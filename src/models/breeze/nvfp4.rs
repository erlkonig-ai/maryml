//! Resident NVFP4 copies of Breeze's linear projections, quantized once from
//! the immutable BF16 pile aliases at load, and the W4A16 GEMV that reads them.
//!
//! WHY. Every Breeze decode step is a one-row (M=1) projection bound by
//! memory. nsys on sky (GB10, live control, 2026-10-10) put ~84 ms of a ~112 ms
//! frame (83 ms of audio) in M=1 BF16 projections at ~152 GB/s: 2.82 GB per
//! backbone step plus 15 depth steps of 667 MB. Four-bit weights move 0.5625
//! bytes per weight instead of 2. JP, 2026-10-10: "we want to use 4bit
//! wherever we can".
//!
//! ## Format: the standard two-level NVFP4 layout
//!
//! * `codes` `[n, k/2]` bytes: E2M1, two per byte, the LOWER k in the LOW
//!   nibble -- the order `inkling::nvfp4` settled against `compressed_tensors`.
//! * `scales` `[n, k/16]` E4M3FN bytes, one per 16 consecutive values along K.
//! * `scale2`, one f32 per tensor.
//!
//! A weight decodes to `E2M1[code] * e4m3(scale) * scale2`.
//!
//! ## The recipe, operation for operation
//!
//! ```text
//! amax  = max |w| over the tensor                 (device partials, host fold)
//! s2    = amax / (6 * 448)                        (f32; 1.0 if the tensor is all zero)
//! per 16-block, bamax = max |w| over the block:
//! sb    = e4m3_rn(min((bamax / 6) / s2, 448))     (clamped BEFORE the NOSAT cast)
//! d     = f32(sb) * s2
//! code  = e2m1_rn(w / d), ties to the even code, saturating at 6;
//!         every code +0 when d == 0 (no 0/0 reaches the converter)
//! ```
//!
//! Every division is a correctly rounded f32 divide (NVRTC's default
//! `-prec-div=true`), and both conversions are the hardware's, exactly as in
//! `inkling::fp4quant`: `e4m3::cast_from` is `__nv_cvt_float_to_fp8(..,
//! __NV_NOSAT, E4M3)`, round to nearest even; `Vector<e2m1x2, 4>::cast_from` of
//! eight f32 is four `cvt.rn.satfinite.e2m1x2.f32`. The host twin in
//! `tests::recipe` performs the same operations and the CUDA gate compares
//! every code and scale byte. Load time is not a hot path, so the recipe buys
//! one real divide per element rather than a reciprocal multiply.
//!
//! ## Which rows take the GEMV
//!
//! `[1, t, K]` with `t <= MAX_ROWS` runs [`gemv_kernel`]: every decode step,
//! and the depth decoder's two-row prefill of each frame. Larger `t` (the
//! backbone and text-encoder prompt prefills, a few hundred rows) goes through
//! the unchanged BF16 projection on the pile alias. That alias is the pile
//! mapping already registered with CUDA for the weights' whole lifetime, so it
//! costs no memory to keep; a dequantize-to-scratch per prefill call would move
//! 0.5625 + 2 + 2 bytes per weight against the alias's 2. The consequence is a
//! prompt state computed at full BF16 precision and decode steps at four bits.
//!
//! ## The GEMV
//!
//! One plane (32 lanes) per output row. A lane takes 32 consecutive weights at
//! a time -- one 16-byte load of codes, the two E4M3 bytes that scale them, and
//! four 16-byte BF16 activation loads per activation row -- and the plane walks
//! K in 512-weight strides, so each code load is one contiguous 512-byte line
//! across the plane. Products accumulate in f32 per 16-block, are scaled by
//! the block's E4M3 once, carried across K in f32, summed across the plane,
//! multiplied by `scale2` once and cast once to BF16 (round to nearest even):
//! the same single final rounding as the BF16 projection it replaces.
//!
//! ## Measured on sky (GB10), 2026-10-10
//!
//! Framing: `cuda_gemv_bandwidth`, one M=1 launch at a time back to back on
//! one stream, 600 launches rotating over >= 512 MB of distinct weight copies
//! so every launch streams from DRAM; bytes = weights + E4M3 scales +
//! activation + output, against the BF16 Cubek GEMV this replaces. Live bot
//! present on the box, not otherwise loaded.
//!
//! | shape                         | NVFP4 us | GB/s | BF16 us | GB/s |
//! | ----------------------------- | -------: | ---: | ------: | ---: |
//! | depth gate/up `8192 x 1024`   |     25.9 |  183 |    86.4 |  195 |
//! | depth down `1024 x 8192`      |     25.0 |  189 |    84.6 |  199 |
//! | backbone gate/up `6144 x 2048`|     32.3 |  220 |   109.8 |  229 |
//! | depth k/v `256 x 1024`        |      4.1 |   37 |     8.2 |   64 |
//!
//! The small shapes sit on a ~4 us per-launch floor. End to end, the resident
//! control's request 2 (seed 42, CFG1, same binary) went from 6.01 s / 54
//! frames (BF16) to 2.37 s / 56 frames (NVFP4) of generation: ~111 to ~42 ms
//! per 83 ms frame.
use super::cuda_ops::Tensor;
use anyhow::{Result, ensure};
use burn::tensor::DType;
use cubecl::{cuda::CudaRuntime, e2m1x2, e4m3, prelude::*, server::Handle};
use half::bf16;

/// Weights per E4M3 block scale.
pub(super) const GROUP: usize = 16;
/// Weights one lane takes per step: two blocks, sixteen code bytes.
const CHUNK: usize = 32;
/// Lanes per plane. CUDA's warp; checked against the device at quantize time.
const PLANE: usize = 32;
/// Planes (output rows) per cube.
const PLANES: usize = 4;
/// Activation rows the GEMV takes; beyond this the BF16 alias projects.
pub(super) const MAX_ROWS: usize = 8;
/// Threads in the load-time quantize and amax launches.
const QUANT_CUBE: u32 = 256;
/// Cubes in the amax launch: 64 x 256 partial maxima, folded on the host.
const AMAX_CUBES: u32 = 64;

/// Which weights the resident decodes from. One explicit choice per load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Weights {
    /// The BF16 pile aliases only: byte-identical to the pre-NVFP4 path.
    #[default]
    Bf16,
    /// A resident NVFP4 copy of every linear projection of the text encoder,
    /// backbone and depth decoder; embeddings, LM heads and norms stay BF16.
    Nvfp4,
}

impl Weights {
    /// `BREEZE_WEIGHTS`: unset or `bf16` keeps BF16, `nvfp4` selects four bits.
    /// Anything else is refused rather than silently read as the default.
    pub fn from_env() -> Result<Self> {
        match std::env::var("BREEZE_WEIGHTS") {
            Err(std::env::VarError::NotPresent) => Ok(Self::Bf16),
            Ok(value) => Self::parse(&value),
            Err(e) => anyhow::bail!("BREEZE_WEIGHTS: {e}"),
        }
    }
    fn parse(value: &str) -> Result<Self> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "nvfp4" => Ok(Self::Nvfp4),
            other => anyhow::bail!("BREEZE_WEIGHTS must be bf16 or nvfp4, not {other:?}"),
        }
    }
}

/// One linear projection: the immutable BF16 pile alias, and under
/// [`Weights::Nvfp4`] its resident four-bit copy.
pub(super) struct Linear {
    pub(super) bf16: Tensor,
    pub(super) nvfp4: Option<Nvfp4>,
}

/// A resident NVFP4 weight `[n, k]` (see the module header for the layout).
pub(super) struct Nvfp4 {
    codes: Handle,
    scales: Handle,
    scale2: f32,
    n: usize,
    k: usize,
}

#[cube(launch_unchecked)]
fn amax_kernel(w: &Array<bf16>, partial: &mut Array<f32>, len: usize, threads: usize) {
    let t = ABSOLUTE_POS;
    let mut m = 0.0f32;
    let mut i = t;
    while i < len {
        m = max(m, Abs::abs(f32::cast_from(w[i])));
        i += threads;
    }
    partial[t] = m;
}

/// One thread per 16-weight block. Scalar BF16 loads on purpose: the alias
/// sits at whatever offset its leaf has in the pile, and that seam promises
/// four-byte alignment, not the sixteen a vector load would need.
#[cube(launch_unchecked)]
fn quantize_kernel(
    w: &Array<bf16>,
    codes: &mut Array<u32>,
    scales: &mut Array<e4m3>,
    blocks: usize,
    scale2: f32,
) {
    let blk = ABSOLUTE_POS;
    if blk < blocks {
        let base = blk * 16;
        let mut l0 = Vector::<f32, Const<8>>::empty();
        let mut l1 = Vector::<f32, Const<8>>::empty();
        #[unroll]
        for i in 0..8usize {
            l0[i] = f32::cast_from(w[base + i]);
            l1[i] = f32::cast_from(w[base + 8 + i]);
        }
        let hi = max(l0.abs(), l1.abs());
        let a01 = max(max(hi[0], hi[1]), max(hi[2], hi[3]));
        let a23 = max(max(hi[4], hi[5]), max(hi[6], hi[7]));
        let bamax = max(a01, a23);
        let se = e4m3::cast_from(min((bamax / 6.0f32) / scale2, f32::new(448.0f32)));
        scales[blk] = se;
        let d = f32::cast_from(se) * scale2;
        let mut q0 = Vector::<f32, Const<8>>::new(0.0f32);
        let mut q1 = Vector::<f32, Const<8>>::new(0.0f32);
        if d > 0.0f32 {
            let dv = Vector::<f32, Const<8>>::new(d);
            q0 = l0 / dv;
            q1 = l1 / dv;
        }
        codes[blk * 2] = u32::reinterpret(Vector::<e2m1x2, Const<4>>::cast_from(q0));
        codes[blk * 2 + 1] = u32::reinterpret(Vector::<e2m1x2, Const<4>>::cast_from(q1));
    }
}

/// The E2M1 value of the low nibble of `code` (higher bits are ignored).
///
/// For a magnitude code `m >= 2` the value `2^((m >> 1) - 1) * (1 + (m & 1)/2)`
/// IS an f32 whose bits are `(m << 22) + bits(0.5)`: the two exponent bits and
/// the mantissa bit land in place and the bias comes from adding 0.5. Codes 0
/// and 1 (`0.0`, `0.5`) are the subnormal pair the formula misses. The sign
/// bit moves straight to bit 31, so code 8 decodes to the `-0.0` it encodes.
#[cube]
fn e2m1(code: u32) -> f32 {
    let m = code & 7u32;
    let low = (m & 1u32) * 0x3F00_0000u32;
    let high = (m << 22u32) + 0x3F00_0000u32;
    f32::reinterpret(select(m >= 2u32, high, low) | ((code & 8u32) << 28u32))
}

/// `sum_i x[i] * E2M1[nibble i of word]` over the eight nibbles of one word.
#[cube]
fn dot8(word: u32, x: Vector<f32, Const<8>>) -> f32 {
    let mut acc = 0.0f32;
    #[unroll]
    for i in 0..8usize {
        acc += x[i] * e2m1(word >> (4 * i) as u32);
    }
    acc
}

/// `out[r, row] = sum_k x[r, k] * w[row, k]` for `rows` activation rows, one
/// plane per weight row. See the module header for the order of operations.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gemv_kernel<O: Scalar + Cast>(
    x: &Array<Vector<bf16, Const<8>>>,
    codes: &Array<Vector<u32, Const<4>>>,
    scales: &Array<e4m3>,
    out: &mut Array<O>,
    n: usize,
    k: usize,
    scale2: f32,
    #[comptime] rows: usize,
) {
    let lane = UNIT_POS_X as usize;
    let row = CUBE_POS_X as usize * comptime!(PLANES) + UNIT_POS_Y as usize;
    // Uniform per plane: every lane of a plane shares UNIT_POS_Y, so the
    // plane_sum below never runs with a partial plane.
    if row < n {
        let chunks = k / comptime!(CHUNK);
        let mut acc = Array::<f32>::new(rows);
        #[unroll]
        for r in 0..rows {
            acc[r] = 0.0f32;
        }
        let mut c = lane;
        while c < chunks {
            let at = row * chunks + c;
            let words = codes[at];
            let s0 = f32::cast_from(scales[at * 2]);
            let s1 = f32::cast_from(scales[at * 2 + 1]);
            #[unroll]
            for r in 0..rows {
                let base = (r * k + c * comptime!(CHUNK)) / 8;
                let b0 = dot8(words[0], Vector::<f32, Const<8>>::cast_from(x[base]))
                    + dot8(words[1], Vector::<f32, Const<8>>::cast_from(x[base + 1]));
                let b1 = dot8(words[2], Vector::<f32, Const<8>>::cast_from(x[base + 2]))
                    + dot8(words[3], Vector::<f32, Const<8>>::cast_from(x[base + 3]));
                acc[r] += b0 * s0 + b1 * s1;
            }
            c += comptime!(PLANE);
        }
        #[unroll]
        for r in 0..rows {
            let total = plane_sum(acc[r]) * scale2;
            if lane == 0 {
                out[r * n + row] = O::cast_from(total);
            }
        }
    }
}

/// Contiguous `[n, k]` BF16 storage of exactly `n * k` elements, or an error.
fn dense_rows(w: &Tensor) -> Result<(usize, usize)> {
    let shape = w.meta.shape().as_slice();
    let strides = w.meta.strides();
    ensure!(
        shape.len() == 2 && strides.len() == 2,
        "NVFP4 quantizes rank-2 weights"
    );
    let (n, k) = (shape[0], shape[1]);
    ensure!(
        n > 0 && k > 0 && k % CHUNK == 0 && strides[1] == 1 && (n == 1 || strides[0] == k),
        "NVFP4 needs contiguous [n, k] with k a multiple of {CHUNK}, got {shape:?} / {strides:?}"
    );
    ensure!(
        w.dtype == DType::BF16 && w.qparams.is_none(),
        "NVFP4 quantizes unquantized BF16 weights"
    );
    let elements = n
        .checked_mul(k)
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| anyhow::anyhow!("NVFP4 weight exceeds u32 indexing"))?;
    ensure!(
        w.handle.size_in_used() >= 2 * elements as u64,
        "NVFP4 source storage is too short"
    );
    Ok((n, k))
}

impl Nvfp4 {
    /// Quantize one immutable BF16 weight `[n, k]`. Reads the alias, writes two
    /// fresh buffers; one host round trip for the tensor amax.
    pub(super) fn quantize(w: &Tensor) -> Result<Self> {
        let (n, k) = dense_rows(w)?;
        ensure!(
            w.client.properties().hardware.plane_size_max as usize == PLANE
                && w.client.properties().hardware.plane_size_min as usize == PLANE,
            "NVFP4 GEMV is written for {PLANE}-lane planes"
        );
        let len = n * k;
        let threads = (AMAX_CUBES * QUANT_CUBE) as usize;
        let partial = w.client.empty(threads * 4);
        // SAFETY: the alias holds len BF16 (checked above); fresh output of
        // `threads` f32, one per thread; the alias is only read.
        unsafe {
            amax_kernel::launch_unchecked::<CudaRuntime>(
                &w.client,
                CubeCount::new_1d(AMAX_CUBES),
                CubeDim::new_1d(QUANT_CUBE),
                ArrayArg::from_raw_parts(w.handle.clone(), len),
                ArrayArg::from_raw_parts(partial.clone(), threads),
                len,
                threads,
            );
        }
        let bytes = w
            .client
            .read_one(partial)
            .map_err(|e| anyhow::anyhow!("NVFP4 amax readback: {e:?}"))?;
        let amax = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .fold(0.0f32, f32::max);
        let scale2 = scale2_of(amax)?;
        let blocks = len / GROUP;
        let codes = w.client.empty(len / 2);
        let scales = w.client.empty(blocks);
        // SAFETY: one thread per complete 16-block of the checked [n, k]
        // alias; each writes its own two code words and one scale byte.
        unsafe {
            quantize_kernel::launch_unchecked::<CudaRuntime>(
                &w.client,
                CubeCount::new_1d(blocks.div_ceil(QUANT_CUBE as usize) as u32),
                CubeDim::new_1d(QUANT_CUBE),
                ArrayArg::from_raw_parts(w.handle.clone(), len),
                ArrayArg::from_raw_parts(codes.clone(), len / 8),
                ArrayArg::from_raw_parts(scales.clone(), blocks),
                blocks,
                scale2,
            );
        }
        Ok(Self {
            codes,
            scales,
            scale2,
            n,
            k,
        })
    }

    /// Device bytes this copy holds (codes and block scales).
    pub(super) fn bytes(&self) -> u64 {
        (self.n * self.k / 2 + self.n * self.k / GROUP) as u64
    }

    /// Activation rows `t` if `x` is a `[1, t, k]` this GEMV takes, else None
    /// (the caller then projects with the BF16 alias).
    pub(super) fn takes(&self, x: &Tensor) -> Option<usize> {
        let s = x.meta.shape().as_slice();
        let st = x.meta.strides();
        let ok = s.len() == 3
            && s[0] == 1
            && (1..=MAX_ROWS).contains(&s[1])
            && s[2] == self.k
            && st[2] == 1
            && (s[1] == 1 || st[1] == self.k)
            && x.dtype == DType::BF16
            && x.qparams.is_none()
            && x.handle.offset_start.unwrap_or(0) % 16 == 0
            && x.handle.size_in_used() >= (2 * s[1] * self.k) as u64;
        ok.then_some(s[1])
    }

    /// `[1, rows, k] x [n, k]^T -> [1, rows, n]` BF16, fresh output.
    pub(super) fn gemv(&self, x: &Tensor, rows: usize) -> Tensor {
        let out = self.launch::<bf16>(x, rows);
        Tensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            [1, rows, self.n].into(),
            out,
            DType::BF16,
        )
    }

    fn launch<O: Scalar + Cast>(&self, x: &Tensor, rows: usize) -> Handle {
        assert!(
            (1..=MAX_ROWS).contains(&rows),
            "NVFP4 GEMV takes 1..={MAX_ROWS} rows"
        );
        let out = x.client.empty(rows * self.n * core::mem::size_of::<O>());
        // SAFETY: `takes` checked x is a contiguous, 16-byte-aligned BF16
        // [rows, k]; codes/scales are this weight's own [n, k] buffers; the
        // output is fresh [rows, n]. k % 32 == 0 bounds every chunk.
        unsafe {
            gemv_kernel::launch_unchecked::<O, CudaRuntime>(
                &x.client,
                CubeCount::new_1d(self.n.div_ceil(PLANES) as u32),
                CubeDim::new_2d(PLANE as u32, PLANES as u32),
                ArrayArg::from_raw_parts(x.handle.clone(), rows * self.k / 8),
                ArrayArg::from_raw_parts(self.codes.clone(), self.n * self.k / CHUNK),
                ArrayArg::from_raw_parts(self.scales.clone(), self.n * self.k / GROUP),
                ArrayArg::from_raw_parts(out.clone(), rows * self.n),
                self.n,
                self.k,
                self.scale2,
                rows,
            );
        }
        out
    }
}

/// `amax / (6 * 448)`, or 1.0 for an all-zero tensor (every block then stores
/// a zero scale and zero codes, whatever `scale2` is).
fn scale2_of(amax: f32) -> Result<f32> {
    ensure!(amax.is_finite(), "NVFP4 source has a non-finite weight");
    Ok(if amax > 0.0 {
        amax / (6.0f32 * 448.0f32)
    } else {
        1.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::mxfp4::{E2M1, e4m3_to_f32};

    /// splitmix64: a fixed, dependency-free stream for reproducible fixtures.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Standard normal by Box-Muller.
        fn normal(&mut self) -> f64 {
            let u = self.unit().max(1e-300);
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
        }
    }

    /// Weight-shaped values: N(0, sigma) with a sprinkling of 8-sigma outliers,
    /// so block scales span several octaves below the tensor's E4M3 ceiling.
    fn weights(rng: &mut Rng, len: usize, sigma: f64) -> Vec<bf16> {
        (0..len)
            .map(|_| {
                let outlier = if rng.next() % 997 == 0 { 8.0 } else { 1.0 };
                bf16::from_f64(rng.normal() * sigma * outlier)
            })
            .collect()
    }

    /// Round to the nearest E4M3FN value, ties to the even pattern, for a
    /// finite `v` in `[0, 448]`. Brackets the 127 non-negative finite patterns.
    fn e4m3_rn(v: f32) -> u8 {
        assert!((0.0..=448.0).contains(&v), "{v} outside the clamped range");
        let v = v as f64;
        let value = |p: u8| e4m3_to_f32(p) as f64;
        let mut lo = 0u8;
        while lo < 0x7E && value(lo + 1) <= v {
            lo += 1;
        }
        if value(lo) == v || lo == 0x7E {
            return lo;
        }
        let (below, above) = (v - value(lo), value(lo + 1) - v);
        if below < above || (below == above && lo % 2 == 0) {
            lo
        } else {
            lo + 1
        }
    }

    /// `cvt.rn.satfinite.e2m1x2.f32` for one finite value: nearest E2M1
    /// magnitude, ties to the even code, saturating at 6, sign kept (also on
    /// a value that rounds to zero, which becomes code 8).
    fn e2m1_rn(q: f32) -> u8 {
        const MIDPOINTS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
        let a = q.abs();
        let mut code = 0u8;
        for (i, &m) in MIDPOINTS.iter().enumerate() {
            if a > m || (a == m && (i + 1) % 2 == 0) {
                code = i as u8 + 1;
            }
        }
        code | if q.is_sign_negative() { 8 } else { 0 }
    }

    /// The module header's recipe on the host, operation for operation.
    /// Returns (codes [n, k/2], scales [n, k/16], scale2).
    fn recipe(w: &[bf16]) -> (Vec<u8>, Vec<u8>, f32) {
        let amax = w.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
        let s2 = scale2_of(amax).unwrap();
        let mut codes = Vec::with_capacity(w.len() / 2);
        let mut scales = Vec::with_capacity(w.len() / GROUP);
        for block in w.chunks_exact(GROUP) {
            let bamax = block.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
            let sb = e4m3_rn(((bamax / 6.0) / s2).min(448.0));
            scales.push(sb);
            let d = e4m3_to_f32(sb) * s2;
            for pair in block.chunks_exact(2) {
                let code = |v: bf16| {
                    if d > 0.0 { e2m1_rn(v.to_f32() / d) } else { 0 }
                };
                codes.push(code(pair[0]) | code(pair[1]) << 4);
            }
        }
        (codes, scales, s2)
    }

    /// Decode packed NVFP4 to f64: exact, since E2M1 x E4M3 x f32 fits.
    fn decode(codes: &[u8], scales: &[u8], s2: f32) -> Vec<f64> {
        let mut out = Vec::with_capacity(codes.len() * 2);
        for (b, &s) in scales.iter().enumerate() {
            let scale = e4m3_to_f32(s) as f64 * s2 as f64;
            for &byte in &codes[b * GROUP / 2..(b + 1) * GROUP / 2] {
                out.push(E2M1[(byte & 15) as usize] as f64 * scale);
                out.push(E2M1[(byte >> 4) as usize] as f64 * scale);
            }
        }
        out
    }

    #[test]
    fn weights_option_is_explicit() {
        assert_eq!(Weights::default(), Weights::Bf16);
        assert_eq!(Weights::parse("bf16").unwrap(), Weights::Bf16);
        assert_eq!(Weights::parse("nvfp4").unwrap(), Weights::Nvfp4);
        for bad in ["", "NVFP4", "fp4", "bf16 "] {
            assert!(Weights::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn host_converters_are_the_formats() {
        // Every E4M3 value maps to itself, and every midpoint to its even
        // neighbour.
        for p in 0u8..=0x7E {
            assert_eq!(e4m3_rn(e4m3_to_f32(p)), p);
            if p < 0x7E {
                let mid = ((e4m3_to_f32(p) as f64 + e4m3_to_f32(p + 1) as f64) / 2.0) as f32;
                assert_eq!(
                    mid as f64 * 2.0,
                    e4m3_to_f32(p) as f64 + e4m3_to_f32(p + 1) as f64
                );
                assert_eq!(
                    e4m3_rn(mid),
                    if p % 2 == 0 { p } else { p + 1 },
                    "pattern {p}"
                );
            }
        }
        assert_eq!(e4m3_rn(1.0 / 2048.0), 0, "below half the least subnormal");
        for code in 0u8..16 {
            assert_eq!(e2m1_rn(E2M1[code as usize]), code);
        }
        let ties = [
            (0.25, 0),
            (0.75, 2),
            (1.25, 2),
            (1.75, 4),
            (2.5, 4),
            (3.5, 6),
            (5.0, 6),
        ];
        for (q, code) in ties {
            assert_eq!(e2m1_rn(q), code, "{q}");
            assert_eq!(e2m1_rn(-q), code | 8, "-{q}");
        }
        assert_eq!(e2m1_rn(1e9), 7);
        assert_eq!(
            e2m1_rn(-0.1),
            8,
            "a negative value that rounds to zero keeps its sign"
        );
        assert_eq!(scale2_of(0.0).unwrap(), 1.0);
        assert!(scale2_of(f32::INFINITY).is_err());
    }

    #[test]
    fn host_decoder_inverts_the_recipe_on_representable_blocks() {
        // A tensor whose amax is 2688 has scale2 == 1, so a block with block
        // amax 6 has scale 1 and its E2M1 values come back exactly.
        let mut w = vec![bf16::ZERO; 32];
        w[0] = bf16::from_f32(2688.0);
        for (i, v) in E2M1.iter().enumerate() {
            w[16 + i] = bf16::from_f32(*v);
        }
        let (codes, scales, s2) = recipe(&w);
        assert_eq!(s2, 1.0);
        assert_eq!(scales[1], 0x38, "E4M3 1.0");
        let back = decode(&codes, &scales, s2);
        for i in 0..16 {
            assert_eq!(back[16 + i].to_bits(), (E2M1[i] as f64).to_bits(), "{i}");
        }
        assert_eq!(back[0], 2688.0);
    }

    fn upload(client: &ComputeClient<CudaRuntime>, values: &[bf16], shape: &[usize]) -> Tensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        Tensor::new_contiguous(
            client.clone(),
            cubecl::cuda::CudaDevice { index: 0 },
            shape.into(),
            client.create_from_slice(&bytes),
            DType::BF16,
        )
    }

    fn read(client: &ComputeClient<CudaRuntime>, handle: &Handle) -> Vec<u8> {
        client.read_one(handle.clone()).unwrap().to_vec()
    }

    /// An adversarial matrix: exact E2M1 ties at scale 1, blocks far below the
    /// tensor amax (subnormal and zero E4M3 scales), all-zero and signed-zero
    /// blocks, then weight-shaped rows.
    fn adversarial(rng: &mut Rng, n: usize, k: usize) -> Vec<bf16> {
        let mut w = weights(rng, n * k, 0.5);
        w[0] = bf16::from_f32(2688.0);
        let ties = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0];
        for i in 0..k {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            w[k + i] = bf16::from_f32(sign * ties[i % 8]);
            w[2 * k + i] = bf16::from_f64(rng.normal() * 1e-3);
            w[3 * k + i] = bf16::from_f64(rng.normal() * 1e-6);
            w[4 * k + i] = if i % 2 == 0 {
                bf16::NEG_ZERO
            } else {
                bf16::ZERO
            };
        }
        w
    }

    #[test]
    #[ignore = "reserved CUDA: device NVFP4 quantizer against the host recipe, bit for bit"]
    fn cuda_quantizer_is_the_host_recipe_bit_for_bit() {
        let client = CudaRuntime::client(&cubecl::cuda::CudaDevice { index: 0 });
        let mut rng = Rng(0x5EED_0F_F4);
        let cases: Vec<(&str, usize, usize, Vec<bf16>)> = vec![
            ("adversarial", 64, 1024, adversarial(&mut rng, 64, 1024)),
            (
                "depth down",
                1024,
                8192,
                weights(&mut rng, 1024 * 8192, 0.02),
            ),
            (
                "text gate",
                6912,
                1152,
                weights(&mut rng, 6912 * 1152, 0.03),
            ),
            (
                "backbone k",
                1024,
                2048,
                weights(&mut rng, 1024 * 2048, 0.05),
            ),
        ];
        for (name, n, k, w) in cases {
            let t = upload(&client, &w, &[n, k]);
            let q = Nvfp4::quantize(&t).unwrap();
            let (codes, scales, s2) = recipe(&w);
            assert_eq!(q.scale2.to_bits(), s2.to_bits(), "{name}: scale2");
            let dev_scales = read(&client, &q.scales);
            let dev_codes = read(&client, &q.codes);
            assert_eq!(dev_scales.len(), scales.len());
            assert_eq!(dev_codes.len(), codes.len());
            let bad_scales = dev_scales
                .iter()
                .zip(&scales)
                .filter(|(a, b)| a != b)
                .count();
            let bad_codes = dev_codes.iter().zip(&codes).filter(|(a, b)| a != b).count();
            assert_eq!(
                (bad_scales, bad_codes),
                (0, 0),
                "{name}: mismatched scale/code bytes"
            );
            let dev = decode(&dev_codes, &dev_scales, q.scale2);
            let host = decode(&codes, &scales, s2);
            assert!(
                dev.iter()
                    .zip(&host)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            // The quantization error itself, for the record (not a gate).
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (v, d) in w.iter().zip(&dev) {
                err += (v.to_f64() - d).powi(2);
                norm += v.to_f64().powi(2);
            }
            eprintln!(
                "{name} [{n}, {k}]: scale2 {s2:e}, relative RMS quantization error {:.4}",
                (err / norm).sqrt()
            );
        }
    }

    /// Host f64 reference: decode the packed weights, multiply in f64.
    fn reference(x: &[bf16], rows: usize, k: usize, w: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut y = vec![0.0f64; rows * n];
        let mut mag = vec![0.0f64; rows * n];
        for r in 0..rows {
            for j in 0..n {
                let (mut s, mut a) = (0.0f64, 0.0f64);
                for i in 0..k {
                    let p = x[r * k + i].to_f64() * w[j * k + i];
                    s += p;
                    a += p.abs();
                }
                y[r * n + j] = s;
                mag[r * n + j] = a;
            }
        }
        (y, mag)
    }

    const SHAPES: [(&str, usize, usize); 17] = [
        ("backbone q", 2048, 2048),
        ("backbone k/v", 1024, 2048),
        ("backbone o", 2048, 2048),
        ("backbone gate/up", 6144, 2048),
        ("backbone down", 2048, 6144),
        ("depth q", 1024, 1024),
        ("depth k/v", 256, 1024),
        ("depth o", 1024, 1024),
        ("depth gate/up", 8192, 1024),
        ("depth down", 1024, 8192),
        ("depth projector", 1024, 2048),
        ("text q", 1024, 1152),
        ("text k/v", 256, 1152),
        ("text o", 1152, 1024),
        ("text gate/up", 6912, 1152),
        ("text down", 1152, 6912),
        ("text projection", 2048, 1152),
    ];

    #[test]
    #[ignore = "reserved CUDA: W4A16 GEMV against a host f64 product over every Breeze shape"]
    fn cuda_gemv_matches_f64_reference_over_real_shapes() {
        let client = CudaRuntime::client(&cubecl::cuda::CudaDevice { index: 0 });
        let mut rng = Rng(0x0BAD_CAFE);
        let mut worst_f32 = 0.0f64;
        for (name, n, k) in SHAPES {
            let w = weights(&mut rng, n * k, 0.02);
            let q = Nvfp4::quantize(&upload(&client, &w, &[n, k])).unwrap();
            let dense = decode(
                &read(&client, &q.codes),
                &read(&client, &q.scales),
                q.scale2,
            );
            for rows in [1usize, 2, MAX_ROWS] {
                let xv: Vec<bf16> = (0..rows * k)
                    .map(|_| bf16::from_f64(rng.normal()))
                    .collect();
                let x = upload(&client, &xv, &[1, rows, k]);
                assert_eq!(q.takes(&x), Some(rows));
                let (y, mag) = reference(&xv, rows, k, &dense, n);
                // f32 output: the accumulation alone. Bound per output by the
                // absolute sum of its products, the scale of f32 summation error.
                let raw = read(&client, &q.launch::<f32>(&x, rows));
                let mut worst = 0.0f64;
                for (i, b) in raw.chunks_exact(4).enumerate() {
                    let got = f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64;
                    worst = worst.max((got - y[i]).abs() / mag[i].max(f64::MIN_POSITIVE));
                }
                worst_f32 = worst_f32.max(worst);
                assert!(
                    worst < 64.0 * f32::EPSILON as f64,
                    "{name} rows={rows}: f32 accumulation error {worst:e} of sum|x w|"
                );
                // BF16 output: one final round to nearest of that f32.
                let out = q.gemv(&x, rows);
                assert_eq!(out.meta.shape().as_slice(), [1, rows, n]);
                let got = read(&client, &out.handle);
                for (i, b) in got.chunks_exact(2).enumerate() {
                    let got = bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64();
                    let want = bf16::from_f64(y[i]).to_f64();
                    let top = want.abs().max(y[i].abs()).max(f64::MIN_POSITIVE);
                    let ulp = top.log2().floor().exp2() / 128.0;
                    assert!(
                        (got - y[i]).abs() <= ulp / 2.0 + 64.0 * f32::EPSILON as f64 * mag[i],
                        "{name} rows={rows} out {i}: {got} vs exact {} (bf16 {want})",
                        y[i]
                    );
                }
            }
            eprintln!("{name} [{n}, {k}]: ok");
        }
        eprintln!(
            "worst f32 accumulation error: {worst_f32:e} of sum|x w| ({:.1} f32 epsilons)",
            worst_f32 / f32::EPSILON as f64
        );
    }

    #[test]
    #[ignore = "reserved CUDA: GEMV bandwidth on Breeze shapes, NVFP4 against the BF16 projection"]
    fn cuda_gemv_bandwidth() {
        use std::time::Instant;
        let device = cubecl::cuda::CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let sync = || cubecl::future::block_on(client.sync()).unwrap();
        let mut rng = Rng(7);
        let iters = 600usize;
        // Rotate through distinct copies of each weight, >= 512 MB of BF16, so
        // every launch streams from DRAM as the model does (layers far exceed
        // L2). Repeating ONE weight measures L2, not the memory system.
        eprintln!(
            "framing: one M=1 launch, back-to-back on one stream, {iters} launches over \
             distinct weights, sky GB10; bytes = weights (+ E4M3 scales) + activation + output"
        );
        eprintln!("shape            | NVFP4 us   GB/s | BF16 us   GB/s | speedup");
        for (name, n, k) in SHAPES {
            let w = weights(&mut rng, n * k, 0.02);
            let copies = (512usize << 20).div_ceil(2 * n * k).max(2);
            let dense: Vec<Linear> = (0..copies)
                .map(|_| Linear {
                    bf16: upload(&client, &w, &[n, k]),
                    nvfp4: None,
                })
                .collect();
            let quant: Vec<Nvfp4> = dense
                .iter()
                .map(|l| Nvfp4::quantize(&l.bf16).unwrap())
                .collect();
            let x = upload(
                &client,
                &(0..k)
                    .map(|_| bf16::from_f64(rng.normal()))
                    .collect::<Vec<_>>(),
                &[1, 1, k],
            );
            for i in 0..2 * copies {
                quant[i % copies].gemv(&x, 1);
                super::super::backbone::linear(&x, &dense[i % copies]).unwrap();
            }
            sync();
            let started = Instant::now();
            for i in 0..iters {
                quant[i % copies].gemv(&x, 1);
            }
            sync();
            let fp4 = started.elapsed().as_secs_f64() / iters as f64;
            let started = Instant::now();
            for i in 0..iters {
                super::super::backbone::linear(&x, &dense[i % copies]).unwrap();
            }
            sync();
            let bf = started.elapsed().as_secs_f64() / iters as f64;
            let fp4_bytes = (quant[0].bytes() + 2 * k as u64 + 2 * n as u64) as f64;
            let dense_bytes = (2 * n * k + 2 * k + 2 * n) as f64;
            eprintln!(
                "{name:16} | {:7.1} {:6.1} | {:7.1} {:6.1} | {:.2}x",
                fp4 * 1e6,
                fp4_bytes / fp4 / 1e9,
                bf * 1e6,
                dense_bytes / bf / 1e9,
                bf / fp4
            );
        }
    }
}
