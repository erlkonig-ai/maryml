//! Resident NVFP4 copies of 16-bit linear weights, quantized once on the
//! device at load, and the W4A16 GEMV that reads them. One recipe and one
//! kernel, instantiated at each model's own 16-bit type: Breeze's BF16
//! projections (`models::breeze::nvfp4`) and the F16 projections of Voxtral's
//! hearing decoder and audio encoder (`models::voxtral::fast`). The weight's
//! source type is also the type of the activations its GEMV takes and of the
//! output it writes.
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
//! `w` is the stored BF16 or F16 value widened to f32, which is exact for both.
//! Every division is a correctly rounded f32 divide (NVRTC's default
//! `-prec-div=true`), and both conversions are the hardware's, exactly as in
//! `inkling::fp4quant`: `e4m3::cast_from` is `__nv_cvt_float_to_fp8(..,
//! __NV_NOSAT, E4M3)`, round to nearest even; `Vector<e2m1x2, 4>::cast_from` of
//! eight f32 is four `cvt.rn.satfinite.e2m1x2.f32`. The host twins in this
//! module's tests and in `breeze::nvfp4`'s perform the same operations and the
//! CUDA gates compare every code and scale byte. Load time is not a hot path,
//! so the recipe buys one real divide per element rather than a reciprocal
//! multiply.
//!
//! ## The GEMV
//!
//! `[1, t, K]` activations with `t <= MAX_ROWS`. One plane (32 lanes) per
//! output row. A lane takes 32 consecutive weights at a time -- one 16-byte
//! load of codes, the two E4M3 bytes that scale them, and four 16-byte
//! activation loads per activation row -- and the plane walks K in 512-weight
//! strides, so each code load is one contiguous 512-byte line across the
//! plane. Products accumulate in f32 per 16-block, are scaled by the block's
//! E4M3 once, carried across K in f32, summed across the plane, multiplied by
//! `scale2` once and cast once to the 16-bit type (round to nearest even): the
//! same single final rounding as the 16-bit projection it replaces.
use anyhow::{Result, ensure};
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    e2m1x2, e4m3,
    prelude::*,
    server::Handle,
};
use half::{bf16, f16};

type Tensor = CubeTensor<CudaRuntime>;

/// Weights per E4M3 block scale.
pub(crate) const GROUP: usize = 16;
/// Weights one lane takes per step: two blocks, sixteen code bytes.
const CHUNK: usize = 32;
/// Lanes per plane. CUDA's warp; checked against the device at quantize time.
const PLANE: usize = 32;
/// Planes (output rows) per cube.
const PLANES: usize = 4;
/// Activation rows the GEMV takes; beyond this the caller projects in 16 bits.
pub(crate) const MAX_ROWS: usize = 8;
/// Threads in the load-time quantize and amax launches.
const QUANT_CUBE: u32 = 256;
/// Cubes in the amax launch: 64 x 256 partial maxima, folded on the host.
const AMAX_CUBES: u32 = 64;

/// A resident NVFP4 weight `[n, k]` (see the module header for the layout).
pub(crate) struct Nvfp4 {
    pub(crate) codes: Handle,
    pub(crate) scales: Handle,
    pub(crate) scale2: f32,
    n: usize,
    k: usize,
    /// BF16 or F16: the type quantized from, taken in and written out.
    dtype: DType,
    device: CudaDevice,
}

#[cube(launch_unchecked)]
fn amax_kernel<E: Scalar + Cast>(
    w: &Array<E>,
    partial: &mut Array<f32>,
    len: usize,
    threads: usize,
) {
    let t = ABSOLUTE_POS;
    let mut m = 0.0f32;
    let mut i = t;
    while i < len {
        m = max(m, Abs::abs(f32::cast_from(w[i])));
        i += threads;
    }
    partial[t] = m;
}

/// One thread per 16-weight block. Scalar loads on purpose: a pile alias sits
/// at whatever offset its leaf has in the pile, and that seam promises
/// four-byte alignment, not the sixteen a vector load would need.
#[cube(launch_unchecked)]
fn quantize_kernel<E: Scalar + Cast>(
    w: &Array<E>,
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
fn gemv_kernel<A: Scalar + Cast, O: Scalar + Cast>(
    x: &Array<Vector<A, Const<8>>>,
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

/// Contiguous `[n, k]` BF16 or F16 storage of exactly `n * k` elements, or an
/// error.
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
        matches!(w.dtype, DType::BF16 | DType::F16) && w.qparams.is_none(),
        "NVFP4 quantizes unquantized BF16 or F16 weights"
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
    /// Quantize one immutable BF16 or F16 weight `[n, k]`. Reads the source,
    /// writes two fresh buffers; one host round trip for the tensor amax.
    pub(crate) fn quantize(w: &Tensor) -> Result<Self> {
        let (n, k) = dense_rows(w)?;
        ensure!(
            w.client.properties().hardware.plane_size_max as usize == PLANE
                && w.client.properties().hardware.plane_size_min as usize == PLANE,
            "NVFP4 GEMV is written for {PLANE}-lane planes"
        );
        match w.dtype {
            DType::BF16 => Self::quantize_as::<bf16>(w, n, k),
            DType::F16 => Self::quantize_as::<f16>(w, n, k),
            other => unreachable!("dense_rows admitted {other:?}"),
        }
    }

    fn quantize_as<E: Scalar + Cast>(w: &Tensor, n: usize, k: usize) -> Result<Self> {
        let len = n * k;
        let threads = (AMAX_CUBES * QUANT_CUBE) as usize;
        let partial = w.client.empty(threads * 4);
        // SAFETY: the source holds len 16-bit values of type E (checked by
        // dense_rows); fresh output of `threads` f32, one per thread; the
        // source is only read.
        unsafe {
            amax_kernel::launch_unchecked::<E, CudaRuntime>(
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
        // source; each writes its own two code words and one scale byte.
        unsafe {
            quantize_kernel::launch_unchecked::<E, CudaRuntime>(
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
            dtype: w.dtype,
            device: w.device.clone(),
        })
    }

    /// Device bytes this copy holds (codes and block scales).
    pub(crate) fn bytes(&self) -> u64 {
        (self.n * self.k / 2 + self.n * self.k / GROUP) as u64
    }

    /// Activation rows `t` if `x` is a `[1, t, k]` this GEMV takes, else None
    /// (the caller then projects with its 16-bit weight).
    pub(crate) fn takes(&self, x: &Tensor) -> Option<usize> {
        let s = x.meta.shape().as_slice();
        let st = x.meta.strides();
        let ok = s.len() == 3
            && s[0] == 1
            && (1..=MAX_ROWS).contains(&s[1])
            && s[2] == self.k
            && st[2] == 1
            && (s[1] == 1 || st[1] == self.k)
            && x.dtype == self.dtype
            && x.qparams.is_none()
            && x.device == self.device
            && x.handle.offset_start.unwrap_or(0) % 16 == 0
            && x.handle.size_in_used() >= (2 * s[1] * self.k) as u64;
        ok.then_some(s[1])
    }

    /// `[1, rows, k] x [n, k]^T -> [1, rows, n]` in the weight's 16-bit type,
    /// fresh output.
    pub(crate) fn gemv(&self, x: &Tensor, rows: usize) -> Tensor {
        let out = match self.dtype {
            DType::BF16 => self.launch::<bf16>(x, rows),
            DType::F16 => self.launch::<f16>(x, rows),
            other => unreachable!("quantized from {other:?}"),
        };
        Tensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            [1, rows, self.n].into(),
            out,
            self.dtype,
        )
    }

    /// The GEMV writing output type `O`; activations are the weight's type.
    pub(crate) fn launch<O: Scalar + Cast>(&self, x: &Tensor, rows: usize) -> Handle {
        match self.dtype {
            DType::BF16 => self.launch_as::<bf16, O>(x, rows),
            DType::F16 => self.launch_as::<f16, O>(x, rows),
            other => unreachable!("quantized from {other:?}"),
        }
    }

    fn launch_as<A: Scalar + Cast, O: Scalar + Cast>(&self, x: &Tensor, rows: usize) -> Handle {
        assert!(
            (1..=MAX_ROWS).contains(&rows),
            "NVFP4 GEMV takes 1..={MAX_ROWS} rows"
        );
        let out = x.client.empty(rows * self.n * core::mem::size_of::<O>());
        // SAFETY: `takes` checked x is a contiguous, 16-byte-aligned [rows, k]
        // of this weight's type A; codes/scales are this weight's own [n, k]
        // buffers; the output is fresh [rows, n]. k % 32 == 0 bounds every chunk.
        unsafe {
            gemv_kernel::launch_unchecked::<A, O, CudaRuntime>(
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
pub(crate) fn scale2_of(amax: f32) -> Result<f32> {
    ensure!(amax.is_finite(), "NVFP4 source has a non-finite weight");
    Ok(if amax > 0.0 {
        amax / (6.0f32 * 448.0f32)
    } else {
        1.0
    })
}

/// F16 coverage. The BF16 instantiation keeps its own gates in
/// `breeze::nvfp4`'s tests, which run against this code unchanged.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::mxfp4::{E2M1, e4m3_to_f32};

    /// splitmix64, as in Breeze's gates.
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
        fn normal(&mut self) -> f64 {
            let u = self.unit().max(1e-300);
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
        }
    }

    /// N(0, sigma) with a sprinkling of 8-sigma outliers.
    fn weights(rng: &mut Rng, len: usize, sigma: f64) -> Vec<f16> {
        (0..len)
            .map(|_| {
                let outlier = if rng.next() % 997 == 0 { 8.0 } else { 1.0 };
                f16::from_f64(rng.normal() * sigma * outlier)
            })
            .collect()
    }

    /// Nearest E4M3FN for a finite `v` in `[0, 448]`, ties to even.
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

    /// `cvt.rn.satfinite.e2m1x2.f32` for one finite value.
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

    /// The module header's recipe on the host over F16 values.
    fn recipe(w: &[f16]) -> (Vec<u8>, Vec<u8>, f32) {
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
                let code = |v: f16| if d > 0.0 { e2m1_rn(v.to_f32() / d) } else { 0 };
                codes.push(code(pair[0]) | code(pair[1]) << 4);
            }
        }
        (codes, scales, s2)
    }

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

    fn client() -> ComputeClient<CudaRuntime> {
        CudaRuntime::client(&CudaDevice { index: 0 })
    }

    fn upload(client: &ComputeClient<CudaRuntime>, values: &[f16], shape: &[usize]) -> Tensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        Tensor::new_contiguous(
            client.clone(),
            CudaDevice { index: 0 },
            shape.into(),
            client.create_from_slice(&bytes),
            DType::F16,
        )
    }

    fn read(client: &ComputeClient<CudaRuntime>, handle: &Handle) -> Vec<u8> {
        client.read_one(handle.clone()).unwrap().to_vec()
    }

    /// Exact E2M1 ties at scale 1, blocks deep in F16's subnormal range and
    /// far below the tensor amax, all-zero and signed-zero rows.
    fn adversarial(rng: &mut Rng, n: usize, k: usize) -> Vec<f16> {
        let mut w = weights(rng, n * k, 0.5);
        w[0] = f16::from_f32(2688.0);
        let ties = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0];
        for i in 0..k {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            w[k + i] = f16::from_f32(sign * ties[i % 8]);
            w[2 * k + i] = f16::from_f64(rng.normal() * 1e-3);
            w[3 * k + i] = f16::from_f64(rng.normal() * 1e-6);
            w[4 * k + i] = if i % 2 == 0 { f16::NEG_ZERO } else { f16::ZERO };
        }
        w
    }

    /// Voxtral's projections as `[n, k]`: the decoder's folded QKV, O,
    /// gate/up and down, and the encoder's wide QKV and gate/up, O and down.
    const SHAPES: [(&str, usize, usize); 7] = [
        ("decoder qkv", 6144, 3072),
        ("decoder o", 3072, 4096),
        ("decoder gate/up", 18432, 3072),
        ("decoder down", 3072, 9216),
        ("encoder qkv, gate/up", 10240, 1280),
        ("encoder o", 1280, 2048),
        ("encoder down", 1280, 5120),
    ];

    #[test]
    fn f16_recipe_inverts_on_representable_blocks() {
        let mut w = vec![f16::ZERO; 32];
        w[0] = f16::from_f32(2688.0);
        for (i, v) in E2M1.iter().enumerate() {
            w[16 + i] = f16::from_f32(*v);
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

    #[test]
    #[ignore = "reserved CUDA: F16 NVFP4 quantizer against the host recipe, bit for bit"]
    fn cuda_f16_quantizer_is_the_host_recipe_bit_for_bit() {
        let client = client();
        let mut rng = Rng(0xF16_0F_F4);
        let mut cases = vec![("adversarial", 64, 1024, adversarial(&mut rng, 64, 1024))];
        for (name, n, k) in SHAPES {
            cases.push((name, n, k, weights(&mut rng, n * k, 0.02)));
        }
        for (name, n, k, w) in cases {
            let q = Nvfp4::quantize(&upload(&client, &w, &[n, k])).unwrap();
            let (codes, scales, s2) = recipe(&w);
            assert_eq!(q.scale2.to_bits(), s2.to_bits(), "{name}: scale2");
            let (dev_codes, dev_scales) = (read(&client, &q.codes), read(&client, &q.scales));
            assert_eq!(
                (dev_codes.len(), dev_scales.len()),
                (codes.len(), scales.len())
            );
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

    #[test]
    #[ignore = "reserved CUDA: F16 W4A16 GEMV against a host f64 product over Voxtral's shapes"]
    fn cuda_f16_gemv_matches_f64_reference_over_voxtral_shapes() {
        let client = client();
        let mut rng = Rng(0x0F16_CAFE);
        let mut worst_f32 = 0.0f64;
        for (name, n, k) in SHAPES {
            let w = weights(&mut rng, n * k, 0.02);
            let q = Nvfp4::quantize(&upload(&client, &w, &[n, k])).unwrap();
            let dense = decode(
                &read(&client, &q.codes),
                &read(&client, &q.scales),
                q.scale2,
            );
            for rows in [1usize, 4, MAX_ROWS] {
                let xv: Vec<f16> = (0..rows * k).map(|_| f16::from_f64(rng.normal())).collect();
                let x = upload(&client, &xv, &[1, rows, k]);
                assert_eq!(q.takes(&x), Some(rows));
                // Only the rows a test reads back are worth the host product.
                let (mut y, mut mag) = (vec![0.0f64; rows * n], vec![0.0f64; rows * n]);
                for r in 0..rows {
                    for j in 0..n {
                        for i in 0..k {
                            let p = xv[r * k + i].to_f64() * dense[j * k + i];
                            y[r * n + j] += p;
                            mag[r * n + j] += p.abs();
                        }
                    }
                }
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
                // F16 output: one final round to nearest of that f32.
                let out = q.gemv(&x, rows);
                assert_eq!(out.meta.shape().as_slice(), [1, rows, n]);
                assert_eq!(out.dtype, DType::F16);
                for (i, b) in read(&client, &out.handle).chunks_exact(2).enumerate() {
                    let got = f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64();
                    let top = got.abs().max(y[i].abs()).max(f64::MIN_POSITIVE);
                    let ulp = top.log2().floor().exp2() / 1024.0;
                    assert!(
                        (got - y[i]).abs() <= ulp / 2.0 + 64.0 * f32::EPSILON as f64 * mag[i],
                        "{name} rows={rows} out {i}: {got} vs exact {}",
                        y[i]
                    );
                }
            }
            // A BF16 activation never reaches an F16 weight's GEMV.
            let x = upload(&client, &vec![f16::ZERO; k], &[1, 1, k]);
            let as_bf16 = Tensor::new_contiguous(
                x.client.clone(),
                x.device.clone(),
                [1, 1, k].into(),
                x.handle.clone(),
                DType::BF16,
            );
            assert_eq!(q.takes(&as_bf16), None);
            let nine = upload(&client, &vec![f16::ZERO; 9 * k], &[1, 9, k]);
            assert_eq!(
                q.takes(&nine),
                None,
                "{name}: nine rows are a 16-bit projection"
            );
            eprintln!("{name} [{n}, {k}]: ok");
        }
        eprintln!(
            "worst f32 accumulation error: {worst_f32:e} of sum|x w| ({:.1} f32 epsilons)",
            worst_f32 / f32::EPSILON as f64
        );
    }
}
