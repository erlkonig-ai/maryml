//! The prompt pass on stored NVFP4 weights: `[1, t, K] x [n, K]^T` for the
//! activation heights the GEMV ([`super::nvfp4_gemv`], `t <= 8`) does not
//! take. No 16-bit copy of a projection exists anywhere: the four-bit planes
//! are the pile alias, and these kernels read them directly.
//!
//! Two lanes, both one plane per 16 x 8 output tile, both derived from
//! Inkling's (`models::inkling::fp4gemm::fp4_linear` and
//! `models::inkling::w4a16gemm::w4a16_linear`, whose headers carry the
//! fragment-layout arguments these rely on):
//!
//! * [`PromptLane::W4a4`]: the activation is quantized on the device to
//!   dynamic per-16 E2M1 with E4M3 block scales (no second-level scale) and
//!   both operands go to `mma.sync...kind::mxf4nvf4.block_scale` on
//!   sm_121a. Inkling's checkpoint was calibrated for that; Breeze and Voxtral
//!   were not, so whether it costs words or voice is measured, not assumed.
//! * [`PromptLane::W4a16`]: the activation stays 16-bit; each lane decodes its
//!   two codes and their block scale to 16 bits in registers and runs the
//!   16-bit `m16n8k16` MMA. The weight is never wider than four bits outside a
//!   register file.
//!
//! Both accumulate in f32, apply the tensor's global scale once to the
//! accumulator and round once to the activation's 16-bit type. Rows past `t`
//! in the last 16-row tile load row `t - 1` instead of a zero pad: an MMA row
//! reaches only its own accumulator row, and those rows are never stored, so
//! there is no padded activation buffer at all.
use super::nvfp4_gemv::{Nvfp4, quantize_kernel};
use anyhow::{Result, ensure};
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::ir::MatrixIdent;
use cubecl::{cuda::CudaRuntime, e2m1x2, e4m3, prelude::*, server::Handle};
use half::{bf16, f16};

type Tensor = CubeTensor<CudaRuntime>;

/// Rows of one MMA tile.
const MTILE: usize = 16;
/// Columns of one MMA tile.
const NTILE: usize = 8;
/// K of the 16-bit instruction.
const KTILE16: usize = 16;
/// K of the four-bit block-scaled instruction.
const KTILE4: usize = 64;
/// Elements per E4M3 block scale.
const GROUP: usize = 16;
/// E2M1 codes per `u32`.
const CODES_PER_WORD: usize = 8;
/// m-tiles one W4A16 plane carries, sharing each decoded weight fragment.
const W4A16_TILES: usize = 4;

pub use crate::nvfp4::PromptLane;

/// The value of one E2M1 code (bit 3 is the sign), as seven comparisons:
/// a runtime-indexed local table would spill.
#[cube]
fn e2m1_value(code: u32) -> f32 {
    let mag = code & 7u32;
    let mut v = f32::new(0.0f32);
    if mag == 1u32 {
        v = f32::new(0.5f32);
    }
    if mag == 2u32 {
        v = f32::new(1.0f32);
    }
    if mag == 3u32 {
        v = f32::new(1.5f32);
    }
    if mag == 4u32 {
        v = f32::new(2.0f32);
    }
    if mag == 5u32 {
        v = f32::new(3.0f32);
    }
    if mag == 6u32 {
        v = f32::new(4.0f32);
    }
    if mag == 7u32 {
        v = f32::new(6.0f32);
    }
    if code >= 8u32 {
        v = -v;
    }
    v
}

/// W4A16: `out[r, c] = O(scale * sum_k a[r, k] * w[c, k])` for `r < m`.
///
/// One plane per `TILES` m-tiles of one 8-column n-tile: each k-step decodes
/// the weight fragment once and multiplies it into every m-tile's
/// accumulator, so a weight word is fetched and decoded once per `TILES * 16`
/// activation rows rather than once per 16. m-tiles wholly past `m` are
/// skipped (uniformly across the plane).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn w4a16_kernel<AB: Scalar + Cast, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    b: &Array<u32>,
    b_sc: &Array<e4m3>,
    out: &mut Array<AB>,
    m: u32,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] tiles: usize,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE16);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();
    let m_base = CUBE_POS_X as usize * comptime!(MTILE * tiles);
    let n_base = CUBE_POS_Y as usize * NTILE;
    let last = m as usize - 1;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut cur = Array::<Vector<f32, NC>>::new(vc_c);
    let mut acc = Array::<Vector<f32, NC>>::new(comptime!(tiles * vc_c));
    #[unroll]
    for i in 0..comptime!(tiles * vc_c) {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let wpr = comptime!(size_k / CODES_PER_WORD);
    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE16);
    for t in 0..k_tiles {
        let kbase = t * KTILE16;
        #[unroll]
        for i in 0..vc_b {
            // B is the stored [n, k]: `col` indexes n, `row` indexes k. A
            // lane's pair starts at an even k, so both elements share one
            // code word and one 16-block scale.
            let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
            let gr = col as usize + n_base;
            let gc = row as usize + kbase;
            let word = b[gr * wpr + gc / CODES_PER_WORD];
            let s = f32::cast_from(b_sc[gr * spr + gc / GROUP]);
            let mut v = Vector::<AB, NA>::empty();
            #[unroll]
            for j in 0..vs_b {
                let kk = gc + j;
                let code = (word >> (4 * (kk % CODES_PER_WORD)) as u32) & 15u32;
                v[j] = AB::cast_from(e2m1_value(code) * s);
            }
            reg_b[i] = v;
        }
        #[unroll]
        for mt in 0..tiles {
            let tile_base = m_base + mt * MTILE;
            if tile_base < m as usize {
                #[unroll]
                for i in 0..vc_a {
                    let (row, col) =
                        def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
                    let gr = min(row as usize + tile_base, last);
                    let gc = col as usize + kbase;
                    reg_a[i] = a[(gr * size_k + gc) / a.vector_size()];
                }
                #[unroll]
                for i in 0..vc_c {
                    cur[i] = acc[mt * vc_c + i];
                }
                let d = def.execute(&reg_a, &reg_b, &cur);
                #[unroll]
                for i in 0..vc_c {
                    acc[mt * vc_c + i] = d[i];
                }
            }
        }
    }

    #[unroll]
    for mt in 0..tiles {
        #[unroll]
        for i in 0..vc_c {
            let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
            let gr = row as usize + m_base + mt * MTILE;
            let gc = col as usize + n_base;
            if gr < m as usize {
                let v = acc[mt * vc_c + i];
                #[unroll]
                for j in 0..vs_c {
                    out[gr * size_n + gc + j] = AB::cast_from(v[j] * scale);
                }
            }
        }
    }
}

/// W4A4: both operands NVFP4, the block-scaled MMA, f32 accumulator.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn w4a4_kernel<AB: Scalar, S: Scalar, O: Scalar + Cast, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    a_sc: &Array<S>,
    b: &Array<Vector<AB, NA>>,
    b_sc: &Array<S>,
    out: &mut Array<O>,
    m: u32,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE4, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();
    let m_base = CUBE_POS_X as usize * MTILE;
    let n_base = CUBE_POS_Y as usize * NTILE;
    let last = m as usize - 1;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();
    let sia = min(
        def.scales_index(lane, MatrixIdent::A) as usize + m_base,
        last,
    );
    let sib = def.scales_index(lane, MatrixIdent::B) as usize + n_base;
    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE4);

    for t in 0..k_tiles {
        let kbase = t * KTILE4;
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
            let gr = min(row as usize + m_base, last);
            let gc = col as usize + kbase;
            reg_a[i] = a[(gr * size_k / 2 + gc / 2) / a.vector_size()];
        }
        #[unroll]
        for i in 0..vc_b {
            let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
            let gr = col as usize + n_base;
            let gc = row as usize + kbase;
            reg_b[i] = b[(gr * size_k / 2 + gc / 2) / b.vector_size()];
        }
        let mut sa = Vector::<S, NS>::empty();
        let mut sb = Vector::<S, NS>::empty();
        #[unroll]
        for i in 0..scales_count {
            sa[i] = a_sc[sia * spr + t * 4 + i];
            sb[i] = b_sc[sib * spr + t * 4 + i];
        }
        let d = def.execute_scaled(&reg_a, &reg_b, &acc, sa, sb);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = d[i];
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gr = row as usize + m_base;
        let gc = col as usize + n_base;
        if gr < m as usize {
            let v = acc[i];
            #[unroll]
            for j in 0..vs_c {
                out[gr * size_n + gc + j] = O::cast_from(v[j] * scale);
            }
        }
    }
}

/// Rows `t` of `x` if it is a contiguous `[1, t, k]` (or `[t, k]`) activation
/// of the weight's 16-bit type on its device, at a 4-byte aligned offset.
fn rows_of(w: &Nvfp4, x: &Tensor) -> Result<usize> {
    let s = x.meta.shape().as_slice();
    let st = x.meta.strides();
    let (t, k) = match s {
        [1, t, k] => (*t, *k),
        [t, k] => (*t, *k),
        other => anyhow::bail!("NVFP4 prompt projection takes [1, t, k], got {other:?}"),
    };
    ensure!(
        k == w.k() && t > 0 && st[st.len() - 1] == 1 && (t == 1 || st[st.len() - 2] == k),
        "NVFP4 prompt projection needs contiguous [t, {}], got {s:?} / {st:?}",
        w.k()
    );
    ensure!(
        x.dtype == w.dtype() && x.qparams.is_none() && &x.device == w.device(),
        "NVFP4 prompt projection: activation is {:?} on {:?}, weight wants {:?}",
        x.dtype,
        x.device,
        w.dtype()
    );
    ensure!(
        x.handle.offset_start.unwrap_or(0) % 4 == 0,
        "NVFP4 prompt projection needs a 4-byte aligned activation"
    );
    ensure!(
        t * k <= u32::MAX as usize && w.n() * k <= u32::MAX as usize,
        "NVFP4 prompt projection exceeds u32 indexing"
    );
    Ok(t)
}

impl Nvfp4 {
    /// `[1, t, k] x [n, k]^T -> [1, t, n]` in the weight's 16-bit type, for any
    /// `t`, on the stored planes. A strided or misaligned activation is made
    /// contiguous first; type and device are checked.
    pub(crate) fn project(&self, x: &Tensor, lane: PromptLane) -> Result<Tensor> {
        self.project_with_scale(x, lane, 1.0)
    }

    /// Calibrated W4A4 on unchanged typed weight planes. `input_scale` is the
    /// positive finite activation dequantization global from THIS operation,
    /// not an inverse scale or the weight global. Existing project() callers
    /// retain unit activation global. Allocation admission belongs to caller.
    ///
    /// Per-block quantization uses (amax/6)/input_scale, then x/(block*global).
    /// The epilogue uses F32(input_scale*weight_global) before the final narrow
    /// cast. This association is explicit, not a claim of fake-quant bit parity.
    pub(crate) fn project_w4a4_scaled(&self, x: &Tensor, input_scale: f32) -> Result<Tensor> {
        calibrated_scale(input_scale, self.scale2)?;
        self.project_with_scale(x, PromptLane::W4a4, input_scale)
    }

    fn project_with_scale(&self, x: &Tensor, lane: PromptLane, input_scale: f32) -> Result<Tensor> {
        let packed;
        let x = if x.is_contiguous() && x.handle.offset_start.unwrap_or(0) % 4 == 0 {
            x
        } else {
            packed = burn_cubecl::kernel::into_contiguous_aligned(x.clone());
            &packed
        };
        let t = rows_of(self, x)?;
        let (n, k) = (self.n(), self.k());
        ensure!(
            n % NTILE == 0,
            "NVFP4 prompt projection needs n a multiple of {NTILE}, got {n}"
        );
        let out = match (lane, self.dtype()) {
            (PromptLane::W4a16, DType::BF16) => self.w4a16::<bf16>(x, t),
            (PromptLane::W4a16, DType::F16) => self.w4a16::<f16>(x, t),
            (PromptLane::W4a4, DType::BF16) => self.w4a4::<bf16>(x, t, input_scale)?,
            (PromptLane::W4a4, DType::F16) => self.w4a4::<f16>(x, t, input_scale)?,
            (_, other) => anyhow::bail!("NVFP4 weight of {other:?}"),
        };
        let _ = k;
        Ok(Tensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            [1, t, n].into(),
            out,
            self.dtype(),
        ))
    }

    fn w4a16<E: Scalar + Cast + CubeElement>(&self, x: &Tensor, t: usize) -> Handle {
        let (n, k) = (self.n(), self.k());
        let client = &x.client;
        let out = client.empty(t * n * core::mem::size_of::<E>());
        let vs = 2usize;
        // SAFETY: rows_of checked x is a contiguous, 4-byte aligned [t, k] of
        // E; the planes are this weight's [n, k/2] codes and [n, k/16] scales;
        // the output is fresh [t, n]; k % 32 == 0 and n % 8 == 0 bound every
        // tile, and the last partial m-tile reads row t - 1 only.
        unsafe {
            w4a16_kernel::launch_unchecked::<E, CudaRuntime>(
                client,
                CubeCount::Static(
                    t.div_ceil(MTILE * W4A16_TILES) as u32,
                    (n / NTILE) as u32,
                    1,
                ),
                CubeDim::new_1d(32),
                vs,
                2,
                ArrayArg::from_raw_parts(x.handle.clone(), t * k / vs),
                ArrayArg::from_raw_parts(self.codes.clone(), n * k / CODES_PER_WORD),
                ArrayArg::from_raw_parts(self.scales.clone(), n * k / GROUP),
                ArrayArg::from_raw_parts(out.clone(), t * n),
                t as u32,
                k,
                n,
                W4A16_TILES,
                self.scale2,
            );
        }
        out
    }

    fn w4a4<O: Scalar + Cast + CubeElement>(&self, x: &Tensor, t: usize, input_scale: f32) -> Result<Handle> {
        let (n, k) = (self.n(), self.k());
        ensure!(
            k % KTILE4 == 0,
            "W4A4 needs k a multiple of {KTILE4}, got {k}"
        );
        let client = &x.client;
        let blocks = t * k / GROUP;
        let codes = client.empty(t * k / 2);
        let scales = client.empty(blocks);
        // Reuse the existing quantizer with the operation's activation global.
        // Existing project() callers pass exactly 1.0, preserving their recipe.
        // SAFETY: one thread per complete 16-block of the checked [t, k]
        // activation; each writes its own two code words and scale byte.
        unsafe {
            quantize_kernel::launch_unchecked::<O, CudaRuntime>(
                client,
                CubeCount::new_1d(blocks.div_ceil(256) as u32),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(x.handle.clone(), t * k),
                ArrayArg::from_raw_parts(codes.clone(), t * k / 8),
                ArrayArg::from_raw_parts(scales.clone(), blocks),
                blocks,
                input_scale,
            );
        }
        let out = client.empty(t * n * core::mem::size_of::<O>());
        let vs = 32 / e2m1x2::cube_type().size_bits();
        // SAFETY: both operands are [rows, k/2] codes and [rows, k/16] scales
        // in the same row-major layout; k % 64 == 0 and n % 8 == 0 bound every
        // tile; the last partial m-tile reads row t - 1 only.
        unsafe {
            w4a4_kernel::launch_unchecked::<e2m1x2, e4m3, O, CudaRuntime>(
                client,
                CubeCount::Static(t.div_ceil(MTILE) as u32, (n / NTILE) as u32, 1),
                CubeDim::new_1d(32),
                vs,
                2,
                ArrayArg::from_raw_parts(codes.clone(), t * (k / 2) / vs),
                ArrayArg::from_raw_parts(scales.clone(), blocks),
                ArrayArg::from_raw_parts(self.codes.clone(), n * (k / 2) / vs),
                ArrayArg::from_raw_parts(self.scales.clone(), n * k / GROUP),
                ArrayArg::from_raw_parts(out.clone(), t * n),
                t as u32,
                k,
                n,
                self.scale2 * input_scale,
            );
        }
        Ok(out)
    }
}

/// Validate before any dispatch/allocation, including the existing contiguous
/// conversion. Reject product underflow/overflow instead of silently erasing
/// or overflowing a calibrated operation.
pub(crate) fn calibrated_scale(input: f32, weight: f32) -> Result<f32> {
    let product = input * weight;
    ensure!(input.is_finite() && input > 0.0 && weight.is_finite() && weight > 0.0
        && product.is_finite() && product > 0.0,
        "W4A4 calibrated globals/product must be finite positive");
    Ok(product)
}
