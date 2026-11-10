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
use super::nvfp4_gemv::{Nvfp4, quantize_block, quantize_kernel};
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

/// Grouped decode quantization: block `blk` of group `g` quantizes the
/// 16-block `x[g * x_group_blocks + within]` with `input_scale[ids[g]]`.
/// `x_group_blocks` is 0 when every group shares one input row.
#[cube(launch_unchecked)]
fn quantize_grouped_kernel<E: Scalar + Cast>(
    x: &Array<E>,
    codes: &mut Array<u32>,
    scales: &mut Array<e4m3>,
    ids: &Array<u32>,
    input_scale: &Array<f32>,
    group_blocks: usize,
    total_blocks: usize,
    x_group_blocks: usize,
) {
    let blk = ABSOLUTE_POS;
    if blk < total_blocks {
        let g = blk / group_blocks;
        let within = blk % group_blocks;
        let scale2 = input_scale[ids[g] as usize];
        quantize_block::<E>(x, (g * x_group_blocks + within) * 16, codes, scales, blk, scale2);
    }
}

/// [`w4a4_kernel`] for one activation row per group, all groups in one
/// launch: cube z is group `g`, whose weight is expert `ids[g]` of a bank
/// laid out as `[experts, n, k]` planes. Per group the arithmetic is
/// exactly `w4a4_kernel`'s with `m = 1`, and its epilogue scale is
/// `alpha[ids[g]]` (the host's f32 `scale2 * input_scale`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn w4a4_grouped_kernel<AB: Scalar, S: Scalar, O: Scalar + Cast, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    a_sc: &Array<S>,
    b: &Array<Vector<AB, NA>>,
    b_sc: &Array<S>,
    out: &mut Array<O>,
    ids: &Array<u32>,
    alpha: &Array<f32>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] unroll: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE4, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();
    let n_base = CUBE_POS_Y as usize * NTILE;
    let g = CUBE_POS_Z as usize;
    let e = ids[g] as usize;
    // Staging, as the one-row FP8/BF16 kernels: per step the 8 weight rows'
    // next `unroll` k-tiles (8 four-byte elements each per row), padded by 4
    // elements per row so a fragment read's 8 rows x 4 lanes hit 32 banks.
    let vec = b.vector_size();
    let per_tile = comptime!(KTILE4 / 8); // elements of 8 codes per row per tile
    let row = comptime!(KTILE4 / 8 * unroll);
    let stride = comptime!(KTILE4 / 8 * unroll + 4);
    let mut slab = SharedMemory::<Vector<AB, NA>>::new(comptime!(8 * (KTILE4 / 8 * unroll + 4)));

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
    let sib = def.scales_index(lane, MatrixIdent::B) as usize + n_base;
    let spr = comptime!(size_k / GROUP);
    let steps = comptime!(size_k / (KTILE4 * unroll));
    // Every MMA row reads the group's single row (row 0 of its tile).
    let a_row = g * size_k / 2;
    let b_rows = e * size_n;
    let row_elems = size_k / 2 / vec;

    for s in 0..steps {
        let k0 = s * row;
        #[unroll]
        for j in 0..comptime!(KTILE4 / 8 * unroll * 8 / 32) {
            let el = j * 32usize + lane as usize;
            let r = el / row;
            let c = el % row;
            slab[r * stride + c] = b[(b_rows + n_base + r) * row_elems + k0 + c];
        }
        sync_cube();
        #[unroll]
        for u in 0..unroll {
            let t = s * unroll + u;
            let kbase = t * KTILE4;
            #[unroll]
            for i in 0..vc_a {
                let (_row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
                let gc = col as usize + kbase;
                reg_a[i] = a[(a_row + gc / 2) / a.vector_size()];
            }
            #[unroll]
            for i in 0..vc_b {
                let (k_in, n_in) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
                reg_b[i] = slab[n_in as usize * stride + u * per_tile + k_in as usize / 2 / vec];
            }
            let mut sa = Vector::<S, NS>::empty();
            let mut sb = Vector::<S, NS>::empty();
            #[unroll]
            for i in 0..scales_count {
                sa[i] = a_sc[g * spr + t * 4 + i];
                sb[i] = b_sc[(b_rows + sib) * spr + t * 4 + i];
            }
            let d = def.execute_scaled(&reg_a, &reg_b, &acc, sa, sb);
            #[unroll]
            for i in 0..vc_c {
                acc[i] = d[i];
            }
        }
        sync_cube();
    }

    let scale = alpha[e];
    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gc = col as usize + n_base;
        if row as usize == 0usize {
            let v = acc[i];
            #[unroll]
            for j in 0..vs_c {
                out[g * size_n + gc + j] = O::cast_from(v[j] * scale);
            }
        }
    }
}

/// The staging depth of the grouped W4A4 kernel: the largest count of k-tiles
/// (64 codes) up to 8 that divides K's tiles (K 1024 -> 8, 2688 -> 7, 4096 ->
/// 8, 5376 -> 7). It changes only how much each step stages, never the MMA
/// order.
fn grouped_unroll(k: usize) -> usize {
    let tiles = k / KTILE4;
    (1..=8).rev().find(|d| tiles % d == 0).unwrap_or(1)
}

/// A bank of `experts` NVFP4 weights `[n, k]`, each plane contiguous
/// `[experts, n, ...]`, with every expert's activation input scale and its
/// f32 epilogue product `scale2 * input_scale`, both indexed by expert ID.
pub(crate) struct GroupedBank<'a> {
    pub codes: &'a Handle,
    pub scales: &'a Handle,
    pub input_scale: &'a Handle,
    pub alpha: &'a Handle,
    pub experts: usize,
    pub n: usize,
    pub k: usize,
}

/// One BF16 activation row per group (`x_per_group`) or one shared row,
/// times bank expert `ids[g]` for each of `groups` groups, in two launches
/// (grouped quantization, grouped W4A4). Returns BF16 `[groups, n]`.
/// Per group this is exactly `project_w4a4_scaled` on one row. The
/// quantized activation planes are pushed to `retain` as they are made, so a
/// caller that settles work late can keep everything the launches read.
pub(crate) fn w4a4_grouped_bf16(
    client: &ComputeClient<CudaRuntime>,
    x: &Handle,
    x_per_group: bool,
    ids: &Handle,
    groups: usize,
    bank: &GroupedBank<'_>,
    retain: &mut Vec<Handle>,
) -> Result<Handle> {
    let (n, k) = (bank.n, bank.k);
    ensure!(k % KTILE4 == 0 && n % NTILE == 0 && groups > 0 && groups <= u16::MAX as usize,
        "grouped W4A4 geometry [{groups} x {n}, {k}]");
    let group_blocks = k / GROUP;
    let blocks = groups * group_blocks;
    let codes = client.empty(groups * k / 2);
    retain.push(codes.clone());
    let scales = client.empty(blocks);
    retain.push(scales.clone());
    // SAFETY: one thread per 16-block of `groups` rows of k; x holds one row
    // (shared) or `groups` rows of k BF16; ids holds `groups` IDs below
    // bank.experts (checked by the caller) indexing input_scale.
    unsafe {
        quantize_grouped_kernel::launch_unchecked::<bf16, CudaRuntime>(
            client,
            CubeCount::new_1d(blocks.div_ceil(256) as u32),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(x.clone(), if x_per_group { groups * k } else { k }),
            ArrayArg::from_raw_parts(codes.clone(), groups * k / 8),
            ArrayArg::from_raw_parts(scales.clone(), blocks),
            ArrayArg::from_raw_parts(ids.clone(), groups),
            ArrayArg::from_raw_parts(bank.input_scale.clone(), bank.experts),
            group_blocks,
            blocks,
            if x_per_group { group_blocks } else { 0 },
        );
    }
    let out = client.empty(groups * n * 2);
    let vs = 32 / e2m1x2::cube_type().size_bits();
    // SAFETY: activation planes are [groups, k/2] codes and [groups, k/16]
    // scales; the bank planes are [experts, n, k/2] and [experts, n, k/16];
    // each cube reads only rows of its own group and expert.
    unsafe {
        w4a4_grouped_kernel::launch_unchecked::<e2m1x2, e4m3, bf16, CudaRuntime>(
            client,
            CubeCount::Static(1, (n / NTILE) as u32, groups as u32),
            CubeDim::new_1d(32),
            vs,
            2,
            ArrayArg::from_raw_parts(codes.clone(), groups * (k / 2) / vs),
            ArrayArg::from_raw_parts(scales.clone(), blocks),
            ArrayArg::from_raw_parts(bank.codes.clone(), bank.experts * n * (k / 2) / vs),
            ArrayArg::from_raw_parts(bank.scales.clone(), bank.experts * n * k / GROUP),
            ArrayArg::from_raw_parts(out.clone(), groups * n),
            ArrayArg::from_raw_parts(ids.clone(), groups),
            ArrayArg::from_raw_parts(bank.alpha.clone(), bank.experts),
            k,
            n,
            grouped_unroll(k),
        );
    }
    Ok(out)
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

// ---------------------------------------------------------------------------
// W4A4 with a per-row activation global scale, register-blocked.
//
// The prompt lane above quantizes activations with block scales alone (unit
// second-level scale), so an activation block whose largest magnitude is
// below 6 * 2^-6 lands its E4M3 scale in the subnormal binade and one above
// 6 * 448 saturates. WeMM's decoder activations span both (post-norm rows
// of O(1), SiLU products of O(0.01)), so this lane gives every activation row
// its own f32 second-level scale, `amax_row / (6 * 448)`, exactly as the
// weight's tensor scale is chosen: the row's largest block lands on E4M3's top
// code. The epilogue multiplies each output row by its row scale and the
// weight's scale2, in f32, before the single narrowing cast. Rows are tokens,
// so the scale never mixes sequences of a batch.
//
// One plane owns `tm` x `tn` MMA tiles (16 x 8 each; 4 x 8 where N divides
// 64, else 4 x 4) and loads each A and B fragment once per k-step for all of
// them; the instruction, its k order and the f32 accumulation are those of
// `w4a4_kernel`. (A shared-memory variant with a cube of 2 x 2 such planes
// was tried on sky, 2026-11-10, and was 2 to 8 percent SLOWER per launch at
// M 1984 on every WeMM shape; it was removed.)
// ---------------------------------------------------------------------------

/// Threads of the row-scale launch.
const ROW_THREADS: usize = 256;
/// MMA tiles one W4A4-rows plane owns along M and along N.
const ROWS_TM: usize = 4;
/// N tiles per plane where N allows it, else [`ROWS_TN_NARROW`].
const ROWS_TN: usize = 8;
const ROWS_TN_NARROW: usize = 4;

/// `rowscale[r] = amax(|x[r, :]|) / (6 * 448)`, or 1.0 for an all-zero row,
/// or NaN for a row holding a NaN.
#[cube(launch_unchecked)]
fn row_scale_kernel<E: Scalar + Cast>(
    x: &Array<E>,
    rowscale: &mut Array<f32>,
    #[comptime] width: usize,
    #[comptime] threads: usize,
) {
    let r = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut m = 0.0f32;
    // `max` drops a NaN operand and E2M1 saturates, so a NaN activation would
    // come out finite. A row holding one gets a NaN scale instead, which the
    // epilogue multiplies into every output of that row.
    // The NaN is the value read, never a literal.
    let mut nan = 0.0f32;
    let mut j = lane;
    while j < width {
        let v = Abs::abs(f32::cast_from(x[r * width + j]));
        if v != v {
            nan = v;
        }
        m = max(m, v);
        j += threads;
    }
    if nan != nan {
        m = nan;
    }
    let mut shared = SharedMemory::<f32>::new(threads);
    shared[lane] = m;
    sync_cube();
    if lane == 0 {
        let mut a = 0.0f32;
        let mut seen = 0.0f32;
        for i in 0..threads {
            let p = shared[i];
            if p != p {
                seen = p;
            }
            a = max(a, p);
        }
        let mut s = 1.0f32;
        if a > 0.0f32 {
            s = a / 2688.0f32;
        }
        if seen != seen {
            s = seen;
        }
        rowscale[r] = s;
    }
}

/// One 16-block per thread under its row's second-level scale.
#[cube(launch_unchecked)]
fn quantize_rows_kernel<E: Scalar + Cast>(
    x: &Array<E>,
    codes: &mut Array<u32>,
    scales: &mut Array<e4m3>,
    rowscale: &Array<f32>,
    row_blocks: usize,
    blocks: usize,
) {
    let blk = ABSOLUTE_POS;
    if blk < blocks {
        quantize_block::<E>(x, blk * 16, codes, scales, blk, rowscale[blk / row_blocks]);
    }
}

/// `out[r, c] = O(acc[r, c] * (rowscale[r] * scale))` for `r < m`, with
/// `acc` the block-scaled FP4 product of the quantized row and weight row.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn w4a4_rows_kernel<AB: Scalar, S: Scalar, O: Scalar + Cast, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    a_sc: &Array<S>,
    rowscale: &Array<f32>,
    b: &Array<Vector<AB, NA>>,
    b_sc: &Array<S>,
    out: &mut Array<O>,
    m: u32,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE4, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();
    let m_base = CUBE_POS_X as usize * comptime!(MTILE * tm);
    let n_base = CUBE_POS_Y as usize * comptime!(NTILE * tn);
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

    let mut all_a = Array::<Vector<AB, NA>>::new(comptime!(tm * vc_a));
    let mut all_b = Array::<Vector<AB, NA>>::new(comptime!(tn * vc_b));
    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut cur = Array::<Vector<f32, NC>>::new(vc_c);
    let mut acc = Array::<Vector<f32, NC>>::new(comptime!(tm * tn * vc_c));
    #[unroll]
    for i in 0..comptime!(tm * tn * vc_c) {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();
    let sia = def.scales_index(lane, MatrixIdent::A) as usize + m_base;
    let sib = def.scales_index(lane, MatrixIdent::B) as usize + n_base;
    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE4);

    for t in 0..k_tiles {
        let kbase = t * KTILE4;
        #[unroll]
        for mt in 0..tm {
            #[unroll]
            for i in 0..vc_a {
                let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
                let gr = min(row as usize + m_base + mt * MTILE, last);
                let gc = col as usize + kbase;
                all_a[mt * vc_a + i] = a[(gr * size_k / 2 + gc / 2) / a.vector_size()];
            }
        }
        #[unroll]
        for nt in 0..tn {
            #[unroll]
            for i in 0..vc_b {
                let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
                let gr = col as usize + n_base + nt * NTILE;
                let gc = row as usize + kbase;
                all_b[nt * vc_b + i] = b[(gr * size_k / 2 + gc / 2) / b.vector_size()];
            }
        }
        #[unroll]
        for mt in 0..tm {
            let ra = min(sia + mt * MTILE, last);
            let mut sa = Vector::<S, NS>::empty();
            #[unroll]
            for i in 0..scales_count {
                sa[i] = a_sc[ra * spr + t * 4 + i];
            }
            #[unroll]
            for i in 0..vc_a {
                reg_a[i] = all_a[mt * vc_a + i];
            }
            #[unroll]
            for nt in 0..tn {
                let mut sb = Vector::<S, NS>::empty();
                #[unroll]
                for i in 0..scales_count {
                    sb[i] = b_sc[(sib + nt * NTILE) * spr + t * 4 + i];
                }
                #[unroll]
                for i in 0..vc_b {
                    reg_b[i] = all_b[nt * vc_b + i];
                }
                #[unroll]
                for i in 0..vc_c {
                    cur[i] = acc[(mt * tn + nt) * vc_c + i];
                }
                let d = def.execute_scaled(&reg_a, &reg_b, &cur, sa, sb);
                #[unroll]
                for i in 0..vc_c {
                    acc[(mt * tn + nt) * vc_c + i] = d[i];
                }
            }
        }
    }

    #[unroll]
    for mt in 0..tm {
        #[unroll]
        for nt in 0..tn {
            #[unroll]
            for i in 0..vc_c {
                let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
                let gr = row as usize + m_base + mt * MTILE;
                let gc = col as usize + n_base + nt * NTILE;
                if gr < m as usize {
                    let f = rowscale[gr] * scale;
                    let v = acc[(mt * tn + nt) * vc_c + i];
                    #[unroll]
                    for j in 0..vs_c {
                        out[gr * size_n + gc + j] = O::cast_from(v[j] * f);
                    }
                }
            }
        }
    }
}

impl Nvfp4 {
    /// W4A4 with a per-row activation global scale (see the section header):
    /// `[t, k]` or `[1, t, k]` BF16 -> `[1, t, n]` BF16, three launches (row
    /// scales, row-scaled quantization, the register-blocked block-scaled
    /// MMA). Every quantity stays on the device; nothing is read back.
    pub(crate) fn project_w4a4_rows(&self, x: &Tensor) -> Result<Tensor> {
        let tn = if self.n() % (NTILE * ROWS_TN) == 0 { ROWS_TN } else { ROWS_TN_NARROW };
        self.project_w4a4_rows_tn(x, tn)
    }

    /// [`Self::project_w4a4_rows`] with `tn` N tiles per plane. The tiling
    /// changes which plane owns a tile, never a tile's MMAs or their order,
    /// so every `tn` gives the same bits.
    pub(crate) fn project_w4a4_rows_tn(&self, x: &Tensor, tn: usize) -> Result<Tensor> {
        ensure!(self.dtype() == DType::BF16, "W4A4-rows takes BF16 weights");
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
            k % KTILE4 == 0 && n % (NTILE * tn) == 0,
            "W4A4-rows needs k a multiple of {KTILE4} and n of {}, got [{n}, {k}]",
            NTILE * tn
        );
        let client = &x.client;
        let rowscale = client.empty(t * 4);
        // SAFETY: one cube per row of the checked contiguous [t, k] BF16
        // activation; each writes only its own row's scale.
        unsafe {
            row_scale_kernel::launch_unchecked::<bf16, CudaRuntime>(
                client,
                CubeCount::Static(t as u32, 1, 1),
                CubeDim::new_1d(ROW_THREADS as u32),
                ArrayArg::from_raw_parts(x.handle.clone(), t * k),
                ArrayArg::from_raw_parts(rowscale.clone(), t),
                k,
                ROW_THREADS,
            );
        }
        let blocks = t * k / GROUP;
        let codes = client.empty(t * k / 2);
        let scales = client.empty(blocks);
        // SAFETY: one thread per complete 16-block of [t, k]; each writes its
        // own two code words and scale byte; rowscale holds t entries.
        unsafe {
            quantize_rows_kernel::launch_unchecked::<bf16, CudaRuntime>(
                client,
                CubeCount::new_1d(blocks.div_ceil(256) as u32),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(x.handle.clone(), t * k),
                ArrayArg::from_raw_parts(codes.clone(), t * k / 8),
                ArrayArg::from_raw_parts(scales.clone(), blocks),
                ArrayArg::from_raw_parts(rowscale.clone(), t),
                k / GROUP,
                blocks,
            );
        }
        let out = client.empty(t * n * 2);
        let vs = 32 / e2m1x2::cube_type().size_bits();
        // SAFETY: activation planes [t, k/2] and [t, k/16] in the weight's
        // row-major layout; k % 64 == 0 and n % (8 tn) == 0 bound every tile;
        // rows past t are clamped to row t - 1 on load and never stored.
        unsafe {
            w4a4_rows_kernel::launch_unchecked::<e2m1x2, e4m3, bf16, CudaRuntime>(
                client,
                CubeCount::Static(t.div_ceil(MTILE * ROWS_TM) as u32, (n / (NTILE * tn)) as u32, 1),
                CubeDim::new_1d(32),
                vs,
                2,
                ArrayArg::from_raw_parts(codes.clone(), t * (k / 2) / vs),
                ArrayArg::from_raw_parts(scales.clone(), blocks),
                ArrayArg::from_raw_parts(rowscale.clone(), t),
                ArrayArg::from_raw_parts(self.codes.clone(), n * (k / 2) / vs),
                ArrayArg::from_raw_parts(self.scales.clone(), n * k / GROUP),
                ArrayArg::from_raw_parts(out.clone(), t * n),
                t as u32,
                k,
                n,
                ROWS_TM,
                tn,
                self.scale2,
            );
        }
        Ok(Tensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            [1, t, n].into(),
            out,
            DType::BF16,
        ))
    }
}

#[cfg(test)]
mod rows_tests {
    use super::*;
    use cubecl::cuda::CudaDevice;

    struct Rng(u64);
    impl Rng {
        fn unit(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn bf16_tensor(client: &ComputeClient<CudaRuntime>, device: &CudaDevice, v: &[f32], shape: [usize; 2]) -> Tensor {
        let bytes: Vec<u8> = v.iter().flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes()).collect();
        Tensor::new_contiguous(client.clone(), device.clone(), shape.as_slice().into(), client.create_from_slice(&bytes), DType::BF16)
    }

    /// Bug detector for the register-blocked per-row W4A4 lane: its output
    /// against an f64 product of the host packing (`nvfp4::pack_nearest`,
    /// the device recipe) of each activation row under its own global scale
    /// and of the weight. Rows span six decades of magnitude; t and n cross
    /// partial plane tiles. A layout or scale-index bug is an O(1) error.
    #[test]
    #[ignore = "reserved CUDA: W4A4-rows kernel against its host-packed f64 oracle"]
    fn w4a4_rows_matches_host_packed_oracle() {
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        for (t, n, k) in [(1usize, 32usize, 64usize), (17, 64, 256), (150, 96, 512), (70, 4096, 4096), (300, 128, 128)] {
            let mut rng = Rng(t as u64 * 7919 + n as u64);
            let x: Vec<f32> = (0..t * k)
                .map(|i| rng.unit() * 10f32.powi((i / k) as i32 % 7 - 3))
                .map(|v| bf16::from_f32(v).to_f32())
                .collect();
            let w: Vec<f32> = (0..n * k).map(|_| bf16::from_f32(rng.unit() * 0.05).to_f32()).collect();
            let weight = Nvfp4::quantize(&bf16_tensor(&client, &device, &w, [n, k])).unwrap();
            let input = bf16_tensor(&client, &device, &x, [t, k]);
            let out = weight.project_w4a4_rows_tn(&input, ROWS_TN_NARROW).unwrap();
            assert_eq!(out.meta.shape().as_slice(), &[1, t, n]);
            if n % (NTILE * ROWS_TN) == 0 {
                // Wider plane tiles issue the same MMAs in the same order.
                let wide = weight.project_w4a4_rows(&input).unwrap();
                assert_eq!(client.read_one(wide.handle.clone()).unwrap()[..t * n * 2],
                    client.read_one(out.handle.clone()).unwrap()[..t * n * 2], "tn 8 != tn 4 at t {t} n {n} k {k}");
            }
            let got: Vec<f32> = client.read_one(out.handle.clone()).unwrap()[..t * n * 2]
                .chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect();
            let wq = crate::nvfp4::pack_nearest(&w, k).unwrap();
            let (codes, scales) = weight.read_planes(&client).unwrap();
            assert_eq!((codes, scales, weight.scale2.to_bits()), (wq.codes.clone(), wq.scales.clone(), wq.scale2.to_bits()),
                "host packing differs from the device quantizer");
            let wd = wq.decode();
            let mut worst = 0f64;
            for r in 0..t {
                let xq = crate::nvfp4::pack_nearest(&x[r * k..(r + 1) * k], k).unwrap().decode();
                let want: Vec<f64> = (0..n).map(|c| (0..k).map(|j| xq[j] as f64 * wd[c * k + j] as f64).sum()).collect();
                let scale = want.iter().map(|v| v * v).sum::<f64>().sqrt() / (n as f64).sqrt();
                for c in 0..n {
                    let e = (got[r * n + c] as f64 - want[c]).abs() / (scale + 1e-30);
                    worst = worst.max(e);
                }
            }
            eprintln!("w4a4_rows t {t} n {n} k {k}: worst |error| / row rms {worst:.5}");
            // BF16 output rounding is 2^-9 relative; allow 2^-6 of the row rms.
            assert!(worst < 1.0 / 64.0, "t {t} n {n} k {k}: worst {worst}");
        }
    }

    /// A NaN never comes out finite. A NaN activation makes every output of
    /// its row NaN and leaves the other rows' bits exactly as without it; a
    /// NaN weight is refused by the quantizer (load time and storage alike).
    #[test]
    #[ignore = "reserved CUDA: NaN propagation through the W4A4 row scale and the weight amax"]
    fn w4a4_rows_and_the_weight_quantizer_carry_nan() {
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let (t, n, k) = (5usize, 64usize, 256usize);
        let mut rng = Rng(4242);
        let x: Vec<f32> = (0..t * k).map(|_| bf16::from_f32(rng.unit()).to_f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| bf16::from_f32(rng.unit() * 0.05).to_f32()).collect();
        let weight = Nvfp4::quantize(&bf16_tensor(&client, &device, &w, [n, k])).unwrap();
        let rows = |x: &[f32], tn: Option<usize>| -> Vec<u16> {
            let input = bf16_tensor(&client, &device, x, [t, k]);
            let out = match tn {
                Some(tn) => weight.project_w4a4_rows_tn(&input, tn).unwrap(),
                None => weight.project_w4a4_rows(&input).unwrap(),
            };
            client.read_one(out.handle.clone()).unwrap()[..t * n * 2]
                .chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        let mut poisoned = x.clone();
        poisoned[2 * k + 77] = f32::NAN;
        for tn in [None, Some(ROWS_TN_NARROW)] {
            let clean = rows(&x, tn);
            let got = rows(&poisoned, tn);
            for r in 0..t {
                for c in 0..n {
                    let v = bf16::from_bits(got[r * n + c]).to_f32();
                    if r == 2 {
                        assert!(v.is_nan(), "row {r} col {c} is {v}, not NaN (tn {tn:?})");
                    } else {
                        assert_eq!(got[r * n + c], clean[r * n + c], "row {r} col {c} changed (tn {tn:?})");
                    }
                }
            }
        }
        let mut bad = w.clone();
        bad[3 * k + 5] = f32::NAN;
        let refused = Nvfp4::quantize(&bf16_tensor(&client, &device, &bad, [n, k]));
        assert!(refused.is_err(), "a NaN weight was quantized");
        eprintln!("nan: row 2 all NaN, other rows unchanged; NaN weight refused: {}", refused.err().unwrap());
    }
}
