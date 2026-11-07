//! Nemotron-H attention on tensor cores over NVFP4 K/V, with `S = Q K^T` on
//! the block-scaled FP4 MMA (phase B2).
//!
//! The native attention owner's only route, at every T including T1 decode.
//! The checked [`Plan`] is shared by admission and launch. One configuration,
//! no switch: [`NK`] = 32 keys per tile, two `cp.async` stages, [`WARPS`] = 4
//! planes per cube, P split into BF16 hi + lo.
//!
//! Design and evidence: `lanes/attn-20261105/phase-b-fp4-design.txt`,
//! `phase-b2-design.txt`, `fp4-b1c-report.txt` and `b2-sky-evidence.tgz`
//! (sky, 2026-11-05: T256 over K 524,288, debug test binary, host wall per
//! launch, median of 6 warm; this exact shape, HiLo P: 44.7 ms, against 53.4
//! for the BF16 K/V kernel at the same staging). B2 is a PRECISION change, not
//! an equivalence: Q is quantized to 4 bits under per-16 E4M3 scales and the
//! hardware sums each k64 product its own way. It is judged by end-to-end
//! capability; the structural tests below guard layouts and maps only.
//!
//! # Shape
//!
//! 32 query heads over 2 KV heads (16 per group), `head_dim` 128, no bias, no
//! window, no gaps. The grid is `(query tiles, KV heads, key splits)`; plane
//! `w` of a cube owns query position `p = tile * 4 + w` and ALL 16 heads of the
//! cube's KV head, the 16 rows of its `m16` fragments. Padded planes (`p >=
//! nq`) take part in every copy, decode and barrier and write nothing.
//!
//! # K/V as stored
//!
//! A run is the cache's NVFP4 rows as `fp4quant::quantize_nvfp4_bf16` writes
//! them, the layout of `inkling::kvpages::Fp4Rows`: per KV row of 2 x 128
//! features, 32 `u32` code words (element `i` of word `w` at bits `4 (i % 8)`,
//! the lowest index in the low nibble) in the [`KeyRun`]'s `k` / `v` handle and
//! 16 E4M3 scale bytes (one per 16 consecutive features, no second level) in
//! `k_scales` / `v_scales`. Head `kvh`'s slice is code words `16 kvh ..+ 16`
//! (64 bytes, four 16-byte units) and scale bytes `8 kvh ..+ 8`: 144 bytes per
//! key per KV head for K and V together ([`FP4_BYTES_PER_KEY_HEAD`]).
//!
//! # Per key tile
//!
//! * Staging ([`copy_tile_fp4_async`]): per key, its head's four code units
//!   (`cp.async.cg`, 16 bytes) and its 8 scale bytes (8-byte `cp.async`) into
//!   rotated, unpadded 64-byte code rows ([`fp4_unit`]) and 8-byte scale rows,
//!   two stages, completion by one block-scope mbarrier:
//!
//! ```text
//! prologue: issue(tile s_lo -> stage 0); commit; arrive                 phase 0
//! tile n:   wait_parity(n % 2); issue(tile n + 1 -> stage (n + 1) % 2); commit;
//!           decode V(stage n % 2); sync_cube; compute; arrive            phase n + 1
//! ```
//!
//!   Passing `wait_parity(n % 2)` means tile `n` has landed and every unit has
//!   arrived after computing tile `n - 1`, so the stage tile `n + 1` is issued
//!   into, and the one BF16 V tile the decode overwrites, are no longer read.
//!   Keys at or past the split's end are zero-filled (source size 0, from a
//!   clamped in-range row), so nothing past `s_hi` is ever read.
//! * Q, once per cube ([`fp4_q_quantize`]): every plane quantizes its 16 heads
//!   by `fp4quant`'s rule, bit for bit, straight into the k64 A operand (8 code
//!   words and 2 scale words held across the key loop). `scaling` multiplies S
//!   after the MMA; folding it into Q would change the codes.
//! * `S = Q K^T` ([`fp4_qk_tile`]): per n8 key tile two
//!   `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`,
//!   dims `0..64` then `64..128`, chained from `+0.0`, the B operands and their
//!   scale words read raw from the staged rows ([`fp4_qk_k_operands`]). The
//!   accumulator comes out in the `m16n8k16` layout the masking reads.
//! * V ([`fp4_decode_tile`]): decoded once per cube by the hardware
//!   (`cvt.rn.f16x2.e2m1x2`, `cvt.rn.f16x2.e4m3x2`, exact) into ONE BF16 V
//!   tile of 68-word rows, read by `ldmatrix.x4.trans` into BF16 PV MMAs.
//! * Online softmax on the accumulator, `flash_kernel`'s guarded recurrence;
//!   P becomes the k16 A fragment of `O += P V` in registers (the map measured
//!   bitwise on this GB10), `p = hi + lo`, both MMAs into one accumulator.
//!
//! The partials are written in EXACTLY `flash_kernel`'s layout and
//! normalisation, and `flash_combine_kernel` turns them into the `[nq, 32 *
//! 128]` F32 output. Splits come from `flash::key_run_splits`, and each split
//! is trimmed by the cube's causal bound exactly as `flash_kernel` trims it.
//!
//! # Register layouts
//!
//! With `g = lane / 4`, `t = lane % 4`, the e2m1 `m16n8k64` maps are
//! `cubecl-cpp`'s (`cuda/processors.rs`, `MmaDefinition::scales_index`),
//! measured by one-hot on this GB10 (`nvfp4_mma_probe`):
//!
//! ```text
//! A reg i (0..4)    row g + 8 (i & 1)    k 8 t + 32 (i >> 1) + 0..8
//! B reg i (0..2)    n g                  k 8 t + 32 i + 0..8
//! D pair i (0..2)   row g + 8 i          n 2 t, 2 t + 1
//! A scales          row g + 8 (t & 1)    byte b = k-block b
//! B scales          n g                  byte b = k-block b
//! ```
//!
//! B reg `i` of step `s` is component `t` of staged unit `2 s + i` of key `8 nt
//! + g`, and the key's scale row's word `s` its scale word: a stored code word
//! IS a register. `attention_mma_tests.rs` replays every map and bank argument
//! on the host.
//!
//! # Shared memory
//!
//! [`shared_bytes`] = the barrier (16) + two NVFP4 stages of K and V (2 x 144 x
//! 32) + one BF16 V tile (32 x 272) = 17,936 bytes, inside the 48 KiB default
//! and 5 blocks per SM by shared alone on this GB10.

use super::attention::{HEADS, HEAD_DIM, KV_HEADS};
use crate::models::inkling::flash::{self, KeyRun, NEG_INF, PLANE};
use crate::models::inkling::fp4quant;
use cubecl::{
    e2m1x2, e4m3,
    features::ScaledMmaConfig,
    ir::MatrixIdent,
    prelude::barrier::{Barrier, copy_async_checked},
    prelude::*,
    server::Handle,
};
use half::bf16;

/// Query heads per KV head.
const GROUPS: usize = HEADS / KV_HEADS;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
/// Query planes per cube, at every T.
pub(crate) const WARPS: usize = 4;
/// Keys per staged tile.
pub(crate) const NK: usize = 32;
/// NVFP4 `cp.async` stages.
pub(crate) const STAGES: usize = 2;
/// 32-bit words (BF16 pairs) per row of the decoded BF16 V tile: 64 for the
/// 128 dims plus 4 of padding, 272 bytes = 17 16-byte units. An ODD unit count
/// puts the 8 rows of every `ldmatrix` matrix on 8 distinct 16-byte groups.
pub(crate) const ASYNC_ROW_WORDS: usize = HEAD_DIM / 2 + 4;
/// One `ldmatrix` row in BF16 pairs (16 bytes).
pub(crate) const ASYNC_CHUNK_PAIRS: usize = 4;
/// The 8-byte block-scope mbarrier at shared offset 0, padded to the first
/// 16-byte-aligned buffer.
pub(crate) const ASYNC_BARRIER_BYTES: usize = 16;
/// 16-byte code units per staged head row (128 E2M1 codes, 64 bytes).
pub(crate) const FP4_ROW_UNITS: usize = HEAD_DIM / 32;
/// 16-byte code units per KV row in global, both heads.
const FP4_KV_ROW_UNITS: usize = KV_WIDTH / 32;
/// `u32` code words per 16-byte unit.
pub(crate) const FP4_UNIT_WORDS: usize = 4;
/// `u32` words (four E4M3 bytes each) per head scale row: 8 bytes.
pub(crate) const FP4_SCALE_WORDS: usize = HEAD_DIM / 16 / 4;
/// Bytes staged per key per KV head: 64 code and 8 scale bytes for each of K
/// and V, exactly the stored head slice.
pub(crate) const FP4_BYTES_PER_KEY_HEAD: usize = 2 * (HEAD_DIM / 2 + HEAD_DIM / 16);
/// The default shared memory per block; no opt-in policy.
pub(crate) const SHARED_LIMIT_BYTES: usize = 48 * 1024;
/// The label recorded beside every timing of this kernel.
#[cfg(test)]
pub(crate) const STAGING: &str =
    "nk32-cp2-b4-nvfp4-rot-rows-k-raw-mxf4nvf4-k64-q-nvfp4-v-coop-cvt-w68-ldmatrix-x4-trans-hilo";

/// Shared bytes one cube uses (module docs, "Shared memory").
pub(crate) fn shared_bytes() -> usize {
    ASYNC_BARRIER_BYTES + STAGES * NK * FP4_BYTES_PER_KEY_HEAD + NK * ASYNC_ROW_WORDS * 4
}

/// The query tiles of one launch.
pub(crate) fn query_tiles(nq: usize) -> usize {
    nq.div_ceil(WARPS)
}

/// Ephemeral launch geometry, not retained model/cache state. Constructed before
/// any scratch allocation; the same split vector is consumed by every launch.
#[derive(Debug)]
pub(crate) struct Plan {
    nq: usize,
    q0: usize,
    query_tiles: usize,
    spans: Vec<usize>,
    split_of: Vec<usize>,
    slots: usize,
    q_elems: usize,
    po_elems: usize,
    pml_elems: usize,
    pub(crate) working_bytes: usize,
}

fn extent(dims: &[usize], width: usize) -> Result<usize, String> {
    let elements = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or("MMA extent overflow")?;
    if elements == 0 || elements.checked_mul(width).is_none_or(|n| n > u32::MAX as usize) {
        return Err("MMA empty or oversized U32 byte extent".into());
    }
    Ok(elements)
}

impl Plan {
    pub(crate) fn new(spans: &[usize], nq: usize, q0: usize) -> Result<Self, String> {
        if !(1..=super::attention::MAX_CHUNK).contains(&nq) || spans.is_empty() {
            return Err("MMA needs T1..256 and key runs".into());
        }
        if (HEADS, KV_HEADS, HEAD_DIM, GROUPS, PLANE) != (32, 2, 128, 16, 32)
            || bf16::cube_type().size_bits() != 16 {
            return Err("MMA requires Nemotron geometry and BF16 register pairs".into());
        }
        let tiles = query_tiles(nq);
        // Padded planes also compute absolute positions before their live guard.
        if q0.checked_add(tiles * WARPS).is_none_or(|n| n > u32::MAX as usize) {
            return Err("MMA padded query position overflow".into());
        }
        spans.iter().try_fold(0usize, |n, &v| n.checked_add(v))
            .filter(|&n| n <= u32::MAX as usize).ok_or("MMA key span overflow")?;
        // Prove the shared split helper's intermediate arithmetic before use.
        let want = (flash::TARGET_CUBES / (tiles * KV_HEADS)).max(1);
        if spans.iter().any(|&v| want.checked_mul(v).is_none()) {
            return Err("MMA split rounding overflow".into());
        }
        // Zero live spans are legal: initialized empty partials.
        let split_of = flash::key_run_splits(spans, tiles, KV_HEADS);
        let slots = split_of.iter().try_fold(0usize, |n, &v| {
            if v == 0 || v > u32::MAX as usize { None } else { n.checked_add(v) }
        }).filter(|&n| n <= u32::MAX as usize).ok_or("MMA invalid split/grid extent")?;
        let q_elems = extent(&[nq, HEADS, HEAD_DIM], 4)?;
        let po_elems = extent(&[slots, nq, HEADS, HEAD_DIM], 4)?;
        let pml_elems = extent(&[slots, nq, HEADS, 2], 4)?;
        // BF16 Q in, F32 output; Q is quantized in registers, never stored.
        let working_bytes = po_elems.checked_add(pml_elems).and_then(|v| v.checked_mul(4))
            .and_then(|v| q_elems.checked_mul(6).and_then(|q| v.checked_add(q)))
            .ok_or("MMA scratch sum overflow")?;
        if shared_bytes() > SHARED_LIMIT_BYTES { return Err("MMA shared storage exceeds 48KiB".into()); }
        Ok(Self { nq, q0, query_tiles: tiles, spans: spans.to_vec(), split_of, slots,
            q_elems, po_elems, pml_elems, working_bytes })
    }

    fn validate_inputs(&self, q: &Handle, runs: &[KeyRun<'_>], scaling: f32) -> Result<(), String> {
        if runs.len() != self.spans.len() || !scaling.is_finite() || scaling <= 0.0
            || q.offset_start.unwrap_or(0) != 0 || q.size_in_used() < (self.q_elems * 2) as u64 {
            return Err("MMA query/run/scaling differs from admitted plan".into());
        }
        for (run, &span) in runs.iter().zip(&self.spans) {
            let (Some(k_scales), Some(v_scales)) = (run.k_scales, run.v_scales) else {
                return Err("MMA runs are NVFP4 codes with their block scales".into());
            };
            let code_bytes = extent(&[run.rows, KV_WIDTH / 2], 1)?;
            let scale_bytes = extent(&[run.rows, KV_WIDTH / fp4quant::GROUP], 1)?;
            // cp.async of 16-byte code units and 8-byte scale rows: a unit
            // starts at byte 128 key + 64 kvh + 16 c, a scale row at 16 key +
            // 8 kvh, so each is aligned iff its buffer's start is.
            let aligned = run.k.offset_start.unwrap_or(0) % 16 == 0
                && run.v.offset_start.unwrap_or(0) % 16 == 0
                && k_scales.offset_start.unwrap_or(0) % 8 == 0
                && v_scales.offset_start.unwrap_or(0) % 8 == 0;
            if run.lo > run.hi || run.hi > run.rows || run.hi - run.lo != span || !aligned
                || run.base.checked_add(run.rows).is_none_or(|n| n > u32::MAX as usize)
                // The key loop increments by NK after its last partial tile.
                || run.hi.checked_add(NK).is_none_or(|n| n > u32::MAX as usize)
                || run.row0 != 0
                || run.k.size_in_used() < code_bytes as u64 || run.v.size_in_used() < code_bytes as u64
                || k_scales.size_in_used() < scale_bytes as u64
                || v_scales.size_in_used() < scale_bytes as u64 {
                return Err("MMA invalid NVFP4 run backing/geometry".into());
            }
        }
        Ok(())
    }
    #[cfg(test)]
    fn validate_splits(&self, splits: &[usize]) -> Result<(), String> {
        if splits != self.split_of || splits.contains(&0) {
            return Err("partials must use admitted positive splits".into());
        }
        Ok(())
    }
}

/// The device capabilities the kernel's instructions need, refused before any
/// allocation: per-thread `cp.async`, the block-scaled e2m1 x e2m1 -> f32
/// `m16n8k64` with `ue4m3` scales one per 16 (`scales_factor` 4), and a pool
/// alignment that keeps every run start 16-byte aligned.
fn device_supports<R: Runtime>(client: &ComputeClient<R>) -> Result<(), String> {
    let props = client.properties();
    let scaled = ScaledMmaConfig {
        a_type: e2m1x2::cube_type(),
        b_type: e2m1x2::cube_type(),
        cd_type: f32::cube_type(),
        scales_type: e4m3::cube_type(),
        m: 16,
        n: 8,
        k: 64,
        scales_factor: 4,
    };
    if !props.features.copy_async || !props.features.matmul.scaled_mma.contains(&scaled)
        || props.memory.alignment % 16 != 0 {
        return Err("MMA needs cp.async, the block-scaled m16n8k64 e2m1/ue4m3 MMA and 16-byte pool \
            alignment".into());
    }
    Ok(())
}

/// Where unit `c` (code words `4 c .. 4 c + 4`, dims `32 c .. 32 c + 32`) of
/// staged NVFP4 row `row` sits, in 16-byte units from the stage's start: the
/// row's four units rotated by half the row index, so the copies, the K reads
/// and the V decode reads are conflict free with no padding.
#[cube]
pub(crate) fn fp4_unit(row: usize, c: usize) -> usize {
    row * FP4_ROW_UNITS + (c + row / 2usize) % FP4_ROW_UNITS
}

/// Tile `t`'s K and V rows of head `kvh` into the code stage at unit offset
/// `code_base` of `skc` / `svc` and the scale stage at row offset `scale_base`
/// of `sks` / `svs`. Code chunk `i = r * units + u` is row `j = i / 4`, unit
/// `c = i % 4`, so a warp instruction covers 8 whole 64-byte head rows; unit
/// `u < NK` copies key `t + u`'s 8 scale bytes. A key `t + j >= s_hi` is a zero
/// fill (source size 0) of all four units and the scale row, from row `t`
/// (callers guarantee `t < s_hi`). Issues only; the caller commits and waits.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_tile_fp4_async(
    kc: &Array<Vector<u32, Const<4>>>,
    vc: &Array<Vector<u32, Const<4>>>,
    ks: &Array<Vector<u32, Const<2>>>,
    vs: &Array<Vector<u32, Const<2>>>,
    skc: &mut SharedMemory<Vector<u32, Const<4>>>,
    svc: &mut SharedMemory<Vector<u32, Const<4>>>,
    sks: &mut SharedMemory<Vector<u32, Const<2>>>,
    svs: &mut SharedMemory<Vector<u32, Const<2>>>,
    t: u32,
    s_hi: u32,
    kvh: usize,
    u: usize,
    code_base: usize,
    scale_base: usize,
) {
    let units = comptime!(WARPS * PLANE as usize);
    let rounds = comptime!(NK * FP4_ROW_UNITS / (WARPS * PLANE as usize));
    let unit_words = comptime!(FP4_UNIT_WORDS as u32);
    let scale_words = comptime!(FP4_SCALE_WORDS as u32);
    #[unroll]
    for r in 0..rounds {
        let i = r * units + u;
        let j = i / FP4_ROW_UNITS;
        let c = i % FP4_ROW_UNITS;
        let key = t + j as u32;
        let live = key < s_hi;
        // The source row, and its length in units: the unit, or nothing.
        let row = select(live, key, t) as usize;
        let len = u32::cast_from(live) as usize;
        let src = row * FP4_KV_ROW_UNITS + kvh * FP4_ROW_UNITS + c;
        let dst = code_base + fp4_unit(j, c);
        copy_async_checked(&kc.slice(src, src + len), &mut skc.slice_mut(dst, dst + 1usize), unit_words);
        copy_async_checked(&vc.slice(src, src + len), &mut svc.slice_mut(dst, dst + 1usize), unit_words);
    }
    // One 8-byte scale row per key: NK <= units, one round.
    if u < NK {
        let key = t + u as u32;
        let live = key < s_hi;
        let row = select(live, key, t) as usize;
        let len = u32::cast_from(live) as usize;
        let src = row * KV_HEADS + kvh;
        let dst = scale_base + u;
        copy_async_checked(&ks.slice(src, src + len), &mut sks.slice_mut(dst, dst + 1usize), scale_words);
        copy_async_checked(&vs.slice(src, src + len), &mut svs.slice_mut(dst, dst + 1usize), scale_words);
    }
}

/// One staged scale row (8 E4M3 bytes, block `b` = byte `b`) to f32 by the
/// hardware pair conversion, `out[b]` block `b`'s scale. Exact: every finite
/// E4M3 value is an f16.
#[cube]
pub(crate) fn fp4_scales_cvt(sc: Vector<u32, Const<2>>, out: &mut Array<f32>) {
    #[unroll]
    for w in 0..2usize {
        #[unroll]
        for h in 0..2usize {
            let bytes = u16::cast_from(sc[w] >> (h * 16usize) as u32);
            let s = Vector::<f32, Const<2>>::cast_from(Vector::<e4m3, Const<2>>::reinterpret(bytes));
            out[w * 4usize + h * 2usize] = s[0usize];
            out[w * 4usize + h * 2usize + 1usize] = s[1usize];
        }
    }
}

/// Unit `c`'s four BF16 pairs for this lane (code words `4 c + i`, byte `t`,
/// `byte_shift = 8 t`): each code byte by one `cvt.rn.f16x2.e2m1x2` (the low
/// nibble, the lower dim, is element 0), widened, times its block's scale in
/// f32 and cast to BF16, exact at both steps. `quad[i]` is dims `32 c + 8 i +
/// 2 t`, `+ 1` under block `2 c + i / 2`.
#[cube]
pub(crate) fn fp4_quad_cvt<N: Size>(
    w: Vector<u32, Const<4>>,
    scales: &Array<f32>,
    c: usize,
    byte_shift: u32,
    quad: &mut Array<Vector<bf16, N>>,
) {
    #[unroll]
    for i in 0..4usize {
        let byte = u8::cast_from(w[i] >> byte_shift);
        let e = Vector::<f32, Const<2>>::cast_from(Vector::<e2m1x2, Const<1>>::reinterpret(byte));
        let s = scales[c * 2usize + i / 2usize];
        let mut f = Vector::<f32, N>::empty();
        f[0usize] = e[0usize] * s;
        f[1usize] = e[1usize] * s;
        quad[i] = Vector::<bf16, N>::cast_from(f);
    }
}

/// The cooperative decode of one staged NVFP4 V tile (code units at
/// `code_base`, scale rows at `scale_base`) into the BF16 tile `out`:
/// row-major `[key][dim]` rows of [`ASYNC_ROW_WORDS`] words, each element
/// written once by one lane of the cube. Warp `warp` takes the 8-row block
/// `warp`; lane `(g, t)` row `8 warp + g` and byte `t` of every code word,
/// stored at word `68 row + 16 c + 4 i + t` (bank `(4 g + t + 16 c + 4 i) %
/// 32`, conflict free). Stores only: the caller syncs before any read.
#[cube]
pub(crate) fn fp4_decode_tile<N: Size>(
    codes: &SharedMemory<Vector<u32, Const<4>>>,
    scales: &SharedMemory<Vector<u32, Const<2>>>,
    out: &mut SharedMemory<Vector<bf16, N>>,
    code_base: usize,
    scale_base: usize,
    warp: usize,
    lane: u32,
) {
    let rounds = comptime!(NK / 8 / WARPS);
    let units = comptime!(FP4_ROW_UNITS);
    let g = lane as usize / 4usize;
    let byte_shift = (lane % 4u32) * 8u32;
    #[unroll]
    for r in 0..rounds {
        let row = (r * WARPS + warp) * 8usize + g;
        let mut sc = Array::<f32>::new(8usize);
        fp4_scales_cvt(scales[scale_base + row], &mut sc);
        let dst = row * ASYNC_ROW_WORDS + lane as usize % 4usize;
        #[unroll]
        for c in 0..units {
            let unit = codes[code_base + fp4_unit(row, c)];
            let mut quad = Array::<Vector<bf16, N>>::new(4usize);
            fp4_quad_cvt::<N>(unit, &sc, c, byte_shift, &mut quad);
            #[unroll]
            for i in 0..4usize {
                out[dst + c * 16usize + i * 4usize] = quad[i];
            }
        }
    }
}

/// Component `t` (0..4) of one staged 16-byte code unit, by three `select`s: a
/// runtime component index into a vector register would not stay in registers.
#[cube]
pub(crate) fn fp4_unit_word(unit: Vector<u32, Const<4>>, t: u32) -> u32 {
    let lo = select(t == 1u32, unit[1usize], unit[0usize]);
    let hi = select(t == 3u32, unit[3usize], unit[2usize]);
    select(t >= 2u32, hi, lo)
}

/// The K operands of one n8 key tile for this lane, straight from the staged
/// rows: `words[2 s + i]` is B reg `i` of k64 step `s` (component `t` of staged
/// unit `2 s + i` of `row`, code word `8 s + 4 i + t`), `scale_words[s]` word
/// `s` of the row's scale row (blocks `4 s .. 4 s + 4`). No decode.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp4_qk_k_operands(
    codes: &SharedMemory<Vector<u32, Const<4>>>,
    scales: &SharedMemory<Vector<u32, Const<2>>>,
    code_base: usize,
    scale_base: usize,
    row: usize,
    tig: u32,
    words: &mut Array<u32>,
    scale_words: &mut Array<u32>,
) {
    let units = comptime!(FP4_ROW_UNITS);
    #[unroll]
    for c in 0..units {
        words[c] = fp4_unit_word(codes[code_base + fp4_unit(row, c)], tig);
    }
    let sc = scales[scale_base + row];
    scale_words[0usize] = sc[0usize];
    scale_words[1usize] = sc[1usize];
}

/// Q for one plane: the 16 heads at pair offset `base` of `q` (zeros when not
/// `live`) to NVFP4 by `fp4quant::quantize_nvfp4`'s rule, straight into the A
/// operand: `codes[4 s + i]` is A reg `i` of k64 step `s`, `scales[s]` the A
/// scale word of step `s` (head `g + 8 (t & 1)`, byte `e` block `4 s + e`).
/// The block amax is the maximum over lanes `t`, `t ^ 1` (exact), the scale
/// word's other parity comes from lane `t ^ 2`, so every lane of the plane
/// calls it. The expressions are `quantize_nvfp4_kernel`'s, written the same
/// way, so the bytes are its bytes (no fast-math in either compile).
#[cube]
pub(crate) fn fp4_q_quantize<N: Size>(
    q: &Array<Vector<bf16, N>>,
    base: usize,
    live: bool,
    lane: u32,
    codes: &mut Array<Vector<e2m1x2, Const<4>>>,
    scales: &mut Array<u32>,
) {
    let head_pairs = comptime!(HEAD_DIM / 2);
    let g = lane as usize / 4usize;
    let tig = lane % 4u32;
    let pair0 = tig as usize * 4usize;
    // full[4 hh + j]: the amax of block 2 j + t / 2 of head g + 8 hh, the
    // block this lane holds dims 32 j + 8 t .. + 8 of.
    let mut full = Array::<f32>::new(8usize);
    #[unroll]
    for hh in 0..2usize {
        #[unroll]
        for j in 0..4usize {
            let at = base + (g + hh * 8usize) * head_pairs + j * 16usize + pair0;
            let mut x = Vector::<f32, Const<8>>::empty();
            #[unroll]
            for k in 0..4usize {
                let mut pair = Vector::<bf16, N>::cast_from(0.0f32);
                if live {
                    pair = q[at + k];
                }
                let f = Vector::<f32, N>::cast_from(pair);
                x[k * 2usize] = f[0usize];
                x[k * 2usize + 1usize] = f[1usize];
            }
            let a = x.abs();
            let part = max(max(max(a[0usize], a[1usize]), max(a[2usize], a[3usize])),
                max(max(a[4usize], a[5usize]), max(a[6usize], a[7usize])));
            let amax = max(part, plane_shuffle_xor(part, 1u32));
            full[hh * 4usize + j] = amax;
            // fp4quant's expressions, operation for operation.
            let sf = min(amax / 6.0, f32::new(448.0f32));
            let s = f32::cast_from(e4m3::cast_from(sf));
            let mut r = f32::new(0.0f32);
            if s > 0.0 {
                r = 1.0 / s;
            }
            let rv = Vector::<f32, Const<8>>::new(r);
            codes[(j / 2usize) * 4usize + (j % 2usize) * 2usize + hh] =
                Vector::<e2m1x2, Const<4>>::cast_from(x * rv);
        }
    }
    // The scale words: head g + 8 (t & 1), all eight blocks. This lane has
    // the four of parity t / 2 (block 2 j + t / 2); lane t ^ 2 the other four.
    let odd = (tig & 1u32) == 1u32;
    let parity = tig / 2u32;
    let mut own = Array::<f32>::new(4usize);
    let mut other = Array::<f32>::new(4usize);
    #[unroll]
    for j in 0..4usize {
        let mine = select(odd, full[4usize + j], full[j]);
        own[j] = mine;
        other[j] = plane_shuffle_xor(mine, 2u32);
    }
    #[unroll]
    for w in 0..2usize {
        let mut sv = Vector::<e4m3, Const<4>>::empty();
        #[unroll]
        for e in 0..4usize {
            // Byte e is block 4 w + e = 2 j + e % 2.
            let j = w * 2usize + e / 2usize;
            let amax = select((e % 2usize) as u32 == parity, own[j], other[j]);
            sv[e] = e4m3::cast_from(min(amax / 6.0, f32::new(448.0f32)));
        }
        scales[w] = u32::reinterpret(sv);
    }
}

/// Raw S over one n8 key tile: two block-scaled `mma.m16n8k64`, step 0 then 1,
/// chained from `+0.0`, A from [`fp4_q_quantize`], B from
/// [`fp4_qk_k_operands`]. `out` gets the accumulator in the `m16n8k16` layout
/// (2 pairs of f32, rows `g`, `g + 8`, keys `2 t`, `2 t + 1`). `scaling` is not
/// applied.
#[cube]
pub(crate) fn fp4_qk_tile<N: Size>(
    qa: &Array<Vector<e2m1x2, Const<4>>>,
    qsw: &Array<u32>,
    kw: &Array<u32>,
    ksw: &Array<u32>,
    out: &mut Array<Vector<f32, N>>,
) {
    let def = cmma::MmaDefinition::<e2m1x2, e2m1x2, f32>::new_scaled::<e4m3>(16usize, 8usize, 64usize,
        4usize);
    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);
    // The maps above assume these per-lane counts, and the masking reads S by
    // the BF16 def's accumulator map: if CubeCL ever disagrees, expansion fails.
    comptime!(assert!(
        (vc_a, vs_a, vc_b, vs_b, ec_c, vs_c) == (4, 4, 2, 4, 4, 2),
        "the block-scaled m16n8k64 per-lane counts are not 4 + 2 b32 operands and 2 f32 pairs"
    ));
    let mut c = Array::<Vector<f32, N>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        c[i] = Vector::<f32, N>::cast_from(0.0f32);
    }
    #[unroll]
    for s in 0..2usize {
        let mut a = Array::<Vector<e2m1x2, Const<4>>>::new(vc_a);
        #[unroll]
        for i in 0..vc_a {
            a[i] = qa[s * 4usize + i];
        }
        let mut b = Array::<Vector<e2m1x2, Const<4>>>::new(vc_b);
        #[unroll]
        for i in 0..vc_b {
            b[i] = Vector::<e2m1x2, Const<4>>::reinterpret(kw[s * 2usize + i]);
        }
        // `mut` is load-bearing: CubeCL binds the scale operands by
        // `reinterpret_cast<uint32&>`, which NVRTC refuses on the `const`
        // local a plain `let` emits (fp4gemm builds its scales the same way).
        #[allow(unused_mut)]
        let mut sa = Vector::<e4m3, Const<4>>::reinterpret(qsw[s]);
        #[allow(unused_mut)]
        let mut sb = Vector::<e4m3, Const<4>>::reinterpret(ksw[s]);
        let d = def.execute_scaled(&a, &b, &c, sa, sb);
        #[unroll]
        for i in 0..vc_c {
            c[i] = d[i];
        }
    }
    #[unroll]
    for i in 0..vc_c {
        out[i] = c[i];
    }
}

/// One `(query tile, KV head, key split)`: writes `flash_kernel`'s unnormalised
/// partial for each of its live query positions and all 16 heads.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mma_attention_kernel<N: Size>(
    q: &Array<Vector<bf16, N>>,
    kcodes: &Array<Vector<u32, Const<4>>>,
    vcodes: &Array<Vector<u32, Const<4>>>,
    kscales: &Array<Vector<u32, Const<2>>>,
    vscales: &Array<Vector<u32, Const<2>>>,
    po: &mut Array<Vector<f32, N>>,
    pml: &mut Array<f32>,
    scaling: f32,
    nq: u32,
    q0: u32,
    k0: u32,
    klo: u32,
    khi: u32,
    slot0: u32,
    splits: u32,
) {
    let def = cmma::MmaDefinition::<bf16, bf16, f32>::new(16usize, 8usize, 16usize);
    // Created in the entry block, so CubeCL's shared allocator places it at
    // offset 0, before any stage buffer ([`ASYNC_BARRIER_BYTES`]).
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    let qt = CUBE_POS_X;
    let kvh = CUBE_POS_Y as usize;
    let split = CUBE_POS_Z;
    let u = UNIT_POS_X as usize;
    let lane = UNIT_POS_PLANE;
    let warp = UNIT_POS_X / PLANE;
    let nk32 = comptime!(NK as u32);
    let n_tiles = comptime!(NK / 8);
    let pv_steps = comptime!(NK / 16);

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);
    comptime!(assert!(
        (ec_c, vs_c) == (4, 2),
        "S tiles are stored by the m16n8k16 accumulator map: 4 f32 per lane as 2 pairs"
    ));

    // Q index arithmetic is in BF16 PAIRS (host-asserted two elements).
    let q_row_pairs = comptime!(HEADS * HEAD_DIM / 2);

    let bq = comptime!(WARPS as u32);
    let p = qt * bq + warp;
    let live = p < nq;
    let q_abs = q0 + p;

    // --- Q as NVFP4, once, straight into the k64 A operand ------------------
    let mut qa = Array::<Vector<e2m1x2, Const<4>>>::new(8usize);
    let mut qsw = Array::<u32>::new(2usize);
    let at = p as usize * q_row_pairs + kvh * comptime!(GROUPS * HEAD_DIM / 2);
    fp4_q_quantize::<N>(q, at, live, lane, &mut qa, &mut qsw);

    // --- this split's slice of the key axis: flash_kernel's trimming ---------
    let q_lo = q0 + qt * bq;
    let mut q_hi = q0 + nq - 1;
    if qt * bq + bq <= nq {
        q_hi = q_lo + bq - 1;
    }
    let lo = klo;
    let mut hi = khi;
    if q_hi + 1 <= k0 {
        hi = lo;
    } else if q_hi + 1 - k0 < hi {
        hi = q_hi + 1 - k0;
    }
    if hi < lo {
        hi = lo;
    }
    let span = hi - lo;
    let per = (span + splits - 1) / splits;
    let mut s_lo = lo + split * per;
    if s_lo > hi {
        s_lo = hi;
    }
    let mut s_hi = s_lo + per;
    if s_hi > hi {
        s_hi = hi;
    }

    // --- running state: two rows per lane, one per accumulator vector --------
    let mut m = Array::<f32>::new(vc_c);
    let mut l = Array::<f32>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        m[i] = f32::new(NEG_INF);
        l[i] = f32::new(0.0f32);
    }
    let o_vectors = comptime!(16 * vc_c);
    let mut of = Array::<Vector<f32, N>>::new(o_vectors);
    #[unroll]
    for i in 0..o_vectors {
        of[i] = Vector::<f32, N>::cast_from(0.0f32);
    }
    let mut sacc = Array::<Vector<f32, N>>::new(comptime!(n_tiles * vc_c));

    // Two NVFP4 stages of K and V code rows (16-byte units, [`fp4_unit`]'s
    // rotation) and scale rows (8 bytes): stage b at unit offset b * NK * 4 /
    // row offset b * NK. One BF16 V tile the decode fills every tile.
    let mut svt = SharedMemory::<Vector<bf16, N>>::new_aligned(comptime!(NK * ASYNC_ROW_WORDS), 16usize);
    let mut skc = SharedMemory::<Vector<u32, Const<4>>>::new_aligned(
        comptime!(STAGES * NK * FP4_ROW_UNITS), 16usize);
    let mut svc = SharedMemory::<Vector<u32, Const<4>>>::new_aligned(
        comptime!(STAGES * NK * FP4_ROW_UNITS), 16usize);
    let mut sks = SharedMemory::<Vector<u32, Const<2>>>::new_aligned(comptime!(STAGES * NK), 16usize);
    let mut svs = SharedMemory::<Vector<u32, Const<2>>>::new_aligned(comptime!(STAGES * NK), 16usize);
    let code_stride = comptime!(NK * FP4_ROW_UNITS);
    let scale_stride = comptime!(NK);

    // The split's first tile into stage 0, completing phase 0. An empty split
    // issues nothing and still arrives (cube-uniform: s_lo, s_hi are the cube's).
    if s_lo < s_hi {
        copy_tile_fp4_async(kcodes, vcodes, kscales, vscales, &mut skc, &mut svc, &mut sks, &mut svs,
            s_lo, s_hi, kvh, u, 0usize, 0usize);
    }
    barrier.commit_copy_async();
    barrier.arrive();

    let g = lane as usize / 4usize;
    let tig = lane % 4u32;
    let mut t = s_lo;
    while t < s_hi {
        let stage = ((t - s_lo) / nk32 % 2u32) as usize;
        // Phase n complete: tile n is in stage n % 2 and every unit has
        // finished tile n - 1, so the other stage and the V tile are free.
        barrier.wait_parity(stage as u32);
        let next = t + nk32;
        if next < s_hi {
            copy_tile_fp4_async(kcodes, vcodes, kscales, vscales, &mut skc, &mut svc, &mut sks,
                &mut svs, next, s_hi, kvh, u, (1usize - stage) * code_stride,
                (1usize - stage) * scale_stride);
        }
        barrier.commit_copy_async();
        let cb = stage * code_stride;
        let sb = stage * scale_stride;

        // --- V decoded once by the cube, then ordered before every read ----
        fp4_decode_tile::<N>(&svc, &svs, &mut svt, cb, sb, warp as usize, lane);
        sync_cube();

        // --- S = Q K^T: K's staged words raw into the block-scaled k64 ------
        #[unroll]
        for nt in 0..n_tiles {
            let mut kw = Array::<u32>::new(4usize);
            let mut ksw = Array::<u32>::new(2usize);
            fp4_qk_k_operands(&skc, &sks, cb, sb, nt * 8usize + g, tig, &mut kw, &mut ksw);
            let mut c = Array::<Vector<f32, N>>::new(vc_c);
            fp4_qk_tile::<N>(&qa, &qsw, &kw, &ksw, &mut c);
            #[unroll]
            for i in 0..vc_c {
                sacc[nt * vc_c + i] = c[i];
            }
        }

        // --- masking and the online update, per row this lane holds ----------
        // Every accumulator element gets its own key-column predicate: inside
        // this split (tail), causal against this plane's ABSOLUTE position,
        // and a live plane. Masked elements are the finite NEG_INF and their
        // probability is exactly 0, as in flash_kernel.
        #[unroll]
        for i in 0..vc_c {
            let (_row, col) =
                def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
            let mut tile_max = f32::new(NEG_INF);
            #[unroll]
            for nt in 0..n_tiles {
                let mut sv = sacc[nt * vc_c + i];
                #[unroll]
                for e in 0..vs_c {
                    let key = t + (nt * 8usize + e) as u32 + col;
                    let vis = live && key < s_hi && k0 + key <= q_abs;
                    let mut s = sv[e] * scaling;
                    if !vis {
                        s = f32::new(NEG_INF);
                    }
                    sv[e] = s;
                    if s > tile_max {
                        tile_max = s;
                    }
                }
                sacc[nt * vc_c + i] = sv;
            }
            // The four lanes of a quad hold the same row.
            let x1 = plane_shuffle_xor(tile_max, 1u32);
            if x1 > tile_max {
                tile_max = x1;
            }
            let x2 = plane_shuffle_xor(tile_max, 2u32);
            if x2 > tile_max {
                tile_max = x2;
            }
            let mut m_new = m[i];
            if tile_max > m_new {
                m_new = tile_max;
            }
            let mut alpha = f32::new(1.0f32);
            if m_new > f32::new(NEG_INF) {
                alpha = Exp::exp(m[i] - m_new);
            }
            let mut row_sum = f32::new(0.0f32);
            #[unroll]
            for nt in 0..n_tiles {
                let mut sv = sacc[nt * vc_c + i];
                #[unroll]
                for e in 0..vs_c {
                    let key = t + (nt * 8usize + e) as u32 + col;
                    let vis = live && key < s_hi && k0 + key <= q_abs;
                    let mut pe = f32::new(0.0f32);
                    if vis {
                        pe = Exp::exp(sv[e] - m_new);
                    }
                    sv[e] = pe;
                    row_sum += pe;
                }
                sacc[nt * vc_c + i] = sv;
            }
            row_sum += plane_shuffle_xor(row_sum, 1u32);
            row_sum += plane_shuffle_xor(row_sum, 2u32);
            l[i] = l[i] * alpha + row_sum;
            m[i] = m_new;
            #[unroll]
            for dt in 0..16usize {
                of[dt * vc_c + i] = of[dt * vc_c + i] * Vector::<f32, N>::cast_from(alpha);
            }
        }

        // --- O += P V: P's A fragment is two adjacent S accumulator tiles ----
        // B(k = key, n = dim) by ldmatrix.x4.trans: lane L gives V row
        // 16 kk + L % 16, dims (2 dt2 + L / 16) * 8, so quad 0, 1 are dim tile
        // 2 dt2's B fragment and 2, 3 dim tile 2 dt2 + 1's.
        let v_lane = (lane as usize % 16usize) * ASYNC_ROW_WORDS + (lane as usize / 16usize) * 4usize;
        #[unroll]
        for kk in 0..pv_steps {
            let mut ph = Array::<Vector<bf16, N>>::new(vc_a);
            let mut pl = Array::<Vector<bf16, N>>::new(vc_a);
            #[unroll]
            for i in 0..vc_c {
                let s0 = sacc[(kk * 2usize) * vc_c + i];
                let s1 = sacc[(kk * 2usize + 1usize) * vc_c + i];
                let h0 = Vector::<bf16, N>::cast_from(s0);
                let h1 = Vector::<bf16, N>::cast_from(s1);
                ph[i] = h0;
                ph[i + vc_c] = h1;
                pl[i] = Vector::<bf16, N>::cast_from(s0 - Vector::<f32, N>::cast_from(h0));
                pl[i + vc_c] = Vector::<bf16, N>::cast_from(s1 - Vector::<f32, N>::cast_from(h1));
            }
            #[unroll]
            for dt2 in 0..8usize {
                let at = kk * comptime!(16 * ASYNC_ROW_WORDS) + dt2 * 8usize + v_lane;
                let quad = def.load_matrix::<_, N>(&svt.slice(at, at + ASYNC_CHUNK_PAIRS),
                    MatrixIdent::B, 4usize, true);
                #[unroll]
                for h in 0..2usize {
                    let dt = dt2 * 2usize + h;
                    let mut b = Array::<Vector<bf16, N>>::new(vc_b);
                    #[unroll]
                    for i in 0..vc_b {
                        b[i] = quad[h * vc_b + i];
                    }
                    let mut c = Array::<Vector<f32, N>>::new(vc_c);
                    #[unroll]
                    for i in 0..vc_c {
                        c[i] = of[dt * vc_c + i];
                    }
                    let d = def.execute(&ph, &b, &c);
                    let d2 = def.execute(&pl, &b, &d);
                    #[unroll]
                    for i in 0..vc_c {
                        of[dt * vc_c + i] = d2[i];
                    }
                }
            }
        }
        // This tile's reads are done: completes phase n + 1, which the next
        // tile waits on before it issues tile n + 2 into this stage.
        barrier.arrive();

        t += nk32;
    }

    // --- the partial, in flash_kernel's layout --------------------------------
    // po[((slot * nq + qrow) * heads + h) * head_dim + d] = o (unnormalised),
    // pml[((slot * nq + qrow) * heads + h) * 2 + {0, 1}] = (m, l). A fully
    // masked plane or split leaves m = NEG_INF, l = 0, o = 0, which the
    // combine weights to zero.
    let slot = slot0 + split;
    if live {
        #[unroll]
        for i in 0..vc_c {
            let (row, col) =
                def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
            let h = kvh as u32 * comptime!(GROUPS as u32) + row;
            let cell = (slot * nq + p) * comptime!(HEADS as u32) + h;
            let base = cell * comptime!(HEAD_DIM as u32);
            #[unroll]
            for dt in 0..16usize {
                po[((base + (dt * 8usize) as u32 + col) / 2) as usize] = of[dt * vc_c + i];
            }
            if col == 0 {
                pml[(cell * 2) as usize] = m[i];
                pml[(cell * 2 + 1) as usize] = l[i];
            }
        }
    }
}

/// The partial slots of one launch: `flash::key_run_splits` over the runs'
/// live spans, at this launch's query tiles. Their sum is the slot count.
#[cfg(test)]
pub(crate) fn split_plan(runs: &[KeyRun<'_>], nq: usize) -> Vec<usize> {
    let spans: Vec<usize> = runs.iter().map(|r| r.hi.saturating_sub(r.lo)).collect();
    flash::key_run_splits(&spans, query_tiles(nq), KV_HEADS)
}

/// The partial pass alone: every run's `mma_attention_kernel` launches, writing
/// `flash_kernel`'s partial layout into `po` (`po_len` F32) and `pml`
/// (`pml_len` F32) for the slots `split_of` names. The buffers may be LONGER
/// than the `slots * nq` rows the launch owns (a test's canary region); the
/// kernel must write nothing past them.
#[allow(clippy::too_many_arguments)]
fn launch_partials<R: Runtime>(
    client: &ComputeClient<R>,
    q: &Handle,
    runs: &[KeyRun<'_>],
    scaling: f32,
    plan: &Plan,
    po: &Handle,
    po_len: usize,
    pml: &Handle,
    pml_len: usize,
) {
    // The only callers validate the complete plan, input runs and output
    // extents before allocation/launch. Consume that geometry without replanning.
    let (nq, q0, split_of) = (plan.nq, plan.q0, plan.split_of.as_slice());
    let vs = 32 / bf16::cube_type().size_bits();
    let q_elems = plan.q_elems;
    let q_tiles = plan.query_tiles;
    let mut slot0 = 0usize;
    for (run, splits) in runs.iter().zip(split_of.iter().copied()) {
        let (Some(k_scales), Some(v_scales)) = (run.k_scales, run.v_scales) else {
            unreachable!("validated NVFP4 runs");
        };
        let code_words = run.rows * KV_WIDTH / fp4quant::CODES_PER_WORD;
        let scale_words = run.rows * KV_WIDTH / fp4quant::GROUP / 4;
        let count = CubeCount::Static(q_tiles as u32, KV_HEADS as u32, splits as u32);
        let dim = CubeDim::new_1d((WARPS * PLANE as usize) as u32);
        unsafe {
            mma_attention_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                vs,
                ArrayArg::from_raw_parts(q.clone(), q_elems),
                ArrayArg::from_raw_parts(run.k.clone(), code_words),
                ArrayArg::from_raw_parts(run.v.clone(), code_words),
                ArrayArg::from_raw_parts(k_scales.clone(), scale_words),
                ArrayArg::from_raw_parts(v_scales.clone(), scale_words),
                ArrayArg::from_raw_parts(po.clone(), po_len),
                ArrayArg::from_raw_parts(pml.clone(), pml_len),
                scaling,
                nq as u32,
                q0 as u32,
                run.base as u32,
                run.lo as u32,
                run.hi as u32,
                slot0 as u32,
                splits as u32,
            )
        };
        slot0 += splits;
    }
}

/// Tensor-core attention over one BF16 query block and any number of NVFP4 key
/// runs, Nemotron geometry only.
///
/// `q` is `[nq, 32 * 128]` BF16, contiguous from offset 0; each run is the
/// cache's NVFP4 rows (module docs, "K/V as stored") with `base` its ABSOLUTE
/// position of row 0. Returns the `[nq, 32 * 128]` F32 handle layout and dtype
/// `flash::flash_attention_launch` returns.
#[cfg(test)]
pub(crate) fn mma_attention_launch<R: Runtime>(
    client: &ComputeClient<R>,
    q: &Handle,
    runs: &[KeyRun<'_>],
    nq: usize,
    q0: usize,
    scaling: f32,
) -> Handle {
    let spans: Vec<_> = runs.iter().map(|r| r.hi.checked_sub(r.lo).expect("invalid run")).collect();
    let plan = Plan::new(&spans, nq, q0).expect("MMA plan");
    launch_planned(client, q, runs, scaling, &plan).expect("MMA input")
}

pub(crate) fn launch_planned<R: Runtime>(client: &ComputeClient<R>, q: &Handle,
    runs: &[KeyRun<'_>], scaling: f32, plan: &Plan) -> Result<Handle, String>
{
    plan.validate_inputs(q, runs, scaling)?; // ALL runs before ANY allocation.
    device_supports(client)?;
    let (nq, slots, po_elems, pml_elems) = (plan.nq, plan.slots, plan.po_elems, plan.pml_elems);
    let po = client.empty(po_elems * core::mem::size_of::<f32>());
    let pml = client.empty(pml_elems * core::mem::size_of::<f32>());
    launch_partials(
        client, q, runs, scaling, plan, &po, po_elems, &pml, pml_elems,
    );

    let q_elems = plan.q_elems;
    let out = client.empty(q_elems * core::mem::size_of::<f32>());
    unsafe {
        flash::flash_combine_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(nq as u32, HEADS as u32, 1),
            CubeDim::new_1d(HEAD_DIM as u32),
            ArrayArg::from_raw_parts(po, po_elems),
            ArrayArg::from_raw_parts(pml, pml_elems),
            ArrayArg::from_raw_parts(out.clone(), q_elems),
            slots as u32,
            nq as u32,
            HEADS as u32,
            HEAD_DIM as u32,
        )
    };
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mma_attention_partials<R: Runtime>(client: &ComputeClient<R>, q: &Handle,
    runs: &[KeyRun<'_>], nq: usize, q0: usize, scaling: f32,
    split_of: &[usize], po: &Handle, po_len: usize, pml: &Handle, pml_len: usize)
{
    let spans: Vec<_> = runs.iter().map(|r| r.hi.checked_sub(r.lo).expect("invalid run")).collect();
    let plan = Plan::new(&spans, nq, q0).expect("MMA plan");
    plan.validate_inputs(q, runs, scaling).expect("MMA inputs");
    device_supports(client).expect("MMA device");
    plan.validate_splits(split_of).expect("MMA splits");
    extent(&[po_len], 4).expect("partial extent");
    extent(&[pml_len], 4).expect("statistics extent");
    assert!(po_len >= plan.po_elems && pml_len >= plan.pml_elems);
    assert!(po.size_in_used() >= (po_len * 4) as u64 && pml.size_in_used() >= (pml_len * 4) as u64);
    launch_partials(client, q, runs, scaling, &plan, po, po_len, pml, pml_len);
}

#[cfg(test)]
#[path = "attention_mma_tests.rs"]
mod tests;
