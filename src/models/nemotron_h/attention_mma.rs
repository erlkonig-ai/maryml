//! Nemotron-H attention on tensor cores, phase A (BF16 Q, BF16 K/V runs).
//!
//! Explicitly selected by the native attention owner; scalar stays the default.
//! Checked Plan is shared by admission and launch. Kernel arithmetic is the
//! frozen NK64 single-buffer phase-A implementation, not the NK32 experiment.
//!
//! # Shape
//!
//! Nemotron's geometry is fixed here: 32 query heads over 2 KV heads (16 heads
//! per group), `head_dim` 128, no relative bias (`eff` 0), no window, no gaps.
//! The grid is `(query tiles, KV heads, key splits)`; a cube is `warps` planes
//! (4 or 8, comptime), and plane `w` owns query position `p = tile * warps + w`
//! and ALL 16 heads of the cube's KV head: the 16 rows of its `m16` fragment
//! are heads `kvh * 16 + 0..16`. Those rows share every key, value and causal
//! bound, so masking is a per-key-column predicate shared by the plane.
//!
//! Per key tile of [`NK`] keys:
//! * K `[64 keys][128 dims]` and V transposed `[128 dims][64 keys]` are staged
//!   cooperatively in shared memory as BF16 pairs, rows padded so the B
//!   fragment reads are bank-conflict free (see [`K_ROW_WORDS`]), and the
//!   V^T staging stores mapped so they are too (see [`V_STAGING`]).
//! * `S = Q K^T`: 8 n8 key tiles x 8 k16 dim steps of `mma.m16n8k16` per plane,
//!   with Q held in registers as 8 k16 A fragments for the whole launch.
//! * Online softmax on the accumulator fragment, `flash_kernel`'s guarded
//!   recurrence: each lane holds two rows, row max and row sum reduce across
//!   the four lanes of a quad (`plane_shuffle_xor` 1, 2).
//! * `P` becomes the k16 A fragment of `O += P V` in registers by the mapping
//!   measured bitwise on this GB10 (`mma_probe_tests.rs`): the A fragment for
//!   keys `16 kk .. 16 kk + 16` is n8 tile `2 kk`'s accumulator vectors
//!   followed by n8 tile `2 kk + 1`'s, each cast to BF16. [`PMode::HiLo`]
//!   splits `p = hi + lo` into two BF16 operands and issues both MMAs into the
//!   same accumulator; [`PMode::Bf16`] issues one.
//!
//! The partials are written in EXACTLY `flash_kernel`'s layout and
//! normalisation (unnormalised `o`, then `m`, `l`), and the existing
//! `flash_combine_kernel` turns them into the same `[nq, 32 * 128]` F32 output
//! `flash_attention_launch` returns. Splits come from the same
//! `flash::key_run_splits`, and each split's key range is trimmed by the
//! cube's causal bound exactly as `flash_kernel` trims it.

use super::attention::{HEADS, HEAD_DIM, KV_HEADS};
use crate::models::inkling::flash::{self, KeyRun, NEG_INF, PLANE};
use cubecl::{ir::MatrixIdent, prelude::*, server::Handle};
use half::bf16;

/// Query heads per KV head.
const GROUPS: usize = HEADS / KV_HEADS;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
/// Keys per staged tile.
pub(crate) const NK: usize = 64;
/// 32-bit words (BF16 pairs) per staged K row: 64 for the 128 dims, plus 4 of
/// padding. A B-fragment read has lane `(g, t)` (key `g`, dim pair `t`) on word
/// `68 g + t + c`, i.e. bank `(4 g + t + c) % 32`: all 32 distinct. Unpadded
/// (64) every key would land on the same 4 banks, an 8-way conflict.
pub(crate) const K_ROW_WORDS: usize = HEAD_DIM / 2 + 4;
/// 32-bit words (key pairs) per staged transposed V row: 32 for the 64 keys,
/// plus 4 of padding, conflict-free for the same reason.
pub(crate) const V_ROW_WORDS: usize = NK / 2 + 4;
/// Which cooperative V^T staging lane map the kernel runs, recorded beside
/// every timing figure: 4 dim pairs x 8 key pairs per warp store, bank
/// conflict free on the store as well as on the B-fragment read. (The first
/// variant, 32 consecutive dim pairs at one key pair, was 8-way on the store.)
pub(crate) const V_STAGING: &str = "vt-store-4dp-x-8kp";
/// Fits the existing 48KiB block limit; no opt-in shared-memory policy.
pub(crate) const SHARED_LIMIT_BYTES: usize = 48 * 1024;

/// How `P` enters the PV tensor-core product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PMode {
    /// `bf16(p)`: one PV MMA per k16 step (the fast candidate).
    Bf16,
    /// `hi = bf16(p)`, `lo = bf16(p - f32(hi))`: two PV MMAs into the same
    /// accumulator (the diagnostic reference arm).
    HiLo,
}
impl PMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::HiLo => "hilo",
        }
    }
}

/// Shared bytes one cube uses. Independent of `warps`: Q, S, P and O never
/// leave registers, so only the K and transposed V tiles are staged.
pub(crate) fn shared_bytes() -> usize {
    (NK * K_ROW_WORDS + HEAD_DIM * V_ROW_WORDS) * 4
}

/// The query tiles of one launch at `warps` query positions per cube.
pub(crate) fn query_tiles(nq: usize, warps: usize) -> usize {
    nq.div_ceil(warps)
}

/// Ephemeral launch geometry, not retained model/cache state. Constructed before
/// any scratch allocation; the same split vector is consumed by every launch.
#[derive(Debug)]
pub(crate) struct Plan {
    nq: usize,
    q0: usize,
    query_tiles: usize,
    warps: usize,
    mode: PMode,
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
    pub(crate) fn new(spans: &[usize], nq: usize, q0: usize, warps: usize, mode: PMode) -> Result<Self, String> {
        if !(1..=super::attention::MAX_CHUNK).contains(&nq) || ![4, 8].contains(&warps) || spans.is_empty() {
            return Err("MMA needs T1..256, B4/B8 and key runs".into());
        }
        if (HEADS, KV_HEADS, HEAD_DIM, GROUPS, PLANE) != (32, 2, 128, 16, 32)
            || bf16::cube_type().size_bits() != 16 {
            return Err("MMA requires Nemotron geometry and BF16 register pairs".into());
        }
        let tiles = query_tiles(nq, warps);
        // Padded planes also compute absolute positions before their live guard.
        if q0.checked_add(tiles * warps).is_none_or(|n| n > u32::MAX as usize) {
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
        let working_bytes = po_elems.checked_add(pml_elems).and_then(|v| v.checked_mul(4))
            .and_then(|v| q_elems.checked_mul(6).and_then(|q| v.checked_add(q)))
            .ok_or("MMA scratch sum overflow")?;
        if shared_bytes() > SHARED_LIMIT_BYTES { return Err("MMA shared storage exceeds 48KiB".into()); }
        Ok(Self { nq, q0, query_tiles: tiles, warps, mode, spans: spans.to_vec(), split_of, slots,
            q_elems, po_elems, pml_elems, working_bytes })
    }

    fn validate_inputs(&self, q: &Handle, runs: &[KeyRun<'_>], scaling: f32) -> Result<(), String> {
        if runs.len() != self.spans.len() || !scaling.is_finite() || scaling <= 0.0
            || q.offset_start.unwrap_or(0) != 0 || q.size_in_used() < (self.q_elems * 2) as u64 {
            return Err("MMA query/run/scaling differs from admitted plan".into());
        }
        for (run, &span) in runs.iter().zip(&self.spans) {
            let elements = extent(&[run.rows, KV_WIDTH], 2)?;
            if run.lo > run.hi || run.hi > run.rows || run.hi - run.lo != span
                || run.base.checked_add(run.rows).is_none_or(|n| n > u32::MAX as usize)
                // The key loop increments by NK after its last partial tile.
                || run.hi.checked_add(NK).is_none_or(|n| n > u32::MAX as usize)
                || run.k_scales.is_some() || run.v_scales.is_some() || run.row0 != 0
                || run.k.size_in_used() < (elements * 2) as u64
                || run.v.size_in_used() < (elements * 2) as u64 {
                return Err("MMA invalid dense BF16 run backing/geometry".into());
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

/// One `(query tile, KV head, key split)`: writes `flash_kernel`'s unnormalised
/// partial for each of its live query positions and all 16 heads.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mma_attention_kernel<N: Size>(
    q: &Array<Vector<bf16, N>>,
    k: &Array<Vector<bf16, N>>,
    v: &Array<Vector<bf16, N>>,
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
    #[comptime] warps: usize,
    #[comptime] hilo: bool,
) {
    let def = cmma::MmaDefinition::<bf16, bf16, f32>::new(16usize, 8usize, 16usize);
    let qt = CUBE_POS_X;
    let kvh = CUBE_POS_Y as usize;
    let split = CUBE_POS_Z;
    let u = UNIT_POS_X as usize;
    let lane = UNIT_POS_PLANE;
    let warp = UNIT_POS_X / PLANE;
    let units = comptime!(warps * PLANE as usize);

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    // Index arithmetic below is in BF16 PAIRS (the vector unit of every
    // operand array), host-asserted to be two elements.
    let q_row_pairs = comptime!(HEADS * HEAD_DIM / 2);
    let kv_row_pairs = comptime!(KV_WIDTH / 2);
    let head_pairs = comptime!(HEAD_DIM / 2);

    // This plane's query position, chunk-local. Padded planes (p >= nq) take
    // part in every cooperative load and every barrier, and write nothing.
    let bq = comptime!(warps as u32);
    let p = qt * bq + warp;
    let live = p < nq;
    let q_abs = q0 + p;

    // --- Q: 16 heads x 128 dims as 8 k16 A fragments, once -------------------
    let mut qf = Array::<Vector<bf16, N>>::new(comptime!(8 * vc_a));
    #[unroll]
    for kk in 0..8usize {
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a) as u32, MatrixIdent::A);
            let mut val = Vector::<bf16, N>::cast_from(0.0f32);
            if live {
                let h = kvh * GROUPS + row as usize;
                let at = p as usize * q_row_pairs + h * head_pairs + (kk * 16usize + col as usize) / 2usize;
                val = q[at];
            }
            qf[kk * vc_a + i] = val;
        }
    }

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
    let mut sacc = Array::<Vector<f32, N>>::new(comptime!(8 * vc_c));

    let mut sk = SharedMemory::<Vector<bf16, N>>::new(comptime!(NK * K_ROW_WORDS));
    let mut svt = SharedMemory::<Vector<bf16, N>>::new(comptime!(HEAD_DIM * V_ROW_WORDS));

    // Cooperative staging rounds: pairs per tile over the cube's units.
    let k_rounds = comptime!(NK * HEAD_DIM / 2 / units);
    let v_rounds = comptime!(NK / 2 * HEAD_DIM / 2 / units);
    let mut t = s_lo;
    while t < s_hi {
        // --- stage K [key][dim] and V^T [dim][key] --------------------------
        // Rows at or past s_hi (the next split's keys, the run's dead rows or
        // its NaN tail) are ZERO, never read: a masked P of 0 times a NaN V
        // would still be NaN.
        #[unroll]
        for r in 0..k_rounds {
            let i = r * units + u;
            let j = i / head_pairs;
            let dp = i % head_pairs;
            let key = t + j as u32;
            let mut val = Vector::<bf16, N>::cast_from(0.0f32);
            if key < s_hi {
                val = k[key as usize * kv_row_pairs + kvh * head_pairs + dp];
            }
            sk[j * K_ROW_WORDS + dp] = val;
        }
        // V^T lane map ([`V_STAGING`]): i = ((kp_hi * 16 + dp_hi) * 8 + kp_lo)
        // * 4 + dp_lo, so the 32 lanes of one store instruction cover 4
        // consecutive dim pairs x 8 consecutive key pairs. The store word
        // 72 dp + kp (+ 36) is then bank (8 dp_lo + kp_lo + c) % 32: all 32
        // distinct. With consecutive dp per lane at one kp it was 8-way. The
        // global read becomes 8 rows x 16 aligned bytes per instruction
        // instead of one 128-byte row.
        #[unroll]
        for r in 0..v_rounds {
            let i = r * units + u;
            let dp = (i / 32usize) % comptime!(head_pairs / 4) * 4usize + i % 4usize;
            let kp = i / comptime!(8 * head_pairs) * 8usize + (i / 4usize) % 8usize;
            let key = t + (kp * 2usize) as u32;
            let mut v0 = Vector::<bf16, N>::cast_from(0.0f32);
            let mut v1 = Vector::<bf16, N>::cast_from(0.0f32);
            if key < s_hi {
                v0 = v[key as usize * kv_row_pairs + kvh * head_pairs + dp];
            }
            if key + 1 < s_hi {
                v1 = v[(key + 1) as usize * kv_row_pairs + kvh * head_pairs + dp];
            }
            let mut w0 = Vector::<bf16, N>::empty();
            let mut w1 = Vector::<bf16, N>::empty();
            w0[0usize] = v0[0usize];
            w0[1usize] = v1[0usize];
            w1[0usize] = v0[1usize];
            w1[1usize] = v1[1usize];
            svt[(dp * 2usize) * V_ROW_WORDS + kp] = w0;
            svt[(dp * 2usize + 1usize) * V_ROW_WORDS + kp] = w1;
        }
        sync_cube();

        // --- S = Q K^T: B(k = dim, n = key) straight from the K tile ---------
        #[unroll]
        for nt in 0..8usize {
            let mut c = Array::<Vector<f32, N>>::new(vc_c);
            #[unroll]
            for i in 0..vc_c {
                c[i] = Vector::<f32, N>::cast_from(0.0f32);
            }
            #[unroll]
            for kk in 0..8usize {
                let mut a = Array::<Vector<bf16, N>>::new(vc_a);
                #[unroll]
                for i in 0..vc_a {
                    a[i] = qf[kk * vc_a + i];
                }
                let mut b = Array::<Vector<bf16, N>>::new(vc_b);
                #[unroll]
                for i in 0..vc_b {
                    let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
                    let key = nt * 8usize + col as usize;
                    let dim = kk * 16usize + row as usize;
                    b[i] = sk[key * K_ROW_WORDS + dim / 2usize];
                }
                let d = def.execute(&a, &b, &c);
                #[unroll]
                for i in 0..vc_c {
                    c[i] = d[i];
                }
            }
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
            for nt in 0..8usize {
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
            for nt in 0..8usize {
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
        #[unroll]
        for kk in 0..4usize {
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
                if comptime![hilo] {
                    pl[i] = Vector::<bf16, N>::cast_from(s0 - Vector::<f32, N>::cast_from(h0));
                    pl[i + vc_c] = Vector::<bf16, N>::cast_from(s1 - Vector::<f32, N>::cast_from(h1));
                }
            }
            #[unroll]
            for dt in 0..16usize {
                // B(k = key, n = dim) from V^T: two adjacent keys per register.
                let mut b = Array::<Vector<bf16, N>>::new(vc_b);
                #[unroll]
                for i in 0..vc_b {
                    let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
                    let key = kk * 16usize + row as usize;
                    let dim = dt * 8usize + col as usize;
                    b[i] = svt[dim * V_ROW_WORDS + key / 2usize];
                }
                let mut c = Array::<Vector<f32, N>>::new(vc_c);
                #[unroll]
                for i in 0..vc_c {
                    c[i] = of[dt * vc_c + i];
                }
                let d = def.execute(&ph, &b, &c);
                if comptime![hilo] {
                    let d2 = def.execute(&pl, &b, &d);
                    #[unroll]
                    for i in 0..vc_c {
                        of[dt * vc_c + i] = d2[i];
                    }
                } else {
                    #[unroll]
                    for i in 0..vc_c {
                        of[dt * vc_c + i] = d[i];
                    }
                }
            }
        }
        sync_cube();

        t += NK as u32;
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
pub(crate) fn split_plan(runs: &[KeyRun<'_>], nq: usize, warps: usize) -> Vec<usize> {
    let spans: Vec<usize> = runs.iter().map(|r| r.hi.saturating_sub(r.lo)).collect();
    flash::key_run_splits(&spans, query_tiles(nq, warps), KV_HEADS)
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
    let (nq, q0, warps, mode, split_of) = (plan.nq, plan.q0, plan.warps, plan.mode, plan.split_of.as_slice());
    // The only callers validate the complete plan, input runs and output
    // extents before allocation/launch. Consume that geometry without replanning.
    let vs = 32 / bf16::cube_type().size_bits();
    let q_elems = plan.q_elems;
    let q_tiles = plan.query_tiles;
    let mut slot0 = 0usize;
    for (run, splits) in runs.iter().zip(split_of.iter().copied()) {
        let kv_elems = run.rows * KV_WIDTH;
        let count = CubeCount::Static(q_tiles as u32, KV_HEADS as u32, splits as u32);
        let dim = CubeDim::new_1d((warps * PLANE as usize) as u32);
        unsafe {
            mma_attention_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                vs,
                ArrayArg::from_raw_parts(q.clone(), q_elems),
                ArrayArg::from_raw_parts(run.k.clone(), kv_elems),
                ArrayArg::from_raw_parts(run.v.clone(), kv_elems),
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
                warps,
                mode == PMode::HiLo,
            )
        };
        slot0 += splits;
    }
}

/// Tensor-core attention over one BF16 query block and any number of BF16 key
/// runs, Nemotron geometry only.
///
/// `q` is `[nq, 32 * 128]` BF16, contiguous from offset 0; each run is
/// `[rows, 2 * 128]` BF16 token-major with `base` its ABSOLUTE position of row
/// 0, exactly as [`flash::flash_attention_launch`] takes them. Returns the same
/// `[nq, 32 * 128]` F32 handle layout and dtype that launcher returns.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn mma_attention_launch<R: Runtime>(
    client: &ComputeClient<R>,
    q: &Handle,
    runs: &[KeyRun<'_>],
    nq: usize,
    q0: usize,
    scaling: f32,
    warps: usize,
    mode: PMode,
) -> Handle {
    let spans: Vec<_> = runs.iter().map(|r| r.hi.checked_sub(r.lo).expect("invalid run")).collect();
    let plan = Plan::new(&spans, nq, q0, warps, mode).expect("MMA plan");
    launch_planned(client, q, runs, scaling, &plan).expect("MMA input")
}

pub(crate) fn launch_planned<R: Runtime>(client: &ComputeClient<R>, q: &Handle,
    runs: &[KeyRun<'_>], scaling: f32, plan: &Plan) -> Result<Handle, String>
{
    plan.validate_inputs(q, runs, scaling)?; // ALL runs before ANY allocation.
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
    runs: &[KeyRun<'_>], nq: usize, q0: usize, scaling: f32, warps: usize, mode: PMode,
    split_of: &[usize], po: &Handle, po_len: usize, pml: &Handle, pml_len: usize)
{
    let spans: Vec<_> = runs.iter().map(|r| r.hi.checked_sub(r.lo).expect("invalid run")).collect();
    let plan = Plan::new(&spans, nq, q0, warps, mode).expect("MMA plan");
    plan.validate_inputs(q, runs, scaling).expect("MMA inputs");
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
