//! Mini's attention on tensor cores, for calls of many queries: BF16 Q, K and
//! V, f32 accumulators, the relative bias, the causal and window masks and an
//! online softmax, one launch per key segment.
//!
//! [`crate::models::inkling::flash`]'s kernel multiplies in f32 on the plain
//! lanes; it is bound by the bytes a cube reads per row it serves. Here a cube
//! serves [`rows_per_cube`] rows from one staged key tile, on
//! `mma.sync.m16n8k16`. The staging and the fragment maps are
//! Nemotron-H's phase-A BF16 kernel
//! ([`crate::models::nemotron_h`], mary d36d7ef); what differs is Mini's shape
//! (three query heads per KV head, so a fragment's sixteen rows are sixteen
//! consecutive QUERIES of one head, not sixteen heads of one query), the bias,
//! the window, and that a block of queries carries its running softmax from
//! one key segment to the next instead of writing a partial per segment.
//!
//! # Shape
//!
//! The grid is `(query tiles, KV heads)`. A cube is `groups * QB` planes; plane
//! `w` owns head `kvh * groups + w % groups` and the sixteen queries `qt * bq +
//! (w / groups) * 16 ..`, `bq = 16 * QB`. Masks and bias are per (query, key)
//! cell.
//!
//! # State
//!
//! `po` is `[nq, heads, head_dim]` (unnormalised `sum exp(s - m) v`) and `pml`
//! `[nq, heads, 2]` (`m`, `l = sum exp(s - m)`), `flash_kernel`'s partial
//! layout with one slot. The first launch of a block starts the recurrence;
//! each later one reads it back, so the working memory does not grow with the
//! number of key segments. [`flash::flash_combine_sinks_kernel`] normalises,
//! with the sink in the denominator.
//!
//! # Precision
//!
//! Q, K and V are BF16, which is what upstream's clips and norms hand to its
//! attention. The scores and the softmax are f32. P is rounded to BF16 for
//! `P V`, as upstream casts its softmax to the query's dtype; the accumulator
//! is f32, and the denominator is the unrounded softmax's.

use cubecl::{ir::MatrixIdent, prelude::*, server::Handle};
use half::bf16;

use crate::models::inkling::flash::{self, KeyRun, NEG_INF, PLANE};

/// Keys per staged tile.
const NK: usize = 64;
/// Sixteen-query fragments per head in a cube. A cube reads every key its
/// queries can see, so more queries per cube read each key less often: two to
/// four took a warm 4,096-token call's attention from 1.16 to 0.88 s at 32,768
/// tokens. Eight (24 planes a cube) was 8 to 12 times slower than four.
const QB: usize = 4;
/// Words (BF16 pairs) per staged V^T row: the keys' 32 plus 4 of padding, so
/// a fragment read's 32 lanes land on distinct banks. K rows are padded the
/// same way ([`k_row_words`]).
const V_ROW_WORDS: usize = NK / 2 + 4;

const fn k_row_words(head_dim: usize) -> usize {
    head_dim / 2 + 4
}

/// `(head, query)` rows one cube serves.
pub fn rows_per_cube(heads: usize, kv_heads: usize) -> usize {
    heads / kv_heads * QB * 16
}

fn shared_bytes(head_dim: usize) -> usize {
    (NK * k_row_words(head_dim) + head_dim * V_ROW_WORDS) * 4
}

/// Whether the kernel covers this shape: whole `k16` steps of a head, no more
/// planes than a cube may have, and staging within the default 48 KiB.
pub fn applies(heads: usize, kv_heads: usize, head_dim: usize) -> bool {
    kv_heads > 0
        && heads > 0
        && heads % kv_heads == 0
        && heads / kv_heads <= 4
        && head_dim > 0
        && head_dim % 16 == 0
        && shared_bytes(head_dim) <= 48 * 1024
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn attend_mma_kernel<N: Size>(
    q: &Array<Vector<bf16, N>>,
    k: &Array<Vector<bf16, N>>,
    v: &Array<Vector<bf16, N>>,
    rel: &Array<f32>,
    po: &mut Array<Vector<f32, N>>,
    pml: &mut Array<f32>,
    scaling: f32,
    nq: u32,
    q0: u32,
    k0: u32,
    khi: u32,
    eff: u32,
    window: u32,
    carry: u32,
    #[comptime] heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] head_dim: usize,
) {
    let def = cmma::MmaDefinition::<bf16, bf16, f32>::new(16usize, 8usize, 16usize);
    let groups = comptime!(heads / kv_heads);
    let units = comptime!(heads / kv_heads * QB * PLANE as usize);
    let qt = CUBE_POS_X;
    let kvh = CUBE_POS_Y as usize;
    let u = UNIT_POS_X as usize;
    let lane = UNIT_POS_PLANE;
    let warp = (UNIT_POS_X / PLANE) as usize;
    let head = kvh * groups + warp % groups;
    let sub = (warp / groups) as u32;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    // Q, K and V indices are in BF16 pairs.
    let q_row_pairs = comptime!(heads * head_dim / 2);
    let kv_row_pairs = comptime!(kv_heads * head_dim / 2);
    let head_pairs = comptime!(head_dim / 2);
    let k_words = comptime!(head_dim / 2 + 4);
    let k_steps = comptime!(head_dim / 16);
    let dim_tiles = comptime!(head_dim / 8);
    let key_tiles = comptime!(NK / 8);
    let pv_steps = comptime!(NK / 16);

    let bq = comptime!((QB * 16) as u32);
    let q_first = qt * bq;
    // This plane's first query, block-local.
    let qp = q_first + sub * 16u32;

    // --- the keys any query of this cube can see -----------------------------
    let q_lo = q0 + q_first;
    let mut q_hi = q0 + nq - 1;
    if q_first + bq <= nq {
        q_hi = q_lo + bq - 1;
    }
    let mut lo = 0u32;
    let mut hi = khi;
    if q_hi + 1 <= k0 {
        hi = lo;
    } else if q_hi + 1 - k0 < hi {
        hi = q_hi + 1 - k0;
    }
    if window > 0 && q_lo + 1 > window + k0 {
        lo = q_lo + 1 - window - k0;
    }
    if hi < lo {
        hi = lo;
    }
    let work = carry == 0u32 || lo < hi;

    // --- Q as k16 A fragments, held for the whole launch ----------------------
    let mut qf = Array::<Vector<bf16, N>>::new(comptime!(k_steps * vc_a));
    #[unroll]
    for kk in 0..k_steps {
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a) as u32, MatrixIdent::A);
            let mut val = Vector::<bf16, N>::cast_from(0.0f32);
            let qrow = qp + row;
            if qrow < nq && lo < hi {
                let at = qrow as usize * q_row_pairs + head * head_pairs + (kk * 16usize + col as usize) / 2usize;
                val = q[at];
            }
            qf[kk * vc_a + i] = val;
        }
    }

    // --- the running softmax of the rows this lane holds ----------------------
    let mut m = Array::<f32>::new(vc_c);
    let mut l = Array::<f32>::new(vc_c);
    let mut of = Array::<Vector<f32, N>>::new(comptime!(dim_tiles * vc_c));
    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let qrow = qp + row;
        let cell = qrow * comptime!(heads as u32) + head as u32;
        let mut mi = f32::new(NEG_INF);
        let mut li = f32::new(0.0f32);
        let resume = carry != 0u32 && qrow < nq && lo < hi;
        if resume {
            mi = pml[(cell * 2u32) as usize];
            li = pml[(cell * 2u32 + 1u32) as usize];
        }
        m[i] = mi;
        l[i] = li;
        #[unroll]
        for dt in 0..dim_tiles {
            let mut ov = Vector::<f32, N>::cast_from(0.0f32);
            if resume {
                let base = cell * comptime!(head_dim as u32);
                ov = po[((base + (dt * 8usize) as u32 + col) / 2u32) as usize];
            }
            of[dt * vc_c + i] = ov;
        }
    }
    let mut sacc = Array::<Vector<f32, N>>::new(comptime!(key_tiles * vc_c));

    let mut sk = SharedMemory::<Vector<bf16, N>>::new(comptime!(NK * (head_dim / 2 + 4)));
    let mut svt = SharedMemory::<Vector<bf16, N>>::new(comptime!(head_dim * V_ROW_WORDS));
    let k_total = comptime!(NK * head_dim / 2);
    let k_rounds = comptime!((NK * head_dim / 2).div_ceil(heads / kv_heads * QB * PLANE as usize));
    let v_total = comptime!(NK / 2 * head_dim / 2);
    let v_rounds = comptime!((NK / 2 * head_dim / 2).div_ceil(heads / kv_heads * QB * PLANE as usize));

    let mut t = lo;
    while t < hi {
        // --- stage K [key][dim] and V^T [dim][key]; keys at or past `hi` are
        // zero and never read
        #[unroll]
        for r in 0..k_rounds {
            let i = r * units + u;
            if i < k_total {
                let j = i / head_pairs;
                let dp = i % head_pairs;
                let key = t + j as u32;
                let mut val = Vector::<bf16, N>::cast_from(0.0f32);
                if key < hi {
                    val = k[key as usize * kv_row_pairs + kvh * head_pairs + dp];
                }
                sk[j * k_words + dp] = val;
            }
        }
        // One store instruction covers 4 consecutive dim pairs by 8
        // consecutive key pairs, conflict free on the store and on the read.
        #[unroll]
        for r in 0..v_rounds {
            let i = r * units + u;
            if i < v_total {
                let dp = (i / 32usize) % comptime!(head_dim / 8) * 4usize + i % 4usize;
                let kp = i / comptime!(4 * head_dim) * 8usize + (i / 4usize) % 8usize;
                let key = t + (kp * 2usize) as u32;
                let mut v0 = Vector::<bf16, N>::cast_from(0.0f32);
                let mut v1 = Vector::<bf16, N>::cast_from(0.0f32);
                if key < hi {
                    v0 = v[key as usize * kv_row_pairs + kvh * head_pairs + dp];
                }
                if key + 1u32 < hi {
                    v1 = v[(key + 1u32) as usize * kv_row_pairs + kvh * head_pairs + dp];
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
        }
        sync_cube();

        // --- S = Q K^T
        #[unroll]
        for nt in 0..key_tiles {
            let mut c = Array::<Vector<f32, N>>::new(vc_c);
            #[unroll]
            for i in 0..vc_c {
                c[i] = Vector::<f32, N>::cast_from(0.0f32);
            }
            #[unroll]
            for kk in 0..k_steps {
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
                    b[i] = sk[key * k_words + dim / 2usize];
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

        // --- bias, masks and the online update, per row this lane holds
        // Whether the nearest query of the cube reaches the bias table in this
        // tile at all.
        let near = q_lo < k0 + t + comptime!(NK as u32) + eff;
        #[unroll]
        for i in 0..vc_c {
            let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
            let qrow = qp + row;
            let q_abs = q0 + qrow;
            let table = (qrow * comptime!(heads as u32) + head as u32) * eff;
            let mut tile_max = f32::new(NEG_INF);
            #[unroll]
            for nt in 0..key_tiles {
                let mut sv = sacc[nt * vc_c + i];
                #[unroll]
                for e in 0..vs_c {
                    let key = t + (nt * 8usize + e) as u32 + col;
                    let key_abs = k0 + key;
                    let mut vis = key < hi && qrow < nq && key_abs <= q_abs;
                    if window > 0 && vis && q_abs - key_abs >= window {
                        vis = false;
                    }
                    let mut s = sv[e] * scaling;
                    if vis {
                        if near && q_abs - key_abs < eff {
                            s += rel[(table + (q_abs - key_abs)) as usize];
                        }
                    } else {
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
            for nt in 0..key_tiles {
                let mut sv = sacc[nt * vc_c + i];
                #[unroll]
                for e in 0..vs_c {
                    let mut pe = f32::new(0.0f32);
                    if sv[e] > f32::new(NEG_INF) {
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
            for dt in 0..dim_tiles {
                of[dt * vc_c + i] = of[dt * vc_c + i] * Vector::<f32, N>::cast_from(alpha);
            }
        }

        // --- O += P V: P's A fragment is two adjacent S accumulator tiles
        #[unroll]
        for kk in 0..pv_steps {
            let mut p = Array::<Vector<bf16, N>>::new(vc_a);
            #[unroll]
            for i in 0..vc_c {
                p[i] = Vector::<bf16, N>::cast_from(sacc[(kk * 2usize) * vc_c + i]);
                p[i + vc_c] = Vector::<bf16, N>::cast_from(sacc[(kk * 2usize + 1usize) * vc_c + i]);
            }
            #[unroll]
            for dt in 0..dim_tiles {
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
                let d = def.execute(&p, &b, &c);
                #[unroll]
                for i in 0..vc_c {
                    of[dt * vc_c + i] = d[i];
                }
            }
        }
        sync_cube();

        t += comptime!(NK as u32);
    }

    // --- the state, for the next segment or the merge -------------------------
    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let qrow = qp + row;
        if work && qrow < nq {
            let cell = qrow * comptime!(heads as u32) + head as u32;
            let base = cell * comptime!(head_dim as u32);
            #[unroll]
            for dt in 0..dim_tiles {
                po[((base + (dt * 8usize) as u32 + col) / 2u32) as usize] = of[dt * vc_c + i];
            }
            if col == 0u32 {
                pml[(cell * 2u32) as usize] = m[i];
                pml[(cell * 2u32 + 1u32) as usize] = l[i];
            }
        }
    }
}

/// Attention of `nq` queries at absolute positions `q0 ..` over `runs`.
///
/// `q` is `[nq, heads * head_dim]` BF16, `rel` `[nq, heads, eff]` f32, `sinks`
/// `[nq, heads]` f32, all contiguous; a run's `k` and `v` are `[rows, kv_heads
/// * head_dim]` BF16 of which the first `hi` rows are keys, `base` the
/// absolute position of row 0 (`lo` must be 0). Returns `[nq, heads *
/// head_dim]` f32.
#[allow(clippy::too_many_arguments)]
pub fn attend_mma_launch<R: Runtime>(
    client: &ComputeClient<R>,
    q: &Handle,
    runs: &[KeyRun<'_>],
    rel: &Handle,
    sinks: &Handle,
    nq: usize,
    q0: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eff: usize,
    window: Option<usize>,
    scaling: f32,
) -> Handle {
    assert!(nq > 0 && !runs.is_empty(), "attention of no queries or over no keys");
    assert!(applies(heads, kv_heads, head_dim), "no tensor-core attention at {heads}/{kv_heads} heads of {head_dim}");
    let vs = 32 / bf16::cube_type().size_bits();
    let q_elems = nq * heads * head_dim;
    let rel_elems = nq * heads * eff;
    assert!(q_elems <= u32::MAX as usize && rel_elems <= u32::MAX as usize, "{nq} queries are past the 32-bit index");
    let po = client.empty(q_elems * 4);
    let pml = client.empty(nq * heads * 2 * 4);
    let count = CubeCount::Static(nq.div_ceil(QB * 16) as u32, kv_heads as u32, 1);
    let dim = CubeDim::new_1d((heads / kv_heads * QB) as u32 * PLANE);
    for (i, run) in runs.iter().enumerate() {
        let kv_elems = run.rows * kv_heads * head_dim;
        assert!(run.lo == 0 && run.hi <= run.rows, "key run {i}: rows {}..{} of {}", run.lo, run.hi, run.rows);
        assert!(kv_elems <= u32::MAX as usize && run.base + run.rows <= u32::MAX as usize, "key run {i} is past the 32-bit index");
        unsafe {
            attend_mma_kernel::launch_unchecked::<R>(
                client,
                count.clone(),
                dim,
                vs,
                ArrayArg::from_raw_parts(q.clone(), q_elems),
                ArrayArg::from_raw_parts(run.k.clone(), kv_elems),
                ArrayArg::from_raw_parts(run.v.clone(), kv_elems),
                ArrayArg::from_raw_parts(rel.clone(), rel_elems),
                ArrayArg::from_raw_parts(po.clone(), q_elems),
                ArrayArg::from_raw_parts(pml.clone(), nq * heads * 2),
                scaling,
                nq as u32,
                q0 as u32,
                run.base as u32,
                run.hi as u32,
                eff as u32,
                window.unwrap_or(0) as u32,
                (i > 0) as u32,
                heads,
                kv_heads,
                head_dim,
            )
        };
    }
    let out = client.empty(q_elems * 4);
    unsafe {
        flash::flash_combine_sinks_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(nq as u32, heads as u32, 1),
            CubeDim::new_1d(head_dim as u32),
            ArrayArg::from_raw_parts(po, q_elems),
            ArrayArg::from_raw_parts(pml, nq * heads * 2),
            ArrayArg::from_raw_parts(sinks.clone(), nq * heads),
            ArrayArg::from_raw_parts(out.clone(), q_elems),
            1u32,
            nq as u32,
            heads as u32,
            head_dim as u32,
        )
    };
    out
}
