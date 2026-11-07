//! Structural and finite synthetic fixtures for the tensor-core attention
//! kernel. Test-only: no model, no pile, no runtime policy, admission or cache
//! ABI.
//!
//! Host tests replay the kernel's maps: shared-memory layouts conflict free,
//! the copy and decode maps covers, the k64 register maps `cubecl-cpp`'s, the
//! in-kernel Q quantizer's lane split the block quantization it claims, and
//! the V decode exact. GPU fixtures assert hard invariants on synthetic NVFP4
//! K/V runs written by the cache's own quantizer: every output finite, bitwise
//! determinism across launches, causal masking (a key changes exactly the
//! queries that may see it, bitwise, and a key past every query changes
//! nothing), padded planes and the dead rows past a run's live end never read
//! or written (dead rows carry the E4M3 NaN scale, a tripwire), and exact
//! masked/boundary values. Model quality is judged by capability, not by
//! numerical agreement with another implementation.
use super::*;
use crate::models::nemotron_h::{attention::HIDDEN, fp8::CudaTensor};
use burn::tensor::DType;
use cubecl::{
    client::ComputeClient,
    cuda::{CudaDevice, CudaRuntime},
};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const SEED: u64 = 0x5eeda77e;
const TAIL: usize = 16;
const LONG_KEYS: usize = 524_288;
const LONG_QUERIES: usize = 256;
const MAX_SECONDS: u64 = 250;
const MAX_RECORDS: usize = 160;
/// The E4M3 NaN pattern: the dead rows' scale bytes. The quantizer never
/// writes it, so any read of a dead row turns its products NaN.
const DEAD_SCALE: u8 = 0x7f;

#[test]
fn mma_host_checked_plan_sizes_and_rejections() {
    let p = Plan::new(&[524288], 256, 524032).unwrap();
    assert_eq!(p.split_of, [8]);
    assert_eq!(p.po_elems * 4, 8 * 256 * 4096 * 4);
    assert_eq!(p.pml_elems * 4, 8 * 256 * 32 * 2 * 4);
    assert_eq!(p.working_bytes, (p.po_elems + p.pml_elems) * 4 + 256 * 4096 * 6);
    assert!(p.validate_splits(&[0]).is_err());
    assert!(p.validate_splits(&[]).is_err());
    assert!(p.validate_splits(&[9]).is_err());
    for (spans, nq, q0) in [
        (vec![], 1, 0), (vec![1], 0, 0), (vec![1], 257, 0),
        (vec![1], 1, u32::MAX as usize - 1),
        (vec![usize::MAX, 1], 1, 0),
    ] {
        assert!(Plan::new(&spans, nq, q0).is_err());
    }
    assert!(extent(&[usize::MAX, 2], 4).is_err());
    assert!(extent(&[u32::MAX as usize], 4).is_err());
    assert_eq!(Plan::new(&[0], 3, 0).unwrap().split_of, [1]);
    assert_eq!(shared_bytes(), 17_936);
    assert!(shared_bytes() <= SHARED_LIMIT_BYTES);
}

#[test]
fn mma_host_prefix_rounding_earlier_endpoint_exceeds_final_width_envelope() {
    // A 256-row tile ending at 8448 over an 8192-row prefix has more partial
    // rows than any final-width tile ending at capacity 65536.
    let earlier = Plan::new(&[8192, 8448 - 8192], 256, 8192).unwrap();
    assert_eq!(earlier.split_of, [8, 1]);
    assert_eq!(earlier.nq * earlier.slots, 2304);
    let final_max = (2..=256).map(|t| {
        let p = Plan::new(&[8192, 65536 - 8192], t, 65536 - t).unwrap();
        p.nq * p.slots
    }).max().unwrap();
    assert_eq!(final_max, 2244);
    assert!(earlier.nq * earlier.slots > final_max);
}

// --------------------------------------------------------- shared layout --

/// Blocks per SM that `shared` bytes allow on this GB10 (102,400 per SM, 1,024
/// reserved per block, the sum allocated in 128-byte units).
fn blocks_by_shared(shared: usize) -> usize {
    102_400 / (shared + 1_024).next_multiple_of(128)
}

/// Most distinct 32-bit words any one bank serves in one shared access (lanes
/// on the same word are a broadcast, not a conflict).
fn bank_degree(words: &[usize]) -> usize {
    let mut per_bank = vec![Vec::new(); 32];
    for &w in words {
        if !per_bank[w % 32].contains(&w) {
            per_bank[w % 32].push(w);
        }
    }
    per_bank.iter().map(Vec::len).max().unwrap()
}

/// [`fp4_unit`] on the host: unit `c` of staged NVFP4 row `row`.
fn fp4_unit_host(row: usize, c: usize) -> usize {
    row * FP4_ROW_UNITS + (c + row / 2) % FP4_ROW_UNITS
}

/// Distinct 16-byte units of a phase, and the most of them on one 16-byte
/// group (4 banks): distinct units each on its own group is conflict free.
fn groups(units: &[usize]) -> (usize, usize) {
    let mut distinct = units.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let mut per_group = [0usize; 8];
    for &u in &distinct {
        per_group[u % 8] += 1;
    }
    (distinct.len(), *per_group.iter().max().unwrap())
}

/// Host only, with the index arithmetic of `copy_tile_fp4_async`,
/// [`fp4_unit`], `fp4_decode_tile`, `fp4_qk_k_operands` and the PV
/// `ldmatrix.x4.trans` lane map:
/// * the shared bytes and the blocks per SM they allow (5);
/// * the `cp.async` code map a cover of every (row, unit), each warp
///   instruction 8 whole head rows, each 8-lane phase on 8 distinct groups;
///   the scale copy a cover, each half warp on distinct banks;
/// * the V decode: every data word of the 68-word BF16 rows written once and
///   no pad word, every store instruction on 32 banks, every unit read
///   conflict free per phase, every stored pair the element its address names
///   under the block its scale index names;
/// * the raw K reads: scale row on distinct banks, unit reads conflict free per
///   phase, every kept word EXACTLY `cubecl-cpp`'s k64 B element and its scale
///   byte the block of its dims;
/// * the V `ldmatrix.x4.trans` matrices conflict free on the 272-byte rows
///   (8-way unpadded) and delivering keys `16 kk + 2 t + 8 r`, `+ 1` at dim
///   `8 dt + g`, the lower key low, which is the k16 B fragment.
#[test]
fn mma_attention_shared_layout_is_conflict_free() {
    let lanes = PLANE as usize;
    let units = WARPS * lanes;
    let ru = FP4_ROW_UNITS;
    let w = ASYNC_ROW_WORDS;
    assert_eq!((lanes, units, NK, STAGES), (32, 128, 32, 2));
    assert_eq!((ru, FP4_UNIT_WORDS, FP4_SCALE_WORDS, HEAD_DIM, w), (4, 4, 2, 128, 68));
    assert_eq!(FP4_BYTES_PER_KEY_HEAD, 144, "bytes per key per KV head");
    assert_eq!(shared_bytes(), ASYNC_BARRIER_BYTES + STAGES * NK * 144 + NK * w * 4);
    assert_eq!(blocks_by_shared(shared_bytes()), 5);
    // Every stage of whole 128-byte bank rows, so the second starts on bank 0.
    assert_eq!((NK * ru * 16 % 128, NK * FP4_SCALE_WORDS * 4 % 128, NK * w * 4 % 128), (0, 0, 0));
    assert_eq!(w * 4 / 16 % 2, 1, "a BF16 V row is not an odd number of 16-byte units");

    // The cp.async code map: chunk i = r * units + u, row i / 4, unit i % 4.
    let rounds = NK * ru / units;
    assert_eq!(rounds * units, NK * ru);
    let mut logical = vec![usize::MAX; NK * ru];
    let mut seen = vec![0usize; NK * ru];
    for r in 0..rounds {
        for warp in 0..WARPS {
            let ids: Vec<_> = (0..lanes).map(|l| r * units + warp * lanes + l).collect();
            assert!(ids[0] % ru == 0 && ids.windows(2).all(|p| p[1] == p[0] + 1), "not whole rows");
            for &i in &ids {
                let p = fp4_unit_host(i / ru, i % ru);
                seen[p] += 1;
                logical[p] = i;
            }
            for phase in ids.chunks(8) {
                let at: Vec<_> = phase.iter().map(|&i| fp4_unit_host(i / ru, i % ru)).collect();
                assert_eq!(groups(&at), (8, 1), "cp.async phase: {at:?}");
            }
        }
    }
    assert!(seen.iter().all(|&n| n == 1), "cp.async code map is not a cover");
    // The scale copy: unit u < NK copies row u, two words.
    assert!(NK <= units);
    let mut rows = vec![0usize; NK];
    for warp in 0..WARPS {
        let words: Vec<_> = (0..lanes)
            .map(|l| warp * lanes + l)
            .filter(|&u| u < NK)
            .inspect(|&u| rows[u] += 1)
            .flat_map(|u| [u * FP4_SCALE_WORDS, u * FP4_SCALE_WORDS + 1])
            .collect();
        for half in words.chunks(32) {
            assert_eq!(bank_degree(half), 1, "scale copy warp {warp}");
        }
    }
    assert!(rows.iter().all(|&n| n == 1), "scale copy is not a cover");

    // The V decode: warp `warp` takes block `r * WARPS + warp` of 8 rows.
    assert_eq!((NK / 8) % WARPS, 0, "whole 8-row decode blocks per warp");
    let mut blocks = vec![0usize; NK / 8];
    let mut written = vec![0usize; NK * w];
    for r in 0..NK / 8 / WARPS {
        for warp in 0..WARPS {
            let b = r * WARPS + warp;
            blocks[b] += 1;
            let row = |l: usize| b * 8 + l / 4;
            let scale_words: Vec<_> = (0..lanes)
                .flat_map(|l| [row(l) * FP4_SCALE_WORDS, row(l) * FP4_SCALE_WORDS + 1])
                .collect();
            assert_eq!(bank_degree(&scale_words), 1, "V decode scale read block {b}");
            for c in 0..ru {
                let at: Vec<_> = (0..lanes).map(|l| fp4_unit_host(row(l), c)).collect();
                for phase in at.chunks(8) {
                    let (n, most) = groups(phase);
                    assert_eq!(most, 1, "V decode unit read block {b} unit {c}: {n} units {phase:?}");
                }
                for l in 0..lanes {
                    let i = logical[at[l]];
                    assert_eq!((i / ru, i % ru), (row(l), c), "V decode reads the wrong staged unit");
                }
                for i in 0..4 {
                    let words: Vec<_> = (0..lanes).map(|l| row(l) * w + 16 * c + 4 * i + l % 4).collect();
                    assert_eq!(bank_degree(&words), 1, "V decode store block {b} unit {c} word {i}");
                    for (l, &word) in words.iter().enumerate() {
                        written[word] += 1;
                        let (key, p) = (word / w, word % w);
                        // Code word 4 c + i, byte t: dims 8 (4 c + i) + 2 t, + 1.
                        let dim = 8 * (4 * c + i) + 2 * (l % 4);
                        assert_eq!((key, 2 * p, 2 * p + 1), (row(l), dim, dim + 1), "V decode pair");
                        assert_eq!((2 * c + i / 2, (dim + 1) / 16), (dim / 16, dim / 16), "V decode block");
                    }
                }
            }
        }
    }
    assert!(blocks.iter().all(|&n| n == 1), "V decode blocks are not a cover");
    for (word, &n) in written.iter().enumerate() {
        assert_eq!(n, usize::from(word % w < HEAD_DIM / 2), "V word {word} written {n} times");
    }
    // fp4_scales_cvt's u16 halves put block b's byte at out[b], given that
    // the pair conversion's element 0 is the low byte (the probe's claim).
    let row: Vec<u8> = (0..FP4_SCALE_WORDS * 4).map(|b| b as u8).collect();
    let mut cvt = vec![u8::MAX; HEAD_DIM / 16];
    for (wi, b) in row.chunks(4).enumerate() {
        let word = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        for h in 0..2 {
            let pair = (word >> (h * 16)) as u16;
            cvt[wi * 4 + h * 2] = (pair & 0xff) as u8;
            cvt[wi * 4 + h * 2 + 1] = (pair >> 8) as u8;
        }
    }
    assert_eq!(cvt, row, "scale decode index");

    // The raw K reads.
    for nt in 0..NK / 8 {
        let scale_words: Vec<_> = (0..lanes)
            .flat_map(|l| {
                let row = nt * 8 + l / 4;
                [row * FP4_SCALE_WORDS, row * FP4_SCALE_WORDS + 1]
            })
            .collect();
        assert_eq!(bank_degree(&scale_words), 1, "K scale read nt{nt}");
        for c in 0..ru {
            let at: Vec<_> = (0..lanes).map(|l| fp4_unit_host(nt * 8 + l / 4, c)).collect();
            for phase in at.chunks(8) {
                assert_eq!(groups(phase).1, 1, "K unit read nt{nt} unit {c}: {phase:?}");
            }
            for l in 0..lanes {
                let (g, t) = (l / 4, l % 4);
                let i = logical[at[l]];
                assert_eq!((i / ru, i % ru), (nt * 8 + g, c), "K reads the wrong staged unit");
                let (s, i) = (c / 2, c % 2);
                let word = FP4_UNIT_WORDS * c + t;
                for e in 0..8 {
                    let dim = 8 * word + e;
                    assert_eq!(dim, 64 * s + 32 * i + 8 * t + e, "K B word of lane {l} unit {c}");
                    assert_eq!((64 * s + cubecl_row_index(l, i * 8 + e, 'b'), cubecl_col_index(l, i * 8 + e, 'b')),
                        (dim, g), "K B register {i} of step {s}, lane {l}, nibble {e}: not cubecl's element");
                    assert_eq!(4 * s + dim % 64 / 16, dim / 16, "K B scale byte of dim {dim}");
                }
            }
        }
    }

    // The V ldmatrix.x4.trans reads of the decoded tile: lane L gives row
    // 16 kk + L % 16, dims (2 dt2 + L / 16) * 8.
    let matrix = |rows: &[usize]| -> usize {
        assert!(rows.iter().all(|&r| r % 4 == 0), "unaligned ldmatrix row: {rows:?}");
        bank_degree(&rows.iter().flat_map(|&r| r..r + 4).collect::<Vec<_>>())
    };
    let v_row = |kk: usize, dt2: usize, l: usize, w: usize| kk * 16 * w + dt2 * 8 + (l % 16) * w + (l / 16) * 4;
    for kk in 0..NK / 16 {
        for dt2 in 0..8 {
            for i in 0..4 {
                let rows: Vec<_> = (0..8).map(|j| v_row(kk, dt2, 8 * i + j, w)).collect();
                assert_eq!(matrix(&rows), 1, "V ldmatrix kk{kk} dt2 {dt2} matrix {i}");
                let unpadded: Vec<_> = (0..8).map(|j| v_row(kk, dt2, 8 * i + j, HEAD_DIM / 2)).collect();
                assert_eq!(matrix(&unpadded), 8, "unpadded V ldmatrix");
            }
            let at = |e: usize| (e / (2 * w), e % (2 * w));
            for l in 0..lanes {
                let (g, t) = (l / 4, l % 4);
                for i in 0..4 {
                    let lo = at(2 * v_row(kk, dt2, 8 * i + 2 * t, w) + g);
                    let hi = at(2 * v_row(kk, dt2, 8 * i + 2 * t + 1, w) + g);
                    let (dt, r) = (2 * dt2 + i / 2, i % 2);
                    let (key, dim) = (kk * 16 + 2 * t + 8 * r, dt * 8 + g);
                    assert_eq!((lo, hi), ((key, dim), (key + 1, dim)),
                        "V ldmatrix.trans register {i} of lane {l} kk{kk} dt2 {dt2}");
                }
            }
        }
    }
}

/// `cubecl-cpp`'s `cuda/processors.rs` `row_index` for the e2m1 `m16n8k64`
/// (eight codes a register), for `A` (`'a'`), `B` (`'b'`) and the accumulator
/// (any other): the source the kernel's closed forms must agree with.
fn cubecl_row_index(lane: usize, n: usize, ident: char) -> usize {
    const EPR: usize = 8;
    match ident {
        'a' => lane / 4 + ((n / EPR) & 1) * 8,
        'b' => (lane % 4) * EPR + (n % EPR) + EPR * 4 * (n / EPR),
        _ => lane / 4 + ((n << 2) & 8),
    }
}

/// `col_index`, same source.
fn cubecl_col_index(lane: usize, n: usize, ident: char) -> usize {
    const EPR: usize = 8;
    match ident {
        'a' => (lane % 4) * EPR + (n % EPR) + 4 * EPR * ((n / (2 * EPR)) & 1),
        'b' => lane / 4,
        _ => (lane % 4) * 2 + (n % 2),
    }
}

/// `cubecl-core`'s `MmaDefinition::scales_index`: the row (A) or column (B)
/// whose scale word a lane holds.
fn cubecl_scales_index(lane: usize, ident: char) -> usize {
    match ident {
        'a' => lane / 4 + (lane % 4 % 2) * 8,
        _ => lane / 4,
    }
}

/// Host only: the k64 maps against `cubecl-cpp`'s, element for element. For
/// every lane, step, A register and nibble: the quantizer's chunk that lands
/// in `qa[4 s + i]` covers cubecl's A element; the A scale word's head is
/// `scales_index` A and its byte `e` the block of k-block `e`; the amax
/// partner `t ^ 1` holds the other half of the same block, the other parity
/// sits in `t ^ 2` of the same head; and the accumulator map is the one the
/// masking reads: rows `g + 8 i`, keys `2 t + e`.
#[test]
fn fp4_qk_register_maps_are_cubecl_maps() {
    for lane in 0..32usize {
        let (g, t) = (lane / 4, lane % 4);
        let mut covered = vec![false; 8];
        for hh in 0..2 {
            for j in 0..4 {
                let idx = (j / 2) * 4 + (j % 2) * 2 + hh;
                assert!(!covered[idx], "two chunks in A register slot {idx}");
                covered[idx] = true;
                let (s, i) = (idx / 4, idx % 4);
                assert_eq!((hh, j), (i & 1, 2 * s + (i >> 1)), "chunk ({hh}, {j}) slot of lane {lane}");
                for k in 0..4 {
                    for half in 0..2 {
                        let (e, dim) = (2 * k + half, 2 * (16 * j + 4 * t + k) + half);
                        let n = i * 8 + e;
                        assert_eq!((cubecl_row_index(lane, n, 'a'), 64 * s + cubecl_col_index(lane, n, 'a')),
                            (g + 8 * hh, dim), "A register {i} step {s} lane {lane} nibble {e}: not cubecl's");
                        assert_eq!(dim / 16, 2 * j + t / 2, "A block of lane {lane} chunk {j}");
                    }
                }
                let mut halves = [32 * j + 8 * t, 32 * j + 8 * (t ^ 1)];
                halves.sort_unstable();
                assert_eq!(halves, [(2 * j + t / 2) * 16, (2 * j + t / 2) * 16 + 8], "a pair is not one block");
            }
        }
        assert!(covered.iter().all(|&c| c), "lane {lane}: an A register slot never written");
        assert_eq!(g + 8 * (t & 1), cubecl_scales_index(lane, 'a'), "A scale row of lane {lane}");
        for s in 0..2 {
            for e in 0..4 {
                let block = 4 * s + e;
                let j = s * 2 + e / 2;
                let holder = if e % 2 == t / 2 { t } else { t ^ 2 };
                assert_eq!(((t ^ 2) & 1, 2 * j + holder / 2), (t & 1, block),
                    "A scale byte {e} of word {s}, lane {lane}: not the holder's block");
                assert_eq!((64 * s + 16 * e) / 16, block, "k-block {e} of step {s}");
            }
        }
        assert_eq!(g, cubecl_scales_index(lane, 'b'), "B scale column of lane {lane}");
        for i in 0..2 {
            for e in 0..2 {
                let n = 2 * i + e;
                assert_eq!((cubecl_row_index(lane, n, 'c'), cubecl_col_index(lane, n, 'c')),
                    (g + 8 * i, 2 * t + e), "accumulator element {n} of lane {lane}");
            }
        }
    }
}

/// `fp4quant`'s documented rule for one block, in host f32: RN-even E4M3 of
/// `min(amax / 6, 448)` and the reciprocal of its value. Both sides of
/// `fp4_q_quantize_replay_is_block_quantization` use it, so that test checks
/// the quantizer's lane split, never the rounding.
fn nvfp4_scale_host(amax: f32) -> (u8, f32) {
    let v = (amax / 6.0).min(448.0);
    let (mut best, mut bestd) = (0u8, f64::INFINITY);
    for b in 0u8..0x7f {
        let d = (f64::from(crate::nvfp4::e4m3_to_f32(b)) - f64::from(v)).abs();
        if d < bestd || (d == bestd && b % 2 == 0) {
            (best, bestd) = (b, d);
        }
    }
    let s = crate::nvfp4::e4m3_to_f32(best);
    (best, if s > 0.0 { 1.0 / s } else { 0.0 })
}

/// [`nvfp4_scale_host`]'s E2M1 half: one code of `x r`.
fn nvfp4_code_host(q: f32) -> u32 {
    let a = f64::from(q.abs());
    let mut m = 7u32;
    if a < 6.0 {
        let mut bestd = f64::INFINITY;
        for c in 0..8u32 {
            let d = (f64::from(crate::nvfp4::E2M1[c as usize]) - a).abs();
            if d < bestd || (d == bestd && c % 2 == 0) {
                (m, bestd) = (c, d);
            }
        }
    }
    if q.is_sign_negative() { m | 8 } else { m }
}

/// Host only: [`fp4_q_quantize`]'s lane split replayed on one plane's 16
/// heads (random rows, a zero block, a `-0.0` block, a clamped block), each
/// lane's partial maxima combined with lane `t ^ 1`'s and its scale words
/// assembled from its own and lane `t ^ 2`'s, against the same rule applied
/// whole-block in `fp4quant`'s layout and read through the A maps: every code
/// word and scale word equal. This pins the index arithmetic only.
#[test]
fn fp4_q_quantize_replay_is_block_quantization() {
    let rows = 16;
    let q: Vec<f32> = (0..rows * HEAD_DIM)
        .map(|i| {
            let (h, d) = (i / HEAD_DIM, i % HEAD_DIM);
            match (h, d / 16) {
                (3, 2) => 0.0,
                (5, 6) => -0.0,
                (9, 1) => if d % 2 == 0 { 4096.0 } else { -0.75 },
                _ => sample(i, 0xb2) * 2f32.powi((h % 7) as i32 - 3),
            }
        })
        .collect();
    let mut codes = vec![0u32; rows * 16];
    let mut scales = vec![0u8; rows * 8];
    for blk in 0..rows * 8 {
        let x = &q[blk * 16..blk * 16 + 16];
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (byte, r) = nvfp4_scale_host(amax);
        scales[blk] = byte;
        for (e, &v) in x.iter().enumerate() {
            codes[blk * 2 + e / 8] |= nvfp4_code_host(v * r) << (4 * (e % 8));
        }
    }
    let chunk = |lane: usize, hh: usize, j: usize| -> Vec<f32> {
        let (g, t) = (lane / 4, lane % 4);
        let base = (g + 8 * hh) * HEAD_DIM + 32 * j + 8 * t;
        q[base..base + 8].to_vec()
    };
    let part = |lane: usize, hh: usize, j: usize| chunk(lane, hh, j).iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let full = |lane: usize, hh: usize, j: usize| part(lane, hh, j).max(part(lane ^ 1, hh, j));
    for lane in 0..32usize {
        let (g, t) = (lane / 4, lane % 4);
        for hh in 0..2 {
            for j in 0..4 {
                let (_, r) = nvfp4_scale_host(full(lane, hh, j));
                let word = chunk(lane, hh, j).iter().enumerate()
                    .fold(0u32, |w, (e, &v)| w | (nvfp4_code_host(v * r) << (4 * e)));
                let (s, i) = (j / 2, (j % 2) * 2 + hh);
                let h = g + 8 * (i & 1);
                assert_eq!(word, codes[16 * h + 8 * s + 4 * (i >> 1) + t],
                    "lane {lane} A register {i} of step {s}: not fp4quant's code word");
            }
        }
        let own = |j: usize| full(lane, t & 1, j);
        let other = |j: usize| full(lane ^ 2, (t ^ 2) & 1, j);
        for s in 0..2 {
            let bytes: Vec<u8> = (0..4)
                .map(|e| {
                    let j = s * 2 + e / 2;
                    let amax = if e % 2 == t / 2 { own(j) } else { other(j) };
                    nvfp4_scale_host(amax).0
                })
                .collect();
            let h = g + 8 * (t & 1);
            assert_eq!(bytes, scales[8 * h + 4 * s..8 * h + 4 * s + 4],
                "lane {lane} A scale word {s}: not fp4quant's scale bytes");
        }
    }
}

/// `f32(e4m3)` by its bit construction, on the host.
fn e4m3_host(byte: u32) -> f32 {
    let (e, m) = ((byte >> 3) & 15, byte & 7);
    let mut value = m as f32 * 0.001953125;
    if e > 0 {
        value = f32::from_bits(((e + 120) << 23) | (m << 20));
    }
    if byte & 128 != 0 {
        value = -value;
    }
    value
}

/// Host only: the V decode cannot round. Every E2M1 value and every finite
/// E4M3 value (the NaN patterns excluded: the quantizer never writes them) is
/// exactly an f16, so the two hardware conversions and the widen are exact;
/// every code x non-negative finite scale product is exact in f32 and in BF16,
/// and never subnormal, so the decoded BF16 V is `dequantize_nvfp4_bf16`'s.
#[test]
fn nvfp4_v_decode_is_exact() {
    assert_eq!((0u32..256).filter(|b| b & 0x7f != 0x7f)
        .filter(|&b| e4m3_host(b).to_bits() == crate::nvfp4::e4m3_to_f32(b as u8).to_bits()).count(), 254);
    for code in 0..16usize {
        let value = crate::nvfp4::E2M1[code & 7] * if code & 8 != 0 { -1.0 } else { 1.0 };
        assert_eq!(half::f16::from_f32(value).to_f32().to_bits(), value.to_bits(), "E2M1 code {code}");
    }
    for byte in (0u32..256).filter(|b| b & 0x7f != 0x7f) {
        let value = e4m3_host(byte);
        assert_eq!(half::f16::from_f32(value).to_f32().to_bits(), value.to_bits(), "E4M3 {byte:#04x}");
    }
    let (mut top, mut least) = (0.0f32, f32::INFINITY);
    for scale_byte in 0u32..0x7f {
        let scale = e4m3_host(scale_byte);
        for code in 0..16usize {
            let value = crate::nvfp4::E2M1[code & 7] * if code & 8 != 0 { -1.0 } else { 1.0 };
            let product = value * scale;
            assert_eq!(f64::from(product), f64::from(value) * f64::from(scale), "inexact in f32");
            assert_eq!(bf16::from_f32(product).to_f32().to_bits(), product.to_bits(), "inexact in BF16");
            assert!(product == 0.0 || product.is_normal(), "{product} is subnormal");
            top = top.max(product.abs());
            if product != 0.0 {
                least = least.min(product.abs());
            }
        }
    }
    assert_eq!((top, least), (2688.0, 2f32.powi(-10)));
}

// ------------------------------------------------------------------ cases --

/// One finite synthetic case. `cut == 0` is one run over `[0, capacity)`;
/// otherwise a prefix run `[0, cut)` and a private run `[cut, capacity)` with
/// their own buffers and absolute bases. Rows at or past `keys` are dead.
#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    queries: usize,
    keys: usize,
    capacity: usize,
    q0: usize,
    cut: usize,
    uniform: bool,
}
impl Case {
    fn validate(self) {
        assert!((1..=LONG_QUERIES).contains(&self.queries));
        assert!(self.q0 + self.queries <= self.keys);
        assert!(self.keys <= self.capacity && self.capacity <= LONG_KEYS + TAIL);
        assert!(self.cut < self.keys);
    }
    fn spans(self) -> Vec<(usize, usize)> {
        if self.cut == 0 {
            vec![(0, self.capacity)]
        } else {
            vec![(0, self.cut), (self.cut, self.capacity - self.cut)]
        }
    }
    /// `(base, lo, hi)` of each run, as the launchers see them.
    fn runs(self) -> Vec<(usize, usize, usize)> {
        self.spans().into_iter().map(|(base, rows)| (base, 0, rows.min(self.keys - base))).collect()
    }
    fn json(self) -> Value {
        json!({"name":self.name,"queries":self.queries,"keys":self.keys,"capacity":self.capacity,
            "q0":self.q0,"cut":self.cut,"uniform":self.uniform,"runs":self.runs()})
    }
}

// ------------------------------------------------------------- coverage --

/// Host replay of the kernel's per-cube trimming and splitting (one KV head;
/// both are identical), counting the masking/padding events a case is meant
/// to exercise. Mirrors `mma_attention_kernel` line for line.
fn coverage(case: Case) -> Value {
    let warps = WARPS;
    let runs = case.runs();
    let spans: Vec<_> = runs.iter().map(|&(_, lo, hi)| hi - lo).collect();
    let q_tiles = query_tiles(case.queries);
    let splits = flash::key_run_splits(&spans, q_tiles, KV_HEADS);
    let (nq, q0) = (case.queries, case.q0);
    let mut c = [0usize; 7];
    for (&(k0, klo, khi), &n) in runs.iter().zip(&splits) {
        for qt in 0..q_tiles {
            let q_lo = q0 + qt * warps;
            let q_hi = if qt * warps + warps <= nq { q_lo + warps - 1 } else { q0 + nq - 1 };
            let lo = klo;
            let mut hi = khi;
            if q_hi + 1 <= k0 {
                hi = lo;
            } else if q_hi + 1 - k0 < hi {
                hi = q_hi + 1 - k0;
            }
            hi = hi.max(lo);
            let per = (hi - lo).div_ceil(n);
            for split in 0..n {
                let s_lo = (lo + split * per).min(hi);
                let s_hi = (s_lo + per).min(hi);
                if s_lo == s_hi {
                    c[1] += 1;
                }
                for w in 0..warps {
                    let p = qt * warps + w;
                    if p >= nq {
                        c[0] += 1;
                        continue;
                    }
                    let q_abs = q0 + p;
                    let (mut any, mut seen) = (false, false);
                    let mut t = s_lo;
                    while t < s_hi {
                        let end = (t + NK).min(s_hi);
                        c[3] += 1;
                        if end - t < NK {
                            c[6] += 1;
                        }
                        let visible = (t..end).filter(|&key| k0 + key <= q_abs).count();
                        if visible == 0 {
                            c[4] += 1;
                            if seen {
                                c[5] += 1;
                            }
                        } else {
                            seen = true;
                            any = true;
                        }
                        t += NK;
                    }
                    if s_lo < s_hi && !any {
                        c[2] += 1;
                    }
                }
            }
        }
    }
    json!({"warps":warps,"query_tiles":q_tiles,"splits":splits,"padded_plane_splits":c[0],
        "empty_splits":c[1],"fully_masked_nonempty_plane_splits":c[2],"plane_tiles":c[3],
        "fully_masked_plane_tiles":c[4],"masked_tile_after_visible":c[5],"partial_tail_tiles":c[6]})
}

fn count(cov: &Value, key: &str) -> u64 {
    cov[key].as_u64().unwrap()
}

// ------------------------------------------------------------- evidence --

fn evidence_root() -> &'static PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = PathBuf::from(
            std::env::var("NEMOTRON_ATTENTION_ROWS_OUTPUT")
                .expect("explicit new fixture evidence directory"),
        );
        assert!(root.is_absolute());
        fs::create_dir(&root).expect("new evidence directory, no overwrite");
        root
    })
}

/// One GPU fixture at a time in this process, so neither the device nor the
/// timing is shared between two of them.
fn gpu_turn() -> std::sync::MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

// One subdirectory per fixture, also created new.
struct Evidence {
    root: PathBuf,
    file: fs::File,
    started: Instant,
    records: usize,
}
impl Evidence {
    fn new(fixture: &str) -> Self {
        assert_eq!(std::env::var("CUBECL_MEMORY_CONFIG").as_deref(), Ok("exclusive"));
        assert!(std::env::var_os("CUBECL_DEBUG_LOG").is_none());
        assert!(std::env::var_os("CUBECL_DEBUG_OPTION").is_none());
        let root = evidence_root().join(fixture);
        fs::create_dir(&root).expect("new fixture evidence directory, no overwrite");
        let file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(root.join("events.jsonl"))
            .unwrap();
        Self { root, file, started: Instant::now(), records: 0 }
    }
    fn check(&self) {
        assert!(
            self.started.elapsed() < Duration::from_secs(MAX_SECONDS),
            "finite fixture exceeded elapsed budget; external timeout also required"
        );
    }
    fn record(&mut self, label: &str, data: Value) {
        self.check();
        self.records += 1;
        assert!(self.records <= MAX_RECORDS);
        let record = json!({"label":label,"process_elapsed_ns":self.started.elapsed().as_nanos(),
            "unix_ns":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),"data":data});
        let encoded = serde_json::to_vec(&record).unwrap();
        assert!(encoded.len() <= 16_384);
        self.file.write_all(&encoded).unwrap();
        self.file.write_all(b"\n").unwrap();
        self.file.flush().unwrap();
    }
}

// --------------------------------------------------------------- inputs --

fn sample(index: usize, salt: u64) -> f32 {
    let mut x = (index as u64).wrapping_add(SEED ^ salt).wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    bf16::from_f32(((x >> 48) as i32 - 32768) as f32 / 131072.0).to_f32()
}

fn fence(client: &ComputeClient<CudaRuntime>) {
    cubecl::future::block_on(client.sync()).expect("fixture completion fence");
}

fn upload(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    shape: &[usize],
    mut value: impl FnMut(usize) -> f32,
) -> CudaTensor {
    let n: usize = shape.iter().product();
    assert!(n * 2 <= (256 << 20) + TAIL * KV_WIDTH * 2);
    let mut bytes = Vec::with_capacity(n * 2);
    for i in 0..n {
        bytes.extend_from_slice(&bf16::from_f32(value(i)).to_bits().to_le_bytes());
    }
    let handle = client.create_from_slice(&bytes);
    // Retire this upload before generating the next large host input.
    fence(client);
    CudaTensor::new_contiguous(client.clone(), device.clone(), shape.into(), handle, DType::BF16)
}

/// One run's NVFP4 rows, as the cache writes them: `values` (`[rows, 256]`)
/// quantized on the device by `fp4quant::quantize_nvfp4_bf16`, then every
/// scale byte of the rows from `dead` on set to [`DEAD_SCALE`].
struct Fp4Run {
    codes: Handle,
    scales: Handle,
}
fn nvfp4_run(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    rows: usize,
    dead: usize,
    value: impl FnMut(usize) -> f32,
) -> Fp4Run {
    let x = upload(client, device, &[rows, KV_WIDTH], value);
    let (codes, scales) = fp4quant::quantize_nvfp4_bf16(client, &x.handle, rows, KV_WIDTH);
    fence(client);
    let width = KV_WIDTH / fp4quant::GROUP;
    let mut bytes = client.read_one(scales).unwrap()[..rows * width].to_vec();
    for b in bytes.iter_mut().skip(dead.min(rows) * width) {
        *b = DEAD_SCALE;
    }
    let scales = client.create_from_slice(&bytes);
    fence(client);
    Fp4Run { codes, scales }
}

fn key_buffer(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    case: Case,
    start: usize,
    rows: usize,
    salt: u64,
    changed: Option<usize>,
) -> Fp4Run {
    assert!(start + rows <= case.capacity);
    nvfp4_run(client, device, rows, case.keys.saturating_sub(start), |i| {
        let global = start * KV_WIDTH + i;
        if changed == Some(global / KV_WIDTH) { sample(global, salt ^ 0x5a5a) } else { sample(global, salt) }
    })
}

fn differing_bits(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

// --------------------------------------------------------------- launch --

struct Inputs {
    q: CudaTensor,
    owners: Vec<(Fp4Run, Fp4Run)>,
}
impl Inputs {
    /// `changed`: one absolute key row whose K and V take different finite values.
    fn new(client: &ComputeClient<CudaRuntime>, device: &CudaDevice, case: Case,
        changed: Option<usize>) -> Self
    {
        case.validate();
        let q = upload(client, device, &[1, case.queries, HIDDEN], |i| {
            if case.uniform { 0.0 } else { sample(i, 1) }
        });
        let owners = case
            .spans()
            .into_iter()
            .map(|(start, rows)| {
                (key_buffer(client, device, case, start, rows, 2, changed),
                    key_buffer(client, device, case, start, rows, 3, changed))
            })
            .collect();
        Self { q, owners }
    }
    fn runs(&self, case: Case) -> Vec<KeyRun<'_>> {
        self.owners
            .iter()
            .zip(case.runs())
            .zip(case.spans())
            .map(|(((k, v), (base, lo, hi)), (_, rows))| KeyRun {
                k: &k.codes,
                v: &v.codes,
                k_scales: Some(&k.scales),
                v_scales: Some(&v.scales),
                rows,
                base,
                lo,
                hi,
                row0: 0,
            })
            .collect()
    }
}

fn contract(evidence: &mut Evidence, cases: &[Case]) {
    evidence.record("contract", json!({"seed":SEED,"heads":HEADS,"kv_heads":KV_HEADS,
        "head_dim":HEAD_DIM,"nk":NK,"warps":WARPS,"stages":STAGES,"max_seconds":MAX_SECONDS,
        "mma_shared_bytes_per_cube":shared_bytes(),"shared_limit_bytes":SHARED_LIMIT_BYTES,
        "staging":STAGING,"kv":"nvfp4 rows by fp4quant::quantize_nvfp4_bf16; dead rows E4M3 NaN scales",
        "cases":cases.iter().map(|c| c.json()).collect::<Vec<_>>()}));
}

/// One complete launch, fenced, read back; every output must be finite.
fn launch(client: &ComputeClient<CudaRuntime>, inputs: &Inputs, case: Case) -> Vec<f32> {
    let runs = inputs.runs(case);
    let out = mma_attention_launch(client, &inputs.q.handle, &runs, case.queries, case.q0,
        1.0 / (HEAD_DIM as f32).sqrt());
    fence(client);
    let bytes = client.read_one(out).unwrap();
    let n = case.queries * HIDDEN;
    assert!(bytes.len() >= n * 4);
    let values: Vec<f32> =
        bytes[..n * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(values.iter().all(|v| v.is_finite()), "nonfinite output: {}", case.name);
    values
}

// ------------------------------------------------------------- fixtures --

const CASES: [Case; 13] = [
    Case { name: "small-17x281-cut0-uniform", queries: 17, keys: 281, capacity: 288, q0: 264,
        cut: 0, uniform: true },
    Case { name: "small-17x281-cut127-uniform", queries: 17, keys: 281, capacity: 288, q0: 264,
        cut: 127, uniform: true },
    Case { name: "small-17x281-cut0", queries: 17, keys: 281, capacity: 288, q0: 264, cut: 0,
        uniform: false },
    Case { name: "small-17x281-cut127", queries: 17, keys: 281, capacity: 288, q0: 264,
        cut: 127, uniform: false },
    Case { name: "small-1x281-cut127", queries: 1, keys: 281, capacity: 288, q0: 280, cut: 127,
        uniform: false },
    // nq = 3 < B4: one query tile, 1 padded plane.
    Case { name: "padded-3x281", queries: 3, keys: 281, capacity: 288, q0: 278, cut: 0,
        uniform: false },
    // One split (100 <= 128 keys); the last tile of the split ends at 4 of 32
    // keys; 17 queries also pad.
    Case { name: "tail-17x100", queries: 17, keys: 100, capacity: 112, q0: 83, cut: 0,
        uniform: false },
    // Prefix/private two-run split at an unaligned cut, over the same tail.
    Case { name: "two-run-17x100-cut37", queries: 17, keys: 100, capacity: 112, q0: 83,
        cut: 37, uniform: false },
    // One split; cube 0 (positions 62..65) trims to [0, 66), so the planes at
    // 62 and 63 see key tiles 0 and 1 and then a FULLY MASKED tile 2 (keys
    // 64..66).
    Case { name: "masked-tile-10x72", queries: 10, keys: 72, capacity: 80, q0: 62, cut: 0,
        uniform: false },
    // q0 = 0 over a 16,384-key run: the causal limit ends long before the
    // run's later tiles, most splits are EMPTY and many planes see none.
    Case { name: "masked-splits-17x16384", queries: 17, keys: 16_384, capacity: 16_400, q0: 0,
        cut: 0, uniform: false },
    // T1 decode: one live plane per cube, 512 key splits at most.
    Case { name: "decode-early-1x281", queries: 1, keys: 281, capacity: 288, q0: 200, cut: 0,
        uniform: false },
    Case { name: "decode-1x16384", queries: 1, keys: 16_384, capacity: 16_400, q0: 16_383,
        cut: 0, uniform: false },
    Case { name: "decode-1x16384-cut16368", queries: 1, keys: 16_384, capacity: 16_400,
        q0: 16_383, cut: 16_368, uniform: false },
];

/// The property each masking/padding fixture is named for must occur (host
/// replay of the launch, recorded beside it).
fn assert_coverage(case: Case, cov: &Value) {
    let need = |key: &str| {
        assert!(count(cov, key) > 0, "{} does not exercise {key}: {cov}", case.name)
    };
    match case.name {
        "padded-3x281" => need("padded_plane_splits"),
        "tail-17x100" | "two-run-17x100-cut37" => {
            need("partial_tail_tiles");
            need("padded_plane_splits");
        }
        "masked-tile-10x72" => need("masked_tile_after_visible"),
        "masked-splits-17x16384" => {
            need("empty_splits");
            need("fully_masked_nonempty_plane_splits");
        }
        _ => {}
    }
    if case.cut > 0 {
        assert_eq!(cov["splits"].as_array().unwrap().len(), 2);
    }
}

#[test]
#[ignore = "finite owned GPU: B2 mma attention finite, deterministic and causal at T1 and T>1 over NVFP4 K/V"]
fn gpu_mma_attention_invariants() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_invariants");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    contract(&mut evidence, &CASES);
    let mut checks = 0usize;
    for case in CASES {
        let cov = coverage(case);
        evidence.record("coverage", json!({"case":case.name,"b4":cov}));
        assert_coverage(case, &cov);
        let inputs = Inputs::new(&client, &device, case, None);
        // The last query's own key is visible to it alone; the next key, when
        // the run has one, to none of them.
        let own = case.q0 + case.queries - 1;
        let own_changed = Inputs::new(&client, &device, case, Some(own));
        let future = (own + 1 < case.keys).then(|| Inputs::new(&client, &device, case, Some(own + 1)));
        let last = (case.queries - 1) * HIDDEN;
        let first = launch(&client, &inputs, case);
        for _ in 0..2 {
            assert_eq!(differing_bits(&launch(&client, &inputs, case), &first), 0,
                "nondeterministic: {}", case.name);
        }
        let changed = launch(&client, &own_changed, case);
        assert_eq!(differing_bits(&changed[..last], &first[..last]), 0,
            "a key changed a query that cannot see it: {}", case.name);
        let own_bits = differing_bits(&changed[last..], &first[last..]);
        assert!(own_bits > 0, "a visible key changed nothing: {}", case.name);
        let future_bits = future.as_ref().map(|f| differing_bits(&launch(&client, f, case), &first));
        assert!(future_bits.is_none_or(|n| n == 0), "a future key leaked: {}", case.name);
        evidence.record("invariants", json!({"case":case.name,"warps":WARPS,
            "deterministic_launches":3,"own_key_differing":own_bits,"future_key_differing":future_bits}));
        checks += 1;
        drop((inputs, own_changed, future));
        fence(&client);
    }
    evidence.record("completed", json!({"passed":true,"checks":checks}));
    fs::write(evidence.root.join("complete.json"), b"{\"passed\":true}\n").unwrap();
}

/// A quiet-NaN bit pattern the kernel never produces (every partial it writes
/// is finite: masked keys are zero-staged, masked scores the finite NEG_INF).
const CANARY: u32 = 0x7fc0_cafe;

/// The partial pass alone, into `po`/`pml` filled with [`CANARY`] and LONGER
/// than the launch owns by [`WARPS`] query rows. A padded plane (p >= nq) that
/// lost its `live` guard writes cell `slot * nq + p`: in the LAST slot that
/// lands past `slots * nq`, inside the canary region, every time. So: every
/// owned word must be written and finite, and every canary word untouched.
fn partial_canary(
    client: &ComputeClient<CudaRuntime>,
    inputs: &Inputs,
    case: Case,
    evidence: &mut Evidence,
) {
    let runs = inputs.runs(case);
    let nq = case.queries;
    let split_of = split_plan(&runs, nq);
    let slots: usize = split_of.iter().sum();
    let rows = slots * nq;
    let past_end = query_tiles(nq) * WARPS - nq;
    let (po_live, po_len) = (rows * HEADS * HEAD_DIM, (rows + WARPS) * HEADS * HEAD_DIM);
    let (pml_live, pml_len) = (rows * HEADS * 2, (rows + WARPS) * HEADS * 2);
    let fill = |n: usize| {
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&CANARY.to_le_bytes());
        }
        client.create_from_slice(&bytes)
    };
    let po = fill(po_len);
    let pml = fill(pml_len);
    fence(client);
    let scaling = 1.0 / (HEAD_DIM as f32).sqrt();
    mma_attention_partials(client, &inputs.q.handle, &runs, nq, case.q0, scaling,
        &split_of, &po, po_len, &pml, pml_len);
    fence(client);
    let words = |h: Handle, n: usize| -> Vec<u32> {
        let bytes = client.read_one(h).unwrap();
        assert!(bytes.len() >= n * 4);
        bytes[..n * 4].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
    };
    let mut buffers = Vec::new();
    let mut clean = true;
    for (name, got, live) in [("po", words(po, po_len), po_live), ("pml", words(pml, pml_len), pml_live)] {
        let unwritten = got[..live].iter().filter(|&&w| w == CANARY).count();
        let nonfinite = got[..live].iter().filter(|&&w| !f32::from_bits(w).is_finite()).count();
        let past_end_written = got[live..].iter().filter(|&&w| w != CANARY).count();
        clean &= unwritten == 0 && nonfinite == 0 && past_end_written == 0;
        buffers.push(json!({"buffer":name,"owned_words":live,"canary_words":got.len() - live,
            "owned_unwritten":unwritten,"owned_nonfinite":nonfinite,
            "canary_overwritten":past_end_written}));
    }
    evidence.record("partial_canary", json!({"case":case.name,"arm":STAGING,"splits":split_of,
        "slots":slots,"padded_rows_past_last_slot":past_end,"canary":format!("{CANARY:#010x}"),
        "buffers":buffers,"clean":clean}));
    if matches!(case.name, "padded-3x281" | "tail-17x100" | "two-run-17x100-cut37") {
        assert!(past_end > 0, "{} has no padded plane past the last slot", case.name);
    }
    assert!(clean, "partial pass wrote outside its cells or left one unwritten: {}", case.name);
}

#[test]
#[ignore = "finite owned GPU: B2 mma partial pass writes every owned cell and nothing past them (NaN canary)"]
fn gpu_mma_attention_partial_canary() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_partial_canary");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    contract(&mut evidence, &CASES);
    let mut checks = 0usize;
    for case in CASES {
        let inputs = Inputs::new(&client, &device, case, None);
        partial_canary(&client, &inputs, case, &mut evidence);
        checks += 1;
        drop(inputs);
        fence(&client);
    }
    evidence.record("completed", json!({"passed":true,"checks":checks}));
    fs::write(evidence.root.join("complete.json"), b"{\"passed\":true}\n").unwrap();
}

#[test]
#[ignore = "reserved GPU: checked B2 MMA boundary nonzero lo and fully masked output; under 1MiB named payload"]
fn gpu_mma_checked_nonzero_lo_and_all_masked() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_checked_ranges");
    // Admit named payload BEFORE upload/launch, not as after-the-fact accounting.
    let max = Plan::new(&[2], 3, 3).unwrap();
    assert!(max.working_bytes + 3 * HIDDEN * 2 + 2 * 4 * KV_WIDTH * 2 < 1 << 20);
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let q = upload(&client, &device, &[3, HIDDEN], |_| 0.0);
    // Rows 0 and 1 sit below lo and carry the NaN scale: never read. V = 3.0
    // is exact in NVFP4 (scale 0.5, code 6), so the mean is exactly 3.0.
    let k = nvfp4_run(&client, &device, 4, 4, |_| 0.0);
    let v = nvfp4_run(&client, &device, 4, 4, |_| 3.0);
    let poison = |run: &Fp4Run| {
        let mut bytes = client.read_one(run.scales.clone()).unwrap()[..4 * 16].to_vec();
        bytes[..2 * 16].fill(DEAD_SCALE);
        client.create_from_slice(&bytes)
    };
    let (ks, vs) = (poison(&k), poison(&v));
    fence(&client);
    for tokens in [1, 3] {
        for masked in [false, true] {
            let run = KeyRun { k: &k.codes, v: &v.codes, k_scales: Some(&ks), v_scales: Some(&vs),
                rows: 4, base: if masked { 10 } else { 0 }, lo: 2, hi: 4, row0: 0 };
            let plan = Plan::new(&[2], tokens, if masked { 0 } else { 3 }).unwrap();
            let output = launch_planned(&client, &q.handle, &[run], 1.0 / 128.0f32.sqrt(), &plan).unwrap();
            fence(&client);
            let bytes = client.read_one(output).unwrap();
            let values: Vec<_> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let expected = if masked { 0.0 } else { 3.0 };
            let wrong = values.iter().filter(|&&v| !v.is_finite() || v != expected).count();
            evidence.record("checked-range", json!({"tokens":tokens,"warps":WARPS,
                "lo":2,"fully_masked":masked,"elements":values.len(),"wrong":wrong}));
            assert_eq!(values.len(), tokens * HIDDEN);
            assert_eq!(wrong, 0, "uniform value/masking contract, not model capability");
        }
    }
    fence(&client);
}

/// Synthetic NVFP4 rows for timing: random code bytes and scale bytes in
/// `0x28..0x38` (finite, 2^-4 to below 1), `rows` of them, made by repeating
/// one 4,096-row block so a long run costs no host time to build.
fn timing_rows(client: &ComputeClient<CudaRuntime>, rows: usize, salt: u64) -> Fp4Run {
    let block = 4096.min(rows);
    let mut state = SEED ^ salt;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let codes: Vec<u8> = (0..block * KV_WIDTH / 2).map(|_| next() as u8).collect();
    let scales: Vec<u8> = (0..block * KV_WIDTH / 16).map(|_| 0x28 + (next() % 16) as u8).collect();
    let repeat = |b: &[u8]| -> Vec<u8> {
        let mut out = Vec::with_capacity(b.len() * rows / block);
        for _ in 0..rows / block {
            out.extend_from_slice(b);
        }
        out
    };
    assert_eq!(rows % block, 0);
    let run = Fp4Run { codes: client.create_from_slice(&repeat(&codes)),
        scales: client.create_from_slice(&repeat(&scales)) };
    fence(client);
    run
}

/// Per-launch time of the production attention (`launch_planned`: the
/// partial pass, its scratch allocation and the combine) at T256 at the end
/// of K = 4,096, 131,072 and 524,288 keys, one run, warm. Host wall from the
/// enqueue to a completion fence after a fence that drained earlier work.
/// Record only; synthetic data.
#[test]
#[ignore = "finite owned GPU: B2 production attention per-launch timing at T256 over K 4k/128k/524k, record only"]
fn gpu_mma_attention_timing() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_timing");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let q = upload(&client, &device, &[LONG_QUERIES, HIDDEN], |i| sample(i, 1));
    let scaling = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut summary = Vec::new();
    for keys in [4096usize, 131_072, 524_288] {
        let (k, v) = (timing_rows(&client, keys, 2), timing_rows(&client, keys, 3));
        let run = KeyRun { k: &k.codes, v: &v.codes, k_scales: Some(&k.scales), v_scales: Some(&v.scales),
            rows: keys, base: 0, lo: 0, hi: keys, row0: 0 };
        let q0 = keys - LONG_QUERIES;
        let plan = Plan::new(&[keys], LONG_QUERIES, q0).unwrap();
        let mut walls = Vec::new();
        for rep in 0..8 {
            fence(&client);
            let start = Instant::now();
            let out = launch_planned(&client, &q.handle, &[run], scaling, &plan).unwrap();
            fence(&client);
            let wall = start.elapsed().as_nanos() as u64;
            drop(out);
            if rep >= 2 {
                walls.push(wall);
            }
        }
        let mut sorted = walls.clone();
        sorted.sort_unstable();
        let median = (sorted[2] + sorted[3]) as f64 / 2e6;
        println!("B2 TIMING {STAGING}: T{LONG_QUERIES} over K {keys}: median {median:.2} ms \
            [{:.2}..{:.2}] per launch (6 warm after 2 cold; splits {:?}; host wall, enqueue to fence)",
            sorted[0] as f64 / 1e6, sorted[5] as f64 / 1e6, plan.split_of);
        evidence.record("timing", json!({"keys":keys,"queries":LONG_QUERIES,"q0":q0,"splits":plan.split_of,
            "warm_walls_ns":walls,"median_ms":median,"arm":STAGING}));
        summary.push(json!({"keys":keys,"median_ms":median}));
        drop((k, v));
        fence(&client);
    }
    evidence.record("completed", json!({"passed":true,"summary":summary,
        "framing":"per attention launch (partials + combine), T256 at the end of K, one run, warm, host wall"}));
    fs::write(evidence.root.join("complete.json"), b"{\"passed\":true}\n").unwrap();
}
