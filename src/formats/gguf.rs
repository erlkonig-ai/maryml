//! GGUF → `(name, f32-data, shape)` tensor extractor for the import path.
//!
//! GGUF (the single-file llama.cpp weight container) ships many LLMs that exist
//! in NO other format. This module parses a `.gguf` file's tensor table with the
//! lean [`gguf_rs_lib`] reader (container only — no backend, no async), then
//! DEQUANTIZES each tensor's raw block bytes to `Vec<f32>` here, following the
//! canonical llama.cpp `ggml-quants.c` block layouts. The resulting
//! `(name, f32-data, shape)` tuples feed [`crate::ingest::ingest_tensors`] — the
//! SAME content-addressed leaf/member path safetensors uses — so a GGUF import
//! lands in the identical model graph (the root id is the pure hash of the f32
//! members, format-independent).
//!
//! Fidelity: F32/F16/BF16 are exact; quantized types are dequantized to the
//! same f32 values llama.cpp produces (the weights were already lossy at quant
//! time — this recovers the stored approximation, it does not invent precision).
//! The IQ* (importance-quant, codebook) families are not decoded here and return
//! an explicit error rather than silently wrong weights.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};

use gguf_rs_lib::prelude::GGUFTensorType;

// Selective, checked import/parity path for modern GGML type tags. The existing
// extract_tensors caller and its output semantics are deliberately unchanged.
pub mod index;
pub mod nvfp4;

/// GGML super-block element count for the K-quant families (Q2_K..Q8_K).
const QK_K: usize = 256;

/// Read a GGUF file and return every tensor as `(name, f32-data, row-major
/// shape)`. The shape is emitted in row-major (C) order — GGUF stores dims
/// fastest-first (ne[0] is the contiguous dimension), so we reverse them to
/// match the safetensors / PyTorch `[out, in, ...]` convention the rest of mary
/// (and the model forwards) assume.
///
/// We take the tensor NAME/TYPE/DIMS/OFFSET from the gguf-rs-lib reader (which
/// parses the header faithfully) but compute the on-disk BYTE SIZE ourselves
/// from the canonical ggml block layout ([`block_bytes`]) — gguf-rs-lib's own
/// size table has "approximate" entries (Q2_K, Q8_K, IQ4_NL) that under-read the
/// data, so we bypass its `load_tensor_data` and slice by our own size at the
/// reader-reported offset.
pub fn extract_tensors(path: &Path) -> Result<Vec<(String, Vec<f32>, Vec<usize>)>> {
    let mut reader = gguf_rs_lib::reader::file_reader::open_gguf_file(path)
        .map_err(|e| anyhow!("open gguf {path:?}: {e}"))?;

    // Snapshot the tensor table (name, type, dims, data offset) so the mutable
    // `read_tensor_data_at` borrow below doesn't collide with the info borrow.
    let infos: Vec<(String, GGUFTensorType, Vec<usize>, u64)> = reader
        .tensor_infos()
        .iter()
        .map(|ti| {
            // GGUF dims are fastest-first; reverse to row-major for mary.
            let mut dims: Vec<usize> = ti.shape().dims().iter().map(|&d| d as usize).collect();
            dims.reverse();
            (
                ti.name().to_string(),
                ti.tensor_type(),
                dims,
                ti.data_offset(),
            )
        })
        .collect();

    let mut out = Vec::with_capacity(infos.len());
    for (name, ty, shape, offset) in infos {
        let n: usize = shape
            .iter()
            .product::<usize>()
            .max(if shape.is_empty() { 0 } else { 1 });
        let nbytes =
            block_bytes(ty, n).with_context(|| format!("size gguf tensor {name:?} ({ty:?})"))?;
        let data = reader
            .read_tensor_data_at(offset, nbytes)
            .map_err(|e| anyhow!("read gguf tensor {name:?} @ {offset} ({nbytes} B): {e}"))?;
        let raw = data.as_slice();
        let f32s = dequantize(ty, raw, n)
            .with_context(|| format!("dequantize gguf tensor {name:?} ({ty:?})"))?;
        out.push((name, f32s, shape));
    }
    Ok(out)
}

/// The exact on-disk byte size of `n` elements of ggml type `ty`, from the
/// canonical block layout (llama.cpp `ggml.c` type traits). This is the source
/// of truth mary reads with — NOT gguf-rs-lib's approximate size table.
fn block_bytes(ty: GGUFTensorType, n: usize) -> Result<usize> {
    use GGUFTensorType as T;
    // (elements per block, bytes per block)
    let (qk, bytes) = match ty {
        T::F32 => (1, 4),
        T::F16 | T::BF16 => (1, 2),
        T::Q4_0 => (32, 18),
        T::Q4_1 => (32, 20),
        T::Q5_0 => (32, 22),
        T::Q5_1 => (32, 24),
        T::Q8_0 => (32, 34),
        T::Q8_1 => (32, 36),
        T::IQ4_NL => (32, 18),
        T::Q2_K => (QK_K, 84),
        T::Q3_K => (QK_K, 110),
        T::Q4_K => (QK_K, 144),
        T::Q5_K => (QK_K, 176),
        T::Q6_K => (QK_K, 210),
        T::Q8_K => (QK_K, 4 + QK_K + 2 * (QK_K / 16)), // d(f32) + qs(256) + 16 bsums(i16) = 292
        other => bail!("no block layout for ggml type {other:?}"),
    };
    Ok(n.div_ceil(qk) * bytes)
}

/// Dequantize `n` elements of ggml type `ty` from `raw` block bytes into f32.
fn dequantize(ty: GGUFTensorType, raw: &[u8], n: usize) -> Result<Vec<f32>> {
    use GGUFTensorType as T;
    match ty {
        T::F32 => Ok(read_f32(raw, n)),
        T::F16 => Ok(read_f16(raw, n)),
        T::BF16 => Ok(read_bf16(raw, n)),
        T::Q4_0 => dequant_q4_0(raw, n),
        T::Q4_1 => dequant_q4_1(raw, n),
        T::Q5_0 => dequant_q5_0(raw, n),
        T::Q5_1 => dequant_q5_1(raw, n),
        T::Q8_0 => dequant_q8_0(raw, n),
        T::Q8_1 => dequant_q8_1(raw, n),
        T::Q2_K => dequant_q2_k(raw, n),
        T::Q3_K => dequant_q3_k(raw, n),
        T::Q4_K => dequant_q4_k(raw, n),
        T::Q5_K => dequant_q5_k(raw, n),
        T::Q6_K => dequant_q6_k(raw, n),
        T::Q8_K => dequant_q8_k(raw, n),
        T::IQ4_NL => dequant_iq4_nl(raw, n),
        other => bail!(
            "ggml type {other:?} is not supported by mary's GGUF importer \
             (the IQ2/IQ3/IQ1 codebook-grid quants are not decoded); re-export \
             the model as F16/Q8_0/Q4_K/Q6_K or import its safetensors instead"
        ),
    }
}

// ---- unquantized readers -------------------------------------------------

fn read_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(4)
        .take(n)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_f16(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(2)
        .take(n)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

fn read_bf16(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(2)
        .take(n)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

// ---- small helpers -------------------------------------------------------

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

// ---- legacy 32-element block quants --------------------------------------
// Block size is QK4_0 = QK5_0 = QK8_0 = 32 elements.

// Element order for the nibble-split legacy quants (Q4_0/Q4_1/Q5_0/Q5_1): the
// 16 low nibbles come first (elements 0..16), then the 16 high nibbles
// (elements 16..32) — NOT interleaved. Matches llama.cpp's `qs[j]&0xF` then
// `qs[j]>>4` two-pass unpack.

/// Q4_0: [f16 d][16 bytes: 32 nibbles], each weight = d * (nibble - 8).
fn dequant_q4_0(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK: usize = 32;
    const BLOCK: usize = 2 + 16;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK)) {
        let d = f16_at(blk, 0);
        let qs = &blk[2..2 + 16];
        for &q in qs {
            out.push(d * ((q & 0x0F) as i32 - 8) as f32);
        }
        for &q in qs {
            out.push(d * ((q >> 4) as i32 - 8) as f32);
        }
    }
    finish(out, n, "Q4_0")
}

/// Q4_1: [f16 d][f16 m][16 bytes: 32 nibbles], weight = d * nibble + m.
fn dequant_q4_1(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK: usize = 32;
    const BLOCK: usize = 2 + 2 + 16;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK)) {
        let d = f16_at(blk, 0);
        let m = f16_at(blk, 2);
        let qs = &blk[4..4 + 16];
        for &q in qs {
            out.push(d * (q & 0x0F) as f32 + m);
        }
        for &q in qs {
            out.push(d * (q >> 4) as f32 + m);
        }
    }
    finish(out, n, "Q4_1")
}

/// Q5_0: [f16 d][u32 qh (high bit per value)][16 bytes low nibbles],
/// weight = d * ((low | (high<<4)) - 16). Low nibbles (elems 0..16) then high
/// nibbles (elems 16..32); the qh bit for element j is bit j of qh.
fn dequant_q5_0(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK: usize = 32;
    const BLOCK: usize = 2 + 4 + 16;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK)) {
        let d = f16_at(blk, 0);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..6 + 16];
        for (j, &q) in qs.iter().enumerate() {
            let hi = ((qh >> j) & 1) << 4;
            out.push(d * (((q & 0x0F) as u32 | hi) as i32 - 16) as f32);
        }
        for (j, &q) in qs.iter().enumerate() {
            let hi = ((qh >> (j + 16)) & 1) << 4;
            out.push(d * (((q >> 4) as u32 | hi) as i32 - 16) as f32);
        }
    }
    finish(out, n, "Q5_0")
}

/// Q5_1: [f16 d][f16 m][u32 qh][16 bytes low nibbles],
/// weight = d * (low | (high<<4)) + m.
fn dequant_q5_1(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK: usize = 32;
    const BLOCK: usize = 2 + 2 + 4 + 16;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK)) {
        let d = f16_at(blk, 0);
        let m = f16_at(blk, 2);
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let qs = &blk[8..8 + 16];
        for (j, &q) in qs.iter().enumerate() {
            let hi = ((qh >> j) & 1) << 4;
            out.push(d * ((q & 0x0F) as u32 | hi) as f32 + m);
        }
        for (j, &q) in qs.iter().enumerate() {
            let hi = ((qh >> (j + 16)) & 1) << 4;
            out.push(d * ((q >> 4) as u32 | hi) as f32 + m);
        }
    }
    finish(out, n, "Q5_1")
}

/// Q8_0: [f16 d][32 i8], weight = d * i8.
fn dequant_q8_0(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK: usize = 32;
    const BLOCK: usize = 2 + 32;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK)) {
        let d = f16_at(blk, 0);
        for j in 0..32 {
            out.push(d * (blk[2 + j] as i8) as f32);
        }
    }
    finish(out, n, "Q8_0")
}

/// Q8_1: [f16 d][f16 s][32 i8], weight = d * i8. (`s` is a precomputed sum used
/// only by matmul accumulation; it does not enter the dequantized value.)
fn dequant_q8_1(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK: usize = 32;
    const BLOCK: usize = 2 + 2 + 32;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK)) {
        let d = f16_at(blk, 0);
        for j in 0..32 {
            out.push(d * (blk[4 + j] as i8) as f32);
        }
    }
    finish(out, n, "Q8_1")
}

// ---- K-quant super-block quants (256 elements per super-block) ------------

/// The 6-bit packed scale/min unpack shared by Q4_K / Q5_K (`get_scale_min_k4`
/// in ggml): 8 sub-blocks, each a (scale, min) pair packed across 12 bytes.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Q2_K super-block (84 bytes / 256 elems):
/// [16 scales (4-bit d + 4-bit m per sub-block)][64 q (2-bit)][f16 d][f16 dmin].
fn dequant_q2_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 16 + 64 + 2 + 2;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK_K)) {
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = f16_at(blk, 80);
        let dmin = f16_at(blk, 82);
        // 2 groups of 128 elems; each group: 8 bit-shifts × 16 lanes.
        let mut is = 0usize;
        for group in 0..2 {
            let qbase = group * 32;
            for shift_i in 0..4 {
                let shift = shift_i * 2;
                for sub in 0..2 {
                    let sc = scales[is];
                    let dl = d * (sc & 0x0F) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    let off = qbase + sub * 16;
                    for l in 0..16 {
                        let q = ((qs[off + l] >> shift) & 3) as f32;
                        out.push(dl * q - ml);
                    }
                    is += 1;
                }
            }
        }
    }
    finish(out, n, "Q2_K")
}

/// Q3_K super-block (110 bytes / 256 elems):
/// [32 hmask][64 qs (2-bit low)][12 scales (6-bit, packed)][f16 d]. Mirrors
/// llama.cpp `dequantize_row_q3_K` exactly (aux-word scale shuffle; scale used
/// as unsigned then `-32`; hmask bit gives a `-4` offset when CLEAR).
fn dequant_q3_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 32 + 64 + 12 + 2;
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK_K)) {
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let scales_raw = &blk[96..108];
        let d_all = f16_at(blk, 108);

        // Unpack the 16 6-bit scales into a byte array (kept UNSIGNED; the -32
        // bias is applied at use, per the reference).
        let mut aux = [0u32; 3];
        for (i, a) in aux.iter_mut().enumerate() {
            *a = u32::from_le_bytes([
                scales_raw[i * 4],
                scales_raw[i * 4 + 1],
                scales_raw[i * 4 + 2],
                scales_raw[i * 4 + 3],
            ]);
        }
        let mut w = [0u32; 4];
        let tmp = aux[2];
        w[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        w[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        w[0] = (aux[0] & kmask2) | (((tmp) & kmask1) << 4);
        w[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let mut sc = [0u8; 16];
        for k in 0..4 {
            sc[k * 4..k * 4 + 4].copy_from_slice(&w[k].to_le_bytes());
        }

        // 2 halves of 128; within each, 4 shift levels. Per shift, two
        // consecutive scales (sc[is++], sc[is++]) cover the low-16 and high-16
        // lanes; `m` walks the 8 hmask bits across BOTH halves (never resets).
        let mut is = 0usize;
        let mut m: u8 = 1;
        for group in 0..2 {
            let qbase = group * 32;
            for shift_i in 0..4 {
                let shift = shift_i * 2;
                // hmask is indexed by the in-32 position ONLY (0..32); the group
                // is distinguished by the bit `m`, not a byte offset (unlike qs,
                // which advances 32 bytes per group via `qbase`).
                let dl0 = d_all * (sc[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let low = ((qs[qbase + l] >> shift) & 3) as i32;
                    let high = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    out.push(dl0 * (low - high) as f32);
                }
                let dl1 = d_all * (sc[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let low = ((qs[qbase + 16 + l] >> shift) & 3) as i32;
                    let high = if (hmask[16 + l] & m) != 0 { 0 } else { 4 };
                    out.push(dl1 * (low - high) as f32);
                }
                m <<= 1;
            }
        }
    }
    finish(out, n, "Q3_K")
}

/// Q4_K super-block (144 bytes / 256 elems):
/// [f16 d][f16 dmin][12 scales (packed 6-bit)][128 qs (4-bit)].
fn dequant_q4_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 2 + 2 + 12 + 128;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK_K)) {
        let d = f16_at(blk, 0);
        let dmin = f16_at(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..16 + 128];
        // 4 groups of 64, each group has 2 sub-blocks (low nibble, high nibble).
        let mut out_group = [0f32; 256];
        let mut written = 0usize;
        for j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(2 * j, scales);
            let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            let base = j * 32;
            for l in 0..32 {
                out_group[written] = d1 * (qs[base + l] & 0x0F) as f32 - mm1;
                written += 1;
            }
            for l in 0..32 {
                out_group[written] = d2 * (qs[base + l] >> 4) as f32 - mm2;
                written += 1;
            }
        }
        out.extend_from_slice(&out_group);
    }
    finish(out, n, "Q4_K")
}

/// Q5_K super-block (176 bytes / 256 elems):
/// [f16 d][f16 dmin][12 scales][32 qh (high bit)][128 qs (4-bit)].
fn dequant_q5_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 2 + 2 + 12 + 32 + 128;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK_K)) {
        let d = f16_at(blk, 0);
        let dmin = f16_at(blk, 2);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..48 + 128];
        let mut out_group = [0f32; 256];
        let mut written = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(2 * j, scales);
            let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            let base = j * 32;
            for l in 0..32 {
                let hi = if (qh[l] & u1) != 0 { 16 } else { 0 };
                out_group[written] = d1 * ((qs[base + l] & 0x0F) as i32 + hi) as f32 - mm1;
                written += 1;
            }
            for l in 0..32 {
                let hi = if (qh[l] & u2) != 0 { 16 } else { 0 };
                out_group[written] = d2 * ((qs[base + l] >> 4) as i32 + hi) as f32 - mm2;
                written += 1;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
        out.extend_from_slice(&out_group);
    }
    finish(out, n, "Q5_K")
}

/// Q6_K super-block (210 bytes / 256 elems):
/// [128 ql (4-bit low)][64 qh (2-bit high)][16 i8 scales][f16 d].
fn dequant_q6_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 128 + 64 + 16 + 2;
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK_K)) {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d = f16_at(blk, 208);
        // Reconstruct into a fixed 256-slot scratch (Q6_K writes out of order),
        // then append. Two 128-element halves; each byte of ql/qh yields 4 elems.
        let mut sb = [0f32; 256];
        for half in 0..2 {
            let ql_base = half * 64;
            let qh_base = half * 32;
            let sc_base = half * 8;
            let out_base = half * 128;
            let scale = |k: usize| d * (sc[sc_base + k] as i8) as f32;
            for l in 0..32 {
                let q1 = ((ql[ql_base + l] & 0x0F) as i32
                    | (((qh[qh_base + l] >> 0) & 3) as i32) << 4)
                    - 32;
                let q2 = ((ql[ql_base + l + 32] & 0x0F) as i32
                    | (((qh[qh_base + l] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 = ((ql[ql_base + l] >> 4) as i32
                    | (((qh[qh_base + l] >> 4) & 3) as i32) << 4)
                    - 32;
                let q4 = ((ql[ql_base + l + 32] >> 4) as i32
                    | (((qh[qh_base + l] >> 6) & 3) as i32) << 4)
                    - 32;
                // Positions l, l+32, l+64, l+96 within the half; scale sub-block
                // is (position)/16.
                sb[out_base + l] = scale(l / 16) * q1 as f32;
                sb[out_base + l + 32] = scale((l + 32) / 16) * q2 as f32;
                sb[out_base + l + 64] = scale((l + 64) / 16) * q3 as f32;
                sb[out_base + l + 96] = scale((l + 96) / 16) * q4 as f32;
            }
        }
        out.extend_from_slice(&sb);
    }
    finish(out, n, "Q6_K")
}

/// Q8_K super-block (292 bytes / 256 elems): [f32 d][256 i8][... aux].
/// Only `d` and the 256 int8 quants enter the dequantized value.
fn dequant_q8_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 4 + 256 + 32; // d(f32) + qs(256×i8) + 16 int16 block sums = 292
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK_K)) {
        let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        for l in 0..256 {
            out.push(d * (blk[4 + l] as i8) as f32);
        }
    }
    finish(out, n, "Q8_K")
}

/// IQ4_NL non-linear 4-bit block (18 bytes / 32 elems): [f16 d][16 nibble
/// bytes]. Each 4-bit index selects a value from a fixed 16-entry codebook
/// (`kvalues`, shared with llama.cpp); weight = d * kvalues[nibble]. The one IQ
/// family mary decodes — it appears constantly on the embedding/attn tensors of
/// "mixed" K-quant files, and is a pure lookup (no codebook grid).
fn dequant_iq4_nl(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK: usize = 32;
    const BLOCK: usize = 2 + 16;
    const KVALUES: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(BLOCK).take(n.div_ceil(QK)) {
        let d = f16_at(blk, 0);
        let qs = &blk[2..2 + 16];
        // Low nibbles first (all 16 lanes), then high nibbles — matches the
        // `>> [0,4]` unpack order in the reference.
        for &q in qs {
            out.push(d * KVALUES[(q & 0x0F) as usize] as f32);
        }
        for &q in qs {
            out.push(d * KVALUES[(q >> 4) as usize] as f32);
        }
    }
    finish(out, n, "IQ4_NL")
}

// ---- output plumbing -----------------------------------------------------

/// Truncate to exactly `n` (the last super-block may over-produce) and verify we
/// produced at least `n` values.
fn finish(mut out: Vec<f32>, n: usize, ty: &str) -> Result<Vec<f32>> {
    if out.len() < n {
        bail!(
            "{ty}: dequantized {} values but tensor needs {n} (truncated GGUF data?)",
            out.len()
        );
    }
    out.truncate(n);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(x: f32) -> [u8; 2] {
        f16::from_f32(x).to_le_bytes()
    }

    #[test]
    fn q8_0_roundtrip() {
        // One 32-element block: d = 0.5, quants 0..32 (wrapping to i8).
        let d = 0.5f32;
        let mut blk = Vec::new();
        blk.extend_from_slice(&f16_bytes(d));
        let qs: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
        blk.extend(qs.iter().map(|&q| q as u8));
        let out = dequant_q8_0(&blk, 32).unwrap();
        for (i, &q) in qs.iter().enumerate() {
            assert!(
                (out[i] - d * q as f32).abs() < 1e-4,
                "elem {i}: {} vs {}",
                out[i],
                d * q as f32
            );
        }
    }

    #[test]
    fn q4_0_roundtrip() {
        // d = 2.0; nibbles n -> weight = d*(n-8). Pack pairs (lo=j, hi=j).
        let d = 2.0f32;
        let mut blk = Vec::new();
        blk.extend_from_slice(&f16_bytes(d));
        // 16 bytes: byte j has low nibble = j%16, high nibble = (j+1)%16
        for j in 0..16u8 {
            let lo = j & 0x0F;
            let hi = (j + 1) & 0x0F;
            blk.push(lo | (hi << 4));
        }
        let out = dequant_q4_0(&blk, 32).unwrap();
        // order: 16 low nibbles first, then 16 high nibbles.
        for j in 0..16usize {
            let lo = (j as i32) & 0x0F;
            let hi = ((j as i32) + 1) & 0x0F;
            assert!((out[j] - d * (lo - 8) as f32).abs() < 1e-4, "lo {j}");
            assert!((out[16 + j] - d * (hi - 8) as f32).abs() < 1e-4, "hi {j}");
        }
    }

    #[test]
    fn q4_1_roundtrip() {
        // weight = d*nibble + m
        let d = 0.25f32;
        let m = -1.0f32;
        let mut blk = Vec::new();
        blk.extend_from_slice(&f16_bytes(d));
        blk.extend_from_slice(&f16_bytes(m));
        for _ in 0..16u8 {
            blk.push(0x30); // lo=0, hi=3
        }
        let out = dequant_q4_1(&blk, 32).unwrap();
        // order: 16 low nibbles (=0) first, then 16 high nibbles (=3).
        for j in 0..16usize {
            assert!((out[j] - (d * 0.0 + m)).abs() < 1e-3);
            assert!((out[16 + j] - (d * 3.0 + m)).abs() < 1e-3);
        }
    }

    #[test]
    fn f16_and_bf16_readers() {
        let vals = [1.5f32, -2.0, 0.0, 100.0];
        let f16b: Vec<u8> = vals
            .iter()
            .flat_map(|&x| f16::from_f32(x).to_le_bytes())
            .collect();
        let got = read_f16(&f16b, 4);
        for (a, b) in got.iter().zip(vals.iter()) {
            assert!((a - b).abs() < 0.1);
        }
        let bf16b: Vec<u8> = vals
            .iter()
            .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
            .collect();
        let got = read_bf16(&bf16b, 4);
        for (a, b) in got.iter().zip(vals.iter()) {
            assert!((a - b).abs() < 1.0);
        }
    }
}
