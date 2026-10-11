//! Host-side NVFP4 codec for typed tensor leaves.
//!
//! A `Tensor<NVFP4, RANK>` payload (triblespace-core `tensor::elements::NVFP4`)
//! is three planes in one blob: E2M1 codes, two per byte with the lower index
//! in the LOW nibble; one E4M3 scale per block of sixteen elements; and one
//! little-endian f32 global scale. Element `i` of the flat row-major tensor is
//! nibble `i % 2` of byte `i / 2` and belongs to block `i / 16`. Its value is
//!
//! ```text
//! E2M1[code & 7] * (code & 8 ? -1 : 1) * e4m3(scale[i / 16]) * scale2
//! ```
//!
//! `NVFP4::payload_len` sizes the payload and the Inkling importer's
//! `split_payload` reads the same offsets; the nibble order was settled against
//! `compressed_tensors` (see `models/inkling/nvfp4.rs`). That decoder lives
//! behind the `inkling-cuda` feature beside the kernels that consume it. This
//! module is the feature-free host form, so a packed leaf can be read by any
//! model loader (`Leaf::to_f32`) and written by a calibrated packer
//! (`calibrate`), on the same bytes.
//!
//! One global scale per TENSOR: that is what the encoding carries, so a packer
//! chooses block scales against it rather than against a per-row maximum. With
//! `scale2 = absmax / (6 * 448)` the largest block lands on E4M3's top code and
//! a block has to be smaller than `absmax / 229376` before its scale flushes
//! to zero.
//!
//! [`pack_nearest`] is the device quantizer's recipe operation for operation
//! (`nn::nvfp4_gemv`, the load-time quantizer Breeze and Voxtral ran before
//! their projections were stored packed): the block scale is
//! `e4m3_rn(min((bamax / 6) / scale2, 448))`, two correctly rounded divides in
//! that order, and a block whose scale flushes to zero codes every element as
//! `+0`. It must stay that way: a stored projection is gated against the
//! device's own codes, bit for bit, before it is published.

use anyhow::{Result, ensure};
use triblespace::core::blob::encodings::tensor::TensorElement;
use triblespace::core::blob::encodings::tensor::elements::{NVFP4, NVFP4_BLOCK};

/// Logical elements per block scale.
pub const BLOCK: usize = NVFP4_BLOCK;
/// The E2M1 magnitudes, indexed by the low three bits of a code.
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// Largest E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;
/// Largest finite E4M3 magnitude.
pub const E4M3_MAX: f32 = 448.0;
/// The E4M3 pattern for 448.
const E4M3_MAX_BYTE: u8 = 0x7E;

/// Decode one E4M3 byte (sign, four exponent bits biased by 7, three mantissa
/// bits; exponent zero is subnormal in steps of 2^-9; `0x7F`/`0xFF` are NaN).
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 7) as f32;
    let mag = if exp == 0 {
        man * 2f32.powi(-9)
    } else if exp == 15 && man == 7.0 {
        f32::NAN
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    sign * mag
}

/// The E4M3 pattern nearest a non-negative scale, ties to the even
/// significand, saturating at 448. A non-positive or NaN input is zero.
///
/// Arithmetic rather than a search over the 127 patterns: scaling by a power
/// of two is exact, so `v / 2^e * 8` is the significand in eighths exactly and
/// one `round_ties_even` is the conversion. The subnormal binade (exponent
/// field zero) is the same step, `2^-9`, as the first normal one.
pub fn e4m3_from_f32(v: f32) -> u8 {
    if !(v > 0.0) {
        return 0;
    }
    if v >= E4M3_MAX {
        return E4M3_MAX_BYTE;
    }
    // Exponent of the binade, floored at the smallest normal (2^-6): below it
    // the step stays 2^-9 and the exponent field is zero.
    let e = (((v.to_bits() >> 23) & 0xFF) as i32 - 127).max(-6);
    let steps = (v * 2f32.powi(3 - e)).round_ties_even() as u32;
    let byte = if e == -6 && steps < 8 {
        steps
    } else {
        // steps is 8..=16 in a normal binade; 16 carries into the next one.
        let (e, m) = if steps == 16 { (e + 1, 0) } else { (e, steps - 8) };
        (((e + 7) as u32) << 3) | m
    };
    (byte as u8).min(E4M3_MAX_BYTE)
}

/// The E2M1 code nearest a magnitude given in E2M1 units, ties to the even
/// code (the GB10's `cvt.rn.satfinite.e2m1x2` rule), saturating at 6.
pub fn e2m1_code(m: f32) -> u8 {
    // The seven midpoints between neighbouring magnitudes; an exact midpoint
    // goes to the even code, which is the upper neighbour exactly when that
    // neighbour's code is even.
    const MIDPOINTS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    if m.is_nan() {
        return 7;
    }
    let mut code = 0u8;
    for (i, &mid) in MIDPOINTS.iter().enumerate() {
        if m > mid || (m == mid && (i + 1) % 2 == 0) {
            code = i as u8 + 1;
        }
    }
    code
}

/// The signed code of `v` at `unit` per E2M1 step: `cvt.rn.satfinite` of
/// `v / unit`, which keeps the sign of a value that rounds to zero. A zero
/// unit codes every value as `+0`, as the device does when a block's scale
/// flushes (no `0 / 0` reaches its converter).
pub fn code_of(v: f32, unit: f32) -> u8 {
    if !(unit > 0.0) {
        return 0;
    }
    let sign = if v.is_sign_negative() { 8 } else { 0 };
    if v == 0.0 || v.is_nan() {
        return sign;
    }
    e2m1_code(v.abs() / unit) | sign
}

/// The signed E2M1 value of a code, in units.
pub fn decode_code(code: u8) -> f32 {
    let m = E2M1[(code & 7) as usize];
    if code & 8 != 0 { -m } else { m }
}

/// One f32 scale for a whole tensor: the largest block scale lands on E4M3's
/// top code. One, not infinity, for an all-zero tensor.
pub fn global_scale(w: &[f32]) -> f32 {
    let absmax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
    if absmax > 0.0 && absmax.is_finite() {
        absmax / (E2M1_MAX * E4M3_MAX)
    } else {
        1.0
    }
}

/// One block's E4M3 scale byte and the value of one E2M1 step under it.
pub fn block_unit(block: &[f32], scale2: f32) -> (u8, f32) {
    let bmax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
    let byte = e4m3_from_f32(((bmax / E2M1_MAX) / scale2).min(E4M3_MAX));
    (byte, e4m3_to_f32(byte) * scale2)
}

/// Split a payload for `elems` logical elements into its three planes.
pub fn split_payload(payload: &[u8], elems: usize) -> Result<(&[u8], &[u8], f32)> {
    let want = NVFP4::payload_len(elems);
    ensure!(
        payload.len() == want,
        "NVFP4 payload is {} bytes, {elems} elements need {want}",
        payload.len()
    );
    let codes_len = elems.div_ceil(2);
    let scales_len = elems.div_ceil(BLOCK);
    let codes = &payload[..codes_len];
    let scales = &payload[codes_len..codes_len + scales_len];
    let tail = &payload[codes_len + scales_len..];
    let scale2 = f32::from_le_bytes(tail.try_into().expect("four trailing bytes"));
    Ok((codes, scales, scale2))
}

/// Decode `elems` logical elements of a payload.
pub fn decode_payload(payload: &[u8], elems: usize) -> Result<Vec<f32>> {
    let (codes, scales, scale2) = split_payload(payload, elems)?;
    let mut out = vec![0f32; elems];
    for (blk, chunk) in out.chunks_mut(BLOCK).enumerate() {
        let unit = e4m3_to_f32(scales[blk]) * scale2;
        for (i, v) in chunk.iter_mut().enumerate() {
            let e = blk * BLOCK + i;
            let code = (codes[e / 2] >> (4 * (e % 2))) & 0xF;
            *v = decode_code(code) * unit;
        }
    }
    Ok(out)
}

/// A packed row-major `[rows, cols]` matrix, the three planes apart.
#[derive(Clone, Debug, PartialEq)]
pub struct Packed {
    pub rows: usize,
    pub cols: usize,
    /// `rows * cols / 2` bytes, low nibble first.
    pub codes: Vec<u8>,
    /// `rows * cols / 16` E4M3 bytes, one per block along `cols`.
    pub scales: Vec<u8>,
    /// The global scale.
    pub scale2: f32,
}

impl Packed {
    pub fn elems(&self) -> usize {
        self.rows * self.cols
    }

    /// The payload a `Tensor<NVFP4, 2>` leaf carries.
    pub fn payload(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(NVFP4::payload_len(self.elems()));
        payload.extend_from_slice(&self.codes);
        payload.extend_from_slice(&self.scales);
        payload.extend_from_slice(&self.scale2.to_le_bytes());
        payload
    }

    /// The inverse of [`Packed::payload`].
    pub fn from_payload(payload: &[u8], rows: usize, cols: usize) -> Result<Self> {
        ensure!(cols % BLOCK == 0, "cols {cols} is not a multiple of {BLOCK}");
        let (codes, scales, scale2) = split_payload(payload, rows * cols)?;
        Ok(Self {
            rows,
            cols,
            codes: codes.to_vec(),
            scales: scales.to_vec(),
            scale2,
        })
    }

    /// The dense values the codes stand for.
    pub fn decode(&self) -> Vec<f32> {
        decode_payload(&self.payload(), self.elems()).expect("a Packed is sized by construction")
    }
}

/// A projection packed for storage as a `Tensor<NVFP4, 2>` leaf.
#[derive(Clone, Debug)]
pub struct StoredProjection {
    /// The tensor name the leaf is stored under.
    pub name: String,
    /// `[out, in]`.
    pub shape: [u64; 2],
    /// Codes, block scales and global scale (see [`Packed::payload`]).
    pub payload: Vec<u8>,
}

/// How a projection stored as NVFP4 multiplies an activation taller than the
/// GEMV's eight rows (the prompt pass). Both read the stored four-bit planes;
/// neither keeps a 16-bit copy of the weight (`nn::nvfp4_gemm`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PromptLane {
    /// Activations quantized per 16 to E2M1 with E4M3 scales, both operands
    /// into the block-scaled four-bit tensor-core MMA.
    W4a4,
    /// Activations stay 16-bit; weights are decoded to 16 bits in registers
    /// for the 16-bit MMA.
    #[default]
    W4a16,
}

impl PromptLane {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "w4a4" => Ok(Self::W4a4),
            "w4a16" => Ok(Self::W4a16),
            other => anyhow::bail!("NVFP4 prompt lane must be w4a4 or w4a16, not {other:?}"),
        }
    }
}

/// Round-to-nearest NVFP4 of a row-major `[rows, cols]` matrix under one
/// global scale, block scales from each block's maximum.
pub fn pack_nearest(w: &[f32], cols: usize) -> Result<Packed> {
    ensure!(cols > 0 && cols % BLOCK == 0, "cols {cols} is not a positive multiple of {BLOCK}");
    ensure!(w.len() % cols == 0, "{} values do not fill rows of {cols}", w.len());
    let rows = w.len() / cols;
    let scale2 = global_scale(w);
    let mut codes = vec![0u8; w.len() / 2];
    let mut scales = vec![0u8; w.len() / BLOCK];
    // Blocks are independent once the global scale is known, so the planes
    // are filled in parallel row bands. A projection is hundreds of millions
    // of elements; one thread took minutes per model.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    let band = rows.div_ceil(threads).max(1) * cols;
    std::thread::scope(|scope| {
        for ((w, codes), scales) in w
            .chunks(band)
            .zip(codes.chunks_mut(band / 2))
            .zip(scales.chunks_mut(band / BLOCK))
        {
            scope.spawn(move || {
                for (blk, block) in w.chunks(BLOCK).enumerate() {
                    let (byte, unit) = block_unit(block, scale2);
                    scales[blk] = byte;
                    for (i, pair) in block.chunks(2).enumerate() {
                        codes[blk * BLOCK / 2 + i] =
                            code_of(pair[0], unit) | code_of(pair[1], unit) << 4;
                    }
                }
            });
        }
    });
    Ok(Packed {
        rows,
        cols,
        codes,
        scales,
        scale2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The search this module used before the arithmetic form: nearest
    /// pattern, ties to the even one.
    fn e4m3_search(v: f32) -> u8 {
        if !(v > 0.0) {
            return 0;
        }
        if v >= E4M3_MAX {
            return E4M3_MAX_BYTE;
        }
        let mut best = 0u8;
        let mut best_err = f64::INFINITY;
        for b in 0..=E4M3_MAX_BYTE {
            let err = (e4m3_to_f32(b) as f64 - v as f64).abs();
            if err < best_err || (err == best_err && b & 1 == 0) {
                best = b;
                best_err = err;
            }
        }
        best
    }

    #[test]
    fn arithmetic_e4m3_is_the_nearest_pattern_everywhere() {
        // Every pattern, every midpoint and its two f32 neighbours, and a
        // dense sweep across the whole range including the subnormals.
        let mut probes = vec![0.0f32, 1e-30, 2f32.powi(-10), 2f32.powi(-9), 447.9, 448.0, 1e9];
        for b in 0..=E4M3_MAX_BYTE {
            let lo = e4m3_to_f32(b);
            probes.push(lo);
            if b < E4M3_MAX_BYTE {
                let mid = ((lo as f64 + e4m3_to_f32(b + 1) as f64) / 2.0) as f32;
                probes.extend([mid, f32::from_bits(mid.to_bits() - 1), f32::from_bits(mid.to_bits() + 1)]);
            }
        }
        let mut v = 2f32.powi(-12);
        while v < 460.0 {
            probes.push(v);
            v *= 1.0007;
        }
        for v in probes {
            assert_eq!(e4m3_from_f32(v), e4m3_search(v), "{v:e}");
        }
    }

    #[test]
    fn every_finite_e4m3_pattern_round_trips() {
        for b in 1..=E4M3_MAX_BYTE {
            assert_eq!(e4m3_from_f32(e4m3_to_f32(b)), b, "pattern {b:#04x}");
        }
        assert_eq!(e4m3_from_f32(0.0), 0);
        assert_eq!(e4m3_to_f32(E4M3_MAX_BYTE), 448.0);
    }

    #[test]
    fn e4m3_saturates_and_ties_to_even() {
        assert_eq!(e4m3_from_f32(1000.0), E4M3_MAX_BYTE);
        // 1.0 is 0x38, 1.125 is 0x39, 1.25 is 0x3A.
        assert_eq!(e4m3_from_f32(1.0625), 0x38, "tie rounds to the even pattern");
        assert_eq!(e4m3_from_f32(1.1875), 0x3A, "tie rounds to the even pattern");
        assert_eq!(e4m3_from_f32(1.1), 0x39);
    }

    #[test]
    fn e2m1_ties_go_to_the_even_code_and_saturate() {
        assert_eq!(e2m1_code(0.25), 0);
        assert_eq!(e2m1_code(1.25), 2);
        assert_eq!(e2m1_code(2.5), 4);
        assert_eq!(e2m1_code(5.0), 6);
        assert_eq!(e2m1_code(0.75), 2);
        assert_eq!(e2m1_code(9.0), 7);
        assert_eq!(code_of(-3.1, 1.0), 8 | 5);
        assert_eq!(code_of(-0.0, 1.0), 8);
    }

    fn sample(rows: usize, cols: usize) -> Vec<f32> {
        // Deterministic, sign-mixed, with a spread of magnitudes across rows.
        (0..rows * cols)
            .map(|i| {
                let r = (i / cols) as f32 + 1.0;
                let x = ((i * 7919) % 1000) as f32 / 1000.0 - 0.5;
                x * r * 0.37
            })
            .collect()
    }

    #[test]
    fn pack_then_decode_is_the_reference_arithmetic() {
        let (rows, cols) = (5, 48);
        let w = sample(rows, cols);
        let p = pack_nearest(&w, cols).expect("packs");
        let dec = p.decode();
        let scale2 = global_scale(&w);
        for (blk, block) in w.chunks(BLOCK).enumerate() {
            let (byte, unit) = block_unit(block, scale2);
            assert_eq!(p.scales[blk], byte);
            for (i, &v) in block.iter().enumerate() {
                let want = decode_code(code_of(v, unit)) * unit;
                assert_eq!(dec[blk * BLOCK + i], want);
                // The E2M1 ladder is not uniform: its widest step is 2 units
                // (4 to 6), so nearest rounding is within one unit.
                assert!((dec[blk * BLOCK + i] - v).abs() <= unit + 1e-6 || v.abs() > 6.0 * unit);
            }
        }
    }

    #[test]
    fn payload_round_trips_and_the_flat_decoder_agrees() {
        let (rows, cols) = (3, 64);
        let w = sample(rows, cols);
        let p = pack_nearest(&w, cols).expect("packs");
        let payload = p.payload();
        assert_eq!(payload.len(), NVFP4::payload_len(rows * cols));
        let back = Packed::from_payload(&payload, rows, cols).expect("splits");
        assert_eq!(back, p);
        assert_eq!(decode_payload(&payload, rows * cols).expect("decodes"), p.decode());
    }

    #[test]
    fn the_lower_index_is_the_low_nibble() {
        let mut block = vec![0f32; BLOCK];
        block[0] = 6.0;
        block[1] = -1.0;
        let p = pack_nearest(&block, BLOCK).expect("packs");
        // unit = 6.0 / 6 = 1: element 0 is code 7, element 1 is code 8|2.
        assert_eq!(p.codes[0], 0x07 | (0x0A << 4));
        assert_eq!(p.decode()[0], 6.0);
        assert_eq!(p.decode()[1], -1.0);
    }

    #[test]
    fn a_zero_block_and_a_zero_tensor_decode_to_zero() {
        let p = pack_nearest(&vec![0f32; 32], 32).expect("packs");
        assert_eq!(p.scale2, 1.0);
        assert!(p.decode().iter().all(|&v| v == 0.0));
    }
}
