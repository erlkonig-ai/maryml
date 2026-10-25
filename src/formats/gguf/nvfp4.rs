//! GGML NVFP4 (type 40), NOT Mary's typed-leaf byte layout.
//!
//! Matches llama.cpp 1ddffe9f70a63e2d260d4bfad9ad200009007f62:
//! ggml-common.h block_nvfp4, ggml-quants.c dequantize_row_nvfp4 and
//! ggml-impl.h ggml_ue4m3_to_fp32. A 36-byte block contains four UE4M3
//! scale bytes followed by four eight-byte code groups. Within a group the
//! low nibbles precede all high nibbles. This only decodes the stored blocks;
//! separate scale_2 tensors/matmul scaling are NOT applied here.

use anyhow::{Result, ensure};

/// Import-only, scale-preserving conversion to the existing typed tensor ABI.
pub mod typed;

pub const ELEMENTS: usize = 64;
pub const BYTES: usize = 36;
const VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Pinned GGML UE4M3 behavior over all byte patterns. In particular 0x7f is
/// zero, not NaN, and the high bit is ignored except in that special case.
/// Do not substitute the signed E4M3 decoder used by typed Mary leaves.
fn scale(byte: u8) -> f32 {
    if byte == 0 || byte == 0x7f {
        return 0.0;
    }
    let exp = (byte >> 3) & 15;
    let man = byte & 7;
    if exp == 0 {
        f32::from(man) * 2f32.powi(-9)
    } else {
        (1.0 + f32::from(man) / 8.0) * 2f32.powi(i32::from(exp) - 7)
    }
}

pub fn decode(raw: &[u8], elements: usize) -> Result<Vec<f32>> {
    ensure!(elements % ELEMENTS == 0, "NVFP4 elements must be a multiple of 64");
    let bytes = (elements / ELEMENTS).checked_mul(BYTES)
        .ok_or_else(|| anyhow::anyhow!("NVFP4 byte length overflow"))?;
    ensure!(raw.len() == bytes, "NVFP4 byte length mismatch: {} != {bytes}", raw.len());
    let mut out = Vec::new();
    out.try_reserve_exact(elements)?;
    out.resize(elements, 0.0);
    for (block, dst) in raw.chunks_exact(BYTES).zip(out.chunks_exact_mut(ELEMENTS)) {
        for group in 0..4 {
            let d = scale(block[group]);
            for j in 0..8 {
                let code = block[4 + group * 8 + j];
                dst[group * 16 + j] = VALUES[(code & 15) as usize] * d;
                dst[group * 16 + j + 8] = VALUES[(code >> 4) as usize] * d;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_size_and_row_block_contract() {
        assert!(decode(&[0; 36], 63).is_err());
        assert!(decode(&[0; 35], 64).is_err());
        assert!(decode(&[0; 37], 64).is_err());
        assert!(decode(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn halves_are_not_adjacent_nibbles() {
        let mut block = [0x21; 36];
        block[..4].fill(0x38); // scale 1
        let decoded = decode(&block, 64).unwrap();
        assert_eq!(&decoded[..8], &[0.5; 8]);
        assert_eq!(&decoded[8..16], &[1.0; 8]);
    }
}
