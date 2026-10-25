//! GGML type40 -> existing Tensor<NVFP4, RANK>, without re-quantization.
//!
//! Codes are reordered, not transposed: GGML's eight low then eight high
//! nibbles become adjacent low-first nibbles. UE4M3's ignored high bit is
//! removed; its special0x7f becomes scale0. GGML code8 means positive zero,
//! so it becomes typed code0 (typed code8 would instead mean negative zero).
//! Negative nonzero codes multiplied by zero retain negative zero in both.
//!
//! Global scale is exactly1. Flint's separate projection/matmul scales MUST
//! remain separate model parameters: folding them into this payload changes
//! arithmetic boundaries. This is an import conversion, not a runtime loader.
//!
//! Source scale0xff means480, outside signed E4M3's finite scale range448.
//! This initial scale-preserving converter rejects it for every code pattern,
//! including all-zero groups that could be represented another way. That is
//! a deliberate domain limit, NOT proof all rejected groups are impossible.
//! Some are genuinely impossible at global1: code7*480=2880 exceeds6*448.
//! Never substitute saturation/nearest rounding or silently choose a global
//! scale. Check actual checkpoint scale bytes before choosing a new dtype.

use anyhow::{Context,Result,ensure};
use anybytes::Bytes;
use triblespace::core::blob::{Blob,encodings::tensor::{
    Tensor,MAX_RANK,TENSOR_HEADER_LEN,elements::NVFP4,tensor_blob,
}};

use super::{BYTES,ELEMENTS};

fn code(nibble:u8)->u8 { if nibble==8 {0} else {nibble} }

/// Convert already-selected complete GGML64-element blocks into the native
/// codes/scales/global-scale payload. Rejection names the offending block/group.
/// Returns nothing on failure; no partially converted tensor is published.
pub fn payload(raw:&[u8],elements:usize)->Result<Vec<u8>> {
    ensure!(elements%ELEMENTS==0,"GGML NVFP4 elements must be a multiple of64");
    let expected=(elements/ELEMENTS).checked_mul(BYTES).context("source length overflow")?;
    ensure!(raw.len()==expected,"GGML NVFP4 source length {} != {expected}",raw.len());
    let codes=elements/2;
    let len=codes.checked_add(elements/16).and_then(|n|n.checked_add(4))
        .context("typed NVFP4 payload length overflow")?;
    // Also admit the core tensor_blob's header+payload allocation before calling it.
    let blob_len=len.checked_add(TENSOR_HEADER_LEN).context("typed tensor length overflow")?;
    ensure!(blob_len<=isize::MAX as usize,"typed tensor exceeds addressable allocation");
    let mut out=Vec::new();
    out.try_reserve_exact(len).context("allocate typed NVFP4 payload")?;
    out.resize(len,0);
    for (block_index,block) in raw.chunks_exact(BYTES).enumerate() {
        for group in 0..4 {
            let source_scale=block[group];
            ensure!(source_scale!=0xff,
                "unsupported GGML NVFP4 scale0xff (480), block{block_index} group{group}: scale-preserving typed E4M3 conversion requires <=448; no re-quantization performed");
            out[codes+block_index*4+group]=if source_scale==0x7f {0} else {source_scale&0x7f};
            let source=&block[4+group*8..4+(group+1)*8];
            let target=block_index*32+group*8;
            for pair in 0..4 {
                let a=source[pair*2]; let b=source[pair*2+1];
                out[target+pair]=code(a&15)|(code(b&15)<<4);
                out[target+4+pair]=code(a>>4)|(code(b>>4)<<4);
            }
        }
    }
    out[len-4..].copy_from_slice(&1.0f32.to_le_bytes());
    Ok(out)
}

/// Build the existing self-describing Tensor leaf, without any pile writes or
/// new schema. Dims are LOGICAL ROW-MAJOR (reverse GGML fastest-first dims).
/// The last dimension is a complete positive multiple of64; projection scales
/// are not arguments because they do not belong in this conversion.
pub fn to_tensor<const RANK:usize>(raw:&[u8],dims:[u64;RANK])->Result<Blob<Tensor<NVFP4,RANK>>> {
    ensure!(RANK>0 && RANK<=MAX_RANK,"NVFP4 tensor rank outside1..={MAX_RANK}");
    ensure!(dims.iter().all(|&d|d>0),"NVFP4 tensor dimensions must be positive");
    let cols=*dims.last().context("NVFP4 requires a row dimension")?;
    ensure!(cols%ELEMENTS as u64==0,"NVFP4 last dimension must be a multiple of64");
    let elements=dims.iter().try_fold(1u64,|n,&d|n.checked_mul(d).context("tensor element count overflow"))?;
    let elements=usize::try_from(elements).context("tensor element count exceeds usize")?;
    let converted=payload(raw,elements)?;
    tensor_blob::<NVFP4,RANK>(dims,Bytes::from_source(converted)).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::blob::{TryFromBlob,encodings::tensor::TensorView};

    #[test]
    fn every_scale_byte_and_packed_code_byte_is_exact_or_explicitly_rejected()->Result<()> {
        // 256 scale patterns x256 packed-code patterns; every finite GGML
        // scale/code combination, including both signs and signed-zero edges.
        for scale in 0..=255u8 {
            for packed in 0..=255u8 {
                let mut raw=[packed;BYTES]; raw[..4].fill(scale);
                let reference=super::super::decode(&raw,ELEMENTS)?;
                assert!(reference.iter().all(|v|v.is_finite()));
                let converted=payload(&raw,ELEMENTS);
                if scale==0xff {
                    assert!(converted.unwrap_err().to_string().contains("scale0xff"));
                    continue;
                }
                let converted=converted?;
                let (_,scales,global)=crate::nvfp4::split_payload(&converted,ELEMENTS)?;
                assert!(scales.iter().all(|&b|b<=0x7e));
                assert_eq!(global.to_bits(),1.0f32.to_bits());
                let decoded=crate::nvfp4::decode_payload(&converted,ELEMENTS)?;
                for (i,(&actual,&expected)) in decoded.iter().zip(&reference).enumerate() {
                    assert_eq!(actual.to_bits(),expected.to_bits(),"scale{scale:#04x} packed{packed:#04x} element{i}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn varied_positions_rows_and_typed_header_keep_logical_shape()->Result<()> {
        let mut raw=vec![0u8;BYTES*4];
        for (block_index,block) in raw.chunks_exact_mut(BYTES).enumerate() {
            for group in 0..4 { block[group]=[0x38,0xb9,0x7f,0x80][(block_index+group)%4]; }
            for (j,byte) in block[4..].iter_mut().enumerate() {
                *byte=((j+block_index*3)%16) as u8|((((j*7+block_index)%16) as u8)<<4);
            }
        }
        let blob=to_tensor(&raw,[2,128])?;
        let view:TensorView=blob.try_from_blob()?;
        assert_eq!(view.dims(),[2,128]);
        assert_eq!(view.payload().len(),128+16+4);
        let a=crate::nvfp4::decode_payload(view.payload(),256)?;
        let b=super::super::decode(&raw,256)?;
        assert_eq!(a.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),b.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        // No transpose: a GGML[64,2,2] expert stack becomes row-major[2,2,64].
        let view:TensorView=to_tensor(&raw,[2,2,64])?.try_from_blob()?;
        assert_eq!(view.dims(),[2,2,64]);
        Ok(())
    }

    #[test]
    fn zero_canonicalization_is_explicit()->Result<()> {
        let mut raw=[0x98;BYTES]; raw[..4].copy_from_slice(&[0x38,0x7f,0x80,0xb8]);
        let converted=payload(&raw,64)?;
        let (codes,scales,global)=crate::nvfp4::split_payload(&converted,64)?;
        assert_eq!(scales,[0x38,0,0,0x38]); assert_eq!(global,1.0);
        assert_eq!(codes[0],0); // source code8 is+0, not typed-0
        assert_eq!(codes[4],0x99);
        let decoded=crate::nvfp4::decode_payload(&converted,64)?;
        assert_eq!(decoded[0].to_bits(),0.0f32.to_bits());
        assert_eq!(decoded[24].to_bits(),(-0.0f32).to_bits());
        Ok(())
    }

    #[test]
    fn unsupported480_is_not_silently_requantized()->Result<()> {
        let mut raw=[0x77;BYTES]; raw[..4].fill(0xff);
        assert_eq!(super::super::decode(&raw,64)?[0],2880.0);
        assert!(2880.0>crate::nvfp4::E4M3_MAX*crate::nvfp4::E2M1_MAX);
        assert!(payload(&raw,64).is_err());
        // Policy refusal is deliberately broader than mathematical impossibility:
        // these zero values COULD be represented, but480 is not silently rescaled.
        raw[4..].fill(0);
        assert_eq!(super::super::decode(&raw,64)?[0],0.0);
        assert!(payload(&raw,64).is_err());
        Ok(())
    }

    #[test]
    fn geometry_and_overflow_refusals_precede_allocation() {
        assert!(payload(&[0;BYTES],63).is_err());
        assert!(payload(&[0;BYTES-1],64).is_err());
        assert!(payload(&[0;BYTES+1],64).is_err());
        assert!(payload(&[],usize::MAX-63).is_err());
        assert!(to_tensor(&[0;BYTES],[0,64]).is_err());
        assert!(to_tensor(&[0;BYTES],[1,63]).is_err());
        assert!(to_tensor(&[0;BYTES],[u64::MAX,64]).is_err());
        assert!(to_tensor(&[],[]).is_err());
        assert!(to_tensor(&[],[64;MAX_RANK+1]).is_err());
        assert_eq!(payload(&[],0).unwrap(),1.0f32.to_le_bytes());
    }
}
