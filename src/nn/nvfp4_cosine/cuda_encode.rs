//! Direct GPU BF16 [1,4096] to the existing canonical row recipe.
//!
//! This deliberately uses fixed serial-per-row binary64 arithmetic. No host
//! normalization, quantization or certificate calculation occurs. Only the
//! completed row bytes and a status word are read back. The source-defined
//! CUDA RN intrinsics prevent contraction/reassociation of the recipe.
//! This API neither chooses a model nor creates collection/schema identities.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{client::ComputeClient, cuda::{CudaDevice, CudaRuntime}, prelude::*};
use crate::nn::raw_cuda::{RawArgs, RawCudaKernel};
use super::{Error, QuantizedRow, QuantizedStage};

const DIMENSION: usize = 4096;
const STAGE_BYTES: usize = 2308;
const ROW_BYTES: usize = STAGE_BYTES * 2 + 8;

/// A device-bound encoder. It retains no row, model, or scratch across calls.
pub struct CudaRowEncoder4096 {
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    kernel: RawCudaKernel,
}

impl CudaRowEncoder4096 {
    pub fn new(device: CudaDevice) -> Result<Self, Error> {
        if device.index > u16::MAX as usize {
            return Err(Error::new("CUDA ordinal exceeds device identity domain"));
        }
        Ok(Self {
            client: CudaRuntime::client(&device), device,
            kernel: RawCudaKernel::new("encode_nvfp4_4096",
                include_str!("cuda_encode.cu"), CubeDim::new_1d(1)),
        })
    }

    fn check(&self, input: &CubeTensor<CudaRuntime>) -> Result<(), Error> {
        let start = input.handle.offset_start.unwrap_or(0);
        let end = input.handle.offset_end.unwrap_or(0);
        let extent = input.handle.size().checked_sub(start).and_then(|n| n.checked_sub(end));
        if input.device != self.device
            || !std::ptr::eq(input.client.properties(), self.client.properties()) {
            return Err(Error::new("NVFP4 input has a different CUDA device/client"));
        }
        if input.dtype != DType::BF16 || input.qparams.is_some()
            || input.meta.shape().as_slice() != [1, DIMENSION]
            || input.meta.strides().len() != 2 || input.meta.strides()[1] != 1
            || start % 2 != 0 || extent.is_none_or(|n| n < (DIMENSION * 2) as u64) {
            return Err(Error::new("NVFP4 encoder requires contiguous BF16 [1,4096] storage"));
        }
        Ok(())
    }

    /// The caller follows the input tensor's valid-handle/stream-ordering
    /// contract; custom unsafe stream producers must establish dependencies.
    /// The input is borrowed unchanged. Device/driver failures retain the
    /// underlying CubeCL error/panic behavior, never a CPU fallback.
    pub fn encode(&self, input: &CubeTensor<CudaRuntime>) -> Result<QuantizedRow, Error> {
        self.check(input)?;
        let scratch = self.client.empty(DIMENSION * 4 * 8);
        let output = self.client.empty(ROW_BYTES);
        let status = self.client.empty(4);
        // SAFETY: exact geometry checked above, all output extents are fixed,
        // one block/one invocation, and the kernel has finite bounded loops.
        unsafe { self.kernel.launch(&self.client, CubeCount::Static(1, 1, 1),
            RawArgs::new().buffer(&input.handle).buffer(&scratch)
                .buffer(&output).buffer(&status)); }
        let result = self.client.read_one(status).map_err(|e| Error::new(format!("CUDA status: {e:?}")))?;
        let status: &[u8] = result.as_ref();
        if status.len() != 4 { return Err(Error::new("CUDA status has wrong extent")); }
        match u32::from_le_bytes(status.try_into().expect("four checked bytes")) {
            0 => {},
            1 => return Err(Error::new("embedding coordinates must all be finite")),
            2 => return Err(Error::new("embedding produced an invalid NVFP4 global scale")),
            other => return Err(Error::new(format!("unexpected CUDA row status {other}"))),
        }
        let bytes = self.client.read_one(output).map_err(|e| Error::new(format!("CUDA row: {e:?}")))?;
        let bytes: &[u8] = bytes.as_ref();
        if bytes.len() != ROW_BYTES { return Err(Error::new("CUDA row has wrong extent")); }
        Ok(QuantizedRow {
            stages: std::array::from_fn(|i| {
                let stage = &bytes[i * STAGE_BYTES..(i + 1) * STAGE_BYTES];
                QuantizedStage { global: stage[..4].try_into().expect("four bytes"),
                    block_scales: stage[4..260].to_vec(), codes: stage[260..].to_vec() }
            }),
            norm: bytes[4616..4620].try_into().expect("four bytes"),
            error: bytes[4620..4624].try_into().expect("four bytes"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn row_geometry_and_explicit_rounding_are_pinned() {
        assert_eq!(STAGE_BYTES, 4 + 4096 / 16 + 4096 / 2);
        assert_eq!(ROW_BYTES, 4624);
        let source = include_str!("cuda_encode.cu");
        for operation in ["__dadd_rn", "__dsub_rn", "__dmul_rn", "__ddiv_rn", "__dsqrt_rn", "__fmaf_rn"] {
            assert!(source.contains(operation));
        }
        assert!(!source.contains("atomic"));
    }

    fn ptx_contract(root: &std::path::Path) {
        #[derive(serde::Deserialize)]
        struct Entry { value: Ptx }
        #[derive(serde::Deserialize)]
        struct Ptx { entrypoint_name: String, ptx: Vec<i8> }
        let mut pending = vec![root.to_path_buf()];
        let mut found = false;
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() { pending.push(path); continue; }
                if !path.file_name().is_some_and(|n| n == "chunk0.cbor") { continue; }
                let bytes = std::fs::read(path).unwrap();
                let mut cursor = std::io::Cursor::new(bytes);
                while cursor.position() < cursor.get_ref().len() as u64 {
                    let entry: Entry = ciborium::from_reader(&mut cursor).unwrap();
                    if entry.value.entrypoint_name != "encode_nvfp4_4096" { continue; }
                    let text = String::from_utf8(entry.value.ptx.into_iter().map(|b| b as u8).take_while(|&b| b != 0).collect()).unwrap();
                    for forbidden in [".ftz", ".approx", "fma.rn.f64", "mad.rn.f64"] {
                        assert!(!text.contains(forbidden), "forbidden PTX arithmetic {forbidden}");
                    }
                    for required in ["add.rn.f64", "mul.rn.f64", "div.rn.f64", "sqrt.rn.f64", "fma.rn.f32"] {
                        assert!(text.contains(required), "missing PTX arithmetic {required}");
                    }
                    println!("PTX contract passed; cache retained at {}", root.display());
                    found = true;
                }
            }
        }
        assert!(found, "encoder PTX must be retained");
    }

    fn tensor(encoder: &CudaRowEncoder4096, bits: &[u16]) -> CubeTensor<CudaRuntime> {
        let bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        CubeTensor::new_contiguous(encoder.client.clone(), encoder.device.clone(),
            [1, bits.len()].as_slice().into(), encoder.client.create_from_slice(&bytes), DType::BF16)
    }

    fn compare(encoder: &CudaRowEncoder4096, bits: &[u16], label: &str) {
        let input = tensor(encoder, bits);
        let before = encoder.client.read_one(input.handle.clone()).unwrap().to_vec();
        // CPU numerical arithmetic exists only in this independent test oracle.
        let values: Vec<f32> = bits.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let expected = QuantizedRow::quantize(&values, DIMENSION).unwrap();
        let actual = encoder.encode(&input).unwrap();
        assert_eq!(actual, expected, "canonical bytes: {label}");
        assert_eq!(encoder.encode(&input).unwrap(), actual, "repeat: {label}");
        assert_eq!(encoder.client.read_one(input.handle.clone()).unwrap().to_vec(), before);
        let transformed = super::super::rotate(&super::super::normalize(&values, DIMENSION).unwrap()).unwrap();
        let canonical = actual.decode_f64();
        let decoded32: Vec<f64> = actual.decode_f32().into_iter().map(f64::from).collect();
        assert!(super::super::outward_l2(&transformed, &canonical).unwrap() <= f64::from(actual.error_bound()));
        assert!(super::super::outward_l2(&canonical, &decoded32).unwrap() <= f64::from(actual.error_bound()));
    }

    #[test]
    #[ignore = "requires exclusively reserved CUDA device"]
    fn gpu_canonical_rows_match_cpu_oracle_and_reject_invalid_inputs() {
        use cubecl::config::{RuntimeConfig, cache::CacheConfig};
        let cache = std::path::PathBuf::from(std::env::var("WEMM_NVFP4_PTX_CACHE").expect("explicit fresh cache directory"));
        assert!(!cache.exists(), "retain old PTX evidence");
        let mut config = cubecl::config::CubeClRuntimeConfig::default();
        config.compilation.cache = Some(CacheConfig::File(cache.clone()));
        cubecl::config::CubeClRuntimeConfig::set(config);
        let encoder = CudaRowEncoder4096::new(CudaDevice { index: 0 }).unwrap();
        compare(&encoder, &[0; DIMENSION], "zero");
        compare(&encoder, &[0x8000; DIMENSION], "negative zero");
        compare(&encoder, &[1; DIMENSION], "all minimum BF16 subnormal");
        let mut spike = vec![0; DIMENSION]; spike[17] = 0x3f80;
        compare(&encoder, &spike, "spike");
        compare(&encoder, &(0..DIMENSION).map(|i| [1, 0x8001, 0x7f7f, 0xff7f, 0x3f80, 0xbf80, 0x3f00, 0x3fc0][i % 8]).collect::<Vec<_>>(), "range");
        let pattern: Vec<_> = (0..DIMENSION).map(|i| ((i * 4051 + 37) & 0xffff) as u16)
            .map(|b| if b & 0x7f80 == 0x7f80 { 0 } else { b }).collect();
        compare(&encoder, &pattern, "finite bit pattern");
        for invalid in [0x7f80, 0xff80, 0x7fc1] {
            let mut bits = vec![0; DIMENSION]; bits[23] = invalid;
            assert!(encoder.encode(&tensor(&encoder, &bits)).unwrap_err().to_string().contains("finite"));
        }
        assert!(encoder.encode(&tensor(&encoder, &[0; 3])).is_err());
        let mut wrong = tensor(&encoder, &[0; DIMENSION]); wrong.dtype = DType::F16;
        assert!(encoder.encode(&wrong).is_err());
        let mut short = tensor(&encoder, &[0; DIMENSION]); short.handle = encoder.client.empty(2);
        assert!(encoder.encode(&short).is_err());
        let mut strided = tensor(&encoder, &[0; DIMENSION]); strided.meta.strides_mut()[1] = 2;
        assert!(encoder.encode(&strided).is_err());
        let mut wrong_device = tensor(&encoder, &[0; DIMENSION]); wrong_device.device.index = 1;
        assert!(encoder.encode(&wrong_device).is_err());
        let mut offset = tensor(&encoder, &[0; DIMENSION + 1]);
        offset.meta = tensor(&encoder, &[0; DIMENSION]).meta;
        offset.handle = offset.handle.offset_start(2);
        assert_eq!(encoder.encode(&offset).unwrap(), QuantizedRow::quantize(&[0.0; DIMENSION], DIMENSION).unwrap());
        let mut odd = tensor(&encoder, &[0; DIMENSION + 1]);
        odd.meta = tensor(&encoder, &[0; DIMENSION]).meta;
        odd.handle = odd.handle.offset_start(1);
        assert!(encoder.encode(&odd).is_err());
        let mut outside = tensor(&encoder, &[0; DIMENSION]);
        outside.handle.offset_start = Some(u64::MAX);
        assert!(encoder.encode(&outside).is_err());
        ptx_contract(&cache);
    }

    #[test]
    #[ignore = "requires reserved CUDA and retained WeMM evidence path"]
    fn gpu_canonical_rows_match_retained_wemm_embeddings() {
        let path = std::env::var("WEMM_NVFP4_ORACLE_JSON").expect("explicit retained evidence path");
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let rows = value["embeddings"].as_array().expect("embeddings");
        assert!(!rows.is_empty());
        let encoder = CudaRowEncoder4096::new(CudaDevice { index: 0 }).unwrap();
        for (i, row) in rows.iter().enumerate() {
            let bits: Vec<u16> = row["bits"].as_array().unwrap().iter().map(|x| u16::try_from(x.as_u64().unwrap()).unwrap()).collect();
            compare(&encoder, &bits, &format!("retained row {i}"));
        }
    }
}
