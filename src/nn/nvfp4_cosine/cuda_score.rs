//! GPU preparation and canonical reconstructed cosine, not certified bounds.
//!
//! The query is BF16 [1,4096] on the existing CubeCL CUDA device. Its F64
//! normalization and rotation never leave that device. Segments upload the
//! existing immutable byte planes once and own their device backing; a borrowed
//! `ScanSegment` is not claimed to carry mmap provenance. Only statuses and
//! final F64 row scores are read back. Rows retain their physical input order:
//! deduplication, handle association and maximum-per-handle are storage policy.
//!
//! Arithmetic is the canonical eight-lane `raw_dot_f64` recipe, not the
//! 32-lane `CudaUpperScanner`. No certificates, persisted bytes or identities
//! change. Preparation validates a query even when a later score has no rows;
//! an index's empty-index shortcut remains its caller's responsibility.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{client::ComputeClient, cuda::{CudaDevice, CudaRuntime}, prelude::*, server::Handle};
use crate::nn::raw_cuda::{RawArgs, RawCudaKernel};
use super::{Error, ScanSegment};

const DIMENSION: usize = 4096;
const SOURCE: &str = concat!(include_str!("cuda_encode.cu"), "\n", include_str!("cuda_score.cu"));

/// Owned, already validated GPU query coordinates. No host embedding copy.
pub struct CudaPreparedQuery4096 {
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    coordinates: Handle,
}

struct Stage { globals: Handle, scales: Handle, codes: Handle }

/// Owned immutable GPU byte planes. No borrowed source survives `upload`.
pub struct CudaScanSegment4096 {
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    rows: usize,
    stages: [Stage; 2],
    norms: Handle,
    errors: Handle,
}

/// Device-bound arithmetic; callers retain returned query/segment owners.
pub struct CudaReconstructedScorer4096 {
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    prepare_kernel: RawCudaKernel,
    score_kernel: RawCudaKernel,
}

impl CudaReconstructedScorer4096 {
    pub fn new(device: CudaDevice) -> Result<Self, Error> {
        if device.index > u16::MAX as usize {
            return Err(Error::new("CUDA ordinal exceeds device identity domain"));
        }
        Ok(Self { client: CudaRuntime::client(&device), device,
            prepare_kernel: RawCudaKernel::new("prepare_query_4096", SOURCE, CubeDim::new_1d(1)),
            score_kernel: RawCudaKernel::new("reconstructed_cosines_4096", SOURCE, CubeDim::new_1d(1)) })
    }

    fn check_device(&self, client: &ComputeClient<CudaRuntime>, device: &CudaDevice) -> Result<(), Error> {
        if device != &self.device || !std::ptr::eq(client.properties(), self.client.properties()) {
            return Err(Error::new("NVFP4 object has a different CUDA device/client"));
        }
        Ok(())
    }

    /// Borrow the existing GPU tensor unchanged. Unsafe custom stream producers
    /// must satisfy CubeTensor's valid-handle and stream-dependency contract.
    /// Runtime/device faults keep CubeCL's error/panic behavior, never fallback.
    pub fn prepare(&self, input: &CubeTensor<CudaRuntime>) -> Result<CudaPreparedQuery4096, Error> {
        self.check_device(&input.client, &input.device)?;
        let start = input.handle.offset_start.unwrap_or(0);
        let end = input.handle.offset_end.unwrap_or(0);
        let extent = input.handle.size().checked_sub(start).and_then(|n| n.checked_sub(end));
        if input.dtype != DType::BF16 || input.qparams.is_some()
            || input.meta.shape().as_slice() != [1, DIMENSION]
            || input.meta.strides().len() != 2 || input.meta.strides()[1] != 1
            || start % 2 != 0 || extent.is_none_or(|n| n < (DIMENSION * 2) as u64) {
            return Err(Error::new("NVFP4 query requires contiguous BF16 [1,4096] storage"));
        }
        let coordinates = self.client.empty(DIMENSION * 8);
        let status = self.client.empty(4);
        // SAFETY: checked input extent/alignment, fixed output geometry, one thread.
        unsafe { self.prepare_kernel.launch(&self.client, CubeCount::Static(1, 1, 1),
            RawArgs::new().buffer(&input.handle).buffer(&coordinates).buffer(&status)); }
        self.check_status(status, 1)?;
        Ok(CudaPreparedQuery4096 { client: self.client.clone(), device: self.device.clone(), coordinates })
    }

    /// Copy a validated borrowed geometry into device-owned immutable planes.
    /// Certificate value checks occur on the GPU when scoring (as does the
    /// canonical reader's finite-score check); arbitrary code bytes keep their
    /// original decoding semantics, including its zero-norm shortcut.
    pub fn upload(&self, segment: ScanSegment<'_>) -> Result<CudaScanSegment4096, Error> {
        if segment.dimension() != DIMENSION || segment.blocks_per_row() != 256 || segment.codes_per_row() != 2048 {
            return Err(Error::new("NVFP4 scorer requires 4096-dimensional row geometry"));
        }
        row_bytes(segment.rows())?;
        // ScanSegment's constructor already checks every plane length. Empty
        // segments receive inert nonzero backing and are never launched.
        let copy = |bytes: &[u8]| if bytes.is_empty() { self.client.empty(1) } else { self.client.create_from_slice(bytes) };
        Ok(CudaScanSegment4096 { client: self.client.clone(), device: self.device.clone(), rows: segment.rows(),
            stages: segment.stages().map(|s| Stage { globals: copy(s.global_scale_bytes()),
                scales: copy(s.block_scales()), codes: copy(s.codes()) }),
            norms: copy(segment.reconstruction_norm_bytes()), errors: copy(segment.error_bound_bytes()) })
    }

    /// Return one reconstructed cosine per physical row in segment/row order.
    /// This is neither an upper bound nor exact source-embedding reranking.
    pub fn score(&self, query: &CudaPreparedQuery4096, segments: &[CudaScanSegment4096]) -> Result<Vec<f64>, Error> {
        self.check_device(&query.client, &query.device)?;
        let mut total = 0usize;
        for segment in segments {
            self.check_device(&segment.client, &segment.device)?;
            row_bytes(segment.rows)?;
            total = total.checked_add(segment.rows).ok_or_else(|| Error::new("NVFP4 row count overflow"))?;
        }
        total.checked_mul(8).filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| Error::new("NVFP4 result allocation overflow"))?;
        let mut result = Vec::new();
        result.try_reserve_exact(total).map_err(|e| Error::new(format!("NVFP4 result allocation: {e}")))?;
        for segment in segments {
            if segment.rows == 0 { continue; }
            let (out_bytes, status_bytes) = row_bytes(segment.rows)?;
            let output = self.client.empty(out_bytes);
            let status = self.client.empty(status_bytes);
            let [p, c] = &segment.stages;
            // SAFETY: opaque owners preserve validated immutable input geometry;
            // each block owns one row, allocation/launch bounds checked above.
            unsafe { self.score_kernel.launch(&self.client, CubeCount::Static(segment.rows as u32, 1, 1),
                RawArgs::new().buffer(&query.coordinates)
                    .buffer(&p.globals).buffer(&p.scales).buffer(&p.codes)
                    .buffer(&c.globals).buffer(&c.scales).buffer(&c.codes)
                    .buffer(&segment.norms).buffer(&segment.errors).buffer(&output).buffer(&status)); }
            self.check_status(status, segment.rows)?;
            let bytes = self.client.read_one(output).map_err(|e| Error::new(format!("CUDA scores: {e:?}")))?;
            let bytes: &[u8] = bytes.as_ref();
            if bytes.len() != out_bytes { return Err(Error::new("CUDA score extent changed")); }
            result.extend(bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().expect("eight bytes"))));
        }
        Ok(result)
    }

    fn check_status(&self, status: Handle, rows: usize) -> Result<(), Error> {
        let bytes = self.client.read_one(status).map_err(|e| Error::new(format!("CUDA status: {e:?}")))?;
        let bytes: &[u8] = bytes.as_ref();
        if bytes.len() != rows * 4 { return Err(Error::new("CUDA status extent changed")); }
        for (row, bytes) in bytes.chunks_exact(4).enumerate() {
            match u32::from_le_bytes(bytes.try_into().expect("four bytes")) {
                0 => {},
                1 => return Err(Error::new("embedding coordinates must all be finite")),
                2 => return Err(Error::new(format!("NVFP4 row {row} certificate is invalid"))),
                3 => return Err(Error::new(format!("NVFP4 row {row} reconstruction produced a nonfinite score"))),
                other => return Err(Error::new(format!("unexpected CUDA status {other}"))),
            }
        }
        Ok(())
    }
}

fn row_bytes(rows: usize) -> Result<(usize, usize), Error> {
    if rows > i32::MAX as usize { return Err(Error::new("NVFP4 row count exceeds CUDA grid bound")); }
    // Includes the largest retained plane and all output/status allocations.
    rows.checked_mul(2048).filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| Error::new("NVFP4 row-plane allocation overflow"))?;
    Ok((rows.checked_mul(8).ok_or_else(|| Error::new("NVFP4 output overflow"))?,
        rows.checked_mul(4).ok_or_else(|| Error::new("NVFP4 status overflow"))?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{PreparedQuery, QuantizedRow, ScanStage, raw_dot_f64};
    use half::bf16;

    // Ephemeral test-only storage planes, never a production index/catalogue.
    struct HostRows { stages: [[Vec<u8>; 3]; 2], norms: Vec<u8>, errors: Vec<u8>, rows: usize }
    impl HostRows {
        fn new(rows: &[QuantizedRow]) -> Self {
            let mut this = Self { stages: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
                norms: Vec::new(), errors: Vec::new(), rows: rows.len() };
            for row in rows {
                for (out, stage) in this.stages.iter_mut().zip(row.stages()) {
                    out[0].extend(stage.global_scale_bytes()); out[1].extend(stage.block_scales()); out[2].extend(stage.codes());
                }
                this.norms.extend(row.reconstruction_norm().to_le_bytes());
                this.errors.extend(row.error_bound().to_le_bytes());
            }
            this
        }
        fn view(&self) -> ScanSegment<'_> {
            ScanSegment::new([0; 32], self.rows, 4096, 256, 2048,
                self.stages.each_ref().map(|s| ScanStage::new(&s[0], &s[1], &s[2])), &self.norms, &self.errors).unwrap()
        }
    }
    fn tensor(scorer: &CudaReconstructedScorer4096, bits: &[u16]) -> CubeTensor<CudaRuntime> {
        let bytes: Vec<_> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        CubeTensor::new_contiguous(scorer.client.clone(), scorer.device.clone(), [1, bits.len()].as_slice().into(),
            scorer.client.create_from_slice(&bytes), DType::BF16)
    }
    fn f32s(bits: &[u16]) -> Vec<f32> { bits.iter().map(|&b| bf16::from_bits(b).to_f32()).collect() }
    fn oracle(query: &PreparedQuery, rows: &HostRows) -> Result<Vec<f64>, Error> {
        let segment = rows.view();
        (0..rows.rows).map(|row| {
            let norm = segment.row_certificate(row)?.reconstruction_norm();
            let score = if norm == 0.0 { 0.0 } else { raw_dot_f64(query.scan_coordinates(), segment, row) / norm };
            if !score.is_finite() { return Err(Error::new("nonfinite score")); }
            Ok(score.clamp(-1.0, 1.0))
        }).collect()
    }
    fn bits(values: &[f64]) -> Vec<u64> { values.iter().map(|v| v.to_bits()).collect() }
    fn prepared(scorer: &CudaReconstructedScorer4096, values: &[u16]) -> CudaPreparedQuery4096 {
        let input = tensor(scorer, values);
        let original = scorer.client.read_one(input.handle.clone()).unwrap().to_vec();
        let query = scorer.prepare(&input).unwrap();
        assert_eq!(scorer.client.read_one(input.handle.clone()).unwrap().to_vec(), original);
        let actual = scorer.client.read_one(query.coordinates.clone()).unwrap();
        let actual: &[u8] = actual.as_ref();
        let expected = PreparedQuery::new(&f32s(values), 4096).unwrap();
        let expected: Vec<u8> = expected.scan_coordinates().iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(actual, expected, "GPU prepared query must be bit-identical");
        query // Input is dropped here; the prepared output owns all needed bytes.
    }
    fn patterns() -> Vec<Vec<u16>> {
        let mut spike = vec![0; 4096]; spike[17] = 0x3f80;
        vec![vec![0; 4096], vec![0x8000; 4096], vec![1; 4096], spike,
            (0..4096).map(|i| [1, 0x8001, 0x7f7f, 0xff7f, 0x3f80, 0xbf80, 0x3f00, 0x3fc0][i % 8]).collect(),
            (0..4096).map(|i| ((i * 4051 + 37) & 0xffff) as u16).map(|b| if b & 0x7f80 == 0x7f80 { 0 } else { b }).collect()]
    }
    fn compare_all(scorer: &CudaReconstructedScorer4096, inputs: &[Vec<u16>]) {
        // Canonical CPU numerics are strictly an independent test oracle.
        let rows: Vec<_> = inputs.iter().map(|v| QuantizedRow::quantize(&f32s(v), 4096).unwrap()).collect();
        let host = HostRows::new(&rows);
        let resident = scorer.upload(host.view()).unwrap();
        let reverse_rows: Vec<_> = rows.iter().rev().cloned().collect();
        let reverse_host = HostRows::new(&reverse_rows);
        let reverse = scorer.upload(reverse_host.view()).unwrap();
        let split = rows.len() / 2;
        let partitioned = vec![scorer.upload(HostRows::new(&rows[..split]).view()).unwrap(),
            scorer.upload(HostRows::new(&rows[split..]).view()).unwrap()];
        // Those temporary host planes have already been freed; the GPU owns them.
        for input in inputs {
            let query = prepared(scorer, input);
            let expected = oracle(&PreparedQuery::new(&f32s(input), 4096).unwrap(), &host).unwrap();
            let actual = scorer.score(&query, std::slice::from_ref(&resident)).unwrap();
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(bits(&scorer.score(&query, std::slice::from_ref(&resident)).unwrap()), bits(&actual));
            assert_eq!(bits(&scorer.score(&query, &partitioned).unwrap()), bits(&actual));
            let mut reversed = scorer.score(&query, std::slice::from_ref(&reverse)).unwrap(); reversed.reverse();
            assert_eq!(bits(&reversed), bits(&actual));
        }
    }

    #[test]
    fn geometry_and_eight_lane_rounding_are_pinned() {
        assert_eq!(row_bytes(0).unwrap(), (0, 0));
        assert_eq!(row_bytes(3).unwrap(), (24, 12));
        assert!(row_bytes(i32::MAX as usize + 1).is_err());
        assert!(row_bytes(usize::MAX).is_err());
        let source = include_str!("cuda_score.cu");
        for required in ["double lanes[8]", "lane < 8", "__dadd_rn", "__dmul_rn", "__ddiv_rn", "__dsqrt_rn", "row_norm == 0.0"] {
            assert!(source.contains(required));
        }
        assert!(!source.contains("atomic"));
        assert!(!source.contains("__shfl"));
    }

    fn ptx_contract(root: &std::path::Path) {
        #[derive(serde::Deserialize)] struct Entry { value: Ptx }
        #[derive(serde::Deserialize)] struct Ptx { entrypoint_name: String, ptx: Vec<i8> }
        let mut pending = vec![root.to_path_buf()];
        let mut prepare = false; let mut score = false;
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() { pending.push(path); continue; }
                if !path.file_name().is_some_and(|n| n == "chunk0.cbor") { continue; }
                let mut cursor = std::io::Cursor::new(std::fs::read(path).unwrap());
                while cursor.position() < cursor.get_ref().len() as u64 {
                    let entry: Entry = ciborium::from_reader(&mut cursor).unwrap();
                    match entry.value.entrypoint_name.as_str() {
                        "prepare_query_4096" => prepare = true,
                        "reconstructed_cosines_4096" => score = true,
                        _ => continue,
                    }
                    let text = String::from_utf8(entry.value.ptx.into_iter().map(|b| b as u8).take_while(|&b| b != 0).collect()).unwrap();
                    for forbidden in [".ftz", ".approx", "fma.rn.f64", "mad.rn.f64"] { assert!(!text.contains(forbidden), "{forbidden}"); }
                    for required in ["add.rn.f64", "mul.rn.f64", "div.rn.f64"] { assert!(text.contains(required), "{required}"); }
                }
            }
        }
        assert!(prepare && score, "both entrypoints must leave exact PTX evidence");
    }

    #[test]
    #[ignore = "requires exclusively reserved CUDA device"]
    fn gpu_reconstructed_cosines_match_oracle_and_reject_invalid_inputs() {
        use cubecl::config::{RuntimeConfig, cache::CacheConfig};
        let cache = std::path::PathBuf::from(std::env::var("WEMM_NVFP4_PTX_CACHE").expect("explicit fresh cache"));
        assert!(!cache.exists());
        let mut config = cubecl::config::CubeClRuntimeConfig::default();
        config.compilation.cache = Some(CacheConfig::File(cache.clone()));
        cubecl::config::CubeClRuntimeConfig::set(config);
        let scorer = CudaReconstructedScorer4096::new(CudaDevice { index: 0 }).unwrap();
        let patterns = patterns(); compare_all(&scorer, &patterns);
        for invalid in [0x7f80, 0xff80, 0x7fc1] {
            let mut v = patterns[0].clone(); v[23] = invalid;
            assert!(scorer.prepare(&tensor(&scorer, &v)).err().unwrap().to_string().contains("finite"));
        }
        assert!(scorer.prepare(&tensor(&scorer, &[0; 3])).is_err());
        for mutation in 0..6 {
            let mut t = tensor(&scorer, &patterns[0]);
            match mutation { 0 => t.dtype = DType::F16, 1 => t.meta.strides_mut()[1] = 2,
                2 => t.handle = scorer.client.empty(2), 3 => t.device.index = 1,
                4 => t.handle.offset_start = Some(1), _ => t.handle.offset_end = Some(u64::MAX) }
            assert!(scorer.prepare(&t).is_err());
        }
        let mut offset_bits = vec![0x4000]; offset_bits.extend(&patterns[3]);
        let mut offset = tensor(&scorer, &offset_bits); offset.handle = offset.handle.offset_start(2);
        offset.meta = tensor(&scorer, &patterns[3]).meta;
        let offset_query = scorer.prepare(&offset).unwrap();
        let expected_query = prepared(&scorer, &patterns[3]);
        assert_eq!(scorer.client.read_one(offset_query.coordinates).unwrap().to_vec(), scorer.client.read_one(expected_query.coordinates).unwrap().to_vec());
        let query = prepared(&scorer, &patterns[3]);
        assert!(scorer.score(&query, &[]).unwrap().is_empty());
        let empty = scorer.upload(HostRows::new(&[]).view()).unwrap();
        assert!(scorer.score(&query, &[empty]).unwrap().is_empty());
        let short = QuantizedRow::quantize(&[1.0; 256], 256).unwrap();
        assert!(scorer.upload(ScanSegment::new([0; 32], 1, 256, 16, 128,
            short.stages().each_ref().map(|s| ScanStage::new(s.global_scale_bytes(), s.block_scales(), s.codes())),
            &short.reconstruction_norm().to_le_bytes(), &short.error_bound().to_le_bytes()).unwrap()).is_err());
        let valid = QuantizedRow::quantize(&f32s(&patterns[3]), 4096).unwrap();
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            for error_plane in [false, true] {
                let mut h = HostRows::new(std::slice::from_ref(&valid));
                if error_plane { h.errors = bad.to_le_bytes().to_vec(); } else { h.norms = bad.to_le_bytes().to_vec(); }
                assert!(oracle(&PreparedQuery::new(&f32s(&patterns[3]), 4096).unwrap(), &h).is_err());
                assert!(scorer.score(&query, &[scorer.upload(h.view()).unwrap()]).unwrap_err().to_string().contains("certificate"));
            }
        }
        let mut h = HostRows::new(std::slice::from_ref(&valid));
        h.stages[0][0] = f32::NAN.to_le_bytes().to_vec();
        assert!(scorer.score(&query, &[scorer.upload(h.view()).unwrap()]).unwrap_err().to_string().contains("nonfinite"));
        h.norms = (-0.0f32).to_le_bytes().to_vec();
        assert_eq!(bits(&scorer.score(&query, &[scorer.upload(h.view()).unwrap()]).unwrap()), vec![0]);
        let mut h = HostRows::new(std::slice::from_ref(&valid)); h.norms = f32::MIN_POSITIVE.to_le_bytes().to_vec();
        assert_eq!(scorer.score(&query, &[scorer.upload(h.view()).unwrap()]).unwrap(), vec![1.0]);
        let mut wrong_segment = scorer.upload(HostRows::new(&[valid]).view()).unwrap(); wrong_segment.device.index = 1;
        assert!(scorer.score(&query, &[wrong_segment]).is_err());
        let mut wrong_query = prepared(&scorer, &patterns[3]); wrong_query.device.index = 1;
        assert!(scorer.score(&wrong_query, &[]).is_err());
        ptx_contract(&cache);
    }

    #[test]
    #[ignore = "requires reserved CUDA and retained WeMM evidence path"]
    fn gpu_reconstructed_cosines_match_retained_wemm_rows() {
        let path = std::env::var("WEMM_NVFP4_ORACLE_JSON").expect("explicit retained evidence path");
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let inputs: Vec<Vec<u16>> = json["embeddings"].as_array().unwrap().iter().map(|r|
            r["bits"].as_array().unwrap().iter().map(|v| u16::try_from(v.as_u64().unwrap()).unwrap()).collect()).collect();
        assert!(!inputs.is_empty());
        compare_all(&CudaReconstructedScorer4096::new(CudaDevice { index: 0 }).unwrap(), &inputs);
        println!("{} retained rows, including repeated inputs; no new model inference", inputs.len());
    }
}
