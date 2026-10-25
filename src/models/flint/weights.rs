//! One selected fixture block, one dense tensor or one routed expert at a time. GGUF scales
//! remain separate, applied after matmul by the math layer. No full-model map.

use std::{path::Path,time::Instant};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use crate::formats::gguf::{index::{IndexedFile,RowReadTimings}, nvfp4};
use super::config::{BlockConfig, METADATA_KEYS};

pub trait Weights {
    /// Expected GGML shape/type are mandatory; expert slices retain identity.
    fn get(&mut self, suffix: &str, dims: &[usize], ty: u32, expert: Option<usize>) -> Result<Vec<f32>>;
    /// Unchanged GGUF type-40 blocks, with the same bounded selection/receipt.
    /// Separate `.scale` tensors are deliberately not included.
    fn get_nvfp4(&mut self, suffix: &str, dims: &[usize], expert: Option<usize>) -> Result<Vec<u8>>;
}

fn row_selection(dims: &[usize], expert: Option<usize>) -> Result<(u64, u64, usize)> {
    ensure!(!dims.is_empty() && dims.iter().all(|&d| d > 0), "empty weight geometry");
    let (first, rows) = if let Some(e) = expert {
        ensure!(dims.len() == 3 && e < dims[2], "invalid expert slice");
        (e.checked_mul(dims[1]).context("expert row offset overflow")?, dims[1])
    } else {
        ensure!(dims.len() <= 2, "full expert-bank loading is forbidden");
        (0, dims.get(1).copied().unwrap_or(1))
    };
    first.checked_add(rows).context("selected row end overflow")?;
    let elements = rows.checked_mul(dims[0]).context("selected element count overflow")?;
    Ok((u64::try_from(first)?, u64::try_from(rows)?, elements))
}

#[derive(Clone, Serialize)]
pub struct ReadReceipt {
    pub name: String,
    pub ggml_dims: Vec<u64>,
    pub ggml_type: u32,
    pub expert: Option<usize>,
    pub first_row: u64,
    pub rows: u64,
    pub absolute_start: u64,
    pub absolute_end: u64,
    pub raw_blake3: String,
}

/// Local diagnostic counters; no retained weight/hash cache or generic profiler.
#[derive(Clone,Default,Serialize)]
pub struct Work {
    pub calls:u64, pub input_bytes:u64, pub output_bytes:u64, pub seconds:f64,
}
impl Work {
    pub fn record(&mut self,start:Instant,input:usize,output:usize) {
        self.seconds+=start.elapsed().as_secs_f64(); self.calls+=1;
        self.input_bytes+=input as u64; self.output_bytes+=output as u64;
    }
}
#[derive(Default,Serialize)]
pub struct WeightHostTimings {
    /// Measured API body inclusive of row reads, digest, conversion and receipt
    /// bookkeeping below. Function-exit drops/return overhead remain unclassified.
    pub api:Work, pub receipt_blake3_hex:Work,
    pub f32_conversion_finite:Work, pub bf16_conversion_finite:Work,
    pub nvfp4_conversion_finite:Work, pub packed_validation:Work,
    /// Outside weight API: global embedding/scalar conversions and block selection.
    pub global_conversion:Work, pub block_selection:Work, pub receipt_drop:Work,
}
#[derive(Default,Serialize)]
pub struct HostPhase {
    pub weights:WeightHostTimings, pub rows:RowReadTimings,
}
impl HostPhase {
    /// Disjoint selected subspans nested INSIDE weights.api, never added to it.
    pub fn weight_selected_seconds(&self)->f64 {
        let w=&self.weights;
        self.rows.selected_seconds()+w.receipt_blake3_hex.seconds+w.f32_conversion_finite.seconds
            +w.bf16_conversion_finite.seconds+w.nvfp4_conversion_finite.seconds+w.packed_validation.seconds
    }
    pub fn weights_other_seconds(&self)->f64 { self.weights.api.seconds-self.weight_selected_seconds() }
    /// Includes unclassified CPU/runtime/driver/device time, not CPU-only time.
    pub fn phase_other_seconds(&self,total:f64)->f64 {
        let w=&self.weights;
        total-w.api.seconds-w.global_conversion.seconds-w.block_selection.seconds-w.receipt_drop.seconds
    }
}

/// Diagnostic provenance only; None never disables data or file-identity guards.
/// Fixed per instance so an observed call cannot silently change later behavior.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq,Serialize)]
#[serde(rename_all="snake_case")]
pub enum ReceiptPolicy { #[default] Collect, None }

pub struct GgufWeights {
    pub file: IndexedFile,
    pub reads: Vec<ReadReceipt>,
    pub config: BlockConfig,
    pub host:WeightHostTimings,
    receipts:ReceiptPolicy,
}

impl GgufWeights {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_block(path, 2)
    }
    pub fn open_block(path: &Path, block: usize) -> Result<Self> {
        Self::open_block_with_receipts(path,block,ReceiptPolicy::Collect)
    }
    /// Explicit opt-out for callers that do not consume selected-weight receipts.
    /// Existing constructors collect by default; absence means an empty collection,
    /// never a receipt with a fabricated or empty digest.
    pub fn open_block_with_receipts(path:&Path,block:usize,receipts:ReceiptPolicy)->Result<Self> {
        let file = IndexedFile::open_with_metadata(path, METADATA_KEYS)?;
        let config = BlockConfig::from_metadata(&file.index.metadata, block)?;
        Ok(Self { file, reads: Vec::new(), config, host:WeightHostTimings::default(), receipts })
    }
    pub fn receipt_policy(&self)->ReceiptPolicy { self.receipts }
    pub fn take_host_phase(&mut self)->HostPhase {
        HostPhase {weights:std::mem::take(&mut self.host),rows:std::mem::take(&mut self.file.row_timings)}
    }
    fn receipt_digest(&mut self,raw:&[u8])->String {
        let start=Instant::now();
        let digest=blake3::hash(raw).to_hex().to_string();
        self.host.receipt_blake3_hex.record(start,raw.len(),digest.len()); digest
    }
    pub fn select_block(&mut self, block: usize) -> Result<()> {
        let start=Instant::now();
        self.config = BlockConfig::from_metadata(&self.file.index.metadata, block)?;
        self.host.block_selection.record(start,0,0);
        Ok(())
    }
    /// Checked global tensor rows. No prefix escape, whole-head read or changed
    /// MAX_READ: the same IndexedFile row/overflow/stat checks apply.
    pub fn global_rows(&mut self, name: &str, dims: &[usize], ty: u32,
        first: usize, rows: usize) -> Result<Vec<u8>> {
        let api_start=Instant::now();
        ensure!(matches!(name,"token_embd.weight"|"output_norm.weight"|"output.weight"|"output.scale"),
            "unsupported global tensor {name}");
        let info = self.file.index.tensor(name)?.clone();
        ensure!(info.dims == dims.iter().map(|&d|d as u64).collect::<Vec<_>>() && info.ggml_type == ty,
            "unexpected global tensor shape/type: {name}");
        let (start,raw) = self.file.read_rows(name,first.try_into()?,rows.try_into()?)?;
        let end=start.checked_add(raw.len().try_into()?).context("global receipt overflow")?;
        if self.receipts==ReceiptPolicy::Collect {
            let raw_blake3=self.receipt_digest(&raw);
            self.reads.push(ReadReceipt {name:name.into(),ggml_dims:info.dims,ggml_type:ty,expert:None,
                first_row:first.try_into()?,rows:rows.try_into()?,absolute_start:start,
                absolute_end:end,raw_blake3});
        }
        self.host.api.record(api_start,raw.len(),raw.len());
        Ok(raw)
    }
    fn read_weight(&mut self, suffix: &str, dims: &[usize], ty: u32, expert: Option<usize>)
        -> Result<(Vec<u8>, usize, Option<ReadReceipt>)> {
        ensure!(!suffix.contains('.') || suffix.ends_with(".weight") || suffix.ends_with(".scale"), "invalid tensor suffix");
        let (first, rows, elements) = row_selection(dims, expert)?;
        let name = format!("blk.{}.{suffix}", self.config.block);
        let info = self.file.index.tensor(&name)?.clone();
        ensure!(info.dims == dims.iter().map(|&d| d as u64).collect::<Vec<_>>() && info.ggml_type == ty,
            "unexpected selected-block shape/type: {name}");
        let (start, raw) = self.file.read_rows(&name, first, rows)?;
        let end = start.checked_add(u64::try_from(raw.len())?).context("read receipt end overflow")?;
        let receipt=if self.receipts==ReceiptPolicy::Collect {
            let raw_blake3=self.receipt_digest(&raw);
            Some(ReadReceipt { name, ggml_dims: info.dims, ggml_type: ty, expert,
                first_row: first, rows, absolute_start: start, absolute_end: end,raw_blake3 })
        } else {None};
        Ok((raw, elements, receipt))
    }
}

impl Weights for GgufWeights {
    fn get(&mut self, suffix: &str, dims: &[usize], ty: u32, expert: Option<usize>) -> Result<Vec<f32>> {
        let api_start=Instant::now();
        let (raw, elements, receipt) = self.read_weight(suffix, dims, ty, expert)?;
        let prep_start=Instant::now();
        let data = match ty {
            0 => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            30 => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)).collect(),
            40 => nvfp4::decode(&raw, elements)?,
            _ => anyhow::bail!("unsupported math weight type {ty}"),
        };
        ensure!(data.iter().all(|v| v.is_finite()), "non-finite weight: blk.{}.{suffix}", self.config.block);
        let measurement=match ty {0=>&mut self.host.f32_conversion_finite,30=>&mut self.host.bf16_conversion_finite,
            40=>&mut self.host.nvfp4_conversion_finite,_=>unreachable!()};
        measurement.record(prep_start,raw.len(),data.len()*4);
        if let Some(receipt)=receipt { self.reads.push(receipt); }
        self.host.api.record(api_start,raw.len(),data.len()*4);
        Ok(data)
    }

    fn get_nvfp4(&mut self, suffix: &str, dims: &[usize], expert: Option<usize>) -> Result<Vec<u8>> {
        let api_start=Instant::now();
        let (raw, elements, receipt) = self.read_weight(suffix, dims, 40, expert)?;
        let prep_start=Instant::now();
        ensure!(dims[0] % nvfp4::ELEMENTS == 0, "NVFP4 row is not block-aligned");
        let bytes = (elements / nvfp4::ELEMENTS).checked_mul(nvfp4::BYTES)
            .context("NVFP4 byte length overflow")?;
        ensure!(raw.len() == bytes, "NVFP4 selected byte length differs");
        self.host.packed_validation.record(prep_start,raw.len(),raw.len());
        if let Some(receipt)=receipt { self.reads.push(receipt); }
        self.host.api.record(api_start,raw.len(),raw.len());
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_span_partition_is_nested_not_additive() {
        let mut phase=HostPhase::default();
        phase.rows.read_exact_seconds=2.0;
        phase.weights.receipt_blake3_hex.seconds=3.0;
        phase.weights.bf16_conversion_finite.seconds=1.0;
        phase.weights.api.seconds=7.0;
        phase.weights.global_conversion.seconds=0.25;
        phase.weights.block_selection.seconds=0.5;
        phase.weights.receipt_drop.seconds=0.25;
        assert_eq!(phase.weight_selected_seconds(),6.0);
        assert_eq!(phase.weights_other_seconds(),1.0);
        assert_eq!(phase.phase_other_seconds(10.0),2.0);
    }

    fn with_payload(name:&str,ty:u32,elements:u64,payload:&[u8],
        check:impl FnOnce(&Path)->Result<()>)->Result<()> {
        use std::{fs::OpenOptions,io::Write,time::{SystemTime,UNIX_EPOCH}};
        let path=std::env::temp_dir().join(format!("flint-host-counts-{}-{}.gguf",
            std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()));
        let mut bytes=b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes()); bytes.extend(1u64.to_le_bytes()); bytes.extend(0u64.to_le_bytes());
        bytes.extend((name.len() as u64).to_le_bytes()); bytes.extend(name.as_bytes());
        bytes.extend(1u32.to_le_bytes()); bytes.extend(elements.to_le_bytes());
        bytes.extend(ty.to_le_bytes()); bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(32)*32,0);
        bytes.extend(payload);
        OpenOptions::new().create_new(true).write(true).open(&path)?.write_all(&bytes)?;
        let result=check(&path);
        std::fs::remove_file(&path)?;
        result
    }

    fn fixture(path:&Path,receipts:ReceiptPolicy)->Result<GgufWeights> {
        Ok(GgufWeights {file:IndexedFile::open(path)?,reads:Vec::new(),config:BlockConfig::LAYER2,
            host:WeightHostTimings::default(),receipts})
    }

    #[test]
    fn host_read_digest_decode_counts_preserve_bytes() -> Result<()> {
        use std::{fs::OpenOptions,io::Write};
        assert_eq!(ReceiptPolicy::default(),ReceiptPolicy::Collect);
        let payload=vec![0x38;36];
        with_payload("blk.2.test.weight",40,64,&payload,|path| {
            let mut instances=vec![fixture(path,ReceiptPolicy::Collect)?,fixture(path,ReceiptPolicy::None)?];
            for w in &mut instances {
                assert_eq!(w.get_nvfp4("test.weight",&[64],None)?,payload);
                assert_eq!(w.get("test.weight",&[64],40,None)?,nvfp4::decode(&payload,64)?);
                let collect=w.receipt_policy()==ReceiptPolicy::Collect;
                assert_eq!(w.reads.len(),if collect {2} else {0});
                let digest=blake3::hash(&payload).to_hex().to_string();
                assert!(w.reads.iter().all(|r|r.raw_blake3==digest));
                let phase=w.take_host_phase();
                assert_eq!((phase.rows.calls,phase.rows.bytes,phase.rows.identity_calls),(2,72,4));
                let host=&phase.weights;
                assert_eq!((host.api.calls,host.api.input_bytes,host.api.output_bytes),(2,72,292));
                assert_eq!((host.receipt_blake3_hex.calls,host.receipt_blake3_hex.input_bytes,host.receipt_blake3_hex.output_bytes),
                    if collect {(2,72,128)} else {(0,0,0)});
                if !collect { assert_eq!(host.receipt_blake3_hex.seconds,0.0); }
                assert_eq!((host.nvfp4_conversion_finite.calls,host.nvfp4_conversion_finite.input_bytes,host.nvfp4_conversion_finite.output_bytes),(1,36,256));
                assert_eq!(host.packed_validation.calls,1);
                assert!(phase.weights_other_seconds()>=0.0);
                assert_eq!(w.take_host_phase().rows.calls,0);
                assert_eq!(w.host.api.calls,0);
            }
            let refusals=|w:&mut GgufWeights|->Vec<String> {vec![
                w.get_nvfp4("test.weight",&[128],None).unwrap_err().to_string(),
                w.get("test.weight",&[64],0,None).unwrap_err().to_string(),
                w.get_nvfp4("test.weight",&[64],Some(0)).unwrap_err().to_string()]};
            assert_eq!(refusals(&mut instances[0]),refusals(&mut instances[1]));
            // Both modes retain the same opened-file identity refusal.
            OpenOptions::new().append(true).open(path)?.write_all(&[0])?;
            let a=instances[0].get_nvfp4("test.weight",&[64],None).unwrap_err().to_string();
            let b=instances[1].get_nvfp4("test.weight",&[64],None).unwrap_err().to_string();
            assert_eq!(a,b); Ok(())
        })
    }

    #[test]
    fn receipt_policy_global_rows_and_finite_refusals() -> Result<()> {
        let payload:Vec<u8>=[1.0f32,2.0].iter().flat_map(|v|v.to_le_bytes()).collect();
        with_payload("output_norm.weight",0,2,&payload,|path| {
            let mut a=fixture(path,ReceiptPolicy::Collect)?;
            let mut b=fixture(path,ReceiptPolicy::None)?;
            assert_eq!(a.global_rows("output_norm.weight",&[2],0,0,1)?,payload);
            assert_eq!(b.global_rows("output_norm.weight",&[2],0,0,1)?,payload);
            assert_eq!(a.reads.len(),1); assert!(b.reads.is_empty());
            assert_eq!(a.reads[0].raw_blake3,blake3::hash(&payload).to_hex().to_string());
            assert_eq!(a.host.receipt_blake3_hex.calls,1);
            assert_eq!(b.host.receipt_blake3_hex.calls,0);
            assert_eq!(b.host.receipt_blake3_hex.input_bytes,0);
            assert_eq!(b.host.receipt_blake3_hex.output_bytes,0);
            assert_eq!(b.host.receipt_blake3_hex.seconds,0.0);
            let refusals=|w:&mut GgufWeights|->Vec<String> {vec![
                w.global_rows("output_norm.weight",&[2],0,1,1).unwrap_err().to_string(),
                w.global_rows("output_norm.weight",&[2],30,0,1).unwrap_err().to_string(),
                w.global_rows("output_norm.weight",&[4],0,0,1).unwrap_err().to_string(),
                w.global_rows("not_allowed",&[2],0,0,1).unwrap_err().to_string()]};
            assert_eq!(refusals(&mut a),refusals(&mut b)); Ok(())
        })?;
        with_payload("blk.2.test.weight",0,1,&f32::NAN.to_le_bytes(),|path| {
            for policy in [ReceiptPolicy::Collect,ReceiptPolicy::None] {
                let mut w=fixture(path,policy)?;
                assert_eq!(w.get("test.weight",&[1],0,None).unwrap_err().to_string(),
                    "non-finite weight: blk.2.test.weight");
                assert!(w.reads.is_empty());
            }
            Ok(())
        })
    }

    #[test]
    fn shared_row_selection_guards_and_expert_offsets() {
        assert_eq!(row_selection(&[64, 9], None).unwrap(), (0, 9, 576));
        assert_eq!(row_selection(&[64, 9, 256], Some(0)).unwrap(), (0, 9, 576));
        assert_eq!(row_selection(&[64, 9, 256], Some(255)).unwrap(), (2295, 9, 576));
        assert_eq!(row_selection(&[256], None).unwrap(), (0, 1, 256));
        for (dims, expert) in [
            (vec![], None), (vec![0, 9], None), (vec![64, 9, 256], None),
            (vec![64, 9, 256], Some(256)), (vec![64, 9], Some(0)),
            (vec![64, usize::MAX, 3], Some(2)), (vec![usize::MAX, 2], None),
            (vec![1, usize::MAX, 2], Some(1)),
        ] { assert!(row_selection(&dims, expert).is_err(), "{dims:?} {expert:?}"); }
    }
}
