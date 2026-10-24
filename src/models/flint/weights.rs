//! Only layer 2, one dense tensor or one routed expert at a time. GGUF scales
//! remain separate, applied after matmul by the math layer. No full-model map.

use std::path::Path;
use anyhow::{Result, ensure};
use serde::Serialize;
use crate::formats::gguf::{index::IndexedFile, nvfp4};

pub trait Weights {
    /// Expected GGML shape/type are mandatory; expert slices retain identity.
    fn get(&mut self, suffix: &str, dims: &[usize], ty: u32, expert: Option<usize>) -> Result<Vec<f32>>;
}

#[derive(Serialize)]
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

pub struct GgufWeights {
    pub file: IndexedFile,
    pub reads: Vec<ReadReceipt>,
}

impl GgufWeights {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self { file: IndexedFile::open(path)?, reads: Vec::new() })
    }
}

impl Weights for GgufWeights {
    fn get(&mut self, suffix: &str, dims: &[usize], ty: u32, expert: Option<usize>) -> Result<Vec<f32>> {
        ensure!(!suffix.contains('.') || suffix.ends_with(".weight") || suffix.ends_with(".scale"), "invalid tensor suffix");
        let name = format!("blk.2.{suffix}");
        let info = self.file.index.tensor(&name)?.clone();
        ensure!(info.dims == dims.iter().map(|&d| d as u64).collect::<Vec<_>>() && info.ggml_type == ty,
            "unexpected layer-2 shape/type: {name}");
        let (first, rows) = if let Some(e) = expert {
            ensure!(dims.len() == 3 && e < dims[2], "invalid expert slice");
            ((e * dims[1]) as u64, dims[1] as u64)
        } else {
            ensure!(dims.len() <= 2, "full expert-bank loading is forbidden");
            (0, info.elements()? / info.dims[0])
        };
        let (start, raw) = self.file.read_rows(&name, first, rows)?;
        let data = match ty {
            0 => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            30 => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)).collect(),
            40 => nvfp4::decode(&raw, rows as usize * dims[0])?,
            _ => anyhow::bail!("unsupported math weight type {ty}"),
        };
        ensure!(data.iter().all(|v| v.is_finite()), "non-finite weight: {name}");
        self.reads.push(ReadReceipt { name, ggml_dims: info.dims, ggml_type: ty, expert,
            first_row: first, rows, absolute_start: start, absolute_end: start + raw.len() as u64,
            raw_blake3: blake3::hash(&raw).to_hex().to_string() });
        Ok(data)
    }
}
