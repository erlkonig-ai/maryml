//! Only layer 2, one dense tensor or one routed expert at a time. GGUF scales
//! remain separate, applied after matmul by the math layer. No full-model map.

use std::path::Path;
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use crate::formats::gguf::{index::IndexedFile, nvfp4};

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
    fn read_weight(&mut self, suffix: &str, dims: &[usize], ty: u32, expert: Option<usize>)
        -> Result<(Vec<u8>, usize, ReadReceipt)> {
        ensure!(!suffix.contains('.') || suffix.ends_with(".weight") || suffix.ends_with(".scale"), "invalid tensor suffix");
        let (first, rows, elements) = row_selection(dims, expert)?;
        let name = format!("blk.2.{suffix}");
        let info = self.file.index.tensor(&name)?.clone();
        ensure!(info.dims == dims.iter().map(|&d| d as u64).collect::<Vec<_>>() && info.ggml_type == ty,
            "unexpected layer-2 shape/type: {name}");
        let (start, raw) = self.file.read_rows(&name, first, rows)?;
        let end = start.checked_add(u64::try_from(raw.len())?).context("read receipt end overflow")?;
        let receipt = ReadReceipt { name, ggml_dims: info.dims, ggml_type: ty, expert,
            first_row: first, rows, absolute_start: start, absolute_end: end,
            raw_blake3: blake3::hash(&raw).to_hex().to_string() };
        Ok((raw, elements, receipt))
    }
}

impl Weights for GgufWeights {
    fn get(&mut self, suffix: &str, dims: &[usize], ty: u32, expert: Option<usize>) -> Result<Vec<f32>> {
        let (raw, elements, receipt) = self.read_weight(suffix, dims, ty, expert)?;
        let data = match ty {
            0 => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            30 => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)).collect(),
            40 => nvfp4::decode(&raw, elements)?,
            _ => anyhow::bail!("unsupported math weight type {ty}"),
        };
        ensure!(data.iter().all(|v| v.is_finite()), "non-finite weight: {}", receipt.name);
        self.reads.push(receipt);
        Ok(data)
    }

    fn get_nvfp4(&mut self, suffix: &str, dims: &[usize], expert: Option<usize>) -> Result<Vec<u8>> {
        let (raw, elements, receipt) = self.read_weight(suffix, dims, 40, expert)?;
        ensure!(dims[0] % nvfp4::ELEMENTS == 0, "NVFP4 row is not block-aligned");
        let bytes = (elements / nvfp4::ELEMENTS).checked_mul(nvfp4::BYTES)
            .context("NVFP4 byte length overflow")?;
        ensure!(raw.len() == bytes, "NVFP4 selected byte length differs");
        self.reads.push(receipt);
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
