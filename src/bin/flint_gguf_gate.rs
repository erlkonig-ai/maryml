//! Stored-weight parity only. The oracle executable calls the actual pinned
//! GGML decoder; no copied decoder is embedded in this gate. Separate .scale
//! tensors are recorded, NOT folded into decoded values or matmul outputs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use mary::formats::gguf::{index::IndexedFile, nvfp4};
use serde_json::{Value, json};

fn hash(data: &[u8]) -> String { blake3::hash(data).to_hex().to_string() }

fn compare(oracle: &Path, out: &Path, label: &str, raw: &[u8]) -> Result<Value> {
    let input = out.join(format!("{label}.nvfp4"));
    let reference = out.join(format!("{label}.reference.f32"));
    fs::write(&input, raw)?;
    let status = Command::new(oracle).arg("decode").arg(&input).arg(&reference).status()?;
    ensure!(status.success(), "reference decoder failed for {label}: {status}");
    let expected = fs::read(&reference)?;
    let elements = raw.len() / nvfp4::BYTES * nvfp4::ELEMENTS;
    let actual = nvfp4::decode(raw, elements)?;
    ensure!(expected.len() == elements * 4, "reference length mismatch");
    let bits: Vec<u8> = actual.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mismatches = expected.chunks_exact(4).zip(bits.chunks_exact(4)).filter(|(a, b)| a != b).count();
    ensure!(mismatches == 0, "{label}: {mismatches}/{elements} f32 bit mismatches");
    Ok(json!({"label": label, "packed_bytes": raw.len(), "elements": elements,
        "packed_blake3": hash(raw), "reference_f32_blake3": hash(&expected),
        "mismatches": mismatches, "comparison": "f32 bits, including signed zero"}))
}

fn check_reference_index(file: &IndexedFile, path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = format!("data_offset\t{}", file.index.data_offset);
    ensure!(lines.next() == Some(header.as_str()), "reference data offset differs");
    let rows: Vec<_> = lines.collect();
    ensure!(rows.len() == file.index.tensors.len(), "tensor count differs");
    for (line, info) in rows.iter().zip(&file.index.tensors) {
        let cols: Vec<_> = line.split('\t').collect();
        ensure!(cols.len() == 5, "invalid reference tensor row");
        let mut dims = info.dims.clone();
        dims.resize(4, 1);
        let reference_dims: Vec<u64> = cols[2].split(',').map(str::parse).collect::<std::result::Result<_, _>>()?;
        ensure!(cols[0] == info.name && cols[1].parse::<u32>()? == info.ggml_type
            && reference_dims == dims && cols[3].parse::<u64>()? == info.offset,
            "metadata differs for {}", info.name);
        if let Some(bytes) = info.bytes {
            ensure!(cols[4].parse::<u64>()? == bytes, "byte size differs for {}", info.name);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(args.len() == 4, "usage: flint_gguf_gate MODEL ORACLE NATIVE_INDEX_TSV NEW_OUTPUT_DIR");
    let [model, oracle, native_index, out] = args.as_slice() else { unreachable!() };
    fs::create_dir(out).context("output directory must not already exist")?;
    let mut file = IndexedFile::open(model)?;
    check_reference_index(&file, native_index)?;

    // Every scale byte and every E2M1 nibble. The oracle's treatment of 0x7f,
    // 0xff and code8 is checked rather than inferred from a generic FP8 name.
    let mut all = Vec::new();
    for scale in 0..=255u8 {
        all.extend([scale; 4]);
        for _ in 0..4 {
            for code in 0..8u8 { all.push(code | ((code + 8) << 4)); }
        }
    }
    let mut comparisons = vec![compare(oracle, out, "all-codes-scales", &all)?];
    let mut order = vec![0x38, 0x40, 0x48, 0x50];
    for group in 0..4u8 {
        for j in 0..8u8 { order.push(((15 - j) << 4) | ((j + group) & 15)); }
    }
    comparisons.push(compare(oracle, out, "nibble-group-order", &order)?);

    let selections = [
        ("ffn_down_shexp", vec![1024, 2048]),
        ("ffn_gate_shexp", vec![2048, 1024]),
        ("ffn_up_shexp", vec![2048, 1024]),
        ("attn_k", vec![2048, 512]),
        ("attn_output", vec![4096, 2048]),
        ("attn_q", vec![2048, 4096]),
        ("attn_v", vec![2048, 512]),
        ("ffn_down_exps", vec![512, 2048, 256]),
        ("ffn_gate_exps", vec![2048, 512, 256]),
        ("ffn_up_exps", vec![2048, 512, 256]),
    ];
    let mut reads = Vec::new();
    let mut associations = Vec::new();
    let mut payload_bytes = 0u64;
    for (base, dims) in selections {
        let name = format!("blk.2.{base}.weight");
        let info = file.index.tensor(&name)?.clone();
        ensure!(info.ggml_type == 40 && info.dims == dims, "unexpected layer2 shape/type: {name}");
        let total_rows = info.elements()? / dims[0];
        let rows = if dims.len() == 3 {
            vec![(0, 1), (127 * dims[1], 1), (total_rows - 1, 1)]
        } else { vec![(0, 2), (total_rows - 2, 2)] };
        for (first, count) in rows {
            let (offset, raw) = file.read_rows(&name, first, count)?;
            payload_bytes += raw.len() as u64;
            let label = format!("{base}-row{first}-n{count}");
            comparisons.push(compare(oracle, out, &label, &raw)?);
            reads.push(json!({"tensor": name, "ggml_type": info.ggml_type,
                "ggml_dims": dims, "mary_shape": info.shape(), "first_row": first,
                "rows": count, "file_start": offset, "file_end_exclusive": offset + raw.len() as u64,
                "packed_blake3": hash(&raw)}));
        }
        let scale_name = format!("blk.2.{base}.scale");
        let scale = file.index.tensor(&scale_name)?.clone();
        let expected = if dims.len() == 3 { vec![256] } else { vec![1] };
        ensure!(scale.ggml_type == 0 && scale.dims == expected, "unexpected scale_2 association shape/type");
        let (offset, raw) = file.read_rows(&scale_name, 0, 1)?;
        payload_bytes += raw.len() as u64;
        fs::write(out.join(format!("{base}.scale.f32")), &raw)?;
        associations.push(json!({"weight": name, "scale_tensor": scale_name,
            "scale_dims": scale.dims, "per_expert": dims.len() == 3,
            "file_start": offset, "file_end_exclusive": offset + raw.len() as u64,
            "scale_blake3": hash(&raw), "applied": false}));
    }
    // Invalid selection checks use the same opened real file but must not read data.
    let name = "blk.2.ffn_gate_exps.weight";
    ensure!(file.read_rows(name, u64::MAX, 2).is_err(), "accepted overflowing row range");
    ensure!(file.read_rows(name, 0, 0).is_err(), "accepted empty row range");
    ensure!(file.read_rows(name, 0, 20000).is_err(), "accepted over-budget read");
    ensure!(file.read_rows("missing", 0, 1).is_err(), "accepted missing tensor");
    file.check_identity()?;
    let layer2: Vec<_> = file.index.tensors.iter().filter(|t| t.name.starts_with("blk.2.")).collect();
    let report = json!({"status": "passed", "scope": "stored-weight decoding only; no scale_2 application, activation/matmul/layer/runtime parity or full-model expansion",
        "model": model, "file_identity_before_and_after": file.identity,
        "whole_model_freshly_hashed": false, "index_data_offset": file.index.data_offset,
        "indexed_tensor_count": file.index.tensors.len(), "native_index_parity": true,
        "selected_payload_bytes_read": payload_bytes, "max_one_read_bytes": mary::formats::gguf::index::MAX_READ,
        "reference_executable": oracle, "reference_executable_blake3": hash(&fs::read(oracle)?),
        "fixture_scale_values": 256, "fixture_code_values": 16,
        "layer2_tensor_metadata": layer2, "reads": reads,
        "scale_2_associations": associations, "comparisons": comparisons,
        "invalid_selection_checks": 4});
    let text = serde_json::to_string_pretty(&report)?;
    fs::write(out.join("report.json"), &text)?;
    println!("{text}");
    Ok(())
}
