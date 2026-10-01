//! Fixed bloom-long diagnostic; NOT a behavioral/admission gate.
//! ordinary MODEL CONFIG FIXTURE HF NATIVE NEW_REPORT
//! profiled MODEL CONFIG FIXTURE HF NATIVE NEW_REPORT ORDINARY_REPORT
//! CubeCL profiling is immutable after client creation. These modes MUST run
//! in separate processes. Profiling synchronizes and uses SYSTEM timestamps.
use anyhow::{Context, Result, ensure};
use burn::tensor::DType;
use cubecl::{
    config::{CubeClRuntimeConfig, RuntimeConfig, profiling::ProfilingLogLevel},
    cuda::CudaDevice,
};
use mary::{
    models::qwen3_5::{config::Qwen3_5Config, multimodal::PreparedMultimodal, prepared},
    nn::cuda_bf16_alias::CudaBf16Aliases,
};
use rand::RngCore;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use triblespace::prelude::Id;
#[path = "wemm_behavior/sources.rs"]
mod sources;

const FIXTURE: &str = "51cd0cc643361a6c9839a8b6b5eeb203a482b8475c2202d2a02e8e6c69563b2c";
const HF: &str = "b7822d387cff5c05edd54773bc8d6c66dea4acf10548b92e1ec0c84b85136598";
const NATIVE: &str = "550e85285d2953061fcdcf975556abd6d436d8864aee4f34ac713267ad0f2047";
const CONFIG: &str = "34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004";
const BASE_CARGO: &str = "f7231f76b2eb48dcff1f19c101411c4386f87cd586536cab0cfd068124cf0653";
const CHECKPOINT: &str = "b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4";
const ROOT: &str = "DC023833AD7E18F1E400DE22A25055CC";
const EMBEDDING: &str = "4c41edbf8c6c2f95557a5c1d7ef153a77044c866cf8bfa7182b5003ab8f503cc";

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn pinned_json(path: &Path, expected: &str) -> Result<Value> {
    ensure!(
        fs::metadata(path)?.len() < 16 * 1024 * 1024,
        "oversized evidence"
    );
    let bytes = fs::read(path)?;
    ensure!(
        hash(&bytes) == expected,
        "frozen evidence SHA differs: {}",
        path.display()
    );
    Ok(serde_json::from_slice(&bytes)?)
}
fn write_new(path: &Path, report: &Value) -> Result<()> {
    let mut out = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut out, report)?;
    out.write_all(b"\n")?;
    Ok(())
}
fn exact_item(fixture: &Value) -> Result<Vec<u32>> {
    ensure!(
        fixture["schema"] == "wemm-prepared-behavior-v1",
        "fixture schema"
    );
    let items = fixture["items"].as_array().context("items")?;
    let matching: Vec<_> = items.iter().filter(|v| v["id"] == "bloom-long").collect();
    ensure!(
        matching.len() == 1 && matching[0]["modality"] == "text",
        "exact text item"
    );
    let ids: Vec<u32> = serde_json::from_value(matching[0]["ids"].clone())?;
    ensure!(ids.len() == 116, "exact 116-token item");
    prepared::validate_ids(&ids).map_err(anyhow::Error::msg)?;
    Ok(ids)
}
fn baseline_bits(native: &Value) -> Result<Vec<u16>> {
    ensure!(
        native["engine"] == "native-CUDA-BF16",
        "native evidence engine"
    );
    let entries: Vec<_> = native["embeddings"]
        .as_array()
        .context("embeddings")?
        .iter()
        .filter(|v| v["id"] == "bloom-long")
        .collect();
    ensure!(
        entries.len() == 2 && entries[0]["bits"] == entries[1]["bits"],
        "baseline repeat"
    );
    let bits: Vec<u16> = serde_json::from_value(entries[0]["bits"].clone())?;
    ensure!(bits.len() == 4096, "baseline extent");
    let bytes: Vec<_> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    ensure!(hash(&bytes) == EMBEDDING, "baseline bytes");
    Ok(bits)
}
#[derive(Deserialize)]
struct Weight {
    name: String,
    shape: Vec<u64>,
    sha256: String,
}

fn invoke(model: &PreparedMultimodal, ids: &[u32], label: &str, summarize: bool) -> Result<Value> {
    let start = Instant::now();
    let embedding = model
        .embed_text(ids)
        .map_err(anyhow::Error::msg)?
        .endpoint
        .embedding;
    let dispatch_ms = start.elapsed().as_secs_f64() * 1000.;
    ensure!(
        embedding.dtype == DType::BF16 && embedding.meta.shape().as_slice() == [1, 4096],
        "embedding shape"
    );
    let readback = Instant::now();
    let bytes = embedding
        .client
        .read_one(embedding.handle.clone())
        .map_err(|e| anyhow::anyhow!("read: {e:?}"))?
        .to_vec();
    let readback_wait_ms = readback.elapsed().as_secs_f64() * 1000.;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
    ensure!(bytes.len() == 8192, "embedding extent");
    let bits: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    ensure!(
        bits.iter().all(|b| b & 0x7f80 != 0x7f80),
        "nonfinite embedding"
    );
    if summarize {
        // Existing API: synchronizes, enqueues summary, resets accumulated
        // logger statistics. It does NOT toggle profiling or acknowledge logger
        // drain. Driver requires both complete summary footers in retained log.
        cubecl::future::block_on(embedding.client.sync())
            .map_err(|e| anyhow::anyhow!("profile summary sync: {e:?}"))?;
    }
    Ok(json!({"label":label,"bits":bits,"sha256":hash(&bytes),
        "elapsed_ms":elapsed_ms,"forward_dispatch_ms":dispatch_ms,"readback_wait_ms":readback_wait_ms}))
}

fn run(args: &[std::ffi::OsString]) -> Result<()> {
    let mode = args
        .first()
        .and_then(|v| v.to_str())
        .context("ordinary or profiled")?;
    let profiling = match mode {
        "ordinary" => false,
        "profiled" => true,
        _ => anyhow::bail!("unknown mode"),
    };
    ensure!(
        args.len() == if profiling { 8 } else { 7 },
        "MODE MODEL CONFIG FIXTURE HF NATIVE NEW_REPORT [ORDINARY_REPORT]"
    );
    ensure!(!Path::new(&args[6]).exists(), "preserve prior output");
    // Before ANY CUDA client, and without inheriting ambient CubeCL config.
    // ServerLogger captures this level; it cannot later be switched in-place.
    let mut runtime_config = CubeClRuntimeConfig::default();
    if profiling {
        runtime_config.profiling.logger.level = ProfilingLogLevel::Full;
        runtime_config.profiling.logger.stderr = true;
    }
    CubeClRuntimeConfig::set(runtime_config);
    let mut nonce = [0; 32];
    rand::rngs::OsRng
        .try_fill_bytes(&mut nonce)
        .map_err(|e| anyhow::anyhow!("nonce: {e}"))?;
    let process = json!({"pid":std::process::id(),"run_nonce":hash(&nonce),
        "started_unix_ns":SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos().to_string()});
    let fixture = pinned_json(Path::new(&args[3]), FIXTURE)?;
    let hf = pinned_json(Path::new(&args[4]), HF)?;
    let native = pinned_json(Path::new(&args[5]), NATIVE)?;
    ensure!(
        hf["engine"] == "HF-CUDA-BF16"
            && hf["fixture_sha256"] == FIXTURE
            && hf["checkpoint_sha256"] == CHECKPOINT,
        "HF evidence contract"
    );
    let ids = exact_item(&fixture)?;
    let expected_bits = baseline_bits(&native)?;
    // Cargo's sole change registers this diagnostic opt-in feature/bin. Every
    // existing model/behavior census entry must remain byte-identical to c8e3.
    let mut census = serde_json::Map::new();
    for &(path, bytes) in sources::SOURCES {
        let actual = hash(bytes);
        let expected = if path == "Cargo.toml" {
            BASE_CARGO
        } else {
            actual.as_str()
        };
        ensure!(
            fixture["native_sources"][path] == expected && hf["native_sources"][path] == expected,
            "base source changed: {path}"
        );
        census.insert(path.into(), json!(actual));
    }
    census.insert(
        "src/bin/wemm_single_profile.rs".into(),
        json!(hash(include_bytes!("wemm_single_profile.rs"))),
    );
    let source = Value::Object(census);
    let ordinary = if profiling {
        ensure!(
            fs::metadata(&args[7])?.len() < 16 * 1024 * 1024,
            "oversized ordinary report"
        );
        let bytes = fs::read(&args[7])?;
        let report: Value = serde_json::from_slice(&bytes)?;
        ensure!(
            report["schema"] == "wemm-single-profile-v1"
                && report["mode"] == "ordinary"
                && report["engine"] == "native-CUDA-BF16-diagnostic"
                && report["baseline_native_sha256"] == NATIVE
                && report["native_sources"] == source
                && report["fixture_sha256"] == FIXTURE
                && report["hf_report_sha256"] == HF
                && report["model_root"] == ROOT
                && report["all_outputs_match_baseline"] == true,
            "ordinary report contract"
        );
        ensure!(
            report["process"]["run_nonce"] != process["run_nonce"],
            "separate process evidence"
        );
        Some((hash(&bytes), report))
    } else {
        None
    };
    let config_bytes = fs::read(&args[2])?;
    ensure!(hash(&config_bytes) == CONFIG, "checkpoint config");
    let config = Qwen3_5Config::from_json(std::str::from_utf8(&config_bytes)?)
        .map_err(anyhow::Error::msg)?;
    let weights: Vec<Weight> = serde_json::from_value(hf["weights"].clone())?;
    ensure!(weights.len() == 759, "759 reference roles");
    let mut names = std::collections::HashSet::new(); // finite probe validation only
    ensure!(
        weights.iter().all(|w| names.insert(&w.name)),
        "duplicate reference role"
    );
    let bind_start = Instant::now();
    let observation = mary::persist::read_model_pile(Path::new(&args[1]))?;
    let root = Id::from_hex(ROOT).context("fixed opaque root")?;
    let mut aliases =
        CudaBf16Aliases::new(CudaDevice { index: 0 }, 762).map_err(anyhow::Error::msg)?;
    let mut selected = Vec::with_capacity(759); // finite diagnostic evidence, not runtime catalogue
    // SAFETY: operator-owned source pile remains an immutable prefix, including
    // preceding mapped pages, until CUDA teardown. No file/tensor copying.
    let model = unsafe { PreparedMultimodal::from_pile(&observation.facts, &observation.store,
        root, &config, &mut aliases, |name, handle, shape, bytes| {
            let expected = weights.iter().find(|w| w.name == name).ok_or_else(|| format!("unknown role {name}"))?;
            let sha256 = hash(bytes);
            if expected.shape != shape || expected.sha256 != sha256 { return Err(format!("checkpoint role differs: {name}")); }
            selected.push(json!({"name":name,"handle":handle.iter().map(|b|format!("{b:02x}")).collect::<String>(),
                "shape":shape,"sha256":sha256}));
            Ok(())
        }) }.map_err(anyhow::Error::msg)?;
    ensure!(selected.len() == 759, "759 selected roles");
    let bind_ms = bind_start.elapsed().as_secs_f64() * 1000.;
    let mut calls = vec![invoke(
        &model,
        &ids,
        if profiling {
            "profiled-warmup"
        } else {
            "ordinary-warmup"
        },
        profiling,
    )?];
    calls.push(invoke(
        &model,
        &ids,
        if profiling {
            "profiled-measured"
        } else {
            "ordinary-measured"
        },
        profiling,
    )?);
    if !profiling {
        calls.push(invoke(&model, &ids, "ordinary-repeat", false)?);
    }
    let all_match = calls.iter().all(|v| v["bits"] == json!(expected_bits));
    let cross_process_match = ordinary.as_ref().map(|(_, report)| {
        report["calls"].as_array().is_some_and(|other| {
            other.len() == 3 && other.iter().all(|v| v["bits"] == calls[1]["bits"])
        })
    });
    let report = json!({"schema":"wemm-single-profile-v1","engine":"native-CUDA-BF16-diagnostic", "mode":mode,
        "item":"bloom-long","token_count":ids.len(),"ids":ids,"fixture_sha256":FIXTURE,"hf_report_sha256":HF,
        "baseline_native_sha256":NATIVE,"model_root":ROOT,"checkpoint_sha256":CHECKPOINT,"native_sources":source,
        "selected_roles":selected,"alias_registrations":aliases.stats().registrations,"bind_ms":bind_ms,
        "calls":calls,"all_outputs_match_baseline":all_match,"ordinary_report_sha256":ordinary.as_ref().map(|v| &v.0),
        "cross_process_byte_exact":cross_process_match,"process":process,"profiling_enabled_at_process_init":profiling,
        "timing":"ordinary dispatch+readback wait; profiled invokes synchronize around EACH kernel and use SYSTEM timestamps; warmup summary reset before measured; no uncontaminated GPU-event timing",
        "scope":"one fixed 116-token input, fresh state, no kernel/model changes or model admission; no runtime profiling toggle available"});
    write_new(Path::new(&args[6]), &report)?;
    ensure!(
        all_match && cross_process_match != Some(false),
        "embedding mismatch; diagnostic report preserved"
    );
    Ok(())
}
fn main() -> Result<()> {
    run(&std::env::args_os().skip(1).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        let mut ids = vec![1; 116];
        ids[115] = prepared::EMBEDDING_TOKEN;
        json!({"schema":"wemm-prepared-behavior-v1","items":[{"id":"bloom-long","modality":"text","ids":ids}]})
    }
    #[test]
    fn fixed_item_rejects_wrong_count_modality_and_duplicate() {
        assert_eq!(exact_item(&fixture()).unwrap().len(), 116);
        let mut f = fixture();
        f["items"][0]["ids"] = json!(vec![1; 115]);
        assert!(exact_item(&f).is_err());
        let mut f = fixture();
        f["items"][0]["modality"] = json!("image");
        assert!(exact_item(&f).is_err());
        let mut f = fixture();
        let item = f["items"][0].clone();
        f["items"].as_array_mut().unwrap().push(item);
        assert!(exact_item(&f).is_err());
    }
    #[test]
    fn baseline_rejects_wrong_engine_or_missing_repeat() {
        assert!(baseline_bits(&json!({"engine":"HF-CUDA-BF16","embeddings":[]})).is_err());
        assert!(baseline_bits(&json!({"engine":"native-CUDA-BF16","embeddings":[]})).is_err());
    }
}
