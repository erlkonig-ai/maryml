//! Fixed prepared-fixture behavioral probe. No reference-coordinate gate.
//! `pack PREPARED.json NEW_INPUTS.pile NEW_FIXTURE.json`
//! `run MODEL.pile ROOT CONFIG.json FIXTURE.json INPUTS.pile HF.json NEW_REPORT.json`
//! All model math is native CUDA; host work is metadata and byte transport.
use anyhow::{Context, Result, ensure};
use burn::tensor::DType;
use cubecl::cuda::CudaDevice;
use mary::{
    models::qwen3_5::{
        config::Qwen3_5Config, multimodal::PreparedMultimodal, multimodal_layout::ImagePlan,
        prepared, vision_geometry::Grid,
    },
    nn::cuda_bf16_alias::CudaBf16Aliases,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use triblespace::{
    core::{
        blob::{
            Blob,
            encodings::tensor::{Tensor, elements::BF16},
        },
        inline::encodings::hash::{Blake3, Handle, Hash},
        repo::pile::PileFile,
    },
    prelude::*,
};
#[path = "wemm_behavior/sources.rs"]
mod sources;
const CONFIG: &str = "34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004";
const CHECKPOINT: &str = "b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4";
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn read_json(path: &Path) -> Result<(Vec<u8>, Value)> {
    ensure!(
        fs::metadata(path)?.len() < 16 * 1024 * 1024,
        "oversized fixture/report"
    );
    let bytes = fs::read(path)?;
    let value = serde_json::from_slice(&bytes)?;
    Ok((bytes, value))
}
fn write_new(path: &Path, value: &Value) -> Result<()> {
    let mut out = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    Ok(())
}
fn source_identity() -> BTreeMap<String, String> {
    sources::SOURCES
        .iter()
        .map(|(p, b)| (p.to_string(), hash(b)))
        .collect()
}
#[derive(Deserialize)]
struct Item {
    id: String,
    modality: String,
    ids: Vec<u32>,
    #[serde(default)]
    pixels_path: Option<String>,
    #[serde(default)]
    pixels_sha256: Option<String>,
    #[serde(default)]
    tensor_handle: Option<String>,
}
fn items(value: &Value) -> Result<Vec<Item>> {
    ensure!(
        value["schema"] == "wemm-prepared-behavior-v1",
        "fixture schema"
    );
    let items: Vec<Item> = serde_json::from_value(value["items"].clone())?;
    ensure!(items.len() == 11, "fixed eleven-input probe only");
    let mut ids = std::collections::HashSet::new(); // one-operation validation
    for item in &items {
        ensure!(ids.insert(&item.id), "duplicate fixture id");
        match item.modality.as_str() {
            "text" => prepared::validate_ids(&item.ids).map_err(anyhow::Error::msg)?,
            "image" => {
                ImagePlan::new(
                    &item.ids,
                    Grid {
                        frames: 1,
                        height: 16,
                        width: 16,
                    },
                    256,
                    64,
                )
                .map_err(anyhow::Error::msg)?;
            }
            _ => anyhow::bail!("unsupported fixture modality"),
        }
        ensure!(
            item.ids
                .iter()
                .filter(|&&id| id == prepared::EMBEDDING_TOKEN)
                .count()
                == 1,
            "exactly one final embedding token required"
        );
    }
    ensure!(
        items.iter().filter(|i| i.modality == "image").count() == 3,
        "three images required"
    );
    Ok(items)
}
fn pack(prepared: &Path, pile_path: &Path, output: &Path) -> Result<()> {
    let (bytes, mut manifest) = read_json(prepared)?;
    ensure!(!output.exists() && !output.is_symlink(), "output exists");
    let decoded = items(&manifest)?;
    ensure!(
        manifest["native_sources"] == serde_json::to_value(source_identity())?,
        "prepared source identity"
    );
    // Validate every input before creating a destination; no Tensor math here.
    let mut payloads = Vec::new();
    for item in decoded.iter().filter(|i| i.modality == "image") {
        let path = Path::new(item.pixels_path.as_deref().context("pixels path")?);
        ensure!(
            fs::metadata(path)?.len() == 256 * 1536 * 2,
            "prepared BF16 extent"
        );
        let raw = fs::read(path)?;
        ensure!(
            Some(hash(&raw)) == item.pixels_sha256,
            "prepared pixels changed"
        );
        payloads.push((item.id.clone(), raw));
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(pile_path)?;
    let mut pile = Pile::open(pile_path)?;
    for (id, raw) in payloads {
        let blob = mary::leaf::leaf_blob::<BF16, 2>([256, 1536], raw.into())?;
        let handle = pile.put(blob)?;
        let row = manifest["items"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|r| r["id"] == id)
            .unwrap();
        row["tensor_handle"] = Value::String(Hash::<Blake3>::to_hex(&Handle::to_hash(handle)));
    }
    pile.close()?;
    manifest["prepared_manifest_sha256"] = json!(hash(&bytes));
    manifest["input_pile_sha256"] = json!(hash(&fs::read(pile_path)?));
    manifest["input_pile_bytes"] = json!(fs::metadata(pile_path)?.len());
    write_new(output, &manifest)
}
#[derive(Deserialize)]
struct Weight {
    name: String,
    shape: Vec<u64>,
    sha256: String,
}
#[derive(Serialize)]
struct Selected {
    name: String,
    handle: String,
    shape: Vec<u64>,
    sha256: String,
}
fn run(args: &[std::ffi::OsString]) -> Result<()> {
    ensure!(
        args.len() == 7,
        "run MODEL ROOT CONFIG FIXTURE INPUTS HF NEW_REPORT"
    );
    let started_unix_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_nanos()
        .to_string();
    let mut nonce = [0u8; 32];
    rand::rngs::OsRng
        .try_fill_bytes(&mut nonce)
        .map_err(|e| anyhow::anyhow!("run nonce: {e}"))?;
    let process = json!({"pid":std::process::id(),"started_unix_ns":started_unix_ns,
        "run_nonce":nonce.iter().map(|b|format!("{b:02x}")).collect::<String>()});
    let root = Id::from_hex(args[1].to_str().context("root UTF8")?).context("opaque root")?;
    let config_bytes = fs::read(&args[2])?;
    ensure!(hash(&config_bytes) == CONFIG, "checkpoint config identity");
    let config = Qwen3_5Config::from_json(std::str::from_utf8(&config_bytes)?)
        .map_err(anyhow::Error::msg)?;
    let (fixture_bytes, fixture) = read_json(Path::new(&args[3]))?;
    let inputs = items(&fixture)?;
    let (hf_bytes, hf) = read_json(Path::new(&args[5]))?;
    ensure!(
        hf["engine"] == "HF-CUDA-BF16",
        "expected actual HF reference report"
    );
    let source = serde_json::to_value(source_identity())?;
    ensure!(
        fixture["native_sources"] == source && hf["native_sources"] == source,
        "source identity"
    );
    ensure!(
        hf["fixture_sha256"] == hash(&fixture_bytes) && hf["checkpoint_sha256"] == CHECKPOINT,
        "HF fixture/checkpoint identity"
    );
    ensure!(
        fs::metadata(&args[4])?.len() < 4 * 1024 * 1024,
        "oversized input pile"
    );
    ensure!(
        fixture["input_pile_sha256"] == hash(&fs::read(&args[4])?),
        "fixture pile identity"
    );
    ensure!(!Path::new(&args[6]).exists(), "preserve old output");
    let weights: Vec<Weight> = serde_json::from_value(hf["weights"].clone())?;
    ensure!(weights.len() == 759, "HF must bind all759 roles");
    let mut names = std::collections::HashSet::new();
    ensure!(
        weights.iter().all(|w| names.insert(&w.name)),
        "duplicate HF role"
    );
    let started = Instant::now();
    let model = mary::persist::read_model_pile(Path::new(&args[0]))?;
    let mut aliases =
        CudaBf16Aliases::new(CudaDevice { index: 0 }, 762).map_err(anyhow::Error::msg)?;
    let mut selected = Vec::with_capacity(759);
    // SAFETY: both operator-owned piles remain genuine immutable prefixes
    // including preceding mmap pages until process/CUDA teardown.
    let model = unsafe {
        PreparedMultimodal::from_pile(
            &model.facts,
            &model.store,
            root,
            &config,
            &mut aliases,
            |name, handle, shape, bytes| {
                let expected = weights
                    .iter()
                    .find(|w| w.name == name)
                    .ok_or_else(|| format!("unknown role {name}"))?;
                let sha256 = hash(bytes);
                if expected.shape != shape || expected.sha256 != sha256 {
                    return Err(format!("checkpoint role differs:{name}"));
                }
                selected.push(Selected {
                    name: name.into(),
                    handle: handle.iter().map(|b| format!("{b:02x}")).collect(),
                    shape: shape.to_vec(),
                    sha256,
                });
                Ok(())
            },
        )
    }
    .map_err(anyhow::Error::msg)?;
    ensure!(selected.len() == 759, "native roles");
    let mut input_store = Pile::new(PileFile::open_read_only(Path::new(&args[4]))?);
    let input_snapshot = input_store.snapshot()?;
    let mut pixels = Vec::with_capacity(3); // finite call fixture, not a model catalogue
    for item in inputs.iter().filter(|i| i.modality == "image") {
        let handle = Handle::<Tensor<BF16, 2>>::from_hash(Hash::<Blake3>::from_hex(
            item.tensor_handle
                .as_deref()
                .context("typed tensor handle")?,
        )?);
        let blob: Blob<Tensor<BF16, 2>> = input_snapshot.get(handle)?;
        let tensor = unsafe { aliases.bind_pile_leaf(blob.clone()) }.map_err(anyhow::Error::msg)?;
        ensure!(
            tensor.meta.shape().as_slice() == [256, 1536] && tensor.dtype == DType::BF16,
            "fixture tensor shape"
        );
        let view = mary::leaf::read_leaf(blob)?; // binder checked header arithmetic first
        ensure!(
            Some(hash(view.payload())) == item.pixels_sha256,
            "typed input differs from HF prepared bytes"
        );
        pixels.push((item.id.clone(), tensor));
    }
    let bind_ms = started.elapsed().as_secs_f64() * 1000.;
    let mut vectors = Vec::with_capacity(22);
    for reverse in [false, true] {
        let order: Vec<usize> = if reverse {
            (0..inputs.len()).rev().collect()
        } else {
            (0..inputs.len()).collect()
        };
        for index in order {
            let item = &inputs[index];
            let start = Instant::now();
            let embedding = if item.modality == "text" {
                model
                    .embed_text(&item.ids)
                    .map_err(anyhow::Error::msg)?
                    .endpoint
                    .embedding
            } else {
                let pixel = &pixels.iter().find(|(id, _)| id == &item.id).unwrap().1;
                model
                    .embed_image(
                        &item.ids,
                        pixel,
                        Grid {
                            frames: 1,
                            height: 16,
                            width: 16,
                        },
                    )
                    .map_err(anyhow::Error::msg)?
                    .endpoint
                    .embedding
            };
            let forward_dispatch_ms = start.elapsed().as_secs_f64() * 1000.;
            ensure!(
                embedding.dtype == DType::BF16 && embedding.meta.shape().as_slice() == [1, 4096],
                "embedding shape"
            );
            let readback_start = Instant::now();
            let raw = embedding
                .client
                .read_one(embedding.handle.clone())
                .map_err(|e| anyhow::anyhow!("read:{e:?}"))?
                .to_vec();
            let readback_wait_ms = readback_start.elapsed().as_secs_f64() * 1000.;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
            ensure!(raw.len() == 8192, "embedding bytes");
            let bits: Vec<u16> = raw
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect();
            ensure!(
                bits.iter().all(|b| b & 0x7f80 != 0x7f80),
                "nonfinite BF16 embedding"
            );
            vectors.push(json!({"id":item.id,"pass":if reverse {"reverse"}else{"forward"},"bits":bits,"sha256":hash(&raw),"elapsed_ms":elapsed_ms,
                "forward_dispatch_ms":forward_dispatch_ms,"readback_wait_ms":readback_wait_ms}));
            println!(
                "{} {} {:.1}ms",
                if reverse { "reverse" } else { "forward" },
                item.id,
                elapsed_ms
            );
        }
    }
    let reproducible = inputs.iter().all(|item| {
        vectors
            .iter()
            .filter(|v| v["id"] == item.id)
            .map(|v| &v["bits"])
            .collect::<Vec<_>>()
            .windows(2)
            .all(|w| w[0] == w[1])
    });
    write_new(
        Path::new(&args[6]),
        &json!({"schema":"wemm-behavior-embeddings-v1","engine":"native-CUDA-BF16","checkpoint_sha256":CHECKPOINT,
        "fixture_sha256":hash(&fixture_bytes),"hf_report_sha256":hash(&hf_bytes),"native_sources":source,"selected_roles":selected,
        "model_root":format!("{root:?}"),"alias_registrations":aliases.stats().registrations,"bind_ms":bind_ms,"embeddings":vectors,
        "reverse_order_byte_exact":reproducible,"process":process,
        "timing":"bind_ms includes selection, payload hashing and input aliasing; elapsed_ms is each forward plus readback wait; forward_dispatch_ms is host call duration, not GPU-only kernel time",
        "scope":"B1 serial calls, fresh state; no vectorized-batch or cross-host claim; coordinate parity diagnostic only"}),
    )?;
    ensure!(
        reproducible,
        "B1 byte reproducibility failure; report preserved"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        let mut value: Value =
            serde_json::from_str(include_str!("../../scripts/wemm_behavior_fixture.json")).unwrap();
        value["schema"] = json!("wemm-prepared-behavior-v1");
        for row in value["items"].as_array_mut().unwrap() {
            row["ids"] = if row["modality"] == "image" {
                let mut ids = vec![248053];
                ids.extend([248056; 64]);
                ids.extend([248054, 248077]);
                json!(ids)
            } else {
                json!([248077])
            };
        }
        value
    }

    #[test]
    fn accepts_the_fixed_eleven_input_layout() {
        assert!(items(&fixture()).is_ok());
    }
    #[test]
    fn rejects_duplicate_item_identity() {
        let mut f = fixture();
        f["items"][1]["id"] = f["items"][0]["id"].clone();
        assert!(items(&f).is_err());
    }
    #[test]
    fn rejects_unknown_modality() {
        let mut f = fixture();
        f["items"][0]["modality"] = json!("audio");
        assert!(items(&f).is_err());
    }
    #[test]
    fn refuses_overlength_instead_of_truncating() {
        let mut f = fixture();
        let mut ids = vec![100; 256];
        ids.push(248077);
        f["items"][0]["ids"] = json!(ids);
        assert!(items(&f).is_err());
    }
    #[test]
    fn requires_exactly_one_terminal_embedding_token() {
        let mut f = fixture();
        f["items"][0]["ids"] = json!([248077, 248077]);
        assert!(items(&f).is_err());
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.first().and_then(|a| a.to_str()) {
        Some("pack") if args.len() == 4 => pack(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
        ),
        Some("run") => run(&args[1..]),
        _ => anyhow::bail!(
            "pack PREPARED NEW_PILE NEW_FIXTURE | run MODEL ROOT CONFIG FIXTURE INPUTS HF NEW_REPORT"
        ),
    }
}
