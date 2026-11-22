//! Import the pinned Inkling 2 Mini release into an explicit existing model
//! collection, and verify an imported root from the pile alone.
//!
//!   inkling2_import import --checkpoint <dir> --pile <pile> --key <key> \
//!       --collection blake3:HEX
//!   inkling2_import verify --pile <pile> --collection blake3:HEX --root HEX \
//!       [--source <dir>]
//!
//! `import` checks the signer against the collection's WRITE policy before
//! writing anything, then stores every tensor at its source dtype, shape and
//! bytes (`models::inkling2_loop::import`). It is idempotent: blobs already in
//! the pile are not appended again, and when every fact of the candidate is
//! already in the collection no commit is written.
//!
//! `verify` is a fresh reader given only the pile, the collection and the
//! root. It resolves every component, rebuilds both safetensors files from
//! the pile and checks their sha256 against the manifest and the release's
//! LFS digests, checks every document against the manifest, the config facts
//! against `config.json`, the chat template, and the tokenizer graph against
//! `tokenizer.json`. With `--source` it then also compares a few real
//! tensors byte for byte with the checkpoint files on disk.
use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use mary::model_collection::{self, ModelCollection};
use mary::models::inkling2_loop::{import, pile::MiniRoot, release};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;
use triblespace::prelude::*;

#[derive(Parser)]
#[command(about = "Import Inkling 2 Mini into a model collection, or verify an imported root")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Import {
        #[arg(long)]
        checkpoint: PathBuf,
        /// Existing pile holding the target collection's descriptor.
        #[arg(long)]
        pile: PathBuf,
        /// Existing signing key the collection's WRITE policy admits.
        #[arg(long)]
        key: PathBuf,
        /// Exact target collection descriptor handle (`blake3:HEX`).
        #[arg(long)]
        collection: String,
    },
    Verify {
        #[arg(long)]
        pile: PathBuf,
        #[arg(long)]
        collection: String,
        #[arg(long)]
        root: String,
        /// Optional: also compare real tensors with these checkpoint files.
        #[arg(long)]
        source: Option<PathBuf>,
    },
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn handle_hex(handle: &triblespace::core::collection::CollectionHandle) -> String {
    hex(handle.raw)
}

fn run_import(checkpoint: &Path, dest: &Path, key: &Path, collection: &str) -> Result<serde_json::Value> {
    let key = signing_key_file::load_existing(key).context("load existing signing key")?;
    let handle = model_collection::parse_collection_handle(collection)?;
    ensure!(
        std::fs::metadata(dest).with_context(|| format!("existing destination {dest:?}"))?.is_file(),
        "destination must be an existing pile file"
    );
    let before = std::fs::metadata(dest)?.len();
    let mut pile = Pile::open(dest).map_err(|e| anyhow::anyhow!("open {dest:?}: {e:?}"))?;
    let result = (|| -> Result<serde_json::Value> {
        pile.refresh().context("refresh destination")?;
        let target: ModelCollection = {
            let snapshot = pile.snapshot().context("freeze destination")?;
            model_collection::open_model_collection(&snapshot, handle)?
        };
        // Before any weight is written: a refused signer costs nothing.
        model_collection::require_model_writer(&mut pile, &key, target)?;
        let started = Instant::now();
        let candidate = import::ingest_checkpoint(&mut pile, checkpoint)?;
        let ingest_seconds = started.elapsed().as_secs_f64();
        let ids = candidate.artifacts;

        // Idempotence: a candidate the collection already holds whole is not
        // committed again.
        let held = {
            let snapshot = pile.snapshot().context("freeze before publication")?;
            model_collection::snapshot_model_collection_for(&snapshot, target)?.facts().clone()
        };
        let facts: TribleSet = candidate.fragment.facts().clone();
        let missing = facts.difference(&held).len();
        let commit = if missing == 0 {
            eprintln!("[inkling2] the collection already holds all {} facts; no commit", facts.len());
            None
        } else {
            let commit = model_collection::publish_model_fragment_into(&mut pile, &key, target, candidate.fragment)?;
            Some(triblespace::core::collection::CollectionRecord::Commit(commit).fingerprint().to_string())
        };
        Ok(serde_json::json!({
            "model_root": format!("{:X}", ids.model_root),
            "drafter_root": format!("{:X}", ids.drafter_root),
            "config_root": format!("{:X}", ids.config_root),
            "tokenizer_root": format!("{:X}", ids.tokenizer_root),
            "manifest_root": format!("{:X}", ids.manifest_root),
            "collection": handle_hex(&handle),
            "facts": facts.len(),
            "facts_new_to_collection": missing,
            "commit": commit,
            "ingest_seconds": ingest_seconds,
            "manifest": candidate.manifest,
        }))
    })();
    let close = pile.close().map_err(|e| anyhow::anyhow!("close {dest:?}: {e:?}"));
    let mut receipt = match (result, close) {
        (Ok(receipt), Ok(())) => receipt,
        (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
    };
    let after = std::fs::metadata(dest)?.len();
    receipt["pile_bytes_before"] = serde_json::json!(before);
    receipt["pile_bytes_after"] = serde_json::json!(after);
    Ok(receipt)
}

/// The byte range of `name` in a safetensors file, read from its header.
fn source_range(path: &Path, name: &str) -> Result<(u64, u64)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header)?;
    let v: serde_json::Value = serde_json::from_slice(&header)?;
    let o = v[name]["data_offsets"].as_array().with_context(|| format!("{name} not in {path:?}"))?;
    Ok((8 + n + o[0].as_u64().unwrap(), 8 + n + o[1].as_u64().unwrap()))
}

fn run_verify(path: &Path, collection: &str, root: &str, source: Option<&Path>) -> Result<serde_json::Value> {
    let handle = model_collection::parse_collection_handle(collection)?;
    let root = Id::from_hex(root).context("--root must be a 32-hex id")?;
    let started = Instant::now();
    let mini = MiniRoot::open(path, handle, root)?;
    let open_seconds = started.elapsed().as_secs_f64();
    let (model_tensors, drafter_tensors) = mini.counts();
    eprintln!("[verify] opened {root:X}: {model_tensors} model + {drafter_tensors} drafter tensors, {} documents, {open_seconds:.1}s",
        mini.document_names().len());

    let manifest = mini.manifest()?;
    ensure!(manifest["revision"] == release::REVISION, "manifest revision {}", manifest["revision"]);
    let recorded = |file: &str| -> Result<(u64, String)> {
        let entry = manifest["files"]
            .as_array()
            .context("manifest has no files")?
            .iter()
            .find(|e| e["path"] == file)
            .with_context(|| format!("manifest lacks {file}"))?;
        Ok((entry["bytes"].as_u64().context("bytes")?, entry["sha256"].as_str().context("sha256")?.to_owned()))
    };

    // Both safetensors files, rebuilt from the pile and hashed.
    let mut files = Vec::new();
    for file in [release::MODEL_FILE, release::MTP_FILE] {
        let t = Instant::now();
        let mut sha = Sha256::new();
        let bytes = mini.reassemble(file, |chunk| sha.update(chunk))?;
        let digest = hex(sha.finalize());
        let (size, want) = recorded(file)?;
        let lfs = release::RELEASE.iter().find(|(p, _, _)| *p == file).and_then(|(_, _, l)| *l).context("pinned")?;
        ensure!(bytes == size, "{file}: rebuilt {bytes} bytes, manifest {size}");
        ensure!(digest == want && digest == lfs, "{file}: rebuilt sha256 {digest}, manifest {want}, release {lfs}");
        eprintln!("[verify] {file}: rebuilt {bytes} bytes from the pile, sha256 {digest} = manifest = release ({:.1}s)", t.elapsed().as_secs_f64());
        files.push(serde_json::json!({"path": file, "bytes": bytes, "sha256": digest, "source": "rebuilt from header document + leaves"}));
    }
    // Every other release file, as a document.
    for &(file, _, lfs) in release::RELEASE {
        if file == release::MODEL_FILE || file == release::MTP_FILE {
            continue;
        }
        let text = mini.document(file)?;
        let digest = hex(Sha256::digest(text.as_bytes()));
        let (size, want) = recorded(file)?;
        ensure!(text.len() as u64 == size && digest == want, "{file}: document {} bytes sha256 {digest}, manifest {size} {want}", text.len());
        if let Some(lfs) = lfs {
            ensure!(digest == lfs, "{file}: document sha256 {digest}, release {lfs}");
        }
        files.push(serde_json::json!({"path": file, "bytes": size, "sha256": digest, "source": "document"}));
    }

    // The root's parts against the documents they came from.
    let config_doc: serde_json::Value = serde_json::from_str(&mini.document("config.json")?)?;
    ensure!(mini.config()? == config_doc, "config facts differ from config.json");
    mary::models::inkling2_loop::Inkling2LoopConfig::from_json(&mini.document("config.json")?)
        .map_err(anyhow::Error::msg)?;
    ensure!(mini.chat_template()? == *mini.document("chat_template.jinja")?, "chat template differs from chat_template.jinja");
    let rebuilt = mary::tokenizer::build_tokenizer(mini.facts(), mini.store(), mini.tokenizer_root())
        .map_err(|e| anyhow::anyhow!("rebuild the tokenizer graph: {e}"))?;
    let file = tokenizers::Tokenizer::from_bytes(mini.document("tokenizer.json")?.as_bytes())
        .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
    let probe = "<|message_user|>Hello, Liora! 你好 🍑\n<|end_message|>";
    let a = rebuilt.encode(probe, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    let b = file.encode(probe, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    ensure!(a.get_ids() == b.get_ids(), "tokenizer graph encodes {:?}, tokenizer.json {:?}", a.get_ids(), b.get_ids());
    let same_serialization = rebuilt.to_string(false).ok() == file.to_string(false).ok();
    eprintln!("[verify] config, chat template and tokenizer agree ({} probe ids; serialization identical: {same_serialization})", a.get_ids().len());

    // A small real tensor, read from the pile.
    let (dtype, shape, bytes) = mini.tensor("model.llm.attn_res.mlp_queries.0").map_err(anyhow::Error::msg)?;
    let first: Vec<f32> = bytes[..16].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    eprintln!("[verify] model.llm.attn_res.mlp_queries.0: {} {shape:?}, first values {first:?}", dtype.name());

    let mut compared = Vec::new();
    if let Some(source) = source {
        use std::io::{Read, Seek, SeekFrom};
        for (file, name) in [
            (release::MODEL_FILE, "model.llm.attn_res.mlp_queries.0"),
            (release::MODEL_FILE, "model.llm.layers.0.attn_norm.weight"),
            (release::MODEL_FILE, "model.llm.layers.0.mlp.experts.w13_weight_scale"),
            (release::MODEL_FILE, "model.llm.layers.20.mlp.experts.w13_weight"),
            (release::MTP_FILE, "fuse_weight"),
        ] {
            let path = source.join(file);
            let (a, b) = source_range(&path, name)?;
            let (dtype, shape, stored) = mini.tensor(name).map_err(anyhow::Error::msg)?;
            ensure!(stored.len() as u64 == b - a, "{name}: pile {} bytes, source {}", stored.len(), b - a);
            // The first and last 64 KiB, read from the file itself.
            let window = (stored.len() as u64).min(64 << 10);
            let mut f = std::fs::File::open(&path)?;
            for start in [0, stored.len() as u64 - window] {
                let mut buf = vec![0u8; window as usize];
                f.seek(SeekFrom::Start(a + start))?;
                f.read_exact(&mut buf)?;
                if buf != stored[start as usize..(start + window) as usize] {
                    bail!("{name}: pile and source differ in bytes {start}..{}", start + window);
                }
            }
            eprintln!("[verify] {name}: {} {shape:?}, pile = source in its first and last {window} bytes", dtype.name());
            compared.push(serde_json::json!({"tensor": name, "dtype": dtype.name(), "shape": shape, "bytes": stored.len(), "compared_bytes": 2 * window}));
        }
    }

    Ok(serde_json::json!({
        "pile": path, "collection": handle_hex(&handle),
        "model_root": format!("{root:X}"), "drafter_root": format!("{:X}", mini.drafter_root()),
        "model_tensors": model_tensors, "drafter_tensors": drafter_tensors,
        "documents": mini.document_names(), "files": files,
        "tokenizer_probe_ids": a.get_ids().len(), "tokenizer_serialization_identical": same_serialization,
        "source_compared": compared, "open_seconds": open_seconds,
        "verify_seconds": started.elapsed().as_secs_f64(),
    }))
}

fn main() -> Result<()> {
    let receipt = match Args::parse().command {
        Command::Import { checkpoint, pile, key, collection } => run_import(&checkpoint, &pile, &key, &collection)?,
        Command::Verify { pile, collection, root, source } => run_verify(&pile, &collection, &root, source.as_deref())?,
    };
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
