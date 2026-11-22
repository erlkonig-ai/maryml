//! Import the published Inkling 2 Mini checkpoint into a model pile, exactly
//! as it was released.
//!
//! Every tensor of both safetensors files becomes a typed leaf with its
//! source dtype, its source shape and its source bytes: BF16 and F32 as
//! themselves, the packed NVFP4 codes and MXFP8 scales as `U8`, the NVFP4
//! block scales and MXFP8 values as `E4M3`. Nothing is repacked, decoded or
//! widened. How the runtime wants its experts laid out is a later, derived
//! root with this one as its parent, not a decision this importer makes.
//!
//! The model root's core is its members plus the parts that decide how text
//! reaches it ([`crate::ingest::RootParts`]): the tokenizer graph, the
//! `config.json` facts and the chat template. The drafter (`mtp.safetensors`)
//! is its own root, listed as a member of the main root and labelled
//! [`DRAFTER`], the way Breeze lists its codec. Every other file of the
//! release, and each safetensors file's own header, is a text document
//! labelled with its file name and the model root. So the pile alone can
//! rebuild both safetensors files byte for byte, and
//! [`super::pile::MiniRoot::reassemble`] does exactly that.
//!
//! The source is checked before the first tensor is written: the file set
//! and sizes against the pinned release inventory, the four LFS digests the
//! release publishes, the layout bijection ([`super::checkpoint::inspect`]),
//! and that each file's tensor ranges tile it from the header to the end.

use super::checkpoint::{self, parse_header};
use super::config::Inkling2LoopConfig;
use super::layout::Precisions;
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::path::Path;
use triblespace::core::{metadata, repo::pile::Pile};
use triblespace::prelude::*;

pub use super::release::*;

/// Ids a reader needs, and what was imported.
#[derive(Debug, Clone, Copy)]
pub struct Artifacts {
    pub model_root: Id,
    pub drafter_root: Id,
    pub config_root: Id,
    pub tokenizer_root: Id,
    pub manifest_root: Id,
}

pub struct Candidate {
    pub fragment: Fragment,
    pub artifacts: Artifacts,
    pub manifest: serde_json::Value,
}

fn mapped(path: &Path) -> Result<anybytes::Bytes> {
    let file = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
    // SAFETY: the caller keeps the checkpoint files immutable during import.
    // The digest is taken over this same mapping, so a changed file shows up
    // as a digest mismatch rather than as silently different leaves.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("map {path:?}"))?;
    Ok(anybytes::Bytes::from_source(mmap))
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// Map a safetensors dtype to the leaf element that stores it unchanged.
pub fn elem_of(dtype: Dtype) -> Option<crate::leaf::Elem> {
    use crate::leaf::Elem;
    match dtype {
        Dtype::BF16 => Some(Elem::Bf16),
        Dtype::F32 => Some(Elem::F32),
        Dtype::U8 => Some(Elem::U8),
        Dtype::F8_E4M3 => Some(Elem::E4m3),
        _ => None,
    }
}

/// One safetensors file: its header bytes, and checks that its tensors tile
/// it exactly, so the leaves plus the header are the whole file.
struct SourceFile {
    name: &'static str,
    bytes: anybytes::Bytes,
    header: String,
    tensors: usize,
}

fn open_safetensors(directory: &Path, name: &'static str) -> Result<SourceFile> {
    let bytes = mapped(&directory.join(name))?;
    ensure!(bytes.len() >= 8, "{name}: shorter than a safetensors length prefix");
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8usize.checked_add(n).context("header length overflow")?;
    ensure!(header_end <= bytes.len(), "{name}: header runs past the file");
    let header = std::str::from_utf8(&bytes[8..header_end])
        .with_context(|| format!("{name}: header is not UTF-8"))?
        .to_owned();
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut ranges: Vec<(usize, usize, String)> = Vec::new();
    for (key, view) in tensors.tensors() {
        ensure!(
            elem_of(view.dtype()).is_some(),
            "{name}:{key}: dtype {:?} has no leaf element; refusing rather than dropping it",
            view.dtype()
        );
        let start = view.data().as_ptr() as usize - bytes.as_ptr() as usize;
        ranges.push((start, start + view.data().len(), key));
    }
    ranges.sort();
    let mut at = header_end;
    for (start, end, key) in &ranges {
        ensure!(
            *start == at,
            "{name}:{key}: data starts at {start}, the previous range ended at {at}; \
             the leaves would not cover the file"
        );
        at = *end;
    }
    ensure!(
        at == bytes.len(),
        "{name}: tensors end at {at} of {} bytes",
        bytes.len()
    );
    Ok(SourceFile {
        name,
        bytes,
        header,
        tensors: ranges.len(),
    })
}

/// Store every tensor of `file` as a member module with its leaf.
fn ingest_file(pile: &mut Pile, file: &SourceFile) -> Result<(Vec<Id>, Fragment, u64)> {
    let tensors = SafeTensors::deserialize(&file.bytes)?;
    let mut names = tensors.names();
    names.sort();
    let kind = pile.put::<blobencodings::UTF8String, _>("tensor".to_owned())?;
    let (mut members, mut facts, mut parameters) = (Vec::new(), Fragment::empty(), 0u64);
    for name in names {
        let tensor = tensors.tensor(name)?;
        let elem = elem_of(tensor.dtype()).expect("checked by open_safetensors");
        let start = tensor.data().as_ptr() as usize - file.bytes.as_ptr() as usize;
        let payload = file.bytes.slice(start..start + tensor.data().len());
        let shape: Vec<u64> = tensor.shape().iter().map(|&x| x as u64).collect();
        parameters += shape.iter().product::<u64>();
        let what = format!("{}:{name}", file.name);
        let leaf = crate::leaf::put_leaf(pile, elem, &shape, payload, &what)?;
        let weight = leaf.root().context("leaf root absent")?;
        facts += leaf;
        let path = pile.put::<blobencodings::UTF8String, _>(name.to_owned())?;
        let member = entity! { _ @ crate::format::attrs::safetensor_path: path,
            crate::format::attrs::kind: kind, crate::format::attrs::weight: weight };
        members.push(member.root().context("member root absent")?);
        facts += member;
    }
    Ok((members, facts, parameters))
}

fn label(pile: &mut Pile, fragment: &mut Fragment, asset: Id, name: &str, model_root: Id) -> Result<()> {
    let name = pile.put::<blobencodings::UTF8String, _>(name.to_owned())?;
    *fragment += entity! { ExclusiveId::force_ref(&asset) @
        metadata::name: name, crate::format::attrs::model_root: model_root };
    Ok(())
}

/// Build an unpublished candidate. Admission, publication and durability are
/// the caller's. A failure can leave unreferenced blobs in the pile; it never
/// publishes a partial model. The source directory is only read.
pub fn ingest_checkpoint(pile: &mut Pile, directory: &Path) -> Result<Candidate> {
    pile.refresh().context("refresh model pile")?;

    // ── the release inventory, before anything else ────────────────────────
    let mut present = Vec::new();
    for entry in walk(directory)? {
        present.push(entry);
    }
    present.sort();
    let mut wanted: Vec<String> = RELEASE.iter().map(|(p, _, _)| (*p).to_owned()).collect();
    wanted.sort();
    ensure!(
        present == wanted,
        "{directory:?} is not exactly the pinned release: has {present:?}, the release has {wanted:?}"
    );

    // ── config, precisions and the header-level layout bijection ───────────
    let config_text = std::fs::read_to_string(directory.join("config.json"))?;
    Inkling2LoopConfig::from_json(&config_text).map_err(anyhow::Error::msg)?;
    let (_, inspection) = checkpoint::inspect_dir(directory).map_err(anyhow::Error::msg)?;
    ensure!(
        inspection.is_bijection(),
        "layout disagrees with the headers: {:?}",
        inspection.findings
    );
    Precisions::from_hf_quant_config(&std::fs::read_to_string(directory.join("hf_quant_config.json"))?)
        .map_err(anyhow::Error::msg)?;
    let config: serde_json::Value = serde_json::from_str(&config_text)?;
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("model.safetensors.index.json"))?)?;
    let weight_map = index["weight_map"].as_object().context("index has no weight_map")?;
    ensure!(
        weight_map.values().all(|v| v.as_str() == Some(MODEL_FILE)),
        "the index names a shard other than {MODEL_FILE}"
    );

    // ── map, hash and tile-check the two safetensors files ─────────────────
    let model = open_safetensors(directory, MODEL_FILE)?;
    let mtp = open_safetensors(directory, MTP_FILE)?;
    ensure!(
        model.tensors == weight_map.len(),
        "{MODEL_FILE} holds {} tensors, the index {}",
        model.tensors,
        weight_map.len()
    );
    let held = parse_header(MODEL_FILE, model.header.as_bytes()).map_err(anyhow::Error::msg)?;
    for key in weight_map.keys() {
        ensure!(held.contains_key(key), "index names {key}, absent from {MODEL_FILE}");
    }

    let mut receipts = Vec::new();
    let mut documents: Vec<(String, String)> = Vec::new();
    for &(path, size, lfs) in RELEASE {
        let bytes = match path {
            MODEL_FILE => model.bytes.clone(),
            MTP_FILE => mtp.bytes.clone(),
            _ => anybytes::Bytes::from_source(std::fs::read(directory.join(path))?),
        };
        ensure!(bytes.len() as u64 == size, "{path}: {} bytes, the release has {size}", bytes.len());
        let sha = hex(Sha256::digest(&bytes));
        if let Some(lfs) = lfs {
            ensure!(sha == lfs, "{path}: sha256 {sha}, the release's LFS pointer says {lfs}");
        }
        eprintln!("[inkling2] {path}: {size} bytes, sha256 {sha}{}", if lfs.is_some() { " (LFS match)" } else { "" });
        receipts.push(serde_json::json!({"path": path, "bytes": size, "sha256": sha,
            "lfs_sha256_checked": lfs.is_some()}));
        if path != MODEL_FILE && path != MTP_FILE {
            let text = String::from_utf8(bytes.to_vec()).with_context(|| format!("{path} is not UTF-8"))?;
            documents.push((path.to_owned(), text));
        }
    }
    for file in [&model, &mtp] {
        documents.push((header_document(file.name), file.header.clone()));
    }
    let template = documents
        .iter()
        .find(|(n, _)| n == "chat_template.jinja")
        .map(|(_, t)| t.clone())
        .context("chat template absent")?;
    let tokenizer_json = std::fs::read(directory.join("tokenizer.json"))?;

    // ── leaves ─────────────────────────────────────────────────────────────
    let (drafter_members, drafter_facts, drafter_parameters) = ingest_file(pile, &mtp)?;
    let drafter = crate::ingest::build_model_root(
        pile,
        &format!("{SOURCE}@{REVISION}/{MTP_FILE}"),
        crate::persist::QUANTIZATION_NATIVE,
        drafter_members,
        drafter_facts,
        &[MTP_FILE.to_owned()],
    )
    .map_err(|e| anyhow::anyhow!("drafter root: {e}"))?;
    let drafter_root = drafter.root().context("drafter root")?;
    eprintln!("[inkling2] drafter: {} tensors, root {drafter_root:X}", mtp.tensors);

    let (mut members, mut facts, parameters) = ingest_file(pile, &model)?;
    eprintln!("[inkling2] model: {} tensors", model.tensors);
    members.push(drafter_root);
    facts += drafter;

    // ── the root and its parts ─────────────────────────────────────────────
    let tokenizer = crate::tokenizer::save_tokenizer_json(&tokenizer_json, SOURCE, pile)
        .map_err(|e| anyhow::anyhow!("store the tokenizer: {e}"))?;
    let tokenizer_root = tokenizer.root().context("tokenizer root")?;
    facts += tokenizer;
    let config_root = crate::jsonfacts::save_json(&config, pile, &mut facts)
        .map_err(|e| anyhow::anyhow!("store config.json: {e}"))?;
    let template = pile.put::<blobencodings::UTF8String, _>(template)?;
    let mut fragment = crate::ingest::build_model_root_with(
        pile,
        &format!("{SOURCE}@{REVISION}"),
        crate::persist::QUANTIZATION_NATIVE,
        members,
        facts,
        &[MODEL_FILE.to_owned(), MTP_FILE.to_owned()],
        crate::ingest::RootParts {
            tokenizer: Some(tokenizer_root),
            config: Some(config_root),
            chat_template: Some(template),
        },
    )
    .map_err(|e| anyhow::anyhow!("model root: {e}"))?;
    let model_root = fragment.root().context("model root")?;
    label(pile, &mut fragment, drafter_root, DRAFTER, model_root)?;

    for (name, text) in documents {
        let text = pile.put::<blobencodings::UTF8String, _>(text)?;
        let document = entity! { _ @ crate::dataset::text: text };
        let asset = document.root().context("document root")?;
        fragment += document;
        label(pile, &mut fragment, asset, &name, model_root)?;
    }

    let manifest = serde_json::json!({
        "source": SOURCE,
        "revision": REVISION,
        "files": receipts,
        "model_tensors": model.tensors,
        "drafter_tensors": mtp.tensors,
        "parameters": parameters + drafter_parameters,
        "storage": "every tensor at its source dtype, shape and bytes (BF16, F32, U8, E4M3); \
                    each safetensors header kept as the document <file>#header",
    });
    let manifest_root = crate::jsonfacts::save_json(&manifest, pile, &mut fragment)
        .map_err(|e| anyhow::anyhow!("manifest facts: {e}"))?;
    label(pile, &mut fragment, manifest_root, MANIFEST, model_root)?;

    Ok(Candidate {
        fragment,
        artifacts: Artifacts { model_root, drafter_root, config_root, tokenizer_root, manifest_root },
        manifest,
    })
}

/// Regular files under `directory`, relative, one level of subdirectory deep
/// as the release has (`tiktoken/`). Anything else is reported by name.
fn walk(directory: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(directory).with_context(|| format!("list {directory:?}"))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            for inner in std::fs::read_dir(entry.path())? {
                let inner = inner?;
                ensure!(inner.file_type()?.is_file(), "{name}/{:?} is not a regular file", inner.file_name());
                out.push(format!("{name}/{}", inner.file_name().to_string_lossy()));
            }
        } else {
            ensure!(kind.is_file(), "{name} is not a regular file");
            out.push(name);
        }
    }
    Ok(out)
}
