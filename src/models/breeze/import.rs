//! Import-only source access. Every source tensor retains its dtype and shape.
use super::{
    SOURCE,
    config::Config,
    load::{self, Artifacts},
};
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    path::{Component, Path},
};
use triblespace::core::{metadata, repo::pile::Pile};
use triblespace::prelude::*;

const DOCUMENTS: &[&str] = &[
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/configuration.json",
    "audio_tokenizer/preprocessor_config.json",
];

pub struct Candidate {
    pub fragment: Fragment,
    pub artifacts: Artifacts,
    pub tensor_count: usize,
    pub parameter_count: u64,
    pub payload_bytes: u64,
}

fn mapped(path: &Path) -> Result<anybytes::Bytes> {
    let file = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
    // SAFETY: caller must retain immutable checkpoint files throughout import.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("map {path:?}"))?;
    Ok(anybytes::Bytes::from_source(mmap))
}

/// Returns an unpublished fragment. Failed imports may leave unreferenced blobs
/// in the caller's new pile, but never publish an incomplete model collection.
pub fn ingest_checkpoint(pile: &mut Pile, directory: &Path, revision: &str) -> Result<Candidate> {
    pile.refresh().context("refresh task model pile")?;
    let config_bytes = std::fs::read(directory.join("config.json"))?;
    let value: serde_json::Value = serde_json::from_slice(&config_bytes)?;
    Config::from_json(&value)?;
    let codec_value: serde_json::Value = serde_json::from_slice(&std::fs::read(
        directory.join("audio_tokenizer/config.json"),
    )?)?;
    super::codec_config::validate_config(&codec_value)?;
    let tokenizer = std::fs::read(directory.join("tokenizer.json"))?;
    tokenizers::Tokenizer::from_bytes(&tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        directory.join("model.safetensors.index.json"),
    )?)?;
    let map = index["weight_map"]
        .as_object()
        .context("index has no weight_map")?;
    // Ephemeral foreign-source layout for this one import, never retained pile
    // state or runtime discovery. The source index owns shard membership.
    let mut shards = Vec::<String>::new();
    for shard in map.values() {
        let name = shard.as_str().context("non-string shard name")?;
        let path = Path::new(name);
        ensure!(
            path.components().count() == 1
                && matches!(path.components().next(), Some(Component::Normal(_)))
                && path.extension().is_some_and(|x| x == "safetensors"),
            "unsafe shard name {name:?}"
        );
        if !shards.iter().any(|s| s == name) {
            shards.push(name.to_owned());
        }
    }
    ensure!(!shards.is_empty(), "empty checkpoint index");
    shards.sort();
    let mut seen = HashSet::<String>::new();
    let mut source_receipts = Vec::new();
    let mut count = 0usize;
    let mut parameters = 0u64;
    let mut payload_bytes = 0u64;
    // Check the entire source closure before writing the first tensor. No
    // dtype filter: unsupported formats fail instead of silently disappearing.
    for name in shards
        .iter()
        .map(String::as_str)
        .chain(["audio_tokenizer/model.safetensors"])
    {
        let bytes = mapped(&directory.join(name))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        for key in tensors.names() {
            let tensor = tensors.tensor(key)?;
            ensure!(
                matches!(tensor.dtype(), Dtype::BF16 | Dtype::F32) && tensor.shape().len() <= 6,
                "{name}:{key}: unsupported {:?} rank {}",
                tensor.dtype(),
                tensor.shape().len()
            );
            if name != "audio_tokenizer/model.safetensors" {
                ensure!(
                    seen.insert(key.to_string()),
                    "duplicate source tensor {key}"
                );
                ensure!(
                    map.get(key).and_then(|v| v.as_str()) == Some(name),
                    "index mismatch for {key}"
                );
            }
            count += 1;
            parameters = parameters
                .checked_add(tensor.shape().iter().try_fold(1u64, |a, &b| {
                    a.checked_mul(b as u64).context("shape element overflow")
                })?)
                .context("parameter count overflow")?;
            payload_bytes = payload_bytes
                .checked_add(tensor.data().len() as u64)
                .context("payload count overflow")?;
        }
        source_receipts.push(serde_json::json!({"path":name,"bytes":bytes.len(),
            "sha256":format!("{:x}", Sha256::digest(&bytes))}));
    }
    ensure!(
        seen.len() == map.len(),
        "checkpoint is missing indexed tensors"
    );
    let mut documents = Vec::new();
    for &name in DOCUMENTS {
        let bytes = std::fs::read(directory.join(name)).with_context(|| format!("read {name}"))?;
        source_receipts.push(serde_json::json!({"path":name,"bytes":bytes.len(),
            "sha256":format!("{:x}", Sha256::digest(&bytes))}));
        documents.push((
            name,
            String::from_utf8(bytes).with_context(|| format!("non-UTF8 document {name}"))?,
        ));
    }
    let mut main_members = Vec::new();
    let mut main_facts = TribleSet::new();
    for name in &shards {
        let (members, facts) = ingest_file(pile, &mapped(&directory.join(name))?)?;
        main_members.extend(members);
        main_facts += facts;
    }
    let (codec_members, codec_facts) = ingest_file(
        pile,
        &mapped(&directory.join("audio_tokenizer/model.safetensors"))?,
    )?;
    let codec = crate::ingest::build_model_root(
        pile,
        &format!("{SOURCE}@{revision}/audio_tokenizer"),
        "native",
        codec_members,
        codec_facts,
        &["audio_tokenizer/model.safetensors".to_owned()],
    )
    .map_err(|e| anyhow::anyhow!("codec root: {e}"))?;
    let external_codec_root = codec.root().context("codec root absent")?;
    main_members.push(external_codec_root);
    main_facts += codec.into_facts();
    let mut provenance = shards;
    provenance.push("audio_tokenizer/model.safetensors".to_owned());
    let mut fragment = crate::ingest::build_model_root(
        pile,
        &format!("{SOURCE}@{revision}"),
        "native",
        main_members,
        main_facts,
        &provenance,
    )
    .map_err(|e| anyhow::anyhow!("model root: {e}"))?;
    let model_root = fragment.root().context("model root absent")?;
    let config_root = crate::jsonfacts::save_json(&value, pile, fragment.facts_mut())
        .map_err(|e| anyhow::anyhow!("config facts: {e}"))?;
    let external_codec_config_root =
        crate::jsonfacts::save_json(&codec_value, pile, fragment.facts_mut())
            .map_err(|e| anyhow::anyhow!("codec config facts: {e}"))?;
    let source_value = serde_json::json!({"source":SOURCE,"revision":revision,"files":source_receipts,
        "tensor_count":count,"parameter_count":parameters,"payload_bytes":payload_bytes});
    let source_root = crate::jsonfacts::save_json(&source_value, pile, fragment.facts_mut())
        .map_err(|e| anyhow::anyhow!("source provenance facts: {e}"))?;
    for (asset, name) in [
        (config_root, "config.json"),
        (external_codec_config_root, "audio_tokenizer/config.json"),
        (source_root, "native-source-manifest"),
        (external_codec_root, "audio_tokenizer"),
    ] {
        fragment += entity! { ExclusiveId::force_ref(&asset) @ metadata::name: name,
        crate::format::attrs::model_root: model_root };
    }
    let mut tokenizer_asset = None;
    for (name, text) in documents {
        let text = pile.put::<blobencodings::UTF8String, _>(text)?;
        let document = entity! { _ @ crate::dataset::text: text };
        let asset = document.root().context("document root absent")?;
        fragment += document;
        fragment += entity! { ExclusiveId::force_ref(&asset) @ metadata::name: name,
        crate::format::attrs::model_root: model_root };
        if name == "tokenizer.json" {
            tokenizer_asset = Some(asset);
        }
    }
    let artifacts = Artifacts {
        model_root,
        config_root,
        tokenizer_asset: tokenizer_asset.context("tokenizer asset absent")?,
        external_codec_root,
        external_codec_config_root,
    };
    let snapshot = pile.snapshot().context("freeze imported candidate")?;
    load::Assets::from_frozen(fragment.facts(), &snapshot, artifacts)?;
    Ok(Candidate {
        fragment,
        artifacts,
        tensor_count: count,
        parameter_count: parameters,
        payload_bytes,
    })
}

fn ingest_file(pile: &mut Pile, bytes: &anybytes::Bytes) -> Result<(Vec<Id>, TribleSet)> {
    let tensors = SafeTensors::deserialize(bytes)?;
    let mut names = tensors.names();
    names.sort();
    let mut members = Vec::new();
    let mut facts = TribleSet::new();
    for name in names {
        let tensor = tensors.tensor(name)?;
        let elem = match tensor.dtype() {
            Dtype::BF16 => crate::leaf::Elem::Bf16,
            Dtype::F32 => crate::leaf::Elem::F32,
            other => anyhow::bail!("{name}: unsupported source dtype {other:?}"),
        };
        let start = tensor.data().as_ptr() as usize - bytes.as_ptr() as usize;
        let payload = bytes.slice(start..start + tensor.data().len());
        let shape: Vec<u64> = tensor.shape().iter().map(|&x| x as u64).collect();
        let leaf = crate::leaf::put_leaf(pile, elem, &shape, payload, name)?;
        let weight = leaf.root().context("leaf root absent")?;
        facts += leaf.into_facts();
        let name = pile.put::<blobencodings::UTF8String, _>(name.to_owned())?;
        let member = entity! { _ @ crate::format::attrs::safetensor_path: name,
        crate::format::attrs::kind: "tensor", crate::format::attrs::weight: weight };
        members.push(member.root().context("member root absent")?);
        facts += member.into_facts();
    }
    Ok((members, facts))
}
