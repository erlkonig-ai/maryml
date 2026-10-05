//! Import only the Qwen2 text/speech decoder. The audio-understanding encoder
//! and the four token2wav networks are distinct future components.
use super::{
    SOURCE,
    codec::MiniCodec,
    config::Config,
    load::{self, Artifacts},
};
use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;
use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use triblespace::core::{metadata, repo::pile::Pile};
use triblespace::prelude::*;

pub struct Candidate {
    pub fragment: Fragment,
    pub artifacts: Artifacts,
    pub tensor_count: usize,
    pub parameter_count: u64,
}

fn mapped(path: &Path) -> Result<anybytes::Bytes> {
    let file = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
    // SAFETY: caller keeps the checkpoint files immutable during ingestion.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("map {path:?}"))?;
    Ok(anybytes::Bytes::from_source(mmap))
}

/// Build an unpublished candidate. The caller owns admission, publication and
/// durability. Failure may leave unreferenced content-addressed blobs, but never
/// publishes an incomplete model. `revision` is caller-supplied provenance;
/// content-addressed leaves identify the actual imported bytes independently.
pub fn ingest_checkpoint(pile: &mut Pile, directory: &Path, revision: &str) -> Result<Candidate> {
    pile.refresh()
        .context("refresh model pile; no automatic repair")?;
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    let config = Config::from_json(&value)?;
    let tokenizer_bytes = std::fs::read(directory.join("tokenizer.json"))?;
    MiniCodec::from_bytes(&tokenizer_bytes, config.vocab_size)?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        directory.join("model.safetensors.index.json"),
    )?)?;
    let weight_map = index["weight_map"]
        .as_object()
        .context("index has no weight_map")?;

    // Ephemeral checkpoint metadata for THIS import operation, not a retained
    // catalogue of pile rows. It checks selected roles before writing weights.
    let mut expected = HashMap::<String, Vec<usize>>::new();
    config.tensors(|name, shape| {
        expected.insert(name.to_string(), shape.to_vec());
        Ok(())
    })?;
    let mut shards = Vec::<PathBuf>::new();
    for name in expected.keys() {
        let shard = weight_map
            .get(name)
            .and_then(|v| v.as_str())
            .with_context(|| format!("index lacks decoder slot {name}"))?;
        let path = Path::new(shard);
        ensure!(
            matches!(path.components().next(), Some(Component::Normal(_)))
                && path.components().count() == 1
                && path.extension().is_some_and(|x| x == "safetensors"),
            "unsafe/non-safetensors shard name {shard:?}"
        );
        let path = directory.join(path);
        if !shards.contains(&path) {
            shards.push(path);
        }
    }
    shards.sort();
    let mut seen = HashSet::new();
    for path in &shards {
        let bytes = mapped(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        for (name, shape) in &expected {
            if let Ok(tensor) = tensors.tensor(name) {
                ensure!(
                    seen.insert(name.clone()),
                    "duplicate selected tensor {name}"
                );
                ensure!(
                    tensor.dtype() == safetensors::Dtype::BF16 && tensor.shape() == shape,
                    "{name}: expected native BF16 {shape:?}, got {:?} {:?}",
                    tensor.dtype(),
                    tensor.shape()
                );
                ensure!(
                    weight_map[name].as_str() == path.file_name().and_then(|x| x.to_str()),
                    "index assigns {name} to a different shard"
                );
            }
        }
    }
    ensure!(
        seen.len() == expected.len(),
        "checkpoint lacks {} selected tensor payloads",
        expected.len() - seen.len()
    );

    let mut members = Vec::new();
    let mut facts = TribleSet::new();
    for path in &shards {
        let bytes = mapped(path)?;
        let (mut shard_members, shard_facts) =
            crate::ingest::ingest_bf16_members(&bytes, pile, |name| expected.contains_key(name))
                .map_err(|e| anyhow::anyhow!("ingest {path:?}: {e}"))?;
        members.append(&mut shard_members);
        facts += shard_facts;
    }
    let provenance: Vec<_> = shards
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let mut fragment = crate::ingest::build_model_root(
        pile,
        &format!("{SOURCE}@{revision}"),
        "native",
        members,
        facts,
        &provenance,
    )
    .map_err(|e| anyhow::anyhow!("build decoder root: {e}"))?;
    let model_root = fragment.root().expect("model root");
    let config_root = crate::jsonfacts::save_json(&value, pile, fragment.facts_mut())
        .map_err(|e| anyhow::anyhow!("persist config: {e}"))?;
    let text = String::from_utf8(tokenizer_bytes)?;
    let text = pile.put::<blobencodings::UTF8String, _>(text)?;
    let tokenizer = entity! { _ @ crate::dataset::text: text };
    let tokenizer_asset = tokenizer.root().expect("tokenizer asset");
    fragment += tokenizer;
    fragment += entity! { ExclusiveId::force_ref(&tokenizer_asset) @
    metadata::name: "tokenizer.json", crate::format::attrs::model_root: model_root };
    fragment += entity! { ExclusiveId::force_ref(&config_root) @
    metadata::name: "config.json", crate::format::attrs::model_root: model_root };
    let artifacts = Artifacts {
        model_root,
        config_root,
        tokenizer_asset,
    };
    load::validate_decoder(fragment.facts(), pile, model_root, &config)?;
    load::Assets::from_frozen(fragment.facts(), pile, artifacts)?;
    Ok(Candidate {
        fragment,
        artifacts,
        tensor_count: expected.len(),
        parameter_count: config.parameter_count()?,
    })
}
