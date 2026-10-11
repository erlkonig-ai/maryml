//! Explicit selected roots in one frozen observation; no weight catalogue.
use super::config::Config;
use anyhow::{Context, Result, bail, ensure};
use triblespace::core::blob::{
    Blob,
    encodings::tensor::{
        Tensor, TensorElement,
        elements::{BF16, F32, NVFP4},
    },
};
use triblespace::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Artifacts {
    pub model_root: Id,
    pub config_root: Id,
    pub tokenizer_asset: Id,
    pub external_codec_root: Id,
    pub external_codec_config_root: Id,
}

impl Artifacts {
    /// Select the Breeze model root stored under `quantization` for this
    /// source revision, and its four assets by the names the importer gave
    /// them under that root. Every coordinate is a stored label or a
    /// relation; no id is computed, and two candidates are an error.
    pub fn select(
        facts: &TribleSet,
        reader: &impl BlobStoreGet,
        quantization: &str,
    ) -> Result<Self> {
        let source = format!("{}@{}", super::SOURCE, super::REVISION);
        let model_root = crate::selection::select_model_root(
            facts,
            reader,
            crate::selection::ModelSelector::Source {
                source: &source,
                quantization,
            },
        )?;
        Self::from_root(facts, model_root)
    }

    /// The four assets the importer named under one explicit model root.
    ///
    /// A name alone does not pick one: the importer stores `config.json` both
    /// as its text and as its parsed JSON, so each asset is asked for with
    /// what it is: the configs as JSON objects, the tokenizer as text, the
    /// codec as a model root.
    pub fn from_root(facts: &TribleSet, model_root: Id) -> Result<Self> {
        use crate::jsonfacts::ty::JSON_OBJECT;
        use triblespace::core::metadata;
        let one = |name: &str, found: Vec<Id>| -> Result<Id> {
            let mut found = found.into_iter();
            let first = found
                .next()
                .with_context(|| format!("Breeze root {model_root:?} has no {name} asset"))?;
            ensure!(
                found.all(|other| other == first),
                "Breeze root {model_root:?} has several {name} assets"
            );
            Ok(first)
        };
        let handle = |name: &str| {
            let blob: Blob<blobencodings::UTF8String> = name.to_owned().to_blob();
            blob.get_handle() // content-addressed value, not an entity-ID lookup
        };
        let json = |name: &str| -> Result<Id> {
            let wanted = handle(name);
            one(
                name,
                find!((asset: Id), pattern!(facts, [{ ?asset @
                    crate::format::attrs::model_root: model_root,
                    metadata::name: wanted,
                    metadata::tag: JSON_OBJECT,
                }]))
                .map(|(asset,)| asset)
                .collect(),
            )
        };
        let wanted = handle("tokenizer.json");
        let tokenizer_asset = one(
            "tokenizer.json",
            find!((asset: Id), pattern!(facts, [{ ?asset @
                crate::format::attrs::model_root: model_root,
                metadata::name: wanted,
                crate::dataset::text: _?text,
            }]))
            .map(|(asset,)| asset)
            .collect(),
        )?;
        let wanted = handle("audio_tokenizer");
        let external_codec_root = one(
            "audio_tokenizer",
            find!((asset: Id), pattern!(facts, [{ ?asset @
                crate::format::attrs::model_root: model_root,
                metadata::name: wanted,
                crate::format::attrs::member: _?member,
            }]))
            .map(|(asset,)| asset)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect(),
        )?;
        Ok(Self {
            model_root,
            config_root: json("config.json")?,
            tokenizer_asset,
            external_codec_root,
            external_codec_config_root: json("audio_tokenizer/config.json")?,
        })
    }
}

pub struct Assets {
    pub config: Config,
    pub tokenizer_json: String,
    pub external_codec_config: serde_json::Value,
}

impl Assets {
    pub fn from_frozen(
        facts: &TribleSet,
        reader: &impl BlobStoreGet,
        ids: Artifacts,
    ) -> Result<Self> {
        for asset in [
            ids.config_root,
            ids.tokenizer_asset,
            ids.external_codec_root,
            ids.external_codec_config_root,
        ] {
            ensure!(
                exists!(
                    pattern!(facts, [{ asset @ crate::format::attrs::model_root: ids.model_root }])
                ),
                "asset {asset:?} is not associated with selected model {:?}",
                ids.model_root
            );
        }
        let value = crate::jsonfacts::load_json(facts, reader, ids.config_root)
            .map_err(|e| anyhow::anyhow!("selected Breeze config: {e}"))?;
        let config = Config::from_json(&value)?;
        let external_codec_config =
            crate::jsonfacts::load_json(facts, reader, ids.external_codec_config_root)
                .map_err(|e| anyhow::anyhow!("selected external codec config: {e}"))?;
        super::codec_config::validate_config(&external_codec_config)?;
        for (handle,) in find!((handle: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(facts, [{ ids.tokenizer_asset @ crate::dataset::text: ?handle }]))
        {
            let text: anybytes::View<str> = reader
                .get(handle)
                .map_err(|e| anyhow::anyhow!("selected tokenizer blob: {e}"))?;
            tokenizers::Tokenizer::from_bytes(text.as_bytes())
                .map_err(|e| anyhow::anyhow!("selected tokenizer document: {e}"))?;
            return Ok(Self {
                config,
                tokenizer_json: text.to_string(),
                external_codec_config,
            });
        }
        bail!(
            "no supported exact tokenizer document at {:?}",
            ids.tokenizer_asset
        )
    }
}

/// The leaf named `name` under `root` in element format `T` and this exact
/// shape, or `None` when the root holds no such leaf. The type is the query: a
/// root may hold the same name as BF16 or as NVFP4, and asking for one never
/// returns the other.
fn find_tensor<T: TensorElement, const R: usize>(
    facts: &TribleSet,
    reader: &impl BlobStoreGet,
    root: Id,
    name: &str,
    shape: [u64; R],
) -> Result<
    Option<(
        Inline<inlineencodings::Handle<Tensor<T, R>>>,
        Blob<Tensor<T, R>>,
    )>,
> {
    let path: Blob<blobencodings::UTF8String> = name.to_owned().to_blob();
    let path = path.get_handle(); // content-addressed value, not an entity-ID lookup
    for (handle,) in find!((handle: Inline<inlineencodings::Handle<Tensor<T, R>>>),
    pattern!(facts, [
        { root @ crate::format::attrs::member: _?member },
        { _?member @ crate::format::attrs::safetensor_path: path, crate::format::attrs::weight: _?weight },
        { _?weight @ crate::leaf::leaf::<T, R>(): ?handle }
    ])) {
        let blob: Blob<Tensor<T, R>> = reader
            .get(handle)
            .map_err(|e| anyhow::anyhow!("{name}: native typed tensor read: {e}"))?;
        if crate::leaf::read_leaf(blob.clone())
            .with_context(|| name.to_owned())?
            .dims()
            == shape
        {
            return Ok(Some((handle, blob)));
        }
    }
    Ok(None)
}

/// A stored NVFP4 projection `[out, in]` under `root`, if the root holds one.
pub fn find_tensor_nvfp4(
    facts: &TribleSet,
    reader: &impl BlobStoreGet,
    root: Id,
    name: &str,
    shape: [u64; 2],
) -> Result<Option<Blob<Tensor<NVFP4, 2>>>> {
    Ok(find_tensor(facts, reader, root, name, shape)?.map(|(_, blob)| blob))
}

fn tensor<T: TensorElement, const R: usize>(
    facts: &TribleSet,
    reader: &impl BlobStoreGet,
    root: Id,
    name: &str,
    shape: [u64; R],
) -> Result<(
    Inline<inlineencodings::Handle<Tensor<T, R>>>,
    Blob<Tensor<T, R>>,
)> {
    if let Some(found) = find_tensor(facts, reader, root, name, shape)? {
        return Ok(found);
    }
    bail!("{name}: no supported rank-{R} leaf with shape {shape:?} under {root:?}")
}

pub fn tensor_bf16<const R: usize>(
    facts: &TribleSet,
    reader: &impl BlobStoreGet,
    root: Id,
    name: &str,
    shape: [u64; R],
) -> Result<(
    Inline<inlineencodings::Handle<Tensor<BF16, R>>>,
    Blob<Tensor<BF16, R>>,
)> {
    tensor(facts, reader, root, name, shape)
}

pub fn tensor_f32<const R: usize>(
    facts: &TribleSet,
    reader: &impl BlobStoreGet,
    root: Id,
    name: &str,
    shape: [u64; R],
) -> Result<(
    Inline<inlineencodings::Handle<Tensor<F32, R>>>,
    Blob<Tensor<F32, R>>,
)> {
    tensor(facts, reader, root, name, shape)
}
