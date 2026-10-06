//! Explicit selected roots in one frozen observation; no weight catalogue.
use super::config::Config;
use anyhow::{Context, Result, bail, ensure};
use triblespace::core::blob::{
    Blob,
    encodings::tensor::{
        Tensor, TensorElement,
        elements::{BF16, F32},
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
            return Ok((handle, blob));
        }
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
