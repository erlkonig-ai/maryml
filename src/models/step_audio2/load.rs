//! Read explicit stored identities from one frozen observation. No global
//! discovery, reconstructed entity IDs, weight catalogue, or source-file access.
use super::{codec::MiniCodec, config::Config};
use anyhow::{Result, bail};
use triblespace::core::blob::{
    Blob,
    encodings::tensor::{Tensor, elements::BF16},
};
use triblespace::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Artifacts {
    pub model_root: Id,
    pub config_root: Id,
    pub tokenizer_asset: Id,
}

pub struct Assets {
    pub config: Config,
    pub codec: MiniCodec,
}

impl Assets {
    pub fn from_frozen(
        facts: &TribleSet,
        reader: &impl BlobStoreGet,
        ids: Artifacts,
    ) -> Result<Self> {
        for asset in [ids.config_root, ids.tokenizer_asset] {
            anyhow::ensure!(
                exists!(pattern!(facts, [
                    { asset @ crate::format::attrs::model_root: ids.model_root }
                ])),
                "selected asset {asset:?} is not associated with selected model {:?}",
                ids.model_root
            );
        }
        let value = crate::jsonfacts::load_json(facts, reader, ids.config_root)
            .map_err(|e| anyhow::anyhow!("read selected config: {e}"))?;
        let config = Config::from_json(&value)?;
        // The tokenizer is a serialized text document; dataset::text is the
        // existing UTF8 payload relation. Additional names/tags are irrelevant.
        for (handle,) in find!(
            (handle: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(facts, [{ ids.tokenizer_asset @ crate::dataset::text: ?handle }])
        ) {
            let text: anybytes::View<str> = reader
                .get(handle)
                .map_err(|e| anyhow::anyhow!("read selected tokenizer {handle:?}: {e}"))?;
            let codec = MiniCodec::from_bytes(text.as_bytes(), config.vocab_size)?;
            return Ok(Self { config, codec });
        }
        bail!(
            "no serialized tokenizer text at selected asset {:?}",
            ids.tokenizer_asset
        )
    }
}

/// Select a supported native BF16 slot by its actual stored root relation.
/// Shape-incompatible alternatives and unknown attributes remain in the graph.
pub fn tensor<const R: usize>(
    facts: &TribleSet,
    reader: &impl BlobStoreGet,
    root: Id,
    name: &str,
    shape: [u64; R],
) -> Result<(
    Inline<inlineencodings::Handle<Tensor<BF16, R>>>,
    Blob<Tensor<BF16, R>>,
)> {
    // This is a content-addressed string value, not a derived entity identity.
    // Constrain the name in the join rather than materializing every member.
    let path: Blob<blobencodings::UTF8String> = name.to_owned().to_blob();
    let path = path.get_handle();
    for (handle,) in find!(
        (handle: Inline<inlineencodings::Handle<Tensor<BF16, R>>>),
        pattern!(facts, [
            { root @ crate::format::attrs::member: _?member },
            { _?member @ crate::format::attrs::safetensor_path: path,
                crate::format::attrs::weight: _?weight },
            { _?weight @ crate::leaf::leaf::<BF16, R>(): ?handle }
        ])
    ) {
        let blob: Blob<Tensor<BF16, R>> = reader
            .get(handle)
            .map_err(|e| anyhow::anyhow!("{name}: read native BF16 leaf: {e}"))?;
        let leaf =
            crate::leaf::read_leaf(blob.clone()).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        if leaf.dims() == shape {
            return Ok((handle, blob));
        }
    }
    bail!("{name}: no native BF16 rank-{R} slot with shape {shape:?} under {root:?}")
}

pub fn validate_decoder(
    facts: &TribleSet,
    reader: &impl BlobStoreGet,
    root: Id,
    config: &Config,
) -> Result<()> {
    config.tensors(|name, shape| {
        match shape {
            [a] => {
                tensor(facts, reader, root, name, [*a as u64])?;
            }
            [a, b] => {
                tensor(facts, reader, root, name, [*a as u64, *b as u64])?;
            }
            _ => unreachable!("Qwen2 has only vector/matrix parameters"),
        }
        Ok(())
    })
}
