//! WeMM's text leg alone: one raw UTF-8 text -> GPU BF16 `[1,4096]`, on
//! whichever runtime the decoder is built for ([`super::Rt`]).
//!
//! The same prepared-token path [`super::native::NativeWemm::embed_text`]
//! runs: the root's own pinned codec frames the text (chat template, then
//! `<embedding>`), [`PreparedDecoder::embed_unpadded`] gathers the token rows
//! and runs the 32 decoder blocks, and the shared output boundary takes the
//! final norm, the last (`<embedding>`) row and its L2 normalization. Nothing
//! here loads the vision tower, so a text-only process (a query embedder on a
//! machine without CUDA) binds the 426 decoder roles and nothing else.
//! [`Assets`] are the root's tokenizer, configuration and chat template, read
//! from the same facts as the weights; the CUDA facade uses them too.

use sha2::{Digest, Sha256};
use triblespace::{
    core::repo::BlobStoreGet,
    prelude::{Id, TribleSet},
};

use super::{
    Binder,
    config::Qwen3_5Config,
    gdn_mixer::DecoderPrecision,
    input_codec::InputCodec,
    multimodal_layout,
    prepared::{CudaTensor, PreparedDecoder},
};

/// Exact WeMM checkpoint configuration used by the native behavior fixtures.
/// This is an asset checksum, not a collection or mapping identity.
pub const CONFIG_SHA256: &str = "34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004";
/// The same configuration read back from facts: the SHA-256 of its
/// [`crate::jsonfacts::canonical_string`], which is what the checkpoint's
/// `config.json` (sha256 [`CONFIG_SHA256`]) canonicalizes to. A configuration
/// stored as facts has no file bytes left to hash, and its canonical text is
/// independent of map order and whitespace, so this pins the same values.
pub const CONFIG_FACTS_SHA256: &str =
    "719264c54581f878babd2d13317ffb1e23f5943dd424e3594a8d107fbd36d4d6";

/// Validated owned assets. Bytes may be exact-fetched by the caller; this
/// type neither opens files nor discovers asset/model records. Construct it
/// BEFORE creating a CUDA alias session if malformed assets must initialize
/// no GPU resources. Other template bytes are refused, not interpreted as Jinja.
pub struct Assets {
    #[cfg_attr(not(feature = "wemm-input"), allow(dead_code))]
    pub(super) config: Qwen3_5Config,
    pub(super) codec: InputCodec,
}

impl Assets {
    /// From the checkpoint's own files, as the behavior fixtures use them.
    pub fn from_bytes(config: &[u8], tokenizer: &[u8], template: &[u8]) -> Result<Self, String> {
        if format!("{:x}", Sha256::digest(config)) != CONFIG_SHA256 {
            return Err("unsupported WeMM configuration asset bytes".into());
        }
        let parsed_config =
            Qwen3_5Config::from_json(std::str::from_utf8(config).map_err(|e| e.to_string())?)?;
        multimodal_layout::validate_model(&parsed_config)?;
        let codec = InputCodec::from_assets(tokenizer, template)?;
        Ok(Self {
            config: parsed_config,
            codec,
        })
    }

    /// From the model root itself: the tokenizer, configuration and chat
    /// template that are part of `root`'s identity
    /// ([`crate::ingest::RootParts`]), read from the same frozen facts and
    /// reader the weights come from. No file is opened and nothing is
    /// discovered; a root without all three parts is refused.
    ///
    /// The tokenizer is rebuilt from facts ([`crate::tokenizer::build_tokenizer`])
    /// and so has no file bytes to pin. It is the root's own: selecting the
    /// root selected it, and the root id is derived from its content, so it
    /// cannot be paired with another model's weights the way a side-file
    /// could. The configuration is pinned by [`CONFIG_FACTS_SHA256`] and the
    /// template by its exact bytes, as before.
    pub fn from_model_root<R: BlobStoreGet>(
        facts: &TribleSet,
        reader: &R,
        root: Id,
    ) -> Result<Self, String> {
        let parts = crate::selection::model_root_parts(facts, root).map_err(|e| e.to_string())?;
        let missing = |what: &str| format!("WeMM model root {root:X} has no {what} in its core");
        let tokenizer = parts.tokenizer.ok_or_else(|| missing("tokenizer"))?;
        let config = parts.config.ok_or_else(|| missing("config"))?;
        let template = parts
            .chat_template
            .ok_or_else(|| missing("chat template"))?;
        let config = crate::jsonfacts::load_json(facts, reader, config)
            .map_err(|e| format!("read the WeMM configuration facts: {e}"))?;
        let config = crate::jsonfacts::canonical_string(&config);
        if format!("{:x}", Sha256::digest(config.as_bytes())) != CONFIG_FACTS_SHA256 {
            return Err("unsupported WeMM configuration facts".into());
        }
        let parsed_config = Qwen3_5Config::from_json(&config)?;
        multimodal_layout::validate_model(&parsed_config)?;
        let tokenizer = crate::tokenizer::build_tokenizer(facts, reader, tokenizer)
            .map_err(|e| format!("rebuild the WeMM tokenizer from facts: {e}"))?;
        let template: anybytes::View<str> = reader
            .get(template)
            .map_err(|e| format!("read the WeMM chat template: {e}"))?;
        let codec = InputCodec::from_tokenizer(tokenizer, template.as_bytes())?;
        Ok(Self {
            config: parsed_config,
            codec,
        })
    }

    /// The codec these assets frame input with, for a caller that also cuts
    /// text into windows with it.
    pub fn codec(&self) -> &InputCodec {
        &self.codec
    }
}

/// One resident text decoder of a selected root. Keep it for repeated
/// queries; constructing one per query is not a cache policy.
pub struct TextWemm {
    decoder: PreparedDecoder,
    codec: InputCodec,
    root: Id,
}

impl TextWemm {
    /// Bind the root's 426 decoder roles (token table, 32 blocks, final norm)
    /// through the caller's binder, with every linear projection on
    /// `precision`. `selected` sees each role as it is resolved.
    ///
    /// # Safety
    /// The binder's own premise for every leaf `reader` returns: on CUDA,
    /// [`super::prepared::PreparedDecoder::from_pile`]'s genuine immutable
    /// pile prefix; on wgpu none, since each payload is copied at bind.
    pub unsafe fn from_frozen<R: BlobStoreGet>(
        facts: &TribleSet,
        reader: &R,
        root: Id,
        assets: Assets,
        binder: &mut Binder,
        selected: impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
        precision: DecoderPrecision,
    ) -> Result<Self, String> {
        // SAFETY: forwarded unchanged.
        let decoder = unsafe {
            PreparedDecoder::from_pile_with(facts, reader, root, binder, selected, precision)?
        };
        Ok(Self {
            decoder,
            codec: assets.codec,
            root,
        })
    }

    /// One raw text, framed exactly as `NativeWemm::embed_text` frames it;
    /// at most 256 framed IDs, nothing cut. The result may still be in
    /// flight on the device's stream.
    pub fn embed_text(&self, text: &str) -> Result<CudaTensor, String> {
        let tokens = self
            .codec
            .text(text)
            .map_err(|e| format!("WeMM text preparation: {e}"))?;
        Ok(self
            .decoder
            .embed_unpadded(tokens.as_slice())
            .map_err(|e| format!("WeMM text forward: {e}"))?
            .endpoint
            .embedding)
    }

    /// The codec this model frames text with.
    pub fn codec(&self) -> &InputCodec {
        &self.codec
    }

    /// See `Stack::decoder_bound`.
    pub fn decoder_bound(&self) -> Option<super::gdn_mixer::BoundProjection> {
        self.decoder.decoder_bound()
    }

    pub fn model_root(&self) -> Id {
        self.root
    }
}
