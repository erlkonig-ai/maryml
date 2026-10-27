//! Qwen2 byte-level BPE, built from the checkpoint's `vocab.json` +
//! `merges.txt` (no tokenizer.json ships with the TTS repo). The Qwen2
//! pre-tokenizer regex differs from GPT-2's (case-insensitive contractions,
//! single-digit numerals), so we assemble Split(qwen2-regex) → ByteLevel
//! explicitly. The 33 added tokens (151643..151675, `<|im_start|>`,
//! `<tts_pad>`, …) are appended in id order so ids line up exactly.
//! Parity-gated against the reference tokenizer in `qwen3tts_probe`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use tokenizers::models::bpe::{BPE, Merges, Vocab};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};
use triblespace::core::metadata;
use triblespace::prelude::blobencodings::UTF8String;
use triblespace::prelude::inlineencodings::Handle;
use triblespace::prelude::*;

/// Qwen2 pre-tokenization regex (from tokenization_qwen2.py PRETOKENIZE_REGEX).
const QWEN2_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const ADDED: [&str; 33] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|audio_pad|>",
];

/// The checkpoint's two tokenizer files. The importer stores each as a text
/// document named on the exact base model root, and the loader asks for them
/// by these names.
pub const VOCAB: &str = "vocab.json";
pub const MERGES: &str = "merges.txt";

/// A document's name as the value `metadata::name` stores.
fn named(name: &str) -> Inline<Handle<UTF8String>> {
    let blob: Blob<UTF8String> = name.to_owned().to_blob();
    blob.get_handle() // content-addressed value, not an entity-ID lookup
}

#[derive(Clone)]
pub struct TextTokenizer {
    tok: Tokenizer,
}

impl TextTokenizer {
    /// The documents named on the model root `root`, the way Breeze names its
    /// `tokenizer.json`.
    pub fn from_root(
        facts: &TribleSet,
        reader: &impl BlobStoreGet,
        root: Id,
    ) -> anyhow::Result<Self> {
        let (vocab_name, merges_name) = (named(VOCAB), named(MERGES));
        for (vocab, merges) in find!(
            (vocab: Inline<Handle<UTF8String>>, merges: Inline<Handle<UTF8String>>),
            pattern!(facts, [
                { _?v @ crate::format::attrs::model_root: root,
                    metadata::name: vocab_name, crate::dataset::text: ?vocab },
                { _?m @ crate::format::attrs::model_root: root,
                    metadata::name: merges_name, crate::dataset::text: ?merges },
            ])
        ) {
            let vocab: anybytes::View<str> = reader
                .get(vocab)
                .map_err(|e| anyhow::anyhow!("Qwen3-TTS vocab.json blob: {e}"))?;
            let merges: anybytes::View<str> = reader
                .get(merges)
                .map_err(|e| anyhow::anyhow!("Qwen3-TTS merges.txt blob: {e}"))?;
            return Self::from_documents(&vocab, &merges);
        }
        anyhow::bail!(
            "Qwen3-TTS model root {root:?} has no vocab.json and merges.txt documents; \
             run qwen3tts_persist on the checkpoint to attach them"
        )
    }

    /// The documents named on a model root of the pile at `path`. A Qwen3-TTS
    /// pile holds one checkpoint family, whose base roots share one tokenizer;
    /// the development tools that load every exact root of it take this.
    pub fn from_model_pile(path: &Path) -> anyhow::Result<Self> {
        let snapshot = crate::model_collection::load_model_collection_local_latest(path)?;
        let vocab_name = named(VOCAB);
        let (root,) = find!((root: Id), pattern!(snapshot.facts(), [{ _?v @
            crate::format::attrs::model_root: ?root,
            metadata::name: vocab_name,
        }]))
        .next()
        .with_context(|| format!("no model root in {path:?} names a vocab.json document"))?;
        Self::from_root(snapshot.facts(), snapshot.store(), root)
    }

    /// `dir` = a checkpoint dir holding vocab.json + merges.txt.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name)).with_context(|| format!("read {name}"))
        };
        Self::from_documents(&read(VOCAB)?, &read(MERGES)?)
    }

    /// Build from the exact text of `vocab.json` and `merges.txt`, parsed the
    /// way `BPE::from_file` parses the files.
    pub fn from_documents(vocab_json: &str, merges_txt: &str) -> anyhow::Result<Self> {
        let vocab: Vocab = serde_json::from_str::<HashMap<String, u32>>(vocab_json)
            .context("parse vocab.json")?
            .into_iter()
            .collect();
        let merges = merges_txt
            .lines()
            .filter(|line| !line.starts_with("#version"))
            .enumerate()
            .map(
                |(rank, line)| match line.split(' ').collect::<Vec<_>>()[..] {
                    [a, b] => Ok((a.to_owned(), b.to_owned())),
                    _ => anyhow::bail!("merges.txt merge {} is not one pair", rank + 1),
                },
            )
            .collect::<anyhow::Result<Merges>>()?;
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|error| anyhow::anyhow!("build Qwen2 BPE: {error}"))?;
        let mut tok = Tokenizer::new(bpe);
        tok.with_normalizer(Some(NFC));
        let split = Split::new(
            tokenizers::pre_tokenizers::split::SplitPattern::Regex(QWEN2_RE.into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .expect("qwen2 split regex");
        tok.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, false, false).into(),
        ])));
        tok.add_special_tokens(
            &ADDED
                .iter()
                .map(|s| AddedToken::from(s.to_string(), true))
                .collect::<Vec<_>>(),
        );
        Ok(Self { tok })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tok
            .encode(text, false)
            .expect("tokenize")
            .get_ids()
            .to_vec()
    }
}
