//! WeMM's pinned, bounded single-user template form, not a Jinja interpreter.
//! Tokenizer postprocessing remains the checkpoint's own executor: no manual
//! embedding-token append, padding or truncation. Other template bytes fail.
use super::{multimodal_layout::ImagePlan, prepared, vision_geometry::Grid};
use sha2::{Digest, Sha256};

pub const TOKENIZER_SHA256: &str =
    "40e444c744512f423da4c8443c47c21e22ff76056ba4e9796a81c04c13a9daf0";
pub const TEMPLATE_SHA256: &str =
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80";
pub const GRID: Grid = Grid {
    frames: 1,
    height: 16,
    width: 16,
};

/// B1, unpadded, at most 256 IDs, with exactly one terminal embedding token.
pub struct Tokens {
    ids: Vec<u32>,
}
impl Tokens {
    pub fn as_slice(&self) -> &[u32] {
        &self.ids
    }
}

#[derive(Clone)]
pub struct InputCodec {
    tokenizer: tokenizers::Tokenizer,
    /// [`content_bytes_per_id`] of `tokenizer`, for [`Self::surely_exceeds`].
    content_bytes_per_id: Option<usize>,
}
impl InputCodec {
    /// From the checkpoint's own `tokenizer.json` bytes, pinned by
    /// [`TOKENIZER_SHA256`] (the behavior fixtures' path). No discovery here.
    pub fn from_assets(tokenizer_json: &[u8], chat_template: &[u8]) -> Result<Self, String> {
        if format!("{:x}", Sha256::digest(tokenizer_json)) != TOKENIZER_SHA256 {
            return Err("unsupported WeMM tokenizer asset bytes".into());
        }
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json).map_err(|e| e.to_string())?;
        Self::from_tokenizer(tokenizer, chat_template)
    }

    /// From a tokenizer the caller already holds: the one rebuilt from a
    /// model root's tokenizer facts (`crate::tokenizer::build_tokenizer`),
    /// which restores the checkpoint's post-processor and so appends the
    /// terminal `<embedding>` itself. The template is pinned by its bytes.
    pub fn from_tokenizer(
        tokenizer: tokenizers::Tokenizer,
        chat_template: &[u8],
    ) -> Result<Self, String> {
        if format!("{:x}", Sha256::digest(chat_template)) != TEMPLATE_SHA256 {
            return Err("unsupported WeMM template asset bytes".into());
        }
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
            return Err("pinned tokenizer unexpectedly enables padding/truncation".into());
        }
        let content_bytes_per_id = content_bytes_per_id(&tokenizer);
        Ok(Self {
            tokenizer,
            content_bytes_per_id,
        })
    }

    /// True only if [`Self::text`] of `text` refuses or frames more than
    /// `max_ids` IDs, decided without tokenizing: the content [`Self::text`]
    /// encodes holds more bytes other than `\n` than `max_ids` IDs can stand
    /// for ([`content_bytes_per_id`]). False says nothing. A caller may skip
    /// [`Self::text`] when this is true and gets exactly the answer it would
    /// have computed; it is never true for a tokenizer the bound is not
    /// proven for.
    pub fn surely_exceeds(&self, text: &str, max_ids: usize) -> bool {
        let Some(per_id) = self.content_bytes_per_id else {
            return false;
        };
        let limit = per_id.saturating_mul(max_ids);
        let content = template_trim(text);
        content.len() > limit && content.bytes().filter(|&b| b != b'\n').nth(limit).is_some()
    }

    /// True only if [`Self::text`] of `longer` returns exactly what it
    /// returns for `shorter`, decided without tokenizing, where `longer`
    /// starts with `shorter` (the caller's promise). [`Self::text`] reads its
    /// argument twice: its raw length against [`MAX_TEXT_BYTES`], and
    /// otherwise only `template_trim` of it. Characters the template trims
    /// at the end change neither trimmed result, so when every character
    /// `longer` adds is one of them, both texts give one answer unless the
    /// added bytes carry `longer` past [`MAX_TEXT_BYTES`]. When `shorter` is
    /// already past it, both are the same refusal. [`Self::surely_exceeds`]
    /// reads only the trimmed content too, so it agrees as well. False says
    /// nothing. The cost is the added bytes, not the whole text.
    pub fn same_text(shorter: &str, longer: &str) -> bool {
        shorter.len() > MAX_TEXT_BYTES
            || (longer.len() <= MAX_TEXT_BYTES
                && longer
                    .get(shorter.len()..)
                    .is_some_and(|added| added.chars().all(template_space)))
    }

    /// Supported template: one user text message, no generation prompt/tools.
    /// Trim semantics match this exact template's render_content|trim branch.
    pub fn text(&self, text: &str) -> Result<Tokens, String> {
        if text.len() > MAX_TEXT_BYTES {
            return Err("bounded WeMM text exceeds1MiB; nothing truncated".into());
        }
        let content = template_trim(text);
        if content.starts_with("<tool_response>") && content.ends_with("</tool_response>") {
            return Err(
                "the pinned template rejects a tool response as its only user query".into(),
            );
        }
        let ids = self.encode(content)?;
        prepared::validate_ids(&ids)?;
        Ok(Tokens { ids })
    }

    /// One still image, no caption, exactly 64 merged feature placeholders.
    /// The 256-patch budget is explicit; it is not upstream arbitrary-resolution
    /// image processing, video, multi-turn chat, or batching.
    pub fn image(&self) -> Result<Tokens, String> {
        let text = format!(
            "<|vision_start|>{}<|vision_end|>",
            "<|image_pad|>".repeat(64)
        );
        let ids = self.encode(&text)?;
        ImagePlan::new(&ids, GRID, 256, 64)?;
        Ok(Tokens { ids })
    }

    fn encode(&self, content: &str) -> Result<Vec<u32>, String> {
        let framed = format!("<|im_start|>user\n{content}<|im_end|>\n");
        let encoded = self
            .tokenizer
            .encode(framed, true)
            .map_err(|e| e.to_string())?;
        let ids = encoded.get_ids();
        if !(1..=256).contains(&ids.len())
            || ids.last() != Some(&248077)
            || ids.iter().filter(|&&i| i == 248077).count() != 1
        {
            return Err("WeMM requires<=256 unpadded IDs and one terminal embedding token; nothing truncated".into());
        }
        Ok(ids.to_vec())
    }
}

/// The content of a user message as the pinned template renders it:
/// `render_content|trim`. Jinja's trim uses Python's whitespace predicate,
/// which also includes the four ASCII separators not in Rust's Unicode
/// White_Space property.
fn template_trim(text: &str) -> &str {
    text.trim_matches(template_space)
}
fn template_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// The longest text [`InputCodec::text`] reads, in bytes, before trimming.
pub const MAX_TEXT_BYTES: usize = 1024 * 1024;

/// Twice the most UTF-8 bytes NFC can turn into one byte: 7, so the ratio is
/// 3.5. [`tests::nfc_shrinks_utf8_at_most_three_and_a_half_times`] proves it
/// against the tables of the NFC the tokenizer itself runs.
const NFC_SHRINK_TWICE: usize = 7;

/// The most content bytes other than `\n` that one ID of `tokenizer` can
/// stand for, or `None` unless the tokenizer has a shape this is proven for.
///
/// The bound is `ceil(3.5 * max(L, A))`, where `L` is the longest vocabulary
/// entry in bytes and `A` the longest added token. It holds because every
/// step of the shape below keeps every byte other than `\n` or shrinks it by
/// at most 3.5:
/// - Added tokens are split out first, each one ID for at most `A` bytes
///   (raw, or normalized when the token is). Every one must have
///   `lstrip` and `rstrip` off, which would let it swallow whitespace.
/// - The normalizer is absent, NFC, the checkpoint's `Replace("\n$", "")`,
///   or a sequence of those. The Replace deletes newlines: under Oniguruma
///   `$` is end of LINE, so every `\n` before another `\n` goes, and a run
///   of newlines of any length becomes at most one. That is why the measure
///   leaves out `\n` and a plain byte length would be unsound. It deletes
///   nothing else. NFC never makes or consumes `\n`, and turns `b` bytes into
///   at least `2b / 7` ([`NFC_SHRINK_TWICE`]).
/// - The pre-tokenizer is a sequence of non-removing Splits ending in
///   ByteLevel, so the model sees one character per byte and nothing is
///   dropped.
/// - The model is BPE whose every vocabulary entry is ByteLevel characters
///   and which holds all 256 of them, so no character is unknown or
///   dropped and every token covers at most `L` bytes.
/// - The post-processor, if any, is a template, which only adds IDs.
///
/// So the normalized text, which holds at least `2/7` of the content's
/// bytes other than `\n`, is exactly covered by the IDs, each at most
/// `max(L, A)` bytes. For the pinned checkpoint `L = 128`, `A = 21`: 448.
fn content_bytes_per_id(tokenizer: &tokenizers::Tokenizer) -> Option<usize> {
    use serde_json::{Value, json};
    fn as_json<T: serde::Serialize>(value: &T) -> Option<Value> {
        serde_json::to_value(value).ok()
    }
    let nfc = json!({ "type": "NFC" });
    let newline = json!({ "type": "Replace", "pattern": { "Regex": "\\n$" }, "content": "" });
    let normalizer_ok = match tokenizer.get_normalizer().map(as_json) {
        None => true,
        Some(Some(n)) if n == nfc || n == newline => true,
        Some(Some(n)) if n["type"] == "Sequence" => n["normalizers"]
            .as_array()
            .is_some_and(|steps| steps.iter().all(|s| *s == nfc || *s == newline)),
        _ => false,
    };
    let split_ok = |s: &Value| {
        s["type"] == "Split"
            && matches!(
                s["behavior"].as_str(),
                Some("Isolated" | "MergedWithPrevious" | "MergedWithNext" | "Contiguous")
            )
    };
    let pre_ok = match tokenizer.get_pre_tokenizer().map(as_json) {
        Some(Some(p)) if p["type"] == "ByteLevel" => true,
        Some(Some(p)) if p["type"] == "Sequence" => p["pretokenizers"]
            .as_array()
            .and_then(|steps| steps.split_last())
            .is_some_and(|(last, splits)| {
                last["type"] == "ByteLevel" && splits.iter().all(split_ok)
            }),
        _ => false,
    };
    let post_ok = match tokenizer.get_post_processor().map(as_json) {
        None => true,
        Some(p) => p.is_some_and(|p| p["type"] == "TemplateProcessing"),
    };
    if !(normalizer_ok && pre_ok && post_ok) {
        return None;
    }
    let tokenizers::ModelWrapper::BPE(bpe) = tokenizer.get_model() else {
        return None;
    };
    let alphabet = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet();
    let vocab = bpe.get_vocab();
    let mut longest = 0;
    for entry in vocab.keys() {
        let mut chars = 0;
        for c in entry.chars() {
            if !alphabet.contains(&c) {
                return None;
            }
            chars += 1;
        }
        longest = longest.max(chars);
    }
    if !alphabet.iter().all(|c| vocab.contains_key(&c.to_string())) {
        return None;
    }
    for token in tokenizer.get_added_tokens_decoder().values() {
        if token.lstrip || token.rstrip {
            return None;
        }
        longest = longest.max(token.content.len());
    }
    Some((longest * NFC_SHRINK_TWICE).div_ceil(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_unpinned_assets() {
        assert!(InputCodec::from_assets(b"{}", b"template").is_err());
    }
    /// [`NFC_SHRINK_TWICE`], against the NFC the tokenizer runs (its own
    /// tables, Unicode 9). NFC is the canonical decomposition of every input
    /// character followed by composition, and a composed output character's
    /// full decomposition is exactly the decomposed characters it absorbed.
    /// Give each input character to the output character that receives the
    /// first character of its decomposition: an output character `o` then
    /// gets at most one input character per character `d` of `o`'s
    /// decomposition, each no longer than the longest character whose
    /// decomposition starts with `d`. So no string shrinks by more than the
    /// largest such sum over `o`'s length, which this computes over every
    /// code point.
    #[test]
    fn nfc_shrinks_utf8_at_most_three_and_a_half_times() {
        use std::collections::HashMap;
        use tokenizers::NormalizedString;
        let nfd = |c: char| NormalizedString::from(c.to_string()).nfd().get().to_owned();
        let chars = || (0..=0x10FFFFu32).filter_map(char::from_u32);
        let mut longest_into: HashMap<char, usize> = HashMap::new();
        for c in chars() {
            let first = nfd(c)
                .chars()
                .next()
                .expect("a decomposition is never empty");
            let at = longest_into.entry(first).or_default();
            *at = (*at).max(c.len_utf8());
        }
        let mut worst = (0, 1, '\0');
        for o in chars() {
            let absorbed: usize = nfd(o)
                .chars()
                .map(|d| longest_into.get(&d).copied().unwrap_or(d.len_utf8()))
                .sum();
            if absorbed * worst.1 > worst.0 * o.len_utf8() {
                worst = (absorbed, o.len_utf8(), o);
            }
        }
        eprintln!(
            "worst NFC shrink {} bytes into {} at U+{:04X}",
            worst.0, worst.1, worst.2 as u32
        );
        assert!(2 * worst.0 <= NFC_SHRINK_TWICE * worst.1, "{worst:?}");
        assert_eq!(longest_into.get(&'\n'), Some(&1));
    }

    #[test]
    fn same_text_only_for_trimmed_additions_within_the_size_bound() {
        assert!(InputCodec::same_text("a b", "a b"));
        assert!(InputCodec::same_text(
            "a b",
            "a b \n\t\r\u{1c}\u{1f}\u{85}\u{3000}"
        ));
        assert!(!InputCodec::same_text("a b", "a b c"));
        assert!(!InputCodec::same_text("a", "a\n\u{200b}"));
        let x = "x".repeat(MAX_TEXT_BYTES + 2);
        let at = |n: usize| &x[..n];
        let padded = |n: usize, pad: usize| format!("{}{}", at(n), "\n".repeat(pad));
        assert!(InputCodec::same_text(
            at(MAX_TEXT_BYTES - 2),
            &padded(MAX_TEXT_BYTES - 2, 2)
        ));
        assert!(!InputCodec::same_text(
            at(MAX_TEXT_BYTES - 2),
            &padded(MAX_TEXT_BYTES - 2, 3)
        ));
        assert!(InputCodec::same_text(
            at(MAX_TEXT_BYTES + 1),
            at(MAX_TEXT_BYTES + 2)
        ));
    }

    #[test]
    #[ignore = "requires exact local checkpoint assets, metadata only; no GPU"]
    fn surely_exceeds_is_proven_for_the_checkpoint_and_agrees_with_text() {
        let root = std::path::PathBuf::from(
            std::env::var("WEMM_CHECKPOINT_DIR").expect("checkpoint directory"),
        );
        let codec = InputCodec::from_assets(
            &std::fs::read(root.join("tokenizer.json")).unwrap(),
            &std::fs::read(root.join("chat_template.jinja")).unwrap(),
        )
        .unwrap();
        assert_eq!(codec.content_bytes_per_id, Some(448));
        let ids = |s: &str| codec.text(s).map(|t| t.as_slice().len());
        // A run of newlines is deleted by the normalizer: bytes alone bound
        // nothing, which is why `\n` is not counted.
        let runs = format!("a{}b", "\n".repeat(200_000));
        assert!(ids(&runs).unwrap() <= 16);
        assert!(!codec.surely_exceeds(&runs, 248));
        // Whitespace the template trims is not content.
        let padded = format!("a{}", " ".repeat(200_000));
        assert!(!codec.surely_exceeds(&padded, 248));
        assert_eq!(ids(&padded).unwrap(), ids("a").unwrap());
        // The longest token, repeated: the densest ordinary content there is.
        let spaces = format!("x{}x", " ".repeat(448 * 248));
        assert!(codec.surely_exceeds(&format!("{spaces}y"), 248));
        assert!(ids(&spaces).map_or(true, |n| n > 248));
        // Wherever same_text says yes, text gives the same answer.
        let answer = |s: &str| codec.text(s).map(|t| t.as_slice().to_vec());
        let near = "y".repeat(MAX_TEXT_BYTES - 3);
        for (shorter, added) in [
            ("one two", "\n\n\n"),
            ("one two\n", " \t\u{1c}\u{3000}\n"),
            ("", "\n"),
            (near.as_str(), "\n\n\n"),
            (near.as_str(), "\n\n\n\n"),
        ] {
            let longer = format!("{shorter}{added}");
            if InputCodec::same_text(shorter, &longer) {
                assert_eq!(answer(shorter), answer(&longer), "{added:?}");
            }
        }
        // Wherever it says yes, text agrees, on a few shapes near the bound.
        for unit in [
            "word ",
            "\u{1ee8}",
            "U\u{308}\u{304}",
            "<|endoftext|>",
            "\u{ac00}\n\n",
        ] {
            for n in [10, 1000, 30_000, 120_000] {
                let s = unit.repeat(n);
                if codec.surely_exceeds(&s, 248) {
                    assert!(ids(&s).map_or(true, |n| n > 248), "{unit:?} x{n}");
                }
            }
        }
    }

    #[test]
    #[ignore = "requires exact local checkpoint assets, metadata only; no GPU"]
    fn pinned_tokens_match_the_frozen_behavior_fixture() {
        let root = std::path::PathBuf::from(
            std::env::var("WEMM_CHECKPOINT_DIR").expect("checkpoint directory"),
        );
        let codec = InputCodec::from_assets(
            &std::fs::read(root.join("tokenizer.json")).unwrap(),
            &std::fs::read(root.join("chat_template.jinja")).unwrap(),
        )
        .unwrap();
        let fixture = super::super::image_prepare::tests::fixture();
        for item in fixture["items"].as_array().unwrap() {
            let tokens = if item["modality"] == "text" {
                codec.text(item["text"].as_str().unwrap()).unwrap()
            } else {
                codec.image().unwrap()
            };
            let expected: Vec<u32> = serde_json::from_value(item["ids"].clone()).unwrap();
            assert_eq!(tokens.as_slice(), expected, "{}", item["id"]);
        }
        assert!(codec.text(&"x ".repeat(1024)).is_err());
        assert!(codec.text("<embedding>").is_err());
        assert!(codec.text("<tool_response>x</tool_response>").is_err());
        assert_eq!(
            codec.text("\u{1c} hello \u{1f}").unwrap().as_slice(),
            codec.text("hello").unwrap().as_slice()
        );
    }
}
