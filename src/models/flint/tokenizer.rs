//! Raw Flint GGUF text codec. No chat rendering, graph migration, model weights,
//! inference backend, Unicode normalization, or lossy per-token UTF-8 conversion.
//!
//! The three sequential splits are from llama.cpp b11853, commit
//! 1ddffe9f70a63e2d260d4bfad9ad200009007f62, src/llama-vocab.cpp:284-290.
//! Matching regex spelling is not itself a parity claim: Unicode tables differ
//! between engines. The independently generated native fixtures are the gate.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use tokenizers::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer};
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::{byte_level::ByteLevel, sequence::Sequence, split::{Split, SplitPattern}};

use crate::formats::gguf::index::{FileIdentity, MetadataEntry, MetadataLimits, MetadataValue, SelectedMetadata};

pub const REFERENCE_COMMIT: &str = "1ddffe9f70a63e2d260d4bfad9ad200009007f62";
pub const DECLARED_EOG_IDS: [u32; 4] = [150001, 150003, 150005, 150018];
/// Historical native discrepancy, NOT the default stopping policy. b11853
/// derives EOGs from spellings and omits the declared developer token.
pub const B11853_EOG_IDS: [u32; 3] = [150001, 150003, 150005];
pub const SPLITS: [&str; 3] = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
];

pub const METADATA_KEYS: &[&str] = &[
    "general.architecture", "tokenizer.ggml.model", "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_sep_token", "tokenizer.ggml.eog_token_ids", "tokenizer.chat_template",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind { Normal, Control, UserDefined }

#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeOptions {
    /// This artifact declares no automatic BOS/SEP and inherits no EOS insertion
    /// from b11853. Both values therefore produce the same IDs, independently of
    /// whether literal controls are recognized.
    pub add_special: bool,
    pub parse_special: bool,
}

struct Piece { kind: TokenKind, bytes: Vec<u8> }

pub struct FlintTokenizer {
    engine: Tokenizer,
    pieces: Vec<Piece>,
    /// Retained verbatim for a separate future template renderer, never executed
    /// by encode. Raw text is not silently sanitized as if it were chat content.
    pub chat_template: String,
}

impl FlintTokenizer {
    /// Reads only metadata through the existing bounded GGUF reader. The return
    /// value identifies that file observation; it is not a full-model digest.
    pub fn open(path: &Path) -> Result<(Self, FileIdentity, u64)> {
        let (identity, selected) = SelectedMetadata::open(path, METADATA_KEYS, MetadataLimits::TOKENIZER)?;
        Ok((Self::from_metadata(&selected.entries)?, identity, selected.end))
    }

    /// Binds the observed 150,144-entry Flint profile, not arbitrary GPT-2 GGUFs.
    pub fn from_metadata(entries: &[MetadataEntry]) -> Result<Self> {
        use MetadataValue::*;
        let optional = |key: &str| -> Result<Option<&MetadataValue>> {
            let mut matching = entries.iter().filter(|entry| entry.key == key);
            let first = matching.next();
            ensure!(matching.next().is_none(), "duplicate tokenizer metadata {key}");
            if let Some(entry) = first {
                ensure!(entry.gguf_type == entry.value.gguf_type(), "wrong tokenizer metadata type {key}");
            }
            Ok(first.map(|entry| &entry.value))
        };
        let get = |key: &str| optional(key)?.with_context(|| format!("missing tokenizer metadata {key}"));
        for (key, value) in [("general.architecture", "flint"), ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "flint")] {
            ensure!(get(key)? == &String(value.into()), "unsupported tokenizer profile: {key}");
        }
        ensure!(get("tokenizer.ggml.bos_token_id")? == &U32(150000), "unsupported BOS ID");
        ensure!(get("tokenizer.ggml.eos_token_id")? == &U32(150001), "unsupported EOS ID");
        for key in ["tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_sep_token"] {
            ensure!(get(key)? == &Bool(false), "unsupported automatic insertion: {key}");
        }
        // Absent in the artifact; false is the explicitly selected b11853 BPE default.
        ensure!(matches!(optional("tokenizer.ggml.add_eos_token")?, None | Some(Bool(false))), "unsupported automatic EOS insertion");
        let I32Array(eog) = get("tokenizer.ggml.eog_token_ids")? else { anyhow::bail!("EOG must be an INT32 array") };
        ensure!(eog.iter().copied().eq(DECLARED_EOG_IDS.map(|id| id as i32)), "unsupported declared Flint EOG set");
        let StringArray(tokens) = get("tokenizer.ggml.tokens")? else { anyhow::bail!("tokens must be a STRING array") };
        let I32Array(types) = get("tokenizer.ggml.token_type")? else { anyhow::bail!("token types must be an INT32 array") };
        let StringArray(merges) = get("tokenizer.ggml.merges")? else { anyhow::bail!("merges must be a STRING array") };
        let String(template) = get("tokenizer.chat_template")? else { anyhow::bail!("chat template must be a STRING") };
        ensure!(tokens.len() == 150144 && types.len() == tokens.len() && merges.len() == 149744, "unsupported Flint vocabulary/merge dimensions");
        let kinds = types.iter().enumerate().map(|(id, &tag)| {
            let (expected, kind) = match id {
                0..=149999 => (1, TokenKind::Normal),
                150006..=150017 => (4, TokenKind::UserDefined),
                _ => (3, TokenKind::Control),
            };
            ensure!(tag == expected, "unexpected token type at {id}: {tag}");
            Ok(kind)
        }).collect::<Result<Vec<_>>>()?;
        for (offset, spelling) in [
            "<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
            "<think>", "</think>", "<tools>", "</tools>", "<tool_call>", "</tool_call>",
            "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>", "</tool_response>", "<|developer|>",
        ].iter().enumerate() {
            ensure!(tokens[150000 + offset] == *spelling, "unexpected Flint marker at {}", 150000 + offset);
        }
        for id in 150019..150144 {
            ensure!(tokens[id] == format!("<|reserved_{id}|>"), "unexpected reserved token {id}");
        }
        let merges = merges.iter().map(|merge| {
            // Native finds the separator after character 0, preserving array rank.
            let split = merge.as_bytes().iter().enumerate().skip(1).find_map(|(i, &b)| (b == b' ').then_some(i))
                .with_context(|| format!("invalid BPE merge {merge:?}"))?;
            ensure!(split + 1 < merge.len(), "empty right BPE merge operand");
            Ok((merge[..split].to_owned(), merge[split + 1..].to_owned()))
        }).collect::<Result<Vec<_>>>()?;
        Self::build(tokens, &kinds, merges, template.clone())
    }

    fn build(tokens: &[String], kinds: &[TokenKind], merges: Vec<(String, String)>, chat_template: String) -> Result<Self> {
        ensure!(tokens.len() == kinds.len(), "token/type length mismatch");
        let vocab: Vocab = tokens.iter().enumerate().map(|(id, text)| (text.clone(), id as u32)).collect();
        ensure!(vocab.len() == tokens.len(), "duplicate vocabulary spelling");
        let model = BPE::builder().vocab_and_merges(vocab, merges).ignore_merges(false)
            .byte_fallback(false).build().map_err(|error| anyhow!("Flint BPE: {error}"))?;
        let mut engine = Tokenizer::new(model);
        let mut splitters = Vec::<PreTokenizerWrapper>::new();
        for pattern in SPLITS {
            splitters.push(Split::new(SplitPattern::Regex(pattern.into()), SplitDelimiterBehavior::Isolated, false)
                .map_err(|error| anyhow!("Flint split: {error}"))?.into());
        }
        // No fourth GPT-2 regex and no synthetic prefix space.
        splitters.push(ByteLevel::new(false, false, false).into());
        engine.with_pre_tokenizer(Some(Sequence::new(splitters)));
        let added = tokens.iter().zip(kinds).filter_map(|(text, kind)| {
            (*kind != TokenKind::Normal).then(|| AddedToken::from(text.clone(), *kind == TokenKind::Control)
                .normalized(false).single_word(false).lstrip(false).rstrip(false))
        }).collect::<Vec<_>>();
        engine.add_tokens(&added);
        for (id, text) in tokens.iter().enumerate() {
            ensure!(engine.token_to_id(text) == Some(id as u32), "added token changed vocabulary ID {id}");
        }
        let mut char_bytes = [None; 512];
        for (byte, c) in byte_chars().into_iter().enumerate() { char_bytes[c as usize] = Some(byte as u8); }
        let pieces = tokens.iter().zip(kinds).map(|(text, &kind)| {
            let bytes = if kind == TokenKind::Normal {
                text.chars().map(|c| char_bytes.get(c as usize).copied().flatten()
                    .with_context(|| format!("normal token contains non-byte-alphabet character {c:?}")))
                    .collect::<Result<Vec<_>>>()?
            } else { text.as_bytes().to_vec() };
            Ok(Piece { kind, bytes })
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self { engine, pieces, chat_template })
    }

    pub fn encode(&mut self, text: &str, options: EncodeOptions) -> Result<Vec<u32>> {
        // HF's switch means encode special spellings as ordinary text: inverse
        // of llama's parse_special. Non-special USER_DEFINED tokens still match.
        self.engine.set_encode_special_tokens(!options.parse_special);
        // This profile has no automatic insertions; intentionally no processor.
        let _ = options.add_special;
        Ok(self.engine.encode(text, false).map_err(|error| anyhow!("Flint encode: {error}"))?.get_ids().to_vec())
    }

    /// Exact bytes, potentially an incomplete UTF-8 prefix. A streaming caller
    /// must carry those bytes across token boundaries, not lossy-decode each piece.
    pub fn token_bytes(&self, id: u32, render_controls: bool) -> Result<&[u8]> {
        let piece = self.pieces.get(id as usize).context("token ID outside Flint vocabulary")?;
        Ok(if piece.kind == TokenKind::Control && !render_controls { &[] } else { &piece.bytes })
    }

    pub fn decode_bytes(&self, ids: &[u32], render_controls: bool) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for &id in ids { bytes.extend_from_slice(self.token_bytes(id, render_controls)?); }
        Ok(bytes)
    }

    /// Strict whole-sequence UTF-8 convenience; arbitrary generated byte tokens
    /// can be invalid/incomplete, in which case use decode_bytes instead.
    pub fn decode(&self, ids: &[u32], render_controls: bool) -> Result<String> {
        String::from_utf8(self.decode_bytes(ids, render_controls)?).context("decoded tokens are not complete UTF-8")
    }

    /// Check IDs before rendering: hiding control spellings does not hide EOG.
    pub fn is_eog(&self, id: u32) -> bool { DECLARED_EOG_IDS.contains(&id) }
    pub fn vocab_size(&self) -> usize { self.pieces.len() }
}

// GPT-2's reversible byte alphabet. Keep bytes, not a lossy Decoder string API.
fn byte_chars() -> [char; 256] {
    let mut next = 256;
    std::array::from_fn(|byte| {
        let byte = byte as u32;
        let code = if matches!(byte, 33..=126 | 161..=172 | 174..=255) { byte } else {
            let code = next; next += 1; code
        };
        char::from_u32(code).expect("GPT-2 alphabet is within Unicode scalar range")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic byte vocabulary tests mechanics, never asserted native oracle IDs.
    fn synthetic() -> FlintTokenizer {
        let mut tokens: Vec<_> = byte_chars().iter().map(char::to_string).collect();
        tokens.extend(["<|user|>", "<think>", "</think>"].map(str::to_owned));
        let mut kinds = vec![TokenKind::Normal; 256];
        kinds.extend([TokenKind::Control, TokenKind::UserDefined, TokenKind::UserDefined]);
        FlintTokenizer::build(&tokens, &kinds, vec![], String::new()).unwrap()
    }

    #[test]
    fn controls_and_user_defined_are_not_conflated() {
        let mut codec = synthetic();
        for add_special in [false, true] {
            let literal = codec.encode("<|user|><think>x</think>", EncodeOptions { add_special, parse_special: false }).unwrap();
            assert!(!literal.contains(&256));
            assert!(literal.contains(&257) && literal.contains(&258));
            let parsed = codec.encode("<|user|><think>x</think>", EncodeOptions { add_special, parse_special: true }).unwrap();
            assert_eq!(parsed, [256, 257, b'x' as u32, 258]);
            assert_eq!(codec.decode(&parsed, false).unwrap(), "<think>x</think>");
            assert_eq!(codec.decode(&parsed, true).unwrap(), "<|user|><think>x</think>");
        }
    }

    #[test]
    fn byte_decode_preserves_incomplete_utf8_without_replacement() {
        let mut codec = synthetic();
        for text in ["", " a\t\r\n b  ", "café", "cafe\u{301}", "中文かなカナ𠀀", "👩🏽‍💻🙂", "a\0b", "\u{a0}\u{2028}\u{3000}"] {
            let ids = codec.encode(text, EncodeOptions::default()).unwrap();
            assert_eq!(codec.decode_bytes(&ids, false).unwrap(), text.as_bytes());
            let streaming: Vec<u8> = ids.iter().flat_map(|&id| codec.token_bytes(id, false).unwrap().iter().copied()).collect();
            assert_eq!(streaming, text.as_bytes());
            assert_eq!(codec.decode(&ids, false).unwrap(), text);
        }
        assert_eq!(codec.token_bytes(0xf0, false).unwrap(), [0xf0]);
        assert!(codec.decode(&[0xf0], false).is_err());
        assert_eq!(codec.decode(&[0xf0, 0x9f, 0x99, 0x82], false).unwrap(), "🙂");
        assert!(codec.token_bytes(999999, true).is_err());
    }

    #[test]
    fn declared_stops_do_not_silently_copy_native_omission() {
        let codec = synthetic();
        for id in DECLARED_EOG_IDS { assert!(codec.is_eog(id)); }
        assert!(!B11853_EOG_IDS.contains(&150018));
        for id in [150000, 150002, 150004, 150006, 150007, 150011, 150017, 150019] { assert!(!codec.is_eog(id)); }
    }

    #[test]
    fn three_splits_preserve_prior_number_and_cjk_boundaries() {
        use tokenizers::{PreTokenizer, PreTokenizedString, OffsetReferential, OffsetType};
        let splitters = SPLITS.into_iter().map(|pattern| {
            Split::new(SplitPattern::Regex(pattern.into()), SplitDelimiterBehavior::Isolated, false).unwrap().into()
        }).collect();
        let mut text = PreTokenizedString::from("abc1234中文かなカナxyz");
        Sequence::new(splitters).pre_tokenize(&mut text).unwrap();
        let pieces: Vec<_> = text.get_splits(OffsetReferential::Original, OffsetType::Byte)
            .iter().map(|(text, _, _)| *text).collect();
        assert_eq!(pieces, ["abc", "123", "4", "中文かなカナ", "xyz"]);
    }
}
