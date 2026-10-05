//! Tekken — Voxtral's tiktoken-style byte-level BPE — as a tokenizer graph in
//! the model pile, plus its DECODE-ONLY runtime (ASR emits ids; the runtime
//! only needs id → bytes).
//!
//! `tekken.json` is read exactly once, at import ([`TekkenSource`]): ids
//! `0..num_special` are special tokens, and id `i >= num_special` is
//! `vocab[i − num_special].token_bytes` (base64 of the raw bytes). The graph is
//! the existing `TIKTOKEN` shape from [`crate::tokenizer`] — vocab entries of
//! `{piece_bytes, token_id}`, special tokens as ordered `added` entries, one
//! pre-tokenizer node holding the split pattern — with every `token_id` the
//! ABSOLUTE model id, so the runtime ([`Tekken::from_graph`]) needs no offset:
//! an id with stored bytes decodes to them, every other id (specials, ids the
//! graph does not name) contributes nothing.
//!
//! The vocab is cut at `default_vocab_size` the way mistral_common's
//! `Tekkenizer` cuts it for this model: the file lists 150 000 pieces, the
//! model's head emits 131 072 ids, and the 19 928 pieces past it are
//! unreachable.

use std::collections::HashMap;

use crate::tokenizer::{attrs, flag, ty};
use triblespace::core::metadata;
use triblespace::prelude::*;

/// Minimal base64 (standard alphabet, `=` padding) — avoids a dep for one call.
fn b64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> i32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as i32,
            b'a'..=b'z' => (c - b'a') as i32 + 26,
            b'0'..=b'9' => (c - b'0') as i32 + 52,
            b'+' => 62,
            b'/' => 63,
            _ => -1,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in s.as_bytes() {
        let v = val(c);
        if v < 0 {
            continue; // '=' padding / whitespace
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

/// `tekken.json`, parsed once at import: what the graph will hold.
pub struct TekkenSource {
    num_special: u32,
    /// `pieces[i]` = raw bytes of model token id `i + num_special`, already
    /// cut at `default_vocab_size`.
    pieces: Vec<Vec<u8>>,
    /// `(token_str, is_control)` for ids `0..specials.len()`.
    specials: Vec<(String, bool)>,
    pattern: String,
}

impl TekkenSource {
    pub fn parse(json: &[u8]) -> anyhow::Result<Self> {
        let json: serde_json::Value = serde_json::from_slice(json)?;
        let config = &json["config"];
        let num_special = config["default_num_special_tokens"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("tekken.json: no config.default_num_special_tokens"))?;
        let pattern = config["pattern"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("tekken.json: no config.pattern"))?
            .to_owned();
        let vocab = json["vocab"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tekken.json: no vocab array"))?;
        let reachable = match config["default_vocab_size"].as_u64() {
            Some(size) => (size.saturating_sub(num_special) as usize).min(vocab.len()),
            None => vocab.len(),
        };
        let pieces = vocab[..reachable]
            .iter()
            .map(|e| {
                e["token_bytes"]
                    .as_str()
                    .map(b64_decode)
                    .ok_or_else(|| anyhow::anyhow!("tekken.json: vocab entry without token_bytes"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let specials = json["special_tokens"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tekken.json: no special_tokens array"))?
            .iter()
            .map(|e| {
                let text = e["token_str"].as_str().ok_or_else(|| {
                    anyhow::anyhow!("tekken.json: special token without token_str")
                })?;
                Ok((text.to_owned(), e["is_control"].as_bool() == Some(true)))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        anyhow::ensure!(
            specials.len() as u64 == num_special,
            "tekken.json: {} special tokens but default_num_special_tokens = {num_special}",
            specials.len()
        );
        Ok(Self {
            num_special: num_special as u32,
            pieces,
            specials,
            pattern,
        })
    }

    /// Ids this source assigns bytes to: `num_special..num_special + len`.
    pub fn piece_ids(&self) -> std::ops::Range<u32> {
        self.num_special..self.num_special + self.pieces.len() as u32
    }

    /// The raw bytes `tekken.json` gives model token `id`, if it gives any.
    pub fn piece(&self, id: u32) -> Option<&[u8]> {
        let index = id.checked_sub(self.num_special)?;
        self.pieces.get(index as usize).map(Vec::as_slice)
    }

    /// The tokenizer graph, rooted at the tokenizer entity and named
    /// `source_name`. Content-addressed throughout, so writing the same file
    /// twice yields the same root.
    pub fn to_fragment(
        &self,
        source_name: &str,
        blobs: &mut impl BlobStorePut,
    ) -> anyhow::Result<Fragment> {
        let mut facts = TribleSet::new();

        let mut vocab_ids = Vec::with_capacity(self.pieces.len());
        for (id, piece) in self.piece_ids().zip(&self.pieces) {
            let bytes = blobs.put::<blobencodings::RawBytes, _>(piece.clone())?;
            let entry = entity! { _ @ attrs::piece_bytes: bytes, attrs::token_id: id as u64 };
            vocab_ids.push(entry.root().expect("vocab entry root"));
            facts += entry.into_facts();
        }

        let mut added_ids = Vec::with_capacity(self.specials.len());
        for (id, (text, control)) in self.specials.iter().enumerate() {
            let text = blobs.put::<blobencodings::UTF8String, _>(text.clone())?;
            let tags: &[Id] = if *control { &[flag::SPECIAL] } else { &[] };
            let entry = entity! { _ @
                attrs::piece: text,
                attrs::token_id: id as u64,
                attrs::index: id as u64,
                metadata::tag*: tags.iter(),
            };
            added_ids.push(entry.root().expect("special token root"));
            facts += entry.into_facts();
        }

        let pattern = blobs.put::<blobencodings::UTF8String, _>(self.pattern.clone())?;
        let pre_tokenizer = entity! { _ @
            metadata::tag: ty::TIKTOKEN_PRE_TOKENIZER,
            attrs::pattern: pattern,
        };
        let pre_tokenizer_id = pre_tokenizer.root().expect("pre-tokenizer root");
        facts += pre_tokenizer.into_facts();

        let tokenizer = entity! { _ @
            metadata::tag: ty::TIKTOKEN,
            attrs::pre_tokenizer: pre_tokenizer_id,
            attrs::vocab*: vocab_ids.iter(),
            attrs::added*: added_ids.iter(),
        };
        let tokenizer_id = tokenizer.root().expect("tokenizer root");
        facts += tokenizer.into_facts();
        let name = blobs.put::<blobencodings::UTF8String, _>(source_name.to_owned())?;
        facts += entity! { ExclusiveId::force_ref(&tokenizer_id) @ attrs::model_name: name }
            .into_facts();
        Ok(Fragment::rooted(tokenizer_id, facts))
    }
}

/// The decode-only runtime: model token id → raw bytes.
pub struct Tekken {
    pieces: HashMap<u32, Vec<u8>>,
}

impl Tekken {
    /// Read the pieces of tokenizer `root` out of a graph. Vocab entries whose
    /// id is not a `u32` cannot be model ids and are skipped.
    pub fn from_graph(tribles: &TribleSet, blobs: &impl BlobStoreGet, root: Id) -> Self {
        let pieces = crate::tokenizer::load_tiktoken_ranks(tribles, blobs, root)
            .into_iter()
            .filter_map(|(bytes, id)| Some((u32::try_from(id).ok()?, bytes)))
            .collect();
        Self { pieces }
    }

    /// The Voxtral tokenizer — the one named [`super::SOURCE`] — out of a
    /// frozen model-collection snapshot.
    pub fn from_snapshot<R: BlobStoreGet>(
        snapshot: &crate::model_collection::ModelSnapshot<R>,
    ) -> anyhow::Result<Self> {
        let root = crate::selection::select_tokenizer_root(
            snapshot.facts(),
            snapshot.store(),
            crate::selection::TokenizerSelector::Name(super::SOURCE),
        )
        .map_err(|error| {
            error.context(
                "no Tekken tokenizer in the Voxtral pile; rerun voxtral_persist \
                 (it imports tekken.json beside the weights)",
            )
        })?;
        Ok(Self::from_graph(snapshot.facts(), snapshot.store(), root))
    }

    /// Number of ids with stored bytes.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Raw bytes of one token; empty for special and unknown ids.
    pub fn piece(&self, id: u32) -> &[u8] {
        self.pieces.get(&id).map_or(&[], Vec::as_slice)
    }

    /// Decode a token stream to text, skipping special ids (BOS/EOS/pads).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.piece(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::blob::MemoryBlobStore;
    use triblespace::core::repo::SnapshotSource;

    fn b64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n = chunk
                .iter()
                .enumerate()
                .fold(0u32, |acc, (i, &b)| acc | (b as u32) << (16 - 8 * i));
            for i in 0..4 {
                if i <= chunk.len() {
                    out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    /// Three specials, five pieces of which the vocab size admits four: 'ä'
    /// (C3 A4) is split across ids 4 and 5, and id 7 lies past the cut.
    fn synthetic_json() -> Vec<u8> {
        let pieces: [&[u8]; 5] = [b"a", b"\xc3", b"\xa4", b" b", b"zz"];
        let vocab: Vec<_> = pieces
            .iter()
            .enumerate()
            .map(|(rank, piece)| serde_json::json!({ "rank": rank, "token_bytes": b64(piece) }))
            .collect();
        serde_json::to_vec(&serde_json::json!({
            "config": {
                "pattern": "\\s+|\\S+",
                "default_vocab_size": 7,
                "default_num_special_tokens": 3,
            },
            "vocab": vocab,
            "special_tokens": [
                { "rank": 0, "token_str": "<unk>", "is_control": true },
                { "rank": 1, "token_str": "<s>", "is_control": true },
                { "rank": 2, "token_str": "</s>", "is_control": true },
            ],
        }))
        .unwrap()
    }

    #[test]
    fn tekken_round_trips_through_its_graph_at_absolute_ids() {
        let source = TekkenSource::parse(&synthetic_json()).unwrap();
        assert_eq!(source.piece_ids(), 3..7);

        let mut blobs = MemoryBlobStore::new();
        let fragment = source.to_fragment("voxtral-test", &mut blobs).unwrap();
        let facts = fragment.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let root = crate::selection::select_tokenizer_root(
            &facts,
            &reader,
            crate::selection::TokenizerSelector::Name("voxtral-test"),
        )
        .unwrap();
        let tekken = Tekken::from_graph(&facts, &reader, root);

        assert_eq!(tekken.len(), 4, "the piece past default_vocab_size is cut");
        for id in source.piece_ids() {
            assert_eq!(Some(tekken.piece(id)), source.piece(id), "id {id}");
        }
        // BOS, 'a', 'ä' split over two ids, ' b', EOS, an id past the cut.
        assert_eq!(tekken.decode(&[1, 3, 4, 5, 6, 2, 7]), "aä b");
    }

    #[test]
    fn rewriting_the_same_file_names_the_same_tokenizer() {
        let source = TekkenSource::parse(&synthetic_json()).unwrap();
        let mut blobs = MemoryBlobStore::new();
        let first = source.to_fragment("voxtral-test", &mut blobs).unwrap();
        let second = source.to_fragment("voxtral-test", &mut blobs).unwrap();
        assert_eq!(first.root(), second.root());
        let facts = first.into_facts() + second.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        crate::selection::select_tokenizer_root(
            &facts,
            &reader,
            crate::selection::TokenizerSelector::Name("voxtral-test"),
        )
        .expect("one tokenizer after a repeated import");
    }
}
