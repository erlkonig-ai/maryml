//! mary — tokenizers as content-addressed graphs in TribleSpace.
//!
//! The companion to [`crate::format`]: where `format` decomposes a model's
//! *weights* into self-describing tensor leaves, this module decomposes a
//! model's *tokenizer* into tribles — so the tokenizer travels **with** the
//! pile instead of riding along as an opaque `tokenizer.json` side-file in the
//! HuggingFace cache (which evaporates the moment the cache is cleared under
//! disk pressure — the failure that motivated this).
//!
//! Two layers, mirroring the two very different cost profiles of a tokenizer:
//!
//!   - **the bulk** — `vocab` (token↔id), `merges` (ordered BPE pairs), and
//!     `added_tokens`. Uniform, content-addressed, dedups across shared vocabs.
//!     Represented here as proper tribles (this module).
//!   - **the config tail** — normalizer / pre-tokenizer / post-processor /
//!     decoder: a small typed AST whose nodes each map 1:1 to a `tokenizers`
//!     constructor call. Types and boolean options are `metadata::tag`
//!     discriminants; regex / behaviour / prefix are flat attrs; `Sequence`
//!     children are ordered `member` edges.
//!
//! The post-processor is stored since 2026-11-07. It used to be left out on
//! the grounds that mary hand-frames its sentinels (`[CLS]`/`[SEP]`, bos/eos,
//! see `embed.rs`), and every caller that does so encodes with
//! `add_special_tokens = false`, which never runs it. WeMM is the model that
//! made that false: its checkpoint's own `TemplateProcessing` appends the
//! terminal `<embedding>` token, and its codec relies on the tokenizer, not on
//! hand framing, to put it there. A tokenizer rebuilt without it encodes
//! WeMM's input one token short. Graphs written before this carry no
//! post-processor edge and rebuild exactly as before.
//!
//! Added tokens carry their `tokenizers::AddedToken` options as flags since
//! the same date: `flag::NOT_SPECIAL`, `flag::NORMALIZED`, `flag::LSTRIP`,
//! `flag::RSTRIP`, `flag::SINGLE_WORD`. `NOT_SPECIAL` is inverted on purpose.
//! Every reader before it rebuilt every added token as special, so the absent
//! flag has always meant "special", on graphs old and new alike; a positive
//! `SPECIAL` flag read with presence semantics would have turned every
//! existing graph's sentinels into ordinary added tokens. The other four
//! default to false in `AddedToken::from(content, true)`, which is what old
//! graphs rebuilt with, so their absence means the same thing in both.
//!
//! Load is **construct-from-graph, no JSON**: [`build_tokenizer`] (feature
//! `tokenizer`) queries the graph and feeds the parts into `tokenizers`'
//! programmatic builders (`Tokenizer::new(model)`, `.with_normalizer(..)`, …)
//! — the exact shape of how `format` reads the weight graph and builds Burn
//! tensors; the `tokenizers` crate is purely the executor, as Burn is for
//! weights. JSON is touched exactly once, at ingest, the way
//! `ingest::save_safetensors` reads a safetensors file once.
//!
//! Attribute provenance: the structural + bulk ids below were minted with
//! `trible genid` on 2026-07-16 (recorded on compass goal 67b09f72). The
//! scaffolding attributes (`model_name`, `kind`, `index`, `member`) are
//! **reused** from [`crate::format`] by re-declaring their exact hex — the same
//! reuse idiom `models/gemma/lora.rs` uses for `model_name`.

use std::collections::HashMap;
use triblespace::core::metadata;
use triblespace::prelude::*;

type Err = Box<dyn std::error::Error>;

/// A [`BlobStorePut`] that counts and times the puts passing through it.
///
/// Ingesting a tokenizer is dominated by blob operations rather than by bytes —
/// Inkling's `tokenizer.json` is 199 998 vocab entries plus 446 189 merges, and
/// the naive shape interns each merge's two halves separately, so 2 MB of text
/// costs over a million puts. The rate that matters is therefore puts per
/// second on a REAL pile, and it has to be counted where the puts happen: a
/// driver that derives the number from `vocab.len() + 2 * merges.len()` is a
/// second transcription of the ingest's control flow and drifts the moment the
/// ingest changes, which is exactly when the number is being looked at.
pub struct CountingBlobs<'a, B: BlobStorePut> {
    inner: &'a mut B,
    pub puts: u64,
    pub nanos: u64,
}

impl<'a, B: BlobStorePut> CountingBlobs<'a, B> {
    pub fn new(inner: &'a mut B) -> Self {
        CountingBlobs {
            inner,
            puts: 0,
            nanos: 0,
        }
    }

    /// Puts per second, or `None` when nothing was put.
    pub fn rate(&self) -> Option<f64> {
        match self.nanos {
            0 => None,
            n => Some(self.puts as f64 * 1e9 / n as f64),
        }
    }
}

impl<'a, B: BlobStorePut> BlobStorePut for CountingBlobs<'a, B> {
    type PutError = B::PutError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<inlineencodings::Handle<S>>, Self::PutError>
    where
        S: triblespace::core::blob::BlobEncoding + 'static,
        T: triblespace::core::blob::IntoBlob<S>,
        inlineencodings::Handle<S>: triblespace::core::inline::InlineEncoding,
    {
        let t = std::time::Instant::now();
        let h = self.inner.put(item);
        self.nanos += t.elapsed().as_nanos() as u64;
        self.puts += 1;
        h
    }
}

// ── classification concepts, via the canonical `metadata::tag` (presence = "this
// concept applies"). There is NO `metadata::kind` attribute — tags ARE the
// classifier (KIND_TAG = kind discriminants, KIND_MULTI = multiple simultaneous
// kinds). Minted 2026-07-16 (compass 67b09f72). `ty::*` are the type
// discriminants (role-agnostic — the *edge* gives the role); `flag::*` the bools.
#[allow(dead_code)]
pub(crate) mod ty {
    use triblespace::macros::id_hex;
    use triblespace::prelude::Id;
    pub const BERT_NORMALIZER: Id = id_hex!("AC009EBADB3488042EDCD3D2C8648342");
    pub const SEQUENCE: Id = id_hex!("33E6588376EEEDCBD4CD16DA197B7F90");
    pub const NFC: Id = id_hex!("4D9204FC08700571D2407B74C0348DE0");
    pub const REPLACE: Id = id_hex!("48927BD29F9321104585A69B1B541812");
    pub const LOWERCASE: Id = id_hex!("C670B7A8105E77D2F6EF0DA05A8F5C99");
    pub const BERT_PRE_TOKENIZER: Id = id_hex!("A6259970638C680F95F061D9EA7F2975");
    pub const SPLIT: Id = id_hex!("B39A4D683D9B44F7D72BE118DA8E46BD");
    pub const BYTE_LEVEL: Id = id_hex!("4FDB5C7C0999B4DECA894AD75E5A94C6");
    pub const WORD_PIECE: Id = id_hex!("85A060015E6E94F70479E0E6B6BE0D98");
    pub const BPE: Id = id_hex!("71F4DAC1D6392375923D7A2A9FA53650");
    // ── minted 2026-08-05 (`trible genid`) for the tiktoken lane ──
    /// Model type: **tiktoken-style byte-level BPE**. Distinct from `BPE`, and
    /// not a flavour of it: there is no merges list at all — the rank table IS
    /// the merge order (a pair merges iff its concatenation is a token, at that
    /// token's rank), so a `TIKTOKEN` tokenizer has `vocab*` but never `merge*`,
    /// and its pieces are raw bytes (`attrs::piece_bytes`) rather than the
    /// byte-level-unicode strings a HuggingFace `tokenizer.json` carries.
    pub const TIKTOKEN: Id = id_hex!("BB88D3A2CAFF2AAA7DC00CA78945CC2E");
    /// Pre-tokenizer type: **one fancy-regex `find_iter` over `pat_str`**. Not
    /// `SPLIT` (which splits *on* a delimiter and reconstructs with a
    /// `SplitDelimiterBehavior`): this one keeps only what the pattern matches,
    /// and needs a backtracking engine — the pattern's `\s+(?!\S)` branch is a
    /// negative lookahead the `regex` crate rejects outright. The pattern
    /// string itself rides the existing `attrs::pattern`.
    pub const TIKTOKEN_PRE_TOKENIZER: Id = id_hex!("AF6D6B71213E710F92E434D451540805");

    // ── minted 2026-08-04 — SentencePiece UNIGRAM support ──
    // PersonaPlex's tokenizer is `model_type = UNIGRAM`: a Viterbi lattice over
    // per-piece log-probs with NO merges. The BPE/WordPiece schema above cannot
    // express it — a unigram model is *defined* by its scores, and only NORMAL
    // pieces are lattice edges, so piece type is load-bearing too.
    /// SentencePiece unigram model (scores, no merges).
    pub const UNIGRAM: Id = id_hex!("AAC02E8BB835F423DF9EA5C0BAA3CAD6");
    /// SentencePiece piece types. Only NORMAL pieces enter the match table;
    /// CONTROL/UNKNOWN never match from text; BYTE backs `byte_fallback`.
    pub const PIECE_NORMAL: Id = id_hex!("EEEA86BCCE689C062D7C26EF1EEF936A");
    pub const PIECE_UNKNOWN: Id = id_hex!("8F7B8C418D447BA7E45CEA8BB4914AC2");
    pub const PIECE_CONTROL: Id = id_hex!("3EE7240B7CB58403D1DF30E39277CFF2");
    pub const PIECE_USER_DEFINED: Id = id_hex!("9796A2F548F7C646D1798EFE7E8B4001");
    pub const PIECE_BYTE: Id = id_hex!("9D9E4D16DD83083B49D34056344FD082");
    pub const PIECE_UNUSED: Id = id_hex!("DD28C23E8DE557BC00F0451A8ED7C1C6");

    // ── minted 2026-11-07 (`trible genid`) — the post-processor ──
    /// Post-processor type: `TemplateProcessing` (single/pair templates over
    /// named special tokens). WeMM's appends its terminal `<embedding>`.
    pub const TEMPLATE_PROCESSING: Id = id_hex!("854D0F97FEDB6F34009FF057E2EA484D");
    /// Post-processor type: `BertProcessing` (`cls` / `sep`).
    pub const BERT_PROCESSING: Id = id_hex!("B279B90BA3274C4EE7E9D4A09FB01C69");
    /// Post-processor type: `RobertaProcessing` (`cls` / `sep` plus the
    /// `TRIM_OFFSETS` / `ADD_PREFIX_SPACE` flags).
    pub const ROBERTA_PROCESSING: Id = id_hex!("4E302695D7A6B275C7D661FE3D62CC38");
    /// A template piece standing for the first input sequence (`$A`).
    pub const TEMPLATE_SEQUENCE_A: Id = id_hex!("6DD021538BCBF99B3391BE505706B5DF");
    /// A template piece standing for the pair sequence (`$B`).
    pub const TEMPLATE_SEQUENCE_B: Id = id_hex!("AC31755BE75895B0CBDB1F32CB5518C1");
}
#[allow(dead_code)]
pub(crate) mod flag {
    use triblespace::macros::id_hex;
    use triblespace::prelude::Id;
    pub const CLEAN_TEXT: Id = id_hex!("C3A27148CDE4AC009B47E1EE141D477B");
    pub const HANDLE_CHINESE_CHARS: Id = id_hex!("56D740D98D843D6E9B3E7CEB09B1949D");
    pub const LOWERCASE: Id = id_hex!("46D626BCA9C575916A51DAAF0D4A3E16");
    pub const INVERT: Id = id_hex!("1C84B18C4CD06F912C6FAF93FD64C48A");
    pub const ADD_PREFIX_SPACE: Id = id_hex!("19B8979BAED4ADF439A7C8867ABC8727");
    pub const TRIM_OFFSETS: Id = id_hex!("E7CC5D08B8995F31A4736B1482124071");
    pub const CLEANUP: Id = id_hex!("42A792CDD8594CE881541C784C65B03F");
    pub const FUSE_UNK: Id = id_hex!("39F4E0AF4C309CB77AADD1EB33495967");
    pub const SPECIAL: Id = id_hex!("BBB4BCA9CA25CAB5F2ECABF787B0A638");
    pub const NORMALIZED: Id = id_hex!("266A99918C6951D5C9F2D0A961C05D4A");
    /// A `ByteLevel` node's `use_regex`. Minted 2026-08-13.
    ///
    /// It was hardcoded `true` on the read side with a comment saying to
    /// revisit it before a tokenizer that sets it false — and Inkling is that
    /// tokenizer: its pre-tokenizer is `Split(Regex, Isolated)` followed by
    /// `ByteLevel { use_regex: false }`. Reconstructed with `true` the
    /// ByteLevel re-applies GPT-2's own split ON TOP of the model's, which
    /// changes the token stream rather than erroring.
    ///
    /// Presence semantics, like every other flag here: a node written before
    /// this id existed carries no tag and rebuilds with `use_regex: false`.
    /// That is a behaviour change for tokenizer graphs already in piles
    /// (CLIP's pre-tokenizer ByteLevel, every decoder ByteLevel) and they must
    /// be re-ingested. A negative `NO_USE_REGEX` flag would have avoided it and
    /// was rejected: an inverted flag makes the absent case mean two different
    /// things depending on when the graph was written, which is the same
    /// ambiguity one layer deeper.
    pub const USE_REGEX: Id = id_hex!("32F9C0904E02BC40F2FF723966A88D8E");
    /// A BPE model's `ignore_merges`: a piece that is already a vocab entry is
    /// emitted whole rather than rebuilt from merges. Minted 2026-08-13.
    ///
    /// Inkling sets it true, and it is not cosmetic — with it false the merge
    /// table re-derives pieces the vocab already names, and long tokens come
    /// back as several short ones.
    pub const IGNORE_MERGES: Id = id_hex!("22B7385E0FFC80CFAB80AE750557F784");

    // ── minted 2026-11-07 (`trible genid`) — added-token options ──
    /// An added token that is NOT special (`"special": false`). Inverted on
    /// purpose; see the module docs: absent has always meant special.
    pub const NOT_SPECIAL: Id = id_hex!("61F8739136328B8CEB56432FABB9EDC6");
    /// An added token's `lstrip`.
    pub const LSTRIP: Id = id_hex!("FEA8C6D2BA219D902B36BE3B3BEA4779");
    /// An added token's `rstrip`.
    pub const RSTRIP: Id = id_hex!("CA5EB59A3C74737199FAE9F9F02134B5");
    /// An added token's `single_word`.
    pub const SINGLE_WORD: Id = id_hex!("A7106FCBE378A206F008841680A82251");
}

pub mod attrs {
    use triblespace::prelude::inlineencodings::{Boolean, F64, GenId, Handle, U256BE};
    use triblespace::prelude::*;

    attributes! {
        // ── reused from `crate::format` (same hex — do NOT re-mint) ──
        /// Ordered position among siblings — merge rank, added-token order,
        /// `Sequence` config-node child order. Reused from `format::attrs::index`.
        "33CE12B1B940B13E48D8E5B0ADFD2421" as index: U256BE;
        /// Homogeneous ordered membership — reused for a `Sequence` config
        /// node → its child steps. From `format::attrs::member`.
        "B4B6EC08A0CD70DE63A690168EE78F0F" as member: GenId;
        /// The tokenizer's source id (e.g. "nomic-ai/nomic-embed-text-v1.5").
        /// Reused from `format::attrs::model_name`.
        "4C1CD1611863E7854C59C7DC706DF77A" as model_name: Handle<blobencodings::UTF8String>;

        // ── minted 2026-07-16 (compass 67b09f72) — structural edges ──
        /// A model root → its tokenizer entity.
        "E7014108A8F9512B19E3E8272E8A71F9" as tokenizer: GenId;
        /// Tokenizer → a vocab entry (repeated).
        "E839AA8F549C0D608FB86476A1EF3416" as vocab: GenId;
        /// Tokenizer → a merge entity (repeated, BPE only).
        "E229769197BB035A2D6F61BC6A7D44BC" as merge: GenId;
        /// Tokenizer → an added/special token entity (repeated).
        "B2553118F4CAAF1D028619956DE7F145" as added: GenId;
        /// Tokenizer → its normalizer config node.
        "53BAF87A0E7F1410F8212B3EDF2A498C" as normalizer: GenId;
        /// Tokenizer → its pre-tokenizer config node.
        "6EEBF39CADD11B7CFBB624019AE21585" as pre_tokenizer: GenId;
        /// Tokenizer → its post-processor config node.
        "98EC58B28F4D0BB43965DF7C5FF22713" as post_processor: GenId;
        /// Tokenizer → its decoder config node.
        "F3AAA4CD8EE04E5592059564A21FE953" as decoder: GenId;

        // ── minted 2026-07-16 — bulk leaves ──
        /// A token/subword string (vocab entry, added-token content). A
        /// `UTF8String` blob: CLIP has 27 tokens over 32 bytes (byte-level
        /// BPE emoji clusters, up to 64 B), so no inline value could hold them.
        /// Content-addressed, so identical pieces (incl. merge halves) dedup.
        "AE7FE29F2F38153F58C542D5CA4A9356" as piece: Handle<blobencodings::UTF8String>;
        /// A token's vocab id (vocab entry, added token). Small int in a U256BE.
        "F0E2E782F7BB62F52B1186DDE0EB5388" as token_id: U256BE;

        // ── minted 2026-08-05 — the tiktoken lane ──
        /// A token's RAW BYTES. The byte-level sibling of `piece`, and the
        /// reason it had to be minted rather than reused: `piece` is a
        /// `UTF8String`, and a tiktoken vocab is not UTF-8 — 1,172 of Kimi-K3's
        /// 163,584 base tokens are mid-codepoint merge fragments that no
        /// `String` can hold. (A HuggingFace `tokenizer.json` sidesteps this by
        /// storing GPT-2 byte-level-unicode *strings*; `tiktoken.model` stores
        /// base64 of the actual bytes, and re-encoding them into a byte-map the
        /// source file never mentions would be inventing a representation.)
        /// Content-addressed like `piece`, so shared tokens still dedup.
        "714AE13F801202EB27C83E3AB2290669" as piece_bytes: Handle<blobencodings::RawBytes>;
        /// A BPE merge's left piece (`UTF8String`, dedups against `piece`).
        "5723ECE1FF426C58879B79D5669A7CF1" as merge_left: Handle<blobencodings::UTF8String>;
        /// A BPE merge's right piece (`UTF8String`, dedups against `piece`).
        "5C78FEB151F35A2C5D07BEC92E860752" as merge_right: Handle<blobencodings::UTF8String>;

        // ── minted 2026-07-16 — config-tail model knobs (flat scalars) ──
        /// The unknown-token string (WordPiece "[UNK]", BPE "<|endoftext|>").
        "68F1A9E6ED735E7C3ADCCA076AFF1742" as unk_token: Handle<blobencodings::UTF8String>;
        /// WordPiece continuing-subword prefix ("##"); empty for BPE.
        "11F76A2C0856C16CB030C4327D5A3B93" as continuing_subword_prefix: Handle<blobencodings::UTF8String>;
        /// BPE end-of-word suffix ("</w>").
        "6FB969E8A3EDD1A657C721DD5A7D42EA" as end_of_word_suffix: Handle<blobencodings::UTF8String>;
        /// WordPiece max input chars per word (100).
        "DF3F88DBFA2B44A7783169C9640014AF" as max_input_chars: U256BE;

        // ── minted 2026-08-04 — SentencePiece UNIGRAM leaves ──
        /// A vocab entry's unigram log-probability. `F64` because scores are
        /// stored f32 in the SPM proto and f64 holds them losslessly; the
        /// Viterbi lattice sums these, so precision is not decorative.
        /// Absent on BPE/WordPiece entries.
        "3BCB70478942DB710ED2A4FB023F3457" as piece_score: F64;
        /// Whether the model falls back to the 256 `<0xXX>` BYTE pieces for
        /// characters no NORMAL piece covers. `Boolean`, on the tokenizer node.
        "EE4C6647619A836326196F0DBF84FA98" as byte_fallback: Boolean;

        // ── minted 2026-07-16 — config-node flat fields ──
        /// A Replace/Split node's regex pattern string (stored raw; both our
        /// tokenizers use the Regex variant, reconstructed as Regex).
        "C8262D5668B8A1F541B3C35D54201BEC" as pattern: Handle<blobencodings::UTF8String>;
        /// A Replace node's replacement content (e.g. " ").
        "3AC7574C07D02D389B4E7AD3B3B084D9" as replace_content: Handle<blobencodings::UTF8String>;
        /// A Split node's SplitDelimiterBehavior name ("Removed"/"Isolated"/…).
        "964B4FCF7477E7E4436F0325F89B7CB5" as behavior: Handle<blobencodings::UTF8String>;

        // ── minted 2026-11-07 (`trible genid`) — post-processor fields ──
        /// A `TemplateProcessing` node → a piece of its `single` template
        /// (repeated; each piece carries its own `index`).
        "14C8B939E24FCFA70C12306A6BEBB1BE" as template_single: GenId;
        /// A `TemplateProcessing` node → a piece of its `pair` template.
        "D4B4725EE0AFDFDCEA32D44A76E25FB6" as template_pair: GenId;
        /// A `TemplateProcessing` node → one of its named special tokens.
        "2405C3D496A57180031ED8C9A10D2B56" as special_token: GenId;
        /// The name a template uses for a special token (`"<embedding>"`,
        /// `"[CLS]"`), on a special-token entity and on a template piece that
        /// refers to it. A name, not token content: one name may stand for
        /// several tokens, which are the special token's `member*`.
        "B4883B40536FB49585FA2FC05063FFF9" as special_id: Handle<blobencodings::UTF8String>;
        /// A template piece's `type_id`.
        "4A59C61D63E2F178979BC4C568C536DC" as type_id: U256BE;
        /// A `BertProcessing`/`RobertaProcessing` node → its `sep` token
        /// (an entity with `piece` and `token_id`).
        "699FCBD65442CF7EDDFD7490361751ED" as sep: GenId;
        /// A `BertProcessing`/`RobertaProcessing` node → its `cls` token.
        "D21DEA042491B27AF5B6929A00BAF193" as cls: GenId;
    }
}

/// Split a HuggingFace `merges` entry into its two pieces. Newer tokenizer.json
/// stores each merge as a 2-element array `["a","b"]`; older ones as a single
/// space-joined string `"a b"`. Handles both.
fn merge_pair(m: &serde_json::Value) -> Option<(String, String)> {
    if let Some(pair) = m.as_array() {
        if pair.len() == 2 {
            return Some((pair[0].as_str()?.to_string(), pair[1].as_str()?.to_string()));
        }
    }
    let s = m.as_str()?;
    let (l, r) = s.split_once(' ')?;
    Some((l.to_string(), r.to_string()))
}

/// Ingest a HuggingFace `tokenizer.json` into a tokenizer graph. Parses the JSON
/// **once** and emits the bulk (vocab / merges / added-tokens) as tribles under
/// a tokenizer entity tagged with its model kind. Returns the Fragment rooted at
/// the tokenizer entity (link it to a model root with `attrs::tokenizer`).
///
/// Alongside the bulk it writes the flat model knobs (`unk_token`, …) and the
/// normalizer / pre-tokenizer / decoder config subtrees ([`save_config_node`])
/// and the post-processor ([`save_post_processor`]).
pub fn save_tokenizer_json(
    json: &[u8],
    source_name: &str,
    blobs: &mut impl BlobStorePut,
) -> Result<Fragment, Err> {
    let v: serde_json::Value = serde_json::from_slice(json)?;
    let model = &v["model"];
    let model_kind = model["type"].as_str().unwrap_or("");
    let mut facts = Fragment::empty();

    // Every distinct string this tokenizer contains, put ONCE.
    //
    // The whole tokenizer is 199 998 distinct strings — every one of Inkling's
    // 892 378 merge sides is already a vocab entry, and merges introduce no new
    // ones — but the shape below asks the store for each of them separately, so
    // 2 MB of text used to cost 1 092 438 blob operations. Nothing about the
    // RESULT changes: a handle is derived from content, so the same string
    // yields the same handle whether it is computed once or five times, and the
    // facts, the blobs and the pile bytes are identical either way. What
    // changes is how many times the store is asked, and on a pile a put is an
    // append rather than a hash-map probe.
    //
    // This is NOT a change of representation. Each piece stays its own
    // content-addressed blob, so the interning that matters — the baseline
    // shared across tokenizers, where CLIP, nomic and Inkling overlap heavily —
    // is exactly as it was. Packing the vocabulary into one blob would have
    // destroyed that; a memo in front of the same puts cannot.
    let mut memo: HashMap<String, Inline<inlineencodings::Handle<blobencodings::UTF8String>>> =
        HashMap::with_capacity(1 << 18);
    macro_rules! intern {
        ($blobs:expr_2021, $s:expr_2021) => {{
            let s: &str = $s;
            match memo.get(s) {
                Some(h) => *h,
                None => {
                    let h = $blobs.put::<blobencodings::UTF8String, _>(s.to_string())?;
                    memo.insert(s.to_string(), h);
                    h
                }
            }
        }};
    }

    // ── vocab: { piece, token_id } per entry ──
    let mut vocab_ids: Vec<Id> = Vec::new();
    if let Some(vocab) = model["vocab"].as_object() {
        for (tok, id) in vocab {
            let id = id.as_u64().ok_or("vocab id not an integer")?;
            let ph = intern!(blobs, tok);
            let e = entity! { _ @ attrs::piece: ph, attrs::token_id: id };
            vocab_ids.push(e.root().expect("vocab entry root"));
            facts += e;
        }
    }

    // ── merges: { merge_left, merge_right, index=rank } (BPE) ──
    let mut merge_ids: Vec<Id> = Vec::new();
    if let Some(merges) = model["merges"].as_array() {
        for (rank, m) in merges.iter().enumerate() {
            let (l, r) = merge_pair(m).ok_or("malformed merge entry")?;
            let lh = intern!(blobs, &l);
            let rh = intern!(blobs, &r);
            let e = entity! { _ @
                attrs::merge_left: lh,
                attrs::merge_right: rh,
                attrs::index: rank as u64,
            };
            merge_ids.push(e.root().expect("merge root"));
            facts += e;
        }
    }

    // ── added / special tokens: { piece, token_id, index=order, options } ──
    // The options are part of the token (a non-special `<think>` is not the
    // same added token as a special one), so they are in its identity.
    let mut added_ids: Vec<Id> = Vec::new();
    if let Some(added) = v["added_tokens"].as_array() {
        for (order, t) in added.iter().enumerate() {
            let content = t["content"].as_str().ok_or("added token missing content")?;
            let id = t["id"].as_u64().ok_or("added token id not an integer")?;
            let ch = intern!(blobs, content);
            let set = |field: &str| t[field].as_bool() == Some(true);
            let options: Vec<Id> = [
                // `special` absent from the JSON is HuggingFace's false.
                (!set("special"), flag::NOT_SPECIAL),
                (set("normalized"), flag::NORMALIZED),
                (set("lstrip"), flag::LSTRIP),
                (set("rstrip"), flag::RSTRIP),
                (set("single_word"), flag::SINGLE_WORD),
            ]
            .into_iter()
            .filter_map(|(on, tag)| on.then_some(tag))
            .collect();
            let e = entity! { _ @
                attrs::piece: ch,
                attrs::token_id: id,
                attrs::index: order as u64,
                metadata::tag*: options.iter(),
            };
            added_ids.push(e.root().expect("added token root"));
            facts += e;
        }
    }

    // ── the tokenizer entity (+ flat model knobs; absent ones are omitted) ──
    let name_h = intern!(blobs, source_name);
    let unk = model["unk_token"].as_str();
    let csp = model["continuing_subword_prefix"]
        .as_str()
        .filter(|s| !s.is_empty());
    let eows = model["end_of_word_suffix"]
        .as_str()
        .filter(|s| !s.is_empty());
    let max_chars = model["max_input_chars_per_word"].as_u64();
    let mut model_tags = vec![match model_kind {
        "WordPiece" => ty::WORD_PIECE,
        "BPE" => ty::BPE,
        other => return Err(format!("unsupported tokenizer model type: {other:?}").into()),
    }];
    if model["ignore_merges"].as_bool() == Some(true) {
        model_tags.push(flag::IGNORE_MERGES);
    }

    // ── config tail: normalizer / pre-tokenizer / post-processor / decoder
    //    subtrees (see the module docs for why the post-processor is here). ──
    let post_id = match v.get("post_processor").filter(|x| !x.is_null()) {
        Some(n) => Some(save_post_processor(n, blobs, &mut facts)?),
        None => None,
    };
    let norm_id = match v.get("normalizer").filter(|x| !x.is_null()) {
        Some(n) => Some(save_config_node(n, blobs, &mut facts)?),
        None => None,
    };
    let pretok_id = match v.get("pre_tokenizer").filter(|x| !x.is_null()) {
        Some(n) => Some(save_config_node(n, blobs, &mut facts)?),
        None => None,
    };
    let dec_id = match v.get("decoder").filter(|x| !x.is_null()) {
        Some(n) => Some(save_config_node(n, blobs, &mut facts)?),
        None => None,
    };

    let unk = put_optional_text(blobs, unk)?;
    let csp = put_optional_text(blobs, csp)?;
    let eows = put_optional_text(blobs, eows)?;
    let tok = entity! { _ @
        metadata::tag*: model_tags.iter(),
        attrs::normalizer?: norm_id,
        attrs::pre_tokenizer?: pretok_id,
        attrs::post_processor?: post_id,
        attrs::decoder?: dec_id,
        attrs::unk_token?: unk,
        attrs::continuing_subword_prefix?: csp,
        attrs::end_of_word_suffix?: eows,
        attrs::max_input_chars?: max_chars,
        attrs::vocab*: vocab_ids.iter(),
        attrs::merge*: merge_ids.iter(),
        attrs::added*: added_ids.iter(),
    };
    let tok_id = tok.root().expect("tokenizer root");
    facts += tok;
    facts += entity! { ExclusiveId::force_ref(&tok_id) @ attrs::model_name: name_h };
    Ok(crate::format::rooted_at(tok_id, facts))
}

/// How many reserved special-token slots a tiktoken tokenizer appends above its
/// base vocab — `TikTokenTokenizer.num_reserved_special_tokens` in the shipped
/// `tokenization_kimi.py`. Kimi-K3: 163,584 base + 256 = 163,840 total.
pub const NUM_RESERVED_SPECIAL_TOKENS: u64 = 256;

/// Ingest a **tiktoken** tokenizer — `tiktoken.model` plus the
/// `tokenizer_config.json` that names its reserved slots — into a tokenizer
/// graph. The tiktoken counterpart of [`save_tokenizer_json`], and structurally
/// simpler: no merges (the rank table is the merge order) and no normalizer or
/// decoder (byte-level BPE is its own inverse), so the graph is the rank table,
/// the 256 special tokens, and one pre-tokenizer node holding `pat_str`.
///
/// The reserved-slot naming mirrors the shipped `__init__` exactly: ids
/// `n_base .. n_base + 256` take their content from `added_tokens_decoder` when
/// the config names them, and `<|reserved_token_{id}|>` (the ABSOLUTE id, not
/// an offset) otherwise. Slots the config marks `"special": true` carry
/// `flag::SPECIAL`; the handful marked `false` (`<|open|>`, `<|close|>`,
/// `<|sep|>`) do not — but all 256 are special *to the encoder*, which is what
/// `added` membership means here.
#[cfg(feature = "k3tok")]
pub fn save_tiktoken(
    model_file: &[u8],
    config_json: &[u8],
    pat_str: &str,
    source_name: &str,
    blobs: &mut impl BlobStorePut,
) -> Result<Fragment, Err> {
    let ranks = crate::tiktoken::parse_tiktoken_model(model_file)?;
    let cfg: serde_json::Value = serde_json::from_slice(config_json)?;
    let mut facts = Fragment::empty();

    // ── the rank table: { piece_bytes, token_id = rank } per entry ──
    let mut vocab_ids: Vec<Id> = Vec::with_capacity(ranks.len());
    for (tok, rank) in &ranks {
        let bh = blobs.put::<blobencodings::RawBytes, _>(tok.clone())?;
        let e = entity! { _ @ attrs::piece_bytes: bh, attrs::token_id: *rank as u64 };
        vocab_ids.push(e.root().expect("vocab entry root"));
        facts += e;
    }

    // ── the 256 reserved special slots: { piece, token_id, index } ──
    let n_base = ranks.len() as u64;
    let named = cfg["added_tokens_decoder"].as_object();
    let mut added_ids: Vec<Id> = Vec::new();
    for (order, id) in (n_base..n_base + NUM_RESERVED_SPECIAL_TOKENS).enumerate() {
        let entry = named.and_then(|m| m.get(&id.to_string()));
        let content = match entry.and_then(|e| e["content"].as_str()) {
            Some(c) => c.to_string(),
            None => format!("<|reserved_token_{id}|>"),
        };
        let mut tags: Vec<Id> = Vec::new();
        if entry
            .map(|e| e["special"].as_bool() == Some(true))
            .unwrap_or(false)
        {
            tags.push(flag::SPECIAL);
        }
        let ch = blobs.put::<blobencodings::UTF8String, _>(content)?;
        let e = entity! { _ @
            attrs::piece: ch,
            attrs::token_id: id,
            attrs::index: order as u64,
            metadata::tag*: tags.iter(),
        };
        added_ids.push(e.root().expect("added token root"));
        facts += e;
    }

    // ── the pre-tokenizer node: the pattern, and the fact that it is applied
    //    by find_iter rather than as a Split (see ty::TIKTOKEN_PRE_TOKENIZER) ──
    let pat_h = blobs.put::<blobencodings::UTF8String, _>(pat_str.to_string())?;
    let pretok = entity! { _ @
        metadata::tag: ty::TIKTOKEN_PRE_TOKENIZER,
        attrs::pattern: pat_h,
    };
    let pretok_id = pretok.root().expect("pre-tokenizer root");
    facts += pretok;

    let name_h = blobs.put::<blobencodings::UTF8String, _>(source_name.to_string())?;
    let unk = put_optional_text(blobs, cfg["unk_token"].as_str())?;
    let tok = entity! { _ @
        metadata::tag: ty::TIKTOKEN,
        attrs::pre_tokenizer: pretok_id,
        attrs::unk_token?: unk,
        attrs::vocab*: vocab_ids.iter(),
        attrs::added*: added_ids.iter(),
    };
    let tok_id = tok.root().expect("tokenizer root");
    facts += tok;
    facts += entity! { ExclusiveId::force_ref(&tok_id) @ attrs::model_name: name_h };
    Ok(crate::format::rooted_at(tok_id, facts))
}

/// Ingest one config node (a normalizer / pre-tokenizer / decoder subtree) into
/// `facts`, returning its entity id. The node's TYPE is a `metadata::tag`
/// discriminant (`ty::*`); boolean options are `metadata::tag` flags (`flag::*`,
/// presence = true); regex/behaviour/prefix are flat attrs; a `Sequence`'s
/// ordered children are `member*` edges carrying an `index`. Recursive.
fn save_config_node(
    v: &serde_json::Value,
    blobs: &mut impl BlobStorePut,
    facts: &mut Fragment,
) -> Result<Id, Err> {
    let node_type = v["type"].as_str().ok_or("config node missing type")?;
    let mut tags: Vec<Id> = Vec::new();
    let mut pattern_h = None;
    let mut replace_content: Option<&str> = None;
    let mut behavior: Option<&str> = None;
    let mut prefix: Option<&str> = None;
    let mut members: Vec<Id> = Vec::new();

    // pattern is a tagged enum object {Regex|String: "…"}; both our tokenizers
    // are Regex, so we store the string and reconstruct as Regex (see spec).
    let pattern_str = || {
        v["pattern"]["Regex"]
            .as_str()
            .or_else(|| v["pattern"]["String"].as_str())
    };

    match node_type {
        "BertNormalizer" => {
            tags.push(ty::BERT_NORMALIZER);
            if v["clean_text"].as_bool() == Some(true) {
                tags.push(flag::CLEAN_TEXT);
            }
            if v["handle_chinese_chars"].as_bool() == Some(true) {
                tags.push(flag::HANDLE_CHINESE_CHARS);
            }
            if v["lowercase"].as_bool() == Some(true) {
                tags.push(flag::LOWERCASE);
            }
            // strip_accents: null in nomic → emit nothing (reconstruct = None).
        }
        "Sequence" => {
            tags.push(ty::SEQUENCE);
            let children = v["normalizers"]
                .as_array()
                .or_else(|| v["pretokenizers"].as_array())
                .ok_or("Sequence node missing normalizers/pretokenizers array")?;
            for (i, child) in children.iter().enumerate() {
                let cid = save_config_node(child, blobs, facts)?;
                *facts +=
                    entity! { ExclusiveId::force_ref(&cid) @ attrs::index: i as u64 };
                members.push(cid);
            }
        }
        "NFC" => tags.push(ty::NFC),
        "Lowercase" => tags.push(ty::LOWERCASE),
        "BertPreTokenizer" => tags.push(ty::BERT_PRE_TOKENIZER),
        "Replace" => {
            tags.push(ty::REPLACE);
            let pat = pattern_str().ok_or("Replace node missing pattern")?;
            pattern_h = Some(blobs.put::<blobencodings::UTF8String, _>(pat.to_string())?);
            replace_content = v["content"].as_str();
        }
        "Split" => {
            tags.push(ty::SPLIT);
            let pat = pattern_str().ok_or("Split node missing pattern")?;
            pattern_h = Some(blobs.put::<blobencodings::UTF8String, _>(pat.to_string())?);
            behavior = v["behavior"].as_str();
            if v["invert"].as_bool() == Some(true) {
                tags.push(flag::INVERT);
            }
        }
        "ByteLevel" => {
            tags.push(ty::BYTE_LEVEL);
            if v["add_prefix_space"].as_bool() == Some(true) {
                tags.push(flag::ADD_PREFIX_SPACE);
            }
            if v["trim_offsets"].as_bool() == Some(true) {
                tags.push(flag::TRIM_OFFSETS);
            }
            // Absent means true — that is HuggingFace's default, and reading
            // `as_bool() == Some(true)` here would silently turn every
            // tokenizer that omits the field into a `use_regex: false` one.
            if v["use_regex"].as_bool().unwrap_or(true) {
                tags.push(flag::USE_REGEX);
            }
        }
        "WordPiece" => {
            // the decoder role (the model WordPiece is the tok-root, not here);
            // reuse the continuing_subword_prefix attr for the decoder prefix.
            tags.push(ty::WORD_PIECE);
            prefix = v["prefix"].as_str();
            if v["cleanup"].as_bool() == Some(true) {
                tags.push(flag::CLEANUP);
            }
        }
        other => return Err(format!("unknown config node type: {other:?}").into()),
    }

    let replace_content = put_optional_text(blobs, replace_content)?;
    let behavior = put_optional_text(blobs, behavior)?;
    let prefix = put_optional_text(blobs, prefix)?;
    let node = entity! { _ @
        metadata::tag*: tags.iter(),
        attrs::pattern?: pattern_h,
        attrs::replace_content?: replace_content,
        attrs::behavior?: behavior,
        attrs::continuing_subword_prefix?: prefix,
        attrs::member*: members.iter(),
    };
    let id = node.root().expect("config node root");
    *facts += node;
    Ok(id)
}

/// Ingest a post-processor subtree into `facts`, returning its node id.
///
/// `TemplateProcessing` stores its two templates as pieces, each carrying its
/// position (`index`) and `type_id` in its own identity, so `$A:0` at index 0
/// and `$A:0` at index 2 are two pieces rather than one entity with two
/// indices; a piece names the first or pair sequence by a type tag, or a
/// special token by its `special_id`. The named special tokens are entities
/// of their own, their tokens ordered `member`s. `Bert`/`RobertaProcessing`
/// carry `cls`/`sep` token entities; `ByteLevel` is the same node a
/// pre-tokenizer uses ([`save_config_node`]); `Sequence` orders its children
/// like every other sequence here.
fn save_post_processor(
    v: &serde_json::Value,
    blobs: &mut impl BlobStorePut,
    facts: &mut Fragment,
) -> Result<Id, Err> {
    let node_type = v["type"].as_str().ok_or("post-processor missing type")?;
    let node = match node_type {
        "ByteLevel" => return save_config_node(v, blobs, facts),
        "Sequence" => {
            let children = v["processors"]
                .as_array()
                .ok_or("post-processor Sequence missing processors array")?;
            let mut members = Vec::with_capacity(children.len());
            for (i, child) in children.iter().enumerate() {
                let cid = save_post_processor(child, blobs, facts)?;
                *facts +=
                    entity! { ExclusiveId::force_ref(&cid) @ attrs::index: i as u64 };
                members.push(cid);
            }
            entity! { _ @ metadata::tag: ty::SEQUENCE, attrs::member*: members.iter() }
        }
        "BertProcessing" | "RobertaProcessing" => {
            let sep = save_cls_sep(&v["sep"], blobs, facts)?;
            let cls = save_cls_sep(&v["cls"], blobs, facts)?;
            let mut tags = Vec::new();
            if node_type == "BertProcessing" {
                tags.push(ty::BERT_PROCESSING);
            } else {
                tags.push(ty::ROBERTA_PROCESSING);
                // Both default to true in `tokenizers`.
                if v["trim_offsets"].as_bool().unwrap_or(true) {
                    tags.push(flag::TRIM_OFFSETS);
                }
                if v["add_prefix_space"].as_bool().unwrap_or(true) {
                    tags.push(flag::ADD_PREFIX_SPACE);
                }
            }
            entity! { _ @ metadata::tag*: tags.iter(), attrs::sep: sep, attrs::cls: cls }
        }
        "TemplateProcessing" => {
            let single = save_template(&v["single"], blobs, facts)?;
            let pair = save_template(&v["pair"], blobs, facts)?;
            let mut specials = Vec::new();
            if let Some(map) = v["special_tokens"].as_object() {
                for special in map.values() {
                    let name = special["id"].as_str().ok_or("special token id")?;
                    let ids = special["ids"].as_array().ok_or("special token ids")?;
                    let tokens = special["tokens"].as_array().ok_or("special token tokens")?;
                    if ids.len() != tokens.len() {
                        return Err(format!(
                            "special token {name:?}: ids and tokens differ in length"
                        )
                        .into());
                    }
                    let mut members = Vec::with_capacity(ids.len());
                    for (index, (id, content)) in ids.iter().zip(tokens).enumerate() {
                        let id = id.as_u64().ok_or("special token id not an integer")?;
                        let content = content.as_str().ok_or("special token content")?;
                        let piece =
                            blobs.put::<blobencodings::UTF8String, _>(content.to_owned())?;
                        let e = entity! { _ @
                            attrs::piece: piece,
                            attrs::token_id: id,
                            attrs::index: index as u64,
                        };
                        members.push(e.root().expect("special token member root"));
                        *facts += e;
                    }
                    let name = blobs.put::<blobencodings::UTF8String, _>(name.to_owned())?;
                    let e = entity! { _ @ attrs::special_id: name, attrs::member*: members.iter() };
                    specials.push(e.root().expect("special token root"));
                    *facts += e;
                }
            }
            entity! { _ @
                metadata::tag: ty::TEMPLATE_PROCESSING,
                attrs::template_single*: single.iter(),
                attrs::template_pair*: pair.iter(),
                attrs::special_token*: specials.iter(),
            }
        }
        other => return Err(format!("unsupported post-processor type: {other:?}").into()),
    };
    let id = node.root().expect("post-processor root");
    *facts += node;
    Ok(id)
}

/// A `BertProcessing`/`RobertaProcessing` `cls`/`sep`: `["[SEP]", 102]`.
fn save_cls_sep(
    pair: &serde_json::Value,
    blobs: &mut impl BlobStorePut,
    facts: &mut Fragment,
) -> Result<Id, Err> {
    let content = pair[0].as_str().ok_or("cls/sep token missing content")?;
    let id = pair[1].as_u64().ok_or("cls/sep token id not an integer")?;
    let piece = blobs.put::<blobencodings::UTF8String, _>(content.to_owned())?;
    let e = entity! { _ @ attrs::piece: piece, attrs::token_id: id };
    let root = e.root().expect("cls/sep token root");
    *facts += e;
    Ok(root)
}

/// One `TemplateProcessing` template's pieces, each carrying its position.
fn save_template(
    template: &serde_json::Value,
    blobs: &mut impl BlobStorePut,
    facts: &mut Fragment,
) -> Result<Vec<Id>, Err> {
    let template = template
        .as_array()
        .ok_or("TemplateProcessing template is not an array")?;
    let mut ids = Vec::with_capacity(template.len());
    for (index, piece) in template.iter().enumerate() {
        let e = if let Some(seq) = piece.get("Sequence") {
            let tag = match seq["id"].as_str() {
                Some("A") => ty::TEMPLATE_SEQUENCE_A,
                Some("B") => ty::TEMPLATE_SEQUENCE_B,
                other => return Err(format!("template sequence id {other:?}").into()),
            };
            let type_id = seq["type_id"].as_u64().ok_or("piece type_id")?;
            entity! { _ @
                metadata::tag: tag,
                attrs::type_id: type_id,
                attrs::index: index as u64,
            }
        } else if let Some(special) = piece.get("SpecialToken") {
            let name = special["id"].as_str().ok_or("template special id")?;
            let type_id = special["type_id"].as_u64().ok_or("piece type_id")?;
            let name = blobs.put::<blobencodings::UTF8String, _>(name.to_owned())?;
            entity! { _ @
                attrs::special_id: name,
                attrs::type_id: type_id,
                attrs::index: index as u64,
            }
        } else {
            return Err(format!("unknown template piece {piece}").into());
        };
        ids.push(e.root().expect("template piece root"));
        *facts += e;
    }
    Ok(ids)
}

/// Store optional text as a `UTF8String` blob and return its handle.
fn put_optional_text(
    blobs: &mut impl BlobStorePut,
    text: Option<&str>,
) -> Result<Option<Inline<inlineencodings::Handle<blobencodings::UTF8String>>>, Err> {
    Ok(match text {
        Some(text) => Some(blobs.put::<blobencodings::UTF8String, _>(text.to_owned())?),
        None => None,
    })
}

/// Read a `UTF8String` blob handle back to an owned `String`.
fn read_piece(
    blobs: &impl BlobStoreGet,
    h: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
) -> String {
    let v: anybytes::View<str> = blobs.get(h).expect("piece blob");
    v.to_string()
}

/// Materialize a tokenizer's vocab back into a `token → id` map by walking its
/// `vocab` members. The dual of the vocab half of [`save_tokenizer_json`].
pub fn load_vocab(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> HashMap<String, u64> {
    // Single joined query — tokenizer → vocab entry → piece + id — the engine
    // does the join (`_?entry` is a pattern-local join var, projected away).
    // Kept on IDIOM grounds (concise, "use the engine"), NOT speed: a
    // warming-controlled 3-round A/B on CLIP's 49k vocab shows this and a
    // find!-per-entry loop are EQUIVALENT (~575ms each, <1% apart, on the
    // Atreides+agglomerative default planner). Load is ~575ms either way; the
    // real cost is the one-time 90s ingest. (Two earlier "findings" — 2.3x
    // slower, then 2x faster — were both measurement artifacts: cross-run
    // ingest noise, then cold-first-query ordering. Only the warm multi-round
    // same-process A/B told the truth.)
    find!(
        (p, i: u64),
        pattern!(tribles, [
            { tok_id @ attrs::vocab: _?entry },
            { _?entry @ attrs::piece: ?p, attrs::token_id: ?i },
        ])
    )
    .map(|(ph, id)| (read_piece(blobs, ph), id))
    .collect()
}

/// Materialize a tokenizer's BPE merges back into a rank-ordered `Vec<(left,
/// right)>`. Empty for WordPiece/WordLevel/Unigram (no merges).
pub fn load_merges(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Vec<(String, String)> {
    // Single joined query — tokenizer → merge → {left, right, rank} (see
    // load_vocab: joined ≈ per-entry on speed; kept for idiom/concision).
    let mut ranked: Vec<(u64, String, String)> = find!(
        (l, r, k: u64),
        pattern!(tribles, [
            { tok_id @ attrs::merge: _?m },
            { _?m @ attrs::merge_left: ?l, attrs::merge_right: ?r, attrs::index: ?k },
        ])
    )
    .map(|(lh, rh, rank)| (rank, read_piece(blobs, lh), read_piece(blobs, rh)))
    .collect();
    ranked.sort_by_key(|(rank, _, _)| *rank);
    ranked.into_iter().map(|(_, l, r)| (l, r)).collect()
}

/// Materialize a tokenizer's added/special tokens back into an order-preserved
/// `Vec<(content, id)>` (their `index` is the original `added_tokens` order).
pub fn load_added(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Vec<(String, u64)> {
    let mut ordered: Vec<(u64, String, u64)> = find!(
        (p, i: u64, k: u64),
        pattern!(tribles, [
            { tok_id @ attrs::added: _?a },
            { _?a @ attrs::piece: ?p, attrs::token_id: ?i, attrs::index: ?k },
        ])
    )
    .map(|(ph, id, order)| (order, read_piece(blobs, ph), id))
    .collect();
    ordered.sort_by_key(|(order, _, _)| *order);
    ordered.into_iter().map(|(_, p, id)| (p, id)).collect()
}

/// Materialize a **tiktoken** tokenizer's rank table back into `(token bytes,
/// rank)` pairs — the dual of the vocab half of [`save_tiktoken`]. Unordered
/// (the rank rides in the pair); feed straight into `Tiktoken::new`.
pub fn load_tiktoken_ranks(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Vec<(Vec<u8>, u64)> {
    find!(
        (p, i: u64),
        pattern!(tribles, [
            { tok_id @ attrs::vocab: _?entry },
            { _?entry @ attrs::piece_bytes: ?p, attrs::token_id: ?i },
        ])
    )
    .map(|(bh, rank)| {
        let b: anybytes::Bytes = blobs.get(bh).expect("piece_bytes blob");
        (b.to_vec(), rank)
    })
    .collect()
}

/// The pre-tokenizer pattern string a tokenizer's pre-tokenizer node carries
/// (`attrs::pattern`), if it has one. For a `TIKTOKEN` tokenizer this is the
/// whole pre-tokenizer: the regex IS the algorithm.
pub fn load_pre_tokenizer_pattern(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Option<String> {
    find!(
        (p,),
        pattern!(tribles, [
            { tok_id @ attrs::pre_tokenizer: _?node },
            { _?node @ attrs::pattern: ?p },
        ])
    )
    .next()
    .map(|(ph,)| read_piece(blobs, ph))
}

/// Tokenizer roots carry a supported model-kind tag and a vocabulary.
/// A decoder may share the same kind tag, but has no vocabulary. Names are
/// optional annotations, so adding aliases or omitting a label cannot change
/// whether the tokenizer is readable. Select a root explicitly when several
/// tokenizers coexist.
pub fn find_tokenizers(tribles: &TribleSet) -> impl Iterator<Item = Id> + '_ {
    find!(
        (e: Id, t: Id),
        pattern!(tribles, [{ ?e @ metadata::tag: ?t, attrs::vocab: _?entry }])
    )
    // The model-type tags, and ONLY those: a tokenizer node also carries flag
    // tags (ADD_PREFIX_SPACE, ...), and `find!` yields one row per tag, so this
    // filter is what distinguishes the tokenizer node from its own flags.
    //
    // All four kinds belong here. TIKTOKEN because Kimi-K3's byte-level BPE has
    // no merges list; UNIGRAM because omitting it made
    // `load_spm_tokenizer_from_pile` report "no tokenizer graph" on a pile that
    // had just been written.
    .filter(|&(_, t)| t == ty::WORD_PIECE || t == ty::BPE || t == ty::TIKTOKEN || t == ty::UNIGRAM)
    .map(|(e, _)| e)
}

/// The first tokenizer root in a fact set, retained for legacy callers.
///
/// New consolidated-graph code should use [`find_tokenizers`] together with an
/// explicit exact-cardinality selector (see [`crate::selection`]).
pub fn find_tokenizer(tribles: &TribleSet) -> Option<Id> {
    find_tokenizers(tribles).next()
}

/// All `metadata::tag` discriminants on a node (type + boolean flags).
pub fn node_tags(tribles: &TribleSet, node: Id) -> Vec<Id> {
    find!((t: Id), pattern!(tribles, [{ node @ metadata::tag: ?t }]))
        .map(|(t,)| t)
        .collect()
}

/// A `Sequence` config node's children, in their persisted `index` order.
pub fn ordered_members(tribles: &TribleSet, node: Id) -> Vec<Id> {
    let mut v: Vec<(u64, Id)> = find!(
        (m: Id, i: u64),
        pattern!(tribles, [{ node @ attrs::member: ?m }, { ?m @ attrs::index: ?i }])
    )
    .map(|(m, i)| (i, m))
    .collect();
    v.sort_by_key(|(i, _)| *i);
    v.into_iter().map(|(_, m)| m).collect()
}

/// Exact-cardinality counterpart to [`ordered_members`] for tokenizer
/// construction. Each sequence child owns one index and indices are unique;
/// otherwise order would depend on query iteration or sort stability.
#[cfg(feature = "tokenizer")]
fn ordered_members_strict(tribles: &TribleSet, node: Id) -> Result<Vec<Id>, Err> {
    use std::collections::{BTreeMap, BTreeSet};

    let members: BTreeSet<Id> = find!(
        (member: Id),
        pattern!(tribles, [{ node @ attrs::member: ?member }])
    )
    .map(|(member,)| member)
    .collect();
    let mut ordered = BTreeMap::new();
    for member in members {
        let index = optional_one(
            find!(
                (index: u64),
                pattern!(tribles, [{ member @ attrs::index: ?index }])
            )
            .map(|(index,)| index),
            member,
            "index",
        )?
        .ok_or_else(|| format!("sequence child {member} has no index"))?;
        if ordered.insert(index, member).is_some() {
            return Err(format!("sequence node {node} has duplicate child index {index}").into());
        }
    }
    Ok(ordered.into_values().collect())
}

// ═══════════════════════════════════════════════════════════════════════════
// Construct-from-graph: query the graph and feed the parts into `tokenizers`'
// programmatic builders. No JSON anywhere in this path — the `tokenizers`
// crate is purely the executor, exactly as Burn is for the weight graph.
// ═══════════════════════════════════════════════════════════════════════════

/// At most one value of a node's field, refusing more.
#[cfg(feature = "tokenizer")]
fn optional_one<T>(
    values: impl IntoIterator<Item = T>,
    node: Id,
    field: &'static str,
) -> Result<Option<T>, Err> {
    let mut values = values.into_iter();
    let value = values.next();
    if values.next().is_some() {
        return Err(format!("node {node} has more than one {field}").into());
    }
    Ok(value)
}

#[cfg(feature = "tokenizer")]
fn required_one<T>(
    values: impl IntoIterator<Item = T>,
    node: Id,
    field: &'static str,
) -> Result<T, Err> {
    optional_one(values, node, field)?.ok_or_else(|| format!("node {node} has no {field}").into())
}

#[cfg(feature = "tokenizer")]
fn load_vocab_strict(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Result<HashMap<String, u64>, Err> {
    use std::collections::{BTreeSet, HashSet};

    let entries: BTreeSet<Id> = find!(
        (entry: Id),
        pattern!(tribles, [{ tok_id @ attrs::vocab: ?entry }])
    )
    .map(|(entry,)| entry)
    .collect();
    let mut vocab = HashMap::with_capacity(entries.len());
    let mut ids = HashSet::with_capacity(entries.len());
    for entry in entries {
        let piece = required_one(
            find!(
                (piece),
                pattern!(tribles, [{ entry @ attrs::piece: ?piece }])
            )
            .map(|(piece,)| read_piece(blobs, piece)),
            entry,
            "piece",
        )?;
        let token_id = required_one(
            find!(
                (token_id: u64),
                pattern!(tribles, [{ entry @ attrs::token_id: ?token_id }])
            )
            .map(|(token_id,)| token_id),
            entry,
            "token_id",
        )?;
        if !ids.insert(token_id) {
            return Err(format!("tokenizer {tok_id} has duplicate token id {token_id}").into());
        }
        if vocab.insert(piece.clone(), token_id).is_some() {
            return Err(format!("tokenizer {tok_id} has duplicate vocab piece {piece:?}").into());
        }
    }
    Ok(vocab)
}

#[cfg(feature = "tokenizer")]
fn load_merges_strict(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Result<Vec<(String, String)>, Err> {
    use std::collections::{BTreeMap, BTreeSet};

    let merges: BTreeSet<Id> = find!(
        (merge: Id),
        pattern!(tribles, [{ tok_id @ attrs::merge: ?merge }])
    )
    .map(|(merge,)| merge)
    .collect();
    let mut ranked = BTreeMap::new();
    for merge in merges {
        let left = required_one(
            find!(
                (left),
                pattern!(tribles, [{ merge @ attrs::merge_left: ?left }])
            )
            .map(|(left,)| read_piece(blobs, left)),
            merge,
            "merge_left",
        )?;
        let right = required_one(
            find!(
                (right),
                pattern!(tribles, [{ merge @ attrs::merge_right: ?right }])
            )
            .map(|(right,)| read_piece(blobs, right)),
            merge,
            "merge_right",
        )?;
        let index = required_one(
            find!((index: u64), pattern!(tribles, [{ merge @ attrs::index: ?index }]))
                .map(|(index,)| index),
            merge,
            "index",
        )?;
        if ranked.insert(index, (left, right)).is_some() {
            return Err(format!("tokenizer {tok_id} has duplicate merge index {index}").into());
        }
    }
    Ok(ranked.into_values().collect())
}

#[cfg(feature = "tokenizer")]
fn load_added_strict(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Result<Vec<(String, u64, Vec<Id>)>, Err> {
    use std::collections::{BTreeMap, BTreeSet};

    let added: BTreeSet<Id> = find!(
        (added: Id),
        pattern!(tribles, [{ tok_id @ attrs::added: ?added }])
    )
    .map(|(added,)| added)
    .collect();
    let mut ordered = BTreeMap::new();
    for added in added {
        let piece = required_one(
            find!(
                (piece),
                pattern!(tribles, [{ added @ attrs::piece: ?piece }])
            )
            .map(|(piece,)| read_piece(blobs, piece)),
            added,
            "piece",
        )?;
        let token_id = required_one(
            find!(
                (token_id: u64),
                pattern!(tribles, [{ added @ attrs::token_id: ?token_id }])
            )
            .map(|(token_id,)| token_id),
            added,
            "token_id",
        )?;
        let index = required_one(
            find!((index: u64), pattern!(tribles, [{ added @ attrs::index: ?index }]))
                .map(|(index,)| index),
            added,
            "index",
        )?;
        let options = node_tags(tribles, added);
        if ordered.insert(index, (piece, token_id, options)).is_some() {
            return Err(
                format!("tokenizer {tok_id} has duplicate added-token index {index}").into(),
            );
        }
    }
    Ok(ordered.into_values().collect())
}

/// Read a node's optional `UTF8String`-handle field back to a `String`.
#[cfg(feature = "tokenizer")]
macro_rules! long_field {
    ($tribles:expr_2021, $blobs:expr_2021, $node:expr_2021, $attr:path) => {
        optional_one(
            find!((h,), pattern!($tribles, [{ ($node) @ $attr: ?h }]))
                .map(|(h,)| read_piece($blobs, h)),
            $node,
            stringify!($attr),
        )?
    };
}

/// Read a node's optional `GenId` edge.
#[cfg(feature = "tokenizer")]
macro_rules! edge_field {
    ($tribles:expr_2021, $node:expr_2021, $attr:path) => {
        optional_one(
            find!((e: Id), pattern!($tribles, [{ ($node) @ $attr: ?e }]))
                .map(|(e,)| e),
            $node,
            stringify!($attr),
        )?
    };
}

/// Build a ready-to-encode [`tokenizers::Tokenizer`] from a tokenizer graph —
/// the dual of [`save_tokenizer_json`], and the whole point of the module:
/// model (vocab/merges + knobs), normalizer, pre-tokenizer, decoder, and
/// added tokens are all queried from the graph and fed to the `tokenizers`
/// builders, and so is the post-processor when the graph stores one (graphs
/// written before 2026-11-07 do not; callers that hand-frame their sentinels
/// encode with `add_special_tokens = false`, which never runs it).
#[cfg(feature = "tokenizer")]
pub fn build_tokenizer(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Result<tokenizers::Tokenizer, Err> {
    build_tokenizer_with_added(tribles, blobs, tok_id, true)
}

/// [`build_tokenizer`], with the choice of leaving the added tokens out.
///
/// Without them the result is the CONTENT-ONLY view a chat codec needs:
/// every added token of our tokenizers is special, so a tokenizer built
/// without them turns a literal marker spelling in untrusted text into
/// ordinary pieces instead of structure.
#[cfg(feature = "tokenizer")]
pub fn build_tokenizer_with_added(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
    with_added: bool,
) -> Result<tokenizers::Tokenizer, Err> {
    use tokenizers::models::bpe::BPE;
    use tokenizers::models::wordpiece::WordPiece;

    let tags = node_tags(tribles, tok_id);
    let vocab: tokenizers::models::bpe::Vocab = load_vocab_strict(tribles, blobs, tok_id)?
        .into_iter()
        .map(|(t, i)| (t, i as u32))
        .collect();
    let unk = long_field!(tribles, blobs, tok_id, attrs::unk_token);
    let csp = long_field!(tribles, blobs, tok_id, attrs::continuing_subword_prefix);
    let eows = long_field!(tribles, blobs, tok_id, attrs::end_of_word_suffix);
    let max_chars = optional_one(
        find!(
            (m: u64),
            pattern!(tribles, [{ tok_id @ attrs::max_input_chars: ?m }])
        )
        .map(|(m,)| m),
        tok_id,
        "max_input_chars",
    )?;

    let model_kinds: Vec<_> = [ty::WORD_PIECE, ty::BPE, ty::TIKTOKEN, ty::UNIGRAM]
        .into_iter()
        .filter(|kind| tags.contains(kind))
        .collect();
    if model_kinds.len() != 1 {
        return Err(format!(
            "tokenizer {tok_id} must carry exactly one model-kind tag; found {}",
            model_kinds.len()
        )
        .into());
    }

    let model: tokenizers::ModelWrapper = if tags.contains(&ty::WORD_PIECE) {
        let mut b = WordPiece::builder().vocab(vocab);
        if let Some(u) = unk {
            b = b.unk_token(u);
        }
        if let Some(p) = csp {
            b = b.continuing_subword_prefix(p);
        }
        if let Some(m) = max_chars {
            b = b.max_input_chars_per_word(m as usize);
        }
        b.build()
            .map_err(|e| format!("build WordPiece model: {e}"))?
            .into()
    } else if tags.contains(&ty::BPE) {
        let merges = load_merges_strict(tribles, blobs, tok_id)?;
        let mut b = BPE::builder().vocab_and_merges(vocab, merges);
        // Not cosmetic: with `ignore_merges` false the merge table re-derives
        // pieces the vocab already names, so a token the model would emit whole
        // comes back as several. Inkling sets it true.
        if tags.contains(&flag::IGNORE_MERGES) {
            b = b.ignore_merges(true);
        }
        if let Some(u) = unk {
            b = b.unk_token(u);
        }
        if let Some(p) = csp {
            b = b.continuing_subword_prefix(p);
        }
        if let Some(s) = eows {
            b = b.end_of_word_suffix(s);
        }
        b.build()
            .map_err(|e| format!("build BPE model: {e}"))?
            .into()
    } else {
        return Err("tokenizer entity carries no model-kind tag (WordPiece/BPE)".into());
    };

    let mut tok = tokenizers::Tokenizer::new(model);
    if let Some(n) = edge_field!(tribles, tok_id, attrs::normalizer) {
        tok.with_normalizer(Some(build_normalizer(tribles, blobs, n)?));
    }
    if let Some(p) = edge_field!(tribles, tok_id, attrs::pre_tokenizer) {
        tok.with_pre_tokenizer(Some(build_pre_tokenizer(tribles, blobs, p)?));
    }
    if let Some(d) = edge_field!(tribles, tok_id, attrs::decoder) {
        tok.with_decoder(Some(build_decoder(tribles, blobs, d)?));
    }
    if let Some(p) = edge_field!(tribles, tok_id, attrs::post_processor) {
        tok.with_post_processor(Some(build_post_processor(tribles, blobs, p)?));
    }

    // Added tokens, in their original order and with their options (absent
    // options are the pre-2026-11-07 reconstruction: special, nothing else).
    // `tokenizers` assigns an id to a token the vocab lacks by appending it
    // after the vocab and the tokens added before it, so the stored order is
    // what makes the ids come out right; each is checked against the stored
    // id, because a graph whose added ids are not contiguous after the vocab
    // would otherwise rebuild with silently different ids.
    if with_added {
        for (content, id, options) in load_added_strict(tribles, blobs, tok_id)? {
            let special = !options.contains(&flag::NOT_SPECIAL);
            let token = tokenizers::AddedToken::from(content.clone(), special)
                .normalized(options.contains(&flag::NORMALIZED))
                .lstrip(options.contains(&flag::LSTRIP))
                .rstrip(options.contains(&flag::RSTRIP))
                .single_word(options.contains(&flag::SINGLE_WORD));
            if special {
                tok.add_special_tokens(&[token]);
            } else {
                tok.add_tokens(&[token]);
            }
            let assigned = tok.token_to_id(&content);
            if assigned != Some(id as u32) {
                return Err(format!(
                    "added token {content:?} has id {id} in tokenizer {tok_id}, \
                     but rebuilding assigns {assigned:?}"
                )
                .into());
            }
        }
    }
    Ok(tok)
}

/// Reconstruct one post-processor node (recursing through `Sequence`).
#[cfg(feature = "tokenizer")]
fn build_post_processor(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    node: Id,
) -> Result<tokenizers::PostProcessorWrapper, Err> {
    use tokenizers::processors as pp;
    let tags = node_tags(tribles, node);
    let has = |id: Id| tags.contains(&id);
    let token = |edge: Id| -> Result<(String, u32), Err> {
        let piece = long_field!(tribles, blobs, edge, attrs::piece)
            .ok_or_else(|| format!("cls/sep token {edge} has no piece"))?;
        let id = required_one(
            find!((i: u64), pattern!(tribles, [{ edge @ attrs::token_id: ?i }])).map(|(i,)| i),
            edge,
            "token_id",
        )?;
        Ok((piece, id as u32))
    };
    if has(ty::TEMPLATE_PROCESSING) {
        let single = build_template(tribles, blobs, node, true)?;
        let pair = build_template(tribles, blobs, node, false)?;
        let mut specials = Vec::new();
        for (special,) in find!(
            (s: Id),
            pattern!(tribles, [{ node @ attrs::special_token: ?s }])
        ) {
            let name = long_field!(tribles, blobs, special, attrs::special_id)
                .ok_or_else(|| format!("special token {special} has no special_id"))?;
            let mut ids = Vec::new();
            let mut contents = Vec::new();
            for member in ordered_members_strict(tribles, special)? {
                let (content, id) = token(member)?;
                contents.push(content);
                ids.push(id);
            }
            specials.push(
                pp::template::SpecialToken::new(name, ids, contents)
                    .map_err(|e| format!("special token: {e}"))?,
            );
        }
        let processor = pp::template::TemplateProcessing::builder()
            .try_single(single)
            .map_err(|e| format!("single template: {e}"))?
            .try_pair(pair)
            .map_err(|e| format!("pair template: {e}"))?
            .special_tokens(specials)
            .build()
            .map_err(|e| format!("build TemplateProcessing: {e}"))?;
        Ok(processor.into())
    } else if has(ty::BERT_PROCESSING) || has(ty::ROBERTA_PROCESSING) {
        let sep =
            token(edge_field!(tribles, node, attrs::sep).ok_or("processing node has no sep")?)?;
        let cls =
            token(edge_field!(tribles, node, attrs::cls).ok_or("processing node has no cls")?)?;
        if has(ty::BERT_PROCESSING) {
            Ok(pp::bert::BertProcessing::new(sep, cls).into())
        } else {
            Ok(pp::roberta::RobertaProcessing::new(sep, cls)
                .trim_offsets(has(flag::TRIM_OFFSETS))
                .add_prefix_space(has(flag::ADD_PREFIX_SPACE))
                .into())
        }
    } else if has(ty::BYTE_LEVEL) {
        Ok(tokenizers::pre_tokenizers::byte_level::ByteLevel::new(
            has(flag::ADD_PREFIX_SPACE),
            has(flag::TRIM_OFFSETS),
            has(flag::USE_REGEX),
        )
        .into())
    } else if has(ty::SEQUENCE) {
        let kids = ordered_members_strict(tribles, node)?
            .into_iter()
            .map(|k| build_post_processor(tribles, blobs, k))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(pp::sequence::Sequence::new(kids).into())
    } else {
        Err(format!("post-processor node {node:?} has no known type tag").into())
    }
}

/// One `TemplateProcessing` template in `tokenizers`' piece syntax
/// (`$A:0`, `<embedding>:0`), the one public constructor it has. That syntax
/// splits on spaces and colons and reads a leading `$` as a sequence, so a
/// special-token name containing either, or starting with `$`, is refused
/// rather than parsed into something else.
#[cfg(feature = "tokenizer")]
fn build_template(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    node: Id,
    single: bool,
) -> Result<Vec<String>, Err> {
    use std::collections::{BTreeMap, BTreeSet};
    let pieces: BTreeSet<Id> = if single {
        find!((p: Id), pattern!(tribles, [{ node @ attrs::template_single: ?p }]))
            .map(|(p,)| p)
            .collect()
    } else {
        find!((p: Id), pattern!(tribles, [{ node @ attrs::template_pair: ?p }]))
            .map(|(p,)| p)
            .collect()
    };
    let mut ordered = BTreeMap::new();
    for piece in pieces {
        let index = required_one(
            find!((i: u64), pattern!(tribles, [{ piece @ attrs::index: ?i }])).map(|(i,)| i),
            piece,
            "index",
        )?;
        let type_id = required_one(
            find!((t: u64), pattern!(tribles, [{ piece @ attrs::type_id: ?t }])).map(|(t,)| t),
            piece,
            "type_id",
        )?;
        let tags = node_tags(tribles, piece);
        let name = if tags.contains(&ty::TEMPLATE_SEQUENCE_A) {
            "$A".to_owned()
        } else if tags.contains(&ty::TEMPLATE_SEQUENCE_B) {
            "$B".to_owned()
        } else {
            let name = long_field!(tribles, blobs, piece, attrs::special_id)
                .ok_or_else(|| format!("template piece {piece} names no sequence or token"))?;
            if name.is_empty() || name.starts_with('$') || name.contains([' ', ':']) {
                return Err(format!(
                    "special token name {name:?} cannot be spelled in tokenizers' template syntax"
                )
                .into());
            }
            name
        };
        if ordered.insert(index, format!("{name}:{type_id}")).is_some() {
            return Err(format!("template of {node} has duplicate piece index {index}").into());
        }
    }
    Ok(ordered.into_values().collect())
}

/// Reconstruct one normalizer node (recursing through `Sequence`).
#[cfg(feature = "tokenizer")]
fn build_normalizer(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    node: Id,
) -> Result<tokenizers::NormalizerWrapper, Err> {
    use tokenizers::normalizers as n;
    let tags = node_tags(tribles, node);
    let has = |id: Id| tags.contains(&id);
    if has(ty::BERT_NORMALIZER) {
        // strip_accents was never emitted (null in our sources) → None.
        Ok(n::BertNormalizer::new(
            has(flag::CLEAN_TEXT),
            has(flag::HANDLE_CHINESE_CHARS),
            None,
            has(flag::LOWERCASE),
        )
        .into())
    } else if has(ty::SEQUENCE) {
        let kids = ordered_members_strict(tribles, node)?
            .into_iter()
            .map(|k| build_normalizer(tribles, blobs, k))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(n::Sequence::new(kids).into())
    } else if has(ty::NFC) {
        Ok(n::NFC.into())
    } else if has(ty::LOWERCASE) {
        Ok(n::Lowercase.into())
    } else if has(ty::REPLACE) {
        let pat = long_field!(tribles, blobs, node, attrs::pattern)
            .ok_or("Replace node missing pattern")?;
        let content = long_field!(tribles, blobs, node, attrs::replace_content).unwrap_or_default();
        // Patterns are stored raw from the Regex variant; rebuild as Regex.
        Ok(
            n::Replace::new(n::replace::ReplacePattern::Regex(pat), content)
                .map_err(|e| format!("build Replace normalizer: {e}"))?
                .into(),
        )
    } else {
        Err(format!("normalizer node {node:?} has no known type tag").into())
    }
}

/// Reconstruct one pre-tokenizer node (recursing through `Sequence`).
#[cfg(feature = "tokenizer")]
fn build_pre_tokenizer(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    node: Id,
) -> Result<tokenizers::PreTokenizerWrapper, Err> {
    use tokenizers::pre_tokenizers as p;
    let tags = node_tags(tribles, node);
    let has = |id: Id| tags.contains(&id);
    if has(ty::BERT_PRE_TOKENIZER) {
        Ok(p::bert::BertPreTokenizer.into())
    } else if has(ty::SEQUENCE) {
        let kids = ordered_members_strict(tribles, node)?
            .into_iter()
            .map(|k| build_pre_tokenizer(tribles, blobs, k))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(p::sequence::Sequence::new(kids).into())
    } else if has(ty::SPLIT) {
        let pat = long_field!(tribles, blobs, node, attrs::pattern)
            .ok_or("Split node missing pattern")?;
        let behavior = long_field!(tribles, blobs, node, attrs::behavior)
            .ok_or("Split node missing behavior")?;
        Ok(p::split::Split::new(
            p::split::SplitPattern::Regex(pat),
            parse_behavior(&behavior)?,
            has(flag::INVERT),
        )
        .map_err(|e| format!("build Split pre-tokenizer: {e}"))?
        .into())
    } else if has(ty::BYTE_LEVEL) {
        // use_regex comes from the graph now. It used to be hardcoded true
        // with a comment saying to revisit it, and Inkling is the tokenizer
        // that made it matter: `Split(Regex, Isolated)` then
        // `ByteLevel { use_regex: false }`, where a true here re-splits what
        // the Split already split.
        Ok(p::byte_level::ByteLevel::new(
            has(flag::ADD_PREFIX_SPACE),
            has(flag::TRIM_OFFSETS),
            has(flag::USE_REGEX),
        )
        .into())
    } else {
        Err(format!("pre-tokenizer node {node:?} has no known type tag").into())
    }
}

/// Reconstruct the decoder node.
#[cfg(feature = "tokenizer")]
fn build_decoder(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    node: Id,
) -> Result<tokenizers::DecoderWrapper, Err> {
    let tags = node_tags(tribles, node);
    let has = |id: Id| tags.contains(&id);
    if has(ty::WORD_PIECE) {
        let prefix = long_field!(tribles, blobs, node, attrs::continuing_subword_prefix)
            .unwrap_or_else(|| "##".to_string());
        Ok(tokenizers::decoders::wordpiece::WordPiece::new(prefix, has(flag::CLEANUP)).into())
    } else if has(ty::BYTE_LEVEL) {
        Ok(tokenizers::pre_tokenizers::byte_level::ByteLevel::new(
            has(flag::ADD_PREFIX_SPACE),
            has(flag::TRIM_OFFSETS),
            has(flag::USE_REGEX),
        )
        .into())
    } else {
        let _ = blobs;
        Err(format!("decoder node {node:?} has no known type tag").into())
    }
}

/// `SplitDelimiterBehavior` from its persisted name.
#[cfg(feature = "tokenizer")]
fn parse_behavior(name: &str) -> Result<tokenizers::SplitDelimiterBehavior, Err> {
    use tokenizers::SplitDelimiterBehavior as B;
    Ok(match name {
        "Removed" => B::Removed,
        "Isolated" => B::Isolated,
        "MergedWithPrevious" => B::MergedWithPrevious,
        "MergedWithNext" => B::MergedWithNext,
        "Contiguous" => B::Contiguous,
        other => return Err(format!("unknown SplitDelimiterBehavior {other:?}").into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny WordPiece-ish tokenizer.json exercising vocab + added tokens.
    // (Full added-token fields + "version" so `Tokenizer::from_bytes` also
    // accepts it — the parity tests compare graph-built vs json-built.)
    const WP: &str = r###"{
      "version": "1.0",
      "added_tokens": [
        {"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
      "pre_tokenizer": {"type": "BertPreTokenizer"},
      "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
      "model": { "type": "WordPiece", "unk_token": "[UNK]",
        "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {
        "[PAD]": 0, "hello": 1, "[CLS]": 2, "##ing": 3, "telecommunications": 4
      } }
    }"###;

    // A tiny BPE tokenizer.json exercising merges (both array + string forms
    // are accepted; HF current form is the array).
    const BPE: &str = r#"{
      "version": "1.0",
      "added_tokens": [],
      "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "NFC"},
        {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
        {"type": "Lowercase"}
      ]},
      "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "foo"}, "behavior": "Removed", "invert": true},
        {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true}
      ]},
      "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
      "model": { "type": "BPE",
        "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3, "abc": 4},
        "merges": [["a","b"], ["ab","c"]]
      }
    }"#;

    #[test]
    fn wordpiece_vocab_round_trips() {
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(WP.as_bytes(), "test/wp", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        let vocab = load_vocab(&tribles, &reader, tok_id);
        assert_eq!(vocab.len(), 5);
        assert_eq!(vocab.get("telecommunications"), Some(&4));
        assert_eq!(vocab.get("##ing"), Some(&3));
        assert_eq!(vocab.get("[PAD]"), Some(&0));
        assert!(load_merges(&tribles, &reader, tok_id).is_empty());

        // flat model knob round-trips (U256BE -> u64 via TryFromInline)
        let max: Option<u64> = find!(
            (m: u64),
            pattern!(&tribles, [{ tok_id @ attrs::max_input_chars: ?m }])
        )
        .next()
        .map(|(m,)| m);
        assert_eq!(max, Some(100));
    }

    #[test]
    fn bpe_merges_round_trip_in_rank_order() {
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(BPE.as_bytes(), "test/bpe", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        let merges = load_merges(&tribles, &reader, tok_id);
        assert_eq!(
            merges,
            vec![
                ("a".to_string(), "b".to_string()),
                ("ab".to_string(), "c".to_string()),
            ]
        );
    }

    // Real-data validation against an actual HuggingFace tokenizer.json (30k+
    // vocabs, byte-level BPE emoji tokens over the 32-byte inline ceiling).
    // Ignored by default; run pointed at a file:
    //   TOK_JSON=<path> TOK_VOCAB=<n> cargo test -p maryml --lib \
    //     tokenizer::tests::real_tokenizer_bulk -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_tokenizer_bulk() {
        let Ok(path) = std::env::var("TOK_JSON") else {
            eprintln!("[real] set TOK_JSON to run");
            return;
        };
        let json = std::fs::read(&path).expect("read TOK_JSON");
        let mut blobs = MemoryBlobStore::new();
        let t = std::time::Instant::now();
        let frag = save_tokenizer_json(&json, "real", &mut blobs).unwrap();
        let ingest_ms = t.elapsed().as_millis();
        let tok_id = frag.root().expect("root");
        let t = std::time::Instant::now();
        let tribles: TribleSet = frag.into();
        let into_ms = t.elapsed().as_millis();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        let t = std::time::Instant::now();
        let vocab = load_vocab(&tribles, &reader, tok_id);
        let load_vocab_ms = t.elapsed().as_millis();
        let t = std::time::Instant::now();
        let merges = load_merges(&tribles, &reader, tok_id);
        let load_merges_ms = t.elapsed().as_millis();
        eprintln!(
            "[real] {path}: {} vocab, {} merges | ingest {ingest_ms}ms  frag->set {into_ms}ms  load_vocab {load_vocab_ms}ms  load_merges {load_merges_ms}ms",
            vocab.len(),
            merges.len()
        );
        // prove an over-inline-ceiling token survived the UTF8String round-trip
        if let Some((tok, id)) = vocab.iter().find(|(t, _)| t.len() > 32) {
            eprintln!("[real]   >32-byte token round-tripped: {tok:?} = {id}");
        }
        if let Ok(n) = std::env::var("TOK_VOCAB") {
            assert_eq!(vocab.len(), n.parse::<usize>().unwrap(), "vocab count");
        }
    }

    // Same-process A/B: point-lookup vs single joined pattern!, both on the SAME
    // ingested TribleSet — kills cross-run ingest variance, isolating the pure
    // planner cost of the two query shapes. Runs on whatever engine triblespace
    // links (currently the residual/agglomerative planner).
    //   TOK_JSON=<path> cargo test -p maryml --lib tokenizer::tests::real_load_ab \
    //     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_load_ab() {
        let Ok(path) = std::env::var("TOK_JSON") else {
            eprintln!("[ab] set TOK_JSON to run");
            return;
        };
        let json = std::fs::read(&path).expect("read TOK_JSON");
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(&json, "real", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        // Alternate joined (B) and point-lookup (A) across rounds to expose any
        // cache-warming / ordering effect (ingest is paid once; queries are cheap).
        for round in 0..3 {
            let t = std::time::Instant::now();
            let b: HashMap<String, u64> = find!(
                (p, i: u64),
                pattern!(&tribles, [
                    { tok_id @ attrs::vocab: _?entry },
                    { _?entry @ attrs::piece: ?p, attrs::token_id: ?i },
                ])
            )
            .map(|(ph, id)| (read_piece(&reader, ph), id))
            .collect();
            let b_ms = t.elapsed().as_millis();

            let t = std::time::Instant::now();
            let a = load_vocab(&tribles, &reader, tok_id);
            let a_ms = t.elapsed().as_millis();

            eprintln!("[ab] round {round}: B joined {b_ms}ms  |  A point {a_ms}ms");
            assert_eq!(a.len(), b.len(), "A and B must agree");
        }
    }

    // the config edge can't be a runtime param (pattern! needs the attr at
    // expansion), so a tiny dispatch:
    fn edge(tribles: &TribleSet, subj: Id, which: &str) -> Option<Id> {
        match which {
            "normalizer" => find!((n: Id), pattern!(tribles, [{ subj @ attrs::normalizer: ?n }]))
                .next()
                .map(|(n,)| n),
            "pre_tokenizer" => {
                find!((n: Id), pattern!(tribles, [{ subj @ attrs::pre_tokenizer: ?n }]))
                    .next()
                    .map(|(n,)| n)
            }
            "decoder" => find!((n: Id), pattern!(tribles, [{ subj @ attrs::decoder: ?n }]))
                .next()
                .map(|(n,)| n),
            _ => None,
        }
    }

    #[test]
    fn config_tail_round_trips() {
        // WordPiece: BertNormalizer + BertPreTokenizer + WordPiece decoder
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(WP.as_bytes(), "test/wp", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();

        let norm = edge(&tribles, tok_id, "normalizer").expect("normalizer edge");
        let nt = node_tags(&tribles, norm);
        assert!(nt.contains(&ty::BERT_NORMALIZER));
        assert!(nt.contains(&flag::CLEAN_TEXT));
        assert!(nt.contains(&flag::LOWERCASE));

        let pretok = edge(&tribles, tok_id, "pre_tokenizer").expect("pre_tokenizer edge");
        assert!(node_tags(&tribles, pretok).contains(&ty::BERT_PRE_TOKENIZER));

        let dec = edge(&tribles, tok_id, "decoder").expect("decoder edge");
        let dt = node_tags(&tribles, dec);
        assert!(dt.contains(&ty::WORD_PIECE));
        assert!(dt.contains(&flag::CLEANUP));

        // BPE: Sequence[NFC, Replace, Lowercase] normalizer, ORDERED
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(BPE.as_bytes(), "test/bpe", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();

        let seq = edge(&tribles, tok_id, "normalizer").expect("normalizer");
        assert!(node_tags(&tribles, seq).contains(&ty::SEQUENCE));
        let kids = ordered_members(&tribles, seq);
        assert_eq!(kids.len(), 3, "NFC, Replace, Lowercase");
        assert!(node_tags(&tribles, kids[0]).contains(&ty::NFC));
        assert!(node_tags(&tribles, kids[1]).contains(&ty::REPLACE));
        assert!(node_tags(&tribles, kids[2]).contains(&ty::LOWERCASE));
    }

    // Regression (real bug, 2026-07-18): the WP fixture's DECODER node is
    // also tagged ty::WORD_PIECE (type ids are role-agnostic), and
    // find_tokenizer once matched it by tag alone — query iteration order
    // decided whether callers got the root or a config node whose empty
    // vocab broke [CLS] resolution downstream. The root is the only entity
    // carrying tag + vocabulary.
    #[test]
    fn find_tokenizer_skips_the_wordpiece_decoder_node() {
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(WP.as_bytes(), "test/wp", &mut blobs).unwrap();
        let root = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        let dec = edge(&tribles, root, "decoder").expect("decoder edge");
        assert!(
            node_tags(&tribles, dec).contains(&ty::WORD_PIECE),
            "fixture precondition: decoder node shares the WordPiece tag"
        );
        assert_eq!(find_tokenizer(&tribles), Some(root));
        assert_ne!(root, dec);
    }

    /// The tiktoken save/load pair on a three-token toy vocab: raw bytes (NOT
    /// a `String` — that is the whole reason `piece_bytes` exists), the 256
    /// reserved slots with their two naming rules, and the pattern. The
    /// full-scale version of this is `k3_tokenizer_gate`, which additionally
    /// checks every id against the shipped Python.
    #[cfg(feature = "k3tok")]
    #[test]
    fn tiktoken_ranks_and_specials_round_trip() {
        // 0xff is not valid UTF-8 on its own — exactly the case a `UTF8String`
        // `piece` could not hold.
        let model = "YQ== 0\nYg== 1\nYWI= 2\n/w== 3\n";
        let config = r#"{"added_tokens_decoder": {"5": {"content": "[BOS]", "special": true}},
                         "unk_token": "[UNK]"}"#;
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tiktoken(
            model.as_bytes(),
            config.as_bytes(),
            "PAT",
            "test/tik",
            &mut blobs,
        )
        .unwrap();
        let root = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        assert_eq!(find_tokenizer(&tribles), Some(root));
        let mut ranks = load_tiktoken_ranks(&tribles, &reader, root);
        ranks.sort_by_key(|(_, r)| *r);
        assert_eq!(
            ranks,
            vec![
                (b"a".to_vec(), 0),
                (b"b".to_vec(), 1),
                (b"ab".to_vec(), 2),
                (vec![0xffu8], 3),
            ]
        );
        assert_eq!(
            load_pre_tokenizer_pattern(&tribles, &reader, root).as_deref(),
            Some("PAT")
        );

        let added = load_added(&tribles, &reader, root);
        assert_eq!(added.len(), NUM_RESERVED_SPECIAL_TOKENS as usize);
        // named by the config where it names them, `<|reserved_token_{ABSOLUTE
        // id}|>` everywhere else — the shipped `__init__`'s two rules.
        assert_eq!(added[0], ("<|reserved_token_4|>".to_string(), 4));
        assert_eq!(added[1], ("[BOS]".to_string(), 5));
        assert_eq!(added[255], ("<|reserved_token_259|>".to_string(), 259));
    }

    // ── construct-from-graph parity: the graph-built tokenizer must encode
    //    exactly like the json-built one (same `tokenizers` executor, two
    //    different loading substrates). ──

    /// A byte-level BPE fixture whose vocab can actually encode text (Ġ = the
    /// byte-level space), so parity is meaningful, not vacuous.
    #[cfg(feature = "tokenizer")]
    const BPE_BYTELEVEL: &str = r#"{
      "version": "1.0",
      "added_tokens": [],
      "normalizer": {"type": "Lowercase"},
      "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true},
      "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
      "model": { "type": "BPE",
        "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4,
                  "Ġ": 5, "Ġa": 6, "Ġab": 7, "Ġabc": 8},
        "merges": [["a","b"], ["ab","c"], ["Ġ","a"], ["Ġa","b"], ["Ġab","c"]]
      }
    }"#;

    #[cfg(feature = "tokenizer")]
    fn assert_encode_parity(json: &str, texts: &[&str]) {
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(json.as_bytes(), "test/parity", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        assert_eq!(find_tokenizer(&tribles), Some(tok_id), "find_tokenizer");
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        let graph_tok = build_tokenizer(&tribles, &reader, tok_id).expect("build from graph");
        let json_tok = tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("from_bytes");
        for text in texts {
            let g = graph_tok.encode(*text, false).expect("graph encode");
            let j = json_tok.encode(*text, false).expect("json encode");
            assert_eq!(
                g.get_ids(),
                j.get_ids(),
                "graph/json encode diverged on {text:?}"
            );
        }
    }

    #[cfg(feature = "tokenizer")]
    fn project_current_model_attributes_to_historical(facts: &TribleSet) -> TribleSet {
        let aliases = crate::model_collection::legacy_model_attribute_aliases();
        facts
            .iter()
            .map(|fact| {
                let Some(alias) = aliases.iter().find(|alias| &alias.canonical == fact.a()) else {
                    return *fact;
                };
                Trible::force(fact.e(), &alias.historical, fact.v::<UnknownInline>())
            })
            .collect()
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn projected_historical_wordpiece_graph_preserves_encoding() {
        let mut blobs = MemoryBlobStore::new();
        let fragment = save_tokenizer_json(WP.as_bytes(), "test/legacy-wp", &mut blobs).unwrap();
        let root = fragment.root().expect("root");
        let current = fragment.into_facts();
        let historical = project_current_model_attributes_to_historical(&current);
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        assert_eq!(find_tokenizer(&historical), None);
        let projection = crate::model_collection::project_legacy_model_attributes(&historical);
        assert!(
            historical
                .iter()
                .all(|fact| projection.facts.contains(fact))
        );
        assert_eq!(find_tokenizer(&projection.facts), Some(root));

        let graph = build_tokenizer(&projection.facts, &reader, root)
            .expect("build projected historical tokenizer");
        let json = tokenizers::Tokenizer::from_bytes(WP.as_bytes()).expect("build JSON tokenizer");
        for text in [
            "hello telecommunications",
            "HELLO Hello hello",
            "telecommunicationsing",
            "[CLS] hello [PAD]",
        ] {
            let graph_ids = graph.encode(text, false).expect("graph encode");
            let json_ids = json.encode(text, false).expect("JSON encode");
            assert_eq!(
                graph_ids.get_ids(),
                json_ids.get_ids(),
                "projected graph/JSON encode diverged on {text:?}"
            );
        }
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn graph_built_wordpiece_encodes_like_json_built() {
        assert_encode_parity(
            WP,
            &[
                "hello telecommunications",
                "HELLO Hello hello",     // BertNormalizer lowercase
                "telecommunicationsing", // WordPiece ##ing continuation
                "[CLS] hello [PAD]",     // added tokens match as specials
            ],
        );
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn graph_built_bytelevel_bpe_encodes_like_json_built() {
        assert_encode_parity(BPE_BYTELEVEL, &["ab abc", "ABC aB", "abc ab a", "a b c"]);
    }

    // Sequence/Split/Replace plumbing: the original BPE fixture's pre-tok
    // (Split invert "foo" + ByteLevel) is degenerate for real text, but the
    // graph and json builds must still agree on whatever it produces.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn graph_built_sequence_split_replace_encodes_like_json_built() {
        assert_encode_parity(BPE, &["foofoo", "ab   abc", "a c"]);
    }

    // Real-data parity: graph-built vs json-built on an actual HuggingFace
    // tokenizer.json, over real prose. Ignored by default; run pointed at a
    // file:
    //   TOK_JSON=<path> cargo test --features tokenizer --lib \
    //     tokenizer::tests::real_graph_parity -- --ignored --nocapture
    /// A tokenizer whose added tokens lie beyond its vocab, one of them not
    /// special, with a post-processor: the shape WeMM's has.
    #[cfg(feature = "tokenizer")]
    fn with_post_processor(post: &str) -> String {
        format!(
            r###"{{
          "version": "1.0",
          "added_tokens": [
            {{"id": 5, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
            {{"id": 6, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}},
            {{"id": 7, "content": "<embedding>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
          ],
          "normalizer": {{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}},
          "pre_tokenizer": {{"type": "BertPreTokenizer"}},
          "post_processor": {post},
          "decoder": {{"type": "WordPiece", "prefix": "##", "cleanup": true}},
          "model": {{ "type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {{
            "[UNK]": 0, "hello": 1, "world": 2, "##ing": 3, "[SEP]": 4 }} }}
        }}"###
        )
    }

    #[cfg(feature = "tokenizer")]
    fn assert_full_parity(json: &str, texts: &[&str]) {
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(json.as_bytes(), "test/post", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let graph = build_tokenizer(&tribles, &reader, tok_id).expect("build from graph");
        let json = tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("from_bytes");
        for text in texts {
            for specials in [false, true] {
                let g = graph.encode(*text, specials).expect("graph encode");
                let j = json.encode(*text, specials).expect("json encode");
                assert_eq!(g.get_ids(), j.get_ids(), "{text:?} specials={specials}");
                assert_eq!(g.get_type_ids(), j.get_type_ids(), "{text:?}");
                assert_eq!(
                    g.get_special_tokens_mask(),
                    j.get_special_tokens_mask(),
                    "{text:?}"
                );
                for skip in [false, true] {
                    assert_eq!(
                        graph.decode(g.get_ids(), skip).unwrap(),
                        json.decode(j.get_ids(), skip).unwrap(),
                        "decode {text:?} skip={skip}"
                    );
                }
            }
            let g = graph.encode((*text, "world hello"), true).unwrap();
            let j = json.encode((*text, "world hello"), true).unwrap();
            assert_eq!(g.get_ids(), j.get_ids(), "pair {text:?}");
            assert_eq!(g.get_type_ids(), j.get_type_ids(), "pair {text:?}");
        }
    }

    const POST_TEXTS: &[&str] = &[
        "hello world",
        "<|im_start|>hello<think>worlding",
        "hello <embedding> world",
        "",
    ];

    #[cfg(feature = "tokenizer")]
    #[test]
    fn template_post_processor_and_added_options_rebuild_exactly() {
        let template = r#"{"type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<embedding>", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<embedding>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<embedding>": {"id": "<embedding>", "ids": [7], "tokens": ["<embedding>"]}}}"#;
        assert_full_parity(&with_post_processor(template), POST_TEXTS);
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn bert_roberta_bytelevel_and_sequence_post_processors_rebuild_exactly() {
        let bert = r#"{"type": "BertProcessing", "sep": ["[SEP]", 4], "cls": ["<|im_start|>", 5]}"#;
        let roberta = r#"{"type": "RobertaProcessing", "sep": ["[SEP]", 4], "cls": ["<|im_start|>", 5], "trim_offsets": false, "add_prefix_space": true}"#;
        let sequence = r#"{"type": "Sequence", "processors": [
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            {"type": "BertProcessing", "sep": ["[SEP]", 4], "cls": ["<|im_start|>", 5]}]}"#;
        for post in [bert, roberta, sequence] {
            assert_full_parity(&with_post_processor(post), POST_TEXTS);
        }
    }

    /// A graph written before the options existed (no flags on its added
    /// tokens) still rebuilds every added token as special.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn added_tokens_without_options_rebuild_as_special() {
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(WP.as_bytes(), "test/legacy", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let mut tribles: TribleSet = frag.into();
        let options = [
            flag::NOT_SPECIAL,
            flag::NORMALIZED,
            flag::LSTRIP,
            flag::RSTRIP,
            flag::SINGLE_WORD,
        ];
        let legacy: TribleSet = tribles
            .iter()
            .filter(|t| {
                !(t.a() == &metadata::tag.id()
                    && options.iter().any(|o| {
                        let tag: Inline<inlineencodings::GenId> = (*o).to_inline();
                        t.v::<inlineencodings::GenId>().raw == tag.raw
                    }))
            })
            .copied()
            .collect();
        tribles = legacy;
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let graph = build_tokenizer(&tribles, &reader, tok_id).unwrap();
        let ids = graph
            .encode("[CLS] hello", false)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(graph.decode(&ids, true).unwrap(), "hello");
    }

    /// WeMM's real tokenizer: the graph-built tokenizer encodes like the
    /// checkpoint's `tokenizer.json`, with and without its post-processor,
    /// on chat-template framing, the image placeholder run and every added
    /// token spelled literally.
    #[cfg(feature = "tokenizer")]
    #[test]
    #[ignore = "set WEMM_TOKENIZER_JSON to WeMM's tokenizer.json"]
    fn wemm_graph_tokenizer_encodes_like_its_json() {
        let path = std::env::var("WEMM_TOKENIZER_JSON").expect("WEMM_TOKENIZER_JSON");
        let json = std::fs::read(&path).unwrap();
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(&json, "wemm", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let graph = build_tokenizer(&tribles, &reader, tok_id).unwrap();
        let reference = tokenizers::Tokenizer::from_bytes(&json).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&json).unwrap();
        let added: Vec<String> = parsed["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["content"].as_str().unwrap().to_owned())
            .collect();
        let mut texts: Vec<String> = [
            "hello",
            "The quick brown fox jumps over the lazy dog.",
            "  leading spaces, trailing newline\n",
            "line one\nline two\r\n\r\nline four\n\n",
            "naïve façade coöperation, Ünïcödé and NFC: e\u{301} vs \u{e9}",
            "数字 123 4567 and 漢字かなカナ 한국어",
            "emoji 🦀🧪 👩‍👩‍👧 zero\u{200b}width",
            "can't won't I'll we've they're it's",
            "fn main() { println!(\"{}\", 1 + 2); } // code\ttabs",
            "<think>reasoning</think><tool_call>{\"a\": 1}</tool_call>",
            "<|fim_prefix|>x<|fim_suffix|>y<|fim_middle|>",
            "text with a literal <embedding> inside",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        // The codec's framing (input_codec.rs) around plain and tricky text.
        for content in ["hello", "a red crab with claws", "  spaced \n", ""] {
            texts.push(format!("<|im_start|>user\n{content}<|im_end|>\n"));
        }
        texts.push(format!(
            "<|im_start|>user\n<|vision_start|>{}<|vision_end|><|im_end|>\n",
            "<|image_pad|>".repeat(64)
        ));
        texts.push(added.concat());
        texts.push(added.join(" plain "));
        for text in &texts {
            for specials in [false, true] {
                let g = graph.encode(text.as_str(), specials).unwrap();
                let j = reference.encode(text.as_str(), specials).unwrap();
                assert_eq!(g.get_ids(), j.get_ids(), "{text:?} specials={specials}");
                assert_eq!(g.get_special_tokens_mask(), j.get_special_tokens_mask());
                for skip in [false, true] {
                    assert_eq!(
                        graph.decode(g.get_ids(), skip).unwrap(),
                        reference.decode(j.get_ids(), skip).unwrap()
                    );
                }
            }
        }
        for content in &added {
            assert_eq!(graph.token_to_id(content), reference.token_to_id(content));
        }
        assert_eq!(graph.get_vocab_size(true), reference.get_vocab_size(true));
        eprintln!(
            "[wemm parity] {} texts x2 identical, {} added tokens",
            texts.len(),
            added.len()
        );
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    #[ignore]
    fn real_graph_parity() {
        let Ok(path) = std::env::var("TOK_JSON") else {
            eprintln!("[parity] set TOK_JSON to run");
            return;
        };
        let json = std::fs::read(&path).expect("read TOK_JSON");
        let mut blobs = MemoryBlobStore::new();
        let frag = save_tokenizer_json(&json, "real", &mut blobs).unwrap();
        let tok_id = frag.root().expect("root");
        let tribles: TribleSet = frag.into();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        let t = std::time::Instant::now();
        let graph_tok = build_tokenizer(&tribles, &reader, tok_id).expect("build from graph");
        let build_ms = t.elapsed().as_millis();
        let json_tok = tokenizers::Tokenizer::from_bytes(&json).expect("from_bytes");

        let texts = [
            "search_document: The pile is the durable store; the HF cache is an evictable download artifact.",
            "search_query: tokenizer as a content-addressed graph",
            "Attention Is All You Need (Vaswani et al., 2017) — 10,000+ citations.",
            "naïve façade coöperation — diacritics exercise the normalizer",
            "[CLS] explicit sentinels survive [SEP]",
            "CamelCase suffixes: telecommunications infrastructure modernization",
            "punctuation, quotes \"double\" and 'single'; hyphen-ated words!",
            "     leading and trailing whitespace     ",
        ];
        for text in texts {
            let g = graph_tok.encode(text, false).expect("graph encode");
            let j = json_tok.encode(text, false).expect("json encode");
            assert_eq!(
                g.get_ids(),
                j.get_ids(),
                "graph/json encode diverged on {text:?}"
            );
        }
        eprintln!(
            "[parity] {path}: {} texts identical (graph build {build_ms}ms)",
            texts.len()
        );
    }
}

// ───────────────────── SentencePiece UNIGRAM (added 2026-08-04) ─────────────

/// Map a SentencePiece piece-type discriminant to its tag id.
fn spm_type_tag(typ: u64) -> Id {
    match typ {
        1 => ty::PIECE_NORMAL,
        2 => ty::PIECE_UNKNOWN,
        3 => ty::PIECE_CONTROL,
        4 => ty::PIECE_USER_DEFINED,
        5 => ty::PIECE_UNUSED,
        6 => ty::PIECE_BYTE,
        other => panic!("spm: unknown piece type {other}"),
    }
}

fn spm_tag_type(tag: Id) -> u64 {
    match tag {
        t if t == ty::PIECE_NORMAL => 1,
        t if t == ty::PIECE_UNKNOWN => 2,
        t if t == ty::PIECE_CONTROL => 3,
        t if t == ty::PIECE_USER_DEFINED => 4,
        t if t == ty::PIECE_UNUSED => 5,
        t if t == ty::PIECE_BYTE => 6,
        _ => panic!("spm: vocab entry carries no piece-type tag"),
    }
}

/// Ingest a SentencePiece UNIGRAM model as a queryable graph.
///
/// Unlike BPE/WordPiece there are no merges: a unigram model IS its scored
/// piece table, so each vocab entry carries `piece_score` (the log-prob that
/// the Viterbi lattice sums) and a piece-type tag (only NORMAL pieces are
/// lattice edges; BYTE pieces back `byte_fallback`). Graph-only — the raw
/// `.model` proto is NOT stored, the pieces ARE the model.
pub fn save_spm_unigram(
    pieces: &[(Vec<u8>, f32, u64)],
    add_dummy_prefix: bool,
    byte_fallback: bool,
    source_name: &str,
    blobs: &mut impl BlobStorePut,
) -> Result<Fragment, Err> {
    let mut facts = Fragment::empty();
    let mut vocab_ids: Vec<Id> = Vec::with_capacity(pieces.len());

    for (id, (bytes, score, typ)) in pieces.iter().enumerate() {
        // Every SPM piece is valid UTF-8: NORMAL pieces are text (still
        // `▁`-escaped), BYTE pieces are the literal "<0xAB>" surface.
        let text = String::from_utf8(bytes.clone())
            .map_err(|e| format!("spm piece {id} is not utf8: {e}"))?;
        let ph = blobs.put::<blobencodings::UTF8String, _>(text)?;
        let e = entity! { _ @
            metadata::tag: spm_type_tag(*typ),
            attrs::piece: ph,
            attrs::token_id: id as u64,
            attrs::piece_score: *score as f64,
        };
        vocab_ids.push(e.root().expect("vocab entry root"));
        facts += e;
    }

    let name_h = blobs.put::<blobencodings::UTF8String, _>(source_name.to_string())?;
    let mut tags: Vec<Id> = vec![ty::UNIGRAM];
    if add_dummy_prefix {
        tags.push(flag::ADD_PREFIX_SPACE);
    }
    let tok = entity! { _ @
        metadata::tag*: tags.iter(),
        attrs::byte_fallback: byte_fallback,
        attrs::vocab*: vocab_ids.iter(),
    };
    let tok_id = tok.root().expect("tokenizer root");
    facts += tok;
    facts += entity! { ExclusiveId::force_ref(&tok_id) @ attrs::model_name: name_h };
    Ok(crate::format::rooted_at(tok_id, facts))
}

/// Read a UNIGRAM tokenizer back as the `(bytes, score, type)` table
/// [`crate::models::personaplex::spm::SpmTokenizer::from_pieces`] consumes,
/// ordered by `token_id`.
pub fn load_spm_pieces(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    tok_id: Id,
) -> Vec<(Vec<u8>, f32, u64)> {
    let mut rows: Vec<(u64, Vec<u8>, f32, u64)> = find!(
        (p, i: u64, sc: f64, t: Id),
        pattern!(tribles, [
            { tok_id @ attrs::vocab: _?entry },
            { _?entry @ attrs::piece: ?p, attrs::token_id: ?i,
                        attrs::piece_score: ?sc, metadata::tag: ?t }
        ])
    )
    .map(|(ph, i, sc, t)| {
        (
            i,
            read_piece(blobs, ph).into_bytes(),
            sc as f32,
            spm_tag_type(t),
        )
    })
    .collect();
    rows.sort_by_key(|r| r.0);
    rows.into_iter().map(|(_, b, s, t)| (b, s, t)).collect()
}

/// Does this tokenizer node carry the `add_prefix_space` flag?
pub fn has_add_prefix_space(tribles: &TribleSet, tok_id: Id) -> bool {
    node_tags(tribles, tok_id).contains(&flag::ADD_PREFIX_SPACE)
}
