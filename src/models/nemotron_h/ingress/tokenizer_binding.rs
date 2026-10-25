//! Native Nemotron tokenizer binding over Mary's existing tokenizer graph.
//! The generic builder treats all added tokens as special; six source markers
//! are ordinary tokens. Keep their actual flags in each added token's identity,
//! not annotations on shared generic nodes. No HF JSON is retained at runtime.
//! This is tokenization, not a chat renderer or trusted-content framing policy.

use crate::{
    format::attrs,
    tokenizer::{self, attrs as tok, flag},
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use triblespace::{core::metadata, prelude::*};

pub const BINDING: &str = "nemotron-h-tokenizer";
pub const BASE: &str = "tokenizer-base";
pub const ADDED: &str = "added-token";

pub fn validate(bytes: &[u8]) -> Result<()> {
    let value: Value = serde_json::from_slice(bytes)?;
    ensure!(
        value["model"]["type"] == "BPE",
        "unsupported tokenizer model"
    );
    ensure!(
        value["padding"].is_null() && value["truncation"].is_null(),
        "unsupported tokenizer padding/truncation"
    );
    let post = &value["post_processor"];
    // This exact checkpoint's ByteLevel postprocessor neither inserts tokens
    // nor trims offsets. Pair type/sequence IDs are not rebuilt. Never silently
    // accept a template that would insert BOS/EOS or alter a single sequence.
    ensure!(
        post.is_null() || (post["type"] == "ByteLevel" && post["trim_offsets"] == false),
        "unsupported tokenizer postprocessor: only no-op single-sequence ByteLevel is supported"
    );
    for token in value["added_tokens"]
        .as_array()
        .context("missing added_tokens")?
    {
        for field in ["lstrip", "rstrip", "single_word"] {
            ensure!(
                token[field] == false,
                "unsupported added-token flag {field}"
            );
        }
        ensure!(
            token["special"].is_boolean() && token["normalized"].is_boolean(),
            "missing added-token flags"
        );
        let text = token["content"]
            .as_str()
            .context("missing added-token content")?;
        ensure!(
            value["model"]["vocab"][text] == token["id"],
            "added token must retain its vocabulary ID: {text}"
        );
    }
    Ok(())
}

pub fn save(bytes: &[u8], source: &str, blobs: &mut impl BlobStorePut) -> Result<Fragment> {
    validate(bytes)?;
    let mut value: Value = serde_json::from_slice(bytes)?;
    let added = value["added_tokens"].take();
    value["added_tokens"] = serde_json::json!([]);
    let base = tokenizer::save_tokenizer_json(&serde_json::to_vec(&value)?, BINDING, blobs)
        .map_err(|e| anyhow::anyhow!("save base tokenizer: {e}"))?;
    let mut members = entity! { attrs::kind: BASE, attrs::member*: base };
    for (index, token) in added
        .as_array()
        .expect("validated added_tokens")
        .iter()
        .enumerate()
    {
        let mut flags = Vec::new();
        if token["special"] == true {
            flags.push(flag::SPECIAL);
        }
        if token["normalized"] == true {
            flags.push(flag::NORMALIZED);
        }
        members += entity! {
            attrs::kind: ADDED, attrs::index: index as u64,
            tok::piece: token["content"].as_str().context("added content")?,
            tok::token_id: token["id"].as_u64().context("added id")?,
            metadata::tag*: flags,
        };
    }
    let mut binding = entity! { attrs::kind: BINDING, attrs::member*: members };
    let root = binding.root().expect("tokenizer binding root");
    binding += entity! { ExclusiveId::force_ref(&root) @ attrs::source: source };
    Ok(binding)
}

/// Select a base from the caller's frozen observation; IDs are opaque and
/// alternatives remain alternatives, not an enforced singleton.
pub fn bases(facts: &TribleSet, binding: Id) -> impl Iterator<Item = Id> + '_ {
    find!(base: Id, pattern!(facts, [
        { binding @ attrs::kind: BINDING, attrs::member: _?node },
        { _?node @ attrs::kind: BASE, attrs::member: ?base }
    ]))
}

#[cfg(feature = "tokenizer")]
pub fn rebuild(
    facts: &TribleSet,
    blobs: &impl BlobStoreGet,
    binding: Id,
    base: Id,
) -> Result<tokenizers::Tokenizer> {
    ensure!(
        exists!(pattern!(facts, [
            { binding @ attrs::kind: BINDING, attrs::member: _?node },
            { _?node @ attrs::kind: BASE, attrs::member: base }
        ])),
        "base is not a member of this tokenizer binding"
    );
    let mut tokenizer = tokenizer::build_tokenizer_with_added(facts, blobs, base, false)
        .map_err(|e| anyhow::anyhow!("rebuild tokenizer base: {e}"))?;
    let mut added = find!((order: u64, node: Id, piece: Inline<inlineencodings::Handle<blobencodings::UTF8String>>, id: u64), pattern!(facts, [
        { binding @ attrs::member: ?node },
        { ?node @ attrs::kind: ADDED, attrs::index: ?order, tok::piece: ?piece, tok::token_id: ?id }
    ])).collect::<Vec<_>>();
    added.sort_by_key(|(index, node, _, _)| (*index, *node));
    for (_, node, piece, id) in added {
        let content: anybytes::View<str> = blobs
            .get(piece)
            .map_err(|e| anyhow::anyhow!("read added token: {e}"))?;
        let content = content.to_string();
        ensure!(
            tokenizer.token_to_id(&content).map(u64::from) == Some(id),
            "added-token vocabulary ID mismatch"
        );
        let special = exists!(pattern!(facts, [{ node @ metadata::tag: flag::SPECIAL }]));
        let normalized = exists!(pattern!(facts, [{ node @ metadata::tag: flag::NORMALIZED }]));
        tokenizer
            .add_tokens(&[tokenizers::AddedToken::from(content, special).normalized(normalized)]);
    }
    Ok(tokenizer)
}
