//! READ-ONLY inspector: what model/tokenizer roots does a legacy pile hold?
//!
//! Scratch tool for the collection migration sweep. Opens the pile, freezes
//! legacy `main`, projects the canonical attribute aliases, then prints every
//! model root with its name/source/quantization, member count, and a sample of
//! member tensor-path prefixes so roots can be told apart by what they CONTAIN
//! rather than by how many members they happen to have.

use anyhow::{anyhow, Context, Result};
use mary::format::attrs;
use mary::model_collection::project_legacy_model_attributes;
use std::collections::{BTreeMap, BTreeSet};
use triblespace::macros::{find, pattern};
use triblespace::prelude::*;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: inspect_roots <pile>")?;
    let path = std::path::Path::new(&path);
    let mut pile = Pile::open(path).map_err(|e| anyhow!("open {path:?}: {e:?}"))?;
    let result = inspect(&mut pile, path);
    let close = pile.close().map_err(|error| anyhow!("close pile: {error}"));
    result?;
    close
}

fn inspect(pile: &mut Pile, path: &std::path::Path) -> Result<()> {
    let frozen =
        mary_model_migration::freeze_legacy_model_main(pile).context("freeze legacy main")?;
    let branch = frozen.branch;
    let head = frozen.head;
    let facts = frozen.facts;
    let reader = frozen.reader;

    println!("pile {}", path.display());
    println!("legacy main branch {branch}, head {head:?}");
    let projection = project_legacy_model_attributes(&facts);
    println!(
        "{} legacy facts, {} aliases would be added",
        facts.len(),
        projection.aliases_added
    );
    let facts = projection.facts;

    let read_ls = |h: Inline<inlineencodings::Handle<blobencodings::UTF8String>>| -> String {
        reader
            .get::<anybytes::View<str>, _>(h)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "<unreadable>".into())
    };

    // Model roots: anything carrying members, by either naming attribute.
    let named: BTreeSet<Id> = find!(
        (m: Id),
        pattern!(&facts, [{ ?m @ attrs::model_name: _?n, attrs::member: _?x }])
    )
    .map(|(m,)| m)
    .collect();
    let sourced: BTreeSet<Id> = find!(
        (m: Id),
        pattern!(&facts, [{ ?m @ attrs::source: _?s, attrs::member: _?x }])
    )
    .map(|(m,)| m)
    .collect();
    let roots: BTreeSet<Id> = named.union(&sourced).copied().collect();
    println!("\n{} model root(s):", roots.len());

    for root in &roots {
        let root = *root;
        let names: Vec<String> = find!(
            (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(&facts, [{ root @ attrs::model_name: ?n }])
        )
        .map(|(n,)| read_ls(n))
        .collect();
        let sources: Vec<String> = find!(
            (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(&facts, [{ root @ attrs::source: ?n }])
        )
        .map(|(n,)| read_ls(n))
        .collect();
        let quants: Vec<String> = find!(
            (q: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(&facts, [{ root @ attrs::quantization: ?q }])
        )
        .map(|(q,)| read_ls(q))
        .collect();
        let members: BTreeSet<Id> = find!(
            (m: Id),
            pattern!(&facts, [{ root @ attrs::member: ?m }])
        )
        .map(|(m,)| m)
        .collect();

        // Tensor paths of the members, bucketed by first dotted segment.
        let mut prefixes: BTreeMap<String, usize> = BTreeMap::new();
        let mut sample: Vec<String> = Vec::new();
        for member in &members {
            let member = *member;
            for (p,) in find!(
                (p: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(&facts, [{ member @ attrs::safetensor_path: ?p }])
            ) {
                let p = read_ls(p);
                let head = p.split('.').next().unwrap_or("").to_string();
                *prefixes.entry(head).or_default() += 1;
                if sample.len() < 4 {
                    sample.push(p);
                }
            }
        }

        println!("\n  root {root}");
        println!("    model_name   {names:?}");
        println!("    source       {sources:?}");
        println!("    quantization {quants:?}");
        println!("    members      {}", members.len());
        let mut top: Vec<_> = prefixes.iter().collect();
        top.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        let top: Vec<String> = top
            .iter()
            .take(12)
            .map(|(k, c)| format!("{k}({c})"))
            .collect();
        println!("    prefixes     {}", top.join(" "));
        println!("    sample       {sample:?}");
    }

    println!(
        "\n  attr ids: format::model_name={} tokenizer::model_name={}",
        attrs::model_name.id(),
        mary::tokenizer::attrs::model_name.id()
    );

    let real_toks: BTreeSet<Id> = mary::tokenizer::find_tokenizers(&facts).collect();
    println!(
        "\n{} REAL tokenizer root(s) (tagged with a tokenizer kind):",
        real_toks.len()
    );
    for t in &real_toks {
        let t = *t;
        let names: Vec<String> = find!(
            (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(&facts, [{ t @ mary::tokenizer::attrs::model_name: ?n }])
        )
        .map(|(n,)| read_ls(n))
        .collect();
        println!("  root {t}  name {names:?}");
    }

    // Tokenizer roots.
    let toks: BTreeSet<Id> = find!(
        (t: Id),
        pattern!(&facts, [{ ?t @ mary::tokenizer::attrs::model_name: _?n }])
    )
    .map(|(t,)| t)
    .collect();
    println!("\n{} tokenizer root(s):", toks.len());
    for t in toks {
        let names: Vec<String> = find!(
            (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(&facts, [{ t @ mary::tokenizer::attrs::model_name: ?n }])
        )
        .map(|(n,)| read_ls(n))
        .collect();
        println!("  root {t}  name {names:?}");
    }
    Ok(())
}
