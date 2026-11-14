//! Ingest a checkpoint's JSON sidecars into a pile as FACTS, and gate the round
//! trip — in memory first, then through the pile.
//!
//! `INK_PILE` moves 159 GiB of weights out of the checkpoint and leaves the run
//! reading `config.json` out of the directory anyway. That is the last thread,
//! and it is a thin one: 40 KB decides the layer count, the head widths, the
//! expert count and the vocabulary, so a pile without it is a pile that cannot
//! be run from.
//!
//! What lands is facts, not files. One entity per JSON scalar (see
//! [`mary::jsonfacts`]), so `text_config.hidden_size` is a path through `member`
//! edges rather than a substring of a stored document — and the exactness of
//! that is checkable, because `serde_json`'s object map is sorted and
//! re-serialising is therefore canonical.
//!
//!   inkling_meta_gate <ckpt-dir> [pile --signing-key <existing-key>]
//!       [--mutate WHAT]
//!
//! `--mutate drop-key|int-to-float|reorder-array` corrupts the graph on the way
//! in so the gate can be watched failing. `int-to-float` is the one worth
//! running: it is what a naive encoding that stored every number as `f64` would
//! do on EVERY integer, silently.
//!
//! Build: `--features import` (needs nothing device-side).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use triblespace::core::signing_key_file;
use triblespace::prelude::*;

/// The sidecars a run actually needs. `config.json` is the load-bearing one;
/// the rest are here because "the pile is authoritative" is a claim about all
/// of them and a list that stops at the one currently read would make the claim
/// false the next time something reaches for a second file.
const DOCS: &[&str] = &[
    "config.json",
    "hf_quant_config.json",
    "processor_config.json",
];

/// Sidecars that are TEXT, stored as a document whose root is a JSON string.
/// One mechanism, not two: the template is a string, and a JSON string node
/// holds a string exactly.
const TEXT_DOCS: &[&str] = &["chat_template.jinja"];

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().map(PathBuf::from).context(
        "usage: inkling_meta_gate <ckpt-dir> \
             [pile --signing-key <existing-key>] [--mutate WHAT]",
    )?;
    let mut pile: Option<String> = None;
    let mut mutate: Option<String> = None;
    let mut signing_key_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--mutate" => mutate = Some(args.next().context("--mutate needs a name")?),
            "--signing-key" => {
                signing_key_path = Some(args.next().context("--signing-key needs a path")?)
            }
            other => {
                if pile.is_none() {
                    pile = Some(other.to_string());
                } else {
                    anyhow::bail!("unexpected argument {other:?}");
                }
            }
        }
    }
    anyhow::ensure!(
        pile.is_some() || signing_key_path.is_none(),
        "--signing-key requires a destination pile"
    );

    println!("checkpoint: {}", dir.display());

    // Read every sidecar as its canonical `Value`. The canonical form is the
    // comparison target throughout: a byte-for-byte comparison against the
    // FILE would fail on whitespace, which is not a fact about the config.
    let mut docs: Vec<(String, serde_json::Value)> = Vec::new();
    for name in DOCS {
        let path = dir.join(name);
        if !path.exists() {
            println!("  {name}: absent from this checkpoint, skipped");
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        let v: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {path:?}"))?;
        println!("  {name}: {} scalars", count_scalars(&v));
        docs.push((name.to_string(), v));
    }
    for name in TEXT_DOCS {
        let path = dir.join(name);
        if !path.exists() {
            println!("  {name}: absent from this checkpoint, skipped");
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        println!("  {name}: {} bytes of template", text.len());
        docs.push((name.to_string(), serde_json::Value::String(text)));
    }
    anyhow::ensure!(!docs.is_empty(), "no sidecars found in {dir:?}");

    // A mutation applies to what is INGESTED. The reference stays the file.
    let ingest: Vec<(String, serde_json::Value)> = match mutate.as_deref() {
        None => docs.clone(),
        Some(what) => {
            println!("MUTATION  : {what}");
            docs.iter()
                .map(|(n, v)| Ok((n.clone(), mutate_value(v, what)?)))
                .collect::<Result<Vec<_>>>()?
        }
    };

    // ── 1. in memory ────────────────────────────────────────────────────────
    let mut blobs = MemoryBlobStore::new();
    let mut facts = Fragment::empty();
    let t0 = std::time::Instant::now();
    for (name, v) in &ingest {
        mary::jsonfacts::save_document(name, v, &mut blobs, &mut facts)
            .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
    }
    println!(
        "graph     : {} facts in {:.2}s",
        facts.len(),
        t0.elapsed().as_secs_f64()
    );
    let reader = blobs
        .snapshot()
        .map_err(|e| anyhow::anyhow!("snapshot: {e:?}"))?;
    let bad = compare("memory", &docs, &facts, &reader)?;
    if bad == 0 {
        println!("PASS (in memory) — every sidecar round-trips exactly.");
    }

    // ── 2. through a real pile ──────────────────────────────────────────────
    let Some(pile) = pile else {
        if bad > 0 {
            println!("\nFAIL (in memory) — {bad} document(s) differ.");
            std::process::exit(1);
        }
        println!("(no pile argument — nothing was written.)");
        return Ok(());
    };
    let signing_key_path = signing_key_path
        .as_deref()
        .context("--signing-key <existing-key> is required when writing a pile")?;
    let signing_key = signing_key_file::load_existing(Path::new(signing_key_path))
        .with_context(|| format!("load existing signing key {signing_key_path:?}"))?;

    println!("\n=== ingesting into {pile} ===");
    anyhow::ensure!(
        mutate.is_none(),
        "refusing to write a mutated graph into {pile}: a pile is append-only, \
         so a deliberately wrong document could not be taken back. Run the \
         mutation without a pile argument — the in-memory stage is where it is \
         supposed to fail."
    );
    match mary::persist::ingest_json_documents(
        Path::new(&pile),
        &dir,
        DOCS,
        TEXT_DOCS,
        &signing_key,
    ) {
        Ok(n) => println!("committed : {n} facts"),
        Err(e) => {
            println!("ingest declined: {e}");
            println!("(continuing to the read-back gate — an existing graph still has to pass)");
        }
    }

    let (pfacts, preader) = mary::persist::pile_facts(Path::new(&pile))?;
    let names = mary::jsonfacts::documents(&pfacts, &preader);
    println!(
        "pile has  : {:?}",
        names.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
    );
    let bad_pile = compare("pile", &docs, &pfacts, &preader)?;

    // The claim that matters: the runtime's own config parser is satisfied by
    // what came out of the pile. Round-tripping the JSON is necessary and not
    // sufficient — `InklingConfig` reads a specific shape, and a document that
    // is byte-identical but reached under the wrong name is still a run that
    // cannot start.
    let cfg_json = mary::jsonfacts::load_document(&pfacts, &preader, "config.json")
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let cfg = mary::models::inkling::config::InklingConfig::from_json(&cfg_json.to_string())
        .context("the pile's config.json does not parse as an InklingConfig")?;
    println!(
        "parsed    : {} layers, hidden {}, {} experts, vocab {} (from the PILE)",
        cfg.text_config.num_hidden_layers,
        cfg.text_config.hidden_size,
        cfg.text_config.n_routed_experts,
        cfg.text_config.vocab_size
    );

    if bad > 0 || bad_pile > 0 {
        println!("\nFAIL — {bad} in-memory and {bad_pile} on-disk document(s) differ.");
        std::process::exit(1);
    }
    println!("\nPASS (on disk) — the checkpoint's JSON sidecars are no longer needed.");
    Ok(())
}

/// Canonical JSON in, canonical JSON out, per document.
fn compare(
    label: &str,
    want: &[(String, serde_json::Value)],
    facts: &TribleSet,
    blobs: &impl BlobStoreGet,
) -> Result<usize> {
    let mut bad = 0usize;
    for (name, v) in want {
        let got = match mary::jsonfacts::load_document(facts, blobs, name) {
            Ok(g) => g,
            Err(e) => {
                println!("  [{label}] {name}: {e}");
                bad += 1;
                continue;
            }
        };
        let (a, b) = (serde_json::to_string(v)?, serde_json::to_string(&got)?);
        if a == b {
            println!(
                "  [{label}] {name}: identical ({} bytes canonical)",
                a.len()
            );
        } else {
            println!("  [{label}] {name}: DIFFERS");
            println!("      first divergence at byte {}", first_diff(&a, &b));
            for (x, y) in [(&a, "file"), (&b, "graph")] {
                println!("      {y:<5} {}", &x[..x.len().min(200)]);
            }
            bad += 1;
        }
    }
    Ok(bad)
}

fn first_diff(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

fn count_scalars(v: &serde_json::Value) -> usize {
    match v {
        serde_json::Value::Array(a) => a.iter().map(count_scalars).sum(),
        serde_json::Value::Object(o) => o.values().map(count_scalars).sum(),
        _ => 1,
    }
}

/// Corrupt a document the way a real encoding bug would.
fn mutate_value(v: &serde_json::Value, what: &str) -> Result<serde_json::Value> {
    use serde_json::Value;
    Ok(match what {
        // What storing every number as `f64` does: 4096 comes back 4096.0.
        "int-to-float" => map_numbers(v),
        // A dropped object member.
        "drop-key" => match v {
            Value::Object(o) => {
                let mut o = o.clone();
                if let Some(k) = o.keys().next().cloned() {
                    o.remove(&k);
                }
                Value::Object(o)
            }
            other => other.clone(),
        },
        // Array order lost — the failure mode `load_json`'s index sort exists
        // to prevent, and `local_layer_ids` is an array whose ORDER is data.
        "reorder-array" => reverse_arrays(v),
        other => anyhow::bail!("unknown --mutate {other:?} (int-to-float|drop-key|reorder-array)"),
    })
}

fn map_numbers(v: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Value::Number(n) => match n.as_i64() {
            Some(i) => Value::Number(serde_json::Number::from_f64(i as f64 + 0.5).unwrap()),
            None => v.clone(),
        },
        Value::Array(a) => Value::Array(a.iter().map(map_numbers).collect()),
        Value::Object(o) => {
            Value::Object(o.iter().map(|(k, x)| (k.clone(), map_numbers(x))).collect())
        }
        other => other.clone(),
    }
}

fn reverse_arrays(v: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Value::Array(a) => {
            let mut a: Vec<Value> = a.iter().map(reverse_arrays).collect();
            a.reverse();
            Value::Array(a)
        }
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, x)| (k.clone(), reverse_arrays(x)))
                .collect(),
        ),
        other => other.clone(),
    }
}
