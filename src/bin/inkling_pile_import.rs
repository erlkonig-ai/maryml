//! Convert Inkling's stacked NVFP4 experts into per-expert tensor blobs.
//!
//! Reads real checkpoint bytes and asserts the conversion against them, because
//! the arithmetic that matters — packed width, block count, where the global
//! scale sits — is exactly the arithmetic a synthetic fixture can agree with
//! while both are wrong.
//!
//! Read-only by default. A fourth positional pile path publishes the verified
//! leaves into that pile's native model collection when accompanied by an
//! existing durable signing key.
//!
//!   inkling_pile_import <checkpoint-dir> [tensor-base] [experts]
//!       [pile --signing-key <existing-key>]

use anyhow::{Context, Result};
use mary::models::inkling::load::Checkpoint;
use mary::models::inkling::pile::{attrs, expert_blob, experts_in_layers, layer_of, split_payload};
use triblespace::core::blob::encodings::tensor::TensorView;
use triblespace::core::metadata;
use triblespace::core::signing_key_file;
use triblespace::macros::entity;
use triblespace::prelude::*;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().context(
        "usage: inkling_pile_import <dir> [base] [experts] \
             [pile --signing-key <existing-key>]",
    )?;
    let mut positionals = Vec::new();
    let mut signing_key_path = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--signing-key" => {
                signing_key_path = Some(args.next().context("--signing-key needs a path")?)
            }
            flag if flag.starts_with('-') => anyhow::bail!("unknown argument {flag}"),
            positional => positionals.push(positional.to_string()),
        }
    }
    anyhow::ensure!(
        positionals.len() <= 3,
        "usage: inkling_pile_import <dir> [base] [experts] \
         [pile --signing-key <existing-key>]"
    );
    let base = positionals
        .first()
        .cloned()
        .unwrap_or_else(|| "model.llm.layers.10.mlp.experts.w13_weight".to_string());
    let count: usize = positionals
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(3);
    // Writing is opt-in and last: a run with no pile path converts, verifies
    // and reports, touching nothing.
    let pile_path = positionals.get(2).cloned();
    anyhow::ensure!(
        pile_path.is_some() || signing_key_path.is_none(),
        "--signing-key requires a destination pile"
    );

    // Held open across the whole run rather than per expert: opening a pile is
    // O(records), and this writes hundreds of them. Writer identity and
    // admission are settled before the checkpoint is opened or any expert is
    // converted.
    let mut writing = match &pile_path {
        None => None,
        Some(p) => {
            let signing_key_path = signing_key_path
                .as_deref()
                .context("--signing-key <existing-key> is required when writing a pile")?;
            let signing_key =
                signing_key_file::load_existing(std::path::Path::new(signing_key_path))
                    .with_context(|| format!("load existing signing key {signing_key_path:?}"))?;
            let path = std::path::Path::new(p);
            if !path.exists() {
                println!("creating a new pile at {p}");
                std::fs::File::create(path)?;
            }
            let mut pile = Pile::open(path).map_err(|e| anyhow::anyhow!("open pile: {e:?}"))?;
            pile.refresh()
                .map_err(|e| anyhow::anyhow!("load pile: {e:?}"))?;
            let _collection =
                mary::model_collection::model_collection_or_create(&mut pile, &signing_key)
                    .map_err(|e| anyhow::anyhow!("model collection writer: {e}"))?;
            Some((pile, signing_key, Fragment::empty()))
        }
    };

    let ck = Checkpoint::open(&dir).with_context(|| format!("opening {dir}"))?;
    println!("checkpoint {dir}");
    println!("tensor     {base}");

    let mut total = 0usize;
    for e in 0..count {
        let q = ck
            .expert_slice_packed(&base, e)
            .with_context(|| format!("slicing expert {e}"))?;
        let logical = q.cols * 2;

        let blob = expert_blob(&q).with_context(|| format!("expert {e} to blob"))?;
        let blob2 = expert_blob(&q)?;
        let bytes = blob.bytes.len();
        let handle = blob.get_handle();
        let view: TensorView = blob
            .try_from_blob()
            .context("decoding the blob just built")?;

        // The claims worth checking against real bytes rather than a fixture.
        anyhow::ensure!(
            view.dims() == [q.rows as u64, logical as u64],
            "expert {e}: dims {:?}, expected [{}, {}]",
            view.dims(),
            q.rows,
            logical
        );
        let (codes, scales, scale2) = split_payload(view.payload(), view.elems())?;
        anyhow::ensure!(
            codes == &q.codes[..],
            "expert {e}: codes differ after a round trip"
        );
        anyhow::ensure!(scales == &q.scales[..], "expert {e}: scales differ");
        anyhow::ensure!(scale2 == q.scale2, "expert {e}: global scale differs");

        if let Some((pile, _, change)) = writing.as_mut() {
            // put() returns the handle the facts then name, so the blob and the
            // fact about it cannot refer to different bytes.
            let handle = pile
                .put(blob2)
                .map_err(|err| anyhow::anyhow!("store expert {e}: {err:?}"))?;
            let name = pile
                .put::<blobencodings::UTF8String, _>(base.to_string())
                .map_err(|e| anyhow::anyhow!("store expert name: {e:?}"))?;
            let facts = entity! { _ @
                attrs::weight_nvfp4_2: handle,
                attrs::expert_index: e as i64,
                metadata::name: name,
                attrs::layer?: layer_of(&base),
            };
            *change += facts;
        }

        total += bytes;
        println!(
            "  expert {e:>3}  dims [{}, {}]  blob {:>10} B  scale2 {:+.6}  {:?}",
            q.rows, logical, bytes, scale2, handle
        );
    }

    if let Some((mut pile, signing_key, change)) = writing {
        if !change.facts().is_empty() {
            mary::model_collection::publish_model_fragment(&mut pile, &signing_key, change)
                .map_err(|e| anyhow::anyhow!("publish model collection: {e}"))?;
        }
        println!("wrote {count} expert(s) to {}", pile_path.clone().unwrap());

        // Query the collection we just wrote, by layer.
        let snapshot = mary::model_collection::snapshot_model_collection_local_latest(&mut pile)
            .map_err(|e| anyhow::anyhow!("snapshot model collection: {e}"))?;
        let held = experts_in_layers(snapshot.facts(), 0..=20);
        let all = experts_in_layers(snapshot.facts(), i64::MIN..=i64::MAX);
        println!(
            "query layers 0..=20 -> {} expert handle(s) of {} total, nothing materialised",
            held.len(),
            all.len()
        );
        for r in all.iter().take(3) {
            println!(
                "  layer {:>3}  expert {:>3}  {:?}",
                r.layer, r.expert, r.handle
            );
        }
        drop(snapshot);
        pile.close()
            .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    }

    println!(
        "{count} expert(s), {total} B total, {} B each on average",
        total / count.max(1)
    );
    Ok(())
}
