//! Import Inkling's DENSE tensors — everything that is not a packed expert —
//! into a pile as typed tensor leaves, in their own dtype.
//!
//! The experts go through `inkling_pile_import`, which packs NVFP4. This is the
//! other half: attention projections, norms, embeddings, the router, the vision
//! tower — stored as `Tensor<BF16, RANK>` or `Tensor<F32, RANK>` according to
//! what the checkpoint actually holds, not widened on the way in.
//!
//! Not widening is the point. `Checkpoint::tensor` widens everything to f32 so
//! the runtime can compute with it, which is right for a runtime and wrong for
//! an importer: it would double the pile and then oblige the loader to narrow
//! the bytes again before handing them to a GPU that wanted BF16 in the first
//! place. Round-tripping BF16 through f32 is lossless — every BF16 is an f32
//! with a truncated mantissa — but the cost is real and avoidable.
//!
//! Element type and rank live in the attribute id, so a reader asking for BF16
//! matrices cannot be handed the f32 router bias by accident. Both dtypes share
//! ONE anchor, so this is the same `weight` attribute at different types rather
//! than a family of parallel ones.
//!
//! Read-only by default: with no pile path it converts, verifies and reports,
//! touching nothing. The dry run is worth having precisely because it still
//! does the encoding and the check — it answers "would this import cleanly"
//! without committing to it.
//!
//!   inkling_dense_import <checkpoint-dir> [pile --signing-key <existing-key>]
//!       [--limit N]

use anyhow::{Context, Result};
use mary::models::inkling::load::Checkpoint;
use mary::models::inkling::pile::{PileSource, attrs, layer_of};
use triblespace::core::blob::TryFromBlob;
use triblespace::core::blob::encodings::tensor::elements::{BF16, F32};
use triblespace::core::blob::encodings::tensor::{TensorView, tensor_blob};
use triblespace::core::metadata;
use triblespace::core::signing_key_file;
use triblespace::macros::entity;
use triblespace::prelude::*;

/// Build the typed blob, check it decodes back to exactly what went in, and —
/// only when writing — store it and record the facts.
///
/// The dispatch is over BOTH dtype and rank, which is what putting them in the
/// attribute id costs. It is paid here, once, at import.
macro_rules! dense_leaf {
    ($elem:ty, $rank:literal, $writing:expr, $dims:expr, $bytes:expr, $name:expr) => {{
        let dims: [u64; $rank] = $dims.as_slice().try_into().expect("rank checked by caller");
        let payload = anybytes::Bytes::from_source($bytes);
        let blob = tensor_blob::<$elem, $rank>(dims, payload.clone())
            .map_err(|e| anyhow::anyhow!("{}: {e}", $name))?;

        // Checked against the bytes that went in, not against a fixture.
        let view: TensorView = TryFromBlob::try_from_blob(blob.clone())
            .map_err(|e| anyhow::anyhow!("{}: decoding the blob just built: {e}", $name))?;
        anyhow::ensure!(
            view.dims() == dims,
            "{}: dims {:?} after a round trip, expected {:?}",
            $name,
            view.dims(),
            dims
        );
        anyhow::ensure!(
            &view.payload()[..] == &payload[..],
            "{}: payload differs after a round trip",
            $name
        );

        let len = blob.bytes.len();
        if let Some((pile, _, change)) = $writing.as_mut() {
            // put() returns the handle the facts then name, so the blob and the
            // fact about it cannot refer to different bytes.
            let handle = pile
                .put(blob)
                .map_err(|e| anyhow::anyhow!("{}: storing leaf: {e:?}", $name))?;
            let name_h = pile
                .put::<blobencodings::UTF8String, _>($name.to_string())
                .map_err(|e| anyhow::anyhow!("{}: storing name: {e:?}", $name))?;
            let facts = entity! { _ @
                attrs::weight::<$elem, $rank>(): handle,
                metadata::name: name_h,
                attrs::layer?: layer_of($name),
            };
            *change += facts;
        }
        len
    }};
}

macro_rules! by_rank {
    ($elem:ty, $writing:expr, $dims:expr, $bytes:expr, $name:expr) => {
        match $dims.len() {
            0 => dense_leaf!($elem, 0, $writing, $dims, $bytes, $name),
            1 => dense_leaf!($elem, 1, $writing, $dims, $bytes, $name),
            2 => dense_leaf!($elem, 2, $writing, $dims, $bytes, $name),
            3 => dense_leaf!($elem, 3, $writing, $dims, $bytes, $name),
            4 => dense_leaf!($elem, 4, $writing, $dims, $bytes, $name),
            5 => dense_leaf!($elem, 5, $writing, $dims, $bytes, $name),
            r => anyhow::bail!("{}: rank {r} is beyond the arms here", $name),
        }
    };
}
/// Re-read every dense tensor from the checkpoint and compare it to what the
/// pile holds under that name.
///
/// The import already checks that each blob decodes back to the bytes that went
/// in, and content addressing carries that to disk. What neither covers is
/// whether the FACT GRAPH points at the right blob for the right name — a
/// mis-wired attribute or a name attached to the wrong leaf produces a pile
/// that is internally consistent and still wrong. That failure mode is not
/// hypothetical: mary spent today unable to read any model pile because its
/// attribute ids had silently drifted, and every byte on disk was fine.
fn verify_pile(dir: &str, pile_path: &str, limit: usize) -> Result<()> {
    let ck = Checkpoint::open(dir).with_context(|| format!("opening {dir}"))?;
    let source = PileSource::open(std::path::Path::new(pile_path))?;
    let tribles = source.facts();
    let reader = source.reader();

    // name -> (dims, payload), read AS its type. One sweep per (dtype, rank).
    let mut index: std::collections::HashMap<String, (Vec<u64>, anybytes::Bytes)> =
        Default::default();
    macro_rules! sweep {
        ($elem:ty, $rank:literal) => {{
            for (n, h) in triblespace::macros::find!(
                (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
                 h: Inline<inlineencodings::Handle<
                     triblespace::core::blob::encodings::tensor::Tensor<$elem, $rank>>>),
                triblespace::macros::pattern!(tribles, [
                    { _?e @ metadata::name: ?n, attrs::weight::<$elem, $rank>(): ?h },
                ])
            ) {
                let name: anybytes::View<str> =
                    reader.get(n).map_err(|e| anyhow::anyhow!("name blob: {e:?}"))?;
                let blob: triblespace::core::blob::Blob<
                    triblespace::core::blob::encodings::tensor::Tensor<$elem, $rank>> =
                    reader.get(h).map_err(|e| anyhow::anyhow!("leaf blob: {e:?}"))?;
                let view: TensorView = TryFromBlob::try_from_blob(blob)
                    .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
                index.insert(name.to_string(), (view.dims().to_vec(), view.payload().clone()));
            }
        }};
    }
    sweep!(BF16, 0);
    sweep!(BF16, 1);
    sweep!(BF16, 2);
    sweep!(BF16, 3);
    sweep!(BF16, 4);
    sweep!(F32, 0);
    sweep!(F32, 1);
    sweep!(F32, 2);
    sweep!(F32, 3);
    sweep!(F32, 4);
    println!("pile       {} typed dense leaves", index.len());

    let names = ck.names();
    let dense: Vec<&String> = names.iter().filter(|n| !n.contains(".experts.")).collect();
    let (mut checked, mut bytes) = (0usize, 0usize);
    for name in dense.into_iter().take(limit) {
        let (dims, payload) = index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{name}: absent from the pile"))?;
        let raw = ck.tensor_raw(name)?;
        let want: Vec<u64> = raw.shape.iter().map(|&d| d as u64).collect();
        anyhow::ensure!(
            dims == &want,
            "{name}: dims {dims:?} != checkpoint {want:?}"
        );
        anyhow::ensure!(
            &payload[..] == &raw.bytes[..],
            "{name}: payload differs from the checkpoint"
        );
        checked += 1;
        bytes += raw.bytes.len();
        if checked % 200 == 0 {
            println!("  {checked} verified ...");
        }
    }
    println!(
        "verified   {checked} tensors, {:.2} GiB, byte-identical to the checkpoint",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().context(
        "usage: inkling_dense_import <checkpoint-dir> \
             [pile --signing-key <existing-key>] [--limit N]",
    )?;
    let mut pile_path: Option<String> = None;
    let mut limit = usize::MAX;
    let mut verify: Option<String> = None;
    let mut repair = false;
    let mut signing_key_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--limit" => limit = args.next().context("--limit needs a value")?.parse()?,
            "--signing-key" => {
                signing_key_path = Some(args.next().context("--signing-key needs a path")?)
            }
            "--verify" => verify = Some(args.next().context("--verify needs a pile")?),
            "--repair" => repair = true,
            other => pile_path = Some(other.to_string()),
        }
    }

    if let Some(v) = verify {
        return verify_pile(&dir, &v, limit);
    }

    // Names the collection already holds. Entities are content-derived, but a
    // scan still turns a re-run into a byte-identical no-op instead of adding a
    // second author's redundant commit, and an interrupted run into a resume.
    let mut present: std::collections::HashSet<String> = Default::default();
    let mut writing = match &pile_path {
        None => {
            println!("mode       read-only (no pile path given)");
            None
        }
        Some(p) => {
            let signing_key_path = signing_key_path
                .as_deref()
                .context("--signing-key <existing-key> is required when a pile path is supplied")?;
            let signing_key =
                signing_key_file::load_existing(std::path::Path::new(signing_key_path))
                    .with_context(|| format!("load existing signing key {signing_key_path:?}"))?;
            let path = std::path::Path::new(p);
            if !path.exists() {
                println!("creating a new pile at {p}");
                std::fs::File::create(path)?;
            }
            let mut store = Pile::open(path).map_err(|e| anyhow::anyhow!("open pile: {e:?}"))?;
            // A torn tail is what an interrupted append leaves; see the same
            // block in inkling_expert_import for why it is a flag and not an
            // open path. It matters more here, not less: this writes into the
            // pile the experts already live in.
            let before = std::fs::metadata(path)?.len();
            if repair {
                store
                    .amputate()
                    .map_err(|e| anyhow::anyhow!("amputating {p}: {e:?}"))?;
                let after = std::fs::metadata(path)?.len();
                println!(
                    "repaired   {} bytes discarded from a torn tail",
                    before - after
                );
            } else {
                store.refresh().map_err(|e| {
                    anyhow::anyhow!(
                        "{p}: {e:?}\n\n\
                         A partial record on the end is what an interrupted \
                         append leaves. Everything before that offset is \
                         intact. Re-run with --repair to truncate there and \
                         resume."
                    )
                })?;
            }
            // Fail before checkpoint indexing or the first tensor conversion
            // unless this durable signer can publish to the existing authority.
            let collection =
                mary::model_collection::model_collection_or_create(&mut store, &signing_key)
                    .map_err(|e| anyhow::anyhow!("model collection writer: {e}"))?;
            let observation = store
                .snapshot()
                .map_err(|e| anyhow::anyhow!("freeze model collection: {e}"))?;
            let snapshot =
                mary::model_collection::snapshot_model_collection_for(&observation, collection)
                    .map_err(|e| anyhow::anyhow!("model collection: {e}"))?;
            let facts =
                mary::model_collection::project_legacy_model_attributes(snapshot.facts()).facts;
            let reader = snapshot.store();
            // One sweep per (dtype, rank), matching the weight handle
            // but never fetching it — a name with no weight beside it
            // is not an imported tensor, and reading the leaves back to
            // find out what is missing would defeat the point.
            macro_rules! seen {
                        ($elem:ty, $rank:literal) => {{
                            for (n, _h) in triblespace::macros::find!(
                                (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
                                 h: Inline<inlineencodings::Handle<
                                     triblespace::core::blob::encodings::tensor::Tensor<$elem, $rank>>>),
                                triblespace::macros::pattern!(&facts, [
                                    { _?e @ metadata::name: ?n, attrs::weight::<$elem, $rank>(): ?h },
                                ])
                            ) {
                                let name: anybytes::View<str> = reader
                                    .get(n)
                                    .map_err(|e| anyhow::anyhow!("name: {e:?}"))?;
                                present.insert(name.to_string());
                            }
                        }};
                    }
            seen!(BF16, 0);
            seen!(BF16, 1);
            seen!(BF16, 2);
            seen!(BF16, 3);
            seen!(BF16, 4);
            seen!(F32, 0);
            seen!(F32, 1);
            seen!(F32, 2);
            seen!(F32, 3);
            seen!(F32, 4);
            println!("resuming   {} tensors already in the pile", present.len());
            drop(snapshot);
            Some((store, signing_key, Fragment::empty()))
        }
    };

    // Writer admission above is deliberately earlier than opening and
    // indexing the checkpoint: an unauthorized invocation must fail before it
    // does useful work, not after gigabytes have been converted.
    let ck = Checkpoint::open(&dir).with_context(|| format!("opening {dir}"))?;
    let names = ck.names();
    println!("checkpoint {dir}");
    println!("tensors    {} in the index", names.len());

    // A packed expert needs its sidecars and a different encoding entirely, so
    // it is skipped here rather than half-handled. The skip is REPORTED with a
    // count: a silent skip in an importer is how a model arrives missing a
    // third of itself and nothing says so.
    let (dense, experts): (Vec<&String>, Vec<&String>) =
        names.iter().partition(|n| !n.contains(".experts."));
    println!(
        "           {} dense, {} expert (skipped here — see inkling_pile_import)",
        dense.len(),
        experts.len()
    );

    let mut by_dtype: std::collections::BTreeMap<String, usize> = Default::default();
    let mut by_rank_count: std::collections::BTreeMap<usize, usize> = Default::default();
    let (mut done, mut total_bytes) = (0usize, 0usize);
    let (mut pending, mut commits, mut skipped) = (0usize, 0usize, 0usize);
    // A commit every FLUSH_EVERY tensors rather than one at the end. The
    // dense side is only ~15 GiB, but it shares a pile with 144 GiB of experts.
    // Publish bounded fragments so an interrupt loses at most one batch.
    const FLUSH_EVERY: usize = 64;

    for name in dense.into_iter().take(limit) {
        if present.contains(name.as_str()) {
            skipped += 1;
            continue;
        }
        let raw = ck
            .tensor_raw(name)
            .with_context(|| format!("reading {name}"))?;
        let dims: Vec<u64> = raw.shape.iter().map(|&d| d as u64).collect();
        *by_dtype.entry(raw.dtype.clone()).or_default() += 1;
        *by_rank_count.entry(dims.len()).or_default() += 1;

        let bytes = match raw.dtype.as_str() {
            "BF16" => by_rank!(BF16, writing, dims, raw.bytes, name),
            "F32" => by_rank!(F32, writing, dims, raw.bytes, name),
            other => {
                anyhow::bail!("{name} holds {other}; add an element type rather than widening it")
            }
        };

        total_bytes += bytes;
        done += 1;
        pending += 1;
        if let Some((pile, signing_key, change)) = writing.as_mut() {
            if pending >= FLUSH_EVERY {
                let batch = std::mem::replace(change, Fragment::empty());
                mary::model_collection::publish_model_fragment(pile, signing_key, batch)
                    .map_err(|e| anyhow::anyhow!("publish model collection: {e}"))?;
                commits += 1;
                pending = 0;
            }
        }
        if done % 100 == 0 {
            println!("  {done} tensors ...");
        }
    }

    println!("dtypes     {by_dtype:?}");
    println!("ranks      {by_rank_count:?}");
    println!(
        "encoded    {done} tensors, {:.2} GiB, each verified to decode back to \
         the bytes that went in",
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    println!("skipped    {skipped} tensors already present");
    if let Some((mut pile, signing_key, change)) = writing {
        if pending > 0 {
            mary::model_collection::publish_model_fragment(&mut pile, &signing_key, change)
                .map_err(|e| anyhow::anyhow!("publish model collection: {e}"))?;
            commits += 1;
        }
        pile.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;
        println!("committed  {commits} commits");
        println!("wrote      {}", pile_path.unwrap());
    }
    Ok(())
}
