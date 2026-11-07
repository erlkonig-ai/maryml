//! Import Inkling's packed experts for a RANGE OF LAYERS into a pile.
//!
//! The probe (`inkling_pile_import`) takes one tensor name and a count. This is
//! the deployment shape: a node imports the layers it holds. Splitting a model
//! across machines is a query — `experts_in_layers` — and this is the writing
//! half of that same idea, so the range is the argument rather than a tensor
//! name the caller has to know.
//!
//! It is also what fits. The full NVFP4 checkpoint is ~552 GB, of which the
//! experts are ~500 GB, so "import everything" is not an option on a single
//! machine and a layer range is not a convenience — it is the unit of work.
//!
//! One layer's experts are plain BF16 rather than NVFP4 (the checkpoint has 64
//! `w13_weight` but 63 `.scale`). That layer is REPORTED and skipped here
//! rather than guessed at: the packed path would read sidecars that do not
//! exist, and inventing them is how a model comes back as plausible noise.
//!
//! It commits PER STACKED MATRIX rather than once at the end, and that is a
//! correctness property rather than a progress bar. The single-commit shape
//! held the whole share in RAM and left the pile file empty until the last
//! expert was encoded — measured at 0 bytes after 71 GiB of experts. For a
//! 144 GiB share that is not a slow start; it is a run that cannot be
//! interrupted.
//!
//! A model-collection commit is appended only after a matrix's blobs, so an
//! interrupted run authorizes only whole matrices. The next run asks the
//! collection what it already holds and skips it. Resuming and re-running are
//! the same code path: a complete re-run writes nothing and leaves the file
//! byte-identical. A run killed mid-append leaves a partial record on the end,
//! which `--repair` truncates — see the block that does it for why that is a
//! flag and not the open path.
//!
//!   inkling_expert_import <ckpt-dir> <pile> --layers A-B [--experts N]
//!       --signing-key <existing-key> [--repair]
//!   inkling_expert_import <ckpt-dir> <pile> --layers A-B [--experts N]
//!       --verify

use anyhow::{Context, Result};
use mary::models::inkling::load::Checkpoint;
use mary::models::inkling::pile::{attrs, expert_blob, experts_in_layers, layer_of, split_payload};
use triblespace::core::blob::TryFromBlob;
use triblespace::core::blob::encodings::tensor::TensorView;
use triblespace::core::metadata;
use triblespace::core::signing_key_file;
use triblespace::macros::entity;
use triblespace::prelude::*;
/// Confirm a pile actually holds a node's whole share, byte for byte.
///
/// The import round-trips each expert as it encodes it, which proves the
/// packing. This proves POSSESSION: for every expert the layer range implies,
/// the pile has a leaf under that name and index, and its bytes match the
/// checkpoint.
///
/// The completeness half is what makes it worth running. A node holding 200 of
/// its 256 experts does not fail — it computes, and returns wrong tokens. So
/// the check is driven by what the CHECKPOINT says should be there, never by
/// what the pile happens to contain; asking the pile to enumerate itself would
/// make a missing expert invisible by construction.
fn verify_share(
    ck: &Checkpoint,
    pile_path: &str,
    lo: i64,
    hi: i64,
    experts_cap: usize,
) -> Result<()> {
    let path = std::path::Path::new(pile_path);
    let mut pile = Pile::open(path).map_err(|e| anyhow::anyhow!("open {path:?}: {e:?}"))?;
    let snapshot = mary::model_collection::snapshot_model_collection_local_latest(&mut pile)
        .map_err(|e| anyhow::anyhow!("{path:?}: model collection snapshot: {e}"))?;
    let facts = mary::model_collection::project_legacy_model_attributes(snapshot.facts()).facts;
    let (_, _, reader) = snapshot.into_parts();
    pile.close()
        .map_err(|e| anyhow::anyhow!("close {path:?}: {e:?}"))?;

    // (name, expert index) -> payload, read AS its type.
    let mut packed_ix: std::collections::HashMap<(String, i64), anybytes::Bytes> =
        Default::default();
    for (n, i, h) in triblespace::macros::find!(
        (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
         i: i64,
         h: Inline<inlineencodings::Handle<
             triblespace::core::blob::encodings::tensor::Tensor<
                 triblespace::core::blob::encodings::tensor::elements::NVFP4, 2>>>),
        triblespace::macros::pattern!(&facts, [
            { _?e @ metadata::name: ?n, attrs::expert_index: ?i, attrs::weight_nvfp4_2: ?h },
        ])
    ) {
        let name: anybytes::View<str> =
            reader.get(n).map_err(|e| anyhow::anyhow!("name: {e:?}"))?;
        let blob: triblespace::core::blob::Blob<
            triblespace::core::blob::encodings::tensor::Tensor<
                triblespace::core::blob::encodings::tensor::elements::NVFP4,
                2,
            >,
        > = reader.get(h).map_err(|e| anyhow::anyhow!("blob: {e:?}"))?;
        let view: TensorView =
            TryFromBlob::try_from_blob(blob).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        packed_ix.insert((name.to_string(), i), view.payload().clone());
    }
    let mut bf16_ix: std::collections::HashMap<(String, i64), anybytes::Bytes> = Default::default();
    for (n, i, h) in triblespace::macros::find!(
        (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
         i: i64,
         h: Inline<inlineencodings::Handle<
             triblespace::core::blob::encodings::tensor::Tensor<
                 triblespace::core::blob::encodings::tensor::elements::BF16, 2>>>),
        triblespace::macros::pattern!(&facts, [
            { _?e @ metadata::name: ?n, attrs::expert_index: ?i,
              attrs::weight::<triblespace::core::blob::encodings::tensor::elements::BF16, 2>(): ?h },
        ])
    ) {
        let name: anybytes::View<str> =
            reader.get(n).map_err(|e| anyhow::anyhow!("name: {e:?}"))?;
        let blob: triblespace::core::blob::Blob<
            triblespace::core::blob::encodings::tensor::Tensor<
                triblespace::core::blob::encodings::tensor::elements::BF16,
                2,
            >,
        > = reader.get(h).map_err(|e| anyhow::anyhow!("blob: {e:?}"))?;
        let view: TensorView =
            TryFromBlob::try_from_blob(blob).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        bf16_ix.insert((name.to_string(), i), view.payload().clone());
    }
    println!(
        "pile       {} NVFP4 + {} BF16 expert leaves",
        packed_ix.len(),
        bf16_ix.len()
    );

    let mut bases: Vec<String> = ck
        .names()
        .into_iter()
        .filter(|n| n.ends_with(".experts.w13_weight") || n.ends_with(".experts.w2_weight"))
        .filter(|n| matches!(layer_of(n), Some(l) if l >= lo && l <= hi))
        .collect();
    bases.sort();

    let (mut ok, mut bytes, mut ok_nvfp4) = (0usize, 0usize, 0usize);
    for base in &bases {
        let count = ck.expert_count(base)?.min(experts_cap);
        for e in 0..count {
            let key = (base.clone(), e as i64);
            if ck.is_nvfp4(base) {
                let have = packed_ix
                    .get(&key)
                    .ok_or_else(|| anyhow::anyhow!("{base}[{e}]: MISSING from the pile"))?;
                let q = ck.expert_slice_packed(base, e)?;
                let (codes, scales, scale2) = split_payload(have, q.rows * q.cols * 2)?;
                anyhow::ensure!(codes == &q.codes[..], "{base}[{e}]: codes differ");
                anyhow::ensure!(scales == &q.scales[..], "{base}[{e}]: scales differ");
                anyhow::ensure!(scale2 == q.scale2, "{base}[{e}]: scale2 differs");
                ok_nvfp4 += 1;
            } else {
                let have = bf16_ix
                    .get(&key)
                    .ok_or_else(|| anyhow::anyhow!("{base}[{e}]: MISSING from the pile"))?;
                let raw = ck.expert_slice_bf16(base, e)?;
                anyhow::ensure!(&have[..] == &raw.bytes[..], "{base}[{e}]: payload differs");
            }
            bytes += have_len(&packed_ix, &bf16_ix, &key);
            ok += 1;
            if ok % 200 == 0 {
                println!("  {ok} experts verified ...");
            }
        }
    }
    println!(
        "verified   {ok} experts across {} matrices, {:.2} GiB, byte-identical to the checkpoint",
        bases.len(),
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // The reader's half of the same question, and a genuinely different one.
    // Everything above asks whether the pile holds the right BYTES under the
    // right name; this asks whether the selector a node actually calls can FIND
    // them. `experts_in_layers` joins on `attrs::layer`, which no check above
    // touches — so an import that wrote every leaf perfectly and dropped the
    // layer facts would pass all of the above and hand a node zero experts.
    // Two faces of one interface; reading one and concluding about the pair is
    // how a green check comes to prove nothing.
    //
    // Against `ok`, not `ok_nvfp4`: the selector sweeps both element formats, so
    // layer 2's BF16 experts are inside its answer. Comparing against the packed
    // count alone would have passed while the selector silently dropped a whole
    // layer, which is the failure this check exists to catch.
    let refs = experts_in_layers(&facts, lo..=hi);
    let _ = ok_nvfp4;
    anyhow::ensure!(
        refs.len() == ok,
        "experts_in_layers({lo}..={hi}) selects {} of the {ok} experts just \
         verified — the leaves are on disk but the layer facts that reach them \
         are not",
        refs.len()
    );
    println!("selector   experts_in_layers({lo}..={hi}) reaches all {ok} experts");
    Ok(())
}

fn have_len(
    a: &std::collections::HashMap<(String, i64), anybytes::Bytes>,
    b: &std::collections::HashMap<(String, i64), anybytes::Bytes>,
    k: &(String, i64),
) -> usize {
    a.get(k)
        .map(|v| v.len())
        .or_else(|| b.get(k).map(|v| v.len()))
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().context(
        "usage: inkling_expert_import <ckpt-dir> <pile> --layers A-B \
             [--experts N] [--verify | --signing-key KEY [--repair]]",
    )?;
    let pile_path = args.next().context(
        "usage: inkling_expert_import <ckpt-dir> <pile> --layers A-B \
             [--experts N] [--verify | --signing-key KEY [--repair]]",
    )?;
    let mut layers: Option<(i64, i64)> = None;
    let mut experts_cap = usize::MAX;
    let mut verify = false;
    let mut repair = false;
    let mut signing_key_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--layers" => {
                let v = args.next().context("--layers needs A-B")?;
                let (a, b) = v.split_once('-').context("--layers wants A-B")?;
                layers = Some((a.parse()?, b.parse()?));
            }
            "--experts" => experts_cap = args.next().context("--experts needs N")?.parse()?,
            "--signing-key" => {
                signing_key_path = Some(args.next().context("--signing-key needs a path")?)
            }
            "--verify" => verify = true,
            "--repair" => repair = true,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let (lo, hi) = layers.context("--layers A-B is required — this imports a node's share")?;

    if verify {
        let ck = Checkpoint::open(&dir).with_context(|| format!("opening {dir}"))?;
        return verify_share(&ck, &pile_path, lo, hi, experts_cap);
    }

    // Import publication needs a durable identity. A fresh process-local key
    // can only create inert commits once this pile already has an authority,
    // so fail before reading or encoding any expert payload.
    let signing_key_path = signing_key_path.context(
        "--signing-key <existing-key> is required when importing; verification is read-only",
    )?;
    let signing_key = signing_key_file::load_existing(std::path::Path::new(&signing_key_path))
        .with_context(|| format!("load existing signing key {signing_key_path:?}"))?;

    let path = std::path::Path::new(&pile_path);
    if !path.exists() {
        println!("creating a new pile at {pile_path}");
        std::fs::File::create(path)?;
    }
    let mut store = Pile::open(path).map_err(|e| anyhow::anyhow!("open pile: {e:?}"))?;

    // ── the tail an interrupted run leaves ──────────────────────────────────
    // Committing per matrix makes an interrupted import resumable, but it does
    // not make the file self-healing, and the difference had to be MEASURED
    // rather than assumed: SIGKILL during a 144 GiB import lands inside a
    // `write_all` more often than not, and the pile that survives has a partial
    // record on the end. `Pile::open` does not validate; the refresh does, and
    // it refuses the file outright rather than guessing where the good data
    // stops.
    //
    // Everything before that offset is intact — a matrix's blobs are appended
    // BEFORE its collection commit, so a torn record cannot expose a partial
    // matrix as authoritative. `amputate` truncates there.
    // It stays behind a flag anyway: it is destructive, and an older binary
    // reading a newer record format would see the same "corruption" and eat
    // real data. Loud by default, surgical on request.
    let before = std::fs::metadata(path)?.len();
    if repair {
        store
            .amputate()
            .map_err(|e| anyhow::anyhow!("amputating {pile_path}: {e:?}"))?;
        let after = std::fs::metadata(path)?.len();
        if after < before {
            println!(
                "repaired   truncated a torn tail: {} bytes ({:.2} MiB) discarded",
                before - after,
                (before - after) as f64 / (1024.0 * 1024.0)
            );
        } else {
            println!("repaired   nothing to do, the pile was already whole");
        }
    } else {
        store.refresh().map_err(|e| {
            anyhow::anyhow!(
                "{pile_path}: {e:?}\n\n\
                 A partial record on the end is what an interrupted append \
                 leaves — the one the process was writing when it died. \
                 Everything before that offset is intact and no collection \
                 commit authorizes the partial matrix. Re-run with --repair to truncate there \
                 and resume; copy the file first if this is not a pile this \
                 importer wrote."
            )
        })?;
    }

    // ── what the pile already holds ─────────────────────────────────────────
    // Asked once, up front, and it is what makes a second run byte-identical.
    // Expert entities are content-derived, so their facts would deduplicate,
    // but a second admitted author could still append a redundant commit. One
    // query turns that into a skip — the same resumption path used after an
    // interruption. Driven by what the pile HAS, matched against what the
    // checkpoint says should be there; nothing is inferred from a file size.
    // Existing collections admit only the authority or a signer with a
    // resident ACTION_WRITE proof; an empty pile is founded under this durable
    // key. This happens before any tensor payload is read or blob appended.
    let collection =
        mary::model_collection::model_collection_or_create(&mut store, &signing_key)
            .map_err(|e| anyhow::anyhow!("model collection writer: {e}"))?;
    let mut present: std::collections::HashSet<(String, i64)> = Default::default();
    let observation = store
        .snapshot()
        .map_err(|e| anyhow::anyhow!("freeze model collection: {e}"))?;
    let snapshot = mary::model_collection::snapshot_model_collection_for(&observation, collection)
        .map_err(|e| anyhow::anyhow!("model collection: {e}"))?;
    let facts = mary::model_collection::project_legacy_model_attributes(snapshot.facts()).facts;
    let reader = snapshot.store();
    // Both element types, because this binary writes both. The weight
    // handle is matched but never fetched: a name and an index with no
    // weight beside them is not an imported expert, and reading 144
    // GiB back to find out what is missing would defeat the point.
    for (nh, i, _h) in triblespace::macros::find!(
                (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
                 i: i64,
                 h: Inline<inlineencodings::Handle<
                     triblespace::core::blob::encodings::tensor::Tensor<
                         triblespace::core::blob::encodings::tensor::elements::NVFP4, 2>>>),
                triblespace::macros::pattern!(&facts, [
                    { _?e @ metadata::name: ?n, attrs::expert_index: ?i,
                      attrs::weight_nvfp4_2: ?h },
                ])
    ) {
        let name: anybytes::View<str> =
            reader.get(nh).map_err(|e| anyhow::anyhow!("name: {e:?}"))?;
        present.insert((name.to_string(), i));
    }
    for (nh, i, _h) in triblespace::macros::find!(
                (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
                 i: i64,
                 h: Inline<inlineencodings::Handle<
                     triblespace::core::blob::encodings::tensor::Tensor<
                         triblespace::core::blob::encodings::tensor::elements::BF16, 2>>>),
                triblespace::macros::pattern!(&facts, [
                    { _?e @ metadata::name: ?n, attrs::expert_index: ?i,
                      attrs::weight::<triblespace::core::blob::encodings::tensor::elements::BF16, 2>(): ?h },
                ])
    ) {
        let name: anybytes::View<str> =
            reader.get(nh).map_err(|e| anyhow::anyhow!("name: {e:?}"))?;
        present.insert((name.to_string(), i));
    }
    println!("resuming   {} experts already in the pile", present.len());
    drop(snapshot);

    // Admission above precedes checkpoint indexing and tensor conversion, so
    // an unauthorized invocation cannot burn through a node's share before it
    // learns that none of its commits would be visible.
    let ck = Checkpoint::open(&dir).with_context(|| format!("opening {dir}"))?;

    // The stacked expert matrices in range. Sidecars are reached through the
    // index by the reader, so only the weights are enumerated here.
    let mut bases: Vec<String> = ck
        .names()
        .into_iter()
        .filter(|n| n.ends_with(".experts.w13_weight") || n.ends_with(".experts.w2_weight"))
        .filter(|n| matches!(layer_of(n), Some(l) if l >= lo && l <= hi))
        .collect();
    bases.sort();
    anyhow::ensure!(!bases.is_empty(), "no expert matrices in layers {lo}-{hi}");
    println!("checkpoint {dir}");
    println!("layers     {lo}..={hi}");
    println!("matrices   {}", bases.len());

    // A checkpoint holds BOTH kinds — Inkling's layer 2 experts are plain BF16
    // while the rest are NVFP4 — and the presence of a `.scale` sidecar is what
    // decides. Both are imported, each through its own path; neither is guessed
    // at, because the packed path on a BF16 stack would read sidecars that do
    // not exist and produce plausible noise.
    let (packed, dense): (Vec<&String>, Vec<&String>) = bases.iter().partition(|b| ck.is_nvfp4(b));
    println!("           {} NVFP4, {} BF16", packed.len(), dense.len());
    if !dense.is_empty() {
        println!("           BF16: {dense:?}");
    }

    let mut change = Fragment::empty();

    let (mut n, mut total) = (0usize, 0usize);
    let (mut pending, mut commits, mut skipped) = (0usize, 0usize, 0usize);

    // ── one commit per stacked matrix, published as it is built ─────────────
    // Blobs land first and the signed collection commit last. Interrupt
    // anywhere and the pile authorizes only whole matrices; the worst case is
    // orphan blobs from a matrix whose commit never landed.
    macro_rules! flush {
        () => {
            if pending > 0 {
                let batch = std::mem::replace(&mut change, Fragment::empty());
                mary::model_collection::publish_model_fragment(&mut store, &signing_key, batch)
                    .map_err(|e| anyhow::anyhow!("publish model collection: {e}"))?;
                commits += 1;
                pending = 0;
            }
        };
    }
    for base in packed {
        let layer = layer_of(base);
        // How many experts this matrix stacks — asked of the checkpoint rather
        // than assumed, so a model with a different expert count imports
        // correctly instead of silently importing a prefix.
        // Asked, not assumed — and not inferred from an error, which would
        // swallow a genuine read failure as "that was the last one".
        let count = ck.expert_count(base)?;
        let take = count.min(experts_cap);
        // This matrix's share of what is already imported, gathered once so
        // the per-expert test is a lookup rather than a scan.
        let done_here: std::collections::HashSet<i64> = present
            .iter()
            .filter(|(name, _)| name == base)
            .map(|(_, i)| *i)
            .collect();
        let (mut e, mut wrote) = (0usize, 0usize);
        while e < take {
            if done_here.contains(&(e as i64)) {
                skipped += 1;
                e += 1;
                continue;
            }
            let q = ck
                .expert_slice_packed(base, e)
                .with_context(|| format!("{base}[{e}]"))?;
            let blob = expert_blob(&q).with_context(|| format!("{base}[{e}] to blob"))?;

            // Same checks the probe makes, kept at scale: the round trip is
            // where a packing bug shows up, and it costs a decode. On a CLONE
            // of the blob rather than a second `expert_blob` — `Blob` is
            // refcounted bytes, so the check costs a decode instead of a
            // decode plus a rebuilt 7 MiB payload.
            let view: TensorView = TryFromBlob::try_from_blob(blob.clone())
                .map_err(|err| anyhow::anyhow!("{base}[{e}]: decode: {err}"))?;
            anyhow::ensure!(
                view.dims() == [q.rows as u64, (q.cols * 2) as u64],
                "{base}[{e}]: dims {:?}",
                view.dims()
            );
            let (codes, scales, scale2) = split_payload(view.payload(), view.elems())?;
            anyhow::ensure!(
                codes == &q.codes[..],
                "{base}[{e}]: codes differ after a round trip"
            );
            anyhow::ensure!(scales == &q.scales[..], "{base}[{e}]: scales differ");
            anyhow::ensure!(scale2 == q.scale2, "{base}[{e}]: global scale differs");

            let bytes = blob.bytes.len();
            let handle = store
                .put(blob)
                .map_err(|err| anyhow::anyhow!("{base}[{e}]: store expert: {err:?}"))?;
            let name_h = store
                .put::<blobencodings::UTF8String, _>(base.to_string())
                .map_err(|err| anyhow::anyhow!("{base}[{e}]: store name: {err:?}"))?;
            let facts = entity! { _ @
                attrs::weight_nvfp4_2: handle,
                attrs::expert_index: e as i64,
                metadata::name: name_h,
                attrs::layer?: layer,
            };
            change += facts;

            total += bytes;
            pending += 1;
            n += 1;
            wrote += 1;
            e += 1;
            if n % 200 == 0 {
                println!(
                    "  {n} experts, {:.1} GiB ...",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            }
        }
        flush!();
        println!(
            "  {base}: {wrote} written, {} already present",
            take - wrote
        );
    }

    // ── the BF16 stacks ─────────────────────────────────────────────────────
    // Same facts, different element type. `weight` is anchored, so
    // Tensor<BF16, 2> is that attribute at BF16 rather than a second attribute
    // beside it — and a reader asking for packed experts cannot be handed one
    // of these by accident.
    let mut bf16_n = 0usize;
    for base in dense {
        let layer = layer_of(base);
        let count = ck.expert_count(base)?;
        let take = count.min(experts_cap);
        let done_here: std::collections::HashSet<i64> = present
            .iter()
            .filter(|(name, _)| name == base)
            .map(|(_, i)| *i)
            .collect();
        let mut wrote = 0usize;
        for e in 0..take {
            if done_here.contains(&(e as i64)) {
                skipped += 1;
                continue;
            }
            let raw = ck
                .expert_slice_bf16(base, e)
                .with_context(|| format!("{base}[{e}]"))?;
            let dims: [u64; 2] = [raw.shape[0] as u64, raw.shape[1] as u64];
            let payload = anybytes::Bytes::from_source(raw.bytes);
            let blob = triblespace::core::blob::encodings::tensor::tensor_blob::<
                triblespace::core::blob::encodings::tensor::elements::BF16,
                2,
            >(dims, payload.clone())
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: {err}"))?;

            let view: TensorView = TryFromBlob::try_from_blob(blob.clone())
                .map_err(|err| anyhow::anyhow!("{base}[{e}]: decode: {err}"))?;
            anyhow::ensure!(
                view.dims() == dims && &view.payload()[..] == &payload[..],
                "{base}[{e}]: did not round-trip"
            );

            let bytes = blob.bytes.len();
            let handle = store
                .put(blob)
                .map_err(|err| anyhow::anyhow!("{base}[{e}]: store expert: {err:?}"))?;
            let name_h = store
                .put::<blobencodings::UTF8String, _>(base.to_string())
                .map_err(|err| anyhow::anyhow!("{base}[{e}]: store name: {err:?}"))?;
            let facts = entity! { _ @
                attrs::weight::<triblespace::core::blob::encodings::tensor::elements::BF16, 2>(): handle,
                attrs::expert_index: e as i64,
                metadata::name: name_h,
                attrs::layer?: layer,
            };
            change += facts;
            total += bytes;
            pending += 1;
            bf16_n += 1;
            wrote += 1;
            if bf16_n % 100 == 0 {
                println!(
                    "  {bf16_n} BF16 experts, {:.1} GiB ...",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            }
        }
        flush!();
        println!(
            "  {base}: {wrote} BF16 written, {} already present",
            take - wrote
        );
    }
    n += bf16_n;

    // Nothing is left staged: every matrix pushed as it finished, so this is
    // a close rather than the one write the whole run was building toward.
    debug_assert_eq!(pending, 0, "a matrix was built and never flushed");
    store.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;

    println!(
        "imported   {n} experts in {commits} commits, {:.2} GiB, each verified \
         to round-trip",
        total as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("skipped    {skipped} experts already present");
    println!("wrote      {pile_path}");
    Ok(())
}
