//! Convert a model pile's leaves from `{data|data_f16, shape}` to typed tensors.
//!
//! Pile to pile, into a NEW file. The source is only ever read.
//!
//! # Substitution, not reconstruction
//!
//! The converter does not rebuild the model graph. It COPIES the pile and
//! substitutes one representation for another:
//!
//!   - every fact is carried over verbatim except the three being replaced
//!     (`data`, `data_f16`, `shape`);
//!   - each leaf keeps its OWN entity id and gains one typed leaf fact;
//!   - every blob is copied except the superseded data and shape blobs.
//!
//! Substitution applies ONLY to the leaves actually converted. Everything else
//! — including leaves in encodings this tool has no path for, such as the q4/q8
//! quantized ones — passes through whole, `shape` included.
//!
//! This matters more than it sounds. A converter that rebuilds only what it
//! understands silently drops what it does not — and these piles carry more
//! than weights. `nomic_text` holds a 30,000-piece tokenizer graph hanging off
//! its own root; the rebuild-the-members version of this tool would have
//! reported a confident "112 leaves converted" and produced a pile with no
//! tokenizer in it. Copy-and-substitute cannot lose data it has no schema for,
//! which is exactly the property worth having when the pile knows things the
//! tool does not.
//!
//! Preserving ids also means nothing dangles and nothing downstream has to be
//! told a new address: module edges, model roots, `member` lists and tokenizer
//! references all still resolve. The weights are bit-identical, so the model an
//! id names is the same model; only how its bytes are framed has changed.
//!
//! It does mean a converted pile and a FRESH import of the same checkpoint hold
//! different ids for the same weights, because a leaf id is the content address
//! of its facts and the two forms are different facts. That is content
//! addressing behaving correctly rather than a defect, and preservation is the
//! right side to take: an id already written down in another pile, a commit, or
//! a note has to keep resolving, whereas an id nobody has yet is free. Two
//! piles converge again once both are on the typed form and reimported.
//!
//! # The COLLECTION is the source, and the destination
//!
//! Read and write both go through the signed model collection, not the
//! deprecated branch pins, because the two do not agree and the collection is
//! the one the production voice selects from.
//!
//! Reading the branch is wrong twice over. `qwen3tts.pile` carries TWO pins
//! both named `main` — `lookup_branch("main")` picks whichever it reaches
//! first, and the one it picks holds 1292 of the model's 1465 weight entities.
//! A branch-based conversion of that pile silently carries seven eighths of a
//! model and reports success. Its collection, meanwhile, holds all 1465: the
//! pins are fragments of a history, the collection is the model.
//!
//! Writing the branch alone is wrong in the other direction: `mary::speak`
//! resolves weights out of a `models` COLLECTION, so a converted pile
//! with only a branch fails to open with "no signed model collection" no matter
//! how correct its tensors are. The conversion was already right about the
//! bytes before this seam existed; it was one epoch behind the reader it has to
//! feed.
//!
//! ## Collection identity is preserved, deliberately
//!
//! The typed leaves land as a NEW COMMIT into the SAME named collection, under
//! the SAME descriptor the source publishes as. A typed collection is
//! identified by its policy-bearing descriptor, and the
//! conversion moves none of those, so the collection handle the destination
//! commits against is byte-identical to the source's. That is load-bearing:
//! model piles resolve by content address, and a collection identity that
//! moved would stop every already-persisted reference from resolving. The tool
//! asserts it rather than trusting it.
//!
//! The signer is explicit and must be admitted by the source collection's
//! authority. A fresh destination consequently needs the authority's own key;
//! a delegated key is also valid when its exact `ACTION_WRITE` proof has
//! already been installed in the destination. The converter checks this before
//! doing the large rewrite. An ephemeral key would still append a structurally
//! valid COMMIT, but that claim would be inert and the production loader would
//! correctly ignore the converted model.
//!
//! # One spelling, not two
//!
//! Pre-epoch piles state their attributes under literal ids that current
//! declarations no longer name, so every reader here projects the canonical
//! aliases in beside them. That projection is a read-side convenience, and
//! persisting its output would make it a defect: the destination would state
//! `kind` 1465 times AND its historical literal 1465 times. So the fact set is
//! canonicalized before it is written — the historical spelling is dropped
//! wherever its canonical twin is present, and refused (loudly) where it is
//! not. Nothing is lost; every current reader queries the canonical id.
//!
//! # What is verified
//!
//! Byte-identity is established twice, and neither half is assumed: each
//! payload is decoded back out of the blob just built and compared to the
//! source bytes (here), and the handle IS the hash of those bytes, which the
//! store checks on every read. `pile_leaf_verify` then re-checks the finished
//! pile cold, through the code path a real loader uses.
//!
//! The conversion never materialises a tensor — no `view::<[f32]>()`, no `Vec`,
//! no cast. The source blob's bytes go into the tensor payload as they are.
//!
//! Writing goes through `mary::leaf::put_leaf_as`, the same writer every
//! importer uses — so the rank dispatch, the length check and the read-back
//! comparison are shared with the write path rather than restated here, and a
//! converted leaf is byte-for-byte what a fresh import would have written.
//!
//!   pile_leaf_migrate <src.pile> <dst.pile> <signing-key>

use anyhow::{Context, Result};
use mary::format::attrs;
use mary::leaf;
use std::collections::HashSet;
use std::path::Path;
use triblespace::core::blob::Blob;
use triblespace::core::blob::encodings::UnknownBlob;
use triblespace::core::id::ExclusiveId;
use triblespace::core::repo::{BlobStoreList, SnapshotSource};
use triblespace::core::signing_key_file;
use triblespace::macros::{find, pattern};
use triblespace::prelude::*;

/// A collection handle as its bare hex, which is how one is compared by eye.
///
/// `Debug` on the handle prints its full generic path around the 32 bytes that
/// actually distinguish it, and this line exists to be read.
fn handle_hex(handle: &triblespace::core::collection::CollectionHandle) -> String {
    handle.raw.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src = args
        .next()
        .context("usage: pile_leaf_migrate <src.pile> <dst.pile> <signing-key>")?;
    let dst = args
        .next()
        .context("usage: pile_leaf_migrate <src.pile> <dst.pile> <signing-key>")?;
    let key = args
        .next()
        .context("usage: pile_leaf_migrate <src.pile> <dst.pile> <signing-key>")?;
    anyhow::ensure!(
        args.next().is_none(),
        "usage: pile_leaf_migrate <src.pile> <dst.pile> <signing-key>"
    );
    let signing_key = signing_key_file::load_existing(Path::new(&key))
        .with_context(|| format!("load existing signing key {key:?}"))?;
    let (src, dst) = (Path::new(&src), Path::new(&dst));
    migrate(src, dst, &signing_key)
}

fn migrate(src: &Path, dst: &Path, signing_key: &ed25519_dalek::SigningKey) -> Result<()> {
    anyhow::ensure!(
        src.canonicalize()? != dst.canonicalize().unwrap_or_else(|_| dst.to_path_buf()),
        "src and dst are the same pile file — refusing to write into the source"
    );

    let mary::persist::ModelPileSource {
        facts: tribles,
        store: reader,
        collection: expected,
    } = mary::persist::read_model_pile(src)?;
    eprintln!(
        "[migrate] {src:?}: native collection, {} facts",
        tribles.len()
    );

    // Every leaf, by its own entity id. `data` XOR `data_f16`, plus `shape`.
    let mut leaves: Vec<(Id, bool)> = Vec::new();
    for (e,) in find!(
        (e: Id),
        pattern!(&tribles, [{ ?e @ attrs::data: _?d, attrs::shape: _?s }])
    ) {
        leaves.push((e, false));
    }
    for (e,) in find!(
        (e: Id),
        pattern!(&tribles, [{ ?e @ attrs::data_f16: _?d, attrs::shape: _?s }])
    ) {
        leaves.push((e, true));
    }
    anyhow::ensure!(!leaves.is_empty(), "no tensor leaves in {src:?}");
    leaves.sort_by_key(|(e, _)| format!("{e}"));
    eprintln!("[migrate] {} leaves", leaves.len());

    if !dst.exists() {
        std::fs::File::create(dst).map_err(|e| anyhow::anyhow!("create {dst:?}: {e}"))?;
    }
    let mut pile = Pile::open(dst).map_err(|e| anyhow::anyhow!("open {dst:?}: {e:?}"))?;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("load {dst:?}: {e:?}; refusing to auto-truncate"))?;

    // Make the source descriptor available before asking the destination's
    // actual admission boundary. This is one tiny inert content-addressed
    // write, and lets an unauthorized key fail before tensor conversion or
    // bulk blob copying begins.
    let descriptor: Blob<blobencodings::SimpleArchive> = reader
        .get(expected.handle())
        .map_err(|error| anyhow::anyhow!("read source collection descriptor: {error:?}"))?;
    let descriptor_handle = pile
        .put::<blobencodings::SimpleArchive, _>(descriptor)
        .map_err(|error| anyhow::anyhow!("stage source collection descriptor: {error:?}"))?;
    anyhow::ensure!(
        descriptor_handle == expected.handle(),
        "source collection descriptor changed identity while copying"
    );
    // Admission is now asked of one frozen destination observation rather than
    // of the mutable pile. That matters here: the descriptor was just staged
    // into `pile`, so this snapshot is taken deliberately AFTER that put and
    // the answer is about the prefix that actually contains it.
    let destination = pile
        .snapshot()
        .map_err(|error| anyhow::anyhow!("freeze destination observation: {error}"))?;
    anyhow::ensure!(
        expected
            .writer_is_admitted(&destination, signing_key.verifying_key())
            .map_err(|error| anyhow::anyhow!("check destination writer admission: {error}"))?,
        "signing key is not admitted to the source model collection in the destination; use the authority key or install its ACTION_WRITE proof first"
    );

    // ── the substitution ────────────────────────────────────────────────────
    // Carry every fact EXCEPT the three being replaced ON THE LEAVES ACTUALLY
    // CONVERTED. The entity check is not a refinement, it is the difference
    // between substitution and quiet mutilation: a quantized leaf carries
    // `data_q4`/`data_q8` + `q_scales` + `shape`, so it is NOT in `leaves`
    // (no `data`/`data_f16`) and gets no typed leaf — but a blanket filter on
    // the attribute would still strip its `shape`, leaving a leaf whose
    // dimensions are simply gone.
    //
    // personaplex_q4.pile: 195 shape facts, 66 convertible leaves, 129 q4. A
    // blanket filter converts 66 and silently unshapes 129, and reports
    // success. Found by auditing this tool against the very failure it was
    // written to avoid — the docs above claimed copy-and-substitute "cannot
    // lose data it has no schema for", and for `shape` that was false.
    //
    // Both spellings of the three go. A pre-epoch pile states them under the
    // literal attribute ids, and the read projects the canonical aliases in
    // beside them; carrying the literal copy forward would leave the converted
    // pile still holding the representation this tool exists to retire, which
    // `pile_leaf_verify` rightly rejects.
    let converted: HashSet<Id> = leaves.iter().map(|(e, _)| *e).collect();
    let canonical = [attrs::data.id(), attrs::data_f16.id(), attrs::shape.id()];
    let mut replaced: HashSet<Id> = canonical.into_iter().collect();
    for alias in mary::model_collection::legacy_model_attribute_aliases() {
        if canonical.contains(&alias.canonical) {
            replaced.insert(alias.historical);
        }
    }
    let carried: TribleSet = tribles
        .iter()
        .filter(|t| !(replaced.contains(t.a()) && converted.contains(t.e())))
        .cloned()
        .collect();
    // ...and then say each surviving fact ONCE, under the name current readers
    // query, rather than persisting the read-side projection's shadow copy.
    let (facts, deduped) = mary::persist::strip_projected_legacy_attributes(&carried)?;
    // The carried facts are the source's, described by the source's own commits;
    // the converted leaves bring the description of the typed leaf attributes.
    let mut facts = Fragment::from(facts);
    eprintln!(
        "[migrate] carrying {} facts ({deduped} pre-epoch duplicate spellings dropped), \
         replacing {}",
        facts.len(),
        tribles.len() - carried.len()
    );

    // The blobs the old representation used, which the new one supersedes.
    let mut superseded: HashSet<[u8; 32]> = HashSet::new();
    let (mut f32n, mut f16n, mut total) = (0usize, 0usize, 0usize);

    for (leaf_id, is_f16) in &leaves {
        let leaf_id = *leaf_id;
        let (dh_raw, sh) = if *is_f16 {
            find!(
                (d: Inline<inlineencodings::Handle<mary::f16enc::F16Array>>,
                 s: Inline<inlineencodings::Handle<mary::format::U64Array>>),
                pattern!(&tribles, [{ leaf_id @ attrs::data_f16: ?d, attrs::shape: ?s }])
            )
            .next()
            .map(|(d, s)| (d.raw, s))
            .context("f16 leaf handles")?
        } else {
            find!(
                (d: Inline<inlineencodings::Handle<mary::format::F32Array>>,
                 s: Inline<inlineencodings::Handle<mary::format::U64Array>>),
                pattern!(&tribles, [{ leaf_id @ attrs::data: ?d, attrs::shape: ?s }])
            )
            .next()
            .map(|(d, s)| (d.raw, s))
            .context("f32 leaf handles")?
        };
        superseded.insert(dh_raw);
        superseded.insert(sh.raw);

        let dims: Vec<u64> = mary::ingest::read_shape(&reader, sh)
            .iter()
            .map(|&d| d as u64)
            .collect();
        let handle: Inline<inlineencodings::Handle<UnknownBlob>> = Inline::new(dh_raw);
        let bytes: anybytes::Bytes = reader
            .get(handle)
            .map_err(|e| anyhow::anyhow!("{leaf_id}: data blob: {e:?}"))?;
        total += bytes.len();

        let name = format!("{leaf_id}");
        let elem = if *is_f16 {
            f16n += 1;
            leaf::Elem::F16
        } else {
            f32n += 1;
            leaf::Elem::F32
        };
        // `put_leaf_as` is the same writer every importer uses, so the rank
        // dispatch and the round-trip check exist once rather than twice.
        let e = leaf::put_leaf_as(
            &mut pile,
            &ExclusiveId::force_ref(&leaf_id),
            elem,
            &dims,
            bytes,
            &name,
        )?;
        facts += e;

        if (f32n + f16n) % 200 == 0 {
            eprintln!("[migrate] {} leaves converted ...", f32n + f16n);
        }
    }

    // ── carry the remaining blobs ───────────────────────────────────────────
    // Everything the pile holds that the substitution did not supersede:
    // tokenizer pieces, names, anything this tool has no schema for. Copied by
    // bytes, so the handles come out identical and every carried fact still
    // resolves.
    let mut copied = 0usize;
    let mut copied_bytes = 0usize;
    for info in reader.blobs() {
        let info = info.map_err(|e| anyhow::anyhow!("list blobs: {e:?}"))?;
        if superseded.contains(&info.handle.raw) {
            continue;
        }
        let bytes: anybytes::Bytes = reader
            .get(info.handle)
            .map_err(|e| anyhow::anyhow!("copy blob: {e:?}"))?;
        copied_bytes += bytes.len();
        let out: Inline<inlineencodings::Handle<blobencodings::RawBytes>> = pile
            .put(Blob::<blobencodings::RawBytes>::new(bytes))
            .map_err(|e| anyhow::anyhow!("copy blob: {e:?}"))?;
        anyhow::ensure!(
            out.raw == info.handle.raw,
            "blob handle changed on copy — content addressing violated"
        );
        copied += 1;
    }
    eprintln!(
        "[migrate] carried {copied} other blobs ({:.2} GiB)",
        copied_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // ── publish, so the production loader can open what we just wrote ───────
    // Publish the converted graph through the same collection identity the
    // production loader selected from the source.
    let commit = pile
        .commit(
            expected,
            signing_key,
            facts,
        )
        .map_err(|e| anyhow::anyhow!("publish model collection commit: {e}"))?;
    // Not a formality: if the descriptor had moved, every already-persisted
    // reference to this model would stop resolving, and the failure would
    // surface as a pile that simply has no model in it rather than as an
    // error. Cheap to check, expensive to discover.
    anyhow::ensure!(
        commit.collection() == expected.handle(),
        "collection identity moved during conversion — refusing to claim the source's name"
    );
    eprintln!(
        "[migrate] published into the source's model collection, identity unchanged\n\
         [migrate]   source commits name {}\n\
         [migrate]   this commit names   {}",
        handle_hex(&expected.handle()),
        handle_hex(&commit.collection())
    );
    pile.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;

    eprintln!(
        "[migrate] {} leaves ({f32n} f32, {f16n} f16), {:.2} GiB of payload, all verified \
         byte-identical",
        f32n + f16n,
        total as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}
