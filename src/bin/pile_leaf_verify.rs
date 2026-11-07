//! Verify a converted pile against its source.
//!
//! `pile_leaf_migrate` checks each payload as it encodes it, which proves the
//! ENCODE step against bytes it already holds. This opens both piles cold and
//! answers the two questions a converter cannot ask itself:
//!
//!   1. is the output LOADABLE — does the typed read path find every tensor,
//!      with the same shape and the same bytes?
//!   2. did anything ELSE get lost — every fact the converter did not replace,
//!      and every blob it did not supersede?
//!
//! (2) is the one worth having. A converter that silently drops what it has no
//! schema for reports success either way; only a comparison of the full fact
//! sets catches a missing tokenizer.
//!
//! Leaves are compared BY ENTITY ID, not by name. The conversion preserves ids,
//! so identity is checkable directly rather than through a naming convention
//! that could itself be wrong.
//!
//! Both sides are read through their complete set of SIGNED COLLECTIONS, which
//! is the seam `mary::speak` selects from — a converted pile that verifies
//! through the deprecated branch pins and cannot be opened by the production
//! loader has passed the wrong exam. It also settles what "every fact" means:
//! `qwen3tts.pile` has two pins both named `main`, so a branch-side comparison
//! is between one fragment of the source and the whole of the destination.
//!
//! And both sides are CANONICALIZED before comparison — projected into the
//! current attribute spellings, then de-duplicated back down to one spelling
//! each. The source states its facts under pre-epoch literal ids and the
//! conversion writes canonical ones; comparing the raw sets would report every
//! surviving fact as both lost and gained. Canonicalizing both is what makes
//! "nothing was lost" a statement about facts rather than about spellings.
//!
//!   pile_leaf_verify <src.pile> <dst.pile>

use anyhow::{Context, Result};
use mary::format::attrs;
use mary::leaf;
use std::collections::HashSet;
use std::path::Path;
use triblespace::macros::{find, pattern};
use triblespace::prelude::*;

/// Entity ids of the leaves a conversion would convert: those carrying
/// `data` or `data_f16` alongside `shape`.
///
/// Leaves in other encodings (q4/q8) are deliberately NOT here — they are not
/// converted, so none of their facts may be dropped.
fn src_leaves_ids(tribles: &TribleSet) -> HashSet<Id> {
    let mut ids = HashSet::new();
    for (e,) in find!(
        (e: Id),
        pattern!(tribles, [{ ?e @ attrs::data: _?d, attrs::shape: _?s }])
    ) {
        ids.insert(e);
    }
    for (e,) in find!(
        (e: Id),
        pattern!(tribles, [{ ?e @ attrs::data_f16: _?d, attrs::shape: _?s }])
    ) {
        ids.insert(e);
    }
    ids
}

/// A collection handle as its bare hex, which is how one is compared by eye.
///
/// `Debug` on the handle prints its full generic path around the 32 bytes that
/// actually distinguish it, and this line exists to be read.
fn handle_hex(handle: &triblespace::core::collection::CollectionHandle) -> String {
    handle.raw.iter().map(|b| format!("{b:02x}")).collect()
}

/// One spelling per fact: project the canonical aliases in, then drop the
/// pre-epoch literals they cover.
///
/// Idempotent, and that is what makes it usable on both sides of a comparison
/// between a pre-epoch source and a post-epoch conversion.
fn canonical_facts(facts: &TribleSet) -> Result<TribleSet> {
    let projected = mary::model_collection::project_legacy_model_attributes(facts).facts;
    Ok(mary::persist::strip_projected_legacy_attributes(&projected)?.0)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src = args
        .next()
        .context("usage: pile_leaf_verify <src.pile> <dst.pile>")?;
    let dst = args
        .next()
        .context("usage: pile_leaf_verify <src.pile> <dst.pile>")?;
    let (src, dst) = (Path::new(&src), Path::new(&dst));

    let source = mary::persist::read_model_pile(src)?;
    let converted = mary::persist::read_model_pile(dst)?;

    // The collection is the thing a converted model pile is FOR. Its
    // policy-bearing descriptor identity must be the source's, byte for byte:
    // model piles resolve by content address.
    let (src_collection, dst_collection) =
        (source.collection.handle(), converted.collection.handle());
    anyhow::ensure!(
        src_collection == dst_collection,
        "collection identity moved: source {}, converted {}",
        handle_hex(&src_collection),
        handle_hex(&dst_collection)
    );
    println!("collection unchanged: {}", handle_hex(&src_collection));

    let (src_tribles, src_reader) = (canonical_facts(&source.facts)?, source.store);
    let (dst_tribles, dst_reader) = (canonical_facts(&converted.facts)?, converted.store);

    // ── 1. every non-replaced fact survived, unchanged ──────────────────────
    //
    // Filtered by (attribute AND entity), matching the converter. Filtering by
    // attribute alone — which this did until 2026-08-11 — excludes a dropped
    // `shape` from BOTH sides of the comparison, so the check cannot see it.
    // That is not a weak check, it is an anti-check: it validated the exact bug
    // it existed to catch, and personaplex_f16 (195 shapes, 194 leaves) passed
    // while missing one.
    //
    // A comparison that filters both sides by the same rule can only ever
    // confirm that rule was applied.
    //
    // Both spellings of the three, for the same reason the converter drops
    // both: a pre-epoch pile states them under the literal attribute ids and
    // the checkout projects the canonical aliases in beside them.
    let canonical = [attrs::data.id(), attrs::data_f16.id(), attrs::shape.id()];
    let mut replaced: HashSet<Id> = canonical.into_iter().collect();
    for alias in mary::model_collection::legacy_model_attribute_aliases() {
        if canonical.contains(&alias.canonical) {
            replaced.insert(alias.historical);
        }
    }
    let mut shape_attrs: HashSet<Id> = [attrs::shape.id()].into_iter().collect();
    for alias in mary::model_collection::legacy_model_attribute_aliases() {
        if alias.canonical == attrs::shape.id() {
            shape_attrs.insert(alias.historical);
        }
    }
    // The data attributes alone: a quantized leaf legitimately keeps its
    // `shape`, so only `data`/`data_f16` mark the old representation.
    let data_attrs: HashSet<Id> = replaced.difference(&shape_attrs).copied().collect();
    let converted_src: HashSet<Id> = src_leaves_ids(&src_tribles);
    let carried: TribleSet = src_tribles
        .iter()
        .filter(|t| !(replaced.contains(t.a()) && converted_src.contains(t.e())))
        .cloned()
        .collect();
    let dst_carried: TribleSet = dst_tribles
        .iter()
        .filter(|t| !(replaced.contains(t.a()) && converted_src.contains(t.e())))
        .cloned()
        .collect();

    let src_set: HashSet<_> = carried.iter().collect();
    let dst_set: HashSet<_> = dst_carried.iter().collect();
    let lost: Vec<_> = src_set.difference(&dst_set).take(5).collect();
    anyhow::ensure!(
        lost.is_empty(),
        "{} source facts did not survive the conversion",
        src_set.difference(&dst_set).count()
    );
    println!("facts       {} carried verbatim, none lost", src_set.len());
    anyhow::ensure!(
        !dst_tribles.iter().any(|t| data_attrs.contains(t.a())),
        "the converted pile still holds old-format leaf facts"
    );

    // ── 2. every leaf reads back, by entity id ──────────────────────────────
    let mut src_leaves: Vec<(Id, bool)> = Vec::new();
    for (e,) in find!(
        (e: Id),
        pattern!(&src_tribles, [{ ?e @ attrs::data: _?d, attrs::shape: _?s }])
    ) {
        src_leaves.push((e, false));
    }
    for (e,) in find!(
        (e: Id),
        pattern!(&src_tribles, [{ ?e @ attrs::data_f16: _?d, attrs::shape: _?s }])
    ) {
        src_leaves.push((e, true));
    }
    anyhow::ensure!(!src_leaves.is_empty(), "no leaves in source {src:?}");

    let typed = leaf::index_typed_all(&dst_tribles, &dst_reader);
    println!(
        "leaves      {} in source, {} typed in converted",
        src_leaves.len(),
        typed.len()
    );
    anyhow::ensure!(
        typed.len() == src_leaves.len(),
        "leaf count differs: {} source vs {} converted",
        src_leaves.len(),
        typed.len()
    );

    let (mut checked, mut bytes_checked, mut zero_copy) = (0usize, 0usize, 0usize);
    for (leaf_id, is_f16) in &src_leaves {
        let leaf_id = *leaf_id;
        let t = typed
            .get(&leaf_id)
            .ok_or_else(|| anyhow::anyhow!("{leaf_id}: no typed leaf in the converted pile"))?;

        let (dh_raw, sh) = if *is_f16 {
            find!(
                (d: Inline<inlineencodings::Handle<mary::f16enc::F16Array>>,
                 s: Inline<inlineencodings::Handle<mary::format::U64Array>>),
                pattern!(&src_tribles, [{ leaf_id @ attrs::data_f16: ?d, attrs::shape: ?s }])
            )
            .next()
            .map(|(d, s)| (d.raw, s))
            .context("f16 handles")?
        } else {
            find!(
                (d: Inline<inlineencodings::Handle<mary::format::F32Array>>,
                 s: Inline<inlineencodings::Handle<mary::format::U64Array>>),
                pattern!(&src_tribles, [{ leaf_id @ attrs::data: ?d, attrs::shape: ?s }])
            )
            .next()
            .map(|(d, s)| (d.raw, s))
            .context("f32 handles")?
        };
        let src_shape = mary::ingest::read_shape(&src_reader, sh);
        let handle: Inline<inlineencodings::Handle<blobencodings::RawBytes>> = Inline::new(dh_raw);
        let src_bytes: anybytes::Bytes = src_reader
            .get(handle)
            .map_err(|e| anyhow::anyhow!("{leaf_id}: source data blob: {e:?}"))?;

        anyhow::ensure!(
            t.shape() == src_shape,
            "{leaf_id}: shape differs: {:?} vs {:?}",
            src_shape,
            t.shape()
        );
        anyhow::ensure!(
            &t.payload()[..] == &src_bytes[..],
            "{leaf_id}: payload differs ({} vs {} bytes)",
            src_bytes.len(),
            t.payload().len()
        );

        if t.elem() == leaf::Elem::F32 {
            let v = t
                .view_f32()
                .ok_or_else(|| anyhow::anyhow!("{leaf_id}: f32 leaf served no zero-copy view"))?;
            anyhow::ensure!(
                v.len() == src_shape.iter().product::<usize>().max(1) || src_shape.is_empty(),
                "{leaf_id}: view length {} != shape {:?}",
                v.len(),
                src_shape
            );
            zero_copy += 1;
        }
        checked += 1;
        bytes_checked += src_bytes.len();
    }

    println!(
        "verified    {checked} tensors, {:.2} GiB, byte-identical",
        bytes_checked as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("zero-copy   {zero_copy} f32 leaves served a view over the mapping");
    Ok(())
}
