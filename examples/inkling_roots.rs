//! Inventory of a model collection's roots and their members, and the one
//! member-only root that names everything the Inkling loader indexes.
//!
//! The 2026-08 Inkling conversion wrote the checkpoint into its collection in
//! pieces, one root per piece, and the resident indexed the whole collection
//! without naming a root. The reader now wants one native root that names the
//! exact member set, so this prints what the pieces hold, then computes the
//! set the loader's own sweeps select (every named expert matrix with its
//! index and layer; every named dense tensor of a native element and rank),
//! reports duplicates, and prints the intrinsic id of the root
//! `mary root --members-file` would publish over it — without writing anything.
//!
//!     cargo run --release --example inkling_roots -- <pile> [--out members.txt]
//!
//! With `--out`, the member ids are written one per line, sorted, in the form
//! `mary root --members-file` reads.

use std::collections::{BTreeMap, BTreeSet};

use mary::format::attrs;
use triblespace::core::attribute::Attribute;
use triblespace::core::blob::encodings::tensor::elements::{BF16, F32, NVFP4};
use triblespace::core::blob::encodings::tensor::{Tensor, TensorElement};
use triblespace::core::id_hex;
use triblespace::core::inline::encodings::hash::Handle;
use triblespace::core::metadata;
use triblespace::prelude::*;

mod inkling {
    use triblespace::prelude::*;

    // The facts that make a member an expert matrix, with the anchors
    // `models::inkling::pile::attrs` declares; repeated here because that
    // module only exists on the CUDA lane and this inventory must not need it.
    attributes! {
        "A6ED6DBA4BE63E4E34F2787DA84AD860" as expert_index: inlineencodings::I256BE;
        "BCDDFBCFF89F67EE0B1E527C4872CED7" as layer: inlineencodings::I256BE;
    }
}

/// Anchor the weight attribute family derives from (`pile::attrs::WEIGHT_ANCHOR`).
const WEIGHT_ANCHOR: Id = id_hex!("0B51DA3E67216213871743E045590DBC");

/// The weight attribute for an element format and rank, as the loader derives it.
fn weight<T: TensorElement, const RANK: usize>() -> Attribute<Handle<Tensor<T, RANK>>> {
    Attribute::anchored(WEIGHT_ANCHOR)
}

type Name = Inline<inlineencodings::Handle<blobencodings::UTF8String>>;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let pile = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: inkling_roots <pile> [--out members.txt]"))?;
    let out = match (args.next(), args.next()) {
        (Some(flag), Some(path)) if flag == "--out" => Some(path),
        (None, _) => None,
        (Some(flag), _) => anyhow::bail!("unexpected argument {flag:?}"),
    };

    let snapshot = mary::model_collection::load_model_collection_local_latest(&pile)?;
    let facts = mary::model_collection::project_legacy_model_attributes(snapshot.facts()).facts;
    let long = |handle: Name| -> String {
        snapshot
            .store()
            .get::<anybytes::View<str>, _>(handle)
            .map(|value| value.to_string())
            .unwrap_or_else(|error| format!("<unreadable: {error}>"))
    };

    // ── what the pieces hold ──
    let mut by_root: BTreeMap<Id, BTreeSet<Id>> = BTreeMap::new();
    for (root, member) in find!(
        (root: Id, member: Id),
        pattern!(&facts, [{ ?root @ attrs::member: ?member }])
    ) {
        by_root.entry(root).or_default().insert(member);
    }
    let mut labels: BTreeMap<Id, Vec<String>> = BTreeMap::new();
    for (root, name) in find!(
        (root: Id, name: Name),
        pattern!(&facts, [{ ?root @ attrs::model_name: ?name }])
    ) {
        labels
            .entry(root)
            .or_default()
            .push(format!("name={:?}", long(name)));
    }
    for (root, source) in find!(
        (root: Id, source: Name),
        pattern!(&facts, [{ ?root @ attrs::source: ?source }])
    ) {
        labels
            .entry(root)
            .or_default()
            .push(format!("source={:?}", long(source)));
    }
    for (root, quantization) in find!(
        (root: Id, quantization: Name),
        pattern!(&facts, [{ ?root @ attrs::quantization: ?quantization }])
    ) {
        labels
            .entry(root)
            .or_default()
            .push(format!("quantization={:?}", long(quantization)));
    }
    let mut legacy_union: BTreeSet<Id> = BTreeSet::new();
    for (root, members) in &by_root {
        println!(
            "root {root:X}: {} members {}",
            members.len(),
            labels.get(root).map_or(String::new(), |l| l.join(" "))
        );
        legacy_union.extend(members.iter().copied());
    }
    println!(
        "{} roots with member edges; {} distinct members between them",
        by_root.len(),
        legacy_union.len()
    );

    // ── what the loader indexes: experts, then dense, by its own patterns ──
    let mut experts: BTreeMap<Id, (String, i64, i64)> = BTreeMap::new();
    macro_rules! sweep_experts {
        ($ty:ty) => {
            for (e, n, i, l) in find!(
                (e: Id, n: Name, i: i64, l: i64),
                pattern!(&facts, [{ ?e @ metadata::name: ?n, inkling::expert_index: ?i,
                    inkling::layer: ?l, weight::<$ty, 2>(): _?h }])
            ) {
                experts.insert(e, (long(n), i, l));
            }
        };
    }
    sweep_experts!(NVFP4);
    sweep_experts!(BF16);
    let mut dense: BTreeMap<Id, String> = BTreeMap::new();
    macro_rules! sweep_dense {
        ($ty:ty, $rank:literal) => {
            for (e, n) in find!(
                (e: Id, n: Name),
                pattern!(&facts, [{ ?e @ metadata::name: ?n, weight::<$ty, $rank>(): _?h }])
            ) {
                if !experts.contains_key(&e) {
                    dense.insert(e, long(n));
                }
            }
        };
    }
    sweep_dense!(BF16, 0);
    sweep_dense!(BF16, 1);
    sweep_dense!(BF16, 2);
    sweep_dense!(BF16, 3);
    sweep_dense!(BF16, 4);
    sweep_dense!(F32, 0);
    sweep_dense!(F32, 1);
    sweep_dense!(F32, 2);
    sweep_dense!(F32, 3);
    sweep_dense!(F32, 4);

    let mut by_key: BTreeMap<(String, i64), Vec<Id>> = BTreeMap::new();
    let mut layers: BTreeSet<i64> = BTreeSet::new();
    for (e, (name, index, layer)) in &experts {
        by_key.entry((name.clone(), *index)).or_default().push(*e);
        layers.insert(*layer);
    }
    let duplicate_experts: Vec<_> = by_key.iter().filter(|(_, ids)| ids.len() > 1).collect();
    let mut by_name: BTreeMap<&String, Vec<Id>> = BTreeMap::new();
    for (e, name) in &dense {
        by_name.entry(name).or_default().push(*e);
    }
    let duplicate_dense: Vec<_> = by_name.iter().filter(|(_, ids)| ids.len() > 1).collect();
    println!(
        "loader view: {} expert entities ({} distinct (name, index) keys, {} layers), {} dense entities ({} distinct names)",
        experts.len(),
        by_key.len(),
        layers.len(),
        dense.len(),
        by_name.len()
    );
    for ((name, index), ids) in duplicate_experts.iter().take(10) {
        println!("  duplicate expert {name} #{index}: {} entities", ids.len());
    }
    for (name, ids) in duplicate_dense.iter().take(10) {
        println!("  duplicate dense {name}: {} entities", ids.len());
    }
    if duplicate_experts.len() > 10 || duplicate_dense.len() > 10 {
        println!(
            "  ({} duplicate expert keys, {} duplicate dense names in all)",
            duplicate_experts.len(),
            duplicate_dense.len()
        );
    }
    let in_legacy = experts
        .keys()
        .chain(dense.keys())
        .filter(|e| legacy_union.contains(*e))
        .count();
    println!("  {in_legacy} of them are members of one of the pieces' roots");

    // ── which roots the reader would select today ──
    // Mirrors `models::inkling::version::version_heads`: a root is a node when
    // one of its members is something the loader indexes; a head is a node no
    // other node names as its parent. The reader wants exactly one head.
    let nodes: BTreeSet<Id> = by_root
        .iter()
        .filter(|(_, members)| {
            members
                .iter()
                .any(|m| experts.contains_key(m) || dense.contains_key(m))
        })
        .map(|(root, _)| *root)
        .collect();
    let mut parents: BTreeSet<Id> = BTreeSet::new();
    for (child, parent) in find!(
        (child: Id, parent: Id),
        pattern!(&facts, [{ ?child @ attrs::parent: ?parent }])
    ) {
        if nodes.contains(&child) {
            parents.insert(parent);
        }
    }
    let heads: Vec<Id> = nodes.difference(&parents).copied().collect();
    println!(
        "native heads today: {} ({})",
        heads.len(),
        heads
            .iter()
            .map(|h| format!("{h:X}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ── the member-only root over that set ──
    // `_ @ member*: ids` is exactly the identity core `mary root` publishes,
    // so this id is the one it will print for the same file.
    let members: BTreeSet<Id> = experts.keys().chain(dense.keys()).copied().collect();
    let fragment = entity! { _ @ attrs::member*: members.iter() };
    let root = fragment
        .root()
        .ok_or_else(|| anyhow::anyhow!("no root for an empty set"))?;
    println!(
        "member-only root over {} loader members: {root:X}",
        members.len()
    );

    if let Some(path) = out {
        let mut text = String::new();
        for member in &members {
            text.push_str(&format!("{member:X}\n"));
        }
        std::fs::write(&path, text)?;
        println!("wrote {} member ids to {path}", members.len());
    }
    Ok(())
}
