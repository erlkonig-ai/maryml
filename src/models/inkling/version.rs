//! The version graph: a learned model as a root in the model collection,
//! with a parent, and what it takes to write one and to find the heads.
//! Backend-free, so a Mac or a Pi build reads the graph a Spark wrote.

use anyhow::{Context, Result};
use triblespace::core::collection::CollectionHandle;
use triblespace::prelude::Id;

use super::load::PackedExpert;
pub use super::resident::{Persisted, VersionRecipe};

/// One whole expert, every rank's cut joined, as the pile would store it.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedExpert {
    pub name: String,
    pub layer: i64,
    pub expert: i64,
    pub packed: PackedExpert,
}

// ── the learned VERSION: a root in the model graph, with a parent ──────────
//
// A learned model is a new ROOT in the same collection as the model it grew
// from: the parent's `member` edges, except that the experts that moved are
// new leaves, plus a `parent` edge to the root it was learned from and the
// recipe that took it there. Nothing about the parent changes, and the leaves
// that did not move are the same entities, so a version costs only the
// experts it moved. Two versions with one parent are a branch. The versions no
// other version names as `parent` are the heads, and the loader takes the
// one head as the model when nothing names a root. New roots derive their ids
// from their member sets for idempotence; existing root ids are opaque. An
// unchanged member set keeps the existing parent without deriving its id again.
// A parent is the training BASE, not the previous save: repeated snapshots
// from one resident training run are siblings, not a chronological chain.
//
// JP, 2026-09-03 17:35Z: not a collection per snapshot -- a DAG in one graph,
// so a model can branch off a model, and the experiment's metadata and its
// explanation sit on the graph where a later analysis can query them.

pub mod attrs {
    use triblespace::prelude::blobencodings::UTF8String;
    use triblespace::prelude::inlineencodings::{F64, Handle, U256BE};
    use triblespace::prelude::*;

    pub use crate::format::attrs::parent;

    attributes! {
        /// The learner's step size on the routed experts' codes.
        "EA48C51D05FC180A5855106F0C0CCCAE" as learn_lr: F64;
        /// How hard her own rows were held to the distribution that said
        /// them (the anchor's weight). Absent when unanchored.
        "8BF7CAB60E71E7186955ACDAF6EF3867" as learn_anchor: F64;
        /// Where the stochastic rounding's step counter started.
        "E8C7ED409F235DA6A787B0B094269DD8" as learn_seed: U256BE;
        /// Scored passes learned from, parent to this version.
        "361889279FA26F7AD22187DF37DC7EC7" as learn_steps: U256BE;
        /// What was learned from, in words: a corpus and its lines, or a
        /// span of the archive.
        "9F65DE719752C3AB83F709B55567B247" as learned_span: Handle<UTF8String>;
        /// Why this version exists, in its author's words.
        "365C41C24AB9FF7DF2C78105DB1FDE97" as explanation: Handle<UTF8String>;
        /// The code that learned it: a git revision.
        "6CEEC82A5A1A1EC65A4609C025E42767" as code_revision: Handle<UTF8String>;
    }
}

/// A version assembled from learned experts, before or after it is committed.
pub struct LearnedVersion {
    /// The ordinary collection descriptor in which this version is published.
    pub collection: CollectionHandle,
    /// The version root.
    pub root: Id,
    /// The root it was learned from.
    pub parent: Id,
    /// A label: the model's name and the moment.
    pub name: String,
    /// The facts to ADD: new leaves, the version root and its annotations.
    /// Never the parent's facts, which the
    /// collection already holds.
    pub facts: triblespace::core::trible::TribleSet,
    /// Leaves whose bytes moved and were replaced by new leaf entities.
    pub replaced: usize,
    /// Members of the version root.
    pub members: usize,
}

/// Native Inkling roots not named as a parent by another native Inkling root.
/// Eligibility is the typed member shape consumed by `PileSource`, not a
/// model label or the shared `member`/`parent` vocabulary alone. An unrelated
/// model's lineage cannot select or suppress an Inkling candidate.
pub fn version_heads(facts: &triblespace::core::trible::TribleSet) -> Vec<Id> {
    use super::pile;
    use std::collections::HashSet;
    use triblespace::core::blob::encodings::tensor::elements::{BF16, F32};
    use triblespace::core::metadata;
    use triblespace::prelude::*;

    let mut nodes: HashSet<Id> = find!(
        (root: Id, expert: i64, layer: i64),
        pattern!(facts, [
            { ?root @ crate::format::attrs::member: _?member },
            { _?member @ metadata::name: _?name,
              pile::attrs::expert_index: ?expert, pile::attrs::layer: ?layer,
              pile::attrs::weight_nvfp4_2: _?weight },
        ])
    )
    .map(|(root, _, _)| root)
    .collect();
    // These are exactly the native dense types/ranks the loader sweeps.
    // BF16 experts are also covered by the BF16 matrix shape.
    macro_rules! native_dense_roots {
        ($ty:ty, $rank:literal) => {
            nodes.extend(find!(
                (root: Id),
                pattern!(facts, [
                    { ?root @ crate::format::attrs::member: _?member },
                    { _?member @ metadata::name: _?name,
                      pile::attrs::weight::<$ty, $rank>(): _?weight },
                ])
            ).map(|(root,)| root));
        };
    }
    native_dense_roots!(BF16, 0);
    native_dense_roots!(BF16, 1);
    native_dense_roots!(BF16, 2);
    native_dense_roots!(BF16, 3);
    native_dense_roots!(BF16, 4);
    native_dense_roots!(F32, 0);
    native_dense_roots!(F32, 1);
    native_dense_roots!(F32, 2);
    native_dense_roots!(F32, 3);
    native_dense_roots!(F32, 4);
    let parents: HashSet<Id> = find!(
        (version: Id, parent: Id),
        and!(
            (&nodes).has(version),
            pattern!(facts, [{ ?version @ attrs::parent: ?parent }])
        )
    )
    .map(|(_, parent)| parent)
    .collect();
    let mut heads: Vec<Id> = nodes.difference(&parents).copied().collect();
    heads.sort();
    heads
}

/// The moment, as `YYYYMMDDTHHMMSSZ`, from the system clock with no crate.
pub fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (rem / 3_600, rem % 3_600 / 60, rem % 60);
    // Civil date from days since the epoch (Hinnant).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(mo <= 2);
    format!("{y:04}{mo:02}{d:02}T{h:02}{m:02}{s:02}Z")
}

/// Assemble a version from the learned experts and the explicitly selected
/// collection and training base. Writes new blobs, but commits no record.
pub fn learned_version(
    pile: &mut triblespace::prelude::Pile,
    collection: CollectionHandle,
    learned: &[LearnedExpert],
    parent: Id,
    recipe: &VersionRecipe,
) -> Result<LearnedVersion> {
    use super::pile::attrs as ink;
    use crate::format::attrs as model;
    use std::collections::{BTreeSet, HashMap};
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::core::metadata;
    use triblespace::core::trible::TribleSet;
    use triblespace::macros::{entity, find, pattern};
    use triblespace::prelude::*;

    let store = pile.snapshot().context("freeze the model collection")?;
    let selected_collection = crate::model_collection::ModelCollection::open(&store, collection)
        .context("open the selected model collection")?;
    let snapshot =
        crate::model_collection::snapshot_model_collection_for(&store, selected_collection)
            .context("read the selected model collection")?;
    let facts = crate::model_collection::project_legacy_model_attributes(snapshot.facts()).facts;
    let (_, _, reader) = snapshot.into_parts();
    let mut added = TribleSet::new();

    // Labels come from the selected base, not from unrelated roots in the
    // collection. Labels do not participate in the version's identity.
    let names: BTreeSet<Inline<Handle<blobencodings::UTF8String>>> = find!(
        (mn: Inline<Handle<blobencodings::UTF8String>>),
        pattern!(&facts, [{ (parent) @ model::model_name: ?mn }])
    )
    .map(|(mn,)| mn)
    .collect();
    let model_name = match names.len() {
        1 => names.into_iter().next(),
        _ => None,
    };
    let label_of = |what: &str| -> String {
        match &model_name {
            Some(h) => {
                let name: Result<anybytes::View<str>, _> = reader.get(*h);
                name.map(|s| format!("{} {what}", &*s))
                    .unwrap_or_else(|_| what.to_string())
            }
            None => what.to_string(),
        }
    };

    // 1. The parent, and its members.
    let parent_members: Vec<Id> =
        find!((m: Id), pattern!(&facts, [{ (parent) @ model::member: ?m }]))
            .map(|(m,)| m)
            .collect();
    anyhow::ensure!(
        !parent_members.is_empty(),
        "parent {parent:X} has no members"
    );
    let parent_set: BTreeSet<Id> = parent_members.iter().copied().collect();

    // 2. New leaves, and the parent's leaves they replace.
    let mut subst: HashMap<Id, Id> = HashMap::new();
    for x in learned {
        let blob = super::pile::expert_blob(&x.packed)
            .with_context(|| format!("{}[{}] as a pile leaf", x.name, x.expert))?;
        let handle = pile
            .put(blob)
            .map_err(|e| anyhow::anyhow!("store {}[{}]: {e:?}", x.name, x.expert))?;
        let name_h = pile
            .put::<blobencodings::UTF8String, _>(x.name.clone())
            .map_err(|e| anyhow::anyhow!("store the name of {}: {e:?}", x.name))?;
        let old: Vec<Id> = find!(
            (e: Id),
            pattern!(&facts, [{ ?e @ metadata::name: (name_h), ink::expert_index: (x.expert), ink::layer: (x.layer) }])
        )
        .map(|(e,)| e)
        .filter(|e| parent_set.contains(e))
        .collect();
        anyhow::ensure!(
            old.len() == 1,
            "{}[{}] (layer {}) names {} leaves among the parent's members, not one",
            x.name,
            x.expert,
            x.layer,
            old.len()
        );
        let leaf = entity! { _ @
            ink::weight_nvfp4_2: handle,
            ink::expert_index: x.expert,
            metadata::name: name_h,
            ink::layer: x.layer,
        };
        let new_id = leaf.root().expect("a leaf has a root");
        added += leaf;
        subst.insert(old[0], new_id);
    }

    // 3. The version root: the parent's members with the moved leaves
    //    replaced. Compare members, not the parent's minting history: an
    //    unchanged model keeps its existing opaque root.
    let members: Vec<Id> = parent_members
        .iter()
        .map(|m| subst.get(m).copied().unwrap_or(*m))
        .collect();
    let unchanged = members.iter().copied().collect::<BTreeSet<_>>() == parent_set;
    let root_e = if unchanged {
        Fragment::empty()
    } else {
        entity! { _ @ model::member*: members.iter() }
    };
    let root = root_e.root().unwrap_or(parent);
    let replaced = subst.iter().filter(|(old, new)| old != new).count();
    let name = label_of(&format!("learned {}", utc_stamp()));
    if root != parent {
        added += root_e;
        let name_h = pile
            .put::<blobencodings::UTF8String, _>(name.clone())
            .map_err(|e| anyhow::anyhow!("store the version label: {e:?}"))?;
        let span_h = pile
            .put::<blobencodings::UTF8String, _>(recipe.span.clone())
            .map_err(|e| anyhow::anyhow!("store the learned span: {e:?}"))?;
        let why_h = pile
            .put::<blobencodings::UTF8String, _>(recipe.explanation.clone())
            .map_err(|e| anyhow::anyhow!("store the explanation: {e:?}"))?;
        let revision_h = pile
            .put::<blobencodings::UTF8String, _>(recipe.code_revision.clone())
            .map_err(|e| anyhow::anyhow!("store the code revision: {e:?}"))?;
        added += entity! { ExclusiveId::force_ref(&root) @
            attrs::parent: parent,
            metadata::name: name_h,
            attrs::learn_lr: recipe.lr,
            attrs::learn_seed: recipe.seed,
            attrs::learn_steps: recipe.steps,
            attrs::learned_span: span_h,
            attrs::explanation: why_h,
            attrs::code_revision: revision_h,
        };
        if let Some(w) = recipe.anchor {
            added += entity! { ExclusiveId::force_ref(&root) @ attrs::learn_anchor: w };
        }
        if let Some(mn) = model_name {
            added += entity! { ExclusiveId::force_ref(&root) @ model::model_name: mn };
        }
    }
    Ok(LearnedVersion {
        collection,
        root,
        parent,
        name,
        facts: added,
        replaced,
        members: members.len(),
    })
}

/// Commit the version's facts into the model graph, signed with `key`. The
/// collection's WRITE policy decides whether that signature counts.
pub fn publish_version(
    pile: &mut triblespace::prelude::Pile,
    key: &ed25519_dalek::SigningKey,
    version: LearnedVersion,
) -> Result<()> {
    use triblespace::prelude::*;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("refresh before publishing '{}': {e:?}", version.name))?;
    let snapshot = pile
        .snapshot()
        .context("freeze version publication authority")?;
    let collection = crate::model_collection::ModelCollection::open(&snapshot, version.collection)
        .context("open the selected version publication collection")?;
    anyhow::ensure!(
        collection.writer_is_admitted(&snapshot, key.verifying_key())?,
        "signing key is not admitted by the selected model collection's WRITE policy"
    );
    let fragment: Fragment = version.facts.into();
    pile.commit(collection, key, fragment)
        .map_err(|e| anyhow::anyhow!("publish '{}': {e}", version.name))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::pile::{self, PileSource};
    use super::*;
    use crate::format::attrs as model;
    use triblespace::core::blob::encodings::tensor::{elements::F32, tensor_blob};
    use triblespace::core::metadata;
    use triblespace::prelude::*;

    fn fixture(
        rooted: bool,
    ) -> Result<(
        std::path::PathBuf,
        Pile,
        ed25519_dalek::SigningKey,
        crate::model_collection::ModelCollection,
        Id,
        LearnedExpert,
    )> {
        let path = std::env::temp_dir().join(format!("inkling-model-reference-{}.pile", genid()));
        std::fs::File::create(&path)?;
        let mut store = Pile::open(&path)?;
        store.refresh()?;
        let key = ed25519_dalek::SigningKey::from_bytes(&[17; 32]);
        let collection =
            crate::model_collection::model_graph_collection_or_create(&mut store, &key)?;
        let expert = LearnedExpert {
            name: "model.llm.layers.41.mlp.experts.w2_weight".to_string(),
            layer: 41,
            expert: 7,
            packed: PackedExpert {
                codes: vec![0; 8],
                scales: vec![0x30],
                scale2: 1.0,
                rows: 1,
                cols: 8,
            },
        };
        let weight = store.put(pile::expert_blob(&expert.packed)?)?;
        let name = store.put::<blobencodings::UTF8String, _>(expert.name.clone())?;
        let mut fragment = entity! { _ @
            pile::attrs::weight_nvfp4_2: weight,
            pile::attrs::expert_index: expert.expert,
            metadata::name: name,
            pile::attrs::layer: expert.layer,
        };
        let expert_member = fragment.root().unwrap();
        let dense_weight = store.put(
            tensor_blob::<F32, 1>([1], anybytes::Bytes::from_source(vec![1.0f32]))
                .map_err(|e| anyhow::anyhow!("fixture dense tensor: {e}"))?,
        )?;
        let dense_name =
            store.put::<blobencodings::UTF8String, _>("model.llm.norm.weight".to_string())?;
        let dense = entity! { _ @
            pile::attrs::weight::<F32, 1>(): dense_weight,
            metadata::name: dense_name,
        };
        let dense_member = dense.root().unwrap();
        fragment += dense;
        let base = fucid();
        if rooted {
            fragment += entity! { &base @ model::member*: [expert_member, dense_member] };
        }
        store.commit(collection, &key, fragment)?;
        store.flush()?;
        Ok((path, store, key, collection, base.id, expert))
    }

    #[test]
    fn unchanged_snapshot_keeps_the_existing_opaque_root() -> Result<()> {
        let (path, mut store, _key, collection, base, _) = fixture(true)?;
        let source = PileSource::open(&path)?;
        assert_eq!(source.model_collection(), collection.handle());
        assert_eq!(source.model_root(), base);
        assert_eq!(source.leaf("model.llm.norm.weight")?.dims, vec![1]);
        let version = learned_version(
            &mut store,
            collection.handle(),
            &[],
            base,
            &VersionRecipe::default(),
        )?;
        assert_eq!(version.root, base);
        assert_eq!(version.parent, base);
        assert!(version.facts.is_empty());
        drop(source);
        store.close()?;
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn snapshots_keep_the_training_base_and_reference_survives_annotation_growth() -> Result<()> {
        let (path, mut store, key, collection, base, mut expert) = fixture(true)?;
        expert.packed.codes[0] = 1;
        let first = learned_version(
            &mut store,
            collection.handle(),
            &[expert.clone()],
            base,
            &VersionRecipe::default(),
        )?;
        let first_root = first.root;
        assert_eq!(first.parent, base);
        publish_version(&mut store, &key, first)?;
        expert.packed.codes[0] = 2;
        let second = learned_version(
            &mut store,
            collection.handle(),
            &[expert.clone()],
            base,
            &VersionRecipe::default(),
        )?;
        let second_root = second.root;
        assert_eq!(second.parent, base);
        assert_ne!(first_root, second_root);
        publish_version(&mut store, &key, second)?;
        let extra =
            entity! { ExclusiveId::force_ref(&base) @ metadata::description: "later annotation" };
        store.commit(collection, &key, extra)?;
        store.flush()?;

        let base_source = PileSource::open_root(&path, Some(base))?;
        let child_source = PileSource::open_root(&path, Some(second_root))?;
        assert_eq!(base_source.model_collection(), collection.handle());
        assert_eq!(child_source.model_collection(), collection.handle());
        assert_eq!(base_source.model_root(), base);
        assert_eq!(child_source.model_root(), second_root);
        assert_eq!(
            base_source.expert_packed_stored(&expert.name, 7)?.codes[0],
            0
        );
        assert_eq!(
            child_source.expert_packed_stored(&expert.name, 7)?.codes[0],
            2
        );
        assert!(
            PileSource::open(&path).is_err(),
            "siblings require explicit selection"
        );
        let snapshot =
            crate::model_collection::snapshot_model_collection_for(&store.snapshot()?, collection)?;
        assert_eq!(
            version_heads(snapshot.facts())
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            [first_root, second_root].into_iter().collect()
        );
        let unchanged = learned_version(
            &mut store,
            collection.handle(),
            &[],
            base,
            &VersionRecipe::default(),
        )?;
        assert_eq!(unchanged.root, base);
        drop((base_source, child_source, snapshot));
        store.close()?;
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn unrelated_model_lineage_and_tokenizer_sequence_do_not_select_an_inkling_head() -> Result<()>
    {
        let (path, mut store, key, collection, base, mut expert) = fixture(true)?;
        let mut additions = crate::format::put_raw(&mut store, &[1.0], &[1, 1])
            .map_err(|e| anyhow::anyhow!("generic model leaf: {e}"))?;
        let generic_leaf = additions.root().unwrap();
        let name = store.put::<blobencodings::UTF8String, _>("unrelated tensor".to_string())?;
        additions += entity! { ExclusiveId::force_ref(&generic_leaf) @ metadata::name: name };
        let unrelated_base = fucid();
        let unrelated_child = fucid();
        additions += entity! { &unrelated_base @ model::member: generic_leaf };
        additions += entity! { &unrelated_child @
            model::member: generic_leaf,
            model::parent: unrelated_base.id,
        };
        // Even a cross-family derivation edge must not suppress the native
        // candidate when its child is not an Inkling member shape.
        additions += entity! { &unrelated_child @ model::parent: base };
        additions += crate::tokenizer::save_tokenizer_json(
            br#"{"model":{"type":"BPE","vocab":{},"merges":[]},
                 "normalizer":{"type":"Sequence","normalizers":[{"type":"NFC"}]}}"#,
            "coexisting tokenizer",
            &mut store,
        )
        .map_err(|e| anyhow::anyhow!("tokenizer Sequence: {e}"))?;
        store.commit(collection, &key, additions)?;
        store.flush()?;

        // The Inkling root is opaque and unlabelled, yet is the only native
        // candidate. Neither the Sequence nor the unrelated graph is a head.
        let source = PileSource::open(&path)?;
        assert_eq!(source.model_root(), base);
        assert_eq!(source.model_collection(), collection.handle());
        drop(source);
        expert.packed.codes[0] = 1;
        let learned = learned_version(
            &mut store,
            collection.handle(),
            &[expert],
            base,
            &VersionRecipe::default(),
        )?;
        let head = learned.root;
        publish_version(&mut store, &key, learned)?;
        store.flush()?;
        let source = PileSource::open(&path)?;
        assert_eq!(source.model_root(), head);
        let snapshot =
            crate::model_collection::snapshot_model_collection_for(&store.snapshot()?, collection)?;
        assert_eq!(version_heads(snapshot.facts()), vec![head]);
        drop((source, snapshot));
        store.close()?;
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn unrooted_or_ambiguous_partial_imports_are_not_synthesized_on_read() -> Result<()> {
        let (path, mut store, key, collection, absent, _) = fixture(false)?;
        let length = std::fs::metadata(&path)?.len();
        assert!(PileSource::open(&path).is_err());
        assert!(PileSource::open_root(&path, Some(absent)).is_err());
        assert_eq!(std::fs::metadata(&path)?.len(), length);
        let a = fucid();
        let b = fucid();
        let snapshot =
            crate::model_collection::snapshot_model_collection_for(&store.snapshot()?, collection)?;
        let member = find!(
            (member: Id),
            pattern!(snapshot.facts(), [{ ?member @ pile::attrs::weight::<F32, 1>(): _?weight }])
        )
        .next()
        .context("fixture dense leaf")?
        .0;
        drop(snapshot);
        let mut partials = entity! { &a @ model::member: member };
        partials += entity! { &b @ model::member: member };
        store.commit(collection, &key, partials)?;
        store.flush()?;
        assert!(PileSource::open(&path).is_err());
        let source = PileSource::open_root(&path, Some(a.id))?;
        assert_eq!(source.model_root(), a.id);
        drop(source);
        store.close()?;
        std::fs::remove_file(path)?;
        Ok(())
    }
}
