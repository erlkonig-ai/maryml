//! Give an existing model pile its one `models` collection.
//!
//!   model_collection_migrate <model.pile> <signing-key>
//!
//! Before mary had one model collection it had two shapes. `mary-model-graph`
//! held model fact archives directly. `mary-model-bundles` held one signed
//! token per model, a single `(root, metadata::archive, H)` row naming the
//! complete fact archive `H`. Mary now reads only `models`, whose members are
//! model fact archives, so this tool makes the archives the old collections
//! held members of `models`: every admitted graph member, and for every
//! admitted bundle token the archive `H` it names. The token itself is not
//! carried; `H` becomes a member directly.
//!
//! When the pile already has a `models` collection (a working pile whose voice
//! models live there, say), the archives join THAT collection, under its own
//! descriptor and policy, and its other members stay as they are; the signer
//! must be admitted by its WRITE policy. No second `models` descriptor is ever
//! created. Only when the pile has none does the tool found one, and then with
//! the old collections' own READ/WRITE policy (refused if graph and bundles
//! differ), never one rooted at whoever runs it: two machines migrating the
//! same synced pile found the same descriptor.
//!
//! What it writes, and nothing else: that descriptor (with its name and policy
//! definitions) when the pile has none, and one COMMIT per (member, metadata)
//! pair not already present (counting only commits by admitted writers),
//! signed by the signer, naming the member archive that is already in the pile
//! and the metadata archive its admitted old COMMIT carried. No archive, tensor or other blob is rewritten or copied, and the
//! old collections and their records stay exactly as they were. The pile is
//! only appended to.
//!
//! Bundle-token rows other than `metadata::archive` are skipped and counted
//! (open world). Each token NAMED its root; the archive alone does not. So the
//! tool refuses a bundle whose archive states another root with the named
//! root's source and quantization: selection by source would then be
//! ambiguous without the token.
//!
//! It refuses, before writing anything, when the pile has no old collection,
//! has two current descriptors under one name, a member archive is not
//! resident, a bundle would become ambiguous as above, or the signer is not
//! admitted (by the existing `models` WRITE policy, or by the old collections'
//! when founding).
//! After writing it proves the result from a fresh observation: the old
//! collections read exactly as before; the `models` members are exactly its
//! members before plus the planned set; its facts are exactly its facts
//! before unioned with the facts unioned from the old collections (one
//! identical TribleSet); there is still one `models` descriptor; and the
//! ordinary loader (`persist::read_model_pile`) selects it and returns those
//! facts. Model ids are content of those facts, so equal facts are equal
//! model identity.
//!
//! An interrupted run leaves a partial set of members that the loader would
//! read as a partial model. Running the tool again completes it: commits
//! already present are not appended twice, and the proof runs again.
//!
//! Never point this at a pile that other machines replicate without deciding
//! that separately: the appended commits replicate too.
use anyhow::{Context, Result, bail, ensure};
use ed25519_dalek::{SigningKey, VerifyingKey};
use mary::format::attrs;
use mary::model_collection::{self, ModelCollection};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::collection::records::empty_metadata_handle;
use triblespace::core::collection::{
    CollectionCommit, CollectionPolicy, CollectionRecord, CollectionStore,
};
use triblespace::core::metadata;
use triblespace::core::repo::StoreRead;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;
use triblespace::prelude::*;

const LEGACY_GRAPH: &str = "mary-model-graph";
const LEGACY_BUNDLES: &str = "mary-model-bundles";

type Archive = Inline<inlineencodings::Handle<SimpleArchive>>;

/// What the old collections hold, read from one frozen observation.
#[derive(Debug, PartialEq)]
struct Plan {
    graph: Option<ModelCollection>,
    bundles: Option<ModelCollection>,
    /// Member archive -> every metadata archive an ADMITTED old COMMIT carried for it.
    members: BTreeMap<Archive, BTreeSet<Archive>>,
    /// The union of every member archive's facts, exactly as stored.
    facts: TribleSet,
    /// Bundle-token rows other than `metadata::archive` (open world: skipped).
    skipped_token_rows: usize,
}

fn hex(raw: &[u8; 32]) -> String {
    raw.iter().map(|b| format!("{b:02x}")).collect()
}

fn sole(store: &impl StoreRead, name: &str) -> Result<Option<ModelCollection>> {
    match model_collection::collections_named_in(store, name)?.as_slice() {
        [] => Ok(None),
        [one] => Ok(Some(*one)),
        many => bail!(
            "{} current descriptors are named {name:?}; refusing to choose",
            many.len()
        ),
    }
}

/// Whether `signer` is admitted by `collection`'s WRITE policy in `store`.
fn admitted(
    store: &impl StoreRead,
    collection: ModelCollection,
    signer: [u8; 32],
    known: &mut BTreeMap<([u8; 32], [u8; 32]), bool>,
) -> Result<bool> {
    if let Some(&answer) = known.get(&(collection.handle().raw, signer)) {
        return Ok(answer);
    }
    let answer = match VerifyingKey::from_bytes(&signer) {
        Ok(key) => collection.writer_is_admitted(store, key).map_err(|error| {
            anyhow::anyhow!("check WRITE admission of {}: {error:?}", hex(&signer))
        })?,
        Err(_) => false,
    };
    known.insert((collection.handle().raw, signer), answer);
    Ok(answer)
}

/// (collection, data) -> metadata of every COMMIT an admitted writer signed,
/// for the given collections.
fn admitted_commits(
    store: &impl StoreRead,
    collections: &[ModelCollection],
) -> Result<BTreeMap<([u8; 32], [u8; 32]), BTreeSet<Archive>>> {
    let mut known = BTreeMap::new();
    let mut commits: BTreeMap<([u8; 32], [u8; 32]), BTreeSet<Archive>> = BTreeMap::new();
    for record in store.records().context("read collection records")? {
        if let CollectionRecord::Commit(commit) = record.context("decode collection record")? {
            let Some(&collection) = collections
                .iter()
                .find(|c| c.handle() == commit.collection())
            else {
                continue;
            };
            if admitted(store, collection, commit.public_key().raw, &mut known)? {
                commits
                    .entry((commit.collection().raw, commit.data().raw))
                    .or_default()
                    .insert(commit.metadata());
            }
        }
    }
    Ok(commits)
}

fn plan(store: &impl StoreRead) -> Result<Plan> {
    let graph = sole(store, LEGACY_GRAPH)?;
    let bundles = sole(store, LEGACY_BUNDLES)?;
    ensure!(
        graph.is_some() || bundles.is_some(),
        "no {LEGACY_GRAPH:?} or {LEGACY_BUNDLES:?} collection in this pile: nothing to migrate"
    );
    let old: Vec<ModelCollection> = [graph, bundles].into_iter().flatten().collect();
    let carried = admitted_commits(store, &old)?;
    let metadata_of = |collection: ModelCollection, data: Archive| {
        carried
            .get(&(collection.handle().raw, data.raw))
            .cloned()
            // A member no admitted COMMIT names directly (a merge result) gets none.
            .unwrap_or_else(|| BTreeSet::from([empty_metadata_handle()]))
    };
    let mut members: BTreeMap<Archive, BTreeSet<Archive>> = BTreeMap::new();
    let mut facts = TribleSet::new();
    let mut skipped_token_rows = 0;
    if let Some(graph) = graph {
        // The collection view reads every admitted member; one not resident fails here.
        let snapshot = model_collection::snapshot_model_collection_for(store, graph)
            .context("read the old graph collection")?;
        for member in snapshot.support().members() {
            members
                .entry(member)
                .or_default()
                .extend(metadata_of(graph, member));
        }
        facts += snapshot.into_facts();
    }
    if let Some(bundles) = bundles {
        let snapshot = model_collection::snapshot_model_collection_for(store, bundles)
            .context("read the old bundle collection")?;
        for token_handle in snapshot.support().members() {
            let token = archive_facts(store, token_handle, "bundle token")?;
            for row in token.iter() {
                if row.a() != &metadata::archive.id() {
                    skipped_token_rows += 1;
                    continue;
                }
                let root = *row.e();
                let archive: Archive = *row.v::<inlineencodings::Handle<SimpleArchive>>();
                let h = archive_facts(store, archive, "bundle archive")?;
                // The token named `root`. Without it, a selector by source and
                // quantization must not find a second root in the same archive.
                let rivals: BTreeSet<Id> = find!(
                    (other: Id),
                    pattern!(&h, [
                        { root @ attrs::source: _?s, attrs::quantization: _?q },
                        { ?other @ attrs::source: _?s, attrs::quantization: _?q }
                    ])
                )
                .map(|(other,)| other)
                .filter(|other| *other != root)
                .collect();
                ensure!(
                    rivals.is_empty(),
                    "bundle token {} names root {root:X}, but its archive {} also states {} other \
                     root(s) with the same source and quantization ({rivals:X?}); without the token, \
                     selecting by source would be ambiguous. Refusing before writing.",
                    hex(&token_handle.raw),
                    hex(&archive.raw),
                    rivals.len()
                );
                facts += h;
                members
                    .entry(archive)
                    .or_default()
                    .extend(metadata_of(bundles, token_handle));
            }
        }
    }
    Ok(Plan {
        graph,
        bundles,
        members,
        facts,
        skipped_token_rows,
    })
}

fn archive_facts(store: &impl StoreRead, member: Archive, what: &str) -> Result<TribleSet> {
    store
        .get::<TribleSet, SimpleArchive>(member)
        .map_err(|error| {
            anyhow::anyhow!("{what} {} is not readable here: {error}", hex(&member.raw))
        })
}

#[derive(Debug, PartialEq)]
struct Outcome {
    collection: ModelCollection,
    members: usize,
    facts: usize,
    commits_appended: usize,
    skipped_token_rows: usize,
}

fn migrate(path: &Path, key: &SigningKey) -> Result<Outcome> {
    let before_len = std::fs::metadata(path)
        .with_context(|| format!("stat {path:?}"))?
        .len();
    let mut pile = Pile::open(path).map_err(|error| anyhow::anyhow!("open {path:?}: {error}"))?;
    let result = migrate_open(&mut pile, key);
    let close = pile
        .close()
        .map_err(|error| anyhow::anyhow!("close {path:?}: {error}"));
    let (mut outcome, facts) = result?;
    close?;
    // The ordinary loader now selects `models` and reads the same facts.
    let loaded = mary::persist::read_model_pile(path)?;
    ensure!(
        loaded.collection == outcome.collection,
        "the loader selected a different collection"
    );
    outcome.facts = facts.len();
    ensure!(
        loaded.facts == model_collection::project_legacy_model_attributes(&facts).facts,
        "the loader's facts differ from the migrated collection's facts"
    );
    ensure!(
        std::fs::metadata(path)?.len() >= before_len,
        "the pile shrank"
    );
    Ok(outcome)
}

/// The descriptor the archives go into: the pile's `models` when it has one,
/// otherwise the one founded with the old collections' own READ/WRITE policy
/// (so every machine migrating the same pile founds the same descriptor).
enum Target {
    Join(ModelCollection),
    Found(CollectionPolicy),
}

fn migrate_open(pile: &mut Pile, key: &SigningKey) -> Result<(Outcome, TribleSet)> {
    pile.refresh().context("refresh the pile")?;
    let before = pile.snapshot().context("freeze the pile")?;
    let planned = plan(&before)?;
    let signer = key.verifying_key();
    let old: Vec<ModelCollection> = [planned.graph, planned.bundles]
        .into_iter()
        .flatten()
        .collect();
    let target = match sole(&before, model_collection::model_collection_name())? {
        Some(existing) => {
            ensure!(
                existing
                    .writer_is_admitted(&before, signer)
                    .map_err(|error| anyhow::anyhow!("check WRITE admission: {error:?}"))?,
                "signer {} is not admitted by the WRITE policy of the existing models collection {}; \
                 nothing written",
                hex(&signer.to_bytes()),
                hex(&existing.handle().raw)
            );
            Target::Join(existing)
        }
        None => {
            let policies = old
                .iter()
                .map(|c| {
                    c.policy(&before).map_err(|error| {
                        anyhow::anyhow!("read the policy of {}: {error:?}", hex(&c.handle().raw))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                policies
                    .windows(2)
                    .all(|p| p[0].fragment().facts() == p[1].fragment().facts()),
                "{LEGACY_GRAPH:?} and {LEGACY_BUNDLES:?} have different READ/WRITE policies; \
                 refusing to choose one for the new models collection"
            );
            for collection in &old {
                ensure!(
                    collection
                        .writer_is_admitted(&before, signer)
                        .map_err(|error| anyhow::anyhow!("check WRITE admission: {error:?}"))?,
                    "signer {} is not admitted by the WRITE policy of {}; nothing written",
                    hex(&signer.to_bytes()),
                    hex(&collection.handle().raw)
                );
            }
            Target::Found(
                policies
                    .into_iter()
                    .next()
                    .expect("at least one old collection"),
            )
        }
    };
    // What `models` already holds, if the pile has one: it is kept as it is.
    let (held_before, facts_before, done) = match &target {
        Target::Join(existing) => {
            let held = model_collection::snapshot_model_collection_for(&before, *existing)?;
            let members: BTreeSet<Archive> = held.support().members().collect();
            let done = admitted_commits(&before, &[*existing])?;
            (members, held.into_facts(), done)
        }
        Target::Found(_) => (BTreeSet::new(), TribleSet::new(), BTreeMap::new()),
    };
    drop(before);
    let collection = match target {
        Target::Join(existing) => existing,
        Target::Found(policy) => pile
            .collection(model_collection::model_collection_name(), policy)
            .map_err(|error| anyhow::anyhow!("found the models collection: {error}"))?,
    };
    let todo: Vec<(Archive, Archive)> = planned
        .members
        .iter()
        .flat_map(|(m, metas)| metas.iter().map(|meta| (*m, *meta)))
        .filter(|(m, meta)| {
            !done
                .get(&(collection.handle().raw, m.raw))
                .is_some_and(|metas| metas.contains(meta))
        })
        .collect();
    for &(member, meta) in &todo {
        let data = inlineencodings::Handle::<SimpleArchive>::to_hash(member);
        pile.insert(CollectionRecord::Commit(CollectionCommit::sign(
            key,
            collection.handle(),
            data,
            meta,
        )))
        .map_err(|error| {
            anyhow::anyhow!("append models commit for {}: {error}", hex(&member.raw))
        })?;
    }
    pile.refresh().context("refresh after appending")?;
    let after = pile.snapshot().context("freeze the migrated pile")?;
    // The proof, from the new observation.
    let replanned = plan(&after)?;
    let unchanged = replanned == planned;
    drop(replanned);
    ensure!(
        unchanged,
        "the old collections changed while migrating (a live sync?); {} commits were appended, \
         rerun to complete",
        todo.len()
    );
    ensure!(
        model_collection::collections_named_in(&after, model_collection::model_collection_name())?
            == vec![collection],
        "the pile no longer has exactly one models collection"
    );
    let new = model_collection::snapshot_model_collection_for(&after, collection)?;
    let held: BTreeSet<Archive> = new.support().members().collect();
    let wanted: BTreeSet<Archive> = planned.members.keys().copied().collect();
    let expected: BTreeSet<Archive> = held_before.union(&wanted).copied().collect();
    ensure!(
        held == expected,
        "models holds {} members, expected its {} before plus the old collections' {}; \
         rerun to complete",
        held.len(),
        held_before.len(),
        wanted.len()
    );
    let facts = facts_before + planned.facts;
    ensure!(
        new.facts() == &facts,
        "models facts differ from its facts before unioned with the old collections' facts"
    );
    Ok((
        Outcome {
            collection,
            members: held.len(),
            facts: 0,
            commits_appended: todo.len(),
            skipped_token_rows: planned.skipped_token_rows,
        },
        facts,
    ))
}

fn main() -> Result<()> {
    let usage = "usage: model_collection_migrate <model.pile> <signing-key>";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [pile, key] = args.as_slice() else {
        bail!("{usage}")
    };
    let key = signing_key_file::load_existing(Path::new(key))
        .with_context(|| format!("load existing signing key {key:?}"))?;
    let outcome = migrate(Path::new(pile), &key)?;
    println!(
        "models {} members {} facts {} commits_appended {} skipped_token_rows {}",
        hex(&outcome.collection.handle().raw),
        outcome.members,
        outcome.facts,
        outcome.commits_appended,
        outcome.skipped_token_rows
    );
    println!(
        "proof: models facts == its facts before + union of {LEGACY_GRAPH} and {LEGACY_BUNDLES} archives \
         (old collections unchanged), members exact, one descriptor, loader agrees"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use triblespace::core::collection::{AdmissionPolicy, CollectionPolicy};

    struct TempPile(PathBuf);
    impl TempPile {
        fn new(label: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "model-collection-migrate-{label}-{}-{nanos}.pile",
                std::process::id()
            ));
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            Self(path)
        }
    }
    impl Drop for TempPile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn policy(key: &SigningKey) -> CollectionPolicy {
        CollectionPolicy::new(
            AdmissionPolicy::direct(key.verifying_key()),
            AdmissionPolicy::direct(key.verifying_key()),
        )
    }

    /// One model root whose facts name blobs (a stand-in for tensor leaves).
    fn model(label: &str) -> Fragment {
        entity! { _ @ mary::format::attrs::kind: "probe-model",
        metadata::name: label.to_owned(), mary::format::attrs::source: format!("{label} tensor bytes") }
    }

    fn write_graph(pile: &mut Pile, key: &SigningKey, fragments: Vec<Fragment>) {
        let graph = pile.collection(LEGACY_GRAPH, policy(key)).unwrap();
        for fragment in fragments {
            pile.commit(graph, key, fragment).unwrap();
        }
    }

    /// The retired bundle writer, reproduced: H is the canonical archive of
    /// the model's facts, the COMMIT data is a one-row token naming it.
    fn write_bundle(pile: &mut Pile, key: &SigningKey, fragment: Fragment) -> (Id, Archive) {
        let bundles = pile.collection(LEGACY_BUNDLES, policy(key)).unwrap();
        let root = fragment.root().unwrap();
        let (_, facts, metafacts, blobs) = fragment.into_parts();
        let mut token = Fragment::empty();
        token.blobs_mut().union(blobs);
        let h = token.put::<SimpleArchive, _>(facts.to_blob());
        token += entity! { ExclusiveId::force_ref(&root) @ metadata::archive: h };
        *token.metafacts_mut() += metafacts;
        pile.commit(bundles, key, token).unwrap();
        (root, h)
    }

    fn old_facts(path: &Path) -> TribleSet {
        let mut pile = Pile::open(path).unwrap();
        let facts = plan(&pile.snapshot().unwrap()).unwrap().facts;
        pile.close().unwrap();
        facts
    }

    fn run(
        label: &str,
        build: impl FnOnce(&mut Pile, &SigningKey),
    ) -> (TempPile, Outcome, TribleSet, Vec<u8>) {
        let path = TempPile::new(label);
        let author = SigningKey::from_bytes(&[0x51; 32]);
        let mut pile = Pile::open(&path.0).unwrap();
        build(&mut pile, &author);
        pile.close().unwrap();
        let before = std::fs::read(&path.0).unwrap();
        let facts = old_facts(&path.0);
        // The founded `models` takes the old collections' policy, so their
        // own writer migrates.
        let outcome = migrate(&path.0, &author).unwrap();
        let after = std::fs::read(&path.0).unwrap();
        assert!(
            after.len() > before.len() && after.starts_with(&before),
            "append-only: the old bytes are a prefix"
        );
        // A second run appends nothing and proves the same.
        let again = migrate(&path.0, &author).unwrap();
        assert_eq!(again.commits_appended, 0);
        assert_eq!(std::fs::read(&path.0).unwrap(), after);
        (path, outcome, facts, before)
    }

    fn loaded(path: &Path) -> mary::persist::ModelPileSource {
        mary::persist::read_model_pile(path).unwrap()
    }

    #[test]
    fn graph_pile_members_become_models_members() {
        let (a, b) = (model("alpha"), model("tokenizer"));
        let roots = [a.root().unwrap(), b.root().unwrap()];
        let archives: BTreeSet<Archive> = [&a, &b]
            .iter()
            .map(|f| f.facts().clone().to_blob().get_handle())
            .collect();
        let (path, outcome, facts, _) =
            run("graph", |pile, key| write_graph(pile, key, vec![a, b]));
        assert_eq!((outcome.members, outcome.commits_appended), (2, 2));
        let source = loaded(&path.0);
        assert_eq!(source.facts, facts);
        let mut pile = Pile::open(&path.0).unwrap();
        let snap = pile.snapshot().unwrap();
        let members: BTreeSet<Archive> =
            model_collection::snapshot_model_collection_for(&snap, outcome.collection)
                .unwrap()
                .support()
                .members()
                .collect();
        assert_eq!(
            members, archives,
            "the graph's own archives, not re-encoded ones"
        );
        // Founded with the old graph's READ/WRITE policy, never the migrator's key.
        let graph = sole(&snap, LEGACY_GRAPH).unwrap().unwrap();
        assert_eq!(
            outcome.collection.policy(&snap).unwrap().fragment().facts(),
            graph.policy(&snap).unwrap().fragment().facts()
        );
        for root in roots {
            assert!(
                source.facts.iter().any(|t| t.e() == &root),
                "model identity {root:X} kept"
            );
        }
        drop(snap);
        pile.close().unwrap();
    }

    #[test]
    fn bundle_archive_itself_becomes_the_member() {
        let fragment = model("bundled");
        let mut h = None;
        let (path, outcome, facts, _) = run("bundle", |pile, key| {
            h = Some(write_bundle(pile, key, fragment))
        });
        let (root, h) = h.unwrap();
        assert_eq!(outcome.members, 1);
        let mut pile = Pile::open(&path.0).unwrap();
        let snap = pile.snapshot().unwrap();
        let new =
            model_collection::snapshot_model_collection_for(&snap, outcome.collection).unwrap();
        assert_eq!(new.support().members().collect::<Vec<_>>(), vec![h]);
        assert_eq!(new.complete_image(), Some(h));
        assert!(
            new.facts().iter().all(|t| t.a() != &metadata::archive.id()),
            "no token row survives"
        );
        assert_eq!(new.facts(), &facts);
        assert!(facts.iter().any(|t| t.e() == &root));
        drop(new);
        drop(snap);
        pile.close().unwrap();
        assert_eq!(loaded(&path.0).facts, facts);
    }

    #[test]
    fn mixed_pile_unions_both_shapes() {
        let (weights, tokenizer) = (model("weights"), model("tokenizer"));
        let mut h = None;
        let (path, outcome, facts, _) = run("mixed", |pile, key| {
            h = Some(write_bundle(pile, key, weights));
            write_graph(pile, key, vec![tokenizer]);
        });
        assert_eq!(outcome.members, 2);
        assert!(facts.len() > 0);
        let source = loaded(&path.0);
        assert_eq!(source.facts, facts);
        assert_eq!(source.collection, outcome.collection);
        let mut pile = Pile::open(&path.0).unwrap();
        let snap = pile.snapshot().unwrap();
        let members: Vec<Archive> =
            model_collection::snapshot_model_collection_for(&snap, outcome.collection)
                .unwrap()
                .support()
                .members()
                .collect();
        assert!(members.contains(&h.unwrap().1));
        drop(snap);
        pile.close().unwrap();
    }

    #[test]
    fn refusals_append_nothing() {
        let key = SigningKey::from_bytes(&[0x53; 32]);
        // Nothing to migrate.
        let empty = TempPile::new("empty");
        assert!(
            migrate(&empty.0, &key)
                .unwrap_err()
                .to_string()
                .contains("nothing to migrate")
        );
        assert_eq!(std::fs::metadata(&empty.0).unwrap().len(), 0);
        // Old graph and bundles under different policies: no policy to found with.
        let split = TempPile::new("split-policy");
        let other = SigningKey::from_bytes(&[0x56; 32]);
        let mut pile = Pile::open(&split.0).unwrap();
        write_graph(&mut pile, &key, vec![model("graph")]);
        write_bundle(&mut pile, &other, model("bundle"));
        pile.close().unwrap();
        let before = std::fs::read(&split.0).unwrap();
        assert!(
            format!("{:#}", migrate(&split.0, &key).unwrap_err())
                .contains("different READ/WRITE policies")
        );
        assert_eq!(std::fs::read(&split.0).unwrap(), before);
        // A signer the old policy does not admit founds nothing.
        let stranger = SigningKey::from_bytes(&[0x57; 32]);
        let lone = TempPile::new("stranger");
        let mut pile = Pile::open(&lone.0).unwrap();
        write_graph(&mut pile, &key, vec![model("graph")]);
        pile.close().unwrap();
        let before = std::fs::read(&lone.0).unwrap();
        assert!(format!("{:#}", migrate(&lone.0, &stranger).unwrap_err()).contains("not admitted"));
        assert_eq!(std::fs::read(&lone.0).unwrap(), before);
        // A bundle whose archive states a second root with the token root's
        // source and quantization: only the token told them apart.
        let twin = TempPile::new("two-roots");
        let mut named = entity! { _ @ attrs::kind: "probe-model",
        attrs::source: "nvidia/personaplex-7b-v1", attrs::quantization: "native",
        metadata::name: "named" };
        let named_root = named.root().unwrap();
        named += entity! { _ @ attrs::kind: "probe-model",
        attrs::source: "nvidia/personaplex-7b-v1", attrs::quantization: "native",
        metadata::name: "ambient" };
        let (_, facts, metafacts, blobs) = named.into_parts();
        let named = Fragment::rooted_from_parts(named_root, facts, metafacts, blobs);
        let mut pile = Pile::open(&twin.0).unwrap();
        write_bundle(&mut pile, &key, named);
        pile.close().unwrap();
        let before = std::fs::read(&twin.0).unwrap();
        let error = format!("{:#}", migrate(&twin.0, &key).unwrap_err());
        assert!(
            error.contains("ambiguous") && error.contains("Refusing before writing"),
            "{error}"
        );
        assert_eq!(std::fs::read(&twin.0).unwrap(), before);
        // An existing models collection whose WRITE policy does not admit the signer.
        let foreign = TempPile::new("foreign");
        let owner = SigningKey::from_bytes(&[0x54; 32]);
        let mut pile = Pile::open(&foreign.0).unwrap();
        write_graph(&mut pile, &key, vec![model("old")]);
        let models = pile
            .collection(model_collection::model_collection_name(), policy(&owner))
            .unwrap();
        pile.commit(models, &owner, model("voice")).unwrap();
        pile.close().unwrap();
        let before = std::fs::read(&foreign.0).unwrap();
        assert!(format!("{:#}", migrate(&foreign.0, &key).unwrap_err()).contains("not admitted"));
        assert_eq!(std::fs::read(&foreign.0).unwrap(), before);
    }

    /// A working pile's `models` collection with unrelated members gains the
    /// old archives under its own descriptor; nothing else in it moves.
    #[test]
    fn existing_models_collection_gains_the_archives_under_its_own_descriptor() {
        let path = TempPile::new("existing-models");
        let owner = SigningKey::from_bytes(&[0x55; 32]);
        let (breeze, voxtral) = (model("breeze"), model("voxtral"));
        let unrelated: BTreeSet<Archive> = [&breeze, &voxtral]
            .iter()
            .map(|f| f.facts().clone().to_blob().get_handle())
            .collect();
        let mut pile = Pile::open(&path.0).unwrap();
        let models = pile
            .collection(model_collection::model_collection_name(), policy(&owner))
            .unwrap();
        pile.commit(models, &owner, breeze).unwrap();
        pile.commit(models, &owner, voxtral).unwrap();
        let (_, h) = write_bundle(&mut pile, &owner, model("bundled"));
        let tokenizer = model("tokenizer");
        let graph_member: Archive = tokenizer.facts().clone().to_blob().get_handle();
        write_graph(&mut pile, &owner, vec![tokenizer]);
        let snap = pile.snapshot().unwrap();
        let facts_before = model_collection::snapshot_model_collection_for(&snap, models)
            .unwrap()
            .facts()
            .clone();
        drop(snap);
        pile.close().unwrap();
        let old = old_facts(&path.0);
        let bytes_before = std::fs::read(&path.0).unwrap();

        let outcome = migrate(&path.0, &owner).unwrap();
        assert_eq!(outcome.collection, models, "joined, not founded");
        assert_eq!((outcome.members, outcome.commits_appended), (4, 2));
        let bytes_after = std::fs::read(&path.0).unwrap();
        assert!(bytes_after.starts_with(&bytes_before));
        let mut pile = Pile::open(&path.0).unwrap();
        let snap = pile.snapshot().unwrap();
        assert_eq!(
            model_collection::collections_named_in(
                &snap,
                model_collection::model_collection_name()
            )
            .unwrap(),
            vec![models],
            "no second models descriptor"
        );
        let new = model_collection::snapshot_model_collection_for(&snap, models).unwrap();
        let members: BTreeSet<Archive> = new.support().members().collect();
        let expected: BTreeSet<Archive> =
            unrelated.iter().copied().chain([h, graph_member]).collect();
        assert_eq!(members, expected);
        assert_eq!(new.facts(), &(facts_before + old.clone()));
        drop(new);
        drop(snap);
        pile.close().unwrap();
        let source = loaded(&path.0);
        assert_eq!(source.collection, models);
        assert!(
            old.iter().all(|t| source.facts.contains(t)),
            "every old fact is read"
        );
        let again = migrate(&path.0, &owner).unwrap();
        assert_eq!(again.commits_appended, 0);
        assert_eq!(std::fs::read(&path.0).unwrap(), bytes_after);
    }

    /// Open world: a token row other than `metadata::archive` is skipped and
    /// counted, and the archive it sits beside still migrates.
    #[test]
    fn bundle_token_with_an_extra_row_migrates() {
        let path = TempPile::new("extra-row");
        let key = SigningKey::from_bytes(&[0x58; 32]);
        let fragment = model("bundled");
        let root = fragment.root().unwrap();
        let (_, facts, metafacts, blobs) = fragment.into_parts();
        let mut token = Fragment::empty();
        token.blobs_mut().union(blobs);
        let h = token.put::<SimpleArchive, _>(facts.clone().to_blob());
        token += entity! { ExclusiveId::force_ref(&root) @ metadata::archive: h,
        metadata::name: "an annotation the old writer never made" };
        *token.metafacts_mut() += metafacts;
        let mut pile = Pile::open(&path.0).unwrap();
        let bundles = pile.collection(LEGACY_BUNDLES, policy(&key)).unwrap();
        pile.commit(bundles, &key, token).unwrap();
        pile.close().unwrap();
        let outcome = migrate(&path.0, &key).unwrap();
        assert_eq!((outcome.members, outcome.skipped_token_rows), (1, 1));
        assert_eq!(
            loaded(&path.0).facts,
            model_collection::project_legacy_model_attributes(&facts).facts
        );
    }

    /// Commits by writers the policy does not admit neither count as already
    /// present nor carry metadata.
    #[test]
    fn unadmitted_commits_are_not_counted() {
        let path = TempPile::new("unadmitted");
        let owner = SigningKey::from_bytes(&[0x59; 32]);
        let outsider = SigningKey::from_bytes(&[0x5a; 32]);
        let mut pile = Pile::open(&path.0).unwrap();
        let graph = pile.collection(LEGACY_GRAPH, policy(&owner)).unwrap();
        let commit = pile.commit(graph, &owner, model("weights")).unwrap();
        let models = pile
            .collection(model_collection::model_collection_name(), policy(&owner))
            .unwrap();
        // An outsider claims the member is already in models (same metadata),
        // and attaches other metadata to it in the old graph.
        for (collection, meta) in [
            (models, commit.metadata()),
            (graph, empty_metadata_handle()),
        ] {
            pile.insert(CollectionRecord::Commit(CollectionCommit::sign(
                &outsider,
                collection.handle(),
                commit.data(),
                meta,
            )))
            .unwrap();
        }
        pile.close().unwrap();
        let outcome = migrate(&path.0, &owner).unwrap();
        assert_eq!(outcome.collection, models);
        assert_eq!(
            (outcome.members, outcome.commits_appended),
            (1, 1),
            "one admitted pair"
        );
    }
}
