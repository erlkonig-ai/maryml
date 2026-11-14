//! A published model commit describes every attribute its data uses.
//!
//! `pile.commit` writes a fragment's facts into the commit's data archive and
//! its metafacts into the metadata archive. Flattening `entity!` fragments to
//! bare facts before publication left that second archive empty, so a reader
//! of the pile could not type a single model attribute without mary's source.

#![cfg(all(feature = "import", feature = "tokenizer"))]

use ed25519_dalek::SigningKey;
use mary::ingest::LeafDtype;
use mary::persist;
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::path::{Path, PathBuf};
use triblespace::core::blob::Blob;
use triblespace::core::collection::CollectionCommit;
use triblespace::core::metadata;
use triblespace::prelude::*;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("mary-publish-metadata-{}", fucid().id));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const TOKENIZER: &str = r###"{"version": "1.0", "added_tokens": [
    {"id": 3, "content": "<e>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
  "normalizer": null, "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": {"type": "TemplateProcessing",
    "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<e>", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
    "special_tokens": {"<e>": {"id": "<e>", "ids": [3], "tokens": ["<e>"]}}},
  "decoder": null,
  "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1, "b": 2}}}"###;

fn write_f32(path: &Path) {
    let matrix: Vec<u8> = (0..8).flat_map(|i| (i as f32 * 0.25).to_le_bytes()).collect();
    let vector: Vec<u8> = (0..4).flat_map(|i| (1.0 - i as f32).to_le_bytes()).collect();
    serialize_to_file(
        [
            ("block.weight", TensorView::new(Dtype::F32, vec![2, 4], &matrix).unwrap()),
            ("norm.weight", TensorView::new(Dtype::F32, vec![4], &vector).unwrap()),
        ],
        &None,
        path,
    )
    .unwrap();
}

fn write_bf16(path: &Path) {
    let bits: [u16; 8] = [0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0x0000, 0x8000];
    let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    serialize_to_file(
        [("proj.weight", TensorView::new(Dtype::BF16, vec![2, 4], &raw).unwrap())],
        &None,
        path,
    )
    .unwrap();
}

/// What one published commit carries, read back from the pile.
struct Published {
    label: &'static str,
    root: Id,
    commit: CollectionCommit,
}

/// Every publish path the fixture exercises: the directory importer at the
/// three storage widths, and a root completed by a tokenizer, a configuration
/// and a chat template.
fn publish_fixture(fixture: &Fixture) -> (PathBuf, Vec<Published>) {
    let f32_dir = fixture.0.join("f32");
    std::fs::create_dir(&f32_dir).unwrap();
    write_f32(&f32_dir.join("model.safetensors"));
    let bf16_dir = fixture.0.join("bf16");
    std::fs::create_dir(&bf16_dir).unwrap();
    write_bf16(&bf16_dir.join("model.safetensors"));

    let path = fixture.0.join("models.pile");
    std::fs::File::create(&path).unwrap();
    let mut pile = Pile::open(&path).unwrap();
    let key = SigningKey::from_bytes(&[0x6d; 32]);
    let mut out = Vec::new();
    for (label, dir, dtype) in [
        ("f32", &f32_dir, LeafDtype::F32),
        ("f16", &f32_dir, LeafDtype::F16),
        ("bf16", &bf16_dir, LeafDtype::Bf16),
    ] {
        let (root, commit) =
            persist::import_model_to_collection(&mut pile, &key, dir, dtype, "fixture/m", label)
                .unwrap();
        out.push(Published { label, root, commit });
    }

    // A root whose core also names a tokenizer, a configuration and a template.
    let bytes = std::fs::read(f32_dir.join("model.safetensors")).unwrap();
    let (members, mut facts) =
        mary::ingest::ingest_members(&bytes, &mut pile, LeafDtype::F32, |_| true).unwrap();
    let tokenizer = mary::tokenizer::save_tokenizer_json(TOKENIZER.as_bytes(), "fixture/m", &mut pile)
        .unwrap();
    let tokenizer_id = tokenizer.root();
    facts += tokenizer;
    let config = mary::jsonfacts::save_json(
        &serde_json::json!({"hidden_size": 8, "rope": {"theta": 1e6, "scaled": true}}),
        &mut pile,
        &mut facts,
    )
    .unwrap();
    let template = pile
        .put::<blobencodings::UTF8String, _>("{{ messages }}".to_owned())
        .unwrap();
    let fragment = mary::ingest::build_model_root_with(
        &mut pile,
        "fixture/m",
        "parts",
        members,
        facts,
        &["model.safetensors".to_owned()],
        mary::ingest::RootParts {
            tokenizer: tokenizer_id,
            config: Some(config),
            chat_template: Some(template),
        },
    )
    .unwrap();
    let root = fragment.root().unwrap();
    let commit = mary::model_collection::publish_model_fragment(&mut pile, &key, fragment).unwrap();
    out.push(Published { label: "parts", root, commit });
    pile.close().unwrap();
    (path, out)
}

#[test]
fn published_model_commits_describe_every_attribute_they_use() {
    let fixture = Fixture::new();
    let (path, published) = publish_fixture(&fixture);
    let snapshot = mary::model_collection::load_model_collection_local_latest(&path).unwrap();
    let store = snapshot.store();
    for p in &published {
        let data_handle: Inline<inlineencodings::Handle<blobencodings::SimpleArchive>> =
            p.commit.data().transmute();
        let data_blob: Blob<blobencodings::SimpleArchive> = store.get(data_handle).unwrap();
        let meta_blob: Blob<blobencodings::SimpleArchive> = store.get(p.commit.metadata()).unwrap();
        let data: TribleSet = store.get(data_handle).unwrap();
        let meta: TribleSet = store.get(p.commit.metadata()).unwrap();
        let attributes: std::collections::BTreeSet<Id> =
            data.iter().map(|t| *t.a()).collect();
        println!(
            "EVIDENCE {} root={:X} data={} data_bytes={} data_tribles={} metadata={} metadata_bytes={} metadata_tribles={} attributes={}",
            p.label,
            p.root,
            hex(&p.commit.data().raw),
            data_blob.bytes.len(),
            data.len(),
            hex(&p.commit.metadata().raw),
            meta_blob.bytes.len(),
            meta.len(),
            attributes.len(),
        );

        // Ids and data are what they were before metadata was kept: the
        // values are the ones origin/main 4368ffe published for this fixture.
        let (root, data_hex) = BEFORE
            .iter()
            .find(|(label, _, _)| *label == p.label)
            .map(|(_, root, data)| (*root, *data))
            .unwrap();
        assert_eq!(format!("{:X}", p.root), root, "{}: root id moved", p.label);
        assert_eq!(hex(&p.commit.data().raw), data_hex, "{}: data archive moved", p.label);

        // An attribute is a handle when its values name resident blobs.
        let handles: std::collections::BTreeSet<Id> = data
            .iter()
            .filter(|t| {
                let v: Inline<inlineencodings::Handle<blobencodings::UnknownBlob>> =
                    Inline::new(t.v::<triblespace::core::inline::encodings::UnknownInline>().raw);
                store.get::<Blob<blobencodings::UnknownBlob>, _>(v).is_ok()
            })
            .map(|t| *t.a())
            .collect();
        assert!(!handles.is_empty(), "{}: the fixture uses handle attributes", p.label);
        let mut undescribed = Vec::new();
        for attribute in &attributes {
            let encodings: Vec<Id> = find!(
                encoding: Id,
                pattern!(&meta, [{ attribute @
                    metadata::tag: metadata::KIND_ATTRIBUTE,
                    metadata::value_encoding: ?encoding,
                }])
            )
            .collect();
            if encodings.is_empty() {
                undescribed.push(format!("{attribute:X}: no attribute record with value_encoding"));
                continue;
            }
            if handles.contains(attribute) {
                let typed = encodings.iter().any(|encoding| {
                    exists!((blob: Id), pattern!(&meta, [{ encoding @ metadata::blob_encoding: ?blob }]))
                });
                if !typed {
                    undescribed.push(format!("{attribute:X}: handle encoding without blob_encoding"));
                }
            }
        }
        assert!(undescribed.is_empty(), "{}: {undescribed:#?}", p.label);
    }
}

/// `(label, root, data archive)` published by origin/main 4368ffe for the
/// fixture above (logs/base-probe.log in the lane that made this change).
const BEFORE: [(&str, &str, &str); 4] = [
    ("f32", "7A3558A6072CD98ABFBE2FB0B52F328A", "DB5D33FDD58292FA414CFEF3B86023677D745CE97B144DF6CEAFAA5A4D09B833"),
    ("f16", "10C76A1F121A2B871BDE396FE3DD10E6", "7E9CCFF1C12C6F8598E5BD78EA7A02241518A456FD39AAD339EAE0DA65C94B47"),
    ("bf16", "E6C8DCDB92B1F413FC7343E34B19719E", "BBC060C920177A2E598719FB9EC94F4AC3431B2C2E70DCFB2228F53B98D4BC28"),
    ("parts", "797AE8AFDFC8FD6460921E5E664CA8CE", "9C40A5C9896CAFDC870935EE4EF13DC39B00C09FD95888873807378DD976A3C9"),
];

/// The ids of the attributes a tensor leaf can be stored under. They are
/// `Attribute::anchored`, so their identity carries an anchor and an encoding
/// and no name: a reader of a pile sees them as unnamed attributes.
#[test]
fn print_the_anchored_tensor_attribute_ids() {
    use triblespace::core::blob::encodings::tensor::elements::{BF16, F16, F32, NVFP4};
    macro_rules! row {
        ($family:literal, $f:ident, $t:ty, $r:literal) => {
            println!("ATTR {} {}<{},{}> {:X}", $family, stringify!($f), stringify!($t), $r, $f::<$t, $r>().id());
        };
    }
    use mary::leaf::leaf;
    use triblespace::core::attribute::Attribute;
    use triblespace::core::blob::encodings::tensor::{Tensor, TensorElement};
    // `models::inkling::pile::attrs::weight`, which only a CUDA build compiles:
    // the same derivation from that module's WEIGHT_ANCHOR.
    fn weight<T: TensorElement, const RANK: usize>(
    ) -> Attribute<inlineencodings::Handle<Tensor<T, RANK>>> {
        Attribute::anchored(triblespace::core::id_hex!("0B51DA3E67216213871743E045590DBC"))
    }
    macro_rules! ranks {
        ($family:literal, $f:ident, $t:ty) => {
            row!($family, $f, $t, 0); row!($family, $f, $t, 1); row!($family, $f, $t, 2);
            row!($family, $f, $t, 3); row!($family, $f, $t, 4); row!($family, $f, $t, 5); row!($family, $f, $t, 6);
        };
    }
    ranks!("mary::leaf", leaf, F32);
    ranks!("mary::leaf", leaf, F16);
    ranks!("mary::leaf", leaf, BF16);
    ranks!("mary::leaf", leaf, NVFP4);
    ranks!("inkling::pile::attrs", weight, F32);
    ranks!("inkling::pile::attrs", weight, F16);
    ranks!("inkling::pile::attrs", weight, BF16);
    ranks!("inkling::pile::attrs", weight, NVFP4);
}

fn hex(raw: &[u8; 32]) -> String {
    raw.iter().map(|b| format!("{b:02X}")).collect()
}

// ── model_metadata_backfill ──────────────────────────────────────────────

use mary::metadata_backfill::{self, Catalog};
use std::collections::BTreeSet;
use triblespace::core::blob::encodings::UnknownBlob;
use triblespace::core::collection::{CollectionRecord, CollectionRecordSelector, CollectionStore};
use triblespace::core::collection::CollectionRead;
use triblespace::core::repo::{BlobStoreMeta, SnapshotSource, potential_handles};
use triblespace::core::trible::{Fragment, Trible};

/// An attribute id no build knows. Minted with `trible genid` on 2026-11-14.
const UNKNOWN_ATTRIBUTE: Id = triblespace::core::id_hex!("8760F0BCDEA0F450E79DF8477509B44E");
/// The entity carrying it. Minted 2026-11-14.
const UNKNOWN_ENTITY: Id = triblespace::core::id_hex!("47E9397AF3476E31C3CC9C78774EA002");

type ArchiveHandle = Inline<inlineencodings::Handle<blobencodings::SimpleArchive>>;

/// What the new writer published for one fixture commit, read back from its pile.
struct Written {
    label: &'static str,
    data: TribleSet,
    data_hash: triblespace::core::collection::records::CollectionData,
    metadata: TribleSet,
    metadata_handle: ArchiveHandle,
    metadata_bytes: usize,
    /// Blobs the data's tribles name (tensors, strings): what an old writer stored.
    content: Vec<Blob<UnknownBlob>>,
    /// Blobs the metadata names that the new writer stored.
    described: BTreeSet<[u8; 32]>,
}

fn written(path: &Path, published: &[Published]) -> Vec<Written> {
    let mut pile = Pile::open(path).unwrap();
    pile.refresh().unwrap();
    let snapshot = pile.snapshot().unwrap();
    let mut out = Vec::new();
    for p in published {
        let data_handle: ArchiveHandle = p.commit.data().transmute();
        let data: TribleSet = snapshot.get(data_handle).unwrap();
        let metadata: TribleSet = snapshot.get(p.commit.metadata()).unwrap();
        let metadata_blob: Blob<blobencodings::SimpleArchive> =
            snapshot.get(p.commit.metadata()).unwrap();
        let mut seen = BTreeSet::new();
        let content = potential_handles(&data)
            .filter(|handle| seen.insert(handle.raw))
            .filter_map(|handle| snapshot.get::<Blob<UnknownBlob>, UnknownBlob>(handle).ok())
            .collect();
        let described = potential_handles(&metadata)
            .filter(|handle| snapshot.metadata(*handle).unwrap().is_some())
            .map(|handle| handle.raw)
            .collect();
        out.push(Written {
            label: p.label,
            data,
            data_hash: p.commit.data(),
            metadata,
            metadata_handle: p.commit.metadata(),
            metadata_bytes: metadata_blob.bytes.len(),
            content,
            described,
        });
    }
    drop(snapshot);
    pile.close().unwrap();
    out
}

/// Commit `data` as the old writer did: its content blobs, its data archive,
/// and an EMPTY metadata archive, signed by `key`.
fn old_commit(
    pile: &mut Pile,
    collection: triblespace::core::collection::records::CollectionHandle,
    key: &SigningKey,
    data: &TribleSet,
    content: &[Blob<UnknownBlob>],
) -> CollectionCommit {
    for blob in content {
        pile.put::<UnknownBlob, _>(blob.clone()).unwrap();
    }
    let data_handle = pile.put::<blobencodings::SimpleArchive, _>(data.clone()).unwrap();
    let empty = pile.put::<blobencodings::SimpleArchive, _>(TribleSet::new()).unwrap();
    let commit = CollectionCommit::sign(
        key,
        collection,
        inlineencodings::Handle::to_hash(data_handle),
        empty,
    );
    pile.insert(CollectionRecord::Commit(commit)).unwrap();
    commit
}

fn collection_commits(
    path: &Path,
    collection: triblespace::core::collection::records::CollectionHandle,
) -> Vec<CollectionCommit> {
    let mut pile = Pile::open(path).unwrap();
    pile.refresh().unwrap();
    let snapshot = pile.snapshot().unwrap();
    let commits = snapshot
        .select_records(&BTreeSet::from([CollectionRecordSelector::Collection(collection)]))
        .unwrap()
        .into_iter()
        .filter_map(|record| match record {
            CollectionRecord::Commit(commit) => Some(commit),
            _ => None,
        })
        .collect();
    drop(snapshot);
    pile.close().unwrap();
    commits
}

fn length(path: &Path) -> u64 {
    std::fs::metadata(path).unwrap().len()
}

/// The other mary declarations of an attribute the fixture's data uses, which
/// the new writer did not reference for that fixture. A commit's tribles do
/// not say which declaration wrote them, so the backfill describes an id by
/// every declaration of it (see `mary::metadata_backfill`).
fn sibling_declarations(label: &str) -> Vec<(&'static str, Fragment)> {
    match label {
        "parts" => vec![
            ("mary::format::attrs::index", mary::format::attrs::index.meta().clone()),
            ("mary::jsonfacts::attrs::index", mary::jsonfacts::attrs::index.meta().clone()),
        ],
        _ => vec![
            ("mary::jsonfacts::attrs::member", mary::jsonfacts::attrs::member.meta().clone()),
            ("mary::tokenizer::attrs::member", mary::tokenizer::attrs::member.meta().clone()),
            ("mary::tokenizer::attrs::model_name", mary::tokenizer::attrs::model_name.meta().clone()),
        ],
    }
}

/// The KEY test. Publish the fixtures through the NEW writer, re-sign each
/// commit's data with an EMPTY metadata archive in a fresh pile (the old
/// writer's shape), backfill, and compare the enriched archive with the one
/// the new writer produced: byte-identical to it united with the sibling
/// declarations' descriptions, and nothing else.
#[test]
fn backfill_reproduces_the_new_writers_metadata() {
    let fixture = Fixture::new();
    let (path, published) = publish_fixture(&fixture);
    let written = written(&path, &published);
    let catalog = Catalog::frozen();
    let key = SigningKey::from_bytes(&[0x6d; 32]);
    let other = SigningKey::from_bytes(&[0x5a; 32]);

    // Every attribute the new writer's fixture commits use is in the catalog,
    // and the catalog's description of it covers what the writer emitted.
    for w in &written {
        let attributes = metadata_backfill::data_attributes(&w.data);
        let missing: Vec<String> = attributes
            .iter()
            .filter(|a| catalog.get(**a).is_none())
            .map(|a| format!("{a:X}"))
            .collect();
        assert!(missing.is_empty(), "{}: catalog misses {missing:?}", w.label);
    }

    let old_path = fixture.0.join("old.pile");
    std::fs::File::create(&old_path).unwrap();
    let mut pile = Pile::open(&old_path).unwrap();
    let collection = mary::model_collection::model_collection_or_create(&mut pile, &key)
        .unwrap()
        .handle();
    let mut old = Vec::new();
    for w in &written {
        old.push(old_commit(&mut pile, collection, &key, &w.data, &w.content));
    }
    // Someone else's old commit: left alone.
    old_commit(&mut pile, collection, &other, &written[0].data, &written[0].content);
    // A commit whose data also uses an attribute nobody declares.
    let mut stranger = written[0].data.clone();
    stranger.insert(&Trible::force(
        &UNKNOWN_ENTITY,
        &UNKNOWN_ATTRIBUTE,
        &Inline::<inlineencodings::U256BE>::new([7; 32]),
    ));
    old_commit(&mut pile, collection, &key, &stranger, &written[0].content);
    pile.close().unwrap();

    // Dry run: writes nothing.
    let before = length(&old_path);
    let dry = metadata_backfill::run_path(&old_path, &key, &catalog, collection, false).unwrap();
    assert_eq!(length(&old_path), before, "a dry run wrote");
    assert!(!dry.applied);
    assert_eq!(dry.commits, 6);
    assert_eq!(dry.signer_commits, 5);
    assert_eq!(dry.other_signers, 1);
    assert_eq!(dry.enriched_commits, 5);
    assert_eq!(dry.new_commits, 5);
    assert!(dry.new_blobs >= 5);
    assert_eq!(dry.unknown.len(), 1);
    assert_eq!(dry.unknown[&UNKNOWN_ATTRIBUTE].commits, 1);
    assert_eq!(dry.unknown[&UNKNOWN_ATTRIBUTE].described, 0);
    assert_eq!(dry.unknown[&UNKNOWN_ATTRIBUTE].sample, Some(([7; 32], false)));

    // Apply: exactly the plan, exactly the bytes it announced.
    let applied = metadata_backfill::run_path(&old_path, &key, &catalog, collection, true).unwrap();
    assert!(applied.applied);
    assert_eq!(applied.new_commits, dry.new_commits);
    assert_eq!(applied.new_blobs, dry.new_blobs);
    assert_eq!(applied.enriched, dry.enriched);
    assert_eq!(length(&old_path) - before, applied.new_bytes);

    let commits = collection_commits(&old_path, collection);
    assert_eq!(commits.len(), 11);
    assert_eq!(
        commits.iter().filter(|c| c.public_key().raw == other.verifying_key().to_bytes()).count(),
        1,
        "the other signer's commit gained a successor"
    );
    let mut pile = Pile::open(&old_path).unwrap();
    pile.refresh().unwrap();
    let snapshot = pile.snapshot().unwrap();
    for w in &written {
        let successor = commits
            .iter()
            .find(|c| {
                c.data() == w.data_hash
                    && c.public_key().raw == key.verifying_key().to_bytes()
                    && !old.contains(c)
            })
            .unwrap_or_else(|| panic!("{}: no successor", w.label));
        successor.verify_strict().unwrap();
        assert_eq!(successor.collection(), collection);
        let enriched_blob: Blob<blobencodings::SimpleArchive> =
            snapshot.get(successor.metadata()).unwrap();
        let enriched: TribleSet = snapshot.get(successor.metadata()).unwrap();
        let identical = successor.metadata() == w.metadata_handle;
        let extra = enriched.difference(&w.metadata);
        let usage_entities: BTreeSet<Id> = find!(
            usage: Id,
            pattern!(&enriched, [{ ?usage @ metadata::tag: metadata::KIND_ATTRIBUTE_USAGE }])
        )
        .collect();
        println!(
            "GOLDEN {} identical={} new_writer={} ({} B, {} tribles) backfill={} ({} B, {} tribles) extra_tribles={} extra_usage_entities={}",
            w.label,
            identical,
            hex(&w.metadata_handle.raw),
            w.metadata_bytes,
            w.metadata.len(),
            hex(&successor.metadata().raw),
            enriched_blob.bytes.len(),
            enriched.len(),
            extra.len(),
            extra.iter().map(|t| *t.e()).collect::<BTreeSet<Id>>().len(),
        );
        // Name the extra usage records: `source_module::name`.
        let mut extra_usages = Vec::new();
        for (module, name) in find!(
            (module: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
             name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(&enriched, [{ _?usage @
                metadata::source_module: ?module,
                metadata::name: ?name,
                metadata::attribute: _?attribute,
            }])
        ) {
            let module: anybytes::View<str> = snapshot.get(module).unwrap();
            let name: anybytes::View<str> = snapshot.get(name).unwrap();
            extra_usages.push(format!("{}::{}", &*module, &*name));
        }
        let mut usages_new = Vec::new();
        for (module, name) in find!(
            (module: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
             name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(&w.metadata, [{ _?usage @
                metadata::source_module: ?module,
                metadata::name: ?name,
                metadata::attribute: _?attribute,
            }])
        ) {
            let module: anybytes::View<str> = snapshot.get(module).unwrap();
            let name: anybytes::View<str> = snapshot.get(name).unwrap();
            usages_new.push(format!("{}::{}", &*module, &*name));
        }
        extra_usages.retain(|u| !usages_new.contains(u));
        extra_usages.sort();
        println!("GOLDEN {} usage records only the backfill adds: {:?}", w.label, extra_usages);
        let siblings = sibling_declarations(w.label);
        let mut names: Vec<String> = siblings.iter().map(|(n, _)| n.to_string()).collect();
        names.sort();
        assert_eq!(extra_usages, names, "{}: unexpected extra usage records", w.label);
        // Byte identity: the new writer's archive united with exactly those
        // declarations' metas IS the backfilled archive.
        let mut expected = w.metadata.clone();
        for (_, meta) in &siblings {
            expected.union(meta.facts().clone());
            expected.union(meta.metafacts().clone());
        }
        let expected_blob: Blob<blobencodings::SimpleArchive> = expected.to_blob();
        assert_eq!(expected_blob.get_handle(), successor.metadata(), "{}: golden mismatch", w.label);
        assert_eq!(expected_blob.bytes, enriched_blob.bytes, "{}: golden bytes differ", w.label);
        println!("GOLDEN {} byte-identical to new writer + {} sibling declarations: true", w.label, siblings.len());
        // Nothing the new writer wrote is missing.
        assert!(
            w.metadata.difference(&enriched).is_empty(),
            "{}: the backfill lacks facts the new writer wrote",
            w.label
        );
        // Anything beyond it is a usage record of another mary declaration of
        // an attribute this data uses: never a typing fact.
        let attributes = metadata_backfill::data_attributes(&w.data);
        for trible in extra.iter() {
            assert!(
                usage_entities.contains(trible.e()),
                "{}: extra fact on {:X} is not a usage record",
                w.label,
                trible.e()
            );
        }
        for (usage, attribute) in find!(
            (usage: Id, attribute: Id),
            pattern!(&extra, [{ ?usage @ metadata::attribute: ?attribute }])
        ) {
            assert!(attributes.contains(&attribute), "{}: usage {usage:X} names {attribute:X}", w.label);
        }
        if identical {
            assert_eq!(enriched_blob.bytes, snapshot.get::<Blob<blobencodings::SimpleArchive>, _>(w.metadata_handle).unwrap().bytes);
        }
        // Closure: every blob the new writer stored for its metadata is
        // resident beside the backfilled one.
        for raw in &w.described {
            let handle: Inline<inlineencodings::Handle<UnknownBlob>> = Inline::new(*raw);
            assert!(snapshot.metadata(handle).unwrap().is_some(), "{}: blob {} missing", w.label, hex(raw));
        }
    }
    drop(snapshot);
    pile.close().unwrap();

    // A second apply finds every successor present and writes nothing.
    let settled = length(&old_path);
    let second = metadata_backfill::run_path(&old_path, &key, &catalog, collection, true).unwrap();
    assert_eq!(second.new_commits, 0);
    assert_eq!(second.new_blobs, 0);
    assert_eq!(second.new_bytes, 0);
    assert_eq!(length(&old_path), settled, "a second apply wrote");
    assert_eq!(second.signer_commits, 10);
    assert_eq!(second.present_commits, 5);
    assert_eq!(second.current_commits, 5);
}

/// An archive whose bytes fail their own hash stops the run, naming it.
#[test]
fn a_corrupt_archive_stops_the_run_naming_it() {
    for role in ["data", "metadata"] {
        let fixture = Fixture::new();
        let path = fixture.0.join("corrupt.pile");
        std::fs::File::create(&path).unwrap();
        let key = SigningKey::from_bytes(&[0x6d; 32]);
        let mut pile = Pile::open(&path).unwrap();
        let collection = mary::model_collection::model_collection_or_create(&mut pile, &key)
            .unwrap()
            .handle();
        let marker = [0xA5_u8; 32];
        let mut damaged = TribleSet::new();
        damaged.insert(&Trible::force(
            &UNKNOWN_ENTITY,
            &UNKNOWN_ATTRIBUTE,
            &Inline::<inlineencodings::U256BE>::new(marker),
        ));
        let damaged = pile.put::<blobencodings::SimpleArchive, _>(damaged).unwrap();
        let plain = pile.put::<blobencodings::SimpleArchive, _>(TribleSet::new()).unwrap();
        let (data, meta) = if role == "data" { (damaged, plain) } else { (plain, damaged) };
        pile.insert(CollectionRecord::Commit(CollectionCommit::sign(
            &key,
            collection,
            inlineencodings::Handle::to_hash(data),
            meta,
        )))
        .unwrap();
        pile.close().unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.windows(32).position(|w| w == marker).expect("value stored verbatim");
        bytes[at] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let before = length(&path);
        for apply in [false, true] {
            let error = metadata_backfill::run_path(&path, &key, &Catalog::frozen(), collection, apply)
                .expect_err("corrupt archive must not be skipped");
            let message = format!("{error:#}");
            assert!(
                message.contains(&format!("{role} archive {}", hex(&damaged.raw)))
                    && message.contains("fails its hash check"),
                "{message}"
            );
        }
        assert_eq!(length(&path), before);
    }
}
