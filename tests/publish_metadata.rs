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
