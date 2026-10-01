//! Byte-level checks of native BF16 publication through the ordinary importer.

#![cfg(feature = "import")]

use ed25519_dalek::SigningKey;
use mary::format::attrs;
use mary::ingest::LeafDtype;
use mary::leaf::Elem;
use mary::persist;
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::path::{Path, PathBuf};
use triblespace::core::collection::CollectionRead;
use triblespace::prelude::*;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("mary-bf16-import-{}", fucid().id));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn pile(&self) -> (PathBuf, Pile) {
        let path = self.0.join("models.pile");
        std::fs::File::create(&path).unwrap();
        let pile = Pile::open(&path).unwrap();
        (path, pile)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_bf16(path: &Path, name: &str, shape: Vec<usize>, bits: &[u16]) {
    let raw: Vec<u8> = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
    serialize_to_file(
        [(name, TensorView::new(Dtype::BF16, shape, &raw).unwrap())],
        &None,
        path,
    )
    .unwrap();
}

#[test]
fn sharded_native_bf16_root_is_exact_idempotent_and_owns_reopened_views() {
    let fixture = Fixture::new();
    let weights = fixture.0.join("weights");
    std::fs::create_dir(&weights).unwrap();
    let matrix = [
        0x0000, 0x8000, 0x0001, 0x007f, 0x0080, 0x7f7f, 0x7f80, 0xff81,
    ];
    let vector = [
        0x8001, 0x807f, 0x8080, 0xff7f, 0xff80, 0x7f81, 0x7fc1, 0xffc1,
    ];
    write_bf16(
        &weights.join("a.safetensors"),
        "block.weight",
        vec![2, 4],
        &matrix,
    );
    write_bf16(
        &weights.join("b.safetensors"),
        "norm.weight",
        vec![8],
        &vector,
    );
    let (path, mut pile) = fixture.pile();
    let key = SigningKey::from_bytes(&[0x65; 32]);
    let first = persist::import_model_to_collection(
        &mut pile,
        &key,
        &weights,
        LeafDtype::Bf16,
        "fixture/native-bf16",
        "native",
    )
    .unwrap();
    let first_size = std::fs::metadata(&path).unwrap().len();
    let repeated = persist::import_model_to_collection(
        &mut pile,
        &key,
        &weights,
        LeafDtype::Bf16,
        "fixture/native-bf16",
        "native",
    )
    .unwrap();
    assert_eq!(first, repeated);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), first_size);
    pile.close().unwrap();

    let snapshot = mary::model_collection::load_model_collection_local_latest(&path).unwrap();
    assert_eq!(snapshot.support().len(), 1);
    let root = first.0;
    assert_eq!(
        mary::selection::select_model_root(
            snapshot.facts(),
            snapshot.store(),
            mary::selection::ModelSelector::Source {
                source: "fixture/native-bf16",
                quantization: "native",
            },
        )
        .unwrap(),
        root,
    );
    let mut provenance: Vec<String> = find!(
        (name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
        pattern!(snapshot.facts(), [{ root @ attrs::model_name: ?name }])
    )
    .map(|(name,)| mary::ingest::read_string(snapshot.store(), name))
    .collect();
    provenance.sort();
    assert_eq!(provenance, ["a.safetensors", "b.safetensors"]);
    let index =
        mary::selection::index_keymap_for_root(snapshot.facts(), snapshot.store(), root).unwrap();
    assert_eq!(index.len(), 2);
    assert_eq!(index["block.weight"].dims(), &[2, 4]);
    assert_eq!(index["norm.weight"].dims(), &[8]);
    let mut views = Vec::new();
    for name in ["block.weight", "norm.weight"] {
        let leaf = &index[name];
        assert_eq!(leaf.elem(), Elem::Bf16);
        assert_eq!(leaf.payload().len(), 16);
        assert!(
            leaf.payload()
                .clone()
                .downcast_to_owner::<memmap2::MmapRaw>()
                .is_ok()
        );
        let view = leaf.view_bf16().unwrap();
        assert_eq!(view.as_ptr() as *const u8, leaf.payload().as_ptr());
        views.push(view);
    }
    drop(index);
    drop(snapshot);
    assert_eq!(
        views[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        matrix
    );
    assert_eq!(
        views[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vector
    );
}

#[test]
fn duplicate_or_wrong_dtype_shards_publish_no_collection_commit() {
    for duplicate in [false, true] {
        let fixture = Fixture::new();
        let weights = fixture.0.join("weights");
        std::fs::create_dir(&weights).unwrap();
        write_bf16(
            &weights.join("a.safetensors"),
            "shared.weight",
            vec![1],
            &[0x7f7f],
        );
        let raw = [0u8; 4];
        let (name, dtype, bytes) = if duplicate {
            ("shared.weight", Dtype::BF16, &raw[..2])
        } else {
            ("other.weight", Dtype::F32, &raw[..])
        };
        serialize_to_file(
            [(name, TensorView::new(dtype, vec![1], bytes).unwrap())],
            &None,
            &weights.join("b.safetensors"),
        )
        .unwrap();
        let (path, mut pile) = fixture.pile();
        let error = persist::import_model_to_collection(
            &mut pile,
            &SigningKey::from_bytes(&[0x66; 32]),
            &weights,
            LeafDtype::Bf16,
            "fixture/rejected",
            "native",
        )
        .unwrap_err();
        let expected = if duplicate {
            "duplicate tensor name"
        } else {
            "requires BF16"
        };
        assert!(error.to_string().contains(expected), "{error:#}");
        let snapshot = pile.snapshot().unwrap();
        assert_eq!(snapshot.records().unwrap().count(), 0);
        drop(snapshot);
        pile.close().unwrap();
        let error = mary::model_collection::load_model_collection_local_latest(&path).unwrap_err();
        assert!(
            error.to_string().contains("no collection named"),
            "{error:#}"
        );
    }
}

#[test]
fn failed_native_append_preserves_the_preexisting_collection() {
    let fixture = Fixture::new();
    let weights = fixture.0.join("weights");
    std::fs::create_dir(&weights).unwrap();
    write_bf16(
        &weights.join("a.safetensors"),
        "first.weight",
        vec![1],
        &[0x0001],
    );
    let (path, mut pile) = fixture.pile();
    let key = SigningKey::from_bytes(&[0x67; 32]);
    persist::import_model_to_collection(
        &mut pile,
        &key,
        &weights,
        LeafDtype::Bf16,
        "fixture/existing",
        "native",
    )
    .unwrap();
    let before = mary::model_collection::snapshot_model_collection_local_latest(&mut pile).unwrap();
    serialize_to_file(
        [(
            "wrong.weight",
            TensorView::new(Dtype::F16, vec![1], &[0u8; 2]).unwrap(),
        )],
        &None,
        &weights.join("b.safetensors"),
    )
    .unwrap();
    assert!(
        persist::import_model_to_collection(
            &mut pile,
            &key,
            &weights,
            LeafDtype::Bf16,
            "fixture/rejected",
            "native",
        )
        .is_err()
    );
    pile.close().unwrap();
    let after = mary::model_collection::load_model_collection_local_latest(&path).unwrap();
    assert_eq!(after.support(), before.support());
    assert_eq!(after.facts(), before.facts());
}

#[test]
fn bf16_rejects_decoded_containers_before_reading_their_tensor_data() {
    for (format, filename, magic) in [
        (
            mary::formats::WeightFormat::Gguf,
            "weights.gguf",
            &b"GGUF"[..],
        ),
        (
            mary::formats::WeightFormat::Pickle,
            "pytorch_model.bin",
            &b"not a zip"[..],
        ),
    ] {
        let fixture = Fixture::new();
        let weights = fixture.0.join("weights");
        std::fs::create_dir(&weights).unwrap();
        // Detection can identify the format, but its decoder would fail.
        std::fs::write(weights.join(filename), magic).unwrap();
        let (_, mut pile) = fixture.pile();
        let error = persist::ingest_model_fragment(
            &mut pile,
            &weights,
            LeafDtype::Bf16,
            "fixture/unsupported",
            "native",
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("requires safetensors"),
            "{error:#}"
        );
        // Explicit-format entrypoint rejects before even opening the file.
        let error = persist::ingest_weight_file_filtered_fragment(
            &mut pile,
            &fixture.0.join("missing.weights"),
            format,
            LeafDtype::Bf16,
            "fixture/unsupported",
            "native",
            |_| true,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("requires safetensors"),
            "{error:#}"
        );
        assert_eq!(pile.snapshot().unwrap().records().unwrap().count(), 0);
        pile.close().unwrap();
    }
}

#[test]
fn filtered_native_entrypoints_keep_only_selected_bf16_tensors() {
    for per_file_root in [false, true] {
        let fixture = Fixture::new();
        let weights = fixture.0.join("components.safetensors");
        let native = [0x01u8, 0x00, 0x7f, 0x7f];
        serialize_to_file(
            [
                (
                    "kept.weight",
                    TensorView::new(Dtype::BF16, vec![2], &native).unwrap(),
                ),
                (
                    "other.weight",
                    TensorView::new(Dtype::F32, vec![1], &[0u8; 4]).unwrap(),
                ),
            ],
            &None,
            &weights,
        )
        .unwrap();
        let (path, mut pile) = fixture.pile();
        let key = SigningKey::from_bytes(&[0x68; 32]);
        if per_file_root {
            pile.close().unwrap();
            persist::persist_safetensors_file_filtered_to_pile(
                &weights,
                "filtered/component",
                &path,
                &key,
                LeafDtype::Bf16,
                |name| name.starts_with("kept."),
            )
            .unwrap();
        } else {
            persist::import_safetensors_file_filtered_to_collection(
                &mut pile,
                &key,
                &weights,
                LeafDtype::Bf16,
                "fixture/filtered",
                "native",
                |name| name.starts_with("kept."),
            )
            .unwrap();
            let before =
                mary::model_collection::snapshot_model_collection_local_latest(&mut pile).unwrap();
            assert!(
                persist::import_safetensors_file_filtered_to_collection(
                    &mut pile,
                    &key,
                    &weights,
                    LeafDtype::Bf16,
                    "fixture/empty",
                    "native",
                    |_| false,
                )
                .is_err()
            );
            assert!(
                persist::import_safetensors_file_filtered_to_collection(
                    &mut pile,
                    &key,
                    &weights,
                    LeafDtype::Bf16,
                    "fixture/wrong",
                    "native",
                    |_| true,
                )
                .is_err()
            );
            let after =
                mary::model_collection::snapshot_model_collection_local_latest(&mut pile).unwrap();
            assert_eq!(before.support(), after.support());
            pile.close().unwrap();
        }
        let snapshot = mary::model_collection::load_model_collection_local_latest(&path).unwrap();
        let index = mary::leaf::index_by_name(snapshot.facts(), snapshot.store()).unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index["kept.weight"].elem(), Elem::Bf16);
        assert_eq!(&index["kept.weight"].payload()[..], &native);
    }
}

#[test]
fn per_file_native_roots_work_through_both_sharded_persist_entrypoints() {
    for directory in [false, true] {
        let fixture = Fixture::new();
        let weights = fixture.0.join("weights");
        std::fs::create_dir(&weights).unwrap();
        let a = weights.join("a.safetensors");
        let b = weights.join("b.safetensors");
        write_bf16(&a, "a.weight", vec![1], &[0x7f7f]);
        write_bf16(&b, "b.weight", vec![1], &[0x0001]);
        let path = fixture.0.join("models.pile");
        let key = SigningKey::from_bytes(&[0x69; 32]);
        if directory {
            persist::persist_safetensors_to_pile(&weights, &path, &key, LeafDtype::Bf16).unwrap();
        } else {
            persist::persist_safetensors_files_to_pile(
                &[(a, "a.safetensors".into()), (b, "b.safetensors".into())],
                &path,
                &key,
                LeafDtype::Bf16,
            )
            .unwrap();
        }
        let snapshot = mary::model_collection::load_model_collection_local_latest(&path).unwrap();
        let index = mary::leaf::index_by_name(snapshot.facts(), snapshot.store()).unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index["a.weight"].view_bf16().unwrap()[0].to_bits(), 0x7f7f);
        assert_eq!(index["b.weight"].view_bf16().unwrap()[0].to_bits(), 0x0001);
        let mut labels: Vec<String> = find!(
            (name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(snapshot.facts(), [{ _?root @ attrs::model_name: ?name }])
        )
        .map(|(name,)| mary::ingest::read_string(snapshot.store(), name))
        .collect();
        labels.sort();
        assert_eq!(labels, ["a.safetensors", "b.safetensors"]);
    }
}
