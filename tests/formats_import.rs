//! The `mary import` storage contract: a tiny synthetic safetensors model is
//! published into the native collection and read through both supported
//! selectors. Optional cached fixtures exercise non-safetensors decoders through
//! the same collection front door.

#![cfg(feature = "import")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ed25519_dalek::SigningKey;
use mary::selection::ModelSelector;
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use triblespace::core::blob::MemoryBlobStore;
use triblespace::core::collection::CollectionRead;
use triblespace::core::repo::SnapshotSource;
use triblespace::core::repo::pile::Pile;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TempFixture {
    dir: PathBuf,
}

impl TempFixture {
    fn new(label: &str) -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "mary-formats-{label}-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).unwrap();
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TempFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn native_import_is_exact_selectable_and_byte_idempotent() {
    let fixture = TempFixture::new("native-import");
    let weights = fixture.path().join("weights");
    std::fs::create_dir(&weights).unwrap();
    let weights_file = weights.join("model.safetensors");

    let weight = vec![1.0_f32, -2.0, 3.5, 0.25];
    let bias = vec![0.5_f32, -0.5];
    let weight_bytes = f32_bytes(&weight);
    let bias_bytes = f32_bytes(&bias);
    serialize_to_file(
        [
            (
                "linear.bias",
                TensorView::new(Dtype::F32, vec![2], &bias_bytes).unwrap(),
            ),
            (
                "linear.weight",
                TensorView::new(Dtype::F32, vec![2, 2], &weight_bytes).unwrap(),
            ),
        ],
        &None,
        &weights_file,
    )
    .unwrap();

    let pile_path = fixture.path().join("models.pile");
    std::fs::File::create(&pile_path).unwrap();
    let signing_key = SigningKey::from_bytes(&[0x51; 32]);
    let mut pile = Pile::open(&pile_path).unwrap();
    let first = mary::persist::import_model_to_collection(
        &mut pile,
        &signing_key,
        &weights,
        mary::ingest::LeafDtype::F32,
        "fixture/source",
        "native",
    )
    .unwrap();
    let bytes_after_first = std::fs::metadata(&pile_path).unwrap().len();
    let repeated = mary::persist::import_model_to_collection(
        &mut pile,
        &signing_key,
        &weights,
        mary::ingest::LeafDtype::F32,
        "fixture/source",
        "native",
    )
    .unwrap();
    let bytes_after_retry = std::fs::metadata(&pile_path).unwrap().len();
    assert_eq!(repeated, first, "stable signer must reproduce the commit");
    assert_eq!(first.1.to_bytes().len(), 224, "a COMMIT is seven 32-byte fields");
    assert_eq!(
        bytes_after_retry, bytes_after_first,
        "an identical retry must append no bytes"
    );
    pile.close().unwrap();

    // Construct the exact expected graph through the same format-agnostic
    // graph primitives, but in independent storage. This checks that the
    // native collection member contains no message or other ambient facts;
    // unrelated non-collection record kinds are intentionally outside what a
    // collection snapshot can observe.
    let mut expected_blobs = MemoryBlobStore::new();
    let (members, member_facts) = mary::ingest::ingest_tensors(
        vec![
            ("linear.bias".to_owned(), bias.clone(), vec![2]),
            ("linear.weight".to_owned(), weight.clone(), vec![2, 2]),
        ]
        .into_iter(),
        &mut expected_blobs,
        mary::ingest::LeafDtype::F32,
    )
    .unwrap();
    let expected = mary::ingest::build_model_root(
        &mut expected_blobs,
        "fixture/source",
        "native",
        members,
        member_facts,
        &["model.safetensors".to_owned()],
    )
    .unwrap();
    assert_eq!(expected.root(), Some(first.0));

    let snapshot = mary::model_collection::load_model_collection_local_latest(&pile_path).unwrap();
    assert_eq!(snapshot.support().len(), 1);
    assert!(
        snapshot
            .support()
            .contains(triblespace::prelude::inlineencodings::Handle::<
                triblespace::prelude::blobencodings::SimpleArchive,
            >::from_hash(first.1.data()))
    );
    assert_eq!(snapshot.facts(), expected.facts());

    let by_source = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Source {
            source: "fixture/source",
            quantization: "native",
        },
    )
    .unwrap();
    let by_root = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Root(first.0),
    )
    .unwrap();
    assert_eq!(by_root, by_source);
    assert_eq!(by_source["linear.bias"], (bias, vec![2]));
    assert_eq!(by_source["linear.weight"], (weight, vec![2, 2]));
}

#[test]
fn duplicate_tensor_names_across_files_publish_no_collection_commit() {
    let fixture = TempFixture::new("duplicate-tensor-names");
    let weights = fixture.path().join("weights");
    std::fs::create_dir(&weights).unwrap();
    let first_values = f32_bytes(&[1.0_f32]);
    let second_values = f32_bytes(&[2.0_f32]);
    serialize_to_file(
        [(
            "shared.weight",
            TensorView::new(Dtype::F32, vec![1], &first_values).unwrap(),
        )],
        &None,
        &weights.join("a.safetensors"),
    )
    .unwrap();
    serialize_to_file(
        [(
            "shared.weight",
            TensorView::new(Dtype::F32, vec![1], &second_values).unwrap(),
        )],
        &None,
        &weights.join("b.safetensors"),
    )
    .unwrap();

    let pile_path = fixture.path().join("models.pile");
    std::fs::File::create(&pile_path).unwrap();
    let mut pile = Pile::open(&pile_path).unwrap();
    let error = mary::persist::import_model_to_collection(
        &mut pile,
        &SigningKey::from_bytes(&[0x53; 32]),
        &weights,
        mary::ingest::LeafDtype::F32,
        "fixture/duplicate",
        "native",
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("duplicate tensor name"),
        "{error:#}"
    );

    let snapshot = pile.snapshot().unwrap();
    assert_eq!(
        snapshot.records().unwrap().count(),
        0,
        "failed import must publish no model data COMMIT"
    );
    drop(snapshot);
    let error = mary::model_collection::snapshot_model_collection_local_latest(&mut pile)
        .expect_err("an unreferenced descriptor blob is not a discoverable collection");
    assert!(
        error.to_string().contains("no collection named"),
        "{error:#}"
    );
    pile.close().unwrap();
}

/// Locate a cached HF snapshot dir for `id`, or `None` if not downloaded.
fn hf_snapshot(id: &str) -> Option<PathBuf> {
    let hf_home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
            PathBuf::from(home).join(".cache/huggingface")
        });
    let repo = format!("models--{}", id.replace('/', "--"));
    let snaps = hf_home.join("hub").join(repo).join("snapshots");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .find(|d| mary::formats::detect_format(d).is_ok())
}

#[test]
#[ignore = "needs hf-internal-testing/tiny-random-MistralForCausalLM in the HF cache"]
fn pytorch_bin_import_roundtrip() {
    let dir = match hf_snapshot("hf-internal-testing/tiny-random-MistralForCausalLM") {
        Some(d) => d,
        None => {
            eprintln!("skip: tiny-random-MistralForCausalLM not cached");
            return;
        }
    };
    let (fmt, files) = mary::formats::detect_format(&dir).unwrap();
    assert_eq!(
        fmt,
        mary::formats::WeightFormat::Pickle,
        "should detect pickle"
    );
    assert_eq!(files.len(), 1);

    let fixture = TempFixture::new("pickle-import");
    let tmp = fixture.path().join("models.pile");
    std::fs::File::create(&tmp).unwrap();
    let mut pile = Pile::open(&tmp).unwrap();
    let signing_key = SigningKey::from_bytes(&[0x52; 32]);
    let (root, _commit) = mary::persist::import_model_to_collection(
        &mut pile,
        &signing_key,
        &dir,
        mary::ingest::LeafDtype::F32,
        "mistral-tiny",
        "native",
    )
    .unwrap();
    pile.close().unwrap();
    eprintln!("imported root {root:X}");

    let snapshot = mary::model_collection::load_model_collection_local_latest(&tmp).unwrap();
    let km = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Source {
            source: "mistral-tiny",
            quantization: "native",
        },
    )
    .unwrap();
    // The tiny Mistral has these tensors with these exact shapes.
    let (embed, eshape) = &km["model.embed_tokens.weight"];
    assert_eq!(eshape, &[32000, 32], "embed shape");
    assert_eq!(embed.len(), 32000 * 32);
    let (ln, lshape) = &km["model.layers.0.input_layernorm.weight"];
    assert_eq!(lshape, &[32]);
    // input_layernorm initializes to all-ones in this fixture.
    for &v in ln.iter() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "layernorm weight should be 1.0, got {v}"
        );
    }
    // Finite, non-degenerate weights everywhere.
    assert!(embed.iter().all(|v| v.is_finite()));
    assert!(embed.iter().any(|&v| v != 0.0));
}
