//! Native BF16 storage checks; no model arithmetic or device execution.

#![cfg(feature = "import")]

use anybytes::Bytes;
use mary::format::attrs;
use mary::ingest::{LeafDtype, ingest_bf16_members, ingest_members, ingest_tensors};
use mary::leaf::{self, Elem};
use safetensors::tensor::{Dtype, TensorView, serialize};
use triblespace::core::blob::Blob;
use triblespace::core::blob::encodings::tensor::elements::{BF16, F16, F32};
use triblespace::core::blob::encodings::tensor::{TENSOR_HEADER_LEN, Tensor};
use triblespace::prelude::*;

fn container(dtype: Dtype, shape: Vec<usize>, data: &[u8]) -> Bytes {
    Bytes::from_source(
        serialize(
            [(
                "linear.weight",
                TensorView::new(dtype, shape, data).unwrap(),
            )],
            &None,
        )
        .unwrap(),
    )
}

#[test]
fn every_bf16_bit_pattern_survives_a_reopened_pile_as_a_native_view() {
    // Covers signed zero, both subnormal boundaries, the full exponent range,
    // infinities, and every quiet/signalling NaN payload without evaluating it.
    let bits: Vec<u16> = (u16::MIN..=u16::MAX).collect();
    let raw: Vec<u8> = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
    let source = container(Dtype::BF16, vec![256, 256], &raw);
    let dir = std::env::temp_dir().join(format!("mary-native-bf16-{}", fucid().id));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("fixture.pile");
    std::fs::File::create(&path).unwrap();
    let mut pile = Pile::open(&path).unwrap();
    pile.refresh().unwrap();
    let (members, facts) = ingest_bf16_members(&source, &mut pile, |_| true).unwrap();
    assert_eq!(members.len(), 1);
    let graph = pile.put::<blobencodings::SimpleArchive, _>(facts).unwrap();
    pile.close().unwrap();
    drop(source);

    let mut pile = Pile::open(&path).unwrap();
    pile.refresh().unwrap();
    let reader = SnapshotSource::snapshot(&mut pile).unwrap();
    let facts: TribleSet = reader.get(graph).unwrap();
    let (weight, handle) = find!(
        (weight: Id, h: Inline<inlineencodings::Handle<Tensor<BF16, 2>>>),
        pattern!(&facts, [
            { _?module @ attrs::weight: ?weight },
            { ?weight @ leaf::leaf::<BF16, 2>(): ?h },
        ])
    )
    .next()
    .unwrap();
    let stored: Blob<Tensor<BF16, 2>> = reader.get(handle).unwrap();
    let blob_address = stored.bytes.as_ptr() as usize;
    let view = leaf::read_leaf::<BF16, 2>(stored).unwrap();
    assert_eq!(view.dims(), &[256, 256]);
    assert_eq!(&view.payload()[..], &raw);
    assert_eq!(
        view.payload().as_ptr() as usize,
        blob_address + TENSOR_HEADER_LEN
    );
    assert_eq!(view.payload().as_ptr() as usize % 256, 0);

    let resolved = leaf::resolve(&facts, &reader, weight).unwrap().unwrap();
    assert_eq!(resolved.elem(), Elem::Bf16);
    assert_eq!(resolved.dims(), &[256, 256]);
    assert!(resolved.view_f16().is_none());
    assert!(resolved.view_f32().is_none());
    let native = resolved.view_bf16().unwrap();
    assert_eq!(native.as_ptr() as *const u8, resolved.payload().as_ptr());
    assert_eq!(native.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    assert!(!exists!(
        pattern!(&facts, [{ weight @ leaf::leaf::<F16, 2>(): _?h }])
    ));
    assert!(!exists!(
        pattern!(&facts, [{ weight @ leaf::leaf::<BF16, 1>(): _?h }])
    ));
    let native_address = native.as_ptr();
    drop(resolved);
    drop(view);
    drop(reader);
    pile.close().unwrap();
    // The view owns its mapping through Bytes, independently of every reader
    // and of the mutable pile that opened it.
    assert_eq!(native.as_ptr(), native_address);
    assert_eq!(native.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    drop(native);
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn bf16_dispatch_preserves_every_supported_rank_and_rejects_wrong_lengths() {
    let mut blobs = MemoryBlobStore::new();
    for rank in 0..=6 {
        let dims = vec![1; rank];
        let fragment = leaf::put_leaf(
            &mut blobs,
            Elem::Bf16,
            &dims,
            Bytes::from_source(vec![0x7fu8, 0x7f]),
            "native rank fixture",
        )
        .unwrap();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let loaded = leaf::resolve(fragment.facts(), &reader, fragment.root().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(loaded.elem(), Elem::Bf16);
        assert_eq!(loaded.dims(), &dims);
        assert_eq!(loaded.view_bf16().unwrap()[0].to_bits(), 0x7f7f);
    }
    assert_ne!(leaf::leaf::<BF16, 2>().id(), leaf::leaf::<F16, 2>().id());
    assert_ne!(leaf::leaf::<BF16, 2>().id(), leaf::leaf::<F32, 2>().id());
    assert!(
        leaf::typed_leaf_attrs()
            .iter()
            .any(|(name, id)| { name == "leaf.bf16.2" && *id == leaf::leaf::<BF16, 2>().id() })
    );
    assert!(
        leaf::put_leaf(
            &mut blobs,
            Elem::Bf16,
            &[2],
            Bytes::from_source(vec![0u8; 2]),
            "short"
        )
        .is_err()
    );
    assert!(
        leaf::put_leaf(
            &mut blobs,
            Elem::Bf16,
            &[1; 7],
            Bytes::from_source(vec![0u8; 2]),
            "rank 7"
        )
        .is_err()
    );
}

#[test]
fn native_import_rejects_other_dtypes_and_never_consumes_an_f32_iterator() {
    for (dtype, width) in [(Dtype::F16, 2), (Dtype::F32, 4), (Dtype::U16, 2)] {
        let source = container(dtype, vec![1], &vec![0; width]);
        let mut blobs = MemoryBlobStore::new();
        let error = ingest_bf16_members(&source, &mut blobs, |_| true).unwrap_err();
        assert!(error.to_string().contains("requires BF16"), "{error}");
        assert!(ingest_members(&source, &mut blobs, LeafDtype::Bf16, |_| true).is_err());
        assert!(
            ingest_bf16_members(&source, &mut blobs, |_| false)
                .unwrap()
                .0
                .is_empty()
        );
    }
    let mut blobs = MemoryBlobStore::new();
    let tensors = std::iter::from_fn(|| -> Option<(String, Vec<f32>, Vec<usize>)> {
        panic!("the native importer must not consume widened tensors")
    });
    assert!(ingest_tensors(tensors, &mut blobs, LeafDtype::Bf16).is_err());
}

#[test]
fn slice_and_mapping_imports_have_identical_native_members() {
    let raw = [0x01, 0x00, 0x7f, 0x7f, 0x01, 0x7f, 0x81, 0xff];
    let source = container(Dtype::BF16, vec![2, 2], &raw);
    let mut blobs = MemoryBlobStore::new();
    let borrowed = ingest_bf16_members(&source, &mut blobs, |_| true).unwrap();
    let copied = ingest_members(&source, &mut blobs, LeafDtype::Bf16, |_| true).unwrap();
    assert_eq!(borrowed, copied);
    let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
    let index = leaf::index_typed_all(&borrowed.1, &reader);
    assert_eq!(index.len(), 1);
    let loaded = index.values().next().unwrap();
    assert_eq!(loaded.elem(), Elem::Bf16);
    assert_eq!(&loaded.payload()[..], &raw);
}

#[test]
fn native_typed_query_accepts_opaque_ids_and_other_annotations() {
    let mut blobs = MemoryBlobStore::new();
    let id = fucid();
    let mut facts = leaf::put_leaf_as(
        &mut blobs,
        &id,
        Elem::Bf16,
        &[1],
        Bytes::from_source(vec![0x80u8, 0x3f]),
        "opaque",
    )
    .unwrap()
    .into_facts();
    facts += entity! { &id @ attrs::kind*: ["weight", "annotation"] };
    facts += leaf::put_leaf_as(
        &mut blobs,
        &id,
        Elem::F32,
        &[1],
        Bytes::from_source(vec![0u8; 4]),
        "other encoding",
    )
    .unwrap()
    .into_facts();
    let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
    // A native reader asks only for its type and follows the stored ID. It
    // does not impose the legacy resolve() helper's cross-dtype cardinality.
    let (handle,) = find!(
        (h: Inline<inlineencodings::Handle<Tensor<BF16, 1>>>),
        pattern!(&facts, [{ id.id @ leaf::leaf::<BF16, 1>(): ?h }])
    )
    .next()
    .unwrap();
    let stored: Blob<Tensor<BF16, 1>> = reader.get(handle).unwrap();
    let native = leaf::read_leaf::<BF16, 1>(stored).unwrap();
    assert_eq!(&native.payload()[..], &[0x80, 0x3f]);
}
