//! Synthetic CPU storage/graph gate only: no checkpoint payload or runtime.
//! Direct modules keep this test independent of model declaration integration.
#![allow(dead_code)]

use mary::{format, leaf, model_collection};
#[path = "../src/models/nemotron_h/ingress/native.rs"]
pub mod native;
#[path = "../src/models/nemotron_h/ingress/packing.rs"]
pub mod packing;
mod ingress {
    pub use super::native;
}
#[path = "../src/models/nemotron_h/pile.rs"]
mod native_pile;

use anybytes::Bytes;
use anyhow::Result;
use ed25519_dalek::SigningKey;
use format::attrs;
use packing::{Dtype, Plane, pack_fp8, pack_nvfp4};
use triblespace::core::blob::encodings::tensor::elements::{BF16, E4M3, F32, NVFP4};
use triblespace::core::collection::{AdmissionPolicy, CollectionPolicy, CollectionStoreExt};
use triblespace::prelude::*;

struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("nemotron-native-{}.pile", genid()));
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn plane<'a>(dtype: Dtype, shape: &'a [usize], bytes: &'a [u8]) -> Plane<'a> {
    Plane {
        dtype,
        shape,
        bytes,
    }
}

fn nv(code: u8, input: f32, global: f32) -> packing::PackedNvfp4 {
    pack_nvfp4(
        [2, 16],
        plane(Dtype::U8, &[2, 8], &[code; 16]),
        plane(Dtype::E4m3Fn, &[2, 1], &[0x38, 0x40]),
        plane(Dtype::F32, &[], &global.to_le_bytes()),
        plane(Dtype::F32, &[1], &input.to_le_bytes()),
    )
    .unwrap()
}

fn fp(code: u8, input: f32, weight: f32) -> packing::PackedFp8 {
    pack_fp8(
        [2, 3],
        plane(Dtype::E4m3Fn, &[2, 3], &[code; 6]),
        plane(Dtype::F32, &[1], &weight.to_le_bytes()),
        plane(Dtype::F32, &[], &input.to_le_bytes()),
    )
    .unwrap()
}

#[test]
fn native_publish_close_reopen_preserves_typed_roles_and_indexed_experts() -> Result<()> {
    let scratch = Scratch::new();
    let mut pile = Pile::open(&scratch.0)?;
    // A fresh throwaway signer: no key file, configured pile, or live service.
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(key.verifying_key()),
        AdmissionPolicy::direct(key.verifying_key()),
    );
    let collection: model_collection::ModelCollection =
        pile.collection("native-synthetic", policy)?;
    model_collection::require_model_writer(&mut pile, &key, collection)?;

    let a = native::store_nvfp4(&mut pile, "projection", nv(0x21, 0.125, 0.25))?;
    let a_id = a.root().unwrap();
    let b = native::store_nvfp4(&mut pile, "projection", nv(0x43, 0.5, 0.75))?;
    let b_id = b.root().unwrap();
    let c = native::store_fp8(&mut pile, "attention", fp(0x38, 2., 3.))?;
    let c_id = c.root().unwrap();
    let d = native::store_fp8(&mut pile, "attention", fp(0x40, 4., 5.))?;
    let d_id = d.root().unwrap();
    let first_expert = native::store_nvfp4(&mut pile, "down", nv(0x65, 6., 7.))?;
    let second_expert = native::store_nvfp4(&mut pile, "down", nv(0x87, 8., 9.))?;
    let experts = entity! { attrs::kind: native_pile::EXPERT, attrs::index: 0u64, attrs::member*: first_expert }
        + entity! { attrs::kind: native_pile::EXPERT, attrs::index: 1u64, attrs::member*: second_expert };
    // Raw bit preservation, not a conversion through f32 (NaNs and signed zero).
    let bf16 = [0x00, 0x80, 0x81, 0x7f];
    let f32 = [0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0xc0, 0x7f];
    let aux = native::store_bf16(&mut pile, "norm", [2], Bytes::from_source(bf16.to_vec()))?
        + native::store_f32(&mut pile, "A_log", [1, 2], Bytes::from_source(f32.to_vec()))?;
    let layer = entity! { attrs::kind: native_pile::LAYER, attrs::index: 3u64,
    attrs::member*: a + b + c + d + experts + aux };
    let layer_id = layer.root().unwrap();
    // Explicit random identity is just as readable as a derived identity.
    let root_id = genid();
    let mut root = entity! { &root_id @ attrs::member*: layer };
    root += entity! { &root_id @ triblespace::core::metadata::name: "unmodeled annotation" };
    assert!(
        model_collection::require_model_writer(
            &mut pile,
            &SigningKey::generate(&mut rand::rngs::OsRng),
            collection
        )
        .is_err()
    );
    model_collection::publish_model_fragment_into(&mut pile, &key, collection, root)?;
    pile.close()?;

    let mut reopened = Pile::new(triblespace::core::repo::pile::PileFile::open_read_only(
        &scratch.0,
    )?);
    reopened.refresh()?;
    let store = reopened.snapshot()?;
    let collection = model_collection::open_model_collection(&store, collection.handle())?;
    let snapshot = model_collection::snapshot_model_collection_for(&store, collection)?;
    assert_eq!(snapshot.support().len(), 1);
    assert_eq!(
        native_pile::layers(&snapshot, root_id.id, 3).collect::<Vec<_>>(),
        [layer_id]
    );
    assert_eq!(native_pile::layers(&snapshot, root_id.id, 4).count(), 0);
    let rows = native_pile::nvfp4(&snapshot, layer_id, "projection").collect::<Result<Vec<_>>>()?;
    assert_eq!(rows.len(), 2, "no cross-operation scale Cartesian product");
    for row in rows {
        let expected = if row.operation == a_id {
            nv(0x21, 0.125, 0.25)
        } else {
            assert_eq!(row.operation, b_id);
            nv(0x43, 0.5, 0.75)
        };
        assert_eq!(row.weight.dims(), [2, 16]);
        assert_eq!(&row.weight.payload()[..], expected.payload);
        assert!(row.input_scale.dims().is_empty());
        assert_eq!(&row.input_scale.payload()[..], expected.input_scale_le);
    }
    let rows = native_pile::fp8(&snapshot, layer_id, "attention").collect::<Result<Vec<_>>>()?;
    assert_eq!(rows.len(), 2, "all three leaves must share their operation");
    for row in rows {
        let expected = if row.operation == c_id {
            fp(0x38, 2., 3.)
        } else {
            assert_eq!(row.operation, d_id);
            fp(0x40, 4., 5.)
        };
        assert_eq!(row.weight.dims(), [2, 3]);
        assert_eq!(&row.weight.payload()[..], expected.e4m3fn);
        assert_eq!(&row.input_scale.payload()[..], expected.input_scale_le);
        assert_eq!(&row.weight_scale.payload()[..], expected.weight_scale_le);
    }
    for (index, expected) in [(0, nv(0x65, 6., 7.)), (1, nv(0x87, 8., 9.))] {
        let experts = native_pile::experts(&snapshot, layer_id, index).collect::<Vec<_>>();
        assert_eq!(experts.len(), 1);
        let rows = native_pile::nvfp4(&snapshot, experts[0], "down").collect::<Result<Vec<_>>>()?;
        assert_eq!(rows.len(), 1);
        assert_eq!(&rows[0].weight.payload()[..], expected.payload);
        assert_eq!(&rows[0].input_scale.payload()[..], expected.input_scale_le);
    }
    assert_eq!(native_pile::nvfp4(&snapshot, layer_id, "down").count(), 0);
    assert_eq!(native_pile::experts(&snapshot, layer_id, 2).count(), 0);
    let norms = native_pile::auxiliary::<BF16, 1, _>(&snapshot, layer_id, "norm")
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(norms.len(), 1);
    assert_eq!(&norms[0].payload()[..], bf16);
    let a_logs = native_pile::auxiliary::<F32, 2, _>(&snapshot, layer_id, "A_log")
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(a_logs.len(), 1);
    assert_eq!(a_logs[0].dims(), [1, 2]);
    assert_eq!(&a_logs[0].payload()[..], f32);
    assert_eq!(
        native_pile::auxiliary::<F32, 1, _>(&snapshot, layer_id, "norm").count(),
        0
    );
    assert_eq!(
        native_pile::auxiliary::<BF16, 2, _>(&snapshot, layer_id, "norm").count(),
        0
    );
    assert_eq!(
        native_pile::nvfp4(&snapshot, layer_id, "attention").count(),
        0
    );
    assert_eq!(
        native_pile::fp8(&snapshot, layer_id, "projection").count(),
        0
    );
    assert_ne!(leaf::leaf::<E4M3, 2>().id(), leaf::leaf::<BF16, 2>().id());
    assert_ne!(leaf::leaf::<NVFP4, 2>().id(), leaf::leaf::<E4M3, 2>().id());
    Ok(())
}

#[test]
fn missing_matched_blob_is_an_error_not_a_negative_match() -> Result<()> {
    let mut repo = MemoryRepo::default();
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(key.verifying_key()),
        AdmissionPolicy::direct(key.verifying_key()),
    );
    let collection: model_collection::ModelCollection =
        repo.collection("missing-synthetic", policy)?;
    let absent = leaf::leaf_blob::<BF16, 1>([1], Bytes::from_source(vec![0u8, 0]))?;
    let leaf = entity! { leaf::leaf::<BF16, 1>(): absent.get_handle() };
    let op = entity! { attrs::kind: "norm", attrs::weight*: leaf };
    let root = entity! { attrs::member*: op };
    let id = root.root().unwrap();
    repo.commit(collection, &key, root)?;
    let snapshot = model_collection::snapshot_model_collection_for(&repo.snapshot()?, collection)?;
    assert!(
        native_pile::auxiliary::<BF16, 1, _>(&snapshot, id, "norm")
            .next()
            .unwrap()
            .is_err()
    );
    assert_eq!(
        native_pile::auxiliary::<F32, 1, _>(&snapshot, id, "norm").count(),
        0
    );
    Ok(())
}

#[test]
fn native_ingress_rejects_bad_geometry_and_payload_lengths() -> Result<()> {
    let scratch = Scratch::new();
    let mut pile = Pile::open(&scratch.0)?;
    let mut packed = nv(0x21, 1., 1.);
    packed.logical_shape = [4, 8]; // total count fits, but a block straddles rows.
    assert!(native::store_nvfp4(&mut pile, "bad", packed).is_err());
    let mut packed = fp(0x38, 1., 1.);
    packed.e4m3fn.pop();
    assert!(native::store_fp8(&mut pile, "bad", packed).is_err());
    assert!(native::store_bf16(&mut pile, "bad", [2], Bytes::from_source(vec![0u8; 3])).is_err());
    assert!(native::store_f32(&mut pile, "bad", [2], Bytes::from_source(vec![0u8; 7])).is_err());
    assert!(
        native::store_bf16(
            &mut pile,
            "overflow",
            [u64::MAX, 2],
            Bytes::from_source(vec![] as Vec<u8>)
        )
        .is_err()
    );
    pile.close()?;
    Ok(())
}

#[test]
fn malformed_matched_tensor_header_is_an_error_not_a_panic() -> Result<()> {
    use triblespace::core::blob::encodings::tensor::{TENSOR_HEADER_LEN, Tensor};
    let mut repo = MemoryRepo::default();
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(key.verifying_key()),
        AdmissionPolicy::direct(key.verifying_key()),
    );
    let collection: model_collection::ModelCollection =
        repo.collection("malformed-synthetic", policy)?;
    let mut bytes = vec![0u8; TENSOR_HEADER_LEN];
    bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    bytes[8..16].copy_from_slice(&2u64.to_le_bytes());
    let malformed: Blob<Tensor<BF16, 2>> = Blob::new(Bytes::from_source(bytes));
    let handle = repo.put(malformed)?;
    let leaf = entity! { leaf::leaf::<BF16, 2>(): handle };
    let op = entity! { attrs::kind: "norm", attrs::weight*: leaf };
    let root = entity! { attrs::member*: op };
    let id = root.root().unwrap();
    repo.commit(collection, &key, root)?;
    let snapshot = model_collection::snapshot_model_collection_for(&repo.snapshot()?, collection)?;
    let error = native_pile::auxiliary::<BF16, 2, _>(&snapshot, id, "norm")
        .next()
        .unwrap()
        .err()
        .unwrap();
    assert!(error.to_string().contains("product overflow"));
    Ok(())
}
