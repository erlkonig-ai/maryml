//! Resolve one concrete native BF16 role from one caller-selected model root.
//!
//! No model/name catalogue is retained and no identifier is reconstructed from
//! a name. Extra facts and unsupported encodings coexist. Among shape-correct
//! native candidates the typed query's stable order chooses the first; this is
//! a selection policy, NOT evidence that other candidates have equal weights.

use crate::{format::attrs, leaf::leaf};
use triblespace::core::{
    blob::{Blob, encodings::tensor::{Tensor, elements::BF16}},
    inline::encodings::hash::Handle,
};
use triblespace::prelude::*;

pub type Slot<const R: usize> = Inline<Handle<Tensor<BF16, R>>>;

/// Reports the selected role immediately. The callback can hash/record bytes
/// for an audit; production need not create a second weight catalogue. The
/// payload is still the original zero-copy native leaf, never a host tensor.
pub fn resolve<const R: usize>(
    facts: &TribleSet,
    snapshot: &impl BlobStoreGet,
    root: Id,
    name: &str,
    shape: &[u64; R],
    selected: &mut impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
) -> Result<Slot<R>, String> {
    let mut wrong_shape = 0;
    for (path, handle) in find!(
        (path: Inline<Handle<blobencodings::UTF8String>>, handle: Slot<R>),
        pattern!(facts, [
            { root @ attrs::member: _?member },
            { _?member @ attrs::safetensor_path: ?path, attrs::weight: _?weight },
            { _?weight @ leaf::<BF16, R>(): ?handle }
        ])
    ) {
        // A missing/invalid name is not evidence of a different name. A read
        // fault stays a read fault, rather than becoming an unsupported role.
        let observed: anybytes::View<str> = snapshot.get(path)
            .map_err(|e| format!("{name}: read candidate path {path:?}: {e}"))?;
        if &*observed != name { continue; }
        let blob: Blob<Tensor<BF16, R>> = snapshot.get(handle)
            .map_err(|e| format!("{name}: read native BF16 leaf {handle:?}: {e}"))?;
        crate::nn::cuda_bf16_alias::checked_shape(&blob)
            .map_err(|e| format!("{name}: invalid native BF16 extent {handle:?}: {e}"))?;
        let view = crate::leaf::read_leaf(blob)
            .map_err(|e| format!("{name}: invalid native BF16 leaf {handle:?}: {e}"))?;
        if view.dims() != shape {
            wrong_shape += 1;
            continue;
        }
        selected(name, handle.raw, shape, view.payload().as_ref())?;
        return Ok(handle);
    }
    Err(format!(
        "{name}: no supported native BF16 rank-{R} role {shape:?} under root {root:?}; {wrong_shape} shape-incompatible candidates (no F16/CPU fallback)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::blob::MemoryBlobStore;

    fn named(facts: &mut TribleSet, blobs: &mut MemoryBlobStore, root: &ExclusiveId,
        payload: Slot<1>) {
        // Explicit random entity IDs prove the loader does not recompute any
        // model/member/leaf identity from the consuming role or payload.
        let member = fucid();
        let weight = fucid();
        let name = blobs.put::<blobencodings::UTF8String, _>("role.weight".to_string()).unwrap();
        *facts += entity! { root @ attrs::member: &member };
        *facts += entity! { &member @ attrs::safetensor_path: name, attrs::weight: &weight };
        *facts += entity! { &weight @ leaf::<BF16, 1>(): payload };
    }

    #[test]
    fn role_query_preserves_opaque_ids_and_ignores_shape_incompatible_candidates() {
        let mut blobs = MemoryBlobStore::new();
        let mut facts = TribleSet::new();
        let root = fucid();
        let good = crate::leaf::leaf_blob::<BF16, 1>([2],
            anybytes::Bytes::from_source(vec![0x80u8, 0x3f, 0, 0])).unwrap();
        let good = blobs.put(good).unwrap();
        let wrong = crate::leaf::leaf_blob::<BF16, 1>([1],
            anybytes::Bytes::from_source(vec![0u8, 0])).unwrap();
        let wrong = blobs.put(wrong).unwrap();
        named(&mut facts, &mut blobs, &root, wrong);
        named(&mut facts, &mut blobs, &root, good);
        let snapshot = blobs.snapshot().unwrap();
        let mut observations = 0;
        let picked = resolve(&facts, &snapshot, root.id, "role.weight", &[2], &mut |name, handle, shape, bytes| {
            observations += 1;
            assert_eq!(name, "role.weight");
            assert_eq!(handle, good.raw);
            assert_eq!(shape, [2]);
            assert_eq!(bytes, [0x80, 0x3f, 0, 0]);
            Ok(())
        }).unwrap();
        assert_eq!(picked, good);
        assert_eq!(observations, 1);
    }

    #[test]
    fn absent_payload_is_a_read_error_not_an_unsupported_role() {
        let mut blobs = MemoryBlobStore::new();
        let mut facts = TribleSet::new();
        let root = fucid();
        let absent = crate::leaf::leaf_blob::<BF16, 1>([1],
            anybytes::Bytes::from_source(vec![0u8, 0])).unwrap().get_handle();
        named(&mut facts, &mut blobs, &root, absent);
        let snapshot = blobs.snapshot().unwrap();
        let error = resolve(&facts, &snapshot, root.id, "role.weight", &[1], &mut |_, _, _, _| Ok(())).unwrap_err();
        assert!(error.contains("read native BF16 leaf"), "{error}");
        assert!(!error.contains("no supported"), "{error}");
    }

    #[test]
    fn malformed_dimensions_fail_before_unchecked_tensor_product() {
        let mut blobs = MemoryBlobStore::new();
        let mut facts = TribleSet::new();
        let root = fucid();
        let mut bytes = vec![0u8; 256];
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        let bad = blobs.put(Blob::<Tensor<BF16, 1>>::new(anybytes::Bytes::from_source(bytes))).unwrap();
        named(&mut facts, &mut blobs, &root, bad);
        let snapshot = blobs.snapshot().unwrap();
        let error = resolve(&facts, &snapshot, root.id, "role.weight", &[1], &mut |_, _, _, _| Ok(())).unwrap_err();
        assert!(error.contains("invalid native BF16 extent"), "{error}");
    }
}
