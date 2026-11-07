//! Host-only temporary stores. No tokenizer, CUDA client or live collection.
use super::*;
use super::super::{checkpoint::tests::{host_checkpoint, nvfp4_kv_payload}, tests::{configuration, limits}};
use triblespace::core::{blob::MemoryBlobStoreSnapshot, trible::Trible};
use triblespace::core::{collection::{AdmissionPolicy, CollectionPolicy,
    CollectionSnapshotExt, CollectionStoreExt}, repo::pile::Pile};

// Only these tests own this create-new temporary file. No caller pile or key.
struct ManifestTestPile(std::path::PathBuf);
impl ManifestTestPile {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("mary-manifest-{}.pile", *genid()));
        std::fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
        Self(path)
    }
}
impl Drop for ManifestTestPile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

fn fixture() -> (PrefixCheckpoint, MemoryBlobStore, Inline<Handle<SimpleArchive>>) {
    let (mut checkpoint, mut store) = host_checkpoint();
    checkpoint.tokens = Arc::from([1, 5, 1]); // ordered duplicates must survive
    let PrefixLayer::Attention(chunks) = &mut checkpoint.layers[1] else { panic!() };
    for chunk in chunks {
        // Unlike the shared host fixture, V must differ from K: swapping their
        // roles must fail the manifest round trip, not pass through equal handles.
        let values = nvfp4_kv_payload(&(0..chunk.rows).collect::<Vec<_>>(), 0x11);
        chunk.values = store.put(leaf::leaf_blob::<NVFP4, 3>([chunk.rows as u64, 2, 128],
            anybytes::Bytes::from_source(values)).unwrap()).unwrap();
        assert_ne!(chunk.keys, chunk.values);
    }
    let mut facts = TribleSet::new();
    let config = jsonfacts::save_json(&serde_json::json!({"text": configuration()}),
        &mut store, &mut facts).unwrap();
    checkpoint.provenance.source.config = config;
    let image = store.put::<SimpleArchive, _>(facts).unwrap();
    (checkpoint, store, image)
}

fn encode_fixture(checkpoint: &PrefixCheckpoint, store: &mut MemoryBlobStore,
    image: Inline<Handle<SimpleArchive>>) -> Fragment
{
    let reader = store.snapshot().unwrap();
    encode(&reader, store, checkpoint, image, limits()).unwrap()
}

fn decode(reader: &MemoryBlobStoreSnapshot, facts: &TribleSet, root: Id,
    image: Inline<Handle<SimpleArchive>>) -> Vec<Result<PrefixCheckpoint>>
{
    let mut results = Vec::new();
    let flow = visit_selected(reader, facts, root, limits(), |actual_image, result| {
        assert_eq!(actual_image, image);
        results.push(result);
        ControlFlow::Continue(())
    });
    assert!(flow.is_continue());
    results
}

fn assert_checkpoint(actual: &PrefixCheckpoint, expected: &PrefixCheckpoint) {
    assert_eq!(actual.provenance.source, expected.provenance.source);
    assert_eq!(actual.provenance.config, expected.provenance.config);
    assert_eq!(actual.provenance.layers, expected.provenance.layers);
    assert_eq!(actual.provenance.execution, expected.provenance.execution);
    assert_eq!(actual.position, expected.position);
    assert_eq!(actual.tokens, expected.tokens);
    assert_eq!(actual.last_logits, expected.last_logits);
    assert_eq!(actual.layers.len(), expected.layers.len());
    for (a, b) in actual.layers.iter().zip(&expected.layers) {
        match (a, b) {
            (PrefixLayer::Attention(a), PrefixLayer::Attention(b)) => {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert_eq!((a.start, a.rows, a.keys, a.values), (b.start, b.rows, b.keys, b.values));
                }
            }
            (PrefixLayer::Mamba { position: a, conv: ac, recurrent: ar },
             PrefixLayer::Mamba { position: b, conv: bc, recurrent: br }) => {
                assert_eq!((a, ac, ar), (b, bc, br));
            }
            (PrefixLayer::Stateless, PrefixLayer::Stateless) => {},
            _ => panic!("layer kind changed"),
        }
    }
}

// Deliberately change one selected test value, not a production uniqueness rule.
fn replace<S: InlineEncoding>(facts: &TribleSet, entity: Id, attribute: &Attribute<S>,
    value: Inline<S>) -> TribleSet
{
    let mut changed = TribleSet::new();
    for row in facts.iter() {
        if row.e() != &entity || row.a() != &attribute.id() { changed.insert(row); }
    }
    changed.insert(&Trible::force(&entity, &attribute.id(), &value));
    changed
}

#[test]
fn manifest_host_le_tokens_are_exact_ordered_bounded_and_framed() {
    let tokens = [0x0102_0304, 7, 7, u32::MAX];
    let blob = TokenIdsLe::encode(&tokens).unwrap();
    assert_eq!(&blob.bytes[..8], &[4, 3, 2, 1, 7, 0, 0, 0]);
    assert_eq!(TokenIdsLe::decode(blob.clone(), 4).unwrap(), tokens);
    assert!(TokenIdsLe::decode(blob, 3).unwrap_err().contains("capacity"));
    for length in [1, 2, 3, 5, 6, 7] {
        let malformed = Blob::new(anybytes::Bytes::from_source(vec![0u8; length]));
        assert_eq!(TokenIdsLe::decode(malformed, 8).unwrap_err(), "partial LE-u32 token");
    }
    assert!(TokenIdsLe::decode(TokenIdsLe::encode(&[]).unwrap(), 0).unwrap().is_empty());
}

#[test]
fn manifest_host_roundtrip_execution_tokens_ranges_and_typed_handles() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let root = fragment.root().unwrap();
    let archive = store.put::<SimpleArchive, _>(fragment.facts()).unwrap();
    let reader = store.snapshot().unwrap();
    let facts: TribleSet = reader.get(archive).unwrap();
    let profiles: Vec<_> = find!(profile: Inline<Handle<UTF8String>>,
        pattern!(&facts, [{ root @ schema::execution_profile: ?profile }])).collect();
    assert_eq!(profiles.len(), 1);
    let text: anybytes::View<str> = reader.get(profiles[0]).unwrap();
    assert_eq!(&*text, "native-b1-nvfp4-kv-f32-mamba/mma-b2-nvfp4-kv-fp4-qk-hilo-p-v1");
    let results = decode(&reader, &facts, root, image);
    assert_eq!(results.len(), 1);
    assert_checkpoint(results[0].as_ref().unwrap(), &checkpoint);
    let PrefixLayer::Mamba { recurrent, .. } = &checkpoint.layers[0] else { panic!() };
    let blob: Blob<Tensor<F32, 4>> = reader.get(*recurrent).unwrap();
    let view = leaf::read_leaf(blob).unwrap();
    assert!(view.payload().chunks_exact(4).all(|b| b == 0x3f80_0001u32.to_le_bytes()));
    let PrefixLayer::Attention(chunks) = &checkpoint.layers[1] else { panic!() };
    let blob: Blob<Tensor<NVFP4, 3>> = reader.get(chunks[0].keys).unwrap();
    let view = leaf::read_leaf(blob).unwrap();
    assert_eq!(view.payload().as_ref(), nvfp4_kv_payload(&[0, 1], 0).as_slice());
    let blob: Blob<Tensor<NVFP4, 3>> = reader.get(chunks[0].values).unwrap();
    let view = leaf::read_leaf(blob).unwrap();
    assert_eq!(view.payload().as_ref(), nvfp4_kv_payload(&[0, 1], 0x11).as_slice());
}

#[test]
fn manifest_host_encode_refuses_config_image_mismatch() {
    let (mut checkpoint, mut store, image) = fixture();
    checkpoint.provenance.config.vocab_size += 8;
    let reader = store.snapshot().unwrap();
    let error = encode(&reader, &mut store, &checkpoint, image, limits()).err().unwrap();
    assert_eq!(error, "checkpoint config differs from supplied model image");
}

#[test]
fn manifest_host_selected_kv_gap_overlap_empty_and_outside_fail() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let facts = fragment.facts();
    let reader = store.snapshot().unwrap();
    let chunks: Vec<_> = find!((chunk: Id, range: Range<u128>),
        pattern!(facts, [{ ?chunk @ schema::token_range: ?range }])).collect();
    assert_eq!(chunks.len(), 2);
    let last = chunks.iter().find(|(_, r)| r.start == 2).unwrap().0;
    for range in [1..3, 3..4, 2..2, 2..4, 2..1] {
        let changed = replace(facts, last, &schema::token_range, (range.start, range.end).to_inline());
        let results = decode(&reader, &changed, fragment.root().unwrap(), image);
        assert!(!results.is_empty());
        assert!(results.iter().all(Result::is_err));
    }
}

#[test]
fn manifest_host_selected_root_and_mamba_positions_fail() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let root = fragment.root().unwrap();
    let facts = fragment.facts();
    let reader = store.snapshot().unwrap();
    let mamba: Vec<_> = find!(layer: Id,
        pattern!(facts, [{ ?layer @ metadata::tag: schema::MAMBA }])).collect();
    assert_eq!(mamba.len(), 1);
    for (subject, position) in [(root, 4u64), (mamba[0], 2)] {
        let changed = replace(facts, subject, &schema::position, position.to_inline());
        let results = decode(&reader, &changed, root, image);
        assert!(!results.is_empty());
        assert!(results.iter().all(Result::is_err));
    }
}

#[test]
fn manifest_host_missing_or_wrong_shape_payload_is_selected_failure() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let facts = fragment.facts();
    let logits: Vec<_> = find!(entity: Id,
        pattern!(facts, [{ ?entity @ metadata::tag: schema::LOGITS }])).collect();
    assert_eq!(logits.len(), 1);
    let absent = leaf::leaf_blob::<F32, 1>([16], anybytes::Bytes::from_source(vec![0u8; 64])).unwrap();
    let missing = absent.get_handle();
    let wrong_shape = store.put(absent.clone()).unwrap();
    assert_eq!(missing, wrong_shape);
    let old = MemoryBlobStore::new().snapshot().unwrap();
    // No image in this reader: selected external dependency failure is explicit.
    assert!(decode(&old, facts, fragment.root().unwrap(), image).iter().all(Result::is_err));
    let changed = replace(facts, logits[0], &leaf::leaf::<F32, 1>(), wrong_shape);
    let reader = store.snapshot().unwrap();
    let results = decode(&reader, &changed, fragment.root().unwrap(), image);
    assert_eq!(results.len(), 1);
    assert!(results[0].as_ref().err().unwrap().contains("shape"));
    let absent = leaf::leaf_blob::<F32, 1>([32], anybytes::Bytes::from_source(vec![1u8; 128])).unwrap();
    let changed = replace(facts, logits[0], &leaf::leaf::<F32, 1>(), absent.get_handle());
    let results = decode(&reader, &changed, fragment.root().unwrap(), image);
    assert_eq!(results.len(), 1);
    assert!(results[0].as_ref().err().unwrap().contains("get"));
}

#[test]
fn manifest_host_opaque_ids_unrelated_facts_and_duplicates_remain_readable() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    // Ephemeral test substitution table, not retained cache state.
    let mut ids = std::collections::HashMap::new();
    for row in fragment.facts().iter() { ids.entry(*row.e()).or_insert_with(|| *genid()); }
    let root = ids[&fragment.root().unwrap()];
    let mut changed = TribleSet::new();
    for row in fragment.facts().iter() {
        let e = ids[row.e()];
        let value = row.v::<GenId>();
        if let Ok(id) = Id::try_from_inline(value) {
            if let Some(replacement) = ids.get(&id) {
                let value: Inline<GenId> = replacement.to_inline();
                changed.insert(&Trible::force(&e, row.a(), &value));
                continue;
            }
        }
        changed.insert(&Trible::force(&e, row.a(), row.v::<triblespace::core::inline::encodings::UnknownInline>()));
    }
    let note = entity! { ExclusiveId::force_ref(&root) @ metadata::tag: *genid() };
    changed.union(note.facts().clone());
    changed.union(changed.clone());
    let reader = store.snapshot().unwrap();
    let results = decode(&reader, &changed, root, image);
    assert_eq!(results.len(), 1);
    assert_checkpoint(results[0].as_ref().unwrap(), &checkpoint);
}

#[test]
fn manifest_host_profile_alternatives_are_visited_and_can_stop() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let root = fragment.root().unwrap();
    let second = store.put::<UTF8String, _>("native-b1-nvfp4-kv-f32-mamba/mma-hilo-p-t1-v1").unwrap();
    let unknown = store.put::<UTF8String, _>("future-unsupported-profile").unwrap();
    // The BF16-KV family before this one is another type: skipped, not refused.
    let bf16_kv = store.put::<UTF8String, _>("native-b1-bf16-kv-f32-mamba/mma-hilo-p-t1-v1").unwrap();
    let extra = entity! { ExclusiveId::force_ref(&root) @ schema::execution_profile*: [second, unknown, bf16_kv] };
    let mut facts = fragment.facts().clone();
    facts.union(extra.facts().clone());
    let reader = store.snapshot().unwrap();
    let results = decode(&reader, &facts, root, image);
    assert_eq!(results.len(), 2);
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(results.iter().filter(|r| r.as_ref().err().is_some_and(|e|
        e == "checkpoint execution profile differs")).count(), 1);
    let mut calls = 0;
    let flow = visit_selected(&reader, &facts, root, limits(), |_, _| {
        calls += 1;
        ControlFlow::Break(())
    });
    assert!(flow.is_break());
    assert_eq!(calls, 1);
}

#[test]
fn manifest_host_retired_profiles_are_refused_before_other_payload_reads() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let root = fragment.root().unwrap();
    // Every attention algorithm this execution replaced, under this family's
    // prefix. The store holds ONLY the profile text: any further read would
    // fail first.
    for retired in ["scalar-f32-v1", "mma-bf16-p-v1", "mma-hilo-p-v1", "mma-hilo-p-t1-v1"] {
        let mut profile_only = MemoryBlobStore::new();
        let text = profile_only.put::<UTF8String, _>(
            format!("native-b1-nvfp4-kv-f32-mamba/{retired}")).unwrap();
        let facts = replace(fragment.facts(), root, &schema::execution_profile, text);
        let reader = profile_only.snapshot().unwrap();
        let results = decode(&reader, &facts, root, image);
        assert_eq!(results.len(), 1, "{retired}");
        assert_eq!(results[0].as_ref().err().unwrap(), "checkpoint execution profile differs");
    }
}

#[test]
fn manifest_host_missing_typed_boundary_is_not_a_partial_checkpoint() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let mut facts = TribleSet::new();
    for row in fragment.facts().iter() {
        if row.a() != &leaf::leaf::<F32, 4>().id() { facts.insert(row); }
    }
    let reader = store.snapshot().unwrap();
    let results = decode(&reader, &facts, fragment.root().unwrap(), image);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].as_ref().err().unwrap(), "missing typed Mamba boundary");
}

#[test]
fn manifest_host_tensor_alternatives_do_not_hide_supported_values() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let facts = fragment.facts();
    let subjects: Vec<_> = find!(entity: Id,
        pattern!(facts, [{ ?entity @ metadata::tag: schema::RECURRENT }])).collect();
    assert_eq!(subjects.len(), 1);
    let dims = [1, 64, 64, 4];
    let alternate = store.put(leaf::leaf_blob::<F32, 4>(dims,
        anybytes::Bytes::from_source(vec![0u8; 64 * 64 * 4 * 4])).unwrap()).unwrap();
    let bad = leaf::leaf_blob::<F32, 4>(dims,
        anybytes::Bytes::from_source(vec![1u8; 64 * 64 * 4 * 4])).unwrap().get_handle();
    let recurrent_leaf = leaf::leaf::<F32, 4>();
    let extra = entity! { ExclusiveId::force_ref(&subjects[0]) @
        recurrent_leaf*: [alternate, bad] };
    let mut changed = facts.clone();
    changed.union(extra.facts().clone());
    let reader = store.snapshot().unwrap();
    let results = decode(&reader, &changed, fragment.root().unwrap(), image);
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 2);
    assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
}

#[test]
fn manifest_host_branching_kv_paths_are_complete_and_stoppable() {
    let (checkpoint, mut store, image) = fixture();
    let fragment = encode_fixture(&checkpoint, &mut store, image);
    let facts = fragment.facts();
    let layers: Vec<_> = find!(layer: Id,
        pattern!(facts, [{ ?layer @ metadata::tag: schema::ATTENTION }])).collect();
    assert_eq!(layers.len(), 1);
    // Original path: 0..2, 2..3. Add 0..1, 1..3 using byte-identical absolute
    // token rows, but a different chunk partition and distinct K/V payloads.
    let mut children = Fragment::empty();
    for (start, end) in [(0usize, 1usize), (1, 3)] {
        let rows = end - start;
        // Existing fixture restarts its pattern at the last 2..3 chunk.
        let source: Vec<_> = (start..end).map(|row| if row == 2 { 0 } else { row }).collect();
        let payload = |xor: u8| anybytes::Bytes::from_source(nvfp4_kv_payload(&source, xor));
        let keys = store.put(leaf::leaf_blob::<NVFP4, 3>([rows as u64, 2, 128],
            payload(0)).unwrap()).unwrap();
        let values = store.put(leaf::leaf_blob::<NVFP4, 3>([rows as u64, 2, 128],
            payload(0x11)).unwrap()).unwrap();
        assert_ne!(keys, values);
        let keys = entity! { metadata::tag: schema::KEYS, leaf::leaf::<NVFP4, 3>(): keys };
        let values = entity! { metadata::tag: schema::VALUES, leaf::leaf::<NVFP4, 3>(): values };
        children += entity! { schema::token_range: (start as u128, end as u128),
            model::member*: keys + values };
    }
    let extra = entity! { ExclusiveId::force_ref(&layers[0]) @ model::member*: children };
    let mut changed = facts.clone();
    changed.union(extra.facts().clone());
    let reader = store.snapshot().unwrap();
    let results = decode(&reader, &changed, fragment.root().unwrap(), image);
    assert_eq!(results.len(), 2);
    let mut paths = Vec::new();
    for result in &results {
        let selected = result.as_ref().unwrap();
        assert_eq!(selected.provenance.layers, checkpoint.provenance.layers);
        let PrefixLayer::Attention(chunks) = &selected.layers[1] else { panic!() };
        paths.push(chunks.iter().map(|c| (c.start, c.start + c.rows)).collect::<Vec<_>>());
        // Per tensor, the rows' code planes and scale planes joined in order.
        let (c, s) = (attention::CODE_ROW_BYTES, attention::SCALE_ROW_BYTES);
        let mut planes = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
        for chunk in chunks {
            for (tensor, handle) in [chunk.keys, chunk.values].into_iter().enumerate() {
                let blob: Blob<Tensor<NVFP4, 3>> = reader.get(handle).unwrap();
                let payload = leaf::read_leaf(blob).unwrap().payload().to_vec();
                planes[tensor][0].extend_from_slice(&payload[..chunk.rows * c]);
                planes[tensor][1].extend_from_slice(&payload[chunk.rows * c..chunk.rows * (c + s)]);
            }
        }
        for (tensor, xor) in [0u8, 0x11].into_iter().enumerate() {
            let whole = nvfp4_kv_payload(&[0, 1, 0], xor);
            assert_eq!(planes[tensor][0], &whole[..3 * c]);
            assert_eq!(planes[tensor][1], &whole[3 * c..3 * (c + s)]);
        }
    }
    paths.sort(); // operation-local test comparison, not a reader selection policy
    assert_eq!(paths, vec![vec![(0, 1), (1, 3)], vec![(0, 2), (2, 3)]]);
    let mut calls = 0;
    let flow = visit_selected(&reader, &changed, fragment.root().unwrap(), limits(), |_, result| {
            assert!(result.is_ok());
            calls += 1;
            ControlFlow::Break(())
        });
    assert!(flow.is_break());
    assert_eq!(calls, 1);
}

#[test]
fn manifest_host_old_snapshot_survives_new_manifest() {
    let (mut checkpoint, mut store, image) = fixture();
    let first = encode_fixture(&checkpoint, &mut store, image);
    let first_handle = store.put::<SimpleArchive, _>(first.facts()).unwrap();
    let old = store.snapshot().unwrap();
    let first_base = checkpoint.provenance.source.tokenizer_base;
    checkpoint.provenance.source.tokenizer_base = *genid();
    let second = encode_fixture(&checkpoint, &mut store, image);
    let second_handle = store.put::<SimpleArchive, _>(second.facts()).unwrap();
    assert_ne!(first_handle, second_handle);
    let unavailable: std::result::Result<TribleSet, _> = old.get(second_handle);
    assert!(unavailable.is_err());
    let facts: TribleSet = old.get(first_handle).unwrap();
    let results = decode(&old, &facts, first.root().unwrap(), image);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].as_ref().unwrap().provenance.source.tokenizer_base, first_base);
}

#[test]
fn manifest_host_pile_reopen_preserves_execution_payloads_and_opaque_alternatives() {
    let path = ManifestTestPile::new();
    let mut pile = Pile::open(&path.0).unwrap();
    let signer = ed25519_dalek::SigningKey::from_bytes(&[0x51; 32]);
    let policy = CollectionPolicy::new(AdmissionPolicy::direct(signer.verifying_key()),
        AdmissionPolicy::direct(signer.verifying_key()));
    let collection = pile.collection("manifest CPU reopen fixture", policy).unwrap();
    let (checkpoint, mut memory, image) = fixture();
    // Synthetic host fixture only: append dependencies individually, as
    // export does. Do not stage the tensor payloads in a publication Fragment.
    for (handle, blob) in memory.snapshot().unwrap().iter() {
        assert_eq!(pile.put(blob).unwrap(), handle);
    }
    let reader = pile.snapshot().unwrap();
    let fragment = encode(&reader, &mut pile, &checkpoint, image, limits()).unwrap();
    let derived_root = fragment.root().unwrap();
    let opaque_root = *genid();
    let (_, original, metafacts, blobs) = fragment.into_parts();
    let mut changed = TribleSet::new();
    for row in original.iter() {
        let subject = if row.e() == &derived_root { opaque_root } else { *row.e() };
        changed.insert(&Trible::force(&subject, row.a(),
            row.v::<triblespace::core::inline::encodings::UnknownInline>()));
    }
    let other_profile = pile.put::<UTF8String, _>(
        "native-b1-nvfp4-kv-f32-mamba/scalar-f32-v1").unwrap();
    let future_profile = pile.put::<UTF8String, _>("unknown-future-profile").unwrap();
    let extra = entity! { ExclusiveId::force_ref(&opaque_root) @
        metadata::tag: *genid(),
        schema::execution_profile*: [other_profile, future_profile]
    };
    changed.union(extra.facts().clone());
    changed.union(changed.clone());
    let expected_facts = changed.clone();
    pile.commit(collection, &signer, Fragment::from_parts(changed, metafacts, blobs)).unwrap();
    pile.flush().unwrap();
    drop(reader);
    drop(memory);
    pile.close().unwrap();

    let mut reopened = Pile::open(&path.0).unwrap();
    let reader = reopened.snapshot().unwrap();
    let observed = reader.collection(collection).unwrap();
    let facts = observed.view::<TribleSet>().unwrap();
    assert_eq!(facts, expected_facts);
    let mut success = 0;
    let mut refused = 0;
    let flow = visit_selected(&reader, &facts, opaque_root, limits(), |actual_image, result| {
            assert_eq!(actual_image, image);
            match result {
                Ok(actual) => { assert_checkpoint(&actual, &checkpoint); success += 1; }
                Err(error) => {
                    assert_eq!(error, "checkpoint execution profile differs");
                    refused += 1;
                }
            }
            ControlFlow::Continue(())
        });
    assert!(flow.is_continue());
    assert_eq!((success, refused), (1, 1)); // unsupported profile is skipped
    let PrefixLayer::Attention(chunks) = &checkpoint.layers[1] else { panic!() };
    for chunk in chunks {
        let keys: Blob<Tensor<NVFP4, 3>> = reader.get(chunk.keys).unwrap();
        let values: Blob<Tensor<NVFP4, 3>> = reader.get(chunk.values).unwrap();
        assert_ne!(keys.bytes, values.bytes);
        assert_eq!(&leaf::read_leaf(keys).unwrap().payload()[..4], &[0x80, 0x81, 0x82, 0x83]);
        assert_eq!(&leaf::read_leaf(values).unwrap().payload()[..4], &[0x91, 0x90, 0x93, 0x92]);
    }
    drop(observed);
    drop(reader);
    reopened.close().unwrap();
}

#[test]
fn manifest_host_pile_precommit_failure_preserves_completed_manifest() {
    let path = ManifestTestPile::new();
    let mut pile = Pile::open(&path.0).unwrap();
    let signer = ed25519_dalek::SigningKey::from_bytes(&[0x52; 32]);
    let policy = CollectionPolicy::new(AdmissionPolicy::direct(signer.verifying_key()),
        AdmissionPolicy::direct(signer.verifying_key()));
    let collection = pile.collection("manifest CPU failure fixture", policy).unwrap();
    let (checkpoint, mut memory, image) = fixture();
    for (handle, blob) in memory.snapshot().unwrap().iter() {
        assert_eq!(pile.put(blob).unwrap(), handle);
    }
    let reader = pile.snapshot().unwrap();
    let completed = encode(&reader, &mut pile, &checkpoint, image, limits()).unwrap();
    let root = completed.root().unwrap();
    let expected_facts = completed.facts().clone();
    pile.commit(collection, &signer, completed).unwrap();
    pile.flush().unwrap();
    drop(reader);
    let mut newer = checkpoint.clone();
    newer.tokens = Arc::from([1, 5, 2]);
    let mut abandoned_root = None;
    let failure: Result<()> = (|| {
        let payload: Vec<u8> = (0..newer.provenance.config.vocab_size)
            .flat_map(|_| 1f32.to_le_bytes()).collect();
        newer.last_logits = pile.put(leaf::leaf_blob::<F32, 1>(
            [newer.provenance.config.vocab_size as u64],
            anybytes::Bytes::from_source(payload)).unwrap()).unwrap();
        let reader = pile.snapshot().unwrap();
        let pending = encode(&reader, &mut pile, &newer, image, limits())?;
        abandoned_root = pending.root();
        // Fault boundary: child/token/profile appends have happened, but the
        // completed-manifest COMMIT has not. No production fault hook needed.
        Err("injected before manifest COMMIT".into())
    })();
    assert_eq!(failure.unwrap_err(), "injected before manifest COMMIT");
    assert_ne!(abandoned_root, Some(root));
    assert_ne!(newer.last_logits, checkpoint.last_logits);
    // Deliberately persist even the abandoned payload: unreferenced bytes do
    // not create a completed candidate or remove the earlier publication.
    pile.flush().unwrap();
    drop(memory);
    pile.close().unwrap();
    let mut reopened = Pile::open(&path.0).unwrap();
    let reader = reopened.snapshot().unwrap();
    let observed = reader.collection(collection).unwrap();
    let facts = observed.view::<TribleSet>().unwrap();
    assert_eq!(facts, expected_facts);
    let _: Blob<Tensor<F32, 1>> = reader.get(newer.last_logits).unwrap();
    let mut successes = 0;
    let flow = visit_selected(&reader, &facts, root, limits(),
        |actual_image, result| {
            assert_eq!(actual_image, image);
            assert_checkpoint(&result.unwrap(), &checkpoint);
            successes += 1;
            ControlFlow::Continue(())
        });
    assert!(flow.is_continue());
    assert_eq!(successes, 1);
    let flow = visit_selected(&reader, &facts, abandoned_root.unwrap(), limits(), |_, _| panic!("uncommitted manifest was exposed"));
    assert!(flow.is_continue());
    drop(observed);
    drop(reader);
    reopened.close().unwrap();
}
