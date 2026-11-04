//! Synthetic host blobs and reserved native fixtures; no live pile/checkpoint.
use super::*;
use super::super::tests::{config, limits, fixture, state_bytes, enqueue_rejected_upload};
use triblespace::core::{blob::{MemoryBlobStore, MemoryBlobStoreSnapshot,
    encodings::{simplearchive::SimpleArchive, tensor::TENSOR_HEADER_LEN}}, repo::SnapshotSource};

fn host_checkpoint() -> (PrefixCheckpoint, MemoryBlobStore) {
    let mut store = MemoryBlobStore::new();
    let c = config();
    let ids: Vec<_> = (0..3).map(|_| *genid()).collect();
    // Lookup coordinates only; the host fixture does not claim an admitted
    // model collection or ask this empty descriptor to authenticate anything.
    let collection = Blob::<SimpleArchive>::new(anybytes::Bytes::from_source(Vec::<u8>::new())).get_handle();
    let source = Source { collection, root: *genid(), config: *genid(),
        tokenizer_binding: *genid(), tokenizer_base: *genid(), pre_tokenizer: *genid() };
    let m = mamba_config(&c);
    let zeros = |dims: &[usize], width: usize| anybytes::Bytes::from_source(vec![0u8; dims.iter().product::<usize>() * width]);
    let conv_shape = [1, m.layout().unwrap().conv_channels, m.conv_kernel];
    let recurrent_shape = [1, m.heads, m.head_dim, m.state_dim];
    let conv = put::<BF16, 3, _>(&mut store, conv_shape, zeros(&conv_shape, 2)).unwrap();
    let recurrent_payload: Vec<u8> = (0..recurrent_shape.iter().product::<usize>())
        .flat_map(|_| 0x3f80_0001u32.to_le_bytes()).collect();
    let recurrent = put::<F32, 4, _>(&mut store, recurrent_shape, anybytes::Bytes::from_source(recurrent_payload)).unwrap();
    let mut chunks = Vec::new();
    for (start, rows) in [(0, 2), (2, 1)] {
        let shape = [rows, 2, 128];
        let payload = anybytes::Bytes::from_source((0..rows * 256)
            .flat_map(|i| ((i as u16) ^ 0x8000).to_le_bytes()).collect::<Vec<_>>());
        let keys = put::<BF16, 3, _>(&mut store, shape, payload.clone()).unwrap();
        let values = put::<BF16, 3, _>(&mut store, shape, payload).unwrap();
        chunks.push(KvChunk { start, rows, keys, values });
    }
    let last_logits = put::<F32, 1, _>(&mut store, [c.vocab_size], zeros(&[c.vocab_size], 4)).unwrap();
    (PrefixCheckpoint { provenance: PrefixProvenance { source, config: c, layers: ids,
        execution: PrefixExecution::NativeB1Bf16KvF32Mamba }, position: 3, tokens: Arc::from([1, 5, 7]),
        last_logits, layers: vec![PrefixLayer::Mamba { position: 3, conv, recurrent },
            PrefixLayer::Attention(chunks), PrefixLayer::Stateless] }, store)
}

fn valid(c: &PrefixCheckpoint, store: &MemoryBlobStoreSnapshot) -> Result<()> {
    c.validate(store, &config(), &c.provenance.layers, limits(), &[1, 5, 7])
}

#[test]
fn checkpoint_host_typed_chunks_preserve_bits_and_own_views() {
    let (checkpoint, mut store) = host_checkpoint();
    let snapshot = store.snapshot().unwrap();
    valid(&checkpoint, &snapshot).unwrap();
    let PrefixLayer::Attention(chunks) = &checkpoint.layers[1] else { panic!() };
    assert_eq!((chunks[0].start, chunks[0].rows, chunks[1].start, chunks[1].rows), (0, 2, 2, 1));
    assert_eq!(chunks[0].keys, chunks[0].values, "identical typed bytes share a handle, without model salt");
    let view = read::<BF16, 3, _>(&snapshot, chunks[0].keys, [2, 2, 128]).unwrap();
    let PrefixLayer::Mamba { recurrent, .. } = &checkpoint.layers[0] else { panic!() };
    let m = mamba_config(&config());
    let recurrence = read::<F32, 4, _>(&snapshot, *recurrent, [1, m.heads, m.head_dim, m.state_dim]).unwrap();
    let expected: Vec<u8> = (0..512).flat_map(|i| ((i as u16) ^ 0x8000).to_le_bytes()).collect();
    drop(snapshot);
    drop(store);
    assert_eq!(view.payload().as_ref(), expected.as_slice(), "owning view outlives the store and snapshot");
    // Includes signed zero and subnormals: no host f32 conversion/requantization.
    assert_eq!(&view.payload()[..4], &[0, 128, 1, 128]);
    assert!(recurrence.payload().chunks_exact(4).all(|v| v == 0x3f80_0001u32.to_le_bytes()),
        "F32 recurrent bits not representable as BF16 must survive unchanged");
}

#[test]
fn checkpoint_host_rejects_kv_gaps_overlaps_empty_and_overflow_ranges() {
    let (checkpoint, mut store) = host_checkpoint();
    let snapshot = store.snapshot().unwrap();
    for (start, rows) in [(1, 1), (3, 1), (2, 0), (usize::MAX, 2), (2, 2)] {
        let mut wrong = checkpoint.clone();
        let PrefixLayer::Attention(chunks) = &mut wrong.layers[1] else { panic!() };
        chunks[1].start = start;
        chunks[1].rows = rows;
        assert!(valid(&wrong, &snapshot).is_err());
    }
    let mut incomplete = checkpoint.clone();
    let PrefixLayer::Attention(chunks) = &mut incomplete.layers[1] else { panic!() };
    chunks.pop();
    assert!(valid(&incomplete, &snapshot).is_err());
    valid(&checkpoint, &snapshot).unwrap();
}

#[test]
fn checkpoint_host_requires_exact_tokens_config_layers_and_hybrid_boundary() {
    let (checkpoint, mut store) = host_checkpoint();
    let snapshot = store.snapshot().unwrap();
    assert!(checkpoint.validate(&snapshot, &config(), &checkpoint.provenance.layers, limits(), &[1, 5]).is_err());
    for change in 0..7 {
        let mut wrong = checkpoint.clone();
        match change {
            0 => wrong.position = 2,
            1 => wrong.tokens = Arc::from([1, 5, 32]),
            2 => wrong.provenance.config.layer_norm_epsilon *= 2.0,
            3 => wrong.provenance.layers.swap(0, 1),
            4 => wrong.layers.swap(0, 1),
            5 => { wrong.layers.pop(); }
            6 => if let PrefixLayer::Mamba { position, .. } = &mut wrong.layers[0] { *position = 2; },
            _ => unreachable!(),
        }
        assert!(wrong.validate(&snapshot, &config(), &checkpoint.provenance.layers, limits(), &[1, 5, 7]).is_err());
    }
    valid(&checkpoint, &snapshot).unwrap();
}

#[test]
fn checkpoint_host_rejects_missing_malformed_shape_dtype_and_logits() {
    let (checkpoint, mut store) = host_checkpoint();
    let empty = MemoryBlobStore::new().snapshot().unwrap();
    assert!(valid(&checkpoint, &empty).is_err());
    let mut wrong = checkpoint.clone();
    // Same byte width does not imply same dimensions; no flatten-and-relabel.
    let conv = put::<BF16, 3, _>(&mut store, [1, 1, mamba_config(&config()).extents(1, 1).unwrap().conv],
        anybytes::Bytes::from_source(vec![0u8; mamba_config(&config()).extents(1, 1).unwrap().conv * 2])).unwrap();
    if let PrefixLayer::Mamba { conv: target, .. } = &mut wrong.layers[0] { *target = conv; }
    assert!(valid(&wrong, &store.snapshot().unwrap()).is_err());
    // A deliberately mislabelled F32 payload cannot enter the BF16 shape.
    // Same-width F16/BF16 provenance is the typed query boundary's contract;
    // the raw bytes cannot prove their own interpretation.
    let fp32 = leaf::leaf_blob::<F32, 3>([2, 2, 128], anybytes::Bytes::from_source(vec![0u8; 2048])).unwrap();
    let bad = store.insert(fp32.transmute::<Tensor<BF16, 3>>());
    wrong = checkpoint.clone();
    if let PrefixLayer::Attention(chunks) = &mut wrong.layers[1] { chunks[0].keys = bad; }
    assert!(valid(&wrong, &store.snapshot().unwrap()).is_err());
    // Exact-length malicious header, rejected before decoder multiplication.
    let mut raw = vec![0; TENSOR_HEADER_LEN + 1024];
    for offset in [0, 8, 16] { raw[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes()); }
    let bad = store.insert(Blob::<Tensor<BF16, 3>>::new(anybytes::Bytes::from_source(raw)));
    if let PrefixLayer::Attention(chunks) = &mut wrong.layers[1] { chunks[0].keys = bad; }
    assert!(valid(&wrong, &store.snapshot().unwrap()).is_err());
    wrong = checkpoint.clone();
    let values: Vec<u8> = vec![f32::NAN; config().vocab_size].iter().flat_map(|v| v.to_le_bytes()).collect();
    wrong.last_logits = put::<F32, 1, _>(&mut store, [config().vocab_size], anybytes::Bytes::from_source(values)).unwrap();
    assert!(valid(&wrong, &store.snapshot().unwrap()).is_err());
    valid(&checkpoint, &store.snapshot().unwrap()).unwrap();
}

#[test]
fn checkpoint_host_new_tensor_preserves_previous_immutable_selection() {
    let (old, mut store) = host_checkpoint();
    let before = store.snapshot().unwrap();
    let mut newer = old.clone();
    newer.last_logits = put::<F32, 1, _>(&mut store, [config().vocab_size],
        anybytes::Bytes::from_source(vec![1f32; config().vocab_size].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())).unwrap();
    assert_ne!(newer.last_logits, old.last_logits);
    valid(&old, &before).unwrap();
    assert!(valid(&newer, &before).is_err());
    valid(&newer, &store.snapshot().unwrap()).unwrap();
    valid(&old, &store.snapshot().unwrap()).unwrap();
}

#[test]
#[ignore = "reserved CUDA: typed prefix export/restore across two local model owners and private continuation"]
fn checkpoint_gpu_restores_exact_state_and_independent_continuation() {
    let (snapshot, source, ids) = fixture();
    let l = limits();
    let o = resident_ownership(&config(), l).unwrap();
    assert!(2 * o.fixed_model_bytes + 5 * o.private_state_bytes
        + 4 * o.private_expert_cache_bytes + l.max_chunk_bytes < 1 << 30);
    let mut producer = Session::from_snapshot_seed(snapshot.clone(), source, &ids, CudaDevice { index: 0 }, l, &[1, 5, 7]).unwrap();
    assert!(producer.saved_prefix().is_none(), "partial seed has no exportable Prefix");
    producer.append_ids(&[1, 5, 7]).unwrap();
    let prefix = producer.saved_prefix().unwrap().clone();
    let expected = state_bytes(&producer);
    let expected_logits = producer.last_logits().unwrap().values.to_vec();
    let mut blobs = MemoryBlobStore::new();
    assert!(prefix.export(&mut blobs, 0).is_err());
    let checkpoint = prefix.export(&mut blobs, 2).unwrap();
    let blobs = blobs.snapshot().unwrap();
    let mut other = Session::from_snapshot(snapshot, source, &ids, CudaDevice { index: 0 }, l).unwrap();
    let destination = other.device_model().clone();
    assert!(!prefix.compatible_with(&destination));
    let rejected = other.restore_checkpoint(&checkpoint, &blobs, &[1, 5, 7], |_, _| Err("caller rejects arithmetic profile".into()));
    assert!(rejected.err().unwrap().contains("caller rejects"));
    assert_eq!(other.status(), Status::Ready);
    assert!(other.committed_ids().is_empty() && other.pending.is_empty());
    other.restore_checkpoint(&checkpoint, &blobs, &[1, 5, 7], |provenance, model| {
        // Concrete policy for this fixture: both loaders used the SAME held
        // immutable snapshot and selections, with the same native executable.
        assert!(std::ptr::eq(model.snapshot(), producer.device_model().snapshot()));
        assert_eq!(provenance.source, model.source());
        Ok(())
    }).unwrap();
    let restored = other.saved_prefix().unwrap().clone();
    assert!(restored.compatible_with(&destination));
    let mut a = other;
    assert_eq!(state_bytes(&a), expected);
    assert_eq!(a.last_logits().unwrap().values, expected_logits);
    assert_eq!(a.prefix_ownership().unwrap().shared_kv_bytes, 3 * 1024);
    assert_eq!(a.prefix_ownership().unwrap().private_kv_bytes, 5 * 1024);
    let mut b = Session::from_prefix(destination, restored, l).unwrap();
    producer.append_ids(&[9]).unwrap();
    a.append_ids(&[9]).unwrap();
    assert_eq!(state_bytes(&a), state_bytes(&producer));
    assert_eq!(a.last_logits().unwrap().values, producer.last_logits().unwrap().values);
    assert_eq!(state_bytes(&b), expected, "private continuation cannot overwrite shared checkpoint");
    b.append_ids(&[2]).unwrap();
    a.restore_prefix().unwrap();
    assert_eq!(state_bytes(&a), expected);
    drop(producer);
    drop(prefix);
    drop(checkpoint);
    drop(blobs);
    // Destination state owns its allocations independently of host blob views.
    a.append_ids(&[2]).unwrap();
    assert_eq!(state_bytes(&a), state_bytes(&b));
}

#[test]
#[ignore = "reserved CUDA: checked origin upload rejection cannot publish restored Prefix"]
fn checkpoint_gpu_failed_origin_upload_never_publishes_restore() {
    let (snapshot, source, ids) = fixture();
    let mut producer = Session::from_snapshot_seed(snapshot, source, &ids, CudaDevice { index: 0 }, limits(), &[1, 5, 7]).unwrap();
    producer.append_ids(&[1, 5, 7]).unwrap();
    let mut blobs = MemoryBlobStore::new();
    let checkpoint = producer.saved_prefix().unwrap().export(&mut blobs, 1).unwrap();
    let snapshot = blobs.snapshot().unwrap();
    let caller = StreamId::current();
    let origin = std::thread::spawn(StreamId::current).join().unwrap();
    let mut destination = origin.executes(|| Session::from_device_model(producer.device_model().clone(), limits())).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        destination.restore_checkpoint_inner(&checkpoint, &snapshot, &[1, 5, 7], |_, _| Ok(()), |_, _| {
            assert_eq!(StreamId::current(), origin);
            panic!("injected restore unwind after actual uploads");
        })
    }));
    assert!(result.is_err());
    assert_eq!(StreamId::current(), caller);
    assert_eq!(destination.status(), Status::Poisoned);
    assert!(destination.saved_prefix().is_none() && destination.pending.len() == 4);
    let retained = destination.pending.len();
    destination.fault = Some(Fault::Reset);
    assert!(destination.reset().is_err());
    assert_eq!(destination.pending.len(), retained, "failed reset cannot retire uncertain restore operands");
    destination.fault = None;
    destination.reset().unwrap();
    assert!(destination.pending.is_empty());
    let result = destination.restore_checkpoint_inner(&checkpoint, &snapshot,
        &[1, 5, 7], |_, _| Ok(()), |model, boundaries| {
            let Boundary::Mamba(state) = &boundaries[0] else { panic!() };
            enqueue_rejected_upload(&model.device, state.conv().handle.clone());
        });
    assert!(result.err().unwrap().contains("checkpoint CUDA completion"));
    assert_eq!(StreamId::current(), caller);
    assert_eq!(destination.status(), Status::Poisoned);
    assert!(destination.saved_prefix().is_none() && destination.pending.len() == 4);
    // No inference/recovery after the defined checked descriptor rejection.
    // The caller still owns the original immutable checkpoint and host blobs.
    assert_eq!(checkpoint.tokens.as_ref(), &[1, 5, 7]);
    // Ordinary owner Drop uses the established managed stream/drop-queue path;
    // it is not a successful recovery or a claimed drain of a failed context.
    drop(destination);
}
