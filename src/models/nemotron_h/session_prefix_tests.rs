//! Tiny synthetic hybrid only. These tests do not open a checkpoint or certify
//! whole-model numerics, simultaneous kernels, throughput or physical admission.
use super::*;

#[test]
#[ignore = "reserved CUDA: synthetic prefix fork/restore/segmented parity, no checkpoint"]
fn session_gpu_prefix_forks_restore_and_keep_independent_suffixes() {
    let (snapshot, source, layers) = fixture();
    let l = limits();
    let o = resident_ownership(&config(), l).unwrap();
    // Conservative logical live payload bound: two fixed uploads, six private
    // quotas/states, one extra saved state, and one active chunk. Runtime/JIT
    // pool reservation and host source bytes are outside this assertion.
    assert!(2 * o.fixed_model_bytes + 7 * o.private_state_bytes
        + 6 * o.private_expert_cache_bytes + l.max_chunk_bytes < 1 << 30);
    let mut parent = Session::from_snapshot_seed(snapshot.clone(), source, &layers,
        CudaDevice { index: 0 }, l, &[1, 5, 7]).unwrap();
    assert_eq!(parent.prefix_ownership().unwrap().private_kv_bytes, 3 * 1024);
    assert!(parent.saved_prefix().is_none());
    assert!(parent.append_ids(&[1, 5, 6]).is_err());
    assert!(parent.committed_ids().is_empty());
    assert_eq!(parent.prefill_ids(&[1, 5, 7, 3], &mut || false).unwrap(), PrefillOutcome::Completed);
    let model = parent.device_model().clone();
    let saved = parent.saved_prefix().unwrap().clone();
    assert_eq!(saved.token_ids(), [1, 5, 7]);
    assert_eq!(parent.prefix_ownership().unwrap().shared_kv_bytes, 3 * 1024);
    assert_eq!(parent.prefix_ownership().unwrap().private_kv_bytes, 5 * 1024);
    let mut a = Session::from_prefix(model.clone(), saved.clone(), l).unwrap();
    let mut b = Session::from_prefix(model.clone(), saved.clone(), l).unwrap();
    let boundary = state_bytes(&a);
    let boundary_logits = a.last_logits().unwrap().values.to_vec();
    assert_eq!(a.prefix_ownership().unwrap().private_mamba_bytes, 0);
    a.append_ids(&[4, 6]).unwrap();
    let a_state = state_bytes(&a);
    let a_logits = a.last_logits().unwrap().values.to_vec();
    b.append_ids(&[9]).unwrap();
    assert_eq!(state_bytes(&a), a_state, "another fork cannot overwrite private state");
    assert_eq!(a.last_logits().unwrap().values, a_logits);
    assert!(a.prefix_ownership().unwrap().private_mamba_bytes > 0);
    let b_state = state_bytes(&b);
    let b_logits = b.last_logits().unwrap().values.to_vec();
    a.restore_prefix().unwrap();
    assert_eq!(state_bytes(&a), boundary);
    assert_eq!(a.last_logits().unwrap().values, boundary_logits);
    a.append_ids(&[9]).unwrap();
    assert_eq!(state_bytes(&a), b_state);
    assert_eq!(a.last_logits().unwrap().values, b_logits);
    a.reset_for_prompt(&[1, 5, 7]).unwrap();
    assert_eq!(a.committed_ids(), [1, 5, 7]);
    assert_eq!(a.last_logits().unwrap().values, boundary_logits);
    assert!(a.reset_for_prompt(&[32]).is_err()); // validation before mutation
    assert_eq!(state_bytes(&a), boundary);
    for fault in [Fault::AfterLayer(1), Fault::PanicAfterLayer(1)] {
        a.fault = Some(fault);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            a.append_ids(&[9]).map(|_| ())
        }));
        if fault == Fault::PanicAfterLayer(1) { assert!(result.is_err()); }
        else { assert!(result.unwrap().is_err()); }
        assert_eq!(a.status(), Status::Poisoned);
        assert!(a.saved_prefix().is_none() && !a.pending.is_empty());
        let held = a.pending.len();
        a.fault = Some(Fault::Reset);
        assert!(a.restore_prefix().is_err());
        assert_eq!(a.pending.len(), held, "uncertain restore retains operands");
        a.fault = None;
        a.restore_prefix().unwrap();
        assert_eq!(state_bytes(&a), boundary);
    }

    // A later Mamba state cannot serve a shorter common prefix. Miss restores
    // genuinely empty full-capacity storage, not a C-P buffer labelled C.
    b.reset_for_prompt(&[1, 5]).unwrap();
    assert!(b.saved_prefix().is_none() && b.committed_ids().is_empty());
    assert_eq!(b.prefix_ownership().unwrap().private_kv_bytes, 8 * 1024);
    b.append_ids(&[1, 5, 7, 9]).unwrap();
    for (&reference, &segmented) in b.last_logits().unwrap().values.iter().zip(&b_logits) {
        assert!((reference - segmented).abs() <= 0.02 + 0.02 * reference.abs());
    }
    // One-run versus two-run grouping is a numerical comparison, not a claim
    // of bitwise equality. Same-layout forks above compare exact state/output.
    let mut full = Session::from_prefix(model.clone(), saved.clone(), Limits { capacity: 3, ..l }).unwrap();
    assert_eq!(full.prefix_ownership().unwrap().private_kv_bytes, 0);
    assert_eq!(full.last_logits().unwrap().values, boundary_logits);
    assert!(full.append_ids(&[2]).is_err());
    full.reset().unwrap();
    assert_eq!(full.prefix_ownership().unwrap().private_kv_bytes, 3 * 1024);
    full.append_ids(&[2, 3, 4]).unwrap();
    drop(full);

    let other = Session::from_snapshot(snapshot, source, &layers, CudaDevice { index: 0 }, l).unwrap();
    // Same source labels and device are insufficient: this is another uploaded
    // model owner. Rejection precedes any new fork allocation.
    assert!(!saved.compatible_with(other.device_model()));
    assert!(Session::from_prefix(other.device_model().clone(), saved.clone(), l).is_err());
    drop(other);
    drop(parent);
    drop(b);
    let origin = a.stream;
    let a = std::thread::spawn(move || {
        let caller = StreamId::current();
        assert_ne!(caller, origin);
        a.restore_prefix().unwrap();
        a.append_ids(&[9]).unwrap();
        assert_eq!(StreamId::current(), caller, "stream binding restores the caller");
        a
    }).join().unwrap();
    assert_eq!(state_bytes(&a), b_state);
    drop(a);
    let mut survivor = Session::from_prefix(model, saved, l).unwrap();
    survivor.append_ids(&[9]).unwrap();
    assert_eq!(survivor.last_logits().unwrap().values, b_logits);
}

#[test]
#[ignore = "reserved CUDA: synthetic cancelled/failed seed and origin rejection; no checkpoint"]
fn session_gpu_prefix_never_publishes_partial_or_failed_completion() {
    let l = Limits { capacity: 512, ..limits() };
    let mut c = config();
    c.max_position_embeddings = 512;
    let o = resident_ownership(&c, l).unwrap();
    assert!(o.fixed_model_bytes + 2 * o.private_state_bytes + o.private_expert_cache_bytes
        + l.max_chunk_bytes < 512 << 20);
    let (snapshot, source, layers) = fixture_capacity(512);
    let ids: Vec<_> = (0..257).map(|i| (i % 31) as u32).collect();
    let mut session = Session::from_snapshot_seed(snapshot, source, &layers,
        CudaDevice { index: 0 }, l, &ids).unwrap();
    assert_eq!(session.prefill_ids(&ids, &mut || true).unwrap(), PrefillOutcome::Cancelled);
    assert!(session.saved_prefix().is_none());
    assert_eq!(session.status(), Status::Ready);
    let mut calls = 0;
    assert_eq!(session.prefill_ids(&ids, &mut || { calls += 1; calls >= 6 }).unwrap(), PrefillOutcome::Cancelled);
    assert_eq!(session.status(), Status::Poisoned);
    assert!(session.saved_prefix().is_none() && session.span_pending.is_some());
    let model = session.device_model().clone();
    session.reset().unwrap();
    drop(session);
    for fault in [Fault::AfterLayer(1), Fault::PanicAfterLayer(1)] {
        let mut failed = Session::from_device_model_seed(model.clone(), l, &ids).unwrap();
        failed.fault = Some(fault);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
            failed.prefill_ids(&ids, &mut || false)));
        if fault == Fault::PanicAfterLayer(1) { assert!(result.is_err()); }
        else { assert!(result.unwrap().is_err()); }
        assert!(failed.saved_prefix().is_none());
        assert_eq!(failed.status(), Status::Poisoned);
        assert!(failed.span_pending.is_some());
        failed.fault = Some(Fault::Reset);
        assert!(failed.reset().is_err());
        assert!(failed.span_pending.is_some());
        failed.fault = None;
        failed.reset().unwrap();
    }
    // A checked UnsupportedStrides rejection on an existing valid buffer,
    // queued exactly before prefix publication's origin fence. No invalid
    // pointer/OOB copy and no inference after this terminal observation.
    let mut failed = Session::from_device_model_seed(model, l, &[1, 5, 7]).unwrap();
    failed.fault = Some(Fault::PrefixUpload);
    let error = failed.prefill_ids(&[1, 5, 7], &mut || false).unwrap_err();
    assert!(error.contains("session CUDA completion"));
    assert!(failed.saved_prefix().is_none());
    assert_eq!(failed.status(), Status::Poisoned);
}
