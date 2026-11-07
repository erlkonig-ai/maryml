//! Scoped diagnostics. GPU tests require explicit reservation and a NEW output
//! directory; retain raw tensors for an independent stock GPU comparison.
//! Native partition agreement alone is not acceptance of the attention model.
use super::*;
use crate::models::inkling::flash;
use std::{fs, path::PathBuf};

fn limits(capacity: usize) -> Limits {
    Limits {
        context_limit: 1 << 20,
        capacity,
        max_resident_bytes: 64 << 20,
        max_working_bytes: 64 << 20,
    }
}

/// Peer projection working bytes for T tokens: FP8 input, F32 accumulator and
/// BF16 output of the four fixed projections.
fn projection_bytes(tokens: usize) -> usize {
    SHAPES.iter().map(|[n, k]| tokens * k + tokens * n * 6).sum()
}

#[test]
fn attention_every_width_plans_tensor_cores_with_exact_budget() {
    assert_eq!(EXECUTION, "mma-hilo-p-t1-v1");
    for retired in ["scalar-f32-v1", "mma-bf16-p-v1", "mma-hilo-p-v1"] {
        assert_ne!(EXECUTION, retired);
    }
    let l = Limits { max_working_bytes: usize::MAX, max_resident_bytes: 2 << 30, ..limits(1 << 20) };
    for start in [0, 3, 257, 65_279, 65_280, 524_032, (1 << 20) - 256] {
        for prefix in [0, start.min(8192), start] {
            for tokens in 1..=MAX_CHUNK {
                let p = plan_with_prefix(start, tokens, l, prefix).unwrap();
                let end = start + tokens;
                let expected = if tokens == 1 || end < 65536 { 4 } else { 8 };
                assert_eq!((p.tokens, p.end, p.mma.warps), (tokens, end, expected),
                    "start={start}, prefix={prefix}, tokens={tokens}");
                let spans = if prefix == 0 { vec![end] } else { vec![prefix, end - prefix] };
                assert_eq!(p.working_bytes, projection_bytes(tokens)
                    + attention_mma::Plan::new(&spans, tokens, start, expected).unwrap().working_bytes);
            }
        }
    }
    for (start, t, prefix) in [(0, 1, 0), ((1 << 20) - 1, 1, 8192), (0, 3, 0), (65535 - 256, 256, 8192),
        (65536 - 256, 256, 8192), (130560, 256, 8192), (131072, 17, 8192)] {
        let p = plan_with_prefix(start, t, l, prefix).unwrap();
        assert!(plan_with_prefix(start, t, Limits { max_working_bytes: p.working_bytes - 1, ..l }, prefix).is_err());
        assert!(plan_with_prefix(start, t, Limits { max_working_bytes: p.working_bytes, ..l }, prefix).is_ok());
    }
}

#[test]
#[ignore = "reserved GPU: MMA owner at T1 and T>1, prefix/private append and poison/reset; named operands under 128MiB"]
fn gpu_attention_mma_prefix_decode_and_pending_ownership() {
    let mut evidence = Evidence::new("NEMOTRON_MMA_INTEGRATION_EVIDENCE");
    let named = 2 * admission(limits(6)).unwrap().resident_bytes
        + 2 * plan_with_prefix(3, 2, limits(6), 3).unwrap().working_bytes;
    assert!(named < 128 << 20); // before any upload; runtime overhead is external
    let weights: [Vec<u8>; 4] = std::array::from_fn(|i| {
        let [n, k] = SHAPES[i];
        let mut bytes = vec![0; n * k];
        for row in 0..n { bytes[row * k + row % k] = 0x38; }
        bytes
    });
    let operations = || std::array::from_fn(|i| Operation {
        e4m3fn: weights[i].as_slice(), input_scale: 0.125, weight_scale: 0.5,
    });
    let device = CudaDevice { index: 0 };
    let mut seed = Attention::upload(device.clone(), source(), operations(), limits(3)).unwrap();
    let client = seed.weights.client.clone();
    let input = tensor(&client, &device, &vec![0.25; 3 * HIDDEN], &[1, 3, HIDDEN]);
    seed.enqueue(&input, 0).unwrap();
    assert!(seed.has_pending());
    seed.commit().unwrap();
    let prefix = seed.seal_kv().unwrap();
    let mut fork = Attention::from_prefix(seed.weights.clone(), prefix.clone(), limits(6)).unwrap();
    // Another owner of identical bytes is not this prefix's owner.
    let other = Attention::upload(device.clone(), source(), operations(), limits(6)).unwrap();
    assert!(Attention::from_prefix(other.weights.clone(), prefix, limits(6)).is_err());
    let suffix = tensor(&client, &device, &vec![0.5; 2 * HIDDEN], &[1, 2, HIDDEN]);
    let output = fork.enqueue(&suffix, 3).unwrap();
    assert_eq!(fork.attended(), 3, "queued is not committed");
    fork.commit().unwrap();
    let values = evidence.tensor("suffix-t2.bf16", &output.values);
    assert!(values.iter().all(|v| v.is_finite()));
    // T1 decode over the prefix and private runs, on the same MMA route.
    let token = tensor(&client, &device, &vec![0.75; HIDDEN], &[1, 1, HIDDEN]);
    let decode = fork.enqueue(&token, 5).unwrap();
    fork.commit().unwrap();
    let values = evidence.tensor("decode-t1.bf16", &decode.values);
    assert!(values.iter().all(|v| v.is_finite()));
    assert_eq!((seed.attended(), fork.attended()), (3, 6));
    fork.reset_capacity(6).unwrap();
    fork.enqueue(&suffix, 0).unwrap();
    fork.invalidate(); // logical cancellation never means device drain
    assert!(fork.enqueue(&suffix, 0).is_err());
    fork.reset().unwrap();
    assert_eq!(fork.attended(), 0);
    assert!(!fork.is_poisoned());
    cubecl::future::block_on(client.sync()).unwrap();
    evidence.finish(serde_json::json!({"scope":"ownership and finite outputs, not capability",
        "execution":EXECUTION,"named_bytes":named}));
}

#[test]
fn attention_geometry_budgets_and_positions_before_device() {
    let a = admission(limits(1024)).unwrap();
    assert_eq!(a.weight_bytes, 35_651_584);
    assert_eq!(a.kv_bytes, 1 << 20);
    assert_eq!(a.resident_bytes, a.weight_bytes + a.kv_bytes);
    let mut large = limits(1 << 20);
    large.max_resident_bytes = 2 << 30;
    assert_eq!(admission(large).unwrap().kv_bytes, 1 << 30);
    // 1024 is a witness size, not an architectural ceiling.
    assert!(admission(limits(1025)).is_ok());
    for capacity in [0, (1 << 20) + 1, usize::MAX] {
        assert!(admission(limits(capacity)).is_err());
    }
    assert!(
        admission(Limits {
            max_resident_bytes: a.resident_bytes - 1,
            ..limits(1024)
        })
        .is_err()
    );
    assert!(plan(usize::MAX, 1, limits(1024)).is_err());
    assert!(plan(1023, 2, limits(1024)).is_err());
    for tokens in [0, MAX_CHUNK + 1, usize::MAX] {
        assert!(plan(0, tokens, limits(1024)).is_err());
    }
    for (start, tokens) in [(0, 1), (0, 3), (511, 3), (1021, 3), (0, 256)] {
        let p = plan(start, tokens, limits(1024)).unwrap();
        assert_eq!(p.end, start + tokens);
        assert!(
            plan(
                start,
                tokens,
                Limits {
                    max_working_bytes: p.working_bytes - 1,
                    ..limits(1024)
                }
            )
            .is_err()
        );
        assert!(
            plan(
                start,
                tokens,
                Limits {
                    max_working_bytes: p.working_bytes,
                    ..limits(1024)
                }
            )
            .is_ok()
        );
    }
}

#[test]
fn attention_budget_counts_prefix_and_private_run_rounding_at_t1_and_full_tiles() {
    let l = Limits { max_resident_bytes: 2 << 30, max_working_bytes: usize::MAX, ..limits(1 << 20) };
    // T tokens: projections 68,608 T + Q 24,576 T = 93,184 T B; every
    // reduction slot adds 16,640 T B of F32 partials and statistics.
    // Splits: TARGET_CUBES 1024 over (query tiles x 2 KV heads), shared by
    // span, at most one per 128 keys; B4 tiles are ceil(T/4), B8 ceil(T/8).
    let last = (1 << 20) - 1;
    for (prefix, start, tokens, expected_splits, expected_bytes) in [
        (0, 0, 256, vec![2], 32_374_784),
        (0, 16_128, 256, vec![8], 57_933_824),
        (0, 524_032, 256, vec![16], 92_012_544),
        (257, 511, 256, vec![3, 4], 53_673_984),
        (524_032, 524_032, 256, vec![16, 1], 96_272_384),
        (0, last, 1, vec![512], 8_612_864),
        (1_048_560, last, 1, vec![512, 1], 8_629_504),
    ] {
        let p = plan_with_prefix(start, tokens, l, prefix).unwrap();
        let spans = if prefix == 0 { vec![p.end] } else { vec![prefix, p.end - prefix] };
        let tiles = attention_mma::query_tiles(tokens, p.mma.warps);
        assert_eq!(flash::key_run_splits(&spans, tiles, KV_HEADS), expected_splits);
        assert_eq!(p.working_bytes, expected_bytes);
        assert!(plan_with_prefix(start, tokens,
            Limits { max_working_bytes: expected_bytes - 1, ..l }, prefix).is_err());
        assert!(plan_with_prefix(start, tokens,
            Limits { max_working_bytes: expected_bytes, ..l }, prefix).is_ok());
    }
}

#[test]
fn attention_pending_failure_cancel_and_reset_protocol() {
    let mut s = Sequence::default();
    assert!(s.ready(1).is_err());
    assert!(
        s.commit(|| panic!("no fence without pending work"))
            .is_err()
    );
    s.phase = Phase::Poisoned; // Set immediately before first dispatch.
    assert!(s.ready(0).is_err());
    s.phase = Phase::Pending(3);
    assert!(s.ready(0).is_err());
    assert_eq!(s.attended, 0);
    assert!(
        s.commit(|| Err("injected completion error".into()))
            .is_err()
    );
    assert_eq!(s.phase, Phase::Poisoned);
    assert_eq!(s.attended, 0);
    assert!(s.reset(|| Err("injected reset error".into())).is_err());
    assert_eq!(s.phase, Phase::Poisoned);
    s.reset(|| Ok(())).unwrap();
    s.ready(0).unwrap();
    s.phase = Phase::Pending(3);
    s.commit(|| Ok(())).unwrap();
    s.ready(3).unwrap();
    s.phase = Phase::Poisoned; // Owner invalidates on cancellation/other-layer error.
    assert_eq!(s.attended, 3);
    assert!(s.ready(3).is_err());
    s.reset(|| Ok(())).unwrap();
    assert_eq!(s.attended, 0);
}

#[test]
fn attention_prefix_split_admission_uses_actual_runs_and_absolute_positions() {
    let l = limits(1024);
    // Each independently launched run owns at least one partial. A 128-row
    // prefix plus one new row therefore needs two, not ceil(129/128) by luck:
    // use the small 3+1 witness where one contiguous run has only one slot.
    let one = plan(3, 1, l).unwrap();
    let two = plan_with_prefix(3, 1, l, 3).unwrap();
    assert_eq!(flash::key_run_splits(&[4], 1, KV_HEADS), [1]);
    assert_eq!(flash::key_run_splits(&[3, 1], 1, KV_HEADS), [1, 1]);
    assert_eq!(two.working_bytes - one.working_bytes, (HEADS * HEAD_DIM + HEADS * 2) * 4);
    assert_eq!(two.end, 4);
    assert!(plan_with_prefix(2, 1, l, 3).is_err());
    assert!(plan_with_prefix(1024, 1, l, 1024).is_err());
    for (prefix, start, count) in [(3, 3, 1), (3, 4, 7), (257, 511, 256), (1023, 1023, 1)] {
        let exact = plan_with_prefix(start, count, l, prefix).unwrap();
        assert!(plan_with_prefix(start, count, Limits { max_working_bytes: exact.working_bytes - 1, ..l }, prefix).is_err());
        assert!(plan_with_prefix(start, count, Limits { max_working_bytes: exact.working_bytes, ..l }, prefix).is_ok());
    }
    for (start, count) in [(0, 1), (0, 256), (511, 5)] {
        assert_eq!(plan(start, count, l).unwrap().working_bytes,
            plan_with_prefix(start, count, l, 0).unwrap().working_bytes);
    }
}

fn source() -> SourceBinding {
    use triblespace::prelude::genid;
    SourceBinding {
        model_root: *genid(),
        layer: *genid(),
        operations: [*genid(), *genid(), *genid(), *genid()],
    }
}

#[test]
fn attention_invalid_upload_has_no_device_work() {
    let op = || Operation {
        e4m3fn: &[],
        input_scale: 1.0,
        weight_scale: 1.0,
    };
    // All lengths are rejected before Projection::upload/CudaRuntime::client.
    assert!(
        Attention::upload(
            CudaDevice { index: 0 },
            source(),
            [op(), op(), op(), op()],
            limits(8)
        )
        .is_err()
    );
}

fn tensor(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    data: &[f32],
    shape: &[usize],
) -> CudaTensor {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    let handle = client.create_from_slice(&bytes);
    CudaTensor::new_contiguous(
        client.clone(),
        device.clone(),
        shape.into(),
        handle,
        DType::BF16,
    )
}

// Ephemeral output inventory for THIS finite diagnostic, not a runtime cache.
struct Evidence {
    root: PathBuf,
    entries: Vec<serde_json::Value>,
    bytes: usize,
}
impl Evidence {
    fn new(variable: &str) -> Self {
        let root = PathBuf::from(
            std::env::var(variable).expect("explicit new evidence directory required"),
        );
        fs::create_dir(&root).expect("evidence destination must be new");
        Self {
            root,
            entries: Vec::new(),
            bytes: 0,
        }
    }
    fn raw(&mut self, name: &str, bytes: &[u8], dtype: &str, shape: &[usize]) {
        self.bytes = self.bytes.checked_add(bytes.len()).unwrap();
        assert!(
            self.bytes <= 96 << 20 && self.entries.len() < 128,
            "bounded fixture output exceeded"
        );
        assert!(!name.contains('/') && !name.contains(".."));
        fs::write(self.root.join(name), bytes).unwrap();
        self.entries.push(serde_json::json!({"path":name,"dtype":dtype,"shape":shape,"bytes":bytes.len(),"blake3":blake3::hash(bytes).to_hex().to_string()}));
    }
    fn tensor(&mut self, name: &str, t: &CudaTensor) -> Vec<f64> {
        let bytes = t.client.read_one(t.handle.clone()).unwrap();
        let count = t.meta.shape().as_slice().iter().product::<usize>();
        let width = if t.dtype == DType::BF16 {
            2
        } else {
            assert_eq!(t.dtype, DType::F32);
            4
        };
        assert!(bytes.len() >= count * width);
        let bytes = &bytes[..count * width];
        self.raw(
            name,
            bytes,
            if width == 2 { "BF16_LE" } else { "F32_LE" },
            t.meta.shape().as_slice(),
        );
        bytes
            .chunks_exact(width)
            .map(|b| {
                if width == 2 {
                    bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f64()
                } else {
                    f32::from_le_bytes(b.try_into().unwrap()) as f64
                }
            })
            .collect()
    }
    fn cache(&mut self, prefix: &str, a: &Attention) {
        for (label, handle) in [("k", &a.keys), ("v", &a.values)] {
            let bytes = a.weights.client.read_one(handle.clone()).unwrap();
            // Discard uninitialized/previous-context tail bytes, never export.
            let live = a.attended() * KV_WIDTH * 2;
            self.raw(
                &format!("{prefix}-{label}-cache.bf16"),
                &bytes[..live],
                "BF16_LE",
                &[a.attended(), KV_HEADS, HEAD_DIM],
            );
        }
    }
    fn finish(&self, report: serde_json::Value) {
        fs::write(
            self.root.join("report.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        fs::write(
            self.root.join("tensors.json"),
            serde_json::to_vec_pretty(&self.entries).unwrap(),
        )
        .unwrap();
    }
}

fn metrics(a: &[f64], b: &[f64]) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    let (mut max_abs, mut max_relative, mut squares, mut differing, mut nonfinite, mut flags) =
        (0.0f64, 0.0f64, 0.0f64, 0usize, 0usize, 0usize);
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            nonfinite += 1;
            continue;
        }
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        max_relative = max_relative.max(d / y.abs().max(1e-8));
        squares += d * d;
        differing += usize::from(x != y);
        flags += usize::from(d > 0.03 + 0.03 * y.abs());
    }
    serde_json::json!({"elements":a.len(),"max_abs":max_abs,"max_relative_floor_1e_8":max_relative,"rms":(squares/a.len() as f64).sqrt(),"differing":differing,"nonfinite":nonfinite,"diagnostic_band_flags":flags,"band":"0.03 + 0.03*abs(reference); descriptive, not acceptance"})
}

#[test]
#[ignore = "reserved GPU: Super-shaped FP8+attention fixtures; independent GPU comparison still required"]
fn gpu_attention_component_partitions_and_state() {
    let mut evidence = Evidence::new("NEMOTRON_ATTENTION_EVIDENCE");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let weights: [Vec<u8>; 4] = std::array::from_fn(|op| {
        let [n, k] = SHAPES[op];
        let mut bytes = vec![0; n * k];
        for row in 0..n {
            let col = (row * 13 + op * 7) % k;
            bytes[row * k + col] = if (row + op) % 3 == 0 { 0xb8 } else { 0x38 }; // +/-1
            bytes[row * k + (col + 11) % k] = 0x30; // +0.5
        }
        evidence.raw(&format!("weight-{op}.e4m3"), &bytes, "E4M3FN", &[n, k]);
        bytes
    });
    let scales = [(0.125, 0.5), (0.25, 0.25), (0.5, 0.5), (0.125, 0.5)];
    let ops = std::array::from_fn(|i| Operation {
        e4m3fn: &weights[i],
        input_scale: scales[i].0,
        weight_scale: scales[i].1,
    });
    let binding = source();
    let mut a = Attention::upload(device.clone(), binding, ops, limits(5)).unwrap();
    let input: Vec<f32> = (0..5 * HIDDEN)
        .map(|i| (((i / HIDDEN) * 3 + i % HIDDEN * 5) % 17) as f32 * 0.125 - 1.0)
        .collect();
    let full = tensor(&client, &device, &input, &[1, 5, HIDDEN]);
    evidence.tensor("input.bf16", &full);
    // Invalid descriptors are rejected before launching any projection/copy.
    let bad_rank = CudaTensor::new_contiguous(
        client.clone(),
        device.clone(),
        [5, HIDDEN].into(),
        full.handle.clone(),
        DType::BF16,
    );
    assert!(a.enqueue(&bad_rank, 0).is_err());
    let mut bad_dtype = full.clone();
    bad_dtype.dtype = DType::F32;
    assert!(a.enqueue(&bad_dtype, 0).is_err());
    let mut other_device = full.clone();
    other_device.device = CudaDevice { index: 1 }; // Descriptor only; never initialize GPU1.
    assert!(a.enqueue(&other_device, 0).is_err());
    let mut bad_span = full.clone();
    bad_span.handle.offset_end = Some(bad_span.handle.size() + 1);
    assert!(a.enqueue(&bad_span, 0).is_err());
    let mut start_alias = full.clone();
    start_alias.handle.offset_start = Some(2);
    assert!(a.enqueue(&start_alias, 0).is_err());
    let saved_budget = a.limits.max_working_bytes;
    a.limits.max_working_bytes = 0;
    assert!(a.enqueue(&full, 0).is_err());
    a.limits.max_working_bytes = saved_budget;
    assert_eq!(a.attended(), 0);
    assert!(!a.is_poisoned() && !a.has_pending());
    let mut joined = Vec::new();
    let mut reports = Vec::new();
    for (arm, counts) in [vec![3, 1, 1], vec![2, 1, 1, 1], vec![1, 1, 1, 1, 1]]
        .into_iter()
        .enumerate()
    {
        a.reset().unwrap();
        assert_eq!(a.source(), binding);
        let mut arm_output = Vec::new();
        for (part, n) in counts.into_iter().enumerate() {
            let start = a.attended();
            let x = tensor(
                &client,
                &device,
                &input[start * HIDDEN..(start + n) * HIDDEN],
                &[1, n, HIDDEN],
            );
            assert!(a.enqueue(&x, start + 1).is_err());
            assert!(!a.is_poisoned() && !a.has_pending());
            let out = a.enqueue(&x, start).unwrap();
            assert_eq!(a.attended(), start);
            assert!(a.has_pending());
            assert!(a.enqueue(&x, start).is_err());
            a.commit().unwrap();
            let prefix = format!("arm{arm}-part{part}");
            let y = evidence.tensor(&format!("{prefix}-output.bf16"), &out.values);
            evidence.tensor(&format!("{prefix}-q.bf16"), &out.stages.q);
            evidence.tensor(&format!("{prefix}-k.bf16"), &out.stages.k);
            evidence.tensor(&format!("{prefix}-v.bf16"), &out.stages.v);
            evidence.tensor(&format!("{prefix}-attention.bf16"), &out.stages.attention);
            arm_output.extend_from_slice(&y);
            reports.push(serde_json::json!({"arm":arm,"part":part,"start":start,"tokens":n,"end":a.attended(),"working_bytes":out.working_bytes,"finite":y.iter().all(|v|v.is_finite())}));
        }
        evidence.cache(&format!("arm{arm}"), &a);
        let x = tensor(&client, &device, &input[..HIDDEN], &[1, 1, HIDDEN]);
        assert!(a.enqueue(&x, 5).is_err()); // Capacity refusal preserves Ready.
        assert!(!a.is_poisoned() && a.attended() == 5);
        joined.push(arm_output);
    }
    // Enqueued cancellation cannot be treated as an empty cache.
    a.reset().unwrap();
    let x = tensor(&client, &device, &input[..HIDDEN], &[1, 1, HIDDEN]);
    let cancelled = a.enqueue(&x, 0).unwrap();
    a.invalidate();
    assert!(a.commit().is_err() && a.enqueue(&x, 0).is_err());
    a.reset().unwrap(); // Drains cancelled work before exposing empty history.
    let fresh = a.enqueue(&x, 0).unwrap();
    a.commit().unwrap();
    let after_reset = evidence.tensor("after-reset-output.bf16", &fresh.values);
    drop(cancelled);
    let comparisons = vec![
        metrics(&joined[1], &joined[0]),
        metrics(&joined[2], &joined[0]),
        metrics(&after_reset, &joined[0][..HIDDEN]),
    ];
    evidence.finish(serde_json::json!({"scope":"synthetic full Super attention geometry, not model acceptance","scales_qkvo":scales,"heads":32,"kv_heads":2,"head_dim":128,"capacity":5,"context_limit":1<<20,"phases":reports,"comparisons":comparisons,"native_policy":"static FP8 MMA projections; BF16 KV; F32 online attention; BF16 before O"}));
    assert!(
        joined
            .iter()
            .flatten()
            .chain(after_reset.iter())
            .all(|x| x.is_finite()),
        "nonfinite outputs retained"
    );
}

#[test]
#[ignore = "reserved GPU: real 514-key online attention, GQA/causality/tail diagnostic and raw fixtures"]
fn gpu_online_attention_514_keys() {
    let mut evidence = Evidence::new("NEMOTRON_ATTENTION_BOUNDARY");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let capacity = 544;
    let keys = 514;
    let q = tensor(&client, &device, &vec![0.0; 2 * HIDDEN], &[1, 2, HIDDEN]);
    let k: Vec<f32> = (0..capacity * KV_WIDTH)
        .map(|i| {
            if i / KV_WIDTH >= keys {
                f32::NAN
            } else {
                (i % 11) as f32 / 32.0
            }
        })
        .collect();
    let v: Vec<f32> = (0..capacity * KV_WIDTH)
        .map(|i| {
            let row = i / KV_WIDTH;
            let h = (i % KV_WIDTH) / HEAD_DIM;
            if row >= keys {
                f32::NAN
            } else if row == 0 {
                128.0 * (h + 1) as f32
            } else if row == 513 {
                64.0 * (h + 1) as f32
            } else {
                0.0
            }
        })
        .collect();
    let k = tensor(&client, &device, &k, &[capacity, KV_WIDTH]);
    let v = tensor(&client, &device, &v, &[capacity, KV_WIDTH]);
    evidence.tensor("q.bf16", &q);
    evidence.tensor("k-reserved.bf16", &k); // Tail is explicit synthetic NaNs, not uninitialized memory.
    evidence.tensor("v-reserved.bf16", &v);
    let run = KeyRun {
        k: &k.handle,
        v: &v.handle,
        k_scales: None,
        v_scales: None,
        rows: capacity,
        base: 0,
        lo: 0,
        hi: keys,
        row0: 0,
    };
    let mut reports = Vec::new();
    // T2 and T1 decode at the boundary, at both plane counts. A position sees
    // keys 0..=pos only: row 0 contributes 128(kv+1), row 513 64(kv+1).
    for (tokens, q0) in [(2, 512), (1, 0), (1, 512), (1, 513)] {
        for warps in [4, 8] {
            let plan = attention_mma::Plan::new(&[keys], tokens, q0, warps).unwrap();
            let out = attention_mma::launch_planned(&client, &q.handle, &[run],
                1.0 / (HEAD_DIM as f32).sqrt(), &plan).unwrap();
            let out = CudaTensor::new_contiguous(
                client.clone(),
                device.clone(),
                [1, tokens, HIDDEN].into(),
                out,
                DType::F32,
            );
            let got = evidence.tensor(&format!("t{tokens}-q0{q0}-b{warps}-output.f32"), &out);
            let expected: Vec<f64> = (0..tokens * HIDDEN)
                .map(|i| {
                    let pos = q0 + i / HIDDEN;
                    let kv = (i % HIDDEN) / HEAD_DIM / 16;
                    let sum = if pos >= 513 { 192.0 } else { 128.0 };
                    sum * (kv + 1) as f64 / (pos + 1) as f64
                })
                .collect();
            let mut report = metrics(&got, &expected);
            report["tokens"] = tokens.into();
            report["q0"] = q0.into();
            report["warps"] = warps.into();
            reports.push(report);
        }
    }
    let uniform_mean_ok = reports
        .iter()
        .all(|r| r["nonfinite"] == 0 && r["max_abs"].as_f64().unwrap() < 1e-5);
    evidence.finish(serde_json::json!({"execution":EXECUTION,"live_keys":514,"capacity":544,"eff":0,"window":null,"comparisons_to_uniform_visible_mean":reports}));
    // Q=0 gives a hand-derived uniform distribution. This specifically detects
    // future-row leakage, an added sink, wrong GQA head, or reading NaN padding.
    assert!(uniform_mean_ok);
}
