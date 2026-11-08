//! Scoped host/control tests and ignored native synthetic GPU fixtures.
//! Finite/shape/refusal assertions are contracts. Native partition agreement
//! alone is NOT an independent numerical/model acceptance test.
use super::*;
use std::{cell::Cell, fs, path::PathBuf};
use triblespace::prelude::genid;

fn source() -> SourceBinding {
    SourceBinding {
        model_root: *genid(),
        layer: *genid(),
    }
}

#[test]
#[ignore = "reserved CUDA: ordered FMA tiles versus existing scatter and independent F32 recurrence"]
fn moe_gpu_prefill_ordered_tiles_match_scatter() {
    let client = CudaRuntime::client(&CudaDevice { index: 0 });
    let tokens = 17;
    let experts = 3;
    let mut y = Vec::new();
    let mut weights = Vec::new();
    for expert in 0..experts {
        for row in 0..tokens {
            weights.push((row + expert + 1) as f32 / 29.0);
            for col in 0..LATENT {
                y.push(bf16::from_f32(match expert {
                    0 => 1024.0 + (col % 7) as f32,
                    1 => -1024.0 + (row % 5) as f32,
                    _ => 0.5 + (col % 3) as f32 / 8.0,
                }));
            }
        }
    }
    let selected = client.create_from_slice(bf16::as_bytes(&y));
    let weight = client.create_from_slice(f32::as_bytes(&weights));
    let row_ids: Vec<u32> = (0..tokens).flat_map(|row|
        (0..experts).map(move |expert| (expert * tokens + row) as u32)).collect();
    let ids = client.create_from_slice(u32::as_bytes(&row_ids));
    let counts = client.create_from_slice(u32::as_bytes(&vec![experts as u32; tokens]));
    let reference = moegroup::scatter_weighted_bf16(&client, &selected, &weight, &ids, &counts,
        tokens * experts, tokens, LATENT, experts);
    let accumulator = client.create_from_slice(f32::as_bytes(&vec![0.0; tokens * LATENT]));
    let mut held = Vec::new();
    for expert in 0..experts {
        for start in [0, 16] {
            let rows = (tokens - start).min(16);
            let from = expert * tokens + start;
            let low = client.create_from_slice(bf16::as_bytes(&y[from * LATENT..(from + rows) * LATENT]));
            let idx: Vec<i32> = (start..start + rows).map(|v| v as i32).collect();
            let idx = client.create_from_slice(i32::as_bytes(&idx));
            let w = client.create_from_slice(f32::as_bytes(&weights[from..from + rows]));
            let len = rows * LATENT;
            unsafe {
                ordered_update::launch_unchecked::<CudaRuntime>(&client,
                    CubeCount::new_1d(len.div_ceil(THREADS as usize) as u32), CubeDim::new_1d(THREADS),
                    ArrayArg::from_raw_parts(low.clone(), len), ArrayArg::from_raw_parts(idx.clone(), rows),
                    ArrayArg::from_raw_parts(w.clone(), rows),
                    ArrayArg::from_raw_parts(accumulator.clone(), tokens * LATENT), len);
            }
            held.extend([low, idx, w]);
        }
    }
    fence(&client).unwrap();
    let actual = client.read_one(accumulator).unwrap();
    let reference = client.read_one(reference).unwrap();
    assert_eq!(&actual[..tokens * LATENT * 4], &reference[..tokens * LATENT * 4]);
    for row in 0..tokens {
        for col in 0..LATENT {
            let expected = (0..experts).fold(0.0f32, |a, expert| {
                let r = expert * tokens + row;
                y[r * LATENT + col].to_f32().mul_add(weights[r], a)
            });
            let offset = (row * LATENT + col) * 4;
            assert_eq!(&actual[offset..offset + 4], &expected.to_le_bytes());
        }
    }
    drop(held);
}
fn limits() -> Limits {
    Limits {
        max_resident_bytes: 2 << 30,
        max_working_bytes: 64 << 20,
    }
}
fn record(t: usize) -> Vec<f32> {
    let mut data = vec![0.0; t * ROUTE_WIDTH];
    for row in 0..t {
        for j in 0..TOP_K {
            data[row * ROUTE_WIDTH + j] = ((row * 3 + j) % EXPERTS) as f32;
            data[row * ROUTE_WIDTH + TOP_K + j] = 5.0 / TOP_K as f32;
        }
    }
    data
}

#[test]
fn moe_host_admission_calibrated_scales_and_finite_operands() {
    let a = admission(limits()).unwrap();
    assert_eq!(a.fixed_bytes, 45_746_184);
    // 512 pairs of four device planes: 2 * (2688 * 1024 / 2 + 2688 * 1024 / 16).
    assert_eq!(expert_device_bytes().unwrap(), 3_096_576);
    assert_eq!(a.bank_bytes, 1_585_446_912);
    let exact = a.fixed_bytes + a.bank_bytes;
    assert!(admission(Limits { max_resident_bytes: exact, ..limits() }).is_ok());
    assert!(admission(Limits { max_resident_bytes: exact - 1, ..limits() }).is_err());
    for t in [1, 3, 16, 17, 256] {
        assert!(working_bytes(t).unwrap() < limits().max_working_bytes);
    }
    for t in [0, 257, usize::MAX] {
        assert!(working_bytes(t).is_err());
    }
    assert!(bytes(&[usize::MAX, 2], 2).is_err());
    for (input, weight) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (f32::NAN, 1.0),
        (1.0, f32::INFINITY),
        (f32::MAX, 2.0),
        (f32::from_bits(1), 0.5),
    ] {
        assert!(calibrated_scale(input, weight).is_err());
    }
    assert_eq!(calibrated_scale(1.0, 0.25).unwrap(), 0.25);
    assert_eq!(calibrated_scale(0.5, 0.25).unwrap(), 0.125);
    let mut data = packed(8, 64, 0.25);
    validate_packed(&operation(&data, 0.5), 8, 64).unwrap();
    data[8 * 64 / 2] = 0x7f;
    assert!(validate_packed(&operation(&data, 0.5), 8, 64).is_err());
    assert!(finite_bf16(&[0x80, 0x7f], 1, 1).is_err());
    assert!(finite_bf16(&[0xc0, 0x7f], 1, 1).is_err());
    let invalid_bias = vec![f32::NAN; EXPERTS];
    let invalid_fixed = FixedOperands {
        operations: [*genid(), *genid(), *genid(), *genid()],
        router_bf16: &[],
        correction_bias: &invalid_bias,
        fc1_bf16: &[],
        fc2_bf16: &[],
        shared_up: operation(&[], 1.0),
        shared_down: operation(&[], 1.0),
    };
    assert!(
        validate_fixed(&invalid_fixed)
            .unwrap_err()
            .contains("correction bias")
    );
    assert!(
        finish::<()>(Err("first".into()), Err("fence".into()))
            .unwrap_err()
            .contains("first; additionally fence")
    );
}

#[test]
fn moe_host_routing_refuses_flags_duplicates_nonfinite_and_bad_cutoff() {
    let valid = record(2);
    let r = decode_routing(f32::as_bytes(&valid), 2).unwrap();
    let s = Schedule::new(&r).unwrap();
    assert_eq!(s.weights.len(), 2 * TOP_K);
    assert_eq!(s.token_rows.len(), 2 * TOP_K);
    for t in 0..2 {
        let mut ids = Vec::new();
        for e in 0..EXPERTS {
            if s.by_expert[e].iter().any(|&(token, _)| token == t) {
                ids.push(e);
            }
        }
        assert_eq!(ids, r.ids[t * TOP_K..(t + 1) * TOP_K]);
    }
    for (at, value) in [
        (0, -1.0),
        (0, 512.0),
        (0, 0.25),
        (1, 0.0),
        (TOP_K, f32::NAN),
        (TOP_K, -0.1),
        (TOP_K, 5.1),
        (ROUTE_WIDTH - 1, 1.0),
        (2 * TOP_K + 1, 1.0),
    ] {
        let mut bad = valid.clone();
        bad[at] = value;
        assert!(
            decode_routing(f32::as_bytes(&bad), 2).is_err(),
            "{at}={value}"
        );
    }
    assert!(decode_routing(&[], 1).is_err());
}

// Deterministic source data, not a copy of a CPU runtime implementation.
fn bf16_matrix(n: usize, k: usize, step: usize, value: f32) -> Vec<u8> {
    let mut data = vec![0u8; n * k * 2];
    for row in 0..n {
        let i = (row * k + (row * step) % k) * 2;
        data[i..i + 2].copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    data
}
fn packed(n: usize, k: usize, global: f32) -> Vec<u8> {
    let mut data = vec![0u8; n * k / 2];
    for row in 0..n {
        for (col, code) in [((row * 7) % k, 2u8), ((row * 11 + 3) % k, 10u8)] {
            data[(row * k + col) / 2] |= code << ((col % 2) * 4);
        }
    }
    data.extend(vec![0x38u8; n * k / 16]); // E4M3 +1
    data.extend(global.to_le_bytes());
    data
}
fn operation(payload: &[u8], input_scale: f32) -> PackedOperation<'_> {
    PackedOperation {
        operation: *genid(),
        payload,
        input_scale,
    }
}
fn input(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    data: &[f32],
    t: usize,
    k: usize,
) -> CudaTensor {
    assert_eq!(data.len(), t * k);
    let raw: Vec<u8> = data
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    CudaTensor::new_contiguous(
        client.clone(),
        device.clone(),
        [1, t, k].into(),
        client.create_from_slice(&raw),
        DType::BF16,
    )
}
fn raw(t: &CudaTensor) -> Vec<u8> {
    let count = t.meta.shape().as_slice().iter().product::<usize>();
    let width = if t.dtype == DType::BF16 {
        2
    } else {
        assert_eq!(t.dtype, DType::F32);
        4
    };
    let bytes = t.client.read_one(t.handle.clone()).unwrap();
    bytes[..count * width].to_vec()
}
fn numbers(t: &CudaTensor) -> Vec<f32> {
    raw(t)
        .chunks_exact(if t.dtype == DType::BF16 { 2 } else { 4 })
        .map(|b| {
            if b.len() == 2 {
                bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
            } else {
                f32::from_le_bytes(b.try_into().unwrap())
            }
        })
        .collect()
}
struct Saved {
    root: PathBuf,
    bytes: usize,
    entries: Vec<serde_json::Value>,
}
impl Saved {
    fn new(name: &str) -> Self {
        let base = PathBuf::from(
            std::env::var_os("NEMOTRON_MOE_OUTPUT").expect("new fixture output parent required"),
        );
        fs::create_dir_all(&base).unwrap();
        let root = base.join(name);
        fs::create_dir(&root).expect("refuse existing fixture directory");
        Self {
            root,
            bytes: 0,
            entries: Vec::new(),
        }
    }
    fn write(&mut self, name: &str, data: &[u8], dtype: &str, shape: &[usize]) {
        self.bytes = self.bytes.checked_add(data.len()).unwrap();
        assert!(self.bytes <= 96 << 20 && self.entries.len() < 512);
        assert!(!name.contains('/') && !name.contains(".."));
        fs::write(self.root.join(name), data).unwrap();
        self.entries
            .push(serde_json::json!({"path":name,"dtype":dtype,"shape":shape,
            "bytes":data.len(),"blake3":blake3::hash(data).to_hex().to_string()}));
    }
    fn tensor(&mut self, name: &str, t: &CudaTensor) {
        self.write(
            name,
            &raw(t),
            if t.dtype == DType::BF16 {
                "BF16_LE"
            } else {
                "F32_LE"
            },
            t.meta.shape().as_slice(),
        );
    }
    fn finish(&self, report: serde_json::Value) {
        fs::write(
            self.root.join("tensors.json"),
            serde_json::to_vec_pretty(&self.entries).unwrap(),
        )
        .unwrap();
        fs::write(
            self.root.join("report.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
}

#[test]
#[ignore = "reserved CUDA: bounded routing diagnostics, not model acceptance"]
fn moe_gpu_route_nonfinite_ties_bias_and_underflow() {
    let mut saved = Saved::new("routing");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let mut cases = Vec::new();
    for (name, logit, bad_input, bad_bias) in [
        ("ties", 0.0, false, false),
        ("biased", 0.0, false, false),
        ("underflow", -1000.0, false, false),
        ("bad_logit", f32::INFINITY, false, false),
        ("bad_input", 0.0, true, false),
        ("bad_bias", 0.0, false, true),
    ] {
        let mut x = vec![0.0; HIDDEN];
        if bad_input {
            x[HIDDEN - 1] = f32::NAN;
        }
        let x = input(&client, &device, &x, 1, HIDDEN);
        let logits = vec![logit; EXPERTS];
        let mut bias = vec![0.0; EXPERTS];
        // Choice-only bias selects the highest IDs, while exact ties among
        // those IDs must follow the native lower-ID tie rule.
        for v in &mut bias[EXPERTS - TOP_K..] {
            *v = 1.0;
        }
        if name == "biased" {
            for (j, v) in bias[EXPERTS - TOP_K..].iter_mut().enumerate() {
                *v = (j + 1) as f32 * 0.1;
            }
        }
        if bad_bias {
            bias[0] = f32::NAN;
        }
        let lh = client.create_from_slice(f32::as_bytes(&logits));
        let bh = client.create_from_slice(f32::as_bytes(&bias));
        let out = client.empty(ROUTE_WIDTH * 4);
        unsafe {
            route_kernel::launch_unchecked::<CudaRuntime>(
                &client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(EXPERTS as u32),
                ArrayArg::from_raw_parts(x.handle.clone(), HIDDEN),
                ArrayArg::from_raw_parts(lh, EXPERTS),
                ArrayArg::from_raw_parts(bh, EXPERTS),
                ArrayArg::from_raw_parts(out.clone(), ROUTE_WIDTH),
            );
        }
        let data = client.read_one(out).unwrap();
        saved.tensor(&format!("{name}-input.bf16"), &x);
        saved.write(
            &format!("{name}-logits.f32"),
            f32::as_bytes(&logits),
            "F32_LE",
            &[1, EXPERTS],
        );
        saved.write(
            &format!("{name}-bias.f32"),
            f32::as_bytes(&bias),
            "F32_LE",
            &[EXPERTS],
        );
        saved.write(
            &format!("{name}-route.f32"),
            &data[..ROUTE_WIDTH * 4],
            "F32_LE",
            &[1, ROUTE_WIDTH],
        );
        let result = decode_routing(&data, 1);
        if name.starts_with("bad") {
            assert!(result.is_err());
        } else {
            let r = result.unwrap();
            let mut expected_ids = (EXPERTS - TOP_K..EXPERTS).collect::<Vec<_>>();
            if name == "biased" {
                expected_ids.reverse();
            }
            assert_eq!(r.ids, expected_ids);
            if name == "underflow" {
                assert!(r.weights.iter().all(|&v| v == 0.0));
            } else {
                assert!(
                    r.weights
                        .iter()
                        .all(|&v| (v - 5.0 / TOP_K as f32).abs() < 1e-6)
                );
            }
        }
        cases.push(serde_json::json!({"case":name,"status":"expected contract observed"}));
    }
    fence(&client).unwrap();
    saved.finish(serde_json::json!({"cases":cases,"independent_model_parity":false}));
}

#[test]
#[ignore = "reserved CUDA: calibrated W4A4 raw fixture, independent stock comparison required"]
fn moe_gpu_scaled_w4a4_and_unit_scale_fixture() {
    use crate::nn::{nvfp4_gemm::PromptLane, nvfp4_gemv::quantize_kernel};
    let mut saved = Saved::new("w4a4");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let (t, n, k) = (3, 8, 64);
    let w = packed(n, k, 0.25);
    saved.write("weight.nvfp4", &w, "NVFP4", &[n, k]);
    let weight = Nvfp4::upload(&w, [n as u64, k as u64], DType::BF16, &device).unwrap();
    let x: Vec<f32> = (0..t * k)
        .map(|i| ((i * 13 % 47) as f32 - 23.0) / 32.0)
        .collect();
    let x = input(&client, &device, &x, t, k);
    saved.tensor("input.bf16", &x);
    let old = weight.project(&x, PromptLane::W4a4).unwrap();
    saved.tensor("old-unit-output.bf16", &old);
    let mut diagnostics = Vec::new();
    for (label, s) in [("unit", 1.0), ("calibrated", 0.0037)] {
        let out = weight.project_w4a4_scaled(&x, s).unwrap();
        saved.tensor(&format!("{label}-output.bf16"), &out);
        assert!(numbers(&out).iter().all(|v| v.is_finite()));
        // Narrow API regression, not bit-equality as a model-quality criterion.
        if label == "unit" {
            assert_eq!(raw(&old), raw(&out));
        }
        let code = client.empty(t * k / 2);
        let scale = client.empty(t * k / 16);
        // Replay the SAME production quantizer on the retained immutable input
        // for an independently consumable plane fixture; not a CPU transcript.
        unsafe {
            quantize_kernel::launch_unchecked::<bf16, CudaRuntime>(
                &client,
                CubeCount::new_1d((t * k / 16).div_ceil(256) as u32),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(x.handle.clone(), t * k),
                ArrayArg::from_raw_parts(code.clone(), t * k / 8),
                ArrayArg::from_raw_parts(scale.clone(), t * k / 16),
                t * k / 16,
                s,
            );
        }
        let c = client.read_one(code).unwrap();
        let sc = client.read_one(scale).unwrap();
        saved.write(
            &format!("{label}-codes.u8"),
            &c[..t * k / 2],
            "E2M1_PACKED",
            &[t, k / 2],
        );
        saved.write(
            &format!("{label}-scales.e4m3"),
            &sc[..t * k / 16],
            "E4M3FN",
            &[t, k / 16],
        );
        let old_values = numbers(&old);
        let new_values = numbers(&out);
        let max = old_values
            .iter()
            .zip(new_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        diagnostics.push(
            serde_json::json!({"arm":label,"input_scale":s,"weight_global":0.25,
            "max_abs_vs_old_unit":max,"byte_equal_old_unit":raw(&old)==raw(&out)}),
        );
    }
    fence(&client).unwrap();
    saved.finish(serde_json::json!({"diagnostics":diagnostics,
        "quantized_planes":"replayed exact production kernel", "independent_stock_comparison":"pending"}));
}

struct TestProvider {
    binding: SourceBinding,
    up: Vec<u8>,
    // One shared payload, or one per expert.
    down: Vec<Vec<u8>>,
    operations: Vec<[Id; 2]>,
    calls: Cell<usize>,
    fail: Cell<Option<usize>>,
}
impl ExpertProvider for TestProvider {
    fn source(&self) -> SourceBinding {
        self.binding
    }
    fn with_expert<T>(&self, index: usize, use_pair: impl FnOnce(ExpertPair<'_>) -> Result<T>) -> Result<T> {
        self.calls.set(self.calls.get() + 1);
        if self.fail.get() == Some(index) {
            return Err("injected expert lookup failure".into());
        }
        use_pair(ExpertPair {
            index,
            up: PackedOperation {
                operation: self.operations[index][0],
                payload: &self.up,
                input_scale: expert_scale(index),
            },
            down: PackedOperation {
                operation: self.operations[index][1],
                payload: &self.down[index % self.down.len()],
                input_scale: 1.0 / 4096.0,
            },
        })
    }
}
fn expert_scale(index: usize) -> f32 {
    [1.0 / 128.0, 1.0 / 256.0, 1.0 / 512.0][index % 3]
}

#[test]
#[ignore = "reserved CUDA: full geometry synthetic stages/control, no model acceptance"]
fn moe_gpu_full_geometry_stages_for_stock_reference() {
    let mut saved = Saved::new("composition");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let binding = source();
    let router = bf16_matrix(EXPERTS, HIDDEN, 3, 0.125);
    let fc1 = bf16_matrix(LATENT, HIDDEN, 1, 0.5);
    let fc2 = bf16_matrix(HIDDEN, LATENT, 1, 0.5);
    let bias: Vec<f32> = (0..EXPERTS)
        .map(|e| {
            if e < TOP_K {
                (TOP_K - e) as f32 * 0.25
            } else {
                -1.0
            }
        })
        .collect();
    let shared_up = packed(SHARED, HIDDEN, 0.25);
    let shared_down = packed(HIDDEN, SHARED, 0.125);
    let up = packed(INTERMEDIATE, LATENT, 0.25);
    let down = packed(LATENT, INTERMEDIATE, 0.125);
    for (name, data, dtype, shape) in [
        (
            "router.bf16",
            router.as_slice(),
            "BF16_LE",
            [EXPERTS, HIDDEN],
        ),
        ("fc1.bf16", fc1.as_slice(), "BF16_LE", [LATENT, HIDDEN]),
        ("fc2.bf16", fc2.as_slice(), "BF16_LE", [HIDDEN, LATENT]),
        (
            "shared-up.nvfp4",
            shared_up.as_slice(),
            "NVFP4",
            [SHARED, HIDDEN],
        ),
        (
            "shared-down.nvfp4",
            shared_down.as_slice(),
            "NVFP4",
            [HIDDEN, SHARED],
        ),
        (
            "common-expert-up.nvfp4",
            up.as_slice(),
            "NVFP4",
            [INTERMEDIATE, LATENT],
        ),
        (
            "common-expert-down.nvfp4",
            down.as_slice(),
            "NVFP4",
            [LATENT, INTERMEDIATE],
        ),
    ] {
        saved.write(name, data, dtype, &shape);
    }
    saved.write(
        "correction-bias.f32",
        f32::as_bytes(&bias),
        "F32_LE",
        &[EXPERTS],
    );
    let provider = TestProvider {
        binding,
        up,
        down: vec![down],
        operations: (0..EXPERTS).map(|_| [*genid(), *genid()]).collect(),
        calls: Cell::new(0),
        fail: Cell::new(None),
    };
    let fixed = || FixedOperands {
        operations: [*genid(), *genid(), *genid(), *genid()],
        router_bf16: &router,
        correction_bias: &bias,
        fc1_bf16: &fc1,
        fc2_bf16: &fc2,
        shared_up: PackedOperation {
            operation: *genid(),
            payload: &shared_up,
            input_scale: 1.0 / 256.0,
        },
        shared_down: PackedOperation {
            operation: *genid(),
            payload: &shared_down,
            input_scale: 1.0 / 4096.0,
        },
    };
    // A source failure on any expert refuses the whole bank at upload.
    provider.fail.set(Some(1));
    let refused = Moe::upload(device.clone(), binding, fixed(), &provider, limits()).err().unwrap();
    assert!(refused.contains("injected"), "{refused}");
    assert_eq!(provider.calls.get(), 2);
    provider.fail.set(None);
    provider.calls.set(0);
    let mut moe = Moe::upload(
        device.clone(),
        binding,
        fixed(),
        &provider,
        limits(),
    )
    .unwrap();
    // Every expert was read exactly once, at upload, and never again below.
    assert_eq!(provider.calls.get(), EXPERTS);
    assert_eq!(moe.weights.experts.len(), EXPERTS);
    let x: Vec<f32> = (0..3 * HIDDEN)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) / 64.0)
        .collect();
    let full = input(&client, &device, &x, 3, HIDDEN);
    let original = raw(&full);
    saved.tensor("input.bf16", &full);
    let all = moe
        .forward_observed(&full, &mut |name, t| {
            saved.tensor(&format!("full-{name}.bin"), t);
            Ok(())
        })
        .unwrap();
    assert_eq!(raw(&full), original);
    assert!(!moe.is_poisoned());
    assert!(all.routing.ids.iter().all(|&e| e < TOP_K));
    let expected = numbers(&all.values);
    assert!(expected.iter().all(|v| v.is_finite()));
    let mut partition = Vec::new();
    for row in 0..3 {
        let one = input(
            &client,
            &device,
            &x[row * HIDDEN..(row + 1) * HIDDEN],
            1,
            HIDDEN,
        );
        let out = moe.forward(&one).unwrap();
        saved.tensor(&format!("row{row}-output.bf16"), &out.values);
        partition.extend(numbers(&out.values));
    }
    let max = expected
        .iter()
        .zip(&partition)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Geometry/budget failure before dispatch leaves owner usable.
    let saved_limit = moe.limits.max_working_bytes;
    moe.limits.max_working_bytes = 0;
    assert!(moe.forward(&full).is_err());
    assert!(!moe.is_poisoned());
    moe.limits.max_working_bytes = saved_limit;
    let mut bad = x[..HIDDEN].to_vec();
    bad[0] = f32::NAN;
    assert!(
        moe.forward(&input(&client, &device, &bad, 1, HIDDEN))
            .is_err()
    );
    assert!(moe.is_poisoned());
    assert!(moe.forward(&full).is_err());
    moe.reset().unwrap();
    // The span owner retains dispatch/readback payloads even when a helper
    // returns an error before its usual diagnostic observer runs.
    let mut held = Vec::new();
    moe.prefill_begin().unwrap();
    assert!(moe.prefill_route(&input(&client, &device, &bad, 1, HIDDEN), &mut held).is_err());
    assert!(held.iter().any(|t| t.meta.shape().as_slice() == [1, 1, EXPERTS]));
    assert!(held.iter().any(|t| t.meta.shape().as_slice() == [1, 1, ROUTE_WIDTH]));
    assert!(moe.is_poisoned());
    moe.reset().unwrap(); // Successful completion precedes owner release.
    held.clear();
    moe.prefill_begin().unwrap();
    let nan = moe.tensor(client.create_from_slice(f32::as_bytes(&vec![f32::NAN; 3 * LATENT])),
        3, LATENT, DType::F32);
    assert!(moe.prefill_finish(&full, &nan, &mut held).is_err());
    assert!(held.iter().filter(|t| t.meta.shape().as_slice() == [1, 3, SHARED]).count() >= 2);
    assert!(held.iter().any(|t| t.meta.shape().as_slice() == [1, 3, HIDDEN] && t.dtype == DType::BF16));
    assert!(held.iter().any(|t| t.meta.shape().as_slice() == [1, 3, 1] && t.dtype == DType::U32));
    moe.reset().unwrap();
    held.clear();
    // The provider is not consulted after upload: a source that now fails
    // changes nothing about forward.
    provider.fail.set(Some(1));
    let again = moe.forward(&full).unwrap();
    assert_eq!(provider.calls.get(), EXPERTS);
    saved.tensor("reset-output.bf16", &again.values);
    assert!(numbers(&again.values).iter().all(|v| v.is_finite()));
    fence(&client).unwrap();
    saved.finish(serde_json::json!({"shape":[1,3,HIDDEN],"top_k":TOP_K,"experts":EXPERTS,
        "source_binding":"synthetic immutable fixture, not checkpoint provenance",
        "selected_ids":all.routing.ids,"selected_weights":all.routing.weights,
        "cutoff_margins":all.routing.cutoff_margins,"max_abs_partition":max,
        "fixed_bytes":moe.admission.fixed_bytes,"bank_bytes":moe.admission.bank_bytes,
        "working_bytes":all.working_bytes,"expert_input_scales":(0..TOP_K).map(expert_scale).collect::<Vec<_>>(),
        "expert_down_input_scale":1.0/4096.0,"shared_up_input_scale":1.0/256.0,
        "shared_down_input_scale":1.0/4096.0,"independent_stock_comparison":"pending",
        "control":"preflight, nonfinite input, upload source failure, poison/refusal/reset observed",
        "numeric_gate":"none: finite output and router contract only; partition error is diagnostic"}));
}

/// Owned synthetic fixed operands for the bank tests.
struct FixedData {
    router: Vec<u8>,
    bias: Vec<f32>,
    fc1: Vec<u8>,
    fc2: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
}
impl FixedData {
    fn new() -> Self {
        Self {
            router: bf16_matrix(EXPERTS, HIDDEN, 3, 0.125),
            bias: (0..EXPERTS).map(|e| if e < TOP_K { (TOP_K - e) as f32 * 0.25 } else { -1.0 }).collect(),
            fc1: bf16_matrix(LATENT, HIDDEN, 1, 0.5),
            fc2: bf16_matrix(HIDDEN, LATENT, 1, 0.5),
            shared_up: packed(SHARED, HIDDEN, 0.25),
            shared_down: packed(HIDDEN, SHARED, 0.125),
        }
    }
    fn operands(&self) -> FixedOperands<'_> {
        FixedOperands {
            operations: [*genid(), *genid(), *genid(), *genid()],
            router_bf16: &self.router,
            correction_bias: &self.bias,
            fc1_bf16: &self.fc1,
            fc2_bf16: &self.fc2,
            shared_up: operation(&self.shared_up, 1.0 / 256.0),
            shared_down: operation(&self.shared_down, 1.0 / 4096.0),
        }
    }
}
fn shared_provider(binding: SourceBinding, down: Vec<Vec<u8>>) -> TestProvider {
    TestProvider {
        binding,
        up: packed(INTERMEDIATE, LATENT, 0.25),
        down,
        operations: (0..EXPERTS).map(|_| [*genid(), *genid()]).collect(),
        calls: Cell::new(0),
        fail: Cell::new(None),
    }
}

#[test]
#[ignore = "reserved CUDA: all 512 experts resident after upload and each run from the bank"]
fn moe_gpu_bank_every_expert_resident_without_source_reads() {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let binding = source();
    let fixed = FixedData::new();
    // Each expert's down projection has its own weight global, 1% apart: more
    // than one BF16 step, so every expert's output is distinct if and only if
    // each index runs its own resident pair.
    let down = packed(LATENT, INTERMEDIATE, 1.0);
    let globals: Vec<f32> = (0..EXPERTS).map(|e| 0.0625 * 1.01f32.powi(e as i32)).collect();
    let downs = globals.iter().map(|g| {
        let mut d = down.clone();
        let at = d.len() - 4;
        d[at..].copy_from_slice(&g.to_le_bytes());
        d
    }).collect();
    let provider = shared_provider(binding, downs);
    let moe = Moe::upload(device.clone(), binding, fixed.operands(), &provider, limits()).unwrap();
    assert_eq!(provider.calls.get(), EXPERTS);
    assert_eq!(moe.weights.experts.len(), EXPERTS);
    // The source now refuses every read; nothing below may consult it.
    provider.fail.set(Some(usize::MAX));
    let latent: Vec<f32> = (0..LATENT).map(|i| ((i * 13 % 47) as f32 - 23.0) / 32.0).collect();
    let latent = input(&client, &device, &latent, 1, LATENT);
    let mut outputs = std::collections::HashSet::new();
    for expert in 0..EXPERTS {
        let accumulator = moe.tensor(
            client.create_from_slice(f32::as_bytes(&vec![0.0f32; LATENT])), 1, LATENT, DType::F32);
        let mut held = Vec::new();
        moe.prefill_expert_tile(expert, &latent, &accumulator, &[(0, 1.0)], &mut held).unwrap();
        fence(&client).unwrap();
        let values = numbers(&accumulator);
        assert!(values.iter().all(|v| v.is_finite()), "expert {expert}");
        assert!(values.iter().any(|&v| v != 0.0), "expert {expert}");
        assert!(outputs.insert(raw(&accumulator)), "expert {expert} repeats another's output");
    }
    assert!(moe.prefill_expert_tile(EXPERTS, &latent, &latent, &[(0, 1.0)], &mut Vec::new()).is_err());
    assert_eq!(outputs.len(), EXPERTS);
    assert_eq!(provider.calls.get(), EXPERTS);
    println!("BANK_RESIDENT {}", serde_json::json!({
        "experts": EXPERTS, "source_reads": provider.calls.get(),
        "distinct_outputs": outputs.len(), "bank_bytes": moe.admission.bank_bytes,
    }));
}

fn rss_anon() -> usize {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("RssAnon:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse::<usize>().unwrap() * 1024
}

#[test]
#[ignore = "reserved CUDA: host copies of a layer's uploads are freed when the layer completes"]
fn moe_gpu_bank_upload_releases_host_staging() {
    // glibc keeps freed ~1.4 MB plane copies in its heap rather than returning
    // them, so a release shows as REUSE: the next upload does not grow
    // RssAnon. Buffers past its mmap threshold (the trailing case) go back.
    const LAYERS: usize = 4;
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let binding = source();
    let fixed = FixedData::new();
    let provider = shared_provider(binding, vec![packed(LATENT, INTERMEDIATE, 0.125)]);
    // Warm the context and allocator before the first observation.
    fence(&client).unwrap();
    let bank = admission(limits()).unwrap().bank_bytes;
    let observe = |label: &str, at: &mut Vec<serde_json::Value>| {
        let rss = rss_anon();
        at.push(serde_json::json!({"at": label, "rss_anon": rss}));
        rss
    };

    // 1. Each layer through Moe::upload, which releases at its end, the last
    // included. Only the first layer may grow the heap.
    let mut layers = Vec::new();
    let mut moe = Vec::new();
    let mut growth = Vec::new();
    for layer in 0..LAYERS {
        let before = observe(&format!("layer{layer}_before"), &mut moe);
        layers.push(Moe::upload(device.clone(), binding, fixed.operands(), &provider, limits()).unwrap());
        let after = observe(&format!("layer{layer}_after_upload_and_release"), &mut moe);
        growth.push(after as i64 - before as i64);
    }

    // 2. Contrast: the same layer's uploads, fenced and NOT released, twice.
    // The heap freed above serves at most one; the second grows it again.
    let raw_layer = || -> Vec<_> {
        let held = (0..EXPERTS).map(|e| load_expert(&provider, e,
            |op, n, k| PackedWeight::upload(op, n, k, &device)).unwrap()).collect();
        fence(&client).unwrap();
        held
    };
    let mut contrast = Vec::new();
    let start = observe("start", &mut contrast);
    let first = raw_layer();
    observe("first_fenced", &mut contrast);
    let second = raw_layer();
    let fenced = observe("second_fenced", &mut contrast);
    release_uploads(&client).unwrap();
    observe("released", &mut contrast);
    drop((first, second));
    let third = raw_layer();
    let reused = observe("third_fenced", &mut contrast);
    release_uploads(&client).unwrap();
    observe("third_released", &mut contrast);

    // 3. A trailing non-MoE upload after the last layer, as the loader's final
    // release handles: fenced alone it stays, released it is returned.
    let trailing = vec![7u8; 256 << 20];
    let mut tail = Vec::new();
    let tail_start = observe("start", &mut tail);
    let handle = client.create_from_slice(&trailing);
    fence(&client).unwrap();
    let tail_fenced = observe("fenced", &mut tail);
    release_uploads(&client).unwrap();
    let tail_released = observe("released", &mut tail);
    // 4. What the loader does last: hand the heap glibc kept back to the OS.
    let mut trim = Vec::new();
    let untrimmed = observe("before_trim", &mut trim);
    trim_host_heap();
    let trimmed = observe("after_trim", &mut trim);
    println!("RSS_RELEASE {}", serde_json::json!({
        "scope": "RssAnon of this process from /proc/self/status, bytes",
        "layer_bank_bytes": bank, "moe_upload": moe, "moe_upload_growth": growth,
        "contrast_fence_only": contrast, "trailing_256MiB": tail, "malloc_trim": trim,
    }));
    assert!(growth[0] < (bank + bank / 10) as i64, "{growth:?}");
    for (layer, &g) in growth.iter().enumerate().skip(1) {
        assert!(g < (bank / 4) as i64, "layer {layer} grew {g}");
    }
    assert!(fenced >= start + bank * 9 / 10, "two fenced layers grew {start} -> {fenced}");
    assert!(reused < fenced + bank / 4, "released memory was not reused: {fenced} -> {reused}");
    assert!(tail_fenced >= tail_start + trailing.len() * 9 / 10);
    assert!(tail_released + trailing.len() / 2 <= tail_fenced);
    assert!(trimmed + bank <= untrimmed, "trim gave back only {untrimmed} -> {trimmed}");
    drop((handle, third, layers));
}

#[test]
#[ignore = "reserved CUDA: parallel route selection versus the one-thread scan it replaced"]
fn moe_gpu_parallel_route_matches_serial_scan_bytes() {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    // Deterministic pseudo-random tokens, plus quantized logits that force
    // many exact ties, plus one token with a nonfinite input.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let t = 7;
    let mut x = vec![0.0f32; t * HIDDEN];
    let mut logits = vec![0.0f32; t * EXPERTS];
    for row in 0..t {
        for v in &mut x[row * HIDDEN..(row + 1) * HIDDEN] {
            *v = next() - 0.5;
        }
        for v in &mut logits[row * EXPERTS..(row + 1) * EXPERTS] {
            *v = match row {
                1 | 2 => ((next() * 8.0).floor() - 4.0) * 0.5, // heavy ties
                3 => 0.0,                                     // all tied
                _ => (next() - 0.5) * 12.0,
            };
        }
    }
    x[5 * HIDDEN + 17] = f32::NAN;
    let bias: Vec<f32> = (0..EXPERTS).map(|e| if e % 3 == 0 { 0.0 } else { (next() - 0.5) * 0.25 }).collect();
    let xh = input(&client, &device, &x, t, HIDDEN);
    let lh = client.create_from_slice(f32::as_bytes(&logits));
    let bh = client.create_from_slice(f32::as_bytes(&bias));
    // A distinct sentinel per run: a launch that never ran cannot pass by
    // reading the other run's bytes back from a reused pool slice.
    let run = |serial: bool| {
        let fill = vec![if serial { -7.0f32 } else { -9.0f32 }; t * ROUTE_WIDTH];
        let out = client.create_from_slice(f32::as_bytes(&fill));
        unsafe {
            let args = (
                ArrayArg::from_raw_parts(xh.handle.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(lh.clone(), t * EXPERTS),
                ArrayArg::from_raw_parts(bh.clone(), EXPERTS),
                ArrayArg::from_raw_parts(out.clone(), t * ROUTE_WIDTH),
            );
            if serial {
                route_kernel_serial::launch_unchecked::<CudaRuntime>(&client, CubeCount::new_1d(t as u32),
                    CubeDim::new_1d(EXPERTS as u32), args.0, args.1, args.2, args.3);
            } else {
                route_kernel::launch_unchecked::<CudaRuntime>(&client, CubeCount::new_1d(t as u32),
                    CubeDim::new_1d(EXPERTS as u32), args.0, args.1, args.2, args.3);
            }
        }
        client.read_one(out).unwrap()[..t * ROUTE_WIDTH * 4].to_vec()
    };
    let (serial, parallel) = (run(true), run(false));
    assert_eq!(serial, parallel, "parallel route record differs from the serial scan");
    let flag = |row: usize| f32::from_le_bytes(serial[(row * ROUTE_WIDTH + ROUTE_WIDTH - 1) * 4..][..4].try_into().unwrap());
    assert_eq!(flag(5), 1.0, "token 5 carries a nonfinite input");
    assert!((0..t).filter(|&r| r != 5).all(|r| flag(r) == 0.0));
}

#[test]
#[ignore = "reserved CUDA: grouped decode experts versus the per-expert tiles, byte for byte"]
fn moe_gpu_grouped_decode_matches_expert_tiles_bytes() {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let binding = source();
    let mut fixed = FixedData::new();
    // No forcing bias: the selection follows the input, so rows pick
    // different, unordered expert sets (checked below).
    fixed.bias = (0..EXPERTS).map(|e| ((e * 37 % 101) as f32 - 50.0) * 1e-4).collect();
    // Distinct down globals per expert: a wrong slot order or a wrong bank
    // view changes the weighted sum, so byte equality checks both.
    let down = packed(LATENT, INTERMEDIATE, 1.0);
    let downs = (0..EXPERTS).map(|e| {
        let mut d = down.clone();
        let at = d.len() - 4;
        d[at..].copy_from_slice(&(0.0625 * 1.01f32.powi(e as i32)).to_le_bytes());
        d
    }).collect();
    let provider = shared_provider(binding, downs);
    let mut moe = Moe::upload(device.clone(), binding, fixed.operands(), &provider, limits()).unwrap();
    let mut selections = std::collections::HashSet::new();
    let mut previous: Option<Vec<u8>> = None;
    for row in 0..6 {
        let x: Vec<f32> = (0..HIDDEN)
            .map(|i| (((i * (17 + 6 * row) + 31 * row * row + i * i % (7 + row)) % 101) as f32 - 50.0) / 64.0)
            .collect();
        let one = input(&client, &device, &x, 1, HIDDEN);
        // Grouped FIRST, on memory last written by a different row's tiles:
        // a grouped launch that never ran cannot pass by reading the
        // per-expert path's slab back from a reused pool slice.
        moe.grouped_decode = true;
        let grouped = moe.forward(&one).unwrap();
        moe.grouped_decode = false;
        let tiles = moe.forward(&one).unwrap();
        if let Some(previous) = &previous {
            assert_ne!(previous, &raw(&grouped.values), "row {row} repeats the previous row");
        }
        previous = Some(raw(&tiles.values));
        assert_eq!(tiles.routing.ids, grouped.routing.ids, "row {row}");
        assert_eq!(raw(&tiles.values), raw(&grouped.values), "row {row}: grouped decode differs");
        assert!(numbers(&grouped.values).iter().all(|v| v.is_finite()));
        selections.insert(grouped.routing.ids.clone());
    }
    println!("GROUPED_DECODE {}", serde_json::json!({"rows": 6, "distinct_selections": selections.len(),
        "selections": selections.iter().collect::<Vec<_>>()}));
    assert!(selections.len() > 1, "fixture never varied the selection");
}

#[test]
#[ignore = "reserved CUDA: one-row BF16 linear versus row 0 of the padded MMA linear, byte for byte"]
fn moe_gpu_bf16_linear_row_matches_padded_bytes() {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    for (n, k, seed) in [(EXPERTS, HIDDEN, 3usize), (LATENT, HIDDEN, 5), (HIDDEN, LATENT, 7)] {
        // Dense weights: every product contributes, so summation order shows.
        let weight: Vec<u8> = (0..n * k).flat_map(|i| {
            let v = (((i * 2654435761usize + seed) >> 7) % 2001) as f32 / 1000.0 - 1.0;
            bf16::from_f32(v * 0.05).to_bits().to_le_bytes()
        }).collect();
        let w = client.create_from_slice(&weight);
        let x: Vec<f32> = (0..k).map(|i| (((i * 29 + seed * 13) % 97) as f32 - 48.0) / 31.0).collect();
        let x = input(&client, &device, &x, 1, k);
        // One-row kernel first: it cannot pass on the padded path's freed buffer.
        let row = client.read_one(bf16gemm::bf16_linear_row_launch(&client, &x.handle, &w, k, n)).unwrap();
        let padded = bf16gemm::pad_bf16_launch(&client, &x.handle, k, 16 * k);
        let full = client.read_one(bf16gemm::bf16_linear_launch(&client, &padded, &w, 16, k, n)).unwrap();
        assert_eq!(&row[..n * 4], &full[..n * 4], "[{n}, {k}] one-row output differs from row 0");
        assert!(row[..n * 4].iter().any(|&b| b != 0));
    }
}
