//! Scoped host/control tests and ignored native synthetic GPU fixtures.
//! Finite/shape/refusal assertions are contracts. Native partition agreement
//! alone is NOT an independent numerical/model acceptance test.
use super::*;
use std::{cell::Cell, fs, path::PathBuf};
use triblespace::prelude::genid;

fn source() -> SourceBinding {
    SourceBinding {
        model_root: genid(),
        layer: genid(),
    }
}
fn limits() -> Limits {
    Limits {
        max_fixed_bytes: 48 << 20,
        max_expert_bytes: 4 << 20,
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
    assert_eq!(a.expert_bytes, 3_096_584);
    assert!(
        admission(Limits {
            max_fixed_bytes: a.fixed_bytes - 1,
            ..limits()
        })
        .is_err()
    );
    assert!(
        admission(Limits {
            max_expert_bytes: a.expert_bytes - 1,
            ..limits()
        })
        .is_err()
    );
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
        operations: [genid(), genid(), genid(), genid()],
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
        operation: genid(),
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
    down: Vec<u8>,
    operations: Vec<[Id; 2]>,
    calls: Cell<usize>,
    fail: Cell<Option<usize>>,
}
impl ExpertProvider for TestProvider {
    fn source(&self) -> SourceBinding {
        self.binding
    }
    fn expert(&self, index: usize) -> Result<ExpertPair<'_>> {
        self.calls.set(self.calls.get() + 1);
        if self.fail.get() == Some(index) {
            return Err("injected expert lookup failure".into());
        }
        Ok(ExpertPair {
            index,
            up: PackedOperation {
                operation: self.operations[index][0],
                payload: &self.up,
                input_scale: expert_scale(index),
            },
            down: PackedOperation {
                operation: self.operations[index][1],
                payload: &self.down,
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
        down,
        operations: (0..EXPERTS).map(|_| [genid(), genid()]).collect(),
        calls: Cell::new(0),
        fail: Cell::new(None),
    };
    let mut moe = Moe::upload(
        device.clone(),
        binding,
        FixedOperands {
            operations: [genid(), genid(), genid(), genid()],
            router_bf16: &router,
            correction_bias: &bias,
            fc1_bf16: &fc1,
            fc2_bf16: &fc2,
            shared_up: PackedOperation {
                operation: genid(),
                payload: &shared_up,
                input_scale: 1.0 / 256.0,
            },
            shared_down: PackedOperation {
                operation: genid(),
                payload: &shared_down,
                input_scale: 1.0 / 4096.0,
            },
        },
        provider,
        limits(),
    )
    .unwrap();
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
    let calls = moe.provider.calls.get();
    let mut bad = x[..HIDDEN].to_vec();
    bad[0] = f32::NAN;
    assert!(
        moe.forward(&input(&client, &device, &bad, 1, HIDDEN))
            .is_err()
    );
    assert!(moe.is_poisoned());
    assert_eq!(moe.provider.calls.get(), calls);
    assert!(moe.forward(&full).is_err());
    moe.reset().unwrap();
    moe.provider.fail.set(Some(1)); // Expert0 already ran; failure must drain and poison.
    let failure = moe.forward(&full).err().unwrap();
    assert!(failure.contains("injected"));
    assert!(moe.is_poisoned());
    moe.provider.fail.set(None);
    moe.reset().unwrap();
    let again = moe.forward(&full).unwrap();
    saved.tensor("reset-output.bf16", &again.values);
    assert!(numbers(&again.values).iter().all(|v| v.is_finite()));
    fence(&client).unwrap();
    saved.finish(serde_json::json!({"shape":[1,3,HIDDEN],"top_k":TOP_K,"experts":EXPERTS,
        "source_binding":"synthetic immutable fixture, not checkpoint provenance",
        "selected_ids":all.routing.ids,"selected_weights":all.routing.weights,
        "cutoff_margins":all.routing.cutoff_margins,"max_abs_partition":max,
        "fixed_bytes":moe.admission.fixed_bytes,"one_expert_bytes":moe.admission.expert_bytes,
        "working_bytes":all.working_bytes,"expert_input_scales":(0..TOP_K).map(expert_scale).collect::<Vec<_>>(),
        "expert_down_input_scale":1.0/4096.0,"shared_up_input_scale":1.0/256.0,
        "shared_down_input_scale":1.0/4096.0,"independent_stock_comparison":"pending",
        "control":"preflight, nonfinite input, partial expert failure, poison/refusal/reset observed",
        "numeric_gate":"none: finite output and router contract only; partition error is diagnostic"}));
}
