//! Reserved synthetic-only CUDA witnesses. No checkpoint/pile or projections.
use super::{
    contract::{CacheDtype, Config, Parameters as CpuParameters},
    gpu, reference,
};
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    Runtime,
    cuda::{CudaDevice, CudaRuntime},
};
use half::bf16;

type Tensor = CubeTensor<CudaRuntime>;
fn tensor(values: &[f64], shape: &[usize], dtype: DType) -> Tensor {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let bytes: Vec<u8> = if dtype == DType::BF16 {
        values
            .iter()
            .flat_map(|x| bf16::from_f64(*x).to_bits().to_le_bytes())
            .collect()
    } else {
        values
            .iter()
            .flat_map(|x| (*x as f32).to_le_bytes())
            .collect()
    };
    let handle = client.create_from_slice(&bytes);
    CubeTensor::new_contiguous(client, device, shape.into(), handle, dtype)
}
fn read(t: &Tensor) -> Vec<f64> {
    // read_one is the completion fence; callers do not inspect partial kernels.
    let bytes = t.client.read_one(t.handle.clone()).unwrap();
    if t.dtype == DType::BF16 {
        bytes
            .chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f64())
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64)
            .collect()
    }
}
fn rounded(values: &[f64], dtype: DType) -> Vec<f64> {
    values
        .iter()
        .map(|&v| {
            if dtype == DType::BF16 {
                bf16::from_f64(v).to_f64()
            } else {
                (v as f32) as f64
            }
        })
        .collect()
}
fn compare(name: &str, got: &[f64], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(got.len(), expected.len());
    let mut max_abs = 0.0f64;
    let mut worst = 0.0f64;
    for (&a, &b) in got.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite(), "{name}: nonfinite");
        let error = (a - b).abs();
        max_abs = max_abs.max(error);
        let allowance = atol + rtol * b.abs();
        worst = worst.max(if allowance == 0.0 {
            if error == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            error / allowance
        });
    }
    eprintln!(
        "{name}: n={} max_abs={max_abs:.8e} max_scaled={worst:.8e} atol={atol} rtol={rtol}",
        got.len()
    );
    assert!(worst <= 1.0, "{name}: outside declared synthetic allowance");
}
fn rows(
    input: &[f64],
    batch: usize,
    tokens: usize,
    width: usize,
    start: usize,
    count: usize,
) -> Vec<f64> {
    (0..batch)
        .flat_map(|b| {
            input[(b * tokens + start) * width..(b * tokens + start + count) * width]
                .iter()
                .copied()
        })
        .collect()
}

fn fixture(c: Config, batch: usize, tokens: usize) {
    let l = c.layout().unwrap();
    let e = c.extents(batch, tokens).unwrap();
    let x = rounded(
        &(0..e.input)
            .map(|i| ((i * 7 % 31) as f64 - 13.0) / 32.0)
            .collect::<Vec<_>>(),
        DType::BF16,
    );
    let w = rounded(
        &(0..l.conv_channels * c.conv_kernel)
            .map(|i| ((i * 3 % 11) as f64 - 4.0) / 16.0)
            .collect::<Vec<_>>(),
        DType::BF16,
    );
    let bias = rounded(
        &(0..l.conv_channels)
            .map(|i| (i % 7) as f64 / 32.0)
            .collect::<Vec<_>>(),
        DType::BF16,
    );
    let a = rounded(
        &(0..c.heads)
            .map(|i| (i % 3) as f64 / 8.0)
            .collect::<Vec<_>>(),
        DType::F32,
    );
    let dt = rounded(
        &(0..c.heads)
            .map(|i| {
                if i == 0 {
                    -12.0
                } else {
                    -0.25 + (i % 4) as f64 / 8.0
                }
            })
            .collect::<Vec<_>>(),
        DType::F32,
    );
    let d = rounded(
        &(0..c.heads)
            .map(|i| 0.25 + (i % 5) as f64 / 8.0)
            .collect::<Vec<_>>(),
        DType::F32,
    );
    let gain = rounded(
        &(0..l.inner)
            .map(|i| 0.5 + (i % 7) as f64 / 8.0)
            .collect::<Vec<_>>(),
        DType::BF16,
    );
    let cpu = CpuParameters {
        conv_weight: &w,
        conv_bias: Some(&bias),
        a_log: &a,
        dt_bias: &dt,
        d: &d,
        norm_gain: &gain,
    };
    let oracle =
        reference::chunk(c, &cpu, &x, batch, tokens, None, CacheDtype::F32, 64 << 20).unwrap();
    let input = tensor(&x, &[batch, tokens, l.projected], DType::BF16);
    let wg = tensor(&w, &[l.conv_channels, 1, c.conv_kernel], DType::BF16);
    let bg = tensor(&bias, &[l.conv_channels], DType::BF16);
    let ag = tensor(&a, &[c.heads], DType::F32);
    let dtg = tensor(&dt, &[c.heads], DType::F32);
    let dg = tensor(&d, &[c.heads], DType::F32);
    let gg = tensor(&gain, &[l.inner], DType::BF16);
    let p = gpu::Parameters {
        conv_weight: &wg,
        conv_bias: Some(&bg),
        a_log: &ag,
        dt_bias: &dtg,
        d: &dg,
        norm_gain: &gg,
    };
    let whole = gpu::chunk(c, &p, &input, None, 64 << 20).unwrap();
    assert_eq!(whole.state.attended(), tokens);
    assert_eq!(
        whole.values.meta.shape().as_slice(),
        [batch, tokens, l.inner]
    );
    assert_eq!(
        whole.state.recurrent().meta.shape().as_slice(),
        [batch, c.heads, c.head_dim, c.state_dim]
    );
    assert_eq!(
        whole.allocated_bytes,
        e.output * 6 + batch * tokens * l.conv_channels * 2 + e.conv * 2 + e.recurrent * 4
    );
    let actual = read(&whole.values);
    let state = read(whole.state.recurrent());
    let history = read(whole.state.conv());
    eprintln!(
        "geometry B={batch} T={tokens} H={} D={} G={} N={} K={} fresh_logical_bytes={}",
        c.heads, c.head_dim, c.groups, c.state_dim, c.conv_kernel, whole.allocated_bytes
    );
    // Fixed before running: BF16 conv/SiLU and output differ from wide algebra.
    // These are finite synthetic allowances, not a model quality/error promise.
    compare("wide-algebra output", &actual, &oracle.values, 0.015, 0.02);
    compare(
        "wide-algebra state",
        &state,
        &oracle.state.recurrent,
        0.0005,
        0.025,
    );
    compare("raw history", &history, &oracle.state.conv, 0.0, 0.0);

    for partition in [vec![1; tokens], vec![tokens - 1, 1]] {
        let mut previous = None;
        let mut joined = vec![0.0; e.output];
        let mut start = 0;
        for count in partition {
            let part = rows(&x, batch, tokens, l.projected, start, count);
            let part = tensor(&part, &[batch, count, l.projected], DType::BF16);
            let out = gpu::chunk(c, &p, &part, previous.as_ref(), 64 << 20).unwrap();
            let values = read(&out.values);
            for b in 0..batch {
                joined[(b * tokens + start) * l.inner..][..count * l.inner]
                    .copy_from_slice(&values[b * count * l.inner..][..count * l.inner]);
            }
            start += count;
            assert_eq!(out.state.attended(), start);
            previous = Some(out.state);
        }
        compare("chunk/decode output", &joined, &actual, 0.008, 0.005);
        compare(
            "chunk/decode state",
            &read(previous.as_ref().unwrap().recurrent()),
            &state,
            1e-6,
            1e-5,
        );
    }
    let reset = whole.state.reset(64 << 20).unwrap();
    assert_eq!(reset.attended(), 0);
    assert!(read(reset.conv()).iter().all(|x| *x == 0.0));
    assert!(read(reset.recurrent()).iter().all(|x| *x == 0.0));
    let again = gpu::chunk(c, &p, &input, Some(&reset), 64 << 20).unwrap();
    compare("reset output", &read(&again.values), &actual, 0.008, 0.005);
    // Mutation and refusal checks are discrete API contracts, not float gates.
    assert!(gpu::chunk(c, &p, &input, None, 0).is_err());
    assert!(whole.state.reset(0).is_err());
    assert!(
        gpu::chunk(
            Config {
                norm_epsilon: f64::NAN,
                ..c
            },
            &p,
            &input,
            None,
            64 << 20
        )
        .is_err()
    );
    assert!(
        gpu::chunk(
            Config {
                time_step_min: 0.001,
                ..c
            },
            &p,
            &input,
            None,
            64 << 20
        )
        .is_err()
    );
    assert!(
        gpu::chunk(
            Config {
                norm_epsilon: c.norm_epsilon * 2.0,
                ..c
            },
            &p,
            &input,
            Some(&whole.state),
            64 << 20
        )
        .is_err()
    );
    compare(
        "old state immutable",
        &read(whole.state.recurrent()),
        &state,
        0.0,
        0.0,
    );
    compare("input immutable", &read(&input), &x, 0.0, 0.0);
    compare("weights immutable", &read(&wg), &w, 0.0, 0.0);
}

#[test]
#[ignore = "reserved CUDA: synthetic projected Mamba2 only, no model weights"]
fn synthetic_gpu_mamba2() {
    fixture(
        Config {
            heads: 4,
            head_dim: 2,
            groups: 2,
            state_dim: 3,
            conv_kernel: 3,
            norm_epsilon: 0.125,
            time_step_min: 0.0,
        },
        2,
        5,
    );
    fixture(
        Config {
            heads: 128,
            head_dim: 64,
            groups: 8,
            state_dim: 128,
            conv_kernel: 4,
            norm_epsilon: 1e-5,
            time_step_min: 0.0,
        },
        1,
        3,
    );
}

// Independent stock-fixture adapter: reads retained raw GPU-reference bytes,
// never regenerates the reference, never calls the scalar recurrence oracle.
// Numerical bands only flag diagnostics; all arrays are exported before review.
struct StockFixture {
    root: std::path::PathBuf,
    manifest: serde_json::Value,
}
impl StockFixture {
    fn read(&self, path: &str, shape: &[usize], dtype: DType) -> Vec<u8> {
        let records = self.manifest["tensors"]
            .as_array()
            .expect("tensor manifest");
        let matches: Vec<_> = records
            .iter()
            .filter(|v| v["path"].as_str() == Some(path))
            .collect();
        assert_eq!(matches.len(), 1, "fixture manifest entry {path}");
        let record = matches[0];
        let declared: Vec<_> = record["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(declared, shape, "reference shape {path}");
        assert_eq!(
            record["dtype"],
            if dtype == DType::BF16 {
                "torch.bfloat16"
            } else {
                "torch.float32"
            }
        );
        let count = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .unwrap();
        let size = count
            .checked_mul(if dtype == DType::BF16 { 2 } else { 4 })
            .unwrap();
        assert!(size <= 8 << 20, "bounded individual fixture");
        assert_eq!(record["bytes"].as_u64().unwrap(), size as u64);
        let file = self.root.join(path);
        assert_eq!(std::fs::metadata(&file).unwrap().len(), size as u64);
        let bytes = std::fs::read(file).unwrap();
        assert_eq!(bytes.len(), size);
        bytes
    }
    fn values(&self, path: &str, shape: &[usize], dtype: DType) -> Vec<f64> {
        raw_values(&self.read(path, shape, dtype), dtype)
    }
    fn tensor(
        &self,
        path: &str,
        stock_shape: &[usize],
        native_shape: &[usize],
        dtype: DType,
    ) -> Tensor {
        assert_eq!(
            stock_shape.iter().product::<usize>(),
            native_shape.iter().product::<usize>()
        );
        let bytes = self.read(path, stock_shape, dtype);
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let handle = client.create_from_slice(&bytes);
        CubeTensor::new_contiguous(client, device, native_shape.into(), handle, dtype)
    }
}
fn raw_values(bytes: &[u8], dtype: DType) -> Vec<f64> {
    if dtype == DType::BF16 {
        bytes
            .chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f64())
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64)
            .collect()
    }
}
fn export_native(root: &std::path::Path, name: &str, t: &Tensor) -> (Vec<f64>, serde_json::Value) {
    use std::io::Write;
    let bytes = t.client.read_one(t.handle.clone()).unwrap();
    let suffix = if t.dtype == DType::BF16 {
        "bf16"
    } else {
        "f32"
    };
    let relative = format!("{name}.{suffix}");
    let path = root.join(&relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(&bytes).unwrap();
    let record = serde_json::json!({"path":relative,"shape":t.meta.shape().as_slice(),"dtype":suffix,
        "bytes":bytes.len(),"blake3":blake3::hash(&bytes).to_hex().to_string()});
    (raw_values(&bytes, t.dtype), record)
}
fn stock_difference(
    name: &str,
    actual: &[f64],
    expected: &[f64],
    atol: f64,
    rtol: f64,
) -> serde_json::Value {
    assert_eq!(actual.len(), expected.len());
    let (mut max_abs, mut max_rel, mut sum_sq) = (0.0f64, 0.0f64, 0.0f64);
    let (mut differing, mut outside, mut nonfinite) = (0usize, 0usize, 0usize);
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() {
            nonfinite += 1;
            continue;
        }
        let e = (a - b).abs();
        max_abs = max_abs.max(e);
        max_rel = max_rel.max(e / b.abs().max(1e-6));
        sum_sq += e * e;
        differing += usize::from(a != b);
        outside += usize::from(e > atol + rtol * b.abs());
    }
    serde_json::json!({"boundary":name,"values":actual.len(),"max_abs":max_abs,"max_rel_floor_1e_6":max_rel,
        "rmse":(sum_sq/actual.len() as f64).sqrt(),"differing_values":differing,
        "outside_diagnostic_band":outside,"nonfinite_pairs":nonfinite,"atol":atol,"rtol":rtol,
        "interpretation":"flags only, not numerical acceptance"})
}

#[test]
#[ignore = "reserved CUDA: consume retained stock GPU fixtures, explicit paths required"]
fn stock_gpu_reference_comparison() {
    let root = std::path::PathBuf::from(
        std::env::var_os("NEMOTRON_STOCK_FIXTURES").expect("reference path required"),
    );
    let out = std::path::PathBuf::from(
        std::env::var_os("NEMOTRON_NATIVE_COMPARISON").expect("new output path required"),
    );
    assert!(
        !root.join("failure.txt").exists(),
        "failed reference is not completed evidence"
    );
    assert!(std::fs::metadata(root.join("tensors.json")).unwrap().len() <= 1 << 20);
    let manifest_bytes = std::fs::read(root.join("tensors.json")).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(manifest["byte_order"], "little");
    assert!(manifest["payload_bytes"].as_u64().unwrap() <= 64 << 20);
    assert!(manifest["tensors"].as_array().unwrap().len() <= 256);
    assert!(std::fs::metadata(root.join("report.json")).unwrap().len() <= 1 << 20);
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["cases"].as_array().unwrap().len(), 2);
    assert_eq!(report["state_dtype"], "float32");
    assert_eq!(report["gate_before_norm"], true);
    assert_eq!(report["dt_limit"], serde_json::json!([0, "inf"]));
    // Artifact SHA256 verification is a separate runner preflight over the
    // retained manifest; this adapter checks every consumed size/shape/dtype.
    let stock = StockFixture { root, manifest };
    std::fs::create_dir(&out).expect("new output directory only");
    let mut artifacts = Vec::new();
    let mut metrics = Vec::new();
    let mut used = 0usize;
    for (name, batch, tokens, c) in [
        (
            "tiny",
            2,
            5,
            Config {
                heads: 4,
                head_dim: 2,
                groups: 2,
                state_dim: 3,
                conv_kernel: 3,
                norm_epsilon: 0.125,
                time_step_min: 0.0,
            },
        ),
        (
            "super",
            1,
            3,
            Config {
                heads: 128,
                head_dim: 64,
                groups: 8,
                state_dim: 128,
                conv_kernel: 4,
                norm_epsilon: 1e-5,
                time_step_min: 0.0,
            },
        ),
    ] {
        let l = c.layout().unwrap();
        let prefix = format!("{name}/input");
        let x = stock.values(
            &format!("{prefix}/projected.bf16"),
            &[batch, tokens, l.projected],
            DType::BF16,
        );
        let w = stock.tensor(
            &format!("{prefix}/conv_weight.bf16"),
            &[l.conv_channels, c.conv_kernel],
            &[l.conv_channels, 1, c.conv_kernel],
            DType::BF16,
        );
        let bias = stock.tensor(
            &format!("{prefix}/conv_bias.bf16"),
            &[l.conv_channels],
            &[l.conv_channels],
            DType::BF16,
        );
        let a = stock.tensor(
            &format!("{prefix}/a_log.f32"),
            &[c.heads],
            &[c.heads],
            DType::F32,
        );
        let dt = stock.tensor(
            &format!("{prefix}/dt_bias.f32"),
            &[c.heads],
            &[c.heads],
            DType::F32,
        );
        let d = stock.tensor(
            &format!("{prefix}/D.f32"),
            &[c.heads],
            &[c.heads],
            DType::F32,
        );
        let gain = stock.tensor(
            &format!("{prefix}/norm_gain.bf16"),
            &[l.inner],
            &[l.inner],
            DType::BF16,
        );
        let p = gpu::Parameters {
            conv_weight: &w,
            conv_bias: Some(&bias),
            a_log: &a,
            dt_bias: &dt,
            d: &d,
            norm_gain: &gain,
        };
        for (arm, plan) in [
            ("prefill", vec![("prefill", tokens)]),
            ("continuation", vec![("prefill", tokens - 1), ("step", 1)]),
            ("steps", vec![("step", 1); tokens]),
        ] {
            let mut state = None;
            let mut start = 0;
            for (index, (kind, count)) in plan.into_iter().enumerate() {
                let label = format!("{name}/{arm}/part{index}-{kind}");
                let input = tensor(
                    &rows(&x, batch, tokens, l.projected, start, count),
                    &[batch, count, l.projected],
                    DType::BF16,
                );
                let got = gpu::chunk(c, &p, &input, state.as_ref(), 64 << 20).unwrap();
                assert_eq!(got.state.attended(), start + count);
                for (boundary, t, shape, atol, rtol) in [
                    (
                        "output",
                        &got.values,
                        vec![batch, count, l.inner],
                        0.02,
                        0.02,
                    ),
                    (
                        "state",
                        got.state.recurrent(),
                        vec![batch, c.heads, c.head_dim, c.state_dim],
                        0.002,
                        0.02,
                    ),
                ] {
                    let (native, record) = export_native(&out, &format!("{label}/{boundary}"), t);
                    used += record["bytes"].as_u64().unwrap() as usize;
                    artifacts.push(record);
                    let suffix = if t.dtype == DType::BF16 {
                        "bf16"
                    } else {
                        "f32"
                    };
                    let expected =
                        stock.values(&format!("{label}/{boundary}.{suffix}"), &shape, t.dtype);
                    let mut metric = stock_difference(boundary, &native, &expected, atol, rtol);
                    metric["part"] = label.clone().into();
                    metrics.push(metric);
                }
                let (history, record) =
                    export_native(&out, &format!("{label}/conv_history_k"), got.state.conv());
                used += record["bytes"].as_u64().unwrap() as usize;
                artifacts.push(record);
                let tail: Vec<_> = history
                    .chunks_exact(c.conv_kernel)
                    .flat_map(|row| row[1..].iter().copied())
                    .collect();
                let expected = stock.values(
                    &format!("{label}/conv_tail.bf16"),
                    &[batch, l.conv_channels, c.conv_kernel - 1],
                    DType::BF16,
                );
                let mut metric =
                    stock_difference("native last K-1 vs stock tail", &tail, &expected, 0.0, 0.0);
                metric["part"] = label.clone().into();
                metrics.push(metric);
                assert!(used <= 64 << 20, "bounded native export");
                let retained = serde_json::json!({"reference_root":stock.root,"reference_manifest_blake3":blake3::hash(&manifest_bytes).to_hex().to_string(),
                    "native_baseline":"0de9c9a3ba2c571a02758302cdde6a2a13c29464","payload_bytes":used,"artifacts":artifacts,"metrics":metrics,
                    "limits":"No CPU oracle invoked. No numerical acceptance asserted. Native conv has intermediate BF16 cast; native scan stays F32 before gate/norm. Native K history compared as last K-1 suffix. Intermediate native conv/scan not exposed, so boundary error alone does not attribute all difference."});
                std::fs::write(
                    out.join("report.json"),
                    serde_json::to_vec_pretty(&retained).unwrap(),
                )
                .unwrap();
                assert!(
                    metrics
                        .iter()
                        .all(|m| m["nonfinite_pairs"].as_u64() == Some(0)),
                    "nonfinite output retained"
                );
                eprintln!("stock comparison retained {label}, cumulative_bytes={used}");
                start += count;
                state = Some(got.state);
            }
        }
    }
}
