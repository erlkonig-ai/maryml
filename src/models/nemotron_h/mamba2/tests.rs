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
