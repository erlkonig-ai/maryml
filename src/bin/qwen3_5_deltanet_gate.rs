//! CUDA parity and same-device exactness gate for the resident DeltaNet slice.
//! Usage: qwen3_5_deltanet_gate <qwen3_5_deltanet_reference.py output.json>
//! Host transfers occur only at fixture ingress and result comparison.

use std::path::Path;

use burn::tensor::{DType, Tensor, TensorPrimitive};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
};
use half::bf16;
use mary::models::qwen3_5::deltanet::{DeltaNetInputs, DeltaNetOutput, gated_delta};
use serde::Deserialize;

type C = CubeTensor<CudaRuntime>;
type B = burn::backend::Cuda<bf16>;

const REFERENCE_SHA256: &str = "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8";

#[derive(Deserialize)]
struct Fixture {
    transformers: String,
    torch: String,
    device: String,
    reference_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    dims: [usize; 6],
    query: Vec<u16>,
    key: Vec<u16>,
    value: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    initial_state: Option<Vec<f32>>,
    recurrent_output: Vec<u16>,
    recurrent_state: Vec<f32>,
    chunk_output: Vec<u16>,
    chunk_state: Vec<f32>,
}

struct Resident {
    q: C,
    k: C,
    v: C,
    a: C,
    b: C,
    a_log: C,
    dt_bias: C,
    initial: Option<C>,
}

fn upload_bf16(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    values: &[u16],
    shape: &[usize],
) -> C {
    assert_eq!(values.len(), shape.iter().product::<usize>());
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        shape.into(),
        client.create_from_slice(&bytes),
        DType::BF16,
    )
}

fn upload_f32(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    values: &[f32],
    shape: &[usize],
) -> C {
    assert_eq!(values.len(), shape.iter().product::<usize>());
    CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        shape.into(),
        client.create_from_slice(f32::as_bytes(values)),
        DType::F32,
    )
}

impl Resident {
    fn new(case: &Case, client: &ComputeClient<CudaRuntime>, device: &CudaDevice) -> Self {
        let [batch, tokens, kh, vh, kd, vd] = case.dims;
        let bf = |x: &[u16], shape: &[usize]| upload_bf16(client, device, x, shape);
        Self {
            q: bf(&case.query, &[batch, tokens, kh, kd]),
            k: bf(&case.key, &[batch, tokens, kh, kd]),
            v: bf(&case.value, &[batch, tokens, vh, vd]),
            a: bf(&case.a, &[batch, tokens, vh]),
            b: bf(&case.b, &[batch, tokens, vh]),
            a_log: bf(&case.a_log, &[vh]),
            dt_bias: bf(&case.dt_bias, &[vh]),
            initial: case
                .initial_state
                .as_ref()
                .map(|s| upload_f32(client, device, s, &[batch, vh, kd, vd])),
        }
    }

    fn run(&self, state: Option<&C>) -> Result<DeltaNetOutput<CudaRuntime>, String> {
        gated_delta(DeltaNetInputs {
            query: &self.q,
            key: &self.k,
            value: &self.v,
            a: &self.a,
            b: &self.b,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            initial_state: state,
        })
    }

    /// Burn slicing and contiguity happen on the device; there is no host read.
    fn slice(&self, batch: std::ops::Range<usize>, tokens: std::ops::Range<usize>) -> Self {
        let slice4 = |tensor: &C| {
            let s = tensor.meta.shape();
            take_slice::<4>(tensor, [batch.clone(), tokens.clone(), 0..s[2], 0..s[3]])
        };
        let slice3 = |tensor: &C| {
            take_slice::<3>(
                tensor,
                [batch.clone(), tokens.clone(), 0..tensor.meta.shape()[2]],
            )
        };
        Self {
            q: slice4(&self.q),
            k: slice4(&self.k),
            v: slice4(&self.v),
            a: slice3(&self.a),
            b: slice3(&self.b),
            a_log: self.a_log.clone(),
            dt_bias: self.dt_bias.clone(),
            initial: self.initial.as_ref().map(|s| {
                take_slice::<4>(
                    s,
                    [
                        batch.clone(),
                        0..s.meta.shape()[1],
                        0..s.meta.shape()[2],
                        0..s.meta.shape()[3],
                    ],
                )
            }),
        }
    }
}

fn take_slice<const D: usize>(tensor: &C, ranges: [std::ops::Range<usize>; D]) -> C {
    let tensor =
        Tensor::<B, D>::from_primitive(TensorPrimitive::Float(tensor.clone())).slice(ranges);
    match tensor.into_primitive() {
        TensorPrimitive::Float(t) => burn_cubecl::kernel::into_contiguous(t),
        TensorPrimitive::QFloat(_) => unreachable!(),
    }
}

fn read(tensor: &C) -> Result<Vec<u8>, String> {
    tensor
        .client
        .read_one(tensor.handle.clone())
        .map(|b| b.to_vec())
        .map_err(|e| format!("{e:?}"))
}

fn read_f32(tensor: &C) -> Result<Vec<f32>, String> {
    let bytes = read(tensor)?;
    Ok(match tensor.dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        _ => unreachable!(),
    })
}

fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: length mismatch"));
    }
    let mut max_abs = 0.0f32;
    let mut max_scaled = 0.0f32;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(format!("{label}: nonfinite at {i}"));
        }
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        max_scaled = max_scaled.max(diff / (atol + rtol * b.abs()));
    }
    println!("{label}: max_abs={max_abs:.8e} max_scaled={max_scaled:.6} atol={atol} rtol={rtol}");
    if max_scaled > 1.0 {
        return Err(format!("{label}: CUDA reference tolerance exceeded"));
    }
    Ok(())
}

fn exact(label: &str, actual: &[u8], expected: &[u8]) -> Result<(), String> {
    if actual != expected {
        let first = actual.iter().zip(expected).position(|(a, b)| a != b);
        return Err(format!(
            "{label}: bit mismatch at byte {first:?}, lengths {} / {}",
            actual.len(),
            expected.len()
        ));
    }
    Ok(())
}

fn gate(
    case: &Case,
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
) -> Result<(), String> {
    let [batch, tokens, _, vh, kd, vd] = case.dims;
    let resident = Resident::new(case, client, device);
    let initial_before = resident.initial.as_ref().map(read).transpose()?;
    let out = resident.run(resident.initial.as_ref())?;
    let output_bytes = read(&out.output)?;
    let state_bytes = read(&out.state)?;
    let actual = read_f32(&out.output)?;
    let state = read_f32(&out.state)?;
    for (name, expected_out, expected_state) in [
        (
            "torch_recurrent",
            &case.recurrent_output,
            &case.recurrent_state,
        ),
        ("torch_chunk", &case.chunk_output, &case.chunk_state),
    ] {
        let expected: Vec<f32> = expected_out
            .iter()
            .map(|&v| bf16::from_bits(v).to_f32())
            .collect();
        compare(
            &format!("{} {name} output", case.name),
            &actual,
            &expected,
            0.001,
            0.02,
        )?;
        compare(
            &format!("{} {name} state", case.name),
            &state,
            expected_state,
            0.00002,
            0.001,
        )?;
    }
    for _ in 0..3 {
        let repeat = resident.run(resident.initial.as_ref())?;
        exact("repeat output", &read(&repeat.output)?, &output_bytes)?;
        exact("repeat state", &read(&repeat.state)?, &state_bytes)?;
    }
    // Same arithmetic run as individual tokens and as irregular chunks.
    for lengths in [vec![1; tokens], vec![2, 1, tokens - 3]] {
        let mut state = resident.initial.clone();
        let mut assembled = vec![0u8; output_bytes.len()];
        let mut start = 0;
        for len in lengths {
            let piece = resident.slice(0..batch, start..start + len);
            let part = piece.run(state.as_ref())?;
            let bytes = read(&part.output)?;
            let row_bytes = vh * vd * 2;
            for item in 0..batch {
                let dst = (item * tokens + start) * row_bytes;
                let src = item * len * row_bytes;
                assembled[dst..dst + len * row_bytes]
                    .copy_from_slice(&bytes[src..src + len * row_bytes]);
            }
            state = Some(part.state);
            start += len;
        }
        exact("chunk output", &assembled, &output_bytes)?;
        exact("chunk state", &read(&state.unwrap())?, &state_bytes)?;
    }
    for item in 0..batch {
        let alone = resident.slice(item..item + 1, 0..tokens);
        let result = alone.run(alone.initial.as_ref())?;
        let out_stride = tokens * vh * vd * 2;
        let state_stride = vh * kd * vd * 4;
        exact(
            "batch independence output",
            &read(&result.output)?,
            &output_bytes[item * out_stride..(item + 1) * out_stride],
        )?;
        exact(
            "batch independence state",
            &read(&result.state)?,
            &state_bytes[item * state_stride..(item + 1) * state_stride],
        )?;
    }
    if let Some(before) = initial_before {
        exact(
            "input state preserved",
            &read(resident.initial.as_ref().unwrap())?,
            &before,
        )?;
    }
    // Reject bad dtypes and noncontiguous/undersized descriptors before launch.
    let mut malformed = resident.q.clone();
    malformed.dtype = DType::F32;
    let check = |query: &C| {
        gated_delta(DeltaNetInputs {
            query,
            key: &resident.k,
            value: &resident.v,
            a: &resident.a,
            b: &resident.b,
            a_log: &resident.a_log,
            dt_bias: &resident.dt_bias,
            initial_state: resident.initial.as_ref(),
        })
    };
    if check(&malformed).is_ok() {
        return Err("dtype guard failed".into());
    }
    malformed = resident.q.clone();
    malformed.meta.strides[3] = 2;
    if check(&malformed).is_ok() {
        return Err("stride guard failed".into());
    }
    malformed = resident.q.clone();
    malformed.handle = client.empty(2);
    if check(&malformed).is_ok() {
        return Err("backing length guard failed".into());
    }
    println!(
        "{}: PASS exact repeat/chunk/single-item isolation and contract guards",
        case.name
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: qwen3_5_deltanet_gate fixture.json")?;
    let fixture: Fixture =
        serde_json::from_slice(&std::fs::read(Path::new(&path)).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if fixture.transformers != "5.2.0"
        || fixture.reference_sha256 != REFERENCE_SHA256
        || fixture.cases.len() != 3
    {
        return Err("wrong or empty reference fixture".into());
    }
    println!(
        "CUDA fixture Transformers={} Torch={} device={} source_sha256={}",
        fixture.transformers, fixture.torch, fixture.device, fixture.reference_sha256
    );
    let device = CudaDevice::default();
    let client = CudaRuntime::client(&device);
    for case in &fixture.cases {
        gate(case, &client, &device)?;
    }
    println!("PASS all resident DeltaNet CUDA gates; no Metal or cross-device equality claim");
    Ok(())
}
