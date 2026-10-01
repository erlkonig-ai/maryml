//! Resident Gated DeltaNet scan, after the depthwise convolution and SiLU.
//!
//! This bounded correctness slice follows Transformers 5.2.0's
//! `torch_recurrent_gated_delta_rule` and Qwen3.5's projected gate semantics.
//! Inputs/outputs are the raw `CubeTensor` underlying Mary's Burn tensors,
//! including pile-aliased tensors: no host read, staging copy, or Fusion graph.
//! The CUDA gate checks this slice on GB10. The kernels are ordinary CubeCL
//! and retain a path to Metal, which has not been validated.
//!
//! Each GPU thread owns one (batch, value head, value coordinate) state column.
//! It walks tokens and key coordinates in ascending order using F32 state and
//! accumulators. There are no cross-item reductions, atomics, or shape-dependent
//! reduction trees. Chunking only persists the same F32 state between launches;
//! this is a recurrent scan over a chunk, not a parallel chunk algorithm.
//! The local state array is intentionally bounded to 256 key coordinates.
//!
//! The pinned Torch fallback performs its L2 normalization BEFORE widening
//! BF16 Q/K: square, sum result, epsilon addition, reciprocal square root, and
//! normalized result are rounded to BF16. We keep those boundaries explicitly.
//! Beta is sigmoid(b) rounded to BF16. Decay is computed in F32 as
//! exp(-exp(A_log) * softplus(a + dt_bias)). Q receives the additional 1/sqrt(K)
//! scale after normalization and widening. No padding mask, projections, short
//! convolution, output RMSNorm/gate, or output projection is implemented here.

use burn::tensor::DType;
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use cubecl::prelude::*;
use half::bf16;

const THREADS: u32 = 64;
pub const MAX_KEY_DIM: usize = 256;
pub const MAX_VALUE_DIM: usize = 256;
pub const MAX_CHUNK_TOKENS: usize = 256;

/// Inputs use contiguous row-major layout. All but `initial_state` are BF16.
/// Q/K have [B,T,Hk,K], V [B,T,Hv,V], a/b [B,T,Hv], and weights [Hv].
/// Hk must divide Hv; each key head serves Hv/Hk adjacent value heads.
pub struct DeltaNetInputs<'a, R: CubeRuntime> {
    pub query: &'a CubeTensor<R>,
    pub key: &'a CubeTensor<R>,
    pub value: &'a CubeTensor<R>,
    pub a: &'a CubeTensor<R>,
    pub b: &'a CubeTensor<R>,
    pub a_log: &'a CubeTensor<R>,
    pub dt_bias: &'a CubeTensor<R>,
    /// F32 [B,Hv,K,V]. None starts from zero. Never mutated.
    pub initial_state: Option<&'a CubeTensor<R>>,
}

/// Both allocations stay on the input device and can feed the next kernel.
pub struct DeltaNetOutput<R: CubeRuntime> {
    /// BF16 [B,T,Hv,V].
    pub output: CubeTensor<R>,
    /// F32 [B,Hv,K,V], including when T=1.
    pub state: CubeTensor<R>,
}

#[cube]
fn round_bf16(x: f32) -> f32 {
    f32::cast_from(bf16::cast_from(x))
}

#[cube(launch_unchecked)]
fn normalize_qk(
    query: &Array<bf16>,
    key: &Array<bf16>,
    q_normal: &mut Array<f32>,
    k_normal: &mut Array<f32>,
    rows: usize,
    scale: f32,
    #[comptime] key_dim: usize,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let base = row * key_dim;
        let mut q_sum = 0.0f32;
        let mut k_sum = 0.0f32;
        for i in 0..key_dim {
            let q = f32::cast_from(query[base + i]);
            let k = f32::cast_from(key[base + i]);
            q_sum += round_bf16(q * q);
            k_sum += round_bf16(k * k);
        }
        let q_inv = round_bf16(1.0f32 / round_bf16(round_bf16(q_sum) + 1.0e-6f32).sqrt());
        let k_inv = round_bf16(1.0f32 / round_bf16(round_bf16(k_sum) + 1.0e-6f32).sqrt());
        for i in 0..key_dim {
            q_normal[base + i] = round_bf16(f32::cast_from(query[base + i]) * q_inv) * scale;
            k_normal[base + i] = round_bf16(f32::cast_from(key[base + i]) * k_inv);
        }
    }
}

#[cube(launch_unchecked)]
fn prepare_gates(
    a: &Array<bf16>,
    b: &Array<bf16>,
    a_log: &Array<bf16>,
    dt_bias: &Array<bf16>,
    decay: &mut Array<f32>,
    beta: &mut Array<f32>,
    rows: usize,
    value_heads: usize,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let head = row % value_heads;
        let dt = f32::cast_from(a[row]) + f32::cast_from(dt_bias[head]);
        // The reference uses softplus's default threshold=20 in F32.
        let softplus = if dt > 20.0f32 { dt } else { dt.exp().log1p() };
        let g = -f32::cast_from(a_log[head]).exp() * softplus;
        decay[row] = g.exp();
        let b_value = f32::cast_from(b[row]);
        beta[row] = round_bf16(1.0f32 / (1.0f32 + (-b_value).exp()));
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn recurrent_scan(
    query: &Array<f32>,
    key: &Array<f32>,
    value: &Array<bf16>,
    decay: &Array<f32>,
    beta: &Array<f32>,
    initial: &Array<f32>,
    output: &mut Array<bf16>,
    final_state: &mut Array<f32>,
    columns: usize,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    value_dim: usize,
    #[comptime] key_dim: usize,
    #[comptime] has_initial: bool,
) {
    let column = ABSOLUTE_POS as usize;
    if column < columns {
        let coordinate = column % value_dim;
        let head = column / value_dim % value_heads;
        let batch = column / (value_dim * value_heads);
        let key_head = head / (value_heads / key_heads);
        let state_base = (batch * value_heads + head) * key_dim * value_dim + coordinate;
        let mut state = Array::<f32>::new(key_dim);
        for i in 0..key_dim {
            state[i] = if has_initial {
                initial[state_base + i * value_dim]
            } else {
                f32::new(0.0f32)
            };
        }
        for token in 0..tokens {
            let qk_base = ((batch * tokens + token) * key_heads + key_head) * key_dim;
            let gate_index = (batch * tokens + token) * value_heads + head;
            let value_index = gate_index * value_dim + coordinate;
            let gamma = decay[gate_index];
            let mut memory = 0.0f32;
            for i in 0..key_dim {
                state[i] = state[i] * gamma;
                memory += state[i] * key[qk_base + i];
            }
            let delta = (f32::cast_from(value[value_index]) - memory) * beta[gate_index];
            let mut out = 0.0f32;
            for i in 0..key_dim {
                state[i] = state[i] + key[qk_base + i] * delta;
                out += state[i] * query[qk_base + i];
            }
            output[value_index] = bf16::cast_from(out);
        }
        for i in 0..key_dim {
            final_state[state_base + i * value_dim] = state[i];
        }
    }
}

fn checked_tensor<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    like: &CubeTensor<R>,
    name: &str,
    shape: &[usize],
    dtype: DType,
) -> Result<usize, String> {
    if tensor.meta.shape().as_slice() != shape {
        return Err(format!(
            "{name}: expected shape {shape:?}, got {:?}",
            tensor.meta.shape()
        ));
    }
    if tensor.dtype != dtype || tensor.qparams.is_some() {
        return Err(format!(
            "{name}: expected plain {dtype:?}, got {:?}",
            tensor.dtype
        ));
    }
    if tensor.device != like.device {
        return Err(format!("{name}: device differs from query"));
    }
    let mut elements = 1usize;
    for (axis, &size) in shape.iter().enumerate().rev() {
        if size == 0 || (size > 1 && tensor.meta.strides()[axis] != elements) {
            return Err(format!(
                "{name}: nonempty contiguous row-major storage required"
            ));
        }
        elements = elements
            .checked_mul(size)
            .ok_or_else(|| format!("{name}: size overflow"))?;
    }
    // CubeCL's device index domain is u32 even on a 64-bit host.
    if elements > u32::MAX as usize {
        return Err(format!("{name}: exceeds the u32 device index domain"));
    }
    let width = if dtype == DType::BF16 { 2 } else { 4 };
    if tensor.handle.size_in_used() < (elements * width) as u64 {
        return Err(format!("{name}: backing handle is shorter than its shape"));
    }
    Ok(elements)
}

/// Execute a bounded resident chunk. Contract errors precede any GPU work.
///
/// All inputs must belong to the same CubeCL device; the caller must honor
/// CubeCL's ordinary stream ordering contract for input producers. Pile-backed
/// input tensors are read-only. Outputs always own separate device storage.
pub fn gated_delta<R: CubeRuntime>(
    inputs: DeltaNetInputs<'_, R>,
) -> Result<DeltaNetOutput<R>, String> {
    let q = inputs.query;
    let shape = q.meta.shape().as_slice();
    if shape.len() != 4 || inputs.value.meta.shape().as_slice().len() != 4 {
        return Err("query and value must have rank 4".into());
    }
    let (batch, tokens, key_heads, key_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let value_shape = inputs.value.meta.shape().as_slice();
    let (value_heads, value_dim) = (value_shape[2], value_shape[3]);
    if batch == 0
        || key_heads == 0
        || value_heads == 0
        || value_heads % key_heads != 0
        || !(1..=MAX_CHUNK_TOKENS).contains(&tokens)
        || !(1..=MAX_KEY_DIM).contains(&key_dim)
        || !(1..=MAX_VALUE_DIM).contains(&value_dim)
    {
        return Err("require nonempty B/H, Hk dividing Hv, T/K/V in 1..=256".into());
    }
    let q_len = checked_tensor(q, q, "query", shape, DType::BF16)?;
    checked_tensor(inputs.key, q, "key", shape, DType::BF16)?;
    let output_shape = [batch, tokens, value_heads, value_dim];
    let output_len = checked_tensor(inputs.value, q, "value", &output_shape, DType::BF16)?;
    let gate_shape = [batch, tokens, value_heads];
    let gate_len = checked_tensor(inputs.a, q, "a", &gate_shape, DType::BF16)?;
    checked_tensor(inputs.b, q, "b", &gate_shape, DType::BF16)?;
    checked_tensor(inputs.a_log, q, "A_log", &[value_heads], DType::BF16)?;
    checked_tensor(inputs.dt_bias, q, "dt_bias", &[value_heads], DType::BF16)?;
    let state_shape = [batch, value_heads, key_dim, value_dim];
    let state_len = state_shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or("state exceeds u32 index domain")?;
    if let Some(state) = inputs.initial_state {
        checked_tensor(state, q, "initial_state", &state_shape, DType::F32)?;
    }

    let client = &q.client;
    let q_normal = client.empty(q_len * 4);
    let k_normal = client.empty(q_len * 4);
    let decay = client.empty(gate_len * 4);
    let beta = client.empty(gate_len * 4);
    let output = client.empty(output_len * 2);
    let state = client.empty(state_len * 4);
    // The false specialization never reads the placeholder. It still receives
    // a correctly sized binding so there is no invalid or null kernel argument.
    let initial = inputs
        .initial_state
        .map(|s| s.handle.clone())
        .unwrap_or_else(|| state.clone());
    let cube = CubeDim::new_1d(THREADS);
    let rows = q_len / key_dim;
    let columns = batch * value_heads * value_dim;
    unsafe {
        normalize_qk::launch_unchecked::<R>(
            client,
            cubecl::calculate_cube_count_elemwise(client, rows, cube),
            cube,
            ArrayArg::from_raw_parts(q.handle.clone(), q_len),
            ArrayArg::from_raw_parts(inputs.key.handle.clone(), q_len),
            ArrayArg::from_raw_parts(q_normal.clone(), q_len),
            ArrayArg::from_raw_parts(k_normal.clone(), q_len),
            rows,
            1.0 / (key_dim as f32).sqrt(),
            key_dim,
        );
        prepare_gates::launch_unchecked::<R>(
            client,
            cubecl::calculate_cube_count_elemwise(client, gate_len, cube),
            cube,
            ArrayArg::from_raw_parts(inputs.a.handle.clone(), gate_len),
            ArrayArg::from_raw_parts(inputs.b.handle.clone(), gate_len),
            ArrayArg::from_raw_parts(inputs.a_log.handle.clone(), value_heads),
            ArrayArg::from_raw_parts(inputs.dt_bias.handle.clone(), value_heads),
            ArrayArg::from_raw_parts(decay.clone(), gate_len),
            ArrayArg::from_raw_parts(beta.clone(), gate_len),
            gate_len,
            value_heads,
        );
        recurrent_scan::launch_unchecked::<R>(
            client,
            cubecl::calculate_cube_count_elemwise(client, columns, cube),
            cube,
            ArrayArg::from_raw_parts(q_normal, q_len),
            ArrayArg::from_raw_parts(k_normal, q_len),
            ArrayArg::from_raw_parts(inputs.value.handle.clone(), output_len),
            ArrayArg::from_raw_parts(decay, gate_len),
            ArrayArg::from_raw_parts(beta, gate_len),
            ArrayArg::from_raw_parts(initial, state_len),
            ArrayArg::from_raw_parts(output.clone(), output_len),
            ArrayArg::from_raw_parts(state.clone(), state_len),
            columns,
            tokens,
            key_heads,
            value_heads,
            value_dim,
            key_dim,
            inputs.initial_state.is_some(),
        );
    }
    Ok(DeltaNetOutput {
        output: CubeTensor::new_contiguous(
            client.clone(),
            q.device.clone(),
            output_shape.into(),
            output,
            DType::BF16,
        ),
        state: CubeTensor::new_contiguous(
            client.clone(),
            q.device.clone(),
            state_shape.into(),
            state,
            DType::F32,
        ),
    })
}
