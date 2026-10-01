//! Resident Qwen3.5 Gated DeltaNet convolution and gated output RMSNorm.
//!
//! These primitives follow Transformers5.2.0 `torch_causal_conv1d_update`
//! and `Qwen3_5RMSNormGated`. They accept the raw CubeTensor buffers underlying
//! Mary/Burn tensors, perform no host reads or staging uploads, and never
//! mutate an input buffer. The kernels are generic CubeCL; Metal is untested.
//!
//! The convolution consumes already projected QKV. In the reference, the
//! padding mask acts on hidden states BEFORE projection, and masked tokens
//! still advance convolution history. This primitive does not apply a mask
//! or infer sequence resets. None history means a new sequence; Some history
//! means continuation, independently for each batch item. It returns the
//! reference's K most recent projected samples (oldest first), not K-1.
//!
//! Convolution accumulates in F32, rounds its result to BF16, then applies
//! SiLU in F32 and rounds to BF16, preserving the fallback's Conv1d -> SiLU
//! boundary. The output norm computes variance and normalization in F32,
//! rounds normalized values to BF16, multiplies by the ordinary BF16 gain
//! and rounds again, then multiplies by F32 SiLU(gate) and returns BF16.
//! The gain is not the `1 + weight` of Qwen3.5's text RMSNorm.

use burn::tensor::DType;
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use cubecl::prelude::*;
use half::bf16;

const THREADS: u32 = 64;
pub const MAX_CONV_KERNEL: usize = 16;
pub const MAX_CHUNK_TOKENS: usize = 256;
pub const MAX_NORM_WIDTH: usize = 256;

/// Both returned tensors own fresh device storage, on the input's device.
pub struct ConvOutput<R: CubeRuntime> {
    /// BF16 [B,T,C], including SiLU activation.
    pub output: CubeTensor<R>,
    /// BF16 [B,C,K], the K most recent input samples, oldest first.
    pub history: CubeTensor<R>,
}

#[cube(launch_unchecked)]
fn conv_silu_kernel(
    input: &Array<bf16>,
    weight: &Array<bf16>,
    initial: &Array<bf16>,
    output: &mut Array<bf16>,
    final_history: &mut Array<bf16>,
    rows: usize,
    tokens: usize,
    channels: usize,
    #[comptime] kernel: usize,
    #[comptime] has_initial: bool,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let batch = row / channels;
        let channel = row % channels;
        let mut history = Array::<bf16>::new(kernel);
        for i in 0..kernel {
            history[i] = bf16::cast_from(0.0f32);
            if has_initial {
                history[i] = initial[row * kernel + i];
            }
        }
        for token in 0..tokens {
            let index = (batch * tokens + token) * channels + channel;
            for i in 0..kernel - 1 {
                history[i] = history[i + 1];
            }
            history[kernel - 1] = input[index];
            let mut sum = 0.0f32;
            for i in 0..kernel {
                sum += f32::cast_from(history[i]) * f32::cast_from(weight[channel * kernel + i]);
            }
            let conv = f32::cast_from(bf16::cast_from(sum));
            output[index] = bf16::cast_from(conv / (1.0f32 + (-conv).exp()));
        }
        for i in 0..kernel {
            final_history[row * kernel + i] = history[i];
        }
    }
}

#[cube(launch_unchecked)]
fn rms_gate_kernel(
    input: &Array<bf16>,
    gate: &Array<bf16>,
    gain: &Array<bf16>,
    output: &mut Array<bf16>,
    rows: usize,
    epsilon: f32,
    #[comptime] width: usize,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let base = row * width;
        let mut square_sum = 0.0f32;
        for i in 0..width {
            let x = f32::cast_from(input[base + i]);
            square_sum += x * x;
        }
        let inverse_rms =
            1.0f32 / (square_sum / f32::new(comptime!(width as f32)) + epsilon).sqrt();
        for i in 0..width {
            let normalized = bf16::cast_from(f32::cast_from(input[base + i]) * inverse_rms);
            let weighted = bf16::cast_from(f32::cast_from(normalized) * f32::cast_from(gain[i]));
            let z = f32::cast_from(gate[base + i]);
            let activated_gate = z / (1.0f32 + (-z).exp());
            output[base + i] = bf16::cast_from(f32::cast_from(weighted) * activated_gate);
        }
    }
}

/// Validate exact BF16 descriptors without silently copying noncontiguous data.
fn check<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    like: &CubeTensor<R>,
    name: &str,
    shape: &[usize],
) -> Result<usize, String> {
    if tensor.meta.shape().as_slice() != shape || tensor.meta.strides().len() != shape.len() {
        return Err(format!(
            "{name}: expected shape {shape:?}, got {:?}",
            tensor.meta.shape()
        ));
    }
    if tensor.dtype != DType::BF16 || tensor.qparams.is_some() {
        return Err(format!("{name}: plain BF16 required"));
    }
    if tensor.device != like.device {
        return Err(format!("{name}: device differs from input"));
    }
    let mut count = 1usize;
    for (axis, &dimension) in shape.iter().enumerate().rev() {
        if dimension == 0 || (dimension > 1 && tensor.meta.strides()[axis] != count) {
            return Err(format!(
                "{name}: nonempty contiguous row-major storage required"
            ));
        }
        count = count
            .checked_mul(dimension)
            .ok_or_else(|| format!("{name}: shape overflow"))?;
    }
    if count > u32::MAX as usize || tensor.handle.size_in_used() < (count * 2) as u64 {
        return Err(format!(
            "{name}: u32 index overflow or insufficient backing storage"
        ));
    }
    Ok(count)
}

/// Depthwise causal convolution followed by SiLU, without a bias.
///
/// Input: BF16 [B,T,C]; weight: BF16 [C,1,K]; optional read-only history:
/// BF16 [B,C,K]. All layouts are contiguous. T is in 1..=256, K in 1..=16.
/// The weight's last position multiplies the current sample. Passing None
/// corresponds to the reference's left-zero-padded prefill convolution.
/// Continuation uses the last K projected samples, just like its update helper.
pub fn causal_conv_silu<R: CubeRuntime>(
    input: &CubeTensor<R>,
    weight: &CubeTensor<R>,
    history: Option<&CubeTensor<R>>,
) -> Result<ConvOutput<R>, String> {
    let shape = input.meta.shape().as_slice();
    let weights = weight.meta.shape().as_slice();
    if shape.len() != 3 || weights.len() != 3 {
        return Err("convolution input and weight must have rank 3".into());
    }
    let (batch, tokens, channels, kernel) = (shape[0], shape[1], shape[2], weights[2]);
    if !(1..=MAX_CHUNK_TOKENS).contains(&tokens) || !(1..=MAX_CONV_KERNEL).contains(&kernel) {
        return Err("convolution requires T in 1..=256 and K in 1..=16".into());
    }
    let count = check(input, input, "input", &[batch, tokens, channels])?;
    let weight_count = check(weight, input, "weight", &[channels, 1, kernel])?;
    let history_shape = [batch, channels, kernel];
    let history_count = history_shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or("history exceeds u32 index domain")?;
    if let Some(h) = history {
        check(h, input, "history", &history_shape)?;
    }
    let client = &input.client;
    let output = client.empty(count * 2);
    let final_history = client.empty(history_count * 2);
    // None specialization never reads the correctly sized placeholder.
    let initial = history
        .map(|h| h.handle.clone())
        .unwrap_or_else(|| final_history.clone());
    let rows = batch * channels;
    let cube = CubeDim::new_1d(THREADS);
    unsafe {
        conv_silu_kernel::launch_unchecked::<R>(
            client,
            cubecl::calculate_cube_count_elemwise(client, rows, cube),
            cube,
            ArrayArg::from_raw_parts(input.handle.clone(), count),
            ArrayArg::from_raw_parts(weight.handle.clone(), weight_count),
            ArrayArg::from_raw_parts(initial, history_count),
            ArrayArg::from_raw_parts(output.clone(), count),
            ArrayArg::from_raw_parts(final_history.clone(), history_count),
            rows,
            tokens,
            channels,
            kernel,
            history.is_some(),
        );
    }
    Ok(ConvOutput {
        output: CubeTensor::new_contiguous(
            client.clone(),
            input.device.clone(),
            shape.into(),
            output,
            DType::BF16,
        ),
        history: CubeTensor::new_contiguous(
            client.clone(),
            input.device.clone(),
            history_shape.into(),
            final_history,
            DType::BF16,
        ),
    })
}

/// Ordinary RMSNorm followed by SiLU(gate), at the reference BF16 boundaries.
///
/// Input/gate: BF16 [B,T,H,V]; gain: BF16 [V]. T and V are in 1..=256.
/// Epsilon must be finite and positive. Output has the input's shape/device.
/// This function is state-free and reduces each V-wide row independently.
pub fn gated_rms_norm<R: CubeRuntime>(
    input: &CubeTensor<R>,
    gate: &CubeTensor<R>,
    gain: &CubeTensor<R>,
    epsilon: f32,
) -> Result<CubeTensor<R>, String> {
    let shape = input.meta.shape().as_slice();
    if shape.len() != 4 {
        return Err("gated RMSNorm input must have rank 4".into());
    }
    let width = shape[3];
    if !(1..=MAX_CHUNK_TOKENS).contains(&shape[1])
        || !(1..=MAX_NORM_WIDTH).contains(&width)
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err("gated RMSNorm requires T/V in 1..=256 and finite positive epsilon".into());
    }
    let count = check(input, input, "input", shape)?;
    check(gate, input, "gate", shape)?;
    check(gain, input, "gain", &[width])?;
    let client = &input.client;
    let output = client.empty(count * 2);
    let rows = count / width;
    let cube = CubeDim::new_1d(THREADS);
    unsafe {
        rms_gate_kernel::launch_unchecked::<R>(
            client,
            cubecl::calculate_cube_count_elemwise(client, rows, cube),
            cube,
            ArrayArg::from_raw_parts(input.handle.clone(), count),
            ArrayArg::from_raw_parts(gate.handle.clone(), count),
            ArrayArg::from_raw_parts(gain.handle.clone(), width),
            ArrayArg::from_raw_parts(output.clone(), count),
            rows,
            epsilon,
            width,
        );
    }
    Ok(CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        shape.into(),
        output,
        DType::BF16,
    ))
}
