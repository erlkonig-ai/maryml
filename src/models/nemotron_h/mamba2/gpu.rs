//! Native projected-input Mamba2, without projection, outer norm or residual.
//!
//! BF16 input/conv history, F32 recurrent state and accumulation, BF16 output.
//! Native fast dt policy is softplus only, dt_limit=[0,infinity], NOT the slow
//! torch_forward floor0.001. Config.time_step_min must be explicitly zero here.
//! F32 state is an explicit native precision choice, NOT parity with the generic
//! Transformers BF16 persisted cache. Conv -> BF16 ->
//! SiLU -> BF16 is explicit. Gate precedes grouped F32 RMSNorm; gain is ordinary.
//! Each state-column thread scans T sequentially; this is not a parallel SSD.
//! Immutable inputs and fresh outputs follow existing CubeTensor stream and
//! binding lifetimes. No host read, synchronization or input mutation occurs.
//! Result covers descriptor/admission errors, not asynchronous CUDA failures:
//! an owning session must fence/handle device errors before committing state.

use super::contract::{Config, Result};
use burn::tensor::DType;
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use cubecl::prelude::*;
use half::bf16;

const THREADS: u32 = 64;
pub const MAX_CHUNK: usize = 256;
pub const MAX_STATE: usize = 256;
pub const MAX_CONV: usize = 16;

/// Read-only decoded parameter buffers. F32 scalar vectors may contain exact
/// promotions of source BF16 values; loading/FP8 projections are not here.
pub struct Parameters<'a, R: CubeRuntime> {
    pub conv_weight: &'a CubeTensor<R>,       // BF16 [channels,1,K]
    pub conv_bias: Option<&'a CubeTensor<R>>, // BF16 [channels]
    pub a_log: &'a CubeTensor<R>,             // F32 [heads]
    pub dt_bias: &'a CubeTensor<R>,           // F32 [heads]
    pub d: &'a CubeTensor<R>,                 // F32 [heads]
    pub norm_gain: &'a CubeTensor<R>,         // BF16 [inner]
}

/// Geometry/device binding only, NOT immutable model/layer identity. The owner
/// must bind parameters and reset attention+conv+recurrence together. Private
/// tensors prevent forging history descriptors through this API.
pub struct State<R: CubeRuntime> {
    config: Config,
    batch: usize,
    attended: usize,
    conv: CubeTensor<R>,
    recurrent: CubeTensor<R>,
}
impl<R: CubeRuntime> State<R> {
    pub fn attended(&self) -> usize {
        self.attended
    }
    pub fn conv(&self) -> &CubeTensor<R> {
        &self.conv
    }
    pub fn recurrent(&self) -> &CubeTensor<R> {
        &self.recurrent
    }

    /// Fresh zero histories; existing state remains usable and unmodified.
    /// Passing None to chunk is the allocation-free logical reset alternative.
    pub fn reset(&self, max_bytes: usize) -> Result<Self> {
        let e = self.config.extents(self.batch, 1)?;
        budget(&[(e.conv, 2), (e.recurrent, 4)], max_bytes)?;
        let client = &self.conv.client;
        let conv = client.empty(e.conv * 2);
        let recurrent = client.empty(e.recurrent * 4);
        let cube = CubeDim::new_1d(THREADS);
        unsafe {
            zero_history::launch_unchecked::<R>(
                client,
                cubecl::calculate_cube_count_elemwise(client, e.conv.max(e.recurrent), cube),
                cube,
                ArrayArg::from_raw_parts(conv.clone(), e.conv),
                ArrayArg::from_raw_parts(recurrent.clone(), e.recurrent),
                e.conv,
                e.recurrent,
            );
        }
        Ok(Self {
            config: self.config,
            batch: self.batch,
            attended: 0,
            conv: CubeTensor::new_contiguous(
                client.clone(),
                self.conv.device.clone(),
                self.conv.meta.shape().clone(),
                conv,
                DType::BF16,
            ),
            recurrent: CubeTensor::new_contiguous(
                client.clone(),
                self.conv.device.clone(),
                self.recurrent.meta.shape().clone(),
                recurrent,
                DType::F32,
            ),
        })
    }
}
pub struct Output<R: CubeRuntime> {
    pub values: CubeTensor<R>, // BF16 [B,T,inner], before output projection
    pub state: State<R>,
    /// Fresh output+history+temporary logical bytes, excluding inputs/runtime.
    pub allocated_bytes: usize,
}

#[cube]
fn silu(x: f32) -> f32 {
    if x >= 0.0f32 {
        x / (1.0f32 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0f32 + e)
    }
}

#[cube(launch_unchecked)]
fn zero_history(conv: &mut Array<bf16>, state: &mut Array<f32>, nc: usize, ns: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < nc {
        conv[i] = bf16::cast_from(0.0f32);
    }
    if i < ns {
        state[i] = 0.0f32;
    }
}

#[cube(launch_unchecked)]
fn convolution(
    input: &Array<bf16>,
    weights: &Array<bf16>,
    bias: &Array<bf16>,
    initial: &Array<bf16>,
    output: &mut Array<bf16>,
    history_out: &mut Array<bf16>,
    rows: usize,
    tokens: usize,
    projected: usize,
    inner: usize,
    channels: usize,
    #[comptime] kernel: usize,
    #[comptime] has_initial: bool,
    #[comptime] has_bias: bool,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let batch = row / channels;
        let ch = row % channels;
        let mut history = Array::<bf16>::new(kernel);
        for k in 0..kernel {
            history[k] = if has_initial {
                initial[row * kernel + k]
            } else {
                bf16::cast_from(0.0f32)
            };
        }
        for t in 0..tokens {
            for k in 0..kernel - 1 {
                history[k] = history[k + 1];
            }
            history[kernel - 1] = input[(batch * tokens + t) * projected + inner + ch];
            let mut sum = 0.0f32;
            if has_bias {
                sum = f32::cast_from(bias[ch]);
            }
            for k in 0..kernel {
                sum += f32::cast_from(history[k]) * f32::cast_from(weights[ch * kernel + k]);
            }
            let rounded = f32::cast_from(bf16::cast_from(sum));
            output[(batch * tokens + t) * channels + ch] = bf16::cast_from(silu(rounded));
        }
        for k in 0..kernel {
            history_out[row * kernel + k] = history[k];
        }
    }
}

#[cube(launch_unchecked)]
fn recurrence(
    input: &Array<bf16>,
    conv: &Array<bf16>,
    a_log: &Array<f32>,
    dt_bias: &Array<f32>,
    d_skip: &Array<f32>,
    initial: &Array<f32>,
    gated: &mut Array<f32>,
    final_state: &mut Array<f32>,
    columns: usize,
    tokens: usize,
    projected: usize,
    inner: usize,
    channels: usize,
    heads: usize,
    head_dim: usize,
    groups: usize,
    dt_floor: f32,
    #[comptime] state_dim: usize,
    #[comptime] has_initial: bool,
) {
    let column = ABSOLUTE_POS as usize;
    if column < columns {
        let batch = column / inner;
        let coordinate = column % inner;
        let head = coordinate / head_dim;
        let group = head / (heads / groups);
        let base = column * state_dim;
        let mut state = Array::<f32>::new(state_dim);
        for n in 0..state_dim {
            state[n] = if has_initial {
                initial[base + n]
            } else {
                f32::new(0.0f32)
            };
        }
        for t in 0..tokens {
            let row = batch * tokens + t;
            let dt_raw =
                f32::cast_from(input[row * projected + inner + channels + head]) + dt_bias[head];
            let soft = if dt_raw > 20.0f32 {
                dt_raw
            } else {
                dt_raw.exp().log1p()
            };
            let dt = if soft < dt_floor { dt_floor } else { soft };
            let decay = (-a_log[head].exp() * dt).exp();
            let x = f32::cast_from(conv[row * channels + coordinate]);
            let mut y = d_skip[head] * x;
            for n in 0..state_dim {
                let b = f32::cast_from(conv[row * channels + inner + group * state_dim + n]);
                let c = f32::cast_from(
                    conv[row * channels + inner + groups * state_dim + group * state_dim + n],
                );
                state[n] = decay * state[n] + dt * b * x;
                y += state[n] * c;
            }
            gated[row * inner + coordinate] =
                y * silu(f32::cast_from(input[row * projected + coordinate]));
        }
        for n in 0..state_dim {
            final_state[base + n] = state[n];
        }
    }
}

#[cube(launch_unchecked)]
fn grouped_norm(
    input: &Array<f32>,
    gain: &Array<bf16>,
    output: &mut Array<bf16>,
    rows: usize,
    inner: usize,
    width: usize,
    epsilon: f32,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let base = row * width;
        let gain_base = base % inner;
        let mut sum = 0.0f32;
        for i in 0..width {
            let x = input[base + i];
            sum += x * x;
        }
        let inv = 1.0f32 / (sum / f32::cast_from(width) + epsilon).sqrt();
        for i in 0..width {
            output[base + i] =
                bf16::cast_from(input[base + i] * inv * f32::cast_from(gain[gain_base + i]));
        }
    }
}

fn check<R: CubeRuntime>(
    t: &CubeTensor<R>,
    like: &CubeTensor<R>,
    name: &str,
    shape: &[usize],
    dtype: DType,
) -> Result<usize> {
    if t.meta.shape().as_slice() != shape || t.meta.strides().len() != shape.len() {
        return Err(format!("{name}: shape differs from {shape:?}"));
    }
    if t.dtype != dtype || t.qparams.is_some() || t.device != like.device {
        return Err(format!("{name}: plain {dtype:?} on input device required"));
    }
    let mut n = 1usize;
    for (axis, &size) in shape.iter().enumerate().rev() {
        if size == 0 || (size > 1 && t.meta.strides()[axis] != n) {
            return Err(format!("{name}: contiguous nonempty required"));
        }
        n = n.checked_mul(size).ok_or("tensor size overflow")?;
    }
    let width = if dtype == DType::BF16 { 2 } else { 4 };
    if n > u32::MAX as usize || t.handle.size_in_used() < (n as u64) * width {
        return Err(format!("{name}: u32 index domain/backing size exceeded"));
    }
    Ok(n)
}
fn budget(parts: &[(usize, usize)], max: usize) -> Result<usize> {
    let mut bytes = 0usize;
    for &(count, width) in parts {
        if count > u32::MAX as usize {
            return Err("u32 device index domain exceeded".into());
        }
        bytes = bytes
            .checked_add(count.checked_mul(width).ok_or("byte size overflow")?)
            .ok_or("byte total overflow")?;
    }
    if bytes > max {
        return Err(format!("fresh buffer budget: {bytes}>{max}"));
    }
    Ok(bytes)
}

/// Accepts BF16 [B,T,projected], T1..256, N1..256, K1..16. Full Super geometry
/// fits. These are initial kernel implementation bounds, not model context
/// limits. Config finite/F32 checks and all descriptor/admission checks happen
/// before allocation or launch. Tensor values must be finite by caller contract;
/// no hidden host read or finite-reduction launch is inserted here.
pub fn chunk<R: CubeRuntime>(
    c: Config,
    p: &Parameters<'_, R>,
    input: &CubeTensor<R>,
    previous: Option<&State<R>>,
    max_bytes: usize,
) -> Result<Output<R>> {
    let l = c.layout()?;
    let shape = input.meta.shape().as_slice();
    if shape.len() != 3 {
        return Err("projected input must be rank3".into());
    }
    let (batch, tokens) = (shape[0], shape[1]);
    let e = c.extents(batch, tokens)?;
    let eps = c.norm_epsilon as f32;
    let floor = c.time_step_min as f32;
    if !(1..=MAX_CHUNK).contains(&tokens)
        || !(1..=MAX_STATE).contains(&c.state_dim)
        || !(1..=MAX_CONV).contains(&c.conv_kernel)
        || !eps.is_finite()
        || eps <= 0.0
        || !floor.is_finite()
    {
        return Err("kernel T/N/K bounds or finite F32 configuration exceeded".into());
    }
    if c.time_step_min != 0.0 {
        return Err("native fast dt policy requires time_step_min=0 (not slow torch floor)".into());
    }
    check(
        input,
        input,
        "input",
        &[batch, tokens, l.projected],
        DType::BF16,
    )?;
    let nw = check(
        p.conv_weight,
        input,
        "conv weight",
        &[l.conv_channels, 1, c.conv_kernel],
        DType::BF16,
    )?;
    if let Some(b) = p.conv_bias {
        check(b, input, "conv bias", &[l.conv_channels], DType::BF16)?;
    }
    for (name, t) in [("A_log", p.a_log), ("dt_bias", p.dt_bias), ("D", p.d)] {
        check(t, input, name, &[c.heads], DType::F32)?;
    }
    check(p.norm_gain, input, "norm gain", &[l.inner], DType::BF16)?;
    let attended = previous
        .map_or(0, |s| s.attended)
        .checked_add(tokens)
        .ok_or("position overflow")?;
    let conv_shape = [batch, l.conv_channels, c.conv_kernel];
    let state_shape = [batch, c.heads, c.head_dim, c.state_dim];
    if let Some(s) = previous {
        if s.config != c || s.batch != batch {
            return Err("state binding differs".into());
        }
        check(&s.conv, input, "conv history", &conv_shape, DType::BF16)?;
        check(&s.recurrent, input, "recurrence", &state_shape, DType::F32)?;
    }
    let nc = batch
        .checked_mul(tokens)
        .and_then(|n| n.checked_mul(l.conv_channels))
        .ok_or("conv output overflow")?;
    let allocated_bytes = budget(
        &[
            (e.output, 2),
            (e.output, 4),
            (nc, 2),
            (e.conv, 2),
            (e.recurrent, 4),
        ],
        max_bytes,
    )?;
    let client = &input.client;
    let cube = CubeDim::new_1d(THREADS);
    let conv = client.empty(nc * 2);
    let history = client.empty(e.conv * 2);
    let state = client.empty(e.recurrent * 4);
    let gated = client.empty(e.output * 4);
    let output = client.empty(e.output * 2);
    let initial_conv = previous
        .map(|s| s.conv.handle.clone())
        .unwrap_or_else(|| history.clone());
    let initial_state = previous
        .map(|s| s.recurrent.handle.clone())
        .unwrap_or_else(|| state.clone());
    // Absent bias/history are compile-time branches: placeholders are never read.
    let bias = p.conv_bias.unwrap_or(p.conv_weight);
    unsafe {
        convolution::launch_unchecked::<R>(
            client,
            cubecl::calculate_cube_count_elemwise(client, batch * l.conv_channels, cube),
            cube,
            ArrayArg::from_raw_parts(input.handle.clone(), e.input),
            ArrayArg::from_raw_parts(p.conv_weight.handle.clone(), nw),
            ArrayArg::from_raw_parts(
                bias.handle.clone(),
                if p.conv_bias.is_some() {
                    l.conv_channels
                } else {
                    nw
                },
            ),
            ArrayArg::from_raw_parts(initial_conv, e.conv),
            ArrayArg::from_raw_parts(conv.clone(), nc),
            ArrayArg::from_raw_parts(history.clone(), e.conv),
            batch * l.conv_channels,
            tokens,
            l.projected,
            l.inner,
            l.conv_channels,
            c.conv_kernel,
            previous.is_some(),
            p.conv_bias.is_some(),
        );
        recurrence::launch_unchecked::<R>(
            client,
            cubecl::calculate_cube_count_elemwise(client, batch * l.inner, cube),
            cube,
            ArrayArg::from_raw_parts(input.handle.clone(), e.input),
            ArrayArg::from_raw_parts(conv, nc),
            ArrayArg::from_raw_parts(p.a_log.handle.clone(), c.heads),
            ArrayArg::from_raw_parts(p.dt_bias.handle.clone(), c.heads),
            ArrayArg::from_raw_parts(p.d.handle.clone(), c.heads),
            ArrayArg::from_raw_parts(initial_state, e.recurrent),
            ArrayArg::from_raw_parts(gated.clone(), e.output),
            ArrayArg::from_raw_parts(state.clone(), e.recurrent),
            batch * l.inner,
            tokens,
            l.projected,
            l.inner,
            l.conv_channels,
            c.heads,
            c.head_dim,
            c.groups,
            floor,
            c.state_dim,
            previous.is_some(),
        );
        grouped_norm::launch_unchecked::<R>(
            client,
            cubecl::calculate_cube_count_elemwise(client, batch * tokens * c.groups, cube),
            cube,
            ArrayArg::from_raw_parts(gated, e.output),
            ArrayArg::from_raw_parts(p.norm_gain.handle.clone(), l.inner),
            ArrayArg::from_raw_parts(output.clone(), e.output),
            batch * tokens * c.groups,
            l.inner,
            l.norm_group,
            eps,
        );
    }
    Ok(Output {
        values: CubeTensor::new_contiguous(
            client.clone(),
            input.device.clone(),
            [batch, tokens, l.inner].into(),
            output,
            DType::BF16,
        ),
        state: State {
            config: c,
            batch,
            attended,
            conv: CubeTensor::new_contiguous(
                client.clone(),
                input.device.clone(),
                conv_shape.into(),
                history,
                DType::BF16,
            ),
            recurrent: CubeTensor::new_contiguous(
                client.clone(),
                input.device.clone(),
                state_shape.into(),
                state,
                DType::F32,
            ),
        },
        allocated_bytes,
    })
}
