//! One unmasked Qwen3.5 GatedDeltaNet mixer, not a decoder layer/backbone.
//!
//! Reference: Transformers 5.2.0 modeling_qwen3_5.py:445–626. Nine native
//! BF16 weight slots; five bias-free [out,in] projections; resident BF16
//! hidden -> convolution/SiLU -> DeltaNet -> ordinary gated RMSNorm -> output.
//! Prefill starts fresh; decode consumes BOTH histories and exactly one token.
//! Masks are refused, including B=1, rather than inheriting the reference's
//! B>1 padding-mask shortcut. Text input norm/residual/MLP are outside this unit.
//!
//! The fixed-order GPU projection deliberately does not call Burn's
//! shape-dependent Cube/autotuned matmul. One thread sums each dot product in
//! increasing input-coordinate order, in F32, then rounds once to BF16.
//! Batch/token counts do not select a reduction tree. This is a correctness
//! implementation; performance and cross-device bit identity are unestablished.
//! Inputs/weights are never uploaded, read on the host, or mutated here.
//! Resident tensor producers must honor CubeCL's ordinary client/handle and
//! stream-ordering contract. Result covers descriptor errors, not allocation,
//! CUDA initialization, kernel compilation or asynchronous driver failures;
//! those retain the upstream runtime's panic/error behavior.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use serde::{Deserialize, Serialize};
use triblespace::core::{
    blob::{Blob, encodings::tensor::{Tensor as NativeTensor, elements::BF16}},
    inline::{Inline, encodings::hash::Handle},
    repo::{BlobStoreGet, pile::PileSnapshot},
};
use super::{deltanet::{self, DeltaNetInputs}, gdn_ops};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;

pub type CudaTensor = CubeTensor<CudaRuntime>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct GdnConfig {
    pub hidden: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub epsilon: f32,
}

impl GdnConfig {
    pub fn validate(self) -> Result<(), String> {
        if self.hidden == 0 || self.key_heads == 0 || self.value_heads == 0
            || self.value_heads % self.key_heads != 0
            || !(1..=256).contains(&self.key_dim)
            || !(1..=256).contains(&self.value_dim)
            || !(1..=16).contains(&self.conv_kernel)
            || !self.epsilon.is_finite() || self.epsilon <= 0.0
        {
            return Err("GDN requires nonempty H/Hk/Hv, Hk divides Hv, K/V<=256, conv<=16, positive finite epsilon".into());
        }
        // Prove every weight shape before using unchecked host products.
        let k = count(&[self.key_heads, self.key_dim])?;
        let v = count(&[self.value_heads, self.value_dim])?;
        let c = k.checked_mul(2).and_then(|x| x.checked_add(v))
            .filter(|&x| x <= u32::MAX as usize).ok_or("QKV width overflow")?;
        count(&[c, self.hidden])?;
        count(&[v, self.hidden])?;
        count(&[self.value_heads, self.hidden])?;
        count(&[c, self.conv_kernel])?;
        Ok(())
    }
    fn key_width(self) -> usize { self.key_heads * self.key_dim }
    fn value_width(self) -> usize { self.value_heads * self.value_dim }
    fn channels(self) -> usize { 2 * self.key_width() + self.value_width() }
}

/// Exactly the nine selected native tensor handles, not a model catalogue.
/// Obtain these through typed leaf queries at the consuming module's slots.
/// Handles/IDs are opaque; no identity is reconstructed from a name or shape.
pub struct GdnSlots {
    pub qkv: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub z: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub a: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub b: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub out: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub conv: Inline<Handle<NativeTensor<BF16, 3>>>,
    pub a_log: Inline<Handle<NativeTensor<BF16, 1>>>,
    pub dt_bias: Inline<Handle<NativeTensor<BF16, 1>>>,
    pub norm: Inline<Handle<NativeTensor<BF16, 1>>>,
}

/// Private fields prevent replacement of a validated, read-only weight.
pub struct GdnMixer {
    config: GdnConfig,
    qkv: CudaTensor,
    z: CudaTensor,
    a: CudaTensor,
    b: CudaTensor,
    out: CudaTensor,
    conv: CudaTensor,
    a_log: CudaTensor,
    dt_bias: CudaTensor,
    norm: CudaTensor,
}

pub struct GdnState {
    /// Raw projected QKV, BF16 [B,C,Kconv], oldest to newest.
    pub conv: CudaTensor,
    /// F32 [B,Hv,K,V], not rounded to BF16 between tokens.
    pub recurrent: CudaTensor,
}

pub struct GdnOutput {
    pub hidden: CudaTensor,
    pub state: GdnState,
}

impl GdnMixer {
    /// Bind typed slots from one actual, validated native pile snapshot.
    ///
    /// # Safety
    /// The snapshot's backing file must remain an immutable append-only prefix
    /// through CUDA runtime teardown, including the partial page preceding each
    /// payload. No rewrite, truncation, in-place mutation, or external writer
    /// violating that premise is permitted. This obligation is passed directly
    /// to the reviewed unsafe binder; a MmapRaw owner alone is NOT provenance.
    /// No generic BlobStore/heap/MmapRaw constructor can enter this boundary.
    /// A late shape/budget error can leave earlier registrations runtime-owned.
    pub unsafe fn from_pile(
        snapshot: &PileSnapshot,
        slots: GdnSlots,
        config: GdnConfig,
        aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        config.validate()?;
        // Typed handles are consumed here, without erasure, legacy fallback,
        // upload, F16 conversion, hash-join lookup, or whole-model enumeration.
        let qkv: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.qkv).map_err(|e| e.to_string())?;
        let z: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.z).map_err(|e| e.to_string())?;
        let a: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.a).map_err(|e| e.to_string())?;
        let b: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.b).map_err(|e| e.to_string())?;
        let out: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.out).map_err(|e| e.to_string())?;
        let conv: Blob<NativeTensor<BF16, 3>> = snapshot.get(slots.conv).map_err(|e| e.to_string())?;
        let a_log: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.a_log).map_err(|e| e.to_string())?;
        let dt_bias: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.dt_bias).map_err(|e| e.to_string())?;
        let norm: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.norm).map_err(|e| e.to_string())?;
        // SAFETY: the caller guarantees the genuine snapshot's immutable
        // file-backed prefix for each exact native leaf, including page prefix.
        let result = unsafe {
            Self {
                config,
                qkv: aliases.bind_pile_leaf(qkv)?,
                z: aliases.bind_pile_leaf(z)?,
                a: aliases.bind_pile_leaf(a)?,
                b: aliases.bind_pile_leaf(b)?,
                out: aliases.bind_pile_leaf(out)?,
                conv: aliases.bind_pile_leaf(conv)?,
                a_log: aliases.bind_pile_leaf(a_log)?,
                dt_bias: aliases.bind_pile_leaf(dt_bias)?,
                norm: aliases.bind_pile_leaf(norm)?,
            }
        };
        result.check_weights()?;
        Ok(result)
    }

    fn check_weights(&self) -> Result<(), String> {
        let c = self.config;
        for (name, tensor, shape) in [
            ("qkv", &self.qkv, vec![c.channels(), c.hidden]),
            ("z", &self.z, vec![c.value_width(), c.hidden]),
            ("a", &self.a, vec![c.value_heads, c.hidden]),
            ("b", &self.b, vec![c.value_heads, c.hidden]),
            ("out", &self.out, vec![c.hidden, c.value_width()]),
            ("conv", &self.conv, vec![c.channels(), 1, c.conv_kernel]),
            ("A_log", &self.a_log, vec![c.value_heads]),
            ("dt_bias", &self.dt_bias, vec![c.value_heads]),
            ("norm", &self.norm, vec![c.value_dim]),
        ] {
            check(tensor, &self.qkv, name, &shape, DType::BF16)?;
            if tensor.handle.can_mut() {
                return Err(format!("{name}: read-only pile alias required"));
            }
        }
        Ok(())
    }

    /// Fresh unmasked sequence, T in 1..=256.
    pub fn prefill(&self, hidden: &CudaTensor, mask: Option<&CudaTensor>) -> Result<GdnOutput, String> {
        self.forward(hidden, None, mask)
    }

    /// One continuation token. Neither caller-owned state buffer is mutated.
    pub fn decode(&self, hidden: &CudaTensor, state: &GdnState, mask: Option<&CudaTensor>) -> Result<GdnOutput, String> {
        self.forward(hidden, Some(state), mask)
    }

    fn forward(&self, hidden: &CudaTensor, state: Option<&GdnState>, mask: Option<&CudaTensor>) -> Result<GdnOutput, String> {
        if mask.is_some() { return Err("GDN mixer supports unmasked inputs only".into()); }
        let shape = hidden.meta.shape().as_slice();
        if shape.len() != 3 { return Err("hidden must be [B,T,H]".into()); }
        let (batch, tokens) = (shape[0], shape[1]);
        let c = self.config;
        if !(1..=256).contains(&tokens) || (state.is_some() && tokens != 1) {
            return Err("prefill requires T=1..256; decode requires exactly T=1".into());
        }
        check(hidden, &self.qkv, "hidden", &[batch, tokens, c.hidden], DType::BF16)?;
        // Check ALL derived allocations and incoming state before any launch.
        count(&[batch, tokens, c.channels()])?;
        count(&[batch, c.channels(), c.conv_kernel])?;
        count(&[batch, c.value_heads, c.key_dim, c.value_dim])?;
        if let Some(s) = state {
            check(&s.conv, hidden, "conv state", &[batch, c.channels(), c.conv_kernel], DType::BF16)?;
            check(&s.recurrent, hidden, "recurrent state", &[batch, c.value_heads, c.key_dim, c.value_dim], DType::F32)?;
        }
        let qkv = project(hidden, &self.qkv)?;
        let z = reshape(project(hidden, &self.z)?, &[batch, tokens, c.value_heads, c.value_dim]);
        let a = project(hidden, &self.a)?;
        let b = project(hidden, &self.b)?;
        let conv = gdn_ops::causal_conv_silu(&qkv, &self.conv, state.map(|s| &s.conv))?;
        let query = reshape(channels(&conv.output, 0, c.key_width())?, &[batch, tokens, c.key_heads, c.key_dim]);
        let key = reshape(channels(&conv.output, c.key_width(), c.key_width())?, &[batch, tokens, c.key_heads, c.key_dim]);
        let value = reshape(channels(&conv.output, 2 * c.key_width(), c.value_width())?, &[batch, tokens, c.value_heads, c.value_dim]);
        let core = deltanet::gated_delta(DeltaNetInputs {
            query: &query, key: &key, value: &value, a: &a, b: &b,
            a_log: &self.a_log, dt_bias: &self.dt_bias,
            initial_state: state.map(|s| &s.recurrent),
        })?;
        let gated = gdn_ops::gated_rms_norm(&core.output, &z, &self.norm, c.epsilon)?;
        let gated = reshape(gated, &[batch, tokens, c.value_width()]);
        let output = project(&gated, &self.out)?;
        Ok(GdnOutput { hidden: output, state: GdnState { conv: conv.history, recurrent: core.state } })
    }
}

fn count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |n, &d| {
        if d == 0 { None } else { n.checked_mul(d) }
    }).filter(|&n| n <= u32::MAX as usize).ok_or_else(|| "empty or overflowing u32 shape".into())
}

fn check(tensor: &CudaTensor, like: &CudaTensor, name: &str, shape: &[usize], dtype: DType) -> Result<usize, String> {
    if tensor.meta.shape().as_slice() != shape || tensor.meta.strides().len() != shape.len()
        || tensor.dtype != dtype || tensor.qparams.is_some() || tensor.device != like.device
    {
        return Err(format!("{name}: wrong shape/dtype/device or quantized storage"));
    }
    let n = count(shape)?;
    let mut stride = 1;
    for (axis, &d) in shape.iter().enumerate().rev() {
        if d > 1 && tensor.meta.strides()[axis] != stride {
            return Err(format!("{name}: contiguous row-major storage required"));
        }
        stride *= d;
    }
    let bytes = n.checked_mul(if dtype == DType::BF16 { 2 } else { 4 }).ok_or("byte extent overflow")?;
    if tensor.handle.size_in_used() < bytes as u64 { return Err(format!("{name}: insufficient storage")); }
    Ok(n)
}

// Called only after exact contiguous element counts have been proved.
fn reshape(tensor: CudaTensor, shape: &[usize]) -> CudaTensor {
    CubeTensor::new_contiguous(tensor.client, tensor.device, shape.into(), tensor.handle, tensor.dtype)
}

#[cube(launch_unchecked)]
fn projection_kernel(
    input: &Array<bf16>, weight: &Array<bf16>, output: &mut Array<bf16>,
    elements: usize, input_width: usize, output_width: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < elements {
        let row = i / output_width;
        let out = i % output_width;
        let mut sum = 0.0f32;
        for k in 0..input_width {
            sum += f32::cast_from(input[row * input_width + k])
                * f32::cast_from(weight[out * input_width + k]);
        }
        output[i] = bf16::cast_from(sum);
    }
}

/// Fixed-order resident BF16 linear projection: [B,T,I] x [O,I] -> [B,T,O].
/// Exposed for a direct nonsquare/transposition and batch-invariance gate.
pub fn project(input: &CudaTensor, weight: &CudaTensor) -> Result<CudaTensor, String> {
    let s = input.meta.shape().as_slice();
    let w = weight.meta.shape().as_slice();
    if s.len() != 3 || w.len() != 2 || s[2] != w[1] {
        return Err("projection requires [B,T,I] and [O,I]".into());
    }
    let ni = check(input, input, "projection input", s, DType::BF16)?;
    let nw = check(weight, input, "projection weight", w, DType::BF16)?;
    let shape = [s[0], s[1], w[0]];
    let no = count(&shape)?;
    let output = input.client.empty(no * 2);
    let cube = CubeDim::new_1d(64);
    // SAFETY: checked contiguous BF16 inputs, u32-indexed extents and disjoint
    // fresh output; each thread owns one output and reads only bounded inputs.
    unsafe {
        projection_kernel::launch_unchecked::<CudaRuntime>(
            &input.client, cubecl::calculate_cube_count_elemwise(&input.client, no, cube), cube,
            ArrayArg::from_raw_parts(input.handle.clone(), ni),
            ArrayArg::from_raw_parts(weight.handle.clone(), nw),
            ArrayArg::from_raw_parts(output.clone(), no), no, s[2], w[0],
        );
    }
    Ok(CubeTensor::new_contiguous(input.client.clone(), input.device.clone(), shape.into(), output, DType::BF16))
}

#[cube(launch_unchecked)]
fn channel_kernel(input: &Array<bf16>, output: &mut Array<bf16>, n: usize, source_width: usize, start: usize, width: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n { output[i] = input[(i / width) * source_width + start + i % width]; }
}

fn channels(input: &CudaTensor, start: usize, width: usize) -> Result<CudaTensor, String> {
    let s = input.meta.shape().as_slice();
    if s.len() != 3 || width == 0 || start.checked_add(width).is_none_or(|end| end > s[2]) {
        return Err("invalid channel selection".into());
    }
    let ni = check(input, input, "channels", s, DType::BF16)?;
    let shape = [s[0], s[1], width];
    let n = count(&shape)?;
    let output = input.client.empty(n * 2);
    let cube = CubeDim::new_1d(64);
    // SAFETY: checked contiguous source, bounded channel range and fresh output.
    unsafe {
        channel_kernel::launch_unchecked::<CudaRuntime>(
            &input.client, cubecl::calculate_cube_count_elemwise(&input.client, n, cube), cube,
            ArrayArg::from_raw_parts(input.handle.clone(), ni),
            ArrayArg::from_raw_parts(output.clone(), n), n, s[2], start, width,
        );
    }
    Ok(CubeTensor::new_contiguous(input.client.clone(), input.device.clone(), shape.into(), output, DType::BF16))
}
