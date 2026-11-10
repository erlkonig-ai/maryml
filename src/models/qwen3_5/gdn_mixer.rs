//! One unmasked Qwen3.5 GatedDeltaNet mixer, not a decoder layer/backbone.
//!
//! Reference: Transformers 5.2.0 modeling_qwen3_5.py:445–626. Nine native
//! BF16 weight slots; five bias-free [out,in] projections; resident BF16
//! hidden -> convolution/SiLU -> DeltaNet -> ordinary gated RMSNorm -> output.
//! Prefill starts fresh; decode consumes BOTH histories and exactly one token.
//! Masks are refused, including B=1, rather than inheriting the reference's
//! B>1 padding-mask shortcut. Text input norm/residual/MLP are outside this unit.
//!
//! Projection uses a forced, fixed BF16 MMA blueprint with F32 accumulation
//! when K is divisible by 16 and N by 8. Other shapes retain the serial F32
//! GPU dot product. Neither dispatch nor the reduction tile depends on B/T/M.
//! This is an experimental accelerated path: execution/reproducibility and
//! performance must be measured before promotion; cross-device identity is
//! not claimed. It does not call Burn's autotuned matmul.
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
    repo::BlobStoreGet,
};
use super::{deltanet::{self, DeltaNetInputs}, gdn_ops, roles::LinearSlot};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use crate::nn::nvfp4_gemv::Nvfp4;

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
    pub qkv: LinearSlot,
    pub z: LinearSlot,
    pub a: LinearSlot,
    pub b: LinearSlot,
    pub out: LinearSlot,
    pub conv: Inline<Handle<NativeTensor<BF16, 3>>>,
    pub a_log: Inline<Handle<NativeTensor<BF16, 1>>>,
    pub dt_bias: Inline<Handle<NativeTensor<BF16, 1>>>,
    pub norm: Inline<Handle<NativeTensor<BF16, 1>>>,
}

/// Private fields prevent replacement of a validated, read-only weight.
pub struct GdnMixer {
    config: GdnConfig,
    qkv: Linear,
    z: Linear,
    a: Linear,
    b: Linear,
    out: Linear,
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
    /// Bind fixed typed slots through a local or exact-acquiring reader.
    ///
    /// # Safety
    /// Every returned leaf must originate in a genuine validated native pile.
    /// Its backing file must remain an immutable append-only prefix
    /// through CUDA runtime teardown, including the partial page preceding each
    /// payload. No rewrite, truncation, in-place mutation, or external writer
    /// violating that premise is permitted. This obligation is passed directly
    /// to the reviewed unsafe binder; a MmapRaw owner alone is NOT provenance.
    /// A generic reader is not provenance: heap blobs remain refused and merely
    /// wrapping an arbitrary MmapRaw does not satisfy this unsafe obligation.
    /// A late shape/budget error can leave earlier registrations runtime-owned.
    pub unsafe fn from_pile<R: BlobStoreGet>(
        snapshot: &R,
        slots: GdnSlots,
        config: GdnConfig,
        aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        // SAFETY: forwarded unchanged.
        unsafe { Self::from_pile_with(snapshot, slots, config, aliases, DecoderPrecision::Bf16) }
    }

    /// [`Self::from_pile`] with the five projections bound for `precision`
    /// ([`bind_linear`]): an NVFP4 precision never leaves a BF16 projection
    /// registered or resident.
    ///
    /// # Safety
    /// As [`Self::from_pile`].
    pub unsafe fn from_pile_with<R: BlobStoreGet>(
        snapshot: &R,
        slots: GdnSlots,
        config: GdnConfig,
        aliases: &mut CudaBf16Aliases,
        precision: DecoderPrecision,
    ) -> Result<Self, String> {
        config.validate()?;
        // Typed handles are consumed here, without erasure, legacy fallback,
        // upload, F16 conversion, hash-join lookup, or whole-model enumeration.
        let conv: Blob<NativeTensor<BF16, 3>> = snapshot.get(slots.conv).map_err(|e| e.to_string())?;
        let a_log: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.a_log).map_err(|e| e.to_string())?;
        let dt_bias: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.dt_bias).map_err(|e| e.to_string())?;
        let norm: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.norm).map_err(|e| e.to_string())?;
        // SAFETY: the caller guarantees the genuine snapshot's immutable
        // file-backed prefix for each exact native leaf, including page prefix.
        let result = unsafe {
            Self {
                config,
                qkv: bind_linear(snapshot, slots.qkv, aliases, precision)?,
                z: bind_linear(snapshot, slots.z, aliases, precision)?,
                a: bind_linear(snapshot, slots.a, aliases, precision)?,
                b: bind_linear(snapshot, slots.b, aliases, precision)?,
                out: bind_linear(snapshot, slots.out, aliases, precision)?,
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
        let anchor = &self.conv;
        for (name, linear, shape) in [
            ("qkv", &self.qkv, [c.channels(), c.hidden]),
            ("z", &self.z, [c.value_width(), c.hidden]),
            ("a", &self.a, [c.value_heads, c.hidden]),
            ("b", &self.b, [c.value_heads, c.hidden]),
            ("out", &self.out, [c.hidden, c.value_width()]),
        ] {
            linear.check(name, shape, anchor)?;
        }
        for (name, tensor, shape) in [
            ("conv", &self.conv, vec![c.channels(), 1, c.conv_kernel]),
            ("A_log", &self.a_log, vec![c.value_heads]),
            ("dt_bias", &self.dt_bias, vec![c.value_heads]),
            ("norm", &self.norm, vec![c.value_dim]),
        ] {
            check(tensor, anchor, name, &shape, DType::BF16)?;
            if tensor.handle.can_mut() {
                return Err(format!("{name}: read-only pile alias required"));
            }
        }
        Ok(())
    }

    pub(crate) fn projections(&self) -> [&Linear; 5] {
        [&self.qkv, &self.z, &self.a, &self.b, &self.out]
    }

    pub(crate) fn projections_mut(&mut self) -> [&mut Linear; 5] {
        [&mut self.qkv, &mut self.z, &mut self.a, &mut self.b, &mut self.out]
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
        check(hidden, &self.conv, "hidden", &[batch, tokens, c.hidden], DType::BF16)?;
        // Check ALL derived allocations and incoming state before any launch.
        count(&[batch, tokens, c.channels()])?;
        count(&[batch, c.channels(), c.conv_kernel])?;
        count(&[batch, c.value_heads, c.key_dim, c.value_dim])?;
        if let Some(s) = state {
            check(&s.conv, hidden, "conv state", &[batch, c.channels(), c.conv_kernel], DType::BF16)?;
            check(&s.recurrent, hidden, "recurrent state", &[batch, c.value_heads, c.key_dim, c.value_dim], DType::F32)?;
        }
        let qkv = project_linear(hidden, &self.qkv)?;
        let z = reshape(project_linear(hidden, &self.z)?, &[batch, tokens, c.value_heads, c.value_dim]);
        let a = project_linear(hidden, &self.a)?;
        let b = project_linear(hidden, &self.b)?;
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
        let output = project_linear(&gated, &self.out)?;
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

/// Resident BF16 linear projection: [B,T,I] x [O,I] -> [B,T,O].
/// K/N-only dispatch; no CPU fallback, autotuning, or split-K reduction.
pub fn project(input: &CudaTensor, weight: &CudaTensor) -> Result<CudaTensor, String> {
    project_impl(input, weight, false)
}

/// How the decoder's linear projections compute. Only the decoder's linear
/// layers change; the vision tower, norms, convolution, DeltaNet scan,
/// attention scores and the token table are BF16 in every arm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum DecoderPrecision {
    /// The fixed BF16 MMA blueprint above.
    #[default]
    Bf16,
    /// NVFP4 weights and NVFP4 activations (E2M1, E4M3 per 16, and an f32
    /// per activation row) into the block-scaled `mma.sync ... kind::mxf4nvf4`
    /// instruction, f32 accumulation (`Nvfp4::project_w4a4_rows`).
    W4a4,
    /// NVFP4 weights decoded to BF16 in registers for the BF16 `m16n8k16`
    /// MMA, BF16 activations (`Nvfp4::project`, `PromptLane::W4a16`).
    W4a16,
}

impl DecoderPrecision {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "w4a4" => Ok(Self::W4a4),
            "w4a16" => Ok(Self::W4a16),
            other => Err(format!("decoder precision must be bf16, w4a4 or w4a16, not {other:?}")),
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::W4a4 => "w4a4",
            Self::W4a16 => "w4a16",
        }
    }
}

/// What one decoder projection runs, as it was actually bound: the identity
/// of an embedding depends on it, so a caller reads it here rather than
/// trusting the precision it asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundProjection {
    /// The BF16 leaf, through the fixed BF16 MMA blueprint.
    Bf16,
    /// NVFP4 planes run on `precision`. `stored` when the planes are the
    /// root's own NVFP4 leaf, uploaded unchanged; false when this process
    /// quantized them from the root's BF16 leaf.
    Nvfp4 { precision: DecoderPrecision, stored: bool },
}

/// One decoder linear weight `[O, I]` and the arithmetic it runs on.
pub(crate) enum Linear {
    Bf16(CudaTensor),
    Nvfp4 { weight: Nvfp4, precision: DecoderPrecision, stored: bool },
}

impl Linear {
    /// The bound BF16 weight; refused once quantized.
    pub(crate) fn bf16(&self) -> Result<&CudaTensor, String> {
        match self {
            Self::Bf16(t) => Ok(t),
            Self::Nvfp4 { .. } => Err("projection is NVFP4, not BF16".into()),
        }
    }

    /// An NVFP4 weight already on the device: the root's stored leaf
    /// (`stored`), or quantized here from its BF16 leaf.
    pub(crate) fn nvfp4(weight: Nvfp4, precision: DecoderPrecision, stored: bool) -> Result<Self, String> {
        if precision == DecoderPrecision::Bf16 {
            return Err("an NVFP4 weight needs an NVFP4 precision".into());
        }
        Ok(Self::Nvfp4 { weight, precision, stored })
    }

    /// What this projection runs, as bound.
    pub(crate) fn bound(&self) -> BoundProjection {
        match self {
            Self::Bf16(_) => BoundProjection::Bf16,
            Self::Nvfp4 { precision, stored, .. } => BoundProjection::Nvfp4 { precision: *precision, stored: *stored },
        }
    }

    /// (BF16 bytes, NVFP4 plane bytes) this projection holds.
    pub(crate) fn bytes(&self) -> (u64, u64) {
        match self {
            Self::Bf16(t) => (t.meta.shape().as_slice().iter().product::<usize>() as u64 * 2, 0),
            Self::Nvfp4 { weight, .. } => (0, weight.bytes()),
        }
    }

    /// Shape `[O, I]` and device check; a BF16 weight must also be an
    /// immutable contiguous binding.
    pub(crate) fn check(&self, name: &str, shape: [usize; 2], like: &CudaTensor) -> Result<(), String> {
        match self {
            Self::Bf16(t) => {
                check(t, like, name, &shape, DType::BF16)?;
                if t.handle.can_mut() {
                    return Err(format!("{name}: read-only pile alias required"));
                }
                Ok(())
            }
            Self::Nvfp4 { weight, .. } => {
                if [weight.n(), weight.k()] != shape || weight.device() != &like.device {
                    return Err(format!("{name}: NVFP4 weight is [{}, {}], wants {shape:?} on the same device", weight.n(), weight.k()));
                }
                Ok(())
            }
        }
    }

    /// This BF16 weight quantized on the device with the NVFP4 recipe of
    /// `nn::nvfp4_gemv` (the same codes `models_import` gates stored
    /// projections against), as a new projection on `precision`; this one is
    /// left as it is. `None` for `Bf16`.
    pub(crate) fn quantized(&self, precision: DecoderPrecision) -> Result<Option<Self>, String> {
        if precision == DecoderPrecision::Bf16 {
            return Ok(None);
        }
        let weight = Nvfp4::quantize(self.bf16()?).map_err(|e| format!("NVFP4 quantize: {e}"))?;
        Self::nvfp4(weight, precision, false).map(Some)
    }
}

/// Bind one decoder projection for `precision`. BF16 binds the leaf through
/// the caller's binder (aliased or copied, as the binder chooses). An NVFP4
/// precision copies the BF16 payload into a transient device buffer,
/// quantizes it there with the `nn::nvfp4_gemv` recipe (the codes a stored
/// NVFP4 leaf is gated against), and drops the transient buffer: only the
/// code and scale planes stay resident, and the binder registers nothing.
///
/// # Safety
/// The binder's genuine immutable pile-prefix premise for the BF16 path.
pub(crate) unsafe fn bind_linear<R: BlobStoreGet>(
    snapshot: &R,
    slot: LinearSlot,
    aliases: &mut CudaBf16Aliases,
    precision: DecoderPrecision,
) -> Result<Linear, String> {
    let slot = match slot {
        LinearSlot::Bf16(slot) => slot,
        LinearSlot::Nvfp4(slot) => {
            // A stored NVFP4 projection: its code and scale planes are copied
            // into device memory once; nothing is converted or quantized.
            if precision == DecoderPrecision::Bf16 {
                return Err("this projection is stored as NVFP4; bind it with an NVFP4 precision".into());
            }
            let blob: Blob<NativeTensor<triblespace::core::blob::encodings::tensor::elements::NVFP4, 2>> =
                snapshot.get(slot).map_err(|e| e.to_string())?;
            let view = crate::leaf::read_leaf(blob).map_err(|e| e.to_string())?;
            let dims: [u64; 2] = view.dims().try_into().map_err(|_| "NVFP4 projection is not rank 2")?;
            let weight = Nvfp4::upload(view.payload(), dims, DType::BF16, aliases.device())
                .map_err(|e| format!("NVFP4 upload: {e}"))?;
            let client = CudaRuntime::client(aliases.device());
            for _ in 0..2 {
                client.flush().map_err(|e| format!("flush after an NVFP4 upload: {e:?}"))?;
            }
            return Linear::nvfp4(weight, precision, true);
        }
    };
    let blob: Blob<NativeTensor<BF16, 2>> = snapshot.get(slot).map_err(|e| e.to_string())?;
    if precision == DecoderPrecision::Bf16 {
        // SAFETY: forwarded genuine immutable-prefix contract.
        return Ok(Linear::Bf16(unsafe { aliases.bind_pile_leaf(blob)? }));
    }
    let shape = crate::nn::cuda_bf16_alias::checked_shape(&blob)?;
    let view = crate::leaf::read_leaf(blob).map_err(|e| e.to_string())?;
    let device = aliases.device().clone();
    let client = CudaRuntime::client(&device);
    let staged = client.create_from_slice(view.payload().as_ref());
    let source = CubeTensor::new_contiguous(client.clone(), device, shape.as_slice().into(), staged, DType::BF16);
    // Quantize reads the tensor amax back, which orders the upload before it.
    let weight = Nvfp4::quantize(&source).map_err(|e| format!("NVFP4 quantize at load: {e}"))?;
    drop(source);
    // Free the upload's host staging and the transient device copy now (see
    // CudaBf16Aliases' DeviceCopy note: two flushes release the host copy).
    for _ in 0..2 {
        client.flush().map_err(|e| format!("flush after an NVFP4 load: {e:?}"))?;
    }
    Linear::nvfp4(weight, precision, false)
}

/// Pack one BF16 decoder projection `[n, k]` (its native leaf payload) for
/// storage as a `Tensor<NVFP4, 2>` leaf: the host packing
/// (`nvfp4::pack_nearest`), required to equal the device quantizer that
/// [`bind_linear`] runs at load, code byte for code byte, scale byte for
/// scale byte and global scale bit for bit. A stored root therefore runs
/// exactly the weights a load-time quantization of the BF16 root runs.
pub fn pack_projection_for_storage(
    name: &str,
    payload: &[u8],
    dims: [u64; 2],
    device: &cubecl::cuda::CudaDevice,
) -> Result<crate::nvfp4::StoredProjection, String> {
    let (n, k) = (dims[0] as usize, dims[1] as usize);
    if payload.len() != n * k * 2 {
        return Err(format!("{name}: {} payload bytes for BF16 [{n}, {k}]", payload.len()));
    }
    let client = CudaRuntime::client(device);
    let source = CubeTensor::new_contiguous(client.clone(), device.clone(), [n, k].as_slice().into(),
        client.create_from_slice(payload), DType::BF16);
    let weight = Nvfp4::quantize(&source).map_err(|e| format!("{name}: device quantize: {e}"))?;
    drop(source);
    let values: Vec<f32> = payload.chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()).collect();
    weight.pack_matching(name, &values).map_err(|e| e.to_string())
}

/// Whether a checkpoint tensor name is one of the decoder's linear
/// projections that the NVFP4 precisions replace (all 32 layers' GDN mixer,
/// attention and MLP projections; never the vision tower, norms, conv or the
/// token table).
pub fn is_decoder_projection(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else { return false };
    let Some((layer, role)) = rest.split_once('.') else { return false };
    layer.parse::<usize>().is_ok_and(|l| l < 32) && matches!(role,
        "linear_attn.in_proj_qkv.weight" | "linear_attn.in_proj_z.weight" | "linear_attn.in_proj_a.weight"
        | "linear_attn.in_proj_b.weight" | "linear_attn.out_proj.weight"
        | "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" | "self_attn.o_proj.weight"
        | "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight")
}

/// Whether a checkpoint tensor name is one of the 83 vision tower linears an
/// NVFP4 vision precision runs: every one of the 27 blocks' `attn.qkv`,
/// `attn.proj` and `mlp.linear_fc1` weights and the merger's `linear_fc1` and
/// `linear_fc2` weights. Each block's `mlp.linear_fc2` (K 4304, which no
/// 64- or 32-wide NVFP4 step divides), the patch convolution, the position
/// table, norms and biases are not. `models_import wemm --vision-nvfp4`
/// stores exactly these as NVFP4 leaves.
pub fn is_vision_nvfp4_linear(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("model.visual.blocks.") {
        let Some((block, role)) = rest.split_once('.') else { return false };
        return block.parse::<usize>().is_ok_and(|b| b < 27 && b.to_string() == block)
            && matches!(role, "attn.qkv.weight" | "attn.proj.weight" | "mlp.linear_fc1.weight");
    }
    matches!(name, "model.visual.merger.linear_fc1.weight" | "model.visual.merger.linear_fc2.weight")
}

/// [`project`] for any decoder [`Linear`]: `[B,T,I] x [O,I]^T -> [B,T,O]`.
/// An NVFP4 weight takes the B*T rows as one activation matrix; every row is
/// computed independently of the others, as in the BF16 blueprint.
pub(crate) fn project_linear(input: &CudaTensor, weight: &Linear) -> Result<CudaTensor, String> {
    let (weight, precision) = match weight {
        Linear::Bf16(t) => return project(input, t),
        Linear::Nvfp4 { weight, precision, .. } => (weight, *precision),
    };
    let s = input.meta.shape().as_slice().to_vec();
    if s.len() != 3 || s[2] != weight.k() {
        return Err("NVFP4 projection requires [B,T,I] with I the weight's input width".into());
    }
    check(input, input, "projection input", &s, DType::BF16)?;
    let rows = count(&[s[0], s[1]])?;
    let flat = reshape(input.clone(), &[rows, s[2]]);
    let out = match precision {
        DecoderPrecision::W4a4 => weight.project_w4a4_rows(&flat),
        DecoderPrecision::W4a16 => weight.project(&flat, crate::nvfp4::PromptLane::W4a16),
        DecoderPrecision::Bf16 => return Err("BF16 precision on an NVFP4 weight".into()),
    }
    .map_err(|e| format!("NVFP4 {} projection [{rows},{}] x [{},{}]: {e}", precision.label(), s[2], weight.n(), weight.k()))?;
    Ok(reshape(out, &[s[0], s[1], weight.n()]))
}

fn project_impl(input: &CudaTensor, weight: &CudaTensor, serial: bool) -> Result<CudaTensor, String> {
    let s = input.meta.shape().as_slice();
    let w = weight.meta.shape().as_slice();
    if s.len() != 3 || w.len() != 2 || s[2] != w[1] {
        return Err("projection requires [B,T,I] and [O,I]".into());
    }
    let ni = check(input, input, "projection input", s, DType::BF16)?;
    let nw = check(weight, input, "projection weight", w, DType::BF16)?;
    let shape = [s[0], s[1], w[0]];
    let no = count(&shape)?;
    if !serial && fixed_mma_shape(s[2], w[0]) {
        return project_mma(input, weight, shape, no);
    }
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

fn fixed_mma_shape(k: usize, n: usize) -> bool {
    k > 0 && n > 0 && k.is_multiple_of(16) && n.is_multiple_of(8)
}

fn projection_elems() -> cubek::matmul::definition::MatmulGlobalElems {
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    let bf16 = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
    cubek::matmul::definition::MatmulGlobalElems {
        lhs: bf16,
        rhs: bf16,
        out: StorageType::Scalar(ElemType::Float(FloatKind::F32)),
    }
}

/// Dependency trace (cubek-matmul 0.2.0): Strategy::SimpleCyclicMma passes
/// Forced through stamp_kind unchanged; SimpleAlgorithm::expand_blueprint
/// clones Forced directly, never entering either M-dependent infer function.
/// launch_tiling chooses IO vector widths from contiguous K/K/N and strides
/// K/K/N, not M. Rank-2 row/column-major bindings are not materialized copies.
/// The row-major global grid owns disjoint M/N partitions; SimpleMatmul walks
/// K stages within each partition, with no atomic or split-K output reduction.
/// Bounds checks are always enabled, even for complete tiles, so crossing an
/// M tile boundary does not switch to a different arithmetic configuration.
fn projection_blueprint(m: usize, n: usize, k: usize) -> Result<cubek::matmul::definition::TilingBlueprint, String> {
    use cubek::matmul::{
        components::{stage::PartitionBuffering, tile::TileMatmulKind},
        definition::{MatmulProblem, TilingBlueprint, TilingScheme},
    };
    use cubek::std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};
    let problem = MatmulProblem::from_shapes_and_strides(
        [m, k].into(), [k, n].into(), [m, n].into(),
        [k, 1].into(), [1, k].into(), [n, 1].into(),
        projection_elems(), cubecl::ir::AddressType::U32, None, None,
    ).map_err(|e| format!("fixed projection problem: {e:?}"))?;
    // The instruction tile (16x8x16) and so the K accumulation order are the
    // same for every shape; only how many output tiles a plane and a cube own
    // differs, which leaves every output bit unchanged (wemm_timing gemm:
    // 0 differing BF16 outputs on all six decoder shapes). Wide weights take
    // 64x64 per plane-row group: 2x8 tiles per plane, four planes stacked in
    // M, 28 against 14 TFLOP/s at M=248 on a GB10. The router-sized a/b
    // projections (N=32) are faster under the original narrow tile.
    let (partition, stage) = if n >= 1024 { ((2, 8, 2), (4, 1, 1)) } else { ((2, 1, 4), (2, 2, 1)) };
    let scheme = TilingScheme::builder()
        .with_tile_size((16, 8, 16).into())
        .with_partition_size(partition.into())
        .with_stage_size(stage.into())
        .build().map_err(str::to_owned)?;
    let grid = HypercubeBlueprint::builder()
        .global_order(GlobalOrder::RowMajor)
        .cube_count_strategy(CubeCountStrategy::FromProblem)
        .build();
    let mut blueprint = TilingBlueprint::builder(TileMatmulKind::Mma, scheme, 32, &problem)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_blueprint(grid)
        .build();
    blueprint.check_m_bounds = true;
    blueprint.check_n_bounds = true;
    blueprint.check_k_bounds = true;
    Ok(blueprint)
}

#[cube(launch_unchecked)]
fn projection_round(input: &Array<f32>, output: &mut Array<bf16>, n: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n { output[i] = bf16::cast_from(input[i]); }
}

// Called only after the public descriptor/device/extent checks. Weight storage
// remains the original aliased BF16 [N,K] handle: [K,N] strides [1,K] is a view,
// not a transpose/cast/copy. F32 is accumulator/output scratch, never weights.
fn project_mma(input: &CudaTensor, weight: &CudaTensor, shape: [usize; 3], no: usize) -> Result<CudaTensor, String> {
    use cubek::matmul::{definition::MatmulElems, launch::Strategy, routines::BlueprintStrategy};
    use cubek::std::InputBinding;
    fn binding(handle: &cubecl::server::Handle, shape: [usize; 2], strides: [usize; 2]) -> TensorBinding<CudaRuntime> {
        TensorBinding {
            handle: handle.clone().binding(), shape: shape.into(), strides: strides.into(),
            runtime: core::marker::PhantomData,
        }
    }
    let (m, n, k) = (shape[0] * shape[1], shape[2], input.meta.shape()[2]);
    let globals = projection_elems();
    let mut dtypes = MatmulElems::from_globals(&globals);
    let scratch = input.client.empty(no.checked_mul(4).ok_or("F32 output byte extent overflow")?);
    let strategy = Strategy::SimpleCyclicMma(BlueprintStrategy::Forced(projection_blueprint(m, n, k)?));
    cubek::matmul::launch::launch_ref(
        &strategy, &input.client,
        InputBinding::new(binding(&input.handle, [m, k], [k, 1]), globals.lhs),
        InputBinding::new(binding(&weight.handle, [k, n], [1, k]), globals.rhs),
        binding(&scratch, [m, n], [n, 1]), &mut dtypes,
    ).map_err(|e| format!("fixed BF16 projection [{m},{k}] x [{n},{k}]^T: {e:?}"))?;
    // Setup errors are not retried through another algorithm: that could make
    // arithmetic depend on resources or M. Unsupported K/N is selected above.
    let output = input.client.empty(no * 2);
    let cube = CubeDim::new_1d(128);
    // SAFETY: both handles own `no` elements, input F32/output BF16, one writer.
    unsafe {
        projection_round::launch_unchecked::<CudaRuntime>(
            &input.client, cubecl::calculate_cube_count_elemwise(&input.client, no, cube), cube,
            ArrayArg::from_raw_parts(scratch, no), ArrayArg::from_raw_parts(output.clone(), no), no,
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

#[cfg(test)]
mod projection_tests {
    use super::*;
    use cubecl::cuda::CudaDevice;

    #[test]
    fn forced_projection_blueprint_is_independent_of_rows_and_tails() {
        let expected = projection_blueprint(1, 40, 48).unwrap();
        for m in [2, 15, 16, 17, 63, 64, 65, 116, 130, 256] {
            assert_eq!(projection_blueprint(m, 40, 48).unwrap(), expected);
        }
        assert!(expected.check_m_bounds && expected.check_n_bounds && expected.check_k_bounds);
        assert_eq!(expected.tiling_scheme.elements_per_stage_along_m(), 64);
        assert_eq!(expected.tiling_scheme.elements_per_stage_along_n(), 16);
        assert_eq!(expected.tiling_scheme.elements_per_stage_along_k(), 64);
        let wide = projection_blueprint(1, 4096, 4096).unwrap();
        for m in [2, 17, 64, 130, 248, 256] {
            assert_eq!(projection_blueprint(m, 4096, 4096).unwrap(), wide);
        }
        assert_eq!(wide.tiling_scheme.elements_per_stage_along_m(), 128);
        assert_eq!(wide.tiling_scheme.elements_per_stage_along_n(), 64);
        assert_eq!(wide.tiling_scheme.elements_per_stage_along_k(), 32);
    }

    #[test]
    fn projection_shape_dispatch_is_explicit_and_accumulation_is_f32() {
        for (k, n) in [(16, 8), (48, 40), (4096, 32), (4096, 12288), (12288, 4096)] {
            assert!(fixed_mma_shape(k, n));
        }
        for (k, n) in [(0, 8), (16, 0), (7, 5), (17, 8), (16, 9)] {
            assert!(!fixed_mma_shape(k, n));
        }
        let dtypes = cubek::matmul::definition::MatmulElems::from_globals(&projection_elems());
        assert_eq!(dtypes.acc_stage, projection_elems().out);
        assert_eq!(dtypes.acc_register, projection_elems().out);
        assert_eq!(dtypes.lhs_register, projection_elems().lhs);
        assert_eq!(dtypes.rhs_register, projection_elems().rhs);
    }

    #[cube(launch_unchecked)]
    fn fixture_values(out: &mut Array<bf16>, rows: &Array<u32>, n: usize, width: usize) {
        let i = ABSOLUTE_POS as usize;
        if i < n {
            // Numeric fixture generation is GPU-only. Host rows are integer
            // identities, not activation or weight payloads.
            let row = rows[i / width] as usize;
            let phase = f32::cast_from(((i % width) * 13 + row * 7) % 251) / 32.0f32;
            out[i] = bf16::cast_from(phase.sin() * 0.125f32);
        }
    }

    fn fixture(shape: &[usize], rows: &[u32]) -> CudaTensor {
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let n = count(shape).unwrap();
        let width = *shape.last().unwrap();
        assert_eq!(rows.len() * width, n);
        let row_bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        let row_handle = client.create_from_slice(&row_bytes);
        let handle = client.empty(n * 2);
        let cube = CubeDim::new_1d(128);
        // SAFETY: integer row table has one entry per output row, exact sizes.
        unsafe {
            fixture_values::launch_unchecked::<CudaRuntime>(
                &client, cubecl::calculate_cube_count_elemwise(&client, n, cube), cube,
                ArrayArg::from_raw_parts(handle.clone(), n),
                ArrayArg::from_raw_parts(row_handle, rows.len()), n, width,
            );
        }
        CubeTensor::new_contiguous(client, device, shape.into(), handle, DType::BF16)
    }

    fn bytes(tensor: CudaTensor) -> Vec<u8> {
        tensor.client.read_one(tensor.handle).unwrap().to_vec()
    }

    fn compare_gpu_oracle(input: &CudaTensor, weight: &CudaTensor) -> Vec<u8> {
        let actual = bytes(project(input, weight).unwrap());
        assert_eq!(actual, bytes(project(input, weight).unwrap()), "fresh-call byte identity");
        let oracle = bytes(project_impl(input, weight, true).unwrap());
        let mut differing = 0usize;
        let mut worst = 0.0f32;
        let mut outside = 0usize;
        for (a, b) in actual.chunks_exact(2).zip(oracle.chunks_exact(2)) {
            differing += usize::from(a != b);
            let a = bf16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
            let b = bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
            assert!(a.is_finite() && b.is_finite());
            // Readback diagnostics only: the reference dot products ran on
            // GPU. This fixture's fixed budget is not a model error envelope,
            // and exact equality to the old serial arithmetic is not required.
            let scaled = (a - b).abs() / (0.001 + 0.01 * b.abs());
            worst = worst.max(scaled);
            outside += usize::from(scaled > 1.0);
        }
        eprintln!("projection {:?} x {:?}: serial differing_words={differing}, worst_scaled={worst}, outside={outside}", input.meta.shape(), weight.meta.shape());
        assert_eq!(outside, 0, "fixed synthetic GPU-oracle budget");
        actual
    }

    #[test]
    #[ignore = "reserved CUDA: fixed projection layout/repeat/batch experiment"]
    fn fixed_projection_matches_gpu_oracle_and_preserves_row_bits() {
        // Nonsquare transposed weight, N/K stage tails, and the explicit small
        // unaligned GPU fallback. M crosses both tile and stage boundaries.
        for (k, n) in [(48, 40), (7, 5)] {
            let weight = fixture(&[n, k], &(0..n as u32).collect::<Vec<_>>());
            let single = fixture(&[1, 1, k], &[777]);
            let wanted = compare_gpu_oracle(&single, &weight);
            for m in [17, 64, 65, 130] {
                let mut rows: Vec<u32> = (0..m as u32).collect();
                for position in [0, 15, m - 1] {
                    rows[position] = 777;
                    let input = fixture(&[1, m, k], &rows);
                    let output = compare_gpu_oracle(&input, &weight);
                    assert_eq!(&output[position * n * 2..(position + 1) * n * 2], &wanted);
                    rows.reverse();
                    let reversed = fixture(&[1, m, k], &rows);
                    let reversed = bytes(project(&reversed, &weight).unwrap());
                    for row in 0..m {
                        assert_eq!(&output[row * n * 2..(row + 1) * n * 2], &reversed[(m - 1 - row) * n * 2..(m - row) * n * 2]);
                    }
                    rows.reverse();
                }
            }
            let rows: Vec<u32> = (0..130).collect();
            let a = fixture(&[1, 130, k], &rows);
            // Reinterpret the SAME GPU bytes, no activation copy, with another
            // valid batch/token factorization.
            let b = reshape(a.clone(), &[2, 65, k]);
            assert_eq!(bytes(project(&a, &weight).unwrap()), bytes(project(&b, &weight).unwrap()));
        }
    }

    #[test]
    #[ignore = "reserved CUDA: finite real WeMM projection dimensions, not model retrieval"]
    fn fixed_projection_real_dimensions_match_gpu_oracle() {
        for (k, n, m) in [(4096, 32, 65), (4096, 4096, 2), (4096, 8192, 2), (4096, 12288, 2), (12288, 4096, 2)] {
            let weight = fixture(&[n, k], &(0..n as u32).collect::<Vec<_>>());
            let rows: Vec<u32> = (0..m as u32).collect();
            let input = fixture(&[1, m, k], &rows);
            let output = compare_gpu_oracle(&input, &weight);
            let single = fixture(&[1, 1, k], &[rows[m - 1]]);
            assert_eq!(&output[(m - 1) * n * 2..m * n * 2], &bytes(project(&single, &weight).unwrap()));
        }
    }
}
