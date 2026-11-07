//! B1 native IDs -> last-row logits. No sampling, codec, tool execution or CPU
//! inference. A completed chunk is the only publication boundary. A failed
//! append is NOT rolled back: all component state becomes unusable until reset.
//!
//! reset() releases retained operands only after successful fences; a sync Err
//! is not a drain. Normal drop returns owned allocations to the stream-ordered
//! runtime, not to an external host allocator. Internal
//! kernel scratch follows CubeCL's managed stream allocator, as in the existing
//! components. This API does not promise recovery from a lost CUDA context or
//! turn dropping the owner into a reliable device drain. No mmap/external
//! aliases or bypass-runtime CUDA launches are used by this assembly.

use super::{
    attention,
    fp8::{CudaTensor, Projection},
    ingress::config::{LayerKind, TextConfig},
    mamba2::{contract, gpu},
    moe,
};
use crate::model_collection::ModelSnapshot;
use burn::tensor::DType;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
    stream_id::StreamId,
};
use half::bf16;
use std::sync::Arc;
use triblespace::{core::collection::CollectionHandle, prelude::*};

#[path = "session_source.rs"]
mod source;

#[path = "session_prefix.rs"]
mod prefix;
pub use prefix::{Prefix, PrefixAllocation, PrefixOwnership, prefix_admission};
#[path = "session_checkpoint.rs"]
mod checkpoint;
pub use checkpoint::{PrefixCheckpoint, PrefixExecution, PrefixProvenance, PrefixLayer, KvChunk};
#[path = "session_checkpoint_manifest.rs"]
pub mod checkpoint_manifest;
#[path = "session_prefill.rs"]
mod prefill;
pub use prefill::{PrefillAllocation, PrefillOutcome, admission as prefill_admission};

type Result<T> = std::result::Result<T, String>;
const H: usize = 4096;
pub const MAX_CHUNK: usize = 256;

/// Explicit caller-selected observation coordinates. Collection is provenance,
/// not an extra authorization check; the held ModelSnapshot carries support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    pub collection: CollectionHandle,
    pub root: Id,
    pub config: Id,
    pub tokenizer_binding: Id,
    pub tokenizer_base: Id,
    pub pre_tokenizer: Id,
}
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub capacity: usize,
    /// Fixed weights including every routed expert of every MoE layer, KV,
    /// and one committed Mamba state per Mamba layer. Logical payload bytes,
    /// not allocator reservation or host staging; loading separately checks
    /// the device's actual free memory.
    pub max_resident_bytes: usize,
    /// Per-call logical transient allowance: short append OR an entire known
    /// prefill span, including its full-span buffers and tile workspace.
    /// NOT physical peak,
    /// host staging, allocator/JIT overhead or a whole-device reservation.
    pub max_chunk_bytes: usize,
}

/// CPU-only logical admission for a cold prompt. These payload totals exclude
/// host preparation/staging, allocator reservation, runtime and JIT overhead;
/// they are neither a physical peak estimate nor a whole-device reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    /// Immutable weights/KV plus one committed state per Mamba layer,
    /// excluding the expert bank.
    pub base_resident_bytes: usize,
    /// Resident experts per MoE layer: all of them (512), or 0 without MoE.
    /// The name predates the bank; it is kept for existing readers.
    pub expert_cache_pairs_per_moe: usize,
    /// The whole expert bank, every pair of every MoE layer, counted once.
    pub expert_cache_bytes: usize,
    /// base_resident_bytes + expert_cache_bytes.
    pub resident_bytes: usize,
    /// Actual ordinary prompt path, including full-span slabs when >256.
    pub prefill: PrefillAllocation,
    /// Short-append comparison only; a long prompt uses prefill above.
    pub first_chunk_tokens: usize,
    pub first_chunk_bytes: usize,
    pub max_prompt_chunk_bytes: usize,
    /// Exact largest logical chunk within capacity, independent of this
    /// prompt. Informational: only the actual prefill path is admitted.
    /// Future appends still enforce max_chunk_bytes before dispatch.
    pub max_capacity_chunk_bytes: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ready,
    Poisoned,
}
/// These logits predict AFTER committed_ids, not a generated/attended token.
pub struct NextLogits<'a> {
    pub committed_len: usize,
    pub values: &'a [f32],
}

struct MambaWeights {
    config: contract::Config,
    input: Projection,
    output: Projection,
    conv: CudaTensor,
    bias: Option<CudaTensor>,
    a_log: CudaTensor,
    dt: CudaTensor,
    d: CudaTensor,
    gain: CudaTensor,
}
struct Mamba {
    weights: Arc<MambaWeights>,
    committed: Option<Arc<gpu::State<CudaRuntime>>>,
    staged: Option<gpu::State<CudaRuntime>>,
}
enum Mixer {
    Mamba(Mamba),
    Attention(attention::Attention),
    Moe(moe::Moe),
}
struct Layer {
    id: Id,
    norm: CudaTensor,
    mixer: Mixer,
}

enum MixerWeights {
    Mamba(Arc<MambaWeights>),
    Attention(Arc<attention::Weights>),
    Moe(Arc<moe::Weights>),
}
struct LayerWeights {
    id: Id,
    norm: CudaTensor,
    mixer: MixerWeights,
}

/// Immutable uploaded operands, every MoE layer's whole expert bank included,
/// and their exact held source observation. Shared only within this
/// process/device. All sequence state belongs to Sessions. Construction observes successful upload completion on
/// the origin stream before this owner can escape. This is not a certificate of
/// future device health. No raw mutable weight handles are exposed.
pub struct DeviceModel<S> {
    snapshot: Arc<ModelSnapshot<S>>,
    source: Source,
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    config: TextConfig,
    embedding: CudaTensor,
    final_norm: CudaTensor,
    head: Projection,
    layers: Vec<LayerWeights>,
}

pub struct Session<S> {
    model: Arc<DeviceModel<S>>,
    // ComputeClient clones resolve the current stream dynamically. Keep every
    // append/reset (including its fence and unwind) on the construction stream,
    // even when this Send owner moves to another thread. Sessions constructed
    // on one thread share its stream; this is not a new concurrent scheduler.
    stream: StreamId,
    limits: Limits,
    layers: Vec<Layer>,
    status: Status,
    committed: Vec<u32>,
    logits: Vec<f32>,
    // Ordinary owning handles, never borrowed raw pointers. Keep outer values
    // across an uncertain append/reset. Component-private pending owners stay
    // inside their respective layers; Mamba staged states are installed early.
    pending: Vec<CudaTensor>,
    // One explicit checkpoint, not an indexed cache of past observations.
    seed: Option<Arc<[u32]>>,
    prefix: Option<Arc<Prefix<S>>>,
    // Span-wide owners survive an error/unwind independently of the current
    // tile. A successful fence retires tile scratch, never these live slabs.
    span_pending: Option<prefill::Pending>,
    #[cfg(test)]
    fault: Option<Fault>,
}

// Control injection except PrefixUpload, which queues the same checked CUDA
// descriptor rejection as the existing origin-upload fixture (no invalid copy).
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    AfterLayer(usize),
    AfterCommit(usize),
    Reset,
    PanicAfterLayer(usize),
    PrefixUpload,
}
#[cfg(test)]
fn inject(fault: Option<Fault>, point: Fault) -> Result<()> {
    if fault == Some(point) {
        return Err("injected session control failure".into());
    }
    Ok(())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or("session byte sum overflow".into())
}
fn bytes(dims: &[usize], width: usize) -> Result<usize> {
    dims.iter()
        .try_fold(width, |n, &d| {
            n.checked_mul(d)
                .filter(|&n| d > 0 && n <= u32::MAX as usize)
        })
        .ok_or("session empty/overflowing device extent".into())
}
fn mamba_config(c: &TextConfig) -> contract::Config {
    contract::Config {
        heads: c.mamba_num_heads,
        head_dim: c.mamba_head_dim,
        groups: c.n_groups,
        state_dim: c.ssm_state_size,
        conv_kernel: c.conv_kernel,
        norm_epsilon: c.layer_norm_epsilon,
        time_step_min: 0.0,
    } // Native fast policy, not the slow reference floor.
}
fn attention_limits(c: &TextConfig, l: Limits) -> attention::Limits {
    attention::Limits {
        context_limit: c.max_position_embeddings,
        capacity: l.capacity,
        max_resident_bytes: l.max_resident_bytes,
        max_working_bytes: l.max_chunk_bytes,
    }
}
fn moe_limits(l: Limits) -> moe::Limits {
    moe::Limits {
        max_resident_bytes: l.max_resident_bytes,
        max_working_bytes: l.max_chunk_bytes,
    }
}
fn validate_config(c: &TextConfig, l: Limits) -> Result<()> {
    if c.model_type != "nemotron_h"
        || c.hidden_size != H
        || c.vocab_size == 0
        || c.vocab_size % 8 != 0
        || c.layers_block_type.is_empty()
        || l.capacity == 0
        || l.capacity > c.max_position_embeddings
        || l.capacity > u32::MAX as usize
        || c.residual_in_fp32
        || c.tie_word_embeddings
        || c.attention_bias
        || c.sliding_window.is_some()
        || (c.num_attention_heads, c.num_key_value_heads, c.head_dim) != (32, 2, 128)
        || !c.layer_norm_epsilon.is_finite()
        || c.layer_norm_epsilon as f32 <= 0.0
        || !(c.layer_norm_epsilon as f32).is_finite()
        || c.mamba_hidden_act != "silu"
        || c.mamba_proj_bias
        || c.use_bias
        || c.time_step_limit[0].0 != 0.0
        || c.time_step_limit[1].0 != f64::INFINITY
        || c.mlp_hidden_act != "relu2"
        || c.mlp_bias
        || c.moe_shared_expert_overlap
        || (
            c.moe_latent_size,
            c.moe_intermediate_size,
            c.moe_shared_expert_intermediate_size,
        ) != (1024, 2688, 5376)
        || (
            c.n_routed_experts,
            c.n_shared_experts,
            c.num_experts_per_tok,
            c.n_group,
            c.topk_group,
        ) != (512, 1, 22, 1, 1)
        || !c.norm_topk_prob
        || c.routed_scaling_factor != 5.0
    {
        return Err("unsupported native hybrid configuration/capacity".into());
    }
    let m = mamba_config(c);
    let geometry = m.layout()?;
    if c.expand.checked_mul(H) != Some(geometry.inner)
        || geometry.projected % 8 != 0
        || geometry.inner % 32 != 0
        || !(1..=gpu::MAX_STATE).contains(&m.state_dim)
        || !(1..=gpu::MAX_CONV).contains(&m.conv_kernel)
    {
        return Err("unsupported native Mamba geometry".into());
    }
    // rescale_prenorm_residual is checkpoint initialization, not a forward op.
    Ok(())
}

/// Pure admission before upload, so a later layer cannot exceed the global
/// logical budget after earlier layers have already reserved their weights.
fn resident_bytes(c: &TextConfig, l: Limits) -> Result<usize> {
    Ok(resident_payload(c, l)?.total)
}

/// One resident total, with two parts of it named: the per-Session private
/// state and the shared expert bank. Neither is counted twice. `window` is
/// the most bytes uploaded between two releases of their host copies (one
/// per MoE layer and one at the end of loading).
struct ResidentPayload {
    total: usize,
    private: usize,
    bank: usize,
    window: usize,
}
fn resident_payload(c: &TextConfig, l: Limits) -> Result<ResidentPayload> {
    let mut private = 0;
    let mut bank = 0;
    let mut total = add(bytes(&[c.vocab_size, H], 3)?, bytes(&[H], 2)?)?;
    let (mut window, mut largest) = (total, 0);
    for kind in &c.layers_block_type {
        total = add(total, bytes(&[H], 2)?)?;
        window = add(window, bytes(&[H], 2)?)?;
        let state_before = private;
        let term = match kind {
                LayerKind::Attention => {
                    let a = attention::admission(attention_limits(c, l))?;
                    private = add(private, a.kv_bytes)?;
                    a.resident_bytes
                }
                LayerKind::Moe => {
                    let a = moe::admission(moe_limits(l))?;
                    bank = add(bank, a.bank_bytes)?;
                    add(a.fixed_bytes, a.bank_bytes)?
                }
                LayerKind::Mamba => {
                    let m = mamba_config(c);
                    let g = m.layout()?;
                    let e = m.extents(1, 1)?;
                    let mut n = add(bytes(&[g.projected, H], 1)?, bytes(&[H, g.inner], 1)?)?;
                    for v in [
                        bytes(&[g.conv_channels, m.conv_kernel], 2)?,
                        bytes(&[m.heads], 12)?,
                        bytes(&[g.inner], 2)?,
                        bytes(&[e.conv], 2)?,
                        bytes(&[e.recurrent], 4)?,
                    ] {
                        n = add(n, v)?;
                    }
                    if c.use_conv_bias {
                        n = add(n, bytes(&[g.conv_channels], 2)?)?;
                    }
                    private = add(private, add(bytes(&[e.conv], 2)?, bytes(&[e.recurrent], 4)?)?)?;
                    n
                }
            };
        total = add(total, term)?;
        // KV and Mamba state are allocated, not uploaded.
        window = add(window, term - (private - state_before))?;
        if matches!(kind, LayerKind::Moe) {
            largest = largest.max(window);
            window = 0;
        }
    }
    if total > l.max_resident_bytes {
        return Err("session aggregate resident budget exceeded".into());
    }
    Ok(ResidentPayload { total, private, bank, window: largest.max(window) })
}

/// The largest working set one call may hold: the cold prefill of the whole
/// capacity, or the widest late decode chunk, within the caller's allowance.
/// A seeded Session's suffix reads two key runs, which can need more than the
/// cold span; the worst split is a one-token prefix with the C-1 suffix after
/// it (Astra's B2 budget report, Files 163cca3d: at most want+1 slots, and P=1
/// attains it), so a seeded preflight prices that span too.
fn transient_bytes(c: &TextConfig, l: Limits, seeded: bool) -> Result<usize> {
    let unbounded = Limits { max_chunk_bytes: usize::MAX, ..l };
    let a = admission(c, unbounded, l.capacity)?;
    let mut working = a.prefill.device_working_bytes.max(a.max_capacity_chunk_bytes);
    if seeded && l.capacity > 1 {
        working = working.max(prefill::admission_with_prefix(c, unbounded, 1, l.capacity - 1, 1)?
            .device_working_bytes);
    }
    Ok(working.min(l.max_chunk_bytes))
}

/// Allocatable memory a load needs: every resident byte, the largest working
/// set, the host copies of the largest upload window (the same memory on an
/// integrated GPU), and, when seeded, the sealed Mamba boundary kept beside
/// the advancing state.
fn load_requirement(c: &TextConfig, l: Limits, seeded: bool) -> Result<usize> {
    let p = resident_payload(c, l)?;
    let n = add(add(p.total, transient_bytes(c, l, seeded)?)?, p.window)?;
    if seeded { add(n, prefix::mamba_bytes(c)?) } else { Ok(n) }
}

/// What one more Session of a loaded model needs: its private state and its
/// working set, plus the sealed boundary when seeded.
fn session_requirement(c: &TextConfig, l: Limits, seeded: bool) -> Result<usize> {
    let n = add(resident_payload(c, l)?.private, transient_bytes(c, l, seeded)?)?;
    if seeded { add(n, prefix::mamba_bytes(c)?) } else { Ok(n) }
}

/// Logical ownership breakdown for ONE Session. For multiple sessions of the
/// same exact DeviceModel, count fixed_model_bytes once and each Session's
/// private_state_bytes separately. This is not physical admission; active or
/// failed chunks retain extra owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentOwnership {
    /// Shared by every Session of the model, the expert bank included.
    pub fixed_model_bytes: usize,
    /// The part of fixed_model_bytes that is the expert bank.
    pub expert_bank_bytes: usize,
    pub private_state_bytes: usize,
}

pub fn resident_ownership(config: &TextConfig, limits: Limits) -> Result<ResidentOwnership> {
    validate_config(config, limits)?;
    let payload = resident_payload(config, limits)?;
    Ok(ResidentOwnership {
        fixed_model_bytes: payload.total - payload.private,
        expert_bank_bytes: payload.bank,
        private_state_bytes: payload.private,
    })
}

/// Validate the native configuration and the cold prompt's resident/chunk
/// budgets without opening a store, creating a CUDA client or allocating device
/// memory. The caller separately admits any response reserve against capacity.
/// This does not validate source payloads or establish physical readiness.
pub fn admission(config: &TextConfig, limits: Limits, prompt_tokens: usize) -> Result<Admission> {
    validate_config(config, limits)?;
    if prompt_tokens == 0 || prompt_tokens > limits.capacity {
        return Err("empty prompt or prompt exceeds session capacity".into());
    }
    let payload = resident_payload(config, limits)?;
    let expert_cache_pairs_per_moe = if payload.bank == 0 { 0 } else { moe::EXPERTS };
    let prefill = prefill_admission(config, limits, 0, prompt_tokens)?;
    let comparison = Limits { max_chunk_bytes: usize::MAX, ..limits };
    let first_chunk_tokens = prompt_tokens.min(MAX_CHUNK);
    let first_chunk_bytes = planned_chunk_bytes(config, comparison, 0, first_chunk_tokens)?;
    let mut max_prompt_chunk_bytes = first_chunk_bytes;
    // Preserve the old chunked schedule as an explicitly descriptive comparison.
    // For fixed width, only attention's split count depends on start, and it
    // is monotone. The last full chunk and the remainder cover every maximum.
    let remainder = prompt_tokens % MAX_CHUNK;
    let full_end = prompt_tokens - remainder;
    if full_end > 0 {
        max_prompt_chunk_bytes = max_prompt_chunk_bytes.max(planned_chunk_bytes(
            config, comparison, full_end - MAX_CHUNK, MAX_CHUNK,
        )?);
    }
    if remainder > 0 {
        max_prompt_chunk_bytes = max_prompt_chunk_bytes.max(planned_chunk_bytes(
            config, comparison, full_end, remainder,
        )?);
    }
    // Report all possible future widths without rejecting a prompt because
    // an unrelated larger chunk would exceed the caller's working budget.
    let unbounded_chunk = Limits { max_chunk_bytes: usize::MAX, ..limits };
    let max_capacity_chunk_bytes = (1..=limits.capacity.min(MAX_CHUNK))
        .try_fold(0usize, |maximum, tokens| -> Result<usize> {
            Ok(maximum.max(planned_chunk_bytes(
                config, unbounded_chunk, limits.capacity - tokens, tokens,
            )?))
        })?;
    Ok(Admission {
        base_resident_bytes: payload.total - payload.bank,
        expert_cache_pairs_per_moe,
        expert_cache_bytes: payload.bank,
        resident_bytes: payload.total,
        prefill,
        first_chunk_tokens, first_chunk_bytes, max_prompt_chunk_bytes, max_capacity_chunk_bytes,
    })
}

/// The shared logical planner for admission and live appends. Config is already
/// validated by admission/source::load. The loader checks actual head [V,H],
/// Mamba input [projected,H]/output [H,inner] and each fixed attention/MoE shape
/// before constructing a Session, so these are the uploaded operand extents.
fn planned_chunk_bytes(c: &TextConfig, l: Limits, start: usize, t: usize) -> Result<usize> {
    planned_prefix_chunk_bytes(c, l, start, t, 0)
}
fn planned_prefix_chunk_bytes(c: &TextConfig, l: Limits, start: usize, t: usize, prefix: usize) -> Result<usize> {
    if !(1..=MAX_CHUNK).contains(&t)
        || start.checked_add(t).filter(|&end| end <= l.capacity).is_none()
    {
        return Err("empty/oversized chunk or capacity exhausted".into());
    }
    // IDs, embedding, final norm and final-row copy; each layer's norm,
    // mixer output and residual are retained until whole-chunk completion.
    let mut total = add(bytes(&[t], 4)?, bytes(&[t, H], 4)?)?;
    total = add(total, bytes(&[H], 2)?)?;
    total = add(total, super::fp8::working_bytes(1, [c.vocab_size, H])?)?;
    for kind in &c.layers_block_type {
        total = add(total, bytes(&[t, H], 6)?)?;
        total = add(total, layer_prefix_working_bytes(c, l, start, t, kind, prefix)?)?;
    }
    if total > l.max_chunk_bytes {
        return Err("session aggregate chunk budget exceeded".into());
    }
    Ok(total)
}

fn layer_prefix_working_bytes(c: &TextConfig, l: Limits, start: usize, t: usize, kind: &LayerKind, prefix: usize) -> Result<usize> {
    Ok(match kind {
        LayerKind::Attention => attention::Attention::planned_prefix_working_bytes(
            start, t, attention_limits(c, l), prefix,
        )?,
        LayerKind::Moe => moe::working_bytes(t)?,
        LayerKind::Mamba => {
            let m = mamba_config(c);
            let e = m.extents(1, t)?;
            let g = m.layout()?;
            let mut n = add(
                super::fp8::working_bytes(t, [g.projected, H])?,
                super::fp8::working_bytes(t, [H, g.inner])?,
            )?;
            for v in [bytes(&[e.output], 6)?, bytes(&[t, g.conv_channels], 2)?,
                bytes(&[e.conv], 2)?, bytes(&[e.recurrent], 4)?] {
                n = add(n, v)?;
            }
            n
        }
    })
}

impl<S> DeviceModel<S> {
    pub fn source(&self) -> Source { self.source }
    pub fn snapshot(&self) -> &ModelSnapshot<S> { &self.snapshot }
    pub fn config(&self) -> &TextConfig { &self.config }
    /// Exact ordered selections used by the original loader, for provenance.
    pub fn layer_ids(&self) -> impl ExactSizeIterator<Item = Id> + '_ {
        self.layers.iter().map(|layer| layer.id)
    }
}

impl<S: BlobStoreGet> Session<S> {
    pub fn device_model(&self) -> &Arc<DeviceModel<S>> { &self.model }

    /// Fresh private state on this thread's stream over the SAME uploaded
    /// operands and expert bank. No source lookup/upload or implicit device
    /// selection. Limits keep their complete-model meaning.
    /// Sharing does not certify CUDA health or drain another failed Session;
    /// a lost device context may affect every Session using this model.
    pub fn from_device_model(model: Arc<DeviceModel<S>>, limits: Limits) -> Result<Self> {
        Self::from_state(model, limits, None, None)
    }
    fn from_state(model: Arc<DeviceModel<S>>, limits: Limits, seed: Option<Arc<[u32]>>,
        prefix: Option<Arc<Prefix<S>>>) -> Result<Self>
    {
        validate_config(&model.config, limits)?;
        prefix::validate_initial(&model, limits, seed.as_deref(), prefix.as_deref())?;
        source::fits(session_requirement(&model.config, limits, seed.is_some())?,
            source::allocatable_bytes(&model.device)?)?;
        let stream = prefix.as_ref().map_or_else(StreamId::current, |p| p.stream);
        stream.executes(|| {
        // KV and state at their own sizes, not in quarter-device pages.
        let layers = crate::nn::exact_alloc::exact(&model.client, || -> Result<Vec<Layer>> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for (index, layer) in model.layers.iter().enumerate() {
            let mixer = match &layer.mixer {
                MixerWeights::Mamba(weights) => Mixer::Mamba(Mamba {
                    weights: weights.clone(),
                    committed: prefix.as_ref().and_then(|p| p.mamba(index)), staged: None,
                }),
                MixerWeights::Attention(weights) => {
                    let mut al = attention_limits(&model.config, limits);
                    if let Some(seed) = &seed { al.capacity = seed.len(); }
                    Mixer::Attention(if let Some(p) = &prefix {
                        attention::Attention::from_prefix(weights.clone(), p.attention(index)?, al)?
                    } else { attention::Attention::from_weights(weights.clone(), al)? })
                }
                MixerWeights::Moe(weights) => Mixer::Moe(moe::Moe::from_weights(
                    weights.clone(), moe_limits(limits),
                )?),
            };
            layers.push(Layer { id: layer.id, norm: layer.norm.clone(), mixer });
        }
        Ok(layers)
        })?;
        let committed = prefix.as_ref().map_or_else(Vec::new, |p| p.ids.to_vec());
        let logits = prefix.as_ref().map_or_else(Vec::new, |p| p.logits.to_vec());
        let session = Self {
            model, stream, limits, layers, status: Status::Ready,
            committed, logits, pending: Vec::new(), span_pending: None, seed, prefix,
            #[cfg(test)]
            fault: None,
        };
        // Prefix publication has already fenced the origin. Observe allocation
        // errors here too before a fork advertises a usable boundary.
        if session.prefix.is_some() { session.fence()?; }
        Ok(session)
        })
    }

    /// No publication, download or default collection selection. Layer IDs are
    /// explicit opaque choices in ordered config positions. Multiple matches
    /// remain legal. Fixed FP8/aux operands choose the first typed tuple and
    /// hold its values. NVFP4 uses the minimum opaque joined tuple, including
    /// complete typed expert pairs at each requested index, stable on repeated
    /// queries of this snapshot. Every expert of every MoE layer is uploaded
    /// here, once, after a check of the device's actual free memory; no source
    /// payloads are retained beside the frozen snapshot.
    /// After the original ordered source checks/uploads, completion is observed
    /// on the upload stream before publishing a shareable model. Deferred CUDA
    /// upload errors therefore return here, before any Session can use it.
    pub fn from_snapshot(
        snapshot: Arc<ModelSnapshot<S>>,
        source: Source,
        layer_ids: &[Id],
        device: CudaDevice,
        limits: Limits,
    ) -> Result<Self> {
        source::load(snapshot, source, layer_ids, device, limits, None,
            #[cfg(test)]
            |_, _, _| {},
        )
    }
    pub fn source(&self) -> Source {
        self.model.source
    }
    pub fn snapshot(&self) -> &ModelSnapshot<S> {
        &self.model.snapshot
    }
    pub fn status(&self) -> Status {
        self.status
    }
    pub fn capacity(&self) -> usize {
        self.limits.capacity
    }
    pub fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }
    /// While poisoned these IDs are diagnostic metadata, NOT a reusable cache.
    pub fn committed_ids(&self) -> &[u32] {
        &self.committed
    }
    pub fn last_logits(&self) -> Option<NextLogits<'_>> {
        (self.status == Status::Ready && !self.committed.is_empty()).then_some(NextLogits {
            committed_len: self.committed.len(),
            values: &self.logits,
        })
    }
    pub fn can_reuse_prefix(&self, rendered_ids: &[u32]) -> bool {
        self.status == Status::Ready && rendered_ids.starts_with(&self.committed)
            && self.seed.as_ref().is_none_or(|ids| rendered_ids.starts_with(ids))
    }
    fn chunk_bytes(&self, t: usize) -> Result<usize> {
        planned_prefix_chunk_bytes(&self.model.config, self.limits, self.committed.len(), t,
            self.prefix.as_ref().map_or(0, |p| p.ids.len()))
    }
    fn preflight(&self, ids: &[u32]) -> Result<()> {
        if self.status != Status::Ready {
            return Err("session poisoned; successful reset required".into());
        }
        if !(1..=MAX_CHUNK).contains(&ids.len())
            || ids.iter().any(|&id| id as usize >= self.model.config.vocab_size)
            || self
                .committed
                .len()
                .checked_add(ids.len())
                .filter(|&n| n <= self.limits.capacity)
                .is_none()
        {
            return Err("empty/oversized chunk, invalid ID or capacity exhausted".into());
        }
        self.chunk_bytes(ids.len())?;
        if let Some(seed) = &self.seed {
            let start = self.committed.len();
            if seed.get(start..start + ids.len()) != Some(ids) {
                return Err("append differs from or crosses the prepared prefix".into());
            }
        }
        self.check_boundary(self.committed.len())
    }
    fn check_boundary(&self, position: usize) -> Result<()> {
        for layer in &self.layers {
            let ready = match &layer.mixer {
                Mixer::Mamba(m) => {
                    m.staged.is_none()
                        && m.committed.as_ref().map_or(0, |s| s.attended()) == position
                }
                Mixer::Attention(a) => {
                    !a.is_poisoned() && !a.has_pending() && a.attended() == position
                }
                Mixer::Moe(m) => !m.is_poisoned(),
            };
            if !ready {
                return Err(format!(
                    "layer {:?} is not at common committed boundary",
                    layer.id
                ));
            }
        }
        Ok(())
    }
    /// Synchronous API; cancellation is polled by caller BEFORE this method or
    /// AFTER its completed chunk. Do not discard an in-flight future to cancel.
    /// All IDs (including EOS when intentionally attended) are ordinary inputs;
    /// a generator must keep emitted versus attended IDs separate.
    pub fn append_ids(&mut self, ids: &[u32]) -> Result<NextLogits<'_>> {
        self.stream.executes(|| self.append_on_stream(ids))
    }
    fn append_on_stream(&mut self, ids: &[u32]) -> Result<NextLogits<'_>> {
        self.preflight(ids)?;
        // Poison BEFORE dispatch: even unwinding cannot expose a Ready prefix.
        self.status = Status::Poisoned;
        #[cfg(feature = "nemotron-timing")]
        let mut timing = super::host_timing::start();
        let result = self.run(
            ids,
            #[cfg(feature = "nemotron-timing")]
            timing.as_mut().map(|(_, timing)| timing),
        );
        #[cfg(feature = "nemotron-timing")]
        if let Some((started, timing)) = timing {
            timing.emit(
                started.elapsed(), ids.len(), self.committed.len(), result.is_ok(),
            );
        }
        match result {
            Ok(logits) => {
                for layer in &mut self.layers {
                    if let Mixer::Mamba(m) = &mut layer.mixer {
                        m.committed = m.staged.take().map(Arc::new);
                    }
                }
                self.committed.extend_from_slice(ids);
                self.logits = logits;
                self.pending.clear();
                self.status = Status::Ready;
                if let Err(error) = self.publish_seed_if_complete() {
                    self.invalidate();
                    return Err(error);
                }
                Ok(self.last_logits().expect("nonempty completed chunk"))
            }
            Err(error) => {
                self.invalidate();
                Err(error)
            }
        }
    }
    fn run(
        &mut self,
        ids: &[u32],
        #[cfg(feature = "nemotron-timing")] mut timing: Option<&mut super::host_timing::Timings>,
    ) -> Result<Vec<f32>> {
        let t = ids.len();
        let h = self.model.client.create_from_slice(u32::as_bytes(ids));
        self.pending.push(CudaTensor::new_contiguous(
            self.model.client.clone(),
            self.model.device.clone(),
            [t].into(),
            h.clone(),
            DType::U32,
        ));
        let mut x = empty(&self.model.client, &self.model.device, t, H);
        self.pending.push(x.clone());
        unsafe {
            embedding_kernel::launch_unchecked::<CudaRuntime>(
                &self.model.client,
                CubeCount::new_1d((t * H).div_ceil(128) as u32),
                CubeDim::new_1d(128),
                ArrayArg::from_raw_parts(h, t),
                ArrayArg::from_raw_parts(self.model.embedding.handle.clone(), self.model.config.vocab_size * H),
                ArrayArg::from_raw_parts(x.handle.clone(), t * H),
                t * H,
            );
        }
        for (_index, layer) in self.layers.iter_mut().enumerate() {
            x = enqueue_layer_tile(layer, &x, self.committed.len(), self.limits,
                self.model.config.layer_norm_epsilon as f32, &mut self.pending,
                #[cfg(feature = "nemotron-timing")] timing.as_deref_mut(),
            )?;
            #[cfg(test)]
            {
                inject(self.fault, Fault::AfterLayer(_index))?;
                if self.fault == Some(Fault::PanicAfterLayer(_index)) {
                    panic!("injected session unwind");
                }
            }
        }
        let values = self.project_last_logits(&x)?;
        for (_index, layer) in self.layers.iter_mut().enumerate() {
            if let Mixer::Attention(a) = &mut layer.mixer {
                a.commit()?;
                #[cfg(test)]
                inject(self.fault, Fault::AfterCommit(_index))?;
            }
        }
        self.fence()?; // Even a no-attention chain needs an explicit completion.
        Ok(values)
    }
    fn project_last_logits(&mut self, x: &CudaTensor) -> Result<Vec<f32>> {
        let t = x.meta.shape().as_slice()[1];
        let norm = normalized(x, &self.model.final_norm, self.model.config.layer_norm_epsilon as f32);
        self.pending.push(norm.clone());
        // Copy the final row rather than manufacture an offset alias accepted
        // by neither Projection's zero-offset contract nor every downstream op.
        let row = empty(&self.model.client, &self.model.device, 1, H);
        self.pending.push(row.clone());
        unsafe {
            last_row_kernel::launch_unchecked::<CudaRuntime>(
                &self.model.client,
                CubeCount::new_1d(32),
                CubeDim::new_1d(128),
                ArrayArg::from_raw_parts(norm.handle.clone(), t * H),
                ArrayArg::from_raw_parts(row.handle.clone(), H),
                (t - 1) * H,
            );
        }
        let logits = self.model.head.project(&row, self.limits.max_chunk_bytes)?;
        self.pending.push(logits.clone());
        let data = self.model
            .client
            .read_one(logits.handle.clone())
            .map_err(|e| format!("session logits read: {e:?}"))?;
        if data.len() < self.model.config.vocab_size * 2 {
            return Err("short session logits read".into());
        }
        let values: Vec<f32> = data[..self.model.config.vocab_size * 2]
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite session logits".into());
        }
        Ok(values)
    }
    fn fence(&self) -> Result<()> {
        cubecl::future::block_on(self.model.client.sync())
            .map_err(|e| format!("session CUDA completion: {e:?}"))
    }
    fn invalidate(&mut self) {
        self.status = Status::Poisoned;
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Attention(a) => a.invalidate(),
                Mixer::Moe(m) => m.invalidate(),
                Mixer::Mamba(_) => {}
            }
        }
    }
    /// Attempts ALL component resets even if one fails. On any error no Ready
    /// state is published, no Mamba/pending/prefix owner is cleared. Attention
    /// may have individually reset: the hybrid remains poisoned, not rolled back.
    pub fn reset(&mut self) -> Result<()> {
        self.stream.executes(|| self.reset_on_stream())
    }
    fn reset_on_stream(&mut self) -> Result<()> {
        self.invalidate();
        let mut errors = Vec::new();
        if let Err(e) = self.fence() {
            errors.push(e);
        }
        #[cfg(test)]
        if let Err(e) = inject(self.fault, Fault::Reset) {
            errors.push(e);
        }
        for layer in &mut self.layers {
            let result = match &mut layer.mixer {
                Mixer::Attention(a) => a.reset_capacity(self.limits.capacity),
                Mixer::Moe(m) => m.reset(),
                Mixer::Mamba(_) => Ok(()),
            };
            if let Err(e) = result {
                errors.push(e);
            }
        }
        if !errors.is_empty() {
            self.invalidate();
            return Err(errors.join("; "));
        }
        if self.seed.is_some() || self.prefix.is_some() { self.fence()?; }
        for layer in &mut self.layers {
            if let Mixer::Mamba(m) = &mut layer.mixer {
                m.committed = None;
                m.staged = None;
            }
        }
        self.pending.clear();
        self.seed = None;
        self.prefix = None;
        self.span_pending = None;
        self.committed.clear();
        self.logits.clear();
        self.status = Status::Ready;
        Ok(())
    }
}

// Both ordinary short append and span prefill share this exact layer dispatch.
// Caller owns publication and chooses when a successful tile may be retired.
fn enqueue_layer_tile(layer: &mut Layer, x: &CudaTensor,
    start: usize, limits: Limits, epsilon: f32, pending: &mut Vec<CudaTensor>,
    #[cfg(feature = "nemotron-timing")] mut timing: Option<&mut super::host_timing::Timings>,
) -> Result<CudaTensor> {
    let norm = normalized(x, &layer.norm, epsilon);
    pending.push(norm.clone());
    let y = match &mut layer.mixer {
        Mixer::Attention(a) => a.enqueue(&norm, start)?.values,
        Mixer::Moe(m) => {
            #[cfg(feature = "nemotron-timing")]
            let result = match timing.as_deref_mut() {
                Some(timing) => m.forward_timed(&norm, timing),
                None => m.forward(&norm),
            };
            #[cfg(not(feature = "nemotron-timing"))]
            let result = m.forward(&norm);
            result?.values
        }
        Mixer::Mamba(m) => {
            let projected = m.weights.input.project(&norm, limits.max_chunk_bytes)?;
            pending.push(projected.clone());
            let out = gpu::chunk(
                m.weights.config,
                &gpu::Parameters {
                    conv_weight: &m.weights.conv,
                    conv_bias: m.weights.bias.as_ref(),
                    a_log: &m.weights.a_log,
                    dt_bias: &m.weights.dt,
                    d: &m.weights.d,
                    norm_gain: &m.weights.gain,
                },
                &projected,
                m.committed.as_deref(),
                limits.max_chunk_bytes,
            )?;
            m.staged = Some(out.state);
            pending.push(out.values.clone());
            m.weights.output.project(&out.values, limits.max_chunk_bytes)?
        }
    };
    pending.push(y.clone());
    let output = residual(x, &y);
    pending.push(output.clone());
    Ok(output)

}

fn empty(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    t: usize,
    n: usize,
) -> CudaTensor {
    CudaTensor::new_contiguous(
        client.clone(),
        device.clone(),
        [1, t, n].into(),
        client.empty(t * n * 2),
        DType::BF16,
    )
}
#[cube(launch_unchecked)]
fn embedding_kernel(ids: &Array<u32>, weight: &Array<bf16>, out: &mut Array<bf16>, len: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < len {
        out[i] = weight[ids[i / 4096usize] as usize * 4096usize + i % 4096usize];
    }
}
#[cube(launch_unchecked)]
fn last_row_kernel(x: &Array<bf16>, out: &mut Array<bf16>, offset: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < 4096usize {
        out[i] = x[offset + i];
    }
}
#[cube(launch_unchecked)]
fn norm_kernel(x: &Array<bf16>, gain: &Array<bf16>, out: &mut Array<bf16>, epsilon: f32) {
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut sums = SharedMemory::<f32>::new(64usize);
    let mut sum = 0.0f32;
    for j in 0usize..64usize {
        let v = f32::cast_from(x[row * 4096usize + lane + j * 64usize]);
        sum += v * v;
    }
    sums[lane] = sum;
    sync_cube();
    let mut step = 32usize;
    while step > 0usize {
        if lane < step {
            sums[lane] += sums[lane + step];
        }
        sync_cube();
        step /= 2usize;
    }
    let inverse = 1.0f32 / (sums[0usize] / 4096.0f32 + epsilon).sqrt();
    for j in 0usize..64usize {
        let col = lane + j * 64usize;
        let rounded = bf16::cast_from(f32::cast_from(x[row * 4096usize + col]) * inverse);
        out[row * 4096usize + col] =
            bf16::cast_from(f32::cast_from(rounded) * f32::cast_from(gain[col]));
    }
}
#[cube(launch_unchecked)]
fn add_kernel(x: &Array<bf16>, y: &Array<bf16>, out: &mut Array<bf16>, len: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < len {
        out[i] = bf16::cast_from(f32::cast_from(x[i]) + f32::cast_from(y[i]));
    }
}
fn normalized(x: &CudaTensor, gain: &CudaTensor, eps: f32) -> CudaTensor {
    let t = x.meta.shape().as_slice()[1];
    let out = empty(&x.client, &x.device, t, H);
    unsafe {
        norm_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            CubeCount::new_1d(t as u32),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(x.handle.clone(), t * H),
            ArrayArg::from_raw_parts(gain.handle.clone(), H),
            ArrayArg::from_raw_parts(out.handle.clone(), t * H),
            eps,
        );
    }
    out
}
fn residual(x: &CudaTensor, y: &CudaTensor) -> CudaTensor {
    let t = x.meta.shape().as_slice()[1];
    let out = empty(&x.client, &x.device, t, H);
    unsafe {
        add_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            CubeCount::new_1d((t * H).div_ceil(128) as u32),
            CubeDim::new_1d(128),
            ArrayArg::from_raw_parts(x.handle.clone(), t * H),
            ArrayArg::from_raw_parts(y.handle.clone(), t * H),
            ArrayArg::from_raw_parts(out.handle.clone(), t * H),
            t * H,
        );
    }
    out
}

#[cfg(test)]
#[path = "session_tests.rs"]
pub(super) mod tests;
