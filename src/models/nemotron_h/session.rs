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
};
use half::bf16;
use std::sync::Arc;
use triblespace::{core::collection::CollectionHandle, prelude::*};

#[path = "session_source.rs"]
mod source;
use source::NativeExperts;

type Result<T> = std::result::Result<T, String>;
const H: usize = 4096;
pub const MAX_CHUNK: usize = 256;

/// Explicit caller-selected observation coordinates. Collection is provenance,
/// not an extra authorization check; the held ModelSnapshot carries support.
#[derive(Clone, Copy, Debug)]
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
    /// Fixed weights/KV plus one committed Mamba state per Mamba layer.
    pub max_resident_bytes: usize,
    /// Sum of fresh logical allocations for one chunk, including staged state,
    /// retained outer outputs and one transient expert pair. NOT physical peak,
    /// host staging, allocator/JIT overhead or a whole-device reservation.
    pub max_chunk_bytes: usize,
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

struct Mamba {
    config: contract::Config,
    input: Projection,
    output: Projection,
    conv: CudaTensor,
    bias: Option<CudaTensor>,
    a_log: CudaTensor,
    dt: CudaTensor,
    d: CudaTensor,
    gain: CudaTensor,
    committed: Option<gpu::State<CudaRuntime>>,
    staged: Option<gpu::State<CudaRuntime>>,
}
enum Mixer<S> {
    Mamba(Mamba),
    Attention(attention::Attention),
    Moe(moe::Moe<NativeExperts<S>>),
}
struct Layer<S> {
    id: Id,
    norm: CudaTensor,
    mixer: Mixer<S>,
}

pub struct Session<S> {
    snapshot: Arc<ModelSnapshot<S>>,
    source: Source,
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    config: TextConfig,
    limits: Limits,
    embedding: CudaTensor,
    final_norm: CudaTensor,
    head: Projection,
    layers: Vec<Layer<S>>,
    status: Status,
    committed: Vec<u32>,
    logits: Vec<f32>,
    // Ordinary owning handles, never borrowed raw pointers. Keep outer values
    // across an uncertain append/reset. Component-private pending owners stay
    // inside their respective layers; Mamba staged states are installed early.
    pending: Vec<CudaTensor>,
    #[cfg(test)]
    fault: Option<Fault>,
}

// Control-injection only, not an imitation of CUDA failure/drain semantics.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    AfterLayer(usize),
    AfterCommit(usize),
    Reset,
    PanicAfterLayer(usize),
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
        max_fixed_bytes: l.max_resident_bytes,
        max_expert_bytes: l.max_chunk_bytes,
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
    let mut total = add(bytes(&[c.vocab_size, H], 3)?, bytes(&[H], 2)?)?;
    for kind in &c.layers_block_type {
        total = add(total, bytes(&[H], 2)?)?;
        total = add(
            total,
            match kind {
                LayerKind::Attention => {
                    attention::admission(attention_limits(c, l))?.resident_bytes
                }
                LayerKind::Moe => moe::admission(moe_limits(l))?.fixed_bytes,
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
                    n
                }
            },
        )?;
    }
    if total > l.max_resident_bytes {
        return Err("session aggregate resident budget exceeded".into());
    }
    Ok(total)
}

impl<S: BlobStoreGet> Session<S> {
    /// No publication, download or default collection selection. Layer IDs are
    /// explicit opaque choices in ordered config positions. Multiple matches
    /// remain legal. Fixed FP8/aux operands choose the first typed tuple and
    /// hold its values. NVFP4 uses the minimum opaque joined tuple (and expert
    /// ID at each requested index), stable on repeated queries of this snapshot.
    /// No full expert catalog is retained.
    pub fn from_snapshot(
        snapshot: Arc<ModelSnapshot<S>>,
        source: Source,
        layer_ids: &[Id],
        device: CudaDevice,
        limits: Limits,
    ) -> Result<Self> {
        source::load(snapshot, source, layer_ids, device, limits)
    }
    pub fn source(&self) -> Source {
        self.source
    }
    pub fn snapshot(&self) -> &ModelSnapshot<S> {
        &self.snapshot
    }
    pub fn status(&self) -> Status {
        self.status
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
    }
    fn chunk_bytes(&self, t: usize) -> Result<usize> {
        // IDs, embedding, final norm and final-row copy; each layer's norm,
        // mixer output and residual are retained until whole-chunk completion.
        let mut total = add(bytes(&[t], 4)?, bytes(&[t, H], 4)?)?;
        total = add(total, bytes(&[H], 2)?)?;
        total = add(total, self.head.working_bytes(1)?)?;
        let mut expert_peak = 0;
        for layer in &self.layers {
            total = add(total, bytes(&[t, H], 6)?)?;
            total = add(
                total,
                match &layer.mixer {
                    Mixer::Attention(_) => attention::Attention::planned_working_bytes(
                        self.committed.len(),
                        t,
                        attention_limits(&self.config, self.limits),
                    )?,
                    Mixer::Moe(m) => {
                        expert_peak = expert_peak.max(m.admission().expert_bytes);
                        moe::working_bytes(t)?
                    }
                    Mixer::Mamba(m) => {
                        let e = m.config.extents(1, t)?;
                        let g = m.config.layout()?;
                        let mut n = add(m.input.working_bytes(t)?, m.output.working_bytes(t)?)?;
                        for v in [
                            bytes(&[e.output], 6)?,
                            bytes(&[t, g.conv_channels], 2)?,
                            bytes(&[e.conv], 2)?,
                            bytes(&[e.recurrent], 4)?,
                        ] {
                            n = add(n, v)?;
                        }
                        n
                    }
                },
            )?;
        }
        total = add(total, expert_peak)?;
        if total > self.limits.max_chunk_bytes {
            return Err("session aggregate chunk budget exceeded".into());
        }
        Ok(total)
    }
    fn preflight(&self, ids: &[u32]) -> Result<()> {
        if self.status != Status::Ready {
            return Err("session poisoned; successful reset required".into());
        }
        if !(1..=MAX_CHUNK).contains(&ids.len())
            || ids.iter().any(|&id| id as usize >= self.config.vocab_size)
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
        for layer in &self.layers {
            let ready = match &layer.mixer {
                Mixer::Mamba(m) => {
                    m.staged.is_none()
                        && m.committed.as_ref().map_or(0, |s| s.attended()) == self.committed.len()
                }
                Mixer::Attention(a) => {
                    !a.is_poisoned() && !a.has_pending() && a.attended() == self.committed.len()
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
        self.preflight(ids)?;
        // Poison BEFORE dispatch: even unwinding cannot expose a Ready prefix.
        self.status = Status::Poisoned;
        let result = self.run(ids);
        match result {
            Ok(logits) => {
                for layer in &mut self.layers {
                    if let Mixer::Mamba(m) = &mut layer.mixer {
                        m.committed = m.staged.take();
                    }
                }
                self.committed.extend_from_slice(ids);
                self.logits = logits;
                self.pending.clear();
                self.status = Status::Ready;
                Ok(self.last_logits().expect("nonempty completed chunk"))
            }
            Err(error) => {
                self.invalidate();
                Err(error)
            }
        }
    }
    fn run(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = ids.len();
        let h = self.client.create_from_slice(u32::as_bytes(ids));
        self.pending.push(CudaTensor::new_contiguous(
            self.client.clone(),
            self.device.clone(),
            [t].into(),
            h.clone(),
            DType::U32,
        ));
        let mut x = empty(&self.client, &self.device, t, H);
        self.pending.push(x.clone());
        unsafe {
            embedding_kernel::launch_unchecked::<CudaRuntime>(
                &self.client,
                CubeCount::new_1d((t * H).div_ceil(128) as u32),
                CubeDim::new_1d(128),
                ArrayArg::from_raw_parts(h, t),
                ArrayArg::from_raw_parts(self.embedding.handle.clone(), self.config.vocab_size * H),
                ArrayArg::from_raw_parts(x.handle.clone(), t * H),
                t * H,
            );
        }
        for (_index, layer) in self.layers.iter_mut().enumerate() {
            let norm = normalized(&x, &layer.norm, self.config.layer_norm_epsilon as f32);
            self.pending.push(norm.clone());
            let y = match &mut layer.mixer {
                Mixer::Attention(a) => a.enqueue(&norm, self.committed.len())?.values,
                Mixer::Moe(m) => m.forward(&norm)?.values,
                Mixer::Mamba(m) => {
                    let projected = m.input.project(&norm, self.limits.max_chunk_bytes)?;
                    self.pending.push(projected.clone());
                    let out = gpu::chunk(
                        m.config,
                        &gpu::Parameters {
                            conv_weight: &m.conv,
                            conv_bias: m.bias.as_ref(),
                            a_log: &m.a_log,
                            dt_bias: &m.dt,
                            d: &m.d,
                            norm_gain: &m.gain,
                        },
                        &projected,
                        m.committed.as_ref(),
                        self.limits.max_chunk_bytes,
                    )?;
                    m.staged = Some(out.state);
                    self.pending.push(out.values.clone());
                    m.output.project(&out.values, self.limits.max_chunk_bytes)?
                }
            };
            self.pending.push(y.clone());
            x = residual(&x, &y);
            self.pending.push(x.clone());
            #[cfg(test)]
            {
                inject(self.fault, Fault::AfterLayer(_index))?;
                if self.fault == Some(Fault::PanicAfterLayer(_index)) {
                    panic!("injected session unwind");
                }
            }
        }
        let norm = normalized(&x, &self.final_norm, self.config.layer_norm_epsilon as f32);
        self.pending.push(norm.clone());
        // Copy the final row rather than manufacture an offset alias accepted
        // by neither Projection's zero-offset contract nor every downstream op.
        let row = empty(&self.client, &self.device, 1, H);
        self.pending.push(row.clone());
        unsafe {
            last_row_kernel::launch_unchecked::<CudaRuntime>(
                &self.client,
                CubeCount::new_1d(32),
                CubeDim::new_1d(128),
                ArrayArg::from_raw_parts(norm.handle.clone(), t * H),
                ArrayArg::from_raw_parts(row.handle.clone(), H),
                (t - 1) * H,
            );
        }
        let logits = self.head.project(&row, self.limits.max_chunk_bytes)?;
        self.pending.push(logits.clone());
        let data = self
            .client
            .read_one(logits.handle.clone())
            .map_err(|e| format!("session logits read: {e:?}"))?;
        if data.len() < self.config.vocab_size * 2 {
            return Err("short session logits read".into());
        }
        let values: Vec<f32> = data[..self.config.vocab_size * 2]
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite session logits".into());
        }
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
    fn fence(&self) -> Result<()> {
        cubecl::future::block_on(self.client.sync())
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
                Mixer::Attention(a) => a.reset(),
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
        for layer in &mut self.layers {
            if let Mixer::Mamba(m) = &mut layer.mixer {
                m.committed = None;
                m.staged = None;
            }
        }
        self.pending.clear();
        self.committed.clear();
        self.logits.clear();
        self.status = Status::Ready;
        Ok(())
    }
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
mod tests;
