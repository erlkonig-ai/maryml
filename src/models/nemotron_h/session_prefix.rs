//! One explicit, immutable hybrid checkpoint. No lookup catalogue, eviction or
//! cross-process identity. Tokens and exact uploaded ownership decide reuse.
use super::*;

enum Boundary {
    Mamba(Arc<gpu::State<CudaRuntime>>),
    Attention(Arc<attention::KvPrefix>),
    Stateless,
}

/// Published only after successful origin-stream completion. Private fields
/// prevent importing or relabelling a checkpoint from another model/runtime.
pub struct Prefix<S> {
    pub(super) model: Arc<DeviceModel<S>>,
    pub(super) stream: StreamId,
    pub(super) ids: Arc<[u32]>,
    pub(super) logits: Arc<[f32]>,
    layers: Vec<Boundary>,
}
impl<S> Prefix<S> {
    pub fn token_ids(&self) -> &[u32] { &self.ids }
    pub fn device_model(&self) -> &Arc<DeviceModel<S>> { &self.model }
    pub fn compatible_with(&self, model: &Arc<DeviceModel<S>>) -> bool {
        Arc::ptr_eq(&self.model, model)
    }
    pub(super) fn mamba(&self, index: usize) -> Option<Arc<gpu::State<CudaRuntime>>> {
        match &self.layers[index] { Boundary::Mamba(s) => Some(s.clone()), _ => None }
    }
    pub(super) fn attention(&self, index: usize) -> Result<Arc<attention::KvPrefix>> {
        match &self.layers[index] {
            Boundary::Attention(a) => Ok(a.clone()),
            _ => Err("prefix layer kind differs".into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixOwnership {
    pub prefix_tokens: usize,
    pub shared_kv_bytes: usize,
    pub private_kv_bytes: usize,
    pub shared_mamba_bytes: usize,
    pub private_mamba_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixAllocation {
    pub seed: PrefillAllocation,
    pub suffix: Option<PrefillAllocation>,
    /// Logical payload after private advancement; excludes fixed weights,
    /// private experts, temporary allocations and allocator/runtime overhead.
    pub after_advance: PrefixOwnership,
}

fn mamba_bytes(c: &TextConfig) -> Result<usize> {
    let e = mamba_config(c).extents(1, 1)?;
    let one = add(bytes(&[e.conv], 2)?, bytes(&[e.recurrent], 4)?)?;
    one.checked_mul(c.layers_block_type.iter().filter(|k| matches!(k, LayerKind::Mamba)).count())
        .ok_or("prefix Mamba byte overflow".into())
}
fn ownership(c: &TextConfig, l: Limits, prefix: usize) -> Result<PrefixOwnership> {
    if prefix == 0 || prefix > l.capacity { return Err("prefix outside context capacity".into()); }
    let row = c.layers_block_type.iter().filter(|k| matches!(k, LayerKind::Attention)).count()
        .checked_mul(2 * 2 * 2 * 128).ok_or("prefix KV row overflow")?;
    Ok(PrefixOwnership {
        prefix_tokens: prefix,
        shared_kv_bytes: prefix.checked_mul(row).ok_or("prefix KV overflow")?,
        private_kv_bytes: (l.capacity - prefix).checked_mul(row).ok_or("suffix KV overflow")?,
        shared_mamba_bytes: mamba_bytes(c)?,
        private_mamba_bytes: if prefix == l.capacity { 0 } else { mamba_bytes(c)? },
    })
}

/// The two actual prefill spans, with exact multi-run scratch for the suffix.
/// The existing complete-model resident allowance still determines private
/// expert quota. Retaining Mamba for restore is additional shared payload.
pub fn prefix_admission(c: &TextConfig, l: Limits, prompt: usize, prefix: usize) -> Result<PrefixAllocation> {
    validate_config(c, l)?;
    resident_ownership(c, l)?;
    if prefix == 0 || prefix > prompt || prompt > l.capacity {
        return Err("prefix/prompt outside context capacity".into());
    }
    Ok(PrefixAllocation {
        seed: prefill::admission(c, l, 0, prefix)?,
        suffix: if prompt == prefix { None } else {
            Some(prefill::admission_with_prefix(c, l, prefix, prompt - prefix, prefix)?)
        },
        after_advance: ownership(c, l, prefix)?,
    })
}

pub(super) fn validate_seed(c: &TextConfig, l: Limits, ids: Option<&[u32]>) -> Result<()> {
    if let Some(ids) = ids {
        if ids.is_empty() || ids.len() > l.capacity || ids.iter().any(|&id| id as usize >= c.vocab_size) {
            return Err("empty/oversized prefix or invalid token ID".into());
        }
    }
    Ok(())
}
pub(super) fn validate_initial<S>(model: &Arc<DeviceModel<S>>, l: Limits,
    seed: Option<&[u32]>, prefix: Option<&Prefix<S>>) -> Result<()>
{
    validate_seed(&model.config, l, seed)?;
    if let Some(p) = prefix {
        if seed.is_some() || !p.compatible_with(model) || p.ids.len() > l.capacity {
            return Err("prefix exact model owner or capacity differs".into());
        }
    }
    Ok(())
}

impl<S: BlobStoreGet> Session<S> {
    /// Cold source loading keeps original fixed upload order and quota. Only
    /// P rows are reserved until that exact prefix completes; no eager full-C
    /// KV allocation or large device copy precedes sealing.
    pub fn from_snapshot_seed(snapshot: Arc<ModelSnapshot<S>>, source: Source,
        layers: &[Id], device: CudaDevice, limits: Limits, ids: &[u32]) -> Result<Self>
    {
        if ids.is_empty() || ids.len() > limits.capacity {
            return Err("empty/oversized prepared prefix".into());
        }
        source::load(snapshot, source, layers, device, limits, Some(Arc::from(ids)),
            #[cfg(test)] |_, _, _| {},
        )
    }
    pub fn from_device_model_seed(model: Arc<DeviceModel<S>>, limits: Limits, ids: &[u32]) -> Result<Self> {
        validate_seed(&model.config, limits, Some(ids))?;
        Self::from_state(model, limits, Some(Arc::from(ids)), None)
    }
    /// Exact same uploaded model/client/device and native execution profile.
    /// Forks use the sealed origin stream, not concurrent kernel scheduling.
    pub fn from_prefix(model: Arc<DeviceModel<S>>, prefix: Arc<Prefix<S>>, limits: Limits) -> Result<Self> {
        Self::from_state(model, limits, None, Some(prefix))
    }
    pub fn saved_prefix(&self) -> Option<&Arc<Prefix<S>>> {
        (self.status == Status::Ready && self.seed.is_none()).then_some(self.prefix.as_ref()).flatten()
    }
    pub fn prefix_ownership(&self) -> Result<PrefixOwnership> {
        let mut result = PrefixOwnership { prefix_tokens: 0, shared_kv_bytes: 0,
            private_kv_bytes: 0, shared_mamba_bytes: 0, private_mamba_bytes: 0 };
        if let Some(prefix) = &self.prefix {
            result = ownership(&self.model.config, self.limits, prefix.ids.len())?;
            result.private_kv_bytes = 0;
            result.private_mamba_bytes = 0;
        }
        let c = mamba_config(&self.model.config).extents(1, 1)?;
        let state_bytes = add(bytes(&[c.conv], 2)?, bytes(&[c.recurrent], 4)?)?;
        for (index, layer) in self.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Attention(a) => result.private_kv_bytes = add(result.private_kv_bytes, a.private_kv_bytes())?,
                Mixer::Mamba(m) => if let Some(state) = &m.committed {
                    let shared = self.prefix.as_ref().and_then(|p| p.mamba(index))
                        .is_some_and(|boundary| Arc::ptr_eq(state, &boundary));
                    if !shared { result.private_mamba_bytes = add(result.private_mamba_bytes, state_bytes)?; }
                },
                Mixer::Moe(_) => {}
            }
        }
        Ok(result)
    }
    fn observe_prefix(&self, outcome: &'static str, reused: usize) {
        if let Ok(bytes) = self.prefix_ownership() {
            // No IDs, prompt text or hashes. The fields describe only this
            // completed transition, not allocator reservation/physical fit.
            let _observation = tracing::info_span!(target: "mary.nemotron.progress", "native_prefix",
                outcome, reused_prefix_tokens = reused, saved_prefix_tokens = bytes.prefix_tokens,
                shared_kv_bytes = bytes.shared_kv_bytes, private_kv_bytes = bytes.private_kv_bytes,
                shared_mamba_bytes = bytes.shared_mamba_bytes, private_mamba_bytes = bytes.private_mamba_bytes,
                session_ready = self.status == Status::Ready).entered();
        }
    }
    pub(super) fn observe_reuse(&self) {
        if self.status == Status::Ready {
            let outcome = if self.prefix.is_some() { "continued" }
                else if self.seed.is_some() { "seed_pending" } else { "no_saved_prefix" };
            self.observe_prefix(outcome, self.prefix.as_ref().map_or(0, |p| p.ids.len()));
        }
    }
    pub(super) fn publish_seed_if_complete(&mut self) -> Result<()> {
        let Some(seed) = self.seed.clone() else { return Ok(()) };
        if self.committed.len() != seed.len() { return Ok(()) }
        self.stream.executes(|| {
            if self.status != Status::Ready || self.span_pending.is_some() || !self.pending.is_empty() {
                return Err("prefix publication needs a completed hybrid boundary".into());
            }
            self.check_boundary(seed.len())?;
            self.status = Status::Poisoned;
            #[cfg(test)]
            if self.fault == Some(Fault::PrefixUpload) {
                tests::enqueue_rejected_upload(&self.model.device, self.model.embedding.handle.clone());
            }
            self.fence()?; // origin error/completion before ANY shareable owner
            let layers = self.layers.iter().map(|layer| Ok(match &layer.mixer {
                Mixer::Mamba(m) => Boundary::Mamba(m.committed.clone().ok_or("missing Mamba boundary")?),
                Mixer::Attention(a) => Boundary::Attention(a.seal_kv()?),
                Mixer::Moe(_) => Boundary::Stateless,
            })).collect::<Result<Vec<_>>>()?;
            let prefix = Arc::new(Prefix { model: self.model.clone(), stream: self.stream,
                ids: seed, logits: Arc::from(self.logits.as_slice()), layers });
            // Retain sealed inputs even if private suffix allocation fails or
            // unwinds. saved_prefix cannot expose an incomplete transition.
            self.prefix = Some(prefix.clone());
            for (index, layer) in self.layers.iter_mut().enumerate() {
                if let Mixer::Attention(a) = &mut layer.mixer {
                    *a = attention::Attention::from_prefix(a.weights.clone(), prefix.attention(index)?,
                        attention_limits(&self.model.config, self.limits))?;
                }
            }
            self.fence()?;
            self.seed = None;
            self.status = Status::Ready;
            self.observe_prefix("published", 0);
            Ok(())
        })
    }

    /// Restore only the saved P boundary, never an inferred shorter prefix.
    /// Failure leaves the private owner poisoned and all uncertain operands
    /// retained. A fatal shared CUDA-context failure is not recovered here.
    pub fn restore_prefix(&mut self) -> Result<()> {
        let prefix = self.prefix.clone().filter(|_| self.seed.is_none())
            .ok_or("no successfully published prefix")?;
        self.stream.executes(|| {
            self.invalidate();
            let mut errors = Vec::new();
            if let Err(e) = self.fence() { errors.push(e); }
            #[cfg(test)]
            if let Err(e) = inject(self.fault, Fault::Reset) { errors.push(e); }
            for layer in &mut self.layers {
                let result = match &mut layer.mixer {
                    Mixer::Attention(a) => a.restore_prefix(),
                    Mixer::Moe(m) => m.reset(),
                    Mixer::Mamba(_) => Ok(()),
                };
                if let Err(e) = result { errors.push(e); }
            }
            if !errors.is_empty() { self.invalidate(); return Err(errors.join("; ")); }
            for (index, layer) in self.layers.iter_mut().enumerate() {
                if let Mixer::Mamba(m) = &mut layer.mixer {
                    m.committed = prefix.mamba(index);
                    m.staged = None;
                }
            }
            self.pending.clear();
            self.span_pending = None;
            self.committed = prefix.ids.to_vec();
            self.logits = prefix.logits.to_vec();
            self.status = Status::Ready;
            self.observe_prefix("restored", prefix.ids.len());
            Ok(())
        })
    }
    /// Complete caller-side prompt validation precedes reset/restore. An exact
    /// seed prompt uses saved last_logits; a mismatch gets full-C empty backing.
    pub fn reset_for_prompt(&mut self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() || ids.len() > self.capacity() || ids.iter().any(|&id| id as usize >= self.vocab_size()) {
            return Err("invalid prompt before prefix reset".into());
        }
        if self.seed.is_none() && self.prefix.as_ref().is_some_and(|p| ids.starts_with(&p.ids)) {
            self.restore_prefix()
        } else {
            let reason = if self.prefix.is_some() || self.seed.is_some() { "token_mismatch" } else { "no_saved_prefix" };
            self.reset()?;
            self.observe_prefix(reason, 0);
            Ok(())
        }
    }
}
