//! Portable values at the existing completed common-memory prefix boundary.
//! This is neither a pile schema nor a second transport. Typed payload handles
//! and inspectable metadata are the future collection adapter's ordinary facts.
//! Keep them under the source's private access/custody policy, including handles.
use super::*;
use super::prefix::Boundary;
use crate::leaf;
use triblespace::core::blob::{Blob, encodings::tensor::{Tensor, TensorElement, TensorView,
    elements::{BF16, F32}}};
use triblespace::core::inline::{Inline, encodings::hash::Handle as BlobHandle};

/// The state interpretation implemented here, not a general cross-device
/// numerical-equivalence promise. Caller policy must approve the producer's
/// arithmetic/kernel profile for the destination execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixExecution { NativeB1Bf16KvF32Mamba }

/// Exact loader selections, not an authenticated portable operand fingerprint.
/// DeviceModel retains a frozen ModelSnapshot; it does not retain a portable
/// descriptor of every lazily selected expert. IDs are opaque lookup keys.
#[derive(Clone)]
pub struct PrefixProvenance {
    pub source: Source,
    pub config: TextConfig,
    pub layers: Vec<Id>,
    pub execution: PrefixExecution,
}

#[derive(Clone)]
pub struct KvChunk {
    pub start: usize,
    pub rows: usize,
    /// Contiguous [rows, KV heads=2, head dimension=128], native BF16 bits.
    pub keys: Inline<BlobHandle<Tensor<BF16, 3>>>,
    pub values: Inline<BlobHandle<Tensor<BF16, 3>>>,
}

#[derive(Clone)]
pub enum PrefixLayer {
    Attention(Vec<KvChunk>),
    Mamba {
        position: usize,
        /// [1, conv_channels, conv_kernel], oldest sample first.
        conv: Inline<BlobHandle<Tensor<BF16, 3>>>,
        /// [1, heads, head_dim, state_dim], F32, without conversion to BF16.
        recurrent: Inline<BlobHandle<Tensor<F32, 4>>>,
    },
    Stateless,
}

/// One selected checkpoint's essential values, not a catalogue of checkpoints.
/// Payloads remain in the caller's store; no whole-KV host assembly is required.
/// Public fields permit construction from typed pile queries. A malformed or
/// incomplete selection is rejected by validate/restore, not made Ready.
#[derive(Clone)]
pub struct PrefixCheckpoint {
    pub provenance: PrefixProvenance,
    pub position: usize,
    pub tokens: Arc<[u32]>,
    pub last_logits: Inline<BlobHandle<Tensor<F32, 1>>>,
    pub layers: Vec<PrefixLayer>,
}

fn put<T: TensorElement, const R: usize, B: BlobStorePut>(store: &mut B,
    dims: [usize; R], payload: anybytes::Bytes) -> Result<Inline<BlobHandle<Tensor<T, R>>>>
{
    let expected = bytes(&dims, T::BITS / 8)?;
    if payload.len() != expected { return Err("checkpoint payload extent differs".into()); }
    let blob = leaf::leaf_blob::<T, R>(dims.map(|n| n as u64), payload).map_err(|e| e.to_string())?;
    store.put(blob).map_err(|e| format!("checkpoint put: {e}"))
}

/// The shared typed decoder validates all extent arithmetic before exposing
/// its owning view. This layer checks only the model's expected geometry.
fn read<T: TensorElement, const R: usize, B: BlobStoreGet>(store: &B,
    handle: Inline<BlobHandle<Tensor<T, R>>>, dims: [usize; R]) -> Result<TensorView>
{
    let blob: Blob<Tensor<T, R>> = store.get(handle).map_err(|e| format!("checkpoint get: {e}"))?;
    let view = leaf::read_leaf(blob).map_err(|e| e.to_string())?;
    if view.dims() != dims.map(|d| d as u64).as_slice() {
        return Err("checkpoint typed shape differs".into());
    }
    Ok(view)
}

impl PrefixCheckpoint {
    /// CPU-only structural and complete typed-payload validation. This does NOT
    /// verify operand equivalence, authority, durability or device readiness.
    /// The exact token slice is the caller's selected common prefix, not the
    /// full divergent prompt. A later Mamba boundary cannot serve a shorter one.
    pub fn validate<B: BlobStoreGet>(&self, store: &B, config: &TextConfig,
        layer_ids: &[Id], limits: Limits, expected_tokens: &[u32]) -> Result<()>
    {
        validate_config(config, limits)?;
        resident_ownership(config, limits)?;
        prefix::validate_seed(config, limits, Some(&self.tokens))?;
        if self.position != self.tokens.len() || self.tokens.as_ref() != expected_tokens {
            return Err("checkpoint token sequence or absolute position differs".into());
        }
        if self.provenance.config != *config || self.provenance.layers != layer_ids
            || self.layers.len() != config.layers_block_type.len()
            || layer_ids.len() != self.layers.len()
            || self.provenance.execution != PrefixExecution::NativeB1Bf16KvF32Mamba {
            return Err("checkpoint config/layer order/execution contract differs".into());
        }
        for (kind, layer) in config.layers_block_type.iter().zip(&self.layers) {
            match (kind, layer) {
                (LayerKind::Attention, PrefixLayer::Attention(chunks)) => {
                    let mut next = 0;
                    for chunk in chunks {
                        let end = chunk.start.checked_add(chunk.rows).ok_or("checkpoint KV range overflow")?;
                        if chunk.rows == 0 || chunk.start != next || end > self.position {
                            return Err("checkpoint KV has gap/overlap/empty/outside range".into());
                        }
                        let shape = [chunk.rows, attention::KV_HEADS, attention::HEAD_DIM];
                        read::<BF16, 3, _>(store, chunk.keys, shape)?;
                        read::<BF16, 3, _>(store, chunk.values, shape)?;
                        next = end;
                    }
                    if next != self.position { return Err("incomplete checkpoint KV prefix".into()); }
                }
                (LayerKind::Mamba, PrefixLayer::Mamba { position, conv, recurrent }) => {
                    if *position != self.position { return Err("checkpoint Mamba boundary differs".into()); }
                    let m = mamba_config(config);
                    read::<BF16, 3, _>(store, *conv, [1, m.layout()?.conv_channels, m.conv_kernel])?;
                    read::<F32, 4, _>(store, *recurrent, [1, m.heads, m.head_dim, m.state_dim])?;
                }
                (LayerKind::Moe, PrefixLayer::Stateless) => {}
                _ => return Err("checkpoint layer kind differs".into()),
            }
        }
        let logits = read::<F32, 1, _>(store, self.last_logits, [config.vocab_size])?;
        if logits.payload().chunks_exact(4).any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite()) {
            return Err("nonfinite checkpoint last logits".into());
        }
        Ok(())
    }
}

fn fence(client: &ComputeClient<CudaRuntime>) -> Result<()> {
    cubecl::future::block_on(client.sync()).map_err(|e| format!("checkpoint CUDA completion: {e:?}"))
}

impl<S> Prefix<S> {
    /// Export only this already completed immutable boundary. Successful put
    /// does not publish checkpoint facts or imply durable flush/custody. Failed
    /// export leaves prior checkpoints unchanged; partial blobs may be retained.
    /// Store each owning typed chunk immediately: temporary host memory scales
    /// with chunk_rows (plus one Mamba boundary), not the full attention prefix.
    /// Tensor construction currently copies its header/payload once.
    pub fn export<B: BlobStorePut>(&self, store: &mut B, chunk_rows: usize) -> Result<PrefixCheckpoint> {
        if chunk_rows == 0 { return Err("checkpoint chunk rows must be positive".into()); }
        self.stream.executes(|| {
            fence(&self.model.client)?;
            let mut layers = Vec::with_capacity(self.layers.len());
            for boundary in &self.layers {
                layers.push(match boundary {
                    Boundary::Attention(kv) => {
                        let mut chunks = Vec::new();
                        let mut start = 0;
                        while start < self.ids.len() {
                            let rows = chunk_rows.min(self.ids.len() - start);
                            let [keys, values] = kv.range(start, rows)?;
                            let mut capture = |handle| -> Result<_> {
                                let data = self.model.client.read_one(handle).map_err(|e| format!("checkpoint KV read: {e:?}"))?;
                                put::<BF16, 3, _>(store, [rows, attention::KV_HEADS, attention::HEAD_DIM],
                                    anybytes::Bytes::from_source(data.to_vec()))
                            };
                            chunks.push(KvChunk { start, rows, keys: capture(keys)?, values: capture(values)? });
                            start += rows;
                        }
                        PrefixLayer::Attention(chunks)
                    }
                    Boundary::Mamba(state) => {
                        if state.attended() != self.ids.len() { return Err("Mamba export boundary differs".into()); }
                        let m = mamba_config(&self.model.config);
                        let conv = self.model.client.read_one(state.conv().handle.clone())
                            .map_err(|e| format!("checkpoint convolution read: {e:?}"))?;
                        let conv = put::<BF16, 3, _>(store, [1, m.layout()?.conv_channels, m.conv_kernel],
                            anybytes::Bytes::from_source(conv.to_vec()))?;
                        let recurrent = self.model.client.read_one(state.recurrent().handle.clone())
                            .map_err(|e| format!("checkpoint recurrence read: {e:?}"))?;
                        let recurrent = put::<F32, 4, _>(store, [1, m.heads, m.head_dim, m.state_dim],
                            anybytes::Bytes::from_source(recurrent.to_vec()))?;
                        PrefixLayer::Mamba { position: state.attended(), conv, recurrent }
                    }
                    Boundary::Stateless => PrefixLayer::Stateless,
                });
            }
            let logits: Vec<u8> = self.logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            let last_logits = put::<F32, 1, _>(store, [self.logits.len()], anybytes::Bytes::from_source(logits))?;
            fence(&self.model.client)?;
            Ok(PrefixCheckpoint {
                provenance: PrefixProvenance { source: self.model.source, config: self.model.config.clone(),
                    layers: self.model.layer_ids().collect(), execution: PrefixExecution::NativeB1Bf16KvF32Mamba },
                position: self.ids.len(), tokens: self.ids.clone(), last_logits, layers,
            })
        })
    }
}

impl<S: BlobStoreGet> Session<S> {
    /// Replace this context with the observed checkpoint on its existing model.
    /// Its completed Prefix is then available through saved_prefix/from_prefix.
    /// The mandatory caller policy must approve actual selected operand content,
    /// codec/token meaning and compatible producer/destination arithmetic. The
    /// supplied provenance and DeviceModel::snapshot/source/config/layer_ids
    /// are inspectable for that query-boundary decision; ID equality by itself
    /// is NOT such verification. This is a numerical trust contract, not Rust
    /// memory unsafety or authentication. No general cross-device parity claim.
    ///
    /// Policy, config/position and every payload are checked before mutation or
    /// allocation. Subsequent error/unwind leaves this Session poisoned, with
    /// staged device tensors in its existing pending owner list. reset() can
    /// release them only after successful fences; dropping an error does not.
    /// StreamId::executes restores the caller's stream even during unwinding.
    ///
    /// Host views are never borrowed by CUDA: create/write owns a staging copy
    /// in the runtime's stream drop queue. Ordinary Session Drop retains the
    /// existing managed-allocation semantics, NOT proof of a drained/lost CUDA
    /// context. No external/mapped pointer is registered or freed here.
    ///
    /// Old context overlaps imported P rows. After successful import fencing,
    /// attention layers replace old buffers one at a time with C-P rows; no
    /// second full-C context or second expert cache is constructed. Limits keep
    /// their logical meaning, not physical admission for temporary overlap.
    pub fn restore_checkpoint<B: BlobStoreGet>(&mut self, checkpoint: &PrefixCheckpoint,
        store: &B, expected_tokens: &[u32],
        caller_policy: impl FnOnce(&PrefixProvenance, &DeviceModel<S>) -> Result<()>) -> Result<()>
    {
        self.restore_checkpoint_inner(checkpoint, store, expected_tokens, caller_policy,
            #[cfg(test)] |_, _| {},
        )
    }

    fn restore_checkpoint_inner<B: BlobStoreGet>(&mut self, checkpoint: &PrefixCheckpoint,
        store: &B, expected_tokens: &[u32],
        caller_policy: impl FnOnce(&PrefixProvenance, &DeviceModel<S>) -> Result<()>,
        #[cfg(test)] before_fence: impl FnOnce(&DeviceModel<S>, &[Boundary]),
    ) -> Result<()> {
        caller_policy(&checkpoint.provenance, &self.model)?;
        checkpoint.validate(store, &self.model.config, &self.model.layer_ids().collect::<Vec<_>>(), self.limits, expected_tokens)?;
        let stream = self.stream;
        stream.executes(|| {
            self.invalidate();
            self.fence()?;
            self.pending.clear();
            self.span_pending = None;
            let model = self.model.clone();
            let mut layers = Vec::with_capacity(model.layers.len());
            for (weights, state) in model.layers.iter().zip(&checkpoint.layers) {
                let boundary = match (&weights.mixer, state) {
                    (MixerWeights::Attention(w), PrefixLayer::Attention(chunks)) => {
                        let size = bytes(&[checkpoint.position, attention::KV_HEADS, attention::HEAD_DIM], 2)?;
                        let shape = [checkpoint.position, attention::KV_HEADS, attention::HEAD_DIM];
                        let keys = CudaTensor::new_contiguous(model.client.clone(), model.device.clone(), shape.into(),
                            model.client.empty(size), DType::BF16);
                        self.pending.push(keys.clone());
                        let values = CudaTensor::new_contiguous(model.client.clone(), model.device.clone(), shape.into(),
                            model.client.empty(size), DType::BF16);
                        self.pending.push(values.clone());
                        // Session owns destinations across every failed fence
                        // or unwind, independently of these local wrappers.
                        let kv = attention::KvPrefix::imported(w.clone(), checkpoint.position, keys.handle.clone(), values.handle.clone())?;
                        for chunk in chunks {
                            let targets = kv.range(chunk.start, chunk.rows)?;
                            for (target, handle) in targets.into_iter().zip([chunk.keys, chunk.values]) {
                                let view = read::<BF16, 3, _>(store, handle, [chunk.rows, attention::KV_HEADS, attention::HEAD_DIM])?;
                                write(&model, target, view.payload())?;
                            }
                        }
                        Boundary::Attention(kv)
                    }
                    (MixerWeights::Mamba(_), PrefixLayer::Mamba { position, conv, recurrent }) => {
                        let m = mamba_config(&model.config);
                        let conv_shape = [1, m.layout()?.conv_channels, m.conv_kernel];
                        let recurrent_shape = [1, m.heads, m.head_dim, m.state_dim];
                        let conv = read::<BF16, 3, _>(store, *conv, conv_shape)?;
                        let conv = CudaTensor::new_contiguous(model.client.clone(), model.device.clone(), conv_shape.into(),
                            model.client.create_from_slice(conv.payload()), DType::BF16);
                        self.pending.push(conv.clone());
                        // Fence each bounded staging operation. A failed sync
                        // is not called a drain; managed owners retain normal
                        // CubeCL stream lifetime, with no external pointer free.
                        fence(&model.client)?;
                        let recurrent = read::<F32, 4, _>(store, *recurrent, recurrent_shape)?;
                        let recurrent = CudaTensor::new_contiguous(model.client.clone(), model.device.clone(), recurrent_shape.into(),
                            model.client.create_from_slice(recurrent.payload()), DType::F32);
                        self.pending.push(recurrent.clone());
                        fence(&model.client)?;
                        Boundary::Mamba(Arc::new(gpu::State::imported(m, *position, conv, recurrent)?))
                    }
                    (MixerWeights::Moe(_), PrefixLayer::Stateless) => Boundary::Stateless,
                    _ => return Err("checkpoint layer kind differs from uploaded owner".into()),
                };
                layers.push(boundary);
            }
            let view = read::<F32, 1, _>(store, checkpoint.last_logits, [model.config.vocab_size])?;
            let logits: Vec<f32> = view.payload().chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect();
            #[cfg(test)] before_fence(&model, &layers);
            fence(&model.client)?;
            let prefix = Arc::new(Prefix { model: model.clone(), stream, ids: checkpoint.tokens.clone(),
                logits: Arc::from(logits), layers });
            // Retain imported operands even if private suffix allocation fails
            // or unwinds. Replace one layer at a time, not a second full-C
            // Session. Existing private expert residency/quotas stay owned.
            self.prefix = Some(prefix.clone());
            for (index, layer) in self.layers.iter_mut().enumerate() {
                match &mut layer.mixer {
                    Mixer::Mamba(m) => { m.committed = prefix.mamba(index); m.staged = None; }
                    Mixer::Attention(a) => {
                        *a = attention::Attention::from_prefix(a.weights.clone(), prefix.attention(index)?,
                            attention_limits(&model.config, self.limits))?;
                    }
                    Mixer::Moe(m) => m.reset()?,
                }
            }
            fence(&model.client)?;
            self.pending.clear();
            self.seed = None;
            self.committed = prefix.ids.to_vec();
            self.logits = prefix.logits.to_vec();
            self.status = Status::Ready;
            self.observe_prefix("imported", prefix.ids.len());
            Ok(())
        })
    }
}

/// Same owning server.write path used by ComputeClient::create, targeting a
/// validated range of already allocated contiguous KV. No raw CUDA pointer or
/// new copy kernel. The server owns staging bytes and the binding through its
/// queued write; the caller retains the full destination and observes errors
/// on THIS stream. Per-chunk fence bounds queued host staging.
fn write<S>(model: &DeviceModel<S>, target: cubecl::server::Handle, payload: &anybytes::Bytes) -> Result<()> {
    use cubecl::{device::Device, device_handle::DeviceHandle, server::{ComputeServer, CopyDescriptor}};
    if target.size_in_used() != payload.len() as u64 { return Err("checkpoint upload range differs".into()); }
    let stream = StreamId::current();
    let descriptor = CopyDescriptor::new(target.clone().binding(), [payload.len()].into(), [1].into(), 1);
    let data = cubecl::bytes::Bytes::from_bytes_vec(payload.to_vec());
    let service = DeviceHandle::<<CudaRuntime as Runtime>::Server>::new(model.device.to_id());
    service.submit_blocking(move |server| server.write(vec![(descriptor, data)], stream))
        .map_err(|e| format!("checkpoint upload submission: {e:?}"))?;
    fence(&model.client)
}

#[cfg(test)]
#[path = "session_checkpoint_tests.rs"]
mod tests;
