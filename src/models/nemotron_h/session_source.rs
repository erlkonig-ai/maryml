//! Private query-at-use loader. Retains execution operands, not a graph mirror.
use super::super::{ingress::tokenizer_binding, pile};
use super::*;
use crate::{format::attrs, jsonfacts, tokenizer};
use triblespace::core::blob::encodings::tensor::{
    TensorElement, TensorView,
    elements::{BF16, F32},
};

pub(super) struct NativeExperts<S> {
    snapshot: Arc<ModelSnapshot<S>>,
    source: moe::SourceBinding,
}
impl<S: BlobStoreGet> moe::ExpertProvider for NativeExperts<S> {
    fn source(&self) -> moe::SourceBinding {
        self.source
    }
    fn with_expert<T>(
        &self,
        index: usize,
        use_pair: impl FnOnce(moe::ExpertPair<'_>) -> Result<T>,
    ) -> Result<T> {
        // Choose only among complete typed expert/up/down/scale tuples. Missing
        // typed facts are negative matches; a selected missing blob is an error.
        let (_, up, down) =
            pile::nvfp4_expert_minimum(&self.snapshot, self.source.layer, index as u64)
                .ok_or_else(|| format!("missing complete native expert {index}"))?
                .map_err(|e| e.to_string())?;
        shape(&up.weight, &[2688, 1024], "moe.up_proj")?;
        shape(&down.weight, &[1024, 2688], "moe.down_proj")?;
        use_pair(moe::ExpertPair {
            index,
            up: packed_op(&up)?,
            down: packed_op(&down)?,
        })
    }
}
fn first<T>(mut values: impl Iterator<Item = anyhow::Result<T>>, role: &str) -> Result<T> {
    values
        .next()
        .ok_or_else(|| format!("missing typed native {role}"))?
        .map_err(|e| e.to_string())
}
fn aux<T: TensorElement, const R: usize, S: BlobStoreGet>(
    s: &ModelSnapshot<S>,
    parent: Id,
    role: &str,
    dims: &[usize],
) -> Result<(Id, TensorView)> {
    let selected = first(pile::auxiliary_operations::<T, R, S>(s, parent, role), role)?;
    shape(&selected.1, dims, role)?;
    Ok(selected)
}
fn shape(view: &TensorView, dims: &[usize], role: &str) -> Result<()> {
    if view
        .dims()
        .iter()
        .copied()
        .ne(dims.iter().map(|&n| n as u64))
    {
        return Err(format!(
            "native {role} shape differs from execution geometry"
        ));
    }
    Ok(())
}
fn scalar(view: &TensorView) -> Result<f32> {
    shape(view, &[], "scale")?;
    let data: [u8; 4] = view
        .payload()
        .as_ref()
        .try_into()
        .map_err(|_| "invalid scalar F32 payload")?;
    let value = f32::from_le_bytes(data);
    if !value.is_finite() || value <= 0.0 {
        return Err("native scale must be finite positive".into());
    }
    Ok(value)
}
fn fp8<S: BlobStoreGet>(
    s: &ModelSnapshot<S>,
    parent: Id,
    role: &str,
    dims: [usize; 2],
) -> Result<pile::Fp8Projection> {
    let selected = first(pile::fp8(s, parent, role), role)?;
    shape(&selected.weight, &dims, role)?;
    if selected.weight.payload().iter().any(|b| b & 0x7f == 0x7f) {
        return Err(format!("nonfinite native FP8 {role}"));
    }
    scalar(&selected.input_scale)?;
    scalar(&selected.weight_scale)?;
    Ok(selected)
}
fn project(op: &pile::Fp8Projection, device: &CudaDevice, limit: usize) -> Result<Projection> {
    let d = op.weight.dims();
    Projection::upload(
        device.clone(),
        [d[0] as usize, d[1] as usize],
        op.weight.payload(),
        scalar(&op.input_scale)?,
        scalar(&op.weight_scale)?,
        limit,
    )
}
fn attention_op(p: &pile::Fp8Projection) -> Result<attention::Operation<'_>> {
    Ok(attention::Operation {
        e4m3fn: p.weight.payload(),
        input_scale: scalar(&p.input_scale)?,
        weight_scale: scalar(&p.weight_scale)?,
    })
}
fn packed<S: BlobStoreGet>(
    s: &ModelSnapshot<S>,
    parent: Id,
    role: &str,
    dims: [usize; 2],
) -> Result<pile::Nvfp4Projection> {
    let selected = pile::nvfp4_minimum(s, parent, role)
        .ok_or_else(|| format!("missing typed native {role}"))?
        .map_err(|e| e.to_string())?;
    shape(&selected.weight, &dims, role)?;
    scalar(&selected.input_scale)?;
    Ok(selected)
}
fn packed_op(op: &pile::Nvfp4Projection) -> Result<moe::PackedOperation<'_>> {
    Ok(moe::PackedOperation {
        operation: op.operation,
        payload: op.weight.payload(),
        input_scale: scalar(&op.input_scale)?,
    })
}
fn upload(
    view: &TensorView,
    dtype: DType,
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
) -> Result<CudaTensor> {
    let width = if dtype == DType::BF16 { 2 } else { 4 };
    let dims: Vec<usize> = view
        .dims()
        .iter()
        .map(|&n| usize::try_from(n).map_err(|_| "native extent exceeds host usize".to_string()))
        .collect::<Result<_>>()?;
    if view.payload().len() != bytes(&dims, width)? {
        return Err("native auxiliary byte extent".into());
    }
    let finite = if width == 2 {
        view.payload()
            .chunks_exact(2)
            .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80)
    } else {
        view.payload()
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
    };
    if !finite {
        return Err("nonfinite native auxiliary".into());
    }
    Ok(CudaTensor::new_contiguous(
        client.clone(),
        device.clone(),
        dims.into(),
        client.create_from_slice(view.payload()),
        dtype,
    ))
}
fn bf16_aux<const R: usize, S: BlobStoreGet>(
    s: &ModelSnapshot<S>,
    p: Id,
    role: &str,
    dims: &[usize],
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
) -> Result<CudaTensor> {
    upload(
        &aux::<BF16, R, S>(s, p, role, dims)?.1,
        DType::BF16,
        client,
        device,
    )
}
fn f32_values<S: BlobStoreGet>(
    s: &ModelSnapshot<S>,
    p: Id,
    role: &str,
    n: usize,
) -> Result<(Id, Vec<f32>)> {
    // Native aux sources may be BF16 or F32. Exact BF16->F32 promotion is a
    // parameter decoding boundary, not a CPU projection/inference path.
    if let Some(value) = pile::auxiliary_operations::<F32, 1, S>(s, p, role).next() {
        let (id, v) = value.map_err(|e| e.to_string())?;
        shape(&v, &[n], role)?;
        let values: Vec<f32> = v
            .payload()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("nonfinite {role}"));
        }
        return Ok((id, values));
    }
    let (id, v) = aux::<BF16, 1, S>(s, p, role, &[n])?;
    let values: Vec<f32> = v
        .payload()
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("nonfinite {role}"));
    }
    Ok((id, values))
}
fn f32_aux<S: BlobStoreGet>(
    s: &ModelSnapshot<S>,
    p: Id,
    role: &str,
    n: usize,
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
) -> Result<CudaTensor> {
    let (_, values) = f32_values(s, p, role, n)?;
    Ok(CudaTensor::new_contiguous(
        client.clone(),
        device.clone(),
        [n].into(),
        client.create_from_slice(f32::as_bytes(&values)),
        DType::F32,
    ))
}

pub(super) fn load<S: BlobStoreGet>(
    snapshot: Arc<ModelSnapshot<S>>,
    source: Source,
    layer_ids: &[Id],
    device: CudaDevice,
    limits: Limits,
) -> Result<Session<S>> {
    let facts = snapshot.facts();
    let root = source.root;
    let config = source.config;
    let binding = source.tokenizer_binding;
    if !exists!(pattern!(facts,[
        {root @ attrs::member:_?config, tokenizer::attrs::tokenizer:binding},
        {_?config @ attrs::kind:"nemotron-h-config",attrs::member:config}
    ])) || !tokenizer_binding::bases(facts, binding).any(|id| id == source.tokenizer_base)
        || !tokenizer_binding::pre_tokenizers(facts, binding).any(|id| id == source.pre_tokenizer)
    {
        return Err("selected config/tokenizer tuple is not linked in the held observation".into());
    }
    let json = jsonfacts::load_json(facts, snapshot.store(), config).map_err(|e| e.to_string())?;
    let config: TextConfig = serde_json::from_value(
        json.get("text")
            .cloned()
            .ok_or("native config has no text section")?,
    )
    .map_err(|e| e.to_string())?;
    validate_config(&config, limits)?;
    let cache_pairs = expert_cache_pairs(
        resident_bytes(&config, limits)?,
        limits.max_resident_bytes,
        config.layers_block_type.iter().filter(|kind| matches!(kind, LayerKind::Moe)).count(),
    )?;
    if layer_ids.len() != config.layers_block_type.len() {
        return Err("selected layer sequence length differs from config".into());
    }
    for (i, &layer) in layer_ids.iter().enumerate() {
        if !pile::layers(&snapshot, source.root, i as u64).any(|id| id == layer) {
            return Err(format!("selected layer is not at native position {i}"));
        }
    }
    let client = CudaRuntime::client(&device);
    let embedding = bf16_aux::<2, S>(
        &snapshot,
        source.root,
        "embedding",
        &[config.vocab_size, H],
        &client,
        &device,
    )?;
    let final_norm =
        bf16_aux::<1, S>(&snapshot, source.root, "final-norm", &[H], &client, &device)?;
    let head = project(
        &fp8(&snapshot, source.root, "lm-head", [config.vocab_size, H])?,
        &device,
        limits.max_resident_bytes,
    )?;
    let mut layers = Vec::with_capacity(layer_ids.len());
    for (&id, kind) in layer_ids.iter().zip(&config.layers_block_type) {
        let norm = bf16_aux::<1, S>(&snapshot, id, "pre-norm", &[H], &client, &device)?;
        let mixer = match kind {
            LayerKind::Mamba => {
                let m = mamba_config(&config);
                let g = m.layout()?;
                Mixer::Mamba(Mamba {
                    config: m,
                    input: project(
                        &fp8(&snapshot, id, "mamba.in_proj", [g.projected, H])?,
                        &device,
                        limits.max_resident_bytes,
                    )?,
                    output: project(
                        &fp8(&snapshot, id, "mamba.out_proj", [H, g.inner])?,
                        &device,
                        limits.max_resident_bytes,
                    )?,
                    conv: bf16_aux::<3, S>(
                        &snapshot,
                        id,
                        "mamba.conv1d.weight",
                        &[g.conv_channels, 1, m.conv_kernel],
                        &client,
                        &device,
                    )?,
                    bias: if config.use_conv_bias {
                        Some(bf16_aux::<1, S>(
                            &snapshot,
                            id,
                            "mamba.conv1d.bias",
                            &[g.conv_channels],
                            &client,
                            &device,
                        )?)
                    } else {
                        None
                    },
                    a_log: f32_aux(&snapshot, id, "mamba.A_log", m.heads, &client, &device)?,
                    dt: f32_aux(&snapshot, id, "mamba.dt_bias", m.heads, &client, &device)?,
                    d: f32_aux(&snapshot, id, "mamba.D", m.heads, &client, &device)?,
                    gain: bf16_aux::<1, S>(
                        &snapshot,
                        id,
                        "mamba.norm",
                        &[g.inner],
                        &client,
                        &device,
                    )?,
                    committed: None,
                    staged: None,
                })
            }
            LayerKind::Attention => {
                let q = fp8(&snapshot, id, "attention.q_proj", [H, H])?;
                let k = fp8(&snapshot, id, "attention.k_proj", [256, H])?;
                let v = fp8(&snapshot, id, "attention.v_proj", [256, H])?;
                let o = fp8(&snapshot, id, "attention.o_proj", [H, H])?;
                Mixer::Attention(attention::Attention::upload(
                    device.clone(),
                    attention::SourceBinding {
                        model_root: source.root,
                        layer: id,
                        operations: [q.operation, k.operation, v.operation, o.operation],
                    },
                    [
                        attention_op(&q)?,
                        attention_op(&k)?,
                        attention_op(&v)?,
                        attention_op(&o)?,
                    ],
                    attention_limits(&config, limits),
                )?)
            }
            LayerKind::Moe => {
                let router = aux::<BF16, 2, S>(&snapshot, id, "moe.router", &[512, H])?;
                let (bias_id, bias) = f32_values(&snapshot, id, "moe.router-bias", 512)?;
                let fc1 = aux::<BF16, 2, S>(&snapshot, id, "moe.fc1_latent_proj", &[1024, H])?;
                let fc2 = aux::<BF16, 2, S>(&snapshot, id, "moe.fc2_latent_proj", &[H, 1024])?;
                let up = packed(&snapshot, id, "moe.shared.up_proj", [5376, H])?;
                let down = packed(&snapshot, id, "moe.shared.down_proj", [H, 5376])?;
                let binding = moe::SourceBinding {
                    model_root: source.root,
                    layer: id,
                };
                Mixer::Moe(moe::Moe::upload_with_expert_cache(
                    device.clone(),
                    binding,
                    moe::FixedOperands {
                        operations: [router.0, bias_id, fc1.0, fc2.0],
                        router_bf16: router.1.payload(),
                        correction_bias: &bias,
                        fc1_bf16: fc1.1.payload(),
                        fc2_bf16: fc2.1.payload(),
                        shared_up: packed_op(&up)?,
                        shared_down: packed_op(&down)?,
                    },
                    NativeExperts {
                        snapshot: snapshot.clone(),
                        source: binding,
                    },
                    moe_limits(limits),
                    cache_pairs,
                )?)
            }
        };
        layers.push(Layer { id, norm, mixer });
    }
    Ok(Session {
        snapshot,
        source,
        client,
        device,
        config,
        limits,
        embedding,
        final_norm,
        head,
        layers,
        status: Status::Ready,
        committed: Vec::new(),
        logits: Vec::new(),
        pending: Vec::new(),
        #[cfg(test)]
        fault: None,
    })
}
