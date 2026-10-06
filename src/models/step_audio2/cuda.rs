//! Native, bounded B1 Qwen2 token decoder. BF16 pile aliases, resident rotated
//! KV state, no Q/K normalization or attention gate, and required QKV biases.
//! All tensor math is CUDA; only tokenization, framing and scalar token transport
//! are host work. Greedy selection is explicit, not the Python sampling policy.
//! This module does not decode speech codes to waveforms.
use super::{
    codec::{EOT, Generated, MiniCodec, Termination},
    config::Config,
    cuda_ops as ops,
    load::{self, Artifacts, Assets},
};
use crate::{models::qwen3_5::gdn_mixer::project, nn::cuda_bf16_alias::CudaBf16Aliases};
use anyhow::{Result, ensure};
use cubecl::cuda::CudaDevice;
use std::time::Instant;
use triblespace::{
    core::repo::BlobStoreGet,
    prelude::{Id, TribleSet},
};

const MAX_CAPACITY: usize = 1024;
type Tensor = ops::Tensor;

/// Refuse unsupported/overflowing shapes before initializing CUDA or selecting
/// weights. FFN18944 is supported; the unrelated Qwen3.5 bound is not inherited.
pub fn validate_geometry(config: &Config, capacity: usize) -> Result<()> {
    config.validate()?;
    ensure!(
        (1..=MAX_CAPACITY).contains(&capacity) && capacity <= config.max_position_embeddings,
        "CUDA token capacity must be 1..={MAX_CAPACITY} and within the model context"
    );
    ensure!(
        config.hidden_size <= 16384
            && config.intermediate_size <= 65536
            && config.num_hidden_layers <= 128
            && config.num_attention_heads <= 128
            && config.head_dim() <= 256
            && config.vocab_size <= 262144,
        "unsupported bounded Qwen2 CUDA geometry"
    );
    ensure!(
        (config.rope_theta as f32).is_finite()
            && (config.rope_theta as f32) > 0.0
            && (config.rms_norm_eps as f32).is_finite()
            && (config.rms_norm_eps as f32) > 0.0,
        "RMS/RoPE parameters cannot be represented in F32"
    );
    config.tensors(|_, shape| {
        extent(shape)?;
        Ok(())
    })?;
    for shape in [
        vec![capacity, config.intermediate_size],
        vec![capacity, config.num_attention_heads, capacity],
        vec![capacity, config.num_key_value_heads, config.head_dim()],
    ] {
        extent(&shape)?;
    }
    Ok(())
}
fn extent(shape: &[usize]) -> Result<usize> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) });
    let n = n
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| anyhow::anyhow!("empty/overflowing CUDA extent"))?;
    n.checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("CUDA scratch byte extent overflow"))?;
    Ok(n)
}

/// Run tiny CUDA operator/transport invariants under the caller's GPU lock.
/// This does not load checkpoint weights or implement a host model oracle.
pub fn smoke_test() -> Result<()> {
    ops::smoke().map_err(anyhow::Error::msg)
}

fn stop_reason(token: u32, emitted: usize, limit: usize) -> Option<Termination> {
    if token == EOT {
        Some(Termination::Eos)
    } else if emitted == limit {
        Some(Termination::TokenLimit)
    } else {
        None
    }
}

/// A complete request is checked before any dispatch; neither context nor tail
/// is truncated. The final sampled token need not be forwarded again.
pub fn validate_request(
    config: &Config,
    capacity: usize,
    prompt: &[u32],
    limit: usize,
) -> Result<()> {
    ensure!(
        !prompt.is_empty() && limit > 0,
        "nonempty prompt and positive token limit required"
    );
    ensure!(
        prompt.iter().all(|&id| (id as usize) < config.vocab_size),
        "prompt ID outside vocabulary"
    );
    let positions = prompt
        .len()
        .checked_add(limit - 1)
        .ok_or_else(|| anyhow::anyhow!("request position overflow"))?;
    ensure!(
        positions <= capacity,
        "prompt plus forwarded decode tokens exceeds capacity; no truncation"
    );
    Ok(())
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Tensor,
    qb: Tensor,
    k: Tensor,
    kb: Tensor,
    v: Tensor,
    vb: Tensor,
    o: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}
struct Kv {
    key: Tensor,
    value: Tensor,
}

/// Model weights and native tokenizer are selected once from explicit opaque
/// roots. The caller owns a single bounded alias binder for this CUDA runtime.
/// KV caches are invocation-local resident tensors, not a model/pile catalogue.
pub struct Decoder {
    config: Config,
    codec: MiniCodec,
    root: Id,
    capacity: usize,
    embedding: Tensor,
    final_norm: Tensor,
    head: Tensor,
    layers: Vec<Layer>,
}

pub struct Generation {
    pub generated: Generated,
    pub text: String,
    pub prompt_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub total_seconds: f64,
}

impl Decoder {
    /// Bind only native BF16 leaves from `load::tensor`; no source checkpoint,
    /// dtype conversion, weight upload or CPU model fallback is reachable.
    ///
    /// # Safety
    /// Every selected blob must belong to a genuine validated frozen pile.
    /// Its file bytes INCLUDING preceding partial pages must remain immutable
    /// or append-only and untruncated until CUDA runtime/storage teardown.
    /// The binder retains mmap owners in external runtime storage, not merely
    /// until this Decoder, the reader or binder drops. An error after partial
    /// binding also retains registrations: reuse the same bounded binder and
    /// do not replace/truncate source files after any attempt. A read-only FD
    /// does not prove external immutability. Driver faults retain CubeCL's policy.
    pub unsafe fn from_frozen<R: BlobStoreGet>(
        facts: &TribleSet,
        reader: &R,
        ids: Artifacts,
        assets: Assets,
        capacity: usize,
        aliases: &mut CudaBf16Aliases,
    ) -> Result<Self> {
        let Assets { config, codec } = assets;
        validate_geometry(&config, capacity)?;
        let root = ids.model_root;
        // Bind at consuming slots, not through a retained tensor-name catalogue.
        unsafe fn bind<const N: usize>(
            facts: &TribleSet,
            reader: &impl BlobStoreGet,
            root: Id,
            name: &str,
            shape: [u64; N],
            aliases: &mut CudaBf16Aliases,
        ) -> Result<Tensor> {
            let (_, blob) = load::tensor(facts, reader, root, name, shape)?;
            // SAFETY: forwarded genuine immutable-prefix contract from caller.
            let tensor = unsafe { aliases.bind_pile_leaf(blob) }.map_err(anyhow::Error::msg)?;
            ensure!(
                !tensor.handle.can_mut(),
                "pile weights must be immutable aliases"
            );
            Ok(tensor)
        }
        macro_rules! weight {
            ($name:expr,$shape:expr) => {{
                // SAFETY: same reader, root, binder and immutable prefix lifetime.
                unsafe { bind(facts, reader, root, $name, $shape, aliases)? }
            }};
        }
        let h = config.hidden_size as u64;
        let f = config.intermediate_size as u64;
        let kv = (config.num_key_value_heads * config.head_dim()) as u64;
        let vocab = config.vocab_size as u64;
        let embedding = weight!("model.embed_tokens.weight", [vocab, h]);
        let final_norm = weight!("model.norm.weight", [h]);
        let head = weight!("lm_head.weight", [vocab, h]);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_norm: weight!(&format!("{p}.input_layernorm.weight"), [h]),
                post_norm: weight!(&format!("{p}.post_attention_layernorm.weight"), [h]),
                q: weight!(&format!("{p}.self_attn.q_proj.weight"), [h, h]),
                qb: weight!(&format!("{p}.self_attn.q_proj.bias"), [h]),
                k: weight!(&format!("{p}.self_attn.k_proj.weight"), [kv, h]),
                kb: weight!(&format!("{p}.self_attn.k_proj.bias"), [kv]),
                v: weight!(&format!("{p}.self_attn.v_proj.weight"), [kv, h]),
                vb: weight!(&format!("{p}.self_attn.v_proj.bias"), [kv]),
                o: weight!(&format!("{p}.self_attn.o_proj.weight"), [h, h]),
                gate: weight!(&format!("{p}.mlp.gate_proj.weight"), [f, h]),
                up: weight!(&format!("{p}.mlp.up_proj.weight"), [f, h]),
                down: weight!(&format!("{p}.mlp.down_proj.weight"), [h, f]),
            });
        }
        Ok(Self {
            config,
            codec,
            root,
            capacity,
            embedding,
            final_norm,
            head,
            layers,
        })
    }
    pub fn model_root(&self) -> Id {
        self.root
    }
    pub fn device(&self) -> &CudaDevice {
        &self.embedding.device
    }
    pub fn synchronize(&self) -> Result<()> {
        cubecl::future::block_on(self.embedding.client.sync())
            .map_err(|e| anyhow::anyhow!("CUDA sync: {e:?}"))
    }

    fn forward(&self, ids: &[u32], past: usize, old: &[Kv]) -> Result<(Tensor, Vec<Kv>)> {
        ensure!(
            !ids.is_empty() && past + ids.len() <= self.capacity,
            "invalid forward extent"
        );
        ensure!(
            ids.iter().all(|&id| (id as usize) < self.config.vocab_size),
            "invalid forward token"
        );
        ensure!(
            (past == 0 && old.is_empty())
                || (past > 0 && ids.len() == 1 && old.len() == self.layers.len()),
            "decode requires one token and one complete resident cache per layer"
        );
        let c = &self.config;
        let t = ids.len();
        let d = c.head_dim();
        let mut x = ops::gather(&self.embedding, ids);
        let mut next = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let normalized = ops::norm(&x, &layer.input_norm, c.rms_norm_eps as f32);
            let q = ops::bias(
                &project(&normalized, &layer.q).map_err(anyhow::Error::msg)?,
                &layer.qb,
            );
            let k = ops::bias(
                &project(&normalized, &layer.k).map_err(anyhow::Error::msg)?,
                &layer.kb,
            );
            let v = ops::bias(
                &project(&normalized, &layer.v).map_err(anyhow::Error::msg)?,
                &layer.vb,
            );
            let q = ops::rope(
                &ops::reshape(q, &[1, t, c.num_attention_heads, d]),
                past,
                c.rope_theta as f32,
            );
            let k = ops::rope(
                &ops::reshape(k, &[1, t, c.num_key_value_heads, d]),
                past,
                c.rope_theta as f32,
            );
            let v = ops::reshape(v, &[1, t, c.num_key_value_heads, d]);
            let old = old.get(index);
            if let Some(old) = old {
                ensure!(
                    old.key.meta.shape().as_slice() == [1, past, c.num_key_value_heads, d]
                        && old.value.meta.shape().as_slice() == [1, past, c.num_key_value_heads, d],
                    "cache shape mismatch"
                );
            }
            let key = ops::append(&k, old.map(|s| &s.key));
            let value = ops::append(&v, old.map(|s| &s.value));
            let attended = ops::attention(&q, &key, &value, past);
            let attended = ops::reshape(attended, &[1, t, c.hidden_size]);
            let attention = project(&attended, &layer.o).map_err(anyhow::Error::msg)?;
            let residual = ops::add(&x, &attention);
            let post = ops::norm(&residual, &layer.post_norm, c.rms_norm_eps as f32);
            let gate = project(&post, &layer.gate).map_err(anyhow::Error::msg)?;
            let up = project(&post, &layer.up).map_err(anyhow::Error::msg)?;
            let down =
                project(&ops::swiglu(&gate, &up), &layer.down).map_err(anyhow::Error::msg)?;
            x = ops::add(&residual, &down);
            next.push(Kv { key, value });
        }
        let last = ops::last(&x);
        let last = ops::norm(&last, &self.final_norm, c.rms_norm_eps as f32);
        let logits = project(&last, &self.head).map_err(anyhow::Error::msg)?;
        Ok((logits, next))
    }

    /// Fresh bounded speech request, greedy only. Each emitted token is kept;
    /// EOS and budget exhaustion are distinct outcomes. Calls do not share KV.
    pub fn generate(&mut self, system: &str, text: &str, limit: usize) -> Result<Generation> {
        let prompt = self.codec.speech_prompt(system, text)?;
        validate_request(&self.config, self.capacity, &prompt, limit)?;
        self.synchronize()?;
        let total = Instant::now();
        let (logits, mut cache) = self.forward(&prompt, 0, &[])?;
        let mut token = ops::greedy(&logits).map_err(anyhow::Error::msg)?;
        let prefill_seconds = total.elapsed().as_secs_f64();
        let decode = Instant::now();
        let mut ids = Vec::with_capacity(limit);
        let mut past = prompt.len();
        let termination = loop {
            ids.push(token);
            if let Some(reason) = stop_reason(token, ids.len(), limit) {
                break reason;
            }
            let (logits, next) = self.forward(&[token], past, &cache)?;
            token = ops::greedy(&logits).map_err(anyhow::Error::msg)?;
            cache = next;
            past += 1;
        };
        let decode_seconds = decode.elapsed().as_secs_f64();
        let total_seconds = total.elapsed().as_secs_f64();
        let generated = self.codec.split_generated(&ids, termination)?;
        let text = self.codec.decode_text(&generated.text_ids)?;
        Ok(Generation {
            generated,
            text,
            prompt_tokens: prompt.len(),
            prefill_seconds,
            decode_seconds,
            total_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mini() -> Config {
        Config {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            vocab_size: 158720,
            max_position_embeddings: 16384,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
        }
    }
    #[test]
    fn actual_ffn_and_grouped_heads_fit_bounded_runtime() {
        let c = mini();
        validate_geometry(&c, 1024).unwrap();
        assert_eq!(c.head_dim(), 128);
        assert!(validate_geometry(&c, 1025).is_err());
        let mut wrong = c.clone();
        wrong.num_key_value_heads = 3;
        assert!(validate_geometry(&wrong, 128).is_err());
        wrong = c.clone();
        wrong.intermediate_size = usize::MAX;
        assert!(validate_geometry(&wrong, 128).is_err());
    }
    #[test]
    fn final_emitted_token_is_not_a_forwarded_cache_slot() {
        let c = mini();
        validate_request(&c, 4, &[1, 2, 3], 2).unwrap();
        validate_request(&c, 4, &[1, 2, 3, 4], 1).unwrap();
        assert!(validate_request(&c, 4, &[1, 2, 3], 3).is_err());
        assert!(validate_request(&c, 4, &[158720], 1).is_err());
        assert!(validate_request(&c, 4, &[], 1).is_err());
        assert!(validate_request(&c, 4, &[1], 0).is_err());
        assert!(validate_request(&c, 4, &[1, 2], usize::MAX).is_err());
        assert_eq!(stop_reason(EOT, 2, 2), Some(Termination::Eos));
        assert_eq!(stop_reason(151696, 2, 2), Some(Termination::TokenLimit));
        assert_eq!(stop_reason(151696, 1, 2), None);
    }
    #[test]
    #[ignore = "requires actual CUDA device 0 and ordinary Stars reservation"]
    fn cuda_primitive_contracts() {
        smoke_test().unwrap();
    }
}
