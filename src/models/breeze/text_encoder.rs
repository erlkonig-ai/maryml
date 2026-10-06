//! The pinned T5Gemma2 shim is bidirectional regardless of the unused outer
//! use_bidirectional_attention flag. Text segments reset positions separately.
use super::{
    backbone::{frequencies, linear},
    config::TextConfig,
    cuda_ops as ops,
    generator::Binder,
};
use anyhow::{Result, ensure};
use ops::Tensor;
use triblespace::core::repo::BlobStoreGet;

struct Layer {
    pre_attn: Tensor,
    post_attn: Tensor,
    pre_ff: Tensor,
    post_ff: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}
pub(super) struct TextEncoder {
    config: TextConfig,
    embedding: Tensor,
    eoi: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    projection: Tensor,
    full_freq: Vec<f32>,
    sliding_freq: Vec<f32>,
}
impl TextEncoder {
    pub(super) unsafe fn bind<R: BlobStoreGet>(
        b: &mut Binder<'_, R>,
        c: TextConfig,
        output: usize,
    ) -> Result<Self> {
        let g = &c.decoder;
        let h = g.hidden_size as u64;
        let f = g.intermediate_size as u64;
        let q = (g.num_attention_heads * g.head_dim) as u64;
        let kv = (g.num_key_value_heads * g.head_dim) as u64;
        let d = g.head_dim as u64;
        let mut layers = Vec::with_capacity(g.num_hidden_layers);
        for i in 0..g.num_hidden_layers {
            let p = format!("text_encoder.layers.{i}");
            // SAFETY: fixed typed consuming slots under Generator's contract.
            layers.push(unsafe {
                Layer {
                    pre_attn: b.weight(&format!("{p}.pre_self_attn_layernorm.weight"), [h])?,
                    post_attn: b.weight(&format!("{p}.post_self_attn_layernorm.weight"), [h])?,
                    pre_ff: b.weight(&format!("{p}.pre_feedforward_layernorm.weight"), [h])?,
                    post_ff: b.weight(&format!("{p}.post_feedforward_layernorm.weight"), [h])?,
                    q: b.weight(&format!("{p}.self_attn.q_proj.weight"), [q, h])?,
                    k: b.weight(&format!("{p}.self_attn.k_proj.weight"), [kv, h])?,
                    v: b.weight(&format!("{p}.self_attn.v_proj.weight"), [kv, h])?,
                    o: b.weight(&format!("{p}.self_attn.o_proj.weight"), [h, q])?,
                    q_norm: b.weight(&format!("{p}.self_attn.q_norm.weight"), [d])?,
                    k_norm: b.weight(&format!("{p}.self_attn.k_norm.weight"), [d])?,
                    gate: b.weight(&format!("{p}.mlp.gate_proj.weight"), [f, h])?,
                    up: b.weight(&format!("{p}.mlp.up_proj.weight"), [f, h])?,
                    down: b.weight(&format!("{p}.mlp.down_proj.weight"), [h, f])?,
                }
            });
        }
        let full_freq = frequencies(&c.full_rope, g.head_dim)?;
        let sliding_freq = frequencies(&c.sliding_rope, g.head_dim)?;
        Ok(unsafe {
            Self {
                embedding: b
                    .weight("text_encoder.embed_tokens.weight", [c.vocab_size as u64, h])?,
                eoi: b.weight("text_encoder.embed_tokens.eoi_embedding", [h])?,
                norm: b.weight("text_encoder.norm.weight", [h])?,
                projection: b.weight("text_encoder_proj.weight", [output as u64, h])?,
                config: c,
                layers,
                full_freq,
                sliding_freq,
            }
        })
    }
    pub(super) fn encode(&self, ids: &[u32]) -> Result<Tensor> {
        let c = &self.config;
        let g = &c.decoder;
        let t = ids.len();
        ensure!(
            t > 0 && t <= 2048 && ids.iter().all(|&id| (id as usize) < c.vocab_size),
            "invalid text segment"
        );
        let mut x = ops::text_embedding(&self.embedding, &self.eoi, ids, c.eoi_token_index);
        for (i, l) in self.layers.iter().enumerate() {
            let n = ops::norm(&x, &l.pre_attn, g.rms_norm_eps as f32, true);
            let q = ops::reshape(
                linear(&n, &l.q)?,
                &[1, t, g.num_attention_heads, g.head_dim],
            );
            let k = ops::reshape(
                linear(&n, &l.k)?,
                &[1, t, g.num_key_value_heads, g.head_dim],
            );
            let v = ops::reshape(
                linear(&n, &l.v)?,
                &[1, t, g.num_key_value_heads, g.head_dim],
            );
            let q = ops::norm(&q, &l.q_norm, g.rms_norm_eps as f32, true);
            let k = ops::norm(&k, &l.k_norm, g.rms_norm_eps as f32, true);
            let sliding = c.layer_types[i] == "sliding_attention";
            let freq = if sliding {
                &self.sliding_freq
            } else {
                &self.full_freq
            };
            let q = ops::rope(&q, 0, freq);
            let k = ops::rope(&k, 0, freq);
            let a = ops::attention(
                &q,
                &k,
                &v,
                0,
                1.0 / (c.query_pre_attn_scalar as f32).sqrt(),
                false,
                sliding.then_some(c.sliding_window),
            );
            let a = ops::reshape(a, &[1, t, g.num_attention_heads * g.head_dim]);
            let a = linear(&a, &l.o)?;
            let a = ops::norm(&a, &l.post_attn, g.rms_norm_eps as f32, true);
            let residual = ops::add(&x, &a);
            let n = ops::norm(&residual, &l.pre_ff, g.rms_norm_eps as f32, true);
            let gate = linear(&n, &l.gate)?;
            let up = linear(&n, &l.up)?;
            let ff = linear(&ops::geglu(&gate, &up), &l.down)?;
            let ff = ops::norm(&ff, &l.post_ff, g.rms_norm_eps as f32, true);
            x = ops::add(&residual, &ff);
        }
        linear(
            &ops::norm(&x, &self.norm, g.rms_norm_eps as f32, true),
            &self.projection,
        )
    }
}
