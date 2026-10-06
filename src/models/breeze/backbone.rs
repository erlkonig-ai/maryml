//! Causal Qwen3 backbone and ordinary Breeze depth layers. Geometry comes from
//! the selected nested configs, never the misleading outer backbone defaults.
use super::{
    config::{DecoderConfig, RopeConfig, RopeKind},
    cuda_ops as ops,
    generator::Binder,
};
use crate::models::qwen3_5::gdn_mixer::project;
use anyhow::{Result, ensure};
pub(super) use ops::Tensor;
use triblespace::core::repo::BlobStoreGet;

pub(super) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    project(x, w).map_err(anyhow::Error::msg)
}
pub(super) fn frequencies(c: &RopeConfig, d: usize) -> Result<Vec<f32>> {
    ensure!(
        d > 0 && d % 2 == 0 && c.theta.is_finite() && c.theta > 0.0,
        "invalid RoPE geometry"
    );
    let mut result = Vec::with_capacity(d / 2);
    for i in 0..d / 2 {
        let mut f = 1.0f64 / c.theta.powf((2 * i) as f64 / d as f64);
        match c.kind {
            RopeKind::Default => (),
            RopeKind::Linear => {
                ensure!(c.factor > 0.0, "invalid linear RoPE factor");
                f /= c.factor;
            }
            RopeKind::Llama3 => {
                ensure!(
                    c.factor > 0.0
                        && c.high_freq_factor > c.low_freq_factor
                        && c.low_freq_factor > 0.0
                        && c.original_max_position_embeddings > 0,
                    "invalid Llama3 RoPE scaling"
                );
                let wave = std::f64::consts::TAU / f;
                let context = c.original_max_position_embeddings as f64;
                if wave > context / c.low_freq_factor {
                    f /= c.factor;
                } else if wave >= context / c.high_freq_factor {
                    let smooth = (context / wave - c.low_freq_factor)
                        / (c.high_freq_factor - c.low_freq_factor);
                    f = (1.0 - smooth) * f / c.factor + smooth * f;
                }
            }
        }
        ensure!(f.is_finite() && f > 0.0, "nonfinite RoPE frequency");
        result.push(f as f32);
    }
    Ok(result)
}

struct Layer {
    input: Tensor,
    post: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}
pub(super) struct Kv {
    key: Tensor,
    value: Tensor,
}
pub(super) struct Decoder {
    config: DecoderConfig,
    layers: Vec<Layer>,
    norm: Tensor,
    freq: Vec<f32>,
}
impl Decoder {
    /// Binder forwards Generator's immutable mapped-prefix lifetime contract.
    pub(super) unsafe fn bind<R: BlobStoreGet>(
        b: &mut Binder<'_, R>,
        c: DecoderConfig,
        prefix: &str,
        qk_norm: bool,
    ) -> Result<Self> {
        let h = c.hidden_size as u64;
        let f = c.intermediate_size as u64;
        let q = (c.num_attention_heads * c.head_dim) as u64;
        let kv = (c.num_key_value_heads * c.head_dim) as u64;
        let d = c.head_dim as u64;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let p = format!("{prefix}.layers.{i}");
            // SAFETY: every fixed slot belongs to the same validated immutable pile.
            layers.push(unsafe {
                Layer {
                    input: b.weight(&format!("{p}.input_layernorm.weight"), [h])?,
                    post: b.weight(&format!("{p}.post_attention_layernorm.weight"), [h])?,
                    q: b.weight(&format!("{p}.self_attn.q_proj.weight"), [q, h])?,
                    k: b.weight(&format!("{p}.self_attn.k_proj.weight"), [kv, h])?,
                    v: b.weight(&format!("{p}.self_attn.v_proj.weight"), [kv, h])?,
                    o: b.weight(&format!("{p}.self_attn.o_proj.weight"), [h, q])?,
                    q_norm: if qk_norm {
                        Some(b.weight(&format!("{p}.self_attn.q_norm.weight"), [d])?)
                    } else {
                        None
                    },
                    k_norm: if qk_norm {
                        Some(b.weight(&format!("{p}.self_attn.k_norm.weight"), [d])?)
                    } else {
                        None
                    },
                    gate: b.weight(&format!("{p}.mlp.gate_proj.weight"), [f, h])?,
                    up: b.weight(&format!("{p}.mlp.up_proj.weight"), [f, h])?,
                    down: b.weight(&format!("{p}.mlp.down_proj.weight"), [h, f])?,
                }
            });
        }
        let norm = unsafe { b.weight(&format!("{prefix}.norm.weight"), [h])? };
        let freq = frequencies(&c.rope, c.head_dim)?;
        Ok(Self {
            config: c,
            layers,
            norm,
            freq,
        })
    }
    /// Invocation-local KV; input at past=0 can contain a whole prefill. Later
    /// calls append one position, producing fresh cache storage, never aliases.
    pub(super) fn forward(
        &self,
        mut x: Tensor,
        past: usize,
        old: &[Kv],
    ) -> Result<(Tensor, Vec<Kv>)> {
        let c = &self.config;
        let t = x.meta.shape()[1];
        ensure!(
            x.meta.shape().as_slice() == [1, t, c.hidden_size] && t > 0 && past + t <= 2048,
            "invalid decoder extent"
        );
        ensure!(
            (past == 0 && old.is_empty()) || (past > 0 && t == 1 && old.len() == self.layers.len()),
            "incomplete decoder KV state"
        );
        let mut cache = Vec::with_capacity(self.layers.len());
        for (i, l) in self.layers.iter().enumerate() {
            let n = ops::norm(&x, &l.input, c.rms_norm_eps as f32, false);
            let mut q = ops::reshape(
                linear(&n, &l.q)?,
                &[1, t, c.num_attention_heads, c.head_dim],
            );
            let mut k = ops::reshape(
                linear(&n, &l.k)?,
                &[1, t, c.num_key_value_heads, c.head_dim],
            );
            let v = ops::reshape(
                linear(&n, &l.v)?,
                &[1, t, c.num_key_value_heads, c.head_dim],
            );
            if let Some(w) = &l.q_norm {
                q = ops::norm(&q, w, c.rms_norm_eps as f32, false);
            }
            if let Some(w) = &l.k_norm {
                k = ops::norm(&k, w, c.rms_norm_eps as f32, false);
            }
            q = ops::rope(&q, past, &self.freq);
            k = ops::rope(&k, past, &self.freq);
            if let Some(s) = old.get(i) {
                ensure!(
                    s.key.meta.shape().as_slice() == [1, past, c.num_key_value_heads, c.head_dim]
                        && s.value.meta.shape() == s.key.meta.shape(),
                    "decoder cache shape mismatch"
                );
            }
            let key = ops::append(&k, old.get(i).map(|s| &s.key));
            let value = ops::append(&v, old.get(i).map(|s| &s.value));
            let a = ops::attention(
                &q,
                &key,
                &value,
                past,
                1.0 / (c.head_dim as f32).sqrt(),
                true,
                None,
            );
            let a = ops::reshape(a, &[1, t, c.num_attention_heads * c.head_dim]);
            let residual = ops::add(&x, &linear(&a, &l.o)?);
            let n = ops::norm(&residual, &l.post, c.rms_norm_eps as f32, false);
            let gate = linear(&n, &l.gate)?;
            let up = linear(&n, &l.up)?;
            x = ops::add(&residual, &linear(&ops::swiglu(&gate, &up), &l.down)?);
            cache.push(Kv { key, value });
        }
        Ok((
            ops::norm(&ops::last(&x), &self.norm, c.rms_norm_eps as f32, false),
            cache,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frequency_families_are_not_outer_config_aliases() {
        let mut c = RopeConfig {
            theta: 1e6,
            kind: RopeKind::Default,
            factor: 1.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let plain = frequencies(&c, 128).unwrap();
        assert_eq!(plain[0], 1.0);
        c.kind = RopeKind::Linear;
        c.factor = 8.0;
        assert_eq!(frequencies(&c, 128).unwrap()[0], 0.125);
        c.kind = RopeKind::Llama3;
        c.theta = 500000.0;
        c.factor = 32.0;
        c.low_freq_factor = 0.001953125;
        c.high_freq_factor = 0.0078125;
        c.original_max_position_embeddings = 16;
        let depth = frequencies(&c, 128).unwrap();
        assert_eq!(depth[0], 1.0);
        assert!(depth[63] < plain[63]);
        assert!(frequencies(&c, 127).is_err());
    }
}
