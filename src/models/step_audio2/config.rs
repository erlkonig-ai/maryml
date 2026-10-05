use anyhow::{Result, ensure};
use serde::Deserialize;

/// Qwen2 decoder geometry, independent of the unused audio-understanding tower.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl Config {
    /// Required decoder fields only. Unknown checkpoint/config fields coexist.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        ensure!(
            value["model_type"] == "step_audio_2",
            "expected Step-Audio2 config"
        );
        let text = &value["text_config"];
        for (name, expected) in [("hidden_act", "silu"), ("torch_dtype", "bfloat16")] {
            if let Some(v) = text.get(name) {
                ensure!(v == expected, "unsupported text_config.{name}: {v}");
            }
        }
        ensure!(
            text.get("rope_scaling").is_none_or(|v| v.is_null()),
            "scaled RoPE is unsupported"
        );
        ensure!(
            text.get("tie_word_embeddings").is_none_or(|v| v == false),
            "Step requires an untied LM head"
        );
        if let Some(groups) = text.get("num_attention_groups") {
            ensure!(
                groups == &text["num_key_value_heads"],
                "attention groups differ from KV heads"
            );
        }
        let config: Self = serde_json::from_value(text.clone())?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_size > 0 && self.intermediate_size > 0 && self.num_hidden_layers > 0,
            "decoder dimensions must be positive"
        );
        ensure!(
            self.num_attention_heads > 0 && self.num_key_value_heads > 0,
            "attention head counts must be positive"
        );
        ensure!(
            self.hidden_size % self.num_attention_heads == 0
                && self.num_attention_heads % self.num_key_value_heads == 0,
            "invalid Qwen2 head geometry"
        );
        ensure!(
            self.head_dim() % 2 == 0,
            "rotate-half RoPE needs an even head dimension"
        );
        ensure!(
            self.vocab_size > 0 && self.max_position_embeddings > 0,
            "vocabulary and context sizes must be positive"
        );
        ensure!(
            self.rms_norm_eps.is_finite()
                && self.rms_norm_eps > 0.0
                && self.rope_theta.is_finite()
                && self.rope_theta > 0.0,
            "invalid RMS/RoPE parameters"
        );
        self.parameter_count()?;
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Visit consuming slots, not a retained catalogue of stored model rows.
    pub fn tensors(&self, mut visit: impl FnMut(&str, &[usize]) -> Result<()>) -> Result<()> {
        let h = self.hidden_size;
        let kv = self
            .num_key_value_heads
            .checked_mul(self.head_dim())
            .ok_or_else(|| anyhow::anyhow!("KV width overflow"))?;
        visit("model.embed_tokens.weight", &[self.vocab_size, h])?;
        visit("model.norm.weight", &[h])?;
        visit("lm_head.weight", &[self.vocab_size, h])?;
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                visit(&format!("{prefix}.{norm}.weight"), &[h])?;
            }
            for (projection, width) in [("q", h), ("k", kv), ("v", kv)] {
                visit(
                    &format!("{prefix}.self_attn.{projection}_proj.weight"),
                    &[width, h],
                )?;
                visit(
                    &format!("{prefix}.self_attn.{projection}_proj.bias"),
                    &[width],
                )?;
            }
            visit(&format!("{prefix}.self_attn.o_proj.weight"), &[h, h])?;
            for projection in ["gate", "up"] {
                visit(
                    &format!("{prefix}.mlp.{projection}_proj.weight"),
                    &[self.intermediate_size, h],
                )?;
            }
            visit(
                &format!("{prefix}.mlp.down_proj.weight"),
                &[h, self.intermediate_size],
            )?;
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> Result<u64> {
        let mut total = 0u64;
        self.tensors(|name, shape| {
            let count = shape
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
                .ok_or_else(|| anyhow::anyhow!("{name}: element count overflow"))?;
            total = total
                .checked_add(count)
                .ok_or_else(|| anyhow::anyhow!("model size overflow"))?;
            Ok(())
        })?;
        Ok(total)
    }
}
