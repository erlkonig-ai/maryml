//! Checkpoint input decoding, not a retained model-graph representation.
//!
//! Contract: NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 at
//! da5b2650a835292c678994a1fa3fb653250fb508. Only the text backbone and
//! lm_head are in scope. Unknown JSON fields (including vision/MTP metadata)
//! are ignored; ordered text layers and explicit quantization selectors are not.

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LayerKind {
    Mamba,
    Moe,
    Attention,
}

/// ModelOpt serializes an infinite timestep bound as a tagged JSON object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeStepBound(pub f64);

impl<'de> Deserialize<'de> for TimeStepBound {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Bound {
            Finite(f64),
            Tagged { __float__: String },
        }
        match Bound::deserialize(deserializer)? {
            Bound::Finite(value) => Ok(Self(value)),
            Bound::Tagged { __float__ } if __float__ == "Infinity" => Ok(Self(f64::INFINITY)),
            Bound::Tagged { .. } => Err(serde::de::Error::custom("unsupported timestep bound")),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub layers_block_type: Vec<LayerKind>,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_epsilon: f64,
    pub residual_in_fp32: bool,
    pub rescale_prenorm_residual: bool,
    pub tie_word_embeddings: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub sliding_window: Option<usize>,
    pub conv_kernel: usize,
    pub expand: usize,
    pub mamba_head_dim: usize,
    pub mamba_num_heads: usize,
    pub mamba_hidden_act: String,
    pub mamba_proj_bias: bool,
    pub mamba_ssm_cache_dtype: String,
    pub use_bias: bool,
    pub use_conv_bias: bool,
    pub ssm_state_size: usize,
    pub n_groups: usize,
    pub chunk_size: usize,
    pub time_step_limit: [TimeStepBound; 2],
    pub time_step_min: f64,
    pub time_step_max: f64,
    pub time_step_floor: f64,
    pub intermediate_size: usize,
    pub mlp_hidden_act: String,
    pub mlp_bias: bool,
    pub moe_intermediate_size: usize,
    pub moe_latent_size: usize,
    pub moe_shared_expert_intermediate_size: usize,
    pub moe_shared_expert_overlap: bool,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    /// Router grouping; distinct from Mamba's `n_groups`.
    pub n_group: usize,
    pub topk_group: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CheckpointConfig {
    pub model_type: String,
    pub llm_config: TextConfig,
    pub quantization_config: QuantizationConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(tag = "quant_algo")]
pub enum LinearQuantization {
    #[serde(rename = "FP8")]
    Fp8,
    #[serde(rename = "NVFP4")]
    Nvfp4 { group_size: usize },
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub quant_algo: String,
    /// Metadata only: decoding it does not choose a runtime KV/cache policy.
    pub kv_cache_quant_algo: Option<String>,
    pub quantized_layers: BTreeMap<String, LinearQuantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfQuantConfig {
    pub quantization: QuantizationConfig,
}

/// Checkpoint *module* name, without `.weight` or companion scale suffixes.
/// No MTP, vision, wildcard expansion or implicit precision fallback.
pub fn is_text_module(name: &str) -> bool {
    name.starts_with("language_model.backbone.") || name == "language_model.lm_head"
}

impl QuantizationConfig {
    pub fn text_linear(&self, module: &str) -> Result<Option<&LinearQuantization>, String> {
        if !is_text_module(module) {
            return Ok(None);
        }
        let quant = self.quantized_layers.get(module);
        if let Some(LinearQuantization::Nvfp4 { group_size }) = quant {
            if *group_size != 16 {
                return Err(format!(
                    "{module}: NVFP4 group size {group_size}, expected 16"
                ));
            }
        }
        Ok(quant)
    }
}

/// Preserve scalar-vs-list EOS syntax without conflating it with tokenizer IDs.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    pub bos_token_id: u32,
    pub eos_token_id: TokenIds,
    pub pad_token_id: u32,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: f64,
}

/// Token spellings are not IDs. Resolving them requires tokenizer.json;
/// generation pad=0 must not overwrite tokenizer pad="<|im_end|>" (ID 11).
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    pub bos_token: String,
    pub eos_token: String,
    pub pad_token: String,
    pub unk_token: String,
    pub padding_side: String,
    pub model_max_length: usize,
}
