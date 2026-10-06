//! Geometry of the selected Breeze runtime, reconstructed from native facts.
//! The nested Qwen3 config, NOT the outer Breeze defaults, owns the backbone.
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeKind {
    Default,
    Linear,
    Llama3,
}

#[derive(Clone, Debug)]
pub struct RopeConfig {
    pub theta: f64,
    pub kind: RopeKind,
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

#[derive(Clone, Debug)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope: RopeConfig,
}

#[derive(Clone, Debug)]
pub struct TextConfig {
    pub decoder: DecoderConfig,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub query_pre_attn_scalar: f64,
    pub layer_types: Vec<String>,
    pub full_rope: RopeConfig,
    pub sliding_rope: RopeConfig,
    pub eoi_token_index: u32,
}

#[derive(Clone, Debug)]
pub struct DepthConfig {
    pub decoder: DecoderConfig,
    pub audio_embed_size: usize,
    pub backbone_hidden_size: usize,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub text: TextConfig,
    pub backbone: DecoderConfig,
    pub depth: DepthConfig,
    pub text_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub num_codebooks: usize,
    pub max_context: usize,
    pub audio_token_id: u32,
    pub audio_eos_token_id: u32,
    pub codebook_eos_token_id: u32,
    pub codebook_pad_token_id: u32,
    pub tie_codebooks_embeddings: bool,
}

fn usize_field(v: &Value, key: &str) -> Result<usize> {
    let n = v[key]
        .as_u64()
        .with_context(|| format!("missing unsigned {key}"))?;
    usize::try_from(n).with_context(|| format!("{key} exceeds host usize"))
}

fn id_field(v: &Value, key: &str) -> Result<u32> {
    u32::try_from(usize_field(v, key)?).with_context(|| format!("{key} exceeds u32"))
}

fn positive(v: &Value, key: &str) -> Result<f64> {
    let x = v[key]
        .as_f64()
        .with_context(|| format!("missing numeric {key}"))?;
    ensure!(
        x.is_finite() && x > 0.0,
        "{key} must be finite and positive"
    );
    Ok(x)
}

impl RopeConfig {
    fn parse(theta: f64, scaling: &Value) -> Result<Self> {
        ensure!(theta.is_finite() && theta > 0.0, "invalid rope theta");
        let kind = match scaling["rope_type"].as_str().unwrap_or("default") {
            "default" => RopeKind::Default,
            "linear" => RopeKind::Linear,
            "llama3" => RopeKind::Llama3,
            other => bail!("unsupported RoPE kind {other}"),
        };
        let factor = if kind == RopeKind::Default {
            1.0
        } else {
            positive(scaling, "factor")?
        };
        let (low, high, original) = if kind == RopeKind::Llama3 {
            let low = positive(scaling, "low_freq_factor")?;
            let high = positive(scaling, "high_freq_factor")?;
            let original = usize_field(scaling, "original_max_position_embeddings")?;
            ensure!(high > low && original > 0, "invalid llama3 frequency bands");
            (low, high, original)
        } else {
            (1.0, 1.0, 0)
        };
        Ok(Self {
            theta,
            kind,
            factor,
            low_freq_factor: low,
            high_freq_factor: high,
            original_max_position_embeddings: original,
        })
    }
}

impl DecoderConfig {
    fn parse(v: &Value, rope: RopeConfig) -> Result<Self> {
        ensure!(
            v["attention_bias"].as_bool() != Some(true),
            "attention bias unsupported"
        );
        ensure!(
            v["mlp_bias"].as_bool() != Some(true),
            "MLP bias unsupported"
        );
        let config = Self {
            hidden_size: usize_field(v, "hidden_size")?,
            intermediate_size: usize_field(v, "intermediate_size")?,
            num_hidden_layers: usize_field(v, "num_hidden_layers")?,
            num_attention_heads: usize_field(v, "num_attention_heads")?,
            num_key_value_heads: usize_field(v, "num_key_value_heads")?,
            head_dim: usize_field(v, "head_dim")?,
            rms_norm_eps: positive(v, "rms_norm_eps")?,
            rope,
        };
        ensure!(
            config.hidden_size > 0
                && config.intermediate_size > 0
                && config.num_hidden_layers > 0
                && config.num_attention_heads > 0
                && config.num_key_value_heads > 0
                && config.head_dim > 0
                && config.head_dim % 2 == 0,
            "invalid decoder geometry"
        );
        ensure!(
            config.num_attention_heads % config.num_key_value_heads == 0,
            "query heads must be grouped by KV heads"
        );
        Ok(config)
    }
}

impl Config {
    pub fn from_json(v: &Value) -> Result<Self> {
        ensure!(v["model_type"] == "breeze", "expected Breeze config");
        ensure!(
            v["text_encoder_proj_type"] == "linear",
            "only final linear text projection supported"
        );
        let final_feature = match v.get("text_encoder_feature_layer_idx") {
            None => true,
            Some(index) if index.as_i64() == Some(-1) => true,
            Some(Value::Array(indices)) => indices.len() == 1 && indices[0].as_i64() == Some(-1),
            _ => false,
        };
        ensure!(final_feature, "only final text encoder feature supported");
        for field in [
            "text_encoder_lora_config",
            "text_encoder_special_tokens_config",
        ] {
            ensure!(
                v[field]["enabled"].as_bool() != Some(true),
                "unmerged active {field} unsupported"
            );
        }
        let b = &v["backbone_config"];
        ensure!(
            b["model_type"] == "qwen3" && b["hidden_act"] == "silu",
            "expected Qwen3 SwiGLU backbone"
        );
        ensure!(
            b["use_sliding_window"].as_bool() != Some(true),
            "active backbone window unsupported"
        );
        let backbone = DecoderConfig::parse(
            b,
            RopeConfig::parse(positive(b, "rope_theta")?, &b["rope_scaling"])?,
        )?;
        let d = &v["depth_decoder_config"];
        ensure!(d["hidden_act"] == "silu", "expected SwiGLU depth decoder");
        let depth = DepthConfig {
            decoder: DecoderConfig::parse(
                d,
                RopeConfig::parse(positive(d, "rope_theta")?, &d["rope_scaling"])?,
            )?,
            audio_embed_size: usize_field(d, "audio_embed_size")?,
            backbone_hidden_size: usize_field(d, "backbone_hidden_size")?,
        };
        let t = &v["text_encoder_config"];
        ensure!(
            t["model_type"] == "t5gemma2_text" && t["hidden_activation"] == "gelu_pytorch_tanh",
            "expected T5Gemma2 text encoder"
        );
        for field in ["attn_logit_softcapping", "final_logit_softcapping"] {
            ensure!(t[field].is_null(), "active {field} unsupported");
        }
        let full = &t["rope_parameters"]["full_attention"];
        let sliding = &t["rope_parameters"]["sliding_attention"];
        let full_rope = RopeConfig::parse(positive(full, "rope_theta")?, full)?;
        let sliding_rope = RopeConfig::parse(positive(sliding, "rope_theta")?, sliding)?;
        let decoder = DecoderConfig::parse(t, full_rope.clone())?;
        let layer_types: Vec<String> = t["layer_types"]
            .as_array()
            .context("missing text layer types")?
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_owned)
                    .context("non-string layer type")
            })
            .collect::<Result<_>>()?;
        ensure!(
            layer_types.len() == decoder.num_hidden_layers
                && layer_types
                    .iter()
                    .all(|x| x == "full_attention" || x == "sliding_attention"),
            "unsupported text layer types"
        );
        let text = TextConfig {
            decoder,
            vocab_size: usize_field(t, "vocab_size")?,
            sliding_window: usize_field(t, "sliding_window")?,
            query_pre_attn_scalar: positive(t, "query_pre_attn_scalar")?,
            layer_types,
            full_rope,
            sliding_rope,
            eoi_token_index: id_field(t, "eoi_token_index")?,
        };
        let config = Self {
            text,
            backbone,
            depth,
            text_vocab_size: usize_field(v, "text_vocab_size")?,
            audio_vocab_size: usize_field(v, "audio_vocab_size")?,
            num_codebooks: usize_field(v, "num_codebooks")?,
            max_context: usize_field(v, "max_position_embeddings")?,
            audio_token_id: id_field(v, "audio_token_id")?,
            audio_eos_token_id: id_field(v, "audio_eos_token_id")?,
            codebook_eos_token_id: id_field(v, "codebook_eos_token_id")?,
            codebook_pad_token_id: id_field(v, "codebook_pad_token_id")?,
            tie_codebooks_embeddings: v["tie_codebooks_embeddings"]
                .as_bool()
                .context("missing audio tie flag")?,
        };
        ensure!(
            config.num_codebooks == 16
                && config.audio_vocab_size == 2051
                && usize_field(d, "num_codebooks")? == 16
                && usize_field(d, "vocab_size")? == 2051,
            "expected 16 Breeze codebooks and 2051 code values"
        );
        ensure!(
            config.codebook_eos_token_id == 0 && config.codebook_pad_token_id == 2050,
            "unsupported audio control identities"
        );
        ensure!(
            config.tie_codebooks_embeddings
                && config.depth.audio_embed_size == config.backbone.hidden_size
                && config.depth.backbone_hidden_size == config.backbone.hidden_size,
            "unsupported audio embedding tie geometry"
        );
        ensure!(
            config.text.vocab_size == config.text_vocab_size
                && config.max_context > 0
                && config.text.sliding_window > 0
                && (config.audio_token_id as usize) < config.text_vocab_size
                && (config.audio_eos_token_id as usize) < config.text_vocab_size,
            "invalid vocabulary/context geometry"
        );
        Ok(config)
    }
}
