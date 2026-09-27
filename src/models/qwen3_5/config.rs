//! Inference-relevant dense Qwen3.5 configuration (Transformers 5.2.0).
//!
//! Unknown metadata is deliberately ignored. Known unsupported behavior is
//! rejected by `validate`; deserialization alone is not validation. MTP fields
//! are metadata here: the reference dense model constructs no MTP modules.

use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub enum Dtype {
    #[serde(rename = "bfloat16")]
    Bf16,
    #[serde(rename = "float16")]
    F16,
    #[serde(rename = "float32")]
    F32,
}

impl Dtype {
    pub fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5Config {
    pub model_type: String,
    pub dtype: Dtype,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: usize,
    pub video_token_id: usize,
    pub vision_start_token_id: usize,
    pub vision_end_token_id: usize,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub dtype: Dtype,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub tie_word_embeddings: bool,
    // The reference unconditionally constructs the output gate. An absent
    // legacy flag therefore means true, but an explicit false is unsupported.
    #[serde(default = "yes")]
    pub attn_output_gate: bool,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default = "four")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub mamba_ssm_dtype: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub model_type: String,
    pub dtype: Dtype,
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

fn yes() -> bool {
    true
}
fn four() -> usize {
    4
}

pub(super) fn product(dims: &[usize]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |n, &d| {
        if d == 0 {
            return Err("dimensions must be nonzero".into());
        }
        n.checked_mul(d)
            .ok_or_else(|| "dimension product overflows usize".into())
    })
}

impl TextConfig {
    /// Call only after `Qwen3_5Config::validate`. Explicit layer types override
    /// the interval, exactly as in the reference configuration constructor.
    pub fn layer_type(&self, index: usize) -> LayerType {
        assert!(index < self.num_hidden_layers);
        match &self.layer_types {
            Some(types) => types[index],
            None if (index + 1) % self.full_attention_interval == 0 => LayerType::FullAttention,
            None => LayerType::LinearAttention,
        }
    }
}

impl Qwen3_5Config {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(json).map_err(|e| e.to_string())?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        Self::from_json(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
    }

    pub fn validate(&self) -> Result<(), String> {
        let t = &self.text_config;
        let v = &self.vision_config;
        if self.model_type != "qwen3_5"
            || t.model_type != "qwen3_5_text"
            || !matches!(v.model_type.as_str(), "qwen3_5" | "qwen3_5_vision")
        {
            return Err("only dense Qwen3.5 text+vision model types are supported".into());
        }
        if self.quantization_config.is_some() {
            return Err(
                "quantized checkpoint configuration is not a dense checkpoint contract".into(),
            );
        }
        if self.dtype != t.dtype || self.dtype != v.dtype {
            return Err("mixed root/text/vision checkpoint dtypes are unsupported".into());
        }
        if self.tie_word_embeddings != t.tie_word_embeddings {
            return Err("root/text tie_word_embeddings disagree".into());
        }
        if t.hidden_act != "silu"
            || !t.attn_output_gate
            || t.attention_dropout != 0.0
            || !t.mlp_only_layers.is_empty()
            || t.num_experts != 0
            || t.use_sliding_window
            || t.sliding_window.is_some()
        {
            return Err("unsupported text behavior: require dense SiLU, gated full attention, zero dropout, no sliding window/MLP-only layers".into());
        }
        if t.mamba_ssm_dtype.as_deref().is_some_and(|s| s != "float32") {
            return Err("DeltaNet recurrent computation must use float32".into());
        }
        if v.hidden_act != "gelu_pytorch_tanh" || !v.deepstack_visual_indexes.is_empty() {
            return Err("vision requires gelu_pytorch_tanh blocks and no deepstack".into());
        }
        // Check individual dimensions without multiplying unrelated dimensions.
        for n in [
            t.vocab_size,
            t.hidden_size,
            t.intermediate_size,
            t.num_hidden_layers,
            t.num_attention_heads,
            t.num_key_value_heads,
            t.head_dim,
            t.max_position_embeddings,
            t.linear_conv_kernel_dim,
            t.linear_key_head_dim,
            t.linear_value_head_dim,
            t.linear_num_key_heads,
            t.linear_num_value_heads,
            v.depth,
            v.hidden_size,
            v.intermediate_size,
            v.num_heads,
            v.in_channels,
            v.patch_size,
            v.temporal_patch_size,
            v.spatial_merge_size,
            v.out_hidden_size,
            v.num_position_embeddings,
        ] {
            product(&[n])?;
        }
        if t.num_attention_heads % t.num_key_value_heads != 0
            || t.linear_num_value_heads % t.linear_num_key_heads != 0
        {
            return Err(
                "query heads / KV heads and DeltaNet value heads / key heads must divide exactly"
                    .into(),
            );
        }
        if let Some(types) = &t.layer_types {
            if types.len() != t.num_hidden_layers {
                return Err("layer_types length must equal num_hidden_layers".into());
            }
        } else if t.full_attention_interval == 0 {
            return Err("full_attention_interval must be nonzero without explicit layer_types".into());
        }
        if !t.rms_norm_eps.is_finite() || t.rms_norm_eps <= 0.0 {
            return Err("rms_norm_eps must be finite and positive".into());
        }
        let r = &t.rope_parameters;
        if r.rope_type != "default" || !r.mrope_interleaved || t.rope_scaling.is_some() {
            return Err("only default interleaved MRoPE in rope_parameters is supported".into());
        }
        if !r.rope_theta.is_finite()
            || r.rope_theta <= 0.0
            || !r.partial_rotary_factor.is_finite()
            || r.partial_rotary_factor <= 0.0
            || r.partial_rotary_factor > 1.0
            || t.partial_rotary_factor
                .is_some_and(|p| p != r.partial_rotary_factor)
        {
            return Err("invalid or inconsistent partial rotary parameters".into());
        }
        let rotary = t.head_dim as f64 * r.partial_rotary_factor;
        let rotary_dim = rotary as usize;
        let half = rotary_dim / 2;
        let section_sum = r
            .mrope_section
            .iter()
            .try_fold(0usize, |n, s| n.checked_add(*s));
        if rotary != rotary_dim as f64
            || rotary_dim == 0
            || rotary_dim % 2 != 0
            || section_sum != Some(half)
            || r.mrope_section[1] > (half + 1) / 3
            || r.mrope_section[2] > half / 3
        {
            return Err("MRoPE sections must fit the even partial rotary head dimension".into());
        }
        let side = (v.num_position_embeddings as f64).sqrt() as usize;
        if side.checked_mul(side) != Some(v.num_position_embeddings)
            || v.hidden_size % v.num_heads != 0
            || (v.hidden_size / v.num_heads) % 4 != 0
            || v.out_hidden_size != t.hidden_size
        {
            return Err("vision requires square learned positions, head dimension divisible by four, and text-width output".into());
        }
        let ids = [
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        ];
        for (i, id) in ids.iter().enumerate() {
            if *id >= t.vocab_size || ids[..i].contains(id) {
                return Err(
                    "vision token IDs must be distinct and inside the text vocabulary".into(),
                );
            }
        }
        // Bound every derived projection dimension before the layout multiplies.
        let keys = product(&[t.linear_num_key_heads, t.linear_key_head_dim, 2])?;
        let values = product(&[t.linear_num_value_heads, t.linear_value_head_dim])?;
        keys.checked_add(values)
            .ok_or("DeltaNet convolution dimension overflows usize")?;
        product(&[t.num_attention_heads, t.head_dim, 2])?;
        product(&[t.num_key_value_heads, t.head_dim])?;
        product(&[v.hidden_size, 3])?;
        product(&[v.hidden_size, v.spatial_merge_size, v.spatial_merge_size])?;
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub fn tiny() -> Qwen3_5Config {
        Qwen3_5Config::from_json(r#"{
            "model_type":"qwen3_5", "dtype":"bfloat16", "tie_word_embeddings":false,
            "image_token_id":12,"video_token_id":13,"vision_start_token_id":14,"vision_end_token_id":15,
            "unrelated_metadata":{"anything":true},
            "text_config":{
                "model_type":"qwen3_5_text","dtype":"bfloat16","vocab_size":16,
                "hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
                "hidden_act":"silu","max_position_embeddings":128,"rms_norm_eps":0.000001,
                "attention_bias":false,"attention_dropout":0.0,"tie_word_embeddings":false,
                "linear_conv_kernel_dim":4,"linear_key_head_dim":2,"linear_value_head_dim":3,
                "linear_num_key_heads":1,"linear_num_value_heads":2,"full_attention_interval":2,
                "rope_parameters":{"rope_type":"default","rope_theta":10000.0,
                    "partial_rotary_factor":1.0,"mrope_section":[1,1,0],"mrope_interleaved":true}},
            "vision_config":{
                "model_type":"qwen3_5_vision","dtype":"bfloat16","depth":1,
                "hidden_size":8,"intermediate_size":12,"hidden_act":"gelu_pytorch_tanh",
                "num_heads":2,"in_channels":3,"patch_size":2,"temporal_patch_size":2,
                "spatial_merge_size":2,"out_hidden_size":8,"num_position_embeddings":16}}
        "#).unwrap()
    }

    #[test]
    fn explicit_layer_types_override_interval() {
        let mut c = tiny();
        assert_eq!(c.text_config.layer_type(0), LayerType::LinearAttention);
        assert_eq!(c.text_config.layer_type(1), LayerType::FullAttention);
        c.text_config.layer_types = Some(vec![LayerType::FullAttention; 2]);
        c.text_config.full_attention_interval = 0; // ignored when explicit types exist
        c.validate().unwrap();
        assert_eq!(c.text_config.layer_type(0), LayerType::FullAttention);
        c.text_config.layer_types = Some(vec![]);
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_unsupported_behavior_and_invalid_dimensions() {
        let mutations: &[fn(&mut Qwen3_5Config)] = &[
            |c| c.text_config.attn_output_gate = false,
            |c| c.text_config.hidden_act = "relu".into(),
            |c| c.text_config.linear_num_value_heads = 0,
            |c| c.text_config.linear_num_key_heads = 3,
            |c| c.text_config.full_attention_interval = 0,
            |c| c.text_config.rope_parameters.mrope_interleaved = false,
            |c| c.text_config.rope_parameters.mrope_section = [1, 0, 1],
            |c| c.text_config.rope_parameters.rope_type = "dynamic".into(),
            |c| c.vision_config.deepstack_visual_indexes = vec![0],
            |c| c.vision_config.num_position_embeddings = 15,
            |c| c.vision_config.hidden_size = usize::MAX,
            |c| c.tie_word_embeddings = true,
            |c| c.image_token_id = c.video_token_id,
            |c| c.text_config.rms_norm_eps = f64::NAN,
        ];
        for mutation in mutations {
            let mut config = tiny();
            mutation(&mut config);
            assert!(config.validate().is_err(), "accepted {config:?}");
        }
    }
}
