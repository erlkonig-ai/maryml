//! Which Inkling a `config.json` describes, and Inkling 2 Mini's configuration.
//!
//! Two different models share the `model.llm.*` tensor namespace and most of
//! its leaf names (`wq_du`, `wk_dv`, `mlp_sconv`, `mlp.gate.weight`,
//! `mlp.experts.w13_weight`), so a name-level loader cannot tell them apart.
//! What separates them is the config's `model_type`:
//!
//! * `inkling_mm_model` is the original Inkling (the 42-layer small and the
//!   66-layer full release), which `models::inkling` runs;
//! * `inkling2_loop` is Inkling 2 Mini, a 24-layer looped decoder.
//!
//! [`family_of`] decides between them before anything parses the rest. A
//! config with no `model_type` is the shape of Mary's older test fixtures and
//! keeps meaning the original Inkling, unless it carries one of Mini's
//! markers. Every other `model_type` (including `inkling2`, Inkling 2 Small)
//! is refused by name rather than read as either.

use std::collections::BTreeSet;

use serde::Deserialize;

/// The original Inkling's `model_type`.
pub const INKLING_MODEL_TYPE: &str = "inkling_mm_model";
/// Inkling 2 Mini's `model_type`.
pub const INKLING2_LOOP_MODEL_TYPE: &str = "inkling2_loop";
/// Inkling 2 Mini's `text_config.model_type`.
pub const INKLING2_LOOP_TEXT_MODEL_TYPE: &str = "inkling2_loop_text";

/// The model lineage a `config.json` describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// The original Inkling, `models::inkling`.
    Inkling,
    /// Inkling 2 Mini, this module.
    Inkling2Loop,
}

/// Why a config does not name exactly one family this crate knows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FamilyError {
    /// The document is not a JSON object.
    NotAnObject,
    /// A `model_type` neither lineage owns.
    Unrecognized { model_type: String },
    /// `model_type` names one lineage while another field names the other.
    Contradiction { model_type: String, marker: String },
}

impl std::fmt::Display for FamilyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FamilyError::NotAnObject => write!(f, "config.json is not a JSON object"),
            FamilyError::Unrecognized { model_type } if model_type == "inkling2" => write!(
                f,
                "model_type \"inkling2\" is Inkling 2 Small, which Mary does not implement; \
                 it is neither the original Inkling (\"{INKLING_MODEL_TYPE}\") nor \
                 Inkling 2 Mini (\"{INKLING2_LOOP_MODEL_TYPE}\")"
            ),
            FamilyError::Unrecognized { model_type } => write!(
                f,
                "model_type {model_type:?} is not an Inkling this crate knows \
                 (\"{INKLING_MODEL_TYPE}\" or \"{INKLING2_LOOP_MODEL_TYPE}\")"
            ),
            FamilyError::Contradiction { model_type, marker } => write!(
                f,
                "config.json says model_type {model_type:?} but also carries {marker}, \
                 which belongs to the other Inkling lineage"
            ),
        }
    }
}

impl std::error::Error for FamilyError {}

/// The fields that only Inkling 2 Mini's config carries, named for errors.
fn loop_markers(v: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(archs) = v.get("architectures").and_then(|a| a.as_array()) {
        for a in archs.iter().filter_map(|a| a.as_str()) {
            if a.starts_with("Inkling2Loop") {
                out.push(format!("architecture {a:?}"));
            }
        }
    }
    let text = v.get("text_config");
    if let Some(t) = text
        .and_then(|t| t.get("model_type"))
        .and_then(|t| t.as_str())
        && t == INKLING2_LOOP_TEXT_MODEL_TYPE
    {
        out.push(format!("text_config.model_type {t:?}"));
    }
    if text.and_then(|t| t.get("loop_config")).is_some() {
        out.push("text_config.loop_config".to_string());
    }
    out
}

/// The fields that only the original Inkling's config carries.
fn inkling_markers(v: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(archs) = v.get("architectures").and_then(|a| a.as_array()) {
        for a in archs.iter().filter_map(|a| a.as_str()) {
            if a == "InklingForConditionalGeneration" {
                out.push(format!("architecture {a:?}"));
            }
        }
    }
    if v.get("text_config")
        .and_then(|t| t.get("dense_mlp_idx"))
        .is_some()
    {
        out.push("text_config.dense_mlp_idx".to_string());
    }
    if v.get("audio_config").is_some() {
        out.push("audio_config".to_string());
    }
    out
}

/// Decide which lineage a parsed `config.json` belongs to.
pub fn family_of(v: &serde_json::Value) -> Result<Family, FamilyError> {
    if !v.is_object() {
        return Err(FamilyError::NotAnObject);
    }
    let model_type = v.get("model_type").and_then(|m| m.as_str());
    let loops = loop_markers(v);
    match model_type {
        Some(INKLING2_LOOP_MODEL_TYPE) => match inkling_markers(v).into_iter().next() {
            Some(marker) => Err(FamilyError::Contradiction {
                model_type: INKLING2_LOOP_MODEL_TYPE.to_string(),
                marker,
            }),
            None => Ok(Family::Inkling2Loop),
        },
        Some(INKLING_MODEL_TYPE) => match loops.into_iter().next() {
            Some(marker) => Err(FamilyError::Contradiction {
                model_type: INKLING_MODEL_TYPE.to_string(),
                marker,
            }),
            None => Ok(Family::Inkling),
        },
        Some(other) => Err(FamilyError::Unrecognized {
            model_type: other.to_string(),
        }),
        // Mary's own fixtures predate the check and carry no model_type. They
        // keep their meaning; a Mini-shaped config without one does not get
        // to borrow it.
        None if !loops.is_empty() => Ok(Family::Inkling2Loop),
        None => Ok(Family::Inkling),
    }
}

/// [`family_of`] on the document's text.
pub fn family_of_str(text: &str) -> Result<Family, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("parsing config.json: {e}"))?;
    family_of(&v).map_err(|e| e.to_string())
}

/// `text_config.loop_config`: which physical layers form the looped core and
/// how many times it runs.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct LoopConfig {
    /// First looped physical layer.
    pub core_start: usize,
    /// One past the last looped physical layer.
    pub core_end: usize,
    /// Passes through the core. The research `varloop` checkpoint may be run
    /// with fewer; the schedule is a function of this number either way.
    pub num_passes: usize,
    /// Rank of the four message gates on every core layer.
    pub gate_rank: usize,
    #[serde(default)]
    pub is_varloop: bool,
}

/// Inkling 2 Mini's decoder configuration, `config.json`'s `text_config`.
///
/// Fields carry the checkpoint's own spelling. Defaults are only given where
/// upstream `Inkling2LoopTextConfig` has one *and* the field changes no tensor
/// shape; anything that decides a shape is required.
#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2LoopTextConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,

    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub swa_head_dim: usize,
    pub sliding_window_size: usize,
    /// Layers that attend over the window. Upstream derives the same split
    /// from `(i + 1) % 6`; [`Inkling2LoopConfig::validate`] requires the two
    /// to agree instead of trusting either.
    pub local_layer_ids: Vec<usize>,
    /// Upstream's explicit override of the split, when a config carries one.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    pub d_rel: usize,
    pub rel_extent: usize,
    #[serde(default)]
    pub q_bias: bool,
    #[serde(default)]
    pub o_bias: bool,
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub use_value_norm_no_gamma: bool,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,

    pub sconv_kernel_size: usize,
    pub n_layers_per_attn_res_block: usize,
    #[serde(default)]
    pub attn_res_normalize_values: bool,

    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    /// Per-expert intermediate width.
    pub intermediate_size: usize,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    pub route_scale: f64,
    #[serde(default)]
    pub use_gate_bias: bool,
    pub final_logit_softcapping: f64,

    #[serde(default)]
    pub model_max_length: usize,
    pub loop_config: LoopConfig,
}

/// Whether a layer attends over a sliding window or the whole sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttnKind {
    Local,
    Global,
}

impl Inkling2LoopTextConfig {
    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        if self.local_layer_ids.contains(&layer) {
            AttnKind::Local
        } else {
            AttnKind::Global
        }
    }

    /// Heads, KV heads and head width for a layer of this kind.
    pub fn heads(&self, kind: AttnKind) -> (usize, usize, usize) {
        match kind {
            AttnKind::Global => (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            ),
            AttnKind::Local => (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            ),
        }
    }

    /// Whether `layer` is one of the looped core's physical layers.
    pub fn is_core(&self, layer: usize) -> bool {
        (self.loop_config.core_start..self.loop_config.core_end).contains(&layer)
    }

    /// Whether `layer` closes a block of the attention residual: the hidden
    /// state *entering* it is appended to the block states (upstream's
    /// `is_block_boundary`, `layer_idx % n_layers_per_attn_res_block == 0`).
    pub fn is_block_boundary(&self, layer: usize) -> bool {
        layer % self.n_layers_per_attn_res_block == 0
    }

    /// Layer visits per token, counting every pass through the core.
    pub fn logical_layers(&self) -> usize {
        let core = self.loop_config.core_end - self.loop_config.core_start;
        self.num_hidden_layers + (self.loop_config.num_passes - 1) * core
    }
}

/// Inkling 2 Mini's vision tower, `config.json`'s `vision_config`: a ViT with
/// a class token, not the original Inkling's HMLP pyramid.
#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2VisionConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    pub text_hidden_size: usize,
    pub temporal_patch_size: usize,
    pub native_patch_size: usize,
    pub adapter_merge_size: usize,
    pub projection_hidden_dim: usize,
    #[serde(default)]
    pub projector_bias: bool,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub mlp_ratio: f64,
    pub posemb_grid_size: usize,
    pub layer_norm_eps: f64,
}

/// The drafter shipped as `mtp.safetensors`: three dense Llama-named blocks
/// over chosen target layers plus a Markov head. Not the original Inkling's
/// MTP stack.
#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2MtpConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub local_layer_ids: Vec<usize>,
    #[serde(default)]
    pub block_size: usize,
    pub markov_rank: usize,
    #[serde(default)]
    pub markov_head_type: String,
    /// Logical layer visits the drafter reads hidden states from.
    pub target_layer_ids: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    #[serde(default)]
    pub quant_algo: String,
}

/// The whole of Inkling 2 Mini's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2LoopConfig {
    pub model_type: Option<String>,
    #[serde(default)]
    pub eos_token_id: Option<u64>,
    pub text_config: Inkling2LoopTextConfig,
    pub vision_config: Inkling2VisionConfig,
    pub mtp_config: Option<Inkling2MtpConfig>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
    /// The image placeholder (`<|unused_200054|>`): each image token's slot
    /// in the text stream, replaced by its projected feature. The release's
    /// `config.json` does not set it, so upstream's default applies
    /// (`Inkling2LoopConfig.image_token_id`, transformers 046bdb8).
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
}

/// Upstream's `Inkling2LoopConfig.image_token_id` default.
pub const DEFAULT_IMAGE_TOKEN_ID: u32 = 200_054;

fn default_image_token_id() -> u32 {
    DEFAULT_IMAGE_TOKEN_ID
}

impl Inkling2LoopConfig {
    /// Parse and validate a `config.json`, refusing any other family first.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("parsing config.json: {e}"))?;
        match family_of(&v).map_err(|e| e.to_string())? {
            Family::Inkling2Loop => {}
            Family::Inkling => {
                return Err("config.json is the original Inkling (models::inkling), \
                            not Inkling 2 Mini"
                    .to_string());
            }
        }
        let cfg: Inkling2LoopConfig = serde_json::from_value(v)
            .map_err(|e| format!("config.json is not an Inkling 2 Mini config: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Upstream's `validate_architecture` checks, plus the agreements this
    /// crate relies on and upstream only assumes.
    pub fn validate(&self) -> Result<(), String> {
        let t = &self.text_config;
        let l = &t.loop_config;
        let block = t.n_layers_per_attn_res_block;
        let mut errs = Vec::new();
        if let Some(mt) = &t.model_type
            && mt != INKLING2_LOOP_TEXT_MODEL_TYPE
        {
            errs.push(format!("text_config.model_type is {mt:?}"));
        }
        if block == 0 {
            errs.push("n_layers_per_attn_res_block is 0".to_string());
        }
        if l.num_passes < 1 {
            errs.push(format!("loop num_passes {} < 1", l.num_passes));
        }
        if l.core_start == 0 {
            errs.push("loop core_start must be > 0".to_string());
        }
        if l.core_start >= l.core_end || l.core_end > t.num_hidden_layers {
            errs.push(format!(
                "loop core {}..{} is not a non-empty range inside {} layers",
                l.core_start, l.core_end, t.num_hidden_layers
            ));
        }
        if block > 0 && (l.core_start % block != 0 || l.core_end % block != 0) {
            errs.push(format!(
                "loop core {}..{} is not aligned to attention-residual blocks of {block}",
                l.core_start, l.core_end
            ));
        }
        // The local/global split: local_layer_ids, upstream's derivation and
        // an explicit layer_types must all say the same thing.
        let local: BTreeSet<usize> = t.local_layer_ids.iter().copied().collect();
        if local.len() != t.local_layer_ids.len() {
            errs.push("local_layer_ids repeats a layer".to_string());
        }
        if let Some(&bad) = local.iter().find(|&&i| i >= t.num_hidden_layers) {
            errs.push(format!(
                "local_layer_ids names layer {bad} of {}",
                t.num_hidden_layers
            ));
        }
        let upstream_local: BTreeSet<usize> = match &t.layer_types {
            Some(types) => {
                if types.len() != t.num_hidden_layers {
                    errs.push(format!(
                        "layer_types has {} entries for {} layers",
                        types.len(),
                        t.num_hidden_layers
                    ));
                }
                for ty in types {
                    if ty != "hybrid" && ty != "hybrid_sliding" {
                        errs.push(format!("layer_types holds {ty:?}"));
                    }
                }
                types
                    .iter()
                    .enumerate()
                    .filter(|(_, ty)| *ty == "hybrid_sliding")
                    .map(|(i, _)| i)
                    .collect()
            }
            None => (0..t.num_hidden_layers)
                .filter(|i| (i + 1) % 6 != 0)
                .collect(),
        };
        if upstream_local != local {
            errs.push(format!(
                "local_layer_ids {:?} disagrees with the split upstream derives {:?}",
                local, upstream_local
            ));
        }
        if self.vision_config.text_hidden_size != t.hidden_size {
            errs.push(format!(
                "vision text_hidden_size {} != hidden_size {}",
                self.vision_config.text_hidden_size, t.hidden_size
            ));
        }
        if self.image_token_id as usize >= t.vocab_size {
            errs.push(format!(
                "image_token_id {} is outside the {}-row embedding",
                self.image_token_id, t.vocab_size
            ));
        }
        if t.unpadded_vocab_size > t.vocab_size {
            errs.push(format!(
                "unpadded_vocab_size {} > vocab_size {}",
                t.unpadded_vocab_size, t.vocab_size
            ));
        }
        if t.num_experts_per_tok > t.n_routed_experts {
            errs.push("num_experts_per_tok exceeds n_routed_experts".to_string());
        }
        for (name, kind) in [("global", AttnKind::Global), ("local", AttnKind::Local)] {
            let (h, kv, _) = t.heads(kind);
            if kv == 0 || h % kv != 0 {
                errs.push(format!(
                    "{name} heads {h} are not a multiple of KV heads {kv}"
                ));
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "invalid Inkling 2 Mini config: {}",
                errs.join("; ")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = include_str!("fixtures/config.json");
    const SMALL: &str = include_str!("fixtures/inkling_small_config.json");

    fn edit(text: &str, f: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut v: serde_json::Value = serde_json::from_str(text).unwrap();
        f(&mut v);
        v.to_string()
    }

    #[test]
    fn the_image_placeholder_is_upstreams_default_unless_set() {
        // the release does not set it: upstream's Inkling2LoopConfig default
        assert!(!MINI.contains("image_token_id"));
        let c = Inkling2LoopConfig::from_json(MINI).unwrap();
        assert_eq!(c.image_token_id, DEFAULT_IMAGE_TOKEN_ID);
        assert_eq!(c.image_token_id, 200_054);
        let set = edit(MINI, |v| v["image_token_id"] = 7.into());
        assert_eq!(Inkling2LoopConfig::from_json(&set).unwrap().image_token_id, 7);
        let outside = edit(MINI, |v| v["image_token_id"] = 1_000_000.into());
        let e = Inkling2LoopConfig::from_json(&outside).unwrap_err();
        assert!(e.contains("image_token_id"), "{e}");
    }

    #[test]
    fn the_published_configs_name_their_own_family() {
        assert_eq!(family_of_str(MINI), Ok(Family::Inkling2Loop));
        assert_eq!(family_of_str(SMALL), Ok(Family::Inkling));
    }

    #[test]
    fn fixtures_without_model_type_keep_meaning_the_original_inkling() {
        // The shape of Mary's own older fixtures: no model_type at all.
        let legacy = r#"{"text_config":{"dense_mlp_idx":2},"mtp_config":{}}"#;
        assert_eq!(family_of_str(legacy), Ok(Family::Inkling));
        let small_untyped = edit(SMALL, |v| {
            v.as_object_mut().unwrap().remove("model_type");
            v.as_object_mut().unwrap().remove("architectures");
        });
        assert_eq!(family_of_str(&small_untyped), Ok(Family::Inkling));
        // ...but Mini without a model_type is still Mini.
        let mini_untyped = edit(MINI, |v| {
            v.as_object_mut().unwrap().remove("model_type");
        });
        assert_eq!(family_of_str(&mini_untyped), Ok(Family::Inkling2Loop));
    }

    #[test]
    fn other_and_contradictory_model_types_are_refused() {
        let small2 = edit(MINI, |v| v["model_type"] = "inkling2".into());
        let e = family_of_str(&small2).unwrap_err();
        assert!(e.contains("Inkling 2 Small"), "{e}");
        let disguised = edit(MINI, |v| v["model_type"] = INKLING_MODEL_TYPE.into());
        assert!(
            family_of_str(&disguised)
                .unwrap_err()
                .contains("other Inkling lineage")
        );
        let reverse = edit(SMALL, |v| v["model_type"] = INKLING2_LOOP_MODEL_TYPE.into());
        assert!(
            family_of_str(&reverse)
                .unwrap_err()
                .contains("other Inkling lineage")
        );
        assert!(family_of_str("[1]").is_err());
    }

    #[test]
    fn the_published_mini_config_parses_and_validates() {
        let c = Inkling2LoopConfig::from_json(MINI).unwrap();
        let t = &c.text_config;
        assert_eq!((t.hidden_size, t.num_hidden_layers), (2304, 24));
        assert_eq!(t.loop_config.core_start..t.loop_config.core_end, 6..18);
        assert_eq!(t.loop_config.num_passes, 3);
        assert_eq!(t.logical_layers(), 48);
        let global: Vec<usize> = (0..24)
            .filter(|&i| t.attn_kind(i) == AttnKind::Global)
            .collect();
        assert_eq!(global, vec![5, 11, 17, 23]);
        assert!(Inkling2LoopConfig::from_json(SMALL).is_err());
    }

    #[test]
    fn a_split_that_disagrees_with_upstream_is_refused() {
        let moved = edit(MINI, |v| {
            let ids = v["text_config"]["local_layer_ids"].as_array_mut().unwrap();
            ids.retain(|i| i.as_u64() != Some(4));
            ids.push(5.into());
        });
        let e = Inkling2LoopConfig::from_json(&moved).unwrap_err();
        assert!(e.contains("disagrees"), "{e}");
        let misaligned = edit(MINI, |v| {
            v["text_config"]["loop_config"]["core_start"] = 4.into()
        });
        assert!(
            Inkling2LoopConfig::from_json(&misaligned)
                .unwrap_err()
                .contains("aligned")
        );
    }
}
