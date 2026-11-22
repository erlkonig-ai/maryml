//! Inkling configuration, read from the checkpoint's `config.json`.
//!
//! Inkling is a 42-layer (small) / 66-layer (full) sparse MoE decoder with
//! native audio and vision input. Structurally it is *closer to conventional*
//! than Kimi-K3 — plain GQA rather than KDA plus MLA — but it is not plain:
//!
//! * a depthwise **short convolution** (`sconv_kernel_size`, 4) on the
//!   attention input, on K, on V and on the MLP input — but not on Q;
//! * **QK-norm** at `head_dim` width;
//! * a **learned relative-position** logit path of rank `d_rel` (16) over
//!   `rel_extent` (1024) — an additive bias, not RoPE;
//! * **log scaling** past `log_scaling_n_floor` tokens, which is what carries
//!   the 1 M context;
//! * **sliding-window** attention on `local_layer_ids`, full attention on the
//!   complement (for the 42-layer model that complement is
//!   `[5,11,17,23,29,35,41]` — stride 6).
//!
//! Two fields decide *name sets*, not just sizes, so they are load-bearing for
//! the layout: `dense_mlp_idx` (layers below it use `mlp.w13_dn`/`mlp.w2_md`
//! rather than the MoE names) and `shared_expert_sink` (which widens the router
//! from `n_routed_experts` rows to `n_routed_experts + n_shared_experts`).

use serde::Deserialize;

/// Whether a layer attends over a sliding window or the whole sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Sliding window of `sliding_window_size` tokens.
    Local,
    /// Full attention over the sequence.
    Global,
}

/// The text decoder's configuration — `config.json`'s `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct InklingTextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    /// Sliding-window layers have their own head configuration. On the 42-layer
    /// model these equal the global fields and are indistinguishable from them;
    /// on the 66-layer model `swa_num_key_value_heads` is 16 against 8, so a
    /// layout that reads only the global fields is wrong on every local layer.
    #[serde(default)]
    pub swa_num_attention_heads: usize,
    #[serde(default)]
    pub swa_num_key_value_heads: usize,
    #[serde(default)]
    pub swa_head_dim: usize,

    pub vocab_size: usize,
    /// Real vocabulary width. `vocab_size` is padded (201024 against 200058 on
    /// both releases) and the head's output is truncated to this. Zero means
    /// the config did not name it, in which case nothing is dropped.
    #[serde(default)]
    pub unpadded_vocab_size: usize,

    /// Rank of the learned relative-position path (16).
    pub d_rel: usize,
    /// How far the relative-position bias reaches (1024).
    pub rel_extent: usize,

    pub rms_norm_eps: f64,
    #[serde(default)]
    pub use_embed_norm: bool,

    /// Depthwise short-convolution kernel width (4).
    pub sconv_kernel_size: usize,
    #[serde(default)]
    pub use_sconv: bool,

    pub sliding_window_size: usize,
    /// Layers that attend over the window; every other layer is global.
    pub local_layer_ids: Vec<usize>,

    /// Layers `0..dense_mlp_idx` carry a dense MLP and *different tensor
    /// names*; MoE starts here.
    pub dense_mlp_idx: usize,
    pub dense_intermediate_size: usize,
    /// Per-expert intermediate width (2048).
    pub intermediate_size: usize,

    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    /// When set, the shared experts occupy their own rows in the router's
    /// output, so the gate is `n_routed + n_shared` wide.
    #[serde(default)]
    pub shared_expert_sink: bool,

    pub route_scale: f64,
    #[serde(default)]
    pub use_gate_bias: bool,
    pub gate_activation: String,
    #[serde(default)]
    pub norm_after_topk: bool,
    #[serde(default)]
    pub use_global_scale: bool,

    #[serde(default)]
    pub log_scaling_n_floor: usize,
    #[serde(default)]
    pub log_scaling_alpha: f64,
    #[serde(default)]
    pub logits_mup_width_multiplier: f64,
    #[serde(default)]
    pub model_max_length: usize,
}

impl InklingTextConfig {
    /// Heads, KV heads and head width for a layer of this kind. Local layers
    /// read the `swa_*` fields, falling back to the global ones when a config
    /// omits them.
    pub fn heads(&self, kind: AttnKind) -> (usize, usize, usize) {
        match kind {
            AttnKind::Global => (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            ),
            AttnKind::Local => (
                if self.swa_num_attention_heads > 0 {
                    self.swa_num_attention_heads
                } else {
                    self.num_attention_heads
                },
                if self.swa_num_key_value_heads > 0 {
                    self.swa_num_key_value_heads
                } else {
                    self.num_key_value_heads
                },
                if self.swa_head_dim > 0 {
                    self.swa_head_dim
                } else {
                    self.head_dim
                },
            ),
        }
    }

    /// Width of the Q projection in a layer of this kind.
    pub fn q_dim(&self, kind: AttnKind) -> usize {
        let (h, _, d) = self.heads(kind);
        h * d
    }

    /// Width of the K and V projections in a layer of this kind.
    pub fn kv_dim(&self, kind: AttnKind) -> usize {
        let (_, kv, d) = self.heads(kind);
        kv * d
    }

    /// Width of the relative-position projection: one rank-`d_rel` block per
    /// attention head.
    pub fn rel_dim(&self, kind: AttnKind) -> usize {
        let (h, _, _) = self.heads(kind);
        h * self.d_rel
    }

    /// Per-head width used by QK-norm in a layer of this kind.
    pub fn norm_dim(&self, kind: AttnKind) -> usize {
        self.heads(kind).2
    }

    /// The vocabulary the head actually emits, falling back to the padded
    /// width when the config does not name a smaller one.
    pub fn effective_vocab(&self) -> usize {
        if self.unpadded_vocab_size > 0 && self.unpadded_vocab_size <= self.vocab_size {
            self.unpadded_vocab_size
        } else {
            self.vocab_size
        }
    }

    /// Rows of the router matrix. This is the field that makes the router
    /// `[258, hidden]` rather than `[256, hidden]` on the small model.
    pub fn gate_rows(&self) -> usize {
        if self.shared_expert_sink {
            self.n_routed_experts + self.n_shared_experts
        } else {
            self.n_routed_experts
        }
    }

    /// Whether `layer` attends locally or globally.
    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        if self.local_layer_ids.contains(&layer) {
            AttnKind::Local
        } else {
            AttnKind::Global
        }
    }

    /// How far the relative-position bias must reach in a layer of this kind:
    /// a local layer can never see past its window, so its table is
    /// `sliding_window_size` wide, while a global layer's spans `rel_extent`.
    pub fn rel_span(&self, kind: AttnKind) -> usize {
        match kind {
            AttnKind::Local => self.sliding_window_size,
            AttnKind::Global => self.rel_extent,
        }
    }

    /// Whether `layer` carries a dense MLP (and therefore the dense names).
    pub fn is_dense(&self, layer: usize) -> bool {
        layer < self.dense_mlp_idx
    }
}

/// The multi-token-prediction stack — `num_nextn_predict_layers` full
/// transformer blocks, each fed the concatenation of an embedding and a hidden
/// state. Every MTP block is dense regardless of `dense_mlp_idx`.
#[derive(Debug, Clone, Deserialize)]
pub struct InklingMtpConfig {
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub chain_hidden_post_norm: bool,
    #[serde(default)]
    pub local_layer_ids: Vec<usize>,
}

/// dMel audio input: each of `n_mel_bins` bins is discretized to one of
/// `mel_vocab_size` levels, so the "encoder" is an embedding table of
/// `n_mel_bins * mel_vocab_size` rows.
#[derive(Debug, Clone, Deserialize)]
pub struct InklingAudioConfig {
    pub decoder_dmodel: usize,
    pub n_mel_bins: usize,
    pub mel_vocab_size: usize,
    #[serde(default)]
    pub audio_mode: String,
    #[serde(default)]
    pub use_audio_norm: bool,
}

impl InklingMtpConfig {
    /// MTP blocks carry their own local/global split, independent of the LLM's.
    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        if self.local_layer_ids.contains(&layer) {
            AttnKind::Local
        } else {
            AttnKind::Global
        }
    }
}

impl InklingAudioConfig {
    /// Rows of the dMel embedding table.
    pub fn encoder_rows(&self) -> usize {
        self.n_mel_bins * self.mel_vocab_size
    }
}

/// HMLP vision input — a hierarchy of MLPs over sub-patches, not a ViT.
///
/// `hidden_dims` are the widths between stages. They are not named in
/// `vision_config`, so they default to the widths the released checkpoints
/// carry; `inkling_layout_gate` re-derives the pyramid arithmetic from
/// `patch_size`, `n_channels` and `temporal_patch_size` and fails if the chain
/// does not close, so a wrong default cannot pass silently.
#[derive(Debug, Clone, Deserialize)]
pub struct InklingVisionConfig {
    pub decoder_dmodel: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub n_channels: usize,
    pub n_layers: usize,
    #[serde(default = "default_vision_hidden_dims")]
    pub hidden_dims: Vec<usize>,
    #[serde(default = "default_vision_subpatch")]
    pub subpatch_size: usize,
}

fn default_vision_hidden_dims() -> Vec<usize> {
    vec![128, 320, 4800]
}

fn default_vision_subpatch() -> usize {
    5
}

impl InklingVisionConfig {
    /// Elements in one sub-patch: `subpatch^2 * channels`.
    pub fn subpatch_elems(&self) -> usize {
        self.subpatch_size * self.subpatch_size * self.n_channels
    }

    /// Elements in a full patch across the temporal window — the input width of
    /// the pyramid's last stage.
    pub fn patch_elems(&self) -> usize {
        self.patch_size * self.patch_size * self.n_channels * self.temporal_patch_size
    }
}

/// The whole checkpoint's configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct InklingConfig {
    pub text_config: InklingTextConfig,
    pub mtp_config: InklingMtpConfig,
    pub audio_config: InklingAudioConfig,
    pub vision_config: InklingVisionConfig,
}

impl InklingConfig {
    /// Parse a checkpoint's `config.json`.
    ///
    /// Refuses a config of any other lineage before reading a field. Inkling 2
    /// Mini shares this model's tensor names and half its config keys, so a
    /// field-level parse is not a family check: a Mini config fails here only
    /// because it says it is Mini, with a pointer to its own session
    /// (`models::inkling2_loop::session`) and what Mary does not run of it.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        use crate::models::inkling2_loop::{Family, checkpoint, family_of};
        use serde::de::Error;
        let v: serde_json::Value = serde_json::from_str(s)?;
        match family_of(&v).map_err(serde_json::Error::custom)? {
            Family::Inkling => serde_json::from_value(v),
            Family::Inkling2Loop => Err(serde_json::Error::custom(
                checkpoint::refusal_for_config(s),
            )),
        }
    }
}
