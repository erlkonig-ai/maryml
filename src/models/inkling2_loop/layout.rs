//! Checkpoint-name to slot mapping for Inkling 2 Mini, with the shape and
//! dtype each slot must hold.
//!
//! Same discipline as `models::inkling::layout`: [`Slot::tensor_name`] and
//! [`Slot::parse`] are written independently, so composing them is a test and
//! not a tautology, and [`describe`] predicts every slot's shape from the
//! config alone. The precision of each tensor is *not* predicted from the
//! config. `config.json` only says "NVFP4"; the per-tensor truth (which layers
//! hold NVFP4 experts, which MXFP8, which tensors are stored F32) is in
//! `hf_quant_config.json`'s parameter groups, and [`Precisions`] reads it from
//! there. The published release has NVFP4 experts on layers 0 to 19 and MXFP8
//! on 20 to 23, and nothing here depends on that split.
//!
//! The names overlap the original Inkling's (`model.llm.layers.N.attn.wq_du`
//! and friends) while the tensors behind them differ: there is no QK-norm,
//! the expert sidecars are `_scale`/`_amax`/`_outer_scale_denominator` rather
//! than `.scale`/`.scale2`, and Mini adds the attention residual, the
//! cross-pass messages, the sink and the output gate. This is a separate
//! layout so that neither can accept the other's checkpoint by accident.
//!
//! The drafter in `mtp.safetensors` is recognised by name and checked where
//! its shapes follow from `mtp_config`; its Markov head and fuse matrix are
//! reported as unpredicted rather than guessed.

use std::collections::{BTreeMap, BTreeSet};

use super::config::{AttnKind, Inkling2LoopConfig};

/// A tensor dtype, in safetensors' spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dtype {
    Bf16,
    F32,
    /// Packed NVFP4 codes, and MXFP8's UE8M0 scales.
    U8,
    /// NVFP4 block scales and MXFP8 values.
    F8E4M3,
}

impl Dtype {
    pub fn parse(s: &str) -> Option<Dtype> {
        match s {
            "BF16" => Some(Dtype::Bf16),
            "F32" => Some(Dtype::F32),
            "U8" => Some(Dtype::U8),
            "F8_E4M3" => Some(Dtype::F8E4M3),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bf16 => "BF16",
            Dtype::F32 => "F32",
            Dtype::U8 => "U8",
            Dtype::F8E4M3 => "F8_E4M3",
        }
    }
}

/// How one parameter group stores its tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    Bf16,
    /// E2M1 codes two per byte, an E4M3 scale per `block` along the input
    /// axis, and a per-expert `amax / outer_scale_denominator`.
    Nvfp4 {
        block: usize,
    },
    /// E4M3 values with a UE8M0 power-of-two scale per `block` along the
    /// input axis. The group declares `block_rows` x `block`; the stored
    /// scale tensor still has one entry per row (see [`describe`]).
    Mxfp8 {
        block_rows: usize,
        block: usize,
    },
}

/// The per-tensor precision `hf_quant_config.json` declares.
#[derive(Debug, Clone, Default)]
pub struct Precisions {
    by_name: BTreeMap<String, Precision>,
}

impl Precisions {
    /// Read the `model` and `mtp` parameter groups. A group of a kind this
    /// crate does not know is an error: guessing its storage is how a tensor
    /// gets decoded as something it is not.
    pub fn from_hf_quant_config(text: &str) -> Result<Precisions, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("parsing hf_quant_config.json: {e}"))?;
        let mut by_name = BTreeMap::new();
        let mut seen_group = false;
        for part in ["model", "mtp"] {
            let Some(groups) = v
                .get(part)
                .and_then(|p| p.get("parameter_groups"))
                .and_then(|g| g.as_array())
            else {
                continue;
            };
            for (i, g) in groups.iter().enumerate() {
                seen_group = true;
                let p = group_precision(g).map_err(|e| format!("{part} group {i}: {e}"))?;
                let tensors = g
                    .get("tensors")
                    .and_then(|t| t.as_array())
                    .ok_or_else(|| format!("{part} group {i} has no tensors"))?;
                for name in tensors {
                    let name = name
                        .as_str()
                        .ok_or_else(|| format!("{part} group {i} names a non-string tensor"))?;
                    if let Some(prev) = by_name.insert(name.to_string(), p)
                        && prev != p
                    {
                        return Err(format!("{name} is in two groups: {prev:?} and {p:?}"));
                    }
                }
            }
        }
        if !seen_group {
            return Err("hf_quant_config.json has no model or mtp parameter_groups".to_string());
        }
        Ok(Precisions { by_name })
    }

    /// The precision a tensor was declared with. Expert sidecars are not
    /// listed in the groups; they inherit their matrix's precision.
    pub fn of(&self, name: &str) -> Option<Precision> {
        self.by_name.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }
}

fn group_precision(g: &serde_json::Value) -> Result<Precision, String> {
    let s = |k: &str| g.get(k).and_then(|v| v.as_str()).unwrap_or("");
    let block: Vec<usize> = g
        .get("weight_block_shape")
        .and_then(|b| b.as_array())
        .map(|b| {
            b.iter()
                .filter_map(|x| x.as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    match (s("compute_dtype"), s("weight_transform")) {
        ("float32", "cast") => Ok(Precision::F32),
        ("bfloat16", "cast") => Ok(Precision::Bf16),
        ("nvfp4", "nvfp4_nearest") => {
            if s("weight_scale_format") != "amax_denominator" {
                return Err(format!(
                    "NVFP4 with weight_scale_format {:?}, not amax_denominator",
                    s("weight_scale_format")
                ));
            }
            match block.as_slice() {
                [1, b] if *b > 0 => Ok(Precision::Nvfp4 { block: *b }),
                other => Err(format!("NVFP4 weight_block_shape {other:?}")),
            }
        }
        ("mxfp8", "mxfp8_nearest") => match block.as_slice() {
            [r, b] if *r > 0 && *b > 0 => Ok(Precision::Mxfp8 {
                block_rows: *r,
                block: *b,
            }),
            other => Err(format!("MXFP8 weight_block_shape {other:?}")),
        },
        (c, w) => Err(format!(
            "unknown parameter group compute_dtype {c:?} weight_transform {w:?}"
        )),
    }
}

/// The two matrices of a routed expert bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertMat {
    /// Gate and up, rows interleaved: `[experts, 2 * intermediate, hidden]`.
    W13,
    /// Down: `[experts, hidden, intermediate]`.
    W2,
}

/// The pieces a stored expert matrix is split into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertPart {
    /// Codes (NVFP4), values (MXFP8) or the plain matrix (BF16).
    Weight,
    /// Block scales: E4M3 for NVFP4, UE8M0 for MXFP8.
    Scale,
    /// NVFP4 only: per-expert amax, `[experts]`.
    Amax,
    /// NVFP4 only: the shared denominator, `[1]`.
    OuterScaleDenominator,
}

impl ExpertPart {
    fn suffix(self) -> &'static str {
        match self {
            ExpertPart::Weight => "",
            ExpertPart::Scale => "_scale",
            ExpertPart::Amax => "_amax",
            ExpertPart::OuterScaleDenominator => "_outer_scale_denominator",
        }
    }

    /// The parts a matrix of this precision is stored as.
    pub fn of(p: Precision) -> &'static [ExpertPart] {
        match p {
            Precision::Nvfp4 { .. } => &[
                ExpertPart::Weight,
                ExpertPart::Scale,
                ExpertPart::Amax,
                ExpertPart::OuterScaleDenominator,
            ],
            Precision::Mxfp8 { .. } => &[ExpertPart::Weight, ExpertPart::Scale],
            Precision::Bf16 | Precision::F32 => &[ExpertPart::Weight],
        }
    }
}

/// The three convolutions on the attention path: a static depthwise kernel
/// on local layers, and on global layers a kernel predicted from the input
/// (`weight_proj`) plus a static part (`static_w`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConvPart {
    Weight,
    WeightProj,
    StaticW,
}

/// The streams a core layer's message gates feed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stream {
    /// `controllers.attn`, upstream's `q_message_gate`.
    Query,
    Key,
    Value,
    Mlp,
}

/// A message gate's three projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GatePart {
    /// `[rank, 2 * hidden]`, over the normed hidden state and message.
    Down,
    /// `[hidden, rank]`, tanh into `(1 - forget) * hidden`.
    Forget,
    /// `[hidden, rank]`, tanh into `+ gain * message`.
    Gain,
}

/// The attention block's tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttnPart {
    Wq,
    Wk,
    Wv,
    Wo,
    /// The relative-position state, rank `d_rel` per head.
    Wr,
    RelLogitsProj,
    /// Per-head learned scale before the query/key L2 clip (in place of the
    /// original Inkling's QK-norm).
    QL2Clip,
    RelL2Clip,
    /// Sigmoid output gate, `[q_dim, hidden]`.
    GateAttn,
    /// Per-head learned attention sink logit from the input, `[heads, hidden]`.
    SinkProj,
    KSconv(ConvPart),
    VSconv(ConvPart),
}

/// Everything under `model.llm.layers.N.`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LayerPart {
    /// Local layers only: one norm for Q, K and V.
    AttnNorm,
    /// Global layers only: separate norms, because Q, K and V get separate
    /// attention-residual inputs there.
    QInputNorm,
    KInputNorm,
    VInputNorm,
    AttnSconv,
    MlpNorm,
    MlpSconv,
    Attn(AttnPart),
    /// Core layers only.
    Controller(Stream, GatePart),
    GateWeight,
    GateBias,
    Expert(ExpertMat, ExpertPart),
}

/// The ViT vision tower and its projector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VisionPart {
    ClassEmbedding,
    ClsPositionEmbedding,
    PatchEmbedding,
    PositionEmbedding,
    LnPreWeight,
    LnPreBias,
    Layer(usize, VisionLayerPart),
    ProjIn,
    ProjOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VisionLayerPart {
    Ln1Weight,
    Ln1Bias,
    Ln2Weight,
    Ln2Bias,
    Ls1Gamma,
    Ls2Gamma,
    FcWeight,
    FcBias,
    ProjWeight,
    ProjBias,
    /// `q`, `k`, `v` or `out`, and whether it is the bias.
    Attn(VisionProj, bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VisionProj {
    Q,
    K,
    V,
    Out,
}

impl VisionProj {
    const ALL: [VisionProj; 4] = [VisionProj::Q, VisionProj::K, VisionProj::V, VisionProj::Out];
    fn stem(self) -> &'static str {
        match self {
            VisionProj::Q => "q_proj",
            VisionProj::K => "k_proj",
            VisionProj::V => "v_proj",
            VisionProj::Out => "out_proj",
        }
    }
}

/// The drafter in `mtp.safetensors`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MtpPart {
    FuseWeight,
    MaskEmbed,
    Norm,
    Layer(usize, MtpLayerPart),
    MarkovHead(MarkovPart),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MtpLayerPart {
    DflareNorm,
    InputLayernorm,
    PostAttentionLayernorm,
    GateProj,
    UpProj,
    DownProj,
    QProj,
    KProj,
    VProj,
    OProj,
    QNorm,
    KNorm,
    KProjCtx,
    VProjCtx,
    KNormCtx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MarkovPart {
    DflareNorm,
    HiddenOutWeight,
    HiddenOutBias,
    MarkovW1,
    RnnProjWeight,
    RnnProjBias,
}

const MTP_LAYER_PARTS: [(MtpLayerPart, &str); 15] = [
    (MtpLayerPart::DflareNorm, "dflare_norm.weight"),
    (MtpLayerPart::InputLayernorm, "input_layernorm.weight"),
    (
        MtpLayerPart::PostAttentionLayernorm,
        "post_attention_layernorm.weight",
    ),
    (MtpLayerPart::GateProj, "mlp.gate_proj.weight"),
    (MtpLayerPart::UpProj, "mlp.up_proj.weight"),
    (MtpLayerPart::DownProj, "mlp.down_proj.weight"),
    (MtpLayerPart::QProj, "self_attn.q_proj.weight"),
    (MtpLayerPart::KProj, "self_attn.k_proj.weight"),
    (MtpLayerPart::VProj, "self_attn.v_proj.weight"),
    (MtpLayerPart::OProj, "self_attn.o_proj.weight"),
    (MtpLayerPart::QNorm, "self_attn.q_norm.weight"),
    (MtpLayerPart::KNorm, "self_attn.k_norm.weight"),
    (MtpLayerPart::KProjCtx, "self_attn.k_proj_ctx.weight"),
    (MtpLayerPart::VProjCtx, "self_attn.v_proj_ctx.weight"),
    (MtpLayerPart::KNormCtx, "self_attn.k_norm_ctx.weight"),
];

const MARKOV_PARTS: [(MarkovPart, &str); 6] = [
    (MarkovPart::DflareNorm, "dflare_norm.weight"),
    (MarkovPart::HiddenOutWeight, "hidden_out.weight"),
    (MarkovPart::HiddenOutBias, "hidden_out.bias"),
    (MarkovPart::MarkovW1, "markov_w1.weight"),
    (MarkovPart::RnnProjWeight, "rnn_proj.weight"),
    (MarkovPart::RnnProjBias, "rnn_proj.bias"),
];

/// Which file of the release a slot lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum File {
    /// The decoder and vision tower, `model*.safetensors`.
    Model,
    /// The drafter, `mtp.safetensors`, whose names carry no prefix.
    Mtp,
}

/// One tensor of the release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Slot {
    Embed,
    EmbedNorm,
    FinalNorm,
    Unembed,
    /// The attention-residual query consumed by layer `N` (`N >= 1`), stored
    /// under `attn_res.attn_queries.{N - 1}`. Layer 0 has none.
    AttnResQuery(usize),
    /// The attention-residual query before layer `N`'s MLP.
    MlpResQuery(usize),
    /// The attention-residual query before the final norm.
    OutputResQuery,
    /// `[2 * core, hidden]`: the Q and MLP message queries of each core layer.
    CrossPassQueries,
    /// `[2 * core, hidden]`: the K and V message queries of each core layer.
    CrossPassKvQueries,
    Layer(usize, LayerPart),
    Vision(VisionPart),
    Mtp(MtpPart),
}

fn attn_suffix(a: AttnPart) -> String {
    let conv = |c: ConvPart| match c {
        ConvPart::Weight => "weight",
        ConvPart::WeightProj => "weight_proj",
        ConvPart::StaticW => "static_w",
    };
    match a {
        AttnPart::Wq => "wq_du.weight".into(),
        AttnPart::Wk => "wk_dv.weight".into(),
        AttnPart::Wv => "wv_dv.weight".into(),
        AttnPart::Wo => "wo_ud.weight".into(),
        AttnPart::Wr => "wr_du.weight".into(),
        AttnPart::RelLogitsProj => "rel_logits_proj.proj".into(),
        AttnPart::QL2Clip => "q_l2_clip.scale".into(),
        AttnPart::RelL2Clip => "rel_l2_clip.scale".into(),
        AttnPart::GateAttn => "gate_attn.weight".into(),
        AttnPart::SinkProj => "attn_sink_proj.weight".into(),
        AttnPart::KSconv(c) => format!("k_sconv.{}", conv(c)),
        AttnPart::VSconv(c) => format!("v_sconv.{}", conv(c)),
    }
}

fn layer_suffix(p: LayerPart) -> String {
    match p {
        LayerPart::AttnNorm => "attn_norm.weight".into(),
        LayerPart::QInputNorm => "q_input_norm.weight".into(),
        LayerPart::KInputNorm => "k_input_norm.weight".into(),
        LayerPart::VInputNorm => "v_input_norm.weight".into(),
        LayerPart::AttnSconv => "attn_sconv.weight".into(),
        LayerPart::MlpNorm => "mlp_norm.weight".into(),
        LayerPart::MlpSconv => "mlp_sconv.weight".into(),
        LayerPart::Attn(a) => format!("attn.{}", attn_suffix(a)),
        LayerPart::Controller(s, g) => {
            let s = match s {
                Stream::Query => "attn",
                Stream::Key => "k",
                Stream::Value => "v",
                Stream::Mlp => "mlp",
            };
            let g = match g {
                GatePart::Down => "down_weight",
                GatePart::Forget => "forget_weight",
                GatePart::Gain => "gain_weight",
            };
            format!("controllers.{s}.{g}")
        }
        LayerPart::GateWeight => "mlp.gate.weight".into(),
        LayerPart::GateBias => "mlp.gate.bias".into(),
        LayerPart::Expert(m, part) => {
            let m = match m {
                ExpertMat::W13 => "w13_weight",
                ExpertMat::W2 => "w2_weight",
            };
            format!("mlp.experts.{m}{}", part.suffix())
        }
    }
}

impl Slot {
    pub fn file(&self) -> File {
        match self {
            Slot::Mtp(_) => File::Mtp,
            _ => File::Model,
        }
    }

    /// The checkpoint name this slot is stored under. Written independently
    /// of [`Slot::parse`].
    pub fn tensor_name(&self) -> String {
        match self {
            Slot::Embed => "model.llm.embed.weight".into(),
            Slot::EmbedNorm => "model.llm.embed_norm.weight".into(),
            Slot::FinalNorm => "model.llm.norm.weight".into(),
            Slot::Unembed => "model.llm.unembed.weight".into(),
            Slot::AttnResQuery(layer) => {
                format!("model.llm.attn_res.attn_queries.{}", layer - 1)
            }
            Slot::MlpResQuery(layer) => format!("model.llm.attn_res.mlp_queries.{layer}"),
            Slot::OutputResQuery => "model.llm.attn_res.output_query".into(),
            Slot::CrossPassQueries => "model.llm.middle_cycle_cross_pass_queries".into(),
            Slot::CrossPassKvQueries => "model.llm.middle_cycle_cross_pass_kv_queries".into(),
            Slot::Layer(i, p) => format!("model.llm.layers.{i}.{}", layer_suffix(*p)),
            Slot::Vision(v) => {
                let b = "model.visual.backbone";
                match v {
                    VisionPart::ClassEmbedding => format!("{b}.embeddings.class_embedding"),
                    VisionPart::ClsPositionEmbedding => {
                        format!("{b}.embeddings.cls_position_embedding")
                    }
                    VisionPart::PatchEmbedding => format!("{b}.embeddings.patch_embedding.weight"),
                    VisionPart::PositionEmbedding => format!("{b}.embeddings.position_embedding"),
                    VisionPart::LnPreWeight => format!("{b}.ln_pre.weight"),
                    VisionPart::LnPreBias => format!("{b}.ln_pre.bias"),
                    VisionPart::ProjIn => "model.visual.proj_in.weight".into(),
                    VisionPart::ProjOut => "model.visual.proj_out.weight".into(),
                    VisionPart::Layer(i, p) => {
                        let tail = match p {
                            VisionLayerPart::Ln1Weight => "ln_1.weight".to_string(),
                            VisionLayerPart::Ln1Bias => "ln_1.bias".into(),
                            VisionLayerPart::Ln2Weight => "ln_2.weight".into(),
                            VisionLayerPart::Ln2Bias => "ln_2.bias".into(),
                            VisionLayerPart::Ls1Gamma => "ls_1.gamma".into(),
                            VisionLayerPart::Ls2Gamma => "ls_2.gamma".into(),
                            VisionLayerPart::FcWeight => "mlp.c_fc.weight".into(),
                            VisionLayerPart::FcBias => "mlp.c_fc.bias".into(),
                            VisionLayerPart::ProjWeight => "mlp.c_proj.weight".into(),
                            VisionLayerPart::ProjBias => "mlp.c_proj.bias".into(),
                            VisionLayerPart::Attn(p, bias) => format!(
                                "self_attn.{}.{}",
                                p.stem(),
                                if *bias { "bias" } else { "weight" }
                            ),
                        };
                        format!("{b}.layers.{i}.{tail}")
                    }
                }
            }
            Slot::Mtp(m) => match m {
                MtpPart::FuseWeight => "fuse_weight".into(),
                MtpPart::MaskEmbed => "mask_embed".into(),
                MtpPart::Norm => "norm.weight".into(),
                MtpPart::Layer(i, p) => {
                    let (_, tail) = MTP_LAYER_PARTS.iter().find(|(q, _)| q == p).unwrap();
                    format!("layers.{i}.{tail}")
                }
                MtpPart::MarkovHead(p) => {
                    let (_, tail) = MARKOV_PARTS.iter().find(|(q, _)| q == p).unwrap();
                    format!("markov_head.{tail}")
                }
            },
        }
    }

    /// Recover a slot from a name in `file`, or `None` if this layout does not
    /// know the name.
    pub fn parse(file: File, name: &str) -> Option<Slot> {
        match file {
            File::Mtp => parse_mtp(name).map(Slot::Mtp),
            File::Model => parse_model(name),
        }
    }
}

fn index(s: &str) -> Option<usize> {
    // Reject "+1", "01" and the like: a checkpoint name is canonical.
    if s.is_empty() || (s.len() > 1 && s.starts_with('0')) || !s.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    s.parse().ok()
}

fn parse_model(name: &str) -> Option<Slot> {
    match name {
        "model.llm.embed.weight" => return Some(Slot::Embed),
        "model.llm.embed_norm.weight" => return Some(Slot::EmbedNorm),
        "model.llm.norm.weight" => return Some(Slot::FinalNorm),
        "model.llm.unembed.weight" => return Some(Slot::Unembed),
        "model.llm.attn_res.output_query" => return Some(Slot::OutputResQuery),
        "model.llm.middle_cycle_cross_pass_queries" => return Some(Slot::CrossPassQueries),
        "model.llm.middle_cycle_cross_pass_kv_queries" => return Some(Slot::CrossPassKvQueries),
        "model.visual.proj_in.weight" => return Some(Slot::Vision(VisionPart::ProjIn)),
        "model.visual.proj_out.weight" => return Some(Slot::Vision(VisionPart::ProjOut)),
        _ => {}
    }
    if let Some(i) = name.strip_prefix("model.llm.attn_res.attn_queries.") {
        return Some(Slot::AttnResQuery(index(i)? + 1));
    }
    if let Some(i) = name.strip_prefix("model.llm.attn_res.mlp_queries.") {
        return Some(Slot::MlpResQuery(index(i)?));
    }
    if let Some(rest) = name.strip_prefix("model.llm.layers.") {
        let (i, tail) = rest.split_once('.')?;
        return Some(Slot::Layer(index(i)?, parse_layer(tail)?));
    }
    if let Some(rest) = name.strip_prefix("model.visual.backbone.") {
        return parse_vision(rest).map(Slot::Vision);
    }
    None
}

fn parse_layer(tail: &str) -> Option<LayerPart> {
    let fixed = match tail {
        "attn_norm.weight" => Some(LayerPart::AttnNorm),
        "q_input_norm.weight" => Some(LayerPart::QInputNorm),
        "k_input_norm.weight" => Some(LayerPart::KInputNorm),
        "v_input_norm.weight" => Some(LayerPart::VInputNorm),
        "attn_sconv.weight" => Some(LayerPart::AttnSconv),
        "mlp_norm.weight" => Some(LayerPart::MlpNorm),
        "mlp_sconv.weight" => Some(LayerPart::MlpSconv),
        "mlp.gate.weight" => Some(LayerPart::GateWeight),
        "mlp.gate.bias" => Some(LayerPart::GateBias),
        _ => None,
    };
    if fixed.is_some() {
        return fixed;
    }
    if let Some(a) = tail.strip_prefix("attn.") {
        let conv = |c: &str| match c {
            "weight" => Some(ConvPart::Weight),
            "weight_proj" => Some(ConvPart::WeightProj),
            "static_w" => Some(ConvPart::StaticW),
            _ => None,
        };
        let part = match a {
            "wq_du.weight" => AttnPart::Wq,
            "wk_dv.weight" => AttnPart::Wk,
            "wv_dv.weight" => AttnPart::Wv,
            "wo_ud.weight" => AttnPart::Wo,
            "wr_du.weight" => AttnPart::Wr,
            "rel_logits_proj.proj" => AttnPart::RelLogitsProj,
            "q_l2_clip.scale" => AttnPart::QL2Clip,
            "rel_l2_clip.scale" => AttnPart::RelL2Clip,
            "gate_attn.weight" => AttnPart::GateAttn,
            "attn_sink_proj.weight" => AttnPart::SinkProj,
            _ => {
                if let Some(c) = a.strip_prefix("k_sconv.") {
                    AttnPart::KSconv(conv(c)?)
                } else if let Some(c) = a.strip_prefix("v_sconv.") {
                    AttnPart::VSconv(conv(c)?)
                } else {
                    return None;
                }
            }
        };
        return Some(LayerPart::Attn(part));
    }
    if let Some(c) = tail.strip_prefix("controllers.") {
        let (s, g) = c.split_once('.')?;
        let s = match s {
            "attn" => Stream::Query,
            "k" => Stream::Key,
            "v" => Stream::Value,
            "mlp" => Stream::Mlp,
            _ => return None,
        };
        let g = match g {
            "down_weight" => GatePart::Down,
            "forget_weight" => GatePart::Forget,
            "gain_weight" => GatePart::Gain,
            _ => return None,
        };
        return Some(LayerPart::Controller(s, g));
    }
    if let Some(e) = tail.strip_prefix("mlp.experts.") {
        let (mat, rest) = if let Some(r) = e.strip_prefix("w13_weight") {
            (ExpertMat::W13, r)
        } else if let Some(r) = e.strip_prefix("w2_weight") {
            (ExpertMat::W2, r)
        } else {
            return None;
        };
        let part = match rest {
            "" => ExpertPart::Weight,
            "_scale" => ExpertPart::Scale,
            "_amax" => ExpertPart::Amax,
            "_outer_scale_denominator" => ExpertPart::OuterScaleDenominator,
            _ => return None,
        };
        return Some(LayerPart::Expert(mat, part));
    }
    None
}

fn parse_vision(rest: &str) -> Option<VisionPart> {
    match rest {
        "embeddings.class_embedding" => return Some(VisionPart::ClassEmbedding),
        "embeddings.cls_position_embedding" => return Some(VisionPart::ClsPositionEmbedding),
        "embeddings.patch_embedding.weight" => return Some(VisionPart::PatchEmbedding),
        "embeddings.position_embedding" => return Some(VisionPart::PositionEmbedding),
        "ln_pre.weight" => return Some(VisionPart::LnPreWeight),
        "ln_pre.bias" => return Some(VisionPart::LnPreBias),
        _ => {}
    }
    let (i, tail) = rest.strip_prefix("layers.")?.split_once('.')?;
    let i = index(i)?;
    let p = match tail {
        "ln_1.weight" => VisionLayerPart::Ln1Weight,
        "ln_1.bias" => VisionLayerPart::Ln1Bias,
        "ln_2.weight" => VisionLayerPart::Ln2Weight,
        "ln_2.bias" => VisionLayerPart::Ln2Bias,
        "ls_1.gamma" => VisionLayerPart::Ls1Gamma,
        "ls_2.gamma" => VisionLayerPart::Ls2Gamma,
        "mlp.c_fc.weight" => VisionLayerPart::FcWeight,
        "mlp.c_fc.bias" => VisionLayerPart::FcBias,
        "mlp.c_proj.weight" => VisionLayerPart::ProjWeight,
        "mlp.c_proj.bias" => VisionLayerPart::ProjBias,
        _ => {
            let a = tail.strip_prefix("self_attn.")?;
            let (stem, wb) = a.split_once('.')?;
            let p = VisionProj::ALL.into_iter().find(|p| p.stem() == stem)?;
            let bias = match wb {
                "weight" => false,
                "bias" => true,
                _ => return None,
            };
            VisionLayerPart::Attn(p, bias)
        }
    };
    Some(VisionPart::Layer(i, p))
}

fn parse_mtp(name: &str) -> Option<MtpPart> {
    match name {
        "fuse_weight" => return Some(MtpPart::FuseWeight),
        "mask_embed" => return Some(MtpPart::MaskEmbed),
        "norm.weight" => return Some(MtpPart::Norm),
        _ => {}
    }
    if let Some(t) = name.strip_prefix("markov_head.") {
        return MARKOV_PARTS
            .iter()
            .find(|(_, s)| *s == t)
            .map(|(p, _)| MtpPart::MarkovHead(*p));
    }
    let (i, tail) = name.strip_prefix("layers.")?.split_once('.')?;
    let i = index(i)?;
    MTP_LAYER_PARTS
        .iter()
        .find(|(_, s)| *s == tail)
        .map(|(p, _)| MtpPart::Layer(i, *p))
}

/// What fills a slot: a dtype and either a predicted shape or, for the few
/// drafter tensors whose shape `mtp_config` does not determine, none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expect {
    pub dtype: Dtype,
    pub shape: Option<Vec<usize>>,
}

/// The precision governing a slot, by the name the parameter groups use. An
/// expert sidecar inherits its matrix's.
fn precision_of(precisions: &Precisions, slot: Slot) -> Option<Precision> {
    let key = match slot {
        Slot::Layer(i, LayerPart::Expert(m, _)) => {
            Slot::Layer(i, LayerPart::Expert(m, ExpertPart::Weight)).tensor_name()
        }
        other => other.tensor_name(),
    };
    precisions.of(&key)
}

/// What the config and the declared precisions say must fill `slot`, or
/// `None` if this config has no such slot. `Err` names a slot whose
/// precision is undeclared or declared as something it cannot be.
pub fn describe(
    cfg: &Inkling2LoopConfig,
    precisions: &Precisions,
    slot: Slot,
) -> Result<Option<Expect>, String> {
    let Some(shape) = logical_shape(cfg, precisions, slot) else {
        return Ok(None);
    };
    let name = slot.tensor_name();
    let p = precision_of(precisions, slot)
        .ok_or_else(|| format!("{name} has no declared precision in hf_quant_config.json"))?;
    let dense = |dtype| {
        Ok(Some(Expect {
            dtype,
            shape: shape.clone(),
        }))
    };
    match (slot, p) {
        (Slot::Layer(_, LayerPart::Expert(_, part)), p) => {
            let dims = shape.expect("an expert matrix's shape is always predicted");
            let (e, rows, cols) = (dims[0], dims[1], dims[2]);
            if !ExpertPart::of(p).contains(&part) {
                return Ok(None);
            }
            let (dtype, dims) = match (p, part) {
                (Precision::Bf16, ExpertPart::Weight) => (Dtype::Bf16, vec![e, rows, cols]),
                (Precision::Nvfp4 { .. }, ExpertPart::Weight) => {
                    divides(&name, 2, cols)?;
                    (Dtype::U8, vec![e, rows, cols / 2])
                }
                (Precision::Nvfp4 { block }, ExpertPart::Scale) => {
                    divides(&name, block, cols)?;
                    (Dtype::F8E4M3, vec![e, rows, cols / block])
                }
                (Precision::Nvfp4 { .. }, ExpertPart::Amax) => (Dtype::F32, vec![e]),
                (Precision::Nvfp4 { .. }, ExpertPart::OuterScaleDenominator) => {
                    (Dtype::F32, vec![1])
                }
                (Precision::Mxfp8 { .. }, ExpertPart::Weight) => {
                    (Dtype::F8E4M3, vec![e, rows, cols])
                }
                // The group declares square blocks, but the stored scale has
                // one entry per row: upstream's decode reads it as
                // `[experts, rows, cols / block]`, one scale per 32 values of
                // a row. Predict what is stored, which the header then checks.
                (Precision::Mxfp8 { block, .. }, ExpertPart::Scale) => {
                    divides(&name, block, cols)?;
                    (Dtype::U8, vec![e, rows, cols / block])
                }
                (p, part) => return Err(format!("{name}: {part:?} cannot be stored as {p:?}")),
            };
            Ok(Some(Expect {
                dtype,
                shape: Some(dims),
            }))
        }
        (_, Precision::F32) => dense(Dtype::F32),
        (_, Precision::Bf16) => dense(Dtype::Bf16),
        (_, p) => Err(format!(
            "{name} is declared {p:?}, which only routed experts use"
        )),
    }
}

fn divides(name: &str, d: usize, n: usize) -> Result<(), String> {
    if d == 0 || n % d != 0 {
        Err(format!("{name}: width {n} is not a multiple of {d}"))
    } else {
        Ok(())
    }
}

/// The logical shape of a slot; `Some(None)` for a drafter tensor whose shape
/// is not determined by the config; `None` for a slot this config does not
/// have.
fn logical_shape(
    cfg: &Inkling2LoopConfig,
    precisions: &Precisions,
    slot: Slot,
) -> Option<Option<Vec<usize>>> {
    let t = &cfg.text_config;
    let h = t.hidden_size;
    let n = t.num_hidden_layers;
    let core = t.loop_config.core_end - t.loop_config.core_start;
    let k = t.sconv_kernel_size;
    let s = |d: &[usize]| Some(Some(d.to_vec()));
    match slot {
        Slot::Embed | Slot::Unembed => s(&[t.vocab_size, h]),
        Slot::EmbedNorm | Slot::FinalNorm | Slot::OutputResQuery => s(&[h]),
        Slot::AttnResQuery(layer) => {
            if layer == 0 || layer >= n {
                return None;
            }
            // A global layer draws separate Q, K and V inputs.
            match t.attn_kind(layer) {
                AttnKind::Global => s(&[3, h]),
                AttnKind::Local => s(&[h]),
            }
        }
        Slot::MlpResQuery(layer) => (layer < n).then(|| Some(vec![h])),
        Slot::CrossPassQueries | Slot::CrossPassKvQueries => s(&[2 * core, h]),
        Slot::Layer(i, p) => {
            if i >= n {
                return None;
            }
            let kind = t.attn_kind(i);
            let (heads, kv, d) = t.heads(kind);
            let global = kind == AttnKind::Global;
            match p {
                LayerPart::AttnNorm => (!global).then(|| Some(vec![h])),
                LayerPart::QInputNorm | LayerPart::KInputNorm | LayerPart::VInputNorm => {
                    global.then(|| Some(vec![h]))
                }
                LayerPart::AttnSconv | LayerPart::MlpSconv => s(&[h, 1, k]),
                LayerPart::MlpNorm => s(&[h]),
                LayerPart::Attn(a) => match a {
                    AttnPart::Wq | AttnPart::GateAttn => s(&[heads * d, h]),
                    AttnPart::Wk | AttnPart::Wv => s(&[kv * d, h]),
                    AttnPart::Wo => s(&[h, heads * d]),
                    AttnPart::Wr => s(&[heads * t.d_rel, h]),
                    AttnPart::RelLogitsProj => s(&[
                        t.d_rel,
                        if global {
                            t.rel_extent
                        } else {
                            t.sliding_window_size
                        },
                    ]),
                    AttnPart::QL2Clip => s(&[heads, d]),
                    AttnPart::RelL2Clip => s(&[heads, t.d_rel]),
                    AttnPart::SinkProj => s(&[heads, h]),
                    AttnPart::KSconv(c) | AttnPart::VSconv(c) => match (global, c) {
                        (false, ConvPart::Weight) => s(&[kv * d, 1, k]),
                        (true, ConvPart::WeightProj) => s(&[kv * k, h]),
                        (true, ConvPart::StaticW) => s(&[k, kv * d]),
                        _ => None,
                    },
                },
                LayerPart::Controller(_, g) => {
                    if !t.is_core(i) {
                        return None;
                    }
                    let r = t.loop_config.gate_rank;
                    match g {
                        GatePart::Down => s(&[r, 2 * h]),
                        GatePart::Forget | GatePart::Gain => s(&[h, r]),
                    }
                }
                LayerPart::GateWeight => s(&[t.n_routed_experts, h]),
                LayerPart::GateBias => t.use_gate_bias.then(|| Some(vec![t.n_routed_experts])),
                LayerPart::Expert(m, _) => {
                    // Present only if the matrix is declared at all; its
                    // sidecars are decided in `describe` from the precision.
                    precision_of(precisions, slot)?;
                    let e = t.n_routed_experts;
                    let inter = t.intermediate_size;
                    match m {
                        ExpertMat::W13 => s(&[e, 2 * inter, h]),
                        ExpertMat::W2 => s(&[e, h, inter]),
                    }
                }
            }
        }
        Slot::Vision(v) => {
            let vc = &cfg.vision_config;
            let vh = vc.hidden_size;
            let mlp = (vh as f64 * vc.mlp_ratio).round() as usize;
            match v {
                VisionPart::ClassEmbedding
                | VisionPart::ClsPositionEmbedding
                | VisionPart::LnPreWeight
                | VisionPart::LnPreBias => s(&[vh]),
                // RGB: the channel count is not in vision_config.
                VisionPart::PatchEmbedding => {
                    s(&[vh, 3 * vc.native_patch_size * vc.native_patch_size])
                }
                VisionPart::PositionEmbedding => {
                    s(&[vc.posemb_grid_size * vc.posemb_grid_size, vh])
                }
                VisionPart::ProjIn => s(&[
                    vc.projection_hidden_dim,
                    vh * vc.adapter_merge_size * vc.adapter_merge_size,
                ]),
                VisionPart::ProjOut => s(&[vc.text_hidden_size, vc.projection_hidden_dim]),
                VisionPart::Layer(i, p) => {
                    if i >= vc.num_hidden_layers {
                        return None;
                    }
                    match p {
                        VisionLayerPart::FcWeight => s(&[mlp, vh]),
                        VisionLayerPart::FcBias => s(&[mlp]),
                        VisionLayerPart::ProjWeight => s(&[vh, mlp]),
                        VisionLayerPart::Attn(_, false) => s(&[vh, vh]),
                        _ => s(&[vh]),
                    }
                }
            }
        }
        Slot::Mtp(m) => {
            let mc = cfg.mtp_config.as_ref()?;
            let mh = mc.hidden_size;
            match m {
                MtpPart::MaskEmbed | MtpPart::Norm => s(&[mh]),
                MtpPart::FuseWeight | MtpPart::MarkovHead(_) => match m {
                    MtpPart::MarkovHead(MarkovPart::DflareNorm) => s(&[mh]),
                    MtpPart::MarkovHead(MarkovPart::MarkovW1) => s(&[t.vocab_size, mc.markov_rank]),
                    MtpPart::MarkovHead(MarkovPart::HiddenOutWeight) => s(&[mh, mc.markov_rank]),
                    MtpPart::MarkovHead(MarkovPart::HiddenOutBias) => s(&[mh]),
                    // The fuse matrix and the RNN projection have no public
                    // reference that fixes their shape.
                    _ => Some(None),
                },
                MtpPart::Layer(i, p) => {
                    if i >= mc.num_hidden_layers {
                        return None;
                    }
                    let q = mc.num_attention_heads * mc.head_dim;
                    let kvd = mc.num_key_value_heads * mc.head_dim;
                    match p {
                        MtpLayerPart::DflareNorm
                        | MtpLayerPart::InputLayernorm
                        | MtpLayerPart::PostAttentionLayernorm => s(&[mh]),
                        MtpLayerPart::GateProj | MtpLayerPart::UpProj => {
                            s(&[mc.intermediate_size, mh])
                        }
                        MtpLayerPart::DownProj => s(&[mh, mc.intermediate_size]),
                        MtpLayerPart::QProj => s(&[q, mh]),
                        MtpLayerPart::KProj
                        | MtpLayerPart::VProj
                        | MtpLayerPart::KProjCtx
                        | MtpLayerPart::VProjCtx => s(&[kvd, mh]),
                        MtpLayerPart::OProj => s(&[mh, q]),
                        MtpLayerPart::QNorm | MtpLayerPart::KNorm | MtpLayerPart::KNormCtx => {
                            s(&[mc.head_dim])
                        }
                    }
                }
            }
        }
    }
}

/// Every slot the config implies, with what must fill it.
pub fn expected(
    cfg: &Inkling2LoopConfig,
    precisions: &Precisions,
) -> Result<BTreeMap<Slot, Expect>, String> {
    let t = &cfg.text_config;
    let mut slots: Vec<Slot> = vec![
        Slot::Embed,
        Slot::EmbedNorm,
        Slot::FinalNorm,
        Slot::Unembed,
        Slot::OutputResQuery,
        Slot::CrossPassQueries,
        Slot::CrossPassKvQueries,
    ];
    let convs = [ConvPart::Weight, ConvPart::WeightProj, ConvPart::StaticW];
    let mut layer_parts = vec![
        LayerPart::AttnNorm,
        LayerPart::QInputNorm,
        LayerPart::KInputNorm,
        LayerPart::VInputNorm,
        LayerPart::AttnSconv,
        LayerPart::MlpNorm,
        LayerPart::MlpSconv,
        LayerPart::GateWeight,
        LayerPart::GateBias,
    ];
    for a in [
        AttnPart::Wq,
        AttnPart::Wk,
        AttnPart::Wv,
        AttnPart::Wo,
        AttnPart::Wr,
        AttnPart::RelLogitsProj,
        AttnPart::QL2Clip,
        AttnPart::RelL2Clip,
        AttnPart::GateAttn,
        AttnPart::SinkProj,
    ] {
        layer_parts.push(LayerPart::Attn(a));
    }
    for c in convs {
        layer_parts.push(LayerPart::Attn(AttnPart::KSconv(c)));
        layer_parts.push(LayerPart::Attn(AttnPart::VSconv(c)));
    }
    for st in [Stream::Query, Stream::Key, Stream::Value, Stream::Mlp] {
        for g in [GatePart::Down, GatePart::Forget, GatePart::Gain] {
            layer_parts.push(LayerPart::Controller(st, g));
        }
    }
    for m in [ExpertMat::W13, ExpertMat::W2] {
        for p in [
            ExpertPart::Weight,
            ExpertPart::Scale,
            ExpertPart::Amax,
            ExpertPart::OuterScaleDenominator,
        ] {
            layer_parts.push(LayerPart::Expert(m, p));
        }
    }
    for i in 0..t.num_hidden_layers {
        slots.push(Slot::AttnResQuery(i));
        slots.push(Slot::MlpResQuery(i));
        for &p in &layer_parts {
            slots.push(Slot::Layer(i, p));
        }
    }
    let vc = &cfg.vision_config;
    for v in [
        VisionPart::ClassEmbedding,
        VisionPart::ClsPositionEmbedding,
        VisionPart::PatchEmbedding,
        VisionPart::PositionEmbedding,
        VisionPart::LnPreWeight,
        VisionPart::LnPreBias,
        VisionPart::ProjIn,
        VisionPart::ProjOut,
    ] {
        slots.push(Slot::Vision(v));
    }
    for i in 0..vc.num_hidden_layers {
        let mut parts = vec![
            VisionLayerPart::Ln1Weight,
            VisionLayerPart::Ln1Bias,
            VisionLayerPart::Ln2Weight,
            VisionLayerPart::Ln2Bias,
            VisionLayerPart::Ls1Gamma,
            VisionLayerPart::Ls2Gamma,
            VisionLayerPart::FcWeight,
            VisionLayerPart::FcBias,
            VisionLayerPart::ProjWeight,
            VisionLayerPart::ProjBias,
        ];
        for p in VisionProj::ALL {
            parts.push(VisionLayerPart::Attn(p, false));
            parts.push(VisionLayerPart::Attn(p, true));
        }
        for p in parts {
            slots.push(Slot::Vision(VisionPart::Layer(i, p)));
        }
    }
    if let Some(mc) = &cfg.mtp_config {
        slots.push(Slot::Mtp(MtpPart::FuseWeight));
        slots.push(Slot::Mtp(MtpPart::MaskEmbed));
        slots.push(Slot::Mtp(MtpPart::Norm));
        for (p, _) in MARKOV_PARTS {
            slots.push(Slot::Mtp(MtpPart::MarkovHead(p)));
        }
        for i in 0..mc.num_hidden_layers {
            for (p, _) in MTP_LAYER_PARTS {
                slots.push(Slot::Mtp(MtpPart::Layer(i, p)));
            }
        }
    }
    let mut out = BTreeMap::new();
    for slot in slots {
        if let Some(e) = describe(cfg, precisions, slot)? {
            out.insert(slot, e);
        }
    }
    Ok(out)
}

/// Every name `hf_quant_config.json` declares that no expected slot covers:
/// a declared tensor this layout would silently drop.
pub fn undeclared_by_layout(
    expected: &BTreeMap<Slot, Expect>,
    precisions: &Precisions,
) -> BTreeSet<String> {
    let names: BTreeSet<String> = expected.keys().map(Slot::tensor_name).collect();
    precisions
        .names()
        .filter(|n| !names.contains(*n))
        .map(str::to_string)
        .collect()
}
