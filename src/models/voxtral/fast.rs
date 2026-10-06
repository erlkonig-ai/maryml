//! Folded realtime lane for the stt — the qwen3tts playbook applied to
//! Voxtral. Same math as the parity-first layout in `encoder.rs`/`decoder.rs`
//! (gated token-identical in f32 by `voxtral_probe --lane fold`), laid out for
//! op-count and weight traffic instead of oracle-mirroring:
//!
//! - one **wide fused matmul** `[q‖k | R(q‖k) | v]` per attention, with
//!   rotate_half pre-applied to the qk weight ROWS — RoPE becomes
//!   `qk·cos + qkR·sin`, no narrow/cat on activations. The encoder's biases
//!   ride along as `[b_q‖0 | R(b_q)‖0 | b_v]` (RoPE applies after the bias,
//!   and rope is linear, so the rotated block carries the rotated bias);
//! - the preceding RMSNorm **weights** live folded into the consuming matmul
//!   rows (attention norm → wide qkv; encoder MLP norm → gate‖up; encoder
//!   final norm → projector rows, tiled ×4 across the frame-stack; decoder
//!   final norm → tied lm_head rows). Decoder post-attention norm weights
//!   fold into the per-session ada scales instead (they share the same
//!   elementwise slot). Norms in the layers are weightless rsqrt chains with
//!   f32 variance (f16 overflows on activation outliers);
//! - 1/√d pre-scaled into the q rows (and q bias) — no score scaling op;
//! - gate‖up fused into one matmul;
//! - single-token decode folds the GQA groups onto the query axis
//!   (`[B,H,1,D] ≅ [B,Hkv,G,D]`) — no kv expand, no mask;
//! - causal/sliding-window masks are built **once per stack forward** (the
//!   raw lane builds one per layer) and only when `l > 1`;
//! - KV caches trim to the sliding window (`keep = window − 1`, see
//!   [`FastKv`]) — bounded memory and bounded per-frame `cat` traffic on
//!   arbitrarily long sessions, and the l==1 path provably needs no mask.
//!
//! Runs on any backend; the realtime lanes are `RealtimeTranscriber<BFusedHalf>` (fusion
//! + f16 weights, `--lane half`) and `RealtimeTranscriber<BHalf>` (raw unfused f16,
//! `--lane rawhalf` — same folded graph, loaded ZERO-COPY: every f16 leaf
//! aliases the native collection mmap straight onto the GPU, the fold transforms
//! read the pile's own pages, and the embed table stays file-backed for the
//! process life). One frozen model-collection snapshot must contain the exact
//! f32 and full derived f16 roots with identical name/shape domains; missing or
//! ambiguous roots fail before model construction. Word-exactness gate:
//! `voxtral_listen` de_short@480 ms 13/13.

use burn::prelude::*;
use burn::tensor::FloatDType;
use burn::tensor::activation::{gelu, silu, softmax};

use super::config::*;
use super::decoder::{AdaScales, time_embedding};
use super::encoder::CausalConv;
use super::layers::{Embedding, Linear, RopeTable};
use super::mel::VoxtralMel;
use super::pipeline::SttPipeline;
use super::tokenizer::Tekken;
use crate::nn::weight_loader::WeightLoader;

/// Weightless RMS normalization: `x · rsqrt(mean(x²)+eps)`; the variance
/// chain runs in f32 and casts back.
pub(super) fn rms<B: Backend>(x: Tensor<B, 3>, eps: f64) -> Tensor<B, 3> {
    #[cfg(feature = "voxtral-cuda")]
    if let Some(normalized) = super::norm_cuda::try_rms(&x, eps) {
        return normalized;
    }
    let dt = x.dtype();
    let x32 = x.cast(FloatDType::F32);
    let var = x32.clone().powf_scalar(2.0).mean_dim(2);
    x32.mul(var.add_scalar(eps).sqrt().recip()).cast(dt)
}

/// Pre-transposed matmul weight `[1, in, out]` with an optional per-input
/// scale folded into the rows (absorbing the preceding RMSNorm's weight).
fn linear_t<B: Backend>(
    loader: &WeightLoader,
    name: &str,
    fold_in: Option<Tensor<B, 1>>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let w: Tensor<B, 2> = loader.load_tensor(&format!("{name}.weight"), device); // [out, in]
    let [o, i] = w.dims();
    let wt = w.transpose();
    let wt = match fold_in {
        Some(s) => wt.mul(s.reshape([i, 1])),
        None => wt,
    };
    wt.reshape([1, i, o])
}

/// Only the folded lane's O/down projections are candidates. Eligibility is
/// checked against actual RawHalf storage; other cases keep Linear::forward.
pub(super) fn output_projection<B: Backend>(linear: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    #[cfg(feature = "voxtral-cuda")]
    if let Some(y) = super::gemv_cuda::try_project(&x, &linear.weight_t)
        .unwrap_or_else(|error| panic!("Voxtral O/down GEMV launch failed: {error:?}"))
    {
        return match &linear.bias {
            Some(bias) => y + bias.clone(),
            None => y,
        };
    }
    linear.forward(x)
}

/// Sliding-window KV cache with absolute-position bookkeeping. Stores at most
/// `window − 1` trailing keys: (a) every dropped key satisfies
/// `q − j ≥ window` for all FUTURE queries (safe to drop), and (b) after an
/// l==1 update `lk ≤ window`, so the single-query decode path needs no mask.
/// For l>1 updates the (per-forward) mask handles both in-block causality and
/// any key that outlived the window between trims.
pub struct FastKv<B: Backend> {
    k: Option<Tensor<B, 4>>,
    v: Option<Tensor<B, 4>>,
    /// Absolute positions processed so far (≥ stored length once trimming).
    pub pos: usize,
    keep: usize,
}

impl<B: Backend> FastKv<B> {
    pub fn new(window: usize) -> Self {
        Self {
            k: None,
            v: None,
            pos: 0,
            keep: window - 1,
        }
    }

    fn stored(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[2])
    }

    /// Key count the next `update` of `l` positions will attend over.
    pub fn next_lk(&self, l: usize) -> usize {
        self.stored() + l
    }

    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let l = k.dims()[2];
        let (fk, fv) = match (self.k.take(), self.v.take()) {
            (Some(pk), Some(pv)) => (Tensor::cat(vec![pk, k], 2), Tensor::cat(vec![pv, v], 2)),
            _ => (k, v),
        };
        self.pos += l;
        let lk = fk.dims()[2];
        if lk > self.keep {
            self.k = Some(fk.clone().narrow(2, lk - self.keep, self.keep));
            self.v = Some(fv.clone().narrow(2, lk - self.keep, self.keep));
        } else {
            self.k = Some(fk.clone());
            self.v = Some(fv.clone());
        }
        (fk, fv)
    }
}

/// Per-layer sliding-window caches for one stack.
pub struct FastCaches<B: Backend>(pub Vec<FastKv<B>>);

/// Block-causal + sliding-window mask over ABSOLUTE positions: query `i` sits
/// at `pos + i`, key `j` at `(pos + l) − lk + j`. `None` when a single query
/// attends only its (window-trimmed) past — no mask needed.
fn build_mask<B: Backend>(
    l: usize,
    lk: usize,
    pos: usize,
    window: usize,
    device: &B::Device,
) -> Option<Tensor<B, 2, Bool>> {
    if l == 1 {
        return None;
    }
    let first_key = (pos + l) as isize - lk as isize;
    let mut blocked = vec![false; l * lk];
    for i in 0..l {
        let q = (pos + i) as isize;
        for j in 0..lk {
            let ja = first_key + j as isize;
            blocked[i * lk + j] = ja > q || q - ja >= window as isize;
        }
    }
    Some(Tensor::<B, 2, Bool>::from_data(
        burn::tensor::TensorData::new(blocked, [l, lk]),
        device,
    ))
}

/// Folded attention: wide fused qkv with pre-rotated rows, biases riding as
/// `[b‖R(b)‖b_v]`, 1/√d in the q rows, GQA group-fold on single-token steps.
struct FastAttention<B: Backend> {
    wide_t: Tensor<B, 3>,            // [1, hidden, (2(h+hkv)+hkv)·d]
    wide_bias: Option<Tensor<B, 3>>, // [1, 1, same]
    o_proj: Linear<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> FastAttention<B> {
    fn load(
        loader: &WeightLoader,
        prefix: &str,
        (h, hkv, d): (usize, usize, usize),
        qvo_bias: bool,
        fold_in: Tensor<B, 1>,
        device: &B::Device,
    ) -> Self {
        let half = d / 2;
        let n_out = (2 * (h + hkv) + hkv) * d;
        let w2 = |n: &str| -> Tensor<B, 2> {
            loader.load_tensor(&format!("{prefix}.{n}.weight"), device)
        };
        let scale = (d as f64).powf(-0.5);

        // rotate_half on OUTPUT rows: per head, rows [d] → [-rows[half..] ‖ rows[..half]]
        let q = w2("q_proj").mul_scalar(scale); // 1/√d folded into the q rows
        let qk: Tensor<B, 2> = Tensor::cat(vec![q, w2("k_proj")], 0); // [(h+hkv)d, in]
        let hidden = qk.dims()[1];
        let qk3 = qk.clone().reshape([h + hkv, d, hidden]);
        let qk_rot = Tensor::cat(
            vec![
                qk3.clone().narrow(1, half, half).neg(),
                qk3.narrow(1, 0, half),
            ],
            1,
        )
        .reshape([(h + hkv) * d, hidden]);
        let wide = Tensor::cat(vec![qk, qk_rot, w2("v_proj")], 0);
        let wide_t = wide
            .transpose()
            .mul(fold_in.reshape([hidden, 1]))
            .reshape([1, hidden, n_out]);

        // RoPE applies AFTER the bias (encoder), and rope is linear — the
        // rotated block carries the rotated bias. k_proj never has a bias.
        let wide_bias = qvo_bias.then(|| {
            let b1 = |n: &str| -> Tensor<B, 1> {
                loader.load_tensor(&format!("{prefix}.{n}.bias"), device)
            };
            let bq = b1("q_proj").mul_scalar(scale);
            let bqk: Tensor<B, 1> = Tensor::cat(vec![bq, Tensor::zeros([hkv * d], device)], 0);
            let b2 = bqk.clone().reshape([h + hkv, d]);
            let b_rot = Tensor::cat(
                vec![
                    b2.clone().narrow(1, half, half).neg(),
                    b2.narrow(1, 0, half),
                ],
                1,
            )
            .reshape([(h + hkv) * d]);
            Tensor::cat(vec![bqk, b_rot, b1("v_proj")], 0).reshape([1, 1, n_out])
        });

        Self {
            wide_t,
            wide_bias,
            o_proj: Linear::load(loader, &format!("{prefix}.o_proj"), qvo_bias, device),
            heads: h,
            kv_heads: hkv,
            head_dim: d,
        }
    }

    /// `x`: pre-normed (weightless) `[B, L, hidden]`; `cos`/`sin` are the
    /// stack's position slices; `mask` is the stack's per-forward mask
    /// (`None` exactly when `l == 1`).
    fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        mask: Option<&Tensor<B, 2, Bool>>,
        cache: &mut FastKv<B>,
    ) -> Tensor<B, 3> {
        let [b, l, _] = x.dims();
        let (h, hkv, d) = (self.heads, self.kv_heads, self.head_dim);
        let hh = h + hkv;

        let mut qkv = x.matmul(self.wide_t.clone()); // [B,L,(2(h+hkv)+hkv)·d]
        if let Some(bias) = &self.wide_bias {
            qkv = qkv + bias.clone();
        }
        // [B,L,heads·D] → [B,heads,L,D]; for L=1 the reshape alone is exact.
        let heads = |t: Tensor<B, 3>, n: usize| -> Tensor<B, 4> {
            if l == 1 {
                t.reshape([b, n, 1, d])
            } else {
                t.reshape([b, l, n, d]).swap_dims(1, 2)
            }
        };
        let qk = heads(qkv.clone().narrow(2, 0, hh * d), hh);
        let qkr = heads(qkv.clone().narrow(2, hh * d, hh * d), hh);
        let v = heads(qkv.narrow(2, 2 * hh * d, hkv * d), hkv);

        let roped = qk.mul(cos.clone()) + qkr.mul(sin.clone());
        let q = roped.clone().narrow(1, 0, h);
        let k = roped.narrow(1, h, hkv);

        let (k, v) = cache.update(k, v);
        let lk = k.dims()[2];
        let groups = h / hkv;

        if l == 1 {
            // Trimmed cache guarantees lk ≤ window: single query, no mask.
            // GQA folds groups onto the query axis; MHA (groups=1) passes through.
            debug_assert!(mask.is_none());
            let q = q.reshape([b, hkv, groups, d]);
            let scores = q.matmul(k.swap_dims(2, 3)); // 1/√d pre-folded
            let probs = softmax(scores, 3);
            let out = probs.matmul(v).reshape([b, 1, h * d]);
            return output_projection(&self.o_proj, out);
        }

        let expand = |t: Tensor<B, 4>| {
            if groups == 1 {
                t
            } else {
                t.reshape([b, hkv, 1, lk, d])
                    .expand([b, hkv, groups, lk, d])
                    .reshape([b, h, lk, d])
            }
        };
        let k = expand(k);
        let v = expand(v);

        let scores = q.matmul(k.swap_dims(2, 3)); // [B,H,L,Lk], 1/√d pre-folded
        let scores = match mask {
            Some(m) => scores.mask_fill(
                m.clone().reshape([1, 1, l, lk]).expand([b, h, l, lk]),
                f32::MIN,
            ),
            None => scores,
        };
        let probs = softmax(scores, 3);
        let out = probs.matmul(v).swap_dims(1, 2).reshape([b, l, h * d]);
        output_projection(&self.o_proj, out)
    }
}

/// SwiGLU MLP with fused gate‖up; `down` optionally biased (encoder).
struct FastMlp<B: Backend> {
    gate_up_t: Tensor<B, 3>, // [1, hidden, 2·inter]
    down: Linear<B>,
    inter: usize,
}

impl<B: Backend> FastMlp<B> {
    fn load(
        loader: &WeightLoader,
        prefix: &str,
        down_bias: bool,
        fold_in: Option<Tensor<B, 1>>,
        device: &B::Device,
    ) -> Self {
        let gate: Tensor<B, 2> = loader.load_tensor(&format!("{prefix}.gate_proj.weight"), device);
        let up: Tensor<B, 2> = loader.load_tensor(&format!("{prefix}.up_proj.weight"), device);
        let gu = Tensor::cat(vec![gate, up], 0); // [2I, hidden]
        let [o2, hidden] = gu.dims();
        let gut = gu.transpose();
        let gut = match fold_in {
            Some(s) => gut.mul(s.reshape([hidden, 1])),
            None => gut,
        };
        Self {
            gate_up_t: gut.reshape([1, hidden, o2]),
            down: Linear::load(loader, &format!("{prefix}.down_proj"), down_bias, device),
            inter: o2 / 2,
        }
    }

    fn forward(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let gu = h.matmul(self.gate_up_t.clone());
        output_projection(
            &self.down,
            silu(gu.clone().narrow(2, 0, self.inter)).mul(gu.narrow(2, self.inter, self.inter)),
        )
    }
}

// ── encoder ────────────────────────────────────────────────────────────────

pub struct FastEncoder<B: Backend> {
    conv1: CausalConv<B>,
    conv2: CausalConv<B>,
    layers: Vec<(FastAttention<B>, FastMlp<B>)>,
    rope: RopeTable<B>,
    /// Projector `linear_1` with the encoder's final-norm weight (tiled ×4
    /// across the frame-stack) folded into the rows.
    proj1_t: Tensor<B, 3>,
    proj2: Linear<B>,
}

impl<B: Backend> FastEncoder<B> {
    pub fn load(loader: &WeightLoader, max_positions: usize, device: &B::Device) -> Self {
        let geo = (ENC_HEADS, ENC_HEADS, ENC_HEAD_DIM);
        let w1 = |n: &str| -> Tensor<B, 1> { loader.load_tensor(n, device) };
        let layers = (0..ENC_LAYERS)
            .map(|i| {
                let p = format!("audio_tower.layers.{i}");
                (
                    FastAttention::load(
                        loader,
                        &format!("{p}.self_attn"),
                        geo,
                        true,
                        w1(&format!("{p}.self_attn_layer_norm.weight")),
                        device,
                    ),
                    FastMlp::load(
                        loader,
                        &format!("{p}.mlp"),
                        true,
                        Some(w1(&format!("{p}.final_layer_norm.weight"))),
                        device,
                    ),
                )
            })
            .collect();
        // final norm weight, tiled ×4 to match the projector's stacked input
        let norm = w1("audio_tower.norm.weight");
        let tiled: Tensor<B, 1> = Tensor::cat(vec![norm; DOWNSAMPLE], 0);
        Self {
            conv1: CausalConv::load(loader, "audio_tower.embedder.conv1", 1, device),
            conv2: CausalConv::load(loader, "audio_tower.embedder.conv2", 2, device),
            layers,
            rope: RopeTable::new(ROPE_THETA, ENC_HEAD_DIM, max_positions, device),
            proj1_t: linear_t(
                loader,
                "multi_modal_projector.linear_1",
                Some(tiled),
                device,
            ),
            proj2: Linear::load(loader, "multi_modal_projector.linear_2", false, device),
        }
    }

    pub fn new_caches(&self) -> FastCaches<B> {
        FastCaches((0..ENC_LAYERS).map(|_| FastKv::new(ENC_WINDOW)).collect())
    }

    /// mel `[1, 128, T_mel]` → conv-stem embeds `[1, T_mel/2, 1280]` (op-for-op
    /// the raw stem — the convs aren't where the frame budget goes).
    pub fn stem(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = gelu(self.conv1.forward(mel));
        let x = gelu(self.conv2.forward(x));
        x.swap_dims(1, 2)
    }

    /// Encoder transformer over the next `l` stem positions. Returns the
    /// final-RMS'd (weightless — weight lives in the projector rows) hidden.
    pub fn forward(&self, embeds: Tensor<B, 3>, caches: &mut FastCaches<B>) -> Tensor<B, 3> {
        let l = embeds.dims()[1];
        let pos = caches.0[0].pos;
        let (cos, sin) = self.rope.slices(pos, l);
        let mask = build_mask::<B>(l, caches.0[0].next_lk(l), pos, ENC_WINDOW, &embeds.device());
        let mut x = embeds;
        for ((attn, mlp), cache) in self.layers.iter().zip(caches.0.iter_mut()) {
            let att = attn.forward(rms(x.clone(), EPS), &cos, &sin, mask.as_ref(), cache);
            let x1 = x + att;
            let m = mlp.forward(rms(x1.clone(), EPS));
            x = x1 + m;
        }
        rms(x, EPS)
    }

    /// Weightless-normed hidden `[1, l, 1280]` (l multiple of 4) → audio
    /// embeds `[1, l/4, 3072]`.
    pub fn project(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, _] = hidden.dims();
        assert!(
            l % DOWNSAMPLE == 0,
            "project needs a multiple of {DOWNSAMPLE} positions"
        );
        let stacked = hidden.reshape([b, l / DOWNSAMPLE, ENC_HIDDEN * DOWNSAMPLE]);
        self.proj2
            .forward(gelu(stacked.matmul(self.proj1_t.clone())))
    }
}

// ── decoder ────────────────────────────────────────────────────────────────

struct FastDecLayer<B: Backend> {
    attn: FastAttention<B>,
    mlp: FastMlp<B>, // gate‖up UNfolded — post-norm weight lives in the ada scales
    ada1_t: Tensor<B, 3>, // [1, 3072, 32]
    ada2_t: Tensor<B, 3>, // [1, 32, 3072]
    post_w: Tensor<B, 1>, // post_attention_layernorm weight [3072]
}

pub struct FastDecoder<B: Backend> {
    pub embed: Embedding<B>,
    layers: Vec<FastDecLayer<B>>,
    /// Tied lm_head `[1, 3072, VOCAB]` with the final-norm weight folded in.
    head_t: Tensor<B, 3>,
    rope: RopeTable<B>,
}

impl<B: Backend> FastDecoder<B> {
    pub fn load(loader: &WeightLoader, max_positions: usize, device: &B::Device) -> Self {
        let geo = (DEC_HEADS, DEC_KV_HEADS, DEC_HEAD_DIM);
        let layers = (0..DEC_LAYERS)
            .map(|i| {
                let p = format!("language_model.model.layers.{i}");
                let w1 = |n: &str| -> Tensor<B, 1> {
                    loader.load_tensor(&format!("{p}.{n}.weight"), device)
                };
                FastDecLayer {
                    attn: FastAttention::load(
                        loader,
                        &format!("{p}.self_attn"),
                        geo,
                        false,
                        w1("input_layernorm"),
                        device,
                    ),
                    mlp: FastMlp::load(loader, &format!("{p}.mlp"), false, None, device),
                    ada1_t: linear_t(loader, &format!("{p}.ada_rms_norm.linear1"), None, device),
                    ada2_t: linear_t(loader, &format!("{p}.ada_rms_norm.linear2"), None, device),
                    post_w: w1("post_attention_layernorm"),
                }
            })
            .collect();
        let embed = Embedding::load(loader, "language_model.model.embed_tokens.weight", device);
        let [v, d] = embed.weight.dims();
        let norm: Tensor<B, 1> = loader.load_tensor("language_model.model.norm.weight", device);
        let head_t = embed
            .weight
            .clone()
            .transpose()
            .mul(norm.reshape([d, 1]))
            .reshape([1, d, v]);
        Self {
            embed,
            layers,
            head_t,
            rope: RopeTable::new(ROPE_THETA, DEC_HEAD_DIM, max_positions, device),
        }
    }

    pub fn new_caches(&self) -> FastCaches<B> {
        FastCaches((0..DEC_LAYERS).map(|_| FastKv::new(DEC_WINDOW)).collect())
    }

    /// The 26 per-session conditioning scales, with the post-attention norm
    /// weight PRE-multiplied: `w_post ⊙ (1 + ada(t_cond))`.
    pub fn ada_scales(&self, num_delay_tokens: usize, device: &B::Device) -> AdaScales<B> {
        let t = time_embedding(num_delay_tokens);
        let t = Tensor::<B, 1>::from_floats(t.as_slice(), device).reshape([1, 1, DEC_HIDDEN]);
        AdaScales(
            self.layers
                .iter()
                .map(|l| {
                    gelu(t.clone().matmul(l.ada1_t.clone()))
                        .matmul(l.ada2_t.clone())
                        .add_scalar(1.0)
                        .mul(l.post_w.clone().reshape([1, 1, DEC_HIDDEN]))
                })
                .collect(),
        )
    }

    /// One decoder pass (prefill or single step), appending to the caches.
    /// Returns the UNnormed residual stream — the final norm lives in
    /// [`Self::logits_last`]'s folded head.
    pub fn forward(
        &self,
        embeds: Tensor<B, 3>,
        ada: &AdaScales<B>,
        caches: &mut FastCaches<B>,
    ) -> Tensor<B, 3> {
        let l = embeds.dims()[1];
        let pos = caches.0[0].pos;
        let (cos, sin) = self.rope.slices(pos, l);
        let mask = build_mask::<B>(l, caches.0[0].next_lk(l), pos, DEC_WINDOW, &embeds.device());
        let mut x = embeds;
        for (i, (layer, cache)) in self.layers.iter().zip(caches.0.iter_mut()).enumerate() {
            let att = layer
                .attn
                .forward(rms(x.clone(), EPS), &cos, &sin, mask.as_ref(), cache);
            let x1 = x + att;
            let h = rms(x1.clone(), EPS).mul(ada.0[i].clone());
            x = x1 + layer.mlp.forward(h);
        }
        x
    }

    /// Logits for the LAST position: narrow → weightless rms → folded head.
    pub fn logits_last(&self, hidden: Tensor<B, 3>) -> Tensor<B, 1> {
        let [_, l, _] = hidden.dims();
        rms(hidden.narrow(1, l - 1, 1), EPS)
            .matmul(self.head_t.clone())
            .reshape([VOCAB])
    }
}

// ── the stage bundle ───────────────────────────────────────────────────────

pub struct RealtimeTranscriber<B: Backend> {
    pub mel: VoxtralMel<B>,
    pub encoder: FastEncoder<B>,
    pub decoder: FastDecoder<B>,
    pub tekken: Tekken,
    device: B::Device,
}

impl<B: Backend> RealtimeTranscriber<B> {
    pub fn load(
        loader: &WeightLoader,
        tekken: Tekken,
        max_tokens: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            mel: VoxtralMel::new(device),
            encoder: FastEncoder::load(loader, max_tokens * DOWNSAMPLE, device),
            decoder: FastDecoder::load(loader, max_tokens, device),
            tekken,
            device: device.clone(),
        }
    }
}

impl<B: Backend> SttPipeline<B> for RealtimeTranscriber<B> {
    type EncCaches = FastCaches<B>;
    type DecCaches = FastCaches<B>;
    fn device(&self) -> &B::Device {
        &self.device
    }
    fn tekken(&self) -> &Tekken {
        &self.tekken
    }
    fn mel(&self, samples: &[f32], center: bool) -> Tensor<B, 3> {
        self.mel.forward(samples, center, &self.device)
    }
    fn stem(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.stem(mel)
    }
    fn new_enc_caches(&self) -> Self::EncCaches {
        self.encoder.new_caches()
    }
    fn new_dec_caches(&self) -> Self::DecCaches {
        self.decoder.new_caches()
    }
    fn encode(&self, embeds: Tensor<B, 3>, caches: &mut Self::EncCaches) -> Tensor<B, 3> {
        self.encoder.forward(embeds, caches)
    }
    fn project(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.project(hidden)
    }
    fn ada_scales(&self, n_delay: usize) -> AdaScales<B> {
        self.decoder.ada_scales(n_delay, &self.device)
    }
    fn embed(&self, ids: &[u32]) -> Tensor<B, 3> {
        self.decoder.embed.forward(ids, &self.device)
    }
    fn decode_step(
        &self,
        embeds: Tensor<B, 3>,
        ada: &AdaScales<B>,
        caches: &mut Self::DecCaches,
    ) -> Tensor<B, 3> {
        self.decoder.forward(embeds, ada, caches)
    }
    fn logits_last(&self, hidden: Tensor<B, 3>) -> Tensor<B, 1> {
        self.decoder.logits_last(hidden)
    }
}

#[cfg(all(test, feature = "voxtral-cuda"))]
mod gemv_loaded_tests {
    use super::*;
    use crate::nn::backend::hear::{Device, RawHalf};
    use std::{fs::OpenOptions, path::PathBuf, time::Instant};

    #[test]
    #[ignore = "one real native decoder load; separately admitted model reservation and explicit pile/output required"]
    fn loaded_odown_eligibility_once() -> anyhow::Result<()> {
        let pile = PathBuf::from(std::env::var("MARY_HEARING_GEMV_PILE")?);
        let output = PathBuf::from(std::env::var("MARY_HEARING_GEMV_OBSERVATION")?);
        let file = OpenOptions::new().write(true).create_new(true).open(output)?;
        let started = Instant::now();
        // Existing native cohort selector/loader, kept alive through decoder
        // and all observation tensors. No source checkpoint or host-model copy.
        let snapshot = crate::model_collection::load_model_collection_local_latest(&pile)?;
        let loader = super::super::VoxtralWeights::from_snapshot(snapshot)?.into_loader();
        let device = Device::default();
        let decoder = FastDecoder::<RawHalf>::load(&loader, crate::hear::MAX_TOKENS, &device);
        RawHalf::sync(&device).map_err(|error| anyhow::anyhow!("decoder load sync: {error:?}"))?;
        let loaded_seconds = started.elapsed().as_secs_f64();
        // Genuine owned tensors, not fabricated input bindings. They establish
        // M1 eligibility only; no claim about each live activation is inferred.
        let o_input = Tensor::<RawHalf,3>::zeros([1,1,4096], &device);
        let down_input = Tensor::<RawHalf,3>::zeros([1,1,9216], &device);
        RawHalf::sync(&device).map_err(|error| anyhow::anyhow!("owned fixture sync: {error:?}"))?;
        let mut records = Vec::new();
        for (layer, item) in decoder.layers.iter().enumerate() {
            for (role, x, weight) in [
                ("o", &o_input, &item.attn.o_proj.weight_t),
                ("down", &down_input, &item.mlp.down.weight_t),
            ] {
                let mut record = super::super::gemv_cuda::observe_loaded_weight(x,weight);
                record["layer"] = serde_json::json!(layer);
                record["role"] = serde_json::json!(role);
                println!("ODOWN_ELIGIBILITY {record}");
                records.push(record);
            }
        }
        assert_eq!(records.len(),52);
        let accepted = records.iter().filter(|r| r["accepted"] == true).count();
        let mut rejected_by_reason = serde_json::Map::new();
        for record in &records {
            for reason in record["reasons"].as_array().expect("reason list") {
                let key = reason.as_str().expect("reason string");
                let count = rejected_by_reason.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
                rejected_by_reason.insert(key.to_owned(),serde_json::json!(count+1));
            }
        }
        let report = serde_json::json!({"kind":"loaded_odown_eligibility", "pile":pile,
            "records":records,"accepted":accepted,"rejected":52-accepted,
            "rejected_by_reason":rejected_by_reason,"load_seconds":loaded_seconds,
            "scope":"one native decoder load; real owned M1 fixture metadata; no GEMV, VAD, warmup or transcription"});
        serde_json::to_writer_pretty(file,&report)?;
        println!("ODOWN_ELIGIBILITY_SUMMARY accepted={accepted} rejected={}",52-accepted);
        // Drop device readers before the native pile owner. The externally
        // supplied pile must remain immutable through process/runtime teardown.
        drop((o_input,down_input,decoder));
        drop(loader);
        Ok(())
    }
}
