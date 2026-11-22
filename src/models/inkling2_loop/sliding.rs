//! One visit of an Inkling 2 Mini *sliding* decoder layer, on the device.
//!
//! From transformers 046bdb8c `Inkling2LoopSlidingDecoderLayer.forward`, for
//! the layers outside the looped core, and the core's first pass (no messages):
//!
//! ```text
//! r = x
//! h = attention_input(blocks, x)            layers > 0 only
//! a = attn(input_layernorm(h))              L2-clipped Q/K, RMS V, rel bias, sink, sigmoid gate
//! a = attn_sconv(a)                         (a + conv(a)) / sqrt(k + 1)
//! if boundary: blocks += [r]; res = h = a   (layer % 6 == 0: the block residual restarts)
//! else:        res = h = r + a
//! m = mlp_input(blocks, h)                  attention over the normed block states
//! y = experts(router(mlp_norm(m)))          softcapped SwiGLU, NVFP4 weights
//! y = mlp_sconv(y)
//! out = res + y
//! ```
//!
//! A sliding core layer (6..17) on a pass after the first is upstream's
//! `Inkling2LoopSlidingLoopDecoderLayer` with `messages` `[t, 4, H]` (Q, K, V,
//! MLP), mixed in by the layer's four message gates ([`MessageGate`], CC's
//! operator, with the released model's BF16 down projection):
//!
//! ```text
//! h  = attention_input(blocks, x)
//! qn = input_layernorm(gate_q(h, m_q))      the Q input also drives the
//! kn = input_layernorm(gate_k(h, m_k))      relative bias, the sink and the
//! vn = input_layernorm(gate_v(h, m_v))      output gate
//! a  = attn(qn, kn, vn) ...                 as above
//! m  = gate_mlp(mlp_input(blocks, h), m_mlp)   before mlp_norm
//! ```
//!
//! With no messages a core layer is exactly the visit above, so the first
//! pass is independent of anything a later pass reads. The gates run on the
//! host: one `[t, H]` download and up to four uploads per gated visit.
//!
//! What is Mary's here: the BF16 projections ([`linear_bf16`]), the NVFP4
//! expert GEMMs ([`linear_fp4`] / [`linear_w4a16`] over [`PackedW`]), the short
//! convolution kernel ([`short_conv_window`], which is upstream's `x + conv(x)`
//! before the `1 / sqrt(k + 1)` Mini adds), [`rms_norm`], and the router's
//! selection rule ([`route_from_logits`] with no shared experts, which is
//! upstream's normalised sigmoid times `route_scale`). The rest is new and in
//! Burn: the clips, the relative bias, the sink, the gate, the block attention
//! residual and the softcaps.
//!
//! State is per visit and real: the post-clip K and post-norm V of every
//! position seen, and the last `k - 1` INPUTS of each of the four short
//! convolutions. Upstream's cache keeps `k` columns of conv input; only its
//! last `k - 1` are ever read, and those are what is kept here.
//!
//! Scope, stated: no eviction beyond the window (positions further back than
//! `sliding_window_size` are masked, not dropped); no log scaling (sliding
//! layers have none upstream); NVFP4 experts only, so layers 20..23 (MXFP8)
//! cannot bind; full-attention layers are not here. The caller keys a
//! [`State`] per visit ([`super::schedule::CacheId`]): a core layer's passes
//! share weights and never share state.

use std::collections::BTreeMap;

use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Bool, Int, TensorData};

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::block::route_from_logits;
use crate::models::inkling::burn::{
    PackedW, linear_bf16, linear_fp4, linear_w4a16, rms_norm, short_conv_window,
};
use crate::models::inkling::seam::{Bk, client_of};
use crate::models::inkling2_loop::config::Inkling2LoopTextConfig;
use crate::models::inkling2_loop::experts::nvfp4_scale2;
use crate::models::inkling2_loop::layout::{
    AttnPart, ConvPart, Dtype, ExpertMat, ExpertPart, GatePart, LayerPart, Slot, Stream,
};
use crate::models::inkling2_loop::messages::{DownPrecision, MessageGate, message_slot};

pub(super) type Dev = burn::backend::cuda::CudaDevice;

/// A stored tensor, as whoever holds the weights hands it over.
pub struct Stored<'a> {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
}

/// Where a layer's weights come from: a checkpoint name in, its bytes out.
///
/// The layer does no storage I/O of its own. A pile-backed model root and a
/// safetensors file used as a test input are both just this.
pub trait Weights {
    fn tensor(&self, name: &str) -> Result<Stored<'_>, String>;
}

pub(super) fn want<'a>(w: &'a dyn Weights, slot: Slot, dtype: Dtype, shape: &[usize]) -> Result<Stored<'a>, String> {
    let name = slot.tensor_name();
    let t = w.tensor(&name)?;
    if t.dtype != dtype || t.shape != shape {
        return Err(format!(
            "{name}: stored {:?} {:?}, this layer reads {:?} {:?}",
            t.dtype, t.shape, dtype, shape
        ));
    }
    Ok(t)
}

pub(super) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

pub(super) fn f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Which activation the routed experts multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertAct {
    /// NVFP4 activations, as `hf_quant_config.json` declares, but quantised by
    /// Mary's `amax / 6` rule, not the declared `fouroversix_mae`.
    Nvfp4,
    /// BF16 activations against the same NVFP4 weights: what transformers
    /// 046bdb8c computes (it dequantises the weights and quantises nothing).
    Bf16,
}

pub(super) struct Expert {
    pub(super) gate: PackedW,
    pub(super) up: PackedW,
    pub(super) down: PackedW,
}

/// The fixed sizes of a sliding layer.
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub window: usize,
    pub kernel: usize,
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub eps: f64,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub gate_softcap: f32,
    pub up_softcap: f32,
    pub route_scale: f32,
    pub block: usize,
}

impl Dims {
    pub fn of(t: &Inkling2LoopTextConfig) -> Dims {
        Dims {
            hidden: t.hidden_size,
            heads: t.swa_num_attention_heads,
            kv_heads: t.swa_num_key_value_heads,
            head_dim: t.swa_head_dim,
            d_rel: t.d_rel,
            // upstream: `rel_extent = sliding_window_size` on sliding layers
            rel_extent: t.sliding_window_size,
            window: t.sliding_window_size,
            kernel: t.sconv_kernel_size,
            experts: t.n_routed_experts,
            top_k: t.num_experts_per_tok,
            inter: t.intermediate_size,
            eps: t.rms_norm_eps,
            qk_clip: t.qk_l2_clip_threshold as f32,
            rel_clip: t.rel_l2_clip_threshold as f32,
            gate_softcap: t.mlp_gate_softcap as f32,
            up_softcap: t.mlp_up_softcap as f32,
            route_scale: t.route_scale as f32,
            block: t.n_layers_per_attn_res_block,
        }
    }

    pub(super) fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub(super) fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// `1 / sqrt(k + 1)`, Mini's output scale on every static short conv.
    pub(super) fn conv_scale(&self) -> f32 {
        1.0 / ((self.kernel + 1) as f32).sqrt()
    }
}

/// A sliding layer's weights, bound on the device.
pub struct SlidingLayer {
    pub layer: usize,
    pub dims: Dims,
    attention_input: Option<Tensor<Bk, 1>>,
    mlp_input: Tensor<Bk, 1>,
    attn_norm: Tensor<Bk, 1>,
    mlp_norm: Tensor<Bk, 1>,
    wq: Bf16W,
    wk: Bf16W,
    wv: Bf16W,
    wr: Bf16W,
    gate_attn: Bf16W,
    wo: Bf16W,
    sink: Bf16W,
    router: Bf16W,
    router_bias: Vec<f32>,
    q_scale: Tensor<Bk, 3>,
    rel_scale: Tensor<Bk, 3>,
    rel_proj: Tensor<Bk, 2>,
    k_sconv: Tensor<Bk, 2>,
    v_sconv: Tensor<Bk, 2>,
    attn_sconv: Tensor<Bk, 2>,
    mlp_sconv: Tensor<Bk, 2>,
    experts: Vec<Expert>,
    /// Core layers only: the Q, K, V and MLP message gates, in
    /// [`message_slot`] order.
    gates: Option<[MessageGate; 4]>,
}

const STREAMS: [Stream; 4] = [Stream::Query, Stream::Key, Stream::Value, Stream::Mlp];

/// One visit's carried state.
pub struct State {
    /// Positions already seen.
    pub pos: usize,
    /// `[pos, kv_dim]`, after the L2 clip.
    pub k: Option<Tensor<Bk, 2>>,
    /// `[pos, kv_dim]`, after the unweighted RMS norm.
    pub v: Option<Tensor<Bk, 2>>,
    /// The last `k - 1` inputs of the K, V, attention-output and MLP-output
    /// convolutions, zero before the first token, as upstream pads.
    pub conv: [Tensor<Bk, 2>; 4],
}

impl State {
    pub fn new(d: &Dims, dev: &Dev) -> State {
        let z = |w: usize| Tensor::zeros([d.kernel - 1, w], dev);
        State {
            pos: 0,
            k: None,
            v: None,
            conv: [z(d.kv_dim()), z(d.kv_dim()), z(d.hidden), z(d.hidden)],
        }
    }
}

/// What one visit returns besides its output, for a caller that wants to look.
pub struct Taps {
    pub attn: Tensor<Bk, 2>,
    pub attn_conv: Tensor<Bk, 2>,
    pub mlp_input: Tensor<Bk, 2>,
    pub router_logits: Vec<f32>,
    pub routing: Vec<(Vec<usize>, Vec<f32>)>,
    pub experts: Tensor<Bk, 2>,
    pub mlp_conv: Tensor<Bk, 2>,
    /// A gated visit's four message-gate outputs `[t, H]`, before their
    /// norms, in [`message_slot`] order.
    pub gates: Option<[Vec<f32>; 4]>,
}

impl SlidingLayer {
    /// Bind layer `layer` from `w`. Only NVFP4 expert layers bind.
    pub fn bind(
        w: &dyn Weights,
        t: &Inkling2LoopTextConfig,
        layer: usize,
        dev: &Dev,
    ) -> Result<SlidingLayer, String> {
        let d = Dims::of(t);
        if (layer + 1) % d.block == 0 {
            return Err(format!("layer {layer} is a full-attention layer, not a sliding one"));
        }
        let l = |p: LayerPart| Slot::Layer(layer, p);
        let a = |p: AttnPart| Slot::Layer(layer, LayerPart::Attn(p));
        let probe = Tensor::<Bk, 1>::zeros([1], dev);
        let client = client_of(&probe);
        let vec1 = |s: Slot, dt: Dtype, n: usize| -> Result<Tensor<Bk, 1>, String> {
            let st = want(w, s, dt, &[n])?;
            let v = if dt == Dtype::F32 { f32_le(st.bytes) } else { bf16_to_f32(st.bytes) };
            Ok(Tensor::from_data(TensorData::new(v, [n]), dev))
        };
        let mat = |s: Slot, rows: usize, cols: usize| -> Result<Tensor<Bk, 2>, String> {
            let st = want(w, s, Dtype::Bf16, &[rows, cols])?;
            Ok(Tensor::from_data(TensorData::new(bf16_to_f32(st.bytes), [rows, cols]), dev))
        };
        // depthwise conv weights are stored [C, 1, k]; Mary's kernel reads [C, k]
        let conv = |s: Slot, c: usize| -> Result<Tensor<Bk, 2>, String> {
            let st = want(w, s, Dtype::Bf16, &[c, 1, d.kernel])?;
            Ok(Tensor::from_data(TensorData::new(bf16_to_f32(st.bytes), [c, d.kernel]), dev))
        };
        let lin = |s: Slot, n: usize, k: usize| -> Result<Bf16W, String> {
            let st = want(w, s, Dtype::Bf16, &[n, k])?;
            if !Bf16W::tileable(n, k) {
                return Err(format!("{}: {n}x{k} does not tile", s.tensor_name()));
            }
            Ok(Bf16W { h: client.create_from_slice(st.bytes), n, k, align: 16 })
        };
        let (h, q, kv, rel) = (d.hidden, d.q_dim(), d.kv_dim(), d.heads * d.d_rel);
        let per_head = |s: Slot, width: usize| -> Result<Tensor<Bk, 3>, String> {
            Ok(mat(s, d.heads, width)?.reshape([1, d.heads, width]))
        };

        let router_bias = f32_le(want(w, l(LayerPart::GateBias), Dtype::F32, &[d.experts])?.bytes);
        let experts = bind_experts(w, &d, layer, &client)?;
        let gates = if t.is_core(layer) {
            let r = t.loop_config.gate_rank;
            let gate = |s: Stream| -> Result<MessageGate, String> {
                let part = |g: GatePart, shape: &[usize]| -> Result<Vec<f32>, String> {
                    Ok(bf16_to_f32(want(w, l(LayerPart::Controller(s, g)), Dtype::Bf16, shape)?.bytes))
                };
                MessageGate::new(
                    part(GatePart::Down, &[r, 2 * h])?,
                    part(GatePart::Forget, &[h, r])?,
                    part(GatePart::Gain, &[h, r])?,
                    h,
                    r,
                    d.eps as f32,
                    DownPrecision::Bf16,
                )
            };
            Some([gate(STREAMS[0])?, gate(STREAMS[1])?, gate(STREAMS[2])?, gate(STREAMS[3])?])
        } else {
            None
        };
        Ok(SlidingLayer {
            layer,
            dims: d,
            attention_input: if layer == 0 {
                None
            } else {
                Some(vec1(Slot::AttnResQuery(layer), Dtype::F32, h)?)
            },
            mlp_input: vec1(Slot::MlpResQuery(layer), Dtype::F32, h)?,
            attn_norm: vec1(l(LayerPart::AttnNorm), Dtype::Bf16, h)?,
            mlp_norm: vec1(l(LayerPart::MlpNorm), Dtype::Bf16, h)?,
            wq: lin(a(AttnPart::Wq), q, h)?,
            wk: lin(a(AttnPart::Wk), kv, h)?,
            wv: lin(a(AttnPart::Wv), kv, h)?,
            wr: lin(a(AttnPart::Wr), rel, h)?,
            gate_attn: lin(a(AttnPart::GateAttn), q, h)?,
            wo: lin(a(AttnPart::Wo), h, q)?,
            sink: lin(a(AttnPart::SinkProj), d.heads, h)?,
            router: lin(l(LayerPart::GateWeight), d.experts, h)?,
            router_bias,
            q_scale: per_head(a(AttnPart::QL2Clip), d.head_dim)?,
            rel_scale: per_head(a(AttnPart::RelL2Clip), d.d_rel)?,
            rel_proj: mat(a(AttnPart::RelLogitsProj), d.d_rel, d.rel_extent)?,
            k_sconv: conv(a(AttnPart::KSconv(ConvPart::Weight)), kv)?,
            v_sconv: conv(a(AttnPart::VSconv(ConvPart::Weight)), kv)?,
            attn_sconv: conv(l(LayerPart::AttnSconv), h)?,
            mlp_sconv: conv(l(LayerPart::MlpSconv), h)?,
            experts,
            gates,
        })
    }

    pub fn is_block_boundary(&self) -> bool {
        self.layer % self.dims.block == 0
    }

    /// Whether this layer has message gates (a core layer).
    pub fn reads_messages(&self) -> bool {
        self.gates.is_some()
    }

    /// Run `x` (`[t, hidden]`, positions `state.pos ..`) through the layer,
    /// updating `state` and, on a block boundary, appending the layer input
    /// to `blocks`. A prefill and a decode step are the same call.
    pub fn forward(
        &self,
        x: Tensor<Bk, 2>,
        blocks: &mut Vec<Tensor<Bk, 2>>,
        state: &mut State,
        act: ExpertAct,
    ) -> (Tensor<Bk, 2>, Taps) {
        self.forward_visit(x, blocks, state, act, None)
    }

    /// [`Self::forward`], with the four messages `[t, 4, H]` (Q, K, V, MLP)
    /// a core layer reads on a pass after the first
    /// ([`super::messages::read_messages`]). `None` is the ungated visit.
    /// Panics if messages reach a layer with no gates.
    pub fn forward_visit(
        &self,
        x: Tensor<Bk, 2>,
        blocks: &mut Vec<Tensor<Bk, 2>>,
        state: &mut State,
        act: ExpertAct,
        messages: Option<&[f32]>,
    ) -> (Tensor<Bk, 2>, Taps) {
        let d = self.dims;
        let [t, _] = x.dims();
        let dev = x.device();
        let block_residual = x.clone();
        let h = match &self.attention_input {
            Some(q) => block_attention(q.clone(), blocks, x.clone()),
            None => x.clone(),
        };
        let gated = messages.map(|m| {
            let g = self.gates.as_ref().unwrap_or_else(|| panic!("layer {} reads no messages", self.layer));
            assert_eq!(m.len(), t * 4 * d.hidden, "messages are [t, 4, H]");
            (g, m)
        });
        let message = |m: &[f32], s: Stream| -> Vec<f32> {
            let j = message_slot(s);
            (0..t).flat_map(|i| m[(i * 4 + j) * d.hidden..(i * 4 + j + 1) * d.hidden].iter().copied()).collect()
        };
        let upload = |v: Vec<f32>| Tensor::<Bk, 2>::from_data(TensorData::new(v, [t, d.hidden]), &dev);
        // upstream casts each normed input back to the layer dtype; here every
        // consumer of the three is a `linear_bf16`, whose entry is that cast
        let norm = |v: Tensor<Bk, 2>| rms_norm(v, self.attn_norm.clone(), d.eps);
        let mut gate_out: Option<[Vec<f32>; 4]> = None;
        let qkv = match gated {
            None => {
                let hn = norm(h);
                [hn.clone(), hn.clone(), hn]
            }
            Some((g, m)) => {
                let hh: Vec<f32> = h.into_data().convert::<f32>().to_vec().expect("f32");
                let [q, k, v] = [0, 1, 2].map(|j| g[j].apply(&hh, &message(m, STREAMS[j]), t));
                let qkv = [norm(upload(q.clone())), norm(upload(k.clone())), norm(upload(v.clone()))];
                gate_out = Some([q, k, v, Vec::new()]);
                qkv
            }
        };
        let attn = self.attention(qkv, state);
        let attn_conv = conv_step(&mut state.conv[2], attn.clone(), &self.attn_sconv, d.conv_scale());

        let residual = if self.is_block_boundary() {
            blocks.push(block_residual);
            attn_conv.clone()
        } else {
            block_residual + attn_conv.clone()
        };
        let mlp_input = block_attention(self.mlp_input.clone(), blocks, residual.clone());
        let mlp_in = match (gated, gate_out.as_mut()) {
            (Some((g, m)), Some(out)) => {
                let hh: Vec<f32> = mlp_input.clone().into_data().convert::<f32>().to_vec().expect("f32");
                out[3] = g[3].apply(&hh, &message(m, Stream::Mlp), t);
                upload(out[3].clone())
            }
            _ => mlp_input.clone(),
        };
        let mn = rms_norm(mlp_in, self.mlp_norm.clone(), d.eps);

        let router_logits: Vec<f32> = linear_bf16(mn.clone(), &self.router)
            .into_data()
            .to_vec()
            .expect("router logits are f32");
        let routes = route_from_logits(
            &router_logits,
            &self.router_bias,
            1.0,
            d.route_scale,
            t,
            d.experts,
            0,
            d.top_k,
        );
        let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
        for (row, r) in routes.iter().enumerate() {
            for (&e, &g) in r.experts.iter().zip(&r.weights) {
                by_expert.entry(e).or_default().push((row, g));
            }
        }
        let mut acc: Tensor<Bk, 2> = Tensor::zeros([t, d.hidden], &dev);
        for (&e, rows) in &by_expert {
            let idx: Vec<i64> = rows.iter().map(|&(r, _)| r as i64).collect();
            let gw: Vec<f32> = rows.iter().map(|&(_, g)| g).collect();
            let n = idx.len();
            let idx = Tensor::<Bk, 1, Int>::from_data(TensorData::new(idx, [n]), &dev);
            let gw = Tensor::<Bk, 2>::from_data(TensorData::new(gw, [n, 1]), &dev);
            let xs = mn.clone().select(0, idx.clone());
            let y = self.expert(e, xs, act) * gw;
            acc = acc.select_assign(0, idx, y, burn::tensor::IndexingUpdateOp::Add);
        }
        let mlp_conv = conv_step(&mut state.conv[3], acc.clone(), &self.mlp_sconv, d.conv_scale());
        state.pos += t;
        let out = residual + mlp_conv.clone();
        let taps = Taps {
            attn,
            attn_conv,
            mlp_input,
            router_logits,
            routing: routes.into_iter().map(|r| (r.experts, r.weights)).collect(),
            experts: acc,
            mlp_conv,
            gates: gate_out,
        };
        (out, taps)
    }

    fn expert(&self, e: usize, x: Tensor<Bk, 2>, act: ExpertAct) -> Tensor<Bk, 2> {
        let d = self.dims;
        let w = &self.experts[e];
        let lin = |x: Tensor<Bk, 2>, p: &PackedW| match act {
            ExpertAct::Nvfp4 => linear_fp4(x, p),
            ExpertAct::Bf16 => linear_w4a16(x, p),
        };
        let g = lin(x.clone(), &w.gate);
        let u = lin(x, &w.up);
        let capped = |v: Tensor<Bk, 2>, c: f32| v.div_scalar(c).tanh().mul_scalar(c);
        let a = capped(g.clone(), d.gate_softcap) * sigmoid(g) * capped(u, d.up_softcap);
        lin(a, &w.down)
    }

    /// Attention over the three normed inputs: `qn` drives the queries, the
    /// relative bias, the sink and the output gate; `kn` and `vn` the keys and
    /// values. Outside a gated visit the three are the same tensor.
    fn attention(&self, [hn, kn, vn]: [Tensor<Bk, 2>; 3], state: &mut State) -> Tensor<Bk, 2> {
        let d = self.dims;
        let [t, _] = hn.dims();
        let dev = hn.device();
        let (heads, kvh, hd) = (d.heads, d.kv_heads, d.head_dim);

        let q = linear_bf16(hn.clone(), &self.wq).reshape([t, heads, hd]) * self.q_scale.clone();
        let q = l2_clip(q, d.qk_clip);

        let kp = linear_bf16(kn, &self.wk);
        let k = conv_step(&mut state.conv[0], kp, &self.k_sconv, d.conv_scale());
        let k = l2_clip(k.reshape([t, kvh, hd]), d.qk_clip).reshape([t, kvh * hd]);
        let vp = linear_bf16(vn, &self.wv);
        let v = conv_step(&mut state.conv[1], vp, &self.v_sconv, d.conv_scale());
        let v = unit_rms(v.reshape([t, kvh, hd]), d.eps).reshape([t, kvh * hd]);
        let cat = |old: Option<Tensor<Bk, 2>>, new: Tensor<Bk, 2>| match old {
            Some(o) => Tensor::cat(vec![o, new], 0),
            None => new,
        };
        let k_all = cat(state.k.take(), k);
        let v_all = cat(state.v.take(), v);
        state.k = Some(k_all.clone());
        state.v = Some(v_all.clone());
        let n = state.pos + t;

        // relative bias: per-token, per-head mix of the trained profiles in `proj`
        let rel = linear_bf16(hn.clone(), &self.wr).reshape([t, heads, d.d_rel]) * self.rel_scale.clone();
        let rel = l2_clip(rel, d.rel_clip).reshape([t * heads, d.d_rel]);
        let rel_logits = rel
            .matmul(self.rel_proj.clone())
            .reshape([t, heads, d.rel_extent])
            .swap_dims(0, 1);
        let mut gather = Vec::with_capacity(heads * t * n);
        let mut masked = Vec::with_capacity(heads * t * n);
        let mut no_bias = Vec::with_capacity(heads * t * n);
        for _ in 0..heads {
            for i in 0..t {
                for j in 0..n {
                    let dist = (state.pos + i) as i64 - j as i64;
                    gather.push(dist.clamp(0, d.rel_extent as i64 - 1));
                    masked.push(dist < 0 || dist >= d.window as i64);
                    no_bias.push(dist < 0 || dist >= d.rel_extent as i64);
                }
            }
        }
        let gather = Tensor::<Bk, 3, Int>::from_data(TensorData::new(gather, [heads, t, n]), &dev);
        let masked = Tensor::<Bk, 3, Bool>::from_data(TensorData::new(masked, [heads, t, n]), &dev);
        let no_bias = Tensor::<Bk, 3, Bool>::from_data(TensorData::new(no_bias, [heads, t, n]), &dev);
        let bias = rel_logits.gather(2, gather).mask_fill(no_bias, 0.0);

        // GQA: heads g * rep .. (g + 1) * rep read kv head g
        let rep = heads / kvh;
        let q = q.swap_dims(0, 1).reshape([kvh, rep * t, hd]);
        let kt = k_all.reshape([n, kvh, hd]).swap_dims(0, 1).swap_dims(1, 2);
        let scores = q.matmul(kt).mul_scalar((hd as f32).powf(-0.5)).reshape([heads, t, n]) + bias;
        let scores = scores.mask_fill(masked, f32::NEG_INFINITY);
        // the sink: one more logit per query whose value is zero
        // reshape while contiguous, then swap: on this backend `[1, heads]`
        // swapped to `[heads, 1]` and then reshaped read zeros for every head
        // but the first (measured at t = 1, 2026-11-22)
        let sinks = linear_bf16(hn.clone(), &self.sink).reshape([t, heads, 1]).swap_dims(0, 1);
        let probs = softmax(Tensor::cat(vec![scores, sinks], 2), 2).slice([0..heads, 0..t, 0..n]);
        let vv = v_all.reshape([n, kvh, hd]).swap_dims(0, 1);
        let o = probs
            .reshape([kvh, rep * t, n])
            .matmul(vv)
            .reshape([heads, t, hd])
            .swap_dims(0, 1)
            .reshape([t, heads * hd]);
        let gate = sigmoid(linear_bf16(hn, &self.gate_attn));
        linear_bf16(o * gate, &self.wo)
    }
}

/// Run one short convolution over `x`'s rows after the carried inputs in
/// `hist`, and carry the last `k - 1` inputs forward.
pub(super) fn conv_step(hist: &mut Tensor<Bk, 2>, x: Tensor<Bk, 2>, w: &Tensor<Bk, 2>, scale: f32) -> Tensor<Bk, 2> {
    let [t, c] = x.dims();
    let [keep, _] = hist.dims();
    let all = Tensor::cat(vec![hist.clone(), x], 0);
    *hist = all.clone().slice([t..t + keep, 0..c]);
    short_conv_window(all, w.clone(), t).mul_scalar(scale)
}

/// Scale each `[.., d]` vector down to `thr` when its L2 norm exceeds it.
pub(super) fn l2_clip(x: Tensor<Bk, 3>, thr: f32) -> Tensor<Bk, 3> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(2).sqrt();
    let f = norm.clone().ones_like().mul_scalar(thr).div(norm.add_scalar(1e-6)).clamp_max(1.0);
    x * f
}

/// RMS-normalise each `[.., d]` vector, with no gain.
pub(super) fn unit_rms(x: Tensor<Bk, 3>, eps: f64) -> Tensor<Bk, 3> {
    let denom = x.clone().powf_scalar(2.0).mean_dim(2).add_scalar(eps).sqrt();
    x / denom
}

/// `Inkling2LoopBlockAttentionResidual` with one query: attention from a
/// learned query over the normed `[blocks.., partial]`, whose values are the
/// same normed states. Upstream builds this norm with its default eps, 1e-6,
/// not the config's.
pub(super) fn block_attention(query: Tensor<Bk, 1>, blocks: &[Tensor<Bk, 2>], partial: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
    let [t, h] = partial.dims();
    let mut all: Vec<Tensor<Bk, 3>> = blocks.iter().map(|b| b.clone().reshape([t, 1, h])).collect();
    all.push(partial.reshape([t, 1, h]));
    let normed = unit_rms(Tensor::cat(all, 1), 1e-6);
    let logits = (normed.clone() * query.reshape([1, 1, h]))
        .sum_dim(2)
        .mul_scalar((h as f32).powf(-0.5));
    let p = softmax(logits, 1);
    (normed * p).sum_dim(1).reshape([t, h])
}

pub(super) fn bind_experts(
    w: &dyn Weights,
    d: &Dims,
    layer: usize,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
) -> Result<Vec<Expert>, String> {
    let (e, h, i) = (d.experts, d.hidden, d.inter);
    let part = |m: ExpertMat, p: ExpertPart| Slot::Layer(layer, LayerPart::Expert(m, p));
    let w13 = want(w, part(ExpertMat::W13, ExpertPart::Weight), Dtype::U8, &[e, 2 * i, h / 2])?;
    let s13 = want(w, part(ExpertMat::W13, ExpertPart::Scale), Dtype::F8E4M3, &[e, 2 * i, h / 16])?;
    let a13 = f32_le(want(w, part(ExpertMat::W13, ExpertPart::Amax), Dtype::F32, &[e])?.bytes);
    let d13 = f32_le(want(w, part(ExpertMat::W13, ExpertPart::OuterScaleDenominator), Dtype::F32, &[1])?.bytes);
    let w2 = want(w, part(ExpertMat::W2, ExpertPart::Weight), Dtype::U8, &[e, h, i / 2])?;
    let s2 = want(w, part(ExpertMat::W2, ExpertPart::Scale), Dtype::F8E4M3, &[e, h, i / 16])?;
    let a2 = f32_le(want(w, part(ExpertMat::W2, ExpertPart::Amax), Dtype::F32, &[e])?.bytes);
    let d2 = f32_le(want(w, part(ExpertMat::W2, ExpertPart::OuterScaleDenominator), Dtype::F32, &[1])?.bytes);
    let sc13 = nvfp4_scale2(&a13, &d13)?;
    let sc2 = nvfp4_scale2(&a2, &d2)?;
    let (cb13, sb13, cb2, sb2) = (h / 2, h / 16, i / 2, i / 16);
    // w13's rows are stored interleaved, gate then up per intermediate unit
    // (upstream de-interleaves them on load with `Interleave(dim=1)`)
    let rows = |src: &[u8], width: usize, e: usize, parity: usize| -> Vec<u8> {
        let base = e * 2 * i * width;
        let mut out = Vec::with_capacity(i * width);
        for r in 0..i {
            let s = base + (2 * r + parity) * width;
            out.extend_from_slice(&src[s..s + width]);
        }
        out
    };
    let packed = |codes: &[u8], scales: &[u8], n: usize, k: usize, scale2: f32| PackedW {
        codes: client.create_from_slice(codes),
        scales: client.create_from_slice(scales),
        n,
        k,
        scale2,
        swizzled: false,
    };
    Ok((0..e)
        .map(|x| Expert {
            gate: packed(&rows(w13.bytes, cb13, x, 0), &rows(s13.bytes, sb13, x, 0), i, h, sc13[x]),
            up: packed(&rows(w13.bytes, cb13, x, 1), &rows(s13.bytes, sb13, x, 1), i, h, sc13[x]),
            down: packed(
                &w2.bytes[x * h * cb2..(x + 1) * h * cb2],
                &s2.bytes[x * h * sb2..(x + 1) * h * sb2],
                h,
                i,
                sc2[x],
            ),
        })
        .collect())
}
