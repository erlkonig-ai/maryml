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
//! pass is independent of anything a later pass reads. The messages and gates
//! run on the device ([`DeviceGate`]), over all of a call's tokens at once.
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
//! position still visible, and the last `k - 1` INPUTS of each of the four short
//! convolutions. Upstream's cache keeps `k` columns of conv input; only its
//! last `k - 1` are ever read, and those are what is kept here.
//!
//! Experts bind in their stored format ([`bind_experts`]): NVFP4 (layers
//! 0..19) as packed codes, MXFP8 (layers 20..23) decoded on the host to BF16
//! and multiplied by [`linear_bf16`] with BF16 activations.
//!
//! Attention itself is [`super::attend`]: keys and values in fixed segments,
//! scored a tile at a time. A sliding visit drops a segment once no later query
//! can see any key in it, so its cache holds at most `sliding_window_size - 1`
//! visible rows plus less than two segments of dead or unfilled ones.
//!
//! Scope, stated: no log scaling (sliding layers have none upstream);
//! full-attention layers are not here. The caller keys a [`State`] per visit
//! ([`super::schedule::CacheId`]): a core layer's passes share weights and
//! never share state.

use std::collections::BTreeMap;

use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Int, TensorData};

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::block::route_from_logits;
use crate::models::inkling::burn::{
    PackedW, linear_bf16, linear_fp4, linear_w4a16, rms_norm, short_conv_window,
};
use crate::models::inkling::seam::{Bk, client_of};
use crate::models::inkling2_loop::attend::{AttnShape, KEY_TILE, Kv, QUERY_TILE, attend};
use crate::models::inkling2_loop::config::Inkling2LoopTextConfig;
use crate::models::inkling2_loop::experts::nvfp4_scale2;
use crate::models::inkling2_loop::layout::{
    AttnPart, ConvPart, Dtype, ExpertMat, ExpertPart, GatePart, LayerPart, Slot, Stream,
};
use crate::models::inkling2_loop::messages::{DeviceGate, DownPrecision, MessageGate, device_gates, message_tensor};

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

/// One matrix of a routed expert, in the format it binds as.
pub(super) enum ExpertW {
    /// NVFP4 codes and scales, multiplied by [`linear_fp4`] or [`linear_w4a16`].
    Nvfp4(PackedW),
    /// MXFP8 weights decoded to BF16 on the host (exact: an E4M3 value times
    /// a power of two is a BF16), multiplied by [`linear_bf16`].
    Bf16(Bf16W),
}

impl ExpertW {
    fn linear(&self, x: Tensor<Bk, 2>, act: ExpertAct) -> Tensor<Bk, 2> {
        match (self, act) {
            (ExpertW::Nvfp4(p), ExpertAct::Nvfp4) => linear_fp4(x, p),
            (ExpertW::Nvfp4(p), ExpertAct::Bf16) => linear_w4a16(x, p),
            // `ExpertAct` chooses only for NVFP4. MXFP8 experts always see
            // BF16 activations: transformers 046bdb8c dequantises the weights
            // and quantises no activation; the declared `mxfp8_e4m3`
            // activation quantisation is not implemented here.
            (ExpertW::Bf16(w), _) => linear_bf16(x, w),
        }
    }
}

pub(super) struct Expert {
    pub(super) gate: ExpertW,
    pub(super) up: ExpertW,
    pub(super) down: ExpertW,
}

impl Expert {
    /// Upstream's `Inkling2LoopSoftcappedSwiGLU` expert: the sigmoid sees the
    /// raw gate, the softcaps bound the linear gate factor and `up`.
    pub(super) fn forward(&self, d: &Dims, x: Tensor<Bk, 2>, act: ExpertAct) -> Tensor<Bk, 2> {
        let g = self.gate.linear(x.clone(), act);
        let u = self.up.linear(x, act);
        let capped = |v: Tensor<Bk, 2>, c: f32| v.div_scalar(c).tanh().mul_scalar(c);
        let a = capped(g.clone(), d.gate_softcap) * sigmoid(g) * capped(u, d.up_softcap);
        self.down.linear(a, act)
    }
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
    /// [`super::messages::message_slot`] order.
    gates: Option<[DeviceGate; 4]>,
}

pub(super) const STREAMS: [Stream; 4] = [Stream::Query, Stream::Key, Stream::Value, Stream::Mlp];

/// One visit's carried state.
pub struct State {
    /// Positions already seen.
    pub pos: usize,
    /// Keys after the L2 clip and values after the unweighted RMS norm, `[.., kv_dim]`
    /// each, from [`Kv::base`] to `pos`.
    pub kv: Kv,
    /// The last `k - 1` inputs of the K, V, attention-output and MLP-output
    /// convolutions, zero before the first token, as upstream pads.
    pub conv: [Tensor<Bk, 2>; 4],
}

impl State {
    pub fn new(d: &Dims, dev: &Dev) -> State {
        let z = |w: usize| Tensor::zeros([d.kernel - 1, w], dev);
        State {
            pos: 0,
            kv: Kv::new(KEY_TILE),
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
    /// norms, in [`super::messages::message_slot`] order, on the device.
    pub gates: Option<[Tensor<Bk, 2>; 4]>,
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
        let gates = bind_gates(w, t, layer, dev)?;
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
        messages: Option<&Tensor<Bk, 3>>,
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
            assert_eq!(m.dims(), [t, 4, d.hidden], "messages are [t, 4, H]");
            (g, m)
        });
        // upstream casts each normed input back to the layer dtype; here every
        // consumer of the three is a `linear_bf16`, whose entry is that cast
        let norm = |v: Tensor<Bk, 2>| rms_norm(v, self.attn_norm.clone(), d.eps);
        let mut gate_out: Option<[Tensor<Bk, 2>; 4]> = None;
        let qkv = match gated {
            None => {
                let hn = norm(h);
                [hn.clone(), hn.clone(), hn]
            }
            Some((g, m)) => {
                let [q, k, v] = [0, 1, 2].map(|j| g[j].apply(h.clone(), message_tensor(m, STREAMS[j])));
                let qkv = [norm(q.clone()), norm(k.clone()), norm(v.clone())];
                gate_out = Some([q.clone(), k, v, q]);
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
                out[3] = g[3].apply(mlp_input.clone(), message_tensor(m, Stream::Mlp));
                out[3].clone()
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
        self.experts[e].forward(&self.dims, x, act)
    }

    /// Attention over the three normed inputs: `qn` drives the queries, the
    /// relative bias, the sink and the output gate; `kn` and `vn` the keys and
    /// values. Outside a gated visit the three are the same tensor.
    fn attention(&self, [hn, kn, vn]: [Tensor<Bk, 2>; 3], state: &mut State) -> Tensor<Bk, 2> {
        let d = self.dims;
        let [t, _] = hn.dims();
        let (heads, kvh, hd) = (d.heads, d.kv_heads, d.head_dim);

        let q = linear_bf16(hn.clone(), &self.wq).reshape([t, heads, hd]) * self.q_scale.clone();
        let q = l2_clip(q, d.qk_clip);

        let kp = linear_bf16(kn, &self.wk);
        let k = conv_step(&mut state.conv[0], kp, &self.k_sconv, d.conv_scale());
        let k = l2_clip(k.reshape([t, kvh, hd]), d.qk_clip).reshape([t, kvh * hd]);
        let vp = linear_bf16(vn, &self.wv);
        let v = conv_step(&mut state.conv[1], vp, &self.v_sconv, d.conv_scale());
        let v = unit_rms(v.reshape([t, kvh, hd]), d.eps).reshape([t, kvh * hd]);
        state.kv.append(k, v);

        // relative bias: per-token, per-head mix of the trained profiles in `proj`
        let rel = linear_bf16(hn.clone(), &self.wr).reshape([t, heads, d.d_rel]) * self.rel_scale.clone();
        let rel = l2_clip(rel, d.rel_clip).reshape([t * heads, d.d_rel]);
        // the sink: one more logit per query whose value is zero
        let sinks = linear_bf16(hn.clone(), &self.sink);
        let shape = AttnShape { heads, kv_heads: kvh, head_dim: hd, rel_extent: d.rel_extent, window: Some(d.window) };
        let o = attend(&shape, q, rel, &self.rel_proj, sinks, &state.kv, state.pos, QUERY_TILE);
        // the next call's first query is `pos + t`; it sees keys from `pos + t + 1 - window`
        state.kv.evict_before((state.pos + t + 1).saturating_sub(d.window));
        let gate = sigmoid(linear_bf16(hn, &self.gate_attn));
        linear_bf16(o * gate, &self.wo)
    }
}

/// A core layer's four message gates (Q, K, V, MLP, in [`super::messages::message_slot`]
/// order) with the released model's BF16 down projection, bound on the
/// device; `None` outside the core. Shared by the sliding and the full-attention core layers.
pub(super) fn bind_gates(
    w: &dyn Weights,
    t: &Inkling2LoopTextConfig,
    layer: usize,
    dev: &Dev,
) -> Result<Option<[DeviceGate; 4]>, String> {
    if !t.is_core(layer) {
        return Ok(None);
    }
    let (h, r) = (t.hidden_size, t.loop_config.gate_rank);
    let gate = |s: Stream| -> Result<MessageGate, String> {
        let part = |g: GatePart, shape: &[usize]| -> Result<Vec<f32>, String> {
            let slot = Slot::Layer(layer, LayerPart::Controller(s, g));
            Ok(bf16_to_f32(want(w, slot, Dtype::Bf16, shape)?.bytes))
        };
        MessageGate::new(
            part(GatePart::Down, &[r, 2 * h])?,
            part(GatePart::Forget, &[h, r])?,
            part(GatePart::Gain, &[h, r])?,
            h,
            r,
            t.rms_norm_eps as f32,
            DownPrecision::Bf16,
        )
    };
    let host = [gate(STREAMS[0])?, gate(STREAMS[1])?, gate(STREAMS[2])?, gate(STREAMS[3])?];
    Ok(Some(device_gates(&host, dev)))
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

/// Bind a layer's routed experts in the format they are stored in: U8 codes
/// are NVFP4 (layers 0..19 of the release), F8_E4M3 values are MXFP8 (layers
/// 20..23).
pub(super) fn bind_experts(
    w: &dyn Weights,
    d: &Dims,
    layer: usize,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
) -> Result<Vec<Expert>, String> {
    let w13 = Slot::Layer(layer, LayerPart::Expert(ExpertMat::W13, ExpertPart::Weight));
    match w.tensor(&w13.tensor_name())?.dtype {
        Dtype::F8E4M3 => bind_mxfp8_experts(w, d, layer, client),
        _ => bind_nvfp4_experts(w, d, layer, client),
    }
}

/// MXFP8 experts: E4M3 values `[e, rows, cols]` and one UE8M0 scale per 32
/// values of a row `[e, rows, cols / 32]`, decoded on the host one expert at
/// a time (upstream's `Inkling2LoopDequantizeExperts`, then `.to(bfloat16)`,
/// which is exact here) and uploaded as BF16. w13's rows are interleaved gate
/// then up per intermediate unit, its scales with them (upstream's
/// `Interleave(dim=1)` on both). About 2 GB per layer at the release's shape.
fn bind_mxfp8_experts(
    w: &dyn Weights,
    d: &Dims,
    layer: usize,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
) -> Result<Vec<Expert>, String> {
    use crate::models::inkling2_loop::experts::{MXFP8_BLOCK, decode_mxfp8_row};
    let (e, h, i) = (d.experts, d.hidden, d.inter);
    let part = |m: ExpertMat, p: ExpertPart| Slot::Layer(layer, LayerPart::Expert(m, p));
    let w13 = want(w, part(ExpertMat::W13, ExpertPart::Weight), Dtype::F8E4M3, &[e, 2 * i, h])?;
    let s13 = want(w, part(ExpertMat::W13, ExpertPart::Scale), Dtype::U8, &[e, 2 * i, h / MXFP8_BLOCK])?;
    let w2 = want(w, part(ExpertMat::W2, ExpertPart::Weight), Dtype::F8E4M3, &[e, h, i])?;
    let s2 = want(w, part(ExpertMat::W2, ExpertPart::Scale), Dtype::U8, &[e, h, i / MXFP8_BLOCK])?;
    for (n, k) in [(i, h), (h, i)] {
        if !Bf16W::tileable(n, k) || k % MXFP8_BLOCK != 0 {
            return Err(format!("layer {layer}: MXFP8 expert {n}x{k} does not tile"));
        }
    }
    // rows `first, first + step, ..` (count of them) of a `[rows, cols]` matrix, as BF16 bytes
    let decode = |vals: &[u8], scales: &[u8], cols: usize, first: usize, step: usize, count: usize| -> Result<Vec<u8>, String> {
        let per = cols / MXFP8_BLOCK;
        let mut row = vec![0f32; cols];
        let mut out = Vec::with_capacity(count * cols * 2);
        for r in (0..count).map(|j| first + j * step) {
            decode_mxfp8_row(&vals[r * cols..(r + 1) * cols], &scales[r * per..(r + 1) * per], MXFP8_BLOCK, &mut row);
            for &v in &row {
                let b = half::bf16::from_f32(v);
                if b.to_f32().to_bits() != v.to_bits() && !v.is_nan() {
                    return Err(format!("layer {layer}: MXFP8 value {v} is not a BF16"));
                }
                out.extend_from_slice(&b.to_le_bytes());
            }
        }
        Ok(out)
    };
    let upload = |bytes: Vec<u8>, n: usize, k: usize| ExpertW::Bf16(Bf16W { h: client.create_from_slice(&bytes), n, k, align: 16 });
    let (m13, p13) = (2 * i * h, 2 * i * (h / MXFP8_BLOCK));
    let (m2, p2) = (h * i, h * (i / MXFP8_BLOCK));
    (0..e)
        .map(|x| {
            let (v13, c13) = (&w13.bytes[x * m13..(x + 1) * m13], &s13.bytes[x * p13..(x + 1) * p13]);
            let (v2, c2) = (&w2.bytes[x * m2..(x + 1) * m2], &s2.bytes[x * p2..(x + 1) * p2]);
            Ok(Expert {
                gate: upload(decode(v13, c13, h, 0, 2, i)?, i, h),
                up: upload(decode(v13, c13, h, 1, 2, i)?, i, h),
                down: upload(decode(v2, c2, i, 0, 1, h)?, h, i),
            })
        })
        .collect()
}

fn bind_nvfp4_experts(
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
    let packed = |codes: &[u8], scales: &[u8], n: usize, k: usize, scale2: f32| {
        ExpertW::Nvfp4(PackedW {
            codes: client.create_from_slice(codes),
            scales: client.create_from_slice(scales),
            n,
            k,
            scale2,
            swizzled: false,
        })
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
