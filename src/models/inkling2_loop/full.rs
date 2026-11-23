//! One visit of an Inkling 2 Mini *full-attention* ("global") decoder layer
//! outside the looped core, on the device: physical layer 5.
//!
//! From transformers 046bdb8c `Inkling2LoopFullDecoderLayer.forward` and
//! `Inkling2LoopFullAttention`. What differs from a sliding layer
//! ([`super::sliding`]):
//!
//! ```text
//! qh, kh, vh = attention_input(blocks, x)     three learned queries, one per stream
//! q = log_scale(l2_clip(wq(qn(qh)) * s))     qn, kn, vn: separate RMS norms
//! k = l2_clip(dconv(wk(kn(kh)), kn(kh)))     dconv: taps static_w[lag] + weight_proj(kn(kh))[head, lag],
//! v = unit_rms(dconv(wv(vn(vh)), vn(vh)))           (x + sum_lag taps * x[-lag]) / sqrt(2k + 1)
//! bias: rel states log-scaled too; rel_extent from the config, no window
//! a = wo(sdpa(q, k, v, bias, sink) * sigmoid(gate(qn(qh))))
//! ```
//!
//! then the same attention conv, residual, block MLP input, router, experts
//! and MLP conv as a sliding layer. `log_scale` is `1 + alpha * ln(max((p + 1)
//! / n_floor, 1))` at the query's absolute position `p`.
//!
//! What is Mary's: the BF16 projections ([`linear_bf16`]), the NVFP4 expert
//! GEMMs over [`PackedW`] (bound by the sliding layer's binder), the static
//! short conv ([`super::sliding::conv_step`]), [`rms_norm`], the router's
//! selection rule ([`route_from_logits`]) and the sliding layer's L2 clip, unit
//! RMS and one-query block attention. New here and in Burn: the dynamic K/V
//! convolution and the log scale.
//!
//! State is [`State`], as for a sliding layer: post-clip K and post-norm V of
//! every position (no eviction: a full layer has no window), in the fixed
//! segments of [`super::attend::Kv`], scored a tile at a time by
//! [`super::attend::attend`]; and the last
//! `k - 1` inputs of the four convolutions; for K and V those are the
//! projected states, before the dynamic conv, as upstream caches them.
//!
//! A full core layer (11, 17) on a pass after the first is upstream's
//! `Inkling2LoopFullLoopDecoderLayer`: each of the three attention-residual
//! outputs gets its own message gate *before* its own norm (a sliding core
//! layer gates its one input three ways), and the MLP gate sits between
//! `mlp_input` and `post_attention_layernorm`:
//!
//! ```text
//! qh, kh, vh = attention_input(blocks, x)
//! qh, kh, vh = gate_q(qh, m_q), gate_k(kh, m_k), gate_v(vh, m_v)
//! a = attn(qn(qh), kn(kh), vn(vh)) ...       as above
//! m = gate_mlp(mlp_input(blocks, h), m_mlp)
//! ```
//!
//! With no messages it is exactly the ungated visit. The gates run on the device
//! ([`DeviceGate`], BF16 down projection), as in a sliding core layer.
//!
//! Experts bind in their stored format, including the MXFP8 experts in layer 23.

use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::TensorData;

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::burn::{linear_bf16, rms_norm};
use crate::models::inkling::seam::{Bk, client_of};
use crate::models::inkling2_loop::attend::{AttnShape, QUERY_TILE, attend};
use crate::models::inkling2_loop::config::{AttnKind, Inkling2LoopTextConfig};
use crate::models::inkling2_loop::layout::{AttnPart, ConvPart, Dtype, LayerPart, Slot};
use crate::models::inkling2_loop::sliding::{
    Dev, Dims, ExpertAct, STREAMS, State, Taps, Weights, bf16_to_f32, bind_bank, bind_gates,
    block_attention, conv_step, f32_le, l2_clip, unit_rms, want,
};
use crate::models::inkling2_loop::messages::{DeviceGate, message_tensor};
use crate::models::inkling2_loop::moe::Bank;

/// A full-attention layer's weights, bound on the device.
pub struct FullLayer {
    pub layer: usize,
    pub dims: Dims,
    log_alpha: f32,
    log_floor: f32,
    /// `[3, hidden]`: the Q, K and V streams' block-attention queries.
    attention_input: Option<[Tensor<Bk, 1>; 3]>,
    mlp_input: Tensor<Bk, 1>,
    /// Q, K, V input norms.
    in_norm: [Tensor<Bk, 1>; 3],
    mlp_norm: Tensor<Bk, 1>,
    wq: Bf16W,
    wk: Bf16W,
    wv: Bf16W,
    wr: Bf16W,
    gate_attn: Bf16W,
    wo: Bf16W,
    sink: Bf16W,
    q_scale: Tensor<Bk, 3>,
    rel_scale: Tensor<Bk, 3>,
    rel_proj: Tensor<Bk, 2>,
    /// K and V dynamic convs: `weight_proj` `[kv_heads * k, hidden]` and
    /// `static_w` `[k, kv_dim]`, row `lag` weighting the input `lag` back.
    k_proj: Bf16W,
    k_static: Tensor<Bk, 2>,
    v_proj: Bf16W,
    v_static: Tensor<Bk, 2>,
    attn_sconv: Tensor<Bk, 2>,
    mlp_sconv: Tensor<Bk, 2>,
    /// The router and the routed experts.
    moe: Bank,
    /// Core layers only: the Q, K, V and MLP message gates, in
    /// [`super::messages::message_slot`] order.
    gates: Option<[DeviceGate; 4]>,
}

/// What a full visit returns besides its output.
pub struct FullTaps {
    /// The Q, K and V streams' block-attention outputs (before their gates
    /// and norms).
    pub inputs: [Tensor<Bk, 2>; 3],
    pub rest: Taps,
}

impl FullLayer {
    /// Bind full-attention layer `layer` from `w`. Only NVFP4 expert layers bind.
    pub fn bind(w: &dyn Weights, t: &Inkling2LoopTextConfig, layer: usize, dev: &Dev) -> Result<FullLayer, String> {
        if t.attn_kind(layer) != AttnKind::Global {
            return Err(format!("layer {layer} is a sliding layer, not a full-attention one"));
        }
        let (heads, kv_heads, head_dim) = t.heads(AttnKind::Global);
        let d = Dims {
            heads,
            kv_heads,
            head_dim,
            rel_extent: t.rel_extent,
            // no window: every earlier position is visible
            window: usize::MAX,
            ..Dims::of(t)
        };
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
        let attention_input = if layer == 0 {
            None
        } else {
            let st = want(w, Slot::AttnResQuery(layer), Dtype::F32, &[3, h])?;
            let v = f32_le(st.bytes);
            let row = |i: usize| Tensor::from_data(TensorData::new(v[i * h..(i + 1) * h].to_vec(), [h]), dev);
            Some([row(0), row(1), row(2)])
        };
        let moe = bind_bank(w, &d, layer, &client, lin(l(LayerPart::GateWeight), d.experts, h)?)?;
        Ok(FullLayer {
            layer,
            dims: d,
            log_alpha: t.log_scaling_alpha as f32,
            log_floor: t.log_scaling_n_floor as f32,
            attention_input,
            mlp_input: vec1(Slot::MlpResQuery(layer), Dtype::F32, h)?,
            in_norm: [
                vec1(l(LayerPart::QInputNorm), Dtype::Bf16, h)?,
                vec1(l(LayerPart::KInputNorm), Dtype::Bf16, h)?,
                vec1(l(LayerPart::VInputNorm), Dtype::Bf16, h)?,
            ],
            mlp_norm: vec1(l(LayerPart::MlpNorm), Dtype::Bf16, h)?,
            wq: lin(a(AttnPart::Wq), q, h)?,
            wk: lin(a(AttnPart::Wk), kv, h)?,
            wv: lin(a(AttnPart::Wv), kv, h)?,
            wr: lin(a(AttnPart::Wr), rel, h)?,
            gate_attn: lin(a(AttnPart::GateAttn), q, h)?,
            wo: lin(a(AttnPart::Wo), h, q)?,
            sink: lin(a(AttnPart::SinkProj), d.heads, h)?,
            q_scale: per_head(a(AttnPart::QL2Clip), d.head_dim)?,
            rel_scale: per_head(a(AttnPart::RelL2Clip), d.d_rel)?,
            rel_proj: mat(a(AttnPart::RelLogitsProj), d.d_rel, d.rel_extent)?,
            k_proj: lin(a(AttnPart::KSconv(ConvPart::WeightProj)), d.kv_heads * d.kernel, h)?,
            k_static: mat(a(AttnPart::KSconv(ConvPart::StaticW)), d.kernel, kv)?,
            v_proj: lin(a(AttnPart::VSconv(ConvPart::WeightProj)), d.kv_heads * d.kernel, h)?,
            v_static: mat(a(AttnPart::VSconv(ConvPart::StaticW)), d.kernel, kv)?,
            attn_sconv: conv(l(LayerPart::AttnSconv), h)?,
            mlp_sconv: conv(l(LayerPart::MlpSconv), h)?,
            moe,
            gates: bind_gates(w, t, layer, dev)?,
        })
    }

    /// Whether this layer has message gates (a core layer).
    pub fn reads_messages(&self) -> bool {
        self.gates.is_some()
    }

    pub fn is_block_boundary(&self) -> bool {
        self.layer % self.dims.block == 0
    }

    /// Run `x` (`[t, hidden]`, positions `state.pos ..`) through the layer. A
    /// prefill and a decode step are the same call.
    pub fn forward(
        &self,
        x: Tensor<Bk, 2>,
        blocks: &mut Vec<Tensor<Bk, 2>>,
        state: &mut State,
        act: ExpertAct,
    ) -> (Tensor<Bk, 2>, FullTaps) {
        self.forward_at(x, blocks, state, act, 0)
    }

    /// [`Self::forward`] for a sequence whose first cached row sits at absolute
    /// position `origin`. Only the log scale reads absolute positions (upstream
    /// takes them as `position_ids`, apart from the cache); distances come
    /// from the state.
    pub fn forward_at(
        &self,
        x: Tensor<Bk, 2>,
        blocks: &mut Vec<Tensor<Bk, 2>>,
        state: &mut State,
        act: ExpertAct,
        origin: usize,
    ) -> (Tensor<Bk, 2>, FullTaps) {
        self.forward_visit(x, blocks, state, act, origin, None)
    }

    /// [`Self::forward_at`], with the four messages `[t, 4, H]` (Q, K, V, MLP)
    /// a core layer reads on a pass after the first
    /// ([`super::messages::read_messages`]). `None` is the ungated visit.
    /// Panics if messages reach a layer with no gates.
    pub fn forward_visit(
        &self,
        x: Tensor<Bk, 2>,
        blocks: &mut Vec<Tensor<Bk, 2>>,
        state: &mut State,
        act: ExpertAct,
        origin: usize,
        messages: Option<&Tensor<Bk, 3>>,
    ) -> (Tensor<Bk, 2>, FullTaps) {
        let d = self.dims;
        let [t, _] = x.dims();
        let block_residual = x.clone();
        let inputs = match &self.attention_input {
            Some(qs) => qs.clone().map(|q| block_attention(q, blocks, x.clone())),
            None => [x.clone(), x.clone(), x.clone()],
        };
        let gated = messages.map(|m| {
            let g = self.gates.as_ref().unwrap_or_else(|| panic!("layer {} reads no messages", self.layer));
            assert_eq!(m.dims(), [t, 4, d.hidden], "messages are [t, 4, H]");
            (g, m)
        });
        let mut gate_out: Option<[Tensor<Bk, 2>; 4]> = None;
        let streams = match gated {
            None => inputs.clone(),
            Some((g, m)) => {
                let out = [0, 1, 2].map(|j| g[j].apply(inputs[j].clone(), message_tensor(m, STREAMS[j])));
                let [q, k, v] = out.clone();
                gate_out = Some([q.clone(), k, v, q]);
                out
            }
        };
        let normed = [0, 1, 2].map(|i| rms_norm(streams[i].clone(), self.in_norm[i].clone(), d.eps));
        let attn = self.attention(normed, state, origin);
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
                out[3] = g[3].apply(mlp_input.clone(), message_tensor(m, STREAMS[3]));
                out[3].clone()
            }
            _ => mlp_input.clone(),
        };
        let mn = rms_norm(mlp_in, self.mlp_norm.clone(), d.eps);

        let (acc, route) = self.moe.forward(&d, &mn, act);
        let mlp_conv = conv_step(&mut state.conv[3], acc.clone(), &self.mlp_sconv, d.conv_scale());
        state.pos += t;
        let out = residual + mlp_conv.clone();
        let rest = Taps {
            gates: gate_out,
            attn,
            attn_conv,
            mlp_input,
            route,
            experts: acc,
            mlp_conv,
        };
        (out, FullTaps { inputs, rest })
    }

    /// `[t, 1, 1]`: upstream's `Inkling2LoopLogScale` at absolute positions
    /// `first ..`.
    fn log_scale(&self, first: usize, t: usize, dev: &Dev) -> Tensor<Bk, 3> {
        let v: Vec<f32> = (0..t)
            .map(|i| {
                let p = (first + i + 1) as f32 / self.log_floor;
                1.0 + self.log_alpha * p.max(1.0).ln()
            })
            .collect();
        Tensor::from_data(TensorData::new(v, [t, 1, 1]), dev)
    }

    fn attention(&self, [qn, kn, vn]: [Tensor<Bk, 2>; 3], state: &mut State, origin: usize) -> Tensor<Bk, 2> {
        let d = self.dims;
        let [t, _] = qn.dims();
        let dev = qn.device();
        let (heads, kvh, hd) = (d.heads, d.kv_heads, d.head_dim);
        let tau = self.log_scale(origin + state.pos, t, &dev);

        let q = linear_bf16(qn.clone(), &self.wq).reshape([t, heads, hd]) * self.q_scale.clone();
        let q = l2_clip(q, d.qk_clip) * tau.clone();

        let k = linear_bf16(kn.clone(), &self.wk);
        let k = dyn_conv_step(&mut state.conv[0], k, linear_bf16(kn, &self.k_proj), &self.k_static, kvh, d.kernel);
        let k = l2_clip(k.reshape([t, kvh, hd]), d.qk_clip).reshape([t, kvh * hd]);
        let v = linear_bf16(vn.clone(), &self.wv);
        let v = dyn_conv_step(&mut state.conv[1], v, linear_bf16(vn, &self.v_proj), &self.v_static, kvh, d.kernel);
        let v = unit_rms(v.reshape([t, kvh, hd]), d.eps).reshape([t, kvh * hd]);
        state.kv.append(k, v);

        let rel = linear_bf16(qn.clone(), &self.wr).reshape([t, heads, d.d_rel]) * self.rel_scale.clone();
        let rel = (l2_clip(rel, d.rel_clip) * tau).reshape([t * heads, d.d_rel]);
        let sinks = linear_bf16(qn.clone(), &self.sink);
        let shape = AttnShape { heads, kv_heads: kvh, head_dim: hd, rel_extent: d.rel_extent, window: None };
        let o = attend(&shape, q, rel, &self.rel_proj, sinks, &state.kv, state.pos, QUERY_TILE);
        let gate = sigmoid(linear_bf16(qn, &self.gate_attn));
        linear_bf16(o * gate, &self.wo)
    }
}

/// `Inkling2LoopDynamicShortConvolution` over `x`'s rows after the carried
/// inputs in `hist`, carrying the last `k - 1` inputs forward.
///
/// `dynamic` is `[t, heads * k]` (the projection of the stream's normed
/// input), shared by a head's channels; `stat` is `[k, C]`. Tap `lag` of row
/// `i` weights the input `lag` rows back.
fn dyn_conv_step(
    hist: &mut Tensor<Bk, 2>,
    x: Tensor<Bk, 2>,
    dynamic: Tensor<Bk, 2>,
    stat: &Tensor<Bk, 2>,
    heads: usize,
    k: usize,
) -> Tensor<Bk, 2> {
    let [t, c] = x.dims();
    let hd = c / heads;
    let keep = k - 1;
    let all = Tensor::cat(vec![hist.clone(), x.clone()], 0);
    *hist = all.clone().slice([t..t + keep, 0..c]);
    let dynamic = dynamic.reshape([t, heads, k]);
    let mut acc: Tensor<Bk, 3> = Tensor::zeros([t, heads, hd], &x.device());
    for lag in 0..k {
        let rows = all.clone().slice([keep - lag..keep - lag + t, 0..c]).reshape([t, heads, hd]);
        // static taps per channel, dynamic taps per row and head: each a
        // one-sided broadcast against the window rows
        let ws = stat.clone().slice([lag..lag + 1, 0..c]).reshape([1, heads, hd]);
        let wd = dynamic.clone().slice([0..t, 0..heads, lag..lag + 1]);
        acc = acc + rows.clone() * ws + rows * wd;
    }
    (acc.reshape([t, c]) + x).div_scalar(((2 * k + 1) as f32).sqrt())
}
