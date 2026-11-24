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
//! and MXFP8 expert GEMMs ([`super::moe`]), the short
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
//! Experts bind in their stored format ([`bind_bank`]): NVFP4 (layers
//! 0..19) as packed codes, MXFP8 (layers 20..23) as packed codes and scales
//! for register conversion to BF16 where the device supports E4M3. Other
//! devices bind decoded BF16. Each projection stacks its experts. The router and
//! the experts run as [`super::moe::Bank`]: routed on the device, one grouped
//! launch per projection.
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

use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::TensorData;

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::burn::{
    linear_bf16, short_conv_window,
};
use crate::models::inkling::seam::{Bk, client_of};
use crate::models::inkling2_loop::prof;
use crate::models::inkling2_loop::attend::{AttnShape, KEY_TILE, Kv, QUERY_TILE, RelProj, attend};
use crate::models::inkling2_loop::config::Inkling2LoopTextConfig;
use crate::models::inkling2_loop::experts::nvfp4_scale2;
use crate::models::inkling2_loop::layout::{
    AttnPart, ConvPart, Dtype, ExpertMat, ExpertPart, GatePart, LayerPart, Slot, Stream,
};
use crate::models::inkling2_loop::messages::{DeviceGate, DownPrecision, MessageGate, device_gates, message_tensor};
use crate::models::inkling2_loop::moe::{Bank, FaultCell, Proj, Route, bf16_proj, fault_cell, nvfp4_proj};

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
    q_scale: Tensor<Bk, 3>,
    rel_scale: Tensor<Bk, 3>,
    rel_proj: RelProj,
    k_sconv: Tensor<Bk, 2>,
    v_sconv: Tensor<Bk, 2>,
    attn_sconv: Tensor<Bk, 2>,
    mlp_sconv: Tensor<Bk, 2>,
    /// The router and the routed experts.
    pub(super) moe: Bank,
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
    /// The router fault cell this visit's experts write ([`super::moe::Bank::forward`]):
    /// one u32 on the device, `1 + logit` once a row's router logits were not
    /// finite. Nothing clears it; whoever owns the state reads it ([`State::fault`]).
    fault: FaultCell,
}

impl State {
    /// A fresh state with a fault cell of its own.
    pub fn new(d: &Dims, dev: &Dev) -> State {
        State { pos: 0, kv: Kv::new(KEY_TILE), conv: Self::zero_conv(d, dev), fault: fault_cell(dev) }
    }

    /// A fresh state writing its router faults into `fault`, a cell shared
    /// with the other visits of one sequence, so the sequence reads one cell.
    pub fn sharing_fault(d: &Dims, dev: &Dev, fault: &FaultCell) -> State {
        assert!(fault.device() == dev, "state: a fault cell on {:?} for a state on {dev:?}", fault.device());
        State { pos: 0, kv: Kv::new(KEY_TILE), conv: Self::zero_conv(d, dev), fault: fault.clone() }
    }

    /// The router fault cell this state's visits write.
    pub fn fault(&self) -> &FaultCell {
        &self.fault
    }

    fn zero_conv(d: &Dims, dev: &Dev) -> [Tensor<Bk, 2>; 4] {
        let z = |w: usize| Tensor::zeros([d.kernel - 1, w], dev);
        [z(d.kv_dim()), z(d.kv_dim()), z(d.hidden), z(d.hidden)]
    }
}

/// What one visit returns besides its output, for a caller that wants to look.
pub struct Taps {
    pub attn: Tensor<Bk, 2>,
    pub attn_conv: Tensor<Bk, 2>,
    pub mlp_input: Tensor<Bk, 2>,
    /// The router's logits and picks, on the device ([`Route::logits`],
    /// [`Route::picks`] read them back).
    pub route: Route,
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

        let moe = bind_bank(w, &d, layer, &client, lin(l(LayerPart::GateWeight), d.experts, h)?)?;
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
            q_scale: per_head(a(AttnPart::QL2Clip), d.head_dim)?,
            rel_scale: per_head(a(AttnPart::RelL2Clip), d.d_rel)?,
            rel_proj: RelProj::new(
                &client,
                want(w, a(AttnPart::RelLogitsProj), Dtype::Bf16, &[d.d_rel, d.rel_extent])?.bytes,
                d.d_rel,
                d.rel_extent,
            ),
            k_sconv: conv(a(AttnPart::KSconv(ConvPart::Weight)), kv)?,
            v_sconv: conv(a(AttnPart::VSconv(ConvPart::Weight)), kv)?,
            attn_sconv: conv(l(LayerPart::AttnSconv), h)?,
            mlp_sconv: conv(l(LayerPart::MlpSconv), h)?,
            moe,
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
    ) -> (Tensor<Bk, 2>, Taps) {
        self.forward_visit(x, blocks, state, None)
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
        messages: Option<&Tensor<Bk, 3>>,
    ) -> (Tensor<Bk, 2>, Taps) {
        let d = self.dims;
        let [t, _] = x.dims();
        let block_residual = x.clone();
        let h = match &self.attention_input {
            Some(q) => prof::span("block_attn", &x.device(), || block_attention(q.clone(), blocks, x.clone())),
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
        let attn = prof::span("attention", &x.device(), || self.attention(qkv, state));
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

        let (acc, route) = self.moe.forward(&mn, state.fault());
        let mlp_conv = conv_step(&mut state.conv[3], acc.clone(), &self.mlp_sconv, d.conv_scale());
        state.pos += t;
        let out = residual + mlp_conv.clone();
        let taps = Taps {
            attn,
            attn_conv,
            mlp_input,
            route,
            experts: acc,
            mlp_conv,
            gates: gate_out,
        };
        (out, taps)
    }

    /// The experts' input of an ungated visit, from its `mlp_input` tap.
    #[cfg(test)]
    pub(super) fn expert_input(&self, mlp_input: &Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        rms_norm(mlp_input.clone(), self.mlp_norm.clone(), self.dims.eps)
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
    if let Some(y) = super::rows::conv_step(hist, x.clone(), w, scale) {
        return y;
    }
    let [t, c] = x.dims();
    let [keep, _] = hist.dims();
    let all = Tensor::cat(vec![hist.clone(), x], 0);
    *hist = all.clone().slice([t..t + keep, 0..c]);
    short_conv_window(all, w.clone(), t).mul_scalar(scale)
}

/// `rms_norm` with a gain over the last axis: the row kernel when it applies,
/// else Burn's composition.
pub(super) fn rms_norm(x: Tensor<Bk, 2>, gain: Tensor<Bk, 1>, eps: f64) -> Tensor<Bk, 2> {
    match super::rows::rms_norm(x.clone(), gain.clone(), eps) {
        Some(y) => y,
        None => crate::models::inkling::burn::rms_norm(x, gain, eps),
    }
}

/// Scale each `[.., d]` vector down to `thr` when its L2 norm exceeds it.
pub(super) fn l2_clip(x: Tensor<Bk, 3>, thr: f32) -> Tensor<Bk, 3> {
    if let Some(y) = super::rows::l2_clip(x.clone(), thr) {
        return y;
    }
    let norm = x.clone().powf_scalar(2.0).sum_dim(2).sqrt();
    let f = norm.clone().ones_like().mul_scalar(thr).div(norm.add_scalar(1e-6)).clamp_max(1.0);
    x * f
}

/// RMS-normalise each `[.., d]` vector, with no gain.
pub(super) fn unit_rms(x: Tensor<Bk, 3>, eps: f64) -> Tensor<Bk, 3> {
    if let Some(y) = super::rows::unit_rms(x.clone(), eps) {
        return y;
    }
    let denom = x.clone().powf_scalar(2.0).mean_dim(2).add_scalar(eps).sqrt();
    x / denom
}

/// `Inkling2LoopBlockAttentionResidual` with one query: attention from a
/// learned query over the normed `[blocks.., partial]`, whose values are the
/// same normed states. Upstream builds this norm with its default eps, 1e-6,
/// not the config's.
pub(super) fn block_attention(query: Tensor<Bk, 1>, blocks: &[Tensor<Bk, 2>], partial: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
    if let Some(y) = super::rows::block_attention(query.clone(), blocks, partial.clone()) {
        return y;
    }
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

/// Bind a layer's router and routed experts ([`Bank`]), the experts in the
/// format they are stored in: U8 codes are NVFP4 (layers 0..19 of the
/// release), F8_E4M3 values are MXFP8 (layers 20..23). Each projection is one
/// device buffer, experts stacked.
pub(super) fn bind_bank(
    w: &dyn Weights,
    d: &Dims,
    layer: usize,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    router: Bf16W,
) -> Result<Bank, String> {
    let bias = f32_le(want(w, Slot::Layer(layer, LayerPart::GateBias), Dtype::F32, &[d.experts])?.bytes);
    let w13 = Slot::Layer(layer, LayerPart::Expert(ExpertMat::W13, ExpertPart::Weight));
    let projs = match w.tensor(&w13.tensor_name())?.dtype {
        Dtype::F8E4M3 => bind_mxfp8_experts(w, d, layer, client)?,
        _ => bind_nvfp4_experts(w, d, layer, client)?,
    };
    Ok(Bank::new(client, router, bias, projs, d.top_k, d.route_scale, d.gate_softcap, d.up_softcap))
}

/// Fill `bufs`, each `e` equal parts, with `f(x, parts)` for every expert `x`,
/// on as many host threads as there are.
pub(super) fn fill_experts<const N: usize>(
    e: usize,
    bufs: [&mut [u8]; N],
    f: impl Fn(usize, [&mut [u8]; N]) -> Result<(), String> + Sync,
) -> Result<(), String> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = e.div_ceil(threads);
    let parts = bufs.each_ref().map(|b| b.len() / e);
    let mut i = 0;
    let mut chunks = bufs.map(|b| {
        i += 1;
        b.chunks_mut(per * parts[i - 1])
    });
    std::thread::scope(|s| {
        let threads: Vec<_> = (0..e)
            .step_by(per)
            .map(|first| {
                let mine = chunks.each_mut().map(|c| c.next().expect("one chunk per thread"));
                let f = &f;
                s.spawn(move || {
                    let mut i = 0;
                    let mut mine = mine.map(|m| {
                        i += 1;
                        m.chunks_mut(parts[i - 1])
                    });
                    (first..e.min(first + per)).try_for_each(|x| f(x, mine.each_mut().map(|m| m.next().expect("one part per expert"))))
                })
            })
            .collect();
        threads.into_iter().try_for_each(|t| t.join().expect("a bind thread panicked"))
    })
}

/// MXFP8 experts: E4M3 values `[e, rows, cols]` and one UE8M0 scale per 32
/// values of a row. Keep both in fragment order on devices that can convert
/// E4M3 in registers. Older devices use the decoded BF16 weight bank.
fn bind_mxfp8_experts(
    w: &dyn Weights,
    d: &Dims,
    layer: usize,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
) -> Result<[Proj; 3], String> {
    use crate::models::inkling2_loop::experts::{MXFP8_BLOCK, decode_mxfp8_row};
    let (e, h, i) = (d.experts, d.hidden, d.inter);
    let part = |m: ExpertMat, p: ExpertPart| Slot::Layer(layer, LayerPart::Expert(m, p));
    let rows13 = i.checked_mul(2).ok_or_else(|| format!("layer {layer}: MXFP8 row count overflow"))?;
    let w13 = want(w, part(ExpertMat::W13, ExpertPart::Weight), Dtype::F8E4M3, &[e, rows13, h])?;
    let s13 = want(w, part(ExpertMat::W13, ExpertPart::Scale), Dtype::U8, &[e, rows13, h / MXFP8_BLOCK])?;
    let w2 = want(w, part(ExpertMat::W2, ExpertPart::Weight), Dtype::F8E4M3, &[e, h, i])?;
    let s2 = want(w, part(ExpertMat::W2, ExpertPart::Scale), Dtype::U8, &[e, h, i / MXFP8_BLOCK])?;
    if e == 0 || h == 0 || i == 0 {
        return Err(format!("layer {layer}: empty MXFP8 expert bank"));
    }
    for (n, k) in [(i, h), (h, i)] {
        if !Bf16W::tileable(n, k) || k % MXFP8_BLOCK != 0 {
            return Err(format!("layer {layer}: MXFP8 expert {n}x{k} does not tile"));
        }
    }
    // Weight sources can include storage padding after the logical tensor.
    // Slice by the declared shape before either consumer divides it into experts.
    fn logical(t: Stored<'_>) -> Result<&[u8], String> {
        let len = t.shape.iter().try_fold(1usize, |len, &n| len.checked_mul(n))
            .ok_or_else(|| "MXFP8 tensor length overflow".to_string())?;
        t.bytes.get(..len).ok_or_else(|| format!("short MXFP8 tensor: {} bytes for shape {:?}", t.bytes.len(), t.shape))
    }
    let (w13, s13, w2, s2) = (logical(w13)?, logical(s13)?, logical(w2)?, logical(s2)?);
    if super::mxfp8::supported(client) {
        let bind = |codes: &[u8], scales: &[u8], n, k, first, step| {
            super::mxfp8::bind(client, codes, scales, e, n, k, first, step).map_err(|err| format!("layer {layer}: {err}"))
        };
        return Ok([bind(w13, s13, i, h, 0, 2)?, bind(w13, s13, i, h, 1, 2)?, bind(w2, s2, h, i, 0, 1)?]);
    }
    // rows `first, first + step, ..` (count of them) of a `[rows, cols]` matrix, as BF16 bytes into `out`
    let decode = |vals: &[u8], scales: &[u8], cols: usize, first: usize, step: usize, out: &mut [u8]| -> Result<(), String> {
        let per = cols / MXFP8_BLOCK;
        let mut row = vec![0f32; cols];
        for (j, out) in out.chunks_exact_mut(cols * 2).enumerate() {
            let r = first + j * step;
            decode_mxfp8_row(&vals[r * cols..(r + 1) * cols], &scales[r * per..(r + 1) * per], MXFP8_BLOCK, &mut row);
            for (&v, out) in row.iter().zip(out.chunks_exact_mut(2)) {
                let b = half::bf16::from_f32(v);
                if b.to_f32().to_bits() != v.to_bits() && !v.is_nan() {
                    return Err(format!("layer {layer}: MXFP8 value {v} is not a BF16"));
                }
                out.copy_from_slice(&b.to_le_bytes());
            }
        }
        Ok(())
    };
    let (m13, p13) = (2 * i * h, 2 * i * (h / MXFP8_BLOCK));
    let (m2, p2) = (h * i, h * (i / MXFP8_BLOCK));
    let part = e * i * h * 2;
    let (mut gate, mut up, mut down) = (vec![0; part], vec![0; part], vec![0; part]);
    fill_experts(e, [&mut gate, &mut up, &mut down], |x, [gate, up, down]| {
        let (v13, c13) = (&w13[x * m13..(x + 1) * m13], &s13[x * p13..(x + 1) * p13]);
        let (v2, c2) = (&w2[x * m2..(x + 1) * m2], &s2[x * p2..(x + 1) * p2]);
        decode(v13, c13, h, 0, 2, gate)?;
        decode(v13, c13, h, 1, 2, up)?;
        decode(v2, c2, i, 0, 1, down)
    })?;
    Ok([bf16_proj(client, &gate, e, i, h), bf16_proj(client, &up, e, i, h), bf16_proj(client, &down, e, h, i)])
}

fn bind_nvfp4_experts(
    w: &dyn Weights,
    d: &Dims,
    layer: usize,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
) -> Result<[Proj; 3], String> {
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
    let rows = |src: &[u8], width: usize, e: usize, parity: usize, out: &mut [u8]| {
        let base = e * 2 * i * width;
        for (r, out) in out.chunks_exact_mut(width).enumerate() {
            let s = base + (2 * r + parity) * width;
            out.copy_from_slice(&src[s..s + width]);
        }
    };
    let (codes, scales) = (e * i * cb13, e * i * sb13);
    let (mut gc, mut gs, mut uc, mut us) = (vec![0; codes], vec![0; scales], vec![0; codes], vec![0; scales]);
    fill_experts(e, [&mut gc, &mut gs, &mut uc, &mut us], |x, [gc, gs, uc, us]| {
        rows(w13.bytes, cb13, x, 0, gc);
        rows(s13.bytes, sb13, x, 0, gs);
        rows(w13.bytes, cb13, x, 1, uc);
        rows(s13.bytes, sb13, x, 1, us);
        Ok(())
    })?;
    Ok([
        nvfp4_proj(client, &gc, &gs, sc13.clone(), i, h),
        nvfp4_proj(client, &uc, &us, sc13, i, h),
        nvfp4_proj(client, &w2.bytes[..e * h * cb2], &s2.bytes[..e * h * sb2], sc2, h, i),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp8_bind_checks_lengths_and_accepts_padded_release_layout() {
        use crate::models::inkling2_loop::{config::Inkling2LoopConfig, moe::Stack};
        use cubecl::{
            cuda::{CudaDevice, CudaRuntime},
            prelude::Runtime,
        };
        use std::collections::HashMap;

        struct Source(HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>);
        impl Weights for Source {
            fn tensor(&self, name: &str) -> Result<Stored<'_>, String> {
                let (dtype, shape, bytes) = self.0.get(name).ok_or_else(|| name.to_owned())?;
                Ok(Stored {
                    dtype: *dtype,
                    shape: shape.clone(),
                    bytes,
                })
            }
        }
        let mut d = Dims::of(
            &Inkling2LoopConfig::from_json(include_str!("fixtures/config.json"))
                .unwrap()
                .text_config,
        );
        (d.experts, d.hidden, d.inter) = (3, 64, 32);
        let mut w = Source(HashMap::new());
        let name = |m, p| Slot::Layer(0, LayerPart::Expert(m, p)).tensor_name();
        for (mat, rows, cols) in [
            (ExpertMat::W13, 2 * d.inter, d.hidden),
            (ExpertMat::W2, d.hidden, d.inter),
        ] {
            for (part, dtype, width) in [
                (ExpertPart::Weight, Dtype::F8E4M3, cols),
                (ExpertPart::Scale, Dtype::U8, cols / 32),
            ] {
                let len = d.experts * rows * width;
                let mut bytes: Vec<_> = (0..len)
                    .map(|j| match part {
                        ExpertPart::Weight => ((j * 17 + j / cols) % 127) as u8,
                        _ => 120 + (j % 13) as u8,
                    })
                    .collect();
                bytes.extend_from_slice(&[255; 7]);
                w.0.insert(name(mat, part), (dtype, vec![d.experts, rows, width], bytes));
            }
        }
        let client = CudaRuntime::client(&CudaDevice::default());
        let projs = bind_mxfp8_experts(&w, &d, 0, &client).unwrap();
        for (projection, (mat, first, step)) in
            projs
                .into_iter()
                .zip([(ExpertMat::W13, 0, 2), (ExpertMat::W13, 1, 2), (ExpertMat::W2, 0, 1)])
        {
            if let Stack::Mxfp8 { codes, scales } = projection.stack {
                let packed = client.read_one(codes).unwrap();
                let exps = client.read_one(scales).unwrap();
                let (n, k) = (projection.n, projection.k);
                let src = w.tensor(&name(mat, ExpertPart::Weight)).unwrap();
                let scale = w.tensor(&name(mat, ExpertPart::Scale)).unwrap();
                assert_eq!((packed.len(), exps.len()), (d.experts * n * k, d.experts * n * k / 32));
                for row in 0..d.experts * n {
                    for col in 0..k {
                        let source = (row * step + first) * k + col;
                        let group = (row / 8) * (k / 32) + col / 32;
                        assert_eq!(
                            packed[group * 256 + (col % 32) / 8 * 64 + row % 8 * 8 + col % 8],
                            src.bytes[source]
                        );
                        assert_eq!(exps[group * 8 + row % 8], scale.bytes[source / 32]);
                    }
                }
            } else {
                assert!(!super::super::mxfp8::supported(&client));
            }
        }
        for key in w.0.keys().cloned().collect::<Vec<_>>() {
            let (_, shape, bytes) = w.0.get_mut(&key).unwrap();
            let tail = bytes.split_off(shape.iter().product::<usize>() - 1);
            assert!(
                bind_mxfp8_experts(&w, &d, 0, &client)
                    .err()
                    .unwrap()
                    .contains("short MXFP8 tensor")
            );
            w.0.get_mut(&key).unwrap().2.extend(tail);
        }
        w.0.get_mut(&name(ExpertMat::W13, ExpertPart::Weight)).unwrap().2[0] = 1;
        w.0.get_mut(&name(ExpertMat::W13, ExpertPart::Scale)).unwrap().2[0] = 0;
        assert!(bind_mxfp8_experts(&w, &d, 0, &client).err().unwrap().contains("BF16"));
    }

    #[test]
    fn every_expert_fills_its_own_parts() {
        for e in [1, 2, 7, 19, 20, 21, 64, 257] {
            let (mut a, mut b) = (vec![0u8; e * 3], vec![0u8; e * 5]);
            fill_experts(e, [&mut a, &mut b], |x, [a, b]| {
                a.fill(x as u8);
                b.fill(!(x as u8));
                Ok(())
            })
            .unwrap();
            assert!(a.chunks(3).zip(b.chunks(5)).enumerate().all(|(x, (a, b))| {
                a.iter().all(|v| *v == x as u8) && b.iter().all(|v| *v == !(x as u8))
            }));
            let stop = e / 2;
            let err = fill_experts(e, [&mut a, &mut b], |x, _| if x == stop { Err("stop".into()) } else { Ok(()) });
            assert_eq!(err, Err("stop".to_string()));
        }
    }
}
