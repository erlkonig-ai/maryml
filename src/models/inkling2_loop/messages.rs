//! Cross-pass messages: what a core layer (6..17) reads from the previous pass
//! through the loop, and how it mixes that in. The f32 reference lane, on plain
//! slices.
//!
//! From transformers 046bdb8c, `Inkling2LoopMessageAttentionResidual`,
//! `Inkling2LoopMessageGate` and `Inkling2LoopMergeMessageQueries`:
//!
//! ```text
//! S      = rms([blocks_prev, partial_prev])     per token, [n + 1, H], unit RMS, eps 1e-6
//! m_j    = softmax_k(q_j . S_k / sqrt(H)) S     j in Q, K, V, MLP: four learned queries per core layer
//! gate(h, m):
//!   d    = W_down [rms(h), rms(m)]              rank r; a BF16 GEMM in the released model
//!   out  = (1 - tanh(W_forget d)) * h + tanh(W_gain d) * m
//! ```
//!
//! What the previous pass hands over, in the release: `blocks_prev` is the
//! previous pass's block states without the one it appended at the core's
//! first layer (upstream's `num_pre_loop_blocks + 1` skip), so exactly one
//! state, the input of layer 12; `partial_prev` is the previous pass's output
//! (layer 17's). The keys are the same for all twelve core layers of a pass;
//! only the queries differ. Nothing here is causal across tokens: each token
//! reads its own states. Layer 0 and the first pass read no messages
//! ([`super::schedule::Visit::reads_messages`]).
//!
//! Two precisions are easy to get wrong:
//!
//! * `down_proj` is the one gate projection *not* in upstream's
//!   `_keep_in_fp32_modules_strict`, so in the released (bfloat16) model its
//!   input is rounded to bf16 and its rank-`r` output is a bf16 GEMM, while the
//!   forget and gain projections and the queries run in f32.
//!   [`DownPrecision::Bf16`] is that; [`DownPrecision::F32`] is the same model
//!   loaded at float32. On the captured fixture the two differ by up to 0.11 on
//!   outputs of magnitude 32, so a gate that skips the rounding is noticed.
//! * The message attention's norm has a fixed eps of 1e-6
//!   ([`MESSAGE_NORM_EPS`], upstream constructs it with no argument); the gate's
//!   norm takes the config's `rms_norm_eps`. Both are 1e-6 in the release.
//!
//! Where the messages enter a core layer is the layer's business, not this
//! module's: sliding core layers gate the attention-residual output before
//! `input_layernorm` (one input, three gates), global core layers gate their
//! three separate Q/K/V attention-residual outputs before their own norms, and
//! the MLP gate sits between `mlp_input` and `post_attention_layernorm`.

use half::bf16;

use super::layout::Stream;

/// Eps of the message attention's unweighted RMS norm (upstream's default,
/// not the config's).
pub const MESSAGE_NORM_EPS: f32 = 1e-6;

/// Where a stream's message sits in the four a core layer reads.
pub fn message_slot(s: Stream) -> usize {
    match s {
        Stream::Query => 0,
        Stream::Key => 1,
        Stream::Value => 2,
        Stream::Mlp => 3,
    }
}

/// `x * rsqrt(mean(x^2) + eps)`, in f32 like upstream after its `.float()`.
pub fn unweighted_rms_norm(row: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(row.len(), out.len());
    let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let scale = (mean_sq + eps).sqrt().recip();
    for (o, &v) in out.iter_mut().zip(row) {
        *o = v * scale;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The message queries of every core layer, `[core, 4, H]` in the order Q, K,
/// V, MLP.
#[derive(Debug, Clone)]
pub struct MessageQueries {
    pub core: usize,
    pub hidden: usize,
    data: Vec<f32>,
}

impl MessageQueries {
    /// From the release's two `[2 * core, H]` tensors:
    /// `middle_cycle_cross_pass_queries` holds `[Q, MLP]` of each core layer,
    /// `middle_cycle_cross_pass_kv_queries` holds `[K, V]`.
    pub fn from_checkpoint(
        queries: &[f32],
        kv_queries: &[f32],
        core: usize,
        hidden: usize,
    ) -> Result<MessageQueries, String> {
        for (name, t) in [("cross_pass_queries", queries), ("cross_pass_kv_queries", kv_queries)] {
            if t.len() != 2 * core * hidden {
                return Err(format!(
                    "{name}: {} values, want [2 * {core}, {hidden}] = {}",
                    t.len(),
                    2 * core * hidden
                ));
            }
        }
        let row = |t: &[f32], r: usize| t[r * hidden..(r + 1) * hidden].to_vec();
        let mut data = Vec::with_capacity(core * 4 * hidden);
        for l in 0..core {
            data.extend(row(queries, 2 * l)); // Q
            data.extend(row(kv_queries, 2 * l)); // K
            data.extend(row(kv_queries, 2 * l + 1)); // V
            data.extend(row(queries, 2 * l + 1)); // MLP
        }
        Ok(MessageQueries { core, hidden, data })
    }

    /// The four queries of core layer `index` (`0..core`, layer `core_start + index`), `[4, H]`.
    pub fn layer(&self, index: usize) -> &[f32] {
        assert!(index < self.core, "core layer {index} of {}", self.core);
        &self.data[index * 4 * self.hidden..(index + 1) * 4 * self.hidden]
    }

    /// All of them, `[core, 4, H]`.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// The four messages one core layer reads, `[tokens, 4, H]`.
///
/// `queries` is that layer's `[4, H]`; `blocks` is the previous pass's
/// `[tokens, n_blocks, H]` (zero blocks is allowed); `partial` is its output,
/// `[tokens, H]`.
pub fn read_messages(
    queries: &[f32],
    blocks: &[f32],
    partial: &[f32],
    tokens: usize,
    n_blocks: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(queries.len(), 4 * hidden);
    assert_eq!(blocks.len(), tokens * n_blocks * hidden);
    assert_eq!(partial.len(), tokens * hidden);
    let keys = n_blocks + 1;
    let scaling = (hidden as f32).powf(-0.5);
    let mut out = vec![0f32; tokens * 4 * hidden];
    let mut stacked = vec![0f32; keys * hidden];
    let mut p = vec![0f32; keys];
    for t in 0..tokens {
        for k in 0..keys {
            let src = if k < n_blocks {
                &blocks[(t * n_blocks + k) * hidden..(t * n_blocks + k + 1) * hidden]
            } else {
                &partial[t * hidden..(t + 1) * hidden]
            };
            unweighted_rms_norm(src, MESSAGE_NORM_EPS, &mut stacked[k * hidden..(k + 1) * hidden]);
        }
        for j in 0..4 {
            let q = &queries[j * hidden..(j + 1) * hidden];
            // upstream: matmul, then `* scaling`, then a float32 softmax
            for (k, pk) in p.iter_mut().enumerate() {
                *pk = dot(q, &stacked[k * hidden..(k + 1) * hidden]) * scaling;
            }
            let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for pk in p.iter_mut() {
                *pk = (*pk - max).exp();
                sum += *pk;
            }
            let o = &mut out[(t * 4 + j) * hidden..(t * 4 + j + 1) * hidden];
            for (k, &pk) in p.iter().enumerate() {
                let w = pk / sum;
                for (oi, &s) in o.iter_mut().zip(&stacked[k * hidden..(k + 1) * hidden]) {
                    *oi += w * s;
                }
            }
        }
    }
    out
}

/// The precision of a message gate's down projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownPrecision {
    /// As the released bfloat16 model runs it: bf16 input and weights, f32
    /// accumulation, bf16 output.
    Bf16,
    /// The same model loaded at float32.
    F32,
}

/// One message gate (`controllers.{attn,k,v,mlp}` of a core layer).
#[derive(Debug, Clone)]
pub struct MessageGate {
    pub hidden: usize,
    pub rank: usize,
    pub eps: f32,
    pub precision: DownPrecision,
    /// `[rank, 2H]`; rounded to bf16 under [`DownPrecision::Bf16`].
    down: Vec<f32>,
    /// `[H, rank]`, f32.
    forget: Vec<f32>,
    /// `[H, rank]`, f32 (`gain_weight`, upstream's `message_gate_proj`).
    gain: Vec<f32>,
}

/// A gate's intermediate values, for comparing against upstream's taps.
#[derive(Debug, Clone)]
pub struct GateTrace {
    /// `[tokens, rank]`, the down projection's output (bf16 values under Bf16).
    pub down: Vec<f32>,
    /// `[tokens, H]`, before the tanh.
    pub forget_pre: Vec<f32>,
    /// `[tokens, H]`, before the tanh.
    pub gain_pre: Vec<f32>,
    /// `[tokens, H]`.
    pub out: Vec<f32>,
}

fn round_bf16(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

impl MessageGate {
    pub fn new(
        down: Vec<f32>,
        forget: Vec<f32>,
        gain: Vec<f32>,
        hidden: usize,
        rank: usize,
        eps: f32,
        precision: DownPrecision,
    ) -> Result<MessageGate, String> {
        if down.len() != rank * 2 * hidden {
            return Err(format!("down: {} values, want [{rank}, {}]", down.len(), 2 * hidden));
        }
        for (name, t) in [("forget", &forget), ("gain", &gain)] {
            if t.len() != hidden * rank {
                return Err(format!("{name}: {} values, want [{hidden}, {rank}]", t.len()));
            }
        }
        let down = match precision {
            DownPrecision::Bf16 => down.into_iter().map(round_bf16).collect(),
            DownPrecision::F32 => down,
        };
        Ok(MessageGate { hidden, rank, eps, precision, down, forget, gain })
    }

    /// From the checkpoint's three BF16 tensors.
    pub fn from_bf16(
        down: &[bf16],
        forget: &[bf16],
        gain: &[bf16],
        hidden: usize,
        rank: usize,
        eps: f32,
        precision: DownPrecision,
    ) -> Result<MessageGate, String> {
        let f = |t: &[bf16]| t.iter().map(|v| v.to_f32()).collect::<Vec<f32>>();
        MessageGate::new(f(down), f(forget), f(gain), hidden, rank, eps, precision)
    }

    /// `(1 - tanh(forget)) * hidden + tanh(gain) * message`, `[tokens, H]`.
    pub fn apply(&self, hidden_states: &[f32], message: &[f32], tokens: usize) -> Vec<f32> {
        self.trace(hidden_states, message, tokens).out
    }

    pub fn trace(&self, hidden_states: &[f32], message: &[f32], tokens: usize) -> GateTrace {
        let (h, r) = (self.hidden, self.rank);
        assert_eq!(hidden_states.len(), tokens * h);
        assert_eq!(message.len(), tokens * h);
        let mut tr = GateTrace {
            down: vec![0f32; tokens * r],
            forget_pre: vec![0f32; tokens * h],
            gain_pre: vec![0f32; tokens * h],
            out: vec![0f32; tokens * h],
        };
        let mut x = vec![0f32; 2 * h];
        for t in 0..tokens {
            let hs = &hidden_states[t * h..(t + 1) * h];
            let ms = &message[t * h..(t + 1) * h];
            unweighted_rms_norm(hs, self.eps, &mut x[..h]);
            unweighted_rms_norm(ms, self.eps, &mut x[h..]);
            if self.precision == DownPrecision::Bf16 {
                x.iter_mut().for_each(|v| *v = round_bf16(*v));
            }
            let d = &mut tr.down[t * r..(t + 1) * r];
            for (i, di) in d.iter_mut().enumerate() {
                let v = dot(&self.down[i * 2 * h..(i + 1) * 2 * h], &x);
                *di = match self.precision {
                    DownPrecision::Bf16 => round_bf16(v),
                    DownPrecision::F32 => v,
                };
            }
            for c in 0..h {
                let f = dot(&self.forget[c * r..(c + 1) * r], d);
                let g = dot(&self.gain[c * r..(c + 1) * r], d);
                tr.forget_pre[t * h + c] = f;
                tr.gain_pre[t * h + c] = g;
                tr.out[t * h + c] = (1.0 - f.tanh()) * hs[c] + g.tanh() * ms[c];
            }
        }
        tr
    }
}

// ---------------------------------------------------------------------------
// The device lane: the same messages and gates, batched over a call's tokens,
// with the previous pass's states kept on the device. The f32 slice functions
// above stay as the reference these are checked against.
#[cfg(feature = "inkling-cuda")]
pub use device::*;

#[cfg(feature = "inkling-cuda")]
mod device {
    use burn::prelude::*;
    use burn::tensor::activation::softmax;
    use burn::tensor::{FloatDType, TensorData};

    use super::{DownPrecision, MESSAGE_NORM_EPS, MessageGate, MessageQueries, message_slot};
    use crate::models::inkling::seam::Bk;
    use crate::models::inkling2_loop::layout::Stream;
    use crate::models::inkling2_loop::sliding::Dev;

    /// The message queries of every core layer on the device, each layer's four
    /// as `[1, 4, H]` (Q, K, V, MLP).
    pub struct DeviceQueries {
        layers: Vec<Tensor<Bk, 3>>,
    }

    impl DeviceQueries {
        pub fn new(q: &MessageQueries, dev: &Dev) -> DeviceQueries {
            let h = q.hidden;
            let layers = (0..q.core)
                .map(|l| Tensor::<Bk, 3>::from_data(TensorData::new(q.layer(l).to_vec(), [1, 4, h]), dev))
                .collect();
            DeviceQueries { layers }
        }

        /// Core layer `index`'s queries, `[1, 4, H]`.
        pub fn layer(&self, index: usize) -> Tensor<Bk, 3> {
            self.layers[index].clone()
        }
    }

    /// `x / sqrt(mean(x^2) + eps)` over each row of `[t, H]`.
    fn unit_rms_rows(x: Tensor<Bk, 2>, eps: f64) -> Tensor<Bk, 2> {
        let denom = x.clone().powf_scalar(2.0).mean_dim(1).add_scalar(eps).sqrt();
        x / denom
    }

    /// Round every value to the nearest bf16 (ties to even), kept as f32.
    fn round_bf16_dev(x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        x.cast(FloatDType::BF16).cast(FloatDType::F32)
    }

    /// What a pass hands the next one: the normed message sources
    /// `rms([blocks_prev, partial_prev])`, `[t, n + 1, H]`, normed once per pass
    /// and shared by all twelve core layers.
    pub struct MessageSources {
        normed: Tensor<Bk, 3>,
    }

    impl MessageSources {
        /// `blocks` are the previous pass's kept block states (each `[t, H]`, zero
        /// is allowed), `partial` its output.
        pub fn new(blocks: &[Tensor<Bk, 2>], partial: Tensor<Bk, 2>) -> MessageSources {
            let [t, h] = partial.dims();
            let mut all: Vec<Tensor<Bk, 2>> = blocks.to_vec();
            all.push(partial);
            let keys = all.len();
            let stacked = Tensor::cat(all.into_iter().map(|b| b.reshape([t, 1, h])).collect(), 1);
            let normed = unit_rms_rows(stacked.reshape([t * keys, h]), MESSAGE_NORM_EPS as f64).reshape([t, keys, h]);
            MessageSources { normed }
        }

        /// The four messages of one core layer, `[t, 4, H]`, from its queries
        /// `[1, 4, H]` ([`DeviceQueries::layer`]). [`read_messages`] on the device.
        ///
        /// Elementwise products and sums over the (one to three) keys, not a
        /// matrix multiply: Burn's f32 matmul on this runtime is TF32, and the
        /// queries and normed states are f32 values upstream multiplies in f32.
        pub fn read(&self, queries: Tensor<Bk, 3>) -> Tensor<Bk, 3> {
            if let Some(m) = crate::models::inkling2_loop::rows::read_messages(&self.normed, queries.clone()) {
                return m;
            }
            let [t, keys, h] = self.normed.dims();
            let scaling = (h as f32).powf(-0.5);
            let key = |k: usize| self.normed.clone().slice([0..t, k..k + 1, 0..h]); // [t, 1, H]
            // upstream: matmul, then `* scaling`, then a float32 softmax over the keys
            let logits = Tensor::cat(
                (0..keys).map(|k| (key(k) * queries.clone()).sum_dim(2)).collect(), // each [t, 4, 1]
                2,
            )
            .mul_scalar(scaling);
            let p = softmax(logits, 2); // [t, 4, keys]
            let mut out: Option<Tensor<Bk, 3>> = None;
            for k in 0..keys {
                let term = p.clone().slice([0..t, 0..4, k..k + 1]) * key(k);
                out = Some(match out {
                    None => term,
                    Some(o) => o + term,
                });
            }
            out.expect("at least the partial state is a key")
        }
    }

    /// The `[t, H]` message of stream `s` out of a visit's `[t, 4, H]` messages.
    pub fn message_tensor(m: &Tensor<Bk, 3>, s: Stream) -> Tensor<Bk, 2> {
        let [t, _, h] = m.dims();
        let j = message_slot(s);
        m.clone().slice([0..t, j..j + 1, 0..h]).reshape([t, h])
    }

    /// One message gate on the device, as f32 matrix multiplies.
    ///
    /// Burn's f32 matmul on this runtime is TF32, and that is exact here under
    /// [`DownPrecision::Bf16`], the released model: every operand is a bf16
    /// value (the bf16-rounded normed inputs and down weights, then the
    /// bf16-rounded rank-`r` output and the checkpoint's bf16 forget and gain
    /// weights), bf16's 8 significant bits fit TF32's 11, and accumulation is
    /// f32. So this is the released bf16 GEMM, then upstream's f32 forget and
    /// gain, up to summation order. Under [`DownPrecision::F32`] the operands
    /// are full f32 and the products are TF32's; nothing binds that lane.
    /// Forget and gain are one multiply against both, stacked.
    pub struct DeviceGate {
        pub hidden: usize,
        pub rank: usize,
        eps: f64,
        precision: DownPrecision,
        /// `[2H, rank]`, the down projection transposed.
        down_t: Tensor<Bk, 2>,
        /// `[rank, 2H]`: columns `0..H` forget, `H..2H` gain.
        forget_gain_t: Tensor<Bk, 2>,
    }

    impl DeviceGate {
        pub fn new(g: &MessageGate, dev: &Dev) -> DeviceGate {
            let (h, r) = (g.hidden, g.rank);
            let down = Tensor::<Bk, 2>::from_data(TensorData::new(g.down.clone(), [r, 2 * h]), dev).transpose();
            let mut fg = g.forget.clone();
            fg.extend_from_slice(&g.gain);
            let fg = Tensor::<Bk, 2>::from_data(TensorData::new(fg, [2 * h, r]), dev).transpose();
            DeviceGate { hidden: h, rank: r, eps: g.eps as f64, precision: g.precision, down_t: down, forget_gain_t: fg }
        }

        /// `(1 - tanh(forget)) * hidden + tanh(gain) * message`, `[t, H]`:
        /// [`MessageGate::apply`] on the device.
        pub fn apply(&self, hidden_states: Tensor<Bk, 2>, message: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
            let [t, h] = hidden_states.dims();
            assert_eq!(h, self.hidden);
            use crate::models::inkling2_loop::rows;
            let bf = self.precision == DownPrecision::Bf16;
            let x = match rows::gate_input(hidden_states.clone(), message.clone(), self.eps, bf) {
                Some(x) => x,
                None => {
                    let x = Tensor::cat(
                        vec![unit_rms_rows(hidden_states.clone(), self.eps), unit_rms_rows(message.clone(), self.eps)],
                        1,
                    );
                    if bf { round_bf16_dev(x) } else { x }
                }
            };
            // The short-call GEMV only where every operand is a bf16 value, so
            // its f32 products equal the TF32 matmul's: under F32 it would
            // change the variant's arithmetic with the row count.
            let mm = |a: Tensor<Bk, 2>, w: &Tensor<Bk, 2>| match bf.then(|| rows::gemv(a.clone(), w)).flatten() {
                Some(y) => y,
                None => a.matmul(w.clone()),
            };
            let d = mm(x, &self.down_t);
            let d = if bf { rows::round_to_bf16(d.clone()).unwrap_or_else(|| round_bf16_dev(d)) } else { d };
            let fg = mm(d, &self.forget_gain_t);
            if let Some(out) = rows::gate_output(fg.clone(), hidden_states.clone(), message.clone()) {
                return out;
            }
            let f = fg.clone().slice([0..t, 0..h]);
            let g = fg.slice([0..t, h..2 * h]);
            f.tanh().neg().add_scalar(1.0) * hidden_states + g.tanh() * message
        }
    }

    /// A core layer's four gates on the device, in [`message_slot`] order.
    pub fn device_gates(g: &[MessageGate; 4], dev: &Dev) -> [DeviceGate; 4] {
        [0, 1, 2, 3].map(|j| DeviceGate::new(&g[j], dev))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `golden/capture_inkling2_messages.py` (transformers 046bdb8c's own
    /// classes, torch 2.11 CPU, vllm/vllm-openai:v0.26.0 offline). sha256 in
    /// `fixtures/messages_reference.json.sha256`.
    const FIXTURE: &str = include_str!("fixtures/messages_reference.json");
    const STREAMS: [Stream; 4] = [Stream::Query, Stream::Key, Stream::Value, Stream::Mlp];
    const NAMES: [&str; 4] = ["q", "k", "v", "mlp"];

    struct Fx(serde_json::Value);

    impl Fx {
        fn load() -> Fx {
            Fx(serde_json::from_str(FIXTURE).unwrap())
        }
        fn meta(&self, k: &str) -> usize {
            self.0["meta"][k].as_u64().unwrap() as usize
        }
        fn t(&self, k: &str) -> Vec<f32> {
            let v = &self.0["tensors"][k];
            assert!(v.is_object(), "fixture has no tensor {k}");
            v["bits"].as_array().unwrap().iter().map(|b| f32::from_bits(b.as_u64().unwrap() as u32)).collect()
        }
    }

    /// The tolerance on every comparison, relative to the tensor's peak. What it
    /// absorbs is f32 summation order (torch's blocked reductions against these
    /// sequential sums), measured at 4e-7 or less on every tap; a gate run at
    /// the other precision is off by 2.4e-3 or more, 1000 times this.
    const TOL: f32 = 2e-6;

    /// Largest |got - want| over the largest |want|.
    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got.iter().zip(want).fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        err / peak
    }

    fn queries(fx: &Fx) -> MessageQueries {
        let (core, h) = (fx.meta("core_layers"), fx.meta("hidden_size"));
        MessageQueries::from_checkpoint(&fx.t("queries"), &fx.t("kv_queries"), core, h).unwrap()
    }

    #[test]
    fn the_checkpoint_queries_merge_like_upstream() {
        let fx = Fx::load();
        let q = queries(&fx);
        let want = fx.t("merged");
        // a permutation of stored values: bit for bit
        assert_eq!(q.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn the_four_messages_match_upstream() {
        let fx = Fx::load();
        let q = queries(&fx);
        let (h, tokens) = (fx.meta("hidden_size"), fx.meta("tokens"));
        let partial = fx.t("partial");
        let mut checked = 0;
        let mut worst = 0f32;
        for n in 0..3 {
            let blocks = fx.t(&format!("blocks{n}"));
            for l in [0, 7, fx.meta("core_layers") - 1] {
                let got = read_messages(q.layer(l), &blocks, &partial, tokens, n, h);
                let want = fx.t(&format!("messages.n{n}.l{l}"));
                // per message and token, so a swapped Q/K/V/MLP or token order is its own failure
                for t in 0..tokens {
                    for j in 0..4 {
                        let r = (t * 4 + j) * h..(t * 4 + j + 1) * h;
                        let e = rel_err(&got[r.clone()], &want[r]);
                        worst = worst.max(e);
                        assert!(e < TOL, "n_blocks {n} layer {l} token {t} message {j}: rel err {e}");
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 3 * 3 * tokens * 4);
        eprintln!("{checked} messages, worst rel err {worst}");
    }

    fn gates(fx: &Fx, precision: DownPrecision) -> Vec<MessageGate> {
        let (h, r) = (fx.meta("hidden_size"), fx.meta("loop_gate_rank"));
        NAMES
            .iter()
            .map(|s| {
                MessageGate::new(
                    fx.t(&format!("gate.{s}.down")),
                    fx.t(&format!("gate.{s}.forget")),
                    fx.t(&format!("gate.{s}.gain")),
                    h,
                    r,
                    1e-6,
                    precision,
                )
                .unwrap()
            })
            .collect()
    }

    /// Gates the four streams of layer 7 against the release's geometry (one
    /// previous block) and returns the worst relative error of each tap.
    fn gate_errors(fx: &Fx, run: &str, gates: &[MessageGate]) -> [f32; 4] {
        let (h, tokens) = (fx.meta("hidden_size"), fx.meta("tokens"));
        let msgs = fx.t(fx.0["meta"]["gate_message"].as_str().unwrap());
        let mut worst = [0f32; 4];
        for (i, (s, gate)) in STREAMS.iter().zip(gates).enumerate() {
            let name = NAMES[i];
            let j = message_slot(*s);
            let message: Vec<f32> = (0..tokens).flat_map(|t| msgs[(t * 4 + j) * h..(t * 4 + j + 1) * h].to_vec()).collect();
            let tr = gate.trace(&fx.t(&format!("hidden.{name}")), &message, tokens);
            for (k, (got, tap)) in [
                (&tr.down, "down_out"),
                (&tr.forget_pre, "forget_pre"),
                (&tr.gain_pre, "gain_pre"),
                (&tr.out, "out"),
            ]
            .into_iter()
            .enumerate()
            {
                worst[k] = worst[k].max(rel_err(got, &fx.t(&format!("{run}.{name}.{tap}"))));
            }
        }
        worst
    }

    #[test]
    fn the_gates_match_upstream_at_f32() {
        let fx = Fx::load();
        let e = gate_errors(&fx, "f32", &gates(&fx, DownPrecision::F32));
        eprintln!("f32 gates, worst rel err of down, forget, gain, out: {e:?}");
        assert!(e.iter().all(|&v| v < TOL), "down, forget, gain, out: {e:?}");
    }

    #[test]
    fn the_gates_match_upstream_as_released() {
        let fx = Fx::load();
        let e = gate_errors(&fx, "bf16", &gates(&fx, DownPrecision::Bf16));
        eprintln!("bf16 gates, worst rel err of down, forget, gain, out: {e:?}");
        // On this fixture the bf16 down outputs, and so the forget and gain
        // pre-activations, are bit-equal to upstream's. On other inputs an f32
        // sum that lands on a bf16 rounding boundary could round the other way;
        // that would move one rank entry by 2^-8 of itself and fail here, and it
        // would be worth a look rather than a looser bound.
        assert!(e.iter().all(|&v| v < TOL), "down, forget, gain, out: {e:?}");
    }

    /// The bf16 fixture refuses a gate that skips the release's rounding, and
    /// the f32 fixture one that adds it: the two tests above can tell.
    #[test]
    fn a_gate_at_the_wrong_precision_is_noticed() {
        let fx = Fx::load();
        let e = gate_errors(&fx, "bf16", &gates(&fx, DownPrecision::F32));
        eprintln!("f32 gate against bf16 fixture: {e:?}");
        assert!(e[3] >= 100.0 * TOL, "an all-f32 gate passed the bf16 fixture: {e:?}");
        let e = gate_errors(&fx, "f32", &gates(&fx, DownPrecision::Bf16));
        eprintln!("bf16 gate against f32 fixture: {e:?}");
        assert!(e[3] >= 100.0 * TOL, "a bf16 gate passed the f32 fixture: {e:?}");
    }

    /// The device lane against the same upstream taps, at the same tolerance:
    /// all five tokens in one call, every token checked.
    #[cfg(feature = "inkling-cuda")]
    mod device {
        use super::*;
        use burn::prelude::*;
        use burn::tensor::TensorData;

        use crate::models::inkling::seam::Bk;

        fn dev() -> burn::backend::cuda::CudaDevice {
            burn::backend::cuda::CudaDevice::default()
        }

        fn up(v: Vec<f32>, shape: [usize; 2]) -> Tensor<Bk, 2> {
            Tensor::from_data(TensorData::new(v, shape), &dev())
        }

        fn host<const D: usize>(t: Tensor<Bk, D>) -> Vec<f32> {
            t.into_data().convert::<f32>().to_vec().unwrap()
        }

        #[test]
        fn the_device_messages_match_upstream() {
            let fx = Fx::load();
            let q = queries(&fx);
            let dq = DeviceQueries::new(&q, &dev());
            let (h, tokens) = (fx.meta("hidden_size"), fx.meta("tokens"));
            let partial = fx.t("partial");
            let mut worst = 0f32;
            for n in 0..3 {
                let blocks = fx.t(&format!("blocks{n}"));
                // the fixture's blocks are [t, n, H]; the device lane takes n tensors [t, H]
                let per_block: Vec<Tensor<Bk, 2>> = (0..n)
                    .map(|k| up((0..tokens).flat_map(|r| blocks[(r * n + k) * h..(r * n + k + 1) * h].to_vec()).collect(), [tokens, h]))
                    .collect();
                let src = MessageSources::new(&per_block, up(partial.clone(), [tokens, h]));
                for l in [0, 7, fx.meta("core_layers") - 1] {
                    let got = host(src.read(dq.layer(l)));
                    let want = fx.t(&format!("messages.n{n}.l{l}"));
                    for t in 0..tokens {
                        for j in 0..4 {
                            let r = (t * 4 + j) * h..(t * 4 + j + 1) * h;
                            let e = rel_err(&got[r.clone()], &want[r]);
                            worst = worst.max(e);
                            assert!(e < TOL, "n_blocks {n} layer {l} token {t} message {j}: rel err {e}");
                        }
                    }
                }
            }
            eprintln!("device messages, worst rel err {worst}");
        }

        fn device_gate_errors(fx: &Fx, run: &str, precision: DownPrecision) -> f32 {
            let (h, tokens) = (fx.meta("hidden_size"), fx.meta("tokens"));
            let msgs = fx.t(fx.0["meta"]["gate_message"].as_str().unwrap());
            let mut worst = 0f32;
            for (i, (s, gate)) in STREAMS.iter().zip(gates(fx, precision)).enumerate() {
                let name = NAMES[i];
                let j = message_slot(*s);
                let message: Vec<f32> = (0..tokens).flat_map(|t| msgs[(t * 4 + j) * h..(t * 4 + j + 1) * h].to_vec()).collect();
                let g = DeviceGate::new(&gate, &dev());
                let got = host(g.apply(up(fx.t(&format!("hidden.{name}")), [tokens, h]), up(message, [tokens, h])));
                let want = fx.t(&format!("{run}.{name}.out"));
                for t in 0..tokens {
                    let e = rel_err(&got[t * h..(t + 1) * h], &want[t * h..(t + 1) * h]);
                    worst = worst.max(e);
                }
            }
            worst
        }

        #[test]
        fn the_device_gates_match_upstream() {
            let fx = Fx::load();
            // the released lane only: under F32 the products are TF32's (see DeviceGate)
            let e16 = device_gate_errors(&fx, "bf16", DownPrecision::Bf16);
            eprintln!("device gates, worst per-token rel err as released: {e16}");
            assert!(e16 < TOL, "bf16 {e16}");
            // and the precision is still told apart on the device
            assert!(device_gate_errors(&fx, "bf16", DownPrecision::F32) >= 100.0 * TOL);
        }

        /// At the release's shapes (H 2304, rank 64), over a 37-row batch:
        /// the device gate against the host one, row by row, so a kernel that
        /// gets only the last row right (compass cbc76004's k = 64 failure)
        /// is caught at row 0. The weights are bf16 values, as the checkpoint's
        /// are. A bug detector, not a tolerance: on sky (2026-11-23) the worst
        /// row was 1.9e-3, which we read as the k = 4608 sum ordered
        /// differently and a few rank entries landing on the other side of a
        /// bf16 rounding (2^-8 each); the k = 64 failure is off by about 26.
        #[test]
        fn the_device_gate_matches_the_host_gate_at_release_shapes() {
            let (h, r, t) = (2304, 64, 37);
            let mut seed = 0x9e3779b97f4a7c15u64;
            let mut rnd = |scale: f32| -> f32 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            };
            let down: Vec<f32> = (0..r * 2 * h).map(|_| rnd(0.05)).collect();
            let b = |v: f32| half::bf16::from_f32(v).to_f32();
            let forget: Vec<f32> = (0..h * r).map(|_| b(rnd(0.1))).collect();
            let gain: Vec<f32> = (0..h * r).map(|_| b(rnd(0.1))).collect();
            let hs: Vec<f32> = (0..t * h).map(|_| rnd(8.0)).collect();
            let ms: Vec<f32> = (0..t * h).map(|_| rnd(1.0)).collect();
            for precision in [DownPrecision::Bf16] {
                let g = MessageGate::new(down.clone(), forget.clone(), gain.clone(), h, r, 1e-6, precision).unwrap();
                let want = g.apply(&hs, &ms, t);
                let got = host(DeviceGate::new(&g, &dev()).apply(up(hs.clone(), [t, h]), up(ms.clone(), [t, h])));
                let mut worst = 0f32;
                for row in 0..t {
                    let e = rel_err(&got[row * h..(row + 1) * h], &want[row * h..(row + 1) * h]);
                    worst = worst.max(e);
                    assert!(e < 1e-2, "{precision:?} row {row}: rel err {e}");
                }
                eprintln!("{precision:?} device vs host gate, {t} rows, worst rel err {worst}");
            }
        }
        /// Whether a token's gate output depends on how many tokens share its
        /// call: the host gate computes every row alone; a device matmul may
        /// pick another kernel, and so another summation order, for one row
        /// than for many, and the bf16 rounding of the down projection turns a
        /// reordering into whole bf16 steps. Reports, per row of a 37-row
        /// batch, the device row against the same row computed alone, and both
        /// against the host. Diagnostic: prints, asserts only the bug bound.
        #[test]
        fn the_device_gate_row_alone_and_in_a_batch() {
            let (h, r, t) = (2304, 64, 37);
            let mut seed = 0x2545f4914f6cdd1du64;
            let mut rnd = |scale: f32| -> f32 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            };
            let b = |v: f32| half::bf16::from_f32(v).to_f32();
            let down: Vec<f32> = (0..r * 2 * h).map(|_| rnd(0.05)).collect();
            let forget: Vec<f32> = (0..h * r).map(|_| b(rnd(0.1))).collect();
            let gain: Vec<f32> = (0..h * r).map(|_| b(rnd(0.1))).collect();
            let hs: Vec<f32> = (0..t * h).map(|_| rnd(8.0)).collect();
            let ms: Vec<f32> = (0..t * h).map(|_| rnd(1.0)).collect();
            let g = MessageGate::new(down, forget, gain, h, r, 1e-6, DownPrecision::Bf16).unwrap();
            let dg = DeviceGate::new(&g, &dev());
            let host_all = g.apply(&hs, &ms, t);
            let batch = host(dg.apply(up(hs.clone(), [t, h]), up(ms.clone(), [t, h])));
            let (mut alone_vs_batch, mut alone_vs_host, mut batch_vs_host, mut differing) = (0f32, 0f32, 0f32, 0);
            for row in 0..t {
                let rr = row * h..(row + 1) * h;
                let alone = host(dg.apply(up(hs[rr.clone()].to_vec(), [1, h]), up(ms[rr.clone()].to_vec(), [1, h])));
                let e = rel_err(&alone, &batch[rr.clone()]);
                if e > 0.0 {
                    differing += 1;
                }
                alone_vs_batch = alone_vs_batch.max(e);
                alone_vs_host = alone_vs_host.max(rel_err(&alone, &host_all[rr.clone()]));
                batch_vs_host = batch_vs_host.max(rel_err(&batch[rr.clone()], &host_all[rr]));
            }
            eprintln!(
                "gate rows alone vs in a {t}-row batch: {differing} of {t} differ, worst {alone_vs_batch}; \
                 worst vs host: alone {alone_vs_host}, batch {batch_vs_host}"
            );
            assert!(alone_vs_host < 1e-2 && batch_vs_host < 1e-2, "a wrong row: alone {alone_vs_host}, batch {batch_vs_host}");
        }

        /// The numbers behind the release-shape test's 1e-2 bound, on its own
        /// inputs (same seed): per row, the device BF16 gate against the host
        /// BF16 gate (what the test bounds), and the spread between two honest
        /// host gates of the same math, BF16 down projection against F32 (how
        /// far rounding alone moves a row). Metric: `rel_err`, the row's max
        /// abs difference over the reference row's max abs value. Prints only.
        #[test]
        #[ignore = "diagnostic: prints the per-row errors behind the release-shape bound"]
        fn the_release_shape_bound_against_the_rounding_spread() {
            let (h, r, t) = (2304, 64, 37);
            let mut seed = 0x9e3779b97f4a7c15u64;
            let mut rnd = |scale: f32| -> f32 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            };
            let down: Vec<f32> = (0..r * 2 * h).map(|_| rnd(0.05)).collect();
            let b = |v: f32| half::bf16::from_f32(v).to_f32();
            let forget: Vec<f32> = (0..h * r).map(|_| b(rnd(0.1))).collect();
            let gain: Vec<f32> = (0..h * r).map(|_| b(rnd(0.1))).collect();
            let hs: Vec<f32> = (0..t * h).map(|_| rnd(8.0)).collect();
            let ms: Vec<f32> = (0..t * h).map(|_| rnd(1.0)).collect();
            let g16 = MessageGate::new(down.clone(), forget.clone(), gain.clone(), h, r, 1e-6, DownPrecision::Bf16).unwrap();
            let g32 = MessageGate::new(down, forget, gain, h, r, 1e-6, DownPrecision::F32).unwrap();
            let h16 = g16.apply(&hs, &ms, t);
            let h32 = g32.apply(&hs, &ms, t);
            let d16 = host(DeviceGate::new(&g16, &dev()).apply(up(hs.clone(), [t, h]), up(ms.clone(), [t, h])));
            let mut rows = Vec::new();
            for row in 0..t {
                let rr = row * h..(row + 1) * h;
                let dev_host = rel_err(&d16[rr.clone()], &h16[rr.clone()]);
                let spread = rel_err(&h16[rr.clone()], &h32[rr.clone()]);
                let dev_f32 = rel_err(&d16[rr.clone()], &h32[rr]);
                eprintln!("BOUND row {row}: device_vs_host_bf16 {dev_host:.3e} host_bf16_vs_host_f32 {spread:.3e} device_vs_host_f32 {dev_f32:.3e}");
                rows.push((dev_host, spread, dev_f32));
            }
            let worst = |f: fn(&(f32, f32, f32)) -> f32| {
                rows.iter().enumerate().map(|(i, x)| (i, f(x))).fold((0, 0f32), |a, b| if b.1 > a.1 { b } else { a })
            };
            let median = |f: fn(&(f32, f32, f32)) -> f32| {
                let mut v: Vec<f32> = rows.iter().map(f).collect();
                v.sort_by(f32::total_cmp);
                v[v.len() / 2]
            };
            let (a, b2, c) = (worst(|x| x.0), worst(|x| x.1), worst(|x| x.2));
            eprintln!(
                "BOUND summary over {t} rows: device_vs_host_bf16 worst row {} = {:.3e} (median {:.3e}); \
                 host_bf16_vs_host_f32 worst row {} = {:.3e} (median {:.3e}); device_vs_host_f32 worst row {} = {:.3e}",
                a.0, a.1, median(|x| x.0), b2.0, b2.1, median(|x| x.1), c.0, c.1
            );
        }
    }

    #[test]
    fn malformed_shapes_are_refused() {
        assert!(MessageQueries::from_checkpoint(&[0.0; 10], &[0.0; 12], 3, 2).is_err());
        assert!(MessageGate::new(vec![0.0; 7], vec![0.0; 4], vec![0.0; 4], 2, 2, 1e-6, DownPrecision::F32).is_err());
    }
}
