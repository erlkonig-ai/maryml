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

    #[test]
    fn malformed_shapes_are_refused() {
        assert!(MessageQueries::from_checkpoint(&[0.0; 10], &[0.0; 12], 3, 2).is_err());
        assert!(MessageGate::new(vec![0.0; 7], vec![0.0; 4], vec![0.0; 4], 2, 2, 1e-6, DownPrecision::F32).is_err());
    }
}
