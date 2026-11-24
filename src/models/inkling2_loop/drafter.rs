//! Inkling 2 Mini's drafter (`mtp.safetensors`, DSpark): a block of seven draft
//! tokens from the target's own hidden states, for speculative decoding.
//!
//! From vLLM 1fd30e4f (`models/inkling2/nvidia/dspark.py`, `ops/dspark.py`,
//! `speculator.py`), with the positions as its DFlash preparation sets them:
//!
//! ```text
//! f[0..4]    the target's hidden state at four points, per committed token
//! ctx_i      = rms_norm(sum_j softmax(fuse[i])_j f[j], dflare_norm_i)
//! K_i, V_i   = rope(rms_norm(k_proj_ctx ctx_i)), v_proj_ctx ctx_i       at the token's position
//! x          = [embed_norm(embed[anchor]), mask x 7]                    at P .. P+7, P the anchor's
//! layer i    : x += attn(norm x), x += mlp(norm x); every query sees the
//!              committed K, V (a local layer, the last 1024 before P) and the
//!              whole block, which is not causal
//! h          = norm(x);  base = unembed h[1..8]                         row i drafts position P+i
//! head       : s = 0, prev = anchor; for each draft
//!              g, c, o = rnn_proj [s, w1[prev], h_i, f_rnn]
//!              s = sigmoid(g) s + (1 - sigmoid(g)) tanh(c)
//!              logits = softcap(base_i + unembed (hidden_out tanh(o)));  prev = argmax
//! ```
//!
//! `f_rnn` is the fourth mix, of the last committed token. The stored weights
//! are BF16 and the reference rounds each operator's output to BF16; so does
//! this, except inside the f32 accumulators the reference also keeps.
//! `unembed hidden_out` is folded once at bind.
//!
//! Scope, stated: greedy drafting of one sequence. The drafter's own state is
//! the committed context's keys and values, which only grow: a rejected draft
//! never reaches them, because the block's are per call.

use anyhow::{Result, anyhow, ensure};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::{Int, TensorData};

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::burn::linear_bf16;
use crate::models::inkling::flash::{self, KeyRun, KvElem};
use crate::models::inkling::seam::{Bk, client_of, handle_of, tensor_of};
use crate::models::inkling2_loop::attend::{KEY_TILE, Kv};
use crate::models::inkling2_loop::config::{Inkling2LoopConfig, Inkling2MtpConfig};
use crate::models::inkling2_loop::head;
use crate::models::inkling2_loop::layout::Dtype;
use crate::models::inkling2_loop::pile::MiniRoot;
use crate::models::inkling2_loop::session::MiniModel;
use crate::models::inkling2_loop::sliding::{Dev, bf16_to_f32, rms_norm};

/// The rotary base the release's `mtp_config` states.
const ROPE_THETA: f64 = 1e6;

fn bf16(x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
    super::rows::round_to_bf16(x).expect("the drafter's tensors are f32")
}

fn lin(x: Tensor<Bk, 2>, w: &Bf16W) -> Tensor<Bk, 2> {
    bf16(linear_bf16(x, w))
}

struct Layer {
    local: bool,
    input_norm: Tensor<Bk, 1>,
    post_norm: Tensor<Bk, 1>,
    dflare_norm: Tensor<Bk, 1>,
    q_norm: Tensor<Bk, 1>,
    k_norm: Tensor<Bk, 1>,
    k_norm_ctx: Tensor<Bk, 1>,
    q: Bf16W,
    k: Bf16W,
    v: Bf16W,
    o: Bf16W,
    k_ctx: Bf16W,
    v_ctx: Bf16W,
    gate: Bf16W,
    up: Bf16W,
    down: Bf16W,
}

/// The drafter's weights, bound once beside a [`MiniModel`]'s.
pub struct Drafter {
    cfg: Inkling2MtpConfig,
    eps: f64,
    window: usize,
    cap: f32,
    vocab: usize,
    layers: Vec<Layer>,
    norm: Tensor<Bk, 1>,
    mask: Tensor<Bk, 2>,
    /// One row of fusion weights per layer, then the head's.
    alpha: Vec<Vec<f32>>,
    head_norm: Tensor<Bk, 1>,
    unembed: Bf16W,
    w1: Tensor<Bk, 2>,
    /// `rnn_proj`'s `[state, previous]` and `[hidden, f_rnn]` columns,
    /// transposed, and its bias.
    rnn_step: Tensor<Bk, 2>,
    rnn_ctx: Tensor<Bk, 2>,
    rnn_bias: Tensor<Bk, 2>,
    fold: Bf16W,
    fold_bias: Tensor<Bk, 2>,
    dev: Dev,
}

/// One sequence's drafting context.
pub struct DraftState {
    kv: Vec<Kv>,
    f_rnn: Option<Tensor<Bk, 2>>,
    position: usize,
}

impl Drafter {
    pub fn bind(model: &MiniModel) -> Result<Drafter> {
        let root = model.root();
        let cfg = Inkling2LoopConfig::from_json(&root.config()?.to_string()).map_err(|e| anyhow!("config: {e}"))?;
        let t = &cfg.text_config;
        let m = cfg.mtp_config.clone().ok_or_else(|| anyhow!("the release declares no drafter"))?;
        ensure!(m.markov_head_type == "rnn_hidden", "markov head {:?}", m.markov_head_type);
        ensure!(m.hidden_size == t.hidden_size, "drafter hidden {}", m.hidden_size);
        ensure!(m.target_layer_ids.len() == 4, "taps {:?}", m.target_layer_ids);
        let dev = Dev::default();
        let client = client_of(&Tensor::<Bk, 1>::zeros([1], &dev));
        let (h, qd, kvd) = (m.hidden_size, m.num_attention_heads * m.head_dim, m.num_key_value_heads * m.head_dim);
        let r = m.markov_rank;
        let stored = |name: &str, shape: &[usize]| -> Result<&[u8]> { stored(root, name, shape) };
        let vec1 = |name: &str, n: usize| -> Result<Tensor<Bk, 1>> {
            Ok(Tensor::from_data(TensorData::new(bf16_to_f32(stored(name, &[n])?), [n]), &dev))
        };
        let mat = |name: &str, n: usize, k: usize| -> Result<Vec<f32>> { Ok(bf16_to_f32(stored(name, &[n, k])?)) };
        let w = |name: String, n: usize, k: usize| -> Result<Bf16W> {
            let bytes = stored(&name, &[n, k])?;
            ensure!(Bf16W::tileable(n, k), "{name}: {n}x{k} does not tile");
            Ok(Bf16W { h: client.create_from_slice(bytes), n, k, align: 16 })
        };
        let layers = (0..m.num_hidden_layers)
            .map(|i| -> Result<Layer> {
                let p = |s: &str| format!("layers.{i}.{s}.weight");
                Ok(Layer {
                    local: m.local_layer_ids.contains(&i),
                    input_norm: vec1(&p("input_layernorm"), h)?,
                    post_norm: vec1(&p("post_attention_layernorm"), h)?,
                    dflare_norm: vec1(&p("dflare_norm"), h)?,
                    q_norm: vec1(&p("self_attn.q_norm"), m.head_dim)?,
                    k_norm: vec1(&p("self_attn.k_norm"), m.head_dim)?,
                    k_norm_ctx: vec1(&p("self_attn.k_norm_ctx"), m.head_dim)?,
                    q: w(p("self_attn.q_proj"), qd, h)?,
                    k: w(p("self_attn.k_proj"), kvd, h)?,
                    v: w(p("self_attn.v_proj"), kvd, h)?,
                    o: w(p("self_attn.o_proj"), h, qd)?,
                    k_ctx: w(p("self_attn.k_proj_ctx"), kvd, h)?,
                    v_ctx: w(p("self_attn.v_proj_ctx"), kvd, h)?,
                    gate: w(p("mlp.gate_proj"), m.intermediate_size, h)?,
                    up: w(p("mlp.up_proj"), m.intermediate_size, h)?,
                    down: w(p("mlp.down_proj"), h, m.intermediate_size)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let taps = m.target_layer_ids.len();
        let fuse = mat("fuse_weight", m.num_hidden_layers + 1, taps)?;
        let alpha = fuse
            .chunks_exact(taps)
            .map(|row| {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
                let s: f32 = e.iter().sum();
                e.iter().map(|v| v / s).collect()
            })
            .collect();

        // the target's unembedding, BF16 as stored
        let (dt, shape, unembed) = root.tensor(head::UNEMBED).map_err(|e| anyhow!(e))?;
        let vocab_rows = t.vocab_size;
        ensure!(dt == Dtype::Bf16 && shape == [vocab_rows, h], "unembedding {dt:?} {shape:?}");
        let out_w = Tensor::<Bk, 2>::from_data(TensorData::new(mat("markov_head.hidden_out.weight", h, r)?, [h, r]), &dev);
        let out_b = vec1("markov_head.hidden_out.bias", h)?.reshape([h, 1]);
        let side = Tensor::cat(vec![out_w, out_b], 1);
        let mut folded = Vec::new();
        for rows in unembed.chunks(8192 * h * 2) {
            let n = rows.len() / (h * 2);
            let part = Tensor::<Bk, 2>::from_data(TensorData::new(bf16_to_f32(rows), [n, h]), &dev);
            folded.push(part.matmul(side.clone()));
        }
        let folded: Vec<f32> = Tensor::cat(folded, 0).into_data().convert::<f32>().to_vec().expect("f32");
        let round = |v: f32| half::bf16::from_f32(v);
        let mut fold = Vec::with_capacity(vocab_rows * r * 2);
        let mut fold_bias = Vec::with_capacity(vocab_rows);
        for row in folded.chunks_exact(r + 1) {
            row[..r].iter().for_each(|&v| fold.extend_from_slice(&round(v).to_le_bytes()));
            fold_bias.push(round(row[r]).to_f32());
        }
        ensure!(Bf16W::tileable(vocab_rows, r), "the folded head {vocab_rows}x{r} does not tile");

        let rnn = mat("markov_head.rnn_proj.weight", 3 * r, 2 * r + 2 * h)?;
        let rnn = Tensor::<Bk, 2>::from_data(TensorData::new(rnn, [3 * r, 2 * r + 2 * h]), &dev);
        let cols = |range: std::ops::Range<usize>| rnn.clone().slice([0..3 * r, range]).transpose();
        Ok(Drafter {
            eps: t.rms_norm_eps,
            window: t.sliding_window_size,
            cap: t.final_logit_softcapping as f32,
            vocab: t.unpadded_vocab_size,
            norm: vec1("norm.weight", h)?,
            mask: vec1("mask_embed", h)?.reshape([1, h]),
            alpha,
            head_norm: vec1("markov_head.dflare_norm.weight", h)?,
            unembed: Bf16W { h: client.create_from_slice(unembed), n: vocab_rows, k: h, align: 16 },
            w1: Tensor::from_data(TensorData::new(mat("markov_head.markov_w1.weight", vocab_rows, r)?, [vocab_rows, r]), &dev),
            rnn_step: cols(0..2 * r),
            rnn_ctx: cols(2 * r..2 * r + 2 * h),
            rnn_bias: vec1("markov_head.rnn_proj.bias", 3 * r)?.reshape([1, 3 * r]),
            fold: Bf16W { h: client.create_from_slice(&fold), n: vocab_rows, k: r, align: 16 },
            fold_bias: Tensor::from_data(TensorData::new(fold_bias, [1, vocab_rows]), &dev),
            layers,
            cfg: m,
            dev,
        })
    }

    /// Where the target's hidden states are read, as the release lists them.
    pub fn taps(&self) -> &[usize] {
        &self.cfg.target_layer_ids
    }

    /// Drafts one block holds.
    pub fn depth(&self) -> usize {
        self.cfg.block_size - 1
    }

    pub fn new_state(&self) -> DraftState {
        DraftState { kv: self.layers.iter().map(|_| Kv::new(KEY_TILE)).collect(), f_rnn: None, position: 0 }
    }

    fn mix(&self, f: &[Tensor<Bk, 2>], row: usize, norm: &Tensor<Bk, 1>) -> Tensor<Bk, 2> {
        let mut acc: Option<Tensor<Bk, 2>> = None;
        for (x, &a) in f.iter().zip(&self.alpha[row]) {
            let y = x.clone().mul_scalar(a);
            acc = Some(match acc {
                None => y,
                Some(s) => s + y,
            });
        }
        bf16(rms_norm(acc.expect("at least one tap"), norm.clone(), self.eps))
    }

    /// Rows `[t, heads * d]`, neox halves, at positions `first ..`.
    fn rope(&self, x: Tensor<Bk, 2>, first: usize) -> Tensor<Bk, 2> {
        let [t, w] = x.dims();
        let d = self.cfg.head_dim;
        let (heads, half) = (w / d, d / 2);
        let (mut cos, mut sin) = (Vec::with_capacity(t * half), Vec::with_capacity(t * half));
        for p in first..first + t {
            for i in 0..half {
                let a = p as f64 / ROPE_THETA.powf(2.0 * i as f64 / d as f64);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        let table = |v: Vec<f32>| Tensor::<Bk, 3>::from_data(TensorData::new(v, [t, 1, half]), &self.dev);
        let (cos, sin) = (table(cos), table(sin));
        let x = x.reshape([t, heads, d]);
        let a = x.clone().slice([0..t, 0..heads, 0..half]);
        let b = x.slice([0..t, 0..heads, half..d]);
        let lo = a.clone() * cos.clone() - b.clone() * sin.clone();
        let hi = b * cos + a * sin;
        bf16(Tensor::cat(vec![lo, hi], 2).reshape([t, w]))
    }

    /// Per-head RMS norm of `[t, heads * d]`, rounded.
    fn head_norm(&self, x: Tensor<Bk, 2>, gain: &Tensor<Bk, 1>) -> Tensor<Bk, 2> {
        let [t, w] = x.dims();
        let d = self.cfg.head_dim;
        bf16(rms_norm(x.reshape([t * w / d, d]), gain.clone(), self.eps)).reshape([t, w])
    }

    /// Take the committed tokens' hidden states (each `[t, hidden]`, positions
    /// `state.position ..`, only their first `keep` rows) into the context.
    pub fn ingest(&self, state: &mut DraftState, taps: &[Tensor<Bk, 2>], keep: usize) {
        let h = self.cfg.hidden_size;
        let f: Vec<Tensor<Bk, 2>> = taps.iter().map(|x| bf16(x.clone().slice([0..keep, 0..h]))).collect();
        for (i, l) in self.layers.iter().enumerate() {
            let ctx = self.mix(&f, i, &l.dflare_norm);
            let k = self.rope(self.head_norm(lin(ctx.clone(), &l.k_ctx), &l.k_norm_ctx), state.position);
            let v = lin(ctx, &l.v_ctx);
            state.kv[i].append(k, v);
        }
        state.position += keep;
        for (kv, l) in state.kv.iter_mut().zip(&self.layers) {
            if l.local {
                kv.evict_before(state.position.saturating_sub(self.window));
            }
        }
        let last: Vec<Tensor<Bk, 2>> = f.iter().map(|x| x.clone().slice([keep - 1..keep, 0..h])).collect();
        state.f_rnn = Some(self.mix(&last, self.layers.len(), &self.head_norm));
    }

    fn attention(&self, l: &Layer, kv: &Kv, x: Tensor<Bk, 2>, first: usize) -> Tensor<Bk, 2> {
        let c = &self.cfg;
        let [t, _] = x.dims();
        let (heads, kvh, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let q = self.rope(self.head_norm(lin(x.clone(), &l.q), &l.q_norm), first);
        let k = self.rope(self.head_norm(lin(x.clone(), &l.k), &l.k_norm), first);
        let v = lin(x, &l.v);
        let client = client_of(&q);
        let oldest = if l.local { first.saturating_sub(self.window) } else { 0 };
        let mut held: Vec<_> = kv
            .whole_segments()
            .filter(|(base, _, _, filled)| base + filled > oldest)
            .map(|(base, k, v, filled)| {
                let [tile, _] = k.dims();
                (handle_of(k.clone()), handle_of(v.clone()), tile, base, oldest.saturating_sub(base), filled)
            })
            .collect();
        held.push((handle_of(k), handle_of(v), t, first, 0, t));
        let runs: Vec<KeyRun<'_>> = held
            .iter()
            .map(|(k, v, rows, base, lo, hi)| KeyRun {
                k,
                v,
                k_scales: None,
                v_scales: None,
                rows: *rows,
                base: *base,
                lo: *lo,
                hi: *hi,
                row0: 0,
            })
            .collect();
        // The block is not causal and has no relative bias: the kernel is told
        // its queries start at the block's last position, so every key is in
        // every query's past, the window is the runs' `lo`, and the bias is zero.
        let zero = handle_of(Tensor::<Bk, 2>::zeros([t * heads, 1], &self.dev));
        let rows = flash::decode_rows(heads / kvh);
        let scale = (hd as f32).powf(-0.5);
        let out = flash::flash_attention_launch(
            &client, &handle_of(q), &runs, &zero, KvElem::F32, t, first + t - 1, heads, kvh, hd, 1, None, scale, rows, None,
        );
        lin(bf16(tensor_of(client, self.dev.clone(), out, t, heads * hd)), &l.o)
    }

    /// `n` (at most [`Self::depth`]) greedy drafts after `anchor`, the token at
    /// the context's end that the target has not seen yet.
    pub fn draft(&self, model: &MiniModel, state: &DraftState, anchor: u32, n: usize) -> Result<Vec<u32>> {
        ensure!((1..=self.depth()).contains(&n), "{n} drafts of a block of {}", self.depth());
        let f_rnn = state.f_rnn.clone().ok_or_else(|| anyhow!("no context to draft from"))?;
        let (h, r, b) = (self.cfg.hidden_size, self.cfg.markov_rank, self.cfg.block_size);
        let embedded = bf16(model.embed(&[anchor]));
        let mut residual = Tensor::cat(vec![embedded, self.mask.clone().repeat_dim(0, b - 1)], 0);
        let mut x = bf16(rms_norm(residual.clone(), self.layers[0].input_norm.clone(), self.eps));
        for (i, l) in self.layers.iter().enumerate() {
            let sum = residual + self.attention(l, &state.kv[i], x, state.position);
            residual = bf16(sum.clone());
            let y = bf16(rms_norm(sum, l.post_norm.clone(), self.eps));
            let gated = bf16(silu(lin(y.clone(), &l.gate)) * lin(y, &l.up));
            let sum = residual + lin(gated, &l.down);
            residual = bf16(sum.clone());
            let norm = self.layers.get(i + 1).map_or(&self.norm, |next| &next.input_norm);
            x = bf16(rms_norm(sum, norm.clone(), self.eps));
        }
        let hidden = x.slice([1..1 + n, 0..h]);
        let base = lin(hidden.clone(), &self.unembed);
        let context = Tensor::cat(vec![hidden, f_rnn.repeat_dim(0, n)], 1).matmul(self.rnn_ctx.clone()) + self.rnn_bias.clone();

        let mut s = Tensor::<Bk, 2>::zeros([1, r], &self.dev);
        let mut prev = Tensor::<Bk, 1, Int>::from_data(TensorData::new(vec![anchor as i32], [1]), &self.dev);
        let mut drafts = Vec::with_capacity(n);
        for i in 0..n {
            let input = Tensor::cat(vec![s.clone(), self.w1.clone().select(0, prev)], 1);
            let raw = bf16(context.clone().slice([i..i + 1, 0..3 * r]) + input.matmul(self.rnn_step.clone()));
            let part = |j: usize| raw.clone().slice([0..1, j * r..(j + 1) * r]);
            let g = bf16(sigmoid(part(0)));
            let c = bf16(part(1).tanh());
            let latent = bf16(part(2).tanh());
            let keep = bf16(g.clone() * s);
            let update = bf16(bf16(g.neg().add_scalar(1.0)) * c);
            s = bf16(keep + update);
            let bias = bf16(linear_bf16(latent, &self.fold) + self.fold_bias.clone());
            let row = base.clone().slice([i..i + 1, 0..self.unembed.n]);
            let logits = bf16(row + bias).slice([0..1, 0..self.vocab]);
            let logits = bf16((logits / self.cap).tanh() * self.cap);
            prev = logits.argmax(1).reshape([1]);
            drafts.push(prev.clone());
        }
        let ids: Vec<i32> = Tensor::cat(drafts, 0).into_data().convert::<i32>().to_vec().expect("i32");
        Ok(ids.into_iter().map(|v| v as u32).collect())
    }
}

fn stored<'a>(root: &'a MiniRoot, name: &str, shape: &[usize]) -> Result<&'a [u8]> {
    let (dt, got, bytes) = root.tensor(name).map_err(|e| anyhow!(e))?;
    ensure!(dt == Dtype::Bf16 && got == shape, "{name}: stored {dt:?} {got:?}, the drafter reads BF16 {shape:?}");
    Ok(bytes)
}
