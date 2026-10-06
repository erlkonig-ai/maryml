//! Transformer primitives for the Voxtral port — parity-first (plain q/k/v
//! projections, explicit RMSNorm weights, activation-side rotate_half RoPE).
//! The op-count folds proven on the qwen3tts talker (wide fused qkv, norm
//! weights into matmul rows, pre-rotated weights) are a later optimization
//! pass, gated exact against this reference layout.
//!
//! Differences from the qwen3tts layer zoo that make this its own module:
//! biased projections (the encoder has q/v/o + mlp-down biases — and RoPE
//! applies AFTER the bias add), no q/k-norm anywhere, and the decoder's
//! ada-RMS-norm delay conditioning (a per-channel scale between the
//! post-attention norm and the MLP).

use burn::prelude::*;
use burn::tensor::FloatDType;
use burn::tensor::activation::{silu, softmax};

use crate::nn::weight_loader::WeightLoader;

/// Bias-optional Linear, stored ready-to-matmul: `[1, in, out]`.
pub struct Linear<B: Backend> {
    pub weight_t: Tensor<B, 3>,     // [1, in, out]
    pub bias: Option<Tensor<B, 3>>, // [1, 1, out]
}

impl<B: Backend> Linear<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, bias: bool, device: &B::Device) -> Self {
        let w: Tensor<B, 2> = loader.load_tensor(&format!("{prefix}.weight"), device); // [out, in]
        let [o, i] = w.dims();
        Self {
            weight_t: w.transpose().reshape([1, i, o]),
            bias: bias.then(|| {
                let b: Tensor<B, 1> = loader.load_tensor(&format!("{prefix}.bias"), device);
                b.reshape([1, 1, o])
            }),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = x.matmul(self.weight_t.clone());
        match &self.bias {
            Some(bias) => y + bias.clone(),
            None => y,
        }
    }
}

/// RMSNorm over the last dim: `x · rsqrt(mean(x²)+eps) · weight`. The variance
/// chain runs in f32 and casts back (activation outliers overflow f16).
pub struct RmsNorm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn load(loader: &WeightLoader, name: &str, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: loader.load_tensor(name, device),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let d = self.weight.dims()[0];
        let dt = x.dtype();
        let x32 = x.cast(FloatDType::F32);
        let var = x32.clone().powf_scalar(2.0).mean_dim(2);
        x32.mul(var.add_scalar(self.eps).sqrt().recip())
            .cast(dt)
            .mul(self.weight.clone().reshape([1, 1, d]))
    }
}

/// Embedding table `[vocab, dim]`; lookup by u32 ids → `[1, L, dim]`.
pub struct Embedding<B: Backend> {
    pub weight: Tensor<B, 2>,
}

impl<B: Backend> Embedding<B> {
    pub fn load(loader: &WeightLoader, name: &str, device: &B::Device) -> Self {
        Self {
            weight: loader.load_tensor(name, device),
        }
    }

    pub fn forward(&self, ids: &[u32], device: &B::Device) -> Tensor<B, 3> {
        let idx: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let idx = Tensor::<B, 1, Int>::from_ints(idx.as_slice(), device);
        let (l, d) = (ids.len(), self.weight.dims()[1]);
        self.weight.clone().select(0, idx).reshape([1, l, d])
    }
}

/// Precomputed NeoX-style RoPE tables: `[max_len, D]` with half-tables
/// duplicated so `rope(x) = x·cos + rotate_half(x)·sin` broadcasts directly.
pub struct RopeTable<B: Backend> {
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
    d: usize,
}

impl<B: Backend> RopeTable<B> {
    pub fn new(theta: f64, head_dim: usize, max_len: usize, device: &B::Device) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0f32; max_len * head_dim];
        let mut sin = vec![0f32; max_len * head_dim];
        for p in 0..max_len {
            for i in 0..half {
                let r = p as f64 * theta.powf(-2.0 * i as f64 / head_dim as f64);
                cos[p * head_dim + i] = r.cos() as f32;
                cos[p * head_dim + half + i] = r.cos() as f32;
                sin[p * head_dim + i] = r.sin() as f32;
                sin[p * head_dim + half + i] = r.sin() as f32;
            }
        }
        Self {
            cos: Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([max_len, head_dim]),
            sin: Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([max_len, head_dim]),
            d: head_dim,
        }
    }

    /// cos/sin slices for query positions `offset..offset+l`, `[1,1,l,D]`.
    pub fn slices(&self, offset: usize, l: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        (
            self.cos
                .clone()
                .narrow(0, offset, l)
                .reshape([1, 1, l, self.d]),
            self.sin
                .clone()
                .narrow(0, offset, l)
                .reshape([1, 1, l, self.d]),
        )
    }
}

/// `rotate_half` on the last dim of `[B,H,L,D]`: `[-x2 ‖ x1]`.
pub(super) fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let d = x.dims()[3];
    let half = d / 2;
    Tensor::cat(
        vec![x.clone().narrow(3, half, half).neg(), x.narrow(3, 0, half)],
        3,
    )
}

/// Growing KV cache: `[B, Hkv, L, D]` per tensor.
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[2])
    }

    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (fk, fv) = match (self.k.take(), self.v.take()) {
            (Some(pk), Some(pv)) => (Tensor::cat(vec![pk, k], 2), Tensor::cat(vec![pv, v], 2)),
            _ => (k, v),
        };
        self.k = Some(fk.clone());
        self.v = Some(fv.clone());
        (fk, fv)
    }
}

/// Geometry + behavior knobs for one attention stack.
#[derive(Clone, Copy)]
pub struct AttnConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// q/v/o projections carry biases (encoder); k never does.
    pub qvo_bias: bool,
    /// Sliding-window size (encoder 750, decoder 8192).
    pub window: usize,
}

/// Plain multi-head / grouped-query attention with RoPE-after-bias, KV cache,
/// and causal sliding-window masking.
pub struct Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    cfg: AttnConfig,
}

impl<B: Backend> Attention<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, cfg: AttnConfig, device: &B::Device) -> Self {
        Self {
            q_proj: Linear::load(loader, &format!("{prefix}.q_proj"), cfg.qvo_bias, device),
            k_proj: Linear::load(loader, &format!("{prefix}.k_proj"), false, device),
            v_proj: Linear::load(loader, &format!("{prefix}.v_proj"), cfg.qvo_bias, device),
            o_proj: Linear::load(loader, &format!("{prefix}.o_proj"), cfg.qvo_bias, device),
            cfg,
        }
    }

    /// `x: [B, L, hidden]` (already normed); `cos`/`sin`: `[1,1,L,D]` slices
    /// for query positions `offset..offset+L` with `offset = cache.seq_len()`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        cache: &mut KvCache<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, l, _] = x.dims();
        let (h, hkv, d) = (self.cfg.heads, self.cfg.kv_heads, self.cfg.head_dim);
        let offset = cache.seq_len();

        let heads =
            |t: Tensor<B, 3>, n: usize| -> Tensor<B, 4> { t.reshape([b, l, n, d]).swap_dims(1, 2) };
        let q = heads(self.q_proj.forward(x.clone()), h);
        let k = heads(self.k_proj.forward(x.clone()), hkv);
        let v = heads(self.v_proj.forward(x), hkv);

        // RoPE (after bias — the encoder's q/v are biased projections)
        let q = q.clone().mul(cos.clone()) + rotate_half(q).mul(sin.clone());
        let k = k.clone().mul(cos.clone()) + rotate_half(k).mul(sin.clone());

        let (k, v) = cache.update(k, v);
        let lk = k.dims()[2];
        let groups = h / hkv;

        // GQA: expand kv heads (groups == 1 for the encoder's MHA)
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

        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((d as f64).powf(-0.5)); // [B,H,L,Lk]

        // causal sliding window: query position qp attends key j iff
        // j <= qp && qp - j < window. Mask built host-side; skipped when the
        // window can't bite (single query attending only its past).
        let need_mask = l > 1 || offset + 1 > self.cfg.window;
        let scores = if need_mask {
            let w = self.cfg.window;
            let mut blocked = vec![false; l * lk];
            for i in 0..l {
                let qp = offset + i;
                for j in 0..lk {
                    blocked[i * lk + j] = j > qp || qp - j >= w;
                }
            }
            let mask = Tensor::<B, 2, Bool>::from_data(
                burn::tensor::TensorData::new(blocked, [l, lk]),
                device,
            )
            .reshape([1, 1, l, lk])
            .expand([b, h, l, lk]);
            scores.mask_fill(mask, f32::MIN)
        } else {
            scores
        };

        let probs = softmax(scores, 3);
        let out = probs.matmul(v).swap_dims(1, 2).reshape([b, l, h * d]);
        self.o_proj.forward(out)
    }
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`; `down` optionally biased
/// (encoder yes, decoder no).
pub struct Mlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, down_bias: bool, device: &B::Device) -> Self {
        Self {
            gate_proj: Linear::load(loader, &format!("{prefix}.gate_proj"), false, device),
            up_proj: Linear::load(loader, &format!("{prefix}.up_proj"), false, device),
            down_proj: Linear::load(loader, &format!("{prefix}.down_proj"), down_bias, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down_proj
            .forward(silu(self.gate_proj.forward(x.clone())).mul(self.up_proj.forward(x)))
    }
}
