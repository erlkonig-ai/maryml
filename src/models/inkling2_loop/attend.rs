//! Inkling 2 Mini's attention, sliding and full, in memory that does not grow
//! with the prompt: keys and values held in fixed segments, scores computed one
//! (query tile, key segment) pair at a time with an online softmax.
//!
//! What it computes is what the layers computed before, a materialised softmax
//! over `[heads, t, n]` (transformers 046bdb8c `Inkling2LoopSlidingAttention`
//! and `Inkling2LoopFullAttention`). For head `h` and the query at absolute
//! position `p`, over the visible keys `j`:
//!
//! ```text
//! s_j = q . k_j / sqrt(head_dim) + bias_h(p - j)
//! out = sum_j exp(s_j) v_j / (sum_j exp(s_j) + exp(sink_h))
//! ```
//!
//! A key is visible when `0 <= p - j` and, on a sliding layer, `p - j <
//! window`. `bias_h(d)` is the query's relative logit at distance `d` when `d <
//! rel_extent` and zero beyond. GQA: heads `g * rep .. (g + 1) * rep` read KV
//! head `g`. The log scale of a full layer is already in `q` and in the
//! relative states the caller passes.
//!
//! The online softmax starts each query row at `m = sink`, `l = 1` (the sink's
//! own term) and `o = 0`, and folds each key segment in: `m' = max(m,
//! max_j s_j)`, `l = l exp(m - m') + sum_j exp(s_j - m')`, `o = o exp(m - m') +
//! sum_j exp(s_j - m') v_j`. So the sink stays in the denominator, and a row
//! that sees no key in a segment (every `s_j` is `-inf`) leaves `m`, `l` and
//! `o` exactly as they were. The answer is the materialised softmax's up to the
//! order of the sums.
//!
//! Distances, the causal and window mask and the bias indices are built on the
//! device from two `arange`s, and only for a segment that needs them: a segment
//! wholly in the past and inside the window gets no mask, one wholly beyond
//! `rel_extent` no bias, one wholly in the future or wholly outside the window
//! is skipped.
//!
//! Memory. [`Kv`] holds a visit's keys and values in segments of
//! [`KEY_TILE`] rows, each allocated whole and filled in place, so a decode
//! step writes one row and allocates nothing. A sliding visit drops a segment
//! once no later query can see any of its keys ([`Kv::evict_before`]); a full
//! visit keeps every segment. The largest transient is a handful of `[heads,
//! QUERY_TILE, KEY_TILE]` tensors (scores, distances, bias indices, masks,
//! probabilities), never one as wide as the prompt.

use burn::prelude::*;
use burn::tensor::{DType, Int};
use cubecl::server::Handle;

use crate::models::inkling::flash::{self, KeyRun, KvElem};
use crate::models::inkling::seam::{Bk, client_of, handle_of, handle_of_any, tensor_of};
use crate::models::inkling2_loop::attend_mma;

/// Rows of one key/value segment, and so of one key tile.
pub const KEY_TILE: usize = 1024;

/// Queries scored together against one key segment.
pub const QUERY_TILE: usize = 1024;

/// Calls of at most this many queries (a decode step, a short draft) take the
/// fused flash-decoding kernel ([`crate::models::inkling::flash`]): one launch
/// per key segment and one to merge. Longer calls take the tensor-core kernel
/// ([`super::attend_mma`]) where it covers the shape, and otherwise the tiled
/// loop below, which launches a chain of about twenty per segment.
pub const FUSED_MAX_QUERIES: usize = 16;

/// One visit's keys and values: post-clip K and post-norm V `[rows, kv_dim]`
/// in BF16 (upstream's clip and norm hand back the layer's dtype, and its
/// cache holds that), in segments of `tile` rows, oldest first, from absolute position `base`.
pub struct Kv {
    tile: usize,
    base: usize,
    /// Every segment but the last is full.
    segs: Vec<[Tensor<Bk, 2>; 2]>,
    /// Rows filled in the last segment.
    last: usize,
}

impl Kv {
    pub fn new(tile: usize) -> Kv {
        assert!(tile > 0, "a segment holds at least one row");
        Kv { tile, base: 0, segs: Vec::new(), last: 0 }
    }

    /// Absolute position of the first row held.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Rows held, dropped segments not counted.
    pub fn rows(&self) -> usize {
        match self.segs.len() {
            0 => 0,
            s => (s - 1) * self.tile + self.last,
        }
    }

    /// One past the absolute position of the last row held.
    pub fn end(&self) -> usize {
        self.base + self.rows()
    }

    /// Segments held, and the rows they are allocated for (full segments,
    /// whatever is filled).
    pub fn allocated_rows(&self) -> usize {
        self.segs.len() * self.tile
    }

    /// Append `k` and `v` (`[t, kv_dim]`) at positions `end() ..`, rounded to
    /// BF16.
    pub fn append(&mut self, k: Tensor<Bk, 2>, v: Tensor<Bk, 2>) {
        let (k, v) = (k.cast(DType::BF16), v.cast(DType::BF16));
        let [t, c] = k.dims();
        assert_eq!(v.dims(), [t, c], "keys and values are the same shape");
        let mut i = 0;
        while i < t {
            if self.segs.is_empty() || self.last == self.tile {
                let opts = (&k.device(), DType::BF16);
                self.segs.push([Tensor::zeros([self.tile, c], opts), Tensor::zeros([self.tile, c], opts)]);
                self.last = 0;
            }
            let take = (self.tile - self.last).min(t - i);
            let rows = self.last..self.last + take;
            let [sk, sv] = self.segs.pop().expect("a segment to fill");
            let sk = sk.slice_assign([rows.clone(), 0..c], k.clone().slice([i..i + take, 0..c]));
            let sv = sv.slice_assign([rows, 0..c], v.clone().slice([i..i + take, 0..c]));
            self.segs.push([sk, sv]);
            self.last += take;
            i += take;
        }
    }

    /// Drop every whole segment whose rows all sit before absolute position
    /// `first`. The segment holding `first`, and the one being filled, stay.
    pub fn evict_before(&mut self, first: usize) {
        while self.segs.len() > 1 && self.base + self.tile <= first {
            self.segs.remove(0);
            self.base += self.tile;
        }
    }

    /// `(absolute position of row 0, keys, values)` of each segment, cut to
    /// its filled rows.
    pub fn segments(&self) -> impl Iterator<Item = (usize, Tensor<Bk, 2>, Tensor<Bk, 2>)> + '_ {
        let n = self.segs.len();
        self.segs.iter().enumerate().map(move |(s, [k, v])| {
            let rows = if s + 1 == n { self.last } else { self.tile };
            let [_, c] = k.dims();
            (self.base + s * self.tile, k.clone().slice([0..rows, 0..c]), v.clone().slice([0..rows, 0..c]))
        })
    }

    /// `(absolute position of row 0, keys, values, filled rows)` of each
    /// segment, whole: the buffers as allocated, `tile` rows each.
    fn whole_segments(&self) -> impl Iterator<Item = (usize, &Tensor<Bk, 2>, &Tensor<Bk, 2>, usize)> + '_ {
        let n = self.segs.len();
        self.segs.iter().enumerate().map(move |(s, [k, v])| {
            let rows = if s + 1 == n { self.last } else { self.tile };
            (self.base + s * self.tile, k, v, rows)
        })
    }

    fn joined(&self, which: usize) -> Option<Tensor<Bk, 2>> {
        let parts: Vec<Tensor<Bk, 2>> = self.segments().map(|(_, k, v)| if which == 0 { k } else { v }).collect();
        (!parts.is_empty()).then(|| Tensor::cat(parts, 0).cast(DType::F32))
    }

    /// Every key held, `[rows, kv_dim]`, as one f32 tensor: for the comparison
    /// binaries, which line a short cache up with upstream's.
    pub fn keys(&self) -> Option<Tensor<Bk, 2>> {
        self.joined(0)
    }

    /// Every value held, as [`Self::keys`].
    pub fn values(&self) -> Option<Tensor<Bk, 2>> {
        self.joined(1)
    }
}

/// The fixed shape of one attention.
#[derive(Debug, Clone, Copy)]
pub struct AttnShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rel_extent: usize,
    /// `Some(w)` on a sliding layer: a key `w` or more positions back is not
    /// visible.
    pub window: Option<usize>,
}

/// Attention of `t` queries at absolute positions `first ..` over every key in
/// `kv`, which already holds the call's own rows.
///
/// `q` is `[t, heads, head_dim]`, `rel` `[t * heads, d_rel]` (both clipped and,
/// on a full layer, log-scaled), `rel_proj` `[d_rel, rel_extent]`, `sinks`
/// `[t, heads]`. Returns `[t, heads * head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attend(
    a: &AttnShape,
    q: Tensor<Bk, 3>,
    rel: Tensor<Bk, 2>,
    rel_proj: &Tensor<Bk, 2>,
    sinks: Tensor<Bk, 2>,
    kv: &Kv,
    first: usize,
    query_tile: usize,
) -> Tensor<Bk, 2> {
    let [t, heads, hd] = q.dims();
    let [_, d_rel] = rel.dims();
    assert_eq!(heads, a.heads, "q carries the shape's heads");
    assert_eq!(kv.end(), first + t, "the cache holds exactly the keys up to the call's last query");
    if t > FUSED_MAX_QUERIES && attend_mma::applies(heads, a.kv_heads, hd) {
        return attend_blocks(a, q, rel, rel_proj, sinks, kv, first, query_tile);
    }
    if t <= FUSED_MAX_QUERIES {
        let rows = flash::decode_rows(heads / a.kv_heads);
        if flash::applies(heads, a.kv_heads, hd, rows) && kv.rows() > 0 {
            return attend_fused(a, q, rel, rel_proj, sinks, kv, first, rows);
        }
    }
    let (kvh, e) = (a.kv_heads, a.rel_extent);
    let rep = heads / kvh;
    let dev = q.device();
    let scale = (hd as f32).powf(-0.5);
    let mut outs = Vec::with_capacity(t.div_ceil(query_tile));
    let mut i0 = 0;
    while i0 < t {
        let tq = query_tile.min(t - i0);
        let p0 = first + i0;
        // the layouts and the swap-then-reshape chains are the materialised
        // attention's, applied to a slice of rows
        let qt = q.clone().slice([i0..i0 + tq, 0..heads, 0..hd]).swap_dims(0, 1).reshape([kvh, rep * tq, hd]);
        let relt = rel
            .clone()
            .slice([i0 * heads..(i0 + tq) * heads, 0..d_rel])
            .matmul(rel_proj.clone())
            .reshape([tq, heads, e])
            .swap_dims(0, 1);
        // `[heads, tq, 1]`, never reshaped: on this backend a reshape after a
        // swap with a unit dimension has read zeros (the sliding layer's note)
        let mut m = sinks.clone().slice([i0..i0 + tq, 0..heads]).reshape([tq, heads, 1]).swap_dims(0, 1);
        let mut l = m.ones_like();
        let mut o: Tensor<Bk, 3> = Tensor::zeros([heads, tq, hd], &dev);
        for (k0, k, v) in kv.segments() {
            let [tk, _] = k.dims();
            // the nearest and the farthest distance any (query, key) pair here has
            let near = p0 as i64 - (k0 + tk - 1) as i64;
            let far = (p0 + tq - 1) as i64 - k0 as i64;
            if far < 0 {
                break;
            }
            if a.window.is_some_and(|w| near >= w as i64) {
                continue;
            }
            let (k, v) = (k.cast(DType::F32), v.cast(DType::F32));
            let kt = k.reshape([tk, kvh, hd]).swap_dims(0, 1).swap_dims(1, 2);
            let mut s = qt.clone().matmul(kt).mul_scalar(scale).reshape([heads, tq, tk]);
            let need_mask = near < 0 || a.window.is_some_and(|w| far >= w as i64);
            let need_bias = near < e as i64;
            if need_mask || need_bias {
                let qp = Tensor::<Bk, 1, Int>::arange(p0 as i64..(p0 + tq) as i64, &dev).reshape([tq, 1]);
                let kp = Tensor::<Bk, 1, Int>::arange(k0 as i64..(k0 + tk) as i64, &dev).reshape([1, tk]);
                let dist = (qp - kp).unsqueeze_dim::<3>(0).repeat_dim(0, heads);
                if need_bias {
                    let idx = dist.clone().clamp(0, e as i64 - 1);
                    let beyond = dist.clone().greater_equal_elem(e as i64);
                    s = s + relt.clone().gather(2, idx).mask_fill(beyond, 0.0);
                }
                if need_mask {
                    let mut hidden = dist.clone().lower_elem(0);
                    if let Some(w) = a.window {
                        hidden = hidden.bool_or(dist.greater_equal_elem(w as i64));
                    }
                    s = s.mask_fill(hidden, f32::NEG_INFINITY);
                }
            }
            let m_new = m.clone().max_pair(s.clone().max_dim(2));
            let alpha = (m - m_new.clone()).exp();
            let p = (s - m_new.clone()).exp();
            l = l * alpha.clone() + p.clone().sum_dim(2);
            let vv = v.reshape([tk, kvh, hd]).swap_dims(0, 1);
            let pv = p.reshape([kvh, rep * tq, tk]).matmul(vv).reshape([heads, tq, hd]);
            o = o * alpha + pv;
            m = m_new;
        }
        outs.push((o / l).swap_dims(0, 1).reshape([tq, heads * hd]));
        i0 += tq;
    }
    if outs.len() == 1 { outs.pop().expect("one tile") } else { Tensor::cat(outs, 0) }
}

/// [`attend`] on tensor cores ([`attend_mma`]), a block of `query_tile` queries
/// at a time: the block's relative-bias table, then one launch per segment it
/// can see, the running softmax carried between them.
#[allow(clippy::too_many_arguments)]
fn attend_blocks(
    a: &AttnShape,
    q: Tensor<Bk, 3>,
    rel: Tensor<Bk, 2>,
    rel_proj: &Tensor<Bk, 2>,
    sinks: Tensor<Bk, 2>,
    kv: &Kv,
    first: usize,
    query_tile: usize,
) -> Tensor<Bk, 2> {
    let [t, heads, hd] = q.dims();
    let [_, d_rel] = rel.dims();
    let dev = q.device();
    let client = client_of(&q);
    let scale = (hd as f32).powf(-0.5);
    let q = q.reshape([t, heads * hd]).cast(DType::BF16);
    let held = held_segments(kv);
    let mut outs = Vec::with_capacity(t.div_ceil(query_tile));
    let mut i0 = 0;
    while i0 < t {
        let tq = query_tile.min(t - i0);
        let p0 = first + i0;
        let runs = visible_runs(&held, a.window, p0, tq);
        let table = rel.clone().slice([i0 * heads..(i0 + tq) * heads, 0..d_rel]).matmul(rel_proj.clone());
        let (q_h, _) = handle_of_any(q.clone().slice([i0..i0 + tq, 0..heads * hd]));
        let rel_h = handle_of(table);
        let sink_h = handle_of(sinks.clone().slice([i0..i0 + tq, 0..heads]));
        let out = attend_mma::attend_mma_launch(
            &client, &q_h, &runs, &rel_h, &sink_h, tq, p0, heads, a.kv_heads, hd, a.rel_extent, a.window, scale,
        );
        outs.push(tensor_of(client.clone(), dev.clone(), out, tq, heads * hd));
        i0 += tq;
    }
    if outs.len() == 1 { outs.pop().expect("one block") } else { Tensor::cat(outs, 0) }
}

/// One held segment's handles: keys, values, allocated rows, absolute position
/// of row 0, filled rows.
struct Held {
    k: Handle,
    v: Handle,
    rows: usize,
    base: usize,
    filled: usize,
}

fn held_segments(kv: &Kv) -> Vec<Held> {
    kv.whole_segments()
        .map(|(base, k, v, filled)| {
            let [rows, _] = k.dims();
            let (k, dt) = handle_of_any(k.clone());
            let (v, _) = handle_of_any(v.clone());
            assert_eq!(dt, DType::BF16, "the cache holds BF16");
            Held { k, v, rows, base, filled }
        })
        .collect()
}

/// The segments with a key some query of `p0 .. p0 + tq` can see, as key runs.
fn visible_runs(held: &[Held], window: Option<usize>, p0: usize, tq: usize) -> Vec<KeyRun<'_>> {
    held.iter()
        .filter(|s| s.base < p0 + tq && window.is_none_or(|w| s.base + s.filled + w > p0 + 1))
        .map(|s| KeyRun {
            k: &s.k,
            v: &s.v,
            k_scales: None,
            v_scales: None,
            rows: s.rows,
            base: s.base,
            lo: 0,
            hi: s.filled,
            row0: 0,
        })
        .collect()
}

/// [`attend`] for a short call, through the fused kernel: the relative-bias
/// table `[t, heads, rel_extent]` is built here (the same product the tiled
/// loop takes per query tile), and every held segment a query can see is one
/// key run whose live rows are its filled ones. Causality, the window and the
/// distance bias are the kernel's, per (query, key) cell; the sink enters each
/// row's denominator in the merge.
#[allow(clippy::too_many_arguments)]
fn attend_fused(
    a: &AttnShape,
    q: Tensor<Bk, 3>,
    rel: Tensor<Bk, 2>,
    rel_proj: &Tensor<Bk, 2>,
    sinks: Tensor<Bk, 2>,
    kv: &Kv,
    first: usize,
    rows: usize,
) -> Tensor<Bk, 2> {
    let [t, heads, hd] = q.dims();
    let dev = q.device();
    let client = client_of(&q);
    let table = rel.matmul(rel_proj.clone());
    let q_h = handle_of(q.reshape([t, heads * hd]));
    let rel_h = handle_of(table);
    let sink_h = handle_of(sinks);
    let held = held_segments(kv);
    let runs = visible_runs(&held, a.window, first, t);
    let scale = (hd as f32).powf(-0.5);
    let out = flash::flash_attention_launch_sinked(
        &client, &q_h, &runs, &rel_h, &sink_h, KvElem::Bf16, t, first, heads, a.kv_heads, hd, a.rel_extent, a.window, scale,
        rows,
    );
    tensor_of(client, dev, out, t, heads * hd)
}

#[cfg(test)]
mod tests {
    //! [`attend`] against the materialised softmax the layers ran before (the
    //! code of mary d17aa9d `SlidingLayer::attention` / `FullLayer::attention`
    //! from the projections on), on the device, with random inputs.

    use super::*;
    use burn::tensor::activation::softmax;
    use burn::tensor::{Bool, TensorData};

    type Dev = burn::backend::cuda::CudaDevice;

    struct Lcg(u64);
    impl Lcg {
        /// Roughly standard normal: a sum of four uniforms, centred and scaled.
        fn normal(&mut self) -> f32 {
            let mut s = 0.0;
            for _ in 0..4 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s += (self.0 >> 40) as f32 / (1u64 << 24) as f32;
            }
            (s - 2.0) * 3f32.sqrt()
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.normal() * scale).collect()
        }
    }

    fn up<const D: usize>(v: Vec<f32>, shape: [usize; D], dev: &Dev) -> Tensor<Bk, D> {
        Tensor::from_data(TensorData::new(v, shape), dev)
    }

    fn host<const D: usize>(t: Tensor<Bk, D>) -> Vec<f32> {
        t.into_data().convert::<f32>().to_vec().expect("f32")
    }

    /// The materialised attention of d17aa9d, generalised only in taking the
    /// window as an option: queries at `first ..`, keys at `0 .. n`.
    #[allow(clippy::too_many_arguments)]
    fn materialised(
        a: &AttnShape,
        q: Tensor<Bk, 3>,
        rel: Tensor<Bk, 2>,
        rel_proj: &Tensor<Bk, 2>,
        sinks: Tensor<Bk, 2>,
        k_all: Tensor<Bk, 2>,
        v_all: Tensor<Bk, 2>,
        first: usize,
    ) -> Tensor<Bk, 2> {
        let [t, heads, hd] = q.dims();
        let [n, _] = k_all.dims();
        let dev = q.device();
        let (kvh, e) = (a.kv_heads, a.rel_extent);
        let rel_logits = rel.matmul(rel_proj.clone()).reshape([t, heads, e]).swap_dims(0, 1);
        let mut gather = Vec::with_capacity(heads * t * n);
        let mut masked = Vec::with_capacity(heads * t * n);
        let mut no_bias = Vec::with_capacity(heads * t * n);
        for _ in 0..heads {
            for i in 0..t {
                for j in 0..n {
                    let dist = (first + i) as i64 - j as i64;
                    gather.push(dist.clamp(0, e as i64 - 1));
                    masked.push(dist < 0 || a.window.is_some_and(|w| dist >= w as i64));
                    no_bias.push(dist < 0 || dist >= e as i64);
                }
            }
        }
        let gather = Tensor::<Bk, 3, Int>::from_data(TensorData::new(gather, [heads, t, n]), &dev);
        let masked = Tensor::<Bk, 3, Bool>::from_data(TensorData::new(masked, [heads, t, n]), &dev);
        let no_bias = Tensor::<Bk, 3, Bool>::from_data(TensorData::new(no_bias, [heads, t, n]), &dev);
        let bias = rel_logits.gather(2, gather).mask_fill(no_bias, 0.0);
        let rep = heads / kvh;
        let q = q.swap_dims(0, 1).reshape([kvh, rep * t, hd]);
        let kt = k_all.reshape([n, kvh, hd]).swap_dims(0, 1).swap_dims(1, 2);
        let scores = q.matmul(kt).mul_scalar((hd as f32).powf(-0.5)).reshape([heads, t, n]) + bias;
        let scores = scores.mask_fill(masked, f32::NEG_INFINITY);
        let sinks = sinks.reshape([t, heads, 1]).swap_dims(0, 1);
        let probs = softmax(Tensor::cat(vec![scores, sinks], 2), 2).slice([0..heads, 0..t, 0..n]);
        let vv = v_all.reshape([n, kvh, hd]).swap_dims(0, 1);
        probs
            .reshape([kvh, rep * t, n])
            .matmul(vv)
            .reshape([heads, t, hd])
            .swap_dims(0, 1)
            .reshape([t, heads * hd])
    }

    /// One random sequence of `n` positions: per-position Q, rel states and
    /// sinks, and K, V.
    struct Case {
        a: AttnShape,
        d_rel: usize,
        n: usize,
        q: Vec<f32>,
        rel: Vec<f32>,
        sinks: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        rel_proj: Tensor<Bk, 2>,
        dev: Dev,
    }

    impl Case {
        fn new(a: AttnShape, d_rel: usize, n: usize, seed: u64) -> Case {
            let dev = Dev::default();
            let mut r = Lcg(seed);
            let (h, kvd) = (a.heads, a.kv_heads * a.head_dim);
            // BF16 values, so the rounding of Q, K and V changes nothing
            let bf = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| half::bf16::from_f32(x).to_f32()).collect() };
            let q = bf(r.vec(n * h * a.head_dim, 2.0));
            let rel = r.vec(n * h * d_rel, 1.0);
            let mut sinks = r.vec(n * h, 2.0);
            // a sink-dominated row: every head of position 7 sinks hard
            for x in &mut sinks[7 * h..8 * h] {
                *x = 60.0;
            }
            let k = bf(r.vec(n * kvd, 1.0));
            let v = bf(r.vec(n * kvd, 1.0));
            let rel_proj = up(r.vec(d_rel * a.rel_extent, 1.0), [d_rel, a.rel_extent], &dev);
            Case { a, d_rel, n, q, rel, sinks, k, v, rel_proj, dev }
        }

        fn rows(&self, r: std::ops::Range<usize>) -> (Tensor<Bk, 3>, Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>) {
            let (h, hd, kvd, t) = (self.a.heads, self.a.head_dim, self.a.kv_heads * self.a.head_dim, r.len());
            (
                up(self.q[r.start * h * hd..r.end * h * hd].to_vec(), [t, h, hd], &self.dev),
                up(self.rel[r.start * h * self.d_rel..r.end * h * self.d_rel].to_vec(), [t * h, self.d_rel], &self.dev),
                up(self.sinks[r.start * h..r.end * h].to_vec(), [t, h], &self.dev),
                up(self.k[r.start * kvd..r.end * kvd].to_vec(), [t, kvd], &self.dev),
                up(self.v[r.start * kvd..r.end * kvd].to_vec(), [t, kvd], &self.dev),
            )
        }

        /// Every position's output, from the materialised attention over the
        /// whole sequence at once.
        fn reference(&self) -> Vec<f32> {
            let (q, rel, sinks, k, v) = self.rows(0..self.n);
            host(materialised(&self.a, q, rel, &self.rel_proj, sinks, k, v, 0))
        }

        /// Every position's output in f64 on the host, from the definition in
        /// the module doc: no device arithmetic at all.
        fn exact(&self) -> Vec<f32> {
            let a = &self.a;
            let (h, hd, kvh, e, dr) = (a.heads, a.head_dim, a.kv_heads, a.rel_extent, self.d_rel);
            let rep = h / kvh;
            let proj = host(self.rel_proj.clone());
            let mut out = vec![0f32; self.n * h * hd];
            let mut w = vec![0f64; self.n];
            for p in 0..self.n {
                for hh in 0..h {
                    let g = hh / rep;
                    let q = &self.q[(p * h + hh) * hd..(p * h + hh + 1) * hd];
                    let rel = &self.rel[(p * h + hh) * dr..(p * h + hh + 1) * dr];
                    let sink = self.sinks[p * h + hh] as f64;
                    let lo = a.window.map_or(0, |win| (p + 1).saturating_sub(win));
                    let mut m = sink;
                    for j in lo..=p {
                        let k = &self.k[(j * kvh + g) * hd..(j * kvh + g + 1) * hd];
                        let mut s: f64 = q.iter().zip(k).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>() / (hd as f64).sqrt();
                        let d = p - j;
                        if d < e {
                            s += (0..dr).map(|r| rel[r] as f64 * proj[r * e + d] as f64).sum::<f64>();
                        }
                        w[j] = s;
                        m = m.max(s);
                    }
                    let mut den = (sink - m).exp();
                    let mut acc = vec![0f64; hd];
                    for j in lo..=p {
                        let pj = (w[j] - m).exp();
                        den += pj;
                        let v = &self.v[(j * kvh + g) * hd..(j * kvh + g + 1) * hd];
                        for (o, x) in acc.iter_mut().zip(v) {
                            *o += pj * *x as f64;
                        }
                    }
                    for (d, o) in acc.iter().enumerate() {
                        out[(p * h + hh) * hd + d] = (o / den) as f32;
                    }
                }
            }
            out
        }

        /// Every position's output from [`attend`], fed in calls of `plan`
        /// lengths, with a sliding window's eviction after each call. Checks
        /// after each call that the cache starts no later than the oldest key
        /// the next query can see, and drops everything a segment earlier.
        fn tiled(&self, plan: &[usize], key_tile: usize, query_tile: usize) -> Vec<f32> {
            let mut kv = Kv::new(key_tile);
            let mut out = Vec::new();
            let mut at = 0;
            for &len in plan {
                let (q, rel, sinks, k, v) = self.rows(at..at + len);
                kv.append(k, v);
                out.extend(host(attend(&self.a, q, rel, &self.rel_proj, sinks, &kv, at, query_tile)));
                at += len;
                if let Some(w) = self.a.window {
                    let oldest = (at + 1).saturating_sub(w);
                    kv.evict_before(oldest);
                    assert!(kv.base() <= oldest, "at {at}: the cache starts at {} after the oldest visible key {oldest}", kv.base());
                    assert!(
                        kv.base() + key_tile > oldest || kv.allocated_rows() <= key_tile,
                        "at {at}: a whole segment before {oldest} survived (base {})",
                        kv.base()
                    );
                    assert!(kv.allocated_rows() <= (w - 1).div_ceil(key_tile) * key_tile + 2 * key_tile);
                }
                assert_eq!(kv.end(), at);
            }
            assert_eq!(at, self.n, "the plan covers the sequence");
            out
        }
    }

    /// Worst absolute difference per row, and the worst of all.
    fn worst(got: &[f32], want: &[f32], width: usize) -> (f32, usize) {
        let mut w = (0f32, 0usize);
        for (i, (g, r)) in got.chunks(width).zip(want.chunks(width)).enumerate() {
            let d = g.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(d.is_finite(), "row {i} is not finite");
            if d > w.0 {
                w = (d, i);
            }
        }
        w
    }

    /// Each plan through [`attend`] against the exact host attention, beside
    /// the materialised device attention against the same: the tiled path
    /// must be no further from the definition than the code it replaces was
    /// (with slack for the order of the sums), and the sink-dominated row
    /// must be zero.
    fn check(c: &Case, plans: &[(&[usize], usize, usize)]) {
        let exact = c.exact();
        let width = c.a.heads * c.a.head_dim;
        let sunk = exact[7 * width..8 * width].iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(sunk < 1e-12, "position 7's sink does not dominate: {sunk}");
        let mat = c.reference();
        let (dm, rm) = worst(&mat, &exact, width);
        println!("materialised (d17aa9d) vs exact: worst abs {dm:.3e} at position {rm}");
        for &(plan, kt, qt) in plans {
            let got = c.tiled(plan, kt, qt);
            let (d, row) = worst(&got, &exact, width);
            let (dt, rt) = worst(&got, &mat, width);
            println!(
                "plan {:?}{} key tile {kt} query tile {qt}: vs exact worst abs {d:.3e} at position {row}; vs materialised {dt:.3e} at {rt}",
                &plan[..plan.len().min(12)],
                if plan.len() > 12 { format!(" .. ({} calls)", plan.len()) } else { String::new() }
            );
            assert!(d <= 1.5 * dm + 1e-5, "plan {plan:?}, tiles {kt}/{qt}: position {row} is {d} from exact, the materialised path {dm}");
            let s = got[7 * width..8 * width].iter().fold(0f32, |m, x| m.max(x.abs()));
            assert!(s < 1e-12, "position 7 under the sink: {s}");
        }
    }

    // Mini's head grouping (24 over 8 is three heads a KV head), small sizes so
    // a sequence crosses many segments, windows and the bias extent.
    fn small(window: Option<usize>, rel_extent: usize) -> AttnShape {
        AttnShape { heads: 6, kv_heads: 2, head_dim: 32, rel_extent, window }
    }

    #[test]
    fn sliding_tiles_match_materialised() {
        // window 48 = rel_extent, as on Mini's sliding layers; key tiles of 16
        // and query tiles of 12 never line up, so query tiles straddle segments
        // and some rows see no key of a live segment (rows 12..15 of the tile
        // at 12 against the segment at 16, and rows beyond the window)
        let c = Case::new(small(Some(48), 48), 4, 203, 11);
        check(
            &c,
            &[
                (&[203], 16, 12),
                (&[203], 1024, 1024),
                (&[30, 20, 1, 1, 1, 50, 47, 3, 1, 49], 16, 12),
                (&[47, 1, 1, 48, 1, 105], 16, 5),
                (&[1; 203], 16, 12),
            ],
        );
    }

    #[test]
    fn full_tiles_match_materialised() {
        // no window, rel_extent 40: many full segments behind every query,
        // some wholly beyond the bias extent
        let c = Case::new(small(None, 40), 4, 211, 23);
        check(
            &c,
            &[
                (&[211], 16, 12),
                (&[211], 64, 64),
                (&[100, 1, 1, 1, 60, 48], 16, 12),
                (&[1; 211], 16, 7),
            ],
        );
    }

    #[test]
    fn decode_appends_pack_into_segments() {
        let dev = Dev::default();
        let mut kv = Kv::new(16);
        let row = || Tensor::<Bk, 2>::ones([1, 4], &dev);
        for i in 1..=40 {
            kv.append(row(), row());
            assert_eq!(kv.end(), i);
            assert_eq!(kv.segments().count(), i.div_ceil(16), "{i} rows");
        }
        assert_eq!(kv.allocated_rows(), 48);
        kv.evict_before(33);
        assert_eq!((kv.base(), kv.rows()), (32, 8));
        let keys = host(kv.keys().unwrap());
        assert_eq!(keys.len(), 8 * 4);
    }

    /// The edges, made sharp: one key dominates every query that can see it,
    /// so a query's output is that key's value exactly when the key is
    /// visible, and an off-by-one at the window's edge or the bias extent
    /// moves an output by O(1), not by the device's rounding.
    ///
    /// Every query is the same vector `u` per head, every key orthogonal-ish
    /// noise except key `j0`, which is `30 u / |u|^2 * sqrt(head_dim)` (a score
    /// of 30 against every query). Sinks are 0, bias off (`rel` zero). On a
    /// sliding layer queries `j0 ..= j0 + window - 1` must return `v[j0]`, and
    /// query `j0 + window` must not; checked one-shot, in Drive-sized chunks
    /// and one token at a time across the window.
    #[test]
    fn the_oldest_visible_key_is_exactly_the_windows_edge() {
        let a = AttnShape { heads: 24, kv_heads: 8, head_dim: 128, rel_extent: 1024, window: Some(1024) };
        let (j0, n) = (5usize, 1300usize);
        let mut c = Case::new(a, 16, n, 41);
        let (h, hd, kvh) = (a.heads, a.head_dim, a.kv_heads);
        let mut r = Lcg(99);
        let u: Vec<f32> = r.vec(hd, 1.0);
        let uu: f32 = u.iter().map(|x| x * x).sum();
        for p in 0..n {
            for hh in 0..h {
                c.q[(p * h + hh) * hd..(p * h + hh + 1) * hd].copy_from_slice(&u);
            }
        }
        c.k.iter_mut().for_each(|x| *x *= 0.05);
        for g in 0..kvh {
            for (d, x) in c.k[(j0 * kvh + g) * hd..(j0 * kvh + g + 1) * hd].iter_mut().enumerate() {
                *x = 30.0 * u[d] / uu * (hd as f32).sqrt();
            }
        }
        c.rel.iter_mut().for_each(|x| *x = 0.0);
        c.sinks.iter_mut().for_each(|x| *x = 0.0);
        let width = h * hd;
        let vj0: Vec<f32> = (0..h).flat_map(|hh| c.v[(j0 * kvh + hh / (h / kvh)) * hd..][..hd].to_vec()).collect();
        let near = |row: &[f32]| row.iter().zip(&vj0).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        for plan in [&[n][..], &[1000, 23, 1, 1, 1, 274], &[1020, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 261]] {
            let got = c.tiled(plan, KEY_TILE, QUERY_TILE);
            for p in 0..n {
                let d = near(&got[p * width..(p + 1) * width]);
                let sees = p >= j0 && p < j0 + 1024;
                if sees {
                    assert!(d < 5e-2, "plan {plan:?}: query {p} sees key {j0} but its output is {d} from v[{j0}]");
                } else {
                    assert!(d > 0.1, "plan {plan:?}: query {p} cannot see key {j0} yet returns v[{j0}] ({d})");
                }
            }
            println!("plan {:?}..: queries {j0}..={} return v[{j0}], {} onward do not", &plan[..plan.len().min(4)], j0 + 1023, j0 + 1024);
        }
    }

    /// The bias extent, made sharp on a full layer: the relative profile
    /// puts +30 at distance `rel_extent - 1` and nothing else, so every query
    /// `p >= rel_extent - 1` returns `v[p - rel_extent + 1]`, and the key one
    /// further back, visible but unbiased, does not take over.
    #[test]
    fn the_bias_ends_exactly_at_rel_extent() {
        let a = AttnShape { heads: 24, kv_heads: 8, head_dim: 128, rel_extent: 1024, window: None };
        let n = 1300usize;
        let mut c = Case::new(a, 16, n, 43);
        let (h, hd, kvh, e) = (a.heads, a.head_dim, a.kv_heads, a.rel_extent);
        c.q.iter_mut().for_each(|x| *x *= 0.02);
        c.sinks.iter_mut().for_each(|x| *x = 0.0);
        // rel states all (1, 0, ..), and a profile whose row 0 is +30 at e - 1
        for p in 0..n * h {
            for r in 0..16 {
                c.rel[p * 16 + r] = if r == 0 { 1.0 } else { 0.0 };
            }
        }
        let mut proj = vec![0f32; 16 * e];
        proj[e - 1] = 30.0;
        c.rel_proj = up(proj, [16, e], &c.dev);
        let width = h * hd;
        for plan in [&[n][..], &[1000, 23, 1, 1, 1, 274]] {
            let got = c.tiled(plan, KEY_TILE, QUERY_TILE);
            for p in (e - 1)..n {
                let j = p + 1 - e;
                let want: Vec<f32> = (0..h).flat_map(|hh| c.v[(j * kvh + hh / (h / kvh)) * hd..][..hd].to_vec()).collect();
                let d = got[p * width..(p + 1) * width].iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(d < 5e-2, "plan {plan:?}: query {p} should return v[{j}] (biased at distance {}), is {d} away", e - 1);
            }
        }
    }

    /// The kernels without the bias table's own rounding: one-hot relative
    /// states over a BF16 profile make the table exact, so what is left is the
    /// attention's arithmetic against the f64 definition, for long calls (the
    /// tensor-core kernel) and for single tokens (the fused kernel) alike. The
    /// tests above cannot see this: the f32 matmul that builds a random table
    /// is about 1e-3 from exact.
    #[test]
    fn the_kernels_match_the_definition_given_an_exact_table() {
        for (window, e, n) in [(Some(48), 48, 203), (None, 40, 211)] {
            let mut c = Case::new(small(window, e), 4, n, 61);
            for (i, x) in c.rel.iter_mut().enumerate() {
                *x = if i % 4 == i / 4 % 4 { 1.0 } else { 0.0 };
            }
            let mut r = Lcg(62);
            let proj = r.vec(4 * e, 1.0).into_iter().map(|x| half::bf16::from_f32(x).to_f32()).collect();
            c.rel_proj = up(proj, [4, e], &c.dev);
            let exact = c.exact();
            let width = c.a.heads * c.a.head_dim;
            for (label, plan, kt, qt) in [
                ("one call", vec![n], 1024, 1024),
                ("one call, small tiles", vec![n], 16, 40),
                ("single tokens", vec![1; n], 16, 12),
            ] {
                let got = c.tiled(&plan, kt, qt);
                let (d, row) = worst(&got, &exact, width);
                println!("window {window:?} {label}: worst abs {d:.3e} at position {row}");
                assert!(d < 2e-5, "window {window:?} {label}: {d} from exact at position {row}");
            }
        }
    }

    /// Mini's real attention shape and window, across the 1024 boundary with
    /// the production tiles, in one call and in Drive-like and decode-like
    /// calls.
    #[test]
    fn mini_shape_across_the_window() {
        let sliding = AttnShape { heads: 24, kv_heads: 8, head_dim: 128, rel_extent: 1024, window: Some(1024) };
        let c = Case::new(sliding, 16, 1300, 5);
        check(
            &c,
            &[(&[1300], KEY_TILE, QUERY_TILE), (&[1000, 23, 1, 1, 1, 274], KEY_TILE, QUERY_TILE), (&[1020, 1, 1, 1, 1, 1, 1, 274], KEY_TILE, QUERY_TILE)],
        );
        let full = AttnShape { window: None, ..sliding };
        let c = Case::new(full, 16, 1300, 6);
        check(&c, &[(&[1300], KEY_TILE, QUERY_TILE), (&[1000, 23, 1, 1, 1, 274], KEY_TILE, QUERY_TILE)]);
    }
}

#[cfg(test)]
mod cost {
    //! Wall time of [`attend`] alone at Mini's full-attention shape, with the
    //! production tiles, synchronised by downloading the output. Not a test of
    //! anything; run with `--ignored --nocapture` to print.

    use super::*;
    use std::time::Instant;

    type Dev = burn::backend::cuda::CudaDevice;

    fn time(kv: &Kv, t: usize, first: usize, a: &AttnShape, proj: &Tensor<Bk, 2>, dev: &Dev, reps: usize) -> f64 {
        let q = Tensor::<Bk, 3>::random([t, a.heads, a.head_dim], burn::tensor::Distribution::Normal(0.0, 1.0), dev);
        let rel = Tensor::<Bk, 2>::random([t * a.heads, 16], burn::tensor::Distribution::Normal(0.0, 1.0), dev);
        let sinks = Tensor::<Bk, 2>::zeros([t, a.heads], dev);
        let _ = attend(a, q.clone(), rel.clone(), proj, sinks.clone(), kv, first, QUERY_TILE).into_data();
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = attend(a, q.clone(), rel.clone(), proj, sinks.clone(), kv, first, QUERY_TILE).into_data();
        }
        t0.elapsed().as_secs_f64() / reps as f64
    }

    #[test]
    #[ignore]
    fn full_attention_cost_at_length() {
        let dev = Dev::default();
        let a = AttnShape { heads: 24, kv_heads: 8, head_dim: 128, rel_extent: 1024, window: None };
        let proj = Tensor::<Bk, 2>::random([16, 1024], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
        let kvd = a.kv_heads * a.head_dim;
        let mut kv = Kv::new(KEY_TILE);
        let mut filled = 0;
        for n in [4096usize, 16384, 65536, 131072] {
            while filled < n {
                let r = Tensor::<Bk, 2>::random([4096, kvd], burn::tensor::Distribution::Normal(0.0, 1.0), &dev);
                kv.append(r.clone(), r);
                filled += 4096;
            }
            // a 4096-query chunk ending at n (Drive's prefill call), and one decode query at n - 1
            let chunk = time(&kv, 4096, n - 4096, &a, &proj, &dev, 2);
            let step = time(&kv, 1, n - 1, &a, &proj, &dev, 5);
            println!(
                "keys {n}: 4096-query chunk {chunk:.3} s ({} tile pairs, {:.2} ms a pair); one decode query {:.1} ms ({} segments)",
                4 * n / KEY_TILE - 6,
                chunk * 1e3 / (4 * n / KEY_TILE - 6) as f64,
                step * 1e3,
                n / KEY_TILE
            );
        }
    }
}
