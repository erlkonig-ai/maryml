//! The post-loop tail, physical layers 18..23 and the output head, held to
//! transformers 046bdb8c on fixed inputs: Mary's shared device layers
//! ([`SlidingLayer`] for 18..22, [`FullLayer`] for 23) and the CPU
//! [`Head`], against `golden/capture_inkling2_tail.py`'s record.
//!
//! Inputs, as upstream's `Inkling2LoopTextModel.forward` hands them to the
//! post-loop layers: the hidden state entering 18 `[t, H]`, the three block
//! states `[t, 3, H]` and the absolute positions (only layer 23's log scale
//! reads them). One prefill, no carried state. 18 is a block boundary: its
//! input becomes the fourth block, which 19..23 and the head attend over.
//!
//! Mary's experts quantise their activations as the model was trained, and
//! upstream's runs multiply unquantised ones, so what lies before a layer's
//! experts is comparable with upstream and what lies behind them is not. Each
//! row is labelled by its input:
//!
//! **Gates**, which decide the result. The tolerance is the layer-0 and
//! layer-5 checks': relative L2 distance to upstream's f32 run within [`TOL`]
//! times upstream's own bf16-vs-f32 distance on the same input.
//! * `iso.<layer>`, every tap before the experts (attention, its conv, the MLP
//!   input, router logits; layer 23's three attention inputs), each layer alone
//!   on the f32 run's own input to it; each token's routed set is upstream
//!   f32's or bf16's; the block count after the visit.
//! * `iso.head`, the head alone on the f32 run's input to it.
//! * `iso.<layer> experts vs one expert at a time`: the grouped experts against
//!   [`OneExpert`], reading the *unmutated* weights itself. Its doc says which
//!   pieces are independent: the weights' selection and layout, the picks, the
//!   products and the combination, and not the quantizers.
//! * the block states that reach the head, counted; every tap finite.
//!
//! **Information**, printed and never failing: the experts, what follows them
//! (`mlp_conv`, `out`), the six layers and the head chained, and the argmax.
//! Their distance to upstream is the trained quantisation, which is meant.
//!
//! [`Mutation`] breaks one piece on purpose; [`check`] then has to fail, at
//! the gate [`Mutation::gate`] names.

use std::collections::{BTreeSet, HashMap};

use burn::prelude::*;
use burn::tensor::TensorData;

use crate::models::inkling::seam::Bk;

type Dev = burn::backend::cuda::CudaDevice;
use crate::models::inkling2_loop::config::{AttnKind, Inkling2LoopTextConfig};
use crate::models::inkling2_loop::full::FullLayer;
use crate::models::inkling2_loop::head::{self, Head, HeadOut};
use crate::models::inkling2_loop::layout::{Dtype, LayerPart, Slot};
use crate::models::inkling2_loop::one_expert::{OneExpert, worst_row};
use crate::models::inkling2_loop::sliding::{
    Dims, SlidingLayer, State, Stored, Taps, Weights, bf16_to_f32, rms_norm, want,
};

/// A pass is within this many times upstream's own bf16-vs-f32 spread.
pub const TOL: f64 = 1.5;
pub const FIRST: usize = 18;
pub const LAST: usize = 23;

/// A safetensors file held in memory: the capture's record.
pub struct Record {
    bytes: Vec<u8>,
    header: serde_json::Map<String, serde_json::Value>,
    base: usize,
}

impl Record {
    pub fn parse(bytes: Vec<u8>) -> Result<Record, String> {
        let n = u64::from_le_bytes(bytes.get(..8).ok_or("short file")?.try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(bytes.get(8..8 + n).ok_or("short header")?).map_err(|e| e.to_string())?;
        let header = header.as_object().ok_or("header is not an object")?.clone();
        Ok(Record { bytes, header, base: 8 + n })
    }

    pub fn read(path: &std::path::Path) -> Result<Record, String> {
        Record::parse(std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
    }

    /// `__metadata__[key]`.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.header.get("__metadata__")?.get(key)?.as_str()
    }

    pub fn raw(&self, name: &str) -> Result<(&str, Vec<usize>, &[u8]), String> {
        let e = self.header.get(name).ok_or_else(|| format!("the record has no {name}"))?;
        let dtype = e["dtype"].as_str().ok_or("dtype")?;
        let shape = e["shape"].as_array().ok_or("shape")?.iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let off = e["data_offsets"].as_array().ok_or("offsets")?;
        let (a, b) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
        Ok((dtype, shape, &self.bytes[self.base + a..self.base + b]))
    }

    pub fn f32s(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (dt, shape, b) = self.raw(name).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(dt, "F32", "{name}");
        (shape, b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn i32s(&self, name: &str) -> Vec<i32> {
        let (dt, _, b) = self.raw(name).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(dt, "I32", "{name}");
        b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

/// The tiny capture also stores its synthetic checkpoint, as `w.<name>`.
impl Weights for Record {
    fn tensor(&self, name: &str) -> Result<Stored<'_>, String> {
        let (dt, shape, bytes) = self.raw(&format!("w.{name}"))?;
        let dtype = match dt {
            "BF16" => Dtype::Bf16,
            "F32" => Dtype::F32,
            "U8" => Dtype::U8,
            "F8_E4M3" => Dtype::F8E4M3,
            other => return Err(format!("{name}: {other}")),
        };
        Ok(Stored { dtype, shape, bytes })
    }
}

/// One piece broken on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    /// Layer 18's w13 per-expert NVFP4 amax times 1.25.
    Nvfp4Amax,
    /// Every UE8M0 scale of layer 20's w13 one step up (weights doubled).
    Mxfp8Scale,
    /// Layer 21's interleaved w13 rows swapped pairwise (gate read as up).
    Mxfp8GateUp,
    /// The block 18 appends dropped before 19 runs.
    NoBoundary,
    /// Layer 23 run 4096 positions later than its input sits.
    Origin,
}

impl Mutation {
    pub const ALL: [Mutation; 5] =
        [Mutation::Nvfp4Amax, Mutation::Mxfp8Scale, Mutation::Mxfp8GateUp, Mutation::NoBoundary, Mutation::Origin];

    /// What every failure of the broken tail starts with: the one gate (or, for
    /// the origin, the one layer's attention) the break sits at.
    pub fn gate(self) -> &'static str {
        match self {
            Mutation::Nvfp4Amax => "iso.18 experts vs one expert at a time",
            Mutation::Mxfp8Scale => "iso.20 experts vs one expert at a time",
            Mutation::Mxfp8GateUp => "iso.21 experts vs one expert at a time",
            Mutation::NoBoundary => "chain: ",
            Mutation::Origin => "iso.23 ",
        }
    }
}

/// A weight source with a few tensors replaced.
struct Patched<'a> {
    inner: &'a dyn Weights,
    owned: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
}

impl Weights for Patched<'_> {
    fn tensor(&self, name: &str) -> Result<Stored<'_>, String> {
        match self.owned.get(name) {
            Some((dtype, shape, bytes)) => Ok(Stored { dtype: *dtype, shape: shape.clone(), bytes }),
            None => self.inner.tensor(name),
        }
    }
}

fn patched<'a>(w: &'a dyn Weights, m: Option<Mutation>) -> Result<Patched<'a>, String> {
    let mut owned = HashMap::new();
    let mut put = |name: String, f: &dyn Fn(&[usize], &mut Vec<u8>)| -> Result<(), String> {
        let s = w.tensor(&name)?;
        let mut b = s.bytes.to_vec();
        f(&s.shape, &mut b);
        owned.insert(name, (s.dtype, s.shape, b));
        Ok(())
    };
    let x = |l: usize, n: &str| format!("model.llm.layers.{l}.mlp.experts.{n}");
    match m {
        Some(Mutation::Nvfp4Amax) => put(x(18, "w13_weight_amax"), &|_, b| {
            for c in b.chunks_exact_mut(4) {
                let v = f32::from_le_bytes(c.try_into().unwrap()) * 1.25;
                c.copy_from_slice(&v.to_le_bytes());
            }
        })?,
        Some(Mutation::Mxfp8Scale) => put(x(20, "w13_weight_scale"), &|_, b| b.iter_mut().for_each(|v| *v += 1))?,
        Some(Mutation::Mxfp8GateUp) => {
            let swap = |shape: &[usize], b: &mut Vec<u8>| {
                let width = shape[2];
                for pair in b.chunks_exact_mut(2 * width) {
                    let (g, u) = pair.split_at_mut(width);
                    g.swap_with_slice(u);
                }
            };
            put(x(21, "w13_weight"), &swap)?;
            put(x(21, "w13_weight_scale"), &swap)?;
        }
        _ => {}
    }
    Ok(Patched { inner: w, owned })
}

fn host<const D: usize>(t: Tensor<Bk, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec().expect("f32")
}

pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// What [`check`] found. Only gates reach `fails`.
#[derive(Default)]
pub struct Report {
    pub lines: Vec<String>,
    pub json: Vec<String>,
    /// Gates compared, and the worst ratio among the upstream ones.
    pub gates: usize,
    pub worst: f64,
    /// Information rows, those beyond the tolerance and their worst ratio.
    pub info: usize,
    pub info_beyond: usize,
    pub info_worst: f64,
    pub fails: Vec<String>,
}

impl Report {
    fn row(&mut self, gate: bool, tag: &str, name: &str, mary: &[f32], r32: &[f32], r16: &[f32]) {
        let e32 = rel_l2(mary, r32);
        let spread = rel_l2(r16, r32).max(1e-7);
        let ratio = e32 / spread;
        let within = ratio <= TOL && e32.is_finite();
        if gate {
            self.gates += 1;
            self.worst = self.worst.max(ratio);
            if !within {
                self.fails.push(format!("{tag} {name}: ratio {ratio:.2}"));
            }
        } else {
            self.info += 1;
            self.info_beyond += usize::from(!within);
            self.info_worst = self.info_worst.max(ratio);
        }
        self.lines.push(format!(
            "{tag:10} {name:14} vs f32 {e32:.3e}  upstream spread {spread:.3e}  ratio {ratio:6.2}  {}",
            match (gate, within) {
                (true, true) => "gate ok",
                (true, false) => "gate OUTSIDE",
                (false, _) => "info",
            }
        ));
        self.json.push(format!(
            "{{\"run\":\"{tag}\",\"tensor\":\"{name}\",\"gate\":{gate},\"rel_l2_vs_f32\":{e32:e},\"upstream_spread\":{spread:e},\"ratio\":{ratio:e},\"within\":{within}}}"
        ));
    }

    /// A gate that is not an upstream distance.
    fn check(&mut self, ok: bool, what: String) {
        self.gates += 1;
        self.lines.push(format!("{what}  gate {}", if ok { "ok" } else { "FAILED" }));
        if !ok {
            self.fails.push(what);
        }
    }

    /// The last line of a run.
    pub fn summary(&self) -> String {
        format!(
            "tail gates: {} compared, worst upstream ratio {:.2} of TOL {TOL}; {} failed. information: {} rows, {} beyond {TOL}, worst ratio {:.2}",
            self.gates, self.worst, self.fails.len(), self.info, self.info_beyond, self.info_worst
        )
    }
}

/// The taps that end at or before the experts' input.
const PRE_EXPERT: [&str; 7] = ["attn", "attn_conv", "mlp_input", "router_logits", "q_in", "k_in", "v_in"];

/// A visit's taps as `(name, values)`, in the record's names.
fn tap_rows(t: Taps, out: Vec<f32>) -> (Vec<(&'static str, Vec<f32>)>, Vec<Vec<usize>>) {
    let rows = vec![
        ("attn", host(t.attn)),
        ("attn_conv", host(t.attn_conv)),
        ("mlp_input", host(t.mlp_input)),
        ("router_logits", t.route.logits()),
        ("experts", host(t.experts)),
        ("mlp_conv", host(t.mlp_conv)),
        ("out", out),
    ];
    (rows, t.route.picks().into_iter().map(|(e, _)| e).collect())
}

enum Bound {
    Sliding(SlidingLayer),
    Full(FullLayer),
}

impl Bound {
    /// One post-loop visit from a fresh state. Returns the output, its taps
    /// and routed sets.
    #[allow(clippy::type_complexity)]
    fn visit(
        &self,
        x: Tensor<Bk, 2>,
        blocks: &mut Vec<Tensor<Bk, 2>>,
        origin: usize,
    ) -> (Tensor<Bk, 2>, Vec<(&'static str, Vec<f32>)>, Vec<Vec<usize>>) {
        let dev = x.device();
        match self {
            Bound::Sliding(l) => {
                let mut st = State::new(&l.dims, &dev);
                let (out, taps) = l.forward(x, blocks, &mut st);
                assert!(taps.gates.is_none(), "a post-loop layer reads no messages");
                let (rows, sets) = tap_rows(taps, host(out.clone()));
                (out, rows, sets)
            }
            Bound::Full(l) => {
                let mut st = State::new(&l.dims, &dev);
                let (out, taps) = l.forward_at(x, blocks, &mut st, origin);
                let [q, k, v] = taps.inputs;
                let (mut rows, sets) = tap_rows(taps.rest, host(out.clone()));
                rows.extend([("q_in", host(q)), ("k_in", host(k)), ("v_in", host(v))]);
                (out, rows, sets)
            }
        }
    }
}

fn bind(w: &dyn Weights, t: &Inkling2LoopTextConfig, layer: usize, dev: &Dev) -> Result<Bound, String> {
    Ok(match t.attn_kind(layer) {
        AttnKind::Local => Bound::Sliding(SlidingLayer::bind(w, t, layer, dev)?),
        AttnKind::Global => Bound::Full(FullLayer::bind(w, t, layer, dev)?),
    })
}

fn stored<'a>(w: &'a dyn Weights, name: &str) -> Result<head::Stored<'a>, String> {
    let s = w.tensor(name)?;
    Ok((s.dtype, s.shape, s.bytes))
}

fn upload(v: Vec<f32>, shape: [usize; 2], dev: &Dev) -> Tensor<Bk, 2> {
    Tensor::from_data(TensorData::new(v, shape), dev)
}

/// `[t, n, H]` as `n` tensors `[t, H]`.
fn split_blocks(v: &[f32], t: usize, n: usize, h: usize, dev: &Dev) -> Vec<Tensor<Bk, 2>> {
    (0..n)
        .map(|j| {
            let rows: Vec<f32> = (0..t).flat_map(|i| v[(i * n + j) * h..(i * n + j + 1) * h].iter().copied()).collect();
            upload(rows, [t, h], dev)
        })
        .collect()
}

/// `n` tensors `[t, H]` as `[t, n, H]` on the host.
fn join_blocks(blocks: &[Tensor<Bk, 2>], t: usize, h: usize) -> Vec<f32> {
    let hosts: Vec<Vec<f32>> = blocks.iter().map(|b| host(b.clone())).collect();
    let mut out = Vec::with_capacity(t * blocks.len() * h);
    for i in 0..t {
        for b in &hosts {
            out.extend_from_slice(&b[i * h..(i + 1) * h]);
        }
    }
    out
}

fn sets(idx: &[i32], k: usize) -> Vec<BTreeSet<usize>> {
    idx.chunks_exact(k).map(|r| r.iter().map(|&e| e as usize).collect()).collect()
}

fn top(v: &[f32], k: usize) -> Vec<usize> {
    let mut i: Vec<usize> = (0..v.len()).collect();
    i.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    i.truncate(k);
    i
}

/// Run the tail on `w` and compare it with `r`. `t` is the text config the
/// record was captured with.
pub fn check(
    w: &dyn Weights,
    t: &Inkling2LoopTextConfig,
    r: &Record,
    mutation: Option<Mutation>,
    dev: &Dev,
) -> Result<Report, String> {
    let unbroken = w;
    let w = patched(w, mutation)?;
    let w: &dyn Weights = &w;
    let h = t.hidden_size;
    let k = t.num_experts_per_tok;
    let (s, x0) = r.f32s("input.x");
    let tokens = s[0];
    let (s, b0) = r.f32s("input.blocks");
    let positions = r.i32s("input.positions");
    let origin = positions[0] as usize;
    let mut rep = Report::default();
    rep.lines.push(format!("tokens {tokens}, {} input blocks, positions from {origin}", s[1]));

    let mut x = upload(x0, [tokens, h], dev);
    let mut blocks = split_blocks(&b0, tokens, s[1], h, dev);
    for layer in FIRST..=LAST {
        let started = std::time::Instant::now();
        let bound = bind(w, t, layer, dev)?;
        let bind_s = started.elapsed().as_secs_f64();
        let at = if layer == LAST && mutation == Some(Mutation::Origin) { origin + 4096 } else { origin };

        // alone, on the f32 run's own input to this layer
        let (_, ix) = r.f32s(&format!("f32.in.{layer}.x"));
        let (bs, ib) = r.f32s(&format!("f32.in.{layer}.blocks"));
        let mut iblocks = split_blocks(&ib, tokens, bs[1], h, dev);
        let (_, rows, ours) = bound.visit(upload(ix, [tokens, h], dev), &mut iblocks, at);
        let want_blocks = bs[1] + usize::from(t.is_block_boundary(layer));
        rep.check(
            iblocks.len() == want_blocks,
            format!("iso.{layer} {} block states after the visit, upstream has {want_blocks}", iblocks.len()),
        );
        for (name, v) in &rows {
            let r32 = r.f32s(&format!("f32.chain.{layer}.{name}")).1;
            let r16 = r.f32s(&format!("bf16.iso.{layer}.{name}")).1;
            rep.row(PRE_EXPERT.contains(name), &format!("iso.{layer}"), name, v, &r32, &r16);
        }
        let finite = rows.iter().filter(|(_, v)| v.iter().all(|x| x.is_finite())).count();
        rep.check(finite == rows.len(), format!("iso.{layer} {finite} of {} taps finite", rows.len()));

        // the grouped experts against one expert at a time, from the unmutated weights
        {
            let d = Dims::of(t);
            let oracle = OneExpert::read(unbroken, &d, layer, dev)?;
            let gain = want(unbroken, Slot::Layer(layer, LayerPart::MlpNorm), Dtype::Bf16, &[h])?;
            let gain = Tensor::<Bk, 1>::from_data(TensorData::new(bf16_to_f32(gain.bytes), [h]), dev);
            let tap = |n: &str| rows.iter().find(|(m, _)| *m == n).expect("a tap").1.clone();
            let mn = rms_norm(upload(tap("mlp_input"), [tokens, h], dev), gain, d.eps);
            let (looped, picks) = oracle.forward(&mn);
            let err = worst_row(&tap("experts"), &looped, h);
            let same = picks
                .iter()
                .zip(&ours)
                .filter(|(a, b)| a.iter().collect::<BTreeSet<_>>() == b.iter().collect::<BTreeSet<_>>())
                .count();
            rep.check(
                err < oracle.bound() && same == tokens,
                format!(
                    "iso.{layer} experts vs one expert at a time: worst row rel err {err:.3e} (bound {:.0e}), the same picks on {same}/{tokens} rows",
                    oracle.bound()
                ),
            );
        }
        let f32_sets = sets(&r.i32s(&format!("f32.chain.{layer}.router_indices")), k);
        let bf16_sets = sets(&r.i32s(&format!("bf16.iso.{layer}.router_indices")), k);
        let mut agree = 0;
        let mut strays = Vec::new();
        for (i, ours) in ours.iter().enumerate() {
            let ours: BTreeSet<usize> = ours.iter().copied().collect();
            if ours == f32_sets[i] {
                agree += 1;
            } else if ours != bf16_sets[i] {
                strays.push(format!("; token {i}: routed {ours:?}, upstream f32 {:?}, bf16 {:?}", f32_sets[i], bf16_sets[i]));
            }
        }
        rep.check(strays.is_empty(), format!("iso.{layer} routed sets are upstream f32's or bf16's{}", strays.concat()));
        rep.lines.push(format!("iso.{layer}    routed sets equal to upstream f32's: {agree}/{tokens}  (bound in {bind_s:.1} s)"));

        // the chain
        let (out, rows, _) = bound.visit(x, &mut blocks, at);
        if layer == FIRST && mutation == Some(Mutation::NoBoundary) {
            blocks.pop();
        }
        for (name, v) in rows.iter().filter(|(n, _)| *n == "out") {
            let r32 = r.f32s(&format!("f32.chain.{layer}.{name}")).1;
            let r16 = r.f32s(&format!("bf16.chain.{layer}.{name}")).1;
            rep.row(false, &format!("chain.{layer}"), name, v, &r32, &r16);
        }
        x = out;
        drop(bound);
    }

    let hd = Head::bind(
        t,
        stored(w, head::QUERY)?,
        stored(w, head::NORM)?,
        stored(w, head::UNEMBED)?,
    )?;
    let head_rows = |o: HeadOut| [("selected", o.selected), ("normed", o.normed), ("raw", o.raw), ("logits", o.logits)];
    let (bs, ib) = r.f32s("f32.in.head.blocks");
    let (_, ix) = r.f32s("f32.in.head.x");
    for (name, v) in head_rows(hd.forward(&ib, bs[1], &ix)) {
        let r32 = r.f32s(&format!("f32.chain.head.{name}")).1;
        let r16 = r.f32s(&format!("bf16.iso.head.{name}")).1;
        rep.row(true, "iso.head", name, &v, &r32, &r16);
    }
    rep.check(blocks.len() == bs[1], format!("chain: {} block states reach the head, upstream has {}", blocks.len(), bs[1]));
    let out = hd.forward(&join_blocks(&blocks, tokens, h), blocks.len(), &host(x));
    let logits = out.logits.clone();
    rep.check(logits.iter().all(|v| v.is_finite()), "chained logits finite".into());
    for (name, v) in head_rows(out) {
        let r32 = r.f32s(&format!("f32.chain.head.{name}")).1;
        let r16 = r.f32s(&format!("bf16.chain.head.{name}")).1;
        rep.row(false, "chain.head", name, &v, &r32, &r16);
    }
    let v = hd.vocab();
    let l32 = r.f32s("f32.chain.head.logits").1;
    let l16 = r.f32s("bf16.chain.head.logits").1;
    for i in 0..tokens {
        let row = |l: &[f32]| l[i * v..(i + 1) * v].to_vec();
        let (ours, a, b) = (top(&row(&logits), 5), top(&row(&l32), 5), top(&row(&l16), 5));
        let overlap = ours.iter().filter(|e| a.contains(e)).count();
        rep.lines.push(format!("token {i}: argmax {} (upstream f32 {}, bf16 {}), top-5 overlap with f32 {overlap}/5  info", ours[0], a[0], b[0]));
        rep.json.push(format!("{{\"token\":{i},\"argmax\":{},\"f32_argmax\":{},\"bf16_argmax\":{},\"top5\":{ours:?},\"f32_top5\":{a:?},\"top5_overlap\":{overlap}}}", ours[0], a[0], b[0]));
    }
    Ok(rep)
}

#[cfg(test)]
pub(super) fn tests_fixture() -> (Record, Inkling2LoopTextConfig) {
    tests::fixture()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `golden/capture_inkling2_tail.py tiny`: a synthetic checkpoint in the
    /// release's stored formats (NVFP4 at 18 and 19, MXFP8 at 20..23) at
    /// hidden 128, 8 experts of width 128, window and relative extent 4, log
    /// scale floor 2, on 7 tokens; its sha256 is beside it. A safetensors file,
    /// named `.st` because the repository ignores `*.safetensors` (weights).
    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/models/inkling2_loop/fixtures/tail_tiny.st");

    pub(super) fn fixture() -> (Record, Inkling2LoopTextConfig) {
        let r = Record::read(std::path::Path::new(FIXTURE)).unwrap();
        let t: Inkling2LoopTextConfig = serde_json::from_str(r.meta("text_config").unwrap()).unwrap();
        (r, t)
    }

    /// The unmodified tail: every gate holds. Synthetic weights in the
    /// release's stored formats.
    #[test]
    fn tail_gates_hold_on_the_tiny_fixture() {
        let (r, t) = fixture();
        let dev = Dev::default();
        let rep = check(&r, &t, &r, None, &dev).unwrap();
        for l in &rep.lines {
            eprintln!("{l}");
        }
        eprintln!("{}", rep.summary());
        assert!(rep.fails.is_empty(), "{:?}", rep.fails);
    }

    /// The same fixture and the same assertion, one piece broken: it must not
    /// hold, and nothing but the gate at the break may fail.
    #[test]
    fn a_broken_tail_fails_at_its_gate() {
        let (r, t) = fixture();
        let dev = Dev::default();
        for m in Mutation::ALL {
            let rep = check(&r, &t, &r, Some(m), &dev).unwrap();
            eprintln!("{m:?}: {} failures {:?}", rep.fails.len(), rep.fails);
            assert!(!rep.fails.is_empty(), "{m:?} passed");
            assert!(rep.fails.iter().all(|f| f.starts_with(m.gate())), "{m:?} failed elsewhere too: {:?}", rep.fails);
        }
    }
}
