//! Inkling 2 Mini from the embedding through layer 5, natively, chained,
//! against transformers 046bdb8c's own run of the same layers.
//!
//! Reads the model from a pile alone (pile, model collection, model root, as
//! [`MiniRoot`]): the normed embedding, sliding layers 0..4 and the first
//! full-attention layer 5, bound through `sliding::Weights`. Mary computes
//! every layer's input from the tokens: nothing upstream produced is fed in.
//! Block states are carried through the pass (layer 0 pushes its input), and
//! each layer has its own token cache. Run for the prompt, for one more token
//! through the caches, and once as one no-cache prefill of both.
//!
//! `golden/capture_inkling2_chain5.py` runs upstream the same way, at float32
//! and at bfloat16, nothing fed, so their distance at each layer is how far
//! rounding alone has moved the chain there. A tensor passes when its distance
//! to upstream's f32 run is within `TOL` times that. The report names the
//! earliest tensor in chain order that does not.
//!
//! usage: inkling2_chain5 <pile> <collection> <root> <reference.safetensors> [receipt.json]

use std::collections::BTreeSet;

use burn::prelude::*;
use burn::tensor::TensorData;
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;

use mary::model_collection::parse_collection_handle;
use mary::models::inkling::burn::rms_norm;
use mary::models::inkling::seam::Bk;
use mary::models::inkling2_loop::Inkling2LoopConfig;
use mary::models::inkling2_loop::full::{FullLayer, FullTaps};
use mary::models::inkling2_loop::layout::{LayerPart, Slot};
use mary::models::inkling2_loop::pile::MiniRoot;
use mary::models::inkling2_loop::sliding::{ExpertAct, SlidingLayer, State, Taps, Weights};
use triblespace::prelude::Id;

/// A pass is within this many times upstream's own bf16-vs-f32 spread.
const TOL: f64 = 1.5;
/// The full-attention layer the chain ends in; layers below it are sliding.
const FULL: usize = 5;

struct File {
    map: Mmap,
    meta: Metadata,
    base: usize,
}

impl File {
    fn open(path: &str) -> File {
        let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let map = unsafe { Mmap::map(&f) }.expect("mmap");
        let (n, meta) = SafeTensors::read_metadata(&map).expect("safetensors header");
        File { map, meta, base: 8 + n }
    }

    fn raw(&self, name: &str) -> Option<(&safetensors::tensor::TensorInfo, &[u8])> {
        let info = self.meta.info(name)?;
        let (a, b) = info.data_offsets;
        Some((info, &self.map[self.base + a..self.base + b]))
    }

    fn f32s(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (info, b) = self.raw(name).unwrap_or_else(|| panic!("reference has no {name}"));
        assert_eq!(info.dtype, safetensors::Dtype::F32, "{name}");
        (info.shape.clone(), b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    fn i32s(&self, name: &str) -> Vec<i32> {
        let (info, b) = self.raw(name).unwrap_or_else(|| panic!("reference has no {name}"));
        assert_eq!(info.dtype, safetensors::Dtype::I32, "{name}");
        b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

fn host<const D: usize>(t: Tensor<Bk, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec().expect("f32")
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max)
}

/// Upstream's cache `[heads, n, d]` as Mary's rows `[n, heads * d]`.
fn heads_to_rows(shape: &[usize], v: &[f32]) -> Vec<f32> {
    let (h, n, d) = (shape[0], shape[1], shape[2]);
    let mut out = vec![0f32; v.len()];
    for hh in 0..h {
        for t in 0..n {
            for x in 0..d {
                out[t * h * d + hh * d + x] = v[(hh * n + t) * d + x];
            }
        }
    }
    out
}

/// Upstream's conv state `[C, k]` (the last `k` inputs, oldest first) as the
/// `[k - 1, C]` rows Mary carries.
fn conv_to_rows(shape: &[usize], v: &[f32]) -> Vec<f32> {
    let (c, k) = (shape[0], shape[1]);
    let mut out = Vec::with_capacity(c * (k - 1));
    for j in 1..k {
        for ch in 0..c {
            out.push(v[ch * k + j]);
        }
    }
    out
}

/// `rel_l2` of each `[h]` row, with the row's norm in the reference.
fn per_row(mary: &[f32], r32: &[f32], h: usize) -> Vec<(usize, f64, f64)> {
    mary.chunks(h)
        .zip(r32.chunks(h))
        .enumerate()
        .map(|(i, (a, b))| (i, rel_l2(a, b), b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()))
        .collect()
}

/// The three rows furthest from f32, and the median row, as one line.
fn worst_rows(mary: &[f32], r32: &[f32], h: usize) -> String {
    let mut rows = per_row(mary, r32, h);
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let median = rows[rows.len() / 2].1;
    let top: Vec<String> = rows.iter().take(3).map(|(i, e, n)| format!("row {i} {e:.2e} (|ref| {n:.1})")).collect();
    format!("worst {}; median row {median:.2e}", top.join(", "))
}

struct Report {
    lines: Vec<String>,
    json: Vec<String>,
    worst: f64,
    fails: usize,
    /// The first gated tensor outside, in chain order.
    earliest: Option<String>,
}

impl Report {
    fn row(&mut self, tag: &str, name: &str, mary: &[f32], r16: &[f32], r32: &[f32], gated: bool) {
        let e32 = rel_l2(mary, r32);
        let e16 = rel_l2(mary, r16);
        let spread = rel_l2(r16, r32);
        let ratio = e32 / spread;
        let pass = ratio <= TOL;
        if gated {
            self.worst = self.worst.max(ratio);
            if !pass {
                self.fails += 1;
                self.earliest.get_or_insert_with(|| format!("{tag} {name} (ratio {ratio:.2})"));
            }
        }
        self.lines.push(format!(
            "{tag:14} {name:16} vs f32 {e32:.3e} (max {:.2e})  vs bf16 {e16:.3e}  upstream spread {spread:.3e}  ratio {ratio:6.2}  {}",
            max_abs(mary, r32),
            if !gated { "report" } else if pass { "ok" } else { "OUTSIDE" },
        ));
        self.json.push(format!(
            "{{\"run\":\"{tag}\",\"tensor\":\"{name}\",\"rel_l2_vs_f32\":{e32:e},\"rel_l2_vs_bf16\":{e16:e},\"upstream_spread\":{spread:e},\"ratio\":{ratio:e},\"gated\":{gated}}}"
        ));
    }
}

/// What one pass of the chain leaves behind, on the host.
struct Pass {
    /// Every layer's input and output, `[t, H]` each; layer 0's input is the
    /// normed embedding.
    inputs: Vec<Vec<f32>>,
    outs: Vec<Vec<f32>>,
    /// Layer 5's block states, one `[t, H]` per completed block.
    blocks: Vec<Vec<f32>>,
    /// Layer 5's taps and routing.
    taps: Vec<(&'static str, Vec<f32>)>,
    routing: Vec<(Vec<usize>, Vec<f32>)>,
    /// Every layer's router logits `[t, experts]` and chosen experts per row.
    layer_logits: Vec<Vec<f32>>,
    layer_experts: Vec<Vec<Vec<usize>>>,
    ms: f64,
}

struct Chain {
    sliding: Vec<SlidingLayer>,
    full: FullLayer,
}

impl Chain {
    fn states(&self, dev: &burn::backend::cuda::CudaDevice) -> Vec<State> {
        let mut s: Vec<State> = self.sliding.iter().map(|l| State::new(&l.dims, dev)).collect();
        s.push(State::new(&self.full.dims, dev));
        s
    }

    fn run(&self, x: Tensor<Bk, 2>, states: &mut [State], act: ExpertAct) -> Pass {
        let t1 = std::time::Instant::now();
        let (mut inputs, mut outs) = (Vec::new(), Vec::new());
        let mut blocks = Vec::new();
        let (mut layer_logits, mut layer_experts) = (Vec::new(), Vec::new());
        let mut x = x;
        for (l, layer) in self.sliding.iter().enumerate() {
            inputs.push(host(x.clone()));
            let (y, taps) = layer.forward(x, &mut blocks, &mut states[l], act);
            x = y;
            layer_logits.push(taps.router_logits);
            layer_experts.push(taps.routing.into_iter().map(|(e, _)| e).collect());
            outs.push(host(x.clone()));
        }
        inputs.push(host(x.clone()));
        let block_rows: Vec<Vec<f32>> = blocks.iter().map(|b| host(b.clone())).collect();
        let (out, taps) = self.full.forward(x, &mut blocks, &mut states[FULL], act);
        let out = host(out);
        outs.push(out.clone());
        let FullTaps { inputs: [qi, ki, vi], rest } = taps;
        let Taps { attn, attn_conv, mlp_input, router_logits, routing, experts, mlp_conv } = rest;
        layer_logits.push(router_logits);
        layer_experts.push(routing.iter().map(|(e, _)| e.clone()).collect());
        Pass {
            inputs,
            outs,
            blocks: block_rows,
            taps: vec![
                ("q_in", host(qi)),
                ("k_in", host(ki)),
                ("v_in", host(vi)),
                ("attn", host(attn)),
                ("attn_conv", host(attn_conv)),
                ("mlp_input", host(mlp_input)),
                ("experts", host(experts)),
                ("mlp_conv", host(mlp_conv)),
                ("out", out),
            ],
            routing,
            layer_logits,
            layer_experts,
            ms: t1.elapsed().as_secs_f64() * 1e3,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (pile, collection, root, reference) = (&args[1], &args[2], &args[3], &args[4]);
    let collection = parse_collection_handle(collection).expect("collection");
    let root = Id::from_hex(root).expect("root must be a 32-hex id");
    let t0 = std::time::Instant::now();
    let model = MiniRoot::open(std::path::Path::new(pile), collection, root).expect("open the model root");
    let (nm, nd) = model.counts();
    println!(
        "opened model root {:X} in {pile} ({nm} + {nd} tensors) in {:.1} s",
        model.root(),
        t0.elapsed().as_secs_f64()
    );
    let cfg = Inkling2LoopConfig::from_json(&model.config().expect("config facts").to_string()).expect("config");
    let t = &cfg.text_config;
    let r = File::open(reference);
    let dev = burn::backend::cuda::CudaDevice::default();
    let h = t.hidden_size;

    let prompt: Vec<usize> = r.i32s("prompt").iter().map(|&x| x as usize).collect();
    let step: Vec<usize> = r.i32s("step_token").iter().map(|&x| x as usize).collect();
    let mut whole_ids = prompt.clone();
    whole_ids.extend(&step);

    // the normed embedding: a gather of stored BF16 rows, then RMSNorm
    let embed = Weights::tensor(&model, &Slot::Embed.tensor_name()).unwrap();
    let gain = Weights::tensor(&model, &Slot::EmbedNorm.tensor_name()).unwrap();
    let bf = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
    };
    let gain = Tensor::<Bk, 1>::from_data(TensorData::new(bf(gain.bytes), [h]), &dev);
    let embed_rows = |ids: &[usize]| -> Tensor<Bk, 2> {
        let mut v = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            v.extend(bf(&embed.bytes[id * h * 2..(id + 1) * h * 2]));
        }
        let x = Tensor::<Bk, 2>::from_data(TensorData::new(v, [ids.len(), h]), &dev);
        rms_norm(x, gain.clone(), t.rms_norm_eps)
    };

    let t1 = std::time::Instant::now();
    let chain = Chain {
        sliding: (0..FULL)
            .map(|l| SlidingLayer::bind(&model, t, l, &dev).unwrap_or_else(|e| panic!("bind layer {l}: {e}")))
            .collect(),
        full: FullLayer::bind(&model, t, FULL, &dev).expect("bind layer 5"),
    };
    println!("bound layers 0..{FULL} in {:.1} s", t1.elapsed().as_secs_f64());
    // each router's score bias, for the margin at a row where the choice differs
    let bias: Vec<Vec<f32>> = (0..=FULL)
        .map(|l| {
            let b = Weights::tensor(&model, &Slot::Layer(l, LayerPart::GateBias).tensor_name()).unwrap();
            b.bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        })
        .collect();
    let ne = t.n_routed_experts;
    let k = t.num_experts_per_tok;
    // upstream's own f32 selection score margin between the k-th and (k+1)-th expert of a row
    let margin = |logits: &[f32], b: &[f32]| -> f64 {
        let mut s: Vec<f64> = logits.iter().zip(b).map(|(&x, &c)| 1.0 / (1.0 + (-(x as f64)).exp()) + c as f64).collect();
        s.sort_by(|a, b| b.total_cmp(a));
        s[k - 1] - s[k]
    };

    let ref_t = |run: &str, key: &str| r.f32s(&format!("{run}.{key}")).1;
    let mut rep = Report { lines: vec![], json: vec![], worst: 0.0, fails: 0, earliest: None };

    // every layer's input and output, then layer 5's block states and taps,
    // in chain order, so the first row outside is the earliest divergence
    let compare = |rep: &mut Report, tag: &str, phase: &str, p: &Pass, gated: bool| {
        let g = |n: &str| (ref_t("bf16", &format!("{phase}.{n}")), ref_t("f32", &format!("{phase}.{n}")));
        for l in 0..=FULL {
            let (a, b) = g(&format!("layer{l}.input"));
            rep.row(tag, &format!("layer{l} input"), &p.inputs[l], &a, &b, gated);
            if l == FULL {
                let (s, a) = r.f32s(&format!("bf16.{phase}.blocks"));
                assert_eq!(s[1], p.blocks.len(), "block count");
                assert_eq!(p.blocks.len(), 1, "layers 0..5 complete one block");
                let b = ref_t("f32", &format!("{phase}.blocks"));
                rep.row(tag, "layer5 blocks", &p.blocks[0], &a, &b, gated);
                for (name, v) in &p.taps {
                    if *name != "out" {
                        let (a, b) = g(name);
                        rep.row(tag, &format!("layer5 {name}"), v, &a, &b, gated);
                    }
                }
            }
            let (a, b) = g(&format!("layer{l}.router_logits"));
            rep.row(tag, &format!("layer{l} router"), &p.layer_logits[l], &a, &b, gated);
            {
                let set = |run: &str| -> Vec<BTreeSet<usize>> {
                    r.i32s(&format!("{run}.{phase}.layer{l}.router_indices"))
                        .chunks(k)
                        .map(|c| c.iter().map(|&x| x as usize).collect())
                        .collect()
                };
                let (r16, r32) = (set("bf16"), set("f32"));
                let ours: Vec<BTreeSet<usize>> = p.layer_experts[l].iter().map(|e| e.iter().copied().collect()).collect();
                let same = |a: &[BTreeSet<usize>], b: &[BTreeSet<usize>]| a.iter().zip(b).filter(|(x, y)| x == y).count();
                let mut flips = Vec::new();
                for (row, (o, u)) in ours.iter().zip(&r32).enumerate() {
                    if o != u {
                        let lo = &b[row * ne..(row + 1) * ne];
                        let mo = &p.layer_logits[l][row * ne..(row + 1) * ne];
                        flips.push(format!(
                            "row {row}: Mary {:?} f32 {:?} (f32 margin {:.2e}, Mary margin {:.2e})",
                            o.difference(u).collect::<Vec<_>>(),
                            u.difference(o).collect::<Vec<_>>(),
                            margin(lo, &bias[l]),
                            margin(mo, &bias[l]),
                        ));
                    }
                }
                rep.lines.push(format!(
                    "{tag:14} layer{l} routing  rows with f32's experts {}/{n}, with bf16's {}/{n} (upstream bf16 vs f32 {}/{n}){}{}",
                    same(&ours, &r32),
                    same(&ours, &r16),
                    same(&r16, &r32),
                    if flips.is_empty() { "" } else { "; " },
                    flips.join("; "),
                    n = ours.len(),
                ));
            }
            let (a, b) = g(&format!("layer{l}.out"));
            rep.row(tag, &format!("layer{l} out"), &p.outs[l], &a, &b, gated);
            if phase != "step" {
                rep.lines.push(format!("{tag:14} layer{l} out rows  {}", worst_rows(&p.outs[l], &b, h)));
            }
        }
    };
    let routing = |rep: &mut Report, tag: &str, phase: &str, p: &Pass| {
        let rset = |run: &str| -> Vec<BTreeSet<i32>> {
            r.i32s(&format!("{run}.{phase}.router_indices"))
                .chunks(t.num_experts_per_tok)
                .map(|c| c.iter().copied().collect())
                .collect()
        };
        let (r16, r32) = (rset("bf16"), rset("f32"));
        let ours: Vec<BTreeSet<i32>> = p.routing.iter().map(|(e, _)| e.iter().map(|&x| x as i32).collect()).collect();
        let overlap = |a: &[BTreeSet<i32>], b: &[BTreeSet<i32>]| -> usize {
            a.iter().zip(b).map(|(x, y)| x.intersection(y).count()).sum()
        };
        let total = ours.len() * t.num_experts_per_tok;
        rep.lines.push(format!(
            "{tag:14} layer5 routing   shared experts with f32 {}/{total}, with bf16 {}/{total}  (upstream bf16 vs f32 {}/{total})  [chain {:.0} ms]",
            overlap(&ours, &r32),
            overlap(&ours, &r16),
            overlap(&r16, &r32),
            p.ms,
        ));
    };
    let caches = |rep: &mut Report, tag: &str, phase: &str, states: &[State], gated: bool| {
        for (l, s) in states.iter().enumerate() {
            let st = |n: &str, conv: bool| {
                let f = |run: &str| {
                    let (shape, v) = r.f32s(&format!("{run}.{phase}.layer{l}.{n}"));
                    if conv { conv_to_rows(&shape, &v) } else { heads_to_rows(&shape, &v) }
                };
                (f("bf16"), f("f32"))
            };
            let (a, b) = st("k", false);
            rep.row(tag, &format!("layer{l} cache k"), &host(s.k.clone().unwrap()), &a, &b, gated);
            let (a, b) = st("v", false);
            rep.row(tag, &format!("layer{l} cache v"), &host(s.v.clone().unwrap()), &a, &b, gated);
            for i in 0..4 {
                let (a, b) = st(&format!("conv{i}"), true);
                rep.row(tag, &format!("layer{l} conv{i}"), &host(s.conv[i].clone()), &a, &b, gated);
            }
        }
    };

    // the prompt and the step token as one prefill, no carried state: the
    // reference the cached step is held to
    let mut states = chain.states(&dev);
    let whole = chain.run(embed_rows(&whole_ids), &mut states, ExpertAct::Bf16);
    compare(&mut rep, "w4a16:whole", "whole", &whole, true);
    routing(&mut rep, "w4a16:whole", "whole", &whole);

    for act in [ExpertAct::Bf16, ExpertAct::Nvfp4] {
        // W4A16 is what upstream computes and is gated; the NVFP4 activation
        // lane is the checkpoint's declared runtime and is reported
        let gated = act == ExpertAct::Bf16;
        let lane = if gated { "w4a16" } else { "w4a4" };
        let mut states = chain.states(&dev);
        for (phase, ids) in [("prefill", &prompt), ("step", &step)] {
            let tag = format!("{lane}:{phase}");
            let p = chain.run(embed_rows(ids), &mut states, act);
            compare(&mut rep, &tag, phase, &p, gated);
            routing(&mut rep, &tag, phase, &p);
            caches(&mut rep, &tag, phase, &states, gated);
            if gated && phase == "step" {
                // the cached step against the replay, Mary against Mary and upstream against upstream
                let n = whole_ids.len();
                for l in [0, FULL] {
                    let last = |v: &[f32]| v[(n - 1) * h..].to_vec();
                    let ours = rel_l2(&p.outs[l], &last(&whole.outs[l]));
                    let up = rel_l2(
                        &ref_t("f32", &format!("step.layer{l}.out")),
                        &last(&ref_t("f32", &format!("whole.layer{l}.out"))),
                    );
                    let up16 = rel_l2(
                        &ref_t("bf16", &format!("step.layer{l}.out")),
                        &last(&ref_t("bf16", &format!("whole.layer{l}.out"))),
                    );
                    rep.lines.push(format!(
                        "cached step vs whole-prefill last row, layer{l} out: Mary {ours:.3e}, upstream f32 {up:.3e}, upstream bf16 {up16:.3e}"
                    ));
                }
            }
        }
    }

    // each layer alone, on upstream's own f32 input of that layer (block
    // states: the f32 normed embedding, or layer 5's recorded blocks), so the
    // error a layer adds is apart from the error it inherits. Reported only:
    // upstream has no bf16 run of a lone layer here to calibrate against.
    for phase in ["prefill", "whole"] {
        let tens = |key: &str| -> Tensor<Bk, 2> {
            let (s, v) = r.f32s(&format!("f32.{phase}.{key}"));
            let n = s[0];
            Tensor::<Bk, 2>::from_data(TensorData::new(v, [n, h]), &dev)
        };
        for l in 0..=FULL {
            let x = tens(&format!("layer{l}.input"));
            let mut blocks = if l == 0 {
                Vec::new()
            } else if l == FULL {
                vec![tens("blocks")]
            } else {
                vec![tens("layer0.input")]
            };
            let out = if l == FULL {
                let mut st = State::new(&chain.full.dims, &dev);
                host(chain.full.forward(x, &mut blocks, &mut st, ExpertAct::Bf16).0)
            } else {
                let mut st = State::new(&chain.sliding[l].dims, &dev);
                host(chain.sliding[l].forward(x, &mut blocks, &mut st, ExpertAct::Bf16).0)
            };
            let b = ref_t("f32", &format!("{phase}.layer{l}.out"));
            let a = ref_t("bf16", &format!("{phase}.layer{l}.out"));
            let tag = format!("alone:{phase}");
            rep.row(&tag, &format!("layer{l} out"), &out, &a, &b, false);
            rep.lines.push(format!("{tag:14} layer{l} out rows  {}", worst_rows(&out, &b, h)));
        }
    }

    for l in &rep.lines {
        println!("{l}");
    }
    println!(
        "gated tensors: worst ratio {:.2} of TOL {TOL}; {} outside; earliest divergence: {}",
        rep.worst,
        rep.fails,
        rep.earliest.as_deref().unwrap_or("none")
    );
    if let Some(p) = args.get(5) {
        std::fs::write(p, format!("[\n{}\n]\n", rep.json.join(",\n"))).expect("receipt");
    }
    std::process::exit(if rep.fails == 0 { 0 } else { 1 });
}
