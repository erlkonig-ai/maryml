//! Inkling 2 Mini layer 0, natively, against transformers 046bdb8c's own run.
//!
//! Reads the model from a pile alone: the pile, the exact model collection
//! and the model root, opened as a [`MiniRoot`] (config from the root's
//! facts, layer 0's tensors as views over the pile mapping). Computes the
//! normed embedding and one visit of layer 0 on the device for the prompt and
//! then for one more token through the carried state, and compares every tap
//! with the capture that `golden/capture_inkling2_layer0.py` made from
//! upstream's code.
//!
//! The tolerance is calibrated, not chosen: upstream ran twice, at bf16 (the
//! shipped precision) and at f32 on the same weights. Their distance is how
//! far rounding alone moves each tensor. A Mary tensor passes when its
//! distance to upstream's f32 run is within `TOL` times that spread.
//!
//! usage: inkling2_layer0 <pile> <collection> <root> <reference.safetensors> [receipt.json]

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
use mary::models::inkling2_loop::layout::Slot;
use mary::models::inkling2_loop::pile::MiniRoot;
use mary::models::inkling2_loop::sliding::{ExpertAct, SlidingLayer, State, Taps, Weights};
use triblespace::prelude::Id;

/// A pass is within this many times upstream's own bf16-vs-f32 spread.
const TOL: f64 = 1.5;

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
/// `[k - 1, C]` rows Mary carries: only the last `k - 1` are ever read.
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

struct Report {
    lines: Vec<String>,
    json: Vec<String>,
    worst: f64,
    fails: usize,
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
            }
        }
        self.lines.push(format!(
            "{tag:10} {name:14} vs f32 {e32:.3e} (max {:.2e})  vs bf16 {e16:.3e}  upstream spread {spread:.3e}  ratio {ratio:6.2}  {}",
            max_abs(mary, r32),
            if !gated { "report" } else if pass { "ok" } else { "OUTSIDE" },
        ));
        self.json.push(format!(
            "{{\"run\":\"{tag}\",\"tensor\":\"{name}\",\"rel_l2_vs_f32\":{e32:e},\"rel_l2_vs_bf16\":{e16:e},\"upstream_spread\":{spread:e},\"ratio\":{ratio:e},\"gated\":{gated}}}"
        ));
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

    let prompt: Vec<usize> = r.i32s("prompt").iter().map(|&x| x as usize).collect();
    let step: Vec<usize> = r.i32s("step_token").iter().map(|&x| x as usize).collect();
    let h = t.hidden_size;

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

    let t0 = std::time::Instant::now();
    let layer = SlidingLayer::bind(&model, t, 0, &dev).expect("bind layer 0");
    println!("bound layer 0 in {:.1} s", t0.elapsed().as_secs_f64());

    let mut rep = Report { lines: vec![], json: vec![], worst: 0.0, fails: 0 };
    let xp = embed_rows(&prompt);
    let xs = embed_rows(&step);
    let ref_in = |run: &str, n: &str| r.f32s(&format!("{run}.{n}")).1;
    rep.row("embed", "prompt", &host(xp.clone()), &ref_in("bf16", "input"), &ref_in("f32", "input"), true);
    rep.row("embed", "step", &host(xs.clone()), &ref_in("bf16", "step_input"), &ref_in("f32", "step_input"), true);

    // prompt and step token as one prefill, no carried state: upstream's
    // capture shows its step equals this last row, so it separates a step-path
    // fault from a layer fault
    {
        let mut state = State::new(&layer.dims, &dev);
        let mut whole_ids = prompt.clone();
        whole_ids.extend(&step);
        let n = whole_ids.len();
        let (out, taps) = layer.forward(embed_rows(&whole_ids), &mut Vec::new(), &mut state, ExpertAct::Bf16);
        let last = |v: Vec<f32>| v[(n - 1) * h..].to_vec();
        for (name, v) in [
            ("attn", host(taps.attn)),
            ("attn_conv", host(taps.attn_conv)),
            ("mlp_input", host(taps.mlp_input)),
            ("mlp_conv", host(taps.mlp_conv)),
            ("out", host(out)),
        ] {
            let f = |run: &str| last(ref_in(run, &format!("whole.{name}")));
            rep.row("w4a16:whole", name, &last(v), &f("bf16"), &f("f32"), true);
        }
    }

    for act in [ExpertAct::Bf16, ExpertAct::Nvfp4] {
        // the W4A16 lane is what upstream computes and is gated; the NVFP4
        // activation lane is the checkpoint's declared runtime with a different
        // scale rule, and is reported
        let gated = act == ExpertAct::Bf16;
        let mut state = State::new(&layer.dims, &dev);
        for (phase, x) in [("prefill", xp.clone()), ("step", xs.clone())] {
            let tag = format!("{}:{phase}", if gated { "w4a16" } else { "w4a4" });
            let mut blocks = Vec::new();
            let t1 = std::time::Instant::now();
            let (out, taps) = layer.forward(x, &mut blocks, &mut state, act);
            let Taps { attn, attn_conv, mlp_input, router_logits, routing, experts, mlp_conv, .. } = taps;
            let out = host(out);
            let ms = t1.elapsed().as_secs_f64() * 1e3;
            let g = |n: &str| (ref_in("bf16", &format!("{phase}.{n}")), ref_in("f32", &format!("{phase}.{n}")));
            for (name, v) in [
                ("attn", host(attn)),
                ("attn_conv", host(attn_conv)),
                ("mlp_input", host(mlp_input)),
                ("router_logits", router_logits),
                ("experts", host(experts)),
                ("mlp_conv", host(mlp_conv)),
                ("out", out),
            ] {
                let (a, b) = g(name);
                rep.row(&tag, name, &v, &a, &b, gated);
            }
            // carried state
            let st = |n: &str, conv: bool| {
                let f = |run: &str| {
                    let (shape, v) = r.f32s(&format!("{run}.{phase}.{n}"));
                    if conv { conv_to_rows(&shape, &v) } else { heads_to_rows(&shape, &v) }
                };
                (f("bf16"), f("f32"))
            };
            let (a, b) = st("k", false);
            rep.row(&tag, "state k", &host(state.k.clone().unwrap()), &a, &b, gated);
            let (a, b) = st("v", false);
            rep.row(&tag, "state v", &host(state.v.clone().unwrap()), &a, &b, gated);
            for i in 0..4 {
                let (a, b) = st(&format!("conv{i}"), true);
                rep.row(&tag, &format!("state conv{i}"), &host(state.conv[i].clone()), &a, &b, gated);
            }
            // routing, as sets per token
            let rset = |run: &str| -> Vec<BTreeSet<i32>> {
                r.i32s(&format!("{run}.{phase}.router_indices"))
                    .chunks(t.num_experts_per_tok)
                    .map(|c| c.iter().copied().collect())
                    .collect()
            };
            let (r16, r32) = (rset("bf16"), rset("f32"));
            let ours: Vec<BTreeSet<i32>> = routing.iter().map(|(e, _)| e.iter().map(|&x| x as i32).collect()).collect();
            let overlap = |a: &[BTreeSet<i32>], b: &[BTreeSet<i32>]| -> usize {
                a.iter().zip(b).map(|(x, y)| x.intersection(y).count()).sum()
            };
            let total = ours.len() * t.num_experts_per_tok;
            rep.lines.push(format!(
                "{tag:10} routing        shared experts with f32 {}/{total}, with bf16 {}/{total}  (upstream bf16 vs f32 {}/{total})  [{ms:.0} ms]",
                overlap(&ours, &r32),
                overlap(&ours, &r16),
                overlap(&r16, &r32),
            ));
        }
    }
    for l in &rep.lines {
        println!("{l}");
    }
    println!(
        "gated tensors: worst ratio {:.2} of TOL {TOL}; {} outside",
        rep.worst, rep.fails
    );
    if let Some(p) = args.get(5) {
        std::fs::write(p, format!("[\n{}\n]\n", rep.json.join(",\n"))).expect("receipt");
    }
    std::process::exit(if rep.fails == 0 { 0 } else { 1 });
}
