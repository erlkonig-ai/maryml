//! Inkling 2 Mini layer 5, the first full-attention layer, natively, against
//! transformers 046bdb8c's own run.
//!
//! Reads the release directory only as a test input (its `config.json` and
//! layer 5's tensors). Layer 5's inputs are fixed: the hidden state and block
//! states upstream's float32 run produced from layers 0..4 on a 1040-token
//! prompt, recorded by `golden/capture_inkling2_global5.py`, which fed the same
//! inputs to its bfloat16 run. Mary runs layer 5 on them for the prompt, then
//! for one more token through the carried state, and once more as one
//! prefill of both; each twice, at positions from 0 and from the capture's
//! `origin` (where the log scale is no longer 1). Every tap and the carried
//! state are compared.
//!
//! The tolerance is calibrated as for layer 0: a tensor passes when its
//! distance to upstream's f32 run is within `TOL` times upstream's own
//! bf16-vs-f32 distance, here layer 5's rounding alone. Gates are the tensors
//! before the experts; the experts, `mlp_conv`, `out` and the last
//! convolution's state are information, because upstream's runs multiply
//! unquantised activations there and Mary's do not.
//!
//! usage: inkling2_global5 <release dir> <reference.safetensors> [receipt.json]

use std::collections::BTreeSet;

use burn::prelude::*;
use burn::tensor::TensorData;
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;

use mary::models::inkling::seam::Bk;
use mary::models::inkling2_loop::Inkling2LoopConfig;
use mary::models::inkling2_loop::full::{FullLayer, FullTaps};
use mary::models::inkling2_loop::layout::Dtype;
use mary::models::inkling2_loop::sliding::{State, Stored, Taps, Weights};

/// A pass is within this many times upstream's own bf16-vs-f32 spread.
const TOL: f64 = 1.5;
const LAYER: usize = 5;
/// Computed from the experts' output.
const POST_EXPERT: [&str; 4] = ["experts", "mlp_conv", "out", "state conv3"];

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

impl Weights for File {
    fn tensor(&self, name: &str) -> Result<Stored<'_>, String> {
        let (info, bytes) = self.raw(name).ok_or_else(|| format!("release has no {name}"))?;
        let dtype = match info.dtype {
            safetensors::Dtype::BF16 => Dtype::Bf16,
            safetensors::Dtype::F32 => Dtype::F32,
            safetensors::Dtype::U8 => Dtype::U8,
            safetensors::Dtype::F8_E4M3 => Dtype::F8E4M3,
            other => return Err(format!("{name}: {other:?}")),
        };
        Ok(Stored { dtype, shape: info.shape.clone(), bytes })
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

struct Report {
    lines: Vec<String>,
    json: Vec<String>,
    worst: f64,
    fails: usize,
    gates: usize,
    info: usize,
    info_beyond: usize,
    info_worst: f64,
}

impl Report {
    fn row(&mut self, tag: &str, name: &str, mary: &[f32], r16: &[f32], r32: &[f32], gated: bool) {
        let e32 = rel_l2(mary, r32);
        let e16 = rel_l2(mary, r16);
        let spread = rel_l2(r16, r32);
        let ratio = e32 / spread;
        let pass = ratio <= TOL;
        if gated {
            self.gates += 1;
            self.worst = self.worst.max(ratio);
            if !pass {
                self.fails += 1;
            }
        } else {
            self.info += 1;
            self.info_beyond += usize::from(!pass);
            self.info_worst = self.info_worst.max(ratio);
        }
        self.lines.push(format!(
            "{tag:18} {name:14} vs f32 {e32:.3e} (max {:.2e})  vs bf16 {e16:.3e}  upstream spread {spread:.3e}  ratio {ratio:6.2}  {}",
            max_abs(mary, r32),
            match (gated, pass) {
                (true, true) => "gate ok",
                (true, false) => "gate OUTSIDE",
                (false, _) => "info",
            },
        ));
        self.json.push(format!(
            "{{\"run\":\"{tag}\",\"tensor\":\"{name}\",\"rel_l2_vs_f32\":{e32:e},\"rel_l2_vs_bf16\":{e16:e},\"upstream_spread\":{spread:e},\"ratio\":{ratio:e},\"gated\":{gated}}}"
        ));
    }

    /// Two upstream tensors that must agree exactly or the comparison means
    /// nothing (for example the far scenario's inputs against pos0's).
    fn note(&mut self, line: String) {
        self.lines.push(line);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, reference) = (&args[1], &args[2]);
    let cfg = Inkling2LoopConfig::from_json(&std::fs::read_to_string(format!("{dir}/config.json")).unwrap())
        .expect("config");
    let t = &cfg.text_config;
    let release = File::open(&format!("{dir}/model-00001-of-00001.safetensors"));
    let r = File::open(reference);
    let dev = burn::backend::cuda::CudaDevice::default();
    let h = t.hidden_size;
    let origin = r.i32s("origin")[0] as usize;

    let t0 = std::time::Instant::now();
    let layer = FullLayer::bind(&release, t, LAYER, &dev).expect("bind layer 5");
    println!("bound layer {LAYER} in {:.1} s", t0.elapsed().as_secs_f64());

    // the fixed inputs: layer 5's hidden state [t, H] and block states [t, nb, H]
    let input = |scen: &str, phase: &str| -> (Tensor<Bk, 2>, Vec<Tensor<Bk, 2>>) {
        let (s, x) = r.f32s(&format!("input.{scen}.{phase}"));
        let x = Tensor::<Bk, 2>::from_data(TensorData::new(x, [s[0], s[1]]), &dev);
        let (s, b) = r.f32s(&format!("blocks.{scen}.{phase}"));
        let (n, nb) = (s[0], s[1]);
        let b = Tensor::<Bk, 3>::from_data(TensorData::new(b, [n, nb, h]), &dev);
        let blocks = (0..nb).map(|i| b.clone().slice([0..n, i..i + 1, 0..h]).reshape([n, h])).collect();
        (x, blocks)
    };
    let ref_t = |run: &str, key: &str| r.f32s(&format!("{run}.{key}")).1;

    let mut rep = Report { lines: vec![], json: vec![], worst: 0.0, fails: 0, gates: 0, info: 0, info_beyond: 0, info_worst: 0.0 };
    let tap_rows = |taps: FullTaps, out: Tensor<Bk, 2>| -> (Vec<(&'static str, Vec<f32>)>, Vec<f32>, Vec<(Vec<usize>, Vec<f32>)>) {
        let FullTaps { inputs: [qi, ki, vi], rest } = taps;
        let Taps { attn, attn_conv, mlp_input, route, experts, mlp_conv, gates } = rest;
        assert!(gates.is_none(), "layer 5 must not receive cross-pass messages");
        (
            vec![
                ("q_in", host(qi)),
                ("k_in", host(ki)),
                ("v_in", host(vi)),
                ("attn", host(attn)),
                ("attn_conv", host(attn_conv)),
                ("mlp_input", host(mlp_input)),
                ("experts", host(experts)),
                ("mlp_conv", host(mlp_conv)),
                ("out", host(out)),
            ],
            route.logits(),
            route.picks(),
        )
    };

    for (scen, at) in [("pos0", 0usize), ("far", origin)] {
        // one prefill of prompt and step, no carried state: its last row is
        // what the step through the state should compute
        {
            let (x, mut blocks) = input(scen, "whole");
            let n = x.dims()[0];
            let mut state = State::new(&layer.dims, &dev);
            let (out, taps) = layer.forward_at(x, &mut blocks, &mut state, at);
            let (rows, _, _) = tap_rows(taps, out);
            let tag = format!("{scen}:whole");
            for (name, v) in rows {
                let a = ref_t("bf16", &format!("{scen}.whole.{name}"));
                let b = ref_t("f32", &format!("{scen}.whole.{name}"));
                rep.row(&tag, name, &v, &a, &b, !POST_EXPERT.contains(&name));
                if name == "out" {
                    let last = |v: &[f32]| v[(n - 1) * h..].to_vec();
                    let s = ref_t("f32", &format!("{scen}.step.out"));
                    rep.note(format!(
                        "{tag:18} upstream whole-vs-step last row (f32) rel {:.3e}",
                        rel_l2(&last(&b), &s)
                    ));
                }
            }
        }
        let mut state = State::new(&layer.dims, &dev);
        for phase in ["prefill", "step"] {
            let tag = format!("cached:{scen}:{phase}");
            let (x, mut blocks) = input(scen, phase);
            let t1 = std::time::Instant::now();
            let (out, taps) = layer.forward_at(x, &mut blocks, &mut state, at);
            let (rows, router_logits, routing) = tap_rows(taps, out);
            let ms = t1.elapsed().as_secs_f64() * 1e3;
            let g = |n: &str| (ref_t("bf16", &format!("{scen}.{phase}.{n}")), ref_t("f32", &format!("{scen}.{phase}.{n}")));
            for (name, v) in rows {
                let (a, b) = g(name);
                rep.row(&tag, name, &v, &a, &b, !POST_EXPERT.contains(&name));
            }
            let (a, b) = g("router_logits");
            rep.row(&tag, "router_logits", &router_logits, &a, &b, true);
            let st = |n: &str, conv: bool| {
                let f = |run: &str| {
                    let (shape, v) = r.f32s(&format!("{run}.{scen}.{phase}.{n}"));
                    if conv { conv_to_rows(&shape, &v) } else { heads_to_rows(&shape, &v) }
                };
                (f("bf16"), f("f32"))
            };
            let (a, b) = st("k", false);
            rep.row(&tag, "state k", &host(state.kv.keys().unwrap()), &a, &b, true);
            let (a, b) = st("v", false);
            rep.row(&tag, "state v", &host(state.kv.values().unwrap()), &a, &b, true);
            for i in 0..4 {
                let (a, b) = st(&format!("conv{i}"), true);
                let name = format!("state conv{i}");
                rep.row(&tag, &name, &host(state.conv[i].clone()), &a, &b, !POST_EXPERT.contains(&name.as_str()));
            }
            let rset = |run: &str| -> Vec<BTreeSet<i32>> {
                r.i32s(&format!("{run}.{scen}.{phase}.router_indices"))
                    .chunks(t.num_experts_per_tok)
                    .map(|c| c.iter().copied().collect())
                    .collect()
            };
            let (r16, r32) = (rset("bf16"), rset("f32"));
            let ours: Vec<BTreeSet<i32>> =
                routing.iter().map(|(e, _)| e.iter().map(|&x| x as i32).collect()).collect();
            let overlap = |a: &[BTreeSet<i32>], b: &[BTreeSet<i32>]| -> usize {
                a.iter().zip(b).map(|(x, y)| x.intersection(y).count()).sum()
            };
            let total = ours.len() * t.num_experts_per_tok;
            rep.note(format!(
                "{tag:18} routing        shared experts with f32 {}/{total}, with bf16 {}/{total}  (upstream bf16 vs f32 {}/{total})  [{ms:.0} ms]",
                overlap(&ours, &r32),
                overlap(&ours, &r16),
                overlap(&r16, &r32),
            ));
        }
    }
    // the log scale must matter at the far origin, or the far rows test nothing
    let a = ref_t("f32", "pos0.prefill.attn");
    let b = ref_t("f32", "far.prefill.attn");
    rep.note(format!("upstream attn, far vs pos0 (f32 prefill): rel {:.3e}", rel_l2(&b, &a)));

    for l in &rep.lines {
        println!("{l}");
    }
    println!(
        "gates: {} compared, worst ratio {:.2} of TOL {TOL}; {} outside. information: {} rows, {} beyond {TOL}, worst ratio {:.2}",
        rep.gates, rep.worst, rep.fails, rep.info, rep.info_beyond, rep.info_worst
    );
    if let Some(p) = args.get(3) {
        std::fs::write(p, format!("[\n{}\n]\n", rep.json.join(",\n"))).expect("receipt");
    }
    std::process::exit(if rep.fails == 0 { 0 } else { 1 });
}
