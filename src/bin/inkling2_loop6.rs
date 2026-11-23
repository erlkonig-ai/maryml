//! Inkling 2 Mini core layer 6, natively, on its first pass and on a later
//! pass that reads cross-pass messages, against transformers 046bdb8c's own run.
//!
//! Reads the model from a pile alone ([`MiniRoot`]), like `inkling2_layer0`.
//! Layer 6 is bound once; its two visits keep their own [`State`], keyed by
//! [`CacheId`] `(6, 0)` and `(6, 1)`. The inputs are the capture's
//! (`golden/capture_inkling2_loop6.py`): layer 6's real input from upstream's
//! layer-5 run on a 1040-token prompt plus one step token, its block states,
//! and two synthetic previous-pass states A and B. The messages are computed
//! here from the previous-pass states with [`read_messages`] and the release's
//! own cross-pass queries ([`MessageQueries::from_checkpoint`]), and compared
//! with upstream's `message_input` before they are used.
//!
//! For each visit and phase (prompt, then the step token through the carried
//! state, and all 1041 rows as one prefill with no state) every tap, the
//! routing and the carried state are compared, with the tolerance of
//! `inkling2_layer0`: a Mary tensor passes when its distance to upstream's f32
//! run is within `TOL` times upstream's own bf16-vs-f32 spread.
//!
//! Then the two structural checks the slice is for:
//! * the first pass is independent of the messages: its output and carried
//!   state are bit-identical whether the second pass then reads A or B;
//! * the gated visit follows them: B moves the second pass's output, and by
//!   about as much as it moves upstream's.
//!
//! usage: inkling2_loop6 <pile> <collection> <root> <reference.safetensors> [receipt.json]

use std::collections::{BTreeMap, BTreeSet};

use burn::prelude::*;
use burn::tensor::TensorData;
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;

use mary::model_collection::parse_collection_handle;
use mary::models::inkling::seam::Bk;
use mary::models::inkling2_loop::Inkling2LoopConfig;
use mary::models::inkling2_loop::layout::Slot;
use mary::models::inkling2_loop::messages::{MessageQueries, read_messages};
use mary::models::inkling2_loop::pile::MiniRoot;
use mary::models::inkling2_loop::schedule::CacheId;
use mary::models::inkling2_loop::sliding::{ExpertAct, SlidingLayer, State, Weights};
use triblespace::prelude::Id;

/// A pass is within this many times upstream's own bf16-vs-f32 spread.
const TOL: f64 = 1.5;
/// Upstream's message read runs in f32 in both runs (`.float()` before the
/// norm, f32 queries) on the same bf16-rounded inputs, so its spread is zero.
/// A tensor with no spread passes within f32 round-off of a 2304-wide sum.
const F32_FLOOR: f64 = 1e-5;
const LAYER: usize = 6;

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

/// The rows Mary and upstream both still hold of a carried K or V cache: both
/// end at the visit's last position, so the matching absolute positions are the
/// last `min(ours, upstream)` rows of each. Mary's `attend::Kv` evicts whole tiles and
/// upstream keeps the window, so either side may hold more.
fn matching_suffix<'a>(ours: &'a [f32], a: &'a [f32], b: &'a [f32], width: usize) -> (&'a [f32], &'a [f32], &'a [f32], usize) {
    assert_eq!(a.len(), b.len(), "the two upstream runs hold the same rows");
    assert!(ours.len() % width == 0 && b.len() % width == 0, "rows of width {width}");
    let n = ours.len().min(b.len()) / width;
    assert!(n > 0, "no position held by both");
    let tail = |v: &'a [f32]| &v[v.len() - n * width..];
    (tail(ours), tail(a), tail(b), n)
}

/// A visit's held keys and values on the host, checked to end at its position.
fn held_kv(st: &State) -> (Vec<f32>, Vec<f32>) {
    let kv = &st.kv;
    assert_eq!(kv.base() + kv.rows(), st.pos, "the cache ends at the visit's position");
    let k = kv.keys().expect("a carried visit holds keys");
    let v = kv.values().expect("a carried visit holds values");
    assert_eq!(k.dims()[0], kv.rows());
    assert_eq!(v.dims()[0], kv.rows());
    (host(k), host(v))
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
    fn row(&mut self, tag: &str, name: &str, mary: &[f32], r16: &[f32], r32: &[f32]) {
        let e32 = rel_l2(mary, r32);
        let e16 = rel_l2(mary, r16);
        let spread = rel_l2(r16, r32);
        let ratio = if spread > 0.0 { e32 / spread } else { e32 / F32_FLOOR * TOL };
        let pass = ratio <= TOL;
        self.worst = self.worst.max(ratio);
        if !pass {
            self.fails += 1;
        }
        self.lines.push(format!(
            "{tag:14} {name:14} vs f32 {e32:.3e} (max {:.2e})  vs bf16 {e16:.3e}  upstream spread {spread:.3e}  ratio {ratio:6.2}{}  {}",
            max_abs(mary, r32),
            if spread > 0.0 { "" } else { " (f32 floor)" },
            if pass { "ok" } else { "OUTSIDE" },
        ));
        self.json.push(format!(
            "{{\"run\":\"{tag}\",\"tensor\":\"{name}\",\"rel_l2_vs_f32\":{e32:e},\"rel_l2_vs_bf16\":{e16:e},\"upstream_spread\":{spread:e},\"ratio\":{ratio:e}}}"
        ));
    }

    fn check(&mut self, what: &str, ok: bool, detail: String) {
        if !ok {
            self.fails += 1;
        }
        self.lines.push(format!("check {what}: {detail}  {}", if ok { "ok" } else { "FAILED" }));
        self.json.push(format!("{{\"check\":\"{what}\",\"ok\":{ok},\"detail\":\"{detail}\"}}"));
    }
}

/// What one visit produced, on the host.
struct Seen {
    taps: Vec<(&'static str, Vec<f32>)>,
    routing: Vec<BTreeSet<i32>>,
    k: Vec<f32>,
    v: Vec<f32>,
    conv: Vec<Vec<f32>>,
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
    let h = t.hidden_size;
    let core = t.loop_config.core_end - t.loop_config.core_start;
    let r = File::open(reference);
    let dev = burn::backend::cuda::CudaDevice::default();

    let t0 = std::time::Instant::now();
    let layer = SlidingLayer::bind(&model, t, LAYER, &dev).expect("bind layer 6");
    assert!(layer.reads_messages() && layer.is_block_boundary());
    let f32_tensor = |s: Slot| -> Vec<f32> {
        let st = Weights::tensor(&model, &s.tensor_name()).unwrap();
        st.bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let queries = MessageQueries::from_checkpoint(
        &f32_tensor(Slot::CrossPassQueries),
        &f32_tensor(Slot::CrossPassKvQueries),
        core,
        h,
    )
    .expect("message queries");
    let q6 = queries.layer(LAYER - t.loop_config.core_start);
    println!("bound layer {LAYER} and its message gates in {:.1} s", t0.elapsed().as_secs_f64());

    let input = |n: &str| r.f32s(n).1;
    let rows = |v: Vec<f32>, n: usize| Tensor::<Bk, 2>::from_data(TensorData::new(v, [n, h]), &dev);
    let mut rep = Report { lines: vec![], json: vec![], worst: 0.0, fails: 0 };

    // One visit: pass 0 reads nothing, pass 1 reads the messages of `prev`.
    // Each visit starts from the pre-loop block states, as upstream's passes do.
    let run = |states: &mut BTreeMap<CacheId, State>, pass: usize, prev: &str, phase: &str, rep: &mut Report| -> Seen {
        let x = input(&format!("x.{phase}"));
        let n = x.len() / h;
        let mut blocks = vec![rows(input(&format!("blocks.{phase}")), n)];
        let messages = (pass > 0).then(|| {
            let m = read_messages(
                q6,
                &input(&format!("prev{prev}_block.{phase}")),
                &input(&format!("prev{prev}_partial.{phase}")),
                n,
                1,
                h,
            );
            let up = |run: &str| input(&format!("{run}.p1{prev}.{phase}.messages"));
            rep.row(&format!("p1{prev}:{phase}"), "messages", &m, &up("bf16"), &up("f32"));
            Tensor::<Bk, 3>::from_data(TensorData::new(m, [n, 4, h]), &dev)
        });
        let state = states.entry(CacheId { layer: LAYER, pass }).or_insert_with(|| State::new(&layer.dims, &dev));
        let (out, taps) = layer.forward_visit(rows(x, n), &mut blocks, state, ExpertAct::Bf16, messages.as_ref());
        assert_eq!(blocks.len(), 2, "layer 6 is a block boundary: its input joins the blocks once");
        let mut seen = vec![
            ("attn", host(taps.attn)),
            ("attn_conv", host(taps.attn_conv)),
            ("mlp_input", host(taps.mlp_input)),
            ("router_logits", taps.router_logits),
            ("experts", host(taps.experts)),
            ("mlp_conv", host(taps.mlp_conv)),
            ("out", host(out)),
        ];
        assert_eq!(taps.gates.is_some(), pass > 0);
        if let Some([q, k, v, m]) = taps.gates {
            seen.extend([("gate_q", host(q)), ("gate_k", host(k)), ("gate_v", host(v)), ("gate_mlp", host(m))]);
        }
        let (k, v) = held_kv(state);
        Seen {
            taps: seen,
            routing: taps.routing.iter().map(|(e, _)| e.iter().map(|&x| x as i32).collect()).collect(),
            k,
            v,
            conv: state.conv.iter().map(|c| host(c.clone())).collect(),
        }
    };

    let compare = |rep: &mut Report, seen: &Seen, visit: &str, phase: &str, carried: bool| {
        let tag = format!("{visit}:{phase}");
        let up = |run: &str, n: &str| input(&format!("{run}.{visit}.{phase}.{n}"));
        for (name, v) in &seen.taps {
            rep.row(&tag, name, v, &up("bf16", name), &up("f32", name));
        }
        if carried {
            // upstream's sliding layer keeps at most the window and Mary's cache
            // drops whole tiles, so compare the positions both still hold
            for (name, ours) in [("k", &seen.k), ("v", &seen.v)] {
                let f = |run: &str| {
                    let (shape, v) = r.f32s(&format!("{run}.{visit}.{phase}.{name}"));
                    heads_to_rows(&shape, &v)
                };
                let (a, b) = (f("bf16"), f("f32"));
                let width = t.swa_num_key_value_heads * t.swa_head_dim;
                let (ours, a, b, n) = matching_suffix(ours, &a, &b, width);
                rep.row(&tag, &format!("state {name} ({n}r)"), ours, a, b);
            }
            for (i, ours) in seen.conv.iter().enumerate() {
                let f = |run: &str| {
                    let (shape, v) = r.f32s(&format!("{run}.{visit}.{phase}.conv{i}"));
                    conv_to_rows(&shape, &v)
                };
                rep.row(&tag, &format!("state conv{i}"), ours, &f("bf16"), &f("f32"));
            }
        }
        let rset = |run: &str| -> Vec<BTreeSet<i32>> {
            r.i32s(&format!("{run}.{visit}.{phase}.router_indices"))
                .chunks(t.num_experts_per_tok)
                .map(|c| c.iter().copied().collect())
                .collect()
        };
        let (r16, r32) = (rset("bf16"), rset("f32"));
        let overlap = |a: &[BTreeSet<i32>], b: &[BTreeSet<i32>]| -> usize {
            a.iter().zip(b).map(|(x, y)| x.intersection(y).count()).sum()
        };
        let total = seen.routing.len() * t.num_experts_per_tok;
        rep.lines.push(format!(
            "{tag:14} routing        shared experts with f32 {}/{total}, with bf16 {}/{total}  (upstream bf16 vs f32 {}/{total})",
            overlap(&seen.routing, &r32),
            overlap(&seen.routing, &r16),
            overlap(&r16, &r32),
        ));
        rep.json.push(format!(
            "{{\"run\":\"{tag}\",\"routing_shared_f32\":{},\"routing_shared_bf16\":{},\"upstream_shared\":{},\"total\":{total}}}",
            overlap(&seen.routing, &r32),
            overlap(&seen.routing, &r16),
            overlap(&r16, &r32)
        ));
    };

    // scenario A, then B on fresh states; pass 0 before pass 1 in each phase
    let mut first: BTreeMap<String, Seen> = BTreeMap::new();
    let mut second: BTreeMap<String, Seen> = BTreeMap::new();
    for prev in ["A", "B"] {
        let mut states: BTreeMap<CacheId, State> = BTreeMap::new();
        for phase in ["prefill", "step"] {
            let t1 = std::time::Instant::now();
            let p0 = run(&mut states, 0, prev, phase, &mut rep);
            let p1 = run(&mut states, 1, prev, phase, &mut rep);
            let ms = t1.elapsed().as_secs_f64() * 1e3;
            if prev == "A" {
                compare(&mut rep, &p0, "p0", phase, true);
            }
            compare(&mut rep, &p1, &format!("p1{prev}"), phase, true);
            rep.lines.push(format!("{prev}:{phase} both visits [{ms:.0} ms]"));
            first.insert(format!("{prev}.{phase}"), p0);
            second.insert(format!("{prev}.{phase}"), p1);
        }
        assert_eq!(states.len(), 2);
        // all 1041 rows at once, no carried state: separates a step-path fault
        let mut fresh: BTreeMap<CacheId, State> = BTreeMap::new();
        let p0 = run(&mut fresh, 0, prev, "whole", &mut rep);
        let p1 = run(&mut fresh, 1, prev, "whole", &mut rep);
        if prev == "A" {
            compare(&mut rep, &p0, "p0", "whole", false);
        }
        compare(&mut rep, &p1, &format!("p1{prev}"), "whole", false);
    }

    // the first pass does not see what a later pass reads
    for phase in ["prefill", "step"] {
        let (a, b) = (&first[&format!("A.{phase}")], &first[&format!("B.{phase}")]);
        let same = a.taps == b.taps && a.k == b.k && a.v == b.v && a.conv == b.conv;
        rep.check(
            &format!("pass 0 independent of messages ({phase})"),
            same,
            "output, taps and carried state bit-identical under A and B".into(),
        );
    }
    // the gated visit follows its messages as upstream's does: Mary's change
    // from A to B, held to upstream's change by the same calibrated tolerance
    for phase in ["prefill", "step"] {
        let out = |m: &BTreeMap<String, Seen>, s: &str| -> Vec<f32> {
            m[&format!("{s}.{phase}")].taps.iter().find(|(n, _)| *n == "out").unwrap().1.clone()
        };
        let up = |run: &str, v: &str| input(&format!("{run}.{v}.{phase}.out"));
        let diff = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
        let ours = diff(&out(&second, "B"), &out(&second, "A"));
        let d32 = diff(&up("f32", "p1B"), &up("f32", "p1A"));
        let d16 = diff(&up("bf16", "p1B"), &up("bf16", "p1A"));
        rep.row(&format!("p1B-p1A:{phase}"), "out change", &ours, &d16, &d32);
        let moved = rel_l2(&out(&second, "B"), &out(&second, "A"));
        let up_moved = rel_l2(&up("f32", "p1B"), &up("f32", "p1A"));
        let gating = rel_l2(&out(&second, "A"), &out(&first, "A"));
        let up_gating = rel_l2(&up("f32", "p1A"), &up("f32", "p0"));
        rep.check(
            &format!("pass 1 follows messages ({phase})"),
            moved > 0.0,
            format!(
                "B vs A out rel-L2 {moved:.3e} (upstream {up_moved:.3e}); gated vs first pass {gating:.3e} (upstream {up_gating:.3e})"
            ),
        );
    }

    for l in &rep.lines {
        println!("{l}");
    }
    println!("tensors: worst ratio {:.2} of TOL {TOL}; {} outside or failed", rep.worst, rep.fails);
    if let Some(p) = args.get(5) {
        std::fs::write(p, format!("[\n{}\n]\n", rep.json.join(",\n"))).expect("receipt");
    }
    std::process::exit(if rep.fails == 0 { 0 } else { 1 });
}
