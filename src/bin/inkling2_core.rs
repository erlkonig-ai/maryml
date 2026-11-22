//! Inkling 2 Mini's looped core (layers 6..17, three passes), natively, against
//! transformers 046bdb8c's own three-pass run.
//!
//! Reads the model from a pile alone ([`MiniRoot`]), like `inkling2_layer0`.
//! The twelve core layers are bound once ([`Core::bind`]) and visited 36 times
//! per token, each visit with its own [`State`] keyed by `(layer, pass)`. The
//! input is the capture's (`golden/capture_inkling2_core.py`): the pre-loop
//! hidden state (upstream's own f32 layer-5 output on the 26-token prompt and
//! its step token, oneDNN off) and the one pre-loop block (the embedding). From
//! there everything is Mary's: every later pass's messages are read from Mary's
//! own previous pass, with each layer's own queries.
//!
//! Phases: the prompt, then the step token through the carried states, then all
//! 27 rows as one prefill with fresh states. For each phase every visit's
//! output, MLP input, messages, gate outputs, routing and (prefill, step) carried
//! state are compared, then the block states at each pass's end and the core's
//! output (hidden and blocks). The tolerance is `inkling2_layer0`'s: a tensor
//! passes when its distance to upstream's f32 run is within `TOL` times
//! upstream's own bf16-vs-f32 spread, both chained through the same visits.
//!
//! Then the structural checks:
//! * the cached step equals the full replay's last row about as closely as
//!   upstream's own step and replay agree;
//! * two planted composition faults are noticed: later passes not restarting
//!   from the pre-loop state, and every layer reading the first layer's queries.
//!
//! usage: inkling2_core <pile> <collection> <root> <reference.safetensors> [receipt.json]

use std::collections::BTreeSet;

use burn::prelude::*;
use burn::tensor::TensorData;
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;

use mary::model_collection::parse_collection_handle;
use mary::models::inkling::seam::Bk;
use mary::models::inkling2_loop::Inkling2LoopConfig;
use mary::models::inkling2_loop::layout::{LayerPart, Slot};
use mary::models::inkling2_loop::loop_core::{Core, CoreState, PassOut, Plant};
use mary::models::inkling2_loop::pile::MiniRoot;
use mary::models::inkling2_loop::schedule::CacheId;
use mary::models::inkling2_loop::sliding::{ExpertAct, Taps, Weights};
use triblespace::prelude::Id;

/// A tensor is within this many times upstream's own bf16-vs-f32 spread.
const TOL: f64 = 1.5;
/// For tensors with no upstream spread (inputs both runs share): f32 round-off.
const F32_FLOOR: f64 = 1e-5;

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

    fn metadata(&self, key: &str) -> String {
        self.meta.metadata().as_ref().and_then(|m| m.get(key).cloned()).unwrap_or_else(|| "(absent)".into())
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

/// Block `k` of an upstream `[t, n, H]` block-state tensor, as `[t, H]`.
fn block_of(v: &[f32], n: usize, k: usize, h: usize) -> Vec<f32> {
    let t = v.len() / (n * h);
    (0..t).flat_map(|row| v[(row * n + k) * h..(row * n + k + 1) * h].iter().copied()).collect()
}

#[derive(Default)]
struct Report {
    lines: Vec<String>,
    json: Vec<String>,
    worst: f64,
    fails: usize,
    rows: usize,
    quiet: bool,
}

impl Report {
    fn row(&mut self, tag: &str, name: &str, mary: &[f32], r16: &[f32], r32: &[f32]) {
        let e32 = rel_l2(mary, r32);
        let e16 = rel_l2(mary, r16);
        let spread = rel_l2(r16, r32);
        let ratio = if spread > 0.0 { e32 / spread } else { e32 / F32_FLOOR * TOL };
        let pass = ratio <= TOL;
        self.worst = self.worst.max(ratio);
        self.rows += 1;
        if !pass {
            self.fails += 1;
        }
        if self.quiet {
            return;
        }
        self.lines.push(format!(
            "{tag:22} {name:16} vs f32 {e32:.3e} (max {:.2e})  vs bf16 {e16:.3e}  upstream spread {spread:.3e}  ratio {ratio:6.2}{}  {}",
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

/// One visit as Mary ran it, on the host.
struct Seen {
    id: CacheId,
    taps: Vec<(&'static str, Vec<f32>)>,
    routing: Vec<BTreeSet<i32>>,
}

fn seen(id: CacheId, out: &Tensor<Bk, 2>, taps: Taps, messages: Option<&[f32]>) -> Seen {
    let mut v = vec![("out", host(out.clone())), ("mlp_input", host(taps.mlp_input)), ("router_logits", taps.router_logits)];
    if let Some([q, k, vv, m]) = taps.gates {
        v.extend([("gate_q", q), ("gate_k", k), ("gate_v", vv), ("gate_mlp", m)]);
    }
    if let Some(m) = messages {
        v.push(("messages", m.to_vec()));
    }
    Seen {
        id,
        taps: v,
        routing: taps.routing.iter().map(|(e, _)| e.iter().map(|&x| x as i32).collect()).collect(),
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
    let h = t.hidden_size;
    let lc = &t.loop_config;
    let r = File::open(reference);
    println!("reference: forward check: {}", r.metadata("forward_check"));
    println!("reference: cpu backend: {}; inputs: {}", r.metadata("cpu_backend"), r.metadata("inputs"));
    let dev = burn::backend::cuda::CudaDevice::default();

    let t0 = std::time::Instant::now();
    let core = Core::bind(&model, t, &dev).expect("bind the core");
    println!(
        "bound core layers {}..{} ({} visits per token) and the message queries in {:.1} s",
        lc.core_start,
        lc.core_end - 1,
        core.passes * core.layers.len(),
        t0.elapsed().as_secs_f64()
    );

    let input = |n: &str| r.f32s(n).1;
    let router_bias: std::collections::BTreeMap<usize, Vec<f32>> = (lc.core_start..lc.core_end)
        .map(|l| {
            let st = Weights::tensor(&model, &Slot::Layer(l, LayerPart::GateBias).tensor_name()).expect("gate bias");
            (l, st.bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        })
        .collect();
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let rows = |v: Vec<f32>, n: usize| Tensor::<Bk, 2>::from_data(TensorData::new(v, [n, h]), &dev);
    let run = |state: &mut CoreState, phase: &str, plant: Option<Plant>| -> (Vec<PassOut>, Vec<Seen>) {
        let x = input(&format!("x.{phase}"));
        let n = x.len() / h;
        let blocks = vec![rows(input(&format!("emb.{phase}")), n)];
        let mut visits = Vec::new();
        let mut hook = |id: CacheId, out: &Tensor<Bk, 2>, taps: Taps, m: Option<&[f32]>| visits.push(seen(id, out, taps, m));
        let t1 = std::time::Instant::now();
        let passes = core.forward(rows(x, n), &blocks, state, ExpertAct::Bf16, 0, plant, Some(&mut hook));
        println!("{phase}{}: {n} rows through 36 visits in {:.0} ms", plant.map(|p| format!(" planted {p:?}")).unwrap_or_default(), t1.elapsed().as_secs_f64() * 1e3);
        (passes, visits)
    };

    let compare = |rep: &mut Report, phase: &str, passes: &[PassOut], visits: &[Seen], state: Option<&CoreState>| {
        for s in visits {
            let tag = format!("p{}.l{}:{phase}", s.id.pass, s.id.layer);
            let up = |run: &str, n: &str| input(&format!("{run}.p{}.l{}.{phase}.{n}", s.id.pass, s.id.layer));
            for (name, v) in &s.taps {
                rep.row(&tag, name, v, &up("bf16", name), &up("f32", name));
            }
            if let Some(st) = state {
                let st = &st.visits[&s.id];
                let base = format!("{{run}}.p{}.l{}.{phase}", s.id.pass, s.id.layer);
                for (name, ours) in [("k", host(st.k.clone().unwrap())), ("v", host(st.v.clone().unwrap()))] {
                    let f = |run: &str| {
                        let (shape, v) = r.f32s(&format!("{}.{name}", base.replace("{run}", run)));
                        heads_to_rows(&shape, &v)
                    };
                    let (a, b) = (f("bf16"), f("f32"));
                    rep.row(&tag, &format!("state {name}"), &ours[ours.len() - b.len()..], &a, &b);
                }
                for i in 0..4 {
                    let f = |run: &str| {
                        let (shape, v) = r.f32s(&format!("{}.conv{i}", base.replace("{run}", run)));
                        conv_to_rows(&shape, &v)
                    };
                    rep.row(&tag, &format!("state conv{i}"), &host(st.conv[i].clone()), &f("bf16"), &f("f32"));
                }
            }
            let rset = |run: &str| -> Vec<BTreeSet<i32>> {
                r.i32s(&format!("{run}.p{}.l{}.{phase}.router_indices", s.id.pass, s.id.layer))
                    .chunks(t.num_experts_per_tok)
                    .map(|c| c.iter().copied().collect())
                    .collect()
            };
            let (r16, r32) = (rset("bf16"), rset("f32"));
            let overlap = |a: &[BTreeSet<i32>], b: &[BTreeSet<i32>]| -> usize {
                a.iter().zip(b).map(|(x, y)| x.intersection(y).count()).sum()
            };
            let total = s.routing.len() * t.num_experts_per_tok;
            if !rep.quiet {
                // where the error sits: rows whose routing differs from f32's, and the per-row ratio of the
                // visit's output (Mary vs f32 over upstream's own bf16-vs-f32, row by row)
                let differ = |a: &[BTreeSet<i32>]| -> Vec<usize> {
                    a.iter().zip(&r32).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect()
                };
                let out = &s.taps[0].1;
                let (o16, o32) = (up("bf16", "out"), up("f32", "out"));
                let per_row: Vec<String> = (0..s.routing.len())
                    .map(|i| {
                        let sl = i * h..(i + 1) * h;
                        let sp = rel_l2(&o16[sl.clone()], &o32[sl.clone()]);
                        format!("{:.2}", rel_l2(&out[sl.clone()], &o32[sl]) / sp.max(1e-12))
                    })
                    .collect();
                // a flip is a near tie when upstream f32's own gap between its 8th and 9th selection score
                // (sigmoid(logit) + bias) is below Mary's largest selection-score error on that row
                let e = t.n_routed_experts;
                let bias = &router_bias[&s.id.layer];
                let l32 = up("f32", "router_logits");
                let ml = &s.taps.iter().find(|(n, _)| *n == "router_logits").unwrap().1;
                let ties: Vec<String> = differ(&s.routing)
                    .into_iter()
                    .map(|row| {
                        let key = |l: &[f32]| -> Vec<f32> { (0..e).map(|x| sig(l[row * e + x]) + bias[x]).collect() };
                        let (k32, km) = (key(&l32), key(ml));
                        let mut sorted = k32.clone();
                        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        let gap = sorted[t.num_experts_per_tok - 1] - sorted[t.num_experts_per_tok];
                        let err = k32.iter().zip(&km).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                        format!("row {row}: gap {gap:.2e} vs Mary key err {err:.2e}{}", if gap <= 2.0 * err { " (tie)" } else { " (NOT a tie)" })
                    })
                    .collect();
                rep.lines.push(format!(
                    "{tag:22} where            routing differs from f32 at rows Mary {:?}, upstream bf16 {:?}; out per-row ratio [{}]; {}",
                    differ(&s.routing),
                    differ(&r16),
                    per_row.join(" "),
                    ties.join("; ")
                ));
                rep.lines.push(format!(
                    "{tag:22} routing          shared experts with f32 {}/{total}, with bf16 {}/{total}  (upstream bf16 vs f32 {}/{total})",
                    overlap(&s.routing, &r32),
                    overlap(&s.routing, &r16),
                    overlap(&r16, &r32),
                ));
                rep.json.push(format!(
                    "{{\"run\":\"{tag}\",\"routing_shared_f32\":{},\"routing_shared_bf16\":{},\"upstream_shared\":{},\"total\":{total}}}",
                    overlap(&s.routing, &r32),
                    overlap(&s.routing, &r16),
                    overlap(&r16, &r32)
                ));
            }
        }
        // the block states at each pass's end, then the core's output
        // upstream's [t, n, H]: n = the pre-loop block, the pass's first input and layer 12's input
        let nb = r.f32s(&format!("f32.p0.{phase}.blocks")).0[1];
        for (p, po) in passes.iter().enumerate() {
            let up = |run: &str| input(&format!("{run}.p{p}.{phase}.blocks"));
            let (u16, u32) = (up("bf16"), up("f32"));
            if po.blocks.len() != nb {
                rep.check(&format!("p{p}:{phase} block count"), false, format!("{} blocks, upstream {nb}", po.blocks.len()));
                continue;
            }
            for (k, b) in po.blocks.iter().enumerate() {
                rep.row(&format!("p{p}:{phase}"), &format!("end block {k}"), &host(b.clone()), &block_of(&u16, nb, k, h), &block_of(&u32, nb, k, h));
            }
        }
        let last = passes.last().unwrap();
        let up = |run: &str, n: &str| input(&format!("{run}.core.{phase}.{n}"));
        rep.row(&format!("core:{phase}"), "hidden", &host(last.hidden.clone()), &up("bf16", "hidden"), &up("f32", "hidden"));
        let (u16, u32) = (up("bf16", "blocks"), up("f32", "blocks"));
        for (k, b) in last.blocks.iter().enumerate().take(nb) {
            rep.row(&format!("core:{phase}"), &format!("block {k}"), &host(b.clone()), &block_of(&u16, nb, k, h), &block_of(&u32, nb, k, h));
        }
    };

    let mut rep = Report::default();
    // prompt, then the step token through the carried states
    let mut state = core.new_state(&dev);
    let (pp, vp) = run(&mut state, "prefill", None);
    compare(&mut rep, "prefill", &pp, &vp, Some(&state));
    let (ps, vs) = run(&mut state, "step", None);
    compare(&mut rep, "step", &ps, &vs, Some(&state));
    assert!(state.visits.values().all(|s| s.pos == 27), "every visit carried all 27 positions");
    assert_eq!(state.visits.len(), 36);
    // all rows at once with fresh states
    let mut fresh = core.new_state(&dev);
    let (pw, vw) = run(&mut fresh, "whole", None);
    compare(&mut rep, "whole", &pw, &vw, None);

    // the cached step against the replay's last row, as closely as upstream's own agree
    let last_row = |v: Vec<f32>| v[v.len() - h..].to_vec();
    for (p, (s, w)) in ps.iter().zip(&pw).enumerate() {
        let ours = rel_l2(&host(s.hidden.clone()), &last_row(host(w.hidden.clone())));
        let up = |run: &str, ph: &str| input(&format!("{run}.p{p}.l{}.{ph}.out", lc.core_end - 1));
        let u32 = rel_l2(&up("f32", "step"), &last_row(up("f32", "whole")));
        let u16 = rel_l2(&up("bf16", "step"), &last_row(up("bf16", "whole")));
        let spread = rel_l2(&up("bf16", "step"), &up("f32", "step"));
        let bound = TOL * u16.max(u32).max(spread);
        rep.check(
            &format!("pass {p} cached step vs replay"),
            ours <= bound,
            format!("Mary step vs whole[26] rel-L2 {ours:.3e}; upstream f32 {u32:.3e}, bf16 {u16:.3e}; bf16-f32 step spread {spread:.3e}; bound {bound:.3e}"),
        );
    }
    let ours: Vec<f32> = vs.iter().map(|s| s.taps[0].1.clone()).flatten().collect();
    let replay: Vec<f32> = vw.iter().map(|s| last_row(s.taps[0].1.clone())).flatten().collect();
    rep.lines.push(format!("all 36 visit outputs, cached step vs replay row 26: rel-L2 {:.3e}", rel_l2(&ours, &replay)));

    // planted faults: each must be noticed by the same comparison
    for plant in [Plant::NoReset, Plant::SharedQuery] {
        let mut st = core.new_state(&dev);
        let (pp, vp) = run(&mut st, "prefill", Some(plant));
        let mut q = Report { quiet: true, ..Report::default() };
        compare(&mut q, "prefill", &pp, &vp, None);
        rep.check(
            &format!("planted {plant:?} noticed"),
            q.fails > 0,
            format!("{} of {} prefill comparisons outside, worst ratio {:.1}", q.fails, q.rows, q.worst),
        );
    }

    for l in &rep.lines {
        println!("{l}");
    }
    println!("tensors: {} compared, worst ratio {:.2} of TOL {TOL}; {} outside or failed", rep.rows, rep.worst, rep.fails);
    if let Some(p) = args.get(5) {
        std::fs::write(p, format!("[\n{}\n]\n", rep.json.join(",\n"))).expect("receipt");
    }
    std::process::exit(if rep.fails == 0 { 0 } else { 1 });
}
