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
//! Mary's experts quantise their activations as the model was trained and
//! upstream's runs do not, so from the first visit's experts on the chained
//! run differs by intention. Labels, by the input of what is compared:
//!
//! **Gates**
//! * upstream, chained: pass 0 of layer 6, whose input is upstream's, before its
//!   experts (MLP input, router logits, keys, values, conv states 0..2).
//! * upstream, replayed: every visit's layer alone on upstream's recorded f32
//!   operands (hidden state, blocks, messages), before its experts. The spread
//!   is upstream's chained one, so looser than a same-input spread.
//! * the same schedule (prompt, then step) repeated from fresh states gives
//!   every tap, routing and carried state bit for bit.
//! * positions and cache extents; layer 6's cached rows are the one-prefill
//!   run's rows, bit for bit and in order, and the step's row lands on row 26.
//! * masks: with the same call shape, another last row leaves every visit's
//!   earlier rows bit-identical; another first row moves the last one, and moves
//!   the cached step.
//! * composition, exact: every pass ends with the pre-loop blocks bit for bit
//!   and with the block count, and each later pass's messages are the previous
//!   pass's states read with that layer's own queries, bit for bit. The two
//!   planted faults must fail these same two gates.
//! * every tap finite.
//!
//! **Information**: every other upstream comparison, and the cached step against
//! the one-prefill run. Those two take different kernels (one query against 27),
//! which differ by rounding before the experts; FP4 thresholds turn such a
//! difference into a step. The bound that once gated it is printed beside it.
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
use mary::models::inkling2_loop::loop_core::{Core, CoreLayer, CoreState, PassOut, Plant};
use mary::models::inkling2_loop::messages::{DeviceQueries, MessageSources};
use mary::models::inkling2_loop::pile::MiniRoot;
use mary::models::inkling2_loop::schedule::CacheId;
use mary::models::inkling2_loop::sliding::{State, Taps, Weights};
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
    gates: usize,
    checks: usize,
    info: usize,
    info_beyond: usize,
    info_worst: f64,
}

impl Report {
    fn row(&mut self, gated: bool, tag: &str, name: &str, mary: &[f32], r16: &[f32], r32: &[f32]) {
        let e32 = rel_l2(mary, r32);
        let e16 = rel_l2(mary, r16);
        let spread = rel_l2(r16, r32);
        let ratio = if spread > 0.0 { e32 / spread } else { e32 / F32_FLOOR * TOL };
        let pass = ratio <= TOL;
        if gated {
            self.gates += 1;
            self.worst = self.worst.max(ratio);
            self.fails += usize::from(!pass);
        } else {
            self.info += 1;
            self.info_beyond += usize::from(!pass);
            self.info_worst = self.info_worst.max(ratio);
        }
        self.lines.push(format!(
            "{tag:22} {name:16} vs f32 {e32:.3e} (max {:.2e})  vs bf16 {e16:.3e}  upstream spread {spread:.3e}  ratio {ratio:6.2}{}  {}",
            max_abs(mary, r32),
            if spread > 0.0 { "" } else { " (f32 floor)" },
            match (gated, pass) {
                (true, true) => "gate ok",
                (true, false) => "gate OUTSIDE",
                (false, _) => "info",
            },
        ));
        self.json.push(format!(
            "{{\"run\":\"{tag}\",\"tensor\":\"{name}\",\"gated\":{gated},\"rel_l2_vs_f32\":{e32:e},\"rel_l2_vs_bf16\":{e16:e},\"upstream_spread\":{spread:e},\"ratio\":{ratio:e}}}"
        ));
    }

    fn check(&mut self, what: &str, ok: bool, detail: String) {
        self.checks += 1;
        if !ok {
            self.fails += 1;
        }
        self.lines.push(format!("check {what}: {detail}  gate {}", if ok { "ok" } else { "FAILED" }));
        self.json.push(format!("{{\"check\":\"{what}\",\"ok\":{ok},\"detail\":\"{detail}\"}}"));
    }
}

/// The taps of the first visit that its given input determines.
const PRE_EXPERT: [&str; 2] = ["mlp_input", "router_logits"];

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Every visit's carried state on the host: position, keys, values, conv rows.
fn carried(st: &CoreState) -> Vec<(usize, usize, Vec<Vec<f32>>)> {
    st.visits
        .values()
        .map(|s| {
            let (k, v) = held_kv(s);
            let mut t = vec![k, v];
            t.extend(s.conv.iter().map(|c| host(c.clone())));
            (s.pos, s.kv.base(), t)
        })
        .collect()
}

/// The visits whose taps or routing are not the same bits, of two runs of one schedule.
fn unequal(a: &[Seen], b: &[Seen]) -> Vec<CacheId> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| {
            x.routing != y.routing
                || x.taps.len() != y.taps.len()
                || x.taps.iter().zip(&y.taps).any(|((n, p), (m, q))| n != m || !same_bits(p, q))
        })
        .map(|(x, _)| x.id)
        .collect()
}

/// What [`Core::forward`] must hold exactly, whatever the arithmetic: `None`, or
/// what a run broke. `x` and `emb` are its input and pre-loop block.
fn composition(
    core: &Core,
    passes: &[PassOut],
    visits: &[Seen],
    x: &[f32],
    emb: &[f32],
    dev: &burn::backend::cuda::CudaDevice,
) -> Vec<(&'static str, String)> {
    let mut broken = Vec::new();
    let queries = DeviceQueries::new(&core.queries, dev);
    for (p, po) in passes.iter().enumerate() {
        let ends = po.blocks.len() == 3 && same_bits(&host(po.blocks[0].clone()), emb) && same_bits(&host(po.blocks[1].clone()), x);
        if !ends {
            broken.push(("restart", format!("pass {p} ends with {} blocks, the first two not the embedding and the pre-loop state", po.blocks.len())));
        }
    }
    let first = core.layers[0].layer();
    for s in visits.iter().filter(|s| s.id.pass > 0) {
        let prev = &passes[s.id.pass - 1];
        let keep = &prev.blocks[2.min(prev.blocks.len())..];
        let want = MessageSources::new(keep, prev.hidden.clone()).read(queries.layer(s.id.layer - first));
        let [t, j, h] = want.dims();
        let want = host(want.reshape([t * j, h]));
        let got = &s.taps.iter().find(|(n, _)| *n == "messages").expect("a later pass reads messages").1;
        if !same_bits(got, &want) {
            broken.push(("queries", format!("pass {} layer {} read other messages than its own queries' ({:.1e})", s.id.pass, s.id.layer, rel_l2(got, &want))));
        }
    }
    broken
}

/// One visit as Mary ran it, on the host.
struct Seen {
    id: CacheId,
    taps: Vec<(&'static str, Vec<f32>)>,
    routing: Vec<BTreeSet<i32>>,
}

fn seen(id: CacheId, out: &Tensor<Bk, 2>, taps: Taps, messages: Option<&Tensor<Bk, 3>>) -> Seen {
    let mut v = vec![("out", host(out.clone())), ("mlp_input", host(taps.mlp_input)), ("router_logits", taps.route.logits())];
    if let Some([q, k, vv, m]) = taps.gates {
        v.extend([("gate_q", host(q)), ("gate_k", host(k)), ("gate_v", host(vv)), ("gate_mlp", host(m))]);
    }
    if let Some(m) = messages {
        let [t, j, h] = m.dims();
        v.push(("messages", host(m.clone().reshape([t * j, h]))));
    }
    Seen {
        id,
        taps: v,
        routing: taps.route.picks().iter().map(|(e, _)| e.iter().map(|&x| x as i32).collect()).collect(),
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
    let run_on = |state: &mut CoreState, phase: &str, x: Vec<f32>, plant: Option<Plant>| -> (Vec<PassOut>, Vec<Seen>) {
        let n = x.len() / h;
        let blocks = vec![rows(input(&format!("emb.{phase}")), n)];
        let mut visits = Vec::new();
        let mut hook = |id: CacheId, out: &Tensor<Bk, 2>, taps: Taps, m: Option<&Tensor<Bk, 3>>| visits.push(seen(id, out, taps, m));
        let t1 = std::time::Instant::now();
        let passes = core.forward(rows(x, n), &blocks, state, 0, plant, Some(&mut hook));
        println!("{phase}{}: {n} rows through 36 visits in {:.0} ms", plant.map(|p| format!(" planted {p:?}")).unwrap_or_default(), t1.elapsed().as_secs_f64() * 1e3);
        (passes, visits)
    };
    let run = |state: &mut CoreState, phase: &str, plant: Option<Plant>| run_on(state, phase, input(&format!("x.{phase}")), plant);

    let compare = |rep: &mut Report, phase: &str, passes: &[PassOut], visits: &[Seen], state: Option<&CoreState>| {
        for s in visits {
            let tag = format!("p{}.l{}:{phase}", s.id.pass, s.id.layer);
            let up = |run: &str, n: &str| input(&format!("{run}.p{}.l{}.{phase}.{n}", s.id.pass, s.id.layer));
            let given = s.id == CacheId { layer: lc.core_start, pass: 0 };
            for (name, v) in &s.taps {
                rep.row(given && PRE_EXPERT.contains(name), &tag, name, v, &up("bf16", name), &up("f32", name));
            }
            if let Some(st) = state {
                let st = &st.visits[&s.id];
                let base = format!("{{run}}.p{}.l{}.{phase}", s.id.pass, s.id.layer);
                let (hk, hv) = held_kv(st);
                for (name, ours) in [("k", hk), ("v", hv)] {
                    let (shape, a) = r.f32s(&format!("{}.{name}", base.replace("{run}", "bf16")));
                    let a = heads_to_rows(&shape, &a);
                    let b = heads_to_rows(&shape, &r.f32s(&format!("{}.{name}", base.replace("{run}", "f32"))).1);
                    let width = shape[0] * shape[2];
                    let (ours, a, b, n) = matching_suffix(&ours, &a, &b, width);
                    rep.row(given, &tag, &format!("state {name} ({n}r)"), ours, a, b);
                }
                for i in 0..4 {
                    let f = |run: &str| {
                        let (shape, v) = r.f32s(&format!("{}.conv{i}", base.replace("{run}", run)));
                        conv_to_rows(&shape, &v)
                    };
                    rep.row(given && i < 3, &tag, &format!("state conv{i}"), &host(st.conv[i].clone()), &f("bf16"), &f("f32"));
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
            {
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
                rep.row(false, &format!("p{p}:{phase}"), &format!("end block {k}"), &host(b.clone()), &block_of(&u16, nb, k, h), &block_of(&u32, nb, k, h));
            }
        }
        let last = passes.last().unwrap();
        let up = |run: &str, n: &str| input(&format!("{run}.core.{phase}.{n}"));
        rep.row(false, &format!("core:{phase}"), "hidden", &host(last.hidden.clone()), &up("bf16", "hidden"), &up("f32", "hidden"));
        let (u16, u32) = (up("bf16", "blocks"), up("f32", "blocks"));
        for (k, b) in last.blocks.iter().enumerate().take(nb) {
            rep.row(false, &format!("core:{phase}"), &format!("block {k}"), &host(b.clone()), &block_of(&u16, nb, k, h), &block_of(&u32, nb, k, h));
        }
    };

    let mut rep = Report::default();
    // prompt, then the step token through the carried states
    let mut state = core.new_state(&dev);
    let (pp, vp) = run(&mut state, "prefill", None);
    compare(&mut rep, "prefill", &pp, &vp, Some(&state));
    let (ps, vs) = run(&mut state, "step", None);
    compare(&mut rep, "step", &ps, &vs, Some(&state));
    // all rows at once with fresh states
    let mut fresh = core.new_state(&dev);
    let (pw, vw) = run(&mut fresh, "whole", None);
    compare(&mut rep, "whole", &pw, &vw, None);

    // positions and cache extents
    let n = input("x.whole").len() / h;
    let held = carried(&state);
    let at = held.iter().filter(|(pos, base, t)| *pos == n && *base == 0 && t[2..].iter().all(|c| !c.is_empty())).count();
    rep.check(
        "positions and extents",
        at == held.len() && held.len() == 36 && state.visits.values().all(|s| s.kv.rows() == n && s.kv.end() == n && s.conv.iter().all(|c| c.dims()[0] == t.sconv_kernel_size - 1)),
        format!("{at} of {} visits at position {n}, rows 0..{n} held, {} conv rows", held.len(), t.sconv_kernel_size - 1),
    );
    // row order: layer 6's first visit computes its keys and values from the
    // given input alone
    {
        let id = CacheId { layer: lc.core_start, pass: 0 };
        let (sk, sv) = held_kv(&state.visits[&id]);
        let (wk, wv) = held_kv(&fresh.visits[&id]);
        for (name, s, w) in [("keys", sk, wk), ("values", sv, wv)] {
            let width = s.len() / n;
            let row = |v: &[f32], i: usize| v[i * width..(i + 1) * width].to_vec();
            let same = (0..n - 1).filter(|&i| same_bits(&row(&s, i), &row(&w, i))).count();
            let last = row(&s, n - 1);
            let nearest = (0..n).min_by(|&i, &j| rel_l2(&row(&w, i), &last).total_cmp(&rel_l2(&row(&w, j), &last))).unwrap();
            rep.check(
                &format!("layer {} {name} in row order", id.layer),
                same == n - 1 && nearest == n - 1,
                format!("{same} of {} prefill rows are the one-prefill run's bits; the step's row is nearest row {nearest}", n - 1),
            );
        }
    }
    // the same schedule again
    let again_step = {
        let mut st = core.new_state(&dev);
        let (_, ap) = run(&mut st, "prefill", None);
        let (_, as_) = run(&mut st, "step", None);
        let (up, us) = (unequal(&vp, &ap), unequal(&vs, &as_));
        let same = carried(&st).iter().zip(&held).filter(|((p, b, x), (q, c, y))| p == q && b == c && x.iter().zip(y).all(|(x, y)| same_bits(x, y))).count();
        rep.check(
            "same schedule repeated",
            up.is_empty() && us.is_empty() && same == 36,
            format!("{} prefill and {} step visits of 36 with other bits; {same} of 36 carried states bit-equal", up.len(), us.len()),
        );
        as_
    };
    // masks, at one call shape
    {
        let x = input("x.whole");
        let swap = |row: usize, with: usize| {
            let mut v = x.clone();
            v.copy_within(with * h..(with + 1) * h, row * h);
            v
        };
        let rows_equal = |a: &Seen, b: &Seen, r: std::ops::Range<usize>| {
            let (p, q) = (&a.taps[0].1, &b.taps[0].1);
            same_bits(&p[r.start * h..r.end * h], &q[r.start * h..r.end * h])
        };
        let (_, later) = run_on(&mut core.new_state(&dev), "whole", swap(n - 1, n - 2), None);
        let kept = vw.iter().zip(&later).filter(|(a, b)| rows_equal(a, b, 0..n - 1)).count();
        let moved = vw.iter().zip(&later).filter(|(a, b)| !rows_equal(a, b, n - 1..n)).count();
        rep.check("no row sees a later one", kept == 36 && moved == 36, format!("another row {}: earlier rows bit-equal on {kept} of 36 visits, its own moved on {moved}", n - 1));
        let (_, earlier) = run_on(&mut core.new_state(&dev), "whole", swap(0, 1), None);
        let moved = vw.iter().zip(&earlier).filter(|(a, b)| !rows_equal(a, b, n - 1..n)).count();
        rep.check("the last row sees the first", moved == 36, format!("another row 0 moves row {} on {moved} of 36 visits", n - 1));
        let mut st = core.new_state(&dev);
        let mut xp = input("x.prefill");
        xp.copy_within(h..2 * h, 0);
        run_on(&mut st, "prefill", xp, None);
        let (_, step) = run(&mut st, "step", None);
        let moved = again_step.iter().zip(&step).filter(|(a, b)| !same_bits(&a.taps[0].1, &b.taps[0].1)).count();
        rep.check("the cached step reads its cache", moved == 36, format!("another row 0 in the prefill moves the step on {moved} of 36 visits"));
    }
    // finite
    {
        let all = [&vp, &vs, &vw];
        let total: usize = all.iter().map(|v| v.iter().map(|s| s.taps.len()).sum::<usize>()).sum();
        let finite: usize = all.iter().map(|v| v.iter().map(|s| s.taps.iter().filter(|(_, t)| t.iter().all(|x| x.is_finite())).count()).sum::<usize>()).sum();
        rep.check("finite", finite == total, format!("{finite} of {total} taps"));
    }
    // each visit alone on upstream's recorded f32 operands, before its experts
    for pass in 0..core.passes {
        for layer in &core.layers {
            let l = layer.layer();
            let x0 = input("x.prefill");
            let out_of = |l: usize| input(&format!("f32.p{pass}.l{l}.prefill.out"));
            let x = if l == lc.core_start { x0.clone() } else { out_of(l - 1) };
            let n = x.len() / h;
            // the pre-loop block, then the inputs of the block boundaries before this layer
            let mut blocks = vec![rows(input("emb.prefill"), n)];
            for b in lc.core_start..l {
                if b % t.n_layers_per_attn_res_block == 0 {
                    blocks.push(rows(if b == lc.core_start { x0.clone() } else { out_of(b - 1) }, n));
                }
            }
            let messages = (pass > 0).then(|| {
                let m = input(&format!("f32.p{pass}.l{l}.prefill.messages"));
                Tensor::<Bk, 3>::from_data(TensorData::new(m, [n, 4, h]), &dev)
            });
            let x = rows(x, n);
            let (st, taps) = match layer {
                CoreLayer::Sliding(s) => {
                    let mut st = State::new(&s.dims, &dev);
                    let taps = s.forward_visit(x, &mut blocks, &mut st, messages.as_ref()).1;
                    (st, taps)
                }
                CoreLayer::Full(f) => {
                    let mut st = State::new(&f.dims, &dev);
                    let taps = f.forward_visit(x, &mut blocks, &mut st, 0, messages.as_ref()).1;
                    (st, taps.rest)
                }
            };
            let tag = format!("p{pass}.l{l}:alone");
            let up = |run: &str, n: &str| input(&format!("{run}.p{pass}.l{l}.prefill.{n}"));
            let mut seen = vec![("mlp_input", host(taps.mlp_input)), ("router_logits", taps.route.logits())];
            if let Some([q, k, v, m]) = taps.gates {
                seen.extend([("gate_q", host(q)), ("gate_k", host(k)), ("gate_v", host(v)), ("gate_mlp", host(m))]);
            }
            for (name, v) in &seen {
                rep.row(true, &tag, name, v, &up("bf16", name), &up("f32", name));
            }
            let (hk, hv) = held_kv(&st);
            for (name, ours) in [("k", hk), ("v", hv)] {
                let f = |run: &str| {
                    let (shape, v) = r.f32s(&format!("{run}.p{pass}.l{l}.prefill.{name}"));
                    (shape[0] * shape[2], heads_to_rows(&shape, &v))
                };
                let ((width, a), (_, b)) = (f("bf16"), f("f32"));
                let (ours, a, b, _) = matching_suffix(&ours, &a, &b, width);
                rep.row(true, &tag, &format!("state {name}"), ours, a, b);
            }
            for i in 0..3 {
                let f = |run: &str| {
                    let (shape, v) = r.f32s(&format!("{run}.p{pass}.l{l}.prefill.conv{i}"));
                    conv_to_rows(&shape, &v)
                };
                rep.row(true, &tag, &format!("state conv{i}"), &host(st.conv[i].clone()), &f("bf16"), &f("f32"));
            }
        }
    }

    // the cached step against the replay's last row: information (other call
    // shapes, see the header), beside the bound that gated it
    let last_row = |v: Vec<f32>| v[v.len() - h..].to_vec();
    for (p, (s, w)) in ps.iter().zip(&pw).enumerate() {
        let ours = rel_l2(&host(s.hidden.clone()), &last_row(host(w.hidden.clone())));
        let up = |run: &str, ph: &str| input(&format!("{run}.p{p}.l{}.{ph}.out", lc.core_end - 1));
        let u32 = rel_l2(&up("f32", "step"), &last_row(up("f32", "whole")));
        let u16 = rel_l2(&up("bf16", "step"), &last_row(up("bf16", "whole")));
        let spread = rel_l2(&up("bf16", "step"), &up("f32", "step"));
        let bound = TOL * u16.max(u32).max(spread);
        rep.lines.push(format!("info pass {p} cached step vs replay: {}", {
            let (beyond, at) = (ours > bound, if ours > bound { "beyond" } else { "within" });
            rep.info += 1;
            rep.info_beyond += usize::from(beyond);
            format!("Mary step vs whole[26] rel-L2 {ours:.3e}; upstream f32 {u32:.3e}, bf16 {u16:.3e}; bf16-f32 step spread {spread:.3e}; {at} the old bound {bound:.3e}")
        }));
    }
    let ours: Vec<f32> = vs.iter().map(|s| s.taps[0].1.clone()).flatten().collect();
    let replay: Vec<f32> = vw.iter().map(|s| last_row(s.taps[0].1.clone())).flatten().collect();
    rep.lines.push(format!("all 36 visit outputs, cached step vs replay row 26: rel-L2 {:.3e}", rel_l2(&ours, &replay)));

    // composition: the same exact gates, held by the run above and broken by each planted fault
    let (x, emb) = (input("x.prefill"), input("emb.prefill"));
    let clean = composition(&core, &pp, &vp, &x, &emb, &dev);
    rep.check("composition of the unplanted run", clean.is_empty(), format!("{} broken {:?}", clean.len(), clean.first()));
    for (plant, gate) in [(Plant::NoReset, "restart"), (Plant::SharedQuery, "queries")] {
        let mut st = core.new_state(&dev);
        let (pp, vp) = run(&mut st, "prefill", Some(plant));
        let broken = composition(&core, &pp, &vp, &x, &emb, &dev);
        rep.check(
            &format!("planted {plant:?} breaks composition ({gate})"),
            broken.iter().any(|(g, _)| *g == gate),
            format!("{} broken, first {:?}", broken.len(), broken.first()),
        );
    }

    for l in &rep.lines {
        println!("{l}");
    }
    println!(
        "gates: {} tensors (worst ratio {:.2} of TOL {TOL}) and {} checks; {} outside or failed. information: {} rows, {} beyond {TOL}, worst ratio {:.2}",
        rep.gates, rep.worst, rep.checks, rep.fails, rep.info, rep.info_beyond, rep.info_worst
    );
    if let Some(p) = args.get(5) {
        std::fs::write(p, format!("[\n{}\n]\n", rep.json.join(",\n"))).expect("receipt");
    }
    std::process::exit(if rep.fails == 0 { 0 } else { 1 });
}
