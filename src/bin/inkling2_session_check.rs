//! The Inkling 2 Mini session ([`MiniSession`]) on the release's weights in a
//! pile, fed fixed token IDs. Nothing is sampled.
//!
//! What is compared:
//!
//! 1. **Replay against the cache.** One session takes the whole sequence in
//!    one call and returns every row's logits (the full replay). A second
//!    session takes the prompt in one call and the remaining tokens one at a
//!    time. Each incremental row is compared with the replay's row at the same
//!    position: relative L2, worst absolute difference, argmax and top-5 set.
//! 2. **Independence.** Two sessions over the one model are interleaved
//!    (prompt, then alternating single tokens; the second session runs a
//!    different, shorter sequence) and each is compared with a run of its own
//!    sequence on a fresh session. Then the second is reset and replayed, and
//!    its logits are compared with its first run bit for bit.
//! 3. **Refusals.** An out-of-table ID, an empty call and an oversized row
//!    request are refused without moving the position.
//! 4. **Upstream** (with `--upstream`): the replay's rows against
//!    upstream's f32 and bf16 logits for the same token IDs, from
//!    `golden/capture_inkling2_tail.py chained`. That reference is composed of
//!    three upstream captures (layers 0..5, the core, the tail), and its f32
//!    and bf16 runs share one seam (the core starts both from chain5's f32
//!    layer-5 output, rounded to bf16), so its bf16-vs-f32 distance understates
//!    a whole bf16 run's. It is not an independent end-to-end upstream run. And
//!    upstream multiplies unquantised activations in the experts, where the
//!    session runs the trained precision, so the distance is information about
//!    that difference and not a gate: it fails nothing. Each row's relative L2,
//!    that over the spread, argmax and top-5 are reported, none dropped.
//!
//! usage: inkling2_session_check <pile> <collection> <root> <ids,comma,separated> <prompt_len>
//!        [--upstream <reference.safetensors>] [--receipt <out.json>]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use mary::model_collection::parse_collection_handle;
use mary::models::inkling2_loop::session::{Logits, MiniModel, MiniSession, Rows};
use serde_json::json;
use triblespace::prelude::Id;

const TOL: f64 = 1.5;

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((*x - *y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(f64::MIN_POSITIVE)).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x - *y).abs() as f64).fold(0.0, f64::max)
}

fn top(v: &[f32], k: usize) -> Vec<usize> {
    let mut i: Vec<usize> = (0..v.len()).collect();
    i.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    i.truncate(k);
    i
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Device memory this process holds, from nvidia-smi, in MiB.
fn gpu_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let me = std::process::id().to_string();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut f = l.split(',').map(str::trim);
            (f.next()? == me).then(|| f.next()?.parse().ok())?
        })
        .next()
}

fn mem_available_mib() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    Some(line.split_whitespace().nth(1)?.parse::<u64>().ok()? / 1024)
}

struct Report {
    lines: Vec<String>,
    failures: Vec<String>,
}

impl Report {
    fn say(&mut self, s: String) {
        println!("{s}");
        self.lines.push(s);
    }

    fn check(&mut self, what: &str, ok: bool, detail: String) {
        self.say(format!("{} {what}: {detail}", if ok { "ok  " } else { "FAIL" }));
        if !ok {
            self.failures.push(format!("{what}: {detail}"));
        }
    }

    /// One row against another: the row comparison every check uses.
    fn row(&mut self, what: &str, got: &[f32], want: &[f32]) -> serde_json::Value {
        let (r, m) = (rel_l2(got, want), max_abs(got, want));
        let (tg, tw) = (top(got, 5), top(want, 5));
        let same_argmax = tg[0] == tw[0];
        let overlap = tg.iter().collect::<BTreeSet<_>>().intersection(&tw.iter().collect()).count();
        self.say(format!(
            "     {what}: rel_l2 {r:.3e} max_abs {m:.3e} argmax {} vs {} top5 overlap {overlap}/5 bit-equal {}",
            tg[0],
            tw[0],
            bits_equal(got, want)
        ));
        json!({"what": what, "rel_l2": r, "max_abs": m, "argmax": tg[0], "want_argmax": tw[0],
               "top5_overlap": overlap, "same_argmax": same_argmax, "bit_equal": bits_equal(got, want)})
    }
}

/// `ids` through a fresh `session`: the first `prompt` in one call, the rest
/// one token per call. Returns one logits row per call after the prompt's
/// last, starting with the prompt's.
fn incremental(s: &mut MiniSession, ids: &[u32], prompt: usize) -> Vec<Vec<f32>> {
    timed(s, ids, prompt).0
}

/// [`incremental`], with each call's wall time in seconds.
fn timed(s: &mut MiniSession, ids: &[u32], prompt: usize) -> (Vec<Vec<f32>>, Vec<f64>) {
    let t = Instant::now();
    let mut rows = vec![s.prefill(&ids[..prompt]).expect("prefill").last().to_vec()];
    let mut secs = vec![t.elapsed().as_secs_f64()];
    for &id in &ids[prompt..] {
        let t = Instant::now();
        rows.push(s.decode(id).expect("decode").last().to_vec());
        secs.push(t.elapsed().as_secs_f64());
    }
    (rows, secs)
}

fn replay(model: &Arc<MiniModel>, ids: &[u32]) -> Logits {
    let mut s = MiniSession::new(Arc::clone(model));
    s.append(ids, Rows::Tail(ids.len())).expect("replay")
}

fn main() {
    // Mini's pool policy is the Inkling engine's: exclusive pages unless the
    // operator set CUBECL_MEMORY_CONFIG. It must run before any CUDA context.
    mary::models::inkling::pool::choose_memory_config();
    let args: Vec<String> = std::env::args().collect();
    let (pile, collection, root, ids, prompt) = (&args[1], &args[2], &args[3], &args[4], &args[5]);
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let collection = parse_collection_handle(collection).expect("collection");
    let root = Id::from_hex(root).expect("root must be a 32-hex id");
    let ids: Vec<u32> = ids.split(',').map(|s| s.trim().parse().expect("token id")).collect();
    let prompt: usize = prompt.parse().expect("prompt length");
    assert!(prompt >= 2 && prompt < ids.len(), "need a prompt of at least 2 and at least one decoded token");
    let mut rep = Report { lines: Vec::new(), failures: Vec::new() };

    let (gpu0, avail0) = (gpu_mib(), mem_available_mib());
    let t0 = Instant::now();
    let model = Arc::new(MiniModel::open(pile.as_ref(), collection, root).expect("open the model"));
    let bind_s = t0.elapsed().as_secs_f64();
    let (gpu1, avail1) = (gpu_mib(), mem_available_mib());
    rep.say(format!(
        "bound root {:X} (tokenizer {:X}) in {bind_s:.1} s; vocab {}; device {:?} MiB, MemAvailable {:?} -> {:?} MiB",
        model.model_root(),
        model.tokenizer_root(),
        model.vocab(),
        gpu1,
        avail0,
        avail1
    ));
    let n = ids.len();

    // 1. replay against the cache
    let t1 = Instant::now();
    let full = replay(&model, &ids);
    let replay_s = t1.elapsed().as_secs_f64();
    let gpu2 = gpu_mib();
    rep.say(format!("full replay of {n} tokens, {} rows, in {replay_s:.1} s; device {gpu2:?} MiB", full.rows));
    let mut b = MiniSession::new(Arc::clone(&model));
    let t2 = Instant::now();
    let (inc, call_s) = timed(&mut b, &ids, prompt);
    let inc_s = t2.elapsed().as_secs_f64();
    rep.check("position after the incremental run", b.position() == n, format!("{} of {n}", b.position()));
    let calls: Vec<String> = call_s.iter().map(|s| format!("{s:.2}")).collect();
    rep.say(format!(
        "incremental: prompt {prompt} then {} single tokens, {inc_s:.1} s (per call, prefill first: {} s)",
        n - prompt,
        calls.join(" ")
    ));
    let mut cached = Vec::new();
    let mut worst = 0f64;
    let mut argmax_same = 0;
    for (k, row) in inc.iter().enumerate() {
        let pos = prompt - 1 + k;
        let r = rep.row(&format!("cached row {pos} vs replay"), row, full.row(pos));
        worst = worst.max(r["rel_l2"].as_f64().unwrap());
        argmax_same += r["same_argmax"].as_bool().unwrap() as usize;
        cached.push(r);
    }
    rep.check(
        "cached next token = replay's",
        argmax_same == inc.len(),
        format!("{argmax_same}/{} rows, worst rel_l2 {worst:.3e}", inc.len()),
    );

    // 2. two interleaved sessions, then reset and replay
    let other: Vec<u32> = ids[..prompt / 2 + 2].iter().rev().copied().collect();
    let p2 = prompt / 2;
    let alone_b = {
        let mut s = MiniSession::new(Arc::clone(&model));
        incremental(&mut s, &ids, prompt)
    };
    let alone_c = {
        let mut s = MiniSession::new(Arc::clone(&model));
        incremental(&mut s, &other, p2)
    };
    let (mut sb, mut sc) = (MiniSession::new(Arc::clone(&model)), MiniSession::new(Arc::clone(&model)));
    let mut got_b = vec![sb.prefill(&ids[..prompt]).unwrap().last().to_vec()];
    let mut got_c = vec![sc.prefill(&other[..p2]).unwrap().last().to_vec()];
    let steps = (n - prompt).max(other.len() - p2);
    for k in 0..steps {
        if let Some(&id) = other.get(p2 + k) {
            got_c.push(sc.decode(id).unwrap().last().to_vec());
        }
        if let Some(&id) = ids.get(prompt + k) {
            got_b.push(sb.decode(id).unwrap().last().to_vec());
        }
    }
    let mut interleaved = Vec::new();
    for (name, got, alone) in [("session B", &got_b, &alone_b), ("session C", &got_c, &alone_c)] {
        let same = got.iter().zip(alone.iter()).filter(|(a, b)| bits_equal(a, b)).count();
        for (k, (a, b)) in got.iter().zip(alone.iter()).enumerate() {
            interleaved.push(rep.row(&format!("{name} interleaved row {k} vs alone"), a, b));
        }
        let argmax = got.iter().zip(alone.iter()).filter(|(a, b)| top(a, 1) == top(b, 1)).count();
        rep.check(
            &format!("{name} interleaved = alone"),
            argmax == got.len() && got.len() == alone.len(),
            format!("argmax {argmax}/{}, bit-equal {same}/{}", got.len(), got.len()),
        );
    }
    sc.reset();
    rep.check("position after reset", sc.position() == 0, format!("{}", sc.position()));
    let again_c = incremental(&mut sc, &other, p2);
    let same = again_c.iter().zip(got_c.iter()).filter(|(a, b)| bits_equal(a, b)).count();
    let argmax = again_c.iter().zip(got_c.iter()).filter(|(a, b)| top(a, 1) == top(b, 1)).count();
    for (k, (a, b)) in again_c.iter().zip(got_c.iter()).enumerate() {
        interleaved.push(rep.row(&format!("session C after reset row {k} vs first run"), a, b));
    }
    rep.check(
        "reset and replay = first run",
        argmax == got_c.len(),
        format!("argmax {argmax}/{}, bit-equal {same}/{}", got_c.len(), got_c.len()),
    );

    // 3. refusals leave the session where it was
    let before = sb.position();
    let bad = model.embedding_rows() as u32;
    let refusals = [
        ("an ID outside the table", sb.append(&[ids[0], bad], Rows::Last).is_err()),
        ("an empty call", sb.append(&[], Rows::Last).is_err()),
        ("Rows::Tail longer than the call", sb.append(&[ids[0]], Rows::Tail(2)).is_err()),
        ("Rows::Tail(0)", sb.append(&[ids[0]], Rows::Tail(0)).is_err()),
    ];
    for (what, refused) in refusals {
        rep.check(&format!("refuses {what}"), refused && sb.position() == before && !sb.is_poisoned(), format!("position {}", sb.position()));
    }

    // 4. upstream
    let mut upstream = Vec::new();
    if let Some(path) = flag("--upstream") {
        let bytes = std::fs::read(&path).expect("read the upstream reference");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("safetensors");
        let get = |name: &str| -> (Vec<usize>, Vec<f32>) {
            let t = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(t.dtype(), safetensors::Dtype::F32, "{name}");
            (t.shape().to_vec(), t.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        };
        let ref_ids: Vec<u32> = st
            .tensor("ids")
            .expect("the reference names its token ids")
            .data()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect();
        rep.check("upstream token ids = ours", ref_ids == ids, format!("{} ids", ref_ids.len()));
        let (shape, u32_) = get("f32.logits");
        let (_, u16_) = get("bf16.logits");
        assert_eq!(shape, vec![n, model.vocab()], "upstream logits shape");
        let v = model.vocab();
        let (mut within, mut am32, mut am16) = (0, 0, 0);
        for p in 0..n {
            let (r32, r16) = (&u32_[p * v..(p + 1) * v], &u16_[p * v..(p + 1) * v]);
            let mine = full.row(p);
            let spread = rel_l2(r16, r32);
            let mut row = rep.row(&format!("replay row {p} vs composed upstream f32"), mine, r32);
            let ratio = row["rel_l2"].as_f64().unwrap() / spread.max(f64::MIN_POSITIVE);
            let a = top(mine, 1)[0];
            let ok = ratio <= TOL;
            within += ok as usize;
            am32 += (a == top(r32, 1)[0]) as usize;
            am16 += (a == top(r16, 1)[0]) as usize;
            rep.say(format!("       upstream bf16-vs-f32 {spread:.3e}, ratio {ratio:.2} {}", if ok { "within 1.5x" } else { "beyond 1.5x" }));
            row["upstream_spread"] = json!(spread);
            row["ratio"] = json!(ratio);
            row["within"] = json!(ok);
            upstream.push(row);
        }
        rep.say(format!(
            "info replay rows within 1.5x the composed (shared-seam) upstream reference's own spread: \
             {within}/{n}; argmax = upstream f32 {am32}/{n}, = upstream bf16 {am16}/{n}"
        ));
    }

    let gpu3 = gpu_mib();
    rep.say(format!(
        "context {n} tokens per session, up to 3 sessions alive; device memory {gpu3:?} MiB (after binding {gpu1:?}, before {gpu0:?})"
    ));
    let verdict = if rep.failures.is_empty() { "pass" } else { "fail" };
    rep.say(format!("{verdict}: {} failures", rep.failures.len()));
    if let Some(path) = flag("--receipt") {
        let receipt = json!({
            "verdict": verdict,
            "failures": rep.failures,
            "pile": pile, "collection": args[2], "root": format!("{:X}", model.model_root()),
            "tokenizer_root": format!("{:X}", model.tokenizer_root()),
            "ids": ids, "prompt": prompt,
            "bind_s": bind_s, "replay_s": replay_s, "incremental_s": inc_s, "incremental_call_s": call_s,
            "profile": if cfg!(debug_assertions) { "dev" } else { "release" },
            "upstream_reference": flag("--upstream"),
            "device_mib": {"before": gpu0, "bound": gpu1, "after_replay": gpu2, "end": gpu3},
            "mem_available_mib": {"before": avail0, "bound": avail1},
            "cached_vs_replay": cached,
            "interleaved_and_reset": interleaved,
            "upstream": upstream,
            "log": rep.lines,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap()).expect("write the receipt");
    }
    std::process::exit(if rep.failures.is_empty() { 0 } else { 1 });
}
