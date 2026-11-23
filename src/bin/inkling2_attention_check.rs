//! Fixed tokens through an Inkling 2 Mini session ([`MiniSession`]), every
//! returned logits row written down with its absolute position, so two builds
//! or two call plans can be compared row for row. Nothing is sampled.
//!
//! `run` feeds a token sequence through one fresh session by a call plan and
//! writes `<out>.f32` (the rows, f32 little-endian, in call order), `<out>.json`
//! (each row's position, each call's wall time and the device pool after it)
//! and `<out>.ids` (the token IDs used). Every call ends by downloading its
//! logits, so each call's time and pool reading follow a device sync.
//!
//! A plan is comma-separated steps, applied in order from position 0:
//!
//! ```text
//! 1000      one call of the next 1000 tokens, its last row
//! 2100:64   one call of the next 2100 tokens, its last 64 rows
//! d64       64 single-token calls, each one's row
//! ```
//!
//! `compare` lines up two runs by position and reports, per shared position,
//! relative L2, worst absolute difference, both argmaxes and both top-2
//! margins, so a disagreement at a near tie can be told from one that is not.
//!
//! usage:
//!   inkling2_attention_check run <pile> <collection> <root> <ids:FILE|text:FILE> <plan> <out>
//!   inkling2_attention_check compare <a> <b> [--near <margin>]

use std::sync::Arc;
use std::time::Instant;

use burn::prelude::*;
use mary::model_collection::parse_collection_handle;
use mary::models::inkling::seam::{Bk, client_of, pool_reserved};
use mary::models::inkling2_loop::session::{MiniModel, MiniOptions, MiniSession, Rows};
use serde_json::json;
use sha2::{Digest, Sha256};
use triblespace::prelude::Id;

fn hexs(d: impl AsRef<[u8]>) -> String {
    d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
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

enum Step {
    Call { n: usize, rows: usize },
    Decode(usize),
}

fn parse_plan(plan: &str) -> Vec<Step> {
    plan.split(',')
        .map(|s| {
            let s = s.trim();
            if let Some(d) = s.strip_prefix('d') {
                Step::Decode(d.parse().expect("decode count"))
            } else if let Some((n, r)) = s.split_once(':') {
                Step::Call { n: n.parse().expect("call length"), rows: r.parse().expect("row count") }
            } else {
                Step::Call { n: s.parse().expect("call length"), rows: 1 }
            }
        })
        .collect()
}

fn plan_len(plan: &[Step]) -> usize {
    plan.iter().map(|s| match s { Step::Call { n, .. } => *n, Step::Decode(d) => *d }).sum()
}

fn run(args: &[String]) {
    let (pile, collection, root, source, plan, out) = (&args[0], &args[1], &args[2], &args[3], &args[4], &args[5]);
    let collection = parse_collection_handle(collection).expect("collection");
    let root = Id::from_hex(root).expect("root must be a 32-hex id");
    let plan_steps = parse_plan(plan);
    let need = plan_len(&plan_steps);

    let (gpu0, avail0) = (gpu_mib(), mem_available_mib());
    let t0 = Instant::now();
    let model = Arc::new(MiniModel::open(pile.as_ref(), collection, root, MiniOptions::default()).expect("open the model"));
    let bind_s = t0.elapsed().as_secs_f64();
    let client = client_of(&Tensor::<Bk, 1>::zeros([1], &Default::default()));
    let (gpu1, avail1, pool1) = (gpu_mib(), mem_available_mib(), pool_reserved(&client));
    println!(
        "bound root {:X} in {bind_s:.1} s; device {gpu0:?} -> {gpu1:?} MiB, MemAvailable {avail0:?} -> {avail1:?} MiB, pool {:.2} GiB",
        model.model_root(),
        pool1 as f64 / (1u64 << 30) as f64
    );

    let (ids, source_sha): (Vec<u32>, String) = if let Some(f) = source.strip_prefix("ids:") {
        let s = std::fs::read_to_string(f).expect("read ids");
        let ids = s.split(|c: char| c == ',' || c.is_whitespace()).filter(|t| !t.is_empty()).map(|t| t.parse().expect("token id")).collect();
        (ids, hexs(Sha256::digest(s.as_bytes())))
    } else if let Some(f) = source.strip_prefix("text:") {
        let bytes = std::fs::read(f).expect("read text");
        let text = String::from_utf8_lossy(&bytes);
        let enc = model.content_tokenizer().encode(text.as_ref(), false).expect("tokenize");
        (enc.get_ids().to_vec(), hexs(Sha256::digest(&bytes)))
    } else {
        panic!("source must be ids:FILE or text:FILE");
    };
    assert!(ids.len() >= need, "the plan needs {need} tokens, the source has {}", ids.len());
    let ids = &ids[..need];
    let id_text: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    std::fs::write(format!("{out}.ids"), id_text.join(",")).expect("write ids");

    let mut s = MiniSession::new(Arc::clone(&model));
    let mut values: Vec<f32> = Vec::new();
    let mut positions: Vec<usize> = Vec::new();
    let mut calls = Vec::new();
    let mut vocab = 0;
    let mut at = 0usize;
    let mut call = |len: usize, rows: usize, s: &mut MiniSession, at: &mut usize| {
        let t = Instant::now();
        let logits = s.append(&ids[*at..*at + len], if rows == 1 { Rows::Last } else { Rows::Tail(rows) }).expect("append");
        let secs = t.elapsed().as_secs_f64();
        vocab = logits.vocab;
        for r in 0..logits.rows {
            positions.push(*at + len - logits.rows + r);
        }
        values.extend_from_slice(&logits.values);
        *at += len;
        let pool = pool_reserved(&client);
        calls.push(json!({"first": *at - len, "tokens": len, "rows": rows, "secs": secs,
                          "pool_reserved": pool, "gpu_mib": gpu_mib(), "mem_available_mib": mem_available_mib()}));
        (secs, pool)
    };
    let total = Instant::now();
    for step in &plan_steps {
        match *step {
            Step::Call { n, rows } => {
                let (secs, pool) = call(n, rows, &mut s, &mut at);
                println!("call {n} tokens to position {at}: {secs:.2} s, pool {:.2} GiB", pool as f64 / (1u64 << 30) as f64);
                mary::models::inkling2_loop::prof::report(&format!("call{n}@{at}"));
            }
            Step::Decode(d) => {
                let t = Instant::now();
                let mut last = (0.0, 0);
                for _ in 0..d {
                    last = call(1, 1, &mut s, &mut at);
                }
                println!(
                    "{d} decodes to position {at}: {:.2} s ({:.3} s each), pool {:.2} GiB",
                    t.elapsed().as_secs_f64(),
                    t.elapsed().as_secs_f64() / d.max(1) as f64,
                    last.1 as f64 / (1u64 << 30) as f64
                );
                mary::models::inkling2_loop::prof::report(&format!("decode{d}@{at}"));
            }
        }
    }
    let total_s = total.elapsed().as_secs_f64();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out}.f32"), &bytes).expect("write rows");
    let report = json!({
        "root": format!("{:X}", model.model_root()), "source": source, "source_sha256": source_sha,
        "plan": plan, "tokens": need, "vocab": vocab, "positions": positions,
        "bind_secs": bind_s, "total_secs": total_s, "calls": calls,
        "gpu_mib_bound": gpu1, "pool_reserved_bound": pool1,
        "rows_sha256": hexs(Sha256::digest(&bytes)),
    });
    std::fs::write(format!("{out}.json"), serde_json::to_string_pretty(&report).unwrap()).expect("write report");
    println!("{} rows at {} positions, {total_s:.1} s; rows sha256 {}", positions.len(), positions.len(), report["rows_sha256"]);
}

struct Rows2 {
    vocab: usize,
    positions: Vec<usize>,
    values: Vec<f32>,
}

fn load(prefix: &str) -> Rows2 {
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{prefix}.json")).expect("report")).expect("json");
    let vocab = j["vocab"].as_u64().unwrap() as usize;
    let positions = j["positions"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as usize).collect();
    let b = std::fs::read(format!("{prefix}.f32")).expect("rows");
    let values = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Rows2 { vocab, positions, values }
}

/// The argmax and the gap from it to the runner-up.
fn top2(v: &[f32]) -> (usize, f32) {
    let mut a = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[a] {
            a = i;
        }
    }
    let b = v.iter().enumerate().filter(|&(i, _)| i != a).map(|(_, &x)| x).fold(f32::NEG_INFINITY, f32::max);
    (a, v[a] - b)
}

fn compare(args: &[String]) {
    let (a, b) = (load(&args[0]), load(&args[1]));
    let near: f32 = args.iter().position(|x| x == "--near").map(|i| args[i + 1].parse().unwrap()).unwrap_or(0.1);
    assert_eq!(a.vocab, b.vocab, "the two runs have different vocabularies");
    let v = a.vocab;
    let (mut shared, mut same, mut worst_rel, mut worst_abs) = (0usize, 0usize, 0f64, 0f64);
    let (mut near_flip, mut far_flip) = (0usize, 0usize);
    for (ia, pa) in a.positions.iter().enumerate() {
        let Some(ib) = b.positions.iter().position(|pb| pb == pa) else { continue };
        let (ra, rb) = (&a.values[ia * v..(ia + 1) * v], &b.values[ib * v..(ib + 1) * v]);
        let (mut d, mut n, mut m) = (0f64, 0f64, 0f64);
        for (x, y) in ra.iter().zip(rb) {
            d += ((*x - *y) as f64).powi(2);
            n += (*y as f64).powi(2);
            m = m.max((*x - *y).abs() as f64);
        }
        let rel = (d / n.max(f64::MIN_POSITIVE)).sqrt();
        let ((aa, ma), (ab, mb)) = (top2(ra), top2(rb));
        shared += 1;
        worst_rel = worst_rel.max(rel);
        worst_abs = worst_abs.max(m);
        let tag = if aa == ab {
            same += 1;
            "same"
        } else if ma.min(mb) < near {
            near_flip += 1;
            "FLIP near tie"
        } else {
            far_flip += 1;
            "FLIP"
        };
        println!("pos {pa}: rel_l2 {rel:.3e} max_abs {m:.3e} argmax {aa} vs {ab} margins {ma:.4} vs {mb:.4} {tag}");
    }
    println!(
        "{shared} shared positions; argmax same {same}, flips at a near tie (margin < {near}) {near_flip}, other flips {far_flip}; worst rel_l2 {worst_rel:.3e}, worst max_abs {worst_abs:.3e}"
    );
    if shared == 0 || far_flip > 0 {
        std::process::exit(1);
    }
}

fn main() {
    // Mini's pool policy is the Inkling engine's: exclusive pages unless the
    // operator set CUBECL_MEMORY_CONFIG. It must run before any CUDA context.
    mary::models::inkling::pool::choose_memory_config();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") if args.len() >= 8 => run(&args[2..]),
        Some("compare") if args.len() >= 4 => compare(&args[2..]),
        _ => {
            eprintln!(
                "usage: inkling2_attention_check run <pile> <collection> <root> <ids:FILE|text:FILE> <plan> <out>\n       \
                 inkling2_attention_check compare <a> <b> [--near <margin>]"
            );
            std::process::exit(2);
        }
    }
}
