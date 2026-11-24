//! One-stream decode of an Inkling 2 Mini session, timed per step from the host.
//!
//! The prompt is a fixture's: `prompts[0]` of a JSON file, optionally behind the
//! first `pad` IDs of its `carrier`, so the decode position is a choice.
//!
//! `gen`: greedy generation. Per repetition a fresh session, the prompt in calls
//! of at most 4096, the first token from its last row, then one call per token.
//! A warm-up repetition of eight selections comes first and is not counted.
//!
//! `rows`: the same forced tokens at the same positions, `k` rows per call, for
//! each `k` in the list, over `span` positions (a multiple of every `k`). The
//! tokens are the prompt's own greedy continuation. Eight one-token calls after
//! the prompt warm each arm. Arms run in list order on even repetitions and
//! reversed on odd ones.
//!
//! `distinct`: `rows`, untimed, reading every visit's picks to count how many
//! distinct experts a call of `k` rows touches.
//!
//! Per call the output holds the host's marks in microseconds from the call's
//! start: the forward built (every operator enqueued by the calling thread),
//! logits read, logits as a host vector, router fault read, and the end
//! of selection. Thread CPU time is read before and after each repetition.
//!
//! usage: inkling2_step_profile <pile> <collection> <root> <fixture.json> <out.json> <gen|rows|distinct> <pad> <reps> <gen: new tokens | rows: span> [ks] [last|tail]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use mary::model_collection::parse_collection_handle;
use mary::models::inkling2_loop::session::{Logits, MiniModel, MiniSession, Rows};
use mary::models::inkling2_loop::steptrace;
use serde_json::{Value, json};
use triblespace::prelude::Id;

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// `name.tid -> (user, system)` clock ticks of every thread of this process.
fn threads() -> BTreeMap<String, (u64, u64)> {
    let mut out = BTreeMap::new();
    let Ok(dir) = std::fs::read_dir("/proc/self/task") else { return out };
    for e in dir.flatten() {
        let Ok(s) = std::fs::read_to_string(e.path().join("stat")) else { continue };
        let (Some(a), Some(b)) = (s.find('('), s.rfind(')')) else { continue };
        let f: Vec<&str> = s[b + 1..].split_whitespace().collect();
        let t = |i: usize| f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0u64);
        out.insert(format!("{}.{}", &s[a + 1..b], e.file_name().to_string_lossy()), (t(11), t(12)));
    }
    out
}

fn thread_delta(a: &BTreeMap<String, (u64, u64)>, b: &BTreeMap<String, (u64, u64)>) -> Value {
    let mut m = serde_json::Map::new();
    for (k, (u, s)) in b {
        let (u0, s0) = a.get(k).copied().unwrap_or((0, 0));
        if u + s > u0 + s0 {
            m.insert(k.clone(), json!([u - u0, s - s0]));
        }
    }
    Value::Object(m)
}

/// One call, and its marks: microseconds from its start.
fn step(s: &mut MiniSession, ids: &[u32], rows: Rows, visits: bool) -> Result<(Logits, Value)> {
    steptrace::take();
    let t0 = Instant::now();
    let logits = s.append(ids, rows)?;
    let append = t0.elapsed();
    let next = argmax(logits.last());
    let end = t0.elapsed();
    let us = |t: Instant| t.duration_since(t0).as_micros() as u64;
    let marks = steptrace::take();
    let at = |name: &str| marks.iter().rev().find(|m| m.0 == name).map(|m| us(m.3));
    let mut v = json!({"rows": ids.len(), "pos": s.position() - ids.len(), "built": at("built"), "read": at("read"), "values": at("values"),
        "fault": at("fault"), "append": append.as_micros() as u64, "end": end.as_micros() as u64, "next": next});
    if visits {
        v["visits"] = marks.iter().filter(|m| m.0 == "at").map(|m| json!([if m.1 == usize::MAX { -1 } else { m.1 as i64 }, m.2, us(m.3)])).collect();
    }
    Ok((logits, v))
}

fn prefill(model: &Arc<MiniModel>, prompt: &[u32]) -> Result<(MiniSession, Logits)> {
    let mut s = MiniSession::new(Arc::clone(model));
    let mut last = None;
    for c in prompt.chunks(4096) {
        last = Some(s.append(c, Rows::Last)?);
    }
    Ok((s, last.context("an empty prompt")?))
}

fn generate(model: &Arc<MiniModel>, prompt: &[u32], n: usize) -> Result<(Vec<u32>, Value)> {
    let t0 = Instant::now();
    let (mut s, first) = prefill(model, prompt)?;
    let mut tok = argmax(first.last());
    let prompt_s = t0.elapsed().as_secs_f64();
    let mut tokens = vec![tok];
    let mut calls = Vec::new();
    let cpu = threads();
    let d0 = Instant::now();
    while tokens.len() < n {
        let (_, v) = step(&mut s, &[tok], Rows::Last, true)?;
        tok = v["next"].as_u64().expect("a token") as u32;
        tokens.push(tok);
        calls.push(v);
    }
    let decode_s = d0.elapsed().as_secs_f64();
    let whole_s = t0.elapsed().as_secs_f64();
    let cpu = thread_delta(&cpu, &threads());
    Ok((tokens, json!({"new": n, "whole_s": whole_s, "prompt_s": prompt_s, "decode_s": decode_s, "decode_calls": n - 1,
        "cpu_ticks": cpu, "calls": calls})))
}

fn main() -> Result<()> {
    mary::models::inkling::pool::choose_memory_config();
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(a.len() >= 9, "see the module doc");
    let (mode, pad, reps, n): (&str, usize, usize, usize) = (&a[5], a[6].parse()?, a[7].parse()?, a[8].parse()?);
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&a[3])?)?;
    let ids = |v: &Value| -> Result<Vec<u32>> { v.as_array().context("ids")?.iter().map(|x| Ok(x.as_u64().context("id")? as u32)).collect() };
    let mut prompt = ids(&fixture["carrier"])?;
    ensure!(prompt.len() >= pad, "the carrier has {} ids", prompt.len());
    prompt.truncate(pad);
    prompt.extend(ids(&fixture["prompts"][0])?);
    let collection = parse_collection_handle(&a[1]).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let t0 = Instant::now();
    let model = Arc::new(MiniModel::open(Path::new(&a[0]), collection, Id::from_hex(&a[2]).context("root")?)?);
    println!("bound in {:.1} s; prompt {} ids", t0.elapsed().as_secs_f64(), prompt.len());
    steptrace::enable(true);
    let mut out = Vec::new();
    match mode {
        "gen" => {
            generate(&model, &prompt, 8)?;
            for rep in 0..reps {
                let (tokens, v) = generate(&model, &prompt, n)?;
                println!(
                    "rep {rep}: {n} tokens in {:.3} s whole ({:.2} tok/s), {} decode calls in {:.3} s ({:.2} tok/s)",
                    v["whole_s"].as_f64().unwrap(),
                    n as f64 / v["whole_s"].as_f64().unwrap(),
                    n - 1,
                    v["decode_s"].as_f64().unwrap(),
                    (n - 1) as f64 / v["decode_s"].as_f64().unwrap()
                );
                out.push(json!({"rep": rep, "tokens": tokens, "run": v}));
            }
        }
        "rows" | "distinct" => {
            ensure!(a.len() == 11, "rows needs <ks> <last|tail>");
            let ks: Vec<usize> = a[9].split(',').map(|k| k.parse().context("k")).collect::<Result<_>>()?;
            ensure!(ks.iter().all(|k| n % k == 0), "span {n} is not a multiple of every k");
            let tail = match a[10].as_str() {
                "tail" => true,
                "last" => false,
                other => bail!("rows: {other}"),
            };
            let (tokens, _) = generate(&model, &prompt, 8 + n + 1)?;
            let mut arm = |k: usize| -> Result<Vec<Value>> {
                let (mut s, _) = prefill(&model, &prompt)?;
                for t in &tokens[..8] {
                    step(&mut s, &[*t], Rows::Last, false)?;
                }
                steptrace::route_counts(mode == "distinct");
                steptrace::take_distinct();
                let mut calls = Vec::new();
                for c in tokens[8..8 + n].chunks(k) {
                    let rows = if tail { Rows::Tail(k) } else { Rows::Last };
                    let (_, mut v) = step(&mut s, c, rows, false)?;
                    if mode == "distinct" {
                        v["distinct"] = steptrace::take_distinct().iter().map(|d| d.1).collect();
                    }
                    calls.push(v);
                }
                steptrace::route_counts(false);
                Ok(calls)
            };
            if mode == "rows" {
                for &k in &ks {
                    arm(k)?;
                }
            }
            for rep in 0..reps {
                let order: Vec<usize> = if rep % 2 == 0 { ks.clone() } else { ks.iter().rev().copied().collect() };
                for k in order {
                    let calls = arm(k)?;
                    let mut t: Vec<u64> = calls.iter().map(|c| c["append"].as_u64().unwrap()).collect();
                    t.sort();
                    println!("rep {rep} k {k}: {} calls, median {:.2} ms", t.len(), t[t.len() / 2] as f64 / 1e3);
                    out.push(json!({"rep": rep, "k": k, "tail": tail, "calls": calls}));
                }
            }
        }
        other => bail!("mode {other}"),
    }
    std::fs::write(&a[4], serde_json::to_string(&json!({"mode": mode, "pad": pad, "prompt": prompt.len(), "out": out}))?)?;
    println!("wrote {}", a[4]);
    Ok(())
}
