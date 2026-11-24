//! Greedy decoding of fixed prompts by Inkling 2 Mini with and without its
//! drafter, on the same model: the same token IDs are asked for, and every
//! token the two runs disagree on is reported with the margin between the two
//! best logits of the plain run's row. A row computed beside others differs in
//! its last bits from one computed alone, so a small-margin disagreement is
//! rounding; a large one is a bug.
//!
//! Per prompt: `off` is [`MiniSession::decode`], one target pass per token.
//! `on` at depth `k` is one draft block and one target pass of `k + 1` tokens,
//! the rejected ones taken back. Times are wall, after the prompt; the prompt
//! is prefilled with the hidden states tapped in both.
//!
//! usage: inkling2_speculate <pile> <collection> <root> <dir of NAME.ids/NAME.json> <names,comma> <new tokens> <depths,comma> <out.json>
//!
//! `NAME.ids` is comma-separated token IDs, of which the first `prompt` (read
//! from `NAME.json`) are the prompt.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use mary::model_collection::parse_collection_handle;
use mary::models::inkling::spectree::accept_linear;
use mary::models::inkling2_loop::drafter::Drafter;
use mary::models::inkling2_loop::session::{Logits, MiniModel, MiniSession, Rows, TapPoints};
use serde_json::{Value, json};
use triblespace::prelude::Id;

/// The best token of a row, the runner-up, and the margin between them.
fn best(row: &[f32]) -> (usize, usize, f32) {
    let (mut a, mut b) = (0usize, 1usize);
    if row[b] > row[a] {
        std::mem::swap(&mut a, &mut b);
    }
    for (i, &v) in row.iter().enumerate().skip(2) {
        if v > row[a] {
            b = a;
            a = i;
        } else if v > row[b] {
            b = i;
        }
    }
    (a, b, row[a] - row[b])
}

fn rows(l: &Logits) -> Vec<(usize, usize, f32)> {
    (0..l.rows).map(|i| best(l.row(i))).collect()
}

struct Plain {
    tokens: Vec<usize>,
    second: Vec<usize>,
    margin: Vec<f32>,
    seconds: f64,
    median: f64,
}

/// The prompt, then `n` tokens one at a time. The first token comes from the
/// prompt's last row and is not timed, as in [`speculative`].
fn plain(model: &Arc<MiniModel>, prompt: &[u32], n: usize) -> Result<Plain> {
    let mut s = MiniSession::new(Arc::clone(model));
    let (mut tokens, mut second, mut margin) = (Vec::new(), Vec::new(), Vec::new());
    let mut step = s.prefill(prompt)?;
    let mut times = Vec::new();
    loop {
        let (a, b, m) = best(step.last());
        tokens.push(a);
        second.push(b);
        margin.push(m);
        if tokens.len() > n {
            break;
        }
        let t0 = Instant::now();
        step = s.decode(a as u32)?;
        times.push(t0.elapsed().as_secs_f64());
    }
    let seconds = times.iter().sum();
    times.sort_by(f64::total_cmp);
    Ok(Plain { tokens, second, margin, seconds, median: times[times.len() / 2] })
}

/// The same committed tokens through a plain session, one at a time: where its
/// own argmax is not the token the speculative run committed, with its margin
/// there, and how far the two sessions' rows for the token after the last
/// committed one are apart (relative L2).
struct Forced {
    disagree: Vec<(usize, f32)>,
    next_rel_l2: f64,
    next_argmax_same: bool,
}

fn forced(model: &Arc<MiniModel>, prompt: &[u32], tokens: &[usize], next: &[f32]) -> Result<Forced> {
    let mut s = MiniSession::new(Arc::clone(model));
    let mut row = s.prefill(prompt)?;
    let mut disagree = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        let (a, _, m) = best(row.last());
        if a != t {
            disagree.push((i, m));
        }
        row = s.decode(t as u32)?;
    }
    let (mut d, mut q) = (0f64, 0f64);
    for (x, y) in next.iter().zip(row.last()) {
        d += (*x as f64 - *y as f64).powi(2);
        q += (*y as f64).powi(2);
    }
    let next_argmax_same = best(next).0 == best(row.last()).0;
    Ok(Forced { disagree, next_rel_l2: (d / q).sqrt(), next_argmax_same })
}

struct Spec {
    /// The session's row after its last committed token, decoded alone.
    next: Vec<f32>,
    tokens: Vec<usize>,
    margin: Vec<f32>,
    passes: usize,
    accepted: Vec<usize>,
    seconds: f64,
    draft: f64,
    verify: f64,
    rest: f64,
}

fn speculative(model: &Arc<MiniModel>, drafter: &Drafter, points: &TapPoints, prompt: &[u32], n: usize, k: usize) -> Result<Spec> {
    let mut s = MiniSession::new(Arc::clone(model));
    let mut st = drafter.new_state();
    let (first, taps) = s.append_tapped(prompt, Rows::Last, points, false)?;
    drafter.ingest(&mut st, &taps, prompt.len());
    drop(taps);
    let (a, _, m) = best(first.last());
    let (mut tokens, mut margin) = (vec![a], vec![m]);
    let mut out = Spec { next: Vec::new(), tokens: Vec::new(), margin: Vec::new(), passes: 0, accepted: Vec::new(), seconds: 0.0, draft: 0.0, verify: 0.0, rest: 0.0 };
    // one block before the clock, so no arm pays the first draft's compilation
    drafter.draft(model, &st, a as u32, k)?;
    let t0 = Instant::now();
    while tokens.len() <= n {
        let anchor = *tokens.last().expect("the first token") as u32;
        let d0 = Instant::now();
        let drafts = drafter.draft(model, &st, anchor, k)?;
        out.draft += d0.elapsed().as_secs_f64();
        let mut ids = vec![anchor];
        ids.extend(&drafts);
        let v0 = Instant::now();
        let (logits, taps) = s.append_tapped(&ids, Rows::Tail(k + 1), points, true)?;
        out.verify += v0.elapsed().as_secs_f64();
        let r0 = Instant::now();
        let got = rows(&logits);
        let preds: Vec<usize> = got.iter().map(|r| r.0).collect();
        let (n_ok, new) = accept_linear(&drafts.iter().map(|&d| d as usize).collect::<Vec<_>>(), &preds);
        s.retract(n_ok + 1)?;
        drafter.ingest(&mut st, &taps, n_ok + 1);
        margin.extend(got[..=n_ok].iter().map(|r| r.2));
        tokens.extend(new);
        out.passes += 1;
        out.accepted.push(n_ok);
        out.rest += r0.elapsed().as_secs_f64();
    }
    out.seconds = t0.elapsed().as_secs_f64();
    out.next = s.decode(*tokens.last().expect("tokens") as u32)?.last().to_vec();
    out.tokens = tokens;
    out.margin = margin;
    Ok(out)
}

fn main() -> Result<()> {
    mary::models::inkling::pool::choose_memory_config();
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(args.len() == 8, "usage: <pile> <collection> <root> <dir> <names> <new tokens> <depths> <out.json>");
    let dir = Path::new(&args[3]);
    let names: Vec<&str> = args[4].split(',').collect();
    let n: usize = args[5].parse()?;
    let depths: Vec<usize> = args[6].split(',').map(|d| d.parse().context("depth")).collect::<Result<_>>()?;
    let collection = parse_collection_handle(&args[1]).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let root = Id::from_hex(&args[2]).context("root is a 32-hex id")?;
    let model = Arc::new(MiniModel::open(Path::new(&args[0]), collection, root)?);
    let t0 = Instant::now();
    let drafter = Drafter::bind(&model)?;
    println!("drafter bound in {:.1} s, taps {:?}", t0.elapsed().as_secs_f64(), drafter.taps());
    let points = TapPoints::new(model.config(), drafter.taps())?;
    let mut results: Vec<Value> = Vec::new();

    // warm both call shapes on a short sequence first
    let warm: Vec<u32> = (0..64).map(|i| 1000 + 37 * i).collect();
    plain(&model, &warm, 4)?;
    for &k in &depths {
        speculative(&model, &drafter, &points, &warm, 4, k)?;
    }

    for name in names {
        let ids: Vec<u32> = std::fs::read_to_string(dir.join(format!("{name}.ids")))?
            .trim()
            .split(',')
            .map(|v| v.trim().parse().context("id"))
            .collect::<Result<_>>()?;
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json")))?)?;
        let len = meta["prompt"].as_u64().context("prompt")? as usize;
        ensure!(len <= ids.len(), "{name}: prompt {len} of {} ids", ids.len());
        let prompt = &ids[..len];
        let off = plain(&model, prompt, n)?;
        println!("{name} prompt {len} off {:.2} tok/s (median {:.4} s/token)", n as f64 / off.seconds, off.median);
        results.push(json!({"name": name, "prompt": len, "arm": "off", "new": n, "seconds": off.seconds,
            "median": off.median, "tokens": off.tokens}));
        for &k in &depths {
            let on = speculative(&model, &drafter, &points, prompt, n, k)?;
            let emitted = on.tokens.len() - 1;
            let accepted: usize = on.accepted.iter().sum();
            let check = forced(&model, prompt, &on.tokens, &on.next)?;
            let differ = (0..off.tokens.len().min(on.tokens.len())).find(|&i| off.tokens[i] != on.tokens[i]);
            let per_pass = emitted as f64 / on.passes as f64;
            println!(
                "{name} on k={k}: {:.2} tok/s, {per_pass:.2} emitted/pass, {:.2} accepted drafts/pass, {} drafted, {} passes, draft {:.4} verify {:.4} rest {:.4} s/pass, {}",
                emitted as f64 / on.seconds,
                accepted as f64 / on.passes as f64,
                on.passes * k,
                on.passes,
                on.draft / on.passes as f64,
                on.verify / on.passes as f64,
                on.rest / on.passes as f64,
                match differ {
                    None => "same tokens".to_string(),
                    Some(i) => format!(
                        "first difference at {i}: off {} (second {}, margin {:.3}) on {} (margin {:.3})",
                        off.tokens[i], off.second[i], off.margin[i], on.tokens[i], on.margin[i]
                    ),
                }
            );
            println!(
                "{name} on k={k}: plain decode of its {} tokens disagrees at {}; next row rel_l2 {:.3e}, argmax same {}",
                on.tokens.len(),
                check.disagree.len(),
                check.next_rel_l2,
                check.next_argmax_same
            );
            for (i, m) in &check.disagree {
                println!("  token {i}: plain margin {m:.3}");
            }
            results.push(json!({"name": name, "prompt": len, "arm": "on", "accepted_total": accepted,
                "drafted": on.passes * k,
                "forced_disagree": check.disagree, "next_rel_l2": check.next_rel_l2,
                "next_argmax_same": check.next_argmax_same, "depth": k, "emitted": emitted,
                "passes": on.passes, "seconds": on.seconds, "draft_s": on.draft, "verify_s": on.verify, "rest_s": on.rest,
                "accepted": on.accepted, "tokens": on.tokens,
                "first_difference": differ, "off_margin_there": differ.map(|i| off.margin[i]),
                "off_second_there": differ.map(|i| off.second[i]), "on_margin_there": differ.map(|i| on.margin[i])}));
        }
    }
    std::fs::write(&args[7], serde_json::to_string(&results)?)?;
    println!("wrote {}", args[7]);
    Ok(())
}
