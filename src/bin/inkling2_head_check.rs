//! Inkling 2 Mini's output head on the release's own weights, read from a
//! model pile, against upstream's classes on the same bytes.
//!
//! The reference directory is written by `golden/capture_inkling2_head.py
//! real` from the release's safetensors file; it records the sha256 of the
//! three weight tensors it used, and this check hashes the pile's tensors and
//! refuses to compare unless they are the same bytes.
//!
//! usage: inkling2_head_check <pile> <collection> <root> <reference dir> <receipt.json> [--mutate partial|softcap]
//!
//! Tolerances as in the fixture test (`models::inkling2_loop::head`), stage by
//! stage on the reference's own input: the residual within 1e-5 (1 + |x|),
//! the norm's and the unembedding's BF16 outputs within one BF16 step of the
//! reference value or an absolute floor (1e-5 normed, 1e-4 raw) and at least
//! 99% bit-equal, the softcapped logits within one raw BF16 step plus f32 tanh
//! error. End to end, the same for the residual and the norm; the logits are
//! held to the exact propagation of the normed values that differ
//! (sum_k |dn_k| |W_vk|) on top of that rule, and the argmax must be equal. `--mutate` breaks the head on
//! purpose (the partial block dropped, or no softcap) and must fail.

use anyhow::{Context, Result, bail, ensure};
use mary::models::inkling2_loop::Inkling2LoopConfig;
use mary::models::inkling2_loop::head::{self, Head, HeadOut, one_query_residual};
use mary::models::inkling2_loop::pile::MiniRoot;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;
use triblespace::prelude::*;

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read(dir: &Path, name: &str) -> Result<Vec<u8>> {
    std::fs::read(dir.join(format!("{name}.bin"))).with_context(|| format!("{name}.bin"))
}
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}
fn top(v: &[f32], k: usize) -> Vec<usize> {
    let mut i: Vec<usize> = (0..v.len()).collect();
    i.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    i.truncate(k);
    i
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 6 || args.len() == 8, "usage: {} <pile> <collection> <root> <reference dir> <receipt.json> [--mutate partial|softcap]", args[0]);
    let pile = PathBuf::from(&args[1]);
    let handle = mary::model_collection::parse_collection_handle(&args[2])?;
    let root_id = Id::from_hex(&args[3]).context("root must be a 32-hex id")?;
    let refdir = PathBuf::from(&args[4]);
    let receipt = PathBuf::from(&args[5]);
    let mutate = if args.len() == 8 {
        ensure!(args[6] == "--mutate", "unknown option {}", args[6]);
        Some(args[7].clone())
    } else {
        None
    };

    let t0 = Instant::now();
    let root = MiniRoot::open(&pile, handle, root_id)?;
    let open_s = t0.elapsed().as_secs_f64();
    let cfg = Inkling2LoopConfig::from_json(&serde_json::to_string(&root.config()?)?).map_err(anyhow::Error::msg)?;
    let text = &cfg.text_config;
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(refdir.join("reference.json"))?)?;

    // Same bytes, or no comparison.
    let mut weights = serde_json::Map::new();
    for name in [head::QUERY, head::NORM, head::UNEMBED] {
        let (dtype, shape, bytes) = root.tensor(name).map_err(anyhow::Error::msg)?;
        let ours = format!("{:x}", Sha256::digest(bytes));
        let theirs = reference["weight_sha256"][name].as_str().with_context(|| format!("reference has no digest for {name}"))?;
        ensure!(ours == theirs, "{name}: pile bytes sha256 {ours}, reference used {theirs}");
        weights.insert(name.into(), serde_json::json!({"dtype": dtype.name(), "shape": shape, "sha256": ours}));
    }
    let head = Head::from_root(text, &root).map_err(anyhow::Error::msg)?;

    let h = text.hidden_size;
    let v = head.vocab();
    let shape = |k: &str| -> Vec<usize> {
        reference["shapes"][k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
    };
    let bshape = shape("block_states");
    ensure!(bshape.len() == 3 && bshape[2] == h, "block_states shape {bshape:?}");
    let (tokens, n_blocks) = (bshape[0], bshape[1]);
    ensure!(shape("logits") == vec![tokens, v], "reference logits {:?}, head emits [{tokens}, {v}]", shape("logits"));
    let blocks = bf16s(&read(&refdir, "block_states")?);
    let partial = bf16s(&read(&refdir, "partial")?);
    let r_selected = f32s(&read(&refdir, "selected")?);
    let r_normed = bf16s(&read(&refdir, "normed")?);
    let r_raw = bf16s(&read(&refdir, "raw")?);
    let r_logits = f32s(&read(&refdir, "logits")?);

    let t1 = Instant::now();
    let q: Vec<f32> = f32s(root.tensor(head::QUERY).unwrap().2);
    let mut out = match mutate.as_deref() {
        None | Some("softcap") => head.forward(&blocks, n_blocks, &partial),
        // The partial block dropped: the residual sees the completed blocks only.
        Some("partial") => head.project(one_query_residual(&q, &blocks, tokens, n_blocks, h)),
        Some(m) => bail!("unknown mutation {m}"),
    };
    // Each stage again on the reference's own input (teacher-forced).
    let tf_normed = head.norm(&r_selected);
    let (tf_raw, mut tf_logits) = head.unembed(&r_normed);
    if mutate.as_deref() == Some("softcap") {
        out = HeadOut { logits: out.raw.clone(), ..out };
        tf_logits = tf_raw.clone();
    }
    let forward_s = t1.elapsed().as_secs_f64();

    let mut failures: Vec<String> = Vec::new();
    let mut fail = |m: String| {
        if failures.len() < 12 {
            failures.push(m)
        }
    };
    let finite = out.logits.iter().chain(&out.selected).chain(&tf_logits).all(|x| x.is_finite());
    if !finite {
        fail("non-finite output".into());
    }

    // A BF16 tap: within one BF16 step of the reference value or an absolute
    // floor near zero, and at least 99% bit-equal.
    let bf16_tap = |name: &str, ours: &[f32], theirs: &[f32], floor: f32, fail: &mut dyn FnMut(String)| {
        let (mut eq, mut max_abs) = (0usize, 0f32);
        for (i, (&a, &b)) in ours.iter().zip(theirs).enumerate() {
            let d = (a - b).abs();
            eq += (a == b) as usize;
            max_abs = max_abs.max(d);
            if d > b.abs() / 128.0 + floor {
                fail(format!("{name}[{i}]: ours {a}, reference {b}"));
            }
        }
        if eq * 100 < ours.len() * 99 {
            fail(format!("{name}: only {eq} of {} bit-equal", ours.len()));
        }
        serde_json::json!({"bit_equal": eq, "of": ours.len(), "max_abs_diff": max_abs})
    };
    // Softcapped logits: within the image of one raw BF16 step plus f32 tanh error.
    let capped = |name: &str, ours: &[f32], theirs: &[f32], raw_ref: &[f32], fail: &mut dyn FnMut(String)| {
        let (mut eq, mut max_abs) = (0usize, 0f32);
        for (i, (&a, &b)) in ours.iter().zip(theirs).enumerate() {
            let d = (a - b).abs();
            eq += (a == b) as usize;
            max_abs = max_abs.max(d);
            if d > raw_ref[i].abs() / 128.0 + 1e-5 * b.abs().max(1.0) {
                fail(format!("{name}[{i}]: ours {a}, reference {b}"));
            }
        }
        serde_json::json!({"bit_equal": eq, "of": ours.len(), "max_abs_diff": max_abs})
    };

    // A. The residual, end to end.
    let mut sel_max = 0f32;
    for (i, (&a, &b)) in out.selected.iter().zip(&r_selected).enumerate() {
        let d = (a - b).abs();
        sel_max = sel_max.max(d);
        if d > 1e-5 * (1.0 + b.abs()) {
            fail(format!("selected[{i}]: ours {a}, reference {b}"));
        }
    }
    // B, C. Norm and unembedding+softcap, each on the reference's input.
    let teacher = serde_json::json!({
        "normed_from_reference_selected": bf16_tap("tf normed", &tf_normed, &r_normed, 1e-5, &mut fail),
        "raw_from_reference_normed": bf16_tap("tf raw", &tf_raw, &r_raw, 1e-4, &mut fail),
        "logits_from_reference_normed": capped("tf logits", &tf_logits, &r_logits, &r_raw, &mut fail),
    });
    // D. End to end. A normed value one BF16 step from the reference moves
    // every logit of its token by exactly dn_k * W_vk before rounding, so the
    // raw logits are held to sum_k |dn_k| |W_vk| plus the one-step-or-floor
    // rule, the softcapped ones to the same bound (tanh' <= 1).
    let e2e_normed = bf16_tap("normed", &out.normed, &r_normed, 1e-5, &mut fail);
    let unembed_bytes = root.tensor(head::UNEMBED).unwrap().2;
    let (mut raw_eq, mut raw_max, mut logit_max, mut flipped) = (0usize, 0f32, 0f32, 0usize);
    for t in 0..tokens {
        let dn: Vec<(usize, f32)> = (0..h)
            .map(|k| (k, (out.normed[t * h + k] - r_normed[t * h + k]).abs()))
            .filter(|&(_, d)| d > 0.0)
            .collect();
        flipped += dn.len();
        for vi in 0..v {
            let i = t * v + vi;
            let row = &unembed_bytes[vi * h * 2..(vi + 1) * h * 2];
            let prop: f32 = dn.iter().map(|&(k, d)| d * f32::from_bits((u16::from_le_bytes([row[2 * k], row[2 * k + 1]]) as u32) << 16).abs()).sum();
            let bound = prop * (1.0 + 1.0 / 128.0) + r_raw[i].abs() / 128.0 + 1e-4;
            let (dr, dl) = ((out.raw[i] - r_raw[i]).abs(), (out.logits[i] - r_logits[i]).abs());
            raw_eq += (out.raw[i] == r_raw[i]) as usize;
            raw_max = raw_max.max(dr);
            logit_max = logit_max.max(dl);
            if dr > bound {
                fail(format!("raw[{i}]: ours {}, reference {}, bound {bound}", out.raw[i], r_raw[i]));
            }
            if dl > bound + 1e-5 * r_logits[i].abs().max(1.0) {
                fail(format!("logits[{i}]: ours {}, reference {}, bound {bound}", out.logits[i], r_logits[i]));
            }
        }
    }
    let mut per_token = Vec::new();
    for t in 0..tokens {
        let ours = &out.logits[t * v..(t + 1) * v];
        let theirs = &r_logits[t * v..(t + 1) * v];
        let (a, b) = (argmax(ours), argmax(theirs));
        if a != b {
            fail(format!("token {t}: argmax ours {a}, reference {b}"));
        }
        per_token.push(serde_json::json!({
            "argmax": a, "reference_argmax": b,
            "top5": top(ours, 5), "reference_top5": top(theirs, 5),
            "max_logit": ours[a], "min_logit": ours.iter().cloned().fold(f32::INFINITY, f32::min),
        }));
    }
    let end_to_end = serde_json::json!({
        "selected_max_abs_diff": sel_max,
        "normed": e2e_normed,
        "normed_values_differing": flipped,
        "raw": {"bit_equal": raw_eq, "of": out.raw.len(), "max_abs_diff": raw_max},
        "logits_max_abs_diff": logit_max,
    });
    let pass = failures.is_empty();
    let rec = serde_json::json!({
        "check": "inkling2_head_check",
        "pile": pile, "collection": args[2], "root": format!("{root_id:X}"),
        "reference": refdir, "reference_source": reference["source"], "inputs": reference["inputs"],
        "mutate": mutate,
        "weights": weights,
        "tokens": tokens, "n_blocks": n_blocks, "hidden": h, "logits_per_token": v,
        "finite": finite,
        "end_to_end": end_to_end,
        "teacher_forced": teacher,
        "per_token": per_token,
        "open_seconds": open_s, "forward_seconds": forward_s,
        "pass": pass, "failures": failures,
    });
    std::fs::write(&receipt, serde_json::to_vec_pretty(&rec)?)?;
    println!("{}", serde_json::to_string_pretty(&rec)?);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
