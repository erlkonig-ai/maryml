//! Inkling 2 Mini's post-loop tail (layers 18..23 and the head) on the
//! release's weights read from a model pile, against transformers 046bdb8c's
//! own run on the same bytes ([`mary::models::inkling2_loop::tail_check`]).
//!
//! The record is written by `golden/capture_inkling2_tail.py real` from the
//! release's safetensors file. It carries the sha256 of every tail tensor
//! upstream read; this check hashes the pile's tensors and refuses to compare
//! unless they are the same bytes.
//!
//! usage: inkling2_tail_check <pile> <collection> <root> <record.safetensors> <receipt.json> [--mutate <mutation>]
//!
//! `--mutate` breaks one piece (nvfp4-amax, mxfp8-scale, mxfp8-gate-up,
//! no-boundary, origin) and the check must then fail; the exit code says
//! whether it did what was asked (0: clean run passed, or mutated run failed).

use anyhow::{Context, Result, bail, ensure};
use mary::models::inkling2_loop::Inkling2LoopConfig;
use mary::models::inkling2_loop::pile::MiniRoot;
use mary::models::inkling2_loop::tail_check::{Mutation, Record, TOL, check};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;
use triblespace::prelude::*;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 6 || args.len() == 8,
        "usage: {} <pile> <collection> <root> <record.safetensors> <receipt.json> [--mutate <mutation>]",
        args[0]
    );
    let pile = PathBuf::from(&args[1]);
    let handle = mary::model_collection::parse_collection_handle(&args[2])?;
    let root_id = Id::from_hex(&args[3]).context("root must be a 32-hex id")?;
    let record = Record::read(std::path::Path::new(&args[4])).map_err(anyhow::Error::msg)?;
    let receipt = PathBuf::from(&args[5]);
    let mutation = match args.get(6..8) {
        None => None,
        Some([flag, m]) if flag == "--mutate" => Some(match m.as_str() {
            "nvfp4-amax" => Mutation::Nvfp4Amax,
            "mxfp8-scale" => Mutation::Mxfp8Scale,
            "mxfp8-gate-up" => Mutation::Mxfp8GateUp,
            "no-boundary" => Mutation::NoBoundary,
            "origin" => Mutation::Origin,
            other => bail!("unknown mutation {other}"),
        }),
        Some(other) => bail!("unknown option {other:?}"),
    };

    let t0 = Instant::now();
    let root = MiniRoot::open(&pile, handle, root_id)?;
    let open_s = t0.elapsed().as_secs_f64();
    let cfg = Inkling2LoopConfig::from_json(&serde_json::to_string(&root.config()?)?).map_err(anyhow::Error::msg)?;
    let text = &cfg.text_config;

    // Same bytes, or no comparison.
    let digests: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(record.meta("weight_sha256").context("the record has no weight digests")?)?;
    ensure!(!digests.is_empty(), "the record lists no weights");
    let t1 = Instant::now();
    for (name, theirs) in &digests {
        let (_, _, bytes) = root.tensor(name).map_err(anyhow::Error::msg)?;
        let ours = format!("{:x}", Sha256::digest(bytes));
        ensure!(Some(ours.as_str()) == theirs.as_str(), "{name}: pile bytes sha256 {ours}, upstream used {theirs}");
    }
    let hash_s = t1.elapsed().as_secs_f64();
    println!("{} tail tensors: the pile's bytes are upstream's ({hash_s:.1} s)", digests.len());

    let dev = burn::backend::cuda::CudaDevice::default();
    let t2 = Instant::now();
    let rep = check(&root, text, &record, mutation, &dev).map_err(anyhow::Error::msg)?;
    let run_s = t2.elapsed().as_secs_f64();
    for l in &rep.lines {
        println!("{l}");
    }
    let passed = rep.fails.is_empty();
    println!("worst ratio {:.2} (tolerance {TOL}), {} failures, {run_s:.1} s", rep.worst, rep.fails.len());
    let json = format!(
        "{{\"scope\":\"Inkling 2 Mini layers 18..23 and the output head, one prefill from fixed inputs, Mary's device layers (CUDA) and CPU head on the pile's weights against transformers 046bdb8c on CPU with oneDNN off\",\
         \"pile\":{:?},\"root\":\"{:X}\",\"inputs\":{:?},\"tolerance\":\"rel L2 to upstream f32 within {TOL} x upstream's bf16-vs-f32 rel L2 on the same input; routed sets equal to upstream f32's or bf16's; argmax equal to upstream f32's or bf16's\",\
         \"mutation\":{:?},\"passed\":{passed},\"worst_ratio\":{},\"failures\":{:?},\"open_s\":{open_s},\"hash_s\":{hash_s},\"run_s\":{run_s},\"rows\":[{}]}}\n",
        pile.display().to_string(),
        root.root(),
        record.meta("inputs").unwrap_or(""),
        mutation.map(|m| format!("{m:?}")),
        rep.worst,
        rep.fails,
        rep.json.join(",")
    );
    std::fs::write(&receipt, json).with_context(|| format!("{}", receipt.display()))?;
    match (mutation, passed) {
        (None, true) => Ok(()),
        (None, false) => bail!("the tail is outside tolerance: {:?}", rep.fails),
        (Some(m), false) => {
            println!("{m:?}: failed as it must");
            Ok(())
        }
        (Some(m), true) => bail!("{m:?} passed: the check cannot see it"),
    }
}
