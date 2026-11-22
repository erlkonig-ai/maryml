//! Inspect an Inkling checkpoint directory from its sidecars and safetensors
//! headers, without reading a tensor byte.
//!
//!   inkling2_inspect <checkpoint-dir>
//!
//! Reads `config.json` and decides which Inkling it is. For Inkling 2 Mini
//! (`inkling2_loop`) it reads `hf_quant_config.json` and every
//! `*.safetensors` header, checks every tensor against the layout (name,
//! dtype, shape), prints the layer schedule with its per-visit state
//! identities, and lists what Mary cannot run yet. It exits 0 only if the
//! checkpoint is an exact bijection with the layout. Running the model is a
//! separate question, which this tool answers in its last lines: Mary has no
//! native Mini forward, and says which operators are missing.
//!
//! For the original Inkling (`inkling_mm_model`) it names the family and
//! points at `inkling_layout_gate`, which checks that layout.

use std::path::PathBuf;
use std::process::ExitCode;

use mary::models::inkling2_loop::checkpoint::{inspect_dir, refuse};
use mary::models::inkling2_loop::layout::Precision;
use mary::models::inkling2_loop::schedule::Segment;
use mary::models::inkling2_loop::{Family, family_of_str};

fn main() -> ExitCode {
    let Some(dir) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: inkling2_inspect <checkpoint-dir>");
        return ExitCode::from(2);
    };
    let config = match std::fs::read_to_string(dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("reading {}: {e}", dir.join("config.json").display());
            return ExitCode::from(2);
        }
    };
    match family_of_str(&config) {
        Err(e) => {
            eprintln!("refused: {e}");
            return ExitCode::from(1);
        }
        Ok(Family::Inkling) => {
            println!(
                "family    : original Inkling (inkling_mm_model); its layout is checked by \
                 inkling_layout_gate, and it runs through models::inkling"
            );
            return ExitCode::SUCCESS;
        }
        Ok(Family::Inkling2Loop) => {}
    }
    let (cfg, ins) = match inspect_dir(&dir) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("refused: {e}");
            return ExitCode::from(1);
        }
    };
    let t = &cfg.text_config;
    let l = &t.loop_config;
    println!("family    : Inkling 2 Mini (inkling2_loop)");
    println!(
        "decoder   : {} physical layers, core {}..{} x {} passes = {} visits, hidden {}, \
         {} experts top-{}",
        t.num_hidden_layers,
        l.core_start,
        l.core_end,
        l.num_passes,
        ins.schedule.len(),
        t.hidden_size,
        t.n_routed_experts,
        t.num_experts_per_tok
    );
    println!(
        "examined  : {} files {:?}, {} tensors against {} expected slots",
        ins.files.len(),
        ins.files,
        ins.tensors,
        ins.expected_slots
    );
    println!(
        "checked   : {} shapes predicted and confirmed; {} dtype-only (no public shape): {:?}",
        ins.shape_checked,
        ins.shape_unpredicted.len(),
        ins.shape_unpredicted
    );
    let mut by: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for (i, p) in &ins.expert_precision {
        let k = match p {
            Precision::Nvfp4 { block } => format!("NVFP4/{block}"),
            Precision::Mxfp8 { block_rows, block } => format!("MXFP8/{block_rows}x{block}"),
            Precision::Bf16 => "BF16".into(),
            Precision::F32 => "F32".into(),
        };
        by.entry(k).or_default().push(*i);
    }
    println!("experts   : {by:?}");
    let seg = |s: Segment| match s {
        Segment::PreLoop => "pre",
        Segment::Core => "core",
        Segment::PostLoop => "post",
    };
    let visits: Vec<String> = ins
        .schedule
        .visits()
        .iter()
        .map(|v| {
            format!(
                "{}:{}{}/p{}{}{}",
                v.logical,
                seg(v.segment),
                v.layer(),
                v.cache.pass,
                if v.reads_messages { "+msg" } else { "" },
                if v.block_boundary { "|" } else { "" }
            )
        })
        .collect();
    println!("schedule  : {}", visits.join(" "));
    let slots: Vec<String> = ins
        .schedule
        .cache_ids()
        .map(|c| format!("({},{})->{}", c.layer, c.pass, c.upstream_slot(t)))
        .collect();
    println!(
        "state ids : (layer,pass)->transformers slot: {}",
        slots.join(" ")
    );
    for f in &ins.findings {
        println!("FINDING   : {f:?}");
    }
    let bijection = ins.is_bijection();
    println!(
        "layout    : {}",
        if bijection {
            "exact bijection".to_string()
        } else {
            format!("{} findings", ins.findings.len())
        }
    );
    if let Err(e) = refuse(&ins.gaps) {
        println!("forward   : {e}");
    }
    if bijection {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
