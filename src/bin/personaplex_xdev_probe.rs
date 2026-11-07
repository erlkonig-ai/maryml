//! Cross-device PersonaPlex temporal probe.
//!
//! Runs the SAME deterministic input through the temporal transformer and dumps
//! hidden states + logits, so a CUDA build and a Metal build can be diffed
//! against each other. Deliberately compares the two ENGINES to one another
//! rather than each to a golden file: the goldens are not on every machine, and
//! engine-vs-engine is the question actually being asked.
//!
//! Input is generated in-process from a fixed seed -- no external npy, no
//! corpus, nothing that could differ between machines.
//!
//! Build (CUDA):  --no-default-features --features qwen3tts,q4,cuda-backend
//! Build (Metal): --no-default-features --features qwen3tts,q4
use std::path::Path;
use std::time::Instant;

use mary::models::personaplex::config as cfg;
use mary::models::personaplex::temporal_metal::{Head, TemporalMetal, WeightFmt};

/// Deterministic pseudo-random activations, identical on every machine.
fn synth(step: usize, dim: usize) -> Vec<f32> {
    let mut s = (step as u32).wrapping_mul(0x9E37_79B9) | 1;
    (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            // small, centred, well inside the activation range
            (s as i32 as f32) / (i32::MAX as f32) * 0.5
        })
        .collect()
}

/// 1-minute load average. A timing run on a loaded machine is not a
/// measurement of the engine, and a run that does not record load cannot be
/// told apart from one that was clean. Recorded at both ends so RAMPING --
/// which silently steepens any curve measured over time -- is visible.
fn loadavg() -> f64 {
    std::process::Command::new("uptime")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.rsplit("average").next().map(|t| {
                t.trim_start_matches(|c: char| !c.is_ascii_digit())
                    .to_string()
            })
        })
        .and_then(|t| t.split(|c| c == ',' || c == ' ').next().map(str::to_string))
        .and_then(|t| t.parse().ok())
        .unwrap_or(f64::NAN)
}

fn stats(v: &[f32]) -> (f64, f64, f64) {
    let n = v.len() as f64;
    let sum: f64 = v.iter().map(|x| *x as f64).sum();
    let mean = sum / n;
    let var: f64 = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
    let amax = v.iter().fold(0f64, |m, x| m.max((*x as f64).abs()));
    (mean, var.sqrt(), amax)
}

/// How many steps of output to dump for the cross-machine diff. The parity
/// question is answered by the first few steps; dumping 2999 is 433 MB.
const DUMP: usize = 8;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pile = mary::paths::model(args.get(1).map(String::as_str), "personaplex.pile")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        })
        .to_string_lossy()
        .into_owned();
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let out = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "xdev.bin".to_string());
    let fmt = match args.get(4).map(|s| s.as_str()) {
        Some("f16") => WeightFmt::F16,
        Some("q8") => WeightFmt::Q8,
        _ => WeightFmt::Q4,
    };
    let head = match args.get(4).map(|s| s.as_str()) {
        Some("f16") => Head::F16,
        _ => Head::Q4,
    };

    let backend = if cfg!(feature = "cuda-backend") {
        "cuda"
    } else {
        "wgpu/metal"
    };
    println!("backend        : {backend}");
    println!("pile           : {pile}");
    println!("steps          : {steps}");
    let headname = if matches!(head, Head::F16) {
        "F16"
    } else {
        "Q4"
    };
    println!("format         : {fmt:?} / head {headname}");

    let t0 = Instant::now();
    let loader = mary::persist::personaplex_loader(Path::new(&pile))
        .unwrap_or_else(|e| panic!("pile load failed: {e}"));
    println!("loader ready   : {:.2}s", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    // Recompute the temporal transform from the signed source model on this
    // CUDA probe path.
    let mut tm = TemporalMetal::load(&loader, fmt);
    println!("model loaded   : {:.2}s", t1.elapsed().as_secs_f64());

    let mut all_h: Vec<f32> = Vec::new();
    let mut all_l: Vec<f32> = Vec::new();
    let t2 = Instant::now();
    let mut per_step_ms: Vec<f64> = Vec::new();
    let load_start = loadavg();
    for s in 0..steps {
        let x = synth(s, cfg::DIM);
        let ts = Instant::now();
        let (h, l) = tm.step(&x, head);
        per_step_ms.push(ts.elapsed().as_secs_f64() * 1e3);
        if s < DUMP {
            all_h.extend_from_slice(&h);
            all_l.extend_from_slice(&l);
        }
    }
    let total = t2.elapsed().as_secs_f64();
    let load_end = loadavg();

    let (hm, hs, ha) = stats(&all_h);
    let (lm, ls, la) = stats(&all_l);
    println!(
        "\nhidden  mean={hm:+.6} sd={hs:.6} amax={ha:.6}  ({} values)",
        all_h.len()
    );
    println!(
        "logits  mean={lm:+.6} sd={ls:.6} amax={la:.6}  ({} values)",
        all_l.len()
    );

    // CONTEXT CURVE: step time as a function of accumulated KV, which is the
    // question. A flat curve means weight-streaming dominates; a rising one
    // means attention does. Windowed medians so a single stall cannot move it.
    let raw = per_step_ms.clone();
    println!("\nctx window     median ms   % of 80ms budget");
    let mut marks: Vec<usize> = vec![];
    let mut w = 64usize;
    while w < raw.len() {
        marks.push(w);
        w *= 2;
    }
    marks.push(raw.len());
    let mut prev = 0usize;
    for m in marks {
        let lo = prev.max(m.saturating_sub(128));
        let mut win: Vec<f64> = raw[lo..m].to_vec();
        if win.is_empty() {
            continue;
        }
        win.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let md = win[win.len() / 2];
        println!("  {:>5}..{:<6} {:>9.2}   {:>6.1}%", lo, m, md, md / 0.8);
        prev = m;
    }
    let mut sorted = raw.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    // first-step cost is kernel compilation, report it apart from steady state
    println!(
        "\nfirst step {:.1}ms (kernel compile)   overall median {:.2}ms   total {:.2}s",
        raw[0], med, total
    );
    println!(
        "12.5Hz frame budget is 80ms -> overall median is {:.1}% of budget",
        med / 0.8
    );
    println!("machine load: {load_start:.2} at start -> {load_end:.2} at end");
    if load_start > 2.0 || load_end > 2.0 || (load_end - load_start).abs() > 1.0 {
        println!("  !! LOADED OR RAMPING -- these timings are not a measurement of the engine");
    }

    // raw f32 dump for an exact cross-machine diff
    let mut bytes = Vec::with_capacity((all_h.len() + all_l.len()) * 4);
    for v in all_h.iter().chain(all_l.iter()) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("write {out}: {e}"));
    println!(
        "\nwrote {} ({} bytes) -- diff this against the other machine",
        out,
        bytes.len()
    );
}
