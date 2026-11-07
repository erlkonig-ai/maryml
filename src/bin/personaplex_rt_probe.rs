//! PersonaPlex-7B **realtime** gates — the q4/Metal temporal build
//! (`personaplex::temporal_metal`) against the moshi CPU-f32 oracle goldens
//! and the 80 ms/frame clock. Companion to `personaplex_probe` (the CPU-f32
//! parity gates); this probe owns the HONEST-q4 bars: q4 is a real numerics
//! change, so the gate is logits cosine (~0.999x expected, NOT 1.0) plus
//! argmax agreement and the first-divergence step — never "token-exact".
//!
//!   cargo run --release --features personaplex,q4 --bin personaplex_rt_probe -- \
//!     <gate|bench|quantcheck|pipeline|framebench|depthab|reset> [q4|q8|f16] \
//!     [--depth-f16] [pile-path]
//!
//! The optional format token picks the stack's [`WeightFmt`] (default q4):
//! f16 is the pipeline-exactness ablation, q8/q4 the two quantized builds.
//!
//! - `gate`  — runs the oracle's 113-step golden stream (50 voice-prompt
//!   embedding rows + 63 token steps, inputs derived exactly as the CPU
//!   parity gate derives them) through the Metal build and reports per-step
//!   hidden/text-logits cosine vs `tt_hidden`/`tt_text_logits`, the text
//!   argmax agreement rate + first mismatch step, per-phase minima, and the
//!   f16-vs-q4 logit-head A/B (both heads run on the SAME hidden state).
//!   PASS bars are per-format regression tripwires calibrated to each
//!   format's measured class (`bars`), with the actual numbers printed for
//!   the PORT_NOTES record.
//! - `bench` — ms/step at cache fills 256/1024/3000 (min-of-medians over
//!   rounds, submit vs full split, both head variants, contention canary).
//!   The fill cursor is pinned per step (`force_len`), so every measured
//!   step attends over exactly the stated prefix.
//! - `depthab` — the PAIRED depth-arm A/B. Both depth arms (`depth_fast` on
//!   the host, `depth_gpu` on the device) are resident in one process and BOTH
//!   run on every frame from the same temporal hidden state, alternating which
//!   goes first; the CPU arm's tokens are the committed ones. Two sequential
//!   `framebench` runs cannot compare the arms on a shared desktop — the
//!   ambient load between them drifted 3x, which is larger than the effect —
//!   so this is the mode that answers "which arm is faster HERE". Reports the
//!   per-frame depth cost of both arms and the LM step reconstructed for each.
//! - `quantcheck` — the why-not-fold-the-norm-alphas measurement: per-matvec
//!   q4 relative RMS error of the raw q_proj vs the alpha-folded q_proj
//!   (folding a high-dynamic-range column scale into a weight quantized in
//!   32-input-channel groups inflates the within-group spread; the first
//!   build folded and collapsed to hidden cos ~0.895 end-to-end).
//! - `pipeline` — the assembled realtime pipeline (`RealtimePipeline`: Metal
//!   quantized temporal + f16 logit head + Accelerate/NEON depformer + CPU
//!   Mimi) FREE-RUNNING the oracle's golden input flow (WAV → Mimi encode →
//!   prompts → 25 user-audio frames → Mimi decode → WAV). Reports token
//!   agreement vs `out_tokens`/`step_tokens`, the FIRST committed-divergence
//!   step (with quantized weights the free run WILL diverge at some point —
//!   the gate verifies the divergence is an argmax near-tie flip, not
//!   garbage: logits cos + flip margins at that step), and audio cos vs the
//!   parity pipeline output over the pre-divergence prefix. Writes
//!   /tmp/mary-personaplex/rt_pipeline_out.wav. It also takes one final
//!   output-only step and gates the public trace's user inputs to `-1`.
//!   `--depth-f16` stores the depformer as f16 (f32 path is the default).
//!
//! Env: RT_ROUNDS (default 5), RT_STEPS (default 16),
//! MARY_DEPTH_THREADS/MARY_PRED_THREADS (depformer pools).

use mary::models::f5::wav;
use mary::models::personaplex::config as cfg;
use mary::models::personaplex::depth::argmax;
use mary::models::personaplex::mimi::config as mimi_cfg;
use mary::models::personaplex::pipeline::{RealtimePipeline, SILENCE, SINE, agent_codes};
use mary::models::personaplex::temporal_metal::{Head, TemporalMetal, WeightFmt};
use mary::nn::npy;
use mary::nn::weight_loader::WeightLoader;
use std::path::Path;
use std::time::Instant;

const GOLD: &str = "/tmp/mary-personaplex/golden";

/// Per-format red-line bars (regression tripwires set just under the
/// MEASURED class of each format — see PORT_NOTES for the measured numbers;
/// they are honesty-calibrated, not quality claims):
/// (min hidden cos, min logits cos, mean logits cos, argmax rate).
fn bars(fmt: WeightFmt) -> (f64, f64, f64, f64) {
    match fmt {
        // f16 is the pipeline-exactness ablation: near-parity or bust.
        WeightFmt::F16 => (0.999, 0.999, 0.9999, 0.99),
        WeightFmt::Q8 => (0.98, 0.98, 0.995, 0.95),
        // q4_0's real class on this checkpoint (per-matvec ~8.7e-2 compounds
        // over 64 residual adds — see temporal_metal docs).
        WeightFmt::Q4 => (0.55, 0.30, 0.90, 0.75),
    }
}

fn golden_f32(name: &str) -> (Vec<f32>, Vec<usize>) {
    npy::load_npy(&Path::new(GOLD).join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
}

fn golden_i64(name: &str) -> (Vec<i64>, Vec<usize>) {
    npy::load_npy_i64(&Path::new(GOLD).join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
}

/// (cosine similarity, max |Δ|) in f64 accumulation.
fn cos_maxd(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxd = maxd.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxd)
}

/// Lazy handle-indexed pile loader (nothing materialized wholesale).
fn pile_loader(pile: &str) -> WeightLoader {
    mary::persist::personaplex_loader(Path::new(pile)).unwrap_or_else(|e| panic!("pile load: {e}"))
}

fn runtime_source(pile: &str) -> mary::models::personaplex::PersonaPlexRuntimeSource {
    mary::persist::personaplex_model(Path::new(pile))
        .unwrap_or_else(|e| panic!("model load: {e}"))
        .into_runtime_source()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn load_model(pile: &str, fmt: WeightFmt) -> TemporalMetal {
    println!("loading ({fmt:?}) temporal transformer from {pile} …");
    let t0 = Instant::now();
    let loader = pile_loader(pile);
    // Unsigned transformed caches are not numerical authority. Use the
    // deterministic transform path until signed-equation admission lands.
    let tm = TemporalMetal::load(&loader, fmt);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    tm
}

// ─────────────────────────────── gate ───────────────────────────────

fn gate(pile: &str, fmt: WeightFmt) {
    let (vp, vps) = golden_f32("vp_embeddings"); // [50, 1, 1, 4096]
    assert_eq!(&vps[1..], &[1, 1, cfg::DIM], "vp_embeddings shape");
    let n_vp = vps[0];
    let (toks, ts) = golden_i64("step_tokens"); // [63, 17]
    assert_eq!(ts[1], cfg::NUM_STREAMS, "step_tokens shape");
    let (gh, ghs) = golden_f32("tt_hidden"); // [113, 4096]
    let (gl, gls) = golden_f32("tt_text_logits"); // [113, 32000]
    let steps = ghs[0];
    assert_eq!(steps, n_vp + ts[0], "temporal steps = vp + token steps");
    assert_eq!(gls, vec![steps, cfg::TEXT_LOGITS]);
    println!(
        "goldens: {steps} temporal steps ({n_vp} embedding-fed + {} token-fed)",
        ts[0]
    );

    let mut tm = load_model(pile, fmt);

    let (mut min_hcos, mut min_lcos, mut min_qcos) = (1f64, 1f64, 1f64);
    let (mut max_hd, mut max_ld) = (0f64, 0f64);
    let (mut worst_h, mut worst_l) = (0usize, 0usize);
    let (mut hits_f16, mut hits_q4) = (0usize, 0usize);
    let (mut first_miss_f16, mut first_miss_q4): (Option<usize>, Option<usize>) = (None, None);
    let mut per_step: Vec<(f64, f64)> = Vec::with_capacity(steps);
    let (mut hsum, mut lsum, mut qsum) = (0f64, 0f64, 0f64);
    let t0 = Instant::now();
    for s in 0..steps {
        let x: Vec<f32> = if s < n_vp {
            vp[s * cfg::DIM..(s + 1) * cfg::DIM].to_vec()
        } else {
            tm.embed_codes(&toks[(s - n_vp) * cfg::NUM_STREAMS..(s - n_vp + 1) * cfg::NUM_STREAMS])
        };
        let (h, l) = tm.step(&x, Head::F16);
        tm.submit_head(Head::Q4);
        let lq = tm.read_logits();

        let (hcos, hd) = cos_maxd(&h, &gh[s * cfg::DIM..(s + 1) * cfg::DIM]);
        let glrow = &gl[s * cfg::TEXT_LOGITS..(s + 1) * cfg::TEXT_LOGITS];
        let (lcos, ld) = cos_maxd(&l, glrow);
        let (qcos, _) = cos_maxd(&lq, glrow);
        if hcos < min_hcos {
            min_hcos = hcos;
            worst_h = s;
        }
        if lcos < min_lcos {
            min_lcos = lcos;
            worst_l = s;
        }
        min_qcos = min_qcos.min(qcos);
        max_hd = max_hd.max(hd);
        max_ld = max_ld.max(ld);
        hsum += hcos;
        lsum += lcos;
        qsum += qcos;
        let gold_am = argmax(glrow);
        if argmax(&l) == gold_am {
            hits_f16 += 1;
        } else if first_miss_f16.is_none() {
            first_miss_f16 = Some(s);
        }
        if argmax(&lq) == gold_am {
            hits_q4 += 1;
        } else if first_miss_q4.is_none() {
            first_miss_q4 = Some(s);
        }
        per_step.push((hcos, lcos));
        if (s + 1) % 16 == 0 || s + 1 == steps {
            eprintln!(
                "  step {:3}/{steps}  hidden cos={hcos:.6}  logits cos={lcos:.6}  ({:.0} ms/step)",
                s + 1,
                t0.elapsed().as_secs_f64() * 1e3 / (s + 1) as f64
            );
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    // per-phase minima (localize any divergence to a prompt phase)
    let phase = |name: &str, lo: usize, hi: usize| {
        let hmin = per_step[lo..hi].iter().map(|p| p.0).fold(1f64, f64::min);
        let lmin = per_step[lo..hi].iter().map(|p| p.1).fold(1f64, f64::min);
        println!(
            "  phase {name:<22} steps {lo:3}..{hi:3}  min hidden cos={hmin:.6}  min logits cos={lmin:.6}"
        );
    };
    println!("phases:");
    phase("voice prompt (embed)", 0, 50);
    phase("silence 1", 50, 56);
    phase("text prompt", 56, 82);
    phase("silence 2", 82, 88);
    phase("user audio (gen)", 88, steps);

    let n = steps as f64;
    let (bar_h, bar_l, bar_lmean, bar_am) = bars(fmt);
    let ok_h = min_hcos >= bar_h;
    let ok_l = min_lcos >= bar_l && lsum / n >= bar_lmean;
    let ok_am = hits_f16 as f64 / n >= bar_am;
    println!(
        "  {} tt hidden        min cos={min_hcos:.6} (step {worst_h})  mean={:.6}  max|Δ|={max_hd:.3e}",
        if ok_h { "OK" } else { "XX" },
        hsum / n
    );
    println!(
        "  {} logits (f16 head) min cos={min_lcos:.6} (step {worst_l})  mean={:.6}  max|Δ|={max_ld:.3e}",
        if ok_l { "OK" } else { "XX" },
        lsum / n
    );
    println!(
        "  -- logits (q4 head)  min cos={min_qcos:.6}                 mean={:.6}",
        qsum / n
    );
    println!(
        "  {} text argmax (f16) {hits_f16}/{steps} ({:.1}%)  first miss: {}",
        if ok_am { "OK" } else { "XX" },
        100.0 * hits_f16 as f64 / n,
        first_miss_f16.map_or("none".into(), |s| format!("step {s}")),
    );
    println!(
        "  -- text argmax (q4)  {hits_q4}/{steps} ({:.1}%)  first miss: {}",
        100.0 * hits_q4 as f64 / n,
        first_miss_q4.map_or("none".into(), |s| format!("step {s}")),
    );
    println!(
        "ran {steps} steps in {secs:.1}s ({:.0} ms/step incl. 3 readbacks + host embed)",
        secs * 1e3 / n
    );
    println!(
        "NOTE: quantized weights are a real numerics change — {fmt:?} bars are min hidden ≥ {bar_h}, \
         min/mean logits ≥ {bar_l}/{bar_lmean}, argmax ≥ {:.0}% (regression tripwires at the \
         format's measured class, not exactness claims).",
        bar_am * 100.0
    );

    if ok_h && ok_l && ok_am {
        println!("PERSONAPLEX TEMPORAL-METAL GATE ({fmt:?}): PASS");
    } else {
        println!("PERSONAPLEX TEMPORAL-METAL GATE ({fmt:?}): FAIL");
        std::process::exit(1);
    }
}

// ─────────────────────────────── bench ───────────────────────────────

/// Deterministic pseudo-random embedding-scale input (timing is
/// value-independent; this just keeps activations finite).
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32;
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bench(pile: &str, fmt: WeightFmt) {
    let rounds: usize = std::env::var("RT_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let steps: usize = std::env::var("RT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    println!(
        "bench ({fmt:?}): {rounds} rounds x {steps} steps, min-of-medians, fill pinned per step"
    );
    println!("(desktop machine — ambient GPU contention inflates; compare within a window)");

    let mut tm = load_model(pile, fmt);
    let x = fill(cfg::DIM, 7, 0.02);

    // one warm pass per shape (JIT compile) before any timing
    for head in [Head::F16, Head::Q4] {
        tm.force_len(0);
        tm.step_submit(&x, head);
        let _ = tm.read_logits();
    }

    println!(
        "\n{:>5}  {:>9}  {:>28}  {:>28}",
        "fill", "", "f16 head (submit/full ms)", "q4 head (submit/full ms)"
    );
    for &filln in &[256usize, 1024, 3000] {
        let fillp = filln.min(mary::models::personaplex::temporal_metal::MAX_SEQ - 1);
        let mut cells = Vec::new();
        for head in [Head::F16, Head::Q4] {
            // warm at this fill
            for _ in 0..3 {
                tm.force_len(fillp);
                tm.step_submit(&x, head);
                let _ = tm.read_logits();
            }
            let mut submit_meds = Vec::new();
            let mut full_meds = Vec::new();
            for _ in 0..rounds {
                let mut subs = Vec::new();
                let mut fulls = Vec::new();
                for _ in 0..steps {
                    tm.force_len(fillp); // every step attends over exactly fillp+1
                    let t0 = Instant::now();
                    tm.step_submit(&x, head);
                    let submit = t0.elapsed().as_secs_f64();
                    let _ = tm.read_logits();
                    let full = t0.elapsed().as_secs_f64();
                    subs.push(submit * 1e3);
                    fulls.push(full * 1e3);
                }
                submit_meds.push(median(subs));
                full_meds.push(median(fulls));
            }
            let sub = submit_meds.iter().cloned().fold(f64::INFINITY, f64::min);
            let full = full_meds.iter().cloned().fold(f64::INFINITY, f64::min);
            cells.push((sub, full));
        }
        println!(
            "{:>5}  {:>9}  {:>13.2} / {:<12.2}  {:>13.2} / {:<12.2}",
            filln, "", cells[0].0, cells[0].1, cells[1].0, cells[1].1
        );
    }
    println!("\nframe budget: 80 ms @ 12.5 Hz; spike projection allots temporal ~15.6 ms +");
    println!("depth ~21.6 + mimi ~5 + submission ~5 (47.6/48.9/52.2 ms at ctx 256/1024/3000).");
}

// ───────────────────────────── quantcheck ─────────────────────────────

/// Per-matvec q4 error, raw weight vs norm-alpha-folded weight — the
/// measurement behind temporal_metal's "alphas apply in the norm kernels,
/// never folded into the q4 weights" rule.
fn quantcheck(pile: &str) {
    use mary::nn::q4::{dequantize_q4, quantize_q4};
    let loader = pile_loader(pile);
    let d = cfg::DIM;
    let rel_rms = |w: &[f32]| -> f64 {
        let (wq, sc) = quantize_q4(w, d, d);
        let wd = dequantize_q4(&wq, &sc, d, d);
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in w.iter().zip(&wd) {
            num += ((a - b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        (num / den).sqrt()
    };
    println!("q4 rel RMS of q_proj: raw (what the build quantizes) vs alpha-folded");
    println!("(the analytic q4_0 class on well-scaled weights is ~3e-2..6e-2)\n");
    for li in [0usize, 8, 16, 24, 31] {
        let src = format!("transformer.layers.{li}");
        let (a1, s) = loader.load_f32(&format!("{src}.norm1.alpha"));
        assert_eq!(s, vec![1, 1, d]);
        let (in_proj, s) = loader.load_f32(&format!("{src}.self_attn.in_proj_weight"));
        assert_eq!(s, vec![3 * d, d]);
        let q_w = &in_proj[..d * d];

        let amin = a1.iter().fold(f32::INFINITY, |m, &v| m.min(v.abs()));
        let amax = a1.iter().fold(0f32, |m, &v| m.max(v.abs()));
        // worst |alpha| dynamic range inside one 32-input-channel q4 group
        let worst_grp = a1
            .chunks(32)
            .map(|g| {
                let mn = g.iter().fold(f32::INFINITY, |m, &v| m.min(v.abs()));
                let mx = g.iter().fold(0f32, |m, &v| m.max(v.abs()));
                mx / mn.max(1e-30)
            })
            .fold(0f32, f32::max);

        let mut folded = q_w.to_vec();
        for row in folded.chunks_exact_mut(d) {
            for (x, a) in row.iter_mut().zip(&a1) {
                *x *= a;
            }
        }
        println!(
            "  layer {li:2}: |alpha| range [{amin:.3e}, {amax:.3e}] (worst in-group ratio {worst_grp:.1}x)   raw {:.3e}   folded {:.3e}",
            rel_rms(q_w),
            rel_rms(&folded)
        );
    }
}

// ───────────────────────────── framebench ─────────────────────────────

/// One measured frame's wall-clock decomposition (ms).
struct FrameRow {
    total: f64,
    /// Temporal stage: host embed + 290-dispatch submit + GPU drain +
    /// the combined hidden+logits readback.
    temporal: f64,
    /// Host-side share of `temporal` (embed + kernel encode, pre-drain).
    submit: f64,
    /// The 16 sequential depformer steps (CPU).
    depth: f64,
    /// Mimi decode of THIS frame (1-frame stateless decode), measured ON THE
    /// DECODE WORKER — off the LM critical path (`total` excludes it). Decode
    /// is downstream-only: frame t's PCM is not an input to any later step,
    /// so a live loop overlaps it with the next LM step; the worker column
    /// verifies its throughput stays far under the frame budget.
    mimi: f64,
}

/// Min-of-medians over consecutive `round`-sized chunks (spike methodology:
/// contention only inflates, so the best round's median approaches the
/// quiet-machine floor).
fn min_of_medians(rows: &[FrameRow], round: usize, f: impl Fn(&FrameRow) -> f64) -> f64 {
    rows.chunks(round)
        .filter(|c| c.len() == round)
        .map(|c| median(c.iter().map(&f).collect()))
        .fold(f64::INFINITY, f64::min)
}

/// The machine's 1-minute load average, so every window carries the
/// contention it was measured under. A frame time without it is not evidence
/// on a shared desktop.
fn loadavg() -> f64 {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.trim()
                .trim_start_matches('{')
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(f64::NAN)
}

/// **Paired** depth-arm A/B on the real LM critical path: both the CPU
/// predictor and the cubecl port resident in ONE process, driven from the
/// SAME temporal hidden state on the SAME frame, alternating which arm goes
/// first.
///
/// [`framebench`] measures one arm per run, and on this machine that is not
/// enough to compare them: two sequential framebench runs of the two arms
/// drifted by 3x in ambient load between them, which is larger than the
/// effect. Pairing removes that entirely — the two numbers in a row saw the
/// same machine at the same microsecond.
///
/// The CPU arm's tokens are the ones committed, so the trajectory is exactly
/// what shipping today produces; the GPU arm's frame is computed and thrown
/// away. Both see the same `forced` array and the same `next_text`.
///
/// The cost of pairing is that each arm runs with the OTHER arm's load also
/// on the machine, so both columns are inflated relative to a solo run — the
/// RATIO is what this mode is for, not the absolute.
fn depth_ab(pile: &str, fmt: WeightFmt) {
    use mary::models::personaplex::depth_gpu::DepthFmt;
    use mary::models::personaplex::pipeline::{DepthChoice, load_depth_arm};
    use mary::models::personaplex::sampling::{Sampler, SamplingConfig};
    use mary::models::personaplex::temporal_metal::MAX_SEQ;
    const WINDOW: usize = 80;
    const BUDGET_MS: f64 = 80.0;
    let targets = [256usize, 1024, MAX_SEQ - 1];
    // RT_AB_SAMPLE runs both arms in the SHIPPING decode config instead of
    // greedy. It is not cosmetic: the host arm pays a 2048-wide sort per step
    // for sampling and the device arm pays two threshold bisections in the
    // kernel, so the greedy ratio is not a stand-in for the sampled one.
    let sampled = std::env::var("RT_AB_SAMPLE").is_ok();
    let scfg = SamplingConfig {
        temp: 0.8,
        top_k: 250,
        top_p: 0.95,
    };
    // One sampler per arm, same seed: neither arm consumes the other's RNG,
    // so the pairing does not change either arm's token stream.
    let mut cpu_smp = sampled.then(|| Sampler::new(scfg, 1234_5678));
    let mut gpu_smp = sampled.then(|| Sampler::new(scfg, 1234_5678));
    let gpu_fmt = std::env::var("RT_AB_FMT")
        .ok()
        .map(|s| DepthFmt::parse(&s).unwrap_or_else(|| panic!("bad RT_AB_FMT {s}")))
        .unwrap_or(DepthFmt::uniform(WeightFmt::Q8));

    let t0 = Instant::now();
    let source = runtime_source(pile);
    let mut p = RealtimePipeline::load_with_depth(&source, fmt, DepthChoice::Cpu { f16: true });
    let mut gpu = load_depth_arm(&source, DepthChoice::Gpu(gpu_fmt));
    println!(
        "depth A/B ({fmt:?} temporal + f16 head + CPU mimi): {} vs {}",
        p.depth.label(),
        gpu.label()
    );
    println!(
        "decode: {}",
        if sampled {
            format!(
                "sampled temp {} top_k {} top_p {} (the shipping config)",
                scfg.temp, scfg.top_k, scfg.top_p
            )
        } else {
            "greedy".to_string()
        }
    );
    println!(
        "loaded both arms in {:.1}s (load {:.1})",
        t0.elapsed().as_secs_f64(),
        loadavg()
    );

    // (fill-window index) -> rows of (temporal, cpu depth, gpu depth, other)
    let mut windows: Vec<Vec<[f64; 4]>> = targets.iter().map(|_| Vec::new()).collect();
    let mut win_load: Vec<f64> = targets.iter().map(|_| f64::NAN).collect();
    let skip: Option<usize> = std::env::var("RT_FB_SKIP")
        .ok()
        .and_then(|s| s.parse().ok());
    let (tx, rx) = std::sync::mpsc::sync_channel::<[u32; mimi_cfg::NUM_CODEBOOKS]>(8);
    let decoder = &p.decoder;
    let run0 = Instant::now();
    std::thread::scope(|sc| {
        // The mimi decode worker is downstream-only, but it is real CPU load
        // that the host depth arm competes with, so it stays on.
        sc.spawn(move || {
            while let Ok(frame) = rx.recv() {
                let _ = decoder.decode(&[frame]);
            }
        });
        let mut frame_no = 0usize;
        loop {
            let fill = p.temporal.len();
            if fill >= MAX_SEQ {
                break;
            }
            if let Some(target) = skip {
                if fill > 4 && fill + WINDOW < target {
                    p.temporal.force_len(target + 1 - WINDOW);
                    continue;
                }
            }
            let win = targets.iter().position(|&t| fill <= t && fill + WINDOW > t);
            let Some(pr) = p.stream.prepare(Some(&SINE), None, None) else {
                continue; // the offset-0 ring-seeding call — no model step
            };
            let ts = Instant::now();
            let x = p.temporal.embed_codes(&pr.input);
            p.temporal.step_submit(&x, p.head);
            let (hidden, logits) = p.temporal.read_hidden_logits();
            let temporal = ts.elapsed().as_secs_f64() * 1e3;
            let sampled = argmax(&logits) as i64;
            let next_text = if pr.provided[0] {
                pr.target[0]
            } else {
                sampled
            };
            let forced = pr.forced();

            // Alternate the order so neither arm keeps the second-mover's
            // warm cache (or its cold queue) for the whole window.
            let (t_cpu, t_gpu, dep);
            if frame_no % 2 == 0 {
                let t = Instant::now();
                dep = p.depth.frame(&hidden, next_text, &forced, cpu_smp.as_mut());
                t_cpu = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let _ = gpu.frame(&hidden, next_text, &forced, gpu_smp.as_mut());
                t_gpu = t.elapsed().as_secs_f64() * 1e3;
            } else {
                let t = Instant::now();
                let _ = gpu.frame(&hidden, next_text, &forced, gpu_smp.as_mut());
                t_gpu = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                dep = p.depth.frame(&hidden, next_text, &forced, cpu_smp.as_mut());
                t_cpu = t.elapsed().as_secs_f64() * 1e3;
            }
            frame_no += 1;

            let tc = Instant::now();
            let out = p.stream.commit(&pr, sampled, &dep);
            if let Some(o) = &out {
                tx.send(agent_codes(o)).expect("decode worker alive");
            }
            let other = tc.elapsed().as_secs_f64() * 1e3;

            if let Some(w) = win {
                if windows[w].len() < WINDOW {
                    if windows[w].is_empty() {
                        win_load[w] = loadavg();
                    }
                    windows[w].push([temporal, t_cpu, t_gpu, other]);
                }
            }
            if fill % 256 == 0 {
                println!(
                    "  fill {fill:4}/{}  ({:.1} min elapsed, load {:.1})",
                    MAX_SEQ - 1,
                    run0.elapsed().as_secs_f64() / 60.0,
                    loadavg()
                );
            }
        }
        drop(tx);
    });

    // ── report ──
    let stat = |v: &mut Vec<f64>| -> (f64, f64, f64) {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (
            mean,
            v[(v.len() - 1) / 2],
            v[((v.len() as f64 - 1.0) * 0.95) as usize],
        )
    };
    println!(
        "\ndepth stage, PAIRED per frame ({WINDOW} frames per window; both arms ran on every frame, so both columns carry the other's load):"
    );
    println!(
        "{:>6}  {:>5}  {:>21}  {:>21}  {:>7}",
        "fill", "load", "cpu depth mean/p50/p95", "gpu depth mean/p50/p95", "speedup"
    );
    for (w, &t) in targets.iter().enumerate() {
        if windows[w].is_empty() {
            continue;
        }
        let (cm, c50, c95) = stat(&mut windows[w].iter().map(|r| r[1]).collect());
        let (gm, g50, g95) = stat(&mut windows[w].iter().map(|r| r[2]).collect());
        println!(
            "{t:>6}  {:>5.1}  {cm:>6.1}/{c50:>6.1}/{c95:>6.1}  {gm:>6.1}/{g50:>6.1}/{g95:>6.1}  {:>6.2}x",
            win_load[w],
            cm / gm
        );
    }
    println!(
        "\nLM step reconstructed per arm (temporal + that arm's depth + commit; the other arm's time removed):"
    );
    println!(
        "{:>6}  {:>8}  {:>21}  {:>13}  {:>21}  {:>13}",
        "fill",
        "temporal",
        "cpu step mean/p50/p95",
        "over 80 / rt",
        "gpu step mean/p50/p95",
        "over 80 / rt"
    );
    for (w, &t) in targets.iter().enumerate() {
        if windows[w].is_empty() {
            continue;
        }
        let (tm, _, _) = stat(&mut windows[w].iter().map(|r| r[0]).collect());
        let mut c: Vec<f64> = windows[w].iter().map(|r| r[0] + r[1] + r[3]).collect();
        let mut g: Vec<f64> = windows[w].iter().map(|r| r[0] + r[2] + r[3]).collect();
        let over = |v: &[f64]| v.iter().filter(|&&x| x > BUDGET_MS).count();
        let (co, go) = (over(&c), over(&g));
        let (cm, c50, c95) = stat(&mut c);
        let (gm, g50, g95) = stat(&mut g);
        println!(
            "{t:>6}  {tm:>8.1}  {cm:>6.1}/{c50:>6.1}/{c95:>6.1}  {co:>4}/{WINDOW} {:>5.2}x  {gm:>6.1}/{g50:>6.1}/{g95:>6.1}  {go:>4}/{WINDOW} {:>5.2}x",
            BUDGET_MS / cm,
            BUDGET_MS / gm
        );
    }
    println!(
        "\nbudget 80 ms @ 12.5 Hz. `rt` = 80 ms / mean step; < 1.00x means the loop falls behind and frames are skipped."
    );
}

/// The measurement that decides whether the pipeline runs realtime: wall clock
/// per EMITTED FRAME on the LM critical path — temporal step + 16 depformer
/// steps + all submission/bookkeeping overhead — at temporal-cache fills
/// ~256 / ~1024 / ~3000 (the static KV cap), with mimi decode running on its
/// own thread (downstream-only; the live-loop shape) and its worker-side
/// per-frame cost reported alongside. A long free-run on synthetic user
/// frames (SINE codes; timing is value-independent) walks the fill to the
/// cap; inside an 80-step window ending at each target every frame is
/// decomposed, then min-of-medians over 5 × 16-step rounds.
fn framebench(pile: &str, fmt: WeightFmt, depth_f16: bool) {
    use mary::models::personaplex::temporal_metal::MAX_SEQ;
    const WINDOW: usize = 80;
    const ROUND: usize = 16;
    const BUDGET_MS: f64 = 80.0;
    let targets = [256usize, 1024, MAX_SEQ - 1];
    let spike = [47.6f64, 48.9, 52.2];

    let t0 = Instant::now();
    let source = runtime_source(pile);
    let mut p = RealtimePipeline::load_auto(&source, fmt, depth_f16);
    // The depth arm is the thing this bench is usually run to compare, so it
    // goes in the header next to the numbers, not in the invocation only.
    println!(
        "framebench ({fmt:?} temporal + f16 head + {} + CPU mimi): ms per emitted frame",
        p.depth.label()
    );
    println!(
        "(desktop machine — ambient contention inflates; min-of-medians + raw best/worst reported)"
    );
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ── mimi decode scaling (stateless CPU decode of t frames): the 1-frame
    // row is the naive live-loop per-frame cost; the marginal Δtotal/Δframes
    // between rows is what a streaming (stateful) decoder's per-frame compute
    // would approach. ──
    let sil: [u32; mimi_cfg::NUM_CODEBOOKS] = std::array::from_fn(|q| cfg::SILENCE_TOKENS[q]);
    let _ = p.decoder.decode(&[sil]); // warm
    println!("\nmimi decode scaling (median of 3 batch decodes):");
    let mut prev: Option<(usize, f64)> = None;
    for t in [1usize, 5, 25, 50] {
        let frames = vec![sil; t];
        let mut ms = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = p.decoder.decode(&frames);
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let m = median(ms);
        let marginal = prev
            .map(|(pt, pm)| format!("  ({:.1} ms/frame marginal)", (m - pm) / (t - pt) as f64))
            .unwrap_or_default();
        println!(
            "  {t:3} frames  {m:8.1} ms total  {:6.1} ms/frame amortized{marginal}",
            m / t as f64
        );
        prev = Some((t, m));
    }

    // ── free-run to the KV cap; mimi decode OFF the LM critical path ──
    // Decode is downstream-only (frame t's PCM is not an input to any later
    // step), so a live loop runs it on its own thread. The worker decodes
    // EVERY emitted frame in order (sustained live-loop load, including the
    // CPU contention it costs the depformer pools); the frame column is the
    // LM critical path: prepare + embed + submit + drain/readback + the 16
    // depformer steps + commit + channel handoff.
    println!(
        "\nfree-run to fill {} (synthetic SINE user frames; decode on its own thread) …",
        MAX_SEQ - 1
    );
    let mut windows: Vec<Vec<FrameRow>> = targets.iter().map(|_| Vec::new()).collect();
    let mut window_idx: Vec<Vec<usize>> = targets.iter().map(|_| Vec::new()).collect();
    let mut last_win: Option<usize> = None;
    // Early-window page-in probe: on the zero-copy path the first frames
    // fault the mmap'd weight pages in on first kernel touch — record the
    // first 256 LM step times individually and compare the early window
    // against the settled one (the 80 ms budget question).
    let mut early: Vec<f64> = Vec::with_capacity(256);
    let run0 = Instant::now();
    // RT_FB_SKIP=<fill>: fast-forward the temporal fill cursor once the ring
    // is past the delay horizon (force_len is the bench-only fill-pinning
    // API; wgpu zero-fills untouched KV slots and timing is value-
    // independent) — lets a short quiet window catch the deep windows
    // without the full 3000-step walk. Windows below the target stay empty
    // and are skipped in the report.
    let skip: Option<usize> = std::env::var("RT_FB_SKIP")
        .ok()
        .and_then(|s| s.parse().ok());
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, [u32; mimi_cfg::NUM_CODEBOOKS])>(8);
    let decoder = &p.decoder;
    let mimi_times: Vec<(usize, f64)> = std::thread::scope(|sc| {
        let worker = sc.spawn(move || {
            let mut times = Vec::new();
            while let Ok((idx, frame)) = rx.recv() {
                let t0 = Instant::now();
                let _ = decoder.decode(&[frame]);
                times.push((idx, t0.elapsed().as_secs_f64() * 1e3));
            }
            times
        });
        let mut emitted = 0usize;
        loop {
            let fill = p.temporal.len();
            if fill >= MAX_SEQ {
                break;
            }
            if let Some(target) = skip {
                if fill > 4 && fill + WINDOW < target {
                    p.temporal.force_len(target + 1 - WINDOW);
                    continue;
                }
            }
            // measured window: the WINDOW steps ending exactly at each target
            let win = targets.iter().position(|&t| fill <= t && fill + WINDOW > t);
            if win != last_win {
                if win.is_some() {
                    let _ = p.depth.take_bench(); // reset the in-situ decomposition
                }
                if let Some(w) = last_win {
                    let (n, tot, cond, gemv, head, scalar) = p.depth.take_bench();
                    println!(
                        "  window @{}: depformer in situ ({n} frames) {tot:.1} ms/frame = cond {cond:.1} + gemv {gemv:.1} + head {head:.1} + scalar {scalar:.1}",
                        targets[w]
                    );
                }
                last_win = win;
            }

            let ts = Instant::now();
            let Some(pr) = p.stream.prepare(Some(&SINE), None, None) else {
                continue; // the offset-0 ring-seeding call — no model step
            };
            let x = p.temporal.embed_codes(&pr.input);
            p.temporal.step_submit(&x, p.head);
            let submit = ts.elapsed().as_secs_f64() * 1e3;
            let (hidden, logits) = p.temporal.read_hidden_logits();
            let temporal = ts.elapsed().as_secs_f64() * 1e3;
            let sampled = argmax(&logits) as i64;
            let next_text = if pr.provided[0] {
                pr.target[0]
            } else {
                sampled
            };
            let td = Instant::now();
            let dep = p.depth.frame(&hidden, next_text, &pr.forced(), None);
            let depth = td.elapsed().as_secs_f64() * 1e3;
            let out = p.stream.commit(&pr, sampled, &dep);
            if let Some(o) = &out {
                tx.send((emitted, agent_codes(o)))
                    .expect("decode worker alive");
                emitted += 1;
            }
            let total = ts.elapsed().as_secs_f64() * 1e3;
            // Early-window page-in probe: on the zero-copy path the first
            // steps fault the mmap'd weight pages in on first kernel touch —
            // record the first 256 LM step times and compare the early
            // window against the settled one (the 80 ms budget question).
            // Meaningless under RT_FB_SKIP (steps are skipped, not run).
            if early.len() < 256 {
                early.push(temporal + depth); // the LM cost, where page-in lands
            }
            if early.len() == 256 {
                let pct = |v: &mut Vec<f64>, p: f64| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[((v.len() as f64 - 1.0) * p) as usize]
                };
                let (mut a, mut b) = (early[..64].to_vec(), early[192..].to_vec());
                println!(
                    "  early-window page-in probe (LM step = temporal+depth, ms): step0 {:.1}; \
                     steps 0..64 p50 {:.1} / p95 {:.1}; steps 192..256 p50 {:.1} / p95 {:.1}",
                    early[0],
                    pct(&mut a, 0.5),
                    pct(&mut a, 0.95),
                    pct(&mut b, 0.5),
                    pct(&mut b, 0.95),
                );
                early.push(f64::NAN); // report once
            }
            if let Some(w) = win {
                assert!(out.is_some(), "window frame past the delay horizon");
                windows[w].push(FrameRow {
                    total,
                    temporal,
                    submit,
                    depth,
                    mimi: 0.0,
                });
                window_idx[w].push(emitted - 1);
            }
            if p.temporal.len().is_multiple_of(256) {
                eprintln!(
                    "  fill {:4}/{}  ({:.1} min elapsed)",
                    p.temporal.len(),
                    MAX_SEQ - 1,
                    run0.elapsed().as_secs_f64() / 60.0
                );
            }
        }
        drop(tx);
        worker.join().expect("decode worker")
    });
    // match the worker's per-frame decode times back into the windows
    let by_idx: std::collections::HashMap<usize, f64> = mimi_times.into_iter().collect();
    for (rows, idxs) in windows.iter_mut().zip(&window_idx) {
        for (row, i) in rows.iter_mut().zip(idxs) {
            row.mimi = by_idx.get(i).copied().unwrap_or(0.0);
        }
    }
    if let Some(w) = last_win {
        let (n, tot, cond, gemv, head, scalar) = p.depth.take_bench();
        println!(
            "  window @{}: depformer in situ ({n} frames) {tot:.1} ms/frame = cond {cond:.1} + gemv {gemv:.1} + head {head:.1} + scalar {scalar:.1}",
            targets[w]
        );
    }

    // ── report ──
    println!(
        "\nfull-frame latency, min-of-medians over {} rounds × {ROUND} steps (ms; components take their own best round, so the sum can differ from the frame column):",
        WINDOW / ROUND
    );
    println!("(frame = LM critical path; mimi decodes on its own thread — its column is the");
    println!(" worker-side per-frame cost, which must only stay under the frame budget)");
    println!(
        "{:>6}  {:>8}  {:>32}  {:>7}  {:>8}  {:>7}  {:>15}",
        "fill",
        "frame",
        "temporal (of which host submit)",
        "depth",
        "mimi(off)",
        "other",
        "raw best/worst"
    );
    let mut all_ok = true;
    for (w, &t) in targets.iter().enumerate() {
        let rows = &windows[w];
        if rows.is_empty() && skip.is_some() {
            continue; // fast-forwarded past this window (RT_FB_SKIP)
        }
        assert_eq!(rows.len(), WINDOW, "window @{t} incomplete");
        let frame = min_of_medians(rows, ROUND, |r| r.total);
        let temporal = min_of_medians(rows, ROUND, |r| r.temporal);
        let submit = min_of_medians(rows, ROUND, |r| r.submit);
        let depth = min_of_medians(rows, ROUND, |r| r.depth);
        let mimi = min_of_medians(rows, ROUND, |r| r.mimi);
        let other = min_of_medians(rows, ROUND, |r| r.total - r.temporal - r.depth);
        let best = rows.iter().map(|r| r.total).fold(f64::INFINITY, f64::min);
        let worst = rows.iter().map(|r| r.total).fold(0f64, f64::max);
        let ok = frame < BUDGET_MS;
        all_ok &= ok;
        println!(
            "{t:>6}  {:>2} {frame:>5.1}  {temporal:>22.1} ({submit:>4.1})  {depth:>7.1}  {mimi:>8.1}  {other:>7.1}  {best:>7.1}/{worst:<7.1}",
            if ok { "OK" } else { "XX" }
        );
    }
    // ── keep-up, EVERY frame counted ──
    // `min_of_medians` is a best-case estimator: it exists to see the
    // machine's own cost through ambient noise, which is the right lens for a
    // port's cost. The live loop does not get the best round — a frame that
    // misses 80 ms is a frame the caller's Ear skips — so the contention
    // question needs the unfiltered distribution instead.
    println!(
        "\nkeep-up over the same {WINDOW}-frame windows, every frame counted (no best-round filter):"
    );
    println!(
        "{:>6}  {:>8}  {:>8}  {:>8}  {:>13}  {:>9}",
        "fill", "mean", "p50", "p95", "over 80 ms", "realtime"
    );
    for (w, &t) in targets.iter().enumerate() {
        let rows = &windows[w];
        if rows.is_empty() {
            continue;
        }
        let mut v: Vec<f64> = rows.iter().map(|r| r.total).collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| v[((v.len() as f64 - 1.0) * q) as usize];
        let over = v.iter().filter(|&&x| x > BUDGET_MS).count();
        println!(
            "{t:>6}  {mean:>8.1}  {:>8.1}  {:>8.1}  {:>7}/{:<5}  {:>8.2}x",
            p(0.5),
            p(0.95),
            over,
            v.len(),
            BUDGET_MS / mean
        );
    }
    println!(
        "\nbudget 80 ms @ 12.5 Hz; spike projection {:.1}/{:.1}/{:.1} ms at ctx {}/{}/{}",
        spike[0], spike[1], spike[2], targets[0], targets[1], targets[2]
    );
    println!(
        "PERSONAPLEX RT FRAMEBENCH ({fmt:?}): {}",
        if all_ok {
            "inside the frame budget at all three fills"
        } else {
            "OVER BUDGET at ≥1 fill"
        }
    );
}

// ───────────────────────────── pipeline gate ─────────────────────────────

/// The oracle's user-side input WAV (same as `personaplex_probe pipeline`).
const INPUT_WAV: &str = "ref_voice.wav";
const OUT_WAV: &str = "/tmp/mary-personaplex/rt_pipeline_out.wav";
/// The CPU-f32 parity pipeline's output (`personaplex_probe pipeline`) — the
/// pre-divergence audio comparison target.
const PARITY_WAV: &str = "/tmp/mary-personaplex/pipeline_out.wav";
const AUDIO_GATE: f64 = 0.999;

/// One oracle prompt-flow step (what `capture_personaplex.py` fed).
enum Phase {
    Vp(usize),
    Silence,
    Text(i64),
    User(usize),
}

/// Snapshot of the first committed divergence: the logits row that flipped,
/// its golden counterpart, and the two tokens.
struct Divergence {
    step: usize,
    what: String,
    rt_row: Vec<f32>,
    gold_row: Vec<f32>,
    rt_tok: i64,
    gold_tok: i64,
}

#[allow(clippy::too_many_lines)]
fn pipeline_gate(pile: &str, fmt: WeightFmt, depth_f16: bool) {
    // ── goldens ──
    let (dep_logits, s) = golden_f32("dep_logits"); // [S, 16, 2048]
    assert_eq!(&s[1..], &[cfg::DEP_Q, cfg::CARD], "dep_logits shape");
    let steps = s[0];
    let (gdep_tokens, s) = golden_i64("dep_tokens");
    assert_eq!(s, vec![steps, cfg::DEP_Q], "dep_tokens shape");
    let (dep_in_text, s) = golden_i64("dep_in_text");
    assert_eq!(s, vec![steps], "dep_in_text shape");
    let (out_tokens, s) = golden_i64("out_tokens");
    assert_eq!(s[1], cfg::NUM_STREAMS, "out_tokens shape");
    let n_gen = s[0];
    let (user, s) = golden_i64("user_codes");
    assert_eq!(s, vec![n_gen, 8], "user_codes shape");
    let (text, _) = golden_i64("text_prompt_tokens");
    let (vp_cache, s) = golden_i64("vp_cache");
    assert_eq!(
        s,
        vec![1, cfg::NUM_STREAMS, mary::models::personaplex::lmgen::CT]
    );
    let (vp, vps) = golden_f32("vp_embeddings"); // [50, 1, 1, 4096]
    assert_eq!(&vps[1..], &[1, 1, cfg::DIM], "vp_embeddings shape");
    let n_vp = vps[0];
    let (gl, s) = golden_f32("tt_text_logits"); // [113, 32000]
    assert_eq!(s, vec![steps, cfg::TEXT_LOGITS], "tt_text_logits shape");
    let (step_toks, s) = golden_i64("step_tokens"); // [63, 17]
    assert_eq!(s, vec![steps - n_vp, cfg::NUM_STREAMS], "step_tokens shape");
    let (gaudio, s) = golden_f32("out_audio");
    assert_eq!(
        s,
        vec![n_gen * mimi_cfg::SAMPLES_PER_FRAME],
        "out_audio shape"
    );

    let n_silence = 6; // int(0.5 s × 12.5 Hz), meta.json phases
    assert_eq!(
        n_vp + 2 * n_silence + text.len() + n_gen,
        steps,
        "phase counts"
    );
    let gen_start = steps - n_gen;
    let mut sched: Vec<Phase> = Vec::with_capacity(steps);
    sched.extend((0..n_vp).map(Phase::Vp));
    sched.extend((0..n_silence).map(|_| Phase::Silence));
    sched.extend(text.iter().map(|&t| Phase::Text(t)));
    sched.extend((0..n_silence).map(|_| Phase::Silence));
    sched.extend((0..n_gen).map(Phase::User));
    println!(
        "goldens: {steps} steps = vp {n_vp} + silence {n_silence} + text {} + silence {n_silence} + user {n_gen}",
        text.len()
    );

    // ── input audio ──
    let (mut samples, sr) = wav::read_pcm16_mono(Path::new(INPUT_WAV));
    assert_eq!(sr, mimi_cfg::SAMPLE_RATE, "input wav sample rate");
    let n_samples = n_gen * mimi_cfg::SAMPLES_PER_FRAME;
    assert!(
        samples.len() >= n_samples,
        "input wav shorter than the oracle window"
    );
    samples.truncate(n_samples);
    println!("input: {INPUT_WAV} ({n_samples} samples = {n_gen} frames)");

    println!(
        "loading realtime pipeline ({fmt:?} temporal + f16 head + {} depformer + CPU mimi) from {pile} …",
        if depth_f16 { "f16" } else { "f32" }
    );
    let t0 = Instant::now();
    let source = runtime_source(pile);
    // Pinned to the CPU depth arm even though the pipeline's default is now
    // the GPU one: this gate's per-format bars were calibrated against the
    // host predictor's numerics, and re-anchoring them on the GPU arm needs
    // the oracle goldens this gate reads, which are session scratch and are
    // not on disk. Gating the shipping arm here is the follow-up, not a
    // silent re-baseline.
    let mut p = RealtimePipeline::load_with_depth(
        &source,
        fmt,
        mary::models::personaplex::pipeline::DepthChoice::Cpu { f16: depth_f16 },
    );
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ── 1. Mimi encode (CPU stage, unchanged from the parity pipeline):
    // integer-exact vs the oracle's streaming encode ──
    let t0 = Instant::now();
    let codes = p.encoder.encode(&samples);
    let enc_secs = t0.elapsed().as_secs_f64();
    assert_eq!(codes.len(), n_gen, "encoded frame count");
    let mut enc_mism = 0usize;
    for (t, frame) in codes.iter().enumerate() {
        for (q, &c) in frame.iter().enumerate() {
            enc_mism += (c as i64 != user[t * 8 + q]) as usize;
        }
    }
    let ok_enc = enc_mism == 0;
    println!(
        "  {} mimi encode   {}/{} codes exact (user_codes)  [{enc_secs:.1}s]",
        if ok_enc { "OK" } else { "XX" },
        n_gen * 8 - enc_mism,
        n_gen * 8
    );

    // ── 2. free-run the whole golden flow through the fast stages ──
    // Divergence bookkeeping: with a quantized temporal stack the committed
    // token stream WILL diverge from the f32 oracle at some gen-phase step
    // (prompt phases teacher-force every stream, so the trajectory cannot
    // diverge before gen_start — only the KV cache carries quantization
    // noise). "Committed" = a sampled token that enters the ring: sampled
    // text (stream 0) and the agent dep tokens (cb 0..8) during gen.
    let mut div: Option<Divergence> = None;
    let mut input_hits = 0usize; // vs step_tokens (token-fed steps)
    let mut next_text_hits = 0usize; // vs dep_in_text (all steps)
    let mut dep_frame_hits = 0usize; // all 16 dep tokens vs dep_tokens
    let mut shadow_text_hits = 0usize; // sampled-text argmax vs golden argmax
    let (mut pre_lmin, mut pre_lsum, mut pre_n) = (1f64, 0f64, 0usize); // text-logits cos, shared prefix
    let mut out_hits = 0usize; // exact 17-stream out frames
    let (mut out_text_hits, mut out_agent_hits, mut out_user_hits) = (0usize, 0usize, 0usize);
    let mut r_div: Option<usize> = None; // first divergent out frame
    let mut agent: Vec<[u32; mimi_cfg::NUM_CODEBOOKS]> = Vec::with_capacity(n_gen);
    let mut gen_ms: Vec<f64> = Vec::with_capacity(n_gen);
    let t0 = Instant::now();
    for s in 0..steps {
        let ts = Instant::now();
        let trace = match &sched[s] {
            Phase::Vp(i) => p.step_embedding(&vp[i * cfg::DIM..(i + 1) * cfg::DIM]),
            Phase::Silence => p.step(
                Some(&SINE),
                Some(&SILENCE),
                Some(cfg::TEXT_PAD_TOKEN as i64),
            ),
            Phase::Text(t) => p.step(Some(&SINE), Some(&SILENCE), Some(*t)),
            Phase::User(r) => {
                let cf: [i64; 8] = codes[*r].map(|c| c as i64);
                p.step(Some(&cf), None, None)
            }
        };
        if s >= gen_start {
            gen_ms.push(ts.elapsed().as_secs_f64() * 1e3);
        }

        // text-logits cos vs the golden (valid as an oracle comparison only
        // while the trajectory is still on the golden prefix)
        let glrow = &gl[s * cfg::TEXT_LOGITS..(s + 1) * cfg::TEXT_LOGITS];
        let (lcos, _) = cos_maxd(&trace.text_logits, glrow);
        if div.is_none() {
            pre_lmin = pre_lmin.min(lcos);
            pre_lsum += lcos;
            pre_n += 1;
        }
        shadow_text_hits += (argmax(&trace.text_logits) == argmax(glrow)) as usize;
        next_text_hits += (trace.next_text == dep_in_text[s]) as usize;
        dep_frame_hits +=
            (trace.dep_tokens[..] == gdep_tokens[s * cfg::DEP_Q..(s + 1) * cfg::DEP_Q]) as usize;
        if s >= n_vp {
            let r = s - n_vp;
            let grow = &step_toks[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
            input_hits += (trace.input.as_ref().map(|i| &i[..]) == Some(grow)) as usize;
        }

        // committed-divergence detection (gen phase only; snapshot the
        // flipped logits row for the near-tie analysis)
        if s >= gen_start && div.is_none() {
            if trace.next_text != dep_in_text[s] {
                div = Some(Divergence {
                    step: s,
                    what: "sampled text (stream 0)".into(),
                    rt_row: trace.text_logits.clone(),
                    gold_row: glrow.to_vec(),
                    rt_tok: trace.next_text,
                    gold_tok: dep_in_text[s],
                });
            } else {
                for cb in 0..8 {
                    let gt = gdep_tokens[s * cfg::DEP_Q + cb];
                    if trace.dep_tokens[cb] != gt {
                        let gr = &dep_logits[(s * cfg::DEP_Q + cb) * cfg::CARD
                            ..(s * cfg::DEP_Q + cb + 1) * cfg::CARD];
                        div = Some(Divergence {
                            step: s,
                            what: format!("agent dep token cb {cb} (stream {})", cb + 1),
                            rt_row: p.depth.logits()[cb * cfg::CARD..(cb + 1) * cfg::CARD].to_vec(),
                            gold_row: gr.to_vec(),
                            rt_tok: trace.dep_tokens[cb],
                            gold_tok: gt,
                        });
                        break;
                    }
                }
            }
        }

        // out-frame agreement + agent codes for the decoder
        if s >= gen_start {
            let r = s - gen_start;
            let out = trace.out.expect("past the delay horizon in the gen phase");
            let grow = &out_tokens[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
            let exact = out[..] == *grow;
            out_hits += exact as usize;
            if !exact && r_div.is_none() {
                r_div = Some(r);
            }
            out_text_hits += (out[0] == grow[0]) as usize;
            out_agent_hits += (1..9).filter(|&k| out[k] == grow[k]).count();
            out_user_hits += (9..17).filter(|&k| out[k] == grow[k]).count();
            agent.push(agent_codes(&out));
        }

        if s + 1 == n_vp {
            p.stream.overwrite(&vp_cache); // oracle: cache.copy_(voice_prompt_cache)
        }
        if (s + 1) % 16 == 0 || s + 1 == steps {
            eprintln!(
                "  step {:3}/{steps}  logits cos={lcos:.6}  ({:.0} ms/step)",
                s + 1,
                t0.elapsed().as_secs_f64() * 1e3 / (s + 1) as f64
            );
        }
    }
    let run_secs = t0.elapsed().as_secs_f64();

    // ── 3. report: agreement, divergence, near-tie analysis ──
    let n_tok = steps - n_vp;
    println!("free-run agreement vs the f32-oracle goldens:");
    println!(
        "  -- text logits (shared prefix, {pre_n} steps)  min cos={pre_lmin:.6}  mean={:.6}",
        pre_lsum / pre_n as f64
    );
    println!(
        "  -- sampled-text argmax {shadow_text_hits}/{steps} ({:.1}%)   next_text {next_text_hits}/{steps}   dep frames (16 cb) {dep_frame_hits}/{steps}",
        100.0 * shadow_text_hits as f64 / steps as f64
    );
    println!("  -- model inputs vs step_tokens  {input_hits}/{n_tok} frames exact");
    println!(
        "  -- out frames vs out_tokens     {out_hits}/{n_gen} exact  (per-stream: text {out_text_hits}/{n_gen}, agent {out_agent_hits}/{}, user {out_user_hits}/{})",
        n_gen * 8,
        n_gen * 8
    );
    let ok_user = out_user_hits == n_gen * 8;
    println!(
        "  {} user streams 9..=16 exact (provided codes round-trip the ring)",
        if ok_user { "OK" } else { "XX" }
    );

    let (_, bar_l, _, _) = bars(fmt);
    let mut ok_div = true;
    match &div {
        None => println!(
            "  -- no committed divergence within the {n_gen}-frame window (first divergence: none)"
        ),
        Some(d) => {
            let (dcos, _) = cos_maxd(&d.rt_row, &d.gold_row);
            let rt_margin = d.rt_row[d.rt_tok as usize] - d.rt_row[d.gold_tok as usize];
            let gold_margin = d.gold_row[d.gold_tok as usize] - d.gold_row[d.rt_tok as usize];
            let gold_rank = d
                .rt_row
                .iter()
                .filter(|&&v| v > d.rt_row[d.gold_tok as usize])
                .count();
            let scale = {
                let mut top = f32::NEG_INFINITY;
                let mut second = f32::NEG_INFINITY;
                for &v in &d.gold_row {
                    if v > top {
                        second = top;
                        top = v;
                    } else if v > second {
                        second = v;
                    }
                }
                (top - second) as f64
            };
            ok_div = dcos >= bar_l;
            println!(
                "  first committed divergence: step {} ({}), {} frames pre-divergence",
                d.step,
                d.what,
                r_div.unwrap_or(n_gen)
            );
            println!(
                "  {} logits at divergence  cos={dcos:.6} vs golden (bar {bar_l} — flip must be a near-tie, not garbage)",
                if ok_div { "OK" } else { "XX" }
            );
            println!(
                "     rt picked {} over golden {} by {rt_margin:.4} logits; golden preferred its token by {gold_margin:.4} (golden top1-top2 gap {scale:.4}); golden token ranks #{} in rt logits",
                d.rt_tok,
                d.gold_tok,
                gold_rank + 1
            );
        }
    }

    // ── 4. Mimi decode → WAV + audio comparison over the pre-divergence
    // prefix (Mimi decode is causal, so the shared token prefix decodes to a
    // comparable sample prefix) ──
    let t0 = Instant::now();
    let pcm = p.decode(&agent);
    let dec_secs = t0.elapsed().as_secs_f64();
    assert_eq!(pcm.len(), gaudio.len(), "decoded sample count");
    wav::write_pcm16_mono(Path::new(OUT_WAV), &pcm, mimi_cfg::SAMPLE_RATE);
    println!(
        "wrote {OUT_WAV} ({} samples, decode {dec_secs:.1}s)",
        pcm.len()
    );

    let pre_frames = r_div.unwrap_or(n_gen);
    let pre = pre_frames * mimi_cfg::SAMPLES_PER_FRAME;
    let mut ok_audio = true;
    if pre == 0 {
        println!(
            "  -- audio prefix empty (divergence at out frame 0) — no pre-divergence audio to compare"
        );
    } else {
        let (cos_g, maxd_g) = cos_maxd(&pcm[..pre], &gaudio[..pre]);
        ok_audio = cos_g >= AUDIO_GATE;
        println!(
            "  {} audio vs oracle (streaming decode), {pre_frames}-frame prefix  cos={cos_g:.9}  max|Δ|={maxd_g:.3e}",
            if ok_audio { "OK" } else { "XX" }
        );
        if Path::new(PARITY_WAV).exists() {
            let (ppcm, psr) = wav::read_pcm16_mono(Path::new(PARITY_WAV));
            assert_eq!(psr, mimi_cfg::SAMPLE_RATE);
            assert_eq!(ppcm.len(), gaudio.len(), "parity wav sample count");
            let (cos_p, maxd_p) = cos_maxd(&pcm[..pre], &ppcm[..pre]);
            println!(
                "  -- audio vs parity pipeline wav (pcm16), same prefix       cos={cos_p:.9}  max|Δ|={maxd_p:.3e}"
            );
        } else {
            println!(
                "  -- {PARITY_WAV} not found (run `personaplex_probe pipeline` first) — skipped"
            );
        }
    }

    // ── timing ──
    gen_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (frames, d_total, d_cond, d_gemv, d_head, d_scalar) = p.depth.take_bench();
    println!(
        "timing: gen-phase median {:.1} ms/step (min {:.1}, max {:.1}; fill ≈ {} — small-context regime, desktop contention applies)",
        gen_ms[gen_ms.len() / 2],
        gen_ms.first().unwrap(),
        gen_ms.last().unwrap(),
        steps
    );
    println!(
        "        depformer (over {frames} frames): {d_total:.1} ms/frame = cond {d_cond:.1} + stack gemv {d_gemv:.1} + head {d_head:.1} + scalar {d_scalar:.1}"
    );
    println!(
        "        mimi encode {enc_secs:.1}s / decode {dec_secs:.1}s for {n_gen} frames (batch; ~{:.0} ms/frame sequential cost in a live loop)",
        dec_secs * 1e3 / n_gen as f64
    );
    println!(
        "ran {steps} steps in {run_secs:.1}s ({:.0} ms/step)",
        run_secs * 1e3 / steps as f64
    );

    // Exercise the public generation-only seam after real user frames. The
    // pure StreamCache test separately gates the corresponding depformer
    // forcing view; the pipeline trace proves those prepared tokens are the
    // ones handed to the temporal embedding path.
    let output_only = p.step_output_only(None, None);
    let ok_output_only = output_only
        .input
        .is_some_and(|input| input[9..17].iter().all(|&token| token == -1));
    println!(
        "  {} output-only trace has user streams 9..=16 set to zero-token -1",
        if ok_output_only { "OK" } else { "XX" }
    );

    // ── verdict ──
    // Exactness expectations are per-format: f16 must reproduce the oracle
    // token stream (wiring ablation); q8/q4 are gated on honesty-calibrated
    // tripwires (divergence is expected, garbage is not).
    let ok_fmt = match fmt {
        WeightFmt::F16 => out_hits == n_gen && div.is_none(),
        _ => ok_div,
    };
    if fmt == WeightFmt::F16 && !ok_fmt {
        println!("  XX f16 stack must be token-exact over the window (wiring ablation)");
    }
    println!(
        "NOTE: {fmt:?} weights are {} — see the module docs; agreement numbers above are the claim, nothing more.",
        if fmt == WeightFmt::F16 {
            "near-exact (ablation format)"
        } else {
            "a REAL numerics change (free-run divergence expected)"
        }
    );
    if ok_enc && ok_user && ok_fmt && ok_audio && ok_output_only {
        println!("PERSONAPLEX RT PIPELINE ({fmt:?}): PASS");
    } else {
        println!("PERSONAPLEX RT PIPELINE ({fmt:?}): FAIL");
        std::process::exit(1);
    }
}

/// The full golden prompt schedule + gen phase, reusable by the reset gate.
/// Returns `(sched, vp_embeddings, vp_cache, n_vp, codes)`.
#[cfg(feature = "q4")]
fn reset_schedule() -> (
    Vec<Phase>,
    Vec<f32>,
    Vec<i64>,
    usize,
    Vec<[u32; mimi_cfg::NUM_CODEBOOKS]>,
) {
    let (gl, s) = golden_f32("tt_text_logits");
    let steps = s[0];
    let (text, _) = golden_i64("text_prompt_tokens");
    let (vp_cache, _) = golden_i64("vp_cache");
    let (vp, vps) = golden_f32("vp_embeddings");
    let n_vp = vps[0];
    let (user, us) = golden_i64("user_codes");
    let n_gen = us[0];
    let _ = gl;
    let n_silence = 6;
    let mut sched: Vec<Phase> = Vec::with_capacity(steps);
    sched.extend((0..n_vp).map(Phase::Vp));
    sched.extend((0..n_silence).map(|_| Phase::Silence));
    sched.extend(text.iter().map(|&t| Phase::Text(t)));
    sched.extend((0..n_silence).map(|_| Phase::Silence));
    sched.extend((0..n_gen).map(Phase::User));
    // user codes as Mimi frames (streams fed on the User phase).
    let codes: Vec<[u32; mimi_cfg::NUM_CODEBOOKS]> = (0..n_gen)
        .map(|r| std::array::from_fn(|q| user[r * 8 + q] as u32))
        .collect();
    (sched, vp, vp_cache, n_vp, codes)
}

/// Drive the whole golden schedule through `p`, collecting the committed token
/// trajectory: per step the `next_text` and 16 dep tokens, and per gen frame
/// the full 17-stream out frame. Deterministic — greedy or a fixed seed.
#[cfg(feature = "q4")]
fn reset_run(
    p: &mut RealtimePipeline,
    sched: &[Phase],
    vp: &[f32],
    vp_cache: &[i64],
    n_vp: usize,
    codes: &[[u32; mimi_cfg::NUM_CODEBOOKS]],
) -> (Vec<i64>, Vec<[i64; cfg::NUM_STREAMS]>) {
    let mut toks: Vec<i64> = Vec::new();
    let mut outs: Vec<[i64; cfg::NUM_STREAMS]> = Vec::new();
    let mut r = 0usize;
    for (s, phase) in sched.iter().enumerate() {
        let trace = match phase {
            Phase::Vp(i) => p.step_embedding(&vp[i * cfg::DIM..(i + 1) * cfg::DIM]),
            Phase::Silence => p.step(
                Some(&SINE),
                Some(&SILENCE),
                Some(cfg::TEXT_PAD_TOKEN as i64),
            ),
            Phase::Text(t) => p.step(Some(&SINE), Some(&SILENCE), Some(*t)),
            Phase::User(_) => {
                let cf: [i64; 8] = codes[r].map(|c| c as i64);
                r += 1;
                p.step(Some(&cf), None, None)
            }
        };
        toks.push(trace.next_text);
        toks.extend_from_slice(&trace.dep_tokens);
        if let Some(o) = trace.out {
            outs.push(o);
        }
        if s + 1 == n_vp {
            p.stream.overwrite(vp_cache);
        }
    }
    (toks, outs)
}

/// `reset_session == reload` — token-exact. Run the golden flow, reset, run it
/// again → identical trajectory; and a freshly `load`ed pipeline → identical
/// too. Proves a new conversation can start without reloading weights (both
/// greedy and a fixed sampling seed).
#[cfg(feature = "q4")]
fn reset_gate(pile: &str, fmt: WeightFmt, depth_f16: bool) {
    use mary::models::personaplex::sampling::SamplingConfig;
    let (sched, vp, vp_cache, n_vp, codes) = reset_schedule();
    println!("loading realtime pipeline ({fmt:?}) from {pile} …");
    let source = runtime_source(pile);
    let mut p = RealtimePipeline::load_auto(&source, fmt, depth_f16);

    let mut ok = true;
    // Two modes: greedy (parity) and seeded sampling (the quality path).
    for mode in ["greedy", "sampling"] {
        let apply = |p: &mut RealtimePipeline| {
            if mode == "sampling" {
                p.set_sampling(
                    SamplingConfig {
                        temp: 0.8,
                        top_k: 64,
                        top_p: 0.95,
                    },
                    0xC0FFEE,
                );
            } else {
                p.set_greedy();
            }
        };

        // Run 1 on the loaded pipeline — reset first: the previous mode's
        // runs leave the session dirty (temporal offset ≈ the schedule
        // length), so without this the sampling arm's run 1 attends over the
        // greedy arm's KV prefix and emits one extra out frame — the arm
        // could never pass (latent gate bug from a9501cc; the pile-requiring
        // gate had not been run to green before). A no-op for the first mode
        // on the fresh load.
        p.reset_session();
        apply(&mut p);
        let (t1, o1) = reset_run(&mut p, &sched, &vp, &vp_cache, n_vp, &codes);

        // Reset in place, run again — must be token-exact.
        p.reset_session();
        apply(&mut p); // re-seed the sampler so its RNG restarts (matches run 1)
        let (t2, o2) = reset_run(&mut p, &sched, &vp, &vp_cache, n_vp, &codes);
        let reset_ok = t1 == t2 && o1 == o2;
        println!(
            "  {} [{mode}] reset-then-run == first run  ({} step-tokens, {} out frames)",
            if reset_ok { "OK" } else { "XX" },
            t1.len(),
            o1.len()
        );

        // Fresh reload — reset must equal reload.
        let mut q = RealtimePipeline::load_auto(&source, fmt, depth_f16);
        apply(&mut q);
        let (t3, o3) = reset_run(&mut q, &sched, &vp, &vp_cache, n_vp, &codes);
        let reload_ok = t1 == t3 && o1 == o3;
        println!(
            "  {} [{mode}] reset-then-run == fresh reload",
            if reload_ok { "OK" } else { "XX" }
        );
        ok = ok && reset_ok && reload_ok;
    }

    if ok {
        println!("PERSONAPLEX RT RESET ({fmt:?}): PASS");
    } else {
        println!("PERSONAPLEX RT RESET ({fmt:?}): FAIL");
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sub = args.get(1).map(String::as_str).unwrap_or("");
    let mut fmt = WeightFmt::Q4;
    let mut depth_f16 = false;
    let mut pile: Option<String> = None;
    for a in &args[2.min(args.len())..] {
        if a == "--depth-f16" {
            depth_f16 = true;
            continue;
        }
        match WeightFmt::parse(a) {
            Some(f) => fmt = f,
            None => pile = Some(a.clone()),
        }
    }
    let pile = mary::paths::model(pile.as_deref(), "personaplex.pile")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        })
        .to_string_lossy()
        .into_owned();
    match sub {
        "gate" => gate(&pile, fmt),
        "bench" => bench(&pile, fmt),
        "quantcheck" => quantcheck(&pile),
        "pipeline" => pipeline_gate(&pile, fmt, depth_f16),
        "framebench" => framebench(&pile, fmt, depth_f16),
        "depthab" => depth_ab(&pile, fmt),
        "reset" => reset_gate(&pile, fmt, depth_f16),
        _ => {
            eprintln!(
                "usage: personaplex_rt_probe <gate|bench|quantcheck|pipeline|framebench|depthab|reset> [q4|q8|f16] [--depth-f16] [pile-path]"
            );
            eprintln!(
                "  gate        113-step golden stream: cos + argmax vs tt_text_logits (per-format bars)"
            );
            eprintln!("  bench       ms/step at cache fill 256/1024/3000, f16 vs q4 logit head");
            eprintln!(
                "  framebench  ms per EMITTED FRAME on the LM critical path (temporal + depformer +"
            );
            eprintln!(
                "              overhead; mimi decode on its own thread, worker cost reported) at"
            );
            eprintln!("              cache fill 256/1024/2999 via a long synthetic free-run");
            eprintln!(
                "  depthab     PAIRED depth-arm A/B: depth_fast and depth_gpu both resident, both run"
            );
            eprintln!(
                "              on the SAME hidden state every frame (order alternated), so the two"
            );
            eprintln!(
                "              numbers saw one machine state. RT_AB_FMT picks the GPU DepthFmt (q8)"
            );
            eprintln!("  quantcheck  per-matvec q4 error, raw vs norm-alpha-folded weights");
            eprintln!(
                "  pipeline    assembled realtime pipeline free-run: WAV → encode → LM → decode → WAV,"
            );
            eprintln!(
                "              agreement % + first divergence + near-tie check + prefix audio cos"
            );
            eprintln!(
                "  reset       reset_session == reload, token-exact (greedy + seeded sampling)"
            );
            std::process::exit(2);
        }
    }
}
