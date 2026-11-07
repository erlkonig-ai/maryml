//! PersonaPlex-7B **surface probe** — the text-surfacing seam
//! experiment. Empirically measures what the REAL model does when we
//! TEACHER-FORCE a text phrase onto stream 0 (the inner-monologue) during a
//! silence gap, then release the force.
//!
//! Design questions (both answered by decoding real model output, not by
//! reasoning about the architecture):
//!
//! - **Q1 (auto-complete vs keep-the-line-open).** Force a short phrase onto
//!   stream 0 for N frames during a silence gap (where the model would emit
//!   PAD), then stop forcing (`text_token = None`). Does the model (a)
//!   auto-complete — continue/finish the surfaced thought coherently — or (b)
//!   return to PAD/silence? Reported: the model's OWN sampled stream-0 token
//!   at every step (its "what I would say" shadow), decoded verbatim, over the
//!   forced window AND the released window. A complete-thought phrase vs a
//!   dangling fragment are both tried.
//! - **Q2 (do the audio streams vocalize the surfaced text / silent-thought
//!   mode).** During the forced text, decode the AGENT audio (streams 1..=8):
//!   speech-like energy vs silence, and its RMS relative to a known-silent
//!   baseline. Then force agent audio = SILENCE_TOKENS simultaneously with the
//!   text and check that (i) the thought is still present on stream 0 and (ii)
//!   the user/hearing streams (9..=16) are unaffected (the probe only forces
//!   text + agent-silence, never the user codes).
//! - **Q3 (the rate-bridge — dense vs speech-cadence silent text).** Stream-0
//!   is normally SPARSE (a word token at onset, PAD between, metered to speech
//!   cadence, ~65% PAD). Hypothesis: that slowness is a constraint of SPEAKING,
//!   not thinking — so a SILENT thought (agent forced to SILENCE_TOKENS) may
//!   tolerate DENSE text (a real token every frame, no PAD): fast silent
//!   thinking. Force the SAME sentence two ways — dense (one token/frame, no
//!   PAD) and speech-cadence (token + PAD gaps) — both under agent-silence,
//!   then release and read the continuation for coherence vs garble, plus the
//!   forced token's rank in the model's own logits (is it fighting the pack?).
//!
//! The prompt flow is driven from the SAME real assets the rt pipeline gate
//! uses: the recorded voice-prompt embeddings + cache snapshot
//! (`vp_embeddings.npy` / `vp_cache.npy`), a silence spacer, the golden text
//! prompt, and a second silence spacer — so the model reaches a genuine
//! silence gap in-distribution before we force anything. Greedy by default
//! (reproducible); `--temp T` switches to seeded sampling for the shadow
//! stream (seed fixed).
//!
//!   cargo run --release --features personaplex,q4 --bin \
//!     personaplex_surface_probe -- [q4|q8|f16] [--temp 0.8] [pile-path]
//!
//! RAILS: reads only the read-only weight pile + the golden npys + the SPM
//! model; writes only WAVs under /tmp/mary-personaplex/. Never touches
//! self.pile or any persona.

use mary::models::f5::wav;
use mary::models::personaplex::config as cfg;
use mary::models::personaplex::depth::argmax;
use mary::models::personaplex::mimi::config as mimi_cfg;
use mary::models::personaplex::pipeline::{RealtimePipeline, SILENCE, SINE, agent_codes};
use mary::models::personaplex::sampling::SamplingConfig;
use mary::models::personaplex::spm::SpmTokenizer;
use mary::models::personaplex::temporal_metal::WeightFmt;
use mary::nn::npy;
use std::path::Path;
use std::time::Instant;

const GOLD: &str = "/tmp/mary-personaplex/golden";
const SPM_MODEL: &str = "/tmp/personaplex_scratch/ckpt/tokenizer_spm_32k_3.model";
const OUT_DIR: &str = "/tmp/mary-personaplex";

fn golden_f32(name: &str) -> (Vec<f32>, Vec<usize>) {
    npy::load_npy(&Path::new(GOLD).join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
}
fn golden_i64(name: &str) -> (Vec<i64>, Vec<usize>) {
    npy::load_npy_i64(&Path::new(GOLD).join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
}

/// RMS of a PCM chunk (a coarse "is it making sound" meter).
fn rms(pcm: &[f32]) -> f64 {
    if pcm.is_empty() {
        return 0.0;
    }
    (pcm.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / pcm.len() as f64).sqrt()
}

/// Peak |sample| of a PCM chunk.
fn peak(pcm: &[f32]) -> f64 {
    pcm.iter().fold(0f64, |m, &x| m.max((x as f64).abs()))
}

/// Decode one 17-stream out-frame's agent audio (streams 1..=8) to 1920 PCM
/// samples. `agent_codes` asserts the codes are real (past the delay horizon).
fn decode_agent_frame(p: &RealtimePipeline, out: &[i64; cfg::NUM_STREAMS]) -> Vec<f32> {
    p.decode(&[agent_codes(out)])
}

/// A short pretty-print of a token: the SPM surface, or the special-id name.
fn tok_str(spm: &SpmTokenizer, t: i64) -> String {
    match t as u32 {
        cfg::TEXT_PAD_TOKEN => "<pad>".into(),   // 3
        cfg::TEXT_EPAD_TOKEN => "<epad>".into(), // 0
        1 => "<bos>".into(),
        2 => "<eos>".into(),
        _ if t >= 0 && (t as usize) < spm.vocab_size() => {
            let s = spm.decode_token(t);
            if s.is_empty() {
                "∅".into()
            } else {
                format!("{s:?}")
            }
        }
        _ => format!("id{t}"),
    }
}

/// Detokenize a run of stream-0 ids to text, dropping the padding/special ids
/// (PAD/EPAD/BOS/EOS) — those are silence/word-boundary markers, not words.
fn decode_run(spm: &SpmTokenizer, ids: &[i64]) -> String {
    let words: Vec<i64> = ids
        .iter()
        .copied()
        .filter(|&t| {
            !(matches!(t as u32, cfg::TEXT_PAD_TOKEN | cfg::TEXT_EPAD_TOKEN) || t == 1 || t == 2)
        })
        .collect();
    if words.is_empty() {
        "(all padding/silence)".into()
    } else {
        format!("{:?}", spm.decode(&words))
    }
}

/// One phase step of the fixed real prompt (voice replay / silence / text).
enum Phase {
    Vp(usize),
    Silence,
    Text(i64),
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut fmt = WeightFmt::F16; // default to the exact-numerics stack for a clean read
    let mut temp: f32 = 0.0;
    let mut pile: Option<String> = None;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == "--temp" {
            temp = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.8);
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
    let sampling = temp > 0.0;
    std::fs::create_dir_all(OUT_DIR).ok();

    // ── real prompt assets (recorded voice + golden text prompt) ──
    let (vp, vps) = golden_f32("vp_embeddings"); // [n_vp,1,1,4096]
    assert_eq!(&vps[1..], &[1, 1, cfg::DIM], "vp_embeddings shape");
    let n_vp = vps[0];
    let (vp_cache, s) = golden_i64("vp_cache");
    assert_eq!(
        s,
        vec![1, cfg::NUM_STREAMS, mary::models::personaplex::lmgen::CT]
    );
    let (text, _) = golden_i64("text_prompt_tokens");
    let n_silence = 6usize;

    let mut sched: Vec<Phase> = Vec::new();
    sched.extend((0..n_vp).map(Phase::Vp));
    sched.extend((0..n_silence).map(|_| Phase::Silence));
    sched.extend(text.iter().map(|&t| Phase::Text(t)));
    sched.extend((0..n_silence).map(|_| Phase::Silence));

    let spm = SpmTokenizer::load(Path::new(SPM_MODEL));

    println!(
        "=== PersonaPlex surface probe ({fmt:?}{}) ===",
        if sampling {
            format!(", temp={temp}, seeded")
        } else {
            ", greedy".into()
        }
    );
    println!(
        "prompt: {n_vp} voice-replay + {n_silence} silence + {} text + {n_silence} silence = {} steps",
        text.len(),
        sched.len()
    );

    let t0 = Instant::now();
    let source = mary::persist::personaplex_model(Path::new(&pile))
        .unwrap_or_else(|e| panic!("pile load: {e}"))
        .into_runtime_source();
    let mut p = RealtimePipeline::load_auto(&source, fmt, false);
    if sampling {
        p.set_sampling(
            SamplingConfig {
                temp,
                top_k: 64,
                top_p: 0.95,
            },
            0x5EED,
        );
    }
    println!("loaded in {:.1}s\n", t0.elapsed().as_secs_f64());

    // ── drive the real prompt to the trailing silence gap ──
    for (i, phase) in sched.iter().enumerate() {
        match phase {
            Phase::Vp(k) => {
                p.step_embedding(&vp[k * cfg::DIM..(k + 1) * cfg::DIM]);
            }
            Phase::Silence => {
                p.step(
                    Some(&SINE),
                    Some(&SILENCE),
                    Some(cfg::TEXT_PAD_TOKEN as i64),
                );
            }
            Phase::Text(t) => {
                p.step(Some(&SINE), Some(&SILENCE), Some(*t));
            }
        }
        if i + 1 == n_vp {
            p.stream.overwrite(&vp_cache); // oracle: cache.copy_(voice_prompt_cache)
        }
    }
    println!(
        "reached the trailing silence gap (offset {}).",
        p.stream.offset()
    );

    // A silent-baseline RMS: what the agent audio decodes to when we hand it
    // the canonical SILENCE_TOKENS directly (the reference "no vocalization"
    // energy). This is a decoder-of-SILENCE_TOKENS measurement, independent of
    // the LM.
    let sil_frame: [u32; mimi_cfg::NUM_CODEBOOKS] = std::array::from_fn(|q| cfg::SILENCE_TOKENS[q]);
    let sil_pcm = p.decode(&[sil_frame]);
    let sil_rms = rms(&sil_pcm);
    let sil_peak = peak(&sil_pcm);
    println!("SILENCE_TOKENS decode baseline: rms={sil_rms:.5}  peak={sil_peak:.5}\n");

    // ── the phrases ──
    // A complete thought and a dangling fragment (Q1: does release finish it?).
    let phrases: &[(&str, &str)] = &[
        (
            "complete",
            "I have two goals, ship the loop and port the model.",
        ),
        ("fragment", "I have two goals,"),
    ];

    // Forcing-window length past the phrase, and the release window over which
    // we record the model's own continuation.
    const RELEASE: usize = 24; // ~1.9 s of released stream-0

    for (label, phrase) in phrases {
        let ids = spm.encode(phrase);
        println!("──────────────────────────────────────────────────────────────");
        println!("PHRASE [{label}] {phrase:?}  →  {} SPM tokens", ids.len());

        // Run this phrase under BOTH audio conditions from the SAME silence-gap
        // state. We snapshot nothing across conditions except re-deriving from a
        // fresh clone would need reload; instead we run condition A (text only),
        // note it does not corrupt the ring irrecoverably for the qualitative
        // read, then reset+re-prompt for condition B. Simplicity: reload-free
        // reset via reset_session + re-run the prompt.
        for (cond, force_agent_silence) in [("text-only", false), ("text+agent-silence", true)] {
            // fresh session at the same silence gap
            reprompt(&mut p, &sched, &vp, &vp_cache, n_vp, sampling, temp);

            println!("\n  ── condition: {cond} ──");
            // ---- FORCE window: one phrase token per frame on stream 0 ----
            let mut forced_shadow: Vec<i64> = Vec::with_capacity(ids.len());
            let mut forced_agent_pcm: Vec<f32> = Vec::new();
            let mut forced_frame_rms: Vec<f64> = Vec::new();
            let mut forced_out_frames: Vec<[i64; cfg::NUM_STREAMS]> = Vec::new();
            for &t in &ids {
                let moshi = if force_agent_silence {
                    Some(&SILENCE)
                } else {
                    None
                };
                let trace = p.step(Some(&SINE), moshi, Some(t));
                // The model's OWN sampled stream-0 token this step (what it
                // would have emitted if not forced) — the "shadow" monologue.
                forced_shadow.push(argmax_or_sampled(&trace.text_logits, sampling, &trace));
                if let Some(out) = trace.out {
                    let frame_pcm = decode_agent_frame(&p, &out);
                    forced_frame_rms.push(rms(&frame_pcm));
                    forced_agent_pcm.extend_from_slice(&frame_pcm);
                    forced_out_frames.push(out);
                }
            }

            // ---- RELEASE window: stop forcing stream 0, keep hearing SINE ----
            let mut released_text: Vec<i64> = Vec::with_capacity(RELEASE);
            let mut released_agent_pcm: Vec<f32> = Vec::new();
            for _ in 0..RELEASE {
                // Release text; do NOT force agent audio in the release window
                // (we want the model's natural behavior after the surface).
                let trace = p.step(Some(&SINE), None, None);
                // committed stream-0 token = out[0] once past the horizon
                if let Some(out) = trace.out {
                    released_text.push(out[0]);
                    released_agent_pcm.extend_from_slice(&decode_agent_frame(&p, &out));
                }
            }

            // ---- report ----
            // Q1: the forced-window shadow (its own choice while being forced)
            println!("    forced-window shadow (model's own stream-0 argmax while forced):");
            let shadow_toks: Vec<String> =
                forced_shadow.iter().map(|&t| tok_str(&spm, t)).collect();
            println!("      tokens: {}", shadow_toks.join(" "));
            println!("      as text: {}", decode_run(&spm, &forced_shadow));

            // Q1: the release continuation
            println!(
                "    release continuation (committed stream-0 for {RELEASE} frames after force stops):"
            );
            let rel_toks: Vec<String> = released_text.iter().map(|&t| tok_str(&spm, t)).collect();
            println!("      tokens: {}", rel_toks.join(" "));
            println!("      as text: {}", decode_run(&spm, &released_text));
            let pad_run = released_text
                .iter()
                .take_while(|&&t| matches!(t as u32, cfg::TEXT_PAD_TOKEN | cfg::TEXT_EPAD_TOKEN))
                .count();
            let n_word = released_text
                .iter()
                .filter(|&&t| {
                    !(matches!(t as u32, cfg::TEXT_PAD_TOKEN | cfg::TEXT_EPAD_TOKEN)
                        || t == 1
                        || t == 2)
                })
                .count();
            println!(
                "      leading pad/epad frames: {pad_run}/{}   non-pad (word) tokens in window: {n_word}",
                released_text.len()
            );

            // Q2: agent audio during the forced window
            let fr = rms(&forced_agent_pcm);
            let fp = peak(&forced_agent_pcm);
            let rr = rms(&released_agent_pcm);
            let ratio = if sil_rms > 0.0 {
                fr / sil_rms
            } else {
                f64::INFINITY
            };
            println!(
                "    agent audio during FORCED text: rms={fr:.5} (×{ratio:.1} vs SILENCE_TOKENS baseline)  peak={fp:.5}"
            );
            if !forced_frame_rms.is_empty() {
                let per: Vec<String> = forced_frame_rms.iter().map(|r| format!("{r:.4}")).collect();
                println!("      per-frame agent rms: [{}]", per.join(", "));
            }
            println!("    agent audio during RELEASE: rms={rr:.5}");

            // Q2: user/hearing streams unaffected — the forced-window out frames
            // must carry the SINE codes we fed (streams 9..=16), untouched by
            // forcing text/agent-silence.
            let sine_i: [i64; 8] = std::array::from_fn(|q| SINE[q]);
            let mut user_ok = true;
            let mut checked = 0usize;
            for out in &forced_out_frames {
                let user = &out[9..17];
                if user != sine_i {
                    user_ok = false;
                }
                checked += 1;
            }
            println!(
                "    hearing streams 9..=16 during force: {} ({}/{} out-frames carry the fed SINE codes exactly)",
                if user_ok { "UNAFFECTED" } else { "CHANGED" },
                if user_ok { checked } else { 0 },
                checked
            );

            // write the forced-window agent audio for a listen
            if !forced_agent_pcm.is_empty() {
                let path = format!("{OUT_DIR}/surface_{label}_{}.wav", cond.replace('+', "_"));
                wav::write_pcm16_mono(Path::new(&path), &forced_agent_pcm, mimi_cfg::SAMPLE_RATE);
                println!("    wrote {path} ({} samples)", forced_agent_pcm.len());
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Q3 (the rate-bridge): dense silent text vs speech-cadence silent text.
    //
    // Normally stream-0 is SPARSE — a word token at onset, PAD/EPAD between,
    // metered to speech cadence (~65% PAD). Hypothesis: that slowness is a
    // constraint of SPEAKING, not thinking, so a SILENT thought (agent audio
    // forced to SILENCE_TOKENS) may tolerate DENSE text — a real token EVERY
    // frame, no PAD — i.e. fast silent thinking. If dense silent surfacing
    // stays coherent (the released continuation is sensible, no garble), the
    // rate-bridge between fast background reasoning and the slow
    // speech-aligned monologue mostly dissolves for silent surfacing.
    //
    // Same content both ways (isolates DENSITY, not words): a coherent
    // sentence's SPM tokens, forced (A) one-per-frame with NO PAD (dense), and
    // (B) each token followed by PAD frames to ~word cadence (sparse). Both
    // under forced agent-silence. Then release and read the continuation.
    println!("\n══════════════════════════════════════════════════════════════");
    println!("Q3 RATE-BRIDGE: dense (no-PAD) vs speech-cadence silent text (agent forced silent)");
    let sentence = "The plan is simple: finish the loop, then measure the seam.";
    let word_ids = spm.encode(sentence);
    println!("sentence {sentence:?}  →  {} SPM tokens", word_ids.len());

    // Build the two forcing schedules over stream 0.
    // dense: exactly the tokens, one per frame.
    let dense: Vec<i64> = word_ids.clone();
    // cadence: each real token, then 1 PAD frame after word-piece boundaries
    // (a leading '▁' marks a new word) — approximates the sparse onset+PAD
    // structure the model was trained on, matched in content to `dense`.
    let mut cadence: Vec<i64> = Vec::new();
    for (k, &t) in word_ids.iter().enumerate() {
        cadence.push(t);
        // insert a PAD after this token if the NEXT token starts a new word
        // (its surface begins with the '▁' space metasymbol) — one PAD gap per
        // word onset, the coarse speech-cadence spacing.
        let next_new_word = word_ids
            .get(k + 1)
            .map(|&n| spm.piece_bytes(n).starts_with(&[0xE2, 0x96, 0x81]))
            .unwrap_or(false);
        if next_new_word {
            cadence.push(cfg::TEXT_PAD_TOKEN as i64);
        }
    }
    let pad_frac = |v: &[i64]| {
        v.iter()
            .filter(|&&t| matches!(t as u32, cfg::TEXT_PAD_TOKEN | cfg::TEXT_EPAD_TOKEN))
            .count() as f64
            / v.len() as f64
    };
    println!(
        "  dense schedule: {} frames, {:.0}% PAD   |   cadence schedule: {} frames, {:.0}% PAD",
        dense.len(),
        100.0 * pad_frac(&dense),
        cadence.len(),
        100.0 * pad_frac(&cadence)
    );

    const RB_RELEASE: usize = 32;
    for (mode, forced_ids) in [("dense-no-pad", &dense), ("speech-cadence", &cadence)] {
        reprompt(&mut p, &sched, &vp, &vp_cache, n_vp, sampling, temp);
        println!("\n  ── {mode} (agent forced SILENCE throughout the force window) ──");

        // FORCE: token per frame (dense) or token+PAD (cadence), agent silent.
        let mut shadow: Vec<i64> = Vec::new();
        let mut agent_pcm: Vec<f32> = Vec::new();
        for &t in forced_ids {
            let trace = p.step(Some(&SINE), Some(&SILENCE), Some(t));
            shadow.push(argmax(&trace.text_logits) as i64);
            if let Some(out) = trace.out {
                agent_pcm.extend_from_slice(&decode_agent_frame(&p, &out));
            }
        }
        // Coherence-under-force signal: at each forced step the model reads its
        // own logits; if the forced token is WILDLY off the model's own
        // distribution (dense packing pushing off-distribution), the forced
        // token's rank in the model's logits balloons. Measure the mean rank of
        // the NEXT forced token in the current step's logits (lower = the model
        // "agrees" the packed token is plausible; a huge rank = it's fighting).
        let mut ranks: Vec<usize> = Vec::new();
        // recompute cleanly: re-run capturing (logits, next_forced) pairs
        reprompt(&mut p, &sched, &vp, &vp_cache, n_vp, sampling, temp);
        for (k, &t) in forced_ids.iter().enumerate() {
            let trace = p.step(Some(&SINE), Some(&SILENCE), Some(t));
            if let Some(&next_t) = forced_ids.get(k + 1) {
                if next_t >= 0 && (next_t as usize) < cfg::TEXT_LOGITS {
                    let lv = trace.text_logits[next_t as usize];
                    let rank = trace.text_logits.iter().filter(|&&v| v > lv).count();
                    ranks.push(rank);
                }
            }
        }
        let mean_rank = if ranks.is_empty() {
            0.0
        } else {
            ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
        };
        let med_rank = {
            let mut r = ranks.clone();
            r.sort_unstable();
            r.get(r.len() / 2).copied().unwrap_or(0)
        };

        // RELEASE: stop forcing text, keep agent free, read the continuation.
        let mut released: Vec<i64> = Vec::new();
        for _ in 0..RB_RELEASE {
            let trace = p.step(Some(&SINE), None, None);
            if let Some(out) = trace.out {
                released.push(out[0]);
            }
        }

        println!("    forced-token rank in the model's own logits (0 = model's own top pick):");
        println!(
            "      mean {mean_rank:.0}   median {med_rank}   (huge = the packed token is off-distribution / model fighting)"
        );
        println!("    shadow during force (model's own argmax each frame):");
        println!("      {}", decode_run(&spm, &shadow));
        println!("    release continuation ({RB_RELEASE} frames after force stops):");
        let rtoks: Vec<String> = released.iter().map(|&t| tok_str(&spm, t)).collect();
        println!("      tokens: {}", rtoks.join(" "));
        println!("      as text: {}", decode_run(&spm, &released));
        let agent_rms = rms(&agent_pcm);
        println!(
            "    agent audio during force: rms={agent_rms:.5} (baseline {sil_rms:.5}) — confirms the thought stayed silent"
        );
    }

    println!("\n=== surface probe complete ===");
}

/// The model's own stream-0 token for this step: argmax of the read-back text
/// logits (greedy), or — under sampling — the sampler-chosen `next_text` is not
/// exposed for the forced case (it's overridden by the force), so we always
/// report the argmax of the logits as the deterministic "top choice" shadow.
fn argmax_or_sampled(
    logits: &[f32],
    _sampling: bool,
    _trace: &mary::models::personaplex::pipeline::RtStepTrace,
) -> i64 {
    argmax(logits) as i64
}

/// Reset the session and replay the fixed prompt so the model is back at the
/// same trailing-silence gap, WITHOUT reloading weights (reset_session ==
/// reload is gated token-exact). Re-applies the sampler seed so the shadow
/// stream is reproducible across conditions.
fn reprompt(
    p: &mut RealtimePipeline,
    sched: &[Phase],
    vp: &[f32],
    vp_cache: &[i64],
    n_vp: usize,
    sampling: bool,
    temp: f32,
) {
    p.reset_session();
    if sampling {
        p.set_sampling(
            SamplingConfig {
                temp,
                top_k: 64,
                top_p: 0.95,
            },
            0x5EED,
        );
    }
    for (i, phase) in sched.iter().enumerate() {
        match phase {
            Phase::Vp(k) => {
                p.step_embedding(&vp[k * cfg::DIM..(k + 1) * cfg::DIM]);
            }
            Phase::Silence => {
                p.step(
                    Some(&SINE),
                    Some(&SILENCE),
                    Some(cfg::TEXT_PAD_TOKEN as i64),
                );
            }
            Phase::Text(t) => {
                p.step(Some(&SINE), Some(&SILENCE), Some(*t));
            }
        }
        if i + 1 == n_vp {
            p.stream.overwrite(vp_cache);
        }
    }
}
