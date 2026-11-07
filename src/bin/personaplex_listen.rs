//! Generate PersonaPlex audio at a chosen weight format, so a human can HEAR
//! what quantisation costs.
//!
//! Every quality figure we have for q4 is a cosine. This produces the artifact
//! those numbers are a proxy for. It deliberately draws no conclusion — it
//! writes a WAV and reports only what is objectively checkable.
//!
//! Golden-free: uses `RealtimePipeline`'s own public surface
//! (`prompt_voice` / `prompt_silence` / `step_user_frame` / `decode`) rather
//! than `personaplex_rt_probe`, which is an oracle-parity harness and needs
//! captured goldens that no longer exist.
//!
//! Everything comes from the pile — weights AND tokenizer. The SPM tokenizer is
//! a graph of 32k scored pieces (see `mary::tokenizer`), not a `.model` file, so
//! this binary reads no loose model files at all.
//!
//! GREEDY on purpose: with sampling off, two runs differ ONLY by the weight
//! format. Any audible difference is quantisation, not a different draw.
//!
//! USER AUDIO IS REQUIRED, and so is STOPPING IT. PersonaPlex is full-duplex,
//! which cuts both ways. Feed it nothing and the agent correctly says NOTHING —
//! a first run that fed pure SILENCE produced rms 0.0002 (-74 dB), the model
//! behaving properly, not a bug. But feed it a CONTINUOUS user turn for the
//! whole window and it does the other correct thing: it yields the floor,
//! speaking only in short interjections and falling quiet again whenever the
//! user is still going. That produces audio that breaks off after a couple of
//! seconds and resumes later — which is good conversational behaviour and a
//! USELESS quantisation comparison, because the agent barely speaks.
//!
//! So the user turn is deliberately SHORT: `user_frames` of real speech, then
//! silence for the rest, which hands the agent the floor and gets a continuous
//! reply worth listening to.
//!
//!   personaplex_listen <fmt> <out.wav> [frames] [user.wav] [pile] [voice.pt]
//!     fmt    f16 | q8 | q4
//!     frames 12.5 Hz, so 125 frames = 10 s (default 125)
//!
//! The three paths also read from PERSONAPLEX_USER_WAV / PERSONAPLEX_PILE /
//! PERSONAPLEX_VOICE_PROMPT when the positional argument is absent.
use std::path::Path;
use std::time::Instant;

use mary::models::f5::wav;
use mary::models::personaplex::config as cfg;
use mary::models::personaplex::mimi::config as mimi_cfg;
use mary::models::personaplex::pipeline::{RealtimePipeline, SILENCE, agent_codes};
use mary::models::personaplex::prompt::{Prompt, wrap_with_system_tags};
use mary::models::personaplex::sampling::SamplingConfig;
use mary::models::personaplex::temporal_metal::WeightFmt;

// The model files are named, not located: positional argument, else the
// matching environment variable, else `$MARY_MODELS/<name>`. The user clip is
// a recording you supply, so it stays relative to the working directory.
const PILE_NAME: &str = "personaplex.pile";
const VOICE_NAME: &str = "voice_prompt.pt";
const DEFAULT_USER: &str = "user_turn_24k.wav";
const SYSTEM_TEXT: &str = "You are a helpful assistant. You speak with a warm, \
                           curious, and direct voice. Answer clearly.";

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let fmt_s = a.get(1).cloned().unwrap_or_else(|| "q4".into());
    let out = a
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/personaplex_listen.wav".into());
    let frames: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(125);
    // positional > environment > relative default
    let pick = |arg: Option<&String>, env: &str, default: &str| -> String {
        arg.cloned()
            .or_else(|| std::env::var(env).ok())
            .unwrap_or_else(|| default.into())
    };
    // positional > environment > $MARY_MODELS/<name>; never a guessed path
    let pick_model = |arg: Option<&String>, env: &str, name: &str| -> String {
        let explicit = arg.cloned().or_else(|| std::env::var(env).ok());
        mary::paths::model(explicit.as_deref(), name)
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2)
            })
            .to_string_lossy()
            .into_owned()
    };
    let user_wav = pick(a.get(4), "PERSONAPLEX_USER_WAV", DEFAULT_USER);
    let pile = pick_model(a.get(5), "PERSONAPLEX_PILE", PILE_NAME);
    let voice_pt = pick_model(a.get(6), "PERSONAPLEX_VOICE_PROMPT", VOICE_NAME);

    let fmt = match fmt_s.as_str() {
        "f16" => WeightFmt::F16,
        "q8" => WeightFmt::Q8,
        "q4" => WeightFmt::Q4,
        other => panic!("unknown format {other} (expected f16|q8|q4)"),
    };

    println!("format : {fmt:?}");
    println!("pile   : {pile}");
    println!("voice  : {voice_pt}");
    println!("user   : {user_wav}");
    println!(
        "frames : {frames}  ({:.1} s at 12.5 Hz)",
        frames as f64 / 12.5
    );

    let t0 = Instant::now();
    let source = mary::persist::personaplex_model(Path::new(&pile))
        .unwrap_or_else(|e| panic!("pile: {e}"))
        .into_runtime_source();
    // `load_auto` deliberately recomputes the quantized runtime form from the
    // authority-bound source.
    let mut pipe = RealtimePipeline::load_auto(&source, fmt, true);
    // GREEDY vs SAMPLING. Greedy is attractive because it makes two runs differ
    // only by weight format — but greedy decoding collapses audio LMs to a
    // near-constant code, which is exactly what a first attempt produced (agent
    // emitted 3 distinct codebook-0 values across 125 frames, rms -74 dB).
    // A fixed seed keeps the comparison fair: same logits -> same draw.
    let greedy = std::env::var("GREEDY").is_ok();
    if greedy {
        pipe.set_greedy();
        println!("decode : greedy — deterministic, so only the weights differ");
    } else {
        pipe.set_sampling(
            SamplingConfig {
                temp: 0.8,
                top_k: 250,
                top_p: 0.95,
            },
            1234_5678_u64,
        );
        println!("decode : sampling temp 0.8 top_k 250 top_p 0.95, seed fixed");
    }
    println!("loaded : {:.1}s", t0.elapsed().as_secs_f64());

    // The SPM tokenizer is LOAD-BEARING, not optional. Without a text system
    // prompt the model never enters conversational mode: it emitted 3 distinct
    // codebook-0 values across 125 frames (rms -74 dB) while the decoder was
    // provably fine (user-code round-trip rms 0.1032).
    let spm = mary::persist::load_spm_tokenizer_from_pile(Path::new(&pile))
        .unwrap_or_else(|e| panic!("tokenizer from pile: {e}"));
    let vs = spm.vocab_size();
    // Compare against TEXT_CARD, not TEXT_VOCAB: TEXT_VOCAB = TEXT_CARD + 1,
    // the extra slot being the initial/ungenerated marker (same pattern the
    // audio streams use with 2048). The tokenizer supplies TEXT_CARD pieces.
    println!(
        "spm    : vocab {vs} (model TEXT_CARD {}, TEXT_VOCAB {})",
        cfg::TEXT_CARD,
        cfg::TEXT_VOCAB
    );
    assert_eq!(
        vs,
        cfg::TEXT_CARD,
        "tokenizer vocab {vs} != TEXT_CARD {} — WRONG TOKENIZER, prompts would be garbage",
        cfg::TEXT_CARD
    );

    let prompt = Prompt::build(
        Path::new(&voice_pt),
        &spm,
        &wrap_with_system_tags(SYSTEM_TEXT),
    );
    println!(
        "prompt : {} voice frames + {} text tokens, {} total steps",
        prompt.voice.n_frames,
        prompt.text_tokens.len(),
        prompt.total_steps()
    );
    pipe.run_prompt(&prompt);

    // Mimi-encode the user turn. Without this the agent has nothing to reply to.
    let (mut samples, sr) = wav::read_pcm16_mono(Path::new(&user_wav));
    assert_eq!(
        sr,
        mimi_cfg::SAMPLE_RATE as u32,
        "user wav must be {} Hz",
        mimi_cfg::SAMPLE_RATE
    );
    let need = frames * mimi_cfg::SAMPLES_PER_FRAME;
    if samples.len() < need {
        samples.resize(need, 0.0);
    }
    samples.truncate(need);
    let user_codes = pipe.encoder.encode(&samples);
    // Hand the floor over after a short turn. Without this the model keeps
    // deferring to a user who never stops talking.
    let user_frames: usize = std::env::var("PERSONAPLEX_USER_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or((frames / 3).clamp(1, 40));
    println!(
        "user   : {} frames Mimi-encoded, speaking for {user_frames} ({:.1}s) then yielding",
        user_codes.len(),
        user_frames as f64 / 12.5
    );

    let t1 = Instant::now();
    let mut agent: Vec<[u32; mimi_cfg::NUM_CODEBOOKS]> = Vec::with_capacity(frames);
    let mut skipped = 0usize;
    for f in 0..frames {
        let c: [i64; 8] = match user_codes.get(f).filter(|_| f < user_frames) {
            Some(u) => std::array::from_fn(|q| u[q] as i64),
            None => SILENCE,
        };
        match pipe.step_user_frame(&c) {
            Some(o) => agent.push(agent_codes(&o)),
            None => skipped += 1,
        }
    }
    let r#gen = t1.elapsed().as_secs_f64();

    // DIAGNOSTIC: identical rms across different user input means the input is
    // not reaching the model, or the agent is emitting a constant. Check both.
    {
        use std::collections::HashSet;
        let u: HashSet<_> = user_codes.iter().take(frames).map(|c| c[0]).collect();
        let a0: HashSet<_> = agent.iter().map(|c| c[0]).collect();
        let all_same = agent.windows(2).all(|w| w[0] == w[1]);
        println!(
            "diag   : user cb0 distinct {} | agent cb0 distinct {} | agent constant {}",
            u.len(),
            a0.len(),
            all_same
        );
        if let Some(f) = agent.first() {
            println!("diag   : agent frame[0] {:?}", f);
        }
        if agent.len() > 60 {
            println!("diag   : agent frame[60] {:?}", agent[60]);
        }
    }
    println!(
        "gen    : {:.2}s for {} frames ({:.1} ms/frame, realtime budget 80 ms){}",
        r#gen,
        agent.len(),
        r#gen / agent.len().max(1) as f64 * 1e3,
        if skipped > 0 {
            format!(", {skipped} pre-horizon frames skipped")
        } else {
            String::new()
        }
    );

    // CONTROL: decode the USER's own codes — known-good, straight from real
    // speech through the encoder. If this is also silent the decoder is at
    // fault; if it is audible, the decoder works and the agent codes are the
    // problem. Different agent codes producing byte-identical audio stats is
    // what motivated this check.
    {
        let uc: Vec<[u32; mimi_cfg::NUM_CODEBOOKS]> =
            user_codes.iter().take(frames).map(|c| *c).collect();
        let upcm = pipe.decode(&uc);
        let urms = (upcm.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
            / upcm.len().max(1) as f64)
            .sqrt();
        let upeak = upcm.iter().fold(0f32, |m, v| m.max(v.abs()));
        println!("CONTROL: user-code round-trip rms {urms:.4} peak {upeak:.4}");
        if urms < 5e-3 {
            println!("  => DECODER IS BROKEN (known-good speech codes decode to silence)");
        } else {
            println!("  => decoder OK (this control says nothing about the agent codes)");
        }
        wav::write_pcm16_mono(
            Path::new("/tmp/listen_control_userroundtrip.wav"),
            &upcm,
            mimi_cfg::SAMPLE_RATE as u32,
        );
    }

    let pcm = pipe.decode(&agent);
    let rms =
        (pcm.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / pcm.len().max(1) as f64).sqrt();
    let peak = pcm.iter().fold(0f32, |m, v| m.max(v.abs()));
    let clipped = pcm.iter().filter(|v| v.abs() >= 0.999).count();
    let silent = rms < 5e-3; // -46 dB: anything below this is not speech

    println!(
        "audio  : {} samples ({:.2} s), rms {rms:.4}, peak {peak:.4}, clipped {clipped}",
        pcm.len(),
        pcm.len() as f64 / mimi_cfg::SAMPLE_RATE as f64
    );
    if silent {
        println!("  !! NEAR-SILENT — the decode produced no signal; do not read this as a");
        println!("     quantisation result, it means the generation or decode path failed.");
    }

    wav::write_pcm16_mono(Path::new(&out), &pcm, mimi_cfg::SAMPLE_RATE as u32);
    println!("wrote  : {out}");
    println!("\n(no judgement offered — this tool cannot evaluate audio. Listen and decide.)");
}
