//! Voxtral parity gates — Burn port vs the CPU-f32 oracle goldens captured by
//! `golden/voxtral_capture.py` (transformers 5.13 voxtral_realtime, greedy):
//!
//!   1. prompt/pad     exact match on input_ids + padded audio construction
//!   2. mel            cos vs `deep_mel`
//!   3. conv stem      cos vs `deep_conv_stem`
//!   4. encoder        cos vs `deep_enc_final` (layer taps informational)
//!   5. projector      cos vs `deep_audio_embeds`
//!   6. delay cond     t_cond bit-exact-ish + 26 ada scales per delay ∈ {6,12,30}
//!   7. decoder        prefill final hidden + logits vs `deep_dec_final`/`deep_prefill_logits`
//!   8. greedy stream  full token match vs oracle (en_short/de_short, d=6,12,30)
//!   9. enc streaming  batch vs incremental (4-pos KV steps) encoder equivalence
//!  10. tokenizer      decoded text == oracle transcript
//!
//!   cargo run --release --features voxtral --bin voxtral_probe -- \
//!     [--pile <path>] [--gold <dir>] [--long] [--lane raw|fold|half]
//!
//! Weights come from `--pile`, else `$MARY_MODELS/voxtral_mini.pile`. The
//! native cohort gate requires one exact/f16 pair bit-identical to the source;
//! there is no missing-root fallback or sibling discovery. `--long` adds the
//! en_long / denglish streams (slow).
//!
//! Lanes: `raw` (default) = the full oracle parity suite on the op-for-op
//! layout — the trust anchor, unchanged. `fold` = the folded fast layout on
//! the fusion f32 backend; same greedy streams, must stay TOKEN-identical to
//! the oracle (folds are exact math). `half` = the folded layout on fusion
//! f16; tokens may drift by AR cascade, gate is WORD-exact transcripts.
//! `rawhalf` = the folded layout on the RAW (unfused) Metal f16 backend,
//! zero-copy loaded from the native f16 root; same gate as `half`, and the
//! printed token digest allows a cross-lane identity check against `half`.

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::models::f5::wav;
use mary::models::voxtral::config::*;
use mary::models::voxtral::decoder::time_embedding;
use mary::models::voxtral::fast::RealtimeTranscriber;
use mary::models::voxtral::pipeline::{
    SttPipeline, Transcriber, pad_audio, prompt_ids, transcribe,
};
use mary::models::voxtral::tokenizer::Tekken;
use mary::nn::backend::{B, BFused, BFusedHalf, BHalf};
use mary::nn::npy;
use std::path::{Path, PathBuf};

fn golden(gold: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    npy::load_npy(&gold.join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
}

fn metrics(name: &str, a: &[f32], b: &[f32], cos_gate: f64) -> bool {
    assert_eq!(a.len(), b.len(), "{name}: len {} vs {}", a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxabs = maxabs.max((x - y).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let ok = cos > cos_gate;
    println!(
        "  {} {name:24} cos={cos:.8}  max|Δ|={maxabs:.3e}",
        if ok { "✓" } else { "✗" }
    );
    ok
}

fn to_host<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .into_vec()
        .expect("host readback")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .map(|i| args[i + 1].clone())
    };
    let pile =
        mary::paths::model(arg("--pile").as_deref(), "voxtral_mini.pile").unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
    let gold = PathBuf::from(arg("--gold").unwrap_or_else(|| "golden/voxtral".into()));
    let long = args.iter().any(|a| a == "--long");

    let lane = arg("--lane").unwrap_or_else(|| "raw".into());
    let dev = Default::default();
    eprintln!("loading stt from {pile:?} (lane {lane}) ...");
    let t0 = std::time::Instant::now();
    let snapshot = mary::model_collection::load_model_collection_local_latest(&pile)
        .expect("load native Voxtral snapshot");
    let tekken = Tekken::from_snapshot(&snapshot).expect("Tekken tokenizer in the Voxtral pile");
    let loader = mary::models::voxtral::VoxtralWeights::from_snapshot(snapshot)
        .expect("select complete native Voxtral cohort")
        .into_loader();
    match lane.as_str() {
        "raw" => {}
        "fold" => {
            let stt = RealtimeTranscriber::<BFused>::load(&loader, tekken, 4096, &dev);
            drop(loader);
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            return fast_lane_gates(&stt, &gold, long, /*exact*/ true);
        }
        "half" => {
            let stt = RealtimeTranscriber::<BFusedHalf>::load(&loader, tekken, 4096, &dev);
            drop(loader);
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            return fast_lane_gates(&stt, &gold, long, /*exact*/ false);
        }
        "rawhalf" => {
            let stt = RealtimeTranscriber::<BHalf>::load(&loader, tekken, 4096, &dev);
            drop(loader);
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            return fast_lane_gates(&stt, &gold, long, /*exact*/ false);
        }
        other => panic!("unknown --lane {other} (raw|fold|half|rawhalf)"),
    }
    let stt = Transcriber::<B>::load(&loader, tekken, 4096, &dev);
    drop(loader);
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let mut all_ok = true;
    let mut gate = |ok: bool| all_ok &= ok;

    // ── 1. prompt + padding construction (host, exact) ──────────────────
    println!("gate 1: prompt/pad construction");
    let (ids_f, _) = golden(&gold, "deep_input_ids");
    let oracle_ids: Vec<u32> = ids_f.iter().map(|&x| x as u32).collect();
    let ours = prompt_ids(6);
    let ids_ok = ours == oracle_ids;
    println!(
        "  {} input_ids ({} tokens)",
        if ids_ok { "✓" } else { "✗" },
        ours.len()
    );
    gate(ids_ok);

    let (clip, sr) = wav::read_pcm16_mono(&gold.join("clips/en_short.wav"));
    assert_eq!(sr, 16000);
    let (opad, _) = golden(&gold, "deep_padded_audio");
    let padded = pad_audio(&clip, 6);
    let pad_ok =
        padded.len() == opad.len() && padded.iter().zip(&opad).all(|(a, b)| (a - b).abs() < 1e-6);
    println!(
        "  {} padded audio ({} vs {} samples)",
        if pad_ok { "✓" } else { "✗" },
        padded.len(),
        opad.len()
    );
    gate(pad_ok);

    // ── 2. mel ───────────────────────────────────────────────────────────
    println!("gate 2: mel front end");
    let mel = stt.mel.forward(&padded, true, &dev);
    let (omel, omel_shape) = golden(&gold, "deep_mel");
    assert_eq!(mel.dims()[2], omel_shape[2], "mel frame count");
    gate(metrics("mel", &to_host(mel.clone()), &omel, 0.9999));

    // ── 3. conv stem ─────────────────────────────────────────────────────
    println!("gate 3: conv stem");
    let stem = stt.encoder.stem(mel);
    let (ostem, _) = golden(&gold, "deep_conv_stem");
    gate(metrics("conv_stem", &to_host(stem.clone()), &ostem, 0.9999));

    // ── 4. encoder (batch) ───────────────────────────────────────────────
    println!("gate 4: audio encoder");
    let mut caches = stt.encoder.new_caches();
    let enc = stt.encoder.forward(stem.clone(), &mut caches);
    let (oenc, _) = golden(&gold, "deep_enc_final");
    gate(metrics(
        "encoder_final",
        &to_host(enc.clone()),
        &oenc,
        0.999,
    ));

    // ── 5. projector ─────────────────────────────────────────────────────
    println!("gate 5: projector");
    let audio_embeds = stt.encoder.project(enc);
    let (oemb, _) = golden(&gold, "deep_audio_embeds");
    gate(metrics(
        "audio_embeds",
        &to_host(audio_embeds.clone()),
        &oemb,
        0.999,
    ));

    // ── 6. delay conditioning (the whole point — tight gates) ───────────
    println!("gate 6: delay conditioning");
    for nd in [6usize, 12, 30] {
        let (ot, _) = golden(&gold, &format!("t_cond_d{nd}"));
        let t = time_embedding(nd);
        gate(metrics(&format!("t_cond_d{nd}"), &t, &ot, 0.999999));
        let (oada, _) = golden(&gold, &format!("ada_scales_d{nd}"));
        let scales = stt.decoder.ada_scales(nd, &dev);
        let ours: Vec<f32> = scales.0.iter().flat_map(|s| to_host(s.clone())).collect();
        // oracle saved (1 + ada(t_cond)) per layer, stacked [26, 1, 3072]
        gate(metrics(
            &format!("ada_scales_d{nd}"),
            &ours,
            &oada,
            0.999999,
        ));
    }

    // ── 7. decoder prefill ───────────────────────────────────────────────
    println!("gate 7: decoder prefill");
    let l = oracle_ids.len();
    let tok = stt.decoder.embed.forward(&oracle_ids, &dev);
    let embeds = tok + audio_embeds.clone().narrow(1, 0, l);
    let (opre, _) = golden(&gold, "deep_prefill_embeds");
    gate(metrics(
        "prefill_embeds",
        &to_host(embeds.clone()),
        &opre,
        0.9999,
    ));

    let ada = stt.decoder.ada_scales(6, &dev);
    let mut dcaches = stt.decoder.new_caches();
    let hidden = stt.decoder.forward(embeds, &ada, &mut dcaches);
    let (ohid, _) = golden(&gold, "deep_dec_final");
    gate(metrics(
        "dec_final_hidden",
        &to_host(hidden.clone()),
        &ohid,
        0.999,
    ));

    let logits = stt.decoder.logits_last(hidden);
    let (ologits, _) = golden(&gold, "deep_prefill_logits");
    let ours_logits = to_host(logits.clone());
    gate(metrics("prefill_logits", &ours_logits, &ologits, 0.999));
    let oargmax = ologits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let aargmax = ours_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let am_ok = oargmax == aargmax;
    println!(
        "  {} prefill argmax {} vs {}",
        if am_ok { "✓" } else { "✗" },
        aargmax,
        oargmax
    );
    gate(am_ok);

    // ── 8+10. greedy streams + transcripts ───────────────────────────────
    println!("gate 8/10: greedy token streams + transcripts");
    let stream_gate = |clip: &str, nd: usize| {
        let (audio, sr) = wav::read_pcm16_mono(&gold.join(format!("clips/{clip}.wav")));
        assert_eq!(sr, 16000);
        let t0 = std::time::Instant::now();
        let out = stt.transcribe(&audio, nd * 80, false);
        let secs = t0.elapsed().as_secs_f64();
        let (otok_f, _) = golden(&gold, &format!("{clip}_d{nd}_tokens"));
        let oracle: Vec<u32> = otok_f.iter().map(|&x| x as u32).collect();
        let matched = out
            .tokens
            .iter()
            .zip(&oracle)
            .take_while(|(a, b)| a == b)
            .count();
        let tok_ok = matched == oracle.len() && out.tokens.len() == oracle.len();
        let otext = std::fs::read_to_string(gold.join(format!("{clip}_d{nd}_text.txt")))
            .unwrap_or_default();
        let text_ok = out.text == otext;
        println!(
            "  {} {clip} d={nd}: {matched}/{} tokens identical ({} ours), text {} [{secs:.0}s]",
            if tok_ok { "✓" } else { "✗" },
            oracle.len(),
            out.tokens.len(),
            if text_ok { "==" } else { "!=" },
        );
        if !text_ok {
            println!("      ours:   {:?}", out.text);
            println!("      oracle: {otext:?}");
        }
        let ms: Vec<f32> = out.timings.iter().map(|t| t.decoder_ms).collect();
        if !ms.is_empty() {
            let mean = ms.iter().sum::<f32>() / ms.len() as f32;
            let mut s = ms.clone();
            s.sort_by(f32::total_cmp);
            println!(
                "      decoder ms/frame: mean {mean:.1}, p50 {:.1}, p95 {:.1} (80 ms budget)",
                s[s.len() / 2],
                s[s.len() * 95 / 100]
            );
        }
        tok_ok && text_ok
    };
    for nd in [6usize, 12, 30] {
        gate(stream_gate("en_short", nd));
    }
    gate(stream_gate("de_short", 6));
    if long {
        gate(stream_gate("denglish", 6));
        gate(stream_gate("en_long", 6));
    }

    // ── 9. incremental encoder ≡ batch encoder ───────────────────────────
    println!("gate 9: incremental vs batch encoder");
    let n_tok = stem.dims()[1] / DOWNSAMPLE;
    let mut inc_caches = stt.encoder.new_caches();
    let mut chunks = Vec::new();
    for t in 0..n_tok {
        let h = stt.encoder.forward(
            stem.clone().narrow(1, t * DOWNSAMPLE, DOWNSAMPLE),
            &mut inc_caches,
        );
        chunks.push(stt.encoder.project(h));
    }
    let inc = Tensor::cat(chunks, 1);
    gate(metrics(
        "enc_incremental",
        &to_host(inc),
        &to_host(audio_embeds),
        0.9999999,
    ));

    println!();
    if all_ok {
        println!("ALL GATES PASSED");
    } else {
        println!("SOME GATES FAILED");
        std::process::exit(1);
    }
    let _ = TensorData::new(vec![0f32], [1]); // keep import used on all paths
}

/// Fast-lane gates: greedy streams vs the oracle goldens through the FOLDED
/// layout. `exact` (f32 fold): tokens + transcript must be identical — the
/// folds are exact math, any drift is a bug. Non-exact (f16): tokens drift by
/// AR cascade; the gate is word-exact transcripts (LCS == oracle words), with
/// token prefix-match reported for information.
///
/// The f16 lane runs the INCREMENTAL encoder (the production streaming
/// schedule, 4-position KV steps): burn 0.21's fused-reduce codegen
/// (`ReduceOptimization` → `GlobalArgsLaunch::strides` bounds panic) breaks
/// on the batch pass's big shapes on the f16 backend — f32 fold passes the
/// identical batch graph, and every streaming-sized f16 shape is fine.
fn fast_lane_gates<BX: Backend, O: SttPipeline<BX>>(stt: &O, gold: &Path, long: bool, exact: bool) {
    let mut all_ok = true;
    let mut clips: Vec<(&str, usize)> = vec![
        ("en_short", 6),
        ("en_short", 12),
        ("en_short", 30),
        ("de_short", 6),
    ];
    if long {
        clips.push(("denglish", 6));
        clips.push(("en_long", 6));
    }
    for (clip, nd) in clips {
        let (audio, sr) = wav::read_pcm16_mono(&gold.join(format!("clips/{clip}.wav")));
        assert_eq!(sr, 16000);
        let t0 = std::time::Instant::now();
        let out = transcribe(stt, &audio, nd * 80, /*incremental_encoder*/ !exact);
        let secs = t0.elapsed().as_secs_f64();
        let (otok_f, _) = golden(gold, &format!("{clip}_d{nd}_tokens"));
        let oracle: Vec<u32> = otok_f.iter().map(|&x| x as u32).collect();
        let matched = out
            .tokens
            .iter()
            .zip(&oracle)
            .take_while(|(a, b)| a == b)
            .count();
        let otext = std::fs::read_to_string(gold.join(format!("{clip}_d{nd}_text.txt")))
            .unwrap_or_default();
        let (lcs, total) = word_lcs(&out.text, &otext);
        let ok = if exact {
            matched == oracle.len() && out.tokens.len() == oracle.len() && out.text == otext
        } else {
            lcs == total
        };
        let ms: Vec<f32> = out.timings.iter().map(|t| t.decoder_ms).collect();
        let (p50, p95) = if ms.is_empty() {
            (0.0, 0.0)
        } else {
            let mut s = ms.clone();
            s.sort_by(f32::total_cmp);
            (s[s.len() / 2], s[s.len() * 95 / 100])
        };
        println!(
            "  {} {clip} d={nd}: tokens {matched}/{} ({} ours), words {lcs}/{total}, \
             digest {:016x}, dec ms/frame p50 {p50:.1} p95 {p95:.1} [{secs:.0}s]",
            if ok { "✓" } else { "✗" },
            oracle.len(),
            out.tokens.len(),
            token_digest(&out.tokens),
        );
        if !ok {
            println!("      ours:   {:?}", out.text);
            println!("      oracle: {otext:?}");
        }
        all_ok &= ok;
    }
    println!();
    if all_ok {
        println!("ALL GATES PASSED");
    } else {
        println!("SOME GATES FAILED");
        std::process::exit(1);
    }
}

/// FNV-1a 64 over the token stream (LE bytes) — equal digests across two
/// lane runs = token-identical streams, without goldens in the middle.
fn token_digest(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// LCS over whitespace words (case/punct-insensitive), vs the oracle count.
fn word_lcs(got: &str, want: &str) -> (usize, usize) {
    let a: Vec<&str> = got.split_whitespace().collect();
    let b: Vec<&str> = want.split_whitespace().collect();
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![0usize; (n + 1) * (m + 1)];
    for i in 1..=n {
        for j in 1..=m {
            dp[i * (m + 1) + j] = if a[i - 1]
                .trim_matches(|c: char| !c.is_alphanumeric())
                .eq_ignore_ascii_case(b[j - 1].trim_matches(|c: char| !c.is_alphanumeric()))
            {
                dp[(i - 1) * (m + 1) + j - 1] + 1
            } else {
                dp[(i - 1) * (m + 1) + j].max(dp[i * (m + 1) + j - 1])
            };
        }
    }
    (dp[n * (m + 1) + m], m)
}
