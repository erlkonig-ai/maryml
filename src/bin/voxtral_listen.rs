//! The stt, live: streaming transcription through `StreamingTranscriber`.
//!
//! File mode (real-time-paced chunk feed; `--fast` = unpaced, for compute
//! benchmarks):
//!   cargo run --release --features voxtral --bin voxtral_listen -- \
//!     [--pile <voxtral_native.pile>] --wav clip.wav [--delay-ms 480] \
//!     [--chunk-ms 80] [--fast] [--expect-text file.txt] [--lane half] [--passes N]
//!
//! `--passes N` streams the clip N times, each a fresh stream over the same
//! loaded model, with a report per pass: pass 1 carries the kernel compiles
//! and autotune, later passes are the warm steady state.
//!
//! Lanes (one backend per process — two fusion runtimes thrash each other).
//! The backend family is `nn::backend::hear`: CUDA when built with
//! `voxtral-cuda`, wgpu (Metal on the Mac, Vulkan on Linux) otherwise.
//!   raw      parity-first layout on the raw f32 backend (trust anchor)
//!   fused    same layout on the fusion-wrapped f32 backend
//!   fold     folded fast layout (wide qkv, norms in matmul rows), fusion f32
//!   half     folded fast layout, fusion f16 (default — the realtime lane)
//!   rawhalf  folded fast layout, RAW (unfused) f16 — on the Mac it loads
//!            ZERO-COPY: f16 leaves alias the native collection mmap straight
//!            onto the GPU (fold sources + the embed table; folded results are
//!            new GPU buffers, the embed stays file-backed for the process life)
//!
//! The Tekken tokenizer is read from the same pile as the weights.
//!
//! Mic mode (mic capture rides the shared `listen` feature, so build
//! voxtral,listen):
//!   cargo run --release --features voxtral,listen --bin voxtral_listen -- \
//!     --mic [--mic-device name] [--delay-ms 480]
//!
//! Prints tokens as they emit (stdout, flushed per token), then a latency
//! report: per-token wall latency ("last sample available" → "token read
//! back") mean/p50/p95, plus per-frame COMPUTE (encoder submit + decoder
//! step incl. the argmax sync) p50/p95 against the 80 ms budget.

use mary::models::f5::wav;
use mary::models::voxtral::config::*;
use mary::models::voxtral::fast::RealtimeTranscriber;
use mary::models::voxtral::pipeline::{
    StreamedToken, StreamingTranscriber, SttPipeline, Transcriber,
};
use mary::models::voxtral::tokenizer::Tekken;
use mary::nn::backend::hear;
use std::io::Write;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

/// Pin to QOS_CLASS_USER_INTERACTIVE (0x21) — the qwen3tts lesson: background
/// daemons otherwise steal enough cores to swing the loop 4-10×.
fn set_interactive_qos() {
    #[cfg(target_os = "macos")]
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x21, 0);
    }
}

struct Args {
    pile: PathBuf,
    delay_ms: usize,
    chunk_ms: usize,
    fast: bool,
    passes: usize,
    wav: Option<String>,
    expect_text: Option<String>,
    mic: bool,
    mic_device: Option<String>,
}

fn main() -> anyhow::Result<()> {
    set_interactive_qos();
    let argv: Vec<String> = std::env::args().collect();
    let arg = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .map(|i| argv[i + 1].clone())
    };
    let flag = |name: &str| argv.iter().any(|a| a == name);

    let args = Args {
        pile: mary::paths::model(arg("--pile").as_deref(), "voxtral_mini.pile")?,
        delay_ms: arg("--delay-ms").map(|s| s.parse().unwrap()).unwrap_or(480),
        chunk_ms: arg("--chunk-ms").map(|s| s.parse().unwrap()).unwrap_or(80),
        fast: flag("--fast"),
        passes: arg("--passes").map(|s| s.parse().unwrap()).unwrap_or(1),
        wav: arg("--wav"),
        expect_text: arg("--expect-text"),
        mic: flag("--mic") || arg("--mic-device").is_some(),
        mic_device: arg("--mic-device"),
    };
    let lane = arg("--lane").unwrap_or_else(|| "half".into());

    let dev = hear::Device::default();
    eprintln!(
        "[listen] loading stt from {:?} (lane {lane}) ...",
        args.pile
    );
    let t0 = std::time::Instant::now();
    let snapshot = mary::model_collection::load_model_collection_local_latest(&args.pile)?;
    let tekken = Tekken::from_snapshot(&snapshot)?;
    let loader = mary::models::voxtral::VoxtralWeights::from_snapshot(snapshot)?.into_loader();
    let max_tokens = 8192;
    match lane.as_str() {
        "raw" => {
            let stt = Transcriber::<hear::Raw>::load(&loader, tekken, max_tokens, &dev);
            drop(loader);
            eprintln!(
                "[listen] loaded in {:.1}s; delay {} ms",
                t0.elapsed().as_secs_f64(),
                args.delay_ms
            );
            go(&stt, &args)
        }
        "fused" => {
            let stt = Transcriber::<hear::Fused>::load(&loader, tekken, max_tokens, &dev);
            drop(loader);
            eprintln!(
                "[listen] loaded in {:.1}s; delay {} ms",
                t0.elapsed().as_secs_f64(),
                args.delay_ms
            );
            go(&stt, &args)
        }
        "fold" => {
            let stt = RealtimeTranscriber::<hear::Fused>::load(&loader, tekken, max_tokens, &dev);
            drop(loader);
            eprintln!(
                "[listen] loaded in {:.1}s; delay {} ms",
                t0.elapsed().as_secs_f64(),
                args.delay_ms
            );
            go(&stt, &args)
        }
        "half" => {
            let stt =
                RealtimeTranscriber::<hear::FusedHalf>::load(&loader, tekken, max_tokens, &dev);
            drop(loader);
            eprintln!(
                "[listen] loaded in {:.1}s; delay {} ms",
                t0.elapsed().as_secs_f64(),
                args.delay_ms
            );
            go(&stt, &args)
        }
        "rawhalf" => {
            let stt = RealtimeTranscriber::<hear::RawHalf>::load(&loader, tekken, max_tokens, &dev);
            drop(loader);
            eprintln!(
                "[listen] loaded in {:.1}s; delay {} ms",
                t0.elapsed().as_secs_f64(),
                args.delay_ms
            );
            go(&stt, &args)
        }
        other => anyhow::bail!("unknown --lane {other} (raw|fused|fold|half|rawhalf)"),
    }
}

fn go<B: burn::prelude::Backend, O: SttPipeline<B>>(stt: &O, args: &Args) -> anyhow::Result<()> {
    if let Some(wav_path) = &args.wav {
        let (audio, sr) = wav::read_pcm16_mono(std::path::Path::new(&wav_path));
        anyhow::ensure!(sr == 16000, "expected 16 kHz wav, got {sr}");
        for pass in 1..=args.passes {
            if args.passes > 1 {
                eprintln!("[listen] pass {pass}/{}", args.passes);
            }
            stream_file(stt, args, wav_path, &audio)?;
        }
        return Ok(());
    }

    if args.mic {
        #[cfg(feature = "listen")]
        return mic_mode(stt, args.delay_ms, args.mic_device.clone());
        #[cfg(not(feature = "listen"))]
        anyhow::bail!(
            "mic mode ({:?}) requires --features listen",
            args.mic_device
        );
    }

    anyhow::bail!("pass --wav <file> or --mic (mic requires --features voxtral,listen)");
}

/// One fresh stream over the whole clip plus the offline path's trailing
/// silence, with the latency report and the optional word check.
fn stream_file<B: burn::prelude::Backend, O: SttPipeline<B>>(
    stt: &O,
    args: &Args,
    wav_path: &str,
    audio: &[f32],
) -> anyhow::Result<()> {
    let mut stream = StreamingTranscriber::new(stt, args.delay_ms);
    let mut emitted: Vec<StreamedToken> = Vec::new();
    let print_tokens = |stt: &O, toks: Vec<StreamedToken>, sink: &mut Vec<StreamedToken>| {
        for t in toks {
            let piece = stt.tekken().decode(&[t.id]);
            print!("{piece}");
            std::io::stdout().flush().ok();
            sink.push(t);
        }
    };

    let chunk = SAMPLE_RATE * args.chunk_ms / 1000;
    let chunk_ms = args.chunk_ms;
    eprintln!(
        "[listen] streaming {wav_path} ({:.1}s) in {chunk_ms} ms chunks{}",
        audio.len() as f32 / SAMPLE_RATE as f32,
        if args.fast {
            " (unpaced)"
        } else {
            " (real-time paced)"
        }
    );
    // warm the pipeline shapes (JIT/autotune) on the silence prefix
    let toks = stream.push(&[]);
    print_tokens(stt, toks, &mut emitted);

    // After the clip, stream the same trailing silence the OFFLINE path
    // right-pads with (align + delay+1+10 tokens) — the delayed tokens
    // catch up and the transcript is comparable to the offline oracle.
    let n_delay = args.delay_ms / 80;
    let align = (SAMPLES_PER_TOK - (audio.len() % SAMPLES_PER_TOK)) % SAMPLES_PER_TOK;
    let tail = align + (n_delay + 1 + OFFLINE_BUFFER_TOKENS) * SAMPLES_PER_TOK + N_FFT / 2;
    let mut feed: Vec<f32> = Vec::with_capacity(audio.len() + tail);
    feed.extend_from_slice(audio);
    feed.extend(std::iter::repeat(0f32).take(tail));

    let wall0 = std::time::Instant::now();
    let mut fed = 0usize;
    while fed < feed.len() && !stream.is_finished() {
        let end = (fed + chunk).min(feed.len());
        if !args.fast {
            // pace: chunk i may not be fed before wall time i*chunk_ms
            let due = wall0 + std::time::Duration::from_millis((fed / chunk * chunk_ms) as u64);
            if let Some(wait) = due.checked_duration_since(std::time::Instant::now()) {
                std::thread::sleep(wait);
            }
        }
        let toks = stream.push(&feed[fed..end]);
        print_tokens(stt, toks, &mut emitted);
        fed = end;
    }
    println!();
    report(&emitted, args.delay_ms);

    if let Some(expect) = &args.expect_text {
        let want = std::fs::read_to_string(expect)?;
        let got = stream.text();
        let (wa, hits, total) = word_accuracy(&got, &want);
        println!(
            "[listen] word match vs {expect}: {hits}/{total} = {wa:.1}% \
             (online mode has no right-pad; tail words may differ from the offline oracle)"
        );
        println!("[listen] ours:   {got:?}");
        println!("[listen] oracle: {:?}", want.trim());
    }
    Ok(())
}

fn report(emitted: &[StreamedToken], delay_ms: usize) {
    if emitted.is_empty() {
        eprintln!("[listen] no tokens emitted");
        return;
    }
    let mut lat: Vec<f32> = emitted.iter().map(|t| t.latency_ms).collect();
    lat.sort_by(f32::total_cmp);
    let mean = lat.iter().sum::<f32>() / lat.len() as f32;
    eprintln!(
        "[listen] {} tokens; compute latency ms/token: mean {mean:.0}, p50 {:.0}, p95 {:.0}, max {:.0}",
        emitted.len(),
        lat[lat.len() / 2],
        lat[lat.len() * 95 / 100],
        lat[lat.len() - 1],
    );
    // per-frame compute: enc + dec wall per token, prefill (first emission)
    // reported separately — it amortizes over the whole prompt.
    if emitted.len() > 1 {
        let frames: Vec<f32> = emitted[1..].iter().map(|t| t.enc_ms + t.dec_ms).collect();
        let mut s = frames.clone();
        s.sort_by(f32::total_cmp);
        let mean = frames.iter().sum::<f32>() / frames.len() as f32;
        let enc_mean = emitted[1..].iter().map(|t| t.enc_ms).sum::<f32>() / frames.len() as f32;
        let dec_mean = emitted[1..].iter().map(|t| t.dec_ms).sum::<f32>() / frames.len() as f32;
        eprintln!(
            "[listen] compute ms/frame ({} frames): mean {mean:.1}, p50 {:.1}, p95 {:.1}, max {:.1} \
             (enc mean {enc_mean:.1} + dec mean {dec_mean:.1}); prefill {:.0} ms; 80 ms budget",
            frames.len(),
            s[s.len() / 2],
            s[s.len() * 95 / 100],
            s[s.len() - 1],
            emitted[0].enc_ms + emitted[0].dec_ms,
        );
    }
    eprintln!(
        "[listen] perceived delay ≈ conditioned {delay_ms} ms + compute p50 {:.0} ms; \
         realtime requires compute < 80 ms/token sustained",
        lat[lat.len() / 2]
    );
}

/// Bag-of-words overlap in order (LCS over whitespace words), as % of oracle.
fn word_accuracy(got: &str, want: &str) -> (f32, usize, usize) {
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
    let hits = dp[n * (m + 1) + m];
    (100.0 * hits as f32 / m.max(1) as f32, hits, m)
}

/// Live mic capture: default (or name-matched) cpal input device, converted
/// to 16 kHz mono f32 and fed through the same StreamingTranscriber.
#[cfg(feature = "listen")]
fn mic_mode<B: burn::prelude::Backend, O: SttPipeline<B>>(
    stt: &O,
    delay_ms: usize,
    device_name: Option<String>,
) -> anyhow::Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = match &device_name {
        Some(name) => host
            .input_devices()?
            .find(|d| d.name().map(|n| n.contains(name.as_str())).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("no input device matching {name:?}"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("no default input device"))?,
    };
    let config = device.default_input_config()?;
    let in_rate = config.sample_rate() as usize;
    let channels = config.channels() as usize;
    eprintln!(
        "[listen] mic: {} @ {in_rate} Hz × {channels}ch → 16 kHz mono; delay {delay_ms} ms; ctrl-c to stop",
        device.name().unwrap_or_default()
    );

    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _| {
            // downmix to mono on the audio thread; resample on the main thread
            let mono: Vec<f32> = data
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect();
            tx.send(mono).ok();
        },
        |e| eprintln!("[listen] input stream error: {e}"),
        None,
    )?;
    stream.play()?;

    let mut ears_stream = StreamingTranscriber::new(stt, delay_ms);
    let mut resample_pos = 0f64;
    let ratio = in_rate as f64 / SAMPLE_RATE as f64;
    let mut hist: Vec<f32> = Vec::new(); // device-rate history for interpolation
    loop {
        let block = rx.recv()?;
        hist.extend_from_slice(&block);
        // linear-interpolation resample to 16 kHz
        let mut out = Vec::new();
        while (resample_pos + ratio) < (hist.len() - 1) as f64 {
            let i = resample_pos as usize;
            let frac = (resample_pos - i as f64) as f32;
            out.push(hist[i] * (1.0 - frac) + hist[i + 1] * frac);
            resample_pos += ratio;
        }
        // drop consumed history (keep one sample of overlap)
        let consumed = resample_pos as usize;
        if consumed > 1 {
            hist.drain(..consumed - 1);
            resample_pos -= (consumed - 1) as f64;
        }
        for t in ears_stream.push(&out) {
            let piece = stt.tekken().decode(&[t.id]);
            print!("{piece}");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        if ears_stream.is_finished() {
            break;
        }
    }
    println!();
    Ok(())
}
