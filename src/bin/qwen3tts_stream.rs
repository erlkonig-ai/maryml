//! Qwen3-TTS **streaming** synthesis: emit audio while the model is still
//! generating. Frames go to a codec thread the moment they are sampled; the
//! codec decodes hop-sized chunks with a left-context window (the reference
//! implementation's own `chunked_decode` primitive, smaller chunks) and emits
//! 24 kHz PCM16-LE to stdout as each chunk is ready. Codec (GPU, f32) and
//! generation (GPU f16 talker + CPU predictor) overlap — the codec fits in
//! the GPU-idle window while the CPU predictor works.
//!
//!   cargo run --release --features qwen3tts --bin qwen3tts_stream -- \
//!     [--out /tmp/stream.wav] [--seed 0] [--f32] [--hop 8] [--ctx 25] \
//!     [--pcm] [text…]
//!
//! Weights come ONLY from the durable qwen3tts pile (written by
//! `qwen3tts_persist`); `QWEN3TTS_PILE` overrides the default path. So does
//! the text tokenizer: the `vocab.json` and `merges.txt` documents the
//! importer names on the base roots.
//!
//! stderr reports TTFA (time to first audio), per-chunk deadline margins
//! against a playback clock that starts at first-chunk-ready, and the
//! sustained realtime factor. `--pcm` streams raw 24 kHz mono PCM16-LE to
//! stdout as chunks become ready (pipe it into an audio sink).
//!
//! Voice-conditioning note: generated frames are decoded with the trailing
//! `ctx` reference/generated codes as left context (the batch path gives the
//! first 300 frames the full reference); the codec's sliding window is 72
//! frames, so `--ctx 72` trades ~3× hop decode cost for full-window context.

use burn::prelude::*;
use mary::models::f5::wav;
use mary::models::qwen3tts::codec::CodecDecoder;
use mary::models::qwen3tts::config::*;
use mary::models::qwen3tts::encoder::CodecEncoder;
use mary::models::qwen3tts::pipeline::{self, ClonePrompt, SamplingParams};
use mary::models::qwen3tts::predictor::CodePredictor;
use mary::models::qwen3tts::speaker::SpeakerEncoder;
use mary::models::qwen3tts::talker::Talker;
use mary::models::qwen3tts::tokenizer::TextTokenizer;
use mary::nn::backend::{BFused, BFusedHalf, WgpuDevice};
use rand::SeedableRng;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

const REF_WAV: &str = "ref_voice.wav";
// Transcript of `ref_voice.wav` — set to match your reference clip.
const REF_TEXT: &str = "The tide rolls in across the flat sand, and the evening light settles slowly over the harbor as the last boats come home.";

struct Args {
    out: String,
    seed: u64,
    f32_: bool,
    hop: usize,
    ctx: usize,
    pcm: bool,
    text: String,
}

/// What the codec thread reports back.
struct StreamStats {
    samples: Vec<f32>,
    ttfa: f64,
    min_margin: f64,
    underruns: usize,
    chunks: usize,
}

fn run<B: Backend>(args: &Args) {
    mary::models::qwen3tts::cpu::set_interactive_qos();
    let dev: B::Device = Default::default();
    let t0 = Instant::now();
    let pile = mary::paths::model(
        std::env::var("QWEN3TTS_PILE").ok().as_deref(),
        "qwen3tts.pile",
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let loader = mary::persist::load_aliased_loader_from_pile(&pile, "talker_f16")
        .unwrap_or_else(|e| panic!("load qwen3tts pile {}: {e:?}", pile.display()));
    let talker = Talker::<B>::load(&loader, &dev);
    let predictor = CodePredictor::load(&loader);
    let spk_enc = SpeakerEncoder::<B>::load(&loader, &dev);
    let encoder = CodecEncoder::load(&loader);
    let tok = TextTokenizer::from_model_pile(&pile)
        .unwrap_or_else(|e| panic!("tokenizer from qwen3tts pile {}: {e:?}", pile.display()));
    eprintln!("weights loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // clone prompt, fully in-process (CPU codec encoder + GPU x-vector)
    let prompt =
        ClonePrompt::from_reference(&encoder, &spk_enc, &tok, Path::new(REF_WAV), REF_TEXT, &dev);
    let text = &args.text;
    let text_ids = tok.encode(&format!(
        "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    ));

    // ── codec thread: consumes frames, decodes hop-chunks, emits PCM ──
    let (tx, rx) = mpsc::channel::<[u32; NUM_CODE_GROUPS]>();
    let (hop, ctx, pcm) = (args.hop, args.ctx, args.pcm);
    let ref_codes = prompt.ref_code.clone();
    // t_start is set when generation begins (after prefill assembly starts);
    // TTFA is measured from there — model load excluded, prefill included.
    let t_start = Instant::now();
    let codec_thread = std::thread::spawn(move || {
        mary::models::qwen3tts::cpu::set_interactive_qos();
        let cdev = WgpuDevice::default();
        // the main thread is done with the pile keymap — it moves in here
        let codec = CodecDecoder::<BFused>::load(&loader, &cdev);
        // warm the decode path at the steady-state chunk shape (shader
        // compile + autotune land here, not in the first real chunk — the
        // first decode was measured ~0.8 s cold vs ~40 ms warm)
        let _ = codec.decode(&vec![[0u32; NUM_CODE_GROUPS]; ctx + hop], &cdev);
        let mut stdout = std::io::stdout();
        let emit_pcm = pcm && !std::io::stdout().is_terminal();

        // history = ref codes ++ generated codes; decode windows slide over it
        let mut history = ref_codes;
        let mut decoded_upto = history.len(); // frame index of first undecoded frame
        let mut out_samples: Vec<f32> = Vec::new();
        let mut ttfa = 0f64;
        let (mut min_margin, mut underruns, mut chunks) = (f64::MAX, 0usize, 0usize);

        let mut flush = |history: &[[u32; NUM_CODE_GROUPS]],
                         from: usize,
                         to: usize,
                         out_samples: &mut Vec<f32>,
                         ttfa: &mut f64,
                         min_margin: &mut f64,
                         underruns: &mut usize,
                         chunks: &mut usize| {
            let c = ctx.min(from);
            let td = Instant::now();
            let wav = codec.decode(&history[from - c..to], &cdev);
            let pcm = &wav[c * SAMPLES_PER_FRAME..];
            let ready = t_start.elapsed().as_secs_f64();
            if *chunks == 0 {
                *ttfa = ready;
                eprintln!(
                    "TTFA: {:.2}s (decode {:.0}ms)",
                    ready,
                    td.elapsed().as_secs_f64() * 1e3
                );
            } else {
                // playback clock starts at TTFA; this chunk's audio starts at
                // its first sample's position in the output stream
                let audio_pos = out_samples.len() as f64 / SAMPLE_RATE as f64;
                let margin = *ttfa + audio_pos - ready;
                *min_margin = min_margin.min(margin);
                if margin < 0.0 {
                    *underruns += 1;
                }
                eprintln!(
                    "chunk {:3}: frames {}..{} ready {:.2}s margin {:+.2}s (decode {:.0}ms)",
                    chunks,
                    from,
                    to,
                    ready,
                    margin,
                    td.elapsed().as_secs_f64() * 1e3
                );
            }
            *chunks += 1;
            if emit_pcm {
                let bytes: Vec<u8> = pcm
                    .iter()
                    .flat_map(|&s| (((s.clamp(-1.0, 1.0)) * 32767.0) as i16).to_le_bytes())
                    .collect();
                let _ = stdout.write_all(&bytes);
                let _ = stdout.flush();
            }
            out_samples.extend_from_slice(pcm);
        };

        while let Ok(frame) = rx.recv() {
            history.push(frame);
            if history.len() - decoded_upto >= hop {
                let (from, to) = (decoded_upto, history.len());
                flush(
                    &history,
                    from,
                    to,
                    &mut out_samples,
                    &mut ttfa,
                    &mut min_margin,
                    &mut underruns,
                    &mut chunks,
                );
                decoded_upto = to;
            }
        }
        // final partial chunk
        if history.len() > decoded_upto {
            let (from, to) = (decoded_upto, history.len());
            flush(
                &history,
                from,
                to,
                &mut out_samples,
                &mut ttfa,
                &mut min_margin,
                &mut underruns,
                &mut chunks,
            );
        }
        StreamStats {
            samples: out_samples,
            ttfa,
            min_margin: if min_margin == f64::MAX {
                0.0
            } else {
                min_margin
            },
            underruns,
            chunks,
        }
    });

    // ── generation (main thread) ──
    let (prefill, trailing, tts_pad) = pipeline::build_prefill(
        &talker,
        &predictor,
        &prompt,
        &text_ids,
        Some(LANG_ENGLISH),
        &dev,
    );
    let params = SamplingParams::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let frames = pipeline::generate_streaming(
        &talker,
        &predictor,
        prefill,
        trailing,
        tts_pad,
        &params,
        &mut rng,
        &dev,
        |f| tx.send(*f).is_ok(),
    );
    drop(tx);
    let t_gen = t_start.elapsed().as_secs_f64();
    let stats = codec_thread.join().expect("codec thread");
    let t_total = t_start.elapsed().as_secs_f64();

    let audio_s = stats.samples.len() as f64 / SAMPLE_RATE as f64;
    wav::write_pcm16_mono(Path::new(&args.out), &stats.samples, SAMPLE_RATE);
    eprintln!(
        "{}{} frames → {:.1}s audio in {:.1}s wall ({:.2}x rt, gen {:.2}x) | TTFA {:.2}s | {} chunks, min margin {:+.2}s, {} underruns | {}",
        if args.f32_ { "[f32] " } else { "[f16] " },
        frames.len(),
        audio_s,
        t_total,
        audio_s / t_total,
        audio_s / t_gen,
        stats.ttfa,
        stats.chunks,
        stats.min_margin,
        stats.underruns,
        args.out
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args {
        out: "/tmp/mary-qwen3tts/stream_sample.wav".to_string(),
        seed: 0,
        f32_: false,
        hop: 8,
        ctx: 25,
        pcm: false,
        text: String::new(),
    };
    let mut words = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--out" => {
                args.out = argv[i + 1].clone();
                i += 2;
            }
            "--seed" => {
                args.seed = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--hop" => {
                args.hop = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--ctx" => {
                args.ctx = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--f32" => {
                args.f32_ = true;
                i += 1;
            }
            "--pcm" => {
                args.pcm = true;
                i += 1;
            }
            w => {
                words.push(w.to_string());
                i += 1;
            }
        }
    }
    args.text = if words.is_empty() {
        "If you can hear this clearly, the port worked: the same reference voice, synthesized end to end by the new engine in real time.".to_string()
    } else {
        words.join(" ")
    };

    if args.f32_ {
        run::<BFused>(&args);
    } else {
        run::<BFusedHalf>(&args);
    }
}
