//! Qwen3-TTS end-to-end: clone a reference voice from `ref_voice.wav` and speak an
//! arbitrary line. Reports the realtime factor (audio seconds per wall second).
//!
//!   cargo run --release --features qwen3tts --bin qwen3tts_say -- \
//!     [--out /tmp/out.wav] [--seed 0] [--greedy] [--f16] \
//!     [--ref clip.wav --ref-text "…"] [text…]
//!
//! `--f16` runs the talker in half precision — the CPU code predictor and
//! the codec stay f32. Fully self-contained voice cloning: the reference
//! clip is encoded in-process (CPU codec encoder + GPU ECAPA x-vector);
//! `--ref`/`--ref-text` clone any 24 kHz mono clip, default is `ref_voice.wav`.
//!
//! Weights come ONLY from the durable qwen3tts pile (both checkpoints, written
//! by `qwen3tts_persist`); `QWEN3TTS_PILE` overrides the default path. So
//! does the text tokenizer: the `vocab.json` and `merges.txt` documents the
//! importer names on the base roots.

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
use mary::nn::backend::{BFused, BFusedHalf};
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

const REF_WAV: &str = "ref_voice.wav";
// Transcript of `ref_voice.wav` — set this (or pass --ref-text) to match your clip.
const REF_TEXT: &str = "The tide rolls in across the flat sand, and the evening light settles slowly over the harbor as the last boats come home.";

struct Args {
    out: String,
    seed: u64,
    greedy: bool,
    f16: bool,
    ref_wav: String,
    ref_text: String,
    text: String,
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
    // the codec stays f32 even under --f16: its im2col conv GEMMs measured
    // *slower* in f16 (0.5 s → 1.2 s per utterance), and it is cheap in f32
    let codec_dev = mary::nn::backend::WgpuDevice::default();
    let codec = CodecDecoder::<BFused>::load(&loader, &codec_dev);
    let encoder = CodecEncoder::load(&loader);
    let tok = TextTokenizer::from_model_pile(&pile)
        .unwrap_or_else(|e| panic!("tokenizer from qwen3tts pile {}: {e:?}", pile.display()));
    eprintln!("weights loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // clone prompt, fully in-process (CPU codec encoder + GPU x-vector)
    let t_ref = Instant::now();
    let prompt = ClonePrompt::from_reference(
        &encoder,
        &spk_enc,
        &tok,
        Path::new(&args.ref_wav),
        &args.ref_text,
        &dev,
    );
    eprintln!(
        "reference encoded in {:.1}s ({} frames)",
        t_ref.elapsed().as_secs_f64(),
        prompt.ref_code.len()
    );

    let text = &args.text;
    let text_ids = tok.encode(&format!(
        "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    ));

    let t1 = Instant::now();
    let (prefill, trailing, tts_pad) = pipeline::build_prefill(
        &talker,
        &predictor,
        &prompt,
        &text_ids,
        Some(LANG_ENGLISH),
        &dev,
    );
    let params = SamplingParams {
        do_sample: !args.greedy,
        subtalker_do_sample: !args.greedy,
        ..Default::default()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let frames = pipeline::generate(
        &talker, &predictor, prefill, trailing, tts_pad, &params, &mut rng, &dev,
    );
    let t_gen = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let wav_out = pipeline::decode_with_ref(&codec, &prompt.ref_code, &frames, &codec_dev);
    let t_dec = t2.elapsed().as_secs_f64();

    let audio_s = wav_out.len() as f64 / SAMPLE_RATE as f64;
    wav::write_pcm16_mono(Path::new(&args.out), &wav_out, SAMPLE_RATE);
    eprintln!(
        "{}{} frames → {:.1}s audio | talker {:.1}s ({:.2}x rt) + codec {:.1}s | total {:.2}x realtime | {}",
        if args.f16 { "[f16] " } else { "" },
        frames.len(),
        audio_s,
        t_gen,
        audio_s / t_gen,
        t_dec,
        audio_s / (t_gen + t_dec),
        args.out
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args {
        out: "/tmp/mary-qwen3tts/port_sample.wav".to_string(),
        seed: 0,
        greedy: false,
        f16: false,
        ref_wav: REF_WAV.to_string(),
        ref_text: REF_TEXT.to_string(),
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
            "--greedy" => {
                args.greedy = true;
                i += 1;
            }
            "--f16" => {
                args.f16 = true;
                i += 1;
            }
            "--ref" => {
                args.ref_wav = argv[i + 1].clone();
                i += 2;
            }
            "--ref-text" => {
                args.ref_text = argv[i + 1].clone();
                i += 2;
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

    if args.f16 {
        run::<BFusedHalf>(&args);
    } else {
        run::<BFused>(&args);
    }
}
