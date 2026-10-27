//! Qwen3-TTS parity gates — Burn port vs the CPU-f32 oracle goldens captured
//! by `golden/capture.py` (in the port worktree):
//!
//!   1. tokenizer      exact-id match on three test strings
//!   2. speaker mel    cos vs `spk_mel`
//!   3. speaker enc    cos vs `ref_spk_embedding`
//!   4. prefill build  cos vs `prefill_embeds` (prompt assembly end-to-end)
//!   5. talker         cos vs `prefill_hidden` + last-pos logits
//!   6. greedy frames  token match on the first frames (talker+predictor loop)
//!   7. codec          cos of decoded waveform vs `codec_refcodes_wav`
//!
//!   cargo run --release --features qwen3tts --bin qwen3tts_probe [-- --f16]
//!
//! Under `--f16` the GPU stages run in half precision: the cos gates relax to
//! 0.999 (the f16 port contract) and the greedy-frame check becomes
//! informational — autoregressive divergence cascades after the first
//! rounding-flipped token, so "leading identical frames" is the honest
//! metric there, not a hard gate.

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::models::f5::wav;
use mary::models::qwen3tts::codec::CodecDecoder;
use mary::models::qwen3tts::config::*;
use mary::models::qwen3tts::encoder::CodecEncoder;
use mary::models::qwen3tts::pipeline::{self, ClonePrompt, SamplingParams};
use mary::models::qwen3tts::predictor::CodePredictor;
use mary::models::qwen3tts::speaker::{SpeakerEncoder, SpeakerMel};
use mary::models::qwen3tts::talker::Talker;
use mary::models::qwen3tts::tokenizer::TextTokenizer;
use mary::nn::backend::{BFused, BFusedHalf};
use mary::nn::npy;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use rand::SeedableRng;
use std::path::Path;

const WEIGHTS: &str = "/tmp/qwen3tts-weights/base";
const GOLD: &str = "/tmp/mary-qwen3tts/golden";
const REF_WAV: &str = "ref_voice.wav";

fn golden(name: &str) -> (Vec<f32>, Vec<usize>) {
    npy::load_npy(&Path::new(GOLD).join(format!("{name}.npy")))
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
        "  {} {name:22} cos={cos:.8}  max|Δ|={maxabs:.3e}",
        if ok { "✓" } else { "✗" }
    );
    ok
}

fn to_f32<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

fn run<B: Backend>(f16: bool) {
    let dev: B::Device = Default::default();
    let mut ok = true;
    // f32 runs use the exactness thresholds; f16 the port-contract 0.999.
    let (tight, loose) = if f16 { (0.999, 0.999) } else { (0.9999, 0.999) };

    // 1. tokenizer
    let tok = TextTokenizer::load(Path::new(WEIGHTS)).expect("checkpoint tokenizer");
    let (tests, tshape) = golden("tokenizer_tests");
    let strings = [
        "<|im_start|>assistant\nIf you can hear this clearly, the port worked: the same reference voice, synthesized end to end by the new engine in real time.<|im_end|>\n<|im_start|>assistant\n",
        "Numbers 12345 and punctuation!!! Don't worry — ümlauts, 中文 too.",
        "A line\nwith newlines\n\nand   spaces.",
    ];
    for (i, s) in strings.iter().enumerate() {
        let want: Vec<u32> = tests[i * tshape[1]..(i + 1) * tshape[1]]
            .iter()
            .filter(|&&v| v >= 0.0)
            .map(|&v| v as u32)
            .collect();
        let got = tok.encode(s);
        let m = got == want;
        ok &= m;
        println!(
            "  {} tokenizer[{i}]           {} tokens",
            if m { "✓" } else { "✗" },
            got.len()
        );
        if !m {
            println!("    want {:?}", &want[..want.len().min(20)]);
            println!("    got  {:?}", &got[..got.len().min(20)]);
        }
    }

    // 2+3. speaker mel + encoder
    let base = WeightLoader::SingleFile(SingleFileLoader::new(
        &Path::new(WEIGHTS).join("model.safetensors"),
    ));
    let (samples, sr) = wav::read_pcm16_mono(Path::new(REF_WAV));
    assert_eq!(sr, 24000);
    let mel_x = SpeakerMel::<B>::new(&dev);
    let mel = mel_x.forward(&samples, &dev);
    let (gmel, _) = golden("spk_mel");
    ok &= metrics("speaker mel", &to_f32(mel.clone()), &gmel, tight);

    let spk = SpeakerEncoder::<B>::load(&base, &dev);
    let emb = spk.forward(mel);
    let (gemb, _) = golden("ref_spk_embedding");
    ok &= metrics("speaker embedding", &to_f32(emb.clone()), &gemb, loose);

    // 4. prefill assembly
    println!("loading talker + predictor…");
    let talker = Talker::<B>::load(&base, &dev);
    let predictor = CodePredictor::load(&base);

    let (rc, rcs) = golden("ref_code_f32");
    let ref_code: Vec<[u32; NUM_CODE_GROUPS]> = (0..rcs[0])
        .map(|t| {
            let mut f = [0u32; NUM_CODE_GROUPS];
            for q in 0..NUM_CODE_GROUPS {
                f[q] = rc[t * NUM_CODE_GROUPS + q] as u32;
            }
            f
        })
        .collect();
    let (ri, _) = golden("ref_ids_f32");
    let ref_ids: Vec<u32> = ri.iter().map(|&v| v as u32).collect();
    let (ti, _) = golden("text_ids_f32");
    let text_ids: Vec<u32> = ti.iter().map(|&v| v as u32).collect();

    // cross-check our tokenizer against the oracle's ids for the exact prompts
    let (meta_ref, meta_text) = (
        "<|im_start|>assistant\nThe tide rolls in across the flat sand, and the evening light settles slowly over the harbor as the last boats come home.<|im_end|>\n",
        "<|im_start|>assistant\nIf you can hear this clearly, the port worked: the same reference voice, synthesized end to end by the new engine in real time.<|im_end|>\n<|im_start|>assistant\n",
    );
    let m = tok.encode(meta_ref) == ref_ids && tok.encode(meta_text) == text_ids;
    ok &= m;
    println!("  {} prompt tokenization", if m { "✓" } else { "✗" });

    let prompt = ClonePrompt {
        ref_code,
        ref_ids,
        spk_embedding: emb,
    };
    let (prefill, trailing, tts_pad) = pipeline::build_prefill(
        &talker,
        &predictor,
        &prompt,
        &text_ids,
        Some(LANG_ENGLISH),
        &dev,
    );
    let (gpre, gpres) = golden("prefill_embeds");
    assert_eq!(prefill.dims()[1], gpres[1], "prefill length");
    ok &= metrics("prefill embeds", &to_f32(prefill.clone()), &gpre, tight);

    // 5. talker prefill parity — drive with the GOLDEN embeds so this gates the
    // 28-layer stack in isolation, then with our own embeds end-to-end.
    let gpre_t = Tensor::<B, 3>::from_data(TensorData::new(gpre.clone(), gpres.clone()), &dev);
    let mut caches = talker.new_caches();
    let hidden = talker.forward(gpre_t, &mut caches, &dev);
    let (gh, _) = golden("prefill_hidden");
    ok &= metrics("talker hidden", &to_f32(hidden.clone()), &gh, loose);
    let logits = talker.logits_last(hidden.clone());
    let (gl, _) = golden("prefill_logits");
    ok &= metrics("prefill logits", &logits, &gl, loose);

    // 6. greedy loop parity — first N frames token-identical (informational
    // under f16, see module doc)
    let (gc, gcs) = golden("greedy_codes_f32");
    let params = SamplingParams {
        do_sample: false,
        subtalker_do_sample: false,
        max_frames: 20,
        ..Default::default()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let frames = pipeline::generate(
        &talker, &predictor, prefill, trailing, tts_pad, &params, &mut rng, &dev,
    );
    let n = frames.len().min(20);
    let mut mism = 0;
    let mut first_div: Option<usize> = None;
    for t in 0..n {
        for q in 0..NUM_CODE_GROUPS {
            if frames[t][q] != gc[t * gcs[1] + q] as u32 {
                mism += 1;
                first_div.get_or_insert(t);
            }
        }
    }
    let m = mism == 0;
    if !f16 {
        ok &= m;
    }
    println!(
        "  {} greedy frames         {}/{} tokens match over {n} frames{}",
        if m {
            "✓"
        } else if f16 {
            "·"
        } else {
            "✗"
        },
        n * NUM_CODE_GROUPS - mism,
        n * NUM_CODE_GROUPS,
        first_div
            .map(|t| format!(" (first divergence at frame {t})"))
            .unwrap_or_default()
    );
    if !m && !f16 {
        for t in 0..n.min(3) {
            println!("    frame {t}: got {:?}", frames[t]);
            let want: Vec<u32> = (0..NUM_CODE_GROUPS)
                .map(|q| gc[t * gcs[1] + q] as u32)
                .collect();
            println!("    frame {t}: want {want:?}");
        }
    }

    // 7. codec decode parity on the reference codes
    println!("loading codec decoder…");
    let codec_loader = WeightLoader::SingleFile(SingleFileLoader::new(
        &Path::new(WEIGHTS).join("speech_tokenizer/model.safetensors"),
    ));
    let codec = CodecDecoder::<B>::load(&codec_loader, &dev);
    let q = codec.quantizer_decode(&prompt.ref_code, &dev);
    let (gq, _) = golden("codec_quantized");
    ok &= metrics("codec quantizer", &to_f32(q), &gq, tight);
    let wav_out = codec.decode(&prompt.ref_code, &dev);
    let (gwav, _) = golden("codec_refcodes_wav");
    ok &= metrics("codec waveform", &wav_out, &gwav, 0.99);
    wav::write_pcm16_mono(
        Path::new("/tmp/mary-qwen3tts/codec_parity.wav"),
        &wav_out,
        SAMPLE_RATE,
    );

    // 8. codec ENCODER (CPU) — encode ref_voice.wav, exact-match the oracle's
    // captured codes (backend-independent, gated in both modes)
    println!("loading codec encoder…");
    let enc = CodecEncoder::load(&codec_loader);
    let enc_codes = enc.encode(&samples);
    let m = enc_codes.len() == prompt.ref_code.len();
    let mut code_mism = 0;
    let mut per_q = [0usize; NUM_CODE_GROUPS];
    for (t, (a, b)) in enc_codes.iter().zip(&prompt.ref_code).enumerate() {
        for q in 0..NUM_CODE_GROUPS {
            if a[q] != b[q] {
                code_mism += 1;
                per_q[q] += 1;
                if code_mism <= 5 {
                    println!("    frame {t} q{q}: got {} want {}", a[q], b[q]);
                }
            }
        }
    }
    if code_mism > 0 {
        println!("    per-quantizer mismatches: {per_q:?}");
    }
    let total = enc_codes.len() * NUM_CODE_GROUPS;
    let em = m && code_mism == 0;
    ok &= em;
    println!(
        "  {} codec encoder         {}/{} ref codes match ({} frames vs {})",
        if em { "✓" } else { "✗" },
        total - code_mism,
        total,
        enc_codes.len(),
        prompt.ref_code.len()
    );

    println!(
        "{}",
        if ok {
            "ALL GATES PASSED"
        } else {
            "GATES FAILED"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}

fn main() {
    if std::env::args().any(|a| a == "--f16") {
        println!("== f16 (GPU stages in half precision) ==");
        run::<BFusedHalf>(true);
    } else {
        run::<BFused>(false);
    }
}
