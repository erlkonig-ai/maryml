//! Silent gate harness for the `mary::speak` seam: everything `voice say/shout`
//! does EXCEPT playback — load the components from the persisted pile, compute the
//! reference x-vector (optionally dumped for cross-checking against the
//! reference implementation), and synthesize a line to a WAV on disk. Gates run
//! on files (resemblyzer / whisper), never through an audio device.
//!
//!   cargo run --release --features speak --bin speak_check -- \
//!     <pile> <ref_wav> <ref_text_file> <ref_codes.npy> \
//!     [--xvec-out xv.npy] [--out out.wav] [--text "line to speak"]
//!
//! Without `--out` it stops after the x-vector (cheap conditioning check).

use anyhow::Context;
use mary::models::f5::wav;
use mary::models::qwen3tts::speaker::{SpeakerEncoder, SpeakerMel};
use mary::nn::backend::BFused as B;
use mary::nn::npy;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let (mut xvec_out, mut out, mut text, mut tok_check) = (None, None, None, None);
    let mut prefill_out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--xvec-out" => {
                xvec_out = Some(args[i + 1].clone());
                i += 2;
            }
            "--tok" => {
                tok_check = Some(args[i + 1].clone());
                i += 2;
            }
            "--prefill-out" => {
                prefill_out = Some(args[i + 1].clone());
                i += 2;
            }
            "--out" => {
                out = Some(args[i + 1].clone());
                i += 2;
            }
            "--text" => {
                text = Some(args[i + 1].clone());
                i += 2;
            }
            p => {
                pos.push(p.to_string());
                i += 1;
            }
        }
    }
    anyhow::ensure!(
        pos.len() == 4,
        "usage: speak_check <pile> <ref_wav> <ref_text_file> <ref_codes.npy> [--xvec-out p] [--out p] [--text s]"
    );
    let (pile, ref_wav, ref_text_file, ref_codes) = (&pos[0], &pos[1], &pos[2], &pos[3]);
    let ref_text = std::fs::read_to_string(ref_text_file)?.trim().to_string();

    if let Some(s) = &tok_check {
        use mary::models::qwen3tts::tokenizer::TextTokenizer;
        let ids = TextTokenizer::from_model_pile(Path::new(pile))?.encode(s);
        eprintln!("tok ({} ids): {:?}", ids.len(), ids);
    }

    if let Some(xp) = &xvec_out {
        let dev = Default::default();
        let loader = mary::persist::load_aliased_loader_from_pile(Path::new(pile), "talker_f16")?;
        let spk_enc = SpeakerEncoder::<B>::load(&loader, &dev);
        let (samples, sr) = wav::read_pcm16_mono(Path::new(ref_wav));
        eprintln!("ref wav: {} samples @ {sr}", samples.len());
        let xv = spk_enc.forward(SpeakerMel::<B>::new(&dev).forward(&samples, &dev));
        let v: Vec<f32> = xv.into_data().to_vec().unwrap();
        npy::save_npy(Path::new(xp), &v, &[v.len()])?;
        eprintln!("x-vector → {xp}");
    }

    if let Some(pp) = &prefill_out {
        use mary::models::qwen3tts::codec::CodecDecoder;
        use mary::models::qwen3tts::config::*;
        use mary::models::qwen3tts::pipeline::{self, ClonePrompt};
        use mary::models::qwen3tts::predictor::CodePredictor;
        use mary::models::qwen3tts::talker::Talker;
        use mary::models::qwen3tts::tokenizer::TextTokenizer;
        let dev = Default::default();
        let loader = mary::persist::load_aliased_loader_from_pile(Path::new(pile), "talker_f16")?;
        let talker = Talker::<B>::load(&loader, &dev);
        let predictor = CodePredictor::load(&loader);
        let spk_enc = SpeakerEncoder::<B>::load(&loader, &dev);
        let _codec = CodecDecoder::<B>::load(&loader, &dev);
        drop(loader);
        let tok = TextTokenizer::from_model_pile(Path::new(pile))?;
        let (samples, _sr) = wav::read_pcm16_mono(Path::new(ref_wav));
        let spk_embedding = spk_enc.forward(SpeakerMel::<B>::new(&dev).forward(&samples, &dev));
        let (rc, rcs) = npy::load_npy(Path::new(ref_codes))?;
        let ref_code: Vec<[u32; NUM_CODE_GROUPS]> = (0..rcs[0])
            .map(|t| {
                let mut f = [0u32; NUM_CODE_GROUPS];
                for q in 0..NUM_CODE_GROUPS {
                    f[q] = rc[t * NUM_CODE_GROUPS + q] as u32;
                }
                f
            })
            .collect();
        let prompt = ClonePrompt {
            ref_code,
            ref_ids: tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n")),
            spk_embedding,
        };
        let line = text
            .clone()
            .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".into());
        let text_ids = tok.encode(&format!(
            "<|im_start|>assistant\n{line}<|im_end|>\n<|im_start|>assistant\n"
        ));
        let (prefill, trailing, _pad) = pipeline::build_prefill(
            &talker,
            &predictor,
            &prompt,
            &text_ids,
            Some(LANG_ENGLISH),
            &dev,
        );
        let dims = prefill.dims();
        eprintln!("prefill dims {:?}, trailing {:?}", dims, trailing.dims());
        let v: Vec<f32> = prefill.into_data().to_vec().unwrap();
        npy::save_npy(Path::new(pp), &v, &[dims[1], dims[2]])?;
        eprintln!("prefill → {pp}");
    }

    if let Some(op) = &out {
        let line = text.unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".into());
        // `synthesize_to_wav` takes a frozen cohort, not a path — the same
        // atomic policy-admitted snapshot the `voice` faculty uses, so the gate
        // renders through the production selection rather than a drifting one.
        let variant = mary::speak::Qwen3TtsVariant::from_env();
        let snapshot = mary::model_collection::load_model_collection_local_latest(Path::new(pile))
            .with_context(|| format!("freeze native Qwen3-TTS snapshot {pile}"))?;
        let weights = mary::speak::Qwen3TtsWeights::from_snapshot(snapshot, variant)
            .with_context(|| format!("select {variant:?} Qwen3-TTS cohort from {pile}"))?;
        let n = mary::speak::synthesize_to_wav(
            weights,
            Path::new(ref_wav),
            &ref_text,
            Path::new(ref_codes),
            &line,
            Path::new(op),
        )?;
        eprintln!("{n} samples → {op}");
    }
    Ok(())
}
