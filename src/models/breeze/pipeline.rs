//! Single-request native Breeze composition. The initial acoustic endpoint is
//! batch decoding, not a stateful/realtime stream and not Qwen clone synthesis.
use anyhow::{Result, ensure};
use burn::prelude::Backend;
use std::{path::Path, time::Instant};

use super::{
    audio::decode_generated,
    generator::{GenerationOptions, GenerationReport, Generator, PromptSegment},
    prompt::BreezeTokenizer,
    reference::{encode_reference, read_wav},
};
use crate::models::qwen3tts::{codec::CodecDecoder, encoder::CodecEncoder};

pub struct PreparedReference {
    pub prompt: Vec<PromptSegment>,
    pub reference_samples: usize,
    pub reference_frames: usize,
    pub wav_read_seconds: f64,
    pub reference_encode_seconds: f64,
    pub prompt_seconds: f64,
}

/// The caller separately times encoder weight loading. Reference processing is
/// native CPU work, once per request; no precomputed oracle reference is read.
pub fn prepare_reference(
    encoder: &CodecEncoder,
    tokenizer: &BreezeTokenizer,
    reference_wav: &Path,
    reference_text: &str,
    text: &str,
    direction: Option<&str>,
) -> Result<PreparedReference> {
    let started = Instant::now();
    let samples = read_wav(reference_wav)?;
    let wav_read_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let codes = encode_reference(encoder, &samples)?;
    let reference_encode_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let prompt = tokenizer.reference_prompt(reference_text, &codes, text, direction)?;
    Ok(PreparedReference {
        prompt,
        reference_samples: samples.len(),
        reference_frames: codes.len(),
        wav_read_seconds,
        reference_encode_seconds,
        prompt_seconds: started.elapsed().as_secs_f64(),
    })
}

pub struct Synthesis {
    pub samples: Vec<f32>,
    pub frames: Vec<[u16; 16]>,
    pub generation: GenerationReport,
    pub codec_decode_seconds: f64,
}

/// Collect only generated complete frames, then decode from fresh acoustic
/// state. Reference codes condition the LM but are never an output prefix.
pub fn run<B: Backend>(
    generator: &mut Generator,
    prompt: &[PromptSegment],
    decoder: &CodecDecoder<B>,
    device: &B::Device,
    options: GenerationOptions,
) -> Result<Synthesis> {
    let limit = options.max_frames;
    ensure!(limit > 0, "positive frame limit required");
    let mut frames = Vec::new();
    let generation = generator.generate(prompt, options, |frame| {
        ensure!(
            frames.len() < limit,
            "generator exceeded requested frame limit"
        );
        ensure!(
            frame.iter().all(|&code| code < 2048),
            "generator emitted a non-speech frame"
        );
        frames.push(frame);
        Ok(())
    })?;
    generator.synchronize()?;
    ensure!(
        generation.frames == frames.len(),
        "generator frame receipt mismatch"
    );
    let started = Instant::now();
    let samples = decode_generated(decoder, &frames, device)?;
    Ok(Synthesis {
        samples,
        frames,
        generation,
        codec_decode_seconds: started.elapsed().as_secs_f64(),
    })
}
