//! Single-request native Breeze composition: [`run`] decodes the whole
//! utterance as one batch once generation ends, [`stream`] decodes it in
//! fixed-size hops while it is generated. Not Qwen clone synthesis.
use anyhow::{Result, ensure};
use burn::prelude::Backend;
use std::{path::Path, time::Instant};

use super::{
    audio::{HopWindow, Hops, decode_generated, decode_hop},
    generator::{GenerationOptions, GenerationReport, Generator},
    prompt::{BreezeTokenizer, GuidedPrompt},
    reference::{encode_reference, read_wav},
};
use crate::models::qwen3tts::{codec::CodecDecoder, encoder::CodecEncoder};

pub struct PreparedReference {
    pub prompt: GuidedPrompt,
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
    cfg_scale: f32,
) -> Result<PreparedReference> {
    let started = Instant::now();
    let samples = read_wav(reference_wav)?;
    let wav_read_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let codes = encode_reference(encoder, &samples)?;
    let reference_encode_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let prompt = tokenizer.reference_prompts(reference_text, &codes, text, direction, cfg_scale)?;
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
    prompt: &GuidedPrompt,
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

/// One utterance decoded in hops, and when its audio left.
pub struct Streamed {
    pub frames: Vec<[u16; 16]>,
    pub generation: GenerationReport,
    /// Wall seconds of each hop's decode in the order the hops left; a tail
    /// shorter than a hop is the last.
    pub hop_seconds: Vec<f64>,
    /// From the call to the first PCM handed on; `None` if none was.
    pub first_pcm_seconds: Option<f64>,
}

/// Generate and decode in hops ([`Hops`]): each time `HOP_FRAMES` new frames
/// are generated they are decoded on this thread, between two frames, and
/// handed to `on_pcm` at once; the frames left at the end go as a shorter
/// tail. The chunks in order are the utterance. An error from `on_pcm` stops
/// generation at that hop. Termination is the caller's to judge: the hops
/// already handed on are not taken back when generation ends at a cap.
pub fn stream<B: Backend>(
    generator: &mut Generator,
    prompt: &GuidedPrompt,
    decoder: &CodecDecoder<B>,
    device: &B::Device,
    options: GenerationOptions,
    mut on_pcm: impl FnMut(Vec<f32>) -> Result<()>,
) -> Result<Streamed> {
    let started = Instant::now();
    let limit = options.max_frames;
    ensure!(limit > 0, "positive frame limit required");
    let mut frames = Vec::new();
    let mut hops = Hops::default();
    let mut hop_seconds = Vec::new();
    let mut first_pcm_seconds = None;
    let mut emit = |window: HopWindow| -> Result<()> {
        let decoding = Instant::now();
        let pcm = decode_hop(decoder, &window, device)?;
        hop_seconds.push(decoding.elapsed().as_secs_f64());
        first_pcm_seconds.get_or_insert_with(|| started.elapsed().as_secs_f64());
        on_pcm(pcm)
    };
    let generation = generator.generate(prompt, options, |frame| {
        ensure!(
            frames.len() < limit,
            "generator exceeded requested frame limit"
        );
        frames.push(frame);
        match hops.push(frame)? {
            Some(window) => emit(window),
            None => Ok(()),
        }
    })?;
    generator.synchronize()?;
    ensure!(
        generation.frames == frames.len(),
        "generator frame receipt mismatch"
    );
    ensure!(!frames.is_empty(), "generation produced no speech frames");
    if let Some(window) = hops.flush() {
        emit(window)?;
    }
    Ok(Streamed {
        frames,
        generation,
        hop_seconds,
        first_pcm_seconds,
    })
}
