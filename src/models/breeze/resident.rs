//! Resident native Breeze weights and one natively encoded reference.
//!
//! One owning worker calls `load` once, then `synthesize_stream` for each line.
//! Output is generated-only PCM decoded in fixed-size hops while the line is
//! generated; each hop decodes its left context and new frames from fresh
//! codec state, and no decoder state is carried between hops. No request
//! can inject precomputed reference codes, a checkpoint path or a CPU LLM.
use anyhow::{Result, ensure};
use cubecl::cuda::CudaDevice;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
use triblespace::core::repo::BlobStoreGet;

use super::{
    audio::hop_window_lengths,
    generator::{GenerationOptions, Generator, Weights},
    load::{Artifacts, Assets},
    nvfp4::PromptLane,
    pipeline::{self, Streamed},
    prompt::BreezeTokenizer,
    reference::{decode_wav, encode_reference, read_wav_bytes},
    reference_cache::{ReferenceKey, encode_reference_cached},
};
use crate::{
    model_collection::ModelSnapshot,
    models::qwen3tts::{codec::CodecDecoder, config::SAMPLES_PER_FRAME, encoder::CodecEncoder},
    nn::{backend::speak::Fused, weight_loader::WeightLoader},
};

/// Explicit per-resident inputs. The model is not named here: it is the
/// Breeze root of the model collection the caller hands to
/// [`BreezeResident::load`], selected by its source revision and `quantization`
/// label, with its assets found by name under that root.
#[derive(Clone, Debug)]
pub struct BreezeVoiceConfig {
    /// The stored weight-format label to select (`persist::QUANTIZATION_NVFP4`).
    pub quantization: String,
    /// How projections stored as NVFP4 multiply prompt-height activations
    /// (`breeze::PROMPT_LANE` unless a comparison asks otherwise).
    pub lane: PromptLane,
    pub reference_wav: PathBuf,
    /// Known transcript, not an ASR request. Outer whitespace is stripped once.
    pub reference_text: String,
    /// Where the clip's encoded codes are kept between loads
    /// ([`super::reference_cache`]); `None` encodes on every load. The codes
    /// are keyed by clip, transcript, encoder weights and encoder version, so
    /// one directory serves every binary and every voice.
    pub reference_cache: Option<PathBuf>,
    /// Optional official target direction, fixed for this resident voice.
    pub direction: Option<String>,
    pub options: GenerationOptions,
}

impl BreezeVoiceConfig {
    /// Cheap input checks only; this opens no file and initializes no CUDA.
    /// Actual tokenizer controls and both context extents are checked by the
    /// shared prompt/generator APIs for each target, never silently truncated.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.quantization.is_empty() && !self.reference_wav.as_os_str().is_empty(),
            "Breeze requires an explicit weight-format label and reference WAV path"
        );
        ensure!(
            !self.reference_text.trim().is_empty() && self.reference_text.len() <= 65536,
            "Breeze requires a known reference transcript of 1..=65536 UTF-8 bytes"
        );
        ensure!(
            self.direction
                .as_ref()
                .is_none_or(|text| text.len() <= 65536),
            "Breeze direction exceeds 65536 UTF-8 bytes"
        );
        let o = &self.options;
        ensure!(
            (1..=1500).contains(&o.max_frames) && (1..=2048).contains(&o.max_context),
            "invalid bounded Breeze frame/context limit"
        );
        ensure!(
            o.temperature.is_finite()
                && o.temperature > 0.0
                && o.top_k > 0
                && o.top_k <= 2051
                && o.top_p.is_finite()
                && o.top_p > 0.0
                && o.top_p <= 1.0
                && o.repetition_penalty.is_finite()
                && o.repetition_penalty > 0.0
                && o.cfg_scale.is_finite()
                && o.cfg_scale > 0.0,
            "invalid Breeze sampling options"
        );
        ensure!(
            o.cfg_scale == 1.0
                || self
                    .direction
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty()),
            "non-unit Breeze CFG requires an explicit nonblank direction"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ResidentLoadReport {
    pub source_seconds: f64,
    /// Zero when the reference codes came from the cache: the encoder is
    /// then not loaded at all.
    pub encoder_load_seconds: f64,
    /// Reading the cached codes on a hit; encoding the clip on a miss.
    pub reference_encode_seconds: f64,
    pub reference_codes_cached: bool,
    pub generator_load_seconds: f64,
    pub codec_load_host_seconds: f64,
    /// Per hop window length (frames), the seconds its first decode took:
    /// kernel compilation when the CubeCL cache is cold, so no line pays it.
    pub codec_warm_seconds: Vec<(usize, f64)>,
    pub total_seconds: f64,
    pub reference_samples: usize,
    pub reference_frames: usize,
    /// Generator weight bytes copied to the device (BF16 and NVFP4).
    pub device_weight_bytes: u64,
}

/// Models/reference belong to the calling worker, never a global voice owner.
/// Every weight is a device copy; nothing aliases the store it came from.
pub struct BreezeResident {
    generator: Generator,
    codec: CodecDecoder<Fused>,
    device: CudaDevice,
    tokenizer: BreezeTokenizer,
    reference_codes: Vec<[u16; 16]>,
    reference_text: String,
    direction: Option<String>,
    options: GenerationOptions,
    load_report: ResidentLoadReport,
}

impl BreezeResident {
    /// Load on the same worker that will synthesize. Reference waveform input
    /// and the CPU encoder are discarded after one real native encoding.
    ///
    /// Every weight is copied to the device, so the store's bytes are not
    /// needed once this returns.
    pub fn load<R: BlobStoreGet>(
        models: &ModelSnapshot<R>,
        config: BreezeVoiceConfig,
    ) -> Result<Self> {
        config.validate()?;
        let total = Instant::now();
        let (facts, store) = (models.facts(), models.store());
        let artifacts = Artifacts::select(facts, store, &config.quantization)?;
        let assets = Assets::from_frozen(facts, store, artifacts)?;
        let tokenizer = BreezeTokenizer::from_bytes(assets.tokenizer_json.as_bytes())?;
        let loader = WeightLoader::Typed(crate::selection::index_keymap_for_root(
            facts,
            store,
            artifacts.external_codec_root,
        )?);
        let source_seconds = total.elapsed().as_secs_f64();

        let wav_bytes = read_wav_bytes(&config.reference_wav)?;
        let samples = decode_wav(&wav_bytes)?;
        let reference_samples = samples.len();
        let key = ReferenceKey::new(
            &wav_bytes,
            config.reference_text.trim(),
            artifacts.external_codec_root,
        );
        drop(wav_bytes);
        let started = Instant::now();
        let mut encoder_load_seconds = 0.0;
        let (reference_codes, cache) = encode_reference_cached(
            config.reference_cache.as_deref(),
            &key,
            reference_samples.div_ceil(SAMPLES_PER_FRAME),
            || {
                let started = Instant::now();
                let encoder = CodecEncoder::load(&loader);
                encoder_load_seconds = started.elapsed().as_secs_f64();
                encode_reference(&encoder, &samples)
            },
        )?;
        let reference_encode_seconds = started.elapsed().as_secs_f64() - encoder_load_seconds;
        let reference_codes_cached = matches!(cache, super::reference_cache::CacheOutcome::Hit(_));
        eprintln!(
            "[breeze] reference codes: {cache}; key {key}, {} frames, {reference_encode_seconds:.3}s",
            reference_codes.len()
        );
        let reference_frames = reference_codes.len();
        drop(samples);

        let device = CudaDevice { index: 0 };
        let started = Instant::now();
        let generator = Generator::from_pile(
            facts,
            store,
            artifacts.model_root,
            assets.config,
            &device,
            Weights::Bf16,
            config.lane,
        )?;
        generator.synchronize()?;
        let generator_load_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let codec = CodecDecoder::<Fused>::load(&loader, &device);
        let codec_load_host_seconds = started.elapsed().as_secs_f64();
        drop(loader);
        let codec_warm_seconds = hop_window_lengths()
            .into_iter()
            .map(|length| {
                let started = Instant::now();
                let _ = codec.decode(&vec![[0u32; 16]; length], &device);
                (length, started.elapsed().as_secs_f64())
            })
            .collect();
        let load_report = ResidentLoadReport {
            source_seconds,
            encoder_load_seconds,
            reference_encode_seconds,
            reference_codes_cached,
            generator_load_seconds,
            codec_load_host_seconds,
            codec_warm_seconds,
            total_seconds: total.elapsed().as_secs_f64(),
            reference_samples,
            reference_frames,
            device_weight_bytes: generator.device_bytes(),
        };
        let ordinal = next_load_ordinal();
        eprintln!(
            "[breeze] resident load {ordinal} in pid {}: {load_report:?}",
            std::process::id()
        );
        if ordinal > 1 {
            eprintln!(
                "[breeze] WARNING: this process now holds {ordinal} Breeze residents; one owning worker loads once, and every further load pays the device copy again"
            );
        }
        Ok(Self {
            generator,
            codec,
            device,
            tokenizer,
            reference_codes,
            reference_text: config.reference_text.trim().to_owned(),
            direction: config.direction,
            options: config.options,
            load_report,
        })
    }

    pub fn load_report(&self) -> &ResidentLoadReport {
        &self.load_report
    }

    /// One fresh generator KV/sampling state; the line's PCM goes to `on_pcm`
    /// hop by hop while it is generated (see [`pipeline::stream`]). Resident
    /// model tensors and the encoded reference are reused. The configured
    /// seed starts anew for each utterance, as in the CLI.
    pub fn synthesize_stream(
        &mut self,
        text: &str,
        on_pcm: impl FnMut(Vec<f32>) -> Result<()>,
    ) -> Result<Streamed> {
        let started = Instant::now();
        let prompt = self.tokenizer.reference_prompts(
            &self.reference_text,
            &self.reference_codes,
            text,
            self.direction.as_deref(),
            self.options.cfg_scale,
        )?;
        let result = pipeline::stream(
            &mut self.generator,
            &prompt,
            &self.codec,
            &self.device,
            GenerationOptions {
                max_frames: frame_budget(text, self.options.max_frames),
                ..self.options.clone()
            },
            on_pcm,
        )?;
        let hops = &result.hop_seconds;
        eprintln!(
            "[breeze] resident utterance: {:?}, {} frames, {:.3}s generation, {} hops of {:.1} ms codec ({:.1} ms max), first PCM {:.3}s, {:.3}s total; CFG{}, prompt {} / {:?} tokens; hop PCM",
            result.generation.termination,
            result.frames.len(),
            result.generation.total_seconds - result.generation.callback_seconds,
            hops.len(),
            hops.iter().sum::<f64>() / hops.len().max(1) as f64 * 1e3,
            hops.iter().copied().fold(0.0, f64::max) * 1e3,
            result.first_pcm_seconds.unwrap_or(f64::NAN),
            started.elapsed().as_secs_f64(),
            result.generation.cfg_scale,
            result.generation.conditional_prompt_tokens,
            result.generation.negative_prompt_tokens,
        );
        Ok(result)
    }

    /// The same frames decoded as one batch, the way [`pipeline::run`] does:
    /// the reference a hop decode is listened against.
    #[cfg(test)]
    pub(crate) fn decode_batch(&self, frames: &[[u16; 16]]) -> Result<Vec<f32>> {
        super::audio::decode_generated(&self.codec, frames, &self.device)
    }
}

/// Loads in this process so far. One process has one owning worker and one
/// resident; the load line names its ordinal and the pid so two reports in
/// one log can be told apart as two loads in one process or two processes
/// (the 2026-10-12 cutover log held one of each binary, read as a double
/// load because neither line said which process it was).
fn next_load_ordinal() -> usize {
    static LOADS: AtomicUsize = AtomicUsize::new(0);
    LOADS.fetch_add(1, Ordering::SeqCst) + 1
}

/// How many frames a line of `text` may take before it counts as a runaway:
/// three frames per character plus two seconds, never more than the
/// configured cap. The live voice speaks 0.75 to 0.9 frames per character
/// (sky, 2026-10-10), so an ordinary line stays far below it. Streamed PCM
/// cannot be taken back, so without this a generation that never reached its
/// end would be heard for the whole configured cap.
fn frame_budget(text: &str, configured: usize) -> usize {
    (text.chars().count().saturating_mul(3) + 24).clamp(1, configured.max(1))
}

#[cfg(test)]
pub(crate) fn test_config() -> BreezeVoiceConfig {
    BreezeVoiceConfig {
        quantization: crate::persist::QUANTIZATION_NVFP4.to_owned(),
        lane: super::PROMPT_LANE,
        reference_wav: PathBuf::from("request-reference.wav"),
        reference_text: "Known reference words.".to_owned(),
        reference_cache: None,
        direction: None,
        options: GenerationOptions {
            max_frames: 256,
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_config_requires_real_inputs_and_bounded_options() {
        let original = test_config();
        original.validate().unwrap();
        let mut bad = original.clone();
        bad.reference_text = " \n".to_owned();
        assert!(bad.validate().is_err());
        bad = original.clone();
        bad.options.max_frames = 0;
        assert!(bad.validate().is_err());
        bad = original.clone();
        bad.options.temperature = f32::NAN;
        assert!(bad.validate().is_err());
        bad = original;
        bad.quantization.clear();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn nonunit_resident_cfg_needs_direction() {
        let mut config = test_config();
        config.options.cfg_scale = 4.0;
        assert!(config.validate().is_err());
        config.direction = Some(" \n".to_owned());
        assert!(config.validate().is_err());
        config.direction = Some("Speak softly.".to_owned());
        config.validate().unwrap();
        config.options.cfg_scale = f32::INFINITY;
        assert!(config.validate().is_err());
    }

    #[test]
    fn load_ordinals_count_up_within_one_process() {
        let first = next_load_ordinal();
        let second = next_load_ordinal();
        assert_eq!(second, first + 1);
        assert!(first >= 1);
    }

    #[test]
    fn a_line_may_run_about_four_times_its_spoken_length_and_no_further() {
        // 27 characters spoke as 23 frames on the live voice.
        assert_eq!(super::frame_budget("This is the four-bit voice.", 1500), 105);
        assert_eq!(super::frame_budget("", 1500), 24);
        assert_eq!(super::frame_budget(&"a".repeat(1000), 1500), 1500);
        assert_eq!(super::frame_budget("short", 10), 10);
    }
}
