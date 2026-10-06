//! Resident native Breeze weights and one natively encoded reference.
//!
//! One owning worker calls `load` once, then `synthesize` for each whole line.
//! Output is a fresh generated-only batch, not stateful streaming. No request
//! can inject precomputed reference codes, a checkpoint path or a CPU LLM.
use anyhow::{Result, ensure};
use cubecl::cuda::CudaDevice;
use std::{path::PathBuf, time::Instant};

use super::{
    generator::{GenerationOptions, Generator},
    load::{Artifacts, Assets},
    pipeline::{self, Synthesis},
    prompt::BreezeTokenizer,
    reference::{encode_reference, read_wav},
};
use crate::{
    models::qwen3tts::{codec::CodecDecoder, encoder::CodecEncoder},
    nn::{backend::speak::Fused, cuda_bf16_alias::CudaBf16Aliases, weight_loader::WeightLoader},
    persist::ModelPileSource,
};

/// Explicit per-resident inputs. The five artifact IDs are opaque, selected
/// stored identities; none is derived from another field or from a path.
#[derive(Clone, Debug)]
pub struct BreezeVoiceConfig {
    pub pile: PathBuf,
    pub artifacts: Artifacts,
    pub reference_wav: PathBuf,
    /// Known transcript, not an ASR request. Outer whitespace is stripped once.
    pub reference_text: String,
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
            !self.pile.as_os_str().is_empty() && !self.reference_wav.as_os_str().is_empty(),
            "Breeze requires explicit pile and reference WAV paths"
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
    pub encoder_load_seconds: f64,
    pub reference_encode_seconds: f64,
    pub generator_load_seconds: f64,
    pub codec_load_host_seconds: f64,
    pub total_seconds: f64,
    pub reference_samples: usize,
    pub reference_frames: usize,
    pub alias_registrations: usize,
    pub aliased_bytes: u64,
}

/// Models/reference belong to the calling worker, never a global voice owner.
/// Fields drop GPU consumers before local alias/source keepers. Independently,
/// CubeCL retains real mmap owners through storage teardown, also on partial
/// construction errors; neither field drop nor sync releases file custody.
pub struct BreezeResident {
    generator: Generator,
    codec: CodecDecoder<Fused>,
    _aliases: CudaBf16Aliases,
    _source: ModelPileSource,
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
    /// # Safety
    /// The selected file must be a genuine validated pile. Its aliased bytes,
    /// including preceding partial file-backed page prefixes, must remain
    /// immutable or append-only/untruncated until CUDA runtime/storage teardown,
    /// including after this resident/worker drops or construction fails. A
    /// read-only file descriptor, owned snapshot or mode0444 is not that proof.
    pub unsafe fn load(config: BreezeVoiceConfig) -> Result<Self> {
        config.validate()?;
        let total = Instant::now();
        let source = crate::persist::read_model_pile_read_only(&config.pile)?;
        let assets = Assets::from_frozen(&source.facts, &source.store, config.artifacts)?;
        let tokenizer = BreezeTokenizer::from_bytes(assets.tokenizer_json.as_bytes())?;
        let snapshot = crate::model_collection::snapshot_model_collection_in(&source.store)?;
        let loader = WeightLoader::selected(snapshot, config.artifacts.external_codec_root);
        let source_seconds = total.elapsed().as_secs_f64();

        let samples = read_wav(&config.reference_wav)?;
        let reference_samples = samples.len();
        let started = Instant::now();
        let encoder = CodecEncoder::load(&loader);
        let encoder_load_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let reference_codes = encode_reference(&encoder, &samples)?;
        let reference_encode_seconds = started.elapsed().as_secs_f64();
        let reference_frames = reference_codes.len();
        drop(encoder);
        drop(samples);

        let device = CudaDevice { index: 0 };
        let started = Instant::now();
        let mut aliases = CudaBf16Aliases::new(device.clone(), 2048).map_err(anyhow::Error::msg)?;
        // SAFETY: the caller establishes real pile custody through runtime
        // teardown; this exact validated observation owns every selected blob.
        let generator = unsafe {
            Generator::from_pile(
                &source.facts,
                &source.store,
                config.artifacts,
                assets.config,
                &mut aliases,
            )?
        };
        generator.synchronize()?;
        let generator_load_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let codec = CodecDecoder::<Fused>::load(&loader, &device);
        let codec_load_host_seconds = started.elapsed().as_secs_f64();
        drop(loader);
        let stats = aliases.stats();
        let load_report = ResidentLoadReport {
            source_seconds,
            encoder_load_seconds,
            reference_encode_seconds,
            generator_load_seconds,
            codec_load_host_seconds,
            total_seconds: total.elapsed().as_secs_f64(),
            reference_samples,
            reference_frames,
            alias_registrations: stats.registrations,
            aliased_bytes: stats.aliased_bytes,
        };
        eprintln!("[breeze] resident loaded once: {load_report:?}");
        Ok(Self {
            generator,
            codec,
            _aliases: aliases,
            _source: source,
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

    /// One fresh generator KV/sampling state and fresh generated-only codec
    /// state; resident model tensors and the encoded reference are reused.
    /// The configured seed starts anew for each utterance, as in the CLI.
    pub fn synthesize(&mut self, text: &str) -> Result<Synthesis> {
        let started = Instant::now();
        let prompt = self.tokenizer.reference_prompts(
            &self.reference_text,
            &self.reference_codes,
            text,
            self.direction.as_deref(),
            self.options.cfg_scale,
        )?;
        let result = pipeline::run(
            &mut self.generator,
            &prompt,
            &self.codec,
            &self.device,
            self.options.clone(),
        )?;
        eprintln!(
            "[breeze] resident utterance: {:?}, {} frames, {:.3}s generation, {:.3}s codec, {:.3}s total; CFG{}, prompt {} / {:?} tokens; batch PCM",
            result.generation.termination,
            result.frames.len(),
            result.generation.total_seconds,
            result.codec_decode_seconds,
            started.elapsed().as_secs_f64(),
            result.generation.cfg_scale,
            result.generation.conditional_prompt_tokens,
            result.generation.negative_prompt_tokens,
        );
        Ok(result)
    }
}

#[cfg(test)]
pub(crate) fn test_config() -> BreezeVoiceConfig {
    use triblespace::prelude::fucid;
    BreezeVoiceConfig {
        pile: std::env::temp_dir().join(format!("absent-breeze-{}.pile", fucid().id)),
        artifacts: Artifacts {
            model_root: fucid().id,
            config_root: fucid().id,
            tokenizer_asset: fucid().id,
            external_codec_root: fucid().id,
            external_codec_config_root: fucid().id,
        },
        reference_wav: PathBuf::from("request-reference.wav"),
        reference_text: "Known reference words.".to_owned(),
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
        bad.pile.clear();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn nonunit_resident_cfg_needs_direction_without_changing_opaque_ids() {
        let mut config = test_config();
        let root = config.artifacts.model_root;
        config.options.cfg_scale = 4.0;
        assert!(config.validate().is_err());
        config.direction = Some(" \n".to_owned());
        assert!(config.validate().is_err());
        config.direction = Some("Speak softly.".to_owned());
        config.validate().unwrap();
        assert_eq!(config.artifacts.model_root, root);
        config.options.cfg_scale = f32::INFINITY;
        assert!(config.validate().is_err());
    }
}
