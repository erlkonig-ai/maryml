//! Finite native Breeze WAV: pile-only models/tokenizer, native CPU reference
//! encoding, CUDA generation and fresh generated-only CUDA batch codec decode.
#![recursion_limit = "256"]

use anyhow::{Context, Result, ensure};
use clap::Parser;
use cubecl::cuda::CudaDevice;
use mary::{
    models::{
        breeze::{
            audio::write_float_wav_new,
            generator::{GenerationOptions, Generator, PromptSegment},
            load::{Artifacts, Assets},
            pipeline,
            prompt::BreezeTokenizer,
            reference::SAMPLE_RATE,
        },
        qwen3tts::{codec::CodecDecoder, encoder::CodecEncoder},
    },
    nn::{backend::speak::Fused, weight_loader::WeightLoader},
};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use triblespace::prelude::Id;

#[derive(Parser)]
#[command(
    about = "Native Breeze text/reference to 24 kHz FLOAT WAV (finite batch codec; not streaming)"
)]
struct Args {
    #[arg(long)]
    pile: PathBuf,
    /// Exact model collection descriptor handle (`blake3:HEX`). The Breeze
    /// root is selected in it by source revision and `--quantization`, and
    /// its assets by name. Without it, the five explicit ids below name a
    /// legacy native model pile.
    #[arg(long, conflicts_with_all = ["model_root", "config_root", "tokenizer_asset", "external_codec_root", "external_codec_config_root"])]
    collection: Option<String>,
    /// The stored weight-format label to select under `--collection`.
    #[arg(long, default_value = "nvfp4")]
    quantization: String,
    /// BF16-stored projections only: `bf16`, or `nvfp4` to quantize them on
    /// the device at load. NVFP4-stored projections are bound as stored.
    #[arg(long, default_value = "bf16")]
    weights: String,
    /// How NVFP4 projections multiply prompt-height activations: `w4a16` or
    /// `w4a4`.
    #[arg(long, default_value = "w4a16")]
    lane: String,
    #[arg(long)]
    model_root: Option<String>,
    #[arg(long)]
    config_root: Option<String>,
    #[arg(long)]
    tokenizer_asset: Option<String>,
    #[arg(long)]
    external_codec_root: Option<String>,
    #[arg(long)]
    external_codec_config_root: Option<String>,
    #[arg(long)]
    text: String,
    /// Official ref_edit_tata target instruction; no singing guarantee.
    #[arg(long)]
    direction: Option<String>,
    /// 24 kHz PCM16 or IEEE FLOAT32 WAV, at most 30 seconds.
    #[arg(long)]
    reference: PathBuf,
    /// Known transcript, UTF-8; outer whitespace stripped as in the reference
    /// runner. No ASR process or automatic transcription.
    #[arg(long)]
    reference_text: PathBuf,
    #[arg(long, default_value_t = 256)]
    max_frames: usize,
    #[arg(long, default_value_t = 2048)]
    max_context: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,
    #[arg(long, default_value_t = 50)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    #[arg(long)]
    greedy: bool,
    /// Non-unit CFG uses an explicit direction and paired positive/negative branches.
    #[arg(long, default_value_t = 1.0)]
    cfg_scale: f32,
    /// Measure the real native CPU reference encoder and write its code receipt,
    /// then exit before CUDA initialization. This is not a saved-code input mode.
    #[arg(long)]
    reference_only: bool,
    /// New audio file; never overwritten.
    #[arg(long)]
    out: Option<PathBuf>,
    /// New JSON receipt, with raw backbone IDs, codec frames and termination.
    #[arg(long)]
    report: PathBuf,
}

fn id(value: &Option<String>, name: &str) -> Result<Id> {
    let value = value
        .as_deref()
        .with_context(|| format!("--{name} is required without --collection"))?;
    Id::from_hex(value).context("expected explicit stored 32-hex entity ID")
}

fn read_text(path: &Path) -> Result<String> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open transcript {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 65536,
        "reference transcript exceeds 65536 bytes"
    );
    let text = String::from_utf8(bytes).context("reference transcript must be UTF-8")?;
    ensure!(!text.trim().is_empty(), "reference transcript is empty");
    Ok(text.trim().to_owned())
}

fn write_report(path: &Path, report: &serde_json::Value) -> Result<()> {
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&serde_json::to_vec_pretty(report)?)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.report.exists()
            && args
                .out
                .as_ref()
                .is_none_or(|out| !out.exists() && out != &args.report),
        "choose distinct new audio/report paths; existing evidence is preserved"
    );
    ensure!(
        args.reference_only == args.out.is_none(),
        "provide --out for synthesis; omit it for --reference-only"
    );
    ensure!(
        (1..=1500).contains(&args.max_frames) && (1..=2048).contains(&args.max_context),
        "bounded endpoint requires frames1..1500 and context1..2048"
    );
    ensure!(
        args.cfg_scale.is_finite()
            && args.cfg_scale > 0.0
            && (args.cfg_scale == 1.0
                || args
                    .direction
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty())),
        "CFG must be finite and positive; non-unit scale requires an explicit direction"
    );
    ensure!(
        args.temperature.is_finite()
            && args.temperature > 0.0
            && args.top_k > 0
            && args.top_k <= 2051
            && args.top_p.is_finite()
            && args.top_p > 0.0
            && args.top_p <= 1.0
            && args.repetition_penalty.is_finite()
            && args.repetition_penalty > 0.0,
        "invalid sampling options"
    );
    let options = GenerationOptions {
        max_frames: args.max_frames,
        max_context: args.max_context,
        seed: args.seed,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        do_sample: !args.greedy,
        cfg_scale: args.cfg_scale,
    };
    let weights = mary::models::breeze::generator::Weights::parse(&args.weights)?;
    let lane = mary::nn::nvfp4_gemm::PromptLane::parse(&args.lane)?;
    let total_started = Instant::now();
    let (facts, store, ids) = if let Some(collection) = &args.collection {
        let store = mary::persist::read_model_pile_snapshot_read_only(&args.pile)?;
        let collection = mary::model_collection::open_model_collection(
            &store,
            mary::model_collection::parse_collection_handle(collection)?,
        )?;
        let models = mary::model_collection::snapshot_model_collection_for(&store, collection)?;
        let ids = Artifacts::select(models.facts(), models.store(), &args.quantization)?;
        let (facts, _, store) = models.into_parts();
        (facts, store, ids)
    } else {
        let source = mary::persist::read_model_pile_read_only(&args.pile)?;
        let ids = Artifacts {
            model_root: id(&args.model_root, "model-root")?,
            config_root: id(&args.config_root, "config-root")?,
            tokenizer_asset: id(&args.tokenizer_asset, "tokenizer-asset")?,
            external_codec_root: id(&args.external_codec_root, "external-codec-root")?,
            external_codec_config_root: id(&args.external_codec_config_root, "external-codec-config-root")?,
        };
        (source.facts, source.store, ids)
    };
    let assets = Assets::from_frozen(&facts, &store, ids)?;
    let tokenizer = BreezeTokenizer::from_bytes(assets.tokenizer_json.as_bytes())?;
    let codec_loader = WeightLoader::Typed(mary::selection::index_keymap_for_root(
        &facts,
        &store,
        ids.external_codec_root,
    )?);
    let reference_text = read_text(&args.reference_text)?;
    let source_seconds = total_started.elapsed().as_secs_f64();

    let started = Instant::now();
    let encoder = CodecEncoder::load(&codec_loader);
    let encoder_load_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "Native CPU reference encoder loaded in {encoder_load_seconds:.3}s; encoding reference"
    );
    let prepared = pipeline::prepare_reference(
        &encoder,
        &tokenizer,
        &args.reference,
        &reference_text,
        &args.text,
        args.direction.as_deref(),
        args.cfg_scale,
    )?;
    drop(encoder);
    if args.reference_only {
        let codes = prepared
            .prompt
            .conditional
            .iter()
            .find_map(|segment| match segment {
                PromptSegment::AudioFrames(codes) => Some(codes),
                _ => None,
            })
            .context("prepared reference has no audio frames")?;
        let report = serde_json::json!({
            "scope":"native Rust CPU reference encoding only; no CUDA initialization, generation or WAV output",
            "pile":args.pile,"collection":args.collection,"model_root":format!("{:X}", ids.model_root),
            "config_root":format!("{:X}", ids.config_root),
            "tokenizer_asset":format!("{:X}", ids.tokenizer_asset),
            "external_codec_root":format!("{:X}", ids.external_codec_root),
            "external_codec_config_root":format!("{:X}", ids.external_codec_config_root),
            "weights":args.weights,"lane":args.lane,
            "reference":args.reference,"reference_text":reference_text,"reference_transcript_path":args.reference_text,
            "reference_samples":prepared.reference_samples,"reference_frames":prepared.reference_frames,
            "codebooks":16,"code_min":codes.iter().flatten().min(),"code_max":codes.iter().flatten().max(),
            "reference_codec_frames":codes,
            "source_seconds":source_seconds,"encoder_load_seconds":encoder_load_seconds,
            "reference_wav_read_seconds":prepared.wav_read_seconds,
            "reference_encode_seconds":prepared.reference_encode_seconds,"prompt_seconds":prepared.prompt_seconds,
            "total_seconds":total_started.elapsed().as_secs_f64(),
            "reference_placement":"native Rust CPU; Linux scalar GEMM; no oracle codes or fallback",
            "receipt_is_not_a_runtime_reference_code_input":true,
        });
        write_report(&args.report, &report)?;
        eprintln!(
            "Native reference-only complete: {} frames in {:.3}s encode time; no CUDA initialized",
            prepared.reference_frames, prepared.reference_encode_seconds
        );
        return Ok(());
    }
    eprintln!(
        "Reference encoded in {:.3}s: {} frames; CUDA model binding starts",
        prepared.reference_encode_seconds, prepared.reference_frames
    );

    let started = Instant::now();
    // Every weight is copied to the device; nothing stays aliased.
    let mut generator = Generator::from_pile(
        &facts,
        &store,
        ids.model_root,
        assets.config,
        &CudaDevice { index: 0 },
        weights,
        lane,
    )?;
    generator.synchronize()?;
    let generator_load_seconds = started.elapsed().as_secs_f64();
    let device = CudaDevice { index: 0 };
    let started = Instant::now();
    let codec = CodecDecoder::<Fused>::load(&codec_loader, &device);
    let codec_load_seconds = started.elapsed().as_secs_f64();
    let result = pipeline::run(&mut generator, &prepared.prompt, &codec, &device, options)?;
    generator.synchronize()?;
    let synthesis_seconds = total_started.elapsed().as_secs_f64();
    let duration = result.samples.len() as f64 / f64::from(SAMPLE_RATE);
    let started = Instant::now();
    write_float_wav_new(
        args.out.as_ref().context("missing output path")?,
        &result.samples,
    )?;
    let wav_write_seconds = started.elapsed().as_secs_f64();
    let report = serde_json::json!({
        "scope":"native Rust reference + CUDA generation + generated-only CUDA batch codec; not stateful streaming",
        "pile":args.pile,"collection":args.collection,"model_root":format!("{:X}", ids.model_root),
        "config_root":format!("{:X}", ids.config_root),
        "tokenizer_asset":format!("{:X}", ids.tokenizer_asset),
        "external_codec_root":format!("{:X}", ids.external_codec_root),
        "external_codec_config_root":format!("{:X}", ids.external_codec_config_root),
        "weights":args.weights,"lane":args.lane,
        "text":args.text,"direction":args.direction,"reference":args.reference,
        "reference_text":reference_text,"reference_transcript_path":args.reference_text,
        "reference_samples":prepared.reference_samples,"reference_frames":prepared.reference_frames,
        "reference_placement":"native Rust CPU; Linux scalar GEMM; no cached oracle reference codes",
        "template":if args.direction.as_deref().is_some_and(|s| !s.trim().is_empty()) {"ref_edit_tata"} else {"ref_clone_tata"},
        "max_frames":args.max_frames,"max_context":args.max_context,"seed":args.seed,
        "temperature":args.temperature,"top_k":args.top_k,"top_p":args.top_p,
        "repetition_penalty":args.repetition_penalty,"do_sample":!args.greedy,"cfg_scale":args.cfg_scale,
        "termination":format!("{:?}",result.generation.termination),
        "conditional_prompt_tokens":result.generation.conditional_prompt_tokens,
        "negative_prompt_tokens":result.generation.negative_prompt_tokens,
        "raw_backbone_ids":result.generation.backbone_ids,"frames":result.frames,
        "sample_rate":SAMPLE_RATE,"samples":result.samples.len(),"audio_seconds":duration,
        "source_seconds":source_seconds,"encoder_load_seconds":encoder_load_seconds,
        "reference_wav_read_seconds":prepared.wav_read_seconds,
        "reference_encode_seconds":prepared.reference_encode_seconds,"prompt_seconds":prepared.prompt_seconds,
        "generator_load_seconds":generator_load_seconds,"codec_load_host_seconds":codec_load_seconds,
        "prefill_seconds":result.generation.prefill_seconds,"decode_seconds":result.generation.decode_seconds,
        "callback_seconds":result.generation.callback_seconds,"generation_seconds":result.generation.total_seconds,
        "codec_decode_seconds":result.codec_decode_seconds,"synthesis_seconds":synthesis_seconds,
        "wav_write_seconds":wav_write_seconds,"cold_total_rtf":synthesis_seconds/duration,
        "generation_and_codec_rtf":(result.generation.total_seconds+result.codec_decode_seconds)/duration,
        "timing_scope":"cold one request; includes native CPU reference and CUDA JIT/page faults; codec load host time may defer GPU work into decode",
        "output_format":"24k mono IEEE FLOAT32 WAV; no normalization or fades",
        "out":args.out,"device_weight_bytes":generator.device_bytes(),
        "runtime_assets":"explicit pile identities only; request WAV/transcript are external inputs",
    });
    write_report(&args.report, &report)?;
    eprintln!(
        "Wrote {:.3}s audio; termination {:?}; cold RTF {:.3}",
        duration,
        result.generation.termination,
        synthesis_seconds / duration
    );
    generator.synchronize()?;
    drop(codec);
    drop(generator);
    drop(codec_loader);
    drop((facts, store));
    Ok(())
}
