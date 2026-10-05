//! Native CUDA token-only entrypoint. No waveform decoding or Python/runtime
//! checkpoint files. The operator keeps the selected pile immutable through
//! process/CUDA teardown, including partial pages before aliased tensor data.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use cubecl::cuda::CudaDevice;
use mary::{
    models::step_audio2::{
        cuda::{Decoder, validate_geometry, validate_request},
        load::{Artifacts, Assets},
    },
    nn::cuda_bf16_alias::CudaBf16Aliases,
};
use std::{fs::OpenOptions, io::Write, path::PathBuf, time::Instant};
use triblespace::prelude::Id;

#[derive(Parser)]
#[command(about = "Native Step-Audio2 Mini CUDA token generator (greedy; no waveform decoder)")]
struct Args {
    #[arg(long)]
    pile: PathBuf,
    #[arg(long)]
    model_root: String,
    #[arg(long)]
    config_root: String,
    #[arg(long)]
    tokenizer_asset: String,
    /// Confirm the genuine pile prefix stays immutable/untruncated until this
    /// process's CUDA runtime exits. A read-only FD alone is insufficient.
    #[arg(long)]
    immutable_pile: bool,
    #[arg(
        long,
        default_value = "Read the user's sentence verbatim. Do not add words or describe your performance. Speak naturally at a normal conversational pace."
    )]
    system: String,
    #[arg(long, default_value = "The moon is smiling, and so am I.")]
    text: String,
    #[arg(long, default_value_t = 192)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 512)]
    capacity: usize,
    /// New JSON output file. Existing evidence is never overwritten.
    #[arg(long)]
    output: PathBuf,
}
fn id(value: &str) -> Result<Id> {
    Id::from_hex(value).context("expected an explicit stored 32-hex entity ID")
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.immutable_pile,
        "--immutable-pile custody confirmation is required through CUDA teardown"
    );
    ensure!(
        !args.output.exists(),
        "preserve existing output; choose a new path"
    );
    let ids = Artifacts {
        model_root: id(&args.model_root)?,
        config_root: id(&args.config_root)?,
        tokenizer_asset: id(&args.tokenizer_asset)?,
    };
    let started = Instant::now();
    // This immutable validated snapshot is retained until the end of main.
    // Runtime external storage additionally retains each actual mmap owner.
    let source = mary::persist::read_model_pile_read_only(&args.pile)?;
    let assets = Assets::from_frozen(&source.facts, &source.store, ids)?;
    validate_geometry(&assets.config, args.capacity)?;
    let prompt = assets.codec.speech_prompt(&args.system, &args.text)?;
    validate_request(&assets.config, args.capacity, &prompt, args.max_new_tokens)?;
    let mut registrations = 0;
    assets.config.tensors(|_, _| {
        registrations += 1;
        Ok(())
    })?;
    let source_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "Loaded pile metadata and {} prompt IDs; binding {} native BF16 tensors",
        prompt.len(),
        registrations
    );
    let started = Instant::now();
    let mut aliases =
        CudaBf16Aliases::new(CudaDevice { index: 0 }, registrations).map_err(anyhow::Error::msg)?;
    // SAFETY: operator explicitly establishes genuine immutable pile custody;
    // read_model_pile_read_only validates and freezes its observation. Neither
    // success nor error here authorizes truncation/replacement before process
    // teardown; external storage retains real mapping owners beyond local drops.
    let mut decoder = unsafe {
        Decoder::from_frozen(
            &source.facts,
            &source.store,
            ids,
            assets,
            args.capacity,
            &mut aliases,
        )?
    };
    decoder.synchronize()?;
    let binding_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "CUDA BF16 aliases bound in {binding_seconds:.3}s; token-only greedy generation starts"
    );
    let result = decoder.generate(&args.system, &args.text, args.max_new_tokens)?;
    let stats = aliases.stats();
    let report = serde_json::json!({
        "scope":"native CUDA text/audio tokens only; not native speech",
        "model_root":args.model_root,"config_root":args.config_root,"tokenizer_asset":args.tokenizer_asset,
        "pile":args.pile,"system":args.system,"text":args.text,
        "selection":"greedy CUDA argmax; earliest ID on ties; not Python sampling",
        "weight_storage":"immutable native BF16 pile aliases; no F16 conversion or checkpoint runtime",
        "cache":"resident rotated KV, fresh immutable append, capacity bounded per request",
        "capacity":args.capacity,"max_new_tokens":args.max_new_tokens,
        "prompt_ids":prompt,"prompt_tokens":result.prompt_tokens,
        "raw_generated_ids":result.generated.raw_ids,
        "text_ids":result.generated.text_ids,"control_ids":result.generated.control_ids,
        "speech_codes":result.generated.speech_codes,"audio_padding_ids":result.generated.audio_padding_ids,
        "termination":format!("{:?}",result.generated.termination),"generated_text":result.text,
        "source_seconds":source_seconds,"binding_seconds":binding_seconds,
        "prefill_seconds":result.prefill_seconds,"decode_seconds":result.decode_seconds,
        "generation_seconds":result.total_seconds,
        "timing_scope":"cold first request includes CUDA JIT/page faults; prefill includes first token selection, decode includes remaining selections; no RTF without waveform",
        "alias_registrations":stats.registrations,"aliased_bytes":stats.aliased_bytes,
        "registered_span_bytes":stats.registered_span_bytes,"mmap_owner_capacity_bytes":stats.owner_capacity_bytes,
        "cuda_device_index":decoder.device().index,
    });
    let bytes = serde_json::to_vec_pretty(&report)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    output.write_all(&bytes)?;
    output.write_all(b"\n")?;
    println!("{}", String::from_utf8(bytes)?);
    decoder.synchronize()?;
    // Explicit final sync before dropping handles; mmap ownership in CUDA's
    // external storage remains until process/runtime teardown regardless.
    drop(decoder);
    drop(aliases);
    drop(source);
    Ok(())
}
