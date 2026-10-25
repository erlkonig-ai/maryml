//! Greedy completion from raw or explicitly trusted-framed text, not a renderer.
//! Requires explicit model, prompt file and new output directory.
use std::{fs::{self,OpenOptions},io::Write,path::{Path,PathBuf},time::Instant};
use anyhow::{Context,Result,ensure};
use clap::{Parser,ValueEnum};
use serde_json::json;
use mary::models::flint::{cache::Limits,forward::Backend,session::{FlintSession,Options,NativeHead,greedy_id},weights::ReceiptPolicy};

#[derive(Clone,Copy,ValueEnum)]
enum Mode { Cpu,Cuda,PackedAll }
#[derive(Clone,Copy,ValueEnum)]
enum Receipts { Collect,None }
#[derive(Clone,Copy,Debug,PartialEq,Eq,ValueEnum)]
enum PromptMode { Raw,TrustedFramed }
#[derive(Clone,Copy,ValueEnum)]
enum HeadMode { Streamed,Resident }
#[derive(Parser)]
struct Args {
    model:PathBuf, prompt:PathBuf, output:PathBuf,
    /// Model entity in the explicitly named positional pile path (not a GGUF).
    #[arg(long)] typed_pile:Option<String>,
    /// Exact native collection descriptor (64 hex digits), separate from root.
    /// Omit only when the pile has one discoverable mary-model-graph collection.
    #[arg(long,requires="typed_pile")] collection:Option<String>,
    /// trusted-framed parses literal controls in caller-rendered trusted text;
    /// it performs no template rendering, escaping or user-content sanitation.
    #[arg(long,value_enum,default_value="raw")] prompt_mode:PromptMode,
    #[arg(long,value_enum,default_value="packed-all")] backend:Mode,
    #[arg(long,default_value_t=1024)] capacity:usize,
    #[arg(long,default_value_t=8)] max_new:usize,
    #[arg(long,default_value_t=256)] max_cache_mib:usize,
    #[arg(long,default_value_t=64)] max_attention_mib:usize,
    /// Native pile only; GGUF stays streamed. Immutable head survives reset.
    #[arg(long,value_enum,default_value="resident")] native_head:HeadMode,
    /// Immutable head buffers only, excluding workspace/runtime/host overhead.
    #[arg(long,default_value_t=192)] max_head_mib:usize,
    #[arg(long)] trace:bool,
    /// Saves each last-row vocabulary distribution for independent prefix replay.
    #[arg(long)] save_logits:bool,
    /// Selected-weight provenance is unused by this raw generator. Collect is
    /// available for paired diagnostics; neither mode changes validation or math.
    #[arg(long,value_enum,default_value="none")] weight_receipts:Receipts,
}
fn save(path:&Path,bytes:&[u8])->Result<()> {
    OpenOptions::new().create_new(true).write(true).open(path)?.write_all(bytes)?; Ok(())
}
fn main()->Result<()> {
    let process_start=Instant::now();
    let args=Args::parse();
    let collection=args.collection.as_deref().map(mary::model_collection::parse_collection_handle).transpose()?;
    let before=fs::symlink_metadata(&args.prompt)?;
    ensure!(before.is_file() && before.len()<=1024*1024,"prompt must be a regular UTF-8 file <=1MiB");
    let raw=fs::read(&args.prompt)?;
    let after=fs::symlink_metadata(&args.prompt)?;
    ensure!(raw.len() as u64==before.len() && before.len()==after.len() && before.modified()?==after.modified()?,"prompt changed while reading");
    let prompt=std::str::from_utf8(&raw).context("prompt is not UTF-8")?;
    let options=Options {backend:match args.backend {Mode::Cpu=>Backend::Cpu,Mode::Cuda=>Backend::Cuda,Mode::PackedAll=>Backend::PackedAll},
        limits:Limits {capacity:args.capacity,max_cache_bytes:args.max_cache_mib.checked_mul(1<<20).context("KV MiB overflow")?,
            max_attention_bytes:args.max_attention_mib.checked_mul(1<<20).context("attention MiB overflow")?},trace:args.trace,
        native_head:match args.native_head {HeadMode::Streamed=>NativeHead::Streamed,
            HeadMode::Resident=>NativeHead::Resident {max_bytes:args.max_head_mib.checked_mul(1<<20).context("head MiB overflow")?}}};
    fs::create_dir(&args.output).context("output directory must be new")?;
    let process_setup_seconds=process_start.elapsed().as_secs_f64();
    let start=Instant::now();
    let mut session=if let Some(root)=&args.typed_pile {
        ensure!(matches!(args.weight_receipts,Receipts::None),"native pile has no GGUF read receipts");
        let root=triblespace::prelude::Id::from_hex(root).context("invalid native model root")?;
        FlintSession::open_pile_with_collection(&args.model,root,collection,options)?
    } else {FlintSession::open_with_receipts(&args.model,options,match args.weight_receipts {
        Receipts::Collect=>ReceiptPolicy::Collect,Receipts::None=>ReceiptPolicy::None})?};
    let model_init_seconds=start.elapsed().as_secs_f64();
    let start=Instant::now();
    let ids=match args.prompt_mode {
        PromptMode::Raw=>session.encode_raw(prompt)?,
        PromptMode::TrustedFramed=>session.encode_trusted_framed(prompt)?,
    };
    let report_encode_seconds=start.elapsed().as_secs_f64();
    let mut choices=Vec::new();
    let mut observer_seconds=0.0;
    let start=Instant::now();
    let (generation,generation_timings)=session.generate_ids_observed_timed(&ids,args.max_new,|attended,logits| {
        let start=Instant::now();
        let id=greedy_id(logits)? as usize;
        let second=logits.iter().enumerate().filter(|&(i,_)|i!=id).map(|(_,v)|*v).fold(f32::NEG_INFINITY,f32::max);
        let bytes:Vec<u8>=logits.iter().flat_map(|v|v.to_le_bytes()).collect();
        if args.save_logits { save(&args.output.join(format!("logits-prefix-{attended}.f32")),&bytes)?; }
        choices.push(json!({"attended":attended,"argmax":id,"logit":logits[id],"top1_minus_top2":logits[id]-second,
            "logits_blake3":blake3::hash(&bytes).to_hex().to_string()}));
        observer_seconds+=start.elapsed().as_secs_f64(); Ok(())
    })?;
    let generation_seconds=start.elapsed().as_secs_f64();
    ensure!(session.attended_len()==generation.attended_tokens,"generation/cache accounting differs");
    let start=Instant::now();
    save(&args.output.join("generation.bin"),&generation.bytes)?;
    save(&args.output.join("prompt_ids.i32"),&ids.iter().flat_map(|&id|(id as i32).to_le_bytes()).collect::<Vec<_>>())?;
    let diagnostic_files_before_report_seconds=start.elapsed().as_secs_f64();
    let report=json!({"scope":"explicit-ID greedy completion; no template renderer or quality claim",
        "prompt_mode":format!("{:?}",args.prompt_mode),
        "input_contract":"Raw does not parse controls; TrustedFramed parses caller-supplied controls verbatim. Neither renders/escapes a chat template or inserts BOS.",
        "timings":{"process_setup_seconds":process_setup_seconds,"model_init_seconds":model_init_seconds,
            "report_encode_seconds":report_encode_seconds,"generation_seconds":generation_seconds,
            "phases":generation_timings,"observer_seconds":observer_seconds,
            "diagnostic_files_before_report_seconds":diagnostic_files_before_report_seconds,
            "main_entry_to_report_seconds":process_start.elapsed().as_secs_f64(),
            "boundaries":"report_encode_seconds encodes once before generation; phases.encode_validate_seconds validates those IDs only. Prefill and decode end after existing synchronous readback/fences; include streamed weight IO and any first-use compilation; exclude observer and file dumps; model_init includes metadata/tokenizer/CUDA/cache allocation fences, not full weight loading; main_entry excludes dynamic loader; final report write/reset/process teardown only in external process wall time"},
        "backend":format!("{:?}",options.backend),"model_identity":session.identity(),"whole_model_hashed":false,
        "source":session.source_identity(),"pile_path":args.typed_pile.as_ref().map(|_|&args.model),
        "model_path":if args.typed_pile.is_none(){Some(&args.model)}else{None},
        "native_host_timings":"GGUF-specific host counters are absent for native pile, not fabricated zeros",
        "weight_receipts":session.receipt_policy(),
        "capacity":session.capacity(),"cache_admission_bytes":session.admission().cache_bytes,
        "attention_admission_bytes":session.admission().attention_bytes,"trace":args.trace,
        "head_admission":session.head_admission(),"max_head_mib":args.max_head_mib,
        "head_admission_scope":"immutable native GPU head buffers only; not total GPU/runtime/workspace/host memory",
        "head_initialization":"resident native head validation/upload is included in model_init_seconds; immutable buffers survive reset; GGUF stays streamed",
        "prompt_ids":ids,"prompt_bytes_blake3":blake3::hash(&raw).to_hex().to_string(),"choices":choices,"saved_logits":args.save_logits,
        "generation_bytes_blake3":blake3::hash(&generation.bytes).to_hex().to_string(),"result":generation,
        "accounting":"attended_tokens excludes the final emitted token when no subsequent prediction is requested; EOG is neither emitted nor fed back"});
    save(&args.output.join("report.json"),&serde_json::to_vec_pretty(&report)?)?;
    session.reset()?;
    println!("emitted={} attended={} stop={:?} raw_bytes={}",generation.ids.len(),generation.attended_tokens,generation.stop,generation.bytes.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_default_and_explicit_trusted_framing() {
        let raw=Args::try_parse_from(["flint_generate","model","prompt","out"]).unwrap();
        assert_eq!(raw.prompt_mode,PromptMode::Raw);assert!(raw.collection.is_none());
        let framed=Args::try_parse_from(["flint_generate","model","prompt","out","--prompt-mode","trusted-framed"]).unwrap();
        assert_eq!(framed.prompt_mode,PromptMode::TrustedFramed);
        assert!(Args::try_parse_from(["flint_generate","model","prompt","out","--prompt-mode","chat"]).is_err());
        assert!(matches!(raw.native_head,HeadMode::Resident));assert_eq!(raw.max_head_mib,192);
        let streamed=Args::try_parse_from(["flint_generate","pile","prompt","out","--native-head","streamed"]).unwrap();
        assert!(matches!(streamed.native_head,HeadMode::Streamed));
    }
    #[test]
    fn descriptor_requires_native_root_and_is_not_a_root_alias() {
        let descriptor="ab".repeat(32);
        assert!(Args::try_parse_from(["flint_generate","pile","prompt","out","--collection",&descriptor]).is_err());
        let args=Args::try_parse_from(["flint_generate","pile","prompt","out",
            "--typed-pile","0123456789abcdef0123456789abcdef","--collection",&descriptor]).unwrap();
        assert_eq!(args.collection.as_deref(),Some(descriptor.as_str()));
        assert!(mary::model_collection::parse_collection_handle(args.collection.as_deref().unwrap()).is_ok());
        assert!(mary::model_collection::parse_collection_handle(args.typed_pile.as_deref().unwrap()).is_err());
    }
}
