//! Raw greedy completion. Requires explicit model, prompt file and new output dir.
use std::{fs::{self,OpenOptions},io::Write,path::{Path,PathBuf}};
use anyhow::{Context,Result,ensure};
use clap::{Parser,ValueEnum};
use serde_json::json;
use mary::models::flint::{cache::Limits,forward::Backend,session::{FlintSession,Options,greedy_id}};

#[derive(Clone,Copy,ValueEnum)]
enum Mode { Cpu,Cuda,PackedAll }
#[derive(Parser)]
struct Args {
    model:PathBuf, prompt:PathBuf, output:PathBuf,
    #[arg(long,value_enum,default_value="packed-all")] backend:Mode,
    #[arg(long,default_value_t=1024)] capacity:usize,
    #[arg(long,default_value_t=8)] max_new:usize,
    #[arg(long,default_value_t=256)] max_cache_mib:usize,
    #[arg(long,default_value_t=64)] max_attention_mib:usize,
    #[arg(long)] trace:bool,
    /// Saves each last-row vocabulary distribution for independent prefix replay.
    #[arg(long)] save_logits:bool,
}
fn save(path:&Path,bytes:&[u8])->Result<()> {
    OpenOptions::new().create_new(true).write(true).open(path)?.write_all(bytes)?; Ok(())
}
fn main()->Result<()> {
    let args=Args::parse();
    let before=fs::symlink_metadata(&args.prompt)?;
    ensure!(before.is_file() && before.len()<=1024*1024,"prompt must be a regular UTF-8 file <=1MiB");
    let raw=fs::read(&args.prompt)?;
    let after=fs::symlink_metadata(&args.prompt)?;
    ensure!(raw.len() as u64==before.len() && before.len()==after.len() && before.modified()?==after.modified()?,"prompt changed while reading");
    let prompt=std::str::from_utf8(&raw).context("prompt is not UTF-8")?;
    let options=Options {backend:match args.backend {Mode::Cpu=>Backend::Cpu,Mode::Cuda=>Backend::Cuda,Mode::PackedAll=>Backend::PackedAll},
        limits:Limits {capacity:args.capacity,max_cache_bytes:args.max_cache_mib.checked_mul(1<<20).context("KV MiB overflow")?,
            max_attention_bytes:args.max_attention_mib.checked_mul(1<<20).context("attention MiB overflow")?},trace:args.trace};
    fs::create_dir(&args.output).context("output directory must be new")?;
    let mut session=FlintSession::open(&args.model,options)?;
    let ids=session.encode_raw(prompt)?;
    let mut choices=Vec::new();
    let generation=session.generate_raw_observed(prompt,args.max_new,|attended,logits| {
        let id=greedy_id(logits)? as usize;
        let second=logits.iter().enumerate().filter(|&(i,_)|i!=id).map(|(_,v)|*v).fold(f32::NEG_INFINITY,f32::max);
        let bytes:Vec<u8>=logits.iter().flat_map(|v|v.to_le_bytes()).collect();
        if args.save_logits { save(&args.output.join(format!("logits-prefix-{attended}.f32")),&bytes)?; }
        choices.push(json!({"attended":attended,"argmax":id,"logit":logits[id],"top1_minus_top2":logits[id]-second,
            "logits_blake3":blake3::hash(&bytes).to_hex().to_string()})); Ok(())
    })?;
    ensure!(session.attended_len()==generation.attended_tokens,"generation/cache accounting differs");
    save(&args.output.join("generation.bin"),&generation.bytes)?;
    save(&args.output.join("prompt_ids.i32"),&ids.iter().flat_map(|&id|(id as i32).to_le_bytes()).collect::<Vec<_>>())?;
    let report=json!({"scope":"raw greedy completion; no chat template, quality or performance claim",
        "backend":format!("{:?}",options.backend),"model_identity":session.identity(),"whole_model_hashed":false,
        "capacity":session.capacity(),"cache_admission_bytes":session.admission().cache_bytes,
        "attention_admission_bytes":session.admission().attention_bytes,"trace":args.trace,
        "prompt_ids":ids,"prompt_bytes_blake3":blake3::hash(&raw).to_hex().to_string(),"choices":choices,"saved_logits":args.save_logits,
        "generation_bytes_blake3":blake3::hash(&generation.bytes).to_hex().to_string(),"result":generation,
        "accounting":"attended_tokens excludes the final emitted token when no subsequent prediction is requested; EOG is neither emitted nor fed back"});
    save(&args.output.join("report.json"),&serde_json::to_vec_pretty(&report)?)?;
    session.reset()?;
    println!("emitted={} attended={} stop={:?} raw_bytes={}",generation.ids.len(),generation.attended_tokens,generation.stop,generation.bytes.len());
    Ok(())
}
