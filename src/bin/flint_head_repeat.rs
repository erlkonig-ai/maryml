//! Two identical trusted-framed generations in one native session, with reset.
//! A small retained measurement driver, not a model-quality/numerical gate.
use std::{fs::{self,OpenOptions},io::Write,path::{Path,PathBuf},time::Instant};
use anyhow::{Context,Result,ensure};
use clap::{Parser,ValueEnum};
use serde_json::json;
use mary::models::flint::{session::{FlintSession,Options,NativeHead,NativeFixed},cache::Limits};

#[derive(Clone,Copy,Debug,ValueEnum)]
enum Mode { Streamed,Resident }
#[derive(Parser)]
struct Args {
    pile:PathBuf, root:String, collection:String, prompt:PathBuf, output:PathBuf,
    #[arg(long,value_enum)] native_head:Mode,
    #[arg(long,default_value_t=192)] max_head_mib:usize,
    #[arg(long,value_enum,default_value="resident")] native_fixed:Mode,
    #[arg(long,default_value_t=1024)] max_fixed_mib:usize,
    #[arg(long,default_value_t=512)] max_new:usize,
    #[arg(long,default_value_t=1024)] capacity:usize,
}
fn save(path:&Path,bytes:&[u8])->Result<()> {
    OpenOptions::new().write(true).create_new(true).open(path)?.write_all(bytes)?; Ok(())
}
fn main()->Result<()> {
    let args=Args::parse();
    let root=triblespace::prelude::Id::from_hex(&args.root).context("invalid native model root")?;
    let collection=mary::model_collection::parse_collection_handle(&args.collection)?;
    let before=fs::symlink_metadata(&args.prompt)?;
    ensure!(before.is_file() && before.len()<=1<<20,"trusted prompt must be a regular UTF-8 file <=1MiB");
    let prompt=fs::read_to_string(&args.prompt)?;
    let after=fs::symlink_metadata(&args.prompt)?;
    ensure!(prompt.len() as u64==before.len() && before.len()==after.len()
        && before.modified()?==after.modified()?,"prompt changed while reading");
    let options=Options {limits:Limits {capacity:args.capacity,..Limits::default()},
        native_head:match args.native_head {Mode::Streamed=>NativeHead::Streamed,
            Mode::Resident=>NativeHead::Resident {max_bytes:args.max_head_mib.checked_mul(1<<20).context("head MiB overflow")?}},
        native_fixed:match args.native_fixed {Mode::Streamed=>NativeFixed::Streamed,
            Mode::Resident=>NativeFixed::Resident {max_bytes:args.max_fixed_mib.checked_mul(1<<20).context("fixed MiB overflow")?}},
        ..Options::default()};
    fs::create_dir(&args.output).context("measurement output must be new")?;
    let start=Instant::now();
    let mut session=FlintSession::open_pile_with_collection(&args.pile,root,Some(collection),options)?;
    let init_seconds=start.elapsed().as_secs_f64();
    let start=Instant::now();
    let ids=session.encode_trusted_framed(&prompt)?;
    let encode_seconds=start.elapsed().as_secs_f64();
    let head=session.head_admission();
    let fixed=session.fixed_admission();
    // Save initialization even if a later run fails; errors are never retried.
    save(&args.output.join("initialization.json"),&serde_json::to_vec_pretty(&json!({
        "source":session.source_identity(),"pile":args.pile,"mode":format!("{:?}",args.native_head),
        "init_seconds":init_seconds,"encode_seconds":encode_seconds,"head_admission":head,
        "fixed_mode":format!("{:?}",args.native_fixed),"fixed_admission":fixed,"fixed_budget_mib":args.max_fixed_mib,
        "head_budget_mib":args.max_head_mib,"cache_bytes":session.admission().cache_bytes,
        "attention_bytes":session.admission().attention_bytes,"capacity":session.capacity(),
        "prompt_mode":"TrustedFramed","prompt_ids":ids,"max_new":args.max_new,
        "prompt_blake3":blake3::hash(prompt.as_bytes()).to_hex().to_string(),
        "boundaries":"init includes native snapshot, codec, fixed/head validation/upload and KV allocation; separate immutable budgets exclude runtime/allocator padding/query scratch/host memory; no logits dumps or numerical acceptance gate"
    }))?)?;
    for run in 0..2 {
        let start=Instant::now();
        let (generation,timings)=session.generate_ids_observed_timed(&ids,args.max_new,|_,_|Ok(()))?;
        let generation_seconds=start.elapsed().as_secs_f64();
        save(&args.output.join(format!("run-{run}.bin")),&generation.bytes)?;
        save(&args.output.join(format!("run-{run}.json")),&serde_json::to_vec_pretty(&json!({
            "run":run,"generation_seconds":generation_seconds,"timings":timings,"result":generation,
            "output_blake3":blake3::hash(&generation.bytes).to_hex().to_string()
        }))?)?;
        let start=Instant::now();session.reset()?;
        let reset_seconds=start.elapsed().as_secs_f64();
        ensure!(session.attended_len()==0 && !session.is_poisoned(),"reset state differs");
        ensure!(session.head_admission()==head,"reset changed immutable head admission");
        ensure!(session.fixed_admission()==fixed,"reset changed immutable fixed-block admission");
        save(&args.output.join(format!("reset-{run}.json")),&serde_json::to_vec_pretty(&json!({
            "reset_seconds":reset_seconds,"attended":session.attended_len(),"poisoned":session.is_poisoned(),
            "head_admission":session.head_admission(),"fixed_admission":session.fixed_admission()
        }))?)?;
        println!("run={run} emitted={} stop={:?} reset_completed=true",generation.ids.len(),generation.stop);
    }
    Ok(())
}
