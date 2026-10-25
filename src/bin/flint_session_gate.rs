//! Optional model witnesses, not a model-quality or bitwise numeric gate.
//! The external pinned GGML continuation oracle is a separate comparison.
use std::{fs::{self,OpenOptions},io::Write,path::{Path,PathBuf}};
use anyhow::{Context,Result,ensure};
use clap::Parser;
use serde_json::{Value,json};
use mary::models::flint::{forward::{self,Backend,VOCAB,BLOCKS},math::{Trace,Values},
    cache::Limits,session::{FlintSession,Options,greedy_id},weights::GgufWeights};

#[derive(Parser)]
struct Args {
    backend:String, model:PathBuf, output:PathBuf,
    #[arg(long)] typed_pile:Option<String>,
    /// Adds trace-off, partial failure/refusal/reset, and fresh-prefix witnesses.
    #[arg(long)] extended:bool,
}
struct Routing { ids:Vec<Vec<i32>>, margins:Vec<Vec<f32>> }
impl Routing {
    fn new()->Self { Self {ids:vec![Vec::new();BLOCKS],margins:vec![Vec::new();BLOCKS]} }
    fn observe(&mut self,block:usize,trace:&Trace)->Result<()> {
        if block<2 { return Ok(()); }
        let Some(stage)=trace.stages.iter().find(|s|s.name=="router_ids") else {anyhow::bail!("missing routing trace")};
        let Values::I32(ids)=&stage.values else {anyhow::bail!("wrong router trace type")};
        self.ids[block].extend(ids); self.margins[block].extend(&trace.router_margins); Ok(())
    }
    fn compare(&self,reference:&Self,tokens:usize)->Result<Value> {
        let mut mismatches=0;
        for block in 0..BLOCKS {
            let n=if block<2 {0} else {6*tokens};
            ensure!(self.ids[block].len()==n && reference.ids[block].len()>=n,"router comparison shape differs");
            mismatches+=self.ids[block].iter().zip(&reference.ids[block][..n]).filter(|(a,b)|a!=b).count();
        }
        Ok(json!({"ordered_id_mismatches":mismatches,"ids":self.ids,"sixth_seventh_margins":self.margins}))
    }
}
fn compare(a:&[f32],b:&[f32])->Result<Value> {
    ensure!(a.len()==b.len() && a.len()==VOCAB,"logit comparison shape differs");
    let mut max=0.0f64; let mut error=0.0f64; let mut norm=0.0f64;
    for (&a,&b) in a.iter().zip(b) {
        ensure!(a.is_finite() && b.is_finite(),"nonfinite comparison");
        let d=a as f64-b as f64; max=max.max(d.abs()); error+=d*d; norm+=(b as f64).powi(2);
    }
    fn top(x:&[f32])->Result<Value> {
        let id=greedy_id(x)? as usize;
        let second=x.iter().enumerate().filter(|&(i,_)|i!=id).map(|(_,v)|*v).fold(f32::NEG_INFINITY,f32::max);
        Ok(json!({"id":id,"logit":x[id],"top1_minus_top2":x[id]-second}))
    }
    Ok(json!({"max_abs":max,"rmse":(error/a.len() as f64).sqrt(),
        "relative_rms":if norm>0.0 {Some((error/norm).sqrt())} else {None},"actual_top":top(a)?,"reference_top":top(b)?}))
}
fn save(path:&Path,bytes:&[u8])->Result<()> {
    OpenOptions::new().write(true).create_new(true).open(path)?.write_all(bytes)?; Ok(())
}
fn main()->Result<()> {
    let args=Args::parse();
    let backend=match args.backend.as_str() {"cpu"=>Backend::Cpu,"cuda"=>Backend::Cuda,"packed-all"=>Backend::PackedAll,_=>anyhow::bail!("backend must be cpu|cuda|packed-all")};
    fs::create_dir(&args.output)?;
    let ids=[150003u32,9479,13]; let signed=ids.map(|id|id as i32);
    let mut progress=OpenOptions::new().create_new(true).write(true).open(args.output.join("progress.jsonl"))?;
    let mut baseline_routes=Routing::new();
    let root=args.typed_pile.as_ref().map(|s|triblespace::prelude::Id::from_hex(s)
        .context("invalid native model root")).transpose()?;
    ensure!(root.is_none() || matches!(backend,Backend::PackedAll),"native pile requires packed-all");
    let mut source=match root {
        Some(root)=>forward::Source::Native(mary::models::flint::pile::PileWeights::open(&args.model,root)?),
        None=>forward::Source::Gguf(GgufWeights::open_block(&args.model,0)?),
    };
    let baseline=forward::run(&mut source,&signed,&[0,1,2],backend,
        |b,t,_|baseline_routes.observe(b,t))?;
    ensure!(baseline.logits.len()==3*VOCAB,"baseline logits shape");
    save(&args.output.join("uncached-logits.f32"),&baseline.logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
    let options=Options {backend,limits:Limits::default(),trace:true,..Options::default()};
    let mut session=match root {Some(root)=>FlintSession::open_pile(&args.model,root,options)?,
        None=>FlintSession::open(&args.model,options)?};
    let mut records=Vec::new();
    for partition in [vec![1,1,1],vec![2,1]] {
        session.reset()?; let mut first=0; let mut routes=Routing::new();
        for &count in &partition {
            let end=first+count;
            let logits=session.append_observed(&ids[first..end],|b,t,_|routes.observe(b,t))?;
            ensure!(session.attended_len()==end,"committed length mismatch");
            let record=json!({"partition":partition,"prefix":end,"logits":compare(&logits,&baseline.logits[(end-1)*VOCAB..end*VOCAB])?,
                "routing":routes.compare(&baseline_routes,end)?});
            writeln!(progress,"{record}")?; progress.flush()?; records.push(record);
            save(&args.output.join(format!("cached-{}-prefix-{end}.f32",partition.iter().map(|n|n.to_string()).collect::<Vec<_>>().join("-"))),
                &logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            first=end;
        }
    }
    if args.extended {
        session.reset()?; session.set_trace(false);
        let logits=session.prefill(&ids)?;
        records.push(json!({"trace_off_vs_on":compare(&logits,&baseline.logits[2*VOCAB..])?}));
        session.reset()?;
        assert!(session.append_observed(&ids[..1],|block,_,_| {
            ensure!(block!=0,"intentional post-block0 observer failure"); Ok(())
        }).is_err());
        ensure!(session.is_poisoned() && session.attended_len()==0,"partial failure was committed or not poisoned");
        ensure!(session.append(&ids[..1]).is_err(),"poisoned append was allowed");
        session.reset()?; ensure!(!session.is_poisoned() && session.attended_len()==0,"reset did not clear sequence");
        let logits=session.prefill(&ids[..1])?;
        records.push(json!({"reset_after_partial_failure":compare(&logits,&baseline.logits[..VOCAB])?}));
    }
    session.reset()?;
    let report=json!({"backend":args.backend,"model_identity":session.identity(),"source":session.source_identity(),
        "model_path":if root.is_none(){Some(&args.model)}else{None},"pile_path":root.map(|_|&args.model),
        "capacity":session.capacity(),"ids":ids,
        "acceptance":"shape, finite values, capacity/state invariants only; numeric/routing/argmax differences remain diagnostics",
        "independence":"this gate compares shared-core execution partitions; pinned GGML continuation comparison is separate",
        "reference_routing":{"ids":baseline_routes.ids,"margins":baseline_routes.margins},"checks":records,"reset_completed":true});
    save(&args.output.join("report.json"),&serde_json::to_vec_pretty(&report)?)?;
    println!("session_witnesses_completed=true extended={}",args.extended); Ok(())
}
