//! Bounded public-session wiring witness, not a steering/affect experiment.
//! Three independent generations: None, explicit zero without capture, and
//! capture-only with NO direction. No nonzero model intervention is exposed.
use std::{cell::Cell,fs::{self,OpenOptions},io::{Read,Write},path::{Path,PathBuf},time::Instant};
use anyhow::{Context,Result,ensure};
use clap::Parser;
use serde::Serialize;
use serde_json::{Value,json};
use mary::models::flint::{cache::Limits,forward::{Backend,VOCAB},math::H,
    residual::{Capture,CaptureSink,CaptureStage,CapturedRow,Phase,Plan,Positions,Request,Steering},
    session::{FlintSession,Generation,NativeFixed,NativeHead,Options,greedy_id}};

const PROMPT:&str="A small bird rests on a branch.";
const PROMPT_TOKENS:usize=4;
const MAX_NEW:usize=2;
const MAX_ATTENDED:usize=PROMPT_TOKENS+MAX_NEW-1;
const BLOCKS:[usize;4]=[0,1,4,48];
const MAX_ROWS:usize=BLOCKS.len()*MAX_ATTENDED*2;
const CAPTURE_BYTES:usize=MAX_ROWS*H*4;
const OUTPUT_CAP:usize=32<<20;

#[derive(Parser)]
struct Args {
    pile:PathBuf, root:String, collection:String,
    /// Retained exact source_identity JSON, including expected support handles.
    expected_source:PathBuf,
    output:PathBuf,
}
// Request-local artifact accounting, not session/model state. Counts attempted
// writes too; on failure no retry can overwrite or evade the finite ceiling.
#[derive(Default)]
struct Output { bytes:Cell<usize> }
impl Output {
    fn reserve(&self,bytes:usize)->Result<()> {
        let end=self.bytes.get().checked_add(bytes).context("artifact byte overflow")?;
        ensure!(end<=OUTPUT_CAP,"probe artifacts exceed32MiB"); self.bytes.set(end); Ok(())
    }
    fn save(&self,path:&Path,bytes:&[u8])->Result<()> {
        self.reserve(bytes.len())?;
        OpenOptions::new().write(true).create_new(true).open(path)?.write_all(bytes)?; Ok(())
    }
    fn json(&self,path:&Path,value:&impl Serialize)->Result<()> {self.save(path,&serde_json::to_vec_pretty(value)?)}
    fn f32(&self,path:&Path,values:&[f32])->Result<()> {
        let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect(); self.save(path,&bytes)
    }
}
#[derive(Debug,Serialize)]
struct Errors { values:usize, unequal_values:usize, unequal_bits:usize, max_abs:f64, max_rel:f64 }
fn errors(a:&[f32],b:&[f32])->Result<Errors> {
    ensure!(a.len()==b.len() && a.iter().chain(b).all(|v|v.is_finite()),"nonfinite or mismatched comparison geometry");
    let mut out=Errors {values:a.len(),unequal_values:0,unequal_bits:0,max_abs:0.0,max_rel:0.0};
    for (&a,&b) in a.iter().zip(b) {
        let delta=(f64::from(a)-f64::from(b)).abs();
        out.max_abs=out.max_abs.max(delta);
        out.max_rel=out.max_rel.max(delta/f64::from(a.abs().max(b.abs())).max(1e-30));
        out.unequal_values+=usize::from(a!=b); out.unequal_bits+=usize::from(a.to_bits()!=b.to_bits());
    }
    Ok(out)
}
struct Observation { attended:usize, argmax:u32, values:Vec<f32> }
#[derive(Serialize)]
struct Row {
    block:usize, absolute_position:usize, input_token_id:u32, phase:Phase, stage:CaptureStage,
    file:String, #[serde(skip)] values:Vec<f32>,
}
fn capture_summary(rows:&[Row],generation:&Generation,prompt:&[u32])->Result<Value> {
    ensure!((PROMPT_TOKENS..=MAX_ATTENDED).contains(&generation.attended_tokens),"unexpected attended horizon");
    ensure!(rows.len()==BLOCKS.len()*generation.attended_tokens*2,"capture count differs from actual attended rows");
    let mut pairs=Vec::new();
    for block in BLOCKS {
        for position in 0..generation.attended_tokens {
            let expected_id=if position<PROMPT_TOKENS {prompt[position]} else {
                *generation.ids.first().context("decode row without emitted input")?
            };
            let expected_phase=if position<PROMPT_TOKENS {Phase::Prefill} else {Phase::Decode};
            let matching:Vec<_>=rows.iter().filter(|r|r.block==block && r.absolute_position==position).collect();
            ensure!(matching.len()==2 && matching.iter().all(|r|r.input_token_id==expected_id && r.phase==expected_phase),
                "capture row identity/phase differs");
            let pre=matching.iter().find(|r|r.stage==CaptureStage::PreAddLout).context("missing pre-add row")?;
            let post=matching.iter().find(|r|r.stage==CaptureStage::PostAddLout).context("missing post-add row")?;
            pairs.push(json!({"block":block,"absolute_position":position,"errors":errors(&pre.values,&post.values)?}));
        }
    }
    Ok(json!({"actual_rows":rows.len(),"actual_f32_bytes":rows.len()*H*4,
        "decode_exercised":generation.attended_tokens>PROMPT_TOKENS,"pre_post":pairs}))
}
fn run(args:&Args,output:&Output)->Result<()> {
    let root=triblespace::prelude::Id::from_hex(&args.root).context("invalid root")?;
    let collection=mary::model_collection::parse_collection_handle(&args.collection)?;
    let metadata=fs::symlink_metadata(&args.expected_source)?;
    ensure!(metadata.is_file() && metadata.len()<=8192,"expected source must be a regular JSON file <=8192 bytes");
    let mut expected_bytes=Vec::new(); fs::File::open(&args.expected_source)?.take(8193).read_to_end(&mut expected_bytes)?;
    ensure!(expected_bytes.len()<=8192,"expected source JSON grew beyond bound");
    let expected:Value=serde_json::from_slice(&expected_bytes)?;
    let numeric_bytes=3usize.checked_mul(MAX_NEW).and_then(|n|n.checked_mul(VOCAB)).and_then(|n|n.checked_mul(4))
        .and_then(|n|n.checked_add(CAPTURE_BYTES)).context("numeric artifact geometry overflow")?;
    ensure!(numeric_bytes<OUTPUT_CAP,"numeric geometry exceeds artifact ceiling");
    output.json(&args.output.join("settings.json"),&json!({
        "pile":args.pile,"root":args.root,"collection":args.collection,"expected_source":expected,
        "prompt_raw":PROMPT,"prompt_tokens":PROMPT_TOKENS,"max_new":MAX_NEW,"capacity":1024,
        "backend":"PackedAll","trace":false,"native_head":"resident","head_cap_bytes":192usize<<20,
        "native_fixed":"resident","fixed_cap_bytes":1usize<<30,"capture_blocks":BLOCKS,
        "capture_positions":"AllAttended","capture_phases":["PreAddLout","PostAddLout"],
        "capture_cap_bytes":CAPTURE_BYTES,"capture_cap_rows":MAX_ROWS,
        "logits_cap_bytes":3*MAX_NEW*VOCAB*4,"total_artifact_cap_bytes":OUTPUT_CAP,
        "explicit_zero_alpha":0.0,"external_alpha_cap":0.0,
        "limits":"logical payload caps exclude runtime/allocator/host overhead; no affect or performance claim"
    }))?;
    let start=Instant::now();
    let mut session=FlintSession::open_pile_with_collection(&args.pile,root,Some(collection),Options {
        backend:Backend::PackedAll,trace:false,limits:Limits {capacity:1024,..Limits::default()},
        native_head:NativeHead::Resident {max_bytes:192<<20},
        native_fixed:NativeFixed::Resident {max_bytes:1<<30},
    })?;
    let init_seconds=start.elapsed().as_secs_f64();
    let source=session.source_identity();
    output.json(&args.output.join("initialization.json"),&json!({"source":source,"init_seconds":init_seconds,
        "head":session.head_admission(),"fixed":session.fixed_admission(),
        "cache_bytes":session.admission().cache_bytes,"attention_bytes":session.admission().attention_bytes}))?;
    ensure!(source==expected,"native source observation differs from retained root/collection/support");
    let start=Instant::now(); let encoded=session.encode_raw(PROMPT)?;
    let encode_seconds=start.elapsed().as_secs_f64();
    output.json(&args.output.join("encoded.json"),&json!({"raw":PROMPT,"ids":encoded,"encode_seconds":encode_seconds}))?;
    ensure!(encoded.len()>=PROMPT_TOKENS && encoded.iter().all(|&id|(id as usize)<VOCAB),"raw encoding needs >=4 valid IDs");
    let ids=&encoded[..PROMPT_TOKENS];
    output.json(&args.output.join("selected-input.json"),&json!({"ids":ids,"absolute_positions":[0,1,2,3],
        "prefill_chunks":[3,1],"meaning":"first four actual raw tokens; not necessarily a complete sentence"}))?;
    let direction=vec![1.0;H]; // Only used by the explicit zero/capture-off pass.
    let head=session.head_admission(); let fixed=session.fixed_admission();
    let mut baseline:Option<(Generation,Vec<Observation>)>=None;
    let mut summaries=Vec::new();
    for mode in ["none","zero-off","capture-only"] {
        let dir=args.output.join(mode); fs::create_dir(&dir)?;
        output.json(&dir.join("before.json"),&json!({"attended":session.attended_len(),"poisoned":session.is_poisoned(),
            "generation_resets_before_prefill":true,"source":source}))?;
        let mut rows=Vec::new(); let mut observations=Vec::new();
        let mut emit=|row:CapturedRow<'_>|->Result<()> {
            ensure!(rows.len()<MAX_ROWS && row.values.len()==H && row.values.iter().all(|v|v.is_finite()),"invalid/cap-exceeding capture");
            ensure!(row.source_identity==&source,"capture source changed");
            let file=format!("capture-{:02}.f32",rows.len());
            output.f32(&dir.join(&file),row.values)?;
            let owned=Row {block:row.block,absolute_position:row.absolute_position,input_token_id:row.input_token_id,
                phase:row.phase,stage:row.stage,file,values:row.values.to_vec()};
            output.json(&dir.join(format!("capture-{:02}.json",rows.len())),&owned)?;
            rows.push(owned); Ok(())
        };
        let request=match mode {
            "none"=>None,
            "zero-off"=>Some(Request {plan:Plan {source_identity:&source,capture:None,
                steering:Some(Steering {block:4,positions:Positions::AllAttended,direction:&direction,alpha:0.0,cap:0.0})},sink:None}),
            _=>Some(Request {plan:Plan {source_identity:&source,steering:None,
                capture:Some(Capture {blocks:&BLOCKS,positions:Positions::AllAttended,post_add:true})},
                sink:Some(CaptureSink::new(CAPTURE_BYTES,&mut emit))}),
        };
        let start=Instant::now();
        let result=session.generate_ids_observed_timed_with_residual(ids,MAX_NEW,request,|attended,values| {
            ensure!(observations.len()<MAX_NEW && (PROMPT_TOKENS..=MAX_ATTENDED).contains(&attended),"unexpected prediction count/horizon");
            let argmax=greedy_id(values)?;
            let index=observations.len(); output.f32(&dir.join(format!("logits-{index}.f32")),values)?;
            output.json(&dir.join(format!("logits-{index}.json")),&json!({"attended_tokens":attended,"argmax":argmax,"values":values.len()}))?;
            observations.push(Observation {attended,argmax,values:values.to_vec()}); Ok(())
        });
        let wall_seconds=start.elapsed().as_secs_f64();
        let (generation,timings)=match result {
            Ok(value)=>value,
            Err(error)=>{
                let partial=json!({"error":format!("{error:#}"),"wall_seconds":wall_seconds,
                    "attended":session.attended_len(),"poisoned":session.is_poisoned(),"captured_rows":rows.len(),
                    "observers":observations.len(),"partial_files_retained":true});
                if let Err(save_error)=output.json(&dir.join("failed.json"),&partial) {eprintln!("saving partial failed: {save_error:#}");}
                return Err(error);
            }
        };
        output.save(&dir.join("generation.bin"),&generation.bytes)?;
        output.json(&dir.join("generation.json"),&json!({"generation":generation,"timings":timings,"wall_seconds":wall_seconds}))?;
        let capture=if mode=="capture-only" {Some(capture_summary(&rows,&generation,ids)?)} else {
            ensure!(rows.is_empty(),"disabled capture produced rows"); None
        };
        let comparison=if let Some((base,base_logits))=&baseline {
            let mut logits=Vec::new();
            for (index,observed) in observations.iter().enumerate() {
                let reference=base_logits.get(index);
                let comparable=reference.is_some_and(|r|r.attended==observed.attended);
                let metrics=if comparable {Some(errors(&reference.unwrap().values,&observed.values)?)} else {None};
                logits.push(json!({"observer":index,"attended_tokens":observed.attended,"argmax":observed.argmax,
                    "baseline_attended":reference.map(|r|r.attended),"baseline_argmax":reference.map(|r|r.argmax),"errors":metrics}));
            }
            Some(json!({"ids_equal":generation.ids==base.ids,"bytes_equal":generation.bytes==base.bytes,
                "stop_equal":generation.stop==base.stop,"baseline_observers":base_logits.len(),"actual_observers":observations.len(),"logits":logits}))
        } else {None};
        let summary=json!({"mode":mode,"comparison":comparison,"capture":capture,
            "attended_tokens":generation.attended_tokens,"emitted_tokens":generation.ids.len(),"stop":generation.stop,
            "decode_exercised":generation.attended_tokens>PROMPT_TOKENS,"wall_seconds":wall_seconds,
            "timing_boundary":"prefill/decode include residual callback IO; greedy host interval includes full-logit observer IO"});
        output.json(&dir.join("summary.json"),&summary)?; summaries.push(summary);
        let start=Instant::now(); session.reset()?;
        output.json(&dir.join("reset.json"),&json!({"seconds":start.elapsed().as_secs_f64(),
            "attended":session.attended_len(),"poisoned":session.is_poisoned(),"head":session.head_admission(),"fixed":session.fixed_admission()}))?;
        ensure!(session.attended_len()==0 && !session.is_poisoned() && session.head_admission()==head && session.fixed_admission()==fixed,
            "reset state/admission differs");
        if baseline.is_none() {baseline=Some((generation,observations));}
    }
    output.json(&args.output.join("report.json"),&json!({"source":source,"runs":summaries,
        "artifact_bytes_before_report":output.bytes.get(),"artifact_cap_bytes":OUTPUT_CAP,
        "numerics":"errors/equalities descriptive, no fitted tolerance or bitwise acceptance requirement",
        "limits":"one short raw prefix; no nonzero steering, affect claim, dataset fitting or performance claim"}))?;
    Ok(())
}
fn main()->Result<()> {
    let args=Args::parse(); fs::create_dir(&args.output).context("output must be new")?;
    let output=Output::default(); let result=run(&args,&output);
    if let Err(error)=&result {
        if let Err(save_error)=output.json(&args.output.join("failure.json"),&json!({"error":format!("{error:#}"),"partials_retained":true})) {
            eprintln!("saving failure failed: {save_error:#}");
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn artifacts_refuse_cap_and_overflow() ->Result<()> {
        let output=Output::default(); output.reserve(OUTPUT_CAP)?;
        assert!(output.reserve(1).is_err()); assert_eq!(output.bytes.get(),OUTPUT_CAP);
        output.bytes.set(usize::MAX); assert!(output.reserve(1).is_err()); Ok(())
    }
    #[test] fn comparison_is_descriptive_and_finite() ->Result<()> {
        let e=errors(&[0.0,1.0,2.0],&[-0.0,1.5,2.0])?;
        assert_eq!(e.unequal_values,1); assert_eq!(e.unequal_bits,2); assert_eq!(e.max_abs,0.5);
        assert!((e.max_rel-1.0/3.0).abs()<1e-12);
        assert!(errors(&[f32::NAN],&[0.0]).is_err()); assert!(errors(&[],&[0.0]).is_err()); Ok(())
    }
    #[test] fn capture_budget_and_early_eog_use_attended_inputs() ->Result<()> {
        assert_eq!(MAX_ROWS,40); assert_eq!(CAPTURE_BYTES,327680);
        let prompt=[10,20,30,40];
        for attended in [4,5] {
            let generation=Generation {ids:if attended==4 {vec![]} else {vec![50,60]},bytes:vec![],prompt_tokens:4,attended_tokens:attended,
                stop:if attended==4 {mary::models::flint::session::Stop::Eog(150003)} else {mary::models::flint::session::Stop::Budget}};
            let mut rows=Vec::new();
            for block in BLOCKS {for pos in 0..attended {for stage in [CaptureStage::PreAddLout,CaptureStage::PostAddLout] {
                rows.push(Row {block,absolute_position:pos,input_token_id:if pos<4 {prompt[pos]} else {50},
                    phase:if pos<4 {Phase::Prefill} else {Phase::Decode},stage,file:String::new(),values:vec![1.0;H]});
            }}}
            let summary=capture_summary(&rows,&generation,&prompt)?;
            assert_eq!(summary["actual_rows"],attended*8); assert_eq!(summary["decode_exercised"],attended==5);
            rows.pop(); assert!(capture_summary(&rows,&generation,&prompt).is_err());
        }
        Ok(())
    }
}
