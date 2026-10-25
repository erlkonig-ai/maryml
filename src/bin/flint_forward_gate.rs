//! Fixed token-ID witnesses for the bounded 49-block F32 diagnostic, not chat.
//! Exit0 means execution/invariants completed; numerical/behavior differences
//! remain measurements, not a fitted acceptance threshold or quality claim.
use std::{fs::{self,File,OpenOptions},io::Write,path::{Path,PathBuf}};
use anyhow::{Context,Result,ensure};
use serde_json::{Value,json};
use mary::models::flint::{forward::{self,Backend},math::{Trace,Values},weights::GgufWeights};

fn bytes(path:&Path,max:usize)->Result<Vec<u8>> {
    let before=fs::symlink_metadata(path)?;
    ensure!(before.is_file() && before.len()<=max as u64,"not bounded regular input {}",path.display());
    let value=fs::read(path)?;
    let after=fs::symlink_metadata(path)?;
    ensure!(before.len()==after.len() && before.modified()?==after.modified()? && value.len() as u64==before.len(),"input changed");
    Ok(value)
}
fn ints(raw:&[u8])->Result<Vec<i32>> {
    ensure!(raw.len()%4==0,"unaligned I32");
    Ok(raw.chunks_exact(4).map(|c|i32::from_le_bytes(c.try_into().unwrap())).collect())
}
fn floats(raw:&[u8])->Result<Vec<f32>> {
    ensure!(raw.len()%4==0,"unaligned F32");
    Ok(raw.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect())
}
fn hash(raw:&[u8])->String { blake3::hash(raw).to_hex().to_string() }
fn create(path:&Path)->Result<File> { Ok(OpenOptions::new().write(true).create_new(true).open(path)?) }
fn save(path:&Path,raw:&[u8])->Result<()> { create(path)?.write_all(raw)?; Ok(()) }
fn fbytes(values:&[f32])->Vec<u8> { values.iter().flat_map(|v|v.to_le_bytes()).collect() }
fn metric(a:&[f32],b:&[f32])->Result<Value> {
    ensure!(a.len()==b.len() && !a.is_empty(),"comparison shape differs");
    let mut max=0.0f64; let mut diff=0.0f64; let mut reference=0.0f64; let mut bits=0;
    for (&a,&b) in a.iter().zip(b) {
        ensure!(a.is_finite() && b.is_finite(),"nonfinite residual/logit comparison");
        let d=a as f64-b as f64; max=max.max(d.abs()); diff+=d*d; reference+=(b as f64)*(b as f64);
        bits+=usize::from(a.to_bits()!=b.to_bits());
    }
    Ok(json!({"elements":a.len(),"bit_mismatches":bits,"max_abs":max,
        "rmse":(diff/a.len() as f64).sqrt(),"reference_rms":(reference/a.len() as f64).sqrt(),
        "relative_rms_error":if reference>0.0 {Some((diff/reference).sqrt())} else {None}}))
}
fn margins(probs:&[f32])->Result<Vec<f32>> {
    ensure!(probs.len()%256==0 && probs.iter().all(|v|v.is_finite()),"invalid routing probabilities");
    Ok(probs.chunks_exact(256).map(|row|{
        let mut sorted=row.to_vec(); sorted.sort_by(|a,b|b.total_cmp(a)); sorted[5]-sorted[6]
    }).collect())
}
fn top(logits:&[f32])->Vec<Value> {
    logits.chunks_exact(forward::VOCAB).map(|row|{
        let mut ids:Vec<usize>=(0..row.len()).collect();
        ids.sort_by(|&a,&b|row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        json!({"argmax":ids[0],"top1_minus_top2":row[ids[0]]-row[ids[1]],
            "top20":ids[..20].iter().map(|&id|json!({"id":id,"logit":row[id]})).collect::<Vec<_>>()})
    }).collect()
}
fn compare_f32(out:&Path,reference:&Path,name:&str,values:&[f32])->Result<Value> {
    let actual=fbytes(values);
    let expected=bytes(&reference.join(format!("{name}.f32")),forward::VOCAB*3*4)?;
    let comparison=metric(values,&floats(&expected)?)?;
    save(&out.join(format!("{name}.f32")),&actual)?;
    Ok(json!({"metric":comparison,"actual_blake3":hash(&actual),"reference_blake3":hash(&expected)}))
}

fn main()->Result<()> {
    let args:Vec<PathBuf>=std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(args.len()==6,"usage: flint_forward_gate cpu|cuda|packed-all MODEL TOKEN_IDS_I32 POSITIONS_I32 REFERENCE_DIR NEW_OUTPUT");
    let [mode,model,ids_path,pos_path,reference,out]=args.as_slice() else {unreachable!()};
    let backend=match mode.to_str() {
        Some("cpu")=>Backend::Cpu,Some("cuda")=>Backend::Cuda,Some("packed-all")=>Backend::PackedAll,
        _=>anyhow::bail!("invalid backend"),
    };
    let id_bytes=bytes(ids_path,12)?; let pos_bytes=bytes(pos_path,12)?;
    let ids=ints(&id_bytes)?; let positions=ints(&pos_bytes)?;
    fs::create_dir(out).context("output directory must be new")?;
    let mut records=create(&out.join("blocks.jsonl"))?;
    let mut weights=GgufWeights::open_block(model,0)?;
    let metadata=weights.file.index.metadata.clone();
    let mut first_float_difference=None; let mut first_router_difference=None;
    let result=forward::run(&mut weights,&ids,&positions,backend,|block,trace:&Trace,reads|{
        let name=format!("block-{block:02}");
        let dir=out.join(&name); fs::create_dir(&dir)?;
        let refdir=reference.join(&name);
        let residual=compare_f32(&dir,&refdir,"l_out",forward::stage_f32(trace,"l_out")?)?;
        if first_float_difference.is_none() && residual["metric"]["bit_mismatches"].as_u64()!=Some(0) {
            first_float_difference=Some(block);
        }
        let routing=if let Some(stage)=trace.stages.iter().find(|s|s.name=="router_ids") {
            let Values::I32(actual)=&stage.values else {anyhow::bail!("router IDs not I32")};
            let expected_bytes=bytes(&refdir.join("router_ids.i32"),6*3*4)?;
            let expected=ints(&expected_bytes)?;
            ensure!(actual.len()==positions.len()*6 && actual.len()==expected.len(),"router ID shape differs");
            let different=actual.iter().zip(&expected).filter(|(a,b)|a!=b).count();
            if different!=0 && first_router_difference.is_none() { first_router_difference=Some(block); }
            save(&dir.join("router_ids.i32"),&actual.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            let probs=forward::stage_f32(trace,"router_probs")?;
            let ref_probs=floats(&bytes(&refdir.join("router_probs.f32"),256*3*4)?)?;
            save(&dir.join("router_probs.f32"),&fbytes(probs))?;
            json!({"ordered_ids":actual,"reference_ordered_ids":expected,"id_mismatches":different,
                "sixth_seventh_margins":trace.router_margins,"reference_sixth_seventh_margins":margins(&ref_probs)?,
                "probability_errors":metric(probs,&ref_probs)?})
        } else {Value::Null};
        let config=mary::models::flint::config::BlockConfig::from_metadata(&metadata,block)?;
        let record=json!({"block":block,"config":config,"residual":residual,"routing":routing,
            "reads":reads,"selected_payload_bytes_returned":reads.iter().map(|r|r.absolute_end-r.absolute_start).sum::<u64>()});
        writeln!(records,"{record}")?; records.flush()?;
        println!("block={block} residual_max_abs={} routing_mismatches={}",record["residual"]["metric"]["max_abs"],record["routing"]["id_mismatches"]);
        Ok(())
    })?;
    let globals=out.join("globals"); fs::create_dir(&globals)?;
    let refglobals=reference.join("globals");
    let embedding=compare_f32(&globals,&refglobals,"embedding",&result.embedding)?;
    let residual=compare_f32(&globals,&refglobals,"residual",&result.residual)?;
    let normalized=compare_f32(&globals,&refglobals,"output_norm",&result.normalized)?;
    let logits=compare_f32(&globals,&refglobals,"logits",&result.logits)?;
    let ref_logits=floats(&bytes(&refglobals.join("logits.f32"),forward::VOCAB*3*4)?)?;
    let report=json!({"backend":format!("{backend:?}"),"scope":"49-block empty-history F32 diagnostic, all T output rows in token-major order; no generation or quantized-runtime equivalence",
        "acceptance":"exit0 only certifies completed structural/finite diagnostics; float and ordered-router differences are reported, not fitted quality thresholds",
        "model_path":model,"model_identity":weights.file.identity,"whole_model_freshly_hashed":false,
        "metadata_end":weights.file.index.metadata_end,"gguf_metadata":metadata,
        "token_ids":ids,"positions":positions,"ids_blake3":hash(&id_bytes),"positions_blake3":hash(&pos_bytes),
        "blocks":49,"head_chunk_rows":forward::HEAD_ROWS,"head_chunks":19,"logit_ggml_shape":[forward::VOCAB,positions.len(),1,1],
        "first_float_bit_difference":first_float_difference,"first_ordered_router_difference":first_router_difference,
        "embedding":embedding,"residual":residual,"output_norm":normalized,"logits":logits,
        "top":top(&result.logits),"reference_top":top(&ref_logits),"global_reads":result.global_reads});
    save(&out.join("report.json"),&serde_json::to_vec_pretty(&report)?)?;
    println!("forward_completed=true first_float_bit_difference={first_float_difference:?} first_ordered_router_difference={first_router_difference:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn metrics_are_diagnostics_not_fitted_acceptance() {
        assert_eq!(metric(&[1.0],&[0.0]).unwrap()["max_abs"],1.0);
        assert!(metric(&[f32::NAN],&[0.0]).is_err());
        assert!(metric(&[f32::INFINITY],&[f32::INFINITY]).is_err());
        assert_eq!(margins(&[0.5;256]).unwrap(),vec![0.0]);
    }
}
