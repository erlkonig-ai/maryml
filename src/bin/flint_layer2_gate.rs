//! CPU or CUDA F32/math layer-2 diagnostics against retained reference outputs.
//! Exit success means diagnostics completed with exact integer stage agreement,
//! NOT floating-point numerical acceptance. CUDA uses the `flint-cuda` feature.

use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result, ensure};
use mary::models::flint::{math::{self, Values}, weights::GgufWeights};
use serde_json::{Value, json};

fn read(path: &Path, max: usize) -> Result<Vec<u8>> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(meta.is_file() && meta.len() <= max as u64, "not a bounded regular file: {}",path.display());
    let bytes = fs::read(path)?;
    let after = fs::symlink_metadata(path)?;
    ensure!(bytes.len() as u64 == meta.len() && after.len() == meta.len()
        && after.modified()? == meta.modified()?, "source changed while reading: {}",path.display());
    Ok(bytes)
}
fn floats(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len()%4 == 0,"f32 length is not aligned");
    Ok(bytes.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect())
}
fn hash(bytes: &[u8]) -> String { blake3::hash(bytes).to_hex().to_string() }

fn errors(actual: &[f32], expected: &[f32]) -> Result<Value> {
    ensure!(actual.len() == expected.len(),"comparison length differs");
    let mut max_abs = 0.0f64; let mut sum_diff = 0.0; let mut sum_ref = 0.0;
    let mut max_ref = 0.0f64; let mut finite = 0usize; let mut bits = 0;
    for (&a,&r) in actual.iter().zip(expected) {
        bits += usize::from(a.to_bits() != r.to_bits());
        if !a.is_finite() || !r.is_finite() {
            ensure!(a == r && !a.is_nan(),"nonfinite mismatch");
            continue;
        }
        let diff = (a as f64)-(r as f64);
        max_abs = max_abs.max(diff.abs()); max_ref = max_ref.max((r as f64).abs());
        sum_diff += diff*diff; sum_ref += (r as f64)*(r as f64); finite += 1;
    }
    let rmse = (sum_diff / finite.max(1) as f64).sqrt();
    let ref_rms = (sum_ref / finite.max(1) as f64).sqrt();
    Ok(json!({"elements":actual.len(),"finite_elements":finite,"bit_mismatches":bits,
        "max_abs":max_abs,"rmse":rmse,"reference_rms":ref_rms,"reference_max_abs":max_ref,
        "relative_rms_error":if ref_rms>0.0 {Some(rmse/ref_rms)} else {None},
        "max_abs_over_reference_max":if max_ref>0.0 {Some(max_abs/max_ref)} else {None}}))
}

fn main() -> Result<()> {
    let mut args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let cuda = args.first().is_some_and(|p| p == Path::new("--cuda"));
    if cuda || args.first().is_some_and(|p| p == Path::new("--cpu")) { args.remove(0); }
    ensure!(!cuda || cfg!(feature = "flint-cuda"), "--cuda requires the flint-cuda feature");
    ensure!(args.len()==5,"usage: flint_layer2_gate [--cpu|--cuda] MODEL INPUT_F32 POSITIONS_I32 REFERENCE_DIR NEW_OUTPUT_DIR");
    let [model,input_path,pos_path,reference,out] = args.as_slice() else { unreachable!() };
    let input_bytes = read(input_path,math::H*3*4)?;
    let input = floats(&input_bytes)?;
    let pos_bytes = read(pos_path,3*4)?;
    ensure!(pos_bytes.len()%4==0,"unaligned positions");
    let positions: Vec<i32> = pos_bytes.chunks_exact(4).map(|c|i32::from_le_bytes(c.try_into().unwrap())).collect();
    let table_bytes = read(&reference.join("tensors.tsv"),1024*1024)?;
    let table = std::str::from_utf8(&table_bytes)?;
    fs::create_dir(out).context("output directory must not already exist")?;
    let mut weights = GgufWeights::open(model)?;
    let trace = if cuda {
        #[cfg(feature = "flint-cuda")]
        { mary::models::flint::gpu::layer2(&mut weights,&input,&positions)? }
        #[cfg(not(feature = "flint-cuda"))]
        { unreachable!("feature checked before file reads") }
    } else { math::layer2(&mut weights,&input,&positions)? };
    weights.file.check_identity()?;
    let mut comparisons = Vec::new();
    let mut exact_integer_stages = Vec::new();
    let mut integer_mismatches = 0usize;
    let mut shape_text = String::from("name\ttype\tne0\tne1\tne2\tne3\tbytes\tblake3\n");
    ensure!(table.lines().skip(1).count() == trace.stages.len(),"reference stage inventory differs");
    for stage in &trace.stages {
        let line = table.lines().skip(1).find(|s|s.split('\t').next()==Some(stage.name.as_str()))
            .with_context(||format!("reference stage absent: {}",stage.name))?;
        let cols: Vec<_> = line.split('\t').collect();
        ensure!(cols.len()>=8,"invalid reference shape row");
        let dims: Vec<usize> = cols[2..6].iter().map(|s|s.parse()).collect::<std::result::Result<_,_>>()?;
        ensure!(dims==stage.dims,"shape differs: {}",stage.name);
        let (ty,raw) = match &stage.values {
            Values::F32(v) => ("f32",v.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
            Values::I32(v) => ("i32",v.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
        };
        ensure!(ty==cols[1],"dtype differs: {}",stage.name);
        let filename = format!("{}.{}",stage.name,ty);
        let expected = read(&reference.join(&filename),1024*1024)?;
        ensure!(expected.len()==raw.len() && expected.len()==cols[6].parse::<usize>()?,"byte length differs");
        let metric = match &stage.values {
            Values::F32(v) => errors(v,&floats(&expected)?)?,
            Values::I32(v) => {
                let mismatches = expected.chunks_exact(4).zip(raw.chunks_exact(4)).filter(|(a,b)|a!=b).count();
                integer_mismatches += mismatches;
                exact_integer_stages.push(json!({"stage":stage.name,"elements":v.len(),
                    "mismatches":mismatches,"exact_ordered_match":mismatches==0}));
                json!({"elements":v.len(),"integer_mismatches":mismatches})
            },
        };
        println!("{} {}",stage.name,metric);
        fs::write(out.join(filename),&raw)?;
        shape_text.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",stage.name,ty,
            dims[0],dims[1],dims[2],dims[3],raw.len(),hash(&raw)));
        comparisons.push(json!({"stage":stage.name,"ggml_dims":dims,"type":ty,"metric":metric,
            "actual_blake3":hash(&raw),"reference_blake3":hash(&expected),
            "reference_manifest_digest_claim":cols[7]}));
    }
    let ref_probs = floats(&read(&reference.join("router_probs.f32"),256*3*4)?)?;
    let reference_margins: Vec<f32> = ref_probs.chunks_exact(256).map(|row|{
        let mut sorted = row.to_vec(); sorted.sort_by(|a,b|b.total_cmp(a)); sorted[5]-sorted[6]
    }).collect();
    let selected_bytes: u64 = weights.reads.iter().map(|r|r.absolute_end-r.absolute_start).sum();
    let scope = if cuda {
        "layer 2 CubeCL CUDA F32 decoded-weight math; per-stage readbacks and CPU-controlled sparse loading; no BF16 or activation quantization"
    } else { "layer 2 CPU F32/math only; wider scalar dot/RMS reductions; no BF16 or activation quantization" };
    let arithmetic = if cuda {
        "F32 values and accumulation; matvec 32-lane strided/tree reduction; RMS 256-lane strided/tree reduction; serial F32 attention/router sums; GPU transcendentals; CUDA compiler may contract multiply-add; fixed kernels without autotune"
    } else { "F32 operation boundaries with f64 scalar dot/RMS and attention denominator/division reductions; CPU transcendentals" };
    let report = json!({"scope":scope,"arithmetic":arithmetic,"backend":if cuda {"cuda"} else {"cpu"},
        "acceptance":"diagnostics only: exact ordered integer stages required; floating errors reported without fitted tolerance or numerical acceptance claim",
        "exact_integer_stages":exact_integer_stages,"integer_stages_pass":integer_mismatches==0,
        "model_path":model,"model_identity":weights.file.identity,"whole_model_freshly_hashed":false,
        "input_path":input_path,"input_blake3":hash(&input_bytes),"positions_path":pos_path,"positions_blake3":hash(&pos_bytes),
        "reference_dir":reference,"reference_table_blake3":hash(&table_bytes),"positions":positions,
        "router_sixth_seventh_margins":trace.router_margins,"reference_router_sixth_seventh_margins":reference_margins,
        "selected_payload_bytes_returned":selected_bytes,"read_accounting":"explicit row returns including repeats; excludes header/buffering/OS read-ahead",
        "reads":weights.reads,"comparisons":comparisons});
    fs::write(out.join("tensors.tsv"),shape_text)?;
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    ensure!(integer_mismatches==0,"integer stage mismatch (including ordered expert IDs); diagnostics retained in {}",out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn metrics_do_not_hide_nonfinite_or_zero_reference() {
        assert!(errors(&[f32::NAN],&[0.0]).is_err());
        assert!(errors(&[f32::INFINITY],&[0.0]).is_err());
        assert!(errors(&[f32::NEG_INFINITY],&[f32::NEG_INFINITY]).is_ok());
        let zero = errors(&[1.0],&[0.0]).unwrap();
        assert_eq!(zero["max_abs"],1.0); assert!(zero["relative_rms_error"].is_null());
    }
}
