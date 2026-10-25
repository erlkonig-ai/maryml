//! Shared 49-block composition for empty-history diagnostics and cached sessions.
//! CPU residual handoff and bounded streamed weights are deliberate first-stage
//! choices, not a resident-weight/performance claim.

use anyhow::{Context, Result, ensure};
use super::{math::{self, Trace, Values, H}, weights::{GgufWeights, ReadReceipt}};
use crate::formats::gguf::nvfp4;

pub const VOCAB: usize = 150144;
pub const HEAD_ROWS: usize = 8192;
pub const BLOCKS: usize = 49;

#[derive(Clone, Copy, Debug)]
pub enum Backend { Cpu, Cuda, PackedAll }
pub(super) enum Head { All, Last, Skip }

pub struct Output {
    pub embedding: Vec<f32>, pub residual: Vec<f32>, pub normalized: Vec<f32>,
    pub logits: Vec<f32>, pub global_reads: Vec<ReadReceipt>,
}

pub fn stage_f32<'a>(trace: &'a Trace, name: &str) -> Result<&'a [f32]> {
    match &trace.stages.iter().find(|s|s.name==name).with_context(||format!("missing stage {name}"))?.values {
        Values::F32(v) => Ok(v), _ => anyhow::bail!("expected F32 stage {name}"),
    }
}

fn f32_rows(w: &mut GgufWeights, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let raw=w.global_rows(name,dims,0,0,1)?;
    let values: Vec<f32>=raw.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect();
    ensure!(values.iter().all(|v|v.is_finite()),"nonfinite global tensor {name}");
    Ok(values)
}

pub fn chunks(rows: usize) -> impl Iterator<Item=(usize,usize)> {
    (0..rows).step_by(HEAD_ROWS).map(move |first|(first,(rows-first).min(HEAD_ROWS)))
}

fn insert_chunk(logits: &mut [f32], values: &[f32], vocab: usize, first: usize, rows: usize, t: usize) -> Result<()> {
    ensure!(rows>0 && first.checked_add(rows).is_some_and(|end|end<=vocab)
        && logits.len()==vocab*t && values.len()==rows*t,"head chunk output geometry differs");
    for token in 0..t {
        logits[token*vocab+first..token*vocab+first+rows].copy_from_slice(&values[token*rows..(token+1)*rows]);
    }
    Ok(())
}

/// Observer borrows one completed block at a time. It must not launch work on
/// weights while the trace is being examined. Runtime errors leave no KV state.
pub fn run(w: &mut GgufWeights, ids: &[i32], positions: &[i32], backend: Backend,
    observe: impl FnMut(usize,&Trace,&[ReadReceipt])->Result<()>) -> Result<Output> {
    run_core(w,ids,positions,backend,None,true,Head::All,observe)
}

pub(super) fn run_core(w:&mut GgufWeights,ids:&[i32],positions:&[i32],backend:Backend,
    mut caches:Option<&mut [super::cache::LayerCache]>,trace:bool,head:Head,
    mut observe:impl FnMut(usize,&Trace,&[ReadReceipt])->Result<()>)->Result<Output> {
    let t=ids.len();
    ensure!((1..=3).contains(&t) && positions.len()==t,"expected T1..3 token IDs and positions");
    ensure!(ids.iter().all(|&id|id>=0 && (id as usize)<VOCAB),"token ID outside vocabulary");
    ensure!(positions.iter().all(|&p|p>=0) && positions.windows(2).all(|p|p[0]<p[1]),"invalid positions");
    ensure!(matches!(backend,Backend::Cpu) || cfg!(feature="flint-cuda"),"CUDA feature required");
    if let Some(c)=caches.as_ref() { ensure!(c.len()==BLOCKS,"expected all49 layer caches"); }
    ensure!(!w.file.index.tensors.iter().any(|info|
        info.name=="output.input_scale" || (info.name.starts_with("blk.") && info.name.ends_with(".bias"))),
        "unsupported input scale or block bias tensor");
    let mut embedding=Vec::with_capacity(t*H);
    for &id in ids {
        let raw=w.global_rows("token_embd.weight",&[H,VOCAB],30,id as usize,1)?;
        embedding.extend(raw.chunks_exact(2).map(|c|
            f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32)<<16)));
    }
    ensure!(embedding.iter().all(|v|v.is_finite()),"nonfinite embedding");
    let mut global_reads=std::mem::take(&mut w.reads);
    let mut residual=embedding.clone();
    for block in 0..BLOCKS {
        w.select_block(block)?;
        let config=w.config;
        let cache=caches.as_deref_mut().map(|c|&mut c[block]);
        let result=match backend {
            Backend::Cpu => {
                let cache=match cache {
                    Some(super::cache::LayerCache::Cpu(c))=>Some(c),None=>None,
                    #[cfg(feature="flint-cuda")] _=>anyhow::bail!("CPU/cache backend mismatch"),
                };
                math::block_core(w,&residual,positions,config,cache,trace)?
            },
            #[cfg(feature="flint-cuda")]
            Backend::Cuda | Backend::PackedAll => {
                use super::gpu::{self,ProjectionMode};
                let mode=if matches!(backend,Backend::PackedAll) {ProjectionMode::PackedAll} else {ProjectionMode::Decoded};
                let cache=match cache {
                    Some(super::cache::LayerCache::Cuda(c))=>Some(c),None=>None,
                    _=>anyhow::bail!("CUDA/cache backend mismatch"),
                };
                gpu::block_core(w,&residual,positions,config,mode,cache,trace)?
            },
            #[cfg(not(feature="flint-cuda"))]
            _ => unreachable!("feature checked before row reads"),
        };
        residual=result.residual;
        ensure!(residual.len()==H*t && residual.iter().all(|v|v.is_finite()),"nonfinite/invalid residual at block {block}");
        observe(block,&result.trace,&w.reads)?;
        w.reads.clear();
        // The full trace and its already-fenced GPU work are dropped here.
    }
    let head_residual=match head { Head::All=>residual.as_slice(),Head::Last=>&residual[(t-1)*H..],Head::Skip=>&[] };
    let (normalized,logits)=if head_residual.is_empty() { (Vec::new(),Vec::new()) }
        else { output_head(w,head_residual,backend)? };
    global_reads.extend(std::mem::take(&mut w.reads));
    w.file.check_identity()?;
    Ok(Output {embedding,residual,normalized,logits,global_reads})
}

fn output_head(w:&mut GgufWeights,residual:&[f32],backend:Backend)->Result<(Vec<f32>,Vec<f32>)> {
    let t=residual.len()/H;
    ensure!((1..=3).contains(&t) && residual.len()==t*H,"invalid head query length");
    let gain=f32_rows(w,"output_norm.weight",&[H])?;
    let scale=f32_rows(w,"output.scale",&[1])?[0];
    let mut logits=vec![0.0;VOCAB*t];
    let normalized;
    match backend {
        Backend::Cpu => {
            normalized=math::norm(&residual,&gain);
            for (first,rows) in chunks(VOCAB) {
                let raw=w.global_rows("output.weight",&[H,VOCAB],40,first,rows)?;
                let decoded=nvfp4::decode(&raw,H*rows)?;
                let values: Vec<f32>=normalized.chunks_exact(H).flat_map(|x|
                    math::matvec(&decoded,x,H).into_iter().map(|v|v*scale)).collect();
                insert_chunk(&mut logits,&values,VOCAB,first,rows,t)?;
            }
        },
        #[cfg(feature="flint-cuda")]
        Backend::Cuda | Backend::PackedAll => {
            use super::gpu::{OutputHead,ProjectionMode};
            let head=OutputHead::new(&residual,&gain)?;
            normalized=head.normalized()?;
            let mode=if matches!(backend,Backend::PackedAll) {ProjectionMode::PackedAll} else {ProjectionMode::Decoded};
            for (first,rows) in chunks(VOCAB) {
                let raw=w.global_rows("output.weight",&[H,VOCAB],40,first,rows)?;
                let values=head.chunk(&raw,rows,scale,mode)?;
                insert_chunk(&mut logits,&values,VOCAB,first,rows,t)?;
            }
        },
        #[cfg(not(feature="flint-cuda"))]
        _ => unreachable!("feature checked before row reads"),
    }
    ensure!(normalized.iter().chain(&logits).all(|v|v.is_finite()),"nonfinite final norm/logits");
    Ok((normalized,logits))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn head_partition_and_token_order() -> Result<()> {
        let parts: Vec<_>=chunks(VOCAB).collect();
        assert_eq!(parts.len(),19); assert_eq!(parts.last(),Some(&(147456,2688)));
        assert_eq!(parts.iter().map(|&(_,rows)|rows).sum::<usize>(),VOCAB);
        for t in [1,3] {
            let vocab=HEAD_ROWS+3;
            let mut output=vec![f32::NAN;vocab*t];
            for (first,rows) in chunks(vocab) {
                let values: Vec<_>=(0..t).flat_map(|token|(0..rows).map(move |row|
                    (token*vocab+first+row) as f32*0.75)).collect();
                insert_chunk(&mut output,&values,vocab,first,rows,t)?;
            }
            assert_eq!(output,(0..vocab*t).map(|i|i as f32*0.75).collect::<Vec<_>>());
            assert!(insert_chunk(&mut output,&[],vocab,vocab,1,t).is_err());
        }
        Ok(())
    }
}
