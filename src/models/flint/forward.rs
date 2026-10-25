//! Shared 49-block composition for empty-history diagnostics and cached sessions.
//! CPU residual handoff and bounded streamed weights are deliberate first-stage
//! choices. Sessions may retain only the native output head on the device.

use anyhow::{Context, Result, ensure};
use std::time::Instant;
use super::{math::{self, Trace, Values, H}, weights::{GgufWeights, ReadReceipt,Weights,PackedWeight,HostPhase,ReceiptPolicy},pile::PileWeights};
use crate::formats::gguf::nvfp4;

pub const VOCAB: usize = 150144;
pub const HEAD_ROWS: usize = 8192;
pub const BLOCKS: usize = 49;

#[derive(Clone, Copy, Debug)]
pub enum Backend { Cpu, Cuda, PackedAll }
pub(super) enum Head { All, Last, Skip }

/// Logical immutable output-head GPU bytes only, separate from KV/attention.
/// Allocator padding, runtime, query scratch and host mappings need headroom.
#[derive(Clone,Copy,Debug,serde::Serialize,PartialEq,Eq)]
pub struct HeadAdmission { pub rows:usize, pub chunks:usize, pub gpu_bytes:usize }
impl HeadAdmission {
    pub fn native(max_bytes:usize)->Result<Self> { Self::for_rows(VOCAB,max_bytes) }
    pub(super) fn for_rows(rows:usize,max_bytes:usize)->Result<Self> {
        ensure!(rows>0 && rows<=VOCAB,"invalid resident head row count");
        let chunks=rows.div_ceil(HEAD_ROWS);
        let elements=rows.checked_mul(H).context("resident head elements overflow")?;
        // NVFP4 codes + scales + one intrinsic global per independently packed
        // chunk, plus F32 RMS gain and ONE paired post-matmul scalar.
        let gpu_bytes=(elements/2).checked_add(elements/16)
            .and_then(|n|chunks.checked_mul(4).and_then(|g|n.checked_add(g)))
            .and_then(|n|n.checked_add(H*4+4)).context("resident head bytes overflow")?;
        ensure!(gpu_bytes<=max_bytes,"output-head admission: {gpu_bytes} > {max_bytes} bytes");
        Ok(Self {rows,chunks,gpu_bytes})
    }
}

/// Session-owned immutable buffers, never a global cache or a second catalogue.
pub(super) enum HeadWeights {
    Streamed,
    #[cfg(feature="flint-cuda")]
    Resident(super::gpu::ResidentOutputHead),
}
impl HeadWeights {
    pub fn resident(w:&PileWeights,max_bytes:usize)->Result<Self> {
        let admission=HeadAdmission::native(max_bytes)?; // Before any device allocation.
        #[cfg(feature="flint-cuda")]
        {
            let gain=w.output_norm()?;
            let (full,scale)=w.output_projection()?;
            Ok(Self::Resident(super::gpu::ResidentOutputHead::new(&gain,&full,scale,admission)?))
        }
        #[cfg(not(feature="flint-cuda"))]
        {let _=(w,admission);anyhow::bail!("resident native head requires flint-cuda")}
    }
    pub fn admission(&self)->Option<HeadAdmission> { match self {
        Self::Streamed=>None,
        #[cfg(feature="flint-cuda")] Self::Resident(head)=>Some(head.admission()),
    }}
    pub fn fence(&self)->Result<()> { match self {
        Self::Streamed=>Ok(()),
        #[cfg(feature="flint-cuda")] Self::Resident(head)=>head.fence(),
    }}
}

/// One storage choice, not a second forward/cache implementation. Native views
/// and tokenizer metadata are retained by the same frozen pile snapshot.
pub enum Source { Gguf(GgufWeights), Native(PileWeights) }
impl Source {
    pub fn gguf(&self)->Option<&GgufWeights> {match self {Self::Gguf(w)=>Some(w),_=>None}}
    pub fn metadata(&self)->Result<Vec<crate::formats::gguf::index::MetadataEntry>> {
        match self {Self::Gguf(w)=>Ok(w.file.index.metadata.clone()),Self::Native(w)=>w.metadata()}
    }
    pub fn identity(&self)->serde_json::Value {
        match self {
            Self::Gguf(w)=>serde_json::json!({"storage":"gguf","file":w.file.identity}),
            Self::Native(w)=>serde_json::json!({"storage":"native-pile","model_root":format!("{:X}",w.root),
                "collection":w.collection(),"support_members":w.support_members(),"support_handles":w.support_handles()}),
        }
    }
    pub fn check_identity(&self)->Result<()> {
        match self {Self::Gguf(w)=>w.file.check_identity(),Self::Native(_)=>Ok(())}
    }
    pub fn receipt_policy(&self)->ReceiptPolicy {
        match self {Self::Gguf(w)=>w.receipt_policy(),Self::Native(_)=>ReceiptPolicy::None}
    }
    pub fn take_host_phase(&mut self)->Option<HostPhase> {
        match self {Self::Gguf(w)=>Some(w.take_host_phase()),Self::Native(_)=>None}
    }
    pub fn reads(&self)->&[ReadReceipt] {match self {Self::Gguf(w)=>&w.reads,Self::Native(_)=>&[]}}
    pub fn take_reads(&mut self)->Vec<ReadReceipt> {
        match self {Self::Gguf(w)=>std::mem::take(&mut w.reads),Self::Native(_)=>Vec::new()}
    }
    pub fn clear_reads(&mut self) {
        if let Self::Gguf(w)=self {let start=Instant::now();w.reads.clear();w.host.receipt_drop.record(start,0,0);}
    }
    fn select_block(&mut self,index:usize)->Result<super::config::BlockConfig> {
        match self {Self::Gguf(w)=>{w.select_block(index)?;Ok(w.config)},
            Self::Native(w)=>{w.select_block(index)?;Ok(w.config)}}
    }
    fn embedding(&mut self,id:usize)->Result<Vec<f32>> {
        match self {
            Self::Native(w)=>w.embedding(id),
            Self::Gguf(w)=>{
                let raw=w.global_rows("token_embd.weight",&[H,VOCAB],30,id,1)?;
                let start=Instant::now();
                let values=raw.chunks_exact(2).map(|c|f32::from_bits(
                    u32::from(u16::from_le_bytes(c.try_into().unwrap()))<<16)).collect();
                w.host.global_conversion.record(start,raw.len(),H*4); Ok(values)
            }
        }
    }
}
impl Weights for Source {
    fn get(&mut self,s:&str,d:&[usize],t:u32,e:Option<usize>)->Result<Vec<f32>> {
        match self {Self::Gguf(w)=>w.get(s,d,t,e),Self::Native(w)=>w.get(s,d,t,e)}
    }
    fn get_nvfp4(&mut self,s:&str,d:&[usize],e:Option<usize>)->Result<Vec<u8>> {
        match self {Self::Gguf(w)=>w.get_nvfp4(s,d,e),Self::Native(w)=>w.get_nvfp4(s,d,e)}
    }
    fn get_packed(&mut self,s:&str,d:&[usize],e:Option<usize>)->Result<PackedWeight> {
        match self {Self::Gguf(w)=>w.get_packed(s,d,e),Self::Native(w)=>w.get_packed(s,d,e)}
    }
    fn packed_projection(&mut self,s:&str,d:&[usize],e:Option<usize>)->Result<(PackedWeight,Option<f32>)> {
        match self {Self::Gguf(w)=>w.packed_projection(s,d,e),Self::Native(w)=>w.packed_projection(s,d,e)}
    }
    fn paired_scales(&self)->bool {matches!(self,Self::Native(_))}
}

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
    let start=Instant::now();
    let values: Vec<f32>=raw.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect();
    ensure!(values.iter().all(|v|v.is_finite()),"nonfinite global tensor {name}");
    w.host.global_conversion.record(start,raw.len(),values.len()*4);
    Ok(values)
}

pub fn chunks(rows: usize) -> impl Iterator<Item=(usize,usize)> {
    (0..rows).step_by(HEAD_ROWS).map(move |first|(first,(rows-first).min(HEAD_ROWS)))
}

pub(super) fn insert_chunk(logits: &mut [f32], values: &[f32], vocab: usize, first: usize, rows: usize, t: usize) -> Result<()> {
    ensure!(rows>0 && first.checked_add(rows).is_some_and(|end|end<=vocab)
        && logits.len()==vocab*t && values.len()==rows*t,"head chunk output geometry differs");
    for token in 0..t {
        logits[token*vocab+first..token*vocab+first+rows].copy_from_slice(&values[token*rows..(token+1)*rows]);
    }
    Ok(())
}

/// Observer borrows one completed block at a time. It must not launch work on
/// weights while the trace is being examined. Runtime errors leave no KV state.
pub fn run(w: &mut Source, ids: &[i32], positions: &[i32], backend: Backend,
    observe: impl FnMut(usize,&Trace,&[ReadReceipt])->Result<()>) -> Result<Output> {
    run_core(w,ids,positions,backend,None,true,Head::All,&HeadWeights::Streamed,observe)
}

pub(super) fn run_core(w:&mut Source,ids:&[i32],positions:&[i32],backend:Backend,
    mut caches:Option<&mut [super::cache::LayerCache]>,trace:bool,head:Head,
    head_weights:&HeadWeights,
    mut observe:impl FnMut(usize,&Trace,&[ReadReceipt])->Result<()>)->Result<Output> {
    let t=ids.len();
    ensure!((1..=3).contains(&t) && positions.len()==t,"expected T1..3 token IDs and positions");
    ensure!(ids.iter().all(|&id|id>=0 && (id as usize)<VOCAB),"token ID outside vocabulary");
    ensure!(positions.iter().all(|&p|p>=0) && positions.windows(2).all(|p|p[0]<p[1]),"invalid positions");
    ensure!(matches!(backend,Backend::Cpu) || cfg!(feature="flint-cuda"),"CUDA feature required");
    if let Some(c)=caches.as_ref() { ensure!(c.len()==BLOCKS,"expected all49 layer caches"); }
    ensure!(!matches!(w,Source::Native(_)) || matches!(backend,Backend::PackedAll),"native pile requires PackedAll");
    ensure!(head_weights.admission().is_none() || matches!(w,Source::Native(_)),"resident head requires its native source");
    if let Source::Gguf(w)=w { ensure!(!w.file.index.tensors.iter().any(|info|
        info.name=="output.input_scale" || (info.name.starts_with("blk.") && info.name.ends_with(".bias"))),
        "unsupported input scale or block bias tensor"); }
    let mut embedding=Vec::with_capacity(t*H);
    for &id in ids {
        embedding.extend(w.embedding(id as usize)?);
    }
    ensure!(embedding.iter().all(|v|v.is_finite()),"nonfinite embedding");
    let mut global_reads=w.take_reads();
    let mut residual=embedding.clone();
    for block in 0..BLOCKS {
        let config=w.select_block(block)?;
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
        observe(block,&result.trace,w.reads())?;
        w.clear_reads();
        // The full trace and its already-fenced GPU work are dropped here.
    }
    let head_residual=match head { Head::All=>residual.as_slice(),Head::Last=>&residual[(t-1)*H..],Head::Skip=>&[] };
    let (normalized,logits)=if head_residual.is_empty() { (Vec::new(),Vec::new()) }
        else { match head_weights {
            #[cfg(feature="flint-cuda")] HeadWeights::Resident(head)=>head.run(head_residual)?,
            HeadWeights::Streamed=>match w {Source::Gguf(w)=>output_head(w,head_residual,backend)?,
                Source::Native(w)=>native_head(w,head_residual)?},
        }};
    global_reads.extend(w.take_reads());
    w.check_identity()?;
    Ok(Output {embedding,residual,normalized,logits,global_reads})
}

fn native_head(w:&PileWeights,residual:&[f32])->Result<(Vec<f32>,Vec<f32>)> {
    #[cfg(feature="flint-cuda")]
    {
        let head=super::gpu::OutputHead::new(residual,&w.output_norm()?)?;
        let normalized=head.normalized()?; let t=residual.len()/H;
        let mut logits=vec![0.0;VOCAB*t];
        for (first,rows) in chunks(VOCAB) {
            let (view,scale)=w.output_rows(first,rows)?;
            let values=head.native_chunk(&view,rows,scale)?;
            insert_chunk(&mut logits,&values,VOCAB,first,rows,t)?;
        }
        ensure!(normalized.iter().chain(&logits).all(|v|v.is_finite()),"nonfinite native final norm/logits");
        Ok((normalized,logits))
    }
    #[cfg(not(feature="flint-cuda"))]
    { let _=(w,residual);anyhow::bail!("native pile requires flint-cuda") }
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
    #[test] fn resident_head_checked_logical_admission() -> Result<()> {
        let a=HeadAdmission::native(192<<20)?;
        assert_eq!(a,HeadAdmission {rows:VOCAB,chunks:19,gpu_bytes:172_974_160});
        assert!(HeadAdmission::native(a.gpu_bytes-1).is_err());
        assert_eq!(HeadAdmission::native(a.gpu_bytes)?,a);
        for rows in [0,VOCAB+1,usize::MAX] {assert!(HeadAdmission::for_rows(rows,usize::MAX).is_err());}
        let tail=HeadAdmission::for_rows(HEAD_ROWS+3,usize::MAX)?;
        assert_eq!(tail.chunks,2);
        assert_eq!(tail.gpu_bytes,(HEAD_ROWS+3)*H*9/16+2*4+H*4+4);
        Ok(())
    }
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
