//! Bounded, single-sequence raw generation; no chat rendering or serving layer.
//! Every append uses the same block/forward core as empty-history diagnostics.
//! Weights come from explicit GGUF or a frozen native pile. KV stays until reset.
use std::{path::Path,time::Instant};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use crate::formats::gguf::index::{FileIdentity,MetadataValue};
use super::{cache::{self,Admission,LayerCache,Limits},forward::{self,Backend,Head,VOCAB},
    math::Trace,tokenizer::{FlintTokenizer,EncodeOptions,DECLARED_EOG_IDS},weights::{GgufWeights,ReadReceipt,HostPhase,ReceiptPolicy}};

#[derive(Clone, Copy, Debug)]
pub struct Options { pub backend:Backend, pub limits:Limits, pub trace:bool }
/// Experimental wall-clock instrumentation only; no arithmetic or cache changes.
/// Prefill/append include streamed weight IO, first-use compilation and the
/// existing synchronous GPU readbacks/fences. No observer or file dump is inside
/// those intervals. The residual greedy interval includes host selection/observer.
#[derive(Default,Serialize)]
pub struct GenerationTimings {
    pub encode_validate_seconds:f64, pub reset_seconds:f64, pub prefill_seconds:f64,
    pub decode_seconds:Vec<f64>, pub greedy_host_observer_seconds:f64,
    pub prefill_host:Option<HostPhase>, pub decode_host:Vec<Option<HostPhase>>,
}
impl Default for Options {
    fn default()->Self { Self {backend:Backend::PackedAll,limits:Limits::default(),trace:false} }
}

#[derive(Default)]
struct Sequence { attended:usize, poisoned:bool }
impl Sequence {
    fn preflight(&self,ids:&[u32],capacity:usize)->Result<usize> {
        ensure!(!self.poisoned,"session is poisoned; successful reset required");
        ensure!(!ids.is_empty(),"empty append/prompt has no next-token distribution");
        ensure!(ids.iter().all(|&id|(id as usize)<VOCAB),"token outside Flint vocabulary");
        let end=self.attended.checked_add(ids.len()).context("sequence length overflow")?;
        ensure!(end<=capacity,"context capacity exhausted"); Ok(end)
    }
    fn execute<T>(&mut self,end:usize,run:impl FnOnce()->Result<T>)->Result<T> {
        ensure!(!self.poisoned,"session is poisoned");
        // Also remains set on unwinding. Never retry a partly advanced layer stack.
        self.poisoned=true;
        let output=run()?;
        self.attended=end; self.poisoned=false; Ok(output)
    }
    fn reset(&mut self,clear:impl FnOnce()->Result<()>)->Result<()> {
        self.poisoned=true; clear()?; *self=Self::default(); Ok(())
    }
}

pub struct FlintSession {
    weights:forward::Source, codec:FlintTokenizer, caches:Vec<LayerCache>,
    sequence:Sequence, options:Options, admission:Admission,
}
impl FlintSession {
    /// Reads metadata/index, admits cache geometry, then allocates KV. No tensor
    /// payloads are deliberately selected until append/prefill. IndexedFile's
    /// buffered header reading may read ahead; this is not a metadata-only probe.
    pub fn open(path:&Path,options:Options)->Result<Self> {
        Self::open_with_receipts(path,options,ReceiptPolicy::Collect)
    }
    /// Explicit per-session provenance policy. None leaves block/global receipt
    /// collections empty, including append_observed; all validation is unchanged.
    pub fn open_with_receipts(path:&Path,options:Options,receipts:ReceiptPolicy)->Result<Self> {
        ensure!(matches!(options.backend,Backend::Cpu)||cfg!(feature="flint-cuda"),"CUDA feature required");
        let weights=GgufWeights::open_block_with_receipts(path,0,receipts)?;
        let context=weights.file.index.metadata.iter().find_map(|e|
            if e.key=="flint.context_length" { if let MetadataValue::U32(n)=&e.value {Some(*n as usize)} else {None} } else {None})
            .context("missing declared context length")?;
        let admission=options.limits.admit(context)?;
        let (codec,identity,_)=FlintTokenizer::open(path)?;
        ensure!(identity==weights.file.identity,"tokenizer and weights are different file observations");
        weights.file.check_identity()?;
        let caches=cache::allocate(options.backend,options.limits.capacity)?;
        Ok(Self {weights:forward::Source::Gguf(weights),codec,caches,sequence:Sequence::default(),options,admission})
    }
    pub fn open_pile(path:&Path,root:triblespace::prelude::Id,options:Options)->Result<Self> {
        ensure!(matches!(options.backend,Backend::PackedAll) && cfg!(feature="flint-cuda"),"native pile requires CUDA PackedAll");
        let weights=super::pile::PileWeights::open(path,root)?;
        let entries=weights.metadata()?;
        let context=entries.iter().find_map(|e|if e.key=="flint.context_length" {
            if let MetadataValue::U32(n)=&e.value {Some(*n as usize)} else {None}
        } else {None}).context("missing native context length")?;
        let admission=options.limits.admit(context)?;
        let codec=weights.tokenizer()?; // Same snapshot/root; no second file observation.
        let caches=cache::allocate(options.backend,options.limits.capacity)?;
        Ok(Self {weights:forward::Source::Native(weights),codec,caches,sequence:Sequence::default(),options,admission})
    }
    pub fn identity(&self)->Option<&FileIdentity> { self.weights.gguf().map(|w|&w.file.identity) }
    pub fn source_identity(&self)->serde_json::Value {self.weights.identity()}
    pub fn receipt_policy(&self)->ReceiptPolicy { self.weights.receipt_policy() }
    pub fn admission(&self)->Admission { self.admission }
    pub fn capacity(&self)->usize { self.options.limits.capacity }
    /// Attended tokens only. The last emitted non-EOG token may not yet be in KV.
    pub fn attended_len(&self)->usize { self.sequence.attended }
    pub fn is_poisoned(&self)->bool { self.sequence.poisoned }
    pub fn set_trace(&mut self,enabled:bool) { self.options.trace=enabled; }
    pub fn encode_raw(&mut self,text:&str)->Result<Vec<u32>> {
        self.codec.encode(text,EncodeOptions {add_special:false,parse_special:false})
    }
    /// A failed CUDA fence leaves the session poisoned. Fatal device/context
    /// failures can require dropping/reconstructing it; reset is not GPU recovery.
    pub fn reset(&mut self)->Result<()> {
        let Self {sequence,caches,weights,..}=self;
        sequence.reset(|| {
            for cache in caches { cache.reset()?; }
            weights.clear_reads(); weights.check_identity()?; Ok(())
        })
    }
    /// Append one chunk of1..3 IDs; return NEXT-token logits for its last row.
    /// Errors after entering the forward conservatively poison the entire session.
    pub fn append(&mut self,ids:&[u32])->Result<Vec<f32>> {
        self.append_observed(ids,|_,_,_|Ok(()))
    }
    /// Observer sees at most one block's optional trace and read receipts. It must
    /// not reenter the session. Observer errors also poison, providing a diagnostic
    /// partial-execution witness without special production failure injection.
    pub fn append_observed(&mut self,ids:&[u32],observe:impl FnMut(usize,&Trace,&[ReadReceipt])->Result<()>)->Result<Vec<f32>> {
        append_parts(&mut self.weights,&mut self.caches,&mut self.sequence,self.options,ids,Head::Last,observe)
    }
    /// New sequence only; all IDs/capacity checked before the first mutation.
    /// Intermediate chunks skip the head. A failed later chunk requires reset.
    pub fn prefill(&mut self,ids:&[u32])->Result<Vec<f32>> {
        ensure!(self.sequence.attended==0,"prefill requires reset/empty session");
        self.sequence.preflight(ids,self.capacity())?;
        let chunks=ids.chunks(cache::MAX_QUERY);
        let count=chunks.len(); let mut logits=Vec::new();
        for (index,chunk) in chunks.enumerate() {
            logits=append_parts(&mut self.weights,&mut self.caches,&mut self.sequence,self.options,chunk,
                if index+1==count {Head::Last} else {Head::Skip},|_,_,_|Ok(()))?;
        }
        Ok(logits)
    }
    /// Raw text, deliberately not a chat prompt. parse_special=false still
    /// recognizes USER_DEFINED tokens. No synthetic BOS and no lossy UTF-8.
    /// Starts a new sequence. max_new=0 performs no forward. Context counts only
    /// attended IDs: a prediction at the final context slot can still be emitted.
    pub fn generate_raw(&mut self,prompt:&str,max_new:usize)->Result<Generation> {
        self.generate_raw_observed(prompt,max_new,|_,_|Ok(()))
    }
    /// One borrowed vocabulary distribution per prediction, including a stopping
    /// EOG. The first argument is the number of attended IDs, not output length.
    /// This supports independent empty-history prefix replay without full traces.
    pub fn generate_raw_observed(&mut self,prompt:&str,max_new:usize,
        observe:impl FnMut(usize,&[f32])->Result<()>)->Result<Generation> {
        self.generate_raw_observed_timed(prompt,max_new,observe).map(|(generation,_)|generation)
    }
    /// Same accepted path, with clocks at completed prefill/cached-append boundaries.
    pub fn generate_raw_observed_timed(&mut self,prompt:&str,max_new:usize,
        observe:impl FnMut(usize,&[f32])->Result<()>)->Result<(Generation,GenerationTimings)> {
        let mut timings=GenerationTimings::default();
        let start=Instant::now();
        let ids=self.encode_raw(prompt)?;
        ensure!(!ids.is_empty(),"empty raw prompt; no implicit BOS");
        ensure!(ids.len()<=self.capacity(),"prompt exceeds context capacity");
        // An implicitly successful reset must not disguise a previously failed
        // session: the caller must explicitly acknowledge it through reset().
        ensure!(!self.is_poisoned(),"session is poisoned; explicit successful reset required");
        timings.encode_validate_seconds=start.elapsed().as_secs_f64();
        let start=Instant::now();
        self.reset()?;
        timings.reset_seconds=start.elapsed().as_secs_f64();
        if max_new==0 { return Ok((Generation {ids:Vec::new(),bytes:Vec::new(),prompt_tokens:ids.len(),attended_tokens:0,stop:Stop::Budget},timings)); }
        let _=self.weights.take_host_phase();
        let start=Instant::now();
        let logits=self.prefill(&ids)?;
        timings.prefill_seconds=start.elapsed().as_secs_f64();
        timings.prefill_host=self.weights.take_host_phase();
        let Self {weights,caches,sequence,options,codec,..}=self;
        let start=Instant::now();
        let generation=greedy(logits,ids.len(),max_new,options.limits.capacity,
            |id| {
                let start=Instant::now();
                let logits=append_parts(weights,caches,sequence,*options,&[id],Head::Last,|_,_,_|Ok(()))?;
                timings.decode_seconds.push(start.elapsed().as_secs_f64());
                timings.decode_host.push(weights.take_host_phase()); Ok(logits)
            },|id| Ok(codec.token_bytes(id,false)?.to_vec()),observe)?;
        timings.greedy_host_observer_seconds=start.elapsed().as_secs_f64()-timings.decode_seconds.iter().sum::<f64>();
        Ok((generation,timings))
    }
}

fn append_parts(w:&mut forward::Source,caches:&mut [LayerCache],sequence:&mut Sequence,options:Options,
    ids:&[u32],head:Head,observe:impl FnMut(usize,&Trace,&[ReadReceipt])->Result<()>)->Result<Vec<f32>> {
    ensure!(ids.len()<=cache::MAX_QUERY,"append query length must be1..3; use prefill for longer prompts");
    let end=sequence.preflight(ids,options.limits.capacity)?;
    if !caches.iter().all(|c|c.len()==sequence.attended) {
        sequence.poisoned=true; anyhow::bail!("layer cache lengths differ from committed sequence");
    }
    let positions:Vec<i32>=(sequence.attended..end).map(|p|p as i32).collect();
    let signed:Vec<i32>=ids.iter().map(|&id|id as i32).collect();
    let expect_logits=!matches!(head,Head::Skip);
    let result=sequence.execute(end,|| {
        w.check_identity()?;
        let result=forward::run_core(w,&signed,&positions,options.backend,Some(&mut *caches),options.trace,head,observe)?;
        ensure!(caches.iter().all(|c|c.len()==end),"incomplete layer cache append");
        ensure!(if expect_logits { result.logits.len()==VOCAB && result.logits.iter().all(|v|v.is_finite()) } else {result.logits.is_empty()},
            "invalid next-token logits");
        Ok(result.logits)
    });
    // Also bound receipts when a weight read/observer fails partway through a block.
    w.clear_reads(); result
}

pub fn greedy_id(logits:&[f32])->Result<u32> {
    ensure!(logits.len()==VOCAB && logits.iter().all(|v|v.is_finite()),"expected finite Flint vocabulary logits");
    let mut best=0;
    for id in 1..logits.len() { if logits[id]>logits[best] { best=id; } }
    Ok(best as u32)
}
#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum Stop { Eog(u32), Budget, Context }
#[derive(Debug, Serialize)]
pub struct Generation {
    /// Emitted non-EOG IDs; hidden non-EOG controls may have zero rendered bytes.
    pub ids:Vec<u32>, pub bytes:Vec<u8>, pub prompt_tokens:usize,
    pub attended_tokens:usize, pub stop:Stop,
}
fn greedy(mut logits:Vec<f32>,prompt_tokens:usize,max_new:usize,capacity:usize,
    mut step:impl FnMut(u32)->Result<Vec<f32>>,mut piece:impl FnMut(u32)->Result<Vec<u8>>,
    mut observe:impl FnMut(usize,&[f32])->Result<()>)->Result<Generation> {
    let mut out=Generation {ids:Vec::new(),bytes:Vec::new(),prompt_tokens,attended_tokens:prompt_tokens,stop:Stop::Budget};
    for index in 0..max_new {
        let id=greedy_id(&logits)?;
        observe(out.attended_tokens,&logits)?;
        if DECLARED_EOG_IDS.contains(&id) { out.stop=Stop::Eog(id); break; }
        out.bytes.extend_from_slice(&piece(id)?); out.ids.push(id);
        if index+1==max_new { break; }
        if out.attended_tokens==capacity { out.stop=Stop::Context; break; }
        logits=step(id)?; out.attended_tokens+=1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn logits(id:u32)->Vec<f32> { let mut v=vec![0.0;VOCAB]; v[id as usize]=1.0; v }
    #[test] fn failed_append_poison_and_preflight() -> Result<()> {
        let mut s=Sequence::default();
        assert!(s.preflight(&[],3).is_err()); assert!(s.preflight(&[VOCAB as u32],3).is_err());
        assert!(!s.poisoned); assert_eq!(s.attended,0);
        let end=s.preflight(&[1,2],3)?;
        assert!(s.execute::<()>(end,||anyhow::bail!("after layer mutation")).is_err());
        assert!(s.poisoned); assert_eq!(s.attended,0); assert!(s.preflight(&[1],3).is_err());
        assert!(s.reset(||anyhow::bail!("failed reset fence")).is_err());
        assert!(s.poisoned); assert!(s.preflight(&[1],3).is_err());
        s.reset(||Ok(()))?; let end=s.preflight(&[1,2,3],3)?; s.execute(end,||Ok(()))?;
        assert!(s.preflight(&[1],3).is_err()); assert_eq!(s.attended,3); assert!(!s.poisoned); Ok(())
    }
    #[test] fn greedy_tie_nonfinite_and_four_stops() -> Result<()> {
        assert_eq!(greedy_id(&vec![0.0;VOCAB])?,0);
        let mut bad=logits(1); bad[7]=f32::NAN; assert!(greedy_id(&bad).is_err());
        assert!(greedy_id(&[]).is_err());
        for id in DECLARED_EOG_IDS {
            let result=greedy(logits(id),1,4,10,|_|panic!("EOG feedback"),|_|panic!("EOG render"),|_,_|Ok(()))?;
            assert_eq!(result.stop,Stop::Eog(id)); assert!(result.ids.is_empty()); assert_eq!(result.attended_tokens,1);
        }
        assert!(!super::super::tokenizer::B11853_EOG_IDS.contains(&150018)); Ok(())
    }
    #[test] fn raw_bytes_feedback_once_budget_and_context() -> Result<()> {
        let mut fed=Vec::new();
        let out=greedy(logits(0),2,3,10,|id|{fed.push(id);Ok(logits(id+1))},
            |id|Ok(vec![[0xe2,0x82,0xac][id as usize]]),|_,_|Ok(()))?;
        assert_eq!(fed,[0,1]); assert_eq!(out.ids,[0,1,2]); assert_eq!(out.bytes,"€".as_bytes());
        assert_eq!(out.attended_tokens,4); assert_eq!(out.stop,Stop::Budget);
        let out=greedy(logits(0),2,5,2,|_|panic!("capacity feedback"),|_|Ok(vec![0xe2]),|_,_|Ok(()))?;
        assert_eq!(out.stop,Stop::Context); assert_eq!(out.bytes,[0xe2]);
        assert!(String::from_utf8(out.bytes).is_err()); Ok(())
    }
}
