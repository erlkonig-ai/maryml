//! Bounded, single-sequence ID generation; no chat rendering or serving layer.
//! Every append uses the same block/forward core as empty-history diagnostics.
//! Weights come from explicit GGUF or a frozen native pile. KV stays until reset;
//! optional immutable native head/fixed block operands stay until session drop.
use std::{path::Path,time::Instant};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use crate::formats::gguf::index::{FileIdentity,MetadataValue};
use super::{cache::{self,Admission,LayerCache,Limits},forward::{self,Backend,Head,VOCAB},
    math::Trace,tokenizer::{FlintTokenizer,DECLARED_EOG_IDS},weights::{GgufWeights,ReadReceipt,HostPhase,ReceiptPolicy}};
use super::residual::{Request as ResidualRequest,Prepared as PreparedResidual,Phase as ResidualPhase};

#[derive(Clone,Copy,Debug)]
pub enum NativeHead { Streamed, Resident {max_bytes:usize} }
impl Default for NativeHead {
    fn default()->Self {Self::Resident {max_bytes:192<<20}}
}
#[derive(Clone,Copy,Debug)]
pub enum NativeFixed { Streamed, Resident {max_bytes:usize} }
impl Default for NativeFixed {
    fn default()->Self {Self::Resident {max_bytes:1<<30}}
}
#[derive(Clone, Copy, Debug)]
pub struct Options { pub backend:Backend, pub limits:Limits, pub trace:bool, pub native_head:NativeHead, pub native_fixed:NativeFixed }
/// Experimental wall-clock instrumentation only; no arithmetic or cache changes.
/// Prefill/append include streamed weight IO, first-use compilation and the
/// existing synchronous GPU readbacks/fences. Residual capture callbacks (and
/// any IO they perform) are inside those intervals. Prediction observers are
/// outside them, in the remaining greedy host/observer interval.
#[derive(Default,Serialize)]
pub struct GenerationTimings {
    pub encode_validate_seconds:f64, pub reset_seconds:f64, pub prefill_seconds:f64,
    pub decode_seconds:Vec<f64>, pub greedy_host_observer_seconds:f64,
    pub prefill_host:Option<HostPhase>, pub decode_host:Vec<Option<HostPhase>>,
    pub prefill_native:Option<super::native_setup::NativeSetupPhase>,
    pub decode_native:Vec<Option<super::native_setup::NativeSetupPhase>>,
}
impl Default for Options {
    fn default()->Self { Self {backend:Backend::PackedAll,limits:Limits::default(),trace:false,
        native_head:NativeHead::default(),native_fixed:NativeFixed::default()} }
}

#[derive(Default)]
struct Sequence { attended:usize, poisoned:bool }
impl Sequence {
    fn prompt_preflight(&self,ids:&[u32],capacity:usize)->Result<()> {
        // A new prompt replaces attended history, but cannot silently recover a
        // failed session. Validate the entire ID slice BEFORE reset/mutation.
        ensure!(!self.poisoned,"session is poisoned; explicit successful reset required");
        Self::default().preflight(ids,capacity)?; Ok(())
    }
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
    head_weights:forward::HeadWeights,
    fixed_weights:forward::FixedWeights,
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
        Ok(Self {weights:forward::Source::Gguf(weights),codec,caches,sequence:Sequence::default(),options,admission,
            head_weights:forward::HeadWeights::Streamed,fixed_weights:forward::FixedWeights::Streamed})
    }
    pub fn open_pile(path:&Path,root:triblespace::prelude::Id,options:Options)->Result<Self> {
        Self::open_pile_with_collection(path,root,None,options)
    }
    /// Descriptor selection and model root are independent coordinates. None
    /// retains sole-name discovery for ordinary single-collection scratch piles.
    pub fn open_pile_with_collection(path:&Path,root:triblespace::prelude::Id,
        collection:Option<triblespace::core::collection::CollectionHandle>,options:Options)->Result<Self> {
        ensure!(matches!(options.backend,Backend::PackedAll) && cfg!(feature="flint-cuda"),"native pile requires CUDA PackedAll");
        let mut weights=super::pile::PileWeights::open_with_collection(path,root,collection)?;
        let entries=weights.metadata()?;
        let context=entries.iter().find_map(|e|if e.key=="flint.context_length" {
            if let MetadataValue::U32(n)=&e.value {Some(*n as usize)} else {None}
        } else {None}).context("missing native context length")?;
        let admission=options.limits.admit(context)?;
        let codec=weights.tokenizer()?; // Same snapshot/root; no second file observation.
        // All actual fixed operands are selected, validated and checked-summed
        // before the first fixed upload (also before head/KV device allocation).
        let fixed_weights=match options.native_fixed {
            NativeFixed::Streamed=>forward::FixedWeights::Streamed,
            NativeFixed::Resident {max_bytes}=>forward::FixedWeights::resident(&mut weights,max_bytes)?,
        };
        let head_weights=match options.native_head {
            NativeHead::Streamed=>forward::HeadWeights::Streamed,
            NativeHead::Resident {max_bytes}=>forward::HeadWeights::resident(&weights,max_bytes)?,
        };
        let caches=cache::allocate(options.backend,options.limits.capacity)?;
        Ok(Self {weights:forward::Source::Native(weights),codec,caches,sequence:Sequence::default(),options,admission,head_weights,fixed_weights})
    }
    pub fn identity(&self)->Option<&FileIdentity> { self.weights.gguf().map(|w|&w.file.identity) }
    pub fn source_identity(&self)->serde_json::Value {self.weights.identity()}
    pub fn receipt_policy(&self)->ReceiptPolicy { self.weights.receipt_policy() }
    pub fn admission(&self)->Admission { self.admission }
    /// Immutable head buffers only, NOT total GPU memory. None means streamed.
    pub fn head_admission(&self)->Option<forward::HeadAdmission> {self.head_weights.admission()}
    /// Immutable non-routed block buffers only; None means streamed (or GGUF).
    pub fn fixed_admission(&self)->Option<forward::FixedAdmission> {self.fixed_weights.admission()}
    pub fn capacity(&self)->usize { self.options.limits.capacity }
    /// Attended tokens only. The last emitted non-EOG token may not yet be in KV.
    pub fn attended_len(&self)->usize { self.sequence.attended }
    pub fn is_poisoned(&self)->bool { self.sequence.poisoned }
    pub fn set_trace(&mut self,enabled:bool) { self.options.trace=enabled; }
    /// Explicit diagnostic after initialization; excludes resident head/fixed
    /// uploads. Resident fixed execution consequently has no fixed setup calls.
    /// Native-only API wall counters. No change to fences or weight validation.
    pub fn enable_native_setup_timing(&mut self)->Result<()> {self.weights.enable_native_setup()}
    /// Drain once per phase; useful for retaining an incomplete failed phase.
    pub fn take_native_setup_timing(&mut self)->Option<super::native_setup::NativeSetupPhase> {
        self.weights.take_native_setup()
    }
    pub fn encode_raw(&mut self,text:&str)->Result<Vec<u32>> {
        self.codec.encode_raw(text)
    }
    /// Only for already-rendered trusted framing. Parses literal controls; does
    /// not render the retained template, sanitize user content, or insert BOS.
    pub fn encode_trusted_framed(&mut self,text:&str)->Result<Vec<u32>> {
        self.codec.encode_trusted_framed(text)
    }
    /// A failed CUDA fence leaves the session poisoned. Fatal device/context
    /// failures can require dropping/reconstructing it; reset is not GPU recovery.
    pub fn reset(&mut self)->Result<()> {
        let Self {sequence,caches,weights,head_weights,fixed_weights,..}=self;
        sequence.reset(|| {
            head_weights.fence()?; // Keep immutable weights, including after poison.
            fixed_weights.fence()?;
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
        self.append_observed_with_residual(ids,None,observe)
    }
    /// Experimental native-only request, never installed on the session. All
    /// plan/source/budget checks precede mutation. Removing a plan does NOT undo
    /// its downstream KV effects: low-level append intentionally retains history.
    /// Trace l_out is before this block's add; captures label pre/post explicitly.
    pub fn append_observed_with_residual(&mut self,ids:&[u32],request:Option<ResidualRequest<'_, '_>>,
        observe:impl FnMut(usize,&Trace,&[ReadReceipt])->Result<()>)->Result<Vec<f32>> {
        ensure!(ids.len()<=cache::MAX_QUERY,"append query length must be1..3; use prefill for longer prompts");
        let end=self.sequence.preflight(ids,self.capacity())?;
        let mut request=request.map(|r|r.prepare(&self.source_identity(),self.capacity(),self.sequence.attended..end))
            .transpose()?.flatten();
        append_parts(&mut self.weights,&mut self.caches,&mut self.sequence,self.options,&self.head_weights,&self.fixed_weights,
            ids,Head::Last,request.as_mut(),ResidualPhase::Append,observe)
    }
    /// New sequence only; all IDs/capacity checked before the first mutation.
    /// Intermediate chunks skip the head. A failed later chunk requires reset.
    pub fn prefill(&mut self,ids:&[u32])->Result<Vec<f32>> {
        self.prefill_with_residual(ids,None)
    }
    /// One invocation spans ALL prefill chunks, including head-skipped chunks.
    /// Absolute input positions and independent layer selection are chunk-invariant.
    pub fn prefill_with_residual(&mut self,ids:&[u32],request:Option<ResidualRequest<'_, '_>>)->Result<Vec<f32>> {
        ensure!(self.sequence.attended==0,"prefill requires reset/empty session");
        self.sequence.preflight(ids,self.capacity())?;
        let mut request=request.map(|r|r.prepare(&self.source_identity(),self.capacity(),0..ids.len()))
            .transpose()?.flatten();
        self.prefill_prepared(ids,request.as_mut())
    }
    fn prefill_prepared(&mut self,ids:&[u32],mut request:Option<&mut PreparedResidual<'_, '_>>)->Result<Vec<f32>> {
        ensure!(self.sequence.attended==0,"prefill requires reset/empty session");
        self.sequence.preflight(ids,self.capacity())?;
        let chunks=ids.chunks(cache::MAX_QUERY);
        let count=chunks.len(); let mut logits=Vec::new();
        for (index,chunk) in chunks.enumerate() {
            logits=append_parts(&mut self.weights,&mut self.caches,&mut self.sequence,self.options,&self.head_weights,&self.fixed_weights,chunk,
                if index+1==count {Head::Last} else {Head::Skip},request.as_deref_mut(),ResidualPhase::Prefill,|_,_,_|Ok(()))?;
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
        let start=Instant::now();
        let ids=self.encode_raw(prompt)?;
        let encode_seconds=start.elapsed().as_secs_f64();
        let (generation,mut timings)=self.generate_ids_observed_timed(&ids,max_new,observe)?;
        timings.encode_validate_seconds+=encode_seconds;
        Ok((generation,timings))
    }
    /// Exact caller-supplied IDs, including any trusted framing controls. No
    /// re-encoding, implicit BOS, filtering, or prompt-EOG removal. Validate all
    /// IDs before reset, even for max_new=0; the latter performs no forward.
    pub fn generate_ids(&mut self,ids:&[u32],max_new:usize)->Result<Generation> {
        self.generate_ids_observed_timed(ids,max_new,|_,_|Ok(())).map(|(generation,_)|generation)
    }
    /// Shared raw/framed/ID generation loop. Encoding, if needed, belongs to the
    /// caller; encode_validate_seconds here measures ID validation only.
    pub fn generate_ids_observed_timed(&mut self,ids:&[u32],max_new:usize,
        observe:impl FnMut(usize,&[f32])->Result<()>)->Result<(Generation,GenerationTimings)> {
        self.generate_ids_observed_timed_with_residual(ids,max_new,None,observe)
    }
    /// Experimental native capture/steering through the existing ID loop.
    /// No tokenizer mode is changed: caller chooses actual raw/trusted IDs.
    pub fn generate_ids_with_residual(&mut self,ids:&[u32],max_new:usize,request:ResidualRequest<'_, '_>)->Result<Generation> {
        self.generate_ids_observed_timed_with_residual(ids,max_new,Some(request),|_,_|Ok(())).map(|(generation,_)|generation)
    }
    /// Request-owned borrowed plan/sink, discarded on success or error; never
    /// sticky configuration. Preflight occurs BEFORE reset, even max_new=0.
    /// New generations reset KV, making independent arms possible. A final
    /// emission may not be attended. A generated stopping EOG is never emitted,
    /// appended or captured; explicit prompt EOG/control IDs remain attended.
    /// Capture callbacks execute inside prefill/append wall spans. Their failure
    /// has the same poison/reset semantics as a failed block observer.
    /// encode_validate_seconds includes plan binding/admission when supplied.
    pub fn generate_ids_observed_timed_with_residual(&mut self,ids:&[u32],max_new:usize,
        request:Option<ResidualRequest<'_, '_>>,observe:impl FnMut(usize,&[f32])->Result<()>)->Result<(Generation,GenerationTimings)> {
        let mut timings=GenerationTimings::default();
        let start=Instant::now();
        self.sequence.prompt_preflight(ids,self.capacity())?;
        // At most max_new-1 emissions are appended. Capacity clips the horizon
        // without overflowing, even when max_new is usize::MAX.
        let horizon=attended_horizon(ids.len(),max_new,self.capacity())?;
        let mut request=request.map(|r|r.prepare(&self.source_identity(),self.capacity(),0..horizon))
            .transpose()?.flatten();
        timings.encode_validate_seconds=start.elapsed().as_secs_f64();
        let start=Instant::now();
        self.reset()?;
        timings.reset_seconds=start.elapsed().as_secs_f64();
        if max_new==0 { return Ok((Generation {ids:Vec::new(),bytes:Vec::new(),prompt_tokens:ids.len(),attended_tokens:0,stop:Stop::Budget},timings)); }
        let _=self.weights.take_host_phase();
        let _=self.weights.take_native_setup();
        let start=Instant::now();
        let logits=self.prefill_prepared(ids,request.as_mut())?;
        timings.prefill_seconds=start.elapsed().as_secs_f64();
        timings.prefill_host=self.weights.take_host_phase();
        timings.prefill_native=self.weights.take_native_setup();
        let Self {weights,caches,sequence,options,codec,head_weights,fixed_weights,..}=self;
        let start=Instant::now();
        let generation=greedy(logits,ids.len(),max_new,options.limits.capacity,
            |id| {
                let start=Instant::now();
                let logits=append_parts(weights,caches,sequence,*options,head_weights,fixed_weights,&[id],Head::Last,
                    request.as_mut(),ResidualPhase::Decode,|_,_,_|Ok(()))?;
                timings.decode_seconds.push(start.elapsed().as_secs_f64());
                timings.decode_host.push(weights.take_host_phase());
                timings.decode_native.push(weights.take_native_setup()); Ok(logits)
            },|id| Ok(codec.token_bytes(id,false)?.to_vec()),observe)?;
        timings.greedy_host_observer_seconds=start.elapsed().as_secs_f64()-timings.decode_seconds.iter().sum::<f64>();
        Ok((generation,timings))
    }
}

fn attended_horizon(prompt:usize,max_new:usize,capacity:usize)->Result<usize> {
    ensure!(prompt>0 && prompt<=capacity,"invalid prompt horizon");
    Ok(if max_new==0 {0} else {prompt+(capacity-prompt).min(max_new-1)})
}

fn append_parts(w:&mut forward::Source,caches:&mut [LayerCache],sequence:&mut Sequence,options:Options,
    head_weights:&forward::HeadWeights,fixed_weights:&forward::FixedWeights,
    ids:&[u32],head:Head,request:Option<&mut PreparedResidual<'_, '_>>,phase:ResidualPhase,
    observe:impl FnMut(usize,&Trace,&[ReadReceipt])->Result<()>)->Result<Vec<f32>> {
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
        let result=forward::run_core(w,&signed,&positions,options.backend,Some(&mut *caches),options.trace,head,head_weights,fixed_weights,
            request,phase,observe)?;
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
    use super::super::residual::{self,Plan,Capture,CaptureSink,CapturedRow,Positions,Steering};
    fn logits(id:u32)->Vec<f32> { let mut v=vec![0.0;VOCAB]; v[id as usize]=1.0; v }
    #[test] fn residual_horizon_is_attended_not_emitted_and_checked() -> Result<()> {
        assert_eq!(attended_horizon(4,0,10)?,0);
        assert_eq!(attended_horizon(4,1,10)?,4);
        assert_eq!(attended_horizon(4,3,10)?,6);
        assert_eq!(attended_horizon(4,usize::MAX,10)?,10);
        assert_eq!(attended_horizon(usize::MAX,2,usize::MAX)?,usize::MAX);
        assert!(attended_horizon(0,1,10).is_err());
        assert!(attended_horizon(11,1,10).is_err()); Ok(())
    }
    #[test] fn residual_none_zero_capture_only_preserve_synthetic_ids_and_bytes() -> Result<()> {
        let source=serde_json::json!({"storage":"native-pile","model_root":"synthetic"});
        let direction=vec![1.0;super::super::math::H];
        for mode in 0..3 {
            let mut captures=Vec::new();
            let mut emit=|row:CapturedRow<'_>| {captures.push((row.input_token_id,row.absolute_position,row.phase));Ok(())};
            let plan=Plan {source_identity:&source,
                steering:(mode==1).then_some(Steering {block:4,positions:Positions::AllAttended,direction:&direction,alpha:0.0,cap:1.0}),
                capture:(mode==2).then_some(Capture {blocks:&[4],positions:Positions::AllAttended,post_add:false})};
            let mut request=if mode==0 {None} else {residual::Request {plan,
                sink:if mode==2 {Some(CaptureSink::new(2*super::super::math::H*4,&mut emit))}else{None}}
                .prepare(&source,10,0..attended_horizon(1,2,10)?)?};
            let mut s=Sequence::default();let mut values=vec![2.0;super::super::math::H];
            s.execute(1,|| {
                if let Some(r)=request.as_mut() {r.apply(4,&[7],&[0],ResidualPhase::Prefill,&mut values)?;}
                Ok(())
            })?;
            let out=greedy(logits(3),1,2,10,|id| {
                let position=s.attended;let end=s.preflight(&[id],10)?;
                s.execute(end,|| {
                    if let Some(r)=request.as_mut() {r.apply(4,&[id as i32],&[position as i32],ResidualPhase::Decode,&mut values)?;}
                    Ok(logits(4))
                })
            },|id|Ok(vec![b'A'+id as u8]),|_,_|Ok(()))?;
            drop(request);
            assert_eq!(out.ids,[3,4]);assert_eq!(out.bytes,b"DE");assert_eq!(out.attended_tokens,2);
            assert!(values.iter().all(|&v|v==2.0));
            if mode==2 {assert_eq!(captures,[(7,0,ResidualPhase::Prefill),(3,1,ResidualPhase::Decode)]);}
            else {assert!(captures.is_empty());}
            // The final emitted4 was not appended; it is not a capture input.
        }
        Ok(())
    }
    #[test] fn residual_preflight_and_capture_failure_keep_poison_reset_contract() -> Result<()> {
        let source=serde_json::json!({"storage":"native-pile","model_root":"synthetic"});
        let mut s=Sequence {attended:2,poisoned:false};
        let bad=residual::Request {plan:Plan {source_identity:&source,
            steering:Some(Steering {block:4,positions:Positions::AllAttended,direction:&[],alpha:0.0,cap:1.0}),capture:None},sink:None};
        let attempt=(|| -> Result<()> {
            s.prompt_preflight(&[7],10)?;
            let _=bad.prepare(&source,10,0..1)?; // Same before-reset ordering as public generation.
            s.reset(||Ok(()))
        })();
        assert!(attempt.is_err());assert_eq!(s.attended,2);assert!(!s.poisoned);
        s.reset(||Ok(()))?;
        let mut fail=|_:CapturedRow<'_>|anyhow::bail!("injected capture sink failure");
        let request=residual::Request {plan:Plan {source_identity:&source,steering:None,
            capture:Some(Capture {blocks:&[4],positions:Positions::AllAttended,post_add:false})},
            sink:Some(CaptureSink::new(super::super::math::H*4,&mut fail))};
        let mut prepared=request.prepare(&source,10,0..1)?.unwrap();
        let mut row=vec![1.0;super::super::math::H];
        assert!(s.execute(1,||prepared.apply(4,&[7],&[0],ResidualPhase::Append,&mut row)).is_err());
        assert!(s.poisoned);assert_eq!(s.attended,0);assert!(s.preflight(&[7],10).is_err());
        drop(prepared); // No request is a Sequence/FlintSession field.
        s.reset(||Ok(()))?;
        let direction=vec![1.0;super::super::math::H];
        let mut steered=residual::Request {plan:Plan {source_identity:&source,capture:None,
            steering:Some(Steering {block:4,positions:Positions::AllAttended,direction:&direction,alpha:0.5,cap:1.0})},sink:None}
            .prepare(&source,10,0..1)?.unwrap();
        s.execute(1,||steered.apply(4,&[7],&[0],ResidualPhase::Append,&mut row))?;
        assert_eq!(row[0],1.5);drop(steered);
        s.reset(||Ok(()))?;
        let baseline=vec![1.0;super::super::math::H];
        let next=s.execute(1,||Ok(baseline.clone()))?; // Next invocation passes no request.
        assert_eq!(next,baseline);assert!(!s.poisoned);assert_eq!(s.attended,1);
        Ok(())
    }
    #[test] fn explicit_prompt_ids_validate_before_reset() -> Result<()> {
        let mut s=Sequence {attended:4,poisoned:false};
        // Previous history does not consume the replacement prompt's capacity.
        s.prompt_preflight(&[150003,9479,13,150004],4)?;
        for bad in [&[][..],&[VOCAB as u32][..],&[u32::MAX][..],&[1,2,3,4,5][..]] {
            assert!(s.prompt_preflight(bad,4).is_err());
            assert_eq!(s.attended,4);assert!(!s.poisoned);
        }
        s.poisoned=true;
        assert!(s.prompt_preflight(&[9479,13],4).is_err());
        assert_eq!(s.attended,4);assert!(s.poisoned);
        s.reset(||Ok(()))?;
        s.prompt_preflight(&[150003,9479,13,150004],4)?;
        assert_eq!(s.attended,0); // Prompt EOG/control IDs are not stripped.
        Ok(())
    }
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
