//! Experimental request-local post-complete-block `l_out` capture/addition.
//!
//! This is the logical residual AFTER attention and the complete FFN residual,
//! before the next block's norm (or final output norm at block48). It reuses
//! today's completed host residual; that host placement is not the definition
//! of the seam. No extra GPU work or full trace is required.
//!
//! Positions name ATTENDED INPUT tokens, not predictions/emissions. Callers
//! obtain sentence/span positions from actual tokenization; punctuation need
//! not occupy a whole token. Captures retain per-row values, not sentence means.
//! Arrival order follows block/chunk execution; identify rows by block, absolute
//! position and pre/post stage, not by callback ordinal across chunk partitions.
//! Direction polarity alone establishes no affect interpretation. No implicit
//! normalization, model-selected coefficient, global control or session state.
//!
//! Clearing a request does NOT clear its downstream KV influence: block b's KV
//! is already written; later blocks consume the modified residual. Independent
//! generation arms reset. Low-level append may intentionally retain history.

use std::ops::Range;
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;
use super::{forward::BLOCKS, math::H};

#[derive(Clone, Copy, Debug)]
pub enum Positions<'a> { AllAttended, Only(&'a [usize]) }
impl Positions<'_> {
    fn validate(self, capacity: usize) -> Result<()> {
        if let Self::Only(rows) = self {
            ensure!(rows.iter().all(|&p| p < capacity) && rows.windows(2).all(|p| p[0] < p[1]),
                "residual positions must be sorted, unique and below capacity");
        }
        Ok(())
    }
    fn contains(self, position: usize) -> bool {
        match self { Self::AllAttended => true, Self::Only(rows) => rows.binary_search(&position).is_ok() }
    }
    fn count(self, range: &Range<usize>) -> usize {
        match self {
            Self::AllAttended => range.end - range.start,
            Self::Only(rows) => rows.partition_point(|&p| p < range.end) - rows.partition_point(|&p| p < range.start),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Steering<'a> {
    pub block: usize,
    pub positions: Positions<'a>,
    pub direction: &'a [f32],
    /// Externally chosen finite 0<=alpha<=cap. The model cannot alter this plan.
    pub alpha: f32,
    pub cap: f32,
}

#[derive(Clone, Copy)]
pub struct Capture<'a> {
    /// Independent of the one steering block: up to49 sorted unique blocks.
    pub blocks: &'a [usize],
    pub positions: Positions<'a>,
    /// Always capture pre-add l_out; optionally capture post-add too.
    pub post_add: bool,
}

#[derive(Clone, Copy)]
pub struct Plan<'a> {
    /// Exact existing FlintSession::source_identity() of the frozen native source.
    /// Caller also records executable/backend and direction content/construction
    /// provenance alongside its captures; this API invents no storage schema.
    pub source_identity: &'a Value,
    pub steering: Option<Steering<'a>>,
    pub capture: Option<Capture<'a>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Phase { Prefill, Decode, Append }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CaptureStage { PreAddLout, PostAddLout }

pub struct CapturedRow<'a> {
    pub source_identity: &'a Value,
    pub block: usize,
    pub absolute_position: usize,
    pub input_token_id: u32,
    pub phase: Phase,
    pub stage: CaptureStage,
    /// Exactly H finite F32 values, borrowed only during the callback.
    pub values: &'a [f32],
}

/// Explicit logical F32-payload budget, not callback allocation/process memory.
/// The sink is request-owned; only copied rows/metadata deliberately retained by
/// the caller survive. A callback must not reenter the session. Error leaves
/// partial captures with the caller and poisons the sequence, not a full result.
pub struct CaptureSink<'a> {
    max_bytes: usize,
    emit: &'a mut dyn for<'r> FnMut(CapturedRow<'r>) -> Result<()>,
}
impl<'a> CaptureSink<'a> {
    pub fn new(max_bytes: usize, emit: &'a mut dyn for<'r> FnMut(CapturedRow<'r>) -> Result<()>) -> Self {
        Self { max_bytes, emit }
    }
}

pub struct Request<'a, 's> { pub plan: Plan<'a>, pub sink: Option<CaptureSink<'s>> }

fn capture_bytes(layers: usize, rows: usize, post_add: bool) -> Result<usize> {
    layers.checked_mul(rows).and_then(|n| n.checked_mul(if post_add { 2 } else { 1 }))
        .and_then(|n| n.checked_mul(H)).and_then(|n| n.checked_mul(size_of::<f32>()))
        .context("residual capture byte budget overflow")
}

/// Call-stack-only validated invocation. No weights, cached rows or retained
/// semantic model. Horizon bounds every possible attended input in this call.
pub(super) struct Prepared<'a, 's> {
    request: Request<'a, 's>, horizon: Range<usize>, admitted_bytes: usize, used_bytes: usize,
}
impl<'a, 's> Request<'a, 's> {
    pub(super) fn prepare(self, actual: &Value, capacity: usize, horizon: Range<usize>) -> Result<Option<Prepared<'a, 's>>> {
        ensure!(actual["storage"] == "native-pile" && self.plan.source_identity == actual,
            "residual request must bind the exact native source identity");
        ensure!(capacity > 0 && horizon.start <= horizon.end && horizon.end <= capacity,
            "invalid residual invocation horizon");
        if let Some(s) = self.plan.steering {
            ensure!(s.block < BLOCKS && s.direction.len() == H, "invalid steering block/direction width");
            ensure!(s.direction.iter().all(|v| v.is_finite()), "nonfinite steering direction");
            ensure!(s.cap.is_finite() && s.cap >= 0.0 && s.alpha.is_finite() && s.alpha >= 0.0 && s.alpha <= s.cap,
                "steering requires finite external 0<=alpha<=cap");
            s.positions.validate(capacity)?;
        }
        let admitted_bytes = if let Some(c) = self.plan.capture {
            ensure!(!c.blocks.is_empty() && c.blocks.len() <= BLOCKS
                && c.blocks.iter().all(|&b| b < BLOCKS) && c.blocks.windows(2).all(|b| b[0] < b[1]),
                "capture blocks must be sorted unique zero-based indices below49");
            c.positions.validate(capacity)?;
            let needed = capture_bytes(c.blocks.len(), c.positions.count(&horizon), c.post_add)?;
            let sink = self.sink.as_ref().context("capture selection requires a bounded sink")?;
            ensure!(needed <= sink.max_bytes, "residual capture budget: {needed} > {} bytes", sink.max_bytes);
            needed
        } else {
            ensure!(self.sink.is_none(), "capture sink supplied without capture selection");
            0
        };
        if self.plan.capture.is_none() && self.plan.steering.is_none_or(|s| s.alpha == 0.0) {
            return Ok(None); // Validated zero/disabled request: no add or capture work.
        }
        Ok(Some(Prepared { request: self, horizon, admitted_bytes, used_bytes: 0 }))
    }
}

impl Prepared<'_, '_> {
    fn emit(&mut self, block: usize, position: usize, input_token_id: u32, phase: Phase,
        stage: CaptureStage, values: &[f32]) -> Result<()> {
        let end = self.used_bytes.checked_add(H * size_of::<f32>()).context("capture byte counter overflow")?;
        ensure!(end <= self.admitted_bytes, "capture exceeded preflight admission");
        self.used_bytes = end; // Attempted delivery counts even when callback fails.
        let sink = self.request.sink.as_mut().context("missing prepared capture sink")?;
        (sink.emit)(CapturedRow { source_identity: self.request.plan.source_identity, block,
            absolute_position: position, input_token_id, phase, stage, values })
    }

    pub(super) fn apply(&mut self, block: usize, ids: &[i32], positions: &[i32], phase: Phase,
        residual: &mut [f32]) -> Result<()> {
        let capture = self.request.plan.capture.filter(|c| c.blocks.binary_search(&block).is_ok());
        let steering = self.request.plan.steering.filter(|s| s.block == block && s.alpha != 0.0);
        if capture.is_none() && steering.is_none() { return Ok(()); }
        ensure!(ids.len() == positions.len() && ids.len().checked_mul(H) == Some(residual.len())
            && ids.iter().all(|&id| id >= 0) && positions.iter().all(|&p| p >= 0 && self.horizon.contains(&(p as usize)))
            && residual.iter().all(|v| v.is_finite()), "invalid completed block residual/input rows");
        for (index, row) in residual.chunks_exact_mut(H).enumerate() {
            let position = positions[index] as usize;
            let c = capture.filter(|c| c.positions.contains(position));
            if c.is_some() { self.emit(block, position, ids[index] as u32, phase, CaptureStage::PreAddLout, row)?; }
            if let Some(s) = steering.filter(|s| s.positions.contains(position)) {
                for (value, &direction) in row.iter_mut().zip(s.direction) { *value += s.alpha * direction; }
                ensure!(row.iter().all(|v| v.is_finite()), "nonfinite post-add residual at block {block}, position {position}");
            }
            if c.is_some_and(|c| c.post_add) {
                self.emit(block, position, ids[index] as u32, phase, CaptureStage::PostAddLout, row)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{forward::stage_f32, math::{Stage, Trace, Values}};

    fn identity() -> Value { serde_json::json!({"storage":"native-pile","model_root":"opaque-test-root","collection":"test","support_handles":["test-observation"]}) }
    fn plan<'a>(source: &'a Value, steering: Option<Steering<'a>>, capture: Option<Capture<'a>>) -> Plan<'a> {
        Plan { source_identity: source, steering, capture }
    }

    #[test]
    fn disabled_and_signed_zero_requests_do_not_touch_residuals() -> Result<()> {
        let source = identity(); let direction = vec![1.0; H];
        let baseline = [-0.0f32, 0.0, 1.5, -2.25].repeat(H / 4);
        for steering in [None, Some(Steering { block: 2, positions: Positions::AllAttended,
            direction: &direction, alpha: 0.0, cap: 1.0 }), Some(Steering { block: 2,
            positions: Positions::AllAttended, direction: &direction, alpha: -0.0, cap: 1.0 })] {
            let request = Request { plan: plan(&source, steering, None), sink: None };
            let mut prepared = request.prepare(&source, 3, 0..1)?;
            assert!(prepared.is_none());
            let mut values = baseline.clone();
            if let Some(p) = prepared.as_mut() { p.apply(2, &[7], &[0], Phase::Append, &mut values)?; }
            assert_eq!(values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(), baseline.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
        Ok(())
    }

    #[test]
    fn capture_trace_preadd_and_one_coordinate_selected_rows_layers() -> Result<()> {
        let source = identity(); let mut direction = vec![0.0; H]; direction[17] = 1.0;
        let original: Vec<f32> = (0..3*H).map(|i|(i / H) as f32 + 2.0).collect();
        let trace = Trace { enabled:true, router_margins:Vec::new(), stages:vec![Stage {name:"l_out".into(),dims:[H,3,1,1],values:Values::F32(original.clone())}] };
        let mut captured = Vec::new();
        let mut emit = |row: CapturedRow<'_>| {
            captured.push((row.block,row.absolute_position,row.input_token_id,row.phase,row.stage,row.values.to_vec())); Ok(())
        };
        let request = Request { plan: plan(&source,Some(Steering {block:8,positions:Positions::Only(&[5]),direction:&direction,alpha:0.5,cap:1.0}),
            Some(Capture {blocks:&[8],positions:Positions::Only(&[5]),post_add:true})),
            sink:Some(CaptureSink::new(2*H*4,&mut emit)) };
        let mut p=request.prepare(&source,10,4..7)?.unwrap();
        let mut residual=original.clone();
        p.apply(7,&[20,21,22],&[4,5,6],Phase::Prefill,&mut residual)?;
        assert_eq!(residual,original);
        p.apply(8,&[20,21,22],&[4,5,6],Phase::Prefill,&mut residual)?;
        assert_eq!(p.used_bytes,2*H*4); drop(p);
        for i in 0..residual.len() {assert_eq!(residual[i],original[i]+if i==H+17 {0.5}else{0.0});}
        assert_eq!(captured.len(),2);
        assert_eq!((captured[0].0,captured[0].1,captured[0].2,captured[0].3,captured[0].4),
            (8,5,21,Phase::Prefill,CaptureStage::PreAddLout));
        assert_eq!(captured[0].5,stage_f32(&trace,"l_out")?[H..2*H]);
        assert_eq!(captured[1].4,CaptureStage::PostAddLout);
        assert_eq!(captured[1].5,residual[H..2*H]);
        assert_eq!(stage_f32(&trace,"l_out")?,original); // Existing trace not overwritten.
        Ok(())
    }

    #[test]
    fn one_forward_captures_twelve_layers_without_direction() -> Result<()> {
        let source=identity(); let blocks:Vec<usize>=(4..49).step_by(4).collect();
        let mut seen=Vec::new();
        let mut emit=|row:CapturedRow<'_>| {seen.push((row.block,row.absolute_position,row.values[0]));Ok(())};
        let request=Request {plan:plan(&source,None,Some(Capture {blocks:&blocks,positions:Positions::Only(&[2]),post_add:false})),
            sink:Some(CaptureSink::new(12*H*4,&mut emit))};
        let mut p=request.prepare(&source,16,0..3)?.unwrap();
        for block in 0..BLOCKS {
            let mut residual=vec![block as f32;3*H];
            p.apply(block,&[1,2,3],&[0,1,2],Phase::Prefill,&mut residual)?;
            assert!(residual.iter().all(|&v|v==block as f32));
        }
        assert_eq!(p.used_bytes,12*H*4);drop(p);
        assert_eq!(seen,blocks.iter().map(|&b|(b,2,b as f32)).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn selection_is_absolute_and_chunk_invariant() -> Result<()> {
        let source=identity(); let direction=vec![1.0;H];
        let mut previous=None;
        for partition in [&[3][..],&[1,1,1][..],&[2,1][..]] {
            let mut captured=Vec::new();
            let mut emit=|row:CapturedRow<'_>| {captured.push((row.absolute_position,row.input_token_id,row.stage,row.values.to_vec()));Ok(())};
            let request=Request {plan:plan(&source,Some(Steering {block:4,positions:Positions::Only(&[10,12]),direction:&direction,alpha:0.25,cap:0.25}),
                Some(Capture {blocks:&[4],positions:Positions::Only(&[10,12]),post_add:true})),
                sink:Some(CaptureSink::new(4*H*4,&mut emit))};
            let mut p=request.prepare(&source,20,10..13)?.unwrap();
            let mut output=Vec::new();let mut first=0;
            for &len in partition {
                let ids:Vec<i32>=(first..first+len).map(|i|30+i as i32).collect();
                let positions:Vec<i32>=(first..first+len).map(|i|10+i as i32).collect();
                let mut values:Vec<f32>=(first..first+len).flat_map(|i|vec![i as f32;H]).collect();
                p.apply(4,&ids,&positions,Phase::Append,&mut values)?;
                output.extend(values);first+=len;
            }
            drop(p);
            let result=(output,captured);
            if let Some(reference)=&previous {assert_eq!(&result,reference);} else {previous=Some(result);}
        }
        Ok(())
    }

    #[test]
    fn preflight_refuses_invalid_binding_geometry_coefficients_and_budget() -> Result<()> {
        let source=identity();let direction=vec![1.0;H];
        let base=Steering {block:2,positions:Positions::AllAttended,direction:&direction,alpha:0.25,cap:1.0};
        for s in [Steering {block:49,..base},Steering {direction:&direction[..H-1],..base},
            Steering {alpha:f32::NAN,..base},Steering {alpha:f32::INFINITY,..base},Steering {alpha:-0.1,..base},
            Steering {alpha:1.1,..base},Steering {cap:f32::INFINITY,..base},Steering {cap:-1.0,..base},
            Steering {positions:Positions::Only(&[1,1]),..base},Steering {positions:Positions::Only(&[2,1]),..base},
            Steering {positions:Positions::Only(&[3]),..base}] {
            assert!(Request {plan:plan(&source,Some(s),None),sink:None}.prepare(&source,3,0..3).is_err());
        }
        let bad_direction=vec![f32::NAN;H];
        assert!(Request {plan:plan(&source,Some(Steering {direction:&bad_direction,alpha:0.0,..base}),None),sink:None}.prepare(&source,3,0..3).is_err());
        let other=serde_json::json!({"storage":"native-pile","model_root":"different"});
        assert!(Request {plan:plan(&other,Some(base),None),sink:None}.prepare(&source,3,0..3).is_err());
        let gguf=serde_json::json!({"storage":"gguf"});
        assert!(Request {plan:plan(&gguf,None,None),sink:None}.prepare(&gguf,3,0..3).is_err());
        for blocks in [&[][..],&[0,0][..],&[49][..],&[4,2][..]] {
            let mut emit=|_:CapturedRow<'_>|Ok(());
            assert!(Request {plan:plan(&source,None,Some(Capture {blocks,positions:Positions::AllAttended,post_add:false})),
                sink:Some(CaptureSink::new(usize::MAX,&mut emit))}.prepare(&source,3,0..3).is_err());
        }
        let c=Capture {blocks:&[4,8],positions:Positions::AllAttended,post_add:true};
        let needed=2*3*2*H*4;
        let mut emit=|_:CapturedRow<'_>|Ok(());
        assert!(Request {plan:plan(&source,None,Some(c)),sink:None}.prepare(&source,3,0..3).is_err());
        assert!(Request {plan:plan(&source,None,Some(c)),sink:Some(CaptureSink::new(needed-1,&mut emit))}.prepare(&source,3,0..3).is_err());
        assert!(Request {plan:plan(&source,None,Some(c)),sink:Some(CaptureSink::new(needed,&mut emit))}.prepare(&source,3,0..3)?.is_some());
        assert!(capture_bytes(49,usize::MAX,true).is_err());
        assert!(capture_bytes(usize::MAX,2,false).is_err());
        Ok(())
    }

    #[test]
    fn finite_inputs_can_overflow_addition_and_must_fail() -> Result<()> {
        let source=identity();let direction=vec![f32::MAX;H];
        let request=Request {plan:plan(&source,Some(Steering {block:48,positions:Positions::AllAttended,direction:&direction,alpha:2.0,cap:2.0}),None),sink:None};
        let mut p=request.prepare(&source,1,0..1)?.unwrap();
        assert!(p.apply(48,&[7],&[0],Phase::Decode,&mut vec![1.0;H]).is_err());
        Ok(())
    }
}
