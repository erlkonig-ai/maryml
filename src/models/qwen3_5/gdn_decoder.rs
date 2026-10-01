//! Bounded unmasked CUDA Qwen3.5 linear-attention DECODER layer.
//! Native BF16 typed slots -> zero-centered input norm -> unchanged GDN mixer
//! -> residual -> zero-centered post norm -> SiLU(gate)*up -> down -> residual.
//! Shared decoder_ops is the same numerical path used by full_attention.
//! No whole backbone, checkpoint-name catalogue, F16 shim or host model math.
//! Runtime initialization/allocation/driver failures retain upstream behavior;
//! Result covers descriptors, not a promise to catch asynchronous GPU errors.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use serde::{Deserialize, Serialize};
use triblespace::core::{
    blob::{Blob, encodings::tensor::{Tensor as NativeTensor, elements::BF16}},
    inline::{Inline, encodings::hash::Handle},
    repo::{BlobStoreGet, pile::PileSnapshot},
};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use super::{decoder_ops::{norm, elementwise},
    gdn_mixer::{GdnConfig, GdnMixer, GdnSlots, GdnState, project}};

pub type CudaTensor = CubeTensor<CudaRuntime>;
pub type Slot<const R: usize> = Inline<Handle<NativeTensor<BF16, R>>>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Config { pub mixer: GdnConfig, pub intermediate: usize }
impl Config {
    pub fn validate(self) -> Result<(), String> {
        self.mixer.validate()?;
        if !(1..=16384).contains(&self.mixer.hidden) || !(1..=16384).contains(&self.intermediate) {
            return Err("decoder H/M must be in 1..=16384".into());
        }
        count(&[self.mixer.hidden, self.intermediate])?;
        Ok(())
    }
}

/// Fourteen concrete typed native slots, queried at their consuming roles.
pub struct Slots {
    pub mixer: GdnSlots,
    pub input_norm: Slot<1>, pub post_norm: Slot<1>,
    pub gate: Slot<2>, pub up: Slot<2>, pub down: Slot<2>,
}
pub struct Block {
    config: Config, mixer: GdnMixer,
    input_norm: CudaTensor, post_norm: CudaTensor,
    gate: CudaTensor, up: CudaTensor, down: CudaTensor,
}
pub struct Output { pub hidden: CudaTensor, pub state: GdnState }

impl Block {
    /// # Safety
    /// Snapshot leaves must originate in a genuine validated immutable
    /// append-only pile prefix, including the preceding partial page. No
    /// truncation/rewrite through CUDA runtime teardown; forwarded unchanged
    /// to the native binder and GdnMixer. Late errors may retain registrations.
    pub unsafe fn from_pile(
        snapshot: &PileSnapshot, slots: Slots, config: Config, aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        config.validate()?;
        macro_rules! bind { ($slot:expr, $rank:literal) => {{
            let blob: Blob<NativeTensor<BF16, $rank>> = snapshot.get($slot).map_err(|e| e.to_string())?;
            // SAFETY: caller provides the actual immutable file-prefix premise.
            unsafe { aliases.bind_pile_leaf(blob)? }
        }}; }
        // SAFETY: same genuine snapshot and immutable lifetime as the outer leaves.
        let mixer = unsafe { GdnMixer::from_pile(snapshot, slots.mixer, config.mixer, aliases)? };
        let result = Self { config, mixer,
            input_norm: bind!(slots.input_norm, 1), post_norm: bind!(slots.post_norm, 1),
            gate: bind!(slots.gate, 2), up: bind!(slots.up, 2), down: bind!(slots.down, 2) };
        let (h, m) = (config.mixer.hidden, config.intermediate);
        for (name, tensor, shape) in [
            ("input norm", &result.input_norm, vec![h]), ("post norm", &result.post_norm, vec![h]),
            ("gate", &result.gate, vec![m,h]), ("up", &result.up, vec![m,h]), ("down", &result.down, vec![h,m]),
        ] {
            check(tensor, &result.input_norm, name, &shape, DType::BF16)?;
            if tensor.handle.can_mut() { return Err(format!("{name}: immutable pile weight required")); }
        }
        Ok(result)
    }

    pub fn prefill(&self, hidden: &CudaTensor, mask: Option<&CudaTensor>) -> Result<Output, String> {
        self.forward(hidden, None, mask)
    }
    pub fn decode(&self, hidden: &CudaTensor, state: &GdnState, mask: Option<&CudaTensor>) -> Result<Output, String> {
        self.forward(hidden, Some(state), mask)
    }
    #[cfg(feature="qwen3_5-real-layer-trace")]
    pub fn prefill_observed(&self,x:&CudaTensor,trace:&mut super::outer_trace::OuterTrace)->Result<Output,String> {
        self.forward_observed(x,None,None,Some(trace))
    }
    #[cfg(feature="qwen3_5-real-layer-trace")]
    pub fn decode_observed(&self,x:&CudaTensor,state:&GdnState,trace:&mut super::outer_trace::OuterTrace)->Result<Output,String> {
        self.forward_observed(x,Some(state),None,Some(trace))
    }
    #[cfg(feature="qwen3_5-real-layer-trace")]
    pub fn replay_outer(&self,x:&CudaTensor,hf:&super::outer_trace::OuterTrace)->Result<[CudaTensor;7],String> {
        super::outer_trace::replay(x,hf,&self.post_norm,&self.gate,&self.up,&self.down,
            self.config.mixer.hidden,self.config.intermediate,self.config.mixer.epsilon)
    }
    #[cfg(feature="qwen3_5-real-layer-trace")]
    pub fn replay_mlp_suffix(&self,hf:&super::outer_trace::OuterTrace,controls:&[CudaTensor;7])
        ->Result<super::outer_trace::MlpSuffixPair,String> {
        super::outer_trace::replay_mlp_suffix(hf,controls,&self.gate,&self.up,&self.down,
            self.config.mixer.hidden,self.config.intermediate)
    }
    fn forward(&self,hidden:&CudaTensor,state:Option<&GdnState>,mask:Option<&CudaTensor>)->Result<Output,String> {
        self.forward_observed(hidden,state,mask,None)
    }
    fn forward_observed(&self, hidden: &CudaTensor, state: Option<&GdnState>, mask: Option<&CudaTensor>,
        mut trace:Option<&mut super::outer_trace::OuterTrace>) -> Result<Output, String> {
        if mask.is_some() { return Err("GDN decoder supports unmasked inputs only".into()); }
        let shape = hidden.meta.shape().as_slice();
        if shape.len() != 3 { return Err("hidden must be [B,T,H]".into()); }
        let (b, t) = (shape[0], shape[1]);
        let c = self.config.mixer;
        if !(1..=256).contains(&t) || (state.is_some() && t != 1) {
            return Err("prefill requires T=1..256; decode requires T1".into());
        }
        check(hidden, &self.input_norm, "hidden", &[b,t,c.hidden], DType::BF16)?;
        let channels = 2*c.key_heads*c.key_dim + c.value_heads*c.value_dim;
        // Validate outer AND inner extents/state before the first norm launch.
        count(&[b,t,self.config.intermediate])?;
        count(&[b,t,channels])?;
        count(&[b,c.value_heads,c.key_dim,c.value_dim])?;
        count(&[b,channels,c.conv_kernel])?;
        if let Some(s) = state {
            check(&s.conv, hidden, "conv state", &[b,channels,c.conv_kernel], DType::BF16)?;
            check(&s.recurrent, hidden, "recurrent state", &[b,c.value_heads,c.key_dim,c.value_dim], DType::F32)?;
        }
        let normalized = norm(hidden, &self.input_norm, c.hidden, c.epsilon);
        if let Some(t)=trace.as_deref_mut(){t.record(0,&normalized);}
        let mixed = match state {
            Some(s) => self.mixer.decode(&normalized, s, None)?,
            None => self.mixer.prefill(&normalized, None)?,
        };
        if let Some(t)=trace.as_deref_mut(){t.record(1,&mixed.hidden);}
        let residual = elementwise(hidden, &mixed.hidden, 0);
        if let Some(t)=trace.as_deref_mut(){t.record(2,&residual);}
        let post = norm(&residual, &self.post_norm, c.hidden, c.epsilon);
        if let Some(t)=trace.as_deref_mut(){t.record(3,&post);}
        let up = project(&post, &self.up)?;
        if let Some(t)=trace.as_deref_mut(){t.record(5,&up);}
        let gate = project(&post, &self.gate)?;
        if let Some(t)=trace.as_deref_mut(){t.record(4,&gate);}
        let activated = elementwise(&gate, &up, 1);
        if let Some(t)=trace.as_deref_mut(){t.record(6,&activated);}
        let output = project(&activated, &self.down)?;
        if let Some(t)=trace.as_deref_mut(){t.record(7,&output);}
        Ok(Output { hidden: elementwise(&residual, &output, 0), state: mixed.state })
    }
}

// Descriptor proof only, not a divergent numerical implementation. Matching
// device does not replace upstream valid client/handle/stream producer duties.
fn count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
        .filter(|&n| n <= u32::MAX as usize).ok_or_else(|| "empty/overflowing u32 shape".into())
}
pub(super) fn check(t: &CudaTensor, like: &CudaTensor, name: &str, shape: &[usize], dtype: DType) -> Result<(), String> {
    if t.meta.shape().as_slice() != shape || t.meta.strides().len() != shape.len()
        || t.dtype != dtype || t.qparams.is_some() || t.device != like.device {
        return Err(format!("{name}: wrong shape/dtype/device or quantized storage"));
    }
    let n = count(shape)?;
    let mut stride = 1;
    for (axis, &d) in shape.iter().enumerate().rev() {
        if d > 1 && t.meta.strides()[axis] != stride { return Err(format!("{name}: contiguous storage required")); }
        stride *= d;
    }
    let bytes = n.checked_mul(if dtype == DType::BF16 { 2 } else { 4 }).ok_or("byte extent overflow")?;
    if t.handle.size_in_used() < bytes as u64 { return Err(format!("{name}: insufficient storage")); }
    Ok(())
}
