//! Diagnostic-only observations of existing decoder operations; no model formula.
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use super::{decoder_ops::{norm,elementwise},gdn_decoder::check,gdn_mixer::project};
pub type CudaTensor=CubeTensor<CudaRuntime>;
pub const NAMES:[&str;8]=["input_norm","mixer_out","residual1","post_norm","mlp_gate","mlp_up","mlp_product","mlp_down"];
pub const CONTROL_NAMES:[&str;7]=["residual1","post_norm","mlp_gate","mlp_up","mlp_product","mlp_down","output"];
/// Ephemeral per-invocation handles, never a model/weight catalogue.
pub struct OuterTrace { stages:[Option<CudaTensor>;8], duplicate:bool }
impl Default for OuterTrace { fn default()->Self { Self{stages:std::array::from_fn(|_|None),duplicate:false} } }
impl OuterTrace {
    pub(super) fn record(&mut self,index:usize,tensor:&CudaTensor) {
        self.duplicate |= self.stages[index].is_some();
        self.stages[index]=Some(tensor.clone());
    }
    pub fn get(&self,index:usize)->Result<&CudaTensor,String> {
        if self.duplicate { return Err("duplicate outer observation".into()); }
        self.stages.get(index).and_then(Option::as_ref).ok_or_else(||"missing outer observation".into())
    }
    /// Diagnostic fixture only: validate all eight descriptors before replay.
    pub fn from_stages(stages:[CudaTensor;8])->Self {
        Self{stages:stages.map(Some),duplicate:false}
    }
}
/// Calls only the existing kernels, using exact immediate HF BF16 operands.
pub(super) fn replay(x:&CudaTensor,hf:&OuterTrace,post:&CudaTensor,gate:&CudaTensor,
    up:&CudaTensor,down:&CudaTensor,h:usize,m:usize,epsilon:f32)->Result<[CudaTensor;7],String> {
    let s=x.meta.shape().as_slice();
    if s.len()!=3 { return Err("outer replay hidden rank".into()); }
    let (b,t)=(s[0],s[1]);
    check(x,post,"outer x",&[b,t,h],DType::BF16)?;
    for i in 0..8 {
        let width=if (4..=6).contains(&i){m}else{h};
        check(hf.get(i)?,x,NAMES[i],&[b,t,width],DType::BF16)?;
    }
    check(post,x,"post gain",&[h],DType::BF16)?;
    check(gate,x,"gate weight",&[m,h],DType::BF16)?;
    check(up,x,"up weight",&[m,h],DType::BF16)?;
    check(down,x,"down weight",&[h,m],DType::BF16)?;
    Ok([
        elementwise(x,hf.get(1)?,0),
        norm(hf.get(2)?,post,h,epsilon),
        project(hf.get(3)?,gate)?,
        project(hf.get(3)?,up)?,
        elementwise(hf.get(4)?,hf.get(5)?,1),
        project(hf.get(6)?,down)?,
        elementwise(hf.get(2)?,hf.get(7)?,0),
    ])
}

#[cfg(feature="qwen3_5-real-layer-trace")]
pub const SUFFIX_NAMES:[&str;5]=["mlp_gate","mlp_up","mlp_product","mlp_down","output"];
/// Ephemeral diagnostic results, not a replacement decoder path.
#[cfg(feature="qwen3_5-real-layer-trace")]
pub struct MlpSuffixPair { pub a:[CudaTensor;5], pub b:[CudaTensor;5] }

/// The caller supplies the SAME invocation's immediate replay results:
/// controls[1]=native norm(HF residual1), controls[2/3]=native gate/up(HF post_norm).
/// Descriptor checks do not establish this provenance; the gate witnesses it.
/// Both suffixes retain their own outputs between operations, with no HF reset.
#[cfg(feature="qwen3_5-real-layer-trace")]
pub(super) fn replay_mlp_suffix(hf:&OuterTrace,controls:&[CudaTensor;7],
    gate:&CudaTensor,up:&CudaTensor,down:&CudaTensor,h:usize,m:usize)->Result<MlpSuffixPair,String> {
    let residual=hf.get(2)?;
    let shape=residual.meta.shape().as_slice();
    if shape.len()!=3 { return Err("suffix residual rank".into()); }
    let (b,t)=(shape[0],shape[1]);
    // All operands and weights are checked before the first additional dispatch.
    check(residual,gate,"suffix HF residual",&[b,t,h],DType::BF16)?;
    check(hf.get(3)?,residual,"suffix HF post",&[b,t,h],DType::BF16)?;
    check(&controls[1],residual,"suffix native post",&[b,t,h],DType::BF16)?;
    check(&controls[2],residual,"suffix A gate",&[b,t,m],DType::BF16)?;
    check(&controls[3],residual,"suffix A up",&[b,t,m],DType::BF16)?;
    check(gate,residual,"suffix gate weight",&[m,h],DType::BF16)?;
    check(up,residual,"suffix up weight",&[m,h],DType::BF16)?;
    check(down,residual,"suffix down weight",&[h,m],DType::BF16)?;
    // A reuses the existing native projections on exact HF post_norm.
    let a_gate=controls[2].clone(); let a_up=controls[3].clone();
    let a_product=elementwise(&a_gate,&a_up,1);
    let a_down=project(&a_product,down)?;
    let a_output=elementwise(residual,&a_down,0);
    // B differs only in the starting post-normalized BF16 tensor. Projection
    // order is gate then up, matching the existing immediate replay controls.
    let b_gate=project(&controls[1],gate)?;
    let b_up=project(&controls[1],up)?;
    let b_product=elementwise(&b_gate,&b_up,1);
    let b_down=project(&b_product,down)?;
    let b_output=elementwise(residual,&b_down,0);
    Ok(MlpSuffixPair{a:[a_gate,a_up,a_product,a_down,a_output],
        b:[b_gate,b_up,b_product,b_down,b_output]})
}
