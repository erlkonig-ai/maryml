//! Bounded CUDA-only, unpadded Qwen3.5 full-attention DECODER block.
//! Native BF16 weights; fresh prefill and one-token immutable KV append.
//! Correctness path: fixed per-row reduction order, no autotuning or CPU math.
//! Reference: pinned Transformers 5.2.0 modeling_qwen3_5.py:174-260,637-880.
//! All masks and projection biases are explicitly unsupported. Runtime driver,
//! allocation and compilation failures retain CubeCL's upstream error policy.
//! Resident producers retain CubeCL's ordinary client/handle and stream-ordering
//! obligations; matching device descriptors alone does not establish them.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use serde::{Deserialize, Serialize};
use triblespace::core::{blob::{Blob, encodings::tensor::{Tensor as NativeTensor, elements::BF16}},
    inline::{Inline, encodings::hash::Handle}, repo::{BlobStoreGet, pile::PileSnapshot}};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use super::gdn_mixer::project;
use super::decoder_ops::{norm, elementwise};

pub type CudaTensor = CubeTensor<CudaRuntime>;
pub type Slot<const R: usize> = Inline<Handle<NativeTensor<BF16, R>>>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Config {
    pub hidden: usize, pub intermediate: usize, pub heads: usize, pub kv_heads: usize,
    pub head_dim: usize, pub rotary_dim: usize, pub sections: [usize; 3],
    pub theta: f32, pub epsilon: f32, pub capacity: usize, pub attention_bias: bool,
}
impl Config {
    pub fn validate(self) -> Result<(), String> {
        if self.attention_bias || self.heads == 0 || self.kv_heads == 0
            || self.heads % self.kv_heads != 0 || !(1..=256).contains(&self.head_dim)
            || !(1..=16384).contains(&self.hidden) || !(1..=16384).contains(&self.intermediate)
            || !(1..=256).contains(&self.capacity) || self.rotary_dim == 0
            || self.rotary_dim > self.head_dim || self.rotary_dim % 2 != 0
            || !self.theta.is_finite() || self.theta <= 0.0
            || !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err("unsupported full-attention config: bias-free, positive bounded dimensions/theta/epsilon required".into());
        }
        let half = self.rotary_dim / 2;
        if self.sections.iter().try_fold(0usize, |n, &s| n.checked_add(s)) != Some(half)
            || self.sections[1] > (half + 1) / 3 || self.sections[2] > half / 3 {
            return Err("invalid interleaved partial MRoPE sections".into());
        }
        count(&[self.heads, self.head_dim, 2, self.hidden])?;
        count(&[self.kv_heads, self.head_dim, self.hidden])?;
        count(&[self.hidden, self.intermediate])?;
        Ok(())
    }
}

/// Position fixture/producer: p[axis,b,t] = start + b*batch_stride +
/// axis_base[axis] + (sequence_offset+t)*axis_step[axis], computed ON GPU.
/// This narrow explicit producer is not image/token position inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Positions { pub start: u32, pub batch_stride: u32, pub axis_base: [u32; 3], pub axis_step: [u32; 3] }
impl Positions {
    fn validate(self, batch: usize, end: usize) -> Result<(), String> {
        for axis in 0..3 {
            let value = (self.start as u64).checked_add((self.batch_stride as u64).checked_mul((batch - 1) as u64).ok_or("position overflow")?)
                .and_then(|n| n.checked_add(self.axis_base[axis] as u64))
                .and_then(|n| n.checked_add((self.axis_step[axis] as u64).checked_mul((end - 1) as u64)?))
                .ok_or("position overflow")?;
            if value > 16_777_216 { return Err("positions exceed exact F32 integer domain".into()); }
        }
        Ok(())
    }
}

pub struct Slots {
    pub input_norm: Slot<1>, pub post_norm: Slot<1>, pub q_norm: Slot<1>, pub k_norm: Slot<1>,
    pub q: Slot<2>, pub k: Slot<2>, pub v: Slot<2>, pub o: Slot<2>,
    pub gate: Slot<2>, pub up: Slot<2>, pub down: Slot<2>,
}
pub struct Block { config: Config, input_norm: CudaTensor, post_norm: CudaTensor,
    q_norm: CudaTensor, k_norm: CudaTensor, q: CudaTensor, k: CudaTensor, v: CudaTensor,
    o: CudaTensor, gate: CudaTensor, up: CudaTensor, down: CudaTensor }
/// Contiguous BF16 [B,L,Nkv,D], rotated normalized K and plain projected V.
/// Both are fresh output allocations. Input state is never modified.
pub struct State { pub key: CudaTensor, pub value: CudaTensor, positions: Positions, length: usize }
impl State { pub fn length(&self) -> usize { self.length } }
pub struct Output { pub hidden: CudaTensor, pub state: State }

impl Block {
    /// # Safety
    /// Genuine validated pile backing, INCLUDING each preceding partial page,
    /// must remain immutable/append-only and untruncated through CUDA runtime
    /// teardown. Same obligation as bind_pile_leaf. Late failure can retain
    /// earlier registrations; no transactional/last-handle reclamation claim.
    pub unsafe fn from_pile(snapshot: &PileSnapshot, s: Slots, config: Config, aliases: &mut CudaBf16Aliases) -> Result<Self, String> {
        config.validate()?;
        macro_rules! bind { ($slot:expr, $rank:literal) => {{
            let blob: Blob<NativeTensor<BF16, $rank>> = snapshot.get($slot).map_err(|e| e.to_string())?;
            // SAFETY: caller establishes genuine immutable file-prefix lifetime.
            unsafe { aliases.bind_pile_leaf(blob)? }
        }}; }
        let b = Self { config, input_norm: bind!(s.input_norm,1), post_norm: bind!(s.post_norm,1),
            q_norm: bind!(s.q_norm,1), k_norm: bind!(s.k_norm,1), q: bind!(s.q,2),
            k: bind!(s.k,2), v: bind!(s.v,2), o: bind!(s.o,2), gate: bind!(s.gate,2),
            up: bind!(s.up,2), down: bind!(s.down,2) };
        let c = config;
        for (t, shape) in [(&b.input_norm,vec![c.hidden]),(&b.post_norm,vec![c.hidden]),
            (&b.q_norm,vec![c.head_dim]),(&b.k_norm,vec![c.head_dim]),
            (&b.q,vec![2*c.heads*c.head_dim,c.hidden]),(&b.k,vec![c.kv_heads*c.head_dim,c.hidden]),
            (&b.v,vec![c.kv_heads*c.head_dim,c.hidden]),(&b.o,vec![c.hidden,c.heads*c.head_dim]),
            (&b.gate,vec![c.intermediate,c.hidden]),(&b.up,vec![c.intermediate,c.hidden]),
            (&b.down,vec![c.hidden,c.intermediate])] {
            check(t, &b.q, &shape)?;
            if t.handle.can_mut() { return Err("immutable pile weights required".into()); }
        }
        Ok(b)
    }
    pub fn prefill(&self, x: &CudaTensor, positions: Positions, mask: Option<&CudaTensor>) -> Result<Output,String> {
        self.forward(x, positions, None, mask)
    }
    pub fn decode(&self, x: &CudaTensor, state: &State, mask: Option<&CudaTensor>) -> Result<Output,String> {
        self.forward(x, state.positions, Some(state), mask)
    }
    #[cfg(feature="qwen3_5-real-layer-trace")]
    pub fn prefill_observed(&self,x:&CudaTensor,pos:Positions,trace:&mut super::outer_trace::OuterTrace)->Result<Output,String> {
        self.forward_observed(x,pos,None,None,Some(trace))
    }
    #[cfg(feature="qwen3_5-real-layer-trace")]
    pub fn decode_observed(&self,x:&CudaTensor,state:&State,trace:&mut super::outer_trace::OuterTrace)->Result<Output,String> {
        self.forward_observed(x,state.positions,Some(state),None,Some(trace))
    }
    #[cfg(feature="qwen3_5-real-layer-trace")]
    pub fn replay_outer(&self,x:&CudaTensor,hf:&super::outer_trace::OuterTrace)->Result<[CudaTensor;7],String> {
        super::outer_trace::replay(x,hf,&self.post_norm,&self.gate,&self.up,&self.down,
            self.config.hidden,self.config.intermediate,self.config.epsilon)
    }
    #[cfg(feature="qwen3_5-real-layer-trace")]
    pub fn replay_mlp_suffix(&self,hf:&super::outer_trace::OuterTrace,controls:&[CudaTensor;7])
        ->Result<super::outer_trace::MlpSuffixPair,String> {
        super::outer_trace::replay_mlp_suffix(hf,controls,&self.gate,&self.up,&self.down,
            self.config.hidden,self.config.intermediate)
    }
    fn forward(&self,x:&CudaTensor,positions:Positions,old:Option<&State>,mask:Option<&CudaTensor>)->Result<Output,String> {
        self.forward_observed(x,positions,old,mask,None)
    }
    fn forward_observed(&self, x: &CudaTensor, positions: Positions, old: Option<&State>, mask: Option<&CudaTensor>,
        mut trace:Option<&mut super::outer_trace::OuterTrace>) -> Result<Output,String> {
        if mask.is_some() { return Err("user masks/padding unsupported; causality is always enforced internally".into()); }
        let s = x.meta.shape().as_slice();
        if s.len()!=3 { return Err("hidden must be [B,T,H]".into()); }
        let (b,t)=(s[0],s[1]); let c=self.config;
        let past=old.map_or(0, |s| s.length);
        let end=past.checked_add(t).ok_or("cache length overflow")?;
        if t==0 || end>c.capacity || (old.is_some() && t!=1) { return Err("prefill/cache capacity exceeded or decode is not T1".into()); }
        check(x,&self.q,&[b,t,c.hidden])?;
        positions.validate(b,end)?;
        // Prove every derived allocation before any dispatch.
        for shape in [vec![b,t,2*c.heads*c.head_dim],vec![b,end,c.kv_heads,c.head_dim],
            vec![b,t,c.intermediate],vec![b,t,c.heads,end]] { count(&shape)?; }
        if let Some(s)=old {
            if past==0 { return Err("empty continuation cache".into()); }
            check(&s.key,x,&[b,past,c.kv_heads,c.head_dim])?;
            check(&s.value,x,&[b,past,c.kv_heads,c.head_dim])?;
        }
        let normalized=norm(x,&self.input_norm,c.hidden,c.epsilon);
        if let Some(t)=trace.as_deref_mut(){t.record(0,&normalized);}
        let packed=project(&normalized,&self.q)?;
        let (q,gate)=split(&packed,b,t,c.heads,c.head_dim);
        let q=norm(&q,&self.q_norm,c.head_dim,c.epsilon);
        let k=reshape(project(&normalized,&self.k)?, &[b,t,c.kv_heads,c.head_dim]);
        let k=norm(&k,&self.k_norm,c.head_dim,c.epsilon);
        let v=reshape(project(&normalized,&self.v)?, &[b,t,c.kv_heads,c.head_dim]);
        let q=rope(&q,c,positions,past); let k=rope(&k,c,positions,past);
        let key=append(&k,old.map(|s| &s.key),past);
        let value=append(&v,old.map(|s| &s.value),past);
        let attended=attention(&q,&key,&value,past,c);
        let gated=elementwise(&attended,&gate,2);
        let output=project(&reshape(gated,&[b,t,c.heads*c.head_dim]),&self.o)?;
        if let Some(t)=trace.as_deref_mut(){t.record(1,&output);}
        let residual=elementwise(x,&output,0);
        if let Some(t)=trace.as_deref_mut(){t.record(2,&residual);}
        let post=norm(&residual,&self.post_norm,c.hidden,c.epsilon);
        if let Some(t)=trace.as_deref_mut(){t.record(3,&post);}
        let gate=project(&post,&self.gate)?;
        if let Some(t)=trace.as_deref_mut(){t.record(4,&gate);}
        let up=project(&post,&self.up)?;
        if let Some(t)=trace.as_deref_mut(){t.record(5,&up);}
        let mlp=elementwise(&gate,&up,1);
        if let Some(t)=trace.as_deref_mut(){t.record(6,&mlp);}
        let down=project(&mlp,&self.down)?;
        if let Some(t)=trace.as_deref_mut(){t.record(7,&down);}
        let hidden=elementwise(&residual,&down,0);
        Ok(Output { hidden, state: State { key,value,positions,length:end } })
    }
}

fn count(s:&[usize])->Result<usize,String> {
    s.iter().try_fold(1usize,|n,&d| if d==0 {None} else {n.checked_mul(d)})
        .filter(|&n|n<=u32::MAX as usize).ok_or_else(||"empty/overflowing u32 shape".into())
}
fn check(t:&CudaTensor,like:&CudaTensor,s:&[usize])->Result<usize,String> {
    if t.meta.shape().as_slice()!=s || t.meta.strides().len()!=s.len() || t.dtype!=DType::BF16
        || t.device!=like.device || t.qparams.is_some() { return Err("wrong BF16 shape/dtype/device or quantized input".into()); }
    let n=count(s)?; let mut stride=1;
    for (i,&d) in s.iter().enumerate().rev() { if d>1 && t.meta.strides()[i]!=stride { return Err("contiguous storage required".into()); } stride*=d; }
    if t.handle.size_in_used()<(n as u64)*2 { return Err("insufficient storage".into()); }
    Ok(n)
}
fn reshape(t:CudaTensor,s:&[usize])->CudaTensor { CubeTensor::new_contiguous(t.client,t.device,s.into(),t.handle,t.dtype) }
fn tensor(like:&CudaTensor,s:&[usize],h:cubecl::server::Handle)->CudaTensor { CubeTensor::new_contiguous(like.client.clone(),like.device.clone(),s.into(),h,DType::BF16) }
fn len(t:&CudaTensor)->usize { t.meta.shape().as_slice().iter().product() }
fn grid(t:&CudaTensor,n:usize)->CubeCount { cubecl::calculate_cube_count_elemwise(&t.client,n,CubeDim::new_1d(64)) }

#[cube(launch_unchecked)]
fn split_kernel(x:&Array<bf16>,q:&mut Array<bf16>,gate:&mut Array<bf16>,n:usize,d:usize) {
    let i=ABSOLUTE_POS as usize; if i<n { let at=(i/d)*2*d+i%d; q[i]=x[at]; gate[i]=x[at+d]; }
}
fn split(x:&CudaTensor,b:usize,t:usize,h:usize,d:usize)->(CudaTensor,CudaTensor) {
    let n=b*t*h*d; let q=x.client.empty(n*2); let g=x.client.empty(n*2);
    unsafe { split_kernel::launch_unchecked::<CudaRuntime>(&x.client,grid(x,n),CubeDim::new_1d(64),
        ArrayArg::from_raw_parts(x.handle.clone(),2*n),ArrayArg::from_raw_parts(q.clone(),n),ArrayArg::from_raw_parts(g.clone(),n),n,d); }
    (tensor(x,&[b,t,h,d],q),tensor(x,&[b,t,h,d],g))
}
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rope_kernel(x:&Array<bf16>,y:&mut Array<bf16>,n:usize,tokens:usize,heads:usize,d:usize,r:usize,
    sh:usize,sw:usize,theta:f32,past:u32,start:u32,bs:u32,a0:u32,a1:u32,a2:u32,s0:u32,s1:u32,s2:u32) {
    let i=ABSOLUTE_POS as usize;
    if i<n { let j=i%d;
        if j>=r { y[i]=x[i]; } else {
            let half=r/2; let f=j%half; let row=i/d; let token=row/heads%tokens; let batch=row/(heads*tokens);
            let mut base=a0; let mut step=s0;
            if f%3==1 && f<sh*3 { base=a1; step=s1; }
            if f%3==2 && f<sw*3 { base=a2; step=s2; }
            let position=start+(batch as u32)*bs+base+(past+token as u32)*step;
            let inv=1.0f32/theta.powf(f32::cast_from(2*f)/f32::cast_from(r));
            let angle=inv*f32::cast_from(position);
            let cos=f32::cast_from(bf16::cast_from(angle.cos())); let sin=f32::cast_from(bf16::cast_from(angle.sin()));
            let low=f32::cast_from(x[row*d+f]); let high=f32::cast_from(x[row*d+half+f]);
            if j<half { y[i]=bf16::cast_from(f32::cast_from(bf16::cast_from(low*cos))-f32::cast_from(bf16::cast_from(high*sin))); }
            else { y[i]=bf16::cast_from(f32::cast_from(bf16::cast_from(high*cos))+f32::cast_from(bf16::cast_from(low*sin))); }
        }
    }
}
fn rope(x:&CudaTensor,c:Config,p:Positions,past:usize)->CudaTensor {
    let s=x.meta.shape().as_slice(); let n=len(x); let out=x.client.empty(n*2);
    unsafe { rope_kernel::launch_unchecked::<CudaRuntime>(&x.client,grid(x,n),CubeDim::new_1d(64),
        ArrayArg::from_raw_parts(x.handle.clone(),n),ArrayArg::from_raw_parts(out.clone(),n),
        n,s[1],s[2],c.head_dim,c.rotary_dim,c.sections[1],c.sections[2],c.theta,past as u32,p.start,p.batch_stride,
        p.axis_base[0],p.axis_base[1],p.axis_base[2],p.axis_step[0],p.axis_step[1],p.axis_step[2]); }
    tensor(x,s,out)
}
#[cube(launch_unchecked)]
fn append_kernel(old:&Array<bf16>,new:&Array<bf16>,out:&mut Array<bf16>,n:usize,past:usize,tokens:usize,width:usize) {
    let i=ABSOLUTE_POS as usize; if i<n { let at=i/width; let coord=i%width; let total=past+tokens; let b=at/total; let t=at%total;
        if t<past { out[i]=old[(b*past+t)*width+coord]; } else { out[i]=new[(b*tokens+t-past)*width+coord]; }
    }
}
fn append(new:&CudaTensor,old:Option<&CudaTensor>,past:usize)->CudaTensor {
    let s=new.meta.shape().as_slice(); let shape=[s[0],past+s[1],s[2],s[3]];
    let n=shape.iter().product(); let out=new.client.empty(n*2); let oldh=old.map(|x|x.handle.clone()).unwrap_or_else(||out.clone());
    unsafe { append_kernel::launch_unchecked::<CudaRuntime>(&new.client,grid(new,n),CubeDim::new_1d(64),
        ArrayArg::from_raw_parts(oldh,old.map_or(n,len)),ArrayArg::from_raw_parts(new.handle.clone(),len(new)),
        ArrayArg::from_raw_parts(out.clone(),n),n,past,s[1],s[2]*s[3]); }
    tensor(new,&shape,out)
}
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scores_kernel(q:&Array<bf16>,k:&Array<bf16>,p:&mut Array<bf16>,rows:usize,tokens:usize,heads:usize,kv:usize,d:usize,total:usize,past:usize,
    #[comptime] capacity:usize) {
    let row=ABSOLUTE_POS as usize;
    if row<rows { let h=row%heads; let t=row/heads%tokens; let b=row/(heads*tokens); let kh=h/(heads/kv);
        let available=past+t+1; let mut scores=Array::<f32>::new(capacity); let mut max=f32::cast_from(-3.4028235e38f32);
        let scale=1.0f32/f32::cast_from(d).sqrt();
        for key in 0..available { let mut dot=0.0f32;
            for j in 0..d { dot+=f32::cast_from(q[row*d+j])*f32::cast_from(k[((b*total+key)*kv+kh)*d+j]); }
            let score=f32::cast_from(bf16::cast_from(f32::cast_from(bf16::cast_from(dot))*scale));
            scores[key]=score; if score>max { max=score; }
        }
        let mut sum=0.0f32;
        for key in 0..available { scores[key]=(scores[key]-max).exp(); sum+=scores[key]; }
        for key in 0..total { if key<available { p[row*total+key]=bf16::cast_from(scores[key]/sum); }
            else { p[row*total+key]=bf16::cast_from(0.0f32); } }
    }
}
#[cube(launch_unchecked)]
fn value_kernel(p:&Array<bf16>,v:&Array<bf16>,out:&mut Array<bf16>,n:usize,tokens:usize,heads:usize,kv:usize,d:usize,total:usize,past:usize) {
    let i=ABSOLUTE_POS as usize; if i<n { let row=i/d; let j=i%d; let h=row%heads; let t=row/heads%tokens;
        let b=row/(heads*tokens); let kh=h/(heads/kv); let mut sum=0.0f32;
        for key in 0..past+t+1 { sum+=f32::cast_from(p[row*total+key])*f32::cast_from(v[((b*total+key)*kv+kh)*d+j]); }
        out[i]=bf16::cast_from(sum);
    }
}
fn attention(q:&CudaTensor,k:&CudaTensor,v:&CudaTensor,past:usize,c:Config)->CudaTensor {
    let s=q.meta.shape().as_slice(); let total=past+s[1]; let rows=s[0]*s[1]*c.heads; let n=len(q);
    let probs=q.client.empty(rows*total*2); let out=q.client.empty(n*2);
    unsafe {
        scores_kernel::launch_unchecked::<CudaRuntime>(&q.client,grid(q,rows),CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(q.handle.clone(),n),ArrayArg::from_raw_parts(k.handle.clone(),len(k)),
            ArrayArg::from_raw_parts(probs.clone(),rows*total),rows,s[1],c.heads,c.kv_heads,c.head_dim,total,past,c.capacity);
        value_kernel::launch_unchecked::<CudaRuntime>(&q.client,grid(q,n),CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(probs,rows*total),ArrayArg::from_raw_parts(v.handle.clone(),len(v)),
            ArrayArg::from_raw_parts(out.clone(),n),n,s[1],c.heads,c.kv_heads,c.head_dim,total,past);
    }
    tensor(q,s,out)
}

// Separate fresh-prefill entry: the existing affine/decode path above is
// byte-unchanged. A position table does not fabricate an affine cache state.
impl Block {
    /// Fresh B1, no padding. The caller's all-ones mask policy means ordinary
    /// causal token order, independent of repeated image temporal positions.
    /// No continuation state is returned or retained by this entry point.
    pub fn prefill_positioned(
        &self, x: &CudaTensor, positions: &super::position_table::PositionTable,
    ) -> Result<CudaTensor, String> {
        let shape = x.meta.shape().as_slice();
        let c = self.config;
        if shape.len() != 3 || shape[0] != 1 || shape[1] == 0 || shape[1] > c.capacity {
            return Err("position-table prefill requires B1, nonempty bounded unpadded tokens".into());
        }
        let tokens = shape[1];
        check(x, &self.q, &[1,tokens,c.hidden])?;
        if !std::ptr::eq(x.client.properties(), self.q.client.properties()) {
            return Err("position-table input and weights require the same CUDA client".into());
        }
        positions.validate(x, tokens)?;
        for shape in [vec![1,tokens,2*c.heads*c.head_dim], vec![1,tokens,c.kv_heads,c.head_dim],
            vec![1,tokens,c.intermediate], vec![1,tokens,c.heads,tokens]] { count(&shape)?; }
        let normalized = norm(x, &self.input_norm, c.hidden, c.epsilon);
        let packed = project(&normalized, &self.q)?;
        let (q, gate) = split(&packed, 1, tokens, c.heads, c.head_dim);
        let q = norm(&q, &self.q_norm, c.head_dim, c.epsilon);
        let k = reshape(project(&normalized, &self.k)?, &[1,tokens,c.kv_heads,c.head_dim]);
        let k = norm(&k, &self.k_norm, c.head_dim, c.epsilon);
        let v = reshape(project(&normalized, &self.v)?, &[1,tokens,c.kv_heads,c.head_dim]);
        let q = positioned_rope(&q, c, positions);
        let k = positioned_rope(&k, c, positions);
        let attended = attention(&q, &k, &v, 0, c);
        let gated = elementwise(&attended, &gate, 2);
        let output = project(&reshape(gated, &[1,tokens,c.heads*c.head_dim]), &self.o)?;
        let residual = elementwise(x, &output, 0);
        let post = norm(&residual, &self.post_norm, c.hidden, c.epsilon);
        let gate = project(&post, &self.gate)?;
        let up = project(&post, &self.up)?;
        let mlp = elementwise(&gate, &up, 1);
        let down = project(&mlp, &self.down)?;
        Ok(elementwise(&residual, &down, 0))
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn positioned_rope_kernel(
    x: &Array<bf16>, positions: &Array<u32>, y: &mut Array<bf16>,
    n: usize, tokens: usize, heads: usize, d: usize, r: usize,
    sh: usize, sw: usize, theta: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let j = i % d;
        if j >= r { y[i] = x[i]; } else {
            let half = r / 2;
            let f = j % half;
            let row = i / d;
            let token = row / heads % tokens;
            let mut axis = 0usize;
            if f % 3 == 1 && f < sh * 3 { axis = 1; }
            if f % 3 == 2 && f < sw * 3 { axis = 2; }
            let position = positions[axis * tokens + token];
            let inv = 1.0f32 / theta.powf(f32::cast_from(2 * f) / f32::cast_from(r));
            let angle = inv * f32::cast_from(position);
            let cos = f32::cast_from(bf16::cast_from(angle.cos()));
            let sin = f32::cast_from(bf16::cast_from(angle.sin()));
            let low = f32::cast_from(x[row * d + f]);
            let high = f32::cast_from(x[row * d + half + f]);
            if j < half {
                y[i] = bf16::cast_from(f32::cast_from(bf16::cast_from(low * cos)) - f32::cast_from(bf16::cast_from(high * sin)));
            } else {
                y[i] = bf16::cast_from(f32::cast_from(bf16::cast_from(high * cos)) + f32::cast_from(bf16::cast_from(low * sin)));
            }
        }
    }
}

fn positioned_rope(x: &CudaTensor, c: Config, positions: &super::position_table::PositionTable) -> CudaTensor {
    let shape = x.meta.shape().as_slice();
    let n = len(x);
    let out = x.client.empty(n * 2);
    unsafe {
        positioned_rope_kernel::launch_unchecked::<CudaRuntime>(
            &x.client, grid(x,n), CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(x.handle.clone(),n),
            ArrayArg::from_raw_parts(positions.tensor().handle.clone(),3*shape[1]),
            ArrayArg::from_raw_parts(out.clone(),n),
            n,shape[1],shape[2],c.head_dim,c.rotary_dim,c.sections[1],c.sections[2],c.theta,
        );
    }
    tensor(x,shape,out)
}

#[cfg(test)]
mod positioned_tests {
    use super::*;
    use cubecl::cuda::CudaDevice;
    use super::super::position_table::PositionTable;

    #[cube(launch_unchecked)]
    fn fill_position_fixture(out: &mut Array<bf16>, n: usize, offset: usize) {
        let i=ABSOLUTE_POS as usize;
        if i<n { out[i]=bf16::cast_from((f32::cast_from(((i+offset)*17)%251)-125.0f32)/128.0f32); }
    }

    fn fixture(tokens: usize, offset: usize) -> CudaTensor {
        let device=CudaDevice{index:0};
        let client=CudaRuntime::client(&device);
        let n=tokens*16*256;
        let out=CubeTensor::new_contiguous(client.clone(),device,[1,tokens,16,256].into(),client.empty(n*2),DType::BF16);
        unsafe { fill_position_fixture::launch_unchecked::<CudaRuntime>(&client,grid(&out,n),CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(out.handle.clone(),n),n,offset); }
        out
    }

    #[test]
    #[ignore="requires reserved CUDA; table indexing/unchanged arithmetic, not HF admission"]
    fn explicit_positions_match_affine_and_per_token_three_axis_control() {
        let c=Config{hidden:4096,intermediate:12288,heads:16,kv_heads:4,head_dim:256,
            rotary_dim:64,sections:[11,11,10],theta:10_000_000.0,epsilon:1e-6,capacity:256,attention_bias:false};
        let x=fixture(4,0);
        let sequential=PositionTable::from_positions(&x,&[[0;3],[1;3],[2;3],[3;3]]).unwrap();
        let expected=rope(&x,c,Positions{start:0,batch_stride:0,axis_base:[0;3],axis_step:[1;3]},0);
        let actual=positioned_rope(&x,c,&sequential);
        assert_eq!(actual.client.read_one(actual.handle.clone()).unwrap().to_vec(),expected.client.read_one(expected.handle.clone()).unwrap().to_vec());
        let axes=[[2,2,2],[2,2,3],[2,3,2],[4,4,4]];
        let table=PositionTable::from_positions(&x,&axes).unwrap();
        assert!(table.validate(&x,3).is_err());
        let actual=positioned_rope(&x,c,&table);
        let actual=actual.client.read_one(actual.handle.clone()).unwrap().to_vec();
        for (token,&axis_base) in axes.iter().enumerate() {
            let one=fixture(1,token*16*256);
            let expected=rope(&one,c,Positions{start:0,batch_stride:0,axis_base,axis_step:[0;3]},0);
            let expected=expected.client.read_one(expected.handle.clone()).unwrap().to_vec();
            assert_eq!(&actual[token*8192..(token+1)*8192],expected.as_slice());
        }
    }
}
