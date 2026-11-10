//! Six native BF16 slots: row LayerNorm -> adjacent 2x2 grouping -> biased
//! linear -> erf GELU -> biased linear. No vision blocks, tower or pooling.
//! Fixed per-row/per-cell ascending reductions; no host numeric path/atomics.
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use serde::{Deserialize,Serialize};
use triblespace::core::{blob::{Blob,encodings::tensor::{Tensor as NativeTensor,elements::BF16}},
    inline::{Inline,encodings::hash::Handle},repo::BlobStoreGet};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use super::gdn_mixer::{DecoderPrecision,Linear,bind_linear};
use super::roles::LinearSlot;
pub type CudaTensor=CubeTensor<CudaRuntime>;
pub type Slot<const R:usize>=Inline<Handle<NativeTensor<BF16,R>>>;
#[derive(Clone,Copy,Debug,Deserialize,Serialize)]
pub struct Config {pub hidden:usize,pub output:usize}
pub struct Slots {pub norm:Slot<1>,pub norm_bias:Slot<1>,pub fc1:Slot<2>,pub bias1:Slot<1>,pub fc2:Slot<2>,pub bias2:Slot<1>}
pub struct Merger {c:Config,norm:CudaTensor,norm_bias:CudaTensor,fc1:Linear,bias1:CudaTensor,fc2:Linear,bias2:CudaTensor}
pub struct Output {pub norm:CudaTensor,pub grouped:CudaTensor,pub fc1:CudaTensor,pub activated:CudaTensor,pub hidden:CudaTensor}
impl Config {
    pub fn validate(self)->Result<(),String> {
        if !(1..=1152).contains(&self.hidden)||!(1..=4096).contains(&self.output) {return Err("merger requires H1..1152,O1..4096,merge exactly2".into());}
        count(&[4*self.hidden,4*self.hidden])?;count(&[self.output,4*self.hidden])?;Ok(())
    }
}
impl Merger {
    /// # Safety
    /// Actual validated immutable append-only pile backing, including preceding
    /// partial pages, must remain unmodified/untruncated through CUDA runtime
    /// teardown. No generic owner proves provenance. Late errors may retain aliases.
    pub unsafe fn from_pile<R: BlobStoreGet>(snapshot:&R,s:Slots,c:Config,a:&mut CudaBf16Aliases)->Result<Self,String> {
        // SAFETY: forwarded unchanged.
        unsafe {Self::from_pile_with(snapshot,s,c,a,DecoderPrecision::Bf16)}
    }
    /// [`Self::from_pile`] with both linears on `precision`, as
    /// `vision_block::Block::from_pile_with` binds a block's.
    ///
    /// # Safety
    /// As [`Self::from_pile`].
    pub unsafe fn from_pile_with<R: BlobStoreGet>(snapshot:&R,s:Slots,c:Config,a:&mut CudaBf16Aliases,precision:DecoderPrecision)->Result<Self,String> {
        c.validate()?;
        macro_rules! bind {($h:expr,$r:literal)=>{{
            let b:Blob<NativeTensor<BF16,$r>>=snapshot.get($h).map_err(|e|e.to_string())?;
            // SAFETY: caller supplies the genuine immutable-prefix premise.
            unsafe {a.bind_pile_leaf(b)?}
        }};}
        let w=4*c.hidden;
        macro_rules! bind_lin {($slot:expr,$n:expr,$k:expr)=>{{
            if super::vision_block::nvfp4_takes(precision,$n,$k){
                // SAFETY: forwarded genuine immutable-prefix contract.
                unsafe {bind_linear(snapshot,LinearSlot::Bf16($slot),a,precision)?}
            } else {Linear::Bf16(bind!($slot,2))}
        }};}
        let out=Self{c,norm:bind!(s.norm,1),norm_bias:bind!(s.norm_bias,1),fc1:bind_lin!(s.fc1,w,w),bias1:bind!(s.bias1,1),fc2:bind_lin!(s.fc2,c.output,w),bias2:bind!(s.bias2,1)};
        for (name,t,shape) in [("norm",&out.norm,vec![c.hidden]),("norm bias",&out.norm_bias,vec![c.hidden]),
            ("bias1",&out.bias1,vec![w]),("bias2",&out.bias2,vec![c.output])] {
            check(t,&out.norm,name,&shape)?;
            if t.handle.can_mut(){return Err(format!("{name}: immutable native weight required"));}
        }
        for (name,l,shape) in [("fc1",&out.fc1,[w,w]),("fc2",&out.fc2,[c.output,w])] {
            if let Linear::Bf16(t)=l {check(t,&out.norm,name,&shape)?;}
            l.check(name,shape,&out.norm)?;
        }
        Ok(out)
    }
    /// The two linears as bound, in order fc1, fc2.
    pub(crate) fn linears(&self)->[&Linear;2]{[&self.fc1,&self.fc2]}
    /// Input already has frontend merge-major row order. Each declared frame/
    /// group must end on a complete four-row unit; no reordering is performed.
    pub fn forward(&self,x:&CudaTensor,groups:&[usize])->Result<Output,String>{self.run(x,groups,false)}
    /// Deliberately wrong normalization placement, ONLY for a discriminating
    /// oracle control. Affine H-vectors repeat four times over a 4H norm row.
    pub fn diagnostic_postshuffle_norm(&self,x:&CudaTensor,groups:&[usize])->Result<Output,String>{self.run(x,groups,true)}
    fn run(&self,x:&CudaTensor,groups:&[usize],postshuffle:bool)->Result<Output,String>{
        let shape=x.meta.shape().as_slice();let h=self.c.hidden;let w=4*h;
        if shape.len()!=2||!(4..=4096).contains(&shape[0])||shape[0]%4!=0{return Err("merger input requires [N,H], N4..4096 divisible by4".into());}
        let n=shape[0];let cells=check(x,&self.norm,"input",&[n,h])?;
        let mut total=0usize;
        for &g in groups {if g==0||g%4!=0{return Err("every group must contain complete nonempty four-row units".into());}
            total=total.checked_add(g).ok_or("group sum overflow")?;}
        if total!=n{return Err("group coverage differs from input".into());}
        let final_cells=count(&[n/4,self.c.output])?;
        let new=|shape:Vec<usize>,size|CudaTensor::new_contiguous(x.client.clone(),x.device.clone(),shape.into(),x.client.empty(size*2),DType::BF16);
        let norm=new(vec![n,h],cells);
        let grouped=CudaTensor::new_contiguous(x.client.clone(),x.device.clone(),[n/4,w].into(),norm.handle.clone(),DType::BF16);
        let mut fc1=new(vec![n/4,w],cells);let activated=new(vec![n/4,w],cells);let mut hidden=new(vec![n/4,self.c.output],final_cells);
        let cube=CubeDim::new_1d(64);let norm_width=if postshuffle{w}else{h};let rows=cells/norm_width;
        // SAFETY: all source descriptors/strides/devices/bounds checked before
        // launch; fresh disjoint destinations except the read-only grouped view.
        // Each norm row or output cell has one owner and reads only checked spans.
        unsafe {
            norm_kernel::launch_unchecked::<CudaRuntime>(&x.client,cubecl::calculate_cube_count_elemwise(&x.client,rows,cube),cube,
                ArrayArg::from_raw_parts(x.handle.clone(),cells),ArrayArg::from_raw_parts(self.norm.handle.clone(),h),
                ArrayArg::from_raw_parts(self.norm_bias.handle.clone(),h),ArrayArg::from_raw_parts(norm.handle.clone(),cells),rows,norm_width,h);
        }
        match &self.fc1 {
            Linear::Bf16(fc1w)=>unsafe{linear_kernel::launch_unchecked::<CudaRuntime>(&x.client,cubecl::calculate_cube_count_elemwise(&x.client,cells,cube),cube,
                ArrayArg::from_raw_parts(grouped.handle.clone(),cells),ArrayArg::from_raw_parts(fc1w.handle.clone(),w*w),
                ArrayArg::from_raw_parts(self.bias1.handle.clone(),w),ArrayArg::from_raw_parts(fc1.handle.clone(),cells),cells,w,w);},
            Linear::Nvfp4{weight,precision,..}=>fc1=super::vision_block::biased_nvfp4(&grouped,weight,*precision,&self.bias1)?,
        }
        // SAFETY: as above; fc1 is complete [n/4, w] BF16 and activated fresh.
        unsafe {
            gelu_kernel::launch_unchecked::<CudaRuntime>(&x.client,cubecl::calculate_cube_count_elemwise(&x.client,cells,cube),cube,
                ArrayArg::from_raw_parts(fc1.handle.clone(),cells),ArrayArg::from_raw_parts(activated.handle.clone(),cells),cells);
        }
        match &self.fc2 {
            Linear::Bf16(fc2w)=>unsafe{linear_kernel::launch_unchecked::<CudaRuntime>(&x.client,cubecl::calculate_cube_count_elemwise(&x.client,final_cells,cube),cube,
                ArrayArg::from_raw_parts(activated.handle.clone(),cells),ArrayArg::from_raw_parts(fc2w.handle.clone(),self.c.output*w),
                ArrayArg::from_raw_parts(self.bias2.handle.clone(),self.c.output),ArrayArg::from_raw_parts(hidden.handle.clone(),final_cells),final_cells,w,self.c.output);},
            Linear::Nvfp4{weight,precision,..}=>hidden=super::vision_block::biased_nvfp4(&activated,weight,*precision,&self.bias2)?,
        }
        Ok(Output{norm,grouped,fc1,activated,hidden})
    }
}
fn count(shape:&[usize])->Result<usize,String>{shape.iter().try_fold(1usize,|n,&d|if d==0{None}else{n.checked_mul(d)})
    .filter(|&n|n<=u32::MAX as usize).ok_or_else(||"empty/overflowing u32 extent".into())}
fn check(t:&CudaTensor,like:&CudaTensor,name:&str,shape:&[usize])->Result<usize,String>{
    if t.meta.shape().as_slice()!=shape||t.meta.strides().len()!=shape.len()||t.dtype!=DType::BF16||t.qparams.is_some()||t.device!=like.device{return Err(format!("{name}: shape/dtype/device/quantization"));}
    let n=count(shape)?;let mut stride=1;
    for (i,&d) in shape.iter().enumerate().rev(){if d>1&&t.meta.strides()[i]!=stride{return Err(format!("{name}: contiguous required"));}stride*=d;}
    if t.handle.size_in_used()<n as u64*2{return Err(format!("{name}: short storage"));}Ok(n)
}
#[cube(launch_unchecked)]
fn norm_kernel(x:&Array<bf16>,weight:&Array<bf16>,bias:&Array<bf16>,out:&mut Array<bf16>,rows:usize,width:usize,h:usize){
    let row=ABSOLUTE_POS as usize;
    if row<rows {
        let mut sum=0.0f32;for k in 0..width{sum+=f32::cast_from(x[row*width+k]);}
        let mean=sum/f32::cast_from(width);let mut variance=0.0f32;
        for k in 0..width{let d=f32::cast_from(x[row*width+k])-mean;variance+=d*d;}
        let inv=(variance/f32::cast_from(width)+1.0e-6f32).sqrt().recip();
        for k in 0..width{out[row*width+k]=bf16::cast_from((f32::cast_from(x[row*width+k])-mean)*inv*f32::cast_from(weight[k%h])+f32::cast_from(bias[k%h]));}
    }
}
#[cube(launch_unchecked)]
fn linear_kernel(x:&Array<bf16>,w:&Array<bf16>,bias:&Array<bf16>,out:&mut Array<bf16>,n:usize,iw:usize,ow:usize){
    let i=ABSOLUTE_POS as usize;if i<n{let row=i/ow;let col=i%ow;let mut sum=0.0f32;
        for k in 0..iw{sum+=f32::cast_from(x[row*iw+k])*f32::cast_from(w[col*iw+k]);}
        // Explicit candidate boundary: F32 dot + bias, then one BF16 store.
        // Conv3d frontend evidence does NOT establish nn.Linear bias rounding.
        out[i]=bf16::cast_from(sum+f32::cast_from(bias[col]));}
}
#[cube(launch_unchecked)]
fn gelu_kernel(x:&Array<bf16>,out:&mut Array<bf16>,n:usize){
    let i=ABSOLUTE_POS as usize;if i<n{let v=f32::cast_from(x[i]);
        out[i]=bf16::cast_from(0.5f32*v*(1.0f32+f32::erf(v*0.7071067811865475f32)));}
}
