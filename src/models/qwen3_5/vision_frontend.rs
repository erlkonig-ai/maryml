//! Three-weight native BF16 Qwen3.5 visual frontend, not a vision tower.
//! Packed [N,C*T*P*P] -> biased patch projection + learned 2D positions.
//! Projection is materialized as BF16 before bias; biased BF16 precedes positions.
//! All numeric work is GPU-only. Grid arithmetic on the host is metadata only.
//! No raw-image preprocessing, rotary embeddings, attention, merger or pooling.
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use serde::{Deserialize, Serialize};
use triblespace::core::{blob::{Blob, encodings::tensor::{Tensor as NativeTensor, elements::BF16}},
    inline::{Inline, encodings::hash::Handle}, repo::BlobStoreGet};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
pub type CudaTensor = CubeTensor<CudaRuntime>;
pub type Slot<const R: usize> = Inline<Handle<NativeTensor<BF16,R>>>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Config { pub hidden: usize, pub channels: usize, pub temporal: usize,
    pub patch: usize, pub merge: usize, pub position_side: usize }
pub use super::vision_geometry::Grid;
pub struct Slots { pub patch: Slot<5>, pub bias: Slot<1>, pub position: Slot<2> }
pub struct Frontend { config: Config, patch: CudaTensor, bias: CudaTensor, position: CudaTensor }
pub struct Output { pub patch: CudaTensor, pub position: CudaTensor, pub combined: CudaTensor }
impl Config {
    pub fn validate(self) -> Result<(),String> {
        if !(1..=4096).contains(&self.hidden) || !(1..=4).contains(&self.channels)
            || !(1..=4).contains(&self.temporal) || !(1..=32).contains(&self.patch)
            || !(1..=4).contains(&self.merge) || !(1..=256).contains(&self.position_side) {
            return Err("frontend requires H<=4096,C/T<=4,P<=32,merge<=4,position side<=256, all positive".into());
        }
        count(&[self.hidden,self.channels,self.temporal,self.patch,self.patch])?;
        count(&[self.position_side,self.position_side,self.hidden])?;
        Ok(())
    }
    pub fn input_width(self) -> usize { self.channels*self.temporal*self.patch*self.patch }
}
impl Frontend {
    /// # Safety
    /// All selected leaves must come from this genuine validated immutable
    /// append-only pile prefix, including preceding partial pages, through CUDA
    /// runtime teardown. Passed unchanged to bind_pile_leaf. Late descriptor
    /// errors can retain earlier registrations. No generic heap store accepted.
    pub unsafe fn from_pile<R: BlobStoreGet>(snapshot: &R, slots: Slots, c: Config,
        aliases: &mut CudaBf16Aliases) -> Result<Self,String> {
        c.validate()?;
        macro_rules! bind { ($slot:expr,$rank:literal) => {{
            let b: Blob<NativeTensor<BF16,$rank>> = snapshot.get($slot).map_err(|e|e.to_string())?;
            // SAFETY: caller guarantees the full genuine immutable-prefix premise.
            unsafe { aliases.bind_pile_leaf(b)? }
        }}; }
        let out = Self {config:c,patch:bind!(slots.patch,5),bias:bind!(slots.bias,1),position:bind!(slots.position,2)};
        for (name,t,shape) in [
            ("patch",&out.patch,vec![c.hidden,c.channels,c.temporal,c.patch,c.patch]),
            ("bias",&out.bias,vec![c.hidden]),
            ("position",&out.position,vec![c.position_side*c.position_side,c.hidden]),
        ] {
            check(t,&out.patch,name,&shape)?;
            if t.handle.can_mut() { return Err(format!("{name}: immutable alias required")); }
        }
        Ok(out)
    }
    /// All descriptors and complete grid coverage are validated before launch.
    /// Runtime allocation/JIT/asynchronous driver errors keep upstream behavior;
    /// Result does not promise panic recovery or prove arbitrary client handles.
    pub fn forward(&self, input: &CudaTensor, grids: &[Grid]) -> Result<Output,String> {
        let c = self.config;
        let shape = input.meta.shape().as_slice();
        if shape.len()!=2 || !(1..=4096).contains(&shape[0]) { return Err("input requires [N,I], N1..4096".into()); }
        let ni = check(input,&self.patch,"input",&[shape[0],c.input_width()])?;
        let no = count(&[shape[0],c.hidden])?;
        let mut total=0usize;
        for g in grids {
            if g.height%c.merge!=0 || g.width%c.merge!=0 || g.height>4096 || g.width>4096 {
                return Err("positive bounded grid dimensions must divide into spatial merge groups".into());
            }
            total=total.checked_add(count(&[g.frames,g.height,g.width])?).ok_or("grid sum overflow")?;
        }
        if total!=shape[0] { return Err("grid coverage differs from input row count".into()); }
        let tensor=|handle| CudaTensor::new_contiguous(input.client.clone(),input.device.clone(),
            [total,c.hidden].into(),handle,DType::BF16);
        let projection=tensor(input.client.empty(no*2));
        let patch=tensor(input.client.empty(no*2));
        let position=tensor(input.client.empty(no*2));
        let combined=tensor(input.client.empty(no*2));
        let cube=CubeDim::new_1d(64);
        // SAFETY: exact contiguous BF16 descriptors, checked u32 extents, fresh
        // disjoint outputs; each thread owns one cell, every read is in bounds.
        unsafe {
            patch_projection_kernel::launch_unchecked::<CudaRuntime>(&input.client,
                cubecl::calculate_cube_count_elemwise(&input.client,no,cube),cube,
                ArrayArg::from_raw_parts(input.handle.clone(),ni),
                ArrayArg::from_raw_parts(self.patch.handle.clone(),c.hidden*c.input_width()),
                ArrayArg::from_raw_parts(projection.handle.clone(),no),no,c.input_width(),c.hidden);
            // A separate BF16 allocation/dispatch fixes the observed rounding
            // boundary. Same bias operation as the measured r3 diagnostic.
            bias_kernel::launch_unchecked::<CudaRuntime>(&input.client,
                cubecl::calculate_cube_count_elemwise(&input.client,no,cube),cube,
                ArrayArg::from_raw_parts(projection.handle.clone(),no),
                ArrayArg::from_raw_parts(self.bias.handle.clone(),c.hidden),
                ArrayArg::from_raw_parts(patch.handle.clone(),no),no,c.hidden);
            let mut offset=0usize;
            for g in grids {
                let cells=g.frames*g.height*g.width*c.hidden;
                position_kernel::launch_unchecked::<CudaRuntime>(&input.client,
                    cubecl::calculate_cube_count_elemwise(&input.client,cells,cube),cube,
                    ArrayArg::from_raw_parts(self.position.handle.clone(),c.position_side*c.position_side*c.hidden),
                    ArrayArg::from_raw_parts(patch.handle.clone(),no),
                    ArrayArg::from_raw_parts(position.handle.clone(),no),
                    ArrayArg::from_raw_parts(combined.handle.clone(),no),
                    cells,offset,c.hidden,g.height,g.width,c.merge,c.position_side);
                offset+=cells;
            }
        }
        Ok(Output {patch,position,combined})
    }
    /// DIAGNOSTIC ONLY: add this frontend's bias to an already materialized
    /// BF16 no-bias projection, using the same bias kernel as forward.
    /// Caller supplies the same-input output of a zero-bias frontend; this
    /// hypothesis is not an assertion about the upstream Conv3d implementation.
    pub fn diagnostic_bias_after_bf16(&self, no_bias: &Output) -> Result<Output,String> {
        let shape=no_bias.patch.meta.shape().as_slice();
        if shape.len()!=2 || !(1..=4096).contains(&shape[0]) {return Err("diagnostic requires [N,H]".into());}
        let n=check(&no_bias.patch,&self.patch,"projection",&[shape[0],self.config.hidden])?;
        check(&no_bias.position,&self.patch,"position",&[shape[0],self.config.hidden])?;
        let tensor=|| CudaTensor::new_contiguous(no_bias.patch.client.clone(),no_bias.patch.device.clone(),
            [shape[0],self.config.hidden].into(),no_bias.patch.client.empty(n*2),DType::BF16);
        let patch=tensor();let combined=tensor();let cube=CubeDim::new_1d(64);
        // SAFETY: checked contiguous BF16 inputs, fresh disjoint outputs, one
        // owner per cell. Projection is stored BF16 before this separate kernel.
        unsafe {
            bias_kernel::launch_unchecked::<CudaRuntime>(&no_bias.patch.client,
                cubecl::calculate_cube_count_elemwise(&no_bias.patch.client,n,cube),cube,
                ArrayArg::from_raw_parts(no_bias.patch.handle.clone(),n),
                ArrayArg::from_raw_parts(self.bias.handle.clone(),self.config.hidden),
                ArrayArg::from_raw_parts(patch.handle.clone(),n),n,self.config.hidden);
            add_position_kernel::launch_unchecked::<CudaRuntime>(&no_bias.patch.client,
                cubecl::calculate_cube_count_elemwise(&no_bias.patch.client,n,cube),cube,
                ArrayArg::from_raw_parts(patch.handle.clone(),n),
                ArrayArg::from_raw_parts(no_bias.position.handle.clone(),n),
                ArrayArg::from_raw_parts(combined.handle.clone(),n),n);
        }
        Ok(Output {patch,position:no_bias.position.clone(),combined})
    }
}
fn count(shape: &[usize]) -> Result<usize,String> {
    shape.iter().try_fold(1usize,|n,&d| if d==0 {None} else {n.checked_mul(d)})
        .filter(|&n| n<=u32::MAX as usize).ok_or_else(||"empty/overflowing u32 extent".into())
}
fn check(t: &CudaTensor, like: &CudaTensor, name:&str, shape:&[usize]) -> Result<usize,String> {
    if t.meta.shape().as_slice()!=shape || t.meta.strides().len()!=shape.len()
        || t.dtype!=DType::BF16 || t.qparams.is_some() || t.device!=like.device {
        return Err(format!("{name}: wrong shape/dtype/device or quantized"));
    }
    let n=count(shape)?;
    let mut stride=1;
    for (i,&d) in shape.iter().enumerate().rev() {
        if d>1 && t.meta.strides()[i]!=stride {return Err(format!("{name}: contiguous storage required"));}
        stride*=d;
    }
    if t.handle.size_in_used()<(n as u64)*2 {return Err(format!("{name}: short storage"));}
    Ok(n)
}
#[cube(launch_unchecked)]
fn patch_projection_kernel(x:&Array<bf16>,w:&Array<bf16>,out:&mut Array<bf16>,n:usize,iw:usize,ow:usize) {
    let i=ABSOLUTE_POS as usize;
    if i<n {
        let row=i/ow;
        let col=i%ow;
        let mut sum=0.0f32;
        for k in 0..iw {sum+=f32::cast_from(x[row*iw+k])*f32::cast_from(w[col*iw+k]);}
        out[i]=bf16::cast_from(sum);
    }
}
#[cube]
fn round_bf16(x:f32)->f32 {f32::cast_from(bf16::cast_from(x))}
#[cube(launch_unchecked)]
fn bias_kernel(projection:&Array<bf16>,bias:&Array<bf16>,patch:&mut Array<bf16>,n:usize,hidden:usize) {
    let i=ABSOLUTE_POS as usize;
    if i<n {
        let biased=bf16::cast_from(f32::cast_from(projection[i])+f32::cast_from(bias[i%hidden]));
        patch[i]=biased;
    }
}
#[cube(launch_unchecked)]
fn add_position_kernel(patch:&Array<bf16>,position:&Array<bf16>,combined:&mut Array<bf16>,n:usize) {
    let i=ABSOLUTE_POS as usize;
    if i<n {combined[i]=bf16::cast_from(f32::cast_from(patch[i])+f32::cast_from(position[i]));}
}
#[cube]
fn coordinate(i:usize,n:usize,side:usize)->f32 {
    if n==1 {f32::cast_from(0.0f32)} else {
        let end=f32::cast_from(side-1);
        let step=end/f32::cast_from(n-1);
        // Endpoint-symmetric F32 linspace, independent of other frames/grids.
        if i<n/2 {step*f32::cast_from(i)} else {end-step*f32::cast_from(n-i-1)}
    }
}
#[cube(launch_unchecked)]
fn position_kernel(table:&Array<bf16>,patch:&Array<bf16>,pos:&mut Array<bf16>,out:&mut Array<bf16>,
    n:usize,offset:usize,hidden:usize,height:usize,width:usize,merge:usize,side:usize) {
    let i=ABSOLUTE_POS as usize;
    if i<n {
        let token=(i/hidden)%(height*width);
        let unit=token/(merge*merge);
        let row=(unit/(width/merge))*merge+(token/merge)%merge;
        let col=(unit%(width/merge))*merge+token%merge;
        let hp=coordinate(row,height,side);
        let wp=coordinate(col,width,side);
        let hf=usize::cast_from(hp);
        let wf=usize::cast_from(wp);
        let hc=if hf+1<side {hf+1} else {side-1};
        let wc=if wf+1<side {wf+1} else {side-1};
        let dh=hp-f32::cast_from(hf);
        let dw=wp-f32::cast_from(wf);
        let feature=i%hidden;
        let p0=round_bf16(f32::cast_from(table[(hf*side+wf)*hidden+feature])*round_bf16((1.0f32-dh)*(1.0f32-dw)));
        let p1=round_bf16(f32::cast_from(table[(hf*side+wc)*hidden+feature])*round_bf16((1.0f32-dh)*dw));
        let p2=round_bf16(f32::cast_from(table[(hc*side+wf)*hidden+feature])*round_bf16(dh*(1.0f32-dw)));
        let p3=round_bf16(f32::cast_from(table[(hc*side+wc)*hidden+feature])*round_bf16(dh*dw));
        let value=round_bf16(round_bf16(round_bf16(p0+p1)+p2)+p3);
        pos[offset+i]=bf16::cast_from(value);
        out[offset+i]=bf16::cast_from(f32::cast_from(patch[offset+i])+value);
    }
}
