//! Shared fixed-order CUDA decoder normalization/residual/gated-MLP operators.
//! Extracted verbatim from frozen full_attention.rs ff186fa2… numerical bodies.
//! Private callers MUST validate contiguous extents, dtype/device, weights and
//! equal element counts before dispatch. Fresh outputs; no host model math.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use super::Rt;
use half::bf16;
type CudaTensor = CubeTensor<Rt>;
fn tensor(like:&CudaTensor,s:&[usize],h:cubecl::server::Handle)->CudaTensor { CubeTensor::new_contiguous(like.client.clone(),like.device.clone(),s.into(),h,DType::BF16) }
fn len(t:&CudaTensor)->usize { t.meta.shape().as_slice().iter().product() }
fn grid(t:&CudaTensor,n:usize)->CubeCount { cubecl::calculate_cube_count_elemwise(&t.client,n,CubeDim::new_1d(64)) }

/// Widest row [`norm`] takes: the hidden size.
const NORM_CAPACITY: usize = 4096;
const NORM_THREADS: u32 = 128;

/// One cube per row. The cube stages the row in shared memory with coalesced
/// loads, one unit sums its squares in the original serial order, and the cube
/// writes the scaled row. Each output is the same expression of the same
/// values as the one-unit-per-row kernel this replaces, so the bits are the
/// same; only the loads and stores moved off a single unit striding an 8 KiB
/// row (361 against ~15 us per [248,4096] norm on a GB10).
#[cube(launch_unchecked)]
fn norm_kernel(x:&Array<bf16>,w:&Array<bf16>,y:&mut Array<bf16>,width:usize,eps:f32) {
    let r=CUBE_POS_X as usize; let lane=UNIT_POS_X as usize; let threads=CUBE_DIM_X as usize;
    let mut row=SharedMemory::<f32>::new(NORM_CAPACITY); let mut scale=SharedMemory::<f32>::new(1usize);
    let mut j=lane; while j<width { row[j]=f32::cast_from(x[r*width+j]); j+=threads; }
    sync_cube();
    if lane==0 { let mut sum=0.0f32; for k in 0..width { let v=row[k]; sum+=v*v; }
        scale[0]=1.0f32/(sum/f32::cast_from(width)+eps).sqrt(); }
    sync_cube();
    let inv=scale[0];
    let mut j=lane; while j<width { y[r*width+j]=bf16::cast_from(row[j]*inv*(1.0f32+f32::cast_from(w[j]))); j+=threads; }
}
pub(super) fn norm(x:&CudaTensor,w:&CudaTensor,width:usize,eps:f32)->CudaTensor {
    assert!(width<=NORM_CAPACITY, "norm row wider than {NORM_CAPACITY}");
    let n=len(x); let out=x.client.empty(n*2); let rows=(n/width) as u32;
    // SAFETY: private callers prevalidate all contiguous BF16 extents; output fresh.
    unsafe { norm_kernel::launch_unchecked::<Rt>(&x.client,CubeCount::Static(rows,1,1),CubeDim::new_1d(NORM_THREADS),
        ArrayArg::from_raw_parts(x.handle.clone(),n),ArrayArg::from_raw_parts(w.handle.clone(),width),
        ArrayArg::from_raw_parts(out.clone(),n),width,eps); }
    tensor(x,x.meta.shape().as_slice(),out)
}
#[cube(launch_unchecked)]
fn element_kernel(a:&Array<bf16>,b:&Array<bf16>,out:&mut Array<bf16>,n:usize,#[comptime] mode:u32) {
    let i=ABSOLUTE_POS as usize; if i<n { let x=f32::cast_from(a[i]); let y=f32::cast_from(b[i]);
        if mode==0 { out[i]=bf16::cast_from(x+y); }
        else if mode==1 { let activated=f32::cast_from(bf16::cast_from(x/(1.0f32+(-x).exp()))); out[i]=bf16::cast_from(activated*y); }
        else { let gate=f32::cast_from(bf16::cast_from(1.0f32/(1.0f32+(-y).exp()))); out[i]=bf16::cast_from(x*gate); }
    }
}
pub(super) fn elementwise(a:&CudaTensor,b:&CudaTensor,mode:u32)->CudaTensor {
    let n=len(a); let out=a.client.empty(n*2);
    unsafe { element_kernel::launch_unchecked::<Rt>(&a.client,grid(a,n),CubeDim::new_1d(64),
        ArrayArg::from_raw_parts(a.handle.clone(),n),ArrayArg::from_raw_parts(b.handle.clone(),n),
        ArrayArg::from_raw_parts(out.clone(),n),n,mode); }
    tensor(a,a.meta.shape().as_slice(),out)
}
