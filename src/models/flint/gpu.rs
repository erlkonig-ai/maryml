//! Bounded shared Flint block CUDA arithmetic through Mary's existing CubeCL runtime.
//!
//! Weights are decoded from stored GGUF NVFP4 (36 bytes / 64 values), then
//! uploaded as F32. Explicit packed-K/all modes decode only attn_k or all
//! type-40 projection families inside the same matvec, directly from unchanged
//! GGUF bytes. A typed pile source supplies native Tensor planes explicitly;
//! their positive finite block scales/global1 are admitted before upload.
//! Norms/sinks and the BF16 router keep their existing paths.
//! Separate `.scale` values stay
//! after the matmul in both modes. This
//! does not use tensor cores, activation quantization,
//! BF16, or llama.cpp's Q8_1/W4A4 expert-down dispatch.
//!
//! All layer arithmetic, including top-six selection, runs on the GPU. This is
//! a bounded graph: optional named stages and required expert IDs return to the
//! host. Cached and diagnostic blocks share the same arithmetic. Explicit fences
//! retire temporary uploads even with tracing off; no full expert bank is retained.
//!
//! Numerical tier, fixed before comparison: F32 input/weight/output and F32
//! accumulation. Matvec uses 32 strided lanes per row and a five-step tree;
//! RMS uses 256 strided lanes and an eight-step tree. Attention and router
//! reductions are serial F32; GPU transcendental functions and CUDA compiler
//! multiply-add contraction may differ from CPU scalar f64 reductions. No
//! autotuning, fitted tolerance, or claim of GGML reduction-order parity.

use anyhow::{Context, Result, anyhow, ensure};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::math::{H, Stage, Trace, Values};
use super::weights::{Weights, PackedWeight};
use super::config::{BlockConfig, FfnKind};

mod packed;
mod fixed;
pub(super) use fixed::{FixedBlock,PreparedFixed,ResidentFixed};
use fixed::{FixedProjection,FixedFfn,FixedBranch};
pub use packed::{KComparison, compare_k as compare_packed_k};
#[cfg(test)]
mod projection_tests;

const D: usize = 128;
const Q: usize = 32;
const KV: usize = 4;
const E: usize = 256;
const K: usize = 6;
const F: usize = 512;
const SH: usize = 1024;
type Client = cubecl::client::ComputeClient<CudaRuntime>;

#[derive(Clone, Copy)]
pub enum ProjectionMode { Decoded, PackedK, PackedAll }
impl ProjectionMode {
    fn packed(self, suffix: &str) -> bool {
        match self { Self::Decoded => false, Self::PackedK => suffix == "attn_k", Self::PackedAll => true }
    }
}

enum ProjectionStorage { Decoded(Buffer), Packed(packed::PackedBuffer) }
struct Projection { ni: usize, no: usize, storage: ProjectionStorage }

#[cube(launch_unchecked)]
fn matvec_kernel(
    x: &Array<f32>, w: &Array<f32>, y: &mut Array<f32>,
    ni: u32, no: u32, batch: u32, input_offset: u32, output_offset: u32,
) {
    let lane = UNIT_POS_X % 32;
    let row = CUBE_POS_X * 8 + UNIT_POS_X / 32;
    let mut acc = f32::new(0.0);
    if row < no * batch {
        let token = row / no;
        let wr = row % no;
        let mut i = lane;
        while i < ni {
            acc += w[(wr * ni + i) as usize] * x[(input_offset + token * ni + i) as usize];
            i += 32;
        }
    }
    let mut red = SharedMemory::<f32>::new(256usize);
    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut stride = u32::new(16);
    while stride > 0 {
        if lane < stride {
            red[UNIT_POS_X as usize] = red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && row < no * batch {
        y[(output_offset + row) as usize] = red[UNIT_POS_X as usize];
    }
}

#[cube(launch_unchecked)]
fn norm_kernel(x: &Array<f32>, weight: &Array<f32>, y: &mut Array<f32>, width: u32) {
    let base = CUBE_POS_X * width;
    let mut acc = f32::new(0.0);
    let mut i = UNIT_POS_X;
    while i < width {
        let v = x[(base + i) as usize];
        acc += v * v;
        i += 256;
    }
    let mut red = SharedMemory::<f32>::new(256usize);
    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut stride = u32::new(128);
    while stride > 0 {
        if UNIT_POS_X < stride {
            red[UNIT_POS_X as usize] = red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let inv = 1.0 / (red[0] / width as f32 + 1e-6).sqrt();
    let mut i = UNIT_POS_X;
    while i < width {
        y[(base + i) as usize] = (x[(base + i) as usize] * inv) * weight[i as usize];
        i += 256;
    }
}

// 0:add, 1:row-scale, 2:sigmoid, 3:SwiGLU, 4:row-divide.
// A row-scale with width == len is a scalar scale, without folding it into W.
#[cube(launch_unchecked)]
fn point_kernel(a: &Array<f32>, b: &Array<f32>, y: &mut Array<f32>,
    len: u32, width: u32, #[comptime] op: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        let x = a[i as usize];
        if comptime![op == 0] { y[i as usize] = x + b[i as usize]; }
        if comptime![op == 1] { y[i as usize] = x * b[(i / width) as usize]; }
        if comptime![op == 2] { y[i as usize] = 1.0 / (1.0 + (-x).exp()); }
        if comptime![op == 3] { y[i as usize] = (x * (1.0 / (1.0 + (-x).exp()))) * b[i as usize]; }
        if comptime![op == 4] { y[i as usize] = x / b[(i / width) as usize]; }
    }
}

#[cube(launch_unchecked)]
fn rope_kernel(x: &Array<f32>, positions: &Array<i32>, y: &mut Array<f32>, heads: u32, pairs: u32) {
    let pair = ABSOLUTE_POS as u32;
    if pair < pairs {
        let j = pair % 64;
        let head_token = pair / 64;
        let token = head_token / heads;
        let angle = positions[token as usize] as f32 * f32::new(10000.0).powf(-2.0 * j as f32 / 128.0);
        let sin = angle.sin();
        let cos = angle.cos();
        let i = head_token * 128 + j;
        y[i as usize] = x[i as usize] * cos - x[(i + 64) as usize] * sin;
        y[(i + 64) as usize] = x[i as usize] * sin + x[(i + 64) as usize] * cos;
    }
}

#[cube(launch_unchecked)]
fn mask_kernel(positions: &Array<i32>, keys: &Array<i32>, mask: &mut Array<f32>,
    t: u32, nt:u32, window: i32, negative_infinity: f32) {
    let i = ABSOLUTE_POS as u32;
    if i < t * nt {
        let query = positions[(i / nt) as usize];
        let key = keys[(i % nt) as usize];
        let mut m = f32::new(0.0);
        // A scalar binding avoids CubeCL emitting the invalid CUDA literal -inf.
        if key > query || (window > 0 && query - key >= window) { m = negative_infinity; }
        mask[i as usize] = m;
    }
}

#[cube(launch_unchecked)]
fn attention_kernel(q: &Array<f32>, k: &Array<f32>, v: &Array<f32>, sinks: &Array<f32>,
    mask: &Array<f32>, scores: &mut Array<f32>, probs: &mut Array<f32>,
    heads: &mut Array<f32>, t: u32, nt:u32) {
    let hq = ABSOLUTE_POS as u32;
    if hq < 32 * t {
        let h = hq / t;
        let query = hq % t;
        let base = hq * nt;
        // Each thread owns one complete [head,query] row of global scratch.
        // No fixed local key array: query<=3 does not limit cached key length.
        let mut maximum = sinks[h as usize];
        for key in 0..nt {
            let mut dot = f32::new(0.0);
            for d in 0..128u32 {
                dot += q[((query * 32 + h) * 128 + d) as usize]
                    * k[((key * 4 + h / 8) * 128 + d) as usize];
            }
            scores[(base + key) as usize] = dot;
            let logit = dot * (1.0 / 128.0f32.sqrt()) + mask[(query * nt + key) as usize];
            probs[(base + key) as usize] = logit;
            maximum = maximum.max(logit);
        }
        let mut denom = f32::new(0.0);
        for key in 0..nt {
            let e = (probs[(base + key) as usize] - maximum).exp();
            probs[(base + key) as usize] = e;
            denom += e;
        }
        denom += (sinks[h as usize] - maximum).exp();
        for key in 0..nt {
            probs[(base + key) as usize] = probs[(base + key) as usize] / denom;
        }
        for d in 0..128u32 {
            let mut sum = f32::new(0.0);
            for key in 0..nt {
                sum += probs[(base + key) as usize] * v[((key * 4 + h / 8) * 128 + d) as usize];
            }
            heads[(hq * 128 + d) as usize] = sum;
        }
    }
}

#[cube(launch_unchecked)]
fn append_kernel(k:&Array<f32>,v:&Array<f32>,keys:&mut Array<f32>,values:&mut Array<f32>,len:u32,offset:u32) {
    let i=ABSOLUTE_POS as u32;
    if i<len { keys[(offset+i) as usize]=k[i as usize]; values[(offset+i) as usize]=v[i as usize]; }
}

#[cube(launch_unchecked)]
fn join_kernel(heads: &Array<f32>, joined: &mut Array<f32>, t: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < t * 32 * 128 {
        let d = i % 128;
        let h = (i / 128) % 32;
        let token = i / (32 * 128);
        joined[i as usize] = heads[((h * t + token) * 128 + d) as usize];
    }
}

#[cube(launch_unchecked)]
fn route_kernel(prob: &Array<f32>, ids: &mut Array<i32>, selected: &mut Array<f32>,
    sums: &mut Array<f32>, denoms: &mut Array<f32>, weights: &mut Array<f32>,
    margins: &mut Array<f32>, t: u32) {
    let token = ABSOLUTE_POS as u32;
    if token < t {
        let mut chosen = Array::<u32>::new(7usize);
        let mut vals = Array::<f32>::new(7usize);
        let mut sum = f32::new(0.0);
        for slot in 0..7u32 {
            let mut best = f32::new(-1.0);
            let mut winner = u32::new(0);
            for e in 0..256u32 {
                let mut used = false;
                for prev in 0..slot { if chosen[prev as usize] == e { used = true; } }
                let p = prob[(token * 256 + e) as usize];
                // Ascending scan plus strict greater-than gives smaller-ID ties.
                if !used && p > best { best = p; winner = e; }
            }
            chosen[slot as usize] = winner;
            vals[slot as usize] = best;
            if slot < 6 {
                ids[(token * 6 + slot) as usize] = winner as i32;
                selected[(token * 6 + slot) as usize] = best;
                sum += best;
            }
        }
        let denom = sum.max(6.103515625e-5);
        sums[token as usize] = sum;
        denoms[token as usize] = denom;
        margins[token as usize] = vals[5] - vals[6];
        for slot in 0..6u32 {
            weights[(token * 6 + slot) as usize] = vals[slot as usize] / denom;
        }
    }
}

#[cube(launch_unchecked)]
fn down_scale_kernel(x: &Array<f32>, scales: &mut Array<f32>, width: u32, rows: u32) {
    let row = ABSOLUTE_POS as u32;
    if row < rows {
        let mut amax = f32::new(0.0);
        for i in 0..width { amax = amax.max(x[(row * width + i) as usize].abs()); }
        let power = ((amax / 1024.0).ln() * std::f32::consts::LOG2_E).ceil().clamp(0.0, 127.0);
        scales[row as usize] = (power * std::f32::consts::LN_2).exp();
    }
}

#[cube(launch_unchecked)]
fn moe_sum_kernel(weighted: &Array<f32>, out: &mut Array<f32>, len: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        let token = i / 2048;
        let d = i % 2048;
        let mut sum = weighted[(token * 6 * 2048 + d) as usize];
        for slot in 1..6u32 { sum += weighted[((token * 6 + slot) * 2048 + d) as usize]; }
        out[i as usize] = sum;
    }
}

// All buffers are four-byte scalars; integer buffers are used only by kernels
// whose signatures say i32. Shapes are checked on the host before raw launches.
struct Buffer { handle: Handle, len: usize }
impl Buffer {
    fn arg(&self) -> ArrayArg<CudaRuntime> {
        // Allocation and logical length are created together and never resized.
        unsafe { ArrayArg::from_raw_parts(self.handle.clone(), self.len) }
    }
}
struct Device { client: Client }
impl Device {
    fn new() -> Self { Self { client: CudaRuntime::client(&Default::default()) } }
    fn empty(&self, len: usize) -> Buffer { Buffer { handle: self.client.empty(len * 4), len } }
    fn upload(&self, x: &[f32]) -> Buffer {
        Buffer { handle: self.client.create_from_slice(f32::as_bytes(x)), len: x.len() }
    }
    // Only explicitly identified immutable parameters use this counter. Never
    // count activation/KV/position uploads as fixed weight traffic.
    fn weight_upload(&self,w:&mut impl Weights,x:&[f32],routed:bool)->Buffer {
        let start=w.native_setup_enabled().then(std::time::Instant::now);
        let buffer=self.upload(x);
        if let Some(start)=start {
            let mut upload=super::weights::Work::default();upload.record(start,x.len()*4,x.len()*4);
            w.record_native_upload(routed,Default::default(),upload);
        }
        buffer
    }
    fn norm_parameter(&self,w:&mut impl Weights,x:&Buffer,suffix:&str,dims:&[usize],resident:Option<&Buffer>)->Result<Buffer> {
        if let Some(gain)=resident {return self.norm_borrowed(x,gain);}
        let values=w.get(suffix,dims,0,None)?;
        assert!(!values.is_empty());assert_eq!(x.len%values.len(),0);
        let gain=self.weight_upload(w,&values,false);
        self.norm_borrowed(x,&gain)
    }
    fn positions(&self, x: &[i32]) -> Buffer {
        Buffer { handle: self.client.create_from_slice(i32::as_bytes(x)), len: x.len() }
    }
    fn sync(&self)->Result<()> {
        cubecl::future::block_on(self.client.sync()).map_err(|e|anyhow!("CUDA synchronization: {e:?}"))
    }
    fn read(&self, x: &Buffer) -> Result<Vec<f32>> {
        let bytes = self.client.read_one(x.handle.clone()).map_err(|e| anyhow!("CUDA readback: {e:?}"))?;
        ensure!(bytes.len() == x.len * 4, "CUDA readback length differs");
        Ok(f32::from_bytes(&bytes).to_vec())
    }
    fn read_ids(&self, x: &Buffer) -> Result<Vec<i32>> {
        let bytes = self.client.read_one(x.handle.clone()).map_err(|e| anyhow!("CUDA ID readback: {e:?}"))?;
        ensure!(bytes.len() == x.len * 4, "CUDA ID readback length differs");
        Ok(i32::from_bytes(&bytes).to_vec())
    }
    fn trace(&self, tr: &mut Trace, name: &str, dims: [usize; 4], x: &Buffer) -> Result<()> {
        ensure!(dims.iter().product::<usize>() == x.len, "trace shape mismatch: {name}");
        if !tr.enabled { return Ok(()); }
        tr.stages.push(Stage { name: name.into(), dims, values: Values::F32(self.read(x)?) });
        Ok(())
    }
    fn matvec(&self, x: &Buffer, w: &Buffer, y: &Buffer, ni: usize, no: usize,
        batch: usize, input_offset: usize, output_offset: usize) -> Result<()> {
        assert_eq!(w.len, ni * no);
        assert!(input_offset + ni * batch <= x.len && output_offset + no * batch <= y.len);
        unsafe { matvec_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d((no * batch).div_ceil(8) as u32), CubeDim::new_1d(256),
            x.arg(), w.arg(), y.arg(), ni as u32, no as u32, batch as u32,
            input_offset as u32, output_offset as u32,
        ); }
        self.sync()
    }
    fn norm(&self, x: &Buffer, weights: &[f32]) -> Result<Buffer> {
        assert!(!weights.is_empty());
        assert_eq!(x.len%weights.len(),0);
        let w = self.upload(weights);
        self.norm_borrowed(x,&w)
    }
    fn norm_borrowed(&self,x:&Buffer,w:&Buffer)->Result<Buffer> {
        assert!(w.len>0);
        assert_eq!(x.len%w.len,0);
        let y = self.empty(x.len);
        unsafe { norm_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d((x.len / w.len) as u32), CubeDim::new_1d(256),
            x.arg(), w.arg(), y.arg(), w.len as u32,
        ); }
        self.sync()?; Ok(y)
    }
    fn point(&self, a: &Buffer, b: &Buffer, width: usize, op: u32) -> Result<Buffer> {
        assert!(width > 0);
        match op { 0 | 3 => assert_eq!(a.len, b.len), 1 | 4 => {
            assert_eq!(a.len % width, 0); assert_eq!(b.len, a.len / width);
        }, 2 => {}, _ => panic!("invalid point operation") }
        let y = self.empty(a.len);
        unsafe { point_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d(a.len.div_ceil(256) as u32), CubeDim::new_1d(256),
            a.arg(), b.arg(), y.arg(), a.len as u32, width as u32, op,
        ); }
        self.sync()?; Ok(y)
    }
    fn rope(&self, x: &Buffer, positions: &Buffer, heads: usize) -> Result<Buffer> {
        assert_eq!(x.len, positions.len * heads * D);
        let y = self.empty(x.len);
        unsafe { rope_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d((x.len / 2).div_ceil(256) as u32), CubeDim::new_1d(256),
            x.arg(), positions.arg(), y.arg(), heads as u32, (x.len / 2) as u32,
        ); }
        self.sync()?; Ok(y)
    }
    fn projection(&self, w: &mut impl Weights, suffix: &str, dims: &[usize],
        expert: Option<usize>, mode: ProjectionMode) -> Result<(Projection,Option<f32>)> {
        ensure!(matches!(dims.len(), 2 | 3), "projection requires matrix/expert dimensions");
        let name = format!("{suffix}.weight");
        let mut post_scale=None;
        let storage = if mode.packed(suffix) {
            let (weight,scale)=w.packed_projection(suffix,dims,expert)?;
            post_scale=scale;
            let packed = match weight {
                PackedWeight::Gguf(bytes) => self.upload_nvfp4(&bytes, dims[0], dims[1])?,
                PackedWeight::Native(view) => {
                    if w.native_setup_enabled() {
                        let mut geometry=super::weights::Work::default();let mut upload=super::weights::Work::default();
                        let result=self.upload_native_nvfp4_observed(&view,dims[0],dims[1],Some((&mut geometry,&mut upload)));
                        w.record_native_upload(expert.is_some(),geometry,upload);
                        result?
                    } else {self.upload_native_nvfp4(&view,dims[0],dims[1])?}
                },
            };
            ProjectionStorage::Packed(packed)
        } else {
            ProjectionStorage::Decoded(self.upload(&w.get(&name, dims, 40, expert)?))
        };
        Ok((Projection { ni: dims[0], no: dims[1], storage },post_scale))
    }
    fn project(&self, weight: &Projection, x: &Buffer, y: &Buffer, t: usize,
        input_offset: usize, output_offset: usize) -> Result<()> {
        match &weight.storage {
            ProjectionStorage::Decoded(w) => self.matvec(x, w, y, weight.ni, weight.no, t, input_offset, output_offset)?,
            ProjectionStorage::Packed(w) => self.packed_matvec(x, w, y, t, input_offset, output_offset)?,
        }
        self.sync()
    }
    fn dense(&self, w: &mut impl Weights, tr: &mut Trace, suffix: &str, label: &str,
        x: &Buffer, ni: usize, no: usize, t: usize, mode: ProjectionMode) -> Result<Buffer> {
        let (weight,paired) = self.projection(w, suffix, &[ni,no], None, mode)?;
        let scale_value=match paired {Some(v)=>v,None=>w.get(&format!("{suffix}.scale"), &[1], 0, None)?[0]};
        let scale = self.weight_upload(w,&[scale_value],false);
        self.dense_borrowed(tr,label,x,&weight,&scale,t)
    }
    fn dense_borrowed(&self,tr:&mut Trace,label:&str,x:&Buffer,weight:&Projection,scale:&Buffer,t:usize)->Result<Buffer> {
        let no=weight.no;
        let raw = self.empty(no * t);
        self.project(weight, x, &raw, t, 0, 0)?;
        self.trace(tr, &format!("{label}_unscaled"), [no,t,1,1], &raw)?;
        let out = self.point(&raw, scale, raw.len, 1)?;
        self.trace(tr, label, [no,t,1,1], &out)?;
        Ok(out)
    }
    fn dense_parameter(&self,w:&mut impl Weights,tr:&mut Trace,suffix:&str,label:&str,
        x:&Buffer,ni:usize,no:usize,t:usize,mode:ProjectionMode,resident:Option<&FixedProjection>)->Result<Buffer> {
        match resident {
            Some(p)=>{
                ensure!(p.weight.ni==ni && p.weight.no==no,"resident fixed projection dimensions differ");
                self.dense_borrowed(tr,label,x,&p.weight,&p.scale,t)
            },
            None=>self.dense(w,tr,suffix,label,x,ni,no,t,mode),
        }
    }
    fn experts(&self, w: &mut impl Weights, tr: &mut Trace, suffix: &str, label: &str,
        x: &Buffer, ni: usize, no: usize, t: usize, ids: &[i32], per_slot: bool, mode: ProjectionMode) -> Result<Buffer> {
        ensure!(ids.len() == K * t && ids.iter().all(|&i| (0..E as i32).contains(&i)), "invalid GPU expert IDs");
        let scales = if w.paired_scales() {Vec::new()}
            else {w.get(&format!("{suffix}.scale"), &[E], 0, None)?};
        let raw = self.empty(no * K * t);
        let mut selected = Vec::with_capacity(ids.len());
        // One packed or decoded expert matrix at a time. Synchronizing the
        // completed slot retires its upload before loading the next one.
        for (slot, &id) in ids.iter().enumerate() {
            let (weight,paired) = self.projection(w, suffix, &[ni,no,E], Some(id as usize), mode)?;
            let row = if per_slot { slot } else { slot / K };
            self.project(&weight, x, &raw, 1, row * ni, slot * no)?;
            cubecl::future::block_on(self.client.sync())
                .map_err(|e| anyhow!("CUDA expert synchronization: {e:?}"))?;
            selected.push(match paired {Some(v)=>v,None=>*scales.get(id as usize)
                .context("paired native expert scale missing")?});
        }
        let scale = self.weight_upload(w,&selected,true);
        let out = self.point(&raw, &scale, no, 1)?;
        self.trace(tr, &format!("{label}_unscaled"), [no,K,t,1], &raw)?;
        self.trace(tr, &format!("{label}_selected_scale"), [1,K,t,1], &scale)?;
        self.trace(tr, label, [no,K,t,1], &out)?;
        Ok(out)
    }
    fn dense_ffn(&self, w: &mut impl Weights, tr: &mut Trace, n: &Buffer,
        hidden: usize, width: usize, t: usize, mode: ProjectionMode,resident:Option<&FixedFfn>) -> Result<Buffer> {
        let up = self.dense_parameter(w,tr,"ffn_up","dense_up",n,hidden,width,t,mode,resident.map(|f|&f.up))?;
        let gate = self.dense_parameter(w,tr,"ffn_gate","dense_gate",n,hidden,width,t,mode,resident.map(|f|&f.gate))?;
        // Clamp=0 follows the exact pinned common loader plus Flint loader.
        let sw = self.point(&gate,&up,width,3)?;
        self.trace(tr,"dense_swiglu",[width,t,1,1],&sw)?;
        self.trace(tr,"dense_down_input",[width,t,1,1],&sw)?;
        self.dense_parameter(w,tr,"ffn_down","dense_down",&sw,width,hidden,t,mode,resident.map(|f|&f.down))
    }
}

/// Reserved token-major [capacity,4,128] buffers on the fixed default device.
/// The owner must poison the whole sequence on a failed partially executed block.
pub(super) struct KvCache {
    dev:Device, k:Buffer, v:Buffer, positions:Vec<i32>, capacity:usize,
}
impl KvCache {
    pub fn new(capacity:usize)->Result<Self> {
        super::cache::Geometry::new(3,capacity)?;
        let elements=capacity.checked_mul(KV*D).ok_or_else(||anyhow!("KV allocation overflow"))?;
        elements.checked_mul(4).ok_or_else(||anyhow!("KV byte count overflow"))?;
        let mut positions=Vec::new(); positions.try_reserve_exact(capacity)?;
        let dev=Device::new(); let k=dev.empty(elements); let v=dev.empty(elements);
        dev.sync()?; Ok(Self {dev,k,v,positions,capacity})
    }
    pub fn len(&self)->usize { self.positions.len() }
    pub fn reset(&mut self)->Result<()> { self.dev.sync()?; self.positions.clear(); Ok(()) }
    fn append(&mut self,k:&Buffer,v:&Buffer,positions:&[i32])->Result<()> {
        super::cache::validate_append(&self.positions,positions,self.capacity)?;
        ensure!(k.len==positions.len()*KV*D && v.len==k.len,"GPU KV row shape differs");
        let offset=self.positions.len()*KV*D;
        unsafe { append_kernel::launch_unchecked::<CudaRuntime>(&self.dev.client,
            CubeCount::new_1d(k.len.div_ceil(256) as u32),CubeDim::new_1d(256),
            k.arg(),v.arg(),self.k.arg(),self.v.arg(),k.len as u32,offset as u32); }
        self.dev.sync()?; self.positions.extend_from_slice(positions); Ok(())
    }
}

/// Exactly layer 2 for 1..=3 supplied residual rows, no KV history. Stage names
/// and GGML fastest-first dimensions match the independent CPU trace.
pub fn layer2(w: &mut impl Weights, input: &[f32], positions: &[i32]) -> Result<Trace> {
    block(w, input, positions, BlockConfig::LAYER2, ProjectionMode::Decoded)
}

/// Only attn_k uses packed GGUF bytes; every other operation uses the accepted
/// decoded-weight path. No tensor-core, activation-quantized or speed claim.
pub fn layer2_packed_k(w: &mut impl Weights, input: &[f32], positions: &[i32]) -> Result<Trace> {
    block(w, input, positions, BlockConfig::LAYER2, ProjectionMode::PackedK)
}

/// All ten type-40 projection families use packed bytes. F32 norms/scales and
/// BF16 router storage are unchanged; no additional layers or runtime state.
pub fn layer2_packed_all(w: &mut impl Weights, input: &[f32], positions: &[i32]) -> Result<Trace> {
    block(w, input, positions, BlockConfig::LAYER2, ProjectionMode::PackedAll)
}

pub fn block(w: &mut impl Weights, input: &[f32], positions: &[i32], config: BlockConfig, mode: ProjectionMode) -> Result<Trace> {
    Ok(block_core(w,input,positions,config,mode,None,true,None)?.trace)
}
pub(super) fn block_core(w:&mut impl Weights,input:&[f32],positions:&[i32],config:BlockConfig,
    mode:ProjectionMode,mut cache:Option<&mut KvCache>,trace:bool,fixed:Option<&FixedBlock>)->Result<super::math::BlockOutput> {
    let t = positions.len();
    ensure!((1..=3).contains(&t) && input.len() == H*t, "expected 1..=3 residual rows of width 2048");
    ensure!(positions.iter().all(|&p| p >= 0) && positions.windows(2).all(|p| p[0] < p[1]), "invalid positions");
    ensure!(input.iter().all(|v| v.is_finite()), "non-finite residual input");
    ensure!(config.sliding_window.is_none_or(|w|w>0),"sliding window must be positive");
    if let Some(f)=fixed {
        ensure!(matches!(mode,ProjectionMode::PackedAll) && f.config==config,"resident fixed configuration differs");
    }
    let dev = match cache.as_ref() { Some(c)=>Device {client:c.dev.client.clone()},None=>Device::new() };
    let mut tr = Trace::new(trace);
    let input_gpu = dev.upload(input);
    let pos = dev.positions(positions);
    dev.trace(&mut tr, "input", [H,t,1,1], &input_gpu)?;
    if trace { tr.stages.push(Stage { name: "positions".into(), dims: [t,1,1,1], values: Values::I32(positions.to_vec()) }); }
    let n = dev.norm_parameter(w,&input_gpu,"attn_norm.weight",&[H],fixed.map(|f|&f.attn_norm))?;
    dev.trace(&mut tr, "attn_norm", [H,t,1,1], &n)?;
    let q = dev.dense_parameter(w,&mut tr,"attn_q","q_proj",&n,H,Q*D,t,mode,fixed.map(|f|&f.q))?;
    let k = dev.dense_parameter(w,&mut tr,"attn_k","k_proj",&n,H,KV*D,t,mode,fixed.map(|f|&f.k))?;
    let v = dev.dense_parameter(w,&mut tr,"attn_v","v_proj",&n,H,KV*D,t,mode,fixed.map(|f|&f.v))?;
    let qn = dev.norm_parameter(w,&q,"attn_q_norm.weight",&[D],fixed.map(|f|&f.q_norm))?;
    let kn = dev.norm_parameter(w,&k,"attn_k_norm.weight",&[D],fixed.map(|f|&f.k_norm))?;
    dev.trace(&mut tr,"q_norm",[D,Q,t,1],&qn)?;
    dev.trace(&mut tr,"k_norm",[D,KV,t,1],&kn)?;
    let q = if config.rope { dev.rope(&qn,&pos,Q)? } else { qn };
    let k = if config.rope { dev.rope(&kn,&pos,KV)? } else { kn };
    dev.trace(&mut tr,if config.rope { "q_rope" } else { "q_no_rope" },[D,Q,t,1],&q)?;
    dev.trace(&mut tr,if config.rope { "k_rope" } else { "k_no_rope" },[D,KV,t,1],&k)?;
    let (k,v,key_positions)=if let Some(c)=cache.as_deref_mut() {
        c.append(&k,&v,positions)?; (&c.k,&c.v,c.positions.as_slice())
    } else { (&k,&v,positions) };
    let nt=key_positions.len();
    let key_pos=dev.positions(key_positions);
    let geometry=super::cache::Geometry::new(t,nt)?;
    let streamed_sinks;
    let sinks=if let Some(f)=fixed {&f.sinks} else {
        let values=w.get("attn_sinks.weight",&[Q],0,None)?;
        streamed_sinks=dev.weight_upload(w,&values,false);&streamed_sinks
    };
    ensure!(sinks.len == Q, "attention sink shape differs");
    let scores = dev.empty(geometry.scores);
    let probs = dev.empty(geometry.scores);
    let mask = dev.empty(geometry.mask);
    let heads = dev.empty(D*t*Q);
    let joined = dev.empty(Q*D*t);
    unsafe {
        mask_kernel::launch_unchecked::<CudaRuntime>(&dev.client, CubeCount::new_1d(geometry.mask.div_ceil(32) as u32), CubeDim::new_1d(32),
            pos.arg(),key_pos.arg(),mask.arg(),t as u32,nt as u32,config.sliding_window.unwrap_or(0),f32::NEG_INFINITY);
        attention_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
            CubeCount::new_1d((Q*t).div_ceil(32) as u32), CubeDim::new_1d(32),
            q.arg(),k.arg(),v.arg(),sinks.arg(),mask.arg(),scores.arg(),probs.arg(),heads.arg(),t as u32,nt as u32);
        join_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
            CubeCount::new_1d(joined.len.div_ceil(256) as u32), CubeDim::new_1d(256),heads.arg(),joined.arg(),t as u32);
    }
    dev.sync()?;
    dev.trace(&mut tr,"attention_scores",[nt,t,Q,1],&scores)?;
    dev.trace(&mut tr,"attention_mask",[nt,t,1,1],&mask)?;
    dev.trace(&mut tr,"attention_probs",[nt,t,Q,1],&probs)?;
    dev.trace(&mut tr,"attention_heads",[D,t,Q,1],&heads)?;
    let ao = dev.dense_parameter(w,&mut tr,"attn_output","attention_output",&joined,Q*D,H,t,mode,fixed.map(|f|&f.o))?;
    let residual = dev.point(&input_gpu,&ao,H,0)?;
    dev.trace(&mut tr,"ffn_inp",[H,t,1,1],&residual)?;
    let n = dev.norm_parameter(w,&residual,"ffn_norm.weight",&[H],fixed.map(|f|&f.ffn_norm))?;
    dev.trace(&mut tr,"ffn_norm",[H,t,1,1],&n)?;
    let ffn = if let FfnKind::Dense { width } = config.ffn {
        let dense=match fixed.map(|f|&f.ffn) {
            Some(FixedBranch::Dense(d))=>Some(d),None=>None,_=>anyhow::bail!("resident dense branch differs"),
        };
        dev.dense_ffn(w,&mut tr,&n,H,width,t,mode,dense)?
    } else {
        let streamed_router;
        let (router,shared)=match fixed.map(|f|&f.ffn) {
            Some(FixedBranch::Moe {router,shared})=>(router,Some(shared)),
            None=>{
                let values=w.get("ffn_gate_inp.weight",&[H,E],30,None)?;
                streamed_router=dev.weight_upload(w,&values,false);(&streamed_router,None)
            },
            _=>anyhow::bail!("resident MoE branch differs"),
        };
        let logits = dev.empty(E*t);
        dev.matvec(&n,router,&logits,H,E,t,0,0)?;
        let prob = dev.point(&logits,&logits,E,2)?;
        dev.trace(&mut tr,"router_logits",[E,t,1,1],&logits)?;
        dev.trace(&mut tr,"router_probs",[E,t,1,1],&prob)?;
        let ids_gpu = dev.empty(K*t);
        let selected = dev.empty(K*t);
        let sums = dev.empty(t);
        let denoms = dev.empty(t);
        let rw = dev.empty(K*t);
        let margins = dev.empty(t);
        unsafe { route_kernel::launch_unchecked::<CudaRuntime>(&dev.client, CubeCount::new_1d(1), CubeDim::new_1d(32),
            prob.arg(),ids_gpu.arg(),selected.arg(),sums.arg(),denoms.arg(),rw.arg(),margins.arg(),t as u32); }
        let ids = dev.read_ids(&ids_gpu)?;
        if trace { tr.stages.push(Stage { name: "router_ids".into(), dims: [K,t,1,1], values: Values::I32(ids.clone()) }); }
        dev.trace(&mut tr,"router_selected_probs",[1,K,t,1],&selected)?;
        dev.trace(&mut tr,"router_sum",[1,t,1,1],&sums)?;
        dev.trace(&mut tr,"router_denominator",[1,t,1,1],&denoms)?;
        dev.trace(&mut tr,"router_weights",[1,K,t,1],&rw)?;
        if trace { tr.router_margins = dev.read(&margins)?; }
        let up = dev.experts(w,&mut tr,"ffn_up_exps","expert_up",&n,H,F,t,&ids,false,mode)?;
        let gate = dev.experts(w,&mut tr,"ffn_gate_exps","expert_gate",&n,H,F,t,&ids,false,mode)?;
        let sw = dev.point(&gate,&up,F,3)?;
        dev.trace(&mut tr,"expert_swiglu",[F,K,t,1],&sw)?;
        let scales = dev.empty(K*t);
        unsafe { down_scale_kernel::launch_unchecked::<CudaRuntime>(&dev.client, CubeCount::new_1d(1), CubeDim::new_1d(32),
            sw.arg(),scales.arg(),F as u32,(K*t) as u32); }
        dev.sync()?;
        dev.trace(&mut tr,"expert_down_scale",[1,K,t,1],&scales)?;
        let di = dev.point(&sw,&scales,F,4)?;
        dev.trace(&mut tr,"expert_down_input",[F,K,t,1],&di)?;
        let before = dev.experts(w,&mut tr,"ffn_down_exps","expert_down_before_rescale",&di,F,H,t,&ids,true,mode)?;
        let down = dev.point(&before,&scales,H,1)?;
        dev.trace(&mut tr,"expert_down",[H,K,t,1],&down)?;
        let weighted = dev.point(&down,&rw,H,1)?;
        dev.trace(&mut tr,"expert_weighted",[H,K,t,1],&weighted)?;
        let moe = dev.empty(H*t);
        unsafe { moe_sum_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
            CubeCount::new_1d(moe.len.div_ceil(256) as u32),CubeDim::new_1d(256),weighted.arg(),moe.arg(),moe.len as u32); }
        dev.sync()?;
        dev.trace(&mut tr,"moe_out",[H,t,1,1],&moe)?;
        let su = dev.dense_parameter(w,&mut tr,"ffn_up_shexp","shared_up",&n,H,SH,t,mode,shared.map(|f|&f.up))?;
        let sg = dev.dense_parameter(w,&mut tr,"ffn_gate_shexp","shared_gate",&n,H,SH,t,mode,shared.map(|f|&f.gate))?;
        let ss = dev.point(&sg,&su,SH,3)?;
        dev.trace(&mut tr,"shared_swiglu",[SH,t,1,1],&ss)?;
        let sd = dev.dense_parameter(w,&mut tr,"ffn_down_shexp","shared_down",&ss,SH,H,t,mode,shared.map(|f|&f.down))?;
        dev.point(&moe,&sd,H,0)?
    };
    dev.trace(&mut tr,"ffn_out",[H,t,1,1],&ffn)?;
    let out = dev.point(&ffn,&residual,H,0)?;
    dev.trace(&mut tr,"l_out",[H,t,1,1],&out)?;
    let residual=dev.read(&out)?;
    ensure!(residual.iter().all(|v|v.is_finite()),"nonfinite CUDA block output");
    Ok(super::math::BlockOutput {residual,trace:tr})
}

/// Diagnostic final RMS plus a streamed output head. Every chunk is read back
/// before its upload is dropped; this is deliberately not a no-fence runtime.
pub struct OutputHead { dev: Device, normalized: Buffer, tokens: usize }
impl OutputHead {
    pub fn new(residual: &[f32], gain: &[f32]) -> Result<Self> {
        ensure!(gain.len()==H && residual.len()%H==0 && (1..=3).contains(&(residual.len()/H)),
            "invalid output RMS geometry");
        ensure!(residual.iter().chain(gain).all(|v|v.is_finite()),"nonfinite output RMS input");
        let dev=Device::new();
        let normalized=dev.norm(&dev.upload(residual),gain)?;
        Ok(Self {dev,normalized,tokens:residual.len()/H})
    }
    pub fn normalized(&self) -> Result<Vec<f32>> { self.dev.read(&self.normalized) }
    pub fn chunk(&self, raw: &[u8], rows: usize, scale: f32, mode: ProjectionMode) -> Result<Vec<f32>> {
        ensure!((1..=8192).contains(&rows) && raw.len()==rows*(H/64)*36 && scale.is_finite(),
            "invalid bounded output chunk");
        let storage = if matches!(mode,ProjectionMode::PackedAll) {
            ProjectionStorage::Packed(self.dev.upload_nvfp4(raw,H,rows)?)
        } else {
            let values=crate::formats::gguf::nvfp4::decode(raw,H*rows)?;
            ProjectionStorage::Decoded(self.dev.upload(&values))
        };
        let weight=Projection {ni:H,no:rows,storage};
        self.project_chunk(&weight,&self.dev.upload(&[scale]))
    }
    pub fn native_chunk(&self,view:&triblespace::core::blob::encodings::tensor::TensorView,
        rows:usize,scale:f32)->Result<Vec<f32>> {
        ensure!((1..=8192).contains(&rows) && scale.is_finite(),"invalid native head chunk");
        let weight=Projection {ni:H,no:rows,
            storage:ProjectionStorage::Packed(self.dev.upload_native_nvfp4(view,H,rows)?)};
        self.project_chunk(&weight,&self.dev.upload(&[scale]))
    }
    fn project_chunk(&self,weight:&Projection,scale:&Buffer)->Result<Vec<f32>> {
        let out=self.dev.empty(weight.no*self.tokens);
        self.dev.project(weight,&self.normalized,&out,self.tokens,0,0)?;
        self.dev.read(&self.dev.point(&out,scale,out.len,1)?)
    }
}

/// Native head only. Immutable paired scale and packed planes live with the
/// session's frozen pile observation; every query keeps the existing fences.
pub(super) struct ResidentOutputHead {
    dev:Device, gain:Buffer, scale:Buffer, chunks:Vec<Projection>,
    admission:super::forward::HeadAdmission,
}
impl ResidentOutputHead {
    pub fn new(gain:&[f32],full:&triblespace::core::blob::encodings::tensor::TensorView,
        scale:f32,admission:super::forward::HeadAdmission)->Result<Self> {
        ensure!(admission==super::forward::HeadAdmission::for_rows(admission.rows,admission.gpu_bytes)?,
            "resident head admission differs");
        ensure!(gain.len()==H && gain.iter().all(|v|v.is_finite()) && scale.is_finite(),"invalid resident norm/postscale");
        ensure!(full.dims()==[admission.rows as u64,H as u64],"resident head shape differs");
        let dev=Device::new();
        let gain=dev.upload(gain); let scale=dev.upload(&[scale]);
        let mut chunks=Vec::with_capacity(admission.chunks);
        for (first,rows) in super::forward::chunks(admission.rows) {
            let view=super::pile::native_rows(full,first,rows)?;
            let storage=ProjectionStorage::Packed(dev.upload_native_nvfp4(&view,H,rows)?);
            dev.sync()?; // Retire the bounded host copy only after upload completes.
            chunks.push(Projection {ni:H,no:rows,storage});
        }
        dev.sync()?;
        Ok(Self {dev,gain,scale,chunks,admission})
    }
    pub fn admission(&self)->super::forward::HeadAdmission { self.admission }
    pub fn fence(&self)->Result<()> { self.dev.sync() }
    pub fn run(&self,residual:&[f32])->Result<(Vec<f32>,Vec<f32>)> {
        ensure!(residual.len()%H==0 && (1..=3).contains(&(residual.len()/H))
            && residual.iter().all(|v|v.is_finite()),"invalid resident head input");
        let normalized=self.dev.norm_borrowed(&self.dev.upload(residual),&self.gain)?;
        let head=OutputHead {dev:Device {client:self.dev.client.clone()},normalized,tokens:residual.len()/H};
        let normalized=head.normalized()?;
        let mut logits=vec![0.0;self.admission.rows*head.tokens];
        let mut first=0;
        for weight in &self.chunks {
            let values=head.project_chunk(weight,&self.scale)?;
            super::forward::insert_chunk(&mut logits,&values,self.admission.rows,first,weight.no,head.tokens)?;
            first+=weight.no;
        }
        ensure!(normalized.iter().chain(&logits).all(|v|v.is_finite()),"nonfinite resident norm/logits");
        Ok((normalized,logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation; synthetic, no model"]
    fn cuda_cached_514_keys_full_local_and_crossing_chunk() -> Result<()> {
        let mut cache=KvCache::new(514)?;
        let dev=Device {client:cache.dev.client.clone()};
        let positions:Vec<i32>=(0..514).collect();
        let key_values:Vec<f32>=(0..514*KV*D).map(|i| {
            let key=i/(KV*D); let head=i/D%KV; let d=i%D;
            match d {0=>((key*7+head*3)%17) as f32/16.0-0.5,7=>(key%11) as f32/32.0-0.125,_=>0.0}
        }).collect();
        let value_values:Vec<f32>=(0..514*KV*D).map(|i| {
            let key=i/(KV*D); let head=i/D%KV; let d=i%D;
            (key%19) as f32/16.0 + head as f32/8.0 - d as f32/256.0 + if key==0 {10.0} else {0.0}
        }).collect();
        for part in positions.chunks(3) {
            let first=part[0] as usize*KV*D; let end=first+part.len()*KV*D;
            cache.append(&dev.upload(&key_values[first..end]),&dev.upload(&value_values[first..end]),part)?;
        }
        assert_eq!(cache.len(),514);
        assert_eq!(dev.read(&cache.k)?,key_values); assert_eq!(dev.read(&cache.v)?,value_values);
        let queries=[511,512,513]; let t=queries.len(); let nt=cache.len();
        let query_values:Vec<f32>=(0..t*Q*D).map(|i| {
            let query=i/(Q*D); let head=i/D%Q;
            match i%D {0=>(1+head%3) as f32/8.0,7=>-((1+query) as f32)/16.0,_=>0.0}
        }).collect();
        let sink_values:Vec<f32>=(0..Q).map(|h|(h%5) as f32/8.0-0.25).collect();
        let q=dev.upload(&query_values); let pos=dev.positions(&queries);
        let keys=dev.positions(&positions); let sinks=dev.upload(&sink_values);
        let geometry=super::super::cache::Geometry::new(t,nt)?;
        for window in [0,513] {
            let mask=dev.empty(geometry.mask); let scores=dev.empty(geometry.scores);
            let probs=dev.empty(geometry.scores); let heads=dev.empty(Q*t*D);
            unsafe {
                mask_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
                    CubeCount::new_1d(geometry.mask.div_ceil(32) as u32),CubeDim::new_1d(32),
                    pos.arg(),keys.arg(),mask.arg(),t as u32,nt as u32,window,f32::NEG_INFINITY);
                attention_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
                    CubeCount::new_1d((Q*t).div_ceil(32) as u32),CubeDim::new_1d(32),
                    q.arg(),cache.k.arg(),cache.v.arg(),sinks.arg(),mask.arg(),scores.arg(),probs.arg(),heads.arg(),t as u32,nt as u32);
            }
            dev.sync()?;
            let actual_mask=dev.read(&mask)?;
            for (row,&query) in queries.iter().enumerate() { for (col,&key) in positions.iter().enumerate() {
                let allowed=key<=query && (window==0 || query-key<window);
                assert_eq!(actual_mask[row*nt+col],if allowed {0.0} else {f32::NEG_INFINITY});
            }}
            let mut expected=vec![0.0;Q*t*D];
            for h in 0..Q { for (qi,&query) in queries.iter().enumerate() {
                let mut logits=vec![f64::NEG_INFINITY;nt];
                for (ki,&key) in positions.iter().enumerate() {
                    if key<=query && (window==0 || query-key<window) {
                        let dot:f64=[0,7].into_iter().map(|d|
                            query_values[(qi*Q+h)*D+d] as f64*key_values[(ki*KV+h/8)*D+d] as f64).sum();
                        logits[ki]=dot/(D as f64).sqrt();
                    }
                }
                let max=logits.iter().copied().fold(sink_values[h] as f64,f64::max);
                let exps:Vec<f64>=logits.iter().map(|&x|(x-max).exp()).collect();
                let denom=exps.iter().sum::<f64>()+(sink_values[h] as f64-max).exp();
                for d in 0..D {
                    expected[(h*t+qi)*D+d]=exps.iter().enumerate().map(|(ki,&p)|
                        p/denom*value_values[(ki*KV+h/8)*D+d] as f64).sum::<f64>() as f32;
                }
            }}
            // Independent scalar synthetic witness, not a model-quality tolerance.
            close(&dev.read(&heads)?,&expected,2e-5);
        }
        let row=dev.upload(&vec![2.0;KV*D]);
        assert!(cache.append(&row,&row,&[514]).is_err()); assert_eq!(cache.len(),514);
        cache.reset()?; assert_eq!(cache.len(),0);
        cache.append(&row,&row,&[0])?; assert_eq!(cache.len(),1);
        assert_eq!(&dev.read(&cache.k)?[..KV*D],&vec![2.0;KV*D]);
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation"]
    fn cuda_full_vs_sliding_mask_gapped_positions_t1_t3() -> Result<()> {
        let dev = Device::new();
        for positions in [vec![0],vec![0,512,513]] {
            let t = positions.len(); let pos = dev.positions(&positions);
            for window in [0,513] {
                let mask = dev.empty(t*t);
                unsafe { mask_kernel::launch_unchecked::<CudaRuntime>(&dev.client,CubeCount::new_1d(1),CubeDim::new_1d(32),
                    pos.arg(),pos.arg(),mask.arg(),t as u32,t as u32,window,f32::NEG_INFINITY); }
                let actual = dev.read(&mask)?;
                let expected: Vec<_> = (0..t*t).map(|i| {
                    let distance = positions[i/t]-positions[i%t];
                    if distance < 0 || (window > 0 && distance >= window) { f32::NEG_INFINITY } else { 0.0 }
                }).collect();
                assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                if t == 3 { assert_eq!(actual[6],if window == 0 {0.0} else {f32::NEG_INFINITY}); }
            }
        }
        Ok(())
    }

    fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(a.is_finite() && (a-e).abs() <= tolerance,
                "element {i}: actual {a}, expected {e}, absolute tolerance {tolerance}");
        }
    }

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation"]
    fn cuda_matvec_norm_and_broadcast_t1_t3() -> Result<()> {
        let dev = Device::new();
        for t in [1, 3] {
            // Width 33 exercises the strided tail, nine outputs exercise a
            // partly occupied cube, and offsets catch token/slot addressing.
            let ni = 33;
            let no = 9;
            let x: Vec<f32> = (0..ni*(t+1)).map(|i| (i as i32 % 17 - 8) as f32 / 16.0).collect();
            let w: Vec<f32> = (0..ni*no).map(|i| (i as i32 % 13 - 6) as f32 / 32.0).collect();
            let y = dev.upload(&vec![-123.0; no*(t+2)]);
            dev.matvec(&dev.upload(&x),&dev.upload(&w),&y,ni,no,t,ni,no)?;
            let mut expected = vec![-123.0; no*(t+2)];
            for token in 0..t { for row in 0..no {
                expected[(token+1)*no+row] = (0..ni).map(|i| w[row*ni+i]*x[(token+1)*ni+i]).sum();
            }}
            close(&dev.read(&y)?,&expected,0.0); // Exactly representable binary fractions.

            let x: Vec<f32> = (0..5*t).map(|i| (i+1) as f32 / 8.0).collect();
            let a = dev.upload(&x);
            let scales: Vec<f32> = (0..t).map(|i| (i+1) as f32).collect();
            let scale = dev.upload(&scales);
            let scaled = dev.point(&a,&scale,5,1)?;
            let expected: Vec<f32> = x.iter().enumerate().map(|(i,&v)| v*scales[i/5]).collect();
            close(&dev.read(&scaled)?,&expected,0.0);
            close(&dev.read(&dev.point(&scaled,&scale,5,4)?)?,&x,0.0);
            let expected: Vec<f32> = x.chunks_exact(5).flat_map(|row| {
                let inv = ((row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()/5.0 + 1e-6).sqrt().recip()) as f32;
                row.iter().map(move |&v| v*inv)
            }).collect();
            close(&dev.read(&dev.norm(&a,&[1.0;5])?)?,&expected,1e-6);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation"]
    fn cuda_route_and_adaptive_scale_t1_t3() -> Result<()> {
        let dev = Device::new();
        for t in [1, 3] {
            let p: Vec<f32> = (0..E*t).map(|i| match i/E {
                0 => 0.0, 1 => 0.5, _ => (i%E) as f32/512.0,
            }).collect();
            let prob = dev.upload(&p);
            let ids = dev.empty(K*t);
            let selected = dev.empty(K*t);
            let sums = dev.empty(t);
            let denoms = dev.empty(t);
            let rw = dev.empty(K*t);
            let margins = dev.empty(t);
            unsafe { route_kernel::launch_unchecked::<CudaRuntime>(&dev.client,CubeCount::new_1d(1),CubeDim::new_1d(32),
                prob.arg(),ids.arg(),selected.arg(),sums.arg(),denoms.arg(),rw.arg(),margins.arg(),t as u32); }
            let expected_ids: Vec<i32> = (0..t).flat_map(|token| (0..K).map(move |s|
                if token == 2 { (E-1-s) as i32 } else { s as i32 })).collect();
            assert_eq!(dev.read_ids(&ids)?,expected_ids);
            let selected_ref: Vec<f32> = expected_ids.iter().enumerate().map(|(s,&id)| p[(s/K)*E+id as usize]).collect();
            close(&dev.read(&selected)?,&selected_ref,0.0);
            let sums_ref: Vec<f32> = selected_ref.chunks_exact(K).map(|r| r.iter().sum()).collect();
            close(&dev.read(&sums)?,&sums_ref,0.0);
            let denoms_ref: Vec<f32> = sums_ref.iter().map(|&s| s.max(6.103515625e-5)).collect();
            close(&dev.read(&denoms)?,&denoms_ref,0.0);
            let weights_ref: Vec<f32> = selected_ref.iter().enumerate().map(|(s,&v)| v/denoms_ref[s/K]).collect();
            close(&dev.read(&rw)?,&weights_ref,1e-7);
            let margin_ref: Vec<f32> = (0..t).map(|i| if i==2 {1.0/512.0} else {0.0}).collect();
            close(&dev.read(&margins)?,&margin_ref,0.0);
        }
        let x = dev.upload(&[0.0,0.0,0.0, 0.0,1025.0,0.0, 0.0,-4096.0,0.0]);
        let scale = dev.empty(3);
        unsafe { down_scale_kernel::launch_unchecked::<CudaRuntime>(&dev.client,CubeCount::new_1d(1),CubeDim::new_1d(32),
            x.arg(),scale.arg(),3,3); }
        close(&dev.read(&scale)?,&[1.0,2.0,4.0],1e-6);
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation"]
    fn cuda_rope_attention_and_head_join_t1_t3() -> Result<()> {
        let dev = Device::new();
        for t in [1, 3] {
            let positions: Vec<i32> = (0..t as i32).collect();
            let pos = dev.positions(&positions);
            let x: Vec<f32> = (0..D*KV*t).map(|i| (i%37) as f32/32.0).collect();
            let mut rope_ref = x.clone();
            for token in 0..t { for h in 0..KV { for j in 0..D/2 {
                let angle = positions[token] as f32 * 10000.0f32.powf(-2.0*j as f32/D as f32);
                let (sin,cos) = angle.sin_cos();
                let i = (token*KV+h)*D+j;
                rope_ref[i] = x[i]*cos-x[i+D/2]*sin;
                rope_ref[i+D/2] = x[i]*sin+x[i+D/2]*cos;
            }}}
            close(&dev.read(&dev.rope(&dev.upload(&x),&pos,KV)?)?,&rope_ref,1e-6);
            let q = dev.upload(&vec![0.0;Q*D*t]);
            let k = dev.upload(&vec![0.0;KV*D*t]);
            let values: Vec<f32> = (0..KV*D*t).map(|i|
                (i/(KV*D)) as f32 * 4.0 + ((i/D)%KV) as f32 + (i%D) as f32/128.0).collect();
            let v = dev.upload(&values);
            let sinks = dev.upload(&[std::f32::consts::LN_2;Q]);
            let mask = dev.empty(t*t);
            let scores = dev.empty(Q*t*t);
            let probs = dev.empty(Q*t*t);
            let heads = dev.empty(Q*t*D);
            let joined = dev.empty(Q*t*D);
            unsafe {
                mask_kernel::launch_unchecked::<CudaRuntime>(&dev.client,CubeCount::new_1d(1),CubeDim::new_1d(32),
                    pos.arg(),pos.arg(),mask.arg(),t as u32,t as u32,513,f32::NEG_INFINITY);
                attention_kernel::launch_unchecked::<CudaRuntime>(&dev.client,CubeCount::new_1d(t as u32),CubeDim::new_1d(32),
                    q.arg(),k.arg(),v.arg(),sinks.arg(),mask.arg(),scores.arg(),probs.arg(),heads.arg(),t as u32,t as u32);
                join_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
                    CubeCount::new_1d(joined.len.div_ceil(256) as u32),CubeDim::new_1d(256),heads.arg(),joined.arg(),t as u32);
            }
            let mask_ref: Vec<f32> = (0..t*t).map(|i| if i%t > i/t {f32::NEG_INFINITY} else {0.0}).collect();
            assert_eq!(dev.read(&mask)?,mask_ref);
            close(&dev.read(&scores)?,&vec![0.0;Q*t*t],0.0);
            let prob_ref: Vec<f32> = (0..Q*t*t).map(|i| {
                let query = (i/t)%t;
                if i%t <= query { 1.0/(query+3) as f32 } else { 0.0 }
            }).collect();
            close(&dev.read(&probs)?,&prob_ref,1e-7);
            let mut head_ref = vec![0.0;Q*t*D];
            let mut join_ref = head_ref.clone();
            for h in 0..Q { for query in 0..t { for d in 0..D {
                let sum: f32 = (0..=query).map(|key| values[(key*KV+h/(Q/KV))*D+d]/(query+3) as f32).sum();
                head_ref[(h*t+query)*D+d] = sum;
                join_ref[(query*Q+h)*D+d] = sum;
            }}}
            close(&dev.read(&heads)?,&head_ref,2e-6);
            close(&dev.read(&joined)?,&join_ref,2e-6);
        }
        Ok(())
    }
}
