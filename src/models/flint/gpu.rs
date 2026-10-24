//! Bounded layer-2 CUDA arithmetic through Mary's existing CubeCL runtime.
//!
//! Weights are decoded from stored GGUF NVFP4 (36 bytes / 64 values), then
//! uploaded as F32. Separate GGUF `.scale` values stay after the matmul. This
//! does not use Mary's other FP4 packing, tensor cores, activation quantization,
//! BF16, or llama.cpp's Q8_1/W4A4 expert-down dispatch.
//!
//! All layer arithmetic, including top-six selection, runs on the GPU. This is
//! a diagnostic graph: every named stage is read back; selected IDs return to
//! the host to choose bounded expert slices. It is not a GPU-resident runtime.
//! Weight uploads are retired after each projection readback; no full bank is
//! retained. There is no KV history or full-model loader.
//!
//! Numerical tier, fixed before comparison: F32 input/weight/output and F32
//! accumulation. Matvec uses 32 strided lanes per row and a five-step tree;
//! RMS uses 256 strided lanes and an eight-step tree. Attention and router
//! reductions are serial F32; GPU transcendental functions and CUDA compiler
//! multiply-add contraction may differ from CPU scalar f64 reductions. No
//! autotuning, fitted tolerance, or claim of GGML reduction-order parity.

use anyhow::{Result, anyhow, ensure};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::math::{H, Stage, Trace, Values};
use super::weights::Weights;

const D: usize = 128;
const Q: usize = 32;
const KV: usize = 4;
const E: usize = 256;
const K: usize = 6;
const F: usize = 512;
const SH: usize = 1024;
type Client = cubecl::client::ComputeClient<CudaRuntime>;

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
fn mask_kernel(positions: &Array<i32>, mask: &mut Array<f32>, t: u32, negative_infinity: f32) {
    let i = ABSOLUTE_POS as u32;
    if i < t * t {
        let query = positions[(i / t) as usize];
        let key = positions[(i % t) as usize];
        let mut m = f32::new(0.0);
        // A scalar binding avoids CubeCL emitting the invalid CUDA literal -inf.
        if key > query || query - key >= 513 { m = negative_infinity; }
        mask[i as usize] = m;
    }
}

#[cube(launch_unchecked)]
fn attention_kernel(q: &Array<f32>, k: &Array<f32>, v: &Array<f32>, sinks: &Array<f32>,
    mask: &Array<f32>, scores: &mut Array<f32>, probs: &mut Array<f32>,
    heads: &mut Array<f32>, t: u32) {
    let hq = ABSOLUTE_POS as u32;
    if hq < 32 * t {
        let h = hq / t;
        let query = hq % t;
        let base = hq * t;
        let mut logits = Array::<f32>::new(3usize);
        let mut maximum = sinks[h as usize];
        for key in 0..t {
            let mut dot = f32::new(0.0);
            for d in 0..128u32 {
                dot += q[((query * 32 + h) * 128 + d) as usize]
                    * k[((key * 4 + h / 8) * 128 + d) as usize];
            }
            scores[(base + key) as usize] = dot;
            let logit = dot * (1.0 / 128.0f32.sqrt()) + mask[(query * t + key) as usize];
            logits[key as usize] = logit;
            maximum = maximum.max(logit);
        }
        let mut denom = f32::new(0.0);
        for key in 0..t {
            let e = (logits[key as usize] - maximum).exp();
            logits[key as usize] = e;
            denom += e;
        }
        denom += (sinks[h as usize] - maximum).exp();
        for key in 0..t {
            logits[key as usize] = logits[key as usize] / denom;
            probs[(base + key) as usize] = logits[key as usize];
        }
        for d in 0..128u32 {
            let mut sum = f32::new(0.0);
            for key in 0..t {
                sum += logits[key as usize] * v[((key * 4 + h / 8) * 128 + d) as usize];
            }
            heads[(hq * 128 + d) as usize] = sum;
        }
    }
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
    fn positions(&self, x: &[i32]) -> Buffer {
        Buffer { handle: self.client.create_from_slice(i32::as_bytes(x)), len: x.len() }
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
        tr.stages.push(Stage { name: name.into(), dims, values: Values::F32(self.read(x)?) });
        Ok(())
    }
    fn matvec(&self, x: &Buffer, w: &Buffer, y: &Buffer, ni: usize, no: usize,
        batch: usize, input_offset: usize, output_offset: usize) {
        assert_eq!(w.len, ni * no);
        assert!(input_offset + ni * batch <= x.len && output_offset + no * batch <= y.len);
        unsafe { matvec_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d((no * batch).div_ceil(8) as u32), CubeDim::new_1d(256),
            x.arg(), w.arg(), y.arg(), ni as u32, no as u32, batch as u32,
            input_offset as u32, output_offset as u32,
        ); }
    }
    fn norm(&self, x: &Buffer, weights: &[f32]) -> Buffer {
        assert!(!weights.is_empty());
        assert_eq!(x.len % weights.len(), 0);
        let w = self.upload(weights);
        let y = self.empty(x.len);
        unsafe { norm_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d((x.len / weights.len()) as u32), CubeDim::new_1d(256),
            x.arg(), w.arg(), y.arg(), weights.len() as u32,
        ); }
        y
    }
    fn point(&self, a: &Buffer, b: &Buffer, width: usize, op: u32) -> Buffer {
        assert!(width > 0);
        match op { 0 | 3 => assert_eq!(a.len, b.len), 1 | 4 => {
            assert_eq!(a.len % width, 0); assert_eq!(b.len, a.len / width);
        }, 2 => {}, _ => panic!("invalid point operation") }
        let y = self.empty(a.len);
        unsafe { point_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d(a.len.div_ceil(256) as u32), CubeDim::new_1d(256),
            a.arg(), b.arg(), y.arg(), a.len as u32, width as u32, op,
        ); }
        y
    }
    fn rope(&self, x: &Buffer, positions: &Buffer, heads: usize) -> Buffer {
        assert_eq!(x.len, positions.len * heads * D);
        let y = self.empty(x.len);
        unsafe { rope_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d((x.len / 2).div_ceil(256) as u32), CubeDim::new_1d(256),
            x.arg(), positions.arg(), y.arg(), heads as u32, (x.len / 2) as u32,
        ); }
        y
    }
    fn dense(&self, w: &mut impl Weights, tr: &mut Trace, suffix: &str, label: &str,
        x: &Buffer, ni: usize, no: usize, t: usize) -> Result<Buffer> {
        let weight = self.upload(&w.get(&format!("{suffix}.weight"), &[ni, no], 40, None)?);
        let scale = self.upload(&w.get(&format!("{suffix}.scale"), &[1], 0, None)?);
        let raw = self.empty(no * t);
        self.matvec(x, &weight, &raw, ni, no, t, 0, 0);
        self.trace(tr, &format!("{label}_unscaled"), [no,t,1,1], &raw)?;
        let out = self.point(&raw, &scale, raw.len, 1);
        self.trace(tr, label, [no,t,1,1], &out)?;
        Ok(out)
    }
    fn experts(&self, w: &mut impl Weights, tr: &mut Trace, suffix: &str, label: &str,
        x: &Buffer, ni: usize, no: usize, t: usize, ids: &[i32], per_slot: bool) -> Result<Buffer> {
        ensure!(ids.len() == K * t && ids.iter().all(|&i| (0..E as i32).contains(&i)), "invalid GPU expert IDs");
        let scales = w.get(&format!("{suffix}.scale"), &[E], 0, None)?;
        let raw = self.empty(no * K * t);
        let mut selected = Vec::with_capacity(ids.len());
        // One decoded/uploaded expert matrix at a time. Synchronizing the
        // completed slot retires its upload before loading the next one.
        for (slot, &id) in ids.iter().enumerate() {
            let weight = self.upload(&w.get(&format!("{suffix}.weight"), &[ni,no,E], 40, Some(id as usize))?);
            let row = if per_slot { slot } else { slot / K };
            self.matvec(x, &weight, &raw, ni, no, 1, row * ni, slot * no);
            cubecl::future::block_on(self.client.sync())
                .map_err(|e| anyhow!("CUDA expert synchronization: {e:?}"))?;
            selected.push(scales[id as usize]);
        }
        let scale = self.upload(&selected);
        let out = self.point(&raw, &scale, no, 1);
        self.trace(tr, &format!("{label}_unscaled"), [no,K,t,1], &raw)?;
        self.trace(tr, &format!("{label}_selected_scale"), [1,K,t,1], &scale)?;
        self.trace(tr, label, [no,K,t,1], &out)?;
        Ok(out)
    }
}

/// Exactly layer 2 for 1..=3 supplied residual rows, no KV history. Stage names
/// and GGML fastest-first dimensions match the independent CPU trace.
pub fn layer2(w: &mut impl Weights, input: &[f32], positions: &[i32]) -> Result<Trace> {
    let t = positions.len();
    ensure!((1..=3).contains(&t) && input.len() == H*t, "expected 1..=3 residual rows of width 2048");
    ensure!(positions.iter().all(|&p| p >= 0) && positions.windows(2).all(|p| p[0] < p[1]), "invalid positions");
    ensure!(input.iter().all(|v| v.is_finite()), "non-finite residual input");
    let dev = Device::new();
    let mut tr = Trace::default();
    let input_gpu = dev.upload(input);
    let pos = dev.positions(positions);
    dev.trace(&mut tr, "input", [H,t,1,1], &input_gpu)?;
    tr.stages.push(Stage { name: "positions".into(), dims: [t,1,1,1], values: Values::I32(dev.read_ids(&pos)?) });
    let n = dev.norm(&input_gpu, &w.get("attn_norm.weight", &[H], 0, None)?);
    dev.trace(&mut tr, "attn_norm", [H,t,1,1], &n)?;
    let q = dev.dense(w,&mut tr,"attn_q","q_proj",&n,H,Q*D,t)?;
    let k = dev.dense(w,&mut tr,"attn_k","k_proj",&n,H,KV*D,t)?;
    let v = dev.dense(w,&mut tr,"attn_v","v_proj",&n,H,KV*D,t)?;
    let qn = dev.norm(&q,&w.get("attn_q_norm.weight",&[D],0,None)?);
    let kn = dev.norm(&k,&w.get("attn_k_norm.weight",&[D],0,None)?);
    dev.trace(&mut tr,"q_norm",[D,Q,t,1],&qn)?;
    dev.trace(&mut tr,"k_norm",[D,KV,t,1],&kn)?;
    let q = dev.rope(&qn,&pos,Q);
    let k = dev.rope(&kn,&pos,KV);
    dev.trace(&mut tr,"q_rope",[D,Q,t,1],&q)?;
    dev.trace(&mut tr,"k_rope",[D,KV,t,1],&k)?;
    let sinks = dev.upload(&w.get("attn_sinks.weight",&[Q],0,None)?);
    ensure!(sinks.len == Q, "attention sink shape differs");
    let scores = dev.empty(t*t*Q);
    let probs = dev.empty(t*t*Q);
    let mask = dev.empty(t*t);
    let heads = dev.empty(D*t*Q);
    let joined = dev.empty(Q*D*t);
    unsafe {
        mask_kernel::launch_unchecked::<CudaRuntime>(&dev.client, CubeCount::new_1d(1), CubeDim::new_1d(32),
            pos.arg(),mask.arg(),t as u32,f32::NEG_INFINITY);
        attention_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
            CubeCount::new_1d((Q*t).div_ceil(32) as u32), CubeDim::new_1d(32),
            q.arg(),k.arg(),v.arg(),sinks.arg(),mask.arg(),scores.arg(),probs.arg(),heads.arg(),t as u32);
        join_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
            CubeCount::new_1d(joined.len.div_ceil(256) as u32), CubeDim::new_1d(256),heads.arg(),joined.arg(),t as u32);
    }
    dev.trace(&mut tr,"attention_scores",[t,t,Q,1],&scores)?;
    dev.trace(&mut tr,"attention_mask",[t,t,1,1],&mask)?;
    dev.trace(&mut tr,"attention_probs",[t,t,Q,1],&probs)?;
    dev.trace(&mut tr,"attention_heads",[D,t,Q,1],&heads)?;
    let ao = dev.dense(w,&mut tr,"attn_output","attention_output",&joined,Q*D,H,t)?;
    let residual = dev.point(&input_gpu,&ao,H,0);
    dev.trace(&mut tr,"ffn_inp",[H,t,1,1],&residual)?;
    let n = dev.norm(&residual,&w.get("ffn_norm.weight",&[H],0,None)?);
    dev.trace(&mut tr,"ffn_norm",[H,t,1,1],&n)?;
    let router = dev.upload(&w.get("ffn_gate_inp.weight",&[H,E],30,None)?);
    let logits = dev.empty(E*t);
    dev.matvec(&n,&router,&logits,H,E,t,0,0);
    let prob = dev.point(&logits,&logits,E,2);
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
    tr.stages.push(Stage { name: "router_ids".into(), dims: [K,t,1,1], values: Values::I32(ids.clone()) });
    dev.trace(&mut tr,"router_selected_probs",[1,K,t,1],&selected)?;
    dev.trace(&mut tr,"router_sum",[1,t,1,1],&sums)?;
    dev.trace(&mut tr,"router_denominator",[1,t,1,1],&denoms)?;
    dev.trace(&mut tr,"router_weights",[1,K,t,1],&rw)?;
    tr.router_margins = dev.read(&margins)?;
    let up = dev.experts(w,&mut tr,"ffn_up_exps","expert_up",&n,H,F,t,&ids,false)?;
    let gate = dev.experts(w,&mut tr,"ffn_gate_exps","expert_gate",&n,H,F,t,&ids,false)?;
    let sw = dev.point(&gate,&up,F,3);
    dev.trace(&mut tr,"expert_swiglu",[F,K,t,1],&sw)?;
    let scales = dev.empty(K*t);
    unsafe { down_scale_kernel::launch_unchecked::<CudaRuntime>(&dev.client, CubeCount::new_1d(1), CubeDim::new_1d(32),
        sw.arg(),scales.arg(),F as u32,(K*t) as u32); }
    dev.trace(&mut tr,"expert_down_scale",[1,K,t,1],&scales)?;
    let di = dev.point(&sw,&scales,F,4);
    dev.trace(&mut tr,"expert_down_input",[F,K,t,1],&di)?;
    let before = dev.experts(w,&mut tr,"ffn_down_exps","expert_down_before_rescale",&di,F,H,t,&ids,true)?;
    let down = dev.point(&before,&scales,H,1);
    dev.trace(&mut tr,"expert_down",[H,K,t,1],&down)?;
    let weighted = dev.point(&down,&rw,H,1);
    dev.trace(&mut tr,"expert_weighted",[H,K,t,1],&weighted)?;
    let moe = dev.empty(H*t);
    unsafe { moe_sum_kernel::launch_unchecked::<CudaRuntime>(&dev.client,
        CubeCount::new_1d(moe.len.div_ceil(256) as u32),CubeDim::new_1d(256),weighted.arg(),moe.arg(),moe.len as u32); }
    dev.trace(&mut tr,"moe_out",[H,t,1,1],&moe)?;
    let su = dev.dense(w,&mut tr,"ffn_up_shexp","shared_up",&n,H,SH,t)?;
    let sg = dev.dense(w,&mut tr,"ffn_gate_shexp","shared_gate",&n,H,SH,t)?;
    let ss = dev.point(&sg,&su,SH,3);
    dev.trace(&mut tr,"shared_swiglu",[SH,t,1,1],&ss)?;
    let sd = dev.dense(w,&mut tr,"ffn_down_shexp","shared_down",&ss,SH,H,t)?;
    let ffn = dev.point(&moe,&sd,H,0);
    dev.trace(&mut tr,"ffn_out",[H,t,1,1],&ffn)?;
    let out = dev.point(&ffn,&residual,H,0);
    dev.trace(&mut tr,"l_out",[H,t,1,1],&out)?;
    Ok(tr)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            dev.matvec(&dev.upload(&x),&dev.upload(&w),&y,ni,no,t,ni,no);
            let mut expected = vec![-123.0; no*(t+2)];
            for token in 0..t { for row in 0..no {
                expected[(token+1)*no+row] = (0..ni).map(|i| w[row*ni+i]*x[(token+1)*ni+i]).sum();
            }}
            close(&dev.read(&y)?,&expected,0.0); // Exactly representable binary fractions.

            let x: Vec<f32> = (0..5*t).map(|i| (i+1) as f32 / 8.0).collect();
            let a = dev.upload(&x);
            let scales: Vec<f32> = (0..t).map(|i| (i+1) as f32).collect();
            let scale = dev.upload(&scales);
            let scaled = dev.point(&a,&scale,5,1);
            let expected: Vec<f32> = x.iter().enumerate().map(|(i,&v)| v*scales[i/5]).collect();
            close(&dev.read(&scaled)?,&expected,0.0);
            close(&dev.read(&dev.point(&scaled,&scale,5,4))?,&x,0.0);
            let expected: Vec<f32> = x.chunks_exact(5).flat_map(|row| {
                let inv = ((row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()/5.0 + 1e-6).sqrt().recip()) as f32;
                row.iter().map(move |&v| v*inv)
            }).collect();
            close(&dev.read(&dev.norm(&a,&[1.0;5]))?,&expected,1e-6);
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
            close(&dev.read(&dev.rope(&dev.upload(&x),&pos,KV))?,&rope_ref,1e-6);
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
                    pos.arg(),mask.arg(),t as u32,f32::NEG_INFINITY);
                attention_kernel::launch_unchecked::<CudaRuntime>(&dev.client,CubeCount::new_1d(t as u32),CubeDim::new_1d(32),
                    q.arg(),k.arg(),v.arg(),sinks.arg(),mask.arg(),scores.arg(),probs.arg(),heads.arg(),t as u32);
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
