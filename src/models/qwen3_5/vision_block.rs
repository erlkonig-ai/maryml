//! Native BF16 vision block; composed by `vision_tower`.
//! Fixed-order GPU arithmetic, 12 typed immutable pile roles, no upload fallback.
//! Rotary recipe: actual HF VisionModel converted wholesale to BF16, including
//! inv_freq. This is explicit synthetic-oracle scope, NOT all checkpoint loaders.
//! Biased linear uses F32 dot+bias then BF16; parity is deliberately unproven.
pub use super::vision_frontend::Grid;
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use serde::{Deserialize, Serialize};
use triblespace::core::{
    blob::{
        Blob,
        encodings::tensor::{Tensor as NativeTensor, elements::BF16},
    },
    inline::{Inline, encodings::hash::Handle},
    repo::BlobStoreGet,
};
pub type CudaTensor = CubeTensor<CudaRuntime>;
pub type Slot<const R: usize> = Inline<Handle<NativeTensor<BF16, R>>>;
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub merge: usize,
}
impl Config {
    pub fn validate(self) -> Result<(), String> {
        if self.hidden == 0
            || self.hidden > 1152
            || self.heads == 0
            || self.heads > 16
            || self.hidden % self.heads != 0
            || (self.hidden / self.heads) % 4 != 0
            || self.intermediate == 0
            || self.intermediate > 4304
            || self.merge != 2
        {
            return Err("vision requires H<=1152,heads<=16,H divisible by heads,D divisible by4,M<=4304,merge2".into());
        }
        Ok(())
    }
}
pub struct Slots {
    pub norm1_weight: Slot<1>,
    pub norm1_bias: Slot<1>,
    pub norm2_weight: Slot<1>,
    pub norm2_bias: Slot<1>,
    pub qkv_weight: Slot<2>,
    pub qkv_bias: Slot<1>,
    pub proj_weight: Slot<2>,
    pub proj_bias: Slot<1>,
    pub fc1_weight: Slot<2>,
    pub fc1_bias: Slot<1>,
    pub fc2_weight: Slot<2>,
    pub fc2_bias: Slot<1>,
}
pub struct Block {
    c: Config,
    n1w: CudaTensor,
    n1b: CudaTensor,
    n2w: CudaTensor,
    n2b: CudaTensor,
    qw: CudaTensor,
    qb: CudaTensor,
    pw: CudaTensor,
    pb: CudaTensor,
    w1: CudaTensor,
    b1: CudaTensor,
    w2: CudaTensor,
    b2: CudaTensor,
}
pub struct Output {
    pub norm1: CudaTensor,
    pub qkv: CudaTensor,
    pub q: CudaTensor,
    pub k: CudaTensor,
    pub v: CudaTensor,
    pub angles: CudaTensor,
    pub cos: CudaTensor,
    pub sin: CudaTensor,
    pub q_rot: CudaTensor,
    pub k_rot: CudaTensor,
    pub attended: CudaTensor,
    pub projected: CudaTensor,
    pub residual: CudaTensor,
    pub norm2: CudaTensor,
    pub fc1: CudaTensor,
    pub activated: CudaTensor,
    pub fc2: CudaTensor,
    pub hidden: CudaTensor,
}
impl Output {
    pub fn stages(&self) -> [(&'static str, &CudaTensor); 18] {
        [
            ("norm1", &self.norm1),
            ("qkv", &self.qkv),
            ("q", &self.q),
            ("k", &self.k),
            ("v", &self.v),
            ("angles", &self.angles),
            ("cos", &self.cos),
            ("sin", &self.sin),
            ("q_rot", &self.q_rot),
            ("k_rot", &self.k_rot),
            ("attended", &self.attended),
            ("projected", &self.projected),
            ("residual", &self.residual),
            ("norm2", &self.norm2),
            ("fc1", &self.fc1),
            ("activated", &self.activated),
            ("fc2", &self.fc2),
            ("hidden", &self.hidden),
        ]
    }
}
impl Block {
    /// # Safety
    /// Genuine validated immutable pile prefix, including preceding partial
    /// pages, must remain unchanged through runtime teardown. A late descriptor
    /// error can retain earlier registrations. No generic heap store accepted.
    pub unsafe fn from_pile<R: BlobStoreGet>(
        snapshot: &R,
        s: Slots,
        c: Config,
        a: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        c.validate()?;
        macro_rules! bind {
            ($slot:expr,$r:literal) => {{
                let b: Blob<NativeTensor<BF16, $r>> =
                    snapshot.get($slot).map_err(|e| e.to_string())?;
                unsafe { a.bind_pile_leaf(b)? }
            }};
        }
        let b = Self {
            c,
            n1w: bind!(s.norm1_weight, 1),
            n1b: bind!(s.norm1_bias, 1),
            n2w: bind!(s.norm2_weight, 1),
            n2b: bind!(s.norm2_bias, 1),
            qw: bind!(s.qkv_weight, 2),
            qb: bind!(s.qkv_bias, 1),
            pw: bind!(s.proj_weight, 2),
            pb: bind!(s.proj_bias, 1),
            w1: bind!(s.fc1_weight, 2),
            b1: bind!(s.fc1_bias, 1),
            w2: bind!(s.fc2_weight, 2),
            b2: bind!(s.fc2_bias, 1),
        };
        let h = c.hidden;
        let m = c.intermediate;
        for (name, t, shape) in [
            ("n1w", &b.n1w, vec![h]),
            ("n1b", &b.n1b, vec![h]),
            ("n2w", &b.n2w, vec![h]),
            ("n2b", &b.n2b, vec![h]),
            ("qw", &b.qw, vec![3 * h, h]),
            ("qb", &b.qb, vec![3 * h]),
            ("pw", &b.pw, vec![h, h]),
            ("pb", &b.pb, vec![h]),
            ("w1", &b.w1, vec![m, h]),
            ("b1", &b.b1, vec![m]),
            ("w2", &b.w2, vec![h, m]),
            ("b2", &b.b2, vec![h]),
        ] {
            check(t, &b.n1w, name, &shape)?;
            if t.handle.can_mut() {
                return Err(format!("{name}: immutable alias required"));
            }
        }
        Ok(b)
    }
    pub fn forward(&self, x: &CudaTensor, grids: &[Grid]) -> Result<Output, String> {
        self.forward_inner(x, grids, false)
    }
    /// Diagnostic mutant only: omits RoPE, preserving every other operation.
    pub fn diagnostic_without_rotary(
        &self,
        x: &CudaTensor,
        grids: &[Grid],
    ) -> Result<Output, String> {
        self.forward_inner(x, grids, true)
    }
    fn forward_inner(
        &self,
        x: &CudaTensor,
        grids: &[Grid],
        no_rotary: bool,
    ) -> Result<Output, String> {
        let s = x.meta.shape().as_slice();
        let c = self.c;
        let h = c.hidden;
        let d = h / c.heads;
        if s.len() != 2 || s[0] == 0 || s[0] > 4096 {
            return Err("input requires [N,H], N1..4096".into());
        }
        let n = s[0];
        check(x, &self.n1w, "input", &[n, h])?;
        // Ephemeral metadata plan only. No host tensor values or numeric masks.
        let frames = super::vision_geometry::frames(grids, n, c.heads)?;
        let norm1 = norm(x, &self.n1w, &self.n1b, h);
        let qkv = linear(&norm1, &self.qw, &self.qb, 3 * h);
        let q = channel(&qkv, 0, h);
        let k = channel(&qkv, h, h);
        let v = channel(&qkv, 2 * h, h);
        let angles = empty(x, &[n, d]);
        let cos = empty(x, &[n, d]);
        let sin = empty(x, &[n, d]);
        let q_rot = if no_rotary {
            q.clone()
        } else {
            empty(x, &[n, h])
        };
        let k_rot = if no_rotary {
            k.clone()
        } else {
            empty(x, &[n, h])
        };
        let attended = empty(x, &[n, h]);
        for frame in &frames {
            let (offset, len, gh, gw) = (frame.offset, frame.patches, frame.height, frame.width);
            // SAFETY: all dimensions/coverage validated before any launch;
            // disjoint fresh outputs; exactly one owner per output coordinate.
            unsafe {
                rotary_table::launch_unchecked::<CudaRuntime>(
                    &x.client,
                    grid(x, len * d),
                    CubeDim::new_1d(64),
                    arg(&angles),
                    arg(&cos),
                    arg(&sin),
                    len * d,
                    offset,
                    d,
                    gh,
                    gw,
                );
                if !no_rotary {
                    rotate::launch_unchecked::<CudaRuntime>(
                        &x.client,
                        grid(x, len * h),
                        CubeDim::new_1d(64),
                        arg(&q),
                        arg(&k),
                        arg(&cos),
                        arg(&sin),
                        arg(&q_rot),
                        arg(&k_rot),
                        len * h,
                        offset,
                        h,
                        d,
                    );
                }
            }
            attention_frame(&q_rot, &k_rot, &v, &attended, offset, len, c.heads, d);
        }
        let projected = linear(&attended, &self.pw, &self.pb, h);
        let residual = element(x, &projected, false);
        let norm2 = norm(&residual, &self.n2w, &self.n2b, h);
        let fc1 = linear(&norm2, &self.w1, &self.b1, c.intermediate);
        let activated = element(&fc1, &fc1, true);
        let fc2 = linear(&activated, &self.w2, &self.b2, h);
        let hidden = element(&residual, &fc2, false);
        Ok(Output {
            norm1,
            qkv,
            q,
            k,
            v,
            angles,
            cos,
            sin,
            q_rot,
            k_rot,
            attended,
            projected,
            residual,
            norm2,
            fc1,
            activated,
            fc2,
            hidden,
        })
    }
}
fn count(s: &[usize]) -> Result<usize, String> {
    s.iter()
        .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| "empty/overflowing extent".into())
}
fn check(t: &CudaTensor, like: &CudaTensor, name: &str, s: &[usize]) -> Result<(), String> {
    if t.meta.shape().as_slice() != s
        || t.meta.strides().len() != s.len()
        || t.dtype != DType::BF16
        || t.qparams.is_some()
        || t.device != like.device
    {
        return Err(format!("{name}: shape/dtype/device/quantization"));
    }
    let n = count(s)?;
    let mut stride = 1;
    for (i, &d) in s.iter().enumerate().rev() {
        if d > 1 && t.meta.strides()[i] != stride {
            return Err(format!("{name}: noncontiguous"));
        }
        stride *= d;
    }
    if t.handle.size_in_used() < (n as u64) * 2 {
        return Err(format!("{name}: short storage"));
    }
    Ok(())
}
fn len(t: &CudaTensor) -> usize {
    t.meta.shape().as_slice().iter().product()
}
fn empty(x: &CudaTensor, s: &[usize]) -> CudaTensor {
    CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        s.into(),
        x.client.empty(s.iter().product::<usize>() * 2),
        DType::BF16,
    )
}
fn grid(x: &CudaTensor, n: usize) -> CubeCount {
    cubecl::calculate_cube_count_elemwise(&x.client, n, CubeDim::new_1d(64))
}
// Private: caller checked complete tensor geometry and frame bounds before launch.
fn attention_frame(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    out: &CudaTensor,
    offset: usize,
    tokens: usize,
    heads: usize,
    d: usize,
) {
    let probs = empty(q, &[tokens, heads, tokens]);
    unsafe {
        scores::launch_unchecked::<CudaRuntime>(
            &q.client,
            grid(q, tokens * heads * tokens),
            CubeDim::new_1d(64),
            arg(q),
            arg(k),
            arg(&probs),
            tokens * heads * tokens,
            offset,
            tokens,
            heads,
            d,
        );
        softmax::launch_unchecked::<CudaRuntime>(
            &q.client,
            grid(q, tokens * heads),
            CubeDim::new_1d(64),
            arg(&probs),
            tokens * heads,
            tokens,
        );
        values::launch_unchecked::<CudaRuntime>(
            &q.client,
            grid(q, tokens * heads * d),
            CubeDim::new_1d(64),
            arg(&probs),
            arg(v),
            arg(out),
            tokens * heads * d,
            offset,
            tokens,
            heads,
            d,
        );
    }
}
// Unsafe launch arguments remain private behind complete public validation.
unsafe fn arg(t: &CudaTensor) -> ArrayArg<CudaRuntime> {
    unsafe { ArrayArg::from_raw_parts(t.handle.clone(), len(t)) }
}
#[cube(launch_unchecked)]
fn norm_kernel(
    x: &Array<bf16>,
    w: &Array<bf16>,
    b: &Array<bf16>,
    out: &mut Array<bf16>,
    rows: usize,
    h: usize,
) {
    let r = ABSOLUTE_POS as usize;
    if r < rows {
        let mut mean = 0.0f32;
        for j in 0..h {
            mean += f32::cast_from(x[r * h + j]);
        }
        mean /= f32::cast_from(h);
        let mut variance = 0.0f32;
        for j in 0..h {
            let v = f32::cast_from(x[r * h + j]) - mean;
            variance += v * v;
        }
        let inv = 1.0f32 / (variance / f32::cast_from(h) + 0.000001f32).sqrt();
        for j in 0..h {
            out[r * h + j] = bf16::cast_from(
                (f32::cast_from(x[r * h + j]) - mean) * inv * f32::cast_from(w[j])
                    + f32::cast_from(b[j]),
            );
        }
    }
}
fn norm(x: &CudaTensor, w: &CudaTensor, b: &CudaTensor, h: usize) -> CudaTensor {
    let y = empty(x, x.meta.shape().as_slice());
    unsafe {
        norm_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            grid(x, len(x) / h),
            CubeDim::new_1d(64),
            arg(x),
            arg(w),
            arg(b),
            arg(&y),
            len(x) / h,
            h,
        );
    }
    y
}
#[cube(launch_unchecked)]
fn linear_kernel(
    x: &Array<bf16>,
    w: &Array<bf16>,
    b: &Array<bf16>,
    y: &mut Array<bf16>,
    n: usize,
    iw: usize,
    ow: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let r = i / ow;
        let o = i % ow;
        let mut sum = 0.0f32;
        for j in 0..iw {
            sum += f32::cast_from(x[r * iw + j]) * f32::cast_from(w[o * iw + j]);
        }
        y[i] = bf16::cast_from(sum + f32::cast_from(b[o]));
    }
}
fn linear(x: &CudaTensor, w: &CudaTensor, b: &CudaTensor, ow: usize) -> CudaTensor {
    let s = x.meta.shape().as_slice();
    let y = empty(x, &[s[0], ow]);
    unsafe {
        linear_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            grid(x, len(&y)),
            CubeDim::new_1d(64),
            arg(x),
            arg(w),
            arg(b),
            arg(&y),
            len(&y),
            s[1],
            ow,
        );
    }
    y
}
#[cube(launch_unchecked)]
fn channel_kernel(
    x: &Array<bf16>,
    y: &mut Array<bf16>,
    n: usize,
    iw: usize,
    start: usize,
    ow: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        y[i] = x[i / ow * iw + start + i % ow];
    }
}
fn channel(x: &CudaTensor, start: usize, ow: usize) -> CudaTensor {
    let s = x.meta.shape().as_slice();
    let y = empty(x, &[s[0], ow]);
    unsafe {
        channel_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            grid(x, len(&y)),
            CubeDim::new_1d(64),
            arg(x),
            arg(&y),
            len(&y),
            s[1],
            start,
            ow,
        );
    }
    y
}
#[cube(launch_unchecked)]
fn rotary_table(
    angle: &mut Array<bf16>,
    cos: &mut Array<bf16>,
    sin: &mut Array<bf16>,
    n: usize,
    offset: usize,
    d: usize,
    gh: usize,
    gw: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let token = i / d;
        let f = i % d % (d / 2);
        let group = token / 4;
        let inner = token % 4;
        let row = group / (gw / 2) * 2 + inner / 2;
        let col = group % (gw / 2) * 2 + inner % 2;
        let mut coordinate = row;
        if f >= d / 4 {
            coordinate = col;
        }
        let theta = f32::cast_from(10000.0f32);
        let inv = f32::cast_from(bf16::cast_from(
            1.0f32 / theta.powf(f32::cast_from(2 * (f % (d / 4))) / f32::cast_from(d / 2)),
        ));
        let pos = f32::cast_from(bf16::cast_from(f32::cast_from(coordinate)));
        let a = bf16::cast_from(pos * inv);
        let at = offset * d + i;
        angle[at] = a;
        cos[at] = bf16::cast_from(f32::cast_from(a).cos());
        sin[at] = bf16::cast_from(f32::cast_from(a).sin());
    }
}
#[cube(launch_unchecked)]
fn rotate(
    q: &Array<bf16>,
    k: &Array<bf16>,
    cos: &Array<bf16>,
    sin: &Array<bf16>,
    qr: &mut Array<bf16>,
    kr: &mut Array<bf16>,
    n: usize,
    offset: usize,
    h: usize,
    d: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let j = i % d;
        let at = offset * h + i;
        let base = at - j;
        let mut other = j + d / 2;
        let mut sign = f32::cast_from(-1.0f32);
        if j >= d / 2 {
            other = j - d / 2;
            sign = 1.0f32;
        }
        let p = (offset + i / h) * d + j;
        let c = f32::cast_from(cos[p]);
        let s = f32::cast_from(sin[p]);
        qr[at] =
            bf16::cast_from(f32::cast_from(q[at]) * c + sign * f32::cast_from(q[base + other]) * s);
        kr[at] =
            bf16::cast_from(f32::cast_from(k[at]) * c + sign * f32::cast_from(k[base + other]) * s);
    }
}
#[cube(launch_unchecked)]
fn scores(
    q: &Array<bf16>,
    k: &Array<bf16>,
    p: &mut Array<bf16>,
    cells: usize,
    offset: usize,
    tokens: usize,
    heads: usize,
    d: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < cells {
        let r = i / tokens;
        let key = i % tokens;
        let h = r % heads;
        let t = r / heads;
        let scale = 1.0f32 / f32::cast_from(d).sqrt();
        let mut dot = 0.0f32;
        for j in 0..d {
            dot += f32::cast_from(q[((offset + t) * heads + h) * d + j])
                * f32::cast_from(k[((offset + key) * heads + h) * d + j]);
        }
        // Same two BF16 materialization boundaries as the bounded block.
        p[i] = bf16::cast_from(f32::cast_from(bf16::cast_from(dot)) * scale);
    }
}
#[cube(launch_unchecked)]
fn softmax(p: &mut Array<bf16>, rows: usize, tokens: usize) {
    let r = ABSOLUTE_POS as usize;
    if r < rows {
        // One thread owns a row. Constant scratch; ascending reductions do not
        // depend on another frame, batch size, warp reduction, or atomics.
        let mut max = f32::cast_from(-3.4028235e38f32);
        for key in 0..tokens {
            let v = f32::cast_from(p[r * tokens + key]);
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for key in 0..tokens {
            sum += (f32::cast_from(p[r * tokens + key]) - max).exp();
        }
        for key in 0..tokens {
            p[r * tokens + key] =
                bf16::cast_from((f32::cast_from(p[r * tokens + key]) - max).exp() / sum);
        }
    }
}
#[cube(launch_unchecked)]
fn values(
    p: &Array<bf16>,
    v: &Array<bf16>,
    y: &mut Array<bf16>,
    n: usize,
    offset: usize,
    tokens: usize,
    heads: usize,
    d: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let r = i / d;
        let h = r % heads;
        let j = i % d;
        let mut sum = 0.0f32;
        for key in 0..tokens {
            sum += f32::cast_from(p[r * tokens + key])
                * f32::cast_from(v[((offset + key) * heads + h) * d + j]);
        }
        y[offset * heads * d + i] = bf16::cast_from(sum);
    }
}
#[cube(launch_unchecked)]
fn element_kernel(
    a: &Array<bf16>,
    b: &Array<bf16>,
    y: &mut Array<bf16>,
    n: usize,
    #[comptime] gelu: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let x = f32::cast_from(a[i]);
        if gelu {
            let inner = 0.7978845608028654f32 * (x + 0.044715f32 * x * x * x);
            y[i] = bf16::cast_from(0.5f32 * x * (1.0f32 + inner.tanh()));
        } else {
            y[i] = bf16::cast_from(x + f32::cast_from(b[i]));
        }
    }
}
fn element(a: &CudaTensor, b: &CudaTensor, gelu: bool) -> CudaTensor {
    let y = empty(a, a.meta.shape().as_slice());
    unsafe {
        element_kernel::launch_unchecked::<CudaRuntime>(
            &a.client,
            grid(a, len(a)),
            CubeDim::new_1d(64),
            arg(a),
            arg(b),
            arg(&y),
            len(a),
            gelu,
        );
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorPrimitive};
    use cubecl::cuda::CudaDevice;
    type B = burn::backend::Cuda<bf16>;

    // GPU-only synthetic arithmetic, not a CPU attention prototype. Host reads
    // below compare output bytes and never compute a reference tensor.
    #[cube(launch_unchecked)]
    fn fill(out: &mut Array<bf16>, n: usize, salt: usize) {
        let i = ABSOLUTE_POS as usize;
        if i < n {
            out[i] = bf16::cast_from((f32::cast_from((i * 17 + salt) % 251) - 125.0f32) / 128.0f32);
        }
    }

    fn fixture(rows: usize, width: usize, salt: usize) -> CudaTensor {
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let out = CudaTensor::new_contiguous(
            client.clone(),
            device,
            [rows, width].into(),
            client.empty(rows * width * 2),
            DType::BF16,
        );
        unsafe {
            fill::launch_unchecked::<CudaRuntime>(
                &client,
                grid(&out, rows * width),
                CubeDim::new_1d(64),
                arg(&out),
                rows * width,
                salt,
            );
        }
        out
    }

    fn tensor(t: &CudaTensor) -> Tensor<B, 2> {
        Tensor::from_primitive(TensorPrimitive::Float(t.clone()))
    }
    fn raw(t: Tensor<B, 2>) -> CudaTensor {
        match t.into_primitive() {
            TensorPrimitive::Float(t) => burn_cubecl::kernel::into_contiguous(t),
            TensorPrimitive::QFloat(_) => unreachable!("no quantized test tensors"),
        }
    }
    fn rows(t: &CudaTensor, start: usize, count: usize) -> CudaTensor {
        raw(tensor(t).slice([start..start + count, 0..t.meta.shape()[1]]))
    }
    fn bytes(t: &CudaTensor) -> Vec<u8> {
        t.client
            .read_one(t.handle.clone())
            .expect("GPU output read")
            .to_vec()
    }

    #[test]
    #[ignore = "requires reserved CUDA; exact isolation test, not HF/checkpoint parity"]
    fn real_width_256_patch_attention_preserves_frame_and_batch_isolation() {
        let (n, h, heads, d) = (256, 1152, 16, 72);
        let q = fixture(2 * n, h, 1);
        let k = fixture(2 * n, h, 17);
        let v = fixture(2 * n, h, 39);
        let together = empty(&q, &[2 * n, h]);
        attention_frame(&q, &k, &v, &together, 0, n, heads, d);
        attention_frame(&q, &k, &v, &together, n, n, heads, d);
        let all = bytes(&together);
        assert_eq!(all.len(), 2 * n * h * 2);
        for start in [0, n] {
            let (one_q, one_k, one_v) =
                (rows(&q, start, n), rows(&k, start, n), rows(&v, start, n));
            let one = empty(&one_q, &[n, h]);
            attention_frame(&one_q, &one_k, &one_v, &one, 0, n, heads, d);
            assert_eq!(bytes(&one), all[start * h * 2..(start + n) * h * 2]);
        }
        let reverse = |t: &CudaTensor| {
            raw(Tensor::cat(
                vec![tensor(&rows(t, n, n)), tensor(&rows(t, 0, n))],
                0,
            ))
        };
        let (rq, rk, rv) = (reverse(&q), reverse(&k), reverse(&v));
        let reordered = empty(&rq, &[2 * n, h]);
        attention_frame(&rq, &rk, &rv, &reordered, 0, n, heads, d);
        attention_frame(&rq, &rk, &rv, &reordered, n, n, heads, d);
        let result = bytes(&reordered);
        assert_eq!(&result[..n * h * 2], &all[n * h * 2..]);
        assert_eq!(&result[n * h * 2..], &all[..n * h * 2]);
        let repeat = empty(&q, &[2 * n, h]);
        attention_frame(&q, &k, &v, &repeat, 0, n, heads, d);
        attention_frame(&q, &k, &v, &repeat, n, n, heads, d);
        assert_eq!(bytes(&repeat), all);
    }

    // The old 64-element implementation is a GPU regression oracle only.
    // No production dispatch uses it and it never accepts a wider frame.
    #[cube(launch_unchecked)]
    fn previous_scores(
        q: &Array<bf16>,
        k: &Array<bf16>,
        p: &mut Array<bf16>,
        rows: usize,
        tokens: usize,
        heads: usize,
        d: usize,
    ) {
        let r = ABSOLUTE_POS as usize;
        if r < rows {
            let h = r % heads;
            let t = r / heads;
            let mut scratch = Array::<f32>::new(64usize);
            let mut max = f32::cast_from(-3.4028235e38f32);
            let scale = 1.0f32 / f32::cast_from(d).sqrt();
            for key in 0..tokens {
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += f32::cast_from(q[(t * heads + h) * d + j])
                        * f32::cast_from(k[(key * heads + h) * d + j]);
                }
                let value = f32::cast_from(bf16::cast_from(
                    f32::cast_from(bf16::cast_from(dot)) * scale,
                ));
                scratch[key] = value;
                if value > max {
                    max = value;
                }
            }
            let mut sum = 0.0f32;
            for key in 0..tokens {
                scratch[key] = (scratch[key] - max).exp();
                sum += scratch[key];
            }
            for key in 0..tokens {
                p[r * tokens + key] = bf16::cast_from(scratch[key] / sum);
            }
        }
    }

    #[test]
    #[ignore = "requires reserved CUDA; compares widened kernel with prior 64-patch arithmetic"]
    fn widened_softmax_keeps_previous_64_patch_bf16_boundaries() {
        for n in [4, 24, 64] {
            let (h, heads, d) = (1152, 16, 72);
            let q = fixture(n, h, 1);
            let k = fixture(n, h, 17);
            let old = empty(&q, &[n, heads, n]);
            let new = empty(&q, &[n, heads, n]);
            unsafe {
                previous_scores::launch_unchecked::<CudaRuntime>(
                    &q.client,
                    grid(&q, n * heads),
                    CubeDim::new_1d(64),
                    arg(&q),
                    arg(&k),
                    arg(&old),
                    n * heads,
                    n,
                    heads,
                    d,
                );
                scores::launch_unchecked::<CudaRuntime>(
                    &q.client,
                    grid(&q, n * heads * n),
                    CubeDim::new_1d(64),
                    arg(&q),
                    arg(&k),
                    arg(&new),
                    n * heads * n,
                    0,
                    n,
                    heads,
                    d,
                );
                softmax::launch_unchecked::<CudaRuntime>(
                    &q.client,
                    grid(&q, n * heads),
                    CubeDim::new_1d(64),
                    arg(&new),
                    n * heads,
                    n,
                );
            }
            assert_eq!(bytes(&new), bytes(&old), "frame patches={n}");
        }
    }
}
