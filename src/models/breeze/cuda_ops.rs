//! B1 Breeze CUDA operators: BF16 weights/activations, F32 reductions.
//! Every output is fresh; immutable pile aliases are never writable destinations.
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;

pub(super) type Tensor = CubeTensor<CudaRuntime>;
pub(super) fn count(t: &Tensor) -> usize {
    t.meta.shape().as_slice().iter().product()
}
pub(super) fn grid(t: &Tensor, n: usize) -> CubeCount {
    cubecl::calculate_cube_count_elemwise(&t.client, n, CubeDim::new_1d(64))
}
fn tensor(t: &Tensor, shape: &[usize], handle: cubecl::server::Handle) -> Tensor {
    CubeTensor::new_contiguous(
        t.client.clone(),
        t.device.clone(),
        shape.into(),
        handle,
        DType::BF16,
    )
}
pub(super) fn reshape(t: Tensor, shape: &[usize]) -> Tensor {
    assert_eq!(count(&t), shape.iter().product::<usize>());
    CubeTensor::new_contiguous(t.client, t.device, shape.into(), t.handle, t.dtype)
}

#[cube(launch_unchecked)]
fn gather_kernel(
    weight: &Array<bf16>,
    ids: &Array<u32>,
    out: &mut Array<bf16>,
    n: usize,
    width: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        out[i] = weight[(ids[i / width] as usize) * width + i % width];
    }
}
pub(super) fn gather(weight: &Tensor, ids: &[u32]) -> Tensor {
    let h = weight.meta.shape()[1];
    let n = ids.len() * h;
    let input = weight.client.create_from_slice(u32::as_bytes(ids));
    let out = weight.client.empty(n * 2);
    // SAFETY: caller checks all IDs against vocabulary and bounded [T,H].
    unsafe {
        gather_kernel::launch_unchecked::<CudaRuntime>(
            &weight.client,
            grid(weight, n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(weight.handle.clone(), count(weight)),
            ArrayArg::from_raw_parts(input, ids.len()),
            ArrayArg::from_raw_parts(out.clone(), n),
            n,
            h,
        );
    }
    tensor(weight, &[1, ids.len(), h], out)
}

#[cube(launch_unchecked)]
fn norm_kernel(
    x: &Array<bf16>,
    w: &Array<bf16>,
    y: &mut Array<bf16>,
    rows: usize,
    width: usize,
    eps: f32,
    #[comptime] gemma: bool,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let mut sum = 0.0f32;
        for j in 0..width {
            let v = f32::cast_from(x[row * width + j]);
            sum += v * v;
        }
        let inv = 1.0f32 / (sum / f32::cast_from(width) + eps).sqrt();
        for j in 0..width {
            let value = f32::cast_from(x[row * width + j]) * inv;
            if gemma {
                y[row * width + j] = bf16::cast_from(value * (1.0f32 + f32::cast_from(w[j])));
            } else {
                let normalized = bf16::cast_from(value);
                y[row * width + j] =
                    bf16::cast_from(f32::cast_from(normalized) * f32::cast_from(w[j]));
            }
        }
    }
}
pub(super) fn norm(x: &Tensor, w: &Tensor, eps: f32, gemma: bool) -> Tensor {
    let n = count(x);
    let h = *x.meta.shape().as_slice().last().unwrap();
    let out = x.client.empty(n * 2);
    // SAFETY: private model provides contiguous [1,T,H] and [H]; fresh output.
    unsafe {
        norm_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            grid(x, n / h),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(x.handle.clone(), n),
            ArrayArg::from_raw_parts(w.handle.clone(), h),
            ArrayArg::from_raw_parts(out.clone(), n),
            n / h,
            h,
            eps,
            gemma,
        );
    }
    tensor(x, x.meta.shape().as_slice(), out)
}

#[cube(launch_unchecked)]
fn element_kernel(
    a: &Array<bf16>,
    b: &Array<bf16>,
    out: &mut Array<bf16>,
    n: usize,
    width: usize,
    #[comptime] mode: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let x = f32::cast_from(a[i]);
        if mode == 0 {
            out[i] = bf16::cast_from(x + f32::cast_from(b[i]));
        } else if mode == 1 {
            out[i] = bf16::cast_from(x + f32::cast_from(b[i % width]));
        } else if mode == 2 {
            let activated = bf16::cast_from(x / (1.0f32 + (-x).exp()));
            out[i] = bf16::cast_from(f32::cast_from(activated) * f32::cast_from(b[i]));
        } else {
            let z = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
            let activated = bf16::cast_from(0.5f32 * x * (1.0f32 + z.tanh()));
            out[i] = bf16::cast_from(f32::cast_from(activated) * f32::cast_from(b[i]));
        }
    }
}
fn element(a: &Tensor, b: &Tensor, mode: u32) -> Tensor {
    let n = count(a);
    let out = a.client.empty(n * 2);
    let width = *a.meta.shape().as_slice().last().unwrap();
    // SAFETY: caller supplies equal arrays, or a width-sized bias for mode1.
    unsafe {
        element_kernel::launch_unchecked::<CudaRuntime>(
            &a.client,
            grid(a, n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(a.handle.clone(), n),
            ArrayArg::from_raw_parts(b.handle.clone(), count(b)),
            ArrayArg::from_raw_parts(out.clone(), n),
            n,
            width,
            mode,
        );
    }
    tensor(a, a.meta.shape().as_slice(), out)
}
pub(super) fn add(a: &Tensor, b: &Tensor) -> Tensor {
    element(a, b, 0)
}
pub(super) fn geglu(a: &Tensor, b: &Tensor) -> Tensor {
    element(a, b, 3)
}
pub(super) fn swiglu(gate: &Tensor, up: &Tensor) -> Tensor {
    element(gate, up, 2)
}

#[cube(launch_unchecked)]
fn rope_kernel(
    x: &Array<bf16>,
    out: &mut Array<bf16>,
    n: usize,
    heads: usize,
    d: usize,
    past: usize,
    inv_freq: &Array<f32>,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let row = i / d;
        let j = i % d;
        let half = d / 2;
        let f = j % half;
        let position = past + row / heads;
        let angle = f32::cast_from(position) * inv_freq[f];
        let cos = f32::cast_from(bf16::cast_from(angle.cos()));
        let sin = f32::cast_from(bf16::cast_from(angle.sin()));
        let low = f32::cast_from(x[row * d + f]);
        let high = f32::cast_from(x[row * d + half + f]);
        if j < half {
            out[i] = bf16::cast_from(
                f32::cast_from(bf16::cast_from(low * cos))
                    - f32::cast_from(bf16::cast_from(high * sin)),
            );
        } else {
            out[i] = bf16::cast_from(
                f32::cast_from(bf16::cast_from(high * cos))
                    + f32::cast_from(bf16::cast_from(low * sin)),
            );
        }
    }
}
pub(super) fn rope(x: &Tensor, past: usize, frequencies: &[f32]) -> Tensor {
    let n = count(x);
    let s = x.meta.shape().as_slice();
    let out = x.client.empty(n * 2);
    assert_eq!(frequencies.len(), s[3] / 2);
    let freq = x.client.create_from_slice(f32::as_bytes(frequencies));
    // SAFETY: bounded B1 head geometry; frequency count exactly D/2.
    unsafe {
        rope_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            grid(x, n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(x.handle.clone(), n),
            ArrayArg::from_raw_parts(out.clone(), n),
            n,
            s[2],
            s[3],
            past,
            ArrayArg::from_raw_parts(freq, frequencies.len()),
        );
    }
    tensor(x, s, out)
}

#[cube(launch_unchecked)]
fn append_kernel(
    old: &Array<bf16>,
    new: &Array<bf16>,
    out: &mut Array<bf16>,
    n: usize,
    old_n: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        if i < old_n {
            out[i] = old[i];
        } else {
            out[i] = new[i - old_n];
        }
    }
}
pub(super) fn append(new: &Tensor, old: Option<&Tensor>) -> Tensor {
    let old_n = old.map_or(0, count);
    let n = old_n + count(new);
    let out = new.client.empty(n * 2);
    let old = old.unwrap_or(new);
    let s = new.meta.shape().as_slice();
    // SAFETY: B1 contiguous caches, old and new lengths bounded by capacity;
    // no old read when absent. Fresh output, no in-place alias mutation.
    unsafe {
        append_kernel::launch_unchecked::<CudaRuntime>(
            &new.client,
            grid(new, n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(old.handle.clone(), count(old)),
            ArrayArg::from_raw_parts(new.handle.clone(), count(new)),
            ArrayArg::from_raw_parts(out.clone(), n),
            n,
            old_n,
        );
    }
    let mut shape = s.to_vec();
    let width: usize = s[2..].iter().product();
    shape[1] = n / width;
    tensor(new, &shape, out)
}

#[cube(launch_unchecked)]
fn score_kernel(
    q: &Array<bf16>,
    k: &Array<bf16>,
    out: &mut Array<f32>,
    n: usize,
    heads: usize,
    kv: usize,
    d: usize,
    total: usize,
    past: usize,
    scale: f32,
    left: usize,
    right: usize,
    #[comptime] causal: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let row = i / total;
        let key = i % total;
        let token = row / heads;
        let kh = (row % heads) / (heads / kv);
        let mut value = f32::cast_from(-3.4028235e38f32);
        let position = past + token;
        if (causal && key <= position)
            || (!causal && key + left >= position && key <= position + right)
        {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += f32::cast_from(q[row * d + j]) * f32::cast_from(k[(key * kv + kh) * d + j]);
            }
            value = sum * scale;
        }
        out[i] = value;
    }
}
#[cube(launch_unchecked)]
fn softmax_kernel(scores: &Array<f32>, out: &mut Array<f32>, rows: usize, total: usize) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let available = total;
        let mut max = f32::cast_from(-3.4028235e38f32);
        for key in 0..available {
            let value = scores[row * total + key];
            if value > max {
                max = value;
            }
        }
        let mut sum = 0.0f32;
        for key in 0..available {
            sum += (scores[row * total + key] - max).exp();
        }
        for key in 0..total {
            if key < available {
                out[row * total + key] = (scores[row * total + key] - max).exp() / sum;
            } else {
                out[row * total + key] = 0.0f32;
            }
        }
    }
}
#[cube(launch_unchecked)]
fn value_kernel(
    p: &Array<f32>,
    v: &Array<bf16>,
    out: &mut Array<bf16>,
    n: usize,
    heads: usize,
    kv: usize,
    d: usize,
    total: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let row = i / d;
        let j = i % d;
        let kh = (row % heads) / (heads / kv);
        let mut sum = 0.0f32;
        for key in 0..total {
            sum += p[row * total + key] * f32::cast_from(v[(key * kv + kh) * d + j]);
        }
        out[i] = bf16::cast_from(sum);
    }
}
pub(super) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past: usize,
    scale: f32,
    causal: bool,
    window: Option<usize>,
) -> Tensor {
    let s = q.meta.shape().as_slice();
    let heads = s[2];
    let d = s[3];
    let total = k.meta.shape()[1];
    let kv = k.meta.shape()[2];
    let rows = s[1] * heads;
    let n = count(q);
    let score_n = rows * total;
    let scores = q.client.empty(score_n * 4);
    let probs = q.client.empty(score_n * 4);
    let out = q.client.empty(n * 2);
    // SAFETY: private B1 runtime validates grouped heads, cache length, causal
    // offset and all u32 extents before dispatch. F32 scratch is not weights.
    unsafe {
        score_kernel::launch_unchecked::<CudaRuntime>(
            &q.client,
            grid(q, score_n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(q.handle.clone(), n),
            ArrayArg::from_raw_parts(k.handle.clone(), count(k)),
            ArrayArg::from_raw_parts(scores.clone(), score_n),
            score_n,
            heads,
            kv,
            d,
            total,
            past,
            scale,
            window.map_or(total, |w| (w - 1) / 2),
            window.map_or(total, |w| w / 2),
            causal,
        );
        softmax_kernel::launch_unchecked::<CudaRuntime>(
            &q.client,
            grid(q, rows),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(scores, score_n),
            ArrayArg::from_raw_parts(probs.clone(), score_n),
            rows,
            total,
        );
        value_kernel::launch_unchecked::<CudaRuntime>(
            &q.client,
            grid(q, n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(probs, score_n),
            ArrayArg::from_raw_parts(v.handle.clone(), count(v)),
            ArrayArg::from_raw_parts(out.clone(), n),
            n,
            heads,
            kv,
            d,
            total,
        );
    }
    tensor(q, s, out)
}

#[cube(launch_unchecked)]
fn last_kernel(x: &Array<bf16>, out: &mut Array<bf16>, width: usize, offset: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < width {
        out[i] = x[offset + i];
    }
}
pub(super) fn last(x: &Tensor) -> Tensor {
    let h = x.meta.shape()[2];
    let n = count(x);
    let out = x.client.empty(h * 2);
    // SAFETY: nonempty B1 input, final row fits within checked contiguous extent.
    unsafe {
        last_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            grid(x, h),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(x.handle.clone(), n),
            ArrayArg::from_raw_parts(out.clone(), h),
            h,
            n - h,
        );
    }
    tensor(x, &[1, 1, h], out)
}

#[cube(launch_unchecked)]
fn text_embedding_kernel(
    w: &Array<bf16>,
    eoi: &Array<bf16>,
    ids: &Array<u32>,
    out: &mut Array<bf16>,
    n: usize,
    h: usize,
    eoi_id: u32,
    scale: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let id = ids[i / h];
        let value = w[(id as usize) * h + i % h];
        out[i] = bf16::cast_from(f32::cast_from(value) * scale);
        if id == eoi_id {
            out[i] = eoi[i % h];
        }
    }
}
pub(super) fn text_embedding(w: &Tensor, eoi: &Tensor, ids: &[u32], eoi_id: u32) -> Tensor {
    let h = w.meta.shape()[1];
    let n = ids.len() * h;
    let tokens = w.client.create_from_slice(u32::as_bytes(ids));
    let out = w.client.empty(n * 2);
    // Upstream creates the sqrt(H) multiplier in activation dtype.
    let scale = bf16::from_f32((h as f32).sqrt()).to_f32();
    unsafe {
        text_embedding_kernel::launch_unchecked::<CudaRuntime>(
            &w.client,
            grid(w, n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(w.handle.clone(), count(w)),
            ArrayArg::from_raw_parts(eoi.handle.clone(), h),
            ArrayArg::from_raw_parts(tokens, ids.len()),
            ArrayArg::from_raw_parts(out.clone(), n),
            n,
            h,
            eoi_id,
            scale,
        );
    }
    tensor(w, &[1, ids.len(), h], out)
}

#[cube(launch_unchecked)]
fn audio_embedding_kernel(
    w: &Array<bf16>,
    codes: &Array<u32>,
    out: &mut Array<bf16>,
    n: usize,
    h: usize,
    vocab: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let frame = i / h;
        let mut sum = 0.0f32;
        for book in 0..16usize {
            let id = book * vocab + codes[frame * 16 + book] as usize;
            sum += f32::cast_from(w[id * h + i % h]);
        }
        out[i] = bf16::cast_from(sum);
    }
}
pub(super) fn audio_embedding(w: &Tensor, frames: &[[u16; 16]], vocab: usize) -> Tensor {
    let h = w.meta.shape()[1];
    let codes: Vec<u32> = frames.iter().flatten().map(|&x| u32::from(x)).collect();
    let input = w.client.create_from_slice(u32::as_bytes(&codes));
    let n = frames.len() * h;
    let out = w.client.empty(n * 2);
    unsafe {
        audio_embedding_kernel::launch_unchecked::<CudaRuntime>(
            &w.client,
            grid(w, n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(w.handle.clone(), count(w)),
            ArrayArg::from_raw_parts(input, codes.len()),
            ArrayArg::from_raw_parts(out.clone(), n),
            n,
            h,
            vocab,
        );
    }
    tensor(w, &[1, frames.len(), h], out)
}

#[cube(launch_unchecked)]
fn head_kernel(
    x: &Array<bf16>,
    w: &Array<bf16>,
    out: &mut Array<f32>,
    h: usize,
    vocab: usize,
    offset: usize,
    #[comptime] transposed: bool,
) {
    let o = ABSOLUTE_POS as usize;
    if o < vocab {
        let mut sum = 0.0f32;
        for j in 0..h {
            let mut wi = offset + o * h + j;
            if transposed {
                wi = offset + j * vocab + o;
            }
            sum += f32::cast_from(x[j]) * f32::cast_from(w[wi]);
        }
        out[o] = sum;
    }
}
/// Output heads use F32 products/reduction without materializing a whole F32
/// copy of immutable BF16 weights. Depth storage is [head,input,output].
pub(super) fn head(x: &Tensor, w: &Tensor, depth_index: Option<usize>) -> Tensor {
    let h = count(x);
    let vocab = if depth_index.is_some() {
        w.meta.shape()[2]
    } else {
        w.meta.shape()[0]
    };
    let offset = depth_index.unwrap_or(0) * h * vocab;
    let out = x.client.empty(vocab * 4);
    unsafe {
        head_kernel::launch_unchecked::<CudaRuntime>(
            &x.client,
            grid(x, vocab),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(x.handle.clone(), h),
            ArrayArg::from_raw_parts(w.handle.clone(), count(w)),
            ArrayArg::from_raw_parts(out.clone(), vocab),
            h,
            vocab,
            offset,
            depth_index.is_some(),
        );
    }
    CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        [1, 1, vocab].into(),
        out,
        DType::F32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::breeze::{generator::GenerationOptions, sampling::Sampler};
    use cubecl::{Runtime, cuda::CudaDevice};
    #[test]
    #[ignore = "actual CUDA device 0 requires ordinary Stars lock"]
    fn cuda_breeze_operator_boundaries() {
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let make = |shape: &[usize], values: &[f32]| {
            assert_eq!(shape.iter().product::<usize>(), values.len());
            let values: Vec<bf16> = values.iter().copied().map(bf16::from_f32).collect();
            Tensor::new_contiguous(
                client.clone(),
                device.clone(),
                shape.into(),
                client.create_from_slice(bf16::as_bytes(&values)),
                DType::BF16,
            )
        };
        let read = |t: &Tensor| {
            let bytes = t.client.read_one(t.handle.clone()).unwrap();
            bytes
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32())
                .collect::<Vec<_>>()
        };
        let x = make(&[1, 1, 2], &[1.0, -1.0]);
        let w = make(&[2], &[0.0, 0.0]);
        assert_eq!(read(&norm(&x, &w, 1e-6, false)), vec![0.0, 0.0]);
        assert_eq!(read(&norm(&x, &w, 1e-6, true)), vec![1.0, -1.0]);
        let q = make(&[1, 3, 2, 2], &[0.0; 12]);
        let k = make(&[1, 3, 1, 2], &[0.0; 6]);
        let v = make(&[1, 3, 1, 2], &[2.0, 4.0, 8.0, 16.0, 14.0, 28.0]);
        assert_eq!(
            read(&attention(&q, &k, &v, 0, 1.0, false, None)),
            vec![
                8.0, 16.0, 8.0, 16.0, 8.0, 16.0, 8.0, 16.0, 8.0, 16.0, 8.0, 16.0
            ]
        );
        assert_eq!(
            read(&attention(&q, &k, &v, 0, 1.0, true, None)),
            vec![
                2.0, 4.0, 2.0, 4.0, 5.0, 10.0, 5.0, 10.0, 8.0, 16.0, 8.0, 16.0
            ]
        );
        // Window2 admits self+one FUTURE, not one past+self.
        assert_eq!(
            read(&attention(&q, &k, &v, 0, 1.0, false, Some(2))),
            vec![
                5.0, 10.0, 5.0, 10.0, 11.0, 22.0, 11.0, 22.0, 14.0, 28.0, 14.0, 28.0
            ]
        );
        let embed = make(&[2, 4], &[1.0; 8]);
        let eoi = make(&[4], &[3.0; 4]);
        assert_eq!(
            read(&text_embedding(&embed, &eoi, &[0, 1], 1)),
            vec![2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
        let a = make(&[1, 1, 2], &[2.0, 3.0]);
        let heads = make(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let logits = head(&a, &heads, Some(0));
        let bytes = logits.client.read_one(logits.handle.clone()).unwrap();
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![14.0, 19.0, 24.0]);
        let mut values = vec![-10.0f32; 2052];
        values[2048] = 100.0;
        values[2051] = 9.0;
        values[7] = 8.0;
        let logits = Tensor::new_contiguous(
            client.clone(),
            device.clone(),
            [1, 1, 2052].into(),
            client.create_from_slice(f32::as_bytes(&values)),
            DType::F32,
        );
        let mut o = GenerationOptions::default();
        o.do_sample = false;
        assert_eq!(
            Sampler::new(42).sample(&logits, &o, &[], true).unwrap(),
            2051
        );
        values.truncate(2051);
        let logits = Tensor::new_contiguous(
            client.clone(),
            device.clone(),
            [1, 1, 2051].into(),
            client.create_from_slice(f32::as_bytes(&values)),
            DType::F32,
        );
        assert_eq!(Sampler::new(42).sample(&logits, &o, &[], false).unwrap(), 7);
        o.do_sample = true;
        o.top_k = 1;
        assert_eq!(Sampler::new(42).sample(&logits, &o, &[], false).unwrap(), 7);
    }
}
