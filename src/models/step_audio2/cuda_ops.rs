//! Private Qwen2 CUDA operators. All model arithmetic stays on CUDA, with BF16
//! weights/activations and F32 reductions. Callers validate the bounded geometry;
//! every output is fresh, and no kernel writes an immutable pile weight alias.
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;

pub(super) type Tensor = CubeTensor<CudaRuntime>;
fn count(t: &Tensor) -> usize {
    t.meta.shape().as_slice().iter().product()
}
fn grid(t: &Tensor, n: usize) -> CubeCount {
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
    assert_eq!(count(&t), shape.iter().product());
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
    let input = weight.client.create(u32::as_bytes(ids));
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
            // Qwen2 casts normalized activations back before weighted multiply.
            let normalized = bf16::cast_from(f32::cast_from(x[row * width + j]) * inv);
            y[row * width + j] = bf16::cast_from(f32::cast_from(normalized) * f32::cast_from(w[j]));
        }
    }
}
pub(super) fn norm(x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
    let n = count(x);
    let h = x.meta.shape()[2];
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
        } else {
            let activated = bf16::cast_from(x / (1.0f32 + (-x).exp()));
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
pub(super) fn bias(a: &Tensor, b: &Tensor) -> Tensor {
    element(a, b, 1)
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
    theta: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let row = i / d;
        let j = i % d;
        let half = d / 2;
        let f = j % half;
        let position = past + row / heads;
        let angle =
            f32::cast_from(position) / theta.powf(f32::cast_from(2 * f) / f32::cast_from(d));
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
pub(super) fn rope(x: &Tensor, past: usize, theta: f32) -> Tensor {
    let n = count(x);
    let s = x.meta.shape().as_slice();
    let out = x.client.empty(n * 2);
    // SAFETY: caller validates [1,T,heads,even D], bounded positions and theta.
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
            theta,
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
    tensor(new, &[1, n / (s[2] * s[3]), s[2], s[3]], out)
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
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let row = i / total;
        let key = i % total;
        let token = row / heads;
        let kh = (row % heads) / (heads / kv);
        let mut value = -3.4028235e38f32;
        if key <= past + token {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += f32::cast_from(q[row * d + j]) * f32::cast_from(k[(key * kv + kh) * d + j]);
            }
            value = sum / f32::cast_from(d).sqrt();
        }
        out[i] = value;
    }
}
#[cube(launch_unchecked)]
fn softmax_kernel(
    scores: &Array<f32>,
    out: &mut Array<f32>,
    rows: usize,
    total: usize,
    heads: usize,
    past: usize,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let available = past + row / heads + 1;
        let mut max = -3.4028235e38f32;
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
    past: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let row = i / d;
        let j = i % d;
        let kh = (row % heads) / (heads / kv);
        let mut sum = 0.0f32;
        for key in 0..past + row / heads + 1 {
            sum += p[row * total + key] * f32::cast_from(v[(key * kv + kh) * d + j]);
        }
        out[i] = bf16::cast_from(sum);
    }
}
pub(super) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, past: usize) -> Tensor {
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
        );
        softmax_kernel::launch_unchecked::<CudaRuntime>(
            &q.client,
            grid(q, rows),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(scores, score_n),
            ArrayArg::from_raw_parts(probs.clone(), score_n),
            rows,
            total,
            heads,
            past,
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
            past,
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
fn argmax_kernel(logits: &Array<bf16>, out: &mut Array<u32>, vocab: usize) {
    if ABSOLUTE_POS == 0 {
        let mut best = -3.4028235e38f32;
        let mut index = 0u32;
        let mut invalid = false;
        for i in 0..vocab {
            let value = f32::cast_from(logits[i]);
            if value != value || value > 3.4028235e38f32 || value < -3.4028235e38f32 {
                invalid = true;
            }
            if value > best {
                best = value;
                index = i as u32;
            }
        }
        if invalid {
            out[0] = 4294967295u32;
        } else {
            out[0] = index;
        }
    }
}
pub(super) fn greedy(logits: &Tensor) -> Result<u32, String> {
    let out = logits.client.empty(4);
    // SAFETY: exactly one contiguous vocabulary row and one fresh u32 output.
    unsafe {
        argmax_kernel::launch_unchecked::<CudaRuntime>(
            &logits.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(logits.handle.clone(), count(logits)),
            ArrayArg::from_raw_parts(out.clone(), 1),
            count(logits),
        );
    }
    // This scalar transport synchronizes the stream; no logits/model arithmetic
    // moves to the host. Earliest ID wins equal finite logits.
    let bytes = logits
        .client
        .read_one(out)
        .map_err(|e| format!("CUDA token readback: {e:?}"))?;
    let word = bytes.get(..4).ok_or("short CUDA token readback")?;
    let id = u32::from_ne_bytes(word.try_into().map_err(|_| "short CUDA token readback")?);
    if id == u32::MAX {
        return Err("nonfinite CUDA logits; generation refused".into());
    }
    Ok(id)
}

/// Tiny operator invariants only, not a second model implementation or a
/// coordinate parity harness. Caller must own the ordinary CUDA reservation.
pub(super) fn smoke() -> Result<(), String> {
    use cubecl::{Runtime, cuda::CudaDevice};
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let make = |shape: &[usize], values: &[f32]| {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let values: Vec<bf16> = values.iter().copied().map(bf16::from_f32).collect();
        Tensor::new_contiguous(
            client.clone(),
            device.clone(),
            shape.into(),
            client.create(bf16::as_bytes(&values)),
            DType::BF16,
        )
    };
    let read = |t: &Tensor| -> Result<Vec<f32>, String> {
        let bytes = t
            .client
            .read_one(t.handle.clone())
            .map_err(|e| format!("smoke readback: {e:?}"))?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32())
            .collect())
    };
    let assert_values = |got: Vec<f32>, want: &[f32], what: &str| -> Result<(), String> {
        if got != want {
            return Err(format!("{what}: {got:?} != {want:?}"));
        }
        Ok(())
    };
    let x = make(&[1, 1, 2], &[1.0, -1.0]);
    let zero_weight = make(&[2], &[0.0, 0.0]);
    assert_values(
        read(&norm(&x, &zero_weight, 1e-6))?,
        &[0.0, 0.0],
        "ordinary RMS weight, not 1+w",
    )?;
    let zero = make(&[1, 1, 2], &[0.0, 0.0]);
    let b = make(&[2], &[2.0, -3.0]);
    assert_values(
        read(&bias(&zero, &b))?,
        &[2.0, -3.0],
        "required projection bias",
    )?;
    let q = make(&[1, 2, 2, 2], &[0.0; 8]);
    let k = make(&[1, 2, 1, 2], &[0.0; 4]);
    let v = make(&[1, 2, 1, 2], &[2.0, 4.0, 10.0, 20.0]);
    assert_values(
        read(&attention(&q, &k, &v, 0))?,
        &[2.0, 4.0, 2.0, 4.0, 6.0, 12.0, 6.0, 12.0],
        "causal grouped attention",
    )?;
    let head = make(&[1, 1, 1, 2], &[2.0, 4.0]);
    assert_values(
        read(&rope(&head, 0, 1e6))?,
        &[2.0, 4.0],
        "zero-position rotate-half",
    )?;
    let tail = make(&[1, 1, 1, 2], &[10.0, 20.0]);
    assert_values(
        read(&append(&tail, Some(&head)))?,
        &[2.0, 4.0, 10.0, 20.0],
        "immutable KV append",
    )?;
    let logits = make(&[1, 1, 3], &[1.0, 2.0, 2.0]);
    if greedy(&logits)? != 1 {
        return Err("greedy tie boundary failed".into());
    }
    let bad = make(&[1, 1, 2], &[0.0, f32::NAN]);
    if greedy(&bad).is_ok() {
        return Err("nonfinite logits were accepted".into());
    }
    Ok(())
}
