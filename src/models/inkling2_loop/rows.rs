//! Row-local kernels for Mini's small per-row chains: one launch each where
//! the Burn composition launched one kernel per elementwise or reduction op.
//!
//! A decode step runs about forty-five layer visits, and each visit's clips,
//! norms and block attention were five to sixteen Burn launches apiece over a
//! few kilobytes. The work is tiny; the launches were the cost (the
//! 2026-11-23 profile: about 10,900 kernels per decode token, the GPU 54%
//! busy). Each kernel here is one plane per row: a strided pass for the row's
//! reduction(s), `plane_sum`, then a strided pass that writes. No stage waits
//! on another cube, so this changes launch count and grid, not the number of
//! dependent device stages.
//!
//! The arithmetic is the Burn composition's, in the same order per element;
//! the reductions are a plane tree instead of Burn's, so the last bits differ.
//! Each entry point takes f32 inputs (a strided view is made contiguous first)
//! within its shape limits and returns `None` otherwise; the caller then runs
//! the Burn composition.

use burn::tensor::Tensor;
use cubecl::prelude::*;
use half::bf16;

use crate::models::inkling::seam::{Bk, client_of, handle_of_any, tensor_of};

const PLANE: u32 = 32;

/// The most block states [`block_attention`] reads besides the partial one.
pub const MAX_BLOCKS: usize = 6;

/// `x * min(thr / (|x| + 1e-6), 1)` per row (`mode` 0); `x / sqrt(mean(x^2) + eps)`
/// (1); the same times `gain` (2).
#[cube(launch_unchecked)]
fn row_scale(
    x: &Array<f32>,
    gain: &Array<f32>,
    out: &mut Array<f32>,
    rows: u32,
    param: f32,
    #[comptime] d: u32,
    #[comptime] mode: u32,
) {
    let r = CUBE_POS_X;
    if r < rows {
        let l = UNIT_POS_PLANE;
        let base = r * d;
        let steps = comptime!(d.div_ceil(32));
        let mut ss = 0.0f32;
        for t in 0..steps {
            let j = l + t * 32u32;
            if j < d {
                let v = x[(base + j) as usize];
                ss += v * v;
            }
        }
        let total = plane_sum(ss);
        let mut f = 1.0f32;
        if comptime!(mode == 0) {
            let norm = Sqrt::sqrt(total);
            f = param / (norm + 1e-6f32);
            if f > 1.0f32 {
                f = 1.0f32;
            }
        } else {
            f = 1.0f32 / Sqrt::sqrt(total / f32::cast_from(d) + param);
        }
        for t in 0..steps {
            let j = l + t * 32u32;
            if j < d {
                let i = (base + j) as usize;
                if comptime!(mode == 2) {
                    out[i] = x[i] * f * gain[j as usize];
                } else {
                    out[i] = x[i] * f;
                }
            }
        }
    }
}

fn f32_handle<const D: usize>(x: Tensor<Bk, D>) -> Option<cubecl::server::Handle> {
    let (h, dt) = handle_of_any(x);
    (dt == burn::tensor::DType::F32).then_some(h)
}

fn launch_scale(x: Tensor<Bk, 2>, gain: Option<Tensor<Bk, 1>>, param: f32, mode: u32) -> Tensor<Bk, 2> {
    let [rows, d] = x.dims();
    let client = client_of(&x);
    let dev = x.device();
    let xh = f32_handle(x).expect("row kernels take f32");
    let gh = match gain {
        Some(g) => f32_handle(g).expect("row kernels take an f32 gain"),
        None => xh.clone(),
    };
    let out = client.empty(rows * d * 4);
    if rows > 0 {
        unsafe {
            row_scale::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::new_1d(rows as u32),
                CubeDim::new_1d(PLANE),
                ArrayArg::from_raw_parts(xh, rows * d),
                ArrayArg::from_raw_parts(gh, if mode == 2 { d } else { rows * d }),
                ArrayArg::from_raw_parts(out.clone(), rows * d),
                rows as u32,
                param,
                d as u32,
                mode,
            );
        }
    }
    tensor_of(client, dev, out, rows, d)
}

fn is_f32<const D: usize>(x: &Tensor<Bk, D>) -> bool {
    x.dtype() == burn::tensor::DType::F32
}

// The entry points' operand contracts, checked on the dimensions before
// anything reaches the device: the kernels index raw arrays, so a mismatch
// would read or write past an operand instead of failing. A mismatch is a
// caller's bug and panics; a dtype or size the kernels do not cover is not,
// and returns `None` for the Burn path.

fn check_gain(x: [usize; 2], gain: [usize; 1]) {
    assert_eq!(gain[0], x[1], "row kernels: rms_norm gain is {gain:?}, x is {x:?}");
}

fn check_block_attention(query: [usize; 1], blocks: &[[usize; 2]], partial: [usize; 2]) {
    assert_eq!(query[0], partial[1], "row kernels: block attention query is {query:?}, partial is {partial:?}");
    for b in blocks {
        assert_eq!(*b, partial, "row kernels: block attention block is {b:?}, partial is {partial:?}");
    }
}

fn check_gate_input(hidden: [usize; 2], message: [usize; 2]) {
    assert_eq!(message, hidden, "row kernels: gate message is {message:?}, hidden is {hidden:?}");
}

fn check_gate_output(fg: [usize; 2], hidden: [usize; 2], message: [usize; 2]) {
    assert_eq!(message, hidden, "row kernels: gate message is {message:?}, hidden is {hidden:?}");
    assert_eq!(fg, [hidden[0], 2 * hidden[1]], "row kernels: gate fg is {fg:?}, hidden is {hidden:?}");
}

fn check_read(normed: [usize; 3], queries: [usize; 3]) {
    assert_eq!(queries, [1, 4, normed[2]], "row kernels: message queries are {queries:?}, normed is {normed:?}");
}

fn check_devices(devs: &[Dev]) {
    assert!(devs.windows(2).all(|w| w[0] == w[1]), "row kernels: operands on different devices: {devs:?}");
}

type Dev = burn::backend::cuda::CudaDevice;

/// [`super::sliding::l2_clip`] over the last axis, one launch.
pub fn l2_clip(x: Tensor<Bk, 3>, thr: f32) -> Option<Tensor<Bk, 3>> {
    if !is_f32(&x) {
        return None;
    }
    let [a, b, d] = x.dims();
    Some(launch_scale(x.reshape([a * b, d]), None, thr, 0).reshape([a, b, d]))
}

/// [`super::sliding::unit_rms`] over the last axis, one launch.
pub fn unit_rms(x: Tensor<Bk, 3>, eps: f64) -> Option<Tensor<Bk, 3>> {
    if !is_f32(&x) {
        return None;
    }
    let [a, b, d] = x.dims();
    Some(launch_scale(x.reshape([a * b, d]), None, eps as f32, 1).reshape([a, b, d]))
}

/// `rms_norm` with a gain, one launch.
pub fn rms_norm(x: Tensor<Bk, 2>, gain: Tensor<Bk, 1>, eps: f64) -> Option<Tensor<Bk, 2>> {
    check_gain(x.dims(), gain.dims());
    check_devices(&[x.device(), gain.device()]);
    if !is_f32(&x) || !is_f32(&gain) {
        return None;
    }
    Some(launch_scale(x, Some(gain), eps as f32, 2))
}

/// State `s` of a row's `[b0.., p]` at element `i`: block `s` below `n`, the
/// partial state at `n`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn pick(
    b0: &Array<f32>,
    b1: &Array<f32>,
    b2: &Array<f32>,
    b3: &Array<f32>,
    b4: &Array<f32>,
    b5: &Array<f32>,
    p: &Array<f32>,
    s: u32,
    n: u32,
    i: usize,
) -> f32 {
    let mut v = 0.0f32;
    if s == n {
        v = p[i];
    } else if s == 0u32 {
        v = b0[i];
    } else if s == 1u32 {
        v = b1[i];
    } else if s == 2u32 {
        v = b2[i];
    } else if s == 3u32 {
        v = b3[i];
    } else if s == 4u32 {
        v = b4[i];
    } else {
        v = b5[i];
    }
    v
}

/// Planes per row in [`block_attend`]: one plane walking seven 2304-wide
/// states left a decode row latency-bound (about 96 us a launch).
const BA_PLANES: u32 = 8;

/// Attention from one learned query over the unit-RMS-normed
/// `[b0.., partial]` states of a row, whose values are the same normed
/// states. `n` block arrays are read; the rest are bound to `p` and unused.
/// One cube of [`BA_PLANES`] planes per row: each plane reduces its strided
/// share, the planes' partial sums meet in shared memory.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn block_attend(
    b0: &Array<f32>,
    b1: &Array<f32>,
    b2: &Array<f32>,
    b3: &Array<f32>,
    b4: &Array<f32>,
    b5: &Array<f32>,
    p: &Array<f32>,
    q: &Array<f32>,
    out: &mut Array<f32>,
    rows: u32,
    n: u32,
    #[comptime] d: u32,
) {
    let r = CUBE_POS_X;
    let l = UNIT_POS_X;
    let plane = UNIT_POS_Y;
    let tid = plane * 32u32 + l;
    let threads = comptime!(32 * BA_PLANES);
    let steps = comptime!(d.div_ceil(32 * BA_PLANES));
    let scale = comptime!(1.0f32 / (d as f32).sqrt());
    let slots = comptime!((MAX_BLOCKS as u32 + 1) * BA_PLANES);
    let mut part_ss = SharedMemory::<f32>::new(slots as usize);
    let mut part_dot = SharedMemory::<f32>::new(slots as usize);
    let live = r < rows;
    let base = r * d;
    for s in 0..n + 1u32 {
        let mut ss = 0.0f32;
        let mut dot = 0.0f32;
        if live {
            for t in 0..steps {
                let j = tid + t * threads;
                if j < d {
                    let v = pick(b0, b1, b2, b3, b4, b5, p, s, n, (base + j) as usize);
                    ss += v * v;
                    dot += v * q[j as usize];
                }
            }
        }
        let a = plane_sum(ss);
        let b = plane_sum(dot);
        if l == 0u32 {
            part_ss[(s * BA_PLANES + plane) as usize] = a;
            part_dot[(s * BA_PLANES + plane) as usize] = b;
        }
    }
    sync_cube();
    if live {
        let mut w = Array::<f32>::new(comptime!(MAX_BLOCKS + 1));
        let mut logit = Array::<f32>::new(comptime!(MAX_BLOCKS + 1));
        let mut mx = f32::new(-3.0e38);
        for s in 0..n + 1u32 {
            let mut ss = 0.0f32;
            let mut dot = 0.0f32;
            #[unroll]
            for k in 0..BA_PLANES {
                ss += part_ss[(s * BA_PLANES + k) as usize];
                dot += part_dot[(s * BA_PLANES + k) as usize];
            }
            let iv = 1.0f32 / Sqrt::sqrt(ss / f32::cast_from(d) + 1e-6f32);
            let lg = dot * iv * scale;
            w[s as usize] = iv;
            logit[s as usize] = lg;
            if lg > mx {
                mx = lg;
            }
        }
        let mut sum = 0.0f32;
        for s in 0..n + 1u32 {
            let e = Exp::exp(logit[s as usize] - mx);
            logit[s as usize] = e;
            sum += e;
        }
        for s in 0..n + 1u32 {
            w[s as usize] = w[s as usize] * (logit[s as usize] / sum);
        }
        for t in 0..steps {
            let j = tid + t * threads;
            if j < d {
                let i = (base + j) as usize;
                let mut acc = 0.0f32;
                for s in 0..n + 1u32 {
                    acc += pick(b0, b1, b2, b3, b4, b5, p, s, n, i) * w[s as usize];
                }
                out[i] = acc;
            }
        }
    }
}

/// [`super::sliding::block_attention`] in one launch, with no concatenation:
/// each block state is read where it is.
pub fn block_attention(query: Tensor<Bk, 1>, blocks: &[Tensor<Bk, 2>], partial: Tensor<Bk, 2>) -> Option<Tensor<Bk, 2>> {
    let dims: Vec<[usize; 2]> = blocks.iter().map(|b| b.dims()).collect();
    check_block_attention(query.dims(), &dims, partial.dims());
    let mut devs: Vec<Dev> = blocks.iter().map(|b| b.device()).collect();
    devs.extend([query.device(), partial.device()]);
    check_devices(&devs);
    if blocks.len() > MAX_BLOCKS || !is_f32(&partial) || !is_f32(&query) || blocks.iter().any(|b| !is_f32(b)) {
        return None;
    }
    let [rows, d] = partial.dims();
    let client = client_of(&partial);
    let dev = partial.device();
    let n = blocks.len();
    let ph = f32_handle(partial).expect("f32");
    let qh = f32_handle(query).expect("f32");
    let bh: Vec<_> = (0..MAX_BLOCKS)
        .map(|i| match blocks.get(i) {
            Some(b) => f32_handle(b.clone()).expect("f32"),
            None => ph.clone(),
        })
        .collect();
    let out = client.empty(rows * d * 4);
    let len = rows * d;
    if rows > 0 {
        unsafe {
            block_attend::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::new_1d(rows as u32),
                CubeDim::new_2d(PLANE, BA_PLANES),
                ArrayArg::from_raw_parts(bh[0].clone(), len),
                ArrayArg::from_raw_parts(bh[1].clone(), len),
                ArrayArg::from_raw_parts(bh[2].clone(), len),
                ArrayArg::from_raw_parts(bh[3].clone(), len),
                ArrayArg::from_raw_parts(bh[4].clone(), len),
                ArrayArg::from_raw_parts(bh[5].clone(), len),
                ArrayArg::from_raw_parts(ph, len),
                ArrayArg::from_raw_parts(qh, d),
                ArrayArg::from_raw_parts(out.clone(), len),
                rows as u32,
                n as u32,
                d as u32,
            );
        }
    }
    Some(tensor_of(client, dev, out, rows, d))
}

/// A message gate's input: `[rms(hidden), rms(message)]` per row, `[rows, 2H]`,
/// rounded to bf16 (kept as f32) when `round`.
#[cube(launch_unchecked)]
fn gate_in(
    hs: &Array<f32>,
    ms: &Array<f32>,
    out: &mut Array<f32>,
    rows: u32,
    eps: f32,
    #[comptime] h: u32,
    #[comptime] round: bool,
) {
    let r = CUBE_POS_X;
    if r < rows {
        let l = UNIT_POS_PLANE;
        let steps = comptime!(h.div_ceil(32));
        let mut sh = 0.0f32;
        let mut sm = 0.0f32;
        for t in 0..steps {
            let j = l + t * 32u32;
            if j < h {
                let a = hs[(r * h + j) as usize];
                let b = ms[(r * h + j) as usize];
                sh += a * a;
                sm += b * b;
            }
        }
        let ih = 1.0f32 / Sqrt::sqrt(plane_sum(sh) / f32::cast_from(h) + eps);
        let im = 1.0f32 / Sqrt::sqrt(plane_sum(sm) / f32::cast_from(h) + eps);
        for t in 0..steps {
            let j = l + t * 32u32;
            if j < h {
                let mut a = hs[(r * h + j) as usize] * ih;
                let mut b = ms[(r * h + j) as usize] * im;
                if comptime!(round) {
                    a = f32::cast_from(bf16::cast_from(a));
                    b = f32::cast_from(bf16::cast_from(b));
                }
                out[(r * 2u32 * h + j) as usize] = a;
                out[(r * 2u32 * h + h + j) as usize] = b;
            }
        }
    }
}

/// Every value rounded to the nearest bf16, kept as f32.
#[cube(launch_unchecked)]
fn round_bf16(x: &Array<f32>, out: &mut Array<f32>, total: u32) {
    let p = ABSOLUTE_POS as u32;
    if p < total {
        out[p as usize] = f32::cast_from(bf16::cast_from(x[p as usize]));
    }
}

/// `(1 - tanh(f)) * hidden + tanh(g) * message` with `fg` `[rows, 2H]`
/// holding `f` then `g` per row.
#[cube(launch_unchecked)]
fn gate_out(fg: &Array<f32>, hs: &Array<f32>, ms: &Array<f32>, out: &mut Array<f32>, total: u32, #[comptime] h: u32) {
    let p = ABSOLUTE_POS as u32;
    if p < total {
        let r = p / h;
        let c = p % h;
        let f = fg[(r * 2u32 * h + c) as usize];
        let g = fg[(r * 2u32 * h + h + c) as usize];
        out[p as usize] = (1.0f32 - Tanh::tanh(f)) * hs[p as usize] + Tanh::tanh(g) * ms[p as usize];
    }
}

const ELEMS: u32 = 256;

/// [`super::messages::DeviceGate::apply`]'s input: `[rms(hidden), rms(message)]`,
/// `[t, 2H]`, bf16-rounded when `round`. One launch.
pub fn gate_input(hidden: Tensor<Bk, 2>, message: Tensor<Bk, 2>, eps: f64, round: bool) -> Option<Tensor<Bk, 2>> {
    check_gate_input(hidden.dims(), message.dims());
    check_devices(&[hidden.device(), message.device()]);
    if !is_f32(&hidden) || !is_f32(&message) {
        return None;
    }
    let [rows, h] = hidden.dims();
    let client = client_of(&hidden);
    let dev = hidden.device();
    let (hh, mh) = (f32_handle(hidden).expect("f32"), f32_handle(message).expect("f32"));
    let out = client.empty(rows * 2 * h * 4);
    if rows > 0 {
        unsafe {
            gate_in::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::new_1d(rows as u32),
                CubeDim::new_1d(PLANE),
                ArrayArg::from_raw_parts(hh, rows * h),
                ArrayArg::from_raw_parts(mh, rows * h),
                ArrayArg::from_raw_parts(out.clone(), rows * 2 * h),
                rows as u32,
                eps as f32,
                h as u32,
                round,
            );
        }
    }
    Some(tensor_of(client, dev, out, rows, 2 * h))
}

/// Round to bf16, kept as f32, in one launch.
pub fn round_to_bf16(x: Tensor<Bk, 2>) -> Option<Tensor<Bk, 2>> {
    if !is_f32(&x) {
        return None;
    }
    let [rows, c] = x.dims();
    let total = rows * c;
    let client = client_of(&x);
    let dev = x.device();
    let xh = f32_handle(x).expect("f32");
    let out = client.empty(total * 4);
    if total > 0 {
        unsafe {
            round_bf16::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::new_1d((total as u32).div_ceil(ELEMS)),
                CubeDim::new_1d(ELEMS),
                ArrayArg::from_raw_parts(xh, total),
                ArrayArg::from_raw_parts(out.clone(), total),
                total as u32,
            );
        }
    }
    Some(tensor_of(client, dev, out, rows, c))
}

/// [`super::messages::DeviceGate::apply`]'s output from the stacked forget
/// and gain pre-activations `fg` (`[t, 2H]`), in one launch.
pub fn gate_output(fg: Tensor<Bk, 2>, hidden: Tensor<Bk, 2>, message: Tensor<Bk, 2>) -> Option<Tensor<Bk, 2>> {
    check_gate_output(fg.dims(), hidden.dims(), message.dims());
    check_devices(&[fg.device(), hidden.device(), message.device()]);
    if !is_f32(&fg) || !is_f32(&hidden) || !is_f32(&message) {
        return None;
    }
    let [rows, h] = hidden.dims();
    let total = rows * h;
    let client = client_of(&hidden);
    let dev = hidden.device();
    let (fh, hh, mh) = (f32_handle(fg).expect("f32"), f32_handle(hidden).expect("f32"), f32_handle(message).expect("f32"));
    let out = client.empty(total * 4);
    if total > 0 {
        unsafe {
            gate_out::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::new_1d((total as u32).div_ceil(ELEMS)),
                CubeDim::new_1d(ELEMS),
                ArrayArg::from_raw_parts(fh, 2 * total),
                ArrayArg::from_raw_parts(hh, total),
                ArrayArg::from_raw_parts(mh, total),
                ArrayArg::from_raw_parts(out.clone(), total),
                total as u32,
                h as u32,
            );
        }
    }
    Some(tensor_of(client, dev, out, rows, h))
}

/// One layer's four messages from the normed sources `[t, keys, H]` and its
/// queries `[4, H]`: per (row, stream), a softmax over the keys of the scaled
/// dot products, then the weighted sum of the keys. One cube per (row, stream).
#[cube(launch_unchecked)]
fn msg_read(normed: &Array<f32>, q: &Array<f32>, out: &mut Array<f32>, cells: u32, keys: u32, #[comptime] h: u32) {
    let cell = CUBE_POS_X;
    if cell < cells {
        let l = UNIT_POS_PLANE;
        let r = cell / 4u32;
        let s = cell % 4u32;
        let steps = comptime!(h.div_ceil(32));
        let scale = comptime!(1.0f32 / (h as f32).sqrt());
        let mut logit = Array::<f32>::new(comptime!(MAX_BLOCKS + 2));
        let mut mx = f32::new(-3.0e38);
        for k in 0..keys {
            let mut dot = 0.0f32;
            for t in 0..steps {
                let j = l + t * 32u32;
                if j < h {
                    dot += normed[((r * keys + k) * h + j) as usize] * q[(s * h + j) as usize];
                }
            }
            let lg = plane_sum(dot) * scale;
            logit[k as usize] = lg;
            if lg > mx {
                mx = lg;
            }
        }
        let mut sum = 0.0f32;
        for k in 0..keys {
            let e = Exp::exp(logit[k as usize] - mx);
            logit[k as usize] = e;
            sum += e;
        }
        for t in 0..steps {
            let j = l + t * 32u32;
            if j < h {
                let mut acc = 0.0f32;
                for k in 0..keys {
                    acc += (logit[k as usize] / sum) * normed[((r * keys + k) * h + j) as usize];
                }
                out[(cell * h + j) as usize] = acc;
            }
        }
    }
}

/// [`super::messages::MessageSources::read`] in one launch: `normed`
/// `[t, keys, H]`, `queries` `[1, 4, H]`, the messages `[t, 4, H]`.
pub fn read_messages(normed: &Tensor<Bk, 3>, queries: Tensor<Bk, 3>) -> Option<Tensor<Bk, 3>> {
    let [t, keys, h] = normed.dims();
    check_read(normed.dims(), queries.dims());
    check_devices(&[normed.device(), queries.device()]);
    if keys > MAX_BLOCKS + 2 || !is_f32(normed) || !is_f32(&queries) {
        return None;
    }
    let client = client_of(normed);
    let dev = normed.device();
    let nh = f32_handle(normed.clone()).expect("f32");
    let qh = f32_handle(queries).expect("f32");
    let cells = t * 4;
    let out = client.empty(cells * h * 4);
    if cells > 0 {
        unsafe {
            msg_read::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::new_1d(cells as u32),
                CubeDim::new_1d(PLANE),
                ArrayArg::from_raw_parts(nh, t * keys * h),
                ArrayArg::from_raw_parts(qh, 4 * h),
                ArrayArg::from_raw_parts(out.clone(), cells * h),
                cells as u32,
                keys as u32,
                h as u32,
            );
        }
    }
    Some(crate::models::inkling::seam::tensor_of3(client, dev, out, t, 4, h))
}

/// One short-convolution step over the window `[hist; x]` without building
/// it: `out[i] = scale * (x_i + sum_j w[d, j] * window[i + j])` per channel,
/// taps in ascending `j` as [`crate::models::inkling::sconv`]'s batched
/// kernel, and `next` the window's last `keep` rows, the history the next call
/// reads. `keep` is the kernel's width less one.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn conv_step_kernel(
    hist: &Array<f32>,
    x: &Array<f32>,
    w: &Array<f32>,
    out: &mut Array<f32>,
    next: &mut Array<f32>,
    rows: u32,
    span: u32,
    scale: f32,
    #[comptime] dim: u32,
    #[comptime] kernel: u32,
) {
    let p = ABSOLUTE_POS as u32;
    if p < span {
        let i = p / dim;
        let d = p % dim;
        let keep = comptime!(kernel - 1);
        if i < rows {
            let xv = x[p as usize];
            let mut acc = 0.0f32;
            #[unroll]
            for j in 0..keep {
                let a = i + j;
                let mut v = 0.0f32;
                if a < keep {
                    v = hist[(a * dim + d) as usize];
                } else {
                    v = x[((a - keep) * dim + d) as usize];
                }
                // from zero: the first fused multiply-add rounds the first
                // product once, as the batched kernel's plain first product does
                acc += w[(d * kernel + j) as usize] * v;
            }
            acc += w[(d * kernel + keep) as usize] * xv;
            out[p as usize] = (xv + acc) * scale;
        }
        if i < keep {
            let a = rows + i;
            let mut v = 0.0f32;
            if a < keep {
                v = hist[(a * dim + d) as usize];
            } else {
                v = x[((a - keep) * dim + d) as usize];
            }
            next[p as usize] = v;
        }
    }
}

/// [`super::sliding::conv_step`] in one launch: the output `[t, c]`, and
/// `hist` replaced by the new history. `w` is `[c, kernel]`.
pub fn conv_step(hist: &mut Tensor<Bk, 2>, x: Tensor<Bk, 2>, w: &Tensor<Bk, 2>, scale: f32) -> Option<Tensor<Bk, 2>> {
    let [t, c] = x.dims();
    let [keep, hc] = hist.dims();
    let [wc, kernel] = w.dims();
    check_devices(&[x.device(), hist.device(), w.device()]);
    if !is_f32(&x) || !is_f32(hist) || !is_f32(w) || t == 0 || hc != c || wc != c || kernel != keep + 1 || keep == 0 {
        return None;
    }
    let client = client_of(&x);
    let dev = x.device();
    let (hh, xh, wh) = (f32_handle(hist.clone()).expect("f32"), f32_handle(x).expect("f32"), f32_handle(w.clone()).expect("f32"));
    let out = client.empty(t * c * 4);
    let next = client.empty(keep * c * 4);
    let span = t.max(keep) * c;
    unsafe {
        conv_step_kernel::launch_unchecked::<cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::new_1d((span as u32).div_ceil(ELEMS)),
            CubeDim::new_1d(ELEMS),
            ArrayArg::from_raw_parts(hh, keep * c),
            ArrayArg::from_raw_parts(xh, t * c),
            ArrayArg::from_raw_parts(wh, c * kernel),
            ArrayArg::from_raw_parts(out.clone(), t * c),
            ArrayArg::from_raw_parts(next.clone(), keep * c),
            t as u32,
            span as u32,
            scale,
            c as u32,
            kernel as u32,
        );
    }
    *hist = tensor_of(client.clone(), dev.clone(), next, keep, c);
    Some(tensor_of(client, dev, out, t, c))
}

/// Outputs per cube in [`gemv_rows`], one plane each.
const GEMV_PLANES: u32 = 8;

/// `out[t, j] = sum_k x[t, k] * w[j, k]`, one plane per output, `w`
/// `[n, k]` row-major (the layout the transposed Burn weight views).
#[cube(launch_unchecked)]
fn gemv_rows(x: &Array<f32>, w: &Array<f32>, out: &mut Array<f32>, cells: u32, #[comptime] n: u32, #[comptime] k: u32) {
    let cell = CUBE_POS_X * GEMV_PLANES + UNIT_POS_Y;
    if cell < cells {
        let l = UNIT_POS_X;
        let t = cell / n;
        let j = cell % n;
        let steps = comptime!(k.div_ceil(32));
        let mut acc = 0.0f32;
        for s in 0..steps {
            let i = l + s * 32u32;
            if i < k {
                acc += x[(t * k + i) as usize] * w[(j * k + i) as usize];
            }
        }
        let total = plane_sum(acc);
        if l == 0u32 {
            out[cell as usize] = total;
        }
    }
}

/// The most rows [`gemv`] takes; longer calls keep Burn's matmul.
pub const GEMV_MAX_ROWS: usize = 16;

/// `x @ w_t` for a short call, `x` `[t, k]`, `w_t` `[k, n]` the transpose of
/// a contiguous `[n, k]` weight. f32 products and an f32 plane-tree sum,
/// where Burn's f32 matmul on this runtime is TF32 with an f32 accumulator;
/// on the released gates every operand is a bf16 value, so the products are
/// exact either way and only the summation order differs. Callers whose
/// operands are full f32 keep the matmul: TF32 and f32 products differ there.
pub fn gemv(x: Tensor<Bk, 2>, w_t: &Tensor<Bk, 2>) -> Option<Tensor<Bk, 2>> {
    let [t, k] = x.dims();
    let [wk, n] = w_t.dims();
    check_devices(&[x.device(), w_t.device()]);
    if t == 0 || t > GEMV_MAX_ROWS || wk != k || !is_f32(&x) || !is_f32(w_t) {
        return None;
    }
    let w = w_t.clone().transpose();
    let wc: burn_cubecl::tensor::CubeTensor<cubecl::cuda::CudaRuntime> = match w.into_primitive() {
        burn::tensor::TensorPrimitive::Float(c) => c,
        _ => return None,
    };
    if !wc.is_contiguous() {
        return None;
    }
    let client = client_of(&x);
    let dev = x.device();
    let xh = f32_handle(x).expect("f32");
    let cells = t * n;
    let out = client.empty(cells * 4);
    unsafe {
        gemv_rows::launch_unchecked::<cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::new_1d((cells as u32).div_ceil(GEMV_PLANES)),
            CubeDim::new_2d(PLANE, GEMV_PLANES),
            ArrayArg::from_raw_parts(xh, t * k),
            ArrayArg::from_raw_parts(wc.handle, n * k),
            ArrayArg::from_raw_parts(out.clone(), cells),
            cells as u32,
            n as u32,
            k as u32,
        );
    }
    Some(tensor_of(client, dev, out, t, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The contracts alone, with no device: each mismatch panics in the check,
    // which every entry point runs before it touches a client.

    #[test]
    #[should_panic(expected = "rms_norm gain")]
    fn a_short_gain_is_refused() {
        check_gain([3, 2304], [2303]);
    }

    #[test]
    #[should_panic(expected = "block attention query")]
    fn a_narrow_block_query_is_refused() {
        check_block_attention([64], &[[3, 2304]], [3, 2304]);
    }

    #[test]
    #[should_panic(expected = "block attention block")]
    fn a_mismatched_block_is_refused() {
        check_block_attention([2304], &[[3, 2304], [2, 2304]], [3, 2304]);
    }

    #[test]
    #[should_panic(expected = "gate message")]
    fn a_gate_input_message_of_other_rows_is_refused() {
        check_gate_input([3, 2304], [1, 2304]);
    }

    #[test]
    #[should_panic(expected = "gate message")]
    fn a_gate_output_message_of_other_width_is_refused() {
        check_gate_output([3, 4608], [3, 2304], [3, 2048]);
    }

    #[test]
    #[should_panic(expected = "gate fg")]
    fn a_gate_output_fg_of_other_width_is_refused() {
        check_gate_output([3, 2304], [3, 2304], [3, 2304]);
    }

    #[test]
    #[should_panic(expected = "message queries")]
    fn message_queries_of_other_width_are_refused() {
        check_read([3, 8, 2304], [1, 4, 2048]);
    }

    #[test]
    fn matching_shapes_pass_every_check() {
        check_gain([3, 2304], [2304]);
        check_block_attention([2304], &[[3, 2304]; 6], [3, 2304]);
        check_gate_input([3, 2304], [3, 2304]);
        check_gate_output([3, 4608], [3, 2304], [3, 2304]);
        check_read([3, 8, 2304], [1, 4, 2304]);
    }

    /// Through the entry points on the device: each mismatch panics with its
    /// check's message (so before the launch code), and the client still
    /// runs a well-formed call afterwards.
    #[test]
    fn the_entry_points_refuse_mismatches_before_launching() {
        let dev = Dev::default();
        let t2 = |r: usize, c: usize| Tensor::<Bk, 2>::ones([r, c], &dev);
        let t1 = |c: usize| Tensor::<Bk, 1>::ones([c], &dev);
        let refused = |what: &str, f: &mut dyn FnMut()| {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err(what);
            let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
            assert!(msg.contains("row kernels:"), "{what}: panicked elsewhere: {msg}");
        };
        refused("rms_norm", &mut || {
            rms_norm(t2(3, 64), t1(63), 1e-6);
        });
        refused("block_attention", &mut || {
            block_attention(t1(32), &[t2(3, 64)], t2(3, 64));
        });
        refused("gate_input", &mut || {
            gate_input(t2(3, 64), t2(2, 64), 1e-6, true);
        });
        refused("gate_output", &mut || {
            gate_output(t2(3, 128), t2(3, 64), t2(3, 32));
        });
        refused("read_messages", &mut || {
            read_messages(&Tensor::<Bk, 3>::ones([3, 4, 64], &dev), Tensor::<Bk, 3>::ones([1, 4, 32], &dev));
        });
        let y = rms_norm(t2(3, 64), t1(64), 1e-6).expect("f32 takes the kernel");
        let v = y.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(v.len(), 3 * 64);
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-5), "rms_norm of ones is ones: {:?}", &v[..4]);
    }
}
