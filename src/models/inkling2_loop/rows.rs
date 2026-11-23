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
//! Only f32 contiguous inputs take these kernels; anything else keeps the
//! Burn path (`MINI_ROWS=burn` forces it, for an A/B in one binary).

use std::sync::OnceLock;

use burn::tensor::Tensor;
use cubecl::prelude::*;
use half::bf16;

use crate::models::inkling::seam::{Bk, client_of, handle_of_any, tensor_of};

const PLANE: u32 = 32;

/// The most block states [`block_attention`] reads besides the partial one.
pub const MAX_BLOCKS: usize = 6;

/// Which row kernels run, cumulatively: `MINI_ROWS=burn` none (Burn's
/// composition everywhere); `scale` the clip and norm kernels; `gates` also
/// the message gates' input, rounding and output; anything else (the
/// default) also the block attention and the message read. The settings
/// exist for an A/B in one binary.
fn setting() -> u8 {
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("MINI_ROWS").as_deref() {
        Ok("burn") => 0,
        Ok("scale") => 1,
        Ok("gates") => 2,
        _ => 3,
    })
}

fn enabled() -> bool {
    setting() >= 1
}

fn gates_enabled() -> bool {
    setting() >= 2
}

fn blocks_enabled() -> bool {
    setting() >= 3
}

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

/// [`super::sliding::l2_clip`] over the last axis, one launch.
pub fn l2_clip(x: Tensor<Bk, 3>, thr: f32) -> Option<Tensor<Bk, 3>> {
    if !enabled() || !is_f32(&x) {
        return None;
    }
    let [a, b, d] = x.dims();
    Some(launch_scale(x.reshape([a * b, d]), None, thr, 0).reshape([a, b, d]))
}

/// [`super::sliding::unit_rms`] over the last axis, one launch.
pub fn unit_rms(x: Tensor<Bk, 3>, eps: f64) -> Option<Tensor<Bk, 3>> {
    if !enabled() || !is_f32(&x) {
        return None;
    }
    let [a, b, d] = x.dims();
    Some(launch_scale(x.reshape([a * b, d]), None, eps as f32, 1).reshape([a, b, d]))
}

/// `rms_norm` with a gain, one launch.
pub fn rms_norm(x: Tensor<Bk, 2>, gain: Tensor<Bk, 1>, eps: f64) -> Option<Tensor<Bk, 2>> {
    if !enabled() || !is_f32(&x) || !is_f32(&gain) {
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

/// Attention from one learned query over the unit-RMS-normed
/// `[b0.., partial]` states of a row, whose values are the same normed
/// states. `n` block arrays are read; the rest are bound to `p` and unused.
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
    if r < rows {
        let l = UNIT_POS_PLANE;
        let base = r * d;
        let steps = comptime!(d.div_ceil(32));
        let scale = comptime!(1.0f32 / (d as f32).sqrt());
        let mut w = Array::<f32>::new(comptime!(MAX_BLOCKS + 1));
        let mut logit = Array::<f32>::new(comptime!(MAX_BLOCKS + 1));
        let mut mx = f32::new(-3.0e38);
        for s in 0..n + 1u32 {
            let mut ss = 0.0f32;
            let mut dot = 0.0f32;
            for t in 0..steps {
                let j = l + t * 32u32;
                if j < d {
                    let v = pick(b0, b1, b2, b3, b4, b5, p, s, n, (base + j) as usize);
                    ss += v * v;
                    dot += v * q[j as usize];
                }
            }
            let iv = 1.0f32 / Sqrt::sqrt(plane_sum(ss) / f32::cast_from(d) + 1e-6f32);
            let lg = plane_sum(dot) * iv * scale;
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
            let j = l + t * 32u32;
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
    if !blocks_enabled() || blocks.len() > MAX_BLOCKS || !is_f32(&partial) || !is_f32(&query) || blocks.iter().any(|b| !is_f32(b)) {
        return None;
    }
    let [rows, d] = partial.dims();
    assert!(blocks.iter().all(|b| b.dims() == [rows, d]), "block attention: every block is [{rows}, {d}]");
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
                CubeDim::new_1d(PLANE),
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
    if !gates_enabled() || !is_f32(&hidden) || !is_f32(&message) {
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
    if !gates_enabled() || !is_f32(&x) {
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
    if !gates_enabled() || !is_f32(&fg) || !is_f32(&hidden) || !is_f32(&message) {
        return None;
    }
    let [rows, h] = hidden.dims();
    assert_eq!(fg.dims(), [rows, 2 * h], "gate output: fg is [t, 2H]");
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
    if !blocks_enabled() || keys > MAX_BLOCKS + 2 || !is_f32(normed) || !is_f32(&queries) {
        return None;
    }
    assert_eq!(queries.dims(), [1, 4, h], "message queries are [1, 4, H]");
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
