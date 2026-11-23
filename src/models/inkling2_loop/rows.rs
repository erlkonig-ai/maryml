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

use crate::models::inkling::seam::{Bk, client_of, handle_of_any, tensor_of};

const PLANE: u32 = 32;

/// The most block states [`block_attention`] reads besides the partial one.
pub const MAX_BLOCKS: usize = 6;

/// Which row kernels run: all (default); `MINI_ROWS=scale` only the clip and
/// norm kernels; `MINI_ROWS=burn` none, Burn's composition everywhere. The
/// settings exist for an A/B in one binary.
fn setting() -> u8 {
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("MINI_ROWS").as_deref() {
        Ok("burn") => 0,
        Ok("scale") => 1,
        _ => 2,
    })
}

fn enabled() -> bool {
    setting() >= 1
}

fn blocks_enabled() -> bool {
    setting() >= 2
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
