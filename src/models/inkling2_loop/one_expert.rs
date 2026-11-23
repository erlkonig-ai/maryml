//! The routed experts one expert at a time, read from a weight source: a
//! second path to the output [`super::moe::Bank`] computes in one grouped call,
//! for the checks that hold the two together.
//!
//! What is its own here, and what is shared, decides what an agreement shows:
//!
//! | piece | here | shared with the grouped path |
//! |---|---|---|
//! | expert weights | each expert's own stored bytes, w13 whole (`[2I, H]`, gate and up rows still interleaved), selected by expert id | the MXFP8 and second-level scale decoders |
//! | picks and weights | [`route_from_logits`] on the host | the router GEMM whose logits it reads |
//! | rows per expert, weighted sum | `select` per expert, a sum on the host | |
//! | SwiGLU | Burn's ops | |
//! | products | the dense W4A16 and BF16 GEMMs on the dequantized activations | |
//! | activation quantizers | | [`quantize_nvfp4_46_bf16`], [`qdq_mxfp8_bf16`] and the dequantizer |
//!
//! So an agreement says the grouped path dispatches, selects and combines as
//! this one does. It says nothing about whether the router's logits or the
//! quantizers are right: the upstream comparison of the router logits carries
//! the first, the quantizers' known-answer and host-twin tests the second.

use std::collections::BTreeMap;

use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::{DType, FloatDType, Int, TensorData};

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::block::{Routing, route_from_logits};
use crate::models::inkling::burn::{PackedW, linear_bf16, linear_w4a16};
use crate::models::inkling::fp4quant::{dequantize_nvfp4, qdq_mxfp8_bf16, quantize_nvfp4_46_bf16};
use crate::models::inkling::seam::{Bk, client_of, handle_of_any, tensor_of, tensor_of_dt};
use crate::models::inkling2_loop::experts::{MXFP8_BLOCK, decode_mxfp8_row, nvfp4_scale2};
use crate::models::inkling2_loop::layout::{Dtype, ExpertMat, ExpertPart, LayerPart, Slot};
use crate::models::inkling2_loop::sliding::{Dev, Dims, Weights, f32_le, want};

/// The worst `|grouped - looped| / max|looped|` of a row: the loop sums in
/// another GEMM and quantizes its own SwiGLU output, so an E2M1 code can move.
/// A wrong expert, offset, scale or row is order one.
pub const NVFP4_BOUND: f32 = 1e-2;
/// [`NVFP4_BOUND`] for the MXFP8 layers.
pub const MXFP8_BOUND: f32 = 1e-2;

/// `x` through the trained activation quantizer and back: BF16, then 4/6
/// NVFP4 per row, as `(code * block scale, outer scale [n, 1])`, f32. The first
/// is an E2M1 times an E4M3, six significant bits, which a BF16 GEMM holds
/// exactly; the outer scale is left off it, as the grouped path leaves it off
/// its MMA, and multiplies the product.
pub(super) fn qdq46(x: Tensor<Bk, 2>) -> (Tensor<Bk, 2>, Tensor<Bk, 2>) {
    let [n, k] = x.dims();
    let client = client_of(&x);
    let dev = x.device();
    let (h, dt) = handle_of_any(x.cast(FloatDType::BF16));
    assert_eq!(dt, DType::BF16);
    let (c, s, a) = quantize_nvfp4_46_bf16(&client, &h, n, k);
    let deq = tensor_of(client.clone(), dev.clone(), dequantize_nvfp4(&client, &c, &s, n, k), n, k);
    (deq, tensor_of(client, dev, a, n, 1))
}

/// `x` through the MXFP8 activation rounding, as f32 (BF16 values), and an
/// outer scale of one.
pub(super) fn qdq_mx(x: Tensor<Bk, 2>) -> (Tensor<Bk, 2>, Tensor<Bk, 2>) {
    let [n, k] = x.dims();
    let client = client_of(&x);
    let dev = x.device();
    let (h, _) = handle_of_any(x.cast(FloatDType::BF16));
    let h = qdq_mxfp8_bf16(&client, &h, n, k);
    let y = tensor_of_dt(client, dev.clone(), h, n, k, DType::BF16).cast(FloatDType::F32);
    (y, Tensor::ones([n, 1], &dev))
}

/// Mini's softcapped SwiGLU in Burn's ops.
pub(super) fn swiglu(g: Tensor<Bk, 2>, u: Tensor<Bk, 2>, gate_cap: f32, up_cap: f32) -> Tensor<Bk, 2> {
    let capped = |v: Tensor<Bk, 2>, c: f32| v.div_scalar(c).tanh().mul_scalar(c);
    capped(g.clone(), gate_cap) * sigmoid(g) * capped(u, up_cap)
}

/// `out[row] = sum over the row's picks of weight * expert(x[row])`, the rows
/// of one expert at a time through `expert`, summed on the host.
pub(super) fn combine(
    x: &Tensor<Bk, 2>,
    picks: &[(Vec<usize>, Vec<f32>)],
    mut expert: impl FnMut(usize, Tensor<Bk, 2>) -> Tensor<Bk, 2>,
) -> Vec<f32> {
    let [rows, h] = x.dims();
    let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
    for (row, (es, ws)) in picks.iter().enumerate() {
        for (&e, &w) in es.iter().zip(ws) {
            by_expert.entry(e).or_default().push((row, w));
        }
    }
    let mut out = vec![0f32; rows * h];
    for (&e, rws) in &by_expert {
        let idx: Vec<i64> = rws.iter().map(|&(r, _)| r as i64).collect();
        let n = idx.len();
        let sel = x.clone().select(0, Tensor::<Bk, 1, Int>::from_data(TensorData::new(idx, [n]), &x.device()));
        let y: Vec<f32> = expert(e, sel).into_data().convert::<f32>().to_vec().expect("f32");
        for (j, &(r, w)) in rws.iter().enumerate() {
            for c in 0..h {
                out[r * h + c] += w * y[j * h + c];
            }
        }
    }
    out
}

/// The worst row of `max|got - want| / max|want|`.
pub(super) fn worst_row(got: &[f32], want: &[f32], h: usize) -> f32 {
    assert_eq!(got.len(), want.len());
    got.chunks(h).zip(want.chunks(h)).fold(0f32, |m, (g, w)| {
        let peak = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        let diff = g.iter().zip(w).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        // a non-finite value on either side, or a row of zeros, is no agreement
        // (f32::max skips NaN, so `diff` alone cannot see one)
        let finite = g.iter().chain(w).all(|v| v.is_finite());
        if finite && peak > 0.0 { m.max(diff / peak) } else { f32::INFINITY }
    })
}

/// A layer's experts as `w` stores them.
pub struct OneExpert<'a> {
    d: Dims,
    nvfp4: bool,
    router: Bf16W,
    bias: Vec<f32>,
    /// w13 and w2, each with its scales, and for NVFP4 its per-expert scales.
    w13: (&'a [u8], &'a [u8], Vec<f32>),
    w2: (&'a [u8], &'a [u8], Vec<f32>),
}

impl<'a> OneExpert<'a> {
    pub fn read(w: &'a dyn Weights, d: &Dims, layer: usize, dev: &Dev) -> Result<OneExpert<'a>, String> {
        let l = |p: LayerPart| Slot::Layer(layer, p);
        let client = client_of(&Tensor::<Bk, 1>::zeros([1], dev));
        let router = want(w, l(LayerPart::GateWeight), Dtype::Bf16, &[d.experts, d.hidden])?;
        let part = |m: ExpertMat, p: ExpertPart| l(LayerPart::Expert(m, p));
        let nvfp4 = w.tensor(&part(ExpertMat::W13, ExpertPart::Weight).tensor_name())?.dtype != Dtype::F8E4M3;
        let mat = |m: ExpertMat, n: usize, k: usize| -> Result<(&'a [u8], &'a [u8], Vec<f32>), String> {
            let e = d.experts;
            if nvfp4 {
                let codes = want(w, part(m, ExpertPart::Weight), Dtype::U8, &[e, n, k / 2])?;
                let scales = want(w, part(m, ExpertPart::Scale), Dtype::F8E4M3, &[e, n, k / 16])?;
                let amax = f32_le(want(w, part(m, ExpertPart::Amax), Dtype::F32, &[e])?.bytes);
                let den = f32_le(want(w, part(m, ExpertPart::OuterScaleDenominator), Dtype::F32, &[1])?.bytes);
                Ok((codes.bytes, scales.bytes, nvfp4_scale2(&amax, &den)?))
            } else {
                let vals = want(w, part(m, ExpertPart::Weight), Dtype::F8E4M3, &[e, n, k])?;
                let scales = want(w, part(m, ExpertPart::Scale), Dtype::U8, &[e, n, k / MXFP8_BLOCK])?;
                Ok((vals.bytes, scales.bytes, Vec::new()))
            }
        };
        Ok(OneExpert {
            d: *d,
            nvfp4,
            router: Bf16W { h: client.create_from_slice(router.bytes), n: d.experts, k: d.hidden, align: 16 },
            bias: f32_le(want(w, l(LayerPart::GateBias), Dtype::F32, &[d.experts])?.bytes),
            w13: mat(ExpertMat::W13, 2 * d.inter, d.hidden)?,
            w2: mat(ExpertMat::W2, d.hidden, d.inter)?,
        })
    }

    /// The bound an agreement with the grouped path is held to.
    pub fn bound(&self) -> f32 {
        if self.nvfp4 { NVFP4_BOUND } else { MXFP8_BOUND }
    }

    /// Expert `e`'s own `[n, k]` matrix times the rows of `x`.
    fn product(&self, m: &(&[u8], &[u8], Vec<f32>), e: usize, n: usize, k: usize, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let client = client_of(&x);
        let (vals, scales) = (m.0, m.1);
        if self.nvfp4 {
            let (pc, ps) = (n * k / 2, n * k / 16);
            let w = PackedW {
                codes: client.create_from_slice(&vals[e * pc..(e + 1) * pc]),
                scales: client.create_from_slice(&scales[e * ps..(e + 1) * ps]),
                n,
                k,
                scale2: m.2[e],
                swizzled: false,
            };
            linear_w4a16(x, &w)
        } else {
            let (pv, ps) = (n * k, n * (k / MXFP8_BLOCK));
            let (vals, scales) = (&vals[e * pv..(e + 1) * pv], &scales[e * ps..(e + 1) * ps]);
            let mut row = vec![0f32; k];
            let mut bytes = Vec::with_capacity(pv * 2);
            for r in 0..n {
                let per = k / MXFP8_BLOCK;
                decode_mxfp8_row(&vals[r * k..(r + 1) * k], &scales[r * per..(r + 1) * per], MXFP8_BLOCK, &mut row);
                bytes.extend(row.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()));
            }
            linear_bf16(x, &Bf16W { h: client.create_from_slice(&bytes), n, k, align: 16 })
        }
    }

    /// The experts over `x` (`[rows, H]`, the normed MLP input), and the picks.
    pub fn forward(&self, x: &Tensor<Bk, 2>) -> (Vec<f32>, Vec<Vec<usize>>) {
        let d = &self.d;
        let [rows, h] = x.dims();
        assert_eq!(h, d.hidden);
        let logits: Vec<f32> = linear_bf16(x.clone(), &self.router)
            .slice([0..rows, 0..d.experts])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .expect("f32");
        let picks: Vec<(Vec<usize>, Vec<f32>)> =
            route_from_logits(&logits, &self.bias, 1.0, d.route_scale, rows, d.experts, 0, d.top_k)
                .into_iter()
                .map(|Routing { experts, weights, .. }| (experts, weights))
                .collect();
        let qdq = |t: Tensor<Bk, 2>| if self.nvfp4 { qdq46(t) } else { qdq_mx(t) };
        let out = combine(x, &picks, |e, rows| {
            let [n, _] = rows.dims();
            let (a, alpha) = qdq(rows);
            // columns 2j and 2j + 1 are unit j's gate and up
            let gu = self.product(&self.w13, e, 2 * d.inter, h, a).slice([0..n, 0..2 * d.inter]) * alpha;
            let gu = gu.reshape([n, d.inter, 2]);
            let half = |j: usize| gu.clone().slice([0..n, 0..d.inter, j..j + 1]).reshape([n, d.inter]);
            let (s, alpha) = qdq(swiglu(half(0), half(1), d.gate_softcap, d.up_softcap));
            self.product(&self.w2, e, h, d.inter, s).slice([0..n, 0..h]) * alpha
        });
        (out, picks.into_iter().map(|(e, _)| e).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::worst_row;

    #[test]
    fn worst_row_rejects_non_finite_on_either_side() {
        // f32::max skips NaN, so a NaN in `want` must not read as zero error.
        assert_eq!(worst_row(&[1.0, 2.0], &[1.0, f32::NAN], 2), f32::INFINITY);
        assert_eq!(worst_row(&[1.0, 2.0], &[1.0, f32::INFINITY], 2), f32::INFINITY);
        assert_eq!(worst_row(&[1.0, f32::NAN], &[1.0, 2.0], 2), f32::INFINITY);
        assert_eq!(worst_row(&[1.0, f32::NEG_INFINITY], &[1.0, 2.0], 2), f32::INFINITY);
        assert_eq!(worst_row(&[1.0, 2.0], &[1.0, 2.0], 2), 0.0);
    }
}
