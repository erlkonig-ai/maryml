//! Inkling 2 Mini's vision tower and projector on the device: the processor's
//! image tokens in, one projected feature per image token out, `text_hidden`
//! wide, ready to stand in for the image placeholders in the text stream.
//!
//! Read off `Inkling2VisionModel.forward` at transformers 046bdb8
//! (`models/inkling2/modeling_inkling2.py`), with the release's
//! `vision_config` (b85c3cfa). Per image:
//!
//! ```text
//! e      = patch_embedding(native rows)                 [P, H], no bias
//!        + bilinear(position_embedding, native grid)    the 32x32 table resampled
//! seq    = [class_embedding + cls_position_embedding; e] the class token in front
//! seq    = layernorm_pre(seq)
//! for each of the 23 layers (pre-norm, layer scale):
//!   seq += lambda_1 * out_proj(attn(rope(q), rope(k), v))  over this image only
//!   seq += lambda_2 * fc2(gelu(fc1(layernorm_after(seq))))
//! feats  = proj_out(gelu(proj_in(seq[1..] as [P / 4, 4H])))  the class row is dropped
//! ```
//!
//! Attention is non-causal and block-diagonal over images (upstream's
//! `cu_seqlens`), so each image runs as its own sequence and the results are
//! concatenated. The rotary embedding is axial over `(h, w)`: the class token
//! at `(0, 0)`, each native patch at its 1-indexed `(row + 1, col + 1)`; the
//! first half of the head's channel pairs rotate by the width position and the
//! second half by the height, interleaved pairs `(x1, x2) -> (-x2, x1)`, theta
//! 10000 (upstream's default; `vision_config` names none).
//!
//! Precision is the inkling2_loop lineage's: the BF16 weights stay BF16 on the
//! device and every projection is [`linear_bf16`] (activation rounded to BF16,
//! f32 accumulate), everything else (norms, rotary, softmax, residuals, layer
//! scale, biases, the position resample) in f32. That sits between upstream's
//! f32 and bf16 runs; the `inkling2_vision_check` binary measures where.
//!
//! Failure. [`VisionTower::bind`] refuses a configuration or a stored tensor
//! it does not read, and [`VisionTower::forward`] refuses an image whose
//! geometry is not the tower's, whose values do not fill its grid, or whose
//! native patch count exceeds [`MAX_NATIVE_PATCHES`]; nothing at either call
//! boundary panics on malformed input. Attention scores are tiled by queries
//! so one image's scratch is at most [`SCORE_BUDGET`] f32 values.

use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::activation::{gelu, softmax};

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::burn::linear_bf16;
use crate::models::inkling::seam::{Bk, client_of};
use crate::models::inkling2_loop::config::Inkling2VisionConfig;
use crate::models::inkling2_loop::layout::{Dtype, Slot, VisionLayerPart, VisionPart, VisionProj};
use crate::models::inkling2_loop::sliding::{Dev, Weights, bf16_to_f32, want};
use crate::models::inkling2_loop::vision_input::{Geometry, ImageTokens, posemb_taps};

/// Upstream's rotary base: `Inkling2VisionConfig` declares an axial rope with
/// no theta, so `RotaryEmbeddingConfigMixin.default_theta` applies.
pub const ROPE_THETA: f32 = 10_000.0;

/// The most native patches one image may bring (a 2560 x 2560 pixel image).
/// Above it the image is refused rather than allocated.
pub const MAX_NATIVE_PATCHES: usize = 16_384;

/// Attention scores one query tile may hold, `heads * tile * sequence` f32s
/// (256 MiB).
pub const SCORE_BUDGET: usize = 1 << 26;

/// The tower's shape, from `vision_config`, checked once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionDims {
    pub geometry: Geometry,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mlp: usize,
    pub layers: usize,
    pub eps: f64,
    /// Side of the learned position table.
    pub posemb_side: usize,
    pub proj_hidden: usize,
    pub text_hidden: usize,
}

impl VisionDims {
    pub fn of(c: &Inkling2VisionConfig) -> Result<VisionDims, String> {
        let geometry = Geometry::of(c)?;
        if c.hidden_size == 0
            || c.num_attention_heads == 0
            || c.hidden_size % c.num_attention_heads != 0
        {
            return Err(format!(
                "vision_config: hidden_size {} is not a positive multiple of num_attention_heads {}",
                c.hidden_size, c.num_attention_heads
            ));
        }
        let head_dim = c.hidden_size / c.num_attention_heads;
        // axial: half the channel pairs per axis
        if head_dim % 4 != 0 {
            return Err(format!(
                "vision head_dim {head_dim} is not a multiple of 4 (axial rotary)"
            ));
        }
        let mlp = c.hidden_size as f64 * c.mlp_ratio;
        if !(mlp.is_finite() && mlp >= 1.0 && mlp.fract() == 0.0) {
            return Err(format!(
                "vision mlp_ratio {} gives no whole width",
                c.mlp_ratio
            ));
        }
        if c.projector_bias {
            return Err(
                "vision projector_bias: the release has none and this tower reads none".into(),
            );
        }
        if c.num_hidden_layers == 0 || c.posemb_grid_size == 0 || !(c.layer_norm_eps > 0.0) {
            return Err(
                "vision_config: no layers, no position table, or a non-positive eps".into(),
            );
        }
        Ok(VisionDims {
            geometry,
            hidden: c.hidden_size,
            heads: c.num_attention_heads,
            head_dim,
            mlp: mlp as usize,
            layers: c.num_hidden_layers,
            eps: c.layer_norm_eps,
            posemb_side: c.posemb_grid_size,
            proj_hidden: c.projection_hidden_dim,
            text_hidden: c.text_hidden_size,
        })
    }

    fn merged(&self) -> usize {
        self.geometry.merge * self.geometry.merge
    }
}

/// `nn.Linear` on the stored BF16, with its bias in f32 when it has one.
struct Linear {
    w: Bf16W,
    b: Option<Tensor<Bk, 2>>,
}

impl Linear {
    fn forward(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let y = linear_bf16(x, &self.w);
        match &self.b {
            Some(b) => y + b.clone(),
            None => y,
        }
    }
}

/// `nn.LayerNorm(eps)` in f32: biased variance over the row.
fn layer_norm(x: Tensor<Bk, 2>, w: &Tensor<Bk, 2>, b: &Tensor<Bk, 2>, eps: f64) -> Tensor<Bk, 2> {
    let mean = x.clone().mean_dim(1);
    let xc = x - mean;
    let var = xc.clone().powi_scalar(2).mean_dim(1);
    xc / var.add_scalar(eps).sqrt() * w.clone() + b.clone()
}

struct Layer {
    ln1: (Tensor<Bk, 2>, Tensor<Bk, 2>),
    ln2: (Tensor<Bk, 2>, Tensor<Bk, 2>),
    ls1: Tensor<Bk, 2>,
    ls2: Tensor<Bk, 2>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    fc1: Linear,
    fc2: Linear,
}

/// The vision tower and projector, bound once.
pub struct VisionTower {
    d: VisionDims,
    dev: Dev,
    patch: Bf16W,
    /// The position table on the host, `[side * side, hidden]` f32: four rows
    /// per patch are read, which is no work for a device.
    posemb: Vec<f32>,
    /// `class_embedding + cls_position_embedding`, `[1, hidden]`.
    cls: Tensor<Bk, 2>,
    ln_pre: (Tensor<Bk, 2>, Tensor<Bk, 2>),
    layers: Vec<Layer>,
    proj_in: Bf16W,
    proj_out: Bf16W,
    inv_freq: Vec<f32>,
}

/// One image's projected features: `tokens` rows of `text_hidden` f32.
#[derive(Debug, Clone)]
pub struct ImageFeatures {
    pub tokens: usize,
    pub width: usize,
    pub values: Vec<f32>,
}

impl VisionTower {
    /// Bind the tower from `w` on `dev`. Every tensor's dtype and shape is
    /// checked against the layout before anything is uploaded.
    pub fn bind(
        w: &dyn Weights,
        cfg: &Inkling2VisionConfig,
        dev: &Dev,
    ) -> Result<VisionTower, String> {
        let d = VisionDims::of(cfg)?;
        let probe = Tensor::<Bk, 1>::zeros([1], dev);
        let client = client_of(&probe);
        let row = |s: Slot, n: usize| -> Result<Vec<f32>, String> {
            Ok(bf16_to_f32(want(w, s, Dtype::Bf16, &[n])?.bytes))
        };
        let dev_row = |s: Slot, n: usize| -> Result<Tensor<Bk, 2>, String> {
            Ok(Tensor::from_data(TensorData::new(row(s, n)?, [1, n]), dev))
        };
        let mat = |s: Slot, n: usize, k: usize| -> Result<Bf16W, String> {
            let st = want(w, s, Dtype::Bf16, &[n, k])?;
            if !Bf16W::tileable(n, k) {
                return Err(format!("{}: {n}x{k} does not tile", s.tensor_name()));
            }
            Ok(Bf16W {
                h: client.create_from_slice(st.bytes),
                n,
                k,
                align: 16,
            })
        };
        let v = |p: VisionPart| Slot::Vision(p);
        let (h, side) = (d.hidden, d.posemb_side);
        let lin = |i: usize, p: VisionProj| -> Result<Linear, String> {
            let l = |x| v(VisionPart::Layer(i, VisionLayerPart::Attn(p, x)));
            Ok(Linear {
                w: mat(l(false), h, h)?,
                b: Some(dev_row(l(true), h)?),
            })
        };
        let mut layers = Vec::with_capacity(d.layers);
        for i in 0..d.layers {
            let lp = |p: VisionLayerPart| v(VisionPart::Layer(i, p));
            layers.push(Layer {
                ln1: (
                    dev_row(lp(VisionLayerPart::Ln1Weight), h)?,
                    dev_row(lp(VisionLayerPart::Ln1Bias), h)?,
                ),
                ln2: (
                    dev_row(lp(VisionLayerPart::Ln2Weight), h)?,
                    dev_row(lp(VisionLayerPart::Ln2Bias), h)?,
                ),
                ls1: dev_row(lp(VisionLayerPart::Ls1Gamma), h)?,
                ls2: dev_row(lp(VisionLayerPart::Ls2Gamma), h)?,
                q: lin(i, VisionProj::Q)?,
                k: lin(i, VisionProj::K)?,
                v: lin(i, VisionProj::V)?,
                o: lin(i, VisionProj::Out)?,
                fc1: Linear {
                    w: mat(lp(VisionLayerPart::FcWeight), d.mlp, h)?,
                    b: Some(dev_row(lp(VisionLayerPart::FcBias), d.mlp)?),
                },
                fc2: Linear {
                    w: mat(lp(VisionLayerPart::ProjWeight), h, d.mlp)?,
                    b: Some(dev_row(lp(VisionLayerPart::ProjBias), h)?),
                },
            });
        }
        let class = row(v(VisionPart::ClassEmbedding), h)?;
        let cls_pos = row(v(VisionPart::ClsPositionEmbedding), h)?;
        let cls: Vec<f32> = class.iter().zip(&cls_pos).map(|(a, b)| a + b).collect();
        let posemb = bf16_to_f32(
            want(
                w,
                v(VisionPart::PositionEmbedding),
                Dtype::Bf16,
                &[side * side, h],
            )?
            .bytes,
        );
        let nf = d.head_dim / 4;
        let spatial = (d.head_dim / 2) as f32;
        let inv_freq = (0..nf)
            .map(|j| 1.0 / ROPE_THETA.powf((2 * j) as f32 / spatial))
            .collect();
        Ok(VisionTower {
            patch: mat(v(VisionPart::PatchEmbedding), h, d.geometry.native_values())?,
            posemb,
            cls: Tensor::from_data(TensorData::new(cls, [1, h]), dev),
            ln_pre: (
                dev_row(v(VisionPart::LnPreWeight), h)?,
                dev_row(v(VisionPart::LnPreBias), h)?,
            ),
            layers,
            proj_in: mat(v(VisionPart::ProjIn), d.proj_hidden, h * d.merged())?,
            proj_out: mat(v(VisionPart::ProjOut), d.text_hidden, d.proj_hidden)?,
            inv_freq,
            dev: dev.clone(),
            d,
        })
    }

    pub fn dims(&self) -> &VisionDims {
        &self.d
    }

    /// Projected features for each image, concatenated in order.
    pub fn forward_images(&self, images: &[ImageTokens]) -> Result<Vec<ImageFeatures>, String> {
        images.iter().map(|im| self.forward(im)).collect()
    }

    /// One image's projected features.
    pub fn forward(&self, image: &ImageTokens) -> Result<ImageFeatures, String> {
        let seq = self.sequence(image, self.d.layers)?;
        let tokens = image.tokens();
        let values = self
            .project(seq, tokens)?
            .into_data()
            .convert::<f32>()
            .to_vec()
            .map_err(|e| format!("{e:?}"))?;
        Ok(ImageFeatures {
            tokens,
            width: self.d.text_hidden,
            values,
        })
    }

    /// The image's sequence `[1 + P, hidden]` (class row first) after the
    /// embedding, the pre-norm and the first `layers` encoder layers: the
    /// prefix the tower's own activations can be compared at.
    pub fn sequence(&self, image: &ImageTokens, layers: usize) -> Result<Tensor<Bk, 2>, String> {
        let d = &self.d;
        if layers > d.layers {
            return Err(format!("the tower has {} layers, not {layers}", d.layers));
        }
        self.check(image)?;
        let positions = image.native_positions();
        let p = positions.len();
        let h = d.hidden;
        let taps = posemb_taps(&positions, image.patch_grid(), d.posemb_side)?;
        let mut pos = vec![0f32; p * h];
        for (out, tap) in pos.chunks_exact_mut(h).zip(&taps) {
            // upstream: (table[idx] * w).sum over the four taps, in tap order
            for (c, o) in out.iter_mut().enumerate() {
                let mut acc = 0f32;
                for &(i, wt) in tap {
                    acc += self.posemb[i * h + c] * wt;
                }
                *o = acc;
            }
        }
        let rows = Tensor::<Bk, 2>::from_data(
            TensorData::new(image.native_patches(), [p, d.geometry.native_values()]),
            &self.dev,
        );
        let e = linear_bf16(rows, &self.patch)
            + Tensor::<Bk, 2>::from_data(TensorData::new(pos, [p, h]), &self.dev);
        let seq = Tensor::cat(vec![self.cls.clone(), e], 0);
        let mut x = layer_norm(seq, &self.ln_pre.0, &self.ln_pre.1, d.eps);
        let (cos, sin) = self.rotary(&image.sequence_positions());
        for layer in &self.layers[..layers] {
            x = self.layer(layer, x, &cos, &sin);
        }
        Ok(x)
    }

    /// The projector over a finished sequence: drop the class row, group each
    /// token's native patches, `proj_out(gelu(proj_in(.)))`.
    pub fn project(&self, seq: Tensor<Bk, 2>, tokens: usize) -> Result<Tensor<Bk, 2>, String> {
        let d = &self.d;
        let [n, h] = seq.dims();
        if h != d.hidden || n != 1 + tokens * d.merged() {
            return Err(format!(
                "project: sequence [{n}, {h}] is not 1 + {tokens} tokens of {}",
                d.merged()
            ));
        }
        let patches = seq.slice([1..n, 0..h]).reshape([tokens, d.merged() * h]);
        Ok(linear_bf16(
            gelu(linear_bf16(patches, &self.proj_in)),
            &self.proj_out,
        ))
    }

    fn check(&self, image: &ImageTokens) -> Result<(), String> {
        let g = self.d.geometry;
        if image.geometry != g {
            return Err(format!(
                "image geometry {:?} is not the tower's {:?}",
                image.geometry, g
            ));
        }
        let tokens = image.tokens();
        if tokens == 0 {
            return Err("an image with no tokens".into());
        }
        let native = tokens * self.d.merged();
        if native > MAX_NATIVE_PATCHES {
            return Err(format!(
                "{native} native patches exceed {MAX_NATIVE_PATCHES}"
            ));
        }
        if image.pixel_values.len() != tokens * g.token_values() {
            return Err(format!(
                "{} pixel values for a {}x{} token grid of {} each",
                image.pixel_values.len(),
                image.grid.0,
                image.grid.1,
                g.token_values()
            ));
        }
        Ok(())
    }

    /// `[n, 1, head_dim]` cos and sin for the sequence's `(h, w)` positions:
    /// channel pair `j` turns by `w * inv_freq[j]` in the first half and by
    /// `h * inv_freq[j - half]` in the second, each angle on both channels of
    /// its pair (upstream's `recomposition_frequencies`).
    fn rotary(&self, positions: &[[usize; 2]]) -> (Tensor<Bk, 3>, Tensor<Bk, 3>) {
        let hd = self.d.head_dim;
        let nf = self.inv_freq.len();
        let n = positions.len();
        let (mut cos, mut sin) = (Vec::with_capacity(n * hd), Vec::with_capacity(n * hd));
        for &[ph, pw] in positions {
            for c in 0..hd {
                let j = c / 2;
                let f = if j < nf {
                    pw as f32 * self.inv_freq[j]
                } else {
                    ph as f32 * self.inv_freq[j - nf]
                };
                cos.push(f.cos());
                sin.push(f.sin());
            }
        }
        let up =
            |v: Vec<f32>| Tensor::<Bk, 3>::from_data(TensorData::new(v, [n, 1, hd]), &self.dev);
        (up(cos), up(sin))
    }

    fn layer(
        &self,
        l: &Layer,
        x: Tensor<Bk, 2>,
        cos: &Tensor<Bk, 3>,
        sin: &Tensor<Bk, 3>,
    ) -> Tensor<Bk, 2> {
        let d = &self.d;
        let [n, h] = x.dims();
        let (heads, hd) = (d.heads, d.head_dim);
        let hn = layer_norm(x.clone(), &l.ln1.0, &l.ln1.1, d.eps);
        let rope = |t: Tensor<Bk, 2>| -> Tensor<Bk, 3> {
            let t = t.reshape([n, heads, hd]);
            let pairs = t.clone().reshape([n, heads, hd / 2, 2]);
            let x1 = pairs.clone().slice([0..n, 0..heads, 0..hd / 2, 0..1]);
            let x2 = pairs.slice([0..n, 0..heads, 0..hd / 2, 1..2]);
            let rotated = Tensor::cat(vec![x2.neg(), x1], 3).reshape([n, heads, hd]);
            t * cos.clone() + rotated * sin.clone()
        };
        let q = rope(l.q.forward(hn.clone())).swap_dims(0, 1);
        let kt = rope(l.k.forward(hn.clone()))
            .swap_dims(0, 1)
            .swap_dims(1, 2);
        let v = l.v.forward(hn).reshape([n, heads, hd]).swap_dims(0, 1);
        let scale = (hd as f32).powf(-0.5);
        let tile = (SCORE_BUDGET / (heads * n)).clamp(1, n);
        let mut outs = Vec::with_capacity(n.div_ceil(tile));
        let mut i0 = 0;
        while i0 < n {
            let tq = tile.min(n - i0);
            let s = q
                .clone()
                .slice([0..heads, i0..i0 + tq, 0..hd])
                .matmul(kt.clone())
                .mul_scalar(scale);
            let o = softmax(s, 2).matmul(v.clone());
            outs.push(o.swap_dims(0, 1).reshape([tq, heads * hd]));
            i0 += tq;
        }
        let attn = if outs.len() == 1 {
            outs.pop().expect("one tile")
        } else {
            Tensor::cat(outs, 0)
        };
        let x = x + l.ls1.clone() * l.o.forward(attn);
        let m = l.fc2.forward(gelu(l.fc1.forward(layer_norm(
            x.clone(),
            &l.ln2.0,
            &l.ln2.1,
            d.eps,
        ))));
        debug_assert_eq!(h, d.hidden);
        x + l.ls2.clone() * m
    }
}
