//! Resident NVFP4 copies of Breeze's linear projections, quantized once from
//! the immutable BF16 pile aliases at load, and the W4A16 GEMV that reads them.
//!
//! WHY. Every Breeze decode step is a one-row (M=1) projection bound by
//! memory. nsys on sky (GB10, live control, 2026-10-10) put ~84 ms of a ~112 ms
//! frame (83 ms of audio) in M=1 BF16 projections at ~152 GB/s: 2.82 GB per
//! backbone step plus 15 depth steps of 667 MB. Four-bit weights move 0.5625
//! bytes per weight instead of 2. JP, 2026-10-10: "we want to use 4bit
//! wherever we can".
//!
//! The format, the quantization recipe and the GEMV are shared with Voxtral's
//! hearing decoder and live in [`crate::nn::nvfp4_gemv`]; this module is the
//! Breeze side: the `BREEZE_WEIGHTS` choice, the projection that carries both
//! copies, and the gates that pin the BF16 instantiation.
//!
//! ## Which rows take the GEMV
//!
//! `[1, t, K]` with `t <= MAX_ROWS` runs the shared NVFP4 GEMV: every decode
//! step, and the depth decoder's two-row prefill of each frame. Larger `t` (the
//! backbone and text-encoder prompt prefills, a few hundred rows) goes through
//! the unchanged BF16 projection on the pile alias. That alias is the pile
//! mapping already registered with CUDA for the weights' whole lifetime, so it
//! costs no memory to keep; a dequantize-to-scratch per prefill call would move
//! 0.5625 + 2 + 2 bytes per weight against the alias's 2. The consequence is a
//! prompt state computed at full BF16 precision and decode steps at four bits.
//!
//! ## Measured on sky (GB10), 2026-10-10
//!
//! Framing: `cuda_gemv_bandwidth`, one M=1 launch at a time back to back on
//! one stream, 600 launches rotating over >= 512 MB of distinct weight copies
//! so every launch streams from DRAM; bytes = weights + E4M3 scales +
//! activation + output, against the BF16 Cubek GEMV this replaces. Live bot
//! present on the box, not otherwise loaded.
//!
//! | shape                         | NVFP4 us | GB/s | BF16 us | GB/s |
//! | ----------------------------- | -------: | ---: | ------: | ---: |
//! | depth gate/up `8192 x 1024`   |     25.9 |  183 |    86.4 |  195 |
//! | depth down `1024 x 8192`      |     25.0 |  189 |    84.6 |  199 |
//! | backbone gate/up `6144 x 2048`|     32.3 |  220 |   109.8 |  229 |
//! | depth k/v `256 x 1024`        |      4.1 |   37 |     8.2 |   64 |
//!
//! The small shapes sit on a ~4 us per-launch floor. End to end, the resident
//! control's request 2 (seed 42, CFG1, same binary) went from 6.01 s / 54
//! frames (BF16) to 2.37 s / 56 frames (NVFP4) of generation: ~111 to ~42 ms
//! per 83 ms frame.
use super::cuda_ops::Tensor;
pub(super) use crate::nn::nvfp4_gemv::Nvfp4;
use anyhow::Result;
#[cfg(test)]
use {
    crate::nn::nvfp4_gemv::{GROUP, MAX_ROWS, scale2_of},
    burn::tensor::DType,
    cubecl::{cuda::CudaRuntime, prelude::*, server::Handle},
    half::bf16,
};

/// Which weights the resident decodes from. One explicit choice per load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Weights {
    /// The BF16 pile aliases only: byte-identical to the pre-NVFP4 path.
    #[default]
    Bf16,
    /// A resident NVFP4 copy of every linear projection of the text encoder,
    /// backbone and depth decoder; embeddings, LM heads and norms stay BF16.
    Nvfp4,
}

impl Weights {
    /// `BREEZE_WEIGHTS`: unset or `bf16` keeps BF16, `nvfp4` selects four bits.
    /// Anything else is refused rather than silently read as the default.
    pub fn from_env() -> Result<Self> {
        match std::env::var("BREEZE_WEIGHTS") {
            Err(std::env::VarError::NotPresent) => Ok(Self::Bf16),
            Ok(value) => Self::parse(&value),
            Err(e) => anyhow::bail!("BREEZE_WEIGHTS: {e}"),
        }
    }
    fn parse(value: &str) -> Result<Self> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "nvfp4" => Ok(Self::Nvfp4),
            other => anyhow::bail!("BREEZE_WEIGHTS must be bf16 or nvfp4, not {other:?}"),
        }
    }
}

/// One linear projection: the immutable BF16 pile alias, and under
/// [`Weights::Nvfp4`] its resident four-bit copy.
pub(super) struct Linear {
    pub(super) bf16: Tensor,
    pub(super) nvfp4: Option<Nvfp4>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::mxfp4::{E2M1, e4m3_to_f32};

    /// splitmix64: a fixed, dependency-free stream for reproducible fixtures.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Standard normal by Box-Muller.
        fn normal(&mut self) -> f64 {
            let u = self.unit().max(1e-300);
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
        }
    }

    /// Weight-shaped values: N(0, sigma) with a sprinkling of 8-sigma outliers,
    /// so block scales span several octaves below the tensor's E4M3 ceiling.
    fn weights(rng: &mut Rng, len: usize, sigma: f64) -> Vec<bf16> {
        (0..len)
            .map(|_| {
                let outlier = if rng.next() % 997 == 0 { 8.0 } else { 1.0 };
                bf16::from_f64(rng.normal() * sigma * outlier)
            })
            .collect()
    }

    /// Round to the nearest E4M3FN value, ties to the even pattern, for a
    /// finite `v` in `[0, 448]`. Brackets the 127 non-negative finite patterns.
    fn e4m3_rn(v: f32) -> u8 {
        assert!((0.0..=448.0).contains(&v), "{v} outside the clamped range");
        let v = v as f64;
        let value = |p: u8| e4m3_to_f32(p) as f64;
        let mut lo = 0u8;
        while lo < 0x7E && value(lo + 1) <= v {
            lo += 1;
        }
        if value(lo) == v || lo == 0x7E {
            return lo;
        }
        let (below, above) = (v - value(lo), value(lo + 1) - v);
        if below < above || (below == above && lo % 2 == 0) {
            lo
        } else {
            lo + 1
        }
    }

    /// `cvt.rn.satfinite.e2m1x2.f32` for one finite value: nearest E2M1
    /// magnitude, ties to the even code, saturating at 6, sign kept (also on
    /// a value that rounds to zero, which becomes code 8).
    fn e2m1_rn(q: f32) -> u8 {
        const MIDPOINTS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
        let a = q.abs();
        let mut code = 0u8;
        for (i, &m) in MIDPOINTS.iter().enumerate() {
            if a > m || (a == m && (i + 1) % 2 == 0) {
                code = i as u8 + 1;
            }
        }
        code | if q.is_sign_negative() { 8 } else { 0 }
    }

    /// The module header's recipe on the host, operation for operation.
    /// Returns (codes [n, k/2], scales [n, k/16], scale2).
    fn recipe(w: &[bf16]) -> (Vec<u8>, Vec<u8>, f32) {
        let amax = w.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
        let s2 = scale2_of(amax).unwrap();
        let mut codes = Vec::with_capacity(w.len() / 2);
        let mut scales = Vec::with_capacity(w.len() / GROUP);
        for block in w.chunks_exact(GROUP) {
            let bamax = block.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
            let sb = e4m3_rn(((bamax / 6.0) / s2).min(448.0));
            scales.push(sb);
            let d = e4m3_to_f32(sb) * s2;
            for pair in block.chunks_exact(2) {
                let code = |v: bf16| {
                    if d > 0.0 { e2m1_rn(v.to_f32() / d) } else { 0 }
                };
                codes.push(code(pair[0]) | code(pair[1]) << 4);
            }
        }
        (codes, scales, s2)
    }

    /// Decode packed NVFP4 to f64: exact, since E2M1 x E4M3 x f32 fits.
    fn decode(codes: &[u8], scales: &[u8], s2: f32) -> Vec<f64> {
        let mut out = Vec::with_capacity(codes.len() * 2);
        for (b, &s) in scales.iter().enumerate() {
            let scale = e4m3_to_f32(s) as f64 * s2 as f64;
            for &byte in &codes[b * GROUP / 2..(b + 1) * GROUP / 2] {
                out.push(E2M1[(byte & 15) as usize] as f64 * scale);
                out.push(E2M1[(byte >> 4) as usize] as f64 * scale);
            }
        }
        out
    }

    #[test]
    fn weights_option_is_explicit() {
        assert_eq!(Weights::default(), Weights::Bf16);
        assert_eq!(Weights::parse("bf16").unwrap(), Weights::Bf16);
        assert_eq!(Weights::parse("nvfp4").unwrap(), Weights::Nvfp4);
        for bad in ["", "NVFP4", "fp4", "bf16 "] {
            assert!(Weights::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn host_converters_are_the_formats() {
        // Every E4M3 value maps to itself, and every midpoint to its even
        // neighbour.
        for p in 0u8..=0x7E {
            assert_eq!(e4m3_rn(e4m3_to_f32(p)), p);
            if p < 0x7E {
                let mid = ((e4m3_to_f32(p) as f64 + e4m3_to_f32(p + 1) as f64) / 2.0) as f32;
                assert_eq!(
                    mid as f64 * 2.0,
                    e4m3_to_f32(p) as f64 + e4m3_to_f32(p + 1) as f64
                );
                assert_eq!(
                    e4m3_rn(mid),
                    if p % 2 == 0 { p } else { p + 1 },
                    "pattern {p}"
                );
            }
        }
        assert_eq!(e4m3_rn(1.0 / 2048.0), 0, "below half the least subnormal");
        for code in 0u8..16 {
            assert_eq!(e2m1_rn(E2M1[code as usize]), code);
        }
        let ties = [
            (0.25, 0),
            (0.75, 2),
            (1.25, 2),
            (1.75, 4),
            (2.5, 4),
            (3.5, 6),
            (5.0, 6),
        ];
        for (q, code) in ties {
            assert_eq!(e2m1_rn(q), code, "{q}");
            assert_eq!(e2m1_rn(-q), code | 8, "-{q}");
        }
        assert_eq!(e2m1_rn(1e9), 7);
        assert_eq!(
            e2m1_rn(-0.1),
            8,
            "a negative value that rounds to zero keeps its sign"
        );
        assert_eq!(scale2_of(0.0).unwrap(), 1.0);
        assert!(scale2_of(f32::INFINITY).is_err());
    }

    #[test]
    fn host_decoder_inverts_the_recipe_on_representable_blocks() {
        // A tensor whose amax is 2688 has scale2 == 1, so a block with block
        // amax 6 has scale 1 and its E2M1 values come back exactly.
        let mut w = vec![bf16::ZERO; 32];
        w[0] = bf16::from_f32(2688.0);
        for (i, v) in E2M1.iter().enumerate() {
            w[16 + i] = bf16::from_f32(*v);
        }
        let (codes, scales, s2) = recipe(&w);
        assert_eq!(s2, 1.0);
        assert_eq!(scales[1], 0x38, "E4M3 1.0");
        let back = decode(&codes, &scales, s2);
        for i in 0..16 {
            assert_eq!(back[16 + i].to_bits(), (E2M1[i] as f64).to_bits(), "{i}");
        }
        assert_eq!(back[0], 2688.0);
    }

    fn upload(client: &ComputeClient<CudaRuntime>, values: &[bf16], shape: &[usize]) -> Tensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        Tensor::new_contiguous(
            client.clone(),
            cubecl::cuda::CudaDevice { index: 0 },
            shape.into(),
            client.create_from_slice(&bytes),
            DType::BF16,
        )
    }

    fn read(client: &ComputeClient<CudaRuntime>, handle: &Handle) -> Vec<u8> {
        client.read_one(handle.clone()).unwrap().to_vec()
    }

    /// An adversarial matrix: exact E2M1 ties at scale 1, blocks far below the
    /// tensor amax (subnormal and zero E4M3 scales), all-zero and signed-zero
    /// blocks, then weight-shaped rows.
    fn adversarial(rng: &mut Rng, n: usize, k: usize) -> Vec<bf16> {
        let mut w = weights(rng, n * k, 0.5);
        w[0] = bf16::from_f32(2688.0);
        let ties = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0];
        for i in 0..k {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            w[k + i] = bf16::from_f32(sign * ties[i % 8]);
            w[2 * k + i] = bf16::from_f64(rng.normal() * 1e-3);
            w[3 * k + i] = bf16::from_f64(rng.normal() * 1e-6);
            w[4 * k + i] = if i % 2 == 0 {
                bf16::NEG_ZERO
            } else {
                bf16::ZERO
            };
        }
        w
    }

    #[test]
    #[ignore = "reserved CUDA: device NVFP4 quantizer against the host recipe, bit for bit"]
    fn cuda_quantizer_is_the_host_recipe_bit_for_bit() {
        let client = CudaRuntime::client(&cubecl::cuda::CudaDevice { index: 0 });
        let mut rng = Rng(0x5EED_0F_F4);
        let cases: Vec<(&str, usize, usize, Vec<bf16>)> = vec![
            ("adversarial", 64, 1024, adversarial(&mut rng, 64, 1024)),
            (
                "depth down",
                1024,
                8192,
                weights(&mut rng, 1024 * 8192, 0.02),
            ),
            (
                "text gate",
                6912,
                1152,
                weights(&mut rng, 6912 * 1152, 0.03),
            ),
            (
                "backbone k",
                1024,
                2048,
                weights(&mut rng, 1024 * 2048, 0.05),
            ),
        ];
        for (name, n, k, w) in cases {
            let t = upload(&client, &w, &[n, k]);
            let q = Nvfp4::quantize(&t).unwrap();
            let (codes, scales, s2) = recipe(&w);
            assert_eq!(q.scale2.to_bits(), s2.to_bits(), "{name}: scale2");
            let dev_scales = read(&client, &q.scales);
            let dev_codes = read(&client, &q.codes);
            assert_eq!(dev_scales.len(), scales.len());
            assert_eq!(dev_codes.len(), codes.len());
            let bad_scales = dev_scales
                .iter()
                .zip(&scales)
                .filter(|(a, b)| a != b)
                .count();
            let bad_codes = dev_codes.iter().zip(&codes).filter(|(a, b)| a != b).count();
            assert_eq!(
                (bad_scales, bad_codes),
                (0, 0),
                "{name}: mismatched scale/code bytes"
            );
            let dev = decode(&dev_codes, &dev_scales, q.scale2);
            let host = decode(&codes, &scales, s2);
            assert!(
                dev.iter()
                    .zip(&host)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            // The quantization error itself, for the record (not a gate).
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (v, d) in w.iter().zip(&dev) {
                err += (v.to_f64() - d).powi(2);
                norm += v.to_f64().powi(2);
            }
            eprintln!(
                "{name} [{n}, {k}]: scale2 {s2:e}, relative RMS quantization error {:.4}",
                (err / norm).sqrt()
            );
        }
    }

    /// Host f64 reference: decode the packed weights, multiply in f64.
    fn reference(x: &[bf16], rows: usize, k: usize, w: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut y = vec![0.0f64; rows * n];
        let mut mag = vec![0.0f64; rows * n];
        for r in 0..rows {
            for j in 0..n {
                let (mut s, mut a) = (0.0f64, 0.0f64);
                for i in 0..k {
                    let p = x[r * k + i].to_f64() * w[j * k + i];
                    s += p;
                    a += p.abs();
                }
                y[r * n + j] = s;
                mag[r * n + j] = a;
            }
        }
        (y, mag)
    }

    const SHAPES: [(&str, usize, usize); 17] = [
        ("backbone q", 2048, 2048),
        ("backbone k/v", 1024, 2048),
        ("backbone o", 2048, 2048),
        ("backbone gate/up", 6144, 2048),
        ("backbone down", 2048, 6144),
        ("depth q", 1024, 1024),
        ("depth k/v", 256, 1024),
        ("depth o", 1024, 1024),
        ("depth gate/up", 8192, 1024),
        ("depth down", 1024, 8192),
        ("depth projector", 1024, 2048),
        ("text q", 1024, 1152),
        ("text k/v", 256, 1152),
        ("text o", 1152, 1024),
        ("text gate/up", 6912, 1152),
        ("text down", 1152, 6912),
        ("text projection", 2048, 1152),
    ];

    #[test]
    #[ignore = "reserved CUDA: W4A16 GEMV against a host f64 product over every Breeze shape"]
    fn cuda_gemv_matches_f64_reference_over_real_shapes() {
        let client = CudaRuntime::client(&cubecl::cuda::CudaDevice { index: 0 });
        let mut rng = Rng(0x0BAD_CAFE);
        let mut worst_f32 = 0.0f64;
        for (name, n, k) in SHAPES {
            let w = weights(&mut rng, n * k, 0.02);
            let q = Nvfp4::quantize(&upload(&client, &w, &[n, k])).unwrap();
            let dense = decode(
                &read(&client, &q.codes),
                &read(&client, &q.scales),
                q.scale2,
            );
            for rows in [1usize, 2, MAX_ROWS] {
                let xv: Vec<bf16> = (0..rows * k)
                    .map(|_| bf16::from_f64(rng.normal()))
                    .collect();
                let x = upload(&client, &xv, &[1, rows, k]);
                assert_eq!(q.takes(&x), Some(rows));
                let (y, mag) = reference(&xv, rows, k, &dense, n);
                // f32 output: the accumulation alone. Bound per output by the
                // absolute sum of its products, the scale of f32 summation error.
                let raw = read(&client, &q.launch::<f32>(&x, rows));
                let mut worst = 0.0f64;
                for (i, b) in raw.chunks_exact(4).enumerate() {
                    let got = f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64;
                    worst = worst.max((got - y[i]).abs() / mag[i].max(f64::MIN_POSITIVE));
                }
                worst_f32 = worst_f32.max(worst);
                assert!(
                    worst < 64.0 * f32::EPSILON as f64,
                    "{name} rows={rows}: f32 accumulation error {worst:e} of sum|x w|"
                );
                // BF16 output: one final round to nearest of that f32.
                let out = q.gemv(&x, rows);
                assert_eq!(out.meta.shape().as_slice(), [1, rows, n]);
                let got = read(&client, &out.handle);
                for (i, b) in got.chunks_exact(2).enumerate() {
                    let got = bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64();
                    let want = bf16::from_f64(y[i]).to_f64();
                    let top = want.abs().max(y[i].abs()).max(f64::MIN_POSITIVE);
                    let ulp = top.log2().floor().exp2() / 128.0;
                    assert!(
                        (got - y[i]).abs() <= ulp / 2.0 + 64.0 * f32::EPSILON as f64 * mag[i],
                        "{name} rows={rows} out {i}: {got} vs exact {} (bf16 {want})",
                        y[i]
                    );
                }
            }
            eprintln!("{name} [{n}, {k}]: ok");
        }
        eprintln!(
            "worst f32 accumulation error: {worst_f32:e} of sum|x w| ({:.1} f32 epsilons)",
            worst_f32 / f32::EPSILON as f64
        );
    }

    #[test]
    #[ignore = "reserved CUDA: GEMV bandwidth on Breeze shapes, NVFP4 against the BF16 projection"]
    fn cuda_gemv_bandwidth() {
        use std::time::Instant;
        let device = cubecl::cuda::CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let sync = || cubecl::future::block_on(client.sync()).unwrap();
        let mut rng = Rng(7);
        let iters = 600usize;
        // Rotate through distinct copies of each weight, >= 512 MB of BF16, so
        // every launch streams from DRAM as the model does (layers far exceed
        // L2). Repeating ONE weight measures L2, not the memory system.
        eprintln!(
            "framing: one M=1 launch, back-to-back on one stream, {iters} launches over \
             distinct weights, sky GB10; bytes = weights (+ E4M3 scales) + activation + output"
        );
        eprintln!("shape            | NVFP4 us   GB/s | BF16 us   GB/s | speedup");
        for (name, n, k) in SHAPES {
            let w = weights(&mut rng, n * k, 0.02);
            let copies = (512usize << 20).div_ceil(2 * n * k).max(2);
            let dense: Vec<Linear> = (0..copies)
                .map(|_| Linear {
                    bf16: upload(&client, &w, &[n, k]),
                    nvfp4: None,
                })
                .collect();
            let quant: Vec<Nvfp4> = dense
                .iter()
                .map(|l| Nvfp4::quantize(&l.bf16).unwrap())
                .collect();
            let x = upload(
                &client,
                &(0..k)
                    .map(|_| bf16::from_f64(rng.normal()))
                    .collect::<Vec<_>>(),
                &[1, 1, k],
            );
            for i in 0..2 * copies {
                quant[i % copies].gemv(&x, 1);
                super::super::backbone::linear(&x, &dense[i % copies]).unwrap();
            }
            sync();
            let started = Instant::now();
            for i in 0..iters {
                quant[i % copies].gemv(&x, 1);
            }
            sync();
            let fp4 = started.elapsed().as_secs_f64() / iters as f64;
            let started = Instant::now();
            for i in 0..iters {
                super::super::backbone::linear(&x, &dense[i % copies]).unwrap();
            }
            sync();
            let bf = started.elapsed().as_secs_f64() / iters as f64;
            let fp4_bytes = (quant[0].bytes() + 2 * k as u64 + 2 * n as u64) as f64;
            let dense_bytes = (2 * n * k + 2 * k + 2 * n) as f64;
            eprintln!(
                "{name:16} | {:7.1} {:6.1} | {:7.1} {:6.1} | {:.2}x",
                fp4 * 1e6,
                fp4_bytes / fp4 / 1e9,
                bf * 1e6,
                dense_bytes / bf / 1e9,
                bf / fp4
            );
        }
    }
}
