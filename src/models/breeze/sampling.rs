//! CUDA logits policy follows the pinned suppress/temperature/top-k/top-p path.
//! Only a seeded uniform scalar and the final sampled ID cross the host seam.
//! SplitMix64 categorical draws are NOT Torch RNG-coordinate parity.
use super::{
    cuda_ops::{Tensor, count, grid},
    generator::GenerationOptions,
};
use anyhow::{Result, ensure};
use burn::tensor::DType;
use cubecl::{cuda::CudaRuntime, prelude::*};

#[cube(launch_unchecked)]
fn cfg_kernel(
    positive: &Array<f32>,
    negative: &Array<f32>,
    out: &mut Array<f32>,
    n: usize,
    scale: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        out[i] = negative[i] + scale * (positive[i] - negative[i]);
    }
}

/// Guidance is on raw F32 logits BEFORE reserved-token masking, repetition,
/// temperature and sampling. It is not interpolation of hidden states or codes.
fn cfg_logits(positive: &Tensor, negative: &Tensor, scale: f32) -> Result<Tensor> {
    let n = count(positive);
    ensure!(
        (n == 2051 || n == 2052) && positive.meta.shape() == negative.meta.shape(),
        "CFG head shape mismatch"
    );
    ensure!(
        positive.dtype == DType::F32 && negative.dtype == DType::F32,
        "CFG requires F32 logits"
    );
    ensure!(scale.is_finite() && scale > 0.0, "invalid CFG scale");
    let out = positive.client.empty(n * 4);
    // SAFETY: checked equal contiguous F32 heads, fresh writable result; neither
    // branch's logits nor any immutable pile leaf is modified.
    unsafe {
        cfg_kernel::launch_unchecked::<CudaRuntime>(
            &positive.client,
            grid(positive, n),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(positive.handle.clone(), n),
            ArrayArg::from_raw_parts(negative.handle.clone(), n),
            ArrayArg::from_raw_parts(out.clone(), n),
            n,
            scale,
        );
    }
    Ok(Tensor::new_contiguous(
        positive.client.clone(),
        positive.device.clone(),
        positive.meta.shape().clone(),
        out,
        DType::F32,
    ))
}

pub(super) struct Sampler {
    state: u64,
}
impl Sampler {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn uniform(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        // Midpoints avoid exact zero and one in the F32 transport.
        ((z >> 41) as f32 + 0.5) / 8388608.0
    }
    pub(super) fn sample_guided(
        &mut self,
        positive: &Tensor,
        negative: Option<&Tensor>,
        options: &GenerationOptions,
        history: &[u32],
        backbone: bool,
    ) -> Result<u32> {
        ensure!(
            negative.is_some() == (options.cfg_scale != 1.0),
            "CFG branch/scale mismatch"
        );
        match negative {
            Some(negative) => self.sample(
                &cfg_logits(positive, negative, options.cfg_scale)?,
                options,
                history,
                backbone,
            ),
            None => self.sample(positive, options, history, backbone),
        }
    }
    pub(super) fn sample(
        &mut self,
        logits: &Tensor,
        o: &GenerationOptions,
        history: &[u32],
        backbone: bool,
    ) -> Result<u32> {
        let n = count(logits);
        ensure!(
            n == if backbone { 2052 } else { 2051 },
            "unexpected audio head width"
        );
        let mut seen = vec![0u32; n];
        for &id in history {
            ensure!((id as usize) < n, "history ID outside head");
            seen[id as usize] = 1;
        }
        let seen = logits.client.create_from_slice(u32::as_bytes(&seen));
        let values = logits.client.empty(n * 4);
        let masses = logits.client.empty(n * 4);
        let output = logits.client.empty(4);
        let uniform = self.uniform();
        // All arrays are fresh bounded vocabulary scratch, not weight aliases.
        unsafe {
            adjust_kernel::launch_unchecked::<CudaRuntime>(
                &logits.client,
                grid(logits, n),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(logits.handle.clone(), n),
                ArrayArg::from_raw_parts(seen, n),
                ArrayArg::from_raw_parts(values.clone(), n),
                n,
                o.repetition_penalty,
                if o.do_sample { o.temperature } else { 1.0 },
                backbone,
            );
            mass_kernel::launch_unchecked::<CudaRuntime>(
                &logits.client,
                grid(logits, n),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(values.clone(), n),
                ArrayArg::from_raw_parts(masses.clone(), n),
                n,
                if o.do_sample { o.top_k } else { 1 },
            );
            choose_kernel::launch_unchecked::<CudaRuntime>(
                &logits.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(values, n),
                ArrayArg::from_raw_parts(masses, n),
                ArrayArg::from_raw_parts(output.clone(), 1),
                n,
                o.top_p,
                uniform,
                o.do_sample,
            );
        }
        let bytes = logits
            .client
            .read_one(output)
            .map_err(|e| anyhow::anyhow!("CUDA sample read: {e:?}"))?;
        let word = bytes
            .get(..4)
            .ok_or_else(|| anyhow::anyhow!("short CUDA sample read"))?;
        let id = u32::from_ne_bytes(word.try_into()?);
        ensure!(id != u32::MAX, "nonfinite/empty CUDA sampling distribution");
        Ok(id)
    }
}

#[cube(launch_unchecked)]
fn adjust_kernel(
    x: &Array<f32>,
    seen: &Array<u32>,
    out: &mut Array<f32>,
    n: usize,
    penalty: f32,
    temperature: f32,
    #[comptime] backbone: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let mut v = x[i];
        if seen[i] != 0 {
            if v > 0.0f32 {
                v /= penalty;
            } else {
                v *= penalty;
            }
        }
        v /= temperature;
        if i >= 2048 && !(backbone && i == 2051) {
            v = -3.4028235e38f32;
        }
        out[i] = v;
    }
}
#[cube(launch_unchecked)]
fn mass_kernel(x: &Array<f32>, out: &mut Array<f32>, n: usize, k: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let value = x[i];
        let mut max = f32::cast_from(-3.4028235e38f32);
        let mut higher = usize::cast_from(0u32);
        let mut invalid = false;
        for j in 0usize..n {
            let v = x[j];
            if v != v || v > 3.4028235e38f32 || v < -3.4028235e38f32 {
                invalid = true;
            }
            if v > value {
                higher += 1usize;
            }
            if v > max {
                max = v;
            }
        }
        let mut mass = 0.0f32;
        if value > -3.4028235e38f32 && (k == 0usize || higher < k) {
            mass = (value - max).exp();
        }
        if invalid {
            mass = -1.0f32;
        }
        out[i] = mass;
    }
}
#[cube(launch_unchecked)]
fn choose_kernel(
    x: &Array<f32>,
    mass: &Array<f32>,
    out: &mut Array<u32>,
    n: usize,
    top_p: f32,
    uniform: f32,
    #[comptime] sample: bool,
) {
    if ABSOLUTE_POS == 0 {
        let mut total = 0.0f32;
        let mut invalid = false;
        let mut best = f32::cast_from(-3.4028235e38f32);
        let mut best_id = 0u32;
        for i in 0..n {
            total += mass[i];
            if mass[i] < 0.0f32 || mass[i] != mass[i] {
                invalid = true;
            }
            if x[i] > best {
                best = x[i];
                best_id = i as u32;
            }
        }
        if invalid || total <= 0.0f32 {
            out[0] = 4294967295u32;
        } else if !sample {
            out[0] = best_id;
        } else {
            // Local per-call scratch; rank ties use ID order. Keep the token
            // crossing top_p, matching upstream's shifted removal boundary.
            let mut keep = Array::<f32>::new(2052usize);
            let mut kept_total = 0.0f32;
            for i in 0..n {
                let mut before = 0.0f32;
                if top_p < 1.0f32 && mass[i] > 0.0f32 {
                    for j in 0..n {
                        if x[j] > x[i] || (x[j] == x[i] && j < i) {
                            before += mass[j];
                        }
                    }
                }
                keep[i] = 0.0f32;
                if before <= top_p * total {
                    keep[i] = mass[i];
                }
                kept_total += keep[i];
            }
            let target = uniform * kept_total;
            let mut cdf = 0.0f32;
            let mut selected = 4294967295u32;
            let mut last = 4294967295u32;
            for i in 0..n {
                if keep[i] > 0.0f32 {
                    last = i as u32;
                }
                cdf += keep[i];
                if selected == 4294967295u32 && cdf > target {
                    selected = i as u32;
                }
            }
            if selected == 4294967295u32 {
                selected = last;
            }
            out[0] = selected;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seeded_uniform_transport_is_bounded_and_reproducible() {
        let mut a = Sampler::new(42);
        let mut b = Sampler::new(42);
        for _ in 0..1000 {
            let u = a.uniform();
            assert!(u > 0.0 && u < 1.0);
            assert_eq!(u, b.uniform());
        }
    }

    #[test]
    #[ignore = "actual CUDA device 0 requires ordinary Stars lock"]
    fn cuda_paired_cfg_changes_both_heads_and_keeps_eos_rules() {
        use cubecl::{Runtime, cuda::CudaDevice};
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let make = |values: &[f32]| {
            Tensor::new_contiguous(
                client.clone(),
                device.clone(),
                [1, 1, values.len()].into(),
                client.create_from_slice(f32::as_bytes(values)),
                DType::F32,
            )
        };
        for backbone in [true, false] {
            let n = if backbone { 2052 } else { 2051 };
            let mut positive = vec![-20.0; n];
            let mut negative = vec![-20.0; n];
            // Planted omission/reversal control: conditional alone picks7,
            // reversed branches pick8, correct paired CFG4 must pick9.
            positive[7] = 4.0;
            negative[7] = 4.0;
            positive[8] = 3.0;
            negative[8] = 8.0;
            positive[9] = 2.0;
            negative[9] = -2.0;
            // Reserved codec token must remain impossible after guidance.
            positive[2048] = 100.0;
            negative[2048] = -100.0;
            let p = make(&positive);
            let n = make(&negative);
            let mut options = GenerationOptions::default();
            options.do_sample = false;
            let mut sampler = Sampler::new(42);
            assert_eq!(
                sampler
                    .sample_guided(&p, None, &options, &[], backbone)
                    .unwrap(),
                7
            );
            options.cfg_scale = 4.0;
            assert!(
                sampler
                    .sample_guided(&p, None, &options, &[], backbone)
                    .is_err()
            );
            assert_eq!(
                sampler
                    .sample_guided(&p, Some(&n), &options, &[], backbone)
                    .unwrap(),
                9
            );
            if backbone {
                positive[2051] = 2.0;
                negative[2051] = -3.0;
                assert_eq!(
                    sampler
                        .sample_guided(
                            &make(&positive),
                            Some(&make(&negative)),
                            &options,
                            &[],
                            true
                        )
                        .unwrap(),
                    2051
                );
            }
        }
    }
}
