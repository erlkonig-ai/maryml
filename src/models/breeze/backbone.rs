//! Causal Qwen3 backbone and ordinary Breeze depth layers. Geometry comes from
//! the selected nested configs, never the misleading outer backbone defaults.
use super::{
    config::{DecoderConfig, RopeConfig, RopeKind},
    cuda_ops as ops,
    generator::Binder,
    nvfp4::Linear,
};
use crate::models::qwen3_5::gdn_mixer::project;
use anyhow::{Result, ensure};
use burn::tensor::DType;
use cubecl::{cuda::CudaRuntime, prelude::*};
pub(super) use ops::Tensor;
use triblespace::core::repo::BlobStoreGet;

/// One projection. A resident NVFP4 copy takes the activation shapes its GEMV
/// accepts (`[1, t, K]`, `t <= nvfp4::MAX_ROWS`); everything else, and every
/// projection under the default BF16 weights, is the unchanged BF16 path.
pub(super) fn linear(x: &Tensor, w: &Linear) -> Result<Tensor> {
    if let Some(q) = &w.nvfp4
        && let Some(rows) = q.takes(x)
        && x.device == w.bf16.device
    {
        return Ok(q.gemv(x, rows));
    }
    linear_bf16(x, &w.bf16)
}

pub(super) fn linear_bf16(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    if gemv_eligible(x) {
        project_gemv(x, w)
    } else {
        // Keep M>1 and unsupported small K on the old path BEFORE launch.
        project(x, w).map_err(anyhow::Error::msg)
    }
}

fn single_row(shape: &[usize]) -> bool {
    shape.len() == 3 && shape[0] == 1 && shape[1] == 1
}

fn gemv_vector_takes(shape: &[usize], plane: usize, vector: usize) -> bool {
    single_row(shape)
        && shape[2] > 0
        && plane
            .checked_mul(vector)
            .is_some_and(|stride| stride > 0 && shape[2].is_multiple_of(stride))
}

fn gemv_eligible(x: &Tensor) -> bool {
    // Match Cubek's documented/preflight K divisibility and BF16 vector-size
    // choice. This is shape/capability dispatch, not catch-and-retry on error.
    let plane = x.client.properties().hardware.plane_size_max as usize;
    x.client
        .io_optimized_vector_sizes(2)
        .any(|vector| gemv_vector_takes(x.meta.shape().as_slice(), plane, vector))
}

// Same descriptor boundary as the existing projection; no hidden contiguous
// materialization or reinterpretation of a non-BF16/quantized tensor.
fn checked_layout(shape: &[usize], strides: &[usize], bytes: u64) -> Result<usize> {
    ensure!(shape.len() == strides.len(), "GEMV stride rank mismatch");
    let mut elements = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        ensure!(dim > 0, "GEMV empty extent");
        ensure!(
            dim == 1 || stride == elements,
            "GEMV requires contiguous storage"
        );
        elements = elements
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("GEMV extent overflow"))?;
    }
    ensure!(elements <= u32::MAX as usize, "GEMV exceeds u32 indexing");
    let required = elements
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("GEMV byte extent overflow"))?;
    ensure!(bytes >= required as u64, "GEMV storage is too short");
    Ok(elements)
}

fn projection_elems() -> cubek::matmul::definition::MatmulGlobalElems {
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    let bf16 = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
    cubek::matmul::definition::MatmulGlobalElems {
        lhs: bf16,
        rhs: bf16,
        out: bf16,
    }
}

fn project_gemv(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    use cubek::matmul::{definition::MatmulElems, launch::Strategy};
    use cubek::std::InputBinding;
    fn binding(
        handle: &cubecl::server::Handle,
        shape: [usize; 2],
        strides: [usize; 2],
    ) -> TensorBinding<CudaRuntime> {
        TensorBinding {
            handle: handle.clone().binding(),
            shape: shape.into(),
            strides: strides.into(),
            runtime: core::marker::PhantomData,
        }
    }
    let s = x.meta.shape().as_slice();
    let ws = w.meta.shape().as_slice();
    ensure!(
        single_row(s) && ws.len() == 2 && s[2] == ws[1],
        "GEMV requires [1,1,K] and [N,K]"
    );
    ensure!(
        x.dtype == DType::BF16
            && w.dtype == DType::BF16
            && x.qparams.is_none()
            && w.qparams.is_none()
            && x.device == w.device,
        "GEMV requires same-device unquantized BF16 tensors"
    );
    checked_layout(s, x.meta.strides(), x.handle.size_in_used())?;
    checked_layout(ws, w.meta.strides(), w.handle.size_in_used())?;
    let (k, n) = (s[2], ws[0]);
    let globals = projection_elems();
    let mut dtypes = MatmulElems::from_globals(&globals);
    let output = x.client.empty(
        n.checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("GEMV output overflow"))?,
    );
    // Immutable BF16 [N,K] is only viewed as [K,N]/[1,K], never copied or
    // cast. from_globals keeps F32 stage/register accumulation for BF16 output.
    // This strategy consumes all K in F32, then casts once at its final store;
    // no split-K/global partials, F32 scratch, or separate rounding launch.
    // Fresh output never aliases either input. Setup errors still propagate,
    // with no MMA retry after launch.
    cubek::matmul::launch::launch_ref(
        &Strategy::GemvPlaneParallel(Default::default()),
        &x.client,
        InputBinding::new(binding(&x.handle, [1, k], [k, 1]), globals.lhs),
        InputBinding::new(binding(&w.handle, [k, n], [1, k]), globals.rhs),
        binding(&output, [1, n], [n, 1]),
        &mut dtypes,
    )
    .map_err(|e| anyhow::anyhow!("Breeze BF16 GEMV [1,{k}] x [{n},{k}]^T: {e:?}"))?;
    Ok(Tensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        [1, 1, n].into(),
        output,
        DType::BF16,
    ))
}
pub(super) fn frequencies(c: &RopeConfig, d: usize) -> Result<Vec<f32>> {
    ensure!(
        d > 0 && d % 2 == 0 && c.theta.is_finite() && c.theta > 0.0,
        "invalid RoPE geometry"
    );
    let mut result = Vec::with_capacity(d / 2);
    for i in 0..d / 2 {
        let mut f = 1.0f64 / c.theta.powf((2 * i) as f64 / d as f64);
        match c.kind {
            RopeKind::Default => (),
            RopeKind::Linear => {
                ensure!(c.factor > 0.0, "invalid linear RoPE factor");
                f /= c.factor;
            }
            RopeKind::Llama3 => {
                ensure!(
                    c.factor > 0.0
                        && c.high_freq_factor > c.low_freq_factor
                        && c.low_freq_factor > 0.0
                        && c.original_max_position_embeddings > 0,
                    "invalid Llama3 RoPE scaling"
                );
                let wave = std::f64::consts::TAU / f;
                let context = c.original_max_position_embeddings as f64;
                if wave > context / c.low_freq_factor {
                    f /= c.factor;
                } else if wave >= context / c.high_freq_factor {
                    let smooth = (context / wave - c.low_freq_factor)
                        / (c.high_freq_factor - c.low_freq_factor);
                    f = (1.0 - smooth) * f / c.factor + smooth * f;
                }
            }
        }
        ensure!(f.is_finite() && f > 0.0, "nonfinite RoPE frequency");
        result.push(f as f32);
    }
    Ok(result)
}

struct Layer {
    input: Tensor,
    post: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    gate: Linear,
    up: Linear,
    down: Linear,
}
pub(super) struct Kv {
    key: Tensor,
    value: Tensor,
}
pub(super) struct Decoder {
    config: DecoderConfig,
    layers: Vec<Layer>,
    norm: Tensor,
    freq: Vec<f32>,
}
impl Decoder {
    /// Binder forwards Generator's immutable mapped-prefix lifetime contract.
    pub(super) unsafe fn bind<R: BlobStoreGet>(
        b: &mut Binder<'_, R>,
        c: DecoderConfig,
        prefix: &str,
        qk_norm: bool,
    ) -> Result<Self> {
        let h = c.hidden_size as u64;
        let f = c.intermediate_size as u64;
        let q = (c.num_attention_heads * c.head_dim) as u64;
        let kv = (c.num_key_value_heads * c.head_dim) as u64;
        let d = c.head_dim as u64;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let p = format!("{prefix}.layers.{i}");
            // SAFETY: every fixed slot belongs to the same validated immutable pile.
            layers.push(unsafe {
                Layer {
                    input: b.weight(&format!("{p}.input_layernorm.weight"), [h])?,
                    post: b.weight(&format!("{p}.post_attention_layernorm.weight"), [h])?,
                    q: b.linear(&format!("{p}.self_attn.q_proj.weight"), [q, h])?,
                    k: b.linear(&format!("{p}.self_attn.k_proj.weight"), [kv, h])?,
                    v: b.linear(&format!("{p}.self_attn.v_proj.weight"), [kv, h])?,
                    o: b.linear(&format!("{p}.self_attn.o_proj.weight"), [h, q])?,
                    q_norm: if qk_norm {
                        Some(b.weight(&format!("{p}.self_attn.q_norm.weight"), [d])?)
                    } else {
                        None
                    },
                    k_norm: if qk_norm {
                        Some(b.weight(&format!("{p}.self_attn.k_norm.weight"), [d])?)
                    } else {
                        None
                    },
                    gate: b.linear(&format!("{p}.mlp.gate_proj.weight"), [f, h])?,
                    up: b.linear(&format!("{p}.mlp.up_proj.weight"), [f, h])?,
                    down: b.linear(&format!("{p}.mlp.down_proj.weight"), [h, f])?,
                }
            });
        }
        let norm = unsafe { b.weight(&format!("{prefix}.norm.weight"), [h])? };
        let freq = frequencies(&c.rope, c.head_dim)?;
        Ok(Self {
            config: c,
            layers,
            norm,
            freq,
        })
    }
    /// Invocation-local KV; input at past=0 can contain a whole prefill. Later
    /// calls append one position, producing fresh cache storage, never aliases.
    pub(super) fn forward(
        &self,
        mut x: Tensor,
        past: usize,
        old: &[Kv],
    ) -> Result<(Tensor, Vec<Kv>)> {
        let c = &self.config;
        let t = x.meta.shape()[1];
        ensure!(
            x.meta.shape().as_slice() == [1, t, c.hidden_size] && t > 0 && past + t <= 2048,
            "invalid decoder extent"
        );
        ensure!(
            (past == 0 && old.is_empty()) || (past > 0 && t == 1 && old.len() == self.layers.len()),
            "incomplete decoder KV state"
        );
        let mut cache = Vec::with_capacity(self.layers.len());
        for (i, l) in self.layers.iter().enumerate() {
            let n = ops::norm(&x, &l.input, c.rms_norm_eps as f32, false);
            let mut q = ops::reshape(
                linear(&n, &l.q)?,
                &[1, t, c.num_attention_heads, c.head_dim],
            );
            let mut k = ops::reshape(
                linear(&n, &l.k)?,
                &[1, t, c.num_key_value_heads, c.head_dim],
            );
            let v = ops::reshape(
                linear(&n, &l.v)?,
                &[1, t, c.num_key_value_heads, c.head_dim],
            );
            if let Some(w) = &l.q_norm {
                q = ops::norm(&q, w, c.rms_norm_eps as f32, false);
            }
            if let Some(w) = &l.k_norm {
                k = ops::norm(&k, w, c.rms_norm_eps as f32, false);
            }
            q = ops::rope(&q, past, &self.freq);
            k = ops::rope(&k, past, &self.freq);
            if let Some(s) = old.get(i) {
                ensure!(
                    s.key.meta.shape().as_slice() == [1, past, c.num_key_value_heads, c.head_dim]
                        && s.value.meta.shape() == s.key.meta.shape(),
                    "decoder cache shape mismatch"
                );
            }
            let key = ops::append(&k, old.get(i).map(|s| &s.key));
            let value = ops::append(&v, old.get(i).map(|s| &s.value));
            let a = ops::attention(
                &q,
                &key,
                &value,
                past,
                1.0 / (c.head_dim as f32).sqrt(),
                true,
                None,
            );
            let a = ops::reshape(a, &[1, t, c.num_attention_heads * c.head_dim]);
            let residual = ops::add(&x, &linear(&a, &l.o)?);
            let n = ops::norm(&residual, &l.post, c.rms_norm_eps as f32, false);
            let gate = linear(&n, &l.gate)?;
            let up = linear(&n, &l.up)?;
            x = ops::add(&residual, &linear(&ops::swiglu(&gate, &up), &l.down)?);
            cache.push(Kv { key, value });
        }
        Ok((
            ops::norm(&ops::last(&x), &self.norm, c.rms_norm_eps as f32, false),
            cache,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    #[test]
    fn gemv_only_routes_one_row_and_keeps_f32_accumulation() {
        for k in [1024, 1152, 2048, 6144, 6912, 8192] {
            assert!(single_row(&[1, 1, k]));
            for shape in [[1, 2, k], [2, 1, k], [1, 16, k]] {
                assert!(!single_row(&shape));
            }
        }
        assert!(!single_row(&[1, 1024]));
        assert!(!single_row(&[]));
        for k in [2, 16, 48] {
            assert!(!gemv_vector_takes(&[1, 1, k], 32, 1));
        }
        assert!(gemv_vector_takes(&[1, 1, 1024], 32, 8));
        assert!(gemv_vector_takes(&[1, 1, 1152], 32, 4));
        assert!(!gemv_vector_takes(&[1, 1, 1152], 32, 8));
        assert!(!gemv_vector_takes(&[1, 2, 1024], 32, 8));
        assert!(!gemv_vector_takes(&[1, 1, 0], 32, 1));
        assert!(!gemv_vector_takes(&[1, 1, 1024], 0, 1));
        let globals = projection_elems();
        let types = cubek::matmul::definition::MatmulElems::from_globals(&globals);
        use cubecl::ir::{ElemType, FloatKind, StorageType};
        let bf16 = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
        let f32 = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        assert_eq!(globals.lhs, bf16);
        assert_eq!(globals.rhs, bf16);
        assert_eq!(globals.out, bf16);
        assert_eq!(types.acc_global, bf16);
        assert_eq!(types.lhs_register, globals.lhs);
        assert_eq!(types.rhs_register, globals.rhs);
        assert_eq!(types.acc_register, f32);
        assert_eq!(types.acc_stage, f32);
    }

    #[test]
    fn gemv_descriptor_boundary_rejects_invalid_storage() {
        assert_eq!(
            checked_layout(&[1, 1, 1152], &[1152, 1152, 1], 2304).unwrap(),
            1152
        );
        assert_eq!(
            checked_layout(&[13, 1152], &[1152, 1], 29952).unwrap(),
            14976
        );
        assert!(checked_layout(&[13, 1152], &[1, 13], 29952).is_err());
        assert!(checked_layout(&[13, 1152], &[1152, 1], 29950).is_err());
        assert!(checked_layout(&[0, 1152], &[1152, 1], 0).is_err());
        assert!(checked_layout(&[13, 1152], &[1152], 29952).is_err());
        assert!(checked_layout(&[2, usize::MAX], &[usize::MAX, 1], u64::MAX).is_err());
        assert!(checked_layout(&[u32::MAX as usize + 1], &[1], u64::MAX).is_err());
    }

    #[test]
    #[ignore = "reserved CUDA: bounded GEMV layout/rounding/immutable-input control, not model parity"]
    fn gemv_cuda_layout_rounding_and_immutable_inputs() {
        use cubecl::cuda::CudaDevice;
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let upload = |values: &[bf16], shape: &[usize]| {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            Tensor::new_contiguous(
                client.clone(),
                device.clone(),
                shape.into(),
                client.create_from_slice(&bytes),
                DType::BF16,
            )
        };
        let read = |t: &Tensor| t.client.read_one(t.handle.clone()).unwrap().to_vec();
        // Sparse analytical probes, not a CPU matrix-multiply twin. Tail N=13
        // covers bounds handling; actual FFN and KV pairs also cover production
        // vectorization/plane configurations. Only one pair is live per loop.
        // The late-K nonzero catches truncation, asymmetric columns catch layout
        // mistakes, and two exact terms exercise F32 -> BF16 output rounding.
        for (k, n) in [
            (1024, 13),
            (1152, 13),
            (2048, 13),
            (6144, 13),
            (6912, 13),
            (8192, 13),
            (1024, 8192),
            (8192, 1024),
            (2048, 6144),
            (6144, 2048),
            (1024, 256),
            (2048, 1024),
        ] {
            let mut xv = vec![bf16::ZERO; k];
            xv[0] = bf16::from_f32(3.0 / 512.0);
            xv[k - 1] = bf16::ONE;
            let mut wv = vec![bf16::ZERO; n * k];
            for row in 0..n {
                wv[row * k] = bf16::ONE;
                // Repeating exactly representable column values avoid adding
                // a separate input-quantization error at large production N.
                wv[row * k + k - 1] = bf16::from_f32(1.0 + (row % 13) as f32 / 8.0);
            }
            let x = upload(&xv, &[1, 1, k]);
            let w = upload(&wv, &[n, k]);
            assert!(
                gemv_eligible(&x),
                "production/tail fixture must exercise GEMV"
            );
            let before_x = read(&x);
            let before_w = read(&w);
            let out = linear_bf16(&x, &w).unwrap();
            assert_eq!(out.dtype, DType::BF16);
            assert_eq!(out.meta.shape().as_slice(), [1, 1, n]);
            let values = read(&out);
            assert_eq!(values.len(), n * 2);
            for row in 0..n {
                let expected =
                    bf16::from_f32(1.0 + (row % 13) as f32 / 8.0 + 3.0 / 512.0).to_bits();
                assert_eq!(
                    u16::from_le_bytes([values[2 * row], values[2 * row + 1]]),
                    expected,
                    "k={k}, n={n}, row={row}"
                );
            }
            assert_eq!(read(&x), before_x);
            assert_eq!(read(&w), before_w);
            let mut wrong_dtype = w.clone();
            wrong_dtype.dtype = DType::F16;
            assert!(linear_bf16(&x, &wrong_dtype).is_err());
        }
        // Preserve small operator fixtures with a pre-launch old-path choice.
        let x = upload(&[bf16::from_f32(2.0), bf16::from_f32(3.0)], &[1, 1, 2]);
        let w = upload(&[bf16::ONE; 6], &[3, 2]);
        assert!(!gemv_eligible(&x));
        let expected: Vec<u8> = [bf16::from_f32(5.0); 3]
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        assert_eq!(read(&linear_bf16(&x, &w).unwrap()), expected);
    }

    #[test]
    #[ignore = "reserved CUDA: actual-stride F32 accumulation and direct BF16 store, not model parity"]
    fn gemv_cuda_direct_bf16_preserves_f32_accumulation() {
        use cubecl::cuda::CudaDevice;
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let (k, n) = (8192usize, 1024usize);
        let plane = client.properties().hardware.plane_size_max as usize;
        // Both inputs are unquantized BF16: reproduce Cubek's maximum valid
        // input vector width, rather than assume which lanes see the terms.
        let vector = client
            .io_optimized_vector_sizes(2)
            .filter(|&v| gemv_vector_takes(&[1, 1, k], plane, v))
            .max()
            .expect("production fixture requires a valid GEMV vector width");
        let stride = plane.checked_mul(vector).unwrap();
        assert!(k.is_multiple_of(stride) && k / stride >= 4);
        let xv = vec![bf16::ONE; k];
        let mut wv = vec![bf16::ZERO; n * k];
        // Column 0 is cube 0 / plane 0, so its segment order is unswizzled.
        // These terms occupy one lane/vector component on four successive
        // iterations. F32 retains 3/512 at 256; a BF16 loop accumulator loses
        // it and would finish at 1.0 (0x3f80), not the required 0x3f81.
        for (segment, term) in [256.0, 3.0 / 512.0, -256.0, 1.0]
            .into_iter()
            .enumerate()
        {
            wv[segment * stride] = bf16::from_f32(term);
        }
        let upload = |values: &[bf16], shape: &[usize]| {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            Tensor::new_contiguous(
                client.clone(),
                device.clone(),
                shape.into(),
                client.create_from_slice(&bytes),
                DType::BF16,
            )
        };
        let read = |t: &Tensor| t.client.read_one(t.handle.clone()).unwrap().to_vec();
        let x = upload(&xv, &[1, 1, k]);
        let w = upload(&wv, &[n, k]);
        assert!(gemv_eligible(&x), "precision fixture must exercise GEMV");
        let before_x = read(&x);
        let before_w = read(&w);
        let out = linear_bf16(&x, &w).unwrap();
        assert_eq!(out.dtype, DType::BF16);
        assert_eq!(out.meta.shape().as_slice(), [1, 1, n]);
        let values = read(&out);
        assert_eq!(values.len(), n * 2);
        assert_eq!(bf16::from_f32(1.0 + 3.0 / 512.0).to_bits(), 0x3f81);
        assert_eq!(u16::from_le_bytes([values[0], values[1]]), 0x3f81);
        assert!(values[2..].iter().all(|&byte| byte == 0));
        assert_eq!(read(&x), before_x);
        assert_eq!(read(&w), before_w);
    }

    #[test]
    fn frequency_families_are_not_outer_config_aliases() {
        let mut c = RopeConfig {
            theta: 1e6,
            kind: RopeKind::Default,
            factor: 1.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let plain = frequencies(&c, 128).unwrap();
        assert_eq!(plain[0], 1.0);
        c.kind = RopeKind::Linear;
        c.factor = 8.0;
        assert_eq!(frequencies(&c, 128).unwrap()[0], 0.125);
        c.kind = RopeKind::Llama3;
        c.theta = 500000.0;
        c.factor = 32.0;
        c.low_freq_factor = 0.001953125;
        c.high_freq_factor = 0.0078125;
        c.original_max_position_embeddings = 16;
        let depth = frequencies(&c, 128).unwrap();
        assert_eq!(depth[0], 1.0);
        assert!(depth[63] < plain[63]);
        assert!(frequencies(&c, 127).is_err());
    }
}
