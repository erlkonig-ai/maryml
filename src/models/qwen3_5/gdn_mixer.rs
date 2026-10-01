//! One unmasked Qwen3.5 GatedDeltaNet mixer, not a decoder layer/backbone.
//!
//! Reference: Transformers 5.2.0 modeling_qwen3_5.py:445–626. Nine native
//! BF16 weight slots; five bias-free [out,in] projections; resident BF16
//! hidden -> convolution/SiLU -> DeltaNet -> ordinary gated RMSNorm -> output.
//! Prefill starts fresh; decode consumes BOTH histories and exactly one token.
//! Masks are refused, including B=1, rather than inheriting the reference's
//! B>1 padding-mask shortcut. Text input norm/residual/MLP are outside this unit.
//!
//! Projection uses a forced, fixed BF16 MMA blueprint with F32 accumulation
//! when K is divisible by 16 and N by 8. Other shapes retain the serial F32
//! GPU dot product. Neither dispatch nor the reduction tile depends on B/T/M.
//! This is an experimental accelerated path: execution/reproducibility and
//! performance must be measured before promotion; cross-device identity is
//! not claimed. It does not call Burn's autotuned matmul.
//! Inputs/weights are never uploaded, read on the host, or mutated here.
//! Resident tensor producers must honor CubeCL's ordinary client/handle and
//! stream-ordering contract. Result covers descriptor errors, not allocation,
//! CUDA initialization, kernel compilation or asynchronous driver failures;
//! those retain the upstream runtime's panic/error behavior.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use serde::{Deserialize, Serialize};
use triblespace::core::{
    blob::{Blob, encodings::tensor::{Tensor as NativeTensor, elements::BF16}},
    inline::{Inline, encodings::hash::Handle},
    repo::BlobStoreGet,
};
use super::{deltanet::{self, DeltaNetInputs}, gdn_ops};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;

pub type CudaTensor = CubeTensor<CudaRuntime>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct GdnConfig {
    pub hidden: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub epsilon: f32,
}

impl GdnConfig {
    pub fn validate(self) -> Result<(), String> {
        if self.hidden == 0 || self.key_heads == 0 || self.value_heads == 0
            || self.value_heads % self.key_heads != 0
            || !(1..=256).contains(&self.key_dim)
            || !(1..=256).contains(&self.value_dim)
            || !(1..=16).contains(&self.conv_kernel)
            || !self.epsilon.is_finite() || self.epsilon <= 0.0
        {
            return Err("GDN requires nonempty H/Hk/Hv, Hk divides Hv, K/V<=256, conv<=16, positive finite epsilon".into());
        }
        // Prove every weight shape before using unchecked host products.
        let k = count(&[self.key_heads, self.key_dim])?;
        let v = count(&[self.value_heads, self.value_dim])?;
        let c = k.checked_mul(2).and_then(|x| x.checked_add(v))
            .filter(|&x| x <= u32::MAX as usize).ok_or("QKV width overflow")?;
        count(&[c, self.hidden])?;
        count(&[v, self.hidden])?;
        count(&[self.value_heads, self.hidden])?;
        count(&[c, self.conv_kernel])?;
        Ok(())
    }
    fn key_width(self) -> usize { self.key_heads * self.key_dim }
    fn value_width(self) -> usize { self.value_heads * self.value_dim }
    fn channels(self) -> usize { 2 * self.key_width() + self.value_width() }
}

/// Exactly the nine selected native tensor handles, not a model catalogue.
/// Obtain these through typed leaf queries at the consuming module's slots.
/// Handles/IDs are opaque; no identity is reconstructed from a name or shape.
pub struct GdnSlots {
    pub qkv: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub z: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub a: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub b: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub out: Inline<Handle<NativeTensor<BF16, 2>>>,
    pub conv: Inline<Handle<NativeTensor<BF16, 3>>>,
    pub a_log: Inline<Handle<NativeTensor<BF16, 1>>>,
    pub dt_bias: Inline<Handle<NativeTensor<BF16, 1>>>,
    pub norm: Inline<Handle<NativeTensor<BF16, 1>>>,
}

/// Private fields prevent replacement of a validated, read-only weight.
pub struct GdnMixer {
    config: GdnConfig,
    qkv: CudaTensor,
    z: CudaTensor,
    a: CudaTensor,
    b: CudaTensor,
    out: CudaTensor,
    conv: CudaTensor,
    a_log: CudaTensor,
    dt_bias: CudaTensor,
    norm: CudaTensor,
}

pub struct GdnState {
    /// Raw projected QKV, BF16 [B,C,Kconv], oldest to newest.
    pub conv: CudaTensor,
    /// F32 [B,Hv,K,V], not rounded to BF16 between tokens.
    pub recurrent: CudaTensor,
}

pub struct GdnOutput {
    pub hidden: CudaTensor,
    pub state: GdnState,
}

impl GdnMixer {
    /// Bind fixed typed slots through a local or exact-acquiring reader.
    ///
    /// # Safety
    /// Every returned leaf must originate in a genuine validated native pile.
    /// Its backing file must remain an immutable append-only prefix
    /// through CUDA runtime teardown, including the partial page preceding each
    /// payload. No rewrite, truncation, in-place mutation, or external writer
    /// violating that premise is permitted. This obligation is passed directly
    /// to the reviewed unsafe binder; a MmapRaw owner alone is NOT provenance.
    /// A generic reader is not provenance: heap blobs remain refused and merely
    /// wrapping an arbitrary MmapRaw does not satisfy this unsafe obligation.
    /// A late shape/budget error can leave earlier registrations runtime-owned.
    pub unsafe fn from_pile<R: BlobStoreGet>(
        snapshot: &R,
        slots: GdnSlots,
        config: GdnConfig,
        aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        config.validate()?;
        // Typed handles are consumed here, without erasure, legacy fallback,
        // upload, F16 conversion, hash-join lookup, or whole-model enumeration.
        let qkv: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.qkv).map_err(|e| e.to_string())?;
        let z: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.z).map_err(|e| e.to_string())?;
        let a: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.a).map_err(|e| e.to_string())?;
        let b: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.b).map_err(|e| e.to_string())?;
        let out: Blob<NativeTensor<BF16, 2>> = snapshot.get(slots.out).map_err(|e| e.to_string())?;
        let conv: Blob<NativeTensor<BF16, 3>> = snapshot.get(slots.conv).map_err(|e| e.to_string())?;
        let a_log: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.a_log).map_err(|e| e.to_string())?;
        let dt_bias: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.dt_bias).map_err(|e| e.to_string())?;
        let norm: Blob<NativeTensor<BF16, 1>> = snapshot.get(slots.norm).map_err(|e| e.to_string())?;
        // SAFETY: the caller guarantees the genuine snapshot's immutable
        // file-backed prefix for each exact native leaf, including page prefix.
        let result = unsafe {
            Self {
                config,
                qkv: aliases.bind_pile_leaf(qkv)?,
                z: aliases.bind_pile_leaf(z)?,
                a: aliases.bind_pile_leaf(a)?,
                b: aliases.bind_pile_leaf(b)?,
                out: aliases.bind_pile_leaf(out)?,
                conv: aliases.bind_pile_leaf(conv)?,
                a_log: aliases.bind_pile_leaf(a_log)?,
                dt_bias: aliases.bind_pile_leaf(dt_bias)?,
                norm: aliases.bind_pile_leaf(norm)?,
            }
        };
        result.check_weights()?;
        Ok(result)
    }

    fn check_weights(&self) -> Result<(), String> {
        let c = self.config;
        for (name, tensor, shape) in [
            ("qkv", &self.qkv, vec![c.channels(), c.hidden]),
            ("z", &self.z, vec![c.value_width(), c.hidden]),
            ("a", &self.a, vec![c.value_heads, c.hidden]),
            ("b", &self.b, vec![c.value_heads, c.hidden]),
            ("out", &self.out, vec![c.hidden, c.value_width()]),
            ("conv", &self.conv, vec![c.channels(), 1, c.conv_kernel]),
            ("A_log", &self.a_log, vec![c.value_heads]),
            ("dt_bias", &self.dt_bias, vec![c.value_heads]),
            ("norm", &self.norm, vec![c.value_dim]),
        ] {
            check(tensor, &self.qkv, name, &shape, DType::BF16)?;
            if tensor.handle.can_mut() {
                return Err(format!("{name}: read-only pile alias required"));
            }
        }
        Ok(())
    }

    /// Fresh unmasked sequence, T in 1..=256.
    pub fn prefill(&self, hidden: &CudaTensor, mask: Option<&CudaTensor>) -> Result<GdnOutput, String> {
        self.forward(hidden, None, mask)
    }

    /// One continuation token. Neither caller-owned state buffer is mutated.
    pub fn decode(&self, hidden: &CudaTensor, state: &GdnState, mask: Option<&CudaTensor>) -> Result<GdnOutput, String> {
        self.forward(hidden, Some(state), mask)
    }

    fn forward(&self, hidden: &CudaTensor, state: Option<&GdnState>, mask: Option<&CudaTensor>) -> Result<GdnOutput, String> {
        if mask.is_some() { return Err("GDN mixer supports unmasked inputs only".into()); }
        let shape = hidden.meta.shape().as_slice();
        if shape.len() != 3 { return Err("hidden must be [B,T,H]".into()); }
        let (batch, tokens) = (shape[0], shape[1]);
        let c = self.config;
        if !(1..=256).contains(&tokens) || (state.is_some() && tokens != 1) {
            return Err("prefill requires T=1..256; decode requires exactly T=1".into());
        }
        check(hidden, &self.qkv, "hidden", &[batch, tokens, c.hidden], DType::BF16)?;
        // Check ALL derived allocations and incoming state before any launch.
        count(&[batch, tokens, c.channels()])?;
        count(&[batch, c.channels(), c.conv_kernel])?;
        count(&[batch, c.value_heads, c.key_dim, c.value_dim])?;
        if let Some(s) = state {
            check(&s.conv, hidden, "conv state", &[batch, c.channels(), c.conv_kernel], DType::BF16)?;
            check(&s.recurrent, hidden, "recurrent state", &[batch, c.value_heads, c.key_dim, c.value_dim], DType::F32)?;
        }
        let qkv = project(hidden, &self.qkv)?;
        let z = reshape(project(hidden, &self.z)?, &[batch, tokens, c.value_heads, c.value_dim]);
        let a = project(hidden, &self.a)?;
        let b = project(hidden, &self.b)?;
        let conv = gdn_ops::causal_conv_silu(&qkv, &self.conv, state.map(|s| &s.conv))?;
        let query = reshape(channels(&conv.output, 0, c.key_width())?, &[batch, tokens, c.key_heads, c.key_dim]);
        let key = reshape(channels(&conv.output, c.key_width(), c.key_width())?, &[batch, tokens, c.key_heads, c.key_dim]);
        let value = reshape(channels(&conv.output, 2 * c.key_width(), c.value_width())?, &[batch, tokens, c.value_heads, c.value_dim]);
        let core = deltanet::gated_delta(DeltaNetInputs {
            query: &query, key: &key, value: &value, a: &a, b: &b,
            a_log: &self.a_log, dt_bias: &self.dt_bias,
            initial_state: state.map(|s| &s.recurrent),
        })?;
        let gated = gdn_ops::gated_rms_norm(&core.output, &z, &self.norm, c.epsilon)?;
        let gated = reshape(gated, &[batch, tokens, c.value_width()]);
        let output = project(&gated, &self.out)?;
        Ok(GdnOutput { hidden: output, state: GdnState { conv: conv.history, recurrent: core.state } })
    }
}

fn count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |n, &d| {
        if d == 0 { None } else { n.checked_mul(d) }
    }).filter(|&n| n <= u32::MAX as usize).ok_or_else(|| "empty or overflowing u32 shape".into())
}

fn check(tensor: &CudaTensor, like: &CudaTensor, name: &str, shape: &[usize], dtype: DType) -> Result<usize, String> {
    if tensor.meta.shape().as_slice() != shape || tensor.meta.strides().len() != shape.len()
        || tensor.dtype != dtype || tensor.qparams.is_some() || tensor.device != like.device
    {
        return Err(format!("{name}: wrong shape/dtype/device or quantized storage"));
    }
    let n = count(shape)?;
    let mut stride = 1;
    for (axis, &d) in shape.iter().enumerate().rev() {
        if d > 1 && tensor.meta.strides()[axis] != stride {
            return Err(format!("{name}: contiguous row-major storage required"));
        }
        stride *= d;
    }
    let bytes = n.checked_mul(if dtype == DType::BF16 { 2 } else { 4 }).ok_or("byte extent overflow")?;
    if tensor.handle.size_in_used() < bytes as u64 { return Err(format!("{name}: insufficient storage")); }
    Ok(n)
}

// Called only after exact contiguous element counts have been proved.
fn reshape(tensor: CudaTensor, shape: &[usize]) -> CudaTensor {
    CubeTensor::new_contiguous(tensor.client, tensor.device, shape.into(), tensor.handle, tensor.dtype)
}

#[cube(launch_unchecked)]
fn projection_kernel(
    input: &Array<bf16>, weight: &Array<bf16>, output: &mut Array<bf16>,
    elements: usize, input_width: usize, output_width: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < elements {
        let row = i / output_width;
        let out = i % output_width;
        let mut sum = 0.0f32;
        for k in 0..input_width {
            sum += f32::cast_from(input[row * input_width + k])
                * f32::cast_from(weight[out * input_width + k]);
        }
        output[i] = bf16::cast_from(sum);
    }
}

/// Resident BF16 linear projection: [B,T,I] x [O,I] -> [B,T,O].
/// K/N-only dispatch; no CPU fallback, autotuning, or split-K reduction.
pub fn project(input: &CudaTensor, weight: &CudaTensor) -> Result<CudaTensor, String> {
    project_impl(input, weight, false)
}

fn project_impl(input: &CudaTensor, weight: &CudaTensor, serial: bool) -> Result<CudaTensor, String> {
    let s = input.meta.shape().as_slice();
    let w = weight.meta.shape().as_slice();
    if s.len() != 3 || w.len() != 2 || s[2] != w[1] {
        return Err("projection requires [B,T,I] and [O,I]".into());
    }
    let ni = check(input, input, "projection input", s, DType::BF16)?;
    let nw = check(weight, input, "projection weight", w, DType::BF16)?;
    let shape = [s[0], s[1], w[0]];
    let no = count(&shape)?;
    if !serial && fixed_mma_shape(s[2], w[0]) {
        return project_mma(input, weight, shape, no);
    }
    let output = input.client.empty(no * 2);
    let cube = CubeDim::new_1d(64);
    // SAFETY: checked contiguous BF16 inputs, u32-indexed extents and disjoint
    // fresh output; each thread owns one output and reads only bounded inputs.
    unsafe {
        projection_kernel::launch_unchecked::<CudaRuntime>(
            &input.client, cubecl::calculate_cube_count_elemwise(&input.client, no, cube), cube,
            ArrayArg::from_raw_parts(input.handle.clone(), ni),
            ArrayArg::from_raw_parts(weight.handle.clone(), nw),
            ArrayArg::from_raw_parts(output.clone(), no), no, s[2], w[0],
        );
    }
    Ok(CubeTensor::new_contiguous(input.client.clone(), input.device.clone(), shape.into(), output, DType::BF16))
}

fn fixed_mma_shape(k: usize, n: usize) -> bool {
    k > 0 && n > 0 && k.is_multiple_of(16) && n.is_multiple_of(8)
}

fn projection_elems() -> cubek::matmul::definition::MatmulGlobalElems {
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    let bf16 = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
    cubek::matmul::definition::MatmulGlobalElems {
        lhs: bf16,
        rhs: bf16,
        out: StorageType::Scalar(ElemType::Float(FloatKind::F32)),
    }
}

/// Dependency trace (cubek-matmul 0.2.0): Strategy::SimpleCyclicMma passes
/// Forced through stamp_kind unchanged; SimpleAlgorithm::expand_blueprint
/// clones Forced directly, never entering either M-dependent infer function.
/// launch_tiling chooses IO vector widths from contiguous K/K/N and strides
/// K/K/N, not M. Rank-2 row/column-major bindings are not materialized copies.
/// The row-major global grid owns disjoint M/N partitions; SimpleMatmul walks
/// K stages within each partition, with no atomic or split-K output reduction.
/// Bounds checks are always enabled, even for complete tiles, so crossing an
/// M tile boundary does not switch to a different arithmetic configuration.
fn projection_blueprint(m: usize, n: usize, k: usize) -> Result<cubek::matmul::definition::TilingBlueprint, String> {
    use cubek::matmul::{
        components::{stage::PartitionBuffering, tile::TileMatmulKind},
        definition::{MatmulProblem, TilingBlueprint, TilingScheme},
    };
    use cubek::std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};
    let problem = MatmulProblem::from_shapes_and_strides(
        [m, k].into(), [k, n].into(), [m, n].into(),
        [k, 1].into(), [1, k].into(), [n, 1].into(),
        projection_elems(), cubecl::ir::AddressType::U32, None, None,
    ).map_err(|e| format!("fixed projection problem: {e:?}"))?;
    let scheme = TilingScheme::builder()
        .with_tile_size((16, 8, 16).into())
        .with_partition_size((2, 1, 4).into())
        .with_stage_size((2, 2, 1).into())
        .build().map_err(str::to_owned)?;
    let grid = HypercubeBlueprint::builder()
        .global_order(GlobalOrder::RowMajor)
        .cube_count_strategy(CubeCountStrategy::FromProblem)
        .build();
    let mut blueprint = TilingBlueprint::builder(TileMatmulKind::Mma, scheme, 32, &problem)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_blueprint(grid)
        .build();
    blueprint.check_m_bounds = true;
    blueprint.check_n_bounds = true;
    blueprint.check_k_bounds = true;
    Ok(blueprint)
}

#[cube(launch_unchecked)]
fn projection_round(input: &Array<f32>, output: &mut Array<bf16>, n: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n { output[i] = bf16::cast_from(input[i]); }
}

// Called only after the public descriptor/device/extent checks. Weight storage
// remains the original aliased BF16 [N,K] handle: [K,N] strides [1,K] is a view,
// not a transpose/cast/copy. F32 is accumulator/output scratch, never weights.
fn project_mma(input: &CudaTensor, weight: &CudaTensor, shape: [usize; 3], no: usize) -> Result<CudaTensor, String> {
    use cubek::matmul::{definition::MatmulElems, launch::Strategy, routines::BlueprintStrategy};
    use cubek::std::InputBinding;
    fn binding(handle: &cubecl::server::Handle, shape: [usize; 2], strides: [usize; 2]) -> TensorBinding<CudaRuntime> {
        TensorBinding {
            handle: handle.clone().binding(), shape: shape.into(), strides: strides.into(),
            runtime: core::marker::PhantomData,
        }
    }
    let (m, n, k) = (shape[0] * shape[1], shape[2], input.meta.shape()[2]);
    let globals = projection_elems();
    let mut dtypes = MatmulElems::from_globals(&globals);
    let scratch = input.client.empty(no.checked_mul(4).ok_or("F32 output byte extent overflow")?);
    let strategy = Strategy::SimpleCyclicMma(BlueprintStrategy::Forced(projection_blueprint(m, n, k)?));
    cubek::matmul::launch::launch_ref(
        &strategy, &input.client,
        InputBinding::new(binding(&input.handle, [m, k], [k, 1]), globals.lhs),
        InputBinding::new(binding(&weight.handle, [k, n], [1, k]), globals.rhs),
        binding(&scratch, [m, n], [n, 1]), &mut dtypes,
    ).map_err(|e| format!("fixed BF16 projection [{m},{k}] x [{n},{k}]^T: {e:?}"))?;
    // Setup errors are not retried through another algorithm: that could make
    // arithmetic depend on resources or M. Unsupported K/N is selected above.
    let output = input.client.empty(no * 2);
    let cube = CubeDim::new_1d(128);
    // SAFETY: both handles own `no` elements, input F32/output BF16, one writer.
    unsafe {
        projection_round::launch_unchecked::<CudaRuntime>(
            &input.client, cubecl::calculate_cube_count_elemwise(&input.client, no, cube), cube,
            ArrayArg::from_raw_parts(scratch, no), ArrayArg::from_raw_parts(output.clone(), no), no,
        );
    }
    Ok(CubeTensor::new_contiguous(input.client.clone(), input.device.clone(), shape.into(), output, DType::BF16))
}

#[cube(launch_unchecked)]
fn channel_kernel(input: &Array<bf16>, output: &mut Array<bf16>, n: usize, source_width: usize, start: usize, width: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n { output[i] = input[(i / width) * source_width + start + i % width]; }
}

fn channels(input: &CudaTensor, start: usize, width: usize) -> Result<CudaTensor, String> {
    let s = input.meta.shape().as_slice();
    if s.len() != 3 || width == 0 || start.checked_add(width).is_none_or(|end| end > s[2]) {
        return Err("invalid channel selection".into());
    }
    let ni = check(input, input, "channels", s, DType::BF16)?;
    let shape = [s[0], s[1], width];
    let n = count(&shape)?;
    let output = input.client.empty(n * 2);
    let cube = CubeDim::new_1d(64);
    // SAFETY: checked contiguous source, bounded channel range and fresh output.
    unsafe {
        channel_kernel::launch_unchecked::<CudaRuntime>(
            &input.client, cubecl::calculate_cube_count_elemwise(&input.client, n, cube), cube,
            ArrayArg::from_raw_parts(input.handle.clone(), ni),
            ArrayArg::from_raw_parts(output.clone(), n), n, s[2], start, width,
        );
    }
    Ok(CubeTensor::new_contiguous(input.client.clone(), input.device.clone(), shape.into(), output, DType::BF16))
}

#[cfg(test)]
mod projection_tests {
    use super::*;
    use cubecl::cuda::CudaDevice;

    #[test]
    fn forced_projection_blueprint_is_independent_of_rows_and_tails() {
        let expected = projection_blueprint(1, 40, 48).unwrap();
        for m in [2, 15, 16, 17, 63, 64, 65, 116, 130, 256] {
            assert_eq!(projection_blueprint(m, 40, 48).unwrap(), expected);
        }
        assert!(expected.check_m_bounds && expected.check_n_bounds && expected.check_k_bounds);
        assert_eq!(expected.tiling_scheme.elements_per_stage_along_m(), 64);
        assert_eq!(expected.tiling_scheme.elements_per_stage_along_n(), 16);
        assert_eq!(expected.tiling_scheme.elements_per_stage_along_k(), 64);
    }

    #[test]
    fn projection_shape_dispatch_is_explicit_and_accumulation_is_f32() {
        for (k, n) in [(16, 8), (48, 40), (4096, 32), (4096, 12288), (12288, 4096)] {
            assert!(fixed_mma_shape(k, n));
        }
        for (k, n) in [(0, 8), (16, 0), (7, 5), (17, 8), (16, 9)] {
            assert!(!fixed_mma_shape(k, n));
        }
        let dtypes = cubek::matmul::definition::MatmulElems::from_globals(&projection_elems());
        assert_eq!(dtypes.acc_stage, projection_elems().out);
        assert_eq!(dtypes.acc_register, projection_elems().out);
        assert_eq!(dtypes.lhs_register, projection_elems().lhs);
        assert_eq!(dtypes.rhs_register, projection_elems().rhs);
    }

    #[cube(launch_unchecked)]
    fn fixture_values(out: &mut Array<bf16>, rows: &Array<u32>, n: usize, width: usize) {
        let i = ABSOLUTE_POS as usize;
        if i < n {
            // Numeric fixture generation is GPU-only. Host rows are integer
            // identities, not activation or weight payloads.
            let row = rows[i / width] as usize;
            let phase = f32::cast_from(((i % width) * 13 + row * 7) % 251) / 32.0f32;
            out[i] = bf16::cast_from(phase.sin() * 0.125f32);
        }
    }

    fn fixture(shape: &[usize], rows: &[u32]) -> CudaTensor {
        let device = CudaDevice { index: 0 };
        let client = CudaRuntime::client(&device);
        let n = count(shape).unwrap();
        let width = *shape.last().unwrap();
        assert_eq!(rows.len() * width, n);
        let row_bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        let row_handle = client.create_from_slice(&row_bytes);
        let handle = client.empty(n * 2);
        let cube = CubeDim::new_1d(128);
        // SAFETY: integer row table has one entry per output row, exact sizes.
        unsafe {
            fixture_values::launch_unchecked::<CudaRuntime>(
                &client, cubecl::calculate_cube_count_elemwise(&client, n, cube), cube,
                ArrayArg::from_raw_parts(handle.clone(), n),
                ArrayArg::from_raw_parts(row_handle, rows.len()), n, width,
            );
        }
        CubeTensor::new_contiguous(client, device, shape.into(), handle, DType::BF16)
    }

    fn bytes(tensor: CudaTensor) -> Vec<u8> {
        tensor.client.read_one(tensor.handle).unwrap().to_vec()
    }

    fn compare_gpu_oracle(input: &CudaTensor, weight: &CudaTensor) -> Vec<u8> {
        let actual = bytes(project(input, weight).unwrap());
        assert_eq!(actual, bytes(project(input, weight).unwrap()), "fresh-call byte identity");
        let oracle = bytes(project_impl(input, weight, true).unwrap());
        let mut differing = 0usize;
        let mut worst = 0.0f32;
        let mut outside = 0usize;
        for (a, b) in actual.chunks_exact(2).zip(oracle.chunks_exact(2)) {
            differing += usize::from(a != b);
            let a = bf16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
            let b = bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
            assert!(a.is_finite() && b.is_finite());
            // Readback diagnostics only: the reference dot products ran on
            // GPU. This fixture's fixed budget is not a model error envelope,
            // and exact equality to the old serial arithmetic is not required.
            let scaled = (a - b).abs() / (0.001 + 0.01 * b.abs());
            worst = worst.max(scaled);
            outside += usize::from(scaled > 1.0);
        }
        eprintln!("projection {:?} x {:?}: serial differing_words={differing}, worst_scaled={worst}, outside={outside}", input.meta.shape(), weight.meta.shape());
        assert_eq!(outside, 0, "fixed synthetic GPU-oracle budget");
        actual
    }

    #[test]
    #[ignore = "reserved CUDA: fixed projection layout/repeat/batch experiment"]
    fn fixed_projection_matches_gpu_oracle_and_preserves_row_bits() {
        // Nonsquare transposed weight, N/K stage tails, and the explicit small
        // unaligned GPU fallback. M crosses both tile and stage boundaries.
        for (k, n) in [(48, 40), (7, 5)] {
            let weight = fixture(&[n, k], &(0..n as u32).collect::<Vec<_>>());
            let single = fixture(&[1, 1, k], &[777]);
            let wanted = compare_gpu_oracle(&single, &weight);
            for m in [17, 64, 65, 130] {
                let mut rows: Vec<u32> = (0..m as u32).collect();
                for position in [0, 15, m - 1] {
                    rows[position] = 777;
                    let input = fixture(&[1, m, k], &rows);
                    let output = compare_gpu_oracle(&input, &weight);
                    assert_eq!(&output[position * n * 2..(position + 1) * n * 2], &wanted);
                    rows.reverse();
                    let reversed = fixture(&[1, m, k], &rows);
                    let reversed = bytes(project(&reversed, &weight).unwrap());
                    for row in 0..m {
                        assert_eq!(&output[row * n * 2..(row + 1) * n * 2], &reversed[(m - 1 - row) * n * 2..(m - row) * n * 2]);
                    }
                    rows.reverse();
                }
            }
            let rows: Vec<u32> = (0..130).collect();
            let a = fixture(&[1, 130, k], &rows);
            // Reinterpret the SAME GPU bytes, no activation copy, with another
            // valid batch/token factorization.
            let b = reshape(a.clone(), &[2, 65, k]);
            assert_eq!(bytes(project(&a, &weight).unwrap()), bytes(project(&b, &weight).unwrap()));
        }
    }

    #[test]
    #[ignore = "reserved CUDA: finite real WeMM projection dimensions, not model retrieval"]
    fn fixed_projection_real_dimensions_match_gpu_oracle() {
        for (k, n, m) in [(4096, 32, 65), (4096, 4096, 2), (4096, 8192, 2), (4096, 12288, 2), (12288, 4096, 2)] {
            let weight = fixture(&[n, k], &(0..n as u32).collect::<Vec<_>>());
            let rows: Vec<u32> = (0..m as u32).collect();
            let input = fixture(&[1, m, k], &rows);
            let output = compare_gpu_oracle(&input, &weight);
            let single = fixture(&[1, 1, k], &[rows[m - 1]]);
            assert_eq!(&output[(m - 1) * n * 2..m * n * 2], &bytes(project(&single, &weight).unwrap()));
        }
    }
}
