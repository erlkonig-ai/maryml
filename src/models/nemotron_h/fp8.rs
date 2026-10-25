//! One operation's static E4M3FN W8A8 projection, without bias or model state.
//!
//! Contract: q=RN_E4M3_SATFINITE(F32(BF16(x))/input_scale); then an E4M3 MMA
//! dot with F32 accumulation; then BF16(dot * F32(input_scale*weight_scale)).
//! The exported scales are dequantization multipliers. Division by that scale
//! is a declared native choice, not assumed bitwise equivalent to ModelOpt's
//! original amax/reciprocal fake-quant arithmetic or BF16-dequantized GEMM.
//! Weight bytes are never cast, transposed, calibrated or expanded to BF16.
//! Each operation retains its own scales, even when activations are shared.
//!
//! This experimental component requires an independent GPU reference before
//! model acceptance. Setup/descriptor/budget errors return Err; allocation,
//! compilation and asynchronous CUDA failures retain runtime behavior. The
//! owning session must fence before committing state. Inputs remain immutable;
//! calls use fresh output and follow the ordinary CubeCL handle/stream contract.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    e4m3,
    features::{MmaConfig, TypeUsage},
    ir::{ElemType, FloatKind, StorageType},
    prelude::*,
    server::Handle,
};
use cubek::{
    matmul::{
        components::{stage::PartitionBuffering, tile::TileMatmulKind},
        definition::{MatmulElems, MatmulGlobalElems, MatmulProblem, TilingBlueprint, TilingScheme},
        launch::Strategy,
        routines::BlueprintStrategy,
    },
    std::{InputBinding, cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint}},
};
use half::bf16;

#[cfg(all(test, feature = "import"))]
#[path = "real_layers.rs"]
mod real_layers;

pub type CudaTensor = CubeTensor<CudaRuntime>;
type Result<T> = std::result::Result<T, String>;
const THREADS: u32 = 128;

/// Immutable, device-owned bytes for one linear operation [out,in]. Native
/// Tensor<E4M3,2> loading can supply its payload to upload; graph queries and
/// source-scale pairing belong to that loader. No raw mutable handle escapes.
pub struct Projection {
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    weight: Handle,
    shape: [usize; 2],
    input_scale: f32,
    weight_scale: f32,
    alpha: f32,
}

impl Projection {
    /// Copy exactly N*K source E4M3FN bytes once. Budget counts logical resident
    /// weight bytes, excluding host staging/runtime allocation overhead. NaN
    /// weight codes are preserved. Scales and their F32 product must be finite
    /// positive; overflow/underflow-to-zero is unsupported by this epilogue.
    pub fn upload(
        device: CudaDevice,
        shape: [usize; 2],
        e4m3fn: &[u8],
        input_scale: f32,
        weight_scale: f32,
        max_weight_bytes: usize,
    ) -> Result<Self> {
        let bytes = weight_elements(shape)?;
        let alpha = scale_product(input_scale, weight_scale)?;
        if e4m3fn.len() != bytes || bytes > max_weight_bytes {
            return Err("FP8 weight byte length or resident budget mismatch".into());
        }
        let client = CudaRuntime::client(&device);
        capability(&client)?;
        let weight = client.create_from_slice(e4m3fn);
        Ok(Self { client, device, weight, shape, input_scale, weight_scale, alpha })
    }

    pub fn shape(&self) -> [usize; 2] { self.shape }
    pub fn scales(&self) -> (f32, f32) { (self.input_scale, self.weight_scale) }

    /// Fresh logical bytes: FP8 activation + F32 accumulator + BF16 result.
    /// Excludes resident input/weight, allocator rounding, shared memory and
    /// runtime compilation/cache allocations; this is not a device-memory cap.
    pub fn working_bytes(&self, rows: usize) -> Result<usize> {
        working_bytes(rows, self.shape)
    }

    /// Contiguous BF16 [B,T,K] -> BF16 [B,T,N], on this operation's device.
    /// Forced native MMA only; unsupported geometry/capability has no fallback.
    /// No caller padding is required: all three MMA bounds checks stay enabled.
    pub fn project(&self, input: &CudaTensor, max_working_bytes: usize) -> Result<CudaTensor> {
        let dims = input.meta.shape().as_slice();
        if dims.len() != 3 || dims[2] != self.shape[1] || input.dtype != DType::BF16
            || input.qparams.is_some() || input.device != self.device
            || !std::ptr::eq(input.client.properties(), self.client.properties())
        {
            return Err("FP8 input requires BF16 [B,T,K] on the weight's CUDA client/device".into());
        }
        let rows = extent(&dims[..2], 1)?;
        let input_len = extent(dims, 2)?;
        let mut stride = 1usize;
        if input.meta.strides().len() != dims.len() {
            return Err("FP8 input strides have wrong rank".into());
        }
        for (axis, &dim) in dims.iter().enumerate().rev() {
            if dim > 1 && input.meta.strides()[axis] != stride {
                return Err("FP8 input must be contiguous row-major".into());
            }
            stride *= dim; // extent() proved this product above.
        }
        // CUDA allocations round up to 512 bytes and trim the unused tail.
        // Keep the aligned base; validate the tail before upstream's unchecked
        // size_in_used() subtraction. Arbitrary start-offset aliases stay out.
        if !base_buffer_covers(input.handle.size(), input.handle.offset_start.unwrap_or(0),
            input.handle.offset_end.unwrap_or(0), (input_len * 2) as u64) {
            return Err("FP8 input requires an aligned base buffer covering its BF16 bytes".into());
        }
        if self.working_bytes(rows)? > max_working_bytes {
            return Err("FP8 projection exceeds logical working-byte budget".into());
        }
        let [n, k] = self.shape;
        let output_len = extent(&[rows, n], 4)?;
        let client = &self.client;
        capability(client)?;
        let quantized = client.empty(input_len);
        let scratch = client.empty(output_len * 4);
        let output = client.empty(output_len * 2);
        let cube = CubeDim::new_1d(THREADS);
        // SAFETY: checked extents/client/contiguity, fresh destinations; each
        // element kernel checks its logical bound and owns one output element.
        unsafe {
            quantize::launch_unchecked::<CudaRuntime>(
                client, cubecl::calculate_cube_count_elemwise(client, input_len, cube), cube,
                ArrayArg::from_raw_parts(input.handle.clone(), input_len),
                ArrayArg::from_raw_parts(quantized.clone(), input_len), input_len, self.input_scale,
            );
        }
        launch_mma(client, &quantized, &self.weight, &scratch, rows, n, k)?;
        unsafe {
            finish::launch_unchecked::<CudaRuntime>(
                client, cubecl::calculate_cube_count_elemwise(client, output_len, cube), cube,
                ArrayArg::from_raw_parts(scratch, output_len),
                ArrayArg::from_raw_parts(output.clone(), output_len), output_len, self.alpha,
            );
        }
        Ok(CubeTensor::new_contiguous(client.clone(), self.device.clone(),
            [dims[0], dims[1], n].into(), output, DType::BF16))
    }
}

fn base_buffer_covers(size: u64, start: u64, end: u64, required: u64) -> bool {
    start == 0 && size.checked_sub(end).is_some_and(|used| used >= required)
}

fn extent(dims: &[usize], element_bytes: usize) -> Result<usize> {
    let elements = dims.iter().try_fold(1usize, |n, &d| {
        if d == 0 { None } else { n.checked_mul(d) }
    }).ok_or("FP8 empty or overflowing shape")?;
    if elements.checked_mul(element_bytes).is_none_or(|n| n > u32::MAX as usize) {
        return Err("FP8 buffer exceeds the bounded u32 byte extent".into());
    }
    Ok(elements)
}

fn weight_elements([n, k]: [usize; 2]) -> Result<usize> {
    let len = extent(&[n, k], 1)?;
    if k % 32 != 0 || n % 8 != 0 {
        return Err("native FP8 requires K divisible by 32 and N by 8".into());
    }
    Ok(len)
}

fn scale_product(input: f32, weight: f32) -> Result<f32> {
    let alpha = input * weight;
    if !input.is_finite() || input <= 0.0 || !weight.is_finite() || weight <= 0.0
        || !alpha.is_finite() || alpha <= 0.0
    {
        return Err("FP8 scales and their F32 product must be finite positive".into());
    }
    Ok(alpha)
}

fn working_bytes(rows: usize, [n, k]: [usize; 2]) -> Result<usize> {
    weight_elements([n, k])?;
    let input = extent(&[rows, k], 2)?;
    let output = extent(&[rows, n], 4)?;
    input.checked_add(output.checked_mul(6).ok_or("FP8 working-byte overflow")?)
        .ok_or_else(|| "FP8 working-byte overflow".into())
}

fn globals() -> MatmulGlobalElems {
    let fp8 = StorageType::Scalar(ElemType::Float(FloatKind::E4M3));
    MatmulGlobalElems { lhs: fp8, rhs: fp8, out: ElemType::Float(FloatKind::F32).into() }
}

fn capability(client: &ComputeClient<CudaRuntime>) -> Result<()> {
    let g = globals();
    let features = &client.properties().features;
    if !features.type_usage(g.lhs).contains(TypeUsage::Conversion)
        || !features.type_usage(g.lhs).contains(TypeUsage::Buffer)
        || !features.matmul.mma.contains(&MmaConfig {
            a_type: g.lhs, b_type: g.rhs, cd_type: g.out, m: 16, n: 8, k: 32,
        })
    {
        return Err("native E4M3 x E4M3 -> F32 m16n8k32 MMA/conversion unavailable".into());
    }
    Ok(())
}

#[cube(launch_unchecked)]
fn quantize(input: &Array<bf16>, output: &mut Array<e4m3>, len: usize, scale: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < len {
        let mut value = f32::cast_from(input[i]) / scale;
        // Branch comparisons preserve NaN and signed zero, unlike fmin/fmax.
        // CubeCL's FP8 cast is NOSAT, so saturate explicitly before that cast.
        if value > 448.0f32 { value = 448.0f32; }
        if value < -448.0f32 { value = -448.0f32; }
        output[i] = e4m3::cast_from(value);
    }
}

#[cube(launch_unchecked)]
fn finish(input: &Array<f32>, output: &mut Array<bf16>, len: usize, alpha: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < len { output[i] = bf16::cast_from(input[i] * alpha); }
}

fn blueprint(m: usize, n: usize, k: usize) -> Result<TilingBlueprint> {
    let problem = MatmulProblem::from_shapes_and_strides(
        [m, k].into(), [k, n].into(), [m, n].into(),
        [k, 1].into(), [1, k].into(), [n, 1].into(),
        globals(), cubecl::ir::AddressType::U32, None, None,
    ).map_err(|e| format!("FP8 problem: {e:?}"))?;
    let scheme = TilingScheme::builder().with_tile_size((16, 8, 32).into())
        .with_partition_size((2, 1, 4).into()).with_stage_size((2, 2, 1).into())
        .build().map_err(str::to_owned)?;
    let grid = HypercubeBlueprint::builder().global_order(GlobalOrder::RowMajor)
        .cube_count_strategy(CubeCountStrategy::FromProblem).build();
    let mut result = TilingBlueprint::builder(TileMatmulKind::Mma, scheme, 32, &problem)
        .partition_buffering(PartitionBuffering::Single).hypercube_blueprint(grid).build();
    result.check_m_bounds = true;
    result.check_n_bounds = true;
    result.check_k_bounds = true;
    Ok(result)
}

fn launch_mma(
    client: &ComputeClient<CudaRuntime>, input: &Handle, weight: &Handle, output: &Handle,
    m: usize, n: usize, k: usize,
) -> Result<()> {
    let binding = |handle: &Handle, shape: [usize; 2], strides: [usize; 2]| TensorBinding {
        handle: handle.clone().binding(), shape: shape.into(), strides: strides.into(),
        runtime: core::marker::PhantomData,
    };
    let g = globals();
    let mut dtypes = MatmulElems::from_globals(&g);
    let strategy = Strategy::SimpleCyclicMma(BlueprintStrategy::Forced(blueprint(m, n, k)?));
    cubek::matmul::launch::launch_ref(
        &strategy, client,
        InputBinding::new(binding(input, [m, k], [k, 1]), g.lhs),
        InputBinding::new(binding(weight, [k, n], [1, k]), g.rhs),
        binding(output, [m, n], [n, 1]), &mut dtypes,
    ).map_err(|e| format!("native FP8 [{m},{k}] x [{n},{k}]^T: {e:?}"))
}

#[cfg(test)]
#[path = "fp8_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "fp8_dense_tests.rs"]
mod dense_tests;
