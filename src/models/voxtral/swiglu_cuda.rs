//! One exact decoder-M1 pointwise launch; projection and rounding stay separate.
//! Unsupported tensors keep Burn's original SiLU(gate) * up expression.

use std::any::{Any, TypeId};

use burn::tensor::{backend::Backend, DType, Tensor, TensorPrimitive};
use burn::tensor::activation::silu;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{calculate_cube_count_elemwise, cuda::CudaRuntime, prelude::*};
use half::f16;

use crate::nn::backend::hear::RawHalf;

const INTER: usize = 9216;
const VECTOR: usize = 4;

// Ephemeral view metadata, not a retained allocation or weight catalogue.
#[derive(Clone, Copy)]
struct Layout {
    shape: [usize; 3],
    strides: [usize; 3],
    dtype: DType,
    quantized: bool,
    bytes: u64,
    start: u64,
    end: u64,
}

fn eligible<B: Backend>(layout: Layout, inter: usize, gpu_vector4: bool) -> bool {
    let available = layout.bytes.checked_sub(layout.start)
        .and_then(|n| n.checked_sub(layout.end));
    TypeId::of::<B>() == TypeId::of::<RawHalf>()
        && gpu_vector4 && inter == INTER && layout.shape == [1, 1, 2 * INTER]
        && layout.strides[2] == 1 && layout.dtype == DType::F16 && !layout.quantized
        && layout.start % (2 * VECTOR) as u64 == 0 && layout.end % 2 == 0
        && available.is_some_and(|n| n >= (4 * INTER) as u64)
}

#[cube(launch_unchecked)]
fn swiglu_kernel(
    gu: &Array<Vector<f16, Const<4>>>,
    out: &mut Array<Vector<f16, Const<4>>>,
) {
    let i = ABSOLUTE_POS as usize;
    if i >= comptime!(INTER / VECTOR) { terminate!(); }
    let gate = gu[i];
    let up = gu[comptime!(INTER / VECTOR) + i];
    // The pinned Burn sigmoid is NOT reciprocal(1+exp(-x)). Preserve its
    // F32 cast, neg-as-multiply, exp, add, log, neg-as-multiply, exp sequence.
    let gate32 = Vector::<f32, Const<4>>::cast_from(gate);
    let minus_one = Vector::<f32, Const<4>>::cast_from(-1.0f32);
    let one = Vector::<f32, Const<4>>::cast_from(1.0f32);
    let ex = Vector::exp(gate32 * minus_one);
    let log = Vector::ln(ex + one);
    let sigmoid = Vector::<f16, Const<4>>::cast_from(Vector::exp(log * minus_one));
    // These are two F16 multiplies, not a reassociated F32 expression with
    // one final cast. The sigmoid and SiLU both round exactly at old stores.
    let activated: Vector<f16, Const<4>> = gate * sigmoid;
    out[i] = activated * up;
}

fn try_swiglu<B: Backend>(gu: &Tensor<B, 3>, inter: usize) -> Option<Tensor<B, 3>> {
    let raw = (gu as &dyn Any).downcast_ref::<Tensor<RawHalf, 3>>()?;
    let TensorPrimitive::Float(input) = raw.clone().into_primitive() else { return None; };
    let shape = input.meta.shape();
    let strides = input.meta.strides();
    if shape.len() != 3 || strides.len() != 3 { return None; }
    let layout = Layout {
        shape: [shape[0], shape[1], shape[2]],
        strides: [strides[0], strides[1], strides[2]],
        dtype: input.dtype, quantized: input.qparams.is_some(),
        bytes: input.handle.size(), start: input.handle.offset_start.unwrap_or(0),
        end: input.handle.offset_end.unwrap_or(0),
    };
    let gpu_vector4 = input.client.properties().hardware.num_cpu_cores.is_none()
        && input.client.io_optimized_vector_sizes(2).any(|v| v == VECTOR);
    if !eligible::<B>(layout, inter, gpu_vector4) { return None; }
    let client = input.client.clone();
    let output = client.empty(INTER * 2);
    let working_units = INTER / VECTOR;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    // SAFETY: exact RawHalf/F16, dense single row, aligned managed-handle view
    // covering both halves. Vector indices remain within the checked spans;
    // each output vector has one writer, no barriers or input writes. Cloned
    // handles keep view offsets and ownership through asynchronous dispatch.
    // Admitted launch failures propagate; they never trigger a fallback retry.
    unsafe {
        swiglu_kernel::launch_unchecked::<CudaRuntime>(
            &client, cube_count, cube_dim,
            ArrayArg::from_raw_parts(input.handle.clone(), 2 * INTER),
            ArrayArg::from_raw_parts(output.clone(), INTER),
        );
    }
    #[cfg(test)]
    FUSED_LAUNCHES.with(|n| n.set(n.get() + 1));
    let result = Tensor::<RawHalf, 3>::from_primitive(TensorPrimitive::Float(
        CubeTensor::new_contiguous(client, input.device, [1, 1, INTER].into(), output, DType::F16),
    ));
    Some((&result as &dyn Any).downcast_ref::<Tensor<B, 3>>()
        .expect("input established exact RawHalf type").clone())
}

pub(super) fn swiglu<B: Backend>(gu: Tensor<B, 3>, inter: usize) -> Tensor<B, 3> {
    try_swiglu(&gu, inter).unwrap_or_else(||
        silu(gu.clone().narrow(2, 0, inter)).mul(gu.narrow(2, inter, inter)))
}

#[cfg(test)]
std::thread_local! {
    static FUSED_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::backend::hear;
    use burn::tensor::{FloatDType, TensorData};

    #[test]
    fn exact_m1_metadata_and_fallbacks() {
        let good = Layout { shape: [1, 1, 2 * INTER], strides: [2 * INTER, 2 * INTER, 1],
            dtype: DType::F16, quantized: false, bytes: (4 * INTER) as u64, start: 0, end: 0 };
        assert!(eligible::<RawHalf>(good, INTER, true));
        assert!(eligible::<RawHalf>(Layout { strides: [99 * INTER, 7 * INTER, 1], ..good }, INTER, true));
        assert!(eligible::<RawHalf>(Layout { start: 64, end: 32, bytes: good.bytes + 96, ..good }, INTER, true));
        assert!(!eligible::<hear::Raw>(good, INTER, true));
        assert!(!eligible::<hear::FusedHalf>(good, INTER, true));
        assert!(!eligible::<RawHalf>(good, INTER, false));
        assert!(!eligible::<RawHalf>(good, 5120, true));
        for bad in [
            Layout { shape: [1, 4, 2 * INTER], ..good },
            Layout { shape: [2, 1, 2 * INTER], ..good },
            Layout { shape: [1, 1, 2 * INTER - 1], ..good },
            Layout { strides: [4 * INTER, 4 * INTER, 2], bytes: good.bytes * 2, ..good },
            Layout { dtype: DType::F32, ..good }, Layout { quantized: true, ..good },
            Layout { bytes: good.bytes - 2, ..good },
            Layout { start: 2, bytes: good.bytes + 2, ..good },
            Layout { end: 1, bytes: good.bytes + 1, ..good },
            Layout { start: u64::MAX, ..good }, Layout { end: u64::MAX, ..good },
        ] { assert!(!eligible::<RawHalf>(bad, INTER, true)); }
    }

    fn old<B: Backend>(gu: Tensor<B, 3>, inter: usize) -> Tensor<B, 3> {
        silu(gu.clone().narrow(2, 0, inter)).mul(gu.narrow(2, inter, inter))
    }

    fn tensor(values: Vec<f16>) -> Tensor<RawHalf, 3> {
        let len = values.len();
        Tensor::from_data(TensorData::new(values, [1, 1, len]), &Default::default())
    }

    fn read(t: Tensor<RawHalf, 3>) -> Vec<f16> { t.into_data().to_vec::<f16>().unwrap() }
    fn bits(t: Tensor<RawHalf, 3>) -> Vec<u16> { read(t).into_iter().map(f16::to_bits).collect() }
    fn raw(t: &Tensor<RawHalf, 3>) -> CubeTensor<CudaRuntime> {
        match t.clone().into_primitive() {
            TensorPrimitive::Float(t) => t, _ => panic!("float storage expected"),
        }
    }
    fn assert_same(got: &[f16], want: &[f16]) {
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            // NaN payloads are not an arithmetic contract; all other bits,
            // including signed zeros, subnormals and infinities, must match.
            assert!((a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                "index {i}: fused={a:?}/{:04x}, old={b:?}/{:04x}", a.to_bits(), b.to_bits());
        }
    }

    #[test]
    #[ignore = "exact old expression, half-rounding and exceptional controls; GPU admission"]
    fn cuda_expression_rounding_extremes_and_dispatch() {
        let special = [f16::ZERO, f16::from_bits(0x8000), f16::ONE, -f16::ONE,
            f16::MAX, f16::MIN, f16::from_bits(1), f16::from_bits(0x8001),
            f16::INFINITY, f16::NEG_INFINITY, f16::NAN];
        FUSED_LAUNCHES.with(|n| n.set(0));
        // All 65536 half gate bit patterns, in eight bounded real-size calls;
        // paired up values also cover extrema, signs, zeros, Inf and NaN.
        for block in 0..8 {
            let mut values = vec![f16::ZERO; 2 * INTER];
            for i in 0..INTER {
                values[i] = f16::from_bits(((block * INTER + i) % 65536) as u16);
                values[INTER + i] = special[i % special.len()];
            }
            let gu = tensor(values);
            let before = bits(gu.clone());
            assert_same(&read(swiglu(gu.clone(), INTER)), &read(old(gu.clone(), INTER)));
            assert_eq!(bits(gu), before, "read-only activation changed projected storage");
        }
        // Dense finite values expose removal of sigmoid/SiLU half rounding;
        // unlike the exceptional matrix above, every output participates.
        let mut values: Vec<f16> = (0..2 * INTER).map(|i|
            f16::from_f32(((i * 37 % 4093) as f32 - 2046.0) / 257.0)).collect();
        for i in [0, 7, 8, 31, 32, 255, 256, INTER - 1] {
            values[i] = f16::from_bits(1);
            values[INTER + i] = f16::MAX;
        }
        let gu = tensor(values);
        let before = bits(gu.clone());
        let want = read(old(gu.clone(), INTER));
        assert_same(&read(swiglu(gu.clone(), INTER)), &want);
        // At the smallest positive half, the half sigmoid is exactly .5.
        // The SiLU store rounds the tie to +0 before multiplying by MAX.
        // Omitting that store instead returns a nonzero representable half.
        let no_silu_store = f16::from_f32(f16::from_bits(1).to_f32() * 0.5 * f16::MAX.to_f32());
        assert_ne!(no_silu_store.to_bits(), 0);
        for i in [0, 7, 8, 31, 32, 255, 256, INTER - 1] {
            assert_eq!(want[i].to_bits(), 0, "old SiLU half-store witness at {i}");
        }
        assert_eq!(bits(gu), before);
        assert_eq!(FUSED_LAUNCHES.with(|n| n.get()), 9,
            "actual public dispatch must fuse every admitted input, not merely agree via fallback");
    }

    #[test]
    #[ignore = "managed offsets, real unsupported views/types/shapes and immutable input; GPU admission"]
    fn cuda_offsets_and_generic_fallbacks() {
        let probe = tensor(vec![f16::ONE]);
        let alignment = raw(&probe).client.properties().memory.alignment as usize;
        assert!(alignment >= 2 * VECTOR && alignment <= 4096 && alignment % 2 == 0);
        let pad = alignment / 2;
        let values = (0..2 * INTER + 2 * pad).map(|i|
            f16::from_f32((i % 31) as f32 / 8.0 - 2.0)).collect();
        let full = tensor(values);
        let before = bits(full.clone());
        let view = full.clone().narrow(2, pad, 2 * INTER);
        assert!(raw(&view).handle.offset_start.unwrap_or(0) > 0);
        FUSED_LAUNCHES.with(|n| n.set(0));
        assert_same(&read(swiglu(view.clone(), INTER)), &read(old(view.clone(), INTER)));
        assert_eq!(FUSED_LAUNCHES.with(|n| n.get()), 1);
        let small = full.clone().narrow(2, 0, 34);
        assert!(try_swiglu(&small, 17).is_none());
        assert_same(&read(swiglu(small.clone(), 17)), &read(old(small, 17)));
        let m4 = Tensor::<RawHalf,3>::ones([1,4,2*INTER], &Default::default());
        assert!(try_swiglu(&m4, INTER).is_none());
        assert_same(&read(swiglu(m4.clone(), INTER)), &read(old(m4, INTER)));
        let f32 = view.clone().cast(FloatDType::F32);
        assert!(try_swiglu(&f32, INTER).is_none());
        assert_eq!(swiglu(f32.clone(), INTER).into_data(), old(f32, INTER).into_data());
        let other = Tensor::<hear::Raw,3>::ones([1,1,2*INTER], &Default::default());
        assert!(try_swiglu(&other, INTER).is_none());
        assert_eq!(swiglu(other.clone(), INTER).into_data(), old(other, INTER).into_data());
        let storage = tensor((0..4 * INTER).map(|i|
            f16::from_f32((i % 19) as f32 / 8.0 - 1.0)).collect());
        let storage_before = bits(storage.clone());
        let mut cube = raw(&storage);
        cube.meta.shape[2] = 2 * INTER;
        cube.meta.strides[2] = 2;
        let strided = Tensor::<RawHalf,3>::from_primitive(TensorPrimitive::Float(cube));
        assert!(try_swiglu(&strided, INTER).is_none());
        assert_same(&read(swiglu(strided.clone(), INTER)), &read(old(strided, INTER)));
        assert_eq!(FUSED_LAUNCHES.with(|n| n.get()), 1, "unsupported metadata must not launch");
        assert_eq!(bits(full), before);
        assert_eq!(bits(storage), storage_before);
    }

    #[test]
    #[ignore = "finite no-model hot old/new/new/old micro; separate GPU admission"]
    fn cuda_hot_old_new_new_old() {
        let gu = tensor((0..2 * INTER).map(|i|
            f16::from_f32((i % 127) as f32 / 16.0 - 4.0)).collect());
        let want = read(old(gu.clone(), INTER));
        assert_same(&read(swiglu(gu.clone(), INTER)), &want);
        for fused in [false, true, true, false] {
            let start = std::time::Instant::now();
            for _ in 0..32 {
                let out = if fused { swiglu(gu.clone(), INTER) } else { old(gu.clone(), INTER) };
                assert_same(&read(out), &want);
            }
            println!("swiglu_hot fused={fused} calls=32 inclusive_host_full_read_seconds={:.9}",
                start.elapsed().as_secs_f64());
        }
    }
}
