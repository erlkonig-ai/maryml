//! Exact CUDA vocabulary argmax using the existing cooperative Cube routine.
//! Logits stay F16 and the result stays i32; no head fusion or host-read change.

use std::any::{Any, TypeId};

use burn::tensor::{backend::Backend, DType, Int, Tensor, TensorPrimitive};
use burn_cubecl::kernel::reduce::{reduce_dim, KernelReduceStrategy};
use cubecl::{features::Plane, tensor_vector_size_parallel};
use cubek::reduce::{
    components::instructions::ReduceOperationConfig,
    launch::{ReduceStrategy, RoutineStrategy, VectorizationStrategy},
    routines::{cube::CubeStrategy, BlueprintStrategy},
    ReduceError,
};

use crate::nn::backend::hear::RawHalf;
use super::config::VOCAB;

// Ephemeral metadata only. No copy, contiguous conversion or retained state.
#[derive(Clone, Copy)]
struct Layout {
    len: usize,
    stride: usize,
    dtype: DType,
    quantized: bool,
    bytes: u64,
    start: u64,
    end: u64,
}

fn eligible<B: Backend>(layout: Layout, vector: usize, planes: bool) -> bool {
    let Some(alignment) = vector.checked_mul(2).filter(|&n| n > 0) else {
        return false;
    };
    let available = layout.bytes.checked_sub(layout.start)
        .and_then(|n| n.checked_sub(layout.end));
    TypeId::of::<B>() == TypeId::of::<RawHalf>()
        && planes && layout.len == VOCAB && layout.stride == 1
        && layout.dtype == DType::F16 && !layout.quantized
        && VOCAB % vector == 0
        && layout.start % alignment as u64 == 0 && layout.end % 2 == 0
        && available.is_some_and(|n| n >= (VOCAB * 2) as u64)
}

fn try_argmax<B: Backend>(logits: &Tensor<B, 1>)
    -> Result<Option<Tensor<B, 1, Int>>, ReduceError>
{
    let Some(raw) = (logits as &dyn Any).downcast_ref::<Tensor<RawHalf, 1>>() else {
        return Ok(None);
    };
    let TensorPrimitive::Float(input) = raw.clone().into_primitive() else {
        return Ok(None);
    };
    if input.meta.shape().len() != 1 || input.meta.strides().len() != 1 {
        return Ok(None);
    }
    let layout = Layout {
        len: input.meta.shape()[0], stride: input.meta.strides()[0],
        dtype: input.dtype, quantized: input.qparams.is_some(),
        bytes: input.handle.size(), start: input.handle.offset_start.unwrap_or(0),
        end: input.handle.offset_end.unwrap_or(0),
    };
    // Same vector choice as Cubek's parallel-axis reducer; validate the view's
    // offset against it rather than silently copying a misaligned view.
    let vector = tensor_vector_size_parallel(input.client.io_optimized_vector_sizes(2),
        input.meta.shape(), input.meta.strides(), 0);
    let props = input.client.properties();
    let planes = props.hardware.num_cpu_cores.is_none()
        && props.hardware.plane_size_min == 32 && props.hardware.plane_size_max == 32
        && props.features.plane.contains(Plane::Ops);
    if !eligible::<B>(layout, vector, planes) {
        return Ok(None);
    }
    let output = reduce_dim(input, Some(DType::I32), 0,
        KernelReduceStrategy::Specific(ReduceStrategy {
            routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(
                CubeStrategy { use_planes: true })),
            vectorization: VectorizationStrategy { parallel_output_vectorization: false },
        }), ReduceOperationConfig::ArgMax)?;
    #[cfg(test)]
    CUBE_LAUNCHES.with(|count| count.set(count.get() + 1));
    let output = Tensor::<RawHalf, 1, Int>::from_primitive(output);
    Ok(Some((&output as &dyn Any).downcast_ref::<Tensor<B, 1, Int>>()
        .expect("input established exact RawHalf type").clone()))
}

/// Unsupported metadata retains the original operation. An admitted launch
/// error remains an error, not a hidden catch-and-retry on the old path.
pub(super) fn argmax<B: Backend>(logits: Tensor<B, 1>) -> Tensor<B, 1, Int> {
    try_argmax(&logits).expect("Voxtral cooperative argmax launch")
        .unwrap_or_else(|| logits.argmax(0))
}

#[cfg(test)]
std::thread_local! {
    static CUBE_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::backend::hear;
    use burn::tensor::{FloatDType, TensorData};
    use burn_cubecl::tensor::CubeTensor;
    use cubecl::cuda::CudaRuntime;
    use half::f16;

    #[test]
    fn exact_vocab_metadata_and_fallbacks() {
        let good = Layout { len: VOCAB, stride: 1, dtype: DType::F16,
            quantized: false, bytes: (VOCAB * 2) as u64, start: 0, end: 0 };
        assert!(eligible::<RawHalf>(good, 8, true));
        assert!(!eligible::<hear::Raw>(good, 8, true));
        assert!(!eligible::<hear::FusedHalf>(good, 8, true));
        assert!(!eligible::<RawHalf>(good, 8, false));
        for vector in [0, 3, usize::MAX] {
            assert!(!eligible::<RawHalf>(good, vector, true));
        }
        for bad in [
            Layout { len: VOCAB - 1, ..good }, Layout { len: VOCAB + 1, ..good },
            Layout { stride: 2, bytes: good.bytes * 2, ..good },
            Layout { dtype: DType::F32, ..good }, Layout { quantized: true, ..good },
            Layout { bytes: good.bytes - 2, ..good },
            Layout { start: 2, bytes: good.bytes + 2, ..good },
            Layout { end: 1, bytes: good.bytes + 1, ..good },
            Layout { start: u64::MAX, ..good }, Layout { end: u64::MAX, ..good },
        ] { assert!(!eligible::<RawHalf>(bad, 8, true)); }
        assert!(eligible::<RawHalf>(Layout {
            start: 64, end: 32, bytes: good.bytes + 96, ..good }, 8, true));
    }

    fn tensor(values: Vec<f16>) -> Tensor<RawHalf, 1> {
        let len = values.len();
        Tensor::from_data(TensorData::new(values, [len]), &Default::default())
    }

    fn scalar(t: Tensor<RawHalf, 1, Int>) -> i32 {
        t.into_data().iter::<i32>().next().expect("one index")
    }

    fn bits(t: Tensor<RawHalf, 1>) -> Vec<u16> {
        t.into_data().iter::<f16>().map(f16::to_bits).collect()
    }

    fn cube(t: &Tensor<RawHalf, 1>) -> CubeTensor<CudaRuntime> {
        match t.clone().into_primitive() {
            TensorPrimitive::Float(raw) => raw,
            _ => panic!("expected float"),
        }
    }

    #[test]
    #[ignore = "full-vocabulary old/Cube token identity incl exceptional values; GPU reservation"]
    fn cuda_vocab_ties_exceptional_and_immutable() {
        let mut cases = Vec::new();
        for at in [0, VOCAB / 2 + 3, VOCAB - 1] {
            let mut values = vec![f16::from_f32(-2.0); VOCAB];
            values[at] = f16::from_f32(4.0);
            cases.push((format!("max_{at}"), values, Some(at as i32)));
        }
        let mut ties = vec![f16::from_f32(-2.0); VOCAB];
        for at in [7, 8, 31, 32, 255, 256, 4095, 4096, VOCAB - 1] {
            ties[at] = f16::from_f32(4.0);
        }
        cases.push(("cross_vector_plane_ties".into(), ties, Some(7)));
        let mut zeros = vec![f16::from_f32(-1.0); VOCAB];
        zeros[9] = f16::from_bits(0x8000); zeros[32] = f16::ZERO;
        cases.push(("signed_zero_tie".into(), zeros, Some(9)));
        let mut infs = vec![f16::NEG_INFINITY; VOCAB];
        infs[111] = f16::INFINITY; infs[777] = f16::INFINITY;
        cases.push(("infinite_tie".into(), infs, Some(111)));
        for (name, value, expected) in [
            ("all_zero", f16::ZERO, Some(0)), ("all_min", f16::MIN, Some(0)),
            ("all_pos_inf", f16::INFINITY, Some(0)),
            ("all_neg_inf", f16::NEG_INFINITY, None), ("all_nan", f16::NAN, None),
        ] { cases.push((name.into(), vec![value; VOCAB], expected)); }
        let mut mixed = vec![f16::from_f32(-1.0); VOCAB];
        for at in [0, 8, 31, 256, 4096, VOCAB - 1] { mixed[at] = f16::NAN; }
        mixed[123] = f16::from_f32(9.0);
        cases.push(("mixed_nan".into(), mixed, None));
        let mut surrounded = vec![f16::NAN; VOCAB];
        surrounded[4097] = f16::from_f32(3.0);
        cases.push(("finite_amid_nan".into(), surrounded, None));
        let exceptional = (0..VOCAB).map(|i| match i % 3 {
            0 => f16::NAN, 1 => f16::NEG_INFINITY, _ => f16::INFINITY,
        }).collect();
        cases.push(("all_exceptional_mixed".into(), exceptional, None));
        let mut failures = Vec::new();
        CUBE_LAUNCHES.with(|count| count.set(0));
        let count = cases.len();
        for (label, values, expected) in cases {
            let input = tensor(values);
            let before = bits(input.clone());
            let old = scalar(input.clone().argmax(0));
            let new = scalar(argmax(input.clone()));
            println!("argmax_case={label} old={old} cube={new}");
            if old != new || expected.is_some_and(|id| id != old) {
                failures.push((label, old, new, expected));
            }
            assert_eq!(bits(input), before, "read-only reduction changed logits");
        }
        assert_eq!(CUBE_LAUNCHES.with(|count| count.get()), count,
            "every comparison must actually use Cube, never a generic fallback");
        assert!(failures.is_empty(), "observable token differences: {failures:?}");
    }

    #[test]
    #[ignore = "valid offset route, unsupported actual tensors and immutable guards; GPU reservation"]
    fn cuda_offsets_and_generic_fallbacks() {
        let probe = tensor(vec![f16::ONE]);
        let alignment = cube(&probe).client.properties().memory.alignment as usize;
        assert!(alignment >= 2 && alignment <= 4096 && alignment % 2 == 0);
        let pad = alignment / 2;
        let mut values = vec![f16::from_f32(-1.0); VOCAB + 2 * pad];
        values[0] = f16::INFINITY; values[VOCAB + 2 * pad - 1] = f16::INFINITY;
        values[pad + 4097] = f16::ONE;
        let full = tensor(values);
        let before = bits(full.clone());
        let view = full.clone().narrow(0, pad, VOCAB);
        assert!(cube(&view).handle.offset_start.unwrap_or(0) > 0);
        CUBE_LAUNCHES.with(|count| count.set(0));
        assert_eq!(scalar(argmax(view.clone())), 4097);
        assert_eq!(scalar(view.argmax(0)), 4097);
        assert_eq!(CUBE_LAUNCHES.with(|count| count.get()), 1);
        assert_eq!(bits(full.clone()), before);
        let small = full.clone().narrow(0, 0, 17);
        assert!(try_argmax(&small).unwrap().is_none());
        assert_eq!(scalar(argmax(small.clone())), scalar(small.argmax(0)));
        let widened = full.clone().narrow(0, pad, VOCAB).cast(FloatDType::F32);
        assert!(try_argmax(&widened).unwrap().is_none());
        assert_eq!(scalar(argmax(widened.clone())), scalar(widened.argmax(0)));
        let f32_backend = Tensor::<hear::Raw, 1>::ones([VOCAB], &Default::default());
        assert!(try_argmax(&f32_backend).unwrap().is_none());
        // A real owned storage view, not slice_with_steps (which can copy).
        // Every addressed element is within the 2*VOCAB-element allocation.
        let mut storage = vec![f16::from_f32(-2.0); 2 * VOCAB];
        storage[2 * 65539] = f16::ONE;
        storage[2 * 65539 + 1] = f16::INFINITY; // excluded interleaved column
        let storage = tensor(storage);
        let storage_before = bits(storage.clone());
        let mut raw = cube(&storage);
        raw.meta.shape[0] = VOCAB;
        raw.meta.strides[0] = 2;
        let strided = Tensor::<RawHalf, 1>::from_primitive(TensorPrimitive::Float(raw));
        assert_eq!(strided.dims(), [VOCAB]);
        assert_eq!(cube(&strided).meta.strides()[0], 2);
        assert!(try_argmax(&strided).unwrap().is_none());
        assert_eq!(scalar(argmax(strided.clone())), 65539);
        assert_eq!(scalar(strided.argmax(0)), 65539);
        assert_eq!(CUBE_LAUNCHES.with(|count| count.get()), 1);
        assert_eq!(bits(full), before);
        assert_eq!(bits(storage), storage_before);
    }

    #[test]
    #[ignore = "finite no-model hot old/new/new/old micro; separate GPU admission"]
    fn cuda_hot_old_new_new_old() {
        let mut values = vec![f16::from_f32(-1.0); VOCAB];
        values[65539] = f16::ONE;
        let input = tensor(values);
        assert_eq!(scalar(input.clone().argmax(0)), 65539);
        assert_eq!(scalar(argmax(input.clone())), 65539);
        for new in [false, true, true, false] {
            let start = std::time::Instant::now();
            for _ in 0..32 {
                let output = if new { argmax(input.clone()) } else { input.clone().argmax(0) };
                assert_eq!(scalar(output), 65539);
            }
            println!("argmax_hot cube={new} calls=32 inclusive_host_read_seconds={:.9}",
                start.elapsed().as_secs_f64());
        }
    }
}
