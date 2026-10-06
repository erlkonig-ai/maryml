//! Isolated O/down GEMV experiment: existing Cubek kernel, no repacking.
//! Exact RawHalf, F16, [1,1,K] @ column-major [1,K,3072], K=4096/9216.
//! Unsupported metadata keeps the original matmul. Launch errors propagate.

use std::any::{Any, TypeId};

use burn::tensor::{DType, Tensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cubek::matmul::definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError};
use cubek::matmul::launch::Strategy;
use cubek::std::InputBinding;

use crate::nn::backend::hear::RawHalf;

/// Ephemeral metadata for this invocation, not a retained tensor catalogue.
#[derive(Clone, Copy)]
struct Layout {
    shape: [usize; 3],
    strides: [usize; 3],
    dtype: DType,
    bytes: u64,
    start: u64,
    end: u64,
    quantized: bool,
}

impl Layout {
    fn of(t: &CubeTensor<CudaRuntime>) -> Self {
        Self {
            shape: [t.meta.shape()[0], t.meta.shape()[1], t.meta.shape()[2]],
            strides: [
                t.meta.strides()[0],
                t.meta.strides()[1],
                t.meta.strides()[2],
            ],
            dtype: t.dtype,
            bytes: t.handle.size(),
            start: t.handle.offset_start.unwrap_or(0),
            end: t.handle.offset_end.unwrap_or(0),
            quantized: t.qparams.is_some(),
        }
    }

    fn covers(self, elements: usize, vector: usize) -> bool {
        let Some(alignment) = vector.checked_mul(2).and_then(|v| u64::try_from(v).ok()) else {
            return false;
        };
        let Some(required) = elements.checked_mul(2).and_then(|v| u64::try_from(v).ok()) else {
            return false;
        };
        let available = self
            .bytes
            .checked_sub(self.start)
            .and_then(|n| n.checked_sub(self.end));
        alignment != 0
            && self.start % alignment == 0
            && self.end % 2 == 0
            && available.is_some_and(|n| n >= required)
    }
}

fn eligible<B: Backend>(
    x: Layout,
    w: Layout,
    same_device: bool,
    plane: usize,
    vector: usize,
) -> bool {
    let k = x.shape[2];
    let Some(tile) = plane.checked_mul(vector).filter(|&v| v != 0) else {
        return false;
    };
    TypeId::of::<B>() == TypeId::of::<RawHalf>()
        && same_device
        && x.dtype == DType::F16
        && w.dtype == DType::F16
        && !x.quantized
        && !w.quantized
        && matches!(k, 4096 | 9216)
        && x.shape == [1, 1, k]
        && w.shape == [1, k, 3072]
        && x.strides[2] == 1
        && w.strides[1] == 1
        && w.strides[2] == k
        && k % tile == 0
        && x.covers(k, vector)
        && w.covers(k * 3072, vector)
}

/// `None` means unsupported metadata, not a failed launch. Bindings preserve
/// offsets and allocation ownership. Inputs remain read-only, output fresh;
/// existing immutable-pile ownership must still outlive all queued GPU uses.
/// Both globals/output are F16; from_globals selects F32 accumulation. The
/// pinned plane-parallel path casts the completed reduction once to F16.
pub(super) fn try_project<B: Backend>(
    x: &Tensor<B, 3>,
    w: &Tensor<B, 3>,
) -> Result<Option<Tensor<B, 3>>, MatmulSetupError> {
    let (Some(raw_x), Some(raw_w)) = (
        (x as &dyn Any).downcast_ref::<Tensor<RawHalf, 3>>(),
        (w as &dyn Any).downcast_ref::<Tensor<RawHalf, 3>>(),
    ) else {
        return Ok(None);
    };
    let (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) = (
        raw_x.clone().into_primitive(),
        raw_w.clone().into_primitive(),
    ) else {
        return Ok(None);
    };
    let plane = lhs.client.properties().hardware.plane_size_max as usize;
    let k = x.dims()[2];
    // Same selection as pinned Cubek for equal F16 operands. Check offsets
    // against that selected width, rather than allowing an implicit copy.
    let Some(vector) = lhs
        .client
        .io_optimized_vector_sizes(2)
        .filter(|&v| {
            plane
                .checked_mul(v)
                .is_some_and(|tile| tile != 0 && k % tile == 0)
        })
        .max()
    else {
        return Ok(None);
    };
    if !eligible::<B>(
        Layout::of(&lhs),
        Layout::of(&rhs),
        lhs.device == rhs.device,
        plane,
        vector,
    ) {
        return Ok(None);
    }
    let out = burn_cubecl::kernel::matmul::init_matmul_output(&lhs, &rhs, DType::F16);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: lhs.dtype.into(),
        rhs: rhs.dtype.into(),
        out: out.dtype.into(),
    });
    let client = lhs.client.clone();
    let (ld, rd) = (lhs.dtype, rhs.dtype);
    cubek::matmul::launch::launch_ref(
        &Strategy::GemvPlaneParallel(Default::default()),
        &client,
        InputBinding::new(lhs.binding(), ld.into()),
        InputBinding::new(rhs.binding(), rd.into()),
        out.clone().binding(),
        &mut dtypes,
    )?;
    let projected = Tensor::<RawHalf, 3>::from_primitive(TensorPrimitive::Float(out));
    Ok(Some(
        (&projected as &dyn Any)
            .downcast_ref::<Tensor<B, 3>>()
            .expect("both inputs established the exact RawHalf type")
            .clone(),
    ))
}

/// Test-only, once per loaded weight. Uses a real owned activation tensor's
/// metadata and the production predicate; never invents a handle/lifetime or
/// launches a GEMV. This observes eligibility, not a per-token execution count.
#[cfg(test)]
pub(super) fn observe_loaded_weight(
    x: &Tensor<RawHalf, 3>,
    w: &Tensor<RawHalf, 3>,
) -> serde_json::Value {
    let (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) =
        (x.clone().into_primitive(), w.clone().into_primitive())
    else {
        panic!("loaded observation requires actual float primitives")
    };
    let (a, b) = (Layout::of(&lhs), Layout::of(&rhs));
    let plane = lhs.client.properties().hardware.plane_size_max as usize;
    let k = a.shape[2];
    let vector = lhs
        .client
        .io_optimized_vector_sizes(2)
        .filter(|&v| {
            plane
                .checked_mul(v)
                .is_some_and(|tile| tile != 0 && k % tile == 0)
        })
        .max();
    let accepted =
        vector.is_some_and(|v| eligible::<RawHalf>(a, b, lhs.device == rhs.device, plane, v));
    let mut reasons = Vec::new();
    if lhs.device != rhs.device {
        reasons.push("device");
    }
    if a.dtype != DType::F16 || b.dtype != DType::F16 {
        reasons.push("dtype");
    }
    if a.quantized || b.quantized {
        reasons.push("quantization");
    }
    if !matches!(k, 4096 | 9216) || a.shape != [1, 1, k] || b.shape != [1, k, 3072] {
        reasons.push("shape");
    }
    if a.strides[2] != 1 {
        reasons.push("activation_stride");
    }
    if b.strides[1] != 1 || b.strides[2] != k {
        reasons.push("weight_stride");
    }
    match vector {
        None => reasons.push("vector_divisibility"),
        Some(v) => {
            if !a.covers(k, v) {
                reasons.push("activation_bounds_alignment");
            }
            if !b.covers(k.saturating_mul(3072), v) {
                reasons.push("weight_bounds_alignment");
            }
        }
    }
    assert_eq!(
        accepted,
        reasons.is_empty(),
        "observation reasons must agree with production gate"
    );
    serde_json::json!({"accepted":accepted,"reasons":reasons,"plane":plane,"vector":vector,
        "weight":{"shape":b.shape,"strides":b.strides,"dtype":format!("{:?}",b.dtype),
            "allocation_bytes":b.bytes,"offset_start":b.start,"offset_end":b.end},
        "activation":{"shape":a.shape,"strides":a.strides,"dtype":format!("{:?}",a.dtype),
            "allocation_bytes":a.bytes,"offset_start":a.start,"offset_end":a.end}})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::backend::hear;
    use burn::tensor::{FloatDType, TensorData};
    use half::f16;

    fn layouts(k: usize) -> (Layout, Layout) {
        (
            Layout {
                shape: [1, 1, k],
                strides: [k, k, 1],
                dtype: DType::F16,
                bytes: (k * 2) as u64,
                start: 0,
                end: 0,
                quantized: false,
            },
            Layout {
                shape: [1, k, 3072],
                strides: [k * 3072, 1, k],
                dtype: DType::F16,
                bytes: (k * 3072 * 2) as u64,
                start: 0,
                end: 0,
                quantized: false,
            },
        )
    }

    #[test]
    fn eligibility_is_only_odown_rawhalf_without_repacking() {
        for k in [4096, 9216] {
            let (x, w) = layouts(k);
            assert!(eligible::<RawHalf>(x, w, true, 32, 8));
            assert!(!eligible::<hear::Raw>(x, w, true, 32, 8));
            assert!(!eligible::<hear::FusedHalf>(x, w, true, 32, 8));
            assert!(!eligible::<RawHalf>(x, w, false, 32, 8));
            for (plane, vector) in [(0, 8), (32, 0), (31, 8), (32, 7), (usize::MAX, 8)] {
                assert!(!eligible::<RawHalf>(x, w, true, plane, vector));
            }
            for bad_x in [
                Layout {
                    shape: [1, 4, k],
                    ..x
                },
                Layout {
                    shape: [2, 1, k],
                    ..x
                },
                Layout {
                    strides: [k, k, 2],
                    ..x
                },
                Layout {
                    dtype: DType::F32,
                    ..x
                },
                Layout {
                    bytes: x.bytes - 1,
                    ..x
                },
                Layout {
                    start: 2,
                    bytes: x.bytes + 2,
                    ..x
                },
                Layout {
                    end: x.bytes + 1,
                    ..x
                },
                Layout {
                    quantized: true,
                    ..x
                },
            ] {
                assert!(!eligible::<RawHalf>(bad_x, w, true, 32, 8));
            }
            for bad_w in [
                Layout {
                    shape: [1, k, 3071],
                    ..w
                },
                Layout {
                    shape: [2, k, 3072],
                    ..w
                },
                Layout {
                    strides: [k * 3072, 3072, 1],
                    ..w
                },
                Layout {
                    strides: [k * 3072, 1, k + 16],
                    ..w
                },
                Layout {
                    dtype: DType::BF16,
                    ..w
                },
                Layout {
                    bytes: w.bytes - 2,
                    ..w
                },
                Layout {
                    start: w.bytes + 1,
                    ..w
                },
                Layout {
                    quantized: true,
                    ..w
                },
            ] {
                assert!(!eligible::<RawHalf>(x, bad_w, true, 32, 8));
            }
            // Singleton strides are irrelevant; aligned offsets retain views.
            assert!(eligible::<RawHalf>(
                Layout {
                    strides: [9 * k, 3 * k, 1],
                    start: 32,
                    end: 16,
                    bytes: x.bytes + 48,
                    ..x
                },
                Layout {
                    start: (2 * k) as u64,
                    end: (2 * k) as u64,
                    bytes: w.bytes + (4 * k) as u64,
                    ..w
                },
                true,
                32,
                8
            ));
        }
        for k in [0, 1280, 3072, 4095, 4097, 5120, 9215, 9217] {
            let (x, w) = layouts(k);
            assert!(!eligible::<RawHalf>(x, w, true, 32, 8));
        }
    }

    fn cube(t: &Tensor<RawHalf, 3>) -> CubeTensor<CudaRuntime> {
        let TensorPrimitive::Float(c) = t.clone().into_primitive() else {
            panic!("float")
        };
        c
    }

    fn read(t: Tensor<RawHalf, 3>) -> Vec<f16> {
        t.into_data().to_vec::<f16>().unwrap()
    }

    fn close(got: &[f16], want: &[f16]) {
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            let (a, b) = (a.to_f32(), b.to_f32());
            assert!(a.is_finite() && b.is_finite(), "nonfinite {i}: {a}/{b}");
            assert!((a - b).abs() <= 0.002 + 0.002 * b.abs(), "{i}: {a}/{b}");
        }
    }

    fn x_values(k: usize) -> Vec<f16> {
        (0..k)
            .map(|i| {
                f16::from_f32(match i {
                    0 | 1 => 256.0,
                    2 => 1.0,
                    _ => ((i % 17) as f32 - 8.0) / 16.0,
                })
            })
            .collect()
    }

    fn w_value(row: usize, col: usize, k: usize) -> f16 {
        f16::from_f32(match row % 4 {
            0 => 0.0,
            1 => {
                if col == k - 1 {
                    1.0
                } else {
                    0.0
                }
            }
            // F16 multiplication would overflow before cancellation; F32
            // arithmetic must produce exactly 1, not NaN/Inf or lost residue.
            2 => match col {
                0 => 256.0,
                1 => -256.0,
                2 => 1.0,
                _ => 0.0,
            },
            _ => (((row * 3 + col * 7) % 31) as f32 - 15.0) / 1024.0,
        })
    }

    #[test]
    #[ignore = "finite native GEMV control; separate GPU reservation required"]
    fn cuda_odown_transposes_cancellation_and_bias() {
        let device = Default::default();
        for k in [4096, 9216] {
            let input = x_values(k);
            let x =
                Tensor::<RawHalf, 3>::from_data(TensorData::new(input.clone(), [1, 1, k]), &device);
            let weights: Vec<f16> = (0..3072)
                .flat_map(|r| (0..k).map(move |c| w_value(r, c, k)))
                .collect();
            let w = Tensor::<RawHalf, 2>::from_data(
                TensorData::new(weights.clone(), [3072, k]),
                &device,
            )
            .transpose()
            .reshape([1, k, 3072]);
            assert_eq!(&cube(&w).meta.strides()[1..], &[1, k]);
            let direct = try_project(&x, &w)
                .unwrap()
                .expect("must launch GEMV, never fallback");
            assert_eq!(direct.dtype(), DType::F16);
            assert_eq!(direct.dims(), [1, 1, 3072]);
            let actual = read(direct.clone());
            let analytic: Vec<f16> = weights
                .chunks_exact(k)
                .map(|row| {
                    f16::from_f64(
                        row.iter()
                            .zip(&input)
                            .map(|(a, b)| a.to_f64() * b.to_f64())
                            .sum::<f64>(),
                    )
                })
                .collect();
            close(&actual, &analytic);
            close(&actual, &read(x.clone().matmul(w.clone())));
            for r in (2..3072).step_by(4) {
                assert_eq!(actual[r], f16::ONE);
            }
            // Show why the cancellation witness is sensitive to F16 products.
            assert!(f16::from_f32(256.0 * 256.0).is_infinite());

            let bias = Tensor::<RawHalf, 3>::from_data(
                TensorData::new(vec![f16::from_f32(0.125); 3072], [1, 1, 3072]),
                &device,
            );
            let linear = super::super::layers::Linear {
                weight_t: w.clone(),
                bias: Some(bias.clone()),
            };
            let routed = super::super::fast::output_projection(&linear, x.clone());
            // Bias remains a separate F16 operation after GEMV's F16 store.
            assert_eq!(read(routed), read(direct.clone() + bias));
            close(
                &read(super::super::fast::output_projection(&linear, x.clone())),
                &read(linear.forward(x.clone())),
            );
            let _ = read(direct.mul_scalar(0.0));
            assert_eq!(read(x), input);
            assert_eq!(read(w.swap_dims(1, 2)), weights);
        }
    }

    #[test]
    #[ignore = "finite GEMV offset/fallback control; no model weights"]
    fn cuda_offsets_fallbacks_and_inputs_immutable() {
        let device = Default::default();
        for k in [4096, 9216] {
            let input = x_values(k);
            let mut padded = vec![f16::from_f32(-7.0); k];
            padded.extend_from_slice(&input);
            padded.extend(vec![f16::from_f32(9.0); k]);
            let full_x = Tensor::<RawHalf, 3>::from_data(
                TensorData::new(padded.clone(), [1, 3, k]),
                &device,
            );
            let x = full_x.clone().narrow(1, 1, 1);
            let weights: Vec<f16> = (0..3074)
                .flat_map(|r| (0..k).map(move |c| w_value(r, c, k)))
                .collect();
            let full_w = Tensor::<RawHalf, 3>::from_data(
                TensorData::new(weights.clone(), [1, 3074, k]),
                &device,
            );
            let w = full_w.clone().narrow(1, 1, 3072).swap_dims(1, 2);
            assert!(cube(&x).handle.offset_start.unwrap_or(0) > 0);
            assert!(cube(&w).handle.offset_start.unwrap_or(0) > 0);
            let actual = try_project(&x, &w)
                .unwrap()
                .expect("aligned offset views launch GEMV");
            close(&read(actual), &read(x.clone().matmul(w.clone())));
            assert_eq!(read(full_x), padded);
            assert_eq!(read(full_w), weights);

            // Same numerical weights but incompatible row-major layout.
            let row = Tensor::<RawHalf, 3>::from_data(w.clone().into_data(), &device);
            assert_eq!(cube(&row).meta.strides()[2], 1);
            assert!(try_project(&x, &row).unwrap().is_none());
            let linear = super::super::layers::Linear {
                weight_t: row,
                bias: None,
            };
            assert_eq!(
                read(super::super::fast::output_projection(&linear, x.clone())),
                read(linear.forward(x.clone()))
            );
            let prefill = Tensor::<RawHalf, 3>::from_data(
                TensorData::new(vec![f16::ONE; 4 * k], [1, 4, k]),
                &device,
            );
            assert!(try_project(&prefill, &w).unwrap().is_none());
            let float_x = x.clone().cast(FloatDType::F32);
            assert!(try_project(&float_x, &w).unwrap().is_none());
            // Burn's slice copies if either removed byte extent is not
            // allocator-aligned. Keep a genuine strided view by making the
            // removed minor-axis tail exactly one alignment unit in bytes.
            let alignment = cube(&x).client.properties().memory.alignment as usize;
            assert!(alignment >= 2 && alignment <= 4096 && alignment % 2 == 0);
            let lanes = alignment / 2 + 1;
            let base = Tensor::<RawHalf, 3>::from_data(
                TensorData::new(vec![f16::ONE; lanes * k], [1, k, lanes]),
                &device,
            );
            // Optimized allocation can pad the physical row pitch beyond
            // logical `lanes`. Preserve and inspect that actual owned view.
            let pitch = cube(&base).meta.strides()[1];
            assert!(pitch >= lanes && pitch > 1);
            let strided = base.clone().swap_dims(1, 2).narrow(1, 0, 1);
            assert_eq!(strided.dims(), [1, 1, k]);
            assert_eq!(cube(&strided).meta.strides()[2], pitch);
            assert!(cube(&strided).meta.strides()[2] > 1);
            assert!(try_project(&strided, &w).unwrap().is_none());
            assert_eq!(read(strided), vec![f16::ONE; k]);
            assert_eq!(read(base), vec![f16::ONE; lanes * k]);
        }
    }
}
