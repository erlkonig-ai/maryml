//! Experimental weightless RMS for Voxtral's exact raw CUDA F16 backend.
//!
//! All other backends, dtypes, widths and non-dense views retain fast::rms's
//! original Burn expression. There is no gain here: fast.rs already folds it
//! into projections or applies the decoder's Ada scale after normalization.
//! One 256-thread cube owns each row, reducing F32 powf(x, 2) in a shared
//! tree. Keep F32 mean + epsilon, sqrt, reciprocal, then multiply, with only
//! the final store rounded to F16. Association of the sum changes; neither
//! F16 bit identity nor a hearing performance improvement is promised.

use std::any::{Any, TypeId};

use burn::tensor::{DType, Tensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::f16;

use crate::nn::backend::hear::RawHalf;

const UNITS: u32 = 256;

/// Dense logical rows only; singleton strides may retain a parent view's
/// values. The handle already carries the slice's byte offset. A conservative
/// grid bound also keeps every row/element index below u32::MAX.
fn geometry<B: Backend>(
    shape: [usize; 3],
    strides: [usize; 3],
    dtype: DType,
    eps: f64,
) -> Option<(usize, usize)> {
    if TypeId::of::<B>() != TypeId::of::<RawHalf>()
        || dtype != DType::F16
        || !matches!(shape[2], 1280 | 3072)
        || !(eps as f32).is_finite()
        || eps as f32 <= 0.0
    {
        return None;
    }
    let rows = shape[0].checked_mul(shape[1])?;
    if rows == 0 || rows > 65_535 {
        return None;
    }
    let mut expected = 1usize;
    for axis in (0..3).rev() {
        if shape[axis] != 1 && strides[axis] != expected {
            return None;
        }
        expected = expected.checked_mul(shape[axis])?;
    }
    Some((rows, shape[2]))
}

#[cube(launch_unchecked)]
fn rms_kernel(
    x: &Array<f16>,
    out: &mut Array<f16>,
    eps: f32,
    units: usize,
    #[comptime] width: usize,
) {
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut red = SharedMemory::<f32>::new(comptime!(UNITS as usize));
    let mut sum = 0.0f32;
    let mut col = lane;
    while col < width {
        let value = f32::cast_from(x[row * width + col]);
        // Match the original powf_scalar(2), not an alternate square opcode.
        sum += value.powf(2.0f32);
        col += units;
    }
    red[lane] = sum;
    sync_cube();
    // A runtime scalar, as in the existing cooperative reducers: CubeCL
    // rejects mutation of a value derived only from a compile-time constant.
    let mut stride = units / 2;
    while stride > 0 {
        if lane < stride {
            red[lane] += red[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    let mean = red[0] / f32::new(comptime!(width as f32));
    let inverse = (mean + eps).sqrt().recip();
    let mut col = lane;
    while col < width {
        let value = f32::cast_from(x[row * width + col]);
        out[row * width + col] = f16::cast_from(value * inverse);
        col += units;
    }
}

/// Local safe type check instead of changing Voxtral's generic backend API.
/// The returned tensor owns fresh device storage; the input is read-only and
/// cloned handles retain its allocation/offset through asynchronous dispatch.
pub(super) fn try_rms<B: Backend>(x: &Tensor<B, 3>, eps: f64) -> Option<Tensor<B, 3>> {
    let raw = (x as &dyn Any).downcast_ref::<Tensor<RawHalf, 3>>()?;
    let TensorPrimitive::Float(cube) = raw.clone().into_primitive() else {
        return None;
    };
    let shape = x.dims();
    let strides = [
        cube.meta.strides()[0],
        cube.meta.strides()[1],
        cube.meta.strides()[2],
    ];
    let (rows, width) = geometry::<B>(shape, strides, cube.dtype, eps)?;
    let count = rows * width;
    if cube.handle.size() < (count * 2) as u64 {
        return None;
    }
    let out = cube.client.empty(count * 2);
    // SAFETY: exact RawHalf/F16 type, checked dense logical rows and slice
    // handle bounds; grid has exactly rows cubes of UNITS threads. Both
    // admitted widths are multiples of UNITS. No early exit crosses a
    // barrier; every output element is written once, never into the input.
    unsafe {
        rms_kernel::launch_unchecked::<CudaRuntime>(
            &cube.client,
            CubeCount::new_1d(rows as u32),
            CubeDim::new_1d(UNITS),
            ArrayArg::from_raw_parts(cube.handle.clone(), count),
            ArrayArg::from_raw_parts(out.clone(), count),
            eps as f32,
            UNITS as usize,
            width,
        );
    }
    let normalized = Tensor::<RawHalf, 3>::from_primitive(TensorPrimitive::Float(
        CubeTensor::new_contiguous(cube.client, cube.device, shape.into(), out, DType::F16),
    ));
    Some(
        (&normalized as &dyn Any)
            .downcast_ref::<Tensor<B, 3>>()
            .expect("the input already established this exact tensor type")
            .clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::backend::hear;
    use burn::tensor::{FloatDType, TensorData};

    #[test]
    fn dispatch_is_only_raw_half_dense_hidden_widths() {
        for width in [1280, 3072] {
            for rows in [1, 4, 39] {
                assert_eq!(
                    geometry::<hear::RawHalf>(
                        [1, rows, width],
                        [rows * width, width, 1],
                        DType::F16,
                        1e-5,
                    ),
                    Some((rows, width)),
                );
            }
            // A narrowed final row may retain the old singleton-axis stride.
            assert_eq!(
                geometry::<hear::RawHalf>([1, 1, width], [39 * width, width, 1], DType::F16, 1e-5),
                Some((1, width)),
            );
            assert!(
                geometry::<hear::Raw>([1, 4, width], [4 * width, width, 1], DType::F16, 1e-5)
                    .is_none()
            );
            assert!(
                geometry::<hear::FusedHalf>([1, 4, width], [4 * width, width, 1], DType::F16, 1e-5)
                    .is_none()
            );
            assert!(
                geometry::<hear::RawHalf>([1, 4, width], [4 * width, width, 1], DType::F32, 1e-5)
                    .is_none()
            );
            assert!(
                geometry::<hear::RawHalf>([1, 4, width], [4 * width, 1, 4], DType::F16, 1e-5)
                    .is_none()
            );
            assert!(
                geometry::<hear::RawHalf>(
                    [1, 4, width],
                    [4 * width, width + 8, 1],
                    DType::F16,
                    1e-5
                )
                .is_none()
            );
            assert!(
                geometry::<hear::RawHalf>([1, 0, width], [0, width, 1], DType::F16, 1e-5).is_none()
            );
            assert!(
                geometry::<hear::RawHalf>(
                    [1, 65_536, width],
                    [65_536 * width, width, 1],
                    DType::F16,
                    1e-5
                )
                .is_none()
            );
            for eps in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-300, 1e300] {
                assert!(
                    geometry::<hear::RawHalf>([1, 1, width], [width, width, 1], DType::F16, eps)
                        .is_none()
                );
            }
        }
        for width in [0, 128, 256, 1024, 1279, 1281, 2048, 3071, 3073] {
            assert!(
                geometry::<hear::RawHalf>([1, 1, width], [width, width, 1], DType::F16, 1e-5)
                    .is_none()
            );
        }
    }

    fn burn_rms(x: Tensor<hear::RawHalf, 3>, eps: f64) -> Tensor<hear::RawHalf, 3> {
        let dt = x.dtype();
        let x32 = x.cast(FloatDType::F32);
        let var = x32.clone().powf_scalar(2.0).mean_dim(2);
        x32.mul(var.add_scalar(eps).sqrt().recip()).cast(dt)
    }

    fn values(rows: usize, width: usize) -> Vec<f16> {
        (0..rows * width)
            .map(|i| {
                let (row, col) = (i / width, i % width);
                let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
                let v = match row % 7 {
                    0 => (((col * 37 + 11) % 127) as f32 - 63.0) / 16.0,
                    1 => sign * 65_504.0, // F16 square/sum would overflow.
                    2 => sign * 0.000030517578125, // Epsilon-dominated row.
                    3 => {
                        if col == width - 1 {
                            4.0
                        } else {
                            sign * 0.0
                        }
                    }
                    4 => sign * f16::from_bits(1).to_f32(),
                    5 => sign * 2.0,
                    _ => {
                        if col == 0 {
                            65_504.0
                        } else {
                            sign * 0.125
                        }
                    }
                };
                f16::from_f32(v)
            })
            .collect()
    }

    fn read(x: Tensor<hear::RawHalf, 3>) -> Vec<f16> {
        x.into_data().to_vec::<f16>().unwrap()
    }

    fn assert_close(got: &[f16], want: &[f16]) {
        assert_eq!(got.len(), want.len());
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            let (a, b) = (a.to_f32(), b.to_f32());
            assert!(
                a.is_finite() && b.is_finite(),
                "nonfinite at {i}: {a} / {b}"
            );
            let mag = f16::from_f32(b.abs());
            let ulp = (f16::from_bits(mag.to_bits() + 1).to_f32() - b.abs())
                .max(f16::from_bits(1).to_f32());
            assert!((a - b).abs() <= 2.0 * ulp, "at {i}: {a} vs {b}, ulp {ulp}");
        }
    }

    #[test]
    #[ignore = "CUDA operator control; run only under a separately admitted GPU reservation"]
    fn cuda_matches_burn_and_weightless_analytic_cases() {
        let device = Default::default();
        for width in [1280, 3072] {
            for rows in [1, 4, 39] {
                let input = values(rows, width);
                for eps in [1e-5, 0.25] {
                    let x = Tensor::<hear::RawHalf, 3>::from_data(
                        TensorData::new(input.clone(), [1, rows, width]),
                        &device,
                    );
                    let direct = try_rms(&x, eps).expect("eligible RawHalf input");
                    assert_eq!(direct.dtype(), DType::F16);
                    let got = read(direct);
                    assert_close(&got, &read(burn_rms(x.clone(), eps)));
                    assert_eq!(got, read(super::super::fast::rms(x.clone(), eps)));

                    let analytic: Vec<f16> = input
                        .chunks_exact(width)
                        .flat_map(|row| {
                            let sum: f64 = row.iter().map(|v| (v.to_f32() as f64).powi(2)).sum();
                            let inv = 1.0 / (sum / width as f64 + (eps as f32) as f64).sqrt();
                            row.iter()
                                .map(move |v| f16::from_f64(v.to_f32() as f64 * inv))
                        })
                        .collect();
                    assert_close(&got, &analytic);
                    // Inputs are read-only; retained aliases survive every launch.
                    assert_eq!(read(x), input);
                }
            }
        }
    }

    #[test]
    #[ignore = "CUDA layout/dtype/boundary control; no model weights needed"]
    fn cuda_offsets_fallbacks_and_nonfinite_rows() {
        let device = Default::default();
        for width in [1280, 3072] {
            let input = values(4, width);
            let x = Tensor::<hear::RawHalf, 3>::from_data(
                TensorData::new(input.clone(), [2, 2, width]),
                &device,
            );
            assert_close(
                &read(try_rms(&x, 1e-5).unwrap()),
                &read(burn_rms(x.clone(), 1e-5)),
            );
            let last = x.reshape([1, 4, width]).narrow(1, 3, 1);
            assert_close(
                &read(try_rms(&last, 1e-5).unwrap()),
                &read(burn_rms(last, 1e-5)),
            );

            let strided = Tensor::<hear::RawHalf, 3>::from_data(
                TensorData::new(input, [1, width, 4]),
                &device,
            )
            .swap_dims(1, 2);
            assert!(try_rms(&strided, 1e-5).is_none());
            assert_eq!(
                read(super::super::fast::rms(strided.clone(), 1e-5)),
                read(burn_rms(strided, 1e-5))
            );

            let wide =
                Tensor::<hear::RawHalf, 1>::from_floats(vec![1.0f32; width].as_slice(), &device)
                    .reshape([1, 1, width])
                    .cast(FloatDType::F32);
            assert!(try_rms(&wide, 1e-5).is_none());
            let wide_out = super::super::fast::rms(wide.clone(), 1e-5);
            assert_eq!(wide_out.dtype(), DType::F32);
            assert_eq!(
                wide_out.into_data().to_vec::<f32>().unwrap(),
                burn_rms(wide, 1e-5).into_data().to_vec::<f32>().unwrap()
            );

            let mut exceptional = vec![f16::from_f32(1.0); 3 * width];
            exceptional[0] = f16::NAN;
            exceptional[width] = f16::INFINITY;
            for v in &mut exceptional[2 * width..] {
                *v = f16::NEG_ZERO;
            }
            let x = Tensor::<hear::RawHalf, 3>::from_data(
                TensorData::new(exceptional, [1, 3, width]),
                &device,
            );
            let got = read(try_rms(&x, 1e-5).unwrap());
            let want = read(burn_rms(x, 1e-5));
            for (g, w) in got.iter().zip(want) {
                if w.is_nan() {
                    assert!(g.is_nan());
                } else {
                    assert_eq!(g.to_bits(), w.to_bits());
                }
            }
        }
        let x = Tensor::<hear::RawHalf, 3>::from_data(
            TensorData::new(values(4, 256), [1, 4, 256]),
            &device,
        );
        assert!(try_rms(&x, 1e-5).is_none());
        assert_eq!(
            read(super::super::fast::rms(x.clone(), 1e-5)),
            read(burn_rms(x, 1e-5))
        );
    }
}
