//! Synthetic diagnostics only. No checkpoint/model acceptance oracle.
//! GPU tests are ignored until an explicitly reserved run. Exact byte checks
//! and analytic sparse sums use zero tolerance; independent stock GPU parity
//! remains required for dense arithmetic and actual checkpoint projections.
use super::*;

#[test]
fn geometry_scales_and_budgets_are_admitted_before_device_use() {
    for shape in [[18560, 4096], [4096, 8192], [4096, 4096], [256, 4096], [131072, 4096]] {
        assert_eq!(weight_elements(shape).unwrap(), shape[0] * shape[1]);
    }
    for shape in [[0, 32], [8, 0], [7, 32], [8, 33], [usize::MAX, 32]] {
        assert!(weight_elements(shape).is_err());
    }
    assert_eq!(working_bytes(65, [24, 128]).unwrap(), 65 * (128 + 6 * 24));
    assert!(working_bytes(0, [24, 128]).is_err());
    assert!(working_bytes(usize::MAX, [24, 128]).is_err());
    assert!(extent(&[u32::MAX as usize], 2).is_err());
    // CUDA create_from_slice uses a 512-byte allocation with a trimmed tail
    // for a 256-byte M=1,K=128 BF16 input. This is not a start-offset alias.
    assert!(base_buffer_covers(512, 0, 256, 256));
    assert!(base_buffer_covers(512, 0, 0, 512));
    assert!(!base_buffer_covers(512, 0, 257, 256));
    assert!(!base_buffer_covers(512, 0, 513, 256));
    assert!(!base_buffer_covers(512, 0, u64::MAX, 256));
    assert!(!base_buffer_covers(512, 1, 0, 256));
    assert!(!base_buffer_covers(512, u64::MAX, 0, 256));
    assert_eq!(scale_product(0.5, 0.25).unwrap(), 0.125);
    for (a, w) in [
        (0.0, 1.0), (-0.0, 1.0), (-1.0, 1.0), (1.0, f32::NAN),
        (f32::INFINITY, 1.0), (f32::MAX, 2.0), (f32::from_bits(1), 0.5),
    ] {
        assert!(scale_product(a, w).is_err(), "scales {a:?}, {w:?}");
    }
    // Invalid scale/length/budget all return before CudaRuntime::client.
    let device = CudaDevice { index: 0 };
    assert!(Projection::upload(device.clone(), [8, 32], &[0; 256], f32::MAX, 2.0, 256).is_err());
    assert!(Projection::upload(device.clone(), [8, 32], &[0; 255], 0.5, 0.25, 256).is_err());
    assert!(Projection::upload(device, [8, 32], &[0; 256], 0.5, 0.25, 255).is_err());
}

#[test]
fn forced_fp8_mma_keeps_types_and_geometry_across_token_tails() {
    let base = blueprint(1, 24, 128).unwrap();
    for m in [3, 16, 63, 64, 65] {
        assert_eq!(blueprint(m, 24, 128).unwrap(), base);
    }
    assert!(base.check_m_bounds && base.check_n_bounds && base.check_k_bounds);
    assert_eq!(base.tiling_scheme.elements_per_stage_along_m(), 64);
    assert_eq!(base.tiling_scheme.elements_per_stage_along_n(), 16);
    assert_eq!(base.tiling_scheme.elements_per_stage_along_k(), 128);
    let g = globals();
    let d = MatmulElems::from_globals(&g);
    assert_eq!(d.lhs_stage, g.lhs);
    assert_eq!(d.rhs_stage, g.rhs);
    assert_eq!(d.lhs_register, g.lhs);
    assert_eq!(d.rhs_register, g.rhs);
    assert_eq!(d.acc_stage, g.out);
    assert_eq!(d.acc_register, g.out);
}

fn tensor(client: &ComputeClient<CudaRuntime>, device: &CudaDevice, data: &[f32], shape: &[usize]) -> CudaTensor {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let bytes: Vec<_> = data.iter().flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes()).collect();
    let handle = client.create_from_slice(&bytes);
    CubeTensor::new_contiguous(client.clone(), device.clone(), shape.into(), handle, DType::BF16)
}

fn read_bf16(tensor: &CudaTensor) -> Vec<u16> {
    // read_one fences the device work; no result is inspected before completion.
    tensor.client.read_one(tensor.handle.clone()).unwrap().chunks_exact(2)
        .map(|v| u16::from_le_bytes([v[0], v[1]])).collect()
}

#[test]
#[ignore = "reserved CUDA: exact FP8 cast/saturation diagnostics, no model acceptance"]
fn gpu_quantizer_boundaries_signed_zero_nan_and_guard() {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    capability(&client).unwrap();
    // Hand-stated E4M3 encodings; no copy of the native quantizer as an oracle.
    let cases = [
        (0.0, 0x00), (-0.0, 0x80),
        (1.0 / 1024.0, 0x00), (-1.0 / 1024.0, 0x80),
        (1.0 / 512.0, 0x01), (3.0 / 1024.0, 0x02),
        (5.0 / 1024.0, 0x02), (15.0 / 1024.0, 0x08),
        (1.0 / 64.0, 0x08), (0.75, 0x34),
        (1.0, 0x38), (1.0625, 0x38), (1.1875, 0x3a), (1.3125, 0x3a),
        (-1.0625, 0xb8), (-1.1875, 0xba),
        (448.0, 0x7e), (464.0, 0x7e), (512.0, 0x7e),
        (-464.0, 0xfe), (f32::INFINITY, 0x7e), (f32::NEG_INFINITY, 0xfe),
    ];
    for scale in [0.5, 1.0, 2.0] {
        let mut data: Vec<f32> = cases.iter().map(|(x, _)| x * scale).collect();
        data.extend([bf16::from_bits(1).to_f32(), -bf16::from_bits(1).to_f32(), f32::NAN]);
        let input = tensor(&client, &device, &data, &[1, 1, data.len()]);
        let output = client.create_from_slice(&vec![0xa5; data.len() + 37]);
        let cube = CubeDim::new_1d(THREADS);
        unsafe {
            quantize::launch_unchecked::<CudaRuntime>(
                &client, cubecl::calculate_cube_count_elemwise(&client, data.len(), cube), cube,
                ArrayArg::from_raw_parts(input.handle, data.len()),
                ArrayArg::from_raw_parts(output.clone(), data.len() + 37), data.len(), scale,
            );
        }
        let bytes = client.read_one(output).unwrap();
        for (i, (_, expected)) in cases.iter().enumerate() {
            assert_eq!(bytes[i], *expected, "case {i}, scale {scale}");
        }
        assert_eq!(bytes[cases.len()], 0x00);
        assert_eq!(bytes[cases.len() + 1], 0x80);
        assert_eq!(bytes[cases.len() + 2] & 0x7f, 0x7f, "NaN stays NaN");
        assert!(bytes[data.len()..].iter().all(|&b| b == 0xa5), "quantizer guard overwritten");
    }
}

fn sparse_fixture(rows: usize) {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let (n, k) = (24, 128); // N-tail at 16, M-tail at 64.
    let mut weight = vec![0u8; n * k];
    for j in 0..n {
        let col = j * 3 % k;
        weight[j * k + col] = if j % 2 == 0 { 0x38 } else { 0xb8 }; // +/-1
        weight[j * k + (col + 5) % k] = if j % 3 == 0 { 0xc0 } else { 0x40 }; // +/-2
    }
    let values: Vec<f32> = (0..rows * k).map(|i| {
        (((i / k * 3 + i % k) % 9) as i32 - 4) as f32 * 0.25
    }).collect();
    let input = tensor(&client, &device, &values, &[1, rows, k]);
    let projection = Projection::upload(device.clone(), [n, k], &weight, 0.5, 0.25, weight.len()).unwrap();
    let budget = working_bytes(rows, [n, k]).unwrap();
    assert!(projection.project(&input, budget - 1).is_err());
    let result = projection.project(&input, budget).unwrap();
    assert_eq!(result.meta.shape().as_slice(), &[1, rows, n]);
    let got = read_bf16(&result);
    for r in 0..rows {
        for j in 0..n {
            let col = j * 3 % k;
            let a = if j % 2 == 0 { 1.0 } else { -1.0 };
            let b = if j % 3 == 0 { -2.0 } else { 2.0 };
            // Only these two nonzero products exist. All operands and sums
            // are exact binary rationals, so zero tolerance is justified.
            let expected = (values[r * k + col] * a + values[r * k + (col + 5) % k] * b) * 0.25;
            let actual = bf16::from_bits(got[r * n + j]).to_f32();
            assert_eq!(actual, expected, "row {r}, column {j}");
        }
    }
    // Distinct token slices retain the same fixed arithmetic configuration.
    for r in 0..rows {
        let one = tensor(&client, &device, &values[r * k..(r + 1) * k], &[1, 1, k]);
        let decoded = read_bf16(&projection.project(&one, budget).unwrap());
        assert_eq!(decoded, got[r * n..(r + 1) * n], "partition row {r}");
    }

    // Surround each destination with a trailing sentinel while the binding's
    // logical matrix remains unpadded. Exercise actual quantize/MMA/finish
    // bounds, not just descriptor arithmetic. Source handles remain private.
    let quantized = client.create_from_slice(&vec![0xa5; rows * k + 37]);
    let scratch = client.create_from_slice(&vec![0xa5; (rows * n + 37) * 4]);
    let output = client.create_from_slice(&vec![0xa5; (rows * n + 37) * 2]);
    let cube = CubeDim::new_1d(THREADS);
    unsafe {
        quantize::launch_unchecked::<CudaRuntime>(
            &client, cubecl::calculate_cube_count_elemwise(&client, rows * k, cube), cube,
            ArrayArg::from_raw_parts(input.handle.clone(), rows * k),
            ArrayArg::from_raw_parts(quantized.clone(), rows * k + 37), rows * k, 0.5f32,
        );
    }
    launch_mma(&client, &quantized, &projection.weight, &scratch, rows, n, k).unwrap();
    unsafe {
        finish::launch_unchecked::<CudaRuntime>(
            &client, cubecl::calculate_cube_count_elemwise(&client, rows * n, cube), cube,
            ArrayArg::from_raw_parts(scratch.clone(), rows * n + 37),
            ArrayArg::from_raw_parts(output.clone(), rows * n + 37), rows * n, 0.125f32,
        );
    }
    for (label, handle, used) in [
        ("quantize", quantized, rows * k), ("MMA", scratch, rows * n * 4),
        ("epilogue", output, rows * n * 2),
    ] {
        let bytes = client.read_one(handle).unwrap();
        assert!(bytes[used..].iter().all(|&b| b == 0xa5), "{label} guard overwritten at M={rows}");
    }
}

#[test]
#[ignore = "reserved CUDA: sparse analytic FP8 projections/tail guards, not stock GPU parity"]
fn gpu_sparse_projections_m1_m3_m65() {
    for rows in [1, 3, 65] { sparse_fixture(rows); }
}

#[test]
#[ignore = "reserved CUDA: descriptor rejection on an existing device, no second GPU"]
fn gpu_descriptor_refusals() {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let projection = Projection::upload(device.clone(), [8, 32], &[0; 256], 0.5, 0.25, 256).unwrap();
    let data = [0.0; 32];
    let mut input = tensor(&client, &device, &data, &[1, 1, 32]);
    input.dtype = DType::F32;
    assert!(projection.project(&input, 4096).is_err());
    input.dtype = DType::BF16;
    input.device.index = 1;
    assert!(projection.project(&input, 4096).is_err());
    input.device = device;
    input.handle.offset_start = Some(u64::MAX);
    assert!(projection.project(&input, 4096).is_err());
}
