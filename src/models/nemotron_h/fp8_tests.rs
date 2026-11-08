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

#[test]
#[ignore = "reserved CUDA: one-row decode kernel versus the general MMA path, byte for byte"]
fn gpu_one_row_decode_matches_general_mma_bytes() {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Shapes of the real decode projections' classes, scaled down where N is
    // large: K 4096/8192 (Mamba in/out, attention), small N (K/V), wide N.
    for (n, k, input_scale, weight_scale) in [
        (256usize, 4096usize, 0.0371f32, 0.00213f32),
        (4096, 8192, 0.0118, 0.00087),
        (1032, 4096, 0.25, 0.0042),
        (24, 256, 0.5, 0.25),
    ] {
        // Every finite E4M3FN code (0x7f/0xff are NaN; keep them out).
        let weight: Vec<u8> = (0..n * k).map(|_| {
            let b = (next() >> 32) as u8;
            if b & 0x7f == 0x7f { b ^ 0x01 } else { b }
        }).collect();
        let values: Vec<f32> = (0..k).map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0).collect();
        let input = tensor(&client, &device, &values, &[1, 1, k]);
        let projection = Projection::upload(device.clone(), [n, k], &weight, input_scale, weight_scale, weight.len()).unwrap();
        let budget = working_bytes(1, [n, k]).unwrap();
        // One-row kernel first, so it cannot pass on the general path's freed output.
        let one_row = read_bf16(&projection.project_with(&input, budget, true).unwrap());
        let general = read_bf16(&projection.project_with(&input, budget, false).unwrap());
        assert!(one_row.iter().any(|&b| b != 0), "[{n}, {k}] produced all zero bits");
        let differing = one_row.iter().zip(&general).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "[{n}, {k}]: {differing} of {n} outputs differ from the general MMA path");
    }
}

#[cube(launch_unchecked)]
fn stream_read(x: &Array<Vector<u32, Const<4>>>, out: &mut Array<u32>, per_thread: usize) {
    let threads = (CUBE_COUNT_X * CUBE_DIM_X) as usize;
    let i = ABSOLUTE_POS as usize;
    let mut acc = 0u32;
    for j in 0..per_thread {
        let v = x[i + j * threads];
        acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
    }
    out[i] = acc;
}

#[test]
#[ignore = "reserved CUDA: decode GEMV bandwidth sweep on the real projection shapes, diagnostic only"]
fn gpu_one_row_gemv_bandwidth_sweep() {
    use std::time::Instant;
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let fence = || cubecl::future::block_on(client.sync()).unwrap();
    // Streaming-read ceiling over 1 GiB.
    let (threads, per) = (48 * 1024 * 8, 128usize);
    let words = threads * per;
    let big = client.create_from_slice(&vec![0x5au8; words * 16]);
    let sink = client.empty(threads * 4);
    let run_stream = || unsafe {
        stream_read::launch_unchecked::<CudaRuntime>(&client, CubeCount::new_1d((threads / 256) as u32),
            CubeDim::new_1d(256), ArrayArg::from_raw_parts(big.clone(), words),
            ArrayArg::from_raw_parts(sink.clone(), threads), per);
    };
    for _ in 0..3 { run_stream(); }
    fence();
    let t = Instant::now();
    for _ in 0..10 { run_stream(); }
    fence();
    let s = t.elapsed().as_secs_f64() / 10.0;
    println!("SWEEP stream_read {:.1} GB/s ({} MiB per pass)", (words * 16) as f64 / s / 1e9, words * 16 >> 20);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    let mut total_bytes = 0f64;
    let mut best_total = 0f64;
    let mut general_total = 0f64;
    let counts = [(18560usize, 4096usize, 40usize), (4096, 8192, 40), (4096, 4096, 16), (256, 4096, 16), (131072, 4096, 1)];
    let variants = [(4usize, 0usize), (8, 0), (16, 0)];
    let mut per_variant = vec![0f64; variants.len()];
    for (n, k, calls) in counts {
        let weight: Vec<u8> = (0..n * k).map(|_| { let b = (next() >> 32) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect();
        let w = client.create_from_slice(&weight);
        let act: Vec<u8> = (0..k).map(|_| { let b = (next() >> 32) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect();
        let a = client.create_from_slice(&act);
        let bytes = (n * k) as f64;
        total_bytes += bytes * calls as f64;
        let time = |f: &dyn Fn()| { for _ in 0..3 { f(); } fence(); let t = Instant::now(); for _ in 0..20 { f(); } fence(); t.elapsed().as_secs_f64() / 20.0 };
        let scratch = client.empty(n * 4);
        let g = time(&|| launch_mma(&client, &a, &w, &scratch, 1, n, k).unwrap());
        general_total += g * calls as f64;
        let reference = { let o = client.empty(n * 2); launch_gemv_staged(&client, &a, &w, &o, 0.001, n, k, 8); client.read_one(o).unwrap() };
        let mut line = format!("SWEEP [{n},{k}] general {:.1} GB/s {:.1} us", bytes / g / 1e9, g * 1e6);
        let mut best = f64::MAX;
        for (vi, &(u, p)) in variants.iter().enumerate() {
            let o = client.empty(n * 2);
            let _ = p;
            let s = time(&|| launch_gemv_staged(&client, &a, &w, &o, 0.001, n, k, u));
            assert_eq!(client.read_one(o).unwrap(), reference, "staged u{u} differs on [{n},{k}]");
            per_variant[vi] += s * calls as f64;
            best = best.min(s);
            line += &format!(" | staged u{u} {:.1} GB/s", bytes / s / 1e9);
        }
        best_total += best * calls as f64;
        println!("{line}");
    }
    println!("SWEEP per-token dense FP8 bytes {:.2} GB: general {:.2} ms; per variant {}; best-per-shape {:.2} ms",
        total_bytes / 1e9, general_total * 1e3,
        variants.iter().zip(&per_variant).map(|(&(u, _), t)| format!("staged u{u} {:.2} ms", t * 1e3)).collect::<Vec<_>>().join(", "),
        best_total * 1e3);
}
