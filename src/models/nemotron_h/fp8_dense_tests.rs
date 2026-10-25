//! Saved independent Torch oracle: diagnostic only, never model acceptance.
//! Requires sha256sum plus explicit fixture/output directories. No downloads.
use super::*;
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
};

const MANIFEST_SHA256: &str = "adff5695e99cbee038b4056e0a5f6da401702e131d1e4156ded4e95e9ce249ea";

fn sha256(bytes: &[u8]) -> String {
    // Hash the exact loaded/exported bytes, not a second read of a mutable file.
    // This ignored GB10 diagnostic uses the host utility, not a new dependency.
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(result.status.success(), "sha256sum: {:?}", result.stderr);
    let text = String::from_utf8(result.stdout).unwrap();
    let hash = text.split_whitespace().next().unwrap();
    assert_eq!(hash.len(), 64);
    hash.to_owned()
}

fn bounded_read(path: &Path, limit: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .unwrap()
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() <= limit, "oversized fixture {}", path.display());
    bytes
}

fn fixture(
    root: &Path,
    manifest: &Value,
    case: &str,
    name: &str,
    shape: &[usize],
    dtype: &str,
    size: usize,
) -> Vec<u8> {
    let path = format!("{case}/{name}"); // Both components are fixed by this test.
    let record = manifest["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["path"].as_str() == Some(path.as_str()))
        .unwrap();
    assert_eq!(record["shape"], json!(shape));
    assert_eq!(record["dtype"], dtype);
    assert_eq!(record["bytes"], size);
    let bytes = bounded_read(&root.join(path), size);
    assert_eq!(bytes.len(), size);
    assert_eq!(sha256(&bytes), record["sha256"].as_str().unwrap());
    bytes
}

fn save(root: &Path, path: &str, shape: &[usize], dtype: &str, bytes: &[u8], records: &mut Vec<Value>) {
    fs::write(root.join(path), bytes).unwrap();
    records.push(json!({"path":path,"shape":shape,"dtype":dtype,"bytes":bytes.len(),"sha256":sha256(bytes)}));
    fs::write(
        root.join("native-tensors.json"),
        serde_json::to_vec_pretty(&json!({
            "byte_order":"little", "reference_manifest_sha256":MANIFEST_SHA256, "tensors":records,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn compare_bf16(actual: &[u8], reference: &[u8]) -> Value {
    assert_eq!(actual.len(), reference.len());
    assert_eq!(actual.len() % 2, 0);
    let (mut nonfinite, mut outside_band, mut worst_index) = (0usize, 0usize, None);
    let (mut max_abs, mut max_relative, mut max_normalized) = (0.0f64, 0.0f64, 0.0f64);
    let mut relative_infinite = false;
    for (i, (got, want)) in actual.chunks_exact(2).zip(reference.chunks_exact(2)).enumerate() {
        let got = bf16::from_bits(u16::from_le_bytes([got[0], got[1]])).to_f32() as f64;
        let want = bf16::from_bits(u16::from_le_bytes([want[0], want[1]])).to_f32() as f64;
        if !got.is_finite() || !want.is_finite() {
            nonfinite += 1;
            continue;
        }
        let error = (got - want).abs();
        let normalized = error / (0.02 + 0.02 * want.abs()); // Frozen before any run.
        if normalized > 1.0 {
            outside_band += 1;
        }
        if normalized > max_normalized || worst_index.is_none() {
            worst_index = Some(i);
        }
        max_abs = max_abs.max(error);
        max_normalized = max_normalized.max(normalized);
        if want != 0.0 {
            max_relative = max_relative.max(error / want.abs());
        } else if error != 0.0 {
            relative_infinite = true;
        }
    }
    json!({"passed":nonfinite == 0 && outside_band == 0,"elements":actual.len()/2,
        "nonfinite_pairs":nonfinite,"outside_band":outside_band,"max_absolute_error":max_abs,
        "max_relative_error":if relative_infinite {None} else {Some(max_relative)},
        "relative_error_infinite":relative_infinite,"worst_normalized_error":max_normalized,
        "worst_normalized_index":worst_index,"atol":0.02,"rtol":0.02})
}

#[test]
#[ignore = "reserved CUDA: saved Torch FP8 byte/MMA/full projection diagnostics, not model acceptance"]
fn gpu_saved_torch_dense_three_arms() {
    let reference = std::env::var_os("NEMOTRON_FP8_REFERENCE_DIR").expect("set frozen Torch fixture directory");
    let reference = Path::new(&reference);
    let output = std::env::var_os("NEMOTRON_FP8_NATIVE_OUT").expect("set a NEW native output directory");
    let output = Path::new(&output);
    let manifest_bytes = bounded_read(&reference.join("tensors.json"), 32 * 1024);
    assert_eq!(sha256(&manifest_bytes), MANIFEST_SHA256);
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(manifest["byte_order"], "little");
    assert_eq!(manifest["tensors"].as_array().unwrap().len(), 21);
    fs::create_dir(output).expect("output must not exist; never overwrite previous evidence");
    let mut records = Vec::new();
    let mut all_passed = true;
    for m in [1usize, 3, 65] {
        let (n, k) = (32usize, 128usize);
        let case = format!("dense-M{m}-N{n}-K{k}");
        fs::create_dir(output.join(&case)).unwrap();
        let x = fixture(
            reference,
            &manifest,
            &case,
            "input.bf16",
            &[m, k],
            "torch.bfloat16",
            m * k * 2,
        );
        let w = fixture(
            reference,
            &manifest,
            &case,
            "weight.e4m3fn",
            &[n, k],
            "torch.float8_e4m3fn",
            n * k,
        );
        let q = fixture(
            reference,
            &manifest,
            &case,
            "quantized_input.e4m3fn",
            &[m, k],
            "torch.float8_e4m3fn",
            m * k,
        );
        let expected = fixture(
            reference,
            &manifest,
            &case,
            "output.bf16",
            &[m, n],
            "torch.bfloat16",
            m * n * 2,
        );
        let sa = f32::from_le_bytes(
            fixture(reference, &manifest, &case, "input_scale.f32", &[1], "torch.float32", 4)
                .try_into()
                .unwrap(),
        );
        let sw = f32::from_le_bytes(
            fixture(
                reference,
                &manifest,
                &case,
                "weight_scale.f32",
                &[1],
                "torch.float32",
                4,
            )
            .try_into()
            .unwrap(),
        );
        assert_eq!(sa.to_bits(), 0.3f32.to_bits());
        assert_eq!(sw.to_bits(), 0.7f32.to_bits());
        let mut metrics = json!({"M":m,"N":n,"K":k,"input_scale":sa,"weight_scale":sw,
            "status":"running", "quantizer":null,"same_stock_bytes":null,"full_projection":null});
        let metrics_path = output.join(&case).join("metrics.json");
        fs::write(&metrics_path, serde_json::to_vec_pretty(&metrics).unwrap()).unwrap();
        let device = CudaDevice { index: 0 };
        let projection = Projection::upload(device.clone(), [n, k], &w, sa, sw, w.len()).unwrap();
        let client = &projection.client;
        let input = CubeTensor::new_contiguous(
            client.clone(),
            device,
            [1, m, k].into(),
            client.create_from_slice(&x),
            DType::BF16,
        );
        let cube = CubeDim::new_1d(THREADS);

        // Arm 1: compare exact bytes, but do not demand reciprocal/division
        // bit identity. NaN boundary policy is tested separately, not here.
        let native_q = client.empty(m * k);
        unsafe {
            quantize::launch_unchecked::<CudaRuntime>(
                client,
                cubecl::calculate_cube_count_elemwise(client, m * k, cube),
                cube,
                ArrayArg::from_raw_parts(input.handle.clone(), m * k),
                ArrayArg::from_raw_parts(native_q.clone(), m * k),
                m * k,
                sa,
            );
        }
        let native_q = client.read_one(native_q).unwrap(); // Fence before export/comparison.
        assert_eq!(native_q.len(), q.len());
        save(
            output,
            &format!("{case}/quantized_input.e4m3fn"),
            &[m, k],
            "e4m3fn",
            &native_q,
            &mut records,
        );
        let mismatches: Vec<usize> = native_q
            .iter()
            .zip(&q)
            .enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i))
            .collect();
        metrics["quantizer"] = json!({"byte_mismatches":mismatches.len(),"first_mismatch":mismatches.first(),
            "native_nan_codes":native_q.iter().filter(|&&b| b & 0x7f == 0x7f).count(),
            "stock_nan_codes":q.iter().filter(|&&b| b & 0x7f == 0x7f).count(),"gate":"reported separately"});
        fs::write(&metrics_path, serde_json::to_vec_pretty(&metrics).unwrap()).unwrap();

        // Arm 2: exact saved stock FP8 bytes; no native activation quantizer.
        let stock_q = client.create_from_slice(&q);
        let accumulator = client.empty(m * n * 4);
        launch_mma(client, &stock_q, &projection.weight, &accumulator, m, n, k).unwrap();
        let raw_acc = client.read_one(accumulator.clone()).unwrap();
        save(
            output,
            &format!("{case}/same_stock_accumulator.f32"),
            &[m, n],
            "f32",
            &raw_acc,
            &mut records,
        );
        let same = client.empty(m * n * 2);
        unsafe {
            finish::launch_unchecked::<CudaRuntime>(
                client,
                cubecl::calculate_cube_count_elemwise(client, m * n, cube),
                cube,
                ArrayArg::from_raw_parts(accumulator, m * n),
                ArrayArg::from_raw_parts(same.clone(), m * n),
                m * n,
                projection.alpha,
            );
        }
        let same = client.read_one(same).unwrap();
        save(
            output,
            &format!("{case}/same_stock_output.bf16"),
            &[m, n],
            "bf16",
            &same,
            &mut records,
        );
        metrics["same_stock_bytes"] = compare_bf16(&same, &expected);
        fs::write(&metrics_path, serde_json::to_vec_pretty(&metrics).unwrap()).unwrap();

        // Arm 3: public immutable Projection path, including native quantize.
        let full = projection
            .project(&input, projection.working_bytes(m).unwrap())
            .unwrap();
        let full = client.read_one(full.handle).unwrap();
        save(
            output,
            &format!("{case}/full_projection_output.bf16"),
            &[m, n],
            "bf16",
            &full,
            &mut records,
        );
        metrics["full_projection"] = compare_bf16(&full, &expected);
        let passed = metrics["same_stock_bytes"]["passed"].as_bool().unwrap()
            && metrics["full_projection"]["passed"].as_bool().unwrap();
        all_passed &= passed;
        metrics["status"] = json!(if passed { "diagnostic_pass" } else { "diagnostic_fail" });
        fs::write(&metrics_path, serde_json::to_vec_pretty(&metrics).unwrap()).unwrap();
        println!("{case}: {metrics}");
    }
    // Keep all sizes' output/metrics even when a numerical diagnostic fails.
    assert!(
        all_passed,
        "frozen FP8 diagnostic band failed; inspect retained outputs/metrics"
    );
}
