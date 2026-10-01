//! Source-bound actual-HF comparison of WeMM's prepared single-image/text leg.
//! MODEL.pile ROOT REFERENCE.json EXPECTED_CHECKPOINT_SHA256 NEW_REPORT.json
//! The model pile is a pre-imported immutable native-BF16 input, never rewritten.
//! This gate performs CPU byte hashing/audit comparisons, not CPU model math.
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
};
use half::bf16;
use mary::{
    models::qwen3_5::{
        config::Qwen3_5Config,
        multimodal::PreparedMultimodal,
        multimodal_layout::{EMBEDDING, IMAGE, ImagePlan, VISION_END, VISION_START},
        vision_geometry::Grid,
    },
    nn::cuda_bf16_alias::CudaBf16Aliases,
};
type CudaTensor = CubeTensor<CudaRuntime>;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::OpenOptions, io::Write, path::Path};
use triblespace::prelude::Id;

const HF: &str = "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8";
const CONFIG: &str = "34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004";
const WRAPPER: &str = "ac255e1fad459cc3e68891d6c3327f4486922aed02fb3c5c13fb53277ba8e94f";

// This exact list is mirrored by scripts/wemm_image_sources.txt, checked
// before any GPU load. Sources are embedded by rustc, not read from cwd later.
const SOURCES: &[(&str, &[u8])] = &[
    ("Cargo.toml", include_bytes!("../../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../../Cargo.lock")),
    ("src/leaf.rs", include_bytes!("../leaf.rs")),
    ("src/format.rs", include_bytes!("../format.rs")),
    ("src/ingest.rs", include_bytes!("../ingest.rs")),
    ("src/persist.rs", include_bytes!("../persist.rs")),
    (
        "src/model_collection.rs",
        include_bytes!("../model_collection.rs"),
    ),
    (
        "src/nn/cuda_bf16_alias.rs",
        include_bytes!("../nn/cuda_bf16_alias.rs"),
    ),
    (
        "src/models/qwen3_5/mod.rs",
        include_bytes!("../models/qwen3_5/mod.rs"),
    ),
    (
        "src/models/qwen3_5/config.rs",
        include_bytes!("../models/qwen3_5/config.rs"),
    ),
    (
        "src/models/qwen3_5/layout.rs",
        include_bytes!("../models/qwen3_5/layout.rs"),
    ),
    (
        "src/models/qwen3_5/deltanet.rs",
        include_bytes!("../models/qwen3_5/deltanet.rs"),
    ),
    (
        "src/models/qwen3_5/gdn_ops.rs",
        include_bytes!("../models/qwen3_5/gdn_ops.rs"),
    ),
    (
        "src/models/qwen3_5/gdn_mixer.rs",
        include_bytes!("../models/qwen3_5/gdn_mixer.rs"),
    ),
    (
        "src/models/qwen3_5/gdn_decoder.rs",
        include_bytes!("../models/qwen3_5/gdn_decoder.rs"),
    ),
    (
        "src/models/qwen3_5/full_attention.rs",
        include_bytes!("../models/qwen3_5/full_attention.rs"),
    ),
    (
        "src/models/qwen3_5/decoder_ops.rs",
        include_bytes!("../models/qwen3_5/decoder_ops.rs"),
    ),
    (
        "src/models/qwen3_5/decoder_stack.rs",
        include_bytes!("../models/qwen3_5/decoder_stack.rs"),
    ),
    (
        "src/models/qwen3_5/outer_trace.rs",
        include_bytes!("../models/qwen3_5/outer_trace.rs"),
    ),
    (
        "src/models/qwen3_5/embedding_boundary.rs",
        include_bytes!("../models/qwen3_5/embedding_boundary.rs"),
    ),
    (
        "src/models/qwen3_5/roles.rs",
        include_bytes!("../models/qwen3_5/roles.rs"),
    ),
    (
        "src/models/qwen3_5/prepared.rs",
        include_bytes!("../models/qwen3_5/prepared.rs"),
    ),
    (
        "src/models/qwen3_5/vision_geometry.rs",
        include_bytes!("../models/qwen3_5/vision_geometry.rs"),
    ),
    (
        "src/models/qwen3_5/vision_frontend.rs",
        include_bytes!("../models/qwen3_5/vision_frontend.rs"),
    ),
    (
        "src/models/qwen3_5/vision_block.rs",
        include_bytes!("../models/qwen3_5/vision_block.rs"),
    ),
    (
        "src/models/qwen3_5/vision_merger.rs",
        include_bytes!("../models/qwen3_5/vision_merger.rs"),
    ),
    (
        "src/models/qwen3_5/vision_tower.rs",
        include_bytes!("../models/qwen3_5/vision_tower.rs"),
    ),
    (
        "src/models/qwen3_5/multimodal_layout.rs",
        include_bytes!("../models/qwen3_5/multimodal_layout.rs"),
    ),
    (
        "src/models/qwen3_5/position_table.rs",
        include_bytes!("../models/qwen3_5/position_table.rs"),
    ),
    (
        "src/models/qwen3_5/multimodal.rs",
        include_bytes!("../models/qwen3_5/multimodal.rs"),
    ),
    (
        "src/bin/wemm_image_gate.rs",
        include_bytes!("wemm_image_gate.rs"),
    ),
    (
        "scripts/wemm_prepared_reference.py",
        include_bytes!("../../scripts/wemm_prepared_reference.py"),
    ),
    (
        "scripts/wemm_image_reference.py",
        include_bytes!("../../scripts/wemm_image_reference.py"),
    ),
    (
        "scripts/wemm_image_sources.txt",
        include_bytes!("../../scripts/wemm_image_sources.txt"),
    ),
];

#[derive(Deserialize)]
struct Weight {
    name: String,
    shape: Vec<u64>,
    sha256: String,
}
#[derive(Deserialize, Serialize)]
struct Layer {
    layer: usize,
    hidden: Vec<u16>,
}
#[derive(Deserialize)]
struct Reference {
    oracle: String,
    hf_sha256: String,
    config_sha256: String,
    wrapper_sha256: String,
    generator_sha256: String,
    native_sources: BTreeMap<String, String>,
    checkpoint_sha256: String,
    checkpoint_bytes: u64,
    transformers: String,
    config_json: String,
    ids: Vec<u32>,
    grid: [usize; 3],
    weights: Vec<Weight>,
    layers: Vec<Layer>,
    pixels: Vec<u16>,
    pixels_sha256: String,
    positions: Vec<[u32; 3]>,
    rope_delta: i64,
    merged: Vec<u16>,
    scattered: Vec<u16>,
    final_hidden: Vec<u16>,
    embedding: Vec<u16>,
    vision_rotary_dtype: String,
    text_rotary_dtype: String,
}
#[derive(Serialize)]
struct Selected {
    name: String,
    handle: String,
    shape: Vec<u64>,
    payload_sha256: String,
}
#[derive(Serialize)]
struct Check {
    name: String,
    count: usize,
    outside_budget: usize,
    different: usize,
    worst_scaled: Option<f32>,
    atol: f32,
    rtol: f32,
}
#[derive(Serialize)]
struct EmbeddingDiagnostic {
    cosine: Option<f64>,
    angular_error_radians: Option<f64>,
    native_l2: Option<f64>,
    reference_l2: Option<f64>,
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
fn bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
// Audit arithmetic only, over final output bytes; this neither implements a
// model operation nor participates in the unchanged coordinate admission gate.
fn embedding_diagnostic(native: &[u16], reference: &[u16]) -> EmbeddingDiagnostic {
    let (mut dot, mut nn, mut rr) = (0.0f64, 0.0f64, 0.0f64);
    for (&n, &r) in native.iter().zip(reference) {
        let n = f64::from(bf16::from_bits(n).to_f32());
        let r = f64::from(bf16::from_bits(r).to_f32());
        dot += n * r;
        nn += n * n;
        rr += r * r;
    }
    let cosine = (native.len() == reference.len()
        && !native.is_empty()
        && dot.is_finite()
        && nn.is_finite()
        && rr.is_finite()
        && nn > 0.0
        && rr > 0.0)
        .then(|| (dot / (nn.sqrt() * rr.sqrt())).clamp(-1.0, 1.0));
    EmbeddingDiagnostic {
        cosine,
        angular_error_radians: cosine.map(f64::acos),
        native_l2: nn.is_finite().then(|| nn.sqrt()),
        reference_l2: rr.is_finite().then(|| rr.sqrt()),
    }
}
fn read(t: &CudaTensor) -> Result<Vec<u8>, String> {
    t.client
        .read_one(t.handle.clone())
        .map(|b| b.to_vec())
        .map_err(|e| format!("GPU read: {e:?}"))
}
fn compare(
    name: &str,
    t: &CudaTensor,
    shape: &[usize],
    expected: &[u16],
    atol: f32,
    rtol: f32,
) -> Result<Check, String> {
    if t.dtype != DType::BF16 || t.meta.shape().as_slice() != shape {
        return Err(format!("{name}: wrong shape/dtype"));
    }
    let bytes = read(t)?;
    if bytes.len() != expected.len() * 2 || expected.is_empty() {
        return Err(format!("{name}: wrong output/reference extent"));
    }
    let (mut bad, mut different, mut worst) = (0, 0, 0.0f32);
    for (b, &bits) in bytes.chunks_exact(2).zip(expected) {
        let actual_bits = u16::from_le_bytes(b.try_into().unwrap());
        let actual = bf16::from_bits(actual_bits).to_f32();
        let wanted = bf16::from_bits(bits).to_f32();
        let scaled = if actual.is_finite() && wanted.is_finite() {
            (actual - wanted).abs() / (atol + rtol * wanted.abs())
        } else {
            f32::INFINITY
        };
        worst = worst.max(scaled);
        bad += usize::from(scaled > 1.0);
        different += usize::from(actual_bits != bits);
    }
    println!(
        "{name}: outside_budget={bad}/{} different={different} max_scaled={worst} atol={atol} rtol={rtol}",
        expected.len()
    );
    Ok(Check {
        name: name.into(),
        count: expected.len(),
        outside_budget: bad,
        different,
        worst_scaled: worst.is_finite().then_some(worst),
        atol,
        rtol,
    })
}

const ROLES: usize = 759;
const PATCHES: usize = 256;
const PIXEL_WIDTH: usize = 1536;

/// Caption IDs retained from the existing exact 9-ID text diagnostic. This
/// prepared fixture is not a tokenizer/chat-template or raw-image quality test.
fn fixture_ids() -> Vec<u32> {
    let mut ids = vec![32, 11012, 13245, 1752, 28428, 264, 14367, 13, VISION_START];
    ids.extend(std::iter::repeat_n(IMAGE, 64));
    ids.extend([VISION_END, EMBEDDING]);
    ids
}

/// Same synthetic normalized RGB image and temporal repeat as the CUDA HF
/// fixture. All values are exact multiples of 1/128, representable in BF16.
/// Input flattening is [merge_row,merge_col,intra_y,intra_x,C,T,patch_y,patch_x].
#[cube(launch_unchecked)]
fn image_fixture_kernel(out: &mut Array<bf16>, count: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < count {
        let row = i / 1536;
        let component = i % 1536;
        let group = row / 4;
        let intra = row % 4;
        let patch_y = (group / 8) * 2 + intra / 2;
        let patch_x = (group % 8) * 2 + intra % 2;
        let channel = component / 512;
        let pixel = component % 256; // temporal index ignored: duplicate still frame
        let flat = channel * 65536 + (patch_y * 16 + pixel / 16) * 256 + patch_x * 16 + pixel % 16;
        out[i] = bf16::cast_from((f32::cast_from((flat * 17 + 3) % 251) - 125.0f32) / 128.0f32);
    }
}
fn image_fixture() -> CudaTensor {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let count = PATCHES * PIXEL_WIDTH;
    let out = client.empty(count * 2);
    let dim = CubeDim::new_1d(64);
    unsafe {
        image_fixture_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, count, dim),
            dim,
            ArrayArg::from_raw_parts(out.clone(), count),
            count,
        );
    }
    CubeTensor::new_contiguous(
        client,
        device,
        [PATCHES, PIXEL_WIDTH].into(),
        out,
        DType::BF16,
    )
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(
        args.len() == 5,
        "MODEL.pile ROOT IMAGE_REFERENCE.json EXPECTED_CHECKPOINT_SHA256 NEW_REPORT.json"
    );
    anyhow::ensure!(
        std::fs::metadata(&args[2])?.len() <= 512 * 1024 * 1024,
        "reference exceeds512MiB"
    );
    let root = Id::from_hex(
        args[1]
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 root"))?,
    )
    .ok_or_else(|| anyhow::anyhow!("opaque model root must be32hex"))?;
    let f: Reference = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let expected_sha = args[3]
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF8 SHA"))?;
    anyhow::ensure!(
        expected_sha.len() == 64
            && expected_sha.bytes().all(|b| b.is_ascii_hexdigit())
            && f.checkpoint_sha256 == expected_sha
            && f.checkpoint_bytes == 18815757458,
        "checkpoint identity"
    );
    anyhow::ensure!(
        f.oracle == "actual-HF-WeMM-prepared-image-CUDA-v1"
            && f.hf_sha256 == HF
            && f.config_sha256 == CONFIG
            && f.wrapper_sha256 == WRAPPER
            && f.transformers == "5.2.0"
            && hash(f.config_json.as_bytes()) == CONFIG
            && f.generator_sha256 == hash(include_bytes!("../../scripts/wemm_image_reference.py")),
        "oracle identity"
    );
    let sources: BTreeMap<String, String> = SOURCES
        .iter()
        .map(|(n, b)| (n.to_string(), hash(b)))
        .collect();
    let listed: Vec<_> = include_str!("../../scripts/wemm_image_sources.txt")
        .lines()
        .collect();
    anyhow::ensure!(
        listed == SOURCES.iter().map(|(p, _)| *p).collect::<Vec<_>>()
            && sources == f.native_sources,
        "image oracle/native source identity"
    );
    anyhow::ensure!(
        f.ids == fixture_ids() && f.grid == [1, 16, 16],
        "exact image/text input layout"
    );
    let grid = Grid {
        frames: 1,
        height: 16,
        width: 16,
    };
    let plan = ImagePlan::new(&f.ids, grid, 256, 64).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        f.positions.as_slice() == plan.positions() && f.rope_delta == plan.rope_delta(),
        "actual HF MRoPE positions differ"
    );
    let tokens = f.ids.len();
    anyhow::ensure!(
        f.weights.len() == ROLES
            && f.layers.len() == 32
            && f.layers
                .iter()
                .enumerate()
                .all(|(i, l)| l.layer == i && l.hidden.len() == tokens * 4096)
            && f.pixels.len() == PATCHES * PIXEL_WIDTH
            && f.merged.len() == 64 * 4096
            && f.scattered.len() == tokens * 4096
            && f.final_hidden.len() == tokens * 4096
            && f.embedding.len() == 4096,
        "reference shapes/roles"
    );
    let mut names = std::collections::HashSet::new();
    anyhow::ensure!(
        f.weights.iter().all(|w| names.insert(&w.name)),
        "duplicate oracle weight name"
    );
    let config = Qwen3_5Config::from_json(&f.config_json).map_err(anyhow::Error::msg)?;
    let mut report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    let pixels = image_fixture();
    let pixel_bytes = read(&pixels).map_err(anyhow::Error::msg)?;
    let pixel_bits = bits(&pixel_bytes);
    anyhow::ensure!(
        pixel_bits == f.pixels && hash(&pixel_bytes) == f.pixels_sha256,
        "CUDA-generated prepared image bytes differ from actual HF input"
    );
    let model = mary::persist::read_model_pile(Path::new(&args[0]))?;
    let mut selected = Vec::with_capacity(ROLES);
    let mut aliases =
        CudaBf16Aliases::new(CudaDevice { index: 0 }, ROLES).map_err(anyhow::Error::msg)?;
    // SAFETY: immutable operator-owned native model pile, retained until CUDA
    // teardown; same frozen observation/root/binder for both modalities.
    let decoder = unsafe {
        PreparedMultimodal::from_pile(
            &model.facts,
            &model.store,
            root,
            &config,
            &mut aliases,
            |name, handle, shape, bytes| {
                let wanted = f
                    .weights
                    .iter()
                    .find(|w| w.name == name)
                    .ok_or_else(|| format!("oracle lacks {name}"))?;
                let payload_sha256 = hash(bytes);
                if wanted.shape != shape || wanted.sha256 != payload_sha256 {
                    return Err(format!("selected role differs: {name}"));
                }
                selected.push(Selected {
                    name: name.into(),
                    handle: hex(&handle),
                    shape: shape.to_vec(),
                    payload_sha256,
                });
                Ok(())
            },
        )
    }
    .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(selected.len() == ROLES, "selected role count");
    let mut checks = Vec::new();
    let mut layers = Vec::with_capacity(32);
    let output = decoder
        .embed_image_observed(&f.ids, &pixels, grid, |layer, tensor| {
            let (atol, rtol) = if layer % 4 == 3 {
                (0.002, 0.03)
            } else {
                (0.001, 0.02)
            };
            checks.push(compare(
                &format!("layer{layer}"),
                tensor,
                &[1, tokens, 4096],
                &f.layers[layer].hidden,
                atol,
                rtol,
            )?);
            layers.push(Layer {
                layer,
                hidden: bits(&read(tensor)?),
            });
            Ok(())
        })
        .map_err(anyhow::Error::msg)?;
    // Existing vision block/merger budgets, not relaxed for tower composition.
    checks.push(
        compare(
            "vision_merged",
            &output.image_features.features,
            &[64, 4096],
            &f.merged,
            0.001,
            0.02,
        )
        .map_err(anyhow::Error::msg)?,
    );
    checks.push(
        compare(
            "scattered",
            &output.decoder_input,
            &[1, tokens, 4096],
            &f.scattered,
            0.001,
            0.02,
        )
        .map_err(anyhow::Error::msg)?,
    );
    checks.push(
        compare(
            "final_hidden",
            &output.endpoint.final_hidden,
            &[1, tokens, 4096],
            &f.final_hidden,
            0.001,
            0.02,
        )
        .map_err(anyhow::Error::msg)?,
    );
    checks.push(
        compare(
            "embedding",
            &output.endpoint.embedding,
            &[1, 4096],
            &f.embedding,
            0.001,
            0.02,
        )
        .map_err(anyhow::Error::msg)?,
    );
    let merged = bits(&read(&output.image_features.features).map_err(anyhow::Error::msg)?);
    let scattered = bits(&read(&output.decoder_input).map_err(anyhow::Error::msg)?);
    let final_hidden = bits(&read(&output.endpoint.final_hidden).map_err(anyhow::Error::msg)?);
    let native_bytes = read(&output.endpoint.embedding).map_err(anyhow::Error::msg)?;
    let repeat = decoder
        .embed_image(&f.ids, &pixels, grid)
        .map_err(anyhow::Error::msg)?;
    let repeat_bytes = read(&repeat.endpoint.embedding).map_err(anyhow::Error::msg)?;
    let repeat_exact = native_bytes == repeat_bytes;
    let native_bits = bits(&native_bytes);
    let reference_bytes: Vec<_> = f.embedding.iter().flat_map(|b| b.to_le_bytes()).collect();
    let diagnostic = embedding_diagnostic(&native_bits, &f.embedding);
    // Exact-copy obligation is independent of vision numerical error: text
    // rows must equal HF's actual gathered rows, and image rows must equal
    // this invocation's native merged features bit-for-bit and in order.
    let scatter_copy_exact = plan
        .feature_for_token()
        .iter()
        .enumerate()
        .all(|(token, &row)| {
            let actual = &scattered[token * 4096..(token + 1) * 4096];
            let wanted = if row == u32::MAX {
                &f.scattered[token * 4096..(token + 1) * 4096]
            } else {
                &merged[row as usize * 4096..(row as usize + 1) * 4096]
            };
            actual == wanted
        });
    let passed = scatter_copy_exact && repeat_exact && checks.iter().all(|c| c.outside_budget == 0);
    let stats = aliases.stats();
    let result = serde_json::json!({
        "scope":"one GPU-generated prepared still image256patches +75tokens; not raw-image/batch/model admission",
        "pass":passed,"model_root":format!("{root:?}"),"native_sources":sources,
        "checkpoint_sha256":f.checkpoint_sha256,"selected_roles":selected,"ids":f.ids,"grid":f.grid,
        "mask_policy":"explicit all-ones B1 unpadded; ordinary causal token order",
        "positions":plan.positions(),"rope_delta":plan.rope_delta(),
        "prepared_pixel_bits":pixel_bits,"prepared_pixel_sha256":hash(&pixel_bytes),
        "vision_merged":merged,"scattered":scattered,"layers":layers,"final_hidden":final_hidden,
        "hf_vision_rotary_dtype":f.vision_rotary_dtype,"hf_text_rotary_dtype":f.text_rotary_dtype,
        "native_vision_rotary_recipe":"BF16 inv_freq and angle boundaries inherited; mismatch remains gate evidence",
        "checks":checks,"repeat_byte_exact":repeat_exact,"scatter_copy_byte_exact":scatter_copy_exact,
        "vision_output_scope":"merged features after all27blocks; no individual vision-layer witness",
        "embedding_diagnostic_not_admission":diagnostic,"embedding_byte_order":"little-endian BF16",
        "native_embedding_bits":native_bits,"native_embedding_sha256":hash(&native_bytes),
        "repeat_embedding_bits":bits(&repeat_bytes),"repeat_embedding_sha256":hash(&repeat_bytes),
        "reference_embedding_bits":f.embedding,"reference_embedding_sha256":hash(&reference_bytes),
        "alias_registrations":stats.registrations,"registered_span_bytes":stats.registered_span_bytes,
        "owner_capacity_bytes":stats.owner_capacity_bytes});
    serde_json::to_writer_pretty(&mut report, &result)?;
    report.write_all(b"\n")?;
    anyhow::ensure!(
        passed,
        "PREPARED IMAGE FAIL: unchanged numerical/repeat budgets; see report"
    );
    println!("PREPARED IMAGE PASS (one prepared case only; no raw-image/full-model admission)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_gate_fixture_layout_matches_the_bounded_entrypoint() {
        let ids = fixture_ids();
        assert_eq!(ids.len(), 75);
        let p = ImagePlan::new(
            &ids,
            Grid {
                frames: 1,
                height: 16,
                width: 16,
            },
            256,
            64,
        )
        .unwrap();
        assert_eq!(p.positions()[9], [9, 9, 9]);
        assert_eq!(p.positions()[72], [9, 16, 16]);
        assert_eq!(p.positions()[73], [17; 3]);
        assert_eq!(p.positions()[74], [18; 3]);
        assert_eq!(p.rope_delta(), -56);
    }
}
