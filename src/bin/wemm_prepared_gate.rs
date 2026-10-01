//! Source-bound actual-HF comparison of WeMM's prepared-token decoder leg.
//! MODEL.pile ROOT REFERENCE.json EXPECTED_CHECKPOINT_SHA256 NEW_REPORT.json
//! The model pile is a pre-imported immutable native-BF16 input, never rewritten.
//! This gate performs CPU byte hashing/audit comparisons, not CPU model math.
use burn::tensor::DType;
use cubecl::cuda::CudaDevice;
use half::bf16;
use mary::{models::qwen3_5::prepared::{self, CudaTensor, PreparedDecoder},
    nn::cuda_bf16_alias::CudaBf16Aliases};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::OpenOptions, io::Write, path::Path};
use triblespace::prelude::Id;

const HF: &str = "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8";
const CONFIG: &str = "34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004";
const WRAPPER: &str = "ac255e1fad459cc3e68891d6c3327f4486922aed02fb3c5c13fb53277ba8e94f";

// This exact list is mirrored by scripts/wemm_prepared_sources.txt, checked
// before any GPU load. Sources are embedded by rustc, not read from cwd later.
const SOURCES: &[(&str, &[u8])] = &[
    ("Cargo.toml", include_bytes!("../../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../../Cargo.lock")),
    ("src/leaf.rs", include_bytes!("../leaf.rs")),
    ("src/format.rs", include_bytes!("../format.rs")),
    ("src/ingest.rs", include_bytes!("../ingest.rs")),
    ("src/persist.rs", include_bytes!("../persist.rs")),
    ("src/model_collection.rs", include_bytes!("../model_collection.rs")),
    ("src/nn/cuda_bf16_alias.rs", include_bytes!("../nn/cuda_bf16_alias.rs")),
    ("src/models/qwen3_5/mod.rs", include_bytes!("../models/qwen3_5/mod.rs")),
    ("src/models/qwen3_5/config.rs", include_bytes!("../models/qwen3_5/config.rs")),
    ("src/models/qwen3_5/layout.rs", include_bytes!("../models/qwen3_5/layout.rs")),
    ("src/models/qwen3_5/deltanet.rs", include_bytes!("../models/qwen3_5/deltanet.rs")),
    ("src/models/qwen3_5/gdn_ops.rs", include_bytes!("../models/qwen3_5/gdn_ops.rs")),
    ("src/models/qwen3_5/gdn_mixer.rs", include_bytes!("../models/qwen3_5/gdn_mixer.rs")),
    ("src/models/qwen3_5/gdn_decoder.rs", include_bytes!("../models/qwen3_5/gdn_decoder.rs")),
    ("src/models/qwen3_5/full_attention.rs", include_bytes!("../models/qwen3_5/full_attention.rs")),
    ("src/models/qwen3_5/decoder_ops.rs", include_bytes!("../models/qwen3_5/decoder_ops.rs")),
    ("src/models/qwen3_5/decoder_stack.rs", include_bytes!("../models/qwen3_5/decoder_stack.rs")),
    ("src/models/qwen3_5/outer_trace.rs", include_bytes!("../models/qwen3_5/outer_trace.rs")),
    ("src/models/qwen3_5/embedding_boundary.rs", include_bytes!("../models/qwen3_5/embedding_boundary.rs")),
    ("src/models/qwen3_5/roles.rs", include_bytes!("../models/qwen3_5/roles.rs")),
    ("src/models/qwen3_5/prepared.rs", include_bytes!("../models/qwen3_5/prepared.rs")),
    ("src/models/qwen3_5/vision_geometry.rs", include_bytes!("../models/qwen3_5/vision_geometry.rs")),
    ("src/models/qwen3_5/vision_frontend.rs", include_bytes!("../models/qwen3_5/vision_frontend.rs")),
    ("src/models/qwen3_5/vision_block.rs", include_bytes!("../models/qwen3_5/vision_block.rs")),
    ("src/models/qwen3_5/vision_merger.rs", include_bytes!("../models/qwen3_5/vision_merger.rs")),
    ("src/models/qwen3_5/vision_tower.rs", include_bytes!("../models/qwen3_5/vision_tower.rs")),
    ("src/models/qwen3_5/multimodal_layout.rs", include_bytes!("../models/qwen3_5/multimodal_layout.rs")),
    ("src/models/qwen3_5/position_table.rs", include_bytes!("../models/qwen3_5/position_table.rs")),
    ("src/models/qwen3_5/multimodal.rs", include_bytes!("../models/qwen3_5/multimodal.rs")),
    ("src/bin/wemm_prepared_gate.rs", include_bytes!("wemm_prepared_gate.rs")),
    ("scripts/wemm_prepared_reference.py", include_bytes!("../../scripts/wemm_prepared_reference.py")),
    ("scripts/wemm_prepared_sources.txt", include_bytes!("../../scripts/wemm_prepared_sources.txt")),
];

#[derive(Deserialize)]
struct Weight { name: String, shape: Vec<u64>, sha256: String }
#[derive(Deserialize)]
struct Layer { layer: usize, hidden: Vec<u16> }
#[derive(Deserialize)]
struct Reference {
    oracle: String, hf_sha256: String, config_sha256: String, wrapper_sha256: String,
    generator_sha256: String, native_sources: BTreeMap<String, String>,
    checkpoint_sha256: String, checkpoint_bytes: u64, transformers: String,
    ids: Vec<u32>, weights: Vec<Weight>, gathered: Vec<u16>, layers: Vec<Layer>,
    final_hidden: Vec<u16>, embedding: Vec<u16>,
}
#[derive(Serialize)]
struct Selected { name: String, handle: String, shape: Vec<u64>, payload_sha256: String }
#[derive(Serialize)]
struct Check { name: String, count: usize, outside_budget: usize, different: usize,
    worst_scaled: Option<f32>, atol: f32, rtol: f32 }
#[derive(Serialize)]
struct EmbeddingDiagnostic { cosine: Option<f64>, angular_error_radians: Option<f64>,
    native_l2: Option<f64>, reference_l2: Option<f64> }
fn hash(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
fn hex(b: &[u8]) -> String { b.iter().map(|v| format!("{v:02x}")).collect() }
fn bits(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|b| u16::from_le_bytes(b.try_into().unwrap())).collect()
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
    let cosine = (native.len() == reference.len() && !native.is_empty()
        && dot.is_finite() && nn.is_finite() && rr.is_finite() && nn > 0.0 && rr > 0.0)
        .then(|| (dot / (nn.sqrt() * rr.sqrt())).clamp(-1.0, 1.0));
    EmbeddingDiagnostic { cosine, angular_error_radians: cosine.map(f64::acos),
        native_l2: nn.is_finite().then(|| nn.sqrt()),
        reference_l2: rr.is_finite().then(|| rr.sqrt()) }
}
fn read(t: &CudaTensor) -> Result<Vec<u8>, String> {
    t.client.read_one(t.handle.clone()).map(|b| b.to_vec()).map_err(|e| format!("GPU read: {e:?}"))
}
fn compare(name: &str, t: &CudaTensor, shape: &[usize], expected: &[u16], atol: f32, rtol: f32)
    -> Result<Check, String> {
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
        } else { f32::INFINITY };
        worst = worst.max(scaled);
        bad += usize::from(scaled > 1.0);
        different += usize::from(actual_bits != bits);
    }
    println!("{name}: outside_budget={bad}/{} different={different} max_scaled={worst} atol={atol} rtol={rtol}", expected.len());
    Ok(Check { name: name.into(), count: expected.len(), outside_budget: bad,
        different, worst_scaled: worst.is_finite().then_some(worst), atol, rtol })
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(args.len() == 5, "MODEL.pile ROOT REFERENCE.json EXPECTED_CHECKPOINT_SHA256 NEW_REPORT.json");
    anyhow::ensure!(std::fs::metadata(&args[2])?.len() <= 512 * 1024 * 1024, "reference exceeds512MiB");
    let root = Id::from_hex(args[1].to_str().ok_or_else(|| anyhow::anyhow!("non-UTF8 root"))?)
        .ok_or_else(|| anyhow::anyhow!("opaque model root must be32hex"))?;
    let f: Reference = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let expected_sha = args[3].to_str().ok_or_else(|| anyhow::anyhow!("non-UTF8 SHA"))?;
    anyhow::ensure!(expected_sha.len() == 64 && expected_sha.bytes().all(|b| b.is_ascii_hexdigit())
        && f.checkpoint_sha256 == expected_sha && f.checkpoint_bytes == 18815757458, "checkpoint identity");
    anyhow::ensure!(f.oracle == "actual-HF-WeMM-prepared-decoder-CUDA-v1" && f.hf_sha256 == HF
        && f.config_sha256 == CONFIG && f.wrapper_sha256 == WRAPPER && f.transformers == "5.2.0"
        && f.generator_sha256 == hash(include_bytes!("../../scripts/wemm_prepared_reference.py")), "oracle identity");
    // Ephemeral maps of audit metadata for THIS invocation only.
    let sources: BTreeMap<String, String> = SOURCES.iter().map(|(name, bytes)| (name.to_string(), hash(bytes))).collect();
    let listed: Vec<_> = include_str!("../../scripts/wemm_prepared_sources.txt").lines().collect();
    anyhow::ensure!(listed == SOURCES.iter().map(|(p, _)| *p).collect::<Vec<_>>()
        && sources == f.native_sources, "native source list/hash differs from the oracle's frozen candidate");
    prepared::validate_ids(&f.ids).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(f.weights.len() == prepared::WEIGHT_ROLES && f.layers.len() == 32
        && f.layers.iter().enumerate().all(|(i, l)| i == l.layer && l.hidden.len() == f.ids.len() * 4096)
        && f.gathered.len() == f.ids.len() * 4096 && f.final_hidden.len() == f.gathered.len()
        && f.embedding.len() == 4096, "reference shapes/roles");
    let mut names = std::collections::HashSet::new();
    anyhow::ensure!(f.weights.iter().all(|w| names.insert(&w.name)), "duplicate oracle weight name");
    // Reserve the new report before initialization; no existing evidence can
    // be overwritten. An operational failure may leave it empty, never PASS.
    let mut report = OpenOptions::new().write(true).create_new(true).open(&args[4])?;
    let model = mary::persist::read_model_pile(Path::new(&args[0]))?;
    let mut selected = Vec::with_capacity(prepared::WEIGHT_ROLES);
    let mut aliases = CudaBf16Aliases::new(CudaDevice { index: 0 }, prepared::WEIGHT_ROLES)
        .map_err(anyhow::Error::msg)?;
    // SAFETY: operator provides a genuine validated immutable model pile; its
    // selected prefix must not be truncated/rewritten until process teardown.
    let decoder = unsafe { PreparedDecoder::from_pile(&model.facts, &model.store, root, &mut aliases,
        |name, handle, shape, bytes| {
            let expected = f.weights.iter().find(|w| w.name == name)
                .ok_or_else(|| format!("oracle lacks role {name}"))?;
            let payload_sha256 = hash(bytes);
            if expected.shape != shape || expected.sha256 != payload_sha256 {
                return Err(format!("{name}: selected native leaf does not match exact checkpoint role"));
            }
            selected.push(Selected { name: name.into(), handle: hex(&handle), shape: shape.to_vec(), payload_sha256 });
            Ok(())
        }) }.map_err(anyhow::Error::msg)?;
    anyhow::ensure!(selected.len() == prepared::WEIGHT_ROLES, "selected role count");
    let mut checks = Vec::with_capacity(35);
    let output = decoder.embed_unpadded_observed(&f.ids, |layer, tensor| {
        // Preserve prior actual L0/L3 budgets for their respective block kinds.
        // No widening for composition or substituting cosine for coordinate error.
        let (atol, rtol) = if layer % 4 == 3 { (0.002, 0.03) } else { (0.001, 0.02) };
        checks.push(compare(&format!("layer{layer}"), tensor, &[1, f.ids.len(), 4096],
            &f.layers[layer].hidden, atol, rtol)?);
        Ok(())
    }).map_err(anyhow::Error::msg)?;
    checks.push(compare("gather", &output.gathered, &[1, f.ids.len(), 4096], &f.gathered, 0.001, 0.02).map_err(anyhow::Error::msg)?);
    let gather_exact = checks.last().unwrap().different == 0;
    checks.push(compare("final_hidden", &output.endpoint.final_hidden, &[1, f.ids.len(), 4096],
        &f.final_hidden, 0.001, 0.02).map_err(anyhow::Error::msg)?);
    checks.push(compare("embedding", &output.endpoint.embedding, &[1,4096],
        &f.embedding, 0.001, 0.02).map_err(anyhow::Error::msg)?);
    let repeat = decoder.embed_unpadded(&f.ids).map_err(anyhow::Error::msg)?;
    let native_bytes = read(&output.endpoint.embedding).map_err(anyhow::Error::msg)?;
    let repeat_bytes = read(&repeat.endpoint.embedding).map_err(anyhow::Error::msg)?;
    let reference_bytes: Vec<u8> = f.embedding.iter().flat_map(|b| b.to_le_bytes()).collect();
    let native_bits = bits(&native_bytes);
    let diagnostic = embedding_diagnostic(&native_bits, &f.embedding);
    let repeat_exact = native_bytes == repeat_bytes;
    let passed = gather_exact && repeat_exact && checks.iter().all(|c| c.outside_budget == 0);
    let stats = aliases.stats();
    let result = serde_json::json!({"scope":"prepared B1 unpadded decoder, not complete multimodal/batch admission",
        "pass":passed, "model_root":format!("{root:?}"), "selected_roles":selected,
        "native_sources":sources, "checkpoint_sha256":f.checkpoint_sha256, "ids":f.ids,
        "gather_byte_exact":gather_exact,"repeat_byte_exact":repeat_exact, "checks":checks,
        "embedding_diagnostic_not_admission":diagnostic,
        "embedding_byte_order":"little-endian BF16",
        "native_embedding_bits":native_bits,"native_embedding_sha256":hash(&native_bytes),
        "repeat_embedding_bits":bits(&repeat_bytes),"repeat_embedding_sha256":hash(&repeat_bytes),
        "reference_embedding_bits":f.embedding,"reference_embedding_sha256":hash(&reference_bytes),
        "alias_registrations":stats.registrations,"registered_span_bytes":stats.registered_span_bytes,
        "owner_capacity_bytes":stats.owner_capacity_bytes});
    serde_json::to_writer_pretty(&mut report, &result)?;
    report.write_all(b"\n")?;
    anyhow::ensure!(passed, "PREPARED DECODER FAIL: numerical/repeat budgets retained; see report");
    println!("PREPARED DECODER PASS (B1 only; no vision/batch/full-model admission)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedding_diagnostics_keep_raw_bits_and_do_not_replace_admission() {
        let one = bf16::from_f32(1.0).to_bits();
        let minus = bf16::from_f32(-1.0).to_bits();
        let bytes: Vec<u8> = [one, minus].into_iter().flat_map(u16::to_le_bytes).collect();
        assert_eq!(bits(&bytes), [one, minus]);
        let same = embedding_diagnostic(&[one, 0], &[one, 0]);
        assert_eq!(same.cosine, Some(1.0));
        assert_eq!(same.angular_error_radians, Some(0.0));
        let opposite = embedding_diagnostic(&[one, 0], &[minus, 0]);
        assert_eq!(opposite.cosine, Some(-1.0));
        assert_eq!(opposite.angular_error_radians, Some(std::f64::consts::PI));
        assert!(embedding_diagnostic(&[0, 0], &[one, 0]).cosine.is_none());
        assert!(embedding_diagnostic(&[0x7fc0], &[one]).cosine.is_none());
    }
}
