//! One L1 reset-input experiment; never a replacement for the cumulative gate.
//! MODEL.pile ROOT RESET_HF.json RETAINED_FULL_HF.json ORIGINAL_NATIVE.json NEW_REPORT.json
//! Model computation stays on CUDA. Host work is witness transport and audit.
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use half::bf16;
use mary::{
    models::qwen3_5::{
        gdn_decoder,
        gdn_mixer::{GdnConfig, GdnSlots},
        outer_trace, roles,
    },
    nn::cuda_bf16_alias::CudaBf16Aliases,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::OpenOptions, io::Write, path::Path};
use triblespace::{
    core::{
        blob::{
            Blob,
            encodings::tensor::{Tensor, elements::BF16},
        },
        repo::BlobStoreGet,
    },
    prelude::Id,
};

const HF: &str = "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8";
const CONFIG: &str = "34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004";
const BASELINE: &str = "036b467c8c9b80c7263a6a18737e1f58c01c3932368bbb7bec39432468a6190b";
const CUMULATIVE: &str = "e707ac8393f512df6aa02b16eec14df3d050491895c52558e0e5daa0643dd603";
const CHECKPOINT: &str = "b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4";
const SOURCES: &[(&str, &[u8])] = &[
    ("Cargo.toml", include_bytes!("../../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../../Cargo.lock")),
    ("src/leaf.rs", include_bytes!("../leaf.rs")),
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
        "src/models/qwen3_5/roles.rs",
        include_bytes!("../models/qwen3_5/roles.rs"),
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
        "src/models/qwen3_5/decoder_ops.rs",
        include_bytes!("../models/qwen3_5/decoder_ops.rs"),
    ),
    (
        "src/models/qwen3_5/outer_trace.rs",
        include_bytes!("../models/qwen3_5/outer_trace.rs"),
    ),
    (
        "src/bin/wemm_layer1_reset.rs",
        include_bytes!("wemm_layer1_reset.rs"),
    ),
    (
        "scripts/wemm_layer1_reset_reference.py",
        include_bytes!("../../scripts/wemm_layer1_reset_reference.py"),
    ),
    (
        "scripts/wemm_layer1_reset_sources.txt",
        include_bytes!("../../scripts/wemm_layer1_reset_sources.txt"),
    ),
];
const NAMES: [&str; 9] = [
    "input_norm",
    "mixer_out",
    "residual1",
    "post_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_product",
    "mlp_down",
    "output",
];
#[derive(Deserialize)]
struct Weight {
    name: String,
    shape: Vec<u64>,
    sha256: String,
}
#[derive(Deserialize)]
struct Stage {
    name: String,
    bits: Vec<u16>,
    sha256: String,
}
#[derive(Deserialize)]
struct Reference {
    oracle: String,
    hf_sha256: String,
    config_sha256: String,
    checkpoint_sha256: String,
    baseline_sha256: String,
    native_sources: BTreeMap<String, String>,
    ids: Vec<u32>,
    input_bits: Vec<u16>,
    input_sha256: String,
    hf_reproduces_retained_layer1: bool,
    weights: Vec<Weight>,
    stages: Vec<Stage>,
}
#[derive(Deserialize)]
struct Layer {
    layer: usize,
    hidden: Vec<u16>,
}
#[derive(Deserialize)]
struct Baseline {
    ids: Vec<u32>,
    layers: Vec<Layer>,
}
#[derive(Serialize)]
struct Selected {
    name: String,
    handle: String,
    shape: Vec<u64>,
    payload_sha256: String,
}
#[derive(Serialize)]
struct Comparison {
    name: String,
    count: usize,
    different: usize,
    outside_budget: usize,
    worst_scaled: Option<f32>,
    native_bits: Vec<u16>,
    native_sha256: String,
    reference_bits: Vec<u16>,
    reference_sha256: String,
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn raw(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn read(t: &CubeTensor<CudaRuntime>, shape: &[usize]) -> anyhow::Result<Vec<u16>> {
    anyhow::ensure!(
        t.dtype == DType::BF16 && t.meta.shape().as_slice() == shape,
        "stage shape/dtype"
    );
    let bytes = t
        .client
        .read_one(t.handle.clone())
        .map_err(|e| anyhow::anyhow!("GPU read: {e:?}"))?;
    anyhow::ensure!(
        bytes.len() == shape.iter().product::<usize>() * 2,
        "stage extent"
    );
    Ok(bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn compare(name: &str, actual: Vec<u16>, expected: &[u16]) -> anyhow::Result<Comparison> {
    anyhow::ensure!(
        actual.len() == expected.len() && !actual.is_empty(),
        "comparison extent"
    );
    let (mut outside, mut different, mut worst) = (0, 0, 0.0f32);
    for (&a, &b) in actual.iter().zip(expected) {
        let (x, y) = (bf16::from_bits(a).to_f32(), bf16::from_bits(b).to_f32());
        let scaled = if x.is_finite() && y.is_finite() {
            (x - y).abs() / (0.001 + 0.02 * y.abs())
        } else {
            f32::INFINITY
        };
        outside += usize::from(scaled > 1.0);
        different += usize::from(a != b);
        worst = worst.max(scaled);
    }
    Ok(Comparison {
        name: name.into(),
        count: actual.len(),
        different,
        outside_budget: outside,
        worst_scaled: worst.is_finite().then_some(worst),
        native_sha256: hash(&raw(&actual)),
        reference_sha256: hash(&raw(expected)),
        native_bits: actual,
        reference_bits: expected.to_vec(),
    })
}
fn json_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        std::fs::metadata(path)?.len() < 64 * 1024 * 1024,
        "input exceeds64MiB"
    );
    Ok(std::fs::read(path)?)
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(
        args.len() == 6,
        "MODEL ROOT RESET_HF RETAINED_FULL_HF ORIGINAL_NATIVE NEW_REPORT"
    );
    let root = Id::from_hex(
        args[1]
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("root UTF8"))?,
    )
    .ok_or_else(|| anyhow::anyhow!("root must be32hex"))?;
    let reference_bytes = json_bytes(Path::new(&args[2]))?;
    let baseline_bytes = json_bytes(Path::new(&args[3]))?;
    let cumulative_bytes = json_bytes(Path::new(&args[4]))?;
    anyhow::ensure!(
        hash(&baseline_bytes) == BASELINE && hash(&cumulative_bytes) == CUMULATIVE,
        "retained evidence identity"
    );
    let f: Reference = serde_json::from_slice(&reference_bytes)?;
    let base: Baseline = serde_json::from_slice(&baseline_bytes)?;
    let original: serde_json::Value = serde_json::from_slice(&cumulative_bytes)?;
    anyhow::ensure!(
        original["pass"] == false,
        "original cumulative result must remain FAIL"
    );
    anyhow::ensure!(
        f.oracle == "actual-HF-WeMM-layer1-reset-CUDA-v1"
            && f.hf_sha256 == HF
            && f.config_sha256 == CONFIG
            && f.checkpoint_sha256 == CHECKPOINT
            && f.baseline_sha256 == BASELINE,
        "reset reference identity"
    );
    // Ephemeral source/evidence maps for this diagnostic invocation only.
    let sources: BTreeMap<String, String> = SOURCES
        .iter()
        .map(|(p, b)| (p.to_string(), hash(b)))
        .collect();
    anyhow::ensure!(
        sources == f.native_sources
            && include_str!("../../scripts/wemm_layer1_reset_sources.txt")
                .lines()
                .eq(SOURCES.iter().map(|(p, _)| *p)),
        "compiled source identity"
    );
    anyhow::ensure!(
        f.ids.len() == 9
            && f.ids == base.ids
            && base.layers.len() == 32
            && base.layers[0].layer == 0
            && base.layers[1].layer == 1
            && f.input_bits == base.layers[0].hidden
            && f.input_bits.len() == 9 * 4096
            && hash(&raw(&f.input_bits)) == f.input_sha256,
        "retained input bits"
    );
    anyhow::ensure!(
        f.hf_reproduces_retained_layer1
            && f.stages.len() == 9
            && f.stages.iter().zip(NAMES).all(|(s, n)| s.name == n)
            && f.stages[8].bits == base.layers[1].hidden,
        "HF reset must reproduce original HF L1 exactly"
    );
    for (i, s) in f.stages.iter().enumerate() {
        anyhow::ensure!(
            s.bits.len() == 9 * if (4..=6).contains(&i) { 12288 } else { 4096 }
                && hash(&raw(&s.bits)) == s.sha256,
            "stage identity/extent"
        );
    }
    let mut unique = std::collections::HashSet::new();
    anyhow::ensure!(
        f.weights.len() == 14 && f.weights.iter().all(|w| unique.insert(&w.name)),
        "oracle fourteen unique roles"
    );
    let mut report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[5])?;
    let model = mary::persist::read_model_pile(Path::new(&args[0]))?;
    let mut selected = Vec::with_capacity(14);
    let mut select =
        |name: &str, handle: [u8; 32], shape: &[u64], bytes: &[u8]| -> Result<(), String> {
            let expected = f
                .weights
                .iter()
                .find(|w| w.name == name)
                .ok_or_else(|| format!("unreferenced role{name}"))?;
            let payload_sha256 = hash(bytes);
            if expected.shape != shape || expected.sha256 != payload_sha256 {
                return Err(format!("{name}: checkpoint role mismatch"));
            }
            selected.push(Selected {
                name: name.into(),
                handle: handle.iter().map(|b| format!("{b:02x}")).collect(),
                shape: shape.to_vec(),
                payload_sha256,
            });
            Ok(())
        };
    macro_rules! role {
        ($name:literal,$shape:expr) => {
            roles::resolve(
                &model.facts,
                &model.store,
                root,
                concat!("model.language_model.layers.1.", $name),
                &$shape,
                &mut select,
            )
            .map_err(anyhow::Error::msg)?
        };
    }
    let slots = gdn_decoder::Slots {
        input_norm: role!("input_layernorm.weight", [4096]),
        post_norm: role!("post_attention_layernorm.weight", [4096]),
        gate: role!("mlp.gate_proj.weight", [12288, 4096]),
        up: role!("mlp.up_proj.weight", [12288, 4096]),
        down: role!("mlp.down_proj.weight", [4096, 12288]),
        mixer: GdnSlots {
            qkv: role!("linear_attn.in_proj_qkv.weight", [8192, 4096]),
            z: role!("linear_attn.in_proj_z.weight", [4096, 4096]),
            a: role!("linear_attn.in_proj_a.weight", [32, 4096]),
            b: role!("linear_attn.in_proj_b.weight", [32, 4096]),
            out: role!("linear_attn.out_proj.weight", [4096, 4096]),
            conv: role!("linear_attn.conv1d.weight", [8192, 1, 4]),
            a_log: role!("linear_attn.A_log", [32]),
            dt_bias: role!("linear_attn.dt_bias", [32]),
            norm: role!("linear_attn.norm.weight", [128]),
        },
    };
    anyhow::ensure!(selected.len() == 14, "selected roles");
    let mut aliases =
        CudaBf16Aliases::new(CudaDevice { index: 0 }, 14).map_err(anyhow::Error::msg)?;
    let anchor_blob: Blob<Tensor<BF16, 1>> = model.store.get(slots.input_norm)?;
    // SAFETY: operator provides the same genuine immutable model pile prefix
    // used by the original run, never rewritten/truncated through CUDA teardown.
    let anchor = unsafe { aliases.bind_pile_leaf(anchor_blob) }.map_err(anyhow::Error::msg)?;
    let config = gdn_decoder::Config {
        mixer: GdnConfig {
            hidden: 4096,
            key_heads: 16,
            value_heads: 32,
            key_dim: 128,
            value_dim: 128,
            conv_kernel: 4,
            epsilon: 1e-6,
        },
        intermediate: 12288,
    };
    let block = unsafe { gdn_decoder::Block::from_pile(&model.store, slots, config, &mut aliases) }
        .map_err(anyhow::Error::msg)?;
    // Only reference activation bytes are uploaded; weights remain native pile aliases.
    let input = CubeTensor::new_contiguous(
        anchor.client.clone(),
        anchor.device.clone(),
        [1, 9, 4096].as_slice().into(),
        anchor.client.create_from_slice(&raw(&f.input_bits)),
        DType::BF16,
    );
    anyhow::ensure!(
        read(&input, &[1, 9, 4096])? == f.input_bits,
        "input transport changed bits"
    );
    let mut trace = outer_trace::OuterTrace::default();
    let out = block
        .prefill_observed(&input, &mut trace)
        .map_err(anyhow::Error::msg)?;
    let mut comparisons = Vec::with_capacity(9);
    for i in 0..8 {
        let width = if (4..=6).contains(&i) { 12288 } else { 4096 };
        comparisons.push(compare(
            NAMES[i],
            read(trace.get(i).map_err(anyhow::Error::msg)?, &[1, 9, width])?,
            &f.stages[i].bits,
        )?);
    }
    comparisons.push(compare(
        "output",
        read(&out.hidden, &[1, 9, 4096])?,
        &f.stages[8].bits,
    )?);
    anyhow::ensure!(
        read(&input, &[1, 9, 4096])? == f.input_bits,
        "diagnostic input was mutated"
    );
    let local_pass = comparisons[8].outside_budget == 0;
    let result = serde_json::json!({"scope":"single L1 reset-input diagnostic; no whole-model admission",
        "local_output_within_prior_L1_budget":local_pass,"atol":0.001,"rtol":0.02,
        "intermediate_bounds_are_diagnostic_only":true,"original_cumulative_report":original,
        "original_cumulative_sha256":CUMULATIVE,"baseline_sha256":BASELINE,
        "reset_reference_sha256":hash(&reference_bytes),"checkpoint_sha256":CHECKPOINT,
        "native_sources":sources,"model_root":format!("{root:?}"),"selected_roles":selected,
        "input_bits":f.input_bits,"input_sha256":f.input_sha256,"comparisons":comparisons,
        "alias_registrations":aliases.stats().registrations});
    serde_json::to_writer_pretty(&mut report, &result)?;
    report.write_all(b"\n")?;
    println!(
        "L1 RESET local_output_within_prior_budget={local_pass}; original cumulative FAIL remains unchanged"
    );
    anyhow::ensure!(
        local_pass,
        "L1 RESET outside unchanged local bound; report preserved"
    );
    Ok(())
}
