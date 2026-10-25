//! One-shot, text-only import of a local ModelOpt HF checkpoint.
//!
//! The directory must remain immutable for this call (mmap has the same
//! requirement as Mary's other importers). Only the current tensor and its
//! companions are retained. HF names/indexes are ingress data, not a runtime
//! parameter catalog. A failed import can leave unreferenced blobs, never a
//! published incomplete model. No downloads, default collection, or key lookup.

use super::{
    config::{CheckpointConfig, HfQuantConfig, LayerKind, LinearQuantization},
    native, packing,
};
use crate::{
    format::attrs,
    jsonfacts,
    model_collection::{self, ModelCollection},
    tokenizer,
};
use anybytes::Bytes;
use anyhow::{Context, Result, bail, ensure};
use ed25519_dalek::SigningKey;
use safetensors::{Dtype, SafeTensors, tensor::Metadata};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Component, Path},
};
use triblespace::{core::collection::CollectionCommit, prelude::*};

pub const SOURCE: &str = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4";
pub const REVISION: &str = "da5b2650a835292c678994a1fa3fb653250fb508";
pub const CONFIG: &str = "nemotron-h-config";
pub const LAYER: &str = "nemotron-h-layer";
pub const EXPERT: &str = "nemotron-h-expert";

pub struct ImportedCheckpoint {
    pub root: Id,
    pub tokenizer: Id,
    pub commit: CollectionCommit,
    pub operations: usize,
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

fn json(directory: &Path, name: &str) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(directory.join(name))?)
        .with_context(|| format!("decode {name}"))
}

fn selected(name: &str) -> bool {
    name.starts_with("language_model.backbone.") || name.starts_with("language_model.lm_head.")
}

struct SourceTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Bytes,
}
impl SourceTensor {
    fn plane(&self) -> Result<packing::Plane<'_>> {
        let dtype = match self.dtype {
            Dtype::U8 => packing::Dtype::U8,
            Dtype::F8_E4M3 => packing::Dtype::E4m3Fn,
            Dtype::F32 => packing::Dtype::F32,
            other => bail!("unsupported quantized plane dtype {other:?}"),
        };
        Ok(packing::Plane {
            dtype,
            shape: &self.shape,
            bytes: &self.bytes,
        })
    }
}

/// Caches only validated safetensors headers, never all weight payloads.
struct Shards<'a> {
    directory: &'a Path,
    index: &'a Index,
    headers: BTreeMap<String, (usize, Metadata)>,
    current: Option<(String, Bytes)>,
}
impl Shards<'_> {
    fn tensor(&mut self, name: &str) -> Result<SourceTensor> {
        let file = self
            .index
            .weight_map
            .get(name)
            .with_context(|| format!("incomplete import: missing {name}"))?;
        if self.current.as_ref().map(|(n, _)| n) != Some(file) {
            let handle = File::open(self.directory.join(file))
                .with_context(|| format!("open shard {file}"))?;
            // Source checkpoint is immutable for the duration of import.
            let mapped = unsafe { memmap2::Mmap::map(&handle) }?;
            let bytes = Bytes::from_source(mapped);
            if !self.headers.contains_key(file) {
                self.headers
                    .insert(file.clone(), SafeTensors::read_metadata(&bytes)?);
            }
            self.current = Some((file.clone(), bytes));
        }
        let (header, metadata) = &self.headers[file];
        let info = metadata
            .info(name)
            .with_context(|| format!("index names absent tensor {name} in {file}"))?;
        let bytes = &self.current.as_ref().expect("mapped shard").1;
        let start = header
            .checked_add(8)
            .and_then(|n| n.checked_add(info.data_offsets.0))
            .context("tensor offset overflow")?;
        let end = header
            .checked_add(8)
            .and_then(|n| n.checked_add(info.data_offsets.1))
            .context("tensor offset overflow")?;
        ensure!(end <= bytes.len(), "shard changed during import: {file}");
        Ok(SourceTensor {
            dtype: info.dtype,
            shape: info.shape.clone(),
            bytes: bytes.slice(start..end),
        })
    }
}

struct Importer<'a> {
    pile: &'a mut Pile,
    shards: Shards<'a>,
    config: &'a CheckpointConfig,
    quant: &'a HfQuantConfig,
    consumed: BTreeSet<String>,
    dry_run: bool,
    operations: usize,
}

impl Importer<'_> {
    fn operation(
        &mut self,
        source: &str,
        role: &str,
        shape: &[usize],
        quantized: bool,
    ) -> Result<Fragment> {
        let module = source.strip_suffix(".weight").unwrap_or(source);
        let a = self
            .config
            .quantization_config
            .text_linear(module)
            .map_err(anyhow::Error::msg)?;
        let b = self
            .quant
            .quantization
            .text_linear(module)
            .map_err(anyhow::Error::msg)?;
        ensure!(a == b, "selected quantization disagreement for {module}");
        ensure!(
            a.is_some() == quantized,
            "incomplete import: {module} expected {}",
            if quantized {
                "an explicit quantization selector"
            } else {
                "an unquantized auxiliary"
            }
        );
        let mut names = vec![source.to_owned()];
        if let Some(q) = a {
            names.extend([
                format!("{module}.weight_scale"),
                format!("{module}.input_scale"),
            ]);
            if matches!(q, LinearQuantization::Nvfp4 { .. }) {
                names.push(format!("{module}.weight_scale_2"));
            }
        }
        for name in &names {
            ensure!(
                self.shards.index.weight_map.contains_key(name),
                "incomplete import: missing {name}"
            );
            self.consumed.insert(name.clone());
        }
        self.operations += 1;
        if self.dry_run {
            return Ok(Fragment::empty());
        }
        let weight = self.shards.tensor(source)?;
        let mut operation = match a {
            Some(q) => {
                let dims: [usize; 2] = shape.try_into().context("linear rank must be two")?;
                let scale = self.shards.tensor(&names[1])?;
                let input = self.shards.tensor(&names[2])?;
                match q {
                    LinearQuantization::Fp8 => native::store_fp8(
                        self.pile,
                        role,
                        packing::pack_fp8(dims, weight.plane()?, scale.plane()?, input.plane()?)
                            .map_err(anyhow::Error::msg)?,
                    )?,
                    LinearQuantization::Nvfp4 { .. } => {
                        let global = self.shards.tensor(&names[3])?;
                        native::store_nvfp4(
                            self.pile,
                            role,
                            packing::pack_nvfp4(
                                dims,
                                weight.plane()?,
                                scale.plane()?,
                                global.plane()?,
                                input.plane()?,
                            )
                            .map_err(anyhow::Error::msg)?,
                        )?
                    }
                }
            }
            None => {
                ensure!(
                    weight.shape == shape,
                    "{source}: expected {shape:?}, got {:?}",
                    weight.shape
                );
                macro_rules! store {
                    ($rank:literal) => {{
                        let dims: [u64; $rank] = shape
                            .iter()
                            .map(|&n| n as u64)
                            .collect::<Vec<_>>()
                            .try_into()
                            .expect("matched rank");
                        match weight.dtype {
                            Dtype::BF16 => native::store_bf16(self.pile, role, dims, weight.bytes)?,
                            Dtype::F32 => native::store_f32(self.pile, role, dims, weight.bytes)?,
                            other => {
                                bail!("{source}: auxiliary dtype {other:?}, expected BF16/F32")
                            }
                        }
                    }};
                }
                match shape.len() {
                    0 => store!(0),
                    1 => store!(1),
                    2 => store!(2),
                    3 => store!(3),
                    _ => bail!("unsupported auxiliary rank"),
                }
            }
        };
        let root = operation.root().expect("operation root");
        operation += entity! { ExclusiveId::force_ref(&root) @ attrs::safetensor_path: source };
        Ok(operation)
    }

    fn graph(&mut self) -> Result<Fragment> {
        let c = self.config.llm_config.clone();
        ensure!(
            !c.tie_word_embeddings
                && !c.use_bias
                && !c.mamba_proj_bias
                && !c.mlp_bias
                && !c.attention_bias,
            "unsupported text bias/tied-embedding configuration"
        );
        ensure!(
            c.moe_latent_size > 0 && c.n_shared_experts == 1,
            "unsupported latent/shared-expert configuration"
        );
        let h = c.hidden_size;
        let inner = c
            .mamba_num_heads
            .checked_mul(c.mamba_head_dim)
            .context("Mamba width overflow")?;
        let state = c
            .n_groups
            .checked_mul(c.ssm_state_size)
            .and_then(|n| n.checked_mul(2))
            .context("Mamba state width overflow")?;
        let conv = inner
            .checked_add(state)
            .context("Mamba conv width overflow")?;
        let projection = inner
            .checked_add(conv)
            .and_then(|n| n.checked_add(c.mamba_num_heads))
            .context("Mamba projection overflow")?;
        let q = c
            .num_attention_heads
            .checked_mul(c.head_dim)
            .context("attention width overflow")?;
        let kv = c
            .num_key_value_heads
            .checked_mul(c.head_dim)
            .context("attention KV width overflow")?;
        let mut members = self.operation(
            "language_model.backbone.embeddings.weight",
            "embedding",
            &[c.vocab_size, h],
            false,
        )?;
        members += self.operation(
            "language_model.backbone.norm_f.weight",
            "final-norm",
            &[h],
            false,
        )?;
        members += self.operation(
            "language_model.lm_head.weight",
            "lm-head",
            &[c.vocab_size, h],
            true,
        )?;
        for (index, kind) in c.layers_block_type.iter().enumerate() {
            let prefix = format!("language_model.backbone.layers.{index}");
            let mut ops =
                self.operation(&format!("{prefix}.norm.weight"), "pre-norm", &[h], false)?;
            let mixer = format!("{prefix}.mixer");
            match kind {
                LayerKind::Mamba => {
                    for (name, role, dims, quant) in [
                        ("in_proj.weight", "mamba.in_proj", vec![projection, h], true),
                        ("out_proj.weight", "mamba.out_proj", vec![h, inner], true),
                        (
                            "conv1d.weight",
                            "mamba.conv1d.weight",
                            vec![conv, 1, c.conv_kernel],
                            false,
                        ),
                        ("norm.weight", "mamba.norm", vec![inner], false),
                        ("A_log", "mamba.A_log", vec![c.mamba_num_heads], false),
                        ("D", "mamba.D", vec![c.mamba_num_heads], false),
                        ("dt_bias", "mamba.dt_bias", vec![c.mamba_num_heads], false),
                    ] {
                        ops += self.operation(&format!("{mixer}.{name}"), role, &dims, quant)?;
                    }
                    if c.use_conv_bias {
                        ops += self.operation(
                            &format!("{mixer}.conv1d.bias"),
                            "mamba.conv1d.bias",
                            &[conv],
                            false,
                        )?;
                    }
                }
                LayerKind::Attention => {
                    for (name, role, dims) in [
                        ("q_proj", "attention.q_proj", [q, h]),
                        ("k_proj", "attention.k_proj", [kv, h]),
                        ("v_proj", "attention.v_proj", [kv, h]),
                        ("o_proj", "attention.o_proj", [h, q]),
                    ] {
                        ops +=
                            self.operation(&format!("{mixer}.{name}.weight"), role, &dims, true)?;
                    }
                }
                LayerKind::Moe => {
                    for (name, role, dims, quant) in [
                        (
                            "gate.weight",
                            "moe.router",
                            vec![c.n_routed_experts, h],
                            false,
                        ),
                        (
                            "gate.e_score_correction_bias",
                            "moe.router-bias",
                            vec![c.n_routed_experts],
                            false,
                        ),
                        (
                            "fc1_latent_proj.weight",
                            "moe.fc1_latent_proj",
                            vec![c.moe_latent_size, h],
                            false,
                        ),
                        (
                            "fc2_latent_proj.weight",
                            "moe.fc2_latent_proj",
                            vec![h, c.moe_latent_size],
                            false,
                        ),
                        (
                            "shared_experts.up_proj.weight",
                            "moe.shared.up_proj",
                            vec![c.moe_shared_expert_intermediate_size, h],
                            true,
                        ),
                        (
                            "shared_experts.down_proj.weight",
                            "moe.shared.down_proj",
                            vec![h, c.moe_shared_expert_intermediate_size],
                            true,
                        ),
                    ] {
                        ops += self.operation(&format!("{mixer}.{name}"), role, &dims, quant)?;
                    }
                    for expert in 0..c.n_routed_experts {
                        let mut projections = self.operation(
                            &format!("{mixer}.experts.{expert}.up_proj.weight"),
                            "moe.up_proj",
                            &[c.moe_intermediate_size, c.moe_latent_size],
                            true,
                        )?;
                        projections += self.operation(
                            &format!("{mixer}.experts.{expert}.down_proj.weight"),
                            "moe.down_proj",
                            &[c.moe_latent_size, c.moe_intermediate_size],
                            true,
                        )?;
                        if !self.dry_run {
                            ops += entity! { attrs::kind: EXPERT, attrs::index: expert as u64, attrs::member*: projections };
                        }
                    }
                }
            }
            if !self.dry_run {
                members +=
                    entity! { attrs::kind: LAYER, attrs::index: index as u64, attrs::member*: ops };
            }
        }
        Ok(members)
    }
}

/// Import all selected text components, never a successful partial checkpoint.
/// The caller owns collection selection, durability/flush, and model activation.
/// `source_name` is caller-asserted provenance (e.g. SOURCE@REVISION), not proof
/// that local tensor bytes came from that revision. It is outside identity.
pub fn import_checkpoint_into(
    pile: &mut Pile,
    signer: &SigningKey,
    collection: ModelCollection,
    directory: &Path,
    source_name: &str,
) -> Result<ImportedCheckpoint> {
    model_collection::require_model_writer(pile, signer, collection)?;
    let value = json(directory, "config.json")?;
    let config: CheckpointConfig = serde_json::from_value(value.clone())?;
    ensure!(
        config.llm_config.model_type == "nemotron_h",
        "unsupported text model_type"
    );
    let quant: HfQuantConfig = serde_json::from_value(json(directory, "hf_quant_config.json")?)?;
    let index: Index = serde_json::from_value(json(directory, "model.safetensors.index.json")?)?;
    for (name, file) in &index.weight_map {
        if !selected(name) {
            continue;
        }
        ensure!(
            Path::new(file)
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
                && Path::new(file)
                    .extension()
                    .is_some_and(|s| s == "safetensors"),
            "unsafe shard path {file}"
        );
    }
    let mut importer = Importer {
        pile,
        shards: Shards {
            directory,
            index: &index,
            headers: BTreeMap::new(),
            current: None,
        },
        config: &config,
        quant: &quant,
        consumed: BTreeSet::new(),
        dry_run: true,
        operations: 0,
    };
    importer.graph()?;
    for name in index.weight_map.keys().filter(|n| selected(n)) {
        ensure!(
            importer.consumed.contains(name),
            "unsupported selected text tensor {name}"
        );
    }
    for module in config
        .quantization_config
        .quantized_layers
        .keys()
        .chain(quant.quantization.quantized_layers.keys())
        .filter(|n| super::config::is_text_module(n))
    {
        ensure!(
            importer.consumed.contains(&format!("{module}.weight")),
            "unsupported selected quantization module {module}"
        );
    }
    let tokenizer_bytes = std::fs::read(directory.join("tokenizer.json"))?;
    super::tokenizer_binding::validate(&tokenizer_bytes)?;
    // Keep token sources separate: generation EOS/pad is not tokenizer EOS/pad.
    let generation = json(directory, "generation_config.json")?;
    let tokenizer_config = json(directory, "tokenizer_config.json")?;
    let _: super::config::GenerationConfig = serde_json::from_value(generation.clone())?;
    let _: super::config::TokenizerConfig = serde_json::from_value(tokenizer_config.clone())?;
    let tokenizer_value: Value = serde_json::from_slice(&tokenizer_bytes)?;
    let mut token_config = serde_json::Map::new();
    for field in [
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "padding_side",
        "model_max_length",
    ] {
        token_config.insert(field.to_owned(), tokenizer_config[field].clone());
    }
    for field in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        let spelling = tokenizer_config[field]
            .as_str()
            .with_context(|| format!("missing tokenizer {field}"))?;
        let id = tokenizer_value["model"]["vocab"][spelling]
            .as_u64()
            .with_context(|| format!("tokenizer spelling not in vocabulary: {spelling}"))?;
        token_config.insert(format!("{field}_id"), Value::from(id));
    }
    let mut text = value["llm_config"].clone();
    if let Some(text) = text.as_object_mut() {
        for key in ["mtp_layers_block_type", "num_nextn_predict_layers"] {
            text.remove(key);
        }
    }
    let native_config = json!({"text": text, "generation": generation,
        "tokenizer": token_config,
        "kv_cache_scheme": value["quantization_config"]["kv_cache_scheme"],
        "kv_cache_quant_algo": value["quantization_config"]["kv_cache_quant_algo"],
        "tokenizer_post_processor": tokenizer_value["post_processor"],
        "tokenizer_post_processor_rebuilt": false});
    importer.dry_run = false;
    importer.operations = 0;
    let mut members = importer.graph()?;
    let operations = importer.operations;
    drop(importer);
    let mut config_facts = Fragment::empty();
    let config_root = jsonfacts::save_json(&native_config, pile, config_facts.facts_mut())
        .map_err(|e| anyhow::anyhow!("native config: {e}"))?;
    let mut config_member = entity! { attrs::kind: CONFIG, attrs::member: config_root };
    config_member += config_facts;
    members += config_member;
    let tokenizer = super::tokenizer_binding::save(&tokenizer_bytes, source_name, pile)?;
    let tokenizer_id = tokenizer.root().expect("tokenizer root");
    // A graph member binds tokenizer semantics into model identity; the direct
    // tokenizer edge is a convenience annotation for existing Mary consumers.
    members += entity! { attrs::kind: "tokenizer", attrs::member*: tokenizer };
    let mut fragment = entity! { attrs::member*: members };
    let root = fragment.root().expect("model root");
    fragment += entity! { ExclusiveId::force_ref(&root) @ attrs::kind: "nemotron-h", attrs::source: source_name, tokenizer::attrs::tokenizer: tokenizer_id };
    let commit = model_collection::publish_model_fragment_into(pile, signer, collection, fragment)?;
    Ok(ImportedCheckpoint {
        root,
        tokenizer: tokenizer_id,
        commit,
        operations,
    })
}
