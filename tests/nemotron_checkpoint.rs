//! Synthetic CPU ingress gate. No GPU or full checkpoint payload acceptance.
#![allow(dead_code)]
#![cfg(all(feature = "import", feature = "tokenizer"))]
use mary::models::nemotron_h::ingress::{checkpoint, tokenizer_binding};
use mary::models::nemotron_h::pile as native_pile;
use mary::{format, jsonfacts, leaf, model_collection, tokenizer};
use anyhow::Result;
use ed25519_dalek::SigningKey;
use format::attrs;
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize_to_file},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};
use triblespace::core::blob::encodings::tensor::elements::{BF16, F32};
use triblespace::{
    core::collection::{AdmissionPolicy, CollectionPolicy, CollectionStoreExt},
    prelude::*,
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("nemotron-checkpoint-{}", genid()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn write_json(path: &std::path::Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn token_json() -> Value {
    let mut value = json!({"version":"1.0", "truncation":null,"padding":null,"normalizer":null,
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[
            {"type":"Split","pattern":{"Regex":"\\S+|\\s+"},"behavior":"Isolated","invert":false},
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
        "post_processor":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
        "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        "model":{"type":"BPE","dropout":null,"unk_token":"[UNK]","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,
            "vocab":{"[BOS]":0,"[UNK]":1,"[EOS]":2,"a":3,"b":4,"Ġ":5,"Ġa":6,"aa":7,"<think>":8,"</think>":9,"<tool_call>":10,"<|im_end|>":11,"</tool_call>":12,"<tool_response>":13,"</tool_response>":14,"z":15},"merges":[["Ġ","a"],["a","a"]]},
        "added_tokens":[]});
    for (text, id, special) in [
        ("[BOS]", 0, true),
        ("[UNK]", 1, true),
        ("[EOS]", 2, true),
        ("<|im_end|>", 11, true),
        ("<think>", 8, false),
        ("</think>", 9, false),
        ("<tool_call>", 10, false),
        ("</tool_call>", 12, false),
        ("<tool_response>", 13, false),
        ("</tool_response>", 14, false),
    ] {
        value["added_tokens"].as_array_mut().unwrap().push(json!({"id":id,"content":text,"special":special,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false}));
    }
    value
}
fn text_config() -> Value {
    serde_json::from_str(r#"{"model_type":"nemotron_h","layers_block_type":["mamba","moe","attention"],
        "hidden_size":16,"vocab_size":16,"max_position_embeddings":128,"layer_norm_epsilon":0.00001,
        "residual_in_fp32":true,"rescale_prenorm_residual":false,"tie_word_embeddings":false,
        "bos_token_id":0,"eos_token_id":11,"pad_token_id":11,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":16,"attention_bias":false,"sliding_window":null,
        "conv_kernel":4,"expand":1,"mamba_head_dim":16,"mamba_num_heads":1,"mamba_hidden_act":"silu","mamba_proj_bias":false,"mamba_ssm_cache_dtype":"float32","use_bias":false,"use_conv_bias":true,"ssm_state_size":2,"n_groups":1,"chunk_size":4,
        "time_step_limit":[0,{"__float__":"Infinity"}],"time_step_min":0.001,"time_step_max":0.1,"time_step_floor":0.0001,
        "intermediate_size":16,"mlp_hidden_act":"relu2","mlp_bias":false,"moe_intermediate_size":16,"moe_latent_size":16,"moe_shared_expert_intermediate_size":16,"moe_shared_expert_overlap":true,"n_routed_experts":2,"n_shared_experts":1,"num_experts_per_tok":1,"norm_topk_prob":true,"routed_scaling_factor":1.0,"n_group":1,"topk_group":1}"#).unwrap()
}
struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
fn fixture(dir: &std::path::Path) {
    let mut tensors = BTreeMap::<String, Tensor>::new();
    let mut quant = serde_json::Map::new();
    let mut add = |name: String, shape: Vec<usize>, mode: &str| {
        let count: usize = shape.iter().product();
        let (dtype, stored_shape, bytes) = match mode {
            "FP8" => (Dtype::F8_E4M3, shape.clone(), vec![0x38; count]),
            "NVFP4" => (
                Dtype::U8,
                vec![shape[0], shape[1] / 2],
                vec![0x21; count / 2],
            ),
            "F32" => (
                Dtype::F32,
                shape.clone(),
                [0x01, 0x00, 0xc0, 0x7f].repeat(count),
            ),
            _ => (Dtype::BF16, shape.clone(), [0x00, 0x80].repeat(count)),
        };
        tensors.insert(
            name.clone(),
            Tensor {
                dtype,
                shape: stored_shape,
                bytes,
            },
        );
        if mode == "FP8" || mode == "NVFP4" {
            let module = name.strip_suffix(".weight").unwrap();
            quant.insert(
                module.to_owned(),
                if mode == "FP8" {
                    json!({"quant_algo":"FP8"})
                } else {
                    json!({"quant_algo":"NVFP4","group_size":16})
                },
            );
            let (dtype, dims, bytes) = if mode == "FP8" {
                (Dtype::F32, vec![], 0.5f32.to_le_bytes().to_vec())
            } else {
                (
                    Dtype::F8_E4M3,
                    vec![shape[0], shape[1] / 16],
                    vec![0x38; count / 16],
                )
            };
            tensors.insert(
                format!("{module}.weight_scale"),
                Tensor {
                    dtype,
                    shape: dims,
                    bytes,
                },
            );
            tensors.insert(
                format!("{module}.input_scale"),
                Tensor {
                    dtype: Dtype::F32,
                    shape: vec![1],
                    bytes: 2f32.to_le_bytes().to_vec(),
                },
            );
            if mode == "NVFP4" {
                tensors.insert(
                    format!("{module}.weight_scale_2"),
                    Tensor {
                        dtype: Dtype::F32,
                        shape: vec![],
                        bytes: 0.25f32.to_le_bytes().to_vec(),
                    },
                );
            }
        }
    };
    for (name, shape, mode) in [
        ("backbone.embeddings.weight", vec![16, 16], "BF16"),
        ("backbone.norm_f.weight", vec![16], "BF16"),
        ("lm_head.weight", vec![16, 16], "FP8"),
    ] {
        add(format!("language_model.{name}"), shape, mode);
    }
    for layer in 0..3 {
        add(
            format!("language_model.backbone.layers.{layer}.norm.weight"),
            vec![16],
            "BF16",
        );
    }
    for (name, shape, mode) in [
        ("in_proj.weight", vec![37, 16], "FP8"),
        ("out_proj.weight", vec![16, 16], "FP8"),
        ("conv1d.weight", vec![20, 1, 4], "BF16"),
        ("conv1d.bias", vec![20], "BF16"),
        ("norm.weight", vec![16], "BF16"),
        ("A_log", vec![1], "F32"),
        ("D", vec![1], "BF16"),
        ("dt_bias", vec![1], "F32"),
    ] {
        add(
            format!("language_model.backbone.layers.0.mixer.{name}"),
            shape,
            mode,
        );
    }
    for (name, shape) in [
        ("gate.weight", vec![2, 16]),
        ("gate.e_score_correction_bias", vec![2]),
        ("fc1_latent_proj.weight", vec![16, 16]),
        ("fc2_latent_proj.weight", vec![16, 16]),
    ] {
        add(
            format!("language_model.backbone.layers.1.mixer.{name}"),
            shape,
            "BF16",
        );
    }
    for expert in ["shared_experts", "experts.0", "experts.1"] {
        for proj in ["up_proj", "down_proj"] {
            add(
                format!("language_model.backbone.layers.1.mixer.{expert}.{proj}.weight"),
                vec![16, 16],
                "NVFP4",
            );
        }
    }
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        add(
            format!("language_model.backbone.layers.2.mixer.{proj}.weight"),
            vec![16, 16],
            "FP8",
        );
    }
    let mut weight_map = serde_json::Map::new();
    for (shard, scales) in [("weights.safetensors", false), ("scales.safetensors", true)] {
        let views = tensors
            .iter()
            .filter(|(name, _)| (name.ends_with("scale") || name.ends_with("scale_2")) == scales)
            .map(|(name, t)| {
                weight_map.insert(name.clone(), Value::from(shard));
                (
                    name.clone(),
                    TensorView::new(t.dtype, t.shape.clone(), &t.bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, &None, &dir.join(shard)).unwrap();
    }
    // Deliberately absent excluded shards must never be opened.
    weight_map.insert(
        "vision_encoder.weight".into(),
        json!("missing-vision.safetensors"),
    );
    weight_map.insert(
        "language_model.mtp.layers.0.weight".into(),
        json!("missing-mtp.safetensors"),
    );
    let q = json!({"quant_algo":"MIXED_PRECISION","quantized_layers":quant,"kv_cache_scheme":{"dynamic":false,"num_bits":8,"type":"float"}});
    write_json(
        &dir.join("config.json"),
        &json!({"model_type":"nemotron_h","llm_config":text_config(),"quantization_config":q}),
    );
    let mut hf = json!({"quantization":q});
    hf["quantization"]["quantized_layers"]["language_model.mtp.layers.0.q_proj"] =
        json!({"quant_algo":"FP8"});
    write_json(&dir.join("hf_quant_config.json"), &hf);
    write_json(
        &dir.join("model.safetensors.index.json"),
        &json!({"weight_map":weight_map}),
    );
    write_json(&dir.join("tokenizer.json"), &token_json());
    write_json(
        &dir.join("tokenizer_config.json"),
        &json!({"bos_token":"[BOS]","eos_token":"<|im_end|>","pad_token":"<|im_end|>","unk_token":"[UNK]","padding_side":"left","model_max_length":128}),
    );
    write_json(
        &dir.join("generation_config.json"),
        &json!({"bos_token_id":0,"eos_token_id":[2,11],"pad_token_id":0,"do_sample":true,"temperature":0.8,"top_p":0.9}),
    );
}
fn destination(
    dir: &std::path::Path,
) -> Result<(Pile, SigningKey, model_collection::ModelCollection)> {
    let file = dir.join("model.pile");
    std::fs::File::create(&file)?;
    let mut pile = Pile::open(&file)?;
    let signer = SigningKey::generate(&mut rand::rngs::OsRng);
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(signer.verifying_key()),
        AdmissionPolicy::direct(signer.verifying_key()),
    );
    let collection = pile.collection("checkpoint-fixture", policy)?;
    Ok((pile, signer, collection))
}

#[test]
fn checkpoint_publish_reopen_native_weights_config_and_tokenizer() -> Result<()> {
    let dir = Scratch::new();
    fixture(&dir.0);
    let (mut pile, key, collection) = destination(&dir.0)?;
    let imported = checkpoint::import_checkpoint_into(
        &mut pile,
        &key,
        collection,
        &dir.0,
        "synthetic-fixture",
    )?;
    assert_eq!(imported.operations, 28);
    let relabeled = checkpoint::import_checkpoint_into(
        &mut pile,
        &key,
        collection,
        &dir.0,
        "another-provenance-label",
    )?;
    assert_eq!(imported.root, relabeled.root, "source is not identity");
    pile.flush()?;
    drop(pile);
    let mut pile = Pile::open(&dir.0.join("model.pile"))?;
    pile.refresh()?;
    let store = pile.snapshot()?;
    let snapshot = model_collection::snapshot_model_collection_for(&store, collection)?;
    let layer = native_pile::layers(&snapshot, imported.root, 0)
        .next()
        .unwrap();
    let mamba = native_pile::fp8(&snapshot, layer, "mamba.in_proj")
        .next()
        .unwrap()?;
    assert_eq!(mamba.weight.dims(), [37, 16]);
    assert_eq!(&mamba.weight.payload()[..], vec![0x38; 37 * 16]);
    let a = native_pile::auxiliary::<F32, 1, _>(&snapshot, layer, "mamba.A_log")
        .next()
        .unwrap()?;
    assert_eq!(&a.payload()[..], [0x01, 0x00, 0xc0, 0x7f]);
    let conv = native_pile::auxiliary::<BF16, 3, _>(&snapshot, layer, "mamba.conv1d.weight")
        .next()
        .unwrap()?;
    assert_eq!(conv.dims(), [20, 1, 4]);
    assert_eq!(&conv.payload()[..], [0x00, 0x80].repeat(80));
    let moe = native_pile::layers(&snapshot, imported.root, 1)
        .next()
        .unwrap();
    let expert = native_pile::experts(&snapshot, moe, 1).next().unwrap();
    let projection = native_pile::nvfp4(&snapshot, expert, "moe.down_proj")
        .next()
        .unwrap()?;
    let mut expected = vec![0x21; 128];
    expected.extend([0x38; 16]);
    expected.extend(0.25f32.to_le_bytes());
    assert_eq!(&projection.weight.payload()[..], expected);
    let cfg = find!(node: Id,pattern!(snapshot.facts(),[
        { imported.root @ attrs::member: _?config }, { _?config @ attrs::kind: checkpoint::CONFIG,attrs::member: ?node }
    ])).next().unwrap();
    let cfg = jsonfacts::load_json(snapshot.facts(), snapshot.store(), cfg)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    assert_eq!(cfg["tokenizer"]["eos_token_id"], 11);
    assert_eq!(cfg["tokenizer"]["pad_token_id"], 11);
    assert_eq!(cfg["generation"]["pad_token_id"], 0);
    assert_eq!(cfg["generation"]["eos_token_id"], json!([2, 11]));
    assert_eq!(cfg["kv_cache_scheme"]["num_bits"], 8);
    assert_eq!(cfg["tokenizer_post_processor_rebuilt"], false);
    let base = tokenizer_binding::bases(snapshot.facts(), imported.tokenizer)
        .next()
        .unwrap();
    let sequence = tokenizer_binding::pre_tokenizers(snapshot.facts(), imported.tokenizer)
        .next()
        .unwrap();
    let rebuilt = tokenizer_binding::rebuild(
        snapshot.facts(),
        snapshot.store(),
        imported.tokenizer,
        base,
        sequence,
    )?;
    assert_tokenizer_parity(&rebuilt, &token_json())?;
    Ok(())
}
fn assert_tokenizer_parity(rebuilt: &tokenizers::Tokenizer, source: &Value) -> Result<()> {
    let original = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(source)?)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for text in [
        "a aa b",
        " aa",
        "<think>a</think>",
        "<tool_call>b</tool_call>",
        "<tool_response>a</tool_response>",
        "[BOS]<|im_end|>",
    ] {
        for special in [false, true] {
            let a = original
                .encode(text, special)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let b = rebuilt
                .encode(text, special)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            assert_eq!(a.get_ids(), b.get_ids(), "{text:?}");
            assert_eq!(a.get_offsets(), b.get_offsets(), "{text:?} offsets");
            for skip in [false, true] {
                assert_eq!(
                    original.decode(a.get_ids(), skip).unwrap(),
                    rebuilt.decode(b.get_ids(), skip).unwrap(),
                    "{text:?} skip={skip}"
                );
            }
        }
    }
    assert_eq!(
        rebuilt.decode(&[8, 9, 10, 12, 13, 14, 11], true).unwrap(),
        "<think></think><tool_call></tool_call><tool_response></tool_response>"
    );
    assert!(rebuilt.get_post_processor().is_none());
    assert!(original.get_post_processor().is_some());
    let a = original.encode((" aa", "b"), true).unwrap();
    let b = rebuilt.encode((" aa", "b"), true).unwrap();
    assert_eq!(a.get_ids(), b.get_ids());
    assert_eq!(a.get_offsets(), b.get_offsets());
    assert_eq!(a.get_type_ids(), b.get_type_ids());
    assert_eq!(a.get_sequence_ids(), b.get_sequence_ids());
    Ok(())
}

#[test]
fn missing_required_component_or_selector_never_publishes_root() -> Result<()> {
    for selector in [false, true] {
        let dir = Scratch::new();
        fixture(&dir.0);
        if selector {
            for filename in ["config.json", "hf_quant_config.json"] {
                let mut v: Value = serde_json::from_slice(&std::fs::read(dir.0.join(filename))?)?;
                let q = if filename == "config.json" {
                    "quantization_config"
                } else {
                    "quantization"
                };
                v[q]["quantized_layers"]
                    .as_object_mut()
                    .unwrap()
                    .remove("language_model.lm_head");
                write_json(&dir.0.join(filename), &v);
            }
        } else {
            let path = dir.0.join("model.safetensors.index.json");
            let mut v: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            v["weight_map"].as_object_mut().unwrap().remove(
                "language_model.backbone.layers.1.mixer.experts.1.down_proj.weight_scale_2",
            );
            write_json(&path, &v);
        }
        let (mut pile, key, collection) = destination(&dir.0)?;
        let error = checkpoint::import_checkpoint_into(
            &mut pile,
            &key,
            collection,
            &dir.0,
            "synthetic-fixture",
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("incomplete import"), "{error:#}");
        pile.flush()?;
        drop(pile);
        let mut pile = Pile::open(&dir.0.join("model.pile"))?;
        pile.refresh()?;
        let store = pile.snapshot()?;
        let snapshot = model_collection::snapshot_model_collection_for(&store, collection)?;
        assert!(!exists!(
            pattern!(snapshot.facts(),[{ _?root @ attrs::kind:"nemotron-h" }])
        ));
    }
    Ok(())
}

#[test]
fn added_flags_are_local_to_each_binding_after_union() -> Result<()> {
    let dir = Scratch::new();
    let (mut pile, key, collection) = destination(&dir.0)?;
    let ordinary = token_json();
    let mut special = ordinary.clone();
    special["added_tokens"][4]["special"] = json!(true);
    special["added_tokens"][4]["normalized"] = json!(true);
    let a = tokenizer_binding::save(&serde_json::to_vec(&ordinary)?, "fixture", &mut pile)?;
    let renamed = tokenizer_binding::save(
        &serde_json::to_vec(&ordinary)?,
        "another-source-label",
        &mut pile,
    )?;
    assert_eq!(a.root(), renamed.root(), "provenance is not identity");
    let b = tokenizer_binding::save(&serde_json::to_vec(&special)?, "fixture", &mut pile)?;
    let ar = a.root().unwrap();
    let br = b.root().unwrap();
    assert_ne!(ar, br);
    model_collection::publish_model_fragment_into(&mut pile, &key, collection, a + b)?;
    let store = pile.snapshot()?;
    let snapshot = model_collection::snapshot_model_collection_for(&store, collection)?;
    let build = |binding| {
        let base = tokenizer_binding::bases(snapshot.facts(), binding)
            .next()
            .unwrap();
        let sequence = tokenizer_binding::pre_tokenizers(snapshot.facts(), binding)
            .next()
            .unwrap();
        tokenizer_binding::rebuild(snapshot.facts(), snapshot.store(), binding, base, sequence)
    };
    let a = build(ar)?;
    let b = build(br)?;
    assert_eq!(a.decode(&[8], true).unwrap(), "<think>");
    assert_eq!(b.decode(&[8], true).unwrap(), "");
    assert!(!a.get_added_tokens_decoder()[&8].normalized);
    assert!(b.get_added_tokens_decoder()[&8].normalized);
    Ok(())
}

#[test]
fn reversed_pre_tokenizer_order_survives_union_with_generic_collision() -> Result<()> {
    let dir = Scratch::new();
    let (mut pile, key, collection) = destination(&dir.0)?;
    let forward = token_json();
    let mut reverse = forward.clone();
    reverse["pre_tokenizer"]["pretokenizers"]
        .as_array_mut()
        .unwrap()
        .reverse();
    let fbytes = serde_json::to_vec(&forward)?;
    let rbytes = serde_json::to_vec(&reverse)?;
    // Reproduce the old writer's identity collision, including polluted child
    // indexes in the SAME collection. The native writer must not share them.
    let generic_f = tokenizer::save_tokenizer_json(&fbytes, "generic-collision", &mut pile)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let generic_r = tokenizer::save_tokenizer_json(&rbytes, "generic-collision", &mut pile)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let generic_root = generic_f.root().unwrap();
    assert_eq!(generic_f.root(), generic_r.root());
    let a = tokenizer_binding::save(&fbytes, "native", &mut pile)?;
    let b = tokenizer_binding::save(&rbytes, "native", &mut pile)?;
    let ar = a.root().unwrap();
    let br = b.root().unwrap();
    assert_ne!(ar, br, "pre-tokenizer order is identity");
    let alias = ufoid();
    let members = find!(member: Id, pattern!(a.facts(), [{ ar @ attrs::member: ?member }]))
        .collect::<Vec<_>>();
    let alias_facts =
        entity! { &alias @ attrs::kind: tokenizer_binding::BINDING, attrs::member*: members };
    model_collection::publish_model_fragment_into(
        &mut pile,
        &key,
        collection,
        generic_f + generic_r + a + b + alias_facts,
    )?;
    pile.close()?;
    let mut pile = Pile::open(&dir.0.join("model.pile"))?;
    pile.refresh()?;
    let store = pile.snapshot()?;
    let snapshot = model_collection::snapshot_model_collection_for(&store, collection)?;
    assert!(
        tokenizer::build_tokenizer(snapshot.facts(), snapshot.store(), generic_root).is_err(),
        "generic shared-index collision reproduced"
    );
    for (binding, expected) in [(ar, &forward), (br, &reverse), (alias.id, &forward)] {
        let base = tokenizer_binding::bases(snapshot.facts(), binding)
            .next()
            .unwrap();
        let sequence = tokenizer_binding::pre_tokenizers(snapshot.facts(), binding)
            .next()
            .unwrap();
        let rebuilt = tokenizer_binding::rebuild(
            snapshot.facts(),
            snapshot.store(),
            binding,
            base,
            sequence,
        )?;
        assert_eq!(
            serde_json::to_value(rebuilt.get_pre_tokenizer())?,
            expected["pre_tokenizer"]
        );
        assert_tokenizer_parity(&rebuilt, expected)?;
    }
    pile.close()?;
    Ok(())
}

#[test]
fn unsupported_bpe_and_config_knobs_are_rejected_explicitly() -> Result<()> {
    for (field, unsupported) in [
        ("byte_fallback", json!(true)),
        ("fuse_unk", json!(true)),
        ("dropout", json!(0.1)),
        ("continuing_subword_prefix", json!("##")),
        ("end_of_word_suffix", json!("</w>")),
    ] {
        let mut value = token_json();
        value["model"][field] = unsupported;
        let error = tokenizer_binding::validate(&serde_json::to_vec(&value)?).unwrap_err();
        assert!(error.to_string().contains(field), "{error:#}");
    }
    let mut value = token_json();
    value["normalizer"] = json!({"type":"Lowercase"});
    assert!(tokenizer_binding::validate(&serde_json::to_vec(&value)?).is_err());
    let mut value = token_json();
    value["pre_tokenizer"]["pretokenizers"][0]["invert"] = json!(true);
    assert!(tokenizer_binding::validate(&serde_json::to_vec(&value)?).is_err());
    Ok(())
}

#[test]
fn late_malformed_payload_and_unadmitted_writer_do_not_publish() -> Result<()> {
    let dir = Scratch::new();
    fixture(&dir.0);
    let (mut pile, key, collection) = destination(&dir.0)?;
    let other = SigningKey::generate(&mut rand::rngs::OsRng);
    let error = checkpoint::import_checkpoint_into(
        &mut pile,
        &other,
        collection,
        &dir.0,
        "synthetic-fixture",
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("not admitted"));
    // Metadata is complete but a later shard contains invalid bytes. A failed
    // streaming import may leave orphan blobs; no model graph is committed.
    std::fs::write(dir.0.join("scales.safetensors"), b"broken")?;
    assert!(
        checkpoint::import_checkpoint_into(
            &mut pile,
            &key,
            collection,
            &dir.0,
            "synthetic-fixture"
        )
        .is_err()
    );
    pile.flush()?;
    let store = pile.snapshot()?;
    let snapshot = model_collection::snapshot_model_collection_for(&store, collection)?;
    assert!(snapshot.facts().is_empty());
    Ok(())
}

/// Optional metadata-only test against retained public originals at the fixed
/// revision. There are deliberately no weight shards in this directory.
#[test]
#[ignore = "set NEMOTRON_METADATA to the pinned public metadata directory"]
fn pinned_inventory_and_real_tokenizer_rebuild() -> Result<()> {
    let metadata = PathBuf::from(std::env::var("NEMOTRON_METADATA")?);
    let dir = Scratch::new();
    let (mut pile, key, collection) = destination(&dir.0)?;
    let source_label = format!("{}@{}", checkpoint::SOURCE, checkpoint::REVISION);
    let error =
        checkpoint::import_checkpoint_into(&mut pile, &key, collection, &metadata, &source_label)
            .err()
            .unwrap();
    // The complete 88-layer inventory and both text selector maps were checked
    // before the first weight open. This is NOT actual-weight shape evidence.
    assert!(error.to_string().starts_with("open shard "), "{error:#}");
    eprintln!("pinned selected index/config inventory accepted; no weight shard opened");
    let bytes = std::fs::read(metadata.join("tokenizer.json"))?;
    let source: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(source["added_tokens"].as_array().unwrap().len(), 1000);
    for token in source["added_tokens"].as_array().unwrap() {
        for field in ["single_word", "lstrip", "rstrip"] {
            assert_eq!(token[field], false, "pinned uniform default {field}");
        }
    }
    assert_eq!(
        source["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| t["special"] == false)
            .count(),
        6
    );
    let binding = tokenizer_binding::save(&bytes, checkpoint::SOURCE, &mut pile)?;
    eprintln!("pinned tokenizer saved as native graph");
    let root = binding.root().unwrap();
    model_collection::publish_model_fragment_into(&mut pile, &key, collection, binding)?;
    pile.flush()?;
    drop(pile);
    let mut pile = Pile::open(&dir.0.join("model.pile"))?;
    pile.refresh()?;
    let store = pile.snapshot()?;
    let snapshot = model_collection::snapshot_model_collection_for(&store, collection)?;
    let base = tokenizer_binding::bases(snapshot.facts(), root)
        .next()
        .unwrap();
    let sequence = tokenizer_binding::pre_tokenizers(snapshot.facts(), root)
        .next()
        .unwrap();
    let rebuilt =
        tokenizer_binding::rebuild(snapshot.facts(), snapshot.store(), root, base, sequence)?;
    eprintln!("pinned tokenizer rebuilt after pile close/reopen");
    let original = tokenizers::Tokenizer::from_bytes(&bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    let markers = source["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["special"] == false)
        .map(|t| t["content"].as_str().unwrap())
        .collect::<String>();
    for text in [
        "Hello, world! 12345\n",
        " naïve café 日本語 🦊",
        "\t a  b\n",
        "<|im_start|>assistant\n<think>test</think><|im_end|>",
        &markers,
    ] {
        for special in [false, true] {
            let a = original.encode(text, special).unwrap();
            let b = rebuilt.encode(text, special).unwrap();
            assert_eq!(a.get_ids(), b.get_ids(), "IDs for {text:?}");
            assert_eq!(a.get_offsets(), b.get_offsets(), "offsets for {text:?}");
            for skip in [false, true] {
                assert_eq!(
                    original.decode(a.get_ids(), skip).unwrap(),
                    rebuilt.decode(b.get_ids(), skip).unwrap(),
                    "decode {text:?} skip={skip}"
                );
            }
        }
    }
    assert!(original.get_post_processor().is_some());
    assert!(rebuilt.get_post_processor().is_none());
    let a = original
        .encode((" first sequence", "second sequence"), true)
        .unwrap();
    let b = rebuilt
        .encode((" first sequence", "second sequence"), true)
        .unwrap();
    assert_eq!(a.get_ids(), b.get_ids());
    assert_eq!(a.get_offsets(), b.get_offsets());
    assert_eq!(a.get_type_ids(), b.get_type_ids());
    assert_eq!(a.get_sequence_ids(), b.get_sequence_ids());
    assert_eq!(rebuilt.token_to_id("<|im_end|>"), Some(11));
    Ok(())
}
