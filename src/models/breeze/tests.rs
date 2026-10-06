use super::{
    config::{Config, RopeKind},
    import, load,
};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use triblespace::core::{metadata, repo::pile::Pile};
use triblespace::prelude::*;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn fixture() -> PathBuf {
    let parent = std::env::var_os("BREEZE_TEST_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&parent).unwrap();
    let path = parent.join(format!(
        "breeze-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&path).unwrap();
    path
}

fn source(root: &std::path::Path) {
    std::fs::create_dir(root).unwrap();
    std::fs::create_dir(root.join("audio_tokenizer")).unwrap();
    std::fs::write(
        root.join("config.json"),
        include_str!("fixtures/config.json"),
    )
    .unwrap();
    std::fs::write(
        root.join("audio_tokenizer/config.json"),
        include_str!("fixtures/codec_config.json"),
    )
    .unwrap();
    for name in [
        "LICENSE",
        "README.md",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "audio_tokenizer/configuration.json",
        "audio_tokenizer/preprocessor_config.json",
    ] {
        std::fs::write(root.join(name), "{}").unwrap();
    }
    std::fs::write(
        root.join("tokenizer.json"),
        serde_json::to_vec(&serde_json::json!({
            "version":"1.0", "truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}
        }))
        .unwrap(),
    )
    .unwrap();
    let bf16 = [0x80u8, 0x7f, 0x01, 0x00]; // infinity + BF16 subnormal: never through F16
    let scalar = 0x7fc12345u32.to_le_bytes(); // retain even NaN payload bits in F32 scalar
    serialize_to_file(
        [
            (
                "test.bf16",
                TensorView::new(Dtype::BF16, vec![2], &bf16).unwrap(),
            ),
            (
                "test.scalar",
                TensorView::new(Dtype::F32, vec![], &scalar).unwrap(),
            ),
        ],
        &None,
        &root.join("model-1.safetensors"),
    )
    .unwrap();
    let matrix: Vec<u8> = [1.25f32, -2.0, 0.0, 4.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    serialize_to_file(
        [(
            "test.matrix",
            TensorView::new(Dtype::F32, vec![2, 2], &matrix).unwrap(),
        )],
        &None,
        &root.join("model-2.safetensors"),
    )
    .unwrap();
    serialize_to_file(
        [(
            "decoder.weight",
            TensorView::new(Dtype::F32, vec![2, 2], &matrix).unwrap(),
        )],
        &None,
        &root.join("audio_tokenizer/model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "weight_map":{"test.bf16":"model-1.safetensors","test.scalar":"model-1.safetensors",
                          "test.matrix":"model-2.safetensors"}
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
fn nested_backbone_geometry_is_distinct_from_outer_and_text_heads_are_explicit() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/config.json")).unwrap();
    let config = Config::from_json(&value).unwrap();
    assert_eq!(config.backbone.rope.theta, 1_000_000.0);
    assert_eq!(config.backbone.rope.kind, RopeKind::Default);
    assert_eq!(config.backbone.rms_norm_eps, 1e-6);
    assert_eq!(config.depth.decoder.rope.theta, 500_000.0);
    assert_eq!(config.depth.decoder.rope.kind, RopeKind::Llama3);
    assert_eq!(config.text.decoder.head_dim, 256);
    assert_ne!(
        config.text.decoder.head_dim,
        config.text.decoder.hidden_size / config.text.decoder.num_attention_heads
    );
    value["future_annotation"] = serde_json::json!({"ignored":true});
    assert!(Config::from_json(&value).is_ok());
    value["text_encoder_feature_layer_idx"] = serde_json::json!([-2]);
    assert!(Config::from_json(&value).is_err());
    value["text_encoder_feature_layer_idx"] = serde_json::json!([-1]);
    assert!(Config::from_json(&value).is_ok());
    value["backbone_config"]["attention_bias"] = true.into();
    assert!(Config::from_json(&value).is_err());
}

#[test]
fn mixed_dtype_scalar_and_documents_reopen_without_source_and_select_opaque_codec() {
    let dir = fixture();
    let checkpoint = dir.join("checkpoint");
    source(&checkpoint);
    let path = dir.join("model.pile");
    std::fs::File::create(&path).unwrap();
    let mut pile = Pile::open(&path).unwrap();
    let key = ed25519_dalek::SigningKey::from_bytes(&[23u8; 32]); // synthetic fixture signer only
    crate::model_collection::model_graph_collection_or_create(&mut pile, &key).unwrap();
    let mut candidate = import::ingest_checkpoint(&mut pile, &checkpoint, "test-revision").unwrap();
    assert_eq!(candidate.tensor_count, 4);
    assert_eq!(candidate.parameter_count, 11);
    assert_eq!(candidate.payload_bytes, 40);
    let ids = candidate.artifacts;
    // A fresh opaque root is equally usable; no identity reconstruction. Add
    // another model with the same leaf name and unrelated annotations.
    let selected = fucid();
    let members: Vec<_> = find!((member: Id), pattern!(candidate.fragment.facts(), [
        { ids.external_codec_root @ crate::format::attrs::member: ?member }
    ]))
    .map(|(m,)| m)
    .collect(); // ephemeral fixture assembly only
    candidate.fragment += entity! { &selected @ crate::format::attrs::member*: members,
    metadata::name: "arbitrary stored codec identity", metadata::json_kind: "future annotation" };
    let other_data = crate::format::put_raw(&mut pile, &[99.0f32], &[1]).unwrap();
    let other_weight = other_data.root().unwrap();
    candidate.fragment += other_data;
    let name = pile
        .put::<blobencodings::UTF8String, _>("decoder.weight".to_owned())
        .unwrap();
    let other_member = entity! { _ @ crate::format::attrs::safetensor_path: name,
    crate::format::attrs::weight: other_weight };
    let other_member_id = other_member.root().unwrap();
    candidate.fragment += other_member;
    let unrelated = fucid();
    candidate.fragment += entity! { &unrelated @ crate::format::attrs::member: other_member_id };
    crate::model_collection::publish_model_fragment(&mut pile, &key, candidate.fragment).unwrap();
    pile.close().unwrap();
    std::fs::rename(&checkpoint, dir.join("source-unavailable-to-runtime")).unwrap();
    let observed = crate::persist::read_model_pile_read_only(&path).unwrap();
    let assets = load::Assets::from_frozen(&observed.facts, &observed.store, ids).unwrap();
    assert!(assets.tokenizer_json.contains("WordLevel"));
    let (_, bf16) = load::tensor_bf16(
        &observed.facts,
        &observed.store,
        ids.model_root,
        "test.bf16",
        [2],
    )
    .unwrap();
    assert_eq!(
        &crate::leaf::read_leaf(bf16).unwrap().payload()[..],
        &[0x80, 0x7f, 0x01, 0x00]
    );
    let (_, scalar) = load::tensor_f32::<0>(
        &observed.facts,
        &observed.store,
        ids.model_root,
        "test.scalar",
        [],
    )
    .unwrap();
    let scalar = crate::leaf::read_leaf(scalar).unwrap();
    assert!(scalar.dims().is_empty());
    assert_eq!(&scalar.payload()[..], &0x7fc12345u32.to_le_bytes());
    assert!(
        load::tensor_bf16(
            &observed.facts,
            &observed.store,
            ids.model_root,
            "test.matrix",
            [2, 2]
        )
        .is_err()
    );
    let snapshot = crate::model_collection::snapshot_model_collection_in(&observed.store).unwrap();
    let loader = crate::nn::weight_loader::WeightLoader::selected(snapshot, selected.id);
    assert_eq!(
        loader.load_f32("decoder.weight"),
        (vec![1.25, -2.0, 0.0, 4.0], vec![2, 2])
    );
    assert!(loader.view_f32("decoder.weight").is_some());
    assert!(!loader.has_weight("test.bf16"));
    eprintln!("retained cold-read fixture: {}", dir.display());
}

#[test]
fn inconsistent_foreign_index_is_rejected_before_publication() {
    let dir = fixture();
    let checkpoint = dir.join("checkpoint");
    source(&checkpoint);
    let index = serde_json::json!({"weight_map":{"test.bf16":"../escape.safetensors"}});
    std::fs::write(
        checkpoint.join("model.safetensors.index.json"),
        index.to_string(),
    )
    .unwrap();
    let path = dir.join("failed.pile");
    std::fs::File::create(&path).unwrap();
    let mut pile = Pile::open(&path).unwrap();
    assert!(import::ingest_checkpoint(&mut pile, &checkpoint, "test").is_err());
    pile.close().unwrap();
}
