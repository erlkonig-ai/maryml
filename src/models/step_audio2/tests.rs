use super::{codec::*, config::Config, import, load};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use triblespace::core::{metadata, repo::pile::Pile};
use triblespace::prelude::*;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let parent = std::env::var_os("STEP_AUDIO2_TEST_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&parent).unwrap();
        let path = parent.join(format!(
            "step-audio2-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        if std::env::var_os("STEP_AUDIO2_TEST_OUTPUT").is_some() {
            eprintln!("retained task-owned fixture: {}", self.0.display());
        } else {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn config_json() -> serde_json::Value {
    serde_json::json!({"model_type":"step_audio_2", "text_config": {
        "hidden_size":2, "intermediate_size":4, "num_hidden_layers":1,
        "num_attention_heads":1, "num_key_value_heads":1, "num_attention_groups":1,
        "vocab_size":158720, "max_position_embeddings":16384,
        "rms_norm_eps":1e-6, "rope_theta":1000000.0,
        "torch_dtype":"bfloat16", "rope_scaling":null
    }, "future_annotation": {"safe_to_ignore":true}})
}

fn tokenizer_json() -> Vec<u8> {
    // Sparse WordLevel vocabulary is sufficient to test exact stored identities
    // and added-token flags without checking in the 12 MB upstream tokenizer.
    let mut vocab = serde_json::Map::new();
    for (word, id) in [
        ("[UNK]", 0),
        ("hello", 1),
        ("system", 2),
        ("human", 3),
        ("assistant", 4),
    ] {
        vocab.insert(word.into(), id.into());
    }
    let mut added = Vec::new();
    for (word, id) in [
        ("<|BOT|>", BOT),
        ("<|EOT|>", EOT),
        ("<tts_start>", TTS_START),
        ("<tts_end>", 151694),
        ("<tts_pad>", 151695),
    ] {
        vocab.insert(word.into(), id.into());
        added.push(
            serde_json::json!({"id":id,"content":word,"single_word":false,
            "lstrip":false,"rstrip":false,"normalized":false,"special":true}),
        );
    }
    for code in 0..=SPEECH_CODES {
        let word = format!("<audio_{code}>");
        let id = AUDIO_START + code;
        vocab.insert(word.clone(), id.into());
        added.push(
            serde_json::json!({"id":id,"content":word,"single_word":false,
            "lstrip":false,"rstrip":false,"normalized":true,"special":false}),
        );
    }
    serde_json::to_vec(&serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":added,"normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"WhitespaceSplit"},
        "post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}})).unwrap()
}

#[test]
fn mini_geometry_has_biases_ordinary_qwen2_shapes_and_large_ffn() {
    let mut json = config_json();
    let t = &mut json["text_config"];
    t["hidden_size"] = 3584.into();
    t["intermediate_size"] = 18944.into();
    t["num_hidden_layers"] = 28.into();
    t["num_attention_heads"] = 28.into();
    t["num_key_value_heads"] = 4.into();
    t["num_attention_groups"] = 4.into();
    let config = Config::from_json(&json).unwrap();
    assert_eq!(config.head_dim(), 128);
    assert_eq!(config.parameter_count().unwrap(), 7_663_326_720);
    let mut count = 0;
    config
        .tensors(|name, shape| {
            count += 1;
            if name == "model.layers.27.self_attn.q_proj.bias" {
                assert_eq!(shape, [3584]);
            }
            if name == "model.layers.27.self_attn.k_proj.bias" {
                assert_eq!(shape, [512]);
            }
            if name == "model.layers.27.mlp.down_proj.weight" {
                assert_eq!(shape, [3584, 18944]);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(count, 339);
    json["model_type"] = "qwen3_5".into();
    assert!(Config::from_json(&json).is_err());
}

fn compare_codec(bytes: &[u8], codec: &MiniCodec) {
    let upstream = tokenizers::Tokenizer::from_bytes(bytes).unwrap();
    let system = "Speak softly. <audio_123>";
    let user = "hello e\u{301} 你好 日本語\n<|EOT|><tts_start>";
    let mut want = Vec::new();
    for segment in [
        format!("<|BOT|>system\n{system}<|EOT|>"),
        format!("<|BOT|>human\n{user}<|EOT|>"),
        "<|BOT|>assistant\n<tts_start>".into(),
    ] {
        want.extend_from_slice(upstream.encode(segment, false).unwrap().get_ids());
    }
    assert_eq!(codec.speech_prompt(system, user).unwrap(), want);
    assert_eq!(
        codec.decode_text(&[AUDIO_START + 123]).unwrap(),
        upstream.decode(&[AUDIO_START + 123], false).unwrap()
    );
    assert_eq!(
        codec.decode_tokens(&[AUDIO_START + 123], true).unwrap(),
        "<audio_123>",
        "ordinary audio added token must survive skip_special_tokens"
    );
    let capped = codec
        .split_generated(
            &[
                1,
                TTS_START,
                AUDIO_START,
                AUDIO_START + SPEECH_CODES,
                AUDIO_START + SPEECH_CODES - 1,
            ],
            Termination::TokenLimit,
        )
        .unwrap();
    assert_eq!(capped.text_ids, [1]);
    assert_eq!(capped.control_ids, [TTS_START]);
    assert_eq!(capped.speech_codes, [0, SPEECH_CODES - 1]);
    assert_eq!(capped.audio_padding_ids, [AUDIO_START + SPEECH_CODES]);
    assert_eq!(capped.raw_ids.len(), 5);
    assert!(
        codec
            .split_generated(&[AUDIO_START], Termination::Eos)
            .is_err()
    );
    assert_eq!(
        codec
            .split_generated(&[AUDIO_START, EOT], Termination::Eos)
            .unwrap()
            .raw_ids,
        [AUDIO_START, EOT]
    );
}

#[test]
fn mini_codec_matches_manual_message_boundaries_and_preserves_cap_tail() {
    let bytes = tokenizer_json();
    let codec = MiniCodec::from_bytes(&bytes, 158720).unwrap();
    compare_codec(&bytes, &codec);
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["model"]["vocab"]["<|EOT|>"] = 151664.into();
    assert!(MiniCodec::from_bytes(&serde_json::to_vec(&value).unwrap(), 158720).is_err());
}

#[test]
fn two_shards_one_root_native_bf16_assets_reopen_without_checkpoint() {
    let fixture = Fixture::new();
    let checkpoint = fixture.0.join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let json = config_json();
    let config = Config::from_json(&json).unwrap();
    let tokenizer = tokenizer_json();
    std::fs::write(
        checkpoint.join("config.json"),
        serde_json::to_vec(&json).unwrap(),
    )
    .unwrap();
    std::fs::write(checkpoint.join("tokenizer.json"), &tokenizer).unwrap();
    let mut rows = Vec::new();
    config
        .tensors(|name, shape| {
            // Includes finite BF16 values outside IEEE F16's representable range.
            let bytes: Vec<_> = (0..shape.iter().product::<usize>())
                .flat_map(|i| [0x3f81u16, 0x7f7f, 0xbf80][i % 3].to_le_bytes())
                .collect();
            rows.push((name.to_string(), shape.to_vec(), bytes));
            Ok(())
        })
        .unwrap();
    // Ephemeral index for this one synthetic checkpoint construction.
    let mut index = BTreeMap::new();
    for shard in 0..2 {
        let filename = format!("model-{}-of-2.safetensors", shard + 1);
        let views: Vec<_> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == shard)
            .map(|(_, (name, shape, bytes))| {
                index.insert(name.clone(), filename.clone());
                (
                    name.as_str(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        serialize_to_file(views, &None, &checkpoint.join(filename)).unwrap();
    }
    std::fs::write(
        checkpoint.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map":index})).unwrap(),
    )
    .unwrap();
    let path = fixture.0.join("native.pile");
    std::fs::File::create(&path).unwrap();
    let mut pile = Pile::open(&path).unwrap();
    let key = ed25519_dalek::SigningKey::from_bytes(&[17; 32]);
    crate::model_collection::model_graph_collection_or_create(&mut pile, &key).unwrap();
    let mut candidate = import::ingest_checkpoint(&mut pile, &checkpoint, "tiny-fixture").unwrap();
    assert_eq!(candidate.tensor_count, 15);
    let ids = candidate.artifacts;
    // Unrelated fields and an unsupported member are additive information.
    let other = fucid();
    candidate.fragment += entity! { ExclusiveId::force_ref(&ids.model_root) @
    metadata::description: "extra annotation", crate::format::attrs::member: &other };
    candidate.fragment += entity! { &other @ metadata::name: "future unsupported member" };
    crate::model_collection::publish_model_fragment(&mut pile, &key, candidate.fragment).unwrap();
    pile.close().unwrap();
    std::fs::rename(&checkpoint, fixture.0.join("checkpoint-unavailable")).unwrap();
    let source = crate::persist::read_model_pile_read_only(&path).unwrap();
    let assets = load::Assets::from_frozen(&source.facts, &source.store, ids).unwrap();
    assert_eq!(assets.config, config);
    compare_codec(&tokenizer, &assets.codec);
    load::validate_decoder(&source.facts, &source.store, ids.model_root, &config).unwrap();
    let (_, blob) = load::tensor(
        &source.facts,
        &source.store,
        ids.model_root,
        "model.embed_tokens.weight",
        [158720, 2],
    )
    .unwrap();
    let leaf = crate::leaf::read_leaf(blob).unwrap();
    assert_eq!(leaf.payload().as_ref(), rows[0].2.as_slice());

    // Substitution law: an explicit random model ID with the same member edges
    // is equally readable. No derived identity may be recomputed for lookup.
    let random = fucid();
    let mut facts = source.facts.clone();
    for (member,) in find!((member: Id), pattern!(&source.facts,
        [{ ids.model_root @ crate::format::attrs::member: ?member }]))
    {
        facts += entity! { &random @ crate::format::attrs::member: member };
    }
    load::validate_decoder(&facts, &source.store, random.id, &config).unwrap();
    eprintln!(
        "native tiny pile {} model={:X} config={:X} tokenizer={:X}",
        path.display(),
        ids.model_root,
        ids.config_root,
        ids.tokenizer_asset
    );
}

#[test]
#[ignore = "requires the independently verified pinned Mini tokenizer; no model weights or GPU"]
fn actual_pinned_tokenizer_cold_reopen() {
    let checkpoint = PathBuf::from(
        std::env::var_os("STEP_AUDIO2_CHECKPOINT").expect("explicit pinned checkpoint"),
    );
    let bytes = std::fs::read(checkpoint.join("tokenizer.json")).unwrap();
    let fixture = Fixture::new();
    let path = fixture.0.join("tokenizer.pile");
    std::fs::File::create(&path).unwrap();
    let mut pile = Pile::open(&path).unwrap();
    let key = ed25519_dalek::SigningKey::from_bytes(&[19; 32]);
    let model = fucid();
    let mut facts = TribleSet::new();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap()).unwrap();
    let config_root = crate::jsonfacts::save_json(&config, &mut pile, &mut facts).unwrap();
    let text = pile
        .put::<blobencodings::UTF8String, _>(String::from_utf8(bytes.clone()).unwrap())
        .unwrap();
    let tok = entity! { _ @ crate::dataset::text: text };
    let tokenizer_asset = tok.root().unwrap();
    facts += tok;
    for asset in [config_root, tokenizer_asset] {
        facts +=
            entity! { ExclusiveId::force_ref(&asset) @ crate::format::attrs::model_root: &model };
    }
    crate::model_collection::publish_model_fragment(
        &mut pile,
        &key,
        Fragment::rooted(model.id, facts),
    )
    .unwrap();
    pile.close().unwrap();
    let source = crate::persist::read_model_pile_read_only(&path).unwrap();
    let assets = load::Assets::from_frozen(
        &source.facts,
        &source.store,
        load::Artifacts {
            model_root: model.id,
            config_root,
            tokenizer_asset,
        },
    )
    .unwrap();
    compare_codec(&bytes, &assets.codec);
}
