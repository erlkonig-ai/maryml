//! Source-only proposed checks until a separately admitted CPU/GPU invocation.
//! Three layers, small vocabulary/state; full native attention/MoE widths.
//! Native partition agreement checks orchestration, NOT model acceptance.
use super::super::{
    ingress::{native, tokenizer_binding},
    pile,
};
use super::*;
use crate::{format::attrs, jsonfacts, leaf, model_collection, tokenizer};
use anybytes::Bytes;
use ed25519_dalek::SigningKey;
use serde_json::{Value, json};
use triblespace::core::{
    blob::encodings::tensor::{
        TensorElement,
        elements::{BF16, E4M3, F32, NVFP4},
    },
    collection::{AdmissionPolicy, CollectionPolicy, CollectionStoreExt},
};

fn configuration() -> Value {
    serde_json::from_str(r#"{
        "model_type":"nemotron_h","layers_block_type":["mamba","attention","moe"],
        "hidden_size":4096,"vocab_size":32,"max_position_embeddings":8,"layer_norm_epsilon":0.00001,
        "residual_in_fp32":false,"rescale_prenorm_residual":true,"tie_word_embeddings":false,
        "bos_token_id":1,"eos_token_id":11,"pad_token_id":0,"num_attention_heads":32,"num_key_value_heads":2,
        "head_dim":128,"attention_bias":false,"sliding_window":null,"conv_kernel":2,"expand":1,
        "mamba_head_dim":64,"mamba_num_heads":64,"mamba_hidden_act":"silu","mamba_proj_bias":false,
        "mamba_ssm_cache_dtype":"float32","use_bias":false,"use_conv_bias":true,"ssm_state_size":4,
        "n_groups":1,"chunk_size":128,"time_step_limit":[0,{"__float__":"Infinity"}],
        "time_step_min":0.001,"time_step_max":0.1,"time_step_floor":0.0001,"intermediate_size":0,
        "mlp_hidden_act":"relu2","mlp_bias":false,"moe_intermediate_size":2688,"moe_latent_size":1024,
        "moe_shared_expert_intermediate_size":5376,"moe_shared_expert_overlap":false,
        "n_routed_experts":512,"n_shared_experts":1,"num_experts_per_tok":22,"norm_topk_prob":true,
        "routed_scaling_factor":5.0,"n_group":1,"topk_group":1
    }"#).unwrap()
}
fn config() -> TextConfig {
    serde_json::from_value(configuration()).unwrap()
}
fn limits() -> Limits {
    Limits {
        capacity: 8,
        max_resident_bytes: 192 << 20,
        max_chunk_bytes: 128 << 20,
    }
}

fn full_config(capacity: usize) -> TextConfig {
    let mut c = config();
    c.layers_block_type = vec![LayerKind::Mamba; 40];
    c.layers_block_type.extend(vec![LayerKind::Moe; 40]);
    c.layers_block_type.extend(vec![LayerKind::Attention; 8]);
    c.vocab_size = 131072;
    c.max_position_embeddings = capacity;
    c.mamba_num_heads = 128;
    c.ssm_state_size = 128;
    c.n_groups = 8;
    c.conv_kernel = 4;
    c.expand = 2;
    c
}

fn full_limits(capacity: usize) -> Limits {
    Limits { capacity, max_resident_bytes: 12 << 30, max_chunk_bytes: 6 << 30 }
}

#[test]
fn session_host_geometry_and_aggregate_admission() {
    let c = config();
    validate_config(&c, limits()).unwrap();
    let n = resident_bytes(&c, limits()).unwrap();
    assert!(n > 100 << 20 && n < 192 << 20);
    let mut l = limits();
    l.max_resident_bytes = n - 1;
    assert!(resident_bytes(&c, l).is_err());
    l = limits();
    l.capacity = 9;
    assert!(validate_config(&c, l).is_err());
    let mut wrong = c.clone();
    wrong.residual_in_fp32 = true;
    assert!(validate_config(&wrong, limits()).is_err());
    wrong = c.clone();
    wrong.mamba_head_dim = usize::MAX;
    assert!(validate_config(&wrong, limits()).is_err());
    wrong = c;
    wrong.routed_scaling_factor = 1.0;
    assert!(validate_config(&wrong, limits()).is_err());
    assert!(bytes(&[usize::MAX, 2], 2).is_err());
}

#[test]
fn session_host_expert_cache_budget_boundaries_and_overflow() {
    let base = resident_bytes(&config(), limits()).unwrap();
    let pair = moe::expert_device_bytes().unwrap();
    assert_eq!(pair, 3_096_576);
    assert_eq!(expert_cache_pairs(base, base, 1).unwrap(), 0);
    assert_eq!(expert_cache_pairs(base, base + pair - 1, 1).unwrap(), 0);
    assert_eq!(expert_cache_pairs(base, base + pair, 1).unwrap(), 1);
    assert_eq!(expert_cache_pairs(base, base + 2 * pair - 1, 2).unwrap(), 0);
    assert_eq!(expert_cache_pairs(base, base + 2 * pair, 2).unwrap(), 1);
    assert_eq!(expert_cache_pairs(base, usize::MAX, 0).unwrap(), 0);
    assert_eq!(expert_cache_pairs(base, usize::MAX, 1).unwrap(), moe::EXPERTS);
    assert!(expert_cache_pairs(base, base - 1, 0).is_err());
    assert!(expert_cache_pairs(base, usize::MAX, usize::MAX).is_err());
}

#[test]
fn session_host_expert_cache_full_geometry_uses_caller_resident_limit() {
    let c = full_config(2048);
    let mut l = Limits { capacity: 2048, max_resident_bytes: 8 << 30, max_chunk_bytes: 6 << 30 };
    validate_config(&c, l).unwrap();
    let base = resident_bytes(&c, l).unwrap();
    assert_eq!(base, 8_302_088_512);
    assert_eq!(expert_cache_pairs(base, l.max_resident_bytes, 40).unwrap(), 2);
    l.max_resident_bytes = 12 << 30;
    let pairs = expert_cache_pairs(base, l.max_resident_bytes, 40).unwrap();
    assert_eq!(pairs, 36);
    let resident = base + 40 * pairs * moe::expert_device_bytes().unwrap();
    assert_eq!(resident, 12_761_157_952);
    assert!(resident <= l.max_resident_bytes);
    assert!(resident + 40 * moe::expert_device_bytes().unwrap() > l.max_resident_bytes);
}

#[test]
fn session_host_admission_full_geometry_matches_existing_accounting() {
    // Arithmetic witnesses from the pre-extraction formulas, not a device
    // measurement or an assertion about allocator/host/runtime overhead.
    let a = admission(&full_config(2048), full_limits(2048), 88).unwrap();
    assert_eq!(a.base_resident_bytes, 8_302_088_512);
    assert_eq!(a.expert_cache_pairs_per_moe, 36);
    assert_eq!(a.expert_cache_bytes, 4_459_069_440);
    assert_eq!(a.resident_bytes, 12_761_157_952);
    assert_eq!(a.first_chunk_tokens, 88);
    assert_eq!(a.first_chunk_bytes, 1_809_020_264);
    assert_eq!(a.max_prompt_chunk_bytes, 1_809_020_264);
    assert_eq!(a.max_capacity_chunk_bytes, 4_992_726_024);

    let a = admission(&full_config(262144), full_limits(262144), 513).unwrap();
    assert_eq!(a.base_resident_bytes, 10_432_794_944);
    assert_eq!(a.expert_cache_pairs_per_moe, 19);
    assert_eq!(a.resident_bytes, 12_786_192_704);
    assert_eq!(a.first_chunk_tokens, MAX_CHUNK);
    assert_eq!(a.first_chunk_bytes, 4_924_568_584);
    assert_eq!(a.max_prompt_chunk_bytes, 4_992_726_024);
    assert_eq!(a.max_capacity_chunk_bytes, 4_992_726_024);
}

#[test]
fn session_host_admission_partial_last_chunk_and_attention_split_boundaries() {
    let c = full_config(2048);
    let l = full_limits(2048);
    // A late 255-token remainder can be larger than the first 256-token
    // chunk because its attention split count has grown from two to four.
    for (tokens, expected) in [
        (1, 217_940_364),
        (255, 4_907_653_764),
        (256, 4_924_568_584),
        (257, 4_924_568_584),
        (300, 4_924_568_584),
        (511, 4_975_544_964),
        (512, 4_992_726_024),
        (513, 4_992_726_024),
    ] {
        assert_eq!(admission(&c, l, tokens).unwrap().max_prompt_chunk_bytes, expected);
    }
    // Max-capacity accounting considers decode too, at its latest start.
    assert_eq!(planned_chunk_bytes(&c, l, 2047, 1).unwrap(), 219_937_164);
    let short = admission(&full_config(1), full_limits(1), 1).unwrap();
    assert_eq!(short.max_capacity_chunk_bytes, short.first_chunk_bytes);
}

#[test]
fn session_host_admission_budget_boundaries_precede_any_session() {
    let c = full_config(2048);
    let mut l = full_limits(2048);
    let base = 8_302_088_512;
    l.max_resident_bytes = base;
    let exact = admission(&c, l, 256).unwrap();
    assert_eq!(exact.expert_cache_pairs_per_moe, 0);
    assert_eq!(exact.expert_cache_bytes, 0);
    assert_eq!(exact.resident_bytes, base);
    l.max_resident_bytes -= 1;
    assert!(admission(&c, l, 256).is_err());
    l = full_limits(2048);
    l.max_chunk_bytes = 4_924_568_584;
    let first = admission(&c, l, 256).unwrap();
    assert_eq!(first.first_chunk_bytes, l.max_chunk_bytes);
    assert!(first.max_capacity_chunk_bytes > l.max_chunk_bytes);
    assert!(admission(&c, l, 300).is_ok());
    // The first chunk fits, but the required late partial chunk does not.
    assert!(admission(&c, l, 511).is_err());
    l.max_chunk_bytes -= 1;
    assert!(admission(&c, l, 256).is_err());
    assert!(admission(&full_config(1048576), full_limits(1048576), 1).is_err());
}

#[test]
fn session_host_admission_rejects_config_capacity_and_overflow() {
    let c = config();
    let l = limits();
    assert!(admission(&c, l, 0).is_err());
    assert!(admission(&c, l, l.capacity + 1).is_err());
    assert!(admission(&c, l, usize::MAX).is_err());
    for capacity in [0, 9, usize::MAX] {
        assert!(admission(&c, Limits { capacity, ..l }, 1).is_err());
    }
    for field in ["hidden_size", "num_attention_heads", "moe_latent_size"] {
        let mut value = configuration();
        value[field] = json!(1);
        let wrong: TextConfig = serde_json::from_value(value).unwrap();
        assert!(admission(&wrong, l, 1).is_err());
    }
    let mut wrong = c.clone();
    wrong.mamba_head_dim = usize::MAX;
    assert!(admission(&wrong, l, 1).is_err());
    wrong = c.clone();
    wrong.vocab_size = usize::MAX - 7; // Divisible by eight, but extent overflows.
    assert!(admission(&wrong, l, 1).is_err());
    assert!(planned_chunk_bytes(&c, l, usize::MAX, 1).is_err());
    assert!(planned_chunk_bytes(&c, l, 0, MAX_CHUNK + 1).is_err());
}

#[test]
fn session_host_config_rejects_unsupported_policies() {
    for field in [
        "attention_bias",
        "residual_in_fp32",
        "tie_word_embeddings",
        "moe_shared_expert_overlap",
    ] {
        let mut v = configuration();
        v[field] = json!(true);
        assert!(validate_config(&serde_json::from_value(v).unwrap(), limits()).is_err());
    }
    let mut c = config();
    c.layer_norm_epsilon = f64::MIN_POSITIVE;
    assert!(validate_config(&c, limits()).is_err());
    assert!(inject(Some(Fault::Reset), Fault::Reset).is_err());
    assert!(inject(Some(Fault::AfterCommit(1)), Fault::AfterLayer(1)).is_ok());
}

#[test]
fn session_host_complete_typed_expert_survives_merged_alternatives() {
    let mut repo = MemoryRepo::default();
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(key.verifying_key()),
        AdmissionPolicy::direct(key.verifying_key()),
    );
    let collection: model_collection::ModelCollection = repo
        .collection("merged-expert-alternatives", policy)
        .unwrap();
    let mut ids = [genid(), genid(), genid()];
    ids.sort_by_key(|id| **id);
    let layer = genid();
    let up = packed(&mut repo, "moe.up_proj", 2688, 1024);
    let down = packed(&mut repo, "moe.down_proj", 1024, 2688);
    let up_id = up.root().unwrap();
    let down_id = down.root().unwrap();
    let complete = up + down;
    let good =
        entity! {&ids[2] @ attrs::kind:pile::EXPERT,attrs::index:7u64,attrs::member*:complete};
    let good = entity! {&layer @ attrs::member*:good};
    repo.commit(collection, &key, good).unwrap();
    // Later independent facts add smaller same-index alternatives: one bare,
    // one with the right semantic roles but unsupported BF16 weight encoding.
    let bare = entity! {&ids[0] @ attrs::kind:pile::EXPERT,attrs::index:7u64};
    let unsupported = operation::<BF16, 2>(&mut repo, "moe.up_proj", [1, 1], bf(1.0, 1))
        + operation::<BF16, 2>(&mut repo, "moe.down_proj", [1, 1], bf(1.0, 1));
    let unsupported =
        entity! {&ids[1] @ attrs::kind:pile::EXPERT,attrs::index:7u64,attrs::member*:unsupported};
    let alternatives = bare + unsupported;
    repo.commit(
        collection,
        &key,
        entity! {&layer @ attrs::member*:alternatives},
    )
    .unwrap();
    let snapshot =
        model_collection::snapshot_model_collection_for(&repo.snapshot().unwrap(), collection)
            .unwrap();
    assert_eq!(pile::experts(&snapshot, *layer, 7).min(), Some(*ids[0]));
    let (expert, up, down) = pile::nvfp4_expert_minimum(&snapshot, *layer, 7)
        .unwrap()
        .unwrap();
    assert_eq!(expert, *ids[2]);
    assert_eq!((up.operation, down.operation), (up_id, down_id));
    assert_eq!(up.weight.dims(), &[2688, 1024]);
    assert_eq!(down.weight.dims(), &[1024, 2688]);
    assert!(pile::nvfp4_expert_minimum(&snapshot, *layer, 8).is_none());
    // The retained provider borrows its attributes across repeated lookups;
    // compare its selected operands with the standalone fresh-attribute path.
    let binding = moe::SourceBinding { model_root: *layer, layer: *layer };
    let provider = NativeExperts::new(Arc::new(snapshot.clone()), binding);
    for _ in 0..2 {
        moe::ExpertProvider::with_expert(&provider, 7, |pair| {
            assert_eq!(pair.index, 7);
            for (selected, original) in [(&pair.up, &up), (&pair.down, &down)] {
                assert_eq!(selected.operation, original.operation);
                assert_eq!(selected.payload, &original.weight.payload()[..]);
                assert_eq!(&selected.input_scale.to_le_bytes()[..], &original.input_scale.payload()[..]);
            }
            Ok(())
        }).unwrap();
    }
    assert!(moe::ExpertProvider::with_expert(&provider, 8, |_| Ok(())).is_err());
    #[cfg(feature = "nemotron-timing")]
    {
        let mut detail = moe::LookupTimings::default();
        let (observed_expert, observed_up, observed_down) =
            pile::nvfp4_expert_minimum_observed(&snapshot, *layer, 7, Some(&mut detail))
                .unwrap().unwrap();
        assert_eq!(observed_expert, expert);
        for (observed, original) in [(&observed_up, &up), (&observed_down, &down)] {
            assert_eq!(observed.operation, original.operation);
            assert_eq!(observed.weight.dims(), original.weight.dims());
            assert_eq!(&observed.weight.payload()[..], &original.weight.payload()[..]);
            assert_eq!(&observed.input_scale.payload()[..], &original.input_scale.payload()[..]);
        }
        assert_eq!(detail.counts(), [(1, 1), (1, 1), (4, 4)]);
        let mut reused = moe::LookupTimings::default();
        moe::ExpertProvider::with_expert_timed(&provider, 7, |pair| {
            assert_eq!((pair.up.operation, pair.down.operation), (up.operation, down.operation));
            Ok(())
        }, &mut reused).unwrap();
        assert_eq!(reused.counts(), detail.counts());
        let mut missing = moe::LookupTimings::default();
        assert!(pile::nvfp4_expert_minimum_observed(
            &snapshot, *layer, 8, Some(&mut missing),
        ).is_none());
        assert_eq!(missing.counts(), [(1, 1), (1, 1), (0, 0)]);
    }

    // A fully typed selected tuple whose blob is absent must still report an
    // error. It must not silently fall through to a different observation.
    let missing = leaf::leaf_blob::<NVFP4, 2>(
        [8, 16],
        Bytes::from_source({
            let mut data = vec![0u8; 64];
            data.extend(vec![0x38; 8]);
            data.extend(1.0f32.to_le_bytes());
            data
        }),
    )
    .unwrap();
    let w = entity! {leaf::leaf::<NVFP4,2>():missing.get_handle()};
    let scale = operation::<F32, 0>(
        &mut repo,
        native::INPUT_SCALE,
        [],
        1.0f32.to_le_bytes().to_vec(),
    );
    let bad_up = entity! {attrs::kind:"moe.up_proj",attrs::weight*:w,attrs::member*:scale};
    let bad_down = packed(&mut repo, "moe.down_proj", 8, 16);
    let pair = bad_up + bad_down;
    let expert = entity! {attrs::kind:pile::EXPERT,attrs::index:8u64,attrs::member*:pair};
    repo.commit(collection, &key, entity! {&layer @ attrs::member*:expert})
        .unwrap();
    let snapshot =
        model_collection::snapshot_model_collection_for(&repo.snapshot().unwrap(), collection)
            .unwrap();
    assert!(
        pile::nvfp4_expert_minimum(&snapshot, *layer, 8)
            .unwrap()
            .is_err()
    );
    let provider = NativeExperts::new(Arc::new(snapshot.clone()), binding);
    let reused_error = moe::ExpertProvider::with_expert(&provider, 8, |_| Ok(())).unwrap_err();
    let fresh_error = pile::nvfp4_expert_minimum(&snapshot, *layer, 8).unwrap().err().unwrap();
    assert_eq!(reused_error, fresh_error.to_string());
    #[cfg(feature = "nemotron-timing")]
    {
        let mut detail = moe::LookupTimings::default();
        let observed = pile::nvfp4_expert_minimum_observed(
            &snapshot, *layer, 8, Some(&mut detail),
        ).unwrap().err().unwrap();
        let original = pile::nvfp4_expert_minimum(&snapshot, *layer, 8)
            .unwrap().err().unwrap();
        assert_eq!(observed.to_string(), original.to_string());
        assert_eq!(detail.counts(), [(1, 1), (1, 1), (1, 0)]);
    }
}

fn tensor<T: TensorElement, const R: usize>(
    repo: &mut MemoryRepo,
    dims: [u64; R],
    bytes: Vec<u8>,
) -> Fragment {
    let blob = leaf::leaf_blob::<T, R>(dims, Bytes::from_source(bytes)).unwrap();
    let h = repo.put(blob).unwrap();
    entity! {leaf::leaf::<T,R>():h}
}
fn operation<T: TensorElement, const R: usize>(
    repo: &mut MemoryRepo,
    role: &str,
    dims: [u64; R],
    bytes: Vec<u8>,
) -> Fragment {
    let w = tensor::<T, R>(repo, dims, bytes);
    entity! {attrs::kind:role,attrs::weight*:w}
}
fn bf(value: f32, n: usize) -> Vec<u8> {
    bf16::from_f32(value).to_bits().to_le_bytes().repeat(n)
}
fn dense(repo: &mut MemoryRepo, role: &str, n: usize, k: usize) -> Fragment {
    let mut w = vec![0u8; n * k];
    for row in 0..n {
        w[row * k + row % k] = if row % 2 == 0 { 0x38 } else { 0xb8 };
    }
    let w = tensor::<E4M3, 2>(repo, [n as u64, k as u64], w);
    let mut scales = operation::<F32, 0>(
        repo,
        native::INPUT_SCALE,
        [],
        (1.0f32 / 64.0).to_le_bytes().to_vec(),
    );
    scales += operation::<F32, 0>(
        repo,
        native::FP8_WEIGHT_SCALE,
        [],
        (1.0f32 / 32.0).to_le_bytes().to_vec(),
    );
    entity! {attrs::kind:role,attrs::weight*:w,attrs::member*:scales}
}
fn packed(repo: &mut MemoryRepo, role: &str, n: usize, k: usize) -> Fragment {
    let mut w = vec![0u8; n * k / 2];
    for row in 0..n {
        w[row * k / 2] = 0x22;
    }
    w.extend(vec![0x38; n * k / 16]);
    w.extend((1.0f32 / 32.0).to_le_bytes());
    let w = tensor::<NVFP4, 2>(repo, [n as u64, k as u64], w);
    let scale = operation::<F32, 0>(
        repo,
        native::INPUT_SCALE,
        [],
        (1.0f32 / 64.0).to_le_bytes().to_vec(),
    );
    entity! {attrs::kind:role,attrs::weight*:w,attrs::member*:scale}
}
fn sparse_bf(n: usize, k: usize) -> Vec<u8> {
    let mut v = vec![0u8; n * k * 2];
    for row in 0..n {
        v[(row * k + row % k) * 2..(row * k + row % k) * 2 + 2]
            .copy_from_slice(&bf16::from_f32(1.0 / 32.0).to_bits().to_le_bytes());
    }
    v
}
fn fixture() -> (Arc<ModelSnapshot<impl BlobStoreGet>>, Source, Vec<Id>) {
    let mut repo = MemoryRepo::default();
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(key.verifying_key()),
        AdmissionPolicy::direct(key.verifying_key()),
    );
    let collection: model_collection::ModelCollection =
        repo.collection("synthetic-hybrid", policy).unwrap();
    let c = config();
    let g = mamba_config(&c).layout().unwrap();
    let mut layer_ids = Vec::new();
    let mut members = Fragment::empty();
    for (index, kind) in c.layers_block_type.iter().enumerate() {
        let mut ops = operation::<BF16, 1>(&mut repo, "pre-norm", [H as u64], bf(1.0, H));
        match kind {
            LayerKind::Mamba => {
                ops += dense(&mut repo, "mamba.in_proj", g.projected, H);
                ops += dense(&mut repo, "mamba.out_proj", H, g.inner);
                ops += operation::<BF16, 3>(
                    &mut repo,
                    "mamba.conv1d.weight",
                    [g.conv_channels as u64, 1, 2],
                    bf(0.5, g.conv_channels * 2),
                );
                ops += operation::<BF16, 1>(
                    &mut repo,
                    "mamba.conv1d.bias",
                    [g.conv_channels as u64],
                    bf(0.01, g.conv_channels),
                );
                ops += operation::<BF16, 1>(
                    &mut repo,
                    "mamba.norm",
                    [g.inner as u64],
                    bf(1.0, g.inner),
                );
                for (role, v) in [
                    ("mamba.A_log", 0.0f32),
                    ("mamba.D", 1.0),
                    ("mamba.dt_bias", -1.0),
                ] {
                    ops += operation::<F32, 1>(&mut repo, role, [64], v.to_le_bytes().repeat(64));
                }
            }
            LayerKind::Attention => {
                for (role, n, k) in [
                    ("attention.q_proj", H, H),
                    ("attention.k_proj", 256, H),
                    ("attention.v_proj", 256, H),
                    ("attention.o_proj", H, H),
                ] {
                    ops += dense(&mut repo, role, n, k);
                }
            }
            LayerKind::Moe => {
                ops += operation::<BF16, 2>(
                    &mut repo,
                    "moe.router",
                    [512, H as u64],
                    bf(0.0, 512 * H),
                );
                ops += operation::<F32, 1>(
                    &mut repo,
                    "moe.router-bias",
                    [512],
                    (0..512)
                        .flat_map(|i| ((512 - i) as f32).to_le_bytes())
                        .collect(),
                );
                ops += operation::<BF16, 2>(
                    &mut repo,
                    "moe.fc1_latent_proj",
                    [1024, H as u64],
                    sparse_bf(1024, H),
                );
                ops += operation::<BF16, 2>(
                    &mut repo,
                    "moe.fc2_latent_proj",
                    [H as u64, 1024],
                    sparse_bf(H, 1024),
                );
                ops += packed(&mut repo, "moe.shared.up_proj", 5376, H);
                ops += packed(&mut repo, "moe.shared.down_proj", H, 5376);
                // Only the selected experts are present: constructor must NOT
                // inventory/require a complete expert bank. Router picks 0..21.
                for index in 0..22u64 {
                    let expert = packed(&mut repo, "moe.up_proj", 2688, 1024)
                        + packed(&mut repo, "moe.down_proj", 1024, 2688);
                    ops +=
                        entity! {attrs::kind:pile::EXPERT,attrs::index:index,attrs::member*:expert};
                }
            }
        }
        let layer = entity! {attrs::kind:pile::LAYER,attrs::index:index as u64,attrs::member*:ops};
        layer_ids.push(layer.root().unwrap());
        members += layer;
    }
    let embedding = (0..32 * H)
        .flat_map(|i| {
            bf16::from_f32(((i * 7 % 29) as f32 - 14.0) / 16.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    members += operation::<BF16, 2>(&mut repo, "embedding", [32, H as u64], embedding);
    members += operation::<BF16, 1>(&mut repo, "final-norm", [H as u64], bf(1.0, H));
    members += dense(&mut repo, "lm-head", 32, H);
    let mut facts = TribleSet::new();
    let config =
        jsonfacts::save_json(&json!({"text":configuration()}), &mut repo, &mut facts).unwrap();
    members += entity! {attrs::kind:"nemotron-h-config",attrs::member:config};
    // A source-binding fixture only; this milestone does not rebuild a codec.
    let base = *genid();
    let base_node = entity! {attrs::kind:tokenizer_binding::BASE,attrs::member:base};
    let pre = entity! {attrs::kind:tokenizer_binding::PRE_TOKENIZER};
    let pre_id = pre.root().unwrap();
    let nodes = base_node + pre;
    let tokenizer = entity! {attrs::kind:tokenizer_binding::BINDING,attrs::member*:nodes};
    let tokenizer_id = tokenizer.root().unwrap();
    members += tokenizer;
    let mut root = entity! {attrs::kind:"nemotron-h",attrs::member*:members,tokenizer::attrs::tokenizer:tokenizer_id};
    let root_id = root.root().unwrap();
    root += Fragment::from(facts);
    repo.commit(collection, &key, root).unwrap();
    let snapshot =
        model_collection::snapshot_model_collection_for(&repo.snapshot().unwrap(), collection)
            .unwrap();
    (
        Arc::new(snapshot),
        Source {
            collection: collection.handle(),
            root: root_id,
            config,
            tokenizer_binding: tokenizer_id,
            tokenizer_base: base,
            pre_tokenizer: pre_id,
        },
        layer_ids,
    )
}

#[test]
#[ignore = "reserved CUDA: synthetic three-kind session/partition/reset, no model acceptance"]
fn session_gpu_three_kind_native_chain() {
    let (snapshot, source, ids) = fixture();
    let mut session =
        Session::from_snapshot(snapshot, source, &ids, CudaDevice { index: 0 }, limits()).unwrap();
    assert_eq!(session.status(), Status::Ready);
    assert!(session.last_logits().is_none());
    assert!(session.append_ids(&[]).is_err());
    assert!(session.append_ids(&[32]).is_err());
    assert_eq!(session.status(), Status::Ready);
    let all = session.append_ids(&[1, 5, 7]).unwrap().values.to_vec();
    assert_eq!(session.committed_ids(), &[1, 5, 7]);
    assert!(all.iter().any(|&v| v != 0.0));
    assert!(session.can_reuse_prefix(&[1, 5, 7, 3]));
    assert!(!session.can_reuse_prefix(&[1, 5, 6]));
    session.reset().unwrap();
    session.append_ids(&[1, 5]).unwrap();
    let split = session.append_ids(&[7]).unwrap().values.to_vec();
    for (a, b) in all.iter().zip(&split) {
        assert!((a - b).abs() <= 0.02 + 0.02 * a.abs());
    }
    session.reset().unwrap();
    for id in [1, 5, 7] {
        session.append_ids(&[id]).unwrap();
    }
    for (a, b) in all.iter().zip(session.last_logits().unwrap().values) {
        assert!((a - b).abs() <= 0.02 + 0.02 * a.abs());
    }
    session.invalidate();
    assert!(session.last_logits().is_none());
    assert!(!session.can_reuse_prefix(&[1, 5, 7]));
    assert!(session.append_ids(&[3]).is_err());
    session.reset().unwrap();
    assert!(session.committed_ids().is_empty());
    let again = session.append_ids(&[1, 5, 7]).unwrap().values.to_vec();
    assert_eq!(all, again);
    for layer in &session.layers {
        match &layer.mixer {
            Mixer::Mamba(m) => assert_eq!(m.committed.as_ref().unwrap().attended(), 3),
            Mixer::Attention(a) => assert_eq!(a.attended(), 3),
            Mixer::Moe(m) => assert!(!m.is_poisoned()),
        }
    }
    session.reset().unwrap();
    // This is a synchronous API. The caller may decline a chunk at a Ready
    // boundary; there is no GPU cancel/rollback promise to exercise here.
    assert_eq!(session.status(), Status::Ready);
    assert!(session.committed_ids().is_empty());
    for fault in [Fault::AfterLayer(1), Fault::AfterCommit(1)] {
        session.fault = Some(fault);
        assert!(session.append_ids(&[1, 5, 7]).is_err());
        assert_eq!(session.status(), Status::Poisoned);
        assert!(session.committed_ids().is_empty());
        assert!(!session.pending.is_empty());
        assert!(session.last_logits().is_none());
        let held = session.pending.len();
        session.fault = Some(Fault::Reset);
        assert!(session.reset().is_err());
        assert_eq!(session.status(), Status::Poisoned);
        assert_eq!(session.pending.len(), held);
        session.fault = None;
        session.reset().unwrap();
        assert!(session.pending.is_empty());
    }
    session.fault = Some(Fault::PanicAfterLayer(1));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = session.append_ids(&[1]);
        }))
        .is_err()
    );
    assert_eq!(session.status(), Status::Poisoned);
    assert!(session.append_ids(&[2]).is_err());
    session.fault = None;
    session.reset().unwrap();
}
