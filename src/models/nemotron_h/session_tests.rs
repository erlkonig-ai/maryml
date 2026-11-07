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

pub(super) fn configuration() -> Value {
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
pub(super) fn config() -> TextConfig {
    serde_json::from_value(configuration()).unwrap()
}
pub(in crate::models::nemotron_h) fn limits() -> Limits {
    Limits {
        capacity: 8,
        max_resident_bytes: 2 << 30,
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
    Limits { capacity, max_resident_bytes: 96 << 30, max_chunk_bytes: 6 << 30 }
}

#[test]
fn session_host_prefix_admission_counts_shared_boundary_and_only_private_suffix() {
    let c = full_config(1 << 20);
    // The exact minimum resident allowance at 1M: the whole model, bank included.
    let l = Limits { capacity: 1 << 20, max_resident_bytes: 80_293_122_368, max_chunk_bytes: usize::MAX };
    assert!(prefix_admission(&c, Limits { max_resident_bytes: l.max_resident_bytes - 1, ..l },
        786_432 + 17, 786_432).is_err());
    let p = prefix_admission(&c, l, 786_432 + 17, 786_432).unwrap();
    assert_eq!(p.after_advance.shared_kv_bytes, 6 << 30);
    assert_eq!(p.after_advance.private_kv_bytes, 2 << 30);
    assert_eq!(p.after_advance.shared_mamba_bytes, 171_048_960);
    assert_eq!(p.after_advance.private_mamba_bytes, 171_048_960);
    assert_eq!(resident_ownership(&c, l).unwrap().expert_bank_bytes, 63_417_876_480);
    assert_eq!(p.suffix.unwrap().device_working_bytes,
        planned_prefix_chunk_bytes(&c, l, 786_432, 17, 786_432).unwrap());
    // Diagnostic from the actual planner, not a measured allocation.
    // Reset retains the prefix while replacing private buffers; 2C bounds KV
    // overlap including an old tensor during replacement, before pool overhead.
    println!("PREFIX_ALLOCATION {}", json!({
        "scope": "synthetic_full_geometry_logical_payload_not_physical_peak",
        "capacity": l.capacity, "prefix_tokens": p.after_advance.prefix_tokens,
        "seed_tokens": p.seed.tokens, "seed_working_bytes": p.seed.device_working_bytes,
        "suffix_tokens": p.suffix.unwrap().tokens,
        "suffix_working_bytes": p.suffix.unwrap().device_working_bytes,
        "shared_kv_bytes": p.after_advance.shared_kv_bytes,
        "private_kv_bytes": p.after_advance.private_kv_bytes,
        "extra_retained_mamba_bytes": p.after_advance.shared_mamba_bytes,
        "cold_miss_kv_overlap_conservative_bytes": 2 * (p.after_advance.shared_kv_bytes + p.after_advance.private_kv_bytes),
    }));
    assert_eq!(prefix_admission(&c, l, l.capacity, l.capacity).unwrap().after_advance.private_kv_bytes, 0);
    assert!(prefix_admission(&c, l, l.capacity, l.capacity).unwrap().suffix.is_none());
    for (prompt, prefix) in [(0, 0), (1, 0), (2, 3), (l.capacity + 1, 1), (1, usize::MAX)] {
        assert!(prefix_admission(&c, l, prompt, prefix).is_err());
    }
    assert!(prefix::validate_seed(&c, l, Some(&[])).is_err());
    assert!(prefix::validate_seed(&c, l, Some(&[c.vocab_size as u32])).is_err());
    assert!(prefix::validate_seed(&c, l, Some(&[1, 2])).is_ok());
}

#[test]
fn session_host_prefill_full_window_pages_and_allocation_inventory() {
    let c = full_config(1 << 20);
    let l = Limits { capacity: 1 << 20, max_resident_bytes: 80_293_122_368,
        max_chunk_bytes: usize::MAX };
    // No pending expert pair: each total is 3_096_584 below its cached-era figure.
    for (tokens, pages, hidden, latent, accumulator, total) in [
        (785_039, 3, 6_431_039_488, 1_607_759_872, 3_215_519_744, 11_354_723_328),
        (1_048_576, 4, 8_589_934_592, 2_147_483_648, 4_294_967_296, 15_132_789_760),
    ] {
        let a = prefill_admission(&c, l, 0, tokens).unwrap();
        assert!(a.layer_major);
        assert_eq!((a.hidden_pages, a.hidden_bytes, a.latent_bytes, a.accumulator_bytes),
            (pages, hidden, latent, accumulator));
        assert_eq!(a.largest_hidden_page_bytes, 1 << 31);
        assert_eq!(a.tile_workspace_bytes, 100_404_224);
        assert_eq!(a.pending_expert_bytes, 0);
        assert_eq!(a.device_working_bytes, total);
        assert_eq!(a.host_routing_bytes, tokens * 272 + tokens.div_ceil(64) * 512 * 8);
        assert!(prefill_admission(&c, Limits { max_chunk_bytes: total, ..l }, 0, tokens).is_ok());
        assert!(prefill_admission(&c, Limits { max_chunk_bytes: total - 1, ..l }, 0, tokens).is_err());
    }
    let tail = prefill_admission(&c, l, 7, prefill::PAGE_ROWS + 1).unwrap();
    assert_eq!(tail.hidden_pages, 2);
    assert_eq!(tail.hidden_bytes, (prefill::PAGE_ROWS + 1) * H * 2);
    let short = prefill_admission(&c, l, 17, 256).unwrap();
    assert!(!short.layer_major);
    assert_eq!(short.device_working_bytes, planned_chunk_bytes(&c, l, 17, 256).unwrap());
    for (start, tokens) in [(0, 0), (l.capacity, 1), (usize::MAX, 2), (1, l.capacity)] {
        assert!(prefill_admission(&c, l, start, tokens).is_err());
    }
}

#[test]
fn session_host_full_tile_chunk_and_span_budgets() {
    // Isolate attention's contribution so an unchanged larger Mamba maximum
    // cannot hide stale native split accounting in the aggregate planners.
    let mut c = full_config(1 << 20);
    c.layers_block_type = vec![LayerKind::Attention];
    let l = Limits { max_chunk_bytes: usize::MAX, ..full_limits(1 << 20) };
    for (prefix, chunk_bytes, span_bytes, tile_bytes) in [
        (0, 103_298_048, 102_509_568, 100_404_224),
        (524_032, 107_557_888, 106_769_408, 104_664_064),
    ] {
        assert_eq!(planned_prefix_chunk_bytes(&c, l, 524_032, 256, prefix).unwrap(), chunk_bytes);
        let short = prefill::admission_with_prefix(&c, l, 524_032, 256, prefix).unwrap();
        assert!(!short.layer_major);
        assert_eq!(short.device_working_bytes, chunk_bytes);
        let span = prefill::admission_with_prefix(&c, l, 524_032, 257, prefix).unwrap();
        assert!(span.layer_major);
        assert_eq!(span.tile_workspace_bytes, tile_bytes);
        assert_eq!(span.device_working_bytes, span_bytes);
        assert_eq!(span.hidden_bytes, 257 * H * 2);
        for (tokens, exact) in [(256, chunk_bytes), (257, span_bytes)] {
            assert!(prefill::admission_with_prefix(&c,
                Limits { max_chunk_bytes: exact, ..l }, 524_032, tokens, prefix).is_ok());
            assert!(prefill::admission_with_prefix(&c,
                Limits { max_chunk_bytes: exact - 1, ..l }, 524_032, tokens, prefix).is_err());
        }
    }
}

#[test]
fn session_host_prefill_covers_actual_prefix_tile_peak() {
    let mut c = full_config(131072);
    c.layers_block_type = vec![LayerKind::Attention];
    let l = Limits { max_chunk_bytes: usize::MAX, ..full_limits(131072) };
    let earlier = layer_prefix_working_bytes(&c, l, 130560, 256, &LayerKind::Attention, 8192).unwrap();
    let final_width_peak = (1..=256).map(|t| layer_prefix_working_bytes(&c, l,
        131072 - t, t, &LayerKind::Attention, 8192).unwrap()).max().unwrap();
    assert!(earlier > final_width_peak, "actual earlier tile must not be hidden by final-width scan");
    let a = prefill::admission_with_prefix(&c, l, 8192, 131072 - 8192, 8192).unwrap();
    let outer = bytes(&[MAX_CHUNK, H], 8).unwrap() + bytes(&[MAX_CHUNK], 12).unwrap();
    assert!(a.tile_workspace_bytes >= outer + earlier);
    assert!(prefill::admission_with_prefix(&c,
        Limits { max_chunk_bytes: a.device_working_bytes - 1, ..l },
        8192, 131072 - 8192, 8192).is_err());
    let whole = prefix_admission(&c, l, 131072, 8192).unwrap();
    assert_eq!(whole.suffix.unwrap().device_working_bytes, a.device_working_bytes);
    let cold = admission(&c, l, 4096).unwrap();
    assert_eq!(cold.prefill.device_working_bytes,
        prefill_admission(&c, l, 0, 4096).unwrap().device_working_bytes);
}

#[test]
fn session_host_shared_ownership_counts_the_bank_once() {
    let c = full_config(1 << 20);
    let small = resident_ownership(&c, full_limits(2048)).unwrap();
    let large = resident_ownership(&c, full_limits(262144)).unwrap();
    // 40 MoE layers x 512 pairs x 3_096_576 device bytes.
    assert_eq!(small.expert_bank_bytes, 63_417_876_480);
    assert_eq!(large.expert_bank_bytes, small.expert_bank_bytes);
    // 8_114_262_336 fixed operands without experts, plus the bank.
    assert_eq!(small.fixed_model_bytes, 71_532_138_816);
    assert_eq!(large.fixed_model_bytes, small.fixed_model_bytes);
    assert_eq!(small.private_state_bytes, 187_826_176);
    assert_eq!(large.private_state_bytes, 2_318_532_608);
    for (capacity, ownership) in [(2048, small), (262144, large)] {
        let admitted = admission(&c, full_limits(capacity), 1).unwrap();
        assert_eq!(admitted.expert_cache_pairs_per_moe, moe::EXPERTS);
        assert_eq!(admitted.expert_cache_bytes, ownership.expert_bank_bytes);
        assert_eq!(ownership.fixed_model_bytes - ownership.expert_bank_bytes
            + ownership.private_state_bytes, admitted.base_resident_bytes);
        assert_eq!(ownership.fixed_model_bytes + ownership.private_state_bytes,
            admitted.resident_bytes);
    }
    // Two Sessions of unequal context share the fixed operands and the bank.
    let combined = small.fixed_model_bytes + small.private_state_bytes + large.private_state_bytes;
    assert_eq!(combined, 74_038_497_600);
    let base = resident_bytes(&c, full_limits(2048)).unwrap();
    assert_eq!(base, 71_719_964_992);
    let rejected = Limits { max_resident_bytes: base - 1, ..full_limits(2048) };
    assert_eq!(resident_ownership(&c, rejected).unwrap_err(),
        admission(&c, rejected, 1).unwrap_err());
    let mut invalid = c;
    invalid.vocab_size = usize::MAX - 7;
    assert_eq!(resident_ownership(&invalid, full_limits(2048)).unwrap_err(),
        admission(&invalid, full_limits(2048), 1).unwrap_err());
}

#[test]
fn session_host_real_owners_satisfy_thread_bounds() {
    fn send<T: Send>() {}
    fn shared<T: Send + Sync>() {}
    type Store = triblespace::core::repo::pile::PileSnapshot;
    shared::<DeviceModel<Store>>();
    shared::<Prefix<Store>>();
    send::<Session<Store>>();
    #[cfg(feature = "tokenizer")]
    send::<super::super::generation::Generator<Store>>();
}

#[test]
fn session_host_execution_stream_scope_restores_after_return_and_unwind() {
    // Use a real other thread's identity; no CUDA client or allocations.
    let owner = std::thread::spawn(StreamId::current).join().unwrap();
    let caller = StreamId::current();
    assert_ne!(owner, caller);
    owner.executes(|| assert_eq!(StreamId::current(), owner));
    assert_eq!(StreamId::current(), caller);
    assert!(std::panic::catch_unwind(|| owner.executes(|| {
        assert_eq!(StreamId::current(), owner);
        panic!("synthetic session unwind");
    })).is_err());
    assert_eq!(StreamId::current(), caller);
}

fn rejected_upload_layout() -> (cubecl::zspace::Shape, cubecl::zspace::Strides) {
    // A transposed 2x2 byte layout: valid ranks and extent, rejected because
    // the last stride is not 1. CUDA Command::write_to_gpu returns
    // UnsupportedStrides BEFORE resource lookup, staging, or memcpy.
    ([2, 2].into(), [1, 2].into())
}

#[test]
fn session_host_upload_fault_is_a_checked_descriptor_rejection() {
    let (shape, strides) = rejected_upload_layout();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape.len(), strides.len());
    assert_eq!(shape.iter().product::<usize>(), 4);
    // This is a defined false return, not the validator's malformed-rank panic.
    assert!(!cubecl::zspace::striding::has_pitched_row_major_strides(&shape, &strides));
}

pub(super) fn enqueue_rejected_upload(device: &CudaDevice, handle: cubecl::server::Handle) {
    use cubecl::{device::Device, device_handle::DeviceHandle, server::{ComputeServer, CopyDescriptor}};
    assert!(handle.size() >= 4);
    let stream = StreamId::current();
    assert_eq!(handle.stream, stream);
    let (shape, strides) = rejected_upload_layout();
    let descriptor = CopyDescriptor::new(handle.binding(), shape, strides, 1);
    let service = DeviceHandle::<<CudaRuntime as Runtime>::Server>::new(device.to_id());
    service.submit_blocking(move |server| {
        server.write(vec![(descriptor, cubecl::bytes::Bytes::from_bytes_vec(vec![1, 2, 3, 4]))], stream);
    }).unwrap();
}

#[path = "session_prefix_tests.rs"]
mod prefix_tests;

#[test]
fn session_host_geometry_and_aggregate_admission() {
    let c = config();
    validate_config(&c, limits()).unwrap();
    let n = resident_bytes(&c, limits()).unwrap();
    // 132_574_040 without experts, plus one layer's bank of 1_585_446_912.
    assert_eq!(n, 1_718_020_952);
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
fn session_host_bank_full_geometry_minimum_resident_limit() {
    for (capacity, minimum) in [(2048, 71_719_964_992), (1 << 20, 80_293_122_368)] {
        let c = full_config(capacity);
        let l = Limits { capacity, max_resident_bytes: minimum, max_chunk_bytes: 6 << 30 };
        validate_config(&c, l).unwrap();
        assert_eq!(resident_bytes(&c, l).unwrap(), minimum);
        assert!(resident_bytes(&c, Limits { max_resident_bytes: minimum - 1, ..l }).is_err());
    }
}

#[test]
fn session_host_admission_full_geometry_matches_existing_accounting() {
    // Arithmetic witnesses from the pre-extraction formulas, attention terms
    // from the tensor-core plan; not a device measurement or an assertion
    // about allocator/host/runtime overhead.
    // Chunks no longer carry a pending expert pair: 3_096_584 below the
    // cached-era figures. Resident adds the whole bank, 63_417_876_480.
    let a = admission(&full_config(2048), full_limits(2048), 88).unwrap();
    assert_eq!(a.base_resident_bytes, 8_302_088_512);
    assert_eq!(a.expert_cache_pairs_per_moe, 512);
    assert_eq!(a.expert_cache_bytes, 63_417_876_480);
    assert_eq!(a.resident_bytes, 71_719_964_992);
    assert_eq!(a.first_chunk_tokens, 88);
    assert_eq!(a.first_chunk_bytes, 1_794_389_344);
    assert_eq!(a.max_prompt_chunk_bytes, 1_794_389_344);
    assert_eq!(a.max_capacity_chunk_bytes, 5_092_389_888);

    let a = admission(&full_config(262144), full_limits(262144), 513).unwrap();
    assert_eq!(a.base_resident_bytes, 10_432_794_944);
    assert_eq!(a.expert_cache_pairs_per_moe, 512);
    assert_eq!(a.resident_bytes, 73_850_671_424);
    assert_eq!(a.first_chunk_tokens, MAX_CHUNK);
    assert_eq!(a.first_chunk_bytes, 4_887_917_568);
    assert_eq!(a.max_prompt_chunk_bytes, 4_956_075_008);
    assert_eq!(a.max_capacity_chunk_bytes, 5_365_019_648);
}

#[test]
fn session_host_admission_partial_last_chunk_and_attention_split_boundaries() {
    let c = full_config(2048);
    let l = full_limits(2048);
    // A late 255-token remainder can be larger than the first 256-token
    // chunk because its attention split count has grown from two to four.
    for (tokens, expected) in [
        (1, 214_712_708),
        (255, 4_871_133_820),
        (256, 4_887_917_568),
        (257, 4_887_917_568),
        (300, 4_887_917_568),
        (511, 4_939_025_020),
        (512, 4_956_075_008),
        (513, 4_956_075_008),
    ] {
        assert_eq!(admission(&c, l, tokens).unwrap().max_prompt_chunk_bytes, expected);
    }
    // Max-capacity accounting considers decode too, at its latest start.
    assert_eq!(planned_chunk_bytes(&c, l, 2047, 1).unwrap(), 216_709_508);
    let short = admission(&full_config(1), full_limits(1), 1).unwrap();
    assert_eq!(short.max_capacity_chunk_bytes, short.first_chunk_bytes);
}

#[test]
fn session_host_admission_budget_boundaries_precede_any_session() {
    let c = full_config(2048);
    let mut l = full_limits(2048);
    let base = 71_719_964_992;
    l.max_resident_bytes = base;
    let exact = admission(&c, l, 256).unwrap();
    assert_eq!(exact.expert_cache_pairs_per_moe, 512);
    assert_eq!(exact.expert_cache_bytes, 63_417_876_480);
    assert_eq!(exact.resident_bytes, base);
    l.max_resident_bytes -= 1;
    assert!(admission(&c, l, 256).is_err());
    l = full_limits(2048);
    l.max_chunk_bytes = 4_887_917_568;
    let first = admission(&c, l, 256).unwrap();
    assert_eq!(first.first_chunk_bytes, l.max_chunk_bytes);
    assert!(first.max_capacity_chunk_bytes > l.max_chunk_bytes);
    assert!(admission(&c, l, 300).is_ok());
    // Long prefill admits its actual tile/slab schedule. The old late chunk
    // comparison may exceed the allowance without being dispatched.
    let span = admission(&c, l, 511).unwrap();
    assert!(span.max_prompt_chunk_bytes > l.max_chunk_bytes);
    assert!(span.prefill.device_working_bytes < l.max_chunk_bytes);
    let span_exact = Limits { max_chunk_bytes: span.prefill.device_working_bytes, ..l };
    assert!(admission(&c, span_exact, 511).is_ok());
    assert!(admission(&c, Limits { max_chunk_bytes: span_exact.max_chunk_bytes - 1, ..l }, 511).is_err());
    l.max_chunk_bytes -= 1;
    assert!(admission(&c, l, 256).is_err());
    assert!(admission(&full_config(1048576), Limits { max_resident_bytes: 80_293_122_367,
        ..full_limits(1048576) }, 1).is_err());
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
    // The bank loader's provider holds its attributes across lookups; compare
    // its selected operands with the standalone fresh-attribute path.
    let binding = moe::SourceBinding { model_root: *layer, layer: *layer };
    let provider = source::NativeExperts::new(&snapshot, binding);
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
    let provider = source::NativeExperts::new(&snapshot, binding);
    let reused_error = moe::ExpertProvider::with_expert(&provider, 8, |_| Ok(())).unwrap_err();
    let fresh_error = pile::nvfp4_expert_minimum(&snapshot, *layer, 8).unwrap().err().unwrap();
    assert_eq!(reused_error, fresh_error.to_string());
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
pub(in crate::models::nemotron_h) fn fixture() -> (Arc<ModelSnapshot<impl BlobStoreGet + Send + Sync>>, Source, Vec<Id>) {
    fixture_capacity(8)
}
pub(in crate::models::nemotron_h) fn fixture_capacity(capacity: usize) -> (Arc<ModelSnapshot<impl BlobStoreGet + Send + Sync>>, Source, Vec<Id>) {
    fixture_experts(capacity, 512)
}
/// `experts` below 512 leaves the bank incomplete, so loading must refuse.
fn fixture_experts(capacity: usize, experts: u64) -> (Arc<ModelSnapshot<impl BlobStoreGet + Send + Sync>>, Source, Vec<Id>) {
    let mut repo = MemoryRepo::default();
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(key.verifying_key()),
        AdmissionPolicy::direct(key.verifying_key()),
    );
    let collection: model_collection::ModelCollection =
        repo.collection("synthetic-hybrid", policy).unwrap();
    let mut value = configuration();
    value["max_position_embeddings"] = json!(capacity);
    let c: TextConfig = serde_json::from_value(value.clone()).unwrap();
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
                // The whole bank: every index loads at construction. One pair of
                // payloads serves all 512; the router picks 0..21.
                let expert = packed(&mut repo, "moe.up_proj", 2688, 1024)
                    + packed(&mut repo, "moe.down_proj", 1024, 2688);
                for index in 0..experts {
                    let expert = expert.clone();
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
        jsonfacts::save_json(&json!({"text":value}), &mut repo, &mut facts).unwrap();
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

// Read only the committed state, on the session's execution stream. Comparing
// bytes before/after another context executes detects aliasing that merely
// checking two Rust wrapper addresses would miss.
#[test]
#[ignore = "reserved CUDA: synthetic layer-major equivalence/cancellation/poison/reset, not model acceptance"]
fn session_gpu_layer_major_prefill_publication_and_failure_owners() {
    let (snapshot, source, layers) = fixture_capacity(260);
    let limits = Limits { capacity: 260, max_chunk_bytes: 256 << 20, ..limits() };
    let mut session = Session::from_snapshot(snapshot, source, &layers,
        CudaDevice { index: 0 }, limits).unwrap();
    let prompt: Vec<_> = (0..257).map(|i| [1, 5, 7][i % 3]).collect();
    session.append_ids(&[9]).unwrap();
    for tile in prompt.chunks(MAX_CHUNK) { session.append_ids(tile).unwrap(); }
    let expected = session.last_logits().unwrap().values.to_vec();
    let expected_state = state_bytes(&session);
    session.reset().unwrap();
    session.append_ids(&[9]).unwrap();
    assert_eq!(session.prefill_ids(&prompt, &mut || false).unwrap(), PrefillOutcome::Completed);
    assert_eq!(session.committed_ids(), [vec![9], prompt.clone()].concat());
    assert!(session.span_pending.is_none() && session.pending.is_empty());
    assert_eq!(state_bytes(&session), expected_state);
    for (&a, &b) in expected.iter().zip(session.last_logits().unwrap().values) {
        assert!((a - b).abs() <= 0.02 + 0.02 * a.abs());
    }
    // Short append still follows the established path after span publication.
    session.append_ids(&[3]).unwrap();
    assert_eq!(session.committed_ids().len(), 259);
    session.reset().unwrap();
    session.append_ids(&[9]).unwrap();
    let old_state = state_bytes(&session);
    assert_eq!(session.prefill_ids(&prompt, &mut || true).unwrap(), PrefillOutcome::Cancelled);
    assert_eq!(session.status(), Status::Ready);
    assert_eq!(session.committed_ids(), [9]);
    assert_eq!(state_bytes(&session), old_state);
    // Cancellation after the first Mamba tile privately advances one layer.
    // It retains owners and cannot expose
    // the diagnostic old prefix as a reusable context.
    let mut polls = 0;
    assert_eq!(session.prefill_ids(&prompt, &mut || { polls += 1; polls == 7 }).unwrap(),
        PrefillOutcome::Cancelled);
    match &session.layers[0].mixer {
        Mixer::Mamba(m) => assert_eq!(m.committed.as_ref().unwrap().attended(), 257),
        _ => panic!("fixture begins with Mamba"),
    }
    assert_eq!(session.status(), Status::Poisoned);
    assert_eq!(session.committed_ids(), [9]);
    assert!(!session.can_reuse_prefix(&[9, 1]) && session.last_logits().is_none());
    assert!(session.span_pending.is_some());
    session.reset().unwrap();
    for fault in [Fault::AfterLayer(1), Fault::PanicAfterLayer(1)] {
        session.fault = Some(fault);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
            session.prefill_ids(&prompt, &mut || false)));
        match fault {
            Fault::PanicAfterLayer(_) => assert!(result.is_err()),
            _ => assert!(result.unwrap().is_err()),
        }
        assert_eq!(session.status(), Status::Poisoned);
        assert!(session.committed_ids().is_empty() && session.last_logits().is_none());
        assert!(session.span_pending.is_some());
        session.fault = Some(Fault::Reset);
        assert!(session.reset().is_err());
        assert!(session.span_pending.is_some(), "uncertain reset retains span owners");
        session.fault = None;
        session.reset().unwrap();
        assert!(session.span_pending.is_none());
    }
}

pub(super) fn state_bytes<S: BlobStoreGet>(session: &Session<S>) -> Vec<Vec<u8>> {
    session.stream.executes(|| {
        let mut result = Vec::new();
        for layer in &session.layers {
            match &layer.mixer {
                Mixer::Mamba(m) => {
                    if let Some(state) = &m.committed {
                        for tensor in [state.conv(), state.recurrent()] {
                            result.push(tensor.client.read_one(tensor.handle.clone()).unwrap().to_vec());
                        }
                    }
                }
                Mixer::Attention(a) => result.extend(a.diagnostic_kv().unwrap()),
                Mixer::Moe(_) => {}
            }
        }
        result
    })
}

#[test]
#[ignore = "reserved CUDA: rejected origin upload cannot publish shared model; no inference kernels"]
fn session_gpu_shared_model_rejects_origin_upload_error() {
    let (snapshot, source, ids) = fixture();
    let ownership = resident_ownership(&config(), limits()).unwrap();
    // One constructor only, no append/expert load or second context. Include
    // the logical committed state although Mamba state is still unallocated.
    assert_eq!(ownership.fixed_model_bytes + ownership.private_state_bytes, 1_718_020_952);
    let caller = StreamId::current();
    let origin = std::thread::spawn(StreamId::current).join().unwrap();
    assert_ne!(caller, origin);
    let device = CudaDevice { index: 0 };
    let error = origin.executes(|| source::load(
        snapshot, source, &ids, device.clone(), limits(), None,
        |client, device, embedding| {
            assert_eq!(StreamId::current(), origin);
            // A real ordinary CUDA-server write Err, on a real fixed operand's
            // origin stream. The checked rejection performs no invalid copy.
            enqueue_rejected_upload(device, embedding.handle.clone());
            caller.executes(|| {
                // Exercise the precise trap: an actual cross-stream handle
                // dependency waits for its origin event, yet reports no origin
                // error. Reading this VALID prior upload is safe; the rejected
                // replacement never touched it. The model must still refuse.
                let bytes = client.read_one(embedding.handle.clone()).unwrap();
                assert_eq!(bytes.len(), 32 * H * 2);
                assert_eq!(&bytes[..2], &bf16::from_f32(-14.0 / 16.0).to_bits().to_le_bytes());
                assert_eq!(&bytes[2..4], &bf16::from_f32(-7.0 / 16.0).to_bits().to_le_bytes());
                cubecl::future::block_on(client.sync()).unwrap();
            });
            assert_eq!(StreamId::current(), origin);
        },
    )).err().expect("an origin upload error must not publish a Session/DeviceModel");
    assert!(error.starts_with("session CUDA completion:"), "{error}");
    // CubeCL IoError's Debug delegates to Display, including this checked rejection.
    assert!(error.contains("the provided strides are not supported for this operation"), "{error}");
    assert_eq!(StreamId::current(), caller);
    // Test teardown only: the error-returning fence was NOT a drain. Observe a
    // separate successful completion after refusal; never retry construction.
    origin.executes(|| cubecl::future::block_on(CudaRuntime::client(&device).sync())).unwrap();
}

#[test]
#[ignore = "reserved CUDA: two synthetic Sessions, shared fixed weights/private state and stream movement"]
fn session_gpu_shared_model_private_state_and_streams() {
    // Only two sessions alive, capacity8 and at most ONE synchronous append at
    // a time. Logical bound: shared W and bank + 2*private state + 128MiB chunk.
    // Fixture source only; no checkpoint, tokenizer, external provider or pile.
    let (snapshot, source, ids) = fixture();
    let ownership = resident_ownership(&config(), limits()).unwrap();
    assert_eq!(ownership.fixed_model_bytes, 1_717_930_808);
    assert_eq!(ownership.expert_bank_bytes, 1_585_446_912);
    assert_eq!(ownership.private_state_bytes, 90_144);
    assert_eq!(ownership.fixed_model_bytes + 2 * ownership.private_state_bytes
        + limits().max_chunk_bytes, 1_852_328_824);
    let mut a = Session::from_snapshot(snapshot, source, &ids, CudaDevice { index: 0 }, limits()).unwrap();
    let model = a.device_model().clone();
    assert_eq!(model.layer_ids().collect::<Vec<_>>(), ids);
    assert_eq!(model.source().root, source.root);
    assert!(std::ptr::eq(model.snapshot(), a.snapshot()));

    let other_model = model.clone();
    // Construct B on another stream, then MOVE its owner back. append/reset
    // must still use B's stream; fixed weights resolve from their origin stream.
    let mut b = std::thread::spawn(move || {
        Session::from_device_model(other_model, limits()).unwrap()
    }).join().unwrap();
    assert_ne!(a.stream, b.stream);
    assert!(Arc::ptr_eq(a.device_model(), b.device_model()));
    for (left, right) in a.layers.iter().zip(&b.layers) {
        match (&left.mixer, &right.mixer) {
            (Mixer::Mamba(x), Mixer::Mamba(y)) => assert!(Arc::ptr_eq(&x.weights, &y.weights)),
            (Mixer::Attention(x), Mixer::Attention(y)) => assert!(Arc::ptr_eq(&x.weights, &y.weights)),
            // One expert bank, in the shared weights, for both Sessions.
            (Mixer::Moe(x), Mixer::Moe(y)) => assert!(Arc::ptr_eq(&x.weights, &y.weights)),
            _ => panic!("shared model changed layer kind"),
        }
    }
    // B's first forward consumes weights whose upload completed on A's stream,
    // before A has executed any chunk.
    let cold_b1 = b.append_ids(&[5]).unwrap().values.to_vec();
    // Exact comparisons below use the SAME append partitions as these isolated
    // reference schedules; no stronger partition/numerical parity is asserted.
    let expected_a2 = a.append_ids(&[1, 5]).unwrap().values.to_vec();
    let expected_a3 = a.append_ids(&[7]).unwrap().values.to_vec();
    a.reset().unwrap();
    let expected_b1 = a.append_ids(&[5]).unwrap().values.to_vec();
    let expected_b2 = a.append_ids(&[3]).unwrap().values.to_vec();
    a.reset().unwrap();
    assert_eq!(cold_b1, expected_b1);
    b.reset().unwrap();
    assert_eq!(b.append_ids(&[5]).unwrap().values, expected_b1);
    assert_eq!(a.append_ids(&[1, 5]).unwrap().values, expected_a2);
    let a_state = state_bytes(&a);
    assert_eq!(b.append_ids(&[3]).unwrap().values, expected_b2);
    assert_eq!(state_bytes(&a), a_state);
    assert_eq!(a.committed_ids(), [1, 5]);
    assert_eq!(a.append_ids(&[7]).unwrap().values, expected_a3);
    let b_state = state_bytes(&b);
    a.reset().unwrap();
    assert_eq!(state_bytes(&b), b_state);
    assert_eq!(b.committed_ids(), [5, 3]);

    // This fault precedes the fixture's MoE layer. It covers retained outer
    // outputs/Mamba staged state/attention state, NOT failed MoE completion.
    a.fault = Some(Fault::AfterLayer(1));
    assert!(a.append_ids(&[1]).is_err());
    let held = a.pending.len();
    assert!(held > 0);
    a.fault = Some(Fault::Reset);
    assert!(a.reset().is_err());
    assert_eq!(a.pending.len(), held);
    assert_eq!(a.status(), Status::Poisoned);
    assert_eq!(b.status(), Status::Ready);
    assert_eq!(state_bytes(&b), b_state);
    a.fault = None;
    a.reset().unwrap();

    // A moves threads as well. The panic scope must restore that caller's
    // stream while preserving pending allocations until a successful reset.
    let mut a = std::thread::spawn(move || {
        let caller = StreamId::current();
        assert_ne!(caller, a.stream);
        a.fault = Some(Fault::PanicAfterLayer(1));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = a.append_ids(&[1]);
        })).is_err());
        assert_eq!(StreamId::current(), caller);
        assert_eq!(a.status(), Status::Poisoned);
        assert!(!a.pending.is_empty());
        assert!(a.pending.iter().all(|tensor| tensor.handle.stream == a.stream));
        a.fault = None;
        a.reset().unwrap();
        assert_eq!(StreamId::current(), caller);
        a
    }).join().unwrap();
    assert_eq!(state_bytes(&b), b_state);
    a.append_ids(&[1, 5]).unwrap();
    assert_eq!(a.append_ids(&[7]).unwrap().values, expected_a3);
    let weak = Arc::downgrade(&model);
    drop(model);
    drop(a); // The original loader/session does not own B's lifetime.
    assert!(weak.upgrade().is_some());
    b.reset().unwrap();
    assert_eq!(b.append_ids(&[5]).unwrap().values, expected_b1);
    assert_eq!(b.append_ids(&[3]).unwrap().values, expected_b2);
    drop(b);
    assert!(weak.upgrade().is_none());
}

fn meminfo_kib(field: &str) -> usize {
    let info = std::fs::read_to_string("/proc/meminfo").unwrap();
    let line = info.lines().find(|l| l.starts_with(field) && l[field.len()..].starts_with(':')).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

#[test]
#[ignore = "reserved CUDA: cuMemGetInfo against /proc/meminfo, with one probe allocation beyond the driver's free figure"]
fn session_gpu_preflight_free_memory_measurement() {
    use cudarc::driver::{result, sys};
    let device = CudaDevice { index: 0 };
    result::init().unwrap();
    let snapshot = || {
        // SAFETY: valid device; the primary context is current only for the query.
        let (free, total) = unsafe {
            let handle = result::device::get(0).unwrap();
            let previous = result::ctx::get_current().unwrap();
            result::ctx::set_current(result::primary_ctx::retain(handle).unwrap()).unwrap();
            let info = result::mem_get_info().unwrap();
            result::ctx::set_current(previous.unwrap_or(std::ptr::null_mut())).unwrap();
            result::primary_ctx::release(handle).unwrap();
            info
        };
        json!({"cu_free": free, "cu_total": total, "allocatable": source::allocatable_bytes(&device).unwrap().0,
            "mem_total": meminfo_kib("MemTotal") * 1024, "mem_free": meminfo_kib("MemFree") * 1024,
            "mem_available": meminfo_kib("MemAvailable") * 1024,
            "cached": meminfo_kib("Cached") * 1024, "buffers": meminfo_kib("Buffers") * 1024})
    };
    let before = snapshot();
    result::init().unwrap();
    let handle = result::device::get(0).unwrap();
    // SAFETY: valid device; the primary context is retained for this test and
    // every allocation is freed before it is released.
    let (integrated, probe) = unsafe {
        let integrated = result::device::get_attribute(handle,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED).unwrap();
        let context = result::primary_ctx::retain(handle).unwrap();
        result::ctx::set_current(context).unwrap();
        // Allocate 4 GiB past the driver's free figure, one GiB at a time,
        // only while the host keeps at least 40 GiB available afterwards.
        let free = before["cu_free"].as_u64().unwrap() as usize;
        let available = before["mem_available"].as_u64().unwrap() as usize;
        let target = free + (4 << 30);
        let mut probe = json!({"attempted": false});
        if available > target + (40 << 30) {
            let mut held = Vec::new();
            let mut error = None;
            while held.len() << 30 < target {
                match result::malloc_sync(1 << 30) {
                    Ok(ptr) => held.push(ptr),
                    Err(e) => { error = Some(e.to_string()); break; }
                }
            }
            let during = snapshot();
            for ptr in held.iter().copied() {
                result::free_sync(ptr).unwrap();
            }
            probe = json!({"attempted": true, "target_bytes": target,
                "allocated_bytes": held.len() << 30, "error": error, "during": during});
        }
        result::primary_ctx::release(handle).unwrap();
        (integrated, probe)
    };
    println!("PREFLIGHT {}", json!({"integrated": integrated, "before": before,
        "probe": probe, "after": snapshot()}));
    // What the preflight relies on: on an integrated device, allocation
    // succeeds past the driver's free figure, up to what the host can reclaim.
    if integrated != 0 && probe["attempted"] == true {
        assert!(probe["error"].is_null(), "{probe}");
        assert!(probe["allocated_bytes"].as_u64().unwrap() > before["cu_free"].as_u64().unwrap());
    }
}

#[test]
fn session_host_preflight_refuses_past_allocatable_with_its_source() {
    source::fits(80_293_122_368, (80_293_122_368, "x")).unwrap();
    let error = source::fits(80_293_122_368, (80_293_122_367, "host MemAvailable (integrated GPU)")).unwrap_err();
    assert!(error.contains("needs 80293122368 bytes"), "{error}");
    assert!(error.contains("only 80293122367 bytes are allocatable, per host MemAvailable"), "{error}");
}

#[test]
fn session_host_physical_requirement_at_1m() {
    // full_config is 40 Mamba, then 40 MoE, then 8 attention layers.
    let c = full_config(1 << 20);
    let l = Limits { capacity: 1 << 20, max_resident_bytes: 80_293_122_368,
        max_chunk_bytes: 15_132_789_760 };
    // Largest upload window: embedding, head and final norm 1_610_620_928,
    // 40 x (norm 8_192 + Mamba weights 109_696_512), then the first MoE layer
    // (norm 8_192 + fixed 45_746_184 + bank 1_585_446_912).
    assert_eq!(resident_payload(&c, l).unwrap().window, 7_630_010_376);
    assert_eq!(transient_bytes(&c, l).unwrap(), 15_132_789_760);
    // 80_293_122_368 resident + 15_132_789_760 span + 7_630_010_376 window.
    assert_eq!(load_requirement(&c, l, false).unwrap(), 103_055_922_504);
    // A seed keeps a sealed boundary: 40 x 4_276_224 Mamba state.
    assert_eq!(load_requirement(&c, l, true).unwrap(), 103_226_971_464);
    // A further Session: KV 8_589_934_592 + Mamba 171_048_960 + span.
    assert_eq!(session_requirement(&c, l, false).unwrap(), 23_893_773_312);
    // A smaller allowance caps the working set it admits.
    let small = Limits { max_chunk_bytes: 6 << 30, ..l };
    assert_eq!(transient_bytes(&c, small).unwrap(), 6 << 30);
}

fn rss_anon_bytes() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("RssAnon:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse::<usize>().unwrap() * 1024
}

#[test]
#[ignore = "reserved CUDA: a load refused at the last expert gives back its device pages and host copies"]
fn session_gpu_refused_load_releases_memory() {
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    cubecl::future::block_on(client.sync()).unwrap();
    // Built before the baseline: the source itself is not what is measured.
    let (snapshot, source, ids) = fixture_experts(8, 511);
    let (reserved, rss) = (client.memory_usage().unwrap().bytes_reserved, rss_anon_bytes());
    let error = Session::from_snapshot(snapshot, source, &ids, device.clone(), limits())
        .err().expect("an incomplete bank must refuse the load");
    assert!(error.contains("missing complete native expert 511"), "{error}");
    let after = (client.memory_usage().unwrap().bytes_reserved, rss_anon_bytes());
    // Contrast: a dropped exact buffer stays reserved until a cleanup.
    drop(moe::exact(&client, || client.empty(64 << 20)));
    let dropped = client.memory_usage().unwrap().bytes_reserved;
    client.memory_cleanup();
    let cleaned = client.memory_usage().unwrap().bytes_reserved;
    println!("REFUSED_LOAD {}", json!({"error": error,
        "device_reserved_before": reserved, "device_reserved_after": after.0,
        "rss_anon_before": rss, "rss_anon_after": after.1,
        "contrast_64MiB_dropped_reserved": dropped, "contrast_after_cleanup": cleaned}));
    assert!(dropped >= reserved + (64 << 20) && cleaned <= reserved, "{dropped} {cleaned}");
    // 1.7 GB had been uploaded when the 512th expert was found missing.
    assert!(after.0 <= reserved + (64 << 20), "device pages kept: {reserved} -> {}", after.0);
    assert!(after.1 <= rss + (256 << 20), "host copies kept: {rss} -> {}", after.1);
}

#[test]
#[ignore = "reserved CUDA, default memory configuration: model buffers allocated at their own size"]
fn session_gpu_physical_matches_logical() {
    let device = CudaDevice { index: 0 };
    // A fresh thread is a fresh stream with its own empty pools.
    let reserve = |exact: bool, bytes: usize| {
        let device = device.clone();
        std::thread::spawn(move || {
            let client = CudaRuntime::client(&device);
            let base = client.memory_usage().unwrap().bytes_reserved;
            let handle = if exact { moe::exact(&client, || client.empty(bytes)) } else { client.empty(bytes) };
            let grown = client.memory_usage().unwrap().bytes_reserved - base;
            drop(handle);
            client.memory_cleanup();
            cubecl::future::block_on(client.sync()).unwrap();
            grown
        }).join().unwrap()
    };
    // A 600 MiB buffer, the size class of a 1M KV plane or the embedding.
    let pooled = reserve(false, 600 << 20);
    let exact = reserve(true, 600 << 20);
    let (logical, physical) = std::thread::spawn(move || {
        let (snapshot, source, ids) = fixture();
        let session = Session::from_snapshot(snapshot, source, &ids, device, limits()).unwrap();
        let usage = session.model.client.memory_usage().unwrap();
        let logical = resident_bytes(&config(), limits()).unwrap();
        drop(session);
        (logical, usage)
    }).join().unwrap();
    println!("PHYSICAL {}", json!({
        "memory_config": std::env::var("CUBECL_MEMORY_CONFIG").ok(),
        "buffer_600MiB": {"logical": 600usize << 20, "default_pool_reserved": pooled, "exact_reserved": exact},
        "synthetic_model": {"logical_resident": logical, "reserved": physical.bytes_reserved,
            "in_use": physical.bytes_in_use, "padding": physical.bytes_padding,
            "allocations": physical.number_allocs},
    }));
    assert!(exact < (600 << 20) + (1 << 20), "{exact}");
    assert!(physical.bytes_reserved < logical as u64 + (logical as u64) / 50, "{physical:?} vs {logical}");
}
