//! Host-only: synthetic marks and CUPTI-shaped records, no device or library.
use super::*;

fn k(start: u64, end: u64, name: &str) -> Activity {
    Activity { start, end, what: What::Kernel(name.into()) }
}
fn copy(start: u64, end: u64, kind: u8, bytes: u64) -> Activity {
    Activity { start, end, what: What::Memcpy { kind, bytes } }
}
fn m(ts: u64, layer: usize, phase: &'static str) -> Mark {
    Mark { ts, layer, phase }
}

fn stage_row<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["stages"].as_array().unwrap().iter().find(|r| r["stage"] == name).unwrap()
}
fn ctx_row<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["idle_by_context"].as_array().unwrap().iter().find(|r| r["context"] == name).unwrap()
}

/// One Mamba, one attention and one MoE layer, every tile fenced and marked
/// the way run_prefill/prefill_moe observe progress, plus the lm-head tail.
fn layer_major() -> (Vec<Mark>, Vec<Kind>, Vec<Activity>) {
    let kinds = vec![Kind::Mamba, Kind::Attention, Kind::Moe];
    let marks = vec![
        m(100, 0, "embedding"), m(200, 0, "stateful"), m(300, 1, "stateful"),
        m(400, 2, "route"), m(450, 2, "accumulator_zero"), m(550, 2, "expert_rows"),
        m(560, 2, "expert_rows"), m(650, 2, "finish"),
    ];
    let acts = vec![
        // embedding
        k(10, 30, "embedding_kernel"), k(30, 40, "copy_rows_kernel_bf16"),
        // mamba
        k(110, 120, "norm_kernel"), k(120, 140, "matmul_entry_f16"), k(140, 150, "convolution"),
        k(150, 170, "recurrence"), k(170, 180, "matmul_entry_f16"), k(180, 185, "add_kernel"),
        k(185, 190, "copy_rows_kernel_bf16"),
        // attention
        k(205, 210, "norm_kernel"), k(210, 230, "quantize"), k(230, 235, "append_tail"),
        k(235, 265, "mma_attention_kernel"), k(265, 270, "flash_combine_kernel"),
        k(270, 280, "finish"), k(280, 290, "add_kernel"),
        // route: router linear, route kernel, readback, then the fc1 latent
        k(310, 315, "norm_kernel"), k(315, 320, "pad_bf16"), k(320, 330, "bf16_linear_bf16"),
        k(330, 335, "route_kernel"), copy(335, 336, 2, 4096), k(340, 350, "bf16_linear_bf16"),
        k(350, 352, "to_bf16"), k(352, 355, "copy_rows_kernel_bf16"),
        // accumulator zero
        k(410, 420, "zero_rows"),
        // one expert: index and routing-weight uploads, gather, FP4 GEMMs, scatter
        copy(460, 480, 1, 1 << 20), copy(481, 482, 1, 8), k(483, 485, "gather_grouped_kernel_bf16_bf16"),
        k(485, 487, "quantize_kernel_bf16"), k(487, 500, "nvfp4_gemm"), k(500, 502, "relu2_kernel"),
        k(502, 510, "nvfp4_gemm"), k(510, 512, "ordered_update"),
        // the forced final expert_rows observation leaves an empty window
        // finish: fc2 latent, shared expert, combine, finite readback, residual
        k(570, 575, "norm_kernel"), k(575, 577, "copy_rows_kernel_f32"), k(577, 578, "to_bf16"),
        k(578, 579, "pad_bf16"), k(579, 590, "bf16_linear_bf16"), k(590, 591, "to_bf16"),
        k(591, 600, "nvfp4_gemm"), k(600, 601, "relu2_kernel"), k(601, 610, "nvfp4_gemm"),
        k(610, 612, "sum_kernel"), copy(612, 613, 2, 16), k(613, 615, "add_kernel"),
        k(615, 618, "copy_rows_kernel_bf16"),
        // tail: final norm, last row, head projection, logits readback
        k(655, 660, "norm_kernel"), k(660, 661, "last_row_kernel"), k(661, 670, "quantize"),
        copy(670, 672, 2, 262144),
        // after the bracket: never attributed
        k(800, 810, "norm_kernel"),
    ];
    (marks, kinds, acts)
}

#[test]
fn layer_major_aggregation_is_exact() {
    let (marks, kinds, acts) = layer_major();
    let r = analyze(&Input { start: 0, end: 700, marks: &marks, kinds: &kinds, activities: &acts });
    let expect = [
        ("embedding", 30, 2), ("mamba_projections", 30, 2), ("mamba_scan", 30, 2),
        ("attention_projections", 30, 2), ("attention_kernel", 35, 2), ("attention_kv_write", 5, 1),
        ("moe_router", 21, 4), ("moe_latent_projections", 26, 6),
        ("expert_dispatch_gather_scatter", 14, 3), ("expert_tile_index_h2d", 21, 2),
        ("expert_gemms", 25, 4), ("shared_expert", 19, 3), ("moe_combine", 3, 2),
        ("norms", 25, 4), ("residuals_and_copies", 30, 7), ("final_norm_lm_head", 17, 4),
        ("transfers_other", 0, 0), ("unclassified", 0, 0),
    ];
    for (name, ns, calls) in expect {
        let row = stage_row(&r, name);
        assert_eq!((row["device_ns"].as_u64().unwrap(), row["calls"].as_u64().unwrap()), (ns, calls), "{name}");
    }
    assert_eq!(stage_row(&r, "attention_kernel")["mean_ns"], 17);
    assert!((stage_row(&r, "attention_kernel")["share"].as_f64().unwrap() - 35.0 / 700.0).abs() < 1e-12);
    assert_eq!(r["bracket_ns"], 700);
    assert_eq!(r["windows"], 9);
    assert_eq!(r["device_busy_ns"], 361);
    assert_eq!(r["device_overlap_ns"], 0);
    assert_eq!(r["device_idle_ns"], 339);
    assert_eq!(r["accounted_ns"], 700);
    assert_eq!((r["records_outside_bracket"].as_u64(), r["outside_bracket_ns"].as_u64()), (Some(1), Some(10)));
    assert_eq!(r["records_straddling_window_end"], 0);
    // Idle split: device idle before/between work vs after the last work.
    for (ctx, windows, busy, gap, trail) in [
        ("embedding", 1, 30, 10, 60), ("mamba", 1, 80, 10, 10), ("attention", 1, 85, 5, 10),
        ("moe_route", 1, 41, 14, 45), ("moe_accumulator_zero", 1, 10, 10, 30),
        ("moe_experts", 2, 50, 22, 38), ("moe_finish", 1, 48, 10, 32),
        ("lm_head_and_commit", 1, 17, 5, 28), ("short_append", 0, 0, 0, 0),
    ] {
        let row = ctx_row(&r, ctx);
        assert_eq!([&row["windows"], &row["device_busy_ns"], &row["host_gap_ns"], &row["commit_flush_ns"]]
            .map(|v| v.as_u64().unwrap()), [windows, busy, gap, trail], "{ctx}");
    }
    let layer1 = r["per_layer"].as_array().unwrap().iter().find(|l| l["layer"] == 1).unwrap();
    assert_eq!((layer1["device_busy_ns"].as_u64(), layer1["device_idle_ns"].as_u64()), (Some(85), Some(15)));
    // Layer 0 is its Mamba window only; the embedding window is no layer's.
    let layer0 = r["per_layer"].as_array().unwrap().iter().find(|l| l["layer"] == 0).unwrap();
    assert_eq!((layer0["device_busy_ns"].as_u64(), layer0["device_idle_ns"].as_u64()), (Some(80), Some(20)));
}

// ---- review fixes (2026-11-07): forced repeats, per-layer, seeded spans ----

#[test]
fn a_forced_repeat_observation_is_not_a_fenced_tile() {
    set_mode(Some(Mode::Bracket));
    let mut probe = Probe::requested(4, 0, || vec![Kind::Moe]).unwrap().unwrap();
    probe.start(std::time::Duration::ZERO).unwrap();
    mark(0, "expert_rows", 8);
    mark(0, "expert_rows", 16);
    // prefill_moe's forced final observation after the expert loop: no fence.
    mark(0, "expert_rows", 16);
    mark(0, "finish", 4);
    probe.finish("completed", 4, true);
    set_mode(None);
    assert_eq!(take_report().unwrap()["fenced_tiles"], 3);
}

/// A seeded append runs two spans in one bracket. Span 1's LM head, logits
/// readback and commit end at its "lm_head" mark, not at span 2's embedding.
#[test]
fn seeded_first_span_tail_is_lm_head_not_embedding() {
    let kinds = [Kind::Moe];
    let marks = [m(100, 0, "embedding"), m(200, 0, "finish"), m(300, 1, "lm_head"),
        m(400, 0, "embedding"), m(500, 0, "finish")];
    let acts = vec![
        k(10, 20, "embedding_kernel"),
        k(210, 215, "norm_kernel"), k(215, 216, "last_row_kernel"), k(216, 230, "quantize"),
        copy(230, 232, 2, 262144),
        k(350, 360, "embedding_kernel"),
        k(510, 515, "norm_kernel"),
    ];
    let r = analyze(&Input { start: 0, end: 600, marks: &marks, kinds: &kinds, activities: &acts });
    let tail = stage_row(&r, "final_norm_lm_head");
    assert_eq!((tail["device_ns"].as_u64(), tail["calls"].as_u64()), (Some(27), Some(5)));
    assert_eq!(stage_row(&r, "embedding")["device_ns"], 20);
    assert_eq!(stage_row(&r, "norms")["calls"], 0, "span 1's final norm is not a layer norm");
    let tail_ctx = ctx_row(&r, "lm_head_and_commit");
    assert_eq!((tail_ctx["windows"].as_u64(), tail_ctx["device_busy_ns"].as_u64()), (Some(2), Some(27)));
    let embedding_ctx = ctx_row(&r, "embedding");
    assert_eq!((embedding_ctx["windows"].as_u64(), embedding_ctx["device_busy_ns"].as_u64()), (Some(2), Some(20)));
    let layers: Vec<u64> = r["per_layer"].as_array().unwrap().iter().map(|l| l["layer"].as_u64().unwrap()).collect();
    assert_eq!(layers, [0], "neither the embedding nor the lm_head window is a layer");
}

#[test]
fn short_append_without_marks_keeps_unknown_work_visible() {
    let acts = vec![k(1, 3, "norm_kernel"), k(3, 9, "matmul_entry_f16"), k(9, 10, "mma_attention_kernel"),
        Activity { start: 10, end: 12, what: What::Memset { bytes: 64 } }];
    let r = analyze(&Input { start: 0, end: 20, marks: &[], kinds: &[], activities: &acts });
    assert_eq!(r["windows"], 1);
    assert_eq!(stage_row(&r, "norms")["device_ns"], 2);
    assert_eq!(stage_row(&r, "unclassified")["device_ns"], 6);
    assert_eq!(stage_row(&r, "attention_kernel")["device_ns"], 1);
    assert_eq!(stage_row(&r, "transfers_other")["device_ns"], 2);
    let short = ctx_row(&r, "short_append");
    assert_eq!((short["host_gap_ns"].as_u64(), short["commit_flush_ns"].as_u64()), (Some(1), Some(8)));
    assert_eq!(r["accounted_ns"], 20);
}

#[test]
fn overlap_and_straddling_are_reported_not_double_counted() {
    let marks = [m(10, 0, "embedding")];
    let acts = vec![k(1, 6, "embedding_kernel"), k(4, 8, "embedding_kernel"), k(9, 12, "embedding_kernel")];
    let r = analyze(&Input { start: 0, end: 20, marks: &marks, kinds: &[Kind::Mamba], activities: &acts });
    assert_eq!(r["device_busy_sum_ns"], 12);
    assert_eq!(r["device_busy_ns"], 8); // union, clipped at the window end
    assert_eq!(r["device_overlap_ns"], 4);
    assert_eq!(r["records_straddling_window_end"], 1);
    assert_eq!(r["accounted_ns"], 20);
}

#[test]
fn table_prints_every_nonzero_stage_and_the_bracket() {
    let (marks, kinds, acts) = layer_major();
    let device = analyze(&Input { start: 0, end: 700, marks: &marks, kinds: &kinds, activities: &acts });
    let report = json!({"mode": "device", "outcome": "completed", "tokens": 7, "committed_before": 0,
        "committed_after": 7, "fenced_tiles": 8, "pre_sync_ns": 5, "bracket_wall_ns": 700_000_000u64,
        "tokens_per_second": 10.0, "device": device});
    let text = table(&report);
    assert!(text.contains("bracket wall 700.000 ms (10.0 tokens/s)"), "{text}");
    for name in ["mamba_scan", "attention_kernel", "expert_gemms", "shared_expert", "final_norm_lm_head",
        "host gap: moe_experts", "commit/flush: lm_head_and_commit"] {
        assert!(text.contains(name), "{name} missing\n{text}");
    }
    assert!(!text.contains("unclassified"), "zero rows are omitted\n{text}");
    let line = text.lines().find(|l| l.starts_with("attention_kernel")).unwrap();
    let cols: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(cols, ["attention_kernel", "0.000", "5.00%", "2", "0.02"], "{line}"); // 35 of 700 ns, 2 calls
    assert!(table(&json!({"mode":"bracket","outcome":"completed","bracket_wall_ns":1})).lines().count() == 2);
}

#[test]
fn mode_override_and_reentrancy_guard_without_device() {
    set_mode(None);
    assert!(Probe::requested(4, 0, || Vec::new()).is_none());
    set_mode(Some(Mode::Bracket));
    let mut probe = Probe::requested(4, 2, || vec![Kind::Mamba]).unwrap().unwrap();
    assert!(Probe::requested(4, 2, || Vec::new()).is_none(), "nested call does not bracket again");
    probe.start(std::time::Duration::from_nanos(3)).unwrap();
    mark(0, "embedding", 4);
    mark(0, "stateful", 4);
    probe.finish("completed", 6, true);
    let report = take_report().unwrap();
    assert_eq!(report["mode"], "bracket");
    assert_eq!(report["fenced_tiles"], 2);
    assert_eq!((report["tokens"].as_u64(), report["committed_after"].as_u64()), (Some(4), Some(6)));
    assert_eq!(report["pre_sync_ns"], 3);
    assert!(report["device"].is_null());
    // The guard closed with the bracket; a dropped probe also releases it.
    drop(Probe::requested(1, 0, || Vec::new()).unwrap().unwrap());
    assert!(Probe::requested(1, 0, || Vec::new()).is_some());
    set_mode(None);
    assert_eq!(Mode::parse(" device "), Some(Mode::Device));
    assert_eq!(Mode::parse("on"), None);
}

#[test]
fn record_offsets_match_the_compiled_cupti_bindings() {
    use cudarc::cupti::sys as c;
    use std::mem::offset_of;
    assert_eq!(offset_of!(c::CUpti_ActivityKernel10, start), cupti::KERNEL_START);
    assert_eq!(offset_of!(c::CUpti_ActivityKernel10, end), cupti::KERNEL_END);
    assert_eq!(offset_of!(c::CUpti_ActivityKernel10, name), cupti::KERNEL_NAME);
    assert_eq!(offset_of!(c::CUpti_ActivityMemcpy6, copyKind), cupti::COPY_KIND);
    assert_eq!(offset_of!(c::CUpti_ActivityMemcpy6, bytes), cupti::COPY_BYTES);
    assert_eq!(offset_of!(c::CUpti_ActivityMemcpy6, start), cupti::COPY_START);
    assert_eq!(offset_of!(c::CUpti_ActivityMemcpy6, end), cupti::COPY_END);
    assert_eq!(offset_of!(c::CUpti_ActivityMemset4, bytes), cupti::COPY_BYTES);
    assert_eq!(offset_of!(c::CUpti_ActivityMemset4, start), cupti::COPY_START);
    assert_eq!(offset_of!(c::CUpti_ActivityMemset4, end), cupti::COPY_END);
}

// ---- fix1 (2026-11-07): profiler failures, (name, stage) rows ----

fn capture(dropped: Result<u64, String>, errors: Vec<String>) -> Capture {
    let (_, _, activities) = layer_major();
    Capture { activities, dropped, errors, version: Ok(130001) }
}

fn device_table(device: Value) -> String {
    table(&json!({"mode": "device", "outcome": "completed", "tokens": 7, "bracket_wall_ns": 700,
        "device": device}))
}

#[test]
fn shared_kernel_name_in_two_stages_gives_two_rows() {
    let (marks, kinds, acts) = layer_major();
    let r = analyze(&Input { start: 0, end: 700, marks: &marks, kinds: &kinds, activities: &acts });
    let rows: Vec<(String, String, u64, u64)> = r["top_kernels"].as_array().unwrap().iter().map(|t| (
        t["kernel"].as_str().unwrap().to_owned(), t["stage"].as_str().unwrap().to_owned(),
        t["device_ns"].as_u64().unwrap(), t["calls"].as_u64().unwrap())).collect();
    let find = |name: &str, stage: &str| rows.iter().find(|r| r.0 == name && r.1 == stage).map(|r| (r.2, r.3));
    // nvfp4_gemm: 2 expert GEMMs (13 + 8) and 2 shared-expert projections (9 + 9).
    assert_eq!(find("nvfp4_gemm", "expert_gemms"), Some((21, 2)));
    assert_eq!(find("nvfp4_gemm", "shared_expert"), Some((18, 2)));
    // bf16_linear: the router (before route_kernel) and two latent projections.
    assert_eq!(find("bf16_linear_bf16", "moe_router"), Some((10, 1)));
    assert_eq!(find("bf16_linear_bf16", "moe_latent_projections"), Some((21, 2)));
    // Each (name, stage) row agrees with the stage table: every top row's
    // stage total is at least that row.
    for (name, stage, ns, _) in &rows {
        assert!(stage_row(&r, stage)["device_ns"].as_u64().unwrap() >= *ns, "{name} {stage}");
    }
    assert_eq!((rows[0].0.as_str(), rows[0].2), ("matmul_entry_f16", 30));
}

#[test]
fn failed_timestamps_make_attribution_unknown_never_zero() {
    let (marks, kinds, _) = layer_major();
    let failed = "cuptiGetTimestamp: CUPTI_ERROR_INVALID_PARAMETER".to_string();
    for (start, end, mark_errors) in [
        (Err(failed.clone()), Ok(700), vec![]),
        (Ok(0), Err(failed.clone()), vec![]),
        (Ok(0), Ok(700), vec![failed.clone()]),
    ] {
        let d = device_report(start, end, &mark_errors, &marks, &kinds, Ok(capture(Ok(0), vec![])));
        let error = d["error"].as_str().unwrap();
        assert!(error.starts_with("profiler state unknown") && error.contains("CUPTI_ERROR_INVALID_PARAMETER"), "{error}");
        assert!(d["stages"].is_null() && d["bracket_ns"].is_null(), "no stages from a zero clock: {d}");
        let text = device_table(d);
        assert!(text.contains("device attribution FAILED: profiler state unknown"), "{text}");
        assert!(!text.contains("device stage"), "{text}");
    }
    // A failed CUPTI call inside capture (flush, disable, record iteration).
    let d = device_report(Ok(0), Ok(700), &[], &marks, &kinds,
        Ok(capture(Ok(0), vec!["cuptiActivityFlushAll: CUPTI_ERROR_UNKNOWN".into()])));
    assert!(d["error"].as_str().unwrap().contains("cuptiActivityFlushAll"));
    let d = device_report(Ok(0), Ok(700), &[], &marks, &kinds, Err("CUPTI record store poisoned".into()));
    assert!(d["error"].as_str().unwrap().contains("poisoned") && d["cupti_dropped_records"].is_null());
}

#[test]
fn failed_dropped_query_is_unknown_not_zero() {
    let mut dropped = Dropped::new();
    dropped.note(Ok(3));
    dropped.note(Ok(2));
    assert_eq!(dropped.result(), Ok(5));
    dropped.note(Err("cuptiActivityGetNumDroppedRecords: CUPTI_ERROR_NOT_INITIALIZED".into()));
    dropped.note(Ok(1));
    let unknown = dropped.result().unwrap_err();
    assert!(unknown.contains("CUPTI_ERROR_NOT_INITIALIZED"));
    let (marks, kinds, _) = layer_major();
    let d = device_report(Ok(0), Ok(700), &[], &marks, &kinds, Ok(capture(Err(unknown), vec![])));
    assert!(d["error"].is_null(), "timings stay valid: {d}");
    assert!(d["cupti_dropped_records"].is_null());
    assert!(d["cupti_dropped_error"].as_str().unwrap().contains("NOT_INITIALIZED"));
    assert_eq!(d["accounted_ns"], 700);
    let text = device_table(d);
    assert!(text.contains("CUPTI dropped UNKNOWN (cuptiActivityGetNumDroppedRecords: CUPTI_ERROR_NOT_INITIALIZED)"), "{text}");
    assert!(!text.contains("CUPTI dropped 0"), "{text}");
    let d = device_report(Ok(0), Ok(700), &[], &marks, &kinds, Ok(capture(Ok(4), vec![])));
    assert!(device_table(d).contains("CUPTI dropped 4)"));
}

#[test]
fn partial_enable_failure_disables_exactly_what_was_enabled() {
    use std::cell::RefCell;
    let calls = RefCell::new(Vec::new());
    // The second enable fails: only the first kind may be rolled back.
    let result = enable_exactly(&["kernel", "memcpy", "memset"],
        |k| { calls.borrow_mut().push(format!("enable {k}")); if k == "memcpy" { Err("enable memcpy: CUPTI_ERROR_UNKNOWN".into()) } else { Ok(()) } },
        |k| { calls.borrow_mut().push(format!("disable {k}")); Ok(()) });
    let error = result.unwrap_err();
    assert_eq!(*calls.borrow(), ["enable kernel", "enable memcpy", "disable kernel"]);
    assert_eq!(error, "enable memcpy: CUPTI_ERROR_UNKNOWN; rolled back [\"kernel\": disabled]");
    // A rollback disable that itself fails keeps its own status.
    let error = enable_exactly(&["kernel", "memcpy", "memset"],
        |k| if k == "memset" { Err("enable memset: E".into()) } else { Ok(()) },
        |k| if k == "kernel" { Err("disable kernel: F".into()) } else { Ok(()) }).unwrap_err();
    assert_eq!(error, "enable memset: E; rolled back [\"memcpy\": disabled, \"kernel\": disable kernel: F]");
    // Success returns exactly the enabled kinds, and the cleanup disables
    // them newest first, one status each.
    let enabled = enable_exactly(&[1, 2, 3], |_| Ok(()), |_| Ok(())).unwrap();
    assert_eq!(enabled, [1, 2, 3]);
    let statuses = disable_exactly(enabled, |k| if k == 2 { Err("disable 2: G".into()) } else { Ok(()) });
    assert_eq!(statuses, [(3, Ok(())), (2, Err("disable 2: G".to_string())), (1, Ok(()))]);
    assert!(disable_exactly(Vec::<u8>::new(), |_| panic!("nothing to disable")).is_empty());
}

// ---- expert-tile host spans (re-ported onto mary 1a4ee30) ----

/// One bracket that drives the expert-tile spans the way prefill_moe does,
/// then returns its report.
fn host_pass(tiles: usize) -> Value {
    set_mode(Some(Mode::Bracket));
    let mut probe = Probe::requested(4, 0, || vec![Kind::Moe]).unwrap().unwrap();
    probe.start(std::time::Duration::ZERO).unwrap();
    for _ in 0..tiles {
        HostSpan::enqueue().done(true);
        HostSpan::fence().done(true);
    }
    drop(HostSpan::fence()); // an early `?` return: counted, not succeeded
    probe.finish("completed", 4, true);
    set_mode(None);
    take_report().unwrap()
}

#[test]
fn host_phases_aggregate_per_pass_and_retired_metrics_are_absent() {
    let r = host_pass(3);
    let h = &r["host_phases"];
    assert_eq!(h["expert_tile_enqueue"]["calls"], 3);
    assert_eq!((h["expert_tile_fence"]["calls"].as_u64(), h["expert_tile_fence"]["succeeded"].as_u64()), (Some(4), Some(3)));
    // Retired metrics are named as absent and carry no value, never a zero.
    let absent: Vec<&str> = h["absent"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(absent, ["cache_hits", "cache_misses", "cache_evictions", "lookup_view", "packed_validation", "weight_upload_api"]);
    for key in ["cache", "cache_hits", "cache_misses", "cache_evictions", "lookup_view", "lookup_query_construction",
        "lookup_min_enumeration", "lookup_native_reads", "packed_validation", "weight_upload_api"] {
        assert!(h[key].is_null(), "{key} must be absent, not a value: {h}");
    }
    let text = host_summary(&r);
    for (_, label) in HOST_ROWS {
        assert!(text.contains(label), "{label} missing\n{text}");
    }
    assert!(text.contains("ABSENT, not zero: cache_hits, cache_misses, cache_evictions, lookup_view, packed_validation, weight_upload_api"), "{text}");
    assert!(text.contains("(1 failed)"), "{text}");
    for forbidden in ["remainder", "other", "repack", "total host", "hits 0", "misses 0"] {
        assert!(!text.to_lowercase().contains(forbidden), "{forbidden}\n{text}");
    }
}

#[test]
fn consecutive_runs_snapshot_their_own_counters() {
    let first = host_pass(5);
    let second = host_pass(1);
    assert_eq!(first["host_phases"]["expert_tile_enqueue"]["calls"], 5);
    assert_eq!(second["host_phases"]["expert_tile_enqueue"]["calls"], 1, "not a process-lifetime total");
    assert_eq!(second["host_phases"]["expert_tile_fence"]["calls"], 2);
}

#[test]
fn no_open_bracket_reads_no_clock_and_leaks_nothing() {
    set_mode(None);
    let mut called = false;
    assert!(with_host(|_| called = true).is_none());
    assert!(!called);
    let span = HostSpan::enqueue();
    assert!(span.0.is_none(), "no clock read without a bracket");
    span.done(true);
    drop(HostSpan::fence());
    // A later bracket starts from zero: nothing above leaked into it.
    let r = host_pass(0);
    assert_eq!(r["host_phases"]["expert_tile_enqueue"]["calls"], 0);
    assert_eq!(r["host_phases"]["expert_tile_fence"]["calls"], 1);
    assert!(Probe::requested(1, 0, || Vec::new()).is_none(), "mode is off again");
}
