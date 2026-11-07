//! Opt-in whole-append prefill bracket, with optional device attribution.
//!
//! Compiled only with `nemotron-prefill-timing`; inert unless a mode is asked
//! for, by [`set_mode`] on the calling thread or `MARY_NEMOTRON_PREFILL_TIMING`
//! (`bracket` or `device`). With no mode, `Session::prefill_ids` never enters
//! this module beyond one thread-local and one environment read.
//!
//! The bracket opens after a fence drains earlier work (its wait is reported as
//! `pre_sync`, outside the bracket) and closes when `prefill_ids` returns: after
//! the final fence, the logits readback and the Session's own commit. So the
//! wall time IS device-synchronised time for the whole append.
//!
//! `device` mode adds CUPTI activity records (kernels, copies, memsets: device
//! start/end on the CUPTI clock) and one host mark per existing progress
//! observation (`native_prefill_step`'s layer/phase), taken right after that
//! tile's fence. Every tile in layer-major prefill ends in a fence, so each
//! device record falls inside exactly one tile window; within a window, the
//! kernel name decides the stage. Device time a window does not cover with work
//! is idle: host enqueue and the sync return/commit.
//!
//! These are not CUDA events. The CubeCL stream handle is private to its
//! runner thread, so events cannot be recorded on it without patching CubeCL;
//! CUPTI reads the same device timestamps without touching the stream.
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Instant;

use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Two clock reads and a counter per tile. The overhead reference.
    Bracket,
    /// Bracket plus CUPTI activity and per-tile host marks.
    Device,
}

impl Mode {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "bracket" => Some(Self::Bracket),
            "device" => Some(Self::Device),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bracket => "bracket",
            Self::Device => "device",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mamba,
    Attention,
    Moe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mark {
    pub ts: u64,
    pub layer: usize,
    pub phase: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum What {
    Kernel(String),
    /// CUPTI copy kind (1 host-to-device, 2 device-to-host, 8 device-to-device).
    Memcpy { kind: u8, bytes: u64 },
    Memset { bytes: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Activity {
    pub start: u64,
    pub end: u64,
    pub what: What,
}

struct Active {
    mode: Mode,
    tiles: u64,
    /// The last observation: a repeat of it follows no new fence.
    last: Option<(usize, &'static str, usize)>,
    marks: Vec<Mark>,
    /// Failed mark timestamps: the window bounds are then unknown, not zero.
    mark_errors: Vec<String>,
}

/// What CUPTI delivered for one bracket. Each failed CUPTI call is an error
/// here, never a zero.
pub(crate) struct Capture {
    pub activities: Vec<Activity>,
    pub dropped: Result<u64, String>,
    pub errors: Vec<String>,
    pub version: Result<u32, String>,
}

/// Dropped-record count summed over queries, each answering for its own
/// queue (a buffer callback's context and stream). One failed query makes
/// the total unknown for the rest of the bracket.
pub(crate) struct Dropped {
    count: u64,
    errors: Vec<String>,
}

impl Dropped {
    pub(crate) const fn new() -> Self {
        Self { count: 0, errors: Vec::new() }
    }
    pub(crate) fn note(&mut self, query: Result<usize, String>) {
        match query {
            Ok(n) => self.count = self.count.saturating_add(n as u64),
            Err(e) => self.errors.push(e),
        }
    }
    pub(crate) fn result(&self) -> Result<u64, String> {
        if self.errors.is_empty() { Ok(self.count) } else { Err(self.errors.join("; ")) }
    }
}

/// The device half of the report. Any failed clock read or CUPTI call makes
/// the attribution UNKNOWN: an error object, never stages computed from a
/// zero timestamp. A failed dropped-record query alone leaves the stages but
/// reports the dropped count as unknown (null plus the error).
pub(crate) fn device_report(start: Result<u64, String>, end: Result<u64, String>,
    mark_errors: &[String], marks: &[Mark], kinds: &[Kind], capture: Result<Capture, String>) -> Value
{
    let capture = match capture {
        Ok(capture) => capture,
        Err(error) => return json!({"error": format!("profiler state unknown: {error}"),
            "profiler_errors": [error], "cupti_dropped_records": null}),
    };
    let mut errors: Vec<String> = Vec::new();
    if let Err(e) = &start { errors.push(format!("bracket start {e}")); }
    if let Err(e) = &end { errors.push(format!("bracket end {e}")); }
    if !mark_errors.is_empty() {
        errors.push(format!("{} tile mark timestamps failed, first: {}", mark_errors.len(), mark_errors[0]));
    }
    errors.extend(capture.errors.iter().cloned());
    let (dropped, dropped_error) = match &capture.dropped {
        Ok(n) => (json!(n), Value::Null),
        Err(e) => (Value::Null, json!(e)),
    };
    let mut device = match (&start, &end) {
        (Ok(start), Ok(end)) if errors.is_empty() => analyze(&Input {
            start: *start, end: *end, marks, kinds, activities: &capture.activities }),
        _ => json!({"error": format!("profiler state unknown: {}", errors.join("; ")),
            "profiler_errors": errors}),
    };
    device["cupti_dropped_records"] = dropped;
    device["cupti_dropped_error"] = dropped_error;
    match capture.version {
        Ok(v) => device["cupti_version"] = json!(v),
        Err(e) => device["cupti_version_error"] = json!(e),
    }
    device
}

thread_local! {
    static MODE: Cell<Option<Option<Mode>>> = const { Cell::new(None) };
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
    static LAST: RefCell<Option<Value>> = const { RefCell::new(None) };
    /// The expert-tile enqueue and fence spans, fresh per bracket.
    static HOST: RefCell<Option<super::host_timing::Timings>> = const { RefCell::new(None) };
}

/// Choose the mode for this thread, overriding the environment. `None` turns
/// the instrument off for this thread whatever the environment says.
pub fn set_mode(mode: Option<Mode>) {
    MODE.with(|m| m.set(Some(mode)));
}

/// The report of the last bracketed append on this thread (one JSON object).
pub fn take_report() -> Option<Value> {
    LAST.with(|l| l.borrow_mut().take())
}

fn requested() -> Option<Mode> {
    MODE.with(Cell::get).unwrap_or_else(|| {
        std::env::var("MARY_NEMOTRON_PREFILL_TIMING").ok().as_deref().and_then(Mode::parse)
    })
}

/// Called from the existing progress observation, after each tile's fence,
/// and once after the LM head's fence ("lm_head"). An observation repeating
/// the previous (layer, phase, rows) follows no fence (prefill_moe's forced
/// final "expert_rows") and is neither counted nor marked.
pub(crate) fn mark(layer: usize, phase: &'static str, rows: usize) {
    ACTIVE.with(|a| {
        if let Some(active) = a.borrow_mut().as_mut() {
            if active.last.replace((layer, phase, rows)) == Some((layer, phase, rows)) {
                return;
            }
            active.tiles += 1;
            if active.mode == Mode::Device {
                match cupti::now() {
                    Ok(ts) => active.marks.push(Mark { ts, layer, phase }),
                    Err(error) => active.mark_errors.push(error),
                }
            }
        }
    });
}

pub(crate) struct Probe {
    mode: Mode,
    tokens: usize,
    committed_before: usize,
    kinds: Vec<Kind>,
    pre_sync_ns: u64,
    started: Option<(Instant, Result<u64, String>)>,
}

impl Probe {
    /// `None` when no mode is asked for, or inside an already bracketed append
    /// (the seeded path calls the span twice; only the outer call brackets).
    pub(crate) fn requested(tokens: usize, committed_before: usize, kinds: impl FnOnce() -> Vec<Kind>)
        -> Option<Result<Self, String>>
    {
        if ACTIVE.with(|a| a.borrow().is_some()) {
            return None;
        }
        let mode = requested()?;
        if mode == Mode::Device {
            if let Err(error) = cupti::available() {
                return Some(Err(format!("prefill timing device mode: {error}")));
            }
        }
        ACTIVE.with(|a| *a.borrow_mut() = Some(Active { mode, tiles: 0, last: None, marks: Vec::new(), mark_errors: Vec::new() }));
        // A new snapshot per pass: these counters are per run, never process totals.
        HOST.with(|h| *h.borrow_mut() = Some(Default::default()));
        Some(Ok(Self { mode, tokens, committed_before, kinds: kinds(), pre_sync_ns: 0, started: None }))
    }

    /// After the pre-fence returned: start CUPTI, then the clocks.
    pub(crate) fn start(&mut self, pre_sync: std::time::Duration) -> Result<(), String> {
        self.pre_sync_ns = ns(pre_sync);
        let ts = if self.mode == Mode::Device {
            if let Err(error) = cupti::start() {
                ACTIVE.with(|a| *a.borrow_mut() = None);
                return Err(format!("prefill timing device mode: {error}"));
            }
            cupti::now()
        } else {
            Ok(0)
        };
        self.started = Some((Instant::now(), ts));
        Ok(())
    }

    /// Close the bracket, collect, analyse, publish. Never changes `result`.
    pub(crate) fn finish(self, outcome: &str, committed_after: usize, ready: bool) {
        let wall_ns = self.started.as_ref().map_or(0, |(t, _)| ns(t.elapsed()));
        let host = HOST.with(|h| h.borrow_mut().take());
        let end = if self.mode == Mode::Device && self.started.is_some() { cupti::now() } else { Ok(0) };
        let active = ACTIVE.with(|a| a.borrow_mut().take());
        let (tiles, marks, mark_errors) =
            active.map_or((0, Vec::new(), Vec::new()), |a| (a.tiles, a.marks, a.mark_errors));
        let mut report = json!({
            "scope": "Session::prefill_ids, after a pre-fence until return (final fence, logits readback, Session commit)",
            "mode": self.mode.as_str(), "outcome": outcome, "tokens": self.tokens,
            "committed_before": self.committed_before, "committed_after": committed_after,
            "session_ready": ready, "fenced_tiles": tiles, "pre_sync_ns": self.pre_sync_ns,
            "bracket_wall_ns": wall_ns,
            "tokens_per_second": rate(self.tokens, wall_ns),
        });
        report["host_phases"] = host.map_or(Value::Null, |h| h.prefill_report());
        if self.mode == Mode::Device {
            let start = self.started.as_ref().map_or(Err("bracket never started".into()), |(_, s)| s.clone());
            report["device"] = device_report(start, end, &mark_errors, &marks, &self.kinds, cupti::stop());
        }
        tracing::info!(target: "mary.nemotron.timing", phase = "prefill_append",
            report = %report);
        LAST.with(|l| *l.borrow_mut() = Some(report));
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        // An unwind between requested() and finish() must not leave the
        // thread believing a bracket is still open.
        ACTIVE.with(|a| {
            if let Ok(mut a) = a.try_borrow_mut() { *a = None; }
        });
        HOST.with(|h| {
            if let Ok(mut h) = h.try_borrow_mut() { *h = None; }
        });
        // finish() already disabled what it enabled, so this is a no-op then.
        // After a panic or an early return it disables exactly the kinds still
        // enabled; Drop has no report to carry failures, so they go to the log.
        if self.mode == Mode::Device {
            let errors = cupti::disable_enabled();
            if !errors.is_empty() {
                tracing::info!(target: "mary.nemotron.timing", phase = "prefill_append_cleanup",
                    errors = %errors.join("; "));
            }
        }
    }
}

/// Run `f` with this bracket's host timings, or return `None` (and never call
/// `f`) when no bracket is open.
fn with_host<R>(f: impl FnOnce(&mut super::host_timing::Timings) -> R) -> Option<R> {
    HOST.with(|h| h.borrow_mut().as_mut().map(f))
}

/// One existing synchronous call, timed on the host: started only when a
/// bracket is open (no clock read otherwise), recorded as failed if dropped
/// before `done` (an early `?` return). Adds no fence or synchronisation.
pub(super) struct HostSpan(Option<(super::host_timing::Phase, Instant)>);

impl HostSpan {
    fn start(phase: super::host_timing::Phase) -> Self {
        Self(HOST.with(|h| h.borrow().is_some()).then(|| (phase, Instant::now())))
    }
    /// The expert tile's row index/routing-weight uploads, gather and launches.
    pub(super) fn enqueue() -> Self {
        Self::start(super::host_timing::Phase::Enqueue)
    }
    /// The per-tile fence that already follows every expert tile.
    pub(super) fn fence() -> Self {
        Self::start(super::host_timing::Phase::ExpertFence)
    }
    pub(super) fn done(mut self, ok: bool) {
        self.record(ok);
    }
    fn record(&mut self, ok: bool) {
        if let Some((phase, started)) = self.0.take() {
            with_host(|t| super::host_timing::record(Some(t), phase, Some(started), 0, ok));
        }
    }
}

impl Drop for HostSpan {
    fn drop(&mut self) {
        self.record(false);
    }
}

/// Enable each kind in order. On the first failure, disable exactly the kinds
/// this call already enabled (best effort, each status kept in the error) and
/// return the error; on success return the enabled kinds.
pub(crate) fn enable_exactly<K: Copy + std::fmt::Debug>(kinds: &[K],
    mut enable: impl FnMut(K) -> Result<(), String>, disable: impl FnMut(K) -> Result<(), String>)
    -> Result<Vec<K>, String>
{
    let mut enabled = Vec::new();
    for &kind in kinds {
        if let Err(error) = enable(kind) {
            let statuses: Vec<String> = disable_exactly(enabled, disable).into_iter()
                .map(|(k, status)| format!("{k:?}: {}", status.err().unwrap_or_else(|| "disabled".into())))
                .collect();
            return Err(format!("{error}; rolled back [{}]", statuses.join(", ")));
        }
        enabled.push(kind);
    }
    Ok(enabled)
}

/// Disable every given kind, newest first, keeping each one's status.
pub(crate) fn disable_exactly<K: Copy>(mut kinds: Vec<K>,
    mut disable: impl FnMut(K) -> Result<(), String>) -> Vec<(K, Result<(), String>)>
{
    let mut statuses = Vec::new();
    while let Some(kind) = kinds.pop() {
        statuses.push((kind, disable(kind)));
    }
    statuses
}

fn ns(d: std::time::Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

fn rate(tokens: usize, ns: u64) -> f64 {
    if ns == 0 { 0.0 } else { tokens as f64 * 1e9 / ns as f64 }
}

// ---------------------------------------------------------------- analysis

/// Device stages, in table order. Each device record lands in exactly one.
pub const STAGES: [&str; 18] = [
    "embedding",
    "mamba_projections",
    "mamba_scan",
    "attention_projections",
    "attention_kernel",
    "attention_kv_write",
    "moe_router",
    "moe_latent_projections",
    "expert_dispatch_gather_scatter",
    "expert_tile_index_h2d",
    "expert_gemms",
    "shared_expert",
    "moe_combine",
    "norms",
    "residuals_and_copies",
    "final_norm_lm_head",
    "transfers_other",
    "unclassified",
];

/// Window contexts: what the tile ending at a mark was doing.
pub const CONTEXTS: [&str; 10] = [
    "embedding", "mamba", "attention", "moe_route", "moe_accumulator_zero",
    "moe_experts", "moe_finish", "lm_head_and_commit", "short_append", "other",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ctx {
    Embedding,
    Mamba,
    Attention,
    Route,
    Zero,
    Experts,
    Finish,
    Tail,
    Short,
    Other,
}

impl Ctx {
    fn index(self) -> usize {
        self as usize
    }
    fn of(mark: Option<&Mark>, kinds: &[Kind], any_marks: bool) -> Self {
        let Some(mark) = mark else {
            return if any_marks { Self::Tail } else { Self::Short };
        };
        match (mark.phase, kinds.get(mark.layer)) {
            ("embedding", _) => Self::Embedding,
            ("stateful", Some(Kind::Mamba)) => Self::Mamba,
            ("stateful", Some(Kind::Attention)) => Self::Attention,
            ("route", _) => Self::Route,
            ("accumulator_zero", _) => Self::Zero,
            ("expert_rows", _) => Self::Experts,
            ("finish", _) => Self::Finish,
            // After the LM head's fence: a seeded append's first span ends here.
            ("lm_head", _) => Self::Tail,
            _ => Self::Other,
        }
    }
}

fn stage(name: &str) -> usize {
    STAGES.iter().position(|s| *s == name).expect("known stage")
}

/// Kernel names are CubeCL entry points: the Rust function name, plus a
/// generic suffix. The window context decides what a shared kernel is for.
fn classify(ctx: Ctx, what: &What, after_route_kernel: bool) -> usize {
    let name = match what {
        What::Kernel(name) => name.as_str(),
        What::Memcpy { kind, .. } => {
            return stage(match (ctx, kind) {
                (Ctx::Experts, 1) => "expert_tile_index_h2d",
                (Ctx::Route, 2) => "moe_router",
                (Ctx::Finish, 2) => "moe_combine",
                (Ctx::Tail, 2) => "final_norm_lm_head",
                _ => "transfers_other",
            });
        }
        What::Memset { .. } => return stage("transfers_other"),
    };
    let has = |p: &str| name.contains(p);
    if name.starts_with("norm_kernel") {
        return stage(if ctx == Ctx::Tail { "final_norm_lm_head" } else { "norms" });
    }
    if has("add_kernel") || has("copy_rows_kernel") || name.starts_with("copy_kernel") {
        return stage(if ctx == Ctx::Embedding { "embedding" } else { "residuals_and_copies" });
    }
    if has("embedding_kernel") {
        return stage("embedding");
    }
    if has("last_row_kernel") {
        return stage("final_norm_lm_head");
    }
    if has("convolution") || has("recurrence") || has("grouped_norm") || has("zero_history") {
        return stage("mamba_scan");
    }
    if has("append_tail") {
        return stage("attention_kv_write");
    }
    if has("attention") || has("flash") {
        return stage("attention_kernel");
    }
    if has("route_kernel") {
        return stage("moe_router");
    }
    if has("gather_grouped") || has("ordered_update") || has("zero_rows") {
        return stage("expert_dispatch_gather_scatter");
    }
    if has("sum_kernel") {
        return stage("moe_combine");
    }
    let bf16 = has("bf16_linear") || has("pad_bf16") || has("to_bf16");
    stage(match ctx {
        Ctx::Embedding => "embedding",
        Ctx::Mamba => "mamba_projections",
        Ctx::Attention => "attention_projections",
        Ctx::Route if !after_route_kernel => "moe_router",
        Ctx::Route => "moe_latent_projections",
        Ctx::Zero => "expert_dispatch_gather_scatter",
        Ctx::Experts => "expert_gemms",
        Ctx::Finish if bf16 => "moe_latent_projections",
        Ctx::Finish => "shared_expert",
        Ctx::Tail => "final_norm_lm_head",
        Ctx::Short | Ctx::Other => "unclassified",
    })
}

pub struct Input<'a> {
    pub start: u64,
    pub end: u64,
    pub marks: &'a [Mark],
    pub kinds: &'a [Kind],
    pub activities: &'a [Activity],
}

#[derive(Default, Clone, Copy)]
struct Sum {
    ns: u64,
    calls: u64,
}

impl Sum {
    fn add(&mut self, ns: u64) {
        self.ns = self.ns.saturating_add(ns);
        self.calls += 1;
    }
}

#[derive(Default, Clone, Copy)]
struct Idle {
    windows: u64,
    lead_internal_ns: u64,
    trailing_ns: u64,
}

/// Pure: CUPTI-clock bracket, marks, records -> one report object.
pub fn analyze(input: &Input) -> Value {
    let any_marks = !input.marks.is_empty();
    let bracket = input.end.saturating_sub(input.start);
    // Window i is (bound[i], bound[i+1]]; the last one closes at the bracket end.
    let mut bounds = Vec::with_capacity(input.marks.len() + 2);
    bounds.push(input.start);
    bounds.extend(input.marks.iter().map(|m| m.ts.clamp(input.start, input.end)));
    bounds.push(input.end);
    let windows = bounds.len() - 1;
    let mut per_window: Vec<Vec<&Activity>> = vec![Vec::new(); windows];
    let (mut outside, mut outside_ns) = (0u64, 0u64);
    for a in input.activities {
        if a.start < input.start || a.start > input.end {
            outside += 1;
            outside_ns = outside_ns.saturating_add(a.end.saturating_sub(a.start));
            continue;
        }
        // Number of marks at or before the start = its window.
        let w = input.marks.partition_point(|m| m.ts <= a.start).min(windows - 1);
        per_window[w].push(a);
    }
    let mut stages = [Sum::default(); STAGES.len()];
    let mut idle = [Idle::default(); CONTEXTS.len()];
    let mut ctx_busy = [0u64; CONTEXTS.len()];
    let mut layer_busy: HashMap<usize, (u64, u64)> = HashMap::new();
    let mut names: HashMap<(String, usize), Sum> = HashMap::new();
    let (mut busy_union, mut busy_sum, mut straddling) = (0u64, 0u64, 0u64);
    for (w, list) in per_window.iter_mut().enumerate() {
        let (ws, we) = (bounds[w], bounds[w + 1]);
        let mark = input.marks.get(w);
        let ctx = Ctx::of(mark, input.kinds, any_marks);
        list.sort_by_key(|a| (a.start, a.end));
        let mut after_route_kernel = false;
        let (mut union, mut cursor, mut first, mut last) = (0u64, ws, None, ws);
        for a in list.iter() {
            let s = stage_of(ctx, &a.what, &mut after_route_kernel);
            let dur = a.end.saturating_sub(a.start);
            stages[s].add(dur);
            busy_sum = busy_sum.saturating_add(dur);
            if let What::Kernel(name) = &a.what {
                names.entry((name.clone(), s)).or_default().add(dur);
            }
            if a.end > we {
                straddling += 1;
            }
            // Union of the window-clipped intervals, in start order.
            let (cs, ce) = (a.start.max(ws), a.end.min(we));
            first.get_or_insert(cs);
            if ce > cursor {
                union += ce - cs.max(cursor);
                cursor = ce;
            }
            last = last.max(ce);
        }
        let span = we - ws;
        // Idle after the last device work until the mark: the fence return,
        // the tile's commit and its host bookkeeping. An empty window is all
        // host gap (no device work to trail).
        let trailing = if first.is_some() { (we - last).min(span - union) } else { 0 };
        let ix = ctx.index();
        idle[ix].windows += 1;
        idle[ix].lead_internal_ns += span - union - trailing;
        idle[ix].trailing_ns += trailing;
        ctx_busy[ix] += union;
        busy_union += union;
        // Embedding and LM-head windows belong to no layer.
        if let Some(mark) = mark.filter(|m| m.phase != "embedding" && m.phase != "lm_head") {
            let e = layer_busy.entry(mark.layer).or_default();
            e.0 += union;
            e.1 += span - union;
        }
    }
    let share = |ns: u64| if bracket == 0 { 0.0 } else { ns as f64 / bracket as f64 };
    let stage_rows: Vec<Value> = STAGES.iter().zip(stages.iter()).map(|(name, s)| json!({
        "stage": name, "device_ns": s.ns, "share": share(s.ns), "calls": s.calls,
        "mean_ns": if s.calls == 0 { 0 } else { s.ns / s.calls },
    })).collect();
    let idle_rows: Vec<Value> = CONTEXTS.iter().zip(idle.iter()).zip(ctx_busy.iter())
        .map(|((name, i), busy)| json!({
            "context": name, "windows": i.windows, "device_busy_ns": busy,
            "host_gap_ns": i.lead_internal_ns, "commit_flush_ns": i.trailing_ns,
        })).collect();
    let idle_total: u64 = idle.iter().map(|i| i.lead_internal_ns + i.trailing_ns).sum();
    let mut top: Vec<_> = names.into_iter().collect();
    // Keyed by (name, stage): a shared GEMM name gets one row per stage.
    top.sort_by(|a, b| b.1.ns.cmp(&a.1.ns).then(a.0.cmp(&b.0)));
    let top: Vec<Value> = top.iter().take(20).map(|((name, st), s)| json!({
        "kernel": name, "stage": STAGES[*st], "device_ns": s.ns, "calls": s.calls,
    })).collect();
    let mut layers: Vec<_> = layer_busy.into_iter().collect();
    layers.sort_by_key(|(l, _)| *l);
    let layers: Vec<Value> = layers.into_iter().map(|(layer, (busy, idle))| json!({
        "layer": layer, "kind": input.kinds.get(layer).map(|k| format!("{k:?}")),
        "device_busy_ns": busy, "device_idle_ns": idle,
    })).collect();
    json!({
        "clock": "CUPTI ns: device record timestamps and host marks share it",
        "bracket_ns": bracket, "windows": windows, "marks": input.marks.len(),
        "records": input.activities.len(), "records_outside_bracket": outside,
        "outside_bracket_ns": outside_ns, "records_straddling_window_end": straddling,
        "device_busy_ns": busy_union, "device_busy_sum_ns": busy_sum,
        "device_overlap_ns": busy_sum.saturating_sub(busy_union),
        "device_idle_ns": idle_total, "accounted_ns": busy_union + idle_total,
        "stages": stage_rows, "idle_by_context": idle_rows, "top_kernels": top,
        "per_layer": layers,
    })
}

fn stage_of(ctx: Ctx, what: &What, after_route_kernel: &mut bool) -> usize {
    let s = classify(ctx, what, *after_route_kernel);
    if let What::Kernel(name) = what {
        if name.contains("route_kernel") {
            *after_route_kernel = true;
        }
    }
    s
}

/// One human-readable table for one report object (either mode).
pub fn table(report: &Value) -> String {
    let mut out = String::new();
    let num = |v: &Value| v.as_u64().unwrap_or(0);
    let ms = |ns: u64| ns as f64 / 1e6;
    let bracket_wall = num(&report["bracket_wall_ns"]);
    let _ = writeln!(out, "prefill append: mode {} outcome {} tokens {} committed {}->{} tiles {}",
        report["mode"].as_str().unwrap_or("?"), report["outcome"].as_str().unwrap_or("?"),
        num(&report["tokens"]), num(&report["committed_before"]), num(&report["committed_after"]),
        num(&report["fenced_tiles"]));
    let _ = writeln!(out, "bracket wall {:.3} ms ({:.1} tokens/s), pre-sync wait {:.3} ms (excluded)",
        ms(bracket_wall), report["tokens_per_second"].as_f64().unwrap_or(0.0),
        ms(num(&report["pre_sync_ns"])));
    host_table(&mut out, &report["host_phases"], bracket_wall);
    let device = &report["device"];
    if device.is_null() {
        return out;
    }
    let dropped = match device["cupti_dropped_records"].as_u64() {
        Some(n) => n.to_string(),
        None => format!("UNKNOWN ({})", device["cupti_dropped_error"].as_str().unwrap_or("not observed")),
    };
    if let Some(error) = device["error"].as_str() {
        let _ = writeln!(out, "device attribution FAILED: {error}; CUPTI dropped {dropped}");
        return out;
    }
    let bracket = num(&device["bracket_ns"]);
    let _ = writeln!(out, "device bracket {:.3} ms on the CUPTI clock; busy {:.3} ms, idle {:.3} ms, accounted {:.3} ms",
        ms(bracket), ms(num(&device["device_busy_ns"])), ms(num(&device["device_idle_ns"])),
        ms(num(&device["accounted_ns"])));
    let _ = writeln!(out, "records {} (outside bracket {}, straddling a window end {}, CUPTI dropped {}), overlap {:.3} ms",
        num(&device["records"]), num(&device["records_outside_bracket"]),
        num(&device["records_straddling_window_end"]), dropped,
        ms(num(&device["device_overlap_ns"])));
    let pct = |ns: u64| if bracket == 0 { 0.0 } else { 100.0 * ns as f64 / bracket as f64 };
    let _ = writeln!(out, "{:<34} {:>12} {:>7} {:>9} {:>11}", "device stage", "total ms", "share", "calls", "mean us");
    for row in device["stages"].as_array().into_iter().flatten() {
        let (t, c) = (num(&row["device_ns"]), num(&row["calls"]));
        if c == 0 { continue; }
        let _ = writeln!(out, "{:<34} {:>12.3} {:>6.2}% {:>9} {:>11.2}",
            row["stage"].as_str().unwrap_or("?"), ms(t), pct(t), c, t as f64 / c as f64 / 1e3);
    }
    let _ = writeln!(out, "{:<34} {:>12} {:>7} {:>9} {:>11}", "device idle, by tile context", "total ms", "share", "windows", "mean us");
    for row in device["idle_by_context"].as_array().into_iter().flatten() {
        let w = num(&row["windows"]);
        if w == 0 { continue; }
        let ctx = row["context"].as_str().unwrap_or("?");
        for (label, key) in [("host gap", "host_gap_ns"), ("commit/flush", "commit_flush_ns")] {
            let t = num(&row[key]);
            let _ = writeln!(out, "{:<34} {:>12.3} {:>6.2}% {:>9} {:>11.2}",
                format!("{label}: {ctx}"), ms(t), pct(t), w, t as f64 / w as f64 / 1e3);
        }
    }
    let _ = writeln!(out, "{:<44} {:<30} {:>12} {:>9}", "top kernels", "stage", "total ms", "calls");
    for row in device["top_kernels"].as_array().into_iter().flatten().take(12) {
        let name = row["kernel"].as_str().unwrap_or("?");
        let short: String = name.chars().take(44).collect();
        let _ = writeln!(out, "{:<44} {:<30} {:>12.3} {:>9}", short,
            row["stage"].as_str().unwrap_or("?"), ms(num(&row["device_ns"])), num(&row["calls"]));
    }
    out
}

/// The host-phase section alone, for one run's report (either mode).
pub fn host_summary(report: &Value) -> String {
    let mut out = String::new();
    host_table(&mut out, &report["host_phases"], report["bracket_wall_ns"].as_u64().unwrap_or(0));
    out
}

/// Host phases of the expert tiles (host wall of existing synchronous calls).
/// Each row is its own measured interval; no total or remainder is printed.
/// The retired cache/lookup/upload metrics are named as absent, never as 0.
fn host_table(out: &mut String, host: &Value, bracket_wall: u64) {
    if host.is_null() {
        return;
    }
    let num = |v: &Value| v.as_u64().unwrap_or(0);
    let pct = |ns: u64| if bracket_wall == 0 { 0.0 } else { 100.0 * ns as f64 / bracket_wall as f64 };
    let _ = writeln!(out, "{:<66} {:>12} {:>7} {:>9} {:>11}",
        "host phase, expert tiles (host wall; rows are NOT summed)", "total ms", "of wall", "calls", "mean us");
    for (key, label) in HOST_ROWS {
        let row = &host[key];
        let (t, c) = (num(&row["wall_ns"]), num(&row["calls"]));
        let failed = c - num(&row["succeeded"]).min(c);
        let _ = writeln!(out, "{:<66} {:>12.3} {:>6.2}% {:>9} {:>11.2}{}", label, t as f64 / 1e6, pct(t), c,
            if c == 0 { 0.0 } else { t as f64 / c as f64 / 1e3 },
            if failed > 0 { format!(" ({failed} failed)") } else { String::new() });
    }
    let absent: Vec<&str> = host["absent"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    if !absent.is_empty() {
        let _ = writeln!(out, "ABSENT, not zero: {} ({})", absent.join(", "),
            host["absent_reason"].as_str().unwrap_or("not measured on this path"));
    }
}

/// (JSON key, table label). Enqueue and fence are separate calls on the same
/// thread and do not overlap one another.
pub const HOST_ROWS: [(&str, &str); 2] = [
    ("expert_tile_enqueue", "expert tile enqueue (index/weight uploads, launches)"),
    ("expert_tile_fence", "expert tile fence wait (the existing per-tile fence)"),
];

// -------------------------------------------------------------------- CUPTI

mod cupti {
    use super::{Activity, Capture, Dropped, What};
    use cudarc::cupti::sys as c;
    use std::sync::{Mutex, OnceLock};

    const BUFFER: usize = 16 << 20;
    const ALIGN: usize = 8;
    // Stable prefixes of CUpti_ActivityKernel*, Memcpy*, Memset* (CUPTI keeps
    // field offsets and appends new ones); checked against the bindings in tests.
    pub(super) const KERNEL_START: usize = 16;
    pub(super) const KERNEL_END: usize = 24;
    pub(super) const KERNEL_NAME: usize = 104;
    pub(super) const COPY_KIND: usize = 4;
    pub(super) const COPY_BYTES: usize = 8;
    pub(super) const COPY_START: usize = 16;
    pub(super) const COPY_END: usize = 24;

    /// What the buffer callbacks deliver: records, the dropped count per
    /// callback queue, and every failed CUPTI call (never silently zero).
    struct Store {
        activities: Vec<Activity>,
        dropped: Dropped,
        errors: Vec<String>,
    }
    static STORE: Mutex<Store> = Mutex::new(Store { activities: Vec::new(), dropped: Dropped::new(), errors: Vec::new() });
    static REGISTERED: OnceLock<Result<(), String>> = OnceLock::new();
    /// Exactly the activity kinds enabled and not yet disabled. Process-global,
    /// like the capture itself (see REPORT: one probe per process).
    static ENABLED: Mutex<Vec<c::CUpti_ActivityKind>> = Mutex::new(Vec::new());

    pub(super) fn check(what: &str, r: c::CUptiResult) -> Result<(), String> {
        if r == c::CUptiResult::CUPTI_SUCCESS { Ok(()) } else { Err(format!("{what}: {r:?}")) }
    }

    const KINDS: [c::CUpti_ActivityKind; 3] = [
        c::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        c::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY,
        c::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMSET,
    ];

    pub(super) fn available() -> Result<(), String> {
        if !unsafe { c::is_culib_present() } {
            return Err("libcupti is not loadable (put the CUDA lib64 directory on LD_LIBRARY_PATH)".into());
        }
        let version = version()?;
        // Record layouts are read at the bindings' offsets: refuse another
        // major CUPTI rather than misread its records.
        if version / 10000 != c::CUPTI_API_VERSION / 10000 {
            return Err(format!("CUPTI API {version} differs from the compiled {}", c::CUPTI_API_VERSION));
        }
        Ok(())
    }

    fn version() -> Result<u32, String> {
        let mut v = 0u32;
        check("cuptiGetVersion", unsafe { c::cuptiGetVersion(&mut v) })?;
        Ok(v)
    }

    pub(super) fn start() -> Result<(), String> {
        REGISTERED.get_or_init(|| check("cuptiActivityRegisterCallbacks",
            unsafe { c::cuptiActivityRegisterCallbacks(Some(requested), Some(completed)) })).clone()?;
        match STORE.lock() {
            Ok(mut s) => *s = Store { activities: Vec::new(), dropped: Dropped::new(), errors: Vec::new() },
            Err(_) => return Err("CUPTI record store poisoned".into()),
        }
        let mut enabled = ENABLED.lock().map_err(|_| "CUPTI enabled-kind set poisoned".to_string())?;
        *enabled = super::enable_exactly(&KINDS,
            |kind| check(&format!("cuptiActivityEnable({kind:?})"), unsafe { c::cuptiActivityEnable(kind) }),
            |kind| check(&format!("cuptiActivityDisable({kind:?})"), unsafe { c::cuptiActivityDisable(kind) }))?;
        Ok(())
    }

    /// Best-effort disable of exactly the kinds this process enabled (and has
    /// not yet disabled); one status per kind. Used by stop() and by Drop.
    pub(super) fn disable_enabled() -> Vec<String> {
        let Ok(mut enabled) = ENABLED.lock() else { return vec!["CUPTI enabled-kind set poisoned".into()] };
        super::disable_exactly(std::mem::take(&mut *enabled),
            |kind| check(&format!("cuptiActivityDisable({kind:?})"), unsafe { c::cuptiActivityDisable(kind) }))
            .into_iter().filter_map(|(_, status)| status.err()).collect()
    }

    pub(super) fn stop() -> Result<Capture, String> {
        let mut errors = Vec::new();
        if let Err(e) = check("cuptiActivityFlushAll", unsafe { c::cuptiActivityFlushAll(1) }) { errors.push(e); }
        errors.extend(disable_enabled());
        // The global queue (NULL context), queried after the flush; the
        // per-queue counts arrived with each buffer in `completed`.
        let mut global = 0usize;
        let global = check("cuptiActivityGetNumDroppedRecords(global)",
            unsafe { c::cuptiActivityGetNumDroppedRecords(std::ptr::null_mut(), 0, &mut global) }).map(|()| global);
        let mut store = STORE.lock().map_err(|_| "CUPTI record store poisoned".to_string())?;
        store.dropped.note(global);
        errors.append(&mut store.errors);
        Ok(Capture { activities: std::mem::take(&mut store.activities), dropped: store.dropped.result(),
            errors, version: version() })
    }

    pub(super) fn now() -> Result<u64, String> {
        let mut ts = 0u64;
        check("cuptiGetTimestamp", unsafe { c::cuptiGetTimestamp(&mut ts) })?;
        Ok(ts)
    }

    unsafe extern "C" fn requested(buffer: *mut *mut u8, size: *mut usize, max: *mut usize) {
        let layout = std::alloc::Layout::from_size_align(BUFFER, ALIGN).expect("layout");
        unsafe {
            let p = std::alloc::alloc(layout);
            *buffer = p;
            *size = if p.is_null() { 0 } else { BUFFER };
            *max = 0;
        }
    }

    unsafe extern "C" fn completed(ctx: cudarc::driver::sys::CUcontext, stream: u32,
        buffer: *mut u8, _size: usize, valid: usize)
    {
        let mut out = Vec::new();
        let mut errors = Vec::new();
        if !buffer.is_null() {
            let mut record: *mut c::CUpti_Activity = std::ptr::null_mut();
            loop {
                match unsafe { c::cuptiActivityGetNextRecord(buffer, valid, &mut record) } {
                    c::CUptiResult::CUPTI_SUCCESS => {
                        if let Some(a) = unsafe { decode(record as *const u8) } { out.push(a); }
                    }
                    c::CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED => break,
                    other => { errors.push(format!("cuptiActivityGetNextRecord: {other:?}")); break; }
                }
            }
            unsafe { std::alloc::dealloc(buffer, std::alloc::Layout::from_size_align_unchecked(BUFFER, ALIGN)) };
        }
        // The API's dropped count is per queue: this callback's context and stream.
        let mut dropped = 0usize;
        let dropped = check("cuptiActivityGetNumDroppedRecords",
            unsafe { c::cuptiActivityGetNumDroppedRecords(ctx, stream, &mut dropped) }).map(|()| dropped);
        if let Ok(mut store) = STORE.lock() {
            store.activities.extend(out);
            store.errors.extend(errors);
            store.dropped.note(dropped);
        }
    }

    unsafe fn at<T: Copy>(p: *const u8, offset: usize) -> T {
        unsafe { std::ptr::read_unaligned(p.add(offset) as *const T) }
    }

    unsafe fn decode(p: *const u8) -> Option<Activity> {
        let kind: u32 = unsafe { at(p, 0) };
        if kind == c::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL as u32
            || kind == c::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_KERNEL as u32
        {
            let name: *const std::ffi::c_char = unsafe { at(p, KERNEL_NAME) };
            let name = if name.is_null() { "<unnamed>".to_owned() }
                else { unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned() };
            return Some(Activity { start: unsafe { at(p, KERNEL_START) }, end: unsafe { at(p, KERNEL_END) },
                what: What::Kernel(name) });
        }
        let (start, end, bytes) = unsafe { (at(p, COPY_START), at(p, COPY_END), at::<u64>(p, COPY_BYTES)) };
        if kind == c::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY as u32 {
            return Some(Activity { start, end, what: What::Memcpy { kind: unsafe { at(p, COPY_KIND) }, bytes } });
        }
        if kind == c::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMSET as u32 {
            return Some(Activity { start, end, what: What::Memset { bytes } });
        }
        None
    }
}

#[cfg(test)]
#[path = "prefill_timing_tests.rs"]
mod tests;
