//! One append's bounded host-return counters, not GPU durations or model state.
//! No subscriber => no clock reads. No extra fences, payloads, or global owner.
//! A panic/abort may leave no final event; absence never means zero work/success.
use std::time::{Duration, Instant};

#[derive(Default, Debug)]
pub(super) struct Work {
    /// Attempts returning normally, including Err; not outstanding/panicked calls.
    calls: u64,
    succeeded: u64,
    wall_ns: u64,
    requested_bytes: u64,
    /// Logical bytes of successful API returns, NOT physical transfer evidence.
    accepted_bytes: u64,
    saturated: bool,
}

impl Work {
    fn merge(&mut self, other: Self) {
        for (dst, value) in [
            (&mut self.calls, other.calls),
            (&mut self.succeeded, other.succeeded),
            (&mut self.wall_ns, other.wall_ns),
            (&mut self.requested_bytes, other.requested_bytes),
            (&mut self.accepted_bytes, other.accepted_bytes),
        ] {
            let (sum, overflow) = dst.overflowing_add(value);
            *dst = if overflow { u64::MAX } else { sum };
            self.saturated |= overflow;
        }
        self.saturated |= other.saturated;
    }

    fn record(&mut self, elapsed: Duration, bytes: usize, ok: bool) {
        fn add(dst: &mut u64, value: u128, saturated: &mut bool) {
            let sum = u128::from(*dst) + value;
            *saturated |= sum > u128::from(u64::MAX);
            *dst = sum.min(u128::from(u64::MAX)) as u64;
        }
        add(&mut self.calls, 1, &mut self.saturated);
        add(&mut self.wall_ns, elapsed.as_nanos(), &mut self.saturated);
        add(
            &mut self.requested_bytes,
            bytes as u128,
            &mut self.saturated,
        );
        if ok {
            add(&mut self.succeeded, 1, &mut self.saturated);
            add(&mut self.accepted_bytes, bytes as u128, &mut self.saturated);
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Phase {
    Mixer,
    Selected,
    Lookup,
    Validate,
    Upload,
    Enqueue,
    ExpertFence,
    MixerFence,
}

#[derive(Clone, Copy)]
pub(super) enum LookupPhase {
    Construction,
    Enumeration,
    NativeRead,
}

/// Optional native-provider detail, scoped to one lookup and then aggregated.
/// Default/empty means unavailable, not measured zero cost. No model data.
/// Byte fields are unused here: read/view traffic is not measured.
#[derive(Default)]
pub struct LookupTimings {
    construction: Work,
    enumeration: Work,
    native_reads: Work,
}

impl LookupTimings {
    pub(super) fn clock(timing: Option<&Self>) -> Option<Instant> {
        timing.map(|_| Instant::now())
    }

    pub(super) fn record(
        timing: Option<&mut Self>, phase: LookupPhase, started: Option<Instant>, ok: bool,
    ) {
        if let (Some(timing), Some(started)) = (timing, started) {
            let work = match phase {
                LookupPhase::Construction => &mut timing.construction,
                LookupPhase::Enumeration => &mut timing.enumeration,
                LookupPhase::NativeRead => &mut timing.native_reads,
            };
            work.record(started.elapsed(), 0, ok);
        }
    }

    fn nested_ns(&self) -> u128 {
        u128::from(self.construction.wall_ns) + u128::from(self.enumeration.wall_ns)
            + u128::from(self.native_reads.wall_ns)
    }

    #[cfg(test)]
    pub(super) fn counts(&self) -> [(u64, u64); 3] {
        [&self.construction, &self.enumeration, &self.native_reads]
            .map(|work| (work.calls, work.succeeded))
    }
}

#[derive(Default)]
pub(super) struct Timings {
    mixer: Work,
    selected: Work,
    lookup: Work,
    lookup_detail: LookupTimings,
    validate: Work,
    upload: Work,
    enqueue: Work,
    expert_fence: Work,
    mixer_fence: Work,
    cache_hits: u64,
    cache_misses: u64,
    cache_evictions: u64,
}

/// Only the actual append entry decides opt-in; inner loops never query env/logging.
pub(super) fn start() -> Option<(Instant, Timings)> {
    tracing::enabled!(target: "mary.nemotron.timing", tracing::Level::INFO)
        .then(|| (Instant::now(), Timings::default()))
}

pub(super) fn clock(timing: Option<&Timings>) -> Option<Instant> {
    timing.map(|_| Instant::now())
}

pub(super) fn record(
    timing: Option<&mut Timings>,
    phase: Phase,
    started: Option<Instant>,
    bytes: usize,
    ok: bool,
) {
    if let (Some(timing), Some(started)) = (timing, started) {
        timing.work(phase).record(started.elapsed(), bytes, ok);
    }
}

/// The provider may fail before invoking its FnOnce callback. End lookup at
/// callback entry, or at the failed return; never include callback work twice.
pub(super) struct Lookup(Option<Instant>);
impl Lookup {
    pub(super) fn start(timing: Option<&Timings>) -> Self {
        Self(clock(timing))
    }
    pub(super) fn entered(&mut self, timing: Option<&mut Timings>) {
        record(timing, Phase::Lookup, self.0.take(), 0, true);
    }
    pub(super) fn returned(self, timing: Option<&mut Timings>) {
        record(timing, Phase::Lookup, self.0, 0, false);
    }
}

impl Timings {
    pub(super) fn merge_lookup(&mut self, detail: LookupTimings) {
        self.lookup_detail.construction.merge(detail.construction);
        self.lookup_detail.enumeration.merge(detail.enumeration);
        self.lookup_detail.native_reads.merge(detail.native_reads);
    }

    // Attempts, including failed miss materialization; not GPU durations.
    pub(super) fn cache_access(&mut self, hit: bool) {
        let count = if hit { &mut self.cache_hits } else { &mut self.cache_misses };
        *count = count.saturating_add(1);
    }

    pub(super) fn cache_eviction(&mut self) {
        self.cache_evictions = self.cache_evictions.saturating_add(1);
    }

    fn work(&mut self, phase: Phase) -> &mut Work {
        match phase {
            Phase::Mixer => &mut self.mixer,
            Phase::Selected => &mut self.selected,
            Phase::Lookup => &mut self.lookup,
            Phase::Validate => &mut self.validate,
            Phase::Upload => &mut self.upload,
            Phase::Enqueue => &mut self.enqueue,
            Phase::ExpertFence => &mut self.expert_fence,
            Phase::MixerFence => &mut self.mixer_fence,
        }
    }

    fn selected_parts_ns(&self) -> u128 {
        [
            &self.lookup,
            &self.validate,
            &self.upload,
            &self.enqueue,
            &self.expert_fence,
        ]
        .iter()
        .map(|work| u128::from(work.wall_ns))
        .sum()
    }

    fn remainder(total: u128, parts: u128) -> (u128, bool) {
        (total.saturating_sub(parts), parts <= total)
    }

    pub(super) fn emit(self, elapsed: Duration, tokens: usize, committed_before: usize, ok: bool) {
        let append_ns = elapsed.as_nanos();
        let selected_parts = self.selected_parts_ns();
        let (selected_other, selected_valid) =
            Self::remainder(u128::from(self.selected.wall_ns), selected_parts);
        let (mixer_other, mixer_valid) = Self::remainder(
            u128::from(self.mixer.wall_ns),
            u128::from(self.selected.wall_ns) + u128::from(self.mixer_fence.wall_ns),
        );
        let (append_other, append_valid) =
            Self::remainder(append_ns, u128::from(self.mixer.wall_ns));
        let lookup_nested = self.lookup_detail.nested_ns();
        let (lookup_other, lookup_valid) =
            Self::remainder(u128::from(self.lookup.wall_ns), lookup_nested);
        // Parent totals are INCLUSIVE. Only the five selected phases are disjoint.
        // The three lookup details are NESTED inside lookup_view, never added
        // again to selected_parts. Construction.calls counts available detail;
        // zero detail calls does not mean lookup itself took zero time.
        // Output/formatting occurs after these clocks and is charged by the outer
        // generation append timer, not by this append-run total.
        tracing::info!(target: "mary.nemotron.timing", phase = "moe_host_append",
            scope = "Session::run return; host wall, not CPU-only or device duration",
            attempted_tokens = tokens, committed_before, run_ok = ok,
            append_run_ns = %append_ns, mixer = ?self.mixer, selected = ?self.selected,
            lookup_view = ?self.lookup, packed_validation = ?self.validate,
            lookup_query_construction = ?self.lookup_detail.construction,
            lookup_min_enumeration = ?self.lookup_detail.enumeration,
            lookup_native_reads = ?self.lookup_detail.native_reads,
            lookup_detail_calls = self.lookup_detail.construction.calls,
            lookup_nested_ns = %lookup_nested, lookup_unclassified_ns = %lookup_other,
            lookup_nesting_valid = lookup_valid,
            weight_upload_api = ?self.upload, gpu_work_enqueue = ?self.enqueue,
            expert_fence = ?self.expert_fence, mixer_fence = ?self.mixer_fence,
            expert_cache_hits = self.cache_hits, expert_cache_misses = self.cache_misses,
            expert_cache_evictions = self.cache_evictions,
            selected_disjoint_ns = %selected_parts, selected_unclassified_ns = %selected_other,
            mixer_other_ns = %mixer_other, append_other_ns = %append_other,
            nesting_valid = selected_valid && mixer_valid && append_valid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_and_partial_upload_returns_are_not_successful_bytes() {
        let mut work = Work::default();
        work.record(Duration::from_nanos(11), 128, true);
        work.record(Duration::from_nanos(17), 256, false);
        assert_eq!((work.calls, work.succeeded, work.wall_ns), (2, 1, 28));
        assert_eq!((work.requested_bytes, work.accepted_bytes), (384, 128));
        assert!(!work.saturated);
    }

    #[test]
    fn disabled_inner_clock_and_record_have_no_observer() {
        tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
            assert!(start().is_none());
        });
        assert!(clock(None).is_none());
        record(None, Phase::Upload, None, usize::MAX, false);
        assert!(LookupTimings::clock(None).is_none());
        LookupTimings::record(None, LookupPhase::NativeRead, None, false);
        let mut lookup = Lookup::start(None);
        assert!(lookup.0.is_none());
        lookup.entered(None);
        lookup.returned(None);
    }

    #[test]
    fn lookup_failure_before_callback_is_counted() {
        let mut t = Timings::default();
        let lookup = Lookup::start(Some(&t));
        lookup.returned(Some(&mut t));
        assert_eq!((t.lookup.calls, t.lookup.succeeded), (1, 0));
    }

    #[test]
    fn callback_failure_does_not_reclassify_or_double_count_lookup() {
        let mut t = Timings::default();
        let mut lookup = Lookup::start(Some(&t));
        lookup.entered(Some(&mut t));
        // Subsequent callback validation/upload failure does not undo lookup.
        lookup.returned(Some(&mut t));
        assert_eq!((t.lookup.calls, t.lookup.succeeded), (1, 1));
    }

    #[test]
    fn parents_are_not_added_to_disjoint_selected_phases() {
        let mut t = Timings::default();
        for phase in [
            Phase::Lookup,
            Phase::Validate,
            Phase::Upload,
            Phase::Enqueue,
            Phase::ExpertFence,
        ] {
            t.work(phase).record(Duration::from_nanos(3), 0, true);
        }
        t.selected.record(Duration::from_nanos(19), 0, false);
        t.mixer.record(Duration::from_nanos(31), 0, false);
        assert_eq!(t.selected_parts_ns(), 15);
        assert_eq!(Timings::remainder(19, t.selected_parts_ns()), (4, true));
        assert_eq!(Timings::remainder(14, t.selected_parts_ns()), (0, false));
    }

    #[test]
    fn lookup_detail_merges_without_double_counting_parents() {
        let mut t = Timings::default();
        t.lookup.record(Duration::from_nanos(41), 0, true);
        t.lookup.record(Duration::from_nanos(29), 0, false);
        let mut first = LookupTimings::default();
        first.construction.record(Duration::from_nanos(3), 0, true);
        first.enumeration.record(Duration::from_nanos(7), 0, true);
        for _ in 0..4 {
            first.native_reads.record(Duration::from_nanos(5), 0, true);
        }
        t.merge_lookup(first);
        let mut failed = LookupTimings::default();
        failed.construction.record(Duration::from_nanos(2), 0, true);
        failed.enumeration.record(Duration::from_nanos(6), 0, true);
        failed.native_reads.record(Duration::from_nanos(11), 0, false);
        t.merge_lookup(failed);
        assert_eq!(t.lookup_detail.counts(), [(2, 2), (2, 2), (5, 4)]);
        assert_eq!(t.lookup_detail.nested_ns(), 49);
        assert_eq!(Timings::remainder(70, t.lookup_detail.nested_ns()), (21, true));
        assert_eq!(t.selected_parts_ns(), 70);
        assert_eq!((t.lookup.calls, t.lookup.succeeded), (2, 1));

        // No native detail is a valid unavailable observation, even when the
        // parent lookup was measured. It is not evidence of zero nested cost.
        let unavailable = LookupTimings::default();
        assert_eq!(unavailable.counts(), [(0, 0); 3]);
        t.merge_lookup(unavailable);
        assert_eq!(t.lookup_detail.nested_ns(), 49);
    }

    #[test]
    fn lookup_detail_merge_preserves_saturation() {
        let mut w = Work { wall_ns: u64::MAX, ..Work::default() };
        let mut detail = Work::default();
        detail.record(Duration::from_nanos(1), 0, true);
        w.merge(detail);
        assert_eq!((w.calls, w.succeeded, w.wall_ns), (1, 1, u64::MAX));
        assert!(w.saturated);
        let mut aggregate = Work::default();
        aggregate.merge(w);
        assert!(aggregate.saturated);
    }

    #[test]
    fn counter_saturation_is_explicit() {
        let mut w = Work {
            calls: u64::MAX,
            ..Work::default()
        };
        w.record(Duration::from_secs(u64::MAX), 0, false);
        assert_eq!(w.calls, u64::MAX);
        assert_eq!(w.wall_ns, u64::MAX);
        assert!(w.saturated);
    }
}
