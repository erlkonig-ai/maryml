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

#[derive(Default)]
pub(super) struct Timings {
    mixer: Work,
    selected: Work,
    lookup: Work,
    validate: Work,
    upload: Work,
    enqueue: Work,
    expert_fence: Work,
    mixer_fence: Work,
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
        // Parent totals are INCLUSIVE. Only the five selected phases are disjoint.
        // Output/formatting occurs after these clocks and is charged by the outer
        // generation append timer, not by this append-run total.
        tracing::info!(target: "mary.nemotron.timing", phase = "moe_host_append",
            scope = "Session::run return; host wall, not CPU-only or device duration",
            attempted_tokens = tokens, committed_before, run_ok = ok,
            append_run_ns = %append_ns, mixer = ?self.mixer, selected = ?self.selected,
            lookup_view = ?self.lookup, packed_validation = ?self.validate,
            weight_upload_api = ?self.upload, gpu_work_enqueue = ?self.enqueue,
            expert_fence = ?self.expert_fence, mixer_fence = ?self.mixer_fence,
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
