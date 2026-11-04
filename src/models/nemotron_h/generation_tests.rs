//! Scripted control-flow tests plus one explicitly ignored native dispatch test.
//! The scripted cases are NOT a CPU model, codec oracle or GPU witness.
//! Existing chat tests own token grammar/UTF-8 semantics; these scripts inject
//! failures and logits at the same private loop boundary used by production.
use super::super::chat::ExecProposal;
use super::*;
use super::super::session::MAX_CHUNK;
use std::{cell::Cell, collections::VecDeque, rc::Rc};

struct Script {
    capacity: usize,
    committed: Vec<u32>,
    status: Status,
    predictions: VecDeque<u32>,
    logits: Vec<f32>,
    appends: Vec<Vec<u32>>,
    resets: usize,
    fail_reset: bool,
    fail_append: Option<usize>,
    cancel_after_append: Option<(usize, Rc<Cell<bool>>)>,
    cancel_inside_prefill: Option<Rc<Cell<bool>>>,
    saved_prefix: Option<Vec<u32>>,
    restored: usize,
}
fn row(id: u32) -> Vec<f32> {
    let mut values = vec![0.0; 16];
    values[id as usize] = 1.0;
    values
}
impl Script {
    fn new(predictions: &[u32]) -> Self {
        Self {
            capacity: 300,
            committed: Vec::new(),
            status: Status::Ready,
            predictions: predictions.iter().copied().collect(),
            logits: Vec::new(),
            appends: Vec::new(),
            resets: 0,
            fail_reset: false,
            fail_append: None,
            cancel_after_append: None,
            cancel_inside_prefill: None,
            saved_prefix: None,
            restored: 0,
        }
    }
}
impl Runtime for Script {
    fn reset_for_prompt(&mut self, ids: &[u32]) -> Result<()> {
        if let Some(prefix) = self.saved_prefix.clone().filter(|p| ids.starts_with(p)) {
            self.status = Status::Poisoned;
            ensure!(!self.fail_reset, "scripted restore failure");
            self.committed = prefix;
            self.logits = row(11); // Saved boundary predicts EOS, unlike the tail.
            self.restored += 1;
            self.status = Status::Ready;
            Ok(())
        } else { self.reset() }
    }
    fn prefill(&mut self, ids: &[u32], cancelled: &mut dyn FnMut() -> bool) -> Result<PrefillOutcome> {
        if cancelled() { return Ok(PrefillOutcome::Cancelled); }
        if ids.len() > MAX_CHUNK {
            if let Some(signal) = &self.cancel_inside_prefill {
                self.appends.push(ids.to_vec());
                self.status = Status::Poisoned;
                signal.set(true); // Stand in for private layer advancement.
                assert!(cancelled());
                return Ok(PrefillOutcome::Cancelled);
            }
        }
        self.append(ids)?;
        Ok(PrefillOutcome::Completed)
    }
    fn context_capacity(&self) -> usize {
        self.capacity
    }
    fn vocabulary_size(&self) -> usize {
        16
    }
    fn committed_ids(&self) -> &[u32] {
        &self.committed
    }
    fn status(&self) -> Status {
        self.status
    }
    fn last_logits(&self) -> Option<&[f32]> {
        (self.status == Status::Ready && !self.committed.is_empty()).then_some(&self.logits)
    }
    fn reset(&mut self) -> Result<()> {
        self.resets += 1;
        self.status = Status::Poisoned;
        ensure!(!self.fail_reset, "scripted reset failure");
        self.committed.clear();
        self.logits.clear();
        self.status = Status::Ready;
        Ok(())
    }
    fn append(&mut self, ids: &[u32]) -> Result<()> {
        self.appends.push(ids.to_vec());
        self.status = Status::Poisoned;
        ensure!(
            self.fail_append != Some(self.appends.len()),
            "scripted append failure"
        );
        self.logits = row(self
            .predictions
            .pop_front()
            .expect("script needs next logits"));
        self.committed.extend_from_slice(ids);
        self.status = Status::Ready;
        if let Some((count, signal)) = &self.cancel_after_append {
            if *count == self.appends.len() {
                signal.set(true);
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct DecodeScript {
    ids: Vec<u32>,
    fail_push: Option<usize>,
    fail_finish: bool,
    protocol_push: Option<usize>,
    protocol_finish: bool,
    cancel_on_push: Option<Rc<Cell<bool>>>,
}
impl DecodeScript {
    fn partial(&self) -> PartialOutput {
        PartialOutput {
            raw_text: self
                .ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
            protocol_error: self
                .protocol_push
                .filter(|&at| self.ids.len() >= at)
                .map(|_| "scripted malformed push".into()),
            ..PartialOutput::default()
        }
    }
}
impl Decoder for DecodeScript {
    fn push(&mut self, id: u32) -> Result<PartialOutput> {
        self.ids.push(id);
        ensure!(
            self.fail_push != Some(self.ids.len()),
            "scripted decode failure"
        );
        if let Some(signal) = &self.cancel_on_push {
            signal.set(true);
        }
        Ok(self.partial())
    }
    fn finish(self, stop: StopReason) -> Result<Completion> {
        ensure!(!self.fail_finish, "scripted finish failure");
        let mut partial = self.partial();
        if self.protocol_finish {
            partial.protocol_error = Some("scripted malformed finish".into());
        }
        // Deliberately offer on every stop to check the owner's unconditional
        // suppression. This is not OutputParser grammar behavior.
        Ok(Completion {
            partial,
            stop,
            proposal: Some(ExecProposal {
                command: "mock, never executed".into(),
            }),
        })
    }
}
fn run(script: &mut Script, prompt: &[u32], budget: usize, decoder: DecodeScript) -> TurnOutput {
    run_turn(
        script,
        &mut Sampler::new(SamplingConfig::greedy(), 0).unwrap(),
        prompt,
        budget,
        &mut || false,
        decoder,
    )
}
fn errored(out: &TurnOutput, needle: &str) {
    assert!(matches!(&out.completion.stop, StopReason::Error(message) if message.contains(needle)));
    assert!(out.completion.proposal.is_none());
}

#[derive(Default)]
struct ProgressSpan {
    fields: Vec<(String, String)>,
    closed: bool,
}

#[derive(Clone)]
struct ProgressCapture(std::sync::Arc<std::sync::Mutex<Vec<ProgressSpan>>>, &'static str);
impl Default for ProgressCapture {
    fn default() -> Self {
        Self(Default::default(), "native_append")
    }
}

struct ProgressFields<'a>(&'a mut Vec<(String, String)>);
impl tracing::field::Visit for ProgressFields<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().into(), format!("{value:?}")));
    }
}
impl tracing::Subscriber for ProgressCapture {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.is_span() && metadata.target() == "mary.nemotron.progress"
            && metadata.name() == self.1
    }
    fn new_span(&self, attributes: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        let mut span = ProgressSpan::default();
        attributes.record(&mut ProgressFields(&mut span.fields));
        let mut spans = self.0.lock().unwrap();
        spans.push(span);
        tracing::span::Id::from_u64(spans.len() as u64)
    }
    fn record(&self, id: &tracing::span::Id, record: &tracing::span::Record<'_>) {
        record.record(&mut ProgressFields(&mut self.0.lock().unwrap()[id.into_u64() as usize - 1].fields));
    }
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) { panic!("progress enables no events") }
    fn try_close(&self, id: tracing::span::Id) -> bool {
        self.0.lock().unwrap()[id.into_u64() as usize - 1].closed = true;
        true
    }
}
impl ProgressSpan {
    fn field(&self, name: &str) -> Option<&str> {
        let mut values = self.fields.iter().filter(|(key, _)| key == name);
        let value = values.next().map(|(_, value)| value.as_str());
        assert!(values.next().is_none(), "{name} must be written only once");
        value
    }
}

#[test]
#[ignore = "reserved CUDA: real Session reuse dispatch, cancelled before prefill"]
fn generation_gpu_session_reuse_dispatch_returns_before_prefill_cancel() {
    use super::super::session::tests::{fixture, limits};
    use cubecl::cuda::CudaDevice;

    let (snapshot, source, layers) = fixture();
    let seed = [1, 5, 7];
    let mut session = Session::from_snapshot_seed(
        snapshot, source, &layers, CudaDevice { index: 0 }, limits(), &seed,
    ).unwrap();
    let ownership = session.prefix_ownership().unwrap();
    assert!(session.can_reuse_prefix(&seed));
    assert!(session.committed_ids().is_empty());
    let capture = ProgressCapture(Default::default(), "native_prefix");
    let cancel_calls = Cell::new(0);
    let output = tracing::subscriber::with_default(capture.clone(), || run_turn(
        &mut session,
        &mut Sampler::new(SamplingConfig::greedy(), 0).unwrap(),
        &seed,
        1,
        &mut || {
            cancel_calls.set(cancel_calls.get() + 1);
            let spans = capture.0.lock().unwrap();
            if cancel_calls.get() == 1 {
                assert!(spans.is_empty());
                false // Enter the actual Runtime<Session> reuse branch.
            } else {
                assert_eq!(cancel_calls.get(), 2);
                assert_eq!(spans.len(), 1, "real Session reuse hook must have run");
                assert!(spans[0].closed, "reuse dispatch must return before cancellation");
                true // Session::prefill_ids checks this before any prefill work.
            }
        },
        DecodeScript::default(),
    ));
    assert_eq!(cancel_calls.get(), 2);
    assert_eq!(output.completion.stop, StopReason::Cancelled);
    assert!(output.emitted_ids.is_empty());
    assert_eq!(output.committed_tokens, 0);
    assert_eq!(output.session_status, Status::Ready);
    assert_eq!(session.prefix_ownership().unwrap(), ownership);
    assert!(session.saved_prefix().is_none());
    assert!(session.can_reuse_prefix(&seed));
    let spans = capture.0.lock().unwrap();
    assert_eq!(spans[0].field("outcome"), Some("\"seed_pending\""));
    assert_eq!(spans[0].field("reused_prefix_tokens"), Some("0"));
    assert_eq!(spans[0].field("saved_prefix_tokens"), Some("0"));
    assert_eq!(spans[0].field("session_ready"), Some("true"));
}

#[test]
fn generation_progress_distinguishes_returned_success_and_failure_without_rich_timing() {
    for fail in [false, true] {
        let make = || {
            let mut script = Script::new(&[5, 11]);
            script.fail_append = fail.then_some(2);
            script
        };
        let mut baseline = make();
        let off = tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(),
            || run(&mut baseline, &[1, 3], 2, DecodeScript::default()));
        let capture = ProgressCapture::default();
        let mut script = make();
        let on = tracing::subscriber::with_default(capture.clone(), || {
            assert!(!tracing::enabled!(target: "mary.nemotron.timing", tracing::Level::INFO));
            #[cfg(feature = "nemotron-timing")]
            {
                assert!(TurnTiming::start(0).is_none());
                assert!(super::super::host_timing::start().is_none());
            }
            run(&mut script, &[1, 3], 2, DecodeScript::default())
        });
        assert_eq!(on.emitted_ids, off.emitted_ids);
        assert_eq!(on.completion.stop, off.completion.stop);
        assert_eq!(on.session_status, off.session_status);
        assert_eq!(on.committed_tokens, off.committed_tokens);
        assert_eq!(script.appends, baseline.appends);
        let spans = capture.0.lock().unwrap();
        assert_eq!(spans.len(), 2);
        for (index, span) in spans.iter().enumerate() {
            let ok = index == 0 || !fail;
            assert!(span.closed);
            assert_eq!(span.field("role"), Some(if index == 0 { "\"prompt\"" } else { "\"generated\"" }));
            assert_eq!(span.field("committed_before"), Some(if index == 0 { "0" } else { "2" }));
            assert_eq!(span.field("attempted_tokens"), Some(if index == 0 { "2" } else { "1" }));
            assert_eq!(span.field("prompt_tokens"), Some("2"));
            assert_eq!(span.field("append_returned"), Some("true"));
            assert_eq!(span.field("append_ok"), Some(if ok { "true" } else { "false" }));
            assert_eq!(span.field("completed_tokens"), Some(if index == 0 { "2" } else if ok { "1" } else { "0" }));
            assert_eq!(span.field("committed_after"), Some(if index == 0 || !ok { "2" } else { "3" }));
            assert_eq!(span.field("session_ready"), Some(if ok { "true" } else { "false" }));
            assert!(span.field("append_wall_ns").unwrap().parse::<u64>().is_ok());
            assert_eq!(span.field("append_cancelled"), Some("false"));
            assert_eq!(span.fields.len(), 11, "bounded numeric/static fields only");
        }
    }
}

#[test]
fn generation_progress_unwind_has_no_return_or_success_fields() {
    let capture = ProgressCapture::default();
    let mut script = Script::new(&[]); // Missing scripted logits panics inside append.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tracing::subscriber::with_default(capture.clone(),
            || run(&mut script, &[1], 1, DecodeScript::default()));
    }));
    assert!(result.is_err());
    assert_eq!(script.status, Status::Poisoned);
    let spans = capture.0.lock().unwrap();
    assert_eq!(spans.len(), 1);
    assert!(spans[0].closed, "an unwind closes the span without claiming a return");
    assert_eq!(spans[0].fields.len(), 4);
    for name in ["append_returned", "append_ok", "append_cancelled", "completed_tokens", "committed_after", "session_ready", "append_wall_ns"] {
        assert_eq!(spans[0].field(name), None, "{name}");
    }
}

#[test]
fn generation_progress_cancellation_counts_only_appends_that_returned() {
    for initially_cancelled in [false, true] {
        let capture = ProgressCapture::default();
        let cancelled = Rc::new(Cell::new(initially_cancelled));
        let mut script = Script::new(&[5]);
        script.cancel_after_append = Some((1, Rc::clone(&cancelled)));
        let output = tracing::subscriber::with_default(capture.clone(), || run_turn(
            &mut script, &mut Sampler::new(SamplingConfig::greedy(), 0).unwrap(),
            &vec![1; 257], 2, &mut || cancelled.get(), DecodeScript::default(),
        ));
        assert_eq!(output.completion.stop, StopReason::Cancelled);
        let spans = capture.0.lock().unwrap();
        if initially_cancelled {
            assert!(spans.is_empty());
        } else {
            assert_eq!(spans.len(), 1, "one known prompt span");
            assert_eq!(spans[0].field("completed_tokens"), Some("257"));
            assert_eq!(spans[0].field("prompt_tokens"), Some("257"));
            assert_eq!(spans[0].field("append_ok"), Some("true"));
        }
    }
}

#[test]
fn generation_long_prefill_cancel_is_not_reusable_and_never_publishes_completion() {
    let mut script = Script::new(&[11]);
    script.committed = vec![9];
    let signal = Rc::new(Cell::new(false));
    script.cancel_inside_prefill = Some(signal.clone());
    let prompt = [vec![9], vec![1; MAX_CHUNK + 1]].concat();
    let capture = ProgressCapture::default();
    let out = tracing::subscriber::with_default(capture.clone(), || run_turn(
        &mut script, &mut Sampler::new(SamplingConfig::greedy(), 0).unwrap(),
        &prompt, 1, &mut || signal.get(), DecodeScript::default(),
    ));
    assert_eq!(out.completion.stop, StopReason::Cancelled);
    assert_eq!(out.session_status, Status::Poisoned);
    assert_eq!(out.committed_tokens, 1, "old IDs are diagnostic only");
    assert!(out.emitted_ids.is_empty());
    assert_eq!(script.appends[0].len(), 257, "entire known suffix, not short fallback");
    let spans = capture.0.lock().unwrap();
    assert_eq!(spans.len(), 1);
    for (name, value) in [("append_returned", "true"), ("append_ok", "false"),
        ("append_cancelled", "true"), ("completed_tokens", "0"),
        ("committed_after", "1"), ("session_ready", "false")] {
        assert_eq!(spans[0].field(name), Some(value));
    }
    drop(spans);
    script.cancel_inside_prefill = None;
    let next = run(&mut script, &prompt, 1, DecodeScript::default());
    assert_eq!(script.resets, 1, "diagnostic matching prefix MUST reset after cancellation");
    assert_eq!(script.appends[1], prompt);
    assert_eq!(next.emitted_ids, [11]);
    assert_eq!(script.committed, prompt, "emitted EOS remains unattended");
    // Exact reuse uses the previously computed last logits, no empty prefill.
    let count = script.appends.len();
    let again = run(&mut script, &prompt, 1, DecodeScript::default());
    assert_eq!(again.emitted_ids, [11]);
    assert_eq!(script.appends.len(), count);
}

// Observe the actual production loop event without a subscriber dependency or
// a CPU model. These remain scripted state/control-flow witnesses, not kernels.
#[cfg(feature = "nemotron-timing")]
#[test]
fn generation_timing_counts_include_failed_attempts_without_state_changes() {
    use std::sync::{Arc, Mutex};
    use tracing::{Event, Metadata, Subscriber, field::Visit, span::{Attributes, Id, Record}};

    #[derive(Clone, Default)]
    struct Capture {
        events: Arc<Mutex<Vec<Vec<(String, String)>>>>,
        spans: Arc<Mutex<Vec<Vec<(String, String)>>>>,
    }
    #[derive(Default)]
    struct Fields(Vec<(String, String)>);
    impl Visit for Fields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0.push((field.name().into(), format!("{value:?}")));
        }
    }
    impl Subscriber for Capture {
        fn enabled(&self, meta: &Metadata<'_>) -> bool {
            meta.target() == "mary.nemotron.timing" || meta.target() == "mary.nemotron.progress"
        }
        fn new_span(&self, attrs: &Attributes<'_>) -> Id {
            assert_eq!(attrs.metadata().name(), "native_append");
            let mut fields = Fields::default();
            attrs.record(&mut fields);
            let mut spans = self.spans.lock().unwrap();
            spans.push(fields.0);
            Id::from_u64(spans.len() as u64)
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
        fn event(&self, event: &Event<'_>) {
            let mut fields = Fields::default();
            event.record(&mut fields);
            self.events.lock().unwrap().push(fields.0);
        }
    }

    for (fail, reuse) in [(false, false), (true, false), (false, true)] {
        let make = || {
            let mut script = Script::new(&[5, 11]);
            script.committed = if reuse { vec![1, 3] } else { vec![9] };
            script.fail_append = fail.then_some(2);
            script
        };
        let mut off = make();
        let baseline = tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(),
            || run(&mut off, &[1, 3, 4], 2, DecodeScript::default()));
        let mut on = make();
        let capture = Capture::default();
        let observed = tracing::subscriber::with_default(capture.clone(),
            || run(&mut on, &[1, 3, 4], 2, DecodeScript::default()));
        assert_eq!(observed.emitted_ids, baseline.emitted_ids);
        assert_eq!(observed.completion.stop, baseline.completion.stop);
        assert_eq!(observed.committed_tokens, baseline.committed_tokens);
        assert_eq!(observed.session_status, baseline.session_status);
        assert_eq!(on.appends, off.appends);
        assert_eq!(on.resets, off.resets);
        let spans = capture.spans.lock().unwrap();
        assert_eq!(spans.len(), 2, "one span per actual append attempt");
        for (span, role, before, tokens) in [
            (
                &spans[0], "\"prompt\"", if reuse { "2" } else { "0" },
                if reuse { "1" } else { "3" },
            ),
            (&spans[1], "\"generated\"", "3", "1"),
        ] {
            let field = |name: &str| {
                span.iter().find(|(key, _)| key == name).unwrap().1.as_str()
            };
            assert_eq!(field("role"), role);
            assert_eq!(field("committed_before"), before);
            assert_eq!(field("attempted_tokens"), tokens);
        }
        // With prefix reuse both chunks have T=1, but retain distinct roles.
        // The failed generated append still owns its correctly labelled span.
        let events = capture.events.lock().unwrap();
        assert_eq!(events.len(), 1, "one aggregate, not per-token events");
        let field = |name: &str| events[0].iter().find(|(key, _)| key == name).unwrap().1.clone();
        assert_eq!(field("phase"), "\"generation\"");
        for (name, value) in [("reset_calls", if reuse { "0" } else { "1" }),
            ("reset_completed", if reuse { "0" } else { "1" }),
            ("committed_before", if reuse { "2" } else { "1" }),
            ("reused_tokens", if reuse { "2" } else { "0" }), ("prompt_tokens", "3"),
            ("prompt_append_calls", "1"), ("prompt_attended_tokens", if reuse { "1" } else { "3" }),
            ("generated_append_calls", "1"), ("generated_attempted_tokens", "1")] {
            assert_eq!(field(name), value, "{name}");
        }
        assert_eq!(field("generated_attended_tokens"), if fail { "0" } else { "1" });
        assert_eq!(field("emitted_tokens"), if fail { "1" } else { "2" });
        assert_eq!(field("attended_tokens"), if fail { "3" } else { "4" });
        // The synthetic error message and emitted/raw text never enter timing.
        assert!(!format!("{events:?}").contains("scripted"));
    }
}

#[test]
fn generation_eos_is_retained_at_budget_boundary_not_attended() {
    for eos in [2, 11] {
        let mut s = Script::new(&[5, eos]);
        let out = run(&mut s, &[1], 2, DecodeScript::default());
        assert_eq!(out.emitted_ids, [5, eos]);
        assert_eq!(out.completion.stop, StopReason::Eos { id: eos });
        assert_eq!(out.committed_tokens, 2);
        assert_eq!(s.committed, [1, 5]);
        assert_eq!(s.appends, [vec![1], vec![5]]);
        assert_eq!(out.completion.partial.raw_text, format!("5,{eos}"));
        assert!(out.decoded_complete && out.limit.is_none());
    }
}

#[test]
fn generation_zero_budget_does_not_reset_or_append() {
    let mut s = Script::new(&[]);
    s.committed = vec![9];
    s.status = Status::Poisoned;
    let out = run(&mut s, &[1], 0, DecodeScript::default());
    assert_eq!(out.completion.stop, StopReason::Budget);
    assert_eq!(out.limit, Some(LimitCause::TokenBudget));
    assert!(out.emitted_ids.is_empty() && out.completion.proposal.is_none());
    assert_eq!(out.session_status, Status::Poisoned);
    assert_eq!(s.committed, [9]);
    assert!(s.appends.is_empty() && s.resets == 0);
}

#[test]
fn generation_context_limit_is_budget_not_eos_and_does_not_overflow() {
    let mut s = Script::new(&[5, 6, 7]);
    s.capacity = 3;
    let out = run(&mut s, &[1], usize::MAX, DecodeScript::default());
    assert_eq!(out.emitted_ids, [5, 6, 7]);
    assert_eq!(out.completion.stop, StopReason::Budget);
    assert_eq!(out.limit, Some(LimitCause::Context));
    assert_eq!(s.committed, [1, 5, 6]);
    assert_eq!(out.committed_tokens, 3);
    assert!(out.completion.proposal.is_none());
}

#[test]
fn generation_cancel_before_work_and_during_private_prefill() {
    let mut s = Script::new(&[5]);
    s.committed = vec![9];
    let mut sampler = Sampler::new(SamplingConfig::greedy(), 0).unwrap();
    let out = run_turn(
        &mut s,
        &mut sampler,
        &[1],
        2,
        &mut || true,
        DecodeScript::default(),
    );
    assert_eq!(out.completion.stop, StopReason::Cancelled);
    assert_eq!(s.committed, [9]);
    assert!(s.appends.is_empty() && s.resets == 0);
    let signal = Rc::new(Cell::new(false));
    s.cancel_inside_prefill = Some(signal.clone());
    let out = run_turn(
        &mut s,
        &mut sampler,
        &vec![1; MAX_CHUNK + 1],
        2,
        &mut || signal.get(),
        DecodeScript::default(),
    );
    assert_eq!(out.completion.stop, StopReason::Cancelled);
    assert_eq!(out.committed_tokens, 0, "no all-layer prefix was published");
    assert_eq!(out.session_status, Status::Poisoned);
    assert_eq!(s.appends.len(), 1);
    assert_eq!(s.resets, 1);
    assert!(out.emitted_ids.is_empty() && out.completion.proposal.is_none());
}

#[test]
fn generation_cancel_after_emission_retains_unattended_id() {
    let mut s = Script::new(&[5]);
    let signal = Rc::new(Cell::new(false));
    let decoder = DecodeScript {
        cancel_on_push: Some(signal.clone()),
        ..Default::default()
    };
    let out = run_turn(
        &mut s,
        &mut Sampler::new(SamplingConfig::greedy(), 0).unwrap(),
        &[1],
        4,
        &mut || signal.get(),
        decoder,
    );
    assert_eq!(out.completion.stop, StopReason::Cancelled);
    assert_eq!(out.emitted_ids, [5]);
    assert_eq!(out.committed_tokens, 1);
    assert_eq!(out.completion.partial.raw_text, "5");
    assert!(out.completion.proposal.is_none());
}

#[test]
fn generation_exact_prefix_reuse_and_replacement() {
    let mut s = Script::new(&[6]);
    s.committed = vec![1];
    s.logits = row(5);
    let out = run(&mut s, &[1], 1, DecodeScript::default());
    assert_eq!(out.emitted_ids, [5]);
    assert!(s.appends.is_empty() && s.resets == 0);
    let out = run(&mut s, &[9], 1, DecodeScript::default());
    assert_eq!(out.emitted_ids, [6]);
    assert_eq!(s.committed, [9]);
    assert_eq!(s.resets, 1);
    assert_eq!(s.appends, [vec![9]]);
}

#[test]
fn generation_reset_failure_retains_actual_status_and_no_old_prefix_promise() {
    let mut s = Script::new(&[]);
    s.committed = vec![9];
    s.fail_reset = true;
    let out = run(&mut s, &[1], 2, DecodeScript::default());
    errored(&out, "reset failure");
    assert_eq!(out.session_status, Status::Poisoned);
    assert_eq!(out.committed_tokens, 1);
    assert!(s.appends.is_empty() && out.emitted_ids.is_empty());
}

#[test]
fn generation_runtime_error_keeps_emission_and_partial() {
    let mut s = Script::new(&[5]);
    s.fail_append = Some(2);
    let out = run(&mut s, &[1], 3, DecodeScript::default());
    errored(&out, "append failure");
    assert_eq!(out.emitted_ids, [5]);
    assert_eq!(out.completion.partial.raw_text, "5");
    assert!(out.decoded_complete);
    assert_eq!(out.session_status, Status::Poisoned);
    assert_eq!(out.committed_tokens, 1);
}

#[test]
fn generation_decoder_failure_keeps_all_ids_and_last_valid_partial() {
    let mut s = Script::new(&[5, 6]);
    let decoder = DecodeScript {
        fail_push: Some(2),
        ..Default::default()
    };
    let out = run(&mut s, &[1], 3, decoder);
    errored(&out, "decode failure");
    assert_eq!(out.emitted_ids, [5, 6]);
    assert_eq!(out.completion.partial.raw_text, "5");
    assert!(!out.decoded_complete);
    assert_eq!(out.session_status, Status::Ready);
    assert_eq!(s.committed, [1, 5]);
}

#[test]
fn generation_finish_failure_does_not_mask_runtime_failure() {
    let mut s = Script::new(&[5]);
    s.fail_append = Some(2);
    let decoder = DecodeScript {
        fail_finish: true,
        ..Default::default()
    };
    let out = run(&mut s, &[1], 3, decoder);
    errored(&out, "append failure");
    errored(&out, "finish failure");
    assert_eq!(out.emitted_ids, [5]);
    assert_eq!(out.completion.partial.raw_text, "5");
    assert!(!out.decoded_complete);
}

#[test]
fn generation_push_protocol_error_retains_newest_decoded_partial() {
    let mut s = Script::new(&[5, 6]);
    let decoder = DecodeScript {
        protocol_push: Some(2),
        ..Default::default()
    };
    let out = run(&mut s, &[1], 3, decoder);
    errored(&out, "malformed push");
    assert_eq!(out.emitted_ids, [5, 6]);
    assert_eq!(out.completion.partial.raw_text, "5,6");
    assert_eq!(
        out.completion.partial.protocol_error.as_deref(),
        Some("scripted malformed push")
    );
    assert!(out.decoded_complete);
    assert_eq!(s.committed, [1, 5]);
}

#[test]
fn generation_finish_protocol_error_keeps_decoded_eos_and_suppresses_proposal() {
    let mut s = Script::new(&[2]);
    let decoder = DecodeScript {
        protocol_finish: true,
        ..Default::default()
    };
    let out = run(&mut s, &[1], 1, decoder);
    errored(&out, "malformed finish");
    assert_eq!(out.emitted_ids, [2]);
    assert_eq!(out.completion.partial.raw_text, "2");
    assert_eq!(
        out.completion.partial.protocol_error.as_deref(),
        Some("scripted malformed finish")
    );
    assert!(out.decoded_complete && out.limit.is_none());
    assert_eq!(s.committed, [1]);
}

#[test]
fn generation_finish_protocol_error_preserves_original_runtime_failure() {
    let mut s = Script::new(&[5]);
    s.fail_append = Some(2);
    let decoder = DecodeScript {
        protocol_finish: true,
        ..Default::default()
    };
    let out = run(&mut s, &[1], 3, decoder);
    errored(&out, "append failure");
    errored(&out, "malformed finish");
    assert_eq!(out.emitted_ids, [5]);
    assert_eq!(out.completion.partial.raw_text, "5");
    assert!(out.completion.partial.protocol_error.is_some());
    assert!(out.decoded_complete);
    assert_eq!(out.session_status, Status::Poisoned);
}

#[test]
fn generation_whole_prompt_preflight_preserves_old_state() {
    let mut invalid = vec![1; MAX_CHUNK + 1];
    invalid[MAX_CHUNK] = 99;
    for prompt in [Vec::new(), invalid, vec![1; 301]] {
        let mut s = Script::new(&[]);
        s.committed = vec![9];
        let out = run(&mut s, &prompt, 1, DecodeScript::default());
        assert!(matches!(out.completion.stop, StopReason::Error(_)));
        assert!(out.completion.proposal.is_none());
        assert_eq!(s.committed, [9]);
        assert_eq!(out.session_status, Status::Ready);
        assert!(s.appends.is_empty() && s.resets == 0);
    }
}

#[test]
fn generation_nonfinite_row_is_not_silently_masked_by_sampler() {
    let mut s = Script::new(&[]);
    s.committed = vec![1];
    s.logits = row(5);
    s.logits[6] = f32::NAN;
    let out = run(&mut s, &[1], 1, DecodeScript::default());
    errored(&out, "invalid native next-logit row");
    assert!(out.emitted_ids.is_empty());
    assert!(s.appends.is_empty() && s.resets == 0);
}

// Admission scripts bypass rendering; the public methods use the actual codec.
fn rendered(ids: &[u32]) -> RenderedPrompt {
    RenderedPrompt {
        text: "scripted complete prompt".into(),
        token_ids: ids.to_vec(),
    }
}

// Drive's fits boundary: only the explicit capacity rejection becomes false.
fn capacity_fits(admission: Result<PromptAdmission>) -> Result<bool> {
    match admission {
        Ok(_) => Ok(true),
        Err(error) if error.downcast_ref::<PromptCapacityExceeded>().is_some() => Ok(false),
        Err(error) => Err(error),
    }
}

#[test]
fn generation_preflight_capacity_rejection_is_typed_and_nonmutating() {
    for status in [Status::Ready, Status::Poisoned] {
        let mut s = Script::new(&[5, 6]);
        s.capacity = 4;
        s.committed = vec![9];
        s.logits = row(5);
        s.status = status;
        assert!(capacity_fits(admit_prompt(&s, rendered(&[1, 3]), 2)).unwrap());
        assert!(capacity_fits(admit_prompt(&s, rendered(&[1, 3, 4, 5]), 0)).unwrap());
        for (ids, budget) in [(vec![1, 3], 3), (vec![1; 5], 0)] {
            let error = admit_prompt(&s, rendered(&ids), budget)
                .unwrap_err()
                .context("caller admission context");
            assert_eq!(
                error.downcast_ref::<PromptCapacityExceeded>(),
                Some(&PromptCapacityExceeded {
                    required_tokens: 5,
                    capacity: 4,
                })
            );
            assert!(!capacity_fits(Err(error)).unwrap());
        }
        assert_eq!(s.committed, [9]);
        assert_eq!(s.logits, row(5));
        assert_eq!(s.predictions, VecDeque::from([5, 6]));
        assert_eq!(s.status, status);
        assert!(s.appends.is_empty() && s.resets == 0);
    }
}

#[test]
fn generation_preflight_invalid_even_oversized_is_not_capacity_rejection() {
    let mut s = Script::new(&[5]);
    s.capacity = 4;
    s.committed = vec![9];
    s.logits = row(5);
    let mut oversized_invalid = vec![1; MAX_CHUNK + 1];
    oversized_invalid[MAX_CHUNK] = 99;
    for ids in [Vec::new(), vec![99], oversized_invalid] {
        let error = capacity_fits(admit_prompt(&s, rendered(&ids), 8)).unwrap_err();
        assert!(error.downcast_ref::<PromptCapacityExceeded>().is_none());
    }
    // Public preflight returns render errors before admission. The scripted
    // boundary must propagate that ordinary error, not guess from its text.
    let render_error = Err(anyhow::anyhow!("scripted render failure")).context("render");
    assert!(capacity_fits(render_error).is_err());
    assert_eq!(s.committed, [9]);
    assert_eq!(s.logits, row(5));
    assert_eq!(s.predictions, VecDeque::from([5]));
    assert_eq!(s.status, Status::Ready);
    assert!(s.appends.is_empty() && s.resets == 0);
}

#[test]
fn generation_preflight_checked_overflow_is_not_capacity_rejection() {
    let mut s = Script::new(&[5]);
    s.capacity = usize::MAX;
    s.committed = vec![9];
    s.logits = row(5);
    let error = capacity_fits(admit_prompt(&s, rendered(&[1]), usize::MAX)).unwrap_err();
    assert!(error.downcast_ref::<PromptCapacityExceeded>().is_none());
    assert_eq!(s.committed, [9]);
    assert_eq!(s.logits, row(5));
    assert_eq!(s.predictions, VecDeque::from([5]));
    assert_eq!(s.status, Status::Ready);
    assert!(s.appends.is_empty() && s.resets == 0);
}

#[test]
fn generation_preflight_reserves_every_emission_without_mutating() {
    let mut s = Script::new(&[]);
    s.capacity = 4;
    s.committed = vec![9];
    s.status = Status::Poisoned;
    let admission = admit_prompt(&s, rendered(&[1, 3]), 2).unwrap();
    assert_eq!(admission.prompt().token_ids, [1, 3]);
    assert_eq!(admission.prompt().text, "scripted complete prompt");
    assert_eq!(admission.requested_new_tokens(), 2);
    assert_eq!(admission.reserved_attended_tokens(), 4);
    assert_eq!(admission.capacity(), 4);
    assert!(admit_prompt(&s, rendered(&[1, 3]), 3).is_err());
    let zero = admit_prompt(&s, rendered(&[1, 3, 4, 5]), 0).unwrap();
    assert_eq!(zero.reserved_attended_tokens(), 4);
    assert_eq!(zero.requested_new_tokens(), 0);
    assert_eq!(s.committed, [9]);
    assert_eq!(s.status, Status::Poisoned);
    assert!(s.appends.is_empty() && s.resets == 0);
}

#[test]
fn generation_replacement_rejects_full_prompt_and_render_failure_before_reset() {
    let mut invalid = vec![1; MAX_CHUNK + 1];
    invalid[MAX_CHUNK] = 99;
    for ids in [Vec::new(), invalid, vec![1; 301]] {
        let mut s = Script::new(&[]);
        s.committed = vec![9];
        let admission = admit_prompt(&s, rendered(&ids), 1);
        assert!(matches!(
            replace_admitted(&mut s, admission),
            ReplacementOutcome::RejectedUnchanged { .. }
        ));
        assert_eq!(s.committed, [9]);
        assert_eq!(s.status, Status::Ready);
        assert!(s.appends.is_empty() && s.resets == 0);
    }
    let mut s = Script::new(&[]);
    s.committed = vec![9];
    let outcome = replace_admitted(&mut s, Err(anyhow::anyhow!("scripted render failure")));
    assert!(
        matches!(outcome, ReplacementOutcome::RejectedUnchanged { error } if error.contains("render failure"))
    );
    assert_eq!(s.committed, [9]);
    assert!(s.appends.is_empty() && s.resets == 0);
}

#[test]
fn generation_replacement_rejects_reserve_and_overflow_before_reset() {
    for budget in [300, usize::MAX] {
        let mut s = Script::new(&[]);
        s.committed = vec![9];
        let admission = admit_prompt(&s, rendered(&[1]), budget);
        let outcome = replace_admitted(&mut s, admission);
        let ReplacementOutcome::RejectedUnchanged { error } = outcome else {
            panic!("must reject")
        };
        assert!(error.contains(if budget == usize::MAX {
            "overflows"
        } else {
            "exceeds"
        }));
        assert_eq!(s.committed, [9]);
        assert_eq!(s.status, Status::Ready);
        assert!(s.appends.is_empty() && s.resets == 0);
    }
}

#[test]
fn generation_replacement_success_is_empty_ready_not_prefilled() {
    let mut s = Script::new(&[]);
    s.committed = vec![1]; // Even an exact old prefix is deliberately reset.
    s.logits = row(5);
    let admission = admit_prompt(&s, rendered(&[1]), 0);
    let ReplacementOutcome::ReadyForAssignment { admission } = replace_admitted(&mut s, admission)
    else {
        panic!("reset must succeed")
    };
    assert_eq!(admission.prompt().token_ids, [1]);
    assert_eq!(s.resets, 1);
    assert!(s.committed.is_empty() && s.logits.is_empty() && s.appends.is_empty());
    assert_eq!(s.status, Status::Ready);
}

#[test]
fn generation_saved_prefix_uses_boundary_logits_and_only_prefills_suffix() {
    let mut s = Script::new(&[]);
    s.saved_prefix = Some(vec![1, 2]);
    s.committed = vec![1, 2, 9];
    s.logits = row(5);
    let output = run(&mut s, &[1, 2], 1, DecodeScript::default());
    assert_eq!(output.emitted_ids, [11], "exact seed uses saved EOS logits, not later-tail logits");
    assert_eq!(s.restored, 1);
    assert!(s.appends.is_empty() && s.resets == 0);
    s.committed.push(9);
    s.predictions.push_back(11);
    let output = run(&mut s, &[1, 2, 3], 1, DecodeScript::default());
    assert_eq!(output.emitted_ids, [11]);
    assert_eq!(s.appends, [vec![3]]);
    assert_eq!(s.restored, 2);
    s.predictions.push_back(11);
    let _ = run(&mut s, &[1], 1, DecodeScript::default());
    assert_eq!(s.resets, 1, "a shorter common prefix cannot use a later Mamba boundary");
    assert_eq!(s.appends.last().unwrap(), &[1]);
}

#[test]
fn generation_saved_prefix_refusal_cancel_and_restore_failure_preserve_control_contract() {
    let mut s = Script::new(&[]);
    s.saved_prefix = Some(vec![1, 2]);
    s.committed = vec![1, 2, 9];
    s.logits = row(5);
    s.capacity = 3;
    let invalid = admit_prompt(&s, rendered(&[1, 2]), 2);
    assert!(matches!(replace_admitted(&mut s, invalid), ReplacementOutcome::RejectedUnchanged { .. }));
    assert_eq!(s.restored, 0);
    let output = run_turn(&mut s, &mut Sampler::new(SamplingConfig::greedy(), 0).unwrap(),
        &[1, 2], 1, &mut || true, DecodeScript::default());
    assert!(matches!(output.completion.stop, StopReason::Cancelled));
    assert_eq!(s.committed, [1, 2, 9]);
    assert_eq!(s.restored, 0);
    s.fail_reset = true;
    let admitted = admit_prompt(&s, rendered(&[1, 2]), 1);
    assert!(matches!(replace_admitted(&mut s, admitted), ReplacementOutcome::ResetFailedTerminal { .. }));
    assert_eq!(s.status, Status::Poisoned);
    assert!(s.appends.is_empty());
}

#[test]
fn generation_replacement_reset_failure_is_terminal_not_rejected_unchanged() {
    let mut s = Script::new(&[]);
    s.committed = vec![9];
    s.fail_reset = true;
    let admission = admit_prompt(&s, rendered(&[1]), 2);
    let outcome = replace_admitted(&mut s, admission);
    assert!(matches!(outcome, ReplacementOutcome::ResetFailedTerminal {
            error, session_status: Status::Poisoned, committed_tokens: 1
        } if error.contains("reset failure")));
    assert_eq!(s.status, Status::Poisoned);
    assert_eq!(s.resets, 1);
    assert!(s.appends.is_empty());
    // Retained count is diagnostic; the outcome makes no rollback assertion.
    assert_eq!(s.committed, [9]);
}
