//! Scripted control-flow tests, NOT a CPU model, codec oracle or GPU witness.
//! Existing chat tests own token grammar/UTF-8 semantics; these scripts inject
//! failures and logits at the same private loop boundary used by production.
use super::super::chat::ExecProposal;
use super::*;
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
        }
    }
}
impl Runtime for Script {
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
fn generation_cancel_before_work_and_between_prefill_chunks() {
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
    s.cancel_after_append = Some((1, signal.clone()));
    let out = run_turn(
        &mut s,
        &mut sampler,
        &vec![1; MAX_CHUNK + 1],
        2,
        &mut || signal.get(),
        DecodeScript::default(),
    );
    assert_eq!(out.completion.stop, StopReason::Cancelled);
    assert_eq!(out.committed_tokens, MAX_CHUNK);
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
