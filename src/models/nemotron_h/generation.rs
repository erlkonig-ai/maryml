//! Synchronous, greedy, text-only Nemotron turns over one native observation.
//!
//! Emitted IDs (including EOS) are authoritative and are NOT the attended
//! prefix. Rendering may change earlier tokens: reuse requires the complete
//! committed prefix, otherwise reset must succeed before prefill. Cancellation
//! is cooperative at completed chunk boundaries, never a GPU interrupt/drain.
//! No executor, sampling policy switch, history truncation or implicit token mask.
//!
//! Available with BOTH nemotron-cuda and tokenizer. Runtime failures preserve
//! the Session's actual state; a failed synchronization is not a drainage claim.

use super::{
    chat::{
        ChatCodec, Completion, History, OutputParser, PartialOutput, RenderOptions, RenderedPrompt,
        StopReason,
    },
    session::{MAX_CHUNK, Session, Status},
};
use crate::sampling::{Sampler, SamplingConfig};
use anyhow::{Context, Result, ensure};
use triblespace::prelude::BlobStoreGet;

#[derive(Clone, Copy, Debug)]
pub struct TurnOptions {
    pub enable_thinking: bool,
    /// Advertises the existing proposal grammar and permits returning a parsed
    /// proposal. False also suppresses unsolicited proposals; never executes.
    pub exec_tool: bool,
    /// External authority's emission bound, INCLUDING a generated EOS.
    pub max_new_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitCause {
    TokenBudget,
    Context,
}

/// A valid rendered prompt and checked generation reserve exceed context.
/// `preflight()` preserves this type inside its `anyhow::Error`: callers may
/// downcast to distinguish insufficient capacity from malformed input/render
/// failures. Arithmetic overflow is a separate error, NOT this rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptCapacityExceeded {
    pub required_tokens: usize,
    pub capacity: usize,
}
impl std::fmt::Display for PromptCapacityExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prompt plus generation reserve ({}) exceeds session capacity ({})",
            self.required_tokens, self.capacity
        )
    }
}
impl std::error::Error for PromptCapacityExceeded {}

/// Immutable, descriptive admission facts, not a transferable execution permit.
/// No History is retained. Replacement re-renders/revalidates its own proposal.
#[derive(Clone, Debug)]
pub struct PromptAdmission {
    prompt: RenderedPrompt,
    requested_new_tokens: usize,
    reserved_attended_tokens: usize,
    capacity: usize,
}
impl PromptAdmission {
    pub fn prompt(&self) -> &RenderedPrompt {
        &self.prompt
    }
    pub fn requested_new_tokens(&self) -> usize {
        self.requested_new_tokens
    }
    /// Conservative bound: prompt length + EVERY requested emission, including
    /// final/EOS tokens that turn() normally does not append. No subtraction or
    /// implicit max(1); zero emission reserve is zero. Checked for usize overflow.
    pub fn reserved_attended_tokens(&self) -> usize {
        self.reserved_attended_tokens
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[derive(Clone, Debug)]
pub enum ReplacementOutcome {
    /// No reset/append occurred; the caller must not assign the proposed History.
    RejectedUnchanged { error: String },
    /// Reset failed. Caller MUST retire/stop this backend session, not retry old
    /// history. No rollback/drain or reusable prefix claim; status/count remain
    /// actual diagnostic observations (the native Session poisons on failure).
    ResetFailedTerminal {
        error: String,
        session_status: Status,
        committed_tokens: usize,
    },
    /// Reset succeeded: Ready with empty attended prefix, NOT prefilled. The
    /// caller may now assign its host History; this owner never stores that copy.
    ReadyForAssignment { admission: PromptAdmission },
}

#[derive(Clone, Debug)]
pub struct TurnOutput {
    /// Every sampled ID, including EOS and an ID whose subsequent decode fails.
    pub emitted_ids: Vec<u32>,
    pub completion: Completion,
    /// The returned partial covers all emitted IDs. This does NOT promise
    /// lossless UTF-8 display: the codec can display an incomplete byte sequence
    /// as U+FFFD. A decoder error keeps the last valid partial and sets false.
    /// Successfully decoded malformed protocol stays complete, with its error
    /// in the partial and an Error stop; no proposal is returned.
    pub decoded_complete: bool,
    /// Budget distinguishes the requested emission bound from context exhaustion.
    pub limit: Option<LimitCause>,
    pub committed_tokens: usize,
    /// Poisoned counts are diagnostic, not reusable cache claims.
    pub session_status: Status,
}

pub struct Generator<S> {
    session: Session<S>,
    codec: ChatCodec,
    sampler: Sampler,
}

impl<S: BlobStoreGet> Generator<S> {
    /// Rebuild the codec from exactly the Session's held native observation and
    /// explicit binding/base/pre-tokenizer. No default collection or new reader.
    pub fn new(session: Session<S>) -> Result<Self> {
        let source = session.source();
        let snapshot = session.snapshot();
        let codec = ChatCodec::rebuild(
            snapshot.facts(),
            snapshot.store(),
            source.tokenizer_binding,
            source.tokenizer_base,
            source.pre_tokenizer,
        )?;
        Ok(Self {
            session,
            codec,
            // Seed is immaterial for greedy draws; reuse the established sampler.
            sampler: Sampler::new(SamplingConfig::greedy(), 0)?,
        })
    }

    pub fn session(&self) -> &Session<S> {
        &self.session
    }

    fn render_prompt(&self, history: &History, options: TurnOptions) -> Result<RenderedPrompt> {
        self.codec.render_adapted(
            history,
            RenderOptions {
                enable_thinking: options.enable_thinking,
                medium_effort: false,
                add_generation_prompt: true,
                exec_tool: options.exec_tool,
            },
        )
    }

    /// Render the COMPLETE proposed history with the real codec, validate all
    /// IDs and capacity, then require prompt_len + max_new_tokens <= capacity.
    /// This optional strict reservation is conservative, not turn()'s behavior:
    /// turn() still accepts a fitting prompt and reports later context limits.
    /// No runtime reset, append, cache mutation or retained host History.
    /// Only a valid prompt with a non-overflowing sum may return the downcastable
    /// `PromptCapacityExceeded`; propagate all other errors rather than treating
    /// them as a capacity miss. An invalid oversized prompt is still invalid.
    pub fn preflight(&self, history: &History, options: TurnOptions) -> Result<PromptAdmission> {
        admit_prompt(
            &self.session,
            self.render_prompt(history, options).context("render")?,
            options.max_new_tokens,
        )
    }

    /// Validate the same COMPLETE proposal and conservative generation reserve
    /// BEFORE attempting reset, even for zero budget or an already-matching
    /// prefix. Success permits host-history assignment, not immediate sampling;
    /// a later turn must prefill it. Failure after reset is NOT rollback/drain.
    /// Inspect session() for actual diagnostic count/status after any outcome.
    pub fn prepare_history_replacement(
        &mut self,
        history: &History,
        options: TurnOptions,
    ) -> ReplacementOutcome {
        let admission = self.preflight(history, options);
        replace_admitted(&mut self.session, admission)
    }

    /// A turn always returns its partial/result, even on render, reset, runtime
    /// or decoding error. Panics/fatal process failures are not caught here.
    /// Validate the whole rendered prompt BEFORE resetting/appending; rejection
    /// leaves old state intact. Zero budget performs no model work, even reset.
    /// History reasoning stays in the render; medium_effort is always false.
    /// A failed replacement after reset does not preserve the old history.
    pub fn turn(
        &mut self,
        history: &History,
        options: TurnOptions,
        mut cancelled: impl FnMut() -> bool,
    ) -> TurnOutput {
        #[cfg(feature = "nemotron-timing")]
        let render_started = tracing::enabled!(target: "mary.nemotron.timing", tracing::Level::INFO)
            .then(std::time::Instant::now);
        let rendered = self.render_prompt(history, options);
        #[cfg(feature = "nemotron-timing")]
        if let Some(started) = render_started {
            tracing::info!(target: "mary.nemotron.timing", phase = "render",
                wall_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                ok = rendered.is_ok(),
                prompt_tokens = rendered.as_ref().map_or(0, |p| p.token_ids.len()));
        }
        let prompt = match rendered {
            Ok(prompt) => prompt,
            Err(error) => return rejected(&self.session, format!("render: {error:#}")),
        };
        let parser =
            OutputParser::new(&self.codec, options.enable_thinking, options.max_new_tokens);
        let mut output = run_turn(
            &mut self.session,
            &mut self.sampler,
            &prompt.token_ids,
            options.max_new_tokens,
            &mut cancelled,
            parser,
        );
        if !options.exec_tool {
            output.completion.proposal = None;
        }
        output
    }
}

// Private control-loop seams solely for scripted host tests; not a model,
// backend registry or numerical oracle. Production uses the actual Session
// and OutputParser below, unchanged kernels and their own state contracts.
trait Runtime {
    fn context_capacity(&self) -> usize;
    fn vocabulary_size(&self) -> usize;
    fn committed_ids(&self) -> &[u32];
    fn status(&self) -> Status;
    fn last_logits(&self) -> Option<&[f32]>;
    fn reset(&mut self) -> Result<()>;
    fn append(&mut self, ids: &[u32]) -> Result<()>;
}
impl<S: BlobStoreGet> Runtime for Session<S> {
    fn context_capacity(&self) -> usize {
        Session::capacity(self)
    }
    fn vocabulary_size(&self) -> usize {
        Session::vocab_size(self)
    }
    fn committed_ids(&self) -> &[u32] {
        Session::committed_ids(self)
    }
    fn status(&self) -> Status {
        Session::status(self)
    }
    fn last_logits(&self) -> Option<&[f32]> {
        Session::last_logits(self).map(|v| v.values)
    }
    fn reset(&mut self) -> Result<()> {
        Session::reset(self).map_err(anyhow::Error::msg)
    }
    fn append(&mut self, ids: &[u32]) -> Result<()> {
        self.append_ids(ids).map(|_| ()).map_err(anyhow::Error::msg)
    }
}
trait Decoder {
    fn push(&mut self, id: u32) -> Result<PartialOutput>;
    fn finish(self, stop: StopReason) -> Result<Completion>;
}
impl Decoder for OutputParser<'_> {
    fn push(&mut self, id: u32) -> Result<PartialOutput> {
        ensure!(
            OutputParser::push(self, &[id]),
            "parser emission bound exceeded"
        );
        self.snapshot()
    }
    fn finish(self, stop: StopReason) -> Result<Completion> {
        OutputParser::finish(self, stop)
    }
}

fn validate_prompt(runtime: &impl Runtime, ids: &[u32]) -> Result<()> {
    ensure!(
        !ids.is_empty() && ids.len() <= runtime.context_capacity(),
        "empty prompt or prompt exceeds session capacity"
    );
    validate_prompt_ids(runtime, ids)
}

fn validate_prompt_ids(runtime: &impl Runtime, ids: &[u32]) -> Result<()> {
    ensure!(
        ids.iter()
            .all(|&id| (id as usize) < runtime.vocabulary_size()),
        "prompt ID outside session vocabulary"
    );
    Ok(())
}

fn admit_prompt(
    runtime: &impl Runtime,
    prompt: RenderedPrompt,
    requested_new_tokens: usize,
) -> Result<PromptAdmission> {
    // Capacity must not hide malformed data: scan the complete prompt's IDs
    // even when its length already exceeds context. turn() keeps its existing
    // validation behavior; this strict preflight supplies the typed rejection.
    ensure!(!prompt.token_ids.is_empty(), "empty prompt");
    validate_prompt_ids(runtime, &prompt.token_ids)?;
    let reserved_attended_tokens = prompt
        .token_ids
        .len()
        .checked_add(requested_new_tokens)
        .context("prompt plus generation reserve overflows usize")?;
    if reserved_attended_tokens > runtime.context_capacity() {
        return Err(PromptCapacityExceeded {
            required_tokens: reserved_attended_tokens,
            capacity: runtime.context_capacity(),
        }
        .into());
    }
    Ok(PromptAdmission {
        prompt,
        requested_new_tokens,
        reserved_attended_tokens,
        capacity: runtime.context_capacity(),
    })
}

fn replace_admitted(
    runtime: &mut impl Runtime,
    admission: Result<PromptAdmission>,
) -> ReplacementOutcome {
    let admission = match admission {
        Ok(value) => value,
        Err(error) => {
            return ReplacementOutcome::RejectedUnchanged {
                error: format!("{error:#}"),
            };
        }
    };
    match runtime.reset() {
        Ok(()) => ReplacementOutcome::ReadyForAssignment { admission },
        Err(error) => ReplacementOutcome::ResetFailedTerminal {
            error: format!("{error:#}"),
            session_status: runtime.status(),
            committed_tokens: runtime.committed_ids().len(),
        },
    }
}

fn rejected(runtime: &impl Runtime, error: String) -> TurnOutput {
    TurnOutput {
        emitted_ids: Vec::new(),
        completion: Completion {
            partial: PartialOutput::default(),
            stop: StopReason::Error(error),
            proposal: None,
        },
        decoded_complete: true,
        limit: None,
        committed_tokens: runtime.committed_ids().len(),
        session_status: runtime.status(),
    }
}

fn run_turn(
    runtime: &mut impl Runtime,
    sampler: &mut Sampler,
    prompt: &[u32],
    max_new: usize,
    cancelled: &mut impl FnMut() -> bool,
    mut decoder: impl Decoder,
) -> TurnOutput {
    #[cfg(feature = "nemotron-timing")]
    let mut timing = TurnTiming::start(runtime.committed_ids().len());
    let mut emitted = Vec::new();
    let mut partial = PartialOutput::default();
    let mut decoded_complete = true;
    let run = (|| -> Result<(StopReason, Option<LimitCause>)> {
        // All prompt IDs, not just the first chunk, before ANY mutation.
        validate_prompt(runtime, prompt)?;
        if max_new == 0 {
            return Ok((StopReason::Budget, Some(LimitCause::TokenBudget)));
        }
        if cancelled() {
            return Ok((StopReason::Cancelled, None));
        }
        let reuse =
            runtime.status() == Status::Ready && prompt.starts_with(runtime.committed_ids());
        if !reuse {
            #[cfg(feature = "nemotron-timing")]
            let started = timing.as_ref().map(|_| std::time::Instant::now());
            let reset = runtime.reset();
            #[cfg(feature = "nemotron-timing")]
            if let (Some(timing), Some(started)) = (&mut timing, started) {
                timing.reset.record(started.elapsed(), 0, reset.is_ok());
            }
            reset.context("reset before prefix replacement")?;
        }
        let mut from = runtime.committed_ids().len();
        #[cfg(feature = "nemotron-timing")]
        if let Some(timing) = &mut timing {
            timing.reused_tokens = from;
        }
        while from < prompt.len() {
            if cancelled() {
                return Ok((StopReason::Cancelled, None));
            }
            let end = from.saturating_add(MAX_CHUNK).min(prompt.len());
            #[cfg(feature = "nemotron-timing")]
            let started = timing.as_ref().map(|_| std::time::Instant::now());
            let append = {
                // Synchronous append: this entered guard never crosses an async
                // suspension. Role comes from the actual generation branch, not T.
                #[cfg(feature = "nemotron-timing")]
                let _append = timing.as_ref().map(|_| {
                    tracing::info_span!(
                        target: "mary.nemotron.timing", "native_append", role = "prompt",
                        committed_before = from, attempted_tokens = end - from,
                    ).entered()
                });
                runtime.append(&prompt[from..end])
            };
            #[cfg(feature = "nemotron-timing")]
            if let (Some(timing), Some(started)) = (&mut timing, started) {
                timing.prompt.record(started.elapsed(), end - from, append.is_ok());
            }
            append.context("prompt append")?;
            from = end;
            if cancelled() {
                return Ok((StopReason::Cancelled, None));
            }
        }
        // An exact already-committed prompt uses last_logits, never append([]).
        loop {
            if cancelled() {
                return Ok((StopReason::Cancelled, None));
            }
            let logits = runtime
                .last_logits()
                .context("ready prefix has no next logits")?;
            // Preserve Session's strict finite-row contract rather than letting
            // Sampler's general nonfinite-exclusion policy hide a runtime fault.
            ensure!(
                logits.len() == runtime.vocabulary_size() && logits.iter().all(|v| v.is_finite()),
                "invalid native next-logit row"
            );
            let id = u32::try_from(sampler.token(logits)?).context("sampled ID exceeds u32")?;
            emitted.push(id); // Retain even if parser/decoder subsequently fails.
            partial = match decoder.push(id) {
                Ok(value) => value,
                Err(error) => {
                    decoded_complete = false;
                    return Err(error.context("decode emission"));
                }
            };
            if let Some(error) = &partial.protocol_error {
                anyhow::bail!("output protocol: {error}");
            }
            // EOS at the last emission slot is EOS, not Budget. No EOS feedback.
            if matches!(id, 2 | 11) {
                return Ok((StopReason::Eos { id }, None));
            }
            if emitted.len() == max_new {
                return Ok((StopReason::Budget, Some(LimitCause::TokenBudget)));
            }
            if cancelled() {
                return Ok((StopReason::Cancelled, None));
            }
            if runtime.committed_ids().len() == runtime.context_capacity() {
                return Ok((StopReason::Budget, Some(LimitCause::Context)));
            }
            // Only request a successor when another emission is allowed. A
            // failed append retains the emitted ID and Session's actual status.
            #[cfg(feature = "nemotron-timing")]
            let started = timing.as_ref().map(|_| std::time::Instant::now());
            let append = {
                #[cfg(feature = "nemotron-timing")]
                let _append = timing.as_ref().map(|_| {
                    tracing::info_span!(
                        target: "mary.nemotron.timing", "native_append", role = "generated",
                        committed_before = runtime.committed_ids().len(), attempted_tokens = 1_usize,
                    ).entered()
                });
                runtime.append(&[id])
            };
            #[cfg(feature = "nemotron-timing")]
            if let (Some(timing), Some(started)) = (&mut timing, started) {
                timing.generated.record(started.elapsed(), 1, append.is_ok());
            }
            append.context("generated-token append")?;
        }
    })();
    let (stop, mut limit) = match run {
        Ok(value) => value,
        Err(error) => (StopReason::Error(format!("{error:#}")), None),
    };
    let fallback = |stop| Completion {
        partial: partial.clone(),
        stop,
        proposal: None,
    };
    let mut completion = if decoded_complete {
        match decoder.finish(stop.clone()) {
            Ok(mut value) => {
                if let Some(error) = &value.partial.protocol_error {
                    // Grammar failure is NOT text-decoding failure. Preserve
                    // this newest partial and any original runtime error.
                    let message = match &stop {
                        StopReason::Error(original) => {
                            format!("{original}; output protocol: {error}")
                        }
                        _ => format!("output protocol: {error}"),
                    };
                    value.stop = StopReason::Error(message);
                    value.proposal = None;
                }
                value
            }
            Err(error) => {
                decoded_complete = false;
                let message = match stop {
                    StopReason::Error(original) => format!("{original}; output finish: {error:#}"),
                    _ => format!("output finish: {error:#}"),
                };
                fallback(StopReason::Error(message))
            }
        }
    } else {
        fallback(stop)
    };
    if !matches!(completion.stop, StopReason::Eos { .. }) {
        completion.proposal = None;
    }
    if !matches!(completion.stop, StopReason::Budget) {
        limit = None;
    }
    #[cfg(feature = "nemotron-timing")]
    if let Some(timing) = timing {
        timing.emit(prompt.len(), emitted.len(), runtime.committed_ids().len(),
            matches!(completion.stop, StopReason::Error(_)));
    }
    TurnOutput {
        emitted_ids: emitted,
        completion,
        decoded_complete,
        limit,
        committed_tokens: runtime.committed_ids().len(),
        session_status: runtime.status(),
    }
}

// Ephemeral counters for this one call, not another trace/history model. Timers
// wrap the existing synchronous returns, including their fences, reads, uploads
// and possible first-use compilation. They are NOT GPU execution durations.
// No feature or no interested subscriber => no Instant reads, no extra fences.
#[cfg(feature = "nemotron-timing")]
#[derive(Default)]
struct CallTiming {
    wall_ns: u64,
    calls: usize,
    completed_calls: usize,
    attempted_tokens: usize,
    completed_tokens: usize,
}

#[cfg(feature = "nemotron-timing")]
impl CallTiming {
    fn record(&mut self, elapsed: std::time::Duration, tokens: usize, ok: bool) {
        self.wall_ns = self.wall_ns.saturating_add(elapsed.as_nanos().min(u64::MAX as u128) as u64);
        self.calls = self.calls.saturating_add(1);
        self.attempted_tokens = self.attempted_tokens.saturating_add(tokens);
        if ok {
            self.completed_calls = self.completed_calls.saturating_add(1);
            self.completed_tokens = self.completed_tokens.saturating_add(tokens);
        }
    }
}

#[cfg(feature = "nemotron-timing")]
struct TurnTiming {
    started: std::time::Instant,
    committed_before: usize,
    reused_tokens: usize,
    reset: CallTiming,
    prompt: CallTiming,
    generated: CallTiming,
}

#[cfg(feature = "nemotron-timing")]
impl TurnTiming {
    fn start(committed_before: usize) -> Option<Self> {
        tracing::enabled!(target: "mary.nemotron.timing", tracing::Level::INFO).then(|| Self {
            started: std::time::Instant::now(), committed_before, reused_tokens: 0,
            reset: CallTiming::default(), prompt: CallTiming::default(),
            generated: CallTiming::default(),
        })
    }

    fn emit(self, prompt_tokens: usize, emitted_tokens: usize, attended_tokens: usize, error: bool) {
        tracing::info!(target: "mary.nemotron.timing", phase = "generation",
            wall_ns = self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            committed_before = self.committed_before, reused_tokens = self.reused_tokens,
            prompt_tokens, emitted_tokens, attended_tokens, error,
            reset_ns = self.reset.wall_ns, reset_calls = self.reset.calls,
            reset_completed = self.reset.completed_calls,
            prompt_append_ns = self.prompt.wall_ns, prompt_append_calls = self.prompt.calls,
            prompt_append_completed = self.prompt.completed_calls,
            prompt_attempted_tokens = self.prompt.attempted_tokens,
            prompt_attended_tokens = self.prompt.completed_tokens,
            generated_append_ns = self.generated.wall_ns,
            generated_append_calls = self.generated.calls,
            generated_append_completed = self.generated.completed_calls,
            generated_attempted_tokens = self.generated.attempted_tokens,
            generated_attended_tokens = self.generated.completed_tokens);
    }
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
