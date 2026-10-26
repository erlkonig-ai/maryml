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
        ChatCodec, Completion, History, OutputParser, PartialOutput, RenderOptions, StopReason,
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
        let prompt = match self.codec.render_adapted(
            history,
            RenderOptions {
                enable_thinking: options.enable_thinking,
                medium_effort: false,
                add_generation_prompt: true,
                exec_tool: options.exec_tool,
            },
        ) {
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
    let mut emitted = Vec::new();
    let mut partial = PartialOutput::default();
    let mut decoded_complete = true;
    let run = (|| -> Result<(StopReason, Option<LimitCause>)> {
        // All prompt IDs, not just the first chunk, before ANY mutation.
        ensure!(
            !prompt.is_empty() && prompt.len() <= runtime.context_capacity(),
            "empty prompt or prompt exceeds session capacity"
        );
        ensure!(
            prompt
                .iter()
                .all(|&id| (id as usize) < runtime.vocabulary_size()),
            "prompt ID outside session vocabulary"
        );
        if max_new == 0 {
            return Ok((StopReason::Budget, Some(LimitCause::TokenBudget)));
        }
        if cancelled() {
            return Ok((StopReason::Cancelled, None));
        }
        let reuse =
            runtime.status() == Status::Ready && prompt.starts_with(runtime.committed_ids());
        if !reuse {
            runtime.reset().context("reset before prefix replacement")?;
        }
        let mut from = runtime.committed_ids().len();
        while from < prompt.len() {
            if cancelled() {
                return Ok((StopReason::Cancelled, None));
            }
            let end = from.saturating_add(MAX_CHUNK).min(prompt.len());
            runtime
                .append(&prompt[from..end])
                .context("prompt append")?;
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
            runtime.append(&[id]).context("generated-token append")?;
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
    TurnOutput {
        emitted_ids: emitted,
        completion,
        decoded_complete,
        limit,
        committed_tokens: runtime.committed_ids().len(),
        session_status: runtime.status(),
    }
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
