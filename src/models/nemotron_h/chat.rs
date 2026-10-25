//! Bounded text-only Super chat protocol (HF revision da5b2650).
//!
//! The reference template's `truncate_history_thinking` is deliberately false.
//! Rendering is not append-only: callers must compare token prefixes before
//! reusing inference state. This module neither owns that state nor executes
//! proposals. Literal marker spellings in content are not control tokens.

use anyhow::{Context, Result, ensure};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecProposal {
    pub command: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    User(String),
    Assistant {
        reasoning: String,
        answer: String,
        exec: Option<ExecProposal>,
    },
    /// Opaque labels supplied by the authority adapter, not invented IDs.
    ToolResult {
        label: String,
        status: String,
        content: String,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct History {
    pub system: String,
    pub messages: Vec<Message>,
}

/// All choices are explicit. History reasoning is never implicitly truncated.
#[derive(Clone, Copy, Debug)]
pub struct RenderOptions {
    pub enable_thinking: bool,
    /// Reference option: appending a user moves this suffix off the prior user.
    /// Initial resident mode sets false; any reuse still needs prefix equality.
    pub medium_effort: bool,
    pub add_generation_prompt: bool,
    pub exec_tool: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub text: String,
    pub token_ids: Vec<u32>,
}

// Source IDs, not new schema identities. The pinned tokenizer reserves 0..1000;
// IDs 12..17 are added tokens with special=false, but remain protocol controls.
const MARKERS: [&str; 18] = [
    "<unk>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "[TOOL_CALLS]",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
];
const MAX_HISTORY_BYTES: usize = 1_048_576;
const EXEC_SCHEMA: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>exec</name>\n<description>Propose a shell command.</description>\n<parameters>\n<parameter>\n<name>command</name>\n<type>string</type>\n<description>Shell command text.</description>\n</parameter>\n<required>[\"command\"]</required>\n</parameters>\n</function>\n</tools>";
const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
const CALL_PREFIX: &str = "\n<function=exec>\n<parameter=command>\n";
const CALL_SUFFIX: &str = "\n</parameter>\n</function>\n";

#[derive(Debug)]
enum Part {
    Control(u32),
    Content(String),
}
fn content(parts: &mut Vec<Part>, text: impl Into<String>) {
    parts.push(Part::Content(text.into()));
}
fn control(parts: &mut Vec<Part>, id: u32) {
    parts.push(Part::Control(id));
}
fn open(parts: &mut Vec<Part>, role: &str) {
    control(parts, 10);
    content(parts, format!("{role}\n"));
}
fn close(parts: &mut Vec<Part>) {
    control(parts, 11);
    content(parts, "\n");
}

fn valid_command(command: &str) -> Result<()> {
    // No XML escaping is defined by the checkpoint. Never guess which closing
    // delimiter was shell data, nor silently trim/escape that data.
    for delimiter in [
        "</parameter>",
        "</function>",
        "</tool_call>",
        "<parameter=",
        "<function=",
        "<tool_call>",
    ] {
        ensure!(
            !command.contains(delimiter),
            "ambiguous command delimiter: {delimiter}"
        );
    }
    ensure!(!command.contains('\0'), "NUL is not shell command text");
    Ok(())
}

fn render_parts(history: &History, options: RenderOptions, adapted: bool) -> Result<Vec<Part>> {
    let mut bytes = history.system.len();
    for message in &history.messages {
        bytes = bytes
            .checked_add(match message {
                Message::User(s) => s.len(),
                Message::Assistant {
                    reasoning,
                    answer,
                    exec,
                } => reasoning
                    .len()
                    .saturating_add(answer.len())
                    .saturating_add(exec.as_ref().map_or(0, |e| e.command.len())),
                Message::ToolResult {
                    label,
                    status,
                    content,
                } => label
                    .len()
                    .saturating_add(status.len())
                    .saturating_add(content.len()),
            })
            .context("history size overflow")?;
    }
    ensure!(
        bytes <= MAX_HISTORY_BYTES && history.messages.len() <= 4096,
        "history bound exceeded"
    );
    let mut parts = Vec::new();
    open(&mut parts, "system");
    content(&mut parts, &history.system);
    if options.exec_tool {
        if !history.system.is_empty() {
            content(&mut parts, "\n\n");
        }
        content(&mut parts, EXEC_SCHEMA);
        // The example is instructional text, not an actual tool-call frame.
        content(&mut parts, TOOL_INSTRUCTIONS);
    }
    close(&mut parts);
    let last_user = history
        .messages
        .iter()
        .rposition(|m| matches!(m, Message::User(_)));
    for (i, message) in history.messages.iter().enumerate() {
        match message {
            Message::User(text) => {
                open(&mut parts, "user");
                content(&mut parts, text);
                if options.medium_effort && last_user == Some(i) {
                    content(&mut parts, "\n\n{reasoning effort: efficient}");
                }
                close(&mut parts);
            }
            Message::Assistant {
                reasoning,
                answer,
                exec,
            } => {
                open(&mut parts, "assistant");
                control(&mut parts, 12);
                if !reasoning.trim().is_empty() {
                    content(&mut parts, format!("\n{reasoning}"));
                }
                control(&mut parts, 13);
                // Reference trims the combined body, whose first bytes are
                // always <think>; thus only the answer's trailing space goes.
                content(&mut parts, answer.trim_end());
                if let Some(exec) = exec {
                    valid_command(&exec.command)?;
                    content(&mut parts, "\n");
                    control(&mut parts, 14);
                    content(&mut parts, CALL_PREFIX);
                    content(&mut parts, &exec.command);
                    content(&mut parts, CALL_SUFFIX);
                    control(&mut parts, 15);
                    content(&mut parts, "\n");
                }
                close(&mut parts);
            }
            Message::ToolResult {
                label,
                status,
                content: text,
            } => {
                let preceding_tool =
                    i > 0 && matches!(history.messages[i - 1], Message::ToolResult { .. });
                if !preceding_tool && (adapted || i > 0) {
                    open(&mut parts, "user");
                }
                control(&mut parts, 16);
                content(&mut parts, "\n");
                if adapted {
                    // JSON here is a CONTENT envelope for result identity, not
                    // the model's XML-like exec proposal grammar.
                    content(
                        &mut parts,
                        serde_json::to_string(&serde_json::json!({
                            "label": label, "status": status, "content": text,
                        }))?,
                    );
                } else {
                    content(&mut parts, text);
                }
                content(&mut parts, "\n");
                control(&mut parts, 17);
                content(&mut parts, "\n");
                if !matches!(
                    history.messages.get(i + 1),
                    Some(Message::ToolResult { .. })
                ) {
                    close(&mut parts);
                }
            }
        }
    }
    if options.add_generation_prompt {
        open(&mut parts, "assistant");
        control(&mut parts, 12);
        if options.enable_thinking {
            content(&mut parts, "\n");
        } else {
            control(&mut parts, 13);
        }
    }
    Ok(parts)
}

fn rendered_text(parts: &[Part]) -> String {
    let mut text = String::new();
    for part in parts {
        match part {
            Part::Control(id) => text.push_str(MARKERS[*id as usize]),
            Part::Content(s) => text.push_str(s),
        }
    }
    text
}

/// Literal reference text only, never safe-to-feed tokenization. Tool result
/// labels/status are dropped and a leading orphan retains the upstream missing
/// opener. Use `render_adapted` for actual prompts. Embedded assistant marker
/// strings are treated as typed content, not upstream's implicit reasoning.
pub fn render_reference_text(history: &History, options: RenderOptions) -> Result<String> {
    Ok(rendered_text(&render_parts(history, options, false)?))
}

#[cfg(feature = "tokenizer")]
pub struct ChatCodec {
    tokenizer: tokenizers::Tokenizer,
    forbidden: std::collections::HashSet<u32>,
}

#[cfg(feature = "tokenizer")]
impl ChatCodec {
    /// Build both content encoding and output decoding from the SAME explicit
    /// native binding selection. Do not use the generic builder on its wrapper.
    pub fn rebuild(
        facts: &triblespace::prelude::TribleSet,
        blobs: &impl triblespace::prelude::BlobStoreGet,
        binding: triblespace::prelude::Id,
        base: triblespace::prelude::Id,
        pre_tokenizer: triblespace::prelude::Id,
    ) -> Result<Self> {
        let tokenizer =
            super::ingress::tokenizer_binding::rebuild(facts, blobs, binding, base, pre_tokenizer)?;
        for (id, marker) in MARKERS.iter().enumerate() {
            ensure!(
                tokenizer.token_to_id(marker) == Some(id as u32)
                    && tokenizer.id_to_token(id as u32).as_deref() == Some(*marker),
                "wrong Super marker ID: {marker}"
            );
        }
        let added = tokenizer.get_added_tokens_decoder();
        ensure!(
            added.len() == 1000 && added.keys().all(|id| *id < 1000),
            "wrong Super reserved token range"
        );
        // Establish the range as an observed pinned fact, not a generic
        // tokenizer convention. Six ordinary added tokens are also controls.
        for (&id, token) in &added {
            ensure!(
                token.special == !(12..=17).contains(&id),
                "wrong Super added-token flag at {id}"
            );
        }
        let mut forbidden: std::collections::HashSet<u32> = added
            .iter()
            .filter_map(|(&id, token)| token.special.then_some(id))
            .collect();
        forbidden.extend(0..MARKERS.len() as u32);
        Ok(Self {
            tokenizer,
            forbidden,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("encode Super text: {e}"))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode Super text: {e}"))
    }

    // ByteLevel's display decoder uses UTF-8 replacement for incomplete bytes.
    // A shell proposal must not inherit that lossy display behavior. Invert the
    // pinned ByteLevel alphabet (the same mapping used by tokenizers 0.21.4),
    // and require complete UTF-8 before accepting command text.
    fn decode_command(&self, ids: &[u32]) -> Result<String> {
        let mut alphabet = std::collections::HashMap::new();
        let mut extra = 256;
        for byte in 0..=255u8 {
            let scalar =
                if (b'!'..=b'~').contains(&byte) || (0xa1..=0xac).contains(&byte) || byte >= 0xae {
                    byte as u32
                } else {
                    let scalar = extra;
                    extra += 1;
                    scalar
                };
            alphabet.insert(char::from_u32(scalar).unwrap(), byte);
        }
        let mut bytes = Vec::new();
        for &id in ids {
            ensure!(!self.forbidden.contains(&id), "control in command bytes");
            for ch in self
                .tokenizer
                .id_to_token(id)
                .context("unknown command token")?
                .chars()
            {
                bytes.push(*alphabet.get(&ch).context("non-ByteLevel command token")?);
            }
        }
        Ok(String::from_utf8(bytes).context("incomplete or invalid command UTF-8")?)
    }

    /// Lossless, marker-safe content policy. Native encoding is unchanged when
    /// it contains no reserved IDs. Otherwise Unicode-scalar boundaries prevent
    /// BOTH added-token matching and BPE from reconstructing a marker token.
    /// The full decode is verified; unsupported content fails closed.
    pub fn encode_content(&self, text: &str) -> Result<Vec<u32>> {
        ensure!(text.len() <= MAX_HISTORY_BYTES, "content bound exceeded");
        let mut ids = self.encode(text)?;
        if ids.iter().any(|id| self.forbidden.contains(id)) {
            ids.clear();
            for ch in text.chars() {
                ids.extend(self.encode(ch.encode_utf8(&mut [0; 4]))?);
            }
        }
        ensure!(
            ids.iter().all(|id| !self.forbidden.contains(id)),
            "content encoded a reserved token"
        );
        ensure!(
            self.decode(&ids)? == text,
            "content encoding is not lossless"
        );
        Ok(ids)
    }

    /// Named adaptation: frames orphan results and preserves opaque labels and
    /// statuses in content envelopes. All content (including prior assistant
    /// text and tool documentation examples) uses the marker-safe policy.
    pub fn render_adapted(
        &self,
        history: &History,
        options: RenderOptions,
    ) -> Result<RenderedPrompt> {
        let parts = render_parts(history, options, true)?;
        let text = rendered_text(&parts);
        let mut token_ids = Vec::new();
        // Coalesce adjacent content so benign text keeps normal BPE boundaries.
        let mut pending = String::new();
        for part in parts {
            match part {
                Part::Content(s) => pending.push_str(&s),
                Part::Control(id) => {
                    token_ids.extend(self.encode_content(&pending)?);
                    pending.clear();
                    token_ids.push(id);
                }
            }
        }
        token_ids.extend(self.encode_content(&pending)?);
        ensure!(
            self.decode(&token_ids)? == text,
            "rendered token/text mismatch"
        );
        Ok(RenderedPrompt { text, token_ids })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    Eos { id: u32 },
    Budget,
    Cancelled,
    Error(String),
}

/// A partial snapshot NEVER contains a proposal. `raw_text` includes markers
/// and incomplete command text, even on malformed or interrupted output.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartialOutput {
    pub reasoning: String,
    pub answer: String,
    pub raw_text: String,
    pub protocol_error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub partial: PartialOutput,
    pub stop: StopReason,
    pub proposal: Option<ExecProposal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Reasoning,
    Answer,
    Call,
    AfterCall,
    End,
}

#[cfg(feature = "tokenizer")]
pub struct OutputParser<'a> {
    codec: &'a ChatCodec,
    phase: Phase,
    max_tokens: usize,
    ids: Vec<u32>,
    reasoning: Vec<u32>,
    answer: Vec<u32>,
    call: Vec<u32>,
    suffix: Vec<u32>,
    terminal: Option<u32>,
    error: Option<String>,
    call_closed: bool,
}

#[cfg(feature = "tokenizer")]
impl<'a> OutputParser<'a> {
    /// Starts AFTER the rendered generation prefix. Runtime supplies every
    /// ordered generated ID including terminal IDs omitted by display decoding.
    pub fn new(codec: &'a ChatCodec, enable_thinking: bool, max_tokens: usize) -> Self {
        Self {
            codec,
            phase: if enable_thinking {
                Phase::Reasoning
            } else {
                Phase::Answer
            },
            max_tokens,
            ids: Vec::new(),
            reasoning: Vec::new(),
            answer: Vec::new(),
            call: Vec::new(),
            suffix: Vec::new(),
            terminal: None,
            error: None,
            call_closed: false,
        }
    }

    fn fail(&mut self, message: &str) {
        if self.error.is_none() {
            self.error = Some(message.to_owned());
        }
    }

    /// Errors latch and permanently suppress proposals, but subsequent bounded
    /// input is retained for diagnostics. Returns false when the bound is hit.
    pub fn push(&mut self, ids: &[u32]) -> bool {
        for &id in ids {
            if self.ids.len() >= self.max_tokens {
                self.fail("generated token bound exceeded");
                return false;
            }
            self.ids.push(id);
            if self.codec.tokenizer.id_to_token(id).is_none() {
                self.fail("unknown token ID");
            }
            if self.phase == Phase::End {
                self.fail("tokens after EOS");
                continue;
            }
            match id {
                2 | 11 => {
                    self.terminal = Some(id);
                    if matches!(self.phase, Phase::Reasoning | Phase::Call) {
                        self.fail("EOS before protocol completion");
                    }
                    self.phase = Phase::End;
                }
                13 if self.phase == Phase::Reasoning => self.phase = Phase::Answer,
                14 if self.phase == Phase::Answer => self.phase = Phase::Call,
                15 if self.phase == Phase::Call => {
                    self.call_closed = true;
                    self.phase = Phase::AfterCall;
                }
                _ if self.codec.forbidden.contains(&id) => self.fail("unexpected control token"),
                _ => match self.phase {
                    Phase::Reasoning => self.reasoning.push(id),
                    Phase::Answer => self.answer.push(id),
                    Phase::Call => self.call.push(id),
                    Phase::AfterCall => self.suffix.push(id),
                    Phase::End => unreachable!(),
                },
            }
        }
        true
    }

    /// Re-decode complete runs: individual tokens may contain only part of a
    /// UTF-8 character. Display snapshots can temporarily end in U+FFFD; they
    /// are replaced by a full-run decode, never concatenated into commands.
    pub fn snapshot(&self) -> Result<PartialOutput> {
        Ok(PartialOutput {
            reasoning: self.codec.decode(&self.reasoning)?,
            answer: self.codec.decode(&self.answer)?,
            raw_text: self.codec.decode(&self.ids)?,
            protocol_error: self.error.clone(),
        })
    }

    pub fn finish(mut self, stop: StopReason) -> Result<Completion> {
        let mut proposal = None;
        if let StopReason::Eos { id } = stop {
            if !matches!(id, 2 | 11) || self.terminal != Some(id) {
                self.fail("stop reason does not match an observed terminal EOS");
            }
            if self.call_closed {
                let suffix = self.codec.decode(&self.suffix)?;
                // One LF is the official template's call terminator, not a
                // natural-language suffix. All other suffix bytes are errors.
                if !suffix.is_empty() && suffix != "\n" {
                    self.fail("suffix after tool call");
                }
                let body = self.codec.decode_command(&self.call);
                match body
                    .as_ref()
                    .ok()
                    .and_then(|s| s.strip_prefix(CALL_PREFIX))
                    .and_then(|s| s.strip_suffix(CALL_SUFFIX))
                {
                    Some(command) => match valid_command(command) {
                        Ok(()) => {
                            proposal = Some(ExecProposal {
                                command: command.to_owned(),
                            })
                        }
                        Err(_) => self.fail("ambiguous command delimiters"),
                    },
                    None => self.fail("expected exactly one exec(command) XML block"),
                }
            }
        }
        if self.error.is_some() || !matches!(stop, StopReason::Eos { .. }) {
            proposal = None;
        }
        Ok(Completion {
            partial: self.snapshot()?,
            stop,
            proposal,
        })
    }
}
