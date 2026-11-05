//! Text-only Super chat protocol (HF revision da5b2650).
//!
//! The inference runtime admits the complete tokenized prompt against its real
//! context capacity. Byte and message counts are not token-capacity bounds.
//!
//! The reference template's `truncate_history_thinking` is deliberately false.
//! Rendering is not append-only: callers must compare token prefixes before
//! reusing inference state. This module neither owns that state nor executes
//! proposals. Literal marker spellings in content are not control tokens.

use anyhow::{Context, Result, ensure};

#[path = "chat_tools.rs"]
mod tools;
pub use tools::{Invocation, ToolDefinition, Tools};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    User(String),
    Assistant {
        reasoning: String,
        answer: String,
        tool_calls: Vec<Invocation>,
    },
    /// Selected model-visible text. Result identity/status remain host-side.
    ToolResult {
        content: String,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct History {
    pub system: String,
    /// Exact observed body for canonical replay. When present, neither the raw
    /// system nor Tools is rendered into this frame again. Selected Tools still
    /// validate calls. Original inputs belong to separate source provenance.
    pub rendered_system: Option<SystemBody>,
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
    pub tools_enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub text: String,
    pub token_ids: Vec<u32>,
}

/// Definition of the renderer actually compiled and tokenizer actually held.
/// Source content is the codec revision when a frozen build git revision is
/// unavailable. This does not assert an upstream Jinja template was executed.
pub struct RendererDefinition {
    pub template: String,
    pub tokenizer: String,
    pub codec: String,
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
const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// One observed renderer segment. Literal content and publisher control IDs
/// remain distinct even when their displayed marker spellings are identical.
/// The same segments are accepted for explicit canonical system-body replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderSegment {
    Control(u32),
    Content(String),
}
impl RenderSegment {
    /// Resolve only the pinned native marker vocabulary, never ordinary BPE.
    /// ChatCodec verifies this spelling/ID relation against its held tokenizer.
    pub fn control(spelling: &str) -> Result<Self> {
        let id = MARKERS.iter().position(|marker| *marker == spelling)
            .context("unknown native system control spelling")?;
        Ok(Self::Control(id as u32))
    }
    pub fn text(&self) -> Result<&str> {
        match self {
            Self::Content(text) => Ok(text),
            Self::Control(id) => MARKERS.get(*id as usize).copied()
                .context("unknown native system control ID"),
        }
    }
}
type Part = RenderSegment;

/// The exact body of the one system message, without its role frame. Retain
/// the ordered segments for replay; feeding joined text back as History.system
/// would duplicate tool instructions and erase trusted-control distinctions.
/// Raw system/definitions are separate provenance. Segments add no role frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemBody {
    text: String,
    segments: Vec<RenderSegment>,
}
impl SystemBody {
    pub fn from_segments(segments: Vec<RenderSegment>) -> Result<Self> {
        for segment in &segments {
            segment.text()?;
            ensure!(!matches!(segment, RenderSegment::Control(10 | 11)),
                "system body contains an interleaved role boundary");
        }
        Ok(Self { text: rendered_text(&segments)?, segments })
    }
    pub fn text(&self) -> &str { &self.text }
    pub fn segments(&self) -> &[RenderSegment] { &self.segments }
}
fn content(parts: &mut Vec<Part>, text: impl Into<String>) {
    parts.push(Part::Content(text.into()));
}
fn control(parts: &mut Vec<Part>, id: u32) {
    parts.push(Part::Control(id));
}
/// Only static publisher grammar takes this path, never caller content.
/// Splitting at added-token markers preserves the tokenizer's own boundaries;
/// all other text still coalesces before BPE, including across trust boundaries.
fn trusted_template(parts: &mut Vec<Part>, mut text: &'static str) {
    while let Some((offset, id, marker)) = MARKERS
        .iter()
        .enumerate()
        .filter_map(|(id, marker)| text.find(*marker).map(|offset| (offset, id, marker)))
        .min_by_key(|(offset, _, _)| *offset)
    {
        content(parts, &text[..offset]);
        control(parts, id as u32);
        text = &text[offset + marker.len()..];
    }
    content(parts, text);
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

fn render_parts(
    history: &History,
    options: RenderOptions,
    tools: &Tools,
    adapted: bool,
) -> Result<Vec<Part>> {
    let mut parts = Vec::new();
    open(&mut parts, "system");
    parts.extend(match &history.rendered_system {
        Some(body) => body.segments.clone(),
        None => system_parts(&history.system, options.tools_enabled, tools)?,
    });
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
                tool_calls,
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
                if !tool_calls.is_empty() {
                    content(&mut parts, "\n");
                }
                for invocation in tool_calls {
                    invocation.validate(tools)?;
                    control(&mut parts, 14);
                    content(&mut parts, invocation.body());
                    control(&mut parts, 15);
                    content(&mut parts, "\n");
                }
                close(&mut parts);
            }
            Message::ToolResult { content: text } => {
                let preceding_tool =
                    i > 0 && matches!(history.messages[i - 1], Message::ToolResult { .. });
                if !preceding_tool && (adapted || i > 0) {
                    open(&mut parts, "user");
                }
                control(&mut parts, 16);
                content(&mut parts, "\n");
                content(&mut parts, text);
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

fn system_parts(system: &str, tools_enabled: bool, tools: &Tools) -> Result<Vec<Part>> {
    let mut parts = Vec::new();
    content(&mut parts, system);
    if tools_enabled && !tools.definitions().is_empty() {
        if !system.is_empty() { content(&mut parts, "\n\n"); }
        content(&mut parts, tools.render()?);
        // Only these static publisher examples are trusted control grammar.
        trusted_template(&mut parts, TOOL_INSTRUCTIONS);
    }
    Ok(parts)
}

/// Observe the same system body used by render_parts; never a second renderer.
pub fn render_system_body(system: &str, tools_enabled: bool, tools: &Tools) -> Result<SystemBody> {
    let segments = system_parts(system, tools_enabled, tools)?;
    SystemBody::from_segments(segments)
}

fn rendered_text(parts: &[Part]) -> Result<String> {
    let bytes = parts.iter().try_fold(0usize, |bytes, part| {
        let len = match part {
            Part::Control(id) => MARKERS[*id as usize].len(),
            Part::Content(s) => s.len(),
        };
        bytes.checked_add(len).context("rendered history size overflow")
    })?;
    let mut text = String::new();
    text.try_reserve_exact(bytes)
        .context("allocate complete rendered history")?;
    for part in parts {
        match part {
            Part::Control(id) => text.push_str(MARKERS[*id as usize]),
            Part::Content(s) => text.push_str(s),
        }
    }
    Ok(text)
}

/// Literal reference text only, never safe-to-feed tokenization. A leading
/// orphan retains the upstream missing opener. Use `render_adapted` for actual
/// prompts. Embedded assistant marker strings are treated as typed content,
/// not upstream's implicit reasoning.
pub fn render_reference_text(
    history: &History,
    options: RenderOptions,
    tools: &Tools,
) -> Result<String> {
    rendered_text(&render_parts(history, options, tools, false)?)
}

#[cfg(feature = "tokenizer")]
pub struct ChatCodec {
    tokenizer: tokenizers::Tokenizer,
    forbidden: std::collections::HashSet<u32>,
    tools: Tools,
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
        // The reverse lookup alone can select one of two IDs with the same
        // spelling. A replayed CONTROL must have one exact held-token meaning.
        for marker in MARKERS {
            ensure!(added.values().filter(|token| token.content == marker).count() == 1,
                "ambiguous or absent Super control spelling: {marker}");
        }
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
            tools: Tools::exec_only(),
        })
    }

    /// Install the actual immutable tool selection BEFORE preparing any wake
    /// or retained prefix. The selection is rendered/token-admitted normally;
    /// it is not a late execution-only attachment or a mutable registry.
    pub fn with_tools(mut self, tools: Tools) -> Self {
        self.tools = tools;
        self
    }

    pub fn tools(&self) -> &Tools {
        &self.tools
    }

    // Share the compiled-source identity with first-fault diagnostics. This
    // path never serializes the held tokenizer or includes generated content.
    fn renderer_source() -> (String, String) {
        let template = format!("// maryml/src/models/nemotron_h/chat.rs\n{}\n// maryml/src/models/nemotron_h/chat_tools.rs\n{}",
            include_str!("chat.rs"), include_str!("chat_tools.rs"));
        let codec = format!("maryml::models::nemotron_h::chat source-blake3:{}",
            blake3::hash(template.as_bytes()).to_hex());
        (template, codec)
    }

    pub fn renderer_definition(&self) -> Result<RendererDefinition> {
        let (template, codec) = Self::renderer_source();
        let tokenizer = self.tokenizer.to_string(false)
            .map_err(|error| anyhow::anyhow!("serialize held native tokenizer: {error}"))?;
        Ok(RendererDefinition { template, tokenizer, codec })
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

    /// Ordinary supported histories retain publisher text AND token IDs.
    /// Named departures: frame leading orphan results and encode literal marker
    /// spellings as marker-safe content, including prior assistant text. Static
    /// template grammar keeps its observed control IDs. Result content is not
    /// wrapped; identity/status belong to the host, not this model transcript.
    pub fn render_adapted(
        &self,
        history: &History,
        options: RenderOptions,
    ) -> Result<RenderedPrompt> {
        let parts = render_parts(history, options, &self.tools, true)?;
        let text = rendered_text(&parts)?;
        let mut token_ids = Vec::new();
        // Coalesce adjacent content so benign text keeps normal BPE boundaries.
        let mut pending = String::new();
        for part in parts {
            match part {
                Part::Content(s) => pending.push_str(&s),
                Part::Control(id) => {
                    let spelling = MARKERS.get(id as usize).context("unknown native control ID")?;
                    ensure!(self.tokenizer.token_to_id(spelling) == Some(id)
                        && self.tokenizer.id_to_token(id).as_deref() == Some(*spelling),
                        "system control does not match held tokenizer");
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
    /// Ordered, complete batch; empty unless every call and terminal validate.
    pub proposals: Vec<Invocation>,
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
#[derive(Clone, Copy)]
struct FaultToken {
    id: u32,
    phase: Phase,
    /// Zero-based emitted position, including a token refused by max_tokens.
    index: usize,
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
    calls: Vec<Vec<u32>>,
    suffix: Vec<u32>,
    terminal: Option<u32>,
    error: Option<String>,
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
            calls: Vec::new(),
            suffix: Vec::new(),
            terminal: None,
            error: None,
        }
    }

    fn fail(&mut self, message: &str, token: Option<FaultToken>) {
        if self.error.is_none() {
            // Only exact, bounded reserved-vocabulary spellings are metadata.
            // Never expose an ordinary BPE piece or truncate it into evidence.
            let (id, index, phase, count, spelling, spelling_status) = match token {
                Some(token) => {
                    let known = if token.id < 1000 {
                        self.codec.tokenizer.id_to_token(token.id)
                    } else {
                        None
                    };
                    let (spelling, status) = match known {
                        Some(text) if text.len() <= 96 =>
                            (serde_json::to_string(&text).expect("serialize vocabulary text"), "known"),
                        Some(_) => ("null".into(), "over-limit"),
                        None if self.codec.forbidden.contains(&token.id) => ("null".into(), "unavailable"),
                        None => ("null".into(), "non-control-or-unknown"),
                    };
                    (token.id.to_string(), token.index.to_string(), token.phase,
                        token.index + 1, spelling, status)
                }
                // finish() can reject a complete call or a claimed stop without
                // knowing one offending token. Do not blame the last emission.
                None => ("none".into(), "none".into(), self.phase, self.ids.len(),
                    "null".into(), "no-token"),
            };
            let (_, codec) = ChatCodec::renderer_source();
            self.error = Some(format!("{message} [token_id={id} phase={phase:?} emitted_index={index} emitted_count={count} retained_count={} spelling={spelling} spelling_status={spelling_status} codec={codec:?}]", self.ids.len()));
        }
    }

    /// Errors latch and permanently suppress proposals, but subsequent bounded
    /// input is retained for diagnostics. Returns false when the bound is hit.
    pub fn push(&mut self, ids: &[u32]) -> bool {
        for &id in ids {
            let token = Some(FaultToken { id, phase: self.phase, index: self.ids.len() });
            if self.ids.len() >= self.max_tokens {
                self.fail("generated token bound exceeded", token);
                return false;
            }
            self.ids.push(id);
            if self.codec.tokenizer.id_to_token(id).is_none() {
                self.fail("unknown token ID", token);
            }
            if self.phase == Phase::End {
                self.fail("tokens after EOS", token);
                continue;
            }
            match id {
                2 | 11 => {
                    self.terminal = Some(id);
                    if matches!(self.phase, Phase::Reasoning | Phase::Call) {
                        self.fail("EOS before protocol completion", token);
                    }
                    self.phase = Phase::End;
                }
                13 if self.phase == Phase::Reasoning => self.phase = Phase::Answer,
                14 if self.phase == Phase::Answer => self.phase = Phase::Call,
                14 if self.phase == Phase::AfterCall => {
                    // The pinned publisher loops over envelopes with exactly
                    // one LF between them. No whitespace trimming or prose.
                    if !matches!(self.codec.decode_command(&self.suffix).as_deref(), Ok("\n")) {
                        self.fail("expected one LF between tool calls", token);
                    }
                    self.suffix.clear();
                    self.phase = Phase::Call;
                }
                15 if self.phase == Phase::Call => {
                    self.calls.push(std::mem::take(&mut self.call));
                    self.phase = Phase::AfterCall;
                }
                _ if self.codec.forbidden.contains(&id) => self.fail("unexpected control token", token),
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
        let mut proposals = Vec::new();
        if let StopReason::Eos { id } = stop {
            if !matches!(id, 2 | 11) || self.terminal != Some(id) {
                self.fail("stop reason does not match an observed terminal EOS", None);
            }
            if !self.calls.is_empty() {
                let suffix = self.codec.decode_command(&self.suffix);
                // One LF is the official template's call terminator, not a
                // natural-language suffix. Preserve the accepted single-call
                // EOS immediately after the closer; no other suffix is valid.
                if !matches!(suffix.as_deref(), Ok("" | "\n")) {
                    self.fail("suffix after tool call", None);
                }
                // No invocation leaves this parser until every body validates
                // and the observed EOS matches. A bad later call voids all.
                if self.error.is_none() {
                    for call in std::mem::take(&mut self.calls) {
                        match self.codec.decode_command(&call)
                            .and_then(|body| self.codec.tools.parse_call(&body))
                        {
                            Ok(invocation) => proposals.push(invocation),
                            Err(error) => {
                                self.fail(&format!("invalid registered tool call: {error:#}"), None);
                                break;
                            }
                        }
                    }
                }
            }
        }
        if self.error.is_some() || !matches!(stop, StopReason::Eos { .. }) {
            proposals.clear();
        }
        Ok(Completion {
            partial: self.snapshot()?,
            stop,
            proposals,
        })
    }
}
