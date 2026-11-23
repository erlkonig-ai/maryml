//! Inkling 2 Mini's chat protocol: the release's `chat_template.jinja`
//! (pinned in [`super::release`]) written out as a renderer, and the parser
//! for what the model generates in reply.
//!
//! No Jinja engine runs here. [`render_text`] produces the same text the
//! template produces for the histories this module can represent, and a test
//! holds it to upstream's own rendering of fixed conversations
//! (`fixtures/chat_reference.json`). [`ChatCodec`] turns that rendering into
//! token ids with the root's own tokenizer.
//!
//! What the template does, and so what this does:
//!
//! * tool declarations come first, as one system message
//!   `<|message_system|>tool_declare<|content_xml|>` + the sorted, compact
//!   JSON list of `{description, name, parameters, type}` + `<|end_message|>`;
//! * every message part is its own frame, `role + content kind + body +
//!   <|end_message|>`; an assistant message is thinking, then text, then one
//!   frame per tool call, closed by `<|content_model_end_sampling|>`, which is
//!   also the model's only end-of-sequence token;
//! * a tool call frame is `<|message_model|>NAME<|content_invoke_tool_json|>`
//!   then `{"name":NAME,"args":{...}}` with the arguments in key order;
//! * a tool result is headed by the tool's name, `<|message_tool|>NAME`;
//! * a `Thinking effort level: E` system message goes before the first
//!   non-system message (at the end when there is none);
//! * there is no generation prompt: the prompt ends with the last message and
//!   the model opens its own `<|message_model|>`.
//!
//! Named departures from executing the template on the whole rendered string:
//!
//! * content is encoded with the tokenizer WITHOUT its added tokens, so a
//!   literal marker spelling inside a message is ordinary text, never
//!   structure. Upstream's tokenizer would turn it into the control token;
//! * nested JSON objects inside a call's arguments are written in the order
//!   this process holds them. The template keeps the order the caller's
//!   dictionary had, which a parsed `serde_json::Map` need not preserve;
//! * rendering is not append-only with respect to what the model generated:
//!   two text frames in one reply render as one, and a reply's tokens are the
//!   tokenizer's, not necessarily the ones sampled. Callers compare token
//!   prefixes before reusing inference state.
//!
//! Images ([`Part::Image`] in [`Message::UserParts`] and [`Message::ToolParts`]):
//!
//! * every part of a user or tool message is its own frame with the role header
//!   repeated (`<|message_tool|>NAME` for each part of a tool result), as the
//!   template renders a list `content`; an image part's frame is
//!   `role <|content_image|><|unused_200054|><|end_message|>`;
//! * [`ChatCodec::render`] then does what the release's `Inkling2Processor` does
//!   to that text: the image's one `<|unused_200054|>` becomes `rows * cols` of
//!   them (its `image_grid_thw` product, one per projected feature row), in
//!   image order. [`RenderedPrompt::images`] names each run of placeholders by
//!   its place in the ids and the image it stands for, so no caller writes
//!   those ids itself; [`RenderedPrompt::text`] stays the template's text, one
//!   placeholder per image;
//! * the expansion is bounded before it is allocated: a render refuses a
//!   history whose image tokens, or whose whole prompt, exceed the codec's
//!   prompt budget ([`ChatCodec::with_prompt_budget`], [`ChatCodec::render_within`]);
//! * the assistant has no image parts, as the template refuses them; neither is
//!   audio supported, which the template refuses too.
//!
//! Further named departures for images: a message with no parts is refused
//! (the template would render nothing for it); a literal `<|unused_200054|>` in
//! a text part is text, as every other marker spelling is, where upstream's
//! processor counts it as one more image (and refuses the conversation).

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

/// The protocol's control tokens, with the ids the pinned release gives them
/// (`tokenizer_config.json`). [`ChatCodec::new`] checks each against the
/// tokenizer it is handed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Marker {
    MessageUser,
    MessageModel,
    MessageSystem,
    MessageTool,
    ContentText,
    ContentThinking,
    ContentXml,
    ContentInvokeToolJson,
    EndMessage,
    EndSampling,
    /// `<|content_image|>`: the content kind of an image part's frame.
    ContentImage,
    /// `<|unused_200054|>`: the processor's image token, one per projected
    /// image feature row once a render expands it.
    ImagePlaceholder,
}

impl Marker {
    pub const ALL: [Marker; 12] = [
        Marker::MessageUser,
        Marker::MessageModel,
        Marker::MessageSystem,
        Marker::MessageTool,
        Marker::ContentText,
        Marker::ContentThinking,
        Marker::ContentXml,
        Marker::ContentInvokeToolJson,
        Marker::EndMessage,
        Marker::EndSampling,
        Marker::ContentImage,
        Marker::ImagePlaceholder,
    ];

    pub fn spelling(self) -> &'static str {
        match self {
            Marker::MessageUser => "<|message_user|>",
            Marker::MessageModel => "<|message_model|>",
            Marker::MessageSystem => "<|message_system|>",
            Marker::MessageTool => "<|message_tool|>",
            Marker::ContentText => "<|content_text|>",
            Marker::ContentThinking => "<|content_thinking|>",
            Marker::ContentXml => "<|content_xml|>",
            Marker::ContentInvokeToolJson => "<|content_invoke_tool_json|>",
            Marker::EndMessage => "<|end_message|>",
            Marker::EndSampling => "<|content_model_end_sampling|>",
            Marker::ContentImage => "<|content_image|>",
            Marker::ImagePlaceholder => "<|unused_200054|>",
        }
    }

    pub fn id(self) -> u32 {
        match self {
            Marker::MessageUser => 200000,
            Marker::MessageModel => 200001,
            Marker::MessageSystem => 200002,
            Marker::MessageTool => 200003,
            Marker::ContentText => 200004,
            Marker::EndSampling => 200006,
            Marker::ContentThinking => 200008,
            Marker::EndMessage => 200010,
            Marker::ContentXml => 200024,
            Marker::ContentInvokeToolJson => 200049,
            Marker::ContentImage => 200005,
            Marker::ImagePlaceholder => 200054,
        }
    }

    pub fn from_spelling(spelling: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|marker| marker.spelling() == spelling)
            .with_context(|| format!("not a Mini protocol control: {spelling:?}"))
    }

    fn from_id(id: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|marker| marker.id() == id)
    }
}

/// The release's `chat_template.jinja`, verbatim: the template this module
/// writes out. A root whose template differs needs a different renderer, so
/// [`check_template`] refuses it.
pub const RELEASE_TEMPLATE: &str = include_str!("fixtures/chat_template.jinja");

/// Refuse a chat template other than [`RELEASE_TEMPLATE`].
pub fn check_template(template: &str) -> Result<()> {
    ensure!(
        template == RELEASE_TEMPLATE,
        "the root's chat template ({} bytes) is not the release template this renderer writes out ({} bytes)",
        template.len(),
        RELEASE_TEMPLATE.len()
    );
    Ok(())
}

/// The token that closes a complete reply (`generation_config.json`'s
/// `eos_token_id`).
pub const END_OF_REPLY: u32 = 200006;

/// The template's `reasoning_effort` levels, by name. The template's default,
/// when none is given, is 0.9 ([`Effort::High`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Effort {
    None,
    Minimal,
    Low,
    Medium,
    #[default]
    High,
    Xhigh,
    Max,
}

impl Effort {
    /// The number as the template prints it (`0` for none, Python's float
    /// spelling otherwise).
    pub fn spelling(self) -> &'static str {
        match self {
            Effort::None => "0",
            Effort::Minimal => "0.1",
            Effort::Low => "0.2",
            Effort::Medium => "0.7",
            Effort::High => "0.9",
            Effort::Xhigh | Effort::Max => "0.99",
        }
    }
}

/// One declared tool. `parameters` is its JSON schema object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// The selected tools, fixed before any prompt is rendered.
///
/// Every definition is declared: calls parse and replay against all of them,
/// whatever the declaration shows. An indexed `Tools` only presents them more
/// compactly (see [`Tools::indexed`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tools {
    definitions: Vec<ToolDefinition>,
    /// When set, the declaration shows these tools in full and every other
    /// one as an index entry: its name and [`summary`] only.
    detailed: Option<Vec<String>>,
}

/// A tool name also appears as plain text in a frame header, before the
/// content marker. Upstream's reply parser reads that header as `[^<]*`, so a
/// name holding `<` could not be read back.
fn valid_name(name: &str) -> Result<()> {
    ensure!(!name.is_empty(), "empty tool name");
    ensure!(!name.contains('<'), "tool name {name:?} contains '<'");
    Ok(())
}

impl Tools {
    pub fn new(definitions: Vec<ToolDefinition>) -> Result<Self> {
        for (index, tool) in definitions.iter().enumerate() {
            valid_name(&tool.name)?;
            ensure!(
                !definitions[..index].iter().any(|prior| prior.name == tool.name),
                "tool {} declared twice",
                tool.name
            );
            ensure!(tool.parameters.is_object(), "tool {} parameters must be an object", tool.name);
        }
        Ok(Self { definitions, detailed: None })
    }

    /// The same declared tools, presented as an index: the `detailed` ones in
    /// full, every other one by its name and the first sentence of its
    /// description, with its parameters shown only as an object. The full
    /// definitions stay here, and parsing and validation use them; a lookup
    /// tool among `detailed` is how the model reads one in full.
    pub fn indexed(definitions: Vec<ToolDefinition>, detailed: &[&str]) -> Result<Self> {
        let tools = Self::new(definitions)?;
        for name in detailed {
            ensure!(tools.is_declared(name), "detailed tool {name:?} is not declared");
        }
        Ok(Self { detailed: Some(detailed.iter().map(|name| (*name).to_owned()).collect()), ..tools })
    }

    /// Whether the declaration is an index rather than every full schema.
    pub fn is_indexed(&self) -> bool {
        self.detailed.is_some()
    }

    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// The full definition of a declared tool.
    pub fn definition(&self, name: &str) -> Option<&ToolDefinition> {
        self.definitions.iter().find(|tool| tool.name == name)
    }

    fn is_declared(&self, name: &str) -> bool {
        self.definitions.iter().any(|tool| tool.name == name)
    }

    /// The `tool_declare` body: the sorted, compact JSON list.
    fn declaration(&self) -> String {
        let mut out = String::from("[");
        for (index, tool) in self.definitions.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let full = self.detailed.as_ref().is_none_or(|detailed| detailed.contains(&tool.name));
            let mut spec = Map::new();
            if full {
                spec.insert("description".into(), Value::String(tool.description.clone()));
                spec.insert("parameters".into(), tool.parameters.clone());
            } else {
                spec.insert("description".into(), Value::String(summary(&tool.description).to_owned()));
                spec.insert("parameters".into(), serde_json::json!({"type": "object"}));
            }
            spec.insert("name".into(), Value::String(tool.name.clone()));
            spec.insert("type".into(), Value::String("function".into()));
            write_sorted(&mut out, &Value::Object(spec));
        }
        out.push(']');
        out
    }

    /// Read one generated call body, `{"name":..,"args":{..}}`. The body is
    /// kept verbatim as the invocation's original; its typed projection is the
    /// name and the argument object.
    pub fn parse_call(&self, body: &str) -> Result<Invocation> {
        let value: Value = serde_json::from_str(body).context("tool call is not JSON")?;
        let object = value.as_object().context("tool call is not a JSON object")?;
        ensure!(
            object.len() == 2 && object.contains_key("name") && object.contains_key("args"),
            "tool call must have exactly the fields name and args"
        );
        let name = object["name"].as_str().context("tool call name is not a string")?;
        ensure!(self.is_declared(name), "call to undeclared tool {name:?}");
        let arguments = object["args"].as_object().context("tool call args is not an object")?.clone();
        Ok(Invocation { name: name.to_owned(), arguments, body: body.to_owned() })
    }
}

/// The first sentence of a description: its first line, up to the first
/// full stop that whitespace and a capital letter follow (so "e.g. by" does
/// not end it). A description with none is its whole first line.
pub fn summary(description: &str) -> &str {
    let line = description.trim_start().lines().next().unwrap_or("").trim_end();
    for (at, _) in line.match_indices('.') {
        let rest = &line[at + 1..];
        let next = rest.trim_start();
        if rest.len() > next.len() && next.chars().next().is_some_and(char::is_uppercase) {
            return &line[..=at];
        }
    }
    line
}

/// One tool call: its name, its arguments, and the exact body text it was
/// read from (or, for a host-made call, the template's canonical body).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invocation {
    name: String,
    arguments: Map<String, Value>,
    body: String,
}

impl Invocation {
    /// A host-made call (a recalled memory, a replayed host action), with the
    /// body the template would render for it.
    pub fn new(name: impl Into<String>, arguments: Map<String, Value>, tools: &Tools) -> Result<Self> {
        let name = name.into();
        ensure!(tools.is_declared(&name), "call to undeclared tool {name:?}");
        let body = canonical_call_body(&name, &arguments);
        Ok(Self { name, arguments, body })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arguments(&self) -> &Map<String, Value> {
        &self.arguments
    }

    pub fn body(&self) -> &str {
        &self.body
    }
}

/// What the template writes for a call: the name, then the arguments in key
/// order, each value compact.
fn canonical_call_body(name: &str, arguments: &Map<String, Value>) -> String {
    let mut out = String::from("{\"name\":");
    out.push_str(&json_string(name));
    out.push_str(",\"args\":{");
    let mut keys: Vec<&String> = arguments.keys().collect();
    keys.sort();
    for (index, key) in keys.into_iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&json_string(key));
        out.push(':');
        out.push_str(&serde_json::to_string(&arguments[key]).expect("a JSON value serializes"));
    }
    out.push_str("}}");
    out
}

fn json_string(text: &str) -> String {
    serde_json::to_string(text).expect("a string serializes")
}

/// Compact JSON with every object's keys sorted, at every depth (Python's
/// `json.dumps(sort_keys=True, separators=(",", ":"))`), whatever order the
/// map holds them in.
fn write_sorted(out: &mut String, value: &Value) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&json_string(key));
                out.push(':');
                write_sorted(out, &map[key]);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_sorted(out, item);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).expect("a JSON value serializes")),
    }
}

/// One image in a message: the image token grid the processor gives it, and
/// the identity of the image it was made from.
///
/// `origin` is the caller's own 32-byte name for the source image (Drive's is
/// a digest of the media type and bytes); this module never interprets it,
/// only carries it into [`ImageSpan`] so the caller can check, before it
/// feeds features, that they come from the image the prompt names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImagePart {
    origin: [u8; 32],
    grid: (usize, usize),
}

impl ImagePart {
    /// An image whose processor grid is `(rows, cols)` image tokens.
    pub fn new(origin: [u8; 32], grid: (usize, usize)) -> Result<Self> {
        let (rows, cols) = grid;
        ensure!(rows > 0 && cols > 0, "an image needs image tokens; its grid is {rows}x{cols}");
        rows.checked_mul(cols).with_context(|| format!("an image grid of {rows}x{cols} overflows"))?;
        Ok(Self { origin, grid })
    }

    /// The image the processor produced `tokens` for.
    pub fn of(origin: [u8; 32], tokens: &super::vision_input::ImageTokens) -> Result<Self> {
        Self::new(origin, tokens.grid)
    }

    pub fn origin(&self) -> [u8; 32] {
        self.origin
    }

    /// `(rows, cols)` of image tokens.
    pub fn grid(&self) -> (usize, usize) {
        self.grid
    }

    /// How many placeholder ids the image becomes: `rows * cols`.
    pub fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1
    }
}

/// One part of a user or tool message, rendered as its own frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    Text(String),
    Image(ImagePart),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    /// A user message of one text part.
    User(String),
    Assistant { reasoning: String, answer: String, tool_calls: Vec<Invocation> },
    /// A tool's result of one text part, headed by the tool's name.
    Tool { name: String, content: String },
    /// A user message of ordered text and image parts.
    UserParts(Vec<Part>),
    /// A tool's result of ordered text and image parts, every part headed by
    /// the tool's name.
    ToolParts { name: String, parts: Vec<Part> },
}

/// One rendered piece: a control token, or text the tokenizer encodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Segment {
    Control(Marker),
    Content(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct History {
    /// The one system message.
    pub system: String,
    /// The exact rendered opening (tool declarations, system message and
    /// effort message), kept for replay. When present it is used verbatim
    /// instead of rendering `system` and the tools again.
    pub rendered_opening: Option<Vec<Segment>>,
    pub messages: Vec<Message>,
}

fn control(parts: &mut Vec<Segment>, marker: Marker) {
    parts.push(Segment::Control(marker));
}

fn content(parts: &mut Vec<Segment>, text: impl Into<String>) {
    let text = text.into();
    if !text.is_empty() {
        parts.push(Segment::Content(text));
    }
}

fn frame(parts: &mut Vec<Segment>, role: Marker, header: &str, kind: Marker, body: &str) {
    control(parts, role);
    content(parts, header);
    control(parts, kind);
    content(parts, body);
    control(parts, Marker::EndMessage);
}

/// The opening the template writes before the conversation: the tool
/// declarations (when there are tools), the system message and the effort
/// message. With exactly one system message, the effort message always
/// follows it directly, whether or not other messages come after.
pub fn render_opening(system: &str, tools: &Tools, effort: Effort) -> Vec<Segment> {
    let mut parts = Vec::new();
    if !tools.definitions().is_empty() {
        frame(&mut parts, Marker::MessageSystem, "tool_declare", Marker::ContentXml, &tools.declaration());
    }
    frame(&mut parts, Marker::MessageSystem, "", Marker::ContentText, system);
    frame(
        &mut parts,
        Marker::MessageSystem,
        "",
        Marker::ContentText,
        &format!("Thinking effort level: {}", effort.spelling()),
    );
    parts
}

/// Each part its own frame under the same header, as the template renders a
/// list `content`. Images are collected in order beside the segments.
fn part_frames(
    parts: &mut Vec<Segment>,
    images: &mut Vec<ImagePart>,
    role: Marker,
    header: &str,
    message_parts: &[Part],
) -> Result<()> {
    ensure!(!message_parts.is_empty(), "a {} message needs at least one part", role.spelling());
    for part in message_parts {
        match part {
            Part::Text(text) => frame(parts, role, header, Marker::ContentText, text),
            Part::Image(image) => {
                control(parts, role);
                content(parts, header);
                control(parts, Marker::ContentImage);
                control(parts, Marker::ImagePlaceholder);
                control(parts, Marker::EndMessage);
                images.push(*image);
            }
        }
    }
    Ok(())
}

/// The whole prompt as segments.
pub fn render_segments(history: &History, tools: &Tools, effort: Effort) -> Result<Vec<Segment>> {
    Ok(render_with_images(history, tools, effort)?.0)
}

/// The whole prompt as segments, and its images in the order of their
/// [`Marker::ImagePlaceholder`] segments.
fn render_with_images(history: &History, tools: &Tools, effort: Effort) -> Result<(Vec<Segment>, Vec<ImagePart>)> {
    let mut parts = match &history.rendered_opening {
        Some(opening) => {
            ensure!(
                !opening.iter().any(|segment| matches!(
                    segment,
                    Segment::Control(Marker::ContentImage | Marker::ImagePlaceholder)
                )),
                "a kept opening holds an image marker, which names no image"
            );
            opening.clone()
        }
        None => render_opening(&history.system, tools, effort),
    };
    let mut images = Vec::new();
    for message in &history.messages {
        match message {
            Message::User(text) => frame(&mut parts, Marker::MessageUser, "", Marker::ContentText, text),
            Message::Tool { name, content: text } => {
                valid_name(name)?;
                frame(&mut parts, Marker::MessageTool, name, Marker::ContentText, text)
            }
            Message::UserParts(message_parts) => {
                part_frames(&mut parts, &mut images, Marker::MessageUser, "", message_parts)?
            }
            Message::ToolParts { name, parts: message_parts } => {
                valid_name(name)?;
                part_frames(&mut parts, &mut images, Marker::MessageTool, name, message_parts)?
            }
            Message::Assistant { reasoning, answer, tool_calls } => {
                if !reasoning.is_empty() {
                    frame(&mut parts, Marker::MessageModel, "", Marker::ContentThinking, reasoning);
                }
                if !answer.is_empty() || (tool_calls.is_empty() && reasoning.is_empty()) {
                    frame(&mut parts, Marker::MessageModel, "", Marker::ContentText, answer);
                }
                for call in tool_calls {
                    ensure!(tools.is_declared(call.name()), "history calls undeclared tool {:?}", call.name());
                    frame(
                        &mut parts,
                        Marker::MessageModel,
                        call.name(),
                        Marker::ContentInvokeToolJson,
                        &canonical_call_body(call.name(), call.arguments()),
                    );
                }
                control(&mut parts, Marker::EndSampling);
            }
        }
    }
    Ok((parts, images))
}

pub fn segments_text(parts: &[Segment]) -> String {
    let mut text = String::new();
    for part in parts {
        match part {
            Segment::Control(marker) => text.push_str(marker.spelling()),
            Segment::Content(body) => text.push_str(body),
        }
    }
    text
}

/// The prompt as text, for comparison with upstream's template rendering.
pub fn render_text(history: &History, tools: &Tools, effort: Effort) -> Result<String> {
    Ok(segments_text(&render_segments(history, tools, effort)?))
}

/// Why generation stopped, as the caller observed it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted [`END_OF_REPLY`].
    EndOfReply,
    Budget,
    Cancelled,
    Error(String),
}

/// A parsed reply. `proposals` is empty unless the reply ended with
/// [`END_OF_REPLY`] and every part was well formed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub reasoning: String,
    pub answer: String,
    /// Every generated token decoded, markers included, malformed or not.
    pub raw_text: String,
    pub protocol_error: Option<String>,
    pub proposals: Vec<Invocation>,
    pub stop: StopReason,
}

/// What a rendered prompt is: its text and its token ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedPrompt {
    /// The template's text: one `<|unused_200054|>` per image.
    pub text: String,
    /// The processor's ids: each image's placeholder repeated `rows * cols`
    /// times.
    pub token_ids: Vec<u32>,
    /// One span per image, in history order.
    pub images: Vec<ImageSpan>,
}

/// Where one image's placeholders are in [`RenderedPrompt::token_ids`]:
/// `token_ids[start..start + len]` is exactly `len = image.tokens()` copies of
/// [`Marker::ImagePlaceholder`] (not the `<|content_image|>` before them or the
/// `<|end_message|>` after). `row` is the first of its rows among all images'
/// feature rows concatenated in order: the sum of the earlier spans' `len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSpan {
    pub start: usize,
    pub len: usize,
    pub row: usize,
    pub image: ImagePart,
}

/// The prompt budget a codec renders within unless told otherwise: the
/// release's `model_max_length` (`tokenizer_config.json`).
pub const DEFAULT_PROMPT_BUDGET: usize = 1 << 20;

/// The renderer's identity, for the archive: the compiled source of this
/// module, the tokenizer it holds, and a codec name with the source's hash.
pub struct RendererDefinition {
    pub template: String,
    pub tokenizer: String,
    pub codec: String,
}

/// The rendered prompt's tokens, and the parser for replies, over one root's
/// tokenizer.
#[cfg(feature = "tokenizer")]
pub struct ChatCodec {
    tokenizer: tokenizers::Tokenizer,
    content: tokenizers::Tokenizer,
    tools: Tools,
    effort: Effort,
    prompt_budget: usize,
}

#[cfg(feature = "tokenizer")]
impl ChatCodec {
    /// `tokenizer` is the root's whole tokenizer; `content` is the same graph
    /// built without its added tokens ([`crate::tokenizer::build_tokenizer_with_added`]
    /// with `false`), which encodes every message body.
    pub fn new(
        tokenizer: tokenizers::Tokenizer,
        content: tokenizers::Tokenizer,
        tools: Tools,
        effort: Effort,
    ) -> Result<Self> {
        for marker in Marker::ALL {
            ensure!(
                tokenizer.token_to_id(marker.spelling()) == Some(marker.id())
                    && tokenizer.id_to_token(marker.id()).as_deref() == Some(marker.spelling()),
                "the tokenizer does not hold {} at {}",
                marker.spelling(),
                marker.id()
            );
        }
        for added in tokenizer.get_added_tokens_decoder().values() {
            ensure!(
                content.token_to_id(&added.content).is_none(),
                "the content tokenizer still holds the added token {:?}",
                added.content
            );
        }
        Ok(Self { tokenizer, content, tools, effort, prompt_budget: DEFAULT_PROMPT_BUDGET })
    }

    /// The same codec rendering within `tokens`: [`Self::render`] refuses a
    /// prompt longer than that, and an image expansion that would exceed it
    /// before allocating it.
    pub fn with_prompt_budget(self, tokens: usize) -> Self {
        Self { prompt_budget: tokens, ..self }
    }

    pub fn prompt_budget(&self) -> usize {
        self.prompt_budget
    }

    /// Build both tokenizers from a root's tokenizer graph.
    pub fn from_graph(
        facts: &triblespace::prelude::TribleSet,
        blobs: &impl triblespace::prelude::BlobStoreGet,
        tokenizer_root: triblespace::prelude::Id,
        tools: Tools,
        effort: Effort,
    ) -> Result<Self> {
        let whole = crate::tokenizer::build_tokenizer_with_added(facts, blobs, tokenizer_root, true)
            .map_err(|e| anyhow::anyhow!("build the Mini tokenizer: {e}"))?;
        let content = crate::tokenizer::build_tokenizer_with_added(facts, blobs, tokenizer_root, false)
            .map_err(|e| anyhow::anyhow!("build the Mini content tokenizer: {e}"))?;
        Self::new(whole, content, tools, effort)
    }

    pub fn tools(&self) -> &Tools {
        &self.tools
    }

    pub fn effort(&self) -> Effort {
        self.effort
    }

    fn source() -> (String, String) {
        let template = format!(
            "// maryml/src/models/inkling2_loop/chat.rs\n{}\n// the release's chat_template.jinja, which it writes out\n{}",
            include_str!("chat.rs"),
            RELEASE_TEMPLATE
        );
        let codec = format!(
            "maryml::models::inkling2_loop::chat source-blake3:{}",
            blake3::hash(template.as_bytes()).to_hex()
        );
        (template, codec)
    }

    pub fn renderer_definition(&self) -> Result<RendererDefinition> {
        let (template, codec) = Self::source();
        let tokenizer = self
            .tokenizer
            .to_string(false)
            .map_err(|e| anyhow::anyhow!("serialize the Mini tokenizer: {e}"))?;
        Ok(RendererDefinition { template, tokenizer, codec })
    }

    /// The opening segments this codec renders for `system`.
    pub fn opening(&self, system: &str) -> Vec<Segment> {
        render_opening(system, &self.tools, self.effort)
    }

    /// Message body text to ids, losslessly, never producing a control token.
    pub fn encode_content(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let ids = self
            .content
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode Mini content: {e}"))?
            .get_ids()
            .to_vec();
        ensure!(self.decode_bytes(&ids)? == text.as_bytes(), "Mini content encoding is not lossless");
        Ok(ids)
    }

    /// The prompt within the codec's prompt budget.
    pub fn render(&self, history: &History) -> Result<RenderedPrompt> {
        self.render_within(history, self.prompt_budget)
    }

    /// The prompt, refused when it would be longer than `budget` tokens. The
    /// images' placeholder count is checked against the budget before any of
    /// it is allocated, and again, with the text, as it grows.
    pub fn render_within(&self, history: &History, budget: usize) -> Result<RenderedPrompt> {
        let (parts, images) = render_with_images(history, &self.tools, self.effort)?;
        let image_tokens = images
            .iter()
            .try_fold(0usize, |sum, image| sum.checked_add(image.tokens()))
            .context("the images' token count overflows")?;
        ensure!(
            image_tokens <= budget,
            "the prompt's {} image(s) need {image_tokens} tokens, more than its budget of {budget}",
            images.len()
        );
        let within = |ids: &Vec<u32>, more: usize| -> Result<()> {
            let total = ids.len().checked_add(more).context("the prompt's length overflows")?;
            ensure!(total <= budget, "the prompt needs more than its budget of {budget} tokens");
            Ok(())
        };
        let mut token_ids = Vec::new();
        let mut spans = Vec::with_capacity(images.len());
        let mut images = images.into_iter();
        let mut row = 0;
        let mut pending = String::new();
        for part in &parts {
            match part {
                Segment::Content(text) => pending.push_str(text),
                Segment::Control(marker) => {
                    let text = self.encode_content(&pending)?;
                    within(&token_ids, text.len())?;
                    token_ids.extend(text);
                    pending.clear();
                    if *marker == Marker::ImagePlaceholder {
                        let image = images.next().context("an image placeholder without its image")?;
                        let len = image.tokens();
                        within(&token_ids, len)?;
                        token_ids
                            .try_reserve(len)
                            .with_context(|| format!("cannot allocate an image's {len} tokens"))?;
                        spans.push(ImageSpan { start: token_ids.len(), len, row, image });
                        token_ids.resize(token_ids.len() + len, marker.id());
                        row += len;
                    } else {
                        within(&token_ids, 1)?;
                        token_ids.push(marker.id());
                    }
                }
            }
        }
        ensure!(images.next().is_none(), "an image without its placeholder");
        let text = self.encode_content(&pending)?;
        within(&token_ids, text.len())?;
        token_ids.extend(text);
        Ok(RenderedPrompt { text: segments_text(&parts), token_ids, images: spans })
    }

    /// The exact bytes of non-control ids, by inverting the byte-level
    /// alphabet. Unlike display decoding, an incomplete character is not
    /// papered over with U+FFFD.
    fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let alphabet = byte_level_alphabet();
        let mut bytes = Vec::new();
        for &id in ids {
            let piece = self.tokenizer.id_to_token(id).with_context(|| format!("unknown token {id}"))?;
            ensure!(
                self.content.token_to_id(&piece) == Some(id),
                "token {id} ({piece:?}) is not ordinary text"
            );
            for ch in piece.chars() {
                bytes.push(*alphabet.get(&ch).with_context(|| format!("token {id} is not byte-level"))?);
            }
        }
        Ok(bytes)
    }

    fn decode_text(&self, ids: &[u32]) -> Result<String> {
        String::from_utf8(self.decode_bytes(ids)?).context("generated text is not complete UTF-8")
    }

    /// Every id spelled out, markers included, for diagnostics.
    pub fn display(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_else(|e| format!("<undecodable: {e}>"))
    }

    /// Parse one complete reply: the ids generated after the prompt, ending
    /// with [`END_OF_REPLY`] when `stop` is [`StopReason::EndOfReply`].
    ///
    /// A reply is a sequence of frames `<|message_model|> header kind body
    /// <|end_message|>` and then the end of reply. Thinking frames join into
    /// `reasoning` and text frames into `answer`, in order, as upstream's
    /// reply parser joins them; a call frame's header must be its own name.
    pub fn parse(&self, ids: &[u32], stop: StopReason) -> Completion {
        let raw_text = self.display(ids);
        let mut reasoning = String::new();
        let mut answer = String::new();
        let mut calls = Vec::new();
        let result = (|| -> Result<()> {
            let body = match stop {
                StopReason::EndOfReply => {
                    ensure!(ids.last() == Some(&END_OF_REPLY), "stopped at end of reply without emitting it");
                    &ids[..ids.len() - 1]
                }
                _ => ids,
            };
            let mut at = 0;
            let next_control = |from: usize| body[from..].iter().position(|id| Marker::from_id(*id).is_some() || self.content.id_to_token(*id).is_none()).map(|p| from + p);
            while at < body.len() {
                ensure!(body[at] == Marker::MessageModel.id(), "a reply frame must open with <|message_model|> (token {} at {at})", body[at]);
                let header_end = next_control(at + 1).context("reply ends inside a frame header")?;
                let header = self.decode_text(&body[at + 1..header_end])?;
                let kind = Marker::from_id(body[header_end])
                    .with_context(|| format!("token {} is not a protocol control", body[header_end]))?;
                let body_start = header_end + 1;
                let body_end = next_control(body_start).context("reply ends inside a frame")?;
                ensure!(
                    body[body_end] == Marker::EndMessage.id(),
                    "a {} frame is closed by token {} instead of <|end_message|>",
                    kind.spelling(),
                    body[body_end]
                );
                let text = self.decode_text(&body[body_start..body_end])?;
                match kind {
                    Marker::ContentThinking | Marker::ContentText => {
                        ensure!(header.is_empty(), "a {} frame has the header {header:?}", kind.spelling());
                        if kind == Marker::ContentThinking {
                            reasoning.push_str(&text);
                        } else {
                            answer.push_str(&text);
                        }
                    }
                    Marker::ContentInvokeToolJson => {
                        let call = self.tools.parse_call(&text)?;
                        ensure!(header == call.name(), "call frame header {header:?} names a different tool than its body {:?}", call.name());
                        calls.push(call);
                    }
                    other => bail!("{} cannot open a reply frame", other.spelling()),
                }
                at = body_end + 1;
            }
            Ok(())
        })();
        let protocol_error = result.err().map(|e| format!("{e:#}"));
        let proposals = if protocol_error.is_none() && stop == StopReason::EndOfReply { calls } else { Vec::new() };
        Completion { reasoning, answer, raw_text, protocol_error, proposals, stop }
    }
}

/// GPT-2's byte-to-unicode table, inverted: the alphabet a ByteLevel
/// tokenizer spells its pieces in.
#[cfg(feature = "tokenizer")]
fn byte_level_alphabet() -> std::collections::HashMap<char, u8> {
    let mut alphabet = std::collections::HashMap::new();
    let mut extra = 256;
    for byte in 0..=255u8 {
        let scalar = if (b'!'..=b'~').contains(&byte) || (0xa1..=0xac).contains(&byte) || byte >= 0xae {
            byte as u32
        } else {
            let scalar = extra;
            extra += 1;
            scalar
        };
        alphabet.insert(char::from_u32(scalar).expect("a byte-level scalar"), byte);
    }
    alphabet
}

#[cfg(test)]
#[path = "chat_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "chat_image_tests.rs"]
mod image_tests;
