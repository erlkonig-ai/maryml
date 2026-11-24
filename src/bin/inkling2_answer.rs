//! Inkling 2 Mini over a pipe: JSON lines of requests on stdin until end of
//! file, one JSON line of response each on stdout, in request order. The
//! weights bind once; every request gets a fresh [`MiniSession`].
//!
//! `kind: "chat"` renders a history through the release's chat template
//! ([`ChatCodec`]), prefills it in calls of [`PREFILL`], samples until
//! `END_OF_REPLY` or the request's budget, and returns the parsed reply. Tool
//! calls are parsed and returned, never run. `kind: "nll"` returns a text's
//! natural-log loss per token: no chat frame, the softmax over the whole logit
//! row in f64. Nothing is truncated but the nll text, at its own `max_tokens`,
//! nothing is retried, and no system text is added beyond the template's.
//!
//! Requests run one at a time: one that names a `cohort` (to be batched with a
//! peer) is answered with an error. A request that cannot be read or fails is
//! answered `stop: "error"` with the cause. A line with no `req` cannot be
//! answered; the process then exits 1 at the end of its input.
//!
//! usage: inkling2_answer <pile> <collection> <root> <tools_catalog.json>

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};
use mary::model_collection::parse_collection_handle;
use mary::models::inkling2_loop::chat::{
    self, ChatCodec, Completion, Effort, END_OF_REPLY, History, Invocation, Marker, Message, StopReason, ToolDefinition, Tools,
};
use mary::models::inkling2_loop::pile::MiniRoot;
use mary::models::inkling2_loop::session::{MAX_LOGIT_ROWS, MiniModel, MiniSession, Rows};
use mary::sampling::{Sampler, SamplingConfig};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use triblespace::prelude::Id;

/// Prompt tokens per call.
const PREFILL: usize = 4096;
/// The catalog's tools an indexed declaration shows in full.
const DETAILED: [&str; 2] = ["exec", "tool_schema"];

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Request {
    Chat(Chat),
    Nll(Nll),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Chat {
    #[serde(rename = "req")]
    _req: Value,
    effort: EffortName,
    system: String,
    tools: ToolSet,
    messages: Vec<Turn>,
    max_new_tokens: usize,
    sampling: Sampling,
    sampler_seed: u64,
    cohort: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Nll {
    #[serde(rename = "req")]
    _req: Value,
    text: String,
    max_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
enum EffortName {
    High,
    Low,
}

impl EffortName {
    fn effort(self) -> Effort {
        match self {
            EffortName::High => Effort::High,
            EffortName::Low => Effort::Low,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ToolSet {
    None,
    Index,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
enum Turn {
    User(String),
    Assistant { reasoning: String, answer: String, tool_calls: Vec<Call> },
    Tool { name: String, content: String },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Call {
    name: String,
    args: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sampling {
    greedy: bool,
    temperature: Option<f32>,
}

impl Sampling {
    fn config(&self) -> Result<SamplingConfig> {
        if self.greedy {
            return Ok(SamplingConfig::greedy());
        }
        let temperature = self.temperature.context("sampling without greedy needs a temperature")?;
        Ok(SamplingConfig { temperature, ..Default::default() })
    }
}

/// The request's messages as the codec's history.
fn history(system: String, turns: Vec<Turn>, tools: &Tools) -> Result<History> {
    let messages = turns
        .into_iter()
        .map(|turn| {
            Ok(match turn {
                Turn::User(text) => Message::User(text),
                Turn::Tool { name, content } => Message::Tool { name, content },
                Turn::Assistant { reasoning, answer, tool_calls } => Message::Assistant {
                    reasoning,
                    answer,
                    tool_calls: tool_calls.into_iter().map(|c| Invocation::new(c.name, c.args, tools)).collect::<Result<_>>()?,
                },
            })
        })
        .collect::<Result<_>>()?;
    Ok(History { system, rendered_opening: None, messages })
}

/// The catalog file: a list of `{name, description, parameters}`.
fn catalog(text: &str) -> Result<Vec<ToolDefinition>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Entry {
        name: String,
        description: String,
        parameters: Value,
    }
    let entries: Vec<Entry> = serde_json::from_str(text).context("the tool catalog")?;
    Ok(entries
        .into_iter()
        .map(|Entry { name, description, parameters }| ToolDefinition { name, description, parameters })
        .collect())
}

/// Sample from `row`, then from every row `decode` returns for the token just
/// emitted, until the end of reply (which is emitted) or `budget` tokens.
fn generate(
    mut row: Vec<f32>,
    sampler: &mut Sampler,
    budget: usize,
    mut decode: impl FnMut(u32) -> Result<Vec<f32>>,
) -> (Vec<u32>, StopReason) {
    let mut emitted = Vec::new();
    while emitted.len() < budget {
        let id = match sampler.token(&row) {
            Ok(id) => id as u32,
            Err(e) => return (emitted, StopReason::Error(format!("{e:#}"))),
        };
        emitted.push(id);
        if id == END_OF_REPLY {
            return (emitted, StopReason::EndOfReply);
        }
        if emitted.len() < budget {
            row = match decode(id) {
                Ok(row) => row,
                Err(e) => return (emitted, StopReason::Error(format!("{e:#}"))),
            };
        }
    }
    (emitted, StopReason::Budget)
}

/// Emitted ids inside thinking frames, markers left out.
fn reasoning_tokens(ids: &[u32]) -> usize {
    let (mut inside, mut n) = (false, 0);
    for &id in ids {
        if id == Marker::ContentThinking.id() {
            inside = true;
        } else if id == Marker::EndMessage.id() {
            inside = false;
        } else if inside {
            n += 1;
        }
    }
    n
}

/// Natural-log loss of `id` under the softmax of `row`.
fn loss(row: &[f32], id: u32) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    sum.ln() + max - row[id as usize] as f64
}

fn failed(req: &Value, cause: &anyhow::Error) -> Value {
    json!({"req": req, "stop": "error", "error": format!("{cause:#}")})
}

/// The response fields of a parsed reply.
fn reply(req: &Value, done: Completion, prompt_tokens: usize, ids: &[u32], prefill_s: f64, decode_s: f64) -> Value {
    let (stop, error) = match done.stop {
        StopReason::EndOfReply => ("end_of_reply", None),
        StopReason::Budget => ("budget", None),
        StopReason::Cancelled => ("cancelled", None),
        StopReason::Error(why) => ("error", Some(why)),
    };
    let proposals: Vec<Value> =
        done.proposals.iter().map(|c| json!({"name": c.name(), "args": c.arguments(), "body": c.body()})).collect();
    let mut out = json!({
        "req": req, "stop": stop, "reasoning": done.reasoning, "answer": done.answer, "raw_text": done.raw_text,
        "protocol_error": done.protocol_error, "proposals": proposals, "prompt_tokens": prompt_tokens,
        "new_tokens": ids.len(), "reasoning_tokens": reasoning_tokens(ids),
        "prefill_s": prefill_s, "decode_s": decode_s,
    });
    if let Some(error) = error {
        out["error"] = error.into();
    }
    out
}

struct Answerer {
    model: Arc<MiniModel>,
    catalog: Vec<ToolDefinition>,
    codecs: HashMap<(ToolSet, EffortName), ChatCodec>,
}

impl Answerer {
    fn codec(&mut self, tools: ToolSet, effort: EffortName) -> Result<&ChatCodec> {
        if !self.codecs.contains_key(&(tools, effort)) {
            let root = self.model.root();
            let declared = match tools {
                ToolSet::None => Tools::default(),
                ToolSet::Index => Tools::indexed(self.catalog.clone(), &DETAILED)?,
            };
            let codec = ChatCodec::from_graph(root.facts(), root.store(), root.tokenizer_root(), declared, effort.effort())?;
            self.codecs.insert((tools, effort), codec);
        }
        Ok(&self.codecs[&(tools, effort)])
    }

    fn chat(&mut self, req: &Value, chat: Chat) -> Result<Value> {
        ensure!(chat.cohort.is_none(), "this process answers one request at a time and cannot run a cohort");
        let mut sampler = Sampler::new(chat.sampling.config()?, chat.sampler_seed)?;
        let model = Arc::clone(&self.model);
        let codec = self.codec(chat.tools, chat.effort)?;
        let history = history(chat.system, chat.messages, codec.tools())?;
        let prompt = codec.render(&history)?.token_ids;
        let mut session = MiniSession::new(model);
        let t = Instant::now();
        let mut row = Err(anyhow!("the prompt is empty"));
        for ids in prompt.chunks(PREFILL) {
            row = session.prefill(ids);
            if row.is_err() {
                break;
            }
        }
        let prefill_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (ids, stop) = match row {
            Ok(logits) => generate(logits.values, &mut sampler, chat.max_new_tokens, |id| Ok(session.decode(id)?.values)),
            Err(e) => (Vec::new(), StopReason::Error(format!("{e:#}"))),
        };
        let decode_s = t.elapsed().as_secs_f64();
        Ok(reply(req, codec.parse(&ids, stop), prompt.len(), &ids, prefill_s, decode_s))
    }

    fn nll(&mut self, req: &Value, nll: Nll) -> Result<Value> {
        let model = Arc::clone(&self.model);
        let mut ids = self.codec(ToolSet::None, EffortName::High)?.encode_content(&nll.text)?;
        ids.truncate(nll.max_tokens);
        let mut session = MiniSession::new(model);
        let mut losses = Vec::with_capacity(ids.len());
        for (call, chunk) in ids.chunks(MAX_LOGIT_ROWS).enumerate() {
            let logits = session.append(chunk, Rows::Tail(chunk.len()))?;
            for i in 0..logits.rows {
                // the last row of the text predicts nothing that is there
                if let Some(&next) = ids.get(call * MAX_LOGIT_ROWS + i + 1) {
                    losses.push(loss(logits.row(i), next));
                }
            }
        }
        Ok(json!({"req": req, "ids": ids, "nll": losses}))
    }

    /// The response to one line, or why it has none.
    fn answer(&mut self, line: &str) -> Result<Value> {
        let value: Value = serde_json::from_str(line).context("the line is not JSON")?;
        let req = value.get("req").context("the line has no req")?.clone();
        Ok(match serde_json::from_value(value).context("the request") {
            Ok(Request::Chat(chat)) => self.chat(&req, chat),
            Ok(Request::Nll(nll)) => self.nll(&req, nll),
            Err(e) => Err(e),
        }
        .unwrap_or_else(|e| failed(&req, &e)))
    }
}

fn main() -> Result<()> {
    // before the first device use
    mary::models::inkling::pool::choose_memory_config();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [pile, collection, root, tools] = args.as_slice() else {
        anyhow::bail!("usage: inkling2_answer <pile> <collection> <root> <tools_catalog.json>");
    };
    let catalog = catalog(&std::fs::read_to_string(tools).with_context(|| tools.clone())?)?;
    Tools::indexed(catalog.clone(), &DETAILED).context("the tool catalog")?;
    let collection = parse_collection_handle(collection)?;
    let id = Id::from_hex(root).context("the model root is not a 32-hex id")?;
    let root = MiniRoot::open(std::path::Path::new(pile), collection, id)?;
    chat::check_template(&root.chat_template()?)?;
    let model = Arc::new(MiniModel::from_root(root)?);
    let mut answerer = Answerer { model, catalog, codecs: HashMap::new() };

    let (mut lines, mut unanswered) = (0usize, 0usize);
    for line in std::io::stdin().lock().lines() {
        let line = line.context("read stdin")?;
        lines += 1;
        match answerer.answer(&line) {
            Ok(response) => {
                let mut out = std::io::stdout().lock();
                writeln!(out, "{response}")?;
                out.flush()?;
            }
            Err(e) => {
                unanswered += 1;
                eprintln!("line {lines}: {e:#}");
            }
        }
    }
    ensure!(unanswered == 0, "{unanswered} of {lines} lines could not be answered");
    Ok(())
}

#[cfg(test)]
mod tests {
    //! No device and no weights: parsing, the history, the stop reasons, the
    //! seeds and the loss.
    use super::*;
    use mary::models::inkling2_loop::chat::render_text;

    fn request(change: impl FnOnce(&mut Value)) -> Result<Request> {
        let mut v = json!({
            "kind": "chat", "req": "a#1#0", "effort": "high", "system": "Sys.", "tools": "index",
            "messages": [{"user": "hi"}], "max_new_tokens": 8,
            "sampling": {"greedy": false, "temperature": 1.0}, "sampler_seed": 7, "cohort": null,
        });
        change(&mut v);
        // through text, as a line arrives
        Ok(serde_json::from_value(serde_json::from_str(&v.to_string())?)?)
    }

    fn chat(change: impl FnOnce(&mut Value)) -> Chat {
        match request(change).unwrap() {
            Request::Chat(chat) => chat,
            other => panic!("{other:?}"),
        }
    }

    fn tools() -> Tools {
        let text = r#"[{"name":"exec","description":"Run.","parameters":{"type":"object"}},
                       {"name":"tool_schema","description":"Read. One.","parameters":{"type":"object"}},
                       {"name":"dim","description":"Dim a lamp. Slowly.","parameters":{"type":"object","properties":{}}}]"#;
        Tools::indexed(catalog(text).unwrap(), &DETAILED).unwrap()
    }

    #[test]
    fn a_seed_is_read_as_u64() {
        for seed in [0u64, (1 << 53) + 1, u64::MAX, u64::MAX - 1] {
            assert_eq!(chat(|v| v["sampler_seed"] = json!(seed)).sampler_seed, seed);
        }
        assert!(request(|v| v["sampler_seed"] = json!(1.5)).is_err());
        assert!(request(|v| v["sampler_seed"] = json!(-1)).is_err());
    }

    #[test]
    fn requests_are_read_exactly() {
        assert_eq!(chat(|v| v["effort"] = json!("low")).effort.effort().spelling(), "0.2");
        assert_eq!(chat(|_| ()).effort.effort().spelling(), "0.9");
        assert_eq!(chat(|v| v["tools"] = json!("none")).tools, ToolSet::None);
        assert!(chat(|v| v["cohort"] = json!({"id": "p", "slot": 0})).cohort.is_some());
        assert!(chat(|_| ()).cohort.is_none());
        for (key, bad) in [("effort", json!("medium")), ("tools", json!("all")), ("messages", json!([{"system": "x"}])),
            ("messages", json!([{"user": "x", "tool": {"name": "a", "content": "b"}}])), ("max_new_tokens", json!(-1)),
            ("kind", json!("batch")), ("new_field", json!(1))] {
            assert!(request(|v| v[key] = bad.clone()).is_err(), "{key} {bad} was accepted");
        }
        let Request::Nll(nll) = serde_json::from_str(r#"{"kind":"nll","req":"nll:x","text":"abc","max_tokens":2048}"#).unwrap() else { panic!() };
        assert_eq!((nll.text.as_str(), nll.max_tokens), ("abc", 2048));
    }

    #[test]
    fn sampling_is_the_releases_or_greedy() {
        let release = chat(|_| ()).sampling.config().unwrap();
        assert_eq!(release, SamplingConfig::default());
        assert_eq!(chat(|v| v["sampling"] = json!({"greedy": false, "temperature": 0.5})).sampling.config().unwrap().temperature, 0.5);
        assert!(chat(|v| v["sampling"] = json!({"greedy": true})).sampling.config().unwrap().is_greedy());
        assert!(chat(|v| v["sampling"] = json!({"greedy": false})).sampling.config().is_err());
    }

    #[test]
    fn messages_become_the_history() {
        let turns = json!([
            {"user": "dim it"},
            {"assistant": {"reasoning": "look", "answer": "", "tool_calls": [{"name": "tool_schema", "args": {"name": "dim"}}]}},
            {"tool": {"name": "tool_schema", "content": "{\"name\":\"dim\"}"}},
            {"assistant": {"reasoning": "", "answer": "Which one?", "tool_calls": []}},
            {"user": "both"},
        ]);
        let chat = chat(|v| v["messages"] = turns);
        let tools = tools();
        let h = history(chat.system, chat.messages, &tools).unwrap();
        let text = render_text(&h, &tools, Effort::High).unwrap();
        let after = text.split_once("Thinking effort level: 0.9<|end_message|>").unwrap().1;
        assert_eq!(
            after,
            "<|message_user|><|content_text|>dim it<|end_message|>\
             <|message_model|><|content_thinking|>look<|end_message|>\
             <|message_model|>tool_schema<|content_invoke_tool_json|>{\"name\":\"tool_schema\",\"args\":{\"name\":\"dim\"}}<|end_message|>\
             <|content_model_end_sampling|>\
             <|message_tool|>tool_schema<|content_text|>{\"name\":\"dim\"}<|end_message|>\
             <|message_model|><|content_text|>Which one?<|end_message|><|content_model_end_sampling|>\
             <|message_user|><|content_text|>both<|end_message|>"
        );
        assert!(text.contains(r#"{"description":"Dim a lamp.","name":"dim","parameters":{"type":"object"},"type":"function"}"#), "{text}");
        assert!(text.contains("<|message_system|><|content_text|>Sys.<|end_message|>"));
        let undeclared = serde_json::from_value(json!([{"assistant": {"reasoning": "", "answer": "", "tool_calls": [{"name": "x", "args": {}}]}}])).unwrap();
        assert!(history(String::new(), undeclared, &tools).is_err());
    }

    /// A row of `n` logits whose argmax is `id`.
    fn peak(n: usize, id: u32) -> Vec<f32> {
        (0..n).map(|i| if i == id as usize { 9.0 } else { 0.0 }).collect()
    }

    /// `script[k]` is the argmax after the k-th emitted token.
    fn run(first: u32, script: &[u32], budget: usize) -> (Vec<u32>, StopReason, Vec<u32>) {
        let n = END_OF_REPLY as usize + 1;
        let mut fed = Vec::new();
        let mut sampler = Sampler::new(SamplingConfig::greedy(), 0).unwrap();
        let (ids, stop) = generate(peak(n, first), &mut sampler, budget, |id| {
            fed.push(id);
            let next = script.get(fed.len() - 1).ok_or_else(|| anyhow!("scripted failure"))?;
            Ok(peak(n, *next))
        });
        (ids, stop, fed)
    }

    #[test]
    fn generation_stops_for_each_reason() {
        // the end of reply is emitted and counted, and is not decoded
        assert_eq!(run(5, &[6, END_OF_REPLY], 8), (vec![5, 6, END_OF_REPLY], StopReason::EndOfReply, vec![5, 6]));
        // at the budget it is still the end of reply
        assert_eq!(run(5, &[6, END_OF_REPLY], 3), (vec![5, 6, END_OF_REPLY], StopReason::EndOfReply, vec![5, 6]));
        // the budget's last token is not decoded
        assert_eq!(run(5, &[6, 7, 8], 2), (vec![5, 6], StopReason::Budget, vec![5]));
        assert_eq!(run(5, &[], 1), (vec![5], StopReason::Budget, vec![]));
        assert_eq!(run(END_OF_REPLY, &[], 1), (vec![END_OF_REPLY], StopReason::EndOfReply, vec![]));
        assert_eq!(run(5, &[6], 8), (vec![5, 6], StopReason::Error("scripted failure".into()), vec![5, 6]));
        let mut sampler = Sampler::new(SamplingConfig::greedy(), 0).unwrap();
        let (ids, stop) = generate(vec![f32::NAN; 4], &mut sampler, 4, |_| unreachable!());
        assert!(ids.is_empty() && matches!(stop, StopReason::Error(_)));
    }

    #[test]
    fn the_seed_alone_decides_a_sampled_reply() {
        let run = |seed| {
            let mut sampler = Sampler::new(SamplingConfig::default(), seed).unwrap();
            generate(vec![0.0; 4096], &mut sampler, 64, |_| Ok(vec![0.0; 4096])).0
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
        // seeds that one f64 would hold as the same number
        assert_ne!(run((1 << 53) + 1), run(1 << 53));
    }

    #[test]
    fn the_response_names_every_field() {
        let done = |stop| Completion {
            reasoning: "r".into(), answer: "a".into(), raw_text: "raw".into(), protocol_error: None,
            proposals: vec![tools().parse_call("{\"name\":\"dim\", \"args\":{\"x\":1}}").unwrap()], stop,
        };
        let thinking = [Marker::MessageModel.id(), Marker::ContentThinking.id(), 7, 8, Marker::EndMessage.id(), 9, END_OF_REPLY];
        let v = reply(&json!("q"), done(StopReason::EndOfReply), 11, &thinking, 0.5, 0.25);
        assert_eq!(v, json!({"req": "q", "stop": "end_of_reply", "reasoning": "r", "answer": "a", "raw_text": "raw",
            "protocol_error": null, "proposals": [{"name": "dim", "args": {"x": 1}, "body": "{\"name\":\"dim\", \"args\":{\"x\":1}}"}],
            "prompt_tokens": 11, "new_tokens": 7, "reasoning_tokens": 2, "prefill_s": 0.5, "decode_s": 0.25}));
        let stop = |s| reply(&json!("q"), done(s), 1, &[], 0.0, 0.0);
        assert_eq!(stop(StopReason::Budget)["stop"], "budget");
        assert_eq!(stop(StopReason::Cancelled)["stop"], "cancelled");
        let v = stop(StopReason::Error("why".into()));
        assert_eq!((&v["stop"], &v["error"]), (&json!("error"), &json!("why")));
        // an unfinished thinking frame counts to its end
        assert_eq!(reasoning_tokens(&thinking[..4]), 2);
    }

    #[test]
    fn a_loss_is_the_natural_log_over_the_whole_row() {
        let row = [0.0f32, 1.0, -2.0, 3.5];
        let total: f64 = row.iter().map(|&v| (v as f64).exp()).sum();
        for (id, &v) in row.iter().enumerate() {
            assert!((loss(&row, id as u32) - (total.ln() - v as f64)).abs() < 1e-12);
        }
        assert!((loss(&[4.0; 8], 3) - 8f64.ln()).abs() < 1e-12);
    }
}
