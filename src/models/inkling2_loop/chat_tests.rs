use super::*;
use serde_json::json;

/// Upstream's own rendering: the release's tokenizer and chat template run by
/// transformers' `apply_chat_template`, offline, in the pinned vLLM v0.26.0
/// image (`fixtures/chat_reference.py`). Digest in
/// `fixtures/chat_reference.json.sha256`.
const REFERENCE: &str = include_str!("fixtures/chat_reference.json");

fn effort_of(name: &Value) -> Effort {
    match name.as_str() {
        None => Effort::default(),
        Some("none") => Effort::None,
        Some("minimal") => Effort::Minimal,
        Some("low") => Effort::Low,
        Some("medium") => Effort::Medium,
        Some("high") => Effort::High,
        Some("xhigh") => Effort::Xhigh,
        Some("max") => Effort::Max,
        Some(other) => panic!("unknown effort {other}"),
    }
}

fn tools_of(case: &Value) -> Tools {
    let Some(list) = case["tools"].as_array() else { return Tools::default() };
    Tools::new(
        list.iter()
            .map(|tool| ToolDefinition {
                name: tool["function"]["name"].as_str().unwrap().into(),
                description: tool["function"]["description"].as_str().unwrap().into(),
                parameters: tool["function"]["parameters"].clone(),
            })
            .collect(),
    )
    .unwrap()
}

fn history_of(case: &Value, tools: &Tools) -> History {
    let mut history = History::default();
    for message in case["messages"].as_array().unwrap() {
        let text = |key: &str| message[key].as_str().unwrap_or("").to_owned();
        match message["role"].as_str().unwrap() {
            "system" => history.system = text("content"),
            "user" => history.messages.push(Message::User(text("content"))),
            "tool" => history.messages.push(Message::Tool { name: text("name"), content: text("content") }),
            "assistant" => {
                let tool_calls = message["tool_calls"]
                    .as_array()
                    .map(|calls| {
                        calls
                            .iter()
                            .map(|call| {
                                Invocation::new(
                                    call["function"]["name"].as_str().unwrap(),
                                    call["function"]["arguments"].as_object().unwrap().clone(),
                                    tools,
                                )
                                .unwrap()
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                history.messages.push(Message::Assistant {
                    reasoning: text("reasoning_content"),
                    answer: text("content"),
                    tool_calls,
                });
            }
            other => panic!("unexpected role {other}"),
        }
    }
    history
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(REFERENCE).unwrap().as_array().unwrap().clone()
}

#[test]
fn the_reference_digest_matches() {
    let digest = include_str!("fixtures/chat_reference.json.sha256");
    assert!(digest.starts_with("b2874bae7a67216703fe0ff3d94aef22f65a8607d0cc48a4a1d22af2f956eb01"));
    assert_eq!(cases().len(), 7, "five conversations at the default effort, plus none and max");
}

#[test]
fn only_the_release_template_is_accepted() {
    check_template(RELEASE_TEMPLATE).unwrap();
    assert!(check_template(&RELEASE_TEMPLATE.replace("tool_declare", "tools")).is_err());
}

#[test]
fn renders_every_reference_conversation_as_upstream_does() {
    for case in cases() {
        let tools = tools_of(&case);
        let history = history_of(&case, &tools);
        let text = render_text(&history, &tools, effort_of(&case["effort"])).unwrap();
        assert_eq!(text, case["text"].as_str().unwrap(), "{} at effort {}", case["name"], case["effort"]);
    }
}

#[test]
fn a_kept_opening_renders_instead_of_the_system_and_tools() {
    let case = &cases()[2];
    let tools = tools_of(case);
    let mut history = history_of(case, &tools);
    history.rendered_opening = Some(render_opening(&history.system, &tools, Effort::default()));
    history.system = "ignored once an opening is kept".into();
    assert_eq!(render_text(&history, &tools, Effort::None).unwrap(), case["text"].as_str().unwrap());
}

#[test]
fn a_host_call_body_is_the_templates_canonical_body() {
    let tools = tools_of(&cases()[2]);
    let mut arguments = Map::new();
    arguments.insert("query".into(), json!("x"));
    arguments.insert("limit".into(), json!(3));
    let call = Invocation::new("memory_search", arguments, &tools).unwrap();
    assert_eq!(call.body(), r#"{"name":"memory_search","args":{"limit":3,"query":"x"}}"#);
    assert_eq!(tools.parse_call(call.body()).unwrap(), call);
    assert!(Invocation::new("undeclared", Map::new(), &tools).is_err());
}

/// The release's own tokenizer, rebuilt from the model root's tokenizer
/// graph, against upstream's ids. Needs the imported release:
/// `MINI_MODEL_PILE`, `MINI_MODEL_COLLECTION` (hex) and `MINI_MODEL_ROOT`.
#[test]
#[ignore = "needs the imported Mini release (MINI_MODEL_PILE, MINI_MODEL_COLLECTION, MINI_MODEL_ROOT)"]
fn tokenizes_every_reference_conversation_as_upstream_does() {
    let env = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"));
    let collection = crate::model_collection::parse_collection_handle(&env("MINI_MODEL_COLLECTION")).unwrap();
    let root = triblespace::prelude::Id::from_hex(&env("MINI_MODEL_ROOT")).unwrap();
    let mini = super::super::pile::MiniRoot::open(std::path::Path::new(&env("MINI_MODEL_PILE")), collection, root).unwrap();
    check_template(&mini.chat_template().unwrap()).unwrap();
    // Build the two tokenizers once; every case shares them.
    let whole = crate::tokenizer::build_tokenizer_with_added(mini.facts(), mini.store(), mini.tokenizer_root(), true).unwrap();
    let content = crate::tokenizer::build_tokenizer_with_added(mini.facts(), mini.store(), mini.tokenizer_root(), false).unwrap();
    for case in cases() {
        let tools = tools_of(&case);
        let codec = ChatCodec::new(whole.clone(), content.clone(), tools.clone(), effort_of(&case["effort"])).unwrap();
        let history = history_of(&case, &tools);
        let rendered = codec.render(&history).unwrap();
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(rendered.text, case["text"].as_str().unwrap());
        assert_eq!(rendered.token_ids, expected, "{} at effort {}", case["name"], case["effort"]);
    }
    // Fixed strings outside any conversation.
    let codec = ChatCodec::new(whole, content, Tools::default(), Effort::default()).unwrap();
    for text in ["Hello there.", "Grüße — what's new? 🙂", "  a\r\n\tb  ", "<|end_message|>"] {
        let ids = codec.encode_content(text).unwrap();
        assert!(ids.iter().all(|id| Marker::from_id(*id).is_none()), "{text:?} encoded a control");
        assert_eq!(codec.decode_text(&ids).unwrap(), text);
    }
}

/// A byte-level tokenizer with no merges whose vocabulary also holds the
/// protocol's controls at the release's ids: enough to test the parser
/// without the release.
fn tiny_codec(tools: Tools) -> ChatCodec {
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, Tokenizer};
    let bytes: tokenizers::models::bpe::Vocab =
        byte_level_alphabet().into_iter().map(|(ch, byte)| (ch.to_string(), byte as u32)).collect();
    let build = |with_markers: bool| {
        let mut vocab = bytes.clone();
        if with_markers {
            for marker in Marker::ALL {
                vocab.insert(marker.spelling().into(), marker.id());
            }
        }
        let model = BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        if with_markers {
            tokenizer.add_special_tokens(
                &Marker::ALL.map(|marker| AddedToken::from(marker.spelling(), true)),
            );
        }
        tokenizer
    };
    ChatCodec::new(build(true), build(false), tools, Effort::default()).unwrap()
}

fn exec_tools() -> Tools {
    Tools::new(vec![ToolDefinition {
        name: "exec".into(),
        description: "Propose a shell command.".into(),
        parameters: json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
    }])
    .unwrap()
}

fn reply(codec: &ChatCodec, parts: &[(Marker, &str, &str)]) -> Vec<u32> {
    let mut ids = Vec::new();
    for (kind, header, body) in parts {
        ids.push(Marker::MessageModel.id());
        ids.extend(codec.encode_content(header).unwrap());
        ids.push(kind.id());
        ids.extend(codec.encode_content(body).unwrap());
        ids.push(Marker::EndMessage.id());
    }
    ids.push(END_OF_REPLY);
    ids
}

#[test]
fn a_complete_reply_parses_into_thinking_text_and_its_calls() {
    let codec = tiny_codec(exec_tools());
    let body = r#"{"name":"exec","args":{"command":"printf hi"}}"#;
    let ids = reply(&codec, &[
        (Marker::ContentThinking, "", "Plan."),
        (Marker::ContentText, "", "Looking."),
        (Marker::ContentInvokeToolJson, "exec", body),
    ]);
    let completion = codec.parse(&ids, StopReason::EndOfReply);
    assert_eq!(completion.protocol_error, None);
    assert_eq!((completion.reasoning.as_str(), completion.answer.as_str()), ("Plan.", "Looking."));
    assert_eq!(completion.proposals.len(), 1);
    assert_eq!(completion.proposals[0].body(), body);
    assert_eq!(completion.proposals[0].arguments()["command"], "printf hi");
    // Re-rendered as history, the reply is the same tokens the model emitted.
    let history = History {
        system: String::new(),
        rendered_opening: Some(Vec::new()),
        messages: vec![Message::Assistant {
            reasoning: completion.reasoning,
            answer: completion.answer,
            tool_calls: completion.proposals,
        }],
    };
    assert_eq!(codec.render(&history).unwrap().token_ids, ids);
}

#[test]
fn split_frames_join_in_order_as_upstreams_reply_parser_joins_them() {
    let codec = tiny_codec(exec_tools());
    let ids = reply(&codec, &[
        (Marker::ContentText, "", "one "),
        (Marker::ContentThinking, "", "aside"),
        (Marker::ContentText, "", "two"),
    ]);
    let completion = codec.parse(&ids, StopReason::EndOfReply);
    assert_eq!(completion.protocol_error, None);
    assert_eq!((completion.reasoning.as_str(), completion.answer.as_str()), ("aside", "one two"));
}

#[test]
fn malformed_or_unfinished_replies_propose_nothing() {
    let codec = tiny_codec(exec_tools());
    let body = r#"{"name":"exec","args":{"command":"rm -rf scratch"}}"#;
    let call = reply(&codec, &[(Marker::ContentInvokeToolJson, "exec", body)]);
    // The same call, cut off before the end of reply.
    let cut = codec.parse(&call[..call.len() - 1], StopReason::Budget);
    assert!(cut.proposals.is_empty());
    assert_eq!(cut.protocol_error, None, "a budget stop is not a protocol fault");
    // A header naming another tool.
    let lying = reply(&codec, &[(Marker::ContentInvokeToolJson, "other", body)]);
    let parsed = codec.parse(&lying, StopReason::EndOfReply);
    assert!(parsed.proposals.is_empty() && parsed.protocol_error.is_some());
    // An undeclared tool.
    let undeclared = reply(&codec, &[(Marker::ContentInvokeToolJson, "shell", r#"{"name":"shell","args":{}}"#)]);
    assert!(codec.parse(&undeclared, StopReason::EndOfReply).protocol_error.is_some());
    // A frame left open at the end of reply.
    let mut open = vec![Marker::MessageModel.id(), Marker::ContentText.id()];
    open.extend(codec.encode_content("unfinished").unwrap());
    open.push(END_OF_REPLY);
    let parsed = codec.parse(&open, StopReason::EndOfReply);
    assert!(parsed.protocol_error.is_some() && parsed.proposals.is_empty());
    assert!(parsed.raw_text.contains("unfinished"));
    // A good call followed by a bad one voids both.
    let mut two = reply(&codec, &[(Marker::ContentInvokeToolJson, "exec", body)]);
    two.pop();
    two.extend(reply(&codec, &[(Marker::ContentInvokeToolJson, "exec", "{not json")]));
    assert!(codec.parse(&two, StopReason::EndOfReply).proposals.is_empty());
}

#[test]
fn a_literal_marker_in_content_stays_text() {
    let codec = tiny_codec(exec_tools());
    let history = History {
        system: "sys".into(),
        rendered_opening: None,
        messages: vec![Message::User("say <|end_message|> please".into())],
    };
    let rendered = codec.render(&history).unwrap();
    let user = &rendered.token_ids[rendered.token_ids.iter().rposition(|id| *id == Marker::ContentText.id()).unwrap() + 1..];
    assert_eq!(user.iter().filter(|id| **id == Marker::EndMessage.id()).count(), 1, "only the frame's own close");
    assert_eq!(codec.decode_text(&user[..user.len() - 1]).unwrap(), "say <|end_message|> please");
}
