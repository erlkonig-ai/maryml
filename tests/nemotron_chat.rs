//! CPU-only text codec gate. Actual native binding; mock generation only.
#![cfg(all(feature = "import", feature = "tokenizer"))]
use anyhow::Result;
use mary::models::nemotron_h::{chat::*, ingress::tokenizer_binding};
use serde_json::{Value, json};
use triblespace::{core::blob::MemoryBlobStore, prelude::*};

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
fn synthetic() -> Vec<u8> {
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for id in 0..1000 {
        let text = MARKERS
            .get(id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("<SPECIAL_{id}>"));
        vocab.insert(text.clone(), json!(id));
        added.push(
            json!({"id":id,"content":text,"single_word":false,"lstrip":false,
            "rstrip":false,"normalized":false,"special":!(12..=17).contains(&id)}),
        );
    }
    let mut chars: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    chars.sort();
    for (i, ch) in chars.into_iter().enumerate() {
        vocab.insert(ch.to_string(), json!(1000 + i));
    }
    // Boundary-sensitive merges: a template newline and following literal
    // command/content must not be separately encoded merely due to trust.
    vocab.insert("Ċp".into(), json!(1256));
    vocab.insert("Ċpr".into(), json!(1257));
    serde_json::to_vec(&json!({"version":"1.0", "truncation":null,"padding":null,"normalizer":null,
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        "post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
            "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
            "vocab":vocab,"merges":[["Ċ","p"],["Ċp","r"]]},"added_tokens":added})).unwrap()
}
fn bound(bytes: &[u8]) -> Result<(ChatCodec, tokenizers::Tokenizer)> {
    let mut blobs = MemoryBlobStore::new();
    let fragment = tokenizer_binding::save(bytes, "chat-test", &mut blobs)?;
    // Native entity! string values travel in the fragment's embedded blobs;
    // the generic tokenizer bulk was written into the explicit destination.
    // Publication joins both; this CPU-only snapshot must do the same.
    blobs.union(fragment.blobs().clone());
    let binding = fragment.root().unwrap();
    let facts = fragment.facts();
    let base = tokenizer_binding::bases(facts, binding).next().unwrap();
    let sequence = tokenizer_binding::pre_tokenizers(facts, binding)
        .next()
        .unwrap();
    let blobs = blobs.snapshot()?;
    Ok((
        ChatCodec::rebuild(facts, &blobs, binding, base, sequence)?,
        // Expected ordinary IDs come directly from the original tokenizer,
        // independently of the native binding reconstruction under test.
        tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|error| anyhow::anyhow!("source tokenizer: {error}"))?,
    ))
}
fn exec_invocation(command: &str) -> Invocation {
    Invocation::new(
        "exec",
        json!({"command":command}).as_object().unwrap().clone(),
        &Tools::exec_only(),
    )
    .unwrap()
}
fn exec_reference(history: &History, options: RenderOptions) -> Result<String> {
    render_reference_text(history, options, &Tools::exec_only())
}
fn options(thinking: bool) -> RenderOptions {
    RenderOptions {
        enable_thinking: thinking,
        medium_effort: false,
        add_generation_prompt: true,
        tools_enabled: false,
    }
}
fn user(text: &str) -> History {
    History {
        system: String::new(),
        messages: vec![Message::User(text.into())],
    }
}
fn result(text: &str) -> Message {
    Message::ToolResult {
        content: text.into(),
    }
}
fn call(codec: &ChatCodec, command: &str, eos: Option<u32>) -> Vec<u32> {
    let mut ids = vec![14];
    ids.extend(
        codec
            .encode_content(&format!(
                "\n<function=exec>\n<parameter=command>\n{command}\n</parameter>\n</function>\n"
            ))
            .unwrap(),
    );
    ids.push(15);
    if let Some(id) = eos {
        ids.push(id);
    }
    ids
}
fn complete(codec: &ChatCodec, ids: &[u32], stop: StopReason) -> Completion {
    let mut parser = OutputParser::new(codec, false, 10000);
    assert!(parser.push(ids));
    parser.finish(stop).unwrap()
}

fn fixture(case: &Value) -> (History, RenderOptions) {
    let mut history = History::default();
    for msg in case["messages"].as_array().unwrap() {
        let text = msg["content"].as_str().unwrap_or_default().to_string();
        match msg["role"].as_str().unwrap() {
            "system" => history.system = text,
            "user" => history.messages.push(Message::User(text)),
            "assistant" => history.messages.push(Message::Assistant {
                reasoning: msg["reasoning_content"].as_str().unwrap_or_default().into(),
                answer: text,
                tool_call: msg["command"].as_str().map(exec_invocation),
            }),
            "tool" => history.messages.push(result(&text)),
            _ => unreachable!(),
        }
    }
    let o = RenderOptions {
        enable_thinking: case["thinking"].as_bool().unwrap(),
        medium_effort: case["medium"].as_bool().unwrap(),
        add_generation_prompt: case["generation"].as_bool().unwrap(),
        tools_enabled: case["tools"].as_bool().unwrap(),
    };
    (history, o)
}

#[test]
fn literal_reference_fixtures() -> Result<()> {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/nemotron_chat_reference.json"))?;
    for case in cases {
        let (history, o) = fixture(&case);
        assert_eq!(
            exec_reference(&history, o)?,
            case["expected"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
    Ok(())
}

fn ordinary_id_checks(codec: &ChatCodec, source: &tokenizers::Tokenizer) -> Result<()> {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/nemotron_chat_reference.json"))?;
    for case in cases {
        match case["parity"].as_str().unwrap() {
            "literal-departure" | "orphan-departure" => continue,
            "ordinary" => (),
            other => panic!("unclassified fixture {other}"),
        }
        let (history, options) = fixture(&case);
        let expected = case["expected"].as_str().unwrap();
        let prompt = codec.render_adapted(&history, options)?;
        assert_eq!(prompt.text, expected, "{}", case["name"]);
        assert_eq!(
            prompt.token_ids,
            source.encode(expected, false).unwrap().get_ids(),
            "ordinary publisher ID parity: {}",
            case["name"]
        );
        assert_eq!(codec.decode(&prompt.token_ids)?, expected);
    }
    Ok(())
}

#[test]
fn ordinary_histories_match_complete_source_tokenization() -> Result<()> {
    let (codec, source) = bound(&synthetic())?;
    assert_eq!(source.encode("\npr", false).unwrap().get_ids(), &[1257]);
    ordinary_id_checks(&codec, &source)
}

#[test]
fn full_window_payload_over_one_mebibyte_is_not_truncated() -> Result<()> {
    let (codec, source) = bound(&synthetic())?;
    // Existing native fixture merges three bytes into one ordinary token.
    // Byte length cannot substitute for the model's actual token capacity.
    let payload = "\npr".repeat(400_000);
    assert!(payload.len() > 1_048_576);
    let ids = codec.encode_content(&payload)?;
    assert_eq!(ids, vec![1257; 400_000]);
    assert_eq!(codec.decode(&ids)?, payload);

    let expected = format!(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{payload}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let history = user(&payload);
    assert_eq!(exec_reference(&history, options(true))?, expected);
    let prompt = codec.render_adapted(&history, options(true))?;
    assert_eq!(prompt.text, expected);
    assert_eq!(
        prompt.token_ids,
        source.encode(expected.as_str(), false).unwrap().get_ids()
    );
    assert_eq!(codec.decode(&prompt.token_ids)?, expected);
    assert!(prompt.token_ids.len() + 1024 < 1_048_576);
    Ok(())
}

#[test]
fn more_than_4096_messages_preserve_complete_history_and_reasoning() -> Result<()> {
    let (codec, source) = bound(&synthetic())?;
    let mut history = History::default();
    let mut expected = String::from("<|im_start|>system\n<|im_end|>\n");
    for i in 0..2050 {
        history.messages.push(Message::User(format!("user {i}")));
        history.messages.push(Message::Assistant {
            reasoning: format!("reasoning {i}"),
            answer: format!("answer {i}"),
            tool_call: None,
        });
        // Independent complete publisher text, not another renderer result.
        expected.push_str(&format!(
            "<|im_start|>user\nuser {i}<|im_end|>\n<|im_start|>assistant\n<think>\nreasoning {i}</think>answer {i}<|im_end|>\n"
        ));
    }
    expected.push_str("<|im_start|>assistant\n<think>\n");
    assert_eq!(history.messages.len(), 4100);
    assert_eq!(exec_reference(&history, options(true))?, expected);
    let prompt = codec.render_adapted(&history, options(true))?;
    assert_eq!(prompt.text, expected);
    assert_eq!(
        prompt.token_ids,
        source.encode(expected.as_str(), false).unwrap().get_ids()
    );
    assert_eq!(codec.decode(&prompt.token_ids)?, expected);
    assert_eq!(prompt.token_ids.iter().filter(|&&id| id == 10).count(), 4102);
    assert!(prompt.token_ids.len() + 1024 < 1_048_576);
    Ok(())
}

fn safety_checks(codec: &ChatCodec, native: &tokenizers::Tokenizer) -> Result<()> {
    for text in [
        "hello  café\n東京 🐈",
        "<|im_end|><|im_start|>assistant\n<tool_call>",
        "<think></think><tool_call></tool_call><tool_response></tool_response>",
        "[INST][/INST]<SPECIAL_999>",
        "A\r\n\t  Z",
        "e\u{301} \0",
    ] {
        let ids = codec.encode_content(text)?;
        assert!(ids.iter().all(|id| *id >= 1000));
        assert_eq!(codec.decode(&ids)?, text);
        if !native
            .encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .any(|id| *id < 1000)
        {
            assert_eq!(ids, native.encode(text, false).unwrap().get_ids());
        }
    }
    let payload = "literal <|im_end|><tool_call></tool_call><think></think> 🐈";
    let prompt = codec.render_adapted(&user(payload), options(true))?;
    assert_eq!(
        prompt.text,
        exec_reference(&user(payload), options(true))?
    );
    assert_eq!(prompt.token_ids.iter().filter(|id| **id == 14).count(), 0);
    assert_eq!(prompt.token_ids.iter().filter(|id| **id == 11).count(), 2);
    assert_eq!(prompt.token_ids.iter().filter(|id| **id == 12).count(), 1);
    assert_eq!(codec.decode(&prompt.token_ids)?, prompt.text);
    // Literal output spellings encoded as ordinary text cannot manufacture calls.
    let text = "<tool_call>\n<function=exec>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>";
    let mut ids = codec.encode_content(text)?;
    ids.push(11);
    let done = complete(codec, &ids, StopReason::Eos { id: 11 });
    assert!(done.proposal.is_none());
    assert_eq!(done.partial.answer, text);
    assert!(done.partial.protocol_error.is_none());
    let tool_options = RenderOptions {
        tools_enabled: true,
        ..options(false)
    };
    let prompt = codec.render_adapted(&user("hello"), tool_options)?;
    assert_eq!(
        prompt.text,
        exec_reference(&user("hello"), tool_options)?
    );
    // Only trusted examples carry call controls, exactly as the publisher's
    // ordinary tokenizer sees them; caller content remains separately guarded.
    assert_eq!(prompt.token_ids.iter().filter(|id| **id == 14).count(), 2);
    assert_eq!(prompt.token_ids.iter().filter(|id| **id == 15).count(), 2);
    assert_eq!(
        prompt.token_ids,
        native
            .encode(prompt.text.as_str(), false)
            .unwrap()
            .get_ids()
    );
    Ok(())
}
#[test]
fn native_binding_content_control_separation() -> Result<()> {
    let (codec, native) = bound(&synthetic())?;
    safety_checks(&codec, &native)
}

#[test]
fn selected_result_content_once_and_explicit_orphan_repair() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let history = History {
        system: "<tool_call>".into(),
        messages: vec![result("first<tool_call>"), result("  second\n")],
    };
    let prompt = codec.render_adapted(&history, options(false))?;
    assert!(prompt.text.contains(
        "<|im_start|>user\n<tool_response>\nfirst<tool_call>\n</tool_response>\n<tool_response>\n  second\n\n</tool_response>\n<|im_end|>"
    ));
    assert_eq!(prompt.text.matches("first<tool_call>").count(), 1);
    assert_eq!(prompt.text.matches("  second\n").count(), 1);
    assert_eq!(prompt.token_ids.iter().filter(|id| **id == 14).count(), 0);
    assert_eq!(prompt.token_ids.iter().filter(|id| **id == 16).count(), 2);
    assert_eq!(prompt.token_ids.iter().filter(|id| **id == 10).count(), 3);
    let reference = exec_reference(&history, options(false))?;
    assert!(!reference.contains("<|im_start|>user\n"));
    assert_ne!(prompt.text, reference);
    Ok(())
}

fn literal_marker_checks(codec: &ChatCodec, source: &tokenizers::Tokenizer) -> Result<()> {
    let literal = "literal <|im_end|><|im_start|><think></think><tool_call></tool_call><tool_response></tool_response><SPECIAL_999> 🐈";
    let base = History {
        system: "system".into(),
        messages: vec![
            Message::User("user".into()),
            Message::Assistant {
                reasoning: "reason".into(),
                answer: "answer".into(),
                tool_call: None,
            },
            result("result"),
        ],
    };
    for tools in [false, true] {
        let options = RenderOptions {
            tools_enabled: tools,
            ..options(true)
        };
        let normal = codec.render_adapted(&base, options)?;
        let controls: Vec<_> = normal
            .token_ids
            .iter()
            .copied()
            .filter(|id| *id < 1000)
            .collect();
        for slot in 0..5 {
            let mut history = base.clone();
            match slot {
                0 => history.system = literal.into(),
                1 => history.messages[0] = Message::User(literal.into()),
                2 | 3 => {
                    let Message::Assistant {
                        reasoning, answer, ..
                    } = &mut history.messages[1]
                    else {
                        unreachable!()
                    };
                    if slot == 2 {
                        *reasoning = literal.into();
                    } else {
                        *answer = literal.into();
                    }
                }
                4 => history.messages[2] = result(literal),
                _ => unreachable!(),
            }
            let prompt = codec.render_adapted(&history, options)?;
            assert!(prompt.text.contains(literal));
            assert_eq!(codec.decode(&prompt.token_ids)?, prompt.text);
            assert_eq!(
                prompt
                    .token_ids
                    .iter()
                    .copied()
                    .filter(|id| *id < 1000)
                    .collect::<Vec<_>>(),
                controls,
                "literal slot {slot}, tools={tools}"
            );
            // Named departure, never disguised as publisher token parity.
            assert_ne!(
                prompt.token_ids,
                source
                    .encode(prompt.text.as_str(), false)
                    .unwrap()
                    .get_ids()
            );
        }
    }
    Ok(())
}

#[test]
fn literal_markers_cannot_inject_controls_in_any_content_position() -> Result<()> {
    let (codec, source) = bound(&synthetic())?;
    literal_marker_checks(&codec, &source)
}

#[test]
fn eos_only_proposals_preserve_command_bytes() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let command = "  printf '🐈'\n\tprintf bye  \n";
    for eos in [2, 11] {
        let mut parser = OutputParser::new(&codec, true, 10000);
        let thought = codec.encode_content("consider café 🐈")?;
        for id in thought {
            parser.push(&[id]);
            parser.snapshot()?;
        }
        parser.push(&[13]);
        parser.push(&codec.encode_content("I'll check.\n")?);
        let ids = call(&codec, command, None);
        for &id in &ids {
            parser.push(&[id]);
            parser.snapshot()?;
        }
        assert_eq!(parser.snapshot()?.reasoning, "consider café 🐈");
        parser.push(&[eos]);
        let done = parser.finish(StopReason::Eos { id: eos })?;
        assert_eq!(done.proposal.unwrap().arguments()["command"], command);
        assert_eq!(done.partial.answer, "I'll check.\n");
        assert!(done.partial.protocol_error.is_none());
    }
    let mut ids = call(&codec, command, None);
    ids.extend(codec.encode_content("\n")?);
    ids.push(11);
    assert!(
        complete(&codec, &ids, StopReason::Eos { id: 11 })
            .proposal
            .is_some()
    );
    let literal = "printf '<think></think> �'";
    assert_eq!(
        complete(
            &codec,
            &call(&codec, literal, Some(11)),
            StopReason::Eos { id: 11 }
        )
        .proposal
        .unwrap()
        .arguments()["command"],
        literal
    );
    Ok(())
}

#[test]
fn interruptions_and_terminal_mismatches_never_propose() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let ids = call(&codec, "printf hi\n", None);
    for end in [ids.len() / 2, ids.len()] {
        for stop in [
            StopReason::Budget,
            StopReason::Cancelled,
            StopReason::Error("mock".into()),
        ] {
            let done = complete(&codec, &ids[..end], stop.clone());
            assert!(done.proposal.is_none());
            assert_eq!(done.stop, stop);
            assert!(!done.partial.raw_text.is_empty());
        }
    }
    for (terminal, reported) in [(None, 11), (Some(2), 11), (Some(11), 2), (Some(11), 12)] {
        let done = complete(
            &codec,
            &call(&codec, "pwd", terminal),
            StopReason::Eos { id: reported },
        );
        assert!(done.proposal.is_none());
        assert!(done.partial.protocol_error.is_some());
    }
    let mut parser = OutputParser::new(&codec, true, 1000);
    parser.push(&[13]);
    assert!(parser.finish(StopReason::Budget)?.proposal.is_none());
    let done = complete(
        &codec,
        &call(&codec, "pwd", Some(11)),
        StopReason::Cancelled,
    );
    assert!(done.proposal.is_none());
    Ok(())
}

#[test]
fn malformed_unknown_duplicate_multiple_and_suffix_are_errors() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let bodies = [
        "\n<function=other>\n<parameter=command>\npwd\n</parameter>\n</function>\n",
        "\n<function=exec>\n<parameter=other>\npwd\n</parameter>\n</function>\n",
        "\n<function=exec>\n<parameter=command>\na\n</parameter>\n<parameter=command>\nb\n</parameter>\n</function>\n",
        "{\"command\":\"pwd\"}",
        "",
        "\n<function=exec>\n<parameter=command>\nx</function>y\n</parameter>\n</function>\n",
    ];
    let mut cases = Vec::new();
    for body in bodies {
        let mut ids = vec![14];
        ids.extend(codec.encode_content(body)?);
        ids.extend([15, 11]);
        cases.push(ids);
    }
    let mut duplicate = call(&codec, "a", None);
    duplicate.extend(call(&codec, "b", Some(11)));
    cases.push(duplicate);
    for suffix in ["no", " ", "\n\n", "\ntext"] {
        let mut ids = call(&codec, "pwd", None);
        ids.extend(codec.encode_content(suffix)?);
        ids.push(11);
        cases.push(ids);
    }
    cases.extend([
        vec![14, 11],
        vec![12, 11],
        vec![16, 17, 11],
        vec![11, 11],
        vec![u32::MAX, 11],
    ]);
    // A complete closing frame cannot turn a truncated UTF-8 byte into a
    // replacement character inside an executable command.
    let mut invalid_utf8 = vec![14];
    invalid_utf8.extend(codec.encode_content("\n<function=exec>\n<parameter=command>\n")?);
    invalid_utf8.push(codec.encode_content("🐈")?[0]);
    invalid_utf8.extend(codec.encode_content("\n</parameter>\n</function>\n")?);
    invalid_utf8.extend([15, 11]);
    cases.push(invalid_utf8);
    for ids in cases {
        let done = complete(&codec, &ids, StopReason::Eos { id: 11 });
        assert!(done.proposal.is_none(), "{ids:?}");
        assert!(done.partial.protocol_error.is_some(), "{ids:?}");
    }
    let mut parser = OutputParser::new(&codec, false, 2);
    assert!(!parser.push(&[14, 15, 11]));
    assert!(
        parser
            .finish(StopReason::Eos { id: 11 })?
            .proposal
            .is_none()
    );
    Ok(())
}

#[test]
#[ignore = "explicit retained full-tokenizer gate; set NEMOTRON_TOKENIZER_JSON"]
fn retained_real_tokenizer_content_policy() -> Result<()> {
    let bytes = std::fs::read(std::env::var("NEMOTRON_TOKENIZER_JSON")?)?;
    let (codec, native) = bound(&bytes)?;
    safety_checks(&codec, &native)?;
    ordinary_id_checks(&codec, &native)?;
    literal_marker_checks(&codec, &native)?;
    let command = "  printf '東京 🐈'\n\tprintf bye  \n";
    let done = complete(
        &codec,
        &call(&codec, command, Some(11)),
        StopReason::Eos { id: 11 },
    );
    assert_eq!(done.proposal.unwrap().arguments()["command"], command);
    Ok(())
}

fn registered_tools() -> Result<Tools> {
    Tools::new(vec![
        ToolDefinition::exec(),
        ToolDefinition {
            name: "fixture_apply".into(),
            description: "Apply typed fixture data.".into(),
            input_schema: json!({"type":"object","properties":{
            "text":{"type":"string"},"empty":{"type":"string"},
            "flag":{"type":"boolean"},"nothing":{"type":"null"},
            "count":{"type":"integer"},"ratio":{"type":"number"},
            "items":{"type":"array","items":{"type":"string"}},
            "object":{"type":"object"}},"required":["text"],"additionalProperties":false}),
        },
        ToolDefinition {
            name: "fixture_empty".into(),
            description: "No arguments.".into(),
            input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
        },
    ])
}

fn body_ids(codec: &ChatCodec, body: &str) -> Result<Vec<u32>> {
    let mut ids = vec![14];
    ids.extend(codec.encode_content(body)?);
    ids.extend([15, 11]);
    Ok(ids)
}

#[test]
fn registered_typed_arguments_preserve_exact_body_and_unicode_boundaries() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let codec = codec.with_tools(registered_tools()?);
    let body = concat!(
        "\n<function=fixture_apply>\n",
        "<parameter=object>\n{ \"nested\": [true, null, {\"café\":\"東京 🐈\"}] }\n</parameter>\n",
        "<parameter=text>\n  true\n東京 🐈  \n</parameter>\n",
        "<parameter=empty>\n\n</parameter>\n",
        "<parameter=flag>\nFalse\n</parameter>\n",
        "<parameter=nothing>\nNone\n</parameter>\n",
        "<parameter=count>\n42\n</parameter>\n",
        "<parameter=ratio>\n1.25\n</parameter>\n",
        "<parameter=items>\n[\"\",\"é\"]\n</parameter>\n</function>\n"
    );
    let ids = body_ids(&codec, body)?;
    let mut parser = OutputParser::new(&codec, true, ids.len() + 100);
    for id in codec.encode_content("consider 🐈")? {
        parser.push(&[id]);
    }
    parser.push(&[13]);
    for id in codec.encode_content("I'll apply it.")? {
        parser.push(&[id]);
    }
    for id in ids {
        assert!(parser.push(&[id]));
        parser.snapshot()?;
    }
    let done = parser.finish(StopReason::Eos { id: 11 })?;
    assert!(done.partial.protocol_error.is_none());
    assert_eq!(done.partial.reasoning, "consider 🐈");
    assert_eq!(done.partial.answer, "I'll apply it.");
    let invocation = done.proposal.unwrap();
    assert_eq!(invocation.name(), "fixture_apply");
    assert_eq!(invocation.body(), body);
    assert_eq!(
        invocation.arguments(),
        json!({"object":{"nested":[true,null,{"café":"東京 🐈"}]},
        "text":"  true\n東京 🐈  ","empty":"","flag":false,"nothing":null,
        "count":42,"ratio":1.25,"items":["","é"]})
        .as_object()
        .unwrap()
    );
    let history = History {
        system: "system".into(),
        messages: vec![Message::Assistant {
            reasoning: done.partial.reasoning,
            answer: done.partial.answer,
            tool_call: Some(invocation),
        }],
    };
    let rendered = codec.render_adapted(&history, options(true))?;
    assert!(
        rendered
            .text
            .contains(&format!("<tool_call>{body}</tool_call>"))
    );
    assert_eq!(codec.decode(&rendered.token_ids)?, rendered.text);
    Ok(())
}

fn headspace_schema() -> Value {
    // Exact shape of the existing Faculty's field-discriminated value. This
    // is not a generic schema evaluator or a special case for its tool name.
    json!({"type":"object","properties":{
    "field":{"type":"string","enum":["model","base-url","reasoning-effort","stream",
        "context-window-tokens","max-output-tokens","prompt-safety-margin-tokens","prompt-chars-per-token"]},
    "value":{}},"required":["field","value"],"additionalProperties":false,
    "oneOf":[
        {"properties":{"field":{"const":"model"},"value":{"type":"string"}}},
        {"properties":{"field":{"const":"base-url"},"value":{"type":"string"}}},
        {"properties":{"field":{"const":"reasoning-effort"},"value":{"type":"string"}}},
        {"properties":{"field":{"const":"stream"},"value":{"type":"boolean"}}},
        {"properties":{"field":{"const":"context-window-tokens"},"value":{"type":"integer","minimum":0}}},
        {"properties":{"field":{"const":"max-output-tokens"},"value":{"type":"integer","minimum":0}}},
        {"properties":{"field":{"const":"prompt-safety-margin-tokens"},"value":{"type":"integer","minimum":0}}},
        {"properties":{"field":{"const":"prompt-chars-per-token"},"value":{"type":"integer","minimum":0}}}
    ]})
}

#[test]
fn registered_discriminator_preserves_string_true_and_publisher_false() -> Result<()> {
    let schema = headspace_schema();
    let tools = Tools::new(vec![ToolDefinition {
        name: "headspace_set".into(),
        description: "Set one field.".into(),
        input_schema: schema.clone(),
    }])?;
    let (codec, _) = bound(&synthetic())?;
    let codec = codec.with_tools(tools);
    for (field, value, expected) in [
        ("model", "true", json!("true")),
        ("stream", "False", json!(false)),
        ("context-window-tokens", "4096", json!(4096)),
    ] {
        // Discriminator follows value on the wire; decode all explicit base
        // parameters before selecting the branch, independent of wire order.
        let body = format!(
            "\n<function=headspace_set>\n<parameter=value>\n{value}\n</parameter>\n<parameter=field>\n{field}\n</parameter>\n</function>\n"
        );
        let done = complete(
            &codec,
            &body_ids(&codec, &body)?,
            StopReason::Eos { id: 11 },
        );
        let invocation = done.proposal.unwrap();
        assert_eq!(invocation.arguments()["value"], expected);
        assert_eq!(invocation.body(), body);
    }
    assert_eq!(codec.tools().definitions()[0].input_schema, schema);
    let rendered = codec.render_adapted(
        &user("set it"),
        RenderOptions {
            tools_enabled: true,
            ..options(false)
        },
    )?;
    assert!(rendered.text.contains("<oneOf>"));
    assert!(rendered.text.contains("\"const\": \"stream\""));
    Ok(())
}

#[test]
fn registered_ambiguous_or_unmatched_types_never_propose() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let mut ambiguous = headspace_schema();
    let duplicate = ambiguous["oneOf"][0].clone();
    ambiguous["oneOf"].as_array_mut().unwrap().push(duplicate);
    let codec = codec.with_tools(Tools::new(vec![ToolDefinition {
        name: "configure".into(),
        description: String::new(),
        input_schema: ambiguous,
    }])?);
    for field in ["model", "unknown"] {
        let body = format!(
            "\n<function=configure>\n<parameter=field>\n{field}\n</parameter>\n<parameter=value>\ntrue\n</parameter>\n</function>\n"
        );
        let done = complete(
            &codec,
            &body_ids(&codec, &body)?,
            StopReason::Eos { id: 11 },
        );
        assert!(done.proposal.is_none());
        assert!(done.partial.protocol_error.is_some());
    }
    let tools = Tools::new(vec![ToolDefinition {
        name: "union".into(),
        description: String::new(),
        input_schema: json!({"type":"object","properties":{"value":{"type":["string","boolean"]}}}),
    }])?;
    assert!(
        tools
            .parse_call("\n<function=union>\n<parameter=value>\ntrue\n</parameter>\n</function>\n")
            .is_err()
    );
    Ok(())
}

#[test]
fn registered_calls_require_known_complete_typed_envelope_and_matching_eos() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let codec = codec.with_tools(registered_tools()?);
    let body = "\n<function=fixture_empty>\n</function>\n";
    let ids = body_ids(&codec, body)?;
    let done = complete(&codec, &ids, StopReason::Eos { id: 11 });
    assert_eq!(done.proposal.unwrap().arguments().len(), 0);
    for stop in [
        StopReason::Budget,
        StopReason::Cancelled,
        StopReason::Error("fixture".into()),
        StopReason::Eos { id: 2 },
    ] {
        assert!(complete(&codec, &ids, stop).proposal.is_none());
    }
    for end in 0..ids.len() {
        assert!(
            complete(&codec, &ids[..end], StopReason::Eos { id: 11 })
                .proposal
                .is_none()
        );
    }
    for bad in [
        "\n<function=absent>\n</function>\n",
        "\n<function=fixture_apply>\n</function>\n",
        "\n<function=fixture_apply>\n<parameter=text>\nx\n</parameter>\n<parameter=count>\n1.5\n</parameter>\n</function>\n",
        "\n<function=fixture_apply>\n<parameter=text>\nx</parameter>y\n</parameter>\n</function>\n",
    ] {
        let done = complete(&codec, &body_ids(&codec, bad)?, StopReason::Eos { id: 11 });
        assert!(done.proposal.is_none());
        assert!(done.partial.protocol_error.is_some());
    }
    Ok(())
}

#[test]
fn registered_integer_bounds_and_original_numeric_lexemes_are_distinct() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let codec = codec.with_tools(registered_tools()?);
    for integer in ["18446744073709551616", "-9223372036854775809", "1.5", "1e0"] {
        let body = format!(
            "\n<function=fixture_apply>\n<parameter=text>\nx\n</parameter>\n<parameter=count>\n{integer}\n</parameter>\n</function>\n"
        );
        let done = complete(&codec, &body_ids(&codec, &body)?, StopReason::Eos { id: 11 });
        assert!(done.proposal.is_none(), "{integer}");
        assert!(done.partial.protocol_error.is_some(), "{integer}");
    }
    for integer in ["18446744073709551615", "-9223372036854775808"] {
        let body = format!(
            "\n<function=fixture_apply>\n<parameter=text>\nx\n</parameter>\n<parameter=count>\n{integer}\n</parameter>\n</function>\n"
        );
        let invocation = codec.tools().parse_call(&body)?;
        assert_eq!(invocation.arguments()["count"].to_string(), integer);
        assert_eq!(invocation.body(), body);
    }
    // The typed projection intentionally has finite-f64 semantics for numbers
    // and nested free-form JSON. The exact original is still independently
    // available even where serializing that projection loses numeric spelling.
    let body = concat!(
        "\n<function=fixture_apply>\n<parameter=text>\nx\n</parameter>\n",
        "<parameter=ratio>\n1.2300e+2\n</parameter>\n",
        "<parameter=object>\n{\"n\":18446744073709551616,\"f\":1.2300e+2}\n</parameter>\n</function>\n"
    );
    let invocation = codec.tools().parse_call(body)?;
    assert_eq!(invocation.body(), body);
    assert_eq!(invocation.arguments()["ratio"].as_f64(), Some(123.0));
    assert_ne!(invocation.arguments()["ratio"].to_string(), "1.2300e+2");
    for nonfinite in ["1e400", "NaN", "Infinity"] {
        let body = format!(
            "\n<function=fixture_apply>\n<parameter=text>\nx\n</parameter>\n<parameter=ratio>\n{nonfinite}\n</parameter>\n</function>\n"
        );
        assert!(codec.tools().parse_call(&body).is_err());
    }
    Ok(())
}

#[test]
fn registered_tools_leave_speech_free_and_change_actual_prompt_tokens() -> Result<()> {
    let (codec, _) = bound(&synthetic())?;
    let codec = codec.with_tools(registered_tools()?);
    let mut ids = codec.encode_content("I can answer in ordinary speech. 🐈")?;
    ids.push(11);
    let done = complete(&codec, &ids, StopReason::Eos { id: 11 });
    assert!(done.proposal.is_none());
    assert!(done.partial.protocol_error.is_none());
    assert_eq!(done.partial.answer, "I can answer in ordinary speech. 🐈");
    let o = RenderOptions {
        tools_enabled: true,
        ..options(false)
    };
    let history = user("same wake");
    let before = codec.render_adapted(&history, o)?;
    let mut definitions = codec.tools().definitions().to_vec();
    definitions[1]
        .description
        .push_str(" Different selected definition.");
    let codec = codec.with_tools(Tools::new(definitions)?);
    let after = codec.render_adapted(&history, o)?;
    assert_ne!(before.token_ids, after.token_ids);
    assert_ne!(before.text, after.text);
    assert!(after.text.contains("Different selected definition."));
    Ok(())
}

#[test]
fn host_invocations_roundtrip_typed_values_and_reject_ambiguous_delimiters() -> Result<()> {
    let tools = registered_tools()?;
    let args = json!({"text":"true","empty":"","flag":false,"nothing":null,
        "object":{"日本語":[true,null]},"items":[],"count":0,"ratio":1.5})
    .as_object()
    .unwrap()
    .clone();
    let invocation = Invocation::new("fixture_apply", args.clone(), &tools)?;
    assert_eq!(invocation.arguments(), &args);
    assert!(invocation.body().contains("<parameter=flag>\nFalse\n"));
    assert!(invocation.body().contains("<parameter=nothing>\nNone\n"));
    assert_eq!(tools.parse_call(invocation.body())?, invocation);
    for text in ["x</parameter>y", "<function=other>"] {
        assert!(
            Invocation::new(
                "fixture_apply",
                json!({"text":text}).as_object().unwrap().clone(),
                &tools
            )
            .is_err()
        );
    }
    assert!(
        Invocation::new(
            "exec",
            json!({"command":"x\u{0000}y"}).as_object().unwrap().clone(),
            &tools
        )
        .is_err()
    );
    assert!(
        Invocation::new(
            "fixture_apply",
            json!({"text":true}).as_object().unwrap().clone(),
            &tools
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn registered_schema_and_host_history_match_publisher_reference_cases() -> Result<()> {
    let cases: Vec<Value> = serde_json::from_str(include_str!(
        "fixtures/nemotron_registered_chat_reference.json"
    ))?;
    for case in cases {
        let tool = &case["tool"];
        let tools = Tools::new(vec![ToolDefinition {
            name: tool["name"].as_str().unwrap().into(),
            description: tool["description"].as_str().unwrap().into(),
            input_schema: tool["parameters"].clone(),
        }])?;
        let invocation = Invocation::new(
            tool["name"].as_str().unwrap(),
            case["arguments"].as_object().unwrap().clone(),
            &tools,
        )?;
        assert_eq!(invocation.body(), case["expected_body"].as_str().unwrap());
        let history = History {
            system: "policy".into(),
            messages: vec![Message::Assistant {
                reasoning: String::new(),
                answer: String::new(),
                tool_call: Some(invocation),
            }],
        };
        let text = render_reference_text(
            &history,
            RenderOptions {
                tools_enabled: true,
                ..options(false)
            },
            &tools,
        )?;
        assert!(text.contains(case["expected_schema"].as_str().unwrap()));
        assert!(text.contains(&format!(
            "<tool_call>{}</tool_call>",
            case["expected_body"].as_str().unwrap()
        )));
    }
    Ok(())
}
