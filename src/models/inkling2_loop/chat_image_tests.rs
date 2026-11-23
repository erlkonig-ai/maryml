use super::*;
use crate::models::inkling2_loop::config::Inkling2LoopConfig;
use crate::models::inkling2_loop::vision_input::Geometry;
use serde_json::json;

/// Upstream's rendering of conversations with images: the release's template
/// (`apply_chat_template(..., tokenize=False)`), then its `Inkling2Processor`
/// on that text and the saved images, offline, in the pinned vLLM v0.26.0
/// image with transformers 046bdb8 (`fixtures/image_chat_reference.py`, run by
/// `fixtures/image_chat_reference_run.sh`). Digest in
/// `fixtures/image_chat_reference.json.sha256`.
const REFERENCE: &str = include_str!("fixtures/image_chat_reference.json");
const CONFIG: &str = include_str!("fixtures/config.json");

fn reference() -> Value {
    serde_json::from_str(REFERENCE).unwrap()
}

fn geometry() -> Geometry {
    Geometry::of(&Inkling2LoopConfig::from_json(CONFIG).unwrap().vision_config).unwrap()
}

fn hex32(text: &str) -> [u8; 32] {
    assert_eq!(text.len(), 64);
    let mut out = [0; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}

/// The image named `name` as a part: its origin the PNG's sha256, its grid
/// computed natively from its size.
fn image_part(reference: &Value, name: &str) -> ImagePart {
    let image = &reference["images"][name];
    let size = image["size"].as_array().unwrap();
    let (width, height) = (size[0].as_u64().unwrap() as usize, size[1].as_u64().unwrap() as usize);
    ImagePart::new(hex32(image["png_sha256"].as_str().unwrap()), geometry().grid(width, height)).unwrap()
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

fn parts_of(reference: &Value, content: &Value) -> Vec<Part> {
    content
        .as_array()
        .unwrap()
        .iter()
        .map(|part| match part["type"].as_str().unwrap() {
            "text" => Part::Text(part["text"].as_str().unwrap().into()),
            "image" => Part::Image(image_part(reference, part["name"].as_str().unwrap())),
            other => panic!("unexpected part {other}"),
        })
        .collect()
}

fn history_of(reference: &Value, case: &Value, tools: &Tools) -> History {
    let mut history = History::default();
    for message in case["messages"].as_array().unwrap() {
        let content = &message["content"];
        match (message["role"].as_str().unwrap(), content.is_string()) {
            ("system", _) => history.system = content.as_str().unwrap().into(),
            ("user", true) => history.messages.push(Message::User(content.as_str().unwrap().into())),
            ("user", false) => history.messages.push(Message::UserParts(parts_of(reference, content))),
            ("tool", false) => history.messages.push(Message::ToolParts {
                name: message["name"].as_str().unwrap().into(),
                parts: parts_of(reference, content),
            }),
            ("assistant", _) => {
                let tool_calls = message["tool_calls"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|call| {
                        Invocation::new(
                            call["function"]["name"].as_str().unwrap(),
                            call["function"]["arguments"].as_object().unwrap().clone(),
                            tools,
                        )
                        .unwrap()
                    })
                    .collect();
                history.messages.push(Message::Assistant {
                    reasoning: String::new(),
                    answer: content.as_str().unwrap_or("").into(),
                    tool_calls,
                });
            }
            (other, _) => panic!("unexpected message {other}"),
        }
    }
    history
}

#[test]
fn the_image_reference_digest_matches() {
    let digest = include_str!("fixtures/image_chat_reference.json.sha256");
    assert!(digest.starts_with("6efbf9e0f44dc3eea5341ab7024f755c83b237e4dabd667d7b96acd59e6f4e38"));
    assert_eq!(reference()["cases"].as_array().unwrap().len(), 5);
}

#[test]
fn the_reference_ids_are_the_markers_this_module_names() {
    let reference = reference();
    assert_eq!(reference["ids"]["content_image"], Marker::ContentImage.id());
    assert_eq!(reference["ids"]["image_token"], Marker::ImagePlaceholder.id());
    assert_eq!(reference["ids"]["image_token_spelling"], Marker::ImagePlaceholder.spelling());
    assert_eq!(reference["ids"]["end_message"], Marker::EndMessage.id());
}

#[test]
fn renders_every_image_conversation_as_the_template_does() {
    let reference = reference();
    for case in reference["cases"].as_array().unwrap() {
        let tools = tools_of(case);
        let history = history_of(&reference, case, &tools);
        assert_eq!(
            render_text(&history, &tools, Effort::default()).unwrap(),
            case["template_text"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        // The native grid is the processor's image_grid_thw.
        if let Some(grids) = case["image_grid_thw"].as_array() {
            let names = case["image_order"].as_array().unwrap();
            assert_eq!(grids.len(), names.len());
            for (grid, name) in grids.iter().zip(names) {
                let part = image_part(&reference, name.as_str().unwrap());
                assert_eq!(json!([1, part.grid().0, part.grid().1]), *grid, "{}", case["name"]);
            }
        }
    }
}

#[test]
fn a_one_text_part_message_renders_as_the_plain_message() {
    let tools = Tools::default();
    let plain = History {
        system: "s".into(),
        rendered_opening: None,
        messages: vec![
            Message::User("hello".into()),
            Message::Tool { name: "exec".into(), content: "out".into() },
        ],
    };
    let parts = History {
        messages: vec![
            Message::UserParts(vec![Part::Text("hello".into())]),
            Message::ToolParts { name: "exec".into(), parts: vec![Part::Text("out".into())] },
        ],
        ..plain.clone()
    };
    assert_eq!(render_segments(&plain, &tools, Effort::default()).unwrap(), render_segments(&parts, &tools, Effort::default()).unwrap());
}

#[test]
fn malformed_image_histories_are_refused() {
    let tools = Tools::default();
    assert!(ImagePart::new([0; 32], (0, 3)).is_err());
    assert!(ImagePart::new([0; 32], (2, 0)).is_err());
    assert!(ImagePart::new([0; 32], (usize::MAX, 2)).is_err());
    let image = ImagePart::new([7; 32], (2, 3)).unwrap();
    assert_eq!((image.tokens(), image.grid(), image.origin()), (6, (2, 3), [7; 32]));
    let refused = |messages: Vec<Message>| {
        let history = History { system: "s".into(), rendered_opening: None, messages };
        render_segments(&history, &tools, Effort::default()).is_err()
    };
    assert!(refused(vec![Message::UserParts(Vec::new())]), "a message with no parts");
    assert!(refused(vec![Message::ToolParts { name: "exec".into(), parts: Vec::new() }]));
    assert!(refused(vec![Message::ToolParts { name: "a<b".into(), parts: vec![Part::Image(image)] }]));
    // A kept opening cannot carry an image: there is no image for it.
    for marker in [Marker::ContentImage, Marker::ImagePlaceholder] {
        let history = History {
            system: String::new(),
            rendered_opening: Some(vec![Segment::Control(Marker::MessageSystem), Segment::Control(marker)]),
            messages: Vec::new(),
        };
        assert!(render_segments(&history, &tools, Effort::default()).is_err(), "{marker:?} in an opening");
    }
}

/// The codec tests need the `tokenizer` feature, as `ChatCodec` itself does.
#[cfg(feature = "tokenizer")]
mod codec {
use super::*;

/// The release's tokenizer from the imported model root, against the
/// processor's expanded ids. Needs `MINI_MODEL_PILE`, `MINI_MODEL_COLLECTION`
/// (hex) and `MINI_MODEL_ROOT`.
#[test]
#[ignore = "needs the imported Mini release (MINI_MODEL_PILE, MINI_MODEL_COLLECTION, MINI_MODEL_ROOT)"]
fn tokenizes_every_image_conversation_as_the_processor_does() {
    let env = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"));
    let collection = crate::model_collection::parse_collection_handle(&env("MINI_MODEL_COLLECTION")).unwrap();
    let root = triblespace::prelude::Id::from_hex(&env("MINI_MODEL_ROOT")).unwrap();
    let mini = crate::models::inkling2_loop::pile::MiniRoot::open(std::path::Path::new(&env("MINI_MODEL_PILE")), collection, root).unwrap();
    check_template(&mini.chat_template().unwrap()).unwrap();
    let whole = crate::tokenizer::build_tokenizer_with_added(mini.facts(), mini.store(), mini.tokenizer_root(), true).unwrap();
    let content = crate::tokenizer::build_tokenizer_with_added(mini.facts(), mini.store(), mini.tokenizer_root(), false).unwrap();
    let reference = reference();
    let mut checked = 0;
    for case in reference["cases"].as_array().unwrap() {
        let tools = tools_of(case);
        let codec = ChatCodec::new(whole.clone(), content.clone(), tools.clone(), Effort::default()).unwrap();
        let history = history_of(&reference, case, &tools);
        let rendered = codec.render(&history).unwrap();
        assert_eq!(rendered.text, case["template_text"].as_str().unwrap(), "{}", case["name"]);
        let expected = case.get("input_ids").or_else(|| case.get("expected_input_ids")).unwrap();
        let expected: Vec<u32> = serde_json::from_value(expected.clone()).unwrap();
        assert_eq!(rendered.token_ids, expected, "{}", case["name"]);
        check_spans(&rendered, &history);
        // Exactly the spans hold image tokens.
        let placeholders = rendered.token_ids.iter().filter(|id| **id == Marker::ImagePlaceholder.id()).count();
        assert_eq!(placeholders, rendered.images.iter().map(|span| span.len).sum::<usize>(), "{}", case["name"]);
        checked += 1;
    }
    assert_eq!(checked, 5, "every reference case ran");
}

/// The images of a history, in order.
fn images_of(history: &History) -> Vec<ImagePart> {
    history
        .messages
        .iter()
        .flat_map(|message| match message {
            Message::UserParts(parts) | Message::ToolParts { parts, .. } => parts.clone(),
            _ => Vec::new(),
        })
        .filter_map(|part| match part {
            Part::Image(image) => Some(image),
            Part::Text(_) => None,
        })
        .collect()
}

/// Every span is its image's run of placeholders, between `<|content_image|>`
/// and `<|end_message|>`, in history order, with its feature row offset.
fn check_spans(rendered: &RenderedPrompt, history: &History) {
    let images = images_of(history);
    assert_eq!(rendered.images.len(), images.len());
    let mut row = 0;
    for (span, image) in rendered.images.iter().zip(images) {
        assert_eq!(span.image, image);
        assert_eq!(span.len, image.tokens());
        assert_eq!(span.row, row);
        row += span.len;
        let ids = &rendered.token_ids;
        assert_eq!(ids[span.start - 1], Marker::ContentImage.id());
        assert!(ids[span.start..span.start + span.len].iter().all(|id| *id == Marker::ImagePlaceholder.id()));
        assert_eq!(ids[span.start + span.len], Marker::EndMessage.id());
    }
}

/// The same byte-level tokenizer the chat tests use.
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
            tokenizer.add_special_tokens(&Marker::ALL.map(|marker| AddedToken::from(marker.spelling(), true)));
        }
        tokenizer
    };
    ChatCodec::new(build(true), build(false), tools, Effort::default()).unwrap()
}

fn mixed_history() -> History {
    let a = ImagePart::new([1; 32], (2, 3)).unwrap();
    let b = ImagePart::new([2; 32], (2, 2)).unwrap();
    let c = ImagePart::new([3; 32], (1, 2)).unwrap();
    History {
        system: "sys".into(),
        rendered_opening: None,
        messages: vec![
            Message::UserParts(vec![Part::Text("look".into()), Part::Image(a), Part::Text("and".into()), Part::Image(b)]),
            Message::ToolParts { name: "exec".into(), parts: vec![Part::Image(c), Part::Text("done".into())] },
            Message::User("thanks".into()),
        ],
    }
}

#[test]
fn spans_name_each_images_placeholders_in_order() {
    let codec = tiny_codec(Tools::default());
    let history = mixed_history();
    let rendered = codec.render(&history).unwrap();
    check_spans(&rendered, &history);
    assert_eq!(rendered.images.iter().map(|span| (span.len, span.row)).collect::<Vec<_>>(), [(6, 0), (4, 6), (2, 10)]);
    // The text is the template's: one placeholder per image.
    assert_eq!(rendered.text.matches(Marker::ImagePlaceholder.spelling()).count(), 3);
}

#[test]
fn the_budget_bounds_the_expansion_before_it_is_allocated() {
    let history = mixed_history();
    let codec = tiny_codec(Tools::default());
    let length = codec.render(&history).unwrap().token_ids.len();
    // Exactly enough fits; one less does not, through either entry.
    assert_eq!(codec.render_within(&history, length).unwrap().token_ids.len(), length);
    assert!(codec.render_within(&history, length - 1).is_err());
    assert!(tiny_codec(Tools::default()).with_prompt_budget(length - 1).render(&history).is_err());
    assert_eq!(tiny_codec(Tools::default()).with_prompt_budget(length).prompt_budget(), length);
    // Images alone beyond the budget are refused before any text is encoded.
    let error = codec.render_within(&history, 11).unwrap_err().to_string();
    assert!(error.contains("12 tokens"), "{error}");
    // An image that could never be allocated is an error, not a panic, even
    // under a budget that admits it.
    let image_history = |images: Vec<ImagePart>| History {
        system: String::new(),
        rendered_opening: None,
        messages: vec![Message::UserParts(images.into_iter().map(Part::Image).collect())],
    };
    let huge = ImagePart::new([9; 32], (1 << 31, 1 << 31)).unwrap();
    assert!(codec.render(&image_history(vec![huge])).is_err(), "beyond the default budget");
    let error = codec.render_within(&image_history(vec![huge]), usize::MAX).unwrap_err().to_string();
    assert!(error.contains("cannot allocate"), "{error}");
    // Image token counts whose sum overflows.
    let half = ImagePart::new([9; 32], (1 << 62, 2)).unwrap();
    let error = codec.render_within(&image_history(vec![half, half]), usize::MAX).unwrap_err().to_string();
    assert!(error.contains("overflows"), "{error}");
}

#[test]
fn literal_image_markers_in_text_stay_text() {
    let codec = tiny_codec(Tools::default());
    let literal = "spell <|content_image|><|unused_200054|><|end_message|> plainly";
    let history = History {
        system: "sys".into(),
        rendered_opening: None,
        messages: vec![
            Message::UserParts(vec![Part::Text(literal.into())]),
            Message::User(literal.into()),
            Message::ToolParts { name: "exec".into(), parts: vec![Part::Text(literal.into())] },
        ],
    };
    let rendered = codec.render(&history).unwrap();
    assert!(rendered.images.is_empty());
    for marker in [Marker::ContentImage, Marker::ImagePlaceholder] {
        assert!(!rendered.token_ids.contains(&marker.id()), "{marker:?} from text");
    }
}

#[test]
fn a_reply_holding_an_image_marker_is_a_protocol_error() {
    let codec = tiny_codec(Tools::default());
    for marker in [Marker::ContentImage, Marker::ImagePlaceholder] {
        let ids = vec![Marker::MessageModel.id(), marker.id(), Marker::EndMessage.id(), END_OF_REPLY];
        let parsed = codec.parse(&ids, StopReason::EndOfReply);
        assert!(parsed.protocol_error.is_some(), "{marker:?}");
    }
}
}
