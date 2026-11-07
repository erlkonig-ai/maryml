//! First bounded native text smoke: explicit scratch-import coordinates only.
//! No default pile/key/collection, publication, tool execution or alternate codec.
//! CPU prompt preflight precedes Session uploads. Logical byte limits are NOT
//! physical memory limits; a finite external unit remains required. Optional
//! unique-answer fixtures are a narrow capability slice, not a quality suite.
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use cubecl::cuda::CudaDevice;
use mary::{
    model_collection,
    models::nemotron_h::{
        attention,
        chat::{ChatCodec, History, Message, RenderOptions, StopReason},
        generation::{Generator, LimitCause, TurnOptions},
        session::{Limits, Session, Source},
    },
    persist,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use triblespace::{
    core::inline::encodings::hash::{Blake3, Hash},
    prelude::Id,
};

const CAPACITY: usize = 256;
const MAX_NEW: usize = 8;
const LAYERS: usize = 88;
const VOCAB: usize = 131072;
const MAX_REPORT_BYTES: usize = 1024 * 1024;
const MAX_PROMPT_BYTES: usize = 4096;
const CAPACITY_CEILING: usize = 32768;
const RESPONSE_CEILING: usize = 256;
const PROMPT_BYTES_CEILING: usize = 128 * 1024;
const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const LIMITS: Limits = Limits {
    capacity: CAPACITY,
    max_resident_bytes: 96 * 1024 * 1024 * 1024, // fixed weights + every expert (63.4 GB) + KV + state
    max_chunk_bytes: 6 * 1024 * 1024 * 1024,
};

#[derive(Parser, Debug)]
#[command(group(clap::ArgGroup::new("input").required(true).args(["prompt", "prompt_file", "case"])))]
struct Args {
    /// Exact successful nemotron_pile_import report.json; not a model identity proof.
    import_report: PathBuf,
    /// NEW directory; kept on any error, never overwritten or automatically removed.
    new_output_dir: PathBuf,
    /// Plain user content, not trusted protocol framing. Ungraded by default.
    #[arg(long)]
    prompt: Option<String>,
    /// UTF-8 plain user content, bounded by --max-prompt-bytes before decoding.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Fixed unique-answer fixture; supplies its own expected answer, not budgets.
    #[arg(long, value_enum, conflicts_with = "expect_answer")]
    case: Option<Case>,
    /// Explicit unique-answer contract: compare decoded answer after trim only.
    /// Omit for generic smoke; this is not a general semantic-answer grader.
    #[arg(long)]
    expect_answer: Option<String>,
    /// Total attended-token capacity, including conservative response reserve.
    #[arg(long, default_value_t = CAPACITY)]
    capacity: usize,
    /// Emission budget INCLUDING EOS. Does not guarantee a complete answer.
    #[arg(long, default_value_t = MAX_NEW)]
    max_new_tokens: usize,
    /// Plain input byte ceiling (not a token count); at most 131072.
    #[arg(long, default_value_t = MAX_PROMPT_BYTES)]
    max_prompt_bytes: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Case {
    ShortAddition,
    EvidenceRetrieval,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::ShortAddition => "short-addition-v1",
            Self::EvidenceRetrieval => "evidence-retrieval-v1",
        }
    }

    fn fixture(self) -> (String, &'static str) {
        match self {
            Self::ShortAddition => (
                "What is 17 + 25? Reply with only the decimal integer, no explanation.".into(),
                "42",
            ),
            Self::EvidenceRetrieval => {
                let mut prompt = String::from(
                    "Read the registry below. Each record has a unique key. Return only the value for key TARGET-731, with no explanation.\nREGISTRY\n",
                );
                for index in 0..256 {
                    if index == 128 {
                        prompt.push_str("key=TARGET-731 value=coral-582\n");
                    }
                    prompt.push_str(&format!(
                        "key=DECOY-{index:03} value=amber-{index:03} note=unrelated registry entry\n"
                    ));
                }
                prompt.push_str("END REGISTRY\nWhat is the value for key TARGET-731? Reply with only its value.");
                (prompt, "coral-582")
            }
        }
    }
}

fn validate_bounds(args: &Args) -> Result<()> {
    ensure!((1..=CAPACITY_CEILING).contains(&args.capacity), "capacity must be 1..=32768 (not 100k+ coverage)");
    ensure!((1..=RESPONSE_CEILING).contains(&args.max_new_tokens), "max-new-tokens must be 1..=256");
    ensure!(args.max_new_tokens <= args.capacity, "response reserve exceeds capacity");
    ensure!((1..=PROMPT_BYTES_CEILING).contains(&args.max_prompt_bytes), "max-prompt-bytes must be 1..=131072");
    if let Some(expected) = &args.expect_answer {
        ensure!(!expected.trim().is_empty() && expected.len() <= MAX_PROMPT_BYTES,
            "expected answer must be nonblank and at most 4096 UTF-8 bytes");
    }
    Ok(())
}

fn read_prompt(args: &Args) -> Result<(String, Option<String>)> {
    validate_bounds(args)?;
    let (prompt, expected) = if let Some(case) = args.case {
        let (prompt, expected) = case.fixture();
        (prompt, Some(expected.to_owned()))
    } else if let Some(path) = &args.prompt_file {
        let mut bytes = Vec::new();
        fs::File::open(path)?.take((args.max_prompt_bytes + 1) as u64).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= args.max_prompt_bytes, "plain prompt exceeds byte bound");
        (String::from_utf8(bytes).context("prompt file is not UTF-8")?, args.expect_answer.clone())
    } else {
        (args.prompt.clone().context("missing prompt")?, args.expect_answer.clone())
    };
    ensure!(!prompt.is_empty() && prompt.len() <= args.max_prompt_bytes,
        "plain prompt must be nonempty and within max-prompt-bytes");
    Ok((prompt, expected))
}

fn prompt_reserve(ids: &[u32], capacity: usize, max_new: usize) -> Result<usize> {
    ensure!(!ids.is_empty() && ids.iter().all(|&id| (id as usize) < VOCAB),
        "empty prompt or prompt ID outside Super vocabulary");
    let required = ids.len().checked_add(max_new).context("prompt reserve overflow")?;
    ensure!(required <= capacity, "prompt plus response reserve ({required}) exceeds capacity ({capacity})");
    Ok(required)
}

// Outcome checks do not compare tokens/logits. An explicitly requested exact
// unique answer is the task contract; generic smoke remains ungraded.
fn task_outcome(expected: Option<&str>, answer: &str, stop: &StopReason,
    decoded: bool, protocol_error: bool, proposal: bool) -> Value
{
    let Some(expected) = expected else {
        return json!({"status":"not_evaluated", "reason":"no-answer-contract"});
    };
    let (status, reason) = if matches!(stop, StopReason::Error(_) | StopReason::Cancelled)
        || !decoded || protocol_error || proposal {
        ("not_evaluated", "operational-failure")
    } else if !matches!(stop, StopReason::Eos { .. }) {
        ("failed", "incomplete-within-budget")
    } else if answer.trim() == expected.trim() {
        ("passed", "expected-answer")
    } else {
        ("failed", "answer-mismatch")
    };
    json!({"status":status, "reason":reason, "expected_answer":expected,
        "comparison":"trimmed-exact-text", "task_scope":"explicit-unique-answer-only"})
}

// Invocation coordinates only, not a materialized model/weight catalogue. Unknown
// report annotations are tolerated. Native snapshot/Session own graph validation.
#[derive(Deserialize)]
struct ImportReport {
    status: String,
    pile: PathBuf,
    collection: String,
    model_root: String,
    source_name: Option<String>,
    selection: Selection,
}
#[derive(Deserialize)]
struct Selection {
    config: String,
    tokenizer_binding: String,
    tokenizer_base: String,
    pre_tokenizer: String,
    layers: Vec<Layer>,
}
#[derive(Deserialize)]
struct Layer {
    index: usize,
    layer: String,
}

fn read_report(path: &Path) -> Result<ImportReport> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take((MAX_REPORT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= MAX_REPORT_BYTES, "import report exceeds1MiB");
    let report: ImportReport = serde_json::from_slice(&bytes)?;
    ensure!(report.status == "complete", "import report is not complete");
    ensure!(
        report.pile.is_absolute(),
        "report pile path must be explicit and absolute"
    );
    ensure!(
        report.selection.layers.len() == LAYERS,
        "smoke requires88 explicitly selected layers"
    );
    ensure!(
        report
            .selection
            .layers
            .iter()
            .enumerate()
            .all(|(i, l)| l.index == i),
        "report layers must be in explicit zero-based position order"
    );
    Ok(report)
}
fn id(value: &str) -> Result<Id> {
    Id::from_hex(value).with_context(|| format!("invalid explicit entity ID {value:?}"))
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() <= MAX_OUTPUT_BYTES,
        "single output exceeds8MiB cap"
    );
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn stop_json(stop: &StopReason) -> Value {
    match stop {
        StopReason::Eos { id } => json!({"kind":"eos", "id":id}),
        StopReason::Budget => json!({"kind":"budget"}),
        StopReason::Cancelled => json!({"kind":"cancelled"}),
        StopReason::Error(error) => json!({"kind":"error", "error":error}),
    }
}

fn execute(args: &Args, record: &mut Value) -> Result<()> {
    record["phase"] = json!("read-prompt");
    let (prompt, expected) = read_prompt(args)?;
    record["plain_prompt"] = json!(prompt);
    record["task"]["expected_answer"] = json!(expected);
    record["phase"] = json!("read-import-report");
    let input = read_report(&args.import_report)?;
    let source = Source {
        collection: model_collection::parse_collection_handle(&input.collection)?,
        root: id(&input.model_root)?,
        config: id(&input.selection.config)?,
        tokenizer_binding: id(&input.selection.tokenizer_binding)?,
        tokenizer_base: id(&input.selection.tokenizer_base)?,
        pre_tokenizer: id(&input.selection.pre_tokenizer)?,
    };
    let layers = input
        .selection
        .layers
        .iter()
        .map(|l| id(&l.layer))
        .collect::<Result<Vec<_>>>()?;
    record["source"] = json!({"pile":input.pile, "collection":input.collection,
        "model_root":input.model_root, "config":input.selection.config,
        "tokenizer_binding":input.selection.tokenizer_binding,
        "tokenizer_base":input.selection.tokenizer_base,
        "pre_tokenizer":input.selection.pre_tokenizer,
        "ordered_layer_ids":input.selection.layers.iter().map(|l| &l.layer).collect::<Vec<_>>(),
        "report_asserted_source_name":input.source_name,
        "report_proves_checkpoint_provenance":false});

    record["phase"] = json!("read-only-snapshot");
    let started = Instant::now();
    let store = persist::read_model_pile_snapshot_read_only(&input.pile)?;
    let collection = model_collection::open_model_collection(&store, source.collection)?;
    let snapshot = Arc::new(model_collection::snapshot_model_collection_for(
        &store, collection,
    )?);
    let support: Vec<String> = snapshot
        .support()
        .members()
        .map(|h| Hash::<Blake3>::to_hex(&h.transmute()))
        .collect();
    record["source"]["actual_support"] = json!(support);
    record["snapshot_seconds"] = json!(started.elapsed().as_secs_f64());

    let history = History {
        system: String::new(),
        rendered_system: None,
        messages: vec![Message::User(prompt.clone())],
    };
    let options = TurnOptions {
        enable_thinking: false,
        tools_enabled: false,
        max_new_tokens: args.max_new_tokens,
    };
    // The same held observation and production renderer, without a CUDA Session.
    // Drop this temporary codec before Session/Generator constructs its own.
    record["phase"] = json!("cpu-prompt-preflight");
    let started = Instant::now();
    let rendered = {
        let codec = ChatCodec::rebuild(snapshot.facts(), snapshot.store(),
            source.tokenizer_binding, source.tokenizer_base, source.pre_tokenizer)?;
        codec.render_adapted(&history, RenderOptions {
            enable_thinking: false, medium_effort: false,
            add_generation_prompt: true, tools_enabled: false,
        })?
    };
    record["prompt"] = json!({"plain_text":prompt, "rendered_text":rendered.text,
        "ids":rendered.token_ids, "reserved_attended_tokens":null});
    let reserved = prompt_reserve(&rendered.token_ids, args.capacity, args.max_new_tokens)?;
    record["prompt"]["reserved_attended_tokens"] = json!(reserved);
    record["cpu_preflight_seconds"] = json!(started.elapsed().as_secs_f64());

    // Session additionally validates linked config/tokenizer choices and every
    // layer, and performs checked aggregate admission before its uploads.
    record["phase"] = json!("construct-session-and-codec");
    let started = Instant::now();
    let limits = Limits { capacity: args.capacity, ..LIMITS };
    let session = Session::from_snapshot(snapshot, source, &layers, CudaDevice::default(), limits)
        .map_err(anyhow::Error::msg)?;
    record["output"]["committed_tokens"] = json!(session.committed_ids().len());
    record["output"]["committed_ids"] = json!(session.committed_ids());
    record["output"]["session_status"] = json!(format!("{:?}", session.status()));
    ensure!(
        session.capacity() == args.capacity && session.vocab_size() == VOCAB,
        "native Session capacity/vocabulary differs from bounded Super smoke"
    );
    let mut generator = Generator::new(session)?;
    record["initialization_seconds"] = json!(started.elapsed().as_secs_f64());
    record["phase"] = json!("prompt-preflight");
    let started = Instant::now();
    let admission = generator.preflight(&history, options)?;
    ensure!(admission.prompt() == &rendered && admission.reserved_attended_tokens() == reserved,
        "production preflight differs from CPU prompt preflight");
    record["preflight_seconds"] = json!(started.elapsed().as_secs_f64());
    drop(admission);
    drop(rendered);

    record["phase"] = json!("generate");
    let started = Instant::now();
    record["evaluation"]["generation_started"] = json!(true);
    let output = generator.turn(&history, options, || false);
    record["generation_seconds"] = json!(started.elapsed().as_secs_f64());
    record["task"] = task_outcome(expected.as_deref(), &output.completion.partial.answer,
        &output.completion.stop, output.decoded_complete,
        output.completion.partial.protocol_error.is_some(), !output.completion.proposals.is_empty());
    let limit = match output.limit {
        None => None,
        Some(LimitCause::TokenBudget) => Some("token-budget"),
        Some(LimitCause::Context) => Some("context"),
    };
    record["output"] = json!({"emitted_ids":output.emitted_ids,
        "partial":{"raw_text":output.completion.partial.raw_text,
            "reasoning":output.completion.partial.reasoning,
            "answer":output.completion.partial.answer,
            "protocol_error":output.completion.partial.protocol_error},
        "decoded_complete":output.decoded_complete, "stop":stop_json(&output.completion.stop),
        "limit":limit, "committed_tokens":output.committed_tokens,
        "committed_ids":generator.session().committed_ids(),
        "session_status":format!("{:?}", output.session_status),
        "proposal_returned":!output.completion.proposals.is_empty()});
    ensure!(
        output.completion.proposals.is_empty(),
        "unexpected proposal with tools disabled"
    );
    if let StopReason::Error(error) = &output.completion.stop {
        anyhow::bail!("generation: {error}");
    }
    record["phase"] = json!("finished");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir(&args.new_output_dir)
        .context("output directory must be NEW; never overwritten")?;
    eprintln!(
        "smoke output retained on success/failure: {}",
        args.new_output_dir.display()
    );
    let started = Instant::now();
    let mut record = json!({"status":"started", "scope":"bounded native text smoke, not quality/parity acceptance",
        "import_report":args.import_report, "plain_prompt":args.prompt,
        "input":{"kind":if args.case.is_some() {"fixture"} else if args.prompt_file.is_some() {"file"} else {"inline"},
            "prompt_file":args.prompt_file,"fixture":args.case.map(Case::name)},
        // Count all contracted invocations, including startup/operational
        // failures. generation_started is a separate subgroup, never success.
        "evaluation":{"contract_requested":args.case.is_some() || args.expect_answer.is_some(),
            "generation_started":false},
        "execution":{"attention":attention::EXECUTION},
        "task":{"status":"not_evaluated","reason":"not-generated","expected_answer":args.expect_answer},
        "output":{"emitted_ids":[],"partial":{"raw_text":"","reasoning":"","answer":"","protocol_error":null},
            "decoded_complete":true,"stop":null,"limit":null,"proposal_returned":false,
            "committed_tokens":null,"committed_ids":null,"session_status":null},
        "settings":{"batch":1,"capacity":args.capacity,"max_new_tokens":args.max_new_tokens,
            "max_prompt_bytes":args.max_prompt_bytes,"capacity_ceiling":CAPACITY_CEILING,
            "budget_provenance":"CLI values or unchanged smoke defaults; fixtures do not override budgets",
            "coverage":"initial at-most-32768-token slice; not 100k+ or 524k geometry",
            "thinking":false,"medium_effort":false,"exec_tool":false,"sampling":"existing greedy",
            "max_resident_bytes":LIMITS.max_resident_bytes,"max_chunk_bytes":LIMITS.max_chunk_bytes,
            "logical_limits_are_not_physical_memory_limits":true,
            "session_upload_precedes_prompt_preflight":false}});
    let result = execute(&args, &mut record);
    record["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    record["status"] = json!(if result.is_ok() { "complete" } else { "failed" });
    if let Err(error) = &result {
        record["task"]["status"] = json!("not_evaluated");
        record["task"]["reason"] = json!("operational-failure");
        record["error"] = json!(format!("{error:#}"));
        if record["output"]["stop"].is_null() {
            record["output"]["stop"] = json!({"kind":"error","error":format!("{error:#}")});
        }
    }
    // Empty raw output on pre-generation failure, actual retained partial after
    // turn. Null state fields mean construction never returned an observation.
    let raw = write_new(
        &args.new_output_dir.join("generation.txt"),
        record["output"]["partial"]["raw_text"]
            .as_str()
            .unwrap_or("")
            .as_bytes(),
    );
    if let Err(error) = &raw {
        record["status"] = json!("failed");
        record["raw_output_error"] = json!(format!("{error:#}"));
    }
    let saved = serde_json::to_vec_pretty(&record)
        .map_err(anyhow::Error::from)
        .and_then(|bytes| write_new(&args.new_output_dir.join("result.json"), &bytes));
    let result = match (result, raw) {
        (Ok(()), value) => value,
        (Err(error), Ok(())) => Err(error),
        (Err(original), Err(save)) => Err(anyhow::anyhow!(
            "{original:#}; additionally save raw output: {save:#}"
        )),
    };
    match (result, saved) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error.context("save smoke result")),
        (Err(original), Err(save)) => Err(anyhow::anyhow!(
            "{original:#}; additionally save result: {save:#}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test-owned, per-instance directory; cleanup names only its one input file
    // and the now-empty directory. No recursive removal or model/pile access.
    struct PromptFile {
        dir: PathBuf,
        path: PathBuf,
    }

    impl PromptFile {
        fn new(bytes: &[u8]) -> Self {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let dir = std::env::temp_dir().join(format!("mary-smoke-prompt-{}-{nonce}-{}",
                std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::create_dir(&dir).unwrap();
            let fixture = Self { path: dir.join("input.txt"), dir };
            let mut file = fs::OpenOptions::new().write(true).create_new(true)
                .open(&fixture.path).unwrap();
            file.write_all(bytes).unwrap();
            fixture
        }

        fn args(&self, bound: usize) -> Args {
            let mut args = args(&["--prompt-file", "replaced-before-read"]);
            args.prompt_file = Some(self.path.clone());
            args.max_prompt_bytes = bound;
            args
        }
    }

    impl Drop for PromptFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_dir(&self.dir);
        }
    }

    fn args(extra: &[&str]) -> Args {
        Args::try_parse_from(["smoke", "report.json", "new-output"].into_iter()
            .chain(extra.iter().copied())).unwrap()
    }

    #[test]
    fn old_inline_invocation_keeps_defaults_and_is_ungraded() {
        let args = args(&["--prompt", "hello"]);
        assert_eq!((args.capacity, args.max_new_tokens, args.max_prompt_bytes), (256, 8, 4096));
        assert_eq!(read_prompt(&args).unwrap(), ("hello".into(), None));
        let result = task_outcome(None, "anything", &StopReason::Eos { id: 2 }, true, false, false);
        assert_eq!(result["status"], "not_evaluated");
        assert_eq!(result["reason"], "no-answer-contract");
    }

    #[test]
    fn attention_has_no_runtime_choice() {
        assert!(Args::try_parse_from(["smoke", "r", "o", "--prompt", "x",
            "--attention-profile", "scalar-f32-v1"]).is_err());
    }

    #[test]
    fn exactly_one_input_and_no_fixture_override() {
        for extra in [
            vec![],
            vec!["--prompt", "x", "--prompt-file", "p"],
            vec!["--prompt", "x", "--case", "short-addition"],
            vec!["--prompt-file", "p", "--case", "short-addition"],
            vec!["--case", "short-addition", "--expect-answer", "wrong"],
        ] {
            assert!(Args::try_parse_from(["smoke", "r", "o"].into_iter().chain(extra)).is_err());
        }
        assert!(args(&["--prompt-file", "p"]).prompt_file.is_some());
    }

    #[test]
    fn explicit_bounds_are_checked_before_any_file_or_model_read() {
        for (flag, value) in [
            ("--capacity", "0"), ("--capacity", "32769"),
            ("--max-new-tokens", "0"), ("--max-new-tokens", "257"),
            ("--max-prompt-bytes", "0"), ("--max-prompt-bytes", "131073"),
        ] {
            assert!(validate_bounds(&args(&["--prompt", "x", flag, value])).is_err());
        }
        assert!(validate_bounds(&args(&["--prompt", "x", "--capacity", "7"])).is_err());
        assert!(validate_bounds(&args(&["--prompt", "x", "--capacity", "32768",
            "--max-new-tokens", "256", "--max-prompt-bytes", "131072"])).is_ok());
    }

    #[test]
    fn byte_limit_counts_utf8_and_exact_boundary() {
        assert!(read_prompt(&args(&["--prompt", "é", "--max-prompt-bytes", "2"])).is_ok());
        assert!(read_prompt(&args(&["--prompt", "é", "--max-prompt-bytes", "1"])).is_err());
        assert!(read_prompt(&args(&["--prompt", ""])).is_err());
        assert!(validate_bounds(&args(&["--prompt", "x", "--expect-answer", " \n"])).is_err());
    }

    #[test]
    fn actual_prompt_file_byte_boundary_and_utf8() {
        let exact = PromptFile::new("é".as_bytes());
        assert_eq!(read_prompt(&exact.args(2)).unwrap(), ("é".into(), None));
        let extra = PromptFile::new("éx".as_bytes());
        assert!(read_prompt(&extra.args(2)).unwrap_err().to_string().contains("exceeds byte bound"));
        let empty = PromptFile::new(b"");
        assert!(read_prompt(&empty.args(2)).unwrap_err().to_string().contains("nonempty"));
        let invalid = PromptFile::new(&[0xff]);
        assert!(read_prompt(&invalid.args(1)).unwrap_err().to_string().contains("not UTF-8"));
        let oversized_invalid = PromptFile::new(&[0xff, 0xff]);
        assert!(read_prompt(&oversized_invalid.args(1)).unwrap_err().to_string().contains("exceeds byte bound"));
    }

    #[test]
    fn missing_prompt_file_is_io_failure_but_invalid_bounds_precede_open() {
        let fixture = PromptFile::new(b"unused");
        let mut args = fixture.args(2);
        args.prompt_file = Some(fixture.dir.join("missing.txt"));
        let error = read_prompt(&args).unwrap_err();
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
        args.capacity = 0;
        let error = read_prompt(&args).unwrap_err();
        assert!(error.to_string().contains("capacity must be"));
        assert!(error.downcast_ref::<std::io::Error>().is_none());
    }

    #[test]
    fn fixtures_are_fixed_unique_answer_tasks_without_hidden_budget_changes() {
        let short = args(&["--case", "short-addition"]);
        assert_eq!(read_prompt(&short).unwrap().1.as_deref(), Some("42"));
        let long = args(&["--case", "evidence-retrieval"]);
        assert!(read_prompt(&long).is_err(), "default 4096-byte smoke bound stays in force");
        assert_eq!((long.capacity, long.max_new_tokens), (256, 8));
        let (prompt, expected) = read_prompt(&args(&["--case", "evidence-retrieval",
            "--capacity", "16384", "--max-new-tokens", "32", "--max-prompt-bytes", "32768"])).unwrap();
        assert_eq!(expected.as_deref(), Some("coral-582"));
        assert_eq!(prompt.matches("key=DECOY-").count(), 256);
        assert_eq!(prompt.matches("key=TARGET-731 value=").count(), 1);
        assert_eq!(prompt.matches("coral-582").count(), 1);
        assert!(prompt.find("key=DECOY-127").unwrap() < prompt.find("key=TARGET-731").unwrap());
        assert!(prompt.find("key=TARGET-731").unwrap() < prompt.find("key=DECOY-128").unwrap());
        assert_eq!(prompt, Case::EvidenceRetrieval.fixture().0);
    }

    #[test]
    fn preflight_reserve_checks_full_ids_and_includes_eos_budget() {
        assert_eq!(prompt_reserve(&[1, 2, 3], 5, 2).unwrap(), 5);
        assert!(prompt_reserve(&[1, 2, 3], 4, 2).is_err());
        assert!(prompt_reserve(&[], 5, 2).is_err());
        assert!(prompt_reserve(&[VOCAB as u32], 5, 2).is_err());
        assert!(prompt_reserve(&[1], usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn exact_answer_contract_trims_only_outer_whitespace() {
        let eos = StopReason::Eos { id: 2 };
        assert_eq!(task_outcome(Some("42"), " 42\n", &eos, true, false, false)["status"], "passed");
        for answer in ["The answer is 42.", "4 2", "042", "43", ""] {
            let result = task_outcome(Some("42"), answer, &eos, true, false, false);
            assert_eq!(result["status"], "failed");
            assert_eq!(result["reason"], "answer-mismatch");
        }
        assert_eq!(task_outcome(Some("coral-582"), "Coral-582", &eos, true, false, false)["status"], "failed");
    }

    #[test]
    fn correct_partial_answer_cannot_pass_without_completed_protocol() {
        let budget = task_outcome(Some("42"), "42", &StopReason::Budget, true, false, false);
        assert_eq!(budget["status"], "failed");
        assert_eq!(budget["reason"], "incomplete-within-budget");
        for stop in [StopReason::Cancelled, StopReason::Error("bad EOS".into())] {
            let result = task_outcome(Some("42"), "42", &stop, true, false, false);
            assert_eq!(result["status"], "not_evaluated");
            assert_eq!(result["reason"], "operational-failure");
        }
        for (decoded, protocol_error, proposal) in [(false, false, false), (true, true, false), (true, false, true)] {
            assert_eq!(task_outcome(Some("42"), "42", &StopReason::Eos { id: 2 },
                decoded, protocol_error, proposal)["status"], "not_evaluated");
        }
    }
}
