//! First bounded native text smoke: explicit scratch-import coordinates only.
//! No default pile/key/collection, publication, tool execution or alternate codec.
//! Session construction uploads before Generator prompt preflight. Logical byte
//! limits are NOT physical memory limits; a finite external unit remains required.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use cubecl::cuda::CudaDevice;
use mary::{
    model_collection,
    models::nemotron_h::{
        chat::{History, Message, StopReason},
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
const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const LIMITS: Limits = Limits {
    capacity: CAPACITY,
    max_resident_bytes: 8 * 1024 * 1024 * 1024,
    max_chunk_bytes: 6 * 1024 * 1024 * 1024,
};

#[derive(Parser)]
struct Args {
    /// Exact successful nemotron_pile_import report.json; not a model identity proof.
    import_report: PathBuf,
    /// NEW directory; kept on any error, never overwritten or automatically removed.
    new_output_dir: PathBuf,
    /// Plain user content, not trusted protocol framing. At most4096 UTF-8 bytes.
    #[arg(long)]
    prompt: String,
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
    ensure!(
        !args.prompt.is_empty() && args.prompt.len() <= MAX_PROMPT_BYTES,
        "plain prompt must have1..4096 UTF-8 bytes"
    );
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

    // This native API validates linked config/tokenizer choices and every layer
    // position, and performs checked aggregate admission BEFORE its uploads.
    // It is not a prompt preflight: construction can allocate/upload first.
    record["phase"] = json!("construct-session-and-codec");
    let started = Instant::now();
    let session = Session::from_snapshot(snapshot, source, &layers, CudaDevice::default(), LIMITS)
        .map_err(anyhow::Error::msg)?;
    record["output"]["committed_tokens"] = json!(session.committed_ids().len());
    record["output"]["committed_ids"] = json!(session.committed_ids());
    record["output"]["session_status"] = json!(format!("{:?}", session.status()));
    ensure!(
        session.capacity() == CAPACITY && session.vocab_size() == VOCAB,
        "native Session capacity/vocabulary differs from bounded Super smoke"
    );
    let mut generator = Generator::new(session)?;
    record["initialization_seconds"] = json!(started.elapsed().as_secs_f64());
    let history = History {
        system: String::new(),
        messages: vec![Message::User(args.prompt.clone())],
    };
    let options = TurnOptions {
        enable_thinking: false,
        exec_tool: false,
        max_new_tokens: MAX_NEW,
    };
    record["phase"] = json!("prompt-preflight");
    let started = Instant::now();
    let admission = generator.preflight(&history, options)?;
    record["prompt"] = json!({"plain_text":args.prompt,
        "rendered_text":admission.prompt().text, "ids":admission.prompt().token_ids,
        "reserved_attended_tokens":admission.reserved_attended_tokens()});
    record["preflight_seconds"] = json!(started.elapsed().as_secs_f64());

    record["phase"] = json!("generate");
    let started = Instant::now();
    let output = generator.turn(&history, options, || false);
    record["generation_seconds"] = json!(started.elapsed().as_secs_f64());
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
        "proposal_returned":output.completion.proposal.is_some()});
    ensure!(
        output.completion.proposal.is_none(),
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
        "output":{"emitted_ids":[],"partial":{"raw_text":"","reasoning":"","answer":"","protocol_error":null},
            "decoded_complete":true,"stop":null,"limit":null,"proposal_returned":false,
            "committed_tokens":null,"committed_ids":null,"session_status":null},
        "settings":{"batch":1,"capacity":CAPACITY,"max_new_tokens":MAX_NEW,
            "thinking":false,"medium_effort":false,"exec_tool":false,"sampling":"existing greedy",
            "max_resident_bytes":LIMITS.max_resident_bytes,"max_chunk_bytes":LIMITS.max_chunk_bytes,
            "logical_limits_are_not_physical_memory_limits":true,
            "session_upload_precedes_prompt_preflight":true}});
    let result = execute(&args, &mut record);
    record["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    record["status"] = json!(if result.is_ok() { "complete" } else { "failed" });
    if let Err(error) = &result {
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
