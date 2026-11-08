//! How fast does native Nemotron-H decode, one generated token at a time?
//!
//! Opens the model pile exactly as `nemotron_prefill_timing` does (read-only
//! snapshot, typed config/tokenizer selection from the root, ordered layers,
//! `Session::from_snapshot` with the given limits). Renders one user turn of
//! about `--tokens` prompt IDs that asks for a long continuation, prefills it
//! once, then decodes `--decode` tokens greedily (argmax), each through the
//! ordinary one-token `Session::prefill_ids` -> `append_ids` path that the
//! Generator's decode uses. Every step is timed by the caller (argmax +
//! append + logits readback) and bracketed by `prefill_timing`; the steps
//! listed in `--device-steps` add CUPTI device attribution. The decoded text
//! is printed so the run can be judged as text, not only as a rate.
//! EOS (2 or 11) ends the decode early; the stop is reported.
//! `--dry-run` prints the logical admission and exits before any upload.
//! No pile is written; nothing is published.
use anyhow::{Context, Result, bail, ensure};
use mary::{
    format::attrs,
    model_collection, persist, tokenizer,
    models::nemotron_h::{
        chat::{ChatCodec, History, Message, RenderOptions},
        ingress::{config::TextConfig, tokenizer_binding},
        pile,
        prefill_timing::{self, Mode},
        session::{self, Limits, Session, Source},
    },
};
use serde_json::{Value, json};
use std::{io::Write, path::PathBuf, sync::Arc, time::Instant};
use triblespace::prelude::*;

const USAGE: &str = "usage: nemotron_decode_timing --model-pile PATH --collection HANDLE --tokens N --decode N
  --resident-bytes N --chunk-bytes N [--root HEX=B42F499283B6FB9BBD11245A619048D3] [--context N=1048576]
  [--device-steps 40,41,42] [--out NEW_PATH] [--dry-run]";

const TEXT: &str = "The harbour office opened at seven, and by half past the clerks had \
sorted the overnight manifests into three piles: cargo that had arrived, cargo that was late, \
and cargo nobody could account for. The third pile was always the most interesting. A crate of \
surveying instruments had been listed twice under different ship names; a pallet of seed potatoes \
had apparently travelled from Bremen to Bremen by way of Rotterdam. Mara, who had worked the desk \
for eleven years, read each entry slowly, checking weights against the dock scales and dates \
against the tide tables. Most mistakes, she liked to say, were honest ones: a tired hand, a \
smudged stamp, a decimal point that wandered. The rare dishonest ones gave themselves away by \
being too tidy. She made a note, poured another cup of coffee, and called the pilot station to \
ask which vessels had actually crossed the bar before dawn.\n\n";
const ASK: &str = "Above is the opening of a story, repeated. Write the next chapter of Mara's \
story in rich detail: at least eight long paragraphs, with dialogue, and do not summarise.";

struct Args {
    pile: PathBuf,
    root: Id,
    collection: String,
    tokens: usize,
    decode: usize,
    context: usize,
    resident_bytes: usize,
    chunk_bytes: usize,
    device_steps: Vec<usize>,
    out: Option<PathBuf>,
    dry_run: bool,
}

fn parse(mut argv: impl Iterator<Item = String>) -> Result<Args> {
    let (mut pile, mut collection, mut tokens, mut decode, mut resident, mut chunk) = (None, None, None, None, None, None);
    let mut args = Args { pile: PathBuf::new(), root: Id::from_hex("B42F499283B6FB9BBD11245A619048D3").unwrap(),
        collection: String::new(), tokens: 0, decode: 0, context: 1_048_576, resident_bytes: 0, chunk_bytes: 0,
        device_steps: Vec::new(), out: None, dry_run: false };
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().with_context(|| format!("{flag} needs a value\n{USAGE}"));
        match flag.as_str() {
            "--model-pile" => pile = Some(PathBuf::from(value()?)),
            "--root" => args.root = Id::from_hex(&value()?).context("--root")?,
            "--collection" => collection = Some(value()?),
            "--tokens" => tokens = Some(value()?.parse::<usize>()?),
            "--decode" => decode = Some(value()?.parse::<usize>()?),
            "--context" => args.context = value()?.parse()?,
            "--resident-bytes" => resident = Some(value()?.parse()?),
            "--chunk-bytes" => chunk = Some(value()?.parse()?),
            "--device-steps" => args.device_steps = value()?.split(',').filter(|s| !s.is_empty())
                .map(|s| s.parse::<usize>().context("--device-steps")).collect::<Result<_>>()?,
            "--out" => args.out = Some(PathBuf::from(value()?)),
            "--dry-run" => args.dry_run = true,
            "-h" | "--help" => bail!("{USAGE}"),
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    args.pile = pile.with_context(|| format!("--model-pile is required\n{USAGE}"))?;
    args.collection = collection.with_context(|| format!("--collection is required\n{USAGE}"))?;
    args.tokens = tokens.with_context(|| format!("--tokens is required\n{USAGE}"))?;
    args.decode = decode.with_context(|| format!("--decode is required\n{USAGE}"))?;
    args.resident_bytes = resident.with_context(|| format!("--resident-bytes is required\n{USAGE}"))?;
    args.chunk_bytes = chunk.with_context(|| format!("--chunk-bytes is required\n{USAGE}"))?;
    ensure!(args.pile.is_absolute(), "--model-pile must be an absolute path");
    ensure!(args.decode > 0 && args.tokens > 0, "--tokens and --decode must be positive");
    Ok(args)
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<()> {
    let args = parse(std::env::args().skip(1))?;
    let mut out = match &args.out {
        Some(path) => Some(std::fs::OpenOptions::new().append(true).create_new(true).open(path)
            .with_context(|| format!("--out {} must be a NEW file", path.display()))?),
        None => None,
    };
    let started = Instant::now();
    let store = persist::read_model_pile_snapshot_read_only(&args.pile)?;
    let collection = model_collection::parse_collection_handle(&args.collection)?;
    let selected = model_collection::open_model_collection(&store, collection)?;
    let snapshot = Arc::new(model_collection::snapshot_model_collection_for(&store, selected)?);
    let root = args.root;
    let (config, binding) = find!((config: Id, binding: Id), pattern!(snapshot.facts(), [
        {root @ attrs::member: _?container, tokenizer::attrs::tokenizer: ?binding},
        {_?container @ attrs::kind: "nemotron-h-config", attrs::member: ?config}
    ])).min().context("selected root has no typed Nemotron config/tokenizer")?;
    let source = Source { collection, root, config, tokenizer_binding: binding,
        tokenizer_base: tokenizer_binding::bases(snapshot.facts(), binding).min().context("missing tokenizer base")?,
        pre_tokenizer: tokenizer_binding::pre_tokenizers(snapshot.facts(), binding).min().context("missing pre-tokenizer")?,
    };
    let json_config = mary::jsonfacts::load_json(snapshot.facts(), snapshot.store(), config)
        .map_err(|error| anyhow::anyhow!("read native configuration: {error}"))?;
    let text_config: TextConfig = serde_json::from_value(json_config.get("text").cloned().context("missing text config")?)?;
    let layers: Vec<Id> = (0..text_config.layers_block_type.len()).map(|index|
        pile::layers(&snapshot, root, index as u64).min().with_context(|| format!("missing native layer {index}"))
    ).collect::<Result<_>>()?;
    let snapshot_seconds = started.elapsed().as_secs_f64();

    // A whole rendered user turn (never truncated): prose repeated until the
    // turn reaches --tokens, then the request for a long continuation.
    let codec = ChatCodec::rebuild(snapshot.facts(), snapshot.store(), source.tokenizer_binding,
        source.tokenizer_base, source.pre_tokenizer)?;
    let render = |body: &str| codec.render_adapted(&History { system: String::new(), rendered_system: None,
        messages: vec![Message::User(format!("{body}{ASK}"))] }, RenderOptions { enable_thinking: false,
        medium_effort: false, add_generation_prompt: true, tools_enabled: false });
    let mut body = TEXT.to_owned();
    let ids = loop {
        let rendered = render(&body)?;
        if rendered.token_ids.len() >= args.tokens { break rendered.token_ids; }
        let per = rendered.token_ids.len().max(1);
        let repeat = ((args.tokens - per) / 170).clamp(1, 4096);
        for _ in 0..repeat { body.push_str(TEXT); }
    };

    let limits = Limits { capacity: args.context, max_resident_bytes: args.resident_bytes,
        max_chunk_bytes: args.chunk_bytes };
    let admission = session::admission(&text_config, limits, ids.len() + args.decode).map_err(anyhow::Error::msg)
        .context("native logical allocation admission")?;
    let admission_json = json!({"layer_major": admission.prefill.layer_major, "resident_bytes": admission.resident_bytes,
        "base_resident_bytes": admission.base_resident_bytes, "expert_bank_bytes": admission.expert_cache_bytes,
        "device_working_bytes": admission.prefill.device_working_bytes});
    eprintln!("prompt {} tokens; admission {admission_json}", ids.len());
    if args.dry_run {
        println!("{}", json!({"tool": "nemotron_decode_timing", "dry_run": true, "prompt_tokens": ids.len(),
            "context": args.context, "admission": admission_json}));
        return Ok(());
    }
    let started = Instant::now();
    let mut session = Session::from_snapshot(snapshot, source, &layers, Default::default(), limits)
        .map_err(anyhow::Error::msg)?;
    let session_seconds = started.elapsed().as_secs_f64();
    eprintln!("session loaded in {session_seconds:.1} s");

    prefill_timing::set_mode(Some(Mode::Bracket));
    let t = Instant::now();
    session.prefill_ids(&ids, &mut || false).map_err(anyhow::Error::msg).context("prompt prefill")?;
    let prefill_ns = t.elapsed().as_nanos() as u64;
    let prefill_report = prefill_timing::take_report();
    eprintln!("prefill {} tokens in {:.3} s", ids.len(), prefill_ns as f64 / 1e9);

    let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY),
        |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0 as u32;
    let mut emitted = Vec::new();
    let mut steps: Vec<Value> = Vec::new();
    let mut step_ms = Vec::new();
    let mut stop = "budget";
    let loop_started = Instant::now();
    for step in 0..args.decode {
        let s = Instant::now();
        let logits = session.last_logits().context("no logits")?.values;
        let id = argmax(logits);
        emitted.push(id);
        if matches!(id, 2 | 11) { stop = "eos"; break; }
        let device = args.device_steps.contains(&step);
        prefill_timing::set_mode(Some(if device { Mode::Device } else { Mode::Bracket }));
        session.prefill_ids(&[id], &mut || false).map_err(anyhow::Error::msg)
            .with_context(|| format!("decode step {step}"))?;
        let caller_ns = s.elapsed().as_nanos() as u64;
        let report = prefill_timing::take_report();
        step_ms.push(caller_ns as f64 / 1e6);
        let bracket = report.as_ref().and_then(|r| r["bracket_wall_ns"].as_u64());
        let pre = report.as_ref().and_then(|r| r["pre_sync_ns"].as_u64());
        steps.push(json!({"step": step, "id": id, "caller_ns": caller_ns, "bracket_ns": bracket, "pre_sync_ns": pre,
            "device": device.then(|| report.clone())}));
        if device {
            if let Some(r) = &report {
                eprintln!("device step {step}:\n{}{}", prefill_timing::table(r), prefill_timing::host_summary(r));
            }
        }
    }
    let loop_ns = loop_started.elapsed().as_nanos() as u64;
    prefill_timing::set_mode(None);
    let text = codec.decode(&emitted).unwrap_or_else(|e| format!("<decode error {e}>"));

    // Warm statistics exclude the first 8 steps (first-use compilation).
    let warm: Vec<f64> = step_ms.iter().skip(8).copied().collect();
    let mut sorted = warm.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if warm.is_empty() { 0.0 } else { warm.iter().sum::<f64>() / warm.len() as f64 };
    let summary = json!({
        "tool": "nemotron_decode_timing",
        "frame": "batch 1, greedy, one generated token per Session append; caller wall per step = argmax + append + logits readback; warm = steps >= 8",
        "source": {"pile": args.pile, "collection": args.collection, "root": format!("{root:X}"), "layers": layers.len()},
        "settings": {"context": args.context, "resident_bytes": args.resident_bytes, "chunk_bytes": args.chunk_bytes,
            "decode": args.decode, "device_steps": args.device_steps},
        "admission": admission_json,
        "prompt_tokens": ids.len(), "snapshot_seconds": snapshot_seconds, "session_seconds": session_seconds,
        "prefill_ns": prefill_ns, "prefill_report": prefill_report,
        "decode": {"stop": stop, "emitted": emitted.len(), "appended": step_ms.len(), "loop_ns": loop_ns,
            "end_to_end_tokens_per_second": emitted.len() as f64 * 1e9 / loop_ns.max(1) as f64,
            "warm_steps": warm.len(), "warm_mean_ms": mean, "warm_tokens_per_second": if mean > 0.0 { 1e3 / mean } else { 0.0 },
            "warm_p10_ms": pct(&sorted, 0.1), "warm_p50_ms": pct(&sorted, 0.5), "warm_p90_ms": pct(&sorted, 0.9),
            "first_steps_ms": step_ms.iter().take(8).collect::<Vec<_>>()},
        "text": text,
        "steps": steps,
    });
    let line = serde_json::to_string(&summary)?;
    println!("{line}");
    eprintln!("decode: {} tokens, warm mean {mean:.2} ms/token = {:.2} tok/s; p50 {:.2} ms; stop {stop}\n--- text ---\n{text}",
        emitted.len(), if mean > 0.0 { 1e3 / mean } else { 0.0 }, pct(&sorted, 0.5));
    if let Some(file) = out.as_mut() {
        writeln!(file, "{line}")?;
        file.sync_all()?;
    }
    Ok(())
}
