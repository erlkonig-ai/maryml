//! Where does one whole Nemotron-H prefill spend its time?
//!
//! Opens a native model pile the way the Drive resident does (read-only
//! snapshot, typed config/tokenizer selection from the root, ordered layers,
//! `Session::from_snapshot` with the given limits), renders one user message
//! to exactly `--tokens` prompt IDs, and appends them through the ordinary
//! `Session::prefill_ids` once per `--runs` entry, resetting between runs.
//! `bracket` runs time the whole append only; `device` runs add CUPTI device
//! attribution (models::nemotron_h::prefill_timing). The first run is cold
//! (first-use compilation) and reported as such. The expert bank is resident
//! from Session load, so no run fills a cache.
//!
//! Prints one JSON record per run as it ends, then one table and one summary
//! JSON line; `--out PATH` (a NEW file) gets the same, appended and synced
//! run by run, so completed runs survive a later failure.
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
        session::{self, Limits, Session, Source, Status},
    },
};
use serde_json::{Value, json};
use std::{io::Write, path::PathBuf, sync::Arc, time::Instant};
use triblespace::prelude::*;

const USAGE: &str = "usage: nemotron_prefill_timing --model-pile PATH --collection HANDLE --tokens N
  --resident-bytes N --chunk-bytes N [--root HEX=B42F499283B6FB9BBD11245A619048D3] [--context N=1048576]
  [--expect-config HEX|none=B789124520F986E33C7A4EFD40F65CDD]
  [--expect-tokenizer-binding HEX|none=17CD966B6B65ECDE8608AF04297B4704]
  [--text-file PATH] [--runs bracket,bracket,device,bracket] [--out NEW_PATH]";

// Ordinary prose, repeated until the prompt reaches --tokens. Override with
// --text-file for a prompt closer to what the resident sees.
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

struct Args {
    pile: PathBuf,
    root: Id,
    collection: String,
    tokens: usize,
    context: usize,
    resident_bytes: usize,
    chunk_bytes: usize,
    expect_config: Option<Id>,
    expect_binding: Option<Id>,
    text_file: Option<PathBuf>,
    runs: Vec<Mode>,
    out: Option<PathBuf>,
}

fn hex(value: &str) -> Result<Id> {
    Id::from_hex(value).with_context(|| format!("invalid entity id {value:?}"))
}
fn expect(value: &str) -> Result<Option<Id>> {
    if value == "none" { Ok(None) } else { hex(value).map(Some) }
}

fn parse(mut argv: impl Iterator<Item = String>) -> Result<Args> {
    let (mut pile, mut collection, mut tokens) = (None, None, None);
    // No defaults: the whole-bank Session's minimum depends on the context,
    // and a stale default would only be refused by admission.
    let (mut resident_bytes, mut chunk_bytes) = (None, None);
    let mut args = Args {
        pile: PathBuf::new(), root: Id::from_hex("B42F499283B6FB9BBD11245A619048D3").unwrap(),
        collection: String::new(), tokens: 0, context: 1_048_576,
        resident_bytes: 0, chunk_bytes: 0,
        expect_config: expect("B789124520F986E33C7A4EFD40F65CDD")?,
        expect_binding: expect("17CD966B6B65ECDE8608AF04297B4704")?,
        text_file: None, runs: vec![Mode::Bracket, Mode::Bracket, Mode::Device, Mode::Bracket], out: None,
    };
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().with_context(|| format!("{flag} needs a value\n{USAGE}"));
        match flag.as_str() {
            "--model-pile" => pile = Some(PathBuf::from(value()?)),
            "--root" => args.root = hex(&value()?)?,
            "--collection" => collection = Some(value()?),
            "--tokens" => tokens = Some(value()?.parse::<usize>()?),
            "--context" => args.context = value()?.parse()?,
            "--resident-bytes" => resident_bytes = Some(value()?.parse()?),
            "--chunk-bytes" => chunk_bytes = Some(value()?.parse()?),
            "--expect-config" => args.expect_config = expect(&value()?)?,
            "--expect-tokenizer-binding" => args.expect_binding = expect(&value()?)?,
            "--text-file" => args.text_file = Some(PathBuf::from(value()?)),
            "--runs" => args.runs = value()?.split(',').map(|r| Mode::parse(r)
                .with_context(|| format!("run {r:?} is neither bracket nor device"))).collect::<Result<_>>()?,
            "--out" => args.out = Some(PathBuf::from(value()?)),
            "-h" | "--help" => bail!("{USAGE}"),
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    args.pile = pile.with_context(|| format!("--model-pile is required\n{USAGE}"))?;
    args.collection = collection.with_context(|| format!("--collection is required\n{USAGE}"))?;
    args.tokens = tokens.with_context(|| format!("--tokens is required\n{USAGE}"))?;
    args.resident_bytes = resident_bytes.with_context(|| format!("--resident-bytes is required\n{USAGE}"))?;
    args.chunk_bytes = chunk_bytes.with_context(|| format!("--chunk-bytes is required\n{USAGE}"))?;
    ensure!(args.pile.is_absolute(), "--model-pile must be an absolute path");
    ensure!(args.tokens > 0 && args.tokens <= args.context, "--tokens must be 1..=--context");
    ensure!(!args.runs.is_empty(), "--runs is empty");
    Ok(args)
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
    // The same typed selection as Drive's native startup (nemotron/startup.rs).
    let (config, binding) = find!((config: Id, binding: Id), pattern!(snapshot.facts(), [
        {root @ attrs::member: _?container, tokenizer::attrs::tokenizer: ?binding},
        {_?container @ attrs::kind: "nemotron-h-config", attrs::member: ?config}
    ])).min().context("selected root has no typed Nemotron config/tokenizer")?;
    if let Some(want) = args.expect_config {
        ensure!(config == want, "root selects config {config:X}, not the expected {want:X}");
    }
    if let Some(want) = args.expect_binding {
        ensure!(binding == want, "root selects tokenizer binding {binding:X}, not the expected {want:X}");
    }
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

    // Exactly --tokens IDs of an ordinary rendered user turn (truncated).
    let text = match &args.text_file {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
        None => TEXT.to_owned(),
    };
    ensure!(!text.is_empty(), "prompt text is empty");
    let codec = ChatCodec::rebuild(snapshot.facts(), snapshot.store(), source.tokenizer_binding,
        source.tokenizer_base, source.pre_tokenizer)?;
    let mut body = text.clone();
    let ids = loop {
        let history = History { system: String::new(), rendered_system: None,
            messages: vec![Message::User(body.clone())] };
        let rendered = codec.render_adapted(&history, RenderOptions { enable_thinking: false,
            medium_effort: false, add_generation_prompt: true, tools_enabled: false })?;
        if rendered.token_ids.len() >= args.tokens {
            break rendered.token_ids[..args.tokens].to_vec();
        }
        let repeat = (args.tokens / rendered.token_ids.len().max(1)).clamp(1, 64);
        for _ in 0..repeat { body.push_str(&text); }
    };
    drop(codec);
    let ids_digest = blake3::hash(&ids.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>());

    let limits = Limits { capacity: args.context, max_resident_bytes: args.resident_bytes,
        max_chunk_bytes: args.chunk_bytes };
    let admission = session::admission(&text_config, limits, ids.len()).map_err(anyhow::Error::msg)
        .context("native logical allocation admission")?;
    let started = Instant::now();
    let mut session = Session::from_snapshot(snapshot, source, &layers, Default::default(), limits)
        .map_err(anyhow::Error::msg)?;
    let session_seconds = started.elapsed().as_secs_f64();

    let runs = run_all(&args.runs, |index, mode| {
        if !session.committed_ids().is_empty() || session.status() != Status::Ready {
            if let Err(error) = session.reset() {
                return (json!({"run": index, "mode": mode.as_str(), "error": format!("reset between runs: {error}")}),
                    Err(anyhow::anyhow!("reset between runs: {error}")));
            }
        }
        prefill_timing::set_mode(Some(mode));
        let outer = Instant::now();
        let result = session.prefill_ids(&ids, &mut || false);
        let outer_ns = outer.elapsed().as_nanos() as u64;
        prefill_timing::set_mode(None);
        let mut report = prefill_timing::take_report().unwrap_or_else(|| json!({"error": "no report"}));
        report["run"] = json!(index);
        report["cold"] = json!(index == 0);
        report["caller_wall_ns"] = json!(outer_ns);
        if let Err(error) = &result { report["error"] = json!(error); }
        eprintln!("run {index} {} done: {:.3} s", mode.as_str(), outer_ns as f64 / 1e9);
        (report, result.map(|_| ()).map_err(anyhow::Error::msg))
    }, out.as_mut())?;

    // Overhead: each device run against the mean of the warm bracket runs.
    let wall = |r: &Value| r["bracket_wall_ns"].as_u64().unwrap_or(0) as f64;
    let warm: Vec<f64> = runs.iter().filter(|r| r["cold"] == false && r["mode"] == "bracket").map(wall).collect();
    let warm_mean = if warm.is_empty() { None } else { Some(warm.iter().sum::<f64>() / warm.len() as f64) };
    let overhead: Vec<Value> = runs.iter().filter(|r| r["mode"] == "device").map(|r| json!({
        "run": r["run"], "cold": r["cold"], "device_wall_ns": wall(r),
        "warm_bracket_mean_ns": warm_mean,
        "overhead_fraction": warm_mean.map(|m| wall(r) / m - 1.0),
    })).collect();
    let spread = (warm.len() > 1).then(|| {
        let (lo, hi) = warm.iter().fold((f64::MAX, 0f64), |(lo, hi), w| (lo.min(*w), hi.max(*w)));
        (hi - lo) / warm_mean.unwrap()
    });

    let mut table = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(table, "nemotron_prefill_timing: {} prompt tokens, context {}, resident {} B, chunk {} B, layer_major {}",
        ids.len(), args.context, args.resident_bytes, args.chunk_bytes, admission.prefill.layer_major);
    let _ = writeln!(table, "{:<5} {:<8} {:<5} {:>14} {:>12} {:>10}", "run", "mode", "cold", "bracket ms", "tokens/s", "tiles");
    for r in &runs {
        let _ = writeln!(table, "{:<5} {:<8} {:<5} {:>14.3} {:>12.1} {:>10}", r["run"].to_string(), r["mode"].as_str().unwrap_or("?"),
            r["cold"].to_string(), wall(r) / 1e6, r["tokens_per_second"].as_f64().unwrap_or(0.0), r["fenced_tiles"].to_string());
    }
    for o in &overhead {
        let _ = writeln!(table, "device-mode overhead (run {}): {} vs warm bracket mean; warm bracket spread {}",
            o["run"], o["overhead_fraction"].as_f64().map_or("n/a (no warm bracket run)".into(), |f| format!("{:+.2}%", 100.0 * f)),
            spread.map_or("n/a".into(), |s| format!("{:.2}%", 100.0 * s)));
    }
    for r in &runs {
        let _ = writeln!(table, "\nhost phases of run {} ({}):", r["run"], r["mode"].as_str().unwrap_or("?"));
        table.push_str(&prefill_timing::host_summary(r));
    }
    if let Some(device) = runs.iter().rev().find(|r| r["mode"] == "device") {
        let _ = writeln!(table, "\nattribution of run {}:", device["run"]);
        table.push_str(&prefill_timing::table(device));
    }
    let line = json!({
        "tool": "nemotron_prefill_timing",
        "source": {"pile": args.pile, "collection": args.collection, "root": format!("{root:X}"),
            "config": format!("{config:X}"), "tokenizer_binding": format!("{binding:X}"),
            "tokenizer_base": format!("{:X}", source.tokenizer_base), "pre_tokenizer": format!("{:X}", source.pre_tokenizer),
            "layers": layers.len()},
        "settings": {"context": args.context, "resident_bytes": args.resident_bytes, "chunk_bytes": args.chunk_bytes,
            "runs": args.runs.iter().map(|m| m.as_str()).collect::<Vec<_>>()},
        "prompt": {"tokens": ids.len(), "text": args.text_file.as_ref().map_or("built-in prose".into(), |p| p.display().to_string()),
            "ids_blake3": ids_digest.to_hex().to_string()},
        "admission": {"layer_major": admission.prefill.layer_major, "resident_bytes": admission.resident_bytes,
            "base_resident_bytes": admission.base_resident_bytes, "expert_bank_bytes": admission.expert_cache_bytes,
            "device_working_bytes": admission.prefill.device_working_bytes},
        "snapshot_seconds": snapshot_seconds, "session_seconds": session_seconds,
        "runs": runs, "overhead": overhead, "warm_bracket_spread_fraction": spread,
    });
    let line = serde_json::to_string(&line)?;
    print!("{table}");
    println!("{line}");
    if let Some(file) = out.as_mut() {
        write!(file, "{table}{line}\n")?;
        file.sync_all()?;
    }
    Ok(())
}

/// Run every mode in order. Each run's record is written (stdout, and
/// appended and synced to --out) as soon as that run ends, BEFORE its error
/// is propagated, so completed runs survive a later failure or cap.
fn run_all(modes: &[Mode], mut run: impl FnMut(usize, Mode) -> (Value, Result<()>),
    mut file: Option<&mut std::fs::File>) -> Result<Vec<Value>>
{
    let mut runs = Vec::new();
    for (index, mode) in modes.iter().enumerate() {
        let (report, result) = run(index, *mode);
        let line = serde_json::to_string(&json!({"tool": "nemotron_prefill_timing", "record": "run", "report": report}))?;
        println!("{line}");
        std::io::stdout().flush()?;
        if let Some(file) = file.as_deref_mut() {
            writeln!(file, "{line}")?;
            file.sync_data()?;
        }
        runs.push(report);
        result.with_context(|| format!("run {index} prefill"))?;
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_and_chunk_bytes_have_no_stale_default() {
        let argv = |extra: &[&str]| ["--model-pile", "/m.pile", "--collection", "00", "--tokens", "4"].iter()
            .chain(extra).map(|s| s.to_string()).collect::<Vec<_>>().into_iter();
        let error = format!("{:#}", parse(argv(&["--chunk-bytes", "1"])).err().unwrap());
        assert!(error.contains("--resident-bytes is required"), "{error}");
        let error = format!("{:#}", parse(argv(&["--resident-bytes", "1"])).err().unwrap());
        assert!(error.contains("--chunk-bytes is required"), "{error}");
        let args = parse(argv(&["--resident-bytes", "7", "--chunk-bytes", "5"])).unwrap();
        assert_eq!((args.resident_bytes, args.chunk_bytes), (7, 5));
    }

    #[test]
    fn a_failure_after_run_one_leaves_its_record_on_disk() {
        let dir = std::env::temp_dir().join(format!("prefill-timing-cli-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("out.txt");
        let mut file = std::fs::OpenOptions::new().append(true).create_new(true).open(&path).unwrap();
        let mut calls = 0;
        let result = run_all(&[Mode::Bracket, Mode::Device, Mode::Bracket], |index, mode| {
            calls += 1;
            if index == 0 {
                (json!({"run": 0, "mode": mode.as_str(), "bracket_wall_ns": 123}), Ok(()))
            } else {
                (json!({"run": index, "mode": mode.as_str(), "error": "injected"}), Err(anyhow::anyhow!("injected cap")))
            }
        }, Some(&mut file));
        drop(file);
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("run 1 prefill") && error.contains("injected cap"), "{error}");
        assert_eq!(calls, 2, "no run after the failure");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["record"], "run");
        assert_eq!(lines[0]["report"]["bracket_wall_ns"], 123);
        assert_eq!(lines[1]["report"]["error"], "injected");
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}
