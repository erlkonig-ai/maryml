//! Offline hearing through the production seam, [`mary::hear::Ears`]: load the
//! ears once, then stream each 16 kHz mono PCM16 WAV through a fresh
//! [`mary::hear::Listening`] at the default 480 ms delay, twice:
//!
//! - unpaced, every 80 ms chunk pushed as soon as the last returned: compute
//!   seconds per second of audio, and the warm pass for the next;
//! - paced at real time, each chunk pushed when its last sample would have
//!   arrived: `finish` is the time from the clip's last sample to the
//!   finished transcript, the wait a conversation sees after someone stops
//!   speaking. A stream that falls behind real time carries its backlog into
//!   that number, as it would live.
//!
//! The model is the `nvfp4` Voxtral root of the model collection
//! `--collection blake3:HEX` names in `<pile>`, or with `--root HEX` one
//! explicit root of a legacy native pile (found in its `mary-model-graph`),
//! whose 16-bit projections `--weights f16|nvfp4` decides. `--lane w4a4|w4a16`
//! is how stored NVFP4 projections multiply prompt-height activations;
//! `--unpaced` skips the real-time pass (its columns read 0). One
//! `RESULT` line per clip, tab separated: clip, audio s, unpaced compute s,
//! paced compute s, finish s, transcript (paced; the unpaced one is checked
//! equal and reported if not).
//!
//!   cargo run --release --features voxtral-cuda --bin hear_offline -- \
//!     <pile> (--collection HEX | --root HEX) [--weights W] [--lane L] [--unpaced] <clip.wav>...

use std::path::Path;
use std::time::{Duration, Instant};

use mary::hear::Ears;
use mary::models::f5::wav;

const RATE: usize = 16_000;
/// One Voxtral token of audio.
const CHUNK: usize = 1280;

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| -> Option<String> {
        let at = args.iter().position(|a| a == name)?;
        let value = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        value
    };
    let collection = flag("--collection");
    let root = flag("--root");
    let weights = flag("--weights").unwrap_or_else(|| "f16".into());
    let lane = flag("--lane").unwrap_or_else(|| "w4a16".into());
    let unpaced_only = match args.iter().position(|a| a == "--unpaced") {
        Some(at) => {
            args.remove(at);
            true
        }
        None => false,
    };
    anyhow::ensure!(
        args.len() >= 2 && collection.is_some() != root.is_some(),
        "usage: hear_offline <pile> (--collection HEX | --root HEX) [--weights f16|nvfp4] [--lane w4a4|w4a16] [--unpaced] <clip.wav>..."
    );
    let weights = mary::models::voxtral::fast::Weights::parse(&weights)?;
    let lane = mary::nvfp4::PromptLane::parse(&lane)?;
    let started = Instant::now();
    let store = mary::persist::read_model_pile_snapshot_read_only(Path::new(&args[0]))?;
    let ears = match (&collection, &root) {
        (Some(collection), _) => {
            let handle = mary::model_collection::parse_collection_handle(collection)?;
            let collection = mary::model_collection::open_model_collection(&store, handle)?;
            let models = mary::model_collection::snapshot_model_collection_for(&store, collection)?;
            Ears::load(&models, lane)?
        }
        (None, Some(root)) => {
            let root = triblespace::prelude::Id::from_hex(root)
                .ok_or_else(|| anyhow::anyhow!("--root is not a 32-hex id"))?;
            let models = mary::model_collection::snapshot_model_collection_in(&store)?;
            Ears::load_root(&models, root, weights, lane)?
        }
        (None, None) => unreachable!("checked above"),
    };
    eprintln!(
        "[hear_offline] loaded in {:.1} s ({}; weights {weights:?}, lane {lane:?})",
        started.elapsed().as_secs_f64(),
        collection.as_deref().map_or("legacy root", |_| "nvfp4 root"),
    );
    report_memory();
    for clip in &args[1..] {
        let (samples, rate) = wav::read_pcm16_mono(Path::new(clip));
        anyhow::ensure!(rate as usize == RATE, "{clip}: expected 16 kHz, got {rate}");
        let audio = samples.len() as f64 / RATE as f64;
        let (unpaced, _, unpaced_text) = stream(&ears, &samples, false);
        // `--unpaced`: transcripts at scale, no real-time pass and no finish.
        let (paced, finish, text) = if unpaced_only {
            (Duration::ZERO, Duration::ZERO, unpaced_text.clone())
        } else {
            stream(&ears, &samples, true)
        };
        if unpaced_text != text {
            eprintln!("[hear_offline] {clip}: unpaced transcript differs: {unpaced_text:?}");
        }
        println!(
            "RESULT\t{clip}\t{audio:.3}\t{:.3}\t{:.3}\t{:.3}\t{}",
            unpaced.as_secs_f64(),
            paced.as_secs_f64(),
            finish.as_secs_f64(),
            text.trim()
        );
    }
    report_memory();
    Ok(())
}

/// Stream `samples` in [`CHUNK`]s; returns (time inside push and finish, time
/// from the last sample's arrival to the finished transcript, transcript).
/// Unpaced, every sample "arrives" at the start, so finish is the whole run.
fn stream(ears: &Ears, samples: &[f32], paced: bool) -> (Duration, Duration, String) {
    let mut listening = ears.listen(480);
    let mut text = String::new();
    let mut compute = Duration::ZERO;
    let start = Instant::now();
    let arrival = |n: usize| {
        if paced {
            start + Duration::from_secs_f64(n as f64 / RATE as f64)
        } else {
            start
        }
    };
    let mut fed = 0;
    for chunk in samples.chunks(CHUNK) {
        fed += chunk.len();
        let due = arrival(fed);
        let now = Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        }
        let t = Instant::now();
        text.push_str(&listening.push(chunk));
        compute += t.elapsed();
    }
    let t = Instant::now();
    text.push_str(&listening.finish());
    compute += t.elapsed();
    let finish = Instant::now().saturating_duration_since(arrival(samples.len()));
    (compute, finish, text)
}

/// This process's resident and peak resident set, and its device memory as
/// the driver reports it (unified memory on GB10: the two overlap).
fn report_memory() {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))
            .map_or("?".to_owned(), |l| l[name.len()..].trim().to_owned())
    };
    let pid = std::process::id().to_string();
    let device = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout).lines().find_map(|l| {
                l.split_once(',')
                    .filter(|(p, _)| p.trim() == pid)
                    .map(|(_, m)| format!("{} MiB", m.trim()))
            })
        })
        .unwrap_or_else(|| "?".to_owned());
    eprintln!(
        "[hear_offline] memory: VmRSS {}, VmHWM {}, device {device}",
        field("VmRSS:"),
        field("VmHWM:")
    );
}
