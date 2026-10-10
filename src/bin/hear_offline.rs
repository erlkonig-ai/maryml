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
//! The weights are whatever `Ears::load` selects (`VOXTRAL_WEIGHTS`). One
//! `RESULT` line per clip, tab separated: clip, audio s, unpaced compute s,
//! paced compute s, finish s, transcript (paced; the unpaced one is checked
//! equal and reported if not).
//!
//!   cargo run --release --features voxtral-cuda --bin hear_offline -- \
//!     <voxtral.pile> <clip.wav>...

use std::path::Path;
use std::time::{Duration, Instant};

use mary::hear::Ears;
use mary::models::f5::wav;

const RATE: usize = 16_000;
/// One Voxtral token of audio.
const CHUNK: usize = 1280;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() >= 2,
        "usage: hear_offline <voxtral.pile> <clip.wav>..."
    );
    let started = Instant::now();
    let ears = Ears::load(Path::new(&args[0]))?;
    eprintln!(
        "[hear_offline] loaded in {:.1} s, VOXTRAL_WEIGHTS={}",
        started.elapsed().as_secs_f64(),
        std::env::var("VOXTRAL_WEIGHTS").unwrap_or_else(|_| "(unset)".into())
    );
    report_memory();
    for clip in &args[1..] {
        let (samples, rate) = wav::read_pcm16_mono(Path::new(clip));
        anyhow::ensure!(rate as usize == RATE, "{clip}: expected 16 kHz, got {rate}");
        let audio = samples.len() as f64 / RATE as f64;
        let (unpaced, _, unpaced_text) = stream(&ears, &samples, false);
        let (paced, finish, text) = stream(&ears, &samples, true);
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
