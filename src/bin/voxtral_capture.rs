//! voxtral-gptq lane (scratch, not for main): stream clips through the 16-bit
//! Voxtral (`--weights f16`, one legacy root) and accumulate every projection's
//! input `H = X^T X` on the device, then write them to `--out DIR`. Prints the
//! F16 transcripts as `RESULT` lines like `hear_offline --unpaced`.
//!
//!   voxtral_capture <pile> --root HEX --out DIR <clip.wav>...

use std::path::Path;
use std::time::Instant;

use mary::hear::Ears;
use mary::models::f5::wav;

const RATE: usize = 16_000;
const CHUNK: usize = 1280;

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| -> Option<String> {
        let at = args.iter().position(|a| a == name)?;
        let value = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        value
    };
    let root = flag("--root").ok_or_else(|| anyhow::anyhow!("--root HEX"))?;
    let out = flag("--out").ok_or_else(|| anyhow::anyhow!("--out DIR"))?;
    anyhow::ensure!(args.len() >= 2, "usage: voxtral_capture <pile> --root HEX --out DIR <clip.wav>...");
    let store = mary::persist::read_model_pile_snapshot_read_only(Path::new(&args[0]))?;
    let root = triblespace::prelude::Id::from_hex(&root)
        .ok_or_else(|| anyhow::anyhow!("--root is not a 32-hex id"))?;
    let models = mary::model_collection::snapshot_model_collection_in(&store)?;
    // SAFETY: nobody rewrites the pile while this process lives.
    let ears = unsafe {
        Ears::load_root(
            &models,
            root,
            mary::models::voxtral::fast::Weights::F16,
            mary::nvfp4::PromptLane::W4a16,
        )?
    };
    mary::models::voxtral::fast::capture::begin();
    let started = Instant::now();
    let mut audio_total = 0.0;
    for clip in &args[1..] {
        let (samples, rate) = wav::read_pcm16_mono(Path::new(clip));
        anyhow::ensure!(rate as usize == RATE, "{clip}: expected 16 kHz, got {rate}");
        let audio = samples.len() as f64 / RATE as f64;
        audio_total += audio;
        let mut listening = ears.listen(480);
        let mut text = String::new();
        for chunk in samples.chunks(CHUNK) {
            text.push_str(&listening.push(chunk));
        }
        text.push_str(&listening.finish());
        println!("RESULT\t{clip}\t{audio:.3}\t0\t0\t0\t{}", text.trim());
    }
    eprintln!(
        "[capture] {audio_total:.1} s of audio in {:.1} s; dumping",
        started.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    mary::models::voxtral::fast::capture::dump::<mary::hear::Backend>(Path::new(&out))?;
    eprintln!("[capture] dumped in {:.1} s", t.elapsed().as_secs_f64());
    Ok(())
}
