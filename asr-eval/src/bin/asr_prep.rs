//! asr_prep: public test/dev sets to 16 kHz mono PCM16 WAV plus a manifest.
//!
//!   asr_prep librispeech <split.tar.gz> <outdir> <N>
//!   asr_prep fleurs <audio.tar.gz> <split.tsv> <outdir> <N>
//!   asr_prep mls <split.parquet> <outdir> <N>
//!
//! Keeps the N utterances whose FNV-1a hash of the id is smallest (a stable
//! subsample that nests: the first K lines of a larger N are the smaller set),
//! writes `<outdir>/<id>.wav` and `<outdir>/manifest.tsv`: id, wav, seconds,
//! reference text (as published, unnormalised), sorted by that hash.

use anyhow::{Context, Result, bail, ensure};
use asr::*;
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

struct Utt {
    id: String,
    text: String,
    audio: Vec<u8>,
    ext: &'static str,
}

fn pick<T>(items: impl IntoIterator<Item = (String, T)>, n: usize) -> BTreeMap<(u64, String), T> {
    let mut all: BTreeMap<(u64, String), T> = items.into_iter().map(|(id, t)| ((fnv(&id), id), t)).collect();
    while all.len() > n {
        let last = all.keys().next_back().unwrap().clone();
        all.remove(&last);
    }
    all
}

fn tar_entries(path: &Path, mut f: impl FnMut(&str, &mut dyn Read) -> Result<()>) -> Result<()> {
    let gz = flate2::read::GzDecoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    let mut ar = tar::Archive::new(gz);
    for e in ar.entries()? {
        let mut e = e?;
        let name = e.path()?.to_string_lossy().into_owned();
        f(&name, &mut e)?;
    }
    Ok(())
}

fn librispeech(tgz: &Path, n: usize) -> Result<Vec<Utt>> {
    // pass 1: transcripts
    let mut text: HashMap<String, String> = HashMap::new();
    tar_entries(tgz, |name, r| {
        if name.ends_with(".trans.txt") {
            let mut s = String::new();
            r.read_to_string(&mut s)?;
            for line in s.lines() {
                if let Some((id, t)) = line.split_once(' ') {
                    text.insert(id.to_string(), t.to_string());
                }
            }
        }
        Ok(())
    })?;
    let chosen = pick(text.into_iter(), n);
    let want: HashMap<String, String> = chosen.into_iter().map(|((_, id), t)| (id, t)).collect();
    let mut out = Vec::new();
    tar_entries(tgz, |name, r| {
        if let Some(stem) = name.strip_suffix(".flac") {
            let id = stem.rsplit('/').next().unwrap();
            if let Some(t) = want.get(id) {
                let mut b = Vec::new();
                r.read_to_end(&mut b)?;
                out.push(Utt { id: id.to_string(), text: t.clone(), audio: b, ext: "flac" });
            }
        }
        Ok(())
    })?;
    ensure!(out.len() == want.len(), "found {} of {} flacs", out.len(), want.len());
    Ok(out)
}

fn fleurs(tgz: &Path, tsv: &Path, n: usize) -> Result<Vec<Utt>> {
    // id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
    let s = std::fs::read_to_string(tsv)?;
    let mut rows = Vec::new();
    for line in s.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        ensure!(f.len() >= 3, "short tsv line: {line}");
        let file = f[1].to_string();
        let id = file.trim_end_matches(".wav").to_string();
        rows.push((id, (file, f[2].to_string())));
    }
    let chosen = pick(rows, n);
    let want: HashMap<String, (String, String)> = chosen.into_iter().map(|((_, id), v)| (v.0.clone(), (id, v.1))).collect();
    let mut out = Vec::new();
    tar_entries(tgz, |name, r| {
        let base = name.rsplit('/').next().unwrap();
        if let Some((id, t)) = want.get(base) {
            let mut b = Vec::new();
            r.read_to_end(&mut b)?;
            out.push(Utt { id: id.clone(), text: t.clone(), audio: b, ext: "wav" });
        }
        Ok(())
    })?;
    ensure!(out.len() == want.len(), "found {} of {} wavs", out.len(), want.len());
    Ok(out)
}

fn mls(pq: &Path, n: usize) -> Result<Vec<Utt>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;
    let reader = SerializedFileReader::new(std::fs::File::open(pq)?)?;
    // pass 1: ids and transcripts only
    let mut rows = Vec::new();
    let mut all = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        let (mut id, mut text, mut audio) = (None, None, None);
        for (name, field) in row.get_column_iter() {
            match (name.as_str(), field) {
                ("id", Field::Str(s)) => id = Some(s.clone()),
                ("transcript", Field::Str(s)) => text = Some(s.clone()),
                ("audio", Field::Group(g)) => {
                    for (n2, f2) in g.get_column_iter() {
                        if let ("bytes", Field::Bytes(b)) = (n2.as_str(), f2) {
                            audio = Some(b.data().to_vec());
                        }
                    }
                }
                _ => {}
            }
        }
        let id = id.context("mls row without id")?;
        rows.push((id.clone(), text.context("no transcript")?));
        all.push((id, audio.context("no audio bytes")?));
    }
    let chosen = pick(rows, n);
    let want: HashMap<String, String> = chosen.into_iter().map(|((_, id), t)| (id, t)).collect();
    let mut out = Vec::new();
    for (id, audio) in all {
        if let Some(t) = want.get(&id) {
            let ext = if audio.starts_with(b"fLaC") { "flac" } else if audio.starts_with(b"OggS") { "ogg" } else { "wav" };
            out.push(Utt { id, text: t.clone(), audio, ext });
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (utts, outdir) = match a.first().map(String::as_str) {
        Some("librispeech") if a.len() == 4 => (librispeech(Path::new(&a[1]), a[3].parse()?)?, &a[2]),
        Some("fleurs") if a.len() == 5 => (fleurs(Path::new(&a[1]), Path::new(&a[2]), a[4].parse()?)?, &a[3]),
        Some("mls") if a.len() == 4 => (mls(Path::new(&a[1]), a[3].parse()?)?, &a[2]),
        _ => bail!("usage: asr_prep librispeech TGZ OUT N | fleurs TGZ TSV OUT N | mls PARQUET OUT N"),
    };
    let out = PathBuf::from(outdir);
    std::fs::create_dir_all(&out)?;
    let mut lines: Vec<(u64, String)> = Vec::new();
    let mut secs = 0.0;
    for u in utts {
        let (x, rate) = decode(u.audio, u.ext).with_context(|| format!("decode {}", u.id))?;
        let x = to_16k(x, rate)?;
        let wav = out.join(format!("{}.wav", u.id));
        write_wav(&wav, &x)?;
        let s = x.len() as f64 / RATE as f64;
        secs += s;
        let text = u.text.replace(['\t', '\n'], " ");
        lines.push((fnv(&u.id), format!("{}\t{}\t{:.3}\t{}", u.id, wav.display(), s, text)));
    }
    lines.sort();
    let body: String = lines.iter().map(|(_, l)| format!("{l}\n")).collect();
    std::fs::write(out.join("manifest.tsv"), body)?;
    eprintln!("{}: {} utterances, {:.1} s ({:.2} h)", out.display(), lines.len(), secs, secs / 3600.0);
    Ok(())
}
