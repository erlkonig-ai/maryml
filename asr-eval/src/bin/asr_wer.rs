//! asr_wer: word and character error rates of one or more systems against
//! published references, per set and in total, with bootstrap 95% intervals
//! and paired differences against the first system.
//!
//!   asr_wer [--boot B] [--first K] --ref M1.tsv[,M2.tsv...] NAME=RESULTS.tsv[,more.tsv]...
//!
//! M*.tsv: `asr_prep` manifests (id, wav, seconds, reference). RESULTS: the
//! `RESULT\t<wav>\t...\t<text>` lines of hear_offline (text is the last
//! field). Utterances are joined on the wav path; only those every system
//! transcribed are scored, and `--first K` keeps each manifest's first K
//! lines (they are in a stable hash order, so smaller K nests in larger).
//! Normalisation: lowercase, every non-alphanumeric character to a space,
//! numbers kept as written. The total resamples within each set (stratified).

use anyhow::{Context, Result, ensure};
use asr::{edits, normalise};
use std::collections::{BTreeMap, HashMap};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Per utterance, per system: (word edits, char edits); plus ref word/char counts.
struct Utt {
    nw: usize,
    nc: usize,
    e: Vec<(usize, usize)>,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn main() -> Result<()> {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| -> Option<String> {
        let at = a.iter().position(|x| x == name)?;
        let v = a.get(at + 1).cloned();
        a.drain(at..(at + 2).min(a.len()));
        v
    };
    let boot: usize = flag("--boot").map_or(Ok(2000), |v| v.parse())?;
    let first: usize = flag("--first").map_or(Ok(usize::MAX), |v| v.parse())?;
    let refs = flag("--ref").context("--ref M.tsv[,M2.tsv]")?;
    ensure!(!a.is_empty(), "no systems");
    // set name -> [(wav, ref text)]
    let mut sets: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for m in refs.split(',') {
        let set = std::path::Path::new(m).parent().and_then(|p| p.file_name()).map_or(m.to_string(), |s| s.to_string_lossy().into());
        let body = std::fs::read_to_string(m).with_context(|| m.to_string())?;
        let v: Vec<(String, String)> = body
            .lines()
            .take(first)
            .map(|l| {
                let f: Vec<&str> = l.splitn(4, '\t').collect();
                (f[1].to_string(), f.get(3).unwrap_or(&"").to_string())
            })
            .collect();
        sets.insert(set, v);
    }
    let mut names = Vec::new();
    let mut hyps: Vec<HashMap<String, String>> = Vec::new();
    for s in &a {
        let (name, files) = s.split_once('=').context("NAME=RESULTS.tsv")?;
        let mut h = HashMap::new();
        for f in files.split(',') {
            for l in std::fs::read_to_string(f).with_context(|| f.to_string())?.lines() {
                if let Some(rest) = l.strip_prefix("RESULT\t") {
                    let fields: Vec<&str> = rest.split('\t').collect();
                    let text = if fields.len() >= 6 { fields[5] } else { "" };
                    h.insert(fields[0].to_string(), text.to_string());
                }
            }
        }
        names.push(name.to_string());
        hyps.push(h);
    }
    let k = names.len();
    let mut scored: BTreeMap<String, Vec<Utt>> = BTreeMap::new();
    for (set, list) in &sets {
        let mut v = Vec::new();
        let mut missing = 0;
        let mut empty = vec![0usize; k];
        for (wav, r) in list {
            if !hyps.iter().all(|h| h.contains_key(wav)) {
                missing += 1;
                continue;
            }
            let rw = normalise(r);
            let rc: Vec<char> = rw.join(" ").chars().collect();
            let mut e = Vec::new();
            for (s, h) in hyps.iter().enumerate() {
                let hw = normalise(&h[wav]);
                if hw.is_empty() && !rw.is_empty() {
                    empty[s] += 1;
                }
                let hc: Vec<char> = hw.join(" ").chars().collect();
                e.push((edits(&rw, &hw), edits(&rc, &hc)));
            }
            v.push(Utt { nw: rw.len(), nc: rc.len(), e });
        }
        eprintln!("{set}: {} scored, {missing} not transcribed by every system, empty outputs {:?}", v.len(), empty);
        if !v.is_empty() {
            scored.insert(set.clone(), v);
        }
    }
    let mut rng = Rng(0x5eed_a5e5);
    let report = |label: &str, groups: &[&Vec<Utt>], rng: &mut Rng| {
        let n: usize = groups.iter().map(|g| g.len()).sum();
        let nw: usize = groups.iter().flat_map(|g| g.iter()).map(|u| u.nw).sum();
        let nc: usize = groups.iter().flat_map(|g| g.iter()).map(|u| u.nc).sum();
        let point: Vec<(f64, f64)> = (0..k)
            .map(|s| {
                let ew: usize = groups.iter().flat_map(|g| g.iter()).map(|u| u.e[s].0).sum();
                let ec: usize = groups.iter().flat_map(|g| g.iter()).map(|u| u.e[s].1).sum();
                (100.0 * ew as f64 / nw as f64, 100.0 * ec as f64 / nc as f64)
            })
            .collect();
        let mut wer = vec![Vec::with_capacity(boot); k];
        let mut cer = vec![Vec::with_capacity(boot); k];
        let mut dw = vec![Vec::with_capacity(boot); k];
        for _ in 0..boot {
            let mut tw = 0usize;
            let mut tc = 0usize;
            let mut ew = vec![0usize; k];
            let mut ec = vec![0usize; k];
            for g in groups {
                for _ in 0..g.len() {
                    let u = &g[rng.below(g.len())];
                    tw += u.nw;
                    tc += u.nc;
                    for s in 0..k {
                        ew[s] += u.e[s].0;
                        ec[s] += u.e[s].1;
                    }
                }
            }
            for s in 0..k {
                let w = 100.0 * ew[s] as f64 / tw.max(1) as f64;
                wer[s].push(w);
                cer[s].push(100.0 * ec[s] as f64 / tc.max(1) as f64);
            }
            for s in 0..k {
                dw[s].push(wer[s].last().unwrap() - wer[0].last().unwrap());
            }
        }
        println!("{label}: {n} utterances, {nw} words");
        for s in 0..k {
            let (lo, hi) = (pct(&mut wer[s], 0.025), pct(&mut wer[s], 0.975));
            let (clo, chi) = (pct(&mut cer[s], 0.025), pct(&mut cer[s], 0.975));
            let mut line = format!(
                "  {:<10} WER {:6.2} [{:5.2}, {:5.2}]   CER {:5.2} [{:5.2}, {:5.2}]",
                names[s], point[s].0, lo, hi, point[s].1, clo, chi
            );
            if s > 0 {
                let above = dw[s].iter().filter(|&&d| d > 0.0).count() as f64 / boot as f64;
                let (dlo, dhi) = (pct(&mut dw[s], 0.025), pct(&mut dw[s], 0.975));
                line += &format!(
                    "   vs {}: {:+.2} [{:+.2}, {:+.2}] P(worse) {:.3}",
                    names[0], point[s].0 - point[0].0, dlo, dhi, above
                );
            }
            println!("{line}");
        }
    };
    for (set, v) in &scored {
        report(set, &[v], &mut rng);
    }
    if scored.len() > 1 {
        let groups: Vec<&Vec<Utt>> = scored.values().collect();
        report("TOTAL", &groups, &mut rng);
        let en: Vec<&Vec<Utt>> = scored.iter().filter(|(s, _)| s.starts_with("ls-") || s.contains("-en-")).map(|(_, v)| v).collect();
        let de: Vec<&Vec<Utt>> = scored.iter().filter(|(s, _)| s.contains("-de-")).map(|(_, v)| v).collect();
        if !en.is_empty() && !de.is_empty() {
            report("TOTAL en", &en, &mut rng);
            report("TOTAL de", &de, &mut rng);
        }
    }
    Ok(())
}
