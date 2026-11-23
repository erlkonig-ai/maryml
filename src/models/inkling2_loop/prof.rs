//! Phase timers for profiling a Mini forward, off unless `MINI_PROFILE` is set.
//!
//! `MINI_PROFILE=1` adds wall time per named phase, with no device sync: host
//! phases are exact, device phases show only their enqueue cost.
//! `MINI_PROFILE=sync` syncs the device before and after every phase, so a
//! device phase's time is its execution. That serialises the stream, so its
//! total is an upper bound on the unsynced forward.
//!
//! Phases are keyed `layer.pass.name`; [`report`] prints them, and per name
//! summed over layers, sorted by time.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

use burn::tensor::backend::Backend;

use crate::models::inkling::seam::Bk;
use crate::models::inkling2_loop::sliding::Dev;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Wall,
    Sync,
}

fn mode() -> Mode {
    static M: OnceLock<Mode> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("MINI_PROFILE").as_deref() {
        Ok("sync") => Mode::Sync,
        Ok(_) => Mode::Wall,
        Err(_) => Mode::Off,
    })
}

static TABLE: Mutex<BTreeMap<String, (f64, u64)>> = Mutex::new(BTreeMap::new());
static COUNTS: Mutex<BTreeMap<String, (u64, u64)>> = Mutex::new(BTreeMap::new());
static CTX: Mutex<(usize, usize)> = Mutex::new((usize::MAX, 0));

pub fn enabled() -> bool {
    mode() != Mode::Off
}

/// The layer and pass the next phases belong to.
pub fn at(layer: usize, pass: usize) {
    if enabled() {
        *CTX.lock().unwrap() = (layer, pass);
    }
}

fn sync(dev: &Dev) {
    let _ = <Bk as Backend>::sync(dev);
}

/// Time `f` as phase `name` of the current layer and pass.
pub fn span<T>(name: &str, dev: &Dev, f: impl FnOnce() -> T) -> T {
    let m = mode();
    if m == Mode::Off {
        return f();
    }
    if m == Mode::Sync {
        sync(dev);
    }
    let t0 = Instant::now();
    let out = f();
    if m == Mode::Sync {
        sync(dev);
    }
    add(name, t0.elapsed().as_secs_f64());
    out
}

fn key(name: &str) -> String {
    let (l, p) = *CTX.lock().unwrap();
    if l == usize::MAX { format!("-.-.{name}") } else { format!("{l:02}.{p}.{name}") }
}

/// Add `secs` measured elsewhere to phase `name`.
pub fn add(name: &str, secs: f64) {
    if !enabled() {
        return;
    }
    let mut tab = TABLE.lock().unwrap();
    let e = tab.entry(key(name)).or_insert((0.0, 0));
    e.0 += secs;
    e.1 += 1;
}

/// Record a count (kept apart from times, reported as `count`).
pub fn count(name: &str, n: usize) {
    if !enabled() {
        return;
    }
    let mut c = COUNTS.lock().unwrap();
    let e = c.entry(name.to_string()).or_insert((0, 0));
    e.0 += n as u64;
    e.1 += 1;
}

/// Print and clear the table: every key, then totals per phase name.
pub fn report(label: &str) {
    if !enabled() {
        return;
    }
    let tab = std::mem::take(&mut *TABLE.lock().unwrap());
    for (k, (sum, n)) in std::mem::take(&mut *COUNTS.lock().unwrap()) {
        eprintln!("PROFILE {label} count {k} sum={sum} n={n} mean={:.1}", sum as f64 / n.max(1) as f64);
    }
    let total: f64 = tab.values().map(|v| v.0).sum();
    eprintln!("PROFILE {label} total_in_phases {total:.3} s");
    for (k, (s, n)) in &tab {
        eprintln!("PROFILE {label} key {k} {s:.4} s n={n}");
    }
    let mut by: BTreeMap<String, (f64, u64)> = BTreeMap::new();
    for (k, (s, n)) in &tab {
        let name = k.splitn(3, '.').nth(2).unwrap_or(k).to_string();
        let e = by.entry(name).or_insert((0.0, 0));
        e.0 += s;
        e.1 += n;
    }
    let mut v: Vec<_> = by.into_iter().collect();
    v.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    for (name, (s, n)) in v {
        eprintln!("PROFILE {label} phase {name} {s:.3} s ({:.1}%) n={n}", 100.0 * s / total.max(1e-9));
    }
}
