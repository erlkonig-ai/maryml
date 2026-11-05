//! Finite synthetic fixtures for the phase-A tensor-core attention kernel.
//! Test-only: no model, no pile, no runtime policy, admission or cache ABI.
//!
//! Every case runs four arms on the same immutable BF16 Q and BF16 K/V runs:
//! * `scalar-cast-rows48`: the existing cast path, `burn` cast of Q to F32 then
//!   `flash::flash_attention_launch` at 48 rows, as the BF16-query
//!   diagnostic's cast arm runs it (the cast is inside the wall bracket);
//! * `mma-hilo-b4`, `mma-hilo-b8`, `mma-bf16-b8`: [`mma_attention_launch`].
//!
//! Small cases are compared with an independent f64 two-pass oracle against
//! the existing 5e-5 / 1e-5 bounds; large cases with the scalar baseline. Both
//! are RECORDED, never asserted: until the hi/lo arm's own error expectation is
//! established, the amended design makes a miss an investigation, not a
//! failure. The HiLo-vs-oracle misses are listed in the `completed` record and
//! in `complete.json`. The hard asserts are finiteness, coverage and the
//! partial-write canary (`gpu_mma_attention_partial_canary`).
//!
//! Each masking/padding fixture also carries a host replay of the kernel's own
//! split and causal trimming (`coverage`), and asserts that the property it is
//! named for actually occurs in the launch, so a fixture cannot silently stop
//! exercising its case.
use super::*;
use crate::models::inkling::flash::KvElem;
use crate::models::nemotron_h::{attention::HIDDEN, fp8::CudaTensor};
use burn::tensor::DType;
use cubecl::{
    client::ComputeClient,
    cuda::{CudaDevice, CudaRuntime},
};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const SEED: u64 = 0x5eeda77e;
const TAIL: usize = 16;
const LONG_KEYS: usize = 524_288;
const LONG_QUERIES: usize = 256;
const MAX_SECONDS: u64 = 250;
const MAX_RECORDS: usize = 160;
const ABS_LIMIT: f64 = 5e-5;
const SCALED_LIMIT: f64 = 1e-5;
const BASELINE_ROWS: usize = 48;

#[test]
fn mma_host_checked_plan_sizes_and_rejections() {
    for (warps, slots) in [(4, 8), (8, 16)] {
        for mode in [PMode::Bf16, PMode::HiLo] {
            let p = Plan::new(&[524288], 256, 524032, warps, mode).unwrap();
            assert_eq!(p.split_of, [slots]);
            assert_eq!(p.po_elems * 4, slots * 256 * 4096 * 4);
            assert_eq!(p.pml_elems * 4, slots * 256 * 32 * 2 * 4);
            assert_eq!(p.working_bytes, (p.po_elems + p.pml_elems) * 4 + 256 * 4096 * 6);
            assert!(p.validate_splits(&[0]).is_err());
            assert!(p.validate_splits(&[]).is_err());
            assert!(p.validate_splits(&[slots + 1]).is_err());
        }
    }
    for (spans, nq, q0, warps) in [
        (vec![], 1, 0, 4), (vec![1], 0, 0, 4), (vec![1], 257, 0, 4),
        (vec![1], 1, 0, 0), (vec![1], 1, 0, 16),
        (vec![1], 1, u32::MAX as usize - 1, 4),
        (vec![usize::MAX, 1], 1, 0, 4),
    ] {
        assert!(Plan::new(&spans, nq, q0, warps, PMode::Bf16).is_err());
    }
    assert!(extent(&[usize::MAX, 2], 4).is_err());
    assert!(extent(&[u32::MAX as usize], 4).is_err());
    assert_eq!(Plan::new(&[0], 3, 0, 4, PMode::Bf16).unwrap().split_of, [1]);
    assert_eq!(shared_bytes(), 35840);
}

#[test]
fn mma_host_prefix_rounding_earlier_endpoint_exceeds_final_width_envelope() {
    let earlier = Plan::new(&[8192, 130816 - 8192], 256, 130560, 8, PMode::Bf16).unwrap();
    assert_eq!(earlier.split_of, [2, 15]);
    assert_eq!(earlier.nq * earlier.slots, 4352);
    let final_max = (2..=256).map(|t| {
        let p = Plan::new(&[8192, 131072 - 8192], t, 131072 - t, 8, PMode::Bf16).unwrap();
        p.nq * p.slots
    }).max().unwrap();
    assert_eq!(final_max, 4320);
    assert!(earlier.nq * earlier.slots > final_max);
}

// ------------------------------------------------------------------ cases --

/// One finite synthetic case. `cut == 0` is one run over `[0, capacity)`;
/// otherwise a prefix run `[0, cut)` and a private run `[cut, capacity)` with
/// their own buffers and absolute bases. Rows at or past `keys` are NaN.
#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    queries: usize,
    keys: usize,
    capacity: usize,
    q0: usize,
    cut: usize,
    uniform: bool,
}
impl Case {
    fn validate(self) {
        assert!((1..=LONG_QUERIES).contains(&self.queries));
        // No query beyond the key extent; q0 + queries < keys is allowed (a
        // block whose causal limit ends well before the run's later keys).
        assert!(self.q0 + self.queries <= self.keys);
        assert!(self.keys <= self.capacity && self.capacity <= LONG_KEYS + TAIL);
        assert!(self.cut < self.keys);
    }
    fn spans(self) -> Vec<(usize, usize)> {
        if self.cut == 0 {
            vec![(0, self.capacity)]
        } else {
            vec![(0, self.cut), (self.cut, self.capacity - self.cut)]
        }
    }
    /// `(base, lo, hi)` of each run, as the launchers see them.
    fn runs(self) -> Vec<(usize, usize, usize)> {
        self.spans().into_iter().map(|(base, rows)| (base, 0, rows.min(self.keys - base))).collect()
    }
    fn small(self) -> bool {
        self.queries * (self.q0 + self.queries) <= 17 * 281
    }
    fn json(self) -> Value {
        json!({"name":self.name,"queries":self.queries,"keys":self.keys,"capacity":self.capacity,
            "q0":self.q0,"cut":self.cut,"uniform":self.uniform,"runs":self.runs()})
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Baseline,
    Mma(usize, PMode),
}
const ARMS: [Arm; 4] =
    [Arm::Baseline, Arm::Mma(4, PMode::HiLo), Arm::Mma(8, PMode::HiLo), Arm::Mma(8, PMode::Bf16)];
impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Baseline => format!("scalar-cast-rows{BASELINE_ROWS}"),
            Arm::Mma(w, mode) => format!("mma-{}-b{w}", mode.as_str()),
        }
    }
    fn q_tiles(self, nq: usize) -> usize {
        match self {
            Arm::Baseline => nq.div_ceil(BASELINE_ROWS / (HEADS / KV_HEADS)),
            Arm::Mma(w, _) => query_tiles(nq, w),
        }
    }
    fn plan(self, case: Case) -> Value {
        let spans: Vec<_> = case.runs().iter().map(|&(_, lo, hi)| hi - lo).collect();
        let tiles = self.q_tiles(case.queries);
        let splits = flash::key_run_splits(&spans, tiles, KV_HEADS);
        let slots: usize = splits.iter().sum();
        let shared = match self {
            Arm::Baseline => flash::shared_floats(BASELINE_ROWS as u32, HEAD_DIM as u32) * 4,
            Arm::Mma(..) => shared_bytes(),
        };
        let partial = slots * case.queries * HIDDEN * 4;
        let stats = slots * case.queries * HEADS * 2 * 4;
        let named = case.capacity * KV_WIDTH * 4 + case.queries * HIDDEN * 10 + partial + stats;
        assert!(named <= 640 << 20, "named allocation ceiling");
        json!({"arm":self.label(),"query_tiles":tiles,"splits":splits,"partial_bytes":partial,
            "stats_bytes":stats,"named_device_bytes":named,"shared_bytes_per_cube":shared})
    }
}

// ------------------------------------------------------------- coverage --

/// Host replay of the kernel's per-cube trimming and splitting (one KV head;
/// both are identical), counting the masking/padding events a case is meant
/// to exercise. Mirrors `mma_attention_kernel` line for line, so it describes
/// what that launch visits, not what the math needs.
fn coverage(case: Case, warps: usize) -> Value {
    let runs = case.runs();
    let spans: Vec<_> = runs.iter().map(|&(_, lo, hi)| hi - lo).collect();
    let q_tiles = query_tiles(case.queries, warps);
    let splits = flash::key_run_splits(&spans, q_tiles, KV_HEADS);
    let (nq, q0) = (case.queries, case.q0);
    let mut c = [0usize; 7];
    // padded plane-splits, empty splits, fully masked plane-splits of a
    // NON-EMPTY split (an empty split never runs the tile loop, so it is
    // counted only as empty), key tiles visited per plane, fully masked
    // plane-tiles, masked tile after a visible tile in the same plane-split,
    // partial (s_hi tail) tiles.
    for (&(k0, klo, khi), &n) in runs.iter().zip(&splits) {
        for qt in 0..q_tiles {
            let q_lo = q0 + qt * warps;
            let q_hi = if qt * warps + warps <= nq { q_lo + warps - 1 } else { q0 + nq - 1 };
            let lo = klo;
            let mut hi = khi;
            if q_hi + 1 <= k0 {
                hi = lo;
            } else if q_hi + 1 - k0 < hi {
                hi = q_hi + 1 - k0;
            }
            hi = hi.max(lo);
            let per = (hi - lo).div_ceil(n);
            for split in 0..n {
                let s_lo = (lo + split * per).min(hi);
                let s_hi = (s_lo + per).min(hi);
                if s_lo == s_hi {
                    c[1] += 1;
                }
                for w in 0..warps {
                    let p = qt * warps + w;
                    if p >= nq {
                        c[0] += 1;
                        continue;
                    }
                    let q_abs = q0 + p;
                    let (mut any, mut seen) = (false, false);
                    let mut t = s_lo;
                    while t < s_hi {
                        let end = (t + NK).min(s_hi);
                        c[3] += 1;
                        if end - t < NK {
                            c[6] += 1;
                        }
                        let visible = (t..end).filter(|&key| k0 + key <= q_abs).count();
                        if visible == 0 {
                            c[4] += 1;
                            if seen {
                                c[5] += 1;
                            }
                        } else {
                            seen = true;
                            any = true;
                        }
                        t += NK;
                    }
                    if s_lo < s_hi && !any {
                        c[2] += 1;
                    }
                }
            }
        }
    }
    json!({"warps":warps,"query_tiles":q_tiles,"splits":splits,"padded_plane_splits":c[0],
        "empty_splits":c[1],"fully_masked_nonempty_plane_splits":c[2],"plane_tiles":c[3],
        "fully_masked_plane_tiles":c[4],"masked_tile_after_visible":c[5],"partial_tail_tiles":c[6]})
}

fn count(cov: &Value, key: &str) -> u64 {
    cov[key].as_u64().unwrap()
}

// ------------------------------------------------------------- evidence --

fn evidence_root() -> &'static PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = PathBuf::from(
            std::env::var("NEMOTRON_ATTENTION_ROWS_OUTPUT")
                .expect("explicit new fixture evidence directory"),
        );
        assert!(root.is_absolute());
        fs::create_dir(&root).expect("new evidence directory, no overwrite");
        root
    })
}

/// One GPU fixture at a time in this process, so neither the device nor the
/// timing is shared between two of them.
fn gpu_turn() -> std::sync::MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

// One subdirectory per fixture, also created new.
struct Evidence {
    root: PathBuf,
    file: fs::File,
    started: Instant,
    records: usize,
}
impl Evidence {
    fn new(fixture: &str) -> Self {
        assert_eq!(std::env::var("CUBECL_MEMORY_CONFIG").as_deref(), Ok("exclusive"));
        assert!(std::env::var_os("CUBECL_DEBUG_LOG").is_none());
        assert!(std::env::var_os("CUBECL_DEBUG_OPTION").is_none());
        let root = evidence_root().join(fixture);
        fs::create_dir(&root).expect("new fixture evidence directory, no overwrite");
        let file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(root.join("events.jsonl"))
            .unwrap();
        Self { root, file, started: Instant::now(), records: 0 }
    }
    fn check(&self) {
        assert!(
            self.started.elapsed() < Duration::from_secs(MAX_SECONDS),
            "finite fixture exceeded elapsed budget; external timeout also required"
        );
    }
    fn record(&mut self, label: &str, data: Value) {
        self.check();
        self.records += 1;
        assert!(self.records <= MAX_RECORDS);
        let record = json!({"label":label,"process_elapsed_ns":self.started.elapsed().as_nanos(),
            "unix_ns":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),"data":data});
        let encoded = serde_json::to_vec(&record).unwrap();
        assert!(encoded.len() <= 16_384);
        self.file.write_all(&encoded).unwrap();
        self.file.write_all(b"\n").unwrap();
        self.file.flush().unwrap();
    }
}

// --------------------------------------------------------------- inputs --

fn sample(index: usize, salt: u64) -> f32 {
    let mut x = (index as u64).wrapping_add(SEED ^ salt).wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    bf16::from_f32(((x >> 48) as i32 - 32768) as f32 / 131072.0).to_f32()
}

fn fence(client: &ComputeClient<CudaRuntime>) {
    cubecl::future::block_on(client.sync()).expect("fixture completion fence");
}

fn upload(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    shape: &[usize],
    mut value: impl FnMut(usize) -> f32,
) -> CudaTensor {
    let n: usize = shape.iter().product();
    assert!(n * 2 <= (256 << 20) + TAIL * KV_WIDTH * 2);
    let mut bytes = Vec::with_capacity(n * 2);
    for i in 0..n {
        bytes.extend_from_slice(&bf16::from_f32(value(i)).to_bits().to_le_bytes());
    }
    let handle = client.create_from_slice(&bytes);
    // Retire this upload before generating the next large host input.
    fence(client);
    CudaTensor::new_contiguous(client.clone(), device.clone(), shape.into(), handle, DType::BF16)
}

fn key_buffer(
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    case: Case,
    start: usize,
    rows: usize,
    salt: u64,
) -> CudaTensor {
    assert!(start + rows <= case.capacity);
    upload(client, device, &[rows, KV_WIDTH], |i| {
        let global = start * KV_WIDTH + i;
        if global / KV_WIDTH >= case.keys { f32::NAN } else { sample(global, salt) }
    })
}

// Independent two-pass f64 reference on the exact rounded BF16 operands:
// query `qi` sees absolute keys `0 ..= q0 + qi`. No tiles, splits, online
// recurrence, fragments or kernel mask arithmetic.
fn oracle(case: Case) -> Vec<f64> {
    assert!(case.small());
    let seen = case.q0 + case.queries;
    let q: Vec<_> = (0..case.queries * HIDDEN)
        .map(|i| if case.uniform { 0.0 } else { f64::from(sample(i, 1)) })
        .collect();
    let k: Vec<_> = (0..seen * KV_WIDTH).map(|i| f64::from(sample(i, 2))).collect();
    let v: Vec<_> = (0..seen * KV_WIDTH).map(|i| f64::from(sample(i, 3))).collect();
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let mut out = vec![0.0; case.queries * HIDDEN];
    for qi in 0..case.queries {
        for h in 0..HEADS {
            let kvh = h / (HEADS / KV_HEADS);
            let qo = qi * HIDDEN + h * HEAD_DIM;
            let scores: Vec<_> = (0..=case.q0 + qi)
                .map(|j| {
                    let ko = j * KV_WIDTH + kvh * HEAD_DIM;
                    (0..HEAD_DIM).map(|d| q[qo + d] * k[ko + d]).sum::<f64>() * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores.iter().map(|s| (s - max).exp()).sum();
            for (j, score) in scores.into_iter().enumerate() {
                let p = (score - max).exp() / denom;
                for d in 0..HEAD_DIM {
                    out[qo + d] += p * v[j * KV_WIDTH + kvh * HEAD_DIM + d];
                }
            }
        }
    }
    out
}

/// Recorded, never asserted here: `within` says whether the existing bounds
/// hold, and the caller decides what a miss means.
fn compare(got: &[f32], expected: &[f64]) -> Value {
    assert_eq!(got.len(), expected.len());
    let (mut absolute, mut scaled, mut worst) = (0.0f64, 0.0f64, 0usize);
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        let e = (f64::from(a) - b).abs();
        if e > absolute {
            worst = i;
        }
        absolute = absolute.max(e);
        scaled = scaled.max(e / b.abs().max(1.0));
    }
    json!({"elements":got.len(),"max_abs":absolute,"max_scaled_floor1":scaled,
        "worst":{"index":worst,"query":worst / HIDDEN,"head":worst % HIDDEN / HEAD_DIM,
            "dim":worst % HEAD_DIM},
        "absolute_limit":ABS_LIMIT,"scaled_limit":SCALED_LIMIT,
        "within":absolute <= ABS_LIMIT && scaled <= SCALED_LIMIT})
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

fn differing_bits(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

// --------------------------------------------------------------- launch --

struct Inputs {
    q: CudaTensor,
    owners: Vec<(CudaTensor, CudaTensor)>,
}
impl Inputs {
    fn new(client: &ComputeClient<CudaRuntime>, device: &CudaDevice, case: Case) -> Self {
        case.validate();
        let q = upload(client, device, &[1, case.queries, HIDDEN], |i| {
            if case.uniform { 0.0 } else { sample(i, 1) }
        });
        let owners = case
            .spans()
            .into_iter()
            .map(|(start, rows)| {
                (key_buffer(client, device, case, start, rows, 2),
                    key_buffer(client, device, case, start, rows, 3))
            })
            .collect();
        Self { q, owners }
    }
    fn runs(&self, case: Case) -> Vec<KeyRun<'_>> {
        self.owners
            .iter()
            .zip(case.runs())
            .zip(case.spans())
            .map(|(((k, v), (base, lo, hi)), (_, rows))| KeyRun {
                k: &k.handle,
                v: &v.handle,
                k_scales: None,
                v_scales: None,
                rows,
                base,
                lo,
                hi,
                row0: 0,
            })
            .collect()
    }
}

/// One launch of one arm, bracketed by completion fences. The wall is host
/// enqueue plus the completion fence (the baseline's Q cast included), NOT a
/// CUDA-event kernel duration. Readback and persistence are outside it.
fn run_arm(
    arm: Arm,
    case: Case,
    inputs: &Inputs,
    evidence: &mut Evidence,
    phase: &str,
) -> Vec<f32> {
    let q = &inputs.q;
    let client = &q.client;
    let runs = inputs.runs(case);
    let scaling = 1.0 / (HEAD_DIM as f32).sqrt();
    fence(client);
    let started = Instant::now();
    let (out, widened) = match arm {
        Arm::Baseline => {
            let q32 = burn_cubecl::kernel::cast(q.clone(), DType::F32);
            let out = flash::flash_attention_launch(client, &q32.handle, &runs, &q32.handle,
                KvElem::Bf16, case.queries, case.q0, HEADS, KV_HEADS, HEAD_DIM, 0, None,
                scaling, BASELINE_ROWS, None);
            (out, Some(q32))
        }
        Arm::Mma(warps, mode) => (
            mma_attention_launch(client, &q.handle, &runs, case.queries, case.q0, scaling,
                warps, mode),
            None,
        ),
    };
    let enqueue_ns = started.elapsed().as_nanos();
    let waiting = Instant::now();
    fence(client);
    let fence_ns = waiting.elapsed().as_nanos();
    let completed_ns = started.elapsed().as_nanos();
    // The cast temporary stays owned until completion, as do Q and the runs.
    drop(widened);
    let bytes = client.read_one(out).unwrap();
    let n = case.queries * HIDDEN;
    assert!(bytes.len() >= n * 4);
    let values: Vec<f32> =
        bytes[..n * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let nonfinite = values.iter().filter(|v| !v.is_finite()).count();
    evidence.record("launch", json!({"case":case.name,"phase":phase,"arm":arm.label(),
        "plan":arm.plan(case),"host_enqueue_ns":enqueue_ns,"host_fence_ns":fence_ns,
        "host_completed_ns":completed_ns,"gpu_duration_ns":Value::Null,"nonfinite":nonfinite,
        "output_blake3":blake3::hash(&bytes[..n * 4]).to_hex().to_string()}));
    assert_eq!(nonfinite, 0, "nonfinite output: {} {phase}", arm.label());
    values
}

/// All four arms once, then the comparisons. Returns the HiLo-vs-oracle
/// misses (recorded by the caller, an investigation and never a failure) and
/// the first outputs, for the timing pass's determinism check.
fn run_case(
    client: &ComputeClient<CudaRuntime>,
    inputs: &Inputs,
    case: Case,
    evidence: &mut Evidence,
) -> (Vec<String>, Vec<Vec<f32>>) {
    let outputs: Vec<_> =
        ARMS.iter().map(|&arm| run_arm(arm, case, inputs, evidence, "first")).collect();
    let baseline = &outputs[0];
    let reference = widen(baseline);
    let expected = case.small().then(|| oracle(case));
    let mut arms = Vec::new();
    let mut misses = Vec::new();
    if let Some(expected) = &expected {
        arms.push(json!({"arm":ARMS[0].label(),"vs_oracle":compare(baseline, expected)}));
    }
    for (arm, got) in ARMS.iter().zip(&outputs).skip(1) {
        let mut row = json!({"arm":arm.label(),"vs_baseline":compare(got, &reference),
            "differing_bits_vs_baseline":differing_bits(got, baseline)});
        if let Some(expected) = &expected {
            let c = compare(got, expected);
            if matches!(arm, Arm::Mma(_, PMode::HiLo)) && c["within"] != json!(true) {
                misses.push(format!("{} {}: {c}", case.name, arm.label()));
            }
            row["vs_oracle"] = c;
        }
        arms.push(row);
    }
    evidence.record("comparison", json!({"case":case.name,"oracle":expected.is_some(),
        "asserted":"nothing: record only, a miss is an investigation",
        "arms":arms}));
    fence(client);
    (misses, outputs)
}

fn contract(evidence: &mut Evidence, cases: &[Case]) {
    evidence.record("contract", json!({"seed":SEED,"heads":HEADS,"kv_heads":KV_HEADS,
        "head_dim":HEAD_DIM,"nk":NK,"max_seconds":MAX_SECONDS,
        "arms":ARMS.iter().map(|a| a.label()).collect::<Vec<_>>(),
        "mma_shared_bytes_per_cube":{"b4":shared_bytes(),"b8":shared_bytes()},
        "shared_limit_bytes":SHARED_LIMIT_BYTES,
        "timing":"host enqueue plus completion fence per launch; baseline includes its Q cast; \
            NOT CUDA-event duration (gpu_duration_ns null)",
        "bounds":{"abs":ABS_LIMIT,"scaled_floor1":SCALED_LIMIT,
            "asserted":"nothing: recorded, a hilo miss vs the f64 oracle is an investigation"},
        "v_staging":V_STAGING,
        "cases":cases.iter().map(|c| c.json()).collect::<Vec<_>>()}));
}

// ------------------------------------------------------------- fixtures --

const ORACLE_CASES: [Case; 10] = [
    Case { name: "small-17x281-cut0-uniform", queries: 17, keys: 281, capacity: 288, q0: 264,
        cut: 0, uniform: true },
    Case { name: "small-17x281-cut127-uniform", queries: 17, keys: 281, capacity: 288, q0: 264,
        cut: 127, uniform: true },
    Case { name: "small-17x281-cut0", queries: 17, keys: 281, capacity: 288, q0: 264, cut: 0,
        uniform: false },
    Case { name: "small-17x281-cut127", queries: 17, keys: 281, capacity: 288, q0: 264,
        cut: 127, uniform: false },
    Case { name: "small-1x281-cut127", queries: 1, keys: 281, capacity: 288, q0: 280, cut: 127,
        uniform: false },
    // nq = 3 < B: one query tile, 1 (B4) or 5 (B8) padded planes.
    Case { name: "padded-3x281", queries: 3, keys: 281, capacity: 288, q0: 278, cut: 0,
        uniform: false },
    // One split (100 <= 128 keys); the last cube's s_hi = 100 ends a key tile
    // at 36 of 64; 17 queries also pad both B.
    Case { name: "tail-17x100", queries: 17, keys: 100, capacity: 112, q0: 83, cut: 0,
        uniform: false },
    // Prefix/private two-run split at an unaligned cut, over the same tail.
    Case { name: "two-run-17x100-cut37", queries: 17, keys: 100, capacity: 112, q0: 83,
        cut: 37, uniform: false },
    // One split; cube 0 (positions 62..65 at B4, 62..69 at B8) trims to
    // [0, 66) / [0, 70), so planes at 62 and 63 see key tile 0 and then a
    // FULLY MASKED key tile 1 (keys 64..) after it.
    Case { name: "masked-tile-10x72", queries: 10, keys: 72, capacity: 80, q0: 62, cut: 0,
        uniform: false },
    // q0 = 0 over a 16,384-key run: the causal limit (17) ends long before the
    // run's later tiles, the trimmed range is split 102 (B4) or 128 (B8) ways,
    // so most splits are EMPTY and many planes see none of their split.
    Case { name: "masked-splits-17x16384", queries: 17, keys: 16_384, capacity: 16_400, q0: 0,
        cut: 0, uniform: false },
];

/// The property each masking/padding fixture is named for must occur at
/// every mma B it runs (host replay of the launch, recorded beside it).
fn assert_coverage(case: Case, warps: usize, cov: &Value) {
    let need = |key: &str| {
        assert!(count(cov, key) > 0, "{} at B{warps} does not exercise {key}: {cov}", case.name)
    };
    match case.name {
        "padded-3x281" => need("padded_plane_splits"),
        "tail-17x100" | "two-run-17x100-cut37" => {
            need("partial_tail_tiles");
            need("padded_plane_splits");
        }
        "masked-tile-10x72" => need("masked_tile_after_visible"),
        "masked-splits-17x16384" => {
            need("empty_splits");
            need("fully_masked_nonempty_plane_splits");
        }
        _ => {}
    }
    if case.cut > 0 {
        assert_eq!(cov["splits"].as_array().unwrap().len(), 2);
    }
}

#[test]
#[ignore = "finite owned GPU: mma attention vs f64 oracle and scalar baseline, small and masking cases"]
fn gpu_mma_attention_oracle_fixtures() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_oracle_fixtures");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    contract(&mut evidence, &ORACLE_CASES);
    let mut misses = Vec::new();
    for case in ORACLE_CASES {
        assert!(case.small());
        let coverage: Vec<_> = [4, 8].map(|w| coverage(case, w)).into_iter().collect();
        evidence.record("coverage", json!({"case":case.name,"b4":coverage[0],"b8":coverage[1]}));
        for (w, cov) in [4, 8].into_iter().zip(&coverage) {
            assert_coverage(case, w, cov);
        }
        let inputs = Inputs::new(&client, &device, case);
        let (m, _) = run_case(&client, &inputs, case, &mut evidence);
        misses.extend(m);
        drop(inputs);
        fence(&client);
    }
    // `passed` covers the hard asserts only (finiteness, coverage); a hilo
    // miss is listed for investigation and fails nothing.
    let summary = json!({"passed":true,"hilo_oracle_within":misses.is_empty(),
        "hilo_oracle_misses":misses,"launches":ORACLE_CASES.len() * ARMS.len()});
    evidence.record("completed", summary.clone());
    if !summary["hilo_oracle_within"].as_bool().unwrap() {
        eprintln!("INVESTIGATE: hilo arm outside the f64 oracle bounds: {}",
            summary["hilo_oracle_misses"]);
    }
    fs::write(evidence.root.join("complete.json"), format!("{summary}\n")).unwrap();
}

/// A quiet-NaN bit pattern the kernel never produces (every partial it writes
/// is finite: masked keys are zero-staged, masked scores the finite NEG_INF).
const CANARY: u32 = 0x7fc0_cafe;

/// The partial pass alone, into `po`/`pml` filled with [`CANARY`] and LONGER
/// than the launch owns by `warps` query rows. A padded plane (p >= nq) that
/// lost its `live` guard writes cell `slot * nq + p`: in every slot but the
/// last that lands in the next slot's rows, where the cube that finishes last
/// wins (race-dependent), but in the LAST slot it lands past `slots * nq`,
/// inside the canary region, every time. So: every owned word must be written
/// and finite, and every canary word must be untouched.
fn partial_canary(
    client: &ComputeClient<CudaRuntime>,
    inputs: &Inputs,
    case: Case,
    warps: usize,
    mode: PMode,
    evidence: &mut Evidence,
) {
    let runs = inputs.runs(case);
    let nq = case.queries;
    let split_of = split_plan(&runs, nq, warps);
    let slots: usize = split_of.iter().sum();
    let rows = slots * nq;
    let past_end = query_tiles(nq, warps) * warps - nq;
    let (po_live, po_len) = (rows * HEADS * HEAD_DIM, (rows + warps) * HEADS * HEAD_DIM);
    let (pml_live, pml_len) = (rows * HEADS * 2, (rows + warps) * HEADS * 2);
    let fill = |n: usize| {
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&CANARY.to_le_bytes());
        }
        client.create_from_slice(&bytes)
    };
    let po = fill(po_len);
    let pml = fill(pml_len);
    fence(client);
    let scaling = 1.0 / (HEAD_DIM as f32).sqrt();
    mma_attention_partials(client, &inputs.q.handle, &runs, nq, case.q0, scaling, warps, mode,
        &split_of, &po, po_len, &pml, pml_len);
    fence(client);
    let words = |h: Handle, n: usize| -> Vec<u32> {
        let bytes = client.read_one(h).unwrap();
        assert!(bytes.len() >= n * 4);
        bytes[..n * 4].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
    };
    let mut buffers = Vec::new();
    let mut clean = true;
    for (name, got, live) in [("po", words(po, po_len), po_live), ("pml", words(pml, pml_len), pml_live)] {
        let unwritten = got[..live].iter().filter(|&&w| w == CANARY).count();
        let nonfinite = got[..live].iter().filter(|&&w| !f32::from_bits(w).is_finite()).count();
        let past_end_written = got[live..].iter().filter(|&&w| w != CANARY).count();
        clean &= unwritten == 0 && nonfinite == 0 && past_end_written == 0;
        buffers.push(json!({"buffer":name,"owned_words":live,"canary_words":got.len() - live,
            "owned_unwritten":unwritten,"owned_nonfinite":nonfinite,
            "canary_overwritten":past_end_written}));
    }
    let label = Arm::Mma(warps, mode).label();
    evidence.record("partial_canary", json!({"case":case.name,"arm":label,"splits":split_of,
        "slots":slots,"padded_rows_past_last_slot":past_end,"canary":format!("{CANARY:#010x}"),
        "buffers":buffers,"clean":clean}));
    if matches!(case.name, "padded-3x281" | "tail-17x100" | "two-run-17x100-cut37") {
        assert!(past_end > 0, "{} at B{warps} has no padded plane past the last slot", case.name);
    }
    assert!(clean, "partial pass wrote outside its cells or left one unwritten: {} {label}", case.name);
}

#[test]
#[ignore = "finite owned GPU: mma partial pass writes every owned cell and nothing past them (NaN canary)"]
fn gpu_mma_attention_partial_canary() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_partial_canary");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    contract(&mut evidence, &ORACLE_CASES);
    let mut checks = 0usize;
    for case in ORACLE_CASES {
        let inputs = Inputs::new(&client, &device, case);
        for arm in ARMS {
            if let Arm::Mma(warps, mode) = arm {
                partial_canary(&client, &inputs, case, warps, mode, &mut evidence);
                checks += 1;
            }
        }
        drop(inputs);
        fence(&client);
    }
    evidence.record("completed", json!({"passed":true,"checks":checks}));
    fs::write(evidence.root.join("complete.json"), b"{\"passed\":true}\n").unwrap();
}

#[test]
#[ignore = "reserved GPU: checked MMA boundary nonzero lo and fully masked output; under 1MiB named payload"]
fn gpu_mma_checked_nonzero_lo_and_all_masked() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_checked_ranges");
    // Admit named payload BEFORE upload/launch, not as after-the-fact accounting.
    let max = Plan::new(&[2], 3, 3, 4, PMode::HiLo).unwrap();
    assert!(max.working_bytes + 3 * HIDDEN * 2 + 2 * 4 * KV_WIDTH * 2 < 1 << 20);
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let q = upload(&client, &device, &[3, HIDDEN], |_| 0.0);
    let k = upload(&client, &device, &[4, KV_WIDTH], |i| if i / KV_WIDTH < 2 { f32::NAN } else { 0.0 });
    let v = upload(&client, &device, &[4, KV_WIDTH], |i| if i / KV_WIDTH < 2 { f32::NAN } else { 2.0 });
    for mode in [PMode::Bf16, PMode::HiLo] {
        for warps in [4, 8] {
            for masked in [false, true] {
                let run = KeyRun { k: &k.handle, v: &v.handle, k_scales: None, v_scales: None,
                    rows: 4, base: if masked { 10 } else { 0 }, lo: 2, hi: 4, row0: 0 };
                let plan = Plan::new(&[2], 3, if masked { 0 } else { 3 }, warps, mode).unwrap();
                let output = launch_planned(&client, &q.handle, &[run], 1.0 / 128.0f32.sqrt(), &plan).unwrap();
                fence(&client);
                let bytes = client.read_one(output).unwrap();
                let values: Vec<_> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                let expected = if masked { 0.0 } else { 2.0 };
                let wrong = values.iter().filter(|&&v| !v.is_finite() || v != expected).count();
                evidence.record("checked-range", json!({"mode":mode.as_str(),"warps":warps,
                    "lo":2,"fully_masked":masked,"elements":values.len(),"wrong":wrong}));
                assert_eq!(values.len(), 3 * HIDDEN);
                assert_eq!(wrong, 0, "uniform value/masking contract, not model capability");
            }
        }
    }
    fence(&client);
}

const FULL_CASES: [Case; 3] = [
    Case { name: "full-256x256", queries: 256, keys: 256, capacity: 272, q0: 0, cut: 0,
        uniform: false },
    Case { name: "full-256x16384-cut8192", queries: 256, keys: 16_384, capacity: 16_400,
        q0: 16_128, cut: 8_192, uniform: false },
    Case { name: "full-256x524288", queries: LONG_QUERIES, keys: LONG_KEYS,
        capacity: LONG_KEYS + TAIL, q0: LONG_KEYS - LONG_QUERIES, cut: 0, uniform: false },
];

#[test]
#[ignore = "finite owned GPU: mma attention vs scalar baseline at T256, record only, plus warm timing"]
fn gpu_mma_attention_full_query() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_full_query");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    contract(&mut evidence, &FULL_CASES);
    for case in FULL_CASES {
        assert!(!case.small());
        let inputs = Inputs::new(&client, &device, case);
        let (_, first) = run_case(&client, &inputs, case, &mut evidence);
        if case.keys == LONG_KEYS {
            // Warm, alternating order over the four arms: two orders x three
            // repeats, each launch its own host completion wall.
            let forward: Vec<_> = (0..ARMS.len()).collect();
            let backward: Vec<_> = forward.iter().rev().copied().collect();
            for (order_index, order) in [forward, backward].into_iter().enumerate() {
                for repeat in 0..3 {
                    for &a in &order {
                        let phase = format!("warm-order{order_index}-repeat{repeat}");
                        let got = run_arm(ARMS[a], case, &inputs, &mut evidence, &phase);
                        // Same inputs, same arm: a nonzero count is itself a
                        // finding (nondeterminism), recorded not asserted.
                        evidence.record("warm_repeat", json!({"case":case.name,"phase":phase,
                            "arm":ARMS[a].label(),
                            "differing_bits_vs_first":differing_bits(&got, &first[a])}));
                    }
                }
            }
        }
        drop(first);
        drop(inputs);
        fence(&client);
    }
    evidence.record("completed", json!({"passed":true,"first_launches":FULL_CASES.len() * ARMS.len(),
        "warm_launches":2 * 3 * ARMS.len()}));
    fs::write(evidence.root.join("complete.json"), b"{\"passed\":true}\n").unwrap();
}

/// The context sweep: T256 at the END of each context (q0 = keys - 256), so
/// every query sees almost all keys, as in a long prefill's late chunks.
const SWEEP_CASES: [Case; 4] = [
    Case { name: "sweep-256x4096", queries: 256, keys: 4_096, capacity: 4_096 + TAIL,
        q0: 4_096 - 256, cut: 0, uniform: false },
    Case { name: "sweep-256x16384", queries: 256, keys: 16_384, capacity: 16_384 + TAIL,
        q0: 16_384 - 256, cut: 0, uniform: false },
    Case { name: "sweep-256x65536", queries: 256, keys: 65_536, capacity: 65_536 + TAIL,
        q0: 65_536 - 256, cut: 0, uniform: false },
    Case { name: "sweep-256x131072", queries: 256, keys: 131_072, capacity: 131_072 + TAIL,
        q0: 131_072 - 256, cut: 0, uniform: false },
];

#[test]
#[ignore = "finite owned GPU: mma attention context sweep at T256, record only, warm timing per K"]
fn gpu_mma_attention_context_sweep() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_context_sweep");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    contract(&mut evidence, &SWEEP_CASES);
    for case in SWEEP_CASES {
        assert!(!case.small());
        let inputs = Inputs::new(&client, &device, case);
        let (_, first) = run_case(&client, &inputs, case, &mut evidence);
        // Warm, alternating order over the four arms: two orders x two
        // repeats (the record cap allows four contexts at this depth).
        let forward: Vec<_> = (0..ARMS.len()).collect();
        let backward: Vec<_> = forward.iter().rev().copied().collect();
        for (order_index, order) in [forward, backward].into_iter().enumerate() {
            for repeat in 0..2 {
                for &a in &order {
                    let phase = format!("warm-order{order_index}-repeat{repeat}");
                    let got = run_arm(ARMS[a], case, &inputs, &mut evidence, &phase);
                    evidence.record("warm_repeat", json!({"case":case.name,"phase":phase,
                        "arm":ARMS[a].label(),
                        "differing_bits_vs_first":differing_bits(&got, &first[a])}));
                }
            }
        }
        drop(first);
        drop(inputs);
        fence(&client);
    }
    evidence.record("completed", json!({"passed":true}));
    fs::write(evidence.root.join("complete.json"), b"{\"passed\":true}\n").unwrap();
}
