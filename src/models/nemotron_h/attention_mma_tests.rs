//! Finite synthetic fixtures for the tensor-core attention kernel.
//! Test-only: no model, no pile, no runtime policy, admission or cache ABI.
//!
//! Hard invariants only, on immutable synthetic BF16 Q and K/V runs at B4 and
//! B8, T1 decode included: every output finite, bitwise determinism across
//! launches, causal masking (a key changes exactly the queries that may see
//! it, bitwise, and a key past every query changes nothing), padded planes
//! and the NaN tail never read or written (`gpu_mma_attention_partial_canary`),
//! and exact masked/boundary values. Model quality is judged by capability,
//! not by numerical agreement with another implementation.
//!
//! Each masking/padding fixture also carries a host replay of the kernel's own
//! split and causal trimming (`coverage`), and asserts that the property it is
//! named for actually occurs in the launch, so a fixture cannot silently stop
//! exercising its case.
use super::*;
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
/// The plane counts the owner plans: B4 (T1, and below live end 65536) and B8.
const WARPS: [usize; 2] = [4, 8];

#[test]
fn mma_host_checked_plan_sizes_and_rejections() {
    for (warps, slots) in [(4, 8), (8, 16)] {
        let p = Plan::new(&[524288], 256, 524032, warps).unwrap();
        assert_eq!(p.split_of, [slots]);
        assert_eq!(p.po_elems * 4, slots * 256 * 4096 * 4);
        assert_eq!(p.pml_elems * 4, slots * 256 * 32 * 2 * 4);
        assert_eq!(p.working_bytes, (p.po_elems + p.pml_elems) * 4 + 256 * 4096 * 6);
        assert!(p.validate_splits(&[0]).is_err());
        assert!(p.validate_splits(&[]).is_err());
        assert!(p.validate_splits(&[slots + 1]).is_err());
    }
    for (spans, nq, q0, warps) in [
        (vec![], 1, 0, 4), (vec![1], 0, 0, 4), (vec![1], 257, 0, 4),
        (vec![1], 1, 0, 0), (vec![1], 1, 0, 16),
        (vec![1], 1, u32::MAX as usize - 1, 4),
        (vec![usize::MAX, 1], 1, 0, 4),
    ] {
        assert!(Plan::new(&spans, nq, q0, warps).is_err());
    }
    assert!(extent(&[usize::MAX, 2], 4).is_err());
    assert!(extent(&[u32::MAX as usize], 4).is_err());
    assert_eq!(Plan::new(&[0], 3, 0, 4).unwrap().split_of, [1]);
    assert_eq!(shared_bytes(), 35840);
}

#[test]
fn mma_host_prefix_rounding_earlier_endpoint_exceeds_final_width_envelope() {
    let earlier = Plan::new(&[8192, 130816 - 8192], 256, 130560, 8).unwrap();
    assert_eq!(earlier.split_of, [2, 15]);
    assert_eq!(earlier.nq * earlier.slots, 4352);
    let final_max = (2..=256).map(|t| {
        let p = Plan::new(&[8192, 131072 - 8192], t, 131072 - t, 8).unwrap();
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
    fn json(self) -> Value {
        json!({"name":self.name,"queries":self.queries,"keys":self.keys,"capacity":self.capacity,
            "q0":self.q0,"cut":self.cut,"uniform":self.uniform,"runs":self.runs()})
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
    changed: Option<usize>,
) -> CudaTensor {
    assert!(start + rows <= case.capacity);
    upload(client, device, &[rows, KV_WIDTH], |i| {
        let global = start * KV_WIDTH + i;
        let row = global / KV_WIDTH;
        if row >= case.keys { f32::NAN }
        else if changed == Some(row) { sample(global, salt ^ 0x5a5a) }
        else { sample(global, salt) }
    })
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
    /// `changed`: one absolute key row whose K and V take different finite values.
    fn new(client: &ComputeClient<CudaRuntime>, device: &CudaDevice, case: Case,
        changed: Option<usize>) -> Self
    {
        case.validate();
        let q = upload(client, device, &[1, case.queries, HIDDEN], |i| {
            if case.uniform { 0.0 } else { sample(i, 1) }
        });
        let owners = case
            .spans()
            .into_iter()
            .map(|(start, rows)| {
                (key_buffer(client, device, case, start, rows, 2, changed),
                    key_buffer(client, device, case, start, rows, 3, changed))
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

fn contract(evidence: &mut Evidence, cases: &[Case]) {
    evidence.record("contract", json!({"seed":SEED,"heads":HEADS,"kv_heads":KV_HEADS,
        "head_dim":HEAD_DIM,"nk":NK,"max_seconds":MAX_SECONDS,"warps":WARPS,
        "mma_shared_bytes_per_cube":shared_bytes(),
        "shared_limit_bytes":SHARED_LIMIT_BYTES,
        "v_staging":V_STAGING,
        "cases":cases.iter().map(|c| c.json()).collect::<Vec<_>>()}));
}

/// One complete launch, fenced, read back; every output must be finite.
fn launch(client: &ComputeClient<CudaRuntime>, inputs: &Inputs, case: Case, warps: usize) -> Vec<f32> {
    let runs = inputs.runs(case);
    let out = mma_attention_launch(client, &inputs.q.handle, &runs, case.queries, case.q0,
        1.0 / (HEAD_DIM as f32).sqrt(), warps);
    fence(client);
    let bytes = client.read_one(out).unwrap();
    let n = case.queries * HIDDEN;
    assert!(bytes.len() >= n * 4);
    let values: Vec<f32> =
        bytes[..n * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(values.iter().all(|v| v.is_finite()), "nonfinite output: {} B{warps}", case.name);
    values
}

// ------------------------------------------------------------- fixtures --

const CASES: [Case; 13] = [
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
    // T1 decode: one live plane per cube, 512 key splits at most. The last
    // two over one long run and over a resident-shaped prefix plus a 16-row
    // private run; the first stops well before its run's later keys.
    Case { name: "decode-early-1x281", queries: 1, keys: 281, capacity: 288, q0: 200, cut: 0,
        uniform: false },
    Case { name: "decode-1x16384", queries: 1, keys: 16_384, capacity: 16_400, q0: 16_383,
        cut: 0, uniform: false },
    Case { name: "decode-1x16384-cut16368", queries: 1, keys: 16_384, capacity: 16_400,
        q0: 16_383, cut: 16_368, uniform: false },
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
#[ignore = "finite owned GPU: mma attention finite, deterministic and causal at T1 and T>1, B4 and B8"]
fn gpu_mma_attention_invariants() {
    let _turn = gpu_turn();
    let mut evidence = Evidence::new("mma_invariants");
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    contract(&mut evidence, &CASES);
    let mut checks = 0usize;
    for case in CASES {
        let coverage: Vec<_> = WARPS.map(|w| coverage(case, w)).into_iter().collect();
        evidence.record("coverage", json!({"case":case.name,"b4":coverage[0],"b8":coverage[1]}));
        for (w, cov) in WARPS.into_iter().zip(&coverage) {
            assert_coverage(case, w, cov);
        }
        let inputs = Inputs::new(&client, &device, case, None);
        // The last query's own key is visible to it alone; the next key, when
        // the run has one, to none of them.
        let own = case.q0 + case.queries - 1;
        let own_changed = Inputs::new(&client, &device, case, Some(own));
        let future = (own + 1 < case.keys).then(|| Inputs::new(&client, &device, case, Some(own + 1)));
        let last = (case.queries - 1) * HIDDEN;
        for warps in WARPS {
            let first = launch(&client, &inputs, case, warps);
            for _ in 0..2 {
                assert_eq!(differing_bits(&launch(&client, &inputs, case, warps), &first), 0,
                    "nondeterministic: {} B{warps}", case.name);
            }
            let changed = launch(&client, &own_changed, case, warps);
            assert_eq!(differing_bits(&changed[..last], &first[..last]), 0,
                "a key changed a query that cannot see it: {} B{warps}", case.name);
            let own_bits = differing_bits(&changed[last..], &first[last..]);
            assert!(own_bits > 0, "a visible key changed nothing: {} B{warps}", case.name);
            let future_bits = future.as_ref().map(|f| differing_bits(&launch(&client, f, case, warps), &first));
            assert!(future_bits.is_none_or(|n| n == 0), "a future key leaked: {} B{warps}", case.name);
            evidence.record("invariants", json!({"case":case.name,"warps":warps,
                "deterministic_launches":3,"own_key_differing":own_bits,"future_key_differing":future_bits}));
            checks += 1;
        }
        drop((inputs, own_changed, future));
        fence(&client);
    }
    evidence.record("completed", json!({"passed":true,"checks":checks}));
    fs::write(evidence.root.join("complete.json"), b"{\"passed\":true}\n").unwrap();
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
    mma_attention_partials(client, &inputs.q.handle, &runs, nq, case.q0, scaling, warps,
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
    let label = format!("mma-hilo-b{warps}");
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
    contract(&mut evidence, &CASES);
    let mut checks = 0usize;
    for case in CASES {
        let inputs = Inputs::new(&client, &device, case, None);
        for warps in WARPS {
            partial_canary(&client, &inputs, case, warps, &mut evidence);
            checks += 1;
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
    let max = Plan::new(&[2], 3, 3, 4).unwrap();
    assert!(max.working_bytes + 3 * HIDDEN * 2 + 2 * 4 * KV_WIDTH * 2 < 1 << 20);
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let q = upload(&client, &device, &[3, HIDDEN], |_| 0.0);
    let k = upload(&client, &device, &[4, KV_WIDTH], |i| if i / KV_WIDTH < 2 { f32::NAN } else { 0.0 });
    let v = upload(&client, &device, &[4, KV_WIDTH], |i| if i / KV_WIDTH < 2 { f32::NAN } else { 2.0 });
    for tokens in [1, 3] {
        for warps in WARPS {
            for masked in [false, true] {
                let run = KeyRun { k: &k.handle, v: &v.handle, k_scales: None, v_scales: None,
                    rows: 4, base: if masked { 10 } else { 0 }, lo: 2, hi: 4, row0: 0 };
                let plan = Plan::new(&[2], tokens, if masked { 0 } else { 3 }, warps).unwrap();
                let output = launch_planned(&client, &q.handle, &[run], 1.0 / 128.0f32.sqrt(), &plan).unwrap();
                fence(&client);
                let bytes = client.read_one(output).unwrap();
                let values: Vec<_> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                let expected = if masked { 0.0 } else { 2.0 };
                let wrong = values.iter().filter(|&&v| !v.is_finite() || v != expected).count();
                evidence.record("checked-range", json!({"tokens":tokens,"warps":warps,
                    "lo":2,"fully_masked":masked,"elements":values.len(),"wrong":wrong}));
                assert_eq!(values.len(), tokens * HIDDEN);
                assert_eq!(wrong, 0, "uniform value/masking contract, not model capability");
            }
        }
    }
    fence(&client);
}
