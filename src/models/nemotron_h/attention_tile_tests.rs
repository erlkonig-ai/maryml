//! Finite synthetic diagnostic for the existing online kernel's row parameter.
//! Test-only: does not change native admission, state publication or cache ABI.
use super::*;
use std::{fs, io::Write, path::PathBuf, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use cubecl::client::ComputeClient;
use serde_json::{Value, json};

const LONG_KEYS: usize = 524_288;
const LONG_QUERIES: usize = 256;
const FULL_QUERY_KEYS: [usize; 3] = [256, 16_384, LONG_KEYS];
const TAIL: usize = 16;
const SEED: u64 = 0x5eeda77e;
const MAX_SECONDS: u64 = 250;
const ABS_LIMIT: f64 = 5e-5;
const SCALED_LIMIT: f64 = 1e-5;

// Ephemeral inputs/records of this one finite test, never a runtime cache.
#[derive(Clone, Copy)]
struct Case { queries: usize, keys: usize, capacity: usize, q0: usize }
impl Case {
    fn validate(self) {
        assert!((1..=256).contains(&self.queries));
        assert_eq!(self.q0 + self.queries, self.keys);
        assert!(self.keys <= self.capacity && self.capacity <= LONG_KEYS + TAIL);
    }
}

fn plan(case: Case, rows: usize, spans: &[usize]) -> Value {
    case.validate();
    assert!([32, 48].contains(&rows));
    assert!(flash::applies(HEADS, KV_HEADS, HEAD_DIM, rows));
    assert_eq!(spans.iter().sum::<usize>(), case.keys);
    let tiles = case.queries.div_ceil(rows / (HEADS / KV_HEADS));
    let splits = flash::key_run_splits(spans, tiles, KV_HEADS);
    let slots: usize = splits.iter().sum();
    let partial = slots * case.queries * HIDDEN * 4;
    let stats = slots * case.queries * HEADS * 2 * 4;
    let named = case.capacity * KV_WIDTH * 4 + case.queries * HIDDEN * 10 + partial + stats;
    assert!(named <= 600 << 20, "named allocation ceiling");
    json!({"rows":rows,"queries":case.queries,"q0":case.q0,"keys":case.keys,
        "capacity":case.capacity,"query_tiles":tiles,"splits":splits,
        "partial_bytes":partial,"stats_bytes":stats,"named_device_bytes":named,
        "shared_bytes_per_cube":flash::shared_floats(rows as u32, HEAD_DIM as u32)*4})
}

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

fn upload(client: &ComputeClient<CudaRuntime>, device: &CudaDevice,
    shape: &[usize], mut value: impl FnMut(usize) -> f32) -> CudaTensor
{
    let n: usize = shape.iter().product();
    assert!(n * 2 <= (256 << 20) + TAIL * KV_WIDTH * 2);
    let mut bytes = Vec::with_capacity(n * 2);
    for i in 0..n { bytes.extend_from_slice(&bf16::from_f32(value(i)).to_bits().to_le_bytes()); }
    let handle = client.create_from_slice(&bytes);
    // Retire this upload before generating the next large host input.
    fence(client);
    CudaTensor::new_contiguous(client.clone(), device.clone(), shape.into(), handle, DType::BF16)
}

fn key_buffer(client: &ComputeClient<CudaRuntime>, device: &CudaDevice,
    case: Case, start: usize, rows: usize, salt: u64) -> CudaTensor
{
    assert!(start + rows <= case.capacity);
    upload(client, device, &[rows, KV_WIDTH], |i| {
        let global = start * KV_WIDTH + i;
        if global / KV_WIDTH >= case.keys { f32::NAN } else { sample(global, salt) }
    })
}

fn queries(client: &ComputeClient<CudaRuntime>, device: &CudaDevice,
    case: Case, uniform: bool) -> (CudaTensor, CudaTensor)
{
    let q = upload(client, device, &[1, case.queries, HIDDEN], |i| {
        if uniform { 0.0 } else { sample(i, 1) }
    });
    let q32 = burn_cubecl::kernel::cast(q.clone(), DType::F32);
    fence(client);
    (q, q32)
}

fn host_counters() -> Value {
    match cubecl::cuda::host_operation_snapshot() {
        None => Value::Null,
        Some(s) => json!({"pid":s.pid,"unix_ms":s.unix_ms,"trace_ms":s.trace_ms,
            "operations":s.operations.iter().map(|o| json!({"operation":o.operation,
                "calls":o.calls_total,"slow_calls":o.slow_calls_total,
                "host_micros":o.host_micros_total,"requested_bytes":o.requested_bytes_total
            })).collect::<Vec<_>>()})
    }
}

struct Evidence { root: PathBuf, file: fs::File, started: Instant, records: usize }
impl Evidence {
    fn new() -> Self {
        assert_eq!(std::env::var("CUBECL_MEMORY_CONFIG").as_deref(), Ok("exclusive"));
        assert!(std::env::var_os("CUBECL_DEBUG_LOG").is_none());
        assert!(std::env::var_os("CUBECL_DEBUG_OPTION").is_none());
        let root = PathBuf::from(std::env::var("NEMOTRON_ATTENTION_ROWS_OUTPUT")
            .expect("explicit new fixture evidence directory"));
        assert!(root.is_absolute());
        fs::create_dir(&root).expect("new evidence directory, no overwrite");
        let file = fs::OpenOptions::new().create_new(true).write(true)
            .open(root.join("events.jsonl")).unwrap();
        Self { root, file, started: Instant::now(), records: 0 }
    }
    fn check(&self) { assert!(self.started.elapsed() < Duration::from_secs(MAX_SECONDS),
        "finite fixture exceeded elapsed budget; external timeout also required"); }
    fn record(&mut self, label: &str, data: Value) {
        self.check();
        self.records += 1;
        assert!(self.records <= 160);
        let record = json!({"label":label,"process_elapsed_ns":self.started.elapsed().as_nanos(),
            "unix_ns":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),"data":data});
        let encoded = serde_json::to_vec(&record).unwrap();
        assert!(encoded.len() <= 16_384);
        self.file.write_all(&encoded).unwrap(); self.file.write_all(b"\n").unwrap();
        self.file.flush().unwrap();
    }
}

fn launch(case: Case, rows: usize, q: &CudaTensor, runs: &[KeyRun<'_>],
    evidence: &mut Evidence, label: &str) -> Vec<f32>
{
    let spans: Vec<_> = runs.iter().map(|r| r.hi-r.lo).collect();
    let inventory = plan(case, rows, &spans);
    let client = &q.client;
    fence(client);
    evidence.record("launch_begin", json!({"phase":label,"plan":inventory,"host":host_counters()}));
    let started = Instant::now();
    let out = flash::flash_attention_launch(client, &q.handle, runs, &q.handle,
        KvElem::Bf16, case.queries, case.q0, HEADS, KV_HEADS, HEAD_DIM, 0,
        None, 1.0 / (HEAD_DIM as f32).sqrt(), rows, None);
    let enqueue_ns = started.elapsed().as_nanos();
    let waiting = Instant::now();
    fence(client);
    let fence_ns = waiting.elapsed().as_nanos();
    let completed_ns = started.elapsed().as_nanos();
    // Reading and persisting results are outside the operation wall bracket.
    let bytes = client.read_one(out).unwrap();
    let n = case.queries * HIDDEN;
    assert!(bytes.len() >= n*4);
    let values: Vec<f32> = bytes[..n*4].chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(values.iter().all(|v| v.is_finite()), "nonfinite online output");
    evidence.record("launch_completed", json!({"phase":label,"rows":rows,
        "host_enqueue_ns":enqueue_ns,"host_fence_ns":fence_ns,
        "host_completed_ns":completed_ns,"gpu_duration_ns":Value::Null,
        "output_blake3":blake3::hash(&bytes[..n*4]).to_hex().to_string(),
        "host":host_counters()}));
    values
}

// Independent two-pass f64 reference on the exact rounded BF16 operands.
// No shared tile, key splitting, online recurrence or kernel mask arithmetic.
fn oracle(case: Case, uniform: bool) -> Vec<f64> {
    assert!(case.keys <= 281 && case.queries <= 17);
    let q: Vec<_> = (0..case.queries*HIDDEN)
        .map(|i| if uniform { 0.0 } else { f64::from(sample(i,1)) }).collect();
    let k: Vec<_> = (0..case.keys*KV_WIDTH).map(|i| f64::from(sample(i,2))).collect();
    let v: Vec<_> = (0..case.keys*KV_WIDTH).map(|i| f64::from(sample(i,3))).collect();
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let mut out = vec![0.0; case.queries*HIDDEN];
    for qi in 0..case.queries {
        for h in 0..HEADS {
            let kvh = h / (HEADS / KV_HEADS);
            let qo = qi*HIDDEN+h*HEAD_DIM;
            let scores: Vec<_> = (0..=case.q0+qi).map(|j| {
                let ko = j*KV_WIDTH+kvh*HEAD_DIM;
                (0..HEAD_DIM).map(|d|q[qo+d]*k[ko+d]).sum::<f64>()*scale
            }).collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores.iter().map(|s|(s-max).exp()).sum();
            for (j, score) in scores.into_iter().enumerate() {
                let p = (score-max).exp()/denom;
                for d in 0..HEAD_DIM { out[qo+d] += p*v[j*KV_WIDTH+kvh*HEAD_DIM+d]; }
            }
        }
    }
    out
}

fn compare(got: &[f32], expected: &[f64]) -> Value {
    assert_eq!(got.len(), expected.len());
    let (mut absolute, mut scaled) = (0.0f64, 0.0f64);
    for (&a,&b) in got.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        let e = (f64::from(a)-b).abs();
        absolute=absolute.max(e); scaled=scaled.max(e/b.abs().max(1.0));
    }
    assert!(absolute <= ABS_LIMIT && scaled <= SCALED_LIMIT,
        "fixed reference bound exceeded: abs={absolute:e}, scaled={scaled:e}");
    json!({"elements":got.len(),"max_abs":absolute,"max_scaled_floor1":scaled,
        "absolute_limit":ABS_LIMIT,"scaled_limit":SCALED_LIMIT})
}

#[test]
fn native_rows32_48_plan_and_oracle_contract() {
    let case = Case { queries:LONG_QUERIES,keys:LONG_KEYS,capacity:LONG_KEYS+TAIL,q0:LONG_KEYS-LONG_QUERIES };
    let a=plan(case,32,&[case.keys]); let b=plan(case,48,&[case.keys]);
    assert_eq!(a["query_tiles"],128); assert_eq!(b["query_tiles"],86);
    assert_eq!(a["splits"],json!([4])); assert_eq!(b["splits"],json!([5]));
    assert_eq!(a["partial_bytes"],16<<20); assert_eq!(b["partial_bytes"],20<<20);
    assert_eq!(a["stats_bytes"],262144); assert_eq!(b["stats_bytes"],327680);
    assert_eq!(a["shared_bytes_per_cube"],36992); assert_eq!(b["shared_bytes_per_cube"],47232);
    for keys in FULL_QUERY_KEYS {
        let c = Case { queries:LONG_QUERIES, keys, capacity:keys+TAIL, q0:keys-LONG_QUERIES };
        assert_eq!(native_plan(c).rows, 48);
        for rows in [32, 48] { plan(c, rows, &[keys]); }
    }
    assert!(!flash::applies(HEADS,KV_HEADS,HEAD_DIM,64)); // current policy, not hardware limit
    let small=Case{queries:2,keys:4,capacity:4,q0:2};
    let out=oracle(small,true);
    for qi in 0..2 { for h in 0..HEADS { for d in 0..HEAD_DIM {
        let kvh=h/(HEADS/KV_HEADS);
        let mean=(0..=small.q0+qi).map(|j|f64::from(sample(j*KV_WIDTH+kvh*HEAD_DIM+d,3)))
            .sum::<f64>()/(small.q0+qi+1) as f64;
        assert!((out[qi*HIDDEN+h*HEAD_DIM+d]-mean).abs()<1e-15);
    }}}
}

fn native_plan(case: Case) -> super::Plan {
    super::plan_with_prefix(case.q0, case.queries, Limits {
        capacity: case.capacity, context_limit: case.capacity,
        max_resident_bytes: usize::MAX, max_working_bytes: usize::MAX,
    }, 0).unwrap()
}

fn full_query_case(client: &ComputeClient<CudaRuntime>, device: &CudaDevice,
    keys: usize, evidence: &mut Evidence)
{
    let case=Case{queries:LONG_QUERIES,keys,capacity:keys+TAIL,q0:keys-LONG_QUERIES};
    // The candidate arm consumes the production planner's selection. The
    // retained ROW32 arm is an explicit comparison, not a second selector.
    let selected=native_plan(case);
    assert_eq!(selected.rows,48);
    evidence.record("native_plan",json!({"keys":keys,"rows":selected.rows,
        "working_bytes":selected.working_bytes,"end":selected.end}));
    let (_q,q)=queries(client,device,case,false);
    let k=key_buffer(client,device,case,0,case.capacity,2);
    let v=key_buffer(client,device,case,0,case.capacity,3);
    let run=KeyRun{k:&k.handle,v:&v.handle,k_scales:None,v_scales:None,
        rows:case.capacity,base:0,lo:0,hi:case.keys,row0:0};
    let baseline=launch(case,32,&q,&[run],evidence,&format!("full-keys{keys}-first-rows32"));
    let expected:Vec<f64>=baseline.iter().map(|&v|f64::from(v)).collect();
    let candidate=launch(case,selected.rows,&q,&[run],evidence,&format!("full-keys{keys}-first-rows48"));
    evidence.record("full_first_comparison",json!({"keys":keys,"comparison":compare(&candidate,&expected)}));
    drop(candidate); drop(baseline);
    for (order_index,order) in [[32,selected.rows],[selected.rows,32]].into_iter().enumerate() {
        for repeat in 0..3 { for rows in order {
            let label=format!("full-keys{keys}-warm-order{order_index}-repeat{repeat}-rows{rows}");
            let got=launch(case,rows,&q,&[run],evidence,&label);
            evidence.record("full_equivalence",json!({"phase":label,"comparison":compare(&got,&expected)}));
        }}
    }
    fence(client);
}

#[test]
#[ignore = "finite owned GPU: synthetic 32/2 GQA rows32/48, no model or pile"]
fn gpu_native_rows32_48_diagnostic() {
    let mut evidence=Evidence::new();
    let device=CudaDevice{index:0}; let client=CudaRuntime::client(&device);
    evidence.record("contract",json!({"seed":SEED,"rows":[32,48],"heads":HEADS,
        "kv_heads":KV_HEADS,"head_dim":HEAD_DIM,"max_seconds":MAX_SECONDS,"full_query_keys":FULL_QUERY_KEYS,
        "timing":"host enqueue plus existing completion fence; NOT CUDA-event duration"}));
    let case=Case{queries:17,keys:281,capacity:288,q0:264};
    for uniform in [true,false] {
        let (_q,q)=queries(&client,&device,case,uniform);
        let expected=oracle(case,uniform);
        for cut in [0,127] {
            let spans=if cut==0 {vec![(0,case.capacity)]} else {vec![(0,cut),(cut,case.capacity-cut)]};
            let owners:Vec<_>=spans.iter().map(|&(start,rows)| (
                key_buffer(&client,&device,case,start,rows,2),
                key_buffer(&client,&device,case,start,rows,3))).collect();
            let runs:Vec<_>=owners.iter().zip(&spans).map(|((k,v),&(base,rows))|KeyRun {
                k:&k.handle,v:&v.handle,k_scales:None,v_scales:None,rows,base,
                lo:0,hi:rows.min(case.keys-base),row0:0,
            }).collect();
            for rows in [32,48] {
                let label=format!("small-uniform{uniform}-cut{cut}-rows{rows}");
                let got=launch(case,rows,&q,&runs,&mut evidence,&label);
                evidence.record("small_oracle",json!({"phase":label,"comparison":compare(&got,&expected)}));
            }
        }
    }
    fence(&client);
    for keys in FULL_QUERY_KEYS { full_query_case(&client,&device,keys,&mut evidence); }
    fence(&client);
    evidence.record("completed",json!({"passed":true,"full_timed_launches":36,
        "first_full_launches":6,"small_launches":8}));
    fs::write(evidence.root.join("complete.json"),b"{\"passed\":true}\n").unwrap();
}
