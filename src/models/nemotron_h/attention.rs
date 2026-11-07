//! Bounded native Nemotron attention mixer, not an assembled hybrid session.
//!
//! Normalized BF16 [1,T,4096] -> four separately scaled static FP8 projections
//! -> tensor-core online attention (BF16 Q/K/V, F32 accumulation) over plain
//! BF16 KV -> BF16 mixer output.
//! No RoPE, Q/K norm, gate, sink, outer norm or residual. F32 online reduction
//! is a declared native choice, not bitwise eager-BF16 attention parity.
//!
//! The owner privately binds uploaded weights, cache, device and provenance.
//! Enqueue stages only beyond the committed KV prefix. Commit fences device
//! work; any uncertain post-dispatch failure/cancellation requires reset.
//! A hybrid owner must stage ALL mixers before committing, and invalidate ALL
//! layer states if any commit/reset fails. Mamba state and the common attended
//! count must be reset together; this component cannot enforce that externally.

use super::fp8::{CudaTensor, Projection};
use super::attention_mma;
use crate::models::inkling::flash::KeyRun;
use burn::tensor::DType;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
    server::Handle,
};
use half::bf16;
use std::sync::Arc;
use triblespace::prelude::Id;

type Result<T> = std::result::Result<T, String>;
pub const HIDDEN: usize = 4096;
pub const HEADS: usize = 32;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 128;
pub const MAX_CHUNK: usize = 256;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const SHAPES: [[usize; 2]; 4] = [
    [HIDDEN, HIDDEN],
    [KV_WIDTH, HIDDEN],
    [KV_WIDTH, HIDDEN],
    [HIDDEN, HIDDEN],
];

/// The one attention algorithm this owner runs, as checkpoint provenance
/// records it: tensor-core MMA at NK64 with P split into BF16 hi + lo, at
/// every T including T1 decode. A fixed identity, not a choice. It differs
/// from every retired profile (`scalar-f32-v1`, `mma-bf16-p-v1`, and
/// `mma-hilo-p-v1`, whose T1 decode was scalar), so state recorded under any
/// of them is refused.
pub const EXECUTION: &str = "mma-hilo-p-t1-v1";

/// Query planes per cube: B4 for T1 (one live plane) and below live end
/// 65536, B8 above. The threshold is a measured-context heuristic, not a
/// proven crossover.
fn warps(tokens: usize, end: usize) -> usize {
    if tokens == 1 || end < 65536 { 4 } else { 8 }
}

/// Caller-selected provenance, not an assertion about an upstream checkpoint.
/// Bytes/scales must be selected together from one frozen native observation.
/// The runtime binding is this owner's immutable uploaded operands, not a hash
/// or equality check on these opaque graph IDs. Cache transfer is not exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceBinding {
    pub model_root: Id,
    pub layer: Id,
    pub operations: [Id; 4], // Q, K, V, O in that order
}

/// One already selected typed E4M3 operation. This is not a file/pile importer.
pub struct Operation<'a> {
    pub e4m3fn: &'a [u8],
    pub input_scale: f32,
    pub weight_scale: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Caller supplies the configured model limit; no implicit 1024-token cap.
    pub context_limit: usize,
    pub capacity: usize,
    /// Four packed projection weights plus two reserved BF16 KV buffers.
    pub max_resident_bytes: usize,
    /// Conservative sum of fresh projection/cast/online-attention allocations.
    pub max_working_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub weight_bytes: usize,
    pub kv_bytes: usize,
    pub resident_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Ready,
    Pending(usize),
    Poisoned,
}

struct Sequence {
    attended: usize,
    phase: Phase,
}
impl Default for Sequence {
    fn default() -> Self {
        Self {
            attended: 0,
            phase: Phase::Ready,
        }
    }
}
impl Sequence {
    fn ready(&self, start: usize) -> Result<()> {
        if self.phase != Phase::Ready {
            return Err("attention has pending work or is poisoned; commit/reset required".into());
        }
        if start != self.attended {
            return Err("attention absolute start differs from committed history".into());
        }
        Ok(())
    }
    fn commit(&mut self, fence: impl FnOnce() -> Result<()>) -> Result<()> {
        let Phase::Pending(end) = self.phase else {
            return Err("no committable attention work".into());
        };
        self.phase = Phase::Poisoned;
        fence()?;
        self.attended = end;
        self.phase = Phase::Ready;
        Ok(())
    }
    fn reset(&mut self, fence: impl FnOnce() -> Result<()>) -> Result<()> {
        self.phase = Phase::Poisoned;
        fence()?;
        *self = Self::default();
        Ok(())
    }
}

pub(super) struct Weights {
    source: SourceBinding,
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    projections: [Projection; 4],
}

pub struct Attention {
    pub(super) weights: Arc<Weights>,
    keys: Handle,
    values: Handle,
    private_rows: usize,
    prefix: Option<Arc<KvPrefix>>,
    limits: Limits,
    admission: Admission,
    sequence: Sequence,
}

/// Sealed contiguous BF16 K/V at a completed boundary. Never writable through
/// a Session. The exact uploaded projection owner is part of compatibility.
pub(super) struct KvPrefix {
    weights: Arc<Weights>,
    keys: Handle,
    values: Handle,
    rows: usize,
}

impl KvPrefix {
    /// Owning handles over one exact contiguous token range. The caller keeps
    /// this immutable prefix alive and reads on its fenced origin stream.
    pub(super) fn range(&self, start: usize, rows: usize) -> Result<[Handle; 2]> {
        let end = start.checked_add(rows).ok_or("KV export range overflow")?;
        if rows == 0 || end > self.rows { return Err("KV export outside prefix".into()); }
        let hi = checked_elements(&[end, KV_WIDTH], 2)? * 2;
        let lo = start * KV_WIDTH * 2; // bounded by checked hi
        let slice = |h: &Handle| -> Result<Handle> {
            if h.size_in_used() < hi as u64 { return Err("short KV backing".into()); }
            Ok(h.clone().offset_start(lo as u64).offset_end(h.size_in_used() - hi as u64))
        };
        Ok([slice(&self.keys)?, slice(&self.values)?])
    }

    /// The hybrid importer owns/fences both uploads before publishing Prefix.
    pub(super) fn imported(weights: Arc<Weights>, rows: usize, keys: Handle, values: Handle) -> Result<Arc<Self>> {
        let size = checked_elements(&[rows, KV_WIDTH], 2)? * 2;
        if keys.size_in_used() != size as u64 || values.size_in_used() != size as u64 {
            return Err("imported KV backing differs from prefix extent".into());
        }
        Ok(Arc::new(Self { weights, rows, keys, values }))
    }
}

pub struct Output {
    /// Queued BF16 [1,T,4096], before the caller's outer residual addition.
    pub values: CudaTensor,
    pub working_bytes: usize,
    #[cfg(test)]
    pub(super) stages: Stages,
}
#[cfg(test)]
pub(super) struct Stages {
    pub(super) q: CudaTensor,
    pub(super) k: CudaTensor,
    pub(super) v: CudaTensor,
    pub(super) attention: CudaTensor,
}

fn checked_elements(dims: &[usize], bytes: usize) -> Result<usize> {
    let n = dims
        .iter()
        .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
        .ok_or("empty/overflowing attention extent")?;
    if n.checked_mul(bytes).is_none_or(|v| v > u32::MAX as usize) {
        return Err("attention buffer exceeds bounded u32 byte extent".into());
    }
    Ok(n)
}

pub(super) fn admission(limits: Limits) -> Result<Admission> {
    if limits.capacity == 0 || limits.capacity > limits.context_limit {
        return Err("attention capacity must be positive and within explicit context limit".into());
    }
    let kv_bytes = checked_elements(&[limits.capacity, KV_WIDTH], 2)? * 4;
    let weight_bytes = SHAPES.iter().try_fold(0usize, |sum, s| -> Result<usize> {
        sum.checked_add(checked_elements(s, 1)?)
            .ok_or("weight byte sum overflow".into())
    })?;
    let resident_bytes = weight_bytes
        .checked_add(kv_bytes)
        .ok_or("resident byte sum overflow")?;
    if resident_bytes > limits.max_resident_bytes {
        return Err("attention exceeds logical resident budget".into());
    }
    Ok(Admission {
        weight_bytes,
        kv_bytes,
        resident_bytes,
    })
}

struct Plan {
    end: usize,
    tokens: usize,
    mma: attention_mma::Plan,
    working_bytes: usize,
}
#[cfg(test)]
fn plan(start: usize, tokens: usize, limits: Limits) -> Result<Plan> {
    plan_with_prefix(start, tokens, limits, 0)
}
fn plan_with_prefix(start: usize, tokens: usize, limits: Limits, prefix: usize) -> Result<Plan> {
    if !(1..=MAX_CHUNK).contains(&tokens) {
        return Err("attention requires B1 and T in 1..=256".into());
    }
    let end = start
        .checked_add(tokens)
        .ok_or("attention position overflow")?;
    if end > limits.capacity || end > u32::MAX as usize {
        return Err("attention append exceeds reserved capacity/position range".into());
    }
    if prefix > start { return Err("attention prefix exceeds committed position".into()); }
    // The launcher's actual per-run split rounding, including the extra
    // partial slot a second run can require. No full-capacity tail is counted.
    let spans = if prefix == 0 { vec![end] } else { vec![prefix, end - prefix] };
    let mma = attention_mma::Plan::new(&spans, tokens, start, warps(tokens, end))?;
    let mut working = mma.working_bytes;
    for [n, k] in SHAPES {
        // Peer Projection::working_bytes = FP8 input + F32 accumulator + BF16 output.
        let bytes = checked_elements(&[tokens, k], 2)?
            .checked_add(checked_elements(&[tokens, n], 4)? * 6)
            .ok_or("projection working sum overflow")?;
        working = working
            .checked_add(bytes)
            .ok_or("attention working sum overflow")?;
    }
    if working > limits.max_working_bytes {
        return Err("attention exceeds logical working budget".into());
    }
    Ok(Plan {
        end,
        tokens,
        mma,
        working_bytes: working,
    })
}

fn validate_operations(operations: &[Operation<'_>; 4]) -> Result<()> {
    for (op, shape) in operations.iter().zip(SHAPES) {
        let product = op.input_scale * op.weight_scale;
        if op.e4m3fn.len() != checked_elements(&shape, 1)?
            || !op.input_scale.is_finite()
            || op.input_scale <= 0.0
            || !op.weight_scale.is_finite()
            || op.weight_scale <= 0.0
            || !product.is_finite()
            || product <= 0.0
        {
            return Err("attention operation length/scales invalid".into());
        }
    }
    Ok(())
}

impl Attention {
    pub(super) fn planned_prefix_working_bytes(start: usize, tokens: usize, limits: Limits, prefix: usize) -> Result<usize> {
        Ok(plan_with_prefix(start, tokens, limits, prefix)?.working_bytes)
    }
    /// Upload each operation once onto ONE client/device and reserve KV once.
    /// All host geometry, scale and aggregate resident checks precede device
    /// allocation. Source finite-value semantics remain the loader's contract;
    /// no host dequantization/NaN replacement is hidden here.
    pub fn upload(
        device: CudaDevice,
        source: SourceBinding,
        operations: [Operation<'_>; 4],
        limits: Limits,
    ) -> Result<Self> {
        let admission = admission(limits)?;
        validate_operations(&operations)?;
        let [q, k, v, o] = operations;
        let upload = |op: Operation<'_>, shape: [usize; 2]| {
            Projection::upload(
                device.clone(),
                shape,
                op.e4m3fn,
                op.input_scale,
                op.weight_scale,
                shape[0] * shape[1],
            )
        };
        let projections = [
            upload(q, SHAPES[0])?,
            upload(k, SHAPES[1])?,
            upload(v, SHAPES[2])?,
            upload(o, SHAPES[3])?,
        ];
        let client = CudaRuntime::client(&device);
        let keys = client.empty(admission.kv_bytes / 2);
        let values = client.empty(admission.kv_bytes / 2);
        Ok(Self {
            weights: Arc::new(Weights { source, client, device, projections }),
            keys,
            values,
            private_rows: limits.capacity,
            prefix: None,
            limits,
            admission,
            sequence: Sequence::default(),
        })
    }
    /// Fresh KV and sequence state on the caller's current stream. The fixed
    /// projections remain bound to the exact uploaded owner.
    pub(super) fn from_weights(weights: Arc<Weights>, limits: Limits) -> Result<Self> {
        let admission = admission(limits)?;
        let keys = weights.client.empty(admission.kv_bytes / 2);
        let values = weights.client.empty(admission.kv_bytes / 2);
        Ok(Self { weights, keys, values, private_rows: limits.capacity, prefix: None,
            limits, admission, sequence: Sequence::default() })
    }
    /// Caller fences the whole hybrid before publication. Requiring a full
    /// seed buffer prevents accidentally retaining an unused full-C tail.
    pub(super) fn seal_kv(&self) -> Result<Arc<KvPrefix>> {
        if self.sequence.phase != Phase::Ready || self.prefix.is_some()
            || self.sequence.attended != self.private_rows || self.private_rows == 0 {
            return Err("attention prefix needs a completed exact-size seed".into());
        }
        Ok(Arc::new(KvPrefix { weights: self.weights.clone(), keys: self.keys.clone(),
            values: self.values.clone(), rows: self.private_rows }))
    }
    pub(super) fn from_prefix(weights: Arc<Weights>, prefix: Arc<KvPrefix>, limits: Limits) -> Result<Self> {
        let admission = admission(limits)?;
        if !Arc::ptr_eq(&weights, &prefix.weights) || prefix.rows > limits.capacity {
            return Err("attention prefix owner or capacity differs".into());
        }
        let private_rows = limits.capacity - prefix.rows;
        // At P=C no append is possible. These owning placeholders are never
        // passed as writable bindings; no zero-byte allocation is manufactured.
        let keys = if private_rows == 0 { prefix.keys.clone() }
            else { weights.client.empty(private_rows * KV_WIDTH * 2) };
        let values = if private_rows == 0 { prefix.values.clone() }
            else { weights.client.empty(private_rows * KV_WIDTH * 2) };
        let sequence = Sequence { attended: prefix.rows, phase: Phase::Ready };
        Ok(Self { weights, keys, values, private_rows, prefix: Some(prefix), limits, admission, sequence })
    }
    pub(super) fn prefix_len(&self) -> usize { self.prefix.as_ref().map_or(0, |p| p.rows) }
    pub(super) fn private_kv_bytes(&self) -> usize { self.private_rows * KV_WIDTH * 4 }
    pub(super) fn restore_prefix(&mut self) -> Result<()> {
        let position = self.prefix.as_ref().ok_or("attention has no saved prefix")?.rows;
        self.reset()?;
        self.sequence.attended = position;
        Ok(())
    }
    pub(super) fn reset_capacity(&mut self, capacity: usize) -> Result<()> {
        self.reset()?;
        let limits = Limits { capacity, ..self.limits };
        let admission = admission(limits)?;
        if self.prefix.is_some() || self.private_rows != capacity {
            // The old state is drained. Partial allocation failure/unwind must
            // not advertise a Ready attention owner with insufficient backing.
            self.sequence.phase = Phase::Poisoned;
            self.keys = self.weights.client.empty(admission.kv_bytes / 2);
            self.values = self.weights.client.empty(admission.kv_bytes / 2);
            self.private_rows = capacity;
            self.prefix = None;
            self.limits = limits;
            self.admission = admission;
            self.sequence = Sequence::default();
        }
        Ok(())
    }
    pub fn source(&self) -> SourceBinding {
        self.weights.source
    }
    pub fn admission(&self) -> Admission {
        self.admission
    }
    pub fn limits(&self) -> Limits {
        self.limits
    }
    pub fn attended(&self) -> usize {
        self.sequence.attended
    }
    pub fn has_pending(&self) -> bool {
        matches!(self.sequence.phase, Phase::Pending(_))
    }
    pub fn is_poisoned(&self) -> bool {
        self.sequence.phase == Phase::Poisoned
    }

    fn preflight(&self, input: &CudaTensor, expected_start: usize) -> Result<Plan> {
        self.sequence.ready(expected_start)?;
        let dims = input.meta.shape().as_slice();
        if dims.len() != 3
            || dims[0] != 1
            || dims[2] != HIDDEN
            || input.dtype != DType::BF16
            || input.qparams.is_some()
            || input.device != self.weights.device
            || !std::ptr::eq(input.client.properties(), self.weights.client.properties())
        {
            return Err(
                "attention needs contiguous BF16 [1,T,4096] on its owning client/device".into(),
            );
        }
        let p = plan_with_prefix(expected_start, dims[1], self.limits, self.prefix_len())?;
        let elements = checked_elements(dims, 2)?;
        let mut stride = 1usize;
        if input.meta.strides().len() != 3 {
            return Err("attention input strides have wrong rank".into());
        }
        for (axis, &d) in dims.iter().enumerate().rev() {
            if d > 1 && input.meta.strides()[axis] != stride {
                return Err("attention input is not contiguous".into());
            }
            stride *= d;
        }
        let available = input
            .handle
            .size()
            .checked_sub(input.handle.offset_end.unwrap_or(0));
        if input.handle.offset_start.unwrap_or(0) != 0
            || available.is_none_or(|n| n < (elements * 2) as u64)
        {
            return Err("attention input start offset or logical buffer span invalid".into());
        }
        // Recheck the peer's exact admitted sizes before any of its launches.
        for projection in &self.weights.projections {
            projection.working_bytes(p.tokens)?;
        }
        Ok(p)
    }

    /// No host read/sync. A successful return is QUEUED, not committed.
    /// Refusals before dispatch leave history intact. After dispatch, any Err
    /// or panic leaves this owner poisoned; never consume its cache again until
    /// reset succeeds. The caller also invalidates every other hybrid state.
    pub fn enqueue(&mut self, input: &CudaTensor, expected_start: usize) -> Result<Output> {
        let p = self.preflight(input, expected_start)?;
        self.sequence.phase = Phase::Poisoned;
        let q = self.weights.projections[0].project(input, self.limits.max_working_bytes)?;
        let k = self.weights.projections[1].project(input, self.limits.max_working_bytes)?;
        let v = self.weights.projections[2].project(input, self.limits.max_working_bytes)?;
        let n = p.tokens * KV_WIDTH;
        let cube = CubeDim::new_1d(128);
        unsafe {
            append_tail::launch_unchecked::<CudaRuntime>(
                &self.weights.client,
                cubecl::calculate_cube_count_elemwise(&self.weights.client, n, cube),
                cube,
                ArrayArg::from_raw_parts(k.handle.clone(), n),
                ArrayArg::from_raw_parts(v.handle.clone(), n),
                ArrayArg::from_raw_parts(self.keys.clone(), self.private_rows * KV_WIDTH),
                ArrayArg::from_raw_parts(self.values.clone(), self.private_rows * KV_WIDTH),
                n,
                (expected_start - self.prefix_len()) * KV_WIDTH,
            );
        }
        let mut runs = Vec::with_capacity(2);
        if let Some(prefix) = &self.prefix {
            runs.push(KeyRun { k: &prefix.keys, v: &prefix.values, k_scales: None, v_scales: None,
                rows: prefix.rows, base: 0, lo: 0, hi: prefix.rows, row0: 0 });
        }
        runs.push(KeyRun {
            k: &self.keys,
            v: &self.values,
            k_scales: None,
            v_scales: None,
            rows: self.private_rows,
            base: self.prefix_len(),
            lo: 0,
            hi: p.end - self.prefix_len(),
            row0: 0,
        });
        let handle = attention_mma::launch_planned(&self.weights.client, &q.handle, &runs,
            1.0 / (HEAD_DIM as f32).sqrt(), &p.mma)?;
        let attention = burn_cubecl::kernel::cast(
            CudaTensor::new_contiguous(
                self.weights.client.clone(),
                self.weights.device.clone(),
                [1, p.tokens, HIDDEN].into(),
                handle,
                DType::F32,
            ),
            DType::BF16,
        );
        let values = self.weights.projections[3].project(&attention, self.limits.max_working_bytes)?;
        self.sequence.phase = Phase::Pending(p.end);
        Ok(Output {
            values,
            working_bytes: p.working_bytes,
            #[cfg(test)]
            stages: Stages { q, k, v, attention },
        })
    }

    /// Fence the shared client before advancing this layer's committed length.
    /// This advances this layer's PRIVATE position, not a hybrid usable prefix.
    /// Short append commits after all layers; layer-major prefill may advance
    /// private tiles while Session stays unavailable. On any failure invalidate
    /// all owners and reset all state families together; no rollback is implied.
    pub fn commit(&mut self) -> Result<()> {
        let client = &self.weights.client;
        self.sequence.commit(|| {
            cubecl::future::block_on(client.sync())
                .map_err(|e| format!("attention completion: {e:?}"))
        })
    }
    /// Logical cancellation only, NOT cancellation of already queued GPU work.
    pub fn invalidate(&mut self) {
        self.sequence.phase = Phase::Poisoned;
    }
    /// Drain outstanding work, empty logical history, retain weights/capacity.
    /// Stale bytes are inaccessible behind hi=0 and overwritten on next append.
    /// Fatal device errors can prevent reset and require owner reconstruction.
    pub fn reset(&mut self) -> Result<()> {
        let client = &self.weights.client;
        self.sequence.reset(|| {
            cubecl::future::block_on(client.sync()).map_err(|e| format!("attention reset: {e:?}"))
        })
    }
    /// Test-only committed prefix readback; never exports uninitialized tail.
    #[cfg(test)]
    pub(super) fn diagnostic_kv(&self) -> Result<[Vec<u8>; 2]> {
        if self.sequence.phase != Phase::Ready {
            return Err("diagnostic KV read requires committed ready state".into());
        }
        let read = |h: &Handle, rows: usize| -> Result<Vec<u8>> {
            let bytes = rows * KV_WIDTH * 2;
            self.weights.client.read_one(h.clone()).map(|v| v[..bytes].to_vec())
                .map_err(|e| format!("diagnostic KV read: {e:?}"))
        };
        let mut result = if let Some(prefix) = &self.prefix {
            [read(&prefix.keys, prefix.rows)?, read(&prefix.values, prefix.rows)?]
        } else { [Vec::new(), Vec::new()] };
        let suffix = self.sequence.attended - self.prefix_len();
        if suffix > 0 {
            result[0].extend(read(&self.keys, suffix)?);
            result[1].extend(read(&self.values, suffix)?);
        }
        Ok(result)
    }
}

#[cube(launch_unchecked)]
fn append_tail(
    k: &Array<bf16>,
    v: &Array<bf16>,
    keys: &mut Array<bf16>,
    values: &mut Array<bf16>,
    n: usize,
    offset: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        keys[offset + i] = k[i];
        values[offset + i] = v[i];
    }
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
