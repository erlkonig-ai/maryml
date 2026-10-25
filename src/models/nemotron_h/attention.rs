//! Bounded native Nemotron attention mixer, not an assembled hybrid session.
//!
//! Normalized BF16 [1,T,4096] -> four separately scaled static FP8 projections
//! -> existing F32 online attention over plain BF16 KV -> BF16 mixer output.
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
use crate::models::inkling::flash::{self, KeyRun, KvElem};
use burn::tensor::DType;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
    server::Handle,
};
use half::bf16;
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

pub struct Attention {
    source: SourceBinding,
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    projections: [Projection; 4],
    keys: Handle,
    values: Handle,
    limits: Limits,
    admission: Admission,
    sequence: Sequence,
}

pub struct Output {
    /// Queued BF16 [1,T,4096], before the caller's outer residual addition.
    pub values: CudaTensor,
    pub working_bytes: usize,
    #[cfg(test)]
    stages: Stages,
}
#[cfg(test)]
struct Stages {
    q: CudaTensor,
    k: CudaTensor,
    v: CudaTensor,
    attention: CudaTensor,
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

fn admission(limits: Limits) -> Result<Admission> {
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
    rows: usize,
    working_bytes: usize,
}
fn plan(start: usize, tokens: usize, limits: Limits) -> Result<Plan> {
    if !(1..=MAX_CHUNK).contains(&tokens) {
        return Err("attention requires B1 and T in 1..=256".into());
    }
    let end = start
        .checked_add(tokens)
        .ok_or("attention position overflow")?;
    if end > limits.capacity || end > u32::MAX as usize {
        return Err("attention append exceeds reserved capacity/position range".into());
    }
    let rows = if tokens == 1 {
        flash::decode_rows(16)
    } else {
        flash::prefill_rows(16)
    };
    if !flash::applies(HEADS, KV_HEADS, HEAD_DIM, rows) {
        return Err("unsupported online attention tile".into());
    }
    // Same single-run split calculation as unchanged flash.rs:1183-1202.
    // Both constants are properties of that launcher, not a new policy.
    let q_tiles = tokens.div_ceil(rows / 16);
    let slots = (1024 / (q_tiles * KV_HEADS))
        .max(1)
        .min(end.div_ceil(128).max(1));
    let partial = checked_elements(&[slots, tokens, HEADS, HEAD_DIM], 4)? * 4;
    let stats = checked_elements(&[slots, tokens, HEADS, 2], 4)? * 4;
    let q = checked_elements(&[tokens, HIDDEN], 4)?;
    // Q cast, flash output, BF16 flash result, plus partials/statistics.
    let mut working = partial
        .checked_add(stats)
        .and_then(|n| n.checked_add(q * 10))
        .ok_or("attention scratch sum overflow")?;
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
        rows,
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
            source,
            client,
            device,
            projections,
            keys,
            values,
            limits,
            admission,
            sequence: Sequence::default(),
        })
    }
    pub fn source(&self) -> SourceBinding {
        self.source
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
            || input.device != self.device
            || !std::ptr::eq(input.client.properties(), self.client.properties())
        {
            return Err(
                "attention needs contiguous BF16 [1,T,4096] on its owning client/device".into(),
            );
        }
        let p = plan(expected_start, dims[1], self.limits)?;
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
        for projection in &self.projections {
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
        let q = self.projections[0].project(input, self.limits.max_working_bytes)?;
        let k = self.projections[1].project(input, self.limits.max_working_bytes)?;
        let v = self.projections[2].project(input, self.limits.max_working_bytes)?;
        let n = p.tokens * KV_WIDTH;
        let cube = CubeDim::new_1d(128);
        unsafe {
            append_tail::launch_unchecked::<CudaRuntime>(
                &self.client,
                cubecl::calculate_cube_count_elemwise(&self.client, n, cube),
                cube,
                ArrayArg::from_raw_parts(k.handle.clone(), n),
                ArrayArg::from_raw_parts(v.handle.clone(), n),
                ArrayArg::from_raw_parts(self.keys.clone(), self.limits.capacity * KV_WIDTH),
                ArrayArg::from_raw_parts(self.values.clone(), self.limits.capacity * KV_WIDTH),
                n,
                expected_start * KV_WIDTH,
            );
        }
        let q_f32 = burn_cubecl::kernel::cast(q.clone(), DType::F32);
        let run = KeyRun {
            k: &self.keys,
            v: &self.values,
            k_scales: None,
            v_scales: None,
            rows: self.limits.capacity,
            base: 0,
            lo: 0,
            hi: p.end,
            row0: 0,
        };
        let handle = flash::flash_attention_launch(
            &self.client,
            &q_f32.handle,
            &[run],
            &q_f32.handle,
            KvElem::Bf16,
            p.tokens,
            expected_start,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            0,
            None,
            1.0 / (HEAD_DIM as f32).sqrt(),
            p.rows,
            None,
        );
        let attention = burn_cubecl::kernel::cast(
            CudaTensor::new_contiguous(
                self.client.clone(),
                self.device.clone(),
                [1, p.tokens, HIDDEN].into(),
                handle,
                DType::F32,
            ),
            DType::BF16,
        );
        let values = self.projections[3].project(&attention, self.limits.max_working_bytes)?;
        self.sequence.phase = Phase::Pending(p.end);
        Ok(Output {
            values,
            working_bytes: p.working_bytes,
            #[cfg(test)]
            stages: Stages { q, k, v, attention },
        })
    }

    /// Fence the shared client before advancing this layer's committed length.
    /// Hybrid owners stage all work first; do not commit one layer during a
    /// partially completed whole-network forward. If any layer commit fails,
    /// invalidate all owners and reset all state families together.
    pub fn commit(&mut self) -> Result<()> {
        let client = &self.client;
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
        let client = &self.client;
        self.sequence.reset(|| {
            cubecl::future::block_on(client.sync()).map_err(|e| format!("attention reset: {e:?}"))
        })
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
