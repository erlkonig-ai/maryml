//! Native Super MoE mixer, before the outer residual, without state/history.
//!
//! BF16 H4096 -> F32 router; BF16 latent1024 -> 22/512 NVFP4 ReLU2
//! experts -> F32 weighted sum -> BF16 -> BF16 latent expansion. Add the
//! independent full-width shared NVFP4 ReLU2 branch. No latent norm/gate.
//! Router BF16 operands use existing BF16 MMA with F32 accumulation/output:
//! this is a reduction variant of HF's widened F32 dot, NOT bit parity.
//! All arithmetic stays on GPU. Small routing metadata alone returns to host
//! to select immutable native expert operands; no retained parameter catalog.
//!
//! Each selected expert pair is fenced before release, as is the final result.
//! Native sessions may retain immutable pairs within their admitted budget;
//! standalone upload keeps the original one-visit lifetime.
//! After dispatch any error/panic poisons this owner until reset succeeds.
//! An outer hybrid owner must also discard/reset attention AND Mamba history;
//! resetting this stateless mixer alone cannot undo another layer's effects.

#[cfg(feature = "nemotron-timing")]
use super::host_timing::{self, Phase, Timings};
use crate::{
    models::inkling::{bf16gemm, moegroup},
    nn::{nvfp4_gemm::calibrated_scale, nvfp4_gemv::Nvfp4},
};
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
    server::Handle,
};
use half::bf16;
use triblespace::prelude::Id;

#[path = "moe_cache.rs"]
mod cache;

pub type CudaTensor = CubeTensor<CudaRuntime>;
pub type Result<T> = std::result::Result<T, String>;
pub const HIDDEN: usize = 4096;
pub const LATENT: usize = 1024;
pub const INTERMEDIATE: usize = 2688;
pub const SHARED: usize = 5376;
pub const EXPERTS: usize = 512;
pub const TOP_K: usize = 22;
pub const MAX_CHUNK: usize = 256;
const ROUTE_WIDTH: usize = TOP_K * 2 + 3; // IDs, weights, two cutoff scores, error
const THREADS: u32 = 256;

/// Caller-selected provenance, not proof that arbitrary bytes are upstream
/// weights. Fixed operands and the owned provider must use ONE held immutable
/// ModelSnapshot. IDs do not replace that observation/descriptor lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceBinding {
    pub model_root: Id,
    pub layer: Id,
}

/// Already selected typed leaf payload: E2M1 codes || E4M3FN scales || F32
/// weight global. Input global belongs to this same operation, not its peer.
pub struct PackedOperation<'a> {
    pub operation: Id,
    pub payload: &'a [u8],
    pub input_scale: f32,
}
pub struct FixedOperands<'a> {
    /// Router, correction bias, fc1, fc2 in that order.
    pub operations: [Id; 4],
    pub router_bf16: &'a [u8],
    pub correction_bias: &'a [f32],
    pub fc1_bf16: &'a [u8],
    pub fc2_bf16: &'a [u8],
    pub shared_up: PackedOperation<'a>,
    pub shared_down: PackedOperation<'a>,
}
pub struct ExpertPair<'a> {
    pub index: usize,
    pub up: PackedOperation<'a>,
    pub down: PackedOperation<'a>,
}

/// Narrow on-demand native operand boundary. Own (or borrow for this owner's
/// lifetime) the SAME frozen snapshot used to select FixedOperands. Query the
/// selected EXPERT child and its joined operation/scale, not a loaded catalog.
/// No provider replacement or runtime source rebinding is exposed by Moe.
pub trait ExpertProvider {
    fn source(&self) -> SourceBinding;
    /// The selected views need only live for this callback. A native provider
    /// can query one immutable snapshot here without retaining an expert bank.
    fn with_expert<T>(
        &self,
        index: usize,
        use_pair: impl FnOnce(ExpertPair<'_>) -> Result<T>,
    ) -> Result<T>;
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_fixed_bytes: usize,
    pub max_expert_bytes: usize,
    /// Conservative sum of logical temporary allocations for ONE call, with
    /// one expert live. Excludes caller input, host staging, allocator/JIT and
    /// runtime internals. This is not a whole-device or cgroup memory bound.
    pub max_working_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub fixed_bytes: usize,
    pub expert_bytes: usize,
}

/// The tiny readback needed for scheduling, not an activation/weight export.
/// Lowest expert ID wins exact biased-score ties. HF sorted=False topk does
/// not guarantee this ordering/tie rule. Compare ID-weight associations and
/// cutoff margins; zero margin is an inherently ambiguous selection witness.
pub struct Routing {
    pub tokens: usize,
    pub ids: Vec<usize>,
    pub weights: Vec<f32>,
    pub cutoff_margins: Vec<f64>,
}
pub struct Output {
    pub values: CudaTensor,
    pub routing: Routing,
    pub working_bytes: usize,
}

struct PackedWeight {
    weight: Nvfp4,
    input_scale: f32,
}
impl PackedWeight {
    fn upload(op: &PackedOperation<'_>, n: usize, k: usize, device: &CudaDevice) -> Result<Self> {
        Ok(Self {
            weight: Nvfp4::upload(op.payload, [n as u64, k as u64], DType::BF16, device)
                .map_err(|e| e.to_string())?,
            input_scale: op.input_scale,
        })
    }
    fn project(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.weight
            .project_w4a4_scaled(x, self.input_scale)
            .map_err(|e| e.to_string())
    }
}

pub struct Moe<P> {
    source: SourceBinding,
    fixed_operations: [Id; 6],
    provider: P,
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    router: Handle,
    bias: Handle,
    fc1: Handle,
    fc2: Handle,
    shared_up: PackedWeight,
    shared_down: PackedWeight,
    limits: Limits,
    admission: Admission,
    // One in-flight recovery bit, not model history or a commit protocol. Set
    // before dispatch so an unwind/uncertain completion cannot silently start
    // another call; reset only drains and acknowledges the failure. Result
    // errors conservatively require that same explicit acknowledgement even
    // when the attempted final fence succeeded. Cross-layer commit is external.
    poisoned: bool,
    // Installed BEFORE projection dispatch. A failed sync is not a drain;
    // retain these operands until a subsequent successful completion/reset.
    expert_operands: cache::Operands<(PackedWeight, PackedWeight)>,
}

fn bytes(dims: &[usize], width: usize) -> Result<usize> {
    dims.iter()
        .try_fold(width, |n, &d| {
            n.checked_mul(d)
                .filter(|&v| d > 0 && v <= u32::MAX as usize)
        })
        .ok_or("MoE empty/overflowing or excessive u32 buffer extent".into())
}
fn packed_bytes(n: usize, k: usize) -> Result<usize> {
    let count = bytes(&[n, k], 1)?;
    if k % 64 != 0 || n % 8 != 0 {
        return Err("MoE packed MMA geometry unsupported".into());
    }
    count
        .checked_div(2)
        .and_then(|v| v.checked_add(count / 16))
        .and_then(|v| v.checked_add(4))
        .ok_or("MoE packed length overflow".into())
}
/// Four device planes; the two trailing global scales remain host scalars.
pub(super) fn expert_device_bytes() -> Result<usize> {
    packed_bytes(INTERMEDIATE, LATENT)?
        .checked_sub(4)
        .and_then(|n| n.checked_mul(2))
        .ok_or("MoE expert device byte overflow".into())
}
pub(super) fn admission(limits: Limits) -> Result<Admission> {
    let fixed_bytes = bytes(&[EXPERTS, HIDDEN], 2)?
        + bytes(&[EXPERTS], 4)?
        + 2 * bytes(&[HIDDEN, LATENT], 2)?
        + 2 * packed_bytes(HIDDEN, SHARED)?;
    let expert_bytes = 2 * packed_bytes(INTERMEDIATE, LATENT)?;
    if fixed_bytes > limits.max_fixed_bytes || expert_bytes > limits.max_expert_bytes {
        return Err("MoE fixed/one-expert logical resident budget exceeded".into());
    }
    Ok(Admission {
        fixed_bytes,
        expert_bytes,
    })
}
fn finite_bf16(data: &[u8], n: usize, k: usize) -> Result<()> {
    if data.len() != bytes(&[n, k], 2)?
        || data
            .chunks_exact(2)
            .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
    {
        return Err("MoE BF16 weight shape/nonfinite refusal".into());
    }
    Ok(())
}
fn validate_packed(op: &PackedOperation<'_>, n: usize, k: usize) -> Result<()> {
    if op.payload.len() != packed_bytes(n, k)? {
        return Err("MoE packed weight length mismatch".into());
    }
    let end = op.payload.len() - 4;
    let global = f32::from_le_bytes(op.payload[end..].try_into().unwrap());
    calibrated_scale(op.input_scale, global).map_err(|e| e.to_string())?;
    if op.payload[n * k / 2..end].iter().any(|v| v & 0x7f == 0x7f) {
        return Err("MoE nonfinite E4M3 block scale".into());
    }
    Ok(())
}
fn validate_fixed(w: &FixedOperands<'_>) -> Result<()> {
    if w.correction_bias.len() != EXPERTS || w.correction_bias.iter().any(|v| !v.is_finite()) {
        return Err("MoE correction bias must be finite F32[512]".into());
    }
    finite_bf16(w.router_bf16, EXPERTS, HIDDEN)?;
    finite_bf16(w.fc1_bf16, LATENT, HIDDEN)?;
    finite_bf16(w.fc2_bf16, HIDDEN, LATENT)?;
    validate_packed(&w.shared_up, SHARED, HIDDEN)?;
    validate_packed(&w.shared_down, HIDDEN, SHARED)
}

/// The same miss path for standalone and cached owners. The uploader seam
/// lets host tests check query/validation/materialization without CUDA calls.
fn load_expert<P: ExpertProvider, W>(
    provider: &P,
    index: usize,
    mut upload: impl FnMut(&PackedOperation<'_>, usize, usize) -> Result<W>,
    #[cfg(feature = "nemotron-timing")] mut timing: Option<&mut Timings>,
) -> Result<(W, W)> {
    #[cfg(feature = "nemotron-timing")]
    let mut lookup = host_timing::Lookup::start(timing.as_deref());
    let result = provider.with_expert(index, |pair| {
        #[cfg(feature = "nemotron-timing")]
        lookup.entered(timing.as_deref_mut());
        #[cfg(feature = "nemotron-timing")]
        let started = host_timing::clock(timing.as_deref());
        let validation = (|| {
            if pair.index != index {
                return Err("MoE selected expert index mismatch".into());
            }
            validate_packed(&pair.up, INTERMEDIATE, LATENT)?;
            validate_packed(&pair.down, LATENT, INTERMEDIATE)
        })();
        #[cfg(feature = "nemotron-timing")]
        host_timing::record(timing.as_deref_mut(), Phase::Validate, started, 0, validation.is_ok());
        validation?;
        #[cfg(feature = "nemotron-timing")]
        let started = host_timing::clock(timing.as_deref());
        let up = upload(&pair.up, INTERMEDIATE, LATENT);
        #[cfg(feature = "nemotron-timing")]
        host_timing::record(
            timing.as_deref_mut(), Phase::Upload, started,
            pair.up.payload.len() - 4, up.is_ok(),
        );
        let up = up?;
        #[cfg(feature = "nemotron-timing")]
        let started = host_timing::clock(timing.as_deref());
        let down = upload(&pair.down, LATENT, INTERMEDIATE);
        #[cfg(feature = "nemotron-timing")]
        host_timing::record(
            timing.as_deref_mut(), Phase::Upload, started,
            pair.down.payload.len() - 4, down.is_ok(),
        );
        Ok((up, down?))
    });
    #[cfg(feature = "nemotron-timing")]
    lookup.returned(timing.as_deref_mut());
    result
}

pub(super) fn working_bytes(t: usize) -> Result<usize> {
    if !(1..=MAX_CHUNK).contains(&t) {
        return Err("MoE requires B1/T1..256".into());
    }
    let p = t.div_ceil(16) * 16;
    let mut sum = 0usize;
    let mut add = |dims: &[usize], width| -> Result<()> {
        sum = sum
            .checked_add(bytes(dims, width)?)
            .ok_or("MoE working sum overflow")?;
        Ok(())
    };
    // Existing BF16 hand MMA pads M; F32 outputs stay padded until narrowed.
    for (k, n) in [(HIDDEN, EXPERTS), (HIDDEN, LATENT), (LATENT, HIDDEN)] {
        add(&[p, k], 2)?;
        add(&[p, n], 4)?;
    }
    add(&[t, ROUTE_WIDTH], 4)?;
    add(&[t, LATENT], 2)?; // fc1 narrowed
    add(&[t, LATENT], 2)?; // one expert gather
    add(&[t], 4)?; // one expert index upload
    for (k, n) in [
        (LATENT, INTERMEDIATE),
        (INTERMEDIATE, LATENT),
        (HIDDEN, SHARED),
        (SHARED, HIDDEN),
    ] {
        // Quantized activation codes+scales; direct BF16 MMA output (no F32 slab).
        add(&[t, k / 16], 9)?;
        add(&[t, n], 2)?;
    }
    add(&[t, INTERMEDIATE], 2)?;
    add(&[t, SHARED], 2)?; // ReLU2
    add(&[t, TOP_K, LATENT], 2)?; // staged selected expert down rows
    add(&[t, TOP_K], 8)?;
    add(&[t], 4)?; // scatter weights/rows/counts
    add(&[t, LATENT], 6)?; // F32 scatter and BF16 combine
    add(&[t, HIDDEN], 4)?; // fc2 narrow and final BF16 branch sum
    add(&[t], 4)?; // final finite flags
    Ok(sum)
}

fn fence(client: &ComputeClient<CudaRuntime>) -> Result<()> {
    cubecl::future::block_on(client.sync()).map_err(|e| format!("MoE CUDA completion: {e:?}"))
}
fn finish<T>(result: Result<T>, completion: Result<()>) -> Result<T> {
    match (result, completion) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(fence)) => Err(format!("{error}; additionally {fence}")),
    }
}

impl<P: ExpertProvider> Moe<P> {
    pub fn upload(
        device: CudaDevice,
        source: SourceBinding,
        fixed: FixedOperands<'_>,
        provider: P,
        limits: Limits,
    ) -> Result<Self> {
        Self::upload_with_expert_cache(device, source, fixed, provider, limits, 0)
    }
    /// Only the native frozen-snapshot assembly enables retention, after
    /// admitting all layer quotas against the aggregate resident limit.
    pub(super) fn upload_with_expert_cache(
        device: CudaDevice,
        source: SourceBinding,
        fixed: FixedOperands<'_>,
        provider: P,
        limits: Limits,
        cache_pairs: usize,
    ) -> Result<Self> {
        let admission = admission(limits)?;
        let expert_operands = cache::Operands::new(cache_pairs)?;
        let resident = expert_device_bytes()?.checked_mul(cache_pairs)
            .and_then(|n| n.checked_add(admission.fixed_bytes))
            .ok_or("MoE cached resident byte overflow")?;
        if resident > limits.max_fixed_bytes {
            return Err("MoE cached resident budget exceeded".into());
        }
        if provider.source() != source {
            return Err("MoE provider/source binding mismatch".into());
        }
        validate_fixed(&fixed)?; // All host geometry/finite/scale checks precede upload.
        let client = CudaRuntime::client(&device);
        let fixed_operations = [
            fixed.operations[0],
            fixed.operations[1],
            fixed.operations[2],
            fixed.operations[3],
            fixed.shared_up.operation,
            fixed.shared_down.operation,
        ];
        Ok(Self {
            source,
            fixed_operations,
            provider,
            router: client.create_from_slice(fixed.router_bf16),
            bias: client.create_from_slice(f32::as_bytes(fixed.correction_bias)),
            fc1: client.create_from_slice(fixed.fc1_bf16),
            fc2: client.create_from_slice(fixed.fc2_bf16),
            shared_up: PackedWeight::upload(&fixed.shared_up, SHARED, HIDDEN, &device)?,
            shared_down: PackedWeight::upload(&fixed.shared_down, HIDDEN, SHARED, &device)?,
            client,
            device,
            limits,
            admission,
            poisoned: false,
            expert_operands,
        })
    }
    pub fn source(&self) -> SourceBinding {
        self.source
    }
    pub fn fixed_operations(&self) -> [Id; 6] {
        self.fixed_operations
    }
    pub fn admission(&self) -> Admission {
        self.admission
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn invalidate(&mut self) {
        self.poisoned = true;
    }
    /// Drain uncertain work and clear the recovery bit. There is no MoE token
    /// history to clear; weights/provider are unchanged. The owning hybrid
    /// decides how to recover its separate attention/Mamba state.
    pub fn reset(&mut self) -> Result<()> {
        self.poisoned = true;
        fence(&self.client)?;
        self.expert_operands.complete(&Ok(()), false);
        self.poisoned = false;
        Ok(())
    }
    fn tensor(&self, h: Handle, t: usize, n: usize, dtype: DType) -> CudaTensor {
        CudaTensor::new_contiguous(
            self.client.clone(),
            self.device.clone(),
            [1, t, n].into(),
            h,
            dtype,
        )
    }
    fn preflight(&self, x: &CudaTensor) -> Result<(usize, usize)> {
        if self.poisoned {
            return Err("MoE poisoned; successful drain/reset required".into());
        }
        if self.provider.source() != self.source {
            return Err("MoE provider binding changed".into());
        }
        let dims = x.meta.shape().as_slice();
        if dims.len() != 3
            || dims[0] != 1
            || dims[2] != HIDDEN
            || x.dtype != DType::BF16
            || x.qparams.is_some()
            || x.device != self.device
            || !std::ptr::eq(x.client.properties(), self.client.properties())
        {
            return Err("MoE expects BF16[1,T,4096] on its bound client/device".into());
        }
        let t = dims[1];
        let working = working_bytes(t)?;
        if working > self.limits.max_working_bytes {
            return Err("MoE working-byte budget exceeded".into());
        }
        let required = bytes(dims, 2)? as u64;
        let s = x.handle.offset_start.unwrap_or(0);
        let e = x.handle.offset_end.unwrap_or(0);
        if s != 0
            || s.checked_add(e).is_none_or(|v| v > x.handle.size())
            || x.handle.size() - s - e < required
            || x.meta.strides().len() != 3
            || x.meta.strides()[2] != 1
            || (t > 1 && x.meta.strides()[1] != HIDDEN)
        {
            return Err("MoE requires a covered contiguous aligned base allocation".into());
        }
        Ok((t, working))
    }
    fn linear(&self, x: &CudaTensor, weight: &Handle, k: usize, n: usize) -> CudaTensor {
        let t = x.meta.shape().as_slice()[1];
        let p = t.div_ceil(16) * 16;
        let padded = bf16gemm::pad_bf16_launch(&self.client, &x.handle, t * k, p * k);
        let out = bf16gemm::bf16_linear_launch(&self.client, &padded, weight, p, k, n);
        self.tensor(out, t, n, DType::F32)
    }
    fn narrow(&self, x: &CudaTensor) -> CudaTensor {
        let s = x.meta.shape().as_slice();
        let out = bf16gemm::to_bf16_launch(&self.client, &x.handle, s[1] * s[2], s[1] * s[2]);
        self.tensor(out, s[1], s[2], DType::BF16)
    }
    fn relu2(&self, x: &CudaTensor) -> CudaTensor {
        let s = x.meta.shape().as_slice();
        let count = s[1] * s[2];
        let out = self.client.empty(count * 2);
        unsafe {
            relu2_kernel::launch_unchecked::<CudaRuntime>(
                &self.client,
                CubeCount::new_1d(count.div_ceil(THREADS as usize) as u32),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(x.handle.clone(), count),
                ArrayArg::from_raw_parts(out.clone(), count),
                count,
            );
        }
        self.tensor(out, s[1], s[2], DType::BF16)
    }

    /// Fully fenced mixer output. Input is immutable. Preflight refusals leave
    /// this owner usable; any later error keeps it poisoned until reset. The
    /// outer hybrid owns common attended-count publication, not GPU rollback.
    pub fn forward(&mut self, x: &CudaTensor) -> Result<Output> {
        self.forward_observed(x, &mut |_, _| Ok(()))
    }
    // Private diagnostic hook: tests save actual native stages for independent
    // GPU comparison. Production does no stage readback/copy through this hook.
    fn forward_observed(
        &mut self,
        x: &CudaTensor,
        observe: &mut impl FnMut(&str, &CudaTensor) -> Result<()>,
    ) -> Result<Output> {
        self.forward_inner(
            x,
            observe,
            #[cfg(feature = "nemotron-timing")]
            None,
        )
    }

    #[cfg(feature = "nemotron-timing")]
    pub(super) fn forward_timed(&mut self, x: &CudaTensor, timing: &mut Timings) -> Result<Output> {
        self.forward_inner(x, &mut |_, _| Ok(()), Some(timing))
    }

    fn forward_inner(
        &mut self,
        x: &CudaTensor,
        observe: &mut impl FnMut(&str, &CudaTensor) -> Result<()>,
        #[cfg(feature = "nemotron-timing")] mut timing: Option<&mut Timings>,
    ) -> Result<Output> {
        let (t, working) = self.preflight(x)?;
        self.poisoned = true;
        #[cfg(feature = "nemotron-timing")]
        let started = host_timing::clock(timing.as_deref());
        let result = self.run(
            x,
            t,
            working,
            observe,
            #[cfg(feature = "nemotron-timing")]
            timing.as_deref_mut(),
        );
        #[cfg(feature = "nemotron-timing")]
        let fence_started = host_timing::clock(timing.as_deref());
        let completion = fence(&self.client);
        #[cfg(feature = "nemotron-timing")]
        host_timing::record(
            timing.as_deref_mut(),
            Phase::MixerFence,
            fence_started,
            0,
            completion.is_ok(),
        );
        self.expert_operands.complete(&completion, false);
        let result = finish(result, completion);
        if result.is_ok() {
            self.poisoned = false;
        }
        #[cfg(feature = "nemotron-timing")]
        host_timing::record(timing, Phase::Mixer, started, 0, result.is_ok());
        result
    }

    fn run(
        &mut self,
        x: &CudaTensor,
        t: usize,
        working: usize,
        observe: &mut impl FnMut(&str, &CudaTensor) -> Result<()>,
        #[cfg(feature = "nemotron-timing")] mut timing: Option<&mut Timings>,
    ) -> Result<Output> {
        let logits = self.linear(x, &self.router, HIDDEN, EXPERTS);
        let route = self.client.empty(t * ROUTE_WIDTH * 4);
        unsafe {
            route_kernel::launch_unchecked::<CudaRuntime>(
                &self.client,
                CubeCount::new_1d(t as u32),
                CubeDim::new_1d(EXPERTS as u32),
                ArrayArg::from_raw_parts(x.handle.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(logits.handle.clone(), t * EXPERTS),
                ArrayArg::from_raw_parts(self.bias.clone(), EXPERTS),
                ArrayArg::from_raw_parts(route.clone(), t * ROUTE_WIDTH),
            );
        }
        // One bounded readback, including error flags, BEFORE any expert lookup.
        let raw = self
            .client
            .read_one(route.clone())
            .map_err(|e| format!("MoE route read: {e:?}"))?;
        let routing = decode_routing(&raw, t)?;
        observe("router_logits", &logits)?;
        observe(
            "routing_record",
            &self.tensor(route, t, ROUTE_WIDTH, DType::F32),
        )?;
        let latent = self.narrow(&self.linear(x, &self.fc1, HIDDEN, LATENT));
        observe("latent", &latent)?;
        let schedule = Schedule::new(&routing)?;
        let selected = self.client.empty(t * TOP_K * LATENT * 2);
        for (index, rows) in schedule.by_expert.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            #[cfg(feature = "nemotron-timing")]
            let selected_started = host_timing::clock(timing.as_deref());
            // This immediate closure only collects Result returns for timing;
            // the existing per-expert and final fences remain on the same paths.
            let selected_result = (|| -> Result<()> {
                if self.provider.source() != self.source {
                    return Err("MoE selected expert/source binding mismatch".into());
                }
                let mut hit = true;
                let staged = self.expert_operands.stage(index, || {
                    hit = false;
                    load_expert(
                        &self.provider,
                        index,
                        |op, n, k| PackedWeight::upload(op, n, k, &self.device),
                        #[cfg(feature = "nemotron-timing")]
                        timing.as_deref_mut(),
                    )
                });
                #[cfg(feature = "nemotron-timing")]
                if (staged.is_ok() || !hit) && let Some(timing) = timing.as_deref_mut() {
                    timing.cache_access(hit);
                }
                staged?;
                let (up, down) = self.expert_operands.pending();
                #[cfg(feature = "nemotron-timing")]
                let enqueue_started = host_timing::clock(timing.as_deref());
                let result = (|| {
                    let idx: Vec<i32> = rows.iter().map(|&(token, _)| token as i32).collect();
                    let idx = self.client.create_from_slice(i32::as_bytes(&idx));
                    let gathered = moegroup::gather_grouped_bf16_from_bf16(
                        &self.client,
                        &latent.handle,
                        &idx,
                        t,
                        rows.len(),
                        LATENT,
                    );
                    let gathered = self.tensor(gathered, rows.len(), LATENT, DType::BF16);
                    let high = up.project(&gathered)?;
                    let active = self.relu2(&high);
                    let low = down.project(&active)?;
                    observe(&format!("expert{index}_input"), &gathered)?;
                    observe(&format!("expert{index}_up"), &high)?;
                    observe(&format!("expert{index}_relu2"), &active)?;
                    observe(&format!("expert{index}_down"), &low)?;
                    let len = rows.len() * LATENT;
                    unsafe {
                        copy_kernel::launch_unchecked::<CudaRuntime>(
                            &self.client,
                            CubeCount::new_1d(len.div_ceil(THREADS as usize) as u32),
                            CubeDim::new_1d(THREADS),
                            ArrayArg::from_raw_parts(low.handle.clone(), len),
                            ArrayArg::from_raw_parts(selected.clone(), t * TOP_K * LATENT),
                            len,
                            schedule.offsets[index] * LATENT,
                        );
                    }
                    Ok(())
                })();
                #[cfg(feature = "nemotron-timing")]
                host_timing::record(
                    timing.as_deref_mut(),
                    Phase::Enqueue,
                    enqueue_started,
                    0,
                    result.is_ok(),
                );
                // Keep both weight owners alive until their queued work completes,
                // including setup/observer failures after the first projection.
                #[cfg(feature = "nemotron-timing")]
                let fence_started = host_timing::clock(timing.as_deref());
                let completion = fence(&self.client);
                #[cfg(feature = "nemotron-timing")]
                host_timing::record(
                    timing.as_deref_mut(),
                    Phase::ExpertFence,
                    fence_started,
                    0,
                    completion.is_ok(),
                );
                let evicted = self.expert_operands.complete(&completion, result.is_ok());
                #[cfg(feature = "nemotron-timing")]
                if evicted && let Some(timing) = timing.as_deref_mut() {
                    timing.cache_eviction();
                }
                #[cfg(not(feature = "nemotron-timing"))]
                let _ = (hit, evicted);
                finish(result, completion)
            })();
            #[cfg(feature = "nemotron-timing")]
            host_timing::record(
                timing.as_deref_mut(),
                Phase::Selected,
                selected_started,
                0,
                selected_result.is_ok(),
            );
            selected_result?;
        }
        let weights = self
            .client
            .create_from_slice(f32::as_bytes(&schedule.weights));
        let row_ids = self
            .client
            .create_from_slice(u32::as_bytes(&schedule.token_rows));
        let counts = vec![TOP_K as u32; t];
        let counts = self.client.create_from_slice(u32::as_bytes(&counts));
        let combined = moegroup::scatter_weighted_bf16(
            &self.client,
            &selected,
            &weights,
            &row_ids,
            &counts,
            t * TOP_K,
            t,
            LATENT,
            TOP_K,
        );
        let combined = self.tensor(combined, t, LATENT, DType::F32);
        observe("combined_f32", &combined)?;
        let combined = self.narrow(&combined);
        observe("combined_bf16", &combined)?;
        let expanded = self.narrow(&self.linear(&combined, &self.fc2, LATENT, HIDDEN));
        let shared_up = self.shared_up.project(x)?;
        let shared_active = self.relu2(&shared_up);
        let shared = self.shared_down.project(&shared_active)?;
        observe("expanded", &expanded)?;
        observe("shared_up", &shared_up)?;
        observe("shared_relu2", &shared_active)?;
        observe("shared_down", &shared)?;
        let out = self.client.empty(t * HIDDEN * 2);
        let flags = self.client.empty(t * 4);
        unsafe {
            sum_kernel::launch_unchecked::<CudaRuntime>(
                &self.client,
                CubeCount::new_1d(t.div_ceil(THREADS as usize) as u32),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(expanded.handle.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(shared.handle.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(out.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(flags.clone(), t),
                t,
            );
        }
        let status = self
            .client
            .read_one(flags)
            .map_err(|e| format!("MoE finite read: {e:?}"))?;
        if status.len() < t * 4
            || status[..t * 4]
                .chunks_exact(4)
                .any(|b| u32::from_le_bytes(b.try_into().unwrap()) != 0)
        {
            return Err("MoE nonfinite mixer result".into());
        }
        let values = self.tensor(out, t, HIDDEN, DType::BF16);
        observe("output", &values)?;
        Ok(Output {
            values,
            routing,
            working_bytes: working,
        })
    }
}

// All state below is ephemeral within one forward, never a parameter catalog.
struct Schedule {
    by_expert: Vec<Vec<(usize, f32)>>,
    offsets: Vec<usize>,
    weights: Vec<f32>,
    token_rows: Vec<u32>,
}
impl Schedule {
    fn new(r: &Routing) -> Result<Self> {
        let mut by_expert = vec![Vec::new(); EXPERTS];
        for t in 0..r.tokens {
            for j in 0..TOP_K {
                let i = t * TOP_K + j;
                by_expert[r.ids[i]].push((t, r.weights[i]));
            }
        }
        let mut offsets = Vec::with_capacity(EXPERTS);
        let mut weights = Vec::with_capacity(r.tokens * TOP_K);
        let mut token_rows = vec![0u32; r.tokens * TOP_K];
        let mut count = vec![0usize; r.tokens];
        for rows in &by_expert {
            offsets.push(weights.len());
            for &(t, weight) in rows {
                if count[t] == TOP_K {
                    return Err("MoE routing row overflow".into());
                }
                token_rows[t * TOP_K + count[t]] = weights.len() as u32;
                count[t] += 1;
                weights.push(weight);
            }
        }
        if count.iter().any(|&v| v != TOP_K) {
            return Err("MoE incomplete routing".into());
        }
        Ok(Self {
            by_expert,
            offsets,
            weights,
            token_rows,
        })
    }
}
fn decode_routing(raw: &[u8], t: usize) -> Result<Routing> {
    if raw.len() < t * ROUTE_WIDTH * 4 {
        return Err("short MoE routing readback".into());
    }
    let values: Vec<f32> = raw[..t * ROUTE_WIDTH * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut result = Routing {
        tokens: t,
        ids: Vec::with_capacity(t * TOP_K),
        weights: Vec::with_capacity(t * TOP_K),
        cutoff_margins: Vec::with_capacity(t),
    };
    for row in values.chunks_exact(ROUTE_WIDTH) {
        if row[ROUTE_WIDTH - 1] != 0.0 || row.iter().any(|v| !v.is_finite()) {
            return Err("MoE nonfinite input/router or invalid route flag".into());
        }
        let mut seen = [false; EXPERTS];
        for j in 0..TOP_K {
            let id = row[j];
            if id < 0.0
                || id >= EXPERTS as f32
                || id.fract() != 0.0
                || seen[id as usize]
                || row[TOP_K + j] < 0.0
                || row[TOP_K + j] > 5.0
            {
                return Err("MoE invalid/duplicate selected expert or weight".into());
            }
            seen[id as usize] = true;
            result.ids.push(id as usize);
            result.weights.push(row[TOP_K + j]);
        }
        let margin = f64::from(row[2 * TOP_K]) - f64::from(row[2 * TOP_K + 1]);
        if margin < 0.0 {
            return Err("MoE negative cutoff margin".into());
        }
        result.cutoff_margins.push(margin);
    }
    Ok(result)
}

#[cube]
fn finite(v: f32) -> bool {
    v == v && Abs::abs(v) <= 3.4028235e38f32
}

#[cube(launch_unchecked)]
fn route_kernel(input: &Array<bf16>, logits: &Array<f32>, bias: &Array<f32>, out: &mut Array<f32>) {
    let t = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    let mut choice = SharedMemory::<f32>::new(EXPERTS);
    let mut scores = SharedMemory::<f32>::new(EXPERTS);
    let mut bad = SharedMemory::<u32>::new(EXPERTS);
    let v = logits[t * EXPERTS + u];
    let score = 1.0f32 / (1.0f32 + Exp::exp(-v));
    let biased = score + bias[u];
    let mut invalid = !finite(v) || !finite(bias[u]) || !finite(score) || !finite(biased);
    for i in 0..8usize {
        invalid = invalid || !finite(f32::cast_from(input[t * HIDDEN + u + i * EXPERTS]));
    }
    bad[u] = select(invalid, 1u32, 0u32);
    choice[u] = biased;
    scores[u] = score;
    sync_cube();
    if u == 0 {
        let mut fail = 0u32;
        for e in 0..EXPERTS {
            fail += bad[e];
        }
        let base = t * ROUTE_WIDTH;
        if fail != 0 {
            for j in 0..ROUTE_WIDTH {
                out[base + j] = 0.0f32;
            }
            out[base + ROUTE_WIDTH - 1] = 1.0f32;
        } else {
            let mut picked = Array::<u32>::new(TOP_K);
            let mut used = Array::<u32>::new(EXPERTS);
            for e in 0..EXPERTS {
                used[e] = 0u32;
            }
            let mut sum = 0.0f32;
            // One extra selection records the actual 22/23 cutoff, without
            // subtracting extreme finite scores in F32 (host uses F64).
            for j in 0..TOP_K + 1 {
                let mut have = false;
                let mut best = 0.0f32;
                let mut index = 0u32;
                for e in 0..EXPERTS {
                    if used[e] == 0u32 && (!have || choice[e] > best) {
                        have = true;
                        best = choice[e];
                        index = e as u32;
                    }
                }
                used[index as usize] = 1u32;
                if j < TOP_K {
                    picked[j] = index;
                    sum += scores[index as usize];
                }
                if j == TOP_K - 1 {
                    out[base + 2 * TOP_K] = best;
                }
                if j == TOP_K {
                    out[base + 2 * TOP_K + 1] = best;
                }
            }
            let denominator = sum + 1e-20f32;
            for j in 0..TOP_K {
                out[base + j] = f32::cast_from(picked[j]);
                out[base + TOP_K + j] = (scores[picked[j] as usize] / denominator) * 5.0f32;
            }
            out[base + ROUTE_WIDTH - 1] = 0.0f32;
        }
    }
}
#[cube(launch_unchecked)]
fn relu2_kernel(x: &Array<bf16>, out: &mut Array<bf16>, n: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let v = f32::cast_from(x[i]);
        // Preserve NaN rather than max() quietly selecting its other operand.
        let positive = select(v < 0.0f32, 0.0f32, v);
        out[i] = bf16::cast_from(positive * positive);
    }
}
#[cube(launch_unchecked)]
fn copy_kernel(x: &Array<bf16>, out: &mut Array<bf16>, n: usize, offset: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        out[offset + i] = x[i];
    }
}
#[cube(launch_unchecked)]
fn sum_kernel(
    a: &Array<bf16>,
    b: &Array<bf16>,
    out: &mut Array<bf16>,
    flags: &mut Array<u32>,
    t: usize,
) {
    let row = ABSOLUTE_POS as usize;
    if row < t {
        let mut bad = 0u32;
        for i in 0..HIDDEN {
            let at = row * HIDDEN + i;
            let v = bf16::cast_from(f32::cast_from(a[at]) + f32::cast_from(b[at]));
            out[at] = v;
            if !finite(f32::cast_from(v)) {
                bad = 1u32;
            }
        }
        flags[row] = bad;
    }
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
