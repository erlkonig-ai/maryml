//! Native Super MoE mixer, before the outer residual, without state/history.
//!
//! BF16 H4096 -> F32 router; BF16 latent1024 -> 22/512 NVFP4 ReLU2
//! experts -> F32 weighted sum -> BF16 -> BF16 latent expansion. Add the
//! independent full-width shared NVFP4 ReLU2 branch. No latent norm/gate.
//! Router BF16 operands use existing BF16 MMA with F32 accumulation/output:
//! this is a reduction variant of HF's widened F32 dot, NOT bit parity.
//! All arithmetic stays on GPU. Small routing metadata alone returns to host
//! to pick which of the already resident expert pairs run.
//!
//! Upload copies every one of the 512 expert pairs into device memory once,
//! beside the fixed operands, and fences. Forward and prefill only read that
//! bank; no expert is looked up, validated or uploaded per call.
//! Selected experts are enqueued back to back on the one stream; the final
//! result is fenced once per call (not once per expert).
//! After dispatch any error/panic poisons this owner until reset succeeds.
//! An outer hybrid owner must also discard/reset attention AND Mamba history;
//! resetting this stateless mixer alone cannot undo another layer's effects.

#[cfg(feature = "nemotron-timing")]
use super::host_timing::{self, Phase, Timings};
use crate::{
    models::inkling::{bf16gemm, moegroup},
    nn::{nvfp4_gemm::{self, GroupedBank, calibrated_scale}, nvfp4_gemv::Nvfp4},
};
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
    server::Handle,
};
use half::bf16;
use std::sync::Arc;
use triblespace::prelude::Id;

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
/// weights. Fixed operands and the expert provider must use ONE held immutable
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

/// Source of the expert pairs, read once per index at upload. Use the SAME
/// frozen snapshot that selected FixedOperands. Moe does not retain it.
pub trait ExpertProvider {
    fn source(&self) -> SourceBinding;
    /// The selected views need only live for this callback.
    fn with_expert<T>(
        &self,
        index: usize,
        use_pair: impl FnOnce(ExpertPair<'_>) -> Result<T>,
    ) -> Result<T>;
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Fixed operands plus the whole expert bank, logical payload bytes.
    pub max_resident_bytes: usize,
    /// Conservative sum of logical temporary allocations for ONE call.
    /// Excludes caller input, host staging, allocator/JIT and runtime
    /// internals. This is not a whole-device or cgroup memory bound.
    pub max_working_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub fixed_bytes: usize,
    /// All EXPERTS pairs' device planes, resident from upload on.
    pub bank_bytes: usize,
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

/// Immutable device operands of one MoE layer, shared by every Session of a
/// DeviceModel: the fixed operands and the bank of all EXPERTS (up, down)
/// pairs, indexed by expert ID.
pub(super) struct Weights {
    source: SourceBinding,
    fixed_operations: [Id; 6],
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    router: Handle,
    bias: Handle,
    fc1: Handle,
    fc2: Handle,
    shared_up: PackedWeight,
    shared_down: PackedWeight,
    /// Every expert's (up, down); each plane is a view into `bank`.
    experts: Vec<(PackedWeight, PackedWeight)>,
    bank: Bank,
}

/// One layer's whole expert bank as contiguous `[EXPERTS, n, ...]` planes,
/// the same bytes the per-expert views in `Weights::experts` read (no second
/// copy), plus every expert's activation input scale and its f32 epilogue
/// product `scale2 * input_scale`, indexed by expert ID. A decode step runs
/// all of its selected experts from here in one grouped launch per side.
struct Bank {
    up_codes: Handle,
    up_scales: Handle,
    up_input: Handle,
    up_alpha: Handle,
    down_codes: Handle,
    down_scales: Handle,
    down_input: Handle,
    down_alpha: Handle,
}

#[cube(launch_unchecked)]
fn copy_words_kernel(x: &Array<u32>, out: &mut Array<u32>, n: usize, offset: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        out[offset + i] = x[i];
    }
}

/// Upload every expert pair (ascending ID, through `load_expert`) into the
/// contiguous bank and return per-expert views of it. Each pair is uploaded
/// on its own as before and copied on the device into its slot; the
/// temporary is dropped and its memory reused in stream order, so at most
/// one extra pair is resident during the upload.
fn upload_bank<P: ExpertProvider>(
    provider: &P,
    client: &ComputeClient<CudaRuntime>,
    device: &CudaDevice,
) -> Result<(Vec<(PackedWeight, PackedWeight)>, Bank)> {
    let (up_c, up_s) = (INTERMEDIATE * LATENT / 2, INTERMEDIATE * LATENT / 16);
    let (down_c, down_s) = (LATENT * INTERMEDIATE / 2, LATENT * INTERMEDIATE / 16);
    let planes = [client.empty(EXPERTS * up_c), client.empty(EXPERTS * up_s),
        client.empty(EXPERTS * down_c), client.empty(EXPERTS * down_s)];
    let mut scales: [Vec<f32>; 4] = Default::default(); // up input, up alpha, down input, down alpha
    let mut experts = Vec::with_capacity(EXPERTS);
    for index in 0..EXPERTS {
        let pair = load_expert(provider, index, |op, n, k| {
            let up = n == INTERMEDIATE;
            let (codes, scales_dst, cb, sb) = if up {
                (&planes[0], &planes[1], up_c, up_s)
            } else {
                (&planes[2], &planes[3], down_c, down_s)
            };
            let temp = PackedWeight::upload(op, n, k, device)?;
            let alpha = calibrated_scale(temp.input_scale, temp.weight.scale2).map_err(|e| e.to_string())?;
            for (src, dst, bytes) in [(&temp.weight.codes, codes, cb), (&temp.weight.scales, scales_dst, sb)] {
                let words = bytes / 4;
                // SAFETY: src holds this operation's `bytes` (whole words); the
                // slot [index * words, (index + 1) * words) is inside dst.
                unsafe {
                    copy_words_kernel::launch_unchecked::<CudaRuntime>(
                        client,
                        CubeCount::new_1d(words.div_ceil(THREADS as usize) as u32),
                        CubeDim::new_1d(THREADS),
                        ArrayArg::from_raw_parts(src.clone(), words),
                        ArrayArg::from_raw_parts(dst.clone(), EXPERTS * words),
                        words,
                        index * words,
                    );
                }
            }
            let view = |h: &Handle, bytes: usize| h.clone().offset_start((index * bytes) as u64)
                .offset_end(((EXPERTS - 1 - index) * bytes) as u64);
            let weight = Nvfp4::from_parts(view(codes, cb), view(scales_dst, sb), temp.weight.scale2,
                n, k, DType::BF16, device.clone()).map_err(|e| e.to_string())?;
            let side = if up { 0 } else { 2 };
            scales[side].push(temp.input_scale);
            scales[side + 1].push(alpha);
            Ok(PackedWeight { weight, input_scale: temp.input_scale })
        })?;
        experts.push(pair);
    }
    let [up_codes, up_scales, down_codes, down_scales] = planes;
    let [up_input, up_alpha, down_input, down_alpha] =
        scales.map(|v| client.create_from_slice(f32::as_bytes(&v)));
    Ok((experts, Bank { up_codes, up_scales, up_input, up_alpha, down_codes, down_scales, down_input, down_alpha }))
}

pub struct Moe {
    pub(super) weights: Arc<Weights>,
    limits: Limits,
    admission: Admission,
    // One in-flight recovery bit, not model history or a commit protocol. Set
    // before dispatch so an unwind/uncertain completion cannot silently start
    // another call; reset only drains and acknowledges the failure. Result
    // errors conservatively require that same explicit acknowledgement even
    // when the attempted final fence succeeded. Cross-layer commit is external.
    poisoned: bool,
    /// Decode (t == 1) runs the selected experts as grouped launches from the
    /// bank. Tests turn it off to compare against the per-expert tiles.
    pub(super) grouped_decode: bool,
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
    let bank_bytes = expert_device_bytes()?
        .checked_mul(EXPERTS)
        .ok_or("MoE expert bank byte overflow")?;
    if fixed_bytes
        .checked_add(bank_bytes)
        .is_none_or(|n| n > limits.max_resident_bytes)
    {
        return Err("MoE fixed/expert-bank logical resident budget exceeded".into());
    }
    Ok(Admission {
        fixed_bytes,
        bank_bytes,
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

/// Read, validate and upload ONE pair at bank construction. Both operands
/// are validated before either upload. The uploader seam lets host tests
/// check query/validation/materialization without CUDA calls.
fn load_expert<P: ExpertProvider, W>(
    provider: &P,
    index: usize,
    mut upload: impl FnMut(&PackedOperation<'_>, usize, usize) -> Result<W>,
) -> Result<(W, W)> {
    provider.with_expert(index, |pair| {
        if pair.index != index {
            return Err("MoE selected expert index mismatch".into());
        }
        validate_packed(&pair.up, INTERMEDIATE, LATENT)?;
        validate_packed(&pair.down, LATENT, INTERMEDIATE)?;
        let up = upload(&pair.up, INTERMEDIATE, LATENT)?;
        Ok((up, upload(&pair.down, LATENT, INTERMEDIATE)?))
    })
}
/// Every expert in ascending ID order; the first failure stops the bank.
fn load_bank<P: ExpertProvider, W>(
    provider: &P,
    mut upload: impl FnMut(&PackedOperation<'_>, usize, usize) -> Result<W>,
) -> Result<Vec<(W, W)>> {
    (0..EXPERTS).map(|index| load_expert(provider, index, &mut upload)).collect()
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
/// Free the host copies that this thread's uploads left queued, then fence.
/// create_from_slice copies each plane to a host buffer, and the stream's drop
/// queue keeps it until a flush. A flush frees only the batch of the PREVIOUS
/// flush (after syncing that batch's fence) and queues the current one, so
/// the second flush frees everything uploaded before the first. sync() never
/// flushes, and a load launches no kernels, so without this every uploaded
/// byte would stay in host memory until loading ends. The fence observes
/// completion and any queued upload error.
pub(super) fn release_uploads(
    client: &ComputeClient<CudaRuntime>,
) -> std::result::Result<(), cubecl::server::ServerError> {
    client.flush()?;
    client.flush()?;
    cubecl::future::block_on(client.sync())
}
/// After a failed upload whose owners are already dropped: drain, free the
/// host copies, and give back the device memory, so a refused load holds
/// nothing. The first drain may only report the stream's queued error; the
/// second then completes. The caller keeps its original error.
pub(super) fn abandon_uploads(client: &ComputeClient<CudaRuntime>) {
    if release_uploads(client).is_err() {
        let _ = release_uploads(client);
    }
    client.memory_cleanup();
    trim_host_heap();
}
/// Return freed host memory to the system. glibc keeps the freed ~1.4 MB
/// upload copies in its heap for reuse; once loading ends nothing reuses them.
pub(super) fn trim_host_heap() {
    // SAFETY: malloc_trim only gives back memory the allocator holds free.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}
fn finish<T>(result: Result<T>, completion: Result<()>) -> Result<T> {
    match (result, completion) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(fence)) => Err(format!("{error}; additionally {fence}")),
    }
}

impl Moe {
    /// Upload the fixed operands, then every expert pair from `provider` in
    /// ascending ID order, and fence. Completion (or the first queued upload
    /// error) is observed here, so the bank is resident before this returns.
    /// The provider is not retained: no later call can read or upload experts.
    pub fn upload<P: ExpertProvider>(
        device: CudaDevice,
        source: SourceBinding,
        fixed: FixedOperands<'_>,
        provider: &P,
        limits: Limits,
    ) -> Result<Self> {
        let admission = admission(limits)?;
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
        let uploaded = (|| -> Result<Weights> {
            let (experts, bank) = upload_bank(provider, &client, &device)?;
            let weights = Weights {
                source,
                fixed_operations,
                router: client.create_from_slice(fixed.router_bf16),
                bias: client.create_from_slice(f32::as_bytes(fixed.correction_bias)),
                fc1: client.create_from_slice(fixed.fc1_bf16),
                fc2: client.create_from_slice(fixed.fc2_bf16),
                shared_up: PackedWeight::upload(&fixed.shared_up, SHARED, HIDDEN, &device)?,
                shared_down: PackedWeight::upload(&fixed.shared_down, HIDDEN, SHARED, &device)?,
                experts,
                bank,
                client: client.clone(),
                device,
            };
            // Bounds the host copies to one layer's uploads, and surfaces any
            // queued upload error before the owner escapes.
            release_uploads(&client).map_err(|e| format!("MoE CUDA completion: {e:?}"))?;
            Ok(weights)
        })();
        match uploaded {
            Ok(weights) => Ok(Self { weights: Arc::new(weights), limits, admission, poisoned: false, grouped_decode: true }),
            // Every owner made above is already dropped.
            Err(error) => {
                abandon_uploads(&client);
                Err(error)
            }
        }
    }
    /// New private recovery state over the SAME resident operands and bank.
    /// No source payload is read or uploaded here.
    pub(super) fn from_weights(weights: Arc<Weights>, limits: Limits) -> Result<Self> {
        let admission = admission(limits)?;
        Ok(Self { weights, limits, admission, poisoned: false, grouped_decode: true })
    }
    pub fn source(&self) -> SourceBinding {
        self.weights.source
    }
    pub fn fixed_operations(&self) -> [Id; 6] {
        self.weights.fixed_operations
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
    /// history to clear; the resident weights are unchanged. The owning hybrid
    /// decides how to recover its separate attention/Mamba state.
    pub fn reset(&mut self) -> Result<()> {
        self.poisoned = true;
        fence(&self.weights.client)?;
        self.poisoned = false;
        Ok(())
    }
    fn tensor(&self, h: Handle, t: usize, n: usize, dtype: DType) -> CudaTensor {
        CudaTensor::new_contiguous(
            self.weights.client.clone(),
            self.weights.device.clone(),
            [1, t, n].into(),
            h,
            dtype,
        )
    }
    fn preflight(&self, x: &CudaTensor) -> Result<(usize, usize)> {
        if self.poisoned {
            return Err("MoE poisoned; successful drain/reset required".into());
        }
        self.check_input(x)
    }
    fn check_input(&self, x: &CudaTensor) -> Result<(usize, usize)> {
        let dims = x.meta.shape().as_slice();
        if dims.len() != 3
            || dims[0] != 1
            || dims[2] != HIDDEN
            || x.dtype != DType::BF16
            || x.qparams.is_some()
            || x.device != self.weights.device
            || !std::ptr::eq(x.client.properties(), self.weights.client.properties())
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
        if t == 1 && n % bf16gemm::NTILE == 0 && k % (bf16gemm::KTILE * bf16gemm::ROW_UNROLL) == 0
            && x.handle.offset_start.unwrap_or(0) == 0
        {
            // Decode: the one-row kernel, byte-identical to row 0 below.
            let out = bf16gemm::bf16_linear_row_launch(&self.weights.client, &x.handle, weight, k, n);
            return self.tensor(out, 1, n, DType::F32);
        }
        let p = t.div_ceil(16) * 16;
        let padded = bf16gemm::pad_bf16_launch(&self.weights.client, &x.handle, t * k, p * k);
        let out = bf16gemm::bf16_linear_launch(&self.weights.client, &padded, weight, p, k, n);
        self.tensor(out, t, n, DType::F32)
    }
    fn narrow(&self, x: &CudaTensor) -> CudaTensor {
        let s = x.meta.shape().as_slice();
        let out = bf16gemm::to_bf16_launch(&self.weights.client, &x.handle, s[1] * s[2], s[1] * s[2]);
        self.tensor(out, s[1], s[2], DType::BF16)
    }
    fn relu2(&self, x: &CudaTensor) -> CudaTensor {
        let s = x.meta.shape().as_slice();
        let count = s[1] * s[2];
        let out = self.weights.client.empty(count * 2);
        unsafe {
            relu2_kernel::launch_unchecked::<CudaRuntime>(
                &self.weights.client,
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
        let completion = fence(&self.weights.client);
        #[cfg(feature = "nemotron-timing")]
        host_timing::record(
            timing.as_deref_mut(),
            Phase::MixerFence,
            fence_started,
            0,
            completion.is_ok(),
        );
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
        let (routing, latent) = self.route_tile(x, t, observe, None)?;
        let schedule = Schedule::new(&routing)?;
        let selected = if t == 1 && self.grouped_decode {
            self.grouped_experts(&routing, &latent)?
        } else {
            self.expert_tiles(&schedule, &latent, t, observe,
                #[cfg(feature = "nemotron-timing")] timing.as_deref_mut())?
        };
        let weights = self.weights
            .client
            .create_from_slice(f32::as_bytes(&schedule.weights));
        let row_ids = self.weights
            .client
            .create_from_slice(u32::as_bytes(&schedule.token_rows));
        let counts = vec![TOP_K as u32; t];
        let counts = self.weights.client.create_from_slice(u32::as_bytes(&counts));
        let combined = moegroup::scatter_weighted_bf16(
            &self.weights.client,
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
        let values = self.finish_tile(x, &combined, t, observe, None)?;
        Ok(Output {
            values,
            routing,
            working_bytes: working,
        })
    }

    /// The per-expert tiles (any t): each selected expert's rows gathered,
    /// projected and staged into the selected slab in ascending expert order.
    fn expert_tiles(
        &self,
        schedule: &Schedule,
        latent: &CudaTensor,
        t: usize,
        observe: &mut impl FnMut(&str, &CudaTensor) -> Result<()>,
        #[cfg(feature = "nemotron-timing")] mut timing: Option<&mut Timings>,
    ) -> Result<Handle> {
        let selected = self.weights.client.empty(t * TOP_K * LATENT * 2);
        for (index, rows) in schedule.by_expert.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            #[cfg(feature = "nemotron-timing")]
            let selected_started = host_timing::clock(timing.as_deref());
            // This immediate closure only collects Result returns for timing;
            // the existing per-expert and final fences remain on the same paths.
            let selected_result = (|| -> Result<()> {
                #[cfg(feature = "nemotron-timing")]
                let enqueue_started = host_timing::clock(timing.as_deref());
                let result = (|| {
                    let idx: Vec<i32> = rows.iter().map(|&(token, _)| token as i32).collect();
                    let idx = self.weights.client.create_from_slice(i32::as_bytes(&idx));
                    let gathered = moegroup::gather_grouped_bf16_from_bf16(
                        &self.weights.client,
                        &latent.handle,
                        &idx,
                        t,
                        rows.len(),
                        LATENT,
                    );
                    let gathered = self.tensor(gathered, rows.len(), LATENT, DType::BF16);
                    let (high, active, low) = self.project_expert(index, &gathered, None)?;
                    observe(&format!("expert{index}_input"), &gathered)?;
                    observe(&format!("expert{index}_up"), &high)?;
                    observe(&format!("expert{index}_relu2"), &active)?;
                    observe(&format!("expert{index}_down"), &low)?;
                    let len = rows.len() * LATENT;
                    unsafe {
                        copy_kernel::launch_unchecked::<CudaRuntime>(
                            &self.weights.client,
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
                // No per-expert fence: the bank is resident, so an expert tile
                // only reads device memory and its scratch is reused in stream
                // order (one stream), never concurrently live. The mixer fence
                // in forward_inner still observes every expert's completion
                // before this call returns, and any queued failure there.
                result
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
        Ok(selected)
    }

    /// Decode (t == 1): all TOP_K selected experts in two grouped W4A4
    /// launches per side from the contiguous bank. Groups are in ascending
    /// expert ID, the same slot order as `Schedule` gives a one-token tile, so
    /// the slab feeds the unchanged weighted scatter. Per expert the
    /// arithmetic is exactly the per-expert tile's (same quantization, same
    /// block-scaled MMA, same f32 epilogue product).
    fn grouped_experts(&self, routing: &Routing, latent: &CudaTensor) -> Result<Handle> {
        let client = &self.weights.client;
        let mut ids: Vec<u32> = routing.ids[..TOP_K].iter().map(|&e| e as u32).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|p| p[0] == p[1]) || ids.iter().any(|&e| e as usize >= EXPERTS) {
            return Err("MoE grouped decode needs distinct in-range expert IDs".into());
        }
        let ids = client.create_from_slice(u32::as_bytes(&ids));
        let b = &self.weights.bank;
        let up = GroupedBank { codes: &b.up_codes, scales: &b.up_scales, input_scale: &b.up_input,
            alpha: &b.up_alpha, experts: EXPERTS, n: INTERMEDIATE, k: LATENT };
        let down = GroupedBank { codes: &b.down_codes, scales: &b.down_scales, input_scale: &b.down_input,
            alpha: &b.down_alpha, experts: EXPERTS, n: LATENT, k: INTERMEDIATE };
        let high = nvfp4_gemm::w4a4_grouped_bf16(client, &latent.handle, false, &ids, TOP_K, &up)
            .map_err(|e| e.to_string())?;
        let active = self.relu2(&self.tensor(high, TOP_K, INTERMEDIATE, DType::BF16));
        nvfp4_gemm::w4a4_grouped_bf16(client, &active.handle, true, &ids, TOP_K, &down)
            .map_err(|e| e.to_string())
    }

    fn route_tile(&self, x: &CudaTensor, t: usize,
        observe: &mut impl FnMut(&str, &CudaTensor) -> Result<()>,
        mut held: Option<&mut Vec<CudaTensor>>) -> Result<(Routing, CudaTensor)>
    {
        let mut hold = |value: &CudaTensor| {
            if let Some(held) = held.as_deref_mut() { held.push(value.clone()); }
        };
        let logits = self.linear(x, &self.weights.router, HIDDEN, EXPERTS);
        hold(&logits);
        let route = self.weights.client.empty(t * ROUTE_WIDTH * 4);
        hold(&self.tensor(route.clone(), t, ROUTE_WIDTH, DType::F32));
        unsafe {
            route_kernel::launch_unchecked::<CudaRuntime>(
                &self.weights.client,
                CubeCount::new_1d(t as u32),
                CubeDim::new_1d(EXPERTS as u32),
                ArrayArg::from_raw_parts(x.handle.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(logits.handle.clone(), t * EXPERTS),
                ArrayArg::from_raw_parts(self.weights.bias.clone(), EXPERTS),
                ArrayArg::from_raw_parts(route.clone(), t * ROUTE_WIDTH),
            );
        }
        // One bounded readback, including error flags, BEFORE any expert lookup.
        let raw = self.weights
            .client
            .read_one(route.clone())
            .map_err(|e| format!("MoE route read: {e:?}"))?;
        let routing = decode_routing(&raw, t)?;
        observe("router_logits", &logits)?;
        observe(
            "routing_record",
            &self.tensor(route, t, ROUTE_WIDTH, DType::F32),
        )?;
        let latent_f32 = self.linear(x, &self.weights.fc1, HIDDEN, LATENT);
        hold(&latent_f32);
        let latent = self.narrow(&latent_f32);
        hold(&latent);
        observe("latent", &latent)?;
        Ok((routing, latent))
    }
    fn project_expert(&self, index: usize, gathered: &CudaTensor,
        mut held: Option<&mut Vec<CudaTensor>>) -> Result<(CudaTensor, CudaTensor, CudaTensor)> {
        let (up, down) = &self.weights.experts[index];
        let high = up.project(gathered)?;
        if let Some(held) = held.as_deref_mut() { held.push(high.clone()); }
        let active = self.relu2(&high);
        if let Some(held) = held.as_deref_mut() { held.push(active.clone()); }
        let low = down.project(&active)?;
        if let Some(held) = held { held.push(low.clone()); }
        Ok((high, active, low))
    }
    fn finish_tile(&self, x: &CudaTensor, combined: &CudaTensor, t: usize,
        observe: &mut impl FnMut(&str, &CudaTensor) -> Result<()>,
        mut held: Option<&mut Vec<CudaTensor>>) -> Result<CudaTensor>
    {
        let mut hold = |value: &CudaTensor| {
            if let Some(held) = held.as_deref_mut() { held.push(value.clone()); }
        };
        let expanded_f32 = self.linear(combined, &self.weights.fc2, LATENT, HIDDEN);
        hold(&expanded_f32);
        let expanded = self.narrow(&expanded_f32);
        hold(&expanded);
        let shared_up = self.weights.shared_up.project(x)?;
        hold(&shared_up);
        let shared_active = self.relu2(&shared_up);
        hold(&shared_active);
        let shared = self.weights.shared_down.project(&shared_active)?;
        hold(&shared);
        observe("expanded", &expanded)?;
        observe("shared_up", &shared_up)?;
        observe("shared_relu2", &shared_active)?;
        observe("shared_down", &shared)?;
        let out = self.weights.client.empty(t * HIDDEN * 2);
        hold(&self.tensor(out.clone(), t, HIDDEN, DType::BF16));
        let flags = self.weights.client.empty(t * 4);
        hold(&self.tensor(flags.clone(), t, 1, DType::U32));
        unsafe {
            sum_kernel::launch_unchecked::<CudaRuntime>(
                &self.weights.client,
                CubeCount::new_1d(t.div_ceil(THREADS as usize) as u32),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(expanded.handle.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(shared.handle.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(out.clone(), t * HIDDEN),
                ArrayArg::from_raw_parts(flags.clone(), t),
                t,
            );
        }
        let status = self.weights
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
        Ok(values)
    }

    // The ordinary Session owns whole-span availability. These internal tile
    // seams run only while that Session and this mixer are unavailable.
    pub(super) fn prefill_begin(&mut self) -> Result<()> {
        if self.poisoned {
            return Err("MoE prefill unavailable".into());
        }
        self.poisoned = true;
        Ok(())
    }
    pub(super) fn prefill_route(&self, x: &CudaTensor, held: &mut Vec<CudaTensor>)
        -> Result<(Routing, CudaTensor)>
    {
        let (t, _) = self.check_input(x)?;
        self.route_tile(x, t, &mut |_, _| Ok(()), Some(held))
    }
    pub(super) fn prefill_expert_tile(&self, expert: usize, latent: &CudaTensor,
        accumulator: &CudaTensor, rows: &[(i32, f32)], held: &mut Vec<CudaTensor>) -> Result<()>
    {
        let shape = latent.meta.shape().as_slice();
        if expert >= EXPERTS || shape.len() != 3 || shape[0] != 1 || shape[2] != LATENT
            || accumulator.meta.shape().as_slice() != shape || latent.dtype != DType::BF16
            || accumulator.dtype != DType::F32 || rows.is_empty() || rows.len() > MAX_CHUNK
            || latent.device != self.weights.device || accumulator.device != self.weights.device
            || !std::ptr::eq(latent.client.properties(), self.weights.client.properties())
            || !std::ptr::eq(accumulator.client.properties(), self.weights.client.properties())
            || shape[1].checked_mul(LATENT).is_none_or(|n| n > u32::MAX as usize)
            || rows.iter().any(|&(i, w)| i < 0 || i as usize >= shape[1] || !w.is_finite())
            || rows.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        { return Err("MoE prefill expert tile binding/range/order mismatch".into()); }
        let indices: Vec<i32> = rows.iter().map(|r| r.0).collect();
        let weights: Vec<f32> = rows.iter().map(|r| r.1).collect();
        let idx = self.tensor(self.weights.client.create_from_slice(i32::as_bytes(&indices)),
            rows.len(), 1, DType::I32);
        held.push(idx.clone());
        let weight = self.tensor(self.weights.client.create_from_slice(f32::as_bytes(&weights)),
            rows.len(), 1, DType::F32);
        held.push(weight.clone());
        let gathered = moegroup::gather_grouped_bf16_from_bf16(&self.weights.client,
            &latent.handle, &idx.handle, shape[1], rows.len(), LATENT);
        let gathered = self.tensor(gathered, rows.len(), LATENT, DType::BF16);
        held.push(gathered.clone());
        let (_high, _active, low) = self.project_expert(expert, &gathered, Some(held))?;
        let len = rows.len() * LATENT;
        unsafe {
            ordered_update::launch_unchecked::<CudaRuntime>(&self.weights.client,
                CubeCount::new_1d(len.div_ceil(THREADS as usize) as u32), CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(low.handle.clone(), len),
                ArrayArg::from_raw_parts(idx.handle.clone(), rows.len()),
                ArrayArg::from_raw_parts(weight.handle.clone(), rows.len()),
                ArrayArg::from_raw_parts(accumulator.handle.clone(), shape[1] * LATENT), len,
            );
        }
        Ok(())
    }
    pub(super) fn prefill_finish(&self, x: &CudaTensor, accumulator: &CudaTensor,
        held: &mut Vec<CudaTensor>) -> Result<CudaTensor>
    {
        let (t, _) = self.check_input(x)?;
        if accumulator.meta.shape().as_slice() != [1, t, LATENT]
            || accumulator.dtype != DType::F32 || accumulator.device != self.weights.device
            || !std::ptr::eq(accumulator.client.properties(), self.weights.client.properties())
        { return Err("MoE prefill accumulator tile mismatch".into()); }
        let combined = self.narrow(accumulator);
        held.push(combined.clone());
        self.finish_tile(x, &combined, t, &mut |_, _| Ok(()), Some(held))
    }
    pub(super) fn prefill_done(&mut self) { self.poisoned = false; }

}

// Expert launches are ordered and each tile contains distinct token IDs.
// Preserve scatter_weighted_bf16's ascending-expert F32 FMA recurrence, with
// no separately rounded product and no atomic/reassociated contributions.
#[cube(launch_unchecked)]
fn ordered_update(y: &Array<bf16>, indices: &Array<i32>, weights: &Array<f32>,
    accumulator: &mut Array<f32>, len: usize)
{
    let i = ABSOLUTE_POS as usize;
    if i < len {
        let row = i / LATENT;
        let out = indices[row] as usize * LATENT + i % LATENT;
        accumulator[out] = fma(f32::cast_from(y[i]), weights[row], accumulator[out]);
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

// One cube of EXPERTS units per token. Each of the TOP_K + 1 selection rounds
// is a shared-memory argmax over the experts not yet picked: higher biased
// score wins, exact ties go to the lower expert ID, exactly the rule of the
// ascending strict-greater scan this replaces (route_kernel_serial in tests).
// Sums and outputs are then written by unit 0 in the same order as before.
#[cube(launch_unchecked)]
fn route_kernel(input: &Array<bf16>, logits: &Array<f32>, bias: &Array<f32>, out: &mut Array<f32>) {
    let t = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    let mut scores = SharedMemory::<f32>::new(EXPERTS);
    let mut bad = SharedMemory::<u32>::new(EXPERTS);
    let mut best_value = SharedMemory::<f32>::new(EXPERTS);
    let mut best_index = SharedMemory::<u32>::new(EXPERTS);
    let mut best_open = SharedMemory::<u32>::new(EXPERTS);
    let mut picked = SharedMemory::<u32>::new(TOP_K + 1);
    let mut cutoff = SharedMemory::<f32>::new(TOP_K + 1);
    let v = logits[t * EXPERTS + u];
    let score = 1.0f32 / (1.0f32 + Exp::exp(-v));
    let biased = score + bias[u];
    let mut invalid = !finite(v) || !finite(bias[u]) || !finite(score) || !finite(biased);
    for i in 0..8usize {
        invalid = invalid || !finite(f32::cast_from(input[t * HIDDEN + u + i * EXPERTS]));
    }
    bad[u] = select(invalid, 1u32, 0u32);
    scores[u] = score;
    let mut open = 1u32;
    for j in 0..TOP_K + 1 {
        best_value[u] = biased;
        best_index[u] = u as u32;
        best_open[u] = open;
        sync_cube();
        let mut step = 256usize; // EXPERTS / 2, as a runtime variable
        while step > 0usize {
            if u < step {
                let (a_open, b_open) = (best_open[u], best_open[u + step]);
                let (a, b) = (best_value[u], best_value[u + step]);
                let (a_index, b_index) = (best_index[u], best_index[u + step]);
                let take = b_open > a_open
                    || (b_open == a_open && (b > a || (b == a && b_index < a_index)));
                if take {
                    best_value[u] = b;
                    best_index[u] = b_index;
                    best_open[u] = b_open;
                }
            }
            sync_cube();
            step /= 2usize;
        }
        let winner = best_index[0usize];
        if u == 0usize {
            picked[j] = winner;
            cutoff[j] = best_value[0usize];
        }
        if winner == u as u32 {
            open = 0u32;
        }
        sync_cube();
    }
    if u == 0usize {
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
            let mut sum = 0.0f32;
            for j in 0..TOP_K {
                sum += scores[picked[j] as usize];
            }
            out[base + 2 * TOP_K] = cutoff[TOP_K - 1];
            out[base + 2 * TOP_K + 1] = cutoff[TOP_K];
            let denominator = sum + 1e-20f32;
            for j in 0..TOP_K {
                out[base + j] = f32::cast_from(picked[j]);
                out[base + TOP_K + j] = (scores[picked[j] as usize] / denominator) * 5.0f32;
            }
            out[base + ROUTE_WIDTH - 1] = 0.0f32;
        }
    }
}
/// The original one-thread selection, kept as the test oracle for
/// route_kernel: both must write byte-identical route records.
#[cfg(test)]
#[cube(launch_unchecked)]
fn route_kernel_serial(input: &Array<bf16>, logits: &Array<f32>, bias: &Array<f32>, out: &mut Array<f32>) {
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
