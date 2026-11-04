//! Known-prompt layer-major execution inside the ordinary Session. Numeric
//! scratch only: no model catalogue, scheduler, independent stream or runner.
use super::*;

// Four pages cover the supported 1M-token window. A full hidden tensor would
// have 2^32 elements; U32 CubeCL metadata must never receive that extent.
pub(super) const PAGE_ROWS: usize = 262_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillOutcome { Completed, Cancelled }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillAllocation {
    pub tokens: usize,
    pub layer_major: bool,
    pub hidden_pages: usize,
    pub hidden_bytes: usize,
    pub largest_hidden_page_bytes: usize,
    pub latent_bytes: usize,
    pub accumulator_bytes: usize,
    pub tile_workspace_bytes: usize,
    pub pending_expert_bytes: usize,
    pub device_working_bytes: usize,
    pub host_routing_bytes: usize,
    pub host_tile_bytes: usize,
}

fn payload(rows: usize, cols: usize, width: usize) -> Result<usize> {
    let elements = rows.checked_mul(cols).filter(|&n| rows > 0 && n <= u32::MAX as usize)
        .ok_or("prefill empty/overflowing U32 element extent")?;
    elements.checked_mul(width).ok_or("prefill byte extent overflow".into())
}

/// CPU-only allocation inventory for the SAME path used by prefill_ids.
/// The caller's existing max_chunk_bytes is a per-call transient allowance.
/// Host route bytes are reported separately, never charged to that device sum.
pub fn admission(c: &TextConfig, l: Limits, start: usize, tokens: usize) -> Result<PrefillAllocation> {
    validate_config(c, l)?;
    let end = start.checked_add(tokens).filter(|&n| tokens > 0 && n <= l.capacity)
        .ok_or("prefill empty suffix or capacity exceeded")?;
    if tokens <= MAX_CHUNK {
        let working = planned_chunk_bytes(c, l, start, tokens)?;
        return Ok(PrefillAllocation {
            tokens, layer_major: false, hidden_pages: 0, hidden_bytes: 0,
            largest_hidden_page_bytes: 0, latent_bytes: 0, accumulator_bytes: 0,
            tile_workspace_bytes: working, pending_expert_bytes: 0,
            device_working_bytes: working, host_routing_bytes: 0, host_tile_bytes: 0,
        });
    }
    let pages = tokens.div_ceil(PAGE_ROWS);
    let page_bytes = payload(tokens.min(PAGE_ROWS), H, 2)?;
    // Sum bytes without ever constructing an unrepresentable whole-hidden
    // ArrayArg. Each actual page is independently checked by payload().
    let hidden = tokens.checked_mul(H).and_then(|n| n.checked_mul(2))
        .ok_or("prefill hidden byte sum overflow")?;
    let has_moe = c.layers_block_type.iter().any(|k| matches!(k, LayerKind::Moe));
    let latent = if has_moe { payload(tokens, moe::LATENT, 2)? } else { 0 };
    let accumulator = if has_moe { payload(tokens, moe::LATENT, 4)? } else { 0 };
    let unbounded = Limits { max_chunk_bytes: usize::MAX, ..l };
    let mut mixer_peak = 0;
    // Attention split count depends on width and endpoint. Check every bounded
    // width, including one-row tails; this never touches model/store/device.
    for width in 1..=MAX_CHUNK {
        for kind in &c.layers_block_type {
            mixer_peak = mixer_peak.max(layer_working_bytes(c, unbounded, end - width, width, kind)?);
        }
    }
    // Copied tile, normalized input, mixer output, residual; tile IDs; the
    // new ordered-update index/weight buffers. The component sums deliberately
    // overcount outputs already included here. No unexplained 128MiB reserve.
    let tile_outer = add(bytes(&[MAX_CHUNK, H], 8)?, bytes(&[MAX_CHUNK], 12)?)?;
    let tile_workspace = add(tile_outer, mixer_peak)?.max(add(
        bytes(&[1, H], 4)?, super::super::fp8::working_bytes(1, [c.vocab_size, H])?,
    )?);
    let expert = if has_moe { moe::admission(moe_limits(unbounded))?.expert_bytes } else { 0 };
    let device = add(add(add(hidden, latent)?, accumulator)?, add(tile_workspace, expert)?)?;
    if device > l.max_chunk_bytes {
        return Err(format!("prefill logical transient budget exceeded: requires {device}, allowance {}", l.max_chunk_bytes));
    }
    // Existing Routing: usize IDs, F32 weights, F64 cutoff margins, plus one
    // dense u64 membership plane per expert. Preallocated, layer-local scratch.
    let (host_routing, host_tile) = if has_moe {
        let row = add(moe::TOP_K * (core::mem::size_of::<usize>() + 4), 8)?;
        let routes = tokens.checked_mul(row).ok_or("prefill route bytes overflow")?;
        let bits = tokens.div_ceil(64).checked_mul(moe::EXPERTS * 8)
            .ok_or("prefill membership bytes overflow")?;
        (add(routes, bits)?, MAX_CHUNK * (row + 188 + 8) + 4096)
    } else { (0, 0) };
    Ok(PrefillAllocation {
        tokens, layer_major: true, hidden_pages: pages, hidden_bytes: hidden,
        largest_hidden_page_bytes: page_bytes, latent_bytes: latent,
        accumulator_bytes: accumulator, tile_workspace_bytes: tile_workspace,
        pending_expert_bytes: expert, device_working_bytes: device,
        host_routing_bytes: host_routing, host_tile_bytes: host_tile,
    })
}

pub(super) struct Pending {
    pages: Vec<CudaTensor>,
    latent: Option<CudaTensor>,
    accumulator: Option<CudaTensor>,
}
impl Pending {
    fn empty() -> Self { Self { pages: Vec::new(), latent: None, accumulator: None } }
}

fn tensor(client: &cubecl::client::ComputeClient<CudaRuntime>, device: &CudaDevice,
    rows: usize, cols: usize, dtype: DType) -> Result<CudaTensor>
{
    let width = match dtype { DType::BF16 => 2, DType::F32 | DType::I32 | DType::U32 => 4,
        _ => return Err("unsupported prefill scratch dtype".into()) };
    let bytes = payload(rows, cols, width)?;
    Ok(CudaTensor::new_contiguous(client.clone(), device.clone(), [1, rows, cols].into(),
        client.empty(bytes), dtype))
}

#[cube(launch_unchecked)]
fn copy_rows_kernel<E: Scalar + Cast>(input: &Array<E>, output: &mut Array<E>,
    input_offset: usize, output_offset: usize, len: usize)
{
    let i = ABSOLUTE_POS as usize;
    if i < len { output[output_offset + i] = input[input_offset + i]; }
}

// Checked copy between complete typed base buffers. No offset alias is passed
// to FP8/attention or any wrapper whose contract rejects one.
fn copy_rows(input: &CudaTensor, output: &CudaTensor, from: usize, to: usize, rows: usize) -> Result<()> {
    let a = input.meta.shape().as_slice();
    let b = output.meta.shape().as_slice();
    if a.len() != 3 || b.len() != 3 || a[0] != 1 || b[0] != 1 || a[2] != b[2]
        || input.dtype != output.dtype || input.device != output.device
        || !std::ptr::eq(input.client.properties(), output.client.properties())
        || from.checked_add(rows).is_none_or(|n| n > a[1])
        || to.checked_add(rows).is_none_or(|n| n > b[1]) || rows == 0
    { return Err("prefill row copy shape/device/range mismatch".into()); }
    let width = match input.dtype { DType::BF16 => 2, DType::F32 => 4,
        _ => return Err("prefill row copy dtype mismatch".into()) };
    payload(a[1], a[2], width)?;
    payload(b[1], b[2], width)?;
    let len = rows * a[2];
    macro_rules! launch { ($ty:ty) => { unsafe {
        copy_rows_kernel::launch_unchecked::<$ty, CudaRuntime>(
            &input.client, CubeCount::new_1d(len.div_ceil(128) as u32), CubeDim::new_1d(128),
            ArrayArg::from_raw_parts(input.handle.clone(), a[1] * a[2]),
            ArrayArg::from_raw_parts(output.handle.clone(), b[1] * b[2]),
            from * a[2], to * b[2], len,
        );
    } }; }
    match input.dtype { DType::BF16 => launch!(bf16), DType::F32 => launch!(f32), _ => unreachable!() }
    Ok(())
}

fn copied_tile(input: &CudaTensor, from: usize, rows: usize, held: &mut Vec<CudaTensor>) -> Result<CudaTensor> {
    let out = tensor(&input.client, &input.device, rows, input.meta.shape().as_slice()[2], input.dtype)?;
    held.push(out.clone());
    copy_rows(input, &out, from, 0, rows)?;
    Ok(out)
}

// Per-layer execution membership, never parameter/storage history. Routing
// weights stay attached to their original token/expert pair. No float atomics.
struct Routes {
    routing: moe::Routing,
    membership: Vec<u64>,
    words: usize,
    capacity: usize,
}
impl Routes {
    fn new(tokens: usize) -> Result<Self> {
        let count = tokens.checked_mul(moe::TOP_K).ok_or("route count overflow")?;
        let mut ids = Vec::new(); let mut weights = Vec::new(); let mut cutoff_margins = Vec::new();
        ids.try_reserve_exact(count).map_err(|e| e.to_string())?;
        weights.try_reserve_exact(count).map_err(|e| e.to_string())?;
        cutoff_margins.try_reserve_exact(tokens).map_err(|e| e.to_string())?;
        let words = tokens.div_ceil(64);
        let len = words.checked_mul(moe::EXPERTS).ok_or("membership count overflow")?;
        let mut membership = Vec::new();
        membership.try_reserve_exact(len).map_err(|e| e.to_string())?;
        membership.resize(len, 0);
        Ok(Self { routing: moe::Routing { tokens: 0, ids, weights, cutoff_margins },
            membership, words, capacity: tokens })
    }
    fn append(&mut self, tile: moe::Routing) -> Result<()> {
        let start = self.routing.tokens;
        if start.checked_add(tile.tokens).is_none_or(|n| n > self.capacity)
            || tile.ids.len() != tile.tokens * moe::TOP_K || tile.weights.len() != tile.ids.len()
            || tile.cutoff_margins.len() != tile.tokens
        { return Err("prefill routing tile extent mismatch".into()); }
        for (row, ids) in tile.ids.chunks_exact(moe::TOP_K).enumerate() {
            let token = start + row;
            for &expert in ids {
                if expert >= moe::EXPERTS { return Err("prefill expert out of range".into()); }
                let word = &mut self.membership[expert * self.words + token / 64];
                let bit = 1u64 << (token % 64);
                if *word & bit != 0 { return Err("prefill duplicate token/expert".into()); }
                *word |= bit;
            }
        }
        self.routing.tokens += tile.tokens;
        self.routing.ids.extend(tile.ids);
        self.routing.weights.extend(tile.weights);
        self.routing.cutoff_margins.extend(tile.cutoff_margins);
        Ok(())
    }
    fn rows(&self, expert: usize) -> impl Iterator<Item = (i32, f32)> + '_ {
        self.membership[expert * self.words..(expert + 1) * self.words].iter().enumerate()
            .flat_map(move |(word, &bits)| (0..64).filter_map(move |bit| {
                if bits & (1u64 << bit) == 0 { return None; }
                let token = word * 64 + bit;
                let base = token * moe::TOP_K;
                let slot = self.routing.ids[base..base + moe::TOP_K].iter()
                    .position(|&id| id == expert).expect("membership came from routing");
                Some((token as i32, self.routing.weights[base + slot]))
            }))
    }
}

struct Progress { last: Option<std::time::Instant> }
impl Progress {
    fn new() -> Self {
        Self { last: tracing::enabled!(target: "mary.nemotron.progress", tracing::Level::INFO)
            .then(std::time::Instant::now) }
    }
    fn completed(&mut self, layer: usize, phase: &'static str, rows: usize,
        tokens: usize, committed: usize, force: bool)
    {
        let Some(last) = &mut self.last else { return; };
        if !force && last.elapsed() < std::time::Duration::from_secs(1) { return; }
        let _step = tracing::info_span!(target: "mary.nemotron.progress", "native_prefill_step",
            layer, phase, stage_rows_completed = rows, span_tokens = tokens,
            committed_before = committed, session_ready = false, fence_ok = true).entered();
        *last = std::time::Instant::now();
    }
}
