//! Known-prompt layer-major execution inside the ordinary Session. Numeric
//! scratch only: no model catalogue, scheduler, independent stream or runner.
use super::*;

// Four pages cover the supported 1M-token window. A full hidden tensor would
// have 2^32 elements; U32 CubeCL metadata must never receive that extent.
pub(super) const PAGE_ROWS: usize = 262_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillOutcome {
    Completed,
    Cancelled,
}

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
    /// Always 0: experts are resident in the model, none is staged per span.
    /// Kept for existing readers.
    pub pending_expert_bytes: usize,
    pub device_working_bytes: usize,
    pub host_routing_bytes: usize,
    pub host_tile_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_host_prefill_progress_is_not_disabled_by_span_only_filter() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        #[derive(Clone, Default)]
        struct Capture(Arc<AtomicUsize>);
        impl tracing::Subscriber for Capture {
            fn enabled(&self, m: &tracing::Metadata<'_>) -> bool {
                m.is_span() && m.target() == "mary.nemotron.progress"
                    && matches!(m.name(), "native_append" | "native_prefill_step")
            }
            fn new_span(&self, a: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                assert_eq!(a.metadata().name(), "native_prefill_step");
                self.0.fetch_add(1, Ordering::SeqCst);
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, _: &tracing::Event<'_>) { panic!("no event admitted") }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }
        let capture = Capture::default();
        tracing::subscriber::with_default(capture.clone(), || {
            assert!(!tracing::enabled!(target: "mary.nemotron.progress", tracing::Level::INFO));
            let mut progress = Progress::new();
            progress.completed(3, "expert_rows", 512, 257, 7, true);
        });
        assert_eq!(capture.0.load(Ordering::SeqCst), 1);
        // Drive's native_timing fixture additionally applies its ACTUAL filter
        // and visitor to this vocabulary, proving durable step->append->turn.
    }

    fn routing(start: usize, tokens: usize) -> moe::Routing {
        let ids: Vec<_> = (start..start + tokens)
            .flat_map(|row| (0..moe::TOP_K).map(move |slot| (row * 3 + slot * 7) % moe::EXPERTS))
            .collect();
        let weights = (0..ids.len())
            .map(|i| (start * moe::TOP_K + i + 1) as f32 / 8192.0)
            .collect();
        moe::Routing {
            tokens,
            ids,
            weights,
            cutoff_margins: vec![0.25; tokens],
        }
    }

    #[test]
    fn session_host_prefill_membership_preserves_token_weights_and_expert_order() {
        let mut routes = Routes::new(257).unwrap();
        routes.append(routing(0, 256)).unwrap();
        routes.append(routing(256, 1)).unwrap();
        let mut selected = vec![Vec::new(); 257];
        for expert in 0..moe::EXPERTS {
            let rows: Vec<_> = routes.rows(expert).collect();
            assert!(rows.windows(2).all(|v| v[0].0 < v[1].0));
            for (token, weight) in rows {
                selected[token as usize].push((expert, weight));
            }
        }
        for (token, selected) in selected.iter().enumerate() {
            let base = token * moe::TOP_K;
            let mut expected: Vec<_> = routes.routing.ids[base..base + moe::TOP_K]
                .iter()
                .copied()
                .zip(
                    routes.routing.weights[base..base + moe::TOP_K]
                        .iter()
                        .copied(),
                )
                .collect();
            expected.sort_by_key(|&(id, _)| id);
            assert_eq!(&expected, selected);
        }
        assert!(routes.append(routing(257, 1)).is_err());
        let mut bad = routing(0, 1);
        bad.ids[1] = bad.ids[0];
        assert!(Routes::new(1).unwrap().append(bad).is_err());
        assert!(Routes::new(0).is_err());
        assert!(Routes::new(i32::MAX as usize + 1).is_err());
    }

    #[test]
    fn session_host_prefill_element_extent_is_distinct_from_byte_extent() {
        assert!(payload(1 << 20, H, 2).is_err());
        assert_eq!(payload(PAGE_ROWS, H, 2).unwrap(), 1 << 31);
        assert_eq!(payload(1 << 20, moe::LATENT, 4).unwrap(), 1usize << 32);
        for (rows, cols, width) in [(0, H, 2), (usize::MAX, H, 2), (1, 1, usize::MAX)] {
            if rows == 1 {
                assert_eq!(payload(rows, cols, width).unwrap(), usize::MAX);
            } else {
                assert!(payload(rows, cols, width).is_err());
            }
        }
    }
}

fn payload(rows: usize, cols: usize, width: usize) -> Result<usize> {
    let elements = rows
        .checked_mul(cols)
        .filter(|&n| rows > 0 && n <= u32::MAX as usize)
        .ok_or("prefill empty/overflowing U32 element extent")?;
    elements
        .checked_mul(width)
        .ok_or("prefill byte extent overflow".into())
}

/// CPU-only allocation inventory for the SAME path used by prefill_ids.
/// The caller's existing max_chunk_bytes is a per-call transient allowance.
/// Host route bytes are reported separately, never charged to that device sum.
pub fn admission(
    c: &TextConfig,
    l: Limits,
    start: usize,
    tokens: usize,
) -> Result<PrefillAllocation> {
    admission_with_prefix(c, l, start, tokens, 0)
}
pub(super) fn admission_with_prefix(c: &TextConfig, l: Limits, start: usize,
    tokens: usize, prefix: usize) -> Result<PrefillAllocation>
{
    validate_config(c, l)?;
    let end = start
        .checked_add(tokens)
        .filter(|&n| tokens > 0 && n <= l.capacity)
        .ok_or("prefill empty suffix or capacity exceeded")?;
    if tokens <= MAX_CHUNK {
        let working = planned_prefix_chunk_bytes(c, l, start, tokens, prefix)?;
        return Ok(PrefillAllocation {
            tokens,
            layer_major: false,
            hidden_pages: 0,
            hidden_bytes: 0,
            largest_hidden_page_bytes: 0,
            latent_bytes: 0,
            accumulator_bytes: 0,
            tile_workspace_bytes: working,
            pending_expert_bytes: 0,
            device_working_bytes: working,
            host_routing_bytes: 0,
            host_tile_bytes: 0,
        });
    }
    let pages = tokens.div_ceil(PAGE_ROWS);
    let page_bytes = payload(tokens.min(PAGE_ROWS), H, 2)?;
    // Sum bytes without ever constructing an unrepresentable whole-hidden
    // ArrayArg. Each actual page is independently checked by payload().
    let hidden = tokens
        .checked_mul(H)
        .and_then(|n| n.checked_mul(2))
        .ok_or("prefill hidden byte sum overflow")?;
    let has_moe = c
        .layers_block_type
        .iter()
        .any(|k| matches!(k, LayerKind::Moe));
    let latent = if has_moe {
        payload(tokens, moe::LATENT, 2)?
    } else {
        0
    };
    let accumulator = if has_moe {
        payload(tokens, moe::LATENT, 4)?
    } else {
        0
    };
    let unbounded = Limits {
        max_chunk_bytes: usize::MAX,
        ..l
    };
    let mut mixer_peak = 0;
    // Attention split count depends on width and endpoint. Check every bounded
    // width, including one-row tails; this never touches model/store/device.
    for width in 1..=MAX_CHUNK {
        for kind in &c.layers_block_type {
            mixer_peak =
                mixer_peak.max(layer_prefix_working_bytes(c, unbounded, end - width, width, kind, prefix)?);
        }
    }
    // Preserve the existing conservative width envelope, but do not assume
    // split rounding is monotone for prefix/private runs. Check every actual
    // attention tile endpoint, including the B4/B8 boundary.
    if c.layers_block_type.iter().any(|kind| matches!(kind, LayerKind::Attention)) {
        for offset in (0..tokens).step_by(MAX_CHUNK) {
            let width = (tokens - offset).min(MAX_CHUNK);
            mixer_peak = mixer_peak.max(layer_prefix_working_bytes(c, unbounded,
                start + offset, width, &LayerKind::Attention, prefix)?);
        }
    }
    // Copied tile, normalized input, mixer output, residual; tile IDs; the
    // new ordered-update index/weight buffers. The component sums deliberately
    // overcount outputs already included here. No unexplained 128MiB reserve.
    let tile_outer = add(bytes(&[MAX_CHUNK, H], 8)?, bytes(&[MAX_CHUNK], 12)?)?;
    let tile_workspace = add(tile_outer, mixer_peak)?.max(add(
        bytes(&[1, H], 6)?,
        super::super::fp8::working_bytes(1, [c.vocab_size, H])?,
    )?);
    let device = add(add(add(hidden, latent)?, accumulator)?, tile_workspace)?;
    if device > l.max_chunk_bytes {
        return Err(format!(
            "prefill logical transient budget exceeded: requires {device}, allowance {}",
            l.max_chunk_bytes
        ));
    }
    // Existing Routing: usize IDs, F32 weights, F64 cutoff margins, plus one
    // dense u64 membership plane per expert. Preallocated, layer-local scratch.
    let (host_routing, host_tile) = if has_moe {
        let row = add(moe::TOP_K * (core::mem::size_of::<usize>() + 4), 8)?;
        let routes = tokens
            .checked_mul(row)
            .ok_or("prefill route bytes overflow")?;
        let bits = tokens
            .div_ceil(64)
            .checked_mul(moe::EXPERTS * 8)
            .ok_or("prefill membership bytes overflow")?;
        // Readback and its decoded F32 copy coexist with bounded Routing.
        // Expert row/index/weight vectors add 16 bytes/row. Fixed slack covers
        // the duplicate bitmap and Vec headers, not general allocator overhead.
        (add(routes, bits)?, MAX_CHUNK * (row + 2 * 188 + 16) + 4096)
    } else {
        (0, 0)
    };
    Ok(PrefillAllocation {
        tokens,
        layer_major: true,
        hidden_pages: pages,
        hidden_bytes: hidden,
        largest_hidden_page_bytes: page_bytes,
        latent_bytes: latent,
        accumulator_bytes: accumulator,
        tile_workspace_bytes: tile_workspace,
        pending_expert_bytes: 0,
        device_working_bytes: device,
        host_routing_bytes: host_routing,
        host_tile_bytes: host_tile,
    })
}

pub(super) struct Pending {
    pages: Vec<CudaTensor>,
    latent: Option<CudaTensor>,
    accumulator: Option<CudaTensor>,
    started: bool,
}
impl Pending {
    fn empty() -> Self {
        Self {
            pages: Vec::new(),
            latent: None,
            accumulator: None,
            started: false,
        }
    }
}

/// Span scratch (hidden pages, latent, accumulator) at its own size: up to
/// 4 GiB at 1M, which the default pool would put in 8 or 32 GB pages.
/// prefill_span gives it back when the span ends.
fn tensor(
    client: &cubecl::client::ComputeClient<CudaRuntime>,
    device: &CudaDevice,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<CudaTensor> {
    let width = match dtype {
        DType::BF16 => 2,
        DType::F32 | DType::I32 | DType::U32 => 4,
        _ => return Err("unsupported prefill scratch dtype".into()),
    };
    let bytes = payload(rows, cols, width)?;
    Ok(CudaTensor::new_contiguous(
        client.clone(),
        device.clone(),
        [1, rows, cols].into(),
        moe::exact(client, || client.empty(bytes)),
        dtype,
    ))
}

#[cube(launch_unchecked)]
fn copy_rows_kernel<E: Scalar + Cast>(
    input: &Array<E>,
    output: &mut Array<E>,
    input_offset: usize,
    output_offset: usize,
    len: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < len {
        output[output_offset + i] = input[input_offset + i];
    }
}

// Checked copy between complete typed base buffers. No offset alias is passed
// to FP8/attention or any wrapper whose contract rejects one.
fn copy_rows(
    input: &CudaTensor,
    output: &CudaTensor,
    from: usize,
    to: usize,
    rows: usize,
) -> Result<()> {
    let a = input.meta.shape().as_slice();
    let b = output.meta.shape().as_slice();
    if a.len() != 3
        || b.len() != 3
        || a[0] != 1
        || b[0] != 1
        || a[2] != b[2]
        || input.dtype != output.dtype
        || input.device != output.device
        || !std::ptr::eq(input.client.properties(), output.client.properties())
        || from.checked_add(rows).is_none_or(|n| n > a[1])
        || to.checked_add(rows).is_none_or(|n| n > b[1])
        || rows == 0
    {
        return Err("prefill row copy shape/device/range mismatch".into());
    }
    let width = match input.dtype {
        DType::BF16 => 2,
        DType::F32 => 4,
        _ => return Err("prefill row copy dtype mismatch".into()),
    };
    payload(a[1], a[2], width)?;
    payload(b[1], b[2], width)?;
    for value in [input, output] {
        let shape = value.meta.shape().as_slice();
        let strides = value.meta.strides();
        let required = payload(shape[1], shape[2], width)? as u64;
        let end = value.handle.offset_end.unwrap_or(0);
        if value.qparams.is_some()
            || value.handle.offset_start.unwrap_or(0) != 0
            || end > value.handle.size()
            || value.handle.size() - end < required
            || strides.len() != 3
            || strides[2] != 1
            || (shape[1] > 1 && strides[1] != shape[2])
        {
            return Err("prefill row copy requires covered contiguous base buffers".into());
        }
    }
    let len = rows * a[2];
    macro_rules! launch {
        ($ty:ty) => {
            unsafe {
                copy_rows_kernel::launch_unchecked::<$ty, CudaRuntime>(
                    &input.client,
                    CubeCount::new_1d(len.div_ceil(128) as u32),
                    CubeDim::new_1d(128),
                    ArrayArg::from_raw_parts(input.handle.clone(), a[1] * a[2]),
                    ArrayArg::from_raw_parts(output.handle.clone(), b[1] * b[2]),
                    from * a[2],
                    to * b[2],
                    len,
                );
            }
        };
    }
    match input.dtype {
        DType::BF16 => launch!(bf16),
        DType::F32 => launch!(f32),
        _ => unreachable!(),
    }
    Ok(())
}

fn copied_tile(
    input: &CudaTensor,
    from: usize,
    rows: usize,
    held: &mut Vec<CudaTensor>,
) -> Result<CudaTensor> {
    let out = tensor(
        &input.client,
        &input.device,
        rows,
        input.meta.shape().as_slice()[2],
        input.dtype,
    )?;
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
        if tokens == 0 || tokens > i32::MAX as usize {
            return Err("route row index exceeds I32".into());
        }
        let count = tokens
            .checked_mul(moe::TOP_K)
            .ok_or("route count overflow")?;
        let mut ids = Vec::new();
        let mut weights = Vec::new();
        let mut cutoff_margins = Vec::new();
        ids.try_reserve_exact(count).map_err(|e| e.to_string())?;
        weights
            .try_reserve_exact(count)
            .map_err(|e| e.to_string())?;
        cutoff_margins
            .try_reserve_exact(tokens)
            .map_err(|e| e.to_string())?;
        let words = tokens.div_ceil(64);
        let len = words
            .checked_mul(moe::EXPERTS)
            .ok_or("membership count overflow")?;
        let mut membership = Vec::new();
        membership
            .try_reserve_exact(len)
            .map_err(|e| e.to_string())?;
        membership.resize(len, 0);
        Ok(Self {
            routing: moe::Routing {
                tokens: 0,
                ids,
                weights,
                cutoff_margins,
            },
            membership,
            words,
            capacity: tokens,
        })
    }
    fn append(&mut self, tile: moe::Routing) -> Result<()> {
        let start = self.routing.tokens;
        if start
            .checked_add(tile.tokens)
            .is_none_or(|n| n > self.capacity)
            || tile.ids.len() != tile.tokens * moe::TOP_K
            || tile.weights.len() != tile.ids.len()
            || tile.cutoff_margins.len() != tile.tokens
        {
            return Err("prefill routing tile extent mismatch".into());
        }
        for (row, ids) in tile.ids.chunks_exact(moe::TOP_K).enumerate() {
            let token = start + row;
            for &expert in ids {
                if expert >= moe::EXPERTS {
                    return Err("prefill expert out of range".into());
                }
                let word = &mut self.membership[expert * self.words + token / 64];
                let bit = 1u64 << (token % 64);
                if *word & bit != 0 {
                    return Err("prefill duplicate token/expert".into());
                }
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
        self.membership[expert * self.words..(expert + 1) * self.words]
            .iter()
            .enumerate()
            .flat_map(move |(word, &bits)| {
                (0..64).filter_map(move |bit| {
                    if bits & (1u64 << bit) == 0 {
                        return None;
                    }
                    let token = word * 64 + bit;
                    let base = token * moe::TOP_K;
                    let slot = self.routing.ids[base..base + moe::TOP_K]
                        .iter()
                        .position(|&id| id == expert)
                        .expect("membership came from routing");
                    Some((token as i32, self.routing.weights[base + slot]))
                })
            })
    }
}

struct Progress {
    last: std::time::Instant,
}
impl Progress {
    fn new() -> Self {
        // The ordinary sink filters spans, not events: tracing::enabled! is
        // an event probe and would falsely disable this progress path.
        Self {
            last: std::time::Instant::now(),
        }
    }
    fn completed(
        &mut self,
        layer: usize,
        phase: &'static str,
        rows: usize,
        tokens: usize,
        committed: usize,
        force: bool,
    ) {
        #[cfg(feature = "nemotron-prefill-timing")]
        crate::models::nemotron_h::prefill_timing::mark(layer, phase, rows);
        if !force && self.last.elapsed() < std::time::Duration::from_secs(1) {
            return;
        }
        let _step = tracing::info_span!(target: "mary.nemotron.progress", "native_prefill_step",
            layer, phase, stage_rows_completed = rows, span_tokens = tokens,
            span_end = committed + tokens,
            committed_before = committed, session_ready = false, fence_ok = true)
        .entered();
        self.last = std::time::Instant::now();
    }
}

#[cube(launch_unchecked)]
fn zero_rows(output: &mut Array<f32>, offset: usize, len: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < len {
        output[offset + i] = 0.0f32;
    }
}

impl<S: BlobStoreGet> Session<S> {
    /// Append the known prompt suffix by layers. No other operation can use a
    /// partly advanced Session. Cancellation is checked between fenced tiles;
    /// no single CUDA call/source read/fence has a host-time bound here.
    pub fn prefill_ids(
        &mut self,
        ids: &[u32],
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<PrefillOutcome> {
        // Opt-in bracket around this whole call; the nested call below runs
        // the unchanged body (the probe refuses to bracket twice).
        #[cfg(feature = "nemotron-prefill-timing")]
        {
            use crate::models::nemotron_h::prefill_timing::{Kind, Probe};
            let kinds = || self.layers.iter().map(|l| match &l.mixer {
                Mixer::Mamba(_) => Kind::Mamba,
                Mixer::Attention(_) => Kind::Attention,
                Mixer::Moe(_) => Kind::Moe,
            }).collect();
            if let Some(probe) = Probe::requested(ids.len(), self.committed.len(), kinds) {
                let mut probe = probe?;
                let synced = std::time::Instant::now();
                self.stream.executes(|| self.fence())?;
                probe.start(synced.elapsed())?;
                let result = self.prefill_ids(ids, cancelled);
                probe.finish(match &result {
                    Ok(PrefillOutcome::Completed) => "completed",
                    Ok(PrefillOutcome::Cancelled) => "cancelled",
                    Err(_) => "error",
                }, self.committed.len(), self.status == Status::Ready);
                return result;
            }
        }
        if let Some(seed) = self.seed.clone() {
            let start = self.committed.len();
            let count = seed.len().checked_sub(start).ok_or("prefix seed position exceeded")?.min(ids.len());
            if self.status != Status::Ready || ids.is_empty()
                || ids.iter().any(|&id| id as usize >= self.vocab_size())
                || seed.get(start..start + count) != Some(&ids[..count]) {
                return Err("invalid prefill or prepared prefix mismatch".into());
            }
            // Admit BOTH actual spans before changing either. The suffix uses
            // two key runs even though its prefix has not yet been published.
            if count > 0 { admission(&self.model.config, self.limits, start, count)?; }
            if count < ids.len() {
                admission_with_prefix(&self.model.config, self.limits, seed.len(), ids.len() - count, seed.len())?;
            }
            if cancelled() { return Ok(PrefillOutcome::Cancelled); }
            if count > 0 && self.prefill_span(&ids[..count], cancelled)? == PrefillOutcome::Cancelled {
                return Ok(PrefillOutcome::Cancelled);
            }
            self.publish_seed_if_complete()?;
            // A completed long seed must finish the same ownership transition
            // as short append before cancellation can expose Ready. Otherwise
            // exact-seed reuse can skip prefill but still lack suffix storage.
            // Cancellation remains Cancelled; publication describes only P.
            if cancelled() { return Ok(PrefillOutcome::Cancelled); }
            if count == ids.len() { return Ok(PrefillOutcome::Completed); }
            return self.prefill_span(&ids[count..], cancelled);
        }
        self.prefill_span(ids, cancelled)
    }

    fn prefill_span(&mut self, ids: &[u32], cancelled: &mut dyn FnMut() -> bool) -> Result<PrefillOutcome> {
        if self.status != Status::Ready
            || self.span_pending.is_some()
            || ids
                .iter()
                .any(|&id| id as usize >= self.model.config.vocab_size)
        {
            return Err("prefill unavailable session or invalid token ID".into());
        }
        let plan = admission_with_prefix(
            &self.model.config,
            self.limits,
            self.committed.len(),
            ids.len(),
            self.prefix.as_ref().map_or(0, |p| p.ids.len()),
        )?;
        self.check_boundary(self.committed.len())?;
        if cancelled() {
            return Ok(PrefillOutcome::Cancelled);
        }
        if !plan.layer_major {
            self.append_ids(ids)?;
            return Ok(PrefillOutcome::Completed);
        }
        self.status = Status::Poisoned;
        self.span_pending = Some(Pending::empty());
        let result = self.stream.executes(|| self.run_prefill(ids, cancelled));
        let outcome = self.finish_span(ids, result);
        // Exact scratch is only reused at an equal size; free what was dropped.
        self.stream.executes(|| self.model.client.memory_cleanup());
        outcome
    }

    fn finish_span(&mut self, ids: &[u32], result: Result<Option<Vec<f32>>>) -> Result<PrefillOutcome> {
        match result {
            Ok(Some(logits)) => {
                self.committed.extend_from_slice(ids);
                self.logits = logits;
                self.pending.clear();
                self.span_pending = None;
                self.status = Status::Ready;
                Ok(PrefillOutcome::Completed)
            }
            Ok(None) => {
                if !self
                    .span_pending
                    .as_ref()
                    .expect("installed before work")
                    .started
                {
                    // Allocation alone did not change the old prefix. No queued
                    // writes have been allowed before started is set.
                    self.span_pending = None;
                    self.pending.clear();
                    self.status = Status::Ready;
                } else {
                    self.invalidate();
                }
                Ok(PrefillOutcome::Cancelled)
            }
            Err(error) => {
                self.invalidate();
                Err(error)
            }
        }
    }

    fn run_prefill(
        &mut self,
        ids: &[u32],
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Option<Vec<f32>>> {
        let tokens = ids.len();
        let start = self.committed.len();
        let mut progress = Progress::new();
        for page_start in (0..tokens).step_by(PAGE_ROWS) {
            if cancelled() {
                return Ok(None);
            }
            let rows = (tokens - page_start).min(PAGE_ROWS);
            let page = tensor(&self.model.client, &self.model.device, rows, H, DType::BF16)?;
            self.span_pending.as_mut().unwrap().pages.push(page);
        }
        for offset in (0..tokens).step_by(MAX_CHUNK) {
            if cancelled() {
                return Ok(None);
            }
            let rows = (tokens - offset).min(MAX_CHUNK);
            self.span_pending.as_mut().unwrap().started = true;
            let ids_handle = self
                .model
                .client
                .create_from_slice(u32::as_bytes(&ids[offset..offset + rows]));
            self.pending.push(CudaTensor::new_contiguous(
                self.model.client.clone(),
                self.model.device.clone(),
                [rows].into(),
                ids_handle.clone(),
                DType::U32,
            ));
            let embedded = empty(&self.model.client, &self.model.device, rows, H);
            self.pending.push(embedded.clone());
            unsafe {
                embedding_kernel::launch_unchecked::<CudaRuntime>(
                    &self.model.client,
                    CubeCount::new_1d((rows * H).div_ceil(128) as u32),
                    CubeDim::new_1d(128),
                    ArrayArg::from_raw_parts(ids_handle, rows),
                    ArrayArg::from_raw_parts(
                        self.model.embedding.handle.clone(),
                        self.model.config.vocab_size * H,
                    ),
                    ArrayArg::from_raw_parts(embedded.handle.clone(), rows * H),
                    rows * H,
                );
            }
            let page = &self.span_pending.as_ref().unwrap().pages[offset / PAGE_ROWS];
            copy_rows(&embedded, page, 0, offset % PAGE_ROWS, rows)?;
            self.fence()?;
            self.pending.clear();
            progress.completed(
                0,
                "embedding",
                offset + rows,
                tokens,
                start,
                offset + rows == tokens,
            );
        }
        for index in 0..self.layers.len() {
            if cancelled() {
                return Ok(None);
            }
            if matches!(&self.layers[index].mixer, Mixer::Moe(_)) {
                if !self.prefill_moe(index, tokens, start, cancelled, &mut progress)? {
                    return Ok(None);
                }
            } else {
                for offset in (0..tokens).step_by(MAX_CHUNK) {
                    if cancelled() {
                        return Ok(None);
                    }
                    let rows = (tokens - offset).min(MAX_CHUNK);
                    let page =
                        self.span_pending.as_ref().unwrap().pages[offset / PAGE_ROWS].clone();
                    let input = copied_tile(&page, offset % PAGE_ROWS, rows, &mut self.pending)?;
                    let output = enqueue_layer_tile(
                        &mut self.layers[index],
                        &input,
                        start + offset,
                        self.limits,
                        self.model.config.layer_norm_epsilon as f32,
                        &mut self.pending,
                        #[cfg(feature = "nemotron-timing")]
                        None,
                    )?;
                    copy_rows(&output, &page, 0, offset % PAGE_ROWS, rows)?;
                    // Includes writeback. Advance PRIVATE state only after
                    // successful completion; the Session remains Poisoned.
                    self.fence()?;
                    match &mut self.layers[index].mixer {
                        Mixer::Attention(a) => a.commit()?,
                        Mixer::Mamba(m) => {
                            m.committed = m.staged.take().map(Arc::new);
                        }
                        Mixer::Moe(_) => unreachable!(),
                    }
                    self.pending.clear();
                    progress.completed(
                        index,
                        "stateful",
                        offset + rows,
                        tokens,
                        start,
                        offset + rows == tokens,
                    );
                    if cancelled() {
                        return Ok(None);
                    }
                }
            }
            #[cfg(test)]
            {
                inject(self.fault, Fault::AfterLayer(index))?;
                if self.fault == Some(Fault::PanicAfterLayer(index)) {
                    panic!("injected span unwind");
                }
            }
        }
        if cancelled() {
            return Ok(None);
        }
        self.check_boundary(start + tokens)?;
        let last = tokens - 1;
        let page = self.span_pending.as_ref().unwrap().pages[last / PAGE_ROWS].clone();
        let row = copied_tile(&page, last % PAGE_ROWS, 1, &mut self.pending)?;
        let logits = self.project_last_logits(&row)?;
        self.fence()?;
        #[cfg(feature = "nemotron-prefill-timing")]
        crate::models::nemotron_h::prefill_timing::mark(self.layers.len(), "lm_head", tokens);
        if cancelled() {
            return Ok(None);
        }
        Ok(Some(logits))
    }

    fn prefill_moe(
        &mut self,
        index: usize,
        tokens: usize,
        start: usize,
        cancelled: &mut dyn FnMut() -> bool,
        progress: &mut Progress,
    ) -> Result<bool> {
        let mut routes = Routes::new(tokens)?;
        if let Mixer::Moe(m) = &mut self.layers[index].mixer {
            m.prefill_begin()?;
        }
        let latent = tensor(
            &self.model.client,
            &self.model.device,
            tokens,
            moe::LATENT,
            DType::BF16,
        )?;
        self.span_pending.as_mut().unwrap().latent = Some(latent.clone());
        for offset in (0..tokens).step_by(MAX_CHUNK) {
            if cancelled() {
                return Ok(false);
            }
            let rows = (tokens - offset).min(MAX_CHUNK);
            let page = self.span_pending.as_ref().unwrap().pages[offset / PAGE_ROWS].clone();
            let input = copied_tile(&page, offset % PAGE_ROWS, rows, &mut self.pending)?;
            let norm = normalized(
                &input,
                &self.layers[index].norm,
                self.model.config.layer_norm_epsilon as f32,
            );
            self.pending.push(norm.clone());
            let (routing, tile) = match &self.layers[index].mixer {
                Mixer::Moe(m) => m.prefill_route(&norm, &mut self.pending)?,
                _ => unreachable!(),
            };
            routes.append(routing)?;
            copy_rows(&tile, &latent, 0, offset, rows)?;
            self.fence()?;
            self.pending.clear();
            progress.completed(
                index,
                "route",
                offset + rows,
                tokens,
                start,
                offset + rows == tokens,
            );
        }
        let accumulator = tensor(
            &self.model.client,
            &self.model.device,
            tokens,
            moe::LATENT,
            DType::F32,
        )?;
        self.span_pending.as_mut().unwrap().accumulator = Some(accumulator.clone());
        for offset in (0..tokens).step_by(MAX_CHUNK) {
            if cancelled() {
                return Ok(false);
            }
            let len = (tokens - offset).min(MAX_CHUNK) * moe::LATENT;
            unsafe {
                zero_rows::launch_unchecked::<CudaRuntime>(
                    &self.model.client,
                    CubeCount::new_1d(len.div_ceil(128) as u32),
                    CubeDim::new_1d(128),
                    ArrayArg::from_raw_parts(accumulator.handle.clone(), tokens * moe::LATENT),
                    offset * moe::LATENT,
                    len,
                );
            }
            self.fence()?;
            progress.completed(
                index,
                "accumulator_zero",
                offset + len / moe::LATENT,
                tokens,
                start,
                false,
            );
        }
        let mut completed_rows = 0;
        for expert in 0..moe::EXPERTS {
            if cancelled() {
                return Ok(false);
            }
            let mut rows = routes.rows(expert).peekable();
            while rows.peek().is_some() {
                if cancelled() {
                    return Ok(false);
                }
                let tile: Vec<_> = rows.by_ref().take(MAX_CHUNK).collect();
                #[cfg(feature = "nemotron-prefill-timing")]
                let enqueue = crate::models::nemotron_h::prefill_timing::HostSpan::enqueue();
                if let Mixer::Moe(m) = &self.layers[index].mixer {
                    m.prefill_expert_tile(expert, &latent, &accumulator, &tile, &mut self.pending)?;
                }
                #[cfg(feature = "nemotron-prefill-timing")]
                enqueue.done(true);
                #[cfg(feature = "nemotron-prefill-timing")]
                {
                    let fence = crate::models::nemotron_h::prefill_timing::HostSpan::fence();
                    let completion = self.fence();
                    fence.done(completion.is_ok());
                    completion?;
                }
                #[cfg(not(feature = "nemotron-prefill-timing"))]
                self.fence()?;
                self.pending.clear();
                completed_rows += tile.len();
                progress.completed(index, "expert_rows", completed_rows, tokens, start, false);
                if cancelled() {
                    return Ok(false);
                }
            }
        }
        if completed_rows != tokens * moe::TOP_K {
            return Err("prefill incomplete routed contributions".into());
        }
        progress.completed(index, "expert_rows", completed_rows, tokens, start, true);
        // All experts and reads of latent have completed. Reclaim this phase,
        // rather than retaining every successful layer's scratch until E.
        drop(routes);
        drop(latent);
        self.span_pending.as_mut().unwrap().latent = None;
        for offset in (0..tokens).step_by(MAX_CHUNK) {
            if cancelled() {
                return Ok(false);
            }
            let rows = (tokens - offset).min(MAX_CHUNK);
            let page = self.span_pending.as_ref().unwrap().pages[offset / PAGE_ROWS].clone();
            let input = copied_tile(&page, offset % PAGE_ROWS, rows, &mut self.pending)?;
            let norm = normalized(
                &input,
                &self.layers[index].norm,
                self.model.config.layer_norm_epsilon as f32,
            );
            self.pending.push(norm.clone());
            let combined = copied_tile(&accumulator, offset, rows, &mut self.pending)?;
            let output = match &self.layers[index].mixer {
                Mixer::Moe(m) => m.prefill_finish(&norm, &combined, &mut self.pending)?,
                _ => unreachable!(),
            };
            let output = residual(&input, &output);
            self.pending.push(output.clone());
            copy_rows(&output, &page, 0, offset % PAGE_ROWS, rows)?;
            self.fence()?;
            self.pending.clear();
            progress.completed(
                index,
                "finish",
                offset + rows,
                tokens,
                start,
                offset + rows == tokens,
            );
            if cancelled() {
                return Ok(false);
            }
        }
        drop(accumulator);
        self.span_pending.as_mut().unwrap().accumulator = None;
        if let Mixer::Moe(m) = &mut self.layers[index].mixer {
            m.prefill_done();
        }
        Ok(true)
    }
}
