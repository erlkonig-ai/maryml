//! Inkling experts as pile blobs.
//!
//! The checkpoint stores one expert matrix as five separate safetensors entries
//! bound only by a naming convention — `w13_weight`, `.scale`, `.scale2`,
//! `.input_amax`, `.original_shape`. Nothing makes that bundle atomic, and a
//! reader holding only the first has bytes it cannot interpret: the packed
//! shape says 2048 where the tensor is 4096 wide, and the truth lives in a
//! sixth place the reader has to know to consult.
//!
//! A [`Tensor<NVFP4, 2>`] blob is the bundle made atomic. One content-addressed
//! handle carries the codes, the block scales and the global scale together,
//! with the LOGICAL dimensions in its header — so `original_shape` has nothing
//! left to correct.
//!
//! # Why per expert
//!
//! The stacked form is 256 independent expert matrices in one array: data,
//! block scales and `scale2` all slice cleanly on the outermost dimension.
//! Storing them separately is what makes a checkpoint shareable — a node
//! fetches the experts it holds rather than a 12 GiB slab it either has or does
//! not, deduplication works per expert, and a layer split across two machines
//! becomes a partition of blob handles rather than a file format problem.
//!
//! # Payload layout
//!
//! `[codes][block scales][global scale]`, each contiguous, in that order.
//! Lengths are implied by the header's dims and the element format, so nothing
//! records the boundaries: `codes` is `elems / 2`, `scales` is `elems / 16`,
//! and the global scale is the trailing four bytes.

use anybytes::Bytes;
use anyhow::{Context, Result};
use triblespace::core::blob::encodings::tensor::{
    Tensor, TensorElement, TensorView,
    elements::{BF16, F32, NVFP4, NVFP4_BLOCK},
    tensor_blob,
};
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::prelude::BlobStoreGet;
use triblespace::prelude::Id;

use super::load::PackedExpert;

/// Startup weight backing. Experimental placement changes no tensor layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightStorage {
    #[default]
    Host,
    HostReadOnly,
    CudaManaged,
}

impl WeightStorage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::HostReadOnly => "host-read-only",
            Self::CudaManaged => "cuda-managed",
        }
    }

    pub const fn inference_only(self) -> bool {
        !matches!(self, Self::Host)
    }
}

fn validate_managed_concurrent_access(concurrent: i32) -> Result<()> {
    anyhow::ensure!(
        concurrent != 0,
        "cuda-managed weights require concurrent managed access: CPU metadata reads may overlap GPU work"
    );
    Ok(())
}

const MIN_ZERO_BYTES_PER_WORKER: usize = 1 << 20;
const MAX_ZERO_WORKERS: usize = 64;

fn weight_zero_workers(len: usize, requested: usize) -> usize {
    if len == 0 {
        return 0;
    }
    requested
        .max(1)
        .min(MAX_ZERO_WORKERS)
        .min((len / MIN_ZERO_BYTES_PER_WORKER).max(1))
}

/// Initialize the whole allocation without ever treating its previous contents
/// as initialized bytes. Disjoint coarse ranges bound thread overhead; scoped
/// threads all join before return, including unwinding after a spawn/worker
/// panic. The allocation owner therefore remains alive for every worker.
///
/// This changes scheduling only: it still writes every byte before the copy
/// workers overwrite payloads, and does not change eager first-touch residency.
fn initialize_weight_bytes(bytes: &mut [std::mem::MaybeUninit<u8>], requested: usize) -> usize {
    fn zero(bytes: &mut [std::mem::MaybeUninit<u8>]) {
        // SAFETY: this uniquely borrowed range is valid for bytes.len() one-byte
        // elements; no previous value is read. A zero bit pattern initializes
        // each u8, including every allocation padding/tail byte.
        unsafe { std::ptr::write_bytes(bytes.as_mut_ptr(), 0, bytes.len()) };
    }

    let workers = weight_zero_workers(bytes.len(), requested);
    if workers <= 1 {
        zero(bytes);
    } else {
        let chunk = bytes.len().div_ceil(workers);
        std::thread::scope(|scope| {
            for range in bytes.chunks_mut(chunk) {
                scope.spawn(move || zero(range));
            }
        });
    }
    workers
}

/// CPU-readable CUDA managed storage, initialized before any u8 slice exists.
/// Its primary context is the same device-0 primary retained by CubeCL.
struct ManagedWeightArena {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    context: Option<std::sync::Arc<cudarc::driver::CudaContext>>,
}

// SAFETY: the allocation never moves. Startup lends only disjoint mutable
// slices to scoped workers; publication occurs after all workers join. Then
// only immutable CPU/GPU reads are allowed (learning is rejected beforehand).
unsafe impl Send for ManagedWeightArena {}
unsafe impl Sync for ManagedWeightArena {}

impl ManagedWeightArena {
    fn new(len: usize, copy_threads: usize) -> Result<Self> {
        anyhow::ensure!(
            len > 0 && len <= isize::MAX as usize,
            "managed weight arena length must be in 1..=isize::MAX"
        );
        let context = cudarc::driver::CudaContext::new(0)
            .context("retain CUDA primary context for managed weights")?;
        validate_managed_concurrent_access(context.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
        ).context("query concurrent managed weight access")?)?;
        let allocated_at = std::time::Instant::now();
        // SAFETY: the retained primary context is current. This returns raw,
        // UNINITIALIZED CPU-readable managed memory, not a Rust byte slice.
        let raw = unsafe {
            cudarc::driver::result::malloc_managed(
                len,
                cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL,
            )
        }
        .with_context(|| format!("allocate {len} bytes of CUDA managed weights"))?;
        let ptr = std::ptr::NonNull::new(raw as *mut u8)
            .context("CUDA returned a null managed weight allocation")?;
        let arena = Self {
            ptr,
            len,
            context: Some(context),
        };
        eprintln!(
            "    managed weight allocation: bytes={len} alloc_seconds={:.6}",
            allocated_at.elapsed().as_secs_f64()
        );
        let started = std::time::Instant::now();
        // SAFETY: the allocation is uniquely owned, len bytes long, and has no
        // GPU alias yet. MaybeUninit permits the allocation's initial contents;
        // no u8 slice or ByteSource is published before every worker joins.
        let uninitialized = unsafe {
            std::slice::from_raw_parts_mut(
                arena.ptr.as_ptr().cast::<std::mem::MaybeUninit<u8>>(),
                len,
            )
        };
        let workers = initialize_weight_bytes(uninitialized, copy_threads);
        eprintln!(
            "    managed weight initialization: bytes={len} zero_seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
        eprintln!(
            "    managed weight initialization workers: requested={copy_threads} active={workers}"
        );
        Ok(arena)
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: new initialized every byte, and self retains its allocation.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: only startup has mutable access, before Bytes/GPU publication.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

// SAFETY: every byte is initialized; moving this owner does not move storage.
// Bytes keeps this same owner alive across all slices and external GPU aliases.
unsafe impl anybytes::ByteSource for ManagedWeightArena {
    type Owner = Self;

    fn as_bytes(&self) -> &[u8] {
        self.as_slice()
    }
    fn get_owner(self) -> Self {
        self
    }
}

impl Drop for ManagedWeightArena {
    fn drop(&mut self) {
        use cudarc::driver::result;
        let Some(context) = self.context.take() else {
            return;
        };
        // An alias can be released on CubeCL's server thread. Never call back
        // into its client here: bind and synchronize the driver directly.
        let previous = match result::ctx::get_current() {
            Ok(previous) => previous,
            Err(error) => {
                std::mem::forget(context);
                eprintln!(
                    "managed weight arena: cannot inspect CUDA context ({error}); leaking allocation and context for safety"
                );
                return;
            }
        };
        let own = context.cu_ctx();
        let released = context
            .bind_to_thread()
            .and_then(|()| result::ctx::synchronize())
            // SAFETY: successful context synchronization proves all submitted
            // GPU work complete; the final owner holds the only free operation.
            .and_then(|()| unsafe { result::memory_free(self.ptr.as_ptr() as u64) });
        match released {
            Ok(()) => drop(context),
            Err(error) => {
                std::mem::forget(context);
                eprintln!(
                    "managed weight arena: CUDA cleanup failed ({error}); leaking allocation and context for safety"
                );
            }
        }
        // CudaContext::drop itself binds its context. Restore a different
        // caller context afterward; never resurrect our own possibly released
        // last primary-context reference.
        if previous != Some(own) {
            // SAFETY: a different context belonged to this thread on entry and
            // has not been released here; null restores no current context.
            if let Err(error) =
                unsafe { result::ctx::set_current(previous.unwrap_or(std::ptr::null_mut())) }
            {
                eprintln!("managed weight arena: cannot restore caller CUDA context: {error}");
            }
        }
    }
}

enum WeightArena {
    Host(memmap2::MmapMut),
    Managed(ManagedWeightArena),
}

impl WeightArena {
    fn new(len: usize, storage: WeightStorage, copy_threads: usize) -> Result<Self> {
        match storage {
            WeightStorage::Host | WeightStorage::HostReadOnly => Ok(Self::Host(
                memmap2::MmapMut::map_anon(len)
                    .with_context(|| format!("map {len} bytes for the startup weight arena"))?,
            )),
            WeightStorage::CudaManaged => {
                Ok(Self::Managed(ManagedWeightArena::new(len, copy_threads)?))
            }
        }
    }

    fn finish(self, storage: WeightStorage) -> Result<Bytes> {
        match (self, storage) {
            (Self::Host(arena), WeightStorage::Host) => finish_weight_arena(arena, false),
            (Self::Host(arena), WeightStorage::HostReadOnly) => finish_weight_arena(arena, true),
            (Self::Managed(arena), WeightStorage::CudaManaged) => Ok(Bytes::from_source(arena)),
            _ => anyhow::bail!("weight arena backing differs from its admitted storage"),
        }
    }
}

impl std::ops::Deref for WeightArena {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Host(arena) => arena,
            Self::Managed(arena) => arena.as_slice(),
        }
    }
}

impl std::ops::DerefMut for WeightArena {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Host(arena) => arena,
            Self::Managed(arena) => arena.as_mut_slice(),
        }
    }
}

/// Finish a dedicated anonymous allocation before publishing any byte views
/// or GPU aliases. Both arms have the same lazy mmap backing; only protection
/// differs. The caller must have joined every startup copy/swizzle worker.
fn finish_weight_arena(arena: memmap2::MmapMut, read_only: bool) -> Result<Bytes> {
    if read_only {
        Ok(Bytes::from_source(
            arena
                .make_read_only()
                .context("seal inference weight arena read-only")?,
        ))
    } else {
        Ok(Bytes::from_source(arena))
    }
}

/// Keep the existing byte owner alive without assuming its concrete allocator.
/// Cloning Bytes retains its mapping; it does not duplicate the weight payload.
fn weight_arena_mapping(
    bytes: &Bytes,
) -> (
    usize,
    usize,
    std::sync::Arc<dyn std::any::Any + Send + Sync>,
) {
    (
        bytes.as_ptr() as usize,
        bytes.len(),
        std::sync::Arc::new(bytes.clone()),
    )
}

/// One line of the kernel's memory accounting, for a startup that competes with
/// the GPU for ONE pool.
///
/// On a unified-memory part the anonymous startup copy, the pile's page cache
/// and everything CUDA reserves are the same 121 GiB. A timer cannot say which
/// of them ran out, and `MemAvailable` alone cannot either: clean page cache
/// counts as available right up to the moment something needs it and the
/// reclaim has to happen synchronously. So print all four.
pub fn mem_line(label: &str) -> String {
    let mut free = 0u64;
    let mut avail = 0u64;
    let mut cached = 0u64;
    let mut anon = 0u64;
    let mut swapfree = 0u64;
    let mut swaptotal = 0u64;
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let kb = |l: &str| -> u64 {
            l.split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        };
        for l in s.lines() {
            if l.starts_with("MemFree:") {
                free = kb(l);
            } else if l.starts_with("MemAvailable:") {
                avail = kb(l);
            } else if l.starts_with("Cached:") {
                cached = kb(l);
            } else if l.starts_with("AnonPages:") {
                anon = kb(l);
            } else if l.starts_with("SwapFree:") {
                swapfree = kb(l);
            } else if l.starts_with("SwapTotal:") {
                swaptotal = kb(l);
            }
        }
    }
    let mut rss = 0u64;
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for l in s.lines() {
            if l.starts_with("VmRSS:") {
                rss = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            }
        }
    }
    let g = |kb: u64| kb as f64 / (1u64 << 20) as f64;
    format!(
        "    mem[{label}]: rss {:.1} GiB, anon {:.1}, free {:.1}, available {:.1}, cached {:.1}, swap used {:.1}",
        g(rss),
        g(anon),
        g(free),
        g(avail),
        g(cached),
        g(swaptotal.saturating_sub(swapfree)),
    )
}

/// Hands each leaf's source pages back to the kernel once it has been copied
/// into the anonymous arena.
///
/// Two calls, and both are needed, because they undo different things.
/// `madvise(MADV_DONTNEED)` zaps this process's page-table entries, which takes
/// the pages out of RSS but leaves them in the page cache; `posix_fadvise(...,
/// POSIX_FADV_DONTNEED)` drops the cache pages themselves, and it only works on
/// pages nobody has mapped -- which is why the madvise has to come first. The
/// pile is mapped shared and read-only, so neither is destructive: a later read
/// of the same bytes re-faults them from the file.
///
/// What this buys is the thing the startup copy is actually fighting for. On a
/// unified-memory part the source page cache, the anonymous arena and
/// everything CUDA reserves are ONE 121 GiB pool, and reclaim gets to choose
/// between them. At `INK_LAYERS=0:20` it chose wrong twice over: without the
/// madvise it held 36 GiB of source pages in RSS and paged 16 GiB of freshly
/// written arena out to swap instead; with the madvise but without the fadvise
/// it still kept the unmapped cache and started swapping again as soon as the
/// share passed ~85 GiB. Handing the pages back as each leaf lands takes the
/// choice away.
///
/// `INK_RELEASE_SOURCE=0` keeps the pages, as the A/B arm.
struct SourceRelease {
    /// The pile file, reopened read-only for `posix_fadvise` alone. The mapping
    /// does not keep an fd and the advice needs one.
    file: Option<std::fs::File>,
    /// Base address of the pile's mapping, so a payload pointer becomes a file
    /// offset by subtraction. `None` when the mapping is not the whole file, in
    /// which case that subtraction would be a guess.
    map_base: Option<usize>,
    /// The file's length, so the advice never names a range past its end.
    file_len: usize,
    page: usize,
    on: bool,
}

impl SourceRelease {
    #[cfg(target_os = "linux")]
    fn new(path: &std::path::Path, map_base: usize, map_len: usize) -> Self {
        let on = !std::env::var("INK_RELEASE_SOURCE")
            .map(|v| v == "0")
            .unwrap_or(false);
        // SAFETY: `sysconf` reads a static system parameter.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = if page > 0 { page as usize } else { 4096 };
        let file = std::fs::File::open(path).ok();
        // The mapping starts at file offset 0, so a payload pointer minus the
        // base IS the file offset -- but the pile RESERVES address space it has
        // not filled (256 GiB of mapping over a 171 GiB file, so appends need no
        // remap), and the reservation is exactly why this cannot be an equality
        // test. What has to hold is that the file is no LONGER than the map,
        // and that the advice never runs past the file's end.
        let file_len = file
            .as_ref()
            .and_then(|f| f.metadata().ok())
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let whole = file_len > 0 && file_len <= map_len;
        println!(
            "    source release: {}, map {} bytes, file {} bytes, offsets {}",
            if on { "on" } else { "OFF" },
            map_len,
            file_len,
            if whole {
                "usable"
            } else {
                "UNUSABLE (page cache will not be dropped)"
            },
        );
        Self {
            file,
            map_base: whole.then_some(map_base),
            file_len,
            page,
            on,
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn new(_path: &std::path::Path, _map_base: usize, _map_len: usize) -> Self {
        Self {
            file: None,
            map_base: None,
            file_len: 0,
            page: 4096,
            on: false,
        }
    }

    /// Rounded INWARD to whole pages, so a leaf never releases a page its
    /// neighbour is still reading.
    #[cfg(target_os = "linux")]
    fn release(&self, src: &Bytes) {
        use std::os::fd::AsRawFd;
        if !self.on {
            return;
        }
        let lo = src.as_ptr() as usize;
        let hi = (lo + src.len()) / self.page * self.page;
        let lo = lo.next_multiple_of(self.page);
        if hi <= lo {
            return;
        }
        // SAFETY: the range lies inside the pile's live mapping, which outlives
        // this call; `MADV_DONTNEED` on a shared file mapping only zaps
        // page-table entries, so the bytes are still readable and unchanged.
        unsafe {
            libc::madvise(lo as *mut libc::c_void, hi - lo, libc::MADV_DONTNEED);
        }
        if let (Some(base), Some(f)) = (self.map_base, self.file.as_ref()) {
            if hi - base > self.file_len {
                return;
            }
            // SAFETY: `f` is open for the duration of the call; the advice is
            // a hint and cannot invalidate anything.
            unsafe {
                libc::posix_fadvise(
                    f.as_raw_fd(),
                    (lo - base) as libc::off_t,
                    (hi - lo) as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn release(&self, _src: &Bytes) {}
}

/// One GiB.
const GIB: u64 = 1 << 30;

/// Total RAM this machine has, capped by a cgroup limit if one is set.
///
/// `MemTotal`, not `MemAvailable`. The two answer different questions and the
/// admission gate needs both: available says whether the copy can be MADE right
/// now, total says whether the finished process can LIVE. A box whose page
/// cache is warm reports most of it as available -- clean file pages are
/// reclaimable, so they count -- which is exactly how a share that cannot fit
/// gets admitted and then thrashes.
fn mem_total_bytes() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    let kb = status
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
        .context("/proc/meminfo has no numeric MemTotal")?;
    let host = kb.checked_mul(1024).context("MemTotal overflow")?;
    let cgroup = match std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        Ok(max) if max.trim() != "max" => max
            .trim()
            .parse::<u64>()
            .context("parsing cgroup memory.max")?,
        _ => u64::MAX,
    };
    Ok(host.min(cgroup))
}

/// Whether one indexed dense leaf belongs in the startup arena.
///
/// Explicit globals are caller-required and take precedence over both the
/// layer range and the drafting flag. Otherwise MTP leaves are optional even
/// when their layer numbers overlap the main model's range.
fn dense_in_startup_share(
    name: &str,
    layer: Option<i64>,
    layers: &std::ops::Range<usize>,
    required_global: bool,
    drafts: bool,
) -> bool {
    required_global
        || ((drafts || !name.starts_with("model.mtp."))
            && layer
                .map(|l| layers.contains(&(l as usize)))
                .unwrap_or(false))
}

/// What ONE dense leaf costs the DEVICE POOL, on top of its place in the arena.
///
/// # The term that was missing
///
/// The startup copy writes this run's selected weights into one anonymous
/// arena, and `total` in [`PileSource::copy_share`] is that arena. Admission
/// charged it once and stopped there, which was right only while every weight
/// the GPU read was ALIASED out of it. Two things break that.
///
/// The first has always been true and was never priced: two of the matrices the
/// loader binds are FUSED in the checkpoint (`mlp.w13_dn.weight` is gate and up
/// interleaved, `shared_w13_weight` likewise) and are split on the host before
/// binding, so their halves are fresh host buffers that no mapping covers and
/// `slice_or_copy` can only COPY. The router's projection is the same story for
/// a different reason: the MMA tiles n by 8 and 258 rows do not divide, so the
/// bf16 arm binds a padded copy. Those copies are a second residency and always
/// were.
///
/// The second is [`super::budget::DenseWeights::DevicePool`], which copies the
/// rest of the plain-BF16 weights too, and buys 179.0 -> 226.8 GB/s achieved
/// and 18.4 -> 20.2 tok/s for it. That arm was measured to hold 4.94 GiB in the
/// pool where the aliased arm holds 1.41, and admission reported the SAME
/// 111.40 GiB estimate for both -- it could not see a 3.53 GiB difference.
///
/// # Why this is a shape and not a coefficient
///
/// Every byte here is `leaf.bytes.len()`, which is the tensor's own dims times
/// its element width. Nothing is fitted and nothing is per-token: this is a
/// RESIDENT set, held from the bind to the end of the run, and it is a
/// different quantity from the 5.301 GB a decode pass READS across the same 143
/// GEMMs. On a 16-layer share those two numbers coincide, because a decode step
/// reads each weight exactly once -- which is a coincidence of the schedule and
/// not an identity, and is exactly the confusion a fitted constant would have
/// baked in.
///
/// Enumerated at `INK_LAYERS=0:16` on the 42-layer checkpoint, layers 0..1
/// dense and 2..15 routed, from the pile's own dims:
///
/// | weight | shape | each | leaves | binds | bytes |
/// |---|---|---|---|---|---|
/// | `attn.w{q_du,o_ud}` | `[4096, 4096]` | 32 MiB | 32 | 32 | 1 073 741 824 |
/// | `attn.w{k_dv,v_dv}` | `[1024, 4096]` | 8 MiB | 32 | 32 | 268 435 456 |
/// | `attn.wr_du` | `[512, 4096]` | 4 MiB | 16 | 16 | 67 108 864 |
/// | `mlp.w13_dn` split | `[32768, 4096]` | 256 MiB | 2 | 4 | 536 870 912 |
/// | `mlp.w2_md` | `[4096, 16384]` | 128 MiB | 2 | 2 | 268 435 456 |
/// | `shared_w13` split+joined | `[2, 4096, 4096]` | 64 MiB | 14 | 14 | 939 524 096 |
/// | `shared_w2` per expert | `[2, 4096, 2048]` | 32 MiB | 14 | 28 | 469 762 048 |
/// | `mlp.gate` padded | `[264, 4096]` | 2.06 MiB | 14 | 14 | 30 277 632 |
/// | `unembed` | `[201024, 4096]` | 1.53 GiB | 1 | 1 | 1 646 788 608 |
/// | **total** | | | | **143** | **5 300 944 896 = 4.94 GiB** |
///
/// A leaf and a bind are not the same thing -- the two fused matrices are bound
/// as their halves and `shared_w2` once per shared expert -- but the BYTES are
/// the leaf's either way, which is why this function can sum leaves while the
/// run's census counts binds and the two still have to agree.
///
/// 143 binds and 5 300 944 896 bytes is what the run's own census prints, and
/// `pool[after stack]` in that arm reads 4.94 GiB live. The aliased arm's four
/// host-transformed rows alone are 1 506 672 640 bytes = 1.40 GiB against a
/// measured 1.41, and its seam reports exactly 32 binds `copied because
/// UNMAPPED` -- 4 + 14 + 14, the same three rows. Both arms, both directions,
/// from the same table.
///
/// The count is not cross-checked here but in the run itself: the stack prints
/// what it bound beside what admission charged, so a lane added later that this
/// function does not know about shows up as a named discrepancy in the log
/// rather than as an OOM kill six weeks on.
///
/// The MTP heads live in the same layer range as the LLM's, but startup copies
/// them only when drafting is on or they are explicit required globals. They
/// are only BOUND when drafting is on, so this second residency is charged on
/// `policy.drafts` and not on the range or explicit arena inclusion.
fn device_weight_bytes(name: &str, leaf: &Leaf, policy: super::budget::AdmissionPolicy) -> u64 {
    use super::budget::DenseWeights;

    /// The four the fused bind concatenates, in its output order.
    const QKVR: [&str; 4] = [
        "attn.wq_du.weight",
        "attn.wk_dv.weight",
        "attn.wv_dv.weight",
        "attn.wr_du.weight",
    ];
    /// The matrices the BF16 GEMM lane binds, by the suffix that names them.
    /// Shared with nothing: the loader constructs these names inline at its
    /// binding sites, and the run's own bind census is what proves the two
    /// lists agree.
    const GEMM: [&str; 9] = [
        "attn.wq_du.weight",
        "attn.wk_dv.weight",
        "attn.wv_dv.weight",
        "attn.wr_du.weight",
        "attn.wo_ud.weight",
        "mlp.w13_dn.weight",
        "mlp.w2_md.weight",
        "mlp.shared_experts.shared_w13_weight",
        "mlp.shared_experts.shared_w2_weight",
    ];

    if name.starts_with("model.mtp.") && !policy.drafts {
        return 0;
    }
    let bytes = leaf.bytes.len() as u64;

    // `INK_FUSE_QKVR`: the four attention projections are ALSO bound as one
    // concatenated `[6656, hidden]` weight. A concatenation is a fresh host
    // buffer and not a span of any registered mapping, so `bind_bf16` cannot
    // alias it -- it is a pool copy in BOTH placement arms, exactly like the
    // two checkpoint-fused matrices below. It is charged ON TOP of whatever the
    // placement arm charges for the four separate binds, not instead of them:
    // the fused weight does not replace them, `AttnWeightsDev` holds both and
    // `project_qkvr` picks. The MTP heads do NOT fuse (`MtpOwned` carries no
    // fused field) and their names end with the same four suffixes, so they are
    // excluded by prefix rather than by suffix.
    //
    // None of these four is a `mlp.gate`, a host-split matrix, an
    // `input_proj` or the unembedding, so this returns rather than falling
    // through: the branches below cannot also claim the name.
    if policy.fused_qkvr
        && !name.starts_with("model.mtp.")
        && QKVR.iter().any(|s| name.ends_with(s))
    {
        return bytes
            + if policy.dense_weights == DenseWeights::Aliased {
                0
            } else {
                bytes
            };
    }

    // Bound row-padded out of a fresh host buffer, so it is a pool copy in
    // BOTH arms -- and the pad is part of the allocation, not a rounding.
    if name.ends_with("mlp.gate.weight") {
        if !policy.router_bf16 || leaf.dims.len() != 2 {
            return 0;
        }
        let rows = leaf.dims[0].next_multiple_of(super::bf16gemm::NTILE as u64);
        return rows * leaf.dims[1] * 2;
    }
    // Fused in the checkpoint, split on the host: the halves are not in any
    // mapping, so aliasing them is not on offer.
    let split = name.ends_with("mlp.w13_dn.weight")
        || name.ends_with("mlp.shared_experts.shared_w13_weight");
    if split {
        return bytes;
    }
    if policy.dense_weights == DenseWeights::Aliased {
        return 0;
    }
    // The MTP head's entry projection, and the largest single weight in the
    // process, neither of which wears a layer suffix the table above covers.
    if name.ends_with("input_proj.weight") || name == "model.llm.unembed.weight" {
        return bytes;
    }
    if GEMM.iter().any(|s| name.ends_with(s)) {
        return bytes;
    }
    0
}

/// What the run needs BESIDES the weight share.
///
/// On a unified-memory part the CUDA context, the device-resident activations
/// and the anonymous arena are one pool, so the share is not the footprint --
/// it is the footprint minus everything that has not been allocated yet. The
/// gate that only compared the share against `MemAvailable` was measuring the
/// first of those against a number that already counts the page cache the copy
/// is about to want back, which is why `INK_LAYERS=0:30` (116.24 GiB of
/// weights on a 121.6 GiB box) was ADMITTED and then thrashed.
///
/// The three terms, measured on this part:
///
/// * the CUDA context is 0.2 GiB;
/// * the per-layer resident activations are 4.1 GiB for a 20-layer share --
///   the KV-adjacent and MLP intermediates, which scale with the RANGE and
///   barely with the sequence;
/// * 4 GiB is left for the kernel, the shell and the page-cache working window,
///   and that number is not a guess either. It is where the measured cliff is;
/// * when SubSlices is selected, cubecl's two largest pool PAGES, from
///   what the ALLOCATOR holds beyond the live set, from
///   [`super::budget::allocator_overhead_bytes`]. Under SubSlices that is the
///   two largest ladder pages -- it reserves 41.74 GiB to hold 1.14 GiB of live
///   tensors at 16,384 tokens, because a page is allocated whole and returned
///   only when every slice of it is free. Under ExclusivePages, which is the
///   default, it is the bucket-rounding ratio of the allocator's own
///   construction applied to `attention_bytes`. Pages the pool is holding FREE
///   for reuse are charged nothing here and bounded at runtime by
///   [`super::pool::CleanupPolicy::WhenStranded`], which is the same hazard
///   compared exactly instead of predicted;
/// * `attention_bytes` is everything that scales with the SEQUENCE, and it is
///   the term this function did not have. See
///   [`super::budget::prefill_activation_bytes`]. It was briefly the score
///   matrices alone, which was not enough by itself: once the dense lane
///   blocked its queries that term stopped growing -- 13.84 GiB at 16,384
///   tokens, 13.34 at 81,920, 13.52 at 100,623 -- and the gate went back to
///   being flat in the one variable it was added to track. What actually
///   scales is the routed-expert lane, whose every stage is
///   `num_experts_per_tok` rows a token: 372 KiB a token against the score
///   blocks' bounded 8 GiB.
///
/// The ladder, on a 121.63 GiB box, caches dropped and swap reset before each
/// row, `INK_GEN=1`, at 01211be plus this change. "free" and "swap" are read
/// straight after the startup copy, "peak" is `/usr/bin/time -v`:
///
/// | `INK_LAYERS` | share | peak RSS | free | swap | outcome |
/// |---|---|---|---|---|---|
/// | 0:20 | 80.72 GiB | 87.16 | 30.9 | 0.0 | forward 5.9 s |
/// | 0:28 | 109.14 | 113.9 | 4.1 | 0.0 | forward 12.5 s |
/// | 0:29 | 112.69 | 116.3 | 1.0 | 0.0 | **`CUDA_ERROR_OUT_OF_MEMORY`** |
/// | 0:30 | 116.24 | 117.5 | 1.0 | 1.4 | forward 45.0 s -- 3.6x 0:28 |
///
/// 0:29 dies outright and 0:30 survives only by swapping, so 0:28 is the
/// honest ceiling and the floor is set to put the refusal between them: this
/// function predicts 119.08 GiB at 0:28 against a machine of 121.63 (admitted,
/// 2.55 GiB of nominal headroom) and 122.84 at 0:29 (refused by 1.21). The
/// 0:30 row was taken with the OLD gate, which admitted it.
///
/// It derives from the machine (`MemTotal`), never from a constant: the two
/// nodes this runs on differ by 2 GiB, and a gate hard-coded to either figure
/// is wrong on the other.
fn run_overhead_bytes(
    layers: usize,
    attention_bytes: u64,
    machine: u64,
    policy: super::budget::AdmissionPolicy,
) -> u64 {
    const CUDA_CONTEXT: u64 = GIB / 5;
    const ACTIVATIONS_PER_LAYER: u64 = 41 * GIB / 200;
    const OS_FLOOR: u64 = 4 * GIB;
    CUDA_CONTEXT
        + ACTIVATIONS_PER_LAYER * layers as u64
        + OS_FLOOR
        + super::budget::allocator_overhead_bytes(policy, machine, attention_bytes)
        + attention_bytes
}

/// Whether a run of a known size may start on a node in a known state.
///
/// Two ways of not fitting, kept apart because they take opposite fixes: one
/// says split the stack across more nodes, the other says leave the stack alone
/// and wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    Fits,
    /// Larger than the node, whatever else is or is not running on it.
    OverMachine,
    /// Fits the node, but not beside what is already resident on it.
    NodeBusy,
}

/// The whole requirement against both of the node's numbers.
///
/// # The half that was missing, and what it cost
///
/// This used to be `need > machine || share > available`: the WHOLE
/// requirement was compared against `MemTotal` and only the raw weight SHARE
/// against `MemAvailable`. Nothing ever asked whether the run fit what the box
/// had free, so every run was priced against an EMPTY machine and whatever else
/// was resident was charged to nobody.
///
/// Measured on spark (119.63 GiB) 2026-08-23, `INK_LAYERS=0:16` at 2048
/// tokens, from the two runs' own logs:
///
/// | run | share | need | available at admission | copy | swap |
/// |---|---|---|---|---|---|
/// | idle node | 66.50 GiB | 111.40 | 114.75 | 15.3 s (4.35 GiB/s) | 0.3 GiB |
/// | 43.9 GiB of AnonPages already held | 66.50 | 112.65 | 69.93 | **1384.5 s (0.05 GiB/s)** | **15.1** |
///
/// The second was ADMITTED, because 66.50 <= 69.93 and 112.65 <= 119.63, and
/// then spent twenty-three minutes writing one anonymous 66.50 GiB arena
/// against a node that did not have it. sshd could not complete a banner
/// exchange for the whole of it -- ICMP answered throughout, so the box looked
/// alive while userspace was starved. The run itself was never the problem: it
/// finished, its pool reserved 4.61 GiB against the 1.28 GiB admission charged,
/// and its forward passes cost 0.1 s each.
///
/// So the failure is not in any per-token coefficient. `MemTotal` cannot see
/// pressure by construction, and this is the number that can.
///
/// `need >= share` always, so testing `need` against `available` subsumes the
/// old share test rather than dropping it.
fn admit(need: u64, machine: u64, available: u64) -> Admission {
    if need > machine {
        Admission::OverMachine
    } else if need > available {
        Admission::NodeBusy
    } else {
        Admission::Fits
    }
}

/// What the node has spare, respecting both host and cgroup limits.
///
/// Public so the pool cleanup policy can compare spare node memory with pages
/// the device allocator is currently stranding.
pub fn mem_available_bytes() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    let kb = status
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
        .context("/proc/meminfo has no numeric MemAvailable")?;
    let host = kb.checked_mul(1024).context("MemAvailable overflow")?;
    let cgroup = match (
        std::fs::read_to_string("/sys/fs/cgroup/memory.max"),
        std::fs::read_to_string("/sys/fs/cgroup/memory.current"),
    ) {
        (Ok(max), Ok(current)) if max.trim() != "max" => {
            let max = max
                .trim()
                .parse::<u64>()
                .context("parsing cgroup memory.max")?;
            let current = current
                .trim()
                .parse::<u64>()
                .context("parsing cgroup memory.current")?;
            max.saturating_sub(current)
        }
        _ => u64::MAX,
    };
    Ok(host.min(cgroup))
}

/// One expert, packed, as a single self-contained blob.
///
/// The dims are LOGICAL — `[rows, cols * 2]` — because `cols` counts packed
/// bytes and two E2M1 values live in each. Writing the packed width here is
/// what makes a checkpoint need an `original_shape` field; writing the logical
/// width is what makes it unnecessary.
pub fn expert_blob(q: &PackedExpert) -> Result<Blob<Tensor<NVFP4, 2>>> {
    let logical = q.cols.checked_mul(2).context("expert logical width overflow")?;
    let elems = q.rows.checked_mul(logical).context("expert logical extent overflow")?;

    // Checked here rather than trusted, because the failure is silent: a
    // scales plane of the wrong length still decodes, just against the wrong
    // blocks, and produces numbers rather than an error.
    anyhow::ensure!(
        q.codes.len() == elems / 2,
        "codes are {} bytes, {} logical elements imply {}",
        q.codes.len(),
        elems,
        elems / 2
    );
    anyhow::ensure!(
        q.scales.len() == elems / NVFP4_BLOCK,
        "scales are {} bytes, {} logical elements in blocks of {NVFP4_BLOCK} imply {}",
        q.scales.len(),
        elems,
        elems / NVFP4_BLOCK
    );

    let mut payload = Vec::with_capacity(NVFP4::payload_len(elems).context("expert payload extent overflow")?);
    payload.extend_from_slice(&q.codes);
    payload.extend_from_slice(&q.scales);
    payload.extend_from_slice(&q.scale2.to_le_bytes());

    tensor_blob::<NVFP4, 2>([q.rows as u64, logical as u64], Bytes::from_source(payload))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Split a blob's payload back into its three planes.
///
/// The inverse of [`expert_blob`]'s layout, offered so a consumer does not have
/// to re-derive the offsets and get them subtly wrong. Both sides compute the
/// boundaries from the same two facts — the element count and the block size —
/// so they cannot disagree.
pub fn split_payload(payload: &[u8], elems: usize) -> Result<(&[u8], &[u8], f32)> {
    let want = NVFP4::payload_len(elems).context("expert payload extent overflow")?;
    anyhow::ensure!(
        payload.len() == want,
        "payload is {} bytes, expected {want}",
        payload.len()
    );
    let codes_len = elems / 2;
    let scales_len = elems / NVFP4_BLOCK;
    let codes = &payload[..codes_len];
    let scales = &payload[codes_len..codes_len + scales_len];
    let tail = &payload[codes_len + scales_len..];
    let scale2 = f32::from_le_bytes(tail.try_into().expect("four trailing bytes"));
    Ok((codes, scales, scale2))
}

/// Facts naming an expert blob.
///
/// The weight attribute is DERIVED per (element, rank) from one anchor, so
/// `Handle<Tensor<NVFP4, 2>>` and `Handle<Tensor<BF16, 3>>` are different
/// attributes with different ids. A query for packed rank-2 experts cannot
/// return a dense rank-3 tensor: the type is the query, not a convention the
/// caller has to remember.
pub mod attrs {
    use super::*;
    use triblespace::core::attribute::Attribute;
    use triblespace::core::id_hex;
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::prelude::*;

    /// Anchor the weight attribute family derives from. Minted 2026-08-10.
    pub const WEIGHT_ANCHOR: Id = id_hex!("0B51DA3E67216213871743E045590DBC");

    /// The weight attribute for any element format and rank.
    ///
    /// One anchor yields a distinct id per `(element, rank)`, which is what
    /// makes the type the query. `weight_nvfp4_2` below is this same attribute
    /// spelled concretely — `attributes!` derives from `(anchor, encoding)`
    /// exactly as `Attribute::anchored` does, so the ids are identical — and it
    /// exists because `entity!` takes an attribute PATH rather than an
    /// expression.
    pub fn weight<T: TensorElement, const RANK: usize>() -> Attribute<Handle<Tensor<T, RANK>>> {
        Attribute::anchored(WEIGHT_ANCHOR)
    }

    attributes! {
        /// A packed rank-2 expert. Same anchor as [`weight`], so this is that
        /// attribute at `(NVFP4, 2)` and not a second one beside it.
        ///
        /// DELIBERATELY the ANCHORED arm (`as`, not `unsafe as`). Every other
        /// minted id in this crate is pinned, because a pinned id is a promise
        /// that data on disk stays reachable. This one is the exception: its
        /// entire purpose is to COINCIDE with `weight::<NVFP4, 2>()`, which is
        /// `Attribute::anchored` and therefore derives. Pin it and the two stop
        /// being the same attribute — the importer writes experts under the
        /// literal while every generic reader looks under the derived id and
        /// finds nothing.
        ///
        /// That is not hypothetical. A bulk pass on 2026-08-11 converted all 52
        /// minted ids to `unsafe as` to repair genuine drift, and swept this one
        /// up with them. Caught before 144 GiB of experts were written under an
        /// id no reader would have asked for. The invariant is asserted below.
        "0B51DA3E67216213871743E045590DBC" as weight_nvfp4_2:
            inlineencodings::Handle<Tensor<NVFP4, 2>>;
        // The checkpoint tensor name lives in `metadata::name` as a UTF8String
        // handle, not here. It was a ShortString attribute until a real name —
        // `model.llm.layers.10.mlp.experts.w13_weight`, 42 characters — panicked
        // the encoder, which answers a too-long value with unwrap() rather than
        // an error. Two copies of one string, and the redundant one was the copy
        // that could not hold it.
        /// Which expert of the stacked matrix.
        "A6ED6DBA4BE63E4E34F2787DA84AD860" as expert_index: inlineencodings::I256BE;
        /// Which transformer layer it belongs to.
        ///
        /// Stored as a fact rather than parsed out of the tensor name at read
        /// time, because splitting a model across machines is a QUERY — "give
        /// me layers 0..21" — and a query over a string you have to parse is
        /// not one.
        "BCDDFBCFF89F67EE0B1E527C4872CED7" as layer: inlineencodings::I256BE;
    }
}

/// Which element format an expert leaf holds, and its handle.
///
/// Two variants because the weight attribute is DERIVED per (element, rank):
/// `Handle<Tensor<NVFP4, 2>>` and `Handle<Tensor<BF16, 2>>` are different
/// attributes with different ids, so "every expert" is two queries and the
/// answer has to be able to say which one it came from. That is the type being
/// the query, paid for at the one place it costs anything.
#[derive(Debug, Clone, Copy)]
pub enum ExpertHandle {
    Nvfp4(
        triblespace::prelude::Inline<
            triblespace::core::inline::encodings::hash::Handle<Tensor<NVFP4, 2>>,
        >,
    ),
    Bf16(
        triblespace::prelude::Inline<
            triblespace::core::inline::encodings::hash::Handle<Tensor<BF16, 2>>,
        >,
    ),
}

/// One expert in a pile, named but NOT loaded.
///
/// A handle, not bytes. Selecting which experts a machine holds must not depend
/// on reading them: a 21/21 layer split is a decision about ~5,000 handles, and
/// materialising even one of them to make it would defeat the split.
#[derive(Debug, Clone, Copy)]
pub struct ExpertRef {
    pub layer: i64,
    pub expert: i64,
    pub handle: ExpertHandle,
}

/// Every expert whose layer falls in `range`, as handles — BOTH element formats.
///
/// This is what makes splitting a model across machines a QUERY. A node asks
/// for the layers it holds and gets references; nothing is read until something
/// is actually computed.
///
/// It sweeps NVFP4 **and** BF16 because Inkling-Small's layer 2 is the odd one
/// out: its experts have no `.scale` sidecar in the checkpoint and land in the
/// pile as `Tensor<BF16, 2>`. A packed-only query is not a filter over that
/// layer, it is a hole — a node told it holds layers 0..=20 would receive every
/// expert of nineteen layers and none of layer 2, compute anyway, and be wrong
/// in exactly one fortieth of the model. Which of the two a leaf is stays in the
/// answer (see [`ExpertHandle`]) rather than being re-derived from a name.
pub fn experts_in_layers(
    space: &triblespace::prelude::TribleSet,
    range: std::ops::RangeInclusive<i64>,
) -> Vec<ExpertRef> {
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::macros::pattern;
    use triblespace::prelude::Inline;

    let mut out: Vec<ExpertRef> = Vec::new();
    for (layer, expert, handle) in triblespace::macros::find!(
        (layer: i64, expert: i64, handle: Inline<Handle<Tensor<NVFP4, 2>>>),
        pattern!(space, [{ _?e @
            attrs::layer: ?layer,
            attrs::expert_index: ?expert,
            attrs::weight_nvfp4_2: ?handle
        }])
    ) {
        if range.contains(&layer) {
            out.push(ExpertRef {
                layer,
                expert,
                handle: ExpertHandle::Nvfp4(handle),
            });
        }
    }
    for (layer, expert, handle) in triblespace::macros::find!(
        (layer: i64, expert: i64, handle: Inline<Handle<Tensor<BF16, 2>>>),
        pattern!(space, [{ _?e @
            attrs::layer: ?layer,
            attrs::expert_index: ?expert,
            attrs::weight::<BF16, 2>(): ?handle
        }])
    ) {
        if range.contains(&layer) {
            out.push(ExpertRef {
                layer,
                expert,
                handle: ExpertHandle::Bf16(handle),
            });
        }
    }
    out.sort_by_key(|r| (r.layer, r.expert));
    out
}

/// The layer a checkpoint tensor name belongs to.
///
/// `model.llm.layers.10.mlp.experts.w13_weight` is layer 10. Returns None for
/// names carrying no layer — the embedding, the final norm — rather than
/// guessing, so a tensor with no layer is visibly absent from a layer query
/// instead of silently landing in layer 0.
pub fn layer_of(tensor_name: &str) -> Option<i64> {
    let rest = tensor_name.split("layers.").nth(1)?;
    rest.split('.').next()?.parse().ok()
}

// ---------------------------------------------------------------------------
// Reading the model back OUT
// ---------------------------------------------------------------------------

/// Which element format a leaf turned out to hold.
///
/// The type parameter is gone by the time a leaf sits in a by-name index — the
/// model spans two dtypes and five ranks — so the fact travels as data instead.
/// This is erasure done ONCE, at the index boundary, from reads that were each
/// typed: a leaf was fetched as `Tensor<BF16, 2>` or not at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    Bf16,
    F32,
}

impl Elem {
    /// Bytes one element occupies on disk.
    pub fn width(self) -> usize {
        match self {
            Elem::Bf16 => 2,
            Elem::F32 => 4,
        }
    }
}

/// One tensor of the model, resolved: its dims from the blob header, its bytes
/// as a VIEW over the pile's mapping.
///
/// Not a `Vec`. `Bytes` here is `Bytes::from_raw_parts(slice, mmap.clone())` —
/// the pile's own mapping with an `Arc` keeping it alive — so an index over the
/// whole model costs handles and headers, never weights, and a lane that hands
/// the bytes to a GPU hands over the mapping itself.
#[derive(Clone)]
pub struct Leaf {
    pub elem: Elem,
    pub dims: Vec<u64>,
    pub bytes: anybytes::Bytes,
    /// Which transformer layer this tensor belongs to, when it belongs to one.
    ///
    /// A FACT the importer wrote, not a substring of the name — which is what
    /// makes "give me layers 0..=19" a query. `None` for the embedding, the
    /// final norm and the unembedding: absent rather than zero, because a
    /// tensor that silently joined layer 0 would ship to the wrong machine.
    pub layer: Option<i64>,
}

impl Leaf {
    /// Shape as the `Vec<usize>` the loaders speak.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().map(|&d| d as usize).collect()
    }

    /// How many elements. From the dims, not from the byte length.
    pub fn elems(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// Widen to f32 — the ONE conversion, made explicit and made the caller's.
    ///
    /// [`crate::models::inkling::load::Checkpoint::tensor`] does this on every
    /// read because a safetensors reader has nothing else to hand back; here the
    /// stored form is reachable, so widening is a thing a caller ASKS for when
    /// it is about to compute in f32, and a device lane that wants the bytes
    /// takes [`Leaf::bytes`] instead.
    pub fn to_f32(&self) -> Vec<f32> {
        match self.elem {
            Elem::Bf16 => self
                .bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            Elem::F32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }
}

/// One expert's packed NVFP4 weight, read out of the pile.
///
/// The three planes are `Bytes` slices of ONE blob, which is what the pile
/// format made atomic: the checkpoint binds `w13_weight`, `.scale` and
/// `.scale2` by naming convention across three different shards, and a reader
/// holding only the first has bytes it cannot interpret. Here there is one
/// handle, and the planes are offsets inside it that both sides compute from the
/// same two facts (see [`split_payload`]).
pub struct PackedSlab {
    pub codes: anybytes::Bytes,
    pub scales: anybytes::Bytes,
    pub scale2: f32,
    /// Output rows of this expert's matrix.
    pub rows: usize,
    /// Packed bytes per row; the logical width is `2 * cols`.
    pub cols: usize,
}

/// One expert's BF16 weight, read out of the pile as a VIEW.
///
/// The unquantised sibling of [`PackedSlab`], and simpler for the reason layer
/// 2 exists at all: nothing was quantised, so there are no planes to split and
/// the payload IS the matrix.
pub struct Bf16Slab {
    pub bytes: Bytes,
    /// Output rows of this expert's matrix.
    pub rows: usize,
    /// Input columns — logical, and here also stored.
    pub cols: usize,
}

/// A model located in a pile: every tensor found, nothing widened, nothing
/// copied.
///
/// The whole reader is two hash maps built once at open. There is no shard
/// index, no header cache, no mapping cache and no span table, because the
/// questions those answer — which file is this tensor in, where in it, is the
/// header parsed yet — do not exist for a content-addressed store. A handle IS
/// the location.
pub struct PileSource {
    /// Where the pile is, kept only so the startup copy can reopen it: the
    /// mapping does not carry an fd and `posix_fadvise` needs one.
    path: std::path::PathBuf,
    reader: triblespace::core::repo::pile::PileSnapshot,
    /// Everything the model collection asserts, kept rather than dropped after
    /// the index is built.
    ///
    /// It costs a few MB against a 159 GiB model and it is what makes the pile
    /// AUTHORITATIVE rather than merely sufficient for weights: the checkpoint's
    /// `config.json` and its siblings live here as facts (see
    /// [`crate::jsonfacts`]), and a runtime that had to reopen the pile to read
    /// them would pay the 18-second index build twice to answer a question the
    /// first open already had in hand.
    facts: triblespace::prelude::TribleSet,
    /// The ordinary model collection descriptor, not a digest of its contents.
    model_collection: triblespace::core::collection::CollectionHandle,
    /// Dense tensors, read as their type at index time. `Leaf` is a view, so
    /// holding all 968 of them costs kilobytes.
    dense: std::collections::HashMap<String, Leaf>,
    /// Experts, as HANDLES: 20 480 of them, and reading even one to build the
    /// index would be 7 MiB of BLAKE3 for a lookup table.
    experts: std::collections::HashMap<(String, i64), ExpertRef>,
    /// How many experts each stacked matrix name has — from the facts, so a
    /// caller never infers a count from an error.
    stacked: std::collections::HashMap<String, usize>,
    /// Anonymous startup copy of the share this process owns. Once present,
    /// every byte a device handle can alias is a view into this allocation,
    /// never into the reclaimable pile mapping.
    copied: Option<anybytes::Bytes>,
    copied_experts: std::collections::HashMap<(String, i64), CopiedExpert>,
    /// Whether the startup copy wrote the NVFP4 expert planes down in
    /// MMA-FRAGMENT ORDER rather than row-major `[n, k]`.
    ///
    /// A property of the BYTES, not a preference: it is false until
    /// [`PileSource::copy_share`] has actually permuted them, so the one path
    /// that skips the copy (`INK_STARTUP_COPY=0`, which leaves the experts
    /// aliasing the pile mapping in place) reports false and every consumer
    /// reads the row-major lane, which is the only lane those bytes support.
    swizzled: bool,
    /// A second copy of one layer's routed experts, taken from the arena
    /// before the learner touched it and never written again: the
    /// checkpoint's experts, in this rank's cut and fragment order, in their
    /// own aligned allocation. See [`PileSource::freeze_experts`].
    frozen: Option<anybytes::Bytes>,
    frozen_experts: std::collections::HashMap<(String, i64), CopiedExpert>,
    /// The existing, opaque model root selected at open. It remains the
    /// training base when learned snapshots are written as its children.
    model_root: Id,
}

#[derive(Clone)]
struct CopiedExpert {
    payload: anybytes::Bytes,
    rows: usize,
    logical: usize,
    nvfp4: bool,
}

impl PileSource {
    /// Open a pile and resolve the unique native Inkling head in its model collection.
    ///
    /// Reads the dense leaves (their headers and their content hashes; the
    /// payloads stay in the mapping) and takes the experts as handles.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        Self::open_root(path, None)
    }

    /// Read an actual model root. A legacy collection-wide import must be
    /// explicitly finalized by its writer; opening never invents its root.
    pub fn open_root(path: &std::path::Path, requested_root: Option<Id>) -> Result<Self> {
        use triblespace::core::inline::encodings::hash::Handle;
        use triblespace::core::metadata;
        use triblespace::macros::{find, pattern};
        use triblespace::prelude::*;

        let t_open = std::time::Instant::now();
        let mut pile = Pile::open(path).map_err(|e| anyhow::anyhow!("open {path:?}: {e:?}"))?;
        // The one model collection. A learned model is a version root inside
        // it (`super::learned`), never a collection beside it.
        let collection_name: &'static str = crate::model_collection::model_collection_name();
        let snapshot = crate::model_collection::snapshot_model_collection_named_local_latest(
            &mut pile,
            collection_name,
        )
        .map_err(|e| anyhow::anyhow!("{path:?}: model collection '{collection_name}': {e}"))?;
        let facts =
            crate::model_collection::project_legacy_model_attributes(snapshot.facts()).facts;
        let model_collection = snapshot.support().collection().handle();
        let (_, _, reader) = snapshot.into_parts();
        pile.close()
            .map_err(|e| anyhow::anyhow!("close {path:?}: {e:?}"))?;
        let open_secs = t_open.elapsed().as_secs_f64();
        let t_experts = std::time::Instant::now();

        // ── which model ─────────────────────────────────────────────────────
        // A root names the exact members to index. Unrelated models,
        // annotations and sibling versions may coexist in this collection;
        // none of them becomes part of the selected checkpoint by union.
        let hex_list = |ids: &[Id]| {
            ids.iter()
                .map(|i| format!("{i:X}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let root = match requested_root {
            Some(id) => {
                anyhow::ensure!(
                    exists!(pattern!(&facts, [{ (id) @ crate::format::attrs::member: _?m }])),
                    "model root {id:X} has no members in {path:?}"
                );
                id
            }
            None => {
                // The shared `member`/`parent` vocabulary also describes
                // other model families and tokenizer Sequence config nodes.
                // Automatic heads are only native Inkling member shapes.
                let heads = super::version::version_heads(&facts);
                match heads.len() {
                    0 => anyhow::bail!(
                        "{path:?} has no selectable native Inkling root. \
                         Name an existing root with --model-root; legacy partial imports need \
                         explicit writer-side publication with `mary root --members-file <ids>` before serving"
                    ),
                    1 => {
                        println!(
                            "    model root: the native Inkling graph's head {:X}",
                            heads[0]
                        );
                        heads[0]
                    }
                    n => anyhow::bail!(
                        "{path:?}: the native Inkling graph has {n} version heads ({}); --model-root names one",
                        hex_list(&heads)
                    ),
                }
            }
        };
        // ── the experts, as handles ─────────────────────────────────────────
        // First, because what it produces is also what tells the dense sweep
        // which entities are NOT dense. An expert entity carries an
        // `expert_index`; a dense one does not. That is the distinction, and it
        // is a FACT rather than a substring test on the name — which matters,
        // because all 256 experts of one matrix share one name and a dense map
        // built without the distinction would hold whichever of them the query
        // happened to yield last.
        let mut experts = std::collections::HashMap::new();
        let mut stacked: std::collections::HashMap<String, usize> = Default::default();
        let mut expert_ids: std::collections::HashSet<Id> = Default::default();
        macro_rules! sweep_experts {
            ($ty:ty, $attr:expr, $wrap:expr) => {{
                type Hit<T> = (
                    Id,
                    Inline<Handle<blobencodings::UTF8String>>,
                    i64,
                    i64,
                    Inline<Handle<Tensor<T, 2>>>,
                );
                let hits: Vec<Hit<$ty>> = find!(
                    (e: Id,
                     n: Inline<Handle<blobencodings::UTF8String>>,
                     i: i64,
                     l: i64,
                     h: Inline<Handle<Tensor<$ty, 2>>>),
                    pattern!(&facts, [
                        { (root) @ crate::format::attrs::member: ?e },
                        { ?e @ metadata::name: ?n, attrs::expert_index: ?i,
                          attrs::layer: ?l, $attr: ?h },
                    ])
                )
                .collect();
                for (e, n, i, l, h) in hits {
                    let name: anybytes::View<str> = reader
                        .get(n)
                        .map_err(|err| anyhow::anyhow!("expert name blob: {err:?}"))?;
                    let name = name.to_string();
                    let c = stacked.entry(name.clone()).or_insert(0);
                    *c = (*c).max(i as usize + 1);
                    expert_ids.insert(e);
                    experts.insert(
                        (name, i),
                        ExpertRef { layer: l, expert: i, handle: $wrap(h) },
                    );
                }
            }};
        }
        sweep_experts!(NVFP4, attrs::weight_nvfp4_2, ExpertHandle::Nvfp4);
        sweep_experts!(BF16, attrs::weight::<BF16, 2>(), ExpertHandle::Bf16);

        let experts_secs = t_experts.elapsed().as_secs_f64();
        let t_dense = std::time::Instant::now();
        // ── the dense tensors, by name ──────────────────────────────────────
        // One query per (element, rank) — ten, not one — because that is what
        // typing the attribute means, and each hit is read AS its type. Nothing
        // is interpreted without one, so a BF16 matrix cannot arrive where f32
        // was asked for.
        let mut dense = std::collections::HashMap::new();
        // How many threads read the dense leaves. Each `get` is a BLAKE3 over
        // the payload and the payloads are the big ones -- the embedding table
        // is 2.40 GiB and the unembedding 1.61 -- so this sweep was 15.0 s of a
        // 23.6 s index build, on one core, for 968 leaves the queries had
        // already located. The reads are independent: `PileSnapshot::get` takes
        // `&self` and its validation cache hashes outside its lock.
        let index_threads: usize = std::env::var("INK_INDEX_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| *n >= 1)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        macro_rules! sweep_dense {
            ($ty:ty, $rank:literal, $tag:expr) => {{
                // Located first, read second. The `find!` iterator borrows
                // `facts` and is not something to hand to a thread; the handles
                // it yields are plain `Copy` values that are.
                type Hit<T> = (
                    Id,
                    Inline<Handle<blobencodings::UTF8String>>,
                    Inline<Handle<Tensor<T, $rank>>>,
                );
                let hits: Vec<Hit<$ty>> = find!(
                    (e: Id,
                     n: Inline<Handle<blobencodings::UTF8String>>,
                     h: Inline<Handle<Tensor<$ty, $rank>>>),
                    pattern!(&facts, [
                        { (root) @ crate::format::attrs::member: ?e },
                        { ?e @ metadata::name: ?n, attrs::weight::<$ty, $rank>(): ?h },
                    ])
                )
                .filter(|(e, _, _)| !expert_ids.contains(e))
                .collect();
                if !hits.is_empty() {
                    let chunk = hits.len().div_ceil(index_threads).max(1);
                    let reader = &reader;
                    let facts = &facts;
                    let parts: Vec<Result<Vec<(String, Leaf)>>> = std::thread::scope(|sc| {
                        let handles: Vec<_> = hits
                            .chunks(chunk)
                            .map(|c| {
                                sc.spawn(move || -> Result<Vec<(String, Leaf)>> {
                                    let mut out = Vec::with_capacity(c.len());
                                    for (e, n, h) in c {
                                        let name: anybytes::View<str> = reader
                                            .get(*n)
                                            .map_err(|err| anyhow::anyhow!("name blob: {err:?}"))?;
                                        let blob: Blob<Tensor<$ty, $rank>> =
                                            reader.get(*h).map_err(|err| {
                                                anyhow::anyhow!("{}: leaf blob: {err:?}", &*name)
                                            })?;
                                        let view: TensorView = TensorView::try_from_blob(blob)
                                            .map_err(|err| {
                                                anyhow::anyhow!("{}: decode: {err}", &*name)
                                            })?;
                                        // The layer is optional in the graph, so it is
                                        // optional here: an `exists!` rather than a second
                                        // required clause, which would silently drop the
                                        // embedding and the head.
                                        let layer = find!(
                                            (l: i64),
                                            pattern!(facts, [{ (*e) @ attrs::layer: ?l }])
                                        )
                                        .next()
                                        .map(|(l,)| l);
                                        out.push((
                                            name.to_string(),
                                            Leaf {
                                                elem: $tag,
                                                dims: view.dims().to_vec(),
                                                bytes: view.payload().clone(),
                                                layer,
                                            },
                                        ));
                                    }
                                    Ok(out)
                                })
                            })
                            .collect();
                        handles.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    for part in parts {
                        for (name, leaf) in part? {
                            dense.insert(name, leaf);
                        }
                    }
                }
            }};
        }
        sweep_dense!(BF16, 0, Elem::Bf16);
        sweep_dense!(BF16, 1, Elem::Bf16);
        sweep_dense!(BF16, 2, Elem::Bf16);
        sweep_dense!(BF16, 3, Elem::Bf16);
        sweep_dense!(BF16, 4, Elem::Bf16);
        sweep_dense!(F32, 0, Elem::F32);
        sweep_dense!(F32, 1, Elem::F32);
        sweep_dense!(F32, 2, Elem::F32);
        sweep_dense!(F32, 3, Elem::F32);
        sweep_dense!(F32, 4, Elem::F32);

        anyhow::ensure!(
            !dense.is_empty(),
            "{path:?}: model collection has no dense leaves"
        );
        println!(
            "    index build: pile open + collection snapshot {open_secs:.1}s, {} expert handles {experts_secs:.1}s, {} dense leaves {:.1}s",
            experts.len(),
            dense.len(),
            t_dense.elapsed().as_secs_f64(),
        );
        Ok(PileSource {
            path: path.to_path_buf(),
            reader,
            facts,
            model_collection,
            dense,
            experts,
            stacked,
            copied: None,
            copied_experts: std::collections::HashMap::new(),
            swizzled: false,
            frozen: None,
            frozen_experts: std::collections::HashMap::new(),
            model_root: root,
        })
    }

    /// The existing opaque root this source loaded from.
    pub fn model_root(&self) -> Id {
        self.model_root
    }

    /// One dense tensor by checkpoint name, as a view.
    pub fn leaf(&self, name: &str) -> Result<&Leaf> {
        self.dense
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{name} is not in the pile"))
    }

    /// Every dense tensor name, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.dense.keys().cloned().collect();
        v.sort();
        v
    }

    /// How many tensors this source located — dense leaves plus experts.
    pub fn len(&self) -> usize {
        self.dense.len() + self.experts.len()
    }

    /// Dense leaves in the index.
    pub fn dense_len(&self) -> usize {
        self.dense.len()
    }

    /// Expert leaves in the index — each expert of each stack, individually.
    pub fn expert_len(&self) -> usize {
        self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many experts a stacked matrix holds.
    pub fn expert_count(&self, base: &str) -> Result<usize> {
        self.stacked
            .get(base)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("{base} is not a stacked expert matrix in this pile"))
    }

    /// Whether a stacked matrix's experts are packed NVFP4 rather than BF16.
    ///
    /// Answered by the ATTRIBUTE the leaves were written under, not by probing
    /// for a `.scale` sidecar's existence. The checkpoint has to ask "is there a
    /// tensor with this name plus `.scale`?", which is a question about a naming
    /// convention; here the element format is part of the leaf's identity.
    pub fn is_nvfp4(&self, base: &str) -> bool {
        matches!(
            self.experts.get(&(base.to_string(), 0)),
            Some(ExpertRef {
                handle: ExpertHandle::Nvfp4(_),
                ..
            })
        )
    }

    /// One expert's NVFP4 planes, read out of the pile and **not decoded**.
    pub fn expert_packed(&self, base: &str, e: usize) -> Result<PackedSlab> {
        if let Some(c) = self.copied_experts.get(&(base.to_string(), e as i64)) {
            anyhow::ensure!(c.nvfp4, "{base}[{e}] is BF16, not packed NVFP4");
            let elems = c.rows * c.logical;
            let codes_len = elems / 2;
            let scales_len = elems / NVFP4_BLOCK;
            let (_, _, scale2) = split_payload(&c.payload, elems)?;
            return Ok(PackedSlab {
                codes: c.payload.slice(..codes_len),
                scales: c.payload.slice(codes_len..codes_len + scales_len),
                scale2,
                rows: c.rows,
                cols: c.logical / 2,
            });
        }
        self.expert_packed_stored(base, e)
    }

    /// One FROZEN expert's packed planes: the checkpoint's, cut and permuted
    /// like the live copy, untouched by the learner. Only the layer
    /// [`PileSource::freeze_experts`] froze answers here.
    pub fn expert_packed_frozen(&self, base: &str, e: usize) -> Result<PackedSlab> {
        let c = self
            .frozen_experts
            .get(&(base.to_string(), e as i64))
            .with_context(|| format!("{base}[{e}] is not frozen"))?;
        anyhow::ensure!(c.nvfp4, "{base}[{e}] is BF16, not packed NVFP4");
        let elems = c.rows * c.logical;
        let codes_len = elems / 2;
        let scales_len = elems / NVFP4_BLOCK;
        let (_, _, scale2) = split_payload(&c.payload, elems)?;
        Ok(PackedSlab {
            codes: c.payload.slice(..codes_len),
            scales: c.payload.slice(codes_len..codes_len + scales_len),
            scale2,
            rows: c.rows,
            cols: c.logical / 2,
        })
    }

    /// Whether a layer's experts were frozen (and which prefix), so a scored
    /// pass knows it has a control to run.
    pub fn frozen_prefix(&self) -> Option<&str> {
        self.frozen_experts
            .keys()
            .next()
            .and_then(|(n, _)| n.rsplit_once("mlp.experts.").map(|(p, _)| p))
    }

    /// Freeze one layer's routed experts: copy this rank's cut of every expert
    /// under `prefix` out of the arena, as it is now, into a second aligned
    /// allocation that the learner never writes. Call it after
    /// [`PileSource::copy_share`] and before any learning pass; the copy is
    /// the checkpoint's experts, and it is what the live loss is measured
    /// against on every scored pass -- the control that costs one expert layer
    /// over the same rows instead of a second model.
    ///
    /// Returns (experts frozen, bytes).
    pub fn freeze_experts(&mut self, prefix: &str) -> Result<(usize, usize)> {
        anyhow::ensure!(self.frozen.is_none(), "experts were already frozen");
        let mut keys: Vec<(String, i64)> = self
            .copied_experts
            .keys()
            .filter(|(n, _)| n.starts_with(prefix))
            .cloned()
            .collect();
        anyhow::ensure!(!keys.is_empty(), "{prefix}: no copied experts to freeze");
        keys.sort();
        const VIEW_ALIGN: usize = 16;
        let mut cursor = 0usize;
        let mut offsets = Vec::with_capacity(keys.len());
        for k in &keys {
            let len = self.copied_experts[k].payload.len();
            offsets.push((cursor, cursor + len));
            cursor = (cursor + len).next_multiple_of(VIEW_ALIGN);
        }
        let mut buf = vec![0u8; cursor + VIEW_ALIGN];
        let skew = buf.as_ptr().align_offset(VIEW_ALIGN);
        for (k, (start, end)) in keys.iter().zip(&offsets) {
            buf[skew + start..skew + end].copy_from_slice(&self.copied_experts[k].payload);
        }
        let bytes = anybytes::Bytes::from_source(buf);
        let view: anybytes::View<[u8]> = bytes
            .clone()
            .view()
            .map_err(|e| anyhow::anyhow!("viewing the frozen expert allocation: {e}"))?;
        let bytes = view.bytes();
        for (k, (start, end)) in keys.iter().zip(&offsets) {
            let c = &self.copied_experts[k];
            self.frozen_experts.insert(
                k.clone(),
                CopiedExpert {
                    payload: bytes.slice(skew + start..skew + end),
                    rows: c.rows,
                    logical: c.logical,
                    nvfp4: c.nvfp4,
                },
            );
        }
        self.frozen = Some(bytes);
        Ok((keys.len(), cursor))
    }

    /// One expert's NVFP4 planes as the PILE stores them, whole and in
    /// row-major order, whatever this process has copied or learned since.
    ///
    /// [`PileSource::expert_packed`] prefers the copied arena, which is the
    /// live expert -- cut to this rank's share, permuted into fragment order,
    /// and updated in place by the learner. This is the other side of that:
    /// the checkpoint the run started from, which is what a learned expert is
    /// diffed against before it is written back.
    pub fn expert_packed_stored(&self, base: &str, e: usize) -> Result<PackedSlab> {
        let h = match self
            .experts
            .get(&(base.to_string(), e as i64))
            .map(|r| r.handle)
        {
            Some(ExpertHandle::Nvfp4(h)) => h,
            Some(ExpertHandle::Bf16(_)) => {
                anyhow::bail!("{base}[{e}] is BF16, not packed NVFP4")
            }
            None => anyhow::bail!("{base}[{e}] is not in the pile"),
        };
        let blob: Blob<Tensor<NVFP4, 2>> = self
            .reader
            .get(h)
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: {err:?}"))?;
        let view: TensorView = TensorView::try_from_blob(blob)
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: decode: {err}"))?;
        let dims = view.dims();
        anyhow::ensure!(dims.len() == 2, "{base}[{e}] is rank {}", dims.len());
        let (rows, logical) = (dims[0] as usize, dims[1] as usize);
        let elems = rows * logical;
        let payload = view.payload();
        // The boundaries are derived, here, from the element count and the
        // block size — the same two facts the writer used. Nothing on disk
        // records them, so the two sides cannot disagree about them.
        let codes_len = elems / 2;
        let scales_len = elems / NVFP4_BLOCK;
        let (_, _, scale2) = split_payload(payload, elems)?;
        Ok(PackedSlab {
            codes: payload.slice(..codes_len),
            scales: payload.slice(codes_len..codes_len + scales_len),
            scale2,
            rows,
            cols: logical / 2,
        })
    }

    /// One expert's BF16 bytes, as a view over the pile's mapping.
    ///
    /// The dual of [`PileSource::expert_packed`], and it refuses the other
    /// format for the same reason that one does: which of the two a leaf holds
    /// is part of its identity here, so asking for the wrong one is an error
    /// and never a reinterpretation.
    pub fn expert_bf16(&self, base: &str, e: usize) -> Result<Bf16Slab> {
        if let Some(c) = self.copied_experts.get(&(base.to_string(), e as i64)) {
            anyhow::ensure!(!c.nvfp4, "{base}[{e}] is packed NVFP4, not BF16");
            anyhow::ensure!(
                c.payload.len() == c.rows * c.logical * 2,
                "{base}[{e}]: {} bytes for {}x{} BF16",
                c.payload.len(),
                c.rows,
                c.logical
            );
            return Ok(Bf16Slab {
                bytes: c.payload.clone(),
                rows: c.rows,
                cols: c.logical,
            });
        }
        let h = match self
            .experts
            .get(&(base.to_string(), e as i64))
            .map(|r| r.handle)
        {
            Some(ExpertHandle::Bf16(h)) => h,
            Some(ExpertHandle::Nvfp4(_)) => {
                anyhow::bail!("{base}[{e}] is packed NVFP4, not BF16")
            }
            None => anyhow::bail!("{base}[{e}] is not in the pile"),
        };
        let blob: Blob<Tensor<BF16, 2>> = self
            .reader
            .get(h)
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: {err:?}"))?;
        let view: TensorView = TensorView::try_from_blob(blob)
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: decode: {err}"))?;
        let dims = view.dims();
        anyhow::ensure!(dims.len() == 2, "{base}[{e}] is rank {}", dims.len());
        let (rows, cols) = (dims[0] as usize, dims[1] as usize);
        let payload = view.payload();
        anyhow::ensure!(
            payload.len() == rows * cols * 2,
            "{base}[{e}]: {} bytes for {rows}x{cols} BF16",
            payload.len()
        );
        Ok(Bf16Slab {
            bytes: payload.clone(),
            rows,
            cols,
        })
    }

    /// The pile's mapping, as `(base, len, keepalive)` — a list of ONE.
    ///
    /// A pile is one file, so a zero-copy lane registers it once and every slab
    /// afterwards is offset arithmetic. The checkpoint's answer is nine shards,
    /// and the only reason that number is not one is that safetensors has a
    /// 2 GiB-ish practical shard ceiling and a 159 GiB model does not fit in it.
    ///
    /// Recovered from a leaf rather than stored: the pile hands out
    /// `Bytes::from_raw_parts(slice, mmap.clone())`, so the mapping IS the
    /// owner of every payload and asking a payload for its owner is exact. A
    /// second `mmap` of the same file would be a different address range and
    /// every offset computed against it would be silently wrong.
    pub fn mappings(
        &self,
    ) -> Result<
        Vec<(
            usize,
            usize,
            std::sync::Arc<dyn std::any::Any + Send + Sync>,
        )>,
    > {
        if let Some(bytes) = &self.copied {
            // The arena, and the frozen experts' allocation beside it when
            // there is one: each is its own mapping, and an expert table is
            // built inside exactly one of them.
            let mut out = Vec::with_capacity(2);
            for bytes in std::iter::once(bytes).chain(self.frozen.iter()) {
                out.push(weight_arena_mapping(bytes));
            }
            return Ok(out);
        }
        let any = self
            .dense
            .values()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no leaves to recover the mapping from"))?;
        let map: std::sync::Arc<memmap2::MmapRaw> = any
            .bytes
            .clone()
            .downcast_to_owner()
            .map_err(|_| anyhow::anyhow!("a pile leaf is not backed by the pile's mapping"))?;
        Ok(vec![(
            map.as_ptr() as usize,
            map.len(),
            map as std::sync::Arc<dyn std::any::Any + Send + Sync>,
        )])
    }

    /// Whether the NVFP4 expert planes are in MMA-fragment order.
    ///
    /// The question a GEMM launcher has to ask before it picks an index
    /// expression, and it is about bytes rather than configuration: see
    /// [`PileSource::swizzled`]'s field docs.
    pub fn experts_swizzled(&self) -> bool {
        self.swizzled
    }

    /// Everything the model collection asserts.
    pub fn facts(&self) -> &triblespace::prelude::TribleSet {
        &self.facts
    }

    /// The ordinary collection descriptor containing the selected root.
    pub fn model_collection(&self) -> triblespace::core::collection::CollectionHandle {
        self.model_collection
    }

    /// The blob reader, so a caller can resolve handles the facts name.
    ///
    /// This is the whole frozen store observation the facts were materialized
    /// from, kept because it also owns the mmap; the name stays `reader`
    /// because nothing past `open` asks it anything but "give me these bytes".
    pub fn reader(&self) -> &triblespace::core::repo::pile::PileSnapshot {
        &self.reader
    }

    /// The same, restricted to a half-open LAYER range.
    ///
    /// The layer is a FACT the importer wrote and this index kept
    /// ([`ExpertRef::layer`]), not a substring of the name, which is what makes
    /// "the experts this node holds" a lookup rather than a parse. A node that
    /// warmed the whole pile would read 159 GiB to prepare for the 85 it runs.
    pub fn expert_keys_in(&self, range: std::ops::Range<usize>) -> Vec<(String, i64)> {
        let mut v: Vec<(String, i64)> = self
            .experts
            .iter()
            .filter(|((_, _), r)| r.layer >= range.start as i64 && r.layer < range.end as i64)
            .map(|((n, e), _)| (n.clone(), *e))
            .collect();
        v.sort();
        v
    }

    /// Copy exactly one node's share out of the file-backed pile mapping into
    /// one dedicated arena. Host arms use the same anonymous mmap; CudaManaged
    /// uses CPU-readable CUDA-managed storage with the same payload layout.
    /// Aliases retain its owner; no second full host weight allocation is made.
    /// Neither arm promises physical pinning or fault-free GPU access.
    ///
    /// `HostReadOnly` seals this mapping after every copy/swizzle worker
    /// joins and before publishing any view. Both non-Host choices require
    /// inference-only callers. Managed initialization zeroes the entire arena
    /// before workers start, including padding; allocation/zero time is logged
    /// separately because this changes startup first-touch/reclaim scheduling.
    ///
    /// Dense leaves follow the layer range, except `model.mtp.*` leaves are
    /// omitted when `policy.drafts` is false. Every `global_dense` name is an
    /// explicit requirement and is copied regardless of range or drafting;
    /// a missing required global is an error. Unselected leaves retain their
    /// original file-backed views instead of occupying this arena.
    ///
    /// # One pass, not two, and why that is a memory question
    ///
    /// This used to be a sequential `fetch+verify` loop that read and BLAKE3'd
    /// every leaf, followed by a threaded `memcpy` loop that copied what the
    /// first loop had faulted in. On a discrete-memory box that is merely two
    /// loops; on a unified-memory one it is the whole problem. Between the two
    /// loops the process holds the share TWICE — once as the pile's mapped page
    /// cache, once as the anonymous arena — and this node's share is 80.72 GiB
    /// against 119-121 GiB of RAM that the GPU also lives in. Measured at
    /// `INK_LAYERS=0:20`: 117.4 GiB resident, 0.7 GiB free, and the entire
    /// 16 GiB swap consumed, because the kernel chose to page out the arena we
    /// had just written rather than evict the mapped file pages we were done
    /// with. The second loop then ran at 3.0 GiB/s instead of 50 — it was
    /// re-reading its own source off the SSD — and on the node with 2 GiB less
    /// of the two this runs on, the CUDA context that comes afterwards could
    /// not be created at all.
    ///
    /// So: the shapes are probed ONCE PER STACKED MATRIX rather than once per
    /// expert (every expert of one stack is the same matrix, and the copy
    /// asserts that), the layout is computed from those, the arena is allocated
    /// zeroed-by-the-kernel rather than written with 80 GiB of zeros nobody
    /// reads, and then one threaded pass fetches, verifies, copies and releases
    /// each leaf's source pages. Peak residency becomes the arena plus a
    /// working window instead of the arena plus the whole share.
    ///
    /// `attention_bytes` is what prefill will hold in ACTIVATIONS at this
    /// sequence length, from [`super::budget::prefill_activation_bytes`]: the
    /// residual stream, every layer's kept keys and values, and the widest
    /// single layer's working set -- which on this model is the routed-expert
    /// lane, six rows a token through buffers as wide as the hidden size.
    /// It is a parameter rather than something this module derives because the
    /// sequence length is a fact about the RUN and the weight share is a fact
    /// about the checkpoint, and folding one into the other is how the gate
    /// came to charge a constant for something linear in the sequence.
    ///
    /// It is also what the ALLOCATOR charge is now computed from, so there is
    /// no longer a `prefill_tokens` parameter beside it: the sequence reaches
    /// admission once, as the live set it implies, and every term downstream is
    /// a function of that. A second, independent path from the token count to a
    /// byte count is exactly how the two came to disagree.
    ///
    /// # `shard`: the within-layer split's routed-expert cut
    ///
    /// `Some(tp)` copies only THIS RANK's half of every routed expert --
    /// `w13`'s rows, `w2`'s columns, both naming the same half of the
    /// intermediate axis (see [`super::tpshard::routed_cut`]). It is done here
    /// and not at bind time because an expert slab is reached as a byte OFFSET
    /// into this arena and never as a host slice: cutting it anywhere else
    /// would mean a second traversal of 70+ GiB, and cutting it not at all
    /// would leave both ranks computing the whole MoE under an all-reduce that
    /// then sums two copies of it.
    ///
    /// The cut is a change of SOURCE index inside a memcpy that already runs,
    /// so it costs nothing the copy was not already paying -- and it moves half
    /// the bytes, so the arena is smaller and the pass is faster. The
    /// `INK_SWZ` permutation below is applied to the CUT dims, which is why the
    /// swizzleability gate reads them and not the stored ones.
    pub fn copy_share(
        &mut self,
        layers: std::ops::Range<usize>,
        global_dense: &[&str],
        attention_bytes: u64,
        policy: super::budget::AdmissionPolicy,
        shard: Option<super::tp::Tp>,
        weight_storage: WeightStorage,
    ) -> Result<(usize, usize, u64, u64)> {
        anyhow::ensure!(self.copied.is_none(), "the weight share was already copied");

        /// The shape every expert of one stacked matrix has, whole and cut.
        ///
        /// Both, deliberately. The STORED dims are what each leaf is validated
        /// against -- a leaf that is not the shape its stack claims is refused
        /// -- and the CUT dims are what is written, indexed and later reported
        /// by `expert_packed`. Keeping only one of the two is how a cut arena
        /// comes to be read with the uncut shape, which is finite numbers from
        /// the wrong place.
        #[derive(Clone)]
        struct Shape {
            rows: usize,
            logical: usize,
            nvfp4: bool,
            payload: usize,
            cut: super::tpshard::Cut,
            cut_rows: usize,
            cut_logical: usize,
            cut_payload: usize,
        }

        let keys = self.expert_keys_in(layers.clone());

        // One probe per stacked matrix, not per expert. A stack is 256 slices
        // of ONE matrix, so its element format and its dimensions are a
        // property of the stack; reading all 9 216 headers to discover 40
        // identical answers is 80 GiB of BLAKE3 paid to learn a layout. The
        // assumption is not silent: the copy below refuses any leaf whose
        // payload is not the length this shape implies.
        let t_probe = std::time::Instant::now();
        let mut shapes: std::collections::HashMap<String, Shape> = Default::default();
        for (name, e) in &keys {
            if shapes.contains_key(name) {
                continue;
            }
            let r = &self.experts[&(name.clone(), *e)];
            let (rows, logical, nvfp4, payload) = match r.handle {
                ExpertHandle::Nvfp4(h) => {
                    let blob: Blob<Tensor<NVFP4, 2>> = self
                        .reader
                        .get(h)
                        .map_err(|err| anyhow::anyhow!("{name}[{e}]: {err:?}"))?;
                    let view = TensorView::try_from_blob(blob)
                        .map_err(|err| anyhow::anyhow!("{name}[{e}]: decode: {err}"))?;
                    (
                        view.dims()[0] as usize,
                        view.dims()[1] as usize,
                        true,
                        view.payload().len(),
                    )
                }
                ExpertHandle::Bf16(h) => {
                    let blob: Blob<Tensor<BF16, 2>> = self
                        .reader
                        .get(h)
                        .map_err(|err| anyhow::anyhow!("{name}[{e}]: {err:?}"))?;
                    let view = TensorView::try_from_blob(blob)
                        .map_err(|err| anyhow::anyhow!("{name}[{e}]: decode: {err}"))?;
                    (
                        view.dims()[0] as usize,
                        view.dims()[1] as usize,
                        false,
                        view.payload().len(),
                    )
                }
            };
            // This rank's cut of THIS stack, decided from the name and the
            // stored dims. `None` -- a single-node run -- is the identity, and
            // `routed_cut` REFUSES a stack it has no rule for rather than
            // leaving it whole: an unsharded operand under the MoE all-reduce
            // is summed twice, which is 2x, finite and fluent.
            let cut = match shard {
                Some(tp) if tp.is_split() => super::tpshard::routed_cut(tp, name, rows, logical)?,
                _ => super::tpshard::Cut::Rows(0..rows),
            };
            let (cut_rows, cut_logical) = cut.dims(rows, logical);
            let cut_elems = cut_rows.checked_mul(cut_logical).context("cut logical extent overflow")?;
            let cut_payload = match nvfp4 {
                true => NVFP4::payload_len(cut_elems),
                false => cut_elems.checked_mul(2),
            }.context("cut payload extent overflow")?;
            shapes.insert(
                name.clone(),
                Shape {
                    rows,
                    logical,
                    nvfp4,
                    payload,
                    cut,
                    cut_rows,
                    cut_logical,
                    cut_payload,
                },
            );
        }
        let probe_secs = t_probe.elapsed().as_secs_f64();

        // Write the routed experts down in MMA-FRAGMENT ORDER, or don't -- one
        // decision, taken here, for the whole share.
        //
        // It belongs in this copy and nowhere else. The bytes have to be
        // permuted somewhere, and every other candidate is worse: the PILE
        // cannot carry the layout (the tensor blob header is dimensions only
        // and `weight_nvfp4_2` is anchored on element format and rank, not byte
        // order, so a permuted blob decodes cleanly and is silently misread --
        // see the module header) and a separate device or host pass would be a
        // second traversal of 70+ GiB. Here it is a change of DESTINATION INDEX
        // inside a memcpy that already happens.
        //
        // All-or-nothing across the share, because the consumer's flag is a
        // comptime one per kernel and a mixture would need it per expert. The
        // shapes are the stacks', so this is 2-3 questions, not 9 216.
        //
        // WHAT IT COSTS, measured rather than argued. Nine `bench-decode.sh`
        // reps at `INK_LAYERS=0:16` (66.50 GiB of share, 20 threads) on a DGX
        // Spark GB10: `fetch+verify+copy` was 14.6 / 14.7 / 14.8 s permuting
        // against 15.0 / 15.1 / 15.9 s not permuting -- the permuting arm is
        // FASTER by the spread, which is to say the difference is nil. This
        // pass runs at 4.4-4.5 GiB/s and is bound by the memory system, not by
        // the address arithmetic, so re-indexing 256-byte blocks inside it is
        // free. Per STARTUP, once, not per token.
        // The CUT dims, not the stored ones: the permutation is applied to what
        // is written, and a cut can leave a stack unswizzleable that the whole
        // one was (or, at these shapes, vice versa). Reading the stored dims
        // here would permute against a tiling the arena does not have.
        let swz = super::moegroup::swizzle_weights()
            && shapes
                .values()
                .all(|s| !s.nvfp4 || super::fp4gemm::swizzleable(s.cut_rows, s.cut_logical));

        let globals: std::collections::HashSet<&str> = global_dense.iter().copied().collect();
        for name in &globals {
            anyhow::ensure!(
                self.dense.contains_key(*name),
                "startup-copy table {name} is not in the pile"
            );
        }
        let mut dense_names: Vec<String> = self
            .dense
            .iter()
            .filter(|(name, leaf)| {
                dense_in_startup_share(
                    name,
                    leaf.layer,
                    &layers,
                    globals.contains(name.as_str()),
                    policy.drafts,
                )
            })
            .map(|(name, _)| name.clone())
            .collect();
        dense_names.sort();

        // The DESTINATION of every leaf, computed before anything is read.
        // Disjoint destinations are what let the pass below be threaded:
        // experts in `expert_keys_in` order, then dense leaves by name, each
        // padded to a SIXTEEN-byte boundary.
        //
        // Sixteen, not the four this used to write. The tuned BF16 GEMM lanes
        // pick their load width from the tensor SHAPE and never from the
        // pointer, so a weight aliased at 4 mod 16 raises
        // `CUDA_ERROR_MISALIGNED_ADDRESS` -- an async fault that poisons the
        // CUDA context -- and the dense lane has to route it to the slower hand
        // kernel instead. Sixteen costs at most 15 bytes per view, about 14 KB
        // on a 20-layer share, and it is the whole benefit `INK_ALIGN_COPY=1`
        // was buying by DUPLICATING 908 MiB of weight into a fresh device
        // allocation; that arm is moot now.
        const VIEW_ALIGN: usize = 16;
        // What each layer costs, so a refusal can name the range that WOULD
        // fit instead of leaving the operator to bisect for it. Accumulated
        // here because this loop is the only place the byte counts exist.
        let mut per_layer: std::collections::BTreeMap<i64, usize> = Default::default();
        let mut fixed_bytes = 0usize;
        // The SAME split, for the second residency the binding lane costs.
        // Kept apart from `per_layer` rather than folded into it so the
        // refusal's "N GiB of weights" stays the arena figure it names, and so
        // the bisect below can add both without counting either twice.
        let mut per_layer_dev: std::collections::BTreeMap<i64, u64> = Default::default();
        let mut fixed_dev = 0u64;
        let mut cursor = 0usize;
        let mut expert_offsets = Vec::with_capacity(keys.len());
        for key in &keys {
            let start = cursor;
            let end = start
                .checked_add(shapes[&key.0].cut_payload)
                .context("weight share byte count overflow")?;
            cursor = end.next_multiple_of(VIEW_ALIGN);
            *per_layer.entry(self.experts[key].layer).or_default() += cursor - start;
            expert_offsets.push((start, end));
        }
        let mut dense_offsets = Vec::with_capacity(dense_names.len());
        for name in &dense_names {
            let start = cursor;
            let end = start
                .checked_add(self.dense[name].bytes.len())
                .context("weight share byte count overflow")?;
            cursor = end.next_multiple_of(VIEW_ALIGN);
            let dev = device_weight_bytes(name, &self.dense[name], policy);
            match self.dense[name]
                .layer
                .filter(|l| layers.contains(&(*l as usize)))
            {
                Some(l) => {
                    *per_layer.entry(l).or_default() += cursor - start;
                    *per_layer_dev.entry(l).or_default() += dev;
                }
                // The embedding and unembedding tables belong to no layer, so
                // they are what a shorter range still has to pay.
                None => {
                    fixed_bytes += cursor - start;
                    fixed_dev += dev;
                }
            }
            dense_offsets.push((start, end));
        }
        let total = cursor;
        let device_weights: u64 = fixed_dev + per_layer_dev.values().sum::<u64>();

        // The admission gate. ONE requirement -- the share plus everything the
        // run allocates after it, see `run_overhead_bytes` -- asked of two
        // numbers, and both have to say yes: does it fit the MACHINE
        // (`MemTotal`, could this box ever hold it) and does it fit what the
        // box has free RIGHT NOW (`MemAvailable`). See [`Admission`].
        let available = mem_available_bytes()?;
        let machine = mem_total_bytes()?;
        let n_layers = layers.len();
        let overhead = run_overhead_bytes(n_layers, attention_bytes, machine, policy);
        // The arena, PLUS the weights the binding lane holds a second time in
        // the device pool. A correct comparison against a number that omits a
        // term is still wrong; see [`device_weight_bytes`].
        let need = total as u64 + device_weights + overhead;
        let gib = |b: u64| b as f64 / GIB as f64;
        let verdict = admit(need, machine, available);
        if verdict != Admission::Fits {
            // The largest range starting at `layers.start` that WOULD fit. Both
            // sides of the test grow with the layer count, so the predicate is
            // monotone and the last k that passes is the answer; the byte
            // counts are the exact ones this layout just computed, not an
            // average, because a refusal that makes the operator bisect for the
            // answer is half a bug.
            let mut acc = fixed_bytes as u64;
            let mut acc_dev = fixed_dev;
            let mut fits: Option<(usize, u64, u64)> = None;
            for (k, (layer, bytes)) in per_layer.iter().enumerate() {
                acc += *bytes as u64;
                acc_dev += per_layer_dev.get(layer).copied().unwrap_or(0);
                let k = k + 1;
                let fits_here =
                    acc + acc_dev + run_overhead_bytes(k, attention_bytes, machine, policy);
                // The WHOLE requirement against both numbers, exactly as the
                // gate above tests it. Comparing the bare share against
                // `available` here is the same half-measurement that admitted
                // the run being refused, and it would name a range that then
                // failed the same way.
                if admit(fits_here, machine, available) == Admission::Fits {
                    fits = Some((k, acc, fits_here));
                }
            }
            let advice = match fits {
                Some((k, share, whole)) => format!(
                    "The largest range that fits here is INK_LAYERS={}:{} -- {:.2} GiB of weights, \
                     {:.2} GiB with the context and activations. Give the rest to another node.",
                    layers.start,
                    layers.start + k,
                    gib(share),
                    gib(whole),
                ),
                None => format!(
                    "Not even one layer fits: layer {} alone is {:.2} GiB on top of {:.2} GiB of \
                     tables this range cannot drop. This model needs more nodes, not a smaller \
                     range.",
                    layers.start,
                    gib(*per_layer.values().next().unwrap_or(&0) as u64),
                    gib(fixed_bytes as u64),
                ),
            };
            // Which half of the sum is the problem. A refusal that only ever
            // says "use fewer layers" sends the operator to buy nodes when the
            // fix is a shorter input: the activation working set is linear in
            // the sequence and very nearly FLAT in the range, because layers
            // run one at a time and each frees its own before the next
            // allocates. No layer split touches it.
            let mut cause = if attention_bytes > total as u64 {
                format!(
                    "\n  THE SEQUENCE, NOT THE SPLIT: {:.2} GiB of that is activations at this \
                     input length, against {:.2} GiB of weights. Splitting the stack across more \
                     nodes does not help -- the widest layer is as wide on every node, and every \
                     node attends over the whole sequence. Shorten the input.",
                    gib(attention_bytes),
                    gib(total as u64),
                )
            } else {
                String::new()
            };
            // ...and whether the run is the problem at all. A merely BUSY node
            // refuses the same range it will admit once the memory comes back,
            // and an operator told to split a stack when the answer was to wait
            // sixty seconds splits one that never needed splitting.
            if verdict == Admission::NodeBusy {
                cause.push_str(&format!(
                    "\n  THE NODE, NOT THE RUN: this fits a {:.2} GiB machine with {:.2} GiB to \
                     spare -- but only {:.2} GiB is free right now, because {:.2} GiB of this node \
                     is already spoken for. Neither the range nor the input length is wrong. Free \
                     that memory or wait for it: a preceding run of this size does not return its \
                     anonymous arena the instant it exits, and a gate that reads only MemTotal \
                     prices every run against an EMPTY machine.",
                    gib(machine),
                    gib(machine - need),
                    gib(available),
                    gib(machine.saturating_sub(available)),
                ));
            }
            anyhow::bail!(
                "INK_LAYERS={}:{} is {n_layers} layers = {:.2} GiB of weights; with the CUDA \
                 context and this node's activations that is {:.2} GiB, and this machine has \
                 {:.2} GiB ({:.2} GiB available right now). Refusing.{cause}\n  \
                 A share this size fits only by taking memory the GPU still needs. Measured on \
                 a 121.63 GiB box: 0:29 (112.69 GiB) was admitted by the old gate and died with \
                 CUDA_ERROR_OUT_OF_MEMORY, and 0:30 (116.24 GiB) survived only by taking 1.4 GiB \
                 of swap, which cost it 45.0 s of forward against 12.5 s at 0:28. Nothing here \
                 is reclaimable -- the arena is anonymous on purpose, so the GPU can alias it -- \
                 so the kernel pages the weights themselves.\n  \
                 Terms: {:.2} GiB weights + {:.2} GiB device weights + {:.2} GiB overhead, of \
                 which {:.2} GiB is one pass's activations with the context's KV and {:.2} GiB \
                 the allocator's slack on them.\n  \
                 {advice}",
                layers.start,
                layers.end,
                gib(total as u64),
                gib(need),
                gib(machine),
                gib(available),
                gib(total as u64),
                gib(device_weights),
                gib(overhead),
                gib(attention_bytes),
                gib(super::budget::allocator_overhead_bytes(
                    policy,
                    machine,
                    attention_bytes
                )),
            );
        }
        println!(
            "    admission: {:.2} GiB of weights + {:.2} GiB bound in the device pool ({}) \
             + {:.2} GiB context/activations = {:.2} GiB of \
             {:.2} GiB ({:.2} GiB of headroom against the {:.2} GiB free right now)",
            gib(total as u64),
            gib(device_weights),
            policy.dense_weights.name(),
            gib(overhead),
            gib(need),
            gib(machine),
            // The BINDING margin, not the one against an empty machine. This
            // reported `machine - need`, so the idle 2026-08-23 run read 8.23
            // GiB of headroom where it had 3.35, and the run that went on to
            // swap for twenty-three minutes read 6.98 where it was 42.72 GiB
            // SHORT. A refusal now catches that case, and the line says which
            // number the margin is against so a thin one is visible before it
            // becomes a refusal.
            gib(machine.min(available) - need),
            gib(available),
        );

        // Resolve once, before allocation: the managed eager-zero sweep and
        // existing fetch/verify/copy workers share this request. Unset means
        // one per core; INK_COPY_THREADS=1 keeps both phases sequential.
        let threads: usize = std::env::var("INK_COPY_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n >= 1)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });

        // Host arms retain the same lazy dedicated mmap. Managed storage is
        // still fully initialized before any u8 slice exists; only the eager
        // sweep's scheduling changed. Speed is an experiment, not a guarantee.
        // The payload layout and later copy/swizzle work are unchanged.
        let (mut arena, skew) = {
            let bytes = total
                .checked_add(VIEW_ALIGN - 1)
                .context("startup weight arena size overflow")?;
            let arena = WeightArena::new(bytes, weight_storage, threads)?;
            let skew = arena.as_ptr().align_offset(VIEW_ALIGN);
            (arena, skew)
        };

        enum Job<'a> {
            Expert(&'a (String, i64), &'a Shape),
            Dense(&'a str),
        }
        let mut jobs: Vec<(usize, usize, Job)> = Vec::with_capacity(keys.len() + dense_names.len());
        for (k, &(start, end)) in keys.iter().zip(expert_offsets.iter()) {
            jobs.push((start, end, Job::Expert(k, &shapes[&k.0])));
        }
        for (name, &(start, end)) in dense_names.iter().zip(dense_offsets.iter()) {
            jobs.push((start, end, Job::Dense(name.as_str())));
        }

        // Split by BYTES, not by leaf count: the leaves are not the same size
        // (a dense leaf is 537 MB and an expert plane is 14 MB), so an even
        // split of the list is an uneven split of the work.
        let per = total.div_ceil(threads);
        let mut bounds: Vec<usize> = Vec::with_capacity(threads + 1);
        let mut j = 0usize;
        bounds.push(0);
        for t in 1..threads {
            let target = per * t;
            while j < jobs.len() && jobs[j].0 < target {
                j += 1;
            }
            bounds.push(j);
        }
        bounds.push(jobs.len());

        let mut rest: &mut [u8] = &mut arena[skew..skew + total];
        let mut shards: Vec<(&mut [u8], usize, &[(usize, usize, Job)])> = Vec::new();
        let mut base = 0usize;
        for w in bounds.windows(2) {
            let (a, b) = (w[0], w[1]);
            let span_end = if b == jobs.len() { total } else { jobs[b].0 };
            let (head, tail) = rest.split_at_mut(span_end - base);
            shards.push((head, base, &jobs[a..b]));
            rest = tail;
            base = span_end;
        }

        let (map_base, map_len, _keep) = {
            let m = self.mappings()?;
            let (b, l, k) = m.into_iter().next().context("the pile has no mapping")?;
            (b, l, k)
        };
        let release = SourceRelease::new(&self.path, map_base, map_len);
        let release = &release;
        let reader = &self.reader;
        let experts = &self.experts;
        let dense = &self.dense;
        let t_copy = std::time::Instant::now();
        let results: Vec<Result<()>> = std::thread::scope(|sc| {
            let handles: Vec<_> = shards
                .into_iter()
                .map(|(buf, base, mine)| {
                    sc.spawn(move || -> Result<()> {
                        for (start, end, job) in mine {
                            let src: Bytes = match job {
                                Job::Expert(k, shape) => {
                                    let r = &experts[*k];
                                    let payload = match r.handle {
                                        ExpertHandle::Nvfp4(h) => {
                                            let blob: Blob<Tensor<NVFP4, 2>> =
                                                reader.get(h).map_err(|err| {
                                                    anyhow::anyhow!("{}[{}]: {err:?}", k.0, k.1)
                                                })?;
                                            let view =
                                                TensorView::try_from_blob(blob).map_err(|err| {
                                                    anyhow::anyhow!(
                                                        "{}[{}]: decode: {err}",
                                                        k.0,
                                                        k.1
                                                    )
                                                })?;
                                            anyhow::ensure!(
                                                shape.nvfp4
                                                    && view.dims()[0] as usize == shape.rows
                                                    && view.dims()[1] as usize == shape.logical,
                                                "{}[{}] is {:?}, but its stack is {}x{}",
                                                k.0,
                                                k.1,
                                                view.dims(),
                                                shape.rows,
                                                shape.logical,
                                            );
                                            view.payload().clone()
                                        }
                                        ExpertHandle::Bf16(h) => {
                                            let blob: Blob<Tensor<BF16, 2>> =
                                                reader.get(h).map_err(|err| {
                                                    anyhow::anyhow!("{}[{}]: {err:?}", k.0, k.1)
                                                })?;
                                            let view =
                                                TensorView::try_from_blob(blob).map_err(|err| {
                                                    anyhow::anyhow!(
                                                        "{}[{}]: decode: {err}",
                                                        k.0,
                                                        k.1
                                                    )
                                                })?;
                                            anyhow::ensure!(
                                                !shape.nvfp4
                                                    && view.dims()[0] as usize == shape.rows
                                                    && view.dims()[1] as usize == shape.logical,
                                                "{}[{}] is {:?}, but its stack is {}x{}",
                                                k.0,
                                                k.1,
                                                view.dims(),
                                                shape.rows,
                                                shape.logical,
                                            );
                                            view.payload().clone()
                                        }
                                    };
                                    anyhow::ensure!(
                                        payload.len() == shape.payload,
                                        "{}[{}] is {} bytes where its stack implies {}",
                                        k.0,
                                        k.1,
                                        payload.len(),
                                        shape.payload,
                                    );
                                    payload
                                }
                                Job::Dense(name) => dense[*name].bytes.clone(),
                            };
                            let dst = &mut buf[start - base..end - base];
                            match job {
                                // BOTH transforms ride inside this one copy:
                                // this rank's CUT of each plane, then the
                                // MMA-fragment permutation of what the cut
                                // kept. The three planes keep their order
                                // within the payload -- codes, scales, the
                                // `scale2` tail -- so every byte offset the
                                // alias seam computes is where it was; only
                                // the LENGTHS shrink with the cut.
                                //
                                // `cut_plane` borrows for a row cut (a span)
                                // and gathers for a column cut, so `w13` costs
                                // nothing over the old memcpy and `w2` pays a
                                // gather into a buffer this pass was going to
                                // write anyway.
                                Job::Expert(_, shape) if shape.nvfp4 => {
                                    use super::tpshard::{Plane, cut_plane};
                                    let elems = shape.rows * shape.logical;
                                    let (c, sc, _) = split_payload(&src, elems)?;
                                    let cc = cut_plane(
                                        c,
                                        shape.rows,
                                        shape.logical,
                                        Plane::NVFP4_CODES,
                                        &shape.cut,
                                    )?;
                                    let cs = cut_plane(
                                        sc,
                                        shape.rows,
                                        shape.logical,
                                        Plane::NVFP4_SCALES,
                                        &shape.cut,
                                    )?;
                                    let (n, k) = (shape.cut_rows, shape.cut_logical);
                                    let codes_len = n * k / 2;
                                    let scales_len = n * k / NVFP4_BLOCK;
                                    let (dc, rest) = dst.split_at_mut(codes_len);
                                    let (dsc, dtail) = rest.split_at_mut(scales_len);
                                    if swz {
                                        super::fp4gemm::swizzle_b_codes_into(&cc, dc, n, k);
                                        super::fp4gemm::swizzle_b_scales_into(&cs, dsc, n, k);
                                    } else {
                                        dc.copy_from_slice(&cc);
                                        dsc.copy_from_slice(&cs);
                                    }
                                    // `scale2` is a per-TENSOR constant, so the
                                    // cut does not touch it -- half of a
                                    // matrix quantised with a given second-level
                                    // scale is still quantised with that scale.
                                    dtail.copy_from_slice(&src[elems / 2 + elems / NVFP4_BLOCK..]);
                                }
                                Job::Expert(_, shape) => {
                                    let cut = super::tpshard::cut_plane(
                                        &src,
                                        shape.rows,
                                        shape.logical,
                                        super::tpshard::Plane::BF16,
                                        &shape.cut,
                                    )?;
                                    dst.copy_from_slice(&cut);
                                }
                                Job::Dense(_) => dst.copy_from_slice(&src),
                            }
                            release.release(&src);
                        }
                        Ok(())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for r in results {
            r?;
        }
        let copy_secs = t_copy.elapsed().as_secs_f64();
        self.swizzled = swz;
        println!(
            "    startup copy: NVFP4 experts written in {} order",
            if swz {
                "MMA-FRAGMENT"
            } else {
                "row-major [n, k]"
            }
        );
        // What was CUT, said in shapes rather than inferred from a byte count.
        // A within-layer split that silently did not cut is the failure this
        // whole path exists to prevent, and it is invisible in every other
        // number the startup prints.
        if let Some(tp) = shard.filter(|t| t.is_split()) {
            let mut named: Vec<&String> = shapes.keys().collect();
            named.sort();
            for n in named {
                let sh = &shapes[n];
                println!(
                    "    startup copy: rank {}/{} keeps {:?} of {} -- [{}, {}] of [{}, {}]",
                    tp.rank(),
                    tp.world(),
                    sh.cut,
                    n.rsplit('.').next().unwrap_or(n),
                    sh.cut_rows,
                    sh.cut_logical,
                    sh.rows,
                    sh.logical,
                );
            }
        }
        println!(
            "    startup copy: {} shape probe{} {:.1}s, fetch+verify+copy {:.1}s ({:.2} GiB/s, {threads} thread{})",
            shapes.len(),
            if shapes.len() == 1 { "" } else { "s" },
            probe_secs,
            copy_secs,
            total as f64 / (1u64 << 30) as f64 / copy_secs.max(1e-9),
            if threads == 1 { "" } else { "s" },
        );
        println!("{}", mem_line("after fetch+verify+copy"));

        // All workers have joined. Seal before any subview or GPU alias can
        // escape, retaining the mmap or managed owner through Bytes.
        let arena_base = arena.as_ptr() as usize;
        let arena_len = arena.len();
        let seal_start = std::time::Instant::now();
        let bytes = arena.finish(weight_storage)?;
        println!(
            "    startup weight arena: base={arena_base:#x} bytes={arena_len} storage={} protection={} seal_seconds={:.6}",
            weight_storage.as_str(),
            if weight_storage == WeightStorage::HostReadOnly {
                "read-only"
            } else {
                "read-write"
            },
            seal_start.elapsed().as_secs_f64(),
        );
        let view: anybytes::View<[u8]> = bytes
            .clone()
            .view()
            .map_err(|e| anyhow::anyhow!("viewing the anonymous weight allocation: {e}"))?;
        let bytes = view.bytes();
        for ((key, shape), (start, end)) in
            keys.iter().map(|k| (k, &shapes[&k.0])).zip(expert_offsets)
        {
            // The CUT dims. `expert_packed` derives its plane boundaries from
            // these, so recording the stored ones would read a half-size slab
            // with a full-size layout -- real weights, wrong places.
            self.copied_experts.insert(
                key.clone(),
                CopiedExpert {
                    payload: bytes.slice(skew + start..skew + end),
                    rows: shape.cut_rows,
                    logical: shape.cut_logical,
                    nvfp4: shape.nvfp4,
                },
            );
        }
        for (name, (start, end)) in dense_names.iter().zip(dense_offsets) {
            self.dense.get_mut(name).expect("selected dense leaf").bytes =
                bytes.slice(skew + start..skew + end);
        }
        self.copied = Some(bytes);
        Ok((
            self.copied_experts.len(),
            dense_names.len(),
            total as u64,
            device_weights,
        ))
    }

    /// Every `(stacked matrix name, expert index)` this pile holds, sorted.
    ///
    /// The index is already built at open, so this is a rename of what is in
    /// memory rather than a query. It exists because an AUDIT has to be driven
    /// by what the pile actually contains — asking it for the experts a layer
    /// range implies would make a leaf nobody indexed invisible to the audit by
    /// construction.
    pub fn expert_keys(&self) -> Vec<(String, i64)> {
        let mut v: Vec<(String, i64)> = self.experts.keys().cloned().collect();
        v.sort();
        v
    }

    /// Whether one expert leaf is packed NVFP4 (rather than BF16).
    pub fn expert_is_nvfp4(&self, base: &str, e: i64) -> Option<bool> {
        self.experts
            .get(&(base.to_string(), e))
            .map(|r| matches!(r.handle, ExpertHandle::Nvfp4(_)))
    }
}

#[cfg(test)]
mod anchor_tests {
    use super::attrs;
    use triblespace::core::blob::encodings::tensor::elements::NVFP4;

    /// `weight_nvfp4_2` and `weight::<NVFP4, 2>()` must be ONE attribute.
    ///
    /// They are declared two different ways — one through `attributes!`, one
    /// through `Attribute::anchored` — and nothing but this test keeps them
    /// equal. A single `unsafe` keyword on the declaration silently separates
    /// them, and the symptom is an importer and a reader that disagree about
    /// where the weights are, with no error from either side.
    #[test]
    fn nvfp4_expert_attribute_matches_the_generic() {
        assert_eq!(
            attrs::weight_nvfp4_2.id(),
            attrs::weight::<NVFP4, 2>().id(),
            "weight_nvfp4_2 must be weight::<NVFP4,2>(); if this fails, check \
             whether the declaration was changed to `unsafe as`"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inkling::budget::{AdmissionPolicy, StorageDType};
    use crate::models::inkling::pool::AllocatorConfig;
    use triblespace::core::blob::encodings::tensor::TensorView;

    #[test]
    fn managed_weight_zero_worker_plan_is_bounded_for_empty_tiny_and_large_spans() {
        assert_eq!(weight_zero_workers(0, 0), 0);
        assert_eq!(weight_zero_workers(0, usize::MAX), 0);
        assert_eq!(weight_zero_workers(1, 0), 1);
        assert_eq!(weight_zero_workers(3, usize::MAX), 1);
        assert_eq!(weight_zero_workers(MIN_ZERO_BYTES_PER_WORKER - 1, 8), 1);
        assert_eq!(weight_zero_workers(2 * MIN_ZERO_BYTES_PER_WORKER + 3, 8), 2);
        assert_eq!(weight_zero_workers(8 * MIN_ZERO_BYTES_PER_WORKER, 3), 3);
        assert_eq!(
            weight_zero_workers(usize::MAX, usize::MAX),
            MAX_ZERO_WORKERS
        );
    }

    #[test]
    fn managed_weight_zero_initialization_clears_poisoned_spans_without_touching_guards() {
        use std::mem::MaybeUninit;
        for len in [0, 1, 15, 16, 17, 8193, 3 * MIN_ZERO_BYTES_PER_WORKER + 17] {
            for requested in [0, 1, 2, 3, usize::MAX] {
                // An odd leading offset and tail exercise exact byte coverage,
                // not just the aligned pages a large allocation usually has.
                let mut bytes = vec![MaybeUninit::new(0xa5u8); len + 19];
                let workers = initialize_weight_bytes(&mut bytes[7..7 + len], requested);
                assert_eq!(workers, weight_zero_workers(len, requested));
                for (index, byte) in bytes.iter().enumerate() {
                    // SAFETY: guards were initialized above; the entire target
                    // range was initialized before its scoped workers returned.
                    let actual = unsafe { byte.assume_init() };
                    let expected = if (7..7 + len).contains(&index) {
                        0
                    } else {
                        0xa5
                    };
                    assert_eq!(
                        actual, expected,
                        "len={len} requested={requested} index={index}"
                    );
                }
            }
        }
    }

    #[test]
    fn managed_weight_zero_initialization_accepts_truly_uninitialized_storage() {
        use std::mem::MaybeUninit;
        let mut bytes = vec![MaybeUninit::<u8>::uninit(); 2 * MIN_ZERO_BYTES_PER_WORKER + 1];
        assert_eq!(initialize_weight_bytes(&mut bytes, 2), 2);
        for byte in bytes {
            // SAFETY: initialization joined both workers before returning.
            assert_eq!(unsafe { byte.assume_init() }, 0);
        }
    }

    #[test]
    fn weight_storage_names_and_default_are_explicit() {
        assert_eq!(WeightStorage::default(), WeightStorage::Host);
        assert_eq!(WeightStorage::Host.as_str(), "host");
        assert_eq!(WeightStorage::HostReadOnly.as_str(), "host-read-only");
        assert_eq!(WeightStorage::CudaManaged.as_str(), "cuda-managed");
        assert!(!WeightStorage::Host.inference_only());
        assert!(WeightStorage::HostReadOnly.inference_only());
        assert!(WeightStorage::CudaManaged.inference_only());
    }

    #[test]
    fn managed_weight_lengths_are_rejected_before_cuda_initialization() {
        assert!(ManagedWeightArena::new(0, 2).is_err());
        assert!(ManagedWeightArena::new(usize::MAX, 2).is_err());
    }

    #[test]
    fn managed_weight_cpu_metadata_requires_concurrent_access() {
        assert!(validate_managed_concurrent_access(0).is_err());
        assert!(validate_managed_concurrent_access(1).is_ok());
    }

    /// No model or kernels: exercise the final ByteSource owner independently
    /// of CubeCL's long-lived external-registration table.
    #[test]
    #[ignore = "requires an exclusive CUDA reservation"]
    fn managed_weight_arena_owner_drops_on_foreign_thread() -> Result<()> {
        // Keep the primary alive independently of the allocation under test.
        let _primary = cudarc::driver::CudaContext::new(0)?;
        let arena_len = 2 * MIN_ZERO_BYTES_PER_WORKER + 17;
        assert_eq!(weight_zero_workers(arena_len, 2), 2);
        let mut arena = ManagedWeightArena::new(arena_len, 2)?;
        assert!(arena.as_slice().iter().all(|byte| *byte == 0));
        arena.as_mut_slice()[32..40].copy_from_slice(b"weights!");
        let base = arena.ptr.as_ptr() as usize;
        let bytes = Bytes::from_source(arena);
        let weak = bytes.downgrade();
        let slice = bytes.slice(32..40);
        let (mapped_base, len, keepalive) = weight_arena_mapping(&bytes);
        assert_eq!((mapped_base, len), (base, arena_len));
        assert_eq!(slice.as_ref(), b"weights!");
        drop(slice);
        drop(bytes);
        let retained = weak
            .upgrade()
            .expect("alias keepalive retains managed bytes");
        assert_eq!(&retained[32..40], b"weights!");
        drop(retained);
        std::thread::spawn(move || -> Result<()> {
            assert!(cudarc::driver::result::ctx::get_current()?.is_none());
            drop(keepalive);
            assert!(
                cudarc::driver::result::ctx::get_current()?.is_none(),
                "managed owner cleanup must restore no current context"
            );
            Ok(())
        })
        .join()
        .map_err(|_| anyhow::anyhow!("managed owner drop thread panicked"))??;
        assert!(
            weak.upgrade().is_none(),
            "the final managed byte owner expired"
        );
        Ok(())
    }

    /// A constructor-only ABBA screen, not a model-readiness benchmark. The
    /// invoking harness must hold the exclusive GB10 lease; memory admission
    /// is an additional guard, not a substitute for that reservation. This
    /// test-only byte count is never read by the production startup path.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires an exclusive GB10 lease and INK_MANAGED_INIT_SCREEN_BYTES"]
    fn managed_weight_initialization_timing_screen() -> Result<()> {
        const MIN_BYTES: u64 = 1 << 20;
        const MAX_BYTES: u64 = 89_022_731_951;
        const MIN_AVAILABLE_BYTES: u64 = 115_000_000 * 1024;
        const SAMPLE_BYTES: usize = 64;
        let bytes = std::env::var("INK_MANAGED_INIT_SCREEN_BYTES")
            .context("explicit test-only INK_MANAGED_INIT_SCREEN_BYTES is required")?
            .parse::<u64>()
            .context("INK_MANAGED_INIT_SCREEN_BYTES must be a byte count")?;
        anyhow::ensure!(
            (MIN_BYTES..=MAX_BYTES).contains(&bytes),
            "INK_MANAGED_INIT_SCREEN_BYTES must be in {MIN_BYTES}..={MAX_BYTES}"
        );
        let len = usize::try_from(bytes).context("screen allocation does not fit usize")?;
        let parallel = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(MAX_ZERO_WORKERS);
        // Keep primary-context startup outside each measured fresh allocation,
        // and retain it across every arena's synchronous destruction.
        let _primary = cudarc::driver::CudaContext::new(0)?;
        for (arm, workers) in [1, parallel, parallel, 1].into_iter().enumerate() {
            let available = mem_available_bytes()?;
            anyhow::ensure!(
                available >= MIN_AVAILABLE_BYTES,
                "managed initialization screen arm {} requires at least 115000000 KiB available; found {} KiB",
                arm + 1,
                available / 1024
            );
            let started = std::time::Instant::now();
            let arena = ManagedWeightArena::new(len, workers)?;
            let new_seconds = started.elapsed().as_secs_f64();
            // The CPU tests check exhaustive coverage. Keep this timing screen
            // bounded to three small samples without another full read pass.
            for offset in [0, len / 2, len - SAMPLE_BYTES] {
                assert!(
                    arena.as_slice()[offset..offset + SAMPLE_BYTES]
                        .iter()
                        .all(|byte| *byte == 0),
                    "nonzero bytes at offset {offset}"
                );
            }
            let dropping = std::time::Instant::now();
            drop(arena);
            let drop_seconds = dropping.elapsed().as_secs_f64();
            println!(
                "{}",
                serde_json::json!({
                    "event": "managed_weight_initialization_screen",
                    "arm": arm + 1,
                    "bytes": bytes,
                    "workers": workers,
                    "active_workers": weight_zero_workers(len, workers),
                    "new_seconds": new_seconds,
                    "drop_seconds": drop_seconds,
                    // Excludes zero sampling and admission/accounting reads.
                    "total_seconds": new_seconds + drop_seconds,
                    "mem_available_before_bytes": available,
                    "mem_available_before_kib": available / 1024,
                })
            );
        }
        Ok(())
    }

    #[test]
    fn mmap_weight_arena_reads_and_alias_keepalive_preserve_the_same_storage() -> Result<()> {
        for storage in [WeightStorage::Host, WeightStorage::HostReadOnly] {
            let mut arena = WeightArena::new(8192, storage, 2)?;
            assert!(arena.iter().all(|byte| *byte == 0));
            arena[32..40].copy_from_slice(b"weights!");
            let base = arena.as_ptr() as usize;
            let bytes = arena.finish(storage)?;
            let weak = bytes.downgrade();
            let slice = bytes.slice(32..40);
            let (mapped_base, len, keepalive) = weight_arena_mapping(&bytes);
            assert_eq!((mapped_base, len), (base, 8192));
            assert_eq!(slice.as_ref(), b"weights!");
            drop(bytes);
            drop(slice);
            let retained = weak.upgrade().expect("the GPU keepalive retains the mmap");
            assert_eq!(retained.as_ptr() as usize, base);
            assert_eq!(&retained[32..40], b"weights!");
            drop(retained);
            drop(keepalive);
            assert!(
                weak.upgrade().is_none(),
                "the final owner releases the mmap"
            );
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mmap_weight_arena_protection_matches_the_selected_arm() -> Result<()> {
        for storage in [WeightStorage::Host, WeightStorage::HostReadOnly] {
            let mut arena = WeightArena::new(8192, storage, 2)?;
            arena.fill(7);
            let bytes = arena.finish(storage)?;
            let ptr = bytes.as_ptr() as usize;
            let maps = std::fs::read_to_string("/proc/self/maps")?;
            let permissions = maps
                .lines()
                .find_map(|line| {
                    let mut fields = line.split_whitespace();
                    let (lo, hi) = fields.next()?.split_once('-')?;
                    let lo = usize::from_str_radix(lo, 16).ok()?;
                    let hi = usize::from_str_radix(hi, 16).ok()?;
                    let permissions = fields.next()?;
                    (lo <= ptr && ptr < hi).then_some(permissions)
                })
                .expect("the dedicated arena is mapped");
            assert!(permissions.starts_with('r'));
            assert_eq!(
                permissions.as_bytes()[1] == b'w',
                storage == WeightStorage::Host
            );
            assert!(bytes.iter().all(|byte| *byte == 7));
        }
        Ok(())
    }

    #[test]
    fn startup_dense_selection_preserves_non_mtp_layer_and_global_rules() {
        let layers = 2..4;
        let cases = [
            ("model.llm.layers.2.attn_norm.weight", Some(2), false, true),
            ("model.llm.layers.3.attn_norm.weight", Some(3), false, true),
            ("model.llm.layers.1.attn_norm.weight", Some(1), false, false),
            ("model.llm.layers.4.attn_norm.weight", Some(4), false, false),
            // Selection uses the indexed layer, not a second parse of the name.
            ("model.llm.layers.2.attn_norm.weight", None, false, false),
            ("model.llm.embed.weight", None, false, false),
            ("model.llm.embed.weight", None, true, true),
            ("model.llm.layers.12.attn_norm.weight", Some(12), true, true),
            // Only the exact MTP namespace is optional.
            ("model.mtp_extra.layers.2.weight", Some(2), false, true),
        ];
        for drafts in [false, true] {
            for (name, layer, required_global, expected) in cases {
                assert_eq!(
                    dense_in_startup_share(name, layer, &layers, required_global, drafts),
                    expected,
                    "{name}, layer={layer:?}, global={required_global}, drafts={drafts}",
                );
            }
        }
    }

    #[test]
    fn startup_dense_selection_gates_mtp_but_honors_explicit_globals() {
        let layers = 2..4;
        let cases = [
            ("model.mtp.layers.2.input_proj.weight", Some(2), true),
            (
                "model.mtp.layers.3.transformer_block.attn.wq_du.weight",
                Some(3),
                true,
            ),
            ("model.mtp.layers.1.input_proj.weight", Some(1), false),
            ("model.mtp.layers.4.input_proj.weight", Some(4), false),
            ("model.mtp.output.weight", None, false),
        ];
        for (name, layer, in_range) in cases {
            for drafts in [false, true] {
                assert_eq!(
                    dense_in_startup_share(name, layer, &layers, false, drafts),
                    drafts && in_range,
                    "implicit {name}, drafts={drafts}",
                );
                assert!(
                    dense_in_startup_share(name, layer, &layers, true, drafts),
                    "explicit {name}, drafts={drafts}",
                );
                assert!(
                    dense_in_startup_share(name, layer, &(2..2), true, drafts),
                    "explicit globals remain required even for an empty layer range",
                );
            }
        }
    }

    /// A dense leaf with the checkpoint's real dims, as `copy_share` sees it.
    fn leaf(dims: &[u64], layer: Option<i64>) -> Leaf {
        let bytes: usize = dims.iter().product::<u64>() as usize * 2;
        Leaf {
            elem: Elem::Bf16,
            dims: dims.to_vec(),
            bytes: anybytes::Bytes::from_source(vec![0u8; bytes]),
            layer,
        }
    }

    /// The lane facts the 0:16 measurement ran with: BF16 router, no drafting.
    fn placed(w: super::super::budget::DenseWeights) -> AdmissionPolicy {
        AdmissionPolicy::new(
            AllocatorConfig::ExclusivePages,
            StorageDType::F32,
            StorageDType::F32,
            StorageDType::Bf16,
        )
        .with_dense_weights(w)
        .with_router_bf16(true)
    }

    /// The table in [`device_weight_bytes`], summed, against what the run
    /// measured its pool holding at `INK_LAYERS=0:16`.
    ///
    /// Not a restatement of the function: the shapes are the pile's and the
    /// two totals are `pool[after stack]` live from the two arms' own logs --
    /// 1.41 GiB aliased and 4.94 GiB device -- so this fails if the enumeration
    /// drifts from either measurement.
    #[test]
    fn the_pool_charge_reproduces_both_measured_arms_at_sixteen_layers() {
        use super::super::budget::DenseWeights;
        // layers 0..1 dense, 2..15 routed, plus the unembedding.
        let mut share: Vec<(String, Leaf)> = Vec::new();
        for l in 0..16i64 {
            let p = format!("model.llm.layers.{l}.");
            for (nm, dims) in [
                ("attn.wq_du.weight", vec![4096, 4096]),
                ("attn.wk_dv.weight", vec![1024, 4096]),
                ("attn.wv_dv.weight", vec![1024, 4096]),
                ("attn.wr_du.weight", vec![512, 4096]),
                ("attn.wo_ud.weight", vec![4096, 4096]),
                // Widened to f32 and uploaded, never bound as a GEMM operand.
                ("attn.k_norm.weight", vec![128]),
                ("attn.k_sconv.weight", vec![1024, 1, 4]),
                ("attn.rel_logits_proj.proj", vec![16, 512]),
                ("attn_norm.weight", vec![4096]),
            ] {
                share.push((format!("{p}{nm}"), leaf(&dims, Some(l))));
            }
            if l < 2 {
                share.push((
                    format!("{p}mlp.w13_dn.weight"),
                    leaf(&[32768, 4096], Some(l)),
                ));
                share.push((
                    format!("{p}mlp.w2_md.weight"),
                    leaf(&[4096, 16384], Some(l)),
                ));
            } else {
                share.push((format!("{p}mlp.gate.weight"), leaf(&[258, 4096], Some(l))));
                share.push((
                    format!("{p}mlp.shared_experts.shared_w13_weight"),
                    leaf(&[2, 4096, 4096], Some(l)),
                ));
                share.push((
                    format!("{p}mlp.shared_experts.shared_w2_weight"),
                    leaf(&[2, 4096, 2048], Some(l)),
                ));
            }
        }
        // In the arena on this node, and NOT bound: the embedding is a lookup.
        share.push(("model.llm.embed.weight".into(), leaf(&[201024, 4096], None)));
        share.push((
            "model.llm.unembed.weight".into(),
            leaf(&[201024, 4096], None),
        ));
        // Candidate MTP leaves: selected into the arena and bound only when
        // drafting. This fixture tests the device charge independently.
        for l in 0..8i64 {
            let p = format!("model.mtp.layers.{l}.");
            share.push((
                format!("{p}input_proj.weight"),
                leaf(&[4096, 8192], Some(l)),
            ));
            share.push((
                format!("{p}transformer_block.attn.wq_du.weight"),
                leaf(&[4096, 4096], Some(l)),
            ));
        }

        let sum = |policy: AdmissionPolicy| -> u64 {
            share
                .iter()
                .map(|(n, l)| device_weight_bytes(n, l, policy))
                .sum()
        };

        let device = sum(placed(DenseWeights::DevicePool));
        assert_eq!(
            device, 5_300_944_896,
            "the device arm's 4.94 GiB, to the byte"
        );
        let aliased = sum(placed(DenseWeights::Aliased));
        assert_eq!(
            aliased, 1_506_672_640,
            "the aliased arm's 1.40 GiB, to the byte"
        );
        // The win's whole memory cost, which is what the gate could not see.
        assert_eq!(device - aliased, 3_794_272_256);
        // Drafting binds the MTP heads as well as selecting their arena bytes.
        let drafting = sum(placed(DenseWeights::DevicePool).with_drafting(true));
        assert_eq!(drafting - device, 8 * (67_108_864 + 33_554_432));
    }

    /// `INK_FUSE_QKVR` costs 52 MiB a layer in BOTH placement arms, and the
    /// MTP heads do not pay it.
    ///
    /// The four projections are `[4096, 4096]`, `[1024, 4096]`, `[1024, 4096]`
    /// and `[512, 4096]` BF16 -- 32 + 8 + 8 + 4 = 52 MiB, which is the 54.5 MB
    /// `burn::fuse_qkvr` names. This is the term that was MISSING while the
    /// switch existed: under the default aliased placement every one of these
    /// four charged zero, so a concatenation admission could not see was
    /// allocated on every layer.
    ///
    /// The MTP clause is not decoration. `model.mtp.layers.N.
    /// transformer_block.attn.wq_du.weight` ends with the same suffix the LLM's
    /// does, and `MtpOwned` carries no fused field, so a suffix-only test would
    /// charge a bind that never happens.
    #[test]
    fn the_fused_qkvr_bind_is_charged_in_both_placement_arms() {
        use super::super::budget::DenseWeights;
        const MIB: u64 = 1 << 20;
        let share: Vec<(String, Leaf)> = vec![
            (
                "model.llm.layers.0.attn.wq_du.weight".into(),
                leaf(&[4096, 4096], Some(0)),
            ),
            (
                "model.llm.layers.0.attn.wk_dv.weight".into(),
                leaf(&[1024, 4096], Some(0)),
            ),
            (
                "model.llm.layers.0.attn.wv_dv.weight".into(),
                leaf(&[1024, 4096], Some(0)),
            ),
            (
                "model.llm.layers.0.attn.wr_du.weight".into(),
                leaf(&[512, 4096], Some(0)),
            ),
            // Same suffix, never fused.
            (
                "model.mtp.layers.0.transformer_block.attn.wq_du.weight".into(),
                leaf(&[4096, 4096], Some(0)),
            ),
        ];
        let sum = |policy: AdmissionPolicy| -> u64 {
            share
                .iter()
                .map(|(n, l)| device_weight_bytes(n, l, policy))
                .sum()
        };

        // Aliased: the four separate binds are views of the arena and cost
        // nothing; the concatenation is the whole charge.
        let alias_off = placed(DenseWeights::Aliased);
        assert_eq!(sum(alias_off), 0);
        assert_eq!(sum(alias_off.with_fused_qkvr(true)), 52 * MIB);

        // Device pool: the separate binds are already charged, and the
        // concatenation is a SECOND residency on top of them.
        let dev_off = placed(DenseWeights::DevicePool);
        assert_eq!(sum(dev_off), 52 * MIB);
        assert_eq!(sum(dev_off.with_fused_qkvr(true)), 104 * MIB);

        // Drafting binds the MTP head; fusing still does not fuse it.
        let drafting = placed(DenseWeights::DevicePool).with_drafting(true);
        assert_eq!(sum(drafting), 52 * MIB + 32 * MIB);
        assert_eq!(sum(drafting.with_fused_qkvr(true)), 104 * MIB + 32 * MIB);

        // 21 layers on the production split, against the 19.68 GiB of headroom
        // the tail node's own admission line printed at INK_LAYERS=21:42.
        assert_eq!(21 * 52 * MIB, 1092 * MIB);
        assert!(21 * 52 * MIB < 19 * (1 << 30));
    }

    /// A synthetic expert with the real checkpoint's proportions, scaled down.
    /// `PackedExpert` is a plain struct, so this needs no checkpoint.
    fn expert(rows: usize, logical: usize) -> PackedExpert {
        let elems = rows * logical;
        PackedExpert {
            codes: (0..elems / 2).map(|i| i as u8).collect(),
            scales: (0..elems / NVFP4_BLOCK).map(|i| (i % 251) as u8).collect(),
            scale2: 0.125,
            rows,
            cols: logical / 2,
        }
    }

    /// SubSlices keeps its two-ladder-page charge; ExclusivePages is charged
    /// the bucket ratio of the live set instead of the same device fraction.
    #[test]
    fn the_allocator_charge_is_per_strategy_and_not_one_device_fraction() {
        let machine = 128 * GIB;
        let attention = 7 * GIB;
        let storage = (StorageDType::F32, StorageDType::F32, StorageDType::Bf16);
        let subslices =
            AdmissionPolicy::new(AllocatorConfig::SubSlices, storage.0, storage.1, storage.2);
        let exclusive = AdmissionPolicy::new(
            AllocatorConfig::ExclusivePages,
            storage.0,
            storage.1,
            storage.2,
        );
        let base = GIB / 5 + 41 * GIB / 200 * 8 + 4 * GIB + attention;
        assert_eq!(
            run_overhead_bytes(8, attention, machine, subslices),
            base + 40 * GIB
        );
        // 0.823 * 7 GiB, not 40: the ladder is the device's, the buckets are
        // the workload's.
        let excl = run_overhead_bytes(8, attention, machine, exclusive);
        assert!(excl > base + 5 * GIB && excl < base + 6 * GIB, "{excl}");
    }

    /// The 2026-08-23 spark incident, in the numbers its own logs printed.
    ///
    /// `INK_LAYERS=0:16` at 2048 tokens needs 112.65 GiB and the node is 119.63,
    /// so it fits the machine -- but 43.9 GiB of AnonPages was already held and
    /// only 69.93 GiB was free. The old gate compared the 66.50 GiB weight
    /// SHARE against those 69.93, found room, and admitted it; the startup copy
    /// then ran 1384.5 s at 0.05 GiB/s and took all 15.1 GiB of swap, with sshd
    /// unable to answer for the whole of it. The identical run on the idle node
    /// copies in 15.3 s.
    ///
    /// Both directions are asserted, because the failure the fix must not
    /// introduce is refusing a run that would have fit.
    #[test]
    fn a_run_that_fits_the_machine_is_refused_when_the_machine_is_busy() {
        let gib = |g: f64| (g * GIB as f64) as u64;
        let (machine, need, share) = (gib(119.63), gib(112.65), gib(66.50));
        // The comparison the old gate made, and passed: the share alone really
        // does fit the busy node. That is why reading it was not enough.
        assert!(
            share < gib(69.93),
            "the old test found room and was still wrong"
        );
        assert_eq!(admit(need, machine, gib(69.93)), Admission::NodeBusy);
        // ...and the same run on the idle node is still admitted, so this is
        // not a blanket tightening that costs work which would have run.
        assert_eq!(admit(need, machine, gib(114.75)), Admission::Fits);
        // The 5-token arm of the same A/B, which ran twelve times cleanly.
        assert_eq!(admit(gib(111.40), machine, gib(114.75)), Admission::Fits);
        // Too big for the node at all is a different verdict because it takes a
        // different fix: more nodes, not more patience.
        assert_eq!(
            admit(gib(130.0), machine, gib(119.0)),
            Admission::OverMachine
        );
        // An exactly-full node is admitted -- the floors inside
        // `run_overhead_bytes` are what leaves the kernel its room, and
        // subtracting a second margin here would double-charge it.
        assert_eq!(admit(machine, machine, machine), Admission::Fits);
    }

    /// The blob states the LOGICAL width, so nothing downstream needs an
    /// `original_shape` field to correct it.
    #[test]
    fn the_blob_states_logical_dims_not_packed_ones() {
        let q = expert(64, 128);
        assert_eq!(q.cols, 64, "the checkpoint's packed width");
        let blob = expert_blob(&q).expect("well formed");
        let view: TensorView = blob.try_from_blob().expect("decodes");
        assert_eq!(view.dims(), &[64, 128], "logical, twice the packed width");
        assert_eq!(view.elems(), 64 * 128);
    }

    /// Codes, scales and the global scale survive as one atomic artifact —
    /// which is the point, since the checkpoint binds them only by name.
    #[test]
    fn all_three_planes_round_trip_through_one_handle() {
        let q = expert(64, 128);
        let blob = expert_blob(&q).expect("well formed");
        let view: TensorView = blob.try_from_blob().expect("decodes");
        let (codes, scales, scale2) = split_payload(view.payload(), view.elems()).expect("splits");
        assert_eq!(codes, &q.codes[..], "codes");
        assert_eq!(scales, &q.scales[..], "block scales");
        assert_eq!(scale2, q.scale2, "global scale");
    }

    /// A scales plane of the wrong length still decodes — against the wrong
    /// blocks — so it has to be refused rather than discovered later as numbers
    /// that look plausible.
    #[test]
    fn a_mis_sized_scales_plane_is_refused() {
        let mut q = expert(64, 128);
        q.scales.truncate(q.scales.len() - 1);
        let err = expert_blob(&q).expect_err("must refuse");
        assert!(format!("{err}").contains("scales are"), "{err}");
    }

    #[test]
    fn mis_sized_codes_are_refused_too() {
        let mut q = expert(64, 128);
        q.codes.push(0);
        let err = expert_blob(&q).expect_err("must refuse");
        assert!(format!("{err}").contains("codes are"), "{err}");
    }

    #[test]
    fn a_layer_is_read_from_the_name_and_absent_when_there_is_none() {
        assert_eq!(
            layer_of("model.llm.layers.10.mlp.experts.w13_weight"),
            Some(10)
        );
        assert_eq!(layer_of("model.llm.layers.0.mlp.w13_dn"), Some(0));
        assert_eq!(layer_of("model.mtp.layers.3.attn.wq_du"), Some(3));
        // No layer at all: absent, not zero. A tensor that silently joined
        // layer 0 would ship to the wrong machine in a 21/21 split.
        assert_eq!(layer_of("model.llm.embed"), None);
        assert_eq!(layer_of("model.audio.encoder.weight"), None);
    }

    /// A layer query returns only the layers asked for. The negative half
    /// matters more than the positive one: a node that received experts from a
    /// layer it does not hold would compute with weights it has no business
    /// having, and the arithmetic would look fine.
    #[test]
    fn a_layer_query_excludes_the_layers_it_did_not_ask_for() {
        use triblespace::macros::entity;
        use triblespace::prelude::*;

        let mut space = TribleSet::new();
        for (layer, idx) in [(3i64, 0i64), (3, 1), (30, 0)] {
            // A distinct handle per row, so nothing collapses by accident.
            let mut q = expert(16, 32);
            q.scale2 = layer as f32 + idx as f32 / 100.0;
            let handle = expert_blob(&q).expect("well formed").get_handle();
            space += entity! { &ufoid() @
                attrs::layer: layer,
                attrs::expert_index: idx,
                attrs::weight_nvfp4_2: handle,
            }
            .into_facts();
        }

        let held = experts_in_layers(&space, 0..=20);
        assert_eq!(held.len(), 2, "two experts in layer 3");
        assert!(
            held.iter().all(|r| r.layer == 3),
            "layer 30 must not appear"
        );
        assert_eq!(held[0].expert, 0, "and they come back ordered");
        assert_eq!(held[1].expert, 1);

        assert_eq!(
            experts_in_layers(&space, 21..=41).len(),
            1,
            "the other half"
        );
        assert_eq!(
            experts_in_layers(&space, 100..=200).len(),
            0,
            "and an empty range is empty"
        );
    }

    /// Inkling's real proportions: a 4096x4096 expert packs to 4096x2048 bytes
    /// with 4096x256 block scales, which is what the checkpoint stores.
    #[test]
    fn the_real_expert_proportions_line_up() {
        let (rows, logical) = (4096usize, 4096usize);
        let elems = rows * logical;
        assert_eq!(
            elems / 2,
            4096 * 2048,
            "packed width matches the checkpoint"
        );
        assert_eq!(elems / NVFP4_BLOCK, 4096 * 256, "scale width matches");
        assert_eq!(
            NVFP4::payload_len(elems),
            Some(elems / 2 + elems / NVFP4_BLOCK + 4)
        );
    }
}
