//! Strict native BF16 pile -> raw CUDA tensor binding.
//!
//! Query the required typed leaf at the consuming model slot, fetch its
//! `Blob<Tensor<BF16, R>>` from the frozen store, then bind it here. This module
//! owns CUDA registrations, not a catalogue of model facts. There is no upload,
//! dtype conversion, source-file reopen, or CPU model path.
//!
//! Reuse one binder per device/session. Each (actual owner, payload offset,
//! payload size) is registered once, up to an explicit registration budget.
//! Each registered region starts at the payload's containing page and ends
//! EXACTLY at the payload end, not at mmap capacity or a rounded-up page end.
//! CubeCL's external allocation table
//! retains owners until runtime/storage teardown, NOT until the last tensor or
//! binder drops. Recreating binders defeats this local registration bound.
//! Owner capacity and registered spans are counted separately; neither counts
//! resident physical RAM. Overlapping registered page prefixes count per span.
//!
//! Binding is unsafe because `Blob` plus `MmapRaw` does not prove pile
//! provenance or immutability of the preceding partial page. The caller must
//! establish the genuine append-only pile premise, including that prefix.
//! Only the validated payload is exposed as a tensor, never a past-EOF tail.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    client::ComputeClient,
    cuda::{CudaDevice, CudaRuntime, supports_zero_copy_host},
    server::Handle,
    Runtime,
};
use memmap2::MmapRaw;
use std::sync::Arc;
use triblespace::core::blob::{
    Blob,
    encodings::tensor::{MAX_RANK, TENSOR_HEADER_LEN, Tensor, elements::BF16},
};

/// Successful binds, including repeated binds, are counted explicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AliasStats {
    pub registrations: usize,
    pub owners: usize,
    pub owner_capacity_bytes: u64,
    pub registered_span_bytes: u64,
    pub bindings: u64,
    pub aliased_bytes: u64,
}

struct Registration {
    owner: Arc<MmapRaw>,
    payload_offset: usize,
    payload_size: u64,
    handle: Handle,
}

/// Device-bound, bounded registration resources; no caller-supplied raw pointer
/// or client/device pair enters the binding API. Pile provenance is an explicit
/// unsafe obligation on `bind_pile_leaf`, not inferred from the owner downcast.
pub struct CudaBf16Aliases {
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
    page_size: usize,
    max_registrations: usize,
    registrations: Vec<Registration>,
    stats: AliasStats,
}

impl CudaBf16Aliases {
    /// Initialize the selected CUDA runtime and require host-page-table access.
    /// Unsupported aliasing returns an error, never an upload path. CUDA runtime
    /// initialization/driver failures retain the upstream runtime's panic policy.
    pub fn new(device: CudaDevice, max_registrations: usize) -> Result<Self, String> {
        if max_registrations == 0 {
            return Err("BF16 alias registration budget must be nonzero".into());
        }
        // CubeCL's DeviceId stores the CUDA ordinal as u16; prevent truncation.
        if device.index > u16::MAX as usize {
            return Err("CUDA device index exceeds CubeCL's u16 device identity".into());
        }
        if !cfg!(target_endian = "little") {
            return Err("native BF16 CUDA aliases require a little-endian host".into());
        }
        let page_size = host_page_size()?;
        // Initialize before the capability helper: its cached CUDA attribute
        // lookup must not run against an uninitialized driver.
        let client = CudaRuntime::client(&device);
        if !supports_zero_copy_host(device.index) {
            return Err(format!(
                "CUDA device {} lacks pageable host memory with host page tables; strict BF16 alias refused",
                device.index
            ));
        }
        Ok(Self {
            client,
            device,
            page_size,
            max_registrations,
            registrations: Vec::new(),
            stats: AliasStats::default(),
        })
    }

    pub fn stats(&self) -> AliasStats {
        self.stats
    }

    /// The actual device this registration session owns. Consumers creating
    /// related GPU operators must not invent a second client/device pairing.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Bind exactly the typed BF16 payload, including odd element counts.
    /// Empty tensors, malformed/oversized shapes, non-pile backing, misaligned
    /// payloads and exhausted registration budgets are descriptive refusals.
    /// Returned handles are immutable under CubeCL's normal reuse policy.
    /// Raw unsafe kernels must still never write to a weight alias.
    ///
    /// # Safety
    /// If the bytes have an `MmapRaw` owner, they must originate from a genuine
    /// validated `Pile`/`PileSnapshot` observation (zero-copy slices are fine).
    /// Its backing file must stay append-only and must not be externally
    /// changed or truncated while any resulting tensor/runtime registration
    /// exists. This includes bytes in the same file-backed page BEFORE the
    /// payload, down to the page boundary: the generic CubeCL API requires
    /// that entire registered prefix-plus-payload span to remain immutable.
    /// A downcast only proves ownership, NOT this provenance/immutability.
    /// Non-MmapRaw owners are refused and require no file-provenance premise.
    ///
    /// F16 is not an alternative interpretation accepted by this API:
    /// ```compile_fail
    /// use mary::nn::cuda_bf16_alias::CudaBf16Aliases;
    /// use triblespace::core::blob::{Blob, encodings::tensor::{Tensor, elements::F16}};
    /// fn wrong_dtype(binder: &mut CudaBf16Aliases, blob: Blob<Tensor<F16, 1>>) {
    ///     let _ = unsafe { binder.bind_pile_leaf(blob) };
    /// }
    /// ```
    pub unsafe fn bind_pile_leaf<const R: usize>(
        &mut self,
        blob: Blob<Tensor<BF16, R>>,
    ) -> Result<CubeTensor<CudaRuntime>, String> {
        // Core's current TensorView decoder multiplies u64 dimensions without
        // checked arithmetic. Preflight before calling it; do not broaden core.
        let shape = checked_shape(&blob)?;
        let view = crate::leaf::read_leaf(blob).map_err(|e| e.to_string())?;
        let payload = view.payload();
        let address = payload.as_ptr() as usize;
        let owner = payload.clone().downcast_to_owner::<MmapRaw>()
            .map_err(|_| "BF16 tensor is not backed by a pile MmapRaw owner".to_string())?;
        if !address.is_multiple_of(256) {
            return Err("BF16 tensor payload is not 256-byte aligned".into());
        }
        let base = owner.as_ptr() as usize;
        let length = owner.len();
        let page_start = address - address % self.page_size;
        let end = address.checked_add(payload.len()).ok_or("BF16 payload address overflow")?;
        let map_end = base.checked_add(length).ok_or("pile mapping address overflow")?;
        if page_start < base || end > map_end {
            return Err("BF16 payload/page prefix lies outside its actual mmap owner".into());
        }
        let payload_offset = address - base;
        let offset = u64::try_from(address - page_start).map_err(|_| "BF16 offset exceeds u64")?;
        let size = u64::try_from(payload.len()).map_err(|_| "BF16 payload exceeds u64")?;
        let map_size = u64::try_from(length).map_err(|_| "pile mapping exceeds u64")?;
        let span_size = offset.checked_add(size).ok_or("BF16 registered span overflow")?;
        let bindings = self.stats.bindings.checked_add(1).ok_or("BF16 bind counter overflow")?;
        let aliased_bytes = self.stats.aliased_bytes.checked_add(size)
            .ok_or("BF16 alias-byte counter overflow")?;

        let index = if let Some(index) = self.registrations.iter().position(|entry| {
            Arc::ptr_eq(&entry.owner, &owner)
                && entry.payload_offset == payload_offset
                && entry.payload_size == size
        }) {
            index
        } else {
            if self.registrations.len() == self.max_registrations {
                return Err(format!("BF16 alias registration budget {} exhausted", self.max_registrations));
            }
            let new_owner = !self.registrations.iter().any(|entry| Arc::ptr_eq(&entry.owner, &owner));
            let owners = self.stats.owners.checked_add(usize::from(new_owner))
                .ok_or("BF16 owner counter overflow")?;
            let owner_capacity_bytes = self.stats.owner_capacity_bytes
                .checked_add(if new_owner { map_size } else { 0 })
                .ok_or("BF16 owner-capacity counter overflow")?;
            let registered_span_bytes = self.stats.registered_span_bytes.checked_add(span_size)
                .ok_or("BF16 registered-span counter overflow")?;
            let keepalive: Arc<dyn std::any::Any + Send + Sync> = owner.clone();
            // SAFETY: the caller establishes a genuine immutable pile prefix.
            // page_start is page-aligned and in this actual owner; the span
            // ends exactly at the checked payload end, never beyond EOF.
            // Earlier bytes in its first page are earlier bytes of that same
            // immutable file prefix. No page-multiple LENGTH is required by
            // the CUDA implementation or the generic client's safety clause.
            // The private client was created for self.device, whose host-page
            // access was checked. CubeCL stores keepalive in external storage.
            let handle = unsafe {
                self.client.register_external_aliased(
                    page_start as *mut core::ffi::c_void,
                    span_size,
                    offset,
                    size,
                    keepalive,
                )
            };
            self.registrations.push(Registration { owner, payload_offset, payload_size: size, handle });
            self.stats.registrations = self.registrations.len();
            self.stats.owners = owners;
            self.stats.owner_capacity_bytes = owner_capacity_bytes;
            self.stats.registered_span_bytes = registered_span_bytes;
            self.registrations.len() - 1
        };
        // CUDA storage has ALREADY applied offset and returned a size-byte
        // handle. Adding owner-relative handle offsets would apply them twice.
        let handle = self.registrations[index].handle.clone();
        self.stats.bindings = bindings;
        self.stats.aliased_bytes = aliased_bytes;
        Ok(CubeTensor::new_contiguous(
            self.client.clone(), self.device.clone(), shape.as_slice().into(), handle, DType::BF16,
        ))
    }
}

pub(crate) fn checked_shape<const R: usize>(blob: &Blob<Tensor<BF16, R>>) -> Result<Vec<usize>, String> {
    if R == 0 || R > MAX_RANK {
        return Err(format!("BF16 CUDA alias rank {R} is outside 1..={MAX_RANK}"));
    }
    if blob.bytes.len() < TENSOR_HEADER_LEN {
        return Err("BF16 tensor is shorter than its 256-byte header".into());
    }
    let mut shape = Vec::with_capacity(R);
    let mut elements = 1usize;
    for axis in 0..R {
        let at = axis * 8;
        let dim = u64::from_le_bytes(blob.bytes[at..at + 8].try_into().expect("checked header"));
        let dim = usize::try_from(dim).map_err(|_| "BF16 dimension exceeds usize")?;
        elements = elements.checked_mul(dim).ok_or("BF16 shape product overflow")?;
        if dim == 0 || elements > u32::MAX as usize {
            return Err("BF16 alias requires nonempty dimensions in the u32 kernel index domain".into());
        }
        shape.push(dim);
    }
    let expected = elements.checked_mul(2).ok_or("BF16 payload length overflow")?;
    if blob.bytes.len() - TENSOR_HEADER_LEN != expected {
        return Err(format!("BF16 payload has {} bytes; shape requires {expected}",
            blob.bytes.len() - TENSOR_HEADER_LEN));
    }
    Ok(shape)
}

#[cfg(unix)]
fn host_page_size() -> Result<usize, String> {
    // SAFETY: sysconf reads an OS property and takes no pointer arguments.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(value).ok().filter(|&n| n != 0 && n.is_power_of_two())
        .ok_or_else(|| "cannot establish host page size for strict BF16 aliasing".into())
}

#[cfg(not(unix))]
fn host_page_size() -> Result<usize, String> {
    Err("strict pile BF16 CUDA aliasing currently requires a Unix mmap host".into())
}
