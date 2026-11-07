//! Device buffers allocated at their own size.
//!
//! CubeCL's default pool packs buffers into pages sized from
//! cuDeviceTotalMem/4: on GB10 a 0.5-4 GB buffer takes an 8.16 GB page, and
//! each thread's stream keeps its own pages. A model's weights are fixed-size
//! buffers that live as long as the model, so they belong in CubeCL's
//! persistent pool, where reservation equals logical size.

use cubecl::{cuda::CudaRuntime, prelude::ComputeClient};

/// Run `f` with this thread's stream allocating every buffer at its own size
/// (CubeCL's persistent pool). Freed exact buffers are reused for the same
/// size until memory_cleanup. Not reentrant: the inner scope would end the
/// outer one.
pub(crate) fn exact<T>(client: &ComputeClient<CudaRuntime>, f: impl FnOnce() -> T) -> T {
    struct Restore<'a>(&'a ComputeClient<CudaRuntime>);
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            // SAFETY: the mode belongs to this thread's stream only.
            unsafe { self.0.allocation_mode(cubecl::MemoryAllocationMode::Auto) }
        }
    }
    // SAFETY: as above; Restore returns the stream to Auto, also on unwind.
    unsafe { client.allocation_mode(cubecl::MemoryAllocationMode::Persistent) }
    let _restore = Restore(client);
    f()
}
