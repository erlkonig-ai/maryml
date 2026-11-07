//! Device buffers allocated at their own size.
//!
//! CubeCL's default pool packs buffers into pages sized from
//! cuDeviceTotalMem/4: on GB10 a 0.5-4 GB buffer takes an 8.16 GB page, and
//! each thread's stream keeps its own pages. A model's weights are fixed-size
//! buffers that live as long as the model, so they belong in CubeCL's
//! persistent pool, where reservation equals logical size.

use cubecl::{cuda::CudaRuntime, prelude::ComputeClient};
use std::cell::RefCell;

thread_local! {
    /// How many `exact` scopes this thread is inside, per device server. The
    /// mode belongs to one server's stream, so a scope on another client must
    /// switch that client's own mode, not count as nested.
    static DEPTH: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

/// One device server: every clone of a client shares its properties.
fn server(client: &ComputeClient<CudaRuntime>) -> usize {
    client.properties() as *const _ as usize
}

/// Add `step` to this thread's depth for `key` and return the new depth.
fn depth(key: usize, step: isize) -> usize {
    DEPTH.with(|d| {
        let mut d = d.borrow_mut();
        let i = d.iter().position(|(k, _)| *k == key).unwrap_or_else(|| { d.push((key, 0)); d.len() - 1 });
        d[i].1 = d[i].1.checked_add_signed(step).expect("exact scope depth");
        let n = d[i].1;
        if n == 0 { d.swap_remove(i); }
        n
    })
}

/// Run `f` with this thread's stream allocating every buffer at its own size
/// (CubeCL's persistent pool). Freed exact buffers are reused for the same
/// size until memory_cleanup. Reentrant per device server: only the outermost
/// scope on a server switches its stream back to Auto, so a buffer type can
/// allocate exactly by itself while a caller already holds a wider scope.
pub(crate) fn exact<T>(client: &ComputeClient<CudaRuntime>, f: impl FnOnce() -> T) -> T {
    struct Restore<'a>(&'a ComputeClient<CudaRuntime>);
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            if depth(server(self.0), -1) == 0 {
                // SAFETY: the mode belongs to this thread's stream only.
                unsafe { self.0.allocation_mode(cubecl::MemoryAllocationMode::Auto) }
            }
        }
    }
    if depth(server(client), 1) == 1 {
        // SAFETY: as above; Restore returns the stream to Auto, also on unwind.
        unsafe { client.allocation_mode(cubecl::MemoryAllocationMode::Persistent) }
    }
    let _restore = Restore(client);
    f()
}
