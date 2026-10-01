//! Hand-written CUDA C++ kernels through the cubecl runtime -- no second
//! runtime.
//!
//! JP, 2026-08-30: "I'd be fine with us writing kernels by hand if it made
//! inkling faster." This module is the hatch that lets a kernel be a string of
//! CUDA C++ instead of a `#[cube]` function, while everything AROUND the kernel
//! stays exactly what it is for a generated one: the same stream, the same
//! arena buffers, the same CUDA-graph capture and the same parameter-rewrite
//! patch path, the same by-value `info_st` for scalars, the same PTX cache.
//!
//! # How it rides the existing runtime
//!
//! cubecl-cuda compiles whatever `CompiledKernel.source` string it is handed
//! with NVRTC (`cubecl-cuda/src/compute/context.rs`, `compile_kernel`:
//! `--gpu-architecture=sm_121a`, `-lineinfo`, the CUDA and CCCL include paths)
//! and loads the entry point named by `entrypoint_name`. It never looks at the
//! cube IR again: the trait a kernel implements to reach that path is
//! [`CubeTask`], whose single method returns the source. [`RawCudaKernel`]
//! implements it by returning the hand-written source verbatim.
//!
//! The one thing the runtime still wants from a compiled kernel is its
//! `repr`, the [`CudaComputeKernel`] a generated kernel gets from the C++
//! backend -- `compile_kernel` unwraps it -- and the only thing it READS off
//! it is `shared_memory_size()`, for `validate_shared` against the device
//! limit and for the launch's dynamic shared-memory attribute. So this module
//! builds an empty `CudaComputeKernel` whose body holds exactly one
//! shared-memory allocation of [`RawCudaKernel::with_dynamic_shared`]'s size.
//! Every field of that struct is public and constructible from here; the
//! `cubecl-cpp` dependency in `Cargo.toml` exists for those five type names
//! and nothing else. No change to the cubecl fork was needed.
//!
//! # The argument convention a hand-written kernel must follow
//!
//! The launch binds parameters in the order the C++ backend generates them
//! (`cubecl-cpp` `shared/kernel.rs` `compile_bindings`, `cuda/dialect.rs`
//! `compile_kernel_signature`; the server passes them in that order at
//! `cubecl-cuda/src/compute/context.rs` `execute_task`):
//!
//! ```text
//! extern "C" __global__ void __launch_bounds__(<cube_dim.num_elems()>)
//! <name>(
//!     <T0>* __restrict__ b0,            // RawArgs::buffer, in call order --
//!     <T1>* __restrict__ b1,            // one raw device pointer each, at
//!     ...                               // the handle's byte offset
//!     const __grid_constant__ info_st info   // RawArgs scalars, BY VALUE
//! )
//! ```
//!
//! * **Buffers** are bare device pointers, one per [`RawArgs::buffer`], in
//!   the order the calls were made. No length, rank, shape or stride travels
//!   with them (the previous change made `Array` binding the rule for every
//!   inkling kernel for that reason); pass a length as a scalar if the kernel
//!   needs one. `const` and `__restrict__` are the kernel's to declare.
//! * **Scalars** arrive in ONE struct, by value, as the last parameter. The
//!   kernel declares the struct itself, and the launch copies
//!   `sizeof(info_st)` bytes from the words [`RawArgs`] packed: so the struct's
//!   fields must be the scalars pushed, in push order, at C layout. The packer
//!   aligns each field to its own size and pads the whole to eight bytes,
//!   which IS the C layout of a struct of primitives, so
//!   `RawArgs::new().f32(a).u32(n)` matches `struct info_st { float a;
//!   unsigned int n; };` and `.u32(n).u64(p)` matches `{ unsigned int n;
//!   unsigned long long p; }` including the four bytes of padding. A struct
//!   that declares a field the packer was not given reads past the blob.
//!   With no scalars, omit the parameter: the driver ignores the extra
//!   `kernelParams` entry the server always appends, exactly as it does for a
//!   generated kernel with no `info`.
//! * **Comptime / shape specialisation** is the source. There is no
//!   `#[comptime]`; bake the constant into the string (`const int HD = 128;`,
//!   a template instantiation, an `#if`) and the kernel id is keyed on the
//!   source bytes, so every distinct specialisation compiles, caches and is
//!   looked up separately, the way a generated kernel is keyed on its comptime
//!   arguments. Build the string with `format!` and let the hash do the rest.
//! * **Block size.** [`RawCudaKernel::new`] takes the [`CubeDim`]; it is the
//!   launch's block dimensions AND part of the kernel id. Keep
//!   `__launch_bounds__` in the source consistent with it.
//! * **Shared memory.** Static `__shared__` arrays in the source need nothing
//!   from here (the 48 KiB static limit applies). Above that, declare
//!   `extern __shared__ __align__(16) unsigned char smem[];` and give the
//!   byte count to [`RawCudaKernel::with_dynamic_shared`]: it becomes the
//!   launch's dynamic shared-memory size and the function's
//!   `MAX_DYNAMIC_SHARED_SIZE_BYTES` attribute, validated against the device
//!   limit at compile time, through the same repr field a generated kernel
//!   uses.
//!
//! # CUDA-graph capture and the patch path
//!
//! A capture records a launch as its function handle, grid, block, shared
//! bytes, the pointer list and a copy of the by-value blob (`CapturedLaunch`
//! in `cubecl-cuda/src/compute/server.rs`), and `graph_patch_launch`
//! (server.rs:546-681) rewrites a node from those alone: `ptrs[i]` for the
//! i-th bound buffer, `info` for the blob, `grid` for the count. A raw kernel
//! is captured and patched IDENTICALLY, because those are exactly the terms
//! this convention is stated in: binding `i` is the i-th [`RawArgs::buffer`],
//! the blob is the packed scalars, and the blob rides as a grid constant
//! (`info_is_grid_constant` is true whenever the device supports it, which
//! `compile` insists on below). The patch asserts the new blob has the SAME
//! word count as the captured one -- a kernel parameter is fixed-size -- and a
//! raw kernel's blob is fixed by its struct, so a re-launch of the same kernel
//! with new scalar values always satisfies it. What a raw kernel never has is
//! the DYNAMIC half of the blob (shapes and strides staged through a memcpy
//! node): `RawArgs` sets `dynamic_metadata_offset` to the blob's length, so
//! the server uploads nothing and there is no staging buffer to keep in step.
//!
//! # Limits
//!
//! * Nothing from cubecl's frontend exists inside the source: no `cmma`, no
//!   `plane_sum`, no `Array`, no bounds checks. `mma.sync` and friends are
//!   inline PTX (`asm volatile`), which is what the generated FP4 kernels lower
//!   to anyway. `cuda_fp16.h`, `cuda_bf16.h` and the CCCL headers are on the
//!   include path.
//! * The kernel must not allocate. No device `malloc`/`new`, no
//!   `cudaMalloc`: every buffer comes from the arena through
//!   [`RawArgs::buffer`], which is what keeps a captured region replayable.
//! * `ExecutionMode` is part of the kernel id but changes nothing in the
//!   source; a raw kernel is unchecked by construction, which is why
//!   [`RawCudaKernel::launch`] is `unsafe`.
//! * Grid constants are required (`sm_70` and up; every device this crate
//!   targets). Without them the server would put the scalars in a device
//!   buffer bound AFTER the buffers, a different signature, and `compile`
//!   refuses rather than let the two conventions disagree silently.
//! * The PTX cache (`CUBECL_COMPILATION_CACHE`) is keyed on the kernel id's
//!   stable hash, which for a raw kernel is the type name, the entry point,
//!   an FNV-1a of the source, the dynamic shared bytes, the cube dim and the
//!   mode. Edit the source and the hash moves; the cache cannot serve a stale
//!   PTX for changed text.

use std::collections::HashSet;
use std::sync::Arc;

use cubecl::backtrace::BackTrace;
use cubecl::cuda::{CudaCompiler, CudaComputeKernel, CudaRuntime};
use cubecl::prelude::{
    AddressType, CompiledKernel, ComputeClient, CubeCount, CubeDim, ExecutionMode, KernelId,
    KernelMetadata, StorageType,
};
use cubecl::server::{Binding, Handle, KernelArguments, MetadataBindingInfo};
use cubecl::{CompilationError, Compiler, CubeTask, Info};
use cubecl_cpp::shared::{Body, Elem, Flags, Item, SharedMemory};

/// A hand-written CUDA C++ kernel, launchable on the cubecl CUDA client.
///
/// Cheap to clone -- the source is shared -- because every launch hands the
/// runtime a fresh `Box<dyn CubeTask>`; the runtime compiles once per
/// [`KernelId`] and finds the loaded module by id on every launch after.
#[derive(Clone, Debug)]
pub struct RawCudaKernel {
    /// The `extern "C" __global__` entry point's symbol.
    name: String,
    /// The whole translation unit NVRTC compiles.
    source: Arc<str>,
    /// FNV-1a of `source`, the part of the kernel id that tracks the text.
    source_hash: u64,
    /// The block dimensions at launch, and part of the kernel id.
    cube_dim: CubeDim,
    /// Dynamic shared memory in bytes; zero unless asked for.
    shared_bytes: usize,
}

/// What distinguishes one raw kernel from another inside the one Rust type
/// they share: the [`KernelId`]'s `info`. Hashed with cubecl's stable hasher
/// for the PTX cache, printed by `Debug` into a capture's launch inventory.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RawKey {
    name: String,
    source: u64,
    shared_bytes: usize,
}

/// FNV-1a over bytes. In-crate so that the kernel id -- and with it the
/// on-disk PTX cache key -- depends on nothing whose hashing could change
/// under it.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl RawCudaKernel {
    /// A kernel from its entry point, its source and its block size.
    ///
    /// `name` must be the symbol of an `extern "C" __global__` function in
    /// `source`; `cube_dim` is the block size it is launched with (see the
    /// module doc for the signature convention).
    pub fn new(name: impl Into<String>, source: impl Into<String>, cube_dim: CubeDim) -> Self {
        let source: Arc<str> = source.into().into();
        let source_hash = fnv1a(source.as_bytes());
        Self {
            name: name.into(),
            source,
            source_hash,
            cube_dim,
            shared_bytes: 0,
        }
    }

    /// Ask for `bytes` of dynamic shared memory (`extern __shared__`).
    pub fn with_dynamic_shared(mut self, bytes: usize) -> Self {
        self.shared_bytes = bytes;
        self
    }

    /// The entry point.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The block size, as given to [`RawCudaKernel::new`].
    pub fn cube_dim(&self) -> CubeDim {
        self.cube_dim
    }

    /// Launch over `count` blocks with `args`.
    ///
    /// # Safety
    ///
    /// The kernel is hand-written and unchecked: the caller guarantees that
    /// every buffer read or written is bound and long enough for the indexing
    /// the source performs, that the scalars pushed match the `info_st` the
    /// source declares field for field, and that the kernel terminates.
    pub unsafe fn launch(
        &self,
        client: &ComputeClient<CudaRuntime>,
        count: CubeCount,
        args: RawArgs,
    ) {
        // SAFETY: forwarded to the caller, who holds the source.
        unsafe { client.launch_unchecked(Box::new(self.clone()), count, args.into_arguments()) }
    }
}

impl KernelMetadata for RawCudaKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<RawCudaKernel>()
            .info(RawKey {
                name: self.name.clone(),
                source: self.source_hash,
                shared_bytes: self.shared_bytes,
            })
            .cube_dim(self.cube_dim)
    }

    fn address_type(&self) -> StorageType {
        // Nothing in a raw kernel is addressed by cubecl; the id needs a value
        // and this is the default a generated kernel carries.
        AddressType::U32.unsigned_type()
    }
}

impl CubeTask<CudaCompiler> for RawCudaKernel {
    fn compile(
        &self,
        _compiler: &mut CudaCompiler,
        options: &<CudaCompiler as Compiler>::CompilationOptions,
        _mode: ExecutionMode,
        _address_type: StorageType,
    ) -> Result<CompiledKernel<CudaCompiler>, CompilationError> {
        if !options.supports_features.grid_constants {
            return Err(CompilationError::Validation {
                reason: format!(
                    "raw kernel `{}`: this device does not support grid constants, so the \
                     runtime would bind the scalars as a device buffer after the buffers \
                     instead of passing `info_st` by value -- a different signature from \
                     the one hand-written kernels are written against",
                    self.name
                ),
                backtrace: BackTrace::capture(),
            });
        }

        // The repr's ONE job here: carry the dynamic shared-memory size. The
        // runtime reads `shared_memory_size()` off it -- the maximum end of
        // any allocation in the body -- and nothing else, so an empty body
        // with a single byte array of the requested length is the whole
        // description. Everything else is the empty value.
        let shared_memories = match self.shared_bytes {
            0 => Vec::new(),
            bytes => vec![SharedMemory::Array {
                index: 0,
                item: Item::scalar(Elem::U8, true),
                length: bytes,
                align: 16,
                offset: 0,
            }],
        };
        let repr = CudaComputeKernel {
            tensor_maps: Vec::new(),
            buffers: Vec::new(),
            scalars: Vec::new(),
            info: Info::default(),
            meta_static_len: 0,
            body: Body {
                instructions: Vec::new(),
                shared_memories,
                pipelines: Vec::new(),
                barriers: Vec::new(),
                const_arrays: Vec::new(),
                local_arrays: Vec::new(),
                info_by_ptr: false,
                has_dynamic_meta: false,
                address_type: Item::scalar(Elem::U32, true),
            },
            cube_dim: self.cube_dim,
            cluster_dim: None,
            extensions: Vec::new(),
            flags: Flags::default(),
            items: HashSet::new(),
            kernel_name: self.name.clone(),
        };

        Ok(CompiledKernel {
            entrypoint_name: self.name.clone(),
            debug_name: Some("mary::models::inkling::rawcuda::RawCudaKernel"),
            source: self.source.to_string(),
            repr: Some(repr),
            cube_dim: self.cube_dim,
            debug_info: None,
        })
    }
}

/// The arguments of one raw launch: buffers in binding order, then the
/// scalars packed at C struct layout into the by-value blob.
#[derive(Default, Debug)]
pub struct RawArgs {
    buffers: Vec<Binding>,
    /// The `info_st` bytes so far, at C layout: each scalar aligned to its own
    /// size. Turned into eight-byte words at the end.
    bytes: Vec<u8>,
}

impl RawArgs {
    /// No buffers, no scalars.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind the next buffer parameter to `handle`.
    pub fn buffer(mut self, handle: &Handle) -> Self {
        self.buffers.push(handle.clone().binding());
        self
    }

    /// The next `info_st` field, a `float`.
    pub fn f32(self, v: f32) -> Self {
        self.scalar(&v.to_ne_bytes())
    }

    /// The next `info_st` field, an `unsigned int`.
    pub fn u32(self, v: u32) -> Self {
        self.scalar(&v.to_ne_bytes())
    }

    /// The next `info_st` field, an `int`.
    pub fn i32(self, v: i32) -> Self {
        self.scalar(&v.to_ne_bytes())
    }

    /// The next `info_st` field, an `unsigned long long`.
    pub fn u64(self, v: u64) -> Self {
        self.scalar(&v.to_ne_bytes())
    }

    /// The next `info_st` field, a `long long`.
    pub fn i64(self, v: i64) -> Self {
        self.scalar(&v.to_ne_bytes())
    }

    /// The next `info_st` field, a `double`.
    pub fn f64(self, v: f64) -> Self {
        self.scalar(&v.to_ne_bytes())
    }

    /// Append one primitive at its natural alignment -- its own size, which
    /// is the C rule for every type this offers.
    fn scalar(mut self, v: &[u8]) -> Self {
        let align = v.len();
        let padded = self.bytes.len().next_multiple_of(align);
        self.bytes.resize(padded, 0);
        self.bytes.extend_from_slice(v);
        self
    }

    /// The by-value blob as the eight-byte words the runtime passes, padded
    /// with zeros to a whole word.
    pub fn words(&self) -> Vec<u64> {
        let mut bytes = self.bytes.clone();
        bytes.resize(bytes.len().next_multiple_of(8), 0);
        bytes
            .chunks_exact(8)
            .map(|w| u64::from_ne_bytes(w.try_into().expect("eight bytes")))
            .collect()
    }

    /// What the runtime launches with. The dynamic-metadata offset is set to
    /// the blob's length: the whole blob is by-value scalars, none of it is
    /// shapes to upload. (`MetadataBindingInfo::custom` sets that offset to
    /// zero, which would make the server treat every word as dynamic metadata
    /// -- upload it and bind the buffer as one more pointer parameter.)
    fn into_arguments(self) -> KernelArguments {
        let words = self.words();
        let len = words.len();
        KernelArguments::new()
            .with_buffers(self.buffers)
            .with_info(MetadataBindingInfo::new(words, len))
    }
}

/// The smoke kernel: `out[i] = alpha * x[i]` over `n` floats, one thread per
/// element, 256 a block. Two buffers and a two-field `info_st` -- the whole
/// convention in nine lines of CUDA.
const SCALE_SRC: &str = r#"
struct info_st {
    float alpha;
    unsigned int n;
};

extern "C" __global__ void __launch_bounds__(256) scale_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    const __grid_constant__ info_st info
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < info.n) {
        out[i] = info.alpha * x[i];
    }
}
"#;

/// Threads per block of [`scale_kernel`].
const SCALE_UNITS: u32 = 256;

/// [`SCALE_SRC`] as a launchable kernel.
pub fn scale_kernel() -> RawCudaKernel {
    RawCudaKernel::new("scale_f32", SCALE_SRC, CubeDim::new_1d(SCALE_UNITS))
}

/// `alpha * x` over `n` f32 elements of `x`, into a fresh buffer, through
/// [`scale_kernel`].
///
/// # Safety
///
/// `x` must hold at least `n` f32 values.
pub unsafe fn scale_f32(
    client: &ComputeClient<CudaRuntime>,
    x: &Handle,
    n: usize,
    alpha: f32,
) -> Handle {
    let out = client.empty(n * core::mem::size_of::<f32>());
    let blocks = n.div_ceil(SCALE_UNITS as usize) as u32;
    // SAFETY: `x` holds `n` floats by the caller's contract, `out` was just
    // allocated to hold `n`, and the kernel guards `i < n`.
    unsafe {
        scale_kernel().launch(
            client,
            CubeCount::new_1d(blocks),
            RawArgs::new()
                .buffer(x)
                .buffer(&out)
                .f32(alpha)
                .u32(n as u32),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packer against the C layout rule, on the host -- this one runs
    /// anywhere.
    ///
    /// `struct { float a; unsigned int b; unsigned long long c; float d; }` is
    /// 24 bytes at offsets 0, 4, 8, 16; `struct { unsigned int n; unsigned
    /// long long p; }` is 16 with four bytes of padding after `n`. Those are
    /// the two cases a packer gets wrong: a wide field after narrow ones, and
    /// the trailing pad.
    #[test]
    fn scalars_pack_at_c_struct_layout() {
        let words = RawArgs::new()
            .f32(1.5)
            .u32(7)
            .u64(0x0102_0304_0506_0708)
            .f32(-2.0)
            .words();
        assert_eq!(words.len(), 3, "24 bytes is three words");
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(&bytes[0..4], &1.5f32.to_ne_bytes());
        assert_eq!(&bytes[4..8], &7u32.to_ne_bytes());
        assert_eq!(&bytes[8..16], &0x0102_0304_0506_0708u64.to_ne_bytes());
        assert_eq!(&bytes[16..20], &(-2.0f32).to_ne_bytes());
        assert_eq!(&bytes[20..24], &[0u8; 4], "the trailing pad is zero");

        let words = RawArgs::new().u32(1).u64(2).words();
        assert_eq!(words.len(), 2);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(&bytes[0..4], &1u32.to_ne_bytes());
        assert_eq!(&bytes[4..8], &[0u8; 4], "u64 after u32 is padded to 8");
        assert_eq!(&bytes[8..16], &2u64.to_ne_bytes());

        assert!(RawArgs::new().words().is_empty(), "no scalars, no words");
    }

    /// The id follows the text and the shape, and nothing else.
    #[test]
    fn kernel_id_tracks_source_cube_dim_and_shared() {
        let a = RawCudaKernel::new("k", "// a", CubeDim::new_1d(64));
        let same = RawCudaKernel::new("k", "// a", CubeDim::new_1d(64));
        let text = RawCudaKernel::new("k", "// b", CubeDim::new_1d(64));
        let dim = RawCudaKernel::new("k", "// a", CubeDim::new_1d(128));
        let smem = RawCudaKernel::new("k", "// a", CubeDim::new_1d(64)).with_dynamic_shared(4096);
        assert_eq!(a.id(), same.id());
        assert_ne!(a.id(), text.id());
        assert_ne!(a.id(), dim.id());
        assert_ne!(a.id(), smem.id());
        assert_eq!(a.id().stable_hash(), same.id().stable_hash());
        assert_ne!(a.id().stable_hash(), text.id().stable_hash());
    }

    /// Deterministic values in `[-1, 1)`, the same bytes on every machine.
    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The hand-written kernel against the eager path -- Burn's `mul_scalar`
    /// on the same CUDA client -- on `n = 1000` floats, a length that is not
    /// a multiple of the block so the `i < n` guard is exercised. Both sides
    /// perform one IEEE f32 multiply, so the results are BIT-IDENTICAL, and
    /// the assertion is exact.
    ///
    /// Launched TWICE with two alphas: the second launch has the same kernel
    /// id as the first, so it hits the loaded module and moves only the
    /// by-value blob -- the property the CUDA-graph patch path relies on.
    ///
    /// NOT EXECUTED on the machine this was written on (no CUDA device; `Bk`
    /// is `burn::backend::Cuda`). On a Spark:
    /// `cargo test --release --features inkling-cuda --lib models::inkling::rawcuda -- --nocapture`
    #[test]
    #[cfg(feature = "inkling-cuda")]
    fn scale_matches_the_eager_path() {
        use crate::models::inkling::seam::{Bk, client_of, handle_of, tensor_of};
        use burn::tensor::{Tensor, TensorData};

        let dev = Default::default();
        let n = 1000usize;
        let xv = lcg(n, 3);
        let x: Tensor<Bk, 1> = Tensor::from_data(TensorData::new(xv, [n]), &dev);
        let client = client_of(&x);
        let xh = handle_of(x.clone());

        for alpha in [0.7f32, -3.25] {
            let want = x
                .clone()
                .mul_scalar(alpha)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            // SAFETY: `xh` holds `n` floats.
            let out = unsafe { scale_f32(&client, &xh, n, alpha) };
            let got = tensor_of(client.clone(), dev.clone(), out, 1, n)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "alpha {alpha}, element {i}: raw {g} vs burn {w}"
                );
            }
        }
    }
}
