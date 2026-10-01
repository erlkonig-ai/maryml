//! Shared Burn toolkit reused across model ports: the concrete backend alias,
//! the safetensors `WeightLoader`, `.npy`/`.npz` I/O, and normalization
//! primitives.
//! the safetensors `WeightLoader`, `.npy` I/O, normalization primitives, and
//! the 4-bit weight codecs (`q4` for mary-quantized weights, `mxfp4` for the
//! microscaling format checkpoints ship in).
//! the safetensors `WeightLoader`, `.npy`/`.npz` I/O, and normalization primitives.
//! Model-specific layers live with their model under `mary::models`.

#[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
pub mod alias;
pub mod backend;
#[cfg(feature = "cuda-bf16-alias")]
pub mod cuda_bf16_alias;
pub mod mxfp4;
pub mod norm;
pub mod npy;
pub mod npz;
/// Two-stage residual NVFP4 arithmetic for exact cosine search.
///
/// This module is deliberately independent of TribleSpace storage. Search
/// collections arrange its rows into blobs; accelerator backends consume its
/// read-only plane views.
pub mod nvfp4_cosine;
// Share the existing raw CUDA launcher without enabling the Inkling model.
#[cfg(feature = "nvfp4-encode-cuda")]
#[path = "../models/inkling/rawcuda.rs"]
pub(crate) mod raw_cuda;
#[cfg(feature = "q4")]
pub mod q4;
pub mod weight_loader;
