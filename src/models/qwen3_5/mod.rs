//! Dense Qwen3.5 structural contracts and resident GPU primitives.
//!
//! Configuration/layout do not load weights or implement a runtime. Their
//! reference is Transformers 5.2.0 `models/qwen3_5`, checked against WeMM
//! 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482 and Ovis
//! 724f4a25d5ede0744eda3a11d2a1ec806a8f5cb0 checkpoint headers.
//! The opt-in `deltanet` module implements the resident post-convolution scan;
//! `gdn_ops` supplies the causal convolution and ordinary gated output norm.
//! The CUDA prepared-token path now assembles all 32 shared decoder blocks
//! and the output boundary. The single-image prepared composition connects
//! the vision tower through GPU scatter and explicit MRoPE positions. Raw
//! image processing is available through the separately gated, bounded
//! `wemm-input` codec/CUDA preparation API; video and batching remain open.
//! The text leg alone (`text::TextWemm`: gather, decoder, boundary) also
//! builds on CubeCL's wgpu runtime under `qwen3_5-wgpu`, BF16 only; see
//! [`Rt`] and [`Binder`].

pub mod config;
pub mod layout;
pub mod multimodal_layout;
#[cfg(any(feature = "wemm-input", feature = "qwen3_5-wgpu"))]
pub mod input_codec;
#[cfg(feature = "wemm-input")]
pub mod image_prepare;
#[cfg(feature = "wemm-input")]
pub mod native;
#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
pub mod position_table;
#[cfg(feature = "qwen3_5-cuda")]
pub mod multimodal;

pub mod vision_geometry;
#[cfg(feature = "qwen3_5-cuda")]
pub mod vision_frontend;
#[cfg(feature = "qwen3_5-cuda")]
pub mod vision_block;
#[cfg(feature = "qwen3_5-cuda")]
pub mod vision_merger;
#[cfg(feature = "qwen3_5-cuda")]
pub mod vision_tower;

#[cfg(feature = "qwen3_5-gpu")]
pub mod deltanet;

#[cfg(feature = "qwen3_5-gpu")]
pub mod gdn_ops;

#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
pub mod gdn_mixer;

#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
pub mod full_attention;

#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
mod decoder_ops;

#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
pub mod gdn_decoder;

#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
pub mod embedding_boundary;

#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
pub mod decoder_stack;

#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
pub mod prepared;

#[cfg(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
pub mod roles;

#[cfg(feature = "qwen3_5-real-layer-trace")]
pub mod outer_trace;
#[cfg(all(any(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"), not(feature = "qwen3_5-real-layer-trace")))]
mod outer_trace;

#[cfg(any(feature = "wemm-input", feature = "qwen3_5-wgpu"))]
pub mod text;

#[cfg(all(feature = "qwen3_5-wgpu", not(feature = "qwen3_5-cuda")))]
mod tiled_projection;

#[cfg(all(feature = "qwen3_5-cuda", feature = "qwen3_5-wgpu"))]
compile_error!("qwen3_5-cuda and qwen3_5-wgpu select the one runtime the Qwen3.5 decoder is built for; enable one");

/// The CubeCL runtime the text decoder (gather, the 32 blocks and the output
/// boundary) is built for. CUDA unless `qwen3_5-wgpu` chooses wgpu, whose
/// compiler is SPIR-V on Vulkan (`wemm-query-vulkan`) and MSL on Metal
/// (`wemm-query-metal`). The same kernels compile for both.
#[cfg(feature = "qwen3_5-cuda")]
pub type Rt = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "qwen3_5-wgpu", not(feature = "qwen3_5-cuda")))]
pub type Rt = cubecl::wgpu::WgpuRuntime;

/// How the decoder's BF16 leaves reach the device: CUDA aliases the pile's
/// mapped pages (or copies them, by choice); wgpu always copies them into
/// device buffers (`nn::wgpu_bf16_upload`).
#[cfg(feature = "qwen3_5-cuda")]
pub type Binder = crate::nn::cuda_bf16_alias::CudaBf16Aliases;
#[cfg(all(feature = "qwen3_5-wgpu", not(feature = "qwen3_5-cuda")))]
pub type Binder = crate::nn::wgpu_bf16_upload::WgpuBf16Uploads;
