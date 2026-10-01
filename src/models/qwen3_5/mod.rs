//! Dense Qwen3.5 structural contracts and resident GPU primitives.
//!
//! Configuration/layout do not load weights or implement a runtime. Their
//! reference is Transformers 5.2.0 `models/qwen3_5`, checked against WeMM
//! 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482 and Ovis
//! 724f4a25d5ede0744eda3a11d2a1ec806a8f5cb0 checkpoint headers.
//! The opt-in `deltanet` module implements the resident post-convolution scan;
//! `gdn_ops` supplies the causal convolution and ordinary gated output norm.
//! The CUDA prepared-token path now assembles all 32 shared decoder blocks
//! and the output boundary. Vision has a separate prepared-patch tower; GPU
//! scatter/arbitrary multimodal positions and numerical admission remain open.

pub mod config;
pub mod layout;

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

#[cfg(feature = "qwen3_5-cuda")]
pub mod gdn_mixer;

#[cfg(feature = "qwen3_5-cuda")]
pub mod full_attention;

#[cfg(feature = "qwen3_5-cuda")]
mod decoder_ops;

#[cfg(feature = "qwen3_5-cuda")]
pub mod gdn_decoder;

#[cfg(feature = "qwen3_5-cuda")]
pub mod embedding_boundary;

#[cfg(feature = "qwen3_5-cuda")]
pub mod decoder_stack;

#[cfg(feature = "qwen3_5-cuda")]
pub mod prepared;

#[cfg(feature = "qwen3_5-cuda")]
pub mod roles;

#[cfg(feature = "qwen3_5-real-layer-trace")]
pub mod outer_trace;
#[cfg(all(feature = "qwen3_5-cuda", not(feature = "qwen3_5-real-layer-trace")))]
mod outer_trace;
