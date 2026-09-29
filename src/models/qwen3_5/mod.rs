//! Dense Qwen3.5 structural contracts and resident GPU primitives.
//!
//! Configuration/layout do not load weights or implement a runtime. Their
//! reference is Transformers 5.2.0 `models/qwen3_5`, checked against WeMM
//! 00c52839de57a6d4fd5b78cf5522ccf0ac8ea482 and Ovis
//! 724f4a25d5ede0744eda3a11d2a1ec806a8f5cb0 checkpoint headers.
//! The opt-in `deltanet` module implements the resident post-convolution scan;
//! this is not yet a complete decoder or embedding backbone.

pub mod config;
pub mod layout;

#[cfg(feature = "qwen3_5-gpu")]
pub mod deltanet;
