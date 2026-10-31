//! Native Nemotron-H components and typed ingress. No assembled model or CPU fallback.
pub mod chat;
pub mod ingress;
pub mod mamba2;
pub mod pile;
#[cfg(feature = "nemotron-timing")]
mod host_timing;
#[cfg(feature = "nemotron-cuda")]
pub mod fp8;
#[cfg(feature = "nemotron-cuda")]
pub mod attention;
#[cfg(feature = "nemotron-cuda")]
pub mod moe;
#[cfg(feature = "nemotron-cuda")]
pub mod session;
#[cfg(all(feature = "nemotron-cuda", feature = "tokenizer"))]
pub mod generation;
