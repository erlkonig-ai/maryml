//! Flint's shared dense/MoE block math, bounded forward diagnostics and raw cached
//! generation. Query chunks<=3; explicitly admitted context capacity; streamed
//! weights and CPU residual handoff, not a serving or performance claim.

pub mod math;
pub mod weights;
pub mod native_setup;
pub mod residual;
pub mod config;
pub mod pile;
pub mod forward;
pub mod cache;
#[cfg(feature = "tokenizer")]
pub mod session;

// Direct raw-text codec, independent of persisted tokenizer graphs or chat framing.
#[cfg(feature = "tokenizer")]
pub mod tokenizer;

#[cfg(feature = "flint-cuda")]
pub mod gpu;
