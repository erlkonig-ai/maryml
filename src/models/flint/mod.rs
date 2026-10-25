//! Flint's shared dense/MoE block math and bounded 49-block F32 diagnostic.
//! Standalone fixtures consume external residuals; forward consumes fixed token
//! IDs and empty history. Neither is a serving/KV or quantized-runtime claim.

pub mod math;
pub mod weights;
pub mod config;
pub mod forward;

#[cfg(feature = "flint-cuda")]
pub mod gpu;
