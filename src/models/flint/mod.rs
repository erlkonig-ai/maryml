//! Flint's isolated dense block-0 and sparse block-2/block-5 fixtures as bounded F32 math. This is an import/parity
//! seam, not a model loader, serving implementation, or quantized kernel claim.
//! The independently supplied residual/position bytes are the entire input.

pub mod math;
pub mod weights;
pub mod config;

#[cfg(feature = "flint-cuda")]
pub mod gpu;
