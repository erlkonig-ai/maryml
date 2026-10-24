//! Flint's first sparse layer as bounded CPU F32/math. This is an import/parity
//! seam, not a model loader, serving implementation, or quantized kernel claim.
//! The independently supplied residual/position bytes are the entire input.

pub mod math;
pub mod weights;
