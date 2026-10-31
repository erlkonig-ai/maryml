//! Projected-input Mamba2 component; outer norm, residual and projections are
//! owned by the future model assembly. The scalar oracle is diagnostic only.
pub mod contract;
#[cfg(test)]
mod reference;
#[cfg(feature = "nemotron-cuda")]
pub mod gpu;
#[cfg(all(test, feature = "nemotron-cuda"))]
mod tests;
