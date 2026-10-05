//! Step-Audio2 Mini's native pile import and exact text/speech token contract.
//!
//! This slice contains no decoder, reference encoder, flow model, or vocoder.
//! Checkpoint ingestion is import-only; runtime assets come from explicit IDs
//! in the caller's frozen model collection. No checkpoint directory is reopened.

pub mod codec;
pub mod config;
#[cfg(feature = "import")]
pub mod import;
pub mod load;

/// The selected upstream artifact. Provenance only, never an entity lookup key.
pub const SOURCE: &str = "stepfun-ai/Step-Audio-2-mini";
pub const REVISION: &str = "e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36";

#[cfg(all(test, feature = "import"))]
mod tests;
