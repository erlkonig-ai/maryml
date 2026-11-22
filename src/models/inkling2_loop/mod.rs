//! Inkling 2 Mini (Thinking Machines, `inkling2_loop`): a 24-layer sparse-MoE
//! decoder whose middle twelve layers run three times, so 48 layer visits per
//! token over 24 layers of weights, with NVFP4 and MXFP8 experts, a ViT, and
//! a separate drafter.
//!
//! This is a lineage of its own, not a generation of `models::inkling`: the
//! original Inkling keeps its module, its layout and its loader unchanged, and
//! both share only the NVFP4 weight codec. What is here so far:
//!
//! * [`config`]: which Inkling a `config.json` describes ([`config::family_of`],
//!   also consulted by the original Inkling's loader so that a Mini config is
//!   never parsed as one), and Mini's configuration, validated;
//! * [`schedule`]: the physical/logical layer schedule and the per-visit state
//!   identity [`schedule::CacheId`] = (physical layer, pass);
//! * [`layout`]: checkpoint names to slots, with shapes from the config and
//!   per-tensor precision from `hf_quant_config.json`;
//! * [`checkpoint`]: a header-only inspection of a checkpoint directory
//!   against that layout, and the explicit list of what Mary cannot run yet;
//! * [`experts`]: the NVFP4 (amax / denominator) and MXFP8 weight decodes.
//!
//! There is no forward here. Anything that would run the model is refused by
//! [`checkpoint::require_native_forward`] with the operators still missing.
//!
//! Pinned sources: the published release `thinkingmachines/Inkling-2-Mini` at
//! b85c3cfa615fc413cc9af3379f491b963f3b586a (`fixtures/`), and transformers
//! 046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3 (`models/inkling2_loop`,
//! `conversion_mapping.py`, `core_model_loading.py`).

pub mod checkpoint;
pub mod config;
pub mod experts;
pub mod head;
pub mod layout;
pub mod pile;
pub mod release;
#[cfg(feature = "inkling2-import")]
pub mod import;
pub mod schedule;

pub use config::{Family, Inkling2LoopConfig, family_of, family_of_str};
