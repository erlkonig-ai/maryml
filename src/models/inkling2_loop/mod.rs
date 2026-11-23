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
//!   against that layout, and the explicit list of what Mary does not run
//!   natively (vision, drafter, the declared activation quantisation);
//! * [`experts`]: the NVFP4 (amax / denominator) and MXFP8 weight decodes;
//! * `sliding` (with `inkling-cuda`): one visit of a sliding layer outside
//!   the looped core, on the device, with its K/V and conv state. Held to
//!   transformers' own layer 0 by the `inkling2_layer0` binary.
//! * [`messages`]: the cross-pass message read and the four message gates of
//!   the core layers, on the CPU, as upstream computes them at bf16 and f32.
//! * `full` (with `inkling-cuda`): one visit of a full-attention layer
//!   outside the core (layer 5): three block-attention inputs, dynamic K/V
//!   convolutions, log scaling. Held to transformers' own layer 5 by the
//!   `inkling2_global5` binary.
//! * `loop_core` (with `inkling-cuda`): the looped core, layers 6..17 over three
//!   passes, from the pre-loop hidden and block states to the post-loop ones,
//!   each visit with its own carried state. Held to transformers' own
//!   three-pass run by the `inkling2_core` binary.
//!
//! * `tail_check` (with `inkling-cuda`): the post-loop tail, layers 18..23
//!   (MXFP8 experts from 20) and the head, against transformers' own run on
//!   fixed inputs; the `inkling2_tail_check` binary on the release's weights.
//!
//! * `session` (with `inkling2-session`): the whole text forward as a model
//!   bound once from a pile root ([`session::MiniModel`]) and independent
//!   token sequences over it ([`session::MiniSession`]): token IDs in,
//!   softcapped logits out, every layer visit's state carried.
//!
//! What the release declares beyond the text forward (the vision tower, the
//! drafter, the declared expert activation quantisation) is not native:
//! [`checkpoint::native_gaps`] names it, and the original Inkling's loader
//! refuses a Mini config with that list and a pointer to the session
//! ([`checkpoint::refusal_for_config`]).
//!
//! Pinned sources: the published release `thinkingmachines/Inkling-2-Mini` at
//! b85c3cfa615fc413cc9af3379f491b963f3b586a (`fixtures/`), and transformers
//! 046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3 (`models/inkling2_loop`,
//! `conversion_mapping.py`, `core_model_loading.py`).

pub mod chat;
pub mod checkpoint;
pub mod config;
pub mod experts;
pub mod head;
pub mod layout;
pub mod messages;
pub mod pile;
pub mod release;
#[cfg(feature = "inkling2-import")]
pub mod import;
pub mod schedule;
#[cfg(feature = "inkling-cuda")]
pub mod attend;
#[cfg(feature = "inkling-cuda")]
pub mod loop_core;
#[cfg(feature = "inkling-cuda")]
pub mod full;
#[cfg(feature = "inkling-cuda")]
pub mod moe;
#[cfg(feature = "inkling-cuda")]
pub mod sliding;
#[cfg(feature = "inkling-cuda")]
pub mod tail_check;
#[cfg(feature = "inkling2-session")]
pub mod session;
#[cfg(feature = "inkling-cuda")]
pub mod prof;
#[cfg(feature = "inkling-cuda")]
pub mod rows;

pub use config::{Family, Inkling2LoopConfig, family_of, family_of_str};
