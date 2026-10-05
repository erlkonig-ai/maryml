//! mary — our Burn model zoo, held in TribleSpace.
//!
//! Named for Mary — Mary Wollstonecraft Shelley, who wrote the creature into
//! being, and her mother Mary Wollstonecraft. The crate brings dead weights to
//! life as composable model-graphs.
//!
//!   - [`format`] — the model-in-TribleSpace storage format (self-describing
//!     tensor leaves, modules as entities, role-edges). The substrate.
//!   - [`nn`] — shared Burn toolkit (backend, weight loader, npy, norms).
//!   - `models::*` — the ports, named by lineage: `f5` (voice), `flux` (face),
//!     `gemma` (senses), `smolvla` (body/action).
//!   - `stitch` — the franken-stitch: graph surgery across models' role-edges.

/// A generic, modality-blind ML dataset / sample-management schema on
/// TribleSpace: Dataset snapshots, multimodal Samples, TrainingRuns, and
/// Preferences. Domain-free — it knows nothing about any one dataset's labels.
pub mod dataset;
/// CLIP ViT-B/32 multi-modal embedder behind `LocalEmbedder` — image+text into
/// one L2-normalized contrastive space (cosine == dot). Gated behind `embed`.
#[cfg(feature = "embed")]
pub mod embed;
pub mod f16enc;
pub mod format;
/// Non-safetensors weight-file importers (GGUF, pickled PyTorch `state_dict`).
/// Each decodes to the same `(name, f32-data, shape)` tuples [`ingest`] consumes,
/// so every format lands in one content-addressed graph. Import-only.
#[cfg(feature = "import")]
pub mod formats;
/// In-process Voxtral-Mini-4B-Realtime speech-to-text (the production ears):
/// weights and tokenizer from one native pile, 16 kHz samples in, text out.
#[cfg(feature = "voxtral")]
pub mod hear;
pub mod ingest;
/// A checkpoint's JSON sidecars — `config.json`, `hf_quant_config.json`, the
/// chat template — as FACTS rather than stored files, so a pile stops needing
/// the checkpoint directory beside it. One entity per JSON scalar, nine
/// attributes for every document; see the module docs for why that beats one
/// minted attribute per config field.
pub mod jsonfacts;
pub mod leaf;
pub mod calibrate;
pub mod nvfp4;
#[cfg(feature = "local-model")]
pub mod local;
/// Compatibility projection for model graphs written before TribleSpace's
/// encoding-aware anchored-attribute epoch. This is a migration primitive;
/// runtime readers continue to use the canonical schemas in [`format`] and
/// [`tokenizer`] directly.
pub mod model_collection;
pub mod models;
pub mod nn;
/// Resolving model files without guessing where they live: an explicit path, or
/// `$MARY_MODELS`, or a loud error naming both. mary ships no default layout.
pub mod paths;
/// Persist/load model weights to a real on-disk pile (the shell-is-physics
/// endpoint). Loading from a pile is THE runtime weight path — every model
/// family goes through here — so the module is unconditional. The
/// safetensors → pile persist direction inside it is `import`-gated.
pub mod persist;
/// In-process F5-TTS voice synthesis — zero-shot, cloning the speaker from a
/// reference clip. A library seam shared by the `say` example — no separately-
/// built production binary that can drift stale against the pile format. The
/// production speak path is [`speak`] (Qwen3-TTS).
pub mod say;
/// Deterministic selection and materialization from an already-open model
/// graph. Storage adapters supply the facts and blob reader; this module owns
/// only the graph semantics.
pub mod selection;
/// Seeded categorical sampling shared by autoregressive model runtimes.
pub mod sampling;
/// In-process Qwen3-TTS voice synthesis (the production speak path):
/// clone the reference kit, speak arbitrary text, weights loaded from a durable
/// standalone pile — no safetensors, no separate binary in the path.
#[cfg(feature = "speak")]
pub mod speak;
/// tiktoken-style byte-level BPE (Kimi-K3): the rank table IS the merge order,
/// so there is no merges list — the engine half of what [`tokenizer`]'s
/// `ty::TIKTOKEN` stores as facts. Gated id-for-id against the shipped
/// `tokenization_kimi.py` by the `k3_tokenizer_gate` binary.
#[cfg(feature = "k3tok")]
pub mod tiktoken;
/// Tokenizers as content-addressed graphs (the companion to [`format`]): vocab,
/// merges, and added-tokens as tribles so the tokenizer travels with the pile
/// instead of as a HuggingFace-cache `tokenizer.json` side-file.
pub mod tokenizer;

pub use format::*;
