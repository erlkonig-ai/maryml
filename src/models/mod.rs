//! Model ports, indexed by what the model *is* (its lineage), not by the role
//! it plays in a deployment: `f5` (TTS), `gemma` (LLM + audio), `flux` (image
//! generation). Each is a graph of `mary::format` modules built with `mary::nn`.

pub mod f5;

pub mod qwen3_5;

// Qwen2.5-VL text backbone (BiQwen2_5 / nomic-embed-multimodal-7b). Reuses
// gemma's RoPE table, so it rides the `gemma` feature.
#[cfg(feature = "gemma")]
pub mod qwen2_5_vl;

#[cfg(feature = "flux")]
pub mod flux;

#[cfg(feature = "gemma")]
pub mod gemma;

#[cfg(feature = "smolvla")]
pub mod smolvla;

#[cfg(feature = "qwen3tts")]
pub mod qwen3tts;

// Mimi neural audio codec (PersonaPlex Phase 1). Reuses the qwen3tts CPU
// primitives (Accelerate sgemm + libm gelu), so it rides the same feature.
#[cfg(feature = "qwen3tts")]
pub mod personaplex;

#[cfg(feature = "voxtral")]
pub mod voxtral;

// Kimi-K3 (`kimi_linear`, 2.78 T MoE). Config + checkpoint-name->module-slot
// layout, the ported operators (SiTU, KDA, MLA, AttnRes, router, latent MoE)
// and the whole decoder layer that composes them.
#[cfg(feature = "k3")]
pub mod k3;

// Inkling (Thinking Machines, 975 B / 276 B MoE, natively multimodal). One
// feature, and it names CUDA: every lane below is a Blackwell tensor-core lane.
#[cfg(feature = "inkling-cuda")]
pub mod inkling;

// Inkling's speculation algebra — tree topology, top-b, the drafting schedule,
// the ancestor masks, the accept walk — is backend-free by construction, and
// the whole point of building it that way is that it can be TESTED where there
// is no GPU. `models::inkling` is gated on the CUDA lane, which no laptop can
// compile, so without that lane the module is re-formed around the one file
// that does not need it. Same path either way:
// `mary::models::inkling::spectree`.
#[cfg(not(feature = "inkling-cuda"))]
pub mod inkling {
    #[path = "cache_pile.rs"]
    pub mod cache_pile;
    #[path = "spectree.rs"]
    pub mod spectree;

    #[path = "target.rs"]
    pub(crate) mod target;

    // The RESIDENT MIND is backend-free for the same reason and to greater
    // effect. The model now runs in the same process as the loop — that was the
    // whole point of the one-binary collapse — so this is no longer "the client
    // half of a protocol". It is the half of the mind that is ABOUT TURNS
    // rather than about kernels: what a delta is, what a carry is, when a
    // response is complete, which bytes a decision spans. None of that needs a
    // GPU, all of it is where the expensive mistakes have historically been,
    // and keeping it compilable here is what keeps it tested on a laptop
    // against a scripted `Model`.
    #[path = "resident.rs"]
    pub mod resident;

    // The dMel front end is a tokenizer and runs where the microphone is.
    #[cfg(feature = "dmel")]
    #[path = "dmel.rs"]
    pub mod dmel;
    pub mod patches;

    // Native READY/TurnEnd evidence, in the same reconstructed module as
    // `resident` for the same reason: durable turn exhaust is a fact about a
    // turn, not about a kernel.
    #[path = "telemetry.rs"]
    pub mod telemetry;
}
