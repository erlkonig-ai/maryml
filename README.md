# mary

The Cargo package is named `maryml`; its Rust library and optional CLI are
still named `mary`. Consumers can retain the familiar dependency alias:

```toml
mary = { package = "maryml", version = "0.1.0" }
```

**A model is data — so store it like data.**

Model weights ship as opaque, multi-gigabyte blobs: a `safetensors` here, an
ONNX graph there, a GGUF somewhere else. You can't query them, you can't tell
when two models share a tensor, and lifting one model's component into another
means wrangling file formats by hand.

`mary` ports models to [Burn](https://burn.dev) and stores them as
**content-addressed graphs** in a database ([TribleSpace](https://github.com/triblespace))
— every tensor a leaf, every module an entity, every wiring an edge. Store your
whole zoo as recombinable parts in one place: identical weights **deduplicate**
across models automatically, loading is **config-free**, and **franken-stitching**
a component out of one model and into another is just rewiring edges — no format
conversion, no glue scripts, no safetensors at runtime.

Named for Mary Shelley, author of *Frankenstein*: mary stitches a living model
from parts.

## The format

A model is a graph of three primitives:

| primitive | is | 
|---|---|
| **tensor** | a self-describing `Tensor<Elem, RANK>` blob with dimensions in its header — identical tensors dedup; historical two-blob leaves remain readable |
| **module** | an entity whose parameters are tensor leaves reached by role-edges; `weight` and `bias` are the universal ones. A Linear is `{weight, bias?}`; a LayerNorm, Conv, Embedding are the same shape of thing — bias is just a parameter a module may or may not have |
| **composition** | role-edges (`q_proj`, `ff_in`, …) from a module to its children, ordered where it matters by an `index` |

Loading is config-free — a tensor carries its own shape. The **franken-stitch** is
rewiring role-edges between two models' module subgraphs.

### Model roots and lineage

Every family shares `format::attrs::{member, parent, model_root,
model_collection}`. A model reference names a collection by its descriptor
handle and a root by its opaque entity ID. The members' internal tensor or
adapter layout remains family-specific. New roots derive their identity from
the set of `member` links alone; names, source labels, weight-format tags, and
ancestry are annotations. Previously stored roots keep their original IDs, and
readers never reconstruct a root's hash to find or validate it.

`parent` means the actual **training or derivation base**, not the preceding
checkpoint save. Successive learned snapshots from the same base are siblings.
Imports with no known base omit the edge; compositions can name several bases.
Nomic packing retains the roots selected for its input weights, as does the
shared f16 derivation path. Gemma LoRA uses the same root/member/parent
vocabulary for its adapter graph.

Legacy partial imports are assembled with an explicit additive write:

```sh
# members.txt contains the exact member entity IDs, separated by whitespace.
mary root --pile models.pile --key model.key --members-file members.txt \
    --name assembled-model
# Optional: --collection COLLECTION_HANDLE, --parent BASE_ROOT,
#           --source SOURCE_LABEL, --quantization FORMAT_LABEL
```

`--member ENTITY_ID` can be repeated instead of (or alongside) the file. This
command publishes only a root over the supplied members and its annotations;
it does not discover all leaves, rewrite old entities, or claim that the
selected members form a complete runnable model. Choosing that complete set
is the explicit authoring operation. A runtime names the resulting stored
root, never a synthesized union of whatever happened to be in a pile.

## Layout

```
mary::format   the model-in-TribleSpace storage format (the substrate)
mary::nn       shared Burn toolkit: backend, weight loader, npy, norms
mary::models   the ports
   ::gemma       Gemma 4 — text generation + audio & vision understanding
   ::f5          F5-TTS — flow-matching expressive text-to-speech
   ::qwen3tts    Qwen3-TTS-12Hz — streaming, discrete-codec text-to-speech
   ::voxtral     Voxtral-Mini-4B-Realtime — delay-conditioned streaming STT
   ::personaplex PersonaPlex-7B — full-duplex speech LM (+ Mimi codec)
   ::qwen2_5_vl  Qwen2.5-VL / BiQwen — multimodal document embedder
   ::smolvla     SmolVLA — vision-language-action
   ::flux        Flux.2 — image generation (Klein + Dev)
mary::stitch   the franken-stitch (graph surgery across models)   [planned]
```

## Where the model files live

mary does not know, and does not guess. It is a library and a set of probes,
not an installation, so it holds no opinion about your disk layout — a baked-in
default path is wrong on every machine but the one it was written on, and it
fails in the worst way: the probe looks in the guessed place, misses, and
reports "not found" as though you did not have the model.

So every probe takes its path one of two ways:

```sh
# 1. explicitly, on the command line (always wins)
cargo run --release --features k3 --bin k3_layout_gate -- /data/kimi-k3

# 2. by name, out of $MARY_MODELS
export MARY_MODELS=/data/models      # → /data/models/kimi-k3
cargo run --release --features k3 --bin k3_layout_gate
```

With neither, the probe says so and exits — it never searches. Per-model
overrides (`GEMMA_PILE`, `QWEN3TTS_PILE`, `K3_MODEL_DIR`, …) still take
precedence over `$MARY_MODELS`; they are explicit, which is the point. Tests
SKIP rather than fail when a model is simply not on the machine.

## Models

### `gemma` — Gemma 4 (text LLM, in-substrate)

Gemma 4 runs as a **first-class in-process text LLM** — no ollama, no HTTP, no
OpenAI-compat shim. The model executes inside the substrate like everything
else: weights load ONLY from content-addressed piles (safetensors exists just
inside the `import`-gated persist importers), and generation
is a direct Rust call. The decoder + KV-cache + dual sliding/global RoPE + logit
softcapping + tied/untied LM head are all native Burn; pure text-gen is a strict
subset of the audio-understanding path (`gemma_hear`).

```sh
# text generation, all in Rust
cargo run --release --features gemma --bin gemma_gen -- \
    --prompt "In one sentence, what is a knowledge graph?"
# any Gemma 4 variant — short aliases (e2b|e4b|12b|26b|31b) or a full HF id
cargo run --release --features gemma --bin gemma_gen -- \
    --model 12b --prompt "..."
# half-precision (f16) weights — the dense 31B fits a 128GB M4 Max
cargo run --release --features "gemma f16gen" --bin gemma_gen -- \
    --model 31b --prompt "..."
```

The whole family runs through one decoder: E2B/E4B (PLE + KV-sharing),
26B-A4B (MoE), and the dense 12B and 31B (K=V global attention) are all
config-level variations. The dense 12B is the text path of the encoder-free
"unified" multimodal release — its text decoder is a 48-layer sibling of the
31B and needs no code of its own. f16 weights halve resident memory
(31B: ~124GB f32 → ~62GB) **and** run faster (memory-bandwidth bound on
M-series), with output identical to f32 on the small E2B. The same module's
`gemma_hear` bin does audio understanding, and `gemma_train_lora`
finetunes LoRA adapters on any variant (E4B and 12B fit f32 training on 128GB).

### `f5` — F5-TTS (expressive text-to-speech)

A from-scratch Burn port of [F5-TTS](https://github.com/SWivid/F5-TTS)
(flow-matching TTS) + the [Vocos](https://github.com/gemelo-ai/vocos) vocoder,
**validated layer-by-layer against the reference PyTorch** — every component at
cosine ≈ 1.0:

| component | rel-err vs reference | cosine |
|---|---|---|
| DiT velocity field | 5.2e-4 | 1.000000 |
| Vocos vocoder (waveform) | 3.6e-6 | 1.000000 |
| CFM sampler | 1.9e-3 | 1.000000 |
| end-to-end generated mel | 1.0e-3 | 0.999999 |
| mel extraction (STFT) | 1.4e-6 | 1.000000 |

The ISTFT and STFT are done as DFT-by-matmul + `conv1d`/`conv_transpose1d`
(Burn has no FFT), and match `torch.istft`/`torchaudio.MelSpectrogram` exactly.

```rust
// speak `gen_text` in the voice of a 24 kHz mono reference clip — no Python.
// Weights (F5 + Vocos) come entirely from the durable voice pile:
mary::say::synthesize_to_wav(f5_pile, ref_wav, ref_text, gen_text, out_wav);
```

Voice is zero-shot: the speaker identity, accent and emotional register all ride
in from the reference clip, not the weights.

Weights are not bundled. Fetch F5-TTS (`SWivid/F5-TTS`) and Vocos
(`charactr/vocos-mel-24khz`) from Hugging Face once, then persist them into the
pile with `f5_persist` (`--features import`); nothing at runtime reads
safetensors.

### `qwen3tts` — Qwen3-TTS-12Hz (streaming TTS)

A Burn port of Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0): a discrete multi-codebook
LM that streams audio at 12.5 Hz — a talker + code predictor + speaker encoder
feeding a 12 Hz codec decoder, zero-shot cloned from a reference clip. The
production seam is `mary::speak` (feature `speak`), loading weights from a
durable pile. Optional `megakernel` (fused CubeCL decode kernels — the lever is
host submission count, not FLOPs) and `q4` (4-bit grouped weights, the
bandwidth lever) accelerate the realtime path.

### `voxtral` — Voxtral-Mini-4B-Realtime (streaming STT)

A Burn port of Voxtral-Mini-4B-Realtime (Apache-2.0): delay-conditioned
streaming speech-to-text, one autoregressive step per 80 ms frame with the text
held 80 ms–2.4 s behind the audio via ada-RMS-norm conditioning. Feature
`voxtral`. Runtime loading selects one exact-f32/full-f16 cohort from a frozen
native model-collection snapshot; missing or ambiguous roots fail closed.

### `personaplex` — PersonaPlex-7B (full-duplex speech LM)

A Burn port of the PersonaPlex-7B full-duplex model — a temporal transformer +
depth transformer over the Mimi neural audio codec, both hearing and speaking on
one 12.5 Hz stream. Ported gate-first and verified against the reference at each
stage; feature `personaplex` (parity lane runs CPU-f32 under `burn-ndarray`).

### `qwen2_5_vl` — BiQwen multimodal embedder

The Qwen2.5-VL / BiQwen2.5 building blocks behind `nomic-embed-multimodal-7b`, a
dense retriever that embeds text, images, and text-in-images (visual documents)
into one space for semantic search. Feature `embed`.

### `smolvla` — SmolVLA (vision-language-action)

A Burn port of `lerobot/smolvla_base`: a SigLIP vision encoder + language model
+ flow-matching action expert that denoises a chunk of future actions from an
image, instruction, and robot state. Feature `smolvla`.

### `flux` — Flux.2 (image generation)

A Burn port of the Flux.2 diffusion transformer (folded in from the `avatar`
crate), supporting **both** released variants, auto-detected from the model
directory:

| variant | text encoder | conditioning |
|---|---|---|
| **Klein** (4B) | Qwen3 | step-distilled, no guidance |
| **Dev** | Mistral3 | classifier-free guidance |

The DiT, VAE, scheduler, and text encoders are all native Burn. Feature `flux`.

## License

MIT OR Apache-2.0.
