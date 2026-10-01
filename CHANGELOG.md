# Changelog

## Unreleased

- Add an unexecuted prepared-image CUDA oracle/native gate over the existing
  imported model pile. The pinned actual HF wrapper owns vision/scatter/MRoPE;
  both runtimes generate identical prepared BF16 patch inputs on GPU. Reports
  retain input bits, merged vision/scatter/decoder/final outputs, rotary-buffer
  dtypes and unchanged coordinate budgets. Existing text-gate code is unchanged;
  this one synthetic still-image case is not raw-image or model admission.

- Add a source-only prepared single-image/text WeMM composition over one
  frozen model root and caller-owned BF16 alias binder: checked integer layout,
  GPU placeholder scatter, explicit three-axis GPU positions, and fresh shared
  decoder prefill. The existing affine text/decode execution is unchanged.
  This is B1/unpadded, with the processor's all-ones mask policy; no raw image
  processor, video, continuation, executed gate or numerical admission claim.

- Add the native BF16 WeMM prepared-token decoder leg: typed opaque-root role
  queries, zero-copy pile aliases, GPU embedding gather, all 32 Qwen3.5 layers,
  and the shared embedding boundary. A separate prepared-patch vision tower
  composes the 27 vision blocks. No image scatter, arbitrary decoder MRoPE,
  batching, Files deployment, or numerical admission is implied. The source-bound
  HF CUDA gate keeps layer-level budgets and records every selected weight.
  Native BF16 safetensors import preserves bytes without an F16 detour; newer
  acquiring model-collection and persistence boundaries remain intact.

- Build reservation heartbeats work with macOS and GNU sed and fail if the
  heartbeat cannot be written, rather than announcing a successful refresh.

- Operational model snapshots retain one selected collection support while
  exact descriptor, label, policy-definition and tensor reads may acquire
  bytes through the caller's storage. Unavailable selection inputs fail
  instead of erasing a candidate or choosing a fallback model. Passive
  resident discovery/selection APIs retain their open-world skip behavior.
  Qwen3-TTS, FLUX and Nomic MM7B operational constructors use the strict
  selectors; native tensor layouts, GPU kernels and mmap alias ownership are
  unchanged.

- Pin the build root to AnyBytes `066c32a7`, matching TribleSpace, Faculties,
  and Drive. Temporary section freezes no longer perform durability flushes;
  explicit `ByteArea::persist` owns that barrier. Model identities, tensor
  encodings, numerical kernels, and feature selections are unchanged.
