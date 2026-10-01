# Changelog

## Unreleased

## 0.1.0 - 2026-10-01

- Publish under the package name `maryml`; the Rust library and the optional
  `mary` command retain their existing names. Path consumers keep their
  `mary` dependency alias by declaring `package = "maryml"`.
- Native WeMM input preparation and the composed BF16 CUDA model are exercised
  on a bounded 11-input behavioral fixture (nine scorable retrieval items).
  The opt-in fixed-MMA projection path preserves the fixture's behavior and
  same-compute-class byte identity across the tested order/process repeats.
  On that GB10 fixture its measured warm speedup is about 12–57× for text and
  6.5× for the image case. These are fixture measurements, not throughput
  guarantees or a claim of numerical identity with the reference. The same
  22 BF16 fixture outputs subsequently matched byte-for-byte on both GB10
  Sparks. The GPU NVFP4 encoder/scorer and opt-in Files text/image adapter
  passed their bounded native gates; a complete Metal model remains open.
  The component-level reference discrepancies recorded during development
  are not reclassified as passing by this behavioral result.

- Add an opt-in CUDA 4096-D BF16 query/reconstructed-cosine API with owned
  uploaded NVFP4 byte planes and canonical eight-lane FP64 scoring. Query
  normalization/rotation stays on the GPU; handle deduplication and maximum
  selection remain storage policy. This is not an upper-bound scanner or
  exact source reranker and does not change the persisted row recipe.

- Expose native WeMM's actual selected root, validated asset content handles
  and bound CUDA device for runtime identity checks; query device name and
  compute capability from the driver rather than host architecture. Add a
  read-only model-pile opener sharing the existing frozen collection reader.
  A read-only descriptor is not an external immutability guarantee: zero-copy
  callers retain the explicit immutable-prefix custody obligation. The Files
  integration gate exercised the loaded root/assets/device binding with 759
  roles and no rebinding, including refusal before a partial overlength leaf.

- Add a bounded direct BF16 `[1,4096]` CUDA encoder for the existing two-stage
  NVFP4 row recipe. Normalization, signed Hadamard transform, quantization and
  outward error certification stay on GPU with explicit arithmetic order;
  only completed encoded bytes/status return to the host. CPU arithmetic is
  used solely by the new byte/certificate oracle tests. No collection identity,
  Files mapping, query scorer or model-selection behavior changes.

- Expose a resident native CUDA WeMM facade over the existing shared model,
  pinned tokenizer/template and GPU image preparation. Text and single-image
  calls return the GPU BF16 `[1,4096]` embedding without host numeric work or
  readback. The caller keeps one bounded alias session; load failures report
  its registration counts and do not claim to unregister partial loads.
  Keep the decoder's fixed eight groups heap-backed to avoid large inline
  array copies overflowing ordinary debug/test-thread constructor stacks.
  Executed on one GB10: library check, three metadata tests and the fixed
  11-input text/image facade test pass on the default test stack; all 22
  forward/reverse outputs match the retained native BF16 bytes, with 759
  selected roles, no rebinding and the source reader dropped before forwards.
  Earlier test compile and inline-stack failures remain preserved. The 680 s
  debug test includes provenance hashing, construction and all forwards, not
  a production inference benchmark; whole-file model hashing took 9 s.

- Native BF16 WeMM constructors accept caller-owned exact-get readers while
  preserving selected model facts/slots and genuine immutable pile backing.
  Decoder client validation shares its already-bound final norm instead of
  reading/registering it twice. No numerical kernel, dtype, model selection,
  preparation or Files indexing behavior changes. Generic-reader and owner
  lifetime witnesses are added; execution is separate from this source slice.

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
