# Changelog

## Unreleased

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
