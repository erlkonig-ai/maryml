//! Global attention that never builds `[heads, n, n]` — online softmax, one pass.
//!
//! # The gap this fills
//!
//! Thirty-five of Inkling-Small's forty-two attention layers are local, and
//! [`super::banded`] fuses them. The other **seven are global**, and until this
//! file they went through burn's generic matmul with a MATERIALISED softmax
//! over the whole key axis: `q @ k^T` into a `[heads, rows, tokens]` block, a
//! scale/bias/mask epilogue over that block, `softmax(scores, 2)` over it
//! again, then `p @ v`. The block is bounded only by `INK_QBLOCK`, which
//! bounds `rows` and nothing else, so the key axis is the sequence and the
//! whole thing is linear in `tokens` per query row with a large constant.
//!
//! ## Why the seven matter more than the thirty-five
//!
//! Not at the context we benchmark. At ~3.7k attention is 7.6 ms of a 54 ms
//! decode step and fusing it buys almost nothing; this file is not for that
//! number and should not be tuned against it.
//!
//! It is for the two regimes where the global layers are the constraint:
//!
//! * **Decode at long context.** A global layer reads the whole cache:
//!   `8 kv_heads * 128 * 2 (K and V) * 2 B` is 4 KiB per token per global
//!   layer, so at 1M tokens the seven of them move **28 GB per decode step** in
//!   BF16 against 5.77 GB of weights — attention becomes ~5x the entire rest of
//!   the model. NVFP4 KV (unconditional now) takes that to ~8.1 GB. The
//!   crossover where global-layer KV traffic equals weight traffic is ~400k
//!   tokens. A fused pass does not reduce the KV read — nothing can, the keys
//!   must be looked at — but it removes everything ELSE the old lane moved: the
//!   `[heads, 1, slots]` score row written by the matmul, read and rewritten by
//!   the epilogue, read twice and written once by the softmax, and read again
//!   by `p @ v`. At 1M that row is 128 MB and it is touched six times.
//! * **Prefill.** The global layers cannot materialise `[n, n]` at long `n` at
//!   all, and the old lane's peak working set was never just the score block:
//!   it GQA-expands K and V to `[heads, tokens, head_dim]` first, which is two
//!   819 MB buffers at 100k tokens in BF16, linear in the sequence, with no
//!   knob on them. This kernel indexes `h / groups` instead, so the expansion
//!   does not happen, and the relative-bias table is the only thing left that
//!   scales with the query block.
//!
//! # The shape
//!
//! One cube per `(query tile, KV head, key split)`, 128 units — four planes of
//! thirty-two. A cube holds `ROWS` query rows, and those rows are a
//! `(head, query)` RECTANGLE: `groups` query heads (all the heads that share
//! this KV head) by `ROWS / groups` consecutive queries. That is the whole GQA
//! saving expressed as a tile shape — K and V are read once for all `groups`
//! heads that need them, rather than once per head off an expanded copy.
//!
//! Within a cube, plane `w` owns rows `[w * rpp, (w+1) * rpp)` and lane `l`
//! owns key `l` of the current key tile and output dimensions
//! `l, l+32, l+64, l+96`. So:
//!
//! * a row's thirty-two scores live in ONE plane, and the online softmax's row
//!   max and row sum are `plane_max` / `plane_sum` — no shared-memory tree, no
//!   `sync_cube` per reduction. (`plane_max` lowers to a shuffle loop on this
//!   backend, not to `redux.sync.max.f32`, which sm_121a does not have.)
//! * the output accumulator is `rpp * head_dim / 32` REGISTERS per lane — 32 at
//!   the prefill tile, 4 at the decode tile — and never touches memory until
//!   the cube ends.
//!
//! Nothing quadratic is written down. The largest thing this file allocates is
//! the partial-output buffer, which is `splits * nq * heads * head_dim`.
//!
//! # Online softmax is the one part that is not ordinary GEMM practice
//!
//! Tiling a product over `k` is what every GEMM does. What flash attention
//! genuinely invented is running the softmax's normalisation as a RUNNING
//! quantity: each cube carries `m` (the max seen so far) and `l` (the sum of
//! `exp(s - m)` so far) per row, and when a tile pushes the max up it rescales
//! the accumulator by `exp(m_old - m_new)` before adding the new tile's
//! contribution. The answer is the same as a two-pass softmax to the last bit
//! of the accumulation order, and the key axis is never materialised.
//!
//! `NEG_INF` here is a large FINITE constant, the way the rest of this module
//! tree spells it, and that forces every guard to be explicit: `-3.4e38 -
//! -3.4e38` is `0`, so `exp(s - m)` for a fully-masked tile would be `1` rather
//! than `0` if it were left to the arithmetic. Both the rescale and the
//! exponential are branch-guarded instead of relying on infinity's algebra.
//!
//! # Staging K, and not staging V
//!
//! The score phase has lane `l` walking all `head_dim` dimensions of key
//! `t + l`. In the `[tokens, kv_heads * head_dim]` layout both the projections
//! and the paged cache produce, that is thirty-two addresses `kv_heads *
//! head_dim` floats apart — thirty-two memory transactions per warp
//! instruction, which is the failure [`super::banded`] measured at 187 ms of a
//! 190 ms kernel and fixed by transposing K to `[kv_heads, head_dim, tokens]`.
//!
//! A transpose is not available here. The decode lane reads PAGES of a cache
//! that is stored token-major and quantised token-major, and transposing them
//! per layer per step would cost a full read and a full write of the very
//! bytes the fusion exists to stop moving. So this kernel STAGES instead: the
//! key tile is copied into shared memory by all 128 units in coalesced
//! `head_dim`-wide lines, and the score phase reads it from there, where the
//! access pattern costs nothing. That is the same class of fix a sibling
//! measured as ~2x on the 4-bit weight GEMMs (`fp4_linear_smem`), and it is why
//! a flash-shaped kernel wants no transposed input: it stages by construction.
//!
//! The shared tile is padded to `head_dim + 1` floats a row. Without the pad,
//! lane `l` reading `sk[l * 128 + d]` puts all thirty-two lanes on bank
//! `d % 32` — a thirty-two-way conflict on every single shared read of the
//! inner loop. With it, lane `l` lands on bank `(l + d) % 32`, which is
//! conflict-free.
//!
//! V is NOT staged, for [`super::banded`]'s reason: in the value phase lane `l`
//! owns output DIMENSIONS at a fixed key, so `[tokens, kv_heads * head_dim]`
//! already has consecutive lanes on consecutive addresses. The four planes read
//! the same V line, which L1 serves.
//!
//! # Splitting the key axis, and why decode needs it
//!
//! At decode `nq` is 1, so the grid without a split is `1 x kv_heads x 1` —
//! eight cubes, on a device with dozens of SMs, each walking a million keys
//! serially. That is not a kernel, it is a queue. So the key axis is SPLIT:
//! split `s` runs the same online softmax over its own slice and writes an
//! UNNORMALISED `(o, m, l)` triple, and [`flash_combine_kernel`] merges them
//! with the same rescale the tile loop uses — `exp(m_s - M)` weighting, one
//! division at the end. This is flash-decoding, and it is what turns eight
//! cubes into as many as the device will take.
//!
//! The combine also runs when there is one split, where it is a normalisation
//! pass. That costs one extra read and write of `nq * heads * head_dim` f32 —
//! nothing next to the key axis at the lengths this file is for, and one code
//! path instead of two.
//!
//! # What it measured
//!
//! **Framing rule for every number in this section.** Milliseconds for ONE
//! attention layer at the 42-layer release's global shape (hidden 4096, 32
//! heads over 8 KV heads, `head_dim` 128, `d_rel` 16, `rel_extent` 1024), on a
//! GB10, with the projections and the two short convolutions INCLUDED — these
//! time `attention_prefill` and `attention_step`, not the kernel alone. The
//! model has SEVEN global layers, so multiply by seven for a step's global
//! attention. Both arms of each pair are the same binary in the same working
//! directory, switched by `INK_FLASH`, because cubecl's autotune cache lives at
//! `$CWD/target/autotune` and a cross-worktree comparison measures the cache.
//! `reserved` is what the cubecl pool holds afterwards — a high-water mark
//! within a process, so a row inherits every row above it.
//! [`super::burn::flash_prefill_cost_at_length`] and
//! `flash_decode_cost_at_context` are the two benchmarks; both are `#[ignore]`d.
//!
//! DECODE, one step, NVFP4 KV:
//!
//! ```text
//!   context   window     fused    dense
//!      4 096        -      1.17     1.58   1.35x
//!     16 384        -      2.60     4.01   1.54x
//!     65 536        -      7.76    29.94   3.86x
//!    262 144        -     28.79   120.17   4.17x
//!     16 384      512      0.76     1.01   1.33x
//! ```
//!
//! The last row is a LOCAL layer — the window trims the cache to its last 512
//! keys, so it measures a 512-key read whatever the context says — and it is
//! there because [`super::burn::attention_step`] does not check `kind`: all
//! forty-two layers take this kernel at decode, not just the seven.
//!
//! PREFILL, whole sequence:
//!
//! ```text
//!    tokens     fused    dense      fused MiB   dense MiB
//!     1 024      14.4     15.3           2542        2542
//!     4 096      90.2    159.6           3021       10199
//!     8 192     283.3    583.1           3051       40824
//!    16 384    1206.5   2210.7           3051       40824
//!    32 768    5515.2   8728.5          11186       41303
//! ```
//!
//! So it pays everywhere, and the answer to "at what N does it start to pay"
//! is "at the shortest length either benchmark runs" — which is not what I
//! expected to find, and the reason is that the arm it replaces was never
//! compute-bound. Its cost is the `[heads, rows, tokens]` score block written
//! by the matmul and walked five more times, plus a GQA expansion of K and V
//! that is linear in the sequence with no knob on it. The memory column is the
//! sharper result: 3.0 GiB at 16k tokens against 40.8, and the fused figure is
//! flat from 4k to 16k where the dense one is not.
//!
//! ## What it does NOT reach, and why, and what would
//!
//! 5.5 s for one layer at 32,768 tokens is 8.8 TFLOP of arithmetic in 5.5 s,
//! i.e. **1.6 TFLOP/s** — under 1% of what this part's tensor cores can do.
//! That is not a mystery and it is not headroom that tuning recovers. One key
//! tile costs `2 * PLANE * head_dim * 4` bytes of K and V and buys
//! `ROWS * PLANE * head_dim * 2` flops, so the kernel's arithmetic intensity is
//! `ROWS / 4` flops a byte — 8 at [`ROWS_PREFILL`] — and 8 flops a byte against
//! this box's ~273 GB/s is a **2.2 TFLOP/s ceiling**. The measurement is at 73%
//! of it. The kernel is at its intensity limit, not at its instruction limit.
//!
//! Two things raise that ceiling and neither is a tuning knob:
//!
//! * **More rows per cube.** Intensity is `ROWS / 4` and nothing else. 64 rows
//!   would double it; what stops it is the 48 KiB a static `__shared__` gets,
//!   since the query tile alone would be 32 KiB at f32. Holding the query tile
//!   in BF16 — which the arm this replaces already does for its matmul — fits
//!   64 rows in 40.5 KiB. That is a change to the kernel's operand precision
//!   and wants its own pairing against `golden/paired/`.
//! * **MMA.** At prefill `m` is large and `mma.sync.aligned.m16n8k16` is what
//!   the score product should be running on; [`super::bf16gemm`] already uses
//!   it in this tree. At DECODE it buys nothing — `m` is 1, the product is a
//!   GEMV, and the decode numbers above are bandwidth-bound already.
//!
//! # Burn already has a flash kernel, and it was rejected on evidence
//!
//! `cubek-attention` 0.2.0 is in our `Cargo.lock` and already compiled into
//! every mary build — one feature word away from `cubek::attention::launch::
//! launch_ref`. It is not used here, and the reasons are three properties of
//! its interface rather than a preference:
//!
//! * **The relative-position bias is not expressible.** Its mask is a BOOLEAN
//!   predicate — `MaterializedTileMask::should_mask() -> bool`, with a `u8`/
//!   `u32` mask dtype — not an addend. (`MaskTile`'s "additive mask" doc
//!   comment is stale.) And our own pinned burn fork closes the door
//!   independently: `burn-cubecl/src/ops/module.rs:325-330` routes to
//!   `attention_fallback` the moment `attn_bias`, `softcap` or `scale` is
//!   present, and the fallback materialises `[b, h, n, n]` — which is the exact
//!   thing this file exists to stop. Inkling's every layer carries a learned
//!   per-head bias gathered by backward distance; there is no arm of this model
//!   without one.
//! * **A paged, NVFP4 KV cache cannot be handed to it.** K and V must be
//!   strided float tensors read through `AttentionGlobalLayout`; the crate
//!   contains no `paged`, `fp4` or `quant`. Using it means materialising one
//!   contiguous BF16 cache per layer per step — the copy `KvStore::parts` was
//!   changed to stop making.
//! * **GQA 32/8 is silently wrong rather than refused.** `AttentionDims` takes
//!   ONE `num_heads`, from `query.shape[1]`; `key.shape[1]` is never read and
//!   nothing validates the pair.
//!
//! Structurally it also has no split over the key axis — its hypercube is
//! `(seq_q_tile, batch * num_heads)` — so at decode it is thirty-two cubes each
//! walking the whole cache serially, which is the shape this file's splits
//! exist to fix. 0.3.0-pre.3 keeps all four and adds a training backward pass,
//! at the cost of rebasing our cubecl fork off 0.10. Upstream is aware: cubek
//! #284 ("Paged flash attention support") has been open since 2026-05-19 and
//! #310 migrates the kernel to the Tile DSL. This is the same answer
//! [`super::banded`] reached for the thirty-five local layers, for the same
//! kind of reason.
//!
//! One thing upstream has that is worth borrowing later: 0.3's
//! `AttentionGlobalLayout::new` takes separate batch and head strides with
//! explicit broadcast, which makes 32/8 GQA expressible as a strided VIEW — Q
//! as `[8, 4, sq, d]` against K/V as `[8, 1, n, d]` with the head axis
//! broadcast. That is the same correspondence this kernel spells as a cube tile
//! (`groups` heads by `rows / groups` queries), and it is the tidier
//! formulation of the two.
//!
//! # Historical packed-reader measurements
//!
//! The historical `INK_FLASH_FP4=1` experiment swapped the fetch for the packed
//! one; that environment switch is no longer read. Sessions now select
//! [`CachedAttentionPolicy`] explicitly, with [`CachedAttentionPolicy::Legacy`]
//! preserving the dense reader and decode-sized tile by default. The following
//! measurements describe the earlier reader experiments, not the batched policy.
//!
//! With the packed reader, the kernel reads the
//! stored E2M1 codes and E4M3 block scales and decodes in registers, and no
//! dequantised page is built at all. It is one comptime branch at two fetch
//! sites; everything below them is the same code. The old note here said this
//! would "save the page-sized round trip through L2" and called it a change to
//! the READER, not to this kernel's shape. The second half was right. The
//! first half was wrong about L2, and the difference is the whole result.
//!
//! ## The round trip is to DRAM, not through L2
//!
//! **Framing rule for this section.** Per DECODE STEP of `INK_LAYERS=0:21` at
//! ctx 3732, `INK_KV=1`, NVFP4 KV, one GB10, median of six warm steps (spread
//! under 0.5%). Bytes are `lts__d_sectors_fill_sysmem.sum * 32 B`, which is the
//! DRAM read on this part: GB10 exposes **no `dram__` metric at all** —
//! `ncu --query-metrics` lists none — and `lts__d_sectors_fill_device.sum` is
//! structurally zero because the memory is unified, so there is no separate
//! device aperture to fill from. Anything asking this part for
//! `dram__bytes_read.sum` gets an error, not a number.
//! (`lts__t_sectors_op_read_lookup_miss` is NOT the counter to use either: it
//! reads 3% miss on the dequant's packed input, which cannot possibly be
//! resident.)
//!
//! | kernel | launches | DRAM read | L2 read | DRAM/L2 |
//! | --- | ---: | ---: | ---: | ---: |
//! | `dequantize_nvfp4` | 84 | 26.5 MiB | 25.7 MiB | 1.03 |
//! | `flash_kernel` (dense) | 42 | 82.9 MiB | 83.8 MiB | **0.99** |
//! | `flash_kernel` (packed) | 42 | 26.3 MiB | 34.0 MiB | 0.77 |
//!
//! **99% of what the dense arm reads back comes from memory.** The pages the
//! dequant wrote microseconds earlier, into a 24 MiB L2, are not there any
//! more. That also settles the write half without assuming anything: flash read
//! those bytes from DRAM, so the lines had been evicted, and evicting a dirty
//! line IS the write-back. Both directions are real traffic, and the headroom
//! is therefore BOTH halves rather than only the one L2 residency could reach.
//!
//! The whole KV path, per 21-layer step: 26.5 MiB of codes read, 94.2 MiB of
//! BF16 written (the read times the kernel's exact 3.556x expansion — there is
//! no DRAM-write counter on this part, and flash's 82.9 MiB read-back is a
//! floor under it), 82.9 MiB read back. **203.6 MiB, against 26.3 for the
//! packed arm: 177 MiB a step removed at 21 layers, ~355 at 42.**
//!
//! One counter needs its own warning. `lts__t_sectors_op_write.sum` on the
//! dequant reads 1393 MiB a step, 16.8x its actual output, because it counts
//! sector REQUESTS and the kernel does sixteen separate 2-byte stores per
//! thread at a 32-byte stride — every store instruction touches 32 sectors for
//! two bytes each. It is not bandwidth (`write_lookup_miss` is ~0, so L2's byte
//! enables absorb it) but it is LSU instructions, and multiplying that counter
//! by 32 gives a number 16.8x too large.
//!
//! ## And at ctx 3732 it is 1.8% SLOWER
//!
//! Paired, ABBA-balanced arm order inside one interleaved run, n = 7, same
//! binary, ctx 3732, `INK_LAYERS=0:21`, `INK_GEN=40`, one GB10: **+1.80% on
//! step time, 95% CI +1.43..+2.17**, against A/A controls of -0.29% and
//! +0.06%. 44.9 ms/step against 45.7. An earlier build that decoded through the
//! `e2m1_value` comparison ladder rather than
//! [`super::fp4quant::e2m1_bits`] read +1.83% (CI +1.73..+1.93) — the same
//! number, which is itself the finding below.
//!
//! The reason is in this file's own doc, four sections up. Split by grid, the
//! 42 flash launches of one warm step at ctx 3732 are:
//!
//! | layers | grid | cubes | DRAM read |
//! | --- | --- | ---: | ---: |
//! | 3 global | `(1, 8, 29)` | 232 | 44.0 MiB |
//! | 18 local | `(1, 8, 4)` | 32 | 36.1 MiB |
//! | 21 tail pages | `(1, 8, 1)` | 8 | 2.9 MiB |
//!
//! **Nearly half the traffic moves in grids of 32 and 8 cubes on a 48-SM part**,
//! at `sm__warps_active` 8.3% of peak. "That is not a kernel, it is a queue":
//! it is LATENCY-bound, not bandwidth-bound, so bytes removed from it are bytes
//! it was not waiting on, while the per-element decode lands straight on the
//! critical path.
//!
//! **And the cost is not arithmetic**, which is measured rather than argued:
//! replacing the seven-comparison ladder with the branchless bit construction
//! roughly halved the decode's operation count and gave back every register it
//! had cost (47 -> 39, one BELOW the dense arm), and moved the step time by
//! nothing at all — 1.83% to 1.80%, each well inside the other's interval.
//!
//! What that leaves, and this part is inference rather than measurement, is the
//! LOADS: the packed reader issues two an element (a code word and a scale
//! byte) where the dense one issues one, in a loop each of the four planes runs
//! redundantly — `vv` depends only on `(j, di, lane)`, so all four decode the
//! same element. Anyone testing that should put
//! `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` on both arms; it was not
//! in the session that produced the numbers above.
//!
//! ## The crossover is real, and it is where the argument said it would be
//!
//! At ctx 14928 with `INK_LAYERS=0:8` — where the model still fits one box and
//! one global layer carries 58.2 MiB of the step's 58.6 — the same paired ABBA
//! design at n = 7 reads **-1.15% on step time, the packed arm FASTER**, 95% CI
//! -1.65..-0.64, every one of the seven reps negative, against A/A controls of
//! -0.06% and -0.38%. The grid there is `(1, 8, 116)` — **928 cubes**, a full
//! device rather than a queue — and the sign flips because the kernel is
//! finally waiting on the bytes this change removes.
//!
//! Read that beside what `bench-decode` prints for the same run, which is
//! "+0.93% tok/s, SMALLER THAN THE SPREAD. Not a result." Both are correct and
//! they are not the same statistic: the script compares each arm's median
//! against the other's, and at long context the per-arm spread is 2-4%. The
//! PAIRED difference inside each rep is what resolves 1%, which is why the
//! arms are interleaved in the first place. An unpaired reading of this run
//! would have concluded nothing.
//!
//! That was the direct-reader result. The staged-value tile below removes the
//! duplicated work without changing the lane-to-output mapping: the 128 units
//! cooperatively expand each 32-key V tile once, one whole sixteen-value NVFP4
//! block per unit, into the key tile's now-dead shared memory. All four planes
//! then consume that one expansion. Each pair of packed words and its scale is
//! loaded once per tile rather than once per value per plane.
//!
//! The same-binary, four-arm ABBA gate after that change is favorable at both
//! lengths. Each row is a median over 39 warm decode steps, and each paired
//! sample is `mean(tile, tileB) / mean(dense, denseB) - 1` within one ABBA
//! block; the interval is Student-t over seven samples. Both lengths use
//! `INK_LAYERS=0:8`, `INK_GEN=40`, one GB10, and differ only in the input
//! context and the historical `INK_FLASH_FP4` switch:
//!
//! | input context | paired step-time delta | 95% CI | favorable blocks |
//! | ---: | ---: | ---: | ---: |
//! | 3732 | **-0.74%** | -1.18..-0.31% | 7 / 7 |
//! | 14928 | **-4.85%** | -5.47..-4.23% | 7 / 7 |
//!
//! The generation sequence is bit-identical across all 28 processes at each
//! length, and the device gate below still requires the packed and dequantized
//! readers to be bit-identical over decode, prefill, page cuts, GQA and windows.
//! Raw logs and TSVs are preserved as Files import
//! `d6431dfa4c1e699b767e424a2785c1a4`.
//!
//! Those timings were decode-only. The initial uncached prefill still reads
//! freshly projected K and V, not packed cache pages. Cached multi-query
//! extensions also use this kernel now; their reader and query tile are selected
//! together by the Session's explicit policy and need separate measurements.
//!
//! ## What it costs in registers, and what it does not spill
//!
//! `flash_kernel` at the decode tile is **39 registers a thread against the
//! dense arm's 40**, and `launch__occupancy_limit_registers` is 12 blocks an SM
//! on both. Neither binds: `launch__occupancy_limit_shared_mem` is 5 on both.
//! Local memory is **0 bytes loaded and 0 stored on both arms**, so nothing
//! spilled — worth having measured rather than inferred, because this is the
//! kernel where a spill would be silent.
//!
//! ## What it removes regardless of the clock
//!
//! In that historical growing-page layout, 84 dequant launches and 84 per-step
//! allocations a step at 21 layers (168 at 42), sized by
//! [`super::kvpages::Pages::read_rows`] and therefore re-sized
//! whenever the read window grows. The flash partials `po` and `pml` remain and
//! are still sized from the span, so this does not by itself make the decode
//! path allocation-free — but it removes the buffer the KV-preallocation work
//! named as the remaining per-step epoch.
//!
//! It also has no MMA path. At decode `m = 1` and tensor cores buy nothing —
//! the product is a GEMV and the kernel is bandwidth-bound. At prefill `m` is
//! large and they would; see the measurements in the commit for where that
//! leaves the prefill lane against burn's cmma matmul.

use super::fp4quant::e2m1_bits;
use cubecl::e4m3;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Units per cube. Four planes of thirty-two on this backend.
pub const UNITS: u32 = 128;

/// Lanes in a plane, and therefore keys in one key tile.
///
/// The two are the same number on purpose: it is what makes a row's whole tile
/// of scores live in one plane, so the online softmax's two reductions are
/// `plane_max` and `plane_sum` rather than a shared-memory tree with a
/// `sync_cube` per level.
pub const PLANE: u32 = 32;

/// Planes per cube.
pub const PLANES: u32 = UNITS / PLANE;

/// Query rows in a PREFILL cube: `groups` heads by `ROWS / groups` queries.
///
/// 32 at Inkling's `groups = 4` is eight consecutive queries across all four
/// heads of a KV head. It sets the arithmetic intensity of the whole kernel —
/// one key tile costs `2 * PLANE * head_dim * 4` bytes of K and V and buys
/// `ROWS * PLANE * head_dim * 2` flops, i.e. `ROWS / 4` flops a byte — and it
/// sets the shared budget, which is what stops it being larger.
pub const ROWS_PREFILL: u32 = 32;

/// Query rows in a DECODE cube: one query, all `groups` heads of a KV head.
///
/// A decode step has one query row, so `ROWS` cannot exceed `groups` without
/// padding the tile with rows that do not exist. Four rows is four accumulator
/// registers a lane and the highest occupancy this kernel has, which is what a
/// bandwidth-bound pass wants.
pub const ROWS_DECODE: u32 = 4;

/// Immutable cached-attention execution policy, selected when a Session loads.
///
/// Initial uncached prefill is unchanged. The candidate reads eligible NVFP4
/// cache pages directly and uses a wider query tile only when there are enough
/// queries to fill it. Changing the tile also changes split/reduction grouping,
/// so this policy is part of persistent cache compatibility, not a live toggle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CachedAttentionPolicy {
    #[default]
    #[serde(rename = "legacy")]
    Legacy,
    #[serde(rename = "packed-batched-v1")]
    PackedBatchedV1,
}

impl CachedAttentionPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::PackedBatchedV1 => "packed-batched-v1",
        }
    }

    /// The default is omitted from serialized cache geometry to preserve the
    /// already-written legacy identity byte for byte.
    pub const fn is_legacy(&self) -> bool {
        matches!(self, Self::Legacy)
    }

    /// Whether this shape may use packed KV. Dense stores and head widths that
    /// end inside an NVFP4 block retain the dense reader under either policy.
    pub fn packed_reader(self, head_dim: usize) -> bool {
        !self.is_legacy() && head_dim != 0 && packed_head_dim_applies(head_dim)
    }

    /// Eligible tile for a cached query batch, or the existing unfused fallback.
    /// A batch of one always has the same tile as single-token decode. If the
    /// wider tile exceeds the shared-memory limit, keep the decode tile.
    pub fn tile_rows(self, heads: usize, kv_heads: usize, head_dim: usize, nq: usize) -> Option<usize> {
        if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 || nq == 0 {
            return None;
        }
        let groups = heads / kv_heads;
        if !self.is_legacy() {
            let rows = prefill_rows(groups);
            if nq >= rows / groups && applies(heads, kv_heads, head_dim, rows) {
                return Some(rows);
            }
        }
        let rows = decode_rows(groups);
        applies(heads, kv_heads, head_dim, rows).then_some(rows)
    }
}

/// Negative infinity as the softmax's identity, spelled the way the rest of
/// this module tree spells it: a large FINITE constant, so every guard against
/// it has to be explicit rather than riding on infinity's algebra.
const NEG_INF: f32 = -3.4028235e38;

/// The shared floats one cube of this shape needs.
///
/// Q tile, key tile padded to `head_dim + 1` a row, and the probability tile.
/// The CUDA static `__shared__` ceiling is 48 KiB and this kernel stays under
/// it deliberately: going above needs an opt-in the runtime does not make, and
/// the occupancy a bigger tile would cost is worth more here than the tile is.
pub fn shared_floats(rows: u32, head_dim: u32) -> usize {
    (rows * head_dim + PLANE * (head_dim + 1) + rows * PLANE) as usize
}

/// Element `idx` of an NVFP4 buffer, dequantised in registers.
///
/// `idx` is the index into the buffer's LOGICAL `[rows, kv_heads * head_dim]`
/// extent — the same index the dense reader uses — because
/// [`super::kvpages::Fp4Rows`] quantizes the page row-major with the block
/// running along the feature axis. So the code lives in word `idx / 8` at
/// nibble `idx % 8`, and its block scale is E4M3 byte `idx / 16`. Nothing here
/// is a function of the page's row count, which is what lets the caller hand
/// over a prefix of a reserved page by shortening a scalar.
///
/// This is [`super::fp4quant::dequantize_nvfp4_kernel`]'s arithmetic for ONE
/// element instead of for a sixteen-element block. It decodes through
/// [`super::fp4quant::e2m1_bits`] rather than the `e2m1_value` ladder the
/// dequant kernel uses, because here it is the innermost loop -- see that
/// function for why there are two, and for what stops them drifting.
#[cube]
fn nvfp4_at(codes: &Array<u32>, scales: &Array<e4m3>, idx: u32) -> f32 {
    let word = codes[(idx / 8) as usize];
    let code = (word >> ((idx % 8) * 4)) & 15;
    e2m1_bits(code) * f32::cast_from(scales[(idx / 16) as usize])
}

/// One `(query tile, KV head, key split)` of global attention, fused.
///
/// Writes an UNNORMALISED partial: `o = sum exp(s - m) v`, beside `m` and
/// `l = sum exp(s - m)`. [`flash_combine_kernel`] turns a column of those into
/// the answer.
///
/// `k` and `v` are generic in their element type because the cache is BF16 (and
/// the prefill operands are too, under `INK_ACT_BF16`) while everything the
/// softmax touches is f32. The cast happens on the way into shared memory and
/// into a register, which is where the reference implementation puts it as
/// well: narrow operands, wide accumulation.
///
/// # The two readers, and why they are one kernel
///
/// `packed` swaps the reader for NVFP4: `kc`/`vc` are the packed E2M1 codes,
/// eight to a `u32`, and `ks`/`vs` are the E4M3 block scales, one per sixteen
/// consecutive features — the layout [`super::kvpages::Fp4Rows`] stores and
/// [`super::fp4quant::dequantize_nvfp4_bf16`] otherwise expands into a BF16
/// page before this kernel ever runs. The index arithmetic is the SAME
/// `row * kv_row + kvh * head_dim + d`; only the fetch differs.
///
/// It is a comptime branch inside one kernel rather than a second kernel
/// because everything below the two fetches — the causal and window trimming,
/// the online softmax's guarded rescale, the split bookkeeping — is subtle,
/// tested once, and must not be able to fork. The unused pair of arrays costs
/// nothing at runtime: a bound array a kernel never reads is a constant-bank
/// pointer, not a register.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn flash_kernel<KV: Numeric>(
    q: &Array<f32>,
    k: &Array<KV>,
    v: &Array<KV>,
    kc: &Array<u32>,
    ks: &Array<e4m3>,
    vc: &Array<u32>,
    vs: &Array<e4m3>,
    rel: &Array<f32>,
    gap_rows: &Array<u32>,
    gap_cums: &Array<u32>,
    po: &mut Array<f32>,
    pml: &mut Array<f32>,
    scaling: f32,
    nq: u32,
    q0: u32,
    k0: u32,
    klo: u32,
    khi: u32,
    eff: u32,
    window: u32,
    slot0: u32,
    splits: u32,
    slots: u32,
    ngaps: u32,
    row0: u32,
    #[comptime] heads: u32,
    #[comptime] kv_heads: u32,
    #[comptime] head_dim: u32,
    #[comptime] rows: u32,
    #[comptime] packed: bool,
    #[comptime] gapped: bool,
) {
    let qt = CUBE_POS_X;
    let kvh = CUBE_POS_Y;
    let split = CUBE_POS_Z;
    let u = UNIT_POS_X;
    let lane = u % PLANE;
    let plane = u / PLANE;

    let groups = comptime!(heads / kv_heads);
    let bq = comptime!(rows / groups);
    let rpp = comptime!(rows / PLANES);
    // Output dimensions a lane owns. `div_ceil`, not `/`: the model's
    // `head_dim` is 128 and divides exactly, but the test shapes are 4 wide and
    // a kernel that only ran on multiples of the plane could not be checked
    // against the harness that decides whether it is right.
    let dpl = comptime!(head_dim.div_ceil(PLANE));
    let q_row = comptime!(heads * head_dim);
    let kv_row = comptime!(kv_heads * head_dim);
    let skw = comptime!(head_dim + 1);

    let mut sq = SharedMemory::<f32>::new(comptime!((rows * head_dim) as usize));
    let mut sk = SharedMemory::<f32>::new(comptime!((PLANE * (head_dim + 1)) as usize));
    let mut sp = SharedMemory::<f32>::new(comptime!((rows * PLANE) as usize));

    // --- the query tile ------------------------------------------------------
    // Row `r` is `(head group r / bq, query qt * bq + r % bq)`. Consecutive
    // units take consecutive dimensions of one row, so the load is coalesced.
    let mut i = u;
    while i < rows * head_dim {
        let r = i / head_dim;
        let d = i % head_dim;
        let qrow = qt * bq + r % bq;
        let h = kvh * groups + r / bq;
        let mut val = f32::new(0.0);
        if qrow < nq {
            val = q[(qrow * q_row + h * head_dim + d) as usize];
        }
        sq[i as usize] = val;
        i += UNITS;
    }

    // --- this split's slice of the key axis ----------------------------------
    // The tile's queries span absolute positions `q0 + qt*bq ..= q_hi`. No key
    // past `q_hi` can be seen by ANY row here (causality), and with a window no
    // key before `q_lo - window + 1` can be seen by any of them. Trimming the
    // slice by both is not the mask — every cell is still masked individually
    // below — it is what keeps the tile loop off ranges that are entirely dead.
    let q_lo = q0 + qt * bq;
    let mut q_hi = q0 + nq - 1;
    if qt * bq + bq <= nq {
        q_hi = q_lo + bq - 1;
    }

    let mut lo = klo;
    let mut hi = khi;
    // Causality, in chunk-local rows. `q_hi + 1 <= k0` means the whole chunk is
    // in this tile's future.
    if q_hi + 1 <= k0 {
        hi = lo;
    } else if q_hi + 1 - k0 < hi {
        hi = q_hi + 1 - k0;
    }
    // The window's lower edge, likewise.
    if window > 0 && q_lo + 1 > window + k0 {
        let wlo = q_lo + 1 - window - k0;
        if wlo > lo {
            lo = wlo;
        }
    }
    if hi < lo {
        hi = lo;
    }

    // Split the surviving range. `per` is rounded up so `splits` slices cover
    // it; the last ones may be empty, and an empty slice still writes its
    // partial (`m = NEG_INF`, `l = 0`), which the combine weights to zero.
    let span = hi - lo;
    let per = (span + splits - 1) / splits;
    let mut s_lo = lo + split * per;
    if s_lo > hi {
        s_lo = hi;
    }
    let mut s_hi = s_lo + per;
    if s_hi > hi {
        s_hi = hi;
    }

    // --- running softmax state, per row this plane owns ----------------------
    let mut m = Array::<f32>::new(comptime!(rpp as usize));
    let mut l = Array::<f32>::new(comptime!(rpp as usize));
    let mut acc = Array::<f32>::new(comptime!(rpp as usize));
    let mut o = Array::<f32>::new(comptime!((rpp * dpl) as usize));
    #[unroll]
    for ri in 0..rpp {
        m[ri as usize] = f32::new(NEG_INF);
        l[ri as usize] = f32::new(0.0);
    }
    #[unroll]
    for oi in 0..rpp * dpl {
        o[oi as usize] = f32::new(0.0);
    }

    sync_cube();

    let mut t = s_lo;
    while t < s_hi {
        // --- stage the key tile ---------------------------------------------
        // 128 units over `PLANE * head_dim` floats: consecutive units take
        // consecutive dimensions of one key, which is a coalesced line in the
        // token-major layout the projections and the cache both produce. See
        // the module doc for why the destination row is `head_dim + 1` wide.
        let mut ki = u;
        while ki < PLANE * head_dim {
            let j = ki / head_dim;
            let d = ki % head_dim;
            let key = t + j;
            let mut val = f32::new(0.0);
            if key < s_hi {
                let idx = key * kv_row + kvh * head_dim + d;
                if comptime![packed] {
                    val = nvfp4_at(kc, ks, idx);
                } else {
                    val = f32::cast_from(k[idx as usize]);
                }
            }
            sk[(j * skw + d) as usize] = val;
            ki += UNITS;
        }
        sync_cube();

        // --- scores: lane `l` owns key `t + l`, for every row this plane holds
        // The dimension loop is outermost so the staged key is read ONCE per
        // dimension and reused across the plane's rows, which is `rpp` times
        // fewer shared reads than the natural nesting.
        #[unroll]
        for ri in 0..rpp {
            acc[ri as usize] = f32::new(0.0);
        }
        for d in 0..head_dim {
            let kd = sk[(lane * skw + d) as usize];
            #[unroll]
            for ri in 0..rpp {
                acc[ri as usize] += sq[((plane * rpp + ri) * head_dim + d) as usize] * kd;
            }
        }

        let key = t + lane;
        let mut key_abs = k0 + key;
        if comptime![gapped] {
            // Positions evicted before this key: the last gap at or before its
            // logical row says how many were skipped up to and including it.
            // Gaps are few (a fold a time), so a scan beats a search here.
            let r = row0 + key;
            let mut skipped = 0u32;
            for g in 0..ngaps {
                if gap_rows[g as usize] <= r {
                    skipped = gap_cums[g as usize];
                }
            }
            key_abs += skipped;
        }
        let visible_key = key < s_hi;

        #[unroll]
        for ri in 0..rpp {
            let r = plane * rpp + ri;
            let qrow = qt * bq + r % bq;
            let h = kvh * groups + r / bq;
            let q_abs = q0 + qrow;

            let mut s = acc[ri as usize] * scaling;
            let mut vis = visible_key && qrow < nq && key_abs <= q_abs;
            if vis && window > 0 && q_abs - key_abs >= window {
                vis = false;
            }
            if vis {
                let dist = q_abs - key_abs;
                if dist < eff {
                    s += rel[((qrow * heads + h) * eff + dist) as usize];
                }
            } else {
                s = f32::new(NEG_INF);
            }

            // The online update. Every step is guarded rather than left to
            // infinity's algebra, because NEG_INF here is finite.
            let tile_max = plane_max(s);
            let mut m_new = m[ri as usize];
            if tile_max > m_new {
                m_new = tile_max;
            }
            let mut alpha = f32::new(1.0);
            if m_new > f32::new(NEG_INF) {
                alpha = Exp::exp(m[ri as usize] - m_new);
            }
            let mut p = f32::new(0.0);
            if vis {
                p = Exp::exp(s - m_new);
            }
            let tile_sum = plane_sum(p);
            l[ri as usize] = l[ri as usize] * alpha + tile_sum;
            m[ri as usize] = m_new;
            #[unroll]
            for di in 0..dpl {
                o[(ri * dpl + di) as usize] *= alpha;
            }
            sp[(r * PLANE + lane) as usize] = p;
        }
        sync_cube();

        // --- the value average ----------------------------------------------
        // The four planes need the SAME V tile.  The dense reader leaves the
        // old direct load alone: its one BF16 load is an L1 hit in the three
        // planes which arrive after the first.  The packed reader cannot do
        // that cheaply -- every apparent load is also a nibble extraction and
        // a scale load -- so its 128 units cooperatively expand the tile ONCE
        // into the key tile's now-dead shared memory.
        //
        // One unit owns one whole sixteen-value NVFP4 block at a time.  That is
        // the storage algebra rather than merely coalesced scalar indexing:
        // one E4M3 scale and two u32 words become sixteen consecutive values.
        // Every packed launch admits only head dimensions that are a multiple
        // of the sixteen-value NVFP4 group, hence a block cannot straddle
        // either a head or a key row.
        if comptime![packed] {
            let blocks = comptime!(PLANE * head_dim / 16);
            let mut block = u;
            while block < blocks {
                let first = block * 16;
                let j = first / head_dim;
                let d0 = first % head_dim;
                let vkey = t + j;

                let mut word0 = u32::new(0);
                let mut word1 = u32::new(0);
                let mut scale = f32::new(0.0);
                if vkey < s_hi {
                    let idx = vkey * kv_row + kvh * head_dim + d0;
                    word0 = vc[(idx / 8) as usize];
                    word1 = vc[(idx / 8 + 1) as usize];
                    scale = f32::cast_from(vs[(idx / 16) as usize]);
                }

                #[unroll]
                for e in 0u32..16u32 {
                    let mut code = (word0 >> (e * 4u32)) & 15u32;
                    if e >= 8u32 {
                        code = (word1 >> ((e - 8u32) * 4u32)) & 15u32;
                    }
                    sk[(j * skw + d0 + e) as usize] = e2m1_bits(code) * scale;
                }
                block += UNITS;
            }
            // Every plane consumes every staged row below.  This barrier is
            // absent from the dense instantiation because `packed` is comptime.
            sync_cube();
        }

        // Lane `l` owns output dimensions `l, l+32, ...`.  Packed V now comes
        // from the one cooperatively expanded tile; dense V keeps its direct
        // consecutive read and L1 sharing.
        for j in 0..PLANE {
            let vkey = t + j;
            if vkey < s_hi {
                let vbase = vkey * kv_row + kvh * head_dim;
                #[unroll]
                for di in 0..dpl {
                    let d = lane + di * PLANE;
                    if d < head_dim {
                        let mut vv = f32::new(0.0);
                        if comptime![packed] {
                            vv = sk[(j * skw + d) as usize];
                        } else {
                            vv = f32::cast_from(v[(vbase + d) as usize]);
                        }
                        #[unroll]
                        for ri in 0..rpp {
                            o[(ri * dpl + di) as usize] +=
                                sp[((plane * rpp + ri) * PLANE + j) as usize] * vv;
                        }
                    }
                }
            }
        }
        sync_cube();

        t += PLANE;
    }

    // --- the partial ---------------------------------------------------------
    let slot = slot0 + split;
    #[unroll]
    for ri in 0..rpp {
        let r = plane * rpp + ri;
        let qrow = qt * bq + r % bq;
        let h = kvh * groups + r / bq;
        if qrow < nq {
            let base = ((slot * nq + qrow) * heads + h) * head_dim;
            #[unroll]
            for di in 0..dpl {
                let d = lane + di * PLANE;
                if d < head_dim {
                    po[(base + d) as usize] = o[(ri * dpl + di) as usize];
                }
            }
            if lane == 0 {
                let mb = ((slot * nq + qrow) * heads + h) * 2;
                pml[mb as usize] = m[ri as usize];
                pml[(mb + 1) as usize] = l[ri as usize];
            }
        }
    }
    let _ = slots;
}

/// Merge a column of partials into the answer.
///
/// One cube per `(query row, head)`, `head_dim` units, unit `d` owning output
/// dimension `d`. The merge is the tile loop's rescale one level up: every
/// split's `o` was accumulated against its own `m`, so weighting by
/// `exp(m_s - M)` puts them all on the same exponent and the sum of the
/// likewise-weighted `l` is the denominator.
#[cube(launch_unchecked)]
fn flash_combine_kernel(
    po: &Array<f32>,
    pml: &Array<f32>,
    out: &mut Array<f32>,
    slots: u32,
    nq: u32,
    #[comptime] heads: u32,
    #[comptime] head_dim: u32,
) {
    let qrow = CUBE_POS_X;
    let h = CUBE_POS_Y;
    let d = UNIT_POS_X;

    let mut mx = f32::new(NEG_INF);
    for s in 0..slots {
        let mb = ((s * nq + qrow) * heads + h) * 2;
        if pml[mb as usize] > mx {
            mx = pml[mb as usize];
        }
    }

    let mut num = f32::new(0.0);
    let mut den = f32::new(0.0);
    for s in 0..slots {
        let mb = ((s * nq + qrow) * heads + h) * 2;
        let ms = pml[mb as usize];
        let ls = pml[(mb + 1) as usize];
        let mut wt = f32::new(0.0);
        if ms > f32::new(NEG_INF) {
            wt = Exp::exp(ms - mx);
        }
        den += wt * ls;
        num += wt * po[(((s * nq + qrow) * heads + h) * head_dim + d) as usize];
    }

    let mut res = f32::new(0.0);
    if den > f32::new(0.0) {
        res = num / den;
    }
    out[(qrow * heads * head_dim + h * head_dim + d) as usize] = res;
}

/// The element type a key/value buffer is held in.
///
/// Not `burn::tensor::DType`: this file is runtime-generic the way
/// [`super::banded`] is, and the one thing it needs to know about the cache's
/// dtype is which instantiation of [`flash_kernel`] to launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvElem {
    F32,
    Bf16,
    /// Packed NVFP4: the run's `k`/`v` are the E2M1 code words and its
    /// `k_scales`/`v_scales` are the E4M3 block scales beside them. Nothing is
    /// expanded before the kernel runs.
    Nvfp4,
}

/// One run of contiguous key/value rows the kernel may read.
///
/// A prefill hands one of these — the whole freshly-projected K and V. A decode
/// step hands one PER PAGE, because that is how the cache is stored and
/// rejoining them would be the copy the paged read exists to avoid.
///
/// `base` is the ABSOLUTE sequence position of row 0, which is what causality,
/// the window and the relative distance are all functions of; `lo .. hi` are
/// the rows of this buffer that are live keys, which is how a page's dropped
/// prefix and its tail padding are excluded without slicing the buffer and
/// making its shape walk.
#[derive(Debug, Clone, Copy)]
pub struct KeyRun<'a> {
    pub k: &'a Handle,
    pub v: &'a Handle,
    /// The E4M3 block scales, on [`KvElem::Nvfp4`] and only there — where `k`
    /// and `v` are the packed code words rather than dense values.
    pub k_scales: Option<&'a Handle>,
    pub v_scales: Option<&'a Handle>,
    /// Rows in the buffer.
    pub rows: usize,
    /// Absolute sequence position of row 0.
    pub base: usize,
    /// First live row.
    pub lo: usize,
    /// One past the last live row.
    pub hi: usize,
    /// Logical row (in the cache's own numbering, dead prefix excluded) of
    /// stored row 0 of this run: what the gap table is indexed by. Zero when
    /// there is one run and no dead prefix, which is every case but a
    /// windowed layer's, and windowed layers never carry gaps.
    pub row0: usize,
}

/// Where a cache has EVICTED rows, and how far each gap moves the absolute
/// position of every key after it.
///
/// `rows[g]` is the logical row the `g`-th gap begins at (the first surviving
/// row after it) and `cums[g]` the number of positions skipped up to and
/// including that gap, non-decreasing in `g`. A key at logical row `r` stands
/// at `base + r + cums[last g with rows[g] <= r]`, or `base + r` with no such
/// gap -- so with no gaps at all the kernel computes exactly what it always
/// has. Both live on the device; `count` is their length.
#[derive(Debug, Clone, Copy)]
pub struct Gaps<'a> {
    pub rows: &'a Handle,
    pub cums: &'a Handle,
    pub count: usize,
}

/// How many cubes the launcher aims for before it stops splitting the key axis.
///
/// A decode step's natural grid is `1 x kv_heads x 1` — eight cubes — and a
/// device with dozens of SMs will run that at a few percent of its bandwidth.
/// Splitting to roughly this many is what fills it. Multi-query batches can
/// supply that parallelism on the query axis alone; wider query tiles may
/// reduce their grid enough to require key splits again.
const TARGET_CUBES: usize = 1024;

/// The fewest keys worth giving a split.
///
/// Four key tiles. Below that the per-split fixed cost — the query tile load,
/// the partial write, and the combine's extra column — outweighs the
/// parallelism. It matters most at the SHORT end: a local layer's decode step
/// reads a 512-key window, and at 512 keys a split floor of 512 would give the
/// whole layer eight cubes.
const MIN_KEYS_PER_SPLIT: usize = 128;

/// Whether a layer's shape is one this kernel handles.
///
/// Each condition is a real limit. An output dimension may end partway through
/// a plane; the packed launcher separately requires whole NVFP4 groups. `rows`
/// must divide into both the planes and the head groups because a cube's rows
/// are a `groups x (rows / groups)` rectangle spread over four planes; and the
/// shared tile must fit the 48 KiB a static `__shared__` gets.
pub fn applies(heads: usize, kv_heads: usize, head_dim: usize, rows: usize) -> bool {
    if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 || head_dim == 0 || rows == 0 {
        return false;
    }
    let groups = heads / kv_heads;
    head_dim <= 1024
        && rows % PLANES as usize == 0
        && rows % groups == 0
        && shared_floats(rows as u32, head_dim as u32) * core::mem::size_of::<f32>() <= 48 * 1024
}

/// Whether one head ends on an NVFP4 block boundary.
///
/// The dense reader has no such restriction. The staged packed-V reader owns
/// one complete block per unit, so allowing a partial terminal block would
/// make that unit cross into the next head row.
fn packed_head_dim_applies(head_dim: usize) -> bool {
    head_dim.is_multiple_of(super::fp4quant::GROUP)
}

/// The cube row count a DECODE step wants: the smallest multiple of [`PLANES`]
/// that also splits into whole head groups.
///
/// One query row across all `groups` heads of a KV head is what a decode step
/// actually has, and at the model's `groups = 4` that is exactly [`PLANES`].
/// Where `groups` does not divide the planes the tile carries query rows that
/// do not exist; they mask out and cost their share of a plane, which is the
/// price of keeping one kernel for both regimes.
pub fn decode_rows(groups: usize) -> usize {
    let mut r = PLANES as usize;
    while !r.is_multiple_of(groups) {
        r += PLANES as usize;
    }
    r
}

/// The cube row count a PREFILL wants: [`ROWS_PREFILL`], rounded up to whole
/// head groups.
pub fn prefill_rows(groups: usize) -> usize {
    let mut r = ROWS_PREFILL as usize;
    while !r.is_multiple_of(groups) {
        r += PLANES as usize;
    }
    r
}

/// How many queries a PREFILL block should hold, from what a block allocates.
///
/// A block allocates three things linear in its rows: the relative-bias table
/// `[rows, heads, eff]`, the partial output `[slots, rows, heads, head_dim]`
/// (one slot at prefill, where the grid is already full on the query axis), and
/// the output itself. That is `heads * (eff + 2 * head_dim)` f32 a row, and
/// none of it is a function of `tokens` — which is the whole point of the
/// change, because the arm this replaces held `heads * tokens` f32 a row and
/// therefore had to shrink `rows` as the sequence grew, until at 1M it could
/// afford eight.
///
/// The cap is 512 MiB, chosen against nothing better than "the same order as
/// the operands": at Inkling's global shape (`heads` 32, `eff` 1024, `head_dim`
/// 128) it comes out at 10,922 rows, and it is a ceiling rather than a target —
/// short sequences take one block.
pub fn query_block(heads: usize, eff: usize, head_dim: usize, tokens: usize) -> usize {
    const CAP: usize = 512 << 20;
    let per_row = heads * (eff + 2 * head_dim) * core::mem::size_of::<f32>();
    (CAP / per_row.max(1)).clamp(1, tokens.max(1))
}

/// The query rows a cube of `rows` holds, which is the grid's X granularity.
pub fn queries_per_cube(heads: usize, kv_heads: usize, rows: usize) -> usize {
    rows / (heads / kv_heads)
}

/// Fused global attention over one query block and any number of key runs.
///
/// `q` is `[nq, heads * head_dim]` f32 and `rel` is `[nq, heads, eff]` f32 —
/// the layouts the projections already produce, block-local, with `q0` giving
/// the absolute position of row 0. The key runs are `[rows, kv_heads *
/// head_dim]` in `kv` element type, token-major, exactly as both the projection
/// and the paged cache leave them.
///
/// Returns `[nq, heads * head_dim]` f32.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_launch<R: Runtime>(
    client: &ComputeClient<R>,
    q: &Handle,
    runs: &[KeyRun<'_>],
    rel: &Handle,
    kv: KvElem,
    nq: usize,
    q0: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eff: usize,
    window: Option<usize>,
    scaling: f32,
    rows: usize,
    gaps: Option<Gaps<'_>>,
) -> Handle {
    assert!(nq > 0, "an empty query block has no attention");
    assert!(!runs.is_empty(), "attention over no keys at all");
    assert!(
        applies(heads, kv_heads, head_dim, rows),
        "this shape is not fused here: {heads}/{kv_heads} heads, head_dim {head_dim}, rows {rows}"
    );
    let packed = kv == KvElem::Nvfp4;
    assert!(
        !packed || packed_head_dim_applies(head_dim),
        "NVFP4 needs head_dim {head_dim} to end on a {}-value block boundary",
        super::fp4quant::GROUP
    );
    let groups = heads / kv_heads;
    let bq = rows / groups;
    let q_tiles = nq.div_ceil(bq);
    let q_elems = nq * heads * head_dim;
    let rel_elems = (nq * heads * eff).max(1);
    assert!(
        q_elems <= u32::MAX as usize && rel_elems <= u32::MAX as usize,
        "{nq} queries x {heads} heads is past the 32-bit index every cubecl usize is on this \
         runtime"
    );

    // How far to split each run's key axis. Prefill fills the grid on the query
    // axis and lands on one split everywhere; a decode step has `q_tiles == 1`
    // and splits until the device is busy. Proportional to the run's span, so a
    // merged 1M-row page and a 128-row tail page do not get the same treatment.
    let spans: Vec<usize> = runs.iter().map(|r| r.hi.saturating_sub(r.lo)).collect();
    let split_of = key_run_splits(&spans, q_tiles, kv_heads);
    let slots: usize = split_of.iter().sum();

    let po_elems = slots * nq * heads * head_dim;
    let pml_elems = slots * nq * heads * 2;
    assert!(
        po_elems <= u32::MAX as usize,
        "{slots} partial slots x {nq} queries is past the 32-bit index"
    );
    let po = client.empty(po_elems * core::mem::size_of::<f32>());
    let pml = client.empty(pml_elems * core::mem::size_of::<f32>());

    // The gap table, or the relative-bias handle standing in for it: with no
    // gaps the kernel is compiled without the branch and never reads the
    // arrays, so any live handle serves as the binding.
    let gapped = gaps.is_some_and(|g| g.count > 0);
    let (gap_rows_h, gap_cums_h, gap_count) = match gaps {
        Some(g) if g.count > 0 => (g.rows.clone(), g.cums.clone(), g.count),
        _ => (rel.clone(), rel.clone(), 0usize),
    };
    let mut slot0 = 0usize;
    for (run, splits) in runs.iter().zip(split_of.iter().copied()) {
        let kv_elems = run.rows * kv_heads * head_dim;
        assert!(
            kv_elems <= u32::MAX as usize,
            "{} key rows x {kv_heads} x {head_dim} is past the 32-bit index",
            run.rows
        );
        assert!(
            run.lo <= run.hi && run.hi <= run.rows,
            "a key run's live range {}..{} is not inside its {} rows",
            run.lo,
            run.hi,
            run.rows
        );
        assert_eq!(
            packed,
            run.k_scales.is_some() && run.v_scales.is_some(),
            "NVFP4 needs both block-scale handles and no other element type may carry them"
        );
        // THE TWO READERS' EXTENTS. Exactly one pair is live, and the other is
        // bound to a live handle at length ONE. The kernel's comptime branch
        // never reads the dead pair, and a length-one binding of a buffer that
        // is certainly at least four bytes long is how you say "absent" to a
        // launcher that has no absent. Binding it at the LIVE extent instead
        // would be a length that is not true of those bytes, which is the kind
        // of lie a later reader believes.
        let (dense_elems, code_elems, scale_elems) = if packed {
            (
                1,
                kv_elems / super::fp4quant::CODES_PER_WORD,
                kv_elems / super::fp4quant::GROUP,
            )
        } else {
            (kv_elems, 1, 1)
        };
        let ks = run.k_scales.unwrap_or(run.k).clone();
        let vs = run.v_scales.unwrap_or(run.v).clone();
        let count = CubeCount::Static(q_tiles as u32, kv_heads as u32, splits as u32);
        let dim = CubeDim::new_1d(UNITS);
        macro_rules! go {
            ($e:ty) => {
                unsafe {
                    flash_kernel::launch_unchecked::<$e, R>(
                        client,
                        count.clone(),
                        dim,
                        ArrayArg::from_raw_parts(q.clone(), q_elems),
                        ArrayArg::from_raw_parts(run.k.clone(), dense_elems),
                        ArrayArg::from_raw_parts(run.v.clone(), dense_elems),
                        ArrayArg::from_raw_parts(run.k.clone(), code_elems),
                        ArrayArg::from_raw_parts(ks.clone(), scale_elems),
                        ArrayArg::from_raw_parts(run.v.clone(), code_elems),
                        ArrayArg::from_raw_parts(vs.clone(), scale_elems),
                        ArrayArg::from_raw_parts(rel.clone(), rel_elems),
                        // Bound whether or not the kernel reads them: an array a
                        // kernel never touches is a constant-bank pointer.
                        ArrayArg::from_raw_parts(gap_rows_h.clone(), gap_count.max(1)),
                        ArrayArg::from_raw_parts(gap_cums_h.clone(), gap_count.max(1)),
                        ArrayArg::from_raw_parts(po.clone(), po_elems),
                        ArrayArg::from_raw_parts(pml.clone(), pml_elems),
                        scaling,
                        nq as u32,
                        q0 as u32,
                        run.base as u32,
                        run.lo as u32,
                        run.hi as u32,
                        eff as u32,
                        window.unwrap_or(0) as u32,
                        slot0 as u32,
                        splits as u32,
                        slots as u32,
                        gap_count as u32,
                        run.row0 as u32,
                        heads as u32,
                        kv_heads as u32,
                        head_dim as u32,
                        rows as u32,
                        packed,
                        gapped,
                    )
                }
            };
        }
        match kv {
            KvElem::F32 => go!(f32),
            KvElem::Bf16 => go!(half::bf16),
            // The element type is dead on this arm — the reader is the packed
            // one — so it names the widest thing the launcher already knows
            // how to instantiate rather than adding a third instantiation.
            KvElem::Nvfp4 => go!(f32),
        };
        slot0 += splits;
    }

    let out = client.empty(q_elems * core::mem::size_of::<f32>());
    unsafe {
        flash_combine_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(nq as u32, heads as u32, 1),
            CubeDim::new_1d(head_dim as u32),
            ArrayArg::from_raw_parts(po, po_elems),
            ArrayArg::from_raw_parts(pml, pml_elems),
            ArrayArg::from_raw_parts(out.clone(), q_elems),
            slots as u32,
            nq as u32,
            heads as u32,
            head_dim as u32,
        )
    };
    out
}

/// The launcher's exact scratch geometry, also used by native admission.
/// Inputs are the live spans of already extent-checked runs, not buffer capacity.
pub(crate) fn key_run_splits(spans: &[usize], q_tiles: usize, kv_heads: usize) -> Vec<usize> {
    let total_span: usize = spans.iter().sum();
    let want = (TARGET_CUBES / (q_tiles * kv_heads)).max(1);
    spans.iter().map(|span| {
        let by_span = if total_span == 0 { 1 } else { (want * span).div_ceil(total_span) };
        by_span.min(span.div_ceil(MIN_KEYS_PER_SPLIT).max(1)).max(1)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_attention_defaults_to_the_legacy_reader_and_tile() {
        let policy = CachedAttentionPolicy::default();
        assert_eq!(policy, CachedAttentionPolicy::Legacy);
        assert_eq!(policy.as_str(), "legacy");
        assert!(!policy.packed_reader(128));
        for nq in [1, 2, 8, 90, 512] {
            assert_eq!(policy.tile_rows(16, 4, 128, nq), Some(4));
        }
    }

    #[test]
    fn packed_batched_policy_waits_for_a_full_query_tile() {
        let policy = CachedAttentionPolicy::PackedBatchedV1;
        assert_eq!(policy.as_str(), "packed-batched-v1");
        assert!(policy.packed_reader(128));
        for nq in [1, 2, 7] {
            assert_eq!(policy.tile_rows(16, 4, 128, nq), Some(4));
        }
        for nq in [8, 90, 512] {
            assert_eq!(policy.tile_rows(16, 4, 128, nq), Some(32));
        }
        // Non-Inkling GQA also uses queries-per-tile, not a hard-coded eight.
        assert_eq!(policy.tile_rows(24, 4, 128, 5), Some(12));
        assert_eq!(policy.tile_rows(24, 4, 128, 6), Some(36));
    }

    #[test]
    fn cached_policy_preserves_shape_and_packed_reader_fallbacks() {
        let policy = CachedAttentionPolicy::PackedBatchedV1;
        assert_eq!(policy.tile_rows(16, 4, 256, 512), Some(4));
        assert_eq!(policy.tile_rows(16, 4, 1024, 512), None);
        assert_eq!(policy.tile_rows(16, 4, 8, 512), Some(32));
        assert!(!policy.packed_reader(8));
        assert!(!policy.packed_reader(0));
        for policy in [CachedAttentionPolicy::Legacy, policy] {
            for (heads, kv_heads, head_dim, nq) in [
                (0, 4, 128, 1), (16, 0, 128, 1), (16, 3, 128, 1),
                (16, 4, 0, 1), (16, 4, 128, 0),
            ] {
                assert_eq!(policy.tile_rows(heads, kv_heads, head_dim, nq), None);
            }
        }
    }

    #[test]
    fn inklings_global_shape_is_fused() {
        // 32 heads over 8 KV heads, head_dim 128: groups 4, so a 32-row cube is
        // eight consecutive queries across all four heads of a KV head.
        assert!(applies(32, 8, 128, prefill_rows(4)));
        assert!(applies(32, 8, 128, decode_rows(4)));
        assert_eq!(prefill_rows(4), 32);
        assert_eq!(decode_rows(4), 4);
        assert_eq!(queries_per_cube(32, 8, 32), 8);
        assert_eq!(queries_per_cube(32, 8, 4), 1);
    }

    #[test]
    fn the_row_counts_always_split_into_planes_and_groups() {
        for groups in [1usize, 2, 4, 8, 16] {
            for rows in [decode_rows(groups), prefill_rows(groups)] {
                assert_eq!(rows % PLANES as usize, 0, "groups {groups}, rows {rows}");
                assert_eq!(rows % groups, 0, "groups {groups}, rows {rows}");
            }
        }
    }

    #[test]
    fn refuses_what_it_cannot_tile() {
        // Rows that do not fill four planes: three of them would idle and the
        // fourth would own rows that do not exist.
        assert!(!applies(32, 8, 128, 6));
        // Rows that do not split into head groups. At 32/8 every multiple of
        // the planes is also a multiple of `groups`, so this needs a geometry
        // where the two do not divide each other: 24 heads over 4 KV heads is
        // `groups = 6`, and a four-row cube cannot hold whole groups of six.
        assert!(!applies(24, 4, 128, 4));
        // Heads that do not divide into KV heads: `h / groups` would alias.
        assert!(!applies(32, 7, 128, 32));
        // A head_dim whose tile does not fit the 48 KiB a static `__shared__`
        // gets.
        assert!(!applies(32, 8, 2048, 32));
    }

    #[test]
    fn packed_heads_end_on_block_boundaries_but_dense_heads_need_not() {
        assert!(packed_head_dim_applies(128));
        assert!(packed_head_dim_applies(
            crate::models::inkling::fp4quant::GROUP
        ));
        assert!(!packed_head_dim_applies(8));
        assert!(applies(32, 8, 8, 4));
    }

    #[test]
    fn the_shared_tile_fits_the_static_ceiling() {
        for (rows, head_dim) in [(32u32, 128u32), (4, 128), (32, 64)] {
            let bytes = shared_floats(rows, head_dim) * 4;
            assert!(bytes <= 48 * 1024, "{rows}x{head_dim} wants {bytes} bytes");
        }
    }

    /// The block a prefill takes never depends on the sequence length, which is
    /// the whole difference from the arm it replaces.
    #[test]
    fn the_query_block_does_not_shrink_with_the_sequence() {
        let a = query_block(32, 1024, 128, 1 << 20);
        let b = query_block(32, 1024, 128, 1 << 14);
        assert_eq!(a, b.min(a), "the cap must not move with `tokens`");
        assert!(a > 1024, "a 512 MiB cap at Inkling's shape is {a} rows");
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use cubecl::cuda::CudaRuntime;

    /// Deterministic filler, the same one [`super::super::banded`]'s device
    /// tests use: a fixed pattern rather than a seeded RNG, so a failure is
    /// reproducible from the source alone.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7919 + seed).sin() * 0.5 + (i as f32 * 0.1237).cos() * 0.25)
            .collect()
    }

    struct Shape {
        nq: usize,
        q0: usize,
        keys: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eff: usize,
        window: Option<usize>,
    }

    /// Run the fused kernel over `cuts` contiguous runs of the key axis.
    ///
    /// `cuts` is what makes this harness worth having: the answer must not
    /// depend on where the key axis is chopped, because the online softmax is
    /// exactly the claim that it does not.
    fn run(sh: &Shape, q: &[f32], k: &[f32], v: &[f32], rel: &[f32], cuts: &[usize]) -> Vec<f32> {
        let kv_row = sh.kv_heads * sh.head_dim;
        let groups = sh.heads / sh.kv_heads;
        let rows = if sh.nq == 1 {
            decode_rows(groups)
        } else {
            prefill_rows(groups)
        };
        let client = <CudaRuntime as Runtime>::client(&Default::default());
        let qh = client.create_from_slice(f32::as_bytes(q));
        let rh = client.create_from_slice(f32::as_bytes(rel));
        // Each cut is uploaded as its OWN buffer, which is what a paged cache
        // hands over and what a single prefill projection does not.
        let mut bounds = vec![0usize];
        bounds.extend_from_slice(cuts);
        bounds.push(sh.keys);
        let held: Vec<(cubecl::server::Handle, cubecl::server::Handle, usize, usize)> = bounds
            .windows(2)
            .map(|w| {
                let (lo, hi) = (w[0], w[1]);
                let kh = client.create_from_slice(f32::as_bytes(&k[lo * kv_row..hi * kv_row]));
                let vh = client.create_from_slice(f32::as_bytes(&v[lo * kv_row..hi * kv_row]));
                (kh, vh, hi - lo, lo)
            })
            .collect();
        let runs: Vec<KeyRun<'_>> = held
            .iter()
            .map(|(kh, vh, rows, base)| KeyRun {
                k: kh,
                v: vh,
                k_scales: None,
                v_scales: None,
                rows: *rows,
                base: *base,
                lo: 0,
                hi: *rows,
                row0: 0,
            })
            .collect();
        let oh = flash_attention_launch(
            &client,
            &qh,
            &runs,
            &rh,
            KvElem::F32,
            sh.nq,
            sh.q0,
            sh.heads,
            sh.kv_heads,
            sh.head_dim,
            sh.eff,
            sh.window,
            1.0 / sh.head_dim as f32,
            rows,
            None,
        );
        f32::from_bytes(&client.read_one(oh).expect("read the fused output")).to_vec()
    }

    fn inputs(sh: &Shape) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            fill(sh.nq * sh.heads * sh.head_dim, 0.1),
            fill(sh.keys * sh.kv_heads * sh.head_dim, 0.3),
            fill(sh.keys * sh.kv_heads * sh.head_dim, 0.5),
            fill(sh.nq * sh.heads * sh.eff, 0.7),
        )
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "comparison lengths differ");
        a.iter()
            .zip(b)
            .enumerate()
            .map(|(i, (x, y))| {
                assert!(x.is_finite() && y.is_finite(),
                    "non-finite comparison at {i}: {x} versus {y}");
                (x - y).abs()
            })
            .fold(0f32, f32::max)
    }

    #[test]
    #[should_panic(expected = "comparison lengths differ")]
    fn worst_rejects_length_mismatch() {
        worst(&[0.0], &[0.0, 1.0]);
    }

    #[test]
    fn worst_rejects_non_finite_values_on_either_side() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(|| worst(&[0.0, bad], &[0.0, 1.0])).is_err());
            assert!(std::panic::catch_unwind(|| worst(&[0.0, 1.0], &[0.0, bad])).is_err());
        }
    }

    /// A reserved read may include dead rows up to the next 512-row epoch.
    /// Compare the production BF16 reader with a tightly sized live buffer,
    /// keeping the key partition identical so exact equality is meaningful.
    /// This isolates masking, not whole-model fixed-versus-growing numerics.
    #[test]
    #[ignore = "requires an explicitly reserved CUDA device; bounded BF16 reserved-tail diagnostic"]
    fn bf16_reserved_tail_is_excluded_at_live_hi() {
        use half::bf16;

        let client = <CudaRuntime as Runtime>::client(&Default::default());
        // (query rows, live keys, first query position). The first two are
        // normal cached batches: a full 512 rows and a short final 90 rows.
        // The last queries a prefix without appending the query itself: the
        // poisoned rows precede it, so causality cannot hide a broken `hi`.
        for (nq, keys, q0) in [(512usize, 641usize, 129usize), (90, 731, 641), (1, 641, 1023)] {
            let sh = Shape {
                nq,
                q0,
                keys,
                heads: 16,
                kv_heads: 4,
                head_dim: 128,
                eff: 17,
                window: None,
            };
            let capacity = keys.next_multiple_of(512);
            assert!(keys < capacity && keys % PLANE as usize != 0);
            let kv_row = sh.kv_heads * sh.head_dim;
            let (q, k, v, rel) = inputs(&sh);
            let qh = client.create_from_slice(f32::as_bytes(&q));
            let rh = client.create_from_slice(f32::as_bytes(&rel));
            let mut k: Vec<bf16> = k.into_iter().map(bf16::from_f32).collect();
            let mut v: Vec<bf16> = v.into_iter().map(bf16::from_f32).collect();
            let live_k = client.create_from_slice(bf16::as_bytes(&k));
            let live_v = client.create_from_slice(bf16::as_bytes(&v));
            // NaNs, rather than zeros: even a masked 0 * V load from the tail
            // must be visible to the finite-output check below.
            k.resize(capacity * kv_row, bf16::from_f32(f32::NAN));
            v.resize(capacity * kv_row, bf16::from_f32(f32::NAN));
            let padded_k = client.create_from_slice(bf16::as_bytes(&k));
            let padded_v = client.create_from_slice(bf16::as_bytes(&v));
            let launch = |kh: &Handle, vh: &Handle, physical_rows: usize| {
                let run = KeyRun {
                    k: kh,
                    v: vh,
                    k_scales: None,
                    v_scales: None,
                    rows: physical_rows,
                    base: 0,
                    lo: 0,
                    hi: keys,
                    row0: 0,
                };
                let out = flash_attention_launch(
                    &client,
                    &qh,
                    &[run],
                    &rh,
                    KvElem::Bf16,
                    sh.nq,
                    sh.q0,
                    sh.heads,
                    sh.kv_heads,
                    sh.head_dim,
                    sh.eff,
                    None,
                    1.0 / sh.head_dim as f32,
                    // flash_cached uses the decode tile even for nq > 1.
                    decode_rows(sh.heads / sh.kv_heads),
                    None,
                );
                f32::from_bytes(&client.read_one(out).expect("read reserved-tail output")).to_vec()
            };
            let expected = launch(&live_k, &live_v, keys);
            let actual = launch(&padded_k, &padded_v, capacity);
            assert_eq!(expected.len(), nq * sh.heads * sh.head_dim);
            assert!(expected.iter().any(|x| x.abs() > 1e-6), "the reference computed nothing");
            assert_eq!(worst(&actual, &expected), 0.0,
                "{nq} queries at {q0}, {keys} live keys in {capacity} physical rows");
        }
    }

    /// NVFP4 code words and E4M3 block-scale bytes for `n` elements, in the
    /// layout [`super::super::kvpages::Fp4Rows`] stores: code `i` in word
    /// `i / 8` at nibble `i % 8`, one scale per sixteen consecutive elements.
    ///
    /// Built here rather than by running `quantize_nvfp4`, because a round
    /// trip through the quantizer would agree with the reader about a layout
    /// they had both got wrong. The scale bytes are deliberately ones with a
    /// NONZERO mantissa — `2^(e-7) * (1 + m/8)` for `e` in 6..9 — so the test
    /// below exercises the claim it rests on rather than only powers of two.
    fn packed_pair(n: usize, seed: usize) -> (Vec<u32>, Vec<u8>) {
        assert_eq!(n % 16, 0, "{n} elements is not whole NVFP4 blocks");
        const SCALES: [u8; 6] = [0x31, 0x35, 0x39, 0x3D, 0x41, 0x45];
        let mut words = vec![0u32; n / 8];
        for i in 0..n {
            words[i / 8] |= (((i * 7 + seed * 5) % 16) as u32) << (4 * (i % 8));
        }
        let scales = (0..n / 16)
            .map(|b| SCALES[(b * 3 + seed) % SCALES.len()])
            .collect();
        (words, scales)
    }

    fn packed_rows(sh: &Shape, seed: usize) -> (Vec<u32>, Vec<u8>) {
        let kv_row = sh.kv_heads * sh.head_dim;
        let mut codes = Vec::with_capacity(sh.keys * kv_row / 8);
        let mut scales = Vec::with_capacity(sh.keys * kv_row / 16);
        for row in 0..sh.keys {
            for head in 0..sh.kv_heads {
                // Distinguish rows and heads, not just feature nibbles.
                let (c, s) = packed_pair(sh.head_dim, seed + row * 5 + head * 3);
                codes.extend(c);
                scales.extend(s);
            }
        }
        (codes, scales)
    }

    /// The same shape, key-axis cuts, live range and tile, read BOTH ways.
    ///
    /// The dense arm is production's, exactly: each cut goes through
    /// [`super::super::fp4quant::dequantize_nvfp4_bf16`] into a BF16 page and
    /// the kernel consumes that. The packed arm hands the kernel the very same
    /// codes and scales. Nothing is decoded on the host, so nothing here can
    /// be wrong about what a byte means without the device being wrong the
    /// same way.
    #[allow(clippy::too_many_arguments)]
    fn run_both_readers(
        sh: &Shape,
        q: &[f32],
        k: &(Vec<u32>, Vec<u8>),
        v: &(Vec<u32>, Vec<u8>),
        rel: &[f32],
        cuts: &[usize],
        live: std::ops::Range<usize>,
        tile_rows: usize,
        gaps: &[(u32, u32)],
    ) -> (Vec<f32>, Vec<f32>) {
        assert!(live.start <= live.end && live.end <= sh.keys);
        assert!(gaps.is_empty() || (live.start == 0 && sh.window.is_none()),
            "only global caches with no dropped prefix carry gaps");
        let kv_row = sh.kv_heads * sh.head_dim;
        let client = <CudaRuntime as Runtime>::client(&Default::default());
        let qh = client.create_from_slice(f32::as_bytes(q));
        let rh = client.create_from_slice(f32::as_bytes(rel));
        let gap_handles = if gaps.is_empty() {
            None
        } else {
            let rows: Vec<u32> = gaps.iter().map(|g| g.0).collect();
            let cums: Vec<u32> = gaps.iter().map(|g| g.1).collect();
            Some((
                client.create_from_slice(u32::as_bytes(&rows)),
                client.create_from_slice(u32::as_bytes(&cums)),
            ))
        };
        let mut bounds = vec![0usize];
        bounds.extend_from_slice(cuts);
        bounds.push(sh.keys);
        // (k codes, k scales, v codes, v scales, dense k, dense v, rows, base)
        let held: Vec<(Handle, Handle, Handle, Handle, Handle, Handle, usize, usize)> = bounds
            .windows(2)
            .map(|w| {
                let (lo, hi) = (w[0], w[1]);
                let (cw, sw) = (kv_row / 8, kv_row / 16);
                let up = |c: &Vec<u32>, s: &Vec<u8>| {
                    (
                        client.create_from_slice(u32::as_bytes(&c[lo * cw..hi * cw])),
                        client.create_from_slice(&s[lo * sw..hi * sw]),
                    )
                };
                let (kc, ks) = up(&k.0, &k.1);
                let (vc, vs) = up(&v.0, &v.1);
                let deq = |c: &Handle, s: &Handle| {
                    super::super::fp4quant::dequantize_nvfp4_bf16(&client, c, s, hi - lo, kv_row)
                };
                let (kd, vd) = (deq(&kc, &ks), deq(&vc, &vs));
                (kc, ks, vc, vs, kd, vd, hi - lo, lo)
            })
            .collect();
        let launch = |runs: &[KeyRun<'_>], elem: KvElem| {
            let oh = flash_attention_launch(
                &client,
                &qh,
                runs,
                &rh,
                elem,
                sh.nq,
                sh.q0,
                sh.heads,
                sh.kv_heads,
                sh.head_dim,
                sh.eff,
                sh.window,
                1.0 / sh.head_dim as f32,
                tile_rows,
                gap_handles.as_ref().map(|(rows, cums)| Gaps {
                    rows,
                    cums,
                    count: gaps.len(),
                }),
            );
            f32::from_bytes(&client.read_one(oh).expect("read the fused output")).to_vec()
        };
        let dense: Vec<KeyRun<'_>> = held
            .iter()
            .map(|(_, _, _, _, kd, vd, rows, base)| KeyRun {
                k: kd,
                v: vd,
                k_scales: None,
                v_scales: None,
                rows: *rows,
                base: *base,
                lo: live.start.saturating_sub(*base).min(*rows),
                hi: live.end.saturating_sub(*base).min(*rows),
                row0: base.saturating_sub(live.start),
            })
            .collect();
        let packed: Vec<KeyRun<'_>> = held
            .iter()
            .map(|(kc, ks, vc, vs, _, _, rows, base)| KeyRun {
                k: kc,
                v: vc,
                k_scales: Some(ks),
                v_scales: Some(vs),
                rows: *rows,
                base: *base,
                lo: live.start.saturating_sub(*base).min(*rows),
                hi: live.end.saturating_sub(*base).min(*rows),
                row0: base.saturating_sub(live.start),
            })
            .collect();
        (launch(&dense, KvElem::Bf16), launch(&packed, KvElem::Nvfp4))
    }

    /// The packed reader is the dequantising reader to the BIT, and that is a
    /// property of NVFP4 rather than luck.
    ///
    /// An E2M1 magnitude carries at most two mantissa bits (`1.1b`, i.e. 1.5
    /// and 3 and 6) and an E4M3 scale at most four significant bits, so their
    /// product needs at most six — and BF16 has eight. The dequantised page is
    /// therefore EXACT in BF16, and reading the codes directly reconstructs the
    /// same f32 the BF16 page widens to. There is no rounding to trade away
    /// here, which is why this asserts equality and not a tolerance.
    ///
    /// A tolerance would also be the wrong instrument for what actually breaks:
    /// every way of getting NVFP4 indexing wrong — the nibble order inside a
    /// word, which sixteen features share a scale, scales indexed per row
    /// instead of per block — is silent. Each produces finite, plausible
    /// attention over the wrong numbers, and each moves the answer far more
    /// than any tolerance a numerical argument would justify.
    #[test]
    fn the_packed_reader_is_the_dequantising_reader_to_the_bit() {
        for (nq, keys, window) in [
            (1usize, 700usize, None),
            (1, 33, None),
            (16, 200, Some(64usize)),
            (37, 91, None),
        ] {
            let sh = Shape {
                nq,
                q0: keys - nq,
                keys,
                heads: 8,
                kv_heads: 2,
                head_dim: 32,
                eff: 5,
                window,
            };
            let n = sh.keys * sh.kv_heads * sh.head_dim;
            let k = packed_pair(n, 1);
            let v = packed_pair(n, 2);
            let q = fill(sh.nq * sh.heads * sh.head_dim, 0.1);
            let rel = fill(sh.nq * sh.heads * sh.eff, 0.7);
            let groups = sh.heads / sh.kv_heads;
            let tile_rows = if nq == 1 {
                decode_rows(groups)
            } else {
                prefill_rows(groups)
            };
            for cuts in [vec![], vec![keys / 2], vec![7, keys / 3, keys - 5]] {
                let (dense, packed) =
                    run_both_readers(&sh, &q, &k, &v, &rel, &cuts, 0..keys, tile_rows, &[]);
                assert!(
                    dense.iter().any(|x| x.abs() > 1e-6),
                    "nq {nq}, {keys} keys: the dense arm computed nothing to compare against"
                );
                assert_eq!(
                    dense,
                    packed,
                    "nq {nq}, {keys} keys, cuts {cuts:?}: worst |delta| {}",
                    worst(&dense, &packed)
                );
            }
        }
    }

    /// The cached production call uses four rows per tile even for 512 query
    /// rows; also gate the larger candidate tiles. Compare the readers only
    /// within the same tile, with identical partitions and live bounds;
    /// excluded NVFP4 rows contain NaN scales so accidental contributions cannot
    /// quietly pass as finite agreement.
    /// This tests the reader seam, not quantization or whole-Session numerics.
    #[test]
    #[ignore = "requires an explicitly reserved CUDA device; bounded packed reserved-range diagnostic"]
    fn packed_reserved_reads_match_bf16_at_production_tile() {
        // (query rows, first query position, live start, live end, window).
        // The prefix-only query at 1023 independently exercises both bounds:
        // neither causality nor a local window can hide the poisoned rows.
        for (nq, q0, lo, hi, window) in [
            (512usize, 129usize, 0usize, 641usize, None),
            (90, 641, 0, 731, None),
            (1, 640, 0, 641, None),
            (1, 1023, 129, 641, None),
            (90, 641, 129, 731, Some(512usize)),
        ] {
            let sh = Shape {
                nq,
                q0,
                keys: 1024,
                heads: 16,
                kv_heads: 4,
                head_dim: 128,
                eff: 17,
                window,
            };
            let kv_row = sh.kv_heads * sh.head_dim;
            assert_eq!(decode_rows(sh.heads / sh.kv_heads), 4);
            assert!(hi < sh.keys && hi % PLANE as usize != 0);
            let pack = |seed: usize| {
                let (mut codes, mut scales) = packed_rows(&sh, seed);
                for row in (0..lo).chain(hi..sh.keys) {
                    codes[row * kv_row / 8..(row + 1) * kv_row / 8].fill(0x7777_7777);
                    // E4M3 0x7f is NaN; the finite code 7 decodes to +6.
                    scales[row * kv_row / 16..(row + 1) * kv_row / 16].fill(0x7f);
                }
                (codes, scales)
            };
            let k = pack(1);
            let v = pack(2);
            let q = fill(sh.nq * sh.heads * sh.head_dim, 0.1);
            let rel = fill(sh.nq * sh.heads * sh.eff, 0.7);
            // The cuts are not tile-aligned; 900..1024 is entirely dead,
            // and with lo=129, 0..7 is dead and 7..133 has a partial head.
            // Equality is only between readers with the SAME cuts: changing
            // partitions changes floating-point reduction order.
            for cuts in [vec![], vec![7, 133, 700, 900]] {
                for tile_rows in [4usize, 8, 16, 32] {
                    let (dense, packed) = run_both_readers(
                        &sh, &q, &k, &v, &rel, &cuts, lo..hi, tile_rows, &[],
                    );
                    assert_eq!(dense.len(), nq * sh.heads * sh.head_dim);
                    assert!(dense.iter().any(|x| x.abs() > 1e-6),
                        "nq {nq}, q0 {q0}, live {lo}..{hi}: the reference computed nothing");
                    assert_eq!(worst(&dense, &packed), 0.0,
                        "tile {tile_rows}, nq {nq}, q0 {q0}, live {lo}..{hi}, \
                         window {window:?}, cuts {cuts:?}");
                }
            }
        }
    }

    /// Two-pass scalar softmax with explicit absolute key positions. This
    /// shares neither the device's tiling/split reduction nor its mask bounds.
    /// The f64 arithmetic checks the kernel algorithm on fixed f32 operands;
    /// it is not a claim that a higher-precision model is ground truth.
    fn cpu_attention(
        sh: &Shape,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rel: &[f32],
        key_positions: &[usize],
    ) -> Vec<f64> {
        let q_row = sh.heads * sh.head_dim;
        let kv_row = sh.kv_heads * sh.head_dim;
        assert_eq!(q.len(), sh.nq * q_row);
        assert_eq!(k.len(), sh.keys * kv_row);
        assert_eq!(v.len(), k.len());
        assert_eq!(rel.len(), sh.nq * sh.heads * sh.eff);
        assert_eq!(key_positions.len(), sh.keys);
        let mut out = vec![0.0; q.len()];
        for qi in 0..sh.nq {
            let q_abs = sh.q0 + qi;
            for h in 0..sh.heads {
                let qoff = qi * q_row + h * sh.head_dim;
                let kvh = h / (sh.heads / sh.kv_heads);
                let mut scores = Vec::new();
                for (ki, &k_abs) in key_positions.iter().enumerate() {
                    if k_abs > q_abs || sh.window.is_some_and(|w| q_abs - k_abs >= w) {
                        continue;
                    }
                    let koff = ki * kv_row + kvh * sh.head_dim;
                    let dot: f64 = (0..sh.head_dim)
                        .map(|d| f64::from(q[qoff + d]) * f64::from(k[koff + d]))
                        .sum();
                    let dist = q_abs - k_abs;
                    let bias = if dist < sh.eff {
                        f64::from(rel[(qi * sh.heads + h) * sh.eff + dist])
                    } else {
                        0.0
                    };
                    scores.push((koff, dot / sh.head_dim as f64 + bias));
                }
                assert!(!scores.is_empty(), "fixture query has no visible key");
                let max = scores.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
                let denom: f64 = scores.iter().map(|(_, s)| (s - max).exp()).sum();
                for (koff, score) in scores {
                    let p = (score - max).exp() / denom;
                    for d in 0..sh.head_dim {
                        out[qoff + d] += p * f64::from(v[koff + d]);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn cpu_attention_oracle_uses_absolute_positions_and_window_edges() {
        // Zero logits make the answer the visible values' arithmetic mean.
        // Positions, rather than stored row numbers, decide visibility.
        for (window, expected) in [
            (None, [5.5, 5.5, 37.0]),
            (Some(3usize), [10.0, 10.0, 100.0]),
        ] {
            let sh = Shape {
                nq: 3,
                q0: 3,
                keys: 4,
                heads: 1,
                kv_heads: 1,
                head_dim: 1,
                eff: 1,
                window,
            };
            let actual = cpu_attention(
                &sh, &[0.0; 3], &[0.0; 4], &[1.0, 10.0, 100.0, 1000.0],
                &[0.0; 3], &[0, 2, 5, 9],
            );
            assert_eq!(actual.len(), expected.len());
            for (got, want) in actual.into_iter().zip(expected) {
                assert!((got - want).abs() < 1e-12, "{got} versus {want}, window {window:?}");
            }
        }
    }

    #[test]
    #[ignore = "requires an explicitly reserved CUDA device; bounded cross-tile CPU-oracle diagnostic"]
    fn cached_tiles_match_cpu_with_ragged_queries_windows_and_gaps() {
        // These are separate production cases: only global caches have gaps.
        // nq=37 leaves a ragged last query tile for rows 8, 16 and 32 at GQA4.
        // With one 897-key run, rows=4 picks six key splits and rows>=8
        // picks eight, so the oracle also covers a changed reduction partition.
        for (window, gaps) in [
            (None, vec![]),
            (Some(19usize), vec![]),
            (None, vec![(17u32, 5u32), (853, 19)]),
        ] {
            let sh = Shape {
                nq: 37,
                q0: 847,
                keys: 897,
                heads: 16,
                kv_heads: 4,
                head_dim: 32,
                eff: 13,
                window,
            };
            let k = packed_rows(&sh, 1);
            let v = packed_rows(&sh, 2);
            // The host codec uses the explicit E2M1 value ladder and E4M3
            // exponent/mantissa decode, not either GPU reader's arithmetic.
            let decode = |packed: &(Vec<u32>, Vec<u8>)| {
                let bytes: Vec<u8> = packed.0.iter().flat_map(|word| word.to_le_bytes()).collect();
                let mut values = vec![0.0; packed.0.len() * 8];
                let count = super::super::nvfp4::decode_row(&bytes, &packed.1, 1.0, &mut values);
                assert_eq!(count, values.len());
                assert!(values.iter().all(|x| x.is_finite() && x.abs() <= 20.0));
                values
            };
            let kd = decode(&k);
            let vd = decode(&v);
            let q = fill(sh.nq * sh.heads * sh.head_dim, 0.1);
            let rel = fill(sh.nq * sh.heads * sh.eff, 0.7);
            let key_positions: Vec<usize> = (0..sh.keys)
                .map(|row| row + gaps.iter().rev()
                    .find(|(at, _)| *at as usize <= row)
                    .map_or(0, |(_, skipped)| *skipped as usize))
                .collect();
            let expected = cpu_attention(&sh, &q, &kd, &vd, &rel, &key_positions);
            assert!(expected.iter().any(|x| x.abs() > 1e-3));
            for cuts in [vec![], vec![7, 519, 859]] {
                for tile_rows in [4usize, 8, 16, 32] {
                    let (dense, packed) = run_both_readers(
                        &sh, &q, &k, &v, &rel, &cuts, 0..sh.keys, tile_rows, &gaps,
                    );
                    assert_eq!(worst(&dense, &packed), 0.0,
                        "reader mismatch: tile {tile_rows}, window {window:?}, gaps {gaps:?}");
                    assert_eq!(dense.len(), expected.len());
                    let (mut max_abs, mut max_scaled) = (0.0f64, 0.0f64);
                    for (&actual, &want) in dense.iter().zip(&expected) {
                        assert!(actual.is_finite() && want.is_finite());
                        let error = (f64::from(actual) - want).abs();
                        max_abs = max_abs.max(error);
                        max_scaled = max_scaled.max(error / want.abs().max(1.0));
                    }
                    eprintln!(
                        "cached tile {tile_rows}, window {window:?}, gaps {gaps:?}, \
                         cuts {cuts:?}: max_abs={max_abs:e} max_scaled={max_scaled:e}"
                    );
                    // Tile width changes query grouping, split counts and
                    // f32 reduction order. Allow ordinary dot/exp/softmax
                    // rounding, not quantization error: these exact decoded
                    // operands have |K|, |V| <= 20 and only 897 keys. The
                    // absolute cap also bounds error at large reference values;
                    // scaled error uses a floor of 1 to handle zero crossings.
                    assert!(max_abs <= 5e-5 && max_scaled <= 1e-5,
                        "CPU mismatch: tile {tile_rows}, window {window:?}, gaps {gaps:?}, \
                         cuts {cuts:?}: max_abs={max_abs:e} max_scaled={max_scaled:e}");
                }
            }
        }
    }

    /// THE defining property of an online softmax, and the one a two-pass
    /// implementation cannot fail: where the key axis is cut must not change
    /// the answer.
    ///
    /// Every rescale bug lives here. A missing `exp(m_old - m_new)` on the
    /// accumulator, a running sum that is not rescaled with it, a combine that
    /// weights by `l` instead of by `exp(m - M) * l` — none of them shows on a
    /// single tile, all of them show the moment a later tile pushes the max up.
    /// The cuts below are deliberately not multiples of the 32-key tile, so the
    /// split boundaries land mid-tile as a page boundary would.
    #[test]
    fn where_the_key_axis_is_cut_does_not_move_the_answer() {
        for (nq, keys) in [(1usize, 700usize), (16, 200), (37, 91), (1, 33)] {
            let sh = Shape {
                nq,
                q0: keys - nq,
                keys,
                heads: 4,
                kv_heads: 2,
                head_dim: 8,
                eff: 5,
                window: None,
            };
            let (q, k, v, rel) = inputs(&sh);
            let whole = run(&sh, &q, &k, &v, &rel, &[]);
            for cuts in [
                vec![keys / 2],
                vec![1],
                vec![keys - 1],
                vec![7, keys / 3, keys / 2, keys - 5],
            ] {
                let cut = run(&sh, &q, &k, &v, &rel, &cuts);
                let w = worst(&whole, &cut);
                assert!(
                    w < 2e-6,
                    "{nq} queries over {keys} keys, cuts {cuts:?}: {w:e}"
                );
            }
        }
    }

    /// Causality, checked against the kernel itself rather than against a
    /// second implementation of it.
    ///
    /// A key AFTER the query cannot reach it, so perturbing that key must leave
    /// the query's output exactly where it was; a key at or before it must move
    /// it. Both halves are necessary: the first alone passes for a kernel that
    /// reads nothing, the second alone for one with no mask at all.
    #[test]
    fn a_key_after_the_query_cannot_reach_it() {
        let sh = Shape {
            nq: 24,
            q0: 0,
            keys: 40,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            eff: 6,
            window: None,
        };
        let (q, k, v, rel) = inputs(&sh);
        let base = run(&sh, &q, &k, &v, &rel, &[]);
        let kv_row = sh.kv_heads * sh.head_dim;
        let last = sh.nq - 1;
        let row = |o: &[f32], i: usize| {
            o[i * sh.heads * sh.head_dim..(i + 1) * sh.heads * sh.head_dim].to_vec()
        };
        for (j, must_move) in [(last + 1, false), (sh.keys - 1, false), (last, true)] {
            let (mut kk, mut vv) = (k.clone(), v.clone());
            for c in 0..kv_row {
                kk[j * kv_row + c] += 0.75;
                vv[j * kv_row + c] -= 0.75;
            }
            let got = run(&sh, &q, &kk, &vv, &rel, &[]);
            let moved = row(&got, last)
                .into_iter()
                .zip(row(&base, last))
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if must_move {
                assert!(
                    moved > 1e-4,
                    "key {j} is visible to query {last} and did not move"
                );
            } else {
                assert_eq!(
                    moved, 0.0,
                    "key {j} is after query {last} but perturbing it moved the answer by {moved:e}"
                );
            }
        }
    }

    /// GQA indexing, again against the kernel itself: perturbing KV head `c`
    /// may move only the query heads in `c`'s group.
    ///
    /// This is the failure a reference implementation catches by accident and
    /// this catches on purpose — a cube whose row-to-head map is off by one is
    /// a wrong ANSWER everywhere and a wrong SHAPE nowhere. It is the one thing
    /// this kernel does that the arm it replaces did not: the rows of a cube
    /// span `groups` heads, so the map from a plane's row index to a head is
    /// arithmetic that can be wrong.
    #[test]
    fn a_kv_head_reaches_only_its_own_group() {
        let sh = Shape {
            nq: 20,
            q0: 0,
            keys: 20,
            heads: 8,
            kv_heads: 2,
            head_dim: 8,
            eff: 6,
            window: None,
        };
        let groups = sh.heads / sh.kv_heads;
        let (q, k, v, rel) = inputs(&sh);
        let base = run(&sh, &q, &k, &v, &rel, &[]);
        let kv_row = sh.kv_heads * sh.head_dim;
        // ONE key, not every key of the head: adding the same constant to all
        // of a head's keys shifts every score equally and the softmax cancels
        // it exactly.
        let at = sh.keys / 2;
        for c in 0..sh.kv_heads {
            let (mut kk, mut vv) = (k.clone(), v.clone());
            for d in 0..sh.head_dim {
                kk[at * kv_row + c * sh.head_dim + d] += 0.5;
                vv[at * kv_row + c * sh.head_dim + d] -= 0.5;
            }
            let got = run(&sh, &q, &kk, &vv, &rel, &[]);
            for h in 0..sh.heads {
                let mine = h / groups == c;
                let moved = (0..sh.nq)
                    .flat_map(|i| {
                        (0..sh.head_dim)
                            .map(move |d| i * sh.heads * sh.head_dim + h * sh.head_dim + d)
                    })
                    .map(|o| (got[o] - base[o]).abs())
                    .fold(0f32, f32::max);
                if mine {
                    assert!(moved > 1e-5, "head {h} shares KV head {c} and did not move");
                } else {
                    assert_eq!(
                        moved, 0.0,
                        "head {h} does not share KV head {c} but moved by {moved:e}"
                    );
                }
            }
        }
    }

    /// The window, when one is asked for: a key `window` or more positions back
    /// is outside, and one inside must move the answer.
    ///
    /// The global layers pass `None` here, but the kernel takes a window
    /// because the DECODE lane runs every layer through it, local ones
    /// included, and a window that is applied but not tested is a window that
    /// is wrong on thirty-five layers of forty-two.
    #[test]
    fn the_window_is_exactly_what_the_query_can_reach() {
        let win = 9usize;
        let sh = Shape {
            nq: 30,
            q0: 0,
            keys: 30,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            eff: 5,
            window: Some(win),
        };
        let (q, k, v, rel) = inputs(&sh);
        let base = run(&sh, &q, &k, &v, &rel, &[]);
        let kv_row = sh.kv_heads * sh.head_dim;
        let last = sh.nq - 1;
        for (j, must_move) in [(last - win, false), (last - win + 1, true)] {
            let (mut kk, mut vv) = (k.clone(), v.clone());
            for c in 0..kv_row {
                kk[j * kv_row + c] += 0.75;
                vv[j * kv_row + c] -= 0.75;
            }
            let got = run(&sh, &q, &kk, &vv, &rel, &[]);
            let moved = (0..sh.heads * sh.head_dim)
                .map(|c| {
                    let o = last * sh.heads * sh.head_dim + c;
                    (got[o] - base[o]).abs()
                })
                .fold(0f32, f32::max);
            if must_move {
                assert!(
                    moved > 1e-4,
                    "key {j} is inside query {last}'s window and did not move"
                );
            } else {
                assert_eq!(
                    moved, 0.0,
                    "key {j} is {win} back from query {last} but moved the answer by {moved:e}"
                );
            }
        }
    }

    /// A decode-shaped launch at a length that forces the key axis to split,
    /// and a prefill-shaped one at the model's own head geometry: both run,
    /// both produce finite output that is not one repeated number.
    ///
    /// No tolerance here on purpose — a kernel that has gone wrong at this size
    /// produces NaNs, zeros or a constant, and one that is subtly off produces
    /// a number only the paired harness can judge.
    #[test]
    fn inklings_shape_at_a_real_length_runs() {
        for (nq, keys) in [(1usize, 20_000usize), (256, 4_000)] {
            let sh = Shape {
                nq,
                q0: keys - nq,
                keys,
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
                eff: 512,
                window: None,
            };
            let (q, k, v, rel) = inputs(&sh);
            let out = run(&sh, &q, &k, &v, &rel, &[]);
            assert_eq!(out.len(), nq * sh.heads * sh.head_dim);
            assert!(
                out.iter().all(|x| x.is_finite()),
                "{nq} queries over {keys} keys produced a non-finite value"
            );
            let lo = out.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                hi - lo > 1e-3,
                "every output is {lo}: the kernel computed nothing"
            );
        }
    }
}
