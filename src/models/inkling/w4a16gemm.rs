//! W4A16: an NVFP4 **weight** against a BF16 **activation**.
//!
//! [`super::fp4gemm::fp4_linear`] runs the hardware block-scaled MMA
//! (`kind::mxf4nvf4`), and that instruction takes E2M1 for BOTH operands — so
//! it quantises the activation too. For the ROUTED experts that is not a
//! liberty: the checkpoint's `hf_quant_config.json` sets `"*input_quantizer"
//! … enable: true`, i.e. the publisher calibrated for a 4-bit activation and
//! W4A4 is what they meant.
//!
//! For everything else in the model they did not. The sink experts, the
//! attention projections and the unembedding carry no `.scale` sidecar and no
//! `input_amax`; the publisher left them BF16 and never calibrated an
//! activation quantiser for them. Quantising their activations is a numerics
//! decision nobody made.
//!
//! But the *reason* to want four bits there is untouched by that. Decode is
//! bound on the DRAM read, not on the MMA: `lm_head` alone is
//! `[201024, 4096]` BF16 = 1.65 GB, read once per step and once per draft
//! depth. The bandwidth win of FP4 lives entirely in that read. So this lane
//! takes the half of NVFP4 that is free — the stored bytes — and declines the
//! half that costs calibration.
//!
//! ## Where the dequantisation happens
//!
//! In registers, during the B-fragment load, and nowhere else. Decoding the
//! weight into a scratch buffer first and multiplying that would spend exactly
//! the bandwidth this exists to save (and then some: the scratch is written as
//! well as read). Each lane reads the packed `u32` word holding its own two
//! codes plus the one E4M3 scale covering them, expands them to BF16 in two
//! registers, and hands those straight to
//! [`cmma::MmaDefinition::execute`]. The weight is never wider than four bits
//! anywhere outside a register file.
//!
//! Structurally this is [`super::bf16gemm::bf16_linear`]: same `m16n8k16`
//! instruction, same one-plane-per-`(m_tile, n_tile)` grid, same
//! `position_of_nth` fragment addressing, same f32 accumulator. The only edit
//! is what sits between the global load and `reg_b`.
//!
//! ## Two facts that make the pair-at-a-time decode exact
//!
//! * `vector_size(B)` is 2 for BF16 and `position_of_nth` returns an EVEN `k`
//!   for the first of each pair, so a lane's two elements are `k` and `k + 1`
//!   with `k` even — one packed byte, low nibble first ([`super::nvfp4`]
//!   settled that ordering against `compressed_tensors`).
//! * A 16-element scale block starts at a multiple of 16, so an even-aligned
//!   pair never straddles one. Both elements share a scale.
//!
//! The loads below are written per-element anyway, indexing `(gc + j)` rather
//! than assuming `gc`. On the layout above both `j` resolve to the same word
//! and the same scale byte, so it is one DRAM line either way and the kernel
//! does not silently depend on a fragment layout it cannot check.

use cubecl::e4m3;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::bf16;

/// Rows of one MMA tile — the M granularity everything here is padded to.
pub const MTILE: usize = 16;
/// Columns of one MMA tile.
pub const NTILE: usize = 8;
/// K covered by one `m16n8k16` instruction — the BF16 operand's k, not the
/// 4-bit operand's. The weight being narrow does not widen the instruction.
pub const KTILE: usize = 16;
/// Logical elements per E4M3 block scale (NVFP4's `group_size`).
pub const GROUP: usize = 16;
/// E2M1 codes packed into one `u32` word (eight nibbles).
pub const CODES_PER_WORD: usize = 8;

/// The value of one E2M1 code, as [`super::nvfp4::FP4_E2M1`] tabulates it.
///
/// Written as seven comparisons rather than an array index because a
/// runtime-indexed local array spills to local memory, and rather than the
/// exponent/mantissa arithmetic because THIS is the host table transcribed —
/// the values on the right are `FP4_E2M1[1..8]` in order, so the two can be
/// diffed by eye. The subnormal (`0x1` = 0.5) is the entry the arithmetic form
/// gets wrong, and it is the one that is hardest to notice.
#[cube]
pub(crate) fn e2m1_value(code: u32) -> f32 {
    let mag = code & 7u32;
    let mut v = f32::new(0.0f32);
    if mag == 1u32 {
        v = f32::new(0.5f32);
    }
    if mag == 2u32 {
        v = f32::new(1.0f32);
    }
    if mag == 3u32 {
        v = f32::new(1.5f32);
    }
    if mag == 4u32 {
        v = f32::new(2.0f32);
    }
    if mag == 5u32 {
        v = f32::new(3.0f32);
    }
    if mag == 6u32 {
        v = f32::new(4.0f32);
    }
    if mag == 7u32 {
        v = f32::new(6.0f32);
    }
    // Bit 3 is the sign. `-0.0` for code `0x8` is what the table says and what
    // the packer emits for a negative value that rounds to zero magnitude.
    if code >= 8u32 {
        v = -v;
    }
    v
}

/// Default for [`live_row_mask`]: OFF until it is measured on this part.
pub const LIVE_ROW_MASK_DEFAULT: bool = false;

/// Skip the A-operand loads whose fragment row is M PADDING, and hand the MMA a
/// register zero in their place.
///
/// ## What this is, and what it is NOT
///
/// It is NOT a bandwidth fix and must not be sold as one. `mma16_lane_dump`
/// settled that for the B operand and the same argument holds for A: over a
/// whole k loop these loads already reach full sector and line utilisation,
/// because the loop walks each row FORWARD and a half-used sector is finished
/// by the next few k-tiles out of L1. And A is the operand this module's header
/// already calls L2-RESIDENT at `m_pad = 16` — 128 KiB, re-read by every cube.
/// A DRAM-traffic model predicts nothing from this. What changes is the L1
/// SECTOR REQUEST COUNT; whether request count was costing anything is a
/// question for a step measurement, not for this comment.
///
/// ## Why there is anything to remove
///
/// `m16n8k16` gives LOAD `i` of lane `l` — the loop index in the A-load body,
/// which is `position_of_nth`'s element index divided by `vs_a = 2` — the A
/// element at `row = l/4 + 8*(i & 1)`, `col = 2*(l%4) + 8*((i>>1) & 1)`.
/// (Derived in cubecl-cpp 0.10.0 `cuda/processors.rs`, `row_index`/`col_index`,
/// and dumped off the device by `mma16_lane_dump`, so it is not assumed.) So of
/// the four loads a lane issues per k-tile, the two with `i` ODD address rows
/// 8..15 and the two with `i` even address rows 0..7. At decode `m` is 1 and
/// `m_pad` is [`MTILE`] = 16: rows 8..15 are entirely padding, and so are rows
/// 1..7. Fifteen of the sixteen rows the tile reads exist only because the
/// instruction is sixteen rows tall. [`super::bf16gemm::pad_bf16`] and
/// `to_bf16` write them as zero and `super::burn::linear_w4a16` slices them off
/// the output again — the kernel is the only place they are ever touched.
///
/// Counted per warp per k-tile at `m_pad = 16`, and keeping REQUESTS and
/// SECTORS apart because that distinction is the whole point: A is four warp
/// load requests, and each one touches eight distinct 32-byte sectors — one per
/// fragment row, 16 useful bytes in each, the row stride `2k` being a multiple
/// of 32 keeping them apart. So A is **4 requests / 32 sectors**. Under the mask
/// at `m = 1` the two `i`-odd loads are predicated off in EVERY lane and issue
/// no request at all, and the two `i`-even ones survive with only the four lanes
/// that hold row 0 active, landing in one sector each: **2 requests / 2
/// sectors**.
///
/// ## Why it is bit-identical
///
/// Two independent reasons, and the second does not depend on the padding being
/// zero:
///
/// 1. The padding rows ARE zero, and `0 * b` is exactly `+0` for every finite
///    `b` a dequantised E2M1 code times an E4M3 scale can be, so the f32
///    accumulator is unchanged bit for bit.
/// 2. `D[r][c] = sum_k A[r][k] * B[k][c] + C[r][c]`: A row `r` reaches
///    accumulator row `r` and NO other. Every row the mask suppresses is a row
///    `linear_w4a16` discards at its `slice([0..m, 0..n])`.
///
/// ## Why it is not the A-side fragment reorder
///
/// Permuting A into fragment order would make each of the four loads 32 lanes x
/// 4 contiguous bytes — 128 B, 4 sectors — so 32 sectors would become 16, at the
/// cost of a permuted copy of the activation on every step plus the registers to
/// address it. This takes them to 2, and it FREES the address arithmetic of the
/// suppressed loads rather than spending registers — which matters on a kernel
/// that sits at 78 registers against a `launch__occupancy_limit_blocks` of 24
/// and measures WORSE at 86 (see [`swz_unroll`]). The two are not exclusive, but
/// at `m = 1` the mask strictly dominates: it removes more and costs less.
///
/// ## Where it does and does not apply
///
/// Both shipped lanes take it: [`w4a16_linear`] (row-major B) and
/// [`w4a16_linear_swz`] (permuted B) load A through the same index space, so
/// the mask is the same three lines in each and the saving is the same 32 -> 2.
/// [`w4a16_linear_wide`] is deliberately left out: it is an experiment kernel no
/// shipped path launches, and masking a kernel nobody runs is dead code.
/// `super::bf16gemm::bf16_linear` has the identical A load and is left out for
/// the same reason — a real run reports `hand BF16 lane: 0 launches`, because
/// every plain-BF16 GEMM reaches a `cubek` tuned lane, and those bounds-check
/// their own tiles and take the true `m` unpadded already.
///
/// ## Two masks, and only one of them can free a register
///
/// This is worth stating precisely because the attractive version of the claim
/// is false. The RUNTIME predicate `gr < m_live` stops a load from issuing; it
/// does NOT free the register the load would have written, because `m_live` is a
/// runtime scalar and the compiler cannot know the value will not be needed. At
/// `swz_unroll`'s depth 4 that is sixteen `a_buf` slots either way.
///
/// The COMPTIME half, `hi_dead`, is the one that reaches the register budget:
/// when nothing from row 8 up is live — which `live_arg` establishes from `m`
/// alone, since `m <= 8` forces a single m-tile — the `i`-odd loads are deleted
/// at compile time, and with them their address arithmetic and their eight
/// `a_buf` slots. That is the version that can move
/// `launch__registers_per_thread`, and it is the reason the flag is two
/// comptime booleans rather than one.
///
/// If it does move it DOWN, the consequence is larger than the mask: `swz_unroll`
/// records depth 8 at 86 registers, pushing `launch__occupancy_limit_registers`
/// below the part's `launch__occupancy_limit_blocks` of 24 and costing occupancy
/// (37.41% against depth 4's 44.77%). Depth 8 measured worse BECAUSE it was
/// register-bound, not because the depth was wrong. Registers freed here are
/// registers depth 8 could spend. That is a hypothesis with an obvious
/// experiment, not a result: read `launch__registers_per_thread` for both arms
/// before believing any of it, and note that a mask which RAISES the count would
/// be a reason to stop, since the lane has single-digit registers of headroom.
///
/// ## Measured
///
/// GB10 (spark), commit `b451912`, one box under the box lock with no other
/// compute app on the device. Framing travels with each number.
///
/// **Bit-identity — the gate, and it PASSES.** `w4a16_swz_probe`'s mask section,
/// masked against unmasked, SAME binary and same process, the two arms differing
/// only in the comptime `mask_rows`/`hi_dead`; every output compared including
/// the padding rows the caller slices away:
///
/// ```text
///   shape                       live/m_pad   outputs   row-major   swizzled
///   [16, 256] x [64, 256]^T          1/16       1024    0.000e0     0.000e0
///   [16, 256] x [64, 256]^T          8/16       1024    0.000e0     0.000e0
///   [16, 256] x [64, 256]^T          9/16       1024    0.000e0     0.000e0
///   [48, 256] x [64, 256]^T         37/48       3072    0.000e0     0.000e0
/// ```
///
/// 6144 outputs, 0 differing f32 bits anywhere. The four shapes are chosen, not
/// convenient: `1/16` is decode; `48` is MULTI-TILE with a partly-padded LAST
/// tile, which a predicate that forgot `m_base` would fail; and `8` against `9`
/// straddles the `hi_dead` boundary, so the comptime-deleted-load variant and
/// the runtime-predicate-only variant are both exercised against the same
/// unmasked reference.
///
/// **The fragment map HOLDS on the device.** `mma16_lane_dump` dumps A off the
/// device and checks `row = lane/4 + 8*(i&1)`, `col = 2*(l%4) + 8*((i>>1)&1)`:
/// HOLDS. Loads `i1` and `i3` address rows 8..15 for every lane, exactly as
/// `hi_dead` assumes.
///
/// **A's cost, per warp per k-tile, at `m_pad = 16`, `k = 4096`** — computed
/// from that dumped map through this kernel's own index arithmetic, so it is a
/// property of the device's layout and not of an assumption:
///
/// ```text
///   live rows   requests   32B sectors   what changed
///        16         4          32        the unmasked lane
///         9         4          18        runtime predicate only (hi_dead off)
///         8         2          16        hi_dead: the i-odd loads are gone
///         1         2           2        decode
/// ```
///
/// Note the two regimes. Above 8 live rows only SECTORS fall, because every load
/// still has some live lane. At or below 8 the `i`-odd loads have no live lane
/// at all and the REQUEST count halves as well — and that is the same threshold
/// at which `hi_dead` lets the compiler delete them outright.
///
/// **Requests, sectors and REGISTERS, from `ncu`.** One process, one binary,
/// `n = 1024`, `k = 4096`, `m_pad = 16`, load depth 4; the mask-on arm is
/// `m_live = 1` and the mask-off arm is the unmasked kernel, so the two differ
/// only in the comptime flags. ncu reports whole-kernel sums; these are divided
/// by `sm__warps_launched.sum` (128) times the k-tile count (`4096/16` = 256) to
/// give PER WARP PER K-TILE, which is the unit the derivation above is in.
/// `ncu` needs `sudo` here — `ERR_NVGPUCTRPERM` otherwise.
///
/// ```text
///                                  requests      32B sectors      registers
///   kernel                        off -> on      off -> on       off -> on
///   w4a16_linear (row-major)      8.0 -> 6.0     64.0 -> 34.0      44 -> 36
///   w4a16_linear_swz (shipped)    7.0 -> 5.0     35.0 ->  5.0      80 -> 73
///   w4a16_linear_swz, sc row-maj  7.0 -> 5.0     42.0 -> 12.0      78 -> 73
/// ```
///
/// Read the DELTAS down the column: **-2 requests and -30 sectors in every
/// lane**, which is exactly A going from 4 requests / 32 sectors to 2 / 2 and
/// nothing else moving. That is the derivation confirmed against the hardware
/// rather than restated. It also answers "does the row-major lane want this
/// too": yes, and by the same ABSOLUTE amount — 64 -> 34 is a 47% cut where the
/// swizzled lane's 35 -> 5 is 86%, and the difference is entirely that row-major
/// B costs more sectors to begin with, not that the mask does less.
///
/// **Registers FALL, in all three.** -7 on the shipped swizzled lane (80 -> 73),
/// -5 against the 78 that [`swz_unroll`]'s table records for depth 4, and -8 on
/// the row-major lane (44 -> 36, whose `launch__occupancy_limit_registers` rises
/// 40 -> 48 with it). This is `hi_dead` doing what the runtime predicate cannot:
/// the `i`-odd loads are gone at compile time, so their address arithmetic and
/// their eight `a_buf` slots are never allocated. The mask is not
/// register-neutral — it is register-POSITIVE, which was the thing most worth
/// checking before spending a box slot on timing.
///
/// **The depth-8 arm was run on 2026-08-27, and the hypothesis was half right.**
/// It predicted that the mask would buy back the registers depth 8 needed, and
/// it did — 95 -> 80, `launch__occupancy_limit_registers` 20 -> 24, achieved
/// occupancy equal to depth 4's to within 0.02 points. The door opened. Depth 8
/// is still slower anyway, by 5.5% at the sink grid and 16.9% at the head, at
/// equal occupancy and byte-identical traffic. So the mask does NOT unlock a
/// deeper load, and the register story is no longer the reason depth 8 loses.
/// The measurement and what it retires are recorded at [`SWZ_UNROLL_DEFAULT`],
/// beside the register table it corrects.
///
/// ## Which decode work this actually reaches
///
/// Not the unembedding, at the shipped default. `inkling_forward`'s
/// `ANN_BUDGET_DEFAULT` is 8192, so a one-row decode step takes `burn::linear_ann`
/// — a shortlist over the sketch, its own kernel — and the exact `[201024, 4096]`
/// W4A16 head GEMM only runs at `INK_ANN_HEAD=0`, under `INK_ANN_VERIFY`, or at
/// `m > 1`. What the mask reaches at `m = 1` is the per-layer W4A16 SINK
/// weights, which every layer launches on both pipe halves; the binary names
/// them at startup ("W4A16 sink weights [n, k] ... written in ... order"). At
/// prefill `m` is 3732 against `m_pad` 3744, so 233 of 234 m-tiles are entirely
/// live and only the last tile has anything to skip — the mask is close to a
/// no-op there by construction, which is the point of it being a predicate and
/// not a second kernel.
///
/// `INK_W4A16_ROWMASK=1` turns it on, so one binary can run both arms.
///
/// ## Measured on this part, 2026-08-27: it costs nothing and buys nothing
///
/// [`LIVE_ROW_MASK_DEFAULT`] says "OFF until it is measured on this part". It is
/// measured. `w4a16_swz_grid` free-running at `INK_M=16`, depth 4, four
/// processes per arm in ALTERNATED order with a memory settle between each, box
/// locked and idle, GB/s over the weight table per launch:
///
/// ```text
///   grid                    masked   unmasked   delta   threshold
///   head  [201024, 4096]    163.0     161.8     +0.8%     1.39%
///   sink  [8192, 4096]      121.6     123.4     -1.5%     1.39%
/// ```
///
/// Both inside the unpaired process-to-process threshold, in opposite
/// directions. The mask is FREE and it is WORTH NOTHING at either shape, and
/// that is consistent with everything above rather than a surprise: it removes
/// 12.9M of 45.1M L1 requests (-28.5%) and 26.5M L2 sectors, and the requests it
/// removes are the ones that HIT — `l1tex__t_sector_hit_rate` is 82.75% unmasked
/// against 53.31% masked — so they never reach the path this lane is short of.
/// The "NOT a bandwidth fix ... recovers no DRAM bytes by construction" above is
/// confirmed to the sector: `lts__d_sectors_fill_sysmem.sum` is 14486323 masked
/// against 14481795 unmasked, 0.03% apart.
///
/// This is the RE-READ the ceiling correction asked for, and the answer is that
/// the ceiling correction does not reach it. "A third of the traffic is
/// unclaimed, so re-read the candidates killed for having nothing left to buy"
/// assumed the third was BYTES; it is transactions (see the ceiling block), and
/// the mask does not remove the transactions that count. The kill stands, on the
/// grounds it always had rather than on the retracted ones.
///
/// One methodological note worth more than the result: `ncu` puts the mask at
/// 6.9% at the head and 5.9% at the sink. That delta does not survive a
/// free-running clock — under clock-lock the extra L1-hitting requests cost
/// issue slots, and at full clock they hide inside a memory shadow that is there
/// either way. Read `ncu` for COUNTS, which are exact, and a free-running box
/// for TIME.
/// Turn a launch's `m_live` into the kernel's `(mask_rows, hi_dead, m_live)`.
///
/// `None` is "load every row as before" and is the only shape in which the
/// count can be absent, so a masked launch cannot be written without one.
///
/// `hi_dead` is the COMPTIME half of the mask and it is the half that can free
/// a register. `m_live` is a runtime scalar, so `gr < m_live` is a runtime
/// predicate: it stops a load from ISSUING but the compiler still has to
/// allocate somewhere to put the result, and at depth 4 that is sixteen `a_buf`
/// slots whatever the mask does. `hi_dead` says something the compiler can act
/// on instead — that the `i`-odd loads, whose fragment row is `lane/4 + 8` and
/// therefore at least 8, address nothing live — and then those loads are not
/// emitted at all and their slots are not allocated.
///
/// It is one BOOL and not a row bound, deliberately: a comptime row bound would
/// compile a kernel variant per distinct `m`, and prefill's `m` varies per
/// prompt. The only comptime question that changes code is whether row 8 can be
/// live, so that is the only thing passed, and the whole model needs exactly two
/// variants — decode's and everyone else's.
///
/// `m <= MTILE / 2` is sufficient AND safe, and the argument needs nothing about
/// tiling: load `i` odd sits at within-tile row `lane/4 + 8 >= 8`, so its global
/// row is `m_base + 8` or more, and `m_base >= 0` — so the global row is at
/// least 8, which is at least `m`. Dead in EVERY tile of every shape. (An
/// earlier version of this comment reasoned via "m <= 8 forces a single m-tile",
/// which is true but is a weaker claim resting on `m_pad`; this one holds even
/// if a caller passes an `m_pad` that does not match `m`.)
///
/// **It depends on the fragment map**, specifically on `row = lane/4 + 8*(i&1)`
/// making `8*(i&1)` the smallest row load `i` can touch. `mma16_lane_dump`
/// checks that closed form against the device and says HOLDS or DOES NOT HOLD,
/// and `w4a16_swz_probe`'s mask gate would catch a violation as a non-identical
/// output at its `(16, 1)` and `(16, 8)` shapes — which is what those two shapes
/// are for. This is the same dependency `swz_word_k16` already carries, declared
/// the same way.
fn live_arg(m_pad: usize, m_live: Option<usize>) -> (bool, bool, u32) {
    match m_live {
        Some(m) => {
            assert!(m <= m_pad, "m_live {m} exceeds the padded {m_pad} rows");
            // `as u32` would TRUNCATE, and a truncated row count is a silently
            // wrong mask rather than a crash. Unreachable at any real shape --
            // 2^32 rows of a 4096-wide BF16 activation is 32 TiB -- which is
            // exactly why it must not be the thing that fails quietly.
            (
                true,
                m <= MTILE / 2,
                u32::try_from(m).expect("m_live fits a u32"),
            )
        }
        // The count the kernel never reads. `m_pad` and not 0, so a masked
        // kernel handed this by mistake would still be CORRECT, only slow.
        None => (false, false, m_pad as u32),
    }
}

pub fn live_row_mask() -> bool {
    // Cached: this is read on every LAUNCH, which is inside the timed region of
    // every harness that measures this lane.
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("INK_W4A16_ROWMASK")
            .map(|v| v != "0")
            .unwrap_or(LIVE_ROW_MASK_DEFAULT)
    })
}

/// `out = (a @ b^T) * scale`, with `a` BF16 and `b` NVFP4.
///
/// `a` is `[m_pad, k]` BF16; `b` is `[n, k/8]` `u32` (element `i` of word `w`
/// at bits `4*(i%8)`, low nibble = lowest index) and `b_sc` is `[n, k/16]`
/// E4M3 — the layout [`super::fp4quant`] writes and the checkpoint stores.
/// `out` is `[m_pad, n]` f32, the accumulator's own type.
///
/// `scale` is the tensor-wide `scale2`; it is applied once to the accumulator
/// rather than per element, which is where [`super::fp4gemm::fp4_linear`] puts
/// it too. That is a deliberate deviation from
/// [`super::nvfp4::decode_row`]'s block-scale-then-`scale2` order — folding it
/// into the accumulator is not associativity-equivalent — and it is the same
/// deviation the NVFP4 lane already makes, so the two lanes agree with each
/// other.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear<AB: Scalar + Cast, S: Scalar, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    b: &Array<u32>,
    b_sc: &Array<S>,
    out: &mut Array<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] mask_rows: bool,
    #[comptime] hi_dead: bool,
    scale: f32,
    m_live: u32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    // 1 for BF16. Kept in the A index arithmetic so this reads as the same
    // kernel as its two parents rather than a third dialect of it.
    let pack = AB::packing_factor();

    // M in x, N in y. Grid x varies fastest, so the cubes that share a weight
    // row — the whole n-tile's `[8, k]` of codes and scales — are the ones
    // launched adjacently, and one DRAM read of that row serves all
    // `m_pad / 16` of them out of L2. The other order puts consumers of the
    // same row `n / 8` cubes apart, which at the head shape is the whole
    // weight table between them, so every m-tile re-reads from DRAM.
    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    // Packed words and scale bytes per weight row.
    let wpr = comptime!(size_k / CODES_PER_WORD);
    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE);

    for t in 0..k_tiles {
        let kbase = t * KTILE;
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
            let gr = row as usize + m_base;
            let gc = col as usize + kbase;
            // Two masks, and they are not the same mask. `hi_dead` is COMPTIME
            // and deletes the load: `i` odd means `row = lane/4 + 8 >= 8`, and
            // `live_arg` only sets `hi_dead` when nothing from row 8 up is live,
            // so there is no address to compute and no register to hold. The
            // `gr < m_live` one is RUNTIME: it stops the load issuing but the
            // result still needs somewhere to land. See `live_row_mask` for the
            // bit-identity argument and `live_arg` for why one of these is a
            // bool and not a row bound.
            if mask_rows {
                if comptime!(hi_dead && (i & 1) == 1) {
                    reg_a[i] = Vector::<AB, NA>::cast_from(0.0f32);
                } else {
                    let mut v = Vector::<AB, NA>::cast_from(0.0f32);
                    if gr < m_live as usize {
                        v = a[(gr * size_k + gc) / a.vector_size()];
                    }
                    reg_a[i] = v;
                }
            } else {
                reg_a[i] = a[(gr * size_k + gc) / a.vector_size()];
            }
        }
        #[unroll]
        for i in 0..vc_b {
            // B is column-major w.r.t. the tile: `col` indexes n, `row`
            // indexes k, and the checkpoint's `[out, in]` rows are exactly
            // that. Same addressing as the BF16 lane; only the fetch differs.
            let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
            let gr = col as usize + n_base;
            let gc = row as usize + kbase;

            // ONE word and ONE scale for the whole fragment element group,
            // not one of each per element. The two facts in the module header
            // say they are the same word and the same scale byte -- `gc` is
            // even and a 16-element scale block starts at a multiple of 16 --
            // so the per-element form was issuing four memory instructions
            // where the BF16 lane issues one, for a quarter of the bytes. That
            // is why the four-bit head measured 69 GB/s against the BF16
            // head's 163: at 2 KB a row the lane is instruction-bound in this
            // load, not bandwidth-bound.
            //
            // This DOES now depend on the fragment layout the header describes.
            // The detector is `linear_w4a16_tracks_linear_bf16_on_the_same_weight`:
            // a wrong nibble or a wrong scale is an order-one error and blows
            // straight past that test's bound, which is what it is for.
            let word = b[gr * wpr + gc / CODES_PER_WORD];
            let s = f32::cast_from(b_sc[gr * spr + gc / GROUP]);
            let mut v = Vector::<AB, NA>::empty();
            #[unroll]
            for j in 0..vs_b {
                let kk = gc + j;
                let code = (word >> (4 * (kk % CODES_PER_WORD)) as u32) & 15u32;
                // The one widening in the lane, and it stops at the register.
                v[j] = AB::cast_from(e2m1_value(code) * s);
            }
            reg_b[i] = v;
        }

        let d = def.execute(&reg_a, &reg_b, &acc);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = d[i];
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gr = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(gr * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// Launch [`w4a16_linear`] for a `[m_pad, k] x [n, k]^T` product.
///
/// Mirrors [`super::fp4gemm::fp4_linear_launch`] minus the activation's codes
/// and scales: `a` is a BF16 handle, not a quantised pair.
///
/// `m_live` is `Some(m)` to MASK the A operand's padding rows — `m` being how
/// many of `m_pad`'s rows are real — and `None` to load them as before. It is
/// one argument and not a flag plus a count so that "mask off" cannot be said
/// with a row count that contradicts it, and so both arms are reachable from
/// ONE process: `Some`/`None` picks between two comptime kernel variants rather
/// than between two builds. Production reads [`live_row_mask`] for the choice;
/// that function carries the bit-identity argument and what the mask does and
/// does not buy.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
    m_live: Option<usize>,
) -> Handle {
    assert_eq!(
        m_pad % MTILE,
        0,
        "m_pad {m_pad} is not a multiple of {MTILE}"
    );
    let (mask_rows, hi_dead, live) = live_arg(m_pad, m_live);
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");
    // N rides grid y, which CUDA caps at 65535 (x is 2^31-1). The largest N in
    // the model is the unembedding's 201024 = 25128 tiles, well inside it, but
    // the cap is silent if it is ever exceeded so it is checked here.
    assert!(
        n / NTILE <= 65535,
        "{} n-tiles exceed the 65535 grid-y limit",
        n / NTILE
    );

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    // Two BF16 per `.b32`, which is what `contiguous_elements` reports for A
    // and B and what the fragment layout actually is.
    let vs = 32 / bf16::cube_type().size_bits();
    let wpr = k / CODES_PER_WORD;
    let spr = k / GROUP;

    unsafe {
        w4a16_linear::launch::<bf16, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            ArrayArg::from_raw_parts(a.clone(), m_pad * k),
            ArrayArg::from_raw_parts(b.clone(), n * wpr),
            ArrayArg::from_raw_parts(b_sc.clone(), n * spr),
            ArrayArg::from_raw_parts(out.clone(), m_pad * n),
            k,
            n,
            mask_rows,
            hi_dead,
            scale,
            live,
        )
    };
    out
}

// ---------------------------------------------------------------------------
// The same product with more warps and wider loads — kept because it MEASURED
// SLOWER, which is the useful part.
//
// The starting suspicion was residency and memory-level parallelism.
// `w4a16_linear` above is one 32-thread cube per 16x8 output tile; on GB10
// `cudaDevAttrMaxBlocksPerMultiprocessor` is 24 against 48 warp slots, so a
// one-warp cube caps residency at 50% and no register budget can recover it (it
// uses 40 registers; 51 blocks' worth would fit). Its K loop is not unrolled
// either, so a warp has one k-tile of weight fetch outstanding at a time behind
// 154 SASS instructions of address arithmetic, bounds clamping and nibble
// expansion.
//
// This variant changes three things: `PLANES` planes per cube instead of one, a
// single 16-byte `LDG.E.128` covering both k-tiles of a step (a lane's two B
// words for k-tile `t` are packed indices `kbase/8` and `kbase/8 + 1`, and the
// next k-tile's are `+2` and `+3`, which at a 32-aligned `kbase` are four
// contiguous aligned words), and one scale byte per k-tile where the original
// loads the same byte twice. It is 269 SASS instructions per two k-tiles
// against 154 per one — 13% fewer per k-tile — at the same 40 registers.
//
// Measured at the head's own shape (m_pad 16, k 4096, n 201024, 0.431 GiB of
// codes + scales, min of four warm launches, launch + sync, GB10, GPU
// otherwise idle), against `w4a16_linear` in the same process:
//
// ```text
//   PLANES   wide         original    verdict
//   1        4.95 ms      4.97 ms     wider loads and 13% fewer instructions: nothing
//   4        5.27-5.35    4.66-4.74   more warps: 13% WORSE
//   8        5.32-6.76    4.81-4.91   more warps still: 10-38% WORSE
// ```
//
// So this lane is not instruction-bound and not occupancy-bound. The
// corroborating measurement is `fp4gemm::fp4_linear`, which runs the hardware
// block-scaled MMA and needs no software dequantisation at all: 12 416
// instructions per warp over the head's K against this kernel's 39 424, 3.2x
// fewer, for 4.53 ms against 4.66. Instruction count moved 3.2x and time moved
// 3%.
//
// What is left is the access pattern itself. Every warp owns eight weight rows
// and consumes 32 bytes from each before advancing, so the memory system sees
// 32-byte reads 2048 bytes apart, thousands of warps deep — and adding warps
// adds streams, which is why more planes made it worse. A fully coalesced read
// of the SAME 0.431 GiB in the same process runs at 242 GB/s against these
// kernels' 98-106. The gap is coalescing, and coalescing needs a cooperative
// stage through shared memory, which is exactly what one warp per output tile
// forecloses.
//
// THE CEILING IS 242 GB/s, AND EVERY EARLIER FIGURE FOR IT WAS LOW. This
// sentence said 158-172 until 2026-08-27, three controls in the tree disagreed
// by 25%, and re-measuring settled it in the direction none of them predicted:
// all three recorded numbers are low and there is only ONE ceiling.
//
// FRAMING RULE for the re-measurement: GB/s over the 0.431 GiB weight table
// (codes + E4M3 scales) of ONE `[201024, 4096]` head operand, per LAUNCH, one
// launch and one sync each, GB10 (sm_121a), `scripts/gb10-lock.sh` held on both
// boxes and both verified idle (no compute apps, load 0.08). It is not a step
// figure and not a two-node figure.
//
// ```text
//   instrument / framing                                     box      GB/s
//   fp4_lane_dump `stream_packed`, min of 4 warm launches     spark2   240.8-244.2  (n=7)
//   fp4_lane_dump `stream_packed`, min of 4 warm launches     spark    240.7-241.3  (n=3)
//   w4a16_swz_probe `stream ceiling`, p50 of 6 warm of 8      spark2   240.5-243.8  (n=3)
//   w4a16_swz_probe `stream ceiling`, p50 of 12 warm of 14    spark2   243.4-243.5  (n=2)
//   fp4_lane_dump `stream_packed`, one sibling GPU tenant     spark2   242.1-242.4  (n=2)
// ```
//
// Seventeen process-level readings, two instruments, two boxes, two rep counts,
// idle and contended: 240.5-244.2 GB/s, a ±0.8% spread. This control does not
// move. It also reconciles with `inkling_membw`'s 247-259 GB/s coalesced rows
// over 1 GiB: those are pipelined, this one syncs per launch, and ~0.1 ms of
// launch+sync on a 1.9 ms kernel is the whole difference.
//
// Where the three old numbers came from, in descending order of how well it is
// understood:
//
// * **158-172 — a BROKEN INSTRUMENT, and the tree already says so.** The
//   instrument was `fp4_lane_dump`'s `stream_packed` as it stood on 2026-08-25,
//   which is the only harness that reads a coalesced stream of these bytes in
//   the SAME PROCESS as these kernels at "min of four warm launches" — this
//   comment's own framing, word for word. Its thread stride was a u32-word
//   count applied to a tensor CubeCL indexes in 16-byte vectors, so every
//   thread strode four times past the buffer, and its store was unconditional,
//   charging a 12.5% write tax to a read figure. Fixed the same day; see
//   `fp4_lane_dump`'s module header. The GEMM half of that old reading was
//   sound and still reproduces exactly: 98-106 GB/s, measured again here at
//   98.7-102.2 (`w4a16_linear`) and 101.6-106.8 (`fp4_linear`).
// * **213.2 and 218.4 — not reproduced, and not explained.** Both are the
//   arms measured above, at 12% and 10% below what those same arms read tonight
//   on either box. Three explanations were tested and all three failed: it is
//   not the box (spark and spark2 agree within 1.1%), not the rep count (the
//   213.2 row's own "12 warm reps of 14" reads 243.4), and not a GPU tenant (a
//   sibling process halves the GEMM arm, 101.8 -> 43.3 GB/s, and moves this arm
//   by 0.3%). What is left is a HYPOTHESIS WITH A NAMED WEAKNESS: both readings
//   predate `scripts/gb10-lock.sh` (2026-08-27 01:5x), so their boxes were
//   unreserved, and the 213.2 table's own end-to-end neighbour below records
//   another agent's job arriving mid-run. But the sibling that failed to move
//   this arm was a short microbenchmark, not a 101 GiB decode process, so that
//   test does not cover the case it needs to. Do not repeat 213.2 or 218.4 as
//   the ceiling; if you need one, run either arm on a locked idle box, which
//   takes about ninety seconds.
//
// WHAT THIS OVERTURNS. Two claims rested on the low figure and neither survives:
//
// * `fp4_linear_grouped`'s 171 GB/s was "at the ceiling" against 158-172. It is
//   70.7% of 242. `moegroup`'s header had already retracted this from the other
//   side ("**The 170.4 was wrong**", same instrument bug); this is the same
//   retraction reaching the sentence above, which is why "at the ROUTED-EXPERT
//   shape it measures right" is deleted rather than softened.
// * The depth-4 swizzled head at 159-161 GB/s is 66% of the ceiling, not the
//   ~95% that 158-172 implied and not the 75.7% that 213.2 implied. So the head
//   lane has roughly a third of its traffic still unclaimed, and any candidate
//   that was killed for having nothing left to buy at this shape — the
//   M-padding predicate among them — was killed against a ceiling that was
//   14-40% too low and should be re-read before it is treated as settled.
//
// Nothing in the live-row mask depends on this: the mask recovers no DRAM bytes
// by construction and is not a GB/s claim at all.
//
// WHERE THE THIRD GOES, MEASURED 2026-08-27. Not to DRAM bytes, not to the
// arithmetic, and not to occupancy. `w4a16_geom_ceiling` builds the rungs this
// paragraph was missing — the SAME load stream and the SAME grid with the
// dequantise and the MMA removed, and the same 25128 per-warp streams read flat
// instead of through the fragment map — all in one process on the same buffers,
// same framing as the table above:
//
// ```text
//   arm                         ms     GB/s   % ceil   sect/req    L2 req
//   coalesced (the 242)      1.831    252.9   100.0          16    3.624M
//   geom width 16 B/lane     1.853    249.9    98.8          12    3.624M
//   geom width 4 B/lane      2.021    229.2    90.6           3    4.524M
//   geom, no A no scales     2.337    176.2    69.7           1    9.495M
//   geom, no A               2.766    167.4    66.2           1   10.717M
//   geom, 1 warp/cube        2.934    157.9    62.4           1   11.089M
//   geom, 8 warps/cube       3.233    143.2    56.6           1   12.676M
//   real w4a16_linear_swz    2.854    162.3    64.2           1   13.462M
// ```
//
// Four back-to-back processes, and the two BANDS are what to quote rather than
// any single row: the flat/wide arms land at 90-104% of the coalesced control
// and the fragment-shaped ones at 56-68%, and the bands do not overlap in any
// process.
//
// Every row moves the SAME 463 MB from DRAM (`lts__d_sectors_fill_sysmem.sum`
// within 0.1%, the codes-only row excepted and charged only for its codes), so
// the numerator was never wrong and there is no over-fetch anywhere. What
// differs is how many TRANSACTIONS carry those bytes, and the `m16n8k16` B
// fragment map is what multiplies them: it puts four lanes on the same 32-bit
// word, so a warp-wide load fetches exactly ONE 32-byte sector
// (`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld` is 1 for
// every fragment-shaped arm and 16 for the control), and the lane spends 11.1M
// L2 requests where the control spends 3.6M for identical bytes.
//
// Two candidate explanations die here and both were plausible:
//
// * THE ARITHMETIC IS FREE. The last two rows differ by the entire E2M1 ladder,
//   the scale multiply, the BF16 cast and the MMA — 525 million instructions a
//   launch, 60% of the stream (`smsp__inst_executed.sum` 870.6M against 345.8M).
//   The real lane is 2.7% FASTER than the control that does none of it on mins
//   and 0.4% slower on p50, i.e. inside the ~1.1% paired resolution. Do not read
//   `sm__throughput` 65.6% as "compute bound": the arithmetic sits inside a
//   memory shadow that is already there without it.
// * OCCUPANCY IS WORTH LESS THAN NOTHING. One warp per cube against this part's
//   24-block-per-SM cap is 24 of 48 warp slots, structurally, whatever the
//   register file does — and the lane already achieves 98% of that cap.
//   Widening the cube to 8 warps lifts achieved occupancy 49.3% -> 91.0% at
//   byte-identical L1 requests and is 10% SLOWER. The request path is saturated
//   at 24 warps an SM.
//
// So the direction this header already pointed at is the one the numbers
// support — "coalescing needs a cooperative stage through shared memory, which
// is exactly what one warp per output tile forecloses" — and the width rows put
// an upper bound on it: 90-99% of the ceiling for the LOAD half. They are not a
// proposed kernel. They read the right bytes in the right order and hand them to
// the WRONG LANES; putting them where `m16n8k16` needs them costs shuffles or a
// shared-memory stage that nothing here measures.
//
// AND THE M-PADDING PREDICATE SURVIVES THE RE-READ WITHOUT MOVING. Re-measured
// free-running at the head, `w4a16_swz_grid` at `INK_M=16`, depth 4, four
// processes per arm in alternated order with a memory settle between each:
// 163.0 GB/s masked against 161.8 unmasked (+0.8%), and at the 1024-cube sink
// 121.6 against 123.4 (-1.5%) — both inside the 1.39% unpaired threshold. The
// mask removes 12.9M of 45.1M L1 requests (-28.5%) and 26.5M L2 sectors and
// buys nothing, which is consistent rather than contradictory: the requests it
// removes are the ones that HIT (L1 sector hit rate 82.75% unmasked against
// 53.31% masked) and never reach the saturated path. Its own doc said it "is
// NOT a bandwidth fix" and "recovers no DRAM bytes by construction", and that is
// confirmed to the sector — 14486323 masked against 14481795 unmasked, 0.03%
// apart. What the ceiling correction implied — that a third of the BYTES were
// available and a byte-recovering candidate should be re-read — does not reach
// it, because the third is transactions and the mask does not remove the
// transactions that count. (`ncu` shows the mask worth 6.9% at the head and 5.9%
// at the sink; that delta does not survive a free-running clock, and the free
// clock is the one production runs at.)
//
// A is left alone throughout: at `m_pad = 16` it is 128 KiB, L2-resident, and
// re-read by every cube.
//
// One thing above IS occupancy-shaped after all, and it is not in the kernel:
// the GRID ORDER. See `fp4gemm`'s module header for the full sweep. The short
// version, same harness and same framing (head shape `k = 4096`, `n = 201024`,
// min of four warm launches, launch + sync, GB10, GPU idle, kernel time):
//
// ```text
//   m_pad   N in x     M in x     ratio        wide, N in x -> M in x
//      16   4.98 ms    4.88 ms    1.02         4.83 -> 4.87   (unchanged)
//      32   9.87       6.99       1.41         9.83 -> 5.90
//      64  18.92      16.43       1.15        19.27 -> 14.38
//     128  38.59      33.33       1.16        40.96 -> 29.82
// ```
//
// Both lanes gain, and BOTH gain less than `fp4_linear` does at the same shapes
// (1.97-2.41x). The obvious difference is A: this lane's activation is BF16, so
// its A traffic is four times the NVFP4 lane's over the same `m_pad`, and A is
// the operand the other launch order was serving out of L2. Obvious is not
// measured, though — that is a hypothesis, and this timer cannot separate it
// from the instruction count the block above already showed this lane is
// carrying.

/// Planes per cube. At 4 the block limit stops binding and the register budget
/// takes over — 40 x 128 = 5120 registers a cube, 12 cubes an SM, all 48 warp
/// slots — and the lane gets 13% SLOWER for it. Left at 1 because that is what
/// measured best; the constant stays so the experiment can be re-run by
/// changing one character.
pub const PLANES: u32 = 1;
/// K covered by one loop iteration: two `m16n8k16` steps, one 16-byte B load.
pub const KSTEP: usize = 2 * KTILE;
/// Packed `u32` words one lane fetches per iteration.
pub const WORDS_PER_STEP: usize = 4;

/// `out = (a @ b^T) * scale`, `a` BF16 and `b` NVFP4 — same contract as
/// [`w4a16_linear`], same numerics, different residency.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_wide<AB: Scalar + Cast, S: Scalar, NA: Size, NC: Size, NB: Size>(
    a: &Array<Vector<AB, NA>>,
    b: &Array<Vector<u32, NB>>,
    b_sc: &Array<S>,
    out: &mut Array<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    // One plane per n-tile; the cube covers `PLANES` of them. M in x for the
    // same reason as `w4a16_linear` above: weight-row sharers launch adjacent.
    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize * comptime!(PLANES as usize) + UNIT_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    // `b` is indexed in 16-byte vectors, so a row is `size_k / 8 / 4` of them.
    let vpr = comptime!(size_k / CODES_PER_WORD / WORDS_PER_STEP);
    let spr = comptime!(size_k / GROUP);
    let steps = comptime!(size_k / KSTEP);

    // The lane's own k offset inside a tile, hoisted: it does not depend on the
    // step, and the original recomputed it from `UNIT_POS_PLANE` six times an
    // iteration.
    let (b_row0, b_col0) = def.position_of_nth(lane, 0u32, MatrixIdent::B);
    let gr = b_col0 as usize + n_base;
    let klane = b_row0 as usize;

    for s in 0..steps {
        let kbase = s * KSTEP;
        // ONE 16-byte load: the four packed words covering both k-tiles.
        let quad = b[gr * vpr + kbase / (CODES_PER_WORD * WORDS_PER_STEP)];

        #[unroll]
        for half in 0..2usize {
            let khalf = kbase + half * KTILE;
            #[unroll]
            for i in 0..vc_a {
                let (row, col) =
                    def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
                let ga = row as usize + m_base;
                let gc = col as usize + khalf;
                reg_a[i] = a[(ga * size_k + gc) / a.vector_size()];
            }
            // Both fragment halves of a k-tile share this scale byte, because
            // `klane` is at most 6 and a 16-element block starts at a multiple
            // of 16. The original read it twice.
            let sc = f32::cast_from(b_sc[gr * spr + khalf / GROUP]);
            #[unroll]
            for i in 0..vc_b {
                // Fragment element `i` sits `i * 8` further along k, which is
                // exactly one packed word: word `2 * half + i` of the quad.
                let word = quad[2 * half + i];
                let mut v = Vector::<AB, NA>::empty();
                #[unroll]
                for j in 0..vs_b {
                    let code = (word >> (4 * (klane + j)) as u32) & 15u32;
                    v[j] = AB::cast_from(e2m1_value(code) * sc);
                }
                reg_b[i] = v;
            }
            let d = def.execute(&reg_a, &reg_b, &acc);
            #[unroll]
            for i in 0..vc_c {
                acc[i] = d[i];
            }
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let go = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(go * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// Launch [`w4a16_linear_wide`]; same signature as [`w4a16_linear_launch`].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_wide_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Handle {
    assert_eq!(
        m_pad % MTILE,
        0,
        "m_pad {m_pad} is not a multiple of {MTILE}"
    );
    let ntiles = n / NTILE;
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(
        ntiles % PLANES as usize,
        0,
        "{ntiles} n-tiles do not divide into cubes of {PLANES} planes"
    );
    assert_eq!(k % KSTEP, 0, "k {k} is not a multiple of {KSTEP}");
    assert!(
        ntiles / PLANES as usize <= 65535,
        "{} n-cubes exceed the 65535 grid-y limit",
        ntiles / PLANES as usize
    );

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / bf16::cube_type().size_bits();
    let vpr = k / CODES_PER_WORD / WORDS_PER_STEP;
    let spr = k / GROUP;

    unsafe {
        w4a16_linear_wide::launch::<bf16, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (ntiles / PLANES as usize) as u32, 1),
            CubeDim::new_2d(32, PLANES),
            vs,
            2,
            WORDS_PER_STEP,
            ArrayArg::from_raw_parts(a.clone(), m_pad * k),
            ArrayArg::from_raw_parts(b.clone(), n * vpr),
            ArrayArg::from_raw_parts(b_sc.clone(), n * spr),
            ArrayArg::from_raw_parts(out.clone(), m_pad * n),
            k,
            n,
            scale,
        )
    };
    out
}

// ---------------------------------------------------------------------------
// Pre-permuted ("swizzled") B layout for m16n8k16
// ---------------------------------------------------------------------------

/// Bytes one `(n_tile, k_tile)` block of codes occupies: `NTILE` rows x
/// `KTILE / 2` packed bytes.
pub const SWZ_BLOCK_CODES: usize = NTILE * KTILE / 2;
/// Bytes one `(n_tile, k_tile)` block of scales occupies: one E4M3 per row.
pub const SWZ_BLOCK_SCALES: usize = NTILE;

/// Where lane `l`'s `i`-th B load lands in a swizzled block.
///
/// `mma16_frag_map` dumps the map off the device and it holds the closed form
///
/// ```text
///   col = lane >> 2                  the n column, 0..8
///   row = 2 * (lane & 3) + 8 * i     the k element, 0..16
/// ```
///
/// exactly, on sm_121a, for all 32 lanes. So a lane's `i`-th load wants the
/// packed word covering k elements `[8i, 8i+8)` of weight row `col`, and the
/// warp's `i`-th load wants eight such words — one per row.
///
/// Row-major that is eight addresses `k/2` bytes apart: eight 32-byte sector
/// requests for 32 useful bytes. Written down as
///
/// ```text
///   dst_word(col, w) = w * NTILE + col        w = row / 8, in 0..2
/// ```
///
/// load `i` is words `[8i, 8i+8)` of the block, i.e. **32 contiguous bytes** —
/// one sector where the row-major form takes eight.
///
/// ## What this does and does not buy
///
/// It does not save DRAM bytes. Over a whole k loop the row-major form already
/// reaches 100% sector and 100% line utilisation, because the k loop walks each
/// of the eight rows FORWARD and a half-used sector is finished by the next few
/// k tiles out of L1 — `mma16_lane_dump` prints both counts side by side. What
/// it saves is REQUESTS: 4096 sector requests per warp k loop against 512
/// distinct sectors, an 8x amplification that the permutation removes.
#[inline]
fn swz_word_k16(col: usize, w: usize) -> usize {
    w * NTILE + col
}

/// Permute `[n, k/8]` packed E2M1 codes into `m16n8k16` B-fragment order.
///
/// Block `(n_tile, k_tile)` occupies [`SWZ_BLOCK_CODES`] consecutive bytes at
/// `((n_tile * k/KTILE) + k_tile) * SWZ_BLOCK_CODES`. Same length, same bytes,
/// different order. `src` and `dst` must not overlap.
pub fn swizzle_w4a16_codes_into(src: &[u8], dst: &mut [u8], n: usize, k: usize) {
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");
    assert_eq!(src.len(), n * k / 2, "codes are not [n, k/2] bytes");
    assert_eq!(
        dst.len(),
        src.len(),
        "destination is not the source's length"
    );
    let kt = k / KTILE;
    let row_w = k / CODES_PER_WORD;
    for nt in 0..n / NTILE {
        for t in 0..kt {
            let blk = (nt * kt + t) * SWZ_BLOCK_CODES;
            for col in 0..NTILE {
                for w in 0..KTILE / CODES_PER_WORD {
                    let s = ((nt * NTILE + col) * row_w + t * (KTILE / CODES_PER_WORD) + w) * 4;
                    let d = blk + swz_word_k16(col, w) * 4;
                    dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
                }
            }
        }
    }
}

/// [`swizzle_w4a16_codes_into`] allocating its own destination.
pub fn swizzle_w4a16_codes(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    swizzle_w4a16_codes_into(src, &mut dst, n, k);
    dst
}

/// Permute `[n, k/16]` E4M3 block scales to match [`swizzle_w4a16_codes_into`].
///
/// One `m16n8k16` covers exactly [`GROUP`] k elements, so a fragment consumes
/// ONE scale byte per weight row: eight bytes, row-major eight separate
/// sectors. Blocked as `[n_tile][k_tile][8]` they are eight contiguous bytes.
pub fn swizzle_w4a16_scales_into(src: &[u8], dst: &mut [u8], n: usize, k: usize) {
    assert_eq!(n % NTILE, 0);
    assert_eq!(k % KTILE, 0);
    assert_eq!(src.len(), n * (k / GROUP), "scales are not [n, k/16]");
    assert_eq!(
        dst.len(),
        src.len(),
        "destination is not the source's length"
    );
    let kt = k / KTILE;
    let spr = k / GROUP;
    for nt in 0..n / NTILE {
        for t in 0..kt {
            let blk = (nt * kt + t) * SWZ_BLOCK_SCALES;
            for col in 0..NTILE {
                dst[blk + col] = src[(nt * NTILE + col) * spr + t];
            }
        }
    }
}

/// [`swizzle_w4a16_scales_into`] allocating its own destination.
pub fn swizzle_w4a16_scales(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    swizzle_w4a16_scales_into(src, &mut dst, n, k);
    dst
}

/// Whether a `[n, k]` weight can be written in the swizzled layout at all.
pub fn swizzleable(n: usize, k: usize) -> bool {
    n % NTILE == 0 && k % KTILE == 0
}

/// Permute a `[n, k/8]` packed-code plane on the DEVICE, into a fresh handle.
///
/// The head's codes are produced on the device by
/// [`super::fp4quant::quantize_nvfp4_bf16`], not copied out of the pile, so
/// there is no host memcpy for the permutation to ride inside the way the
/// routed experts' has. A device pass is the cheap alternative: one linear
/// write of the destination with a gathered read, 0.43 GiB at the head's shape,
/// once per process at startup.
///
/// Written destination-linear on purpose. The gather side is the scattered one,
/// and a scattered READ of a resident table is what this whole permutation
/// exists to make the GEMM stop doing per step — paying it once is the trade.
#[cube(launch)]
pub fn swizzle_codes_dev(src: &Array<u32>, dst: &mut Array<u32>, #[comptime] k_tiles: usize) {
    let d = ABSOLUTE_POS as usize;
    if d < dst.len() {
        let wpb = comptime!(SWZ_BLOCK_CODES / 4);
        let per_row = comptime!(KTILE / CODES_PER_WORD);
        let blk = d / wpb;
        let j = d % wpb;
        let w = j / NTILE;
        let col = j % NTILE;
        let nt = blk / k_tiles;
        let t = blk % k_tiles;
        dst[d] = src[(nt * NTILE + col) * (k_tiles * per_row) + t * per_row + w];
    }
}

/// Permute a `[n, k/16]` E4M3 scale plane on the DEVICE. See
/// [`swizzle_codes_dev`].
#[cube(launch)]
pub fn swizzle_scales_dev<S: Scalar>(
    src: &Array<S>,
    dst: &mut Array<S>,
    #[comptime] k_tiles: usize,
) {
    let d = ABSOLUTE_POS as usize;
    if d < dst.len() {
        let blk = d / NTILE;
        let col = d % NTILE;
        let nt = blk / k_tiles;
        let t = blk % k_tiles;
        dst[d] = src[(nt * NTILE + col) * k_tiles + t];
    }
}

/// Permute both planes of an already-quantised `[n, k]` weight on the device.
///
/// Returns fresh handles; the caller drops the row-major ones. Panics rather
/// than silently declining on a shape the layout cannot express — a weight that
/// half-permuted would be a kernel reading the wrong bytes and producing
/// NUMBERS, which is the one failure mode this must not have.
pub fn swizzle_w4a16_device<R: Runtime>(
    client: &ComputeClient<R>,
    codes: &Handle,
    scales: &Handle,
    n: usize,
    k: usize,
) -> (Handle, Handle) {
    assert!(swizzleable(n, k), "[{n}, {k}] is not swizzleable");
    let k_tiles = k / KTILE;
    let words = n * k / CODES_PER_WORD;
    let sc = n * (k / GROUP);
    let dc = client.empty(words * 4);
    let ds = client.empty(sc);
    let threads = 256u32;
    unsafe {
        swizzle_codes_dev::launch::<R>(
            client,
            CubeCount::Static(words.div_ceil(threads as usize) as u32, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(codes.clone(), words),
            ArrayArg::from_raw_parts(dc.clone(), words),
            k_tiles,
        )
    };
    unsafe {
        swizzle_scales_dev::launch::<e4m3, R>(
            client,
            CubeCount::Static(sc.div_ceil(threads as usize) as u32, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(scales.clone(), sc),
            ArrayArg::from_raw_parts(ds.clone(), sc),
            k_tiles,
        )
    };
    (dc, ds)
}

/// Whether the head/sink W4A16 weights are written in MMA-fragment order.
///
/// On by default. `INK_W4A16_SWZ=0` is the A/B arm.
///
/// ## What it measured
///
/// **The kernel, alone.** `w4a16_swz_probe`, one process, four interleaved
/// arms, first two reps discarded, per LAUNCH of one `[16, 4096] x
/// [201024, 4096]^T` product (the unembedding's own shape at decode, `m = 1`
/// padded to one m-tile), one GB10 box, GB/s over the weight planes only
/// (0.431 GiB) and not over a step:
///
/// | arm              | p50      | GB/s  |
/// |------------------|---------:|------:|
/// | row-major        | 4.832 ms |  95.9 |
/// | both planes swz  | 3.984 ms | 116.3 |
/// | codes only swz   | 4.069 ms | 113.8 |
/// | coalesced ceiling| 2.172 ms | 213.2 |  <- LOW, see below
///
/// The ceiling row does not reproduce and should not be quoted. Re-run
/// 2026-08-27 at this exact framing (`INK_SWZ_REPS=14`, so 12 warm reps of 14,
/// same binary, same arm, box locked and verified idle) it reads 1.903 and
/// 1.902 ms — 243.4 and 243.5 GB/s. The other three rows are a DEPTH-1
/// measurement, superseded by [`SWZ_UNROLL_DEFAULT`]'s table rather than wrong;
/// the ceiling row is simply 12% low for reasons nobody has recovered. The full
/// reconciliation, and the two conclusions it overturns, is in the ceiling block
/// above `w4a16_linear_wide`.
///
/// 12 warm reps of 14. The two swizzled arms are indistinguishable within
/// their spread, so this does not establish that the scale plane's permutation
/// is worth anything on its own; both are permuted because keeping one plane
/// row-major would be a second layout for no measured gain.
///
/// **End to end.** `bench-decode.sh -n 4 --gen 12 --layers 21:42`, `INK_KV=1`,
/// a 3720-token prompt, ctx 3732, ONE GB10 box holding layers 21..42 and the
/// head (not the two-node pipe), arms interleaved, per DECODE STEP, p50 of 11
/// warm passes a rep:
///
/// | arm       | reps (p50 each)        | p50      |
/// |-----------|------------------------|---------:|
/// | swz       | 55.9, 56.1, 56.1 ms    | 56.1 ms  |
/// | row-major | 57.1, 57.2, 57.3 ms    | 57.2 ms  |
///
/// 1.1 ms a step, 1.9%, and the two arms' rep bands do not overlap. Eight reps
/// were run and TWO were discarded as contended -- `swz` rep 1 and `row-major`
/// rep 4, identified by their PREFILL, 165.5 s and 95.5 s against 14.2-15.3 s
/// for the other six; another agent's job arrived on the box mid-run. One
/// discard fell on each arm.
///
/// The gap between 17.5% on the kernel and 1.9% on the step is not a
/// discrepancy: the head is one term of a step that also streams 5.36 GiB of
/// attention, dense-MLP and routed-expert weight, and 1.1 ms of a 57.2 ms step
/// is most of what a 0.85 ms kernel saving can be worth once it is enqueued
/// against a host that is also doing everything else.
///
/// **Numerics.** Bit-identical, as a permutation must be: max deviation
/// 0.000e0 over 1024 outputs of a `[16, 256] x [64, 256]^T` product against the
/// row-major lane. Reported as an observation; there is no gate on it.
///
/// # RETRACTED FOR THE SINKS, 2026-08-26. Do not quote the table above at any
/// # shape but the head's.
///
/// SUPERSEDED IN SCOPE by the section below it, which measured the range this
/// one inferred from two points. The +25% is real and the lesson is right; what
/// is wrong here is the UNIT. "The sinks" is not a shape -- it is two shapes,
/// and only one of them loses. Read both sections, and take the rule from the
/// second: `swizzle_pays`.
///
/// The end-to-end row (`swz` 56.1 against `row-major` 57.2 ms/step) was taken
/// with `ANN_BUDGET_DEFAULT` already at 8192, so the head went through
/// `linear_ann` in BOTH arms and read identical bytes -- meaning the 1.1 ms was
/// attributed to the SINKS. Two things were wrong with that. It was confounded:
/// the `swz` arm of that A/B was the one carrying the `linear_ann` seam bug, so
/// the two arms generated different token streams and routed to different
/// experts. And when the unconfounded A/B was finally run -- a109514 against
/// e30f22a, ctx 3732, split 21, 7 reps, 441 warm passes a side -- it came back
/// the other way:
///
/// ```text
///   w4a16_linear, both nodes   6.84 -> 8.55 ms   (+1.71, +25% SLOWER permuted)
///     head 3.30 -> 4.05, tail 3.54 -> 4.50
/// ```
///
/// The kernel name changed `w4a16_linear_ab` -> `w4a16_linear_swz_ab`, so the
/// permuted path is unambiguously what ran.
///
/// The KERNEL table at the top stands -- it is the head's shape, 201024 x 4096,
/// ~25128 cubes, and 95.9 -> 116.3 GB/s there is real. The sinks are 8192x4096
/// and 4096x2048, i.e. 1024 and 512 cubes. `swz_grid_scaling` had already shown
/// the multiplier shrinking as the grid starves; at the sink shapes it inverts.
///
/// So `w4a16_bind` now permutes the HEAD only, and only when the approximate
/// lane is off. The sinks are correct in either layout -- they never touch
/// `ann_logits` -- so leaving them row-major costs nothing but the speed it
/// buys back. THE LESSON, which is the reason this retraction is written out
/// rather than deleted: a bandwidth figure is a property of a SHAPE, and this
/// one was carried across a 25x difference in cube count without anybody
/// noticing that the framing rule had been dropped.
///
/// # WHY the multiplier is shape-dependent, and what the shape actually is
/// # (2026-08-26)
///
/// The retraction above is right that it lost and right about the lesson. It is
/// wrong about the UNIT: "the sinks" is not a shape, it is two shapes, and they
/// fall on opposite sides of the line. `w4a16_swz_grid` is the A/B neither
/// earlier measurement had -- rotating buffers so neither arm is the L2-warm
/// one, 20 pipelined launches to ONE sync so the host round trip is not the
/// floor, arms round-robined, min of 6 rounds after 2 discarded, `m_pad` 16,
/// one GB10 box, GB/s over the weight planes only, `ratio` = row-major time /
/// swizzled time:
///
/// ```text
///   k=2048     512 cubes   70.7 ->  62.0 GB/s   0.88   <- sink `down`, ACTUAL
///             1024         92.6 -> 100.5        1.09
///            25128        116.7 -> 133.0        1.14
///   k=4096     256         49.0 ->  34.3        0.70
///              512         77.9 ->  65.7        0.84
///              768         90.7 ->  93.0        1.03   <- crossover
///             1024         94.3 -> 106.6        1.13   <- sink `gate_up`, ACTUAL
///             2048         90.9 -> 113.5        1.25   <- dense `g`/`u`
///            25128        107.3 -> 132.6        1.24   <- the head
///   k=16384    512         46.9 ->  58.4        1.25   <- dense `down`, ACTUAL
///             1024         66.4 ->  95.1        1.43
///             2048         67.8 ->  98.3        1.45
/// ```
///
/// TWO knobs, not one. The multiplier rises monotonically with CUBE COUNT and
/// crosses 1.0 near 750 cubes -- 0.65 of a wave, a wave being 48 SMs x 24 blocks
/// = 1152 single-warp cubes -- and the value it saturates to is set by K:
/// ~1.10 at k=2048, ~1.24 at k=4096, ~1.45 at k=16384. Cube count alone does not
/// decide it, which is why `down` at 512 cubes LOSES for the sinks (k=2048) and
/// WINS for the dense MLP (k=16384) at the identical cube count.
///
/// That cube count is the cause and not a correlate is the one thing a sweep
/// over `n` cannot show, so it was isolated: same `[4096, 4096]` table, same
/// bytes, only the m-tile count varied, so only the warps moved.
///
/// ```text
///   m_pad  16 ->  512 cubes  0.86       m_pad  64 -> 2048 cubes  1.15
///   m_pad  32 -> 1024 cubes  1.09       m_pad 128 -> 4096 cubes  1.20
/// ```
///
/// ## The mechanism, off `ncu`
///
/// Load sectors and requests per warp per k-tile, at both sink shapes, from
/// `l1tex__t_{requests,sectors}_pipe_lsu_mem_global_op_ld.sum` over
/// `cubes * k_tiles`. The decomposition closes to the integer on all three
/// arms and both shapes, which is why it can be read as a mechanism and not as
/// a correlation:
///
/// ```text
///                             requests   sectors   =    A  + codes + scales
///   row-major                     8         64        4x8=32   2x8    2x8
///   swizzled, scales permuted     7         35        4x8=32   2x1    1x1
///   swizzled, scales row-major    7         42        4x8=32   2x1    1x8
/// ```
///
/// So the permutation is worth 8x on each B stream -- 32 sectors of B become 3
/// -- and only **1.83x on the kernel**, because A is 32 of the 64 sectors and
/// the permutation does not touch it. After permuting B, A is 32 of the
/// remaining 35: **91% of the swizzled kernel's sector traffic is now the
/// activation.** PARKED, not a task, and it wants this rule shipped and measured
/// in-model first: A's floor is 16 sectors per warp-k-tile (one k-tile of A is
/// 16 rows x 16 k x 2 B = 512 B), so a fragment-ordered A would cut 35 -> 19, a
/// further 1.84x, on top of B's 1.83x. At decode there is ONE m-tile, so every
/// cube reads the identical A -- 16 x k x 2 bytes, 128 KiB at k = 4096 -- and a
/// per-step permutation of that is ~0.5 us against a 0.2 ms kernel. It is the
/// largest single number here and it is deliberately not being acted on yet.
///
/// And the row-major form's "wasted" seven-eighths is not waste, which is the
/// part that decides the sign. A 32-byte sector holds eight packed words = FOUR
/// k-tiles of one weight row, so row-major's burst of 8 parallel sector requests
/// is an incidental 4-deep PREFETCH: it misses once and then hits L1 for three
/// k-tiles (`l1tex__t_sector_hit_rate` 93.9-94.7% against the swizzled form's
/// 89.0-91.3%). The swizzled 64-byte block is consumed by the k-tile that
/// fetched it, so it exposes a fresh 2-sector miss EVERY k-tile instead of an
/// 8-sector burst every fourth -- four times as many exposed latencies at a
/// quarter of the per-warp memory-level parallelism, on a k loop that is not
/// unrolled and so has one k-tile of fetch outstanding at a time.
///
/// With enough resident warps the machine hides that with thread-level
/// parallelism and the 8x request saving is all that is left, so the permutation
/// wins. Below ~0.65 waves there is nothing to hide it with and the lost
/// prefetch dominates. `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active`
/// says exactly that: at 1024 cubes the swizzled kernel stalls LESS on memory
/// (13.2-13.8 against 16.7-17.1), and at 512 cubes that advantage is gone
/// (16.6-18.7 against 16.7-17.0).
///
/// THREE candidates die here, and all three had been stated confidently.
///
/// * It is NOT occupancy. Both lanes report the identical achieved occupancy
///   (44.4% at 1024 cubes, 22.1% at 512), both are capped by
///   `launch__occupancy_limit_blocks` = 24 -- one warp per cube against 48 warp
///   slots -- and below that by the grid itself.
/// * It is NOT register pressure. The swizzled kernel uses FEWER registers, 36
///   against 44 (38 with the scales left row-major).
/// * It is NOT "a separate kernel tuned for the head's regime", which is what
///   `20a0b06`'s commit message says and what this doc has to outlive, because a
///   commit message cannot be corrected. Strip the comments and the two kernels
///   differ by SIX LINES, all of them index arithmetic: same `CubeCount`, same
///   `CubeDim::new_1d(32)`, same tile, same unrolls, same accumulator, same
///   output store, 74 code lines against 79 (the permuted one is the LONGER).
///   There is no tuning knob that differs, so there is nothing to re-tune; the
///   whole difference in behaviour is the access pattern above.
///
/// One inference to retire with them, because it is the intuitive one and it is
/// backwards: "at low occupancy there is less to hide latency with, so halving
/// the request count matters MORE". Fewer requests is not more latency hiding.
/// At low occupancy what is scarce is outstanding requests PER WARP, and the
/// permutation removes them -- eight sectors in flight from one instruction
/// become one. That is why the sign inverts downward and not upward.
///
/// A measurement caveat that belongs with the numbers: `ncu`'s CYCLE counts are
/// inflated 2-7x for this kernel under kernel replay and reverse the ranking, so
/// only its COUNTERS above are usable. Every time figure here comes from
/// `w4a16_swz_grid`, not from `ncu`.
///
/// ## Confirmed IN-MODEL, not only on the probe
///
/// `bench-decode.sh -n 4 --gen 12 --layers 0:21`, `INK_KV=1`, `ctx3732.ids`
/// (3744 ctx), one GB10 holding layers 0..21, arms INTERLEAVED, per DECODE STEP,
/// 11 warm passes a rep, CORPUS-UNBOUNDED (the right shape for a question about a
/// kernel -- its cost does not read the text). One binary throughout, identical
/// sha256 on all three arms; they differ only by environment:
///
/// ```text
///   arm      reps (ms/step)             median     min
///   off      47.0 47.2 48.0 47.4        47.300    47.000   both sinks row-major
///   rule     46.1 46.4 46.2 46.8        46.300    46.100   `swizzle_pays`
///   forced   46.5 48.0 47.1             47.100    46.500   both sinks permuted
/// ```
///
/// The rule is worth 1.0 ms a step; the harness scores it +2.16% and the two
/// bands do NOT overlap -- `rule`'s worst rep (46.8) beats `off`'s best (47.0).
/// That is the claim.
///
/// `forced` is NOT a second claim. The harness scores it +0.32% against `off`
/// and says so itself: "SMALLER THAN THE SPREAD. Not a result." It is consistent
/// with permuting `down` giving the gain back, which is what a 0.88x arm should
/// do, but this run cannot distinguish it from `off`; it is a direction, not a
/// measurement, and it landed three reps rather than four.
///
/// RE-MEASURED at the shipped load depth of 4, after [`swz_unroll`] landed and
/// `swizzle_pays` relaxed to permute BOTH sinks. Same harness, same corpus, same
/// layers, one binary across both arms:
///
/// ```text
///   arm    reps (ms/step)              median    spread
///   off    47.5 47.7 47.5 46.6         47.500     2.4%
///   rule   46.4 46.2 45.8 45.8         46.000     1.3%
/// ```
///
/// +3.26%, up from +2.16% before the load-ahead, and again the bands do not
/// overlap -- `rule`'s worst rep (46.4) beats `off`'s best (46.6). The startup
/// log confirms what was bound: `rule` writes both sinks MMA-FRAGMENT, `off`
/// writes both row-major.
///
/// A STAMP BOTH RUNS CARRY, and a correction to what the first one blamed. The
/// harness marks these UNGATED because its POST-run gate reads `loadavg 6.65`
/// against a limit of 2.0 -- but that load is the benchmark's OWN decode process
/// decaying on a 20-core box, and it appears identically in the second run,
/// where the box was not touched from outside at all. So it is a property of
/// measuring decode on this part, not evidence of a contender; an earlier
/// version of this paragraph blamed the author's polling, and the second run
/// falsified that. The tree was also DIRTY at `20a0b06`, so the commit does not
/// identify what ran; the shared binary sha256 across arms is what makes each
/// comparison internally valid. The kernel-level tables, where the mechanism
/// comes from, do not depend on either run.
///
/// ## What it says about the four consumers
///
/// * sink `gate_up` `[8192, 4096]`, 1024 cubes: the permutation WINS 1.13-1.22x.
///   `w4a16_bind` used to decline it, on the weight-kind rule; `swizzle_pays`
///   now takes it.
/// * sink `down` `[4096, 2048]`, 512 cubes: LOSES 0.88x, and is declined. This is
///   the shape a deliberate load-ahead is aimed at -- see [`swz_unroll`].
/// * the head `[201024, 4096]`, 25128 cubes: WINS 1.24x, and is declined for a
///   CORRECTNESS reason (the `ann_logits` seam), not this one.
/// * the dense MLP, which is the next candidate and the reason this was chased:
///   `g`/`u` `[16384, 4096]` at 2048 cubes WINS 1.25x, and `down`
///   `[4096, 16384]` at 512 cubes WINS 1.25x. Reasoning from the sinks as a
///   proxy would have got the second one backwards -- it is at the same starved
///   cube count that loses for the sinks, and it wins anyway because k is 8x
///   longer.
///
/// A NOTE ON THE +25% ABOVE, which this does not reproduce for `gate_up`:
/// `a109514 -> e30f22a` spans `de482cb`, which the comment in `w4a16_bind`
/// already flags as bundling the sink permutation with moving the pool
/// `memory_usage` barrier off the decode path; and at `a109514`
/// `ann_owns_m1` was `ann_budget() > 0` with no `for_ann` gate, so NOTHING was
/// permuted in that arm. The kernel-level A/B above varies the layout and
/// nothing else.
///
/// Finally, the reason `w4a16_swz_probe` must not be run at a sink shape: its
/// four arms share buffers -- the `stream ceiling` arm reads the same `b_row` /
/// `bs_row` the row-major arm reads, immediately before it -- and the two plane
/// sets are 36 MiB against a 24 MiB L2, so the row-major arm is L2-warm every
/// rep and the swizzled arms are L2-cold every rep. It reports 0.176 against
/// 0.306 ms there, a 74% "loss" that is an artifact of the arm ORDER. At the
/// head's 463 MiB nothing is resident and the confound does not exist, which is
/// why the table at the top of this doc stands.
pub fn swizzle_w4a16() -> bool {
    std::env::var("INK_W4A16_SWZ")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Does the `m16n8k16` permutation PAY at a `[n, k]` weight's decode grid?
///
/// The shipped policy used to be a WEIGHT-KIND rule -- `!for_ann`, i.e. never a
/// sink -- derived from one shape. It is a CUBE-COUNT-AND-K rule, because that
/// is what the crossover is a function of. Measured by `w4a16_swz_grid`
/// (rotating buffers so neither arm is the L2-warm one, 20 pipelined launches to
/// one sync so the host round trip is not the floor, arms round-robined, min of
/// 6 rounds after 2 discarded, `m_pad` 16, one GB10 box, GB/s over the weight
/// planes only). `ratio` is row-major time / swizzled time; above 1 it pays:
///
/// ```text
///                 cubes    row -> swz GB/s   ratio      this predicate
///   k=2048          512     70.7 ->  62.0     0.88   no   <- sink `down`
///                  1024     92.6 -> 100.5     1.09   YES
///                 25128    116.7 -> 133.0     1.14   YES
///   k=4096          256     49.0 ->  34.3     0.70   no
///                   512     77.9 ->  65.7     0.84   no
///                   768     90.7 ->  93.0     1.03   no   (crossover, declined)
///                  1024     94.3 -> 106.6     1.13   YES  <- sink `gate_up`
///                  2048     90.9 -> 113.5     1.25   YES  <- dense `g`/`u`
///                 25128    107.3 -> 132.6     1.24   YES  <- the head
///   k=16384         512     46.9 ->  58.4     1.25   YES  <- dense `down`
///                  1024     66.4 ->  95.1     1.43   YES
///                  2048     67.8 ->  98.3     1.45   YES
/// ```
///
/// RELAXED 2026-08-26 once [`swz_unroll`] shipped. The table above is the
/// UN-PIPELINED kernel; at the shipped load depth of 4 every shape in it wins,
/// the two losing rows included (512 cubes goes 0.84 -> 1.20 at k=4096 and
/// 0.80 -> 1.02 at k=2048). So the predicate no longer encodes a crossover, only
/// the edge of what has been measured:
///
/// * `cubes >= 512` with `k >= 2048` is the whole measured region, and every
///   point in it pays at depth 4 -- worst case 1.02, at sink `down`.
/// * Below 512 cubes is DECLINED as unmeasured at depth 4, not as known-bad.
///   256 and 384 cubes measured 0.70 and 0.77 un-pipelined and have not been
///   re-run; the model has no such weight, so nothing turns on it.
/// * `k < 2048` is below the measured floor entirely and is declined.
///
/// The two-knob crossover below is kept because it is the MECHANISM, and because
/// it is what a future kernel change has to be judged against. It is no longer
/// the shipped rule.
///
/// WHY it is a function of these two things, since a threshold without its
/// mechanism becomes folklore -- which is how the +25% came to be applied 25x
/// outside its range in the first place:
///
/// A 32-byte sector holds eight packed words = FOUR k-tiles of one weight row,
/// so the row-major form's burst of eight sector requests is an incidental
/// 4-deep L1 PREFETCH: it misses once and then hits L1 for three k-tiles
/// (`l1tex__t_sector_hit_rate` 94.7% against the permuted form's 89.0-91.3%).
/// Its "wasted seven-eighths" was never waste, which is why every framing built
/// on sector efficiency alone was going to mislead. The permuted form's 64-byte
/// block is consumed by the k-tile that fetched it, so it exposes a fresh
/// 2-sector miss EVERY k-tile instead of an 8-sector burst every fourth -- four
/// times the exposed latencies at a quarter of the per-warp memory-level
/// parallelism, on a k loop that is not unrolled.
///
/// So the trade is 8x fewer requests against a lost prefetch, and which side
/// pays depends on whether there are enough resident warps to hide the latency
/// the prefetch was covering. Cube count is that warp count: each cube is ONE
/// warp, `launch__occupancy_limit_blocks` is 24, and 48 SMs x 24 = a 1152-cube
/// wave, so the ~750-cube crossover is 0.65 of a wave. K is the second knob
/// because a longer k loop spreads the L1 working set and makes the row-major
/// form's reuse less viable, so the saturated multiplier grows with it: ~1.10 at
/// k=2048, ~1.24 at k=4096, ~1.45 at k=16384. That is why dense `down` at 512
/// cubes WINS 1.25x where sink `down` at 512 cubes LOSES 0.88x -- same starved
/// geometry, opposite sign, because k is 8x longer.
///
/// ## Judged at the DECODE grid, deliberately
///
/// The layout is chosen once per weight at bind time and cannot vary per call,
/// so it must be judged at one `m_pad`, and that one is decode's `MTILE`. This
/// is safe in the direction that matters: cubes are `(m_pad / MTILE) * (n /
/// NTILE)`, so a prefill has strictly MORE cubes than the decode this was
/// judged at, and more cubes only moves further into the region where the
/// permutation pays. A layout that is right for decode cannot be wrong for
/// prefill on this axis.
pub fn swizzle_pays(n: usize, k: usize) -> bool {
    let cubes = n / NTILE;
    k >= 2048 && cubes >= 512
}

/// Does `INK_W4A16_SWZ=1` explicitly ask for the permutation where the shipped
/// policy declines it?
///
/// [`swizzle_w4a16`] answers "is it allowed"; this answers "is it demanded". The
/// distinction exists because the shipped policy never permutes a sink, on a
/// measurement (25% slower at 512-1024 cubes) that has no counterpart at any
/// other shape -- we know 25128 cubes wins and 512-1024 loses and NOTHING in
/// between. Without a way to turn it back on, that crossover cannot be measured
/// without a rebuild, and an unmeasurable policy hardens into folklore, which is
/// exactly how the original figure came to be applied 25x outside its range.
///
/// This never overrides the CORRECTNESS guard. `ann_owns_m1` stays absolute:
/// `linear_ann` reads row-major and permuting the head while the approximate
/// lane owns m=1 makes every exact rescore read permuted bytes, which produced
/// plausible logits rather than an error. Forcing is a speed knob only.
pub fn swizzle_w4a16_forced() -> bool {
    std::env::var("INK_W4A16_SWZ")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// How many k-tiles the swizzled lane loads before it consumes any of them.
///
/// The permutation removes 8x of B's sector REQUESTS and, with them, the
/// row-major form's accidental 4-deep L1 prefetch -- see [`swizzle_pays`] for
/// the measurement and the mechanism. This is the attempt to keep both: issue
/// several k-tiles' loads up front so the warp carries them in flight
/// deliberately, rather than inheriting depth from a wasteful access pattern.
///
/// Depth 4 is the number to match, because that is what one row-major 32-byte
/// sector covers. The registers are free: `ncu` puts this lane's occupancy under
/// `launch__occupancy_limit_blocks` = 24, not under the register limit, and the
/// swizzled kernel sits at 36 registers against the row-major lane's 44.
///
/// `INK_W4A16_SWZ_UNROLL` overrides for a sweep; it must divide `k / KTILE`.
///
/// # What it measured (`w4a16_swz_grid`, same framing as [`swizzle_pays`])
///
/// `ratio` is row-major time / swizzled time. Depth 1 is the un-pipelined kernel
/// and reproduces the table in [`swizzle_pays`], which is the regression check
/// on this refactor:
///
/// ```text
///            cubes    depth 1   depth 2   depth 4
///   k=4096     512      0.87      0.92      1.20
///             1024      1.23      1.23      1.48
///             2048      1.18      1.28      1.52
///            25128      1.22      1.30      1.49
///   k=2048     512      0.80      0.87      1.02   <- sink `down`
///             1024      1.03      1.14      1.22
///            25128      1.14      1.14      1.39
/// ```
///
/// AT DEPTH 4 EVERY MEASURED SHAPE WINS, including both shapes that lost at
/// depth 1. The losing region is not a property of the permutation; it was the
/// permutation being charged for memory-level parallelism it had removed and
/// nobody had put back. Restoring it deliberately costs 3 registers a depth step
/// and buys the trade outright.
///
/// It is also worth far more than the crossover it fixes. The head goes
/// 107.9 -> 161.3 GB/s, against the 95.9 -> 116.3 in this module's original
/// table: the permutation alone was collecting less than half of what was
/// available, at EVERY cube count, because the un-pipelined loop kept one k-tile
/// of fetch outstanding at a time whatever the layout. THAT is the correction to
/// the mechanism in [`swizzle_pays`] -- the story there is right about which way
/// the trade tips and wrong to imply the high-cube-count end had no latency left
/// to hide. It did, and depth 4 collects it.
///
/// ## Why 4 and not 8, exactly
///
/// Depth 8 is measured and is WORSE: 1.17 against depth 4's 1.48 at 1024 cubes
/// (k=4096), 1.00 against 1.22 (k=2048), 1.12 against 1.57 (k=16384). `ncu` says
/// why, and the boundary is sharp:
///
/// ```text
///   depth   registers   launch__occupancy_limit_registers   achieved occupancy
///     1        38            48 blocks (slack)                   45.07%
///     4        78            24 blocks (exactly co-binding)      44.77%
///     8        86            20 blocks (BINDING)                 37.41%
/// ```
///
/// `launch__occupancy_limit_blocks` is 24 on this part, so depth 4 is the
/// largest depth whose register demand still lands at or above that cap --
/// occupancy is untouched. Depth 8 pushes the register limit BELOW the block cap
/// and occupancy falls with it. "The registers are free" was true, and it stops
/// being true between 4 and 8.
///
/// The register COLUMN has since drifted: re-read 2026-08-27 on the same kernel
/// with the mask off, depth 4 is 80 and depth 8 is 95, not 78 and 86. The
/// `limit_registers` column is unchanged (24 and 20) and so is the conclusion,
/// which is why the table is annotated rather than rewritten -- but do not quote
/// 78 or 86 as this lane's register count.
///
/// The control that makes this a schedule effect and not a traffic one: at
/// `[16384, 4096]` the sector count is 22020096 at depth 1, 4 AND 8, byte for
/// byte, with `sectors_per_request` 6 throughout. Same bytes, same requests, same
/// sectors -- only WHEN they are issued differs.
///
/// ## Depth 8 reopened on the live-row mask, and closed again for a NEW reason
///
/// The paragraph above is the reason depth 8 was killed, and by 2026-08-27 it
/// had MOVED: [`live_row_mask`] is register-positive, so the register budget the
/// kill rested on was no longer the one being spent. Reopened deliberately, and
/// the door does open -- `ncu` on `w4a16_linear_swz` launched by
/// `w4a16_swz_grid` at `INK_M=16`, `INK_MLIVE=1`, `k=4096`, `n=8192` (1024
/// cubes, the sink `gate_up` grid), one launch profiled, ncu 2025.3.1, spark2,
/// box locked and idle, NO timing in this reading:
///
/// ```text
///   depth  mask  registers  limit_registers  limit_blocks  achieved occ.  sectors
///     4    off      80          24 blocks       24 blocks       --           --
///     8    off      95          20 blocks       24 blocks       --           --
///     4    ON       73          24 blocks       24 blocks     44.30%      1327101
///     8    ON       80          24 blocks       24 blocks     44.28%      1327124
/// ```
///
/// The mask is worth 15 registers at depth 8 and 7 at depth 4, and it lifts
/// depth 8's register limit off 20 and back onto the 24-block cap. Masked depth
/// 8 sits exactly where masked depth 4 sits. Achieved occupancy is 44.28%
/// against 44.30%, `l1tex__t_requests.sum` is 1312768 for both, and the sector
/// counts differ by 23 out of 1.33 million (0.002%). Whatever separates them, it
/// is not occupancy and it is not traffic.
///
/// AND DEPTH 8 IS STILL SLOWER. `w4a16_swz_grid`, mask ON (`INK_MLIVE=1` of
/// `INK_M=16`), five processes per depth in ALTERNATED order (4,8,8,4,4,8,8,4,4,8)
/// with a memory settle between each, each process being min of 6 rounds of 20
/// pipelined launches over rotating buffers with 2 rounds discarded, spark2,
/// box locked and idle. `ratio` is row-major time / swizzled time and is PAIRED
/// inside each process -- the row-major arm is round-robined against the
/// swizzled one and is unaffected by depth, so it divides out anything that
/// drifts between processes:
///
/// ```text
///                             depth 4 ratio        depth 8 ratio    d4/d8
///   k=4096  25128 cubes (head)  1.58 [1.58-1.63]   1.36 [1.34-1.38]  1.169
///   k=4096   1024 cubes (sink)  1.42 [1.37-1.46]   1.34 [1.31-1.38]  1.055
///   k=2048    512 cubes (sink)  1.03 [0.99-1.08]   1.05 [0.94-1.06]  0.981
/// ```
///
/// Medians of five, brackets are the full range of the five. The head and the
/// 1024-cube sink are real losses -- 16.9% and 5.5% -- and at 512 cubes it is a
/// tie, which is what a shape where the permutation itself barely pays looks
/// like. The unpaired process-to-process floor on this harness, measured from
/// the same runs as the spread of the row-major arm (which depth cannot touch),
/// is 2.3% at the 1024-cube shape; the 5.5% clears it and the ratio removes it
/// anyway.
///
/// So the answer to "why 4 and not 8" is no longer the register table above. It
/// was the true reason while depth 8 ran at 37.41% occupancy; with the mask on,
/// occupancy is equal to two decimal places and depth 8 loses by MORE at the
/// head than the occupancy gap could ever have explained.
///
/// ## The schedule hypothesis, measured 2026-08-27: half right, and the half
/// that was wrong was the mechanism
///
/// This paragraph used to read "the warp spends longer at the head of each group
/// issuing eight k-tiles of loads before it can consume any, and the loads it is
/// adding are ones the memory system was already keeping busy at depth 4",
/// flagged as a HYPOTHESIS because nothing measured issue stalls. They are
/// measured now, and the ISSUE half is refuted while the MEMORY half is
/// confirmed with a mechanism it did not have.
///
/// `ncu` 2025.3.1 on `w4a16_linear_swz` launched by `w4a16_swz_grid` at
/// `INK_M=16`, `INK_MLIVE=1`, `k=4096`, `n=201024` (25128 cubes, the head grid),
/// one launch profiled, spark2, box locked and idle:
///
/// ```text
///   depth  L1 requests   L2 requests   sect/L2 req   DRAM sectors   long_sb   mio_thr
///     1     32163840      14961432        1.013        14485808      --        --
///     4     32163840      13465439        1.145        14482982      4.07      0.00
///     8     32163840      14963365        1.024        14484408      5.80      0.00
/// ```
///
/// `long_sb` and `mio_thr` are
/// `smsp__average_warps_issue_stalled_{long_scoreboard,mio_throttle}_per_issue_active`.
///
/// THE ISSUE BURST IS NOT WHAT COSTS. `mio_throttle` is 0.00 at both depths,
/// `lg_throttle` 0.01 against 0.07, `barrier` 0 (one warp per cube has nothing
/// to sync on), and the instruction counts are within 0.6%. A longer burst of
/// load instructions at the head of a group does not back the memory-IO queue up
/// at all on this part.
///
/// WHAT COSTS IS LOST MISS-MERGING. The L1 request count is BYTE-IDENTICAL at
/// all three depths -- 32163840, not approximately -- and so is the DRAM traffic.
/// What depth 4 buys, and depth 8 gives straight back, is the merging of those
/// single-sector L1 misses into shared 128-byte L2 requests: depth 4 issues
/// 13.465M L2 requests against depth 1's 14.961M, and depth 8 lands on 14.963M,
/// which is depth 1's number to within 0.01%. Depth 8 asks for exactly the same
/// bytes in 11.1% more transactions, and transactions are the currency this lane
/// is short of -- see the ceiling block above, where the whole missing third is
/// denominated in them. Time follows: free-running at the head, four alternated
/// processes per depth, 161.8 GB/s at depth 4 against 140.6 at depth 8 (-13.1%),
/// paired row/swz ratios 1.58-1.60 against 1.36-1.39.
///
/// Depth 4 therefore buys TWO different things and depth 8 keeps only one: the
/// memory-level parallelism that depth 1 lacks (depth 1 stalls hardest of the
/// three despite merging no worse than depth 8), and the miss-merging that
/// depth 8 loses. Why 8 loses it is not measured here; the shape of the answer
/// is a finite per-L1 pending-request window, since 8 k-tiles x 5 requests x 24
/// resident warps is 960 outstanding sector requests an SM against depth 4's
/// 480. That is the next hypothesis, and it is labelled as one.
///
/// The finding is narrower and solid: depth 8 is not register-bound any more, it
/// is not issue-throttled, it moves identical bytes in identical L1 requests, and
/// it is still not worth taking.
///
/// This is recorded rather than merely re-killed because the previous kill was
/// re-opened on good evidence exactly BECAUSE its stated reason had expired. A
/// door closed without its reason gets reopened; a door closed with the wrong
/// reason gets reopened the moment that reason moves.
///
/// Bit-identical to the row-major lane at depth 4: max deviation 0.000e0 over
/// 1024 outputs of a `[16, 256] x [64, 256]^T` product (`w4a16_swz_probe`).
pub const SWZ_UNROLL_DEFAULT: usize = 4;

/// [`SWZ_UNROLL_DEFAULT`], overridable for a sweep. Powers of two up to 8.
pub fn swz_unroll() -> usize {
    // Cached: this is read on every LAUNCH, which is inside the timed region of
    // every harness that measures this lane.
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("INK_W4A16_SWZ_UNROLL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|d| *d > 0 && d.is_power_of_two() && *d <= 8)
            .unwrap_or(SWZ_UNROLL_DEFAULT)
    })
}

/// Load the weight planes COALESCED and redistribute to the fragment lanes with
/// warp shuffles, instead of loading them fragment-shaped. Off by default.
///
/// ## What it is
///
/// The `m16n8k16` B fragment map gives lane `l` column `l >> 2`, so FOUR LANES
/// LAND ON THE SAME 32-BIT WORD. A fragment-shaped warp load therefore touches
/// 8 distinct words -- 32 bytes, one sector -- where a flat load of the same
/// instruction touches 16. That is the whole of the W4A16 head's missing third:
/// `w4a16_geom_ceiling` shows every arm moving the same 463 MB from DRAM within
/// 0.2% while the shipped lane spends 13.4M L2 requests against a coalesced
/// control's 3.6M for those same bytes. The lane is short of TRANSACTIONS, not
/// bytes.
///
/// This flag buys them back without touching the stored format. One warp step
/// reads 32 consecutive words of the shipped swizzle -- two whole
/// `(n_tile, k_tile)` blocks -- and lane `l` then pulls its own four words out
/// of them with four `plane_shuffle`s (`__shfl_sync` on the CUDA dialect),
/// filling the same `w_buf` the fragment-shaped loads filled. At the shipped
/// `kunroll = 4` that is 2 loads and 8 shuffles where there were 8 loads. The
/// scale plane gets the same treatment one granularity coarser (32 E4M3 bytes
/// is four k-tiles), and only when it is permuted -- the row-major scale plane
/// strides by `spr` per lane and is not a warp-contiguous read at all.
///
/// It is EXACT over the existing swizzle: same words, same slots, same order,
/// no shared memory, no barrier (the lane is one warp per cube, so
/// `stall_barrier` reads 0.00 and there is nothing to synchronise on).
///
/// ## What it measures on the SHIPPED lane (2026-08-27, spark2 and spark, both
/// boxes locked and idle)
///
/// The rungs that motivated it are the LOAD HALF -- no A operand, no
/// accumulator, no MMA -- and they are not a claim about this kernel. These are.
///
/// FRAMING RULE: GB/s over the head's 0.431 GiB weight table (0.383 GiB of
/// codes plus 0.048 GiB of E4M3 scales) of ONE `[201024, 4096]` operand, PER
/// LAUNCH of a `[16, 4096] x [201024, 4096]^T` product with ONE live row of 16,
/// one launch and one sync each, p50 of 10 warm reps of 12, arms round-robined
/// with the order reversed on odd reps (`w4a16_geom_ceiling`). Not a step
/// figure and not a two-node figure.
///
/// ```text
///   arm                        spark2 p50 ms   GB/s      spark p50 ms   GB/s
///   coalesced (the ceiling)        1.940      238.8         2.121      218.4
///   w4a16_linear_swz, flag OFF     3.121      148.4         3.093      149.7
///   w4a16_linear_swz, flag ON      2.658      174.2         2.642      175.3
/// ```
///
/// Medians of three processes a box. The PAIRED within-process ratio is the
/// only figure here that resolves ~1.1% -- both flag states run in ONE process
/// on one pair of buffers, ratioed rep by rep -- and over six processes across
/// both boxes its p50 is
///
/// ```text
///   flag ON / flag OFF   0.860   (0.818 .. 0.898 across the six)
/// ```
///
/// i.e. 1.16x, and the flag-off arm is the noisier of the two by a wide margin
/// (its p50 moves 144-155 GB/s between processes where the flag-on arm moves
/// 174-184).
///
/// AND IT IS A WIN AT EVERY CUBE COUNT, which is what makes it a candidate for
/// a global default rather than a head-only one. `w4a16_swz_grid`, which
/// pipelines 20 launches over rotating buffers precisely so a small shape
/// cannot be answered by the L2, min of 6 rounds after 2 discarded, k = 4096,
/// `m_pad = 16` with one live row, both boxes:
///
/// ```text
///   n         cubes    swz GB/s -> +shuffle    spark2   spark
///   4096        512     77.0 ->  78.9           1.03    1.06
///   8192       1024    130.5 -> 144.1           1.10    1.14
///   16384      2048    135.1 -> 153.9           1.14    1.16
///   32768      4096    142.6 -> 171.4           1.20    1.16
///   65536      8192    144.8 -> 174.8           1.21    1.23
///   131072    16384    141.3 -> 183.5           1.30    1.16
///   201024    25128    160.7 -> 186.5           1.16    1.22   <- the head
/// ```
///
/// Nothing regresses, including the 512-cube shape where the permutation itself
/// barely pays.
///
/// ## Why it costs nothing, in counters rather than in time
///
/// `ncu` 2025.3.1 on `w4a16_linear_swz` launched by `w4a16_geom_ceiling` at the
/// head shape, one launch profiled per arm, spark2, box locked and idle:
///
/// ```text
///   flag   reg  limit_reg  limit_blk   occ.    sect/req    L1 req    L2 req   DRAM sect
///   OFF     73     24         24      49.25%     1.00     32.164M   13.454M   14.483M
///   ON      61     32         24      49.49%     1.55     17.690M    5.451M   14.485M
/// ```
///
/// REGISTERS GO DOWN, 73 to 61. The staged coalesced words are the COMPRESSED
/// form -- four words hold what sixteen fragment words would -- so shuffling out
/// of a staging register just in time is a smaller live set than holding the
/// fragments. `launch__occupancy_limit_blocks` stays 24 and the register limit
/// moves OFF it (24 -> 32 blocks), so the ~80-register cliff that killed load
/// depth 8 is further away than before, not closer.
///
/// THE TRANSACTIONS ARE THE WHOLE OF IT, and the L1 arithmetic is exact rather
/// than approximate. Per launch the codes cost `25128 * 256 * 2` = 12.866M
/// requests fragment-shaped and `25128 * 128` = 3.216M coalesced; the scales
/// cost 6.433M and 1.608M; A is untouched at 12.866M. Predicted saving
/// 14.475M, measured 14.474M. `lts__d_sectors_fill_sysmem` is 14.485M sectors
/// (463.5 MB) either way, 0.014% apart -- the SAME BYTES in 2.47x fewer L2
/// requests.
///
/// The instruction stream gets SHORTER, not longer (870.6M -> 774.2M
/// `smsp__inst_executed.sum`), even though 19.3M warp shuffles were added: what
/// went with the 14.5M load instructions was their address arithmetic.
/// `stall_barrier` is 0.00 in both, `lg_throttle` 0.01 -> 0.00.
///
/// ## What it is worth PER STEP, which is why it is now ON by default
///
/// Every figure above is a PER-LAUNCH weight-read figure at `m_pad = 16`, and
/// what production spends is a step. So the default was held OFF until a step
/// had been measured. It has been.
///
/// FRAMING RULE: PER one decode step of the 42-layer two-node pipeline -- head
/// layers 0:21 on spark2, tail 21:42 on spark -- AT ctx 3732
/// (`~/refprompts/ctx3732.ids`), `INK_GEN=64`, `INK_KV=1`, split 21,
/// `--overlap`, 62 warm passes a process, AGAINST the same binary in the same
/// rep with the flag forced off. `scripts/pipe-bench.sh --order abba`, 16 reps
/// = 32 processes, ~77 s each, `scripts/gb10-lock.sh` held on both boxes and
/// both idle-gated; ONE binary, staged byte-identical on both boxes, so the
/// arms differ in the environment and in nothing else. NOT ncu, NOT nsys: a
/// free-running locked idle pair.
///
/// ```text
///   arm     reps   median ms/step   median tok/s
///   off      16        88.8            11.262
///   on       16        86.8            11.517
/// ```
///
/// The arms run back to back inside one rep, so the PAIRED ratio is the figure
/// and the unpaired medians are only its shadow:
///
/// ```text
///   on / off, per rep, ms/step   mean 0.9769   median 0.9764   16/16 reps < 1
///   sd of the paired ratio       0.77%   (THIS run's, over its own 16 pairs)
///   resolution, 2 sem            0.39%
///   95% CI                       -2.70% .. -1.92% of step time
/// ```
///
/// **-2.31% of step time, i.e. 1.0236x, against a resolution of 0.39%.** The
/// sd is quoted from this run's own 16 paired ratios rather than from
/// `pipe-bench.sh`'s 1.45% constant, which is a between-process figure for
/// arms that are not this tightly paired.
///
/// AND THE LANE ACCOUNTS FOR IT, which is what makes it a mechanism rather than
/// a coincidence. `nsys -t cuda` on BOTH halves of the same config, 16 steps,
/// warm passes only (the prefill pass and the two cold decode passes dropped):
///
/// ```text
///   half   swz launches/pass   lane ms/pass OFF -> ON
///   head        57  (19 MoE layers x 3)   4.433 -> 3.609   (-0.824)
///   tail        63  (21 MoE layers x 3)   4.910 -> 3.965   (-0.945)
///   both       120                        9.343 -> 7.574   (-1.769, -18.9%)
/// ```
///
/// The launch count is IDENTICAL in both arms (969 head, 1071 tail, over 17
/// passes) -- same shapes, same schedule, only the load form differs. The two
/// halves are SERIAL in this pipeline, so their savings add, and -1.769 ms
/// covers 86% of the -2.056 ms the free-running step measurement lost; the
/// 0.29 ms residue is inside that measurement's own 0.35 ms resolution band.
/// The step's own weights are the SINKS (`gate_up` [8192, 4096] and `down`
/// [4096, 2048] x 2 an MoE layer) and never the head, which `w4a16_bind` keeps
/// row-major while the approximate lane owns m = 1.
///
/// NOTE THE PRODUCTION LANE GAINS MORE THAN THE GRID PREDICTS: -18.9% against
/// the 1.10 and 0.99 `shf/swz` that `w4a16_swz_grid` reads at exactly these two
/// shapes. The grid pipelines 20 launches over rotating buffers with the SM to
/// itself, which is the lane at its throughput ceiling; production interleaves
/// it with `fp4_linear_grouped` and the attention lane, where the 73 -> 61
/// register drop buys resident blocks that the isolated harness had no use for.
/// Same change, two different currencies -- and the step is the one that is
/// spent.
///
/// `INK_W4A16_SWZ_SHUFFLE=0` remains the ablation, so both arms stay runnable
/// from one build and this measurement can be repeated without a rebuild.
///
/// `w4a16_swz_probe` gates the numerics: flag on against flag off, four
/// `(m_pad, live, k, swz_sc)` shapes, EVERY output compared including the
/// padding rows, 0 bits differ. It is a pure data-movement change and anything
/// but bit-identity would be a wrong source lane.
///
/// The 16 B/lane variant is worth ~3% more on the load rungs and is NOT
/// reachable from here: the current swizzle puts all sixteen wanted words in
/// one vector component, which sixteen shuffles cannot reach. It presumes a
/// re-swizzle and is a separate piece of work.
pub fn swz_shuffle() -> bool {
    // Cached for the same reason `swz_unroll` is: read on every LAUNCH, inside
    // the timed region of every harness that measures this lane.
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        // ON by default. `=0` (or `false`) is the ablation and is the only
        // thing that turns it off -- the arms have to stay reachable from one
        // build, because the step measurement above is worth nothing if the
        // next person cannot repeat it without a rebuild.
        std::env::var("INK_W4A16_SWZ_SHUFFLE")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true)
    })
}

/// [`w4a16_linear`] reading a B operand written by [`swizzle_w4a16_codes_into`].
///
/// `m_live` is `Some(m)` to MASK the A operand's padding rows — `m` being how
/// many of `m_pad`'s rows are real — and `None` to load them as before. It is
/// one argument and not a flag plus a count so that "mask off" cannot be said
/// with a row count that contradicts it, and so both arms are reachable from
/// ONE process: `Some`/`None` picks between two comptime kernel variants rather
/// than between two builds. Production reads [`live_row_mask`] for the choice;
/// that function carries the bit-identity argument and what the mask does and
/// does not buy.
///
/// The ONLY difference is the two global indices. Everything else — the
/// dequantise, the scale application, the accumulator, the output store — is
/// the same code, because the permutation moves bytes and changes nothing about
/// what they mean.
///
/// `redist` selects the COALESCED LOAD + WARP SHUFFLE form of the two weight
/// reads; [`swz_shuffle`] carries what it is for and what it costs.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_swz<AB: Scalar + Cast, S: Scalar, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    b: &Array<u32>,
    b_sc: &Array<S>,
    out: &mut Array<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] swz_sc: bool,
    #[comptime] kunroll: usize,
    #[comptime] mask_rows: bool,
    #[comptime] hi_dead: bool,
    #[comptime] redist: bool,
    scale: f32,
    m_live: u32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE);
    let wpb = comptime!(SWZ_BLOCK_CODES / 4);
    let groups = comptime!(k_tiles / kunroll);

    // The redistributed load's step geometry, all comptime. `bps` is how many
    // whole `(n_tile, k_tile)` blocks one 32-lane warp step of 32-bit words
    // covers on the codes plane, `sbps` the same on the E4M3 scale plane, and
    // the two `steps` are how many such steps one `kunroll` group needs. The
    // scale plane needs a coarser group (`32 / NTILE` k-tiles a step against
    // `32 / wpb`), which is why `redist_sc` carries its own divisibility test
    // rather than riding on `redist`; the launch refuses `redist` outright when
    // `kunroll` does not divide the codes step.
    let bps = comptime!(32 / (SWZ_BLOCK_CODES / 4));
    let bsteps = comptime!(kunroll / bps);
    let sbps = comptime!(32 / NTILE);
    let ssteps = comptime!(kunroll / sbps);
    let redist_sc = comptime!(redist && swz_sc && kunroll % sbps == 0);

    // `kunroll` k-tiles are LOADED before any of them is consumed, so the warp
    // carries `kunroll * 2` B sectors in flight instead of 2. That depth is what
    // the permutation took away: a row-major 32-byte sector spans four k-tiles,
    // so the row-major lane gets depth 4 for free out of an access pattern that
    // costs it 8x the requests. See `swizzle_pays`.
    //
    // Both buffer indices below are comptime (`u` and `i` are unrolled loop
    // variables), so these stay in registers -- a RUNTIME index would spill the
    // array to local memory, which is the hazard `e2m1_value` is written the way
    // it is to avoid.
    let mut w_buf = Array::<u32>::new(comptime!(kunroll * vc_b));
    let mut s_buf = Array::<f32>::new(kunroll);
    let mut a_buf = Array::<Vector<AB, NA>>::new(comptime!(kunroll * vc_a));

    for g in 0..groups {
        #[unroll]
        for u in 0..kunroll {
            let t = g * kunroll + u;
            let kbase = t * KTILE;
            #[unroll]
            for i in 0..vc_a {
                let (row, col) =
                    def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
                let gr = row as usize + m_base;
                let gc = col as usize + kbase;
                // `hi_dead` is the one that reaches the REGISTER BUDGET, and
                // this is the loop where that matters: at depth 4 `a_buf` is
                // sixteen slots, and under `hi_dead` eight of them are an
                // immediate zero the compiler can see, so the load, its address
                // arithmetic and its slot all go. The runtime `gr < m_live`
                // predicate cannot do that -- it stops a load ISSUING, not the
                // allocation of somewhere to put it. `live_row_mask` carries the
                // bit-identity argument; `live_arg` carries why `hi_dead` is
                // sound and what it assumes about the fragment map.
                if mask_rows {
                    if comptime!(hi_dead && (i & 1) == 1) {
                        a_buf[u * vc_a + i] = Vector::<AB, NA>::cast_from(0.0f32);
                    } else {
                        let mut v = Vector::<AB, NA>::cast_from(0.0f32);
                        if gr < m_live as usize {
                            v = a[(gr * size_k + gc) / a.vector_size()];
                        }
                        a_buf[u * vc_a + i] = v;
                    }
                } else {
                    a_buf[u * vc_a + i] = a[(gr * size_k + gc) / a.vector_size()];
                }
            }
            if comptime!(!redist) {
                #[unroll]
                for i in 0..vc_b {
                    // Written out of `position_of_nth` rather than assumed, so it
                    // tracks the target's own fragment layout: `row` is the k
                    // element, `col` the n column, and `swz_word_k16`'s `w` is
                    // `row / 8`.
                    let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
                    let w = row as usize / CODES_PER_WORD;
                    let blk = (n_tile * k_tiles + t) * wpb;
                    w_buf[u * vc_b + i] = b[blk + w * NTILE + col as usize];
                }
            }
            // One scale per k-tile, not one per fragment element. Both elements
            // of a fragment sit at `row` in 0..14 and `kbase` is a multiple of
            // KTILE = GROUP, so `(kbase + row) / GROUP` is `t` for every row --
            // the module header's second fact. The old form wrote it per element
            // and let the compiler notice; this says it once.
            if comptime!(!redist_sc) {
                let (_r0, c0) = def.position_of_nth(lane, 0u32, MatrixIdent::B);
                s_buf[u] = if swz_sc {
                    f32::cast_from(b_sc[(n_tile * k_tiles + t) * NTILE + c0 as usize])
                } else {
                    f32::cast_from(b_sc[(c0 as usize + n_base) * spr + kbase / GROUP])
                };
            }
        }

        // The SAME words, in the SAME `w_buf` slots, fetched coalesced and then
        // handed to the lanes that want them. The `m16n8k16` B map puts four
        // lanes on one 32-bit word (`col = lane >> 2`), so the fragment-shaped
        // load above spends a whole warp request on 8 distinct words = one
        // 32-byte sector, where a flat one covers 16. `w4a16_geom_ceiling`
        // prices both ends of that and the redistribution between them.
        //
        // One warp step reads 32 consecutive words of this n-tile's swizzled
        // plane, which is `bps` whole `(n_tile, k_tile)` blocks; lane `l` then
        // pulls its own `vc_b` words out of each of them by their step-local
        // index. Each `plane_shuffle` delivers one word to one lane and each
        // word is wanted by four lanes, so `bps * vc_b` shuffles a step is the
        // SATURATED count, not a chosen one.
        if comptime!(redist) {
            #[unroll]
            for s in 0..bsteps {
                let t0 = g * kunroll + s * bps;
                let v = b[(n_tile * k_tiles + t0) * wpb + lane as usize];
                #[unroll]
                for p in 0..bps {
                    #[unroll]
                    for i in 0..vc_b {
                        let (row, col) =
                            def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
                        let src = col
                            + (row / (CODES_PER_WORD as u32)) * (NTILE as u32)
                            + (p * wpb) as u32;
                        w_buf[(s * bps + p) * vc_b + i] = plane_shuffle(v, src);
                    }
                }
            }
        }
        // Scales, same shape: `NTILE` E4M3 bytes a block, so one 32-byte warp
        // step covers `32 / NTILE` k-tiles and every lane wants one byte out of
        // each. Only reachable on the permuted scale plane -- the row-major one
        // strides by `spr` per lane and is not a warp-contiguous read at all.
        if comptime!(redist_sc) {
            let (_r0, c0) = def.position_of_nth(lane, 0u32, MatrixIdent::B);
            #[unroll]
            for s in 0..ssteps {
                let t0 = g * kunroll + s * sbps;
                let sv = f32::cast_from(b_sc[(n_tile * k_tiles + t0) * NTILE + lane as usize]);
                #[unroll]
                for p in 0..sbps {
                    s_buf[s * sbps + p] = plane_shuffle(sv, c0 + (p * NTILE) as u32);
                }
            }
        }

        #[unroll]
        for u in 0..kunroll {
            let kbase = (g * kunroll + u) * KTILE;
            #[unroll]
            for i in 0..vc_a {
                reg_a[i] = a_buf[u * vc_a + i];
            }
            #[unroll]
            for i in 0..vc_b {
                let (row, _col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
                let word = w_buf[u * vc_b + i];
                let s = s_buf[u];
                let mut v = Vector::<AB, NA>::empty();
                #[unroll]
                for j in 0..vs_b {
                    let kk = row as usize + kbase + j;
                    let code = (word >> (4 * (kk % CODES_PER_WORD)) as u32) & 15u32;
                    v[j] = AB::cast_from(e2m1_value(code) * s);
                }
                reg_b[i] = v;
            }

            let d = def.execute(&reg_a, &reg_b, &acc);
            #[unroll]
            for i in 0..vc_c {
                acc[i] = d[i];
            }
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gr = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(gr * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// Launch [`w4a16_linear_swz`]. `swz_sc` says whether `b_sc` is permuted too.
///
/// The weight-load form comes from [`swz_shuffle`], i.e. from the environment.
/// A harness that must run BOTH forms in one process — which is the only way to
/// resolve a few percent on this part — calls [`w4a16_linear_swz_launch_redist`]
/// and says which one it means.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_swz_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    swz_sc: bool,
    scale: f32,
    m_live: Option<usize>,
) -> Handle {
    w4a16_linear_swz_launch_redist(
        client,
        a,
        b,
        b_sc,
        m_pad,
        k,
        n,
        swz_sc,
        scale,
        m_live,
        swz_shuffle(),
    )
}

/// [`w4a16_linear_swz_launch`] with the weight-load form said explicitly.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_swz_launch_redist<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    swz_sc: bool,
    scale: f32,
    m_live: Option<usize>,
    redist: bool,
) -> Handle {
    assert_eq!(
        m_pad % MTILE,
        0,
        "m_pad {m_pad} is not a multiple of {MTILE}"
    );
    let (mask_rows, hi_dead, live) = live_arg(m_pad, m_live);
    assert!(swizzleable(n, k), "[{n}, {k}] is not swizzleable");
    assert!(
        n / NTILE <= 65535,
        "{} n-tiles exceed the 65535 grid-y limit",
        n / NTILE
    );

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / bf16::cube_type().size_bits();
    let wpr = k / CODES_PER_WORD;
    let spr = k / GROUP;
    // Depth must divide the k loop exactly; a shape that does not divide takes
    // depth 1 rather than a remainder loop, because a remainder loop would be a
    // second code path through the dequantise for the sake of a few k-tiles.
    let kunroll = {
        let d = swz_unroll();
        if (k / KTILE) % d == 0 { d } else { 1 }
    };
    // The coalesced step is 32 words wide and a swizzled block is `wpb` of
    // them, so one step covers `32 / wpb` k-tiles and `kunroll` has to be a
    // whole number of steps. It is not a fallback worth writing a second load
    // path for: the only depth that fails it is 1, which is itself the fallback
    // for a k the group does not divide.
    let redist = redist && kunroll % (32 / (SWZ_BLOCK_CODES / 4)) == 0;

    unsafe {
        w4a16_linear_swz::launch::<bf16, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            ArrayArg::from_raw_parts(a.clone(), m_pad * k),
            ArrayArg::from_raw_parts(b.clone(), n * wpr),
            ArrayArg::from_raw_parts(b_sc.clone(), n * spr),
            ArrayArg::from_raw_parts(out.clone(), m_pad * n),
            k,
            n,
            swz_sc,
            kunroll,
            mask_rows,
            hi_dead,
            redist,
            scale,
            live,
        )
    };
    out
}

#[cfg(test)]
mod swizzle_k16_tests {
    use super::*;

    /// Every source byte lands somewhere, exactly once, on both planes.
    ///
    /// The failure this excludes is a colliding destination formula, which
    /// loses one byte and duplicates another and which no same-shaped output
    /// check would call an error. A non-square shape, so a transposed n/k
    /// cannot pass.
    /// The predicate agrees with the measured table at the four real consumers.
    ///
    /// Every one of them pays at the shipped load depth of 4; the history worth
    /// keeping is that sink `down` did NOT before it, and that the un-pipelined
    /// table would have declined it forever. What is left in the predicate is the
    /// edge of what has been measured, not a crossover.
    #[test]
    fn swizzle_pays_matches_the_measured_shapes() {
        // head [201024, 4096], 25128 cubes, measured 1.24x.
        assert!(swizzle_pays(201024, 4096));
        // sink gate_up [8192, 4096], 1024 cubes, measured 1.13-1.22x.
        assert!(swizzle_pays(8192, 4096));
        // dense g/u [16384, 4096], 2048 cubes, measured 1.25x.
        assert!(swizzle_pays(16384, 4096));

        // sink down [4096, 2048], 512 cubes: 0.88x UN-PIPELINED, 1.02x at the
        // shipped load depth of 4. Taken now, and this is the row that flipped.
        assert!(swizzle_pays(4096, 2048));
        // dense down [4096, 16384], the SAME 512 cubes: 1.25x, 1.48x at depth 4.
        assert!(swizzle_pays(4096, 16384));

        // Declined as UNMEASURED at depth 4, not as known-bad. 256 cubes read
        // 0.70x un-pipelined and has not been re-run; no weight in the model has
        // that shape, so nothing turns on it.
        assert!(!swizzle_pays(2048, 4096), "256 cubes not re-run at depth 4");
        assert!(!swizzle_pays(201024, 1024), "k below the measured floor");
    }

    #[test]
    fn the_k16_permutation_is_a_bijection_on_both_planes() {
        let (n, k) = (16usize, 128usize);
        let codes: Vec<u8> = (0..n * k / 2).map(|i| (i % 251) as u8).collect();
        let mut a = swizzle_w4a16_codes(&codes, n, k);
        let mut b = codes.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);

        let mut seen = vec![false; codes.len()];
        let kt = k / KTILE;
        for nt in 0..n / NTILE {
            for t in 0..kt {
                for col in 0..NTILE {
                    for w in 0..KTILE / CODES_PER_WORD {
                        let d = (nt * kt + t) * SWZ_BLOCK_CODES + swz_word_k16(col, w) * 4;
                        for x in 0..4 {
                            assert!(!seen[d + x], "destination {} written twice", d + x);
                            seen[d + x] = true;
                        }
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|v| *v),
            "some destination was never written"
        );

        let scales: Vec<u8> = (0..n * (k / GROUP)).map(|i| (i % 241) as u8).collect();
        let mut a = swizzle_w4a16_scales(&scales, n, k);
        let mut b = scales.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}

// ---------------------------------------------------------------------------
// The m16n8k16 B fragment map, off the device
// ---------------------------------------------------------------------------

/// Dump `position_of_nth` for the `m16n8k16` fragment, as the DEVICE answers it.
///
/// [`super::fp4gemm::fp4_frag_b_map`] does this for `m16n8k64`
/// (`new_scaled::<e4m3>`); this is its twin for the shape BOTH four-bit-head and
/// BF16 lanes actually issue. It is a separate dump and not a re-read of the
/// other because the two are different instructions: a different `k`, a
/// different operand width, a different constructor. The FP4 permutation was
/// derived from the first map, and nothing about that derivation transfers.
///
/// One `MmaDefinition` serves both consumers here:
/// [`w4a16_linear`] and [`super::bf16gemm::bf16_linear`] both construct
/// `MmaDefinition::<bf16, bf16, f32>::new(16, 8, 16)` — same types, same
/// constructor, same shape — so one dump settles the layout for 3.70 GiB/step
/// of BF16 traffic and the 0.43 GiB/step head at once.
///
/// Layout of `out` (u32 words):
///
/// ```text
///   0   .. 256   B:   [lane * 4 + i] -> (row, col)      i < 4
///   256 .. 512   A:   same indexing
///   512 .. 768   Acc: same indexing
///   768 ..       counts: see the writes below
/// ```
#[cube(launch)]
pub fn mma16_frag_map<AB: Scalar>(out: &mut Array<u32>) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    // B is indexed EXACTLY as `w4a16_linear` indexes it -- `i * vs_b`, no
    // packing factor -- so this dumps the addresses that kernel computes and
    // not a neighbouring convention.
    #[unroll]
    for i in 0..vc_b {
        let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
        out[(lane as usize * 4 + i) * 2] = row;
        out[(lane as usize * 4 + i) * 2 + 1] = col;
    }
    #[unroll]
    for i in 0..vc_a {
        let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
        out[256 + (lane as usize * 4 + i) * 2] = row;
        out[256 + (lane as usize * 4 + i) * 2 + 1] = col;
    }
    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        out[512 + (lane as usize * 4 + i) * 2] = row;
        out[512 + (lane as usize * 4 + i) * 2 + 1] = col;
    }
    if lane == 0 {
        out[768] = vc_b as u32;
        out[769] = vs_b as u32;
        out[770] = ec_b as u32;
        out[771] = vc_a as u32;
        out[772] = vs_a as u32;
        out[773] = ec_a as u32;
        out[774] = vc_c as u32;
        out[775] = vs_c as u32;
        out[776] = ec_c as u32;
        out[777] = pack as u32;
    }
}

/// Launch [`mma16_frag_map`] and return the raw `u32` dump.
pub fn mma16_frag_map_launch<R: Runtime>(client: &ComputeClient<R>) -> Handle {
    let out = client.empty(1024 * 4);
    unsafe {
        mma16_frag_map::launch::<bf16, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            ArrayArg::from_raw_parts(out.clone(), 1024),
        )
    };
    out
}
