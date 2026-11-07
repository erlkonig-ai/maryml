//! `fp4quant` — dynamic **NVFP4 activation quantization** as a CubeCL kernel.
//!
//! The weight side of Inkling arrives already quantized and only has to be
//! *decoded* ([`super::nvfp4`]). The activation side does not: `hf_quant_config`
//! asks for `"*input_quantizer": { num_bits: [2,1], block_sizes: {-1: 16,
//! type: "dynamic", scale_bits: [4,3]}, algorithm: "max" }`, i.e. the
//! activations must be quantized **at run time, per forward pass**, into the
//! same NVFP4 shape the weights are already in. That is this module.
//!
//! ## The recipe (one 16-element block along the contiguous axis)
//!
//! ```text
//! amax   = max |x_i|                      over the 16 elements
//! scale  = amax / 6.0                     6.0 = largest E2M1 magnitude
//! byte   = E4M3(scale)                    round-to-nearest-even
//! s      = decode(byte)                   the value actually stored
//! code_i = argmin_c |E2M1[c] - x_i / s|   round-to-nearest
//! ```
//!
//! There is **no second (`scale2`) level** here. The weights have one because
//! they are quantized offline against a whole tensor; a dynamic per-block
//! activation quantizer has only the block, so `scale` must land in E4M3's own
//! range (`2^-9 .. 448`) unaided. `scale` is clamped to 448 before the cast so
//! that overflow behaviour is this kernel's decision and not the conversion
//! instruction's — CubeCL emits `__nv_cvt_float_to_fp8(..., __NV_NOSAT, ...)`,
//! which would otherwise turn an overflowing scale into the E4M3 NaN pattern.
//!
//! ## Tie-breaking, stated rather than left to chance
//!
//! * **The E4M3 scale** rounds to nearest, ties to **even** significand — that
//!   is what `__nv_cvt_float_to_fp8` does, and `fp4quant_gate` reproduces it on
//!   the host by bracketing the 127 non-negative E4M3 patterns.
//! * **The E2M1 code** is chosen by the GB10's own
//!   `cvt.rn.satfinite.e2m1x2.f32`, which rounds to nearest with ties to the
//!   **even code**. That is not the same rule the seven-comparison `>=` ladder
//!   this kernel used to run applied — that one sent an exact tie *away from
//!   zero* — and the two part on four of the seven midpoints (`0.25 -> 0` vs
//!   `1`, `1.25 -> 2` vs `3`, `2.5 -> 4` vs `5`, `5.0 -> 6` vs `7`; on `0.75`,
//!   `1.75` and `3.5` rounding up already *is* the even code). Ties-to-even is
//!   what the hardware, `cuda_fp4.hpp`, and every NVFP4 producer built on them
//!   do, so it is the rule to be identical to. An exact midpoint requires the
//!   quotient `x/s` to land on a dyadic value with ≤3 significant bits, which
//!   is measure-zero for activation-shaped input: over 1,048,576 real elements
//!   zero codes differ between the two rules. On input drawn from a coarse
//!   dyadic grid 2.2% differ, every one by exactly one E2M1 step.
//! * The instruction also **saturates**: `|x/s| > 6` clamps to `±6.0` rather
//!   than overflowing, and NaN saturates to `±6.0` instead of failing every
//!   comparison and falling out as zero.
//! * A negative input that rounds to magnitude zero keeps its sign and becomes
//!   code `0x8` (`-0.0`), not code `0x0` — including a `-0.0` input, which the
//!   old ladder mapped to `0x0` because `-0.0 < 0.0` is false. Both codes
//!   decode to the same number.
//! * An all-zero block emits a zero scale byte and a zero reciprocal, so every
//!   element becomes `±0.0` and its code is `0x0`/`0x8` — no per-element `s > 0`
//!   guard, and no reliance on `0/0` producing a NaN.
//!
//! ## Output layout
//!
//! `codes` is `[rows, k/8]` `u32`: word `w` holds elements `8w..8w+7`, element
//! `i` at bits `4*(i%8)` — lowest index in the **low** nibble, the ordering
//! [`super::nvfp4`] settled against `compressed_tensors`. On a little-endian
//! host that is byte-identical to the checkpoint's packed `e2m1x2` rows, so the
//! same buffer can be bound straight into the NVFP4 tensor-core MMA (see
//! `nvfp4_mma_probe`) without a repack. `scales` is `[rows, k/16]` `e4m3`, one
//! byte per block, in the same row-major order.
//!
//! ## What the GB10 rewrite cost, measured
//!
//! Framing rule: **static** SASS instruction counts for one thread = one
//! 16-element NVFP4 block, `NOP`/`BRA` excluded. The source is the CUDA CubeCL
//! itself generated for `quantize_nvfp4_kernel` (captured by running
//! `minifloat_caps_probe` under `CUBECL_DEBUG_LOG`), assembled with CUDA 13.0
//! V13.0.88 `nvcc --gpu-architecture=sm_121a -lineinfo -cubin`. *Static*, not
//! wall time — nothing here was timed, and the box was shared. "body" is the
//! straight-line kernel through its last `EXIT`; "helper" is the out-of-line
//! correctly-rounded-division routine `ptxas` plants behind an `FCHK`, which
//! costs I-cache but is not executed on the normal path. The two must not be
//! added and quoted as one number.
//!
//! | lane | | body | helper | loads per block |
//! | ---- | -------- | ---: | ---: | --------------- |
//! | f32  | before   |  660 |   87 | 16 × `LDG.E`     |
//! | f32  | **after**|**170**| 132 | 2 × `LDG.E.256`  |
//! | BF16 | before   |  741 |   87 | 16 × `LDG.E.U16` |
//! | BF16 | **after**|**195**| 133 | 2 × `LDG.E.128`  |
//!
//! Three levers, all of them landing in the default path because each is either
//! free or measured bit-identical: the hardware `F2FP` conversion in place of
//! the ladder (−130 `FSETP`, −110 `SEL`), one hoisted reciprocal in place of
//! sixteen correctly-rounded divides (−16 `FCHK`/`CALL` pairs, −11 `MUFU`), and
//! an eight-wide line in place of scalar loads. The helper column grows because
//! `amax / 6` and `1.0 / s` are two differently-shaped divides where the old
//! kernel's seventeen were all the same shape and shared one routine.
//!
//! Behind the `q4` feature, which is what pulls the `cubecl` dependency in
//! (`cuda-backend` enables it). Gated by `fp4quant_gate`.

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::{e2m1x2, e4m3};

/// Logical elements covered by one E4M3 block scale (NVFP4's `group_size`).
pub const GROUP: usize = 16;
/// Largest representable E2M1 magnitude — the divisor in `scale = amax / 6`.
pub const FP4_MAX: f32 = 6.0;
/// Largest finite E4M3 magnitude; `scale` is clamped here before the cast.
pub const E4M3_MAX: f32 = 448.0;
/// Elements per packed `u32` word (eight nibbles).
pub const CODES_PER_WORD: usize = 8;

/// Elements per vectorized activation load, and the width of one output word.
///
/// Eight is not a compromise between the two element types — it is the widest
/// load each one has: for f32 the eight elements are 32 B and `ptxas` issues one
/// `LDG.E.256`, for BF16 they are 16 B and it issues one `LDG.E.128`. A block is
/// two loads either way, where sixteen scalar `LDG.E` used to be four times that
/// on the f32 lane and eight times on the BF16 one. The block base is 64 B (f32)
/// or 32 B (BF16) aligned, so every one of them is naturally aligned.
///
/// Eight is also exactly one output `u32`, which is why nothing has to be
/// shifted or or-ed together afterwards: eight scaled f32 cast straight to a
/// four-byte `Vector<e2m1x2, 4>` *is* the word.
pub const LINE: usize = 8;

/// One thread per 16-element block, this many threads per cube.
const CUBE_SIZE: u32 = 256;

/// One thread per 16-element block: reduce `amax`, emit the E4M3 scale byte and
/// the block's two packed `u32` code words.
///
/// A block is exactly two output words, so no two threads ever touch the same
/// word — the pack needs no atomics and no shared memory.
///
/// ## The encode is the hardware's, not a comparison ladder
///
/// `Vector::<e2m1x2, 4>::cast_from` of eight f32 lowers, through CubeCL's
/// `__nv_cvt_float2_to_fp4x2`, to four `cvt.rn.satfinite.e2m1x2.f32` — SASS
/// `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`, two f32 in and one byte holding
/// two E2M1 codes out. Eight of them cover the block's sixteen elements, in
/// place of the 130 `FSETP` + 110 `SEL` the seven-midpoint ladder cost. The
/// header's inline-PTX path is gated on `__CUDA_ARCH_FAMILY_SPECIFIC__`, which
/// only `sm_121a` defines; CubeCL passes `--gpu-architecture=sm_{arch}a` for
/// every arch ≥ 90, so this lane gets it. On plain `sm_121` the same header
/// silently *emulates* the conversion at ~88 SASS instructions, i.e. the `a`
/// suffix is the difference between a 3× win and a loss.
///
/// The pair packing is little-endian — the first f32 of a pair lands in the LOW
/// nibble — which is exactly the order [`super::nvfp4`] settled against
/// `compressed_tensors`, so the four bytes reinterpret straight to the `u32`.
///
/// ## Why the block is *one* divide and sixteen multiplies
///
/// The obvious spelling, `x_i / s` per element, is what this kernel used to be,
/// and it was the single most expensive thing in it: CUDA's `/` is correctly
/// rounded, so `ptxas` plants an `FCHK` + `CALL` slow-path pair *per element* —
/// seventeen of them counting `amax / 6` — plus eighteen `MUFU`. Hoisting one
/// reciprocal deletes all sixteen element divides in favour of sixteen `FMUL`.
///
/// `amax / 6.0` **stays a real divide**. It runs once per block rather than once
/// per element, so it is a sixteenth of the cost, and it is the one rounding the
/// whole block hangs on: the E4M3 scale byte is a bit-checked output, and
/// `amax * (1/6)` rounds twice and can land on the other side of an E4M3
/// midpoint. Exactness is cheap here and expensive per element, so it is bought
/// where it is cheap. (Measured, `sm_121a`: spending it costs 93 static SASS
/// instructions, all but ~12 of them in a never-executed out-of-line helper.)
///
/// The per-element reciprocal is not free of that concern either — `x * fl(1/s)`
/// rounds twice where `x / s` rounds once, so a quotient within ~1e-7 relative
/// of an E2M1 midpoint can fall the other way. `fp4quant_gate` counts those
/// near-boundary elements explicitly and checks every code bitwise against an
/// f64 host reference, so the claim is measured on real Inkling data rather than
/// assumed.
///
/// ## The block amax has no hardware assist on this chip, and does not need one
///
/// Probed against `ptxas` 13.0.88 rather than the ISA docs, which disagree with
/// it here: `redux.sync.max.f32` is a *die-family* gap, not a suffix gap — it
/// assembles to `CREDUX.MAX.F32` on `sm_100a`/`sm_103a` and is rejected
/// identically on `sm_120a`, `sm_121`, `sm_121a` and `sm_121f`. Nor is there an
/// f16 door around it: `redux` is 32-bit-integer-only on every architecture, and
/// while `red.global.max.noftz.v2.bf16` and
/// `cp.reduce.async.bulk…max.bf16` *do* exist where their f32 forms are
/// rejected, both are element-wise memory reductions, not horizontal ones. Every
/// reduction unit on GB10 is cross-*lane* (`REDUX`, `SHFL`, `ATOM`, `UBLKRED`);
/// none of them reduces a thread's own registers.
///
/// That is not a limitation to work around — it is why this layout is cheap. A
/// `redux` is one instruction for 32 elements; one `FMNMX` here is one
/// instruction for 32 independent *blocks*. Re-laying the kernel out as sixteen
/// threads per block to reach `redux` costs ~10× more warp-instruction slots per
/// block and drops the loads from `LDG.E.256` to `LDG.E.32`.
///
/// So the amax is software, written to cost as little as software can: `FMNMX`
/// takes an `|src|` operand modifier, so `max(|a|, |b|)` is one instruction and
/// the `abs` is free — where the old `if v < 0 { a = -v }` / `if a > amax` pair
/// spent an `FSETP` and a `SEL` each. Sixteen elements, fifteen `FMNMX`, as a
/// tree rather than a chain so the dependency depth is 4 instead of 16.
#[cube(launch_unchecked)]
fn quantize_nvfp4_kernel<E: Scalar + Cast>(
    x: &Array<Vector<E, Const<8>>>,
    codes: &mut Array<u32>,
    scales: &mut Array<e4m3>,
    blocks: usize,
) {
    let blk = ABSOLUTE_POS;
    if blk < blocks {
        let base = blk * 2;

        let l0 = Vector::<f32, Const<8>>::cast_from(x[base]);
        let l1 = Vector::<f32, Const<8>>::cast_from(x[base + 1]);

        let hi = max(l0.abs(), l1.abs());
        let a01 = max(max(hi[0], hi[1]), max(hi[2], hi[3]));
        let a23 = max(max(hi[4], hi[5]), max(hi[6], hi[7]));
        let amax = max(a01, a23);

        let sf = min(amax / 6.0, f32::new(448.0f32));
        let se = e4m3::cast_from(sf);
        // Round-trip through the stored byte: the codes must be chosen against
        // the scale the consumer will actually read back, not the f32 that was
        // rounded to produce it.
        let s = f32::cast_from(se);
        scales[blk] = se;

        // An all-zero block stores a zero scale byte; `r = 0` then sends every
        // element to `±0.0`, which is code `0x0`/`0x8` — the same number either
        // way — with no per-element guard and no `0/0`.
        let mut r = f32::new(0.0f32);
        if s > 0.0 {
            r = 1.0 / s;
        }
        let rv = Vector::<f32, Const<8>>::new(r);

        codes[blk * 2] = u32::reinterpret(Vector::<e2m1x2, Const<4>>::cast_from(l0 * rv));
        codes[blk * 2 + 1] = u32::reinterpret(Vector::<e2m1x2, Const<4>>::cast_from(l1 * rv));
    }
}

/// Quantize `x` (`[rows, k]`, f32, row-major, contiguous) to NVFP4 on device.
///
/// Returns `(codes, scales)`: `codes` is `[rows, k/8]` `u32` (element `i` of
/// word `w` at bits `4*(i%8)`, low nibble = lowest index), `scales` is
/// `[rows, k/16]` [`e4m3`] bytes. Both buffers are freshly allocated.
///
/// `k` must be a positive multiple of 64 (every Inkling activation width is);
/// `rows` is unconstrained. Blocks never straddle a row because `k % 16 == 0`
/// follows from `k % 64 == 0`.
pub fn quantize_nvfp4<R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    rows: usize,
    k: usize,
) -> (Handle, Handle) {
    quantize_nvfp4_as::<f32, R>(client, x, rows, k)
}

/// The same, reading a BF16 activation.
///
/// The quantizer's own arithmetic does not move: every element is widened to
/// f32 on the way in and the block amax, the E4M3 scale and the seven midpoint
/// comparisons are the f32 ones they were. What changes is the buffer the
/// producer had to write, and on a prefill that is the whole point -- the
/// stacked `[k * n, hidden]` activation this reads is the largest thing a
/// routed-expert layer holds.
///
/// It costs nothing in accuracy that the lane was not already paying: the
/// destination is FOUR BITS with one E4M3 scale per sixteen elements, so the
/// question is whether BF16's eight mantissa bits can miss an E2M1 code the f32
/// would have hit, and the codes are seven comparisons against midpoints two
/// orders of magnitude coarser than that.
pub fn quantize_nvfp4_bf16<R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    rows: usize,
    k: usize,
) -> (Handle, Handle) {
    quantize_nvfp4_as::<half::bf16, R>(client, x, rows, k)
}

/// [`quantize_nvfp4`] at a named input element type.
fn quantize_nvfp4_as<E: Scalar + Cast, R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    rows: usize,
    k: usize,
) -> (Handle, Handle) {
    assert!(k > 0 && k % 64 == 0 && rows > 0, "NVFP4 needs rows > 0 and k a positive multiple of 64");
    let n = rows * k;
    let codes = client.empty(n / CODES_PER_WORD * core::mem::size_of::<u32>());
    let scales = client.empty(n / GROUP);
    quantize_nvfp4_into::<E, R>(client, x, rows, k, &codes, &scales);
    (codes, scales)
}

/// [`quantize_nvfp4_bf16`] into caller-owned destinations: `rows` rows of
/// codes and scales written at the START of `codes` and `scales`, which may be
/// offset views ([`Handle::offset_start`]) into longer buffers. An appending
/// cache uses it to write new rows behind the ones it already holds, which it
/// never rewrites: the rows are independent, so the bytes are the ones
/// [`quantize_nvfp4_bf16`] would return for the same input. The views must
/// hold at least `rows * k / 8` `u32` and `rows * k / 16` bytes.
pub fn quantize_nvfp4_bf16_into<R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    rows: usize,
    k: usize,
    codes: &Handle,
    scales: &Handle,
) {
    quantize_nvfp4_into::<half::bf16, R>(client, x, rows, k, codes, scales)
}

fn quantize_nvfp4_into<E: Scalar + Cast, R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    rows: usize,
    k: usize,
    codes: &Handle,
    scales: &Handle,
) {
    assert!(
        k > 0 && k % 64 == 0,
        "k must be a positive multiple of 64, got {k}"
    );
    assert!(rows > 0, "rows must be non-zero");

    let n = rows * k;
    let blocks = n / GROUP;
    let words = n / CODES_PER_WORD;
    assert!(
        codes.size_in_used() >= (words * core::mem::size_of::<u32>()) as u64
            && scales.size_in_used() >= blocks as u64,
        "NVFP4 destinations shorter than {rows} rows of {k}"
    );

    let cubes = blocks.div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        quantize_nvfp4_kernel::launch_unchecked::<E, R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            // Length in LINES, not elements: `k % 64 == 0` makes `n % 4 == 0`.
            ArrayArg::from_raw_parts(x.clone(), n / LINE),
            ArrayArg::from_raw_parts(codes.clone(), words),
            ArrayArg::from_raw_parts(scales.clone(), blocks),
            blocks,
        );
    }
}

/// The eight representable E2M1 magnitudes, indexed by the code's low three
/// bits — the exact inverse of what [`quantize_nvfp4_kernel`] encodes.
///
/// Written as seven `>=` tests rather than as a lookup table because a table in
/// device memory is an indexed load per element. Sign last, so a code that
/// rounded to magnitude zero comes back as `-0.0` exactly as it was stored.
#[cube]
pub(crate) fn e2m1_value(code: u32) -> f32 {
    let a = f32::cast_from(code & 7);
    let mut m = f32::new(0.0f32);
    if a >= 0.5 {
        m = f32::new(0.5f32);
    }
    if a >= 1.5 {
        m = f32::new(1.0f32);
    }
    if a >= 2.5 {
        m = f32::new(1.5f32);
    }
    if a >= 3.5 {
        m = f32::new(2.0f32);
    }
    if a >= 4.5 {
        m = f32::new(3.0f32);
    }
    if a >= 5.5 {
        m = f32::new(4.0f32);
    }
    if a >= 6.5 {
        m = f32::new(6.0f32);
    }
    if f32::cast_from(code & 8) > 0.0 {
        m = -m;
    }
    m
}

/// [`e2m1_value`] with the ladder replaced by bit construction.
///
/// The seven nonzero E2M1 magnitudes are already IEEE-754 floats, spelled in
/// the wrong field widths: for `m = code & 7` in `1..8` the value is
/// `2^((m >> 1) - 1) * (1 + (m & 1) / 2)`, so the f32 exponent field is
/// `126 + (m >> 1)` and the mantissa is one bit at position 22. `m == 1` is
/// the single exception — `0.5` is `2^-1 * 1.0`, and its low bit is not a
/// mantissa — and `m == 0` is a signed zero. Both are selects, not branches.
///
/// # Why both exist
///
/// They compute the same eight numbers and the difference is entirely where
/// they are called from. [`dequantize_nvfp4_kernel`] decodes each element ONCE
/// per launch and is bound by the store, so the ladder's ~18 operations are
/// free there and its shape is the clearer statement of what a code means.
/// [`super::flash`]'s packed reader decodes each element once per PLANE per key
/// tile — 160 decodes a unit a tile at the decode shape — where it is the
/// innermost loop and ~7 operations against ~18 is the whole difference between
/// the arms.
///
/// The two cannot be allowed to drift, and the thing that stops them is not
/// this comment: `flash`'s `the_packed_reader_is_the_dequantising_reader_to_the_bit`
/// runs a page through the ladder and the same page through this, and demands
/// the attention outputs be equal to the bit.
#[cube]
pub(crate) fn e2m1_bits(code: u32) -> f32 {
    let m = code & 7;
    // The sign bit lands in bit 31, and on its own it is the signed zero that
    // `m == 0` decodes to — including the `-0.0` a value that rounded to zero
    // was stored as.
    let mut bits = (code & 8) << 28;
    if m > 0 {
        let mut frac = m & 1;
        if m == 1 {
            frac = 0;
        }
        bits |= ((126 + (m >> 1)) << 23) | (frac << 22);
    }
    f32::reinterpret(bits)
}

/// One thread per 16-element block, mirroring [`quantize_nvfp4_kernel`]: read
/// the block's E4M3 scale and its two packed `u32` code words, write sixteen
/// dense values.
///
/// The same one-block-is-two-words property that let the pack skip atomics lets
/// the unpack skip bounds arithmetic: a thread owns a contiguous sixteen and no
/// two threads share a word or a scale.
#[cube(launch_unchecked)]
fn dequantize_nvfp4_kernel<O: Scalar + Cast>(
    codes: &Array<u32>,
    scales: &Array<e4m3>,
    out: &mut Array<O>,
    blocks: usize,
) {
    let blk = ABSOLUTE_POS;
    if blk < blocks {
        let s = f32::cast_from(scales[blk]);
        let base = blk * 16;
        let w0 = codes[blk * 2];
        let w1 = codes[blk * 2 + 1];
        #[unroll]
        for i in 0..8usize {
            out[base + i] = O::cast_from(e2m1_value((w0 >> (4 * i) as u32) & 15) * s);
        }
        #[unroll]
        for i in 0..8usize {
            out[base + 8 + i] = O::cast_from(e2m1_value((w1 >> (4 * i) as u32) & 15) * s);
        }
    }
}

/// Undo [`quantize_nvfp4`]: `[rows, k/8]` codes plus `[rows, k/16]` E4M3 scales
/// back to a dense `[rows, k]` f32 buffer.
///
/// This is not a *lossless* inverse and cannot be — four bits went in. What it
/// is is the DEFINITION of what the stored codes mean, and the reason it exists
/// as a kernel rather than as a host loop is that the FP4 KV cache reads its
/// whole retained context back on every decode step. A host round trip there
/// would cost more than the four bits saved.
pub fn dequantize_nvfp4<R: Runtime>(
    client: &ComputeClient<R>,
    codes: &Handle,
    scales: &Handle,
    rows: usize,
    k: usize,
) -> Handle {
    dequantize_nvfp4_as::<f32, R>(client, codes, scales, rows, k)
}

/// The same, writing a BF16 buffer.
///
/// The KV cache's consumer holds its operands BF16 (see `attn_bf16`), so
/// widening to f32 on the way out only to narrow again at the matmul would
/// double what the dequant writes for nothing. The arithmetic inside the kernel
/// is f32 either way; only the store narrows.
pub fn dequantize_nvfp4_bf16<R: Runtime>(
    client: &ComputeClient<R>,
    codes: &Handle,
    scales: &Handle,
    rows: usize,
    k: usize,
) -> Handle {
    dequantize_nvfp4_as::<half::bf16, R>(client, codes, scales, rows, k)
}

/// [`dequantize_nvfp4`] at a named output element type.
fn dequantize_nvfp4_as<O: Scalar + Cast, R: Runtime>(
    client: &ComputeClient<R>,
    codes: &Handle,
    scales: &Handle,
    rows: usize,
    k: usize,
) -> Handle {
    assert!(
        k > 0 && k % 64 == 0,
        "k must be a positive multiple of 64, got {k}"
    );
    assert!(rows > 0, "rows must be non-zero");

    let n = rows * k;
    let blocks = n / GROUP;
    let words = n / CODES_PER_WORD;

    let out = client.empty(n * core::mem::size_of::<O>());

    let cubes = blocks.div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        dequantize_nvfp4_kernel::launch_unchecked::<O, R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(codes.clone(), words),
            ArrayArg::from_raw_parts(scales.clone(), blocks),
            ArrayArg::from_raw_parts(out.clone(), n),
            blocks,
        );
    }

    out
}

#[cfg(test)]
mod tests {
    /// [`e2m1_bits`]' arithmetic, on the host, so the algebra can be checked
    /// without a GPU.
    ///
    /// This is not a duplicate of the device test in [`super::super::flash`]:
    /// that one proves the two decoders agree AS LOWERED, this one proves the
    /// bit construction is the right formula in the first place. The `m == 1`
    /// line is the whole reason it exists — `0.5` is the one magnitude whose
    /// low bit is not its mantissa, and a formula that forgets it returns
    /// `0.75` and nothing else moves.
    fn e2m1_bits_host(code: u32) -> f32 {
        let m = code & 7;
        let mut bits = (code & 8) << 28;
        if m > 0 {
            let frac = if m == 1 { 0 } else { m & 1 };
            bits |= ((126 + (m >> 1)) << 23) | (frac << 22);
        }
        f32::from_bits(bits)
    }

    #[test]
    fn the_bit_construction_is_the_e2m1_ladder() {
        const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for code in 0u32..16 {
            let want = if code & 8 != 0 {
                -MAG[(code & 7) as usize]
            } else {
                MAG[(code & 7) as usize]
            };
            let got = e2m1_bits_host(code);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "code {code}: {got} is not {want} (signed zero included: \
                 a code that rounded to zero was stored as -0.0)"
            );
        }
    }
}
