//! Native, explicit-destination ingress for one already-packed operation.
//!
//! Only the current tensor is resident while it is put. Returned fragments
//! retain graph facts and small role strings, not tensor payloads. The caller
//! checks its explicitly selected collection's writer before ingress, composes
//! operations with `entity!`, and publishes the completed fragment last using
//! `model_collection::publish_model_fragment_into`. Errors may leave orphan
//! blobs; this module neither chooses storage nor publishes partial graphs.

use anybytes::Bytes;
use anyhow::{Context, Result, ensure};
use triblespace::core::blob::encodings::tensor::{
    TensorElement,
    elements::{BF16, E4M3, F32, NVFP4},
};
use triblespace::prelude::*;

use super::packing::{PackedFp8, PackedNvfp4};
use crate::{format::attrs, leaf};

/// Scale roles are not interchangeable. NVFP4's weight global remains inside
/// its typed payload; it is never replaced with one or turned into a postscale.
pub const INPUT_SCALE: &str = "activation-input-scale";
pub const FP8_WEIGHT_SCALE: &str = "fp8-weight-scale";

fn store<T: TensorElement, const R: usize>(
    pile: &mut Pile,
    dims: [u64; R],
    bytes: Bytes,
) -> Result<Fragment> {
    // Core's Tensor constructor multiplies dims unchecked. Admit arithmetic
    // before entering it; the encoding itself still owns length validation.
    let elements = dims
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(d))
        .context("native tensor dimension product overflow")?;
    usize::try_from(elements)?
        .checked_mul(T::BITS)
        .context("native tensor bit length overflow")?;
    let blob = leaf::leaf_blob::<T, R>(dims, bytes)?;
    let handle = pile
        .put(blob)
        .map_err(|e| anyhow::anyhow!("put native tensor: {e}"))?;
    Ok(entity! { leaf::leaf::<T, R>(): handle })
}

fn scale(pile: &mut Pile, role: &str, bytes: [u8; 4]) -> Result<Fragment> {
    // The packing contract normalizes either source scalar shape [] or [1]
    // to four unchanged little-endian bytes. Its native role is a rank-0 leaf.
    let weight = store::<F32, 0>(pile, [], Bytes::from_source(bytes.to_vec()))?;
    Ok(entity! { attrs::kind: role, attrs::weight*: weight })
}

/// `kind` names the operation's role (for example `mamba.in_proj`), not a
/// source-file path. Attach optional source labels afterwards, outside its core.
pub fn store_nvfp4(pile: &mut Pile, kind: &str, packed: PackedNvfp4) -> Result<Fragment> {
    let [out, input] = packed.logical_shape;
    ensure!(
        out > 0 && input > 0 && input % 16 == 0,
        "invalid native NVFP4 row geometry"
    );
    let weight = store::<NVFP4, 2>(
        pile,
        [out as u64, input as u64],
        Bytes::from_source(packed.payload),
    )?;
    let input_scale = scale(pile, INPUT_SCALE, packed.input_scale_le)?;
    Ok(entity! { attrs::kind: kind, attrs::weight*: weight, attrs::member*: input_scale })
}

/// E4M3FN stays `Tensor<E4M3, 2>`; neither its bytes nor its two distinct
/// scales pass through Mary's erased `Elem` or a floating-point conversion.
pub fn store_fp8(pile: &mut Pile, kind: &str, packed: PackedFp8) -> Result<Fragment> {
    let [out, input] = packed.logical_shape;
    ensure!(out > 0 && input > 0, "invalid native FP8 row geometry");
    let weight = store::<E4M3, 2>(
        pile,
        [out as u64, input as u64],
        Bytes::from_source(packed.e4m3fn),
    )?;
    let scales = scale(pile, INPUT_SCALE, packed.input_scale_le)?
        + scale(pile, FP8_WEIGHT_SCALE, packed.weight_scale_le)?;
    Ok(entity! { attrs::kind: kind, attrs::weight*: weight, attrs::member*: scales })
}

/// Preserve an auxiliary's original rank, shape and BF16 bits, including
/// signed zero/NaN payloads. This is storage, not a numerical admission gate.
pub fn store_bf16<const R: usize>(
    pile: &mut Pile,
    kind: &str,
    dims: [u64; R],
    bytes: Bytes,
) -> Result<Fragment> {
    let weight = store::<BF16, R>(pile, dims, bytes)?;
    Ok(entity! { attrs::kind: kind, attrs::weight*: weight })
}

/// Preserve an auxiliary's original rank, shape and little-endian F32 bits.
pub fn store_f32<const R: usize>(
    pile: &mut Pile,
    kind: &str,
    dims: [u64; R],
    bytes: Bytes,
) -> Result<Fragment> {
    let weight = store::<F32, R>(pile, dims, bytes)?;
    Ok(entity! { attrs::kind: kind, attrs::weight*: weight })
}
