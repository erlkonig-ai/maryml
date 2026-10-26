//! Native Nemotron operation graph, queried at use in one frozen observation.
//!
//! The root's `member` edges select indexed `nemotron-h-layer` containers.
//! Each layer contains operations and/or indexed `nemotron-h-expert`
//! containers, whose members are operations. An operation has `kind` (its
//! semantic role), `weight` (a typed leaf), and scale-role `member` nodes.
//! Compose these ordinary relations with `entity!`/`Fragment`; no name catalog,
//! synthesized lookup IDs, closed fact-set check or singleton rule is needed.
//!
//! The projection queries bind every scale and weight through the SAME
//! operation. Alternatives remain alternatives, including several same-role
//! operations. Missing typed facts are a negative match; a matched handle with
//! unavailable/invalid bytes is an error, not a silently skipped candidate.
//! These APIs establish storage semantics only, not compute/KV policy.

use super::ingress::native::{FP8_WEIGHT_SCALE, INPUT_SCALE};
use crate::{format::attrs, leaf, model_collection::ModelSnapshot};
use anyhow::{Context, Result, ensure};
use triblespace::core::blob::{
    Blob,
    encodings::tensor::{
        MAX_RANK, TENSOR_HEADER_LEN, Tensor, TensorElement, TensorView,
        elements::{E4M3, F32, NVFP4},
    },
};
use triblespace::prelude::*;

pub const LAYER: &str = "nemotron-h-layer";
pub const EXPERT: &str = "nemotron-h-expert";

pub fn layers<S>(
    snapshot: &ModelSnapshot<S>,
    root: Id,
    index: u64,
) -> impl Iterator<Item = Id> + '_ {
    find!(layer: Id, pattern!(snapshot.facts(), [
        { root @ attrs::member: ?layer },
        { ?layer @ attrs::kind: LAYER, attrs::index: index }
    ]))
}

pub fn experts<S>(
    snapshot: &ModelSnapshot<S>,
    layer: Id,
    index: u64,
) -> impl Iterator<Item = Id> + '_ {
    find!(expert: Id, pattern!(snapshot.facts(), [
        { layer @ attrs::member: ?expert },
        { ?expert @ attrs::kind: EXPERT, attrs::index: index }
    ]))
}

/// Views borrow the snapshot's mapped blobs; no dequantization or weight copy.
pub struct Nvfp4Projection {
    pub operation: Id,
    pub weight: TensorView,
    pub input_scale: TensorView,
}

pub struct Fp8Projection {
    pub operation: Id,
    pub weight: TensorView,
    pub input_scale: TensorView,
    pub weight_scale: TensorView,
}

fn read<T: TensorElement, const R: usize>(
    store: &impl BlobStoreGet,
    handle: Inline<inlineencodings::Handle<Tensor<T, R>>>,
) -> Result<TensorView> {
    let blob: Blob<Tensor<T, R>> = store
        .get(handle)
        .map_err(|e| anyhow::anyhow!("read native tensor: {e}"))?;
    // Guard core's unchecked dim product before its ordinary TensorView
    // decoder validates the payload. No second tensor format is introduced.
    ensure!(
        R <= MAX_RANK && blob.bytes.len() >= TENSOR_HEADER_LEN,
        "invalid native tensor header"
    );
    let elements = blob.bytes[..R * 8]
        .chunks_exact(8)
        .try_fold(1u64, |n, d| {
            n.checked_mul(u64::from_le_bytes(
                d.try_into().expect("eight-byte dimension"),
            ))
        })
        .context("native tensor dimension product overflow")?;
    usize::try_from(elements)?
        .checked_mul(T::BITS)
        .context("native tensor bit length overflow")?;
    leaf::read_leaf(blob)
}

/// `parent` is a selected layer, expert, or the root for global operations.
pub fn nvfp4<'a, S: BlobStoreGet>(
    snapshot: &'a ModelSnapshot<S>,
    parent: Id,
    kind: &'a str,
) -> impl Iterator<Item = Result<Nvfp4Projection>> + 'a {
    find!((op: Id, w: Inline<inlineencodings::Handle<Tensor<NVFP4, 2>>>,
    input: Inline<inlineencodings::Handle<Tensor<F32, 0>>>),
    pattern!(snapshot.facts(), [
        { parent @ attrs::member: ?op },
        { ?op @ attrs::kind: kind, attrs::weight: _?weight, attrs::member: _?input },
        { _?weight @ leaf::leaf::<NVFP4, 2>(): ?w },
        { _?input @ attrs::kind: INPUT_SCALE, attrs::weight: _?input_leaf },
        { _?input_leaf @ leaf::leaf::<F32, 0>(): ?input }
    ]))
    .map(move |(operation, w, input)| {
        Ok(Nvfp4Projection {
            operation,
            weight: read(snapshot.store(), w)?,
            input_scale: read(snapshot.store(), input)?,
        })
    })
}

/// A named reader policy for repeated on-demand execution: the least opaque
/// (operation, weight handle, input-scale handle) tuple in this observation.
/// Select BEFORE loading payloads. This is not validation of ID derivation,
/// a singleton constraint or a claim that one alternative is more authoritative.
pub fn nvfp4_minimum<S: BlobStoreGet>(
    snapshot: &ModelSnapshot<S>, parent: Id, kind: &str,
) -> Option<Result<Nvfp4Projection>> {
    find!((op: Id, w: Inline<inlineencodings::Handle<Tensor<NVFP4, 2>>>,
    input: Inline<inlineencodings::Handle<Tensor<F32, 0>>>),
    pattern!(snapshot.facts(), [
        { parent @ attrs::member: ?op },
        { ?op @ attrs::kind: kind, attrs::weight: _?weight, attrs::member: _?input },
        { _?weight @ leaf::leaf::<NVFP4, 2>(): ?w },
        { _?input @ attrs::kind: INPUT_SCALE, attrs::weight: _?input_leaf },
        { _?input_leaf @ leaf::leaf::<F32, 0>(): ?input }
    ]))
    .min()
    .map(|(operation, w, input)| Ok(Nvfp4Projection {
        operation, weight: read(snapshot.store(), w)?, input_scale: read(snapshot.store(), input)?,
    }))
}

pub fn fp8<'a, S: BlobStoreGet>(
    snapshot: &'a ModelSnapshot<S>,
    parent: Id,
    kind: &'a str,
) -> impl Iterator<Item = Result<Fp8Projection>> + 'a {
    find!((op: Id, w: Inline<inlineencodings::Handle<Tensor<E4M3, 2>>>,
    input: Inline<inlineencodings::Handle<Tensor<F32, 0>>>,
    scale: Inline<inlineencodings::Handle<Tensor<F32, 0>>>),
    pattern!(snapshot.facts(), [
        { parent @ attrs::member: ?op },
        { ?op @ attrs::kind: kind, attrs::weight: _?weight,
            attrs::member: _?input, attrs::member: _?scale },
        { _?weight @ leaf::leaf::<E4M3, 2>(): ?w },
        { _?input @ attrs::kind: INPUT_SCALE, attrs::weight: _?input_leaf },
        { _?input_leaf @ leaf::leaf::<F32, 0>(): ?input },
        { _?scale @ attrs::kind: FP8_WEIGHT_SCALE, attrs::weight: _?scale_leaf },
        { _?scale_leaf @ leaf::leaf::<F32, 0>(): ?scale }
    ]))
    .map(move |(operation, w, input, scale)| {
        Ok(Fp8Projection {
            operation,
            weight: read(snapshot.store(), w)?,
            input_scale: read(snapshot.store(), input)?,
            weight_scale: read(snapshot.store(), scale)?,
        })
    })
}

/// Typed auxiliary query; callers request the actual source rank and encoding.
pub fn auxiliary<'a, T: TensorElement, const R: usize, S: BlobStoreGet>(
    snapshot: &'a ModelSnapshot<S>,
    parent: Id,
    kind: &'a str,
) -> impl Iterator<Item = Result<TensorView>> + 'a {
    auxiliary_operations::<T, R, S>(snapshot, parent, kind)
        .map(|selected| selected.map(|(_, view)| view))
}

/// Same typed alternatives, retaining the selected opaque operation identity.
pub fn auxiliary_operations<'a, T: TensorElement, const R: usize, S: BlobStoreGet>(
    snapshot: &'a ModelSnapshot<S>,
    parent: Id,
    kind: &'a str,
) -> impl Iterator<Item = Result<(Id, TensorView)>> + 'a {
    find!((op: Id, w: Inline<inlineencodings::Handle<Tensor<T, R>>>), pattern!(snapshot.facts(), [
        { parent @ attrs::member: ?op },
        { ?op @ attrs::kind: kind, attrs::weight: _?leaf },
        { _?leaf @ leaf::leaf::<T, R>(): ?w }
    ]))
    .map(move |(op, w)| Ok((op, read(snapshot.store(), w)?)))
}
