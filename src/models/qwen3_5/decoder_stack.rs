//! Source-only assembly of the ONE shared WeMM multimodal decoder backbone.
//!
//! Prepared resident BF16 [1,T,4096], T1..256 -> 32 existing decoder blocks in
//! actual WeMM order -> shared final norm / last-token / L2 output boundary.
//! No tokenizer, gather, vision encoder, arbitrary position-ID tensor, padding,
//! mask, decode/cache API, LM head, host numerical path, or model admission.
//! Actual layer-0/layer-3 numerical failures remain unresolved by this wiring.
//!
//! Configuration is pinned to upstream/wemm/config.json SHA256
//! 34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004.
//! Each group is three GDN blocks followed by gated full attention. Eight
//! groups are structural model weights, NOT a catalogue of store facts.
//! All arithmetic is delegated to the unchanged resident GPU implementations.

use burn::tensor::DType;
use triblespace::core::repo::BlobStoreGet;
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use super::{
    embedding_boundary::{self, Boundary, CudaTensor, FinalNormSlot},
    full_attention::{self, Positions},
    gdn_decoder,
    gdn_mixer::GdnConfig,
};

pub const GROUPS: usize = 8;
pub const LAYERS: usize = GROUPS * 4;
pub const GDN_LAYERS: usize = GROUPS * 3;
pub const ATTENTION_LAYERS: usize = GROUPS;
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 12288;
pub const MAX_TOKENS: usize = 256;
/// 24 * 14 GDN roles + 8 * 11 attention roles + one shared final norm.
/// Allow additional registrations for prepared inputs and other session users.
/// This is a role count, NOT a resident-byte estimate or a new binder budget.
pub const WEIGHT_ROLES: usize = GDN_LAYERS * 14 + ATTENTION_LAYERS * 11 + 1;

/// Group g selects exact model layers 4g, 4g+1, 4g+2, 4g+3, in that order.
/// Slots retain the existing native BF16 rank types; IDs stay opaque. Resolve
/// them at the consuming typed model roles, not by enumerating the pile.
/// Supplying the correct layer/role handles is the caller's semantic contract.
pub struct GroupSlots {
    pub gdn: [gdn_decoder::Slots; 3],
    pub attention: full_attention::Slots,
}

pub struct Slots {
    pub groups: [GroupSlots; GROUPS],
    pub final_norm: FinalNormSlot,
}

struct Group {
    gdn: [gdn_decoder::Block; 3],
    attention: full_attention::Block,
}

/// Only bound immutable model weights are retained; prefill stores no cache or
/// observations in this object. Reuse one caller-owned binder for this entire
/// model/session, including repeated constructions. Dropping the stack does
/// NOT revoke CubeCL's runtime-owned external memory registrations.
pub struct Stack {
    // Keep the architecture's exact group count without copying its large
    // bound weight descriptors through every enclosing constructor's stack.
    groups: Box<[Group; GROUPS]>,
    endpoint: Boundary,
    // Same final-norm alias, not an extra model role. The frozen blocks hide
    // their weights, so this anchor permits input-client validation BEFORE L0.
    input_anchor: CudaTensor,
}

impl Stack {
    /// Bind exactly the explicit 425 typed roles through ONE supplied binder.
    ///
    /// # Safety
    /// The reader may acquire exact bytes for these fixed slots; every returned
    /// leaf must come from a genuine validated native pile observation. Every
    /// selected payload AND its preceding partial page must remain immutable
    /// and untruncated through CUDA runtime teardown. No mutable-file, generic
    /// heap blob, dtype conversion or legacy alias path is provided.
    ///
    /// This forwards the existing blocks'/endpoint's unsafe binding contract.
    /// A late error can retain earlier registrations; this is not transactional.
    /// The client-validation anchor clones the already-bound final norm. No
    /// second get can fetch/remap that content under another mmap owner:
    /// 425 binding calls for 425 roles, at most 425 weight registrations.
    pub unsafe fn from_pile<R: BlobStoreGet>(
        snapshot: &R, slots: Slots, aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        // SAFETY: forwarded unchanged.
        unsafe { Self::from_pile_with(snapshot, slots, aliases, super::gdn_mixer::DecoderPrecision::Bf16) }
    }

    /// [`Self::from_pile`] with every decoder linear projection bound for
    /// `precision` (see `gdn_mixer::bind_linear`); the final norm and every
    /// other weight bind as before.
    ///
    /// # Safety
    /// As [`Self::from_pile`].
    pub unsafe fn from_pile_with<R: BlobStoreGet>(
        snapshot: &R, slots: Slots, aliases: &mut CudaBf16Aliases,
        precision: super::gdn_mixer::DecoderPrecision,
    ) -> Result<Self, String> {
        let Slots { groups, final_norm } = slots;
        // SAFETY: same reader, same supplied binder, same immutable prefixes.
        let endpoint = unsafe { Boundary::from_pile(snapshot, final_norm, aliases)? };
        // Retain the actual bound client/device anchor without reading or
        // registering its source again through a possibly acquiring reader.
        let input_anchor = endpoint.weight_anchor().clone();
        check_tensor(&input_anchor, &input_anchor, &[HIDDEN])?;
        if input_anchor.handle.can_mut() {
            return Err("immutable final-norm alias required".into());
        }
        // Ephemeral construction scratch only. Retained structure has exactly
        // eight groups, not a configurable graph or a weight catalogue.
        let mut bound = Vec::with_capacity(GROUPS);
        for (index, group) in groups.into_iter().enumerate() {
            // SAFETY: forwards the caller's same genuine-prefix obligation.
            bound.push(unsafe { bind_group(snapshot, group, index * 4, aliases, precision)? });
        }
        let groups = bound.into_boxed_slice().try_into()
            .map_err(|_| "internal fixed group count mismatch".to_string())?;
        Ok(Self { groups, endpoint, input_anchor })
    }

    /// Put every decoder linear projection (all 32 layers' mixer, attention
    /// and MLP projections; never the token table, norms or the vision
    /// tower) on `precision`. Returns the NVFP4 plane bytes created.
    ///
    /// Every replacement is built before any is installed, so a failure
    /// partway leaves every projection exactly as it was bound, never a
    /// stack of mixed precision.
    pub fn quantize_projections(&mut self, precision: super::gdn_mixer::DecoderPrecision) -> Result<u64, String> {
        let mut built = Vec::new();
        for (at, w) in self.projections().into_iter().enumerate() {
            built.push(w.quantized(precision).map_err(|e| format!("decoder projection {at}: {e}"))?);
        }
        let mut bytes = 0;
        for (slot, new) in self.projections_mut().into_iter().zip(built) {
            if let Some(new) = new {
                bytes += new.bytes().1;
                *slot = new;
            }
        }
        Ok(bytes)
    }

    /// Every decoder linear projection, layer by layer.
    fn projections(&self) -> Vec<&super::gdn_mixer::Linear> {
        let mut all = Vec::new();
        for group in self.groups.iter() {
            for block in &group.gdn {
                all.extend(block.projections());
            }
            all.extend(group.attention.projections());
        }
        all
    }

    /// [`Self::projections`], in the same order.
    fn projections_mut(&mut self) -> Vec<&mut super::gdn_mixer::Linear> {
        let mut all = Vec::new();
        for group in self.groups.iter_mut() {
            for block in group.gdn.iter_mut() {
                all.extend(block.projections_mut());
            }
            all.extend(group.attention.projections_mut());
        }
        all
    }

    /// (projections, BF16 bytes, NVFP4 plane bytes) over every decoder linear
    /// projection: what the decoder's projection weights hold resident.
    pub fn projection_bytes(&self) -> (usize, u64, u64) {
        let (mut n, mut bf16, mut fp4) = (0, 0, 0);
        for w in self.projections() {
            let (b, f) = w.bytes();
            n += 1;
            bf16 += b;
            fp4 += f;
        }
        (n, bf16, fp4)
    }

    /// What every decoder linear projection runs, as bound, when they all
    /// run the same; `None` when they differ.
    pub fn decoder_bound(&self) -> Option<super::gdn_mixer::BoundProjection> {
        let mut kinds = self.projections().into_iter().map(|w| w.bound());
        let first = kinds.next()?;
        kinds.all(|kind| kind == first).then_some(first)
    }

    /// One fresh, unpadded sequence. Same existing affine three-axis positions
    /// reach all eight attention blocks. Their GPU producer computes
    /// p[axis,0,t] = start + axis_base[axis] + t*axis_step[axis].
    /// This is NOT arbitrary image/video position inference. Padding, masks,
    /// custom position-ID tensors and decode are absent, not silently ignored.
    ///
    /// Every invocation starts fresh. Per-block recurrent/conv/KV states are
    /// intentionally dropped after extracting hidden; no continuation state
    /// is returned. No host read, upload, synchronization or observer is added.
    ///
    /// The prepared tensor must have a valid CubeCL client/handle pair and
    /// established stream dependencies. Public descriptor checks cannot prove
    /// arbitrary forged handles or unsafe external stream ordering. Allocation,
    /// JIT/driver/asynchronous failures retain upstream panic/error behavior;
    /// Result::Ok is neither synchronization nor numerical admission.
    pub fn prefill_unpadded(
        &self, input: &CudaTensor, positions: Positions,
    ) -> Result<embedding_boundary::Output, String> {
        self.prefill_unpadded_observed(input, positions, |_, _| Ok(()))
    }

    /// Same executed path, with a diagnostic callback after every complete
    /// layer. Production's wrapper above installs no observer. A gate may
    /// synchronize/read witnesses here; the numerical operators are unchanged.
    pub fn prefill_unpadded_observed(
        &self, input: &CudaTensor, positions: Positions,
        mut observed: impl FnMut(usize, &CudaTensor) -> Result<(), String>,
    ) -> Result<embedding_boundary::Output, String> {
        let shape = input.meta.shape().as_slice();
        if shape.len() != 3 || shape[0] != 1 || !(1..=MAX_TOKENS).contains(&shape[1])
            || shape[2] != HIDDEN {
            return Err("stack requires prepared unpadded BF16 [1,T,4096], T1..256".into());
        }
        let hidden = self.layers(input, positions, &mut observed)?;
        self.endpoint.finish_unpadded(&hidden)
            .map_err(|e| format!("shared output boundary: {e}"))
    }
}

impl Stack {
    /// The 32 layers over a prepared BF16 [B,T,4096], B >= 1. Every operator
    /// is per row (norms, projections, MLP) or causal within its own sequence
    /// (convolution, DeltaNet scan, attention, positions by token index), so
    /// a sequence's first n rows do not depend on what follows them in its
    /// own row or on the other sequences: right padding needs no mask.
    fn layers(
        &self, input: &CudaTensor, positions: Positions,
        observed: &mut impl FnMut(usize, &CudaTensor) -> Result<(), String>,
    ) -> Result<CudaTensor, String> {
        let shape = input.meta.shape().as_slice();
        if shape.len() != 3 || shape[0] == 0 || !(1..=MAX_TOKENS).contains(&shape[1])
            || shape[2] != HIDDEN {
            return Err("stack requires prepared BF16 [B,T,4096], T1..256".into());
        }
        check_tensor(input, &self.input_anchor, shape)?;
        // Same bound as the existing full-attention producer, checked before
        // any of the preceding GDN blocks can launch. Callers here pass
        // batch_stride 0 (every sequence starts at 0); no position data or
        // model arithmetic is read on CPU.
        for axis in 0..3 {
            let end = (positions.start as u64)
                .checked_add(positions.axis_base[axis] as u64)
                .and_then(|v| v.checked_add(
                    (positions.axis_step[axis] as u64).checked_mul((shape[1] - 1) as u64)?
                )).ok_or("position overflow")?;
            if end > 16_777_216 {
                return Err("positions exceed exact F32 integer domain".into());
            }
        }

        let mut hidden = input.clone();
        for (group_index, group) in self.groups.iter().enumerate() {
            for (offset, block) in group.gdn.iter().enumerate() {
                let layer = group_index * 4 + offset;
                let gdn_decoder::Output { hidden: next, state } = block.prefill(&hidden, None)
                    .map_err(|e| format!("layer {layer} GDN prefill: {e}"))?;
                drop(state);
                hidden = next;
                observed(layer, &hidden)?;
            }
            let layer = group_index * 4 + 3;
            let full_attention::Output { hidden: next, state } =
                group.attention.prefill(&hidden, positions, None)
                    .map_err(|e| format!("layer {layer} attention prefill: {e}"))?;
            drop(state);
            hidden = next;
            observed(layer, &hidden)?;
        }
        // Hidden is PRE-final-norm. Finish exactly once; never feed an already
        // final-normalized HF hidden through this endpoint.
        Ok(hidden)
    }

    /// Right-padded batch of fresh sequences, each starting at position 0:
    /// `input` is [B,T,4096] and sequence b holds `lengths[b]` valid tokens
    /// at its front. Returns one [1,4096] embedding per sequence, each pooled
    /// at its own last valid token.
    pub fn prefill_padded(&self, input: &CudaTensor, lengths: &[usize]) -> Result<Vec<CudaTensor>, String> {
        let positions = Positions { start: 0, batch_stride: 0, axis_base: [0; 3], axis_step: [1; 3] };
        let hidden = self.layers(input, positions, &mut |_, _| Ok(()))?;
        self.endpoint.finish_padded(&hidden, lengths)
            .map_err(|e| format!("shared output boundary: {e}"))
    }
}

fn gdn_config() -> gdn_decoder::Config {
    gdn_decoder::Config {
        mixer: GdnConfig {
            hidden: HIDDEN, key_heads: 16, value_heads: 32, key_dim: 128,
            value_dim: 128, conv_kernel: 4, epsilon: 1e-6,
        },
        intermediate: INTERMEDIATE,
    }
}

fn attention_config() -> full_attention::Config {
    full_attention::Config {
        hidden: HIDDEN, intermediate: INTERMEDIATE, heads: 16, kv_heads: 4,
        head_dim: 256, rotary_dim: 64, sections: [11, 11, 10],
        theta: 10_000_000.0, epsilon: 1e-6, capacity: MAX_TOKENS,
        attention_bias: false,
    }
}

unsafe fn bind_group<R: BlobStoreGet>(
    snapshot: &R, slots: GroupSlots, first_layer: usize,
    aliases: &mut CudaBf16Aliases, precision: super::gdn_mixer::DecoderPrecision,
) -> Result<Group, String> {
    let [s0, s1, s2] = slots.gdn;
    // SAFETY: all calls forward from_pile's same snapshot/prefix lifetime.
    let g0 = unsafe { gdn_decoder::Block::from_pile_with(snapshot, s0, gdn_config(), aliases, precision) }
        .map_err(|e| format!("layer {first_layer}: {e}"))?;
    let g1 = unsafe { gdn_decoder::Block::from_pile_with(snapshot, s1, gdn_config(), aliases, precision) }
        .map_err(|e| format!("layer {}: {e}", first_layer + 1))?;
    let g2 = unsafe { gdn_decoder::Block::from_pile_with(snapshot, s2, gdn_config(), aliases, precision) }
        .map_err(|e| format!("layer {}: {e}", first_layer + 2))?;
    let attention = unsafe {
        full_attention::Block::from_pile_with(snapshot, slots.attention, attention_config(), aliases, precision)
    }.map_err(|e| format!("layer {}: {e}", first_layer + 3))?;
    Ok(Group { gdn: [g0, g1, g2], attention })
}

/// Descriptor/extent preflight, not validation of forged raw runtime handles.
fn check_tensor(t: &CudaTensor, anchor: &CudaTensor, shape: &[usize]) -> Result<(), String> {
    if t.meta.shape().as_slice() != shape || t.meta.strides().len() != shape.len()
        || t.dtype != DType::BF16 || t.qparams.is_some() || t.device != anchor.device
        || !std::ptr::eq(t.client.properties(), anchor.client.properties()) {
        return Err("wrong BF16 shape/dtype/device/client or quantized storage".into());
    }
    let mut extent = 1usize;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        if dim == 0 || (dim > 1 && t.meta.strides()[axis] != extent) {
            return Err("nonempty contiguous BF16 storage required".into());
        }
        extent = extent.checked_mul(dim).ok_or("tensor extent overflow")?;
    }
    let bytes = extent.checked_mul(2).ok_or("tensor byte extent overflow")?;
    let start = t.handle.offset_start.unwrap_or(0);
    let remaining = t.handle.size().checked_sub(start)
        .and_then(|n| n.checked_sub(t.handle.offset_end.unwrap_or(0)))
        .ok_or("tensor handle offsets outside allocation")?;
    if start % 2 != 0 || extent > u32::MAX as usize || remaining < bytes as u64 {
        return Err("unaligned/short BF16 storage or extent outside u32 domain".into());
    }
    Ok(())
}

impl Stack {
    /// Explicit positions, fresh B1 prefill with the processor's all-ones
    /// unpadded mask policy. No packed mask inference or continuation state.
    /// The affine producer and its existing call graph remain unchanged.
    pub fn prefill_positioned_unpadded_observed(
        &self, input: &CudaTensor, positions: &super::position_table::PositionTable,
        mut observed: impl FnMut(usize, &CudaTensor) -> Result<(), String>,
    ) -> Result<embedding_boundary::Output, String> {
        let shape = input.meta.shape().as_slice();
        if shape.len() != 3 || shape[0] != 1 || !(1..=MAX_TOKENS).contains(&shape[1])
            || shape[2] != HIDDEN {
            return Err("positioned stack requires unpadded BF16 [1,T,4096], T1..256".into());
        }
        check_tensor(input, &self.input_anchor, shape)?;
        positions.validate(input, shape[1])?;
        let mut hidden = input.clone();
        for (group_index, group) in self.groups.iter().enumerate() {
            for (offset, block) in group.gdn.iter().enumerate() {
                let layer = group_index * 4 + offset;
                let gdn_decoder::Output { hidden: next, state } = block.prefill(&hidden, None)
                    .map_err(|e| format!("layer {layer} GDN prefill: {e}"))?;
                drop(state);
                hidden = next;
                observed(layer, &hidden)?;
            }
            let layer = group_index * 4 + 3;
            hidden = group.attention.prefill_positioned(&hidden, positions)
                .map_err(|e| format!("layer {layer} positioned attention prefill: {e}"))?;
            observed(layer, &hidden)?;
        }
        self.endpoint.finish_unpadded(&hidden)
            .map_err(|e| format!("shared output boundary: {e}"))
    }
}

#[cfg(test)]
mod representation_tests {
    use super::*;

    #[test]
    fn fixed_groups_are_heap_backed_without_an_inline_array() {
        // Field-type evidence keeps the fixed architecture while making the
        // owning model handle small. No model/device is needed for this check.
        let _: fn(&Stack) -> &Box<[Group; GROUPS]> = |stack| &stack.groups;
        assert_eq!(
            std::mem::size_of::<Box<[Group; GROUPS]>>(),
            std::mem::size_of::<*const [Group; GROUPS]>(),
        );
    }
}
