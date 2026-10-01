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
use triblespace::core::{
    blob::{Blob, encodings::tensor::{Tensor as NativeTensor, elements::BF16}},
    repo::{BlobStoreGet, pile::PileSnapshot},
};
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
    groups: [Group; GROUPS],
    endpoint: Boundary,
    // Same final-norm alias, not an extra model role. The frozen blocks hide
    // their weights, so this anchor permits input-client validation BEFORE L0.
    input_anchor: CudaTensor,
}

impl Stack {
    /// Bind exactly the explicit 425 typed roles through ONE supplied binder.
    ///
    /// # Safety
    /// The snapshot must be a genuine validated native pile observation. Every
    /// selected payload AND its preceding partial page must remain immutable
    /// and untruncated through CUDA runtime teardown. No mutable-file, generic
    /// heap blob, dtype conversion or legacy alias path is provided.
    ///
    /// This forwards the existing blocks'/endpoint's unsafe binding contract.
    /// A late error can retain earlier registrations; this is not transactional.
    /// The final norm is bound twice through the SAME binder, from the same
    /// snapshot owner/offset, so its registration is reused: 426 successful
    /// binding calls for 425 roles, at most 425 distinct weight registrations.
    pub unsafe fn from_pile(
        snapshot: &PileSnapshot, slots: Slots, aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        let Slots { groups, final_norm } = slots;
        // Retain an actual binder-created client/device anchor. Never trust a
        // caller-supplied advertised device as proof of client identity.
        let blob: Blob<NativeTensor<BF16, 1>> =
            snapshot.get(final_norm).map_err(|e| format!("final norm: {e}"))?;
        // SAFETY: genuine immutable-prefix lifetime is required of the caller.
        let input_anchor = unsafe { aliases.bind_pile_leaf(blob)? };
        check_tensor(&input_anchor, &input_anchor, &[HIDDEN])?;
        if input_anchor.handle.can_mut() {
            return Err("immutable final-norm alias required".into());
        }
        // SAFETY: same snapshot, same supplied binder, same immutable prefix.
        let endpoint = unsafe { Boundary::from_pile(snapshot, final_norm, aliases)? };

        // Ephemeral construction scratch only. Retained structure has exactly
        // eight groups, not a configurable graph or a weight catalogue.
        let mut bound = Vec::with_capacity(GROUPS);
        for (index, group) in groups.into_iter().enumerate() {
            // SAFETY: forwards the caller's same genuine-prefix obligation.
            bound.push(unsafe { bind_group(snapshot, group, index * 4, aliases)? });
        }
        let groups = bound.try_into()
            .map_err(|_| "internal fixed group count mismatch".to_string())?;
        Ok(Self { groups, endpoint, input_anchor })
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
        check_tensor(input, &self.input_anchor, shape)?;
        // Same bound as the existing full-attention producer, checked before
        // any of the preceding GDN blocks can launch. B=1 makes batch_stride
        // irrelevant; no position data or model arithmetic is read on CPU.
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
        self.endpoint.finish_unpadded(&hidden)
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

unsafe fn bind_group(
    snapshot: &PileSnapshot, slots: GroupSlots, first_layer: usize,
    aliases: &mut CudaBf16Aliases,
) -> Result<Group, String> {
    let [s0, s1, s2] = slots.gdn;
    // SAFETY: all calls forward from_pile's same snapshot/prefix lifetime.
    let g0 = unsafe { gdn_decoder::Block::from_pile(snapshot, s0, gdn_config(), aliases) }
        .map_err(|e| format!("layer {first_layer}: {e}"))?;
    let g1 = unsafe { gdn_decoder::Block::from_pile(snapshot, s1, gdn_config(), aliases) }
        .map_err(|e| format!("layer {}: {e}", first_layer + 1))?;
    let g2 = unsafe { gdn_decoder::Block::from_pile(snapshot, s2, gdn_config(), aliases) }
        .map_err(|e| format!("layer {}: {e}", first_layer + 2))?;
    let attention = unsafe {
        full_attention::Block::from_pile(snapshot, slots.attention, attention_config(), aliases)
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
