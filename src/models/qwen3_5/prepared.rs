//! Finite prepared-token leg of the ONE WeMM multimodal model.
//!
//! Caller-prepared integer IDs (including the final embedding token) -> native
//! BF16 GPU table gather -> all 32 decoder layers -> shared embedding endpoint.
//! B1, unpadded T1..256, ordinary sequential positions only. No tokenization,
//! vision scatter, mask, batching, continuation, Files integration or claim of
//! numerical admission. Existing layer error budgets remain obligations.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use triblespace::core::blob::{Blob, encodings::tensor::{Tensor as NativeTensor, elements::BF16}};
use triblespace::prelude::{BlobStoreGet, Id, TribleSet};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use super::{
    decoder_stack::{self, GroupSlots, Stack}, embedding_boundary,
    full_attention, gdn_decoder, gdn_mixer::GdnSlots, roles,
};

pub const VOCAB: usize = 248_078;
/// Upstream tokenizer.json's `<embedding>`, not config's EOS 248044.
pub const EMBEDDING_TOKEN: u32 = 248_077;
pub const WEIGHT_ROLES: usize = decoder_stack::WEIGHT_ROLES + 1;
pub type CudaTensor = CubeTensor<CudaRuntime>;

pub struct PreparedDecoder {
    token_embedding: CudaTensor,
    stack: Stack,
}

pub struct Output {
    /// GPU gather witness; no host read is performed by this API.
    pub gathered: CudaTensor,
    pub endpoint: embedding_boundary::Output,
}

impl PreparedDecoder {
    /// Bind the selected model's 426 native BF16 roles through one caller-owned
    /// binder. Required path/blob faults remain errors; unrelated facts do not.
    ///
    /// # Safety
    /// `snapshot` may acquire exact bytes but must preserve the selected facts
    /// and return leaves from genuine validated pile prefixes. Payloads AND their
    /// preceding partial pages stay immutable and untruncated until the CUDA
    /// runtime releases its registrations. Dropping this object is insufficient.
    /// A failure can retain registrations in the supplied binder/runtime.
    pub unsafe fn from_pile<R: BlobStoreGet>(
        facts: &TribleSet, snapshot: &R, root: Id,
        aliases: &mut CudaBf16Aliases,
        mut selected: impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        let embedding = roles::resolve(facts, snapshot, root,
            "model.language_model.embed_tokens.weight", &[VOCAB as u64, 4096], &mut selected)?;
        let final_norm = roles::resolve(facts, snapshot, root,
            "model.language_model.norm.weight", &[4096], &mut selected)?;
        // Construction scratch has exactly the architecture's eight groups.
        let mut groups = Vec::with_capacity(decoder_stack::GROUPS);
        for group in 0..decoder_stack::GROUPS {
            groups.push(GroupSlots {
                gdn: [
                    gdn_slots(facts, snapshot, root, group * 4, &mut selected)?,
                    gdn_slots(facts, snapshot, root, group * 4 + 1, &mut selected)?,
                    gdn_slots(facts, snapshot, root, group * 4 + 2, &mut selected)?,
                ],
                attention: attention_slots(facts, snapshot, root, group * 4 + 3, &mut selected)?,
            });
        }
        let groups = groups.try_into().map_err(|_| "internal group count")?;
        let blob: Blob<NativeTensor<BF16, 2>> = snapshot.get(embedding)
            .map_err(|e| format!("embedding leaf: {e}"))?;
        // SAFETY: forwarded genuine immutable-prefix contract, one binder.
        let token_embedding = unsafe { aliases.bind_pile_leaf(blob)? };
        if token_embedding.meta.shape().as_slice() != [VOCAB, 4096]
            || token_embedding.dtype != DType::BF16 || token_embedding.handle.can_mut() {
            return Err("immutable BF16 token embedding [248078,4096] required".into());
        }
        let stack = unsafe { Stack::from_pile(snapshot,
            decoder_stack::Slots { groups, final_norm }, aliases)? };
        Ok(Self { token_embedding, stack })
    }

    /// IDs are control input, not CPU model tensors. Only at most 1 KiB of
    /// integer IDs is uploaded. The weight lookup and ALL model arithmetic run
    /// on GPU; BF16 weight bytes never pass through F16 or host float buffers.
    /// The supplied sequence already includes `<embedding>`; nothing is added
    /// or truncated. Image/video tokens are refused until scatter is wired.
    pub fn embed_unpadded(&self, ids: &[u32]) -> Result<Output, String> {
        self.embed_unpadded_observed(ids, |_, _| Ok(()))
    }

    /// Identical path with explicit per-layer gate observation. The ordinary
    /// API performs no readbacks; observer effects belong to its caller.
    pub fn embed_unpadded_observed(
        &self, ids: &[u32], observed: impl FnMut(usize, &CudaTensor) -> Result<(), String>,
    ) -> Result<Output, String> {
        validate_ids(ids)?;
        let weight = &self.token_embedding;
        let controls: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let ids_handle = weight.client.create_from_slice(&controls);
        let count = ids.len() * decoder_stack::HIDDEN;
        let out = weight.client.empty(count * 2);
        let dim = CubeDim::new_1d(64);
        // Every id and all fixed table/output extents were checked before the
        // dispatch. Raw table alias is never an output and is immutable.
        unsafe { gather::launch_unchecked::<CudaRuntime>(
            &weight.client, cubecl::calculate_cube_count_elemwise(&weight.client, count, dim), dim,
            ArrayArg::from_raw_parts(weight.handle.clone(), VOCAB * decoder_stack::HIDDEN),
            ArrayArg::from_raw_parts(ids_handle, ids.len()),
            ArrayArg::from_raw_parts(out.clone(), count), count,
        ); }
        let gathered = CubeTensor::new_contiguous(weight.client.clone(), weight.device.clone(),
            [1, ids.len(), decoder_stack::HIDDEN].as_slice().into(), out, DType::BF16);
        let endpoint = self.stack.prefill_unpadded_observed(&gathered, full_attention::Positions {
            start: 0, batch_stride: 0, axis_base: [0; 3], axis_step: [1; 3],
        }, observed)?;
        Ok(Output { gathered, endpoint })
    }
}

// Same bound model, not independently selected text weights for another
// embedder. The original affine gather and its public execution stay intact.
impl PreparedDecoder {
    pub(crate) fn embedding_table(&self) -> &CudaTensor { &self.token_embedding }
    pub(crate) fn decoder_stack(&self) -> &Stack { &self.stack }
}

pub fn validate_ids(ids: &[u32]) -> Result<(), String> {
    if !(1..=decoder_stack::MAX_TOKENS).contains(&ids.len()) {
        return Err("prepared WeMM input requires 1..=256 unpadded integer IDs".into());
    }
    if ids.last() != Some(&EMBEDDING_TOKEN) {
        return Err("prepared single-sequence input must end with <embedding> ID248077".into());
    }
    if ids.iter().any(|&id| id as usize >= VOCAB) {
        return Err("token ID outside the actual WeMM vocabulary".into());
    }
    if ids.iter().any(|id| matches!(id, 248_053 | 248_054 | 248_056 | 248_057)) {
        return Err("vision tokens need GPU scatter and image MRoPE, not this prepared decoder leg".into());
    }
    Ok(())
}

#[cube(launch_unchecked)]
fn gather(table: &Array<bf16>, ids: &Array<u32>, out: &mut Array<bf16>, count: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < count {
        let row = ids[i / 4096] as usize;
        out[i] = table[row * 4096 + i % 4096];
    }
}

fn gdn_slots<R: BlobStoreGet>(
    facts: &TribleSet, snapshot: &R, root: Id, layer: usize,
    selected: &mut impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
) -> Result<gdn_decoder::Slots, String> {
    let prefix = format!("model.language_model.layers.{layer}.");
    macro_rules! role { ($name:literal, $shape:expr) => {
        roles::resolve(facts, snapshot, root, &format!("{prefix}{}", $name), &$shape, selected)?
    }; }
    Ok(gdn_decoder::Slots {
        input_norm: role!("input_layernorm.weight", [4096]),
        post_norm: role!("post_attention_layernorm.weight", [4096]),
        gate: role!("mlp.gate_proj.weight", [12288, 4096]),
        up: role!("mlp.up_proj.weight", [12288, 4096]),
        down: role!("mlp.down_proj.weight", [4096, 12288]),
        mixer: GdnSlots {
            qkv: role!("linear_attn.in_proj_qkv.weight", [8192, 4096]),
            z: role!("linear_attn.in_proj_z.weight", [4096, 4096]),
            a: role!("linear_attn.in_proj_a.weight", [32, 4096]),
            b: role!("linear_attn.in_proj_b.weight", [32, 4096]),
            out: role!("linear_attn.out_proj.weight", [4096, 4096]),
            conv: role!("linear_attn.conv1d.weight", [8192, 1, 4]),
            a_log: role!("linear_attn.A_log", [32]),
            dt_bias: role!("linear_attn.dt_bias", [32]),
            norm: role!("linear_attn.norm.weight", [128]),
        },
    })
}

fn attention_slots<R: BlobStoreGet>(
    facts: &TribleSet, snapshot: &R, root: Id, layer: usize,
    selected: &mut impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
) -> Result<full_attention::Slots, String> {
    let prefix = format!("model.language_model.layers.{layer}.");
    macro_rules! role { ($name:literal, $shape:expr) => {
        roles::resolve(facts, snapshot, root, &format!("{prefix}{}", $name), &$shape, selected)?
    }; }
    Ok(full_attention::Slots {
        input_norm: role!("input_layernorm.weight", [4096]),
        post_norm: role!("post_attention_layernorm.weight", [4096]),
        gate: role!("mlp.gate_proj.weight", [12288, 4096]),
        up: role!("mlp.up_proj.weight", [12288, 4096]),
        down: role!("mlp.down_proj.weight", [4096, 12288]),
        q: role!("self_attn.q_proj.weight", [8192, 4096]),
        k: role!("self_attn.k_proj.weight", [1024, 4096]),
        v: role!("self_attn.v_proj.weight", [1024, 4096]),
        o: role!("self_attn.o_proj.weight", [4096, 4096]),
        q_norm: role!("self_attn.q_norm.weight", [256]),
        k_norm: role!("self_attn.k_norm.weight", [256]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_contract_neither_invents_nor_duplicates_embedding_token() {
        assert!(validate_ids(&[EMBEDDING_TOKEN]).is_ok());
        assert!(validate_ids(&[12, 13, EMBEDDING_TOKEN]).is_ok());
        for bad in [vec![], vec![248044], vec![VOCAB as u32, EMBEDDING_TOKEN],
            vec![248056, EMBEDDING_TOKEN], vec![EMBEDDING_TOKEN; 257]] {
            assert!(validate_ids(&bad).is_err(), "{bad:?}");
        }
    }
}
