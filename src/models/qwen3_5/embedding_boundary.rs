//! The OUTPUT boundary of the one shared WeMM multimodal backbone.
//!
//! Prepared unpadded BF16 [1,T,4096] -> final zero-centered RMSNorm -> last
//! token -> BF16 L2 normalization. No tokenizer, token gather, decoder, vision
//! tower, model catalogue, host numerical path, or embedding-model admission.
//! The caller supplies the PRE-final-norm decoder output, not an already
//! normalized HF last_hidden_state. T is bounded to 1..=256.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use triblespace::core::{
    blob::{Blob, encodings::tensor::{Tensor as NativeTensor, elements::BF16}},
    inline::{Inline, encodings::hash::Handle},
    repo::{BlobStoreGet, pile::PileSnapshot},
};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use super::decoder_ops;

pub type CudaTensor = CubeTensor<CudaRuntime>;
pub type FinalNormSlot = Inline<Handle<NativeTensor<BF16, 1>>>;
pub const HIDDEN: usize = 4096;
pub const MAX_TOKENS: usize = 256;
pub const RMS_EPSILON: f32 = 1e-6;
pub const L2_EPSILON: f32 = 1e-12;

/// One weight of the shared multimodal model, not a separate text embedder.
pub struct Boundary { final_norm: CudaTensor }

/// Per-invocation witnesses. Only `embedding` is the wrapper's return value.
/// No observations are retained by Boundary between calls.
pub struct Output {
    pub final_hidden: CudaTensor,
    pub selected: CudaTensor,
    pub embedding: CudaTensor,
}

impl Boundary {
    /// # Safety
    /// The selected leaf must belong to a genuine validated append-only pile
    /// prefix. Its payload AND preceding partial page must remain immutable
    /// and untruncated through CUDA runtime teardown. A late descriptor error
    /// may retain the registration. Forwarded to the existing native binder.
    pub unsafe fn from_pile(
        snapshot: &PileSnapshot, slot: FinalNormSlot, aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        let blob: Blob<NativeTensor<BF16, 1>> = snapshot.get(slot).map_err(|e| e.to_string())?;
        // SAFETY: the caller establishes the genuine immutable-prefix premise.
        let final_norm = unsafe { aliases.bind_pile_leaf(blob)? };
        check(&final_norm, &final_norm, &[HIDDEN])?;
        if final_norm.handle.can_mut() { return Err("immutable final-norm weight required".into()); }
        Ok(Self { final_norm })
    }

    /// No padding/mask API: exactly one unpadded sequence, pooled at T-1.
    /// The producer must follow CubeCL's valid-handle and stream-ordering
    /// contracts. We check actual shared server-properties identity, not only
    /// the advertised CUDA device. CubeCL does not expose its stream id here;
    /// unsafe custom-stream producers must establish dependencies themselves.
    /// Driver/allocation/asynchronous failures retain upstream behavior.
    pub fn finish_unpadded(&self, hidden: &CudaTensor) -> Result<Output, String> {
        let shape = hidden.meta.shape().as_slice();
        if shape.len() != 3 || shape[0] != 1 || !(1..=MAX_TOKENS).contains(&shape[1])
            || shape[2] != HIDDEN {
            return Err("embedding boundary requires unpadded BF16 [1,T,4096], T1..256".into());
        }
        check(hidden, &self.final_norm, shape)?;
        // All input/weight extents and client identities checked before launch.
        let final_hidden = decoder_ops::norm(hidden, &self.final_norm, HIDDEN, RMS_EPSILON);
        let selected_handle = hidden.client.empty(HIDDEN * 2);
        let embedding_handle = hidden.client.empty(HIDDEN * 2);
        // One invocation, one ascending-coordinate reduction. No autotuning,
        // subgroup-size-dependent reduction or host readback/model arithmetic.
        unsafe {
            finish_kernel::launch_unchecked::<CudaRuntime>(
                &hidden.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(final_hidden.handle.clone(), shape[1] * HIDDEN),
                ArrayArg::from_raw_parts(selected_handle.clone(), HIDDEN),
                ArrayArg::from_raw_parts(embedding_handle.clone(), HIDDEN),
                (shape[1] - 1) * HIDDEN,
            );
        }
        let tensor = |handle| CubeTensor::new_contiguous(
            hidden.client.clone(), hidden.device.clone(), [1, HIDDEN].as_slice().into(),
            handle, DType::BF16,
        );
        Ok(Output { final_hidden, selected: tensor(selected_handle), embedding: tensor(embedding_handle) })
    }
}

fn check(t: &CudaTensor, weight: &CudaTensor, shape: &[usize]) -> Result<(), String> {
    if t.meta.shape().as_slice() != shape || t.meta.strides().len() != shape.len()
        || t.dtype != DType::BF16 || t.qparams.is_some() || t.device != weight.device
        || !std::ptr::eq(t.client.properties(), weight.client.properties()) {
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
    if extent > u32::MAX as usize || t.handle.size_in_used() < bytes as u64 {
        return Err("tensor storage too short or extent outside u32 domain".into());
    }
    Ok(())
}

#[cube(launch_unchecked)]
fn finish_kernel(x: &Array<bf16>, selected: &mut Array<bf16>, out: &mut Array<bf16>, offset: usize) {
    if ABSOLUTE_POS == 0 {
        let mut sum = 0.0f32;
        for j in 0..4096usize {
            let v = f32::cast_from(x[offset + j]);
            selected[j] = x[offset + j];
            sum += v * v;
        }
        // torch BF16 norm returns BF16; clamp_min(eps) also returns BF16.
        // Keep those roundings before the final BF16 division. F32 accumulation
        // is computation, never a changed input/weight storage interpretation.
        let mut denominator = f32::cast_from(bf16::cast_from(sum.sqrt()));
        let epsilon = f32::cast_from(bf16::cast_from(1.0e-12f32));
        if denominator < epsilon { denominator = epsilon; }
        for j in 0..4096usize {
            out[j] = bf16::cast_from(f32::cast_from(x[offset + j]) / denominator);
        }
    }
}
