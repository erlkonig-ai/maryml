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
    repo::BlobStoreGet,
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
    /// An exact-acquiring reader is permitted; its returned selected leaf must
    /// belong to a genuine validated append-only pile
    /// prefix. Its payload AND preceding partial page must remain immutable
    /// and untruncated through CUDA runtime teardown. A late descriptor error
    /// may retain the registration. Forwarded to the existing native binder.
    pub unsafe fn from_pile<R: BlobStoreGet>(
        snapshot: &R, slot: FinalNormSlot, aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        let blob: Blob<NativeTensor<BF16, 1>> = snapshot.get(slot).map_err(|e| e.to_string())?;
        // SAFETY: the caller establishes the genuine immutable-prefix premise.
        let final_norm = unsafe { aliases.bind_pile_leaf(blob)? };
        check(&final_norm, &final_norm, &[HIDDEN])?;
        if final_norm.handle.can_mut() { return Err("immutable final-norm weight required".into()); }
        Ok(Self { final_norm })
    }

    /// Share the already-validated immutable alias without another store get.
    pub(crate) fn weight_anchor(&self) -> &CudaTensor { &self.final_norm }

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

#[cfg(test)]
mod reader_tests {
    use super::*;
    use std::{cell::Cell, error::Error, path::PathBuf};
    use triblespace::core::{
        blob::{BlobEncoding, TryFromBlob},
        inline::InlineEncoding,
        repo::{BlobStorePut, SnapshotSource, StorageClose, pile::{Pile, PileSnapshot}},
    };
    use triblespace::prelude::{Id, TribleSet, fucid};
    use crate::models::qwen3_5::{
        config::Qwen3_5Config, decoder_stack, full_attention, gdn_decoder,
        gdn_mixer, multimodal::PreparedMultimodal, prepared::PreparedDecoder,
    };

    // Deliberately no Deref/AsRef<PileSnapshot>, Clone, StoreRead or catalogue:
    // constructors must need only the caller's exact typed gets.
    struct Reader<R> { source: R, gets: Cell<usize> }

    impl<R: BlobStoreGet> BlobStoreGet for Reader<R> {
        type GetError<E: Error + Send + Sync + 'static> = R::GetError<E>;

        fn get<T, S>(&self, handle: Inline<Handle<S>>) -> Result<T, Self::GetError<T::Error>>
        where
            S: BlobEncoding + 'static,
            T: TryFromBlob<S>,
            Handle<S>: InlineEncoding,
        {
            self.gets.set(self.gets.get() + 1);
            self.source.get(handle)
        }
    }

    // Assigning each constructor proves its generic boundary without creating
    // a CUDA client, fake aliases, uninitialized resources or model weights.
    fn constructors_accept<R: BlobStoreGet>() {
        type Observer = fn(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>;
        let _: unsafe fn(&TribleSet, &R, Id, &Qwen3_5Config, &mut CudaBf16Aliases, Observer)
            -> Result<PreparedMultimodal, String> = PreparedMultimodal::from_pile::<R>;
        let _: unsafe fn(&TribleSet, &R, Id, &mut CudaBf16Aliases, Observer)
            -> Result<PreparedDecoder, String> = PreparedDecoder::from_pile::<R>;
        let _: unsafe fn(&R, decoder_stack::Slots, &mut CudaBf16Aliases)
            -> Result<decoder_stack::Stack, String> = decoder_stack::Stack::from_pile::<R>;
        let _: unsafe fn(&R, gdn_decoder::Slots, gdn_decoder::Config, &mut CudaBf16Aliases)
            -> Result<gdn_decoder::Block, String> = gdn_decoder::Block::from_pile::<R>;
        let _: unsafe fn(&R, gdn_mixer::GdnSlots, gdn_mixer::GdnConfig, &mut CudaBf16Aliases)
            -> Result<gdn_mixer::GdnMixer, String> = gdn_mixer::GdnMixer::from_pile::<R>;
        let _: unsafe fn(&R, full_attention::Slots, full_attention::Config, &mut CudaBf16Aliases)
            -> Result<full_attention::Block, String> = full_attention::Block::from_pile::<R>;
        let _: unsafe fn(&R, FinalNormSlot, &mut CudaBf16Aliases)
            -> Result<Boundary, String> = Boundary::from_pile::<R>;
    }

    #[test]
    fn native_constructors_accept_exact_reader_without_snapshot_downcast() {
        constructors_accept::<Reader<PileSnapshot>>();
        use triblespace::core::repo::async_store::{AcquiringReader, SyncAsAsync};
        constructors_accept::<AcquiringReader<SyncAsAsync<PileSnapshot>>>();
    }

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("mary-fetching-leaf-{}", fucid().id));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn pile(&self) -> Pile {
            let path = self.0.join("weights.pile");
            std::fs::File::create_new(&path).unwrap();
            Pile::open(&path).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    fn norm_blob() -> Blob<NativeTensor<BF16, 1>> {
        // Native fixture bytes, not a host model computation or conversion.
        let bits: Vec<u8> = (0..HIDDEN).flat_map(|i| (i as u16).to_le_bytes()).collect();
        crate::leaf::leaf_blob::<BF16, 1>([HIDDEN as u64], bits.into()).unwrap()
    }

    #[test]
    fn forwarded_native_leaf_keeps_its_owner_after_reader_and_store_close() {
        let fixture = Fixture::new();
        let mut pile = fixture.pile();
        let slot = pile.put::<NativeTensor<BF16, 1>, _>(norm_blob()).unwrap();
        let reader = Reader { source: pile.snapshot().unwrap(), gets: Cell::new(0) };
        let blob: Blob<NativeTensor<BF16, 1>> = reader.get(slot).unwrap();
        assert_eq!(reader.gets.get(), 1);
        let leaf = crate::leaf::read_leaf(blob).unwrap();
        let address = leaf.payload().as_ptr();
        let owner = leaf.payload().clone().downcast_to_owner::<memmap2::MmapRaw>().unwrap();
        let weak = std::sync::Arc::downgrade(&owner);
        drop(owner);
        drop(reader);
        pile.close().unwrap();
        assert!(weak.upgrade().is_some());
        assert_eq!(leaf.payload().as_ptr(), address);
        assert_eq!(&leaf.payload()[..4], &[0, 0, 1, 0]);
        assert_eq!(leaf.payload().len(), HIDDEN * 2);
    }

    #[test]
    #[ignore = "requires an explicitly reserved CUDA host-page-table device"]
    fn generic_boundary_alias_survives_reader_store_and_binder_drop() {
        let fixture = Fixture::new();
        let mut pile = fixture.pile();
        let slot = pile.put::<NativeTensor<BF16, 1>, _>(norm_blob()).unwrap();
        let reader = Reader { source: pile.snapshot().unwrap(), gets: Cell::new(0) };
        let mut aliases = CudaBf16Aliases::new(cubecl::cuda::CudaDevice { index: 0 }, 1).unwrap();
        // SAFETY: this private pile and all preceding pages remain unchanged;
        // closing its handle does not truncate/rewrite its retained mmap.
        let boundary = unsafe { Boundary::from_pile(&reader, slot, &mut aliases) }.unwrap();
        assert_eq!(reader.gets.get(), 1);
        let anchor = boundary.weight_anchor().clone();
        assert_eq!(reader.gets.get(), 1, "sharing the anchor must not read again");
        assert_eq!(aliases.stats().registrations, 1);
        assert!(!anchor.handle.can_mut());
        drop(reader);
        pile.close().unwrap();
        drop(aliases);
        drop(boundary);
        let raw = anchor.client.read_one(anchor.handle.clone()).unwrap().to_vec();
        let expected = crate::leaf::read_leaf(norm_blob()).unwrap();
        assert_eq!(raw.as_slice(), expected.payload().as_ref());
    }
}
