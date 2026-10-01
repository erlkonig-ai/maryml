//! One selected native BF16 WeMM model: prepared patches + prepared IDs ->
//! 27-block vision tower -> GPU placeholder scatter -> the SAME 32-layer
//! decoder and last-token embedding endpoint. Source composition, not a claim
//! of numerical admission. Image preprocessing, video, padding, batching and
//! continuation remain outside this finite one-still-image entry point.
//!
//! CPU work is bounded integer control metadata only. Tensor values remain on
//! CUDA throughout. No F16 conversion, fallback upload of weights or second
//! model/root/binder is hidden in this wrapper.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cuda::CudaRuntime, prelude::*};
use half::bf16;
use triblespace::{
    core::repo::pile::PileSnapshot,
    prelude::{Id, TribleSet},
};

use super::{
    config::Qwen3_5Config,
    embedding_boundary,
    multimodal_layout::{self, ImagePlan},
    position_table::PositionTable,
    prepared::PreparedDecoder,
    vision_geometry::Grid,
    vision_tower::{self, PreparedImageFeatures, Tower},
};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;

pub type CudaTensor = CubeTensor<CudaRuntime>;

pub struct PreparedMultimodal {
    decoder: PreparedDecoder,
    vision: Tower,
}

pub struct Output {
    /// Device witnesses are returned without synchronizing or reading values.
    pub image_features: PreparedImageFeatures,
    pub decoder_input: CudaTensor,
    pub layout: ImagePlan,
    pub endpoint: embedding_boundary::Output,
}

impl PreparedMultimodal {
    /// Text-only input through this same selected multimodal model. Weights
    /// remain bound once; each call has fresh decoder state, as image calls do.
    pub fn embed_text(&self, ids: &[u32]) -> Result<super::prepared::Output, String> {
        self.decoder.embed_unpadded(ids)
    }

    /// All 759 roles come from this ONE opaque root and frozen observation;
    /// the constructor never discovers/reselects a model between modalities.
    /// `selected` is immediate provenance exhaust, not a retained catalogue.
    ///
    /// # Safety
    /// Forward the binder's genuine append-only pile-prefix obligation for
    /// every parameter and its preceding partial page, until CUDA runtime
    /// teardown. Caller owns source/session and reuses one bounded binder.
    /// A late error can retain earlier registrations; drop is not unregister.
    pub unsafe fn from_pile(
        facts: &TribleSet,
        snapshot: &PileSnapshot,
        root: Id,
        config: &Qwen3_5Config,
        aliases: &mut CudaBf16Aliases,
        mut selected: impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        multimodal_layout::validate_model(config)?;
        let vision_config = vision_tower::Config::from_model(config)?;
        let vision_slots =
            vision_tower::Slots::resolve(facts, snapshot, root, vision_config, &mut selected)?;
        // SAFETY: same genuine frozen source, same caller-owned binder.
        let decoder =
            unsafe { PreparedDecoder::from_pile(facts, snapshot, root, aliases, &mut selected)? };
        let vision = unsafe { Tower::from_pile(snapshot, vision_slots, vision_config, aliases)? };
        Ok(Self { decoder, vision })
    }

    /// Exactly one prepared still image. `pixels` is merge-major native BF16
    /// `[H*W,1536]`; IDs already contain its H*W/4 placeholders and the final
    /// <embedding>. Never invent, truncate or replace caller tokens.
    ///
    /// Mask policy equals an explicit all-ones [1,T] processor mask: unpadded
    /// causal token order, no packed inference from repeated MRoPE positions.
    /// The CUDA producer/client's usual valid-handle/stream obligations apply.
    pub fn embed_image(
        &self,
        ids: &[u32],
        pixels: &CudaTensor,
        grid: Grid,
    ) -> Result<Output, String> {
        self.embed_image_observed(ids, pixels, grid, |_, _| Ok(()))
    }

    pub fn embed_image_observed(
        &self,
        ids: &[u32],
        pixels: &CudaTensor,
        grid: Grid,
        observed: impl FnMut(usize, &CudaTensor) -> Result<(), String>,
    ) -> Result<Output, String> {
        let shape = pixels.meta.shape().as_slice();
        if shape.len() != 2 || shape[1] != 1536 {
            return Err("WeMM prepared pixels require [patches,1536]".into());
        }
        let layout = ImagePlan::new(ids, grid, shape[0], shape[0] / 4)?;
        let table = self.decoder.embedding_table();
        check_bf16(pixels, table, &[shape[0], 1536])?;
        let positions = PositionTable::from_positions(table, layout.positions())?;
        let image_features = self.vision.forward(pixels, &[grid])?;
        validate_features(&layout, &image_features, table)?;
        let decoder_input = scatter(table, ids, &image_features.features, &layout);
        let endpoint = self
            .decoder
            .decoder_stack()
            .prefill_positioned_unpadded_observed(&decoder_input, &positions, observed)?;
        Ok(Output {
            image_features,
            decoder_input,
            layout,
            endpoint,
        })
    }
}

fn validate_features(
    plan: &ImagePlan,
    features: &PreparedImageFeatures,
    like: &CudaTensor,
) -> Result<(), String> {
    if features.images.len() != 1
        || features.images[0].grid != plan.grid()
        || features.images[0].rows != (0..plan.feature_rows())
    {
        return Err("prepared image feature span differs from selected input image".into());
    }
    plan.check_coordinates(&features.grid_coordinates)?;
    check_bf16(&features.features, like, &[plan.feature_rows(), 4096])
}

fn check_bf16(t: &CudaTensor, like: &CudaTensor, shape: &[usize]) -> Result<(), String> {
    if t.meta.shape().as_slice() != shape
        || t.meta.strides().len() != shape.len()
        || t.dtype != DType::BF16
        || t.qparams.is_some()
        || t.device != like.device
        || !std::ptr::eq(t.client.properties(), like.client.properties())
    {
        return Err(
            "multimodal tensors need contiguous BF16 and the SAME bound model CUDA client".into(),
        );
    }
    let mut extent = 1usize;
    for (axis, &dimension) in shape.iter().enumerate().rev() {
        if dimension == 0 || (dimension > 1 && t.meta.strides()[axis] != extent) {
            return Err("nonempty contiguous multimodal tensor required".into());
        }
        extent = extent
            .checked_mul(dimension)
            .ok_or("tensor extent overflow")?;
    }
    let bytes = extent.checked_mul(2).ok_or("tensor byte extent overflow")?;
    let start = t.handle.offset_start.unwrap_or(0);
    let available = t
        .handle
        .size()
        .checked_sub(start)
        .and_then(|n| n.checked_sub(t.handle.offset_end.unwrap_or(0)))
        .ok_or("invalid tensor offsets")?;
    if extent > u32::MAX as usize || start % 2 != 0 || available < bytes as u64 {
        return Err("tensor storage is short/unaligned or outside kernel index bounds".into());
    }
    Ok(())
}

fn scatter(
    table: &CudaTensor,
    ids: &[u32],
    features: &CudaTensor,
    layout: &ImagePlan,
) -> CudaTensor {
    let id_bytes: Vec<_> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let row_bytes: Vec<_> = layout
        .feature_for_token()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let id_handle = table.client.create_from_slice(&id_bytes);
    let row_handle = table.client.create_from_slice(&row_bytes);
    let count = ids.len() * 4096;
    let out = table.client.empty(count * 2);
    let dim = CubeDim::new_1d(64);
    // The private plan proves all IDs, image row indices and output extents.
    // Each GPU thread copies one BF16 coordinate. No reduction or atomic and
    // no overwrite of aliased model/input storage occurs.
    unsafe {
        scatter_kernel::launch_unchecked::<CudaRuntime>(
            &table.client,
            cubecl::calculate_cube_count_elemwise(&table.client, count, dim),
            dim,
            ArrayArg::from_raw_parts(
                table.handle.clone(),
                multimodal_layout::VOCAB as usize * 4096,
            ),
            ArrayArg::from_raw_parts(id_handle, ids.len()),
            ArrayArg::from_raw_parts(row_handle, ids.len()),
            ArrayArg::from_raw_parts(features.handle.clone(), layout.feature_rows() * 4096),
            ArrayArg::from_raw_parts(out.clone(), count),
            count,
        );
    }
    CubeTensor::new_contiguous(
        table.client.clone(),
        table.device.clone(),
        [1, ids.len(), 4096].as_slice().into(),
        out,
        DType::BF16,
    )
}

#[cube(launch_unchecked)]
fn scatter_kernel(
    table: &Array<bf16>,
    ids: &Array<u32>,
    rows: &Array<u32>,
    features: &Array<bf16>,
    out: &mut Array<bf16>,
    count: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count {
        let token = i / 4096;
        let column = i % 4096;
        let image_row = rows[token];
        if image_row == u32::MAX {
            out[i] = table[(ids[token] as usize) * 4096 + column];
        } else {
            out[i] = features[(image_row as usize) * 4096 + column];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cuda::CudaDevice;

    #[cube(launch_unchecked)]
    fn fill_scatter_fixture(out: &mut Array<bf16>, n: usize, salt: usize) {
        let i = ABSOLUTE_POS as usize;
        if i < n {
            out[i] = bf16::cast_from(f32::cast_from((i * 13 + salt) % 251) / 128.0f32);
        }
    }

    #[test]
    #[ignore = "requires reserved CUDA; exact BF16 scatter/order, not model parity"]
    fn gpu_scatter_preserves_text_and_feature_row_bits() {
        let client = CudaRuntime::client(&CudaDevice { index: 0 });
        let table = client.empty(2 * 4096 * 2);
        let features = client.empty(2 * 4096 * 2);
        let dim = CubeDim::new_1d(64);
        for (handle, salt) in [(&table, 3), (&features, 127)] {
            unsafe {
                fill_scatter_fixture::launch_unchecked::<CudaRuntime>(
                    &client,
                    cubecl::calculate_cube_count_elemwise(&client, 8192, dim),
                    dim,
                    ArrayArg::from_raw_parts(handle.clone(), 8192),
                    8192,
                    salt,
                );
            }
        }
        let ids = [1u32, 0, 0, 0];
        let rows = [u32::MAX, 1, 0, u32::MAX];
        let ids =
            client.create_from_slice(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let rows = client.create_from_slice(
            &rows
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let out = client.empty(4 * 4096 * 2);
        unsafe {
            scatter_kernel::launch_unchecked::<CudaRuntime>(
                &client,
                cubecl::calculate_cube_count_elemwise(&client, 16384, dim),
                dim,
                ArrayArg::from_raw_parts(table.clone(), 8192),
                ArrayArg::from_raw_parts(ids, 4),
                ArrayArg::from_raw_parts(rows, 4),
                ArrayArg::from_raw_parts(features.clone(), 8192),
                ArrayArg::from_raw_parts(out.clone(), 16384),
                16384,
            );
        }
        // Readback is test evidence only: compare copied bytes, no host tensor arithmetic.
        let table = client.read_one(table).unwrap();
        let features = client.read_one(features).unwrap();
        let actual = client.read_one(out).unwrap();
        for (token, expected) in [
            &table[8192..16384],
            &features[8192..16384],
            &features[..8192],
            &table[..8192],
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(&actual[token * 8192..(token + 1) * 8192], *expected);
        }
    }
}
