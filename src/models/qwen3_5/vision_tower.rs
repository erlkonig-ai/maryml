//! Native BF16 packed-patch frontend -> configured vision blocks -> merger.
//!
//! Supports the checkpoint's 27 blocks at H1152 / M4304 / 16 heads and
//! decoder-width 4096, including 256-patch frames. This composition is not a
//! full-checkpoint numerical parity claim. The inherited block rotary recipe
//! is the whole-HF-model BF16 recipe; loader-specific rotary parity remains a
//! required numerical check, not a silently selected alternate precision.
//!
//! Input is already prepared, merge-major BF16 `[N,C*T*P*P]` on CUDA. This is
//! neither a raw-image processor nor decoder scatter. The result stays on the
//! device and carries local grid coordinates; only the decoder can combine
//! those coordinates with prompt offsets, image-token locations or timestamps.

use std::ops::Range;

use triblespace::core::{id::Id, repo::BlobStoreGet, trible::TribleSet};

use crate::nn::cuda_bf16_alias::CudaBf16Aliases;

use super::{
    config::{Dtype, Qwen3_5Config},
    gdn_mixer::{BoundProjection, DecoderPrecision},
    vision_block::{self, Block},
    vision_frontend::{self, CudaTensor, Frontend},
    vision_geometry::{self, Grid},
    vision_merger::{self, Merger},
};

pub const MAX_DEPTH: usize = 27;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub frontend: vision_frontend::Config,
    pub block: vision_block::Config,
    pub merger: vision_merger::Config,
    pub depth: usize,
}

impl Config {
    /// Derive the operator dimensions from the validated checkpoint config,
    /// never from a filename or an assumed hidden-size fallback.
    pub fn from_model(config: &Qwen3_5Config) -> Result<Self, String> {
        config.validate()?;
        if config.dtype != Dtype::Bf16 {
            return Err("native vision tower requires BF16 checkpoint leaves".into());
        }
        let v = &config.vision_config;
        let out = Self {
            frontend: vision_frontend::Config {
                hidden: v.hidden_size,
                channels: v.in_channels,
                temporal: v.temporal_patch_size,
                patch: v.patch_size,
                merge: v.spatial_merge_size,
                position_side: v.num_position_embeddings.isqrt(),
            },
            block: vision_block::Config {
                hidden: v.hidden_size,
                heads: v.num_heads,
                intermediate: v.intermediate_size,
                merge: v.spatial_merge_size,
            },
            merger: vision_merger::Config {
                hidden: v.hidden_size,
                output: v.out_hidden_size,
            },
            depth: v.depth,
        };
        out.validate()?;
        Ok(out)
    }

    pub fn validate(self) -> Result<(), String> {
        self.frontend.validate()?;
        self.block.validate()?;
        self.merger.validate()?;
        if !(1..=MAX_DEPTH).contains(&self.depth)
            || self.frontend.hidden != self.block.hidden
            || self.block.hidden != self.merger.hidden
            || self.frontend.merge != 2
        {
            return Err("vision tower requires depth 1..27, equal hidden widths and merge2".into());
        }
        Ok(())
    }

    /// Number of selected native parameter leaves, not mmap owners or RAM.
    pub fn parameter_slots(self) -> Result<usize, String> {
        self.validate()?;
        Ok(3 + 12 * self.depth + 6)
    }
}

/// Exact typed leaves selected by the model/source layer. Their order is the
/// checkpoint block order, not a map enumeration or a second weight catalogue.
pub struct Slots {
    pub frontend: vision_frontend::Slots,
    pub blocks: Vec<vision_block::Slots>,
    pub merger: vision_merger::Slots,
}

impl Slots {
    /// Read only the roles used by this tower from the caller-selected root
    /// and fixed fact observation. A fetching reader may supply exact missing
    /// path/weight bytes without selecting another model or observation.
    pub fn resolve(
        facts: &TribleSet,
        source: &impl BlobStoreGet,
        root: Id,
        config: Config,
        selected: &mut impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        config.validate()?;
        let h = config.block.hidden as u64;
        let m = config.block.intermediate as u64;
        let f = config.frontend;
        macro_rules! role {
            ($name:expr, $shape:expr) => {
                super::roles::resolve(facts, source, root, $name, &$shape, selected)?
            };
        }
        // A block or merger linear: its BF16 leaf, or the NVFP4 leaf a root
        // made by `models_import wemm --vision-nvfp4` stores in its place.
        macro_rules! linear {
            ($name:expr, $shape:expr) => {
                super::roles::resolve_linear(facts, source, root, $name, &$shape, selected)?
            };
        }
        let frontend = vision_frontend::Slots {
            patch: role!(
                "model.visual.patch_embed.proj.weight",
                [
                    h,
                    f.channels as u64,
                    f.temporal as u64,
                    f.patch as u64,
                    f.patch as u64
                ]
            ),
            bias: role!("model.visual.patch_embed.proj.bias", [h]),
            position: role!(
                "model.visual.pos_embed.weight",
                [(f.position_side * f.position_side) as u64, h]
            ),
        };
        let mut blocks = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            let prefix = format!("model.visual.blocks.{index}");
            blocks.push(vision_block::Slots {
                norm1_weight: role!(&format!("{prefix}.norm1.weight"), [h]),
                norm1_bias: role!(&format!("{prefix}.norm1.bias"), [h]),
                norm2_weight: role!(&format!("{prefix}.norm2.weight"), [h]),
                norm2_bias: role!(&format!("{prefix}.norm2.bias"), [h]),
                qkv_weight: linear!(&format!("{prefix}.attn.qkv.weight"), [3 * h, h]),
                qkv_bias: role!(&format!("{prefix}.attn.qkv.bias"), [3 * h]),
                proj_weight: linear!(&format!("{prefix}.attn.proj.weight"), [h, h]),
                proj_bias: role!(&format!("{prefix}.attn.proj.bias"), [h]),
                fc1_weight: linear!(&format!("{prefix}.mlp.linear_fc1.weight"), [m, h]),
                fc1_bias: role!(&format!("{prefix}.mlp.linear_fc1.bias"), [m]),
                fc2_weight: linear!(&format!("{prefix}.mlp.linear_fc2.weight"), [h, m]),
                fc2_bias: role!(&format!("{prefix}.mlp.linear_fc2.bias"), [h]),
            });
        }
        let o = config.merger.output as u64;
        let merger = vision_merger::Slots {
            norm: role!("model.visual.merger.norm.weight", [h]),
            norm_bias: role!("model.visual.merger.norm.bias", [h]),
            fc1: linear!("model.visual.merger.linear_fc1.weight", [4 * h, 4 * h]),
            bias1: role!("model.visual.merger.linear_fc1.bias", [4 * h]),
            fc2: linear!("model.visual.merger.linear_fc2.weight", [o, 4 * h]),
            bias2: role!("model.visual.merger.linear_fc2.bias", [o]),
        };
        Ok(Self {
            frontend,
            blocks,
            merger,
        })
    }
}

/// The vision tower's linears as bound, from each block's [qkv, proj, fc1,
/// fc2] and the merger's [fc1, fc2], in the two shapes a kernel profile can
/// name: `Bf16` when every linear is BF16, and `Nvfp4 { precision, stored }`
/// when every block's qkv, proj and fc1 and both merger linears (the 83
/// linears of `gdn_mixer::is_vision_nvfp4_linear`) run that one NVFP4 kind
/// while every block's fc2 (K 4304, which no NVFP4 kernel takes) is BF16.
/// Anything else, a mixture of NVFP4 kinds or of NVFP4 and BF16 among those
/// 83 included, is `None`. A tower with no blocks is `None` too.
pub fn vision_bound_of(
    blocks: impl IntoIterator<Item = [BoundProjection; 4]>,
    merger: [BoundProjection; 2],
) -> Option<BoundProjection> {
    let mut nvfp4 = merger.to_vec();
    let mut fc2 = Vec::new();
    for [qkv, proj, fc1, block_fc2] in blocks {
        nvfp4.extend([qkv, proj, fc1]);
        fc2.push(block_fc2);
    }
    if fc2.is_empty() || fc2.iter().any(|b| *b != BoundProjection::Bf16) {
        return None;
    }
    let first = nvfp4[0];
    nvfp4.iter().all(|b| *b == first).then_some(first)
}

/// Rows contributed by one input image/grid. Rows are contiguous and retain
/// input-item order. Temporal groups are not assigned wall-clock timestamps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageFeatureSpan {
    pub grid: Grid,
    pub rows: Range<usize>,
}

pub struct PreparedImageFeatures {
    /// Native BF16 `[sum(T * H/2 * W/2), decoder_hidden]`, no host copy.
    pub features: CudaTensor,
    pub images: Vec<ImageFeatureSpan>,
    /// Local `[temporal_group, merged_row, merged_column]` for each GPU row.
    /// Integer shape metadata only, not final decoder MRoPE positions.
    pub grid_coordinates: Vec<[u32; 3]>,
}

pub struct Tower {
    config: Config,
    frontend: Frontend,
    blocks: Vec<Block>,
    merger: Merger,
}

impl Tower {
    /// The source may be an acquiring reader over a frozen pile observation.
    /// Missing/malformed selected leaves remain errors; no upload or F16 path
    /// is substituted. Caller owns/reuses the alias binder and its budget.
    ///
    /// # Safety
    /// Every fetched leaf must originate in a genuine validated append-only
    /// pile prefix, including preceding partial pages, which remains immutable
    /// until CUDA runtime teardown. Binder drop alone does not unregister the
    /// runtime's external allocations. Late errors may retain earlier aliases.
    pub unsafe fn from_pile<R: BlobStoreGet>(
        source: &R,
        slots: Slots,
        config: Config,
        aliases: &mut CudaBf16Aliases,
    ) -> Result<Self, String> {
        // SAFETY: forwarded unchanged.
        unsafe { Self::from_pile_with(source, slots, config, aliases, DecoderPrecision::Bf16) }
    }

    /// [`Self::from_pile`] with the blocks' and the merger's linears on
    /// `precision` (`vision_block::Block::from_pile_with`). The frontend's
    /// patch convolution stays BF16 in every precision.
    ///
    /// # Safety
    /// As [`Self::from_pile`].
    pub unsafe fn from_pile_with<R: BlobStoreGet>(
        source: &R,
        slots: Slots,
        config: Config,
        aliases: &mut CudaBf16Aliases,
        precision: DecoderPrecision,
    ) -> Result<Self, String> {
        config.validate()?;
        if slots.blocks.len() != config.depth {
            return Err(format!(
                "vision tower needs {} ordered blocks, got {}",
                config.depth,
                slots.blocks.len()
            ));
        }
        // SAFETY: the caller supplies the same pile-prefix premise for all
        // selected leaves, forwarded without changing their byte ownership.
        let frontend =
            unsafe { Frontend::from_pile(source, slots.frontend, config.frontend, aliases)? };
        let mut blocks = Vec::with_capacity(config.depth);
        for (index, block) in slots.blocks.into_iter().enumerate() {
            blocks.push(
                unsafe { Block::from_pile_with(source, block, config.block, aliases, precision) }
                    .map_err(|e| format!("vision block {index}: {e}"))?,
            );
        }
        let merger = unsafe { Merger::from_pile_with(source, slots.merger, config.merger, aliases, precision)? };
        Ok(Self {
            config,
            frontend,
            blocks,
            merger,
        })
    }

    /// What the vision tower's linears run, as bound, in the two shapes a
    /// kernel profile can name ([`vision_bound_of`]).
    pub fn vision_bound(&self) -> Option<BoundProjection> {
        vision_bound_of(
            self.blocks.iter().map(|b| b.linears().map(|l| l.bound())),
            self.merger.linears().map(|l| l.bound()),
        )
    }

    /// The blocks' and the merger's linears as bound: (count, BF16 count,
    /// BF16 weight bytes, NVFP4 count, NVFP4 plane bytes, bound kinds seen).
    pub fn linear_bytes(&self) -> (usize, usize, u64, usize, u64, Vec<BoundProjection>) {
        let mut out = (0, 0, 0, 0, 0, Vec::new());
        let all = self.blocks.iter().flat_map(|b| b.linears()).chain(self.merger.linears());
        for l in all {
            out.0 += 1;
            let (b, f) = l.bytes();
            match l.bound() {
                BoundProjection::Bf16 => { out.1 += 1; out.2 += b; }
                BoundProjection::Nvfp4 { .. } => { out.3 += 1; out.4 += f; }
            }
            if !out.5.contains(&l.bound()) {
                out.5.push(l.bound());
            }
        }
        out
    }

    /// All grid/frame/score extents are checked before any GPU allocation.
    /// Only one block's diagnostic intermediates are alive at a time. No
    /// tensor values cross to the host between frontend, blocks, and merger.
    pub fn forward(
        &self,
        input: &CudaTensor,
        grids: &[Grid],
    ) -> Result<PreparedImageFeatures, String> {
        let shape = input.meta.shape().as_slice();
        if shape.len() != 2 || shape[1] != self.config.frontend.input_width() {
            return Err("vision input must be packed [patches,C*T*P*P]".into());
        }
        let frames = vision_geometry::frames(grids, shape[0], self.config.block.heads)?;
        let (images, grid_coordinates) = feature_geometry(grids);
        let mut hidden = self.frontend.forward(input, grids)?.combined;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block
                .forward(&hidden, grids)
                .map_err(|e| format!("vision block {index}: {e}"))?
                .hidden;
        }
        let groups: Vec<_> = frames.iter().map(|frame| frame.patches).collect();
        let features = self.merger.forward(&hidden, &groups)?.hidden;
        debug_assert_eq!(
            features.meta.shape().as_slice(),
            &[grid_coordinates.len(), self.config.merger.output]
        );
        Ok(PreparedImageFeatures {
            features,
            images,
            grid_coordinates,
        })
    }
}

// Called only after complete geometry validation; at most 1024 output rows.
fn feature_geometry(grids: &[Grid]) -> (Vec<ImageFeatureSpan>, Vec<[u32; 3]>) {
    let mut images = Vec::with_capacity(grids.len());
    let mut coordinates = Vec::new();
    for &grid in grids {
        let start = coordinates.len();
        for frame in 0..grid.frames {
            for row in 0..grid.height / 2 {
                for column in 0..grid.width / 2 {
                    coordinates.push([frame as u32, row as u32, column as u32]);
                }
            }
        }
        images.push(ImageFeatureSpan {
            grid,
            rows: start..coordinates.len(),
        });
    }
    (images, coordinates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint_dimensions() -> Config {
        Config {
            frontend: vision_frontend::Config {
                hidden: 1152,
                channels: 3,
                temporal: 2,
                patch: 16,
                merge: 2,
                position_side: 48,
            },
            block: vision_block::Config {
                hidden: 1152,
                heads: 16,
                intermediate: 4304,
                merge: 2,
            },
            merger: vision_merger::Config {
                hidden: 1152,
                output: 4096,
            },
            depth: 27,
        }
    }

    /// Every block and merger linear of the checkpoint, `(name, [n, k])`.
    fn checkpoint_linears() -> Vec<(String, [usize; 2])> {
        let c = checkpoint_dimensions();
        let (h, m, o) = (c.block.hidden, c.block.intermediate, c.merger.output);
        let mut all = Vec::new();
        for b in 0..c.depth {
            let p = format!("model.visual.blocks.{b}");
            all.push((format!("{p}.attn.qkv.weight"), [3 * h, h]));
            all.push((format!("{p}.attn.proj.weight"), [h, h]));
            all.push((format!("{p}.mlp.linear_fc1.weight"), [m, h]));
            all.push((format!("{p}.mlp.linear_fc2.weight"), [h, m]));
        }
        all.push(("model.visual.merger.linear_fc1.weight".into(), [4 * h, 4 * h]));
        all.push(("model.visual.merger.linear_fc2.weight".into(), [o, 4 * h]));
        all
    }

    #[test]
    fn the_stored_vision_linears_are_exactly_the_83_both_nvfp4_kernels_take() {
        use super::super::gdn_mixer::{is_decoder_projection, is_vision_nvfp4_linear};
        let all = checkpoint_linears();
        assert_eq!(all.len(), 110);
        let mut stored = 0;
        for (name, [n, k]) in &all {
            let named = is_vision_nvfp4_linear(name);
            assert_eq!(named, vision_block::nvfp4_takes(DecoderPrecision::W4a4, *n, *k), "{name}");
            assert_eq!(named, vision_block::nvfp4_takes(DecoderPrecision::W4a16, *n, *k), "{name}");
            assert!(!vision_block::nvfp4_takes(DecoderPrecision::Bf16, *n, *k));
            assert!(!is_decoder_projection(name), "{name}");
            stored += named as usize;
        }
        assert_eq!(stored, 83);
        for other in [
            "model.visual.blocks.27.attn.qkv.weight",
            "model.visual.blocks.01.attn.qkv.weight",
            "model.visual.blocks.0.attn.qkv.bias",
            "model.visual.blocks.0.mlp.linear_fc2.weight",
            "model.visual.blocks.0.norm1.weight",
            "model.visual.merger.linear_fc1.bias",
            "model.visual.patch_embed.proj.weight",
            "model.visual.pos_embed.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
        ] {
            assert!(!is_vision_nvfp4_linear(other), "{other}");
        }
    }

    #[test]
    fn the_vision_tower_is_bound_in_a_profile_shape_or_in_none() {
        use BoundProjection::{Bf16, Nvfp4};
        let w4a16 = Nvfp4 { precision: DecoderPrecision::W4a16, stored: true };
        let at_load = Nvfp4 { precision: DecoderPrecision::W4a16, stored: false };
        let w4a4 = Nvfp4 { precision: DecoderPrecision::W4a4, stored: true };
        let tower = |block: [BoundProjection; 4], merger| vision_bound_of(vec![block; 27], merger);
        assert_eq!(tower([Bf16; 4], [Bf16; 2]), Some(Bf16));
        assert_eq!(tower([w4a16, w4a16, w4a16, Bf16], [w4a16; 2]), Some(w4a16));
        assert_eq!(tower([at_load, at_load, at_load, Bf16], [at_load; 2]), Some(at_load));
        assert_eq!(tower([w4a4, w4a4, w4a4, Bf16], [w4a4; 2]), Some(w4a4));
        for (block, merger) in [
            // fc2 is never NVFP4 in a profile shape.
            ([w4a16; 4], [w4a16; 2]),
            ([Bf16, Bf16, Bf16, w4a16], [Bf16; 2]),
            // NVFP4 blocks with a BF16 merger, and the reverse.
            ([w4a16, w4a16, w4a16, Bf16], [Bf16; 2]),
            ([Bf16; 4], [w4a16; 2]),
            // Two NVFP4 kinds.
            ([w4a16, w4a16, w4a16, Bf16], [w4a4; 2]),
            ([w4a16, at_load, w4a16, Bf16], [w4a16; 2]),
        ] {
            assert_eq!(tower(block, merger), None, "{block:?} {merger:?}");
        }
        // One block unlike the other 26.
        let mut blocks = vec![[w4a16, w4a16, w4a16, Bf16]; 27];
        blocks[13] = [Bf16; 4];
        assert_eq!(vision_bound_of(blocks, [w4a16; 2]), None);
        assert_eq!(vision_bound_of(Vec::new(), [Bf16; 2]), None);
    }

    #[test]
    fn checkpoint_real_width_and_27_blocks_are_supported() {
        let c = checkpoint_dimensions();
        c.validate().unwrap();
        assert_eq!(c.parameter_slots().unwrap(), 333);
        assert_eq!(c.frontend.input_width(), 1536);
        let grid = Grid {
            frames: 1,
            height: 16,
            width: 16,
        };
        vision_geometry::frames(&[grid], 256, c.block.heads).unwrap();
        let (images, coordinates) = feature_geometry(&[grid]);
        assert_eq!(images[0].rows, 0..64);
        assert_eq!(coordinates[0], [0, 0, 0]);
        assert_eq!(coordinates[63], [0, 7, 7]);
    }

    #[test]
    fn dimensions_come_from_the_model_config_and_other_storage_dtypes_are_refused() {
        let mut model = super::super::config::tests::tiny();
        let actual = checkpoint_dimensions();
        model.text_config.hidden_size = actual.merger.output;
        let v = &mut model.vision_config;
        v.depth = actual.depth;
        v.hidden_size = actual.block.hidden;
        v.intermediate_size = actual.block.intermediate;
        v.num_heads = actual.block.heads;
        v.patch_size = actual.frontend.patch;
        v.num_position_embeddings = actual.frontend.position_side * actual.frontend.position_side;
        v.out_hidden_size = actual.merger.output;
        let derived = Config::from_model(&model).unwrap();
        assert_eq!(derived.parameter_slots().unwrap(), 333);
        assert_eq!(derived.block.hidden, 1152);
        assert_eq!(derived.block.intermediate, 4304);
        assert_eq!(derived.merger.output, 4096);
        for dtype in [Dtype::F16, Dtype::F32] {
            model.dtype = dtype;
            model.text_config.dtype = dtype;
            model.vision_config.dtype = dtype;
            assert!(
                Config::from_model(&model)
                    .unwrap_err()
                    .contains("BF16 checkpoint")
            );
        }
    }

    #[test]
    fn prepared_coordinates_preserve_items_and_temporal_groups() {
        let grids = [
            Grid {
                frames: 2,
                height: 4,
                width: 6,
            },
            Grid {
                frames: 1,
                height: 16,
                width: 16,
            },
        ];
        vision_geometry::frames(&grids, 304, 16).unwrap();
        let (images, positions) = feature_geometry(&grids);
        assert_eq!(images[0].rows, 0..12);
        assert_eq!(images[1].rows, 12..76);
        assert_eq!(positions[5], [0, 1, 2]);
        assert_eq!(positions[6], [1, 0, 0]);
        assert_eq!(positions[12], [0, 0, 0]);
        let (_, separate) = feature_geometry(&grids[1..]);
        assert_eq!(&positions[12..], separate.as_slice());
    }

    #[test]
    fn rejects_mismatched_composition_instead_of_reinterpreting_weights() {
        let mut c = checkpoint_dimensions();
        c.depth = 0;
        assert!(c.validate().is_err());
        c.depth = 28;
        assert!(c.validate().is_err());
        c = checkpoint_dimensions();
        c.frontend.hidden = 1024;
        assert!(c.validate().is_err());
        c = checkpoint_dimensions();
        c.frontend.merge = 4;
        assert!(c.validate().is_err());
    }
}
