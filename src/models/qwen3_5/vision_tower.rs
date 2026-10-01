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
                qkv_weight: role!(&format!("{prefix}.attn.qkv.weight"), [3 * h, h]),
                qkv_bias: role!(&format!("{prefix}.attn.qkv.bias"), [3 * h]),
                proj_weight: role!(&format!("{prefix}.attn.proj.weight"), [h, h]),
                proj_bias: role!(&format!("{prefix}.attn.proj.bias"), [h]),
                fc1_weight: role!(&format!("{prefix}.mlp.linear_fc1.weight"), [m, h]),
                fc1_bias: role!(&format!("{prefix}.mlp.linear_fc1.bias"), [m]),
                fc2_weight: role!(&format!("{prefix}.mlp.linear_fc2.weight"), [h, m]),
                fc2_bias: role!(&format!("{prefix}.mlp.linear_fc2.bias"), [h]),
            });
        }
        let o = config.merger.output as u64;
        let merger = vision_merger::Slots {
            norm: role!("model.visual.merger.norm.weight", [h]),
            norm_bias: role!("model.visual.merger.norm.bias", [h]),
            fc1: role!("model.visual.merger.linear_fc1.weight", [4 * h, 4 * h]),
            bias1: role!("model.visual.merger.linear_fc1.bias", [4 * h]),
            fc2: role!("model.visual.merger.linear_fc2.weight", [o, 4 * h]),
            bias2: role!("model.visual.merger.linear_fc2.bias", [o]),
        };
        Ok(Self {
            frontend,
            blocks,
            merger,
        })
    }
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
                unsafe { Block::from_pile(source, block, config.block, aliases) }
                    .map_err(|e| format!("vision block {index}: {e}"))?,
            );
        }
        let merger = unsafe { Merger::from_pile(source, slots.merger, config.merger, aliases)? };
        Ok(Self {
            config,
            frontend,
            blocks,
            merger,
        })
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
