//! Metadata-only packing contract shared by the native vision components.
//! No image values, masks, or tensor arithmetic are materialized on the host.

use serde::{Deserialize, Serialize};

pub const MAX_PATCHES: usize = 4096;
pub const MAX_HEADS: usize = 16;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Grid {
    /// Number of temporal patch groups, not decoder timestamps.
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub offset: usize,
    pub patches: usize,
    pub height: usize,
    pub width: usize,
}

/// Validate every extent before allocating even the frame metadata. All
/// kernels use u32 indexing. The largest probability allocation is
/// 4096 * 16 * 4096 * sizeof(BF16) = 512 MiB, never per-thread scratch.
pub(crate) fn frames(grids: &[Grid], rows: usize, heads: usize) -> Result<Vec<Frame>, String> {
    if !(1..=MAX_PATCHES).contains(&rows) || !(1..=MAX_HEADS).contains(&heads) {
        return Err("vision requires total patches 1..4096 and heads 1..16".into());
    }
    let mut total = 0usize;
    let mut frame_count = 0usize;
    for grid in grids {
        if grid.frames == 0
            || grid.height == 0
            || grid.width == 0
            || grid.height % 2 != 0
            || grid.width % 2 != 0
        {
            return Err("positive merge2 grids required".into());
        }
        let patches = grid.height.checked_mul(grid.width).ok_or("grid overflow")?;
        if patches > MAX_PATCHES {
            return Err("vision supports at most 4096 patches per frame".into());
        }
        let score_cells = patches
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(patches))
            .ok_or("attention extent overflow")?;
        if score_cells > u32::MAX as usize || score_cells.checked_mul(2).is_none() {
            return Err("attention exceeds kernel/allocation index domain".into());
        }
        let extent = grid
            .frames
            .checked_mul(patches)
            .ok_or("frame extent overflow")?;
        total = total.checked_add(extent).ok_or("grid sum overflow")?;
        if total > rows {
            return Err("grid coverage exceeds input rows".into());
        }
        frame_count = frame_count
            .checked_add(grid.frames)
            .ok_or("frame count overflow")?;
    }
    if total != rows {
        return Err("grid coverage differs from input rows".into());
    }
    let mut plan = Vec::with_capacity(frame_count);
    let mut offset = 0;
    for grid in grids {
        let patches = grid.height * grid.width; // checked above
        for _ in 0..grid.frames {
            plan.push(Frame {
                offset,
                patches,
                height: grid.height,
                width: grid.width,
            });
            offset += patches;
        }
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_reference_256_patch_frame_and_maximum_have_checked_bounds() {
        let grid = Grid {
            frames: 1,
            height: 16,
            width: 16,
        };
        assert_eq!(frames(&[grid], 256, 16).unwrap()[0].patches, 256);
        let largest = Grid {
            frames: 1,
            height: 64,
            width: 64,
        };
        assert_eq!(
            frames(&[largest], 4096, 16).unwrap()[0].patches,
            MAX_PATCHES
        );
        assert_eq!(MAX_PATCHES * MAX_HEADS * MAX_PATCHES * 2, 512 * 1024 * 1024);
    }

    #[test]
    fn item_frame_offsets_do_not_change_attention_boundaries() {
        let first = Grid {
            frames: 2,
            height: 16,
            width: 16,
        };
        let second = Grid {
            frames: 1,
            height: 8,
            width: 16,
        };
        let together = frames(&[first, second], 640, 16).unwrap();
        assert_eq!(
            together
                .iter()
                .map(|f| (f.offset, f.patches))
                .collect::<Vec<_>>(),
            [(0, 256), (256, 256), (512, 128)]
        );
        assert_eq!(
            frames(&[second], 128, 16).unwrap()[0].patches,
            together[2].patches
        );
        assert_eq!(
            frames(&[second, first], 640, 16).unwrap()[1].patches,
            together[0].patches
        );
    }

    #[test]
    fn rejects_all_bad_geometry_before_allocating_or_launching() {
        for grid in [
            Grid {
                frames: 0,
                height: 16,
                width: 16,
            },
            Grid {
                frames: 1,
                height: 15,
                width: 16,
            },
            Grid {
                frames: 1,
                height: 0,
                width: 16,
            },
            Grid {
                frames: 1,
                height: 128,
                width: 64,
            },
            Grid {
                frames: usize::MAX,
                height: 2,
                width: 2,
            },
            Grid {
                frames: 1,
                height: usize::MAX - 1,
                width: 2,
            },
        ] {
            assert!(frames(&[grid], 256, 16).is_err(), "{grid:?}");
        }
        let grid = Grid {
            frames: 1,
            height: 16,
            width: 16,
        };
        assert!(frames(&[grid], 255, 16).is_err());
        assert!(frames(&[grid], 257, 16).is_err());
        assert!(frames(&[], 256, 16).is_err());
        assert!(frames(&[grid], 256, 0).is_err());
        assert!(frames(&[grid], 256, 17).is_err());
        assert!(frames(&[grid], 4097, 16).is_err());
    }
}
