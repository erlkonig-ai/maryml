//! Inkling 2 Mini's image front end: a decoded 8-bit image in, the vision
//! tower's input rows out, on the CPU.
//!
//! Like the original Inkling's [`crate::models::inkling::patches`] this is a
//! tokenizer, not model math: an alpha composite, a pad, one affine map and a
//! reindexing, with no learned parameter. The codec stays with the caller (as
//! Drive decodes for the original Inkling); what arrives here is raw RGB8 or
//! RGBA8. Read off the published `Inkling2ImageProcessor`
//! (`image_processing_inkling2.py` at transformers 046bdb8, configured by the
//! release's `processor_config.json`) and the patch reshape at the top of
//! `Inkling2VisionEmbeddings.forward`. It differs from the original Inkling's
//! front end in four places that are easy to carry over by mistake:
//!
//! - an image with transparency is composited onto a grey checkerboard of
//!   8-pixel squares (204 where `(y / 8 + x / 8)` is even, 153 where odd) as
//!   `(c * a + bg * (255 - a) + 127) / 255` in integers, before anything else.
//!   An opaque pixel composites to itself, so an RGBA8 raster of an image
//!   without transparency gives the same values as its RGB8 raster;
//! - the grid is `ceil(height / 40)` by `ceil(width / 40)`: an exact multiple
//!   of forty gets no pad and no extra column;
//! - the pad is `-1` on the raw 0..255 scale, then normalised like a pixel;
//!   the normalisation is the processor's fused one, `(v - 127.5) / 127.5` in
//!   f32 (mean and std 0.5 over a rescale of 1/255);
//! - there is no time axis: a token is `[40][40][rgb]`, and the tower's linear
//!   patch embedding (in_features 1200) takes 20-pixel native patches, four per
//!   token in `(my, mx)` order, each `[y 0..20][x 0..20][rgb]`. The config's
//!   `temporal_patch_size` (2) plays no part for a still image.
//!
//! Multiple images are the concatenation of their tokens, each image with its
//! own grid, as the reference batches them.

use super::config::Inkling2VisionConfig;

const CHANNELS: usize = 3;
/// The processor's mean and std (0.5) over its rescale (1/255), fused as the
/// torchvision backend fuses them: `(v - 127.5) / 127.5`.
const CENTRE: f32 = 127.5;
/// The pad on the raw 0..255 scale, applied before the normalisation.
const PAD_RAW: f32 = -1.0;
/// Side of the checkerboard squares an image with transparency is put on.
const CHECKER: usize = 8;
const CHECKER_EVEN: u32 = 204;
const CHECKER_ODD: u32 = 153;

/// A decoded 8-bit image, row-major, top-left first.
#[derive(Clone, Copy, Debug)]
pub enum Pixels<'a> {
    /// `width * height * 3` bytes: an image without transparency.
    Rgb8(&'a [u8]),
    /// `width * height * 4` bytes, straight (not premultiplied) alpha: an
    /// image with transparency, composited onto the checkerboard.
    Rgba8(&'a [u8]),
}

#[derive(Clone, Copy, Debug)]
pub struct Raster<'a> {
    pub width: usize,
    pub height: usize,
    pub pixels: Pixels<'a>,
}

/// The tower's patch geometry from `vision_config`: a token is a
/// `merge x merge` block of `native x native` patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub native_patch: usize,
    pub merge: usize,
}

impl Geometry {
    pub fn of(cfg: &Inkling2VisionConfig) -> Result<Self, String> {
        if cfg.native_patch_size == 0 || cfg.adapter_merge_size == 0 {
            return Err(
                "vision_config: native_patch_size and adapter_merge_size must be positive".into(),
            );
        }
        Ok(Self {
            native_patch: cfg.native_patch_size,
            merge: cfg.adapter_merge_size,
        })
    }

    /// Side of one image token's square of pixels (40 for the release).
    pub fn tile(&self) -> usize {
        self.native_patch * self.merge
    }

    /// `(rows, cols)` of image tokens over a `width x height` image.
    pub fn grid(&self, width: usize, height: usize) -> (usize, usize) {
        (height.div_ceil(self.tile()), width.div_ceil(self.tile()))
    }

    /// f32 values in one image token, `tile * tile * 3`.
    pub fn token_values(&self) -> usize {
        self.tile() * self.tile() * CHANNELS
    }

    /// f32 values in one native patch row, `native * native * 3` (1200).
    pub fn native_values(&self) -> usize {
        self.native_patch * self.native_patch * CHANNELS
    }
}

/// One image's processor output.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageTokens {
    pub geometry: Geometry,
    /// `(rows, cols)` of image tokens; the reference's `image_grid_thw` is
    /// `[1, rows, cols]`.
    pub grid: (usize, usize),
    /// The reference's `pixel_values`: `rows * cols` tokens in raster order,
    /// each `[y 0..tile][x 0..tile][rgb]`.
    pub pixel_values: Vec<f32>,
}

/// The processor on one decoded image.
pub fn preprocess(geometry: Geometry, raster: Raster<'_>) -> Result<ImageTokens, String> {
    let Raster {
        width,
        height,
        pixels,
    } = raster;
    if width == 0 || height == 0 {
        return Err(format!("an image needs pixels; {width}x{height} has none"));
    }
    let area = width
        .checked_mul(height)
        .ok_or_else(|| format!("{width}x{height} overflows"))?;
    let (bytes, stride) = match pixels {
        Pixels::Rgb8(b) => (b, 3),
        Pixels::Rgba8(b) => (b, 4),
    };
    if area.checked_mul(stride) != Some(bytes.len()) {
        return Err(format!(
            "a {width}x{height} {} raster is {} bytes; {} arrived",
            if stride == 3 { "RGB8" } else { "RGBA8" },
            area.saturating_mul(stride),
            bytes.len()
        ));
    }

    // Every value the output can hold: 256 pixel values and the pad.
    let norm = |raw: f32| (raw - CENTRE) / CENTRE;
    let table: [f32; 256] = std::array::from_fn(|v| norm(v as f32));
    let pad = norm(PAD_RAW);

    let tile = geometry.tile();
    let (rows, cols) = geometry.grid(width, height);
    let mut out = Vec::with_capacity(rows * cols * geometry.token_values());
    for r in 0..rows {
        for c in 0..cols {
            for y in 0..tile {
                let py = r * tile + y;
                for x in 0..tile {
                    let px = c * tile + x;
                    if py >= height || px >= width {
                        out.extend_from_slice(&[pad; CHANNELS]);
                        continue;
                    }
                    let at = (py * width + px) * stride;
                    match pixels {
                        Pixels::Rgb8(_) => {
                            for ch in 0..CHANNELS {
                                out.push(table[bytes[at + ch] as usize]);
                            }
                        }
                        Pixels::Rgba8(_) => {
                            let a = u32::from(bytes[at + 3]);
                            let bg = if (py / CHECKER + px / CHECKER) % 2 == 0 {
                                CHECKER_EVEN
                            } else {
                                CHECKER_ODD
                            };
                            for ch in 0..CHANNELS {
                                let v =
                                    (u32::from(bytes[at + ch]) * a + bg * (255 - a) + 127) / 255;
                                out.push(table[v as usize]);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(ImageTokens {
        geometry,
        grid: (rows, cols),
        pixel_values: out,
    })
}

impl ImageTokens {
    pub fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1
    }

    /// The reference's `patch_grid_thw` row: `[1, rows * merge, cols * merge]`.
    pub fn patch_grid_thw(&self) -> [usize; 3] {
        let m = self.geometry.merge;
        [1, self.grid.0 * m, self.grid.1 * m]
    }

    /// The rows the tower's linear patch embedding multiplies:
    /// `tokens * merge^2` rows of `native * native * 3`, token-major, the
    /// token's native patches in `(my, mx)` order, each `[y][x][rgb]`. The
    /// reference's `pixel_values.reshape(-1, m, p, m, p, 3).transpose(2, 3)`.
    pub fn native_patches(&self) -> Vec<f32> {
        let Geometry {
            native_patch: p,
            merge: m,
        } = self.geometry;
        let tile = self.geometry.tile();
        let mut out = Vec::with_capacity(self.pixel_values.len());
        for token in self.pixel_values.chunks_exact(self.geometry.token_values()) {
            for my in 0..m {
                for mx in 0..m {
                    for y in 0..p {
                        let row = (my * p + y) * tile + mx * p;
                        out.extend_from_slice(&token[row * CHANNELS..(row + p) * CHANNELS]);
                    }
                }
            }
        }
        out
    }

    /// `(row, col)` of each native patch row in the image's native grid, in
    /// [`Self::native_patches`]'s order: the reference's
    /// `get_vision_position_ids(patch_grid_thw, merge)` for one image. The
    /// tower puts a class token before them, at `(0, 0)`, and these 1-indexed.
    pub fn native_positions(&self) -> Vec<[usize; 2]> {
        let m = self.geometry.merge;
        let (rows, cols) = self.grid;
        let mut out = Vec::with_capacity(self.tokens() * m * m);
        for r in 0..rows {
            for c in 0..cols {
                for my in 0..m {
                    for mx in 0..m {
                        out.push([r * m + my, c * m + mx]);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "vision_input_tests.rs"]
mod tests;
