//! Bounded still-image preparation: codec RGBA -> CUDA aspect-fit 256,
//! white alpha/padding, F32 antialiased bicubic, normalization, BF16 patches.
//!
//! CPU does codec and integer geometry only. Every pixel operation is CUDA.
//! The native model consumes the returned GPU tensor directly; typed leaf
//! export is optional byte transport, never a CPU numerical implementation.
//! This 256-patch budget is not arbitrary-resolution upstream preprocessing.
//!
//! Filter/edge/order reference: PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb,
//! aten/src/ATen/native/cuda/{UpSample.cuh,UpSampleBilinear2d.cu}: a=-0.5,
//! clipped normalized support, horizontal F32 before vertical F32. Equality
//! to the retained Torch CUDA BF16 fixtures is an explicit unexecuted gate.
use super::vision_geometry::Grid;
use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    Runtime,
    client::ComputeClient,
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
};
use half::bf16;
use triblespace::core::blob::{
    Blob,
    encodings::tensor::{Tensor, elements::BF16},
};

pub struct DecodedRgba {
    width: usize,
    height: usize,
    bytes: Vec<u8>,
}
impl DecodedRgba {
    /// CPU image codec only; formats are the crate's PNG/JPEG feature set.
    /// No CPU crop/resize/alpha composition, HDR/tone mapping or fallback.
    pub fn decode(encoded: &[u8]) -> Result<Self, String> {
        if encoded.len() > 64 * 1024 * 1024 {
            return Err("encoded image exceeds64MiB".into());
        }
        let mut reader = image::ImageReader::new(std::io::Cursor::new(encoded))
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(4096);
        limits.max_image_height = Some(4096);
        limits.max_alloc = Some(256 * 1024 * 1024);
        reader.limits(limits);
        let image = reader.decode().map_err(|e| e.to_string())?;
        let color = image.color();
        if color.bits_per_pixel() / u16::from(color.channel_count()) != 8 {
            return Err("only 8-bit decoded channels are supported; no CPU tone mapping".into());
        }
        let image = image.into_rgba8();
        Self::from_rgba(
            image.width() as usize,
            image.height() as usize,
            image.into_raw(),
        )
    }
    pub fn from_rgba(width: usize, height: usize, bytes: Vec<u8>) -> Result<Self, String> {
        if !(1..=4096).contains(&width)
            || !(1..=4096).contains(&height)
            || bytes.len() != width * height * 4
        {
            return Err("RGBA requires exact bytes and width/height1..4096".into());
        }
        Ok(Self {
            width,
            height,
            bytes,
        })
    }
}

/// Exclusive right/bottom bounds in the original decoded raster; no guessing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crop {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub original: [usize; 2],
    pub crop: Crop,
    pub resized: [usize; 2],
    pub pad_left_top: [usize; 2],
}
impl Geometry {
    pub fn new(width: usize, height: usize, crop: Option<Crop>) -> Result<Self, String> {
        if !(1..=4096).contains(&width) || !(1..=4096).contains(&height) {
            return Err("image extent1..4096 required".into());
        }
        let crop = crop.unwrap_or(Crop {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        });
        if crop.left >= crop.right
            || crop.top >= crop.bottom
            || crop.right > width
            || crop.bottom > height
        {
            return Err("crop outside decoded image or empty".into());
        }
        let w = crop.right - crop.left;
        let h = crop.bottom - crop.top;
        let denominator = w.max(h);
        // Exact integer half-even rounding of rational dimensions, matching
        // the tested Python round geometry without CPU pixel/tensor arithmetic.
        let resize = |n: usize| {
            let v = n * 256;
            let q = v / denominator;
            let r = v % denominator;
            (q + usize::from(r * 2 > denominator || (r * 2 == denominator && q % 2 == 1))).max(1)
        };
        let resized = [resize(w), resize(h)];
        Ok(Self {
            original: [width, height],
            crop,
            resized,
            pad_left_top: [(256 - resized[0]) / 2, (256 - resized[1]) / 2],
        })
    }
}

/// Always contiguous native BF16[256,1536], constructed only by this module.
pub struct PreparedPixels {
    tensor: CubeTensor<CudaRuntime>,
    pub geometry: Geometry,
}
impl PreparedPixels {
    pub fn tensor(&self) -> &CubeTensor<CudaRuntime> {
        &self.tensor
    }
    pub fn grid(&self) -> Grid {
        Grid {
            frames: 1,
            height: 16,
            width: 16,
        }
    }
    /// Explicit byte-export seam for persistence/oracles. The default model
    /// path passes tensor() directly, with no readback/reupload of pixels.
    pub fn to_leaf(&self) -> Result<Blob<Tensor<BF16, 2>>, String> {
        let raw = self
            .tensor
            .client
            .read_one(self.tensor.handle.clone())
            .map_err(|e| format!("pixels readback: {e:?}"))?
            .to_vec();
        crate::leaf::leaf_blob::<BF16, 2>([256, 1536], raw.into()).map_err(|e| e.to_string())
    }
}
pub struct GpuPreparer {
    client: ComputeClient<CudaRuntime>,
    device: CudaDevice,
}
impl GpuPreparer {
    pub fn new(device: CudaDevice) -> Result<Self, String> {
        if device.index > u16::MAX as usize {
            return Err("CUDA ordinal exceeds runtime domain".into());
        }
        Ok(Self {
            client: CudaRuntime::client(&device),
            device,
        })
    }
    pub fn prepare(
        &self,
        image: &DecodedRgba,
        crop: Option<Crop>,
    ) -> Result<PreparedPixels, String> {
        let g = Geometry::new(image.width, image.height, crop)?;
        let w = g.crop.right - g.crop.left;
        let h = g.crop.bottom - g.crop.top;
        let [nw, nh] = g.resized;
        let n = w * h * 3;
        let rgba = self.client.create_from_slice(&image.bytes); // codec bytes, not CPU model values
        let foreground = self.client.empty(n * 4);
        let background = self.client.empty(n * 4);
        let rgb = self.client.empty(n * 4);
        let horizontal = self.client.empty(nw * h * 3 * 4);
        let resized = self.client.empty(nw * nh * 3 * 4);
        let scaled = self.client.empty(3 * 256 * 256 * 4);
        let packed = self.client.empty(256 * 1536 * 2);
        let cube = CubeDim::new_1d(64);
        let launch = |n| cubecl::calculate_cube_count_elemwise(&self.client, n, cube);
        // SAFETY: all input extents validated, bounded below u32, each output
        // fresh/disjoint, each lane owns one cell, crop/packing indices bounded.
        unsafe {
            composite_parts::launch_unchecked::<CudaRuntime>(
                &self.client,
                launch(n),
                cube,
                ArrayArg::from_raw_parts(rgba, image.bytes.len()),
                ArrayArg::from_raw_parts(foreground.clone(), n),
                ArrayArg::from_raw_parts(background.clone(), n),
                n,
                image.width,
                w,
                h,
                g.crop.left,
                g.crop.top,
            );
            composite_sum::launch_unchecked::<CudaRuntime>(
                &self.client,
                launch(n),
                cube,
                ArrayArg::from_raw_parts(foreground, n),
                ArrayArg::from_raw_parts(background, n),
                ArrayArg::from_raw_parts(rgb.clone(), n),
                n,
            );
            resample::launch_unchecked::<CudaRuntime>(
                &self.client,
                launch(nw * h * 3),
                cube,
                ArrayArg::from_raw_parts(rgb, n),
                ArrayArg::from_raw_parts(horizontal.clone(), nw * h * 3),
                nw * h * 3,
                w,
                h,
                nw,
                h,
                true,
            );
            resample::launch_unchecked::<CudaRuntime>(
                &self.client,
                launch(nw * nh * 3),
                cube,
                ArrayArg::from_raw_parts(horizontal, nw * h * 3),
                ArrayArg::from_raw_parts(resized.clone(), nw * nh * 3),
                nw * nh * 3,
                nw,
                h,
                nw,
                nh,
                false,
            );
            pad_scale::launch_unchecked::<CudaRuntime>(
                &self.client,
                launch(3 * 256 * 256),
                cube,
                ArrayArg::from_raw_parts(resized, nw * nh * 3),
                ArrayArg::from_raw_parts(scaled.clone(), 3 * 256 * 256),
                nw,
                nh,
                g.pad_left_top[0],
                g.pad_left_top[1],
            );
            pack::launch_unchecked::<CudaRuntime>(
                &self.client,
                launch(256 * 1536),
                cube,
                ArrayArg::from_raw_parts(scaled, 3 * 256 * 256),
                ArrayArg::from_raw_parts(packed.clone(), 256 * 1536),
            );
        }
        Ok(PreparedPixels {
            geometry: g,
            tensor: CubeTensor::new_contiguous(
                self.client.clone(),
                self.device.clone(),
                [256, 1536].as_slice().into(),
                packed,
                DType::BF16,
            ),
        })
    }
}

#[cube(launch_unchecked)]
fn composite_parts(
    rgba: &Array<u8>,
    fg: &mut Array<f32>,
    bg: &mut Array<f32>,
    n: usize,
    full_w: usize,
    w: usize,
    h: usize,
    left: usize,
    top: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let c = i / (w * h);
        let y = (i / w) % h;
        let x = i % w;
        let p = ((y + top) * full_w + x + left) * 4;
        let a = f32::cast_from(rgba[p + 3]) / 255.0;
        fg[i] = f32::cast_from(rgba[p + c]) * a;
        bg[i] = 255.0 * (1.0 - a);
    }
}
#[cube(launch_unchecked)]
fn composite_sum(fg: &Array<f32>, bg: &Array<f32>, rgb: &mut Array<f32>, n: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        rgb[i] = fg[i] + bg[i];
    }
}
#[cube]
fn cubic(distance: f32) -> f32 {
    let x = distance.abs();
    let mut w = 0.0f32;
    if x < 1.0 {
        w = ((1.5 * x - 2.5) * x) * x + 1.0;
    } else if x < 2.0 {
        w = (((x - 5.0) * x + 8.0) * x - 4.0) * (-0.5);
    }
    w
}
#[cube(launch_unchecked)]
fn resample(
    input: &Array<f32>,
    out: &mut Array<f32>,
    n: usize,
    iw: usize,
    ih: usize,
    ow: usize,
    oh: usize,
    #[comptime] horizontal: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n {
        let c = i / (ow * oh);
        let y = (i / ow) % oh;
        let x = i % ow;
        let mut size = ih;
        let mut dst = oh;
        let mut p = y;
        if horizontal {
            size = iw;
            dst = ow;
            p = x;
        }
        let scale = f32::cast_from(size) / f32::cast_from(dst);
        let center = scale * (f32::cast_from(p) + 0.5);
        let filter_scale = scale.max(1.0);
        let support = 2.0 * filter_scale;
        let lo = usize::cast_from((center - support + 0.5).max(0.0));
        let hi = usize::cast_from((center + support + 0.5).min(f32::cast_from(size)));
        let inverse = 1.0 / filter_scale;
        let mut total = 0.0f32;
        for j in lo..hi {
            total += cubic((f32::cast_from(j) - center + 0.5) * inverse);
        }
        let mut value = 0.0f32;
        for j in lo..hi {
            let mut weight = cubic((f32::cast_from(j) - center + 0.5) * inverse);
            if total != 0.0 {
                weight /= total;
            }
            let mut offset = (c * ih + j) * iw + x;
            if horizontal {
                offset = (c * ih + y) * iw + j;
            }
            let term = input[offset] * weight;
            if j == lo {
                value = term;
            } else {
                value += term;
            }
        }
        out[i] = value;
    }
}
#[cube(launch_unchecked)]
fn pad_scale(
    image: &Array<f32>,
    out: &mut Array<f32>,
    w: usize,
    h: usize,
    left: usize,
    top: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < 3 * 256 * 256 {
        let c = i / (256 * 256);
        let y = (i / 256) % 256;
        let x = i % 256;
        let mut v = 255.0f32;
        if x >= left && x < left + w && y >= top && y < top + h {
            v = image[(c * h + y - top) * w + x - left].max(0.0).min(255.0);
        }
        out[i] = v / 255.0;
    }
}
#[cube(launch_unchecked)]
fn pack(scaled: &Array<f32>, out: &mut Array<bf16>) {
    let i = ABSOLUTE_POS as usize;
    if i < 256 * 1536 {
        let row = i / 1536;
        let q = i % 1536;
        let x = (((row / 4) % 8) * 2 + row % 2) * 16 + q % 16;
        let y = ((row / 32) * 2 + (row / 2) % 2) * 16 + (q / 16) % 16;
        let c = q / 512; // q's temporal index deliberately repeats the same frame
        out[i] = bf16::cast_from((scaled[c * 256 * 256 + y * 256 + x] - 0.5) / 0.5);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    pub(crate) fn fixture() -> serde_json::Value {
        let path = std::env::var("WEMM_PREPARED_FIXTURE").expect("reviewed prepared.json");
        let expected =
            std::env::var("WEMM_PREPARED_FIXTURE_SHA256").expect("reviewed exact SHA256");
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected);
        serde_json::from_slice(&bytes).unwrap()
    }
    #[test]
    fn geometry_is_explicit_and_rejects_empty_crops() {
        let g = Geometry::new(
            594,
            768,
            Some(Crop {
                left: 132,
                top: 395,
                right: 463,
                bottom: 562,
            }),
        )
        .unwrap();
        assert_eq!(g.resized, [256, 129]);
        assert_eq!(g.pad_left_top, [0, 63]);
        assert!(Geometry::new(0, 1, None).is_err());
        assert!(
            Geometry::new(
                10,
                10,
                Some(Crop {
                    left: 2,
                    top: 0,
                    right: 2,
                    bottom: 8
                })
            )
            .is_err()
        );
    }
    #[test]
    fn geometry_preserves_small_image_aspect_and_bounds() {
        assert_eq!(Geometry::new(128, 94, None).unwrap().resized, [256, 188]);
        assert_eq!(Geometry::new(4096, 1, None).unwrap().resized, [256, 1]);
        assert!(DecodedRgba::from_rgba(2, 2, vec![0; 15]).is_err());
    }
    #[test]
    #[ignore = "CUDA only; no CPU fallback"]
    fn transparent_pixel_is_white_and_repeat_bits_are_identical() {
        let p = GpuPreparer::new(CudaDevice { index: 0 }).unwrap();
        let input = DecodedRgba::from_rgba(1, 1, vec![37, 129, 211, 0]).unwrap();
        let a = p.prepare(&input, None).unwrap().to_leaf().unwrap();
        let b = p.prepare(&input, None).unwrap().to_leaf().unwrap();
        let a = crate::leaf::read_leaf(a).unwrap();
        let b = crate::leaf::read_leaf(b).unwrap();
        assert_eq!(&a.payload()[..], &b.payload()[..]);
        assert!(a.payload().chunks_exact(2).all(|x| x == [0x80, 0x3f]));
    }
    #[test]
    #[ignore = "CUDA + exact retained Torch fixture; reports genuine preprocessing mismatch"]
    fn pixels_match_the_retained_torch_cuda_fixture() {
        let f = fixture();
        let p = GpuPreparer::new(CudaDevice { index: 0 }).unwrap();
        for item in f["items"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["modality"] == "image")
        {
            let bytes = std::fs::read(item["source_path"].as_str().unwrap()).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                item["sha256"].as_str().unwrap()
            );
            let crop = item["crop_xyxy"].as_array().map(|v| Crop {
                left: v[0].as_u64().unwrap() as usize,
                top: v[1].as_u64().unwrap() as usize,
                right: v[2].as_u64().unwrap() as usize,
                bottom: v[3].as_u64().unwrap() as usize,
            });
            let prepared = p
                .prepare(&DecodedRgba::decode(&bytes).unwrap(), crop)
                .unwrap();
            let blob = prepared.to_leaf().unwrap();
            let view = crate::leaf::read_leaf(blob).unwrap();
            let expected = std::fs::read(item["pixels_path"].as_str().unwrap()).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&expected)),
                item["pixels_sha256"].as_str().unwrap()
            );
            assert_eq!(view.payload().len(), expected.len());
            let differing = view
                .payload()
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(differing, 0, "{} BF16 pixel words differ", item["id"]);
        }
    }
}
