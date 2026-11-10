//! Resident native WeMM: raw text or one encoded image -> GPU BF16 `[1,4096]`.
//!
//! This is the existing tested pipeline, not a model finder, worker/cache,
//! collection mapping or alternate embedder. One caller-selected frozen root
//! supplies BOTH modalities. No CPU tensor math, F16, fallback model, embedding
//! readback or synchronization is hidden here. CPU work is tokenization, image
//! codec and integer input geometry; preparation and model arithmetic are CUDA.
//!
//! The supported contract is deliberately bounded: one user turn, B1, at most
//! 256 total framed token IDs, no truncation; one PNG/JPEG still image, no
//! caption/video/batch, at most 4096x4096 decoded 8-bit pixels and 64MiB encoded
//! bytes. Image processing is the existing aspect-fit/white-pad 256-patch
//! recipe, not arbitrary-resolution upstream processing. An optional explicit
//! crop changes input semantics; no crop is inferred. Unsupported inputs are
//! errors, never successful empty embeddings.

use std::fmt;

use cubecl::cuda::CudaDevice;
use sha2::{Digest, Sha256};
use triblespace::{
    core::repo::BlobStoreGet,
    prelude::{Id, TribleSet},
};

use super::{
    config::Qwen3_5Config,
    image_prepare::{Crop, DecodedRgba, GpuPreparer},
    input_codec::InputCodec,
    multimodal::{CudaTensor, PreparedMultimodal},
    multimodal_layout,
};
use crate::nn::cuda_bf16_alias::{AliasStats, CudaBf16Aliases};

/// Exact WeMM checkpoint configuration used by the native behavior fixtures.
/// This is an asset checksum, not a collection or mapping identity.
pub const CONFIG_SHA256: &str = "34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004";
/// The same configuration read back from facts: the SHA-256 of its
/// [`crate::jsonfacts::canonical_string`], which is what the checkpoint's
/// `config.json` (sha256 [`CONFIG_SHA256`]) canonicalizes to. A configuration
/// stored as facts has no file bytes left to hash, and its canonical text is
/// independent of map order and whitespace, so this pins the same values.
pub const CONFIG_FACTS_SHA256: &str =
    "719264c54581f878babd2d13317ffb1e23f5943dd424e3594a8d107fbd36d4d6";

/// Validated owned assets. Bytes may be exact-fetched by the caller; this
/// type neither opens files nor discovers asset/model records. Construct it
/// BEFORE creating a CUDA alias session if malformed assets must initialize
/// no GPU resources. Other template bytes are refused, not interpreted as Jinja.
pub struct Assets {
    config: Qwen3_5Config,
    codec: InputCodec,
}

impl Assets {
    /// From the checkpoint's own files, as the behavior fixtures use them.
    pub fn from_bytes(config: &[u8], tokenizer: &[u8], template: &[u8]) -> Result<Self, String> {
        if format!("{:x}", Sha256::digest(config)) != CONFIG_SHA256 {
            return Err("unsupported WeMM configuration asset bytes".into());
        }
        let parsed_config =
            Qwen3_5Config::from_json(std::str::from_utf8(config).map_err(|e| e.to_string())?)?;
        multimodal_layout::validate_model(&parsed_config)?;
        let codec = InputCodec::from_assets(tokenizer, template)?;
        Ok(Self {
            config: parsed_config,
            codec,
        })
    }

    /// From the model root itself: the tokenizer, configuration and chat
    /// template that are part of `root`'s identity
    /// ([`crate::ingest::RootParts`]), read from the same frozen facts and
    /// reader the weights come from. No file is opened and nothing is
    /// discovered; a root without all three parts is refused.
    ///
    /// The tokenizer is rebuilt from facts ([`crate::tokenizer::build_tokenizer`])
    /// and so has no file bytes to pin. It is the root's own: selecting the
    /// root selected it, and the root id is derived from its content, so it
    /// cannot be paired with another model's weights the way a side-file
    /// could. The configuration is pinned by [`CONFIG_FACTS_SHA256`] and the
    /// template by its exact bytes, as before.
    pub fn from_model_root<R: BlobStoreGet>(
        facts: &TribleSet,
        reader: &R,
        root: Id,
    ) -> Result<Self, String> {
        let parts = crate::selection::model_root_parts(facts, root).map_err(|e| e.to_string())?;
        let missing = |what: &str| format!("WeMM model root {root:X} has no {what} in its core");
        let tokenizer = parts.tokenizer.ok_or_else(|| missing("tokenizer"))?;
        let config = parts.config.ok_or_else(|| missing("config"))?;
        let template = parts
            .chat_template
            .ok_or_else(|| missing("chat template"))?;
        let config = crate::jsonfacts::load_json(facts, reader, config)
            .map_err(|e| format!("read the WeMM configuration facts: {e}"))?;
        let config = crate::jsonfacts::canonical_string(&config);
        if format!("{:x}", Sha256::digest(config.as_bytes())) != CONFIG_FACTS_SHA256 {
            return Err("unsupported WeMM configuration facts".into());
        }
        let parsed_config = Qwen3_5Config::from_json(&config)?;
        multimodal_layout::validate_model(&parsed_config)?;
        let tokenizer = crate::tokenizer::build_tokenizer(facts, reader, tokenizer)
            .map_err(|e| format!("rebuild the WeMM tokenizer from facts: {e}"))?;
        let template: anybytes::View<str> = reader
            .get(template)
            .map_err(|e| format!("read the WeMM chat template: {e}"))?;
        let codec = InputCodec::from_tokenizer(tokenizer, template.as_bytes())?;
        Ok(Self {
            config: parsed_config,
            codec,
        })
    }

    /// The codec these assets frame input with, for a caller that also cuts
    /// text into windows with it.
    pub fn codec(&self) -> &InputCodec {
        &self.codec
    }
}

/// A failed load does NOT unregister aliases. Keep/reuse the same caller's
/// binder; creating a fresh binder for each retry defeats its registration
/// budget. These counts expose earlier successful registrations on failure.
#[derive(Debug)]
pub struct LoadError {
    pub phase: &'static str,
    pub before: AliasStats,
    pub after: AliasStats,
    message: String,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WeMM {}: {}; alias registrations {} -> {} (drop does not unregister)",
            self.phase, self.message, self.before.registrations, self.after.registrations
        )
    }
}
impl std::error::Error for LoadError {}

/// One resident selected multimodal model. Keep it for repeated queries;
/// constructing one per input is not a cache or lifecycle policy.
pub struct NativeWemm {
    model: PreparedMultimodal,
    codec: InputCodec,
    pixels: GpuPreparer,
    root: Id,
    device: CudaDevice,
}

impl NativeWemm {
    /// Bind once through the caller's existing bounded alias session. `selected`
    /// receives immediate role/handle/shape/payload provenance, not a retained
    /// catalogue. The caller chooses whether/how to record it and the mapping
    /// identity; the opaque root alone does not certify a checkpoint.
    ///
    /// The reader is used only during construction. Exact gets may acquire
    /// missing selected bytes, but selection must remain the caller's frozen
    /// observation. Completed weights retain actual mmap owners through the
    /// existing alias/runtime storage; this facade does not need a live reader
    /// for subsequent forwards and does not re-open or re-select anything.
    ///
    /// # Safety
    /// Forward `PreparedMultimodal::from_pile`'s genuine pile-backed immutable
    /// prefix contract for every leaf AND its preceding partial page until
    /// CUDA runtime teardown. `BlobStoreGet` alone does not prove it. Drop of
    /// the reader, facade, binder, or a returned error does NOT unregister the
    /// runtime's aliases or permit truncating/replacing the backing bytes.
    /// Keep one bounded binder/session across failed/repeated load attempts;
    /// do not repeatedly create fresh binders in a long-lived runtime.
    /// Driver/initialization/allocation faults retain CubeCL's error/panic
    /// behavior, never a CPU or upload fallback.
    pub unsafe fn from_frozen<R: BlobStoreGet>(
        facts: &TribleSet,
        reader: &R,
        root: Id,
        assets: Assets,
        aliases: &mut CudaBf16Aliases,
        selected: impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
    ) -> Result<Self, LoadError> {
        // SAFETY: forwarded unchanged.
        unsafe { Self::from_frozen_with(facts, reader, root, assets, aliases, selected, super::gdn_mixer::DecoderPrecision::Bf16) }
    }

    /// [`Self::from_frozen`] with the decoder's linear projections bound for
    /// `precision`. An NVFP4 precision quantizes each BF16 projection on the
    /// device as it is selected (`gdn_mixer::bind_linear`) and keeps only its
    /// NVFP4 planes, in device memory; the binder never registers those
    /// projections. Everything else binds exactly as `from_frozen` binds it.
    ///
    /// # Safety
    /// As [`Self::from_frozen`].
    pub unsafe fn from_frozen_with<R: BlobStoreGet>(
        facts: &TribleSet,
        reader: &R,
        root: Id,
        assets: Assets,
        aliases: &mut CudaBf16Aliases,
        selected: impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
        precision: super::gdn_mixer::DecoderPrecision,
    ) -> Result<Self, LoadError> {
        // SAFETY: forwarded unchanged.
        unsafe {
            Self::from_frozen_with_vision(facts, reader, root, assets, aliases, selected, precision,
                super::gdn_mixer::DecoderPrecision::Bf16)
        }
    }

    /// [`Self::from_frozen_with`] with the vision tower's linears on `vision`
    /// as well: `W4a4` quantizes each block and merger linear the W4A4-rows
    /// kernel takes at load, exactly as the decoder's (`Tower::from_pile_with`).
    /// No kernel profile names such a model; it is for evaluation.
    ///
    /// # Safety
    /// As [`Self::from_frozen`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_frozen_with_vision<R: BlobStoreGet>(
        facts: &TribleSet,
        reader: &R,
        root: Id,
        assets: Assets,
        aliases: &mut CudaBf16Aliases,
        selected: impl FnMut(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>,
        precision: super::gdn_mixer::DecoderPrecision,
        vision: super::gdn_mixer::DecoderPrecision,
    ) -> Result<Self, LoadError> {
        let before = aliases.stats();
        let pixels = GpuPreparer::new(aliases.device().clone()).map_err(|message| LoadError {
            phase: "image preparation initialization",
            before,
            after: aliases.stats(),
            message,
        })?;
        // SAFETY: exactly the caller's frozen genuine-pile premise above;
        // same supplied binder and no independent model selection.
        let model = unsafe {
            PreparedMultimodal::from_pile_with_vision(facts, reader, root, &assets.config, aliases, selected, precision, vision)
        }
        .map_err(|message| LoadError {
            phase: "model binding",
            before,
            after: aliases.stats(),
            message,
        })?;
        Ok(Self {
            model,
            codec: assets.codec,
            pixels,
            root,
            device: aliases.device().clone(),
        })
    }

    /// (decoder linear projections, their BF16 bytes, their NVFP4 plane
    /// bytes) as bound now.
    pub fn projection_bytes(&self) -> (usize, u64, u64) {
        self.model.projection_bytes()
    }

    /// What every decoder linear projection runs, as actually bound: BF16,
    /// or NVFP4 on a precision, from the root's stored planes or quantized
    /// here. `None` when they differ (no bind produces that). An index's
    /// identity is derived from this, never from the precision asked for.
    pub fn decoder_bound(&self) -> Option<super::gdn_mixer::BoundProjection> {
        self.model.decoder_bound()
    }

    /// The vision tower's linears as bound (`Tower::linear_bytes`): (count,
    /// BF16 count, BF16 bytes, NVFP4 count, NVFP4 plane bytes, kinds seen).
    /// A kernel profile must read this as well as `decoder_bound` before it
    /// names a model whose vision linears are NVFP4.
    pub fn vision_linear_bytes(&self) -> (usize, usize, u64, usize, u64, Vec<super::gdn_mixer::BoundProjection>) {
        self.model.vision_linear_bytes()
    }

    /// Run the decoder's linear projections on `precision` from now on
    /// (`Bf16` changes nothing). The BF16 weights are quantized on the device
    /// with the NVFP4 recipe of `nn::nvfp4_gemv`; their bound BF16 handles
    /// stay registered in the caller's binder. Returns the NVFP4 plane bytes
    /// created. The vision tower is not touched.
    pub fn quantize_decoder(&mut self, precision: super::gdn_mixer::DecoderPrecision) -> Result<u64, String> {
        self.model.quantize_decoder(precision)
    }

    /// Actual selected root. The contributing collection is external store
    /// provenance and must be established separately by the caller.
    pub fn model_root(&self) -> Id {
        self.root
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Query the bound device's actual CUDA name and compute capability;
    /// neither host architecture nor a caller label establishes compute class.
    pub fn device_identity(&self) -> Result<(String, i32, i32), String> {
        device_identity(&self.device)
    }

    /// One raw UTF-8 user text, framed by the exact pinned codec. The total
    /// token limit includes framing and the embedding token; nothing is cut.
    /// `&mut self` keeps this public seam serial B1. The GPU result may still
    /// be in flight; consumers obey the ordinary CubeCL stream contract.
    pub fn embed_text(&mut self, text: &str) -> Result<CudaTensor, String> {
        let tokens = self
            .codec
            .text(text)
            .map_err(|e| format!("WeMM text preparation: {e}"))?;
        Ok(self
            .model
            .embed_text(tokens.as_slice())
            .map_err(|e| format!("WeMM text forward: {e}"))?
            .endpoint
            .embedding)
    }

    /// Several raw texts through ONE batched forward, each framed exactly as
    /// [`Self::embed_text`] frames it and pooled at its own last token. The
    /// batch is right-padded to the longest framed text; padding is never
    /// read by a valid row. Returns one [1,4096] embedding per text, in order.
    /// The caller chooses the batch size, which bounds activation memory.
    pub fn embed_texts(&mut self, texts: &[&str]) -> Result<Vec<CudaTensor>, String> {
        let tokens = texts.iter()
            .map(|text| self.codec.text(text).map_err(|e| format!("WeMM text preparation: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        let sequences: Vec<&[u32]> = tokens.iter().map(|t| t.as_slice()).collect();
        self.model.embed_texts(&sequences).map_err(|e| format!("WeMM batched text forward: {e}"))
    }

    /// One still image with an optional caller-specified crop in original
    /// pixel coordinates. Pass `None` for the complete raster. Prepared pixel
    /// tensors go directly to this SAME model, with no readback/reupload.
    pub fn embed_image(
        &mut self,
        encoded: &[u8],
        crop: Option<Crop>,
    ) -> Result<CudaTensor, String> {
        let tokens = self
            .codec
            .image()
            .map_err(|e| format!("WeMM image tokens: {e}"))?;
        let decoded = DecodedRgba::decode(encoded).map_err(|e| format!("WeMM image codec: {e}"))?;
        let pixels = self
            .pixels
            .prepare(&decoded, crop)
            .map_err(|e| format!("WeMM image preparation: {e}"))?;
        Ok(self
            .model
            .embed_image(tokens.as_slice(), pixels.tensor(), pixels.grid())
            .map_err(|e| format!("WeMM image forward: {e}"))?
            .endpoint
            .embedding)
    }
}

/// Observe hardware before allocating a binder or loading any model weights.
/// Runtime construction also checks the bound model's own device afterwards.
pub fn device_identity(device: &CudaDevice) -> Result<(String, i32, i32), String> {
    use cudarc::driver::{result, sys::CUdevice_attribute};
    result::init().map_err(|e| format!("initialize CUDA device observation: {e}"))?;
    let ordinal = i32::try_from(device.index).map_err(|e| e.to_string())?;
    let device = result::device::get(ordinal).map_err(|e| e.to_string())?;
    let name = result::device::get_name(device).map_err(|e| e.to_string())?;
    // SAFETY: valid device obtained from the initialized CUDA driver.
    let major = unsafe {
        result::device::get_attribute(
            device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .map_err(|e| e.to_string())?;
    let minor = unsafe {
        result::device::get_attribute(
            device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .map_err(|e| e.to_string())?;
    Ok((name, major, minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_assets_fail_before_a_device_or_reader_is_needed() {
        let err = match Assets::from_bytes(b"{}", b"{}", b"not the template") {
            Ok(_) => panic!("unsupported assets accepted"),
            Err(err) => err,
        };
        assert!(err.contains("configuration asset"));
    }

    #[test]
    fn load_failure_keeps_registration_retention_visible() {
        let before = AliasStats {
            registrations: 2,
            ..AliasStats::default()
        };
        let after = AliasStats {
            registrations: 5,
            ..AliasStats::default()
        };
        let error = LoadError {
            phase: "model binding",
            before,
            after,
            message: "missing leaf".into(),
        };
        assert!(error.to_string().contains("missing leaf"));
        assert!(error.to_string().contains("2 -> 5"));
        assert!(error.to_string().contains("does not unregister"));
    }

    // Typecheck the minimal exact-get boundary without creating a GPU client.
    #[allow(dead_code)]
    fn accepts_exact_reader<R: BlobStoreGet>() {
        type Observer = fn(&str, [u8; 32], &[u64], &[u8]) -> Result<(), String>;
        let _: unsafe fn(
            &TribleSet,
            &R,
            Id,
            Assets,
            &mut CudaBf16Aliases,
            Observer,
        ) -> Result<NativeWemm, LoadError> = NativeWemm::from_frozen::<R>;
    }

    #[test]
    #[ignore = "requires reserved CUDA, immutable model pile and retained exact native evidence"]
    fn facade_matches_retained_text_and_images_without_rebinding() {
        use burn::tensor::DType;
        use cubecl::cuda::CudaDevice;
        use std::{collections::BTreeMap, path::PathBuf};

        let path = |name| PathBuf::from(std::env::var(name).expect(name));
        let fixture_bytes = std::fs::read(path("WEMM_FACADE_FIXTURE")).unwrap();
        let fixture: serde_json::Value = serde_json::from_slice(&fixture_bytes).unwrap();
        let expected: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path("WEMM_FACADE_NATIVE")).unwrap()).unwrap();
        assert_eq!(expected["engine"], "native-CUDA-BF16");
        assert_eq!(
            expected["fixture_sha256"],
            format!("{:x}", Sha256::digest(&fixture_bytes))
        );
        let assets_dir = path("WEMM_CHECKPOINT_DIR");
        let assets = Assets::from_bytes(
            &std::fs::read(assets_dir.join("config.json")).unwrap(),
            &std::fs::read(assets_dir.join("tokenizer.json")).unwrap(),
            &std::fs::read(assets_dir.join("chat_template.jinja")).unwrap(),
        )
        .unwrap();
        assert!(assets.codec.text(&"x ".repeat(1024)).is_err());
        let model = crate::persist::read_model_pile_read_only(&path("WEMM_FACADE_MODEL")).unwrap();
        let root = Id::from_hex(&std::env::var("WEMM_FACADE_ROOT").unwrap()).unwrap();
        assert_eq!(expected["model_root"], format!("{root:?}"));
        let mut aliases = CudaBf16Aliases::new(CudaDevice { index: 0 }, 759).unwrap();
        // Scratch for this one test invocation, not retained model state.
        let roles: BTreeMap<_, _> = expected["selected_roles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| (v["name"].as_str().unwrap(), v))
            .collect();
        assert_eq!(roles.len(), 759);
        let mut selected = 0;
        // SAFETY: explicit immutable pile fixture remains intact through
        // this process's runtime teardown; no test rewrites/deletes it.
        let mut native = unsafe {
            NativeWemm::from_frozen(
                &model.facts,
                &model.store,
                root,
                assets,
                &mut aliases,
                |name, handle, shape, bytes| {
                    let want = roles
                        .get(name)
                        .ok_or_else(|| format!("unexpected role {name}"))?;
                    assert_eq!(
                        want["handle"],
                        handle
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<String>()
                    );
                    assert_eq!(want["shape"], serde_json::json!(shape));
                    assert_eq!(want["sha256"], format!("{:x}", Sha256::digest(bytes)));
                    selected += 1;
                    Ok(())
                },
            )
        }
        .unwrap();
        assert_eq!(native.model_root(), root);
        assert_eq!(native.device(), aliases.device());
        assert_eq!(
            native.device_identity().unwrap(),
            ("NVIDIA GB10".into(), 12, 1)
        );
        assert_eq!(selected, 759);
        assert_eq!(aliases.stats().registrations, 759);
        let bindings = aliases.stats();
        drop(model); // The actual registered mmap owners, not this reader, keep weights alive.
        let items = fixture["items"].as_array().unwrap();
        assert_eq!(items.len(), 11);
        for reverse in [false, true] {
            for position in 0..items.len() {
                let item = &items[if reverse {
                    items.len() - 1 - position
                } else {
                    position
                }];
                let output = if item["modality"] == "text" {
                    native.embed_text(item["text"].as_str().unwrap()).unwrap()
                } else {
                    assert_eq!(item["modality"], "image");
                    let bytes = std::fs::read(item["source_path"].as_str().unwrap()).unwrap();
                    assert_eq!(item["sha256"], format!("{:x}", Sha256::digest(&bytes)));
                    let crop = item["crop_xyxy"].as_array().map(|v| Crop {
                        left: v[0].as_u64().unwrap() as usize,
                        top: v[1].as_u64().unwrap() as usize,
                        right: v[2].as_u64().unwrap() as usize,
                        bottom: v[3].as_u64().unwrap() as usize,
                    });
                    native.embed_image(&bytes, crop).unwrap()
                };
                assert_eq!(output.dtype, DType::BF16);
                assert_eq!(output.meta.shape().as_slice(), &[1, 4096]);
                let raw = output.client.read_one(output.handle.clone()).unwrap(); // TEST-ONLY byte observation
                let want = expected["embeddings"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|v| v["id"] == item["id"] && v["pass"] == "forward")
                    .unwrap();
                let expected_bits: Vec<u8> = want["bits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|v| u16::try_from(v.as_u64().unwrap()).unwrap().to_le_bytes())
                    .collect();
                let raw: &[u8] = raw.as_ref();
                assert_eq!(raw, expected_bits.as_slice(), "{}", item["id"]);
                assert_eq!(
                    aliases.stats(),
                    bindings,
                    "forward must not bind model weights again"
                );
            }
        }
        assert!(
            native
                .embed_text(&"x ".repeat(1024))
                .err()
                .unwrap()
                .contains("preparation")
        );
        assert!(
            native
                .embed_image(b"not an image", None)
                .err()
                .unwrap()
                .contains("codec")
        );
    }
}
