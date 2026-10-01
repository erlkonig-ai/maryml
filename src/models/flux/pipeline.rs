use std::collections::HashMap;
use std::path::Path;

use crate::model_collection::ModelSnapshot;
use burn::prelude::*;
use burn::tensor::TensorData;
use triblespace::prelude::BlobStoreGet;

use crate::leaf::Leaf;
use crate::models::flux::mistral_encoder::Mistral3Model;
use crate::models::flux::mistral_encoder::config::Mistral3Config;
use crate::models::flux::scheduler::FlowMatchEulerDiscreteScheduler;
use crate::models::flux::text_encoder::Qwen3Model;
use crate::models::flux::text_encoder::config::Qwen3Config;
use crate::models::flux::tokenizer::{MistralTokenizer, Qwen2Tokenizer};
use crate::models::flux::transformer::Flux2Transformer2DModel;
use crate::models::flux::transformer::config::Flux2TransformerConfig;
use crate::models::flux::utils;
use crate::models::flux::vae::AutoencoderKLFlux2;
use crate::models::flux::vae::config::VaeConfig;
use crate::nn::backend::{B, BHalf, WgpuDevice};
use crate::nn::weight_loader::WeightLoader;
use crate::selection::ModelSelector;

const TEXT_ENCODER: &str = "text_encoder";
const TRANSFORMER: &str = "transformer";
const VAE: &str = "vae";

/// Model variant auto-detected from directory contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    /// FLUX.2-klein-4B: Qwen3 text encoder, no guidance, step-distilled
    Klein,
    /// FLUX.2-dev: Mistral3 text encoder, guidance conditioning
    Dev,
}

impl ModelVariant {
    /// Detect the variant from the text-encoder config shipped beside the
    /// durable weight collection.
    pub fn detect(model_dir: &Path) -> anyhow::Result<Self> {
        let path = model_dir.join(TEXT_ENCODER).join("config.json");
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        Ok(
            match json.get("model_type").and_then(|value| value.as_str()) {
                Some("mistral3") => Self::Dev,
                _ => Self::Klein,
            },
        )
    }

    /// Canonical source coordinate shared by all components of this variant.
    pub const fn source(self) -> &'static str {
        match self {
            Self::Klein => "black-forest-labs/FLUX.2-klein-4B",
            Self::Dev => "black-forest-labs/FLUX.2-dev",
        }
    }

    /// Source coordinate of one independently imported component.
    pub fn component_source(self, component: &str) -> String {
        format!("{}#{component}", self.source())
    }
}

/// The three FLUX weight components selected from one frozen native model
/// collection snapshot.
///
/// Only the compact tensor indexes stay resident. Each phase materializes its
/// component, then drops the resulting keymap before the next phase.
/// Construction validates all three source coordinates up front, so a mixed,
/// missing, or ambiguous local model cohort fails before GPU execution starts.
///
/// No reader is retained: a leaf holds its bytes as a view over the pile's
/// mapping and keeps that mapping alive by itself.
pub struct FluxWeights {
    variant: ModelVariant,
    text_encoder: HashMap<String, Leaf>,
    transformer: HashMap<String, Leaf>,
    vae: HashMap<String, Leaf>,
}

impl FluxWeights {
    /// Select the text encoder, transformer, and VAE components from one
    /// immutable native model-collection snapshot. A component may span
    /// several real roots (the Klein text encoder has two weight shards).
    pub fn from_snapshot<R: BlobStoreGet>(
        snapshot: ModelSnapshot<R>,
        variant: ModelVariant,
    ) -> anyhow::Result<Self> {
        fn select_component(
            facts: &triblespace::prelude::TribleSet,
            reader: &impl BlobStoreGet,
            variant: ModelVariant,
            component: &str,
        ) -> anyhow::Result<HashMap<String, Leaf>> {
            let source = variant.component_source(component);
            let roots = crate::selection::select_model_roots_acquiring(
                facts,
                reader,
                ModelSelector::Source {
                    source: &source,
                    quantization: crate::persist::QUANTIZATION_NATIVE,
                },
            )?;
            crate::selection::index_keymap_for_roots(facts, reader, &roots)
        }

        let text_encoder =
            select_component(snapshot.facts(), snapshot.store(), variant, TEXT_ENCODER)?;
        let transformer =
            select_component(snapshot.facts(), snapshot.store(), variant, TRANSFORMER)?;
        let vae = select_component(snapshot.facts(), snapshot.store(), variant, VAE)?;
        drop(snapshot);
        Ok(Self {
            variant,
            text_encoder,
            transformer,
            vae,
        })
    }

    pub const fn variant(&self) -> ModelVariant {
        self.variant
    }

    fn materialize(
        &self,
        index: &HashMap<String, Leaf>,
    ) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        index
            .iter()
            .map(|(name, leaf)| (name.clone(), leaf.to_f32_shape()))
            .collect()
    }

    fn text_encoder(&self) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        self.materialize(&self.text_encoder)
    }

    fn transformer(&self) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        self.materialize(&self.transformer)
    }

    fn vae(&self) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        self.materialize(&self.vae)
    }
}

/// Unified FLUX.2 inference pipeline (supports both Klein and Dev).
pub struct Flux2Pipeline;

impl Flux2Pipeline {
    /// Generate an image from a text prompt.
    ///
    /// WEIGHTS load per-component from one frozen [`FluxWeights`] snapshot.
    /// `model_dir` supplies only the small side-files: configs + tokenizer.
    /// NOTE: the Dev streaming path materializes the transformer keymap in host
    /// RAM before streaming blocks to the GPU — a lazy handle-backed loader is
    /// the known follow-up before Dev-scale (60GB) models are practical.
    ///
    /// Loads models sequentially to minimize memory usage:
    /// 1. Tokenize + text encode → drop text encoder
    /// 2. Load transformer → denoise → drop transformer
    /// 3. Load VAE → decode → output image
    ///
    /// Auto-detects Klein vs Dev from the text-encoder config.
    pub fn generate<B: Backend>(
        prompt: &str,
        height: usize,
        width: usize,
        num_steps: usize,
        guidance_scale: f32,
        seed: u64,
        model_dir: &Path,
        weights: &FluxWeights,
        lora_path: Option<&Path>,
        device: &B::Device,
    ) -> image::RgbImage {
        let vae_scale_factor: usize = 8;

        // Align dimensions to be divisible by vae_scale_factor * 2 = 16
        let latent_h = 2 * (height / (vae_scale_factor * 2));
        let latent_w = 2 * (width / (vae_scale_factor * 2));

        let variant = ModelVariant::detect(model_dir)
            .unwrap_or_else(|error| panic!("detect FLUX model variant: {error:#}"));
        assert_eq!(
            variant,
            weights.variant(),
            "FLUX config variant and selected native weights disagree"
        );
        let transformer_config_path = model_dir.join("transformer").join("config.json");
        let mut transformer_config = Flux2TransformerConfig::load(&transformer_config_path);
        // Dev config may be missing guidance_embeds field; override from variant detection
        if matches!(variant, ModelVariant::Dev) {
            transformer_config.guidance_embeds = true;
        }

        eprintln!(
            "Generating {}x{} image (latent {}x{}) with {} steps [{:?}]",
            width, height, latent_w, latent_h, num_steps, variant
        );

        // ========== Phase 1: Text Encoding ==========
        eprintln!("Phase 1: Text encoding...");

        let (prompt_embeds, seq_len) = match variant {
            ModelVariant::Klein => Self::encode_text_klein::<B>(prompt, model_dir, weights, device),
            ModelVariant::Dev => Self::encode_text_dev::<B>(prompt, model_dir, weights, device),
        };

        // Prepare text position IDs (for full sequence)
        let txt_ids = utils::prepare_text_ids::<B>(1, seq_len, device);

        // ========== Phase 2: Prepare Latents ==========
        eprintln!("Phase 2: Preparing latents...");

        let latent_channels: usize = 128;
        let packed_h = latent_h / 2;
        let packed_w = latent_w / 2;
        let image_seq_len = packed_h * packed_w;

        // Generate random noise: [1, latent_channels, packed_h, packed_w]
        let latents = generate_noise::<B>(1, latent_channels, packed_h, packed_w, seed, device);

        // Pack: [1, C, H, W] -> [1, H*W, C]
        let mut latents = utils::pack_latents(latents);
        eprintln!(
            "  Latent shape: [1, {}, {}]",
            image_seq_len, latent_channels
        );

        // Prepare image position IDs
        let img_ids = utils::prepare_latent_ids::<B>(1, packed_h, packed_w, device);

        // ========== Phase 3: Denoising ==========
        eprintln!("Phase 3: Loading transformer and denoising...");

        let transformer_loader = WeightLoader::Pile(weights.transformer());

        let transformer =
            Flux2Transformer2DModel::<B>::load(&transformer_loader, transformer_config, device);

        if lora_path.is_some() {
            eprintln!("Warning: LoRA merging is not part of this milestone; --lora ignored.");
        }

        // Initialize scheduler
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(num_steps, image_seq_len);

        for i in 0..num_steps {
            let t = scheduler.timesteps[i];
            eprintln!("  Step {}/{} (t={:.3})", i + 1, num_steps, t);

            // Timestep: divide by 1000 for transformer (it multiplies back internally)
            let timestep =
                Tensor::<B, 1>::from_data(TensorData::new(vec![t / 1000.0], [1]), device);

            // Guidance conditioning (Dev only)
            let guidance = match variant {
                ModelVariant::Dev => Some(Tensor::<B, 1>::from_data(
                    TensorData::new(vec![guidance_scale], [1]),
                    device,
                )),
                ModelVariant::Klein => None,
            };

            // Transformer forward pass
            let txt_ids_2d: Tensor<B, 2> = txt_ids.clone().squeeze();
            let img_ids_2d: Tensor<B, 2> = img_ids.clone().squeeze();
            let noise_pred = transformer.forward(
                latents.clone(),
                prompt_embeds.clone(),
                timestep,
                guidance,
                img_ids_2d,
                txt_ids_2d,
                device,
            );
            // Scheduler step
            latents = scheduler.step(noise_pred, latents);
        }

        // Drop transformer
        drop(transformer);
        drop(transformer_loader);
        eprintln!("  Transformer freed from memory");

        // ========== Phase 4: VAE Decode ==========
        eprintln!("Phase 4: VAE decoding...");

        let vae_config_path = model_dir.join("vae").join("config.json");
        let vae_config = VaeConfig::load(&vae_config_path);
        let vae_loader = WeightLoader::Pile(weights.vae());

        let vae = AutoencoderKLFlux2::<B>::load(&vae_loader, vae_config, device);

        // Unpack latents: [1, H*W, C] -> [1, C, H, W]
        let latents_spatial = utils::unpack_latents_with_ids(latents, img_ids, device);

        // Denormalize using VAE BatchNorm statistics
        let bn_mean = vae
            .bn_running_mean
            .clone()
            .reshape([1, latent_channels, 1, 1]);
        let bn_std = (vae.bn_running_var.clone() + 1e-4)
            .sqrt()
            .reshape([1, latent_channels, 1, 1]);
        let latents_denorm = latents_spatial * bn_std + bn_mean;

        // Unpatchify: [1, 128, packed_h, packed_w] -> [1, 32, latent_h, latent_w]
        let latents_unpacked = utils::unpatchify_latents(latents_denorm);

        // VAE decode
        let image = vae.decode(latents_unpacked);

        eprintln!("Phase 5: Saving image...");
        utils::tensor_to_image(image)
    }

    /// Text encoding for Klein: Qwen2Tokenizer + Qwen3Model, extract layers [9, 18, 27].
    fn encode_text_klein<B: Backend>(
        prompt: &str,
        model_dir: &Path,
        weights: &FluxWeights,
        device: &B::Device,
    ) -> (Tensor<B, 3>, usize) {
        let tokenizer_path = model_dir.join("tokenizer").join("tokenizer.json");
        let tokenizer = Qwen2Tokenizer::from_file(&tokenizer_path);
        let (input_ids, attention_mask) = tokenizer.encode_prompt(prompt);
        let seq_len = input_ids.len();

        let te_config_path = model_dir.join("text_encoder").join("config.json");
        let te_config = Qwen3Config::load(&te_config_path);
        let te_loader = WeightLoader::Pile(weights.text_encoder());

        let text_encoder = Qwen3Model::<B>::load(&te_loader, te_config, device);

        let input_ids_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                [input_ids.len()],
            ),
            device,
        )
        .unsqueeze::<2>();

        let extract_layers = [9, 18, 27];
        let prompt_embeds = text_encoder.forward(
            input_ids_tensor,
            &extract_layers,
            Some(&attention_mask),
            device,
        );

        let [_b, _l, d] = prompt_embeds.dims();
        eprintln!("  Text embeddings: [1, {}, {}]", seq_len, d);

        drop(text_encoder);
        drop(te_loader);
        eprintln!("  Text encoder freed from memory");

        (prompt_embeds, seq_len)
    }

    /// Text encoding for Dev: MistralTokenizer + Mistral3Model, extract layers [10, 20, 30].
    fn encode_text_dev<B: Backend>(
        prompt: &str,
        model_dir: &Path,
        weights: &FluxWeights,
        device: &B::Device,
    ) -> (Tensor<B, 3>, usize) {
        let tokenizer_path = model_dir.join("tokenizer").join("tokenizer.json");
        let tokenizer = MistralTokenizer::from_file(&tokenizer_path);
        let (input_ids, attention_mask) = tokenizer.encode_prompt(prompt);
        let seq_len = input_ids.len();

        let te_config_path = model_dir.join("text_encoder").join("config.json");
        let te_config = Mistral3Config::load(&te_config_path);
        let te_loader = WeightLoader::Pile(weights.text_encoder());

        let extract_layers = [10, 20, 30];
        let text_encoder = Mistral3Model::<B>::load(&te_loader, te_config, &extract_layers, device);

        let input_ids_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                [input_ids.len()],
            ),
            device,
        )
        .unsqueeze::<2>();

        let prompt_embeds = text_encoder.forward(
            input_ids_tensor,
            &extract_layers,
            Some(&attention_mask),
            device,
        );

        let [_b, _l, d] = prompt_embeds.dims();
        eprintln!("  Text embeddings: [1, {}, {}]", seq_len, d);

        drop(text_encoder);
        drop(te_loader);
        eprintln!("  Text encoder freed from memory");

        (prompt_embeds, seq_len)
    }

    /// Generate an image using f16 precision.
    ///
    /// - Klein: f16 text encoder only, f32 transformer (fits in memory, better precision)
    /// - Dev: f16 text encoder + transformer (60GB transformer doesn't fit in f32)
    /// - VAE always runs in f32.
    pub fn generate_f16(
        prompt: &str,
        height: usize,
        width: usize,
        num_steps: usize,
        guidance_scale: f32,
        seed: u64,
        model_dir: &Path,
        weights: &FluxWeights,
        lora_path: Option<&Path>,
        device: &WgpuDevice,
    ) -> image::RgbImage {
        let vae_scale_factor: usize = 8;
        let latent_h = 2 * (height / (vae_scale_factor * 2));
        let latent_w = 2 * (width / (vae_scale_factor * 2));

        let variant = ModelVariant::detect(model_dir)
            .unwrap_or_else(|error| panic!("detect FLUX model variant: {error:#}"));
        assert_eq!(
            variant,
            weights.variant(),
            "FLUX config variant and selected native weights disagree"
        );
        let transformer_config_path = model_dir.join("transformer").join("config.json");
        let mut transformer_config = Flux2TransformerConfig::load(&transformer_config_path);
        if matches!(variant, ModelVariant::Dev) {
            transformer_config.guidance_embeds = true;
        }

        let streaming = matches!(variant, ModelVariant::Dev);
        eprintln!(
            "Generating {}x{} image (latent {}x{}) with {} steps [{:?}{}]",
            width,
            height,
            latent_w,
            latent_h,
            num_steps,
            variant,
            match variant {
                ModelVariant::Klein => "", // Klein: all f32 (fits in memory)
                ModelVariant::Dev => ", f16 text encoder, streaming f32 transformer",
            }
        );

        // ========== Phase 1: Text Encoding ==========
        // Klein: f32 always (fits in ~24GB, f16 introduces too much error for 4-step distilled model)
        // Dev: f16 text encoder (90GB f32 doesn't fit, and 28 steps tolerate f16 precision)
        let (prompt_embeds, seq_len) = match variant {
            ModelVariant::Klein => {
                eprintln!("Phase 1: Text encoding (f32)...");
                Self::encode_text_klein::<B>(prompt, model_dir, weights, device)
            }
            ModelVariant::Dev => {
                eprintln!("Phase 1: Text encoding (f16)...");
                let (embeds_half, seq_len) =
                    Self::encode_text_dev::<BHalf>(prompt, model_dir, weights, device);
                let embeds: Tensor<B, 3> = Tensor::from_data(embeds_half.into_data(), device);
                (embeds, seq_len)
            }
        };

        if streaming {
            // Dev: streaming f32 transformer (loads blocks one-at-a-time, ~7GB peak)
            Self::denoise_streaming_and_decode(
                prompt_embeds,
                seq_len,
                variant,
                transformer_config,
                latent_h,
                latent_w,
                num_steps,
                guidance_scale,
                seed,
                model_dir,
                weights,
                lora_path,
                device,
            )
        } else {
            // Klein: full f32 transformer (all blocks in memory, ~12GB)
            Self::denoise_and_decode_f32(
                prompt_embeds,
                seq_len,
                variant,
                transformer_config,
                latent_h,
                latent_w,
                num_steps,
                guidance_scale,
                seed,
                model_dir,
                weights,
                lora_path,
                device,
            )
        }
    }

    /// Denoise in f32 + VAE decode (used for Klein --f16 where only text encoder is f16).
    fn denoise_and_decode_f32(
        prompt_embeds: Tensor<B, 3>,
        seq_len: usize,
        variant: ModelVariant,
        transformer_config: Flux2TransformerConfig,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        seed: u64,
        model_dir: &Path,
        weights: &FluxWeights,
        lora_path: Option<&Path>,
        device: &WgpuDevice,
    ) -> image::RgbImage {
        let latent_channels: usize = 128;
        let packed_h = latent_h / 2;
        let packed_w = latent_w / 2;
        let image_seq_len = packed_h * packed_w;

        let txt_ids = utils::prepare_text_ids::<B>(1, seq_len, device);
        let img_ids = utils::prepare_latent_ids::<B>(1, packed_h, packed_w, device);

        eprintln!("Phase 2: Preparing latents...");
        let latents = generate_noise::<B>(1, latent_channels, packed_h, packed_w, seed, device);
        let mut latents = utils::pack_latents(latents);
        eprintln!(
            "  Latent shape: [1, {}, {}]",
            image_seq_len, latent_channels
        );

        eprintln!("Phase 3: Loading transformer and denoising (f32)...");
        let transformer_loader = WeightLoader::Pile(weights.transformer());
        let transformer =
            Flux2Transformer2DModel::<B>::load(&transformer_loader, transformer_config, device);

        if lora_path.is_some() {
            eprintln!("Warning: LoRA merging is not part of this milestone; --lora ignored.");
        }

        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(num_steps, image_seq_len);
        for i in 0..num_steps {
            let t = scheduler.timesteps[i];
            eprintln!("  Step {}/{} (t={:.3})", i + 1, num_steps, t);

            let timestep =
                Tensor::<B, 1>::from_data(TensorData::new(vec![t / 1000.0], [1]), device);
            let guidance = match variant {
                ModelVariant::Dev => Some(Tensor::<B, 1>::from_data(
                    TensorData::new(vec![guidance_scale], [1]),
                    device,
                )),
                ModelVariant::Klein => None,
            };

            let noise_pred = transformer.forward(
                latents.clone(),
                prompt_embeds.clone(),
                timestep,
                guidance,
                img_ids.clone().squeeze(),
                txt_ids.clone().squeeze(),
                device,
            );
            latents = scheduler.step(noise_pred, latents);
        }
        drop(transformer);
        drop(transformer_loader);
        eprintln!("  Transformer freed from memory");

        Self::vae_decode(
            latents,
            img_ids,
            latent_channels,
            model_dir,
            weights,
            device,
        )
    }

    /// Denoise with streaming f32 transformer (device blocks loaded one-at-a-time) + VAE decode.
    /// Used for Dev where the 60GB transformer doesn't fit in device memory all at once.
    /// The native snapshot loader currently still materializes the complete transformer on the
    /// host; streaming here bounds device residency, not total process memory.
    fn denoise_streaming_and_decode(
        prompt_embeds: Tensor<B, 3>,
        seq_len: usize,
        variant: ModelVariant,
        transformer_config: Flux2TransformerConfig,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        seed: u64,
        model_dir: &Path,
        weights: &FluxWeights,
        lora_path: Option<&Path>,
        device: &WgpuDevice,
    ) -> image::RgbImage {
        let latent_channels: usize = 128;
        let packed_h = latent_h / 2;
        let packed_w = latent_w / 2;
        let image_seq_len = packed_h * packed_w;

        let txt_ids = utils::prepare_text_ids::<B>(1, seq_len, device);
        let img_ids = utils::prepare_latent_ids::<B>(1, packed_h, packed_w, device);

        eprintln!("Phase 2: Preparing latents...");
        let latents = generate_noise::<B>(1, latent_channels, packed_h, packed_w, seed, device);
        let mut latents = utils::pack_latents(latents);
        eprintln!(
            "  Latent shape: [1, {}, {}]",
            image_seq_len, latent_channels
        );

        if lora_path.is_some() {
            eprintln!(
                "Warning: LoRA is not yet supported with streaming transformer (Dev). LoRA will be ignored."
            );
        }
        eprintln!("Phase 3: Loading transformer header and denoising (streaming f32)...");
        let transformer_loader = WeightLoader::Pile(weights.transformer());
        let transformer = Flux2Transformer2DModel::<B>::load_header_only(
            &transformer_loader,
            transformer_config,
            device,
        );
        eprintln!("  Header loaded (~3GB). Blocks will be streamed per step.");

        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(num_steps, image_seq_len);
        for i in 0..num_steps {
            let t = scheduler.timesteps[i];
            eprintln!("  Step {}/{} (t={:.3})", i + 1, num_steps, t);

            let timestep =
                Tensor::<B, 1>::from_data(TensorData::new(vec![t / 1000.0], [1]), device);
            let guidance = match variant {
                ModelVariant::Dev => Some(Tensor::<B, 1>::from_data(
                    TensorData::new(vec![guidance_scale], [1]),
                    device,
                )),
                ModelVariant::Klein => None,
            };

            let noise_pred = transformer.forward_streaming(
                &transformer_loader,
                latents.clone(),
                prompt_embeds.clone(),
                timestep,
                guidance,
                img_ids.clone().squeeze(),
                txt_ids.clone().squeeze(),
                device,
            );
            latents = scheduler.step(noise_pred, latents);
        }
        drop(transformer);
        drop(transformer_loader);
        eprintln!("  Transformer freed from memory");

        Self::vae_decode(
            latents,
            img_ids,
            latent_channels,
            model_dir,
            weights,
            device,
        )
    }

    /// VAE decode: unpack latents, denormalize, decode, convert to image. Always f32.
    fn vae_decode(
        latents: Tensor<B, 3>,
        img_ids: Tensor<B, 3>,
        latent_channels: usize,
        model_dir: &Path,
        weights: &FluxWeights,
        device: &WgpuDevice,
    ) -> image::RgbImage {
        eprintln!("Phase 4: VAE decoding...");

        let vae_config_path = model_dir.join("vae").join("config.json");
        let vae_config = VaeConfig::load(&vae_config_path);
        let vae_loader = WeightLoader::Pile(weights.vae());

        let vae = AutoencoderKLFlux2::<B>::load(&vae_loader, vae_config, device);

        let latents_spatial = utils::unpack_latents_with_ids(latents, img_ids, device);

        let bn_mean = vae
            .bn_running_mean
            .clone()
            .reshape([1, latent_channels, 1, 1]);
        let bn_std = (vae.bn_running_var.clone() + 1e-4)
            .sqrt()
            .reshape([1, latent_channels, 1, 1]);
        let latents_denorm = latents_spatial * bn_std + bn_mean;

        let latents_unpacked = utils::unpatchify_latents(latents_denorm);
        let image = vae.decode(latents_unpacked);

        eprintln!("Phase 5: Saving image...");
        utils::tensor_to_image(image)
    }
}

/// Generate random noise tensor using a simple seeded RNG.
fn generate_noise<B: Backend>(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    seed: u64,
    device: &B::Device,
) -> Tensor<B, 4> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let n = batch * channels * height * width;
    let mut data = Vec::with_capacity(n);

    // Box-Muller transform for normal distribution
    for _ in 0..((n + 1) / 2) {
        let u1: f64 = rng.r#gen::<f64>().max(1e-10);
        let u2: f64 = rng.r#gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push((r * theta.cos()) as f32);
        data.push((r * theta.sin()) as f32);
    }
    data.truncate(n);

    Tensor::from_data(
        TensorData::new(data, [batch, channels, height, width]),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{F32Array, U64Array, attrs};
    use ed25519_dalek::SigningKey;
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use triblespace::core::repo::pile::Pile;
    use triblespace::prelude::blobencodings::UTF8String;
    use triblespace::prelude::*;

    static NEXT_TEST_PILE: AtomicU64 = AtomicU64::new(0);

    struct TestPile(PathBuf);

    impl TestPile {
        fn new() -> Self {
            let ordinal = NEXT_TEST_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-native-flux-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create synthetic FLUX pile");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestPile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn component_fragment(
        variant: ModelVariant,
        component: &str,
        tensor: &str,
        value: f32,
    ) -> Fragment {
        let mut fragment = Fragment::empty();
        let data = fragment.put::<F32Array, _>(vec![value]);
        let shape = fragment.put::<U64Array, _>(vec![1_u64]);
        let leaf = entity! { _ @ attrs::data: data, attrs::shape: shape };
        let leaf_id = leaf.root().expect("tensor leaf root");
        fragment += leaf;

        let name = fragment.put::<UTF8String, _>(tensor.to_owned());
        let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: &leaf_id };
        let member_id = member.root().expect("model member root");
        fragment += member;

        let source = fragment.put::<UTF8String, _>(variant.component_source(component));
        fragment += entity! { _ @
            attrs::source: source,
            attrs::quantization: crate::persist::QUANTIZATION_NATIVE,
            attrs::member: &member_id,
        };
        fragment
    }

    fn publish(path: &Path, fragments: impl IntoIterator<Item = Fragment>) {
        let mut pile = Pile::open(path).expect("open synthetic FLUX pile");
        for fragment in fragments {
            crate::model_collection::publish_model_fragment(
                &mut pile,
                &SigningKey::from_bytes(&[0x46; 32]),
                fragment,
            )
            .expect("publish native FLUX component");
        }
        pile.close().expect("close synthetic FLUX pile");
    }

    #[test]
    fn one_snapshot_owns_three_explicit_flux_components_and_all_text_shards() {
        let file = TestPile::new();
        publish(
            file.path(),
            [
                component_fragment(ModelVariant::Klein, TEXT_ENCODER, "te.0.weight", 1.0),
                component_fragment(ModelVariant::Klein, TEXT_ENCODER, "te.1.weight", 1.5),
                component_fragment(ModelVariant::Klein, TRANSFORMER, "tr.weight", 2.0),
                component_fragment(ModelVariant::Klein, VAE, "vae.weight", 3.0),
            ],
        );

        let snapshot = crate::model_collection::load_model_collection_local_latest(file.path())
            .expect("freeze native FLUX snapshot");
        let weights = FluxWeights::from_snapshot(snapshot, ModelVariant::Klein)
            .expect("index all FLUX components");

        // A conflicting later root cannot change the already-owned snapshot.
        publish(
            file.path(),
            [component_fragment(
                ModelVariant::Klein,
                TEXT_ENCODER,
                "conflict.weight",
                9.0,
            )],
        );
        assert_eq!(weights.text_encoder()["te.0.weight"], (vec![1.0], vec![1]));
        assert_eq!(weights.text_encoder()["te.1.weight"], (vec![1.5], vec![1]));
        assert_eq!(weights.transformer()["tr.weight"], (vec![2.0], vec![1]));
        assert_eq!(weights.vae()["vae.weight"], (vec![3.0], vec![1]));

        // A later root claiming the text coordinate but shadowing one tensor
        // is not another shard. The widened snapshot fails deterministically.
        publish(
            file.path(),
            [component_fragment(
                ModelVariant::Klein,
                TEXT_ENCODER,
                "te.0.weight",
                9.0,
            )],
        );
        let widened = crate::model_collection::load_model_collection_local_latest(file.path())
            .expect("load widened native FLUX snapshot");
        let error = FluxWeights::from_snapshot(widened, ModelVariant::Klein)
            .err()
            .expect("a shadowed tensor must fail closed");
        assert!(
            error.to_string().contains("not shards of one component"),
            "unexpected shard diagnostic: {error:#}"
        );
    }
}
