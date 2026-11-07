//! CLIP ViT-B/32 as a multi-modal `LocalEmbedder` — image and text encoded into
//! one shared 512-d contrastive space, L2-normalized so cosine == dot product.
//!
//! This is the embedding seam a file-indexing service can use for image+text
//! similarity search: load a warm handle once (`load_clip_from_dir`), then call
//! `embed_image` / `embed_text` many times against `&self`.
//!
//! Port of `openai/clip-vit-base-patch32` (HF transformers `CLIPModel`):
//!   - VISION ViT-B/32: hidden 768, 12 layers, 12 heads, patch 32, image 224 →
//!     49 patches + 1 CLS = 50 tokens; learned class + position embeddings;
//!     PRE-LayerNorm blocks; quickgelu activation; `pre_layrnorm` before blocks,
//!     `post_layernorm` applied to the CLS token; then `visual_projection`
//!     (768→512, no bias).
//!   - TEXT transformer: hidden 512, 12 layers, 8 heads, ctx 77, vocab 49408 BPE;
//!     token + position embeddings; CAUSAL attention mask; quickgelu; final LN;
//!     take the hidden state at the EOT token (argmax of ids = 49407), then
//!     `text_projection` (512→512, no bias).
//!
//! Gotchas pinned during the parity pass (see `src/bin/clip_embed_test.rs`):
//!   - CLIP uses **quickgelu** (`x * sigmoid(1.702 * x)`), NOT tanh-gelu.
//!   - The vision pre-LN key is spelled `vision_model.pre_layrnorm` (HF typo).
//!   - Text pooling is at the EOT position = `argmax(token_ids)`, not the last
//!     non-pad token (here they coincide, but the argmax rule is canonical).
//!   - Image preprocessing must be PIL-bicubic resize (shortest side 224) +
//!     center-crop 224 + the exact CLIP mean/std normalization.

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::nn::backend::{B, WgpuDevice};
#[cfg(feature = "import")]
use crate::nn::weight_loader::SingleFileLoader;
use crate::nn::weight_loader::WeightLoader;

/// clip-vit-base-patch32 shared contrastive space dimension.
pub const CLIP_DIM: usize = 512;

/// siglip2-so400m-patch14-384 shared contrastive space dimension.
pub const SIGLIP_DIM: usize = 1152;

/// The exact tensor inventory consumed by one live embedding architecture.
///
/// These contracts are deliberately runtime-shaped: checkpoint tensors that
/// inference never reads (for example CLIP's scalar `logit_scale`) are not part
/// of the canonical native model. Importers and native constructors validate
/// against the same inventory before an infallible Burn loader sees it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingArchitecture {
    ClipVitBasePatch32,
    NomicTextV15,
    NomicVisionV15,
}

impl EmbeddingArchitecture {
    /// Required `tensor name -> exact shape` map for this fixed architecture.
    pub fn tensor_shapes(self) -> BTreeMap<String, Vec<usize>> {
        match self {
            Self::ClipVitBasePatch32 => clip_vit_b32_tensor_shapes(),
            Self::NomicTextV15 => nomic_text_v15_tensor_shapes(),
            Self::NomicVisionV15 => nomic_vision_v15_tensor_shapes(),
        }
    }

    /// Reject missing, unexpected, or dimensionally incompatible tensors.
    pub fn validate_tensor_shapes(self, actual: &BTreeMap<String, Vec<usize>>) -> Result<()> {
        let expected = self.tensor_shapes();
        for (name, shape) in &expected {
            let found = actual
                .get(name)
                .with_context(|| format!("{self:?} is missing required tensor {name:?}"))?;
            anyhow::ensure!(
                found == shape,
                "{self:?} tensor {name:?} has shape {found:?}, expected {shape:?}"
            );
        }
        if actual.len() != expected.len() {
            let unexpected = actual
                .keys()
                .find(|name| !expected.contains_key(*name))
                .expect("different exact-map lengths imply an unexpected key");
            anyhow::bail!("{self:?} contains unexpected tensor {unexpected:?}");
        }
        Ok(())
    }

    fn validate_keymap(self, keymap: &HashMap<String, (Vec<f32>, Vec<usize>)>) -> Result<()> {
        for (name, (values, shape)) in keymap {
            let elements = shape.iter().try_fold(1_usize, |product, &dimension| {
                product.checked_mul(dimension).with_context(|| {
                    format!("{self:?} tensor {name:?} shape element count overflows usize")
                })
            })?;
            anyhow::ensure!(
                values.len() == elements,
                "{self:?} tensor {name:?} has {} values but shape {shape:?} describes {elements}",
                values.len()
            );
        }
        let actual = keymap
            .iter()
            .map(|(name, (_, shape))| (name.clone(), shape.clone()))
            .collect();
        self.validate_tensor_shapes(&actual)
    }

    /// Validate every tokenizer id that the fixed embedding table may receive.
    pub fn validate_tokenizer(self, tokenizer: &tokenizers::Tokenizer) -> Result<()> {
        let (label, rows) = match self {
            Self::ClipVitBasePatch32 => ("CLIP", 49_408_u32),
            Self::NomicTextV15 => ("Nomic text", 30_528_u32),
            Self::NomicVisionV15 => anyhow::bail!("Nomic vision has no tokenizer"),
        };
        let max_id = tokenizer
            .get_vocab(true)
            .into_values()
            .max()
            .with_context(|| format!("{label} tokenizer vocabulary is empty"))?;
        anyhow::ensure!(
            max_id < rows,
            "{label} tokenizer can produce id {max_id}, outside its {rows}-row embedding table"
        );

        match self {
            Self::ClipVitBasePatch32 => {
                let bos = tokenizer
                    .token_to_id("<|startoftext|>")
                    .context("CLIP tokenizer has no start-of-text token")?;
                let eos = tokenizer
                    .token_to_id("<|endoftext|>")
                    .context("CLIP tokenizer has no end-of-text token")?;
                anyhow::ensure!(
                    bos == 49_406 && eos == 49_407,
                    "CLIP tokenizer sentinel ids are ({bos}, {eos}), expected (49406, 49407)"
                );
            }
            Self::NomicTextV15 => {
                let cls = tokenizer
                    .token_to_id("[CLS]")
                    .context("Nomic tokenizer has no [CLS] token")?;
                let sep = tokenizer
                    .token_to_id("[SEP]")
                    .context("Nomic tokenizer has no [SEP] token")?;
                anyhow::ensure!(cls != sep, "Nomic [CLS] and [SEP] ids collide");
            }
            Self::NomicVisionV15 => unreachable!("rejected before vocabulary validation"),
        }
        Ok(())
    }
}

fn tensor(
    tensors: &mut BTreeMap<String, Vec<usize>>,
    name: impl Into<String>,
    shape: impl Into<Vec<usize>>,
) {
    let name = name.into();
    assert!(
        tensors.insert(name.clone(), shape.into()).is_none(),
        "duplicate embedding tensor contract entry {name}"
    );
}

fn layer_norm(tensors: &mut BTreeMap<String, Vec<usize>>, prefix: &str, width: usize) {
    tensor(tensors, format!("{prefix}.weight"), [width]);
    tensor(tensors, format!("{prefix}.bias"), [width]);
}

fn linear(
    tensors: &mut BTreeMap<String, Vec<usize>>,
    prefix: &str,
    output: usize,
    input: usize,
    bias: bool,
) {
    tensor(tensors, format!("{prefix}.weight"), [output, input]);
    if bias {
        tensor(tensors, format!("{prefix}.bias"), [output]);
    }
}

fn clip_block(
    tensors: &mut BTreeMap<String, Vec<usize>>,
    prefix: &str,
    width: usize,
    inner: usize,
) {
    layer_norm(tensors, &format!("{prefix}.layer_norm1"), width);
    for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        linear(
            tensors,
            &format!("{prefix}.self_attn.{projection}"),
            width,
            width,
            true,
        );
    }
    layer_norm(tensors, &format!("{prefix}.layer_norm2"), width);
    linear(tensors, &format!("{prefix}.mlp.fc1"), inner, width, true);
    linear(tensors, &format!("{prefix}.mlp.fc2"), width, inner, true);
}

fn clip_vit_b32_tensor_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut tensors = BTreeMap::new();
    tensor(
        &mut tensors,
        "vision_model.embeddings.class_embedding",
        [768],
    );
    tensor(
        &mut tensors,
        "vision_model.embeddings.patch_embedding.weight",
        [768, 3, 32, 32],
    );
    tensor(
        &mut tensors,
        "vision_model.embeddings.position_embedding.weight",
        [50, 768],
    );
    layer_norm(&mut tensors, "vision_model.pre_layrnorm", 768);
    for layer in 0..12 {
        clip_block(
            &mut tensors,
            &format!("vision_model.encoder.layers.{layer}"),
            768,
            3072,
        );
    }
    layer_norm(&mut tensors, "vision_model.post_layernorm", 768);
    linear(&mut tensors, "visual_projection", 512, 768, false);

    tensor(
        &mut tensors,
        "text_model.embeddings.token_embedding.weight",
        [49_408, 512],
    );
    tensor(
        &mut tensors,
        "text_model.embeddings.position_embedding.weight",
        [77, 512],
    );
    for layer in 0..12 {
        clip_block(
            &mut tensors,
            &format!("text_model.encoder.layers.{layer}"),
            512,
            2048,
        );
    }
    layer_norm(&mut tensors, "text_model.final_layer_norm", 512);
    linear(&mut tensors, "text_projection", 512, 512, false);
    tensors
}

fn nomic_text_v15_tensor_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut tensors = BTreeMap::new();
    tensor(
        &mut tensors,
        "embeddings.word_embeddings.weight",
        [30_528, 768],
    );
    tensor(
        &mut tensors,
        "embeddings.token_type_embeddings.weight",
        [2, 768],
    );
    layer_norm(&mut tensors, "emb_ln", 768);
    for layer in 0..12 {
        let prefix = format!("encoder.layers.{layer}");
        linear(
            &mut tensors,
            &format!("{prefix}.attn.Wqkv"),
            2304,
            768,
            false,
        );
        linear(
            &mut tensors,
            &format!("{prefix}.attn.out_proj"),
            768,
            768,
            false,
        );
        layer_norm(&mut tensors, &format!("{prefix}.norm1"), 768);
        linear(
            &mut tensors,
            &format!("{prefix}.mlp.fc11"),
            3072,
            768,
            false,
        );
        linear(
            &mut tensors,
            &format!("{prefix}.mlp.fc12"),
            3072,
            768,
            false,
        );
        linear(&mut tensors, &format!("{prefix}.mlp.fc2"), 768, 3072, false);
        layer_norm(&mut tensors, &format!("{prefix}.norm2"), 768);
    }
    tensors
}

fn nomic_vision_mlp(tensors: &mut BTreeMap<String, Vec<usize>>, prefix: &str, inner_norm: bool) {
    linear(tensors, &format!("{prefix}.fc11"), 2048, 768, true);
    linear(tensors, &format!("{prefix}.fc12"), 2048, 768, true);
    if inner_norm {
        layer_norm(tensors, &format!("{prefix}.norm"), 2048);
    }
    linear(tensors, &format!("{prefix}.fc2"), 768, 2048, true);
}

fn nomic_vision_v15_tensor_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut tensors = BTreeMap::new();
    tensor(&mut tensors, "embeddings.cls_token", [1, 1, 768]);
    tensor(&mut tensors, "embeddings.pos_embed", [1, 197, 768]);
    linear(&mut tensors, "embeddings.proj", 768, 768, true);
    for layer in 0..12 {
        let prefix = format!("layers.{layer}");
        linear(
            &mut tensors,
            &format!("{prefix}.attn.Wqkv"),
            2304,
            768,
            true,
        );
        linear(
            &mut tensors,
            &format!("{prefix}.attn.out_proj"),
            768,
            768,
            true,
        );
        layer_norm(&mut tensors, &format!("{prefix}.norm1"), 768);
        nomic_vision_mlp(&mut tensors, &format!("{prefix}.mlp"), true);
        layer_norm(&mut tensors, &format!("{prefix}.norm2"), 768);
    }
    tensor(&mut tensors, "selector.attn.latent", [1, 1, 768]);
    linear(&mut tensors, "selector.attn.Wq", 768, 768, true);
    linear(&mut tensors, "selector.attn.Wkv", 1536, 768, true);
    linear(&mut tensors, "selector.attn.out_proj", 768, 768, true);
    layer_norm(&mut tensors, "selector.norm1", 768);
    nomic_vision_mlp(&mut tensors, "selector.mlp", false);
    tensors
}

#[cfg(test)]
mod embedding_contract_tests {
    use super::*;

    #[test]
    fn fixed_architecture_contracts_have_expected_inventories() {
        let clip = EmbeddingArchitecture::ClipVitBasePatch32.tensor_shapes();
        assert_eq!(clip.len(), 397);
        assert_eq!(
            clip["vision_model.embeddings.patch_embedding.weight"],
            [768, 3, 32, 32]
        );
        assert_eq!(clip["text_projection.weight"], [512, 512]);
        assert!(!clip.contains_key("logit_scale"));

        let text = EmbeddingArchitecture::NomicTextV15.tensor_shapes();
        assert_eq!(text.len(), 112);
        assert_eq!(text["encoder.layers.11.mlp.fc2.weight"], [768, 3072]);

        let vision = EmbeddingArchitecture::NomicVisionV15.tensor_shapes();
        assert_eq!(vision.len(), 211);
        assert_eq!(vision["layers.11.mlp.norm.weight"], [2048]);
        assert_eq!(vision["selector.attn.Wkv.weight"], [1536, 768]);
    }

    #[test]
    fn architecture_contract_rejects_missing_extra_and_wrong_shape() {
        let architecture = EmbeddingArchitecture::NomicTextV15;
        let complete = architecture.tensor_shapes();
        architecture.validate_tensor_shapes(&complete).unwrap();

        let mut missing = complete.clone();
        missing.remove("emb_ln.bias");
        assert!(
            architecture
                .validate_tensor_shapes(&missing)
                .unwrap_err()
                .to_string()
                .contains("missing required tensor")
        );

        let mut extra = complete.clone();
        extra.insert("unused.weight".to_owned(), vec![1]);
        assert!(
            architecture
                .validate_tensor_shapes(&extra)
                .unwrap_err()
                .to_string()
                .contains("unexpected tensor")
        );

        let mut wrong = complete;
        wrong.insert("emb_ln.bias".to_owned(), vec![767]);
        assert!(
            architecture
                .validate_tensor_shapes(&wrong)
                .unwrap_err()
                .to_string()
                .contains("expected [768]")
        );
    }

    #[test]
    fn tokenizer_contract_rejects_ids_beyond_the_embedding_table() {
        let vocab = [
            ("[UNK]".to_owned(), 0_u32),
            ("[CLS]".to_owned(), 1_u32),
            ("[SEP]".to_owned(), 2_u32),
            ("outside".to_owned(), 30_528_u32),
        ];
        let model = tokenizers::models::wordpiece::WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let tokenizer = tokenizers::Tokenizer::new(model);
        let error = EmbeddingArchitecture::NomicTextV15
            .validate_tokenizer(&tokenizer)
            .unwrap_err();
        assert!(error.to_string().contains("outside its 30528-row"));
    }

    #[test]
    fn native_keymap_contract_rejects_payload_shape_mismatches() {
        let keymap = HashMap::from([("emb_ln.bias".to_owned(), (vec![0.0; 767], vec![768]))]);
        let error = EmbeddingArchitecture::NomicTextV15
            .validate_keymap(&keymap)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("has 767 values but shape [768] describes 768"),
            "{error}"
        );
    }
}

/// A warm, content-addressed multi-modal embedder: image and text land in one
/// L2-normalized space (cosine == dot). Load once, embed many.
pub trait LocalEmbedder: Send {
    /// Decode PNG/JPEG bytes, preprocess (CLIP), encode, L2-normalize.
    fn embed_image(&self, bytes: &[u8]) -> Result<Vec<f32>>;
    /// Tokenize (CLIP BPE), encode, L2-normalize.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    /// Embedding dimensionality (`CLIP_DIM`).
    fn dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// `y = x @ wᵀ (+ b)` against a PyTorch `[out, in]` weight.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>, // [out, in]
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> Linear<B> {
    fn load(loader: &WeightLoader, prefix: &str, has_bias: bool, device: &B::Device) -> Self {
        Self {
            weight: loader.load_tensor(&format!("{prefix}.weight"), device),
            bias: has_bias.then(|| loader.load_tensor(&format!("{prefix}.bias"), device)),
        }
    }
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let out = x.matmul(self.weight.clone().transpose().unsqueeze());
        match &self.bias {
            Some(b) => out + b.clone().unsqueeze(),
            None => out,
        }
    }
}

/// Standard affine LayerNorm over the last dim.
struct LayerNorm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    fn load(loader: &WeightLoader, prefix: &str, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: loader.load_tensor(&format!("{prefix}.weight"), device),
            bias: loader.load_tensor(&format!("{prefix}.bias"), device),
            eps,
        }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let d = self.weight.dims()[0];
        let mean = x.clone().mean_dim(2);
        let xc = x.sub(mean);
        let var = xc.clone().powf_scalar(2.0).mean_dim(2);
        let norm = xc.div(var.add_scalar(self.eps).sqrt());
        norm.mul(self.weight.clone().reshape([1, 1, d]))
            .add(self.bias.clone().reshape([1, 1, d]))
    }
}

/// CLIP quickgelu: `x * sigmoid(1.702 * x)`. NOT the tanh approximation.
fn quickgelu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let s = burn::tensor::activation::sigmoid(x.clone().mul_scalar(1.702));
    x.mul(s)
}

/// One pre-LN CLIP transformer block. `causal_mask` is added to the attention
/// scores (`[1, s, s]`, 0 / -inf) for the text tower; `None` for vision.
struct Block<B: Backend> {
    ln1: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    ln2: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Block<B> {
    fn load(
        loader: &WeightLoader,
        p: &str,
        eps: f64,
        n_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let lin = |n: &str, bias: bool| Linear::load(loader, &format!("{p}.{n}"), bias, device);
        Self {
            ln1: LayerNorm::load(loader, &format!("{p}.layer_norm1"), eps, device),
            q: lin("self_attn.q_proj", true),
            k: lin("self_attn.k_proj", true),
            v: lin("self_attn.v_proj", true),
            out: lin("self_attn.out_proj", true),
            ln2: LayerNorm::load(loader, &format!("{p}.layer_norm2"), eps, device),
            fc1: lin("mlp.fc1", true),
            fc2: lin("mlp.fc2", true),
            n_heads,
            head_dim,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, causal_mask: Option<&Tensor<B, 3>>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);
        let hidden = self.ln1.forward(x.clone());
        let shape = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2); // [b,h,s,hd]
        let q = shape(self.q.forward(hidden.clone()));
        let k = shape(self.k.forward(hidden.clone()));
        let v = shape(self.v.forward(hidden));
        let mut scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5));
        if let Some(m) = causal_mask {
            // [1,s,s] -> broadcast over heads [b,h,s,s]
            scores = scores.add(m.clone().reshape([1, 1, s, s]));
        }
        let probs = softmax(scores, 3);
        let att = probs.matmul(v).swap_dims(1, 2).reshape([b, s, d]);
        let x = x.add(self.out.forward(att)); // residual 1
        let h2 = self.ln2.forward(x.clone());
        let mlp = self.fc2.forward(quickgelu(self.fc1.forward(h2)));
        x.add(mlp) // residual 2
    }
}

// ---------------------------------------------------------------------------
// Vision tower
// ---------------------------------------------------------------------------

struct VisionTower<B: Backend> {
    class_embedding: Tensor<B, 1>,    // [768]
    patch_weight: Tensor<B, 4>,       // [768,3,32,32]
    position_embedding: Tensor<B, 2>, // [50,768]
    pre_ln: LayerNorm<B>,
    layers: Vec<Block<B>>,
    post_ln: LayerNorm<B>,
    projection: Linear<B>, // visual_projection [512,768], no bias
    patch_size: usize,
}

impl<B: Backend> VisionTower<B> {
    fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let vp = "vision_model";
        let (n_layers, n_heads, head_dim, eps) = (12usize, 12usize, 64usize, 1e-5);
        let layers = (0..n_layers)
            .map(|i| {
                Block::load(
                    loader,
                    &format!("{vp}.encoder.layers.{i}"),
                    eps,
                    n_heads,
                    head_dim,
                    device,
                )
            })
            .collect();
        Self {
            class_embedding: loader
                .load_tensor(&format!("{vp}.embeddings.class_embedding"), device),
            patch_weight: loader
                .load_tensor(&format!("{vp}.embeddings.patch_embedding.weight"), device),
            position_embedding: loader.load_tensor(
                &format!("{vp}.embeddings.position_embedding.weight"),
                device,
            ),
            // NOTE: HF spells this `pre_layrnorm` (typo preserved in checkpoint).
            pre_ln: LayerNorm::load(loader, &format!("{vp}.pre_layrnorm"), eps, device),
            layers,
            post_ln: LayerNorm::load(loader, &format!("{vp}.post_layernorm"), eps, device),
            projection: Linear::load(loader, "visual_projection", false, device),
            patch_size: 32,
        }
    }

    /// `image` is `[1,3,224,224]` already normalized. Returns `[1,512]` (pre-norm).
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 2> {
        let [b, _, _, _] = image.dims();
        // patch conv embed: [b,768,7,7]
        let opts = ConvOptions::new([self.patch_size, self.patch_size], [0, 0], [1, 1], 1);
        let patch = conv2d(image, self.patch_weight.clone(), None, opts);
        let [_, c, gh, gw] = patch.dims();
        let np = gh * gw; // 49
        let patches = patch.reshape([b, c, np]).swap_dims(1, 2); // [b,49,768]

        // prepend CLS token
        let cls = self
            .class_embedding
            .clone()
            .reshape([1, 1, c])
            .repeat_dim(0, b);
        let x = Tensor::cat(vec![cls, patches], 1); // [b,50,768]

        // add learned position embedding, pre-LN, blocks, then post-LN on CLS
        let x = x.add(self.position_embedding.clone().reshape([1, np + 1, c]));
        let mut x = self.pre_ln.forward(x);
        for layer in &self.layers {
            x = layer.forward(x, None);
        }
        // pooled = post_layernorm(CLS token)
        let cls_tok = x.narrow(1, 0, 1); // [b,1,768]
        let pooled = self.post_ln.forward(cls_tok).reshape([b, c]); // [b,768]
        self.projection.forward(pooled) // [b,512]
    }
}

// ---------------------------------------------------------------------------
// Text tower
// ---------------------------------------------------------------------------

struct TextTower<B: Backend> {
    token_embedding: Tensor<B, 2>,    // [49408,512]
    position_embedding: Tensor<B, 2>, // [77,512]
    layers: Vec<Block<B>>,
    final_ln: LayerNorm<B>,
    projection: Linear<B>, // text_projection [512,512], no bias
}

impl<B: Backend> TextTower<B> {
    fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let tp = "text_model";
        let (n_layers, n_heads, head_dim, eps) = (12usize, 8usize, 64usize, 1e-5);
        let layers = (0..n_layers)
            .map(|i| {
                Block::load(
                    loader,
                    &format!("{tp}.encoder.layers.{i}"),
                    eps,
                    n_heads,
                    head_dim,
                    device,
                )
            })
            .collect();
        Self {
            token_embedding: loader
                .load_tensor(&format!("{tp}.embeddings.token_embedding.weight"), device),
            position_embedding: loader.load_tensor(
                &format!("{tp}.embeddings.position_embedding.weight"),
                device,
            ),
            layers,
            final_ln: LayerNorm::load(loader, &format!("{tp}.final_layer_norm"), eps, device),
            projection: Linear::load(loader, "text_projection", false, device),
        }
    }

    /// `ids` are the token ids (length s ≤ 77). Returns `[1,512]` (pre-norm).
    /// Pools the hidden state at the EOT position (argmax of ids).
    fn forward(&self, ids: &[u32], device: &B::Device) -> Tensor<B, 2> {
        let s = ids.len();
        let d = self.position_embedding.dims()[1];

        // token embedding via index_select on the embedding table
        let idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), [s]),
            device,
        );
        let tok = self.token_embedding.clone().select(0, idx); // [s,512]
        let pos = self.position_embedding.clone().narrow(0, 0, s); // [s,512]
        let x = tok.add(pos).reshape([1, s, d]);

        // causal mask: [1,s,s], 0 on/below diagonal, -inf above
        let mask = causal_mask::<B>(s, device);
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, Some(&mask));
        }
        let x = self.final_ln.forward(x); // [1,s,512]

        // pool at EOT = argmax(ids)
        let eot = ids
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(s - 1);
        let pooled = x.narrow(1, eot, 1).reshape([1, d]); // [1,512]
        self.projection.forward(pooled)
    }
}

/// Lower-triangular additive causal mask `[1,s,s]` (0 allowed, large-negative
/// blocked).
fn causal_mask<B: Backend>(s: usize, device: &B::Device) -> Tensor<B, 3> {
    let mut m = vec![0f32; s * s];
    for i in 0..s {
        for j in (i + 1)..s {
            m[i * s + j] = f32::MIN;
        }
    }
    Tensor::<B, 1>::from_data(TensorData::new(m, [s * s]), device).reshape([1, s, s])
}

// ---------------------------------------------------------------------------
// The embedder
// ---------------------------------------------------------------------------

/// Warm CLIP handle: both towers + a CLIP BPE tokenizer + the wgpu device.
pub struct ClipEmbedder<B: Backend> {
    vision: VisionTower<B>,
    text: TextTower<B>,
    tokenizer: tokenizers::Tokenizer,
    bos: u32,
    eos: u32,
    device: B::Device,
}

impl<B: Backend> ClipEmbedder<B> {
    fn encode_text_ids(&self, text: &str) -> Vec<u32> {
        // CLIP's tokenizer.json already lowercases, applies BPE, and (via its
        // post-processor) wraps with <|startoftext|>/<|endoftext|>. We add the
        // sentinels ourselves to be robust to tokenizer.json variants, then
        // truncate to the 77-token context (last slot stays EOT).
        let enc = self
            .tokenizer
            .encode(text, false)
            .expect("clip tokenizer encode");
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        // Strip any sentinels the tokenizer already added so we control framing.
        ids.retain(|&i| i != self.bos && i != self.eos);
        let max_body = 77 - 2;
        if ids.len() > max_body {
            ids.truncate(max_body);
        }
        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(self.bos);
        out.extend(ids);
        out.push(self.eos);
        out
    }
}

impl<B: Backend> LocalEmbedder for ClipEmbedder<B>
where
    B::Device: Send,
{
    fn embed_image(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let pixels = preprocess_image_bytes(bytes)?; // [3*224*224] normalized
        let image =
            Tensor::<B, 4>::from_data(TensorData::new(pixels, [1, 3, 224, 224]), &self.device);
        let feat = self.vision.forward(image); // [1,512]
        Ok(l2_normalize(feat))
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.encode_text_ids(text);
        let feat = self.text.forward(&ids, &self.device); // [1,512]
        Ok(l2_normalize(feat))
    }

    fn dim(&self) -> usize {
        CLIP_DIM
    }
}

/// L2-normalize a `[1,512]` feature into a length-512 `Vec<f32>`.
fn l2_normalize<B: Backend>(feat: Tensor<B, 2>) -> Vec<f32> {
    let v = feat.into_data().to_vec::<f32>().unwrap();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

/// A default wgpu/Metal device — CLIP-B is small, no raised buffer limit needed.
/// Convenience so callers don't depend on the backend device type directly.
pub fn default_device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Core: build a `ClipEmbedder` from explicit `model.safetensors` + `tokenizer.json`
/// paths. The ViT-B/32 config is fixed in the towers, so config.json is not read.
#[cfg(feature = "import")]
pub fn load_clip_from_files(
    weights: &Path,
    tokenizer: &Path,
    device: WgpuDevice,
) -> Result<ClipEmbedder<B>> {
    if !weights.exists() {
        anyhow::bail!("missing clip weights: {}", weights.display());
    }
    let tok = tokenizers::Tokenizer::from_file(tokenizer)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer.display()))?;
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(weights));
    Ok(ClipEmbedder {
        vision: VisionTower::load(&loader, &device),
        text: TextTower::load(&loader, &device),
        tokenizer: tok,
        bos: 49406, // <|startoftext|>
        eos: 49407, // <|endoftext|>
        device,
    })
}

/// Load a `ClipEmbedder` from an HF snapshot dir holding `model.safetensors`
/// and `tokenizer.json`.
#[cfg(feature = "import")]
pub fn load_clip_from_dir(dir: &Path, device: WgpuDevice) -> Result<ClipEmbedder<B>> {
    load_clip_from_files(
        &dir.join("model.safetensors"),
        &dir.join("tokenizer.json"),
        device,
    )
}

/// Resolve the exact snapshot named by a cached Hugging Face `refs/main`.
///
/// Returning the directory, rather than resolving files independently, makes
/// it impossible for a caller to combine weights from one revision with a
/// tokenizer from another. This is a pure cache lookup; it never downloads.
pub fn hf_cache_main_snapshot(model_id: &str) -> Result<std::path::PathBuf> {
    let hub_cache = std::env::var_os("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HF_HOME")
                .map(std::path::PathBuf::from)
                .map(|home| home.join("hub"))
        })
        .or_else(|| {
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .map(|home| home.join(".cache/huggingface/hub"))
        })
        .context("none of HF_HUB_CACHE, HF_HOME, or HOME is set")?;
    hf_cache_main_snapshot_at(&hub_cache, model_id)
}

fn hf_cache_main_snapshot_at(hub_cache: &Path, model_id: &str) -> Result<std::path::PathBuf> {
    anyhow::ensure!(
        !model_id.is_empty()
            && !model_id.contains('\\')
            && model_id
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != ".."),
        "invalid Hugging Face model id {model_id:?}"
    );
    let repo = hub_cache.join(format!("models--{}", model_id.replace('/', "--")));
    let reference = repo.join("refs/main");
    let revision = std::fs::read_to_string(&reference)
        .with_context(|| format!("read cached Hugging Face ref {}", reference.display()))?;
    let revision = revision.trim();
    anyhow::ensure!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "cached Hugging Face ref {} contains an invalid revision {revision:?}",
        reference.display()
    );

    let snapshots = repo.join("snapshots");
    let snapshots = snapshots.canonicalize().with_context(|| {
        format!(
            "canonicalize Hugging Face snapshot directory {}",
            snapshots.display()
        )
    })?;
    let snapshot = snapshots
        .join(revision)
        .canonicalize()
        .with_context(|| format!("resolve Hugging Face main revision {revision} for {model_id}"))?;
    anyhow::ensure!(
        snapshot.parent() == Some(snapshots.as_path()) && snapshot.is_dir(),
        "Hugging Face main revision for {model_id} escapes its snapshot directory"
    );
    Ok(snapshot)
}

#[cfg(test)]
mod hf_cache_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    const REVISION_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const REVISION_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("mary-hf-cache-{}-{sequence}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn repo(temp: &TempDir) -> std::path::PathBuf {
        temp.0.join("models--org--model")
    }

    #[test]
    fn main_ref_selects_one_snapshot_without_cross_revision_fallback() {
        let temp = TempDir::new();
        let repo = repo(&temp);
        let first = repo.join("snapshots").join(REVISION_A);
        let second = repo.join("snapshots").join(REVISION_B);
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(repo.join("refs/main"), format!("{REVISION_A}\n")).unwrap();
        std::fs::write(first.join("model.safetensors"), b"weights-a").unwrap();
        std::fs::write(second.join("tokenizer.json"), b"tokenizer-b").unwrap();

        let selected = hf_cache_main_snapshot_at(&temp.0, "org/model").unwrap();
        assert_eq!(selected, first.canonicalize().unwrap());
        assert!(selected.join("model.safetensors").is_file());
        assert!(!selected.join("tokenizer.json").exists());

        std::fs::write(repo.join("refs/main"), REVISION_B).unwrap();
        assert_eq!(selected, first.canonicalize().unwrap());
        assert_eq!(
            hf_cache_main_snapshot_at(&temp.0, "org/model").unwrap(),
            second.canonicalize().unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_invalid_refs_and_snapshot_escape_but_allows_blob_symlinks() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new();
        let repo = repo(&temp);
        let snapshots = repo.join("snapshots");
        let valid = snapshots.join(REVISION_A);
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::create_dir_all(&valid).unwrap();
        std::fs::create_dir_all(repo.join("blobs")).unwrap();
        std::fs::write(repo.join("blobs/tokenizer"), b"{}").unwrap();
        symlink("../../blobs/tokenizer", valid.join("tokenizer.json")).unwrap();
        std::fs::write(repo.join("refs/main"), REVISION_A).unwrap();
        let selected = hf_cache_main_snapshot_at(&temp.0, "org/model").unwrap();
        assert!(selected.join("tokenizer.json").is_file());

        std::fs::write(repo.join("refs/main"), "../../outside").unwrap();
        assert!(hf_cache_main_snapshot_at(&temp.0, "org/model").is_err());
        assert!(hf_cache_main_snapshot_at(&temp.0, "org\\model").is_err());

        let outside = temp.0.join("outside");
        std::fs::create_dir(&outside).unwrap();
        let escaping_revision = "cccccccccccccccccccccccccccccccccccccccc";
        symlink(&outside, snapshots.join(escaping_revision)).unwrap();
        std::fs::write(repo.join("refs/main"), escaping_revision).unwrap();
        assert!(hf_cache_main_snapshot_at(&temp.0, "org/model").is_err());
    }
}

/// Load a `ClipEmbedder` by HF model id (e.g. `"openai/clip-vit-base-patch32"`),
/// resolving the official `pytorch_model.bin` + `tokenizer.json` pair from one
/// cached `main` revision. The model must already be in the cache (fetch once
/// with `huggingface-cli download <id>` or `hf_hub_download`).
#[cfg(feature = "import")]
pub fn load_clip_from_hf(model_id: &str, device: WgpuDevice) -> Result<ClipEmbedder<B>> {
    let snapshot = hf_cache_main_snapshot(model_id)?;
    let weights = snapshot.join("pytorch_model.bin");
    anyhow::ensure!(
        weights.is_file(),
        "cached main revision for {model_id} has no pytorch_model.bin"
    );
    let tokenizer_path = snapshot.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        anyhow::anyhow!(
            "load tokenizer {} from cached main revision: {error}",
            tokenizer_path.display()
        )
    })?;

    let contract = EmbeddingArchitecture::ClipVitBasePatch32.tensor_shapes();
    let mut keymap = HashMap::with_capacity(contract.len());
    for (name, values, shape) in
        crate::formats::extract_tensors(crate::formats::WeightFormat::Pickle, &weights)?
    {
        if contract.contains_key(&name) {
            anyhow::ensure!(
                keymap.insert(name.clone(), (values, shape)).is_none(),
                "duplicate CLIP tensor {name:?} in {}",
                weights.display()
            );
        }
    }
    clip_from_parts(keymap, tokenizer, device)
}

/// Assemble a `ClipEmbedder` from parts: a content-addressed weight keymap and
/// an already-built tokenizer. Both parts can therefore come from one frozen
/// model-collection snapshot without a tokenizer side-file.
pub fn clip_from_parts(
    keymap: HashMap<String, (Vec<f32>, Vec<usize>)>,
    tokenizer: tokenizers::Tokenizer,
    device: WgpuDevice,
) -> Result<ClipEmbedder<B>> {
    let architecture = EmbeddingArchitecture::ClipVitBasePatch32;
    architecture.validate_tokenizer(&tokenizer)?;
    architecture.validate_keymap(&keymap)?;
    let bos = tokenizer
        .token_to_id("<|startoftext|>")
        .expect("validated CLIP start-of-text token");
    let eos = tokenizer
        .token_to_id("<|endoftext|>")
        .expect("validated CLIP end-of-text token");
    let loader = WeightLoader::Pile(keymap);
    Ok(ClipEmbedder {
        vision: VisionTower::load(&loader, &device),
        text: TextTower::load(&loader, &device),
        tokenizer,
        bos,
        eos,
        device,
    })
}

/// Build a `ClipEmbedder` from a content-addressed weight keymap plus a
/// `tokenizer.json` path. This is the file-tokenizer variant of
/// [`clip_from_parts`]; native collection callers should select both parts
/// from one frozen snapshot instead.
pub fn load_clip_from_keymap(
    keymap: HashMap<String, (Vec<f32>, Vec<usize>)>,
    tokenizer: &Path,
    device: WgpuDevice,
) -> Result<ClipEmbedder<B>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer.display()))?;
    clip_from_parts(keymap, tok, device)
}

#[cfg(test)]
mod clip_parts_tests {
    use super::*;

    #[test]
    fn native_parts_reject_wrong_sentinel_ids_before_loading_weights() {
        let vocab = [
            ("[UNK]".to_owned(), 0_u32),
            ("<|startoftext|>".to_owned(), 1_u32),
            ("<|endoftext|>".to_owned(), 2_u32),
        ];
        let model = tokenizers::models::wordpiece::WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let tokenizer = tokenizers::Tokenizer::new(model);
        let error = match clip_from_parts(HashMap::new(), tokenizer, WgpuDevice::default()) {
            Ok(_) => panic!("wrong CLIP sentinel ids must fail before weight loading"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("sentinel ids"), "{error}");
    }
}

// ===========================================================================
// SigLIP2 so400m — second LocalEmbedder (image+text → one 1152-d space)
// ===========================================================================
//
// Port of `google/siglip2-so400m-patch14-384` (HF `SiglipModel`):
//   - VISION (no CLS token): conv patch-embed (patch 14, image 384 → 27×27=729
//     patches) + learned position table; 27 PRE-LN encoder layers; post-LN; then
//     a **MultiheadAttentionPoolingHead** — a learned probe query cross-attends
//     over the 729 patch tokens (torch.nn.MultiheadAttention, packed in_proj),
//     followed by a residual LN+MLP block. The pooled vector IS the 1152-d image
//     embedding (no extra projection).
//   - TEXT (full, non-causal attention): token + position embeddings; 27 PRE-LN
//     layers; final LN; pool the **LAST position** (index seq_len-1, here 63 —
//     a pad slot, but SigLIP pools `[:, -1, :]` unconditionally with full
//     bidirectional attention) → `text_model.head` Linear (1152→1152) → embedding.
//
// Gotchas pinned during the parity pass (see `src/bin/siglip_embed_test.rs`):
//   - Activation is **gelu_pytorch_tanh** in BOTH towers (NOT quickgelu).
//   - Vision pooling head's attention is a packed nn.MultiheadAttention:
//     `in_proj_weight` is [3*1152,1152] = q,k,v stacked → split into thirds.
//     Q comes from the probe; K,V from the encoder tokens.
//   - Text pooling is the LAST token (last sequence position), not argmax/EOT.
//   - Tokenizer is GemmaTokenizer (sentencepiece): lowercase the text, the
//     tokenizer.json appends `<eos>`=1 and pads to 64 with `<pad>`=0 itself.
//     No BOS. So encode(lowercase(text)) yields the exact 64-length id vector.
//   - Image preprocessing: resize to 384×384 (PIL bicubic, no center crop),
//     normalize (x/255 - 0.5)/0.5 = x/127.5 - 1 (mean=std=0.5).
//   - layer_norm eps is 1e-6 (SiglipConfig default).

/// `y = x @ wᵀ (+ b)` against a PyTorch `[out, in]` weight. Distinct from the
/// CLIP `Linear` only in that it's reused by the SigLIP towers verbatim.
type SLinear<B> = Linear<B>;

/// gelu_pytorch_tanh: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
fn gelu_tanh<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let x3 = x.clone().powf_scalar(3.0);
    let inner = x
        .clone()
        .add(x3.mul_scalar(0.044715))
        .mul_scalar(0.7978845608028654);
    x.mul(inner.tanh().add_scalar(1.0)).mul_scalar(0.5)
}

/// One pre-LN SigLIP encoder block (full bidirectional attention, gelu-tanh MLP).
/// Same topology as the CLIP `Block` but with gelu_pytorch_tanh and no mask.
struct SiglipBlock<B: Backend> {
    ln1: LayerNorm<B>,
    q: SLinear<B>,
    k: SLinear<B>,
    v: SLinear<B>,
    out: SLinear<B>,
    ln2: LayerNorm<B>,
    fc1: SLinear<B>,
    fc2: SLinear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> SiglipBlock<B> {
    fn load(
        loader: &WeightLoader,
        p: &str,
        eps: f64,
        n_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let lin = |n: &str| Linear::load(loader, &format!("{p}.{n}"), true, device);
        Self {
            ln1: LayerNorm::load(loader, &format!("{p}.layer_norm1"), eps, device),
            q: lin("self_attn.q_proj"),
            k: lin("self_attn.k_proj"),
            v: lin("self_attn.v_proj"),
            out: lin("self_attn.out_proj"),
            ln2: LayerNorm::load(loader, &format!("{p}.layer_norm2"), eps, device),
            fc1: lin("mlp.fc1"),
            fc2: lin("mlp.fc2"),
            n_heads,
            head_dim,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);
        let hidden = self.ln1.forward(x.clone());
        let shape = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2); // [b,h,s,hd]
        let q = shape(self.q.forward(hidden.clone()));
        let k = shape(self.k.forward(hidden.clone()));
        let v = shape(self.v.forward(hidden));
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5));
        let probs = softmax(scores, 3);
        let att = probs.matmul(v).swap_dims(1, 2).reshape([b, s, d]);
        let x = x.add(self.out.forward(att)); // residual 1
        let h2 = self.ln2.forward(x.clone());
        let mlp = self.fc2.forward(gelu_tanh(self.fc1.forward(h2)));
        x.add(mlp) // residual 2
    }
}

// ---------------------------------------------------------------------------
// Vision tower + MultiheadAttentionPoolingHead
// ---------------------------------------------------------------------------

/// `MultiheadAttentionPoolingHead`: a learned probe query cross-attends over the
/// encoder token sequence (torch.nn.MultiheadAttention, packed in_proj split into
/// q/k/v), then a residual LN+MLP. Returns the pooled `[b, 1152]`.
struct AttnPoolHead<B: Backend> {
    probe: Tensor<B, 3>, // [1,1,1152]
    // packed in_proj split into per-projection [out, in] weights + biases
    q_w: Tensor<B, 2>,
    q_b: Tensor<B, 1>,
    k_w: Tensor<B, 2>,
    k_b: Tensor<B, 1>,
    v_w: Tensor<B, 2>,
    v_b: Tensor<B, 1>,
    out: SLinear<B>, // attention.out_proj
    ln: LayerNorm<B>,
    fc1: SLinear<B>,
    fc2: SLinear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> AttnPoolHead<B> {
    fn load(
        loader: &WeightLoader,
        eps: f64,
        n_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let hp = "vision_model.head";
        let dim = n_heads * head_dim;
        // packed in_proj_weight [3*dim, dim], in_proj_bias [3*dim] → split q,k,v.
        let in_w: Tensor<B, 2> =
            loader.load_tensor(&format!("{hp}.attention.in_proj_weight"), device);
        let in_b: Tensor<B, 1> =
            loader.load_tensor(&format!("{hp}.attention.in_proj_bias"), device);
        let q_w = in_w.clone().narrow(0, 0, dim);
        let k_w = in_w.clone().narrow(0, dim, dim);
        let v_w = in_w.narrow(0, 2 * dim, dim);
        let q_b = in_b.clone().narrow(0, 0, dim);
        let k_b = in_b.clone().narrow(0, dim, dim);
        let v_b = in_b.narrow(0, 2 * dim, dim);
        Self {
            probe: loader.load_tensor(&format!("{hp}.probe"), device),
            q_w,
            q_b,
            k_w,
            k_b,
            v_w,
            v_b,
            out: Linear::load(loader, &format!("{hp}.attention.out_proj"), true, device),
            ln: LayerNorm::load(loader, &format!("{hp}.layernorm"), eps, device),
            fc1: Linear::load(loader, &format!("{hp}.mlp.fc1"), true, device),
            fc2: Linear::load(loader, &format!("{hp}.mlp.fc2"), true, device),
            n_heads,
            head_dim,
        }
    }

    /// `x` is the encoder output `[b, n, dim]`. Returns pooled `[b, dim]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [b, n, dim] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);
        // y = wᵀ on [out,in] weight: x @ wᵀ + b.
        let proj = |t: Tensor<B, 3>, w: &Tensor<B, 2>, bias: &Tensor<B, 1>| {
            t.matmul(w.clone().transpose().unsqueeze())
                .add(bias.clone().unsqueeze())
        };
        // probe → query (length 1); keys/values from encoder tokens.
        let probe = self.probe.clone().repeat_dim(0, b); // [b,1,dim]
        let q = proj(probe, &self.q_w, &self.q_b)
            .reshape([b, 1, h, hd])
            .swap_dims(1, 2); // [b,h,1,hd]
        let k = proj(x.clone(), &self.k_w, &self.k_b)
            .reshape([b, n, h, hd])
            .swap_dims(1, 2); // [b,h,n,hd]
        let v = proj(x, &self.v_w, &self.v_b)
            .reshape([b, n, h, hd])
            .swap_dims(1, 2);
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5)); // [b,h,1,n]
        let probs = softmax(scores, 3);
        let att = probs.matmul(v).swap_dims(1, 2).reshape([b, 1, dim]); // [b,1,dim]
        let hidden = self.out.forward(att); // out_proj
        // residual LN + MLP block
        let residual = hidden.clone();
        let h2 = self.ln.forward(hidden);
        let mlp = self.fc2.forward(gelu_tanh(self.fc1.forward(h2)));
        let pooled = residual.add(mlp); // [b,1,dim]
        pooled.narrow(1, 0, 1).reshape([b, dim])
    }
}

struct SiglipVisionTower<B: Backend> {
    patch_weight: Tensor<B, 4>, // [1152,3,14,14]
    patch_bias: Tensor<B, 1>,
    position_embedding: Tensor<B, 2>, // [729,1152]
    layers: Vec<SiglipBlock<B>>,
    post_ln: LayerNorm<B>,
    head: AttnPoolHead<B>,
    patch_size: usize,
}

impl<B: Backend> SiglipVisionTower<B> {
    fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let vp = "vision_model";
        let (n_layers, n_heads, head_dim, eps) = (27usize, 16usize, 72usize, 1e-6);
        let layers = (0..n_layers)
            .map(|i| {
                SiglipBlock::load(
                    loader,
                    &format!("{vp}.encoder.layers.{i}"),
                    eps,
                    n_heads,
                    head_dim,
                    device,
                )
            })
            .collect();
        Self {
            patch_weight: loader
                .load_tensor(&format!("{vp}.embeddings.patch_embedding.weight"), device),
            patch_bias: loader
                .load_tensor(&format!("{vp}.embeddings.patch_embedding.bias"), device),
            position_embedding: loader.load_tensor(
                &format!("{vp}.embeddings.position_embedding.weight"),
                device,
            ),
            layers,
            post_ln: LayerNorm::load(loader, &format!("{vp}.post_layernorm"), eps, device),
            head: AttnPoolHead::load(loader, eps, n_heads, head_dim, device),
            patch_size: 14,
        }
    }

    /// `image` is `[1,3,384,384]` normalized. Returns the pooled `[1,1152]`.
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 2> {
        let [b, _, _, _] = image.dims();
        let opts = ConvOptions::new([self.patch_size, self.patch_size], [0, 0], [1, 1], 1);
        let patch = conv2d(
            image,
            self.patch_weight.clone(),
            Some(self.patch_bias.clone()),
            opts,
        ); // [b,1152,27,27]
        let [_, c, gh, gw] = patch.dims();
        let np = gh * gw; // 729
        let patches = patch.reshape([b, c, np]).swap_dims(1, 2); // [b,729,1152]
        // NO CLS token: just add learned position embedding.
        let mut x = patches.add(self.position_embedding.clone().reshape([1, np, c]));
        for layer in &self.layers {
            x = layer.forward(x);
        }
        let x = self.post_ln.forward(x);
        self.head.forward(x) // [b,1152]
    }
}

// ---------------------------------------------------------------------------
// Text tower
// ---------------------------------------------------------------------------

struct SiglipTextTower<B: Backend> {
    token_embedding: Tensor<B, 2>,    // [256000,1152]
    position_embedding: Tensor<B, 2>, // [64,1152]
    layers: Vec<SiglipBlock<B>>,
    final_ln: LayerNorm<B>,
    head: SLinear<B>, // text_model.head Linear [1152,1152] + bias
}

impl<B: Backend> SiglipTextTower<B> {
    fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let tp = "text_model";
        let (n_layers, n_heads, head_dim, eps) = (27usize, 16usize, 72usize, 1e-6);
        let layers = (0..n_layers)
            .map(|i| {
                SiglipBlock::load(
                    loader,
                    &format!("{tp}.encoder.layers.{i}"),
                    eps,
                    n_heads,
                    head_dim,
                    device,
                )
            })
            .collect();
        Self {
            token_embedding: loader
                .load_tensor(&format!("{tp}.embeddings.token_embedding.weight"), device),
            position_embedding: loader.load_tensor(
                &format!("{tp}.embeddings.position_embedding.weight"),
                device,
            ),
            layers,
            final_ln: LayerNorm::load(loader, &format!("{tp}.final_layer_norm"), eps, device),
            head: Linear::load(loader, &format!("{tp}.head"), true, device),
        }
    }

    /// `ids` is the full padded sequence (length 64). Full bidirectional
    /// attention (pads attend too — SigLIP uses no attention mask here). Pools
    /// the LAST position, then the head Linear. Returns `[1,1152]`.
    fn forward(&self, ids: &[u32], device: &B::Device) -> Tensor<B, 2> {
        let s = ids.len();
        let d = self.position_embedding.dims()[1];
        let idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), [s]),
            device,
        );
        let tok = self.token_embedding.clone().select(0, idx); // [s,1152]
        let pos = self.position_embedding.clone().narrow(0, 0, s); // [s,1152]
        let mut x = tok.add(pos).reshape([1, s, d]);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        let x = self.final_ln.forward(x); // [1,s,1152]
        let pooled = x.narrow(1, s - 1, 1).reshape([1, d]); // LAST position
        self.head.forward(pooled)
    }
}

// ---------------------------------------------------------------------------
// The embedder
// ---------------------------------------------------------------------------

/// Warm SigLIP2 handle: both towers + the Gemma sentencepiece tokenizer + device.
pub struct SiglipEmbedder<B: Backend> {
    vision: SiglipVisionTower<B>,
    text: SiglipTextTower<B>,
    tokenizer: tokenizers::Tokenizer,
}

impl<B: Backend> SiglipEmbedder<B> {
    /// Lowercase, encode (tokenizer.json appends `<eos>` and pads to 64), and
    /// truncate to 64 as a safety net. Matches HF SiglipProcessor
    /// (padding="max_length", max_length=64, lowercase).
    fn encode_text_ids(&self, text: &str) -> Vec<u32> {
        // `true` = apply the post-processor, which appends `<eos>`=1; the
        // tokenizer.json's own padding then fills to 64 with `<pad>`=0. Matches
        // HF SiglipProcessor (lowercase + padding="max_length", max_length=64).
        let enc = self
            .tokenizer
            .encode(text.to_lowercase(), true)
            .expect("siglip tokenizer encode");
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        ids.truncate(64);
        ids
    }
}

impl<B: Backend> LocalEmbedder for SiglipEmbedder<B>
where
    B::Device: Send,
{
    fn embed_image(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let pixels = preprocess_image_siglip(bytes)?; // [3*384*384] normalized
        let device = self.vision.patch_weight.device();
        let image = Tensor::<B, 4>::from_data(TensorData::new(pixels, [1, 3, 384, 384]), &device);
        let feat = self.vision.forward(image); // [1,1152]
        Ok(l2_normalize(feat))
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.encode_text_ids(text);
        let device = self.text.token_embedding.device();
        let feat = self.text.forward(&ids, &device); // [1,1152]
        Ok(l2_normalize(feat))
    }

    fn dim(&self) -> usize {
        SIGLIP_DIM
    }
}

/// Core: build a `SiglipEmbedder` from explicit `model.safetensors` +
/// `tokenizer.json` paths. The so400m config is fixed in the towers.
#[cfg(feature = "import")]
pub fn load_siglip_from_files(
    weights: &Path,
    tokenizer_json: &Path,
    device: WgpuDevice,
) -> Result<SiglipEmbedder<B>> {
    if !weights.exists() {
        anyhow::bail!("missing siglip weights: {}", weights.display());
    }
    let tok = tokenizers::Tokenizer::from_file(tokenizer_json)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_json.display()))?;
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(weights));
    Ok(SiglipEmbedder {
        vision: SiglipVisionTower::load(&loader, &device),
        text: SiglipTextTower::load(&loader, &device),
        tokenizer: tok,
    })
}

/// Load a `SiglipEmbedder` from an HF snapshot dir holding `model.safetensors`
/// and `tokenizer.json`.
#[cfg(feature = "import")]
pub fn load_siglip_from_dir(dir: &Path, device: WgpuDevice) -> Result<SiglipEmbedder<B>> {
    load_siglip_from_files(
        &dir.join("model.safetensors"),
        &dir.join("tokenizer.json"),
        device,
    )
}

/// Load a `SiglipEmbedder` by HF model id
/// (e.g. `"google/siglip2-so400m-patch14-384"`), resolving `model.safetensors` +
/// `tokenizer.json` from one cached `main` revision. Must already be cached
/// (`huggingface-cli download <id>`).
#[cfg(feature = "import")]
pub fn load_siglip_from_hf(model_id: &str, device: WgpuDevice) -> Result<SiglipEmbedder<B>> {
    let snapshot = hf_cache_main_snapshot(model_id)?;
    let weights = snapshot.join("model.safetensors");
    let tokenizer = snapshot.join("tokenizer.json");
    load_siglip_from_files(&weights, &tokenizer, device)
}

/// Build a `SiglipEmbedder` from a content-addressed weight keymap. Same model
/// build as [`load_siglip_from_files`] but loading the towers from a
/// [`WeightLoader::Pile`]. `tokenizer.json` stays a small file.
pub fn load_siglip_from_keymap(
    keymap: HashMap<String, (Vec<f32>, Vec<usize>)>,
    tokenizer: &Path,
    device: WgpuDevice,
) -> Result<SiglipEmbedder<B>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer.display()))?;
    let loader = WeightLoader::Pile(keymap);
    Ok(SiglipEmbedder {
        vision: SiglipVisionTower::load(&loader, &device),
        text: SiglipTextTower::load(&loader, &device),
        tokenizer: tok,
    })
}

/// Decode PNG/JPEG, resize to 384×384 (PIL bicubic, no crop), normalize
/// (x/127.5 − 1, mean=std=0.5). Returns CHW `[3*384*384]`. SiglipImageProcessor.
fn preprocess_image_siglip(bytes: &[u8]) -> Result<Vec<f32>> {
    const TARGET: u32 = 384;
    let img = image::load_from_memory(bytes)?.to_rgb8();
    // SigLIP resizes directly to a square (no aspect-preserving + crop).
    // preprocessor_config.json: resample=2 = PIL BILINEAR (NOT bicubic).
    let resized = pil_resize(&img, TARGET, TARGET, BILINEAR);
    let s = TARGET as usize;
    let mut out = vec![0f32; 3 * s * s];
    for y in 0..TARGET {
        for x in 0..TARGET {
            let px = resized.get_pixel(x, y);
            for c in 0..3 {
                // (v/255 - 0.5)/0.5 = v/127.5 - 1
                out[c * s * s + (y as usize) * s + (x as usize)] = (px[c] as f32) / 127.5 - 1.0;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Image preprocessing (CLIPImageProcessor parity)
// ---------------------------------------------------------------------------

/// Decode PNG/JPEG, resize shortest side to 224 (PIL bicubic), center-crop 224,
/// scale to [0,1], normalize by CLIP mean/std. Returns CHW `[3*224*224]`.
fn preprocess_image_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
    const TARGET: u32 = 224;

    let img = image::load_from_memory(bytes)?.to_rgb8();
    let (w, h) = img.dimensions();

    // resize shortest side to 224, preserving aspect
    let (nw, nh) = if w <= h {
        (
            TARGET,
            ((h as f64) * (TARGET as f64) / (w as f64)).round() as u32,
        )
    } else {
        (
            ((w as f64) * (TARGET as f64) / (h as f64)).round() as u32,
            TARGET,
        )
    };
    let resized = pil_resize(&img, nw, nh, BICUBIC);

    // center-crop 224x224
    let cx = (nw - TARGET) / 2;
    let cy = (nh - TARGET) / 2;

    let mut out = vec![0f32; 3 * (TARGET as usize) * (TARGET as usize)];
    let s = TARGET as usize;
    for y in 0..TARGET {
        for x in 0..TARGET {
            let px = resized.get_pixel(cx + x, cy + y);
            for c in 0..3 {
                let v = (px[c] as f32) / 255.0;
                let v = (v - MEAN[c]) / STD[c];
                out[c * s * s + (y as usize) * s + (x as usize)] = v;
            }
        }
    }
    Ok(out)
}

// ---- PIL-compatible bicubic resize (self-contained; matches PIL BICUBIC) ----
// Mirrors CPython's PIL.Image.resize(size, Resampling.BICUBIC): Keys cubic
// kernel a=-0.5, antialiased filter scaling on downsample, separable H then V,
// u8 round+clamp between passes. (Same algorithm as mary::models::gemma's
// preprocess, copied here so the `embed` feature stays light.)

fn bicubic_kernel(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// PIL BILINEAR triangle kernel (support 1.0). PIL resample code = 2.
fn bilinear_kernel(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

/// A separable resampling filter: kernel fn + its support radius.
#[derive(Clone, Copy)]
struct Filter {
    kernel: fn(f64) -> f64,
    support: f64,
}

const BICUBIC: Filter = Filter {
    kernel: bicubic_kernel,
    support: 2.0,
};
const BILINEAR: Filter = Filter {
    kernel: bilinear_kernel,
    support: 1.0,
};

/// Precompute (bounds, weights) per destination pixel for a 1-D resample.
fn precompute_coeffs(
    src: usize,
    dst: usize,
    filter: Filter,
) -> (Vec<(usize, usize)>, Vec<Vec<f64>>) {
    let scale = src as f64 / dst as f64;
    let filterscale = scale.max(1.0);
    let support_s = filter.support * filterscale;

    let mut bounds = Vec::with_capacity(dst);
    let mut weights = Vec::with_capacity(dst);
    for xx in 0..dst {
        let center = (xx as f64 + 0.5) * scale;
        let ww = center - support_s + 0.5;
        let ee = center + support_s + 0.5;
        let mut xmin = ww.floor() as isize;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = ee.floor() as isize;
        if xmax > src as isize {
            xmax = src as isize;
        }
        let xmin = xmin as usize;
        let xmax = xmax as usize;
        let mut w = Vec::with_capacity(xmax - xmin);
        let mut total = 0.0;
        for x in xmin..xmax {
            let k = (filter.kernel)((x as f64 + 0.5 - center) / filterscale);
            w.push(k);
            total += k;
        }
        if total != 0.0 {
            for k in &mut w {
                *k /= total;
            }
        }
        bounds.push((xmin, xmax - xmin));
        weights.push(w);
    }
    (bounds, weights)
}

fn resample_horizontal_u8(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    filter: Filter,
) -> Vec<u8> {
    let (bounds, weights) = precompute_coeffs(src_w, dst_w, filter);
    let mut out = vec![0u8; dst_w * src_h * 3];
    for y in 0..src_h {
        for xx in 0..dst_w {
            let (xmin, n) = bounds[xx];
            let w = &weights[xx];
            for c in 0..3 {
                let mut acc = 0.0f64;
                for i in 0..n {
                    acc += w[i] * src[((y * src_w) + (xmin + i)) * 3 + c] as f64;
                }
                out[((y * dst_w) + xx) * 3 + c] = (acc + 0.5).floor().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn resample_vertical_u8(
    src: &[u8],
    width: usize,
    src_h: usize,
    dst_h: usize,
    filter: Filter,
) -> Vec<u8> {
    let (bounds, weights) = precompute_coeffs(src_h, dst_h, filter);
    let mut out = vec![0u8; width * dst_h * 3];
    for yy in 0..dst_h {
        let (ymin, n) = bounds[yy];
        let w = &weights[yy];
        for x in 0..width {
            for c in 0..3 {
                let mut acc = 0.0f64;
                for i in 0..n {
                    acc += w[i] * src[(((ymin + i) * width) + x) * 3 + c] as f64;
                }
                out[((yy * width) + x) * 3 + c] = (acc + 0.5).floor().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn pil_resize(img: &image::RgbImage, dst_w: u32, dst_h: u32, filter: Filter) -> image::RgbImage {
    let src_w = img.width() as usize;
    let src_h = img.height() as usize;
    let dst_w_u = dst_w as usize;
    let dst_h_u = dst_h as usize;
    let src_bytes: &[u8] = img.as_raw();
    let h_out = resample_horizontal_u8(src_bytes, src_w, src_h, dst_w_u, filter);
    let v_out = resample_vertical_u8(&h_out, dst_w_u, src_h, dst_h_u, filter);
    let mut out = image::RgbImage::new(dst_w, dst_h);
    for y in 0..dst_h_u {
        for x in 0..dst_w_u {
            let off = (y * dst_w_u + x) * 3;
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([v_out[off], v_out[off + 1], v_out[off + 2]]),
            );
        }
    }
    out
}

// ===========================================================================
// nomic-embed-text-v1.5 — long-context TEXT LocalEmbedder (768-d)
// ===========================================================================
//
// Port of `nomic-ai/nomic-embed-text-v1.5` (HF `NomicBertModel`, arch
// `nomic_bert`). A BERT-shaped bidirectional encoder, but POST-NORM, with RoPE
// instead of learned position embeddings and a SwiGLU MLP:
//   - Embeddings: word_embeddings[ids] + token_type_embeddings[row 0] (all
//     token_type_ids are 0), then `emb_ln` LayerNorm (eps 1e-12). No learned
//     position embeddings — RoPE carries position (rotary_emb_fraction 1.0).
//   - 12 POST-norm encoder layers. Per layer the residual lands BEFORE the norm:
//       h  = norm1(attn(x) + x)
//       h2 = norm2(mlp(h) + h)
//     (config `prenorm: false`; verified against modeling_hf_nomic_bert.py's
//     non-prenorm branch.) Attention is full bidirectional (config `causal:
//     false`) over the padding mask; scale 1/sqrt(head_dim).
//   - Packed `attn.Wqkv` [3*768, 768] (NO bias, `qkv_proj_bias: false`) → split
//     into q,k,v thirds. RoPE applied to Q and K over the FULL head_dim
//     (rotate_half / non-interleaved), base = 1000 (config `rotary_emb_base`).
//     `attn.out_proj` [768,768] (no bias).
//   - SwiGLU MLP (NO biases, `mlp_fc{1,2}_bias: false`): fc2(fc11(x) *
//     silu(fc12(x))). fc11 is the value branch, fc12 the gate.
//   - Pooling: masked MEAN over token hidden states (sum h*mask / sum mask),
//     then L2-normalize. Use the FULL 768 dims (Matryoshka not truncated).
//   - Task prefix (REQUIRED by v1.5): "search_document: " for stored content,
//     "search_query: " for queries, prepended before tokenizing.
//   - Tokenizer: BERT WordPiece (lowercase), tokenizer.json's TemplateProcessing
//     wraps [CLS]=101 … [SEP]=102 itself.

/// nomic-embed-text-v1.5 embedding dimension (full Matryoshka width).
pub const NOMIC_TEXT_DIM: usize = 768;

/// How every linear's input is fake-quantised before its multiply, to measure
/// what an fp4 activation path costs this model. Read once per model load from
/// `NOMIC_ACT_QUANT`: unset or `f32` is the plain path; `nvfp4` is one E2M1
/// lane under NVFP4 block scales; `lanes16` is the sixteen-lane fractional
/// carrier from the Inkling pilot, sixteen fp4 lanes whose mean lands within a
/// sixteenth of the E2M1 step. The residual stream stays f32 either way; only
/// the operand handed to each matmul is rounded, which is what an fp4 kernel
/// would see. All of it is tensor ops on the device.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ActQuant {
    None,
    Nvfp4,
    Lanes16,
}

impl ActQuant {
    fn from_env() -> Self {
        match std::env::var("NOMIC_ACT_QUANT").as_deref() {
            Ok("nvfp4") => ActQuant::Nvfp4,
            Ok("lanes16") => ActQuant::Lanes16,
            _ => ActQuant::None,
        }
    }
}

/// Per-linear input statistics gathered while a model runs with capture on:
/// the mean absolute value and mean square of every input channel over all
/// rows seen, and a reservoir of whole rows for a calibration search that
/// needs the real inputs (AWQ's scale search, GPTQ's Hessian).
#[derive(Clone, Debug, Default)]
pub struct NomicActStats {
    pub rows_seen: usize,
    pub sum_abs: Vec<f64>,
    pub sum_sq: Vec<f64>,
    pub rows: Vec<Vec<f32>>,
}

impl NomicActStats {
    pub fn mean_abs(&self) -> Vec<f32> {
        let n = self.rows_seen.max(1) as f64;
        self.sum_abs.iter().map(|&v| (v / n) as f32).collect()
    }
    pub fn mean_sq(&self) -> Vec<f32> {
        let n = self.rows_seen.max(1) as f64;
        self.sum_sq.iter().map(|&v| (v / n) as f32).collect()
    }
}

struct ActCapture {
    rows_per_call: usize,
    lcg: u64,
    stats: HashMap<String, NomicActStats>,
}

static ACT_CAPTURE: std::sync::Mutex<Option<ActCapture>> = std::sync::Mutex::new(None);

/// Start recording every nomic linear's input (see [`NomicActStats`]); each
/// forward call contributes its channel sums and up to `rows_per_call`
/// randomly chosen token rows to the reservoir.
pub fn nomic_activation_capture_start(rows_per_call: usize) {
    *ACT_CAPTURE.lock().expect("capture lock") = Some(ActCapture {
        rows_per_call,
        lcg: 0x9E37_79B9_7F4A_7C15,
        stats: HashMap::new(),
    });
}

/// Stop recording and hand back what was gathered, keyed by the linear's
/// weight prefix (`encoder.layers.3.attn.Wqkv`; the MLP's two input linears
/// share one entry under `mlp.fc1` because they read the same input).
pub fn nomic_activation_capture_take() -> HashMap<String, NomicActStats> {
    ACT_CAPTURE
        .lock()
        .expect("capture lock")
        .take()
        .map(|c| c.stats)
        .unwrap_or_default()
}

fn capture_linear_input<B: Backend>(x: &Tensor<B, 3>, name: &str) {
    let mut guard = ACT_CAPTURE.lock().expect("capture lock");
    let Some(capture) = guard.as_mut() else {
        return;
    };
    let [b, s, k] = x.dims();
    let flat = x.clone().reshape([b * s, k]);
    let sum_abs = flat.clone().abs().sum_dim(0).into_data().to_vec::<f32>().expect("sum_abs");
    let sum_sq = flat
        .clone()
        .powf_scalar(2.0)
        .sum_dim(0)
        .into_data()
        .to_vec::<f32>()
        .expect("sum_sq");
    let rows = b * s;
    let take = capture.rows_per_call.min(rows);
    let mut picks = Vec::with_capacity(take);
    for _ in 0..take {
        capture.lcg = capture.lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        picks.push(((capture.lcg >> 33) as usize % rows) as i64);
    }
    let picked = if take > 0 {
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(picks, [take]), &x.device());
        flat.select(0, idx).into_data().to_vec::<f32>().expect("rows")
    } else {
        Vec::new()
    };
    let entry = capture.stats.entry(name.to_owned()).or_insert_with(|| NomicActStats {
        rows_seen: 0,
        sum_abs: vec![0.0; k],
        sum_sq: vec![0.0; k],
        rows: Vec::new(),
    });
    entry.rows_seen += rows;
    for (acc, v) in entry.sum_abs.iter_mut().zip(&sum_abs) {
        *acc += *v as f64;
    }
    for (acc, v) in entry.sum_sq.iter_mut().zip(&sum_sq) {
        *acc += *v as f64;
    }
    for row in picked.chunks(k) {
        entry.rows.push(row.to_vec());
    }
}

/// The operand handed to one linear: captured if a capture is running, then
/// fake-quantised according to `mode`.
fn linear_input<B: Backend>(x: Tensor<B, 3>, mode: ActQuant, name: &str) -> Tensor<B, 3> {
    capture_linear_input(&x, name);
    fake_quant_act(x, mode)
}

/// NVFP4 fake quantisation of `[b, s, k]` activations: one f32 scale per
/// token row (row max maps to 6 x 448), one E4M3 scale per block of sixteen
/// along `k`, E2M1 codes {0, 0.5, 1, 1.5, 2, 3, 4, 6} by nearest on the
/// non-uniform grid, and for `Lanes16` the lane mean `lo + (hi - lo) * k / 16`
/// with `k = round(16 * frac)`.
fn fake_quant_act<B: Backend>(x: Tensor<B, 3>, mode: ActQuant) -> Tensor<B, 3> {
    if mode == ActQuant::None {
        return x;
    }
    let [b, s, k] = x.dims();
    assert!(k % 16 == 0, "activation width {k} is not a multiple of the NVFP4 block");
    let blocks = k / 16;
    let x4 = x.reshape([b, s, blocks, 16]);
    let ax = x4.clone().abs();
    let sign = x4.sign();
    // Per-row scale; a zero row would divide by zero, and its values are all
    // zero anyway, so give it any positive scale.
    let row_max = ax.clone().max_dim(3).max_dim(2); // [b,s,1,1]
    let ts = row_max.div_scalar(6.0 * 448.0);
    let ts = ts.clone().mask_where(ts.clone().equal_elem(0.0), ts.ones_like());
    // Block scale in E4M3: three mantissa bits, exponent floor at -6, cap 448.
    let ln2 = std::f64::consts::LN_2;
    let s_raw = ax.clone().max_dim(3).div(ts.clone()).div_scalar(6.0); // [b,s,blocks,1]
    let e = s_raw.clone().clamp_min(1e-30).log().div_scalar(ln2).floor().clamp_min(-6.0);
    let step = e.sub_scalar(3.0).mul_scalar(ln2).exp();
    let scale = s_raw.div(step.clone()).round().mul(step).clamp_max(448.0);
    let unit = scale.mul(ts); // [b,s,blocks,1]
    let unit = unit.clone().mask_where(unit.clone().equal_elem(0.0), unit.ones_like());
    let m = ax.div(unit.clone()).clamp(0.0, 6.0); // [b,s,blocks,16]
    // E2M1 step: 0.5 below 2, 1 from 2 to 4, 2 from 4 to 6.
    let st = m.zeros_like().add_scalar(0.5);
    let st = st.mask_where(m.clone().greater_equal_elem(2.0), m.ones_like());
    let st = st.mask_where(m.clone().greater_equal_elem(4.0), m.ones_like().mul_scalar(2.0));
    let q = match mode {
        ActQuant::Nvfp4 => m.div(st.clone()).round().mul(st),
        ActQuant::Lanes16 => {
            let lo = m.clone().div(st.clone()).floor().mul(st.clone());
            let frac = m.sub(lo.clone()).div(st.clone());
            let lanes = frac.mul_scalar(16.0).round().div_scalar(16.0);
            lo.add(lanes.mul(st)).clamp_max(6.0)
        }
        ActQuant::None => unreachable!(),
    };
    q.mul(unit).mul(sign).reshape([b, s, k])
}

/// One POST-norm Nomic encoder layer: bidirectional MHA with RoPE on Q/K, then
/// a SwiGLU MLP; the norm is applied AFTER the residual add (BERT post-norm).
struct NomicLayer<B: Backend> {
    wqkv: Linear<B>, // packed [3*768, 768], no bias
    out_proj: Linear<B>,
    norm1: LayerNorm<B>,
    fc11: Linear<B>, // value, no bias
    fc12: Linear<B>, // gate, no bias
    fc2: Linear<B>,  // no bias
    norm2: LayerNorm<B>,
    n_heads: usize,
    head_dim: usize,
    act: ActQuant,
    prefix: String,
}

impl<B: Backend> NomicLayer<B> {
    fn load(
        loader: &WeightLoader,
        p: &str,
        eps: f64,
        n_heads: usize,
        head_dim: usize,
        act: ActQuant,
        device: &B::Device,
    ) -> Self {
        let lin = |n: &str| Linear::load(loader, &format!("{p}.{n}"), false, device);
        Self {
            wqkv: lin("attn.Wqkv"),
            out_proj: lin("attn.out_proj"),
            norm1: LayerNorm::load(loader, &format!("{p}.norm1"), eps, device),
            fc11: lin("mlp.fc11"),
            fc12: lin("mlp.fc12"),
            fc2: lin("mlp.fc2"),
            norm2: LayerNorm::load(loader, &format!("{p}.norm2"), eps, device),
            n_heads,
            head_dim,
            act,
            prefix: p.to_owned(),
        }
    }

    /// `x` is `[b,s,d]`; `cos`/`sin` are the RoPE tables `[s, head_dim/2]`;
    /// `add_mask` is the additive attention mask `[b,1,1,s]` (0 keep, -inf pad).
    fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 2>,
        sin: &Tensor<B, 2>,
        add_mask: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);
        // packed qkv → [b,s,3d] then split into q,k,v each [b,s,d].
        let qkv = self
            .wqkv
            .forward(linear_input(x.clone(), self.act, &format!("{}.attn.Wqkv", self.prefix)));
        let q = qkv.clone().narrow(2, 0, d);
        let k = qkv.clone().narrow(2, d, d);
        let v = qkv.narrow(2, 2 * d, d);
        let shape = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2); // [b,h,s,hd]
        let q = apply_rope(shape(q), cos, sin);
        let k = apply_rope(shape(k), cos, sin);
        let v = shape(v);
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5)); // [b,h,s,s]
        let scores = scores.add(add_mask.clone()); // broadcast [b,1,1,s]
        let probs = softmax(scores, 3);
        let att = probs.matmul(v).swap_dims(1, 2).reshape([b, s, d]);
        // POST-norm: norm AFTER the residual add.
        let att = linear_input(att, self.act, &format!("{}.attn.out_proj", self.prefix));
        let h1 = self.norm1.forward(self.out_proj.forward(att).add(x));
        // SwiGLU: fc2(fc11(x) * silu(fc12(x))). fc11 and fc12 read the same
        // input, recorded once under `mlp.fc1`.
        let h1q = linear_input(h1.clone(), self.act, &format!("{}.mlp.fc1", self.prefix));
        let gated = self
            .fc11
            .forward(h1q.clone())
            .mul(silu(self.fc12.forward(h1q)));
        let mlp = self
            .fc2
            .forward(linear_input(gated, self.act, &format!("{}.mlp.fc2", self.prefix)));
        self.norm2.forward(mlp.add(h1))
    }
}

/// Apply RoPE (non-interleaved rotate_half) to `[b,h,s,hd]` using cos/sin tables
/// `[s, hd/2]`. `rotate_half(x) = [-x2, x1]`; `out = x*cos_full + rotate_half*sin_full`
/// where cos/sin are duplicated per pair → matches the gemma RopeTable convention.
fn apply_rope<B: Backend>(x: Tensor<B, 4>, cos: &Tensor<B, 2>, sin: &Tensor<B, 2>) -> Tensor<B, 4> {
    let [b, h, s, hd] = x.dims();
    let half = hd / 2;
    let cos = cos.clone().reshape([1, 1, s, half]);
    let sin = sin.clone().reshape([1, 1, s, half]);
    let cos = cos.expand([b, h, s, half]);
    let sin = sin.expand([b, h, s, half]);
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    let out1 = x1.clone().mul(cos.clone()).sub(x2.clone().mul(sin.clone()));
    let out2 = x1.mul(sin).add(x2.mul(cos));
    Tensor::cat(vec![out1, out2], 3)
}

/// Build the RoPE cos/sin tables `[max_len, head_dim/2]` for base `theta`.
fn nomic_rope_tables<B: Backend>(
    head_dim: usize,
    max_len: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let positions: Vec<f32> = (0..max_len).map(|p| p as f32).collect();
    let pos = Tensor::<B, 1>::from_floats(&positions[..], device).reshape([max_len, 1]);
    let invf = Tensor::<B, 1>::from_floats(&inv_freq[..], device).reshape([1, half]);
    let freqs = pos.matmul(invf); // [max_len, half]
    (freqs.clone().cos(), freqs.sin())
}

struct NomicTextModel<B: Backend> {
    word_embeddings: Tensor<B, 2>,       // [30528, 768]
    token_type_embeddings: Tensor<B, 2>, // [2, 768] — only row 0 used
    emb_ln: LayerNorm<B>,
    layers: Vec<NomicLayer<B>>,
    head_dim: usize,
    rope_theta: f64,
}

impl<B: Backend> NomicTextModel<B> {
    fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let (n_layers, n_heads, head_dim, eps, theta) =
            (12usize, 12usize, 64usize, 1e-12, 1000.0f64);
        let act = ActQuant::from_env();
        if act != ActQuant::None {
            eprintln!("nomic text: activations fake-quantised at every linear input: {act:?}");
        }
        let layers = (0..n_layers)
            .map(|i| {
                NomicLayer::load(
                    loader,
                    &format!("encoder.layers.{i}"),
                    eps,
                    n_heads,
                    head_dim,
                    act,
                    device,
                )
            })
            .collect();
        Self {
            word_embeddings: loader.load_tensor("embeddings.word_embeddings.weight", device),
            token_type_embeddings: loader
                .load_tensor("embeddings.token_type_embeddings.weight", device),
            emb_ln: LayerNorm::load(loader, "emb_ln", eps, device),
            layers,
            head_dim,
            rope_theta: theta,
        }
    }

    /// `ids` are the token ids (incl [CLS]/[SEP]); all are real tokens (single
    /// sequence, no padding) so the mean-pool mask is all-ones. Returns the
    /// L2-normalized `[768]` embedding.
    fn embed(&self, ids: &[u32], device: &B::Device) -> Vec<f32> {
        let s = ids.len();
        let d = self.word_embeddings.dims()[1];
        let idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), [s]),
            device,
        );
        let tok = self.word_embeddings.clone().select(0, idx); // [s,768]
        let tok_type = self
            .token_type_embeddings
            .clone()
            .narrow(0, 0, 1)
            .reshape([1, d]); // row 0
        let x = tok.add(tok_type).reshape([1, s, d]);
        let x = self.emb_ln.forward(x);

        let (cos, sin) = nomic_rope_tables::<B>(self.head_dim, s, self.rope_theta, device);
        // No padding (single sequence) → additive mask is all-zeros.
        let add_mask = Tensor::<B, 4>::zeros([1, 1, 1, s], device);
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, &cos, &sin, &add_mask);
        }
        // Masked MEAN pool: mask is all-ones here → plain mean over the seq dim.
        let pooled = x.mean_dim(1).reshape([1, d]); // [1,768]
        // v1.5 recipe (model card): `F.layer_norm(embeddings, (768,))` between
        // the mean pool and the L2 normalisation, no affine, eps 1e-5. It
        // recentres and rescales each vector; text-to-text rankings barely
        // move, but the vision model was aligned against vectors that had
        // been through it, so without it a text query has near-zero
        // similarity to every image (measured 2026-06-21 and 2026-09-12).
        let mean = pooled.clone().mean_dim(1); // [1,1]
        let centred = pooled.sub(mean);
        let var = centred.clone().powf_scalar(2.0).mean_dim(1); // [1,1]
        let pooled = centred.div(var.add_scalar(1e-5).sqrt());
        l2_normalize(pooled)
    }
}

/// Warm nomic-embed-text-v1.5 handle: the encoder + a BERT WordPiece tokenizer.
///
/// Sentinel framing is done BY HAND here ([CLS] body [SEP] via the resolved
/// `cls_id`/`sep_id`), never by a tokenizer post-processor: both the
/// graph-built tokenizer (`crate::tokenizer::build_tokenizer`, which restores
/// a post-processor when its graph stores one) and the json-built one are
/// encoded with `add_special_tokens=false`, which never runs it, so both
/// substrates produce byte-identical sequences.
pub struct NomicTextEmbedder<B: Backend> {
    model: NomicTextModel<B>,
    tokenizer: tokenizers::Tokenizer,
    cls_id: u32,
    sep_id: u32,
    device: B::Device,
}

/// Resolve the BERT sentinels a `NomicTextEmbedder` frames with.
fn nomic_sentinel_ids(tokenizer: &tokenizers::Tokenizer) -> Result<(u32, u32)> {
    EmbeddingArchitecture::NomicTextV15.validate_tokenizer(tokenizer)?;
    let cls = tokenizer
        .token_to_id("[CLS]")
        .expect("validated Nomic [CLS] token");
    let sep = tokenizer
        .token_to_id("[SEP]")
        .expect("validated Nomic [SEP] token");
    Ok((cls, sep))
}

impl<B: Backend> NomicTextEmbedder<B> {
    /// Embed with a task prefix prepended before tokenizing, hand-framing the
    /// result as `[CLS] body [SEP]`.
    fn embed_prefixed(&self, prefix: &str, text: &str) -> Result<Vec<f32>> {
        let full = format!("{prefix}{text}");
        let enc = self
            .tokenizer
            .encode(full, false)
            .map_err(|e| anyhow::anyhow!("nomic tokenizer encode: {e}"))?;
        // Cap the sequence length: self-attention is O(n^2) in both compute and
        // GPU memory, so an over-long input (a full research fragment, the whole
        // cookbook) can try to allocate multiple GB and abort the process. nomic
        // mean-pools, so the leading tokens carry the topic — truncating the tail
        // is safe for a search embedding, and queries are always short.
        const MAX_TOKENS: usize = 2048;
        let body = enc.get_ids();
        let body = &body[..body.len().min(MAX_TOKENS - 2)];
        let mut ids: Vec<u32> = Vec::with_capacity(body.len() + 2);
        ids.push(self.cls_id);
        ids.extend_from_slice(body);
        ids.push(self.sep_id);
        Ok(self.model.embed(&ids, &self.device))
    }

    /// Number of tokens (incl [CLS]/[SEP]) for a prefixed text — for diagnostics.
    pub fn token_count(&self, prefix: &str, text: &str) -> usize {
        let full = format!("{prefix}{text}");
        self.tokenizer
            .encode(full, false)
            .map(|e| e.get_ids().len() + 2)
            .unwrap_or(0)
    }

    /// Embed stored content (`"search_document: "` prefix).
    pub fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_prefixed("search_document: ", text)
    }

    /// Embed a query (`"search_query: "` prefix).
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_prefixed("search_query: ", text)
    }
}

impl<B: Backend> LocalEmbedder for NomicTextEmbedder<B>
where
    B::Device: Send,
{
    /// Default text embedding = document side.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_document(text)
    }
    fn embed_image(&self, _bytes: &[u8]) -> Result<Vec<f32>> {
        anyhow::bail!("nomic-embed-text is text-only")
    }
    fn dim(&self) -> usize {
        NOMIC_TEXT_DIM
    }
}

/// Core: build a `NomicTextEmbedder` from explicit `model.safetensors` +
/// `tokenizer.json` paths. The nomic-v1.5 config is fixed in the model.
#[cfg(feature = "import")]
pub fn load_nomic_text_from_files(
    weights: &Path,
    tokenizer: &Path,
    device: WgpuDevice,
) -> Result<NomicTextEmbedder<B>> {
    if !weights.exists() {
        anyhow::bail!("missing nomic weights: {}", weights.display());
    }
    let tok = tokenizers::Tokenizer::from_file(tokenizer)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer.display()))?;
    let (cls_id, sep_id) = nomic_sentinel_ids(&tok)?;
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(weights));
    Ok(NomicTextEmbedder {
        model: NomicTextModel::load(&loader, &device),
        tokenizer: tok,
        cls_id,
        sep_id,
        device,
    })
}

/// Load a `NomicTextEmbedder` by HF model id
/// (e.g. `"nomic-ai/nomic-embed-text-v1.5"`), resolving `model.safetensors` +
/// `tokenizer.json` from one cached `main` revision. Must already be cached
/// (`huggingface-cli download <id>`).
#[cfg(feature = "import")]
pub fn load_nomic_text_from_hf(model_id: &str, device: WgpuDevice) -> Result<NomicTextEmbedder<B>> {
    let snapshot = hf_cache_main_snapshot(model_id)?;
    let weights = snapshot.join("model.safetensors");
    let tokenizer = snapshot.join("tokenizer.json");
    load_nomic_text_from_files(&weights, &tokenizer, device)
}

/// Assemble a `NomicTextEmbedder` from a content-addressed weight keymap and
/// an already-built tokenizer. Native callers select both from the same frozen
/// collection snapshot via [`crate::selection::load_keymap_from_graph`] and
/// [`crate::selection::load_tokenizer_from_graph`].
pub fn nomic_text_from_parts(
    keymap: HashMap<String, (Vec<f32>, Vec<usize>)>,
    tokenizer: tokenizers::Tokenizer,
    device: WgpuDevice,
) -> Result<NomicTextEmbedder<B>> {
    let (cls_id, sep_id) = nomic_sentinel_ids(&tokenizer)?;
    EmbeddingArchitecture::NomicTextV15.validate_keymap(&keymap)?;
    let loader = WeightLoader::Pile(keymap);
    Ok(NomicTextEmbedder {
        model: NomicTextModel::load(&loader, &device),
        tokenizer,
        cls_id,
        sep_id,
        device,
    })
}

/// Build a `NomicTextEmbedder` from a content-addressed weight keymap plus a
/// `tokenizer.json` path (the json-substrate variant of
/// [`nomic_text_from_parts`]).
pub fn load_nomic_text_from_keymap(
    keymap: HashMap<String, (Vec<f32>, Vec<usize>)>,
    tokenizer: &Path,
    device: WgpuDevice,
) -> Result<NomicTextEmbedder<B>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer.display()))?;
    nomic_text_from_parts(keymap, tok, device)
}

// ===========================================================================
// nomic-embed-vision-v1.5 — IMAGE LocalEmbedder, aligned to the SAME 768-d
// space as nomic-embed-text-v1.5 (cross-modal: text query ↔ image).
// ===========================================================================
//
// Port of `nomic-ai/nomic-embed-vision-v1.5` (HF `NomicVisionModel`, the
// `nomic_bert` ViT). A 12-layer PRE-NORM ViT with a learned-query attention
// pooling head (the "selector"). Verified against the trust_remote_code
// `modeling_hf_nomic_bert.py` shipped with nomic-bert-2048:
//   - Patch embed: a `nn.Linear` (NOT a conv) over flattened 16×16 patches,
//     ordered `(c, p1, p2)` channel-major — `embeddings.proj` [768, 3*16*16].
//     Equivalent to a stride-16 conv2d whose [768,3,16,16] kernel is the proj
//     weight reshaped, so we run it as a conv. img 224 / patch 16 → 14×14 = 196
//     patch tokens; prepend a learned `cls_token` → 197 tokens.
//   - Position: BOTH a learned absolute `pos_embed` [1,197,768] (added to the
//     CLS+patch sequence) AND a 2D axial RoPE applied inside attention to the
//     196 PATCH tokens only (the CLS prefix token is excluded). The RoPE is
//     timm's `RotaryEmbeddingCat` (in_pixels=False, feat=ref=14×14): freq_bands
//     of dim/4=16 over temperature 1e4; per-axis grid 0..14; sin/cos each
//     [196,32] → repeat_interleave(2) → [196,64]; applied as GPT-J-style
//     INTERLEAVED rotation `x*cos + rot(x)*sin`, `rot(x)=[-x1,x0, -x3,x2, ...]`.
//   - 12 PRE-NORM blocks with a running residual (GPT-NeoX style): per block
//       residual = x + residual           (x is the block input)
//       h        = norm1(residual)
//       h        = attn(h)               (RoPE on patch Q/K, full bidir attn)
//       residual = h + residual
//       h        = norm2(residual)
//       h        = mlp(h)
//       return (h, residual)
//     After all layers: `hidden = h + residual` (finalize the last MLP add).
//     There is NO final layernorm (`no_last_ln: true`).
//   - MLP is a gated SwiGLU WITH an inner LayerNorm on the 2048 hidden
//     (`norm_mlp: true`): `fc2(norm(fc11(x) * silu(fc12(x))))`. All of attn,
//     mlp, patch-proj carry biases here (unlike nomic-TEXT which is bias-free).
//   - Pooling = the model card's `F.normalize(last_hidden_state[:, 0])`, where
//     `last_hidden_state` is `[b,197,768]`: the encoder output PLUS the
//     `selector`'s term broadcast over every token. The selector is one
//     learned `latent` query cross-attending over the tokens (`[b,1,768]`),
//     then `mlp(norm1(attn_out))`, and the reference does
//     `hidden_states + mlp(normed)`; token 0 of that is the encoder CLS plus
//     the pooled MLP term (verified 2026-09-13 with the published remote code,
//     transformers 4.46). An earlier version of this port returned
//     `attn_out + mlp` as the embedding; those vectors had cosine 0.08-0.11 to
//     the reference's and no text query could find an image.
//
// Gotchas pinned during the parity pass (see `src/bin/nomic_vision_test.rs`):
//   - eps: encoder/selector LayerNorms use layer_norm_epsilon=1e-6; the MLP's
//     inner `norm` is a plain `nn.LayerNorm` with its DEFAULT eps=1e-5.
//   - RoPE excludes the CLS token (`num_prefix_tokens = max(register_tokens,1)
//     = 1`) and uses interleaved pairs (not rotate_half), with cos/sin
//     repeat-interleaved to match.
//   - Image preprocessing is CLIPImageProcessor: resize shortest side to 224
//     (PIL BICUBIC, resample=3) + center-crop 224 + the CLIP mean/std — the
//     SAME `preprocess_image_bytes` used by CLIP here.

/// nomic-embed-vision-v1.5 grid (14×14 patches) for the axial RoPE.
const NOMIC_VISION_GRID: usize = 14;

/// One PRE-NORM Nomic vision encoder layer with running-residual semantics and
/// 2D-RoPE attention over the patch tokens. Biased linears throughout.
struct NomicVisionLayer<B: Backend> {
    wqkv: Linear<B>, // packed [3*768, 768] + bias
    out_proj: Linear<B>,
    norm1: LayerNorm<B>,
    fc11: Linear<B>,        // value branch + bias
    fc12: Linear<B>,        // gate branch + bias
    mlp_norm: LayerNorm<B>, // inner norm on 2048 hidden, eps 1e-5
    fc2: Linear<B>,
    norm2: LayerNorm<B>,
    n_heads: usize,
    head_dim: usize,
    n_prefix: usize, // RoPE-excluded prefix tokens (CLS) = 1
    /// Activation quantisation and input capture, as the text layer has them
    /// (`NOMIC_ACT_QUANT`, `nomic_activation_capture_start`), so the same
    /// calibration recipe packs the vision model.
    act: ActQuant,
    prefix: String,
}

impl<B: Backend> NomicVisionLayer<B> {
    fn load(
        loader: &WeightLoader,
        p: &str,
        eps: f64,
        n_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let lin = |n: &str| Linear::load(loader, &format!("{p}.{n}"), true, device);
        Self {
            wqkv: lin("attn.Wqkv"),
            out_proj: lin("attn.out_proj"),
            norm1: LayerNorm::load(loader, &format!("{p}.norm1"), eps, device),
            fc11: lin("mlp.fc11"),
            fc12: lin("mlp.fc12"),
            // inner MLP LayerNorm uses torch's DEFAULT eps (1e-5), not 1e-6.
            mlp_norm: LayerNorm::load(loader, &format!("{p}.mlp.norm"), 1e-5, device),
            fc2: lin("mlp.fc2"),
            norm2: LayerNorm::load(loader, &format!("{p}.norm2"), eps, device),
            n_heads,
            head_dim,
            n_prefix: 1,
            act: ActQuant::from_env(),
            prefix: p.to_owned(),
        }
    }

    /// `x` is the block INPUT `[b,s,d]`; `residual` is the running residual sum
    /// (`None` only for the first layer); `cos`/`sin` are the patch RoPE tables
    /// `[n_patch, head_dim]`. Returns `(mlp_out, new_residual)`.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        residual: Option<Tensor<B, 3>>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);
        // running residual: residual = x + residual (or x for the first layer).
        let residual = match residual {
            Some(r) => x.add(r),
            None => x,
        };
        let hidden = self.norm1.forward(residual.clone());
        // packed qkv → q,k,v each [b,h,s,hd]
        let qkv = self
            .wqkv
            .forward(linear_input(hidden, self.act, &format!("{}.attn.Wqkv", self.prefix)));
        let q = qkv.clone().narrow(2, 0, d);
        let k = qkv.clone().narrow(2, d, d);
        let v = qkv.narrow(2, 2 * d, d);
        let shape = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2); // [b,h,s,hd]
        let q = self.apply_patch_rope(shape(q), cos, sin);
        let k = self.apply_patch_rope(shape(k), cos, sin);
        let v = shape(v);
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5)); // [b,h,s,s]
        let probs = softmax(scores, 3);
        let att = probs.matmul(v).swap_dims(1, 2).reshape([b, s, d]);
        let att = linear_input(att, self.act, &format!("{}.attn.out_proj", self.prefix));
        let attn_out = self.out_proj.forward(att);

        let residual = attn_out.add(residual);
        let hidden = self.norm2.forward(residual.clone());
        // gated SwiGLU with an inner LayerNorm: fc2(norm(fc11(x)*silu(fc12(x)))).
        // fc11 and fc12 read the same input, captured once under `.mlp.fc1`.
        let hq = linear_input(hidden, self.act, &format!("{}.mlp.fc1", self.prefix));
        let gated = self
            .fc11
            .forward(hq.clone())
            .mul(silu(self.fc12.forward(hq)));
        let inner = linear_input(self.mlp_norm.forward(gated), self.act, &format!("{}.mlp.fc2", self.prefix));
        let mlp = self.fc2.forward(inner);
        (mlp, residual)
    }

    /// Apply 2D axial RoPE (GPT-J interleaved) to the PATCH tokens of `[b,h,s,hd]`,
    /// leaving the first `n_prefix` (CLS) tokens untouched. `cos`/`sin` are the
    /// patch tables `[1,1,n_patch,hd]`.
    fn apply_patch_rope(
        &self,
        x: Tensor<B, 4>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [bb, hh, _s, hd] = x.dims();
        let np = NOMIC_VISION_GRID * NOMIC_VISION_GRID;
        let prefix = x.clone().narrow(2, 0, self.n_prefix); // [b,h,n_prefix,hd]
        let patches = x.narrow(2, self.n_prefix, np); // [b,h,np,hd]
        let cos = cos.clone().expand([bb, hh, np, hd]);
        let sin = sin.clone().expand([bb, hh, np, hd]);
        let rotated = interleaved_rotate(patches.clone())
            .mul(sin)
            .add(patches.mul(cos));
        Tensor::cat(vec![prefix, rotated], 2)
    }
}

/// GPT-J `rot(x)`: interleave `(-x[1], x[0], -x[3], x[2], ...)` over the last
/// dim (assumed even). Matches timm's `rot()` used by `apply_rot_embed_cat`.
fn interleaved_rotate<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let xr = x.reshape([b, h, s, half, 2]); // pairs along the last axis
    let x0 = xr.clone().narrow(4, 0, 1); // [.,.,.,half,1]
    let x1 = xr.narrow(4, 1, 1);
    // rot pair = (-x1, x0)
    Tensor::cat(vec![x1.neg(), x0], 4).reshape([b, h, s, d])
}

/// Build the nomic-vision 2D axial RoPE cos/sin tables `[1,1,n_patch,head_dim]`
/// on CPU (deterministic), matching timm `build_rotary_pos_embed`
/// (in_pixels=False, feat=ref=grid×grid, num_bands=head_dim/4):
///   bands[i] = 1 / 1e4^(i/(head_dim/4)) for i in 0..head_dim/4
///   per token (row r, col c): freqs = [r*bands ; c*bands]  (2*num_bands values)
///   sin/cos of freqs, then repeat_interleave(2) → head_dim values each.
fn nomic_vision_rope_tables<B: Backend>(
    grid: usize,
    head_dim: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let num_bands = head_dim / 4; // 16
    let temperature = 10000.0f64;
    let bands: Vec<f64> = (0..num_bands)
        .map(|i| 1.0 / temperature.powf(i as f64 / num_bands as f64))
        .collect();
    let np = grid * grid;
    let mut cos = vec![0f32; np * head_dim];
    let mut sin = vec![0f32; np * head_dim];
    for r in 0..grid {
        for c in 0..grid {
            let tok = r * grid + c;
            // freqs = concat(r*bands, c*bands) → 2*num_bands = head_dim/2 values,
            // then each value is repeated twice (repeat_interleave(2)).
            let mut freqs = Vec::with_capacity(2 * num_bands);
            for &b in &bands {
                freqs.push(r as f64 * b);
            }
            for &b in &bands {
                freqs.push(c as f64 * b);
            }
            for (j, &f) in freqs.iter().enumerate() {
                let (cf, sf) = (f.cos() as f32, f.sin() as f32);
                cos[tok * head_dim + 2 * j] = cf;
                cos[tok * head_dim + 2 * j + 1] = cf;
                sin[tok * head_dim + 2 * j] = sf;
                sin[tok * head_dim + 2 * j + 1] = sf;
            }
        }
    }
    let cos = Tensor::<B, 1>::from_data(TensorData::new(cos, [np * head_dim]), device)
        .reshape([1, 1, np, head_dim]);
    let sin = Tensor::<B, 1>::from_data(TensorData::new(sin, [np * head_dim]), device)
        .reshape([1, 1, np, head_dim]);
    (cos, sin)
}

/// The `selector` attention-pooling head: a learned `latent` query cross-attends
/// over the encoder hidden tokens, then a residual gated-SwiGLU MLP (no inner
/// norm). Returns the pooled `[b,768]` = the model's `last_hidden_state`.
struct NomicSelector<B: Backend> {
    latent: Tensor<B, 3>, // [1,1,768]
    wq: Linear<B>,        // [768,768] + bias
    wkv: Linear<B>,       // [2*768,768] + bias (packed k,v)
    out_proj: Linear<B>,
    norm1: LayerNorm<B>,
    fc11: Linear<B>,
    fc12: Linear<B>,
    fc2: Linear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> NomicSelector<B> {
    fn load(
        loader: &WeightLoader,
        eps: f64,
        n_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let sp = "selector";
        let lin = |n: &str| Linear::load(loader, &format!("{sp}.{n}"), true, device);
        Self {
            latent: loader.load_tensor(&format!("{sp}.attn.latent"), device),
            wq: lin("attn.Wq"),
            wkv: lin("attn.Wkv"),
            out_proj: lin("attn.out_proj"),
            norm1: LayerNorm::load(loader, &format!("{sp}.norm1"), eps, device),
            fc11: lin("mlp.fc11"),
            fc12: lin("mlp.fc12"),
            fc2: lin("mlp.fc2"),
            n_heads,
            head_dim,
        }
    }

    /// `x` is the encoder output `[b,n,d]`. Returns pooled `[b,d]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [b, n, d] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);
        // latent query (length 1); k,v from the encoder tokens.
        let q_latent = self.latent.clone().repeat_dim(0, b); // [b,1,d]
        let q = self
            .wq
            .forward(q_latent)
            .reshape([b, 1, h, hd])
            .swap_dims(1, 2); // [b,h,1,hd]
        let kv = self.wkv.forward(x); // [b,n,2d]
        let k = kv
            .clone()
            .narrow(2, 0, d)
            .reshape([b, n, h, hd])
            .swap_dims(1, 2); // [b,h,n,hd]
        let v = kv.narrow(2, d, d).reshape([b, n, h, hd]).swap_dims(1, 2);
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5)); // [b,h,1,n]
        let probs = softmax(scores, 3);
        let att = probs.matmul(v).swap_dims(1, 2).reshape([b, 1, d]); // [b,1,d]
        let attn_out = self.out_proj.forward(att); // [b,1,d]
        // The reference adds `mlp(norm1(attn_out))`, one vector, to EVERY
        // encoder token (`hidden_states + self.mlp(normed)` broadcasts the
        // `[b,1,d]` pooled term over `[b,197,d]`), and the model card takes
        // token 0 of that. So the selector's contribution is this term alone;
        // `attn_out` itself is never part of the embedding. An earlier port
        // returned `attn_out + mlp` and its vectors had cosine 0.08 to 0.11 to
        // the reference's (verified 2026-09-13).
        let normed = self.norm1.forward(attn_out);
        let gated = self
            .fc11
            .forward(normed.clone())
            .mul(silu(self.fc12.forward(normed)));
        self.fc2.forward(gated).reshape([b, d])
    }
}

struct NomicVisionModel<B: Backend> {
    cls_token: Tensor<B, 3>,          // [1,1,768]
    patch_weight: Tensor<B, 4>,       // proj weight reshaped to [768,3,16,16]
    patch_bias: Tensor<B, 1>,         // [768]
    position_embedding: Tensor<B, 3>, // [1,197,768]
    layers: Vec<NomicVisionLayer<B>>,
    selector: NomicSelector<B>,
    patch_size: usize,
    head_dim: usize,
}

impl<B: Backend> NomicVisionModel<B> {
    fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let (n_layers, n_heads, head_dim, eps) = (12usize, 12usize, 64usize, 1e-6);
        let layers = (0..n_layers)
            .map(|i| {
                NomicVisionLayer::load(
                    loader,
                    &format!("layers.{i}"),
                    eps,
                    n_heads,
                    head_dim,
                    device,
                )
            })
            .collect();
        // proj is a Linear [768, 3*16*16] ordered (c,p1,p2) — reshape to a conv
        // kernel [768,3,16,16] so a stride-16 conv reproduces it exactly.
        let proj_w: Tensor<B, 2> = loader.load_tensor("embeddings.proj.weight", device); // [768,768]
        let patch_weight = proj_w.reshape([768, 3, 16, 16]);
        Self {
            cls_token: loader.load_tensor("embeddings.cls_token", device),
            patch_weight,
            patch_bias: loader.load_tensor("embeddings.proj.bias", device),
            position_embedding: loader.load_tensor("embeddings.pos_embed", device),
            layers,
            selector: NomicSelector::load(loader, eps, n_heads, head_dim, device),
            patch_size: 16,
            head_dim,
        }
    }

    /// `image` is `[1,3,224,224]` normalized. Returns the L2-normalized `[768]`.
    fn embed(&self, image: Tensor<B, 4>, device: &B::Device) -> Vec<f32> {
        let [b, _, _, _] = image.dims();
        let opts = ConvOptions::new([self.patch_size, self.patch_size], [0, 0], [1, 1], 1);
        let patch = conv2d(
            image,
            self.patch_weight.clone(),
            Some(self.patch_bias.clone()),
            opts,
        ); // [b,768,14,14]
        let [_, c, gh, gw] = patch.dims();
        let np = gh * gw; // 196
        let patches = patch.reshape([b, c, np]).swap_dims(1, 2); // [b,196,768]
        let cls = self.cls_token.clone().repeat_dim(0, b); // [b,1,768]
        let x = Tensor::cat(vec![cls, patches], 1); // [b,197,768]
        // add learned absolute position embedding (over CLS+patch sequence).
        let mut x = x.add(self.position_embedding.clone());

        let (cos, sin) = nomic_vision_rope_tables::<B>(NOMIC_VISION_GRID, self.head_dim, device);
        let mut residual: Option<Tensor<B, 3>> = None;
        for layer in &self.layers {
            let (h, r) = layer.forward(x, residual, &cos, &sin);
            x = h;
            residual = Some(r);
        }
        // finalize the last MLP residual add (no final layernorm).
        let hidden = match residual {
            Some(r) => x.add(r),
            None => x,
        };
        // The embedding is token 0 of the reference's `last_hidden_state`,
        // which is the encoder output plus the selector's pooled MLP term
        // broadcast over every token (see `NomicSelector::forward`): the
        // encoder CLS plus that term, L2-normalised. Measured 2026-09-13
        // against the published models on CPU: the encoder CLS alone has
        // cosine 0.81 to 0.86 to the reference vector, the old selector
        // output 0.08 to 0.11.
        let [b, _, d] = hidden.dims();
        let cls = hidden.clone().narrow(1, 0, 1).reshape([b, d]); // [b,768]
        let pooled = cls.add(self.selector.forward(hidden));
        l2_normalize(pooled)
    }
}

/// Warm nomic-embed-vision-v1.5 handle: the ViT encoder + selector, image-only.
/// Embeddings land in the SAME 768-d space as `NomicTextEmbedder`.
pub struct NomicVisionEmbedder<B: Backend> {
    model: NomicVisionModel<B>,
    device: B::Device,
}

impl<B: Backend> LocalEmbedder for NomicVisionEmbedder<B>
where
    B::Device: Send,
{
    fn embed_image(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        // CLIPImageProcessor: bicubic resize shortest side 224 + center-crop 224
        // + CLIP mean/std — the exact same preprocessing as CLIP here.
        let pixels = preprocess_image_bytes(bytes)?; // [3*224*224]
        let image =
            Tensor::<B, 4>::from_data(TensorData::new(pixels, [1, 3, 224, 224]), &self.device);
        Ok(self.model.embed(image, &self.device))
    }
    fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
        anyhow::bail!("nomic-embed-vision is image-only; use NomicTextEmbedder for text")
    }
    fn dim(&self) -> usize {
        NOMIC_TEXT_DIM // 768, the same shared space as nomic-text.
    }
}

/// Core: build a `NomicVisionEmbedder` from an explicit `model.safetensors` path.
/// No tokenizer (image-only); the v1.5 vision config is fixed in the model.
#[cfg(feature = "import")]
pub fn load_nomic_vision_from_files(
    weights: &Path,
    device: WgpuDevice,
) -> Result<NomicVisionEmbedder<B>> {
    if !weights.exists() {
        anyhow::bail!("missing nomic-vision weights: {}", weights.display());
    }
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(weights));
    Ok(NomicVisionEmbedder {
        model: NomicVisionModel::load(&loader, &device),
        device,
    })
}

/// Load a `NomicVisionEmbedder` by HF model id
/// (e.g. `"nomic-ai/nomic-embed-vision-v1.5"`), resolving `model.safetensors`
/// from one cached `main` revision. Must already be cached.
#[cfg(feature = "import")]
pub fn load_nomic_vision_from_hf(
    model_id: &str,
    device: WgpuDevice,
) -> Result<NomicVisionEmbedder<B>> {
    let snapshot = hf_cache_main_snapshot(model_id)?;
    let weights = snapshot.join("model.safetensors");
    load_nomic_vision_from_files(&weights, device)
}

/// Build a `NomicVisionEmbedder` from a content-addressed weight keymap, such
/// as one selected from a frozen native collection snapshot with
/// [`crate::selection::load_keymap_from_graph`].
pub fn load_nomic_vision_from_keymap(
    keymap: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: WgpuDevice,
) -> Result<NomicVisionEmbedder<B>> {
    EmbeddingArchitecture::NomicVisionV15.validate_keymap(&keymap)?;
    let loader = WeightLoader::Pile(keymap);
    Ok(NomicVisionEmbedder {
        model: NomicVisionModel::load(&loader, &device),
        device,
    })
}
