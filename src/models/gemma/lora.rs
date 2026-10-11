//! LoRA (Low-Rank Adaptation) for the Gemma 4 text decoder.
//!
//! Adapter weights trained while the base model stays frozen. LoRA decomposes
//! a weight update into low-rank matrices: ΔW = scale * B @ A, where A is
//! [rank, in_features] and B is [out_features, rank]. B starts at zero, so the
//! adapted model is exactly the base model at step 0.
//!
//! Adapters are keyed by the checkpoint's projection paths:
//! `layers.{i}.self_attn.{q,k,v,o}_proj` and `layers.{i}.mlp.{gate,up,down}_proj`.
//! KV-shared layers (the last `num_kv_shared_layers`) never run k_proj/v_proj —
//! see `Gemma4Attention::forward` — so they get q/o + MLP adapters only; k/v
//! adapters there would be dead weight that never receives a gradient. The same
//! logic drops k_proj adapters on K=V full-attention layers (dense 12B/31B):
//! there k is defined as v, k_proj never runs, and the v_proj adapter adapts
//! both k and v at once.
//!
//! Persistence: safetensors (PEFT-style `{key}.lora_A.weight` names) and the
//! pile (content-addressed adapter entities). The pile attribute IDs are shared
//! verbatim with avatar/gaze, so LoRA sets are queryable across model families.

use std::collections::HashMap;
#[cfg(feature = "import")]
use std::path::Path;

use burn::nn::Linear;
use burn::prelude::*;
#[cfg(feature = "import")]
use burn::tensor::TensorData;
use triblespace::prelude::*;

use super::gemma4::config::{Gemma4TextConfig, LayerType};
use crate::format::F32Array;

pub mod attrs {
    use crate::format::F32Array;
    use triblespace::prelude::blobencodings::UTF8String;
    use triblespace::prelude::inlineencodings::{F64, GenId, Handle, U256BE};
    use triblespace::prelude::*;

    attributes! {
        // Shared IDs with avatar/gaze — same attributes, cross-model LoRA queries.
        /// Low-rank dimension of LoRA adapters.
        "1A682F45CE40171DD5C6FDB4F086AD69" as lora_rank: U256BE;
        /// LoRA alpha (scale = alpha / rank).
        "198B03AF556B7505CCC9ABD4A1D6E724" as lora_alpha: F64;
        /// Reference to a LoRA adapter entity (repeated on the LoRA set entity).
        "B93C4E66F4B9553BF0E8B5DBAD116ECF" as lora_adapter: GenId;
        /// Projection name for a LoRA target (e.g. "layers.0.self_attn.q_proj").
        "FF8335C187823A267E26B4E33EF157E9" as lora_projection: Handle<UTF8String>;
        /// LoRA A matrix blob [rank, in_features].
        "7CD7F0DC8BDA328735A22DF02B4B8828" as lora_a: Handle<F32Array>;
        /// LoRA B matrix blob [out_features, rank].
        "1F21DAE68652A4D8CAD973400F04124D" as lora_b: Handle<F32Array>;
        /// Model name (same shared id as `crate::format::attrs::model_name`).
        "4C1CD1611863E7854C59C7DC706DF77A" as model_name: Handle<UTF8String>;
    }
}

/// A single LoRA adapter for one linear layer.
pub struct LoraAdapter<B: Backend> {
    pub lora_a: Tensor<B, 2>, // [rank, in_features]
    pub lora_b: Tensor<B, 2>, // [out_features, rank]
    pub scale: f32,           // alpha / rank
}

impl<B: Backend> LoraAdapter<B> {
    /// Initialize with Kaiming uniform for A, zeros for B, so the LoRA
    /// contribution starts at zero (B @ A = 0) and step 0 IS the base model.
    pub fn init(
        rank: usize,
        out_features: usize,
        in_features: usize,
        scale: f32,
        device: &B::Device,
    ) -> Self {
        let bound = (3.0 / in_features as f64).sqrt();
        let lora_a = Tensor::random(
            [rank, in_features],
            burn::tensor::Distribution::Uniform(-bound, bound),
            device,
        );
        let lora_b = Tensor::zeros([out_features, rank], device);
        Self {
            lora_a,
            lora_b,
            scale,
        }
    }
}

/// Homogeneous-rank/alpha LoRA adapters for a Gemma 4 text decoder, keyed by
/// checkpoint weight path. Each adapter carries its own numerical scale;
/// `rank` and `alpha` summarize the common parameters for training/export.
pub struct LoraWeights<B: Backend> {
    pub adapters: HashMap<String, LoraAdapter<B>>,
    pub rank: usize,
    pub alpha: f32,
}

impl<B: Backend> LoraWeights<B> {
    /// Create LoRA adapters for every trainable projection of a Gemma 4 text
    /// decoder. Per-layer shapes follow the layer type (full-attention layers
    /// have a wider head_dim, so q/k/v/o dims differ from sliding layers);
    /// KV-shared layers (index >= `first_shared_kv_layer`) skip k/v adapters
    /// because their forward never runs k_proj/v_proj, and K=V full-attention
    /// layers (`attention_k_eq_v`, dense 12B/31B) skip k_proj because k is
    /// defined as v there — the v_proj adapter adapts both.
    pub fn init_gemma4(
        config: &Gemma4TextConfig,
        rank: usize,
        alpha: f32,
        device: &B::Device,
    ) -> Self {
        let scale = alpha / rank as f32;
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let first_shared = config.first_shared_kv_layer();
        let mut adapters = HashMap::new();

        for i in 0..config.num_hidden_layers {
            let (n_kv_heads, head_dim) = match config.layer_type(i) {
                LayerType::SlidingAttention => (config.num_key_value_heads, config.head_dim),
                LayerType::FullAttention => (config.global_kv_heads(), config.global_head_dim()),
            };
            let q_dim = config.num_attention_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;

            let mut targets: Vec<(String, usize, usize)> = vec![
                (format!("layers.{i}.self_attn.q_proj"), q_dim, hidden),
                (format!("layers.{i}.self_attn.o_proj"), hidden, q_dim),
                (format!("layers.{i}.mlp.gate_proj"), inter, hidden),
                (format!("layers.{i}.mlp.up_proj"), inter, hidden),
                (format!("layers.{i}.mlp.down_proj"), hidden, inter),
            ];
            if i < first_shared {
                let k_is_v =
                    config.attention_k_eq_v && config.layer_type(i) == LayerType::FullAttention;
                if !k_is_v {
                    targets.push((format!("layers.{i}.self_attn.k_proj"), kv_dim, hidden));
                }
                targets.push((format!("layers.{i}.self_attn.v_proj"), kv_dim, hidden));
            }
            for (key, out_dim, in_dim) in targets {
                adapters.insert(key, LoraAdapter::init(rank, out_dim, in_dim, scale, device));
            }
        }

        eprintln!(
            "Initialized {} LoRA adapters (rank={}, alpha={}, scale={:.4})",
            adapters.len(),
            rank,
            alpha,
            scale
        );
        Self {
            adapters,
            rank,
            alpha,
        }
    }

    /// Get a LoRA adapter by key, if it exists.
    pub fn get(&self, key: &str) -> Option<&LoraAdapter<B>> {
        self.adapters.get(key)
    }

    /// Count total trainable parameters.
    pub fn num_params(&self) -> usize {
        self.adapters
            .values()
            .map(|a| {
                let [r, in_f] = a.lora_a.dims();
                let [out_f, r2] = a.lora_b.dims();
                r * in_f + out_f * r2
            })
            .sum()
    }

    /// Save LoRA weights to a safetensors file (PEFT-style tensor names,
    /// rank/alpha in the file metadata).
    #[cfg(feature = "import")]
    pub fn save(&self, path: &Path) {
        use safetensors::Dtype;
        use std::borrow::Cow;

        struct RawTensor {
            data: Vec<u8>,
            shape: Vec<usize>,
        }
        impl safetensors::View for RawTensor {
            fn dtype(&self) -> Dtype {
                Dtype::F32
            }
            fn shape(&self) -> &[usize] {
                &self.shape
            }
            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Borrowed(&self.data)
            }
            fn data_len(&self) -> usize {
                self.data.len()
            }
        }

        let mut tensors: Vec<(String, RawTensor)> = Vec::new();
        for (key, adapter) in &self.adapters {
            let a_data: Vec<f32> = adapter.lora_a.clone().to_data().to_vec().unwrap();
            tensors.push((
                format!("{key}.lora_A.weight"),
                RawTensor {
                    data: a_data.iter().flat_map(|f| f.to_le_bytes()).collect(),
                    shape: adapter.lora_a.dims().to_vec(),
                },
            ));
            let b_data: Vec<f32> = adapter.lora_b.clone().to_data().to_vec().unwrap();
            tensors.push((
                format!("{key}.lora_B.weight"),
                RawTensor {
                    data: b_data.iter().flat_map(|f| f.to_le_bytes()).collect(),
                    shape: adapter.lora_b.dims().to_vec(),
                },
            ));
        }

        let metadata = HashMap::from([
            ("rank".to_string(), self.rank.to_string()),
            ("alpha".to_string(), self.alpha.to_string()),
        ]);
        safetensors::serialize_to_file(tensors, &Some(metadata), path).unwrap();
        eprintln!("Saved LoRA weights to: {}", path.display());
    }

    /// Load LoRA weights from a safetensors file.
    #[cfg(feature = "import")]
    pub fn load(path: &Path, device: &B::Device) -> Self {
        use safetensors::SafeTensors;

        let file_data = std::fs::read(path).unwrap();
        let (_, meta) = SafeTensors::read_metadata(&file_data).unwrap();
        let metadata = meta.metadata();
        let rank: usize = metadata
            .as_ref()
            .and_then(|m| m.get("rank"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let alpha: f32 = metadata
            .as_ref()
            .and_then(|m| m.get("alpha"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(16.0);

        let st = SafeTensors::deserialize(&file_data).unwrap();
        let scale = alpha / rank as f32;

        let mut adapters = HashMap::new();
        for name in st.names() {
            let Some(key) = name.strip_suffix(".lora_A.weight") else {
                continue;
            };
            let get = |n: &str| -> Tensor<B, 2> {
                let view = st.tensor(n).unwrap();
                assert_eq!(view.dtype(), safetensors::Dtype::F32, "{n}: expected f32");
                let data: Vec<f32> = view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Tensor::from_data(TensorData::new(data, view.shape().to_vec()), device)
            };
            adapters.insert(
                key.to_string(),
                LoraAdapter {
                    lora_a: get(name),
                    lora_b: get(&format!("{key}.lora_B.weight")),
                    scale,
                },
            );
        }

        eprintln!(
            "Loaded {} LoRA adapters from {} (rank={}, alpha={})",
            adapters.len(),
            path.display(),
            rank,
            alpha
        );
        Self {
            adapters,
            rank,
            alpha,
        }
    }

    /// Save LoRA weights as structured entities into a blob store (pile or
    /// memory). Each adapter becomes a content-derived entity carrying its
    /// projection name, rank/alpha, and A/B blob handles; the returned fragment
    /// uses the shared member-only model root. Rank/alpha interpret an adapter's
    /// arrays and scale, so they belong to the members' content, not the root's
    /// identity core. The family-native set fields remain root annotations.
    /// `parent` is the actual training base when known; a model name cannot
    /// supply it. Older stored adapter/set IDs are never reconstructed or
    /// rewritten by this new-writer shape.
    pub fn save_to_pile(
        &self,
        name: &str,
        parent: Option<Id>,
        blobs: &mut impl BlobStorePut,
    ) -> Result<Fragment, Box<dyn std::error::Error>> {
        let mut adapter_ids: Vec<Id> = Vec::new();
        let mut facts = TribleSet::new();

        // Deterministic order so the fragment (and thus the set entity's
        // content-derived id) is stable for identical adapter sets.
        let mut keys: Vec<&String> = self.adapters.keys().collect();
        keys.sort();
        for key in keys {
            let adapter = &self.adapters[key];
            let a_data: Vec<f32> = adapter.lora_a.clone().to_data().to_vec().unwrap();
            let b_data: Vec<f32> = adapter.lora_b.clone().to_data().to_vec().unwrap();

            let adapter_ent = entity! { _ @
                attrs::lora_projection: blobs.put::<blobencodings::UTF8String, _>(key.clone())?,
                attrs::lora_rank: self.rank as u32,
                attrs::lora_alpha: self.alpha as f64,
                attrs::lora_a: blobs.put::<F32Array, _>(a_data)?,
                attrs::lora_b: blobs.put::<F32Array, _>(b_data)?,
            };
            adapter_ids.push(adapter_ent.root().expect("adapter entity has root"));
            facts += adapter_ent.into_facts();
        }

        let set_ent = entity! { _ @ crate::format::attrs::member*: adapter_ids.iter() };
        let root = set_ent.root().expect("lora set entity has root");
        facts += set_ent.into_facts();
        facts += entity! { ExclusiveId::force_ref(&root) @
            attrs::lora_rank: self.rank as u32,
            attrs::lora_alpha: self.alpha as f64,
            attrs::model_name: blobs.put::<blobencodings::UTF8String, _>(name.to_string())?,
            attrs::lora_adapter*: adapter_ids.iter(),
            crate::format::attrs::parent?: parent.filter(|parent| *parent != root),
        };

        eprintln!(
            "Saved {} LoRA adapters to pile (rank={}, alpha={}).",
            adapter_ids.len(),
            self.rank,
            self.alpha
        );
        Ok(Fragment::rooted(root, facts))
    }

    /// Load the explicitly named, opaque LoRA root from pile facts + blobs.
    /// Shared members carry their own rank (A is [rank, in], B is [out, rank])
    /// and alpha; root annotations never determine their shapes or scale.
    /// Legacy native-link roots instead carry those parameters on the set.
    /// This numerical type and its writers require homogeneous rank/alpha, so
    /// its summary fields come from the selected adapters' common parameters.
    pub fn load_from_pile(
        tribles: &TribleSet,
        blobs: &impl BlobStoreGet,
        root: Id,
        device: &B::Device,
    ) -> Self {
        let mut adapters = HashMap::new();
        let mut parameters = None;
        let mut load_adapter =
            |proj: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
             rank_v: Inline<inlineencodings::U256BE>,
             alpha: f64,
             a_h: Inline<inlineencodings::Handle<F32Array>>,
             b_h: Inline<inlineencodings::Handle<F32Array>>| {
                let proj: anybytes::View<str> = blobs.get(proj).expect("lora_projection blob");
                let proj = proj.to_string();
                // U256BE stores the u32 rank big-endian in the low (last) 8 bytes.
                let rank = u64::from_be_bytes(rank_v.raw[24..32].try_into().unwrap()) as usize;
                let alpha = alpha as f32;
                let scale = alpha / rank as f32;
                if let Some(common) = parameters {
                    assert_eq!(
                        common,
                        (rank, alpha),
                        "selected root {root} cannot be represented as homogeneous LoRA weights: \
                     adapter {proj} has different rank/alpha"
                    );
                } else {
                    parameters = Some((rank, alpha));
                }

                let a_bytes: anybytes::Bytes = blobs.get(a_h).expect("lora_a blob");
                let a_data: anybytes::View<[f32]> = a_bytes.view().expect("lora_a view");
                let b_bytes: anybytes::Bytes = blobs.get(b_h).expect("lora_b blob");
                let b_data: anybytes::View<[f32]> = b_bytes.view().expect("lora_b view");

                let in_features = a_data.len() / rank;
                let out_features = b_data.len() / rank;
                let lora_a =
                    Tensor::<B, 1>::from_floats(&a_data[..], device).reshape([rank, in_features]);
                let lora_b =
                    Tensor::<B, 1>::from_floats(&b_data[..], device).reshape([out_features, rank]);
                adapters.insert(
                    proj,
                    LoraAdapter {
                        lora_a,
                        lora_b,
                        scale,
                    },
                );
            };

        if exists!(pattern!(tribles, [{
            root @ crate::format::attrs::member: _?member,
        }])) {
            for (proj, rank, alpha, a_h, b_h) in find!(
                (proj: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
                 rank: Inline<inlineencodings::U256BE>, alpha: f64,
                 a_h: Inline<inlineencodings::Handle<F32Array>>,
                 b_h: Inline<inlineencodings::Handle<F32Array>>),
                pattern!(tribles, [{ root @ crate::format::attrs::member: _?adapter }, {
                    _?adapter @
                    attrs::lora_projection: ?proj,
                    attrs::lora_rank: ?rank,
                    attrs::lora_alpha: ?alpha,
                    attrs::lora_a: ?a_h,
                    attrs::lora_b: ?b_h,
                }])
            ) {
                load_adapter(proj, rank, alpha, a_h, b_h);
            }
        } else {
            // Old stored sets use native links and set-local parameters. Keep
            // those opaque roots readable without manufacturing shared roots.
            for (proj, rank, alpha, a_h, b_h) in find!(
                (proj: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
                 rank: Inline<inlineencodings::U256BE>, alpha: f64,
                 a_h: Inline<inlineencodings::Handle<F32Array>>,
                 b_h: Inline<inlineencodings::Handle<F32Array>>),
                pattern!(tribles, [{
                    root @
                    attrs::lora_adapter: _?adapter,
                    attrs::lora_rank: ?rank,
                    attrs::lora_alpha: ?alpha,
                }, {
                    _?adapter @
                    attrs::lora_projection: ?proj,
                    attrs::lora_a: ?a_h,
                    attrs::lora_b: ?b_h,
                }])
            ) {
                load_adapter(proj, rank, alpha, a_h, b_h);
            }
        }

        let (rank, alpha) =
            parameters.unwrap_or_else(|| panic!("LoRA root {root} has no supported adapters"));

        eprintln!(
            "Loaded {} LoRA adapters from pile (rank={}, alpha={}).",
            adapters.len(),
            rank,
            alpha
        );
        Self {
            adapters,
            rank,
            alpha,
        }
    }
}

/// Linear forward with an optional LoRA delta: base + scale * (x @ Aᵀ @ Bᵀ).
/// The None path (or a key without an adapter) is exactly `linear.forward(x)` —
/// no clone, no lookup-formatting cost at inference.
pub fn maybe_lora<B: Backend>(
    linear: &Linear<B>,
    x: Tensor<B, 3>,
    lora: Option<&LoraWeights<B>>,
    key: &str,
) -> Tensor<B, 3> {
    match lora.and_then(|l| l.get(key)) {
        Some(adapter) => {
            let base = linear.forward(x.clone());
            let at = adapter.lora_a.clone().transpose().unsqueeze::<3>(); // [1, in, rank]
            let bt = adapter.lora_b.clone().transpose().unsqueeze::<3>(); // [1, rank, out]
            base + x.matmul(at).matmul(bt) * adapter.scale
        }
        None => linear.forward(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ed25519_dalek::SigningKey;

    type TB = NdArray;

    fn tensor_vec<const D: usize>(t: &Tensor<TB, D>) -> Vec<f32> {
        t.to_data().to_vec().unwrap()
    }

    #[test]
    fn lora_root_identity_keeps_scale_in_members_and_labels_outside() {
        let device = burn::prelude::Device::<TB>::default();
        let weights = |alpha| LoraWeights::<TB> {
            rank: 1,
            alpha,
            adapters: HashMap::from([(
                "layers.0.self_attn.q_proj".to_owned(),
                LoraAdapter {
                    lora_a: Tensor::from_floats([[1.0, 2.0]], &device),
                    lora_b: Tensor::from_floats([[3.0], [4.0]], &device),
                    scale: alpha,
                },
            )]),
        };
        let mut blobs = MemoryBlobStore::new();
        let first = weights(2.0)
            .save_to_pile("first", Some(fucid().id), &mut blobs)
            .unwrap();
        let alias = weights(2.0)
            .save_to_pile("alias", None, &mut blobs)
            .unwrap();
        let rescaled = weights(4.0)
            .save_to_pile("first", None, &mut blobs)
            .unwrap();
        assert_eq!(first.root(), alias.root());
        assert_ne!(
            first.root(),
            rescaled.root(),
            "adapter scale changes the model's meaning"
        );
    }

    #[test]
    fn lora_load_uses_only_the_named_opaque_roots_members_and_scale() {
        let device = burn::prelude::Device::<TB>::default();
        let mut blobs = MemoryBlobStore::new();
        let first_adapter = entity! { _ @
            attrs::lora_projection: blobs
                .put::<blobencodings::UTF8String, _>("layers.0.self_attn.q_proj".to_owned())
                .unwrap(),
            attrs::lora_rank: 1u32,
            attrs::lora_alpha: 2.0f64,
            attrs::lora_a: blobs.put::<F32Array, _>(vec![1.0, 2.0]).unwrap(),
            attrs::lora_b: blobs.put::<F32Array, _>(vec![3.0, 4.0]).unwrap(),
        };
        let second_adapter = entity! { _ @
            attrs::lora_projection: blobs
                .put::<blobencodings::UTF8String, _>("layers.0.self_attn.q_proj".to_owned())
                .unwrap(),
            attrs::lora_rank: 2u32,
            attrs::lora_alpha: 8.0f64,
            attrs::lora_a: blobs.put::<F32Array, _>(vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
            attrs::lora_b: blobs.put::<F32Array, _>(vec![50.0, 60.0, 70.0, 80.0]).unwrap(),
        };
        let unselected_adapter = entity! { _ @
            attrs::lora_projection: blobs
                .put::<blobencodings::UTF8String, _>("layers.0.mlp.down_proj".to_owned())
                .unwrap(),
            attrs::lora_a: blobs.put::<F32Array, _>(vec![90.0, 91.0]).unwrap(),
            attrs::lora_b: blobs.put::<F32Array, _>(vec![92.0, 93.0]).unwrap(),
        };
        let first_root = fucid();
        let second_root = fucid();
        let legacy_root = fucid();
        let mut facts = entity! { &first_root @
            crate::format::attrs::member: first_adapter.root().unwrap(),
            attrs::lora_rank: 1u32,
            attrs::lora_alpha: 2.0f64,
        }
        .into_facts();
        facts += entity! { &second_root @
            crate::format::attrs::member: second_adapter.root().unwrap(),
        };
        // A pre-shared-membership root remains usable at its stored ID, with
        // its own set-local scale and no graph-wide adapter discovery.
        facts += entity! { &legacy_root @
            attrs::lora_adapter: first_adapter.root().unwrap(),
            attrs::lora_rank: 1u32,
            attrs::lora_alpha: 3.0f64,
        };
        facts += first_adapter.into_facts();
        facts += second_adapter.into_facts();
        facts += unselected_adapter.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        let first = LoraWeights::<TB>::load_from_pile(&facts, &reader, first_root.id, &device);
        assert_eq!((first.rank, first.alpha, first.adapters.len()), (1, 2.0, 1));
        let adapter = first.get("layers.0.self_attn.q_proj").unwrap();
        assert_eq!(adapter.lora_a.dims(), [1, 2]);
        assert_eq!(adapter.lora_b.dims(), [2, 1]);
        assert_eq!(tensor_vec(&adapter.lora_a), vec![1.0, 2.0]);
        assert_eq!(tensor_vec(&adapter.lora_b), vec![3.0, 4.0]);
        assert_eq!(adapter.scale, 2.0);

        let second = LoraWeights::<TB>::load_from_pile(&facts, &reader, second_root.id, &device);
        assert_eq!(
            (second.rank, second.alpha, second.adapters.len()),
            (2, 8.0, 1)
        );
        let adapter = second.get("layers.0.self_attn.q_proj").unwrap();
        assert_eq!(adapter.lora_a.dims(), [2, 2]);
        assert_eq!(adapter.lora_b.dims(), [2, 2]);
        assert_eq!(tensor_vec(&adapter.lora_a), vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(tensor_vec(&adapter.lora_b), vec![50.0, 60.0, 70.0, 80.0]);
        assert_eq!(adapter.scale, 4.0);

        let legacy = LoraWeights::<TB>::load_from_pile(&facts, &reader, legacy_root.id, &device);
        assert_eq!(
            (legacy.rank, legacy.alpha, legacy.adapters.len()),
            (1, 3.0, 1)
        );
        let adapter = legacy.get("layers.0.self_attn.q_proj").unwrap();
        assert_eq!(tensor_vec(&adapter.lora_a), vec![1.0, 2.0]);
        assert_eq!(tensor_vec(&adapter.lora_b), vec![3.0, 4.0]);
        assert_eq!(adapter.scale, 3.0);
    }

    #[test]
    fn shared_lora_computation_ignores_conflicting_root_annotations() {
        let device = burn::prelude::Device::<TB>::default();
        let key = "layers.0.self_attn.q_proj";
        let weights = LoraWeights::<TB> {
            rank: 1,
            alpha: 2.0,
            adapters: HashMap::from([(
                key.to_owned(),
                LoraAdapter {
                    lora_a: Tensor::from_floats([[1.0, 2.0]], &device),
                    lora_b: Tensor::from_floats([[3.0], [4.0]], &device),
                    scale: 2.0,
                },
            )]),
        };
        let mut blobs = MemoryBlobStore::new();
        let fragment = weights.save_to_pile("adapter", None, &mut blobs).unwrap();
        let root = fragment.root().unwrap();
        let mut facts = fragment.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let before = LoraWeights::<TB>::load_from_pile(&facts, &reader, root, &device);

        facts += entity! { ExclusiveId::force_ref(&root) @
            attrs::lora_rank*: [0u32, 4u32],
            attrs::lora_alpha*: [0.0f64, 64.0f64],
        };
        let after = LoraWeights::<TB>::load_from_pile(&facts, &reader, root, &device);
        assert_eq!((after.rank, after.alpha), (before.rank, before.alpha));
        assert_eq!((after.rank, after.alpha), (1, 2.0));
        let adapter = after.get(key).unwrap();
        assert_eq!(adapter.lora_a.dims(), [1, 2]);
        assert_eq!(adapter.lora_b.dims(), [2, 1]);
        assert_eq!(adapter.scale, 2.0);

        let linear = burn::nn::LinearConfig::new(2, 2)
            .with_bias(false)
            .init(&device);
        let x = Tensor::<TB, 3>::from_floats([[[2.0, 1.0]]], &device);
        let base = linear.forward(x.clone());
        let before = maybe_lora(&linear, x.clone(), Some(&before), key);
        let after = maybe_lora(&linear, x, Some(&after), key);
        assert_ne!(
            tensor_vec(&before),
            tensor_vec(&base),
            "the adapter contributes"
        );
        assert_eq!(tensor_vec(&after), tensor_vec(&before));
    }

    /// Round-trip a small adapter set through a REAL temp on-disk pile
    /// (native collection commit, cold reopen, exact snapshot, load) and assert the
    /// tensors come back bit-identical with the right shapes.
    #[test]
    fn lora_pile_roundtrip() {
        let device = burn::prelude::Device::<TB>::default();

        let rank = 2;
        let alpha = 4.0f32;
        let scale = alpha / rank as f32;
        let mut adapters = HashMap::new();
        adapters.insert(
            "layers.0.self_attn.q_proj".to_string(),
            LoraAdapter::<TB> {
                lora_a: Tensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device),
                lora_b: Tensor::from_floats(
                    [[0.5, -0.5], [1.5, 2.5], [0.0, 1.0], [-1.0, 0.25]],
                    &device,
                ),
                scale,
            },
        );
        adapters.insert(
            "layers.0.mlp.down_proj".to_string(),
            LoraAdapter::<TB> {
                lora_a: Tensor::from_floats([[7.0, 8.0], [9.0, 10.0]], &device),
                lora_b: Tensor::from_floats([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], &device),
                scale,
            },
        );
        let lora = LoraWeights::<TB> {
            adapters,
            rank,
            alpha,
        };

        let dir = std::env::temp_dir().join(format!("mary_lora_pile_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pile_path = dir.join("lora.pile");
        let _ = std::fs::remove_file(&pile_path);
        std::fs::File::create(&pile_path).unwrap();

        // Save: adapter entities plus set entity in Mary's model collection.
        let root = {
            let mut pile = Pile::open(&pile_path).unwrap();
            pile.refresh().unwrap();
            let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
            let base = fucid().id;
            let fragment = lora
                .save_to_pile("gemma-4-E4B-it", Some(base), &mut pile)
                .unwrap();
            let root = fragment.root().unwrap();
            assert!(exists!(
                (),
                pattern!(fragment.facts(), [{
                    root @ crate::format::attrs::parent: base,
                }])
            ));
            let members: std::collections::BTreeSet<Id> = find!(
                member: Id,
                pattern!(fragment.facts(), [{ root @ crate::format::attrs::member: ?member }])
            )
            .collect();
            assert_eq!(members.len(), lora.adapters.len());
            let relabelled = lora.save_to_pile("renamed", None, &mut pile).unwrap();
            assert_eq!(fragment.root(), relabelled.root());
            crate::model_collection::publish_model_fragment(&mut pile, &signing_key, fragment)
                .unwrap();
            pile.close().unwrap();
            root
        };

        // Load from a FRESH open of the pile file.
        let loaded = {
            let source = crate::persist::read_model_pile(&pile_path).unwrap();
            LoraWeights::<TB>::load_from_pile(&source.facts, &source.store, root, &device)
        };

        assert_eq!(loaded.rank, rank);
        assert_eq!(loaded.alpha, alpha);
        assert_eq!(loaded.adapters.len(), lora.adapters.len());
        for (key, orig) in &lora.adapters {
            let got = loaded
                .get(key)
                .unwrap_or_else(|| panic!("missing adapter {key}"));
            assert_eq!(got.lora_a.dims(), orig.lora_a.dims());
            assert_eq!(got.lora_b.dims(), orig.lora_b.dims());
            assert_eq!(tensor_vec(&got.lora_a), tensor_vec(&orig.lora_a));
            assert_eq!(tensor_vec(&got.lora_b), tensor_vec(&orig.lora_b));
            assert_eq!(got.scale, scale);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Safetensors round-trip: save to a temp file, load back, bit-identical
    /// tensors and preserved rank/alpha metadata. (Import-only: safetensors is
    /// the training-export format, never a runtime load path.)
    #[cfg(feature = "import")]
    #[test]
    fn lora_safetensors_roundtrip() {
        let device = burn::prelude::Device::<TB>::default();
        let rank = 2;
        let alpha = 4.0f32;
        let mut adapters = HashMap::new();
        adapters.insert(
            "layers.1.self_attn.o_proj".to_string(),
            LoraAdapter::<TB> {
                lora_a: Tensor::from_floats([[1.5, -2.5, 0.25], [0.0, 3.0, -1.0]], &device),
                lora_b: Tensor::from_floats([[0.5, -0.5], [1.5, 2.5]], &device),
                scale: alpha / rank as f32,
            },
        );
        let lora = LoraWeights::<TB> {
            adapters,
            rank,
            alpha,
        };

        let path = std::env::temp_dir().join(format!(
            "mary_lora_st_test_{}.safetensors",
            std::process::id()
        ));
        lora.save(&path);
        let loaded = LoraWeights::<TB>::load(&path, &device);
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.rank, rank);
        assert_eq!(loaded.alpha, alpha);
        let orig = &lora.adapters["layers.1.self_attn.o_proj"];
        let got = loaded
            .get("layers.1.self_attn.o_proj")
            .expect("adapter present");
        assert_eq!(tensor_vec(&got.lora_a), tensor_vec(&orig.lora_a));
        assert_eq!(tensor_vec(&got.lora_b), tensor_vec(&orig.lora_b));
        assert_eq!(got.scale, orig.scale);
    }

    /// E4B-shaped init: kv-shared layers carry no k/v adapters; the rest carry
    /// all seven projections; shapes follow the layer type.
    #[test]
    fn lora_init_gemma4_shapes() {
        let device = burn::prelude::Device::<TB>::default();
        // A miniature config with the E4B *structure*: 4 layers, last 2 kv-shared,
        // layer 1 full-attention with a doubled global head_dim.
        let config: Gemma4TextConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "intermediate_size": 32,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25}
            },
            "global_head_dim": 8,
            "num_kv_shared_layers": 2
        }))
        .unwrap();

        let lora = LoraWeights::<TB>::init_gemma4(&config, 2, 4.0, &device);
        // Layers 0,1: 7 adapters each; layers 2,3 (kv-shared): 5 each.
        assert_eq!(lora.adapters.len(), 7 * 2 + 5 * 2);
        assert!(lora.get("layers.2.self_attn.k_proj").is_none());
        assert!(lora.get("layers.3.self_attn.v_proj").is_none());
        // Sliding layer 0: q_dim = 4*4 = 16, kv_dim = 2*4 = 8.
        assert_eq!(
            lora.get("layers.0.self_attn.q_proj").unwrap().lora_b.dims(),
            [16, 2]
        );
        assert_eq!(
            lora.get("layers.0.self_attn.k_proj").unwrap().lora_b.dims(),
            [8, 2]
        );
        // Full layer 1: q_dim = 4*8 = 32, kv_dim = 2*8 = 16.
        assert_eq!(
            lora.get("layers.1.self_attn.q_proj").unwrap().lora_b.dims(),
            [32, 2]
        );
        assert_eq!(
            lora.get("layers.1.self_attn.v_proj").unwrap().lora_b.dims(),
            [16, 2]
        );
        assert_eq!(
            lora.get("layers.1.self_attn.o_proj").unwrap().lora_a.dims(),
            [2, 32]
        );
        // MLP shapes are layer-type independent.
        assert_eq!(
            lora.get("layers.3.mlp.down_proj").unwrap().lora_b.dims(),
            [16, 2]
        );

        // K=V (dense 12B/31B structure): full-attention layers drop the
        // k_proj adapter (k is defined as v; k_proj never runs) but keep
        // v_proj; sliding layers keep all seven.
        let config_keqv: Gemma4TextConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "intermediate_size": 32,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25}
            },
            "global_head_dim": 8,
            "num_global_key_value_heads": 1,
            "attention_k_eq_v": true
        }))
        .unwrap();
        let lora_keqv = LoraWeights::<TB>::init_gemma4(&config_keqv, 2, 4.0, &device);
        assert_eq!(lora_keqv.adapters.len(), 7 + 6);
        assert!(lora_keqv.get("layers.0.self_attn.k_proj").is_some());
        assert!(lora_keqv.get("layers.1.self_attn.k_proj").is_none());
        // Full layer 1: kv_dim = 1 * 8 = 8.
        assert_eq!(
            lora_keqv
                .get("layers.1.self_attn.v_proj")
                .unwrap()
                .lora_b
                .dims(),
            [8, 2]
        );

        // maybe_lora: zero-init B ⇒ adapted forward == base forward.
        let linear = burn::nn::LinearConfig::new(16, 16)
            .with_bias(false)
            .init(&device);
        let lora_fresh = LoraWeights::<TB>::init_gemma4(&config, 2, 4.0, &device);
        let x = Tensor::<TB, 3>::random([1, 3, 16], burn::tensor::Distribution::Default, &device);
        let base = linear.forward(x.clone());
        let adapted = maybe_lora(
            &linear,
            x.clone(),
            Some(&lora_fresh),
            "layers.0.self_attn.q_proj",
        );
        let none = maybe_lora(&linear, x, None, "layers.0.self_attn.q_proj");
        assert_eq!(tensor_vec(&base), tensor_vec(&adapted));
        assert_eq!(tensor_vec(&base), tensor_vec(&none));
    }
}
