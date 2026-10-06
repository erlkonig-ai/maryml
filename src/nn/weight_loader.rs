//! Weight loading. The RUNTIME path is [`WeightLoader::Pile`] — a keymap
//! materialized from a content-addressed pile (see [`crate::persist`]). The
//! safetensors readers ([`SingleFileLoader`], [`MultiShardLoader`], and the
//! free helpers) exist only under the `import` feature, for the persist
//! importers and parity gates: an inference/serve build cannot compile a
//! safetensors path at all.

use burn::prelude::*;
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "import")]
use burn::tensor::{ElementConversion, TensorData};
#[cfg(feature = "import")]
use half::{bf16, f16};
#[cfg(feature = "import")]
use safetensors::SafeTensors;
#[cfg(feature = "import")]
use std::fs;

/// Load a safetensors file from disk, returning the raw bytes.
#[cfg(feature = "import")]
pub fn read_safetensors_file(path: &Path) -> Vec<u8> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("Failed to read safetensors file {}: {}", path.display(), e))
}

/// Extract a named tensor from a SafeTensors container, converting to f32.
/// Returns (data, shape).
#[cfg(feature = "import")]
pub fn get_tensor_f32(st: &SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("Missing tensor '{}': {}", name, e));
    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();

    let floats: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F64 => data
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        dtype => panic!("Unsupported dtype: {:?}", dtype),
    };

    (floats, shape)
}

/// Load a tensor from a SafeTensors container as a Burn Tensor<B, D>.
///
/// Converts the mmap'd on-disk bytes DIRECTLY into the backend's element type
/// (`B::FloatElem`) — no intermediate `Vec<f32>`. For an f16 device-tensor
/// backend (`Metal<f16>`) the bf16→f32→f16 rounding happens per-scalar in a
/// register and only a `Vec<f16>` (== the final size) is allocated, instead of
/// the old route's `Vec<f32>` (4 B/elem) plus a second convert buffer inside
/// `from_data`. The numeric result is bit-identical to the old path (same
/// bf16→f32→f16 double-rounding). For an f32 backend (F5 / Vocos run on
/// `Metal`) `B::FloatElem = f32` and this is exactly the previous behavior.
#[cfg(feature = "import")]
pub fn load_tensor<B: Backend, const D: usize>(
    st: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Tensor<B, D> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("Missing tensor '{}': {}", name, e));
    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();

    let elems: Vec<B::FloatElem> = match view.dtype() {
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32().elem())
            .collect(),
        safetensors::Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32().elem())
            .collect(),
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).elem())
            .collect(),
        dtype => panic!("Unsupported dtype: {:?}", dtype),
    };

    Tensor::from_data(TensorData::new(elems, shape), device)
}

/// Load a 1D tensor.
#[cfg(feature = "import")]
pub fn load_tensor_1d<B: Backend>(
    st: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 1> {
    load_tensor::<B, 1>(st, name, device)
}

/// Load a 2D tensor.
#[cfg(feature = "import")]
pub fn load_tensor_2d<B: Backend>(
    st: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 2> {
    load_tensor::<B, 2>(st, name, device)
}

/// Load a 4D tensor (for conv weights).
#[cfg(feature = "import")]
pub fn load_tensor_4d<B: Backend>(
    st: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 4> {
    load_tensor::<B, 4>(st, name, device)
}

/// Holds opened multi-shard safetensors data.
/// Keeps the raw bytes alive so that SafeTensors references remain valid.
#[cfg(feature = "import")]
pub struct MultiShardLoader {
    /// Map from weight name -> shard index
    pub(crate) weight_map: HashMap<String, usize>,
    /// Raw bytes for each shard file (kept alive for SafeTensors lifetime)
    pub(crate) shard_bytes: Vec<Vec<u8>>,
}

#[cfg(feature = "import")]
impl MultiShardLoader {
    /// Create from a directory containing model.safetensors.index.json.
    pub fn new(dir: &Path) -> Self {
        let index_path = dir.join("model.safetensors.index.json");
        let index_str = fs::read_to_string(&index_path)
            .unwrap_or_else(|e| panic!("Failed to read index {}: {}", index_path.display(), e));
        let index: serde_json::Value = serde_json::from_str(&index_str).unwrap();
        let wm = index["weight_map"].as_object().unwrap();

        // Collect unique shard filenames
        let mut shard_files: Vec<String> = Vec::new();
        let mut shard_index_map: HashMap<String, usize> = HashMap::new();

        let mut weight_map = HashMap::new();
        for (key, shard_name) in wm {
            let shard = shard_name.as_str().unwrap().to_string();
            let idx = if let Some(&idx) = shard_index_map.get(&shard) {
                idx
            } else {
                let idx = shard_files.len();
                shard_files.push(shard.clone());
                shard_index_map.insert(shard, idx);
                idx
            };
            weight_map.insert(key.clone(), idx);
        }

        // Load all shard files
        let shard_bytes: Vec<Vec<u8>> = shard_files
            .iter()
            .map(|f| read_safetensors_file(&dir.join(f)))
            .collect();

        Self {
            weight_map,
            shard_bytes,
        }
    }

    /// Load a named tensor from the correct shard.
    pub fn load_tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Tensor<B, D> {
        let &shard_idx = self
            .weight_map
            .get(name)
            .unwrap_or_else(|| panic!("Weight '{}' not found in index", name));
        let st = SafeTensors::deserialize(&self.shard_bytes[shard_idx]).unwrap();
        load_tensor::<B, D>(&st, name, device)
    }

    /// Check if a weight name exists.
    pub fn has_weight(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }
}

/// Simple single-file loader that keeps bytes alive.
#[cfg(feature = "import")]
pub struct SingleFileLoader {
    pub(crate) bytes: Vec<u8>,
}

#[cfg(feature = "import")]
impl SingleFileLoader {
    pub fn new(path: &Path) -> Self {
        Self {
            bytes: read_safetensors_file(path),
        }
    }

    pub fn load_tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Tensor<B, D> {
        let st = SafeTensors::deserialize(&self.bytes).unwrap();
        load_tensor::<B, D>(&st, name, device)
    }

    /// Get all tensor names in this file.
    pub fn tensor_names(&self) -> Vec<String> {
        let st = SafeTensors::deserialize(&self.bytes).unwrap();
        st.names().into_iter().map(|s| s.to_string()).collect()
    }
}

/// Host-resident f32 tensor data: an OWNED buffer (the materializing load
/// path) or a zero-copy mmap VIEW of the pile blob (the zero-copy load path —
/// the `View` keeps the pile mmap alive for the model's life, and first
/// access pages the bytes in lazily). Derefs to `[f32]` either way, so
/// consumers (embedding row lookups, CPU gemv operands) are
/// storage-agnostic.
pub enum HostF32 {
    Owned(Vec<f32>),
    Mapped(anybytes::View<[f32]>),
}

impl std::ops::Deref for HostF32 {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            HostF32::Owned(v) => v,
            HostF32::Mapped(v) => v,
        }
    }
}

/// Unified weight loader. At runtime this is always [`WeightLoader::Pile`] or
/// (on macOS, for the fused qwen3tts/voxtral backends) [`WeightLoader::Aliased`];
/// the safetensors variants exist only under `import` for the importers and
/// parity gates.
pub enum WeightLoader {
    /// One explicitly selected exact-F32 model in an immutable observation.
    /// Requests query the stored root/name/typed-leaf relation at point of use;
    /// there is no retained key catalogue or union across unrelated models.
    Selected {
        snapshot: crate::model_collection::ModelPileSnapshot,
        root: triblespace::prelude::Id,
    },
    #[cfg(feature = "import")]
    SingleFile(SingleFileLoader),
    #[cfg(feature = "import")]
    MultiShard(MultiShardLoader),
    /// Model materialized out of a mary pile: key → (flat data, shape).
    Pile(HashMap<String, (Vec<f32>, Vec<usize>)>),
    /// Model read from a pile of TYPED tensor leaves: key → a view over the
    /// pile's mapping, not a copy.
    ///
    /// The general zero-copy path. [`WeightLoader::Aliased`] below is the older
    /// one and is gated to macOS AND to two model features, so everywhere else
    /// every load materialized. A typed leaf carries its shape in the blob
    /// header, so serving an aligned `[f32]` view is a slice — no platform
    /// check, no per-model feature, no alias preconditions to fail.
    Typed(HashMap<String, crate::leaf::Leaf>),
    /// ZERO-COPY pile loader: tensor requests on the fused Metal backends alias
    /// the mmap'd pile blobs straight onto the GPU (no host materialization);
    /// everything else (CPU stages via [`WeightLoader::load_f32`], non-fused
    /// backends) materializes lazily from the same handle index.
    #[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
    Aliased(AliasedPile),
}

/// The state behind [`WeightLoader::Aliased`]: cheap handle indexes (no tensor
/// data) for the half-width and exact leaf families, plus the pile readers
/// whose mmaps the aliased tensors bind. Native Qwen3-TTS and Voxtral cohorts
/// pass clones of one shared snapshot reader; the two-reader low-level shape
/// also supports callers that intentionally compose independent stores.
/// Requests on `BFusedHalf` upload `f16` leaves at native width; requests on
/// `BFused` upload `f32` leaves; requests on the RAW (unfused) backends
/// (`BHalf`/`B`) alias the matching-width leaves' mmap'd pile pages straight
/// onto the GPU — TRUE zero-copy (see [`crate::nn::alias::alias_flat_raw`]).
/// Anything else — and any leaf whose alias preconditions fail (logged) —
/// materializes through [`crate::ingest::read_leaf`].
#[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
pub struct AliasedPile {
    f16: HashMap<String, crate::leaf::Leaf>,
    f32: HashMap<String, crate::leaf::Leaf>,
    device: burn::backend::wgpu::WgpuDevice,
}

#[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
impl AliasedPile {
    /// The two indexes and the device they load onto.
    ///
    /// No reader: a leaf's payload is `Bytes` over the pile's mapping and keeps
    /// that mapping alive by itself, so the index outlives the repository it
    /// came from without holding a handle to it.
    pub fn new(
        f16: HashMap<String, crate::leaf::Leaf>,
        f32: HashMap<String, crate::leaf::Leaf>,
        device: burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        Self { f16, f32, device }
    }

    /// Number of half-width / exact leaves indexed.
    pub fn counts(&self) -> (usize, usize) {
        (self.f16.len(), self.f32.len())
    }

    /// Fast GPU load for the Metal backends without the f32 keymap
    /// materialization: `BFusedHalf` ← the f16 leaf uploaded at native width,
    /// `BFused` ← the f32 leaf uploaded, and the RAW backends (`BHalf`/`B`,
    /// the speak talker lane) ← the matching-width leaf aliased
    /// ZERO-COPY straight from the mmap'd pile pages
    /// ([`crate::nn::alias::alias_flat_raw`], the gemma seam — upload
    /// fallback if a precondition fails, logged). `None` = not applicable
    /// (other backend / no matching-width leaf) so the caller materializes
    /// from the exact leaf.
    ///
    /// The fused backends must NOT alias: burn 0.21's fusion codegen
    /// miscompiles graphs over many externally-registered buffers (cos≈0.41
    /// end-to-end talker; the fusion-import route was probed and DELETED at
    /// bf92171 — see PORT_NOTES.md "Zero-copy alias probe"). The raw
    /// backends have no fusion graph; gemma runs the same seam at 31B scale.
    fn gpu_tensor<B: Backend, const D: usize>(&self, name: &str) -> Option<Tensor<B, D>> {
        use crate::leaf::Elem;
        use crate::nn::backend::{BFused, BFusedHalf};
        use burn::tensor::TensorData;
        use std::any::TypeId;

        let want_f16 = TypeId::of::<B>() == TypeId::of::<BFusedHalf>();
        let want_f32 = TypeId::of::<B>() == TypeId::of::<BFused>();
        let want_raw_f16 = TypeId::of::<B>() == TypeId::of::<crate::nn::backend::BHalf>();
        let want_raw_f32 = TypeId::of::<B>() == TypeId::of::<crate::nn::backend::B>();

        let dev = &self.device; // concretely WgpuDevice, the device of all four backends
        // Each arm picks its index, insists on the width that index is for,
        // and takes the leaf's own bytes. There is no shape blob to fetch:
        // the dims came out of the tensor header the payload is attached to.
        let (flat, shape): (Tensor<B, 1>, Vec<usize>) = if want_f16 {
            let leaf = self.f16.get(name)?;
            let bytes = (leaf.elem() == Elem::F16).then(|| leaf.payload().clone())?;
            let concrete: Tensor<BFusedHalf, 1> = upload_f16::<BFusedHalf>(bytes, dev);
            (same_type::<_, Tensor<B, 1>>(concrete), leaf.shape())
        } else if want_raw_f16 {
            let leaf = self.f16.get(name)?;
            let bytes = (leaf.elem() == Elem::F16).then(|| leaf.payload().clone())?;
            let concrete: Tensor<crate::nn::backend::BHalf, 1> =
                match crate::nn::alias::alias_flat_raw::<half::f16>(bytes.clone(), dev) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[mary] {name}: zero-copy alias failed ({e}); uploading");
                        upload_f16::<crate::nn::backend::BHalf>(bytes, dev)
                    }
                };
            (same_type::<_, Tensor<B, 1>>(concrete), leaf.shape())
        } else if want_raw_f32 {
            let leaf = self.f32.get(name)?;
            let bytes = (leaf.elem() == Elem::F32).then(|| leaf.payload().clone())?;
            let concrete: Tensor<crate::nn::backend::B, 1> =
                match crate::nn::alias::alias_flat_raw::<f32>(bytes, dev) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[mary] {name}: zero-copy alias failed ({e}); uploading");
                        let data = leaf.to_f32();
                        let n = data.len();
                        Tensor::from_data(TensorData::new(data, [n]), dev)
                    }
                };
            (same_type::<_, Tensor<B, 1>>(concrete), leaf.shape())
        } else if want_f32 {
            let leaf = self.f32.get(name)?;
            if leaf.elem() != Elem::F32 {
                return None;
            }
            let data = leaf.to_f32();
            let n = data.len();
            let concrete: Tensor<BFused, 1> =
                Tensor::<BFused, 1>::from_data(TensorData::new(data, [n]), dev);
            (same_type::<_, Tensor<B, 1>>(concrete), leaf.shape())
        } else {
            return None;
        };
        assert_eq!(shape.len(), D, "rank mismatch for {name}");
        let mut dims = [0usize; D];
        dims[..D].copy_from_slice(&shape[..D]);
        Some(flat.reshape(dims))
    }

    /// The leaf for `name`, exact leaves preferred (for the materializing
    /// fallback and the CPU stages).
    fn leaf(&self, name: &str) -> Option<&crate::leaf::Leaf> {
        self.f32.get(name).or_else(|| self.f16.get(name))
    }
}

/// Identity "cast" between two monomorphizations the caller has proven to be
/// the SAME type (`TypeId` equality) — how the generic `load_tensor` hands out
/// a concretely-typed aliased tensor (and how the speak lane hands out a
/// concretely-typed folded talker).
#[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
pub(crate) fn same_type<T: 'static, U: 'static>(t: T) -> U {
    let any: Box<dyn std::any::Any> = Box::new(t);
    *any.downcast::<U>()
        .expect("same_type called across distinct types")
}

/// Upload f16 leaf bytes to `B` (an f16 backend) DIRECTLY — no f32
/// materialization, no cast loop. The pile stores the exact f16 the talker
/// runs at (`f16::from_f32(bf16→f32)`), so `B::FloatElem = f16` means this is a
/// straight `bytemuck` view → device buffer: half the bytes of the old f32
/// path and bit-identical to the materialized+cast tensor.
#[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
fn upload_f16<B: Backend>(bytes: anybytes::Bytes, device: &B::Device) -> Tensor<B, 1> {
    use burn::tensor::TensorData;
    let halves: Vec<half::f16> = bytes.view::<[half::f16]>().expect("f16 view")[..].to_vec();
    let n = halves.len();
    // TensorData carries the dtype; for an f16 backend this is the native path
    // (no element cast), for f32 it up-converts on upload — either way no host
    // f32 buffer is built first.
    Tensor::<B, 1>::from_data(TensorData::new(halves, [n]), device)
}

impl WeightLoader {
    pub fn selected(
        snapshot: crate::model_collection::ModelPileSnapshot,
        root: triblespace::prelude::Id,
    ) -> Self {
        Self::Selected { snapshot, root }
    }

    /// Auto-detect and create from a directory.
    /// Looks for `diffusion_pytorch_model.safetensors.index.json` (multi-shard)
    /// or `diffusion_pytorch_model.safetensors` (single-file).
    #[cfg(feature = "import")]
    pub fn from_dir(dir: &Path) -> Self {
        let index_path = dir.join("diffusion_pytorch_model.safetensors.index.json");
        if index_path.exists() {
            // Multi-shard: reuse MultiShardLoader but with diffusion_pytorch_model prefix
            let index_str = fs::read_to_string(&index_path)
                .unwrap_or_else(|e| panic!("Failed to read index {}: {}", index_path.display(), e));
            let index: serde_json::Value = serde_json::from_str(&index_str).unwrap();
            let wm = index["weight_map"].as_object().unwrap();

            let mut shard_files: Vec<String> = Vec::new();
            let mut shard_index_map: HashMap<String, usize> = HashMap::new();
            let mut weight_map = HashMap::new();

            for (key, shard_name) in wm {
                let shard = shard_name.as_str().unwrap().to_string();
                let idx = if let Some(&idx) = shard_index_map.get(&shard) {
                    idx
                } else {
                    let idx = shard_files.len();
                    shard_files.push(shard.clone());
                    shard_index_map.insert(shard, idx);
                    idx
                };
                weight_map.insert(key.clone(), idx);
            }

            let shard_bytes: Vec<Vec<u8>> = shard_files
                .iter()
                .map(|f| read_safetensors_file(&dir.join(f)))
                .collect();

            WeightLoader::MultiShard(MultiShardLoader {
                weight_map,
                shard_bytes,
            })
        } else {
            let single_path = dir.join("diffusion_pytorch_model.safetensors");
            WeightLoader::SingleFile(SingleFileLoader::new(&single_path))
        }
    }

    /// Build the runtime loader from a persisted pile: materialize the union
    /// keymap (see [`crate::persist::load_keymap_from_pile`]).
    /// Build the runtime loader from a persisted pile, preferring the typed
    /// layout.
    ///
    /// One constructor, so every caller gets zero-copy loading the moment its
    /// pile is converted, and keeps working when it is not. The fallback is on
    /// EMPTY rather than on error: a pile with no typed leaves is an
    /// unconverted pile, not a broken one.
    pub fn from_pile(pile_path: &Path) -> anyhow::Result<Self> {
        match crate::persist::load_typed_keymap_from_pile(pile_path) {
            Ok(map) if !map.is_empty() => return Ok(WeightLoader::Typed(map)),
            Ok(_) => {}
            Err(e) => eprintln!("[weights] typed index unavailable ({e}); materializing"),
        }
        Ok(WeightLoader::Pile(crate::persist::load_keymap_from_pile(
            pile_path,
        )?))
    }

    pub fn load_tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Tensor<B, D> {
        match self {
            WeightLoader::Selected { snapshot, root } => {
                let (data, shape) = selected_f32_rank::<D>(snapshot, *root, name)
                    .unwrap_or_else(|e| panic!("selected model tensor {name}: {e}"))
                    .unwrap_or_else(|| panic!("selected model lacks F32 rank-{D} tensor {name}"));
                let dims: [usize; D] = shape.try_into().expect("typed rank");
                Tensor::<B, 1>::from_floats(&data[..], device).reshape(dims)
            }
            #[cfg(feature = "import")]
            WeightLoader::SingleFile(loader) => loader.load_tensor(name, device),
            #[cfg(feature = "import")]
            WeightLoader::MultiShard(loader) => loader.load_tensor(name, device),
            WeightLoader::Pile(map) => {
                let (data, shape) = map
                    .get(name)
                    .unwrap_or_else(|| panic!("pile missing tensor {name}"));
                assert_eq!(shape.len(), D, "rank mismatch for {name}");
                let mut dims = [0usize; D];
                dims[..D].copy_from_slice(&shape[..D]);
                Tensor::<B, 1>::from_floats(&data[..], device).reshape(dims)
            }
            WeightLoader::Typed(map) => {
                let leaf = map
                    .get(name)
                    .unwrap_or_else(|| panic!("pile missing tensor {name}"));
                let shape = leaf.shape();
                assert_eq!(shape.len(), D, "rank mismatch for {name}");
                let mut dims = [0usize; D];
                dims[..D].copy_from_slice(&shape[..D]);
                // A copy still happens here because Burn wants to own the
                // tensor; what is avoided is the SECOND copy the materialized
                // keymap makes when it builds the map up front.
                Tensor::<B, 1>::from_floats(&leaf.to_f32()[..], device).reshape(dims)
            }
            #[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
            WeightLoader::Aliased(pile) => {
                if let Some(t) = pile.gpu_tensor::<B, D>(name) {
                    return t;
                }
                let leaf = pile
                    .leaf(name)
                    .unwrap_or_else(|| panic!("pile missing tensor {name}"));
                let (data, shape) = leaf.to_f32_shape();
                assert_eq!(shape.len(), D, "rank mismatch for {name}");
                let mut dims = [0usize; D];
                dims[..D].copy_from_slice(&shape[..D]);
                Tensor::<B, 1>::from_floats(&data[..], device).reshape(dims)
            }
        }
    }

    /// Load a tensor as raw host f32 (data, shape) — for stages that run on
    /// the CPU (Accelerate) instead of a Burn backend.
    pub fn load_f32(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        match self {
            WeightLoader::Selected { snapshot, root } => {
                let (data, shape) = selected_f32(snapshot, *root, name)
                    .unwrap_or_else(|e| panic!("selected model tensor {name}: {e}"))
                    .unwrap_or_else(|| panic!("selected model lacks F32 tensor {name}"));
                (data.to_vec(), shape)
            }
            #[cfg(feature = "import")]
            WeightLoader::SingleFile(loader) => {
                let st = SafeTensors::deserialize(&loader.bytes).unwrap();
                get_tensor_f32(&st, name)
            }
            #[cfg(feature = "import")]
            WeightLoader::MultiShard(loader) => {
                let &shard_idx = loader
                    .weight_map
                    .get(name)
                    .unwrap_or_else(|| panic!("Weight '{}' not found in index", name));
                let st = SafeTensors::deserialize(&loader.shard_bytes[shard_idx]).unwrap();
                get_tensor_f32(&st, name)
            }
            WeightLoader::Pile(map) => map
                .get(name)
                .unwrap_or_else(|| panic!("pile missing tensor {name}"))
                .clone(),
            WeightLoader::Typed(map) => {
                let leaf = map
                    .get(name)
                    .unwrap_or_else(|| panic!("pile missing tensor {name}"));
                (leaf.to_f32(), leaf.shape())
            }
            #[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
            WeightLoader::Aliased(pile) => {
                let leaf = pile
                    .leaf(name)
                    .unwrap_or_else(|| panic!("pile missing tensor {name}"));
                leaf.to_f32_shape()
            }
        }
    }

    /// ZERO-COPY mmap view of an exact-f32 leaf plus its shape — `None` when
    /// this loader (or this leaf) can't serve one (safetensors/materialized
    /// loaders, f16 leaves), in which case the caller falls back to
    /// [`Self::load_f32`] (an owned copy, same bytes).
    pub fn view_f32(&self, name: &str) -> Option<(anybytes::View<[f32]>, Vec<usize>)> {
        match self {
            WeightLoader::Selected { snapshot, root } => selected_f32(snapshot, *root, name)
                .unwrap_or_else(|e| panic!("selected model tensor {name}: {e}")),
            // Serves on every platform, for every model — see the variant docs.
            WeightLoader::Typed(map) => {
                let leaf = map.get(name)?;
                Some((leaf.view_f32()?, leaf.shape()))
            }
            #[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
            WeightLoader::Aliased(pile) => {
                let leaf = pile.leaf(name)?;
                Some((leaf.view_f32()?, leaf.shape()))
            }
            _ => None,
        }
    }

    /// [`Self::load_f32`] with the zero-copy view preferred: a mapped
    /// [`HostF32`] when the pile can alias the bytes, an owned copy
    /// otherwise. Same values either way.
    pub fn load_host_f32(&self, name: &str) -> (HostF32, Vec<usize>) {
        if let Some((v, shape)) = self.view_f32(name) {
            return (HostF32::Mapped(v), shape);
        }
        let (data, shape) = self.load_f32(name);
        (HostF32::Owned(data), shape)
    }

    pub fn has_weight(&self, name: &str) -> bool {
        match self {
            WeightLoader::Selected { snapshot, root } => selected_f32(snapshot, *root, name)
                .unwrap_or_else(|e| panic!("selected model tensor {name}: {e}"))
                .is_some(),
            #[cfg(feature = "import")]
            WeightLoader::SingleFile(loader) => loader.tensor_names().iter().any(|n| n == name),
            #[cfg(feature = "import")]
            WeightLoader::MultiShard(loader) => loader.has_weight(name),
            WeightLoader::Pile(map) => map.contains_key(name),
            WeightLoader::Typed(map) => map.contains_key(name),
            #[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
            WeightLoader::Aliased(pile) => pile.leaf(name).is_some(),
        }
    }
}

fn selected_f32_rank<const R: usize>(
    snapshot: &crate::model_collection::ModelPileSnapshot,
    root: triblespace::prelude::Id,
    name: &str,
) -> anyhow::Result<Option<(anybytes::View<[f32]>, Vec<usize>)>> {
    use triblespace::core::blob::{
        Blob,
        encodings::tensor::{Tensor as NativeTensor, elements::F32},
    };
    use triblespace::prelude::*;
    let path: Blob<blobencodings::UTF8String> = name.to_owned().to_blob();
    let path = path.get_handle();
    for (handle,) in find!((handle: Inline<inlineencodings::Handle<NativeTensor<F32, R>>>),
    pattern!(snapshot.facts(), [
        { root @ crate::format::attrs::member: _?member },
        { _?member @ crate::format::attrs::safetensor_path: path, crate::format::attrs::weight: _?weight },
        { _?weight @ crate::leaf::leaf::<F32, R>(): ?handle }
    ])) {
        let blob: Blob<NativeTensor<F32, R>> = snapshot
            .store()
            .get(handle)
            .map_err(|e| anyhow::anyhow!("{name}: read selected F32 leaf: {e}"))?;
        let leaf = crate::leaf::read_leaf(blob)?;
        let shape = leaf
            .dims()
            .iter()
            .map(|&d| usize::try_from(d))
            .collect::<Result<Vec<_>, _>>()?;
        let view = leaf
            .payload()
            .clone()
            .view::<[f32]>()
            .map_err(|e| anyhow::anyhow!("{name}: F32 view: {e}"))?;
        return Ok(Some((view, shape)));
    }
    Ok(None)
}

fn selected_f32(
    snapshot: &crate::model_collection::ModelPileSnapshot,
    root: triblespace::prelude::Id,
    name: &str,
) -> anyhow::Result<Option<(anybytes::View<[f32]>, Vec<usize>)>> {
    // CPU codec callers ask for raw host values rather than a compile-time
    // rank. Probe only supported native F32 types; unrelated facts coexist.
    macro_rules! rank {
        ($r:literal) => {
            if let Some(value) = selected_f32_rank::<$r>(snapshot, root, name)? {
                return Ok(Some(value));
            }
        };
    }
    rank!(0);
    rank!(1);
    rank!(2);
    rank!(3);
    rank!(4);
    rank!(5);
    rank!(6);
    Ok(None)
}

#[cfg(all(test, feature = "import"))]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView, serialize};

    #[test]
    fn f64_is_widened_through_the_supported_float_path() {
        let source = [1.25_f64, -2.5];
        let raw: Vec<u8> = source
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let tensor = TensorView::new(Dtype::F64, vec![2], &raw).unwrap();
        let bytes = serialize([("wide.weight", tensor)], &None).unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();

        assert_eq!(
            get_tensor_f32(&tensors, "wide.weight"),
            (vec![1.25_f32, -2.5], vec![2])
        );
    }
}
