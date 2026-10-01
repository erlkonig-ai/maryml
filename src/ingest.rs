//! Bridge between weight files and the `mary::format` graph. Ingest tensors into
//! the pile — each a content-addressed [`crate::leaf`] inside a named module,
//! gathered under a model entity — and index them back by name for the
//! `WeightLoader` reconstruction paths. Generic over model AND over
//! source format: safetensors decode here ([`ingest_members`]); GGUF and pytorch
//! pickle decode in [`crate::formats`] and feed the same format-agnostic core
//! ([`ingest_tensors`]), so every format lands in one content-addressed graph.

use crate::format::attrs;
#[cfg(feature = "import")]
use crate::format::{put_raw, put_raw_bf16, put_raw_f16};
#[cfg(feature = "import")]
use crate::nn::weight_loader::get_tensor_f32;
#[cfg(feature = "import")]
use safetensors::SafeTensors;
use std::collections::HashMap;
use triblespace::prelude::blobencodings::UTF8String;
use triblespace::prelude::*;

#[cfg(feature = "import")]
type Err = Box<dyn std::error::Error>;

/// Storage element format. `F32` and `F16` use the legacy conversion importer.
/// `Bf16` requires native BF16 safetensors and preserves their exact bytes;
/// it never converts BF16 through IEEE f16's narrower exponent range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafDtype {
    F32,
    F16,
    Bf16,
}

/// Ingest a single-file safetensors blob into the pile as a model graph. Returns
/// the model Fragment (rooted at the model entity), carrying all module + leaf facts.
/// Import-only: this is the safetensors → pile direction.
#[cfg(feature = "import")]
pub fn save_safetensors(
    bytes: &[u8],
    model_name: &str,
    blobs: &mut impl BlobStorePut,
    dtype: LeafDtype,
) -> Result<Fragment, Err> {
    save_safetensors_filtered(bytes, model_name, blobs, dtype, |_| true)
}

/// [`save_safetensors`] restricted to the tensors whose name passes `keep` —
/// the way to persist ONE component of a multi-component checkpoint under its own
/// model entity (e.g. the qwen3tts talker as a half-width `talker_f16` variant
/// next to the exact f32 leaves).
///
/// The root's identity core is its member links, just as in
/// [`build_model_root`]. `model_name` is a non-core label. This affects new
/// imports only: readers keep addressing previously stored per-file roots by
/// their opaque IDs, including roots whose old writer included the name.
#[cfg(feature = "import")]
pub fn save_safetensors_filtered(
    bytes: &[u8],
    model_name: &str,
    blobs: &mut impl BlobStorePut,
    dtype: LeafDtype,
    keep: impl Fn(&str) -> bool,
) -> Result<Fragment, Err> {
    let (members, facts) = ingest_members(bytes, blobs, dtype, keep)?;
    named_file_root(blobs, model_name, members, facts)
}

/// The shared-bytes counterpart to [`save_safetensors_filtered`]. Native BF16
/// payloads borrow the source backing through [`ingest_bf16_members`], avoiding
/// the borrowed-slice interface's copy. Other storage dtypes keep their existing
/// conversion behavior. Root identity and file-name annotations are identical
/// for the two input interfaces.
#[cfg(feature = "import")]
pub fn save_safetensors_filtered_bytes(
    bytes: &anybytes::Bytes,
    model_name: &str,
    blobs: &mut impl BlobStorePut,
    dtype: LeafDtype,
    keep: impl Fn(&str) -> bool,
) -> Result<Fragment, Err> {
    let (members, facts) = match dtype {
        LeafDtype::Bf16 => ingest_bf16_members(bytes, blobs, keep)?,
        LeafDtype::F32 | LeafDtype::F16 => ingest_members(bytes, blobs, dtype, keep)?,
    };
    named_file_root(blobs, model_name, members, facts)
}

/// Both file input interfaces publish the same member-only root and label.
#[cfg(feature = "import")]
fn named_file_root(
    blobs: &mut impl BlobStorePut,
    model_name: &str,
    members: Vec<Id>,
    mut facts: TribleSet,
) -> Result<Fragment, Err> {
    let mn = blobs.put::<UTF8String, _>(model_name.to_string())?;
    let model = entity! { _ @ attrs::member*: members.iter() };
    let model_root_id = model.root().expect("model root");
    facts += model.into_facts();
    facts += entity! { ExclusiveId::force_ref(&model_root_id) @ attrs::model_name: mn };
    Ok(Fragment::rooted(model_root_id, facts))
}

/// Ingest one safetensors blob's float tensors (those whose name passes `keep`)
/// into content-addressed member MODULES, returning `(member module ids, facts)`
/// — the shared front half of both the per-file model entity
/// ([`save_safetensors_filtered`]) and the content-addressed model ROOT
/// ([`build_model_root`]). No model/root entity is created here: the caller
/// decides how the members are grouped (per-file, or ONE root composing every
/// shard's members). Each tensor is a typed tensor leaf reached by a
/// `{kind, safetensor_path, weight}` module; identical tensors dedup by content.
///
/// The legacy F32/F16 extractor decodes the container to
/// `(name, f32-data, shape)` tuples and hands them to the format-agnostic
/// [`ingest_tensors`]. Other formats (GGUF, pytorch pickle — see
/// [`crate::formats`]) produce the SAME tuples and reuse `ingest_tensors`, so a
/// model imported from any format lands in the identical content-addressed graph
/// (the member ids, and hence the model-root id, are the pure hash of the
/// `(name, f32-bytes, shape)` set, format-independent). `Bf16` instead stores
/// one tensor's native bytes at a time and rejects selected non-BF16 tensors.
/// With an owned input mapping, prefer [`ingest_bf16_members`] to avoid even
/// the per-tensor input copy required by this borrowed-slice interface.
#[cfg(feature = "import")]
pub fn ingest_members(
    bytes: &[u8],
    blobs: &mut impl BlobStorePut,
    dtype: LeafDtype,
    keep: impl Fn(&str) -> bool,
) -> Result<(Vec<Id>, TribleSet), Err> {
    if dtype == LeafDtype::Bf16 {
        return ingest_bf16_members_with(bytes, blobs, keep, |data| {
            anybytes::Bytes::from_source(data.to_vec())
        });
    }
    let st = SafeTensors::deserialize(bytes)?;
    let tensors = st
        .names()
        .into_iter()
        .filter(|k| keep(k))
        .filter_map(|key| {
            // skip non-float tensors (int buffers etc.) — the forward never loads them
            use safetensors::Dtype;
            let view = st.tensor(key).ok()?;
            if !matches!(
                view.dtype(),
                Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16
            ) {
                return None;
            }
            let (data, shape) = get_tensor_f32(&st, key);
            Some((key.to_string(), data, shape))
        })
        .collect::<Vec<_>>();
    ingest_tensors(tensors.into_iter(), blobs, dtype)
}

/// Import selected native BF16 safetensors from shared bytes (for example an
/// `anybytes::Bytes` backed by an mmap), borrowing one payload at a time.
///
/// No tensor is decoded into floats. The only weight allocation is the
/// header-plus-payload blob currently being written by the tensor encoding.
/// The importer retains member IDs and facts, not a collection of weight
/// buffers. All selected tensors must be BF16; callers can use `keep` to omit
/// integer buffers or other components explicitly. Errors do not roll back
/// blobs already written, but no partial member graph is returned.
#[cfg(feature = "import")]
pub fn ingest_bf16_members(
    bytes: &anybytes::Bytes,
    blobs: &mut impl BlobStorePut,
    keep: impl Fn(&str) -> bool,
) -> Result<(Vec<Id>, TribleSet), Err> {
    ingest_bf16_members_with(bytes, blobs, keep, |data| {
        // SafeTensors returns slices into the same input buffer. Byte offsets
        // let Bytes retain that buffer's owner without copying the payload.
        let start = data.as_ptr() as usize - bytes.as_ptr() as usize;
        bytes.slice(start..start + data.len())
    })
}

#[cfg(feature = "import")]
fn ingest_bf16_members_with(
    bytes: &[u8],
    blobs: &mut impl BlobStorePut,
    keep: impl Fn(&str) -> bool,
    payload: impl Fn(&[u8]) -> anybytes::Bytes,
) -> Result<(Vec<Id>, TribleSet), Err> {
    let st = SafeTensors::deserialize(bytes)?;
    let mut members = Vec::new();
    let mut facts = TribleSet::new();
    for name in st.names().into_iter().filter(|name| keep(name)) {
        let view = st.tensor(name)?;
        if view.dtype() != safetensors::Dtype::BF16 {
            return Err(format!(
                "{name}: native BF16 import requires BF16, found {:?}; no conversion performed",
                view.dtype()
            )
            .into());
        }
        let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
        let leaf = put_raw_bf16(blobs, payload(view.data()), &shape)?;
        let leaf_id = leaf.root().expect("leaf root");
        facts += leaf.into_facts();
        let kind = match shape.len() {
            1 => "vector",
            2 => "matrix",
            3 => "conv",
            _ => "tensor",
        };
        let name_h = blobs.put::<UTF8String, _>(name.to_string())?;
        let module = entity! { _ @
            attrs::kind: kind, attrs::safetensor_path: name_h, attrs::weight: leaf_id
        };
        members.push(module.root().expect("module root"));
        facts += module.into_facts();
    }
    Ok((members, facts))
}

/// Ingest a stream of already-decoded `(name, f32-data, shape)` tensors into
/// content-addressed member MODULES — the FORMAT-AGNOSTIC core shared by every
/// importer (safetensors, GGUF, pytorch pickle). Each tensor becomes a
/// typed tensor leaf reached by a `{kind, safetensor_path, weight}`
/// module; identical tensors dedup by content. Returns `(member module ids,
/// facts)`. Because the member id is derived purely from the tensor's bytes and
/// shape and name (never the source format), the SAME weights imported from two
/// different container formats produce the SAME members — and hence the same
/// content-addressed model root.
#[cfg(feature = "import")]
pub fn ingest_tensors(
    tensors: impl Iterator<Item = (String, Vec<f32>, Vec<usize>)>,
    blobs: &mut impl BlobStorePut,
    dtype: LeafDtype,
) -> Result<(Vec<Id>, TribleSet), Err> {
    if dtype == LeafDtype::Bf16 {
        return Err("native BF16 import requires safetensors bytes, not decoded f32 tensors".into());
    }
    let mut members: Vec<Id> = Vec::new();
    let mut facts = TribleSet::new();
    for (name, data, shape) in tensors {
        let shp: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let leaf = match dtype {
            LeafDtype::F32 => put_raw(blobs, &data, &shp)?,
            LeafDtype::F16 => put_raw_f16(blobs, &data, &shp)?,
            LeafDtype::Bf16 => unreachable!("rejected before consuming any tensors"),
        };
        let leaf_id = leaf.root().expect("leaf root");
        facts += leaf.into_facts();
        let kind = match shape.len() {
            1 => "vector",
            2 => "matrix",
            3 => "conv",
            _ => "tensor",
        };
        let name_h = blobs.put::<UTF8String, _>(name)?;
        let m = entity! { _ @ attrs::kind: kind, attrs::safetensor_path: name_h, attrs::weight: leaf_id };
        members.push(m.root().expect("module root"));
        facts += m.into_facts();
    }
    Ok((members, facts))
}

/// Build a content-addressed model-ROOT entity over already-ingested member
/// modules — "identity IS content" for a model. The root id is derived by
/// `entity!` from the CORE attr ALONE:
///   - `member*` — the tensor-leaf modules (already content-addressed, so the
///                 weight CONTENT flows into the id).
/// so the id is the PURE content-address of the weight set:
/// `entity!{ _ @ member* }`. Importing the same weights twice yields the SAME
/// root id (dedup across piles); a genuinely different model has different member
/// leaves, so a different id. `member*` is a set: shard/order/duplicates don't
/// move the id. Native vs fp4 differ by their actual weight bytes → different
/// members → different ids automatically, which is why `quantization` need NOT
/// be core.
///
/// `quantization` (the weight-format tag), `source` (the HF id it came from, or a
/// `--name` for a local dir), and `provenance` (the shard file names, as
/// `model_name` facts) are ALL NON-core: attached to the content-derived root id
/// via `ExclusiveId::force_ref`, they are queryable LABELS that never influence
/// the identity. (A consequence: the SAME weights tagged with two different
/// `quantization` labels resolve to the SAME entity id — the label rides on the
/// weights' identity.) Returns the root Fragment carrying the member facts plus
/// the root's own core + label facts.
#[cfg(feature = "import")]
pub fn build_model_root(
    blobs: &mut impl BlobStorePut,
    source: &str,
    quantization: &str,
    members: Vec<Id>,
    member_facts: TribleSet,
    provenance: &[String],
) -> Result<Fragment, Err> {
    let mut facts = member_facts;
    // CORE: the id is the PURE content-address of the weight set (members alone).
    let root = entity! { _ @ attrs::member*: members.iter() };
    let root_id = root.root().expect("model root");
    facts += root.into_facts();
    // NON-core labels on the (content-derived) root id — queryable, no id
    // influence: the weight-format tag, the `source` it was imported from, and
    // which files it came from.
    let source_h = blobs.put::<UTF8String, _>(source.to_string())?;
    facts += entity! { ExclusiveId::force_ref(&root_id) @
        attrs::quantization: quantization,
        attrs::source: source_h,
    }
    .into_facts();
    for name in provenance {
        let mn = blobs.put::<UTF8String, _>(name.clone())?;
        facts += entity! { ExclusiveId::force_ref(&root_id) @ attrs::model_name: mn }.into_facts();
    }
    Ok(Fragment::rooted(root_id, facts))
}

pub fn read_string(
    blobs: &impl BlobStoreGet,
    h: Inline<inlineencodings::Handle<UTF8String>>,
) -> String {
    let v: anybytes::View<str> = blobs.get(h).expect("string blob");
    v.to_string()
}

/// Index a model graph's leaves by tensor name.
///
/// The index costs handles, tensor headers and names — never a copy of a
/// weight: a [`crate::leaf::Leaf`] holds a view over the pile's mapping. That
/// is what lets a large model load from a pile without materializing its whole
/// f32 keymap in RAM at once; the caller decides per tensor whether it ever
/// wants a copy.
///
/// Typed leaves resolve first; a pile still holding the two-blob form resolves
/// through the legacy adapter.
pub fn index_keymap(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    model_id: Id,
) -> HashMap<String, crate::leaf::Leaf> {
    let mut map = HashMap::new();
    let members: Vec<_> =
        find!((m: Id), pattern!(tribles, [{ model_id @ attrs::member: ?m }])).collect();
    for (mid,) in members {
        let (name_h, w_id) = find!(
            (n, w: Id),
            pattern!(tribles, [{ mid @ attrs::safetensor_path: ?n, attrs::weight: ?w }])
        )
        .next()
        .expect("module name/weight");
        let key = read_string(blobs, name_h);
        let leaf = crate::leaf::resolve(tribles, blobs, w_id)
            .expect("resolve tensor leaf")
            .expect("tensor leaf");
        map.insert(key, leaf);
    }
    map
}

/// Materialize a model graph back into a `key → (data, shape)` map by walking
/// its member modules. Feeds `WeightLoader::Pile`. Reads every tensor's f32
/// data into RAM — fine for small models; large models stream via
/// [`index_keymap`] and materialize one leaf at a time instead.
pub fn load_keymap(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    model_id: Id,
) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    index_keymap(tribles, blobs, model_id)
        .into_iter()
        .map(|(k, leaf)| (k, leaf.to_f32_shape()))
        .collect()
}

/// Read just a legacy leaf's shape blob.
///
/// Only the pile converter still needs this: it reads the two-blob form's
/// dimensions in order to write them into a tensor header. Nothing on a load
/// path calls it — a typed leaf's dims come out of its own blob.
pub fn read_shape(
    blobs: &impl BlobStoreGet,
    sh: Inline<inlineencodings::Handle<crate::format::U64Array>>,
) -> Vec<usize> {
    let sbb: anybytes::Bytes = blobs.get(sh).expect("shape blob");
    sbb.view::<[u64]>()
        .expect("shape view")
        .iter()
        .map(|&d| d as usize)
        .collect()
}
