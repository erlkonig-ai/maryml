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
        let kind_h = blobs.put::<UTF8String, _>(kind.to_owned())?;
        let module = entity! { _ @
            attrs::kind: kind_h, attrs::safetensor_path: name_h, attrs::weight: leaf_id
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
        let kind_h = blobs.put::<UTF8String, _>(kind.to_owned())?;
        let m = entity! { _ @ attrs::kind: kind_h, attrs::safetensor_path: name_h, attrs::weight: leaf_id };
        members.push(m.root().expect("module root"));
        facts += m.into_facts();
    }
    Ok((members, facts))
}

/// Build a content-addressed model-ROOT entity over already-ingested member
/// modules, with nothing but the weights in its core. This is
/// [`build_model_root_with`] with an empty [`RootParts`]: the root id is the
/// pure content address of the weight set, `entity!{ _ @ member* }`, exactly
/// the shape every root had before 2026-11-07.
///
/// `quantization` (the weight-format tag), `source` (the HF id it came from, or a
/// `--name` for a local dir), and `provenance` (the shard file names, as
/// `model_name` facts) are ALL NON-core: attached to the content-derived root id
/// via `ExclusiveId::force_ref`, they are queryable LABELS that never influence
/// the identity. (A consequence: the SAME weights tagged with two different
/// `quantization` labels resolve to the SAME entity id — the label rides on the
/// weights' identity.) Returns the root Fragment carrying the member facts plus
/// the root's own core + label facts.
pub fn build_model_root(
    blobs: &mut impl BlobStorePut,
    source: &str,
    quantization: &str,
    members: Vec<Id>,
    member_facts: TribleSet,
    provenance: &[String],
) -> Result<Fragment, Box<dyn std::error::Error>> {
    build_model_root_with(
        blobs,
        source,
        quantization,
        members,
        member_facts,
        provenance,
        RootParts::default(),
    )
}

/// The parts of a model besides its weights that are part of WHAT the model
/// is, and so of its root's identity: the tokenizer it reads text with, the
/// configuration its weights are laid out by, and the chat template its input
/// is framed with. Two checkpoints with the same weights and a different
/// tokenizer or template are different models: they map the same text to
/// different inputs. Each part is optional, because not every model has all
/// three; a root with none of them has today's member-only core.
///
/// The representations, chosen deliberately:
///   - `tokenizer`: the tokenizer ENTITY that
///     [`crate::tokenizer::save_tokenizer_json`] returns, as facts. Its id is
///     itself content-derived (vocab, merges, added tokens and their options,
///     normalizer, pre-tokenizer, post-processor, decoder), so its content
///     flows into the root id, and a reader rebuilds it with
///     [`crate::tokenizer::build_tokenizer`]; no `tokenizer.json` is kept.
///   - `config`: the root node of the `config.json` facts that
///     [`crate::jsonfacts::save_json`] writes. Mary already stores checkpoint
///     sidecars as facts that way, with an exact round trip (integers stay
///     integers, keys are part of node identity), so a field is a query, not
///     a substring of a blob. A reader loads it with
///     [`crate::jsonfacts::load_json`].
///   - `chat_template`: a `UTF8String` blob handle of the template, verbatim.
///     It is Jinja source, not structured data; its bytes are what a codec
///     pins (`qwen3_5::input_codec`), and any re-encoding would change them.
///
/// The facts these ids stand on are the caller's (the `Fragment`s the savers
/// return); this function only links them from the root.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RootParts {
    pub tokenizer: Option<Id>,
    pub config: Option<Id>,
    pub chat_template: Option<Inline<inlineencodings::Handle<UTF8String>>>,
}

/// The root entity's CORE alone, with no labels: `member*` plus the
/// [`RootParts`]. [`build_model_root_with`] is this plus its labels; a tool
/// that completes an existing root and carries that root's own annotations
/// over unchanged uses it directly.
pub fn model_root_core(members: &[Id], parts: RootParts) -> Fragment {
    entity! { _ @
        attrs::member*: members.iter(),
        crate::tokenizer::attrs::tokenizer?: parts.tokenizer,
        attrs::config?: parts.config,
        attrs::chat_template?: parts.chat_template,
    }
}

/// Build a content-addressed model-ROOT entity over already-ingested member
/// modules and the [`RootParts`] that complete it — "identity IS content" for
/// a model. The root id is derived by `entity!` from the CORE attrs:
///   - `member*` — the tensor-leaf modules (already content-addressed, so the
///                 weight CONTENT flows into the id);
///   - `tokenizer?`, `config?`, `chat_template?` — see [`RootParts`].
///
/// Importing the same weights and parts twice yields the SAME root id (dedup
/// across piles and collections); changing any weight, the tokenizer, the
/// configuration or the template yields a different one. `member*` is a set:
/// shard/order/duplicates don't move the id. Native vs fp4 differ by their
/// actual weight bytes → different members → different ids automatically,
/// which is why `quantization` need NOT be core. The labels are as in
/// [`build_model_root`].
pub fn build_model_root_with(
    blobs: &mut impl BlobStorePut,
    source: &str,
    quantization: &str,
    members: Vec<Id>,
    member_facts: TribleSet,
    provenance: &[String],
    parts: RootParts,
) -> Result<Fragment, Box<dyn std::error::Error>> {
    let mut facts = member_facts;
    let root = model_root_core(&members, parts);
    let root_id = root.root().expect("model root");
    facts += root.into_facts();
    // NON-core labels on the (content-derived) root id — queryable, no id
    // influence: the weight-format tag, the `source` it was imported from, and
    // which files it came from.
    let source_h = blobs.put::<UTF8String, _>(source.to_string())?;
    let quantization_h = blobs.put::<UTF8String, _>(quantization.to_string())?;
    facts += entity! { ExclusiveId::force_ref(&root_id) @
        attrs::quantization: quantization_h,
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

#[cfg(test)]
mod root_parts_tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use triblespace::core::collection::{AdmissionPolicy, CollectionPolicy, CollectionStoreExt};
    use triblespace::core::repo::SnapshotSource;
    use triblespace::core::repo::memoryrepo::MemoryRepo;

    const TOKENIZER: &str = r###"{"version": "1.0", "added_tokens": [
        {"id": 3, "content": "<e>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
      "normalizer": null, "pre_tokenizer": {"type": "BertPreTokenizer"},
      "post_processor": {"type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<e>", "type_id": 0}}],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {"<e>": {"id": "<e>", "ids": [3], "tokens": ["<e>"]}}},
      "decoder": null,
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1, "b": 2}}}"###;
    const TEMPLATE: &str = "{{ messages }}";

    fn config() -> serde_json::Value {
        serde_json::json!({"hidden_size": 8})
    }

    /// A stand-in model put into `blobs`: one module naming a leaf, a
    /// tokenizer, a configuration and a template.
    fn model(
        blobs: &mut impl BlobStorePut,
        weight: &str,
        tokenizer: &str,
        config: serde_json::Value,
        template: &str,
    ) -> (Vec<Id>, TribleSet, RootParts) {
        let mut facts = TribleSet::new();
        let path = blobs.put::<UTF8String, _>("w".to_owned()).ok().unwrap();
        let kind = blobs.put::<UTF8String, _>(weight.to_owned()).ok().unwrap();
        let leaf = entity! { _ @ attrs::kind: kind };
        let leaf_id = leaf.root().unwrap();
        facts += leaf.into_facts();
        let module = entity! { _ @ attrs::safetensor_path: path, attrs::weight: leaf_id };
        let members = vec![module.root().unwrap()];
        facts += module.into_facts();
        let tok = crate::tokenizer::save_tokenizer_json(tokenizer.as_bytes(), "t", blobs).unwrap();
        let tokenizer = tok.root();
        facts += tok.into_facts();
        let config = Some(crate::jsonfacts::save_json(&config, blobs, &mut facts).unwrap());
        let chat_template = Some(blobs.put::<UTF8String, _>(template.to_owned()).ok().unwrap());
        (members, facts, RootParts { tokenizer, config, chat_template })
    }

    fn root_of(
        blobs: &mut impl BlobStorePut,
        weight: &str,
        tokenizer: &str,
        config: serde_json::Value,
        template: &str,
    ) -> Fragment {
        let (members, facts, parts) = model(blobs, weight, tokenizer, config, template);
        build_model_root_with(blobs, "org/m", "native", members, facts, &[], parts).unwrap()
    }

    #[test]
    fn every_part_of_the_core_moves_the_root_and_labels_do_not() {
        let mut blobs = MemoryBlobStore::new();
        let base = root_of(&mut blobs, "weights", TOKENIZER, config(), TEMPLATE).root().unwrap();
        let again = root_of(&mut blobs, "weights", TOKENIZER, config(), TEMPLATE).root().unwrap();
        assert_eq!(base, again, "same content, same root");
        let no_post = TOKENIZER.replace(
            r#", {"SpecialToken": {"id": "<e>", "type_id": 0}}],"#,
            "],",
        );
        let other_vocab = TOKENIZER.replace(r#""b": 2"#, r#""c": 2"#);
        assert!(no_post != TOKENIZER && other_vocab != TOKENIZER);
        for (what, other) in [
            ("weights", root_of(&mut blobs, "other weights", TOKENIZER, config(), TEMPLATE)),
            ("tokenizer", root_of(&mut blobs, "weights", &other_vocab, config(), TEMPLATE)),
            ("post-processor", root_of(&mut blobs, "weights", &no_post, config(), TEMPLATE)),
            ("config", root_of(&mut blobs, "weights", TOKENIZER, serde_json::json!({"hidden_size": 9}), TEMPLATE)),
            ("template", root_of(&mut blobs, "weights", TOKENIZER, config(), "{{ messages }} ")),
        ] {
            assert_ne!(other.root().unwrap(), base, "a changed {what} must give another root");
        }
        let (members, facts, parts) = model(&mut blobs, "weights", TOKENIZER, config(), TEMPLATE);
        let relabelled = build_model_root_with(
            &mut blobs, "another/source", "fp4", members, facts, &["x".into()], parts,
        )
        .unwrap();
        assert_eq!(relabelled.root().unwrap(), base, "labels annotate, they never move the id");
    }

    #[test]
    fn a_root_without_parts_keeps_the_member_only_core() {
        let mut blobs = MemoryBlobStore::new();
        let (members, facts, _) = model(&mut blobs, "weights", TOKENIZER, config(), TEMPLATE);
        let plain = build_model_root(&mut blobs, "org/m", "native", members.clone(), facts, &[]).unwrap();
        let member_only = entity! { _ @ attrs::member*: members.iter() };
        assert_eq!(plain.root(), member_only.root());
    }

    #[test]
    fn the_same_model_in_two_collections_is_one_root_with_the_same_parts() {
        let mut roots = Vec::new();
        let mut handles = Vec::new();
        let mut all_parts = Vec::new();
        for seed in [0x11u8, 0x22] {
            let mut repo = MemoryRepo::default();
            let fragment = root_of(&mut repo, "weights", TOKENIZER, config(), TEMPLATE);
            let key = SigningKey::from_bytes(&[seed; 32]);
            let policy = CollectionPolicy::new(
                AdmissionPolicy::direct(key.verifying_key()),
                AdmissionPolicy::direct(key.verifying_key()),
            );
            let collection = repo.collection("models", policy).unwrap();
            repo.commit(collection, &key, fragment).unwrap();
            let store = SnapshotSource::snapshot(&mut repo).unwrap();
            let snapshot =
                crate::model_collection::snapshot_model_collection_for(&store, collection).unwrap();
            let root = crate::selection::select_model_root(
                snapshot.facts(),
                &store,
                crate::selection::ModelSelector::Only,
            )
            .unwrap();
            all_parts.push(crate::selection::model_root_parts(snapshot.facts(), root).unwrap());
            roots.push(root);
            handles.push(collection.handle());
        }
        assert_ne!(handles[0], handles[1], "two different collections");
        assert_eq!(roots[0], roots[1], "one model");
        assert_eq!(all_parts[0], all_parts[1]);
        assert!(all_parts[0].tokenizer.is_some() && all_parts[0].config.is_some());
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn the_tokenizer_named_by_the_root_rebuilds_with_its_post_processor() {
        let mut blobs = MemoryBlobStore::new();
        let fragment = root_of(&mut blobs, "weights", TOKENIZER, config(), TEMPLATE);
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let root = fragment.root().unwrap();
        let parts = crate::selection::model_root_parts(fragment.facts(), root).unwrap();
        let tok = crate::tokenizer::build_tokenizer(fragment.facts(), &reader, parts.tokenizer.unwrap())
            .unwrap();
        let json = tokenizers::Tokenizer::from_bytes(TOKENIZER.as_bytes()).unwrap();
        for text in ["a b", "a <e> b", ""] {
            assert_eq!(
                tok.encode(text, true).unwrap().get_ids(),
                json.encode(text, true).unwrap().get_ids()
            );
        }
        assert_eq!(tok.encode("a", true).unwrap().get_ids(), &[1, 3]);
        let config = crate::jsonfacts::load_json(fragment.facts(), &reader, parts.config.unwrap()).unwrap();
        assert_eq!(config, self::config());
    }
}
