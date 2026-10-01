//! Deterministic model and tokenizer selection over an already-open graph.
//!
//! A consolidated collection can contain several models and tokenizers. The
//! old pile wrappers predated that: they selected the first matching root or
//! extended a `HashMap`, making iteration order decide ambiguous data. This
//! module keeps storage out of the decision. Callers pass a materialized
//! [`TribleSet`], its blob reader, and an explicit selector. Names and source
//! labels are annotations: adding an alias does not invalidate a model. A
//! caller choosing one model must still disambiguate distinct matching roots.

use crate::format::attrs;
use crate::leaf::Leaf;
use crate::model_collection::ModelSnapshot;
use anyhow::{Context, anyhow, bail};
use std::collections::{BTreeSet, HashMap};
use triblespace::prelude::*;

/// How to identify one model component in a consolidated graph.
///
/// `Only`, `Root`, and `Name` retain singular semantics. A `Source`
/// coordinate is the one selector that may deliberately name several roots:
/// weight-file shards of one component carry the same source and
/// quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSelector<'a> {
    /// Succeed only when the graph contains exactly one model root.
    Only,
    /// Select an existing root by its opaque entity ID.
    Root(Id),
    /// Select the one legacy/root entity carrying this exact `model_name`.
    Name(&'a str),
    /// Select every shard root carrying this source and weight-format label.
    Source {
        source: &'a str,
        quantization: &'a str,
    },
}

/// How to identify one tokenizer root in a consolidated graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerSelector<'a> {
    /// Succeed only when the graph contains exactly one tokenizer root.
    Only,
    /// Select an existing tokenizer by its opaque entity ID.
    Root(Id),
    /// Select the one tokenizer carrying this exact `model_name`.
    Name(&'a str),
}

/// One explicitly selected model component, the real roots that make it up,
/// its strict tensor-handle index, and the blob reader that owns every indexed
/// attachment.
///
/// Selection consumes a frozen [`ModelSnapshot`]. Once the roots and
/// functional fields have been validated, the collection facts and exact
/// cover are no longer needed by weight loading; the reader is retained so
/// the content handles remain resolvable without reopening storage. This is
/// the storage-policy-free boundary for lazy, streaming, and mmap-aliased
/// loaders.
pub struct SelectedModelIndex<R> {
    // Invariant: sorted, duplicate-free, and nonempty. The field stays private
    // so every construction path has to preserve that canonical form.
    roots: Vec<Id>,
    handles: HashMap<String, Leaf>,
    reader: R,
}

impl<R> SelectedModelIndex<R> {
    /// Canonical real roots of the selected component.
    pub fn roots(&self) -> &[Id] {
        &self.roots
    }

    /// Return the root only when this component is represented by exactly one.
    ///
    /// Callers whose domain contract is singular should check this at their
    /// admission boundary. A sharded component deliberately returns `None`.
    pub fn single_root(&self) -> Option<Id> {
        match self.roots.as_slice() {
            [root] => Some(*root),
            _ => None,
        }
    }

    /// Strict `tensor name -> leaf` index for the selected roots.
    pub fn handles(&self) -> &HashMap<String, Leaf> {
        &self.handles
    }

    /// Reader that owns the attachment snapshot named by [`Self::handles`].
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Consume the selection into its canonical roots, leaf index, and reader.
    pub fn into_parts(self) -> (Vec<Id>, HashMap<String, Leaf>, R) {
        (self.roots, self.handles, self.reader)
    }
}

fn read_long_string(
    blobs: &impl BlobStoreGet,
    handle: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
    field: &str,
) -> anyhow::Result<String> {
    let value: anybytes::View<str> = blobs
        .get(handle)
        .map_err(|error| anyhow!("read {field} blob: {error}"))?;
    Ok(value.to_string())
}

fn exactly_one<T>(
    values: impl IntoIterator<Item = T>,
    what: impl std::fmt::Display,
) -> anyhow::Result<T> {
    let mut values = values.into_iter();
    let Some(value) = values.next() else {
        bail!("no {what}");
    };
    if values.next().is_some() {
        bail!("ambiguous {what}");
    }
    Ok(value)
}

fn model_roots(tribles: &TribleSet) -> BTreeSet<Id> {
    find!(
        (model: Id),
        pattern!(tribles, [
            { ?model @ attrs::member: _?member },
            { _?member @ attrs::safetensor_path: _?path, attrs::weight: _?weight },
        ])
    )
    .map(|(model,)| model)
    .collect()
}

/// Resolve exactly one model root according to `selector`.
pub fn select_model_root(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<Id> {
    match selector {
        ModelSelector::Only => exactly_one(model_roots(tribles), "model root in graph"),
        ModelSelector::Root(root) => {
            if exists!(pattern!(tribles, [{ root @ attrs::member: _?member }])) {
                Ok(root)
            } else {
                bail!("model root {root} is absent or has no members")
            }
        }
        ModelSelector::Name(wanted) => {
            let matches = find!(
                (model: Id, name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?model @ attrs::model_name: ?name, attrs::member: _?member }])
            )
            .filter_map(|(model, name)| {
                (read_long_string(blobs, name, "model_name").ok()?.as_str() == wanted)
                    .then_some(model)
            })
            .collect::<BTreeSet<_>>();
            exactly_one(matches, format_args!("model root named {wanted:?}"))
        }
        ModelSelector::Source {
            source: wanted_source,
            quantization,
        } => {
            let matches = find!(
                (model: Id, source: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?model @
                    attrs::source: ?source,
                    attrs::quantization: quantization,
                    attrs::member: _?member,
                }])
            )
            .filter_map(|(model, source)| {
                (read_long_string(blobs, source, "source").ok()?.as_str() == wanted_source)
                    .then_some(model)
            })
            .collect::<BTreeSet<_>>();
            exactly_one(
                matches,
                format_args!(
                    "model root with source {wanted_source:?} and quantization {quantization:?}"
                ),
            )
        }
    }
}

/// Resolve every model root matching `selector`, for a component that may be
/// sharded across several of them.
///
/// [`select_model_root`] answers "which root is this?" and is right whenever a
/// coordinate names exactly one. This answers "which roots make this up?", and
/// is the entry point [`index_keymap_for_roots`] needs: a sharded checkpoint
/// gives every one of its files the same coordinate, so the singular selector
/// fails `exactly_one` on precisely the models that need selecting.
///
/// More than one root coming back is a CLAIM that they are shards of one
/// component, not a verified fact — this only knows they share a coordinate.
/// [`index_keymap_for_roots`] is what tests the claim, by requiring their
/// tensor names to be disjoint. Keeping those two steps apart is deliberate:
/// selection reads coordinates, and the merge is where a wrong grouping is
/// caught, so a mislabelled root fails loudly at load rather than quietly
/// shadowing a tensor.
///
/// `Only`, `Name`, and `Root` retain their exact-one-root semantics. Only
/// `Source` can name a cohort: sharing `(source, quantization)` is the explicit
/// claim that several roots are shards of one component. This prevents `Only`
/// from silently combining unrelated components merely because their tensor
/// names happen to be disjoint.
///
/// The result is sorted and duplicate-free, so callers see one deterministic
/// order regardless of query iteration.
pub fn select_model_roots(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<Vec<Id>> {
    let roots: BTreeSet<Id> = match selector {
        ModelSelector::Only | ModelSelector::Root(_) | ModelSelector::Name(_) => {
            return Ok(vec![select_model_root(tribles, blobs, selector)?]);
        }
        ModelSelector::Source {
            source: wanted_source,
            quantization,
        } => {
            let matches: BTreeSet<Id> = find!(
                (model: Id, source: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?model @
                    attrs::source: ?source,
                    attrs::quantization: quantization,
                    attrs::member: _?member,
                }])
            )
            .filter_map(|(model, source)| {
                (read_long_string(blobs, source, "source").ok()?.as_str() == wanted_source)
                    .then_some(model)
            })
            .collect();
            matches
        }
    };

    if roots.is_empty() {
        bail!("no model root matches {selector:?}");
    }
    Ok(roots.into_iter().collect())
}

/// Operational selection over one fixed graph. An empty result means that
/// every queried label was readable and no root matched; an unavailable label
/// is an error, never a reason to select a different model. Undecodable label
/// bytes still supply no typed row, as in the passive selectors above.
pub fn matching_model_roots_acquiring(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<Vec<Id>> {
    let mut roots = BTreeSet::new();
    match selector {
        ModelSelector::Only => roots = model_roots(tribles),
        ModelSelector::Root(root) => {
            if exists!(pattern!(tribles, [{ root @ attrs::member: _?member }])) {
                roots.insert(root);
            }
        }
        ModelSelector::Name(wanted) => {
            for (model, name) in find!(
                (model: Id, name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?model @ attrs::model_name: ?name, attrs::member: _?member }])
            ) {
                if acquiring_label(blobs, name, "model_name")?.as_deref() == Some(wanted) {
                    roots.insert(model);
                }
            }
        }
        ModelSelector::Source { source: wanted, quantization } => {
            for (model, source) in find!(
                (model: Id, source: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?model @
                    attrs::source: ?source,
                    attrs::quantization: quantization,
                    attrs::member: _?member,
                }])
            ) {
                if acquiring_label(blobs, source, "source")?.as_deref() == Some(wanted) {
                    roots.insert(model);
                }
            }
        }
    }
    Ok(roots.into_iter().collect())
}

fn acquiring_label(
    blobs: &impl BlobStoreGet,
    handle: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
    field: &str,
) -> anyhow::Result<Option<anybytes::View<str>>> {
    use triblespace::core::blob::{Blob, TryFromBlob};
    let bytes: Blob<blobencodings::UTF8String> = blobs
        .get(handle)
        .with_context(|| format!("read {field} label {handle:?}"))?;
    Ok(anybytes::View::<str>::try_from_blob(bytes).ok())
}

/// Exactly one operational model root; unlike passive discovery, a failed
/// label read cannot erase an otherwise ambiguous candidate.
pub fn select_model_root_acquiring(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<Id> {
    exactly_one(
        matching_model_roots_acquiring(tribles, blobs, selector)?,
        format_args!("model root matching {selector:?}"),
    )
}

/// Operational component selection with the same shard/single-root contract
/// as [`select_model_roots`]. All label failures propagate before selection.
pub fn select_model_roots_acquiring(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<Vec<Id>> {
    if !matches!(selector, ModelSelector::Source { .. }) {
        return Ok(vec![select_model_root_acquiring(tribles, blobs, selector)?]);
    }
    let roots = matching_model_roots_acquiring(tribles, blobs, selector)?;
    if roots.is_empty() {
        bail!("no model root matches {selector:?}");
    }
    Ok(roots)
}

/// Index a component named by `selector`, however many roots carry it.
///
/// The pair of [`select_model_roots`] and [`index_keymap_for_roots`], which is
/// how a caller should normally reach a sharded component: one coordinate in,
/// one strict `name -> leaf` index out, whether the writer split the weights
/// across one file or four.
pub fn index_keymap_for_selector(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<HashMap<String, Leaf>> {
    let roots = select_model_roots(tribles, blobs, selector)?;
    index_keymap_for_roots(tribles, blobs, &roots)
}

fn model_members(tribles: &TribleSet, root: Id) -> anyhow::Result<BTreeSet<Id>> {
    let members: BTreeSet<_> = find!(
        (member: Id),
        pattern!(tribles, [{ root @ attrs::member: ?member }])
    )
    .map(|(member,)| member)
    .collect();
    if members.is_empty() {
        bail!("model root {root} has no members");
    }
    Ok(members)
}

/// Build a strict `name -> leaf` index for one model root.
///
/// Every module must have exactly one name and weight edge, every weight entity
/// must carry exactly one readable tensor leaf, and tensor names must be
/// globally unique within the model. Violations are errors rather than
/// iteration-order-dependent `HashMap` overwrites.
///
/// A leaf's element format and shape come out of the leaf itself — the
/// attribute's id names the element and rank, and the blob header names the
/// dims — so there is nothing here that pairs a payload with a shape and hopes.
pub fn index_keymap_for_root(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    root: Id,
) -> anyhow::Result<HashMap<String, Leaf>> {
    let mut map = HashMap::new();
    for member in model_members(tribles, root)? {
        let name_handle = exactly_one(
            find!(
                (name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ member @ attrs::safetensor_path: ?name }])
            )
            .map(|(name,)| name),
            format_args!("safetensor_path on model member {member}"),
        )?;
        let weight = exactly_one(
            find!(
                (weight: Id),
                pattern!(tribles, [{ member @ attrs::weight: ?weight }])
            )
            .map(|(weight,)| weight),
            format_args!("weight edge on model member {member}"),
        )?;
        let handles = crate::leaf::resolve(tribles, blobs, weight)
            .with_context(|| format!("read tensor leaf {weight}"))?
            .ok_or_else(|| anyhow!("tensor leaf {weight} carries no readable tensor"))?;

        let name = read_long_string(blobs, name_handle, "safetensor_path")?;
        if map.insert(name.clone(), handles).is_some() {
            bail!("duplicate tensor name {name:?} under model root {root}");
        }
    }
    Ok(map)
}

/// Index one component whose tensors are split across several roots.
///
/// A sharded checkpoint writes one root per `model-0000N-of-0000M.safetensors`
/// file, and those roots are not alternatives: together they are one component,
/// and no single one of them can load. [`index_keymap_for_root`] cannot express
/// that, and [`select_model_root`] actively refuses it, since a coordinate
/// naming two roots fails `exactly_one`.
///
/// The merge is by TENSOR NAME rather than by any structural split, because the
/// files do not divide structurally. In the gemma-3-1b checkpoint layer 17 has
/// tensors in *both* shards while layers 22, 24 and 25 sit in the first, so a
/// merge that assigned a layer range per shard would work on a clean two-way
/// boundary and silently drop tensors here.
///
/// DISJOINTNESS IS THE INVARIANT, and it is checked rather than assumed. Two
/// shards of one component cannot legitimately name the same tensor: the writer
/// split a single namespace across files. So a collision is not a case to
/// resolve by preferring a shard -- it means the roots are not shards of one
/// thing, and quietly taking either one would hide that. Roots are visited in
/// sorted order so the collision reported does not depend on the caller's
/// argument order.
///
/// This composes with component multiplicity rather than replacing it: a model
/// like FLUX has three components of which one is itself sharded, so its loader
/// holds three indexes and builds one of them from two roots.
pub fn index_keymap_for_roots(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    roots: &[Id],
) -> anyhow::Result<HashMap<String, Leaf>> {
    if roots.is_empty() {
        bail!("cannot index a component from zero roots");
    }

    let mut ordered: Vec<Id> = roots.to_vec();
    ordered.sort();
    ordered.dedup();

    let mut merged: HashMap<String, Leaf> = HashMap::new();
    let mut owner: HashMap<String, Id> = HashMap::new();
    for root in ordered {
        for (name, leaf) in index_keymap_for_root(tribles, blobs, root)? {
            if let Some(previous) = owner.insert(name.clone(), root) {
                bail!(
                    "tensor {name:?} appears in both root {previous} and root {root}; \
                     these are not shards of one component"
                );
            }
            merged.insert(name, leaf);
        }
    }
    Ok(merged)
}

impl<R: BlobStoreGet> SelectedModelIndex<R> {
    /// Strictly index an explicit set of real model roots.
    ///
    /// This is the domain-composition seam for a model assembled from several
    /// independently selected components. The roots are canonicalized without
    /// inventing an aggregate identity, and tensor names must be disjoint over
    /// the whole set.
    pub fn from_roots(
        facts: &TribleSet,
        reader: R,
        roots: impl IntoIterator<Item = Id>,
    ) -> anyhow::Result<Self> {
        let mut roots: Vec<Id> = roots.into_iter().collect();
        roots.sort();
        roots.dedup();
        if roots.is_empty() {
            bail!("cannot select a model component from zero roots");
        }
        let handles = index_keymap_for_roots(facts, &reader, &roots)
            .context("index explicit model roots from graph")?;
        Ok(Self {
            roots,
            handles,
            reader,
        })
    }

    /// Resolve and strictly index one model from an explicit graph plus its
    /// owning reader.
    ///
    /// This is the unpublished-candidate seam: callers may union staged facts
    /// with an authorized snapshot, validate the resulting view, and only then
    /// publish authority. Storage admission remains entirely with the caller.
    pub fn from_graph(
        facts: &TribleSet,
        reader: R,
        selector: ModelSelector<'_>,
    ) -> anyhow::Result<Self> {
        let roots = select_model_roots(facts, &reader, selector)
            .context("select model roots from explicit graph")?;
        Self::from_roots(facts, reader, roots)
    }

    /// Resolve and strictly index one model from an already-frozen collection.
    ///
    /// Storage opening and admission policy remain with the caller. Ambiguous
    /// selectors, malformed functional fields, mixed data encodings on one
    /// leaf, and duplicate tensor names all fail before the snapshot is
    /// consumed.
    pub fn from_snapshot(
        snapshot: ModelSnapshot<R>,
        selector: ModelSelector<'_>,
    ) -> anyhow::Result<Self> {
        // `R` is now the whole frozen store observation rather than a blob
        // reader leased separately from it, so the facts and the attachments
        // this index will fetch are guaranteed to be the same pile prefix.
        let (facts, _, store) = snapshot.into_parts();
        Self::from_graph(&facts, store, selector)
            .context("select model from native collection snapshot")
    }
}

/// Select one model and materialize its tensor keymap from an already-open
/// graph and blob reader.
pub fn load_keymap_from_graph(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let roots = select_model_roots(tribles, blobs, selector)?;
    load_keymap_for_roots(tribles, blobs, &roots)
}

/// Load already-selected roots without repeating label discovery after I/O.
pub fn load_keymap_for_roots(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    roots: &[Id],
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let mut keymap: HashMap<String, (Vec<f32>, Vec<usize>)> =
        index_keymap_for_roots(tribles, blobs, roots)?
            .into_iter()
            .map(|(name, leaf)| (name, leaf.to_f32_shape()))
            .collect();
    fold_input_scales(&mut keymap)?;
    Ok(keymap)
}

/// Fold every `<weight>.input_scale` vector into its matrix and drop it.
///
/// A packed linear written by [`crate::calibrate`] carries its activation-aware
/// channel scales as a sibling F32 vector rather than inside its codes. A
/// consumer that reads f32 matrices wants the product: column `j` of the
/// matrix times `input_scale[j]`, which is what applying the vector to the
/// input would compute. Folded here, once, so no model loader has to know the
/// convention; a kernel that reads the packed codes directly applies the vector
/// to its input instead. Returns how many vectors were folded.
pub fn fold_input_scales(
    keymap: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> anyhow::Result<usize> {
    const SUFFIX: &str = ".input_scale";
    let names: Vec<String> = keymap
        .keys()
        .filter(|k| k.ends_with(SUFFIX))
        .cloned()
        .collect();
    for name in &names {
        let base = &name[..name.len() - SUFFIX.len()];
        let (scale, scale_shape) = keymap.remove(name).expect("listed key");
        let (values, shape) = keymap
            .get_mut(base)
            .ok_or_else(|| anyhow::anyhow!("{name} has no matrix {base:?} to fold into"))?;
        anyhow::ensure!(
            shape.len() == 2 && scale_shape.len() == 1 && scale.len() == shape[1],
            "{name}: a vector of {} for a matrix {shape:?}",
            scale.len()
        );
        for row in values.chunks_mut(shape[1]) {
            for (v, s) in row.iter_mut().zip(&scale) {
                *v *= s;
            }
        }
    }
    Ok(names.len())
}

fn tokenizer_roots(tribles: &TribleSet) -> BTreeSet<Id> {
    crate::tokenizer::find_tokenizers(tribles).collect()
}

/// Resolve exactly one tokenizer root according to `selector`.
pub fn select_tokenizer_root(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: TokenizerSelector<'_>,
) -> anyhow::Result<Id> {
    match selector {
        TokenizerSelector::Only => exactly_one(tokenizer_roots(tribles), "tokenizer root in graph"),
        TokenizerSelector::Root(root) => {
            if tokenizer_roots(tribles).contains(&root) {
                Ok(root)
            } else {
                bail!("tokenizer root {root} is absent or lacks a tokenizer kind/vocabulary")
            }
        }
        TokenizerSelector::Name(wanted) => {
            let roots = tokenizer_roots(tribles);
            let matches = find!(
                (tokenizer: Id, name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?tokenizer @ crate::tokenizer::attrs::model_name: ?name }])
            )
            .filter(|(tokenizer, _)| roots.contains(tokenizer))
            .filter_map(|(tokenizer, name)| {
                (read_long_string(blobs, name, "tokenizer model_name")
                    .ok()?
                    .as_str()
                    == wanted)
                    .then_some(tokenizer)
            })
            .collect::<BTreeSet<_>>();
            exactly_one(matches, format_args!("tokenizer root named {wanted:?}"))
        }
    }
}

/// Operational tokenizer selection; missing labels are not absent candidates.
pub fn select_tokenizer_root_acquiring(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: TokenizerSelector<'_>,
) -> anyhow::Result<Id> {
    let TokenizerSelector::Name(wanted) = selector else {
        return select_tokenizer_root(tribles, blobs, selector);
    };
    let roots = tokenizer_roots(tribles);
    let mut matches = BTreeSet::new();
    for (tokenizer, name) in find!(
        (tokenizer: Id, name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
        pattern!(tribles, [{ ?tokenizer @ crate::tokenizer::attrs::model_name: ?name }])
    ).filter(|(tokenizer, _)| roots.contains(tokenizer)) {
        if acquiring_label(blobs, name, "tokenizer model_name")?.as_deref() == Some(wanted) {
            matches.insert(tokenizer);
        }
    }
    exactly_one(matches, format_args!("tokenizer root named {wanted:?}"))
}

/// Select and construct one HuggingFace tokenizer from an already-open graph.
#[cfg(feature = "tokenizer")]
pub fn load_tokenizer_from_graph(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: TokenizerSelector<'_>,
) -> anyhow::Result<tokenizers::Tokenizer> {
    let root = select_tokenizer_root(tribles, blobs, selector)?;
    crate::tokenizer::build_tokenizer(tribles, blobs, root)
        .map_err(|error| anyhow!("build tokenizer {root}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::blob::MemoryBlobStore;
    #[cfg(feature = "tokenizer")]
    use triblespace::core::metadata;

    struct ModelFixture {
        root: Id,
        members: Vec<Id>,
    }

    /// How a fixture writes its leaves. Both forms are in the wild: the typed
    /// tensor every importer writes now, and the two-blob pair every existing
    /// model pile holds. Selection has to work on both, so the tests say which.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LeafForm {
        Typed,
        TwoBlob,
    }

    fn add_leaf(
        facts: &mut TribleSet,
        blobs: &mut MemoryBlobStore,
        form: LeafForm,
        value: f32,
    ) -> Id {
        let leaf = match form {
            LeafForm::Typed => crate::format::put_raw(blobs, &[value], &[1]).unwrap(),
            LeafForm::TwoBlob => {
                let data = blobs
                    .put::<crate::format::F32Array, _>(vec![value])
                    .unwrap();
                let shape = blobs.put::<crate::format::U64Array, _>(vec![1]).unwrap();
                entity! { _ @ attrs::data: data, attrs::shape: shape }
            }
        };
        let leaf_id = leaf.root().unwrap();
        *facts += leaf.into_facts();
        leaf_id
    }

    fn add_model(
        facts: &mut TribleSet,
        blobs: &mut MemoryBlobStore,
        name: &str,
        source: &str,
        quantization: &str,
        tensors: &[(&str, f32)],
    ) -> ModelFixture {
        add_model_as(
            facts,
            blobs,
            LeafForm::Typed,
            name,
            source,
            quantization,
            tensors,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_model_as(
        facts: &mut TribleSet,
        blobs: &mut MemoryBlobStore,
        form: LeafForm,
        name: &str,
        source: &str,
        quantization: &str,
        tensors: &[(&str, f32)],
    ) -> ModelFixture {
        let mut members = Vec::new();
        for &(tensor_name, value) in tensors {
            let leaf_id = add_leaf(facts, blobs, form, value);

            let name = blobs
                .put::<blobencodings::UTF8String, _>(tensor_name.to_string())
                .unwrap();
            let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: &leaf_id };
            let member_id = member.root().unwrap();
            *facts += member.into_facts();
            members.push(member_id);
        }

        let name = blobs
            .put::<blobencodings::UTF8String, _>(name.to_string())
            .unwrap();
        let source = blobs
            .put::<blobencodings::UTF8String, _>(source.to_string())
            .unwrap();
        let model = entity! { _ @
            attrs::model_name: name,
            attrs::source: source,
            attrs::quantization: quantization,
            attrs::member*: members.iter(),
        };
        let root = model.root().unwrap();
        *facts += model.into_facts();
        ModelFixture { root, members }
    }

    #[test]
    fn acquiring_selection_does_not_erase_an_unread_label() {
        use triblespace::core::blob::IntoBlob;
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let alpha = add_model(
            &mut facts, &mut blobs, "alpha", "org/alpha", "native",
            &[("alpha.weight", 1.0)],
        );
        let late: triblespace::core::blob::Blob<blobencodings::UTF8String> =
            "org/other".to_owned().to_blob();
        let missing = late.get_handle();
        facts += entity! {
            attrs::source: missing,
            attrs::quantization: "native",
            attrs::member: fucid(),
        }.into_facts();
        let selector = ModelSelector::Source { source: "org/alpha", quantization: "native" };
        let before = blobs.snapshot().unwrap();
        assert_eq!(select_model_roots(&facts, &before, selector).unwrap(), vec![alpha.root]);
        let error = matching_model_roots_acquiring(&facts, &before, selector).unwrap_err();
        assert!(error.to_string().contains("read source label"));
        // Making bytes available changes what can be answered, not the facts
        // or the roots named by the already-selected graph.
        blobs.put::<blobencodings::UTF8String, _>(late).unwrap();
        let after = blobs.snapshot().unwrap();
        assert_eq!(matching_model_roots_acquiring(&facts, &after, selector).unwrap(), vec![alpha.root]);
        assert!(matching_model_roots_acquiring(&facts, &before, selector).is_err());
        assert!(matching_model_roots_acquiring(&facts, &after, ModelSelector::Source {
            source: "org/absent", quantization: "native",
        }).unwrap().is_empty());
    }

    #[test]
    fn acquiring_selection_keeps_undecodable_labels_out_of_the_typed_query() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let alpha = add_model(
            &mut facts, &mut blobs, "alpha", "org/alpha", "native",
            &[("alpha.weight", 1.0)],
        );
        let invalid = blobs.put::<blobencodings::RawBytes, _>(vec![0xff_u8]).unwrap();
        let invalid: Inline<inlineencodings::Handle<blobencodings::UTF8String>> = invalid.transmute();
        facts += entity! {
            attrs::source: invalid,
            attrs::quantization: "native",
            attrs::member: fucid(),
        }.into_facts();
        let reader = blobs.snapshot().unwrap();
        assert_eq!(select_model_root_acquiring(&facts, &reader, ModelSelector::Source {
            source: "org/alpha", quantization: "native",
        }).unwrap(), alpha.root);
    }

    #[test]
    fn model_selectors_are_exact_in_a_consolidated_graph() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let alpha = add_model(
            &mut facts,
            &mut blobs,
            "alpha",
            "org/alpha",
            "native",
            &[("alpha.weight", 1.0)],
        );
        let beta = add_model(
            &mut facts,
            &mut blobs,
            "beta",
            "org/beta",
            "fp4",
            &[("beta.weight", 2.0)],
        );
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        assert!(select_model_root(&facts, &reader, ModelSelector::Only).is_err());
        assert!(select_model_roots(&facts, &reader, ModelSelector::Only).is_err());
        assert_eq!(
            select_model_root(&facts, &reader, ModelSelector::Name("alpha")).unwrap(),
            alpha.root
        );
        assert_eq!(
            select_model_root(
                &facts,
                &reader,
                ModelSelector::Source {
                    source: "org/beta",
                    quantization: "fp4",
                },
            )
            .unwrap(),
            beta.root
        );
        assert_eq!(
            select_model_root(&facts, &reader, ModelSelector::Root(beta.root)).unwrap(),
            beta.root
        );

        let keymap = load_keymap_from_graph(
            &facts,
            &reader,
            ModelSelector::Source {
                source: "org/alpha",
                quantization: "native",
            },
        )
        .unwrap();
        assert_eq!(keymap["alpha.weight"], (vec![1.0], vec![1]));
        assert!(select_model_root(&facts, &reader, ModelSelector::Name("missing")).is_err());
    }

    #[test]
    fn duplicate_model_names_and_functional_fields_are_errors() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let first = add_model(
            &mut facts,
            &mut blobs,
            "same",
            "org/one",
            "native",
            &[("weight", 1.0)],
        );
        add_model(
            &mut facts,
            &mut blobs,
            "same",
            "org/two",
            "native",
            &[("weight", 2.0)],
        );
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        assert!(select_model_root(&facts, &reader, ModelSelector::Name("same")).is_err());

        let extra_leaf_id = add_leaf(&mut facts, &mut blobs, LeafForm::Typed, 3.0);
        let member = first.members[0];
        facts += entity! { ExclusiveId::force_ref(&member) @ attrs::weight: &extra_leaf_id }
            .into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let error = match index_keymap_for_root(&facts, &reader, first.root) {
            Ok(_) => panic!("duplicate weight edge was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("ambiguous weight edge"), "{error}");
    }

    /// One coordinate, two shard roots. The singular selector refuses this
    /// graph, which is the whole reason the plural one exists.
    #[test]
    fn a_coordinate_naming_two_shards_selects_both_and_indexes_as_one() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let first = add_model(
            &mut facts,
            &mut blobs,
            "model-00001-of-00002.safetensors",
            "google/gemma-3-1b",
            "native",
            &[("model.layers.0.weight", 1.0)],
        );
        let second = add_model(
            &mut facts,
            &mut blobs,
            "model-00002-of-00002.safetensors",
            "google/gemma-3-1b",
            "native",
            &[("model.layers.1.weight", 2.0)],
        );
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let selector = ModelSelector::Source {
            source: "google/gemma-3-1b",
            quantization: "native",
        };

        // The singular selector cannot express a sharded component.
        assert!(select_model_root(&facts, &reader, selector).is_err());

        let mut expected = vec![first.root, second.root];
        expected.sort();
        assert_eq!(
            select_model_roots(&facts, &reader, selector).unwrap(),
            expected
        );

        let index = index_keymap_for_selector(&facts, &reader, selector).unwrap();
        let mut names: Vec<&String> = index.keys().collect();
        names.sort();
        assert_eq!(
            names,
            vec!["model.layers.0.weight", "model.layers.1.weight"]
        );
        assert_eq!(index["model.layers.0.weight"].to_f32(), vec![1.0]);
        assert_eq!(index["model.layers.1.weight"].to_f32(), vec![2.0]);

        let selected = SelectedModelIndex::from_graph(&facts, reader, selector).unwrap();
        assert_eq!(selected.roots(), expected);
        assert_eq!(selected.single_root(), None);
        assert_eq!(selected.handles().len(), 2);
    }

    /// Sharing a coordinate is a claim, not proof. Selection groups the roots;
    /// the merge is what rejects a grouping that cannot be shards.
    #[test]
    fn a_mislabelled_root_is_caught_by_the_merge_not_by_selection() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        add_model(
            &mut facts,
            &mut blobs,
            "shard-a",
            "vendor/model",
            "native",
            &[("model.layers.0.weight", 1.0)],
        );
        add_model(
            &mut facts,
            &mut blobs,
            "mislabelled",
            "vendor/model",
            "native",
            &[("model.layers.0.weight", 2.0)],
        );
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let selector = ModelSelector::Source {
            source: "vendor/model",
            quantization: "native",
        };

        // Selection is happy — it only knows they share a coordinate.
        assert_eq!(
            select_model_roots(&facts, &reader, selector).unwrap().len(),
            2
        );
        // The merge is where the wrong grouping fails, loudly.
        let error = index_keymap_for_selector(&facts, &reader, selector)
            .expect_err("a shadowed tensor must not load");
        assert!(error.to_string().contains("not shards of one component"));
    }

    #[test]
    fn an_exact_root_is_never_a_group_and_a_missing_coordinate_is_an_error() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let only = add_model(
            &mut facts,
            &mut blobs,
            "solo",
            "vendor/model",
            "native",
            &[("model.layers.0.weight", 1.0)],
        );
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        assert_eq!(
            select_model_roots(&facts, &reader, ModelSelector::Root(only.root)).unwrap(),
            vec![only.root]
        );
        assert!(
            select_model_roots(
                &facts,
                &reader,
                ModelSelector::Source {
                    source: "vendor/absent",
                    quantization: "native",
                }
            )
            .is_err()
        );

        let selected =
            SelectedModelIndex::from_graph(&facts, reader, ModelSelector::Root(only.root)).unwrap();
        assert_eq!(selected.roots(), &[only.root]);
        assert_eq!(selected.single_root(), Some(only.root));
    }

    /// The gemma-3-1b shape: one component, two files, layers INTERLEAVED
    /// rather than split at a boundary. Layer 17 has tensors in both shards
    /// while layer 22 sits only in the first — which is exactly the case a
    /// merge keyed on layer ranges would get wrong.
    #[test]
    fn interleaved_shards_of_one_component_merge_by_tensor_name() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let first = add_model(
            &mut facts,
            &mut blobs,
            "model-00001-of-00002.safetensors",
            "google/gemma-3-1b",
            "native",
            &[
                ("model.layers.17.self_attn.q_proj.weight", 1.0),
                ("model.layers.22.mlp.up_proj.weight", 2.0),
            ],
        );
        let second = add_model(
            &mut facts,
            &mut blobs,
            "model-00002-of-00002.safetensors",
            "google/gemma-3-1b",
            "native",
            &[
                ("model.layers.17.post_attention_layernorm.weight", 3.0),
                ("model.layers.59.mlp.down_proj.weight", 4.0),
            ],
        );

        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        // Neither shard alone is the component.
        assert_eq!(
            index_keymap_for_root(&facts, &reader, first.root)
                .unwrap()
                .len(),
            2
        );

        let merged = index_keymap_for_roots(&facts, &reader, &[first.root, second.root]).unwrap();
        let mut names: Vec<&String> = merged.keys().collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "model.layers.17.post_attention_layernorm.weight",
                "model.layers.17.self_attn.q_proj.weight",
                "model.layers.22.mlp.up_proj.weight",
                "model.layers.59.mlp.down_proj.weight",
            ]
        );
        // The values follow their own shard, so the merge is not silently
        // taking one root's leaf for a name the other owns.
        assert_eq!(
            merged["model.layers.17.self_attn.q_proj.weight"].to_f32(),
            vec![1.0]
        );
        assert_eq!(
            merged["model.layers.17.post_attention_layernorm.weight"].to_f32(),
            vec![3.0]
        );

        // Argument order is not observable.
        let reversed = index_keymap_for_roots(&facts, &reader, &[second.root, first.root]).unwrap();
        assert_eq!(merged.len(), reversed.len());
        for (name, leaf) in &merged {
            assert_eq!(reversed[name].to_f32(), leaf.to_f32());
        }

        let selected =
            SelectedModelIndex::from_roots(&facts, reader, [second.root, first.root, second.root])
                .unwrap();
        let mut expected_roots = vec![first.root, second.root];
        expected_roots.sort();
        assert_eq!(selected.roots(), expected_roots);
    }

    /// A name in two roots means they are not shards of one component. Taking
    /// either leaf would hide that, so it is an error naming both roots.
    #[test]
    fn roots_sharing_a_tensor_name_are_not_shards() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let first = add_model(
            &mut facts,
            &mut blobs,
            "shard-a",
            "vendor/model",
            "native",
            &[("model.layers.0.weight", 1.0)],
        );
        let second = add_model(
            &mut facts,
            &mut blobs,
            "shard-b",
            "vendor/model",
            "native",
            &[("model.layers.0.weight", 2.0)],
        );

        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let error = index_keymap_for_roots(&facts, &reader, &[first.root, second.root])
            .expect_err("a shared tensor name must not merge");
        let message = error.to_string();
        assert!(
            message.contains("model.layers.0.weight"),
            "error names the colliding tensor: {message}"
        );
        assert!(
            message.contains("not shards of one component"),
            "error says what the collision means: {message}"
        );

        // The report is deterministic: roots are visited in sorted order, so
        // the same two roots produce the same message whichever way the caller
        // passes them. Without that, a collision would be reported differently
        // run to run and would look like two distinct problems.
        let reversed = index_keymap_for_roots(&facts, &reader, &[second.root, first.root])
            .expect_err("a shared tensor name must not merge either way");
        assert_eq!(message, reversed.to_string());

        let (low, high) = if first.root < second.root {
            (first.root, second.root)
        } else {
            (second.root, first.root)
        };
        assert!(
            message.contains(&format!("root {low} and root {high}")),
            "error names the roots in sorted order: {message}"
        );
    }

    #[test]
    fn indexing_a_component_from_zero_roots_is_an_error() {
        let facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        assert!(index_keymap_for_roots(&facts, &reader, &[]).is_err());
        assert!(SelectedModelIndex::from_roots(&facts, reader, std::iter::empty::<Id>()).is_err());
    }

    /// One root passed twice is one root, not a self-collision.
    #[test]
    fn a_repeated_root_is_deduplicated_rather_than_colliding() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let only = add_model(
            &mut facts,
            &mut blobs,
            "solo",
            "vendor/model",
            "native",
            &[("model.layers.0.weight", 1.0)],
        );
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let merged = index_keymap_for_roots(&facts, &reader, &[only.root, only.root]).unwrap();
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn duplicate_tensor_names_are_errors() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let model = add_model(
            &mut facts,
            &mut blobs,
            "duplicate",
            "org/duplicate",
            "native",
            &[("weight", 1.0), ("weight", 2.0)],
        );
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let error = match index_keymap_for_root(&facts, &reader, model.root) {
            Ok(_) => panic!("duplicate tensor name was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("duplicate tensor name"), "{error}");
    }

    #[test]
    fn model_aliases_select_the_same_root() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let model = add_model(
            &mut facts,
            &mut blobs,
            "primary",
            "org/model",
            "native",
            &[("weight", 1.0)],
        );
        let alias = blobs
            .put::<blobencodings::UTF8String, _>("alias".to_string())
            .unwrap();
        facts +=
            entity! { ExclusiveId::force_ref(&model.root) @ attrs::model_name: alias }.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        for name in ["primary", "alias"] {
            assert_eq!(
                select_model_root(&facts, &reader, ModelSelector::Name(name)).unwrap(),
                model.root
            );
        }
    }

    #[test]
    fn model_selection_ignores_unreadable_labels_and_non_model_members() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let model = add_model(
            &mut facts,
            &mut blobs,
            "model",
            "org/model",
            "native",
            &[("weight", 1.0)],
        );
        let absent: Blob<blobencodings::UTF8String> = "unavailable label".to_owned().to_blob();
        facts += entity! { ExclusiveId::force_ref(&model.root) @
            attrs::model_name: absent.get_handle(), attrs::source: absent.get_handle(),
        }
        .into_facts();
        // Config sequences also use member, but do not contain model weights.
        let child = fucid();
        facts += entity! { attrs::member: &child }.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        for selector in [
            ModelSelector::Only,
            ModelSelector::Name("model"),
            ModelSelector::Source {
                source: "org/model",
                quantization: "native",
            },
        ] {
            assert_eq!(
                select_model_root(&facts, &reader, selector).unwrap(),
                model.root
            );
        }
    }

    /// The model piles that exist today hold the two-blob form, and selection
    /// must keep reading them. Same graph, same keymap, one leaf form apart.
    #[test]
    fn a_two_blob_pile_indexes_the_same_as_a_typed_one() {
        let mut typed_facts = TribleSet::new();
        let mut typed_blobs = MemoryBlobStore::new();
        let typed = add_model_as(
            &mut typed_facts,
            &mut typed_blobs,
            LeafForm::Typed,
            "m",
            "org/m",
            "native",
            &[("a.weight", 1.5), ("b.weight", -2.5)],
        );
        let typed_reader = SnapshotSource::snapshot(&mut typed_blobs).unwrap();

        let mut old_facts = TribleSet::new();
        let mut old_blobs = MemoryBlobStore::new();
        let old = add_model_as(
            &mut old_facts,
            &mut old_blobs,
            LeafForm::TwoBlob,
            "m",
            "org/m",
            "native",
            &[("a.weight", 1.5), ("b.weight", -2.5)],
        );
        let old_reader = SnapshotSource::snapshot(&mut old_blobs).unwrap();

        let typed_map = index_keymap_for_root(&typed_facts, &typed_reader, typed.root).unwrap();
        let old_map = index_keymap_for_root(&old_facts, &old_reader, old.root).unwrap();
        assert_eq!(typed_map.len(), 2);
        assert_eq!(old_map.len(), 2);
        for name in ["a.weight", "b.weight"] {
            assert_eq!(typed_map[name].to_f32_shape(), old_map[name].to_f32_shape());
            assert_eq!(typed_map[name].elem(), old_map[name].elem());
        }
        // The roots differ, and that is correct: the two forms are different
        // bytes, so they are different content addresses.
        assert_ne!(typed.root, old.root);
    }

    #[cfg(feature = "tokenizer")]
    const WORDPIECE: &str = r###"{
      "added_tokens": [],
      "normalizer": {"type": "BertNormalizer", "clean_text": true,
                     "handle_chinese_chars": true, "strip_accents": null,
                     "lowercase": true},
      "pre_tokenizer": {"type": "BertPreTokenizer"},
      "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "hello": 1}}
    }"###;

    #[cfg(feature = "tokenizer")]
    #[test]
    fn tokenizer_labels_are_annotations_and_explicit_roots_need_no_name() {
        let mut blobs = MemoryBlobStore::new();
        let first =
            crate::tokenizer::save_tokenizer_json(WORDPIECE.as_bytes(), "first-label", &mut blobs)
                .unwrap();
        let alias = crate::tokenizer::save_tokenizer_json(
            WORDPIECE.as_bytes(),
            "another-label",
            &mut blobs,
        )
        .unwrap();
        let root = first.root().unwrap();
        assert_eq!(Some(root), alias.root());
        let facts = first.into_facts() + alias.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        for label in ["first-label", "another-label"] {
            assert_eq!(
                select_tokenizer_root(&facts, &reader, TokenizerSelector::Name(label)).unwrap(),
                root
            );
        }

        let names: TribleSet = facts
            .iter()
            .filter(|fact| fact.a() == &attrs::model_name.id())
            .copied()
            .collect();
        let unnamed = facts.difference(&names);
        assert_eq!(
            select_tokenizer_root(&unnamed, &reader, TokenizerSelector::Root(root)).unwrap(),
            root
        );
        let tokenizer =
            load_tokenizer_from_graph(&unnamed, &reader, TokenizerSelector::Only).unwrap();
        assert_eq!(tokenizer.token_to_id("hello"), Some(1));
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn tokenizer_name_selection_disambiguates_consolidated_graphs() {
        let mut blobs = MemoryBlobStore::new();
        let alpha =
            crate::tokenizer::save_tokenizer_json(WORDPIECE.as_bytes(), "org/alpha", &mut blobs)
                .unwrap();
        let alpha_root = alpha.root().unwrap();
        let beta_json = WORDPIECE.replace("\"hello\": 1", "\"hello\": 1, \"world\": 2");
        let beta =
            crate::tokenizer::save_tokenizer_json(beta_json.as_bytes(), "org/beta", &mut blobs)
                .unwrap();
        let beta_root = beta.root().unwrap();
        let mut facts = alpha.into_facts();
        facts += beta.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        assert!(select_tokenizer_root(&facts, &reader, TokenizerSelector::Only).is_err());
        assert_eq!(
            select_tokenizer_root(&facts, &reader, TokenizerSelector::Name("org/alpha")).unwrap(),
            alpha_root
        );
        assert_eq!(
            select_tokenizer_root(&facts, &reader, TokenizerSelector::Root(beta_root)).unwrap(),
            beta_root
        );
        let tokenizer =
            load_tokenizer_from_graph(&facts, &reader, TokenizerSelector::Name("org/beta"))
                .unwrap();
        assert_eq!(tokenizer.token_to_id("hello"), Some(1));
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn duplicate_tokenizer_functional_edge_is_an_error() {
        let mut blobs = MemoryBlobStore::new();
        let tokenizer = crate::tokenizer::save_tokenizer_json(
            WORDPIECE.as_bytes(),
            "org/tokenizer",
            &mut blobs,
        )
        .unwrap();
        let root = tokenizer.root().unwrap();
        let mut facts = tokenizer.into_facts();
        let other = entity! { _ @ metadata::tag: metadata::KIND_MULTI };
        let other_root = other.root().unwrap();
        facts += other.into_facts();
        facts += entity! { ExclusiveId::force_ref(&root) @
            crate::tokenizer::attrs::normalizer: &other_root
        }
        .into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        let error = load_tokenizer_from_graph(&facts, &reader, TokenizerSelector::Only)
            .unwrap_err()
            .to_string();
        assert!(error.contains("more than one attrs::normalizer"), "{error}");
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn duplicate_vocab_entry_functional_field_is_an_error() {
        let mut blobs = MemoryBlobStore::new();
        let tokenizer = crate::tokenizer::save_tokenizer_json(
            WORDPIECE.as_bytes(),
            "org/tokenizer",
            &mut blobs,
        )
        .unwrap();
        let root = tokenizer.root().unwrap();
        let mut facts = tokenizer.into_facts();
        let entry = find!(
            (entry: Id),
            pattern!(&facts, [{ root @ crate::tokenizer::attrs::vocab: ?entry }])
        )
        .next()
        .unwrap()
        .0;
        facts += entity! { ExclusiveId::force_ref(&entry) @
            crate::tokenizer::attrs::token_id: 99_u64
        }
        .into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

        let error = load_tokenizer_from_graph(&facts, &reader, TokenizerSelector::Only)
            .unwrap_err()
            .to_string();
        assert!(error.contains("more than one token_id"), "{error}");
    }
}
