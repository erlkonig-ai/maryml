//! Read an imported Inkling 2 Mini root out of a model pile: given the pile,
//! the exact model collection and the model root, and nothing else.
//!
//! [`MiniRoot::open`] resolves every component the importer
//! ([`super::import`]) writes, and refuses when any is missing or ambiguous:
//! the main root's tensor members, the drafter root (a member of the main
//! root, labelled [`super::import::DRAFTER`]) and its tensors, the root's
//! parts (tokenizer graph, `config.json` facts, chat template), every
//! document labelled under the root, and the source manifest. Tensors are
//! views over the pile mapping, so opening costs headers and handles, not
//! weights.
//!
//! [`MiniRoot::tensor`] answers a checkpoint name with the stored dtype,
//! shape and bytes, which are the source's own. [`MiniRoot::reassemble`]
//! rebuilds a safetensors file from its stored header and leaves, so a
//! reader can hash it against the release digest without the source.

use super::release::{self as import, DRAFTER, MANIFEST, MODEL_FILE, MTP_FILE};
use super::layout::Dtype;
use crate::format::attrs;
use crate::leaf::{Elem, Leaf};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use triblespace::core::blob::Blob;
use triblespace::core::collection::CollectionHandle;
use triblespace::core::metadata;
use triblespace::core::repo::pile::PileSnapshot;
use triblespace::prelude::*;

type TextHandle = Inline<inlineencodings::Handle<blobencodings::UTF8String>>;

/// One imported Mini model, resolved.
pub struct MiniRoot {
    store: PileSnapshot,
    facts: TribleSet,
    collection: CollectionHandle,
    root: Id,
    drafter: Id,
    model: HashMap<String, Leaf>,
    mtp: HashMap<String, Leaf>,
    documents: BTreeMap<String, TextHandle>,
    parts: crate::ingest::RootParts,
    manifest: Id,
}

fn name_handle(name: &str) -> TextHandle {
    let blob: Blob<blobencodings::UTF8String> = name.to_owned().to_blob();
    blob.get_handle()
}

/// The asset labelled `name` under `root`; exactly one, or an error.
fn labelled(facts: &TribleSet, root: Id, name: &str) -> Result<Id> {
    let wanted = name_handle(name);
    let found: BTreeSet<Id> = find!(
        (asset: Id),
        pattern!(facts, [{ ?asset @ metadata::name: wanted, attrs::model_root: root }])
    )
    .map(|(a,)| a)
    .collect();
    match found.len() {
        1 => Ok(*found.iter().next().unwrap()),
        0 => bail!("model root {root:X} has no asset labelled {name:?}"),
        n => bail!("model root {root:X} has {n} assets labelled {name:?}"),
    }
}

/// `name -> leaf` for the tensor members of `root`. A member that is not a
/// tensor (the drafter root under the main root) is left alone.
fn tensors(facts: &TribleSet, store: &PileSnapshot, root: Id) -> Result<HashMap<String, Leaf>> {
    let mut map = HashMap::new();
    for (path, weight) in find!(
        (path: TextHandle, weight: Id),
        pattern!(facts, [
            { root @ attrs::member: _?member },
            { _?member @ attrs::safetensor_path: ?path, attrs::weight: ?weight },
        ])
    ) {
        let name: anybytes::View<str> =
            store.get(path).map_err(|e| anyhow::anyhow!("tensor name: {e}"))?;
        let leaf = crate::leaf::resolve(facts, store, weight)?
            .with_context(|| format!("{}: no readable leaf", &*name))?;
        ensure!(map.insert(name.to_string(), leaf).is_none(), "{} named twice under {root:X}", &*name);
    }
    ensure!(!map.is_empty(), "root {root:X} has no tensor members");
    Ok(map)
}

/// The safetensors spelling of a stored leaf's element.
pub fn dtype_of(elem: Elem) -> Option<Dtype> {
    match elem {
        Elem::Bf16 => Some(Dtype::Bf16),
        Elem::F32 => Some(Dtype::F32),
        Elem::U8 => Some(Dtype::U8),
        Elem::E4m3 => Some(Dtype::F8E4M3),
        Elem::F16 | Elem::Nvfp4 => None,
    }
}

impl MiniRoot {
    /// Open `path` read-only and resolve `root` inside the model collection
    /// `collection`. Only those three are inputs.
    pub fn open(path: &Path, collection: CollectionHandle, root: Id) -> Result<Self> {
        let store = crate::persist::read_model_pile_snapshot_read_only(path)?;
        let opened = crate::model_collection::open_model_collection(&store, collection)?;
        let snapshot = crate::model_collection::snapshot_model_collection_for(&store, opened)
            .with_context(|| format!("{path:?}: model collection {collection:?}"))?;
        let facts = snapshot.facts().clone();
        ensure!(
            exists!(pattern!(&facts, [{ root @ attrs::member: _?m }])),
            "model root {root:X} has no members in collection {collection:?}"
        );

        let drafter = labelled(&facts, root, DRAFTER)?;
        ensure!(
            exists!(pattern!(&facts, [{ root @ attrs::member: drafter }])),
            "the drafter {drafter:X} is labelled under {root:X} but is not its member"
        );
        let model = tensors(&facts, &store, root)?;
        let mtp = tensors(&facts, &store, drafter)?;
        if let Some(name) = model.keys().find(|n| mtp.contains_key(*n)) {
            bail!("{name} is both a model and a drafter tensor");
        }

        let mut documents = BTreeMap::new();
        for (name, text) in find!(
            (name: TextHandle, text: TextHandle),
            pattern!(&facts, [{ _?asset @ attrs::model_root: root, metadata::name: ?name,
                crate::dataset::text: ?text }])
        ) {
            let name: anybytes::View<str> =
                store.get(name).map_err(|e| anyhow::anyhow!("document name: {e}"))?;
            if let Some(other) = documents.insert(name.to_string(), text)
                && other != text
            {
                bail!("two documents named {:?} under {root:X}", &*name);
            }
        }

        let parts = crate::selection::model_root_parts(&facts, root)?;
        ensure!(parts.tokenizer.is_some(), "root {root:X} has no tokenizer");
        ensure!(parts.config.is_some(), "root {root:X} has no config");
        ensure!(parts.chat_template.is_some(), "root {root:X} has no chat template");
        let manifest = labelled(&facts, root, MANIFEST)?;

        let this = MiniRoot { store, facts, collection, root, drafter, model, mtp, documents, parts, manifest };
        // The release's text files and both headers must all be here.
        for &(path, _, _) in import::RELEASE {
            if path != MODEL_FILE && path != MTP_FILE {
                this.document(path)?;
            }
        }
        for file in [MODEL_FILE, MTP_FILE] {
            this.document(&import::header_document(file))?;
        }
        Ok(this)
    }

    pub fn root(&self) -> Id {
        self.root
    }

    pub fn drafter_root(&self) -> Id {
        self.drafter
    }

    pub fn collection(&self) -> CollectionHandle {
        self.collection
    }

    pub fn facts(&self) -> &TribleSet {
        &self.facts
    }

    pub fn store(&self) -> &PileSnapshot {
        &self.store
    }

    /// Tensor counts: (model, drafter).
    pub fn counts(&self) -> (usize, usize) {
        (self.model.len(), self.mtp.len())
    }

    /// Every tensor name of one file, sorted.
    pub fn names(&self, file: &str) -> Result<Vec<&str>> {
        let map = self.map(file)?;
        let mut names: Vec<&str> = map.keys().map(String::as_str).collect();
        names.sort();
        Ok(names)
    }

    fn map(&self, file: &str) -> Result<&HashMap<String, Leaf>> {
        match file {
            MODEL_FILE => Ok(&self.model),
            MTP_FILE => Ok(&self.mtp),
            other => bail!("{other} is not one of the two safetensors files"),
        }
    }

    /// The stored leaf for a checkpoint tensor name, model or drafter.
    pub fn leaf(&self, name: &str) -> Option<&Leaf> {
        self.model.get(name).or_else(|| self.mtp.get(name))
    }

    /// A tensor by checkpoint name: its source dtype, shape and bytes.
    pub fn tensor(&self, name: &str) -> Result<(Dtype, Vec<usize>, &[u8]), String> {
        let leaf = self.leaf(name).ok_or_else(|| format!("no tensor named {name} under the model root"))?;
        let dtype = dtype_of(leaf.elem()).ok_or_else(|| format!("{name}: stored as {:?}", leaf.elem()))?;
        Ok((dtype, leaf.shape(), &leaf.payload()[..]))
    }

    /// A document under the root, by the file name it had in the release.
    pub fn document(&self, name: &str) -> Result<anybytes::View<str>> {
        let handle = self
            .documents
            .get(name)
            .with_context(|| format!("no document {name:?} under {:X}; it has {:?}", self.root, self.documents.keys().collect::<Vec<_>>()))?;
        self.store.get(*handle).map_err(|e| anyhow::anyhow!("document {name}: {e}"))
    }

    pub fn document_names(&self) -> Vec<&str> {
        self.documents.keys().map(String::as_str).collect()
    }

    /// `config.json`, from the root's config facts.
    pub fn config(&self) -> Result<serde_json::Value> {
        crate::jsonfacts::load_json(&self.facts, &self.store, self.parts.config.expect("checked at open"))
            .map_err(|e| anyhow::anyhow!("config facts: {e}"))
    }

    /// The chat template, verbatim, from the root's core.
    pub fn chat_template(&self) -> Result<String> {
        let v: anybytes::View<str> = self
            .store
            .get(self.parts.chat_template.expect("checked at open"))
            .map_err(|e| anyhow::anyhow!("chat template: {e}"))?;
        Ok(v.to_string())
    }

    /// The tokenizer graph's root, for `crate::tokenizer::build_tokenizer`.
    pub fn tokenizer_root(&self) -> Id {
        self.parts.tokenizer.expect("checked at open")
    }

    pub fn manifest(&self) -> Result<serde_json::Value> {
        crate::jsonfacts::load_json(&self.facts, &self.store, self.manifest)
            .map_err(|e| anyhow::anyhow!("manifest facts: {e}"))
    }

    /// Rebuild `file` (one of the two safetensors files) from its stored
    /// header and leaves, handing the bytes to `sink` in file order. Every
    /// header tensor must have a leaf of the same dtype and shape and the
    /// declared length, the ranges must tile the file, and every leaf must
    /// be used. Returns the byte count.
    pub fn reassemble(&self, file: &str, mut sink: impl FnMut(&[u8])) -> Result<u64> {
        let map = self.map(file)?;
        let header = self.document(&import::header_document(file))?;
        let parsed: serde_json::Value = serde_json::from_str(&header)?;
        let mut entries = Vec::new();
        for (name, t) in parsed.as_object().context("header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let offsets = t["data_offsets"].as_array().context("no data_offsets")?;
            let (a, b) = (offsets[0].as_u64().context("offset")?, offsets[1].as_u64().context("offset")?);
            entries.push((a, b, name.clone(), t["dtype"].as_str().unwrap_or("").to_owned(), t["shape"].clone()));
        }
        entries.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        ensure!(entries.len() == map.len(), "{file}: header has {} tensors, the root {}", entries.len(), map.len());
        sink(&(header.len() as u64).to_le_bytes());
        sink(header.as_bytes());
        let mut at = 0u64;
        for (a, b, name, dtype, shape) in entries {
            ensure!(a == at, "{file}:{name}: starts at {a}, expected {at}");
            let leaf = map.get(&name).with_context(|| format!("{file}:{name}: no leaf"))?;
            let stored = dtype_of(leaf.elem()).map(Dtype::name).unwrap_or("?");
            ensure!(stored == dtype, "{file}:{name}: header says {dtype}, leaf is {stored}");
            let shape: Vec<u64> = serde_json::from_value(shape)?;
            ensure!(shape == leaf.dims(), "{file}:{name}: header shape {shape:?}, leaf {:?}", leaf.dims());
            ensure!(leaf.payload().len() as u64 == b - a, "{file}:{name}: {} bytes, header {}", leaf.payload().len(), b - a);
            sink(&leaf.payload()[..]);
            at = b;
        }
        Ok(8 + header.len() as u64 + at)
    }
}
