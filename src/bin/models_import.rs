//! Import into an explicit existing model collection: Flint GGUF -> native typed
//! leaves on the host, or Breeze/Voxtral native piles -> device-gated NVFP4.
//!
//! Flint's source is GGUF, never opened as a pile. Other sources are existing
//! native model piles, opened read-only. The
//! destination is a pile holding the target collection's descriptor
//! (`trible pile collection init <PILE> models ...`), named by its exact
//! handle. The signer must be admitted by that descriptor's WRITE policy,
//! which is checked before anything is written.
//!
//! For Breeze/Voxtral the `nvfp4` root holds: every linear projection the runtime used to
//! quantize at load, stored as a `Tensor<NVFP4, 2>` leaf packed on the host by
//! `nvfp4::pack_nearest` and gated byte for byte against the device quantizer
//! on the same 16-bit values (Breeze: `Generator::export_projections`,
//! Voxtral: `fast::export_nvfp4_projections`, which stores the fused, folded
//! QKV and gate/up the folded lane multiplies). Every other tensor is copied
//! unchanged at its source precision. Configs, tokenizer and documents travel
//! with it, labelled as before.
//!
//!   models_import breeze --source <model.pile> --source-root HEX \
//!     --pile <dest.pile> --key <signing.key> --collection blake3:HEX
//!   models_import voxtral --source <voxtral.pile> --source-root HEX \
//!     --tekken <tekken.json> --pile <dest.pile> --key <signing.key> --collection blake3:HEX
//!   models_import flint --source <flint.gguf> \
//!     --pile <existing.pile> --key <signing.key> --collection blake3:HEX
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use mary::model_collection::{self, ModelCollection};
use std::path::{Path, PathBuf};
use std::time::Instant;
use triblespace::core::collection::CollectionHandle;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;
use triblespace::prelude::*;

#[derive(Parser)]
#[command(
    about = "Import Flint GGUF or repack an enabled native model into an explicit existing collection"
)]
struct Args {
    #[command(subcommand)]
    model: Model,
    /// Flint GGUF, or an existing native model pile for Breeze/Voxtral; read only.
    #[arg(long, global = true)]
    source: Option<PathBuf>,
    /// The source model root (32 hex): Breeze's native root, Voxtral's f16 root.
    #[arg(long, global = true)]
    source_root: Option<String>,
    /// Destination pile holding the target collection's descriptor.
    #[arg(long, global = true)]
    pile: Option<PathBuf>,
    /// Existing signing key the target collection's WRITE policy admits.
    #[arg(long, global = true)]
    key: Option<PathBuf>,
    /// Exact target collection descriptor handle (`blake3:HEX`).
    #[arg(long, global = true)]
    collection: Option<String>,
}

#[derive(Subcommand)]
enum Model {
    #[cfg(feature = "tokenizer")]
    Flint,
    #[cfg(feature = "breeze-cuda")]
    Breeze,
    #[cfg(feature = "voxtral-cuda")]
    Voxtral {
        /// The checkpoint's tekken.json, gated piece for piece against the
        /// tokenizer the source pile already holds.
        #[arg(long)]
        tekken: PathBuf,
    },
}

impl Args {
    fn check_source_kind(&self) -> Result<()> {
        match &self.model {
            #[cfg(feature = "tokenizer")]
            Model::Flint => ensure!(self.source_root.is_none(), "Flint --source is GGUF; --source-root is not used"),
            #[cfg(feature = "breeze-cuda")]
            Model::Breeze => ensure!(self.source_root.is_some(), "--source-root is required"),
            #[cfg(feature = "voxtral-cuda")]
            Model::Voxtral { .. } => ensure!(self.source_root.is_some(), "--source-root is required"),
        }
        Ok(())
    }
}

// Existing native-source repacking is absent from the host-only Flint build.
#[cfg(any(feature = "breeze-cuda", feature = "voxtral-cuda"))]
mod native {
use super::*;
use mary::{format::attrs, leaf::Leaf, nvfp4::StoredProjection};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use triblespace::core::{metadata, repo::pile::PileSnapshot};

/// The source pile's one observation and the explicit root being converted.
pub(super) struct Source {
    facts: TribleSet,
    store: PileSnapshot,
    root: Id,
    models: model_collection::ModelPileSnapshot,
}

pub(super) fn open_source(path: &Path, root: &str) -> Result<Source> {
    let root = Id::from_hex(root).context("--source-root must be a 32-hex id")?;
    let store = mary::persist::read_model_pile_snapshot_read_only(path)?;
    let models = model_collection::snapshot_model_collection_in(&store)
        .context("discover the source pile's model collection")?;
    ensure!(
        exists!(pattern!(models.facts(), [{ root @ attrs::member: _?member }])),
        "source root {root} has no members in {path:?}"
    );
    Ok(Source {
        facts: models.facts().clone(),
        store: models.store().clone(),
        root,
        models,
    })
}

/// `name -> leaf` for the tensor members of `root`. A member that is not a
/// tensor (Breeze's main root lists its codec root as a member) is not one of
/// its tensors and is left alone.
pub(super) fn tensors(
    facts: &TribleSet,
    store: &impl BlobStoreGet,
    root: Id,
) -> Result<HashMap<String, Leaf>> {
    let mut map = HashMap::new();
    for (path, weight) in find!(
        (path: Inline<inlineencodings::Handle<blobencodings::UTF8String>>, weight: Id),
        pattern!(facts, [
            { root @ attrs::member: _?member },
            { _?member @ attrs::safetensor_path: ?path, attrs::weight: ?weight },
        ])
    ) {
        let name: anybytes::View<str> = store
            .get(path)
            .map_err(|e| anyhow::anyhow!("tensor name: {e}"))?;
        let leaf = mary::leaf::resolve(facts, store, weight)?
            .with_context(|| format!("{}: no readable leaf", &*name))?;
        ensure!(
            map.insert(name.to_string(), leaf).is_none(),
            "{} named twice under {root}",
            &*name
        );
    }
    ensure!(!map.is_empty(), "root {root} has no tensor members");
    Ok(map)
}

fn text(
    pile: &mut Pile,
    value: &str,
) -> Result<Inline<inlineencodings::Handle<blobencodings::UTF8String>>> {
    pile.put::<blobencodings::UTF8String, _>(value.to_owned())
        .map_err(|e| anyhow::anyhow!("store text: {e}"))
}

/// Copy `index` into `pile` as model members, replacing the leaves named in
/// `stored` by their NVFP4 payloads and leaving out `dropped`. Returns the
/// members, their facts, and the bytes written for dense and packed leaves.
fn members(
    pile: &mut Pile,
    index: &HashMap<String, Leaf>,
    stored: &BTreeMap<String, StoredProjection>,
    dropped: &BTreeSet<String>,
) -> Result<(Vec<Id>, TribleSet, u64, u64)> {
    let mut names: Vec<&String> = index.keys().chain(stored.keys()).collect();
    names.sort();
    names.dedup();
    let kind = text(pile, "tensor")?;
    let (mut members, mut facts, mut dense, mut packed) =
        (Vec::new(), TribleSet::new(), 0u64, 0u64);
    for name in names {
        let leaf = if let Some(projection) = stored.get(name) {
            packed += projection.payload.len() as u64;
            mary::leaf::put_leaf(
                pile,
                mary::leaf::Elem::Nvfp4,
                &projection.shape,
                anybytes::Bytes::from_source(projection.payload.clone()),
                name,
            )?
        } else if dropped.contains(name) {
            continue;
        } else {
            let leaf = &index[name];
            dense += leaf.payload().len() as u64;
            mary::leaf::put_leaf(pile, leaf.elem(), leaf.dims(), leaf.payload().clone(), name)?
        };
        let weight = leaf.root().context("leaf root")?;
        facts += leaf.into_facts();
        let path = text(pile, name)?;
        let member =
            entity! { _ @ attrs::safetensor_path: path, attrs::kind: kind, attrs::weight: weight };
        members.push(member.root().context("member root")?);
        facts += member.into_facts();
    }
    Ok((members, facts, dense, packed))
}

/// The source root's provenance names (`model_name`), carried over.
fn provenance(source: &Source) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for (handle,) in find!(
        (handle: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
        pattern!(&source.facts, [{ source.root @ attrs::model_name: ?handle }])
    ) {
        let name: anybytes::View<str> = source
            .store
            .get(handle)
            .map_err(|e| anyhow::anyhow!("source provenance name: {e}"))?;
        names.push(name.to_string());
    }
    names.sort();
    Ok(names)
}

/// Name `asset` under `model_root` in `fragment`.
fn label(
    pile: &mut Pile,
    fragment: &mut Fragment,
    asset: Id,
    name: &str,
    model_root: Id,
) -> Result<()> {
    let name = text(pile, name)?;
    *fragment += entity! { ExclusiveId::force_ref(&asset) @ metadata::name: name, attrs::model_root: model_root };
    Ok(())
}

fn derivation(
    pile: &mut Pile,
    fragment: &mut Fragment,
    model_root: Id,
    source: &Source,
    source_path: &Path,
    projections: usize,
    packed: u64,
) -> Result<()> {
    let value = serde_json::json!({
        "derivation": "nvfp4 storage of the projections the runtime quantized at load",
        "source_pile": source_path.file_name().map(|n| n.to_string_lossy().into_owned()),
        "source_root": format!("{:X}", source.root),
        "projections": projections,
        "packed_bytes": packed,
        "packer": "mary nvfp4::pack_nearest, gated byte for byte against nn::nvfp4_gemv::Nvfp4::quantize on the same 16-bit values",
        "other_tensors": "copied unchanged at source precision",
    });
    let root = mary::jsonfacts::save_json(&value, pile, fragment.facts_mut())
        .map_err(|e| anyhow::anyhow!("derivation facts: {e}"))?;
    label(pile, fragment, root, "nvfp4-derivation", model_root)
}

#[cfg(feature = "breeze-cuda")]
pub(super) fn breeze(
    args: &Args,
    source: &Source,
    source_path: &Path,
    pile: &mut Pile,
) -> Result<(Id, Fragment, serde_json::Value)> {
    use mary::models::breeze::{
        REVISION, SOURCE,
        generator::Generator,
        load::{Artifacts, Assets},
    };
    let ids = Artifacts::from_root(&source.facts, source.root)?;
    let assets = Assets::from_frozen(&source.facts, &source.store, ids)?;
    let started = Instant::now();
    let exported = Generator::export_projections(
        &source.facts,
        &source.store,
        source.root,
        assets.config,
        &cubecl::cuda::CudaDevice { index: 0 },
    )?;
    let export_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "[breeze] {} projections packed and gated in {export_seconds:.1}s",
        exported.len()
    );
    let stored: BTreeMap<String, StoredProjection> =
        exported.into_iter().map(|p| (p.name.clone(), p)).collect();

    // The external codec: copied unchanged as its own native root.
    let codec_index = tensors(&source.facts, &source.store, ids.external_codec_root)?;
    let (codec_members, codec_facts, codec_bytes, _) =
        members(pile, &codec_index, &BTreeMap::new(), &BTreeSet::new())?;
    let codec = mary::ingest::build_model_root(
        pile,
        &format!("{SOURCE}@{REVISION}/audio_tokenizer"),
        mary::persist::QUANTIZATION_NATIVE,
        codec_members,
        codec_facts,
        &["audio_tokenizer/model.safetensors".to_owned()],
    )
    .map_err(|e| anyhow::anyhow!("codec root: {e}"))?;
    let codec_root = codec.root().context("codec root")?;

    let index = tensors(&source.facts, &source.store, source.root)?;
    for name in stored.keys() {
        ensure!(
            index.contains_key(name),
            "exported {name} is not a source tensor"
        );
    }
    let (mut main_members, mut facts, dense, packed) =
        members(pile, &index, &stored, &BTreeSet::new())?;
    // The source root also lists the codec root as a member; it is rebuilt
    // above, not copied as a tensor.
    main_members.push(codec_root);
    facts += codec.into_facts();
    let mut fragment = mary::ingest::build_model_root(
        pile,
        &format!("{SOURCE}@{REVISION}"),
        mary::persist::QUANTIZATION_NVFP4,
        main_members,
        facts,
        &provenance(source)?,
    )
    .map_err(|e| anyhow::anyhow!("model root: {e}"))?;
    let model_root = fragment.root().context("model root")?;

    for (asset, name) in [
        (ids.config_root, "config.json"),
        (
            ids.external_codec_config_root,
            "audio_tokenizer/config.json",
        ),
    ] {
        let value = mary::jsonfacts::load_json(&source.facts, &source.store, asset)
            .map_err(|e| anyhow::anyhow!("source {name}: {e}"))?;
        let root = mary::jsonfacts::save_json(&value, pile, fragment.facts_mut())
            .map_err(|e| anyhow::anyhow!("{name} facts: {e}"))?;
        label(pile, &mut fragment, root, name, model_root)?;
    }
    // The native import's provenance manifest, a JSON document, carried over.
    let wanted: Blob<blobencodings::UTF8String> = "native-source-manifest".to_owned().to_blob();
    let wanted = wanted.get_handle();
    let manifests: BTreeSet<Id> = find!(
        (asset: Id),
        pattern!(&source.facts, [{ ?asset @
            attrs::model_root: source.root,
            metadata::name: wanted,
            metadata::tag: mary::jsonfacts::ty::JSON_OBJECT,
        }])
    )
    .map(|(asset,)| asset)
    .collect();
    for manifest in manifests {
        let value = mary::jsonfacts::load_json(&source.facts, &source.store, manifest)
            .map_err(|e| anyhow::anyhow!("source manifest: {e}"))?;
        let root = mary::jsonfacts::save_json(&value, pile, fragment.facts_mut())
            .map_err(|e| anyhow::anyhow!("manifest facts: {e}"))?;
        label(
            pile,
            &mut fragment,
            root,
            "native-source-manifest",
            model_root,
        )?;
    }
    label(
        pile,
        &mut fragment,
        codec_root,
        "audio_tokenizer",
        model_root,
    )?;

    // Every text document under the source root, by its name.
    let mut documents = 0usize;
    for (asset, name, handle) in find!(
        (asset: Id, name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>, handle: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
        pattern!(&source.facts, [{ ?asset @ attrs::model_root: source.root, metadata::name: ?name, mary::dataset::text: ?handle }])
    ) {
        let _ = asset;
        let name: anybytes::View<str> = source
            .store
            .get(name)
            .map_err(|e| anyhow::anyhow!("document name: {e}"))?;
        let body: anybytes::View<str> = source
            .store
            .get(handle)
            .map_err(|e| anyhow::anyhow!("document {}: {e}", &*name))?;
        let body = pile.put::<blobencodings::UTF8String, _>(body.to_string())?;
        let document = entity! { _ @ mary::dataset::text: body };
        let id = document.root().context("document root")?;
        fragment += document;
        label(pile, &mut fragment, id, &name, model_root)?;
        documents += 1;
    }
    ensure!(documents > 0, "the source root carries no documents");
    derivation(
        pile,
        &mut fragment,
        model_root,
        source,
        source_path,
        stored.len(),
        packed,
    )?;
    let _ = args;
    Ok((
        model_root, fragment,
        serde_json::json!({
            "model": "breeze", "model_root": format!("{model_root:X}"), "codec_root": format!("{codec_root:X}"),
            "projections": stored.len(), "packed_bytes": packed, "dense_bytes": dense, "codec_bytes": codec_bytes,
            "documents": documents, "export_seconds": export_seconds,
        }),
    ))
}

#[cfg(feature = "voxtral-cuda")]
pub(super) fn voxtral(
    tekken: &Path,
    source: &Source,
    source_path: &Path,
    pile: &mut Pile,
) -> Result<(Id, Fragment, serde_json::Value)> {
    use mary::models::voxtral::{
        SOURCE, fast,
        tokenizer::{Tekken, TekkenSource},
    };
    let index = tensors(&source.facts, &source.store, source.root)?;
    for (name, leaf) in &index {
        ensure!(
            leaf.elem() == mary::leaf::Elem::F16,
            "Voxtral source tensor {name:?} is not F16"
        );
    }
    let started = Instant::now();
    let loader = mary::nn::weight_loader::WeightLoader::Typed(index.clone());
    let device = mary::nn::backend::hear::Device::default();
    let mut stored = BTreeMap::new();
    let mut dropped = BTreeSet::new();
    fast::export_nvfp4_projections(&loader, &device, |projection, replaces| {
        dropped.extend(replaces);
        stored.insert(projection.name.clone(), projection);
        Ok(())
    })?;
    drop(loader);
    let export_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "[voxtral] {} projections packed and gated in {export_seconds:.1}s",
        stored.len()
    );
    for name in &dropped {
        ensure!(
            index.contains_key(name),
            "replaced {name} is not a source tensor"
        );
    }
    let (members, facts, dense, packed) = members(pile, &index, &stored, &dropped)?;
    let mut fragment = mary::ingest::build_model_root(
        pile,
        SOURCE,
        mary::persist::QUANTIZATION_NVFP4,
        members,
        facts,
        &provenance(source)?,
    )
    .map_err(|e| anyhow::anyhow!("model root: {e}"))?;
    let model_root = fragment.root().context("model root")?;

    // The tokenizer from its source file, gated against the one the source
    // pile holds: every id the file gives bytes decodes to exactly those.
    let tekken_source = TekkenSource::parse(&std::fs::read(tekken)?)?;
    let held = Tekken::from_snapshot(&source.models)?;
    for id in tekken_source.piece_ids() {
        ensure!(
            tekken_source.piece(id) == Some(held.piece(id)),
            "Tekken piece {id} differs between {tekken:?} and the source pile"
        );
    }
    ensure!(
        held.len() == tekken_source.piece_ids().len(),
        "Tekken piece counts differ"
    );
    fragment += tekken_source.to_fragment(SOURCE, pile)?;
    derivation(
        pile,
        &mut fragment,
        model_root,
        source,
        source_path,
        stored.len(),
        packed,
    )?;
    Ok((
        model_root, fragment,
        serde_json::json!({
            "model": "voxtral", "model_root": format!("{model_root:X}"), "projections": stored.len(),
            "replaced_tensors": dropped.len(), "packed_bytes": packed, "dense_bytes": dense,
            "tekken_pieces": held.len(), "export_seconds": export_seconds,
        }),
    ))
}

} // native

#[cfg(feature = "tokenizer")]
fn flint(source_path: &Path, pile: &mut Pile) -> Result<(Id, Fragment, serde_json::Value)> {
    use mary::formats::gguf::index::{SelectedMetadata, MetadataLimits};
    use mary::models::flint::{weights::GgufWeights, tokenizer, pile::import};
    let mut source = GgufWeights::open_block(source_path, 0)?;
    let (identity, metadata) = SelectedMetadata::open(source_path, tokenizer::METADATA_KEYS, MetadataLimits::TOKENIZER)?;
    ensure!(identity == source.file.identity, "tokenizer and weights are different source observations");
    source.file.check_identity()?;
    // full() validates all block configs and the actual tokenizer before its
    // first leaf write; the destination writer has already been admitted.
    let (fragment, summary) = import::full(&mut source, pile, &metadata.entries)?;
    source.file.check_identity()?;
    let root = fragment.root().context("Flint model root")?;
    Ok((root, fragment, serde_json::json!({
        "model": "flint", "source_identity": source.file.identity,
        "source_gguf": source_path.file_name().map(|n| n.to_string_lossy().into_owned()),
        "whole_model_freshly_hashed": false, "metadata_end": source.file.index.metadata_end,
        "tokenizer_metadata_end": metadata.end, "import": summary,
        "layout": "native BF16/F32/NVFP4 Tensor leaves; indexed rank2 experts; separate post-matmul scales",
    })))
}

/// Existing destination transaction, factored so synthetic tests can verify
/// that a refused writer never reaches the leaf-producing callback. Not a
/// descriptor creator, transfer operation or alternative import framework.
fn publish_into(dest: &Path, key: &ed25519_dalek::SigningKey, handle: CollectionHandle,
    import: impl FnOnce(&mut Pile) -> Result<(Id, Fragment, serde_json::Value)>) -> Result<(Id, serde_json::Value)> {
    ensure!(std::fs::metadata(dest).with_context(|| format!("existing destination {dest:?}"))?.is_file(),
        "destination must be an existing pile file");
    let mut pile = Pile::open(&dest).map_err(|e| anyhow::anyhow!("open {dest:?}: {e:?}"))?;
    let result = (|| -> Result<(Id, serde_json::Value)> {
        pile.refresh().context("refresh destination")?;
        let collection: ModelCollection = {
            let snapshot = pile.snapshot().context("freeze destination")?;
            model_collection::open_model_collection(&snapshot, handle)?
        };
        // Before any weight is written: a refused signer must cost nothing.
        model_collection::require_model_writer(&mut pile, key, collection)?;
        // The model's root is explicit: native fragments can export additional
        // document/tokenizer roots without changing the model being published.
        let (root, fragment, mut receipt) = import(&mut pile)?;
        let commit =
            model_collection::publish_model_fragment_into(&mut pile, key, collection, fragment)?;
        receipt["model_root"] = serde_json::json!(format!("{root:X}"));
        receipt["collection"] = serde_json::json!(handle.raw.iter().map(|b| format!("{b:02x}")).collect::<String>());
        receipt["commit"] = serde_json::json!(
            triblespace::core::collection::CollectionRecord::Commit(commit)
                .fingerprint()
                .to_string()
        );
        Ok((root, receipt))
    })();
    let close = pile
        .close()
        .map_err(|e| anyhow::anyhow!("close {dest:?}: {e:?}"));
    match (result, close) {
        (Ok(receipt), Ok(())) => Ok(receipt),
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    args.check_source_kind()?;
    let need =
        |v: &Option<PathBuf>, n: &str| v.clone().with_context(|| format!("--{n} is required"));
    let source_path = need(&args.source, "source")?;
    let dest = need(&args.pile, "pile")?;
    let key = signing_key_file::load_existing(&need(&args.key, "key")?).context("load signing key")?;
    let handle = model_collection::parse_collection_handle(args.collection.as_deref().context("--collection is required")?)?;
    let started = Instant::now();
    let size_before = std::fs::metadata(&dest).with_context(|| format!("{dest:?}"))?.len();
    let (model_root, mut receipt) = publish_into(&dest, &key, handle, |pile| match &args.model {
        #[cfg(feature = "tokenizer")]
        Model::Flint => flint(&source_path, pile),
        #[cfg(feature = "breeze-cuda")]
        Model::Breeze => {
            let source = native::open_source(&source_path, args.source_root.as_deref().context("--source-root is required")?)?;
            native::breeze(&args, &source, &source_path, pile)
        },
        #[cfg(feature = "voxtral-cuda")]
        Model::Voxtral { tekken } => {
            let source = native::open_source(&source_path, args.source_root.as_deref().context("--source-root is required")?)?;
            native::voxtral(tekken, &source, &source_path, pile)
        },
    })?;

    // Read it back the way the runtime will: the collection by handle, Flint's
    // explicit root, or the existing native readers' model labels.
    let store = mary::persist::read_model_pile_snapshot_read_only(&dest)?;
    let collection = model_collection::open_model_collection(&store, handle)?;
    let models = model_collection::snapshot_model_collection_for(&store, collection)?;
    match &args.model {
        #[cfg(feature = "tokenizer")]
        Model::Flint => {
            let mut weights = mary::models::flint::pile::PileWeights::from_snapshot(models, model_root)?;
            for block in 0..49 { weights.select_block(block)?; }
            weights.tokenizer()?;
            receipt["readback"] = serde_json::json!("all49 block bindings and root-linked tokenizer in explicit frozen collection");
            receipt["support_handles"] = serde_json::json!(weights.support_handles());
        },
        #[cfg(feature = "breeze-cuda")]
        Model::Breeze => {
            let ids = mary::models::breeze::load::Artifacts::select(
                models.facts(),
                models.store(),
                mary::persist::QUANTIZATION_NVFP4,
            )?;
            mary::models::breeze::load::Assets::from_frozen(models.facts(), models.store(), ids)?;
            let index = native::tensors(models.facts(), models.store(), ids.model_root)?;
            receipt["readback_nvfp4_leaves"] = serde_json::json!(
                index
                    .values()
                    .filter(|l| l.elem() == mary::leaf::Elem::Nvfp4)
                    .count()
            );
            receipt["readback_tensors"] = serde_json::json!(index.len());
        }
        #[cfg(feature = "voxtral-cuda")]
        Model::Voxtral { .. } => {
            let model = mary::models::voxtral::VoxtralNvfp4::from_snapshot(&models)?;
            mary::models::voxtral::tokenizer::Tekken::from_snapshot(&models)?;
            receipt["readback_nvfp4_leaves"] = serde_json::json!(model.nvfp4_count());
        }
    }
    let size_after = std::fs::metadata(&dest)?.len();
    receipt["pile_bytes_added"] = serde_json::json!(size_after - size_before);
    receipt["seconds"] = serde_json::json!(started.elapsed().as_secs_f64());
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mary::{format::attrs, leaf};
    use triblespace::core::{metadata, collection::{AdmissionPolicy, CollectionPolicy},
        blob::encodings::tensor::elements::F32};

    // One-operation synthetic fixture; public deterministic key, no key file.
    fn destination() -> Result<(PathBuf, ed25519_dalek::SigningKey, CollectionHandle, Id)> {
        let path=std::env::temp_dir().join(format!("mary-models-import-{}.pile",fucid().id));
        std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
        let mut pile=Pile::open(&path).map_err(|e|anyhow::anyhow!("{e}"))?;
        let key=ed25519_dalek::SigningKey::from_bytes(&[71;32]);
        let policy=CollectionPolicy::new(AdmissionPolicy::direct(key.verifying_key()),AdmissionPolicy::direct(key.verifying_key()));
        let collection=pile.collection("models",policy)?;
        let existing=entity! {metadata::name:"unrelated existing model"};
        let root=existing.root().context("fixture root")?;
        model_collection::publish_model_fragment_into(&mut pile,&key,collection,existing)?;
        pile.close().map_err(|e|anyhow::anyhow!("{e}"))?;
        Ok((path,key,collection.handle(),root))
    }

    #[cfg(feature="tokenizer")]
    #[test]
    fn flint_cli_uses_gguf_without_source_root() -> Result<()> {
        let args=Args::try_parse_from(["models_import","flint","--source","input.gguf",
            "--pile","existing.pile","--key","existing.key","--collection",&"ab".repeat(32)])?;
        args.check_source_kind()?;
        assert!(args.source_root.is_none());
        assert_eq!(args.source.as_deref(),Some(Path::new("input.gguf")));
        let wrong=Args::try_parse_from(["models_import","flint","--source-root","0123456789abcdef0123456789abcdef"])?;
        assert!(wrong.check_source_kind().is_err());
        #[cfg(not(feature="breeze-cuda"))]
        assert!(Args::try_parse_from(["models_import","breeze"]).is_err());
        #[cfg(not(feature="voxtral-cuda"))]
        assert!(Args::try_parse_from(["models_import","voxtral","--tekken","tekken.json"]).is_err());
        Ok(())
    }

    #[test]
    fn refused_writer_never_reaches_leaf_writes_and_missing_pile_is_not_created() -> Result<()> {
        let (path,key,handle,_)=destination()?;
        let before=std::fs::metadata(&path)?.len();
        let outsider=ed25519_dalek::SigningKey::from_bytes(&[72;32]);
        let mut called=false;
        let error=publish_into(&path,&outsider,handle,|_| {called=true;anyhow::bail!("unreachable leaf writes")}).unwrap_err();
        assert!(!called);assert!(error.to_string().contains("not admitted"),"{error:#}");
        assert_eq!(std::fs::metadata(&path)?.len(),before);
        let absent=path.with_extension("absent");
        assert!(publish_into(&absent,&key,handle,|_|panic!("missing destination reached importer")).is_err());
        assert!(!absent.exists());
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn admitted_publication_reopens_explicit_collection_and_preserves_existing_facts() -> Result<()> {
        let (path,key,handle,existing)=destination()?;
        let (root,receipt)=publish_into(&path,&key,handle,|pile| {
            let blob=leaf::leaf_blob::<F32,1>([1],anybytes::Bytes::from_source(1.25f32.to_le_bytes().to_vec()))?;
            let weight=pile.put(blob)?;
            let leaf=entity! {leaf::leaf::<F32,1>():weight};
            let mut fragment=entity! {attrs::member*:leaf,metadata::name:"typed import fixture"};
            let root=fragment.root().context("typed fixture root")?;
            fragment+=entity! {metadata::name:"another exported document root"};
            Ok((root,fragment,serde_json::json!({})))
        })?;
        assert_eq!(receipt["model_root"],format!("{root:X}"));
        let store=mary::persist::read_model_pile_snapshot_read_only(&path)?;
        let collection=model_collection::open_model_collection(&store,handle)?;
        let models=model_collection::snapshot_model_collection_for(&store,collection)?;
        assert!(exists!(pattern!(models.facts(),[{existing @ metadata::name:"unrelated existing model"}])));
        let weight=find!(w:Id,pattern!(models.facts(),[{root @ attrs::member:?w}])).next().context("typed leaf missing")?;
        let leaf=leaf::resolve(models.facts(),models.store(),weight)?.context("typed leaf unreadable")?;
        assert_eq!(&leaf.payload()[..],1.25f32.to_le_bytes());
        assert_eq!(models.support().collection().handle(),handle);
        drop(leaf);drop(models);drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[cfg(feature="tokenizer")]
    #[test]
    fn invalid_gguf_ingress_does_not_publish_or_write_leaves() -> Result<()> {
        use std::io::Write;
        let (path,key,handle,_)=destination()?;
        let source=path.with_extension("gguf");
        std::fs::OpenOptions::new().write(true).create_new(true).open(&source)?.write_all(b"not a GGUF")?;
        let before=std::fs::metadata(&path)?.len();
        assert!(publish_into(&path,&key,handle,|pile|flint(&source,pile)).is_err());
        assert_eq!(std::fs::metadata(&path)?.len(),before);
        std::fs::remove_file(source)?;std::fs::remove_file(path)?;
        Ok(())
    }
}
