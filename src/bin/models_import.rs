//! Re-import a native Breeze or Voxtral model pile as one `nvfp4` root in an
//! explicit model collection.
//!
//! The source is an existing native model pile, opened read-only; the
//! destination is a pile holding the target collection's descriptor
//! (`trible pile collection init <PILE> models ...`), named by its exact
//! handle. The signer must be admitted by that descriptor's WRITE policy,
//! which is checked before anything is written.
//!
//! What the `nvfp4` root holds: every linear projection the runtime used to
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
//!     --tekken <tekken.json> [--calibration DIR[,DIR...]] \
//!     --pile <dest.pile> --key <signing.key> --collection blake3:HEX
//!
//! Voxtral with `--calibration` (the per-projection input statistics
//! `voxtral_capture` writes) packs every projection by GPTQ with searched
//! block scales instead, where that lowers the output error on the captured
//! inputs (`calibrate::pack_gptq_h`). Those planes have no device recipe to
//! equal byte for byte, so their gate is that they decode through the same
//! kernels: bound as the runtime binds them, read back unchanged, and their
//! GEMV and W4A16 prompt-GEMM products agree with an f64 product of the
//! stored values (`Nvfp4::gate_stored`). The stored format is plain NVFP4.
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use mary::format::attrs;
use mary::leaf::Leaf;
use mary::model_collection::{self, ModelCollection};
use mary::nvfp4::StoredProjection;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;
use triblespace::core::metadata;
use triblespace::core::repo::pile::{Pile, PileSnapshot};
use triblespace::core::signing_key_file;
use triblespace::prelude::*;

#[derive(Parser)]
#[command(
    about = "Re-import a native Breeze or Voxtral model pile as an nvfp4 root in a model collection"
)]
struct Args {
    #[command(subcommand)]
    model: Model,
    /// Existing native model pile, read only.
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
    Breeze,
    Voxtral {
        /// The checkpoint's tekken.json, gated piece for piece against the
        /// tokenizer the source pile already holds.
        #[arg(long)]
        tekken: PathBuf,
        /// Capture directories (`voxtral_capture --out`), comma separated;
        /// their H are summed. Packs by GPTQ instead of rounding to nearest.
        #[arg(long, value_delimiter = ',')]
        calibration: Vec<PathBuf>,
        /// Block scales under GPTQ: `search` (least error) or `max`.
        #[arg(long, default_value = "search")]
        block_scale: String,
    },
}

/// The source pile's one observation and the explicit root being converted.
struct Source {
    facts: TribleSet,
    store: PileSnapshot,
    root: Id,
    models: model_collection::ModelPileSnapshot,
}

fn open_source(path: &Path, root: &str) -> Result<Source> {
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
fn tensors(
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

const NEAREST_PACKER: &str = "mary nvfp4::pack_nearest, gated byte for byte against nn::nvfp4_gemv::Nvfp4::quantize on the same 16-bit values";

fn derivation(
    pile: &mut Pile,
    fragment: &mut Fragment,
    model_root: Id,
    source: &Source,
    source_path: &Path,
    projections: usize,
    packed: u64,
    packer: serde_json::Value,
) -> Result<()> {
    let value = serde_json::json!({
        "derivation": "nvfp4 storage of the projections the runtime quantized at load",
        "source_pile": source_path.file_name().map(|n| n.to_string_lossy().into_owned()),
        "source_root": format!("{:X}", source.root),
        "projections": projections,
        "packed_bytes": packed,
        "packer": packer,
        "other_tensors": "copied unchanged at source precision",
    });
    let root = mary::jsonfacts::save_json(&value, pile, fragment.facts_mut())
        .map_err(|e| anyhow::anyhow!("derivation facts: {e}"))?;
    label(pile, fragment, root, "nvfp4-derivation", model_root)
}

fn breeze(
    args: &Args,
    source: &Source,
    source_path: &Path,
    pile: &mut Pile,
) -> Result<(Fragment, serde_json::Value)> {
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
        serde_json::json!(NEAREST_PACKER),
    )?;
    let _ = args;
    Ok((
        fragment,
        serde_json::json!({
            "model": "breeze", "model_root": format!("{model_root:X}"), "codec_root": format!("{codec_root:X}"),
            "projections": stored.len(), "packed_bytes": packed, "dense_bytes": dense, "codec_bytes": codec_bytes,
            "documents": documents, "export_seconds": export_seconds,
        }),
    ))
}

fn voxtral(
    tekken: &Path,
    calibration: Option<mary::calibrate::Calibration>,
    source: &Source,
    source_path: &Path,
    pile: &mut Pile,
) -> Result<(Fragment, serde_json::Value)> {
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
    fast::export_nvfp4_projections(&loader, &device, calibration.as_ref(), |projection, replaces| {
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
    let packer = match &calibration {
        None => serde_json::json!(NEAREST_PACKER),
        Some(c) => serde_json::json!({
            "packer": format!(
                "mary calibrate::pack_gptq_h: GPTQ from captured input statistics H = X^T X \
                 (voxtral_capture), damped {} of the mean diagonal, block scales {:?}; per \
                 projection where it lowers the H-weighted error below rounding to nearest, \
                 else {NEAREST_PACKER}",
                c.damp, c.scale
            ),
            "gate": "decodes through the same kernels: nn::nvfp4_gemv::Nvfp4::gate_stored binds the stored planes, reads them back unchanged, and checks GEMV and W4A16 prompt-GEMM products against an f64 product of the stored values",
            "calibration": c.dirs.iter().map(|d| d.file_name().map(|n| n.to_string_lossy().into_owned())).collect::<Vec<_>>(),
        }),
    };
    derivation(
        pile,
        &mut fragment,
        model_root,
        source,
        source_path,
        stored.len(),
        packed,
        packer,
    )?;
    Ok((
        fragment,
        serde_json::json!({
            "model": "voxtral", "model_root": format!("{model_root:X}"), "projections": stored.len(),
            "replaced_tensors": dropped.len(), "packed_bytes": packed, "dense_bytes": dense,
            "tekken_pieces": held.len(), "export_seconds": export_seconds,
        }),
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let need =
        |v: &Option<PathBuf>, n: &str| v.clone().with_context(|| format!("--{n} is required"));
    let source_path = need(&args.source, "source")?;
    let dest = need(&args.pile, "pile")?;
    let key =
        signing_key_file::load_existing(&need(&args.key, "key")?).context("load signing key")?;
    let handle = model_collection::parse_collection_handle(
        args.collection
            .as_deref()
            .context("--collection is required")?,
    )?;
    let source = open_source(
        &source_path,
        args.source_root
            .as_deref()
            .context("--source-root is required")?,
    )?;

    let started = Instant::now();
    let size_before = std::fs::metadata(&dest)
        .with_context(|| format!("{dest:?}"))?
        .len();
    let mut pile = Pile::open(&dest).map_err(|e| anyhow::anyhow!("open {dest:?}: {e:?}"))?;
    let result = (|| -> Result<serde_json::Value> {
        pile.refresh().context("refresh destination")?;
        let collection: ModelCollection = {
            let snapshot = pile.snapshot().context("freeze destination")?;
            model_collection::open_model_collection(&snapshot, handle)?
        };
        // Before any weight is written: a refused signer must cost nothing.
        model_collection::require_model_writer(&mut pile, &key, collection)?;
        let (fragment, mut receipt) = match &args.model {
            Model::Breeze => breeze(&args, &source, &source_path, &mut pile)?,
            Model::Voxtral { tekken, calibration, block_scale } => {
                let calibration = if calibration.is_empty() {
                    None
                } else {
                    let mut c = mary::calibrate::Calibration::new(calibration.clone());
                    c.scale = mary::calibrate::BlockScale::parse(block_scale)?;
                    Some(c)
                };
                voxtral(tekken, calibration, &source, &source_path, &mut pile)?
            }
        };
        let commit =
            model_collection::publish_model_fragment_into(&mut pile, &key, collection, fragment)?;
        receipt["commit"] = serde_json::json!(
            triblespace::core::collection::CollectionRecord::Commit(commit)
                .fingerprint()
                .to_string()
        );
        Ok(receipt)
    })();
    let close = pile
        .close()
        .map_err(|e| anyhow::anyhow!("close {dest:?}: {e:?}"));
    let mut receipt = match (result, close) {
        (Ok(receipt), Ok(())) => receipt,
        (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
    };

    // Read it back the way the runtime will: the collection by handle, the
    // root by its labels.
    let store = mary::persist::read_model_pile_snapshot_read_only(&dest)?;
    let collection = model_collection::open_model_collection(&store, handle)?;
    let models = model_collection::snapshot_model_collection_for(&store, collection)?;
    match &args.model {
        Model::Breeze => {
            let ids = mary::models::breeze::load::Artifacts::select(
                models.facts(),
                models.store(),
                mary::persist::QUANTIZATION_NVFP4,
            )?;
            mary::models::breeze::load::Assets::from_frozen(models.facts(), models.store(), ids)?;
            let index = tensors(models.facts(), models.store(), ids.model_root)?;
            receipt["readback_nvfp4_leaves"] = serde_json::json!(
                index
                    .values()
                    .filter(|l| l.elem() == mary::leaf::Elem::Nvfp4)
                    .count()
            );
            receipt["readback_tensors"] = serde_json::json!(index.len());
        }
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
