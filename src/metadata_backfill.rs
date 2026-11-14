//! Additive backfill of attribute descriptions into old model COMMIT metadata.
//!
//! Until a7308814 every model commit mary published named an EMPTY metadata
//! archive: the builders flattened `entity!` fragments to bare facts, which
//! drops the metafacts (`attr.meta()`, the description of each attribute an
//! entity uses) before `pile.commit` could write them. The data is unaffected;
//! only the second archive is missing.
//!
//! An empty archive names no attribute, so the enrichment starts from the
//! DATA: for every `COMMIT(collection, data, M)` in the given collection that
//! the given key signed, this module reads `data`, collects the attribute ids
//! its tribles use, and builds `M'` = `M` united with the description the
//! [`Catalog`] holds for each of those ids. When `M' == M` nothing is emitted.
//! Otherwise it signs `COMMIT(collection, data, M')` with the same key and puts
//! the blobs the added facts reference. Nothing is removed or rewritten: `M'`
//! is a superset of `M`, and the old commit stays.
//!
//! Re-running is a no-op without bookkeeping: Ed25519 signing is
//! deterministic, a `SimpleArchive` is canonical, and the pile skips a record
//! or blob it already holds.
//!
//! Attribute ids the catalog does not know are counted and listed, never
//! fatal; so are archives that are not resident or do not decode, and commits
//! whose signature does not verify. An archive whose bytes fail their own hash
//! stops the run, naming it: that is damage to our own history, and a
//! migration that skipped it would report success over a corrupt pile.
//! Commits by any other key are left alone: each author re-signs its own.
//!
//! # One id, several declarations
//!
//! `attributes!` lets two modules declare the same attribute (same anchor,
//! same encoding, so the same id): `format`, `tokenizer` and `jsonfacts` all
//! declare `member` and `index`, for instance. Each declaration's `meta()`
//! carries the same attribute record (identity, `KIND_ATTRIBUTE`,
//! `value_encoding`, the encoding's description) plus a usage record of its
//! own (`metadata::attribute` + `source_module`, named and documented). A
//! commit's tribles do not say which declaration wrote them, so the catalog
//! entry for an id is the union of EVERY mary declaration of it. For a commit
//! that used all of them this is exactly what the writer emits; for one that
//! used fewer, `M'` additionally holds the other declarations' usage records,
//! which are true statements about the attribute. The typing facts are the
//! same either way.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use ed25519_dalek::SigningKey;
use triblespace::core::attribute::Attribute;
use triblespace::core::blob::encodings::UnknownBlob;
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::{Blob, IntoBlob, TryFromBlob};
use triblespace::core::collection::covered::Covered;
use triblespace::core::collection::records::{CollectionData, CollectionHandle};
use triblespace::core::collection::{
    CollectionCommit, CollectionRead, CollectionRecord, CollectionRecordSelector, CollectionStore,
};
use triblespace::core::id::Id;
use triblespace::core::inline::encodings::hash::Handle;
use triblespace::core::inline::{Inline, InlineEncoding};
use triblespace::core::metadata;
use triblespace::core::repo::pile::{GetBlobError, Pile, PileFile, PileSnapshot};
use triblespace::core::repo::{BlobStoreGet, BlobStoreMeta, BlobStorePut, SnapshotSource};
use triblespace::core::trible::TribleSet;
use triblespace::prelude::{exists, pattern};

/// Pile frame sizes, for the byte estimate: every record is one 256-byte
/// frame, and a blob is a 256-byte envelope plus its payload padded to 256.
const FRAME: u64 = 256;

fn blob_frame_bytes(len: usize) -> u64 {
    FRAME + (len as u64).div_ceil(FRAME) * FRAME
}

/// Everything mary says about one attribute id.
pub struct CatalogEntry {
    pub id: Id,
    /// Where each declaration lives, for the report only
    /// (`module::name<Encoding>`).
    pub declarations: Vec<String>,
    /// `meta().facts() ∪ meta().metafacts()` of every declaration, which is
    /// exactly what `entity!` unions into a fragment's metafacts per attribute.
    pub facts: TribleSet,
    /// The blobs those metas carry (names, descriptions, value formatters).
    pub blobs: Vec<Blob<UnknownBlob>>,
}

/// The frozen, typed list of every concrete attribute mary declares or
/// instantiates, keyed by id. Sorted by id; built once per run and only read
/// afterwards.
pub struct Catalog {
    entries: Vec<CatalogEntry>,
}

/// `declare!(catalog, path::to::module => a, b, c)`: one entry per static.
macro_rules! declare {
    ($catalog:ident, $($segment:ident)::+ => $($name:ident),+ $(,)?) => {{
        use $($segment)::+ as module;
        let path = stringify!($($segment)::+).replace(' ', "");
        $(
            $catalog.add(&format!("{path}::{}", stringify!($name)), &*module::$name);
        )+
    }};
}

/// Every rank a tensor header admits (`tensor::MAX_RANK` = 32).
macro_rules! ranks {
    ($catalog:ident, $family:literal, $f:ident, $t:ty) => {
        ranks!(@ $catalog, $family, $f, $t;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32);
    };
    (@ $catalog:ident, $family:literal, $f:ident, $t:ty; $($r:literal)+) => {
        $(
            $catalog.add(
                concat!($family, "<", stringify!($t), ",", stringify!($r), ">"),
                &$f::<$t, $r>(),
            );
        )+
    };
}

/// The derived tensor-attribute families, over every element format.
macro_rules! tensor_family {
    ($catalog:ident, $family:literal, $($f:ident)::+) => {{
        use $($f)::+ as family;
        use triblespace::core::blob::encodings::tensor::elements::{
            BF16, E4M3, F16, F32, F64, NVFP4, U8,
        };
        ranks!($catalog, $family, family, F32);
        ranks!($catalog, $family, family, F64);
        ranks!($catalog, $family, family, F16);
        ranks!($catalog, $family, family, BF16);
        ranks!($catalog, $family, family, U8);
        ranks!($catalog, $family, family, E4M3);
        ranks!($catalog, $family, family, NVFP4);
    }};
}

/// `models::inkling::pile::attrs::weight` without the CUDA lane that compiles
/// that module: the same `Attribute::anchored` derivation from the same
/// anchor, so the same id and the same `meta()` (an anchored runtime
/// attribute's meta is its identity fragment, a function of anchor and
/// encoding alone). `inkling_weight_family_matches_the_module` pins the
/// equality where the module compiles.
#[cfg(not(feature = "inkling-cuda"))]
fn inkling_weight<
    T: triblespace::core::blob::encodings::tensor::TensorElement,
    const RANK: usize,
>() -> Attribute<Handle<triblespace::core::blob::encodings::tensor::Tensor<T, RANK>>> {
    Attribute::anchored(INKLING_WEIGHT_ANCHOR)
}

/// `models::inkling::pile::attrs::WEIGHT_ANCHOR`, minted 2026-08-10 there.
pub const INKLING_WEIGHT_ANCHOR: Id =
    triblespace::core::id_hex!("0B51DA3E67216213871743E045590DBC");

impl Catalog {
    /// Every `attributes!` declaration in mary that this build compiles, the
    /// `triblespace::core::metadata` attributes mary writes as data, and the
    /// two derived tensor families. `catalog_covers_every_declaration` and
    /// `catalog_lists_every_attributes_block` keep the list honest.
    pub fn frozen() -> Self {
        let mut catalog = Self {
            entries: Vec::new(),
        };
        declare!(catalog, crate::format::attrs =>
            data, data_f16, shape, data_q4, data_q8, q_scales, format_marker, weight, bias,
            kind, safetensor_path, index, model_collection, model_root, member, config,
            chat_template, parent, model_name, source, quantization);
        declare!(catalog, crate::tokenizer::attrs =>
            index, member, model_name, tokenizer, vocab, merge, added, normalizer,
            pre_tokenizer, post_processor, decoder, piece, token_id, piece_bytes, merge_left,
            merge_right, unk_token, continuing_subword_prefix, end_of_word_suffix,
            max_input_chars, piece_score, byte_fallback, pattern, replace_content, behavior,
            template_single, template_pair, special_token, special_id, type_id, sep, cls);
        declare!(catalog, crate::jsonfacts::attrs =>
            member, index, json_key, json_string, json_int, json_float, json_bool,
            document_root);
        declare!(catalog, crate::dataset =>
            dataset_version, image, text, sound, split, sample_dataset, source, run_dataset,
            run_model, run_hyperparams, run_metrics, pref_chosen, pref_rejected, pref_score,
            pref_judge);
        // Pre-6b65f278 ids, pinned. Mary no longer writes them from entities,
        // but the legacy projection and migration carries put them into data.
        declare!(catalog, crate::model_collection::historical =>
            data, data_f16, shape, data_q4, data_q8, q_scales, format_marker, weight, bias,
            safetensor_path, index, model_root, member, model_name, source, lora_rank,
            lora_alpha, lora_adapter, lora_a, lora_b, tokenizer, vocab, merge, added,
            normalizer, pre_tokenizer, post_processor, decoder, piece, token_id, piece_bytes,
            merge_left, merge_right, max_input_chars, piece_score, byte_fallback, pattern);
        declare!(catalog, crate::models::inkling::cache_pile::schema =>
            prefix, rank, manifest, chunk, compatibility);
        declare!(catalog, crate::models::inkling::telemetry::schema =>
            tokenizer_identity, tokenizer, execution_identity, execution_profile,
            execution_unavailable, tp_world, layer_start, layer_end, stack_layers, partial,
            vocabulary_width, prefill_budget, context_budget, load_seconds, turn,
            generated_tokens, delta_tokens, carried_tokens, stop_reason, first_token_seconds,
            delta_nll_nats, delta_scored_tokens, delta_nll_frozen_nats, persisted_version,
            persisted_experts, turn_seconds, position, context_placement, max_response_tokens,
            required_end, fits, previous_position, previous_turns, initialization_tokens,
            context_epoch);
        #[cfg(feature = "inkling-cuda")]
        {
            declare!(catalog, crate::models::inkling::pile::attrs =>
                weight_nvfp4_2, expert_index, layer);
            declare!(catalog, crate::models::inkling::version::attrs =>
                learn_lr, learn_anchor, learn_seed, learn_steps, learned_span, explanation,
                code_revision);
        }
        #[cfg(feature = "nemotron-cuda")]
        declare!(catalog, crate::models::nemotron_h::session::checkpoint_manifest::schema =>
            model_facts, token_ids, token_range, source_layer, selected_config,
            tokenizer_base, execution_profile, position);
        #[cfg(feature = "gemma")]
        declare!(catalog, crate::models::gemma::lora::attrs =>
            lora_rank, lora_alpha, lora_adapter, lora_projection, lora_a, lora_b, model_name);
        // Core attributes mary writes as DATA: JSON documents and checkpoint
        // tensor names (`metadata::name`), document kinds (`metadata::tag`),
        // descriptions. The whole module, so none is missed.
        declare!(catalog, triblespace::core::metadata =>
            archive, description, value_encoding, anchor, blob_encoding, array_item_schema,
            tensor_rank, hash_schema, value_formatter, name, iri, attribute, source,
            source_module, tag, created_at, updated_at, started_at, finished_at, expires_at,
            supersedes);
        tensor_family!(catalog, "crate::leaf::leaf", crate::leaf::leaf);
        #[cfg(feature = "inkling-cuda")]
        tensor_family!(
            catalog,
            "crate::models::inkling::pile::attrs::weight",
            crate::models::inkling::pile::attrs::weight
        );
        #[cfg(not(feature = "inkling-cuda"))]
        tensor_family!(
            catalog,
            "crate::models::inkling::pile::attrs::weight",
            crate::metadata_backfill::inkling_weight
        );
        catalog
    }

    /// Add one declaration of an attribute: the description `entity!` would
    /// carry for it, united into the entry for its id.
    pub fn add<S: InlineEncoding>(&mut self, declaration: &str, attribute: &Attribute<S>) {
        let id = attribute.id();
        let meta = attribute.meta();
        let mut facts = meta.facts().clone();
        facts.union(meta.metafacts().clone());
        let blobs: Vec<Blob<UnknownBlob>> = meta
            .blobs()
            .clone()
            .snapshot()
            .expect("MemoryBlobStore::snapshot is infallible")
            .into_iter()
            .map(|(_, blob)| blob)
            .collect();
        let declaration = format!("{declaration}<{}>", std::any::type_name::<S>());
        let position = self.entries.partition_point(|entry| entry.id < id);
        match self.entries.get_mut(position) {
            Some(entry) if entry.id == id => {
                entry.declarations.push(declaration);
                entry.facts.union(facts);
                let held: BTreeSet<[u8; 32]> = entry
                    .blobs
                    .iter()
                    .map(|blob| blob.get_handle().raw)
                    .collect();
                entry.blobs.extend(
                    blobs
                        .into_iter()
                        .filter(|blob| !held.contains(&blob.get_handle().raw)),
                );
            }
            _ => self.entries.insert(
                position,
                CatalogEntry {
                    id,
                    declarations: vec![declaration],
                    facts,
                    blobs,
                },
            ),
        }
    }

    /// The entry for one attribute id, if the catalog knows it.
    pub fn get(&self, id: Id) -> Option<&CatalogEntry> {
        self.entries
            .binary_search_by(|entry| entry.id.cmp(&id))
            .ok()
            .map(|index| &self.entries[index])
    }

    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    /// Union of every entry's facts.
    pub fn facts(&self) -> TribleSet {
        let mut facts = TribleSet::new();
        for entry in &self.entries {
            facts.union(entry.facts.clone());
        }
        facts
    }

    /// Handle of the canonical archive of [`Self::facts`]. Two builds with the
    /// same hash add identical facts for identical data. Every entry's facts
    /// name its attribute id, so the hash also fixes which ids are known.
    pub fn hash(&self) -> Inline<Handle<SimpleArchive>> {
        let blob: Blob<SimpleArchive> = self.facts().to_blob();
        blob.get_handle()
    }
}

/// What enriching one commit found.
pub struct Enrichment {
    /// `M'`: the old metadata plus the description of every known attribute.
    pub facts: TribleSet,
    /// Blobs of the entries whose facts were added.
    pub blobs: Vec<Blob<UnknownBlob>>,
    /// Known attribute ids in the data.
    pub known: BTreeSet<Id>,
    /// Attribute ids in the data the catalog does not know.
    pub unknown: BTreeSet<Id>,
}

impl Enrichment {
    /// `M' == M`: `M'` is a superset, so equal length is equality.
    pub fn unchanged(&self, metadata: &TribleSet) -> bool {
        self.facts.len() == metadata.len()
    }
}

/// The attribute ids the tribles of `data` use.
pub fn data_attributes(data: &TribleSet) -> BTreeSet<Id> {
    data.iter().map(|trible| *trible.a()).collect()
}

/// `M'` for one commit: `metadata` united with the catalog's description of
/// every attribute in `attributes`.
pub fn enrich(catalog: &Catalog, attributes: &BTreeSet<Id>, metadata: &TribleSet) -> Enrichment {
    let mut facts = metadata.clone();
    let mut blobs = Vec::new();
    let mut known = BTreeSet::new();
    let mut unknown = BTreeSet::new();
    for attribute in attributes {
        let Some(entry) = catalog.get(*attribute) else {
            unknown.insert(*attribute);
            continue;
        };
        known.insert(*attribute);
        if entry.facts.difference(metadata).is_empty() {
            continue;
        }
        facts.union(entry.facts.clone());
        blobs.extend(entry.blobs.iter().cloned());
    }
    Enrichment {
        facts,
        blobs,
        known,
        unknown,
    }
}

/// One attribute id the catalog does not know.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UnknownAttribute {
    /// The signer's commits whose data uses it.
    pub commits: usize,
    /// Of those, commits whose old metadata already has a `KIND_ATTRIBUTE`
    /// record for it.
    pub described: usize,
}

/// Everything one collection showed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Report {
    pub catalog_hash: Inline<Handle<SimpleArchive>>,
    pub signer: [u8; 32],
    pub collection: CollectionHandle,
    pub applied: bool,
    /// Every COMMIT record in the collection.
    pub commits: usize,
    /// Commits by the signer before this run.
    pub signer_commits: usize,
    /// Commits by other keys, left alone.
    pub other_signers: usize,
    /// Commits whose signature does not verify, skipped.
    pub bad_signatures: usize,
    /// Signer commits whose data archive is not resident / does not decode.
    pub data_non_resident: usize,
    pub data_undecodable: usize,
    /// Signer commits whose metadata archive is not resident / does not decode.
    pub metadata_non_resident: usize,
    pub metadata_undecodable: usize,
    /// Signer commits whose `M'` equals `M`.
    pub current_commits: usize,
    /// Signer commits whose `M'` differs from `M`.
    pub enriched_commits: usize,
    /// Of those, the `COMMIT(collection, data, M')` already in the collection.
    pub present_commits: usize,
    /// `COMMIT(collection, data, M')` records this run writes (or would).
    pub new_commits: usize,
    /// Blobs this run writes (or would): new archives and description blobs
    /// not resident before the run, each counted once.
    pub new_blobs: usize,
    /// Framed pile bytes of `new_commits` and `new_blobs`.
    pub new_bytes: u64,
    /// Known attribute ids seen in the signer's data, with commit counts.
    pub known: BTreeMap<Id, usize>,
    pub unknown: BTreeMap<Id, UnknownAttribute>,
    /// One row per enriched signer commit, for the report.
    pub enriched: Vec<EnrichedCommit>,
}

/// One signer commit whose `M'` differs from its `M`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnrichedCommit {
    pub data: CollectionData,
    /// `M`, the metadata the old commit names.
    pub old_metadata: Inline<Handle<SimpleArchive>>,
    /// `M'`, the metadata the successor names.
    pub metadata: Inline<Handle<SimpleArchive>>,
    /// Bytes of the `M'` archive.
    pub metadata_bytes: usize,
    /// Tribles `M'` holds beyond `M`.
    pub added_tribles: usize,
    /// Distinct attribute ids in the data.
    pub attributes: usize,
}

enum Read<T> {
    Ok(T),
    NonResident,
    Undecodable,
}

/// Read one archive. A blob that fails its own hash stops the run.
fn read_archive(
    snapshot: &PileSnapshot,
    handle: Inline<Handle<SimpleArchive>>,
    role: &str,
) -> Result<Read<TribleSet>> {
    Ok(
        match snapshot.get::<Blob<SimpleArchive>, SimpleArchive>(handle) {
            Err(GetBlobError::BlobNotFound(_)) => Read::NonResident,
            Err(GetBlobError::ValidationError(_)) => bail!(
                "{role} archive {} fails its hash check: refusing to continue over corrupt bytes",
                hex_upper(&handle.raw)
            ),
            Err(GetBlobError::ConversionError(_)) => Read::Undecodable,
            Ok(blob) => match TribleSet::try_from_blob(blob) {
                Err(_) => Read::Undecodable,
                Ok(set) => Read::Ok(set),
            },
        },
    )
}

/// Scratch for one run: every blob already counted (and, when applying,
/// written) and every successor planned. Lives for one operation.
struct Run<'a> {
    snapshot: &'a PileSnapshot,
    apply: bool,
    staged: BTreeSet<[u8; 32]>,
}

impl Run<'_> {
    fn stage(
        &mut self,
        store: &mut Pile,
        blob: Blob<UnknownBlob>,
        report: &mut Report,
    ) -> Result<()> {
        let handle = blob.get_handle();
        if !self.staged.insert(handle.raw) {
            return Ok(());
        }
        let resident = self
            .snapshot
            .metadata(handle)
            .map_err(|error| anyhow!("check blob residency: {error}"))?
            .is_some();
        if !resident {
            report.new_blobs += 1;
            report.new_bytes += blob_frame_bytes(blob.bytes.len());
        }
        if self.apply {
            // A resident blob is put too: the pile returns the existing handle
            // without appending, and the new commit's closure is certain.
            store
                .put::<UnknownBlob, _>(blob)
                .map_err(|error| anyhow!("put blob: {error:?}"))?;
        }
        Ok(())
    }
}

/// Enrich one collection against one frozen snapshot of `store`. Writes only
/// when `apply` is set.
pub fn run(
    store: &mut Pile,
    catalog: &Catalog,
    signer: &SigningKey,
    collection: CollectionHandle,
    apply: bool,
) -> Result<Report> {
    let snapshot = store
        .snapshot()
        .map_err(|error| anyhow!("freeze model-metadata source snapshot: {error:?}"))?;
    let mut report = Report {
        catalog_hash: catalog.hash(),
        signer: signer.verifying_key().to_bytes(),
        collection,
        applied: apply,
        commits: 0,
        signer_commits: 0,
        other_signers: 0,
        bad_signatures: 0,
        data_non_resident: 0,
        data_undecodable: 0,
        metadata_non_resident: 0,
        metadata_undecodable: 0,
        current_commits: 0,
        enriched_commits: 0,
        present_commits: 0,
        new_commits: 0,
        new_blobs: 0,
        new_bytes: 0,
        known: BTreeMap::new(),
        unknown: BTreeMap::new(),
        enriched: Vec::new(),
    };
    let records = snapshot
        .select_records(&BTreeSet::from([CollectionRecordSelector::Collection(
            collection,
        )]))
        .map_err(|error| anyhow!("select collection records: {error}"))?;
    let held: BTreeSet<CollectionCommit> = records
        .iter()
        .filter_map(|record| match record {
            CollectionRecord::Commit(commit) => Some(*commit),
            _ => None,
        })
        .collect();
    let key = signer.verifying_key().to_bytes();
    let mut run = Run {
        snapshot: &snapshot,
        apply,
        staged: BTreeSet::new(),
    };
    let mut planned = BTreeSet::new();

    for commit in &held {
        report.commits += 1;
        if commit.verify_strict().is_err() {
            report.bad_signatures += 1;
            continue;
        }
        if commit.public_key().raw != key {
            report.other_signers += 1;
            continue;
        }
        report.signer_commits += 1;
        let data = match read_archive(&snapshot, commit.data().transmute(), "data")? {
            Read::Ok(data) => data,
            Read::NonResident => {
                report.data_non_resident += 1;
                continue;
            }
            Read::Undecodable => {
                report.data_undecodable += 1;
                continue;
            }
        };
        let metadata = match read_archive(&snapshot, commit.metadata(), "metadata")? {
            Read::Ok(metadata) => metadata,
            Read::NonResident => {
                report.metadata_non_resident += 1;
                continue;
            }
            Read::Undecodable => {
                report.metadata_undecodable += 1;
                continue;
            }
        };
        let attributes = data_attributes(&data);
        drop(data);
        let enrichment = enrich(catalog, &attributes, &metadata);
        for attribute in &enrichment.known {
            *report.known.entry(*attribute).or_default() += 1;
        }
        for attribute in &enrichment.unknown {
            let entry = report.unknown.entry(*attribute).or_default();
            entry.commits += 1;
            let attribute = *attribute;
            if exists!(
                (encoding: Id),
                pattern!(&metadata, [{ attribute @
                    metadata::tag: metadata::KIND_ATTRIBUTE,
                    metadata::value_encoding: ?encoding,
                }])
            ) {
                entry.described += 1;
            }
        }
        if enrichment.unchanged(&metadata) {
            report.current_commits += 1;
            continue;
        }
        report.enriched_commits += 1;
        let added = enrichment.facts.len() - metadata.len();
        for blob in enrichment.blobs {
            run.stage(store, blob, &mut report)?;
        }
        let archive: Blob<SimpleArchive> = enrichment.facts.to_blob();
        let handle = archive.get_handle();
        report.enriched.push(EnrichedCommit {
            data: commit.data(),
            old_metadata: commit.metadata(),
            metadata: handle,
            metadata_bytes: archive.bytes.len(),
            added_tribles: added,
            attributes: attributes.len(),
        });
        run.stage(store, archive.transmute(), &mut report)?;
        let successor = CollectionCommit::sign(signer, collection, commit.data(), handle);
        if held.contains(&successor) || !planned.insert(successor) {
            report.present_commits += 1;
            continue;
        }
        report.new_commits += 1;
        report.new_bytes += FRAME;
        if apply {
            store
                .insert(CollectionRecord::Commit(successor))
                .map_err(|error| anyhow!("insert enriched COMMIT: {error:?}"))?;
        }
    }
    drop(run);
    drop(snapshot);
    Ok(report)
}

/// Open `pile`, run, and close it. A dry run opens the file READ-ONLY, so it
/// cannot write by construction; `apply` opens it for appending.
pub fn run_path(
    pile: &Path,
    signer: &SigningKey,
    catalog: &Catalog,
    collection: CollectionHandle,
    apply: bool,
) -> Result<Report> {
    if apply {
        let mut store =
            Pile::open(pile).with_context(|| format!("open pile {}", pile.display()))?;
        store
            .refresh()
            .map_err(|error| anyhow!("replay pile {}: {error:?}", pile.display()))?;
        let result = run(&mut store, catalog, signer, collection, true);
        let flushed = match &result {
            Ok(_) => store
                .flush()
                .map_err(|error| anyhow!("flush pile: {error:?}")),
            Err(_) => Ok(()),
        };
        let close = store
            .close()
            .map_err(|error| anyhow!("close pile: {error:?}"));
        let report = result?;
        flushed?;
        close?;
        Ok(report)
    } else {
        let file = PileFile::open_read_only(pile)
            .with_context(|| format!("open pile {} read-only", pile.display()))?;
        let mut store = Covered::new(file);
        store
            .refresh()
            .map_err(|error| anyhow!("replay pile {}: {error:?}", pile.display()))?;
        let result = run(&mut store, catalog, signer, collection, false);
        let close = store
            .close()
            .map_err(|error| anyhow!("close pile: {error:?}"));
        let report = result?;
        close?;
        Ok(report)
    }
}

/// Refuse unless `expected` names this build's catalog hash.
pub fn check_catalog(catalog: &Catalog, expected: &str) -> Result<()> {
    let expected = expected.trim();
    let expected = expected.strip_prefix("blake3:").unwrap_or(expected);
    let actual = hex_upper(&catalog.hash().raw);
    if !actual.eq_ignore_ascii_case(expected) {
        bail!("catalog hash is blake3:{actual}, expected {expected}; refusing to run");
    }
    Ok(())
}

pub fn hex_upper(raw: &[u8]) -> String {
    raw.iter().map(|byte| format!("{byte:02X}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::trible::Fragment;
    use triblespace::prelude::{entity, find};

    /// Every `attributes!` module the catalog declares, by its `describe()`.
    fn module_descriptions() -> Vec<(&'static str, fn() -> Fragment)> {
        let mut modules: Vec<(&'static str, fn() -> Fragment)> = vec![
            ("crate::format::attrs", crate::format::attrs::describe),
            ("crate::tokenizer::attrs", crate::tokenizer::attrs::describe),
            ("crate::jsonfacts::attrs", crate::jsonfacts::attrs::describe),
            ("crate::dataset", crate::dataset::describe),
            (
                "crate::model_collection::historical",
                crate::model_collection::historical::describe,
            ),
            (
                "crate::models::inkling::cache_pile::schema",
                crate::models::inkling::cache_pile::schema::describe,
            ),
            (
                "crate::models::inkling::telemetry::schema",
                crate::models::inkling::telemetry::schema::describe,
            ),
            ("triblespace::core::metadata", metadata::describe),
        ];
        #[cfg(feature = "inkling-cuda")]
        {
            modules.push((
                "crate::models::inkling::pile::attrs",
                crate::models::inkling::pile::attrs::describe,
            ));
            modules.push((
                "crate::models::inkling::version::attrs",
                crate::models::inkling::version::attrs::describe,
            ));
        }
        #[cfg(feature = "nemotron-cuda")]
        modules.push((
            "crate::models::nemotron_h::session::checkpoint_manifest::schema",
            crate::models::nemotron_h::session::checkpoint_manifest::schema::describe,
        ));
        #[cfg(feature = "gemma")]
        modules.push((
            "crate::models::gemma::lora::attrs",
            crate::models::gemma::lora::attrs::describe,
        ));
        modules
    }

    #[test]
    fn catalog_covers_every_declaration() {
        let catalog = Catalog::frozen();
        let all = catalog.facts();
        for (module, describe) in module_descriptions() {
            let (_, facts, metafacts, _) = describe().into_parts();
            let mut description = facts;
            description.union(metafacts);
            let declared: Vec<Id> = find!(
                attribute: Id,
                pattern!(&description, [{ ?attribute @ metadata::tag: metadata::KIND_ATTRIBUTE }])
            )
            .collect();
            assert!(!declared.is_empty(), "{module} declares nothing");
            for attribute in declared {
                let entry = catalog
                    .get(attribute)
                    .unwrap_or_else(|| panic!("{module}: {attribute:X} not in the catalog"));
                assert!(
                    entry.declarations.iter().any(|d| d.starts_with(module)),
                    "{module}: {attribute:X} is in the catalog only as {:?}",
                    entry.declarations
                );
            }
            assert!(
                description.difference(&all).is_empty(),
                "{module}: describe() says more than the catalog"
            );
        }
        println!(
            "CATALOG blake3:{} ids={} tribles={} blobs={}",
            hex_upper(&catalog.hash().raw),
            catalog.entries().len(),
            all.len(),
            catalog
                .entries()
                .iter()
                .map(|e| e.blobs.len())
                .sum::<usize>()
        );
        for entry in catalog.entries() {
            if !entry.declarations[0].contains("::leaf::leaf<")
                && !entry.declarations[0].contains("::attrs::weight<")
            {
                println!("ATTR {:X} {}", entry.id, entry.declarations.join(" | "));
            }
        }
    }

    /// Every file with an `attributes!` block is in the catalog, name by name.
    /// A new block, or a new attribute in an old one, fails here until the
    /// catalog lists it.
    #[test]
    fn catalog_lists_every_attributes_block() {
        let table: &[(&str, &str, bool)] = &[
            ("src/format.rs", "crate::format::attrs", true),
            ("src/tokenizer.rs", "crate::tokenizer::attrs", true),
            ("src/jsonfacts.rs", "crate::jsonfacts::attrs", true),
            ("src/dataset.rs", "crate::dataset", true),
            (
                "src/model_collection.rs",
                "crate::model_collection::historical",
                true,
            ),
            (
                "src/models/inkling/cache_pile.rs",
                "crate::models::inkling::cache_pile::schema",
                true,
            ),
            (
                "src/models/inkling/telemetry.rs",
                "crate::models::inkling::telemetry::schema",
                true,
            ),
            (
                "src/models/inkling/pile.rs",
                "crate::models::inkling::pile::attrs",
                cfg!(feature = "inkling-cuda"),
            ),
            (
                "src/models/inkling/version.rs",
                "crate::models::inkling::version::attrs",
                cfg!(feature = "inkling-cuda"),
            ),
            (
                "src/models/nemotron_h/session_checkpoint_manifest.rs",
                "crate::models::nemotron_h::session::checkpoint_manifest::schema",
                cfg!(feature = "nemotron-cuda"),
            ),
            (
                "src/models/gemma/lora.rs",
                "crate::models::gemma::lora::attrs",
                cfg!(feature = "gemma"),
            ),
        ];
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        let mut stack = vec![root.join("src")];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    files.push(path);
                }
            }
        }
        let catalog = Catalog::frozen();
        let declared: BTreeSet<String> = catalog
            .entries()
            .iter()
            .flat_map(|entry| entry.declarations.iter())
            .map(|d| d.split('<').next().unwrap().to_owned())
            .collect();
        let mut seen = BTreeSet::new();
        for path in files {
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            if relative == "src/metadata_backfill.rs" {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let names = attributes_block_names(&text);
            if names.is_empty() {
                continue;
            }
            let Some((_, module, compiled)) = table.iter().find(|(file, _, _)| *file == relative)
            else {
                panic!("{relative} has an attributes! block the catalog does not list: {names:?}");
            };
            seen.insert(relative.clone());
            if !compiled {
                println!("SKIPPED {relative}: its feature is off in this build");
                continue;
            }
            for name in names {
                assert!(
                    declared.contains(&format!("{module}::{name}")),
                    "{module}::{name} is declared in {relative} and missing from the catalog"
                );
            }
        }
        assert_eq!(
            seen.len(),
            table.len(),
            "a listed file has no attributes! block any more"
        );
    }

    /// The names an `attributes!` block declares, `"HEX" [unsafe] as [pub] name:`.
    fn attributes_block_names(text: &str) -> Vec<String> {
        let mut names = Vec::new();
        let mut inside = false;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") {
                continue;
            }
            if trimmed.starts_with("attributes! {") || trimmed.contains("macros::attributes! {") {
                inside = true;
                continue;
            }
            if !inside {
                continue;
            }
            if trimmed == "}" {
                inside = false;
                continue;
            }
            let Some(rest) = trimmed.strip_prefix('"') else {
                continue;
            };
            let Some((hex, rest)) = rest.split_once('"') else {
                continue;
            };
            if hex.len() != 32 {
                continue;
            }
            let rest = rest.trim_start();
            let rest = rest.strip_prefix("unsafe").unwrap_or(rest).trim_start();
            let Some(rest) = rest.strip_prefix("as") else {
                continue;
            };
            let rest = rest.trim_start();
            let rest = rest.strip_prefix("pub ").unwrap_or(rest).trim_start();
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            names.push(name);
        }
        names
    }

    /// An attribute with one declaration: its entry is exactly the metafacts
    /// `entity!` writes when it uses that attribute.
    #[test]
    fn an_entry_is_what_entity_writes() {
        let catalog = Catalog::frozen();
        let fragment = entity! { crate::format::attrs::kind: "linear" };
        let entry = catalog.get(crate::format::attrs::kind.id()).unwrap();
        assert_eq!(entry.declarations.len(), 1);
        assert!(*fragment.metafacts() == entry.facts);

        // A derived tensor attribute: identity plus encoding description.
        use triblespace::core::blob::encodings::tensor::elements::F32;
        let leaf = crate::leaf::leaf::<F32, 2>();
        let entry = catalog.get(leaf.id()).unwrap();
        let mut expected = leaf.meta().facts().clone();
        expected.union(leaf.meta().metafacts().clone());
        assert!(expected == entry.facts);
        assert_eq!(
            catalog.entries().len(),
            catalog
                .entries()
                .iter()
                .map(|e| e.id)
                .collect::<BTreeSet<_>>()
                .len()
        );
    }

    /// The non-CUDA stand-in for Inkling's weight family has the id the
    /// module's own declaration pins (`weight_nvfp4_2` coincides with
    /// `weight::<NVFP4, 2>()`; the value is the one the CUDA build prints).
    #[cfg(not(feature = "inkling-cuda"))]
    #[test]
    fn inkling_weight_stand_in_has_the_modules_id() {
        use triblespace::core::blob::encodings::tensor::elements::NVFP4;
        assert_eq!(
            format!("{:X}", inkling_weight::<NVFP4, 2>().id()),
            "35078B7C84F7910EFDDD8AA721BA61E1"
        );
    }

    #[cfg(feature = "inkling-cuda")]
    #[test]
    fn inkling_weight_family_matches_the_declaration() {
        use triblespace::core::blob::encodings::tensor::elements::NVFP4;
        assert_eq!(
            crate::models::inkling::pile::attrs::weight::<NVFP4, 2>().id(),
            crate::models::inkling::pile::attrs::weight_nvfp4_2.id()
        );
    }

    #[test]
    fn the_catalog_check_refuses_a_mismatch() {
        let catalog = Catalog::frozen();
        let hash = hex_upper(&catalog.hash().raw);
        check_catalog(&catalog, &hash).unwrap();
        check_catalog(&catalog, &format!("blake3:{}", hash.to_lowercase())).unwrap();
        assert!(check_catalog(&catalog, &"0".repeat(64)).is_err());
    }
}
