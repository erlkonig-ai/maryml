use anyhow::Context;
use ed25519_dalek::SigningKey;
use mary::ingest::LeafDtype;
use mary::selection::{ModelSelector, SelectedModelIndex, TokenizerSelector};
use std::collections::BTreeMap;
use std::path::Path;
use triblespace::core::collection::CollectionCommit;
use triblespace::core::repo::SnapshotSource;
use triblespace::core::repo::pile::Pile;
use triblespace::prelude::*;

/// Check the things the encoding cannot: that the root is a model at all, that
/// every leaf is exact f32, and that the tensor names and shapes are the ones
/// this architecture contracts for.
///
/// What is NOT here any more is the shape check. A tensor leaf is one
/// `Tensor<F32, RANK>` blob whose header states its dims, and decoding refuses
/// a payload that is not the length those dims imply — so "does this shape
/// describe this data" is settled before a `Leaf` exists, for every reader, not
/// just this one. Comparing a shape blob against a data blob here was the only
/// way to know that under the two-blob form; under this one it would be
/// re-deriving a fact the leaf already carries.
///
/// The element check survives because it is about the leaf, not its shape: an
/// f16 leaf resolves perfectly well and is simply not what an exact embedding
/// root may contain. The rest survives because it is about the GRAPH — which
/// tensors this root reaches, under which names — which no tensor encoding
/// speaks to.
fn validate_f32_model<R: BlobStoreGet>(
    selected: &SelectedModelIndex<R>,
    contract: &BTreeMap<String, Vec<usize>>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !selected.handles().is_empty(),
        "embedding model root contains no tensors"
    );
    let mut actual_shapes = BTreeMap::new();
    for (name, leaf) in selected.handles() {
        anyhow::ensure!(
            leaf.elem() == mary::leaf::Elem::F32,
            "embedding tensor {name:?} is not an exact f32 leaf"
        );
        anyhow::ensure!(
            (leaf.payload().as_ptr() as usize).is_multiple_of(256),
            "embedding tensor {name:?} is not 256-byte aligned"
        );
        leaf.view_f32()
            .with_context(|| format!("decode embedding tensor {name:?}"))?;
        actual_shapes.insert(name.clone(), leaf.shape());
    }
    for (name, expected) in contract {
        let actual = actual_shapes
            .get(name)
            .with_context(|| format!("embedding model is missing required tensor {name:?}"))?;
        anyhow::ensure!(
            actual == expected,
            "embedding tensor {name:?} has shape {actual:?}, expected {expected:?}"
        );
    }
    if actual_shapes.len() != contract.len() {
        let unexpected = actual_shapes
            .keys()
            .find(|name| !contract.contains_key(*name))
            .expect("different exact-map lengths imply an unexpected key");
        anyhow::bail!("embedding model contains unexpected tensor {unexpected:?}");
    }
    Ok(())
}

fn publish_embedding_candidate_with_contract_impl(
    pile: &mut Pile,
    signing_key: &SigningKey,
    weights: &Path,
    format: mary::formats::WeightFormat,
    tokenizer_json: Option<&Path>,
    source: &str,
    architecture: mary::embed::EmbeddingArchitecture,
    contract: &BTreeMap<String, Vec<usize>>,
) -> anyhow::Result<(Id, CollectionCommit, usize)> {
    match architecture {
        mary::embed::EmbeddingArchitecture::ClipVitBasePatch32
        | mary::embed::EmbeddingArchitecture::NomicTextV15 => anyhow::ensure!(
            tokenizer_json.is_some(),
            "{architecture:?} requires its tokenizer in the same signed cohort"
        ),
        mary::embed::EmbeddingArchitecture::NomicVisionV15 => anyhow::ensure!(
            tokenizer_json.is_none(),
            "{architecture:?} has no tokenizer and rejects a tokenizer attachment"
        ),
    }

    let mut candidate = mary::persist::ingest_weight_file_filtered_fragment(
        pile,
        weights,
        format,
        LeafDtype::F32,
        source,
        mary::persist::QUANTIZATION_NATIVE,
        |name| contract.contains_key(name),
    )?;
    let model_root = candidate
        .root()
        .context("embedding candidate has no unique model root")?;

    if let Some(path) = tokenizer_json {
        let json =
            std::fs::read(path).with_context(|| format!("read tokenizer graph source {path:?}"))?;
        let tokenizer = mary::tokenizer::save_tokenizer_json(&json, source, candidate.blobs_mut())
            .map_err(|error| anyhow::anyhow!("ingest tokenizer graph for {source}: {error}"))?;
        candidate += tokenizer;
    }

    // Stage every dependency before exposing authority. Validate through the
    // destination's own reader, then append the signed commit as the final
    // publication step.
    let candidate_facts = candidate.facts().clone();
    let collection = mary::model_collection::model_collection_or_create(pile, signing_key)?;
    let prepared = mary::model_collection::prepare_model_fragment(candidate);
    let mut staged = prepared
        .stage_for(pile, collection, signing_key)
        .map_err(|error| anyhow::anyhow!("stage embedding commit dependencies: {error}"))?;

    // Freeze one coherent observation after staging, so the candidate's
    // dependencies are resident in the exact prefix used to validate them.
    let store = staged
        .store_mut()
        .snapshot()
        .map_err(|error| anyhow::anyhow!("freeze staged embedding observation: {error}"))?;
    let snapshot = mary::model_collection::snapshot_model_collection_for(&store, collection)?;
    let (mut facts, _, reader) = snapshot.into_parts();
    facts += candidate_facts;
    let selected = SelectedModelIndex::from_graph(
        &facts,
        reader,
        ModelSelector::Source {
            source,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )?;
    anyhow::ensure!(
        selected.single_root() == Some(model_root),
        "staged embedding root differs from the unique Source/native root"
    );
    validate_f32_model(&selected, contract)?;

    if tokenizer_json.is_some() {
        let tokenizer = mary::selection::load_tokenizer_from_graph(
            &facts,
            selected.reader(),
            TokenizerSelector::Name(source),
        )?;
        architecture.validate_tokenizer(&tokenizer)?;
    }
    let tensor_count = selected.handles().len();
    drop(selected);

    let commit = staged
        .finalize()
        .map_err(|error| anyhow::anyhow!("publish validated embedding commit: {error}"))?;
    Ok((model_root, commit, tensor_count))
}

/// Publish one architecture-exact embedding model and its optional tokenizer
/// as a single signed native model-collection commit.
pub(super) fn publish_embedding_candidate(
    pile: &mut Pile,
    signing_key: &SigningKey,
    weights: &Path,
    format: mary::formats::WeightFormat,
    tokenizer_json: Option<&Path>,
    source: &str,
    architecture: mary::embed::EmbeddingArchitecture,
) -> anyhow::Result<(Id, CollectionCommit, usize)> {
    let contract = architecture.tensor_shapes();
    publish_embedding_candidate_with_contract_impl(
        pile,
        signing_key,
        weights,
        format,
        tokenizer_json,
        source,
        architecture,
        &contract,
    )
}

/// Tiny synthetic bin tests need to exercise publication without constructing
/// hundreds of real architecture tensors. Production callers cannot supply a
/// contract independently of the architecture.
#[cfg(test)]
#[allow(dead_code)] // This support module is also test-compiled by the two parity bins.
pub(super) fn publish_embedding_candidate_with_contract(
    pile: &mut Pile,
    signing_key: &SigningKey,
    weights: &Path,
    format: mary::formats::WeightFormat,
    tokenizer_json: Option<&Path>,
    source: &str,
    architecture: mary::embed::EmbeddingArchitecture,
    contract: &BTreeMap<String, Vec<usize>>,
) -> anyhow::Result<(Id, CollectionCommit, usize)> {
    publish_embedding_candidate_with_contract_impl(
        pile,
        signing_key,
        weights,
        format,
        tokenizer_json,
        source,
        architecture,
        contract,
    )
}
