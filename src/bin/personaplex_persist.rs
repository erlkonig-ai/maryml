//! Persist the PersonaPlex-7B checkpoint into Mary's native append-only model
//! collection. The LM and Mimi codec become one exact f32 content-addressed
//! root under the canonical `nvidia/personaplex-7b-v1` source coordinate:
//!
//!   - `model.safetensors` — the 7B LM (temporal transformer + depth
//!     transformer + embeddings), 475 bf16 tensors → exact f32 leaves
//!     (bf16→f32 widening is lossless).
//!   - `tokenizer-e351c8d8-checkpoint125.safetensors` — the Mimi codec
//!     (byte-identical to the ungated kyutai checkpoint the mimi port was
//!     gated against; persisted here so the pile is the SELF-CONTAINED voice
//!     stack: codec encoder + decoder + LM).
//!
//! After this runs, the pile file and its signed `models` COMMIT, whose member
//! archive states the whole model, are the durable weight authority; the HF download is no longer needed. Tensor
//! names don't collide across the two checkpoints
//! (`transformer.*`/`depformer.*`/`emb.*`… vs
//! `encoder.*`/`decoder.*`/`quantizer.*`…), so one strict root serves both
//! components.
//!
//!   cargo run --release --features personaplex,import --bin personaplex_persist -- \
//!     <ckpt-dir> <pile-path> <signing-key>
//!
//! `<ckpt-dir>` holds `model.safetensors` and
//! `tokenizer-e351c8d8-checkpoint125.safetensors` (the HF snapshot layout).
//!
//! ── gate (always runs): pile round-trip bit-exactness ──
//! Every float tensor of both source files is re-read from the pile and
//! compared BIT-EXACT (f32 to_bits) against the bf16→f32 widening of the
//! safetensors bytes; shapes must match; every data blob must sit 256-aligned
//! in the pile mmap (the V3 payload-alignment invariant); and the LM leaf
//! count must equal `config::CHECKPOINT_TENSORS`. On top of the round-trip,
//! the load-bearing architecture constants in
//! `mary::models::personaplex::config` are asserted against the REAL tensor
//! shapes — the config file stays mechanically verified, not just documented.

use mary::ingest::LeafDtype;
use mary::models::personaplex::{PersonaPlexWeights, SOURCE, config as cfg};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;
use triblespace::core::blob::IntoBlob;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;
use triblespace::prelude::SnapshotSource;

const LM_FILE: &str = "model.safetensors";
const MIMI_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";

/// Validate one tensor's shape against the architecture config.
fn expect_shape(st: &SafeTensors, name: &str, want: &[usize]) -> anyhow::Result<()> {
    let view = st
        .tensor(name)
        .map_err(|error| anyhow::anyhow!("missing {name}: {error}"))?;
    anyhow::ensure!(
        view.shape() == want,
        "{name}: shape {:?} != config {want:?}",
        view.shape()
    );
    Ok(())
}

/// Fallibly widen one source tensor to f32 while retaining its exact shape.
fn tensor_f32(view: &TensorView<'_>) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    let shape = view.shape().to_vec();
    let elements = shape.iter().try_fold(1_usize, |product, &dimension| {
        product.checked_mul(dimension)
    });
    let elements = elements.ok_or_else(|| anyhow::anyhow!("tensor shape product overflow"))?;
    let width = match view.dtype() {
        Dtype::F64 => 8,
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        dtype => anyhow::bail!("unsupported source tensor dtype {dtype:?}"),
    };
    let expected_bytes = elements
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("tensor payload byte length overflow"))?;
    anyhow::ensure!(
        view.data().len() == expected_bytes,
        "tensor payload has {} bytes, expected {} for shape {shape:?} and {:?}",
        view.data().len(),
        expected_bytes,
        view.dtype()
    );
    // The checked total length makes every `chunks_exact(width)` item exactly
    // wide enough for direct indexing, without per-scalar fallibility overhead.
    let data = match view.dtype() {
        Dtype::F64 => view
            .data()
            .chunks_exact(8)
            .map(|bytes| {
                f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]) as f32
            })
            .collect(),
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect(),
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            .collect(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            .collect(),
        dtype => anyhow::bail!("unsupported source tensor dtype {dtype:?}"),
    };
    Ok((data, shape))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: personaplex_persist <ckpt-dir> <pile-path> <signing-key>");
        std::process::exit(2);
    }
    let ckpt_dir = Path::new(&args[1]);
    let pile_path = Path::new(&args[2]);
    let signing_key = signing_key_file::load_existing(Path::new(&args[3]))?;

    let files = [
        (ckpt_dir.join(LM_FILE), LM_FILE),
        (ckpt_dir.join(MIMI_FILE), MIMI_FILE),
    ];
    for (path, _) in &files {
        anyhow::ensure!(path.is_file(), "checkpoint file missing: {path:?}");
    }

    // The generic importer intentionally ingests every safetensors file in a
    // directory. PersonaPlex's authority is narrower: exactly these two files
    // compose the one LM + Mimi root. Reject stale copies and surprise shards
    // before writing anything.
    let (format, detected) = mary::formats::detect_format(ckpt_dir)?;
    anyhow::ensure!(
        format == mary::formats::WeightFormat::Safetensors,
        "PersonaPlex checkpoint must use safetensors, found {format:?}"
    );
    let detected_names: BTreeSet<_> = detected
        .iter()
        .map(|path| {
            path.file_name()
                .map(std::ffi::OsStr::to_owned)
                .ok_or_else(|| anyhow::anyhow!("weight path has no file name: {path:?}"))
        })
        .collect::<anyhow::Result<_>>()?;
    let expected_files: BTreeSet<_> = [LM_FILE, MIMI_FILE]
        .into_iter()
        .map(std::ffi::OsString::from)
        .collect();
    anyhow::ensure!(
        detected_names == expected_files,
        "PersonaPlex checkpoint must contain exactly {expected_files:?}, found {detected_names:?}"
    );

    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(pile_path)
    {
        Ok(_) => eprintln!("personaplex_persist: created new empty pile {pile_path:?}"),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }

    let started = Instant::now();
    let mut pile = Pile::open(pile_path)
        .map_err(|error| anyhow::anyhow!("open model pile {pile_path:?}: {error}"))?;
    let imported = (|| -> anyhow::Result<_> {
        eprintln!("Persisting PersonaPlex checkpoint {SOURCE} from {ckpt_dir:?} ...");
        let candidate = mary::persist::ingest_model_fragment(
            &mut pile,
            ckpt_dir,
            LeafDtype::F32,
            SOURCE,
            mary::persist::QUANTIZATION_NATIVE,
        )?;
        let root = candidate
            .root()
            .ok_or_else(|| anyhow::anyhow!("PersonaPlex candidate has no unique model root"))?;
        let collection =
            mary::model_collection::model_collection_or_create(&mut pile, &signing_key)?;

        // The member must be independently complete. Validate only the
        // candidate facts against the already-staged attachment prefix; the
        // union of other members is not allowed to fill holes in it.
        let reader = pile.snapshot()?;
        let weights = PersonaPlexWeights::from_graph(candidate.facts(), reader)?;
        anyhow::ensure!(
            weights.root() == root,
            "staged PersonaPlex root differs from the candidate's exact Source/native root"
        );
        let model_archive_data = triblespace::prelude::inlineencodings::Handle::<
            triblespace::core::blob::encodings::simplearchive::SimpleArchive,
        >::to_hash(candidate.facts().clone().to_blob().get_handle());
        let prepared = mary::model_collection::prepare_model_fragment(candidate);
        let store = pile.snapshot()?;
        let existing = mary::model_collection::snapshot_model_collection_for(&store, collection)?;
        if let Some(existing) = PersonaPlexWeights::find_in_snapshot(existing)? {
            anyhow::ensure!(
                existing.authority().model_root() == root
                    && existing.authority().model_archive_data() == model_archive_data,
                "a different PersonaPlex model is already authoritative in this pile"
            );
        }
        eprintln!(
            "Native root holds {} tensors; verifying both safetensors sources ...",
            weights.count()
        );

        let mut expected_names = BTreeSet::new();
        let mut lm_count = 0usize;
        let (mut checked, mut elems) = (0usize, 0usize);
        for (path, entity) in &files {
            let bytes = std::fs::read(path)
                .map_err(|error| anyhow::anyhow!("read source checkpoint {path:?}: {error}"))?;
            let st = SafeTensors::deserialize(&bytes)?;

            if *entity == LM_FILE {
                // Config truth: the load-bearing dims, asserted against real shapes.
                expect_shape(
                    &st,
                    "transformer.layers.0.self_attn.in_proj_weight",
                    &[3 * cfg::DIM, cfg::DIM],
                )?;
                expect_shape(
                    &st,
                    "transformer.layers.0.self_attn.out_proj.weight",
                    &[cfg::DIM, cfg::DIM],
                )?;
                expect_shape(
                    &st,
                    "transformer.layers.0.gating.linear_in.weight",
                    &[cfg::FFN_FUSED_IN, cfg::DIM],
                )?;
                expect_shape(
                    &st,
                    "transformer.layers.0.gating.linear_out.weight",
                    &[cfg::DIM, cfg::FFN_HIDDEN],
                )?;
                expect_shape(
                    &st,
                    &format!("transformer.layers.{}.norm2.alpha", cfg::NUM_LAYERS - 1),
                    &[1, 1, cfg::DIM],
                )?;
                expect_shape(&st, "out_norm.alpha", &[1, 1, cfg::DIM])?;
                expect_shape(
                    &st,
                    "depformer.layers.0.self_attn.in_proj_weight",
                    &[cfg::WEIGHTS_PER_STEP * 3 * cfg::DEP_DIM, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    "depformer.layers.0.self_attn.out_proj.weight",
                    &[cfg::WEIGHTS_PER_STEP * cfg::DEP_DIM, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    &format!(
                        "depformer.layers.{}.gating.{}.linear_in.weight",
                        cfg::DEP_LAYERS - 1,
                        cfg::WEIGHTS_PER_STEP - 1
                    ),
                    &[2 * cfg::DEP_FFN_HIDDEN, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    &format!("emb.{}.weight", cfg::N_Q - 1),
                    &[cfg::AUDIO_VOCAB, cfg::DIM],
                )?;
                expect_shape(&st, "text_emb.weight", &[cfg::TEXT_VOCAB, cfg::DIM])?;
                expect_shape(&st, "text_linear.weight", &[cfg::TEXT_LOGITS, cfg::DIM])?;
                expect_shape(
                    &st,
                    &format!("depformer_in.{}.weight", cfg::DEP_Q - 1),
                    &[cfg::DEP_DIM, cfg::DIM],
                )?;
                expect_shape(
                    &st,
                    &format!("depformer_emb.{}.weight", cfg::DEP_Q - 2),
                    &[cfg::AUDIO_VOCAB, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    "depformer_text_emb.weight",
                    &[cfg::TEXT_VOCAB, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    &format!("linears.{}.weight", cfg::DEP_Q - 1),
                    &[cfg::CARD, cfg::DEP_DIM],
                )?;
                eprintln!(
                    "config-truth shape assertions PASSED (temporal ffn {} fused {}, depth ffn {}, {} per-step)",
                    cfg::FFN_HIDDEN,
                    cfg::FFN_FUSED_IN,
                    cfg::DEP_FFN_HIDDEN,
                    cfg::WEIGHTS_PER_STEP
                );
            }

            let mut misaligned = 0usize;
            for name in st.names() {
                let view = st.tensor(name)?;
                if !matches!(
                    view.dtype(),
                    Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16
                ) {
                    continue; // importer skips non-float buffers
                }
                anyhow::ensure!(
                    expected_names.insert(name.to_owned()),
                    "tensor name {name:?} occurs in both PersonaPlex source files"
                );
                if *entity == LM_FILE {
                    lm_count += 1;
                }
                let (want, want_shape) = tensor_f32(&view)?;
                let leaf = weights
                    .exact()
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("{entity}/{name}: no native root leaf"))?;
                anyhow::ensure!(
                    leaf.elem() == mary::leaf::Elem::F32,
                    "{entity}/{name}: expected an f32 leaf"
                );
                // The typed tensor payload must sit 256-aligned in the mmap.
                if !(leaf.payload().as_ptr() as usize).is_multiple_of(256) {
                    eprintln!("  MISALIGNED (ptr % 256 != 0): {name}");
                    misaligned += 1;
                }
                let got_shape = leaf.shape();
                let got = leaf
                    .view_f32()
                    .ok_or_else(|| anyhow::anyhow!("{entity}/{name}: invalid f32 payload"))?;
                anyhow::ensure!(
                    got_shape == want_shape,
                    "{name}: shape {got_shape:?} != {want_shape:?}"
                );
                anyhow::ensure!(
                    got.len() == want.len(),
                    "{name}: len {} != {}",
                    got.len(),
                    want.len()
                );
                for (index, (&got, &want)) in got.iter().zip(want.iter()).enumerate() {
                    anyhow::ensure!(
                        got.to_bits() == want.to_bits(),
                        "{name}[{index}]: pile {got} != source {want} (bit mismatch)"
                    );
                }
                checked += 1;
                elems += got.len();
            }
            anyhow::ensure!(
                misaligned == 0,
                "{entity}: {misaligned} leaves misaligned — V3 alignment invariant violated"
            );
            eprintln!("{entity}: round-trip verified");
        }

        let actual_names: BTreeSet<_> = weights.exact().keys().cloned().collect();
        anyhow::ensure!(
            actual_names == expected_names,
            "native root name set differs from the exact LM + Mimi source union"
        );
        anyhow::ensure!(
            lm_count == cfg::CHECKPOINT_TENSORS,
            "LM leaf count {lm_count} != expected {} (config::CHECKPOINT_TENSORS)",
            cfg::CHECKPOINT_TENSORS
        );
        drop(weights);
        let staged = prepared
            .stage_for(&mut pile, collection, &signing_key)
            .map_err(|error| anyhow::anyhow!("stage validated PersonaPlex model: {error}"))?;
        staged
            .finalize()
            .map_err(|error| anyhow::anyhow!("finalize validated PersonaPlex model: {error}"))?;
        Ok((root, checked, elems, lm_count))
    })();

    // `close` is the sole durability boundary, even when a source gate fails.
    let close = pile
        .close()
        .map_err(|error| anyhow::anyhow!("close model pile {pile_path:?}: {error}"));
    let (root, checked, elems, lm_count) = match (imported, close) {
        (Ok(result), Ok(())) => result,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(close_error)) => {
            return Err(error.context(format!("import also failed to close pile: {close_error}")));
        }
    };

    let size = std::fs::metadata(pile_path)?.len();
    println!(
        "PersonaPlex native root {root} valid: {checked} tensors / {elems} elements \
         bit-identical (LM {lm_count} leaves), all payloads 256-aligned; \
         pile {:.2} GiB in {:.1}s",
        size as f64 / (1_u64 << 30) as f64,
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}
