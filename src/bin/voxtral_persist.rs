//! Import one complete Voxtral-Mini-4B-Realtime-2602 cohort into Mary's native
//! append-only model collection.
//!
//! The result is two ordinary model roots in one pile, distinguished only by
//! their quantization coordinate:
//!
//! - `native`: the exact f32 checkpoint (bf16 → f32 is lossless);
//! - `f16`: the full `f16::from_f32` derivation used by the realtime lanes.
//!
//! Beside them sits the Tekken tokenizer graph from the directory's
//! `tekken.json`, named after the same source, gated byte-for-byte against the
//! file for every piece.
//!
//! The derivation reads the frozen exact root and writes back through the same
//! open pile one tensor at a time. Runtime selection uses one immutable native
//! collection snapshot; there is no Repository branch, sibling pile, random
//! signer, fallback root, or filename discovery.
//!
//! Rerunning the same source with the same signing key is content-idempotent.
//! Changing bytes under either existing source coordinate makes the final
//! local-latest gate ambiguous and fails closed.
//!
//! ```text
//! cargo run --release --features voxtral,import --bin voxtral_persist -- \
//!   <model-dir> <pile-path> <signing-key>
//! ```

use mary::ingest::LeafDtype;
use mary::leaf::Elem;
use mary::models::voxtral::tokenizer::{Tekken, TekkenSource};
use mary::models::voxtral::{QUANTIZATION_F16, SOURCE, VoxtralWeights};
use mary::selection::{ModelSelector, SelectedModelIndex};
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;
use triblespace::core::repo::SnapshotSource;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;

fn verify_exact_source(
    model_dir: &Path,
    weights: &VoxtralWeights<triblespace::core::repo::pile::PileSnapshot>,
) -> anyhow::Result<(usize, usize)> {
    let path = model_dir.join("model.safetensors");
    let file = std::fs::File::open(&path)?;
    // Keep the multi-gigabyte source out of a second host allocation during
    // the gate. Each tensor is decoded/materialized independently below.
    let mapped = unsafe { memmap2::Mmap::map(&file)? };
    let source = SafeTensors::deserialize(&mapped)?;
    let expected: BTreeSet<_> = source
        .names()
        .into_iter()
        .filter(|name| {
            matches!(
                source.tensor(name).map(|tensor| tensor.dtype()),
                Ok(Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16)
            )
        })
        .collect();
    anyhow::ensure!(
        expected.len() == weights.exact().len(),
        "native root has {} tensors but source safetensors has {} float tensors",
        weights.exact().len(),
        expected.len()
    );

    let mut elements = 0;
    for name in &expected {
        let handles = weights
            .exact()
            .get(*name)
            .ok_or_else(|| anyhow::anyhow!("native root is missing source tensor {name:?}"))?;
        anyhow::ensure!(
            handles.elem() == Elem::F32,
            "native tensor {name:?} is not f32"
        );
        // A typed leaf carries its own dims and its payload is the tensor data
        // itself, so neither the shape blob nor the reader fetch survives.
        let stored: anybytes::Bytes = handles.payload().clone();
        anyhow::ensure!(
            (stored.as_ptr() as usize).is_multiple_of(256),
            "native tensor {name:?} is not 256-byte aligned"
        );
        let stored = stored
            .view::<[f32]>()
            .map_err(|error| anyhow::anyhow!("decode native tensor {name:?}: {error}"))?;
        let (wanted, wanted_shape) = mary::nn::weight_loader::get_tensor_f32(&source, name);
        anyhow::ensure!(
            handles.shape() == wanted_shape,
            "native tensor {name:?} shape differs from source"
        );
        anyhow::ensure!(
            stored.len() == wanted.len(),
            "native tensor {name:?} length differs from source"
        );
        for (index, (&stored, &wanted)) in stored.iter().zip(wanted.iter()).enumerate() {
            anyhow::ensure!(
                stored.to_bits() == wanted.to_bits(),
                "native tensor {name:?}[{index}] differs from source"
            );
        }
        elements += stored.len();
    }
    Ok((expected.len(), elements))
}

fn verify_alignment(
    weights: &VoxtralWeights<triblespace::core::repo::pile::PileSnapshot>,
) -> anyhow::Result<()> {
    for (name, handles) in weights.exact().iter().chain(weights.f16()) {
        // Alignment is a property of the payload, which is where the tensor
        // data actually starts — one 256-byte header into a 256-aligned record
        // for a typed leaf, and at the record itself for a legacy one.
        let bytes: anybytes::Bytes = handles.payload().clone();
        anyhow::ensure!(
            (bytes.as_ptr() as usize).is_multiple_of(256),
            "tensor {name:?} is not 256-byte aligned"
        );
    }
    Ok(())
}

/// Every id the source assigns bytes to decodes to exactly those bytes from
/// the pile, and the pile names no id the source does not.
fn verify_tekken(source: &TekkenSource, stored: &Tekken) -> anyhow::Result<usize> {
    for id in source.piece_ids() {
        anyhow::ensure!(
            source.piece(id) == Some(stored.piece(id)),
            "Tekken piece {id} differs between tekken.json and the pile"
        );
    }
    let pieces = source.piece_ids().len();
    anyhow::ensure!(
        stored.len() == pieces,
        "the pile's Tekken has {} pieces, tekken.json {pieces}",
        stored.len()
    );
    Ok(pieces)
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: voxtral_persist <model-dir> <pile-path> <signing-key>");
        std::process::exit(2);
    }
    let model_dir = Path::new(&args[1]);
    let pile_path = Path::new(&args[2]);
    let signing_key = signing_key_file::load_existing(Path::new(&args[3]))?;

    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(pile_path)
    {
        Ok(_) => eprintln!("voxtral_persist: created new empty pile {pile_path:?}"),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }

    let started = Instant::now();
    let mut pile = Pile::open(pile_path)
        .map_err(|error| anyhow::anyhow!("open model pile {pile_path:?}: {error}"))?;
    let imported = (|| -> anyhow::Result<_> {
        eprintln!("[voxtral] importing exact checkpoint {SOURCE}");
        let (exact_root, _exact_commit) = mary::persist::import_model_to_collection(
            &mut pile,
            &signing_key,
            model_dir,
            LeafDtype::F32,
            SOURCE,
            mary::persist::QUANTIZATION_NATIVE,
        )?;

        // Freeze before deriving: later appends cannot move the source root or
        // the bytes under the conversion. This line previously tried to say it
        // by naming the just-published commit — `&[exact_commit]` — which never
        // typechecked, because `Support` is not a slice of commits and
        // TribleSpace exposes NO public constructor for one (every
        // `Cover::from_data`/`from_members`/`from_patch` is `pub(crate)`, see
        // triblespace-core/src/collection/api.rs:277-303). Freezing the
        // observation is what the comment actually wanted, and the snapshot
        // epoch gives it directly: this cover is discovered inside a prefix
        // that cannot advance, so a later append cannot widen it. The
        // subsequent selector still picks exactly this import's
        // (SOURCE, NATIVE) root out of it.
        let store = pile
            .snapshot()
            .map_err(|error| anyhow::anyhow!("freeze model pile observation: {error}"))?;
        let exact_snapshot = mary::model_collection::snapshot_model_collection_in(&store)?;
        let exact = SelectedModelIndex::from_snapshot(
            exact_snapshot,
            ModelSelector::Source {
                source: SOURCE,
                quantization: mary::persist::QUANTIZATION_NATIVE,
            },
        )?;
        eprintln!("[voxtral] deriving full f16 root in the same pile");
        let (f16_root, _f16_commit, derived_count, derived_elements) =
            mary::persist::derive_selected_f16_to_collection(
                &mut pile,
                &signing_key,
                exact,
                SOURCE,
                QUANTIZATION_F16,
            )?;

        // The tokenizer travels in the same pile: `tekken.json` is read here
        // and never again at runtime.
        eprintln!("[voxtral] importing Tekken tokenizer");
        let tekken_source = TekkenSource::parse(&std::fs::read(model_dir.join("tekken.json"))?)?;
        let tokenizer = tekken_source.to_fragment(SOURCE, &mut pile)?;
        mary::model_collection::publish_model_fragment(&mut pile, &signing_key, tokenizer)?;

        // Gate the exact local prefix the live runtime admits, including any
        // previously published coordinate conflicts or invalid native records.
        let complete = mary::model_collection::snapshot_model_collection_local_latest(&mut pile)?;
        let tekken = Tekken::from_snapshot(&complete)?;
        let tekken_gate = verify_tekken(&tekken_source, &tekken)?;
        let weights = VoxtralWeights::from_snapshot(complete)?;
        anyhow::ensure!(
            weights.roots() == (exact_root, f16_root),
            "locally admitted Voxtral roots differ from this import"
        );
        let source_gate = verify_exact_source(model_dir, &weights)?;
        let f16_gate = weights.validate_f16_parity()?;
        verify_alignment(&weights)?;
        anyhow::ensure!(source_gate == f16_gate, "source and f16 gates disagree");
        anyhow::ensure!(
            f16_gate == (derived_count, derived_elements),
            "derived counters disagree with the admitted cohort"
        );
        Ok((exact_root, f16_root, f16_gate, tekken_gate))
    })();

    // `close` is the sole durability boundary, on success and on an import
    // error. No helper above flushes, reopens, repairs, or truncates the pile.
    let close = pile
        .close()
        .map_err(|error| anyhow::anyhow!("close model pile {pile_path:?}: {error}"));
    let (exact_root, f16_root, (tensors, elements), pieces) = match (imported, close) {
        (Ok(result), Ok(())) => result,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(close_error)) => {
            return Err(error.context(format!("import also failed to close pile: {close_error}")));
        }
    };

    let size = std::fs::metadata(pile_path)?.len();
    println!(
        "Voxtral native cohort valid: exact={exact_root}, f16={f16_root}; \
         {tensors} tensors / {elements} elements source- and f16-bit-identical; \
         all payloads 256-aligned; Tekken {pieces} pieces byte-identical to tekken.json; \
         pile {:.2} GiB in {:.1}s",
        size as f64 / (1_u64 << 30) as f64,
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    run()
}
