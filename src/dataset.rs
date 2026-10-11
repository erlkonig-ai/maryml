//! A generic, modality-blind ML dataset / sample-management schema on
//! TribleSpace.
//!
//! Four entity shapes, distinguished by their discriminating attributes (the
//! presence of `dataset_version`, `sample_dataset`, `run_dataset`, or
//! `pref_chosen` says which shape an entity is — no separate kind tag needed):
//!
//!   - **Dataset** — a named, versioned snapshot. Identified by the canonical
//!     [`metadata::name`] plus [`dataset_version`]. Append-only: each version is
//!     a *distinct* entity, so "the v3 snapshot" is a stable address forever and
//!     a new version never mutates an old one.
//!   - **Sample** — one training example. Every modality payload is a `Handle`
//!     to a content-addressed blob: [`image`],
//!     [`text`], [`sound`] are all optional and independent, so a sample can be
//!     unimodal or multimodal with the same shape. [`split`] tags train/val/test,
//!     [`sample_dataset`] links it to its Dataset, [`source`] optionally records
//!     where it came from, and [`metadata::tag`] carries free-form labels.
//!   - **TrainingRun** — provenance: which data trained which model with what
//!     result. [`run_dataset`] → the Dataset snapshot it consumed, [`run_model`]
//!     the resulting weights, [`run_hyperparams`] / [`run_metrics`] the JSON of
//!     the config and the outcome. "What produced this model?" is one query.
//!   - **Preference** — a pairwise judgement for DPO/RLHF/self-curation:
//!     [`pref_chosen`] / [`pref_rejected`] → two Samples, [`pref_score`] the
//!     (signed) margin, [`pref_judge`] who or what judged.
//!
//! This namespace is deliberately domain-free: it knows nothing about any one
//! dataset's labels. Domain attributes (a style, an expression, a difficulty…)
//! are just *user-minted attributes hung on the same Sample entities* — the open
//! schema means a consumer extends the model by minting its own typed attributes,
//! never by forking this library.

use triblespace::prelude::blobencodings::{RawBytes, UTF8String};
use triblespace::prelude::inlineencodings::{F64, GenId, Handle};
use triblespace::prelude::*;

attributes! {
    // ── Dataset ──────────────────────────────────────────────────────────
    /// The version label of a Dataset snapshot (e.g. "v3", "2026-06-30").
    /// Identity is (`metadata::name`, `dataset_version`); each version is a
    /// distinct, immutable entity, so the history of a dataset is just the set
    /// of its snapshots — append-only, never overwritten.
    /// Minted 2026-06-30 (trible genid).
    "8644CC9146EA9348DB5CF401CD183724" as pub dataset_version: Handle<UTF8String>;

    // ── Sample (modality payloads — all Handles, all optional) ───────────
    /// The image payload of a sample (raw encoded bytes — PNG/JPEG/…; the codec
    /// is the dataset's concern, not the schema's). Minted 2026-06-30.
    "58659FC179EED13E9385B82F1FDF95FE" as pub image: Handle<RawBytes>;
    /// The text payload of a sample (a prompt, caption, transcript, document…).
    /// `UTF8String` blob: text routinely exceeds the 32-byte inline ceiling.
    /// Minted 2026-06-30.
    "806AF895E3D21D3147908D36D542F367" as pub text: Handle<UTF8String>;
    /// The audio payload of a sample (raw encoded bytes — WAV/FLAC/MP3/…). No
    /// audio-specific blob schema exists in the substrate and none is warranted:
    /// `RawBytes` carries the encoded stream and the container format is implied
    /// by the dataset, exactly as for `image`. Minted 2026-06-30.
    "B83C144EF2DDEFD3378E13486908D18C" as pub sound: Handle<RawBytes>;
    /// Which split a sample belongs to: "train" / "val" / "test", as text.
    /// Minted 2026-06-30.
    "70644EA68C6CF91C2C6ABC8BF96982F9" as pub split: Handle<UTF8String>;
    /// Membership edge: a sample → the Dataset snapshot it belongs to.
    /// Minted 2026-06-30.
    "7AB92787B113098EFD8700183002892A" as pub sample_dataset: GenId;
    /// Optional provenance edge: a sample → the entity it was derived from
    /// (an upstream sample, a source record, a generating run…). Minted 2026-06-30.
    "CC4B192933D3182E7BB3BC892156690D" as pub source: GenId;

    // ── TrainingRun ──────────────────────────────────────────────────────
    /// A training run → the Dataset snapshot it trained on. Minted 2026-06-30.
    "C303BE459419B7AF6D5CF32C2B447BE8" as pub run_dataset: GenId;
    /// The trained weights produced by a run (raw serialized bytes — the run's
    /// own model blob; the mary model-graph format is a separate, finer-grained
    /// representation). Minted 2026-06-30.
    "779F9D64D1783D0BDB19EF8E192C3798" as pub run_model: Handle<RawBytes>;
    /// A run's hyperparameters as a JSON document (`UTF8String` blob).
    /// Minted 2026-06-30.
    "A8FE4B4E92F2819827553C462E41C013" as pub run_hyperparams: Handle<UTF8String>;
    /// A run's final metrics as a JSON document (`UTF8String` blob).
    /// Minted 2026-06-30.
    "CEA456D63AE1F5ABC84292C511EA27EA" as pub run_metrics: Handle<UTF8String>;

    // ── Preference (DPO / RLHF / self-curation) ──────────────────────────
    /// A preference → the Sample that was chosen (preferred). Minted 2026-06-30.
    "DDB8E9FB2EAB95F510281E4E20DF1178" as pub pref_chosen: GenId;
    /// A preference → the Sample that was rejected. Minted 2026-06-30.
    "CE977BD5B53BFF5B9C16D25AA6D44AB5" as pub pref_rejected: GenId;
    /// The (signed) preference margin / reward score. A real-valued `F64` inline
    /// value — standard double precision is the right grain for a learning
    /// signal. Minted 2026-06-30.
    "DF0496BEE2DC1AA8FFE27DB90187D5E2" as pub pref_score: F64;
    /// Who or what produced the judgement (a model handle, a rater name…), as
    /// text. Minted 2026-06-30.
    "60811D43F4E1DA286624403CBC1BFB91" as pub pref_judge: Handle<UTF8String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::metadata;

    /// Build a Dataset snapshot, a multimodal (image + text) Sample in it, a
    /// TrainingRun over it, and a Preference between two Samples — then query
    /// them back through `pattern!`, proving the schema round-trips facts +
    /// blobs through a TribleSet / BlobStore exactly as the rest of mary does.
    #[test]
    fn dataset_sample_run_preference_roundtrip() {
        let mut blobs = MemoryBlobStore::new();
        let mut tribles = TribleSet::new();

        // ── a Dataset snapshot ────────────────────────────────────────────
        let dataset = ufoid();
        let dataset_id: Id = *dataset;
        tribles += entity! { &dataset @
            metadata::name: blobs.put::<UTF8String, _>("portraits".to_string()).unwrap(),
            dataset_version: "v1",
        };

        // ── a multimodal Sample (image + text) in that dataset ────────────
        let png: &[u8] = &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let sample_a = ufoid();
        let sample_a_id: Id = *sample_a;
        tribles += entity! { &sample_a @
            image: blobs.put::<RawBytes, _>(png.to_vec()).unwrap(),
            text: blobs.put::<UTF8String, _>("a soft portrait in warm light".to_string()).unwrap(),
            split: "train",
            sample_dataset: &dataset,
        };

        // ── a second, text-only Sample (for the preference pair) ──────────
        let sample_b = ufoid();
        let sample_b_id: Id = *sample_b;
        tribles += entity! { &sample_b @
            text: blobs.put::<UTF8String, _>("a harsh portrait in flat light".to_string()).unwrap(),
            split: "train",
            sample_dataset: &dataset,
        };

        // ── a TrainingRun over the dataset ────────────────────────────────
        let run = ufoid();
        let run_id: Id = *run;
        tribles += entity! { &run @
            run_dataset: &dataset,
            run_model: blobs.put::<RawBytes, _>(vec![1u8, 2, 3, 4]).unwrap(),
            run_hyperparams: blobs.put::<UTF8String, _>(r#"{"lr":3e-4,"epochs":3}"#.to_string()).unwrap(),
            run_metrics: blobs.put::<UTF8String, _>(r#"{"loss":0.12,"acc":0.94}"#.to_string()).unwrap(),
        };

        // ── a Preference between the two samples ──────────────────────────
        let pref = ufoid();
        let pref_id: Id = *pref;
        tribles += entity! { &pref @
            pref_chosen: &sample_a,
            pref_rejected: &sample_b,
            pref_score: 1.5f64,
            pref_judge: "human-rater",
        };

        // ── query: find the train samples of this dataset ─────────────────
        let train_samples: Vec<Id> = find!(
            (s: Id),
            pattern!(&tribles, [{
                ?s @
                    sample_dataset: dataset_id,
                    split: "train",
            }])
        )
        .map(|(s,)| s)
        .collect();
        assert_eq!(
            train_samples.len(),
            2,
            "both samples are train-split members"
        );
        assert!(train_samples.contains(&sample_a_id));
        assert!(train_samples.contains(&sample_b_id));

        // ── query: the multimodal sample (has BOTH image and text) ────────
        let multimodal: Vec<Id> = find!(
            (s: Id, img: Inline<Handle<RawBytes>>, txt: Inline<Handle<UTF8String>>),
            pattern!(&tribles, [{
                ?s @
                    sample_dataset: dataset_id,
                    image: ?img,
                    text: ?txt,
            }])
        )
        .map(|(s, _, _)| s)
        .collect();
        assert_eq!(
            multimodal,
            vec![sample_a_id],
            "only sample_a carries both modalities"
        );

        // ── query: provenance — which dataset trained the run's model ─────
        let (trained_on,) = find!(
            (d: Id),
            pattern!(&tribles, [{ run_id @ run_dataset: ?d }])
        )
        .next()
        .expect("run links to its dataset");
        assert_eq!(trained_on, dataset_id);

        // ── query: the preference's chosen sample + its score ─────────────
        let (chosen, score) = find!(
            (c: Id, sc: f64),
            pattern!(&tribles, [{ pref_id @ pref_chosen: ?c, pref_score: ?sc }])
        )
        .next()
        .expect("preference has a chosen sample and a score");
        assert_eq!(chosen, sample_a_id);
        assert_eq!(score, 1.5f64);

        // blobs really landed (the multimodal sample's text resolves).
        let reader = SnapshotSource::snapshot(&mut blobs).unwrap();
        let (txt_handle,) = find!(
            (t: Inline<Handle<UTF8String>>),
            pattern!(&tribles, [{ sample_a_id @ text: ?t }])
        )
        .next()
        .unwrap();
        let resolved: anybytes::View<str> = reader.get(txt_handle).unwrap();
        assert_eq!(&*resolved, "a soft portrait in warm light");
    }
}
