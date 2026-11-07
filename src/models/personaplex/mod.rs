//! PersonaPlex-7B full-duplex model port. Phase 1: the Mimi neural audio codec
//! ([`mimi`]). Phase 0 of the LM port: verified architecture constants in
//! [`config`] (from the real checkpoint's tensor shapes) and the weight pile
//! (`personaplex_persist`). LM part 1: the 7B [`temporal`] transformer
//! forward. LM part 2: the [`depth`] transformer (depformer) and the
//! [`lmgen`] delay/undelay step machinery (all CPU-f32 parity vs the moshi
//! oracle — `personaplex_probe`). LM part 3: the end-to-end [`pipeline`] —
//! input WAV → Mimi encode → LM free-run → agent streams 1..=8 → Mimi decode
//! → 24 kHz audio out. Realtime lane A: the [`temporal_metal`] q4/Metal
//! decode build (feature `q4`; gated by `personaplex_rt_probe`). Realtime
//! lane B: the depformer on either arm — [`depth_fast`], the Accelerate/NEON
//! CPU predictor (preloaded per-step weight sets, fixed buffers, optional f16
//! storage with f32 accumulate; gate+bench `moshi_depth_probe`), and [`depth_gpu`], the
//! cubecl port of the same math with the autoregressive chain AND the sampler
//! on the device (gate+bench `depth_gpu_probe`, paired against the CPU arm by
//! `personaplex_rt_probe depthab`). [`pipeline`] picks one via
//! `PERSONAPLEX_DEPTH`. Phase 5: the prompt
//! machinery — [`spm`] (pure-Rust SentencePiece unigram text tokenizer, encode
//! + decode), [`voice_prompt`] (packaged voice `.pt` reader) and [`prompt`]
//! (system prompts assembled from primary sources instead of golden npys;
//! gated by `personaplex_probe prompt`). Realtime foundation: [`sampling`]
//! (seedable temperature / top-k / top-p over the text + audio heads; greedy
//! stays the parity default) and the `reset_session` seam on the [`lmgen`] /
//! [`pipeline`] step machines (a new conversation without a weight reload).

use crate::leaf::{Elem, Leaf};
use crate::model_collection::ModelSnapshot;
use crate::nn::weight_loader::WeightLoader;
use crate::selection::{ModelSelector, SelectedModelIndex};
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::core::collection::{CollectionData, Support};
use triblespace::core::repo::pile::PileSnapshot;
use triblespace::prelude::{BlobStoreGet, Id, Inline, TribleSet, blobencodings, inlineencodings};

/// Canonical source coordinate of the complete PersonaPlex LM + Mimi model.
pub const SOURCE: &str = "nvidia/personaplex-7b-v1";

/// One exact PersonaPlex model selected from a self-contained model graph.
///
/// The root is the union of the LM and Mimi checkpoint tensors. Runtime code
/// deliberately accepts only faithful f32 leaves under [`SOURCE`] and
/// `quantization="native"`. Steady-state loads select it from one member
/// archive of the signed `models` collection; direct graph selection exists
/// only for validating a staged candidate before its COMMIT is finalized.
pub struct PersonaPlexWeights<R> {
    selected: SelectedModelIndex<R>,
}

/// Exact signed authority retained alongside one PersonaPlex loader: the
/// collection member that states the whole model, and its root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonaPlexAuthority {
    model_root: Id,
    model_archive_data: CollectionData,
    support: Support,
}

impl PersonaPlexAuthority {
    pub fn collection(&self) -> crate::model_collection::ModelCollection {
        self.support.collection()
    }

    pub fn model_root(&self) -> Id {
        self.model_root
    }

    /// The collection member: the complete canonical model-fact archive.
    pub fn model_archive_data(&self) -> CollectionData {
        self.model_archive_data
    }

    /// Exact admitted foundational support frozen with this loader.
    pub fn support(&self) -> &Support {
        &self.support
    }
}

/// PersonaPlex weights plus the exact authority from which they were loaded.
pub struct PersonaPlexModel<R> {
    weights: PersonaPlexWeights<R>,
    authority: PersonaPlexAuthority,
}

/// Authority-bound loader for deterministic runtime transforms.
///
/// Its fields are private so a caller cannot label weights from one member
/// with another member's `(collection, root, archive)` identity.
pub struct PersonaPlexRuntimeSource {
    authority: PersonaPlexAuthority,
    loader: WeightLoader,
}

impl PersonaPlexRuntimeSource {
    pub fn authority(&self) -> &PersonaPlexAuthority {
        &self.authority
    }

    pub(crate) fn loader(&self) -> &WeightLoader {
        &self.loader
    }
}

impl<R> PersonaPlexModel<R> {
    pub fn authority(&self) -> &PersonaPlexAuthority {
        &self.authority
    }

    pub fn weights(&self) -> &PersonaPlexWeights<R> {
        &self.weights
    }
}

impl<R: BlobStoreGet> PersonaPlexWeights<R> {
    fn admit(selected: SelectedModelIndex<R>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            selected.single_root().is_some(),
            "PersonaPlex exact weights require one complete model root, found {}",
            selected.roots().len()
        );
        if let Some(name) = selected
            .handles()
            .iter()
            .filter_map(|(name, leaf)| (leaf.elem() == Elem::F16).then_some(name))
            .min()
        {
            anyhow::bail!("PersonaPlex exact tensor {name:?} is not f32");
        }
        Ok(Self { selected })
    }

    /// Select the canonical exact root from an explicit graph and owning
    /// reader. This supports commit-last validation of staged candidate facts.
    pub fn from_graph(facts: &TribleSet, reader: R) -> anyhow::Result<Self> {
        let selected = SelectedModelIndex::from_graph(
            facts,
            reader,
            ModelSelector::Source {
                source: SOURCE,
                quantization: crate::persist::QUANTIZATION_NATIVE,
            },
        )?;
        Self::admit(selected)
    }

    /// Select PersonaPlex from the members of one `models` snapshot. Each
    /// member is a self-contained model-fact archive; a member stating one
    /// Source/native PersonaPlex root is that model. Members stating none are
    /// other models or annotations and are skipped. The same root stated by
    /// several members is the same model (its id is its content); the first
    /// member in canonical order is its authority. Two different roots fail
    /// closed, and so does one member stating two such roots, because nothing
    /// names which of them is meant.
    pub fn find_in_snapshot(
        snapshot: ModelSnapshot<R>,
    ) -> anyhow::Result<Option<PersonaPlexModel<R>>>
    where
        R: Clone,
    {
        use anyhow::{Context, anyhow};
        let (_, support, reader) = snapshot.into_parts();
        let mut selected: Option<(Self, Id, CollectionData)> = None;
        for member in support.members() {
            let model_archive_data = inlineencodings::Handle::<SimpleArchive>::to_hash(member);
            let blob: Blob<SimpleArchive> = reader
                .get(member)
                .map_err(|error| anyhow!("read model member {member:?}: {error}"))?;
            triblespace::core::collection::simplearchive_union::validate_element(&blob)
                .map_err(|error| anyhow!("model member {member:?} is not canonical: {error}"))?;
            let facts = TribleSet::try_from_blob(blob)
                .with_context(|| format!("decode model member {member:?}"))?;
            let mut roots = Vec::new();
            for (root,) in triblespace::prelude::find!(
                (root: Id),
                triblespace::prelude::pattern!(
                    &facts,
                    [{ ?root @ crate::format::attrs::quantization: crate::persist::QUANTIZATION_NATIVE }]
                )
            ) {
                let mut source_matches = false;
                for (handle,) in triblespace::prelude::find!(
                    (source: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                    triblespace::prelude::pattern!(&facts, [{ root @ crate::format::attrs::source: ?source }])
                ) {
                    let value: anybytes::View<str> = reader
                        .get(handle)
                        .map_err(|error| anyhow!("read PersonaPlex source coordinate: {error}"))?;
                    source_matches |= value.as_ref() == SOURCE;
                }
                // Other source/format labels are annotations on the same root.
                if source_matches && !roots.contains(&root) {
                    roots.push(root);
                }
            }
            let root = match roots.as_slice() {
                [] => continue,
                [root] => *root,
                many => anyhow::bail!(
                    "model member {member:?} states {} PersonaPlex Source/native roots; nothing names one",
                    many.len()
                ),
            };
            let weights = Self::admit(SelectedModelIndex::from_graph(
                &facts,
                reader.clone(),
                ModelSelector::Root(root),
            )?)
            .with_context(|| format!("validate PersonaPlex member {member:?}"))?;
            anyhow::ensure!(
                weights.root() == root,
                "model member {member:?} states root {root}, but selects {}",
                weights.root()
            );
            if let Some((_, existing_root, existing_h)) = &selected {
                anyhow::ensure!(
                    *existing_root == root,
                    "ambiguous PersonaPlex authority: ({existing_root}, {existing_h:?}) and ({root}, {model_archive_data:?})"
                );
            } else {
                selected = Some((weights, root, model_archive_data));
            }
        }

        let Some((weights, model_root, model_archive_data)) = selected else {
            return Ok(None);
        };
        Ok(Some(PersonaPlexModel {
            weights,
            authority: PersonaPlexAuthority {
                model_root,
                model_archive_data,
                support,
            },
        }))
    }

    /// Required form of [`Self::find_in_snapshot`].
    pub fn from_snapshot(snapshot: ModelSnapshot<R>) -> anyhow::Result<PersonaPlexModel<R>>
    where
        R: Clone,
    {
        Self::find_in_snapshot(snapshot)?
            .ok_or_else(|| anyhow::anyhow!("no admitted PersonaPlex model in exact support"))
    }

    /// Content-addressed root of the complete exact model.
    pub fn root(&self) -> triblespace::prelude::Id {
        self.selected
            .single_root()
            .expect("PersonaPlex admission requires one exact root")
    }

    /// Number of tensors in the LM + Mimi union.
    pub fn count(&self) -> usize {
        self.selected.handles().len()
    }

    /// Exact tensor index retained for source-parity gates.
    pub fn exact(&self) -> &std::collections::HashMap<String, Leaf> {
        self.selected.handles()
    }

    /// Reader owning every attachment named by [`Self::exact`].
    pub fn reader(&self) -> &R {
        self.selected.reader()
    }
}

impl PersonaPlexWeights<PileSnapshot> {
    /// Consume the exact model into the platform's established lazy loader.
    pub fn into_loader(self) -> WeightLoader {
        let (_, exact, _reader) = self.selected.into_parts();
        #[cfg(target_os = "macos")]
        {
            return WeightLoader::Aliased(crate::nn::weight_loader::AliasedPile::new(
                std::collections::HashMap::new(),
                exact,
                crate::nn::backend::WgpuDevice::default(),
            ));
        }
        #[cfg(not(target_os = "macos"))]
        {
            let keymap = exact
                .into_iter()
                .map(|(name, leaf)| (name, leaf.to_f32_shape()))
                .collect();
            WeightLoader::Pile(keymap)
        }
    }
}

impl PersonaPlexModel<PileSnapshot> {
    /// Split a collection-native load into its immutable authority and runtime
    /// loader without reopening the pile or observing a wider prefix.
    pub fn into_parts(self) -> (PersonaPlexAuthority, WeightLoader) {
        (self.authority, self.weights.into_loader())
    }

    /// Preserve the authority/loader binding for runtime transforms.
    pub fn into_runtime_source(self) -> PersonaPlexRuntimeSource {
        PersonaPlexRuntimeSource {
            authority: self.authority,
            loader: self.weights.into_loader(),
        }
    }
}

pub mod config;
pub mod depth;
pub mod depth_fast;
#[cfg(feature = "q4")]
pub mod depth_gpu;
pub mod lmgen;
pub mod mimi;
pub mod pipeline;
pub mod prompt;
pub mod sampling;
pub mod spm;
pub mod temporal;
#[cfg(feature = "q4")]
pub mod temporal_metal;
pub mod voice_prompt;

#[cfg(test)]
mod native_authority_tests {
    use super::*;
    use crate::format::{F32Array, U64Array, attrs};
    use ed25519_dalek::SigningKey;
    use std::fs::OpenOptions;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use triblespace::core::repo::pile::Pile;
    use triblespace::prelude::blobencodings::UTF8String;
    use triblespace::prelude::*;

    static NEXT_TEST_PILE: AtomicU64 = AtomicU64::new(0);

    struct TestPile(PathBuf);

    impl TestPile {
        fn new() -> Self {
            let ordinal = NEXT_TEST_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-personaplex-native-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create synthetic PersonaPlex pile");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestPile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn model_fragment(
        tensors: &[(&str, &[f32], &[u64])],
        f16: bool,
        form: crate::leaf::Form,
    ) -> Fragment {
        let mut fragment = Fragment::empty();
        let mut members = Vec::new();
        for &(tensor, values, dimensions) in tensors {
            let leaf_id = crate::leaf::fixture_leaf(
                &mut fragment,
                form,
                if f16 {
                    crate::leaf::Elem::F16
                } else {
                    crate::leaf::Elem::F32
                },
                dimensions,
                values,
            );
            let name = fragment.put::<UTF8String, _>(tensor.to_owned());
            let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: leaf_id };
            members.push(member.root().expect("model member root"));
            fragment += member;
        }
        let root = entity! { _ @ attrs::member*: members.iter() };
        let root_id = root.root().expect("model root");
        fragment += root;
        let source = fragment.put::<UTF8String, _>(SOURCE.to_owned());
        fragment += entity! { ExclusiveId::force_ref(&root_id) @
            attrs::source: source,
            attrs::quantization: crate::persist::QUANTIZATION_NATIVE,
        };
        let (_, facts, metafacts, blobs) = fragment.into_parts();
        Fragment::rooted_from_parts(root_id, facts, metafacts, blobs)
    }

    #[test]
    fn exact_member_is_selected_and_repeated_publication_appends_nothing() {
        let file = TestPile::new();
        let fragment = model_fragment(
            &[
                ("transformer.weight", &[1.0, 2.0], &[2]),
                ("encoder.weight", &[3.0, 4.0], &[1, 2]),
            ],
            false,
            crate::leaf::Form::TwoBlob,
        );
        let root = fragment.root().expect("PersonaPlex model root");
        let signing_key = SigningKey::from_bytes(&[0x50; 32]);
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let first = crate::model_collection::publish_model_fragment(&mut pile, &signing_key, fragment.clone())
        .expect("publish exact PersonaPlex root");
        let len_after_first = std::fs::metadata(file.path()).unwrap().len();
        let repeated = crate::model_collection::publish_model_fragment(&mut pile, &signing_key, fragment)
        .expect("repeat exact PersonaPlex publication");
        let len_after_retry = std::fs::metadata(file.path()).unwrap().len();
        assert_eq!(first, repeated);
        assert_eq!(len_after_first, len_after_retry);

        let snapshot =
            crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
                .expect("freeze exact PersonaPlex prefix");
        let model = PersonaPlexWeights::from_snapshot(snapshot)
            .expect("select exact PersonaPlex model");
        let authority = model.authority();
        assert_eq!(authority.model_root(), root);
        assert_eq!(authority.model_archive_data(), first.data());
        assert_eq!(authority.support().len(), 1);
        assert!(
            authority
                .support()
                .contains(inlineencodings::Handle::<SimpleArchive>::from_hash(
                    first.data()
                ))
        );
        let weights = model.weights();
        assert_eq!(weights.root(), root);
        assert_eq!(weights.count(), 2);
        assert!(weights.exact().contains_key("transformer.weight"));
        assert!(weights.exact().contains_key("encoder.weight"));
        drop(model);
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    /// Without a token naming the root, a member stating two PersonaPlex
    /// Source/native roots does not choose one by accident.
    #[test]
    fn member_with_two_personaplex_roots_fails_closed() {
        let file = TestPile::new();
        let mut asserted = model_fragment(
            &[("asserted.weight", &[1.0], &[1])],
            false,
            crate::leaf::Form::TwoBlob,
        );
        let root = asserted.root().expect("asserted PersonaPlex root");
        let ambient = model_fragment(
            &[("ambient.weight", &[2.0], &[1])],
            false,
            crate::leaf::Form::TwoBlob,
        );
        asserted += ambient;
        let (_, facts, metafacts, blobs) = asserted.into_parts();
        let combined = Fragment::rooted_from_parts(root, facts, metafacts, blobs);

        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        crate::model_collection::publish_model_fragment(
            &mut pile,
            &SigningKey::from_bytes(&[0x50; 32]),
            combined,
        )
        .expect("publish a member with two roots");
        let snapshot = crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
            .expect("freeze multi-root PersonaPlex member");
        let error = PersonaPlexWeights::from_snapshot(snapshot)
            .err()
            .expect("two roots in one member must fail closed");
        assert!(format!("{error:#}").contains("2 PersonaPlex Source/native roots"), "{error:#}");
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    /// The same root in two members is one model, not an ambiguity.
    #[test]
    fn same_root_in_two_members_is_one_model() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let key = SigningKey::from_bytes(&[0x50; 32]);
        let fragment = model_fragment(&[("weight", &[1.0], &[1])], false, crate::leaf::Form::TwoBlob);
        let root = fragment.root().unwrap();
        let mut annotated = fragment.clone();
        annotated += entity! { _ @ attrs::kind: "an unrelated annotation" };
        crate::model_collection::publish_model_fragment(&mut pile, &key, fragment).unwrap();
        crate::model_collection::publish_model_fragment(&mut pile, &key, annotated).unwrap();
        let snapshot = crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
            .expect("freeze two members");
        assert_eq!(snapshot.support().len(), 2);
        let model = PersonaPlexWeights::from_snapshot(snapshot).expect("one model in two members");
        assert_eq!(model.authority().model_root(), root);
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    /// A tokenizer or other model published as its own member is skipped.
    #[test]
    fn other_members_are_skipped() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let key = SigningKey::from_bytes(&[0x50; 32]);
        crate::model_collection::publish_model_fragment(
            &mut pile,
            &key,
            entity! { _ @ attrs::kind: "tokenizer" },
        )
        .expect("publish an unrelated member");
        let fragment = model_fragment(&[("weight", &[1.0], &[1])], false, crate::leaf::Form::TwoBlob);
        let root = fragment.root().unwrap();
        crate::model_collection::publish_model_fragment(&mut pile, &key, fragment)
            .expect("publish PersonaPlex member");
        let snapshot = crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
            .expect("freeze two members");
        assert_eq!(snapshot.support().len(), 2);
        let model = PersonaPlexWeights::from_snapshot(snapshot).expect("select the PersonaPlex member");
        assert_eq!(model.authority().model_root(), root);
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn snapshot_retains_its_collection_identity() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let fragment = model_fragment(
            &[("weight", &[1.0], &[1])],
            false,
            crate::leaf::Form::TwoBlob,
        );
        crate::model_collection::publish_model_fragment(&mut pile, &SigningKey::from_bytes(&[0x50; 32]), fragment)
        .expect("publish PersonaPlex model");
        let snapshot =
            crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
                .expect("freeze PersonaPlex model");
        let collection = snapshot.support().collection();
        let model = PersonaPlexWeights::from_snapshot(snapshot)
            .expect("select exact PersonaPlex model");
        assert_eq!(model.authority().collection(), collection);
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn non_f32_exact_coordinate_fails_closed() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let fragment = model_fragment(
            &[("weight", &[1.0], &[1])],
            true,
            crate::leaf::Form::TwoBlob,
        );
        crate::model_collection::publish_model_fragment(&mut pile, &SigningKey::from_bytes(&[0x50; 32]), fragment)
        .expect("publish incompatible PersonaPlex root");
        let snapshot =
            crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
                .expect("freeze incompatible PersonaPlex prefix");
        let error = PersonaPlexWeights::from_snapshot(snapshot)
            .err()
            .expect("f16 exact coordinate must fail");
        assert!(format!("{error:#}").contains("is not f32"), "{error:#}");
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn root_accepts_an_additional_source_annotation() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let mut fragment = model_fragment(
            &[("weight", &[1.0], &[1])],
            false,
            crate::leaf::Form::TwoBlob,
        );
        let root = fragment.root().unwrap();
        let other_source = fragment.put::<UTF8String, _>("example/other".to_owned());
        fragment += entity! { ExclusiveId::force_ref(&root) @ attrs::source: other_source };
        crate::model_collection::publish_model_fragment(&mut pile, &SigningKey::from_bytes(&[0x50; 32]), fragment)
        .expect("publish multiply labelled PersonaPlex root");
        let snapshot =
            crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
                .expect("freeze multiply labelled PersonaPlex prefix");
        let model = PersonaPlexWeights::from_snapshot(snapshot)
            .expect("an additional source label must not invalidate a matching root");
        assert_eq!(model.authority().model_root(), root);
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn root_accepts_an_additional_quantization_annotation() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let mut fragment = model_fragment(
            &[("weight", &[1.0], &[1])],
            false,
            crate::leaf::Form::TwoBlob,
        );
        let root = fragment.root().unwrap();
        fragment += entity! { ExclusiveId::force_ref(&root) @ attrs::quantization: "other" };
        crate::model_collection::publish_model_fragment(&mut pile, &SigningKey::from_bytes(&[0x50; 32]), fragment)
        .expect("publish multiply labelled PersonaPlex root");
        let snapshot =
            crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
                .expect("freeze multiply labelled PersonaPlex prefix");
        let model = PersonaPlexWeights::from_snapshot(snapshot)
            .expect("an additional format label must not invalidate a matching root");
        assert_eq!(model.authority().model_root(), root);
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn conflicting_member_roots_fail_closed() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let existing = model_fragment(
            &[("weight", &[1.0], &[1])],
            false,
            crate::leaf::Form::TwoBlob,
        );
        let existing_commit = crate::model_collection::publish_model_fragment(&mut pile, &SigningKey::from_bytes(&[0x50; 32]), existing)
        .expect("publish existing PersonaPlex authority");

        let candidate = model_fragment(
            &[("weight", &[2.0], &[1])],
            false,
            crate::leaf::Form::TwoBlob,
        );
        let candidate_commit = crate::model_collection::publish_model_fragment(&mut pile, &SigningKey::from_bytes(&[0x50; 32]), candidate)
        .expect("publish conflicting PersonaPlex authority");
        let snapshot =
            crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
                .expect("freeze conflicting model authority");
        let error = PersonaPlexWeights::from_snapshot(snapshot)
            .err()
            .expect("conflicting Source/native roots must fail closed");
        assert!(format!("{error:#}").contains("ambiguous"), "{error:#}");
        assert_ne!(existing_commit.data(), candidate_commit.data());
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    /// A member selects and the exact coordinate keeps its shape and width
    /// under either leaf form. `models/personaplex.pile` is two-blob today; a
    /// typed conversion of it must land on the same weights, and this is where
    /// that stops being an assumption.
    #[test]
    fn personaplex_member_selects_in_either_leaf_form() {
        for form in crate::leaf::FORMS {
            let file = TestPile::new();
            let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
            let fragment = model_fragment(
                &[
                    ("transformer.weight", &[1.0, 2.0], &[2]),
                    ("encoder.weight", &[3.0, 4.0], &[1, 2]),
                ],
                false,
                form,
            );
            let root = fragment.root().expect("PersonaPlex model root");
            crate::model_collection::publish_model_fragment(&mut pile, &SigningKey::from_bytes(&[0x50; 32]), fragment)
            .expect("publish PersonaPlex root");
            let snapshot =
                crate::model_collection::snapshot_model_collection_local_latest(&mut pile)
                    .expect("freeze PersonaPlex prefix");
            let model = PersonaPlexWeights::from_snapshot(snapshot)
                .unwrap_or_else(|e| panic!("{} model must select: {e:#}", form.label()));
            let weights = model.weights();
            assert_eq!(weights.root(), root, "{}", form.label());
            assert_eq!(weights.count(), 2, "{}", form.label());
            let encoder = &weights.exact()["encoder.weight"];
            assert_eq!(encoder.shape(), vec![1, 2], "{}", form.label());
            assert_eq!(encoder.to_f32(), vec![3.0, 4.0], "{}", form.label());
            drop(model);
            pile.close().expect("close synthetic PersonaPlex pile");
        }
    }
}
