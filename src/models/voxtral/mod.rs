//! Voxtral-Mini-4B-Realtime-2602 — delay-conditioned streaming
//! speech-to-text (one AR step per 80 ms frame, text delayed 80 ms–2.4 s
//! behind the audio via ada-RMS-norm conditioning). Architecture + port log:
//! `docs/VOXTRAL_ARCH.md`. Runtime weights are two explicit roots in one
//! frozen native model-collection snapshot: the exact f32 checkpoint and its
//! full f16 derivation.

use crate::leaf::{Elem, Leaf};
use crate::model_collection::ModelSnapshot;
use crate::nn::weight_loader::WeightLoader;
use crate::selection::{ModelSelector, index_keymap_for_root, select_model_root};
use std::collections::HashMap;
use triblespace::core::repo::pile::PileSnapshot;
use triblespace::prelude::*;

/// Canonical source coordinate shared by Voxtral's exact and derived roots.
pub const SOURCE: &str = "mistralai/Voxtral-Mini-4B-Realtime-2602";

/// Quantization coordinate of Voxtral's full half-width derivation.
pub const QUANTIZATION_F16: &str = "f16";

/// The complete live Voxtral weight cohort selected from one immutable native
/// model-collection snapshot.
///
/// Both roots must describe exactly the same tensor-name/shape domain. The
/// exact root contains only f32 leaves and the derived root only f16 leaves.
/// Once those invariants are checked, only the two compact indexes and their
/// one owning reader remain resident; no sibling-file naming, repository
/// ancestry, or fallback root participates in runtime loading.
pub struct VoxtralWeights<R> {
    exact_root: Id,
    f16_root: Id,
    exact: HashMap<String, Leaf>,
    f16: HashMap<String, Leaf>,
    reader: R,
}

impl<R: BlobStoreGet> VoxtralWeights<R> {
    /// Select and validate the exact/f16 pair from one already-frozen snapshot.
    pub fn from_snapshot(snapshot: ModelSnapshot<R>) -> anyhow::Result<Self> {
        fn select(
            snapshot: &ModelSnapshot<impl BlobStoreGet>,
            quantization: &str,
        ) -> anyhow::Result<(Id, HashMap<String, Leaf>)> {
            let root = select_model_root(
                snapshot.facts(),
                snapshot.store(),
                ModelSelector::Source {
                    source: SOURCE,
                    quantization,
                },
            )?;
            let index = index_keymap_for_root(snapshot.facts(), snapshot.store(), root)?;
            Ok((root, index))
        }

        let (exact_root, exact) = select(&snapshot, crate::persist::QUANTIZATION_NATIVE)?;
        let (f16_root, f16) = select(&snapshot, QUANTIZATION_F16)?;

        for (name, leaf) in &exact {
            if leaf.elem() != Elem::F32 {
                anyhow::bail!("Voxtral exact tensor {name:?} is not f32");
            }
        }
        for (name, leaf) in &f16 {
            if leaf.elem() != Elem::F16 {
                anyhow::bail!("Voxtral derived tensor {name:?} is not f16");
            }
        }
        anyhow::ensure!(
            exact.len() == f16.len(),
            "Voxtral cohort name-set mismatch: {} exact tensors, {} f16 tensors",
            exact.len(),
            f16.len()
        );
        // Two different tensors agreeing on a shape is a cross-tensor fact, so
        // it is still checked here — but on the dims themselves rather than on
        // whether the two happened to share one content-addressed shape blob.
        for (name, exact_leaf) in &exact {
            let derived = f16
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Voxtral f16 root is missing tensor {name:?}"))?;
            anyhow::ensure!(
                exact_leaf.dims() == derived.dims(),
                "Voxtral tensor {name:?} is {:?} exact but {:?} derived",
                exact_leaf.dims(),
                derived.dims()
            );
        }

        let (_, _, reader) = snapshot.into_parts();
        Ok(Self {
            exact_root,
            f16_root,
            exact,
            f16,
            reader,
        })
    }

    /// Content-addressed roots of the exact and f16 models.
    pub const fn roots(&self) -> (Id, Id) {
        (self.exact_root, self.f16_root)
    }

    /// Number of tensors in each root (equal for every valid cohort).
    pub fn counts(&self) -> (usize, usize) {
        (self.exact.len(), self.f16.len())
    }

    /// Exact tensor index retained for source-parity gates.
    pub fn exact(&self) -> &HashMap<String, Leaf> {
        &self.exact
    }

    /// Half-width tensor index retained for derivation-parity gates.
    pub fn f16(&self) -> &HashMap<String, Leaf> {
        &self.f16
    }

    /// Reader owning every attachment named by both indexes.
    pub const fn reader(&self) -> &R {
        &self.reader
    }

    /// Verify every stored f16 bit against `f16::from_f32` of the exact root.
    /// Data is fetched and compared one tensor at a time.
    pub fn validate_f16_parity(&self) -> anyhow::Result<(usize, usize)> {
        let mut names: Vec<_> = self.exact.keys().collect();
        names.sort_unstable();
        let mut elements = 0;
        for name in &names {
            let exact_values = self.exact[*name]
                .view_f32()
                .ok_or_else(|| anyhow::anyhow!("decode exact tensor {name:?}"))?;
            let f16_values = self.f16[*name]
                .view_f16()
                .ok_or_else(|| anyhow::anyhow!("decode f16 tensor {name:?}"))?;
            anyhow::ensure!(
                exact_values.len() == f16_values.len(),
                "Voxtral tensor {name:?} has {} exact elements but {} f16 elements",
                exact_values.len(),
                f16_values.len()
            );
            for (index, (&exact, &derived)) in
                exact_values.iter().zip(f16_values.iter()).enumerate()
            {
                let wanted = half::f16::from_f32(exact);
                anyhow::ensure!(
                    wanted.to_bits() == derived.to_bits(),
                    "Voxtral tensor {name:?}[{index}] f16 bits differ from exact cast"
                );
            }
            elements += exact_values.len();
        }
        Ok((names.len(), elements))
    }
}

impl VoxtralWeights<PileSnapshot> {
    /// Consume the validated cohort into the platform's lazy runtime loader.
    ///
    /// macOS uses one `AliasedPile` over the shared reader. Other platforms
    /// load from the exact index's typed leaves, which view the pile's mapping:
    /// each tensor is decoded when the model asks for it, so the host holds
    /// one tensor's f32 copy at a time. Materializing the whole index up front
    /// held all 4.43 G elements as f32 (16.5 GiB) before the first upload.
    pub fn into_loader(self) -> WeightLoader {
        #[cfg(target_os = "macos")]
        {
            return WeightLoader::Aliased(crate::nn::weight_loader::AliasedPile::new(
                self.f16,
                self.exact,
                crate::nn::backend::WgpuDevice::default(),
            ));
        }
        #[cfg(not(target_os = "macos"))]
        {
            WeightLoader::Typed(self.exact)
        }
    }
}

pub mod config;
#[cfg(feature = "voxtral-cuda")]
mod argmax_cuda;
pub mod decoder;
pub mod encoder;
pub mod fast;
#[cfg(feature = "voxtral-cuda")]
mod gemv_cuda;
pub mod layers;
pub mod mel;
#[cfg(feature = "voxtral-cuda")]
mod norm_cuda;
pub mod pipeline;
#[cfg(feature = "voxtral-cuda")]
mod swiglu_cuda;
pub mod tokenizer;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{F32Array, U64Array, attrs};
    use ed25519_dalek::SigningKey;
    use std::fs::OpenOptions;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use triblespace::core::repo::pile::Pile;
    use triblespace::prelude::blobencodings::UTF8String;

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
                "mary-voxtral-native-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create synthetic Voxtral pile");
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

    fn component_fragment(
        quantization: &str,
        tensors: &[(&str, &[f32], &[u64])],
        f16: bool,
    ) -> Fragment {
        let mut fragment = Fragment::empty();
        let mut members = Vec::new();
        for &(tensor, values, dims) in tensors {
            let shape = fragment.put::<U64Array, _>(dims.to_vec());
            let leaf = if f16 {
                let values: Vec<_> = values.iter().copied().map(half::f16::from_f32).collect();
                let data = fragment.put::<crate::f16enc::F16Array, _>(values);
                entity! { _ @ attrs::data_f16: data, attrs::shape: shape }
            } else {
                let data = fragment.put::<F32Array, _>(values.to_vec());
                entity! { _ @ attrs::data: data, attrs::shape: shape }
            };
            let leaf_id = leaf.root().expect("tensor leaf root");
            fragment += leaf;
            let name = fragment.put::<UTF8String, _>(tensor.to_owned());
            let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: &leaf_id };
            members.push(member.root().expect("model member root"));
            fragment += member;
        }
        let source = fragment.put::<UTF8String, _>(SOURCE.to_owned());
        fragment += entity! { _ @
            attrs::source: source,
            attrs::quantization: quantization,
            attrs::member*: members.iter(),
        };
        fragment
    }

    fn publish(path: &Path, fragments: impl IntoIterator<Item = Fragment>) {
        let mut pile = Pile::open(path).expect("open synthetic Voxtral pile");
        for fragment in fragments {
            crate::model_collection::publish_model_fragment(
                &mut pile,
                &SigningKey::from_bytes(&[0x56; 32]),
                fragment,
            )
            .expect("publish native Voxtral component");
        }
        pile.close().expect("close synthetic Voxtral pile");
    }

    fn valid_cohort() -> [Fragment; 2] {
        let tensors = [
            ("a.weight", &[1.0_f32, 2.0][..], &[2_u64][..]),
            ("b.weight", &[3.0_f32, 4.0][..], &[1_u64, 2][..]),
        ];
        [
            component_fragment(crate::persist::QUANTIZATION_NATIVE, &tensors, false),
            component_fragment(QUANTIZATION_F16, &tensors, true),
        ]
    }

    fn load(path: &Path) -> anyhow::Result<VoxtralWeights<PileSnapshot>> {
        let snapshot = crate::model_collection::load_model_collection_local_latest(path)?;
        VoxtralWeights::from_snapshot(snapshot)
    }

    #[test]
    fn valid_cohort_is_frozen_and_later_coordinate_ambiguity_fails_fresh_loads() {
        let pile = TestPile::new();
        publish(pile.path(), valid_cohort());
        let frozen = load(pile.path()).expect("valid Voxtral cohort");
        assert_eq!(frozen.counts(), (2, 2));
        assert_eq!(frozen.validate_f16_parity().unwrap(), (2, 4));

        publish(
            pile.path(),
            [component_fragment(
                QUANTIZATION_F16,
                &[("other.weight", &[9.0], &[1])],
                true,
            )],
        );
        assert_eq!(frozen.counts(), (2, 2));
        let error = load(pile.path()).err().expect("ambiguous f16 coordinate");
        assert!(
            error.to_string().contains("ambiguous model root"),
            "{error:#}"
        );
    }

    #[test]
    fn missing_or_wrong_width_roots_fail_closed() {
        let missing = TestPile::new();
        publish(
            missing.path(),
            [component_fragment(
                crate::persist::QUANTIZATION_NATIVE,
                &[("weight", &[1.0], &[1])],
                false,
            )],
        );
        let error = load(missing.path()).err().expect("missing f16 root");
        assert!(error.to_string().contains("no model root"), "{error:#}");

        let wrong = TestPile::new();
        publish(
            wrong.path(),
            [
                component_fragment(
                    crate::persist::QUANTIZATION_NATIVE,
                    &[("weight", &[1.0], &[1])],
                    true,
                ),
                component_fragment(QUANTIZATION_F16, &[("weight", &[1.0], &[1])], true),
            ],
        );
        let error = load(wrong.path()).err().expect("wrong exact width");
        assert!(error.to_string().contains("exact tensor"), "{error:#}");
    }

    #[test]
    fn tensor_name_and_shape_domains_must_match_exactly() {
        let names = TestPile::new();
        publish(
            names.path(),
            [
                component_fragment(
                    crate::persist::QUANTIZATION_NATIVE,
                    &[("exact.weight", &[1.0], &[1])],
                    false,
                ),
                component_fragment(QUANTIZATION_F16, &[("derived.weight", &[1.0], &[1])], true),
            ],
        );
        let error = load(names.path()).err().expect("different tensor names");
        assert!(error.to_string().contains("missing tensor"), "{error:#}");

        let shapes = TestPile::new();
        publish(
            shapes.path(),
            [
                component_fragment(
                    crate::persist::QUANTIZATION_NATIVE,
                    &[("weight", &[1.0, 2.0], &[2])],
                    false,
                ),
                component_fragment(QUANTIZATION_F16, &[("weight", &[1.0, 2.0], &[1, 2])], true),
            ],
        );
        let error = load(shapes.path()).err().expect("different shapes");
        assert!(
            format!("{error:#}").contains("[2] exact but [1, 2] derived"),
            "{error:#}"
        );
    }

    #[cfg(feature = "import")]
    #[test]
    fn repeated_f16_derivation_is_physically_idempotent() {
        let pile = TestPile::new();
        publish(
            pile.path(),
            [component_fragment(
                crate::persist::QUANTIZATION_NATIVE,
                &[
                    ("a.weight", &[1.0_f32, -2.25][..], &[2_u64][..]),
                    ("b.weight", &[3.5_f32, 4.0][..], &[1_u64, 2][..]),
                ],
                false,
            )],
        );

        let signing_key = SigningKey::from_bytes(&[0x56; 32]);
        let mut open = Pile::open(pile.path()).expect("open exact-only Voxtral pile");
        let derive = |open: &mut Pile| {
            let snapshot = crate::model_collection::snapshot_model_collection_local_latest(open)
                .expect("freeze exact Voxtral prefix");
            let exact = crate::selection::SelectedModelIndex::from_snapshot(
                snapshot,
                ModelSelector::Source {
                    source: SOURCE,
                    quantization: crate::persist::QUANTIZATION_NATIVE,
                },
            )
            .expect("select exact Voxtral root");
            crate::persist::derive_selected_f16_to_collection(
                open,
                &signing_key,
                exact,
                SOURCE,
                QUANTIZATION_F16,
            )
            .expect("derive synthetic f16 root")
        };

        let (first_root, _, first_tensors, first_elements) = derive(&mut open);
        let len_after_first = std::fs::metadata(pile.path())
            .expect("stat first derivation")
            .len();
        let (second_root, _, second_tensors, second_elements) = derive(&mut open);
        let len_after_second = std::fs::metadata(pile.path())
            .expect("stat repeated derivation")
            .len();

        assert_eq!(first_root, second_root);
        assert_eq!((first_tensors, first_elements), (2, 4));
        assert_eq!((second_tensors, second_elements), (2, 4));
        assert_eq!(
            len_after_first, len_after_second,
            "repeating an identical derivation appended bytes"
        );

        let complete = crate::model_collection::snapshot_model_collection_local_latest(&mut open)
            .expect("freeze complete repeated cohort");
        let weights = VoxtralWeights::from_snapshot(complete).expect("select repeated cohort");
        assert_eq!(weights.roots().1, first_root);
        assert_eq!(weights.validate_f16_parity().unwrap(), (2, 4));
        drop(weights);
        open.close().expect("close repeated-derivation pile");
    }

    /// Opt-in deployment gate for a full native pile without constructing the
    /// model. The ordinary test suite skips it when no artifact is configured.
    #[test]
    fn configured_native_pile_has_a_bit_exact_complete_cohort() {
        let Ok(path) = std::env::var("MARY_VOXTRAL_NATIVE_PILE") else {
            return;
        };
        let path = Path::new(&path);
        let snapshot = crate::model_collection::load_model_collection_local_latest(path)
            .expect("load configured native Voxtral collection");
        let weights =
            VoxtralWeights::from_snapshot(snapshot).expect("select configured Voxtral cohort");
        assert_eq!(weights.counts(), (711, 711));
        let (tensors, elements) = weights
            .validate_f16_parity()
            .expect("validate configured exact/f16 bit parity");
        eprintln!("validated {tensors} Voxtral tensors / {elements} elements");
        assert_eq!(tensors, 711);
        assert!(elements > 0);
    }
}
