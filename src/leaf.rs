//! Typed tensor leaves.
//!
//! A leaf is one stored tensor of a model. It is a [`Tensor<T, RANK>`] blob:
//! the shape lives in the blob's own header, so one handle replaces two and a
//! leaf cannot be paired with the wrong shape.
//!
//! The form this replaced carried TWO handles — the data and, separately, the
//! shape — with nothing binding them. "Does this shape describe this data" was
//! then an invariant no type and no encoding held, so the only way to know was
//! to fetch both and compare, which is what every reader that cared had to do.
//! Here the comparison happens once, inside the encoding, on the way in and on
//! the way out.
//!
//! # The type IS the query
//!
//! One anchor yields a distinct attribute id per `(element, rank)`, so
//! `Handle<Tensor<F32, 2>>` and `Handle<Tensor<F16, 2>>` are different
//! attributes. A reader asking for f32 matrices cannot receive f16 ones, and a
//! rank-2 query cannot receive a rank-4 tensor.
//!
//! The cost, stated plainly: a reader cannot iterate "every leaf" in one query.
//! It queries per (element, rank). That is the guarantee working rather than a
//! limitation to route around — an untyped iteration is exactly what lets a q4
//! leaf be read as f16 — but it does change the shape of a keymap builder.
//!
//! # Old piles
//!
//! Every model pile written before this holds the two-blob form, and they are
//! large enough that making them unreadable would be a far worse outcome than
//! the duplication. So [`resolve`] reads BOTH: the typed leaf first, the
//! `{data|data_f16, shape}` pair second. The runtime shape check that used to
//! be scattered across readers now lives in exactly one place — [`legacy`],
//! the adapter — and it goes away with the last unmigrated pile rather than
//! having to be remembered.
//!
//! Writing is typed only. `pile_leaf_migrate` converts a pile in place-ish
//! (source read-only, into a new file), preserving every leaf's entity id, so
//! nothing downstream has to be told a new address.

use anyhow::{Context, Result};
use triblespace::core::attribute::Attribute;
use triblespace::core::blob::encodings::tensor::elements::{BF16, F16, F32, NVFP4};
use triblespace::core::blob::encodings::tensor::{Tensor, TensorElement, TensorView, tensor_blob};
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::core::id_hex;
use triblespace::core::inline::encodings::hash::Handle;
use triblespace::prelude::*;

/// Anchor every leaf attribute derives from. Minted 2026-08-10.
///
/// One anchor, not one per dtype: `Attribute::anchored` derives the id from
/// (anchor, value encoding), and the encoding already carries the element type
/// and the rank. Fourteen combinations cost one minted id.
pub const LEAF_ANCHOR: Id = id_hex!("743E98D23794CA9BEFE727D07482D8D5");

/// The leaf attribute for one element format and rank.
pub fn leaf<T: TensorElement, const RANK: usize>() -> Attribute<Handle<Tensor<T, RANK>>> {
    Attribute::anchored(LEAF_ANCHOR)
}

/// Which element format a leaf holds.
///
/// The type parameter is gone by the time a leaf is in a by-name index — a
/// keymap spans every dtype in the model — so the fact travels as data instead.
/// This is erasure done ONCE, at the index boundary, from a read that was
/// typed: the leaf was fetched as `Tensor<F32, 2>` or not at all. It is not the
/// untyped iteration the module docs warn about, where a q4 leaf can be read as
/// f16, because no read here is performed without its type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F32,
    F16,
    /// Native bfloat16, distinct from IEEE f16 despite sharing its width.
    Bf16,
    /// Packed NVFP4: E2M1 codes with E4M3 block scales and one global scale in
    /// the payload (`crate::nvfp4`). Reading it as f32 decodes on the host.
    Nvfp4,
}

impl Elem {
    /// Payload bytes `elems` logical elements of this format occupy.
    fn payload_len(self, elems: usize) -> usize {
        match self {
            Elem::F32 => <F32 as TensorElement>::payload_len(elems),
            Elem::F16 => <F16 as TensorElement>::payload_len(elems),
            Elem::Bf16 => <BF16 as TensorElement>::payload_len(elems),
            Elem::Nvfp4 => <NVFP4 as TensorElement>::payload_len(elems),
        }
    }
}

/// One resolved leaf: what it holds, its logical shape, and its bytes.
///
/// The payload is a view over the pile's mapping, not a copy. Building an index
/// of a whole model therefore costs handles and headers, not weights.
#[derive(Clone)]
pub struct Leaf {
    elem: Elem,
    dims: Vec<u64>,
    payload: anybytes::Bytes,
}

impl Leaf {
    /// Which element format.
    pub fn elem(&self) -> Elem {
        self.elem
    }

    /// Logical dims, as stored.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Shape as the `Vec<usize>` the loaders speak.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().map(|&d| d as usize).collect()
    }

    /// Logical element count.
    pub fn elems(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// The payload bytes. Aligned to 256 by construction — a typed leaf's
    /// payload starts one 256-byte header into a 256-aligned pile record, and a
    /// legacy data blob starts at the record itself.
    pub fn payload(&self) -> &anybytes::Bytes {
        &self.payload
    }

    /// ZERO-COPY view of an f32 leaf. `None` for f16 — the caller wants
    /// [`Self::to_f32`] there, which must convert and therefore must allocate.
    pub fn view_f32(&self) -> Option<anybytes::View<[f32]>> {
        match self.elem {
            Elem::F32 => self.payload.clone().view::<[f32]>().ok(),
            Elem::F16 | Elem::Bf16 | Elem::Nvfp4 => None,
        }
    }

    /// ZERO-COPY view of an f16 leaf, for the lanes that upload at native
    /// width. `None` for f32.
    pub fn view_f16(&self) -> Option<anybytes::View<[half::f16]>> {
        match self.elem {
            Elem::F16 => self.payload.clone().view::<[half::f16]>().ok(),
            Elem::F32 | Elem::Bf16 | Elem::Nvfp4 => None,
        }
    }

    /// Borrow the native BF16 payload without conversion or allocation.
    /// Returns `None` for every other element format, including IEEE f16.
    pub fn view_bf16(&self) -> Option<anybytes::View<[half::bf16]>> {
        match self.elem {
            Elem::Bf16 => self.payload.clone().view::<[half::bf16]>().ok(),
            Elem::F32 | Elem::F16 | Elem::Nvfp4 => None,
        }
    }

    /// Materialise as f32. Allocates — call [`Self::view_f32`] first and only
    /// fall back to this when it returns `None`.
    pub fn to_f32(&self) -> Vec<f32> {
        match self.elem {
            Elem::F32 => self
                .payload
                .clone()
                .view::<[f32]>()
                .expect("f32 payload view")[..]
                .to_vec(),
            Elem::F16 => self
                .payload
                .clone()
                .view::<[half::f16]>()
                .expect("f16 payload view")
                .iter()
                .map(|h| h.to_f32())
                .collect(),
            Elem::Bf16 => self
                .view_bf16()
                .expect("bf16 payload view")
                .iter()
                .map(|h| h.to_f32())
                .collect(),
            Elem::Nvfp4 => crate::nvfp4::decode_payload(&self.payload, self.elems())
                .expect("an NVFP4 leaf payload is sized by its encoding"),
        }
    }

    /// The `(f32 data, shape)` pair the f32-centric model loaders speak.
    pub fn to_f32_shape(&self) -> (Vec<f32>, Vec<usize>) {
        (self.to_f32(), self.shape())
    }

    /// A leaf from a decoded typed tensor. Infallible: the encoding already
    /// checked that the payload is the length the dims imply.
    fn from_view(elem: Elem, view: TensorView) -> Self {
        Self {
            elem,
            dims: view.dims().to_vec(),
            payload: view.payload().clone(),
        }
    }
}

impl std::fmt::Debug for Leaf {
    /// What it is and how big — never the weights.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Leaf")
            .field("elem", &self.elem)
            .field("dims", &self.dims)
            .field("payload_len", &self.payload.len())
            .finish()
    }
}

/// A leaf from the two-blob form, with the check the encoding would have made.
///
/// This is the ONLY place mary compares a shape against a payload, and it
/// exists solely because pre-migration piles state the two separately. A typed
/// leaf reaches [`Leaf`] without passing through here.
pub fn legacy(elem: Elem, dims: Vec<u64>, payload: anybytes::Bytes) -> Result<Leaf> {
    let elems: u64 = dims.iter().product();
    let expected = elem.payload_len(elems as usize);
    anyhow::ensure!(
        payload.len() == expected,
        "legacy {elem:?} leaf: payload is {} bytes, shape {dims:?} implies {expected}",
        payload.len()
    );
    Ok(Leaf {
        elem,
        dims,
        payload,
    })
}

/// Store a tensor as a typed leaf blob.
///
/// Fallible because the payload is checked against what the dims and element
/// format imply — here, once, rather than as a misread tensor later that
/// produces plausible numbers instead of an error.
pub fn leaf_blob<T: TensorElement, const RANK: usize>(
    dims: [u64; RANK],
    payload: anybytes::Bytes,
) -> Result<Blob<Tensor<T, RANK>>> {
    tensor_blob::<T, RANK>(dims, payload).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Read a leaf WITHOUT materialising it.
pub fn read_leaf<T: TensorElement, const RANK: usize>(
    blob: Blob<Tensor<T, RANK>>,
) -> Result<TensorView> {
    TensorView::try_from_blob(blob).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Build the typed blob, verify it against the source bytes, put it, and attach
/// the handle to `$head` (`_` for a content-derived id, an `ExclusiveId` ref to
/// keep an existing one).
///
/// Verified where both halves are in hand: decode what was just built and
/// compare. `tensor_blob` copies the payload anyway, so the comparison is one
/// more pass over bytes that are already hot — cheap next to writing them.
macro_rules! leaf_entity {
    ($blobs:expr_2021, $head:tt, $elem:ty, $rank:literal, $dims:expr_2021, $payload:expr_2021, $what:expr_2021) => {{
        let dims: [u64; $rank] = $dims
            .as_ref()
            .try_into()
            .expect("rank checked by the dispatch");
        let src: anybytes::Bytes = $payload;
        let blob = leaf_blob::<$elem, $rank>(dims, src.clone())
            .with_context(|| format!("{}: build typed leaf", $what))?;
        let view = read_leaf::<$elem, $rank>(blob.clone())
            .with_context(|| format!("{}: read back typed leaf", $what))?;
        anyhow::ensure!(
            view.dims() == dims && &view.payload()[..] == &src[..],
            "{}: typed leaf did not round-trip its own bytes",
            $what
        );
        let handle = $blobs
            .put::<Tensor<$elem, $rank>, _>(blob)
            .map_err(|e| anyhow::anyhow!("{}: store typed leaf: {e}", $what))?;
        triblespace::macros::entity! { $head @ leaf::<$elem, $rank>(): handle }
    }};
}

/// Dispatch over the ranks models actually use.
///
/// Both ends of the range are real cases found in the piles, not defensive
/// padding: `clip`'s `logit_scale` is a rank-0 scalar, and `nomic_mm7b` holds a
/// rank-5 tensor. Both were found by this dispatch REFUSING them — which is the
/// argument for refusing rather than flattening. A converter that reshaped the
/// rank-5 tensor to fit would have reported success and written a model whose
/// weights are silently misframed.
///
/// Beyond rank 6 it still refuses. The encoding allows up to 32; the arms stop
/// where the evidence stops.
macro_rules! leaf_by_rank {
    ($blobs:expr_2021, $head:tt, $elem:expr_2021, $dims:expr_2021, $payload:expr_2021, $what:expr_2021) => {{
        let dims: &[u64] = $dims;
        match ($elem, dims.len()) {
            (Elem::F32, 0) => leaf_entity!($blobs, $head, F32, 0, dims, $payload, $what),
            (Elem::F32, 1) => leaf_entity!($blobs, $head, F32, 1, dims, $payload, $what),
            (Elem::F32, 2) => leaf_entity!($blobs, $head, F32, 2, dims, $payload, $what),
            (Elem::F32, 3) => leaf_entity!($blobs, $head, F32, 3, dims, $payload, $what),
            (Elem::F32, 4) => leaf_entity!($blobs, $head, F32, 4, dims, $payload, $what),
            (Elem::F32, 5) => leaf_entity!($blobs, $head, F32, 5, dims, $payload, $what),
            (Elem::F32, 6) => leaf_entity!($blobs, $head, F32, 6, dims, $payload, $what),
            (Elem::F16, 0) => leaf_entity!($blobs, $head, F16, 0, dims, $payload, $what),
            (Elem::F16, 1) => leaf_entity!($blobs, $head, F16, 1, dims, $payload, $what),
            (Elem::F16, 2) => leaf_entity!($blobs, $head, F16, 2, dims, $payload, $what),
            (Elem::F16, 3) => leaf_entity!($blobs, $head, F16, 3, dims, $payload, $what),
            (Elem::F16, 4) => leaf_entity!($blobs, $head, F16, 4, dims, $payload, $what),
            (Elem::F16, 5) => leaf_entity!($blobs, $head, F16, 5, dims, $payload, $what),
            (Elem::F16, 6) => leaf_entity!($blobs, $head, F16, 6, dims, $payload, $what),
            (Elem::Bf16, 0) => leaf_entity!($blobs, $head, BF16, 0, dims, $payload, $what),
            (Elem::Bf16, 1) => leaf_entity!($blobs, $head, BF16, 1, dims, $payload, $what),
            (Elem::Bf16, 2) => leaf_entity!($blobs, $head, BF16, 2, dims, $payload, $what),
            (Elem::Bf16, 3) => leaf_entity!($blobs, $head, BF16, 3, dims, $payload, $what),
            (Elem::Bf16, 4) => leaf_entity!($blobs, $head, BF16, 4, dims, $payload, $what),
            (Elem::Bf16, 5) => leaf_entity!($blobs, $head, BF16, 5, dims, $payload, $what),
            (Elem::Bf16, 6) => leaf_entity!($blobs, $head, BF16, 6, dims, $payload, $what),
            (Elem::Nvfp4, 1) => leaf_entity!($blobs, $head, NVFP4, 1, dims, $payload, $what),
            (Elem::Nvfp4, 2) => leaf_entity!($blobs, $head, NVFP4, 2, dims, $payload, $what),
            (Elem::Nvfp4, 3) => leaf_entity!($blobs, $head, NVFP4, 3, dims, $payload, $what),
            (_, r) => anyhow::bail!(
                "{}: rank {r} exceeds the ranks this format dispatches (0..=6); \
                 add an arm rather than flattening",
                $what
            ),
        }
    }};
}

/// Store one tensor as a typed leaf under a CONTENT-DERIVED entity id.
///
/// The write path for every importer. Identical tensors collapse to one entity,
/// as before — the id now derives from one handle instead of two, so a leaf and
/// its shape can no longer disagree about which tensor they are.
pub fn put_leaf(
    blobs: &mut impl BlobStorePut,
    elem: Elem,
    dims: &[u64],
    payload: anybytes::Bytes,
    what: &str,
) -> Result<Fragment> {
    Ok(leaf_by_rank!(blobs, _, elem, dims, payload, what))
}

/// Store one tensor as a typed leaf under an EXISTING entity id.
///
/// What a pile-to-pile conversion needs: the leaf keeps its own address, so
/// module edges, model roots and `member` lists still resolve.
pub fn put_leaf_as(
    blobs: &mut impl BlobStorePut,
    id: &ExclusiveId,
    elem: Elem,
    dims: &[u64],
    payload: anybytes::Bytes,
    what: &str,
) -> Result<Fragment> {
    Ok(leaf_by_rank!(blobs, id, elem, dims, payload, what))
}

/// Read the leaf hanging off one weight entity: typed first, legacy second.
///
/// Every `(element, rank)` attribute is asked, not just enough of them to find
/// an answer, because "a leaf carries ONE payload" is a fact about the graph
/// that the encoding has nothing to say about — an entity with both an f32 and
/// an f16 leaf is malformed however well each of them decodes. Asking is
/// sixteen point probes; only the one that answers is fetched.
pub fn resolve(tribles: &TribleSet, blobs: &impl BlobStoreGet, weight: Id) -> Result<Option<Leaf>> {
    let mut hits = 0usize;
    let mut found: Option<Leaf> = None;

    macro_rules! typed {
        ($elem:ty, $rank:literal, $tag:expr_2021) => {{
            if let Some((h,)) = triblespace::macros::find!(
                (h: Inline<Handle<Tensor<$elem, $rank>>>),
                triblespace::macros::pattern!(tribles, [{ weight @ leaf::<$elem, $rank>(): ?h }])
            )
            .next()
            {
                hits += 1;
                if found.is_none() {
                    let blob: Blob<Tensor<$elem, $rank>> = blobs
                        .get(h)
                        .map_err(|e| anyhow::anyhow!("read typed leaf {weight}: {e}"))?;
                    let view = read_leaf::<$elem, $rank>(blob)
                        .with_context(|| format!("decode typed leaf {weight}"))?;
                    found = Some(Leaf::from_view($tag, view));
                }
            }
        }};
    }
    typed!(F32, 0, Elem::F32);
    typed!(F32, 1, Elem::F32);
    typed!(F32, 2, Elem::F32);
    typed!(F32, 3, Elem::F32);
    typed!(F32, 4, Elem::F32);
    typed!(F32, 5, Elem::F32);
    typed!(F32, 6, Elem::F32);
    typed!(F16, 0, Elem::F16);
    typed!(F16, 1, Elem::F16);
    typed!(F16, 2, Elem::F16);
    typed!(F16, 3, Elem::F16);
    typed!(F16, 4, Elem::F16);
    typed!(F16, 5, Elem::F16);
    typed!(F16, 6, Elem::F16);
    typed!(BF16, 0, Elem::Bf16);
    typed!(BF16, 1, Elem::Bf16);
    typed!(BF16, 2, Elem::Bf16);
    typed!(BF16, 3, Elem::Bf16);
    typed!(BF16, 4, Elem::Bf16);
    typed!(BF16, 5, Elem::Bf16);
    typed!(BF16, 6, Elem::Bf16);
    typed!(NVFP4, 1, Elem::Nvfp4);
    typed!(NVFP4, 2, Elem::Nvfp4);
    typed!(NVFP4, 3, Elem::Nvfp4);

    if hits > 0 {
        anyhow::ensure!(
            hits == 1,
            "tensor leaf {weight} carries {hits} typed leaves; a leaf has one payload"
        );
        return Ok(found);
    }
    resolve_legacy(tribles, blobs, weight)
}

/// The two-blob form, for piles written before the typed encoding.
///
/// Carries the same one-payload rule and the shape check the encoding would
/// have made. This is the ONLY place mary compares a shape against a payload.
fn resolve_legacy(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    weight: Id,
) -> Result<Option<Leaf>> {
    use crate::format::attrs;
    let f32_data: Vec<_> = triblespace::macros::find!(
        (d: Inline<inlineencodings::Handle<crate::format::F32Array>>),
        triblespace::macros::pattern!(tribles, [{ weight @ attrs::data: ?d }])
    )
    .map(|(d,)| d)
    .collect();
    let f16_data: Vec<_> = triblespace::macros::find!(
        (d: Inline<inlineencodings::Handle<crate::f16enc::F16Array>>),
        triblespace::macros::pattern!(tribles, [{ weight @ attrs::data_f16: ?d }])
    )
    .map(|(d,)| d)
    .collect();
    // Neither: not a float tensor leaf at all. The quantized leaves
    // (`data_q4`/`data_q8` + `q_scales` + `shape`) land here and are read by
    // their own module, not this one.
    if f32_data.is_empty() && f16_data.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        f32_data.len() + f16_data.len() == 1,
        "tensor leaf {weight} must have exactly one of data/data_f16 (found {} f32, {} f16)",
        f32_data.len(),
        f16_data.len()
    );

    let shape_handle = {
        let mut shapes = triblespace::macros::find!(
            (s: Inline<inlineencodings::Handle<crate::format::U64Array>>),
            triblespace::macros::pattern!(tribles, [{ weight @ attrs::shape: ?s }])
        );
        let first = shapes
            .next()
            .ok_or_else(|| anyhow::anyhow!("two-blob leaf {weight} has no shape"))?
            .0;
        anyhow::ensure!(
            shapes.next().is_none(),
            "ambiguous shape on tensor leaf {weight}"
        );
        first
    };
    let shape: anybytes::Bytes = blobs
        .get(shape_handle)
        .map_err(|e| anyhow::anyhow!("read shape blob for leaf {weight}: {e}"))?;
    let dims: Vec<u64> = shape
        .view::<[u64]>()
        .with_context(|| format!("decode shape blob for leaf {weight}"))?
        .to_vec();

    let (elem, payload) = if let Some(d) = f32_data.first() {
        let payload: anybytes::Bytes = blobs
            .get(*d)
            .map_err(|e| anyhow::anyhow!("read f32 data blob for leaf {weight}: {e}"))?;
        (Elem::F32, payload)
    } else {
        let payload: anybytes::Bytes = blobs
            .get(f16_data[0])
            .map_err(|e| anyhow::anyhow!("read f16 data blob for leaf {weight}: {e}"))?;
        (Elem::F16, payload)
    };
    legacy(elem, dims, payload)
        .with_context(|| format!("leaf {weight}"))
        .map(Some)
}

/// Every `(module, tensor name, weight entity)` in a graph.
///
/// Joins module to leaf directly rather than walking `member` edges from a
/// model root, so a pile holding several models (or one whose root layout
/// differs) indexes the same way.
fn named_weights(
    tribles: &TribleSet,
) -> Vec<(
    Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
    Id,
)> {
    use crate::format::attrs;
    triblespace::macros::find!(
        (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>, w: Id),
        triblespace::macros::pattern!(tribles, [
            { _?m @ attrs::safetensor_path: ?n, attrs::weight: ?w },
        ])
    )
    .collect()
}

/// Index every leaf in a pile by its `safetensor_path` name.
///
/// Costs headers and handles, not weights: the payloads alias the pile's
/// mapping.
pub fn index_by_name(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
) -> Result<std::collections::HashMap<String, Leaf>> {
    let mut map = std::collections::HashMap::new();
    for (name_handle, weight) in named_weights(tribles) {
        let Some(resolved) = resolve(tribles, blobs, weight)? else {
            continue;
        };
        let name = crate::ingest::read_string(blobs, name_handle);
        map.insert(name, resolved);
    }
    Ok(map)
}

/// Index EVERY TYPED leaf in a pile by its entity id, regardless of which model
/// root (if any) reaches it.
///
/// [`index_by_name`] walks module edges and so only sees leaves a module names.
/// This sees all of them, which is what a verifier wants: it must not take the
/// graph's word for what exists. Typed only, deliberately — its one caller is
/// checking that a conversion produced typed leaves.
pub fn index_typed_all(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
) -> std::collections::HashMap<Id, Leaf> {
    let mut map = std::collections::HashMap::new();

    macro_rules! sweep_all {
        ($elem:ty, $rank:literal, $tag:expr_2021) => {{
            for (e, h) in triblespace::macros::find!(
                (e: Id, h: Inline<Handle<Tensor<$elem, $rank>>>),
                triblespace::macros::pattern!(tribles, [
                    { ?e @ leaf::<$elem, $rank>(): ?h },
                ])
            ) {
                let blob: Blob<Tensor<$elem, $rank>> = blobs.get(h).expect("typed leaf blob");
                let view = TensorView::try_from_blob(blob).expect("typed leaf decodes");
                map.insert(e, Leaf::from_view($tag, view));
            }
        }};
    }

    sweep_all!(F32, 0, Elem::F32);
    sweep_all!(F32, 1, Elem::F32);
    sweep_all!(F32, 2, Elem::F32);
    sweep_all!(F32, 3, Elem::F32);
    sweep_all!(F32, 4, Elem::F32);
    sweep_all!(F32, 5, Elem::F32);
    sweep_all!(F32, 6, Elem::F32);
    sweep_all!(F16, 0, Elem::F16);
    sweep_all!(F16, 1, Elem::F16);
    sweep_all!(F16, 2, Elem::F16);
    sweep_all!(F16, 3, Elem::F16);
    sweep_all!(F16, 4, Elem::F16);
    sweep_all!(F16, 5, Elem::F16);
    sweep_all!(F16, 6, Elem::F16);
    sweep_all!(BF16, 0, Elem::Bf16);
    sweep_all!(BF16, 1, Elem::Bf16);
    sweep_all!(BF16, 2, Elem::Bf16);
    sweep_all!(BF16, 3, Elem::Bf16);
    sweep_all!(BF16, 4, Elem::Bf16);
    sweep_all!(BF16, 5, Elem::Bf16);
    sweep_all!(BF16, 6, Elem::Bf16);
    sweep_all!(NVFP4, 1, Elem::Nvfp4);
    sweep_all!(NVFP4, 2, Elem::Nvfp4);
    sweep_all!(NVFP4, 3, Elem::Nvfp4);

    map
}

/// Every typed leaf attribute this build dispatches, as `(label, id)`.
///
/// The ids are DERIVED from (anchor, encoding), so they cannot be read off the
/// source — a census tool that wants to name them has to ask the same code the
/// readers ask, rather than restating a table that would drift the first time
/// an arm is added.
pub fn typed_leaf_attrs() -> Vec<(String, Id)> {
    let mut out = Vec::new();

    macro_rules! name_all {
        ($elem:ty, $rank:literal, $tag:expr_2021) => {{
            out.push((
                format!("leaf.{}.{}", $tag, $rank),
                leaf::<$elem, $rank>().id(),
            ));
        }};
    }

    name_all!(F32, 0, "f32");
    name_all!(F32, 1, "f32");
    name_all!(F32, 2, "f32");
    name_all!(F32, 3, "f32");
    name_all!(F32, 4, "f32");
    name_all!(F32, 5, "f32");
    name_all!(F32, 6, "f32");
    name_all!(F16, 0, "f16");
    name_all!(F16, 1, "f16");
    name_all!(F16, 2, "f16");
    name_all!(F16, 3, "f16");
    name_all!(F16, 4, "f16");
    name_all!(F16, 5, "f16");
    name_all!(F16, 6, "f16");
    name_all!(BF16, 0, "bf16");
    name_all!(BF16, 1, "bf16");
    name_all!(BF16, 2, "bf16");
    name_all!(BF16, 3, "bf16");
    name_all!(BF16, 4, "bf16");
    name_all!(BF16, 5, "bf16");
    name_all!(BF16, 6, "bf16");

    out
}

/// Which storage form a fixture builds its leaves in.
///
/// Test-only, and it exists because the two are not interchangeable in
/// practice: every importer WRITES [`Form::Typed`], while every model pile on
/// disk still HOLDS [`Form::TwoBlob`]. A fixture pinned to either one tests a
/// path half the world is not on. Running a model's selection tests over both
/// is what makes "this loader survives the conversion" a checked claim rather
/// than a plan.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    /// One `Tensor<T, RANK>` blob, shape in its header.
    Typed,
    /// `{data|data_f16, shape}`: two blobs, and the pairing is a convention.
    TwoBlob,
}

/// Both forms, for a test that should hold under either.
#[cfg(test)]
pub(crate) const FORMS: [Form; 2] = [Form::Typed, Form::TwoBlob];

#[cfg(test)]
impl Form {
    /// Short label for a test's failure message.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Form::Typed => "typed",
            Form::TwoBlob => "two-blob",
        }
    }
}

/// Put one tensor leaf into `fragment` in `form`; returns its entity id.
///
/// `values` are f32 whatever the element format; an [`Elem::F16`] leaf is
/// down-cast here, exactly as [`crate::format::put_raw_f16`] does on the real
/// import path.
#[cfg(test)]
pub(crate) fn fixture_leaf(
    fragment: &mut Fragment,
    form: Form,
    elem: Elem,
    dims: &[u64],
    values: &[f32],
) -> Id {
    use crate::format::attrs;
    let payload = || -> anybytes::Bytes {
        match elem {
            Elem::F32 => anybytes::Bytes::from_source(values.to_vec()),
            Elem::F16 => anybytes::Bytes::from_source(
                values
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect::<Vec<half::f16>>(),
            ),
            Elem::Bf16 => anybytes::Bytes::from_source(
                values
                    .iter()
                    .map(|&v| half::bf16::from_f32(v))
                    .collect::<Vec<half::bf16>>(),
            ),
            Elem::Nvfp4 => panic!("fixture leaves are dense"),
        }
    };

    // A fragment is not a blob STORE — it carries its own blobs — so the typed
    // arm builds the blob and hands it to `Fragment::put` rather than going
    // through `put_leaf`. Same encoding, same header, same bytes; only the
    // sink differs.
    macro_rules! typed {
        ($elem:ty, $rank:literal) => {{
            let d: [u64; $rank] = dims.try_into().expect("fixture rank matches its dims");
            let blob = leaf_blob::<$elem, $rank>(d, payload()).expect("fixture typed leaf");
            let h = fragment.put::<Tensor<$elem, $rank>, _>(blob);
            triblespace::macros::entity! { _ @ leaf::<$elem, $rank>(): h }
        }};
    }

    let leaf = match form {
        Form::Typed => match (elem, dims.len()) {
            (Elem::F32, 0) => typed!(F32, 0),
            (Elem::F32, 1) => typed!(F32, 1),
            (Elem::F32, 2) => typed!(F32, 2),
            (Elem::F32, 3) => typed!(F32, 3),
            (Elem::F16, 0) => typed!(F16, 0),
            (Elem::F16, 1) => typed!(F16, 1),
            (Elem::F16, 2) => typed!(F16, 2),
            (Elem::F16, 3) => typed!(F16, 3),
            (Elem::Bf16, 0) => typed!(BF16, 0),
            (Elem::Bf16, 1) => typed!(BF16, 1),
            (Elem::Bf16, 2) => typed!(BF16, 2),
            (Elem::Bf16, 3) => typed!(BF16, 3),
            (_, r) => panic!("fixture leaf rank {r} has no arm; add one"),
        },
        Form::TwoBlob => {
            let shape = fragment.put::<crate::format::U64Array, _>(dims.to_vec());
            match elem {
                Elem::F32 => {
                    let data = fragment.put::<crate::format::F32Array, _>(values.to_vec());
                    triblespace::macros::entity! { _ @ attrs::data: data, attrs::shape: shape }
                }
                Elem::F16 => {
                    let data = fragment.put::<crate::f16enc::F16Array, _>(
                        values
                            .iter()
                            .map(|&v| half::f16::from_f32(v))
                            .collect::<Vec<half::f16>>(),
                    );
                    triblespace::macros::entity! { _ @ attrs::data_f16: data, attrs::shape: shape }
                }
                Elem::Bf16 | Elem::Nvfp4 => panic!("the two-blob form has no bf16 or packed leaves"),
            }
        }
    };
    let id = leaf.root().expect("fixture leaf root");
    *fragment += leaf;
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::blob::MemoryBlobStore;

    fn bytes(n: usize) -> anybytes::Bytes {
        anybytes::Bytes::from_source(vec![0u8; n])
    }

    /// One handle, not two, and the shape travels with the data so they cannot
    /// be paired wrongly.
    #[test]
    fn a_leaf_carries_its_own_shape() {
        let blob = leaf_blob::<F32, 2>([8, 16], bytes(8 * 16 * 4)).expect("well formed");
        let view = read_leaf::<F32, 2>(blob).expect("reads");
        assert_eq!(view.dims(), &[8, 16]);
        assert_eq!(view.payload().len(), 512);
    }

    /// THE property. Element type and rank are both in the attribute id, so a
    /// reader cannot be handed the wrong kind of leaf.
    #[test]
    fn element_and_rank_are_both_in_the_attribute_id() {
        let f32_2 = leaf::<F32, 2>().id();
        let f16_2 = leaf::<F16, 2>().id();
        let f32_4 = leaf::<F32, 4>().id();
        assert_ne!(f32_2, f16_2, "an f16 leaf is not an f32 leaf");
        assert_ne!(f32_2, f32_4, "a rank-4 leaf is not a rank-2 leaf");
        assert_eq!(f32_2, leaf::<F32, 2>().id(), "and the id is stable");
    }

    /// A payload that contradicts the dims is refused where refusing is cheap.
    /// The same bytes read as the wrong shape produce numbers, not an error.
    #[test]
    fn a_payload_that_does_not_match_the_dims_is_refused() {
        assert!(leaf_blob::<F32, 2>([8, 16], bytes(500)).is_err());
        // and an f16 payload offered as f32 is caught by its length alone
        assert!(leaf_blob::<F32, 2>([8, 16], bytes(8 * 16 * 2)).is_err());
        assert!(leaf_blob::<F16, 2>([8, 16], bytes(8 * 16 * 2)).is_ok());
    }

    /// The write path refuses a wrong shape the same way, before anything is
    /// stored — so a pile cannot come to hold a leaf whose shape is a lie.
    #[test]
    fn the_write_path_refuses_a_shape_that_does_not_describe_the_bytes() {
        let mut blobs = MemoryBlobStore::new();
        let ok = put_leaf(&mut blobs, Elem::F32, &[3, 4], bytes(48), "w").expect("well formed");
        assert!(ok.root().is_some());
        let err = put_leaf(&mut blobs, Elem::F32, &[3, 4], bytes(40), "w")
            .expect_err("must refuse")
            .to_string();
        assert!(err.contains("build typed leaf"), "{err}");
    }

    /// A rank the dispatch has no arm for is refused rather than flattened.
    #[test]
    fn an_unsupported_rank_is_refused_not_reshaped() {
        let mut blobs = MemoryBlobStore::new();
        let err = put_leaf(&mut blobs, Elem::F32, &[1, 1, 1, 1, 1, 1, 1], bytes(4), "w")
            .expect_err("must refuse")
            .to_string();
        assert!(err.contains("rank 7"), "{err}");
    }

    /// A written leaf reads back through the same seam every loader uses, with
    /// its element format and shape intact.
    #[test]
    fn a_written_leaf_resolves_back_to_itself() {
        let mut blobs = MemoryBlobStore::new();
        let payload = anybytes::Bytes::from_source(
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<u8>>(),
        );
        let fragment = put_leaf(&mut blobs, Elem::F32, &[2, 3], payload, "w").expect("stored");
        let id = fragment.root().expect("leaf root");
        let facts = fragment.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).expect("reader");
        let leaf = resolve(&facts, &reader, id)
            .expect("resolves")
            .expect("present");
        assert_eq!(leaf.elem(), Elem::F32);
        assert_eq!(leaf.shape(), vec![2, 3]);
        assert_eq!(leaf.to_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(
            leaf.view_f32().is_some(),
            "an f32 leaf views without a copy"
        );
    }

    /// "A leaf carries one payload" is a fact about the graph, not about any
    /// encoding, so it is still checked — for both forms. An entity wearing two
    /// leaves decodes perfectly well and is still malformed.
    #[test]
    fn a_leaf_with_two_payloads_is_refused_in_either_form() {
        use crate::format::{F32Array, U64Array, attrs};
        let mut blobs = MemoryBlobStore::new();

        // Typed: one entity, an f32 leaf AND an f16 leaf.
        let f32_leaf = put_leaf(&mut blobs, Elem::F32, &[2], bytes(8), "w").expect("stored");
        let id = f32_leaf.root().expect("root");
        let mut facts = f32_leaf.into_facts();
        facts += put_leaf_as(
            &mut blobs,
            &ExclusiveId::force_ref(&id),
            Elem::F16,
            &[2],
            bytes(4),
            "w",
        )
        .expect("stored")
        .into_facts();

        // Two-blob: one entity, `data` AND `data_f16`.
        let data = blobs.put::<F32Array, _>(vec![1.0f32, 2.0]).expect("data");
        let half = blobs
            .put::<crate::f16enc::F16Array, _>(vec![half::f16::ONE, half::f16::ONE])
            .expect("half");
        let shape = blobs.put::<U64Array, _>(vec![2u64]).expect("shape");
        let both = triblespace::macros::entity! { _ @
            attrs::data: data,
            attrs::data_f16: half,
            attrs::shape: shape,
        };
        let both_id = both.root().expect("root");
        facts += both.into_facts();

        let reader = SnapshotSource::snapshot(&mut blobs).expect("reader");
        let typed_error = format!(
            "{:#}",
            resolve(&facts, &reader, id).expect_err("two typed leaves must fail")
        );
        assert!(typed_error.contains("2 typed leaves"), "{typed_error}");
        let legacy_error = format!(
            "{:#}",
            resolve(&facts, &reader, both_id).expect_err("data and data_f16 must fail")
        );
        assert!(
            legacy_error.contains("exactly one of data/data_f16"),
            "{legacy_error}"
        );
    }

    /// Old piles still read. The two-blob form resolves through the same seam,
    /// and the shape check that used to live in every reader lives here.
    #[test]
    fn the_two_blob_form_still_resolves_and_is_still_checked() {
        use crate::format::{F32Array, U64Array, attrs};
        let mut blobs = MemoryBlobStore::new();
        let data = blobs
            .put::<F32Array, _>(vec![1.0f32, 2.0, 3.0, 4.0])
            .expect("data");
        let shape = blobs.put::<U64Array, _>(vec![2u64, 2]).expect("shape");
        let good = triblespace::macros::entity! { _ @ attrs::data: data, attrs::shape: shape };
        let good_id = good.root().expect("root");

        let lying_shape = blobs.put::<U64Array, _>(vec![2u64, 3]).expect("shape");
        let bad = triblespace::macros::entity! { _ @ attrs::data: data, attrs::shape: lying_shape };
        let bad_id = bad.root().expect("root");

        let mut facts = good.into_facts();
        facts += bad.into_facts();
        let reader = SnapshotSource::snapshot(&mut blobs).expect("reader");

        let leaf = resolve(&facts, &reader, good_id)
            .expect("resolves")
            .expect("present");
        assert_eq!(leaf.shape(), vec![2, 2]);
        assert_eq!(leaf.to_f32(), vec![1.0, 2.0, 3.0, 4.0]);

        let error = format!(
            "{:#}",
            resolve(&facts, &reader, bad_id)
                .expect_err("a shape that does not describe the data must fail")
        );
        assert!(error.contains("implies"), "{error}");
    }
}
