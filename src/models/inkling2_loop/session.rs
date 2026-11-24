//! Inkling 2 Mini as a model you can hold: the weights bound once from a model
//! root in a pile ([`MiniModel`]), and any number of independent token
//! sequences over them ([`MiniSession`]), each appending token IDs and getting
//! logits back.
//!
//! This composes the operators the comparison binaries hold to upstream, in
//! upstream's order (transformers 046bdb8c, `Inkling2LoopTextModel.forward` and
//! `Inkling2LoopForCausalLM.forward`):
//!
//! ```text
//! x      = rms_norm(embed[ids], embed_norm)          the normed embedding, [t, H]
//! blocks = []
//! for l in 0..core_start:   x = layer_l(x, blocks)    layer 0 appends its input: blocks = [x0]
//! x, blocks = core(x, blocks)                         three passes, each from the pre-loop state
//! for l in core_end..L:     x = layer_l(x, blocks)    layer 18 appends its input
//! logits = head(blocks, x)                            final residual, norm, unembedding, softcap
//! ```
//!
//! What a session carries between calls is every layer visit's state, keyed by
//! [`CacheId`] (physical layer, pass): 6 + 36 + 6 = 48 states, each with its
//! own keys, values and convolution tails, and the position. Block states are
//! per token and live only inside one call: a token's blocks are built from
//! that token's own hidden states, so there is nothing of them to carry.
//!
//! Prefill and decode are the same call, [`MiniSession::append`]. A session
//! starts at position 0, so every full-attention layer's `origin` (the
//! absolute position of its first cached row, read only by the log scale) is 0.
//!
//! Failure. `append` checks its input before it touches any state: an empty
//! call, an ID outside the embedding table or an unsupported row request is
//! refused and the session is unchanged. A failure during the forward itself
//! (a device error surfaces as a panic in the backend) may leave some visits
//! updated and others not; the session is then *poisoned* and refuses every
//! call until [`MiniSession::reset`]. A non-finite router logit is such a
//! failure: the experts write the session's own fault cell, which
//! `append` reads after the forward and turns into a panic. The cell
//! is the session's, so a fault never reaches another session over the same
//! model. The caller replays the sequence from its
//! archive. There is no per-token snapshot.
//!
//! Memory. A sliding visit drops keys and values no later query can see; a
//! full visit keeps all of them ([`super::attend::Kv`]). So a session's state
//! grows with its length only in the eight full-attention visits (layers 5 and
//! 23 once, 11 and 17 on each of three passes): `2 * kv_dim * 4` bytes, 8 KiB,
//! per token per full visit. Attention's scratch is bounded by its tiles
//! ([`super::attend::QUERY_TILE`] by [`super::attend::KEY_TILE`]), not by the
//! sequence.
//!
//! A healthy session can fork in process. Its state stays on its bound device. The head runs on the
//! device ([`DeviceHead`]), the host [`Head`]'s arithmetic in its order.
//! Logits are the release's: `unpadded_vocab_size` softcapped f32 values per
//! row. Nothing here samples.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use burn::prelude::*;
use burn::tensor::TensorData;
use triblespace::core::collection::CollectionHandle;
use triblespace::prelude::Id;

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::burn::{linear_bf16, rms_norm};
use crate::models::inkling::seam::client_of;
use burn::tensor::FloatDType;
use burn::tensor::activation::softmax;
use crate::models::inkling::seam::Bk;
use crate::models::inkling2_loop::prof;
use crate::models::inkling2_loop::config::{AttnKind, Inkling2LoopConfig, Inkling2LoopTextConfig};
use crate::models::inkling2_loop::full::FullLayer;
use crate::models::inkling2_loop::head::{self, Head};
use crate::models::inkling2_loop::layout::{Dtype, Slot};
use crate::models::inkling2_loop::loop_core::{Core, CoreState};
use crate::models::inkling2_loop::moe::{FaultCell, fault_cell, read_fault};
use crate::models::inkling2_loop::pile::MiniRoot;
use crate::models::inkling2_loop::schedule::CacheId;
use crate::models::inkling2_loop::sliding::{Dev, SlidingLayer, State, Weights, bf16_to_f32, want};

/// The most logit rows one call returns, so a long prefill cannot ask for
/// `rows x 200058` floats on the host by accident.
pub const MAX_LOGIT_ROWS: usize = 64;

/// Which rows of a call's logits to compute and return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rows {
    /// The last row: the distribution over the token after the call's last.
    Last,
    /// The last `n` rows (`1..=MAX_LOGIT_ROWS`, at most the call's length).
    Tail(usize),
}

/// `[rows, vocab]` softcapped f32 logits. The last row predicts the token at
/// [`MiniSession::position`].
#[derive(Debug, Clone)]
pub struct Logits {
    pub rows: usize,
    pub vocab: usize,
    pub values: Vec<f32>,
}

impl Logits {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.values[i * self.vocab..(i + 1) * self.vocab]
    }

    pub fn last(&self) -> &[f32] {
        self.row(self.rows - 1)
    }
}

/// A layer outside the core, bound once.
enum Outer {
    Sliding(SlidingLayer),
    Full(FullLayer),
}

impl Outer {
    fn bind(w: &dyn Weights, t: &Inkling2LoopTextConfig, layer: usize, dev: &Dev) -> Result<Outer, String> {
        Ok(match t.attn_kind(layer) {
            AttnKind::Global => Outer::Full(FullLayer::bind(w, t, layer, dev)?),
            _ => Outer::Sliding(SlidingLayer::bind(w, t, layer, dev)?),
        })
    }

    fn layer(&self) -> usize {
        match self {
            Outer::Sliding(l) => l.layer,
            Outer::Full(l) => l.layer,
        }
    }

    fn new_state(&self, dev: &Dev, fault: &FaultCell) -> State {
        match self {
            Outer::Sliding(l) => State::sharing_fault(&l.dims, dev, fault),
            Outer::Full(l) => State::sharing_fault(&l.dims, dev, fault),
        }
    }

    fn forward(&self, x: Tensor<Bk, 2>, blocks: &mut Vec<Tensor<Bk, 2>>, state: &mut State) -> Tensor<Bk, 2> {
        match self {
            Outer::Sliding(l) => l.forward(x, blocks, state).0,
            // a session starts at position 0: the first cached row is absolute position 0
            Outer::Full(l) => l.forward_at(x, blocks, state, 0).0,
        }
    }
}

/// Inkling 2 Mini, bound once from a model root in a pile: config, chat template
/// and weights, all from that root. Text becomes ids in a
/// [`super::chat::ChatCodec`] over the same root's tokenizer graph.
pub struct MiniModel {
    root: MiniRoot,
    cfg: Inkling2LoopConfig,
    dev: Dev,
    embed_gain: Tensor<Bk, 1>,
    pre: Vec<Outer>,
    core: Core,
    post: Vec<Outer>,
    head: DeviceHead,
}

/// The output head on the device: [`Head`]'s four steps in its order and at
/// its precisions (the block residual and the learned norm in f32, the normed
/// state rounded to BF16, the BF16 unembedding with an f32 accumulator rounded
/// to BF16, the softcap in f32). The unembedding is bound once as the stored
/// BF16, padded rows and all (`vocab_size` is a whole number of tiles); the
/// pad rows' outputs are sliced away.
struct DeviceHead {
    query: Tensor<Bk, 1>,
    gain: Tensor<Bk, 1>,
    unembed: Bf16W,
    eps: f64,
    cap: f32,
    vocab: usize,
}

impl DeviceHead {
    fn bind(t: &Inkling2LoopTextConfig, root: &MiniRoot, dev: &Dev) -> Result<DeviceHead> {
        let h = t.hidden_size;
        // the host head checks every dtype and shape this reads
        Head::from_root(t, root).map_err(|e| anyhow!("bind the head: {e}"))?;
        let (_, _, query) = root.tensor(head::QUERY).map_err(|e| anyhow!(e))?;
        let (_, _, norm) = root.tensor(head::NORM).map_err(|e| anyhow!(e))?;
        let (_, _, unembed) = root.tensor(head::UNEMBED).map_err(|e| anyhow!(e))?;
        let query: Vec<f32> = query.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let query = Tensor::<Bk, 1>::from_data(TensorData::new(query, [h]), dev);
        let gain = Tensor::<Bk, 1>::from_data(TensorData::new(bf16_to_f32(norm), [h]), dev);
        ensure!(Bf16W::tileable(t.vocab_size, h), "the unembedding {}x{h} does not tile", t.vocab_size);
        let client = client_of(&query);
        let unembed = Bf16W { h: client.create_from_slice(unembed), n: t.vocab_size, k: h, align: 16 };
        Ok(DeviceHead {
            query,
            gain,
            unembed,
            eps: t.rms_norm_eps,
            cap: t.final_logit_softcapping as f32,
            vocab: t.unpadded_vocab_size,
        })
    }

    /// `blocks` and `partial` are `[rows, hidden]` each; returns the
    /// softcapped `[rows, vocab]` logits.
    fn forward(&self, blocks: Vec<Tensor<Bk, 2>>, partial: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let [rows, h] = partial.dims();
        let mut states = blocks;
        states.push(partial);
        let n = states.len();
        // the one-query block residual over the unweighted-normed states
        let x = Tensor::stack::<3>(states, 1);
        let ms = x.clone().powf_scalar(2.0).mean_dim(2);
        let normed = x / ms.add_scalar(head::RESIDUAL_EPS).sqrt();
        let scores = (normed.clone() * self.query.clone().reshape([1, 1, h])).sum_dim(2) * (h as f32).powf(-0.5);
        let w = softmax(scores, 1);
        let selected = (normed * w).sum_dim(1).reshape([rows, h]);
        debug_assert!(n > 0);
        let bf16 = |x: Tensor<Bk, 2>| x.cast(FloatDType::BF16).cast(FloatDType::F32);
        let y = bf16(rms_norm(selected, self.gain.clone(), self.eps));
        let raw = bf16(linear_bf16(y, &self.unembed).slice([0..rows, 0..self.vocab]));
        (raw / self.cap).tanh() * self.cap
    }
}

impl MiniModel {
    /// Open `root` in `collection` of the pile at `path` and bind every layer
    /// on the default CUDA device. The weights are views over the pile
    /// mapping until bound; binding uploads them.
    pub fn open(path: &Path, collection: CollectionHandle, root: Id) -> Result<MiniModel> {
        Self::from_root(MiniRoot::open(path, collection, root).context("open the Mini model root")?)
    }

    /// [`Self::open`] from a root that is open already, so that a caller that
    /// opened it to admit a prompt does not open and validate it again.
    pub fn from_root(root: MiniRoot) -> Result<MiniModel> {
        let cfg = Inkling2LoopConfig::from_json(&root.config().context("config facts")?.to_string())
            .map_err(|e| anyhow!("config: {e}"))?;
        let dev = Dev::default();
        let t = &cfg.text_config;
        let l = &t.loop_config;
        let h = t.hidden_size;
        let gain = want(&root, Slot::EmbedNorm, Dtype::Bf16, &[h]).map_err(|e| anyhow!(e))?;
        let embed_gain = Tensor::<Bk, 1>::from_data(TensorData::new(bf16_to_f32(gain.bytes), [h]), &dev);
        want(&root, Slot::Embed, Dtype::Bf16, &[t.vocab_size, h]).map_err(|e| anyhow!(e))?;
        let outer = |range: std::ops::Range<usize>| -> Result<Vec<Outer>> {
            range
                .map(|layer| Outer::bind(&root, t, layer, &dev).map_err(|e| anyhow!("bind layer {layer}: {e}")))
                .collect()
        };
        let pre = outer(0..l.core_start)?;
        let core = Core::bind(&root, t, &dev).map_err(|e| anyhow!("bind the core: {e}"))?;
        let post = outer(l.core_end..t.num_hidden_layers)?;
        let head = DeviceHead::bind(t, &root, &dev)?;
        Ok(MiniModel { root, cfg, dev, embed_gain, pre, core, post, head })
    }

    pub fn config(&self) -> &Inkling2LoopTextConfig {
        &self.cfg.text_config
    }

    /// The root's chat template, as imported.
    pub fn chat_template(&self) -> Result<String> {
        self.root.chat_template()
    }

    pub fn model_root(&self) -> Id {
        self.root.root()
    }

    pub fn tokenizer_root(&self) -> Id {
        self.root.tokenizer_root()
    }

    pub fn collection(&self) -> CollectionHandle {
        self.root.collection()
    }

    /// The resolved model root itself, for anything else read from it.
    pub fn root(&self) -> &MiniRoot {
        &self.root
    }

    /// Logits per row: `unpadded_vocab_size`.
    pub fn vocab(&self) -> usize {
        self.config().unpadded_vocab_size
    }

    /// Rows of the embedding table; an input ID must be below this.
    pub fn embedding_rows(&self) -> usize {
        self.config().vocab_size
    }

    /// Fresh caches for one sequence: every visit's state, and the one router
    /// fault cell all of them write.
    fn fresh(&self) -> Caches {
        let fault = fault_cell(&self.dev);
        Caches {
            pre: self.pre.iter().map(|l| l.new_state(&self.dev, &fault)).collect(),
            core: self.core.new_state_sharing_fault(&self.dev, &fault),
            post: self.post.iter().map(|l| l.new_state(&self.dev, &fault)).collect(),
            fault,
            #[cfg(test)]
            nan_input: false,
        }
    }

    fn embed(&self, ids: &[u32]) -> Tensor<Bk, 2> {
        let h = self.config().hidden_size;
        let table = Weights::tensor(&self.root, &Slot::Embed.tensor_name()).expect("checked at open");
        let mut v = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            let id = id as usize;
            v.extend(bf16_to_f32(&table.bytes[id * h * 2..(id + 1) * h * 2]));
        }
        let x = Tensor::<Bk, 2>::from_data(TensorData::new(v, [ids.len(), h]), &self.dev);
        rms_norm(x, self.embed_gain.clone(), self.config().rms_norm_eps)
    }

    /// The whole forward of `ids` through `caches`, and the head on the last
    /// `rows` rows. Input is checked by the caller.
    fn forward(&self, ids: &[u32], caches: &mut Caches, rows: usize) -> Logits {
        let t = self.config();
        let (n, h) = (ids.len(), t.hidden_size);
        prof::at(usize::MAX, 0);
        let mut x = prof::span("embed", &self.dev, || self.embed(ids));
        #[cfg(test)]
        if std::mem::take(&mut caches.nan_input) {
            x = x.mul_scalar(f32::NAN);
        }
        let mut blocks: Vec<Tensor<Bk, 2>> = Vec::new();
        for (layer, st) in self.pre.iter().zip(caches.pre.iter_mut()) {
            prof::at(layer.layer(), 0);
            x = layer.forward(x, &mut blocks, st);
        }
        let mut passes = self.core.forward(x, &blocks, &mut caches.core, 0, None, None);
        let last = passes.pop().expect("the core runs at least one pass");
        let (mut x, mut blocks) = (last.hidden, last.blocks);
        for (layer, st) in self.post.iter().zip(caches.post.iter_mut()) {
            prof::at(layer.layer(), 0);
            x = layer.forward(x, &mut blocks, st);
        }
        prof::at(usize::MAX, 0);
        // The head needs only the requested trailing rows.
        let tail = |v: Tensor<Bk, 2>| v.slice([n - rows..n, 0..h]);
        let logits = self.head.forward(blocks.into_iter().map(tail).collect(), tail(x));
        let values = logits.into_data().convert::<f32>().to_vec().expect("f32");
        Logits { rows, vocab: self.head.vocab, values }
    }
}

/// Every layer visit's state.
struct Caches {
    pre: Vec<State>,
    core: CoreState,
    post: Vec<State>,
    /// The router fault cell every visit above writes. A sequence owns it:
    /// another session over the same model has its own, and [`MiniSession::reset`]
    /// replaces it with the caches.
    fault: FaultCell,
    /// Test only: the next forward multiplies its embedded input by NaN, so
    /// the first layer's router sees non-finite logits by the ordinary path.
    #[cfg(test)]
    nan_input: bool,
}

impl Caches {
    /// The panic of `Route::picks`, after a forward: a set cell is a non-finite router logit somewhere in
    /// the call. The read is a sync; the head has already synced by then.
    fn check_router(&self) {
        if let Some(flag) = read_fault(&self.fault) {
            panic!("router logit is non-finite, logit {}", flag - 1);
        }
    }
}

/// One token sequence over a shared [`MiniModel`]: its own state for every
/// layer visit and its own position.
pub struct MiniSession {
    model: Arc<MiniModel>,
    caches: Caches,
    position: usize,
    poisoned: bool,
}

impl MiniSession {
    /// A fresh sequence at position 0 against `model`'s warm weights.
    pub fn new(model: Arc<MiniModel>) -> MiniSession {
        let caches = model.fresh();
        MiniSession { model, caches, position: 0, poisoned: false }
    }

    pub fn model(&self) -> &Arc<MiniModel> {
        &self.model
    }

    /// Tokens this session has attended to.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether a failed call left the caches partly updated; only
    /// [`Self::reset`] clears it.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Snapshot a completed append on this model and device. Full immutable KV
    /// pages share storage. Partial pages and convolution tails follow Burn's
    /// copy-on-write semantics. Each branch gets an independent router fault cell.
    pub fn fork(&self) -> Result<Self> {
        ensure!(!self.poisoned, "cannot fork a failed Mini session");
        // CubeCL reports allocation failures by unwinding. This optional fork
        // has not touched the original session and can be declined.
        let fault = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fault_cell(&self.model.dev)))
            .map_err(|_| anyhow!("could not allocate a private Mini router fault cell"))?;
        let caches = Caches {
            pre: self.caches.pre.iter().map(|s| s.fork(&fault)).collect(),
            core: CoreState { visits: self.caches.core.visits.iter()
                .map(|(id, s)| (*id, s.fork(&fault))).collect() },
            post: self.caches.post.iter().map(|s| s.fork(&fault)).collect(),
            fault,
            #[cfg(test)]
            nan_input: false,
        };
        Ok(Self { model: Arc::clone(&self.model), caches, position: self.position, poisoned: false })
    }

    /// Drop every cache and return to position 0. The weights stay bound.
    pub fn reset(&mut self) {
        self.caches = self.model.fresh();
        self.position = 0;
        self.poisoned = false;
    }

    /// The state identities this session holds, in execution order.
    pub fn cache_ids(&self) -> Vec<CacheId> {
        let mut ids: Vec<CacheId> = self.model.pre.iter().map(|l| CacheId { layer: l.layer(), pass: 0 }).collect();
        ids.extend(self.caches.core.visits.keys().copied());
        ids.extend(self.model.post.iter().map(|l| CacheId { layer: l.layer(), pass: 0 }));
        ids
    }

    /// Attend to `ids` after everything already seen and return the logits of
    /// the requested rows. A prompt and a single decoded token are the same
    /// call.
    pub fn append(&mut self, ids: &[u32], rows: Rows) -> Result<Logits> {
        if self.poisoned {
            bail!("this session failed mid-forward and must be reset and replayed");
        }
        ensure!(!ids.is_empty(), "append needs at least one token");
        let limit = self.model.embedding_rows();
        if let Some((i, &id)) = ids.iter().enumerate().find(|(_, id)| **id as usize >= limit) {
            bail!("token {i} is id {id}, the embedding has {limit} rows");
        }
        let rows = match rows {
            Rows::Last => 1,
            Rows::Tail(n) => {
                ensure!(
                    (1..=MAX_LOGIT_ROWS).contains(&n) && n <= ids.len(),
                    "Rows::Tail({n}): want 1..={} and at most the call's {} tokens",
                    MAX_LOGIT_ROWS,
                    ids.len()
                );
                n
            }
        };
        let model = Arc::clone(&self.model);
        let caches = &mut self.caches;
        let call = || {
            let logits = model.forward(ids, caches, rows);
            caches.check_router();
            logits
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
            Ok(logits) => {
                self.position += ids.len();
                Ok(logits)
            }
            Err(cause) => {
                self.poisoned = true;
                let why = cause
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| cause.downcast_ref::<&str>().copied())
                    .unwrap_or("non-text panic");
                Err(anyhow!("forward failed at position {}: {why}; the session is poisoned", self.position))
            }
        }
    }

    /// [`Self::append`] of a prompt, returning the last row.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<Logits> {
        self.append(ids, Rows::Last)
    }

    /// [`Self::append`] of one token, returning its row.
    pub fn decode(&mut self, id: u32) -> Result<Logits> {
        self.append(&[id], Rows::Last)
    }
}

// Drive's `Mind` is `Send`: it owns an `Arc<MiniModel>` and one session.
const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<MiniModel>();
    send::<MiniSession>();
};

#[cfg(test)]
mod tests {
    //! On the release's weights: the router-fault boundary (a non-finite router
    //! logit fails its own session's append and no other session's), and the
    //! experts' own determinism. Needs the
    //! imported release (`MINI_MODEL_PILE`, `MINI_MODEL_COLLECTION`,
    //! `MINI_MODEL_ROOT`) and a GPU; the fault is induced by the test-only
    //! `nan_input` flag, so the cell is written by the real experts in
    //! layer 0 and read by the real `append`.
    use super::*;

    fn open() -> Arc<MiniModel> {
        let env = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"));
        let collection = crate::model_collection::parse_collection_handle(&env("MINI_MODEL_COLLECTION")).unwrap();
        let root = Id::from_hex(&env("MINI_MODEL_ROOT")).expect("MINI_MODEL_ROOT is a 32-hex id");
        Arc::new(MiniModel::open(Path::new(&env("MINI_MODEL_PILE")), collection, root).unwrap())
    }

    fn bits(l: &Logits) -> Vec<u32> {
        l.values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    #[ignore = "needs the imported Mini release (MINI_MODEL_PILE, MINI_MODEL_COLLECTION, MINI_MODEL_ROOT) and a GPU"]
    fn a_router_fault_poisons_only_its_own_session() {
        let model = open();
        let ids_a: Vec<u32> = (0..24).map(|i| 1000 + 37 * i).collect();
        let ids_b: Vec<u32> = (0..17).map(|i| 2000 + 53 * i).collect();
        let (next_a, next_b) = (4321, 5678);

        let mut a = MiniSession::new(Arc::clone(&model));
        let mut b = MiniSession::new(Arc::clone(&model));
        let a0 = a.prefill(&ids_a).expect("A prefill");
        let b0 = b.prefill(&ids_b).expect("B prefill");

        a.caches.nan_input = true;
        let at = a.position();
        let err = a.decode(next_a).expect_err("A's faulted decode must fail").to_string();
        eprintln!("SESSION fault: A failed with: {err}");
        assert!(err.contains("router logit is non-finite"), "A failed for another reason: {err}");
        assert!(a.is_poisoned(), "A is not poisoned");
        assert_eq!(a.position(), at, "A's position moved");
        assert!(a.decode(next_a).is_err(), "a poisoned A accepted a call");

        let b1 = b.decode(next_b).expect("B after A's fault");
        assert!(b1.values.iter().all(|v| v.is_finite()), "B's logits are not finite");
        let mut c = MiniSession::new(Arc::clone(&model));
        let c0 = c.prefill(&ids_b).expect("C prefill");
        let c1 = c.decode(next_b).expect("C decode");
        let same = |x: &Logits, y: &Logits| bits(x).iter().zip(bits(y)).filter(|(p, q)| **p == *q).count();
        eprintln!(
            "SESSION fault: B vs a fresh run of B's sequence, prefill {} of {} equal, decode {} of {} equal",
            same(&b0, &c0), b0.values.len(), same(&b1, &c1), b1.values.len()
        );
        assert_eq!(bits(&b0), bits(&c0), "B's prefill differs from a fresh run");
        assert_eq!(bits(&b1), bits(&c1), "B's decode after A's fault differs from a fresh run");

        a.reset();
        assert!(!a.is_poisoned() && a.position() == 0);
        let a0r = a.prefill(&ids_a).expect("A replay after reset");
        assert_eq!(bits(&a0), bits(&a0r), "A's replay after reset differs from its first prefill");
        a.decode(next_a).expect("A decodes after reset");
        assert_eq!(a.position(), ids_a.len() + 1);
        eprintln!("SESSION fault: A reset, replayed bit for bit, decoded to position {}", a.position());
    }

    /// Release weights: a captured decode-step row at the experts' input, and
    /// the last row of a replay, each alone and inside the replay's call under
    /// one routing, give the same activation codes and the same expert output,
    /// bit for bit. Layer 0 (NVFP4) on its own operands; layer 20 (MXFP8) on
    /// the embedding, which is not what reaches it, but the claim is per operand.
    #[test]
    #[ignore = "needs the imported Mini release (MINI_MODEL_PILE, MINI_MODEL_COLLECTION, MINI_MODEL_ROOT) and a GPU"]
    fn identical_expert_input_gives_identical_codes_and_output() {
        let model = open();
        let ids: Vec<u32> = (0..27).map(|i| 1000 + 37 * i).collect();
        let n = ids.len();
        for layer in [0usize, 20] {
            let outer = if layer == 0 { &model.pre[0] } else { &model.post[layer - model.config().loop_config.core_end] };
            let Outer::Sliding(l) = outer else { panic!("layer {layer} is not sliding") };
            assert_eq!((l.layer, l.moe.is_nvfp4()), (layer, layer == 0));
            let visit = |ids: &[u32], st: &mut State| {
                let x = model.embed(ids);
                let mut blocks = if layer == 0 { Vec::new() } else { vec![x.clone()] };
                l.expert_input(&l.forward(x, &mut blocks, st).1.mlp_input)
            };
            let whole = visit(&ids, &mut State::new(&l.dims, &model.dev));
            let mut st = State::new(&l.dims, &model.dev);
            visit(&ids[..n - 1], &mut st);
            let step = visit(&ids[n - 1..], &mut st);
            let [_, h] = whole.dims();
            let mixed = Tensor::cat(vec![whole.clone().slice([0..n - 1, 0..h]), step], 0);
            for (what, x) in [("the replay's row", &whole), ("the cached step's row", &mixed)] {
                let line = l.moe.row_is_its_own(x, &l.moe.route(x), n - 1).unwrap_or_else(|e| panic!("layer {layer}, {what}: {e}"));
                eprintln!("EXPERT INPUT layer {layer}, {what}: {line}");
            }
        }
    }
}
