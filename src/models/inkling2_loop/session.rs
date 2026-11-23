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
//! call until [`MiniSession::reset`]. The caller replays the sequence from its
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
//! Scope, stated. No checkpoint, rewind or cache export. The head runs on the
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
use crate::models::inkling2_loop::config::{AttnKind, Inkling2LoopConfig, Inkling2LoopTextConfig};
use crate::models::inkling2_loop::full::FullLayer;
use crate::models::inkling2_loop::head::{self, Head};
use crate::models::inkling2_loop::layout::{Dtype, Slot};
use crate::models::inkling2_loop::loop_core::{Core, CoreState};
use crate::models::inkling2_loop::pile::MiniRoot;
use crate::models::inkling2_loop::schedule::CacheId;
use crate::models::inkling2_loop::sliding::{Dev, ExpertAct, SlidingLayer, State, Weights, bf16_to_f32, want};

/// The most logit rows one call returns, so a long prefill cannot ask for
/// `rows x 200058` floats on the host by accident.
pub const MAX_LOGIT_ROWS: usize = 64;

/// How a model binds.
#[derive(Debug, Clone, Copy)]
pub struct MiniOptions {
    /// What the NVFP4 experts multiply. [`ExpertAct::Bf16`] (the default) is
    /// what transformers 046bdb8c computes and what the comparisons checked.
    pub experts: ExpertAct,
}

impl Default for MiniOptions {
    fn default() -> Self {
        MiniOptions { experts: ExpertAct::Bf16 }
    }
}

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

    fn new_state(&self, dev: &Dev) -> State {
        match self {
            Outer::Sliding(l) => State::new(&l.dims, dev),
            Outer::Full(l) => State::new(&l.dims, dev),
        }
    }

    fn forward(&self, x: Tensor<Bk, 2>, blocks: &mut Vec<Tensor<Bk, 2>>, state: &mut State, act: ExpertAct) -> Tensor<Bk, 2> {
        match self {
            Outer::Sliding(l) => l.forward(x, blocks, state, act).0,
            // a session starts at position 0: the first cached row is absolute position 0
            Outer::Full(l) => l.forward_at(x, blocks, state, act, 0).0,
        }
    }
}

/// Inkling 2 Mini, bound once from a model root in a pile: config, tokenizer,
/// chat template and weights, all from that root.
pub struct MiniModel {
    root: MiniRoot,
    cfg: Inkling2LoopConfig,
    tokenizer: tokenizers::Tokenizer,
    content_tokenizer: tokenizers::Tokenizer,
    opts: MiniOptions,
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
    pub fn open(path: &Path, collection: CollectionHandle, root: Id, opts: MiniOptions) -> Result<MiniModel> {
        let root = MiniRoot::open(path, collection, root).context("open the Mini model root")?;
        let cfg = Inkling2LoopConfig::from_json(&root.config().context("config facts")?.to_string())
            .map_err(|e| anyhow!("config: {e}"))?;
        let tok = root.tokenizer_root();
        let tokenizer = crate::tokenizer::build_tokenizer(root.facts(), root.store(), tok)
            .map_err(|e| anyhow!("build the tokenizer from the root's graph: {e:?}"))?;
        let content_tokenizer = crate::tokenizer::build_tokenizer_with_added(root.facts(), root.store(), tok, false)
            .map_err(|e| anyhow!("build the content-only tokenizer from the root's graph: {e:?}"))?;
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
        Ok(MiniModel { root, cfg, tokenizer, content_tokenizer, opts, dev, embed_gain, pre, core, post, head })
    }

    pub fn config(&self) -> &Inkling2LoopTextConfig {
        &self.cfg.text_config
    }

    /// The whole tokenizer, added (special) tokens included, from the root's
    /// own tokenizer graph.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// The same graph without its added tokens: literal marker spellings in
    /// untrusted text become ordinary pieces.
    pub fn content_tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.content_tokenizer
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

    fn fresh(&self) -> Caches {
        Caches {
            pre: self.pre.iter().map(|l| l.new_state(&self.dev)).collect(),
            core: self.core.new_state(&self.dev),
            post: self.post.iter().map(|l| l.new_state(&self.dev)).collect(),
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
        let act = self.opts.experts;
        let mut x = self.embed(ids);
        let mut blocks: Vec<Tensor<Bk, 2>> = Vec::new();
        for (layer, st) in self.pre.iter().zip(caches.pre.iter_mut()) {
            x = layer.forward(x, &mut blocks, st, act);
        }
        let mut passes = self.core.forward(x, &blocks, &mut caches.core, act, 0, None, None);
        let last = passes.pop().expect("the core runs at least one pass");
        let (mut x, mut blocks) = (last.hidden, last.blocks);
        for (layer, st) in self.post.iter().zip(caches.post.iter_mut()) {
            x = layer.forward(x, &mut blocks, st, act);
        }
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
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.forward(ids, caches, rows))) {
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
