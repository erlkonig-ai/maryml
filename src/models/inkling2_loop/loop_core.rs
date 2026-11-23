//! Inkling 2 Mini's looped core, on the device: layers `core_start..core_end`
//! (6..17 in the release), run `num_passes` (3) times over shared weights.
//!
//! From transformers 046bdb8c `Inkling2LoopTextModel.forward`, the lines
//! between the pre-loop and the post-loop layers:
//!
//! ```text
//! pre_h, pre_blocks = h, blocks             what layers 0..5 produced: one block, the embedding
//! pass 0:   for l in core: h = layer_l(h, blocks)                      a boundary layer appends its input
//! pass p>0: prev_blocks = blocks[n_pre + 1 ..]                         drops the pass's first block (= pre_h):
//!                                                                      the release keeps one, layer 12's input
//!           prev_h = h                                                 layer 17's output
//!           h, blocks = pre_h, pre_blocks                              every pass restarts from the pre-loop state
//!           for l in core: h = layer_l(h, blocks, message_input(prev_blocks, prev_h, l - core_start))
//! return h, blocks                                                     pass 3's output; [emb, pre_h, layer-12 input]
//! ```
//!
//! Each layer's four messages use that layer's own queries
//! ([`MessageQueries::layer`]) over the same previous-pass states. The sliding
//! core layers are [`SlidingLayer`], the full ones (11, 17) [`FullLayer`], both
//! bound once and visited once per pass. Each visit carries its own [`State`]
//! across token steps, keyed by [`CacheId`] `(layer, pass)`: shared weights,
//! separate keys, values and convolution tails.
//!
//! The messages are read on the device ([`MessageSources`], the device form of
//! [`super::messages::read_messages`]): the previous pass's states stay where
//! they are, are normed once per pass, and each layer reads its four messages
//! for all of the call's tokens in two matrix multiplies.

use std::collections::BTreeMap;

use burn::prelude::*;

use crate::models::inkling::seam::Bk;
use crate::models::inkling2_loop::prof;
use crate::models::inkling2_loop::config::{AttnKind, Inkling2LoopTextConfig};
use crate::models::inkling2_loop::full::FullLayer;
use crate::models::inkling2_loop::layout::{Dtype, Slot};
use crate::models::inkling2_loop::messages::{DeviceQueries, MessageQueries, MessageSources};
use crate::models::inkling2_loop::moe::{FaultCell, fault_cell};
use crate::models::inkling2_loop::schedule::CacheId;
use crate::models::inkling2_loop::sliding::{Dev, SlidingLayer, State, Taps, Weights, f32_le, want};

/// One core layer, bound once.
pub enum CoreLayer {
    Sliding(SlidingLayer),
    Full(FullLayer),
}

impl CoreLayer {
    pub fn layer(&self) -> usize {
        match self {
            CoreLayer::Sliding(l) => l.layer,
            CoreLayer::Full(l) => l.layer,
        }
    }

    fn new_state(&self, dev: &Dev, fault: &FaultCell) -> State {
        match self {
            CoreLayer::Sliding(l) => State::sharing_fault(&l.dims, dev, fault),
            CoreLayer::Full(l) => State::sharing_fault(&l.dims, dev, fault),
        }
    }

    fn visit(
        &self,
        x: Tensor<Bk, 2>,
        blocks: &mut Vec<Tensor<Bk, 2>>,
        state: &mut State,
        origin: usize,
        messages: Option<&Tensor<Bk, 3>>,
    ) -> (Tensor<Bk, 2>, Taps) {
        match self {
            CoreLayer::Sliding(l) => l.forward_visit(x, blocks, state, messages),
            CoreLayer::Full(l) => {
                let (out, taps) = l.forward_visit(x, blocks, state, origin, messages);
                (out, taps.rest)
            }
        }
    }
}

/// The looped core's weights: its layers and the cross-pass message queries.
pub struct Core {
    pub layers: Vec<CoreLayer>,
    pub queries: MessageQueries,
    device_queries: DeviceQueries,
    pub passes: usize,
    hidden: usize,
}

/// Every core visit's carried state, keyed by `(layer, pass)`.
pub struct CoreState {
    pub visits: BTreeMap<CacheId, State>,
}

/// What a pass ended with, on the device.
pub struct PassOut {
    /// The pass's output, layer 17's.
    pub hidden: Tensor<Bk, 2>,
    /// The block states at the pass's end, `[pre_blocks.., pre_h, layer-12 input]`.
    pub blocks: Vec<Tensor<Bk, 2>>,
}

/// A deliberate fault in the core's composition, for checking that a
/// comparison notices one. Never used outside such a check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plant {
    /// Passes after the first continue from the previous pass's output and
    /// blocks instead of restarting from the pre-loop state.
    NoReset,
    /// Every core layer reads its messages with the first core layer's queries.
    SharedQuery,
}

/// Sees each visit as it ends: its identity, output, taps and the messages it
/// read (`[t, 4, H]`, on the device).
pub type VisitHook<'a> = &'a mut dyn FnMut(CacheId, &Tensor<Bk, 2>, Taps, Option<&Tensor<Bk, 3>>);

impl Core {
    /// Bind the core layers and the message queries from `w`.
    pub fn bind(w: &dyn Weights, t: &Inkling2LoopTextConfig, dev: &Dev) -> Result<Core, String> {
        let l = &t.loop_config;
        let (h, core) = (t.hidden_size, l.core_end - l.core_start);
        let layers = (l.core_start..l.core_end)
            .map(|layer| -> Result<CoreLayer, String> {
                let bound = match t.attn_kind(layer) {
                    AttnKind::Global => CoreLayer::Full(FullLayer::bind(w, t, layer, dev)?),
                    _ => CoreLayer::Sliding(SlidingLayer::bind(w, t, layer, dev)?),
                };
                Ok(bound)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let f32_of = |s: Slot| -> Result<Vec<f32>, String> { Ok(f32_le(want(w, s, Dtype::F32, &[2 * core, h])?.bytes)) };
        let queries = MessageQueries::from_checkpoint(
            &f32_of(Slot::CrossPassQueries)?,
            &f32_of(Slot::CrossPassKvQueries)?,
            core,
            h,
        )?;
        let device_queries = DeviceQueries::new(&queries, dev);
        Ok(Core { layers, queries, device_queries, passes: l.num_passes, hidden: h })
    }

    /// Fresh state for every visit, `passes x layers`, all writing one
    /// router fault cell of their own.
    pub fn new_state(&self, dev: &Dev) -> CoreState {
        self.new_state_sharing_fault(dev, &fault_cell(dev))
    }

    /// [`Self::new_state`], every visit writing its router faults into `fault`.
    pub fn new_state_sharing_fault(&self, dev: &Dev, fault: &FaultCell) -> CoreState {
        let mut visits = BTreeMap::new();
        for pass in 0..self.passes {
            for l in &self.layers {
                visits.insert(CacheId { layer: l.layer(), pass }, l.new_state(dev, fault));
            }
        }
        CoreState { visits }
    }

    /// Run the core over `x` (`[t, H]`, the pre-loop hidden state) and the
    /// pre-loop `blocks` (each `[t, H]`), positions `state`'s next ones. A
    /// prefill and a decode step are the same call. Returns every pass's end;
    /// the last is the core's output (`hidden`, `blocks`) for the post-loop
    /// layers.
    ///
    /// `origin` is the absolute position of the first cached row (only the full
    /// layers' log scale reads it). `hook` sees every visit's output and the
    /// messages it read, in execution order.
    pub fn forward(
        &self,
        x: Tensor<Bk, 2>,
        blocks: &[Tensor<Bk, 2>],
        state: &mut CoreState,
        origin: usize,
        plant: Option<Plant>,
        mut hook: Option<VisitHook<'_>>,
    ) -> Vec<PassOut> {
        let [_, h] = x.dims();
        assert_eq!(h, self.hidden);
        let n_pre = blocks.len();
        let mut passes: Vec<PassOut> = Vec::with_capacity(self.passes);
        // the previous pass's message sources, normed, on the device
        let mut prev: Option<MessageSources> = None;
        for pass in 0..self.passes {
            let (mut hs, mut bs) = match (plant, passes.last()) {
                (Some(Plant::NoReset), Some(p)) => (p.hidden.clone(), p.blocks.clone()),
                _ => (x.clone(), blocks.to_vec()),
            };
            for (i, layer) in self.layers.iter().enumerate() {
                let q = if plant == Some(Plant::SharedQuery) { 0 } else { i };
                let id = CacheId { layer: layer.layer(), pass };
                prof::at(id.layer, pass);
                let dev = hs.device();
                let messages = prof::span("msg_read", &dev, || prev.as_ref().map(|src| src.read(self.device_queries.layer(q))));
                let st = state.visits.get_mut(&id).unwrap_or_else(|| panic!("no state for {id:?}"));
                let (out, taps) = layer.visit(hs, &mut bs, st, origin, messages.as_ref());
                if let Some(f) = hook.as_mut() {
                    f(id, &out, taps, messages.as_ref());
                }
                hs = out;
            }
            // upstream: previous_block_states = block_states[..., num_pre_loop_blocks + 1:, :]
            prof::at(usize::MAX, pass);
            let prev_t0 = prof::enabled().then(std::time::Instant::now);
            let keep = &bs[(n_pre + 1).min(bs.len())..];
            prev = Some(MessageSources::new(keep, hs.clone()));
            if let Some(t0) = prev_t0 {
                prof::add("pass_sources", t0.elapsed().as_secs_f64());
            }
            passes.push(PassOut { hidden: hs, blocks: bs });
        }
        passes
    }
}
