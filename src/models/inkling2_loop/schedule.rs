//! The order in which Inkling 2 Mini visits its layers, and what each visit
//! owns.
//!
//! Mini stores 24 physical layers and runs 48 visits per token: layers
//! `0..core_start` once, the core `core_start..core_end` `num_passes` times,
//! then the rest once. Every pass through the core reuses the same *weights*.
//! It does not reuse *state*: each visit attends over its own keys and values
//! and carries its own short-convolution tails. So the weights are keyed by the
//! physical layer and the live state by the pair (physical layer, pass).
//! [`CacheId`] is that pair. It is the identity a resumable session stores
//! state under, which is why it is a pair and not the visit's position: if the
//! post-loop layers were keyed by logical index, a checkpoint run with fewer
//! passes would put layer 18's state where a three-pass run keeps a core
//! visit's.
//!
//! Upstream (transformers `Inkling2LoopTextModel.forward`) addresses the same
//! state by slot index instead: slots `0..num_hidden_layers` for every
//! physical layer's first visit, then
//! `num_hidden_layers + (pass - 1) * core_len + (layer - core_start)` for the
//! later passes. [`CacheId::upstream_slot`] computes that index so a state
//! dump from the reference can be lined up with ours.
//!
//! Each core pass after the first starts again from the hidden state and
//! block states the pre-loop layers produced, and reads cross-pass messages
//! from the previous pass's outputs. [`Visit::reads_messages`] marks those
//! visits.

use super::config::{AttnKind, Inkling2LoopTextConfig};

/// The state one layer visit owns: its keys and values and its convolution
/// tails. Shared weights, separate state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheId {
    /// The physical layer whose weights the visit runs.
    pub layer: usize,
    /// Zero for every layer outside the core and for the core's first pass.
    pub pass: usize,
}

impl CacheId {
    /// The slot transformers keeps this state in.
    pub fn upstream_slot(&self, t: &Inkling2LoopTextConfig) -> usize {
        if self.pass == 0 {
            self.layer
        } else {
            let l = &t.loop_config;
            let core = l.core_end - l.core_start;
            t.num_hidden_layers + (self.pass - 1) * core + (self.layer - l.core_start)
        }
    }
}

/// Where in the stack a visit sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    PreLoop,
    Core,
    PostLoop,
}

/// One layer visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visit {
    /// Position in execution order, `0..logical_layers()`.
    pub logical: usize,
    pub segment: Segment,
    pub cache: CacheId,
    pub attn: AttnKind,
    /// A core visit in a pass after the first: it reads four messages (Q, K,
    /// V and MLP inputs) from the previous pass and mixes them in through its
    /// message gates.
    pub reads_messages: bool,
    /// The hidden state entering this visit is appended to the attention
    /// residual's block states.
    pub block_boundary: bool,
}

impl Visit {
    pub fn layer(&self) -> usize {
        self.cache.layer
    }
}

/// Every visit of one token's forward, in execution order.
#[derive(Debug, Clone)]
pub struct Schedule {
    visits: Vec<Visit>,
}

impl Schedule {
    pub fn of(t: &Inkling2LoopTextConfig) -> Schedule {
        let l = &t.loop_config;
        let mut visits = Vec::with_capacity(t.logical_layers());
        let mut push = |layer: usize, segment: Segment, pass: usize| {
            visits.push(Visit {
                logical: visits.len(),
                segment,
                cache: CacheId { layer, pass },
                attn: t.attn_kind(layer),
                reads_messages: segment == Segment::Core && pass > 0,
                block_boundary: t.is_block_boundary(layer),
            });
        };
        for layer in 0..l.core_start {
            push(layer, Segment::PreLoop, 0);
        }
        for pass in 0..l.num_passes {
            for layer in l.core_start..l.core_end {
                push(layer, Segment::Core, pass);
            }
        }
        for layer in l.core_end..t.num_hidden_layers {
            push(layer, Segment::PostLoop, 0);
        }
        Schedule { visits }
    }

    pub fn visits(&self) -> &[Visit] {
        &self.visits
    }

    pub fn len(&self) -> usize {
        self.visits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.visits.is_empty()
    }

    /// The visits that run `layer`'s weights.
    pub fn visits_of(&self, layer: usize) -> impl Iterator<Item = &Visit> {
        self.visits.iter().filter(move |v| v.cache.layer == layer)
    }

    /// Every state identity a session must hold, in execution order.
    pub fn cache_ids(&self) -> impl Iterator<Item = CacheId> + '_ {
        self.visits.iter().map(|v| v.cache)
    }

    /// Visits with a global (full) attention cache, which grows with the
    /// context; the rest keep at most the sliding window.
    pub fn global_visits(&self) -> impl Iterator<Item = &Visit> {
        self.visits.iter().filter(|v| v.attn == AttnKind::Global)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::Inkling2LoopConfig;
    use super::*;
    use std::collections::BTreeSet;

    fn mini() -> Inkling2LoopTextConfig {
        Inkling2LoopConfig::from_json(include_str!("fixtures/config.json"))
            .unwrap()
            .text_config
    }

    #[test]
    fn the_published_schedule() {
        let t = mini();
        let s = Schedule::of(&t);
        assert_eq!(s.len(), 48);
        let layers: Vec<usize> = s.visits().iter().map(Visit::layer).collect();
        let mut want: Vec<usize> = (0..6).collect();
        for _ in 0..3 {
            want.extend(6..18);
        }
        want.extend(18..24);
        assert_eq!(layers, want);
        // Every visit owns its own state; weights are shared by layer.
        let ids: BTreeSet<CacheId> = s.cache_ids().collect();
        assert_eq!(ids.len(), 48);
        for layer in 0..24 {
            let n = s.visits_of(layer).count();
            assert_eq!(
                n,
                if (6..18).contains(&layer) { 3 } else { 1 },
                "layer {layer}"
            );
        }
        // Messages only on the core's second and third passes.
        assert_eq!(s.visits().iter().filter(|v| v.reads_messages).count(), 24);
        // Four global layers, two of them in the core: 2 + 2 * 3 global caches.
        assert_eq!(s.global_visits().count(), 8);
        let boundaries: Vec<usize> = s
            .visits()
            .iter()
            .filter(|v| v.block_boundary)
            .map(|v| v.logical)
            .collect();
        assert_eq!(boundaries, vec![0, 6, 12, 18, 24, 30, 36, 42]);
    }

    #[test]
    fn upstream_slots_are_a_bijection_onto_its_cache_layout() {
        let t = mini();
        let s = Schedule::of(&t);
        let slots: Vec<usize> = s.cache_ids().map(|c| c.upstream_slot(&t)).collect();
        let distinct: BTreeSet<usize> = slots.iter().copied().collect();
        assert_eq!(distinct, (0..48).collect());
        // Spot values from transformers' `cache_offset` arithmetic.
        let at = |layer, pass| CacheId { layer, pass }.upstream_slot(&t);
        assert_eq!(at(6, 0), 6);
        assert_eq!(at(6, 1), 24);
        assert_eq!(at(17, 1), 35);
        assert_eq!(at(6, 2), 36);
        assert_eq!(at(17, 2), 47);
        assert_eq!(at(23, 0), 23);
    }

    #[test]
    fn identities_do_not_move_when_the_pass_count_does() {
        // The varloop research checkpoint runs the same weights with fewer
        // passes; a post-loop layer's state keeps its identity.
        let mut t = mini();
        let three: Vec<CacheId> = Schedule::of(&t).cache_ids().collect();
        t.loop_config.num_passes = 1;
        let one = Schedule::of(&t);
        assert_eq!(one.len(), 24);
        assert!(one.visits().iter().all(|v| !v.reads_messages));
        for id in one.cache_ids() {
            assert!(three.contains(&id), "{id:?}");
        }
    }
}
