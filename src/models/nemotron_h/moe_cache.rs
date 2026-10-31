//! Execution operands for one immutable provider/layer, never a pile catalogue.
//! A miss owns one extra pair until completion. Failed completion retains it;
//! only successfully executed and fenced pairs enter the bounded resident set.
use super::{EXPERTS, Result};
use std::sync::Arc;
use triblespace::core::patch::{Entry, IdentitySchema, PATCH};

struct Cached<T> {
    value: Arc<T>,
    touched: u64,
}
struct Pending<T> {
    key: [u8; 2],
    value: Arc<T>,
    hit: bool,
}

pub(super) struct Operands<T> {
    capacity: usize,
    entries: PATCH<2, IdentitySchema, Cached<T>>,
    clock: u64,
    pending: Option<Pending<T>>,
}

impl<T> Operands<T> {
    pub(super) fn new(capacity: usize) -> Result<Self> {
        if capacity > EXPERTS {
            return Err("MoE expert cache exceeds expert count".into());
        }
        Ok(Self {
            capacity,
            entries: PATCH::new(),
            clock: 0,
            pending: None,
        })
    }

    /// Install the sole pending owner before projection dispatch. A hit skips
    /// the complete query/validation/upload closure, not just the device copy.
    pub(super) fn stage(&mut self, index: usize, load: impl FnOnce() -> Result<T>) -> Result<()> {
        if self.pending.is_some() || index >= EXPERTS {
            return Err("MoE expert operands pending or index out of range".into());
        }
        let key = (index as u16).to_be_bytes();
        let (value, hit) = match self.entries.get(&key) {
            Some(entry) => (entry.value.clone(), true),
            None => (Arc::new(load()?), false),
        };
        self.pending = Some(Pending { key, value, hit });
        Ok(())
    }

    pub(super) fn pending(&self) -> &T {
        self.pending
            .as_ref()
            .expect("installed before dispatch")
            .value
            .as_ref()
    }

    /// Recency is private to this owner. PATCH equality observes keys only;
    /// neither equality nor set algebra is used to observe mutable recency.
    fn tick(&mut self) -> u64 {
        if self.clock == u64::MAX {
            // Ephemeral scratch for an exceptionally rare counter rebase.
            let mut order: Vec<_> = self
                .entries
                .iter_entries_ordered()
                .map(|(key, entry)| (entry.touched, *key))
                .collect();
            order.sort_unstable();
            for (rank, (_, key)) in order.iter().enumerate() {
                let value = self.entries.get(key).unwrap().value.clone();
                self.entries.replace(&Entry::with_value(
                    key,
                    Cached {
                        value,
                        touched: rank as u64,
                    },
                ));
            }
            self.clock = order.len() as u64;
        }
        self.clock += 1;
        self.clock
    }

    /// Returns whether a completed miss evicted one resident pair. An Err is
    /// not a drain, even if execution failed before its last projection.
    pub(super) fn complete(&mut self, completion: &Result<()>, executed: bool) -> bool {
        if completion.is_err() {
            return false;
        }
        let Some(pending) = self.pending.take() else {
            return false;
        };
        if !executed || self.capacity == 0 {
            return false;
        }
        let touched = self.tick();
        if pending.hit {
            self.entries.replace(&Entry::with_value(
                &pending.key,
                Cached {
                    value: pending.value,
                    touched,
                },
            ));
            return false;
        }
        let evicted = self.entries.len() == self.capacity as u64;
        if evicted {
            let key = self
                .entries
                .iter_entries_ordered()
                .min_by_key(|(key, entry)| (entry.touched, **key))
                .map(|(key, _)| *key)
                .expect("full nonempty cache");
            self.entries.remove(&key);
        }
        self.entries.insert(&Entry::with_value(
            &pending.key,
            Cached {
                value: pending.value,
                touched,
            },
        ));
        evicted
    }
}

#[cfg(test)]
#[path = "moe_cache_tests.rs"]
mod tests;
