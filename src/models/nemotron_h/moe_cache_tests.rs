use super::super::{
    ExpertPair, ExpertProvider, INTERMEDIATE, LATENT, PackedOperation, SourceBinding, load_expert,
};
use super::*;
use std::{cell::Cell, rc::Rc};
use triblespace::prelude::{Id, genid};

struct Provider {
    source: SourceBinding,
    operations: [Id; 2],
    up: Vec<u8>,
    down: Vec<u8>,
    input_scales: [f32; 2],
    queries: Cell<usize>,
}
impl Provider {
    fn new() -> Self {
        let packed = |code, global: f32| {
            let mut bytes = vec![code; INTERMEDIATE * LATENT / 2];
            bytes.extend(vec![0x38; INTERMEDIATE * LATENT / 16]);
            bytes.extend(global.to_le_bytes());
            bytes
        };
        Self {
            source: SourceBinding {
                model_root: *genid(),
                layer: *genid(),
            },
            operations: [*genid(), *genid()],
            up: packed(0x12, 0.5),
            down: packed(0x34, 0.25),
            input_scales: [0.75, 0.125],
            queries: Cell::new(0),
        }
    }
}
impl ExpertProvider for Provider {
    fn source(&self) -> SourceBinding {
        self.source
    }
    fn with_expert<T>(
        &self,
        index: usize,
        use_pair: impl FnOnce(ExpertPair<'_>) -> Result<T>,
    ) -> Result<T> {
        self.queries.set(self.queries.get() + 1);
        use_pair(ExpertPair {
            index,
            up: PackedOperation {
                operation: self.operations[0],
                payload: &self.up,
                input_scale: self.input_scales[0],
            },
            down: PackedOperation {
                operation: self.operations[1],
                payload: &self.down,
                input_scale: self.input_scales[1],
            },
        })
    }
}

// Records the actual validated uploader arguments, not a numerical oracle.
#[derive(Debug, PartialEq)]
struct Upload {
    operation: Id,
    input_scale: f32,
    weight_scale: f32,
    code: u8,
    shape: [usize; 2],
}
fn stage(
    cache: &mut Operands<(Upload, Upload)>,
    provider: &Provider,
    uploads: &Cell<usize>,
) -> Result<()> {
    cache.stage(7, || {
        load_expert(
            provider,
            7,
            |op, n, k| {
                uploads.set(uploads.get() + 1);
                Ok(Upload {
                    operation: op.operation,
                    input_scale: op.input_scale,
                    weight_scale: f32::from_le_bytes(
                        op.payload[op.payload.len() - 4..].try_into().unwrap(),
                    ),
                    code: op.payload[0],
                    shape: [n, k],
                })
            },
            #[cfg(feature = "nemotron-timing")]
            None,
        )
    })
}

#[test]
fn moe_host_cache_hit_skips_query_validation_and_upload() {
    let provider = Provider::new();
    let uploads = Cell::new(0);
    let mut cache = Operands::new(1).unwrap();
    stage(&mut cache, &provider, &uploads).unwrap();
    let expected_up = Upload {
        operation: provider.operations[0],
        input_scale: 0.75,
        weight_scale: 0.5,
        code: 0x12,
        shape: [INTERMEDIATE, LATENT],
    };
    let expected_down = Upload {
        operation: provider.operations[1],
        input_scale: 0.125,
        weight_scale: 0.25,
        code: 0x34,
        shape: [LATENT, INTERMEDIATE],
    };
    assert_eq!(cache.pending(), &(expected_up, expected_down));
    assert_eq!((provider.queries.get(), uploads.get()), (1, 2));
    assert!(!cache.complete(&Ok(()), true));
    let first = cache
        .entries
        .get(&7u16.to_be_bytes())
        .unwrap()
        .value
        .clone();
    stage(&mut cache, &provider, &uploads).unwrap();
    assert!(Arc::ptr_eq(&first, &cache.pending.as_ref().unwrap().value));
    assert_eq!((provider.queries.get(), uploads.get()), (1, 2));
    assert!(!cache.complete(&Ok(()), true));

    // Separate owners never share a cache even with the same opaque binding.
    let mut other_provider = Provider::new();
    other_provider.source = provider.source;
    other_provider.input_scales[0] = 0.375;
    let mut other = Operands::new(1).unwrap();
    stage(&mut other, &other_provider, &uploads).unwrap();
    assert_eq!(other.pending().0.input_scale, 0.375);
    assert_eq!(other_provider.queries.get(), 1);
    assert_eq!(first.0.input_scale, 0.75);
}

#[test]
fn moe_host_cache_miss_validates_both_operands_before_any_upload() {
    let mut provider = Provider::new();
    provider.down[INTERMEDIATE * LATENT / 2] = 0x7f;
    let uploads = Cell::new(0);
    let mut cache = Operands::new(1).unwrap();
    assert!(
        stage(&mut cache, &provider, &uploads)
            .unwrap_err()
            .contains("nonfinite")
    );
    assert_eq!((provider.queries.get(), uploads.get()), (1, 0));
    assert!(cache.pending.is_none());
    assert_eq!(cache.entries.len(), 0);
}

#[test]
fn moe_host_cache_zero_capacity_keeps_standalone_miss_behavior() {
    let provider = Provider::new();
    let uploads = Cell::new(0);
    let mut cache = Operands::new(0).unwrap();
    for _ in 0..2 {
        stage(&mut cache, &provider, &uploads).unwrap();
        assert!(!cache.complete(&Ok(()), true));
        assert!(cache.pending.is_none());
        assert_eq!(cache.entries.len(), 0);
    }
    assert_eq!((provider.queries.get(), uploads.get()), (2, 4));
}

struct Tracked(Rc<Cell<usize>>);
impl Drop for Tracked {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn moe_host_cache_failed_completion_retains_pending_and_reset_preserves_residents() {
    let resident_drops = Rc::new(Cell::new(0));
    let miss_drops = Rc::new(Cell::new(0));
    let mut cache = Operands::new(1).unwrap();
    cache
        .stage(1, || Ok(Tracked(resident_drops.clone())))
        .unwrap();
    cache.complete(&Ok(()), true);
    cache.stage(2, || Ok(Tracked(miss_drops.clone()))).unwrap();
    let failure = Err("uncertain CUDA completion".into());
    assert!(!cache.complete(&failure, true));
    assert!(!cache.complete(&failure, false));
    assert!(cache.pending.is_some());
    assert_eq!((resident_drops.get(), miss_drops.get()), (0, 0));
    assert_eq!(cache.entries.len(), 1);
    assert!(
        cache
            .stage(3, || panic!("must not start while pending"))
            .is_err()
    );
    // A later successful drain clears the failed miss, without admitting it.
    assert!(!cache.complete(&Ok(()), false));
    assert_eq!((resident_drops.get(), miss_drops.get()), (0, 1));
    cache
        .stage(1, || panic!("reset should retain immutable residents"))
        .unwrap();
    assert!(!cache.complete(&failure, true));
    assert!(cache.pending.is_some());
    assert_eq!(resident_drops.get(), 0);
    cache.complete(&Ok(()), false);
    drop(cache);
    assert_eq!(resident_drops.get(), 1);
}

#[test]
fn moe_host_cache_partial_upload_failure_is_not_admitted() {
    let provider = Provider::new();
    let uploads = Cell::new(0);
    let drops = Rc::new(Cell::new(0));
    let mut cache = Operands::new(1).unwrap();
    let mut attempt = || {
        cache.stage(7, || {
            load_expert(
                &provider,
                7,
                |_, _, _| {
                    uploads.set(uploads.get() + 1);
                    if uploads.get() == 2 {
                        return Err("down upload failed".into());
                    }
                    Ok(Tracked(drops.clone()))
                },
                #[cfg(feature = "nemotron-timing")]
                None,
            )
        })
    };
    assert!(attempt().unwrap_err().contains("down upload failed"));
    assert_eq!(drops.get(), 1); // the first up never became a complete pair
    attempt().unwrap();
    assert_eq!((provider.queries.get(), uploads.get()), (2, 4));
    cache.complete(&Ok(()), true);
    drop(cache);
    assert_eq!(drops.get(), 3);
}

#[test]
fn moe_host_cache_lru_evicts_only_after_success_and_preserves_value_associations() {
    let mut cache = Operands::new(2).unwrap();
    for (index, value) in [(3, "three"), (1, "one")] {
        cache.stage(index, || Ok(value)).unwrap();
        assert!(!cache.complete(&Ok(()), true));
    }
    cache.stage(3, || panic!("hit")).unwrap();
    cache.complete(&Ok(()), true);
    cache.stage(2, || Ok("two")).unwrap();
    assert_eq!(cache.entries.len(), 2); // two residents plus ONE pending miss
    assert_eq!(cache.pending(), &"two");
    assert!(cache.entries.get(&1u16.to_be_bytes()).is_some());
    assert!(cache.complete(&Ok(()), true));
    assert!(cache.entries.get(&1u16.to_be_bytes()).is_none());
    assert_eq!(cache.entries.len(), 2);
    cache
        .stage(3, || panic!("most recent surviving value"))
        .unwrap();
    assert_eq!(cache.pending(), &"three");
    cache.complete(&Ok(()), true);
    cache.stage(1, || Ok("reloaded one")).unwrap();
    assert_eq!(cache.pending(), &"reloaded one");
    // Execution failure after a successful fence does not admit the miss.
    assert!(!cache.complete(&Ok(()), false));
    assert!(cache.entries.get(&1u16.to_be_bytes()).is_none());
    assert_eq!(
        cache
            .entries
            .get(&2u16.to_be_bytes())
            .unwrap()
            .value
            .as_ref(),
        &"two"
    );
}

#[test]
fn moe_host_cache_rebases_recency_without_changing_payloads() {
    let mut cache = Operands::new(2).unwrap();
    for index in [8, 2] {
        cache.stage(index, || Ok(index + 100)).unwrap();
        cache.complete(&Ok(()), true);
    }
    cache.clock = u64::MAX;
    cache
        .stage(8, || panic!("hit across counter rebase"))
        .unwrap();
    cache.complete(&Ok(()), true);
    cache.stage(4, || Ok(104)).unwrap();
    assert!(cache.complete(&Ok(()), true));
    assert!(cache.entries.get(&2u16.to_be_bytes()).is_none());
    assert_eq!(*cache.entries.get(&8u16.to_be_bytes()).unwrap().value, 108);
    assert!(Operands::<()>::new(EXPERTS + 1).is_err());
    assert!(cache.stage(EXPERTS, || panic!("invalid index")).is_err());
}
