//! Actual GGUF semantics for bounded dense/sparse block fixtures, not HF defaults.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use crate::formats::gguf::index::{MetadataEntry, MetadataValue};

pub const METADATA_KEYS: &[&str] = &[
    "general.architecture", "flint.block_count", "flint.context_length",
    "flint.embedding_length", "flint.feed_forward_length", "flint.attention.head_count", "flint.attention.head_count_kv",
    "flint.attention.key_length", "flint.attention.value_length", "flint.rope.freq_base",
    "flint.attention.layer_norm_rms_epsilon", "flint.leading_dense_block_count",
    "flint.expert_count", "flint.expert_used_count", "flint.expert_gating_func",
    "flint.expert_feed_forward_length", "flint.expert_shared_feed_forward_length",
    "flint.expert_shared_count", "flint.expert_weights_norm", "flint.expert_weights_scale",
    "flint.attention.sliding_window", "flint.attention.sliding_window_pattern",
    "flint.attention.rope_pattern",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FfnKind { Dense { width: usize }, Moe }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct BlockConfig {
    pub block: usize, pub sliding_window: Option<i32>, pub rope: bool, pub ffn: FfnKind,
}

impl BlockConfig {
    /// Legacy synthetic/reference arithmetic entry point; file-backed gates must
    /// use from_metadata before reading weights. This does not assert model identity.
    pub const LAYER2: Self = Self { block: 2, sliding_window: Some(513), rope: true, ffn: FfnKind::Moe };

    pub fn from_metadata(entries: &[MetadataEntry], block: usize) -> Result<Self> {
        use MetadataValue::*;
        ensure!(matches!(block, 0 | 2 | 5), "only isolated blocks 0, 2 and 5 are implemented");
        let get = |key: &str| -> Result<&MetadataValue> {
            let mut matching = entries.iter().filter(|e| e.key == key);
            let entry = matching.next().with_context(|| format!("missing Flint metadata {key}"))?;
            ensure!(matching.next().is_none(), "duplicate Flint metadata {key}");
            let tag = match &entry.value { U32(_) => 4, F32(_) => 6, Bool(_) => 7, String(_) => 8, BoolArray(_) => 9 };
            ensure!(entry.gguf_type == tag, "Flint metadata type differs: {key}");
            Ok(&entry.value)
        };
        ensure!(get("general.architecture")? == &String("flint".into()), "not Flint GGUF architecture");
        for (key, value) in [
            ("block_count",49), ("context_length",262144), ("embedding_length",2048), ("feed_forward_length",6144),
            ("attention.head_count",32), ("attention.head_count_kv",4),
            ("attention.key_length",128), ("attention.value_length",128),
            ("leading_dense_block_count",2), ("expert_count",256), ("expert_used_count",6),
            ("expert_gating_func",2), ("expert_feed_forward_length",512),
            ("expert_shared_feed_forward_length",1024), ("expert_shared_count",2),
            ("attention.sliding_window",513),
        ] { ensure!(get(&format!("flint.{key}"))? == &U32(value), "unsupported Flint metadata {key}"); }
        for (key, value) in [("rope.freq_base",10000.0f32), ("attention.layer_norm_rms_epsilon",1e-6), ("expert_weights_scale",1.0)] {
            ensure!(get(&format!("flint.{key}"))? == &F32(value), "unsupported Flint metadata {key}");
        }
        ensure!(get("flint.expert_weights_norm")? == &Bool(true), "unsupported expert normalization");
        let pattern = |key: &str| -> Result<bool> {
            let BoolArray(values) = get(key)? else { anyhow::bail!("{key} must be BOOL array") };
            ensure!(values.len() == 49, "{key} must have one flag per block");
            Ok(values[block])
        };
        let swa = pattern("flint.attention.sliding_window_pattern")?;
        let rope = pattern("flint.attention.rope_pattern")?;
        ensure!(swa == (block == 2) && rope == (block == 2), "fixture attention semantics differ from actual GGUF");
        // Leading-dense count and dense width were checked above against bytes.
        // Clamp=0 is pinned loader behavior, not an absent-key assumption:
        // llama.cpp 1ddffe9 llama-model.cpp:1471, flint.cpp:3-18.
        let ffn = if block < 2 { FfnKind::Dense { width: 6144 } } else { FfnKind::Moe };
        Ok(Self { block, sliding_window: swa.then_some(513), rope, ffn })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use MetadataValue::*;

    fn fixture() -> Vec<MetadataEntry> {
        let mut entries = vec![MetadataEntry { key: "general.architecture".into(), gguf_type: 8, value: String("flint".into()) }];
        for (key, value) in [
            ("block_count",49), ("context_length",262144), ("embedding_length",2048), ("feed_forward_length",6144),
            ("attention.head_count",32), ("attention.head_count_kv",4),
            ("attention.key_length",128), ("attention.value_length",128),
            ("leading_dense_block_count",2), ("expert_count",256), ("expert_used_count",6),
            ("expert_gating_func",2), ("expert_feed_forward_length",512),
            ("expert_shared_feed_forward_length",1024), ("expert_shared_count",2),
            ("attention.sliding_window",513),
        ] { entries.push(MetadataEntry { key: format!("flint.{key}"), gguf_type: 4, value: U32(value) }); }
        for (key, value) in [("rope.freq_base",10000.0f32), ("attention.layer_norm_rms_epsilon",1e-6), ("expert_weights_scale",1.0)] {
            entries.push(MetadataEntry { key: format!("flint.{key}"), gguf_type: 6, value: F32(value) });
        }
        entries.push(MetadataEntry { key: "flint.expert_weights_norm".into(), gguf_type: 7, value: Bool(true) });
        let mut flags = vec![true;49];
        for i in [0,1,5,9,13,17,21,25,29,33,37,41,45,48] { flags[i] = false; }
        for key in ["flint.attention.sliding_window_pattern", "flint.attention.rope_pattern"] {
            entries.push(MetadataEntry { key: key.into(), gguf_type: 9, value: BoolArray(flags.clone()) });
        }
        entries
    }

    #[test]
    fn actual_semantics_select_only_supported_isolated_blocks() {
        let mut entries = fixture();
        entries.push(MetadataEntry { key: "unrelated.open.world".into(), gguf_type: 8, value: String("ignored".into()) });
        assert_eq!(BlockConfig::from_metadata(&entries,2).unwrap(),BlockConfig::LAYER2);
        assert_eq!(BlockConfig::from_metadata(&entries,5).unwrap(),BlockConfig {block:5,sliding_window:None,rope:false,ffn:FfnKind::Moe});
        assert_eq!(BlockConfig::from_metadata(&entries,0).unwrap(),BlockConfig {block:0,sliding_window:None,rope:false,ffn:FfnKind::Dense {width:6144}});
        assert!(BlockConfig::from_metadata(&entries,1).is_err());
        assert!(BlockConfig::from_metadata(&entries,49).is_err());
    }

    #[test]
    fn reject_missing_ambiguous_wrong_typed_or_changed_semantics() {
        let entries = fixture();
        for index in 0..entries.len() {
            let mut missing = entries.clone(); missing.remove(index);
            assert!(BlockConfig::from_metadata(&missing,5).is_err());
            let mut duplicate = entries.clone(); duplicate.push(entries[index].clone());
            assert!(BlockConfig::from_metadata(&duplicate,5).is_err());
            let mut wrong_type = entries.clone(); wrong_type[index].gguf_type = 99;
            assert!(BlockConfig::from_metadata(&wrong_type,5).is_err());
            let mut changed = entries.clone();
            changed[index].value = match &entries[index].value {
                U32(v) => U32(v+1), F32(_) => F32(f32::NAN), Bool(v) => Bool(!v),
                String(_) => String("not-flint".into()), BoolArray(_) => BoolArray(vec![false;48]),
            };
            assert!(BlockConfig::from_metadata(&changed,5).is_err());
        }
        let mut changed = entries;
        let BoolArray(flags) = &mut changed.last_mut().unwrap().value else { unreachable!() };
        flags[5] = true;
        assert!(BlockConfig::from_metadata(&changed,5).is_err());
    }
}
