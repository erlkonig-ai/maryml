//! Checkpoint names/shapes, in PyTorch `[out, in]` order; no runtime loader.
//!
//! Shapes follow Transformers 5.2.0 `modeling_qwen3_5.py`: DeltaNet lines
//! 445–517, attention 709–734, vision 915–1128. Text RMSNorm is zero-centered
//! (`1 + weight`), GDN output RMSNorm is ordinary gain + SiLU gate, and vision
//! norms are LayerNorm with bias. Those meanings must survive future loading.

use super::config::{Dtype, LayerType, Qwen3_5Config, product};

/// The checkpoint's head policy is explicit, not inferred from architectures.
/// Both observed configs say untied conditional generation, but Ovis omits the
/// head because it is an embedding checkpoint. There is no embedding projector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmHead {
    /// Require the separate `lm_head.weight` (WeMM).
    Untied,
    /// No head parameter; caller intends only the embedding backbone (Ovis).
    Absent,
    /// Safetensors deduplication omitted the tied head; alias token embeddings.
    /// A redundantly stored head is rejected, not assumed numerically equal.
    TiedToEmbedding,
}

/// Enumerate expected parameters without retaining a second checkpoint catalog.
/// The callback borrows temporary names/shapes; a loader consumes them in place.
/// All tensors have the declared checkpoint dtype, including GDN A_log/dt_bias;
/// their float32 *computation* does not imply float32 storage.
pub fn for_each_parameter(
    config: &Qwen3_5Config,
    head: LmHead,
    mut visit: impl FnMut(&str, &[usize], Dtype),
) -> Result<usize, String> {
    config.validate()?;
    if (head == LmHead::Untied && config.tie_word_embeddings)
        || (head == LmHead::TiedToEmbedding && !config.tie_word_embeddings)
    {
        return Err("lm_head policy disagrees with tie_word_embeddings".into());
    }
    let mut count = 0;
    let mut emit = |name: &str, shape: &[usize]| -> Result<(), String> {
        product(shape).map_err(|e| format!("{name}: {e}"))?;
        count += 1;
        visit(name, shape, config.dtype);
        Ok(())
    };
    let t = &config.text_config;
    let h = t.hidden_size;
    emit(
        "model.language_model.embed_tokens.weight",
        &[t.vocab_size, h],
    )?;
    emit("model.language_model.norm.weight", &[h])?;
    if head == LmHead::Untied {
        emit("lm_head.weight", &[t.vocab_size, h])?;
    }
    for i in 0..t.num_hidden_layers {
        let p = format!("model.language_model.layers.{i}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            emit(&format!("{p}.{norm}.weight"), &[h])?;
        }
        for proj in ["gate_proj", "up_proj"] {
            emit(&format!("{p}.mlp.{proj}.weight"), &[t.intermediate_size, h])?;
        }
        emit(
            &format!("{p}.mlp.down_proj.weight"),
            &[h, t.intermediate_size],
        )?;
        match t.layer_type(i) {
            LayerType::LinearAttention => {
                let key = t.linear_num_key_heads * t.linear_key_head_dim;
                let value = t.linear_num_value_heads * t.linear_value_head_dim;
                let conv = 2 * key + value;
                let p = format!("{p}.linear_attn");
                emit(
                    &format!("{p}.conv1d.weight"),
                    &[conv, 1, t.linear_conv_kernel_dim],
                )?;
                for name in ["A_log", "dt_bias"] {
                    emit(&format!("{p}.{name}"), &[t.linear_num_value_heads])?;
                }
                for (name, out) in [
                    ("in_proj_qkv", conv),
                    ("in_proj_z", value),
                    ("in_proj_a", t.linear_num_value_heads),
                    ("in_proj_b", t.linear_num_value_heads),
                ] {
                    emit(&format!("{p}.{name}.weight"), &[out, h])?;
                }
                emit(&format!("{p}.norm.weight"), &[t.linear_value_head_dim])?;
                emit(&format!("{p}.out_proj.weight"), &[h, value])?;
            }
            LayerType::FullAttention => {
                let q = t.num_attention_heads * t.head_dim;
                let kv = t.num_key_value_heads * t.head_dim;
                let p = format!("{p}.self_attn");
                // Q and gate interleave within each head: [heads, 2 * head_dim].
                for (name, out, input) in [
                    ("q_proj", 2 * q, h),
                    ("k_proj", kv, h),
                    ("v_proj", kv, h),
                    ("o_proj", h, q),
                ] {
                    emit(&format!("{p}.{name}.weight"), &[out, input])?;
                    if t.attention_bias {
                        emit(&format!("{p}.{name}.bias"), &[out])?;
                    }
                }
                for name in ["q_norm", "k_norm"] {
                    emit(&format!("{p}.{name}.weight"), &[t.head_dim])?;
                }
            }
        }
    }
    let v = &config.vision_config;
    let h = v.hidden_size;
    emit(
        "model.visual.patch_embed.proj.weight",
        &[
            h,
            v.in_channels,
            v.temporal_patch_size,
            v.patch_size,
            v.patch_size,
        ],
    )?;
    emit("model.visual.patch_embed.proj.bias", &[h])?;
    emit(
        "model.visual.pos_embed.weight",
        &[v.num_position_embeddings, h],
    )?;
    for i in 0..v.depth {
        let p = format!("model.visual.blocks.{i}");
        for norm in ["norm1", "norm2"] {
            for suffix in ["weight", "bias"] {
                emit(&format!("{p}.{norm}.{suffix}"), &[h])?;
            }
        }
        for (name, out, input) in [
            ("attn.qkv", 3 * h, h),
            ("attn.proj", h, h),
            ("mlp.linear_fc1", v.intermediate_size, h),
            ("mlp.linear_fc2", h, v.intermediate_size),
        ] {
            emit(&format!("{p}.{name}.weight"), &[out, input])?;
            emit(&format!("{p}.{name}.bias"), &[out])?;
        }
    }
    let merged = h * v.spatial_merge_size * v.spatial_merge_size;
    for suffix in ["weight", "bias"] {
        emit(&format!("model.visual.merger.norm.{suffix}"), &[h])?;
    }
    for (name, out, input) in [
        ("linear_fc1", merged, merged),
        ("linear_fc2", v.out_hidden_size, merged),
    ] {
        emit(&format!("model.visual.merger.{name}.weight"), &[out, input])?;
        emit(&format!("model.visual.merger.{name}.bias"), &[out])?;
    }
    Ok(count)
}

/// Verify a complete checkpoint header census: no missing, extra, duplicate,
/// wrong-shape or wrong-dtype parameters. Exclude safetensors `__metadata__`.
/// Only ephemeral metadata references are collected for this one operation.
pub fn validate_checkpoint<'a>(
    config: &Qwen3_5Config,
    head: LmHead,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize], &'a str)>,
) -> Result<usize, String> {
    let mut remaining = std::collections::HashMap::new();
    for (name, shape, dtype) in tensors {
        if remaining.insert(name, (shape, dtype)).is_some() {
            return Err(format!("duplicate checkpoint parameter: {name}"));
        }
    }
    let mut errors = Vec::new();
    let count = for_each_parameter(config, head, |name, shape, dtype| {
        match remaining.remove(name) {
            None => errors.push(format!("missing parameter: {name}")),
            Some((actual_shape, actual_dtype))
                if actual_shape != shape || actual_dtype != dtype.safetensors_name() =>
            {
                errors.push(format!(
                    "{name}: expected {shape:?} {}, got {actual_shape:?} {actual_dtype}",
                    dtype.safetensors_name()
                ));
            }
            Some(_) => {}
        }
    })?;
    errors.extend(
        remaining
            .keys()
            .map(|name| format!("unexpected parameter: {name}")),
    );
    errors.sort();
    if errors.is_empty() {
        Ok(count)
    } else {
        Err(errors.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::tiny;
    use super::*;

    fn census(c: &Qwen3_5Config, head: LmHead) -> Vec<(String, Vec<usize>, &'static str)> {
        let mut entries = Vec::new();
        for_each_parameter(c, head, |n, s, d| {
            entries.push((n.into(), s.to_vec(), d.safetensors_name()))
        })
        .unwrap();
        entries
    }

    #[test]
    fn projection_and_norm_shapes_have_reference_meaning() {
        let c = tiny();
        let entries = census(&c, LmHead::Untied);
        assert_eq!(entries.len(), 49);
        for (name, shape) in [
            (
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                vec![10, 1, 4],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                vec![6, 8],
            ),
            (
                "model.language_model.layers.0.linear_attn.norm.weight",
                vec![3],
            ),
            (
                "model.language_model.layers.1.self_attn.q_proj.weight",
                vec![16, 8],
            ),
            (
                "model.language_model.layers.1.self_attn.k_proj.weight",
                vec![4, 8],
            ),
            (
                "model.language_model.layers.1.self_attn.q_norm.weight",
                vec![4],
            ),
            ("model.visual.patch_embed.proj.weight", vec![8, 3, 2, 2, 2]),
            ("model.visual.merger.norm.weight", vec![8]),
            ("model.visual.merger.linear_fc1.weight", vec![32, 32]),
            ("model.visual.merger.linear_fc2.weight", vec![8, 32]),
        ] {
            assert_eq!(
                entries.iter().find(|e| e.0 == name).unwrap().1,
                shape,
                "{name}"
            );
        }
    }

    #[test]
    fn totality_rejects_missing_extra_duplicate_shape_and_dtype() {
        let c = tiny();
        let good = census(&c, LmHead::Untied);
        let check = |entries: &Vec<(String, Vec<usize>, &'static str)>| {
            validate_checkpoint(
                &c,
                LmHead::Untied,
                entries
                    .iter()
                    .map(|(n, s, d)| (n.as_str(), s.as_slice(), *d)),
            )
        };
        assert_eq!(check(&good).unwrap(), 49);
        let mut bad = good.clone();
        bad.pop();
        assert!(check(&bad).is_err());
        let mut bad = good.clone();
        bad.push(("extra".into(), vec![1], "BF16"));
        assert!(check(&bad).is_err());
        let mut bad = good.clone();
        bad.push(good[0].clone());
        assert!(check(&bad).is_err());
        let mut bad = good.clone();
        bad[0].1.reverse();
        assert!(check(&bad).is_err());
        let mut bad = good.clone();
        bad[0].2 = "F32";
        assert!(check(&bad).is_err());
    }

    #[test]
    fn head_and_attention_bias_policies_are_explicit() {
        let mut c = tiny();
        assert_eq!(census(&c, LmHead::Absent).len(), 48);
        assert!(for_each_parameter(&c, LmHead::TiedToEmbedding, |_, _, _| {}).is_err());
        c.tie_word_embeddings = true;
        c.text_config.tie_word_embeddings = true;
        assert_eq!(census(&c, LmHead::TiedToEmbedding).len(), 48);
        assert!(for_each_parameter(&c, LmHead::Untied, |_, _, _| {}).is_err());
        c.text_config.attention_bias = true;
        assert_eq!(census(&c, LmHead::Absent).len(), 52);
    }

    /// Optional local header-only gate; no tensor payload, GPU or model import.
    /// Run with QWEN3_5_CHECKPOINT=<directory> and QWEN3_5_HEAD=untied|absent|tied.
    #[test]
    #[ignore = "requires a local checkpoint selected by QWEN3_5_CHECKPOINT"]
    fn real_checkpoint_headers() {
        use std::io::Read;
        let directory = std::path::PathBuf::from(
            std::env::var_os("QWEN3_5_CHECKPOINT").expect("checkpoint directory"),
        );
        let head = match std::env::var("QWEN3_5_HEAD")
            .expect("explicit checkpoint head policy")
            .as_str()
        {
            "untied" => LmHead::Untied,
            "absent" => LmHead::Absent,
            "tied" => LmHead::TiedToEmbedding,
            _ => panic!("QWEN3_5_HEAD must be untied, absent or tied"),
        };
        let config = Qwen3_5Config::load(&directory.join("config.json")).unwrap();
        // Scratch metadata for this single validation, never retained by a loader.
        let mut entries: Vec<(String, Vec<usize>, String)> = Vec::new();
        let mut paths: Vec<_> = std::fs::read_dir(&directory)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|s| s == "safetensors"))
            .collect();
        paths.sort();
        assert!(!paths.is_empty(), "no safetensors shards");
        for path in paths {
            let mut file = std::fs::File::open(path).unwrap();
            let mut size = [0; 8];
            file.read_exact(&mut size).unwrap();
            let size = u64::from_le_bytes(size);
            assert!(size <= 64 * 1024 * 1024, "unexpectedly large header");
            let mut bytes = vec![0; size as usize];
            file.read_exact(&mut bytes).unwrap();
            let header: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            for (name, metadata) in header.as_object().unwrap() {
                if name == "__metadata__" {
                    continue;
                }
                let shape = serde_json::from_value(metadata["shape"].clone()).unwrap();
                let dtype = metadata["dtype"].as_str().unwrap().to_owned();
                entries.push((name.clone(), shape, dtype));
            }
        }
        let count = validate_checkpoint(
            &config,
            head,
            entries
                .iter()
                .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_str())),
        )
        .unwrap();
        println!(
            "{}: {count} exact names/shapes/dtypes; head {head:?}",
            directory.display()
        );
    }
}
