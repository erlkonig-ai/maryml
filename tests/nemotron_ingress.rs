//! CPU-only ingress contract tests. Modules are included directly until the
//! Nemotron model declaration is integrated; no GPU, graph, pile or weights.
#![allow(dead_code)]

#[path = "../src/models/nemotron_h/ingress/config.rs"]
mod config;
#[path = "../src/models/nemotron_h/ingress/packing.rs"]
mod packing;

use config::*;
use packing::*;

fn plane<'a>(dtype: Dtype, shape: &'a [usize], bytes: &'a [u8]) -> Plane<'a> {
    Plane {
        dtype,
        shape,
        bytes,
    }
}

#[test]
fn nvfp4_preserves_bytes_nibbles_rows_and_scale_boundaries() {
    let codes: Vec<u8> = (0..32).map(|i| ((i % 16) << 4) | (15 - i % 16)).collect();
    let scales = [0x38, 0x40, 0x48, 0x50];
    let global = 0.25f32.to_le_bytes();
    let input = 0.125f32.to_le_bytes();
    let packed = pack_nvfp4(
        [2, 32],
        plane(Dtype::U8, &[2, 16], &codes),
        plane(Dtype::E4m3Fn, &[2, 2], &scales),
        plane(Dtype::F32, &[], &global),
        plane(Dtype::F32, &[1], &input),
    )
    .unwrap();
    assert_eq!(packed.logical_shape, [2, 32]);
    assert_eq!(&packed.payload[..32], codes);
    assert_eq!(&packed.payload[32..36], scales);
    assert_eq!(&packed.payload[36..], global);
    assert_eq!(packed.input_scale_le, input);
    // First and second values of each byte remain low- then high-nibble;
    // each row has two groups of 16 and uses its own two scale bytes.
    let e2m1 = [
        0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
    ];
    for i in 0..64 {
        let nibble = (packed.payload[i / 2] >> ((i % 2) * 4)) & 15;
        let source = if i % 2 == 0 {
            15 - (i / 2) % 16
        } else {
            (i / 2) % 16
        };
        assert_eq!(nibble as usize, source);
    }
    let decode = |i: usize| {
        let nibble = (packed.payload[i / 2] >> ((i % 2) * 4)) & 15;
        let s = packed.payload[32 + i / 16];
        let scale = (1. + f32::from(s & 7) / 8.) * 2f32.powi(i32::from(s >> 3) - 7);
        let global = f32::from_le_bytes(packed.payload[36..40].try_into().unwrap());
        e2m1[nibble as usize] * scale * global
    };
    // Explicit independent expectations across both 16-element block and
    // 32-element row boundaries. Input scale does not multiply stored weights.
    for (index, expected) in [
        (0, -1.5),
        (15, 1.5),
        (16, 3.),
        (31, -3.),
        (32, -6.),
        (63, -12.),
    ] {
        assert_eq!(decode(index), expected);
    }
}

#[test]
fn nvfp4_rejects_shapes_lengths_and_swizzled_scale_planes() {
    let codes = [0; 16];
    let scales = [0x38; 2];
    let scalar_bytes = 1f32.to_le_bytes();
    let scalar = plane(Dtype::F32, &[], &scalar_bytes);
    let valid_codes = plane(Dtype::U8, &[2, 8], &codes);
    let valid_scales = plane(Dtype::E4m3Fn, &[2, 1], &scales);
    assert!(pack_nvfp4([2, 16], valid_codes, valid_scales, scalar, scalar).is_ok());
    for bad_codes in [
        plane(Dtype::E4m3Fn, &[2, 8], &codes),
        plane(Dtype::U8, &[8, 2], &codes),
        plane(Dtype::U8, &[2, 8], &codes[..15]),
    ] {
        assert!(pack_nvfp4([2, 16], bad_codes, valid_scales, scalar, scalar).is_err());
    }
    for bad_scales in [
        plane(Dtype::U8, &[2], &scales), // TRT-LLM layout is not this format.
        plane(Dtype::E4m3Fn, &[1, 2], &scales),
        plane(Dtype::E4m3Fn, &[2, 1], &scales[..1]),
    ] {
        assert!(pack_nvfp4([2, 16], valid_codes, bad_scales, scalar, scalar).is_err());
    }
    for shape in [[2, 8], [1, 17], [0, 16], [usize::MAX, 16]] {
        assert!(pack_nvfp4(shape, valid_codes, valid_scales, scalar, scalar).is_err());
    }
}

#[test]
fn fp8_is_not_bf16_and_both_scales_survive() {
    let weight = [0x00, 0x80, 0x38, 0x7e, 0x01, 0x48];
    let weight_scale = 0.5f32.to_le_bytes();
    let input_scale = 0.125f32.to_le_bytes();
    let scalar = plane(Dtype::F32, &[], &weight_scale);
    let input = plane(Dtype::F32, &[1], &input_scale);
    let result = pack_fp8(
        [2, 3],
        plane(Dtype::E4m3Fn, &[2, 3], &weight),
        scalar,
        input,
    )
    .unwrap();
    assert_eq!(result.e4m3fn, weight);
    assert_eq!(result.weight_scale_le, weight_scale);
    assert_eq!(result.input_scale_le, input_scale);
    assert!(pack_fp8([2, 3], plane(Dtype::U8, &[2, 3], &weight), scalar, input).is_err());
    assert!(
        pack_fp8(
            [3, 2],
            plane(Dtype::E4m3Fn, &[2, 3], &weight),
            scalar,
            input
        )
        .is_err()
    );
}

#[test]
fn scalar_shape_dtype_and_byte_length_are_checked() {
    let weight = [0; 16];
    let scales = [0x38; 2];
    let bytes = 1f32.to_le_bytes();
    let scalar = plane(Dtype::F32, &[], &bytes);
    for bad in [
        plane(Dtype::F32, &[1, 1], &bytes),
        plane(Dtype::F32, &[2], &bytes),
        plane(Dtype::F32, &[], &bytes[..3]),
        plane(Dtype::E4m3Fn, &[], &bytes),
    ] {
        let codes = plane(Dtype::U8, &[2, 8], &weight);
        let block_scales = plane(Dtype::E4m3Fn, &[2, 1], &scales);
        assert!(pack_nvfp4([2, 16], codes, block_scales, bad, scalar).is_err());
        assert!(pack_nvfp4([2, 16], codes, block_scales, scalar, bad).is_err());
    }
}

#[test]
fn explicit_quantization_has_no_precision_or_scope_fallback() {
    let quant: QuantizationConfig = serde_json::from_str(r#"{
        "quant_algo":"MIXED_PRECISION", "quantized_layers": {
            "language_model.backbone.layers.0.mixer.in_proj":{"quant_algo":"FP8"},
            "language_model.backbone.layers.1.mixer.experts.0.up_proj":{"quant_algo":"NVFP4","group_size":16},
            "language_model.backbone.bad":{"quant_algo":"NVFP4","group_size":32},
            "language_model.mtp.layers.1.mixer.experts":{"quant_algo":"NVFP4","group_size":16}
        }, "future_metadata": true
    }"#).unwrap();
    assert_eq!(
        quant
            .text_linear("language_model.backbone.layers.0.mixer.in_proj")
            .unwrap(),
        Some(&LinearQuantization::Fp8)
    );
    assert_eq!(
        quant
            .text_linear("language_model.backbone.layers.1.mixer.experts.0.up_proj")
            .unwrap(),
        Some(&LinearQuantization::Nvfp4 { group_size: 16 })
    );
    assert!(quant.text_linear("language_model.backbone.bad").is_err());
    for module in [
        "language_model.backbone.missing",
        "language_model.mtp.layers.1.mixer.experts",
        "vision_model.layer.0",
        "language_model.lm_head.weight",
    ] {
        assert_eq!(quant.text_linear(module).unwrap(), None);
    }
}

#[test]
fn layer_order_and_tagged_infinity_are_not_defaulted() {
    let layers: Vec<LayerKind> =
        serde_json::from_str(r#"["moe","mamba","attention","mamba"]"#).unwrap();
    assert_eq!(
        layers,
        [
            LayerKind::Moe,
            LayerKind::Mamba,
            LayerKind::Attention,
            LayerKind::Mamba
        ]
    );
    let limits: [TimeStepBound; 2] =
        serde_json::from_str(r#"[0.0,{"__float__":"Infinity"}]"#).unwrap();
    assert_eq!(limits, [TimeStepBound(0.), TimeStepBound(f64::INFINITY)]);
    assert!(serde_json::from_str::<TimeStepBound>(r#"{"__float__":"NaN"}"#).is_err());
    assert!(serde_json::from_str::<LayerKind>(r#""unknown""#).is_err());
    assert_eq!(
        serde_json::from_str::<TokenIds>("11").unwrap(),
        TokenIds::One(11)
    );
    assert_eq!(
        serde_json::from_str::<TokenIds>("[2,11]").unwrap(),
        TokenIds::Many(vec![2, 11])
    );
}

#[test]
#[ignore = "requires the pinned public metadata directory in NEMOTRON_EVIDENCE"]
fn pinned_official_metadata() {
    let root = std::path::PathBuf::from(std::env::var("NEMOTRON_EVIDENCE").unwrap());
    let read = |name: &str| std::fs::read(root.join("hf").join(name)).unwrap();
    let config: CheckpointConfig = serde_json::from_slice(&read("config.json")).unwrap();
    let quant: HfQuantConfig = serde_json::from_slice(&read("hf_quant_config.json")).unwrap();
    assert_eq!(config.model_type, "nemotron_h_omni");
    assert_eq!(config.llm_config.model_type, "nemotron_h");
    let text = &config.llm_config;
    assert_eq!(text.layers_block_type.len(), 88);
    assert_eq!(
        text.layers_block_type
            .iter()
            .filter(|&&k| k == LayerKind::Mamba)
            .count(),
        40
    );
    assert_eq!(
        text.layers_block_type
            .iter()
            .filter(|&&k| k == LayerKind::Moe)
            .count(),
        40
    );
    let attention: Vec<_> = text
        .layers_block_type
        .iter()
        .enumerate()
        .filter_map(|(i, k)| (*k == LayerKind::Attention).then_some(i))
        .collect();
    assert_eq!(attention, [7, 16, 25, 36, 47, 58, 69, 78]);
    assert_eq!(
        text.time_step_limit,
        [TimeStepBound(0.), TimeStepBound(f64::INFINITY)]
    );
    assert_eq!(
        (text.hidden_size, text.mamba_num_heads, text.mamba_head_dim),
        (4096, 128, 64)
    );
    assert_eq!((text.n_groups, text.n_group), (8, 1));
    // config.json omits the three MTP selectors present in hf_quant_config.
    // Only text selectors must agree; do not silently synthesize a merged map.
    assert_eq!(config.quantization_config.quantized_layers.len(), 41153);
    assert_eq!(quant.quantization.quantized_layers.len(), 41156);
    for (name, value) in &config.quantization_config.quantized_layers {
        assert!(is_text_module(name), "unexpected config scope: {name}");
        assert_eq!(
            quant.quantization.text_linear(name).unwrap(),
            Some(value),
            "{name}"
        );
    }
    let mut counts = [0, 0];
    for name in quant.quantization.quantized_layers.keys() {
        match quant.quantization.text_linear(name).unwrap() {
            Some(LinearQuantization::Fp8) => counts[0] += 1,
            Some(LinearQuantization::Nvfp4 { group_size: 16 }) => counts[1] += 1,
            None => (),
            other => panic!("unexpected quantization: {other:?}"),
        }
    }
    assert_eq!(counts, [113, 41040]);
    assert_eq!(
        quant
            .quantization
            .text_linear("language_model.lm_head")
            .unwrap(),
        Some(&LinearQuantization::Fp8)
    );
    let generation: GenerationConfig =
        serde_json::from_slice(&read("generation_config.json")).unwrap();
    let tokenizer: TokenizerConfig =
        serde_json::from_slice(&read("tokenizer_config.json")).unwrap();
    assert_eq!(text.eos_token_id, 11);
    assert_eq!(generation.eos_token_id, TokenIds::Many(vec![2, 11]));
    assert_eq!(generation.pad_token_id, 0);
    assert_eq!(tokenizer.pad_token, "<|im_end|>");
    assert_eq!(tokenizer.pad_token, tokenizer.eos_token);
}
