//! Compatibility of the fixed shared Qwen codec implementation, not an exact
//! metadata/fact-set constraint. Unrelated annotations are deliberately ignored.
use anyhow::{Result, ensure};
use serde_json::Value;

fn integers(value: &Value, fields: &[(&str, u64)]) -> Result<()> {
    for &(name, expected) in fields {
        ensure!(
            value[name].as_u64() == Some(expected),
            "unsupported codec {name}: expected {expected}"
        );
    }
    Ok(())
}

fn flags(value: &Value, fields: &[(&str, bool)]) -> Result<()> {
    for &(name, expected) in fields {
        ensure!(
            value[name].as_bool() == Some(expected),
            "unsupported codec {name}: expected {expected}"
        );
    }
    Ok(())
}

pub fn validate_config(value: &Value) -> Result<()> {
    ensure!(
        value["model_type"] == "qwen3_tts_tokenizer_12hz",
        "unsupported external audio tokenizer"
    );
    integers(
        value,
        &[
            ("input_sample_rate", 24000),
            ("output_sample_rate", 24000),
            ("encode_downsample_rate", 1920),
            ("decode_upsample_rate", 1920),
            ("encoder_valid_num_quantizers", 16),
        ],
    )?;
    let decoder = &value["decoder_config"];
    integers(
        decoder,
        &[
            ("latent_dim", 1024),
            ("codebook_dim", 512),
            ("codebook_size", 2048),
            ("decoder_dim", 1536),
            ("hidden_size", 512),
            ("intermediate_size", 1024),
            ("head_dim", 64),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 16),
            ("num_hidden_layers", 8),
            ("num_quantizers", 16),
            ("num_semantic_quantizers", 1),
            ("sliding_window", 72),
        ],
    )?;
    flags(decoder, &[("attention_bias", false)])?;
    ensure!(
        decoder["hidden_act"] == "silu"
            && decoder["rms_norm_eps"].as_f64() == Some(1e-5)
            && decoder["rope_theta"].as_f64() == Some(10000.0)
            && decoder["upsample_rates"] == serde_json::json!([8, 5, 4, 3])
            && decoder["upsampling_ratios"] == serde_json::json!([2, 2]),
        "unsupported external codec decoder operations/rates"
    );
    let encoder = &value["encoder_config"];
    integers(
        encoder,
        &[
            ("sampling_rate", 24000),
            ("audio_channels", 1),
            ("codebook_dim", 256),
            ("codebook_size", 2048),
            ("compress", 2),
            ("dilation_growth_rate", 2),
            ("head_dim", 64),
            ("hidden_size", 512),
            ("intermediate_size", 2048),
            ("kernel_size", 7),
            ("last_kernel_size", 3),
            ("num_attention_heads", 8),
            ("num_key_value_heads", 8),
            ("num_filters", 64),
            ("num_hidden_layers", 8),
            ("num_residual_layers", 1),
            ("num_semantic_quantizers", 1),
            ("residual_kernel_size", 3),
        ],
    )?;
    // The native encoder consumes only semantic0 and acoustic0..14, matching
    // encoder_valid_num_quantizers. Additional trained banks are not required.
    ensure!(
        encoder["num_quantizers"].as_u64().is_some_and(|n| n >= 16),
        "external encoder has fewer than sixteen quantizers"
    );
    flags(
        encoder,
        &[
            ("attention_bias", false),
            ("normalize", false),
            ("use_causal_conv", true),
            ("use_conv_shortcut", false),
            ("use_streaming", false),
        ],
    )?;
    ensure!(
        encoder["hidden_act"] == "gelu"
            && encoder["pad_mode"] == "constant"
            && encoder["norm_eps"].as_f64() == Some(1e-5)
            && encoder["rope_theta"].as_f64() == Some(10000.0)
            && encoder["upsampling_ratios"] == serde_json::json!([8, 6, 5, 4]),
        "unsupported external codec encoder operations/rates"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::json!({
            "model_type":"qwen3_tts_tokenizer_12hz","input_sample_rate":24000,
            "output_sample_rate":24000,"encode_downsample_rate":1920,"decode_upsample_rate":1920,
            "encoder_valid_num_quantizers":16,
            "decoder_config":{"latent_dim":1024,"codebook_dim":512,"codebook_size":2048,
                "decoder_dim":1536,"hidden_size":512,"intermediate_size":1024,"head_dim":64,
                "num_attention_heads":16,"num_key_value_heads":16,"num_hidden_layers":8,
                "num_quantizers":16,"num_semantic_quantizers":1,"sliding_window":72,
                "attention_bias":false,"hidden_act":"silu","rms_norm_eps":1e-5,"rope_theta":10000,
                "upsample_rates":[8,5,4,3],"upsampling_ratios":[2,2]},
            "encoder_config":{"sampling_rate":24000,"audio_channels":1,"codebook_dim":256,
                "codebook_size":2048,"compress":2,"dilation_growth_rate":2,"head_dim":64,
                "hidden_size":512,"intermediate_size":2048,"kernel_size":7,"last_kernel_size":3,
                "num_attention_heads":8,"num_key_value_heads":8,"num_filters":64,"num_hidden_layers":8,
                "num_residual_layers":1,"num_semantic_quantizers":1,"residual_kernel_size":3,
                "num_quantizers":32,"attention_bias":false,"normalize":false,"use_causal_conv":true,
                "use_conv_shortcut":false,"use_streaming":false,"hidden_act":"gelu",
                "pad_mode":"constant","norm_eps":1e-5,"rope_theta":10000,"upsampling_ratios":[8,6,5,4]}
        })
    }

    #[test]
    fn codec_accepts_extra_annotations_but_not_changed_geometry() {
        let mut config = fixture();
        config["future_annotation"] = serde_json::json!({"speaker":"unrelated"});
        config["decoder_config"]["semantic_codebook_size"] = 4096.into();
        validate_config(&config).unwrap();
        for (path, value) in [
            ("/decode_upsample_rate", serde_json::json!(960)),
            ("/encoder_config/normalize", serde_json::json!(true)),
            ("/decoder_config/num_attention_heads", serde_json::json!(8)),
            ("/encoder_config/pad_mode", serde_json::json!("reflect")),
        ] {
            let mut incompatible = config.clone();
            *incompatible.pointer_mut(path).unwrap() = value;
            assert!(validate_config(&incompatible).is_err(), "{path}");
        }
    }
}
