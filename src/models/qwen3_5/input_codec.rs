//! WeMM's pinned, bounded single-user template form, not a Jinja interpreter.
//! Tokenizer postprocessing remains the checkpoint's own executor: no manual
//! embedding-token append, padding or truncation. Other template bytes fail.
use super::{multimodal_layout::ImagePlan, prepared, vision_geometry::Grid};
use sha2::{Digest, Sha256};

pub const TOKENIZER_SHA256: &str =
    "40e444c744512f423da4c8443c47c21e22ff76056ba4e9796a81c04c13a9daf0";
pub const TEMPLATE_SHA256: &str =
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80";
pub const GRID: Grid = Grid {
    frames: 1,
    height: 16,
    width: 16,
};

/// B1, unpadded, at most 256 IDs, with exactly one terminal embedding token.
pub struct Tokens {
    ids: Vec<u32>,
}
impl Tokens {
    pub fn as_slice(&self) -> &[u32] {
        &self.ids
    }
}

pub struct InputCodec {
    tokenizer: tokenizers::Tokenizer,
}
impl InputCodec {
    /// Assets may be exact-fetched typed blobs; no file/model discovery here.
    /// The tokenizer JSON retains the postprocessor that the older tokenizer
    /// graph reader explicitly does not reconstruct. No graph fallback.
    pub fn from_assets(tokenizer_json: &[u8], chat_template: &[u8]) -> Result<Self, String> {
        for (label, bytes, want) in [
            ("tokenizer", tokenizer_json, TOKENIZER_SHA256),
            ("template", chat_template, TEMPLATE_SHA256),
        ] {
            if format!("{:x}", Sha256::digest(bytes)) != want {
                return Err(format!("unsupported WeMM {label} asset bytes"));
            }
        }
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(tokenizer_json).map_err(|e| e.to_string())?;
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
            return Err("pinned tokenizer unexpectedly enables padding/truncation".into());
        }
        Ok(Self { tokenizer })
    }

    /// Supported template: one user text message, no generation prompt/tools.
    /// Trim semantics match this exact template's render_content|trim branch.
    pub fn text(&self, text: &str) -> Result<Tokens, String> {
        if text.len() > 1024 * 1024 {
            return Err("bounded WeMM text exceeds1MiB; nothing truncated".into());
        }
        // Jinja's trim uses Python's whitespace predicate, which also includes
        // the four ASCII separators not in Rust's Unicode White_Space property.
        let content =
            text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c));
        if content.starts_with("<tool_response>") && content.ends_with("</tool_response>") {
            return Err(
                "the pinned template rejects a tool response as its only user query".into(),
            );
        }
        let ids = self.encode(content)?;
        prepared::validate_ids(&ids)?;
        Ok(Tokens { ids })
    }

    /// One still image, no caption, exactly 64 merged feature placeholders.
    /// The 256-patch budget is explicit; it is not upstream arbitrary-resolution
    /// image processing, video, multi-turn chat, or batching.
    pub fn image(&self) -> Result<Tokens, String> {
        let text = format!(
            "<|vision_start|>{}<|vision_end|>",
            "<|image_pad|>".repeat(64)
        );
        let ids = self.encode(&text)?;
        ImagePlan::new(&ids, GRID, 256, 64)?;
        Ok(Tokens { ids })
    }

    fn encode(&self, content: &str) -> Result<Vec<u32>, String> {
        let framed = format!("<|im_start|>user\n{content}<|im_end|>\n");
        let encoded = self
            .tokenizer
            .encode(framed, true)
            .map_err(|e| e.to_string())?;
        let ids = encoded.get_ids();
        if !(1..=256).contains(&ids.len())
            || ids.last() != Some(&248077)
            || ids.iter().filter(|&&i| i == 248077).count() != 1
        {
            return Err("WeMM requires<=256 unpadded IDs and one terminal embedding token; nothing truncated".into());
        }
        Ok(ids.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_unpinned_assets() {
        assert!(InputCodec::from_assets(b"{}", b"template").is_err());
    }
    #[test]
    #[ignore = "requires exact local checkpoint assets, metadata only; no GPU"]
    fn pinned_tokens_match_the_frozen_behavior_fixture() {
        let root = std::path::PathBuf::from(
            std::env::var("WEMM_CHECKPOINT_DIR").expect("checkpoint directory"),
        );
        let codec = InputCodec::from_assets(
            &std::fs::read(root.join("tokenizer.json")).unwrap(),
            &std::fs::read(root.join("chat_template.jinja")).unwrap(),
        )
        .unwrap();
        let fixture = super::super::image_prepare::tests::fixture();
        for item in fixture["items"].as_array().unwrap() {
            let tokens = if item["modality"] == "text" {
                codec.text(item["text"].as_str().unwrap()).unwrap()
            } else {
                codec.image().unwrap()
            };
            let expected: Vec<u32> = serde_json::from_value(item["ids"].clone()).unwrap();
            assert_eq!(tokens.as_slice(), expected, "{}", item["id"]);
        }
        assert!(codec.text(&"x ".repeat(1024)).is_err());
        assert!(codec.text("<embedding>").is_err());
        assert!(codec.text("<tool_response>x</tool_response>").is_err());
        assert_eq!(
            codec.text("\u{1c} hello \u{1f}").unwrap().as_slice(),
            codec.text("hello").unwrap().as_slice()
        );
    }
}
