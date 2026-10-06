//! Official ref_clone_tata/ref_edit_tata framing, using the exact pile tokenizer.
//! Each text segment has its own BOS. This is not a Qwen chat/clone prompt.
use anyhow::{Result, ensure};
use tokenizers::Tokenizer;

use super::generator::PromptSegment;

pub const BOS: u32 = 2;
pub const AUDIO: u32 = 262144;
pub const AUDIO_EOS: u32 = 262145;
pub const SPEAKER_ZERO: u32 = 262146;
pub const INS_BOS: u32 = 262156;
pub const INS_EOS: u32 = 262157;

const CONTROLS: [(&str, u32); 8] = [
    ("<pad>", 0),
    ("<eos>", 1),
    ("<bos>", BOS),
    ("<|AUDIO|>", AUDIO),
    ("<|audio_eos|>", AUDIO_EOS),
    ("[S0]", SPEAKER_ZERO),
    ("<ins_bos>", INS_BOS),
    ("<ins_eos>", INS_EOS),
];

/// Paired instruction guidance removes only the instruction from the negative
/// branch. Both branches retain the same reference transcript/audio and target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuidedPrompt {
    pub conditional: Vec<PromptSegment>,
    pub negative: Option<Vec<PromptSegment>>,
}

pub struct BreezeTokenizer {
    tokenizer: Tokenizer,
}

impl BreezeTokenizer {
    /// Keep the complete serialized tokenizer, including its BPE decoder,
    /// normalizer, postprocessor and all added-token flags. No external assets.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| anyhow::anyhow!("decode selected Breeze tokenizer: {error}"))?;
        ensure!(
            tokenizer.get_truncation().is_none() && tokenizer.get_padding().is_none(),
            "Breeze prompt requires untruncated, unpadded single-request tokenization"
        );
        let this = Self { tokenizer };
        for (spelling, id) in CONTROLS {
            ensure!(
                this.tokenizer.token_to_id(spelling) == Some(id),
                "wrong Breeze control identity: {spelling}"
            );
            ensure!(
                this.encode(spelling, false)? == [id],
                "Breeze control must be atomic: {spelling}"
            );
        }
        ensure!(
            this.encode("", true)? == [BOS],
            "Breeze text postprocessor must add BOS and no EOS"
        );
        Ok(this)
    }

    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>> {
        let encoded = self
            .tokenizer
            .encode(text, special)
            .map_err(|error| anyhow::anyhow!("tokenize Breeze prompt: {error}"))?;
        ensure!(
            encoded.get_ids().iter().all(|&id| id < 262158),
            "Breeze prompt token outside text vocabulary"
        );
        Ok(encoded.get_ids().to_vec())
    }

    fn render_text(&self, text: &str) -> Result<(String, Vec<u32>)> {
        let ids = self.encode(text, true)?;
        let rendered = self
            .tokenizer
            .decode(&ids, false)
            .map_err(|error| anyhow::anyhow!("render Breeze text segment: {error}"))?;
        let ids = self.encode(&rendered, false)?;
        ensure!(
            ids.first() == Some(&BOS) && ids.iter().filter(|&&id| id == BOS).count() == 1,
            "each Breeze text segment must retain exactly one leading BOS"
        );
        ensure!(
            !ids.contains(&AUDIO) && !ids.contains(&AUDIO_EOS),
            "text segment collides with audio placeholders"
        );
        Ok((rendered, ids))
    }

    /// Pinned templates.py: ref_clone_tata has no negative builder; ref_edit_tata
    /// uses ref_clone_tata as its negative, not an empty/text-only prompt.
    pub fn reference_prompts(
        &self,
        reference_text: &str,
        reference_codes: &[[u16; 16]],
        text: &str,
        direction: Option<&str>,
        cfg_scale: f32,
    ) -> Result<GuidedPrompt> {
        ensure!(
            cfg_scale.is_finite() && cfg_scale > 0.0,
            "CFG scale must be finite and positive"
        );
        let direction = direction.filter(|value| !value.trim().is_empty());
        ensure!(
            cfg_scale == 1.0 || direction.is_some(),
            "neutral ref_clone_tata has no negative prompt; use CFG1"
        );
        Ok(GuidedPrompt {
            conditional: self.reference_branch(reference_text, reference_codes, text, direction)?,
            negative: if cfg_scale != 1.0 {
                Some(self.reference_branch(reference_text, reference_codes, text, None)?)
            } else {
                None
            },
        })
    }

    fn reference_branch(
        &self,
        reference_text: &str,
        reference_codes: &[[u16; 16]],
        text: &str,
        direction: Option<&str>,
    ) -> Result<Vec<PromptSegment>> {
        validate_literal(reference_text)?;
        validate_literal(text)?;
        ensure!(
            !reference_codes.is_empty() && reference_codes.len() <= 375,
            "reference must contain 1..=375 codec frames"
        );
        ensure!(
            reference_codes.iter().flatten().all(|&code| code < 2048),
            "reference contains non-speech codec values"
        );
        let target = match direction.filter(|value| !value.trim().is_empty()) {
            Some(direction) => {
                validate_literal(direction)?;
                format!("[S0]<ins_bos>{direction}<ins_eos>{text}")
            }
            None => format!("[S0]{text}"),
        };
        let (reference_rendered, reference_ids) =
            self.render_text(&format!("[S0]{reference_text}"))?;
        let (target_rendered, target_ids) = self.render_text(&target)?;
        let audio_rendered = format!("{}<|audio_eos|>", "<|AUDIO|>".repeat(reference_codes.len()));
        // Upstream re-tokenizes the concatenated rendered segments. Check that
        // their boundaries survive before handing typed segments to generation.
        let combined = self.encode(
            &format!("{reference_rendered}{audio_rendered}{target_rendered}"),
            false,
        )?;
        let mut expected = reference_ids.clone();
        expected.extend(std::iter::repeat_n(AUDIO, reference_codes.len()));
        expected.push(AUDIO_EOS);
        expected.extend_from_slice(&target_ids);
        ensure!(
            combined == expected,
            "Breeze tokenizer changed a rendered segment boundary"
        );
        Ok(vec![
            PromptSegment::Text(reference_ids),
            PromptSegment::AudioFrames(reference_codes.to_vec()),
            PromptSegment::AudioEos,
            PromptSegment::Text(target_ids),
        ])
    }
}

fn validate_literal(text: &str) -> Result<()> {
    ensure!(
        !text.trim().is_empty() && text.len() <= 65536,
        "Breeze text/direction must contain 1..=65536 UTF-8 bytes"
    );
    for (spelling, _) in CONTROLS {
        ensure!(
            !text.contains(spelling),
            "literal request collides with prompt control {spelling}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // A transient tiny tokenizer fixture proves framing, not model numerics.
    // The importer additionally tests the exact selected tokenizer cold reopen.
    fn fixture() -> Vec<u8> {
        let mut vocab = serde_json::Map::new();
        for (spelling, id) in [("[UNK]", 3), ("reference", 4), ("target", 5), ("softly", 6)] {
            vocab.insert(spelling.into(), id.into());
        }
        let mut added = Vec::new();
        for (spelling, id) in CONTROLS {
            vocab.insert(spelling.into(), id.into());
            added.push(
                serde_json::json!({"id":id,"content":spelling,"single_word":false,
                "lstrip":false,"rstrip":false,"normalized":false,"special":true}),
            );
        }
        serde_json::to_vec(&serde_json::json!({
            "version":"1.0","truncation":null,"padding":null,"added_tokens":added,
            "normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"decoder":null,
            "post_processor":{"type":"TemplateProcessing",
                "single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
                "pair":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
                "special_tokens":{"<bos>":{"id":"<bos>","ids":[2],"tokens":["<bos>"]}}},
            "model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}
        })).unwrap()
    }

    fn texts(segments: &[PromptSegment]) -> Vec<&[u32]> {
        segments
            .iter()
            .filter_map(|segment| match segment {
                PromptSegment::Text(ids) => Some(ids.as_slice()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn tata_has_two_text_bos_and_keeps_last_reference_frame() {
        let codec = BreezeTokenizer::from_bytes(&fixture()).unwrap();
        let codes = [[0u16; 16], [2047; 16]];
        let segments = codec
            .reference_prompts("reference", &codes, "target", None, 1.0)
            .unwrap()
            .conditional;
        assert_eq!(segments.len(), 4);
        assert_eq!(
            texts(&segments),
            [vec![BOS, SPEAKER_ZERO, 4], vec![BOS, SPEAKER_ZERO, 5]]
        );
        match &segments[1] {
            PromptSegment::AudioFrames(frames) => assert_eq!(frames, &codes),
            _ => panic!("reference audio segment missing"),
        }
        assert!(matches!(segments[2], PromptSegment::AudioEos));
        // Text masks are true only for the two Text segments: audio EOS is not text.
        let audio_positions = codes.len() + 1;
        assert_eq!(
            texts(&segments).iter().map(|ids| ids.len()).sum::<usize>() + audio_positions,
            9
        );
    }

    #[test]
    fn direction_only_changes_target_and_blank_direction_is_clone() {
        let codec = BreezeTokenizer::from_bytes(&fixture()).unwrap();
        let codes = [[1u16; 16]];
        let plain = codec
            .reference_prompts("reference", &codes, "target", None, 1.0)
            .unwrap()
            .conditional;
        let edit = codec
            .reference_prompts("reference", &codes, "target", Some("softly"), 1.0)
            .unwrap()
            .conditional;
        let blank = codec
            .reference_prompts("reference", &codes, "target", Some(" \n"), 1.0)
            .unwrap()
            .conditional;
        assert_eq!(texts(&plain)[0], texts(&edit)[0]);
        assert_eq!(texts(&edit)[1], [BOS, SPEAKER_ZERO, INS_BOS, 6, INS_EOS, 5]);
        assert_eq!(texts(&plain), texts(&blank));
    }

    #[test]
    fn paired_negative_keeps_reference_and_removes_only_direction() {
        let codec = BreezeTokenizer::from_bytes(&fixture()).unwrap();
        let codes = [[7; 16], [8; 16]];
        let prompt = codec
            .reference_prompts("reference", &codes, "target", Some("softly"), 4.0)
            .unwrap();
        let negative = prompt.negative.unwrap();
        assert_eq!(&prompt.conditional[..3], &negative[..3]);
        assert_eq!(
            texts(&prompt.conditional)[1],
            [BOS, SPEAKER_ZERO, INS_BOS, 6, INS_EOS, 5]
        );
        assert_eq!(texts(&negative)[1], [BOS, SPEAKER_ZERO, 5]);
        assert!(
            codec
                .reference_prompts("reference", &codes, "target", None, 4.0)
                .is_err()
        );
        assert!(
            codec
                .reference_prompts("reference", &codes, "target", Some(" \n"), 4.0)
                .is_err()
        );
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                codec
                    .reference_prompts("reference", &codes, "target", Some("softly"), scale)
                    .is_err()
            );
        }
        let unit = codec
            .reference_prompts("reference", &codes, "target", Some("softly"), 1.0)
            .unwrap();
        assert!(unit.negative.is_none());
    }

    #[test]
    fn tokenizer_cold_reconstruction_retains_special_controls() {
        let serialized = fixture();
        let before = BreezeTokenizer::from_bytes(&serialized).unwrap();
        let after = BreezeTokenizer::from_bytes(&serialized).unwrap();
        let codes = [[7u16; 16]];
        let first = before
            .reference_prompts("reference", &codes, "target", None, 1.0)
            .unwrap()
            .conditional;
        let cold = after
            .reference_prompts("reference", &codes, "target", None, 1.0)
            .unwrap()
            .conditional;
        assert_eq!(texts(&first), texts(&cold));
        let mut bad: serde_json::Value = serde_json::from_slice(&serialized).unwrap();
        bad["post_processor"] = serde_json::Value::Null;
        assert!(BreezeTokenizer::from_bytes(&serde_json::to_vec(&bad).unwrap()).is_err());
    }

    #[test]
    fn literal_control_collisions_and_reference_padding_are_errors() {
        let codec = BreezeTokenizer::from_bytes(&fixture()).unwrap();
        for text in ["", "<|AUDIO|>", "target<|audio_eos|>", "<bos>target"] {
            assert!(
                codec
                    .reference_prompts("reference", &[[0; 16]], text, None, 1.0)
                    .is_err()
            );
        }
        assert!(
            codec
                .reference_prompts("reference", &[[2050; 16]], "target", None, 1.0)
                .is_err()
        );
        assert!(
            codec
                .reference_prompts("reference", &[], "target", None, 1.0)
                .is_err()
        );
    }
}
