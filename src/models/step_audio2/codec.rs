use anyhow::{Result, ensure};

pub const BOT: u32 = 151666;
pub const EOT: u32 = 151665;
pub const TTS_START: u32 = 151693;
pub const AUDIO_START: u32 = 151696;
pub const SPEECH_CODES: u32 = 6561;
const TEXT_VOCAB: u32 = 151643;

/// The complete serialized tokenizer preserves added-token flags and NFC.
/// Mary's older graph tokenizer reconstruction marks all added tokens special;
/// Mini audio tokens are ordinary normalized added tokens, so that is lossy here.
pub struct MiniCodec {
    tokenizer: tokenizers::Tokenizer,
    vocab_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    Human,
    Assistant,
}

pub struct Message<'a> {
    pub role: Role,
    pub content: &'a str,
    pub end_turn: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    Eos,
    TokenLimit,
    CallerStopped,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Generated {
    pub raw_ids: Vec<u32>,
    pub text_ids: Vec<u32>,
    pub control_ids: Vec<u32>,
    pub speech_codes: Vec<u32>,
    /// Original model IDs, not valid decoder codes.
    pub audio_padding_ids: Vec<u32>,
    pub termination: Termination,
}

impl MiniCodec {
    pub fn from_bytes(bytes: &[u8], vocab_size: usize) -> Result<Self> {
        let vocab_size = u32::try_from(vocab_size)?;
        ensure!(
            vocab_size > AUDIO_START + SPEECH_CODES,
            "Mini vocabulary is too small"
        );
        let tokenizer = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("decode stored tokenizer: {e}"))?;
        // Upstream tokenizes each message as a batch of one with padding=True.
        // Its serialized BatchLongest padding is therefore a no-op. Fixed or
        // multiple-of padding would change the prompt; truncation would lose it.
        ensure!(
            tokenizer.get_padding().is_none_or(|p| {
                matches!(p.strategy, tokenizers::PaddingStrategy::BatchLongest)
                    && p.pad_to_multiple_of.is_none()
            }) && tokenizer.get_truncation().is_none(),
            "Mini codec requires untruncated single-message tokenization without added padding"
        );
        for (spelling, id) in [
            ("<|BOT|>", BOT),
            ("<|EOT|>", EOT),
            ("<tts_start>", TTS_START),
            ("<tts_end>", 151694),
            ("<tts_pad>", 151695),
        ] {
            ensure!(
                tokenizer.token_to_id(spelling) == Some(id),
                "wrong Mini token identity for {spelling}"
            );
            let ids = tokenizer
                .encode(spelling, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            ensure!(ids.get_ids() == [id], "{spelling} is not one atomic token");
        }
        for code in 0..=SPEECH_CODES {
            let spelling = format!("<audio_{code}>");
            ensure!(
                tokenizer.token_to_id(&spelling) == Some(AUDIO_START + code),
                "wrong Mini audio token identity for {spelling}"
            );
        }
        Ok(Self {
            tokenizer,
            vocab_size,
        })
    }

    /// Match stepaudio2.py message boundaries: tokenize each message separately.
    /// Literal control spellings retain upstream tokenizer behavior.
    pub fn encode_messages(&self, messages: &[Message<'_>]) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for message in messages {
            let role = match message.role {
                Role::System => "system",
                Role::Human => "human",
                Role::Assistant => "assistant",
            };
            let end = if message.end_turn { "<|EOT|>" } else { "" };
            let framed = format!("<|BOT|>{role}\n{}{end}", message.content);
            let encoded = self
                .tokenizer
                .encode(framed, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            ensure!(
                encoded.get_ids().iter().all(|&id| id < self.vocab_size),
                "prompt token outside model vocabulary"
            );
            ids.extend_from_slice(encoded.get_ids());
        }
        Ok(ids)
    }

    /// Mini conversational speech framing; this does not promise literal reading.
    pub fn speech_prompt(&self, system: &str, user_text: &str) -> Result<Vec<u32>> {
        self.encode_messages(&[
            Message {
                role: Role::System,
                content: system,
                end_turn: true,
            },
            Message {
                role: Role::Human,
                content: user_text,
                end_turn: true,
            },
            Message {
                role: Role::Assistant,
                content: "<tts_start>",
                end_turn: false,
            },
        ])
    }

    /// Preserve every generated ID, including a valid final token at the cap.
    /// Pre-audio added/control IDs are separated from ordinary text pieces.
    pub fn split_generated(&self, ids: &[u32], termination: Termination) -> Result<Generated> {
        ensure!(
            ids.iter().all(|&id| id < self.vocab_size),
            "generated ID outside model vocabulary"
        );
        ensure!(
            termination != Termination::Eos || ids.last() == Some(&EOT),
            "EOS termination without Mini EOT"
        );
        let mut out = Generated {
            raw_ids: ids.to_vec(),
            text_ids: Vec::new(),
            control_ids: Vec::new(),
            speech_codes: Vec::new(),
            audio_padding_ids: Vec::new(),
            termination,
        };
        for &id in ids {
            if id < TEXT_VOCAB {
                out.text_ids.push(id);
            } else if id < AUDIO_START {
                out.control_ids.push(id);
            } else if id - AUDIO_START < SPEECH_CODES {
                out.speech_codes.push(id - AUDIO_START);
            } else {
                out.audio_padding_ids.push(id);
            }
        }
        Ok(out)
    }

    pub fn decode_text(&self, ids: &[u32]) -> Result<String> {
        self.decode_tokens(ids, false)
    }

    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}
