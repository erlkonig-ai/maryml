//! `mary::hear` — the production ears seam: Voxtral-Mini-4B-Realtime
//! streaming speech-to-text, loaded whole from one native model pile (the
//! exact/f16 weight cohort AND the Tekken tokenizer, see the `voxtral_persist`
//! bin), running the folded f16 realtime lane on the hearing backend
//! ([`crate::nn::backend::hear`]: CUDA under `voxtral-cuda`, the wgpu lane
//! otherwise).
//!
//! One [`Ears`] per process holds the model; each utterance or stream is a
//! [`Listening`] borrowed from it. Feed 16 kHz mono f32 samples with
//! [`Listening::push`] and read back the text that became final; text arrives
//! `delay_ms` behind the audio (the model's conditioning delay, a multiple of
//! 80 ms) plus compute. [`Listening::finish`] streams the trailing silence the
//! delayed tokens need and returns the rest of the text.
//!
//! The model and its GPU state are not shared across threads here: keep
//! `Ears` and its `Listening`s on one thread and send samples to it.

use std::path::Path;

use crate::models::voxtral::config::{
    N_FFT, N_LEFT_PAD_TOKENS, OFFLINE_BUFFER_TOKENS, SAMPLES_PER_TOK, delay_tokens,
};
use crate::models::voxtral::fast::RealtimeTranscriber;
use crate::models::voxtral::pipeline::StreamingTranscriber;
use crate::models::voxtral::tokenizer::Tekken;
use crate::nn::backend::hear;

/// The backend the ears run on: the folded layout on fusion f16.
pub type Backend = hear::FusedHalf;

/// Decoder positions one [`Listening`] may use: the RoPE tables are built for
/// this many, and the silence prefix, the flush tail and the audio share them.
/// 8192 positions of 80 ms leave about 10 minutes 50 seconds of audio per
/// `Listening`; start a new one (for example at a pause) to go on.
pub const MAX_TOKENS: usize = 8192;

/// The resident ears: the realtime transcriber, weights on the GPU.
pub struct Ears {
    stt: RealtimeTranscriber<Backend>,
}

impl Ears {
    /// Load the Voxtral cohort and its tokenizer from the native pile at
    /// `pile` onto the default device of [`Backend`].
    pub fn load(pile: &Path) -> anyhow::Result<Self> {
        let snapshot = crate::model_collection::load_model_collection_local_latest(pile)?;
        let tekken = Tekken::from_snapshot(&snapshot)?;
        let loader = crate::models::voxtral::VoxtralWeights::from_snapshot(snapshot)?.into_loader();
        let device = hear::Device::default();
        let stt = RealtimeTranscriber::load(&loader, tekken, MAX_TOKENS, &device);
        Ok(Self { stt })
    }

    /// Start a stream with text delayed `delay_ms` behind the audio (a
    /// multiple of 80 between 80 and 1200, or 2400; 480 is the model's
    /// default).
    pub fn listen(&self, delay_ms: usize) -> Listening<'_> {
        let n_delay = delay_tokens(delay_ms);
        let tail = (n_delay + 1 + OFFLINE_BUFFER_TOKENS) * SAMPLES_PER_TOK + SAMPLES_PER_TOK;
        let budget = (MAX_TOKENS - N_LEFT_PAD_TOKENS - 1) * SAMPLES_PER_TOK - tail - N_FFT;
        Listening {
            stream: StreamingTranscriber::new(&self.stt, delay_ms),
            tekken: &self.stt.tekken,
            pending: Vec::new(),
            accepted: 0,
            budget,
            n_delay,
        }
    }
}

/// One stream of speech being turned into text.
pub struct Listening<'a> {
    stream: StreamingTranscriber<'a, Backend, RealtimeTranscriber<Backend>>,
    tekken: &'a Tekken,
    /// Token bytes that do not yet end on a UTF-8 character boundary.
    pending: Vec<u8>,
    accepted: usize,
    /// Samples this stream accepts before its positions run out.
    budget: usize,
    n_delay: usize,
}

impl Listening<'_> {
    /// Feed 16 kHz mono samples; returns the text that became final. Samples
    /// past the position budget are dropped and [`Self::is_full`] turns true.
    pub fn push(&mut self, samples: &[f32]) -> String {
        let room = self.budget - self.accepted;
        let samples = &samples[..samples.len().min(room)];
        self.accepted += samples.len();
        self.feed(samples)
    }

    /// Stream the trailing silence the delayed tokens need (alignment to a
    /// token boundary, the delay, and the same buffer the offline path pads
    /// with) and return the remaining text, including any incomplete
    /// character as replacement text.
    pub fn finish(mut self) -> String {
        let align = (SAMPLES_PER_TOK - self.accepted % SAMPLES_PER_TOK) % SAMPLES_PER_TOK;
        let tail = align + (self.n_delay + 1 + OFFLINE_BUFFER_TOKENS) * SAMPLES_PER_TOK + N_FFT / 2;
        let mut text = self.feed(&vec![0.0; tail]);
        text.push_str(&String::from_utf8_lossy(&self.pending));
        text
    }

    /// The model emitted end-of-stream; further pushes return nothing.
    pub fn is_finished(&self) -> bool {
        self.stream.is_finished()
    }

    /// The position budget is spent: [`Self::finish`] this one and start a
    /// new [`Ears::listen`] to keep listening.
    pub fn is_full(&self) -> bool {
        self.accepted == self.budget
    }

    /// Everything transcribed so far.
    pub fn transcript(&self) -> String {
        self.stream.text()
    }

    fn feed(&mut self, samples: &[f32]) -> String {
        for token in self.stream.push(samples) {
            self.pending.extend_from_slice(self.tekken.piece(token.id));
        }
        take_complete_utf8(&mut self.pending)
    }
}

/// Remove and return the longest prefix of `pending` that ends on a character
/// boundary; an incomplete trailing sequence stays for the next token's bytes.
/// Bytes that can never become valid UTF-8 become U+FFFD.
fn take_complete_utf8(pending: &mut Vec<u8>) -> String {
    let mut text = String::new();
    loop {
        match std::str::from_utf8(pending) {
            Ok(valid) => {
                text.push_str(valid);
                pending.clear();
                return text;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                text.push_str(std::str::from_utf8(&pending[..valid]).expect("valid prefix"));
                match error.error_len() {
                    None => {
                        pending.drain(..valid);
                        return text;
                    }
                    Some(invalid) => {
                        text.push(char::REPLACEMENT_CHARACTER);
                        pending.drain(..valid + invalid);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_character_split_across_tokens_waits_for_its_last_byte() {
        let mut pending = b"a\xc3".to_vec();
        assert_eq!(take_complete_utf8(&mut pending), "a");
        assert_eq!(pending, b"\xc3");
        pending.extend_from_slice(b"\xa4 b");
        assert_eq!(take_complete_utf8(&mut pending), "\u{e4} b");
        assert!(pending.is_empty());
    }

    #[test]
    fn bytes_that_can_never_be_utf8_become_replacement_characters() {
        let mut pending = b"x\xffy\xe2\x82".to_vec();
        assert_eq!(take_complete_utf8(&mut pending), "x\u{fffd}y");
        assert_eq!(
            pending, b"\xe2\x82",
            "an incomplete euro sign keeps waiting"
        );
    }

    /// Opt-in end-to-end gate on a real pile and GPU: set
    /// `MARY_VOXTRAL_NATIVE_PILE`, `MARY_HEAR_WAV` (16 kHz mono PCM16) and
    /// `MARY_HEAR_EXPECT` (the clip's exact words). Every expected word must
    /// come back, in order.
    #[test]
    fn configured_clip_is_heard_word_for_word() {
        let (Ok(pile), Ok(wav), Ok(expect)) = (
            std::env::var("MARY_VOXTRAL_NATIVE_PILE"),
            std::env::var("MARY_HEAR_WAV"),
            std::env::var("MARY_HEAR_EXPECT"),
        ) else {
            return;
        };
        let (audio, rate) = crate::models::f5::wav::read_pcm16_mono(Path::new(&wav));
        assert_eq!(rate, 16000);
        let expect = std::fs::read_to_string(expect).unwrap();

        let ears = Ears::load(Path::new(&pile)).unwrap();
        let mut listening = ears.listen(480);
        let mut text = String::new();
        for chunk in audio.chunks(SAMPLES_PER_TOK) {
            text += &listening.push(chunk);
        }
        text += &listening.finish();
        eprintln!("heard: {text:?}");

        let words = |s: &str| -> Vec<String> {
            s.split_whitespace()
                .map(|w| {
                    w.chars()
                        .filter(|c| c.is_alphanumeric())
                        .collect::<String>()
                })
                .filter(|w| !w.is_empty())
                .map(|w| w.to_lowercase())
                .collect()
        };
        assert_eq!(words(&text), words(&expect));
    }
}
