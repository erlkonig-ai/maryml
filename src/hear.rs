//! `mary::hear` — the production ears seam: Voxtral-Mini-4B-Realtime
//! streaming speech-to-text, loaded whole from one native model pile (the
//! exact/f16 weight cohort AND the Tekken tokenizer, see the `voxtral_persist`
//! bin), running the folded f16 realtime layout on the hearing backend
//! ([`crate::nn::backend::hear`]: raw CUDA under `voxtral-cuda`, the fused
//! wgpu lane otherwise; see [`Backend`]).
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
use crate::models::voxtral::fast::{FastCaches, RealtimeTranscriber};
use crate::models::voxtral::pipeline::{EncoderPrefix, StreamingTranscriber};
use crate::models::voxtral::tokenizer::Tekken;
use crate::nn::backend::hear;

/// The backend the ears run on, always the folded f16 layout. On CUDA it is
/// the raw (unfused) backend: on sky (GB10) it measured p50 263-267 / p95
/// 281-284 ms of compute per 80 ms frame against 282 / 302-309 for the fusion
/// backend (voxtral_listen `--lane rawhalf` vs `--lane half`, warm passes, same
/// clip). Elsewhere it stays the fusion backend, the lane the Mac measured
/// realtime on.
#[cfg(feature = "voxtral-cuda")]
pub type Backend = hear::RawHalf;
#[cfg(not(feature = "voxtral-cuda"))]
pub type Backend = hear::FusedHalf;

/// Decoder positions one [`Listening`] may use: the RoPE tables are built for
/// this many, and the silence prefix, the flush tail and the audio share them.
/// 8192 positions of 80 ms leave about 10 minutes 50 seconds of audio per
/// `Listening`; start a new one (for example at a pause) to go on.
pub const MAX_TOKENS: usize = 8192;

/// Silence streamed once by [`Ears::load`] to compile the stream's kernels:
/// two seconds at 16 kHz.
pub const WARM_UP_SAMPLES: usize = 32_000;

/// The resident ears: model plus its immutable 31-step encoder silence prefix.
/// Both stay on the same owner thread/device; no seed is reused across reloads.
pub struct Ears {
    stt: RealtimeTranscriber<Backend>,
    prefix: EncoderPrefix<Backend, FastCaches<Backend>>,
}

impl Ears {
    /// Load the Voxtral cohort and its tokenizer from the native pile at
    /// `pile` onto the default device of [`Backend`], then stream
    /// [`WARM_UP_SAMPLES`] of silence through a throwaway [`Listening`] at
    /// the default 480 ms delay, so the kernels a stream compiles on first
    /// use (the prefill above all) are compiled here rather than inside the
    /// caller's first utterance.
    pub fn load(pile: &Path) -> anyhow::Result<Self> {
        let snapshot = crate::model_collection::load_model_collection_local_latest(pile)?;
        let tekken = Tekken::from_snapshot(&snapshot)?;
        let loader = crate::models::voxtral::VoxtralWeights::from_snapshot(snapshot)?.into_loader();
        let device = hear::Device::default();
        let stt = RealtimeTranscriber::load(&loader, tekken, MAX_TOKENS, &device);
        let prefix = {
            let mut stream = StreamingTranscriber::new(&stt, 480);
            let prefix = stream.prepare_encoder_prefix();
            // Continue this stream: stock warmup must not encode the prefix twice.
            let mut warm = Listening::from_stream(stream, &stt.tekken, 480);
            warm.push(&vec![0.0; WARM_UP_SAMPLES]);
            warm.finish();
            prefix
        };
        Ok(Self { stt, prefix })
    }

    /// Start a stream with text delayed `delay_ms` behind the audio (a
    /// multiple of 80 between 80 and 1200, or 2400; 480 is the model's
    /// default).
    pub fn listen(&self, delay_ms: usize) -> Listening<'_> {
        Listening::from_stream(
            StreamingTranscriber::from_encoder_prefix(&self.stt, delay_ms, &self.prefix),
            &self.stt.tekken, delay_ms,
        )
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
        let samples = &samples[..samples.len().min(self.room())];
        self.accepted += samples.len();
        self.feed(samples)
    }

    /// Stream the trailing silence the delayed tokens need (alignment to a
    /// token boundary, the delay, and the same buffer the offline path pads
    /// with) and return the remaining text, including any incomplete
    /// character as replacement text.
    pub fn finish(mut self) -> String {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> String {
        let align = (SAMPLES_PER_TOK - self.accepted % SAMPLES_PER_TOK) % SAMPLES_PER_TOK;
        let tail = align + (self.n_delay + 1 + OFFLINE_BUFFER_TOKENS) * SAMPLES_PER_TOK + N_FFT / 2;
        for token in self.stream.push_finish(&vec![0.0; tail]) {
            self.pending.extend_from_slice(self.tekken.piece(token.id));
        }
        let mut text = take_complete_utf8(&mut self.pending);
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

    /// Samples [`Self::push`] still accepts, so a caller can hand the rest
    /// of its audio to the next stream instead of losing it.
    pub fn room(&self) -> usize {
        self.budget - self.accepted
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

impl<'a> Listening<'a> {
    fn from_stream(
        stream: StreamingTranscriber<'a, Backend, RealtimeTranscriber<Backend>>,
        tekken: &'a Tekken, delay_ms: usize,
    ) -> Self {
        let n_delay = delay_tokens(delay_ms);
        let tail = (n_delay + 1 + OFFLINE_BUFFER_TOKENS) * SAMPLES_PER_TOK + SAMPLES_PER_TOK;
        let budget = (MAX_TOKENS - N_LEFT_PAD_TOKENS - 1) * SAMPLES_PER_TOK - tail - N_FFT;
        Self { stream, tekken, pending: Vec::new(), accepted: 0, budget, n_delay }
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

    #[test]
    #[cfg(all(feature = "voxtral-cuda", feature = "breeze"))]
    #[ignore = "one Ears load, four admitted segments with eager/grouped finish host counters; separately admitted"]
    fn loaded_tail_block_four_segments() -> anyhow::Result<()> {
        use std::{fs::OpenOptions, io::Write, time::Instant};
        use sha2::{Digest, Sha256};
        let pile = std::env::var("MARY_HEARING_TAIL_BLOCK_PILE")?;
        let wav = std::env::var("MARY_HEARING_TAIL_BLOCK_WAV")?;
        let output = std::env::var("MARY_HEARING_TAIL_BLOCK_REPORT")?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(output)?;
        let bytes = std::fs::read(&wav)?;
        let wav_hash = format!("{:x}", Sha256::digest(&bytes));
        anyhow::ensure!(wav_hash == "ff1f2b743163a4039d4dea695d617b2b7fd1073ca67cab164eedf762e4eeacb0",
            "not the admitted WAV");
        let (pcm,sr) = crate::models::f5::wav::read_pcm16_mono(Path::new(&wav));
        anyhow::ensure!(sr==16000,"expected16k PCM");
        let started = Instant::now();
        let ears = Ears::load(Path::new(&pile))?;
        let load_seconds = started.elapsed().as_secs_f64();
        writeln!(file,"{}",serde_json::json!({"kind":"load","pile":pile,
            "wav_sha256":wav_hash,"load_stock_warmup_seconds":load_seconds,
            "scope":"host-only counters; stock warmup also exercises eligible grouped finish"}))?;
        file.flush()?;
        let mut rows = Vec::new();
        for (index,(start,end)) in [(12160,23680),(32960,87680),(96000,146880),(158720,183360)]
            .into_iter().enumerate() {
            let samples = &pcm[start..end];
            let mut hash = Sha256::new();
            for sample in samples { hash.update(sample.to_le_bytes()); }
            let sample_hash = format!("{:x}",hash.finalize());
            for eager in [true,false] {
                let api_start = Instant::now();
                let mut listening = ears.listen(480);
                let mut text = listening.push(samples);
                let push_api_seconds = api_start.elapsed().as_secs_f64();
                let before = listening.stream.tail_block_test_state();
                let accepted = listening.accepted;
                let cap = listening.is_full();
                let align = (SAMPLES_PER_TOK-accepted%SAMPLES_PER_TOK)%SAMPLES_PER_TOK;
                let tail = align+(listening.n_delay+1+OFFLINE_BUFFER_TOKENS)*SAMPLES_PER_TOK+N_FFT/2;
                let finish_start = Instant::now();
                let rest = if eager {
                    // Literal old finish policy, compiled only into this test.
                    let mut rest = listening.feed(&vec![0.0;tail]);
                    rest.push_str(&String::from_utf8_lossy(&listening.pending));
                    rest
                } else {
                    listening.finish_inner()
                };
                let after = listening.stream.tail_block_test_state();
                drop(listening);
                let finish_seconds = finish_start.elapsed().as_secs_f64();
                text.push_str(&rest);
                let row = serde_json::json!({"kind":"finish","segment":index,
                    "policy":if eager {"old_eager"} else {"grouped"},
                    "samples":samples.len(),"start_sample":start,"pcm_f32le_sha256":sample_hash,
                    "accepted":accepted,"cap_before":cap,"delay_ms":480,"padding_budget":tail,
                    "padding_fed":after.samples-before.samples,"encoded_delta":after.encoded-before.encoded,
                    "encoder_calls_delta":after.encoder_calls-before.encoder_calls,
                    "projector_calls_delta":after.projector_calls-before.projector_calls,
                    "grouped_finishes_delta":after.grouped_finishes-before.grouped_finishes,
                    "decoder_steps_delta":after.decoder_steps-before.decoder_steps,
                    "before":before,"after":after,"raw_text":text,
                    "push_api_seconds":push_api_seconds,"finish_seconds":finish_seconds,
                    "api_seconds":push_api_seconds+finish_seconds});
                writeln!(file,"{row}")?;
                file.flush()?;
                println!("TAIL_BLOCK_WITNESS {row}");
                rows.push(row);
            }
        }
        // Keep all raw evidence before route/state assertions; no words/speedup
        // gate or tensor synchronization is hidden in these host-only checks.
        for pair in rows.chunks_exact(2) {
            for (i,row) in pair.iter().enumerate() {
                anyhow::ensure!(row["cap_before"]==false && row["accepted"]==row["samples"],
                    "input cap/acceptance changed");
                anyhow::ensure!(row["padding_fed"]==row["padding_budget"],"padding not fully offered");
                anyhow::ensure!(row["encoded_delta"]==18 && row["projector_calls_delta"]==18,
                    "frontend/projector step count changed");
                anyhow::ensure!(row["encoder_calls_delta"]==if i==0 {18} else {1},
                    "wrong encoder call count");
                anyhow::ensure!(row["grouped_finishes_delta"]==if i==0 {0} else {1},
                    "finish specialization not selected as expected");
                let before = row["before"]["encoder_positions"].as_array().unwrap();
                let after = row["after"]["encoder_positions"].as_array().unwrap();
                anyhow::ensure!(before.len()==32 && after.len()==32,"wrong layer count");
                anyhow::ensure!(before.iter().zip(after).all(|(a,b)|
                    b.as_u64().unwrap()==a.as_u64().unwrap()+72),"encoder positions changed");
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "voxtral-cuda", feature = "breeze"))]
    #[ignore = "one native Ears load, tiny prefix/first40 witnesses and three short finishes; separately admitted"]
    fn loaded_encoder_prefix_boundary_sessions_and_finish() -> anyhow::Result<()> {
        use std::{fs::OpenOptions, time::Instant};
        use crate::models::voxtral::pipeline::PrefixTestState;
        let pile = std::env::var("MARY_HEARING_PREFIX_PILE")?;
        let output = std::env::var("MARY_HEARING_PREFIX_REPORT")?;
        let file = OpenOptions::new().write(true).create_new(true).open(output)?;
        let started = Instant::now();
        let ears = Ears::load(Path::new(&pile))?;
        let loaded_seconds = started.elapsed().as_secs_f64();
        let seed_before = ears.prefix.test_snapshot();
        let state_ok = |state: &PrefixTestState, encoded, positions| {
            state.encoded == encoded && state.queued == encoded && state.tokens == 0
                && state.positions.len() == 32 && state.positions.iter().all(|&p| p == positions)
        };
        let mut zero = ears.listen(480);
        let mut impulse = ears.listen(80);
        let initial_zero = zero.stream.prefix_test_state();
        let initial_impulse = impulse.stream.prefix_test_state();
        let mut zero_text = zero.push(&[]);
        let zero_empty = zero.stream.prefix_test_state();
        zero_text += &zero.push(&[0.0; 39]);
        let zero_39 = zero.stream.prefix_test_state();
        zero_text += &zero.push(&[0.0]);
        let zero_40 = zero.stream.prefix_test_state();
        // The other branch must still be at the seed's private absolute pos.
        let other_after_zero = impulse.stream.prefix_test_state();
        let mut first = [0.0; 39];
        first[0] = 1.0;
        let mut impulse_text = impulse.push(&first);
        let impulse_39 = impulse.stream.prefix_test_state();
        impulse_text += &impulse.push(&[0.0]);
        let impulse_40 = impulse.stream.prefix_test_state();
        let seed_after_branches = ears.prefix.test_snapshot();
        let zero_cap = zero.is_full();
        let impulse_cap = impulse.is_full();
        let zero_finished_before = zero.is_finished();
        let impulse_finished_before = impulse.is_finished();
        let zero_finish_started = Instant::now();
        zero_text += &zero.finish();
        let zero_finish_seconds = zero_finish_started.elapsed().as_secs_f64();
        let impulse_finish_started = Instant::now();
        impulse_text += &impulse.finish();
        let impulse_finish_seconds = impulse_finish_started.elapsed().as_secs_f64();
        let empty_started = Instant::now();
        let empty = ears.listen(480);
        let empty_cap = empty.is_full();
        let empty_text = empty.finish();
        let empty_seconds = empty_started.elapsed().as_secs_f64();
        let seed_after_finishes = ears.prefix.test_snapshot();
        let count = |s: &PrefixTestState| serde_json::json!({
            "encoded":s.encoded,"queued":s.queued,"tokens":s.tokens,"positions":s.positions});
        let impulse_visible = zero_40.last_embedding != impulse_40.last_embedding;
        let result = serde_json::json!({"kind":"loaded_encoder_prefix_boundary_sessions_and_finish",
            "pile":pile,"load_including_prefix_and_stock_warmup_seconds":loaded_seconds,
            "initial_zero":count(&initial_zero),"initial_impulse":count(&initial_impulse),
            "zero_empty":count(&zero_empty),"zero_39":count(&zero_39),"zero_40":count(&zero_40),
            "other_after_zero":count(&other_after_zero),"impulse_39":count(&impulse_39),
            "impulse_40":count(&impulse_40),"first_sample_impulse_visible":impulse_visible,
            "seed_unchanged_after_branches":seed_before==seed_after_branches,
            "seed_unchanged_after_finishes":seed_before==seed_after_finishes,
            "raw_finishes":[
                {"delay_ms":480,"input_samples":40,"text":zero_text,"cap_before_finish":zero_cap,
                    "eos_before_finish":zero_finished_before,"finish_seconds":zero_finish_seconds},
                {"delay_ms":80,"input_samples":40,"text":impulse_text,"cap_before_finish":impulse_cap,
                    "eos_before_finish":impulse_finished_before,"finish_seconds":impulse_finish_seconds},
                {"delay_ms":480,"input_samples":0,"text":empty_text,"cap_before_finish":empty_cap,
                    "request_seconds":empty_seconds}],
            "post_finish_eos":"not exposed by consuming public finish API; not inferred from text",
            "scope":"first/last layer KV 8-value samples and8values per seed embedding; one full first40 projected vector, not model-wide parity"});
        serde_json::to_writer_pretty(file, &result)?;
        println!("ENCODER_PREFIX_WITNESS {result}");
        for state in [&initial_zero,&initial_impulse,&zero_empty,&zero_39,&other_after_zero,&impulse_39] {
            anyhow::ensure!(state_ok(state,31,124), "unexpected reusable prefix boundary: {state:?}");
        }
        anyhow::ensure!(state_ok(&zero_40,32,128) && state_ok(&impulse_40,32,128), "first40 step missing");
        anyhow::ensure!(impulse_visible, "first real impulse hidden by reuse");
        anyhow::ensure!(seed_before==seed_after_branches && seed_before==seed_after_finishes,
            "shared prefix sample changed");
        anyhow::ensure!(!zero_cap && !impulse_cap && !empty_cap, "unexpected short-request cap");
        Ok(())
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

        let started = std::time::Instant::now();
        let ears = Ears::load(Path::new(&pile)).unwrap();
        let loaded = started.elapsed();
        let mut listening = ears.listen(480);
        let mut text = String::new();
        for chunk in audio.chunks(SAMPLES_PER_TOK) {
            text += &listening.push(chunk);
        }
        text += &listening.finish();
        eprintln!(
            "heard: {text:?}\nload incl. warm-up {:.1} s; {:.2} s of audio heard in {:.1} s",
            loaded.as_secs_f64(),
            audio.len() as f64 / 16000.0,
            (started.elapsed() - loaded).as_secs_f64()
        );

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
