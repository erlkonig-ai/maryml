//! The stt' end-to-end pipeline: audio padding + prompt construction
//! (mirroring mistral_common's offline-streaming transcription encoding),
//! and the delay-conditioned autoregressive transcription loop.
//!
//! Schedule (batch=1): every decoder position is `tok_embed(id) + audio_embed`
//! — prompt positions use the fixed `[BOS, PAD×(32+delay)]` ids, generated
//! positions use the previously sampled token. The encoder side advances 4
//! stem positions (= 8 mel frames = 80 ms) per decoder position.

use burn::prelude::*;
use std::time::Instant;

use super::config::*;
use super::decoder::{Decoder, DecoderCaches};
use super::encoder::AudioEncoder;
use super::mel::VoxtralMel;
use super::tokenizer::Tekken;
use crate::nn::weight_loader::WeightLoader;

/// Left/right-pad a 16 kHz clip the way mistral_common does for OFFLINE
/// streaming: 32 tokens of leading silence; trailing pad to a token multiple
/// plus `(delay + 1 + 10)` extra silence tokens.
pub fn pad_audio(audio: &[f32], num_delay_tokens: usize) -> Vec<f32> {
    let left = N_LEFT_PAD_TOKENS * SAMPLES_PER_TOK;
    let align = (SAMPLES_PER_TOK - (audio.len() % SAMPLES_PER_TOK)) % SAMPLES_PER_TOK;
    let right = align + (num_delay_tokens + 1 + OFFLINE_BUFFER_TOKENS) * SAMPLES_PER_TOK;
    let mut out = vec![0f32; left + audio.len() + right];
    out[left..left + audio.len()].copy_from_slice(audio);
    out
}

/// `[BOS] + [STREAMING_PAD] × (32 + delay)`.
pub fn prompt_ids(num_delay_tokens: usize) -> Vec<u32> {
    let mut ids = vec![BOS];
    ids.extend(std::iter::repeat(STREAMING_PAD).take(N_LEFT_PAD_TOKENS + num_delay_tokens));
    ids
}

/// The stage surface the transcription loops run against. Two implementations:
/// [`Transcriber`] — the parity-first op-for-op layout (the trust anchor, what the
/// probe gates against the oracle) — and [`super::fast::RealtimeTranscriber`] — the
/// folded realtime lane (wide fused qkv, norm weights in matmul rows), gated
/// token-identical against this one in f32.
pub trait SttPipeline<B: Backend> {
    type EncCaches;
    type DecCaches;
    fn device(&self) -> &B::Device;
    fn tekken(&self) -> &Tekken;
    fn mel(&self, samples: &[f32], center: bool) -> Tensor<B, 3>;
    fn stem(&self, mel: Tensor<B, 3>) -> Tensor<B, 3>;
    fn new_enc_caches(&self) -> Self::EncCaches;
    fn new_dec_caches(&self) -> Self::DecCaches;
    /// Encoder transformer over the next stem positions (append-only KV).
    fn encode(&self, embeds: Tensor<B, 3>, caches: &mut Self::EncCaches) -> Tensor<B, 3>;
    fn project(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3>;
    fn ada_scales(&self, n_delay: usize) -> AdaScales<B>;
    fn embed(&self, ids: &[u32]) -> Tensor<B, 3>;
    /// One decoder pass (prefill or single step), appending to the caches.
    /// Returns hidden states in whatever form the lane's [`SttPipeline::logits_last`]
    /// expects (raw: final-normed; fast: unnormed residual).
    fn decode_step(
        &self,
        embeds: Tensor<B, 3>,
        ada: &AdaScales<B>,
        caches: &mut Self::DecCaches,
    ) -> Tensor<B, 3>;
    fn logits_last(&self, hidden: Tensor<B, 3>) -> Tensor<B, 1>;
}

use super::decoder::AdaScales;

pub struct Transcriber<B: Backend> {
    pub mel: VoxtralMel<B>,
    pub encoder: AudioEncoder<B>,
    pub decoder: Decoder<B>,
    pub tekken: Tekken,
    device: B::Device,
}

impl<B: Backend> SttPipeline<B> for Transcriber<B> {
    type EncCaches = super::encoder::EncoderCaches<B>;
    type DecCaches = DecoderCaches<B>;
    fn device(&self) -> &B::Device {
        &self.device
    }
    fn tekken(&self) -> &Tekken {
        &self.tekken
    }
    fn mel(&self, samples: &[f32], center: bool) -> Tensor<B, 3> {
        self.mel.forward(samples, center, &self.device)
    }
    fn stem(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.stem(mel)
    }
    fn new_enc_caches(&self) -> Self::EncCaches {
        self.encoder.new_caches()
    }
    fn new_dec_caches(&self) -> Self::DecCaches {
        self.decoder.new_caches()
    }
    fn encode(&self, embeds: Tensor<B, 3>, caches: &mut Self::EncCaches) -> Tensor<B, 3> {
        self.encoder.forward(embeds, caches)
    }
    fn project(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.project(hidden)
    }
    fn ada_scales(&self, n_delay: usize) -> AdaScales<B> {
        self.decoder.ada_scales(n_delay, &self.device)
    }
    fn embed(&self, ids: &[u32]) -> Tensor<B, 3> {
        self.decoder.embed.forward(ids, &self.device)
    }
    fn decode_step(
        &self,
        embeds: Tensor<B, 3>,
        ada: &AdaScales<B>,
        caches: &mut Self::DecCaches,
    ) -> Tensor<B, 3> {
        self.decoder.forward(embeds, ada, caches)
    }
    fn logits_last(&self, hidden: Tensor<B, 3>) -> Tensor<B, 1> {
        self.decoder.logits_last(hidden)
    }
}

/// Per-frame timing (ms) for the honest latency report.
pub struct FrameTiming {
    pub encoder_ms: f32,
    pub decoder_ms: f32,
}

pub struct Transcription {
    /// Full token sequence: prompt + generated (oracle-comparable).
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub text: String,
    pub timings: Vec<FrameTiming>,
}

impl<B: Backend> Transcriber<B> {
    pub fn load(
        loader: &WeightLoader,
        tekken: Tekken,
        max_tokens: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            mel: VoxtralMel::new(device),
            encoder: AudioEncoder::load(loader, max_tokens * DOWNSAMPLE, device),
            decoder: Decoder::load(loader, max_tokens, device),
            tekken,
            device: device.clone(),
        }
    }

    /// Offline transcription of a full 16 kHz clip at the given delay.
    /// `incremental_encoder`: false = one batch encoder pass (the oracle's
    /// own semantic, cheapest for files); true = advance the encoder 4
    /// positions per frame through its KV cache (the streaming path — gated
    /// in the probe). "Identical output" there means the TRANSCRIPT: streaming
    /// must not change what the ears hear. It measured bit-identical because
    /// the KV cache re-derives the same attention over the same prefix, but
    /// that is an observation, not the bar — retiling the encoder is allowed to
    /// move the tensors and must be judged on the transcript
    /// (wiki:f5dcc88988bb28e472e50fa030332adb).
    pub fn transcribe(
        &self,
        audio: &[f32],
        delay_ms: usize,
        incremental_encoder: bool,
    ) -> Transcription {
        transcribe(self, audio, delay_ms, incremental_encoder)
    }
}

/// Offline transcription over any [`SttPipeline`] lane (see [`Transcriber::transcribe`]).
pub fn transcribe<B: Backend, O: SttPipeline<B>>(
    organs: &O,
    audio: &[f32],
    delay_ms: usize,
    incremental_encoder: bool,
) -> Transcription {
    let n_delay = delay_tokens(delay_ms);
    let padded = pad_audio(audio, n_delay);
    let prompt = prompt_ids(n_delay);
    let mel = organs.mel(&padded, true);
    let n_tokens = mel.dims()[2] / MEL_PER_TOK;

    // conv stem over the whole mel (streaming-equivalent: causal convs)
    let stem = organs.stem(mel); // [1, n_tokens*4, 1280]

    // audio embeds: batch (one pass) or incremental (4 positions/step)
    let mut enc_caches = organs.new_enc_caches();
    let audio_embeds: Tensor<B, 3> = if incremental_encoder {
        let mut chunks = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let h = organs.encode(
                stem.clone().narrow(1, t * DOWNSAMPLE, DOWNSAMPLE),
                &mut enc_caches,
            );
            chunks.push(organs.project(h));
        }
        Tensor::cat(chunks, 1)
    } else {
        let h = organs.encode(stem, &mut enc_caches);
        organs.project(h)
    }; // [1, n_tokens, 3072]

    let ada = organs.ada_scales(n_delay);
    let mut caches = organs.new_dec_caches();

    // prefill: prompt ids + aligned audio embeds
    let l = prompt.len();
    let tok = organs.embed(&prompt);
    let embeds = tok + audio_embeds.clone().narrow(1, 0, l);
    let hidden = organs.decode_step(embeds, &ada, &mut caches);
    let mut next = argmax_host::<B>(organs.logits_last(hidden));

    let mut tokens = prompt.clone();
    let mut timings = Vec::new();
    tokens.push(next);

    // decode: one token per remaining audio position
    for pos in l + 1..n_tokens {
        if next == EOS {
            break;
        }
        let t0 = Instant::now();
        let tok = organs.embed(&[next]);
        let embeds = tok + audio_embeds.clone().narrow(1, pos - 1, 1);
        let enc_ms = 0.0; // batch-encoder mode: encoder cost paid up front
        let hidden = organs.decode_step(embeds, &ada, &mut caches);
        next = argmax_host::<B>(organs.logits_last(hidden));
        tokens.push(next);
        timings.push(FrameTiming {
            encoder_ms: enc_ms,
            decoder_ms: t0.elapsed().as_secs_f32() * 1000.0,
        });
    }

    let text = organs.tekken().decode(&tokens);
    Transcription {
        tokens,
        prompt_len: l,
        text,
        timings,
    }
}

/// Argmax with host readback — one sync per frame.
fn argmax_host<B: Backend>(logits: Tensor<B, 1>) -> u32 {
    let idx = logits.argmax(0);
    let data = idx.into_data();
    let id = data.iter::<i64>().next().expect("argmax scalar") as u32;
    id
}

/// A token emitted by the streaming path, with its honest latency: wall time
/// from "the last audio sample this token needed became available" to "the
/// token id was read back from the GPU".
pub struct StreamedToken {
    pub id: u32,
    pub latency_ms: f32,
    /// Position in the full sequence (prompt included).
    pub pos: usize,
    /// Wall time spent producing this token's audio embed (mel + stem +
    /// encoder step + projector — host submission; the GPU drain lands in
    /// `dec_ms`'s sync). The first emission carries the whole prompt's worth.
    pub enc_ms: f32,
    /// Wall time of the decoder step through the argmax readback (the
    /// per-frame GPU sync). First emission = prefill.
    pub dec_ms: f32,
}

/// Incremental (online) transcription: push 16 kHz samples in, get tokens
/// out at the conditioned delay. The 32-token silence prefix is synthetic and
/// pre-loaded; every stage is chunk-exact against the batch path:
///   - mel: frame g needs samples [g·160−200, g·160+200) — recomputed from
///     the raw buffer per token, so torch's center=True semantics hold
///     exactly (the first 200 virtual samples fall in the silence prefix);
///   - conv stem: per token, re-run over 4 context mel frames + 8 new ones
///     and keep the last 4 positions (convs are local; parity-checked
///     against the batch stem in the listen bin's file mode);
///   - encoder: 4-position KV steps (probe gate 9: bit-identical to batch);
///   - decoder: prompt prefill once enough audio queued, then 1 step/token.
pub struct StreamingTranscriber<'a, B: Backend, O: SttPipeline<B>> {
    stt: &'a O,
    ada: super::decoder::AdaScales<B>,
    enc_caches: O::EncCaches,
    dec_caches: O::DecCaches,
    prompt: Vec<u32>,
    samples: Vec<f32>,
    tokens_encoded: usize, // audio tokens turned into embeds so far
    /// Per-token audio embeds + the instant their samples completed
    /// (latency is measured from here — encoder + queueing + decoder) + the
    /// wall time the encoder side spent on this token.
    queue: std::collections::VecDeque<(Tensor<B, 3>, Instant, f32)>,
    pub tokens: Vec<u32>, // full sequence: prompt + generated
    finished: bool,       // saw EOS
}

/// Immutable encoder-only silence prefix belonging to one loaded model/device
/// and its owning thread. Ears owns it; never persist or share across reloads.
/// Cloned cache containers append into fresh tensors; seed handles stay alive.
/// No decoder, samples, delay conditioning or request timestamps are retained.
pub(crate) struct EncoderPrefix<B: Backend, C> {
    caches: C,
    embeddings: Vec<Tensor<B, 3>>,
}

impl<'a, B: Backend, O: SttPipeline<B>> StreamingTranscriber<'a, B, O> {
    pub fn new(stt: &'a O, delay_ms: usize) -> Self {
        let n_delay = delay_tokens(delay_ms);
        Self {
            ada: stt.ada_scales(n_delay),
            enc_caches: stt.new_enc_caches(),
            dec_caches: stt.new_dec_caches(),
            prompt: prompt_ids(n_delay),
            samples: vec![0f32; N_LEFT_PAD_TOKENS * SAMPLES_PER_TOK],
            tokens_encoded: 0,
            queue: std::collections::VecDeque::new(),
            tokens: Vec::new(),
            finished: false,
            stt,
        }
    }

    /// Called once on the fresh load-time warmup stream, which then continues
    /// normally. Preserve the original 31 M4 calls, not one large encoder pass.
    pub(crate) fn prepare_encoder_prefix(&mut self) -> EncoderPrefix<B, O::EncCaches>
    where O::EncCaches: Clone {
        assert_eq!(self.tokens_encoded, 0, "prefix preparation needs a fresh stream");
        assert_eq!(self.samples.len(), N_LEFT_PAD_TOKENS * SAMPLES_PER_TOK);
        assert!(self.tokens.is_empty() && self.queue.is_empty() && !self.finished);
        assert!(self.push(&[]).is_empty(), "encoder prefix must not decode");
        // k31 needs the first40 real samples; it must never enter the seed.
        assert_eq!(self.tokens_encoded, N_LEFT_PAD_TOKENS - 1);
        assert_eq!(self.queue.len(), self.tokens_encoded);
        EncoderPrefix {
            caches: self.enc_caches.clone(),
            embeddings: self.queue.iter().map(|(t, _, _)| t.clone()).collect(),
        }
    }

    /// The caller must use the seed from this same loaded model/device/owner.
    /// Only Ears uses this in production. Each request owns its mutable state.
    pub(crate) fn from_encoder_prefix(
        stt: &'a O, delay_ms: usize, prefix: &EncoderPrefix<B, O::EncCaches>,
    ) -> Self where O::EncCaches: Clone {
        let mut stream = Self::new(stt, delay_ms);
        stream.enc_caches = prefix.caches.clone();
        stream.tokens_encoded = prefix.embeddings.len();
        let available = Instant::now();
        stream.queue = prefix.embeddings.iter()
            .map(|t| (t.clone(), available, 0.0)).collect();
        stream
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Feed new samples; returns any tokens that became ready.
    pub fn push(&mut self, new_samples: &[f32]) -> Vec<StreamedToken> {
        self.samples.extend_from_slice(new_samples);
        let mut out = Vec::new();
        if self.finished {
            return out;
        }

        // 1. encode every audio token whose samples are complete
        loop {
            let k = self.tokens_encoded;
            let last_needed = (8 * k + 7) * HOP + N_FFT / 2; // exclusive
            if self.samples.len() < last_needed {
                break;
            }
            let avail = Instant::now();
            // mel frames [g0, 8k+8) from samples [g0·160−200, (8k+7)·160+200)
            let g0 = (8 * k).saturating_sub(DOWNSAMPLE);
            let ctx = 8 * k - g0; // 0 for the first token, 4 after
            let s0 = (g0 * HOP) as isize - (N_FFT / 2) as isize;
            let s1 = (8 * k + 7) * HOP + N_FFT / 2;
            let slice: Vec<f32> = if s0 < 0 {
                let mut v = vec![0f32; (-s0) as usize];
                v.extend_from_slice(&self.samples[..s1]);
                v
            } else {
                self.samples[s0 as usize..s1].to_vec()
            };
            let mel = self.stt.mel(&slice, false); // [1,128,ctx+8]
            let stem = self.stt.stem(mel); // [1,(ctx+8)/2,1280]
            let new = stem.clone().narrow(1, ctx / 2, DOWNSAMPLE);
            let h = self.stt.encode(new, &mut self.enc_caches);
            let proj = self.stt.project(h);
            self.queue
                .push_back((proj, avail, avail.elapsed().as_secs_f32() * 1000.0));
            self.tokens_encoded += 1;
        }

        // 2. prefill once the prompt's audio positions are all queued
        let l = self.prompt.len();
        if self.tokens.is_empty() && self.queue.len() >= l {
            // latency counts from the LAST prompt-position embed becoming
            // available — the binding constraint for the first emission.
            let mut avail: Option<Instant> = None;
            let mut enc_ms = 0f32;
            let audio: Vec<_> = (0..l)
                .map(|_| {
                    let (t, a, e) = self.queue.pop_front().unwrap();
                    avail = Some(avail.map_or(a, |x| x.max(a)));
                    enc_ms += e;
                    t
                })
                .collect();
            let avail = avail.expect("l > 0");
            let t0 = Instant::now();
            let audio = Tensor::cat(audio, 1);
            let tok = self.stt.embed(&self.prompt);
            let hidden = self
                .stt
                .decode_step(tok + audio, &self.ada, &mut self.dec_caches);
            let id = argmax_host::<B>(self.stt.logits_last(hidden));
            self.tokens.extend_from_slice(&self.prompt);
            self.tokens.push(id);
            out.push(StreamedToken {
                id,
                latency_ms: avail.elapsed().as_secs_f32() * 1000.0,
                pos: self.tokens.len() - 1,
                enc_ms,
                dec_ms: t0.elapsed().as_secs_f32() * 1000.0,
            });
            if id == EOS {
                self.finished = true;
                return out;
            }
        }

        // 3. one decoder step per queued audio token
        while !self.tokens.is_empty() && !self.queue.is_empty() {
            let (audio, avail, enc_ms) = self.queue.pop_front().unwrap();
            let t0 = Instant::now();
            let prev = *self.tokens.last().unwrap();
            let tok = self.stt.embed(&[prev]);
            let hidden = self
                .stt
                .decode_step(tok + audio, &self.ada, &mut self.dec_caches);
            let id = argmax_host::<B>(self.stt.logits_last(hidden));
            self.tokens.push(id);
            out.push(StreamedToken {
                id,
                latency_ms: avail.elapsed().as_secs_f32() * 1000.0,
                pos: self.tokens.len() - 1,
                enc_ms,
                dec_ms: t0.elapsed().as_secs_f32() * 1000.0,
            });
            if id == EOS {
                self.finished = true;
                break;
            }
        }
        out
    }

    /// Transcript so far.
    pub fn text(&self) -> String {
        self.stt.tekken().decode(&self.tokens)
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq)]
pub(crate) struct PrefixTestState {
    pub encoded: usize,
    pub queued: usize,
    pub tokens: usize,
    pub positions: Vec<usize>,
    pub last_embedding: Vec<f32>,
}

#[cfg(test)]
impl<B: Backend> EncoderPrefix<B, super::fast::FastCaches<B>> {
    pub(crate) fn test_snapshot(&self) -> (Vec<(usize, usize, Vec<f32>, Vec<f32>)>, Vec<Vec<f32>>) {
        let caches = [&self.caches.0[0], self.caches.0.last().unwrap()].into_iter()
            .map(|c| c.prefix_test_sample()).collect();
        let embeds = self.embeddings.iter().map(|t| {
            t.clone().narrow(2, 0, t.dims()[2].min(8))
                .into_data().iter::<f32>().collect()
        }).collect();
        (caches, embeds)
    }
}

#[cfg(test)]
impl<'a, B: Backend, O: SttPipeline<B, EncCaches = super::fast::FastCaches<B>>>
    StreamingTranscriber<'a, B, O>
{
    pub(crate) fn prefix_test_state(&self) -> PrefixTestState {
        PrefixTestState {
            encoded: self.tokens_encoded, queued: self.queue.len(), tokens: self.tokens.len(),
            positions: self.enc_caches.0.iter().map(|c| c.pos).collect(),
            last_embedding: self.queue.back().map_or_else(Vec::new, |(t, _, _)|
                t.clone().into_data().iter::<f32>().collect()),
        }
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;
    use super::super::fast::{FastCaches, FastKv};
    use std::cell::{Cell, RefCell};
    type Cpu = burn_ndarray::NdArray<f32>;

    // Tiny deterministic stages exercise the REAL streaming schedule and
    // FastKv append/COW semantics. They are not a CPU model implementation.
    #[derive(Default)]
    struct Probe {
        device: <Cpu as Backend>::Device,
        encodes: Cell<usize>,
        windows: RefCell<Vec<Vec<f32>>>,
        decode_shapes: RefCell<Vec<(usize, usize)>>,
    }

    impl SttPipeline<Cpu> for Probe {
        type EncCaches = FastCaches<Cpu>;
        type DecCaches = usize;
        fn device(&self) -> &<Cpu as Backend>::Device { &self.device }
        fn tekken(&self) -> &Tekken { panic!("schedule fixture does not decode text") }
        fn mel(&self, samples: &[f32], center: bool) -> Tensor<Cpu, 3> {
            assert!(!center);
            self.windows.borrow_mut().push(samples.to_vec());
            let frames = (samples.len() - N_FFT) / HOP + 1;
            Tensor::full([1, 1, frames], samples.iter().sum::<f32>(), &self.device)
        }
        fn stem(&self, mel: Tensor<Cpu, 3>) -> Tensor<Cpu, 3> {
            let frames = mel.dims()[2] / 2;
            let value = mel.into_data().iter::<f32>().next().unwrap();
            Tensor::full([1, frames, 1], value, &self.device)
        }
        fn new_enc_caches(&self) -> FastCaches<Cpu> { FastCaches(vec![FastKv::new(ENC_WINDOW)]) }
        fn new_dec_caches(&self) -> usize { 0 }
        fn encode(&self, x: Tensor<Cpu, 3>, caches: &mut FastCaches<Cpu>) -> Tensor<Cpu, 3> {
            self.encodes.set(self.encodes.get() + 1);
            assert_eq!(x.dims(), [1, DOWNSAMPLE, 1]);
            let kv = x.clone().reshape([1, 1, DOWNSAMPLE, 1]);
            let _ = caches.0[0].update(kv.clone(), kv);
            x
        }
        fn project(&self, hidden: Tensor<Cpu, 3>) -> Tensor<Cpu, 3> { hidden.sum_dim(1) }
        fn ada_scales(&self, delay: usize) -> AdaScales<Cpu> {
            AdaScales(vec![Tensor::full([1, 1, 1], delay as f32, &self.device)])
        }
        fn embed(&self, ids: &[u32]) -> Tensor<Cpu, 3> { Tensor::zeros([1, ids.len(), 1], &self.device) }
        fn decode_step(&self, x: Tensor<Cpu, 3>, ada: &AdaScales<Cpu>, calls: &mut usize) -> Tensor<Cpu, 3> {
            let delay = ada.0[0].clone().into_data().iter::<f32>().next().unwrap() as usize;
            self.decode_shapes.borrow_mut().push((x.dims()[1], delay));
            *calls += 1;
            Tensor::full([1, 1, 1], *calls as f32, &self.device)
        }
        fn logits_last(&self, hidden: Tensor<Cpu, 3>) -> Tensor<Cpu, 1> {
            let calls = hidden.into_data().iter::<f32>().next().unwrap();
            let id = if calls >= 2.0 { EOS as usize } else { 3 };
            let mut logits = vec![0.0f32; 4];
            logits[id] = 1.0;
            Tensor::from_data(TensorData::new(logits, [4]), &self.device)
        }
    }

    #[test]
    fn encoder_prefix_reuse_skips_completed_work() {
        let probe = Probe::default();
        let mut warm = StreamingTranscriber::new(&probe, 480);
        let seed = warm.prepare_encoder_prefix();
        assert_eq!(probe.encodes.get(), 31);
        assert_eq!(warm.prefix_test_state().positions, [124]);
        assert_eq!(seed.embeddings.len(), 31);
        assert!(warm.push(&[0.0; 40]).is_empty());
        assert_eq!(probe.encodes.get(), 32, "warmup continues without redoing prefix");

        let before = Instant::now();
        let mut request = StreamingTranscriber::from_encoder_prefix(&probe, 480, &seed);
        let after = Instant::now();
        assert!(request.queue.iter().all(|(_, at, cost)| *at >= before && *at <= after && *cost == 0.0));
        assert_eq!(request.samples.len(), N_LEFT_PAD_TOKENS * SAMPLES_PER_TOK);
        assert!(request.push(&[]).is_empty());
        // Genuine behavior: old whole-prefix scheduling spends another31 calls.
        assert_eq!(probe.encodes.get(), 32, "new request must reuse completed encoder work");
        assert!(request.push(&[0.0; 39]).is_empty());
        assert_eq!(probe.encodes.get(), 32);
        assert!(request.push(&[0.0]).is_empty());
        assert_eq!(probe.encodes.get(), 33);
        assert_eq!(request.prefix_test_state().positions, [128]);
    }

    #[test]
    fn encoder_prefix_boundary_keeps_first_real_impulse() {
        let probe = Probe::default();
        let mut fresh = StreamingTranscriber::new(&probe, 480);
        let seed = fresh.prepare_encoder_prefix();
        let original = seed.test_snapshot();
        let mut zero = StreamingTranscriber::from_encoder_prefix(&probe, 480, &seed);
        let mut impulse = StreamingTranscriber::from_encoder_prefix(&probe, 480, &seed);
        let mut first = [0.0; 39];
        first[0] = 1.0;
        let before = probe.encodes.get();
        assert!(impulse.push(&first).is_empty());
        assert_eq!(probe.encodes.get(), before);
        assert_eq!(impulse.prefix_test_state().encoded, 31);
        assert!(impulse.push(&[0.0]).is_empty());
        assert_eq!(probe.encodes.get(), before + 1);
        assert_eq!(probe.windows.borrow().last().unwrap().iter().sum::<f32>(), 1.0);
        assert_eq!(impulse.prefix_test_state().last_embedding, [4.0]);
        assert_eq!(zero.prefix_test_state().positions, [124]);
        assert!(zero.push(&[0.0; 40]).is_empty());
        assert_eq!(zero.prefix_test_state().last_embedding, [0.0]);
        assert_eq!(seed.test_snapshot(), original);
        assert_eq!(fresh.prefix_test_state().positions, [124]);
        // Advancing/consuming one queue never consumes another's31 seed slots.
        assert_eq!(fresh.queue.len(), 31);
        assert_eq!(zero.queue.len(), 32);
        assert_eq!(impulse.queue.len(), 32);
    }

    #[test]
    fn encoder_prefix_preserves_delay_prefill_empty_short_and_eos() {
        let probe = Probe::default();
        let mut warm = StreamingTranscriber::new(&probe, 480);
        let seed = warm.prepare_encoder_prefix();
        for (delay, samples) in [(80, Vec::new()), (480, vec![1.0]), (2400, vec![0.0; 40])] {
            let mut fresh = StreamingTranscriber::new(&probe, delay);
            let mut reused = StreamingTranscriber::from_encoder_prefix(&probe, delay, &seed);
            assert!(fresh.push(&samples).is_empty());
            assert!(reused.push(&samples).is_empty());
            let n_delay = delay_tokens(delay);
            let align = (SAMPLES_PER_TOK - samples.len() % SAMPLES_PER_TOK) % SAMPLES_PER_TOK;
            let tail = vec![0.0; align + (n_delay + 1 + OFFLINE_BUFFER_TOKENS) * SAMPLES_PER_TOK + N_FFT / 2];
            probe.decode_shapes.borrow_mut().clear();
            let a: Vec<_> = fresh.push(&tail).into_iter().map(|t| t.id).collect();
            let expected_shapes = probe.decode_shapes.borrow().clone();
            probe.decode_shapes.borrow_mut().clear();
            let b: Vec<_> = reused.push(&tail).into_iter().map(|t| t.id).collect();
            assert_eq!(a, b);
            assert_eq!(fresh.tokens, reused.tokens);
            assert_eq!(expected_shapes, *probe.decode_shapes.borrow());
            assert_eq!(expected_shapes[0], (prompt_ids(n_delay).len(), n_delay));
            assert_eq!(expected_shapes[1], (1, n_delay));
            assert!(fresh.is_finished() && reused.is_finished());
            assert!(fresh.push(&[1.0; 40]).is_empty());
            assert!(reused.push(&[1.0; 40]).is_empty());
        }
        assert_eq!(seed.caches.0[0].pos, 124);
    }
}
