//! PersonaPlex end-to-end voice pipeline — LM part 3: the audio-out
//! integration around [`LmGen`], wiring input WAV → Mimi encode (user codes)
//! → temporal + depth free-run → agent codes → Mimi decode → 24 kHz PCM.
//!
//! Stream semantics (moshi `offline.py`): of the 17 undelayed output streams
//! `[text, agent 1..8, user 1..8]`, ONLY the agent audio streams `1..=8` are
//! Mimi-decoded (`decode_tokens_to_pcm` slices `out[:, 1:9]`); stream 0 is
//! text, streams 9..=16 are the model's prediction of the USER's audio and
//! are never decoded.
//!
//! **The skip / partial-frame rule** (moshi `LMGen.step`, read from lm.py —
//! there is no partial frame, only `None`):
//!
//! 1. the very first call (`offset == 0`) seeds the token ring with initial
//!    tokens and returns `None` — no model step at all
//!    (`prepare_step_input`'s `state.offset == 0` branch);
//! 2. while `offset <= max_delay` (= 1 here) the model steps but
//!    `process_transformer_output` still returns `None` — the undelayed read
//!    position of the acoustic (delay-1) streams has not been written yet;
//! 3. from `offset == max_delay + 1` onward EVERY step returns a complete
//!    frame, `out[k] = cache[k][(offset − max_delay + delay_k) % CT]`. By
//!    construction each emitted slot has been written by a sampled or
//!    provided token — initial (2048) and ungenerated (−2) tokens never
//!    surface. [`agent_codes`] asserts that invariant before handing codes
//!    to the decoder.
//!
//! In the production prompt flow (voice prompt → silence → text → silence →
//! user audio) both `None` steps happen inside the voice-prompt phase, so
//! every user-audio step yields a decodable frame; prompt-phase frames are
//! discarded (the oracle's `_step_*` helpers ignore them too).
//!
//! CPU-f32 parity path (`personaplex_probe pipeline` gates the chain against
//! the oracle's `out_audio` golden). The realtime build is
//! [`RealtimePipeline`] (feature `q4`): same step semantics on the fast
//! stages — Metal quantized temporal + Accelerate/NEON depformer + CPU Mimi
//! — gated by `personaplex_rt_probe pipeline`.

use burn::prelude::*;

use super::config as cfg;
use super::depth::DepthTransformer;
use super::lmgen::LmGen;
use super::mimi::config as mimi_cfg;
use super::mimi::{MimiDecoder, MimiEncoder};
use super::temporal::TemporalTransformer;
use crate::nn::weight_loader::WeightLoader;

#[cfg(feature = "q4")]
use super::depth::argmax;
#[cfg(feature = "q4")]
use super::depth_fast::DepthFast;
#[cfg(feature = "q4")]
use super::depth_gpu::{DepthFmt, DepthGpu, NO_FORCE};
#[cfg(feature = "q4")]
use super::lmgen::{Prepared, StreamCache};
#[cfg(feature = "q4")]
use super::temporal_metal::{Head, TemporalMetal, WeightFmt};

/// Agent-side spacer frame (`lm.py SILENCE_TOKENS`) as step input.
pub const SILENCE: [i64; 8] = {
    let mut a = [0i64; 8];
    let mut i = 0;
    while i < 8 {
        a[i] = cfg::SILENCE_TOKENS[i] as i64;
        i += 1;
    }
    a
};
/// User-side reference sine frame (`lm.py SINE_TOKENS`) as step input.
pub const SINE: [i64; 8] = {
    let mut a = [0i64; 8];
    let mut i = 0;
    while i < 8 {
        a[i] = cfg::SINE_TOKENS[i] as i64;
        i += 1;
    }
    a
};

/// The full voice pipeline: Mimi encoder (user audio in), the 7B LM step
/// machine, and the Mimi decoder (agent audio out). All three load from the
/// one union weight pile (`models/personaplex.pile`).
pub struct VoicePipeline<B: Backend> {
    pub lm: LmGen<B>,
    pub encoder: MimiEncoder,
    pub decoder: MimiDecoder,
}

impl<B: Backend> VoicePipeline<B> {
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        Self {
            lm: LmGen::new(
                TemporalTransformer::load(loader, device),
                DepthTransformer::load(loader, device),
            ),
            encoder: MimiEncoder::load(loader),
            decoder: MimiDecoder::load(loader),
        }
    }

    /// Switch the step machine onto seeded sampling (`cfg` + `seed`) for both
    /// the text and audio heads. A greedy `cfg` reproduces argmax exactly.
    pub fn set_sampling(&mut self, cfg: super::sampling::SamplingConfig, seed: u64) {
        self.lm.set_sampling(cfg, seed);
    }

    /// Drop back to greedy argmax.
    pub fn set_greedy(&mut self) {
        self.lm.set_greedy();
    }

    /// Reset the streaming state so a new conversation can start WITHOUT
    /// reloading weights (see [`LmGen::reset_session`]). Token-exact vs a
    /// fresh [`VoicePipeline::load`] of the same weights.
    pub fn reset_session(&mut self) {
        self.lm.reset_session();
    }

    /// Voice-prompt replay (moshi `_step_voice_prompt` on a packaged voice):
    /// feed the pre-recorded embeddings (`[n, 4096]` row-major) one step at a
    /// time, then overwrite the token ring with the packaged cache snapshot
    /// (`[17, CT]` row-major — the oracle's `state.cache.copy_(...)`).
    pub fn prompt_voice(&mut self, embeddings: &[f32], cache_snapshot: &[i64], device: &B::Device) {
        assert_eq!(
            embeddings.len() % cfg::DIM,
            0,
            "vp embeddings not row-aligned"
        );
        for row in embeddings.chunks_exact(cfg::DIM) {
            let x = Tensor::<B, 1>::from_floats(row, device).reshape([1, 1, cfg::DIM]);
            self.lm.step_embeddings(x);
        }
        self.lm.stream.overwrite(cache_snapshot);
    }

    /// Silence spacer (moshi `_step_audio_silence`): `frames` steps of agent
    /// SILENCE + user SINE + text PAD.
    pub fn prompt_silence(&mut self, frames: usize, device: &B::Device) {
        for _ in 0..frames {
            self.lm.step(
                Some(&SINE),
                Some(&SILENCE),
                Some(cfg::TEXT_PAD_TOKEN as i64),
                device,
            );
        }
    }

    /// Text system prompt (moshi `_step_text_prompt`): one token per step,
    /// agent SILENCE + user SINE alongside.
    pub fn prompt_text(&mut self, tokens: &[i64], device: &B::Device) {
        for &t in tokens {
            self.lm.step(Some(&SINE), Some(&SILENCE), Some(t), device);
        }
    }

    /// The full system-prompt flow from an assembled [`super::prompt::Prompt`]
    /// (moshi `step_system_prompts`): voice replay → silence → text → silence.
    pub fn run_prompt(&mut self, p: &super::prompt::Prompt, device: &B::Device) {
        self.prompt_voice(&p.voice.embeddings, &p.voice.cache, device);
        self.prompt_silence(p.silence_frames, device);
        self.prompt_text(&p.text_tokens, device);
        self.prompt_silence(p.silence_frames, device);
    }

    /// One user-audio frame (8 Mimi codes) in → the undelayed 17-stream
    /// output frame, once past the delay horizon (see the skip rule in the
    /// module docs — in the prompt flow this is always `Some` by the time
    /// user audio starts). Feed [`agent_codes`] of it to [`Self::decode`].
    pub fn step_user_frame(
        &mut self,
        codes: &[i64; 8],
        device: &B::Device,
    ) -> Option<[i64; cfg::NUM_STREAMS]> {
        self.lm.step(Some(codes), None, None, device).out
    }

    /// Agent frames → 24 kHz mono PCM (1920 samples per frame).
    pub fn decode(&self, frames: &[[u32; mimi_cfg::NUM_CODEBOOKS]]) -> Vec<f32> {
        self.decoder.decode(frames)
    }
}

/// One [`RealtimePipeline`] step's trace — the [`super::lmgen::StepTrace`]
/// fields plus the read-back text logits (the probe gates them; a production
/// loop can ignore them — they are read back anyway for the host argmax).
#[cfg(feature = "q4")]
pub struct RtStepTrace {
    /// The 17 input tokens fed to the temporal stack (`None` for
    /// embedding-fed voice-prompt steps and the offset-0 seeding call).
    pub input: Option<[i64; cfg::NUM_STREAMS]>,
    /// `next_text_token` handed to the depformer (forced target or sampled).
    pub next_text: i64,
    /// The depformer's 16 greedy tokens.
    pub dep_tokens: [i64; cfg::DEP_Q],
    /// Undelayed output frame, once past the delay horizon.
    pub out: Option<[i64; cfg::NUM_STREAMS]>,
    /// The temporal stack's text logits `[32000]` (empty for the offset-0
    /// seeding call).
    pub text_logits: Vec<f32>,
}

/// Which depth transformer [`RealtimePipeline`] drives.
///
/// The depformer is the last host stage of the realtime frame, and it is the
/// one that competes with whatever else the machine is compiling: the temporal
/// stack is already on the device ([`TemporalMetal`]) and cannot be starved by
/// CPU load, so a build storm lands entirely on this stage. [`Self::Gpu`]
/// selects the cubecl port ([`DepthGpu`]), which takes it off the host too.
///
/// **Default is [`Self::Gpu`] at uniform q8**, measured — see
/// [`Self::from_env`] for the escape hatch and the wiring commits for the
/// numbers. The short version, on M4 Max in the shipping decode config
/// (temp 0.8 / top-k 250 / top-p 0.95), paired so both arms saw one machine
/// state: the depth stage goes 105.9 -> 34.1 ms per frame at temporal fill
/// 256, and the reconstructed LM step goes from 0.56x realtime (73 of 80
/// frames over the 80 ms budget) to 1.13x (22 of 80). The CPU arm has no
/// headroom left to reclaim either — it is at the host's memory ceiling,
/// where fewer weight bytes buy only more unpack work.
#[cfg(feature = "q4")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthChoice {
    /// [`DepthFast`], the Accelerate/NEON CPU predictor. `f16` picks its
    /// weight storage width (f32 accumulate either way).
    Cpu { f16: bool },
    /// [`DepthGpu`], the cubecl port, with each matvec at its [`DepthFmt`]
    /// slot. Uniform q8 is the only format that gates — see [`DepthFmt`] for
    /// the measured mixed-format table.
    Gpu(DepthFmt),
}

#[cfg(feature = "q4")]
impl DepthChoice {
    /// Read the arm from `PERSONAPLEX_DEPTH`: unset or `gpu` selects the
    /// cubecl port at uniform q8 (the measured default), `gpu:SPEC` selects it
    /// at an explicit [`DepthFmt`] spec (`gpu:f16`, `gpu:q8:qkv=f16`, …), and
    /// `cpu` falls back to the host predictor with `depth_f16` deciding its
    /// storage.
    ///
    /// The CPU fallback exists because the GPU arm needs a working device and
    /// the host arm does not — that is the one thing it still buys, and it is
    /// why it is a value here rather than deleted.
    ///
    /// An unparseable value panics rather than falling back: a typo that
    /// silently changed the arm would make a whole measurement mean the
    /// opposite of what it says.
    pub fn from_env(depth_f16: bool) -> Self {
        match std::env::var("PERSONAPLEX_DEPTH") {
            Err(_) => Self::Gpu(DepthFmt::uniform(WeightFmt::Q8)),
            Ok(v) => match v.as_str() {
                "cpu" => Self::Cpu { f16: depth_f16 },
                "gpu" => Self::Gpu(DepthFmt::uniform(WeightFmt::Q8)),
                other => {
                    let spec = other.strip_prefix("gpu:").unwrap_or_else(|| {
                        panic!("PERSONAPLEX_DEPTH={other} (expected cpu | gpu | gpu:<fmt>)")
                    });
                    Self::Gpu(DepthFmt::parse(spec).unwrap_or_else(|| {
                        panic!("PERSONAPLEX_DEPTH={other}: bad depth format {spec}")
                    }))
                }
            },
        }
    }

    /// One-line arm label for a run header — every timing this pipeline
    /// produces is per-arm, so the arm belongs next to the number.
    pub fn label(&self) -> String {
        match self {
            Self::Cpu { f16 } => {
                format!("cpu depth_fast ({})", if *f16 { "f16" } else { "f32" })
            }
            Self::Gpu(fmt) => format!("gpu depth_gpu ({})", fmt.label()),
        }
    }
}

/// The loaded depth transformer, either arm — the one call site in
/// [`RealtimePipeline::forward`] goes through [`Self::frame`].
#[cfg(feature = "q4")]
pub enum DepthArm {
    Cpu(DepthFast),
    Gpu(DepthGpu),
}

#[cfg(feature = "q4")]
impl DepthArm {
    /// One temporal frame's depformer pass. The two arms take the forcing
    /// rule in different shapes — `Option<i64>` per step on the host, a
    /// [`NO_FORCE`]-sentinel `[16]` u32 upload on the device — and this is
    /// where that translation lives.
    ///
    /// Both arms sample. The GPU arm cannot hand its logits to a host
    /// `Sampler` — its argmax and the prev-token chain live on the device, and
    /// that is exactly what keeps a frame at ONE sync — so it draws the 16
    /// uniforms here, from the same seeded RNG, and does the
    /// temperature/top-k/top-p work in `dep_sample_kernel`. Greedy audio
    /// decode is not an acceptable substitute: it collapses the agent to a
    /// near-constant code (measured: 3 distinct codebook-0 values across 125
    /// frames, rms −74 dB; see `personaplex_listen`).
    pub fn frame(
        &mut self,
        transformer_out: &[f32],
        text_token: i64,
        forced: &[Option<i64>; cfg::DEP_Q],
        sampler: Option<&mut super::sampling::Sampler>,
    ) -> [i64; cfg::DEP_Q] {
        match self {
            Self::Cpu(d) => d.frame(transformer_out, text_token, forced, None, sampler),
            Self::Gpu(d) => {
                assert!(text_token >= 0, "text token {text_token}");
                let forced: [u32; cfg::DEP_Q] =
                    std::array::from_fn(|s| forced[s].map_or(NO_FORCE, |t| t as u32));
                let out = match sampler {
                    None => d.frame(transformer_out, text_token as u32, &forced, cfg::DEP_Q),
                    Some(smp) => {
                        // The RNG never leaves the host: 16 uniforms are drawn
                        // here, in step order, and ride up with the frame.
                        let scfg = *smp.config();
                        let mut u = [0f32; cfg::DEP_Q];
                        smp.uniforms(&mut u);
                        d.frame_sampled(
                            transformer_out,
                            text_token as u32,
                            &forced,
                            cfg::DEP_Q,
                            &scfg,
                            &u,
                        )
                    }
                };
                std::array::from_fn(|s| out[s])
            }
        }
    }

    /// Which arm this is, for a run header — a depth timing without its arm
    /// is not evidence.
    pub fn label(&self) -> String {
        match self {
            Self::Cpu(d) => format!(
                "cpu depth_fast ({})",
                if d.is_f16() { "f16" } else { "f32" }
            ),
            Self::Gpu(d) => format!("gpu depth_gpu ({})", d.fmt().label()),
        }
    }

    /// The `[16, 2048]` logit slab of the last [`Self::frame`], row-major.
    /// The GPU arm reads it back from the device on demand (a gate's view,
    /// not the realtime path's — a greedy frame never needs it).
    pub fn logits(&self) -> Vec<f32> {
        match self {
            Self::Cpu(d) => d.logits().to_vec(),
            Self::Gpu(d) => d.logits(cfg::DEP_Q),
        }
    }

    /// Weight bytes a full 16-codebook frame streams.
    pub fn frame_weight_bytes(&self) -> usize {
        match self {
            Self::Cpu(d) => d.frame_weight_bytes(),
            Self::Gpu(d) => d.frame_weight_bytes(),
        }
    }

    /// The CPU arm's in-situ timing decomposition (frames, total, cond, stack
    /// gemv, head, scalar-rest) in ms/frame. The GPU arm has no host-side
    /// decomposition to drain — its stages are device kernels, so it reports
    /// zero frames and callers must skip the breakdown rather than divide.
    pub fn take_bench(&mut self) -> (u64, f64, f64, f64, f64, f64) {
        match self {
            Self::Cpu(d) => d.take_bench(),
            Self::Gpu(_) => (0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    }
}

/// The **realtime** voice pipeline: the same step semantics as [`VoicePipeline`]
/// (the CPU-f32 parity oracle) rebuilt on the fast stages — the Metal
/// quantized temporal transformer ([`TemporalMetal`], q4/q8/f16 stack with
/// the f16 logit head as the documented production choice), the depformer on
/// either arm ([`DepthArm`] — the cubecl port by default, the CPU predictor
/// under `PERSONAPLEX_DEPTH=cpu`), and the CPU Mimi codec.
///
/// **Numerics honesty:** with a quantized (q4/q8) temporal stack this is a
/// REAL numerics change — free-run token streams are expected to diverge
/// from the f32 oracle at some step (an argmax near-tie flips, then the
/// autoregressive paths separate). `personaplex_rt_probe pipeline` measures
/// agreement %, the first-divergence step, and the logits at the divergence
/// point; the f16 stack is the wiring-exactness ablation. Never claim
/// token-exactness for the quantized formats.
///
/// **Mimi threading:** Mimi runs sequentially in this build (batch encode up
/// front, batch decode at the end in the golden gate flow; ~5 ms/frame if
/// decoded per frame in a live duplex loop). Moving it onto its own thread
/// (as in `qwen3tts_stream`) is deferred until a live session loop exists to
/// consume streaming frames — the LM step machine here is synchronous, so a
/// decode thread would only overlap the ~5 ms Mimi cost, not the LM.
#[cfg(feature = "q4")]
pub struct RealtimePipeline {
    pub temporal: TemporalMetal,
    pub depth: DepthArm,
    pub stream: StreamCache,
    pub encoder: MimiEncoder,
    pub decoder: MimiDecoder,
    /// Logit head variant (f16 is the measured production choice — see
    /// `temporal_metal` module docs).
    pub head: Head,
    /// `None` = greedy argmax (parity). `Some` = seeded temp/top-k/top-p over
    /// the text head and each depformer audio step (the quality path).
    sampler: Option<super::sampling::Sampler>,
}

#[cfg(feature = "q4")]
impl RealtimePipeline {
    /// Load all components from one authority-bound runtime source. `fmt` picks
    /// the temporal stack's weight format (q4/q8/f16); `depth_f16` stores the
    /// CPU depformer's weights as f16 (f32 accumulate) instead of f32, and is
    /// ignored unless `PERSONAPLEX_DEPTH=cpu` selects that arm (the default
    /// GPU arm carries its own per-tensor [`DepthFmt`]). Accepting the authority/loader pair
    /// prevents a future native cache from being checked against one member
    /// while its unchanged weights come from another.
    pub fn load(source: &super::PersonaPlexRuntimeSource, fmt: WeightFmt, depth_f16: bool) -> Self {
        Self::load_with_depth(source, fmt, DepthChoice::from_env(depth_f16))
    }

    /// [`Self::load`] with the depth arm named outright instead of read from
    /// `PERSONAPLEX_DEPTH` — the form an A/B measurement wants, so both arms
    /// can be built in one process against one machine state.
    pub fn load_with_depth(
        source: &super::PersonaPlexRuntimeSource,
        fmt: WeightFmt,
        depth: DepthChoice,
    ) -> Self {
        let loader = source.loader();
        Self {
            temporal: TemporalMetal::load(loader, fmt),
            depth: match depth {
                DepthChoice::Cpu { f16 } => DepthArm::Cpu(DepthFast::load(loader, f16)),
                DepthChoice::Gpu(dfmt) => DepthArm::Gpu(DepthGpu::load(loader, dfmt)),
            },
            stream: StreamCache::new(),
            encoder: MimiEncoder::load(loader),
            decoder: MimiDecoder::load(loader),
            head: Head::F16,
            sampler: None,
        }
    }

    /// Authority-safe automatic load.
    ///
    /// The legacy filename-discovered cache could not prove that transformed
    /// bytes belonged to this source. Automatic loading therefore recomputes
    /// from the verified model while preserving the authority/loader binding
    /// needed by any future cache design.
    pub fn load_auto(
        source: &super::PersonaPlexRuntimeSource,
        fmt: WeightFmt,
        depth_f16: bool,
    ) -> Self {
        Self::load(source, fmt, depth_f16)
    }

    /// Switch the realtime step machine onto seeded sampling (`cfg` + `seed`)
    /// for both the temporal text head and the depformer audio heads. A greedy
    /// `cfg` reproduces argmax exactly.
    pub fn set_sampling(&mut self, cfg: super::sampling::SamplingConfig, seed: u64) {
        self.sampler = Some(super::sampling::Sampler::new(cfg, seed));
    }

    /// Drop back to greedy argmax (clears the sampler).
    pub fn set_greedy(&mut self) {
        self.sampler = None;
    }

    /// Reset the streaming state so a new conversation can start WITHOUT
    /// reloading weights (moshi's per-session `streaming` reset): clears the
    /// temporal KV cache + offset, the stream ring/offset, AND re-seeds an
    /// existing sampler back to its original seed — so a reset session samples
    /// the identical token stream as the first, no `set_sampling` re-call
    /// needed. Token-exact vs a fresh [`RealtimePipeline::load`] of the same
    /// weights (gated).
    pub fn reset_session(&mut self) {
        self.temporal.reset();
        self.stream.reset();
        if let Some(smp) = self.sampler.as_mut() {
            smp.reseed();
        }
    }

    fn forward(
        &mut self,
        x: &[f32],
        p: Prepared,
        input: Option<[i64; cfg::NUM_STREAMS]>,
        arbiter: Option<&mut dyn FnMut(&[f32], i64) -> i64>,
    ) -> RtStepTrace {
        self.temporal.step_submit(x, self.head);
        let (hidden, text_logits) = self.temporal.read_hidden_logits();
        // Text first, then audio — one RNG threaded per frame (reproducible).
        let mut sampled_text = match self.sampler.as_mut() {
            Some(smp) => smp.token(&text_logits) as i64,
            None => argmax(&text_logits) as i64,
        };
        // The arbitration point (see [`Self::step_arbitrated`]). It sits HERE,
        // between reading the row and conditioning the depformer, because
        // those happen inside one call: a caller that watched
        // `RtStepTrace.text_logits` and reacted next frame would be steering
        // the audio one frame after the audio was generated. Bypassed when the
        // caller provided stream 0 outright — an explicit force is already a
        // decision and must not be second-guessed.
        if !p.provided[0] {
            if let Some(decide) = arbiter {
                sampled_text = decide(&text_logits, sampled_text);
            }
        }
        let next_text = if p.provided[0] {
            p.target[0]
        } else {
            sampled_text
        };
        let dep_tokens = self
            .depth
            .frame(&hidden, next_text, &p.forced(), self.sampler.as_mut());
        // `sampled_text` — post-arbitration — is what `commit` writes into the
        // ring, so the substituted token enters the model's own history as if
        // it had chosen it. That is the whole basis of the no-garble property:
        // on release it continues from a prefix it owns.
        let out = self.stream.commit(&p, sampled_text, &dep_tokens);
        RtStepTrace {
            input,
            next_text,
            dep_tokens,
            out,
            text_logits,
        }
    }

    fn token_step(
        &mut self,
        prepared: Option<Prepared>,
        arbiter: Option<&mut dyn FnMut(&[f32], i64) -> i64>,
    ) -> RtStepTrace {
        let Some(p) = prepared else {
            return RtStepTrace {
                input: None,
                next_text: -1,
                dep_tokens: [0; cfg::DEP_Q],
                out: None,
                text_logits: Vec::new(),
            };
        };
        let input = p.input;
        let x = self.temporal.embed_codes(&input);
        self.forward(&x, p, Some(input), arbiter)
    }

    /// moshi `LMGen.step` on the fast stages (see [`super::lmgen::LmGen::step`]
    /// for the contract).
    pub fn step(
        &mut self,
        input_tokens: Option<&[i64; 8]>,
        moshi_tokens: Option<&[i64; 8]>,
        text_token: Option<i64>,
    ) -> RtStepTrace {
        let prepared = self.stream.prepare(input_tokens, moshi_tokens, text_token);
        self.token_step(prepared, None)
    }

    /// Generation-only step: user streams `9..17` are `-1` in every
    /// post-seed temporal input and the user half of the depformer is never
    /// forced. Unlike passing `Some(&[-1; 8])` to [`Self::step`], this cannot
    /// turn the zero-embedding sentinel into a forced depth token. Agent
    /// audio and text forcing retain their ordinary semantics.
    pub fn step_output_only(
        &mut self,
        moshi_tokens: Option<&[i64; 8]>,
        text_token: Option<i64>,
    ) -> RtStepTrace {
        let prepared = self.stream.prepare_output_only(moshi_tokens, text_token);
        self.token_step(prepared, None)
    }

    /// Step with stream 0 decided AT THE SAMPLING BOUNDARY: the model samples
    /// its own text token, `decide` sees that token and the row it came from,
    /// and whatever `decide` returns is what conditions the depformer for this
    /// same frame and what enters the token ring.
    ///
    /// This is a different division of labour from [`Self::step`]'s
    /// `text_token`. Forcing a token supplies BOTH what is said and when —
    /// the caller has to invent a rhythm, and every rhythm it can invent is a
    /// schedule the model was not trained on. Arbitrating instead lets the
    /// model keep the timing it does know — where pauses fall, how long they
    /// run, where a word boundary is marked — while the caller substitutes
    /// only the content of the tokens that are actually words.
    ///
    /// `decide` returning its second argument unchanged is exactly
    /// [`Self::step`] with `text_token: None`.
    pub fn step_arbitrated(
        &mut self,
        input_tokens: Option<&[i64; 8]>,
        moshi_tokens: Option<&[i64; 8]>,
        decide: &mut dyn FnMut(&[f32], i64) -> i64,
    ) -> RtStepTrace {
        let prepared = self.stream.prepare(input_tokens, moshi_tokens, None);
        self.token_step(prepared, Some(decide))
    }

    /// Generation-only counterpart to [`Self::step_arbitrated`]. The model
    /// keeps control of text timing while the temporal user contribution is
    /// exactly zero and the user depformer steps remain unforced.
    pub fn step_output_only_arbitrated(
        &mut self,
        moshi_tokens: Option<&[i64; 8]>,
        decide: &mut dyn FnMut(&[f32], i64) -> i64,
    ) -> RtStepTrace {
        let prepared = self.stream.prepare_output_only(moshi_tokens, None);
        self.token_step(prepared, Some(decide))
    }

    /// Voice-prompt replay step: the cache advances with dummy initial
    /// tokens + PAD text while the temporal stack consumes the pre-recorded
    /// embedding row `[4096]` (see [`super::lmgen::LmGen::step_embeddings`]).
    pub fn step_embedding(&mut self, x: &[f32]) -> RtStepTrace {
        let dummy = [cfg::CARD as i64; 8];
        let p = loop {
            if let Some(p) =
                self.stream
                    .prepare(Some(&dummy), Some(&dummy), Some(cfg::TEXT_PAD_TOKEN as i64))
            {
                break p;
            }
        };
        self.forward(x, p, None, None)
    }

    /// Voice-prompt replay (moshi `_step_voice_prompt`): feed the packaged
    /// embeddings (`[n, 4096]` row-major), then overwrite the token ring
    /// with the packaged cache snapshot (`[17, CT]` row-major).
    pub fn prompt_voice(&mut self, embeddings: &[f32], cache_snapshot: &[i64]) {
        assert_eq!(
            embeddings.len() % cfg::DIM,
            0,
            "vp embeddings not row-aligned"
        );
        for row in embeddings.chunks_exact(cfg::DIM) {
            self.step_embedding(row);
        }
        self.stream.overwrite(cache_snapshot);
    }

    /// Silence spacer (moshi `_step_audio_silence`).
    pub fn prompt_silence(&mut self, frames: usize) {
        for _ in 0..frames {
            self.step(
                Some(&SINE),
                Some(&SILENCE),
                Some(cfg::TEXT_PAD_TOKEN as i64),
            );
        }
    }

    /// Text system prompt (moshi `_step_text_prompt`).
    pub fn prompt_text(&mut self, tokens: &[i64]) {
        for &t in tokens {
            self.step(Some(&SINE), Some(&SILENCE), Some(t));
        }
    }

    /// The full system-prompt flow from an assembled [`super::prompt::Prompt`]
    /// (moshi `step_system_prompts`): voice replay → silence → text → silence.
    pub fn run_prompt(&mut self, p: &super::prompt::Prompt) {
        self.prompt_voice(&p.voice.embeddings, &p.voice.cache);
        self.prompt_silence(p.silence_frames);
        self.prompt_text(&p.text_tokens);
        self.prompt_silence(p.silence_frames);
    }

    /// One user-audio frame (8 Mimi codes) in → the undelayed 17-stream
    /// output frame (always `Some` once the prompt flow consumed the `None`
    /// steps). Feed [`agent_codes`] of it to [`Self::decode`].
    pub fn step_user_frame(&mut self, codes: &[i64; 8]) -> Option<[i64; cfg::NUM_STREAMS]> {
        self.step(Some(codes), None, None).out
    }

    /// Agent frames → 24 kHz mono PCM (1920 samples per frame).
    pub fn decode(&self, frames: &[[u32; mimi_cfg::NUM_CODEBOOKS]]) -> Vec<f32> {
        self.decoder.decode(frames)
    }
}

/// Load a depth arm on its own, from the same authority-bound source a
/// [`RealtimePipeline`] was loaded from.
///
/// This exists for the PAIRED A/B measurement, and the pairing is not
/// fastidiousness: on a shared desktop the drift in ambient load between two
/// sequential runs is routinely larger than the difference between the arms,
/// so two separate runs cannot answer which arm is faster. Holding both arms
/// in one process and driving them from the same hidden state at the same
/// instant is the only comparison the machine will actually support.
#[cfg(feature = "q4")]
pub fn load_depth_arm(source: &super::PersonaPlexRuntimeSource, depth: DepthChoice) -> DepthArm {
    let loader = source.loader();
    match depth {
        DepthChoice::Cpu { f16 } => DepthArm::Cpu(DepthFast::load(loader, f16)),
        DepthChoice::Gpu(fmt) => DepthArm::Gpu(DepthGpu::load(loader, fmt)),
    }
}

/// Extract the agent's Mimi codes (streams `1..=8`) from an undelayed output
/// frame. Streams `9..=16` (user prediction) are never decoded; stream 0 is
/// text. Asserts the delay machinery's invariant: emitted frames contain
/// only real codebook entries — an initial (2048) or ungenerated token here
/// means the skip rule was violated upstream.
pub fn agent_codes(out: &[i64; cfg::NUM_STREAMS]) -> [u32; mimi_cfg::NUM_CODEBOOKS] {
    std::array::from_fn(|q| {
        let t = out[1 + q];
        assert!(
            (0..cfg::CARD as i64).contains(&t),
            "agent stream {} token {t} outside the Mimi codebook — an initial/ungenerated \
             token leaked past the delay horizon",
            1 + q
        );
        t as u32
    })
}
