//! Pile-native B1 Breeze generator. No checkpoint file, Python model, hidden
//! weight catalogue, CPU LLM or reference-prefixed acoustic output is reachable.
//! CFG1 and paired conditional/negative guidance, with independent branch KV.
use super::{
    backbone::{Decoder, Kv},
    config::{Config, DecoderConfig},
    cuda_ops as ops,
    depth::Depth,
    load::{self, Artifacts},
    nvfp4::{Linear, Nvfp4},
    prompt::GuidedPrompt,
    sampling::{self, Sampler},
    text_encoder::TextEncoder,
};
use crate::nn::cuda_bf16_alias::CudaBf16Aliases;
use anyhow::{Result, ensure};
use ops::Tensor;
use std::time::Instant;
use triblespace::{
    core::repo::BlobStoreGet,
    prelude::{Id, TribleSet},
};

pub use super::nvfp4::Weights;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PromptSegment {
    Text(Vec<u32>),
    AudioFrames(Vec<[u16; 16]>),
    AudioEos,
}
#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub max_frames: usize,
    pub max_context: usize,
    pub seed: u64,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub cfg_scale: f32,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_frames: 1500,
            max_context: 2048,
            seed: 42,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.1,
            do_sample: true,
            cfg_scale: 1.0,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    Eos,
    FrameLimit,
    ContextLimit,
}
#[derive(Debug)]
pub struct GenerationReport {
    pub termination: Termination,
    pub frames: usize,
    pub backbone_ids: Vec<u32>,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub callback_seconds: f64,
    pub total_seconds: f64,
    pub cfg_scale: f32,
    pub conditional_prompt_tokens: usize,
    pub negative_prompt_tokens: Option<usize>,
}

/// A typed selector only for the duration of construction, not a retained
/// catalogue. Each consumer asks for its fixed leaf with explicit opaque root.
pub(super) struct Binder<'a, R> {
    pub facts: &'a TribleSet,
    pub reader: &'a R,
    pub root: Id,
    pub aliases: &'a mut CudaBf16Aliases,
    pub weights: Weights,
    /// NVFP4 copies made so far and their device bytes, for the load line.
    pub nvfp4: (usize, u64),
}
impl<R: BlobStoreGet> Binder<'_, R> {
    pub(super) unsafe fn weight<const N: usize>(
        &mut self,
        name: &str,
        shape: [u64; N],
    ) -> Result<Tensor> {
        let (_, blob) = load::tensor_bf16(self.facts, self.reader, self.root, name, shape)?;
        // SAFETY: forwarded genuine mapped immutable-prefix contract from caller.
        let value = unsafe { self.aliases.bind_pile_leaf(blob) }.map_err(anyhow::Error::msg)?;
        ensure!(
            !value.handle.can_mut(),
            "weight alias unexpectedly writable"
        );
        Ok(value)
    }
    /// A linear projection `[out, in]`: always the BF16 alias, plus a resident
    /// NVFP4 copy quantized from it under [`Weights::Nvfp4`].
    pub(super) unsafe fn linear(&mut self, name: &str, shape: [u64; 2]) -> Result<Linear> {
        // SAFETY: forwarded genuine mapped immutable-prefix contract from caller.
        let bf16 = unsafe { self.weight(name, shape)? };
        let nvfp4 = match self.weights {
            Weights::Bf16 => None,
            Weights::Nvfp4 => {
                let q = Nvfp4::quantize(&bf16).map_err(|e| e.context(name.to_owned()))?;
                self.nvfp4.0 += 1;
                self.nvfp4.1 += q.bytes();
                Some(q)
            }
        };
        Ok(Linear { bf16, nvfp4 })
    }
}

fn validate_geometry(c: &Config) -> Result<()> {
    fn decoder(c: &DecoderConfig, want: (usize, usize, usize, usize, usize, usize)) -> Result<()> {
        ensure!(
            (
                c.hidden_size,
                c.intermediate_size,
                c.num_hidden_layers,
                c.num_attention_heads,
                c.num_key_value_heads,
                c.head_dim
            ) == want,
            "unsupported bounded Breeze transformer geometry"
        );
        ensure!(
            c.rms_norm_eps.is_finite() && c.rms_norm_eps > 0.0,
            "invalid norm epsilon"
        );
        Ok(())
    }
    decoder(&c.text.decoder, (1152, 6912, 26, 4, 1, 256))?;
    decoder(&c.backbone, (2048, 6144, 28, 16, 8, 128))?;
    decoder(&c.depth.decoder, (1024, 8192, 12, 8, 2, 128))?;
    ensure!(
        c.text.vocab_size == 262158
            && c.text_vocab_size == 262158
            && c.audio_vocab_size == 2051
            && c.num_codebooks == 16,
        "unsupported vocabulary geometry"
    );
    ensure!(
        c.text.layer_types.len() == 26
            && c.text
                .layer_types
                .iter()
                .all(|s| s == "full_attention" || s == "sliding_attention"),
        "invalid encoder layer types"
    );
    ensure!(
        c.text.sliding_window > 0
            && c.text.sliding_window <= 2048
            && c.text.query_pre_attn_scalar.is_finite()
            && c.text.query_pre_attn_scalar > 0.0,
        "invalid encoder attention geometry"
    );
    ensure!(
        c.depth.audio_embed_size == 2048
            && c.depth.backbone_hidden_size == 2048
            && c.tie_codebooks_embeddings
            && c.codebook_eos_token_id == 0,
        "unsupported tied audio embedding geometry"
    );
    ensure!(
        c.max_context >= 1 && c.text.eoi_token_index < 262158,
        "invalid context/EOI identity"
    );
    Ok(())
}
pub fn validate_request(prompt: &GuidedPrompt, o: &GenerationOptions) -> Result<usize> {
    ensure!(
        o.cfg_scale.is_finite() && o.cfg_scale > 0.0,
        "CFG scale must be finite and positive"
    );
    ensure!(
        prompt.negative.is_some() == (o.cfg_scale != 1.0),
        "nonunit CFG needs an explicit negative prompt; CFG1 must not carry an unused branch"
    );
    ensure!(
        (1..=1500).contains(&o.max_frames) && (1..=2048).contains(&o.max_context),
        "invalid bounded frame/context limit"
    );
    ensure!(
        o.temperature.is_finite()
            && o.temperature > 0.0
            && o.top_p.is_finite()
            && o.top_p > 0.0
            && o.top_p <= 1.0
            && o.repetition_penalty.is_finite()
            && o.repetition_penalty > 0.0,
        "invalid sampling options"
    );
    let count = validate_branch(&prompt.conditional, o.max_context)?;
    if let Some(negative) = &prompt.negative {
        validate_branch(negative, o.max_context)?;
    }
    Ok(count)
}

fn validate_branch(prompt: &[PromptSegment], max_context: usize) -> Result<usize> {
    ensure!(!prompt.is_empty(), "empty prompt");
    let mut count = 0usize;
    for segment in prompt {
        let n = match segment {
            PromptSegment::Text(ids) => {
                ensure!(
                    !ids.is_empty() && ids.iter().all(|&id| id < 262158),
                    "empty/out-of-vocabulary text segment"
                );
                ids.len()
            }
            PromptSegment::AudioFrames(frames) => {
                ensure!(
                    !frames.is_empty() && frames.iter().flatten().all(|&code| code < 2048),
                    "empty/reserved reference frame"
                );
                frames.len()
            }
            PromptSegment::AudioEos => 1,
        };
        count = count
            .checked_add(n)
            .ok_or_else(|| anyhow::anyhow!("prompt length overflow"))?;
    }
    ensure!(
        count <= max_context,
        "prompt exceeds bounded context; no truncation"
    );
    Ok(count)
}

/// Per-request state of ONE branch. Separate prefills establish independent
/// RoPE positions/caches even where prompt prefixes are identical. Never alias
/// one branch's KV as the other's state, nor retain these across requests.
struct Branch {
    hidden: Tensor,
    cache: Vec<Kv>,
    past: usize,
}

pub struct Generator {
    config: Config,
    text: TextEncoder,
    backbone: Decoder,
    depth: Depth,
    audio_embedding: Tensor,
    head: Tensor,
}
impl Generator {
    /// # Safety
    /// All selected leaves must be genuine validated mapped pile blobs. Their
    /// backing files INCLUDING preceding partial pages must stay immutable or
    /// append-only and untruncated through CUDA client/runtime STORAGE TEARDOWN,
    /// not merely until Generator/reader/binder drop or synchronization. Runtime
    /// external registrations retain mmap owners. Partial construction failure
    /// also retains registrations; a read-only FD is not external immutability.
    pub unsafe fn from_pile<R: BlobStoreGet>(
        facts: &TribleSet,
        reader: &R,
        ids: Artifacts,
        config: Config,
        aliases: &mut CudaBf16Aliases,
        weights: Weights,
    ) -> Result<Self> {
        validate_geometry(&config)?;
        let started = Instant::now();
        let mut b = Binder {
            facts,
            reader,
            root: ids.model_root,
            aliases,
            weights,
            nvfp4: (0, 0),
        };
        // SAFETY: same genuine pile/root/binder and immutable lifetime premise.
        let generator = unsafe {
            Self {
                text: TextEncoder::bind(&mut b, config.text.clone(), config.backbone.hidden_size)?,
                backbone: Decoder::bind(&mut b, config.backbone.clone(), "backbone_model", true)?,
                depth: Depth::bind(&mut b, config.depth.clone(), config.audio_vocab_size)?,
                audio_embedding: b
                    .weight("depth_decoder.model.embed_tokens.weight", [32816, 2048])?,
                head: b.weight("lm_head.weight", [2052, 2048])?,
                config,
            }
        };
        if weights == Weights::Nvfp4 {
            generator.synchronize()?;
            eprintln!(
                "[breeze] NVFP4 weights: {} projections, {} bytes resident, bound and quantized in {:.3}s",
                b.nvfp4.0,
                b.nvfp4.1,
                started.elapsed().as_secs_f64()
            );
        }
        Ok(generator)
    }
    pub fn synchronize(&self) -> Result<()> {
        cubecl::future::block_on(self.head.client.sync())
            .map_err(|e| anyhow::anyhow!("Breeze CUDA sync: {e:?}"))
    }
    fn prefill(&self, prompt: &[PromptSegment]) -> Result<Branch> {
        let mut merged: Option<Tensor> = None;
        for segment in prompt {
            let x = match segment {
                PromptSegment::Text(ids) => self.text.encode(ids)?,
                PromptSegment::AudioFrames(frames) => {
                    ops::audio_embedding(&self.audio_embedding, frames, 2051)
                }
                PromptSegment::AudioEos => {
                    ops::audio_embedding(&self.audio_embedding, &[[0u16; 16]], 2051)
                }
            };
            merged = Some(ops::append(&x, merged.as_ref()));
        }
        let merged = merged.ok_or_else(|| anyhow::anyhow!("empty merged prompt"))?;
        let past = merged.meta.shape()[1];
        let (hidden, cache) = self.backbone.forward(merged, 0, &[])?;
        Ok(Branch {
            hidden,
            cache,
            past,
        })
    }
    pub fn generate(
        &mut self,
        prompt: &GuidedPrompt,
        options: GenerationOptions,
        mut on_frame: impl FnMut([u16; 16]) -> Result<()>,
    ) -> Result<GenerationReport> {
        let prompt_len = validate_request(prompt, &options)?;
        ensure!(
            options.max_context <= self.config.max_context,
            "request exceeds model context"
        );
        self.synchronize()?;
        let started = Instant::now();
        let mut positive = self.prefill(&prompt.conditional)?;
        let mut negative = prompt
            .negative
            .as_ref()
            .map(|p| self.prefill(p))
            .transpose()?;
        let negative_prompt_tokens = negative.as_ref().map(|branch| branch.past);
        self.synchronize()?;
        let prefill_seconds = started.elapsed().as_secs_f64();
        let decode_start = Instant::now();
        let mut sampler = Sampler::new(options.seed);
        let mut backbone_ids = Vec::with_capacity(options.max_frames);
        let mut frames = 0usize;
        let mut callback_seconds = 0.0f64;
        let termination = loop {
            let logits = ops::head(&positive.hidden, &self.head, None);
            let negative_logits = negative
                .as_ref()
                .map(|b| ops::head(&b.hidden, &self.head, None));
            // The frame's sixteen sampled IDs stay on the device and reach the
            // host in one read at its end. The backbone's code is copied out
            // as soon as it is sampled, but waited for only once the depth
            // decoder's first step is queued behind the copy, so the GPU stays
            // busy across the host's EOS test; on EOS that one step is dropped.
            let ids = self.head.client.empty(16 * 4);
            sampler.sample_guided_into(
                &logits,
                negative_logits.as_ref(),
                &options,
                &backbone_ids,
                true,
                &ids,
                0,
            )?;
            let first = sampling::read_id_later(&self.head.client, &ids, 0)?;
            let depth = self.depth.begin(
                &positive.hidden,
                negative.as_ref().map(|b| &b.hidden),
                &ids,
                &self.audio_embedding,
                &options,
                &mut sampler,
            );
            // Waited for even when `begin` failed: its host buffer is in flight.
            let first = first();
            let depth = depth?;
            let first = first?;
            ensure!(
                first != u32::MAX,
                "nonfinite/empty CUDA sampling distribution"
            );
            backbone_ids.push(first);
            if first == 2051 {
                break Termination::Eos;
            }
            ensure!(first < 2048, "backbone emitted reserved codec ID");
            self.depth
                .finish(depth, &ids, &self.audio_embedding, &options, &mut sampler)?;
            let codes: [u32; 16] = sampling::read_ids(&self.head.client, ids)?;
            let mut frame = [0u16; 16];
            for (code, &id) in frame.iter_mut().zip(&codes) {
                ensure!(id != u32::MAX, "nonfinite/empty CUDA sampling distribution");
                ensure!(id < 2048, "depth sampler emitted reserved code");
                *code = id as u16;
            }
            let callback_start = Instant::now();
            on_frame(frame)?;
            callback_seconds += callback_start.elapsed().as_secs_f64();
            frames += 1;
            if frames == options.max_frames {
                break Termination::FrameLimit;
            }
            if positive.past == options.max_context
                || negative
                    .as_ref()
                    .is_some_and(|b| b.past == options.max_context)
            {
                break Termination::ContextLimit;
            }
            let input = ops::audio_embedding(&self.audio_embedding, &[frame], 2051);
            (positive.hidden, positive.cache) =
                self.backbone
                    .forward(input.clone(), positive.past, &positive.cache)?;
            positive.past += 1;
            if let Some(branch) = &mut negative {
                // Exactly the SAME sampled complete frame goes to both branches.
                (branch.hidden, branch.cache) =
                    self.backbone.forward(input, branch.past, &branch.cache)?;
                branch.past += 1;
            }
        };
        self.synchronize()?;
        let decode_seconds = decode_start.elapsed().as_secs_f64() - callback_seconds;
        Ok(GenerationReport {
            termination,
            frames,
            backbone_ids,
            prefill_seconds,
            decode_seconds,
            callback_seconds,
            total_seconds: started.elapsed().as_secs_f64(),
            cfg_scale: options.cfg_scale,
            conditional_prompt_tokens: prompt_len,
            negative_prompt_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_requests_refuse_missing_cfg_branch_and_truncation() {
        let mut p = GuidedPrompt {
            conditional: vec![
                PromptSegment::Text(vec![2, 100]),
                PromptSegment::AudioFrames(vec![[17; 16]]),
                PromptSegment::AudioEos,
            ],
            negative: None,
        };
        let mut o = GenerationOptions::default();
        assert_eq!(validate_request(&p, &o).unwrap(), 4);
        o.cfg_scale = 4.0;
        assert!(validate_request(&p, &o).is_err());
        p.negative = Some(vec![PromptSegment::Text(vec![2, 101])]);
        assert_eq!(validate_request(&p, &o).unwrap(), 4);
        p.negative = Some(vec![PromptSegment::Text(vec![2; 2049])]);
        assert!(validate_request(&p, &o).is_err());
        p.negative = None;
        o.cfg_scale = 1.0;
        o.max_context = 3;
        assert!(validate_request(&p, &o).is_err());
        o.max_context = 4;
        assert!(validate_request(&p, &o).is_ok());
        let bad = GuidedPrompt {
            conditional: vec![PromptSegment::AudioFrames(vec![[2048; 16]])],
            negative: None,
        };
        assert!(validate_request(&bad, &o).is_err());
        assert!(
            validate_request(
                &GuidedPrompt {
                    conditional: vec![],
                    negative: None
                },
                &o
            )
            .is_err()
        );
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            o.cfg_scale = scale;
            assert!(validate_request(&p, &o).is_err());
        }
    }
}
