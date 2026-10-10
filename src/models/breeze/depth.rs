//! Fifteen additional codebooks per backbone sample; fresh depth KV per frame.
use super::{
    backbone::{Decoder, linear},
    config::DepthConfig,
    cuda_ops as ops,
    generator::{Binder, GenerationOptions},
    nvfp4::Linear,
    sampling::Sampler,
};
use anyhow::{Result, ensure};
use ops::Tensor;
use triblespace::core::repo::BlobStoreGet;

pub(super) struct Depth {
    decoder: Decoder,
    projection: Linear,
    heads: Tensor,
    vocab: usize,
}
impl Depth {
    pub(super) unsafe fn bind<R: BlobStoreGet>(
        b: &mut Binder<'_, R>,
        c: DepthConfig,
        vocab: usize,
    ) -> Result<Self> {
        ensure!(
            c.audio_embed_size == c.backbone_hidden_size,
            "separate depth backbone projector unsupported"
        );
        Ok(unsafe {
            Self {
                projection: b.linear(
                    "depth_decoder.model.inputs_embeds_projector.weight",
                    [c.decoder.hidden_size as u64, c.audio_embed_size as u64],
                )?,
                heads: b.weight(
                    "depth_decoder.codebooks_head.weight",
                    [15, c.decoder.hidden_size as u64, vocab as u64],
                )?,
                decoder: Decoder::bind(b, c.decoder, "depth_decoder.model", false)?,
                vocab,
            }
        })
    }
    pub(super) fn frame(
        &self,
        hidden: &Tensor,
        negative_hidden: Option<&Tensor>,
        first: u32,
        embedding: &Tensor,
        options: &GenerationOptions,
        sampler: &mut Sampler,
    ) -> Result<[u16; 16]> {
        ensure!(first < 2048, "invalid initial codec sample");
        let mut frame = [0u16; 16];
        frame[0] = first as u16;
        let initial = ops::gather(embedding, &[first]);
        let input = ops::append(&initial, Some(hidden));
        let (mut state, mut cache) =
            self.decoder
                .forward(linear(&input, &self.projection)?, 0, &[])?;
        // Fresh depth state for each branch and each frame. The only common
        // inputs are sampled codebook IDs, never a copied conditional KV cache.
        let mut negative = negative_hidden
            .map(|hidden| {
                let input = ops::append(&initial, Some(hidden));
                self.decoder
                    .forward(linear(&input, &self.projection)?, 0, &[])
            })
            .transpose()?;
        for head in 0..15 {
            let logits = ops::head(&state, &self.heads, Some(head));
            let negative_logits = negative
                .as_ref()
                .map(|(state, _)| ops::head(state, &self.heads, Some(head)));
            let code =
                sampler.sample_guided(&logits, negative_logits.as_ref(), options, &[], false)?;
            ensure!(code < 2048, "depth sampler emitted reserved code");
            frame[head + 1] = code as u16;
            if head < 14 {
                let row = (head + 1) * self.vocab + code as usize;
                let next = ops::gather(embedding, &[row as u32]);
                let next = linear(&next, &self.projection)?;
                (state, cache) = self.decoder.forward(next.clone(), head + 2, &cache)?;
                if let Some((state, cache)) = &mut negative {
                    (*state, *cache) = self.decoder.forward(next, head + 2, cache)?;
                }
            }
        }
        Ok(frame)
    }
}
