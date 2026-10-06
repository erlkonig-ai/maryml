//! Fifteen additional codebooks per backbone sample; fresh depth KV per frame.
use super::{
    backbone::{Decoder, linear},
    config::DepthConfig,
    cuda_ops as ops,
    generator::{Binder, GenerationOptions},
    sampling::Sampler,
};
use anyhow::{Result, ensure};
use ops::Tensor;
use triblespace::core::repo::BlobStoreGet;

pub(super) struct Depth {
    decoder: Decoder,
    projection: Tensor,
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
                projection: b.weight(
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
        for head in 0..15 {
            let logits = ops::head(&state, &self.heads, Some(head));
            let code = sampler.sample(&logits, options, &[], false)?;
            ensure!(code < 2048, "depth sampler emitted reserved code");
            frame[head + 1] = code as u16;
            if head < 14 {
                let row = (head + 1) * self.vocab + code as usize;
                let next = ops::gather(embedding, &[row as u32]);
                (state, cache) =
                    self.decoder
                        .forward(linear(&next, &self.projection)?, head + 2, &cache)?;
            }
        }
        Ok(frame)
    }
}
