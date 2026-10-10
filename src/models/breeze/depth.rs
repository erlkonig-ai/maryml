//! Fifteen additional codebooks per backbone sample; fresh depth KV per frame.
use super::{
    backbone::{Decoder, Kv, linear},
    config::DepthConfig,
    cuda_ops as ops,
    generator::{Binder, GenerationOptions},
    sampling::Sampler,
};
use anyhow::{Result, ensure};
use cubecl::server::Handle;
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
    /// Launches the frame's first codebook step from the backbone's code in
    /// `ids[0]` and samples it into `ids[1]`, reading nothing back. The caller
    /// tests `ids[0]` for EOS while this step runs; on EOS it drops the frame.
    pub(super) fn begin(
        &self,
        hidden: &Tensor,
        negative_hidden: Option<&Tensor>,
        ids: &Handle,
        embedding: &Tensor,
        options: &GenerationOptions,
        sampler: &mut Sampler,
    ) -> Result<DepthFrame> {
        let initial = ops::gather_sampled(embedding, ids, 0, 0);
        let input = ops::append(&initial, Some(hidden));
        let (state, cache) = self
            .decoder
            .forward(linear(&input, &self.projection)?, 0, &[])?;
        // Fresh depth state for each branch and each frame. The only common
        // inputs are sampled codebook IDs, never a copied conditional KV cache.
        let negative = negative_hidden
            .map(|hidden| {
                let input = ops::append(&initial, Some(hidden));
                self.decoder
                    .forward(linear(&input, &self.projection)?, 0, &[])
            })
            .transpose()?;
        let frame = DepthFrame {
            state,
            cache,
            negative,
        };
        self.sample(&frame, 0, ids, options, sampler)?;
        Ok(frame)
    }
    /// Launches the remaining fourteen steps into `ids[2..16]`. Each step's
    /// codebook input is gathered on the device from the previous step's
    /// sampled ID; the caller reads the frame's sixteen IDs once afterwards
    /// and refuses invalid codes there.
    pub(super) fn finish(
        &self,
        mut frame: DepthFrame,
        ids: &Handle,
        embedding: &Tensor,
        options: &GenerationOptions,
        sampler: &mut Sampler,
    ) -> Result<()> {
        for head in 1..15 {
            let next = ops::gather_sampled(embedding, ids, head, head * self.vocab);
            let next = linear(&next, &self.projection)?;
            (frame.state, frame.cache) =
                self.decoder.forward(next.clone(), head + 1, &frame.cache)?;
            if let Some((state, cache)) = &mut frame.negative {
                (*state, *cache) = self.decoder.forward(next, head + 1, cache)?;
            }
            self.sample(&frame, head, ids, options, sampler)?;
        }
        Ok(())
    }
    fn sample(
        &self,
        frame: &DepthFrame,
        head: usize,
        ids: &Handle,
        options: &GenerationOptions,
        sampler: &mut Sampler,
    ) -> Result<()> {
        let logits = ops::head(&frame.state, &self.heads, Some(head));
        let negative_logits = frame
            .negative
            .as_ref()
            .map(|(state, _)| ops::head(state, &self.heads, Some(head)));
        sampler.sample_guided_into(
            &logits,
            negative_logits.as_ref(),
            options,
            &[],
            false,
            ids,
            head + 1,
        )
    }
}

/// One frame's depth state between its first codebook step and the rest.
pub(super) struct DepthFrame {
    state: Tensor,
    cache: Vec<Kv>,
    negative: Option<(Tensor, Vec<Kv>)>,
}
