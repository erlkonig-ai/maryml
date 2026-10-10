//! Fresh generated-only acoustic output. Never prefix the reference codes.
//!
//! Two decodes of the same generated frames: [`decode_generated`], the batch
//! of a whole utterance, and [`Hops`], fixed-size windows decoded while the
//! frames are still being generated.
use anyhow::{Context, Result, ensure};
use burn::prelude::Backend;
use std::{fs::OpenOptions, io::Write, ops::Range, path::Path};

use super::reference::SAMPLE_RATE;
use crate::models::qwen3tts::{codec::CodecDecoder, config::SAMPLES_PER_FRAME};

/// Generated frames per streaming hop: 8 frames are 667 ms of audio, so the
/// first PCM waits for the prefill and eight frames, not the utterance.
pub const HOP_FRAMES: usize = 8;
/// Frames of left context decoded again before each hop, the value upstream's
/// `chunked_decode` and Mary's Qwen stream use. The decoder is causal, so a
/// window only looks back; 25 frames of its 72-frame attention window is the
/// upstream trade between fidelity at the seams and the cost of a hop.
pub const HOP_CONTEXT: usize = 25;

fn decoder_frame(frame: &[u16; 16]) -> Result<[u32; 16]> {
    ensure!(
        frame.iter().all(|&code| code < 2048),
        "control/padding/out-of-range token reached acoustic decoder"
    );
    Ok(frame.map(u32::from))
}

fn decoder_frames(frames: &[[u16; 16]]) -> Result<Vec<[u32; 16]>> {
    ensure!(!frames.is_empty(), "generation produced no speech frames");
    frames.iter().map(decoder_frame).collect()
}

/// One hop to decode: the codes of a window and which of its samples are new.
#[derive(Debug)]
pub struct HopWindow {
    /// Left context, the hop's new frames, then the last frame repeated up to
    /// a whole hop, so that every window has one of [`hop_window_lengths`].
    /// The decoder is causal: the repeats change none of the real samples.
    pub codes: Vec<[u32; 16]>,
    /// The new frames' samples in the decoded window.
    pub keep: Range<usize>,
}

/// Generated frames, cut into hops as they arrive: every [`HOP_FRAMES`] new
/// frames make a window with up to [`HOP_CONTEXT`] frames of left context,
/// and [`flush`](Self::flush) makes the last, shorter one. Decoding the
/// windows in order and concatenating their kept samples gives one sample run
/// of exactly `frames × SAMPLES_PER_FRAME`, starting from fresh decoder state
/// at the first generated frame, as [`decode_generated`] does.
#[derive(Debug, Default)]
pub struct Hops {
    frames: Vec<[u32; 16]>,
    emitted: usize,
}

impl Hops {
    /// Take one generated frame; once a whole hop of new frames waits, the
    /// window that decodes it.
    pub fn push(&mut self, frame: [u16; 16]) -> Result<Option<HopWindow>> {
        self.frames.push(decoder_frame(&frame)?);
        Ok(self.window(HOP_FRAMES))
    }

    /// The window for the frames not yet in a hop, at the end of generation.
    pub fn flush(&mut self) -> Option<HopWindow> {
        self.window(1)
    }

    fn window(&mut self, at_least: usize) -> Option<HopWindow> {
        let new = self.frames.len() - self.emitted;
        if new < at_least {
            return None;
        }
        let new = new.min(HOP_FRAMES);
        let context = HOP_CONTEXT.min(self.emitted);
        let start = self.emitted - context;
        let mut codes = self.frames[start..self.emitted + new].to_vec();
        let last = *codes.last().expect("a window holds at least one new frame");
        codes.resize(context + HOP_FRAMES, last);
        self.emitted += new;
        Some(HopWindow {
            codes,
            keep: context * SAMPLES_PER_FRAME..(context + new) * SAMPLES_PER_FRAME,
        })
    }
}

/// Every length a hop window has: shorter while the left context fills at
/// the start of an utterance, then `HOP_CONTEXT + HOP_FRAMES` for good. The
/// codec compiles one set of kernels per length, so these few are all.
pub fn hop_window_lengths() -> Vec<usize> {
    let mut lengths: Vec<usize> = Vec::new();
    for emitted in (0..).step_by(HOP_FRAMES) {
        let length = HOP_CONTEXT.min(emitted) + HOP_FRAMES;
        if lengths.last() == Some(&length) {
            break;
        }
        lengths.push(length);
    }
    lengths
}

/// Decode one hop and keep its new samples.
pub fn decode_hop<B: Backend>(
    decoder: &CodecDecoder<B>,
    window: &HopWindow,
    device: &B::Device,
) -> Result<Vec<f32>> {
    let samples = decoder.decode(&window.codes, device);
    ensure!(
        samples.len() == window.codes.len() * SAMPLES_PER_FRAME,
        "codec returned unexpected sample count"
    );
    let samples = samples[window.keep.clone()].to_vec();
    ensure!(
        samples.iter().all(|sample| sample.is_finite()),
        "codec returned nonfinite samples"
    );
    Ok(samples)
}

/// First endpoint uses the shared tokenizer's 300-frame/25-context batch path,
/// not Breeze's stateful streaming codec. Decoder state begins at generated t=0.
pub fn decode_generated<B: Backend>(
    decoder: &CodecDecoder<B>,
    frames: &[[u16; 16]],
    device: &B::Device,
) -> Result<Vec<f32>> {
    let frames = decoder_frames(frames)?;
    let expected = frames
        .len()
        .checked_mul(SAMPLES_PER_FRAME)
        .context("sample count overflow")?;
    let samples = decoder.chunked_decode(&frames, device);
    ensure!(
        samples.len() == expected,
        "codec returned unexpected sample count"
    );
    ensure!(
        samples.iter().all(|sample| sample.is_finite()),
        "codec returned nonfinite samples"
    );
    Ok(samples)
}

/// FLOAT WAV retains the actual generated waveform. Exclusive creation keeps
/// previous audio intact; no normalization, fade or PCM16 roundtrip is applied.
pub fn write_float_wav_new(path: &Path, samples: &[f32]) -> Result<()> {
    let bytes = float_wav(samples)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create new WAV {}", path.display()))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

pub fn float_wav(samples: &[f32]) -> Result<Vec<u8>> {
    ensure!(
        !samples.is_empty() && samples.iter().all(|x| x.is_finite()),
        "WAV requires nonempty finite samples"
    );
    let size = samples
        .len()
        .checked_mul(4)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n <= u32::MAX - 36)
        .context("WAV exceeds RIFF size limit")?;
    let mut bytes = Vec::with_capacity(size as usize + 44);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(size + 36).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&3u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 4).to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&32u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&size.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_only_frames_keep_zero_codes_and_last_frame() {
        let generated = [[0u16; 16], [2047; 16]];
        let decoded = decoder_frames(&generated).unwrap();
        assert_eq!(decoded, [[0u32; 16], [2047; 16]]);
        assert_eq!(decoded.len() * SAMPLES_PER_FRAME, 3840);
        // There is deliberately no reference-code argument to this conversion.
    }

    #[test]
    fn control_rows_are_not_clamped_into_speech() {
        assert!(decoder_frames(&[]).is_err());
        assert!(decoder_frames(&[[2050; 16]]).is_err());
        let mut mixed = [1u16; 16];
        mixed[15] = 2048;
        assert!(decoder_frames(&[mixed]).is_err());
    }

    /// A causal stand-in for the codec: each frame's samples depend on it
    /// and the three frames before it in the window, so a window with fewer
    /// than three frames of history differs, as the real decoder's would.
    fn causal(codes: &[[u32; 16]]) -> Vec<f32> {
        let mut samples = Vec::with_capacity(codes.len() * SAMPLES_PER_FRAME);
        for (t, frame) in codes.iter().enumerate() {
            let history: u32 = codes[t.saturating_sub(3)..t].iter().map(|f| f[0]).sum();
            let value = (frame[0] * 1000 + frame[15] + history * 7) as f32;
            samples.extend((0..SAMPLES_PER_FRAME).map(|i| value + i as f32 * 1e-3));
        }
        samples
    }

    fn generated(count: usize) -> Vec<[u16; 16]> {
        (0..count)
            .map(|t| {
                let mut frame = [(t * 37 % 2048) as u16; 16];
                frame[15] = (t * 11 % 2048) as u16;
                frame
            })
            .collect()
    }

    /// Stream `frames` through hops with `decode`; the kept samples and the
    /// windows' lengths.
    fn stream(frames: &[[u16; 16]]) -> (Vec<f32>, Vec<usize>) {
        let mut hops = Hops::default();
        let (mut samples, mut lengths) = (Vec::new(), Vec::new());
        let mut take = |window: HopWindow| {
            lengths.push(window.codes.len());
            samples.extend_from_slice(&causal(&window.codes)[window.keep]);
        };
        for &frame in frames {
            if let Some(window) = hops.push(frame).unwrap() {
                take(window);
            }
        }
        if let Some(window) = hops.flush() {
            take(window);
        }
        assert!(hops.flush().is_none(), "the tail is flushed once");
        (samples, lengths)
    }

    #[test]
    fn hop_windows_have_a_few_fixed_lengths() {
        assert_eq!(hop_window_lengths(), [8, 16, 24, 32, 33]);
    }

    #[test]
    fn hops_cover_every_frame_once_with_fixed_window_lengths() {
        for count in [1, 5, 7, 8, 9, 15, 16, 17, 32, 33, 40, 41, 54, 100] {
            let frames = generated(count);
            let (samples, lengths) = stream(&frames);
            assert_eq!(samples.len(), count * SAMPLES_PER_FRAME, "{count} frames");
            assert_eq!(lengths.len(), count.div_ceil(HOP_FRAMES), "{count} frames");
            assert!(
                lengths
                    .iter()
                    .all(|length| hop_window_lengths().contains(length)),
                "{count} frames: {lengths:?}"
            );
            // Context (25) covers the stand-in's history (3), and the tail's
            // repeated frames come after every real one: streaming equals
            // decoding the whole utterance at once.
            let codes: Vec<[u32; 16]> = frames.iter().map(|f| f.map(u32::from)).collect();
            assert_eq!(samples, causal(&codes), "{count} frames");
        }
    }

    #[test]
    fn an_utterance_shorter_than_a_hop_is_one_padded_tail() {
        let mut hops = Hops::default();
        for frame in generated(5) {
            assert!(hops.push(frame).unwrap().is_none());
        }
        let tail = hops.flush().unwrap();
        assert_eq!(tail.codes.len(), HOP_FRAMES);
        assert_eq!(tail.keep, 0..5 * SAMPLES_PER_FRAME);
        assert_eq!(tail.codes[4..], [tail.codes[4]; 4]);
        assert!(Hops::default().flush().is_none(), "no frames, no tail");
    }

    #[test]
    fn a_hop_keeps_only_its_new_frames_after_full_context() {
        let mut hops = Hops::default();
        let windows: Vec<HopWindow> = generated(48)
            .into_iter()
            .filter_map(|frame| hops.push(frame).unwrap())
            .collect();
        let last = windows.last().unwrap();
        assert_eq!(last.codes.len(), HOP_CONTEXT + HOP_FRAMES);
        assert_eq!(
            last.keep,
            HOP_CONTEXT * SAMPLES_PER_FRAME..(HOP_CONTEXT + HOP_FRAMES) * SAMPLES_PER_FRAME
        );
        assert!(hops.flush().is_none(), "48 frames are six whole hops");
        assert!(Hops::default().push([2048; 16]).is_err());
    }

    #[test]
    fn output_float_wav_roundtrip_has_no_gain_or_quantization() {
        let samples = [0.0, 0.12345679, -0.8, 1.25];
        let bytes = float_wav(&samples).unwrap();
        assert_eq!(
            super::super::reference::decode_wav(&bytes).unwrap(),
            samples
        );
        assert!(float_wav(&[f32::INFINITY]).is_err());
    }
}
