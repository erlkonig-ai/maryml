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

/// Generated frames per streaming hop once the stream is going: 8 frames are
/// 667 ms of audio, enough that the cost of a hop's left context is spread
/// over many new frames.
pub const HOP_FRAMES: usize = 8;
/// The hops before the steady [`HOP_FRAMES`]: the first PCM waits for the
/// prefill and two frames, not eight. The ramp grows more slowly than the
/// generator runs ahead of real time (~40 ms of GPU per 83 ms frame), so a
/// player that starts on the first hop is not caught up by any later one,
/// with ~70 ms to spare: the first 2 frames play 167 ms while the next 2 are
/// generated and decoded in ~100 ms, those 4 play 333 ms while the next 3
/// take ~140 ms, and so on up to 8. A steeper 2, 3, 5 measured only 20 to
/// 30 ms to spare before its second hop (sky, nvfp4, 2026-10-11), too little
/// on a GPU shared with hearing.
pub const FIRST_HOPS: [usize; 4] = [2, 2, 3, 5];
/// Frames of left context decoded again before each hop, the value upstream's
/// `chunked_decode` and Mary's Qwen stream use. The decoder is causal, so a
/// window only looks back; 25 frames of its 72-frame attention window is the
/// upstream trade between fidelity at the seams and the cost of a hop.
pub const HOP_CONTEXT: usize = 25;

/// The early hops in effect: [`FIRST_HOPS`], or in the test binary the
/// comma-separated frame counts in `BREEZE_FIRST_HOPS` (empty for none), so
/// that one build can time and be listened to under another ramp.
fn first_hops() -> Vec<usize> {
    #[cfg(test)]
    if let Ok(ramp) = std::env::var("BREEZE_FIRST_HOPS") {
        return ramp
            .split(',')
            .filter(|hop| !hop.trim().is_empty())
            .map(|hop| hop.trim().parse().expect("BREEZE_FIRST_HOPS: frame counts"))
            .filter(|&hop| (1..=HOP_FRAMES).contains(&hop))
            .collect();
    }
    FIRST_HOPS.to_vec()
}

/// The new frames of each hop in order: the early hops, then [`HOP_FRAMES`].
fn hop_sizes() -> impl Iterator<Item = usize> {
    first_hops()
        .into_iter()
        .chain(std::iter::repeat(HOP_FRAMES))
}

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
    /// the hop's whole size, so that every window has one of
    /// [`hop_window_lengths`]. The decoder is causal: the repeats change none
    /// of the real samples.
    pub codes: Vec<[u32; 16]>,
    /// The new frames' samples in the decoded window.
    pub keep: Range<usize>,
}

/// Generated frames, cut into hops as they arrive: the early hops
/// ([`FIRST_HOPS`]), then every [`HOP_FRAMES`] new frames, each make a window
/// with up to [`HOP_CONTEXT`] frames of left context, and
/// [`flush`](Self::flush) makes the last, shorter one. Decoding the windows in
/// order and concatenating their kept samples gives one sample run of exactly
/// `frames × SAMPLES_PER_FRAME`, starting from fresh decoder state at the
/// first generated frame, as [`decode_generated`] does.
#[derive(Debug)]
pub struct Hops {
    frames: Vec<[u32; 16]>,
    emitted: usize,
    first_hops: Vec<usize>,
    /// Hops emitted so far.
    hops: usize,
}

impl Default for Hops {
    fn default() -> Self {
        Self {
            frames: Vec::new(),
            emitted: 0,
            first_hops: first_hops(),
            hops: 0,
        }
    }
}

impl Hops {
    /// Take one generated frame; once the hop now filling is whole, the
    /// window that decodes it.
    pub fn push(&mut self, frame: [u16; 16]) -> Result<Option<HopWindow>> {
        self.frames.push(decoder_frame(&frame)?);
        Ok(self.window(self.size()))
    }

    /// The window for the frames not yet in a hop, at the end of generation.
    pub fn flush(&mut self) -> Option<HopWindow> {
        self.window(1)
    }

    /// The new frames of the hop now filling.
    fn size(&self) -> usize {
        self.first_hops
            .get(self.hops)
            .copied()
            .unwrap_or(HOP_FRAMES)
    }

    fn window(&mut self, at_least: usize) -> Option<HopWindow> {
        let new = self.frames.len() - self.emitted;
        if new < at_least {
            return None;
        }
        let size = self.size();
        let new = new.min(size);
        let context = HOP_CONTEXT.min(self.emitted);
        let start = self.emitted - context;
        let mut codes = self.frames[start..self.emitted + new].to_vec();
        let last = *codes.last().expect("a window holds at least one new frame");
        codes.resize(context + size, last);
        self.emitted += new;
        self.hops += 1;
        Some(HopWindow {
            codes,
            keep: context * SAMPLES_PER_FRAME..(context + new) * SAMPLES_PER_FRAME,
        })
    }
}

/// Every length a hop window has: shorter while the early hops run and the
/// left context fills at the start of an utterance, then
/// `HOP_CONTEXT + HOP_FRAMES` for good. A tail pads to its hop's whole size,
/// so it has its hop's length too. The codec compiles one set of kernels per
/// length, so these few are all.
pub fn hop_window_lengths() -> Vec<usize> {
    let mut lengths: Vec<usize> = Vec::new();
    let (mut emitted, mut previous, early) = (0, 0, first_hops().len());
    for (index, size) in hop_sizes().enumerate() {
        let length = HOP_CONTEXT.min(emitted) + size;
        if index > early && length == previous {
            break;
        }
        if !lengths.contains(&length) {
            lengths.push(length);
        }
        (emitted, previous) = (emitted + size, length);
    }
    lengths
}

/// The new frames in each chunk a stream of `frames` frames is handed on as.
#[cfg(test)]
pub(crate) fn hop_chunks(frames: usize) -> Vec<usize> {
    let mut chunks = Vec::new();
    let mut left = frames;
    for size in hop_sizes() {
        if left == 0 {
            break;
        }
        chunks.push(size.min(left));
        left -= size.min(left);
    }
    chunks
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
        assert_eq!(FIRST_HOPS, [2, 2, 3, 5]);
        assert_eq!(hop_window_lengths(), [2, 4, 7, 12, 20, 28, 33]);
    }

    #[test]
    fn early_hops_ramp_up_to_the_steady_hop() {
        assert_eq!(hop_chunks(1), [1]);
        assert_eq!(hop_chunks(2), [2]);
        assert_eq!(hop_chunks(4), [2, 2]);
        assert_eq!(hop_chunks(12), [2, 2, 3, 5]);
        assert_eq!(hop_chunks(29), [2, 2, 3, 5, 8, 8, 1]);
        assert_eq!(hop_chunks(0), Vec::<usize>::new());
        // Played from the first hop's arrival, every later hop arrives
        // before the audio ahead of it runs out: generation at 40 ms per
        // 83 ms frame plus a decode of 10 ms and 1.6 ms per window frame
        // (measured on sky: 13 ms at 2, 22 ms at 10, 61 ms at 33), with 50 ms
        // to spare.
        let (mut played_ms, mut ready_ms, mut emitted) = (0.0, 0.0, 0);
        for (hop, frames) in hop_chunks(200).into_iter().enumerate() {
            let window = HOP_CONTEXT.min(emitted) + frames;
            ready_ms += 40.0 * frames as f64 + 10.0 + 1.6 * window as f64;
            if hop == 0 {
                played_ms = ready_ms;
            }
            assert!(
                hop == 0 || ready_ms + 50.0 <= played_ms,
                "hop {hop} of {frames}: ready at {ready_ms} ms, playback there at {played_ms} ms"
            );
            played_ms += 1000.0 * (frames * SAMPLES_PER_FRAME) as f64 / f64::from(SAMPLE_RATE);
            emitted += frames;
        }
    }

    #[test]
    fn hops_cover_every_frame_once_with_fixed_window_lengths() {
        for count in [
            1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 26, 27, 32, 33, 34, 40, 41, 54, 100,
        ] {
            let frames = generated(count);
            let (samples, lengths) = stream(&frames);
            assert_eq!(samples.len(), count * SAMPLES_PER_FRAME, "{count} frames");
            assert_eq!(lengths.len(), hop_chunks(count).len(), "{count} frames");
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
    fn the_first_hop_is_two_frames_without_context() {
        let mut hops = Hops::default();
        assert!(hops.push(generated(1)[0]).unwrap().is_none());
        let first = hops.push(generated(2)[1]).unwrap().unwrap();
        assert_eq!(first.codes.len(), 2);
        assert_eq!(first.keep, 0..2 * SAMPLES_PER_FRAME);
        assert!(Hops::default().flush().is_none(), "no frames, no tail");
    }

    #[test]
    fn a_tail_shorter_than_its_hop_pads_to_that_hops_window() {
        // One frame: the first hop's two-frame window.
        let mut hops = Hops::default();
        hops.push(generated(1)[0]).unwrap();
        let tail = hops.flush().unwrap();
        assert_eq!(tail.codes.len(), 2);
        assert_eq!(tail.keep, 0..SAMPLES_PER_FRAME);
        assert_eq!(tail.codes[1], tail.codes[0]);
        // Nine frames: hops of 2, 2 and 3, then 2 of the fourth hop's 5
        // after 7 frames of context.
        let mut hops = Hops::default();
        let windows: Vec<HopWindow> = generated(9)
            .into_iter()
            .filter_map(|frame| hops.push(frame).unwrap())
            .collect();
        assert_eq!(windows.len(), 3);
        let tail = hops.flush().unwrap();
        assert_eq!(tail.codes.len(), 12);
        assert_eq!(tail.keep, 7 * SAMPLES_PER_FRAME..9 * SAMPLES_PER_FRAME);
        assert_eq!(tail.codes[8..], [tail.codes[8]; 4]);
    }

    #[test]
    fn a_hop_keeps_only_its_new_frames_after_full_context() {
        let mut hops = Hops::default();
        let windows: Vec<HopWindow> = generated(52)
            .into_iter()
            .filter_map(|frame| hops.push(frame).unwrap())
            .collect();
        let last = windows.last().unwrap();
        assert_eq!(last.codes.len(), HOP_CONTEXT + HOP_FRAMES);
        assert_eq!(
            last.keep,
            HOP_CONTEXT * SAMPLES_PER_FRAME..(HOP_CONTEXT + HOP_FRAMES) * SAMPLES_PER_FRAME
        );
        assert!(
            hops.flush().is_none(),
            "52 frames are 2 + 2 + 3 + 5 and five whole hops"
        );
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
