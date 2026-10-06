//! Native reference input. No ASR, resampling, gain change or oracle code file.
use anyhow::{Context, Result, ensure};
use std::{fs::File, io::Read, path::Path};

use crate::models::qwen3tts::{config::SAMPLES_PER_FRAME, encoder::CodecEncoder};

pub const SAMPLE_RATE: u32 = 24_000;
pub const MAX_REFERENCE_SAMPLES: usize = SAMPLE_RATE as usize * 30;
const MAX_WAV_BYTES: u64 = 16 * 1024 * 1024;

/// Read a bounded 24 kHz PCM16/FLOAT32 WAV, mean-downmixing channels once.
/// Other sample rates are explicit errors: we do not substitute a resampler.
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("open reference {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_WAV_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_WAV_BYTES,
        "reference WAV exceeds 16 MiB"
    );
    decode_wav(&bytes)
}

pub fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len() >= 12, "truncated WAV header");
    ensure!(
        &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "expected RIFF/WAVE"
    );
    let declared = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    ensure!(
        declared.checked_add(8) == Some(bytes.len()),
        "RIFF length mismatch"
    );
    let mut format = None;
    let mut data = None;
    let mut pos = 12usize;
    while pos < bytes.len() {
        let head = bytes
            .get(pos..pos + 8)
            .context("truncated WAV chunk header")?;
        let size = u32::from_le_bytes(head[4..8].try_into()?) as usize;
        let start = pos + 8;
        let end = start
            .checked_add(size)
            .context("WAV chunk length overflow")?;
        let body = bytes.get(start..end).context("truncated WAV chunk")?;
        match &head[..4] {
            b"fmt " => {
                ensure!(
                    format.is_none() && body.len() >= 16,
                    "invalid/duplicate WAV format"
                );
                format = Some(body);
            }
            b"data" => {
                ensure!(data.is_none(), "multiple WAV data chunks are unsupported");
                data = Some(body);
            }
            _ => {}
        }
        pos = end
            .checked_add(size & 1)
            .context("WAV alignment overflow")?;
        ensure!(pos <= bytes.len(), "missing WAV chunk alignment byte");
    }
    let fmt = format.context("WAV has no format")?;
    let data = data.context("WAV has no samples")?;
    let encoding = u16::from_le_bytes(fmt[0..2].try_into()?);
    let channels = u16::from_le_bytes(fmt[2..4].try_into()?) as usize;
    let rate = u32::from_le_bytes(fmt[4..8].try_into()?);
    let byte_rate = u32::from_le_bytes(fmt[8..12].try_into()?);
    let align = u16::from_le_bytes(fmt[12..14].try_into()?) as usize;
    let bits = u16::from_le_bytes(fmt[14..16].try_into()?);
    ensure!(
        rate == SAMPLE_RATE,
        "reference must be 24 kHz; no implicit resampling"
    );
    ensure!(
        (1..=32).contains(&channels),
        "unsupported WAV channel count"
    );
    ensure!(
        (encoding == 1 && bits == 16) || (encoding == 3 && bits == 32),
        "reference must be PCM16 or IEEE FLOAT32 WAV"
    );
    let width = bits as usize / 8;
    ensure!(
        align == channels * width && byte_rate as usize == rate as usize * align,
        "invalid WAV sample layout"
    );
    ensure!(
        !data.is_empty() && data.len() % align == 0,
        "empty/partial WAV frame"
    );
    let frames = data.len() / align;
    ensure!(
        frames <= MAX_REFERENCE_SAMPLES,
        "reference exceeds 30 seconds"
    );
    let mut mono = Vec::with_capacity(frames);
    for frame in data.chunks_exact(align) {
        let mut sum = 0.0f32;
        for channel in frame.chunks_exact(width) {
            let value = if encoding == 1 {
                i16::from_le_bytes(channel.try_into()?) as f32 / 32768.0
            } else {
                f32::from_le_bytes(channel.try_into()?)
            };
            ensure!(value.is_finite(), "reference contains nonfinite samples");
            sum += value;
        }
        let value = sum / channels as f32;
        ensure!(value.is_finite(), "reference downmix overflow");
        mono.push(value);
    }
    Ok(mono)
}

/// Existing native Rust Mimi encoder is CPU-resident. Its cold/load and encode
/// time must be reported separately from CUDA generation and acoustic decode.
pub fn encode_reference(encoder: &CodecEncoder, samples: &[f32]) -> Result<Vec<[u16; 16]>> {
    ensure!(
        !samples.is_empty() && samples.len() <= MAX_REFERENCE_SAMPLES,
        "reference must contain 1..=720000 samples"
    );
    ensure!(
        samples.iter().all(|x| x.is_finite()),
        "nonfinite reference samples"
    );
    let codes = encoder.encode(samples);
    ensure!(
        codes.len() == samples.len().div_ceil(SAMPLES_PER_FRAME),
        "reference encoder returned an unexpected frame count"
    );
    codes
        .into_iter()
        .map(|frame| {
            ensure!(
                frame.iter().all(|&code| code < 2048),
                "reference code outside codec vocabulary"
            );
            Ok(frame.map(|code| code as u16))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(format: u16, channels: u16, bits: u16, data: &[u8]) -> Vec<u8> {
        let align = channels * bits / 8;
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&format.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        bytes.extend_from_slice(&(SAMPLE_RATE * align as u32).to_le_bytes());
        bytes.extend_from_slice(&align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn float_reference_preserves_samples_and_mean_downmix() {
        let samples = [0.5f32, -0.25, 0.125, 0.75];
        let bytes: Vec<_> = samples.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(decode_wav(&wav(3, 1, 32, &bytes)).unwrap(), samples);
        assert_eq!(decode_wav(&wav(3, 2, 32, &bytes)).unwrap(), [0.125, 0.4375]);
    }

    #[test]
    fn pcm16_scales_without_normalization() {
        let bytes: Vec<_> = [-32768i16, 0, 16384]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(
            decode_wav(&wav(1, 1, 16, &bytes)).unwrap(),
            [-1.0, 0.0, 0.5]
        );
    }

    #[test]
    fn malformed_or_nonfinite_reference_fails_before_encoder() {
        let mut bytes = wav(3, 1, 32, &0.25f32.to_le_bytes());
        for length in 0..bytes.len() {
            assert!(decode_wav(&bytes[..length]).is_err());
        }
        bytes[24..28].copy_from_slice(&48000u32.to_le_bytes());
        assert!(decode_wav(&bytes).is_err());
        assert!(decode_wav(&wav(3, 1, 32, &f32::NAN.to_le_bytes())).is_err());
        assert!(decode_wav(&wav(3, 1, 32, &[])).is_err());
        assert!(decode_wav(&wav(1, 2, 16, &[0, 0])).is_err());
    }
}
