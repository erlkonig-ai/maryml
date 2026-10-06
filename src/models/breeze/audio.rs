//! Fresh generated-only acoustic output. Never prefix the reference codes.
use anyhow::{Context, Result, ensure};
use burn::prelude::Backend;
use std::{fs::OpenOptions, io::Write, path::Path};

use super::reference::SAMPLE_RATE;
use crate::models::qwen3tts::{codec::CodecDecoder, config::SAMPLES_PER_FRAME};

fn decoder_frames(frames: &[[u16; 16]]) -> Result<Vec<[u32; 16]>> {
    ensure!(!frames.is_empty(), "generation produced no speech frames");
    frames
        .iter()
        .map(|frame| {
            ensure!(
                frame.iter().all(|&code| code < 2048),
                "control/padding/out-of-range token reached acoustic decoder"
            );
            Ok(frame.map(u32::from))
        })
        .collect()
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
