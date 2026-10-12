//! Shared bits for the ASR evaluation tools: audio decoding to 16 kHz mono,
//! text normalisation, word/char edit distance.

use anyhow::{Context, Result};

pub const RATE: u32 = 16_000;

/// FNV-1a 64 then the splitmix64 finaliser: a stable order for subsampling
/// by id that does not cluster neighbouring ids (plain FNV-1a does, since its
/// last bytes barely reach the high bits).
pub fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

/// Decode any symphonia-supported container in `bytes` to mono f32 at its own rate.
pub fn decode(bytes: Vec<u8>, hint_ext: &str) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    hint.with_extension(hint_ext);
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("probe")?;
    let mut format = probed.format;
    let track = format.default_track().context("no track")?.clone();
    if track.codec_params.codec == symphonia::core::codecs::CODEC_TYPE_OPUS {
        return decode_opus(format, track.id).map(|x| (x, RATE));
    }
    let mut dec = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let rate = track.codec_params.sample_rate.context("no rate")?;
    let mut out = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track.id {
            continue;
        }
        let buf = dec.decode(&packet)?;
        let spec = *buf.spec();
        let ch = spec.channels.count();
        let mut sb = SampleBuffer::<f32>::new(buf.capacity() as u64, spec);
        sb.copy_interleaved_ref(buf);
        for frame in sb.samples().chunks(ch) {
            out.push(frame.iter().sum::<f32>() / ch as f32);
        }
    }
    Ok((out, rate))
}

unsafe extern "C" {
    fn opus_decoder_create(fs: i32, channels: i32, error: *mut i32) -> *mut std::ffi::c_void;
    fn opus_decode_float(st: *mut std::ffi::c_void, data: *const u8, len: i32, pcm: *mut f32, frame_size: i32, fec: i32) -> i32;
    fn opus_decoder_destroy(st: *mut std::ffi::c_void);
}

/// Ogg/Opus packets through libopus, decoded natively at 16 kHz mono, with
/// the stream's pre-skip (OpusHead, in 48 kHz samples) dropped.
fn decode_opus(mut format: Box<dyn symphonia::core::formats::FormatReader>, track: u32) -> Result<Vec<f32>> {
    use symphonia::core::errors::Error;
    let pre_skip = format
        .tracks()
        .iter()
        .find(|t| t.id == track)
        .and_then(|t| t.codec_params.extra_data.as_ref())
        .filter(|h| h.len() >= 12 && h.starts_with(b"OpusHead"))
        .map_or(312, |h| u16::from_le_bytes([h[10], h[11]]) as usize);
    let mut err = 0;
    let st = unsafe { opus_decoder_create(RATE as i32, 1, &mut err) };
    anyhow::ensure!(!st.is_null() && err == 0, "opus_decoder_create: {err}");
    let mut out = Vec::new();
    let mut pcm = vec![0f32; 5760];
    let result = (|| -> Result<()> {
        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            };
            if packet.track_id() != track {
                continue;
            }
            let n = unsafe { opus_decode_float(st, packet.data.as_ptr(), packet.data.len() as i32, pcm.as_mut_ptr(), pcm.len() as i32, 0) };
            anyhow::ensure!(n >= 0, "opus_decode_float: {n}");
            out.extend_from_slice(&pcm[..n as usize]);
        }
        Ok(())
    })();
    unsafe { opus_decoder_destroy(st) };
    result?;
    let skip = (pre_skip / 3).min(out.len());
    Ok(out.split_off(skip))
}

/// Resample to 16 kHz if needed (rubato sinc).
pub fn to_16k(x: Vec<f32>, rate: u32) -> Result<Vec<f32>> {
    if rate == RATE {
        return Ok(x);
    }
    use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut r = SincFixedIn::<f32>::new(RATE as f64 / rate as f64, 2.0, params, x.len(), 1)?;
    let out = r.process(&[x], None)?;
    Ok(out.into_iter().next().unwrap())
}

pub fn write_wav(path: &std::path::Path, x: &[f32]) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate: RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &v in x {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// Normalise for scoring: lowercase, punctuation to spaces (apostrophes and
/// hyphens included), whitespace collapsed. Numbers are kept as written.
pub fn normalise(t: &str) -> Vec<String> {
    let lower = t.to_lowercase();
    let mut s = String::with_capacity(lower.len());
    for c in lower.chars() {
        if c.is_alphanumeric() {
            s.push(c);
        } else {
            s.push(' ');
        }
    }
    s.split_whitespace().map(str::to_string).collect()
}

/// Levenshtein distance between token sequences.
pub fn edits<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + (x != y) as usize);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn norm_and_edits() {
        assert_eq!(normalise("Hello, World! It's 1990 — Straße."), ["hello", "world", "it", "s", "1990", "straße"]);
        assert_eq!(normalise("ÄRGER"), ["ärger"]);
        let r = normalise("the cat sat on the mat");
        let h = normalise("the cat sat on mat now");
        assert_eq!(edits(&r, &h), 2);
        assert_eq!(edits(&[1, 2, 3], &[] as &[i32]), 3);
    }
}
