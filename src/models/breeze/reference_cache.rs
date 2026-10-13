//! The encoded reference, kept on disk between starts.
//!
//! [`super::reference::encode_reference`] is a pure function of the reference
//! clip and the codec encoder's weights, and it runs on the CPU: 21 to 24 s
//! for the live 11 s clip at every bot start (sky, 2026-10-12/13), out of a
//! 43 to 45 s cold start. Its output is the same 139 frames of codes each
//! time, so the first start after a change computes them and every later
//! start reads them back.
//!
//! The key names everything the codes depend on (the clip's bytes, the
//! encoder weight root, the encoder code version) and the transcript they
//! are spoken against, so a different clip, transcript, weight root or
//! encoder is a miss. A file that is unreadable, carries another key, fails
//! its checksum or disagrees with the clip's frame count is a miss as well.
//! A miss is only the cold path: compute, then write for the next start.
//! Nothing here can substitute one voice's codes for another's.
use anyhow::{Context, Result, bail, ensure};
use std::{
    fmt,
    path::{Path, PathBuf},
};
use triblespace::core::id::Id;

/// Part of every key. Bump it when `CodecEncoder::encode`'s arithmetic
/// changes: the cache holds that function's outputs.
pub const ENCODER_CODE_VERSION: &str = "mimi-encoder-cpu/1";
const MAGIC: &[u8; 8] = b"BRZREF01";
const HEADER_LEN: usize = MAGIC.len() + 32 + 4;
const CHECKSUM_LEN: usize = 32;
const FRAME_LEN: usize = 16 * 2;

/// blake3 over the length-prefixed inputs the codes depend on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ReferenceKey([u8; 32]);

impl ReferenceKey {
    /// `wav_bytes` is the clip file as read, not its decoded samples;
    /// `codec_root` is the model root the encoder's weights are loaded from.
    pub fn new(wav_bytes: &[u8], reference_text: &str, codec_root: Id) -> Self {
        let mut hasher = blake3::Hasher::new();
        for part in [
            ENCODER_CODE_VERSION.as_bytes(),
            wav_bytes,
            reference_text.as_bytes(),
            &codec_root.raw()[..],
        ] {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
        Self(*hasher.finalize().as_bytes())
    }

    pub fn path_in(&self, dir: &Path) -> PathBuf {
        dir.join(format!("{self}.codes"))
    }
}

impl fmt::Display for ReferenceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for ReferenceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReferenceKey({self})")
    }
}

/// Where the codes came from, for the one log line a load writes.
#[derive(Debug, PartialEq, Eq)]
pub enum CacheOutcome {
    /// Read back from `path`; the encoder was not loaded.
    Hit(PathBuf),
    /// Computed, because of `reason`, and written to `path`.
    Written { reason: String, path: PathBuf },
    /// Computed, because of `reason`, and the write failed as `error` says.
    Unwritten { reason: String, error: String },
    /// Computed; no directory was configured.
    Disabled,
}

impl fmt::Display for CacheOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hit(path) => write!(f, "hit, read from {}", path.display()),
            Self::Written { reason, path } => {
                write!(
                    f,
                    "miss ({reason}); computed and written to {}",
                    path.display()
                )
            }
            Self::Unwritten { reason, error } => {
                write!(f, "miss ({reason}); computed, not written: {error}")
            }
            Self::Disabled => write!(f, "no cache directory; computed"),
        }
    }
}

/// The codes under `dir` for `key`, or `encode()`'s, stored for next time.
/// `expected_frames` is the clip's frame count, which a stored file must
/// agree with. `encode` runs only on a miss. Nothing about the cache is a
/// hard error: every problem with it is a miss, and a write that fails
/// leaves the computed codes as they are.
pub fn encode_reference_cached(
    dir: Option<&Path>,
    key: &ReferenceKey,
    expected_frames: usize,
    encode: impl FnOnce() -> Result<Vec<[u16; 16]>>,
) -> Result<(Vec<[u16; 16]>, CacheOutcome)> {
    let Some(dir) = dir else {
        return Ok((encode()?, CacheOutcome::Disabled));
    };
    let path = key.path_in(dir);
    let reason = match std::fs::read(&path) {
        Ok(bytes) => match decode(&bytes, key, expected_frames) {
            Ok(codes) => return Ok((codes, CacheOutcome::Hit(path))),
            Err(error) => format!("{error}"),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => "no file yet".to_owned(),
        Err(error) => format!("unreadable: {error}"),
    };
    let codes = encode()?;
    ensure!(
        codes.len() == expected_frames,
        "reference encoder returned {} frames for a clip of {expected_frames}",
        codes.len()
    );
    let outcome = match write_atomically(&path, &encode_file(key, &codes)) {
        Ok(()) => CacheOutcome::Written { reason, path },
        Err(error) => CacheOutcome::Unwritten {
            reason,
            error: format!("{error:#}"),
        },
    };
    Ok((codes, outcome))
}

/// `MAGIC`, the key, the frame count, the codes little-endian, and a blake3
/// of all of that.
fn encode_file(key: &ReferenceKey, codes: &[[u16; 16]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_LEN + codes.len() * FRAME_LEN + CHECKSUM_LEN);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&key.0);
    bytes.extend_from_slice(&(codes.len() as u32).to_le_bytes());
    for frame in codes {
        for code in frame {
            bytes.extend_from_slice(&code.to_le_bytes());
        }
    }
    let checksum = blake3::hash(&bytes);
    bytes.extend_from_slice(checksum.as_bytes());
    bytes
}

fn decode(bytes: &[u8], key: &ReferenceKey, expected_frames: usize) -> Result<Vec<[u16; 16]>> {
    ensure!(bytes.len() >= HEADER_LEN + CHECKSUM_LEN, "file too short");
    ensure!(&bytes[..MAGIC.len()] == MAGIC, "not a reference code file");
    let (body, checksum) = bytes.split_at(bytes.len() - CHECKSUM_LEN);
    ensure!(
        blake3::hash(body).as_bytes()[..] == checksum[..],
        "checksum mismatch"
    );
    ensure!(
        body[MAGIC.len()..MAGIC.len() + 32] == key.0[..],
        "stored under another key"
    );
    let frames = u32::from_le_bytes(body[HEADER_LEN - 4..HEADER_LEN].try_into()?) as usize;
    ensure!(
        frames == expected_frames,
        "holds {frames} frames, the clip has {expected_frames}"
    );
    let codes = &body[HEADER_LEN..];
    ensure!(
        codes.len() == frames * FRAME_LEN,
        "frame count disagrees with length"
    );
    codes
        .chunks_exact(FRAME_LEN)
        .map(|frame| {
            let mut out = [0u16; 16];
            for (code, pair) in out.iter_mut().zip(frame.chunks_exact(2)) {
                *code = u16::from_le_bytes([pair[0], pair[1]]);
                if *code >= 2048 {
                    bail!("code {code} outside the codec vocabulary");
                }
            }
            Ok(out)
        })
        .collect()
}

/// Whole file or nothing: a reader never sees a partial write, and two
/// starts writing the same key both end with a complete file.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().context("cache path has no directory")?;
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let temporary = dir.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("codes"),
        std::process::id()
    ));
    std::fs::write(&temporary, bytes).with_context(|| format!("write {}", temporary.display()))?;
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error).with_context(|| format!("rename into {}", path.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mary-reference-cache-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn root(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn codes(frames: usize) -> Vec<[u16; 16]> {
        (0..frames)
            .map(|t| std::array::from_fn(|q| ((t * 37 + q * 101) % 2048) as u16))
            .collect()
    }

    /// Runs `encode_reference_cached` with an encoder that counts its calls.
    fn cached(
        dir: Option<&Path>,
        key: &ReferenceKey,
        frames: usize,
        calls: &Cell<usize>,
    ) -> (Vec<[u16; 16]>, CacheOutcome) {
        encode_reference_cached(dir, key, frames, || {
            calls.set(calls.get() + 1);
            Ok(codes(frames))
        })
        .unwrap()
    }

    #[test]
    fn key_follows_every_input_and_nothing_else() {
        let key = ReferenceKey::new(b"RIFF clip", "words", root(1));
        assert_eq!(key, ReferenceKey::new(b"RIFF clip", "words", root(1)));
        assert_ne!(key, ReferenceKey::new(b"RIFF clop", "words", root(1)));
        assert_ne!(key, ReferenceKey::new(b"RIFF clip", "Words", root(1)));
        assert_ne!(key, ReferenceKey::new(b"RIFF clip", "words", root(2)));
        // Length prefixes keep the parts apart: moving a byte between the
        // clip and the transcript is a different key.
        assert_ne!(
            ReferenceKey::new(b"ab", "c", root(1)),
            ReferenceKey::new(b"a", "bc", root(1))
        );
        let hex = key.to_string();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            key.path_in(Path::new("/c")),
            Path::new(&format!("/c/{hex}.codes"))
        );
        assert_eq!(format!("{key:?}"), format!("ReferenceKey({hex})"));
    }

    #[test]
    fn miss_computes_and_writes_then_hit_reads_the_same_bytes_without_encoding() {
        let dir = scratch("round-trip");
        let key = ReferenceKey::new(b"clip", "text", root(3));
        let calls = Cell::new(0);
        let (first, outcome) = cached(Some(&dir), &key, 139, &calls);
        assert_eq!(calls.get(), 1);
        let path = key.path_in(&dir);
        assert_eq!(
            outcome,
            CacheOutcome::Written {
                reason: "no file yet".into(),
                path: path.clone()
            }
        );
        assert_eq!(first, codes(139));
        let stored = std::fs::read(&path).unwrap();
        assert_eq!(stored.len(), HEADER_LEN + 139 * FRAME_LEN + CHECKSUM_LEN);
        assert_eq!(
            stored,
            encode_file(&key, &first),
            "the file is exactly the encoding"
        );

        let (second, outcome) = cached(Some(&dir), &key, 139, &calls);
        assert_eq!(calls.get(), 1, "a hit does not run the encoder");
        assert_eq!(outcome, CacheOutcome::Hit(path.clone()));
        assert_eq!(second, first);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            stored,
            "a hit rewrites nothing"
        );
        assert!(
            std::fs::read_dir(&dir).unwrap().count() == 1,
            "no temporary file is left behind"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tampered_or_foreign_file_is_recomputed_and_replaced() {
        let dir = scratch("tampered");
        let key = ReferenceKey::new(b"clip", "text", root(4));
        let calls = Cell::new(0);
        let (good, _) = cached(Some(&dir), &key, 10, &calls);
        let path = key.path_in(&dir);
        let pristine = std::fs::read(&path).unwrap();

        // One code byte flipped: the checksum catches it.
        let mut bytes = pristine.clone();
        bytes[HEADER_LEN + 5] ^= 0x40;
        std::fs::write(&path, &bytes).unwrap();
        let (codes, outcome) = cached(Some(&dir), &key, 10, &calls);
        assert_eq!(codes, good);
        assert_eq!(calls.get(), 2);
        assert!(
            matches!(&outcome, CacheOutcome::Written { reason, .. } if reason == "checksum mismatch"),
            "{outcome}"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            pristine,
            "replaced by a good file"
        );

        // Another voice's codes stored under this name, with a valid checksum.
        let other = ReferenceKey::new(b"other clip", "text", root(4));
        std::fs::write(&path, encode_file(&other, &self::codes(10))).unwrap();
        let (codes, outcome) = cached(Some(&dir), &key, 10, &calls);
        assert_eq!((codes, calls.get()), (good.clone(), 3));
        assert!(
            matches!(&outcome, CacheOutcome::Written { reason, .. } if reason == "stored under another key"),
            "{outcome}"
        );

        // A frame count that is not this clip's.
        std::fs::write(&path, encode_file(&key, &self::codes(11))).unwrap();
        let (_, outcome) = cached(Some(&dir), &key, 10, &calls);
        assert!(
            matches!(&outcome, CacheOutcome::Written { reason, .. } if reason == "holds 11 frames, the clip has 10"),
            "{outcome}"
        );

        // A code outside the vocabulary, with a valid checksum.
        let mut bad = self::codes(10);
        bad[3][7] = 2048;
        std::fs::write(&path, encode_file(&key, &bad)).unwrap();
        let (codes, outcome) = cached(Some(&dir), &key, 10, &calls);
        assert_eq!(codes, good);
        assert!(
            matches!(&outcome, CacheOutcome::Written { reason, .. } if reason.contains("outside the codec vocabulary")),
            "{outcome}"
        );

        // Truncated, and garbage.
        std::fs::write(&path, &pristine[..pristine.len() - 1]).unwrap();
        let (_, outcome) = cached(Some(&dir), &key, 10, &calls);
        assert!(
            matches!(&outcome, CacheOutcome::Written { reason, .. } if reason == "checksum mismatch"),
            "{outcome}"
        );
        std::fs::write(&path, b"junk").unwrap();
        let (_, outcome) = cached(Some(&dir), &key, 10, &calls);
        assert!(
            matches!(&outcome, CacheOutcome::Written { reason, .. } if reason == "file too short"),
            "{outcome}"
        );
        assert_eq!(calls.get(), 7);
        assert_eq!(std::fs::read(&path).unwrap(), pristine);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_directory_or_an_unwritable_one_still_yields_the_codes() {
        let calls = Cell::new(0);
        let key = ReferenceKey::new(b"clip", "text", root(5));
        let (codes, outcome) = cached(None, &key, 4, &calls);
        assert_eq!((codes, outcome), (self::codes(4), CacheOutcome::Disabled));

        // A plain file where the directory should be: the write fails, the
        // codes are still returned, and the outcome says why.
        let blocker = scratch("blocker");
        std::fs::write(&blocker, b"").unwrap();
        let (codes, outcome) = cached(Some(&blocker), &key, 4, &calls);
        assert_eq!(codes, self::codes(4));
        assert!(
            matches!(&outcome, CacheOutcome::Unwritten { reason, .. } if reason.starts_with("unreadable")),
            "{outcome}"
        );
        assert_eq!(calls.get(), 2);
        std::fs::remove_file(&blocker).unwrap();
    }

    #[test]
    fn an_encoder_frame_count_that_disagrees_with_the_clip_is_an_error() {
        let dir = scratch("frames");
        let key = ReferenceKey::new(b"clip", "text", root(6));
        let error = encode_reference_cached(Some(&dir), &key, 10, || Ok(codes(9))).unwrap_err();
        assert!(error.to_string().contains("9 frames for a clip of 10"));
        assert!(!dir.exists(), "nothing is written for a rejected encoding");
    }
}
