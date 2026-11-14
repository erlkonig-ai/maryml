//! Durable inference checkpoints, not process images or learned-model versions.
//!
//! Payloads are bounded raw packed-byte blobs. A native collection commit is
//! published only after its complete payload has been flushed. A crash before
//! publication leaves unreferenced blobs, never an advertised partial cache.
//! TP ranks publish independently; the engine resumes only their common prefix.
//! This module is backend-free so storage and prefix selection have CPU tests.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use triblespace::core::metadata;
use triblespace::prelude::blobencodings::{RawBytes, SimpleArchive, UTF8String};
use triblespace::prelude::inlineencodings::{Blake3, Handle, Hash, U256BE};
use triblespace::prelude::*;

pub const MAX_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const COLLECTION: &str = "inkling-inference-cache";
// Version 2 replaces the collection-facts fingerprint with a descriptor and root.
const MANIFEST_VERSION: u32 = 2;

/// Each rank uses an explicitly named local pile and an existing durable key.
/// Nothing implicitly copies caches into the model or the shared home pile.
#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub pile: PathBuf,
    pub signing_key: PathBuf,
    pub checkpoint_tokens: usize,
}

// All anchors/kind below were minted with `trible genid` on 2026-09-09.
pub(crate) mod schema {
    use super::*;
    pub const KIND: Id = triblespace::macros::id_hex!("2B63C12AB244407A486498721AD2295E");
    attributes! {
        /// Hash of the exact ordered token prefix (u32 little endian).
        "EC16E68615D6FB652253D9FD6954B7A9" as pub prefix: Hash<Blake3>;
        /// This checkpoint's tensor-parallel rank, not a placement hint.
        "90837F98D8F95A3F28DE37EE986F017B" as pub rank: U256BE;
        /// Versioned JSON metadata; all tensor payloads remain external blobs.
        "4E0024AF186CF6A29404B4CDAEED4FD5" as pub manifest: Handle<UTF8String>;
        /// Packed payload dependency, repeated, so collection traversal sees it.
        "0B64970E7E2CF30203E89C44A00FBAD0" as pub chunk: Handle<RawBytes>;
        /// Model, config, cache layout and rank compatibility, not build hash.
        "758F56F44FA32051D4BA462300DCF751" as pub compatibility: Hash<Blake3>;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    pub model_collection: [u8; 32],
    pub model_root: [u8; 16],
    pub compatibility: [u8; 32],
    pub rank: usize,
    pub world: usize,
    pub prefix: Prefix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Prefix {
    pub position: usize,
    pub hash: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Chunk {
    hash: [u8; 32],
    bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    key: CacheKey,
    state: serde_json::Value,
    chunks: Vec<Chunk>,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub prefix: Prefix,
    manifest: [u8; 32],
}

pub struct CacheStore {
    pile: Pile,
    key: SigningKey,
    collection: crate::model_collection::ModelCollection,
    staged: BTreeMap<[u8; 32], usize>,
    pub checkpoint_tokens: usize,
}

impl CacheStore {
    pub fn open(config: &CacheConfig) -> Result<Self> {
        anyhow::ensure!(
            config.checkpoint_tokens > 0,
            "cache checkpoint interval must be positive"
        );
        // Resolve authority BEFORE creating any file. Never generate or replace
        // a signing key as a side effect of reading/restoring a cache.
        let key = triblespace::core::signing_key_file::load_existing(&config.signing_key)
            .context("load existing cache signing key")?;
        Self::open_with_key(config, key)
    }

    fn open_with_key(config: &CacheConfig, key: SigningKey) -> Result<Self> {
        let mut create = std::fs::OpenOptions::new();
        create.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            create.mode(0o600);
        }
        match create.open(&config.pile) {
            Ok(file) => {
                file.sync_all()?;
                let parent = config
                    .pile
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or_else(|| std::path::Path::new("."));
                std::fs::File::open(parent)?.sync_all()?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error).context("create private cache pile"),
        }
        let mut pile = Pile::open(&config.pile).context("open cache pile")?;
        pile.refresh()
            .context("read cache pile; corrupt tails require explicit operator repair")?;
        let collection =
            crate::model_collection::collection_or_create(&mut pile, &key, COLLECTION)?;
        let snapshot = pile.snapshot()?;
        anyhow::ensure!(
            collection.reader_is_admitted(&snapshot, key.verifying_key())?,
            "cache signing identity is not admitted by the collection READ policy"
        );
        Ok(Self {
            pile,
            key,
            collection,
            staged: BTreeMap::new(),
            checkpoint_tokens: config.checkpoint_tokens,
        })
    }

    pub fn begin(&mut self) {
        self.staged.clear();
    }

    /// Finish the local writer explicitly. Published checkpoints were already
    /// flushed; closing does not publish any merely staged payloads.
    pub fn close(self) -> Result<()> {
        self.pile.close().context("close inference cache pile")
    }

    pub fn put_chunk(&mut self, bytes: &[u8]) -> Result<[u8; 32]> {
        anyhow::ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_CHUNK_BYTES,
            "cache chunks must contain 1..={MAX_CHUNK_BYTES} bytes"
        );
        let handle = self
            .pile
            .put::<RawBytes, _>(bytes)
            .context("append packed cache chunk")?;
        self.staged.insert(handle.raw, bytes.len());
        Ok(handle.raw)
    }

    pub fn publish(&mut self, key: CacheKey, state: serde_json::Value) -> Result<()> {
        anyhow::ensure!(
            key.prefix.position > 0 && key.rank < key.world,
            "invalid cache coordinates"
        );
        let chunks = self
            .staged
            .iter()
            .map(|(&hash, &bytes)| Chunk { hash, bytes })
            .collect();
        let manifest = Manifest {
            version: MANIFEST_VERSION,
            key: key.clone(),
            state,
            chunks,
        };
        let json = serde_json::to_string(&manifest)?;
        anyhow::ensure!(
            json.len() <= MAX_MANIFEST_BYTES,
            "cache metadata exceeds its byte budget"
        );
        let handle = self.pile.put::<UTF8String, _>(json)?;
        let chunk_handles: Vec<Inline<Handle<RawBytes>>> =
            self.staged.keys().map(|&hash| Inline::new(hash)).collect();
        let fragment = entity! { _ @
            metadata::tag: schema::KIND,
            super::telemetry::schema::model_collection: Inline::<Handle<SimpleArchive>>::new(key.model_collection),
            super::telemetry::schema::model_root: Id::new(key.model_root).context("cache model root is nil")?,
            schema::compatibility: Inline::<Hash<Blake3>>::new(key.compatibility),
            schema::rank: key.rank as u128,
            super::telemetry::schema::tp_world: key.world as u128,
            super::telemetry::schema::position: key.prefix.position as u128,
            schema::prefix: Inline::<Hash<Blake3>>::new(key.prefix.hash),
            schema::manifest: handle,
            schema::chunk*: chunk_handles,
        };
        // A checkpoint is durable only after BOTH flushes, not merely after a
        // successful write or an in-memory collection update.
        self.pile
            .flush()
            .context("make cache payload durable before publication")?;
        self.pile
            .commit(self.collection, &self.key, fragment)
            .context("publish cache checkpoint")?;
        self.pile
            .flush()
            .context("make cache publication durable")?;
        self.staged.clear();
        Ok(())
    }

    pub fn candidates(&mut self, compatibility: [u8; 32], limit: usize) -> Result<Vec<Candidate>> {
        self.pile.refresh()?;
        let snapshot = self.pile.snapshot()?;
        let facts = self.collection.read::<TribleSet, _>(&snapshot)?;
        let mut candidates: Vec<Candidate> = find!(
            (position: u128, prefix: Inline<Hash<Blake3>>, manifest: Inline<Handle<UTF8String>>),
            and!(
                pattern!(&facts, [{ _?cache @
                    metadata::tag: schema::KIND,
                    schema::compatibility: Inline::<Hash<Blake3>>::new(compatibility),
                    super::telemetry::schema::position: ?position,
                    schema::prefix: ?prefix,
                    schema::manifest: ?manifest,
                }]),
                value_range(position, 1u128.to_inline(), (limit as u128).to_inline()),
            )
        )
        .map(|(position, prefix, manifest)| Candidate {
            prefix: Prefix {
                position: position as usize,
                hash: prefix.raw,
            },
            manifest: manifest.raw,
        })
        .collect();
        candidates.sort_by(|a, b| b.prefix.cmp(&a.prefix).then(a.manifest.cmp(&b.manifest)));
        candidates.dedup_by(|a, b| a.prefix == b.prefix && a.manifest == b.manifest);
        Ok(candidates)
    }

    pub fn read(&mut self, candidate: &Candidate, expected: &CacheKey) -> Result<CacheReader> {
        let snapshot = self.pile.snapshot()?;
        let blob: Blob<UTF8String> = snapshot.get(Inline::new(candidate.manifest))?;
        anyhow::ensure!(
            blob.bytes.len() <= MAX_MANIFEST_BYTES,
            "cache metadata exceeds its byte budget"
        );
        anyhow::ensure!(
            *blake3::hash(&blob.bytes).as_bytes() == candidate.manifest,
            "cache metadata content hash mismatch"
        );
        let manifest: Manifest =
            serde_json::from_slice(&blob.bytes).context("decode cache metadata")?;
        anyhow::ensure!(
            manifest.version == MANIFEST_VERSION
                && manifest.key == *expected
                && candidate.prefix == expected.prefix,
            "cache version, compatibility or prefix mismatch"
        );
        let mut chunks = BTreeMap::new();
        for chunk in manifest.chunks {
            anyhow::ensure!(
                chunk.bytes > 0 && chunk.bytes <= MAX_CHUNK_BYTES,
                "invalid packed chunk length"
            );
            anyhow::ensure!(
                chunks.insert(chunk.hash, chunk.bytes).is_none(),
                "duplicate cache chunk metadata"
            );
        }
        Ok(CacheReader {
            snapshot,
            state: manifest.state,
            chunks,
        })
    }
}

pub struct CacheReader {
    snapshot: PileSnapshot,
    pub state: serde_json::Value,
    chunks: BTreeMap<[u8; 32], usize>,
}

impl CacheReader {
    pub fn chunk(&self, hash: [u8; 32], expected: usize) -> Result<Vec<u8>> {
        anyhow::ensure!(
            self.chunks.get(&hash) == Some(&expected),
            "cache state names an undeclared or wrong-sized chunk"
        );
        let blob: Blob<RawBytes> = self
            .snapshot
            .get(Inline::new(hash))
            .context("read packed cache chunk")?;
        anyhow::ensure!(
            blob.bytes.len() == expected && *blake3::hash(&blob.bytes).as_bytes() == hash,
            "packed cache chunk length or content hash mismatch"
        );
        Ok(blob.bytes.to_vec())
    }
}

/// Compute only the requested prefix digests in one scan of the token stream.
/// This includes arbitrary cancellation boundaries, not just regular intervals.
pub fn matching_prefixes(ids: &[usize], offers: &[Prefix]) -> Result<Vec<Prefix>> {
    let mut positions: Vec<usize> = offers
        .iter()
        .map(|p| p.position)
        .filter(|&p| p > 0 && p <= ids.len())
        .collect();
    positions.sort_unstable();
    positions.dedup();
    let mut hasher = blake3::Hasher::new();
    let mut cursor = 0;
    let mut matched = Vec::new();
    for position in positions {
        for &id in &ids[cursor..position] {
            let id = u32::try_from(id).context("token id exceeds cache encoding width")?;
            hasher.update(&id.to_le_bytes());
        }
        cursor = position;
        let prefix = Prefix {
            position,
            hash: *hasher.clone().finalize().as_bytes(),
        };
        if offers.contains(&prefix) {
            matched.push(prefix);
        }
    }
    matched.reverse();
    Ok(matched)
}

pub fn token_prefix(ids: &[usize]) -> Result<Prefix> {
    let mut hash = blake3::Hasher::new();
    for &id in ids {
        hash.update(&u32::try_from(id)?.to_le_bytes());
    }
    Ok(Prefix {
        position: ids.len(),
        hash: *hash.finalize().as_bytes(),
    })
}

/// Newest exact prefix available on every rank. A save that reached only one
/// pile is useful data, but never authority to combine different TP states.
pub fn common_prefix(ids: &[usize], offers: &[Vec<Prefix>]) -> Result<Option<Prefix>> {
    let first = offers
        .first()
        .context("cache selection requires at least one rank")?;
    Ok(matching_prefixes(ids, first)?
        .into_iter()
        .find(|prefix| offers.iter().all(|rank| rank.contains(prefix))))
}

/// Small host-side control messages; tensor payloads never travel on this wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheCommand {
    Candidates { limit: usize },
    Restore(Prefix),
    Save(Prefix),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CacheReply {
    pub error: Option<String>,
    pub prefixes: Vec<Prefix>,
}

impl CacheReply {
    pub fn from_result(result: Result<Vec<Prefix>>) -> Self {
        match result {
            Ok(prefixes) => Self {
                error: None,
                prefixes,
            },
            Err(error) => Self {
                error: Some(format!("{error:#}")),
                prefixes: Vec::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (CacheConfig, SigningKey) {
        let pile = std::env::temp_dir().join(format!("inkling-cache-{}.pile", genid()));
        (
            CacheConfig {
                pile,
                signing_key: PathBuf::from("unused-existing-key"),
                checkpoint_tokens: 32,
            },
            SigningKey::generate(&mut rand::rngs::OsRng),
        )
    }

    fn key(prefix: Prefix) -> CacheKey {
        CacheKey {
            model_collection: [1; 32],
            model_root: [3; 16],
            compatibility: [2; 32],
            rank: 0,
            world: 2,
            prefix,
        }
    }

    #[test]
    fn cache_manifest_version_and_reference_shape_are_explicit() {
        assert_eq!(MANIFEST_VERSION, 2);
        let prefix = token_prefix(&[3, 4, 5]).unwrap();
        let model = [1u8; 32];
        let compatibility = [2u8; 32];
        let legacy = serde_json::json!({
            "version": 1,
            "key": { "model": model, "compatibility": compatibility,
                "rank": 0, "world": 2, "prefix": prefix },
            "state": {}, "chunks": []
        });
        assert!(serde_json::from_value::<Manifest>(legacy).is_err());
    }

    #[test]
    fn arbitrary_prefix_resume_and_changed_tail() -> Result<()> {
        let ids = [9, 8, 7, 6, 5];
        let p2 = token_prefix(&ids[..2])?;
        let p4 = token_prefix(&ids[..4])?;
        let wrong = token_prefix(&[9, 8, 0])?;
        assert_eq!(matching_prefixes(&ids, &[p2, wrong, p4])?, [p4, p2]);
        assert_eq!(matching_prefixes(&[9, 8, 2, 6], &[p2, p4])?, [p2]);
        Ok(())
    }

    #[test]
    fn incomplete_pair_uses_last_shared_exact_prefix() -> Result<()> {
        let ids = [9, 8, 7, 6, 5];
        let old = token_prefix(&ids[..2])?;
        let new = token_prefix(&ids)?;
        assert_eq!(
            common_prefix(&ids, &[vec![new, old], vec![old]])?,
            Some(old)
        );
        assert_eq!(common_prefix(&ids, &[vec![new, old], vec![]])?, None);
        assert_eq!(
            common_prefix(&[9, 8, 0], &[vec![new, old], vec![new, old]])?,
            Some(old)
        );
        assert_eq!(common_prefix(&[], &[vec![old], vec![old]])?, None);
        assert!(common_prefix(&ids, &[]).is_err());
        Ok(())
    }

    #[test]
    fn published_checkpoint_roundtrips_and_unsaved_blobs_do_not() -> Result<()> {
        let (config, signing) = fixture();
        let prefix = token_prefix(&[3, 4, 5])?;
        {
            let mut store = CacheStore::open_with_key(&config, signing.clone())?;
            store.begin();
            store.put_chunk(b"unpublished")?;
            assert!(store.candidates([2; 32], 99)?.is_empty());
            store.begin();
            let hash = store.put_chunk(b"packed codes, not floats")?;
            store.publish(key(prefix), serde_json::json!({"chunk": hash}))?;
            // Deliberately omit close here: publication itself must be durable.
        }
        {
            let mut store = CacheStore::open_with_key(&config, signing)?;
            assert!(store.candidates([8; 32], 99)?.is_empty());
            assert!(store.candidates([2; 32], 2)?.is_empty());
            let found = store.candidates([2; 32], 99)?;
            assert_eq!(found.len(), 1);
            let reader = store.read(&found[0], &key(prefix))?;
            let hash = serde_json::from_value(reader.state["chunk"].clone())?;
            assert_eq!(reader.chunk(hash, 24)?, b"packed codes, not floats");
            assert!(reader.chunk(hash, 23).is_err());
            assert!(reader.chunk([0; 32], 1).is_err());
            let mut wrong = key(prefix);
            wrong.rank = 1;
            assert!(store.read(&found[0], &wrong).is_err());
            let mut wrong = key(prefix);
            wrong.model_collection[0] ^= 1;
            assert!(store.read(&found[0], &wrong).is_err());
            let mut wrong = key(prefix);
            wrong.model_root[0] ^= 1;
            assert!(store.read(&found[0], &wrong).is_err());
            store.close()?;
        }
        std::fs::remove_file(config.pile)?;
        Ok(())
    }

    #[test]
    fn chunk_bounds_and_missing_key_refuse_before_publication() -> Result<()> {
        let (config, signing) = fixture();
        assert!(CacheStore::open(&config).is_err());
        assert!(!config.pile.exists());
        {
            let mut store = CacheStore::open_with_key(&config, signing)?;
            assert!(store.put_chunk(&[]).is_err());
            assert!(store.put_chunk(&vec![0; MAX_CHUNK_BYTES + 1]).is_err());
            assert!(store.candidates([2; 32], 99)?.is_empty());
            store.close()?;
        }
        std::fs::remove_file(config.pile)?;
        Ok(())
    }
}
