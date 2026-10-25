//! Bounded GGUF v3 little-endian index and selective row reads for import/parity.
//!
//! Raw type tags remain opaque unless explicitly supported. Indexing does not
//! decode or materialize tensors; buffering/OS read-ahead may fetch payload
//! bytes beyond the header. Unknown types can be named but never decoded or
//! assigned a guessed byte size. This is not the runtime weight-loader path.

use std::collections::HashSet;
use std::fs::{File, Metadata};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Instant,UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

const MAX_HEADER: u64 = 128 * 1024 * 1024;
const MAX_ENTRIES: u64 = 1_000_000;
const MAX_STRING: u64 = 16 * 1024 * 1024;
const MAX_ARRAY: u64 = 16 * 1024 * 1024;
pub const MAX_READ: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FileIdentity {
    pub bytes: u64,
    pub modified_ns: u128,
    #[cfg(unix)]
    pub dev: u64,
    #[cfg(unix)]
    pub ino: u64,
}

impl FileIdentity {
    fn from_metadata(meta: &Metadata) -> Result<Self> {
        ensure!(meta.is_file(), "GGUF source is not a regular file");
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            bytes: meta.len(),
            modified_ns: meta.modified()?.duration_since(UNIX_EPOCH)?.as_nanos(),
            #[cfg(unix)]
            dev: meta.dev(),
            #[cfg(unix)]
            ino: meta.ino(),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: u32,
    /// GGML fastest-first dimensions, preserving the expert axis. Reverse for
    /// Mary's row-major shape. scale_2 remains a separately named tensor.
    pub dims: Vec<u64>,
    pub offset: u64,
    pub bytes: Option<u64>,
}

impl TensorInfo {
    pub fn shape(&self) -> Vec<u64> {
        self.dims.iter().rev().copied().collect()
    }

    pub fn elements(&self) -> Result<u64> {
        self.dims.iter().try_fold(1u64, |n, &d| {
            n.checked_mul(d).context("tensor element count overflow")
        })
    }

    pub fn row_bytes(&self) -> Result<u64> {
        let width = self.dims[0];
        let (elements, bytes) = match self.ggml_type {
            0 => (1, 4),             // F32
            1 | 30 => (1, 2),        // F16 / BF16 (canonical GGML tags)
            40 => (64, 36),          // NVFP4
            tag => bail!("opaque GGML type {tag}: no supported row layout"),
        };
        ensure!(width % elements == 0, "tensor {} row is not block-aligned", self.name);
        (width / elements).checked_mul(bytes).context("row byte length overflow")
    }
}

#[derive(Debug, Serialize)]
pub struct Index {
    pub version: u32,
    pub data_offset: u64,
    pub alignment: u64,
    pub tensors: Vec<TensorInfo>,
    /// Only caller-selected semantics; unrelated metadata is skipped as before.
    pub metadata: Vec<MetadataEntry>,
    pub metadata_end: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MetadataEntry { pub key: String, pub gguf_type: u32, pub value: MetadataValue }

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum MetadataValue {
    U32(u32), F32(f32), Bool(bool), String(String), BoolArray(Vec<bool>),
    I32Array(Vec<i32>), StringArray(Vec<String>),
}

impl MetadataValue {
    pub fn gguf_type(&self) -> u32 {
        match self {
            Self::U32(_) => 4, Self::F32(_) => 6, Self::Bool(_) => 7, Self::String(_) => 8,
            Self::BoolArray(_) | Self::I32Array(_) | Self::StringArray(_) => 9,
        }
    }
}

/// Explicit allocation/read limits, not inferred from an untrusted GGUF count.
#[derive(Clone, Copy, Debug)]
pub struct MetadataLimits {
    pub header_bytes: u64,
    pub string_bytes: u64,
    pub array_entries: u64,
    pub retained_bytes: u64,
}

impl MetadataLimits {
    pub const SCALARS: Self = Self {
        header_bytes: MAX_HEADER, string_bytes: 4096, array_entries: 4096,
        retained_bytes: 1024 * 1024,
    };
    /// Fits Flint's 5,436,996-byte metadata section and 150,144-token vocabulary.
    /// The retained budget includes string-array slots as well as UTF-8 bytes.
    pub const TOKENIZER: Self = Self {
        header_bytes: 8 * 1024 * 1024, string_bytes: 64 * 1024, array_entries: 262_144,
        retained_bytes: 16 * 1024 * 1024,
    };
}

/// Metadata only: stops before the tensor directory, never seeks to a tensor.
#[derive(Debug, Serialize)]
pub struct SelectedMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub alignment: u64,
    pub entries: Vec<MetadataEntry>,
    pub end: u64,
}

impl SelectedMetadata {
    pub fn read<R: Read + Seek>(reader: R, len: u64, keys: &[&str], limits: MetadataLimits) -> Result<Self> {
        Header { reader, len: len.min(limits.header_bytes) }.metadata(keys, limits)
    }

    /// Unbuffered userspace reads prevent read-ahead into tensor payloads. OS
    /// page-cache read-ahead is outside this API's byte-read contract.
    pub fn open(path: &Path, keys: &[&str], limits: MetadataLimits) -> Result<(FileIdentity, Self)> {
        let before = path.symlink_metadata()?;
        ensure!(!before.file_type().is_symlink(), "GGUF source must not be a symlink");
        let mut file = File::open(path)?;
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        ensure!(FileIdentity::from_metadata(&before)? == identity, "GGUF source changed while opening");
        let selected = Self::read(&mut file, identity.bytes, keys, limits)?;
        ensure!(FileIdentity::from_metadata(&file.metadata()?)? == identity, "GGUF source changed during metadata read");
        Ok((identity, selected))
    }
}

struct Header<R> {
    reader: R,
    len: u64,
}

impl<R: Read + Seek> Header<R> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.check(N as u64)?;
        let mut data = [0; N];
        self.reader.read_exact(&mut data).context("truncated GGUF header")?;
        Ok(data)
    }

    fn check(&mut self, bytes: u64) -> Result<u64> {
        let end = self.reader.stream_position()?.checked_add(bytes).context("header offset overflow")?;
        ensure!(end <= self.len && end <= MAX_HEADER, "truncated or oversized GGUF header");
        Ok(end)
    }

    fn skip(&mut self, bytes: u64) -> Result<()> {
        let end = self.check(bytes)?;
        self.reader.seek(SeekFrom::Start(end))?;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take()?)) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take()?)) }

    fn string(&mut self) -> Result<String> {
        let size = self.u64()?;
        ensure!(size <= MAX_STRING, "GGUF string exceeds bounded index limit");
        self.check(size)?;
        let mut data = vec![0; usize::try_from(size)?];
        self.reader.read_exact(&mut data)?;
        String::from_utf8(data).context("GGUF name is not UTF-8")
    }

    fn value(&mut self, tag: u32) -> Result<()> {
        match tag {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4..=6 => self.skip(4),
            10..=12 => self.skip(8),
            8 => {
                let len = self.u64()?;
                ensure!(len <= MAX_STRING, "GGUF metadata string exceeds limit");
                self.skip(len)
            }
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                ensure!(count <= MAX_ARRAY, "GGUF metadata array exceeds limit");
                let width: u64 = match element {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    10..=12 => 8,
                    8 => {
                        for _ in 0..count { self.value(8)?; }
                        return Ok(());
                    }
                    _ => bail!("unsupported GGUF array element type {element}"),
                };
                self.skip(count.checked_mul(width).context("metadata array overflow")?)
            }
            _ => bail!("unknown GGUF metadata type {tag}"),
        }
    }

    fn selected_value(&mut self, tag: u32, limits: MetadataLimits, budget: &mut u64) -> Result<MetadataValue> {
        fn boolean(byte: u8) -> Result<bool> {
            ensure!(byte <= 1, "invalid GGUF BOOL value");
            Ok(byte != 0)
        }
        Ok(match tag {
            4 => MetadataValue::U32(self.u32()?),
            6 => MetadataValue::F32(f32::from_le_bytes(self.take()?)),
            7 => MetadataValue::Bool(boolean(self.take::<1>()?[0])?),
            8 => MetadataValue::String(self.selected_string(limits, budget)?),
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                ensure!(count <= limits.array_entries, "selected metadata array exceeds limit");
                let width = match element {
                    7 => 1, 5 => 4, 8 => std::mem::size_of::<String>() as u64,
                    _ => bail!("unsupported selected GGUF array element type {element}"),
                };
                Self::charge(budget, count.checked_mul(width).context("selected array overflow")?)?;
                let capacity = usize::try_from(count)?;
                match element {
                    7 => {
                        self.check(count)?;
                        let mut values = Vec::with_capacity(capacity);
                        for _ in 0..count { values.push(boolean(self.take::<1>()?[0])?); }
                        MetadataValue::BoolArray(values)
                    }
                    5 => {
                        self.check(count * 4)?;
                        let mut values = Vec::with_capacity(capacity);
                        for _ in 0..count { values.push(i32::from_le_bytes(self.take()?)); }
                        MetadataValue::I32Array(values)
                    }
                    8 => {
                        self.check(count * 8)?;
                        let mut values = Vec::with_capacity(capacity);
                        for _ in 0..count { values.push(self.selected_string(limits, budget)?); }
                        MetadataValue::StringArray(values)
                    }
                    _ => unreachable!(),
                }
            }
            _ => bail!("unsupported selected GGUF metadata type {tag}"),
        })
    }

    fn charge(budget: &mut u64, bytes: u64) -> Result<()> {
        *budget = budget.checked_sub(bytes).context("selected metadata retained-byte limit exceeded")?;
        Ok(())
    }

    fn selected_string(&mut self, limits: MetadataLimits, budget: &mut u64) -> Result<String> {
        let size = self.u64()?;
        ensure!(size <= limits.string_bytes, "selected metadata string exceeds limit");
        Self::charge(budget, size)?;
        self.check(size)?;
        let mut bytes = vec![0; usize::try_from(size)?];
        self.reader.read_exact(&mut bytes)?;
        String::from_utf8(bytes).context("selected metadata is not UTF-8")
    }

    fn metadata(&mut self, keys: &[&str], limits: MetadataLimits) -> Result<SelectedMetadata> {
        ensure!(keys.len() <= 64, "too many selected metadata keys");
        ensure!(limits.header_bytes <= MAX_HEADER && limits.string_bytes <= MAX_STRING
            && limits.array_entries <= MAX_ARRAY && limits.retained_bytes <= MAX_HEADER,
            "selected metadata limits exceed hard bounds");
        let h = self;
        ensure!(h.reader.stream_position()? == 0, "GGUF reader must start at byte zero");
        let magic = h.take::<4>()?;
        ensure!(magic != *b"FUGG", "big-endian GGUF is not supported by this reader");
        ensure!(magic == *b"GGUF", "invalid GGUF magic");
        let version = h.u32()?;
        ensure!(version == 3, "only GGUF v3 little-endian is supported (version word {version:#x})");
        let tensors = h.u64()?;
        let metadata = h.u64()?;
        ensure!(tensors <= MAX_ENTRIES && metadata <= MAX_ENTRIES, "GGUF table exceeds bounded index limit");
        let mut alignment = 32u64;
        let mut alignment_seen = false;
        let mut selected = Vec::<MetadataEntry>::new();
        let mut budget = limits.retained_bytes;
        for _ in 0..metadata {
            let key = h.string()?;
            let ty = h.u32()?;
            if key == "general.alignment" {
                ensure!(!alignment_seen && ty == 4, "alignment must be one UINT32 value");
                alignment_seen = true;
                alignment = u64::from(h.u32()?);
                ensure!(alignment.is_power_of_two() && alignment <= MAX_HEADER, "invalid GGUF alignment");
            } else if keys.contains(&key.as_str()) {
                ensure!(!selected.iter().any(|entry| entry.key == key), "duplicate selected metadata key {key}");
                Self::charge(&mut budget, key.len() as u64 + std::mem::size_of::<MetadataEntry>() as u64)?;
                selected.push(MetadataEntry { key, gguf_type: ty, value: h.selected_value(ty, limits, &mut budget)? });
            } else {
                h.value(ty)?;
            }
        }
        let metadata_end = h.reader.stream_position()?;
        Ok(SelectedMetadata { version, tensor_count: tensors, alignment, entries: selected, end: metadata_end })
    }
}

impl Index {
    pub fn read<R: Read + Seek>(reader: R, len: u64) -> Result<Self> {
        Self::read_with_metadata(reader, len, &[])
    }

    pub fn read_with_metadata<R: Read + Seek>(reader: R, len: u64, keys: &[&str]) -> Result<Self> {
        let mut h = Header { reader, len };
        let SelectedMetadata { version, tensor_count: tensors, alignment, entries: selected, end: metadata_end } =
            h.metadata(keys, MetadataLimits::SCALARS)?;
        let mut infos = Vec::new();
        let mut names = HashSet::new(); // Ephemeral format validation, not retained collection state.
        for _ in 0..tensors {
            let name = h.string()?;
            ensure!(names.insert(name.clone()), "duplicate GGUF tensor name {name}");
            let rank = h.u32()?;
            ensure!((1..=4).contains(&rank), "GGUF tensor rank must be 1..=4");
            let mut dims = Vec::new();
            for _ in 0..rank {
                let dim = h.u64()?;
                ensure!(dim > 0 && dim <= i64::MAX as u64, "invalid GGUF tensor dimension");
                dims.push(dim);
            }
            let ggml_type = h.u32()?;
            let offset = h.u64()?;
            ensure!(offset % alignment == 0, "unaligned tensor offset");
            let mut info = TensorInfo { name, ggml_type, dims, offset, bytes: None };
            let count = info.elements()?;
            if matches!(ggml_type, 0 | 1 | 30 | 40) {
                info.bytes = Some(info.row_bytes()?.checked_mul(count / info.dims[0]).context("tensor byte length overflow")?);
            }
            infos.push(info);
        }
        let header_end = h.reader.stream_position()?;
        let data_offset = if tensors == 0 { header_end } else {
            header_end.checked_add(alignment - 1).context("data alignment overflow")? / alignment * alignment
        };
        ensure!(data_offset <= len, "truncated GGUF data alignment");
        // Sorted scratch references validate non-overlap, without changing file order.
        let mut ordered: Vec<_> = infos.iter().collect();
        ordered.sort_by_key(|t| t.offset);
        for (i, info) in ordered.iter().enumerate() {
            let start = data_offset.checked_add(info.offset).context("tensor offset overflow")?;
            ensure!(start <= len, "tensor offset beyond EOF");
            if let Some(bytes) = info.bytes {
                ensure!(start.checked_add(bytes).context("tensor end overflow")? <= len, "truncated tensor {}", info.name);
                if let Some(next) = ordered.get(i + 1) {
                    ensure!(info.offset.checked_add(bytes).context("tensor extent overflow")? <= next.offset, "overlapping GGUF tensors");
                }
            }
            if i > 0 { ensure!(ordered[i - 1].offset != info.offset, "duplicate tensor offset"); }
        }
        Ok(Self { version, data_offset, alignment, tensors: infos, metadata: selected, metadata_end })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name).with_context(|| format!("missing GGUF tensor {name}"))
    }
}

/// Completed selected-row suboperations, not metadata parsing or OS syscall
/// counts. calls/bytes count successful whole reads; errors may leave partial
/// subspans. Consecutive spans are disjoint; allocation may be lazy zeroing.
#[derive(Clone,Default,Serialize)]
pub struct RowReadTimings {
    pub calls:u64, pub bytes:u64, pub identity_calls:u64,
    pub geometry_seconds:f64, pub identity_seconds:f64, pub seek_seconds:f64,
    pub allocation_seconds:f64, pub read_exact_seconds:f64,
}
impl RowReadTimings {
    pub fn selected_seconds(&self)->f64 {
        self.geometry_seconds+self.identity_seconds+self.seek_seconds+self.allocation_seconds+self.read_exact_seconds
    }
}
pub struct IndexedFile {
    file: File,
    pub identity: FileIdentity,
    pub index: Index,
    pub row_timings:RowReadTimings,
}

impl IndexedFile {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_metadata(path, &[])
    }

    pub fn open_with_metadata(path: &Path, keys: &[&str]) -> Result<Self> {
        let before = path.symlink_metadata()?;
        ensure!(!before.file_type().is_symlink(), "GGUF source must not be a symlink");
        let file = File::open(path)?;
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        ensure!(FileIdentity::from_metadata(&before)? == identity, "GGUF source changed while opening");
        let index = Index::read_with_metadata(BufReader::new(file.try_clone()?), identity.bytes, keys)?;
        ensure!(FileIdentity::from_metadata(&file.metadata()?)? == identity, "GGUF source changed during indexing");
        Ok(Self { file, identity, index, row_timings:RowReadTimings::default() })
    }

    /// Read whole rows only, capped at 16 MiB. Returns absolute file offset and
    /// unchanged packed bytes; no decoding or scale_2 association is implicit.
    pub fn read_rows(&mut self, name: &str, first: u64, rows: u64) -> Result<(u64, Vec<u8>)> {
        let start=Instant::now();
        ensure!(rows > 0, "empty row read");
        let info = self.index.tensor(name)?;
        let total = info.elements()? / info.dims[0];
        ensure!(first.checked_add(rows).context("row range overflow")? <= total, "row range outside tensor");
        let row_bytes = info.row_bytes()?;
        let bytes = rows.checked_mul(row_bytes).context("read byte length overflow")?;
        ensure!(bytes <= MAX_READ, "selective read exceeds 16 MiB limit");
        let offset = self.index.data_offset.checked_add(info.offset)
            .and_then(|v| first.checked_mul(row_bytes).and_then(|b| v.checked_add(b)))
            .context("read offset overflow")?;
        self.row_timings.geometry_seconds+=start.elapsed().as_secs_f64();
        let start=Instant::now();
        self.check_identity()?;
        self.row_timings.identity_seconds+=start.elapsed().as_secs_f64();
        self.row_timings.identity_calls+=1;
        let start=Instant::now();
        self.file.seek(SeekFrom::Start(offset))?;
        self.row_timings.seek_seconds+=start.elapsed().as_secs_f64();
        let start=Instant::now();
        let mut data = vec![0; usize::try_from(bytes)?];
        self.row_timings.allocation_seconds+=start.elapsed().as_secs_f64();
        let start=Instant::now();
        self.file.read_exact(&mut data).context("truncated selected rows")?;
        self.row_timings.read_exact_seconds+=start.elapsed().as_secs_f64();
        let start=Instant::now();
        self.check_identity()?;
        self.row_timings.identity_seconds+=start.elapsed().as_secs_f64();
        self.row_timings.identity_calls+=1;
        self.row_timings.calls+=1; self.row_timings.bytes+=bytes;
        Ok((offset, data))
    }

    pub fn check_identity(&self) -> Result<()> {
        ensure!(FileIdentity::from_metadata(&self.file.metadata()?)? == self.identity, "GGUF source identity/size/mtime changed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn string(out: &mut Vec<u8>, text: &str) { out.extend((text.len() as u64).to_le_bytes()); out.extend(text.as_bytes()); }
    fn fixture(dims: &[u64], ty: u32, offset: u64) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes()); out.extend(1u64.to_le_bytes()); out.extend(0u64.to_le_bytes());
        string(&mut out, "blk.2.test.weight"); out.extend((dims.len() as u32).to_le_bytes());
        for dim in dims { out.extend(dim.to_le_bytes()); }
        out.extend(ty.to_le_bytes()); out.extend(offset.to_le_bytes());
        out.resize(out.len().div_ceil(32) * 32 + 144, 0);
        out
    }
    fn parse(data: &[u8]) -> Result<Index> { Index::read(Cursor::new(data), data.len() as u64) }

    #[test]
    fn nvfp4_shape_and_opaque_types() {
        let index = parse(&fixture(&[64, 2, 2], 40, 0)).unwrap();
        assert_eq!(index.tensors[0].shape(), [2, 2, 64]);
        assert_eq!(index.tensors[0].bytes, Some(144));
        let opaque = parse(&fixture(&[64], 999, 0)).unwrap();
        assert_eq!(opaque.tensors[0].bytes, None);
        assert!(opaque.tensors[0].row_bytes().is_err());
        assert_eq!(parse(&fixture(&[64], 30, 0)).unwrap().tensors[0].bytes, Some(128));
        assert_eq!(parse(&fixture(&[64], 28, 0)).unwrap().tensors[0].bytes, None);
    }

    #[test]
    fn rejects_version_endian_rank_alignment_and_overflow() {
        let mut data = fixture(&[64], 40, 0);
        data[4..8].copy_from_slice(&2u32.to_le_bytes()); assert!(parse(&data).is_err());
        data[4..8].copy_from_slice(&3u32.to_be_bytes()); assert!(parse(&data).is_err());
        data[..4].copy_from_slice(b"FUGG"); assert!(parse(&data).is_err());
        for dims in [vec![], vec![0], vec![63], vec![64; 5], vec![64, i64::MAX as u64]] {
            assert!(parse(&fixture(&dims, 40, 0)).is_err(), "{dims:?}");
        }
        assert!(parse(&fixture(&[64], 40, 1)).is_err());
        assert!(parse(&fixture(&[64], 40, u64::MAX - 31)).is_err());
    }

    #[test]
    fn rejects_each_header_and_payload_truncation() {
        let data = fixture(&[64, 2, 2], 40, 0);
        for len in 0..data.len() { assert!(parse(&data[..len]).is_err(), "accepted truncation at {len}"); }
        assert!(parse(&data).is_ok());
    }

    #[test]
    fn tokenizer_metadata_is_bounded_and_stops_before_tensor_directory() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes()); bytes.extend(1264u64.to_le_bytes()); bytes.extend(2u64.to_le_bytes());
        string(&mut bytes, "tokens"); bytes.extend(9u32.to_le_bytes()); bytes.extend(8u32.to_le_bytes());
        bytes.extend(2u64.to_le_bytes()); string(&mut bytes, "hello"); string(&mut bytes, "Ġ世界");
        string(&mut bytes, "types"); bytes.extend(9u32.to_le_bytes()); bytes.extend(5u32.to_le_bytes());
        bytes.extend(2u64.to_le_bytes()); bytes.extend(1i32.to_le_bytes()); bytes.extend(4i32.to_le_bytes());
        let end = bytes.len();
        bytes.extend(b"NOT A TENSOR DIRECTORY OR PAYLOAD");
        let read = |data: &[u8], limits| SelectedMetadata::read(Cursor::new(data), data.len() as u64, &["tokens", "types"], limits);
        let selected = read(&bytes, MetadataLimits::TOKENIZER).unwrap();
        assert_eq!(selected.tensor_count, 1264);
        assert_eq!(selected.end, end as u64);
        assert_eq!(selected.entries[0].value, MetadataValue::StringArray(vec!["hello".into(), "Ġ世界".into()]));
        assert_eq!(selected.entries[1].value, MetadataValue::I32Array(vec![1, 4]));
        for length in 0..end { assert!(read(&bytes[..length], MetadataLimits::TOKENIZER).is_err()); }
        for limits in [
            MetadataLimits { header_bytes: end as u64 - 1, ..MetadataLimits::TOKENIZER },
            MetadataLimits { string_bytes: 4, ..MetadataLimits::TOKENIZER },
            MetadataLimits { array_entries: 1, ..MetadataLimits::TOKENIZER },
            MetadataLimits { retained_bytes: 1, ..MetadataLimits::TOKENIZER },
        ] { assert!(read(&bytes, limits).is_err()); }
        // No tensor table is needed for metadata-only reading; indexing still rejects it.
        assert!(parse(&bytes).is_err());
    }
}
    #[test]
    fn selected_metadata_is_typed_bounded_and_unrelated_values_are_skipped() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes()); bytes.extend(0u64.to_le_bytes()); bytes.extend(3u64.to_le_bytes());
        let entry = |dst: &mut Vec<u8>, key: &str, tag: u32| {
            dst.extend((key.len() as u64).to_le_bytes()); dst.extend(key.as_bytes()); dst.extend(tag.to_le_bytes());
        };
        entry(&mut bytes,"architecture",8); bytes.extend(5u64.to_le_bytes()); bytes.extend(b"flint");
        entry(&mut bytes,"unrelated",9); bytes.extend(10u32.to_le_bytes()); bytes.extend(2u64.to_le_bytes()); bytes.extend([0u8;16]);
        entry(&mut bytes,"pattern",9); bytes.extend(7u32.to_le_bytes()); bytes.extend(3u64.to_le_bytes()); bytes.extend([0,1,0]);
        let parse_selected = |data: &[u8]| Index::read_with_metadata(std::io::Cursor::new(data),data.len() as u64,&["architecture","pattern"]);
        let result = parse_selected(&bytes).unwrap();
        assert_eq!(result.metadata.len(),2);
        assert_eq!(result.metadata[0].value,MetadataValue::String("flint".into()));
        assert_eq!(result.metadata[1].value,MetadataValue::BoolArray(vec![false,true,false]));
        assert_eq!(result.metadata_end,bytes.len() as u64);
        for size in 0..bytes.len() { assert!(parse_selected(&bytes[..size]).is_err()); }
        let mut bad = bytes.clone(); *bad.last_mut().unwrap() = 2;
        assert!(parse_selected(&bad).is_err());
        let n = bytes.len(); bytes[n-11..n-3].copy_from_slice(&4097u64.to_le_bytes());
        assert!(parse_selected(&bytes).is_err());
    }
