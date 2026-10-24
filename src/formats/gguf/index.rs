//! Bounded GGUF v3 little-endian index and selective row reads for import/parity.
//!
//! Raw type tags remain opaque unless explicitly supported. No tensor payload
//! is read while indexing. Unknown types can be named but never decoded or
//! assigned a guessed byte size. This is not the runtime weight-loader path.

use std::collections::HashSet;
use std::fs::{File, Metadata};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::UNIX_EPOCH;

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
}

impl Index {
    pub fn read<R: Read + Seek>(reader: R, len: u64) -> Result<Self> {
        let mut h = Header { reader, len };
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
        for _ in 0..metadata {
            let key = h.string()?;
            let ty = h.u32()?;
            if key == "general.alignment" {
                ensure!(!alignment_seen && ty == 4, "alignment must be one UINT32 value");
                alignment_seen = true;
                alignment = u64::from(h.u32()?);
                ensure!(alignment.is_power_of_two() && alignment <= MAX_HEADER, "invalid GGUF alignment");
            } else {
                h.value(ty)?;
            }
        }
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
        Ok(Self { version, data_offset, alignment, tensors: infos })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name).with_context(|| format!("missing GGUF tensor {name}"))
    }
}

pub struct IndexedFile {
    file: File,
    pub identity: FileIdentity,
    pub index: Index,
}

impl IndexedFile {
    pub fn open(path: &Path) -> Result<Self> {
        let before = path.symlink_metadata()?;
        ensure!(!before.file_type().is_symlink(), "GGUF source must not be a symlink");
        let file = File::open(path)?;
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        ensure!(FileIdentity::from_metadata(&before)? == identity, "GGUF source changed while opening");
        let index = Index::read(BufReader::new(file.try_clone()?), identity.bytes)?;
        ensure!(FileIdentity::from_metadata(&file.metadata()?)? == identity, "GGUF source changed during indexing");
        Ok(Self { file, identity, index })
    }

    /// Read whole rows only, capped at 16 MiB. Returns absolute file offset and
    /// unchanged packed bytes; no decoding or scale_2 association is implicit.
    pub fn read_rows(&mut self, name: &str, first: u64, rows: u64) -> Result<(u64, Vec<u8>)> {
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
        self.check_identity()?;
        self.file.seek(SeekFrom::Start(offset))?;
        let mut data = vec![0; usize::try_from(bytes)?];
        self.file.read_exact(&mut data).context("truncated selected rows")?;
        self.check_identity()?;
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
}
