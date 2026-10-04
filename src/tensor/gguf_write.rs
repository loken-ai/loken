//! Minimal GGUF v3 writer - exactly the subset `quant::gguf_file::Content::read` consumes
//! (zero metadata KVs, the default data alignment). Used to persist one-time checkpoint
//! conversions (fp8-scaled safetensors -> block-quantized sidecar) so later loads read the
//! quantized bytes straight from disk instead of re-decoding the source checkpoint.
//!
//! What the container IS is not restated here: the magic, the version, the header field
//! order, the tensor-descriptor field order (including the dimension reversal) and the
//! alignment rule all come from `gguf_file::layout`, which the reader consumes as well.

use super::quantized::gguf_file::layout::{
    self, value_type, TensorDescriptor, DEFAULT_ALIGNMENT, FILE_HEADER, MAGIC, WRITE_VERSION,
};
use super::quantized::gguf_file::Value;
use super::quantized::GgmlDType;
use super::{Error, Result};
use std::io::Write;
use std::path::Path;

pub struct GgufEntry {
    pub name: String,
    /// Row-major dims (the GGUF file stores them reversed; the descriptor flips).
    pub dims: Vec<usize>,
    pub dtype: GgmlDType,
    /// The tensor's raw (possibly block-quantized) bytes.
    pub data: Vec<u8>,
}

/// The type tag a value is written under, mirroring `read_value`'s dispatch.
fn value_tag(v: &Value) -> u32 {
    match v {
        Value::U8(_) => value_type::U8,
        Value::I8(_) => value_type::I8,
        Value::U16(_) => value_type::U16,
        Value::I16(_) => value_type::I16,
        Value::U32(_) => value_type::U32,
        Value::I32(_) => value_type::I32,
        Value::F32(_) => value_type::F32,
        Value::Bool(_) => value_type::BOOL,
        Value::String(_) => value_type::STRING,
        Value::Array(_) => value_type::ARRAY,
        Value::U64(_) => value_type::U64,
        Value::I64(_) => value_type::I64,
        Value::F64(_) => value_type::F64,
    }
}

/// A value's payload, the exact inverse of `read_value`. The type tag is written by the caller,
/// once, before the payload; an array writes its element type tag and count, then each element's
/// payload with no per-element tag.
fn write_value_payload(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(*x as u8),
        Value::String(s) => layout::write_string(out, s),
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Array(items) => {
            // The element type is written once. An empty array lost its element type on read
            // (the Value enum does not carry it), so it is emitted as I32 - model metadata
            // arrays that matter (tokens, scores, merges, token types) are never empty.
            let elem = items.first().map(value_tag).unwrap_or(value_type::I32);
            out.extend_from_slice(&elem.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                write_value_payload(out, it);
            }
        }
    }
}

/// Write `entries` as a GGUF v3 file with no metadata. See `write_gguf_with_metadata`.
pub fn write_gguf(path: &Path, entries: &[GgufEntry]) -> Result<()> {
    write_gguf_with_metadata(path, &[], entries)
}

/// Write `entries` as a GGUF v3 file carrying `metadata`. Atomic: writes `<path>.partial` then
/// renames, so a crashed or cancelled write never leaves a readable half-file behind.
///
/// The metadata is the model's own key/value block - architecture, hyper-parameters, tokenizer,
/// chat template - carried through byte for byte from the source so a requantised model is served
/// as the format it declares. Dropping a key here is the failure that makes a model close its
/// turn after a dozen tokens.
/// The GGUF header for `metadata` and the descriptors of `entries`: the magic, the counts, the
/// metadata KVs, then each tensor's descriptor with its data offset. This is the whole file up to
/// the aligned tensor-data section; `write_gguf_with_metadata` appends the data, and a caller that
/// only needs the header (a metadata-only transfer) uses it as is.
fn build_header(metadata: &[(String, Value)], entries: &[GgufEntry]) -> Vec<u8> {
    let mut header: Vec<u8> = Vec::with_capacity(64 * entries.len());
    layout::write_scalars(
        &mut header,
        &FILE_HEADER,
        &[
            u64::from(MAGIC),
            u64::from(WRITE_VERSION.to_u32()),
            entries.len() as u64,
            metadata.len() as u64,
        ],
    );
    for (key, value) in metadata {
        layout::write_string(&mut header, key);
        header.extend_from_slice(&value_tag(value).to_le_bytes());
        write_value_payload(&mut header, value);
    }

    let mut offset = 0u64; // relative to the (aligned) tensor-data start
    for e in entries {
        TensorDescriptor {
            name: &e.name,
            dims: &e.dims,
            dtype: e.dtype,
            offset,
        }
        .encode(&mut header);
        offset = layout::align_up(offset + e.data.len() as u64, DEFAULT_ALIGNMENT);
    }
    header
}

/// A GGUF header carrying only metadata and no tensors, as bytes. `Content::read` parses this into
/// the metadata map, so a node can hand its model configuration to a peer without a file - the peer
/// reads it exactly as it reads a real GGUF's header.
pub fn metadata_only_bytes(metadata: &[(String, Value)]) -> Vec<u8> {
    build_header(metadata, &[])
}

pub fn write_gguf_with_metadata(
    path: &Path,
    metadata: &[(String, Value)],
    entries: &[GgufEntry],
) -> Result<()> {
    let io_err = |e: std::io::Error| Error(format!("gguf write {}: {e}", path.display()));

    let header = build_header(metadata, entries);

    let tmp = path.with_extension("gguf.partial");
    let f = std::fs::File::create(&tmp).map_err(io_err)?;
    let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
    (|| -> std::io::Result<()> {
        w.write_all(&header)?;
        let data_start = layout::align_up(header.len() as u64, DEFAULT_ALIGNMENT);
        w.write_all(&vec![0u8; (data_start - header.len() as u64) as usize])?;
        let mut pos = 0u64;
        for e in entries {
            w.write_all(&e.data)?;
            pos += e.data.len() as u64;
            let padded = layout::align_up(pos, DEFAULT_ALIGNMENT);
            w.write_all(&vec![0u8; (padded - pos) as usize])?;
            pos = padded;
        }
        w.flush()
    })()
    .map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        io_err(e)
    })?;
    drop(w);
    std::fs::rename(&tmp, path).map_err(io_err)?;
    Ok(())
}

/// A tensor a streaming write will carry: its descriptor now, its bytes later, in order.
pub struct PlannedEntry {
    pub name: String,
    /// Row-major dims.
    pub dims: Vec<usize>,
    pub dtype: GgmlDType,
    pub byte_len: u64,
}

/// Where a streaming write puts each tensor. Every tensor starts at a multiple of `alignment`,
/// the value the file declares under `general.alignment`; `starts[i]`, when larger, is the
/// multiple of the file's own offset tensor `i` starts at, so a group of tensors read together
/// can begin on a boundary the device reads in one request.
pub struct Placement {
    pub alignment: u64,
    /// Per planned tensor, a multiple of `alignment`; empty for `alignment` everywhere.
    pub starts: Vec<u64>,
}

impl Placement {
    /// Every tensor at the next multiple of the default alignment: the layout `create` writes.
    pub fn packed() -> Self {
        Self {
            alignment: DEFAULT_ALIGNMENT,
            starts: Vec::new(),
        }
    }
}

/// The metadata value's bytes as a file carries them: the type tag, then the payload. Two values
/// are the same value when these bytes are equal.
pub fn encode_value(v: &Value) -> Vec<u8> {
    let mut out = value_tag(v).to_le_bytes().to_vec();
    write_value_payload(&mut out, v);
    out
}

/// The header for `metadata` and `entries` with each tensor's data offset, given the absolute
/// file position each tensor starts at and where the data section starts.
fn stream_header(
    metadata: &[(String, Value)],
    entries: &[PlannedEntry],
    starts: &[u64],
    data_start: u64,
) -> Vec<u8> {
    let mut header: Vec<u8> = Vec::with_capacity(64 * entries.len());
    layout::write_scalars(
        &mut header,
        &FILE_HEADER,
        &[
            u64::from(MAGIC),
            u64::from(WRITE_VERSION.to_u32()),
            entries.len() as u64,
            metadata.len() as u64,
        ],
    );
    for (key, value) in metadata {
        layout::write_string(&mut header, key);
        header.extend_from_slice(&encode_value(value));
    }
    for (e, &start) in entries.iter().zip(starts) {
        TensorDescriptor {
            name: &e.name,
            dims: &e.dims,
            dtype: e.dtype,
            offset: start - data_start,
        }
        .encode(&mut header);
    }
    header
}

/// A GGUF written tensor by tensor: the header - metadata and every descriptor, whose byte
/// lengths are known ahead - goes out first, then each tensor's bytes are appended in the
/// planned order. For a file too large to hold in memory, such as a requantised expert set.
/// Atomic like `write_gguf_with_metadata`: a `.partial` file renamed on `finish`. The bytes are
/// hashed as they go out, so a store that names files by digest need not read them back.
pub struct GgufStreamWriter {
    path: std::path::PathBuf,
    tmp: std::path::PathBuf,
    w: std::io::BufWriter<std::fs::File>,
    digest: sha2::Sha256,
    planned: Vec<u64>,
    /// The absolute file position each planned tensor starts at.
    starts: Vec<u64>,
    alignment: u64,
    next: usize,
    /// The absolute file position written up to.
    pos: u64,
}

impl GgufStreamWriter {
    pub fn create(
        path: &Path,
        metadata: &[(String, Value)],
        entries: &[PlannedEntry],
    ) -> Result<Self> {
        Self::create_placed(path, metadata, entries, &Placement::packed())
    }

    /// `create` with each tensor where `placement` puts it. An alignment other than the default
    /// is declared in the metadata under `general.alignment`, added when the caller has not.
    pub fn create_placed(
        path: &Path,
        metadata: &[(String, Value)],
        entries: &[PlannedEntry],
        placement: &Placement,
    ) -> Result<Self> {
        let io_err = |e: std::io::Error| Error(format!("gguf write {}: {e}", path.display()));
        let alignment = placement.alignment;
        if alignment == 0 || !alignment.is_multiple_of(DEFAULT_ALIGNMENT) {
            return Err(Error(format!(
                "gguf write {}: alignment {alignment} is not a multiple of {DEFAULT_ALIGNMENT}",
                path.display()
            )));
        }
        if !placement.starts.is_empty() && placement.starts.len() != entries.len() {
            return Err(Error(format!(
                "gguf write {}: {} starts for {} tensors",
                path.display(),
                placement.starts.len(),
                entries.len()
            )));
        }
        let mut metadata = metadata.to_vec();
        match metadata
            .iter()
            .find(|(k, _)| k == layout::ALIGNMENT_KEY)
            .map(|(_, v)| v.to_u64())
        {
            Some(Ok(declared)) if declared == alignment => {}
            Some(_) => {
                return Err(Error(format!(
                    "gguf write {}: the metadata declares an alignment other than {alignment}",
                    path.display()
                )))
            }
            None if alignment != DEFAULT_ALIGNMENT => metadata.push((
                layout::ALIGNMENT_KEY.to_string(),
                Value::U32(alignment as u32),
            )),
            None => {}
        }
        // Offsets are fixed-width, so the header's length does not depend on them: measured with
        // placeholders, then written with the real ones.
        let zeros = vec![0u64; entries.len()];
        let header_len = stream_header(&metadata, entries, &zeros, 0).len() as u64;
        let data_start = layout::align_up(header_len, alignment);
        let mut starts = Vec::with_capacity(entries.len());
        let mut end = data_start;
        for (i, e) in entries.iter().enumerate() {
            let a = placement
                .starts
                .get(i)
                .copied()
                .unwrap_or(alignment)
                .max(alignment);
            if a % alignment != 0 {
                return Err(Error(format!(
                    "gguf write {}: tensor {} starts on {a}, not a multiple of {alignment}",
                    path.display(),
                    e.name
                )));
            }
            let start = layout::align_up(end, a);
            starts.push(start);
            end = start + e.byte_len;
        }
        let mut header = stream_header(&metadata, entries, &starts, data_start);
        header.resize(data_start as usize, 0);
        let tmp = path.with_extension("gguf.partial");
        let f = std::fs::File::create(&tmp).map_err(io_err)?;
        let mut writer = Self {
            path: path.to_path_buf(),
            tmp,
            w: std::io::BufWriter::with_capacity(8 << 20, f),
            digest: sha2::Digest::new(),
            planned: entries.iter().map(|e| e.byte_len).collect(),
            starts,
            alignment,
            next: 0,
            pos: 0,
        };
        writer.put(&header)?;
        Ok(writer)
    }

    /// Bytes out to the file and into the digest.
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        sha2::Digest::update(&mut self.digest, bytes);
        self.pos += bytes.len() as u64;
        self.w
            .write_all(bytes)
            .map_err(|e| Error(format!("gguf write {}: {e}", self.path.display())))
    }

    /// Zeros up to `to`, an absolute file position at or after the current one.
    fn pad_to(&mut self, to: u64) -> Result<()> {
        let zeros = [0u8; 4096];
        let mut left = (to - self.pos) as usize;
        while left > 0 {
            let n = left.min(zeros.len());
            self.put(&zeros[..n])?;
            left -= n;
        }
        Ok(())
    }

    /// The next planned tensor's length, with the file padded up to where it starts.
    fn begin(&mut self) -> Result<u64> {
        let Some(&want) = self.planned.get(self.next) else {
            return Err(Error(format!(
                "gguf write {}: more tensors than planned",
                self.path.display()
            )));
        };
        self.pad_to(self.starts[self.next])?;
        Ok(want)
    }

    /// The next planned tensor's bytes, which must be exactly its planned length.
    pub fn append(&mut self, data: &[u8]) -> Result<()> {
        let want = self.begin()?;
        if data.len() as u64 != want {
            return Err(Error(format!(
                "gguf write {}: tensor {} has {} bytes, {want} planned",
                self.path.display(),
                self.next,
                data.len()
            )));
        }
        self.put(data)?;
        self.next += 1;
        Ok(())
    }

    /// The next planned tensor's bytes, in as many pieces as the caller produces them: a stack too
    /// large to build whole is written a slice at a time. The pieces must add up to its planned
    /// length.
    pub fn append_parts<I>(&mut self, parts: I) -> Result<()>
    where
        I: IntoIterator<Item = Result<Vec<u8>>>,
    {
        let want = self.begin()?;
        let mut wrote = 0u64;
        for part in parts {
            let part = part?;
            wrote += part.len() as u64;
            if wrote > want {
                return Err(Error(format!(
                    "gguf write {}: tensor {} overruns its {want} planned bytes",
                    self.path.display(),
                    self.next
                )));
            }
            self.put(&part)?;
        }
        if wrote != want {
            return Err(Error(format!(
                "gguf write {}: tensor {} has {wrote} bytes, {want} planned",
                self.path.display(),
                self.next
            )));
        }
        self.next += 1;
        Ok(())
    }

    /// Push what is written so far to the disk and drop it from the page cache: a file larger
    /// than memory then does not evict everything else on its way out.
    pub fn release_written(&mut self) -> Result<()> {
        let io_err = |e: std::io::Error| Error(format!("gguf write {}: {e}", self.path.display()));
        self.w.flush().map_err(io_err)?;
        self.w.get_ref().sync_data().map_err(io_err)?;
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // Safety: a valid descriptor; advice only.
            unsafe {
                libc::posix_fadvise(
                    self.w.get_ref().as_raw_fd(),
                    0,
                    0,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
        }
        Ok(())
    }

    /// Every planned tensor written: flush and move the file into place.
    pub fn finish(self) -> Result<()> {
        self.finish_with_digest().map(|_| ())
    }

    /// `finish`, returning the SHA-256 of the file written, as lowercase hex.
    pub fn finish_with_digest(mut self) -> Result<String> {
        let path = self.path.clone();
        let io_err = |e: std::io::Error| Error(format!("gguf write {}: {e}", path.display()));
        if self.next != self.planned.len() {
            let _ = std::fs::remove_file(&self.tmp);
            return Err(Error(format!(
                "gguf write {}: {} of {} planned tensors written",
                self.path.display(),
                self.next,
                self.planned.len()
            )));
        }
        // The file ends on its alignment, as every tensor does.
        let end = layout::align_up(self.pos, self.alignment);
        self.pad_to(end)?;
        self.w.flush().map_err(io_err)?;
        std::fs::rename(&self.tmp, &self.path).map_err(io_err)?;
        let digest = sha2::Digest::finalize(std::mem::take(&mut self.digest));
        Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
    }
}

impl Drop for GgufStreamWriter {
    fn drop(&mut self) {
        // A writer dropped before `finish` leaves no readable half-file behind.
        if self.next != self.planned.len() {
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::Content;

    /// The metadata a model carries - architecture, hyper-parameters, tokenizer arrays - has to
    /// come back byte-equal after a write, or a requantised model is served as a format it does
    /// not declare and closes its turn after a dozen tokens. Covers every value shape a model
    /// GGUF uses: a string, a scalar, a float, an array of strings and an array of ints.
    #[test]
    fn metadata_survives_the_write_read_round_trip() {
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("qwen3".into()),
            ),
            ("qwen3.block_count".to_string(), Value::U32(28)),
            ("qwen3.rope.freq_base".to_string(), Value::F32(1_000_000.0)),
            (
                "tokenizer.ggml.tokens".to_string(),
                Value::Array(vec![
                    Value::String("<a>".into()),
                    Value::String("bb".into()),
                    Value::String("c".into()),
                ]),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                Value::Array(vec![Value::I32(1), Value::I32(3), Value::I32(4)]),
            ),
        ];
        let entries = vec![GgufEntry {
            name: "blk.0.attn_norm.weight".to_string(),
            dims: vec![4],
            dtype: GgmlDType::F32,
            data: (0..16u8).collect(),
        }];
        let path = std::env::temp_dir().join(format!("loken-gguf-md-{}.gguf", std::process::id()));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let mut f = std::fs::File::open(&path).unwrap();
        let content = Content::read(&mut f).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(
            content
                .metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok().cloned())
                .as_deref(),
            Some("qwen3")
        );
        assert_eq!(
            content
                .metadata
                .get("qwen3.block_count")
                .and_then(|v| v.to_u64().ok()),
            Some(28)
        );
        match content.metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Array(a)) => {
                assert_eq!(a.len(), 3);
                assert_eq!(a[0].to_string().ok().cloned().as_deref(), Some("<a>"));
            }
            other => panic!("tokens round-tripped wrong: {other:?}"),
        }
        match content.metadata.get("tokenizer.ggml.token_type") {
            Some(Value::Array(a)) => assert_eq!(a.len(), 3),
            other => panic!("token_type round-tripped wrong: {other:?}"),
        }
        assert!(content.tensor_infos.contains_key("blk.0.attn_norm.weight"));
    }

    /// The digest a streaming write returns is the SHA-256 of the file it leaves, padding and
    /// header included, whichever way its tensors were appended.
    #[test]
    fn a_streamed_file_carries_its_own_digest() {
        use sha2::{Digest, Sha256};
        let md = vec![(
            "general.architecture".to_string(),
            Value::String("test".into()),
        )];
        let plan = vec![
            PlannedEntry {
                name: "a".into(),
                dims: vec![3],
                dtype: GgmlDType::F32,
                byte_len: 12,
            },
            PlannedEntry {
                name: "b".into(),
                dims: vec![5],
                dtype: GgmlDType::F32,
                byte_len: 20,
            },
        ];
        let path =
            std::env::temp_dir().join(format!("loken-gguf-digest-{}.gguf", std::process::id()));
        let mut w = GgufStreamWriter::create(&path, &md, &plan).unwrap();
        w.append(&[7u8; 12]).unwrap();
        w.append_parts([Ok(vec![1u8; 8]), Ok(vec![2u8; 12])])
            .unwrap();
        let digest = w.finish_with_digest().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let want: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(digest, want);
    }

    /// A placed write declares its alignment, starts every tensor on it and each tensor asked
    /// for on its own boundary, and reads back with the bytes it was given.
    #[test]
    fn a_placed_file_starts_each_tensor_where_asked() {
        use crate::tensor::quantized::gguf_file::open_mapped;
        let md = vec![(
            "general.architecture".to_string(),
            Value::String("test".into()),
        )];
        let plan: Vec<PlannedEntry> = [("a", 3usize), ("b", 5), ("c", 7)]
            .iter()
            .map(|&(n, len)| PlannedEntry {
                name: n.into(),
                dims: vec![len],
                dtype: GgmlDType::F32,
                byte_len: 4 * len as u64,
            })
            .collect();
        let placement = Placement {
            alignment: 64,
            starts: vec![64, 256, 64],
        };
        let path =
            std::env::temp_dir().join(format!("loken-gguf-placed-{}.gguf", std::process::id()));
        let mut w = GgufStreamWriter::create_placed(&path, &md, &plan, &placement).unwrap();
        for (i, e) in plan.iter().enumerate() {
            w.append(&vec![i as u8 + 1; e.byte_len as usize]).unwrap();
        }
        w.finish().unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            g.content
                .metadata
                .get(layout::ALIGNMENT_KEY)
                .and_then(|v| v.to_u64().ok()),
            Some(64)
        );
        for (i, (e, start)) in plan.iter().zip(&placement.starts).enumerate() {
            let at = g.content.tensor_data_offset + g.content.tensor_infos[&e.name].offset;
            assert_eq!(at % start, 0, "{} starts at {at}", e.name);
            let bytes = &g.mmap[at as usize..(at + e.byte_len) as usize];
            assert!(bytes.iter().all(|&b| b == i as u8 + 1));
        }
        assert_eq!(g.mmap.len() as u64 % placement.alignment, 0);
    }
}
