use super::host::{AlignedBytes, QHostStorage};
use super::*;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

// De-wrapped surface (compat->native): the GGUF reader takes/returns the
// concrete native `Device`/`Error`/`Result` (and `Tensor`, already native
// via the quantized module).
use self::layout::{read_string, read_u32, read_u64, value_type};
use crate::tensor::{Device, Error, Result};

/// The GGUF container layout: the single declaration the reader in this module
/// and the writer in [`crate::tensor::gguf_write`] both consume.
///
/// A container is bytes before it is anything else, and every byte of one is
/// decided here - the magic, the version numbers, the order and width of the
/// header fields, the order of the tensor-descriptor fields including the
/// dimension reversal, the metadata value-type ids and the payload alignment
/// rule. Stated once, a reader and a writer of the same file cannot disagree
/// about what the file is; stated twice, they agree only until one is edited.
pub mod layout {
    use super::{TensorInfo, VersionedMagic};
    use crate::tensor::quantized::GgmlDType;
    use crate::tensor::{Error, Result};
    use std::io::Read;

    /// File magic: the four bytes `GGUF`, read little-endian.
    pub const MAGIC: u32 = 0x4655_4747;

    /// The magic as it sits on disk, for checks that peek at a file's first
    /// four bytes instead of parsing it.
    pub const MAGIC_BYTES: [u8; 4] = MAGIC.to_le_bytes();

    /// The version [`crate::tensor::gguf_write::write_gguf`] emits.
    pub const WRITE_VERSION: VersionedMagic = VersionedMagic::GgufV3;

    /// Payload alignment when the metadata declares none.
    pub const DEFAULT_ALIGNMENT: u64 = 32;

    /// The metadata key that overrides [`DEFAULT_ALIGNMENT`].
    pub const ALIGNMENT_KEY: &str = "general.alignment";

    /// Metadata value-type ids, in the order the format assigns them. The
    /// reader dispatches on these; a writer emitting key/value pairs tags them
    /// with the same ids.
    pub mod value_type {
        pub const U8: u32 = 0;
        pub const I8: u32 = 1;
        pub const U16: u32 = 2;
        pub const I16: u32 = 3;
        pub const U32: u32 = 4;
        pub const I32: u32 = 5;
        pub const F32: u32 = 6;
        pub const BOOL: u32 = 7;
        pub const STRING: u32 = 8;
        pub const ARRAY: u32 = 9;
        pub const U64: u32 = 10;
        pub const I64: u32 = 11;
        pub const F64: u32 = 12;
    }

    /// The width of one fixed-width field. Every one is little-endian.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Scalar {
        U32,
        U64,
    }

    /// The container header, in file order: magic, version, tensor count,
    /// key/value count. Both directions traverse this one list, so the order
    /// and the widths are stated once and honoured the same way by each.
    pub const FILE_HEADER: [Scalar; 4] = [Scalar::U32, Scalar::U32, Scalar::U64, Scalar::U64];

    /// Read the listed fields in order, each widened to `u64`.
    pub fn read_scalars<R: Read, const N: usize>(
        r: &mut R,
        fields: &[Scalar; N],
    ) -> Result<[u64; N]> {
        let mut out = [0u64; N];
        for (slot, field) in out.iter_mut().zip(fields) {
            *slot = match field {
                Scalar::U32 => u64::from(read_u32(r)?),
                Scalar::U64 => read_u64(r)?,
            };
        }
        Ok(out)
    }

    /// Append the listed fields in order, each narrowed to its declared width.
    pub fn write_scalars<const N: usize>(
        out: &mut Vec<u8>,
        fields: &[Scalar; N],
        values: &[u64; N],
    ) {
        for (field, value) in fields.iter().zip(values) {
            match field {
                Scalar::U32 => out.extend_from_slice(&(*value as u32).to_le_bytes()),
                Scalar::U64 => out.extend_from_slice(&value.to_le_bytes()),
            }
        }
    }

    pub(super) fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)
            .map_err(|e| Error(format!("gguf read: {e}")))?;
        Ok(u32::from_le_bytes(b))
    }

    pub(super) fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)
            .map_err(|e| Error(format!("gguf read: {e}")))?;
        Ok(u64::from_le_bytes(b))
    }

    /// A string is its byte length as a `u64`, then the bytes.
    pub(super) fn read_string<R: Read>(r: &mut R) -> Result<String> {
        let len = read_u64(r)? as usize;
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf)
            .map_err(|e| Error(format!("gguf read: {e}")))?;
        String::from_utf8(buf).map_err(|e| Error(format!("gguf string: {e}")))
    }

    /// The writing half of [`read_string`].
    pub fn write_string(out: &mut Vec<u8>, s: &str) {
        write_scalars(out, &[Scalar::U64], &[s.len() as u64]);
        out.extend_from_slice(s.as_bytes());
    }

    /// The file stores dimensions innermost-first; the rest of the stack keeps
    /// them row-major. The mapping is its own inverse, so this one statement of
    /// it serves the reader and the writer alike.
    pub fn flip_dim_order(dims: &mut [usize]) {
        dims.reverse();
    }

    /// The payload begins at the first multiple of `alignment` at or after the
    /// directory's end, and each tensor at the first multiple at or after the
    /// previous one's end; the gaps between are zero padding.
    pub fn align_up(offset: u64, alignment: u64) -> u64 {
        offset.next_multiple_of(alignment)
    }

    /// One tensor-directory entry, in file order:
    ///
    /// 1. name - `u64` byte length, then the bytes
    /// 2. dimension count - `u32`
    /// 3. the dimensions innermost-first - `u64` each
    /// 4. block dtype id - `u32`
    /// 5. payload offset - `u64`, relative to the aligned start of the data
    ///
    /// The two directions keep their own shapes, because they want different
    /// ones: the writer borrows the name and dims its caller already holds, the
    /// reader hands back owned ones. What they share is this field list and the
    /// order it is walked in.
    pub struct TensorDescriptor<'a> {
        pub name: &'a str,
        /// Row-major, the order the rest of the stack states shapes in.
        pub dims: &'a [usize],
        pub dtype: GgmlDType,
        pub offset: u64,
    }

    impl TensorDescriptor<'_> {
        /// Append this descriptor to `out`.
        pub fn encode(&self, out: &mut Vec<u8>) {
            write_string(out, self.name);
            write_scalars(out, &[Scalar::U32], &[self.dims.len() as u64]);
            let mut dims = self.dims.to_vec();
            flip_dim_order(&mut dims);
            for d in dims {
                write_scalars(out, &[Scalar::U64], &[d as u64]);
            }
            write_scalars(
                out,
                &[Scalar::U32, Scalar::U64],
                &[u64::from(self.dtype.to_u32()), self.offset],
            );
        }

        /// Read one descriptor: the tensor's name and the directory entry it
        /// describes, dimensions restored to row-major.
        pub fn decode<R: Read>(r: &mut R) -> Result<(String, TensorInfo)> {
            let name = read_string(r)?;
            let [n_dims] = read_scalars(r, &[Scalar::U32])?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let [d] = read_scalars(r, &[Scalar::U64])?;
                dims.push(d as usize);
            }
            flip_dim_order(&mut dims);
            let [dtype, offset] = read_scalars(r, &[Scalar::U32, Scalar::U64])?;
            let info = TensorInfo {
                ggml_dtype: GgmlDType::from_u32(dtype as u32)?,
                shape: dims.into(),
                offset,
            };
            Ok((name, info))
        }
    }
}

/// GGUF metadata value with `Result` accessors.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

fn read_value<R: Read>(r: &mut R, vtype: u32) -> Result<Value> {
    let mut b8 = [0u8; 8];
    Ok(match vtype {
        value_type::U8 => {
            r.read_exact(&mut b8[..1])
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::U8(b8[0])
        }
        value_type::I8 => {
            r.read_exact(&mut b8[..1])
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::I8(b8[0] as i8)
        }
        value_type::U16 => {
            r.read_exact(&mut b8[..2])
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::U16(u16::from_le_bytes([b8[0], b8[1]]))
        }
        value_type::I16 => {
            r.read_exact(&mut b8[..2])
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::I16(i16::from_le_bytes([b8[0], b8[1]]))
        }
        value_type::U32 => Value::U32(read_u32(r)?),
        value_type::I32 => {
            r.read_exact(&mut b8[..4])
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::I32(i32::from_le_bytes([b8[0], b8[1], b8[2], b8[3]]))
        }
        value_type::F32 => {
            r.read_exact(&mut b8[..4])
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::F32(f32::from_le_bytes([b8[0], b8[1], b8[2], b8[3]]))
        }
        value_type::BOOL => {
            r.read_exact(&mut b8[..1])
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::Bool(b8[0] != 0)
        }
        value_type::STRING => Value::String(read_string(r)?),
        value_type::ARRAY => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            let mut items = Vec::with_capacity(count.min(1 << 20));
            for _ in 0..count {
                items.push(read_value(r, elem_type)?);
            }
            Value::Array(items)
        }
        value_type::U64 => Value::U64(read_u64(r)?),
        value_type::I64 => {
            r.read_exact(&mut b8)
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::I64(i64::from_le_bytes(b8))
        }
        value_type::F64 => {
            r.read_exact(&mut b8)
                .map_err(|e| Error(format!("gguf: {e}")))?;
            Value::F64(f64::from_le_bytes(b8))
        }
        other => return Err(Error(format!("gguf: unknown value type {other}"))),
    })
}

macro_rules! value_accessor {
    ($fn:ident, $variant:ident, $ty:ty) => {
        pub fn $fn(&self) -> Result<$ty> {
            match self {
                Self::$variant(v) => Ok(*v),
                other => Err(Error::msg(format!(
                    concat!(stringify!($fn), ": not a ", stringify!($variant), " ({:?})"),
                    other
                ))),
            }
        }
    };
}

impl Value {
    value_accessor!(to_u8, U8, u8);
    value_accessor!(to_i8, I8, i8);
    value_accessor!(to_u16, U16, u16);
    value_accessor!(to_i16, I16, i16);
    value_accessor!(to_i32, I32, i32);
    value_accessor!(to_f64, F64, f64);
    value_accessor!(to_bool, Bool, bool);

    /// Which variants carry a whole number, and what number, stated once.
    ///
    /// A metadata key's width is the writer's choice, not the reader's: the same
    /// `block_count` arrives as `U32` from one converter and `I32` from the next. So the
    /// integer is widened to a type that holds every GGUF integer width with its sign
    /// intact, and each accessor below narrows to what it promised.
    fn whole_number(&self) -> Option<i128> {
        Some(match *self {
            Self::U8(v) => v.into(),
            Self::I8(v) => v.into(),
            Self::U16(v) => v.into(),
            Self::I16(v) => v.into(),
            Self::U32(v) => v.into(),
            Self::I32(v) => v.into(),
            Self::U64(v) => v.into(),
            Self::I64(v) => v.into(),
            _ => return None,
        })
    }

    /// Any integer family converts, provided the value fits.
    pub fn to_u32(&self) -> Result<u32> {
        self.to_u64().map(|v| v as u32)
    }

    /// A whole number that is not negative, whichever width it was written at.
    pub fn to_u64(&self) -> Result<u64> {
        self.whole_number()
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| Error::msg(format!("to_u64: not an unsigned value ({self:?})")))
    }

    /// A whole number written signed, or written unsigned in a width narrower than the
    /// answer. `U64` is refused rather than range-checked: a key stored at the full
    /// unsigned width is not a key that was meant to be read as signed.
    pub fn to_i64(&self) -> Result<i64> {
        self.whole_number()
            .filter(|_| !matches!(self, Self::U64(_)))
            .and_then(|v| i64::try_from(v).ok())
            .ok_or_else(|| Error::msg(format!("to_i64: not an integer value ({self:?})")))
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            Self::F64(v) => Ok(*v as f32),
            other => Err(Error::msg(format!("to_f32: not a float value ({other:?})"))),
        }
    }

    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(s) => Ok(s),
            other => Err(Error::msg(format!("to_string: not a string ({other:?})"))),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            other => Err(Error::msg(format!("to_vec: not an array ({other:?})"))),
        }
    }
}

/// Tensor directory entry. `shape` is in the row-major order the rest
/// of the stack uses (the GGUF file stores dims reversed; the reader
/// un-reverses).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: Shape,
    pub offset: u64,
}

impl TensorInfo {
    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.elem_count() / self.ggml_dtype.block_size() * self.ggml_dtype.type_size()
    }

    /// legacy-shaped: read this tensor's blocks from `reader` (offsets
    /// relative to `tensor_data_offset`) and place them on `device`.
    pub fn read<R: Read + Seek>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
        device: &Device,
    ) -> Result<QTensor> {
        use std::io::SeekFrom;
        let n = self.size_in_bytes();
        reader
            .seek(SeekFrom::Start(tensor_data_offset + self.offset))
            .map_err(|e| Error::msg(format!("gguf tensor seek: {e}")))?;
        let mut raw = vec![0u8; n];
        reader
            .read_exact(&mut raw)
            .map_err(|e| Error::msg(format!("gguf tensor read: {e}")))?;
        let qt = QHostTensor::from_bytes(&raw, self.ggml_dtype, self.shape.dims().to_vec())?;
        QTensor::from_native(Arc::new(qt), device)
    }

    /// Zero-copy read from an mmap'd file: slice the tensor bytes DIRECTLY
    /// out of `mmap` instead of `read_exact`ing them into a fresh Vec first.
    /// The loader already holds `mmap: &[u8]` + `tensor_data_offset`, so the
    /// `read`-via-Cursor path above copies the whole model an extra time
    /// (mmap -> raw Vec -> from_bytes' AlignedBytes -> GPU = 2 host copies of
    /// 7.5 GB). This drops the `raw` copy (mmap slice -> AlignedBytes -> GPU).
    pub fn read_slice(
        &self,
        mmap: &[u8],
        tensor_data_offset: u64,
        device: &Device,
    ) -> Result<QTensor> {
        self.read_slice_owned(mmap, tensor_data_offset, device, None)
    }

    /// As `read_slice`, but when `owner` (the `Arc<Mmap>` that backs `mmap`) is
    /// provided, build the native QTensor as a ZERO-COPY VIEW into the mmap
    /// instead of copying its bytes into an owned AlignedBytes buffer. The
    /// `_owner` Arc pins the mmap for the tensor's lifetime (safe: read-only,
    /// never mutated/moved). This removes the last host copy of the model
    /// (mmap -> AlignedBytes, ~7.5 GB) AND the committed heap it occupied - the
    /// GPU upload then reads straight from the mmap view. Falls back to the copy
    /// path when there's no owner or the tensor isn't 8-byte aligned in the file.
    pub fn read_slice_owned(
        &self,
        mmap: &[u8],
        tensor_data_offset: u64,
        device: &Device,
        owner: Option<&Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Result<QTensor> {
        let n = self.size_in_bytes();
        let start = (tensor_data_offset + self.offset) as usize;
        let end = start
            .checked_add(n)
            .ok_or_else(|| Error::msg("gguf tensor slice overflow"))?;
        let raw = mmap.get(start..end).ok_or_else(|| {
            Error::msg(format!(
                "gguf tensor out of mmap bounds: [{start}..{end}] > {}",
                mmap.len()
            ))
        })?;
        let dims = self.shape.dims().to_vec();
        if let Some(owner) = owner {
            // Zero-copy view (owner Arc pins the backing). view() checks 8-byte
            // alignment; if the tensor isn't aligned in the file it errors, and
            // we fall through to the owned copy below.
            // SAFETY: `raw` above came from `mmap.get(start..end)`, so the
            // range is in bounds of this mapping, and `owner` keeps the mapping
            // alive for as long as the view exists.
            if let Ok(qt) = unsafe {
                QHostTensor::view(
                    owner.clone(),
                    mmap.as_ptr(),
                    start,
                    n,
                    self.ggml_dtype,
                    dims.clone(),
                )
            } {
                return QTensor::from_native(Arc::new(qt), device);
            }
        }
        let qt = QHostTensor::from_bytes(raw, self.ggml_dtype, dims)?;
        QTensor::from_native(Arc::new(qt), device)
    }
}

/// A parsed GGUF whose tensor data stays memory-mapped: the header for lookups
/// plus the mapping every read views into. THE handle to thread through loaders
/// instead of re-opening and re-parsing the file - a re-parse through a plain
/// reader silently drops the mapping and copies every tensor into owned host
/// memory, visible only as committed RAM on exactly the models least able to
/// afford it.
pub struct MappedGguf {
    pub content: Content,
    pub mmap: std::sync::Arc<memmap2::Mmap>,
}

impl std::ops::Deref for MappedGguf {
    type Target = Content;
    fn deref(&self) -> &Content {
        &self.content
    }
}

impl MappedGguf {
    /// Zero-copy read of one tensor, pinned by the mapping.
    pub fn tensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        self.content.tensor_mapped(&self.mmap[..], name, device)
    }

    /// A cursor over the mapped bytes, where callers used to hold `&mut File`.
    pub fn reader(&self) -> std::io::Cursor<&[u8]> {
        std::io::Cursor::new(&self.mmap[..])
    }
}

/// Open + map + parse in one step, the mapping adopted as the tensor-data owner.
pub fn open_mapped<P: AsRef<std::path::Path>>(path: P) -> Result<MappedGguf> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).map_err(|e| Error(format!("open {}: {e}", path.display())))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| Error(format!("mmap {}: {e}", path.display())))?;
    let _ = mmap.advise(memmap2::Advice::WillNeed);
    let mmap = std::sync::Arc::new(mmap);
    let content = Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())?;
    Ok(MappedGguf { content, mmap })
}

/// Parse over a fresh private mapping of `file`, adopting it as the tensor-data
/// owner - the drop-in upgrade for reader-based sites that hold the File anyway.
pub fn read_mapped_file(file: &std::fs::File) -> Result<Content> {
    let mmap = unsafe { memmap2::Mmap::map(file) }.map_err(|e| Error(format!("gguf mmap: {e}")))?;
    let _ = mmap.advise(memmap2::Advice::WillNeed);
    let mmap = std::sync::Arc::new(mmap);
    Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
}

/// Header-only parse for metadata probes. The explicit entry point that states no
/// tensor payload will be read, so the call does not look like the dropped-mmap
/// defect the invariant gate hunts for.
pub fn open_header<P: AsRef<std::path::Path>>(path: P) -> Result<Content> {
    let path = path.as_ref();
    let mut f =
        std::fs::File::open(path).map_err(|e| Error(format!("open {}: {e}", path.display())))?;
    Content::read(&mut f)
}

/// GGUF container version (the reader accepts v2/v3; one loader
/// synthesizes a Content literal with it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    /// The version number a variant is written as.
    pub fn to_u32(self) -> u32 {
        match self {
            Self::GgufV1 => 1,
            Self::GgufV2 => 2,
            Self::GgufV3 => 3,
        }
    }

    /// The variant a version number names, for the versions whose container is
    /// the one [`layout`] declares. Version 1 sizes its counts and its string
    /// lengths differently, so it is named here but not parsed.
    pub fn from_u32(version: u32) -> Result<Self> {
        match version {
            2 => Ok(Self::GgufV2),
            3 => Ok(Self::GgufV3),
            v => Err(Error(format!("unsupported GGUF version {v}"))),
        }
    }
}

/// Parsed GGUF container: metadata + tensor directory. ALL-public
/// fields (a loader builds a synthetic Content via a struct literal).
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
    /// Optional `Arc<Mmap>` (as `dyn Any`) backing the tensor data. When set,
    /// the loader reads weights as zero-copy VIEWS into the mmap (no host copy,
    /// the Arc pins it) instead of copying each tensor into owned host bytes.
    /// Default None -> the copy path. The engine sets this; dev bins don't.
    pub mmap_owner: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

impl Content {
    /// The directory entry for `name`, or the one error a missing tensor gives.
    ///
    /// Every read path starts here - through the mapping, through a reader, or host-side  -
    /// so a caller that asks for a name the file does not carry cannot tell which path it
    /// took from the message, there being nothing to tell apart.
    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor_infos
            .get(name)
            .ok_or_else(|| Error(format!("gguf: no tensor `{name}`")))
    }

    /// Parse a GGUF header AND record what backs the tensor data, in one step.
    ///
    /// `read` leaves `mmap_owner` at None, which is right for a reader that owns no
    /// mapping and wrong everywhere the bytes came from one: the loader then copies
    /// every tensor into owned host memory instead of viewing it. Nothing reports the
    /// difference - the model loads either way - so the copy is only visible as
    /// committed RAM, and it lands on exactly the models that can least afford it,
    /// the ones large enough to have triggered a re-parse in the first place.
    ///
    /// Parsing and adopting the owner together is what keeps them from drifting apart:
    /// a re-parse that forgets the second call is silent, and there is no third place
    /// to notice.
    pub fn read_mapped<R: Read + Seek>(
        r: &mut R,
        owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<Self> {
        let mut c = Self::read(r)?;
        c.mmap_owner = Some(owner);
        Ok(c)
    }

    /// Read one tensor out of `data` (the full file bytes this header was parsed
    /// from), honouring `mmap_owner`: parsed over a mapping, the weight is a
    /// zero-copy view pinned by the owner; otherwise an owned host copy. The
    /// reader-based [`Content::tensor`] cannot do this - it has no owner to pin
    /// with - which is how ten call sites silently lost their mmap backing.
    pub fn tensor_mapped(&self, data: &[u8], name: &str, device: &Device) -> Result<QTensor> {
        let info = self.info(name)?;
        info.read_slice_owned(
            data,
            self.tensor_data_offset,
            device,
            self.mmap_owner.as_ref(),
        )
    }

    /// Drop the page-cache residency of one layer's tensors, once they live on a card.
    ///
    /// Loading reads the file front to back, and every byte read stays cached until the
    /// kernel needs the room. On a machine whose RAM is smaller than the file that room is
    /// found by swapping the process's own anonymous memory - its repacked host layers, its
    /// embeddings - which then comes back page by page, mid-decode, from a compressed swap.
    /// Measured on a 42.5 GB model over 64 GB of RAM: 12 GB of the process pushed to zram
    /// during the load, and the same cell decoding anywhere between 1.5 and 2.2 tok/s from one
    /// run to the next depending on what had been pushed. Released here, layer by layer as
    /// each one is uploaded, the cache never holds more than a layer beyond what the host will
    /// actually read. Advice only: a page dropped and touched again comes back from the file.
    pub fn release_layer_pages(&self, layer: usize) {
        let Some(owner) = self.mmap_owner.as_ref() else {
            return;
        };
        let Some(mmap) = owner.downcast_ref::<memmap2::Mmap>() else {
            return;
        };
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
        let prefix = format!("blk.{layer}.");
        let total = mmap.len();
        for (name, info) in self.tensor_infos.iter() {
            if !name.starts_with(&prefix) {
                continue;
            }
            let start = (self.tensor_data_offset + info.offset) as usize;
            let a = start / page * page;
            let b = (start + info.size_in_bytes()).div_ceil(page) * page;
            let b = b.min(total);
            if a < b {
                // Read-only shared file mapping: DontNeed discards residency, never data.
                let _ = unsafe {
                    mmap.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, a, b - a)
                };
            }
        }
    }
    /// Parse a GGUF v2/v3 header: magic, metadata KVs, tensor
    /// directory (dims un-reversed to row-major), aligned data offset.
    pub fn read<R: Read + Seek>(r: &mut R) -> Result<Self> {
        let [magic, version, tensor_count, kv_count] =
            layout::read_scalars(r, &layout::FILE_HEADER)?;
        if magic != u64::from(layout::MAGIC) {
            return Err(Error(format!("not a GGUF file (magic {magic:#x})")));
        }
        let magic = VersionedMagic::from_u32(version as u32)?;
        let (tensor_count, kv_count) = (tensor_count as usize, kv_count as usize);

        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = read_string(r)?;
            let vtype = read_u32(r)?;
            metadata.insert(key, read_value(r, vtype)?);
        }

        let mut tensor_infos = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let (name, info) = layout::TensorDescriptor::decode(r)?;
            tensor_infos.insert(name, info);
        }

        let alignment = metadata
            .get(layout::ALIGNMENT_KEY)
            .and_then(|v| v.to_u64().ok())
            .unwrap_or(layout::DEFAULT_ALIGNMENT);
        let position = r
            .stream_position()
            .map_err(|e| Error(format!("gguf pos: {e}")))?;
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset: layout::align_up(position, alignment),
            mmap_owner: None,
        })
    }

    /// Read tensor `name`'s blocks and place them on `device`
    /// (signature: reader + name + device).
    pub fn tensor<R: Read + Seek>(
        &self,
        r: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<QTensor> {
        let info = self.info(name)?;
        // Parsed over a mapping, read through it: a zero-copy view pinned by the
        // owner instead of a copy through the reader. Every reader-based caller
        // upgrades the moment its header is parsed mapped, with no signature to
        // change - the reader is only used when there is nothing better.
        if let Some(owner) = self.mmap_owner.as_ref() {
            if let Ok(m) = owner.clone().downcast::<memmap2::Mmap>() {
                return info.read_slice_owned(
                    &m[..],
                    self.tensor_data_offset,
                    device,
                    self.mmap_owner.as_ref(),
                );
            }
        }
        info.read(r, self.tensor_data_offset, device)
    }

    /// Read tensor `name`'s raw blocks into a host-side
    /// [`QHostTensor`] (no device upload - the DiT/QVarBuilder and
    /// zero-copy-view loaders' entry point).
    pub fn host_tensor<R: Read + Seek>(&self, r: &mut R, name: &str) -> Result<QHostTensor> {
        let info = self.info(name)?;
        r.seek(SeekFrom::Start(self.tensor_data_offset + info.offset))
            .map_err(|e| Error(format!("gguf seek: {e}")))?;
        let mut data = AlignedBytes::zeroed(info.size_in_bytes());
        r.read_exact(data.as_mut_slice())
            .map_err(|e| Error(format!("gguf tensor read: {e}")))?;
        Ok(QHostTensor {
            storage: QHostStorage::Owned(data),
            dtype: info.ggml_dtype,
            dims: info.shape.dims().to_vec(),
            id: crate::tensor::quantized::host::next_tensor_id(),
        })
    }
}
