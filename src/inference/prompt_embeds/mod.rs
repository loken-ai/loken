//! A prompt sent as embeddings: the forms a completion's `prompt_embeds` arrives in, validated
//! and decoded into rows of the model's width.
//!
//! A base64 string holds one of three payloads, told apart by their first bytes:
//! - loken's own format, described in `docs/API.md`: a header naming the dtype, rows and width,
//!   the values, and a CRC-32 of everything before it;
//! - `torch.save` of one tensor, as vLLM clients send it, read by [`torch`] without unpickling;
//! - bare little-endian f32 values, rows of the model's width.
//!
//! A JSON array of rows of numbers is the fourth form.
//!
//! [`PromptEmbeds::from_json`] checks a payload's structure: base64, header, exact length,
//! checksum, archive and pickle. [`PromptEmbeds::into_values`] checks it against the model
//! (width, rows within the context window) before converting a value, then refuses a value that
//! is not finite.

mod torch;

#[cfg(test)]
mod tests;

/// The first bytes of loken's format. Its first four bytes read as a little-endian f32 are a
/// NaN, which a bare f32 payload is refused for, so the two forms never collide.
pub const MAGIC: [u8; 8] = [0x00, 0x00, 0xC0, 0xFF, b'L', b'K', b'E', b'M'];
/// The version of loken's format this reader takes.
pub const VERSION: u16 = 1;
/// Bytes of loken's header: magic, version, dtype, a reserved byte, rows, width and a reserved
/// word, so the values start 8-byte aligned.
pub const HEADER_BYTES: usize = 24;
/// Bytes of the CRC-32 that ends a payload in loken's format.
const CRC_BYTES: usize = 4;
/// The first bytes of a zip archive's local file header, which `torch.save` writes.
const ZIP_MAGIC: [u8; 4] = *b"PK\x03\x04";
/// The first bytes of `torch.save`'s legacy format: protocol 2 and the pickled magic number.
const TORCH_LEGACY_MAGIC: [u8; 14] = [
    0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19,
];

/// The element types a packed payload carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    F64,
}

impl Dtype {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "float32",
            Dtype::F16 => "float16",
            Dtype::Bf16 => "bfloat16",
            Dtype::F64 => "float64",
        }
    }

    /// The dtype a code of loken's header names: 0 f32, 1 f16, 2 bf16.
    fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Dtype::F32),
            1 => Some(Dtype::F16),
            2 => Some(Dtype::Bf16),
            _ => None,
        }
    }

    /// The little-endian element at the start of `b`, which holds at least `self.size()` bytes.
    fn read(self, b: &[u8]) -> f32 {
        match self {
            Dtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Dtype::F16 => half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
            Dtype::Bf16 => half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
            Dtype::F64 => {
                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
            }
        }
    }
}

/// `rows` by `width` values in `bytes`: element `(r, c)` is the element at index
/// `offset + r * row_stride + c * col_stride` counted from byte `start`. Every index the view
/// reaches lies inside `bytes`, checked by [`Packed::new`].
#[derive(Debug)]
struct Packed {
    bytes: Vec<u8>,
    start: usize,
    dtype: Dtype,
    rows: usize,
    width: usize,
    offset: usize,
    row_stride: usize,
    col_stride: usize,
}

/// The view's geometry, apart from the bytes it reads.
#[derive(Clone, Copy, Debug)]
struct Layout {
    dtype: Dtype,
    rows: usize,
    width: usize,
    offset: usize,
    row_stride: usize,
    col_stride: usize,
}

impl Packed {
    /// The view `layout` describes over `bytes[start..end]`; refused when it is empty or reaches
    /// past `end`.
    fn new(bytes: Vec<u8>, start: usize, end: usize, layout: Layout) -> Result<Self, String> {
        let Layout {
            dtype,
            rows,
            width,
            offset,
            row_stride,
            col_stride,
        } = layout;
        if rows == 0 || width == 0 {
            return Err(format!(
                "`prompt_embeds` holds {rows} rows of {width} values"
            ));
        }
        let last = (rows - 1)
            .checked_mul(row_stride)
            .and_then(|r| (width - 1).checked_mul(col_stride).map(|c| (r, c)))
            .and_then(|(r, c)| offset.checked_add(r)?.checked_add(c));
        let reach = last
            .and_then(|l| l.checked_add(1)?.checked_mul(dtype.size()))
            .and_then(|n| start.checked_add(n));
        match reach {
            Some(r) if end <= bytes.len() && r <= end => Ok(Packed {
                bytes,
                start,
                dtype,
                rows,
                width,
                offset,
                row_stride,
                col_stride,
            }),
            _ => Err("`prompt_embeds` describes values past the end of its data".into()),
        }
    }

    /// The values row after row, as f32.
    fn values(&self) -> Vec<f32> {
        let size = self.dtype.size();
        let mut out = Vec::with_capacity(self.rows * self.width);
        for r in 0..self.rows {
            let row = self.offset + r * self.row_stride;
            for c in 0..self.width {
                let at = self.start + (row + c * self.col_stride) * size;
                out.push(self.dtype.read(&self.bytes[at..at + size]));
            }
        }
        out
    }
}

#[derive(Debug)]
enum Form {
    Packed(Packed),
    RawF32(Vec<u8>),
    Rows { width: usize, values: Vec<f32> },
}

/// A completion's prompt as embeddings, its structure checked, not yet checked against a model.
#[derive(Debug)]
pub struct PromptEmbeds(Form);

impl PromptEmbeds {
    /// The payload of a `prompt_embeds` field: a base64 string, an array of one, or an array of
    /// rows of numbers.
    pub fn from_json(field: &serde_json::Value) -> Result<Self, String> {
        use serde_json::Value;
        match field {
            Value::String(b64) => Self::from_base64(b64),
            Value::Array(items) => match items.as_slice() {
                [Value::String(b64)] => Self::from_base64(b64),
                _ => Self::from_rows(items),
            },
            _ => Err(
                "`prompt_embeds` must be a base64 string, an array of them, or an array of rows"
                    .into(),
            ),
        }
    }

    fn from_base64(b64: &str) -> Result<Self, String> {
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("`prompt_embeds` is not base64: {e}"))?;
        Self::from_bytes(bytes)
    }

    /// A decoded payload, its form read from its first bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.starts_with(&MAGIC) {
            return read_loken(bytes).map(|p| PromptEmbeds(Form::Packed(p)));
        }
        if bytes.starts_with(&ZIP_MAGIC) {
            return torch::read(bytes).map(|p| PromptEmbeds(Form::Packed(p)));
        }
        if bytes.starts_with(&TORCH_LEGACY_MAGIC) {
            return Err(
                "`prompt_embeds` is in torch.save's legacy format; save it with the \
                        default zip serialization"
                    .into(),
            );
        }
        if bytes.is_empty() || !bytes.len().is_multiple_of(Dtype::F32.size()) {
            return Err(format!(
                "`prompt_embeds` holds {} bytes, not whole f32 values",
                bytes.len()
            ));
        }
        Ok(PromptEmbeds(Form::RawF32(bytes)))
    }

    fn from_rows(rows: &[serde_json::Value]) -> Result<Self, String> {
        const NOT_ROWS: &str = "`prompt_embeds` rows must be arrays of numbers";
        let width = rows
            .first()
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        if width == 0 {
            return Err("`prompt_embeds` must hold at least one row of numbers".into());
        }
        let mut values = Vec::with_capacity(rows.len() * width);
        for row in rows {
            let row = row.as_array().ok_or(NOT_ROWS)?;
            if row.len() != width {
                return Err(format!(
                    "`prompt_embeds` rows must all be {width} wide, one is {}",
                    row.len()
                ));
            }
            for v in row {
                values.push(v.as_f64().ok_or(NOT_ROWS)? as f32);
            }
        }
        Ok(PromptEmbeds(Form::Rows { width, values }))
    }

    /// The width the payload declares, `None` for bare f32 values, which take the model's.
    pub fn width(&self) -> Option<usize> {
        match &self.0 {
            Form::Packed(p) => Some(p.width),
            Form::Rows { width, .. } => Some(*width),
            Form::RawF32(_) => None,
        }
    }

    /// The values row after row, refused before any is converted when the rows are not `width`
    /// wide or when they fill `window` positions, and after when one is not finite.
    pub fn into_values(self, width: usize, window: usize) -> Result<Vec<f32>, String> {
        if width == 0 {
            return Err("the model declares no embedding width".into());
        }
        if let Some(sent) = self.width().filter(|&w| w != width) {
            return Err(format!(
                "`prompt_embeds` rows are {sent} wide but the model's embeddings are {width} wide"
            ));
        }
        let rows = match &self.0 {
            Form::Packed(p) => p.rows,
            Form::Rows { values, .. } => values.len() / width,
            Form::RawF32(bytes) => {
                let n = bytes.len() / Dtype::F32.size();
                if !n.is_multiple_of(width) {
                    return Err(format!(
                        "`prompt_embeds` holds {n} values, not whole rows of the model's width \
                         {width}"
                    ));
                }
                n / width
            }
        };
        if rows >= window {
            return Err(format!(
                "the prompt embeddings occupy {rows} KV positions but the context window is only \
                 {window}; increase context_length / num_ctx or send a shorter prompt"
            ));
        }
        let values = match self.0 {
            Form::Packed(p) => p.values(),
            Form::Rows { values, .. } => values,
            Form::RawF32(bytes) => bytes
                .chunks_exact(Dtype::F32.size())
                .map(|b| Dtype::F32.read(b))
                .collect(),
        };
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "`prompt_embeds` value {i} (row {}) is not a finite number",
                i / width
            ));
        }
        Ok(values)
    }
}

/// A payload in loken's format, checked whole before a value is read: magic, version, dtype,
/// reserved fields at zero, a length equal to what the header declares, and the CRC-32 of every
/// byte before the trailing one.
fn read_loken(bytes: Vec<u8>) -> Result<Packed, String> {
    let le_u16 = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let le_u32 =
        |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    if bytes.len() < HEADER_BYTES + CRC_BYTES {
        return Err(format!(
            "`prompt_embeds` holds {} bytes, shorter than the {} of a header and checksum",
            bytes.len(),
            HEADER_BYTES + CRC_BYTES
        ));
    }
    let version = le_u16(8);
    if version != VERSION {
        return Err(format!(
            "`prompt_embeds` is version {version} of loken's format; this server reads version \
             {VERSION}"
        ));
    }
    let dtype = Dtype::from_code(bytes[10]).ok_or_else(|| {
        format!(
            "`prompt_embeds` names dtype code {}; the codes are 0 float32, 1 float16, 2 bfloat16",
            bytes[10]
        )
    })?;
    if bytes[11] != 0 || le_u32(20) != 0 {
        return Err("`prompt_embeds` has a reserved header field that is not zero".into());
    }
    let (rows, width) = (le_u32(12) as usize, le_u32(16) as usize);
    let expected = rows
        .checked_mul(width)
        .and_then(|n| n.checked_mul(dtype.size()))
        .and_then(|n| n.checked_add(HEADER_BYTES + CRC_BYTES));
    if expected != Some(bytes.len()) {
        return Err(format!(
            "`prompt_embeds` declares {rows} rows of {width} {} values but holds {} bytes, \
             not the {} they take: truncated or padded",
            dtype.name(),
            bytes.len(),
            expected.map_or_else(|| "overflowing".to_string(), |n| n.to_string()),
        ));
    }
    let body = bytes.len() - CRC_BYTES;
    let sent = le_u32(body);
    let computed = crc32(&bytes[..body]);
    if sent != computed {
        return Err(format!(
            "`prompt_embeds` fails its checksum (CRC-32 {computed:#010x}, sent {sent:#010x}): \
             the payload is corrupted"
        ));
    }
    Packed::new(
        bytes,
        HEADER_BYTES,
        body,
        Layout {
            dtype,
            rows,
            width,
            offset: 0,
            row_stride: width,
            col_stride: 1,
        },
    )
}

/// The CRC-32 of `bytes`, the polynomial zip, gzip and zlib's `crc32` use.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = flate2::Crc::new();
    crc.update(bytes);
    crc.sum()
}
