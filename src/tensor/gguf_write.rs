//! Minimal GGUF v3 writer - exactly the subset `quant::gguf_file::Content::read` consumes
//! (zero metadata KVs, the default data alignment). Used to persist one-time checkpoint
//! conversions (fp8-scaled safetensors -> block-quantized sidecar) so later loads read the
//! quantized bytes straight from disk instead of re-decoding the source checkpoint.
//!
//! What the container IS is not restated here: the magic, the version, the header field
//! order, the tensor-descriptor field order (including the dimension reversal) and the
//! alignment rule all come from `gguf_file::layout`, which the reader consumes as well.

use super::quantized::gguf_file::layout::{
    self, TensorDescriptor, DEFAULT_ALIGNMENT, FILE_HEADER, MAGIC, WRITE_VERSION,
};
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

/// Write `entries` as a GGUF v3 file. Atomic: writes `<path>.partial` then renames, so a
/// crashed or cancelled write never leaves a readable half-file behind.
pub fn write_gguf(path: &Path, entries: &[GgufEntry]) -> Result<()> {
    let io_err = |e: std::io::Error| Error(format!("gguf write {}: {e}", path.display()));

    let mut header: Vec<u8> = Vec::with_capacity(64 * entries.len());
    layout::write_scalars(
        &mut header,
        &FILE_HEADER,
        &[
            u64::from(MAGIC),
            u64::from(WRITE_VERSION.to_u32()),
            entries.len() as u64,
            0, // no metadata key/value pairs
        ],
    );

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
