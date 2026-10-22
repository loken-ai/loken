//! The one file format of everything a cache keeps on disk: a header that names the kind of
//! file and its version, then a body that is stored as it is or compressed. A reader checks the
//! header before it trusts the body, and a body is written whole or not at all.
//!
//! ```text
//! "LKC1" | kind 4 bytes | version u16 | flags u16 | body length u64 | body
//! ```
//!
//! Numbers are little-endian. Flag bit 0 says the body is compressed: its bytes regrouped by
//! position in each 4-byte word, then zstd, which gives long runs when the words are floats
//! that carry few distinct values. A body is compressed only when that makes it clearly smaller.

use std::io::{Error, ErrorKind, Result};

const MAGIC: &[u8; 4] = b"LKC1";
/// Bytes of the header: the magic, the kind, the version, the flags and the body length.
pub const HEADER_LEN: usize = 20;
const FLAG_PACKED: u16 = 1;
/// A body smaller than this is stored as it is: the saving would not be worth a frame.
const PACK_MIN_BYTES: usize = 4096;
/// A body is stored compressed only when it comes out below this share of its size.
const PACK_MAX_SHARE: f64 = 0.9;

/// What a file holds, four bytes that appear in the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kind(pub [u8; 4]);

/// The index of a decode-state cache: how recently and how often each state was used.
pub const STATE_INDEX: Kind = Kind(*b"SIDX");
/// The block size of each scope of a decode-state cache.
pub const STATE_GEOMETRY: Kind = Kind(*b"SGEO");
/// A decode-state snapshot: the blocks it names and the model's own payload.
pub const STATE_MANIFEST: Kind = Kind(*b"SMAN");
/// A block of decode state shared by the snapshots that begin with the same tokens.
pub const STATE_BLOCK: Kind = Kind(*b"SBLK");
/// The routing counts of a model's experts.
pub const EXPERT_PROFILE: Kind = Kind(*b"XPRF");
/// A snapshot of the KV disk tier: its tokens and the blocks that hold their rows.
pub const KV_MANIFEST: Kind = Kind(*b"KVMF");

fn bad(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

/// `bytes` with the bytes at each position of a 4-byte word gathered together. A tail shorter
/// than a word stays as it is.
fn shuffle4(bytes: &[u8]) -> Vec<u8> {
    let words = bytes.len() / 4;
    let mut out = vec![0u8; bytes.len()];
    for (i, w) in bytes.chunks_exact(4).enumerate() {
        for (k, b) in w.iter().enumerate() {
            out[k * words + i] = *b;
        }
    }
    out[words * 4..].copy_from_slice(&bytes[words * 4..]);
    out
}

/// The inverse of [`shuffle4`].
fn unshuffle4(bytes: &[u8]) -> Vec<u8> {
    let words = bytes.len() / 4;
    let mut out = vec![0u8; bytes.len()];
    for i in 0..words {
        for k in 0..4 {
            out[i * 4 + k] = bytes[k * words + i];
        }
    }
    out[words * 4..].copy_from_slice(&bytes[words * 4..]);
    out
}

/// The file for a body of `kind` at `version`: compressed when `compress` is set and that
/// saves enough, else as it is.
pub fn seal(kind: Kind, version: u16, body: &[u8], compress: bool) -> Vec<u8> {
    let packed = (compress && body.len() >= PACK_MIN_BYTES)
        .then(|| zstd::bulk::compress(&shuffle4(body), 1).ok())
        .flatten()
        .filter(|p| (p.len() + HEADER_LEN) as f64 <= body.len() as f64 * PACK_MAX_SHARE);
    let (flags, stored) = match &packed {
        Some(p) => (FLAG_PACKED, p.as_slice()),
        None => (0, body),
    };
    let mut out = Vec::with_capacity(HEADER_LEN + stored.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&kind.0);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(stored);
    out
}

/// The version and flags of the file whose first bytes are `prefix`, `None` when they are not
/// the header of a file of `kind`. A reader that wants only the start of a file checks it so.
pub fn peek(kind: Kind, prefix: &[u8]) -> Option<(u16, bool)> {
    if prefix.len() < HEADER_LEN || &prefix[..4] != MAGIC || prefix[4..8] != kind.0 {
        return None;
    }
    let version = u16::from_le_bytes([prefix[8], prefix[9]]);
    let flags = u16::from_le_bytes([prefix[10], prefix[11]]);
    Some((version, flags & FLAG_PACKED != 0))
}

/// The version and body of `file`, refused when it is not a file of `kind` or is cut short.
pub fn open(kind: Kind, file: Vec<u8>) -> Result<(u16, Vec<u8>)> {
    let (version, packed) = peek(kind, &file).ok_or_else(|| bad("not a file of this kind"))?;
    let len = u64::from_le_bytes(file[12..20].try_into().unwrap()) as usize;
    if !packed {
        if file.len() - HEADER_LEN != len {
            return Err(bad("the body is cut short"));
        }
        let mut body = file;
        body.drain(..HEADER_LEN);
        return Ok((version, body));
    }
    let shuffled =
        zstd::bulk::decompress(&file[HEADER_LEN..], len).map_err(|e| bad(&e.to_string()))?;
    if shuffled.len() != len {
        return Err(bad("the body has the wrong length"));
    }
    Ok((version, unshuffle4(&shuffled)))
}

/// Appends the fields of a record.
#[derive(Default)]
pub struct Writer(pub Vec<u8>);

impl Writer {
    pub fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    /// A length, then the bytes.
    pub fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.0.extend_from_slice(v);
    }
    pub fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }
    /// A count, then the values.
    pub fn u32s(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.u32(*x);
        }
    }
    pub fn u64s(&mut self, v: &[u64]) {
        self.u64(v.len() as u64);
        for x in v {
            self.u64(*x);
        }
    }
}

/// Reads the fields of a record in the order they were written; every read is `None` once
/// the record is exhausted or a count points past its end.
pub struct Reader<'a>(pub &'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, tail) = (self.0.get(..n)?, self.0.get(n..)?);
        self.0 = tail;
        Some(head)
    }
    pub fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }
    pub fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Option<u64> {
        self.take(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }
    pub fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = usize::try_from(self.u64()?).ok()?;
        self.take(n)
    }
    pub fn str(&mut self) -> Option<String> {
        String::from_utf8(self.bytes()?.to_vec()).ok()
    }
    pub fn u32s(&mut self) -> Option<Vec<u32>> {
        let n = usize::try_from(self.u64()?).ok()?;
        let raw = self.take(n.checked_mul(4)?)?;
        Some(
            raw.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }
    pub fn u64s(&mut self) -> Option<Vec<u64>> {
        let n = usize::try_from(self.u64()?).ok()?;
        let raw = self.take(n.checked_mul(8)?)?;
        Some(
            raw.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }
    /// What is left, unread.
    pub fn rest(&self) -> &'a [u8] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_body_reads_back_whole_stored_or_compressed() {
        let few: Vec<u8> = (0..8192u32)
            .flat_map(|i| ((i % 7) as f32 * 0.5).to_le_bytes())
            .collect();
        let sealed = seal(STATE_BLOCK, 3, &few, true);
        assert!(sealed.len() < few.len() / 4, "few distinct floats compress");
        assert_eq!(peek(STATE_BLOCK, &sealed), Some((3, true)));
        assert_eq!(open(STATE_BLOCK, sealed).unwrap(), (3, few.clone()));
        let plain = seal(STATE_BLOCK, 3, &few, false);
        assert_eq!(plain.len(), few.len() + HEADER_LEN);
        assert_eq!(open(STATE_BLOCK, plain).unwrap().1, few);
        let tiny = seal(STATE_INDEX, 1, b"abc", true);
        assert_eq!(peek(STATE_INDEX, &tiny), Some((1, false)));
    }

    #[test]
    fn a_file_of_another_kind_or_cut_short_is_refused() {
        let sealed = seal(STATE_INDEX, 1, b"0123456789", false);
        assert!(open(STATE_BLOCK, sealed.clone()).is_err());
        let mut cut = sealed;
        cut.pop();
        assert!(open(STATE_INDEX, cut).is_err());
        assert!(open(STATE_INDEX, b"{\"uses\":{}}".to_vec()).is_err());
    }

    #[test]
    fn fields_read_back_in_the_order_they_were_written() {
        let mut w = Writer::default();
        w.u8(7);
        w.u32(9);
        w.str("model");
        w.u32s(&[1, 2, 3]);
        w.u64s(&[u64::MAX, 5]);
        w.bytes(b"rest");
        let mut r = Reader(&w.0);
        assert_eq!(r.u8(), Some(7));
        assert_eq!(r.u32(), Some(9));
        assert_eq!(r.str().as_deref(), Some("model"));
        assert_eq!(r.u32s(), Some(vec![1, 2, 3]));
        assert_eq!(r.u64s(), Some(vec![u64::MAX, 5]));
        assert_eq!(r.bytes(), Some(b"rest".as_slice()));
        assert_eq!(r.u8(), None);
        assert_eq!(Reader(&[255; 8]).u32s(), None, "a count past the end");
    }
}
