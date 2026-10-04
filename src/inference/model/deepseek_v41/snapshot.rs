//! A decode state as bytes: what a sequence has built up to a position, in a form that comes back
//! bit for bit.
//!
//! The state is every layer's window of rope'd KV rows, compressed KV, index keys, the raw rows of
//! the filling compression group, the engram's recent compressed ids and the position - all f32
//! or integers, already quantised at their own positions - so a round trip through this format
//! hands the model exactly the state it had, and a decode that continues from it gives the logits
//! the live state would give. The prompt-end logits can ride along, so a prompt that comes back
//! whole needs no pass at all.
//!
//! Content: KV tensors, hidden rows, and the engram's last few ids after the model's own token
//! compression (a class map, not token ids). No token text and no prompt token id is written; a
//! run of prompt tokens cannot be read off the bytes. The state is still derived from the prompt,
//! so a file of it is as private as the prompt.
//!
//! Layout, little endian: magic, format version, the 32-byte identity of the model and numerics
//! the state belongs to, the position, the sections, and a SHA-256 over everything before it.
//! A reader refuses a file whose checksum, magic, version, identity or shape disagrees with what
//! it expects, and never reads past the end of the bytes it was given.

use super::cache::{AttnCache, DecodeState};
use super::engram::EngramState;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

const MAGIC: &[u8; 8] = b"LKDSSNAP";

/// Bumped whenever the byte layout or the meaning of a field changes; a file of another version
/// is refused rather than reinterpreted.
pub const FORMAT_VERSION: u32 = 1;

const DIGEST_LEN: usize = 32;

/// The bytes that name the model, its quantisation and the numerics a state was built under.
pub type Identity = [u8; 32];

/// Why a snapshot was not accepted.
#[derive(Debug, PartialEq, Eq)]
pub enum SnapshotError {
    /// Shorter than the fixed parts, or a section runs past the end.
    Truncated,
    /// The checksum does not match the bytes: damaged in storage or in transit.
    Corrupt,
    /// Not a snapshot of this kind.
    NotASnapshot,
    /// Written by another version of the format.
    Version(u32),
    /// Built for another model, quantisation or numerics.
    Identity,
    /// Well formed but not the shape of this model.
    Shape(&'static str),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated"),
            Self::Corrupt => write!(f, "checksum mismatch"),
            Self::NotASnapshot => write!(f, "not a decode-state snapshot"),
            Self::Version(v) => write!(f, "format version {v}, this build reads {FORMAT_VERSION}"),
            Self::Identity => write!(f, "built for another model or numerics"),
            Self::Shape(what) => write!(f, "shape mismatch: {what}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

/// A decoded snapshot: the state at its position, and the logits of the position before it when
/// the snapshot was taken straight after a prefill.
pub struct Snapshot {
    pub state: DecodeState,
    pub logits: Option<Vec<f32>>,
}

/// The identity of a model's numerics from the parts that define them, in order. A part that
/// changes - a file digest, a numerics setting, the operator's salt - gives another identity.
pub fn identity(parts: &[&[u8]]) -> Identity {
    let mut h = Sha256::new();
    h.update(b"loken decode-state identity");
    h.update(FORMAT_VERSION.to_le_bytes());
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p);
    }
    h.finalize().into()
}

struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32s(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        self.0.reserve(v.len() * 4);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn u32s(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.u32(*x);
        }
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], SnapshotError> {
        if self.0.len() < n {
            return Err(SnapshotError::Truncated);
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, SnapshotError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// A length that must fit in what is left at `width` bytes per element, checked before
    /// anything is allocated for it.
    fn len(&mut self, width: usize) -> Result<usize, SnapshotError> {
        let n = usize::try_from(self.u64()?).map_err(|_| SnapshotError::Truncated)?;
        if n.checked_mul(width).is_none_or(|b| b > self.0.len()) {
            return Err(SnapshotError::Truncated);
        }
        Ok(n)
    }
    fn f32s(&mut self) -> Result<Vec<f32>, SnapshotError> {
        let n = self.len(4)?;
        Ok(self
            .take(n * 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn u32s(&mut self) -> Result<Vec<u32>, SnapshotError> {
        let n = self.len(4)?;
        Ok(self
            .take(n * 4)?
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}

/// `state` and optionally the logits of its last position as bytes under `identity`. The DSpark
/// capture (`capture_layers`, `captured`) is per-decode scratch and is not carried.
pub fn encode(state: &DecodeState, logits: Option<&[f32]>, identity: &Identity) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(MAGIC);
    w.u32(FORMAT_VERSION);
    w.0.extend_from_slice(identity);
    w.u64(state.pos as u64);
    w.u32(state.layers.len() as u32);
    for layer in &state.layers {
        w.u64(layer.n_groups as u64);
        w.u64(layer.win.len() as u64);
        for row in &layer.win {
            w.f32s(row);
        }
        w.f32s(&layer.comp_kv);
        w.f32s(&layer.index_k);
        w.u64(layer.group_x.len() as u64);
        for row in &layer.group_x {
            w.f32s(row);
        }
    }
    w.u32s(&state.engram.recent());
    match logits {
        Some(l) => {
            w.u32(1);
            w.f32s(l);
        }
        None => w.u32(0),
    }
    let sum: [u8; DIGEST_LEN] = Sha256::digest(&w.0).into();
    w.0.extend_from_slice(&sum);
    w.0
}

/// The snapshot in `bytes`, if it is one of this format and version, written under `identity`
/// for a model of `n_layers` layers, and whole.
pub fn decode(
    bytes: &[u8],
    identity: &Identity,
    n_layers: usize,
) -> Result<Snapshot, SnapshotError> {
    if bytes.len() < MAGIC.len() + 4 + DIGEST_LEN {
        return Err(SnapshotError::Truncated);
    }
    // Magic and version first: a file of another kind or version has no checksum to trust.
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(SnapshotError::NotASnapshot);
    }
    let version = u32::from_le_bytes(bytes[MAGIC.len()..MAGIC.len() + 4].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(SnapshotError::Version(version));
    }
    let (body, sum) = bytes.split_at(bytes.len() - DIGEST_LEN);
    if Sha256::digest(body).as_slice() != sum {
        return Err(SnapshotError::Corrupt);
    }
    let mut r = Reader(&body[MAGIC.len() + 4..]);
    if r.take(DIGEST_LEN)? != identity {
        return Err(SnapshotError::Identity);
    }
    let pos = usize::try_from(r.u64()?).map_err(|_| SnapshotError::Truncated)?;
    if r.u32()? as usize != n_layers {
        return Err(SnapshotError::Shape("layer count"));
    }
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let n_groups = usize::try_from(r.u64()?).map_err(|_| SnapshotError::Truncated)?;
        // Each window row carries at least its own length word.
        let n_win = r.len(8)?;
        let mut win = VecDeque::with_capacity(n_win);
        for _ in 0..n_win {
            win.push_back(r.f32s()?);
        }
        let comp_kv = r.f32s()?;
        let index_k = r.f32s()?;
        let n_x = r.len(8)?;
        let mut group_x = Vec::with_capacity(n_x);
        for _ in 0..n_x {
            group_x.push(r.f32s()?);
        }
        layers.push(AttnCache {
            win,
            comp_kv,
            index_k,
            group_x,
            n_groups,
        });
    }
    let engram = EngramState::from_recent(r.u32s()?);
    let logits = match r.u32()? {
        0 => None,
        1 => Some(r.f32s()?),
        _ => return Err(SnapshotError::Shape("logits flag")),
    };
    if !r.0.is_empty() {
        return Err(SnapshotError::Shape("trailing bytes"));
    }
    let mut state = DecodeState::new(n_layers);
    state.layers = layers;
    state.engram = engram;
    state.pos = pos;
    Ok(Snapshot { state, logits })
}

/// Whether two states hold the same bits: the position, every layer's rows and counts, and the
/// engram's ids. A test's definition of a faithful round trip.
pub fn same_state(a: &DecodeState, b: &DecodeState) -> bool {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    a.pos == b.pos
        && a.engram.recent() == b.engram.recent()
        && a.layers.len() == b.layers.len()
        && a.layers.iter().zip(&b.layers).all(|(x, y)| {
            x.n_groups == y.n_groups
                && x.win.len() == y.win.len()
                && x.win.iter().zip(&y.win).all(|(p, q)| bits(p) == bits(q))
                && bits(&x.comp_kv) == bits(&y.comp_kv)
                && bits(&x.index_k) == bits(&y.index_k)
                && x.group_x.len() == y.group_x.len()
                && x.group_x
                    .iter()
                    .zip(&y.group_x)
                    .all(|(p, q)| bits(p) == bits(q))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DecodeState {
        let mut s = DecodeState::new(2);
        s.pos = 7;
        s.layers[0].win = (0..3).map(|i| vec![i as f32 + 0.5, -1.25]).collect();
        s.layers[1].comp_kv = vec![f32::MIN_POSITIVE, 3.0, -0.0];
        s.layers[1].index_k = vec![1.0];
        s.layers[1].n_groups = 1;
        s.layers[1].group_x = vec![vec![9.0, 8.0], vec![7.0, f32::NAN]];
        s.engram = EngramState::from_recent([4, 5, 6]);
        s
    }

    fn id() -> Identity {
        identity(&[b"model", b"numerics"])
    }

    #[test]
    fn a_round_trip_keeps_every_bit_and_the_logits() {
        let s = sample();
        let bytes = encode(&s, Some(&[0.25, -2.0]), &id());
        let back = decode(&bytes, &id(), 2).unwrap();
        assert!(same_state(&s, &back.state));
        assert_eq!(back.logits.as_deref(), Some(&[0.25, -2.0][..]));
        let none = decode(&encode(&s, None, &id()), &id(), 2).unwrap();
        assert!(none.logits.is_none());
    }

    #[test]
    fn a_damaged_file_is_refused_not_read() {
        let bytes = encode(&sample(), None, &id());
        // Truncated at every length, flipped at every byte: an error, never a panic.
        for n in 0..bytes.len() {
            assert!(decode(&bytes[..n], &id(), 2).is_err(), "truncated to {n}");
        }
        for i in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[i] ^= 0x40;
            assert!(decode(&bad, &id(), 2).is_err(), "byte {i} flipped");
        }
    }

    #[test]
    fn another_version_model_or_shape_is_refused() {
        let bytes = encode(&sample(), None, &id());
        assert_eq!(
            decode(&bytes, &identity(&[b"other"]), 2).err(),
            Some(SnapshotError::Identity)
        );
        assert_eq!(
            decode(&bytes, &id(), 3).err(),
            Some(SnapshotError::Shape("layer count"))
        );
        // A future version, checksum and all: refused for its version, not read.
        let mut v2 = bytes.clone();
        v2[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        let n = v2.len() - DIGEST_LEN;
        let sum: [u8; DIGEST_LEN] = Sha256::digest(&v2[..n]).into();
        v2[n..].copy_from_slice(&sum);
        assert_eq!(
            decode(&v2, &id(), 2).err(),
            Some(SnapshotError::Version(FORMAT_VERSION + 1))
        );
        assert_eq!(
            decode(
                b"not a snapshot at all, but long enough to have a tail....",
                &id(),
                2
            )
            .err(),
            Some(SnapshotError::NotASnapshot)
        );
    }

    #[test]
    fn a_length_word_past_the_end_does_not_allocate() {
        // A checksummed body whose first section claims more rows than it holds.
        let mut w = Writer(Vec::new());
        w.0.extend_from_slice(MAGIC);
        w.u32(FORMAT_VERSION);
        w.0.extend_from_slice(&id());
        w.u64(1);
        w.u32(1);
        w.u64(0);
        w.u64(u64::MAX / 16);
        let sum: [u8; DIGEST_LEN] = Sha256::digest(&w.0).into();
        w.0.extend_from_slice(&sum);
        assert_eq!(decode(&w.0, &id(), 1).err(), Some(SnapshotError::Truncated));
    }
}
