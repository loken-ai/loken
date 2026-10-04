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
use super::packed::{fp4_len, fp8_len, pack_fp4, pack_fp8, unpack_fp4, unpack_fp8};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

const MAGIC: &[u8; 8] = b"LKDSSNAP";
const BLOCK_MAGIC: &[u8; 8] = b"LKDSBLK1";

/// Bumped whenever the byte layout or the meaning of a field changes; a file of another version
/// is refused rather than reinterpreted.
pub const FORMAT_VERSION: u32 = 3;

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
    /// `v` in the packed form of `kind` when it is on that grid bit for bit, as f32 otherwise.
    fn rows(&mut self, kind: Kind, v: &[f32]) {
        match kind.pack(v) {
            Some(bytes) => {
                self.0.push(1);
                self.u64(v.len() as u64);
                self.u64(bytes.len() as u64);
                self.0.extend_from_slice(&bytes);
            }
            None => {
                self.0.push(0);
                self.f32s(v);
            }
        }
    }
}

/// The quantised forms the cache rows come in.
#[derive(Clone, Copy)]
enum Kind {
    /// e4m3, power-of-two scale per block of 32: the window rows.
    Fp8,
    /// e2m1, f32 scale per block of 16: the compressed KV.
    Fp4Block16,
    /// e2m1, power-of-two scale per block of 32: the index keys.
    Fp4Pow2Block32,
}

impl Kind {
    fn pack(self, v: &[f32]) -> Option<Vec<u8>> {
        match self {
            Self::Fp8 => pack_fp8(v),
            Self::Fp4Block16 => pack_fp4(v, 16, false),
            Self::Fp4Pow2Block32 => pack_fp4(v, 32, true),
        }
    }
    /// Packed bytes for `n` values, `None` when `n` is not a whole number of blocks.
    fn packed_len(self, n: usize) -> Option<usize> {
        let (block, len) = match self {
            Self::Fp8 => (32, fp8_len(n)),
            Self::Fp4Block16 => (16, fp4_len(n, 16, false)),
            Self::Fp4Pow2Block32 => (32, fp4_len(n, 32, true)),
        };
        n.is_multiple_of(block).then_some(len)
    }
    fn unpack(self, bytes: &[u8], n: usize) -> Vec<f32> {
        match self {
            Self::Fp8 => unpack_fp8(bytes, n),
            Self::Fp4Block16 => unpack_fp4(bytes, n, 16, false),
            Self::Fp4Pow2Block32 => unpack_fp4(bytes, n, 32, true),
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
    fn rows(&mut self, kind: Kind) -> Result<Vec<f32>, SnapshotError> {
        match self.take(1)?[0] {
            0 => self.f32s(),
            1 => {
                let n = usize::try_from(self.u64()?).map_err(|_| SnapshotError::Truncated)?;
                let len = self.len(1)?;
                if kind.packed_len(n) != Some(len) {
                    return Err(SnapshotError::Shape("packed rows"));
                }
                Ok(kind.unpack(self.take(len)?, n))
            }
            _ => Err(SnapshotError::Shape("rows mode")),
        }
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

/// How a state's compressed rows divide into blocks of tokens: each layer's compression ratio and
/// the block size. A block of `block_tokens` tokens adds `block_tokens / ratio` groups to a layer.
pub struct BlockLayout<'a> {
    pub ratios: &'a [usize],
    pub block_tokens: usize,
}

/// Per layer: groups a block holds, floats of compressed KV and of index keys per group. `None`
/// for a state the layout cannot cut at block boundaries (a layer with rows but no ratio, a ratio
/// that does not divide the block, rows that are not a whole number per group).
fn block_shape(state: &DecodeState, layout: &BlockLayout) -> Option<Vec<(usize, usize, usize)>> {
    if layout.ratios.len() != state.layers.len() || layout.block_tokens == 0 {
        return None;
    }
    state
        .layers
        .iter()
        .zip(layout.ratios)
        .map(|(l, &ratio)| {
            let has_rows = !l.comp_kv.is_empty() || !l.index_k.is_empty();
            if !has_rows {
                return Some((0, 0, 0));
            }
            if ratio == 0 || !layout.block_tokens.is_multiple_of(ratio) || l.n_groups == 0 {
                return None;
            }
            let (c, i) = (l.comp_kv.len(), l.index_k.len());
            (c % l.n_groups == 0 && i % l.n_groups == 0)
                .then(|| (layout.block_tokens / ratio, c / l.n_groups, i / l.n_groups))
        })
        .collect()
}

/// Whole blocks of tokens a state can hand to block files: the position's, when every layer has
/// the groups for them, else none.
pub fn block_count(state: &DecodeState, layout: &BlockLayout) -> usize {
    let Some(shape) = block_shape(state, layout) else {
        return 0;
    };
    let nb = state.pos / layout.block_tokens;
    let fits = state
        .layers
        .iter()
        .zip(&shape)
        .all(|(l, &(g, _, _))| nb * g <= l.n_groups);
    if fits {
        nb
    } else {
        0
    }
}

/// The compressed rows block `b` of `state` adds, every layer's, under `identity`'s version.
pub fn encode_block(state: &DecodeState, b: usize, layout: &BlockLayout) -> Vec<u8> {
    let shape = block_shape(state, layout).unwrap_or_default();
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(BLOCK_MAGIC);
    w.u32(FORMAT_VERSION);
    w.u32(state.layers.len() as u32);
    for (l, &(g, cl, il)) in state.layers.iter().zip(&shape) {
        w.rows(Kind::Fp4Block16, &l.comp_kv[b * g * cl..(b + 1) * g * cl]);
        w.rows(
            Kind::Fp4Pow2Block32,
            &l.index_k[b * g * il..(b + 1) * g * il],
        );
    }
    let sum: [u8; DIGEST_LEN] = Sha256::digest(&w.0).into();
    w.0.extend_from_slice(&sum);
    w.0
}

/// `state` and optionally the logits of its last position as one self-contained snapshot under
/// `identity`. The DSpark capture (`capture_layers`, `captured`) is per-decode scratch and is not
/// carried.
pub fn encode(state: &DecodeState, logits: Option<&[f32]>, identity: &Identity) -> Vec<u8> {
    encode_manifest(state, logits, identity, None).0
}

/// The manifest of `state`: everything but the compressed rows the first `blocks` blocks of
/// `layout` carry, and that count of blocks (0 without a layout or when the state cannot be cut).
pub fn encode_manifest(
    state: &DecodeState,
    logits: Option<&[f32]>,
    identity: &Identity,
    layout: Option<&BlockLayout>,
) -> (Vec<u8>, usize) {
    let nblocks = layout.map_or(0, |l| block_count(state, l));
    let shape = layout
        .and_then(|l| block_shape(state, l))
        .unwrap_or_else(|| vec![(0, 0, 0); state.layers.len()]);
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(MAGIC);
    w.u32(FORMAT_VERSION);
    w.0.extend_from_slice(identity);
    w.u64(state.pos as u64);
    w.u64(nblocks as u64);
    w.u32(state.layers.len() as u32);
    for (layer, &(g, cl, il)) in state.layers.iter().zip(&shape) {
        w.u64(layer.n_groups as u64);
        let row_len = layer.win.front().map_or(0, Vec::len);
        w.u64(layer.win.len() as u64);
        w.u64(row_len as u64);
        let flat: Vec<f32> = layer.win.iter().flatten().copied().collect();
        if flat.len() == layer.win.len() * row_len {
            w.0.push(1);
            w.rows(Kind::Fp8, &flat);
        } else {
            w.0.push(0);
            for row in &layer.win {
                w.f32s(row);
            }
        }
        w.rows(
            Kind::Fp4Block16,
            &layer.comp_kv[(nblocks * g * cl).min(layer.comp_kv.len())..],
        );
        w.rows(
            Kind::Fp4Pow2Block32,
            &layer.index_k[(nblocks * g * il).min(layer.index_k.len())..],
        );
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
    (w.0, nblocks)
}

/// The snapshot in `bytes`, if it is one of this format and version, written under `identity`
/// for a model of `n_layers` layers, whole and with no blocks.
pub fn decode(
    bytes: &[u8],
    identity: &Identity,
    n_layers: usize,
) -> Result<Snapshot, SnapshotError> {
    decode_split(bytes, &[], identity, n_layers)
}

/// One block's compressed rows per layer, as `(compressed KV, index keys)`.
fn decode_block(bytes: &[u8], n_layers: usize) -> Result<Vec<(Vec<f32>, Vec<f32>)>, SnapshotError> {
    if bytes.len() < BLOCK_MAGIC.len() + 4 + DIGEST_LEN {
        return Err(SnapshotError::Truncated);
    }
    if &bytes[..BLOCK_MAGIC.len()] != BLOCK_MAGIC {
        return Err(SnapshotError::NotASnapshot);
    }
    let version = u32::from_le_bytes(
        bytes[BLOCK_MAGIC.len()..BLOCK_MAGIC.len() + 4]
            .try_into()
            .unwrap(),
    );
    if version != FORMAT_VERSION {
        return Err(SnapshotError::Version(version));
    }
    let (body, sum) = bytes.split_at(bytes.len() - DIGEST_LEN);
    if Sha256::digest(body).as_slice() != sum {
        return Err(SnapshotError::Corrupt);
    }
    let mut r = Reader(&body[BLOCK_MAGIC.len() + 4..]);
    if r.u32()? as usize != n_layers {
        return Err(SnapshotError::Shape("layer count"));
    }
    let mut out = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        out.push((r.rows(Kind::Fp4Block16)?, r.rows(Kind::Fp4Pow2Block32)?));
    }
    if !r.0.is_empty() {
        return Err(SnapshotError::Shape("trailing bytes"));
    }
    Ok(out)
}

/// The snapshot whose manifest is `bytes` and whose first compressed rows are in `blocks`, in
/// order: the same checks as `decode`, and the block count must be the manifest's.
pub fn decode_split(
    bytes: &[u8],
    blocks: &[Vec<u8>],
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
    if usize::try_from(r.u64()?).ok() != Some(blocks.len()) {
        return Err(SnapshotError::Shape("block count"));
    }
    if r.u32()? as usize != n_layers {
        return Err(SnapshotError::Shape("layer count"));
    }
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let n_groups = usize::try_from(r.u64()?).map_err(|_| SnapshotError::Truncated)?;
        let n_win = usize::try_from(r.u64()?).map_err(|_| SnapshotError::Truncated)?;
        let row_len = usize::try_from(r.u64()?).map_err(|_| SnapshotError::Truncated)?;
        let mut win = VecDeque::new();
        match r.take(1)?[0] {
            1 => {
                let flat = r.rows(Kind::Fp8)?;
                if n_win.checked_mul(row_len) != Some(flat.len()) {
                    return Err(SnapshotError::Shape("window rows"));
                }
                if row_len > 0 {
                    win.extend(flat.chunks_exact(row_len).map(<[f32]>::to_vec));
                }
            }
            0 => {
                // Each window row carries at least its own length word.
                if n_win.checked_mul(8).is_none_or(|b| b > r.0.len()) {
                    return Err(SnapshotError::Truncated);
                }
                for _ in 0..n_win {
                    win.push_back(r.f32s()?);
                }
            }
            _ => return Err(SnapshotError::Shape("window mode")),
        }
        let comp_kv = r.rows(Kind::Fp4Block16)?;
        let index_k = r.rows(Kind::Fp4Pow2Block32)?;
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
    // The blocks' rows come first, then the manifest's tail.
    for block in blocks.iter().rev() {
        for (layer, (comp, index)) in layers.iter_mut().zip(decode_block(block, n_layers)?) {
            layer.comp_kv.splice(..0, comp);
            layer.index_k.splice(..0, index);
        }
    }
    for l in &layers {
        let rows_ok = |len: usize| {
            l.n_groups == 0 && len == 0 || l.n_groups > 0 && len.is_multiple_of(l.n_groups)
        };
        if !rows_ok(l.comp_kv.len()) || !rows_ok(l.index_k.len()) {
            return Err(SnapshotError::Shape("rows per group"));
        }
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

    /// A state whose rows are what the quantisers produce is written packed: smaller by a factor
    /// of several, and back to the bit.
    #[test]
    fn quantised_rows_are_stored_packed_and_come_back_exact() {
        use crate::inference::model::deepseek_v41::attention::act_quant_fp8_e4m3;
        use crate::inference::model::deepseek_v41::band::fp4_act_quant;
        use crate::tensor::{Device, Tensor};
        let raw: Vec<f32> = (0..64 * 40)
            .map(|i| ((i * 2654435761u64 as usize) % 1000) as f32 / 97.0 - 5.0)
            .collect();
        let t = |rows: usize| {
            Tensor::from_vec(raw[..rows * 64].to_vec(), (rows, 64), &Device::Cpu).unwrap()
        };
        let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut s = DecodeState::new(1);
        s.pos = 30;
        let win = flat(act_quant_fp8_e4m3(&t(30), 32).unwrap());
        s.layers[0].win = win.chunks(64).map(<[f32]>::to_vec).collect();
        s.layers[0].comp_kv = flat(fp4_act_quant(&t(20), 16, false).unwrap());
        s.layers[0].index_k = flat(fp4_act_quant(&t(20), 32, true).unwrap());
        s.layers[0].n_groups = 20;
        let bytes = encode(&s, None, &id());
        let floats = (win.len() + 2 * 20 * 64) * 4;
        assert!(bytes.len() * 3 < floats, "{} against {floats}", bytes.len());
        assert!(same_state(&s, &decode(&bytes, &id(), 1).unwrap().state));
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
        w.u64(0);
        w.u32(1);
        w.u64(0);
        w.u64(u64::MAX / 16);
        let sum: [u8; DIGEST_LEN] = Sha256::digest(&w.0).into();
        w.0.extend_from_slice(&sum);
        assert_eq!(decode(&w.0, &id(), 1).err(), Some(SnapshotError::Truncated));
    }
}
