//! A disk cache of a recurrent hybrid's decode state, found by the longest prefix of a prompt
//! it covers.
//!
//! A hybrid carries two kinds of state: the recurrent mixers' (a convolution window, a state
//! matrix) and the attention layers' KV. The recurrent part cannot be cut back to a shorter
//! prefix, so a state is stored at the positions a prefill passes, and a prompt restores the
//! longest stored state it extends and prefills the rest.
//!
//! One stored state is a manifest and blocks, both held by [`StateStore`].
//! - The manifest is everything that belongs to one exact position: every recurrent mixer's
//!   state, every windowed layer's resident rows, and the rows of each global layer past the last
//!   whole block.
//! - A block holds the rows of one block of tokens for every global layer, in the cache's own
//!   element type (rows that an f16 holds exactly are stored as f16, the others as f32, so a
//!   restore returns every value bit for bit). It is named by the chain hash of the tokens up to
//!   its end, so states of one conversation, and of conversations sharing a prefix, store it once.
//!
//! - A model with a draft head also keeps the head's state in the manifest: its layer's rows
//!   for the pairs it has read, its indexer, and the prompt row whose pair waits for the next
//!   token, so a restored head drafts as if it had read the prefix.
//!
//! The layer kinds are read from the layers the model really has: a state is refused when the
//! kinds, the row geometry or the placement it was written under are not the model's. Every file
//! carries a version, the identity it was written under and a checksum; a file that fails any of
//! them is refused and dropped.

use super::mixer::{ConvState, Mixer, MixerState, SsmState};
use super::*;
use crate::inference::cache::state_disk::{Entry, StateStore};
use crate::inference::cache::state_tune;
use anyhow::{anyhow, bail, Result as AnyResult};
use sha2::{Digest, Sha256};

const MANIFEST_MAGIC: &[u8; 8] = b"LKHYBM01";
const BLOCK_MAGIC: &[u8; 8] = b"LKHYBB01";
const HANDOVER_MAGIC: &[u8; 8] = b"LKHYBH01";
/// Bumped when the layout of a manifest or a block changes.
pub const FORMAT_VERSION: u32 = 2;
const SUM_BYTES: usize = 16;

/// What a layer contributes to a stored state, fixed by the layer and not by the position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StateKind {
    /// No state: a block that mixes nothing.
    Nothing,
    /// Attention over every position: its rows are in the blocks.
    Global,
    /// Attention over a sliding window: the resident rows are in the manifest.
    Window,
    /// A recurrent mixer: its state is in the manifest.
    Mixer,
}

impl StateKind {
    fn tag(self) -> u8 {
        match self {
            StateKind::Nothing => 0,
            StateKind::Global => 1,
            StateKind::Window => 2,
            StateKind::Mixer => 3,
        }
    }
}

/// Rows of a layer's KV, token-major `[n, n_kv, head_dim]`.
#[derive(Clone)]
struct Rows {
    dtype: u8,
    n_kv: usize,
    head_dim: usize,
    n: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

/// A recurrent state tensor copied to the host.
#[derive(Clone)]
struct Tens {
    dims: Vec<usize>,
    dtype: u8,
    /// Held on a card (a tensor) rather than on the host (a flat vector).
    card: bool,
    data: Vec<f32>,
}

#[derive(Clone)]
enum MixerHost {
    Conv(Option<Tens>),
    Ssm(Option<(Tens, Tens)>),
    Delta(Option<(Tens, Tens)>),
}

#[derive(Clone)]
enum Held {
    Nothing,
    Global(Rows),
    Window(Rows),
    Mixer(MixerHost),
}

/// A model's state after `pos` tokens, copied to the host: all of it but the blocks.
#[derive(Clone)]
pub(crate) struct Point {
    pub pos: usize,
    layers: Vec<Held>,
    /// The draft head's state at `pos`; `None` for a model without one, or when the head did
    /// not hold the pairs of the `pos` tokens.
    draft: Option<DraftHeld>,
}

/// A draft head's state after the target read `pos` tokens: it has read `committed` pairs,
/// `pos - 1`, and the last prompt row waits for the token after it.
#[derive(Clone)]
pub(crate) struct DraftHeld {
    committed: usize,
    /// The head layer's rows for positions `0..committed`.
    rows: Rows,
    /// The head layer's indexer with `committed` pairs read: the raw rows, the pooled keys,
    /// their lanes and the tokens held.
    indexer: Option<(Vec<f32>, Vec<f32>, Vec<f32>, usize)>,
    /// The last prompt row, `[1, 1, streams * dim]`.
    carried: Option<Tens>,
}

fn dtype_code(d: DType) -> Result<u8> {
    match d {
        DType::F16 => Ok(0),
        DType::F32 => Ok(1),
        DType::BF16 => Ok(2),
        other => Err(crate::tensor::Error::msg(format!(
            "state cache: no code for a {other:?} state"
        ))),
    }
}

fn dtype_of(code: u8) -> AnyResult<DType> {
    match code {
        0 => Ok(DType::F16),
        1 => Ok(DType::F32),
        2 => Ok(DType::BF16),
        c => bail!("state cache: unknown element type {c}"),
    }
}

impl Tens {
    fn of(t: &Tensor) -> Result<Self> {
        Ok(Self {
            dims: t.dims().to_vec(),
            dtype: dtype_code(t.dtype())?,
            card: true,
            data: t
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        })
    }

    fn host(data: &[f32]) -> Self {
        Self {
            dims: vec![data.len()],
            dtype: 1,
            card: false,
            data: data.to_vec(),
        }
    }

    fn tensor(&self, device: &Device) -> Result<Tensor> {
        let dt = dtype_of(self.dtype).map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        Tensor::from_vec(self.data.clone(), self.dims.as_slice(), &Device::Cpu)?
            .to_dtype(dt)?
            .to_device(device)
    }
}

impl MixerHost {
    fn of(state: &MixerState) -> Result<Self> {
        let conv = |c: &ConvState| -> Result<Tens> {
            match c {
                ConvState::Card(t) => Tens::of(t),
                ConvState::Host(v) => Ok(Tens::host(v)),
            }
        };
        let ssm = |s: &SsmState| -> Result<(Tens, Tens)> {
            match s {
                SsmState::Card { conv, h } => Ok((Tens::of(conv)?, Tens::of(h)?)),
                SsmState::Host { conv, h } => Ok((Tens::host(conv), Tens::host(h))),
            }
        };
        Ok(match state {
            MixerState::ShortConv(s) => MixerHost::Conv(s.as_ref().map(conv).transpose()?),
            MixerState::Ssm(s) => MixerHost::Ssm(s.as_ref().map(ssm).transpose()?),
            MixerState::DeltaNet(s) => MixerHost::Delta(s.as_ref().map(ssm).transpose()?),
            MixerState::Wide(_) => {
                return Err(crate::tensor::Error::msg(
                    "a wide-residual layer's state has no stored form",
                ))
            }
        })
    }

    fn state(&self, device: &Device) -> Result<MixerState> {
        let conv = |t: &Tens| -> Result<ConvState> {
            Ok(if t.card {
                ConvState::Card(t.tensor(device)?)
            } else {
                ConvState::Host(t.data.clone())
            })
        };
        let ssm = |(c, h): &(Tens, Tens)| -> Result<SsmState> {
            Ok(if c.card {
                SsmState::Card {
                    conv: c.tensor(device)?,
                    h: h.tensor(device)?,
                }
            } else {
                SsmState::Host {
                    conv: c.data.clone(),
                    h: h.data.clone(),
                }
            })
        };
        Ok(match self {
            MixerHost::Conv(s) => MixerState::ShortConv(s.as_ref().map(conv).transpose()?),
            MixerHost::Ssm(s) => MixerState::Ssm(s.as_ref().map(ssm).transpose()?),
            MixerHost::Delta(s) => MixerState::DeltaNet(s.as_ref().map(ssm).transpose()?),
        })
    }

    fn variant(&self) -> u8 {
        match self {
            MixerHost::Conv(_) => 0,
            MixerHost::Ssm(_) => 1,
            MixerHost::Delta(_) => 2,
        }
    }
}

// --- Wire format ---------------------------------------------------------------------------

fn checksum(bytes: &[u8]) -> [u8; SUM_BYTES] {
    Sha256::digest(bytes)[..SUM_BYTES].try_into().unwrap()
}

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, x: u8) {
        self.0.push(x);
    }
    fn u32(&mut self, x: usize) {
        self.0.extend((x as u32).to_le_bytes());
    }
    fn u64(&mut self, x: usize) {
        self.0.extend((x as u64).to_le_bytes());
    }

    /// Values as f16 when each is exactly an f16, else as f32: a read returns the same bits.
    fn packed(&mut self, xs: &[f32]) {
        let half = |x: f32| half::f16::from_f32(x);
        if xs
            .iter()
            .all(|&x| half(x).to_f32().to_bits() == x.to_bits())
        {
            self.u8(1);
            for &x in xs {
                self.0.extend(half(x).to_bits().to_le_bytes());
            }
        } else {
            self.u8(0);
            self.raw(xs);
        }
    }

    fn raw(&mut self, xs: &[f32]) {
        for &x in xs {
            self.0.extend(x.to_bits().to_le_bytes());
        }
    }

    fn rows(&mut self, r: &Rows) {
        self.u8(r.dtype);
        self.u32(r.n_kv);
        self.u32(r.head_dim);
        self.u32(r.n);
        self.packed(&r.k);
        self.packed(&r.v);
    }

    fn tens(&mut self, t: &Tens) {
        self.u32(t.dims.len());
        for &d in &t.dims {
            self.u64(d);
        }
        self.u8(t.dtype);
        self.u8(t.card as u8);
        self.u64(t.data.len());
        self.raw(&t.data);
    }

    fn seal(mut self) -> Vec<u8> {
        let sum = checksum(&self.0);
        self.0.extend(sum);
        self.0
    }
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    /// A reader over the payload of `bytes` once its checksum holds.
    fn open(bytes: &'a [u8]) -> AnyResult<Self> {
        let Some(body) = bytes.len().checked_sub(SUM_BYTES).map(|n| &bytes[..n]) else {
            bail!("state cache: file shorter than its checksum");
        };
        if checksum(body) != bytes[body.len()..] {
            bail!("state cache: checksum mismatch");
        }
        Ok(Self { b: body, i: 0 })
    }

    fn take(&mut self, n: usize) -> AnyResult<&'a [u8]> {
        let end = self
            .i
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| anyhow!("state cache: file ends inside a section"))?;
        let out = &self.b[self.i..end];
        self.i = end;
        Ok(out)
    }

    fn u8(&mut self) -> AnyResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> AnyResult<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn u64(&mut self) -> AnyResult<usize> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()) as usize)
    }

    fn raw(&mut self, n: usize) -> AnyResult<Vec<f32>> {
        let bytes = self.take(
            n.checked_mul(4)
                .ok_or_else(|| anyhow!("state cache: size"))?,
        )?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect())
    }

    fn packed(&mut self, n: usize) -> AnyResult<Vec<f32>> {
        match self.u8()? {
            1 => {
                let bytes = self.take(
                    n.checked_mul(2)
                        .ok_or_else(|| anyhow!("state cache: size"))?,
                )?;
                Ok(bytes
                    .chunks_exact(2)
                    .map(|c| {
                        half::f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32()
                    })
                    .collect())
            }
            0 => self.raw(n),
            t => bail!("state cache: unknown packing {t}"),
        }
    }

    fn rows(&mut self) -> AnyResult<Rows> {
        let dtype = self.u8()?;
        let (n_kv, head_dim, n) = (self.u32()?, self.u32()?, self.u32()?);
        let count = n
            .checked_mul(n_kv)
            .and_then(|x| x.checked_mul(head_dim))
            .ok_or_else(|| anyhow!("state cache: size"))?;
        Ok(Rows {
            dtype,
            n_kv,
            head_dim,
            n,
            k: self.packed(count)?,
            v: self.packed(count)?,
        })
    }

    fn tens(&mut self) -> AnyResult<Tens> {
        let rank = self.u32()?;
        let dims = (0..rank)
            .map(|_| self.u64())
            .collect::<AnyResult<Vec<_>>>()?;
        let dtype = self.u8()?;
        let card = self.u8()? != 0;
        let len = self.u64()?;
        if dims.iter().product::<usize>() != len && !(dims.len() == 1 && dims[0] == len) {
            bail!("state cache: a state tensor's shape and size disagree");
        }
        Ok(Tens {
            dims,
            dtype,
            card,
            data: self.raw(len)?,
        })
    }

    fn done(&self) -> AnyResult<()> {
        if self.i != self.b.len() {
            bail!("state cache: trailing bytes");
        }
        Ok(())
    }
}

/// What a stored state is checked against: the model's layer kinds and row geometry.
pub(crate) struct Layout {
    pub kinds: Vec<StateKind>,
    /// `(n_kv_head, head_dim)` of each layer.
    pub geometry: Vec<(usize, usize)>,
    pub block_tokens: usize,
}

impl Layout {
    fn global_geometry(&self) -> Vec<(usize, usize)> {
        self.kinds
            .iter()
            .zip(&self.geometry)
            .filter(|(k, _)| **k == StateKind::Global)
            .map(|(_, g)| *g)
            .collect()
    }
}

fn encode_manifest(identity: &[u8; 32], layout: &Layout, p: &Point) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    w.0.extend(MANIFEST_MAGIC);
    w.0.extend(FORMAT_VERSION.to_le_bytes());
    w.0.extend(identity);
    w.u64(p.pos);
    w.u64(layout.block_tokens);
    w.u32(p.layers.len());
    for held in &p.layers {
        match held {
            Held::Nothing => w.u8(StateKind::Nothing.tag()),
            Held::Global(r) => {
                w.u8(StateKind::Global.tag());
                w.rows(r);
            }
            Held::Window(r) => {
                w.u8(StateKind::Window.tag());
                w.rows(r);
            }
            Held::Mixer(m) => {
                w.u8(StateKind::Mixer.tag());
                w.u8(m.variant());
                let tens = |w: &mut Writer, t: &Option<Tens>| match t {
                    Some(t) => {
                        w.u8(1);
                        w.tens(t);
                    }
                    None => w.u8(0),
                };
                match m {
                    MixerHost::Conv(t) => tens(&mut w, t),
                    MixerHost::Ssm(s) | MixerHost::Delta(s) => match s {
                        Some((c, h)) => {
                            w.u8(1);
                            w.tens(c);
                            w.tens(h);
                        }
                        None => w.u8(0),
                    },
                }
            }
        }
    }
    match &p.draft {
        Some(d) => {
            w.u8(1);
            write_draft(&mut w, d);
        }
        None => w.u8(0),
    }
    w.seal()
}

fn write_draft(w: &mut Writer, d: &DraftHeld) {
    w.u64(d.committed);
    w.rows(&d.rows);
    match &d.indexer {
        Some((raw, pooled, lanes, tokens)) => {
            w.u8(1);
            w.u64(*tokens);
            for xs in [raw, pooled, lanes] {
                w.u64(xs.len());
                w.raw(xs);
            }
        }
        None => w.u8(0),
    }
    match &d.carried {
        Some(t) => {
            w.u8(1);
            w.tens(t);
        }
        None => w.u8(0),
    }
}

fn read_draft(r: &mut Reader) -> AnyResult<DraftHeld> {
    let committed = r.u64()?;
    let rows = r.rows()?;
    if rows.n != committed {
        bail!(
            "state cache: the draft head holds {} rows for {committed} pairs",
            rows.n
        );
    }
    let indexer = match r.u8()? {
        0 => None,
        _ => {
            let tokens = r.u64()?;
            let mut part = || -> AnyResult<Vec<f32>> {
                let n = r.u64()?;
                r.raw(n)
            };
            Some((part()?, part()?, part()?, tokens))
        }
    };
    let carried = match r.u8()? {
        0 => None,
        _ => Some(r.tens()?),
    };
    Ok(DraftHeld {
        committed,
        rows,
        indexer,
        carried,
    })
}

fn decode_manifest(bytes: &[u8], identity: &[u8; 32], layout: &Layout) -> AnyResult<Point> {
    let mut r = Reader::open(bytes)?;
    if r.take(8)? != MANIFEST_MAGIC {
        bail!("state cache: not a hybrid state manifest");
    }
    let version = u32::from_le_bytes(r.take(4)?.try_into().unwrap());
    if version != FORMAT_VERSION {
        bail!("state cache: format version {version}, this build reads {FORMAT_VERSION}");
    }
    if r.take(32)? != identity {
        bail!("state cache: written under another model or numerics");
    }
    let pos = r.u64()?;
    if r.u64()? != layout.block_tokens {
        bail!("state cache: written under another block size");
    }
    if r.u32()? != layout.kinds.len() {
        bail!("state cache: written for another layer count");
    }
    let mut layers = Vec::with_capacity(layout.kinds.len());
    for (il, kind) in layout.kinds.iter().enumerate() {
        if r.u8()? != kind.tag() {
            bail!("state cache: layer {il} is not the kind it was written as");
        }
        let rows = |r: &mut Reader| -> AnyResult<Rows> {
            let rows = r.rows()?;
            if (rows.n_kv, rows.head_dim) != layout.geometry[il] {
                bail!("state cache: layer {il} has another row geometry");
            }
            Ok(rows)
        };
        layers.push(match kind {
            StateKind::Nothing => Held::Nothing,
            StateKind::Global => Held::Global(rows(&mut r)?),
            StateKind::Window => Held::Window(rows(&mut r)?),
            StateKind::Mixer => {
                let variant = r.u8()?;
                let present = r.u8()? != 0;
                let pair = |r: &mut Reader| -> AnyResult<Option<(Tens, Tens)>> {
                    Ok(if present {
                        Some((r.tens()?, r.tens()?))
                    } else {
                        None
                    })
                };
                Held::Mixer(match variant {
                    0 => MixerHost::Conv(if present { Some(r.tens()?) } else { None }),
                    1 => MixerHost::Ssm(pair(&mut r)?),
                    2 => MixerHost::Delta(pair(&mut r)?),
                    v => bail!("state cache: unknown mixer {v}"),
                })
            }
        });
    }
    let draft = match r.u8()? {
        0 => None,
        _ => Some(read_draft(&mut r)?),
    };
    r.done()?;
    Ok(Point { pos, layers, draft })
}

fn encode_block(rows: &[(Vec<f32>, Vec<f32>)]) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    w.0.extend(BLOCK_MAGIC);
    w.0.extend(FORMAT_VERSION.to_le_bytes());
    w.u32(rows.len());
    for (k, v) in rows {
        w.packed(k);
        w.packed(v);
    }
    w.seal()
}

/// The `(k, v)` rows of every global layer in a block.
fn decode_block(bytes: &[u8], layout: &Layout) -> AnyResult<Vec<(Vec<f32>, Vec<f32>)>> {
    let mut r = Reader::open(bytes)?;
    if r.take(8)? != BLOCK_MAGIC {
        bail!("state cache: not a hybrid state block");
    }
    let version = u32::from_le_bytes(r.take(4)?.try_into().unwrap());
    if version != FORMAT_VERSION {
        bail!("state cache: format version {version}, this build reads {FORMAT_VERSION}");
    }
    let geometry = layout.global_geometry();
    if r.u32()? != geometry.len() {
        bail!("state cache: a block of another layer count");
    }
    let mut out = Vec::with_capacity(geometry.len());
    for (n_kv, hd) in geometry {
        let n = layout.block_tokens * n_kv * hd;
        out.push((r.packed(n)?, r.packed(n)?));
    }
    r.done()?;
    Ok(out)
}

// --- The model's side ----------------------------------------------------------------------

/// Rows resident in a layer's KV, whichever store holds them.
fn resident_len(l: &GenericTransformerLayer) -> usize {
    let spec = l.kv_cache.current_seq_len();
    let host = l.cpu_f16_kv.as_ref().map_or(0, |c| c.len());
    #[cfg(feature = "cuda")]
    let quant = l.quant_kv.as_ref().map_or(0, |q| q.current_seq_len());
    #[cfg(not(feature = "cuda"))]
    let quant = 0;
    spec.max(host).max(quant)
}

/// The element type a layer's resident KV rows are read back in.
fn cache_dtype(l: &GenericTransformerLayer) -> Result<u8> {
    if l.cpu_f16_kv.as_ref().is_some_and(|c| !c.is_empty()) {
        return Ok(0);
    }
    #[cfg(feature = "cuda")]
    if l.quant_kv.as_ref().is_some_and(|q| q.current_seq_len() > 0) {
        return Ok(1);
    }
    match l.kv_cache.k_buffer() {
        Some(k) => dtype_code(k.dtype()),
        None => Ok(0),
    }
}

/// Tokens `[from, to)` of a layer's resident KV as host f32 rows, with the element type the
/// cache holds them in.
fn live_rows(l: &GenericTransformerLayer, from: usize, to: usize) -> Result<Rows> {
    let (n_kv, head_dim) = (l.n_kv_head, l.head_dim);
    if from == to {
        return Ok(Rows {
            dtype: cache_dtype(l)?,
            n_kv,
            head_dim,
            n: 0,
            k: Vec::new(),
            v: Vec::new(),
        });
    }
    let (dtype, (k, v)) = if let Some(c) = l
        .cpu_f16_kv
        .as_ref()
        .filter(|c| c.len() >= to && !c.is_empty())
    {
        (0, c.host_rows(from, to))
    } else {
        #[cfg(feature = "cuda")]
        let quant = match l
            .quant_kv
            .as_ref()
            .filter(|q| q.current_seq_len() >= to && q.current_seq_len() > 0)
        {
            Some(q) => {
                let (k, v) = q
                    .dequantize_kv(DType::F32)
                    .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
                let snap = crate::inference::serve::spec_kv_cache::SpecKvSnapshot::from_kv_tensors(
                    k,
                    v,
                    q.current_seq_len(),
                );
                Some((1, snap.host_rows(from, to)?))
            }
            None => None,
        };
        #[cfg(not(feature = "cuda"))]
        let quant: Option<(u8, (Vec<f32>, Vec<f32>))> = None;
        match quant {
            Some(q) => q,
            None => {
                let (Some(k), Some(v)) = (l.kv_cache.k_buffer(), l.kv_cache.v_buffer()) else {
                    return Err(crate::tensor::Error::msg(
                        "state cache: a layer holds no KV rows".to_string(),
                    ));
                };
                if l.kv_cache.current_seq_len() < to {
                    return Err(crate::tensor::Error::msg(
                        "state cache: a layer holds fewer rows than asked".to_string(),
                    ));
                }
                let dtype = dtype_code(k.dtype())?;
                let snap = crate::inference::serve::spec_kv_cache::SpecKvSnapshot::from_kv_tensors(
                    k,
                    v,
                    l.kv_cache.current_seq_len(),
                );
                (dtype, snap.host_rows(from, to)?)
            }
        }
    };
    Ok(Rows {
        dtype,
        n_kv,
        head_dim,
        n: to - from,
        k,
        v,
    })
}

impl GenericHeteroTransformer {
    /// What each layer contributes to a stored state, and the geometry its rows have.
    pub(crate) fn state_layout(&self, block_tokens: usize) -> Layout {
        let kinds = self
            .layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Attention if l.kv_cache.is_windowed() => StateKind::Window,
                Mixer::Attention => StateKind::Global,
                Mixer::Absent => StateKind::Nothing,
                _ => StateKind::Mixer,
            })
            .collect();
        let geometry = self
            .layers
            .iter()
            .map(|l| (l.n_kv_head, l.head_dim))
            .collect();
        Layout {
            kinds,
            geometry,
            block_tokens,
        }
    }

    /// Opens the disk cache over `store` for this model. `identity` names the model files and the
    /// numerics; the layers' kinds, geometry and placement are added to it. Refused, with `false`,
    /// for a model with no recurrent state (its KV is reused by trimming) or whose KV lives in a
    /// store a state cannot be read from.
    pub(crate) fn attach_state_cache(
        &mut self,
        store: Arc<StateStore>,
        identity: [u8; 32],
    ) -> bool {
        // A wide-residual layer's indexer and n-gram state have no stored form yet.
        if !self.is_recurrent()
            || self.is_wide()
            || self.layers.iter().any(|l| l.cpu_q8_kv.is_some())
        {
            return false;
        }
        // The identity first: the block size is settled under it.
        let mut h = Sha256::new();
        h.update(b"loken hybrid state identity");
        h.update(FORMAT_VERSION.to_le_bytes());
        h.update(identity);
        let probe = self.state_layout(1);
        for (il, l) in self.layers.iter().enumerate() {
            h.update([probe.kinds[il].tag(), l.cos.device().is_cuda() as u8]);
            h.update((probe.geometry[il].0 as u64).to_le_bytes());
            h.update((probe.geometry[il].1 as u64).to_le_bytes());
        }
        h.update([self.draft.is_some() as u8]);
        let identity: [u8; 32] = h.finalize().into();
        // What one token adds to the blocks: both rows of every global layer, at the widest
        // element a cache holds.
        let token_bytes: usize = probe
            .kinds
            .iter()
            .zip(&probe.geometry)
            .filter(|(k, _)| **k == StateKind::Global)
            .map(|(_, (n_kv, hd))| 2 * n_kv * hd * std::mem::size_of::<f32>())
            .sum();
        let Some(block) = state_tune::settle_block_tokens(&store, &identity, 1, token_bytes) else {
            tracing::warn!("state cache off: the directory cannot be written to measure it");
            return false;
        };
        let layout = self.state_layout(block);
        self.hybrid_cache = Some(HybridStateCache {
            store,
            identity,
            layout: Arc::new(layout),
            last_stride: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        true
    }

    pub(crate) fn state_cache(&self) -> Option<HybridStateCache> {
        self.hybrid_cache.clone()
    }

    /// Every recurrent mixer's state, every windowed layer's rows and every global layer's rows
    /// past the last whole block, as the model stands after `pos` tokens.
    pub(crate) fn capture_point(&self, pos: usize, layout: &Layout) -> Result<Point> {
        let g = layout.block_tokens;
        let tail_from = pos / g * g;
        let mut layers = Vec::with_capacity(self.layers.len());
        for (l, kind) in self.layers.iter().zip(&layout.kinds) {
            layers.push(match kind {
                StateKind::Nothing => Held::Nothing,
                StateKind::Mixer => match l.mixer.snapshot()? {
                    Some(s) => Held::Mixer(MixerHost::of(&s)?),
                    None => {
                        return Err(crate::tensor::Error::msg(
                            "state cache: a mixer holds no state".to_string(),
                        ))
                    }
                },
                StateKind::Global => {
                    let held = resident_len(l);
                    if held != pos {
                        return Err(crate::tensor::Error::msg(format!(
                            "state cache: a global layer holds {held} rows at position {pos}"
                        )));
                    }
                    Held::Global(live_rows(l, tail_from, pos)?)
                }
                StateKind::Window => Held::Window(live_rows(l, 0, resident_len(l))?),
            });
        }
        Ok(Point {
            pos,
            layers,
            draft: self.capture_draft(pos)?,
        })
    }

    /// The draft head's state after the target read `pos` tokens; `None` without a head, or
    /// when the head does not hold the pairs of those tokens.
    pub(crate) fn capture_draft(&self, pos: usize) -> Result<Option<DraftHeld>> {
        let Some(d) = self.draft.as_ref() else {
            return Ok(None);
        };
        let held = resident_len(&d.layer);
        let indexer = match (&d.layer.sparse, &d.indexer_mark) {
            (None, _) => None,
            (Some(_), Some((at, c))) if *at == d.committed => Some(c.clone()),
            (Some(s), _) => Some(s.snapshot_cache()?).filter(|c| c.len() == d.committed),
        };
        let agrees = pos > 0
            && d.committed + 1 == pos
            && d.carried.is_some()
            && held >= d.committed
            && (d.layer.sparse.is_none() || indexer.is_some());
        if !agrees {
            tracing::debug!(
                "state cache: no draft head state at token {pos}: the head holds {} pairs, {held} rows",
                d.committed
            );
            return Ok(None);
        }
        Ok(Some(DraftHeld {
            committed: d.committed,
            rows: live_rows(&d.layer, 0, d.committed)?,
            indexer: indexer.map(|c| {
                let (raw, pooled, lanes, tokens) = c.parts();
                (raw.to_vec(), pooled.to_vec(), lanes.to_vec(), tokens)
            }),
            carried: d.carried.as_ref().map(Tens::of).transpose()?,
        }))
    }

    /// Makes `held` the draft head's state: its layer's rows, its indexer and the waiting row.
    /// `None` leaves the head as it is.
    pub(crate) fn restore_draft(&mut self, held: Option<&DraftHeld>) -> Result<()> {
        let (Some(h), true) = (held, self.draft.is_some()) else {
            return Ok(());
        };
        self.draft_reset();
        let d = self.draft.as_mut().expect("checked above");
        let err = |m: &str| crate::tensor::Error::msg(format!("state cache: {m}"));
        if (h.rows.n_kv, h.rows.head_dim) != (d.layer.n_kv_head, d.layer.head_dim) {
            return Err(err("the draft head has another row geometry"));
        }
        if h.indexer.is_some() != d.layer.sparse.is_some() {
            return Err(err(
                "the draft head's indexer is not the one it was written with",
            ));
        }
        let snap = rows_snapshot(&d.layer, &h.rows.k, &h.rows.v, h.rows.dtype)?;
        super::graph::restore_layers(std::slice::from_mut(&mut d.layer), &[snap])?;
        rebuild_host_rows(&mut d.layer);
        if let (Some(s), Some((raw, pooled, lanes, tokens))) = (d.layer.sparse.as_mut(), &h.indexer)
        {
            let cache = super::qsa::IndexerCache::from_parts(
                raw.clone(),
                pooled.clone(),
                lanes.clone(),
                *tokens,
            );
            s.restore_cache(&cache, false);
            d.indexer_mark = Some((h.committed, cache));
        }
        let device = d.layer.cos.device();
        d.carried = h.carried.as_ref().map(|t| t.tensor(&device)).transpose()?;
        d.committed = h.committed;
        d.len = h.committed;
        #[cfg(feature = "cuda")]
        {
            d.step = None;
        }
        Ok(())
    }

    /// Block `b` of the global layers' rows, encoded.
    pub(crate) fn capture_block(&self, b: usize, layout: &Layout) -> Result<Vec<u8>> {
        let g = layout.block_tokens;
        let mut rows = Vec::new();
        for (l, kind) in self.layers.iter().zip(&layout.kinds) {
            if *kind == StateKind::Global {
                let r = live_rows(l, b * g, (b + 1) * g)?;
                rows.push((r.k, r.v));
            }
        }
        Ok(encode_block(&rows))
    }

    /// The per-layer snapshots of `point` with the rows of `blocks` before its tail.
    fn point_snapshots(
        &self,
        point: &Point,
        blocks: &[Vec<(Vec<f32>, Vec<f32>)>],
    ) -> Result<Vec<super::LayerKvSnapshot>> {
        let mut snaps = Vec::with_capacity(self.layers.len());
        let mut global = 0;
        for (l, held) in self.layers.iter().zip(&point.layers) {
            snaps.push(match held {
                Held::Nothing => super::LayerKvSnapshot {
                    spec: None,
                    quant: None,
                    cpu_f16: None,
                    mixer: None,
                    shards: Vec::new(),
                },
                Held::Mixer(m) => super::LayerKvSnapshot {
                    spec: None,
                    quant: None,
                    cpu_f16: None,
                    mixer: Some(m.state(&l.cos.device())?),
                    shards: Vec::new(),
                },
                Held::Window(r) => rows_snapshot(l, &r.k, &r.v, r.dtype)?,
                Held::Global(tail) => {
                    let mut k = Vec::new();
                    let mut v = Vec::new();
                    for block in blocks {
                        k.extend_from_slice(&block[global].0);
                        v.extend_from_slice(&block[global].1);
                    }
                    k.extend_from_slice(&tail.k);
                    v.extend_from_slice(&tail.v);
                    global += 1;
                    rows_snapshot(l, &k, &v, tail.dtype)?
                }
            });
        }
        Ok(snaps)
    }

    /// Makes `point`, with the rows of `blocks` before its tail, the resident state.
    fn restore_point(&mut self, point: &Point, blocks: &[Vec<(Vec<f32>, Vec<f32>)>]) -> Result<()> {
        let snaps = self.point_snapshots(point, blocks)?;
        super::graph::restore_layers(&mut self.layers, &snaps)?;
        for l in &mut self.layers {
            rebuild_host_rows(l);
        }
        self.restore_draft(point.draft.as_ref())?;
        self.invalidate_graph_state();
        Ok(())
    }

    /// Builds again, from the resident rows, the host stores that the layers attending on the host
    /// decode from. A snapshot restored in an instance that did not prefill the rows has none.
    pub(super) fn rebuild_host_stores(&mut self) {
        for l in &mut self.layers {
            if !matches!(l.cos.device(), Device::Cpu) || !matches!(l.mixer, Mixer::Attention) {
                continue;
            }
            let n = l.kv_cache.current_seq_len();
            l.cpu_q8_kv = None;
            l.cpu_f16_kv = None;
            if n > 0 {
                if let Some((k, v)) = l.kv_cache.current_kv() {
                    l.cpu_q8_append(&k, &v, n, 0, false);
                }
            }
        }
    }

    /// Whether this model's decode state is more than rows of KV (a recurrent mixer, a windowed
    /// layer) and can still travel in a state handover: no wide-residual layer, no host q8 cache,
    /// no layer split over cards or cut over cards.
    pub(crate) fn carries_handover_state(&self) -> bool {
        let stateful = self.is_recurrent() || self.layers.iter().any(|l| l.kv_cache.is_windowed());
        stateful
            && !self.is_wide()
            && !self.has_split_layers()
            && !self.has_context_shards()
            && self.layers.iter().all(|l| l.cpu_q8_kv.is_none())
    }

    /// The state of snapshot `index` as one self-contained buffer: a manifest holding every
    /// layer's whole state (recurrent state, windowed rows, every global layer's rows), then the
    /// logits the snapshot resumes decoding from, when it has them. The block size is one more
    /// than the snapshot's length, so no rows go to blocks.
    pub(crate) fn export_state(&self, index: usize) -> Result<Vec<u8>> {
        let err = |m: &str| crate::tensor::Error::msg(format!("state handover: {m}"));
        let snap = self
            .kv_snapshots
            .get(index)
            .ok_or_else(|| err("the snapshot is gone"))?;
        if snap.layers.iter().any(|s| !s.shards.is_empty()) {
            return Err(err("a layer is split over cards"));
        }
        let pos = snap.kv_len;
        let layout = self.state_layout(pos + 1);
        let mut layers = Vec::with_capacity(self.layers.len());
        for ((l, s), kind) in self.layers.iter().zip(&snap.layers).zip(&layout.kinds) {
            layers.push(match kind {
                StateKind::Nothing => Held::Nothing,
                StateKind::Mixer => {
                    let state = s
                        .mixer
                        .as_ref()
                        .ok_or_else(|| err("a mixer holds no state"))?;
                    Held::Mixer(MixerHost::of(state)?)
                }
                StateKind::Global => {
                    let rows = snapshot_rows(l, s)?;
                    if rows.n != pos {
                        return Err(err("a global layer does not hold the whole prompt"));
                    }
                    Held::Global(rows)
                }
                StateKind::Window => Held::Window(snapshot_rows(l, s)?),
            });
        }
        let point = Point {
            pos,
            layers,
            draft: None,
        };
        let manifest = encode_manifest(&handover_identity(&layout), &layout, &point);
        let mut w = Writer(Vec::with_capacity(manifest.len() + 64));
        w.0.extend(HANDOVER_MAGIC);
        w.0.extend(FORMAT_VERSION.to_le_bytes());
        w.u64(manifest.len());
        w.0.extend(&manifest);
        match &snap.logits {
            Some(t) => {
                w.u8(1);
                w.tens(&Tens::of(t)?);
            }
            None => w.u8(0),
        }
        Ok(w.seal())
    }

    /// Enters the state `bytes` carry as the snapshot of `tokens`, after the same checks a stored
    /// state passes: the layer kinds, the row geometry and the length. A recurrent model's state
    /// without the logits it resumes from is refused.
    pub(crate) fn import_state(&mut self, tokens: Vec<u32>, bytes: &[u8]) -> AnyResult<()> {
        let pos = tokens.len();
        if pos == 0 {
            bail!("state handover: no tokens");
        }
        let layout = self.state_layout(pos + 1);
        let mut r = Reader::open(bytes)?;
        if r.take(8)? != HANDOVER_MAGIC {
            bail!("state handover: not a state handover");
        }
        let version = u32::from_le_bytes(r.take(4)?.try_into().unwrap());
        if version != FORMAT_VERSION {
            bail!("state handover: format version {version}, this build reads {FORMAT_VERSION}");
        }
        let manifest_len = r.u64()?;
        let point = decode_manifest(r.take(manifest_len)?, &handover_identity(&layout), &layout)?;
        if point.pos != pos {
            bail!("state handover: the state is not the tokens' length");
        }
        let logits = match r.u8()? {
            0 => None,
            _ => Some(r.tens()?.tensor(&Device::Cpu).map_err(|e| anyhow!("{e}"))?),
        };
        r.done()?;
        if logits.is_none() && self.is_recurrent() {
            bail!("state handover: a recurrent state arrives without its logits");
        }
        let layers = self
            .point_snapshots(&point, &[])
            .map_err(|e| anyhow!("{e}"))?;
        self.store_snapshot(tokens, pos, layers, logits);
        Ok(())
    }

    /// Forward `prompt[from..]` in chunks of at most `chunk` tokens, cut at multiples of `stride`
    /// so the state can be kept where a prefill passes it. Returns the last logits and the states
    /// at the cuts short of the end.
    pub(crate) fn prefill_with_grid(
        &mut self,
        prompt: &[u32],
        from: usize,
        chunk: usize,
        stride: usize,
        layout: &Layout,
    ) -> Result<(Tensor, Vec<Point>)> {
        let stride = stride.max(1);
        let chunk = chunk.max(1);
        let mut grid = Vec::new();
        let mut pos = from;
        let mut last = None;
        while pos < prompt.len() {
            let boundary = (pos / stride + 1) * stride;
            let end = boundary.min(prompt.len());
            while pos < end {
                let n = chunk.min(end - pos);
                let x = Tensor::new(&prompt[pos..pos + n], &Device::Cpu)?.unsqueeze(0)?;
                last = Some(self.forward(&x, pos)?);
                pos += n;
            }
            if end == boundary && end < prompt.len() {
                match self.capture_point(end, layout) {
                    Ok(p) => grid.push(p),
                    Err(e) => tracing::info!("state cache: no state kept at token {end}: {e}"),
                }
            }
        }
        let logits = last.ok_or_else(|| crate::tensor::Error::msg("empty prefill".to_string()))?;
        Ok((logits, grid))
    }
}

/// What a state handover is checked against besides the manifest's own checks: the layer kinds
/// and row geometry, hashed so the two instances need no store to agree on them.
fn handover_identity(layout: &Layout) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"loken state handover");
    h.update(FORMAT_VERSION.to_le_bytes());
    for (kind, (n_kv, head_dim)) in layout.kinds.iter().zip(&layout.geometry) {
        h.update([kind.tag()]);
        h.update((*n_kv as u64).to_le_bytes());
        h.update((*head_dim as u64).to_le_bytes());
    }
    h.finalize().into()
}

/// The rows a layer's snapshot holds, as host f32 with the element type they are held in.
fn snapshot_rows(l: &GenericTransformerLayer, s: &super::LayerKvSnapshot) -> Result<Rows> {
    let (dtype, n, (k, v)) = match (&s.spec, &s.quant, &s.cpu_f16) {
        (Some(spec), _, _) => (
            dtype_code(spec.dtype())?,
            spec.len(),
            spec.host_rows(0, spec.len())?,
        ),
        // A quantised layer's rows come out dequantised to F32, which holds them exactly.
        (None, Some(q), _) => (
            dtype_code(DType::F32)?,
            q.len(),
            q.host_rows(0, q.len())
                .map_err(|e| crate::tensor::Error::msg(e.to_string()))?,
        ),
        (None, None, Some(c)) => (0, c.len(), c.host_rows(0, c.len())),
        (None, None, None) => (0, 0, (Vec::new(), Vec::new())),
    };
    Ok(Rows {
        dtype,
        n_kv: l.n_kv_head,
        head_dim: l.head_dim,
        n,
        k,
        v,
    })
}

/// A layer attending on the host reads its decode from a store beside the KV cache, built from
/// the same rows as a prefill appends them: built again from the restored rows.
fn rebuild_host_rows(l: &mut GenericTransformerLayer) {
    let n = l.kv_cache.current_seq_len();
    if matches!(l.mixer, Mixer::Attention) && n > 0 {
        l.cpu_q8_kv = None;
        l.cpu_f16_kv = None;
        if let Some((k, v)) = l.kv_cache.current_kv() {
            l.cpu_q8_append(&k, &v, n, 0, false);
        }
    }
}

fn rows_snapshot(
    l: &GenericTransformerLayer,
    k: &[f32],
    v: &[f32],
    dtype: u8,
) -> Result<super::LayerKvSnapshot> {
    if k.is_empty() {
        return Ok(super::LayerKvSnapshot {
            spec: None,
            quant: None,
            cpu_f16: None,
            mixer: None,
            shards: Vec::new(),
        });
    }
    let dt = dtype_of(dtype).map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
    super::graph::layer_snapshot_from_rows(l, k, v, dt)
}

// --- The cache -----------------------------------------------------------------------------

/// The store a model's states are written to and read from, with what names them.
#[derive(Clone)]
pub(crate) struct HybridStateCache {
    store: Arc<StateStore>,
    identity: [u8; 32],
    layout: Arc<Layout>,
    last_stride: Arc<std::sync::atomic::AtomicUsize>,
}

impl HybridStateCache {
    pub(crate) fn tuning(&self) -> &state_tune::Tuning {
        self.store.tuning()
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Restores the longest stored state `prompt` extends, shorter than `prompt` by at least a
    /// token so one is left to produce logits, and returns the tokens it covers; zero when none
    /// is usable. A stored state that proves unusable is dropped.
    pub(crate) fn restore(&self, model: &mut GenericHeteroTransformer, prompt: &[u32]) -> usize {
        // A token is left to produce logits; with a draft head, two, so the prefill of the
        // rest is a chunk the head reads.
        let left = 1 + model.has_draft_head() as usize;
        if prompt.len() <= left {
            return 0;
        }
        let keys = self.store.keys(&self.identity, prompt);
        for entry in self.store.candidates(&keys, prompt.len() - left) {
            let bytes = self.store.entry_bytes(&entry);
            let m = self.store.tuning().get();
            if !state_tune::restore_pays(
                entry.len,
                bytes,
                m.restore.secs_per(),
                m.prefill.secs_per(),
            ) {
                tracing::info!(
                    "state cache: a {}-token state is not restored: reading {} MiB is expected to cost more than prefilling the tokens",
                    entry.len,
                    bytes >> 20
                );
                continue;
            }
            let started = std::time::Instant::now();
            match self.restore_entry(model, &entry) {
                Ok(()) => {
                    let secs = started.elapsed().as_secs_f64();
                    self.store.tuning().restored(bytes, secs);
                    tracing::info!(
                        "state cache: restored {} tokens ({} MiB) in {:.2} s",
                        entry.len,
                        bytes >> 20,
                        secs
                    );
                    return entry.len;
                }
                Err(e) => {
                    tracing::warn!("state cache: dropping a {}-token state: {e}", entry.len);
                    self.store.forget(&entry);
                }
            }
        }
        0
    }

    fn restore_entry(&self, model: &mut GenericHeteroTransformer, entry: &Entry) -> AnyResult<()> {
        let layout = &*self.layout;
        let point = decode_manifest(&self.store.read(entry)?, &self.identity, layout)?;
        if point.pos != entry.len || entry.blocks.len() != point.pos / layout.block_tokens {
            bail!("state cache: a manifest that does not match its name");
        }
        let blocks = entry
            .blocks
            .iter()
            .map(|h| decode_block(&self.store.read_block(h)?, layout))
            .collect::<AnyResult<Vec<_>>>()?;
        model
            .restore_point(&point, &blocks)
            .map_err(|e| anyhow!("{e}"))
    }

    /// The spacing of the states a prefill keeps, from what the store has measured, for a prefill
    /// that runs forwards of `chunk` tokens. Logged when it changes.
    pub(crate) fn stride(&self, chunk: usize) -> usize {
        let m = self.store.tuning().get();
        let stride = state_tune::grid_stride(
            self.layout.block_tokens,
            chunk,
            m.manifest_bytes,
            m.write.per_sec(),
            m.prefill.secs_per(),
        );
        if self
            .last_stride
            .swap(stride, std::sync::atomic::Ordering::Relaxed)
            != stride
        {
            tracing::info!(
                "state cache: a state every {stride} tokens (chunk {chunk}, manifest {} KiB, write {} MB/s, prefill {} tokens/s)",
                m.manifest_bytes >> 10,
                m.write.per_sec().map_or(0.0, |w| w / 1e6) as u64,
                m.prefill.per_sec().unwrap_or(0.0) as u64
            );
        }
        stride
    }

    /// Keeps the state the model stands in after `prompt`, and the `grid` states a prefill passed
    /// on the way. The model's rows are copied here; encoding and writing run on a thread of their
    /// own, queued behind earlier writes only while the disk drains them within `prefill_secs`,
    /// the time the prefill took. A prompt shorter than a block is not worth a file.
    pub(crate) fn save(
        &self,
        model: &GenericHeteroTransformer,
        prompt: &[u32],
        grid: Vec<Point>,
        prefill_secs: Option<f64>,
    ) {
        let layout = &*self.layout;
        let g = layout.block_tokens;
        if prompt.len() < g {
            return;
        }
        let end = match model.capture_point(prompt.len(), layout) {
            Ok(p) => p,
            Err(e) => {
                tracing::info!("state cache: not stored: {e}");
                return;
            }
        };
        let keys = self.store.keys(&self.identity, prompt);
        let nblocks = prompt.len() / g;
        let first = self.store.first_missing_block(&keys, nblocks);
        let mut blocks = Vec::with_capacity(nblocks - first);
        for b in first..nblocks {
            match model.capture_block(b, layout) {
                Ok(bytes) => blocks.push(bytes),
                Err(e) => {
                    tracing::info!("state cache: not stored: {e}");
                    return;
                }
            }
        }
        let m = self.store.tuning().get();
        let bytes = blocks.iter().map(|b| b.len() as u64).sum::<u64>()
            + m.manifest_bytes * (grid.len() as u64 + 1);
        let Some(ticket) = self.store.admit(bytes, prefill_secs) else {
            tracing::info!("state cache: a write is skipped, the disk is still behind");
            return;
        };
        let this = self.clone();
        let tokens = prompt.to_vec();
        std::thread::spawn(move || {
            ticket.run(|| {
                let keys = this.store.keys(&this.identity, &tokens);
                let mut points = grid;
                points.push(end);
                for p in points {
                    let manifest = encode_manifest(&this.identity, &this.layout, &p);
                    let mut lost = false;
                    let put = this.store.put(&keys, p.pos, &manifest, p.pos / g, |b| {
                        match b.checked_sub(first).and_then(|i| blocks.get(i)) {
                            Some(bytes) => bytes.clone(),
                            None => {
                                lost = true;
                                Vec::new()
                            }
                        }
                    });
                    match put {
                        Ok(()) if lost => {
                            // A block that was on disk when the rows were copied has gone since.
                            for e in this.store.candidates(&keys, p.pos) {
                                if e.len == p.pos {
                                    this.store.forget(&e);
                                }
                            }
                        }
                        Ok(()) => tracing::info!(
                            "state cache: stored a {}-token state ({} KiB manifest, {} blocks)",
                            p.pos,
                            manifest.len() >> 10,
                            p.pos / g
                        ),
                        Err(e) => tracing::warn!("state cache: not stored at {}: {e}", p.pos),
                    }
                }
            })
        });
    }
}

/// A draft head's state encoded and decoded again, for the round-trip test.
#[cfg(test)]
pub(crate) fn draft_round_trip(d: &DraftHeld) -> DraftHeld {
    let mut w = Writer(Vec::new());
    write_draft(&mut w, d);
    let bytes = w.seal();
    let mut r = Reader::open(&bytes).unwrap();
    let out = read_draft(&mut r).unwrap();
    r.done().unwrap();
    out
}

/// A block encoded and decoded again, for the round-trip test.
#[cfg(test)]
pub(crate) fn block_round_trip(
    rows: &[(Vec<f32>, Vec<f32>)],
    layout: &Layout,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    decode_block(&encode_block(rows), layout).unwrap()
}
