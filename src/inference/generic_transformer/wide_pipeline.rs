//! A Sinkhorn wide-residual model cut into layer ranges that nodes run in turn.
//!
//! A cut between 2 layers carries more than the residual:
//! - the residual, `streams` copies of each row,
//! - the pre mix each row's next token mixer collapses with,
//! - what the band's sources before the cut left for the readers after it,
//! - the compressed ids the engram rows after the cut hash, on a model with engram layers.
//!
//! The first range embeds the tokens.
//! The last range runs the final norm and the head.
//! Each range holds the mixer states, the band cell and the token log of its own layers.
//!
//! A verify runs its rows 1 at a time, so the band's sources leave 1 value per row.
//! Any other forward leaves 1 value for all its rows.
//!
//! The bytes of a boundary are a header of 3 numbers, then a body.
//! The body of a decode step's boundary is the byte format of the V4.1 path's `DecodeBoundary`.
//!
//! A verify cut back inside its rows runs the kept rows again through each range.
//! The band's sources then leave 1 value for the kept rows, which the cut carries on.

use super::compressed_band::BandLeft;
use super::wide_sparse_layer::{TokenLog, WideSparseHead};
use super::*;

/// Bytes of a boundary's header: the first position, the rows and the band values.
pub const HEADER_BYTES: usize = 12;

/// What crosses a cut of a Sinkhorn wide-residual stack, for the rows of 1 forward.
#[derive(Debug, Clone, PartialEq)]
pub struct WideBoundary {
    /// The position of the first row.
    pub first: usize,
    /// Parallel copies of the hidden state.
    pub streams: usize,
    /// Values of 1 copy of a row.
    pub hidden: usize,
    /// `rows * streams * hidden` values, row after row, each row stream after stream.
    pub residual: Vec<f32>,
    /// Per row, the pre mix the next layer's token mixer collapses with, `streams` values.
    pub pre: Vec<Vec<f32>>,
    /// The compressed ids the engram rows of the forward read, oldest first.
    ///
    /// They are the ids of the positions `TokenLog::window_start(first)..first + rows`.
    /// `None` for a model without engram layers.
    pub recent: Option<Vec<u32>>,
    /// Values of a compressed row and of an index key.
    pub widths: (usize, usize),
    /// What the band's sources left: 1 value for all the rows, or 1 per row.
    pub band: Vec<BandLeft>,
}

/// What the band's sources left for the rows a cut-back verify keeps.
#[derive(Debug, Clone, PartialEq)]
pub struct WideSettle {
    /// Values of a compressed row and of an index key.
    pub widths: (usize, usize),
    pub band: BandLeft,
}

/// Where the band's sources leave their values for a forward of `rows` rows from `first`: 1
/// place per row when the rows run 1 at a time, else 1 for all of them.
fn band_places(first: usize, rows: usize, stepped: bool) -> Vec<(usize, usize)> {
    match stepped && rows > 1 {
        true => (first..first + rows).map(|p| (p, 1)).collect(),
        false => vec![(first, rows)],
    }
}

/// Whether the rows of a forward of `rows` rows from `first` run 1 at a time, as the token
/// mixers of a Sinkhorn layer decide it.
fn steps_rows(first: usize, rows: usize) -> bool {
    rows > 1 && first > 0 && crate::tensor::quantized::DecodeOrder::active()
}

fn refuse(msg: impl Into<String>) -> crate::tensor::Error {
    crate::tensor::Error::msg(msg.into())
}

fn put_u32(out: &mut Vec<u8>, v: usize) -> Result<()> {
    let v = u32::try_from(v).map_err(|_| refuse("a boundary number above u32"))?;
    out.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Rows of `width` values, as a flag, the dimensions `[1, rows, width]` and the values.
fn put_rows(out: &mut Vec<u8>, rows: Option<&[f32]>, width: usize) -> Result<()> {
    let Some(values) = rows else {
        return put_u32(out, 0);
    };
    if width == 0 || !values.len().is_multiple_of(width) {
        crate::tensor::bail!("{} band values do not make rows of {width}", values.len());
    }
    put_u32(out, 1)?;
    put_u32(out, 3)?;
    for d in [1, values.len() / width, width] {
        put_u32(out, d)?;
    }
    put_f32s(out, values);
    Ok(())
}

/// 1 forward's band values: the compressed rows, the index keys, the picks, the candidate mask.
fn put_band(out: &mut Vec<u8>, left: &BandLeft, (row, key): (usize, usize)) -> Result<()> {
    put_rows(out, left.compressed.as_deref(), row)?;
    put_rows(out, left.index_keys.as_deref(), key)?;
    match &left.picks {
        None => put_u32(out, 0)?,
        Some(picks) => {
            put_u32(out, 1)?;
            put_u32(out, picks.len())?;
            for p in picks {
                out.extend_from_slice(&p.to_le_bytes());
            }
        }
    }
    match &left.candidates {
        None => put_u32(out, 0)?,
        Some(mask) => {
            put_u32(out, 1)?;
            put_u32(out, mask.len())?;
            out.extend(mask.iter().map(|&b| u8::from(b)));
        }
    }
    Ok(())
}

/// A cursor over the bytes of a boundary.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        match self.bytes.get(self.at..self.at + n) {
            Some(b) => {
                self.at += n;
                Ok(b)
            }
            None => Err(refuse("a wide boundary cut short")),
        }
    }

    fn word(&mut self) -> Result<[u8; 4]> {
        let b = self.take(4)?;
        Ok([b[0], b[1], b[2], b[3]])
    }

    fn u32(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.word()?) as usize)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        (0..n)
            .map(|_| Ok(f32::from_le_bytes(self.word()?)))
            .collect()
    }

    /// Rows written by `put_rows`, with their width.
    fn rows(&mut self) -> Result<Option<(Vec<f32>, usize)>> {
        if self.u32()? == 0 {
            return Ok(None);
        }
        let dims = (0..self.u32()?)
            .map(|_| self.u32())
            .collect::<Result<Vec<_>>>()?;
        let width = dims.last().copied().unwrap_or(0);
        let values = self.f32s(dims.iter().product())?;
        Ok(Some((values, width)))
    }

    /// A band written by `put_band`, with the widths of its rows, 0 where it carries none.
    fn band(&mut self) -> Result<(BandLeft, (usize, usize))> {
        let compressed = self.rows()?;
        let index_keys = self.rows()?;
        let picks = match self.u32()? {
            0 => None,
            _ => {
                let n = self.u32()?;
                Some(
                    (0..n)
                        .map(|_| Ok(i32::from_le_bytes(self.word()?)))
                        .collect::<Result<Vec<_>>>()?,
                )
            }
        };
        let candidates = match self.u32()? {
            0 => None,
            _ => {
                let n = self.u32()?;
                Some(self.take(n)?.iter().map(|&b| b != 0).collect())
            }
        };
        let widths = (
            compressed.as_ref().map_or(0, |c| c.1),
            index_keys.as_ref().map_or(0, |k| k.1),
        );
        let left = BandLeft {
            compressed: compressed.map(|c| c.0),
            index_keys: index_keys.map(|k| k.0),
            picks,
            candidates,
        };
        Ok((left, widths))
    }
}

/// `held` where it is set, else `other`.
fn either_width(held: (usize, usize), other: (usize, usize)) -> (usize, usize) {
    (
        if held.0 > 0 { held.0 } else { other.0 },
        if held.1 > 0 { held.1 } else { other.1 },
    )
}

impl WideBoundary {
    /// Rows of the forward.
    pub fn rows(&self) -> usize {
        self.pre.len()
    }

    /// The bytes after the header.
    ///
    /// For 1 row and 1 band value they are the bytes of the V4.1 `DecodeBoundary` of the same
    /// cut.
    pub fn body_bytes(&self) -> Result<Vec<u8>> {
        let rows = self.rows();
        if self.residual.len() != rows * self.streams * self.hidden {
            crate::tensor::bail!(
                "a residual of {} values for {rows} rows of {} streams of {}",
                self.residual.len(),
                self.streams,
                self.hidden
            );
        }
        let mut out = Vec::new();
        put_u32(&mut out, self.streams)?;
        put_u32(&mut out, self.hidden)?;
        put_f32s(&mut out, &self.residual);
        put_u32(&mut out, rows)?;
        for row in &self.pre {
            put_u32(&mut out, row.len())?;
            put_f32s(&mut out, row);
        }
        match &self.recent {
            Some(ids) => {
                put_u32(&mut out, 1)?;
                put_u32(&mut out, ids.len())?;
                for id in ids {
                    out.extend_from_slice(&id.to_le_bytes());
                }
            }
            None => put_u32(&mut out, 0)?,
        }
        for left in &self.band {
            put_band(&mut out, left, self.widths)?;
        }
        Ok(out)
    }

    /// The header, then the body.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(HEADER_BYTES);
        put_u32(&mut out, self.first)?;
        put_u32(&mut out, self.rows())?;
        put_u32(&mut out, self.band.len())?;
        out.extend(self.body_bytes()?);
        Ok(out)
    }

    /// The boundary `to_bytes` wrote.
    ///
    /// Refused when the bytes end early or carry more than the boundary.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader { bytes, at: 0 };
        let first = r.u32()?;
        let rows = r.u32()?;
        let places = r.u32()?;
        if places != 1 && places != rows {
            crate::tensor::bail!("{places} band values for {rows} rows");
        }
        let streams = r.u32()?;
        let hidden = r.u32()?;
        let residual = r.f32s(rows * streams * hidden)?;
        let pre_rows = r.u32()?;
        if pre_rows != rows {
            crate::tensor::bail!("{pre_rows} pre mixes for {rows} rows");
        }
        let pre = (0..rows)
            .map(|_| {
                let n = r.u32()?;
                r.f32s(n)
            })
            .collect::<Result<Vec<_>>>()?;
        let recent = match r.u32()? {
            0 => None,
            _ => {
                let n = r.u32()?;
                Some(
                    (0..n)
                        .map(|_| r.u32().map(|v| v as u32))
                        .collect::<Result<Vec<_>>>()?,
                )
            }
        };
        let mut widths = (0, 0);
        let mut band = Vec::with_capacity(places);
        for _ in 0..places {
            let (left, w) = r.band()?;
            widths = either_width(widths, w);
            band.push(left);
        }
        if r.at != bytes.len() {
            crate::tensor::bail!("{} bytes after a wide boundary", bytes.len() - r.at);
        }
        Ok(Self {
            first,
            streams,
            hidden,
            residual,
            pre,
            recent,
            widths,
            band,
        })
    }
}

impl WideSettle {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        put_band(&mut out, &self.band, self.widths)?;
        Ok(out)
    }

    /// The settle `to_bytes` wrote.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader { bytes, at: 0 };
        let (band, widths) = r.band()?;
        if r.at != bytes.len() {
            crate::tensor::bail!("{} bytes after a band settle", bytes.len() - r.at);
        }
        Ok(Self { widths, band })
    }
}

impl GenericHeteroTransformer {
    /// Whether a cut of this model carries a [`WideBoundary`]: the model is a Sinkhorn wide
    /// residual.
    pub fn cuts_wide(&self) -> bool {
        self.wide_sparse.is_some()
    }

    /// The part of a Sinkhorn wide residual the layers pass to each other.
    ///
    /// Refused for another model, and for a model whose block draft reads layers on both sides
    /// of a cut.
    fn cut_head(&self) -> Result<&WideSparseHead> {
        let ws = self
            .wide_sparse
            .as_ref()
            .ok_or_else(|| refuse("a wide boundary needs a Sinkhorn wide residual"))?;
        if ws.draft.is_some() {
            crate::tensor::bail!("a block draft reads the layers of every range of the stack");
        }
        Ok(ws)
    }

    /// Values of a compressed row and of an index key of the band layers held, 0 without one.
    fn band_widths(&self) -> (usize, usize) {
        self.layers
            .iter()
            .find_map(|l| match &l.mixer {
                mixer::Mixer::CompressedBand(b) => {
                    Some((b.base.dims.head_dim, b.band.index_head_dim))
                }
                _ => None,
            })
            .unwrap_or((0, 0))
    }

    /// 1 forward of `rows` rows from `first` through the layers held, as `forward` runs it, or
    /// as `forward_all` does with `all_positions`.
    ///
    /// `start` gives the residual the layers start from.
    /// `finish` takes the residual after them.
    /// Both learn whether the rows run 1 at a time.
    /// `ids` are the rows' token ids a verify records, `None` on a range that has no ids.
    fn wide_pass<R>(
        &mut self,
        first: usize,
        rows: usize,
        all_positions: bool,
        ids: Option<&Tensor>,
        start: impl FnOnce(&mut Self, bool) -> Result<Tensor>,
        finish: impl FnOnce(&mut Self, Tensor, bool, bool) -> Result<R>,
    ) -> Result<R> {
        self.cut_head()?;
        let verify = all_positions
            && first > 0
            && rows <= crate::inference::engine::decode_step::VERIFY_ROWS;
        let recording = verify && self.is_recurrent();
        if recording {
            match ids {
                Some(ids) => self.spec_open(ids, first)?,
                None => {
                    // The rows' ids feed only an n-gram history, which a wide residual lacks.
                    let ids = Tensor::from_vec(vec![0u32; rows], (1, rows), &Device::Cpu)?;
                    self.spec_open(&ids, first)?;
                }
            }
        }
        #[cfg(feature = "gpu")]
        let _in_place = recording.then(|| crate::inference::moe_cuda::in_place_state(true));
        let _order = verify.then(crate::tensor::quantized::DecodeOrder::enter);
        let stepped = steps_rows(first, rows);
        let out = self.wide_layers(first, rows, stepped, start, finish);
        if recording {
            self.spec_close(out.is_ok());
        }
        out
    }

    fn wide_layers<R>(
        &mut self,
        first: usize,
        rows: usize,
        stepped: bool,
        start: impl FnOnce(&mut Self, bool) -> Result<Tensor>,
        finish: impl FnOnce(&mut Self, Tensor, bool, bool) -> Result<R>,
    ) -> Result<R> {
        self.spec_clear();
        let hidden = start(self, stepped)?;
        crate::tensor::quant_cpu::set_small_model_prefill(
            rows > 1 && self.config.embedding_length <= 1024,
        );
        let (hidden, on_cuda) = self.run_layers(hidden, None, first, rows)?;
        finish(self, hidden, on_cuda, stepped)
    }

    /// The boundary after the layers held, for the residual `hidden` of the rows from `first`.
    ///
    /// `widths` are the widths a boundary received, for band values the layers held do not
    /// make.
    fn boundary_after(
        &self,
        hidden: Tensor,
        first: usize,
        stepped: bool,
        widths: (usize, usize),
    ) -> Result<WideBoundary> {
        let ws = self.cut_head()?;
        let hidden = hidden.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let (_, rows, width) = hidden.dims3()?;
        let recent = match (&ws.tokens, &ws.engram) {
            (Some(tokens), Some(meta)) => Some(tokens.window(meta, first, rows)?),
            _ => None,
        };
        let band = band_places(first, rows, stepped)
            .into_iter()
            .map(|at| ws.cell.left_at(at))
            .collect();
        Ok(WideBoundary {
            first,
            streams: ws.streams,
            hidden: width / ws.streams.max(1),
            residual: hidden.flatten_all()?.to_vec1::<f32>()?,
            pre: ws.carry.get(first, rows)?,
            recent,
            widths: either_width(self.band_widths(), widths),
            band,
        })
    }

    /// The residual of `b`'s rows, with what `b` carries held for the layers this model holds:
    /// the pre mixes, the band values and the compressed ids.
    fn open_boundary(&self, b: &WideBoundary, stepped: bool) -> Result<Tensor> {
        let ws = self.cut_head()?;
        let rows = b.rows();
        if b.streams != ws.streams || b.streams * b.hidden != self.stream_width() {
            crate::tensor::bail!(
                "a boundary of {} streams of {} for a residual of {} streams {} wide",
                b.streams,
                b.hidden,
                ws.streams,
                self.stream_width()
            );
        }
        let places = band_places(b.first, rows, stepped);
        if b.band.len() != places.len() {
            crate::tensor::bail!(
                "a boundary of {} band values for a forward that leaves {}",
                b.band.len(),
                places.len()
            );
        }
        match (&ws.tokens, &ws.engram, &b.recent) {
            (Some(tokens), Some(meta), Some(ids)) => {
                tokens.receive(TokenLog::window_start(meta, b.first), ids.clone())
            }
            (None, _, None) => {}
            _ => crate::tensor::bail!("the boundary and the model disagree on engram layers"),
        }
        ws.carry.put(b.first, rows, b.pre.clone());
        ws.cell.open(b.first, rows);
        for (at, left) in places.into_iter().zip(&b.band) {
            ws.cell.leave_at(at, left.clone());
        }
        Tensor::from_vec(
            b.residual.clone(),
            (1, rows, b.streams * b.hidden),
            &Device::Cpu,
        )
    }

    /// The boundary the first range of a cut stack hands on: the rows `ids` `[1, rows]` from
    /// `first` through the layers this model holds.
    ///
    /// `all_positions` is the request the last range answers: a verify records its rows here
    /// as `forward_all` does.
    pub fn wide_head_forward(
        &mut self,
        ids: &Tensor,
        first: usize,
        all_positions: bool,
    ) -> Result<WideBoundary> {
        let rows = ids.dims2()?.1;
        self.wide_pass(
            first,
            rows,
            all_positions,
            Some(ids),
            |model, _| Ok(model.embed_stage(ids, first)?.0),
            |model, hidden, _, stepped| model.boundary_after(hidden, first, stepped, (0, 0)),
        )
    }

    /// The boundary an interior range hands on: `b` through the layers this model holds.
    pub fn wide_interior_forward(
        &mut self,
        b: &WideBoundary,
        all_positions: bool,
    ) -> Result<WideBoundary> {
        self.wide_pass(
            b.first,
            b.rows(),
            all_positions,
            None,
            |model, stepped| model.open_boundary(b, stepped),
            |model, hidden, _, stepped| model.boundary_after(hidden, b.first, stepped, b.widths),
        )
    }

    /// The logits the last range answers with: `b` through the layers this model holds, the
    /// final norm and the head.
    ///
    /// Every row's `[1, rows, vocab]` with `all_positions`, else the last row's `[1, vocab]`.
    pub fn wide_tail_forward(&mut self, b: &WideBoundary, all_positions: bool) -> Result<Tensor> {
        let rows = b.rows();
        self.wide_pass(
            b.first,
            rows,
            all_positions,
            None,
            |model, stepped| model.open_boundary(b, stepped),
            |model, hidden, on_cuda, _| model.output_stage(hidden, on_cuda, rows, all_positions),
        )
    }

    /// Cuts the layers held back to `new_len` tokens, as `trim_kv` does.
    ///
    /// When the cut falls inside the rows of the verify just run, the kept rows run again
    /// through the layers, and the result is what the band's sources left for them.
    /// The range after the cut takes it with [`Self::wide_trim_after`].
    pub fn wide_trim(&mut self, new_len: usize) -> Result<Option<WideSettle>> {
        let kept = self.kept_rows(new_len);
        self.trim_kv(new_len);
        let Some(at) = kept else {
            return Ok(None);
        };
        let ws = self.cut_head()?;
        Ok(Some(WideSettle {
            widths: self.band_widths(),
            band: ws.cell.left_at(at),
        }))
    }

    /// Cuts the layers held back to `new_len` tokens, the kept rows reading what the band's
    /// sources before the cut left for them in `settle`.
    ///
    /// Gives what the sources left once the layers held ran the kept rows again, for the range
    /// after this one.
    pub fn wide_trim_after(
        &mut self,
        new_len: usize,
        settle: Option<&WideSettle>,
    ) -> Result<Option<WideSettle>> {
        let kept = self.kept_rows(new_len);
        if let Some(at) = kept {
            let left = settle.map(|s| s.band.clone()).unwrap_or_default();
            self.cut_head()?.cell.leave_at(at, left);
        }
        self.trim_kv(new_len);
        let Some(at) = kept else {
            return Ok(None);
        };
        let ws = self.cut_head()?;
        let received = settle.map_or((0, 0), |s| s.widths);
        Ok(Some(WideSettle {
            widths: either_width(self.band_widths(), received),
            band: ws.cell.left_at(at),
        }))
    }

    /// The place the band's sources leave values in for the rows a cut to `new_len` keeps of
    /// the verify just recorded, `None` when the cut keeps none of them or all of them.
    fn kept_rows(&self, new_len: usize) -> Option<(usize, usize)> {
        let w = self.spec_window.as_ref()?;
        (new_len > w.base && new_len < w.base + w.rows).then(|| (w.base, new_len - w.base))
    }
}
