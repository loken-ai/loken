//! Compressed band attention over 1 shared key head, on the host.
//!
//! A band layer attends as a window layer does (`latent_mqa.rs`), over 2 sets of rows at once.
//! The first set is its own window of key rows.
//! The second is a compressed copy of the past, 1 row per `ratio` tokens. An indexer keeps at
//! most `index_topk` of those rows per query.
//!
//! The layers of a band share the compressed rows through a [`BandCell`]. A layer has some of
//! these roles:
//! - a kv-source pools every `ratio` input rows into 1 compressed row through a softmax gate per
//!   channel, rotates it at the group's first position and rounds it to fp4,
//! - an index-source scores the compressed rows with a small fp4 attention and keeps the best,
//! - an index-source that is also a kv-source keeps an fp4 index key per group,
//! - a candidate source also keeps the best blocks of compressed rows per query, and a later
//!   index-source scores only inside them.
//!
//! A layer reads from the cell what it does not build itself.
//! Layers run in order, so a source leaves its rows in the cell before a later layer reads them.
//!
//! A forward of 1 row after position 0 is a decode step.
//! A kv-source then pools a group once the group's last input row arrives.
//!
//! The mixer returns the groups' slices of the grouped output projection, as the window mixer
//! does.

use super::latent_mqa::{
    host_rows, phases, pow2_ceil, rotate_tails, LatentMqa, LatentMqaState, SCALE_FLOOR,
};
use super::wide_sparse_config::{LayerKind, WideSparseMeta};
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::ops::rms_norm;
use crate::tensor::{Device, Result, Tensor};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// The e2m1 magnitudes.
pub(super) const FP4_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The largest e2m1 magnitude. The round trip clamps there.
pub(super) const FP4_MAX: f32 = 6.0;

/// Values of an index query or index key that share 1 power-of-two scale.
pub(super) const INDEX_BLOCK: usize = 32;

/// Values of a compressed row that share 1 scale.
pub(super) const COMPRESSED_BLOCK: usize = 16;

/// What a band layer adds to the window mixer's widths, and its roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandDims {
    /// Tokens pooled into 1 compressed row.
    pub ratio: usize,
    pub kv_source: bool,
    pub index_source: bool,
    /// The layer keeps the best blocks per query for a later index-source.
    pub candidate_source: bool,
    /// The layer scores only inside the blocks a candidate source kept.
    pub uses_candidates: bool,
    /// Compressed rows a query attends at most.
    pub index_topk: usize,
    pub index_heads: usize,
    pub index_head_dim: usize,
    /// Blocks a candidate source keeps per query.
    pub candidate_blocks: usize,
    /// Compressed rows per block.
    pub candidate_block_size: usize,
}

impl BandDims {
    /// Layer `il`'s, `None` for a window layer.
    pub fn from_meta(meta: &WideSparseMeta, il: usize) -> Option<Self> {
        let LayerKind::Compressed {
            ratio,
            kv_source,
            index_source,
        } = meta.layer_kind(il)
        else {
            return None;
        };
        let ix = &meta.indexer;
        let cand = usize::try_from(ix.candidate_source_layer).ok();
        Some(Self {
            ratio,
            kv_source,
            index_source,
            candidate_source: index_source && cand == Some(il),
            uses_candidates: index_source && cand.is_some_and(|c| c < il),
            index_topk: ix.top_k,
            index_heads: ix.heads,
            index_head_dim: ix.head_dim,
            candidate_blocks: ix.candidate_top_k_blocks,
            candidate_block_size: ix.candidate_block_size,
        })
    }
}

/// A kv-source's pooling weights, F32.
pub struct CompressorWeights {
    /// `[head_dim, hidden]`.
    pub wkv: Tensor,
    /// The pooling gate `[head_dim, hidden]`, absent at ratio 1.
    pub wgate: Option<Tensor>,
    /// Scale of the norm on a pooled row, `[head_dim]`.
    pub norm: Tensor,
}

/// An index-source's weights.
pub struct BandIndexerWeights {
    /// `[index_heads * index_head_dim, q_lora_rank]`.
    pub wq_b: Projection,
    /// The weight of each index head from the layer's input, `[index_heads, hidden]` F32.
    pub weights_proj: Tensor,
    /// The index key of a pooled row, `[index_head_dim, head_dim]` F32, on a kv-source.
    pub wk: Option<Tensor>,
    /// Scale of the norm on an index key, `[index_head_dim]` F32, on a kv-source.
    pub k_norm: Option<Tensor>,
}

/// What a band layer keeps of the tokens it has seen.
#[derive(Clone, Debug, Default)]
pub struct BandState {
    /// Tokens seen and the window of key rows.
    pub window: LatentMqaState,
    /// Compressed rows built, on a kv-source.
    pub groups: usize,
    /// `groups` rows of `head_dim`, rotated and rounded to fp4, on a kv-source.
    pub compressed: Vec<f32>,
    /// `groups` index keys of `index_head_dim`, on a kv-source that is an index-source.
    pub index_keys: Vec<f32>,
    /// The input rows of the group still filling, `hidden` values each, on a kv-source.
    pub filling: Vec<f32>,
}

/// A value a layer left for the forward over the rows `first..first + rows`.
pub(super) struct Stamped<T> {
    first: usize,
    rows: usize,
    value: T,
}

/// What the sources of a band left, one value per forward over its rows.
#[derive(Default)]
pub(super) struct Published {
    pub(super) compressed: Vec<Stamped<Vec<f32>>>,
    pub(super) index_keys: Vec<Stamped<Vec<f32>>>,
    pub(super) picks: Vec<Stamped<Vec<i32>>>,
    pub(super) candidates: Vec<Stamped<Vec<bool>>>,
}

/// Where the layers of a band leave and find what their sources built.
///
/// A value is kept per forward, so the rows of a verify that attend 1 at a time each find
/// their own.
#[derive(Default)]
pub struct BandCell(Mutex<Published>);

impl BandCell {
    pub(super) fn with<R>(&self, f: impl FnOnce(&mut Published) -> R) -> R {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut held)
    }

    /// Opens the pass over the rows `first..first + rows`.
    ///
    /// The values left for a row outside them go.
    pub fn open(&self, first: usize, rows: usize) {
        self.with(|p| {
            keep_inside(&mut p.compressed, first, rows);
            keep_inside(&mut p.index_keys, first, rows);
            keep_inside(&mut p.picks, first, rows);
            keep_inside(&mut p.candidates, first, rows);
        });
    }

    /// Exchanges what this cell and `other` hold.
    ///
    /// A batch of sequences keeps 1 cell per sequence and brings it into the model's cell
    /// for the sequence's own rows.
    pub fn exchange(&self, other: &BandCell) {
        if std::ptr::eq(self, other) {
            return;
        }
        let mut mine = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut theirs = other.0.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::swap(&mut *mine, &mut *theirs);
    }
}

/// Drops the values of `slot` left for a forward that reaches outside `first..first + rows`.
fn keep_inside<T>(slot: &mut Vec<Stamped<T>>, first: usize, rows: usize) {
    slot.retain(|s| s.first >= first && s.first + s.rows <= first + rows);
}

/// Leaves `value` in `slot` for the forward over the rows `first..first + rows`.
///
/// It replaces what an earlier forward over the same rows left.
pub(super) fn put<T>(slot: &mut Vec<Stamped<T>>, (first, rows): (usize, usize), value: T) {
    slot.retain(|s| (s.first, s.rows) != (first, rows));
    slot.push(Stamped { first, rows, value });
}

/// The value `slot` holds for the forward over `first..first + rows`, or why there is none.
pub(super) fn read<T: Clone>(
    slot: &[Stamped<T>],
    (first, rows): (usize, usize),
    what: &str,
) -> Result<T> {
    match slot.iter().find(|s| s.first == first && s.rows == rows) {
        Some(s) => Ok(s.value.clone()),
        None => Err(crate::tensor::Error::msg(format!(
            "no {what} for the {rows} rows from position {first}: their source has not run"
        ))),
    }
}

/// A compressed band attention layer.
pub struct CompressedBand {
    /// The window mixer's weights and widths.
    pub(super) base: LatentMqa,
    pub(super) band: BandDims,
    compressor: Option<CompressorWeights>,
    indexer: Option<BandIndexerWeights>,
    pub(super) cell: Arc<BandCell>,
    pub state: Option<BandState>,
    /// Times the state was dropped or replaced.
    pub(super) moves: u64,
    /// The compressor and the indexer on a card, when the layer runs there.
    #[cfg(feature = "gpu")]
    card_band: Option<Arc<super::card_band::CardBand>>,
}

impl CompressedBand {
    /// Bytes of the weights every token reads, as held: the window mixer's projections, the
    /// indexer's query factor, head weights and key projection, and the pooling weights.
    ///
    /// The F32 weights count 4 bytes a value.
    /// A layer on a card counts what the card holds.
    pub fn resident_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        if self.card_band.is_some() {
            return self.card_bytes();
        }
        let dense = |t: &Tensor| t.elem_count() * std::mem::size_of::<f32>();
        let indexer = self.indexer.as_ref().map_or(0, |i| {
            i.wq_b.bytes() + dense(&i.weights_proj) + i.wk.as_ref().map_or(0, dense)
        });
        let compressor = self
            .compressor
            .as_ref()
            .map_or(0, |c| dense(&c.wkv) + c.wgate.as_ref().map_or(0, dense));
        self.base.resident_bytes() + indexer + compressor
    }

    /// The layer over the window mixer `base`, with the weights of its roles.
    pub fn new(
        base: LatentMqa,
        band: BandDims,
        compressor: Option<CompressorWeights>,
        indexer: Option<BandIndexerWeights>,
        cell: Arc<BandCell>,
    ) -> Result<Self> {
        let d = base.dims;
        let refuse = |what: &str| crate::tensor::Error::msg(format!("compressed band: {what}"));
        if band.ratio == 0 {
            return Err(refuse("a ratio of no token per row"));
        }
        if band.kv_source != compressor.is_some() || band.index_source != indexer.is_some() {
            return Err(refuse("its weights do not match its roles"));
        }
        if band.ratio > 1 && compressor.as_ref().is_some_and(|c| c.wgate.is_none()) {
            return Err(refuse("a kv-source above ratio 1 needs a pooling gate"));
        }
        if !d.head_dim.is_multiple_of(COMPRESSED_BLOCK) {
            return Err(refuse("a compressed row does not split into its blocks"));
        }
        if band.index_source
            && (band.index_topk == 0
                || band.index_heads == 0
                || band.index_head_dim < d.rope_dim.max(1)
                || !band.index_head_dim.is_multiple_of(INDEX_BLOCK))
        {
            return Err(refuse(
                "an index without a head, a kept row or whole blocks",
            ));
        }
        let keyless = |ix: &BandIndexerWeights| ix.wk.is_none() || ix.k_norm.is_none();
        if band.kv_source && indexer.as_ref().is_some_and(keyless) {
            return Err(refuse("an index-source that pools keeps index keys"));
        }
        if (band.candidate_source || band.uses_candidates) && band.candidate_block_size == 0 {
            return Err(refuse("candidate blocks of no row"));
        }
        Ok(Self {
            base,
            band,
            compressor,
            indexer,
            cell,
            state: None,
            moves: 0,
            #[cfg(feature = "gpu")]
            card_band: None,
        })
    }

    /// Runs the layer's projections and attention on the card `ordinal`, and its compressor,
    /// index keys and indexer on a decode step.
    ///
    /// The host weights and the host state stay.
    #[cfg(feature = "gpu")]
    pub fn on_card(&mut self, ordinal: usize) -> Result<()> {
        self.base.on_card(ordinal)?;
        let Some(card) = self.base.card.as_ref() else {
            return Err(crate::tensor::Error::msg("the layer is not on a card"));
        };
        self.card_band = Some(Arc::new(super::card_band::CardBand::new(
            card.dev(),
            self.compressor.as_ref(),
            self.indexer.as_ref(),
        )?));
        Ok(())
    }

    /// Bytes of weights the layer holds on its card.
    #[cfg(feature = "gpu")]
    pub fn card_bytes(&self) -> usize {
        self.base.card_bytes() + self.card_band.as_ref().map_or(0, |c| c.weight_bytes())
    }

    /// Tokens the layer's window ring on its card stands for, 0 before the first decode step.
    #[cfg(feature = "gpu")]
    pub fn card_ring_tokens(&self) -> usize {
        self.base.card_ring_tokens()
    }

    /// Tokens seen.
    pub fn len(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.window.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Values per row of what the layer keeps: a window key row, a compressed row and an
    /// index key.
    ///
    /// A layer that builds no compressed row or no index key keeps rows of 0 values of it.
    pub(super) fn state_widths(&self) -> (usize, usize, usize) {
        let row = self.base.dims.head_dim;
        let compressed = if self.band.kv_source { row } else { 0 };
        let index = if self.owns_keys() {
            self.band.index_head_dim
        } else {
            0
        };
        (row, compressed, index)
    }

    /// The layer keeps index keys of its own.
    pub(super) fn owns_keys(&self) -> bool {
        self.band.kv_source && self.band.index_source
    }

    /// The groups' slices for the rows of `x` `[1, seq, hidden]`, which stand at positions
    /// `index_pos..index_pos + seq`. A forward at position 0 starts the state again. Any other
    /// must continue it. The result has the shape `[1, seq, o_groups * o_lora_rank]` and `x`'s
    /// device and type.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        if b != 1 {
            return Err(crate::tensor::Error::msg(
                "compressed band attention keeps 1 sequence's rows",
            ));
        }
        if index_pos == 0 && self.state.is_some() {
            self.state = None;
            self.moves += 1;
        }
        if self.len() != index_pos {
            return Err(crate::tensor::Error::msg(format!(
                "compressed band attention holds {} tokens and cannot continue from position {index_pos}",
                self.len()
            )));
        }
        let (device, dtype) = (x.device(), x.dtype());
        let x2 = host_rows(x, seq, hidden)?;
        // A failed forward leaves no state, so the next one must start at position 0.
        let mut state = self.state.take().unwrap_or_default();
        let out = match seq == 1 && index_pos > 0 {
            true => self.decode(&x2, index_pos, &mut state)?,
            false => self.prefill(&x2, index_pos, &mut state)?,
        };
        self.state = Some(state);
        out.reshape((1, seq, self.base.dims.out_width()))?
            .to_device(&device)?
            .to_dtype(dtype)
    }

    /// The rows `x2` `[seq, hidden]` at positions `base..base + seq` at once.
    fn prefill(&self, x2: &Tensor, base: usize, state: &mut BandState) -> Result<Tensor> {
        let (seq, hidden) = x2.dims2()?;
        let m = &self.base;
        let (d, band) = (m.dims, self.band);
        let (hd, ratio) = (d.head_dim, band.ratio);
        let at = (base, seq);
        let (cos, sin) = phases(&m.inv_freq, base..base + seq);
        let front = m.front(x2, &cos, &sin, self.indexer.is_some())?;
        let qr = &front.qr;

        // The window: the kept rows, then this forward's. Row `old_rows + i` is query `i`'s own.
        let kv = &front.kv;
        let old_rows = state.window.window.len() / hd;
        let mut keys = std::mem::take(&mut state.window.window);
        keys.extend_from_slice(kv);
        let kept = (old_rows + seq).min(d.window) * hd;
        state.window.window = keys[keys.len() - kept..].to_vec();
        state.window.len += seq;

        // The groups this forward completes, from the rows still filling and this forward's.
        let held_groups = state.groups;
        let mut pooled_x = std::mem::take(&mut state.filling);
        pooled_x.extend_from_slice(&x2.flatten_all()?.to_vec1::<f32>()?);
        let groups = pooled_x.len() / hidden / ratio;
        let latent = match (&self.compressor, groups) {
            (Some(c), g) if g > 0 => Some(pool(c, &pooled_x, hidden, ratio, hd, d.eps)?),
            _ => None,
        };
        // A group's rows turn at the group's first position.
        let firsts = (held_groups..held_groups + groups).map(|g| g * ratio);
        let (gcos, gsin) = phases(&m.inv_freq, firsts);

        // The index keys of a layer that owns them: the held ones, then this forward's groups.
        let own_keys = match (&self.indexer, self.owns_keys()) {
            (Some(ix), true) => {
                let mut all = std::mem::take(&mut state.index_keys);
                if let Some(latent) = &latent {
                    all.extend(self.index_keys_of(ix, latent, groups, &gcos, &gsin, false)?);
                }
                let published = all.clone();
                self.cell.with(|p| put(&mut p.index_keys, at, published));
                Some(all)
            }
            _ => None,
        };
        let picks = match &self.indexer {
            Some(ix) => {
                let index_keys = match &own_keys {
                    Some(k) => k.clone(),
                    None => self.cell.with(|p| read(&p.index_keys, at, "index keys"))?,
                };
                let offset = old_rows + seq;
                self.pick_prefill(ix, x2, qr, &index_keys, base, offset, &cos, &sin)?
            }
            None => self.cell.with(|p| read(&p.picks, at, "compressed picks"))?,
        };

        // A kv-source's compressed rows: the held ones, then this forward's groups rotated and
        // rounded to fp4.
        let compressed = match &self.compressor {
            Some(_) => {
                let mut all = std::mem::take(&mut state.compressed);
                if let Some(latent) = &latent {
                    let mut fresh = latent.clone();
                    rotate_tails(&mut fresh, 1, hd, d.rope_dim, &gcos, &gsin);
                    fp4_round_trip(&mut fresh, COMPRESSED_BLOCK, false);
                    all.extend_from_slice(&fresh);
                }
                let published = all.clone();
                self.cell.with(|p| put(&mut p.compressed, at, published));
                state.compressed = all.clone();
                state.groups = held_groups + groups;
                if let Some(k) = own_keys {
                    state.index_keys = k;
                }
                state.filling = pooled_x.split_off(groups * ratio * hidden);
                all
            }
            None => self
                .cell
                .with(|p| read(&p.compressed, at, "compressed rows"))?,
        };

        // Per query, the window rows it sees, then its picked compressed rows.
        let (win, itopk) = (d.window, band.index_topk);
        if picks.len() != seq * itopk {
            return Err(crate::tensor::Error::msg(format!(
                "{} compressed picks for {seq} rows of {itopk}",
                picks.len()
            )));
        }
        let width = win + itopk;
        let mut slots = vec![-1i32; seq * width];
        for (i, row) in slots.chunks_exact_mut(width).enumerate() {
            let iq = old_rows + i;
            let lo = iq.saturating_sub(win - 1);
            for (k, slot) in row.iter_mut().take(win.min(old_rows + seq)).enumerate() {
                if lo + k <= iq {
                    *slot = (lo + k) as i32;
                }
            }
            row[win..].copy_from_slice(&picks[i * itopk..(i + 1) * itopk]);
        }
        keys.extend_from_slice(&compressed);
        #[cfg(feature = "gpu")]
        if let (Some(card), Some(cf)) = (&m.card, &front.card) {
            return card.attend_rows(m, cf, &keys, &slots, width, seq);
        }
        let q = m.queries(qr, &cos, &sin)?;
        let o = attend_slots(
            &q,
            &keys,
            &m.w.sink,
            &slots,
            width,
            (d.heads, hd),
            m.scale(),
        );
        m.project_out(o, &cos, &sin, seq)
    }

    /// The row `x2` `[1, hidden]` at position `pos`, after the rows the state holds.
    fn decode(&self, x2: &Tensor, pos: usize, state: &mut BandState) -> Result<Tensor> {
        #[cfg(feature = "gpu")]
        if let Some(card_band) = &self.card_band {
            return card_band.decode(self, x2, pos, state);
        }
        let hidden = x2.dim(1)?;
        let m = &self.base;
        let (d, band) = (m.dims, self.band);
        let (hd, ratio) = (d.head_dim, band.ratio);
        let at = (pos, 1);
        let (cos, sin) = phases(&m.inv_freq, pos..pos + 1);
        let front = m.front(x2, &cos, &sin, true)?;
        let qr = &front.qr;

        // The window after this row's key, oldest first.
        let ring = &mut state.window.window;
        ring.extend_from_slice(&front.kv);
        let excess = ring.len().saturating_sub(d.window * hd);
        ring.drain(..excess);
        let w = ring.len() / hd;
        state.window.len += 1;

        // A kv-source keeps the row, and pools the group the row completes.
        if let Some(c) = &self.compressor {
            state
                .filling
                .extend_from_slice(&x2.flatten_all()?.to_vec1::<f32>()?);
            if state.filling.len() == ratio * hidden {
                let g = state.groups;
                let latent = pool(c, &state.filling, hidden, ratio, hd, d.eps)?;
                let (gcos, gsin) = phases(&m.inv_freq, std::iter::once(g * ratio));
                let mut row = latent.clone();
                rotate_tails(&mut row, 1, hd, d.rope_dim, &gcos, &gsin);
                fp4_round_trip(&mut row, COMPRESSED_BLOCK, false);
                state.compressed.extend_from_slice(&row);
                if let (Some(ix), true) = (&self.indexer, self.owns_keys()) {
                    let k = self.index_keys_of(ix, &latent, 1, &gcos, &gsin, true)?;
                    state.index_keys.extend(k);
                }
                state.filling.clear();
                state.groups += 1;
            }
            let compressed = state.compressed.clone();
            self.cell.with(|p| put(&mut p.compressed, at, compressed));
            if self.owns_keys() {
                let index_keys = state.index_keys.clone();
                self.cell.with(|p| put(&mut p.index_keys, at, index_keys));
            }
        }

        let picks = match &self.indexer {
            Some(ix) => self.pick_decode(ix, x2, qr, pos, w, state)?,
            None => self.cell.with(|p| read(&p.picks, at, "compressed picks"))?,
        };
        let compressed = match &self.compressor {
            Some(_) => state.compressed.clone(),
            None => self
                .cell
                .with(|p| read(&p.compressed, at, "compressed rows"))?,
        };
        let itopk = band.index_topk;
        if picks.len() != itopk {
            return Err(crate::tensor::Error::msg(format!(
                "{} compressed picks for 1 row of {itopk}",
                picks.len()
            )));
        }

        // Every window row, then the picked compressed rows.
        let width = w + itopk;
        let mut slots: Vec<i32> = (0..w as i32).collect();
        slots.extend_from_slice(&picks);
        let mut keys = state.window.window.clone();
        keys.extend_from_slice(&compressed);
        let q = m.queries(qr, &cos, &sin)?;
        let o = attend_slots(
            &q,
            &keys,
            &m.w.sink,
            &slots,
            width,
            (d.heads, hd),
            m.scale(),
        );
        m.project_out(o, &cos, &sin, 1)
    }

    /// The index keys of `groups` pooled rows `latent` `[groups, head_dim]`: projection, norm,
    /// rotation by the groups' phases, fp4 round trip. A decode step takes the projection as a
    /// matrix product, a prefill as the dense product of the window mixer.
    #[allow(clippy::too_many_arguments)]
    fn index_keys_of(
        &self,
        ix: &BandIndexerWeights,
        latent: &[f32],
        groups: usize,
        cos: &[f32],
        sin: &[f32],
        decode: bool,
    ) -> Result<Vec<f32>> {
        let (d, band) = (self.base.dims, self.band);
        let missing = || crate::tensor::Error::msg("an index-source that pools keeps index keys");
        let wk = ix.wk.as_ref().ok_or_else(missing)?;
        let k_norm = ix.k_norm.as_ref().ok_or_else(missing)?;
        let latent = Tensor::from_vec(latent.to_vec(), (groups, d.head_dim), &Device::Cpu)?;
        let k = match decode {
            true => latent.matmul(&wk.t()?)?,
            false => linear(&latent, wk)?,
        };
        let k = rms_norm(&k, k_norm, d.eps)?;
        let mut k = k.flatten_all()?.to_vec1::<f32>()?;
        rotate_tails(&mut k, 1, band.index_head_dim, d.rope_dim, cos, sin);
        fp4_round_trip(&mut k, INDEX_BLOCK, true);
        Ok(k)
    }

    /// The fp4 index queries of the factor `qr` `[n, q_lora_rank]`, rotated by the phases of
    /// their rows: `[n, index_heads * index_head_dim]`.
    fn index_queries(
        &self,
        ix: &BandIndexerWeights,
        qr: &Tensor,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<f32>> {
        let band = self.band;
        let q = ix.wq_b.apply_here(qr)?;
        let mut q = q.flatten_all()?.to_vec1::<f32>()?;
        let (nh, ihd) = (band.index_heads, band.index_head_dim);
        rotate_tails(&mut q, nh, ihd, self.base.dims.rope_dim, cos, sin);
        fp4_round_trip(&mut q, INDEX_BLOCK, true);
        Ok(q)
    }

    /// The compressed rows each prefill query attends to, `[seq, index_topk]`, shifted by
    /// `offset`, `-1` for an empty slot. Query `i` reaches the first `(base + i + 1) / ratio`
    /// compressed rows. Leaves the picks, and on a candidate source its blocks, in the cell.
    #[allow(clippy::too_many_arguments)]
    fn pick_prefill(
        &self,
        ix: &BandIndexerWeights,
        x2: &Tensor,
        qr: &Tensor,
        index_keys: &[f32],
        base: usize,
        offset: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<i32>> {
        let seq = x2.dim(0)?;
        let band = self.band;
        let (nh, ihd, ratio, itopk) = (
            band.index_heads,
            band.index_head_dim,
            band.ratio,
            band.index_topk,
        );
        let at = (base, seq);
        let gtot = index_keys.len() / ihd;
        if gtot == 0 {
            let out = vec![-1i32; seq * itopk];
            let published = out.clone();
            self.cell.with(|p| put(&mut p.picks, at, published));
            return Ok(out);
        }
        let q = self.index_queries(ix, qr, cos, sin)?;
        let scale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
        let weights = linear(x2, &ix.weights_proj)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reach = |i: usize| (base + i + 1) / ratio;

        // The score of every reachable compressed row, `-inf` beyond.
        let mut scores = vec![f32::NEG_INFINITY; seq * gtot];
        for (i, row) in scores.chunks_exact_mut(gtot).enumerate() {
            let (qi, wi) = (
                &q[i * nh * ihd..(i + 1) * nh * ihd],
                &weights[i * nh..(i + 1) * nh],
            );
            for (t, s) in row.iter_mut().enumerate().take(reach(i).min(gtot)) {
                *s = index_score(qi, &index_keys[t * ihd..(t + 1) * ihd], wi, scale);
            }
        }
        let reaches: Vec<usize> = (0..seq).map(reach).collect();
        if band.candidate_source {
            let (blocks, size) = (band.candidate_blocks, band.candidate_block_size);
            let mask = select_candidate_blocks(&scores, seq, gtot, &reaches, blocks, size);
            self.cell.with(|p| put(&mut p.candidates, at, mask));
        } else if band.uses_candidates {
            let mask = self
                .cell
                .with(|p| read(&p.candidates, at, "candidate blocks"))?;
            scores = apply_candidate_mask(&scores, &mask);
        }

        // The best `index_topk` reachable rows by score, ties to the lower row, compacted.
        let topk = itopk.min(gtot);
        let mut out = vec![-1i32; seq * itopk];
        out.par_chunks_mut(itopk)
            .enumerate()
            .for_each(|(qi, slots)| {
                let row = &scores[qi * gtot..(qi + 1) * gtot];
                let mut order: Vec<usize> = (0..gtot).collect();
                let by =
                    |a: &usize, c: &usize| row[*c].partial_cmp(&row[*a]).unwrap().then(a.cmp(c));
                if topk < gtot {
                    order.select_nth_unstable_by(topk - 1, by);
                }
                let kept = &mut order[..topk];
                kept.sort_by(by);
                let reachable = kept.iter().filter(|&&t| t < reach(qi));
                for (slot, &t) in slots.iter_mut().zip(reachable) {
                    *slot = (t + offset) as i32;
                }
            });
        let published = out.clone();
        self.cell.with(|p| put(&mut p.picks, at, published));
        Ok(out)
    }

    /// The compressed rows the decode query at `pos` attends to, `[index_topk]`, shifted by
    /// `w_offset`, `-1` for an empty slot. Leaves the picks, and on a candidate source its
    /// blocks, in the cell.
    fn pick_decode(
        &self,
        ix: &BandIndexerWeights,
        x2: &Tensor,
        qr: &Tensor,
        pos: usize,
        w_offset: usize,
        state: &BandState,
    ) -> Result<Vec<i32>> {
        let band = self.band;
        let (nh, ihd, ratio, itopk) = (
            band.index_heads,
            band.index_head_dim,
            band.ratio,
            band.index_topk,
        );
        let at = (pos, 1);
        let index_keys = match self.owns_keys() {
            true => state.index_keys.clone(),
            false => self.cell.with(|p| read(&p.index_keys, at, "index keys"))?,
        };
        let gtot = index_keys.len() / ihd;
        let (cos, sin) = phases(&self.base.inv_freq, pos..pos + 1);
        let q = self.index_queries(ix, qr, &cos, &sin)?;
        let weights = x2.matmul(&ix.weights_proj.t()?)?;
        let weights = weights.flatten_all()?.to_vec1::<f32>()?;
        let scale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);

        let reach = ((pos + 1) / ratio).min(gtot);
        let mut scores = vec![f32::NEG_INFINITY; gtot];
        for (t, s) in scores.iter_mut().enumerate().take(reach) {
            *s = index_score(&q, &index_keys[t * ihd..(t + 1) * ihd], &weights, scale);
        }
        if band.candidate_source {
            let (blocks, size) = (band.candidate_blocks, band.candidate_block_size);
            let mask = select_candidate_blocks(&scores, 1, gtot, &[reach], blocks, size);
            self.cell.with(|p| put(&mut p.candidates, at, mask));
        } else if band.uses_candidates {
            let mask = self
                .cell
                .with(|p| read(&p.candidates, at, "candidate blocks"))?;
            scores = apply_candidate_mask(&scores, &mask);
        }

        // The best `index_topk` rows by score, in a stable order, the reachable ones compacted.
        let mut order: Vec<usize> = (0..gtot).collect();
        order.sort_by(|&a, &c| scores[c].partial_cmp(&scores[a]).unwrap());
        let mut out = vec![-1i32; itopk];
        let reachable = order.iter().take(itopk.min(gtot)).filter(|&&t| t < reach);
        for (slot, &t) in out.iter_mut().zip(reachable) {
            *slot = (t + w_offset) as i32;
        }
        let published = out.clone();
        self.cell.with(|p| put(&mut p.picks, at, published));
        Ok(out)
    }
}

/// `x` `[n, in]` through `w` `[out, in]` on this thread's cores.
fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    Projection::Dense(w.clone()).apply_here(x)
}

/// Pools every `ratio` rows of `rows` `[n, hidden]` into 1 row of `hd`. Per channel, a softmax
/// over the group's gate scores weighs the group's projected rows. Only whole groups are pooled.
/// The pooled rows are normalised. Returns `[n / ratio, hd]`, not rotated.
fn pool(
    c: &CompressorWeights,
    rows: &[f32],
    hidden: usize,
    ratio: usize,
    hd: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    let n = rows.len() / hidden;
    let groups = n / ratio;
    let x2 = Tensor::from_vec(rows.to_vec(), (n, hidden), &Device::Cpu)?;
    let kv = linear(&x2, &c.wkv)?.flatten_all()?.to_vec1::<f32>()?;
    let pooled = match &c.wgate {
        // Ratio 1: each row is its own group.
        None => kv,
        Some(wgate) => {
            let score = linear(&x2, wgate)?.flatten_all()?.to_vec1::<f32>()?;
            let mut pooled = vec![0f32; groups * hd];
            for (g, out) in pooled.chunks_exact_mut(hd).enumerate() {
                for (ch, o) in out.iter_mut().enumerate() {
                    let at = |r: usize| (g * ratio + r) * hd + ch;
                    let mut mx = f32::NEG_INFINITY;
                    for r in 0..ratio {
                        mx = mx.max(score[at(r)]);
                    }
                    let (mut den, mut acc) = (0f32, 0f32);
                    for r in 0..ratio {
                        let w = (score[at(r)] - mx).exp();
                        den += w;
                        acc += w * kv[at(r)];
                    }
                    *o = acc / den;
                }
            }
            pooled
        }
    };
    let pooled = Tensor::from_vec(pooled, (groups, hd), &Device::Cpu)?;
    rms_norm(&pooled, &c.norm, eps)?
        .flatten_all()?
        .to_vec1::<f32>()
}

/// The index score of 1 compressed row: over the index heads, the positive part of the head's
/// query `q` against the row's `key`, weighed by the head's weight and by `scale`.
fn index_score(q: &[f32], key: &[f32], weights: &[f32], scale: f32) -> f32 {
    let mut acc = 0f32;
    for (qh, &w) in q.chunks_exact(key.len()).zip(weights) {
        let mut dot = 0f32;
        for (a, k) in qh.iter().zip(key) {
            dot += a * k;
        }
        acc += dot.max(0.0) * w * scale;
    }
    acc
}

/// Every query head of `q` `[n, heads * hd]` against the rows of `keys` `[rows, hd]` its row's
/// `width` slots name, `-1` naming none. The rows are the values too. The sink logit of the head
/// takes softmax mass and adds no value. Returns `[n, heads * hd]`.
fn attend_slots(
    q: &[f32],
    keys: &[f32],
    sink: &[f32],
    slots: &[i32],
    width: usize,
    (heads, hd): (usize, usize),
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; q.len()];
    out.par_chunks_mut(hd).enumerate().for_each(|(pair, o)| {
        let (row, h) = (pair / heads, pair % heads);
        let qv = &q[pair * hd..(pair + 1) * hd];
        // The logits over the named rows and their running maximum, the sink included.
        let mut m = sink[h];
        let mut logits = Vec::with_capacity(width);
        for &idx in &slots[row * width..(row + 1) * width] {
            if idx < 0 {
                continue;
            }
            let key = &keys[idx as usize * hd..(idx as usize + 1) * hd];
            let l = qv.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
            if l > m {
                m = l;
            }
            logits.push((idx as usize, l));
        }
        // The softmax denominator with the sink, and the rows summed by their weights.
        let mut denom = (sink[h] - m).exp();
        for &(idx, l) in &logits {
            let w = (l - m).exp();
            denom += w;
            for (acc, &v) in o.iter_mut().zip(&keys[idx * hd..(idx + 1) * hd]) {
                *acc += w * v;
            }
        }
        for v in o.iter_mut() {
            *v /= denom;
        }
    });
    out
}

/// Rounds every value of `v` to what fp4 e2m1 holds under 1 scale per `block` values, and reads
/// it back. The scale puts the block's largest magnitude at the format's largest, rounded up to
/// a power of two with `pow2_scale`. `v.len()` is a multiple of `block`.
pub(crate) fn fp4_round_trip(v: &mut [f32], block: usize, pow2_scale: bool) {
    for seg in v.chunks_exact_mut(block) {
        let amax = seg.iter().fold(0f32, |m, &e| m.max(e.abs()));
        let mut scale = (amax / FP4_MAX).max(SCALE_FLOOR);
        if pow2_scale {
            scale = pow2_ceil(scale);
        }
        for e in seg.iter_mut() {
            let scaled = (*e / scale).clamp(-FP4_MAX, FP4_MAX);
            *e = e2m1_round(scaled.abs()).copysign(scaled) * scale;
        }
    }
}

/// The e2m1 magnitude nearest `m` (`m >= 0`), the lower one on a tie.
fn e2m1_round(m: f32) -> f32 {
    let mut best = FP4_GRID[0];
    let mut bestd = (m - FP4_GRID[0]).abs();
    for &g in &FP4_GRID[1..] {
        let d = (m - g).abs();
        if d < bestd {
            bestd = d;
            best = g;
        }
    }
    best
}

/// The first level of the candidate selection. `logits` is `[n_queries, width]`, a row a query
/// cannot reach at `-inf`. Query `q` reaches `compress_lens[q]` rows. Per query, each block of
/// `block_size` rows scores its best row, the block of the query's newest row scores `+inf`,
/// and the best `topk_blocks` blocks above `-inf` are kept. Returns the mask `[n_queries, width]`
/// of the rows inside a kept block.
pub fn select_candidate_blocks(
    logits: &[f32],
    n_queries: usize,
    width: usize,
    compress_lens: &[usize],
    topk_blocks: usize,
    block_size: usize,
) -> Vec<bool> {
    let num_blocks = width.div_ceil(block_size);
    let mut out = vec![false; n_queries * width];
    for q in 0..n_queries {
        let mut bscore = vec![f32::NEG_INFINITY; num_blocks];
        for (j, bs) in bscore.iter_mut().enumerate() {
            for p in j * block_size..((j + 1) * block_size).min(width) {
                *bs = bs.max(logits[q * width + p]);
            }
        }
        let last = (compress_lens[q].saturating_sub(1)) / block_size;
        if last < num_blocks {
            bscore[last] = f32::INFINITY;
        }
        let k = topk_blocks.min(num_blocks);
        let mut order: Vec<usize> = (0..num_blocks).collect();
        order.sort_by(|&a, &c| bscore[c].partial_cmp(&bscore[a]).unwrap());
        for &j in order.iter().take(k) {
            if bscore[j] > f32::NEG_INFINITY {
                for p in j * block_size..((j + 1) * block_size).min(width) {
                    out[q * width + p] = true;
                }
            }
        }
    }
    out
}

/// The second level: every score outside the mask of the first level set to `-inf`.
pub fn apply_candidate_mask(score: &[f32], mask: &[bool]) -> Vec<f32> {
    score
        .iter()
        .zip(mask)
        .map(|(&s, &keep)| if keep { s } else { f32::NEG_INFINITY })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::latent_mqa::{frequencies, LatentMqaDims, LatentMqaWeights};
    use super::super::mixer::Mixer;
    use super::*;
    use crate::inference::model::deepseek_v41::band::{
        fp4_act_quant, BandAttention, Compressor, Indexer, SharedAttn,
    };
    use crate::inference::model::deepseek_v41::cache::AttnCache;
    use crate::inference::model::deepseek_v41::candidates;
    use crate::inference::model::deepseek_v41::model::rope_table;

    const HIDDEN: usize = 48;
    const Q_RANK: usize = 24;
    const THETA: f32 = 10_000.0;
    const ORIGINAL: usize = 16;
    const FACTOR: f32 = 4.0;

    fn dims() -> LatentMqaDims {
        LatentMqaDims {
            heads: 4,
            head_dim: 64,
            rope_dim: 16,
            o_groups: 2,
            o_lora_rank: 12,
            window: 4,
            eps: 1e-6,
        }
    }

    fn band(kv: bool, index: bool, candidate_source: bool, uses_candidates: bool) -> BandDims {
        BandDims {
            ratio: 2,
            kv_source: kv,
            index_source: index,
            candidate_source,
            uses_candidates,
            index_topk: 3,
            index_heads: 2,
            index_head_dim: 32,
            candidate_blocks: 1,
            candidate_block_size: 2,
        }
    }

    /// A kv-source that owns its index keys and keeps candidate blocks, an index-source that
    /// scores inside those blocks, and a layer that reads what both left.
    fn roles() -> [BandDims; 3] {
        [
            band(true, true, true, false),
            band(false, true, false, true),
            band(false, false, false, false),
        ]
    }

    fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    fn mat(out: usize, inp: usize, seed: u64) -> Tensor {
        Tensor::from_vec(values(out * inp, seed, 0.4), (out, inp), &Device::Cpu).unwrap()
    }

    fn dense(out: usize, inp: usize, seed: u64) -> Projection {
        Projection::Dense(mat(out, inp, seed))
    }

    fn norm(n: usize, seed: u64) -> Tensor {
        let scale: Vec<f32> = values(n, seed, 0.5).iter().map(|v| 1.0 + v).collect();
        Tensor::from_vec(scale, n, &Device::Cpu).unwrap()
    }

    fn input(seq: usize, seed: u64) -> Tensor {
        Tensor::from_vec(
            values(seq * HIDDEN, seed, 1.0),
            (1, seq, HIDDEN),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn v41_layer(seed: u64, b: BandDims) -> BandAttention {
        let d = dims();
        let (hd, ihd) = (d.head_dim, b.index_head_dim);
        let compressor = b.kv_source.then(|| Compressor {
            norm: norm(hd, seed + 10),
            wkv: mat(hd, HIDDEN, seed + 11),
            wgate: Some(mat(hd, HIDDEN, seed + 12)),
            ratio: b.ratio,
            head_dim: hd,
            eps: d.eps,
        });
        let indexer = b.index_source.then(|| Indexer {
            wq_b: dense(b.index_heads * ihd, Q_RANK, seed + 13),
            weights_proj: mat(b.index_heads, HIDDEN, seed + 14),
            wk: b.kv_source.then(|| mat(ihd, hd, seed + 15)),
            k_norm: b.kv_source.then(|| norm(ihd, seed + 16)),
            n_heads: b.index_heads,
            index_head_dim: ihd,
            rope_head_dim: d.rope_dim,
            index_topk: b.index_topk,
            eps: d.eps,
            owns_k: b.kv_source,
            is_candidate_source: b.candidate_source,
            uses_candidates: b.uses_candidates,
            candidate_topk_blocks: b.candidate_blocks,
            candidate_block_size: b.candidate_block_size,
        });
        let sink = Tensor::from_vec(values(d.heads, seed + 5, 1.0), d.heads, &Device::Cpu).unwrap();
        BandAttention {
            wq_a: dense(Q_RANK, HIDDEN, seed),
            q_norm: norm(Q_RANK, seed + 1),
            wq_b: dense(d.heads * hd, Q_RANK, seed + 2),
            wkv: dense(hd, HIDDEN, seed + 3),
            kv_norm: norm(hd, seed + 4),
            attn_sink: sink,
            wo_a: dense(d.out_width(), d.heads * hd / d.o_groups, seed + 6),
            wo_b: dense(HIDDEN, d.out_width(), seed + 7),
            compressor,
            indexer,
            n_heads: d.heads,
            head_dim: hd,
            rope_head_dim: d.rope_dim,
            o_groups: d.o_groups,
            o_lora_rank: d.o_lora_rank,
            window_size: d.window,
            ratio: b.ratio,
            index_topk: b.index_topk,
            eps: d.eps,
        }
    }

    /// The window mixer and the role weights over the V4.1 layer's weights.
    fn parts(
        old: &BandAttention,
    ) -> (
        LatentMqa,
        Option<CompressorWeights>,
        Option<BandIndexerWeights>,
    ) {
        let weights = LatentMqaWeights {
            wq_a: old.wq_a.clone(),
            q_norm: old.q_norm.clone(),
            wq_b: old.wq_b.clone(),
            wkv: old.wkv.clone(),
            kv_norm: old.kv_norm.clone(),
            sink: flat(&old.attn_sink),
            wo_a: old.wo_a.clone(),
        };
        let freqs = frequencies(dims().rope_dim, ORIGINAL, THETA, FACTOR);
        let base = LatentMqa::new(dims(), weights, freqs).unwrap();
        let compressor = old.compressor.as_ref().map(|c| CompressorWeights {
            wkv: c.wkv.clone(),
            wgate: c.wgate.clone(),
            norm: c.norm.clone(),
        });
        let indexer = old.indexer.as_ref().map(|ix| BandIndexerWeights {
            wq_b: ix.wq_b.clone(),
            weights_proj: ix.weights_proj.clone(),
            wk: ix.wk.clone(),
            k_norm: ix.k_norm.clone(),
        });
        (base, compressor, indexer)
    }

    /// The V4.1 band and the mixers over the same weights, the mixers sharing 1 cell.
    fn pair() -> (Vec<BandAttention>, Vec<CompressedBand>) {
        let cell = Arc::new(BandCell::default());
        let olds: Vec<BandAttention> = roles()
            .iter()
            .zip(1u64..)
            .map(|(&b, i)| v41_layer(100 * i, b))
            .collect();
        let news = olds
            .iter()
            .zip(roles())
            .map(|(old, b)| {
                let (base, compressor, indexer) = parts(old);
                CompressedBand::new(base, b, compressor, indexer, cell.clone()).unwrap()
            })
            .collect();
        (olds, news)
    }

    fn caches() -> Vec<AttnCache> {
        (0..roles().len()).map(|_| AttnCache::default()).collect()
    }

    fn tables(n: usize) -> (Tensor, Tensor) {
        rope_table(dims().rope_dim, n, ORIGINAL, THETA, FACTOR).unwrap()
    }

    /// The mixer's rows through `wo_b` as the V4.1 band applies it, `[seq * hidden]`.
    fn full(old: &BandAttention, mixed: &Tensor) -> Vec<f32> {
        let (_, seq, width) = mixed.dims3().unwrap();
        flat(
            &old.wo_b
                .apply_here(&mixed.reshape((seq, width)).unwrap())
                .unwrap(),
        )
    }

    /// The rows from `base` through every layer of both, each layer on its own input, every
    /// layer's output compared.
    fn prompt(
        olds: &[BandAttention],
        news: &mut [CompressedBand],
        caches: &mut [AttnCache],
        (base, seq): (usize, usize),
        seed: u64,
    ) {
        let (cos, sin) = tables(base + seq);
        let mut shared = SharedAttn::default();
        let layers = olds.iter().zip(news.iter_mut()).zip(caches.iter_mut());
        for (i, ((old, new), cache)) in layers.enumerate() {
            let x = input(seq, seed + i as u64);
            cache.base = base;
            let want = old
                .forward_prefill_cached(&x, &cos, &sin, &mut shared, Some(cache))
                .unwrap();
            let got = new.forward(&x, base).unwrap();
            assert_eq!(
                bits(&full(old, &got)),
                bits(&flat(&want)),
                "layer {i} from {base}"
            );
        }
    }

    /// The token at `pos` through every layer of both, every layer's output compared.
    fn step(
        olds: &[BandAttention],
        news: &mut [CompressedBand],
        caches: &mut [AttnCache],
        pos: usize,
        seed: u64,
    ) {
        let (cos, sin) = tables(pos + 1);
        let mut shared = SharedAttn::default();
        let layers = olds.iter().zip(news.iter_mut()).zip(caches.iter_mut());
        for (i, ((old, new), cache)) in layers.enumerate() {
            let x = input(1, seed + i as u64);
            let want = old
                .forward_decode(&x, &cos, &sin, pos, cache, &mut shared)
                .unwrap();
            let got = new.forward(&x, pos).unwrap();
            assert_eq!(
                bits(&full(old, &got)),
                bits(&flat(&want)),
                "layer {i} at {pos}"
            );
        }
    }

    /// The kv-source's state equals its V4.1 cache, bit for bit.
    fn same_state(new: &CompressedBand, cache: &AttnCache, when: &str) {
        let s = new.state.as_ref().unwrap();
        let window: Vec<f32> = cache.win.iter().flatten().copied().collect();
        let filling: Vec<f32> = cache.group_x.iter().flatten().copied().collect();
        assert_eq!(bits(&s.window.window), bits(&window), "window {when}");
        assert_eq!(s.groups, cache.n_groups, "groups {when}");
        assert_eq!(
            bits(&s.compressed),
            bits(&cache.comp_kv),
            "compressed rows {when}"
        );
        assert_eq!(
            bits(&s.index_keys),
            bits(&cache.index_k),
            "index keys {when}"
        );
        assert_eq!(bits(&s.filling), bits(&filling), "filling group {when}");
    }

    #[test]
    fn a_prompt_through_the_band_gives_the_bits_of_the_v41_band() {
        let (olds, mut news) = pair();
        let mut caches = caches();
        // 5 compressed rows, more than the index keeps, and a group left filling.
        prompt(&olds, &mut news, &mut caches, (0, 11), 7);
        same_state(&news[0], &caches[0], "after the prompt");
    }

    #[test]
    fn decoding_through_the_band_gives_the_bits_of_the_v41_decode() {
        let (olds, mut news) = pair();
        let mut caches = caches();
        prompt(&olds, &mut news, &mut caches, (0, 5), 7);
        // 3 more groups complete, and the window turns over.
        for pos in 5..11 {
            step(&olds, &mut news, &mut caches, pos, 100 + 10 * pos as u64);
            same_state(&news[0], &caches[0], &format!("at {pos}"));
        }
    }

    #[test]
    fn a_prompt_that_continues_the_band_gives_the_bits_of_the_v41_continuation() {
        let (olds, mut news) = pair();
        let mut caches = caches();
        prompt(&olds, &mut news, &mut caches, (0, 5), 7);
        // The continuation completes the group the first prompt left filling.
        prompt(&olds, &mut news, &mut caches, (5, 6), 17);
        same_state(&news[0], &caches[0], "after the continuation");
    }

    #[test]
    fn a_restored_band_snapshot_decodes_the_same_bits() {
        let (_, news) = pair();
        let mut mixers: Vec<Mixer> = news.into_iter().map(Mixer::CompressedBand).collect();
        for (i, m) in mixers.iter_mut().enumerate() {
            m.forward(&input(5, 30 + i as u64), 0).unwrap();
        }
        let marks: Vec<_> = mixers
            .iter()
            .map(|m| m.snapshot().unwrap().unwrap())
            .collect();
        let run = |mixers: &mut Vec<Mixer>| -> Vec<Vec<u32>> {
            let rows = mixers.iter_mut().enumerate();
            rows.map(|(i, m)| bits(&flat(&m.forward(&input(1, 40 + i as u64), 5).unwrap())))
                .collect()
        };
        let first = run(&mut mixers);
        for (m, mark) in mixers.iter_mut().zip(&marks) {
            m.restore(mark).unwrap();
        }
        assert_eq!(run(&mut mixers), first);
    }

    #[test]
    fn a_reader_before_its_sources_is_refused_and_keeps_no_state() {
        let (olds, _) = pair();
        let (base, compressor, indexer) = parts(&olds[2]);
        let cell = Arc::new(BandCell::default());
        let mut reader = CompressedBand::new(base, roles()[2], compressor, indexer, cell).unwrap();
        assert!(reader.forward(&input(3, 1), 0).is_err());
        assert!(reader.state.is_none());
    }

    #[test]
    fn weights_that_do_not_match_the_roles_are_refused() {
        let (olds, _) = pair();
        let source = roles()[0];
        let cell = || Arc::new(BandCell::default());
        let (base, compressor, indexer) = parts(&olds[0]);
        assert!(CompressedBand::new(base, source, compressor, indexer, cell()).is_ok());
        let (base, _, indexer) = parts(&olds[0]);
        assert!(CompressedBand::new(base, source, None, indexer, cell()).is_err());
        let (base, compressor, indexer) = parts(&olds[0]);
        let gateless = compressor.map(|c| CompressorWeights { wgate: None, ..c });
        assert!(CompressedBand::new(base, source, gateless, indexer, cell()).is_err());
        let (base, compressor, indexer) = parts(&olds[0]);
        let narrow = BandDims {
            index_head_dim: 24,
            ..source
        };
        assert!(CompressedBand::new(base, narrow, compressor, indexer, cell()).is_err());
    }

    #[test]
    fn the_fp4_round_trip_gives_the_bits_of_the_v41_one() {
        // Blocks of ordinary, of large and of tiny magnitudes, then a block of zeros.
        let mut v = values(4 * INDEX_BLOCK, 3, 5.0);
        v.extend(values(INDEX_BLOCK, 4, 2.0e3));
        v.extend(values(INDEX_BLOCK, 5, 1.0e-5));
        v.extend(vec![0f32; INDEX_BLOCK]);
        let n = v.len();
        for (block, pow2) in [
            (INDEX_BLOCK, true),
            (COMPRESSED_BLOCK, false),
            (COMPRESSED_BLOCK, true),
        ] {
            let t = Tensor::from_vec(v.clone(), (n / block, block), &Device::Cpu).unwrap();
            let want = flat(&fp4_act_quant(&t, block, pow2).unwrap());
            let mut got = v.clone();
            fp4_round_trip(&mut got, block, pow2);
            assert_eq!(
                bits(&got),
                bits(&want),
                "blocks of {block}, power of two {pow2}"
            );
        }
    }

    #[test]
    fn a_magnitude_rounds_to_the_nearest_e2m1_value_the_lower_on_a_tie() {
        assert_eq!(e2m1_round(0.25), 0.0);
        assert_eq!(e2m1_round(2.6), 3.0);
        assert_eq!(e2m1_round(5.0), 4.0);
        assert_eq!(e2m1_round(FP4_MAX), FP4_MAX);
    }

    #[test]
    fn both_candidate_levels_give_the_bits_of_the_v41_ones() {
        let (n, width) = (4, 7);
        let lens = [0, 2, 5, 7];
        let mut logits = values(n * width, 9, 2.0);
        for (q, &len) in lens.iter().enumerate() {
            for l in &mut logits[q * width + len..(q + 1) * width] {
                *l = f32::NEG_INFINITY;
            }
        }
        let scores = values(n * width, 10, 1.0);
        for (blocks, size) in [(1, 2), (2, 3), (5, 2)] {
            let want = candidates::select_candidate_blocks(&logits, n, width, &lens, blocks, size);
            let got = select_candidate_blocks(&logits, n, width, &lens, blocks, size);
            assert_eq!(got, want, "{blocks} blocks of {size}");
            // The block of each query's newest reachable row is kept.
            for (q, &len) in lens.iter().enumerate().filter(|&(_, &len)| len > 0) {
                assert!(
                    got[q * width + len - 1],
                    "query {q}, {blocks} blocks of {size}"
                );
            }
            let masked = candidates::apply_candidate_mask(&scores, &want);
            assert_eq!(bits(&apply_candidate_mask(&scores, &got)), bits(&masked));
        }
    }
}
