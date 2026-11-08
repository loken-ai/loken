//! The decode step of a compressed band layer on a card.
//!
//! From the input row to the output row, the layer runs there:
//! - the pooling of the group a row completes, its norm, rotation and fp4 round trip,
//! - the index key of the new compressed row,
//! - the index queries, the scores over the index keys, the candidate blocks and the top-k,
//! - the attention over the window ring and the picked compressed rows.
//!
//! The host state stays the source of truth for every snapshot, restore and rewind.
//! The card keeps a copy of the rows it needs, stamped with the restart it was built under.
//! A copy that does not match the host state is rebuilt from it.
//!
//! The rows the host state keeps come back with the output row in one download.
//! The picks and the candidate blocks come back with them, for the layers that read them from
//! the cell of the band.

use super::card_mixers::{sync_rows, vector, Arenas, CardFront, CardLatent};
use super::card_packed::{Choices, PackedDev};
use super::compressed_band::{
    put, read, BandIndexerWeights, BandState, CompressedBand, CompressorWeights, COMPRESSED_BLOCK,
    INDEX_BLOCK,
};
use super::latent_mqa::phases;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::cuda::{
    alloc_f32, alloc_i32, apply_candidate_mask, candidate_mask, compress_pool, copy_f32, fp4_round,
    index_scores_dev, pack_download, rms_norm_f32, rope_tokens, topk_select, upload_f32,
    upload_i32, CudaDevice, GrowDev,
};
use crate::tensor::gpu_runtime::driver::{CudaSlice, CudaView};
use crate::tensor::{Device, Error, Result, Tensor};
use std::sync::{Arc, Mutex, MutexGuard};

/// A kv-source's pooling weights on a card.
struct CardCompressor {
    wkv: PackedDev,
    wgate: Option<PackedDev>,
    norm: CudaSlice<f32>,
}

/// An index-source's weights on a card.
struct CardIndexer {
    wq_b: PackedDev,
    weights: PackedDev,
    wk: Option<PackedDev>,
    k_norm: Option<CudaSlice<f32>>,
}

/// The projections of the rows of the group still filling, one row of `head_dim` per input row.
#[derive(Default)]
struct Fill {
    /// The restart and the rows the buffers hold.
    stamp: Option<(u64, usize)>,
    kv: Option<CudaSlice<f32>>,
    score: Option<CudaSlice<f32>>,
}

/// What the card keeps beside the ring and the compressed rows of the layer.
#[derive(Default)]
struct BandArenas {
    /// The index keys, rows of `index_head_dim`.
    keys: Option<GrowDev>,
    keys_moves: u64,
    fill: Fill,
}

/// A band layer's compressor and indexer on a card.
pub(super) struct CardBand {
    compressor: Option<CardCompressor>,
    indexer: Option<CardIndexer>,
    arenas: Mutex<BandArenas>,
}

/// The compressed row a decode step built, and the index key that goes with it.
struct Built {
    row: CudaSlice<f32>,
    key: Option<CudaSlice<f32>>,
}

/// What the indexer left of a decode step.
struct Picked {
    /// The slots of the compressed rows the query reaches, shifted by the window's length.
    slots: CudaSlice<i32>,
    /// The same slots as floats, for the host.
    floats: Option<CudaSlice<f32>>,
    /// The candidate blocks as 0 and 1, on a candidate source.
    mask: Option<CudaSlice<f32>>,
}

/// Writes `src` into `dst` from value `at`.
fn write_at(
    dev: &CudaDevice,
    dst: &mut CudaSlice<f32>,
    at: usize,
    src: &CudaView<'_, f32>,
) -> Result<()> {
    dev.stream()
        .memcpy_dtod(src, &mut dst.slice_mut(at..at + src.len()))
        .map_err(|e| Error::msg(format!("group rows copy: {e}")))
}

impl CardBand {
    /// The weights of a band layer on `dev`.
    pub(super) fn new(
        dev: &Arc<CudaDevice>,
        compressor: Option<&CompressorWeights>,
        indexer: Option<&BandIndexerWeights>,
        choices: &Choices,
    ) -> Result<Self> {
        let dense = |t: &Tensor| PackedDev::new(dev, &Projection::Dense(t.clone()), choices);
        let compressor = compressor
            .map(|c| -> Result<CardCompressor> {
                Ok(CardCompressor {
                    wkv: dense(&c.wkv)?,
                    wgate: c.wgate.as_ref().map(dense).transpose()?,
                    norm: vector(dev, &c.norm)?,
                })
            })
            .transpose()?;
        let indexer = indexer
            .map(|i| -> Result<CardIndexer> {
                Ok(CardIndexer {
                    wq_b: PackedDev::new(dev, &i.wq_b, choices)?,
                    weights: dense(&i.weights_proj)?,
                    wk: i.wk.as_ref().map(dense).transpose()?,
                    k_norm: i.k_norm.as_ref().map(|t| vector(dev, t)).transpose()?,
                })
            })
            .transpose()?;
        Ok(Self {
            compressor,
            indexer,
            arenas: Mutex::default(),
        })
    }

    fn arenas(&self) -> MutexGuard<'_, BandArenas> {
        self.arenas.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Bytes of weights held on the card.
    pub(super) fn weight_bytes(&self) -> usize {
        let floats = |v: &CudaSlice<f32>| v.len() * std::mem::size_of::<f32>();
        let compressor = self.compressor.as_ref().map_or(0, |c| {
            c.wkv.bytes() + c.wgate.as_ref().map_or(0, PackedDev::bytes) + floats(&c.norm)
        });
        let indexer = self.indexer.as_ref().map_or(0, |i| {
            i.wq_b.bytes()
                + i.weights.bytes()
                + i.wk.as_ref().map_or(0, PackedDev::bytes)
                + i.k_norm.as_ref().map_or(0, floats)
        });
        compressor + indexer
    }

    /// Pools the group the row of `f` completes, if it does.
    /// Returns the compressed row and the index key it gives, both appended to the buffers of the
    /// card.
    ///
    /// The projections of the rows of the group are kept as they come, so a step projects its
    /// own row alone.
    #[allow(clippy::too_many_arguments)]
    fn pool(
        &self,
        b: &CompressedBand,
        card: &CardLatent,
        arenas: &mut Arenas,
        ba: &mut BandArenas,
        f: &CardFront,
        state: &BandState,
        hidden: usize,
    ) -> Result<Option<Built>> {
        let Some(c) = &self.compressor else {
            return Ok(None);
        };
        let dev: &CudaDevice = card.dev();
        let m = &b.base;
        let (d, band) = (m.dims, b.band);
        let (hd, ratio, moves) = (d.head_dim, band.ratio, b.moves);
        let held = state.filling.len() / hidden;
        let rows = ratio * hd;
        let Fill { stamp, kv, score } = &mut ba.fill;

        // The buffers hold the projections of the rows the host state still fills with.
        if *stamp != Some((moves, held)) || kv.is_none() || score.is_none() {
            let mut latents = alloc_f32(dev, "group latents", rows)?;
            let mut scores = alloc_f32(dev, "group scores", rows)?;
            if held > 0 {
                let filling = upload_f32(dev, "group rows", &state.filling)?;
                let k = c.wkv.apply(&filling, held)?;
                write_at(dev, &mut latents, 0, &k.slice(0..held * hd))?;
                if let Some(g) = &c.wgate {
                    let s = g.apply(&filling, held)?;
                    write_at(dev, &mut scores, 0, &s.slice(0..held * hd))?;
                }
            }
            *kv = Some(latents);
            *score = Some(scores);
        }
        let (Some(latents), Some(scores)) = (kv.as_mut(), score.as_mut()) else {
            return Err(Error::msg("the group buffers are not on the card"));
        };
        let row = c.wkv.apply(&f.x, 1)?;
        write_at(dev, latents, held * hd, &row.slice(0..hd))?;
        if let Some(g) = &c.wgate {
            let s = g.apply(&f.x, 1)?;
            write_at(dev, scores, held * hd, &s.slice(0..hd))?;
        }
        if held + 1 < ratio {
            *stamp = Some((moves, held + 1));
            return Ok(None);
        }
        *stamp = Some((moves, 0));

        let pooled = match &c.wgate {
            Some(_) => compress_pool(
                dev,
                &latents.slice(0..rows),
                &scores.slice(0..rows),
                ratio,
                hd,
            )?,
            None => copy_f32(dev, "group latent", &row.slice(0..hd))?,
        };
        // A group's row turns at the group's first position.
        let half = d.rope_dim / 2;
        let (gcos, gsin) = phases(&m.inv_freq, std::iter::once(state.groups * ratio));
        let tables = upload_f32(dev, "group phases", &[gcos, gsin].concat())?;
        let (gc, gs) = (tables.slice(0..half), tables.slice(half..2 * half));

        let latent = rms_norm_f32(dev, &pooled, &c.norm, 1, hd, d.eps)?;
        let mut out = copy_f32(dev, "compressed row", &latent.slice(0..hd))?;
        rope_tokens(dev, &mut out, (1, 1, hd, d.rope_dim), &gc, &gs)?;
        fp4_round(dev, &mut out, COMPRESSED_BLOCK, false)?;
        let Some(comp) = arenas.comp.as_mut() else {
            return Err(Error::msg("the compressed rows are not on the card"));
        };
        comp.append(dev, &out.slice(0..hd))?;

        let key = match (&self.indexer, b.owns_keys()) {
            (Some(ix), true) => {
                let (Some(wk), Some(k_norm)) = (&ix.wk, &ix.k_norm) else {
                    return Err(Error::msg("an index-source that pools keeps index keys"));
                };
                let ihd = band.index_head_dim;
                let k = wk.apply(&latent, 1)?;
                let mut k = rms_norm_f32(dev, &k, k_norm, 1, ihd, d.eps)?;
                rope_tokens(dev, &mut k, (1, 1, ihd, d.rope_dim), &gc, &gs)?;
                fp4_round(dev, &mut k, INDEX_BLOCK, true)?;
                let Some(keys) = ba.keys.as_mut() else {
                    return Err(Error::msg("the index keys are not on the card"));
                };
                keys.append(dev, &k.slice(0..ihd))?;
                Some(k)
            }
            _ => None,
        };
        Ok(Some(Built { row: out, key }))
    }

    /// The compressed rows the query at `pos` picks, by the index keys on the card.
    /// `w` is the window's length, which the slots are shifted by.
    fn pick(
        &self,
        b: &CompressedBand,
        card: &CardLatent,
        ba: &BandArenas,
        f: &CardFront,
        (pos, w): (usize, usize),
    ) -> Result<Picked> {
        let Some(ix) = &self.indexer else {
            return Err(Error::msg("the layer has no indexer"));
        };
        let dev: &CudaDevice = card.dev();
        let (d, band) = (b.base.dims, b.band);
        let (nh, ihd, ratio, itopk) = (
            band.index_heads,
            band.index_head_dim,
            band.ratio,
            band.index_topk,
        );
        let gtot = ba.keys.as_ref().map_or(0, |k| k.len());
        let Some(keys) = ba.keys.as_ref().filter(|_| gtot > 0) else {
            return Ok(Picked {
                slots: upload_i32(dev, "picks", &vec![-1; itopk])?,
                floats: Some(upload_f32(dev, "picks", &vec![-1.0; itopk])?),
                mask: None,
            });
        };
        let mut q = ix.wq_b.apply(&f.qr, 1)?;
        rope_tokens(dev, &mut q, (1, nh, ihd, d.rope_dim), &f.cos(), &f.sin())?;
        fp4_round(dev, &mut q, INDEX_BLOCK, true)?;
        let weights = ix.weights.apply(&f.x, 1)?;
        let scale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
        let mut scores = index_scores_dev(
            dev,
            &q,
            &keys.view(),
            &weights.slice(0..nh),
            (nh, ihd, gtot),
            ratio,
            scale,
            pos,
        )?;
        let reach = ((pos + 1) / ratio).min(gtot);
        let mut mask = None;
        if band.candidate_source {
            let (blocks, size) = (band.candidate_blocks, band.candidate_block_size);
            mask = Some(candidate_mask(dev, &scores, reach, size, blocks)?);
        } else if band.uses_candidates {
            let kept = b
                .cell
                .with(|p| read(&p.candidates, (pos, 1), "candidate blocks"))?;
            if kept.len() != gtot {
                return Err(Error::msg(format!(
                    "{} candidate flags for {gtot} compressed rows",
                    kept.len()
                )));
            }
            let flags: Vec<f32> = kept.iter().map(|&k| f32::from(u8::from(k))).collect();
            let up = upload_f32(dev, "candidate mask", &flags)?;
            apply_candidate_mask(dev, &mut scores, &up)?;
        }
        let (slots, floats) = topk_select(dev, &scores, reach, itopk, w)?;
        Ok(Picked {
            slots,
            floats: Some(floats),
            mask,
        })
    }

    /// The decode step of the layer `b` for the row `x2` `[1, hidden]` at position `pos`: the
    /// output row `[1, o_groups * o_lora_rank]`.
    ///
    /// The host state advances as the host decode advances it, from the rows that come back.
    pub(super) fn decode(
        &self,
        b: &CompressedBand,
        x2: &Tensor,
        pos: usize,
        state: &mut BandState,
    ) -> Result<Tensor> {
        let m = &b.base;
        let Some(card) = m.card.clone() else {
            return Err(Error::msg("the layer is not on a card"));
        };
        let dev: &CudaDevice = card.dev();
        let (d, band) = (m.dims, b.band);
        let (hd, ihd, itopk, moves) = (d.head_dim, band.index_head_dim, band.index_topk, b.moves);
        let hidden = x2.dim(1)?;
        let at = (pos, 1);
        let owns = b.owns_keys();
        let (cos, sin) = phases(&m.inv_freq, pos..pos + 1);
        let f = card.front_dev(m, x2, &cos, &sin)?;

        let mut arenas = card.arenas();
        let mut ba = self.arenas();
        arenas.sync_ring(
            dev,
            (hd, d.window),
            (moves, state.window.len + 1),
            &state.window.window,
            &f.kv,
        )?;
        let w = arenas.ring.as_ref().map_or(0, |r| r.len());

        // The compressed rows, and the index keys of a layer that scores.
        let built = if self.compressor.is_some() {
            arenas.sync_comp(dev, hd, moves, &state.compressed)?;
            if owns {
                let ba = &mut *ba;
                let keys = &state.index_keys;
                sync_rows(&mut ba.keys, &mut ba.keys_moves, dev, ihd, moves, keys)?;
            }
            self.pool(b, &card, &mut arenas, &mut ba, &f, state, hidden)?
        } else {
            let flat = b
                .cell
                .with(|p| read(&p.compressed, at, "compressed rows"))?;
            arenas.sync_comp(dev, hd, moves, &flat)?;
            if self.indexer.is_some() {
                let keys = b.cell.with(|p| read(&p.index_keys, at, "index keys"))?;
                let ba = &mut *ba;
                sync_rows(&mut ba.keys, &mut ba.keys_moves, dev, ihd, moves, &keys)?;
            }
            None
        };

        let picked = match &self.indexer {
            Some(_) => self.pick(b, &card, &ba, &f, (pos, w))?,
            None => {
                let picks = b.cell.with(|p| read(&p.picks, at, "compressed picks"))?;
                if picks.len() != itopk {
                    return Err(Error::msg(format!(
                        "{} compressed picks for 1 row of {itopk}",
                        picks.len()
                    )));
                }
                Picked {
                    slots: upload_i32(dev, "picks", &picks)?,
                    floats: None,
                    mask: None,
                }
            }
        };

        // Every window row, then the picked compressed rows.
        let width = w + itopk;
        let mut idx = alloc_i32(dev, "mixer slots", width)?;
        if w > 0 {
            dev.stream()
                .memcpy_dtod(&card.iota().slice(0..w), &mut idx.slice_mut(0..w))
                .map_err(|e| Error::msg(format!("window slots copy: {e}")))?;
        }
        if itopk > 0 {
            dev.stream()
                .memcpy_dtod(&picked.slots.slice(0..itopk), &mut idx.slice_mut(w..width))
                .map_err(|e| Error::msg(format!("picked slots copy: {e}")))?;
        }
        let wide = card.attend_decode(m, &f, &arenas, &idx, width)?;
        let gtot = ba.keys.as_ref().map_or(0, |k| k.len());
        drop(arenas);
        drop(ba);

        // One download: the output row, the key row, the rows built, the blocks and the picks.
        let out_w = d.out_width();
        let mut parts: Vec<CudaView<'_, f32>> = vec![wide.slice(0..out_w), f.kv.slice(0..hd)];
        if let Some(bt) = &built {
            parts.push(bt.row.slice(0..hd));
            if let Some(k) = &bt.key {
                parts.push(k.slice(0..ihd));
            }
        }
        if let Some(mask) = &picked.mask {
            parts.push(mask.slice(0..gtot));
        }
        if let Some(floats) = &picked.floats {
            parts.push(floats.slice(0..itopk));
        }
        let host = pack_download(dev, &parts)?;
        let mut cursor = 0;
        let mut take = |n: usize| {
            cursor += n;
            &host[cursor - n..cursor]
        };

        let out = take(out_w).to_vec();
        let ring = &mut state.window.window;
        ring.extend_from_slice(take(hd));
        let excess = ring.len().saturating_sub(d.window * hd);
        ring.drain(..excess);
        state.window.len += 1;

        if self.compressor.is_some() {
            let row = x2.flatten_all()?.to_vec1::<f32>()?;
            state.filling.extend_from_slice(&row);
            if let Some(bt) = &built {
                state.compressed.extend_from_slice(take(hd));
                if bt.key.is_some() {
                    state.index_keys.extend_from_slice(take(ihd));
                }
                state.filling.clear();
                state.groups += 1;
            }
            let compressed = state.compressed.clone();
            b.cell.with(|p| put(&mut p.compressed, at, compressed));
            if owns {
                let keys = state.index_keys.clone();
                b.cell.with(|p| put(&mut p.index_keys, at, keys));
            }
        }
        if self.indexer.is_some() {
            if band.candidate_source {
                let mask: Vec<bool> = match picked.mask {
                    Some(_) => take(gtot).iter().map(|&v| v != 0.0).collect(),
                    None => Vec::new(),
                };
                b.cell.with(|p| put(&mut p.candidates, at, mask));
            }
            let picks: Vec<i32> = take(itopk).iter().map(|&v| v as i32).collect();
            b.cell.with(|p| put(&mut p.picks, at, picks));
        }
        Tensor::from_vec(out, (1, out_w), &Device::Cpu)
    }
}
