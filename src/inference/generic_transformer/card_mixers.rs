//! The latent attention and compressed band mixers on a card.
//!
//! A mixer keeps its host weights and its host state, which stay the oracle and the source of
//! every snapshot, restore and rewind.
//! On a card it also holds its projections in the form the file stores them, see `card_packed.rs`,
//! and runs these steps there:
//! - the query factor, the queries and the key row, with their norms, rotation and fp8 round trip,
//! - the attention over the window rows and the picked compressed rows, with the sink,
//! - the turn back of the heads' outputs and the grouped output projection.
//!
//! A band layer also runs its compressor, index keys and indexer there on a decode step, see
//! `card_band.rs`.
//!
//! A decode step crosses the bus twice: the input row goes up, and the output row comes back
//! with the rows the host state keeps.
//!
//! The window lives in a ring and the compressed rows in a growing buffer, both in device memory.
//! Each is flat rows that the gather kernel reads by slot, as the decode stream reads them.
//! A prefill gathers from the rows it uploads and leaves the buffers to the next decode.
//!
//! A buffer holds the state it was stamped with, and a decode that finds another stamp rebuilds
//! it from the host state.
//! A restore, a rewind or a restart changes the stamp, so a buffer never serves a state it does
//! not hold.

use super::card_packed::{Choices, PackedDev};
use super::latent_mqa::{phases, Front, LatentMqa, LatentMqaState, FP8_BLOCK};
use crate::tensor::cuda::{
    alloc_f32, fp8_round, gather_ring, pack_download, put_cols, rms_norm_f32, rope_tokens,
    slots_attention, sparse_attention_dev, take_cols, upload_f32, upload_i32, CudaDevice, GrowDev,
    RingDev,
};
use crate::tensor::gpu_runtime::driver::{CudaSlice, CudaView};
use crate::tensor::{DType, Device, Error, Result, Tensor};
use std::sync::{Arc, Mutex, MutexGuard};

/// What the device holds of the query side and key rows of one forward.
pub(super) struct CardFront {
    /// The input rows `[seq, hidden]`, followed by the phases tables.
    /// The tail is not part of the input.
    pub(super) x: CudaSlice<f32>,
    /// The normalised query factor `[seq, q_lora_rank]`.
    pub(super) qr: CudaSlice<f32>,
    /// The rotated queries `[seq, heads * head_dim]`.
    pub(super) q: CudaSlice<f32>,
    /// The rotated key rows, rounded to the fp8 grid, `[seq, head_dim]`.
    pub(super) kv: CudaSlice<f32>,
    /// Where the 3 tables of the rows' phases start in `x`: the cosines, the sines, then the
    /// sines turned back, each `table` values.
    tables: usize,
    table: usize,
}

impl CardFront {
    pub(super) fn cos(&self) -> CudaView<'_, f32> {
        self.x.slice(self.tables..self.tables + self.table)
    }

    pub(super) fn sin(&self) -> CudaView<'_, f32> {
        let at = self.tables + self.table;
        self.x.slice(at..at + self.table)
    }

    fn back(&self) -> CudaView<'_, f32> {
        let at = self.tables + 2 * self.table;
        self.x.slice(at..at + self.table)
    }
}

/// Makes `slot` hold `flat`, rows of `width`, which stand for the restart `moves`.
///
/// Rows only grow within one restart, so a buffer of that restart takes the rows it lacks.
/// A buffer of another restart, or longer than `flat`, is rebuilt.
pub(super) fn sync_rows(
    slot: &mut Option<GrowDev>,
    stamp: &mut u64,
    dev: &CudaDevice,
    width: usize,
    moves: u64,
    flat: &[f32],
) -> Result<()> {
    let rows = flat.len() / width;
    let grows = match slot {
        Some(c) => *stamp == moves && c.len() <= rows,
        None => false,
    };
    if !grows {
        *slot = Some(GrowDev::open(dev, flat, width)?);
        *stamp = moves;
        return Ok(());
    }
    if let Some(c) = slot.as_mut() {
        for row in flat.chunks_exact(width).skip(c.len()) {
            let up = upload_f32(dev, "device row", row)?;
            c.append(dev, &up.slice(0..width))?;
        }
    }
    Ok(())
}

/// The ring and the compressed rows on the device, with the state they were stamped with.
#[derive(Default)]
pub(super) struct Arenas {
    pub(super) ring: Option<RingDev>,
    /// The restarts and the tokens the ring holds.
    ring_stamp: Option<(u64, usize)>,
    pub(super) comp: Option<GrowDev>,
    /// The restarts the compressed rows were built under.
    comp_moves: u64,
}

impl Arenas {
    /// Makes the ring hold the window after a row, which stands for `tokens` tokens seen.
    ///
    /// `before` is the host window without the row, `newest` the row on the card.
    /// A ring stamped with the tokens before takes the row alone.
    /// Any other ring is rebuilt from `before` and the row read back.
    pub(super) fn sync_ring(
        &mut self,
        dev: &CudaDevice,
        (hd, cap): (usize, usize),
        (moves, tokens): (u64, usize),
        before: &[f32],
        newest: &CudaSlice<f32>,
    ) -> Result<()> {
        let current = self.ring_stamp == Some((moves, tokens.saturating_sub(1)));
        self.ring_stamp = None;
        match (&mut self.ring, current) {
            (Some(ring), true) => ring.append(dev, &newest.slice(0..hd))?,
            _ => {
                let mut all = before.to_vec();
                all.extend(download(dev, newest, hd)?);
                let kept = all.len().min(cap * hd);
                self.ring = Some(RingDev::open(dev, &all[all.len() - kept..], cap, hd)?);
            }
        }
        self.ring_stamp = Some((moves, tokens));
        Ok(())
    }

    /// Makes the compressed rows hold `flat`, rows of `hd`.
    pub(super) fn sync_comp(
        &mut self,
        dev: &CudaDevice,
        hd: usize,
        moves: u64,
        flat: &[f32],
    ) -> Result<()> {
        sync_rows(&mut self.comp, &mut self.comp_moves, dev, hd, moves, flat)
    }
}

/// A latent attention layer's weights and buffers on a card.
pub(super) struct CardLatent {
    dev: Arc<CudaDevice>,
    wq_a: PackedDev,
    q_norm: CudaSlice<f32>,
    wq_b: PackedDev,
    wkv: PackedDev,
    kv_norm: CudaSlice<f32>,
    sink: CudaSlice<f32>,
    /// The groups of the output projection.
    wo: Vec<PackedDev>,
    /// The slots `0..window`, which a decode step names to attend over the whole window.
    iota: CudaSlice<i32>,
    arenas: Mutex<Arenas>,
}

/// `n` values of `s` read back.
pub(super) fn download(dev: &CudaDevice, s: &CudaSlice<f32>, n: usize) -> Result<Vec<f32>> {
    let mut host = dev
        .stream()
        .clone_dtoh(s)
        .map_err(|e| Error::msg(format!("card mixer download: {e}")))?;
    host.truncate(n);
    Ok(host)
}

/// A norm scale or another vector, uploaded as F32.
pub(super) fn vector(dev: &CudaDevice, t: &Tensor) -> Result<CudaSlice<f32>> {
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    upload_f32(dev, "mixer vector", &v)
}

/// The key rows `[seq, window]` each query sees among the rows of a buffer that holds `old` kept
/// rows and then the `seq` new ones, `-1` for an empty slot.
pub(super) fn window_slots(old: usize, seq: usize, window: usize) -> Vec<i32> {
    let mut slots = vec![-1i32; seq * window];
    for (i, row) in slots.chunks_exact_mut(window).enumerate() {
        let iq = old + i;
        let lo = iq.saturating_sub(window - 1);
        for (k, slot) in row.iter_mut().enumerate() {
            if lo + k <= iq {
                *slot = (lo + k) as i32;
            }
        }
    }
    slots
}

impl CardLatent {
    /// The layer `m` on `dev`, its block weights in their units of `choices`.
    pub(super) fn new(dev: Arc<CudaDevice>, m: &LatentMqa, choices: &Choices) -> Result<Self> {
        Ok(Self {
            wq_a: PackedDev::new(&dev, &m.w.wq_a, choices)?,
            q_norm: vector(&dev, &m.w.q_norm)?,
            wq_b: PackedDev::new(&dev, &m.w.wq_b, choices)?,
            wkv: PackedDev::new(&dev, &m.w.wkv, choices)?,
            kv_norm: vector(&dev, &m.w.kv_norm)?,
            sink: upload_f32(&dev, "mixer sink", &m.w.sink)?,
            wo: m
                .wo_groups
                .iter()
                .map(|g| PackedDev::new(&dev, g, choices))
                .collect::<Result<Vec<_>>>()?,
            iota: upload_i32(
                &dev,
                "window slots",
                &(0..m.dims.window as i32).collect::<Vec<_>>(),
            )?,
            arenas: Mutex::default(),
            dev,
        })
    }

    pub(super) fn dev(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    /// The slots `0..window`.
    pub(super) fn iota(&self) -> &CudaSlice<i32> {
        &self.iota
    }

    pub(super) fn arenas(&self) -> MutexGuard<'_, Arenas> {
        self.arenas.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Tokens the window ring on the card stands for, 0 before the first decode step.
    pub(super) fn ring_tokens(&self) -> usize {
        self.arenas().ring_stamp.map_or(0, |(_, tokens)| tokens)
    }

    /// Bytes of the weights held on the card.
    pub(super) fn weight_bytes(&self) -> usize {
        let floats = self.q_norm.len() + self.kv_norm.len() + self.sink.len();
        self.wq_a.bytes()
            + self.wq_b.bytes()
            + self.wkv.bytes()
            + self.wo.iter().map(PackedDev::bytes).sum::<usize>()
            + floats * std::mem::size_of::<f32>()
    }

    /// The query factor, the queries and the key rows of the rows `x2` `[seq, hidden]` on the
    /// card, whose phases are `cos` and `sin`.
    ///
    /// Nothing comes back to the host.
    pub(super) fn front_dev(
        &self,
        m: &LatentMqa,
        x2: &Tensor,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<CardFront> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let seq = x2.dim(0)?;
        let (hd, rd) = (d.head_dim, d.rope_dim);
        // The rows and the 3 phases tables go up in one upload.
        let mut up = x2.flatten_all()?.to_vec1::<f32>()?;
        let (tables, table) = (up.len(), cos.len());
        up.extend_from_slice(cos);
        up.extend_from_slice(sin);
        up.extend(sin.iter().map(|&s| 0.0 - s));
        let x = upload_f32(dev, "mixer input", &up)?;
        let (cos_v, sin_v) = (
            x.slice(tables..tables + table),
            x.slice(tables + table..tables + 2 * table),
        );

        let q_rank = self.wq_a.out();
        let qa = self.wq_a.apply(&x, seq)?;
        let qr = rms_norm_f32(dev, &qa, &self.q_norm, seq, q_rank, d.eps)?;
        let mut q = self.wq_b.apply(&qr, seq)?;
        rope_tokens(dev, &mut q, (seq, d.heads, hd, rd), &cos_v, &sin_v)?;

        let raw = self.wkv.apply(&x, seq)?;
        let mut kv = rms_norm_f32(dev, &raw, &self.kv_norm, seq, hd, d.eps)?;
        rope_tokens(dev, &mut kv, (seq, 1, hd, rd), &cos_v, &sin_v)?;
        fp8_round(dev, &mut kv, FP8_BLOCK)?;

        Ok(CardFront {
            x,
            qr,
            q,
            kv,
            tables,
            table,
        })
    }

    /// [`Self::front_dev`] with the key rows, and the query factor when `need_qr`, read back for
    /// the host state and the host indexer.
    ///
    /// The query factor is empty when it is not needed.
    pub(super) fn front(
        &self,
        m: &LatentMqa,
        x2: &Tensor,
        cos: &[f32],
        sin: &[f32],
        need_qr: bool,
    ) -> Result<Front> {
        let dev: &CudaDevice = &self.dev;
        let seq = x2.dim(0)?;
        let hd = m.dims.head_dim;
        let q_rank = self.wq_a.out();
        let card = self.front_dev(m, x2, cos, sin)?;
        let qr = match need_qr {
            true => download(dev, &card.qr, seq * q_rank)?,
            false => Vec::new(),
        };
        Ok(Front {
            qr: Tensor::from_vec(qr, (if need_qr { seq } else { 0 }, q_rank), &Device::Cpu)?,
            kv: download(dev, &card.kv, seq * hd)?,
            card: Some(card),
        })
    }

    /// The heads' outputs `o` `[seq, heads * head_dim]` turned back and through the grouped
    /// output projection: `[seq, o_groups * o_lora_rank]` on the card.
    pub(super) fn project_dev(
        &self,
        m: &LatentMqa,
        mut o: CudaSlice<f32>,
        f: &CardFront,
        seq: usize,
    ) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let (hd, rd, nh) = (d.head_dim, d.rope_dim, d.heads);
        rope_tokens(dev, &mut o, (seq, nh, hd, rd), &f.cos(), &f.back())?;
        let (groups, rank) = (d.o_groups, d.o_lora_rank);
        let p = nh * hd / groups;
        let mut wide = alloc_f32(dev, "grouped output", seq * groups * rank)?;
        for (g, wo) in self.wo.iter().enumerate() {
            let slice = take_cols(dev, &o, (seq, nh * hd), (g * p, p))?;
            let y = wo.apply(&slice, seq)?;
            put_cols(dev, &y, &mut wide, (seq, groups * rank), (g * rank, rank))?;
        }
        Ok(wide)
    }

    /// The attention of `seq` queries over `keys`, rows of `head_dim` uploaded for the call.
    ///
    /// `slots` holds `width` row numbers per query, `-1` for an empty slot.
    pub(super) fn attend_rows(
        &self,
        m: &LatentMqa,
        f: &CardFront,
        keys: &[f32],
        slots: &[i32],
        width: usize,
        seq: usize,
    ) -> Result<Tensor> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let kd = upload_f32(dev, "mixer keys", keys)?;
        let idx = upload_i32(dev, "mixer slots", slots)?;
        let o = slots_attention(
            dev,
            &f.q,
            &kd,
            &self.sink,
            &idx,
            (seq, d.heads, d.head_dim, width),
            m.scale(),
        )?;
        let wide = self.project_dev(m, o, f, seq)?;
        let n = seq * d.out_width();
        Tensor::from_vec(download(dev, &wide, n)?, (seq, d.out_width()), &Device::Cpu)
    }

    /// The attention of the one decoded query over the slots `idx` of the window ring and the
    /// compressed rows, then the output projection, on the card: `[o_groups * o_lora_rank]`.
    pub(super) fn attend_decode(
        &self,
        m: &LatentMqa,
        f: &CardFront,
        arenas: &Arenas,
        idx: &CudaSlice<i32>,
        width: usize,
    ) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let Some(ring) = arenas.ring.as_ref() else {
            return Err(Error::msg("the window ring is not on the card"));
        };
        let keys = gather_ring(dev, ring, arenas.comp.as_ref(), idx, width)?;
        let o = sparse_attention_dev(
            dev,
            &f.q,
            &keys,
            &self.sink.slice(0..d.heads),
            idx,
            d.heads,
            d.head_dim,
            width,
            m.scale(),
        )?;
        self.project_dev(m, o, f, 1)
    }

    /// The decode step of a window layer: the output row `[o_groups * o_lora_rank]`, with the
    /// key row appended to the host window of `state`.
    ///
    /// `moves` is the restarts of the host state.
    /// The input row goes up and the output row and the key row come back, in one crossing each.
    pub(super) fn decode_window(
        &self,
        m: &LatentMqa,
        x2: &Tensor,
        pos: usize,
        state: &mut LatentMqaState,
    ) -> Result<Tensor> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let hd = d.head_dim;
        let (cos, sin) = phases(&m.inv_freq, pos..pos + 1);
        let f = self.front_dev(m, x2, &cos, &sin)?;
        let wide = {
            let mut arenas = self.arenas();
            arenas.sync_ring(
                dev,
                (hd, d.window),
                (m.moves, state.len + 1),
                &state.window,
                &f.kv,
            )?;
            let w = arenas.ring.as_ref().map_or(0, |r| r.len());
            self.attend_decode(m, &f, &arenas, &self.iota, w)?
        };
        let width = d.out_width();
        let host = pack_download(dev, &[wide.slice(0..width), f.kv.slice(0..hd)])?;
        state.window.extend_from_slice(&host[width..width + hd]);
        let excess = state.window.len().saturating_sub(d.window * hd);
        state.window.drain(..excess);
        state.len += 1;
        Tensor::from_vec(host[..width].to_vec(), (1, width), &Device::Cpu)
    }
}

impl LatentMqa {
    /// Runs the layer's projections and attention on the card `ordinal`, its block weights in
    /// their units of `choices`.
    ///
    /// The host weights and the host state stay.
    pub(super) fn on_card(&mut self, ordinal: usize, choices: &Choices) -> Result<()> {
        let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
            return Err(Error::msg("card mixers need a CUDA device"));
        };
        self.card = Some(Arc::new(CardLatent::new(dev, self, choices)?));
        Ok(())
    }

    /// Bytes of weights the layer holds on its card.
    pub fn card_bytes(&self) -> usize {
        self.card.as_ref().map_or(0, |c| c.weight_bytes())
    }

    /// The ordinal of the card the layer runs on, `None` on the host.
    pub fn card_ordinal(&self) -> Option<usize> {
        self.card.as_ref().map(|c| c.dev().ordinal())
    }

    /// Tokens the layer's window ring on its card stands for, 0 before the first decode step.
    pub fn card_ring_tokens(&self) -> usize {
        self.card.as_ref().map_or(0, |c| c.ring_tokens())
    }

    /// The groups' slices for the rows `x2` `[seq, hidden]` on the host, at positions from
    /// `index_pos`, computed on the card.
    ///
    /// The host state advances as the host forward advances it.
    pub(super) fn forward_card(&mut self, x2: &Tensor, index_pos: usize) -> Result<Tensor> {
        let Some(card) = self.card.clone() else {
            return Err(Error::msg("the layer is not on a card"));
        };
        let seq = x2.dim(0)?;
        let d = self.dims;
        let hd = d.head_dim;
        let mut state = self.state.take().unwrap_or_default();
        if seq == 1 && index_pos > 0 {
            let out = card.decode_window(self, x2, index_pos, &mut state)?;
            self.state = Some(state);
            return Ok(out);
        }
        let (cos, sin) = phases(&self.inv_freq, index_pos..index_pos + seq);
        let front = card.front(self, x2, &cos, &sin, false)?;
        let Some(cf) = front.card.as_ref() else {
            return Err(Error::msg("the card gave no front"));
        };
        let old = state.window.len() / hd;
        let mut keys = std::mem::take(&mut state.window);
        keys.extend_from_slice(&front.kv);
        let kept = (old + seq).min(d.window) * hd;
        state.window = keys[keys.len() - kept..].to_vec();
        state.len += seq;
        let slots = window_slots(old, seq, d.window);
        let out = card.attend_rows(self, cf, &keys, &slots, d.window, seq)?;
        self.state = Some(state);
        Ok(out)
    }
}
