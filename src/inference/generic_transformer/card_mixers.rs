//! The latent attention and compressed band mixers on a card.
//!
//! A mixer keeps its host weights and its host state, which stay the oracle and the source of
//! every snapshot, restore and rewind.
//! On a card it also holds its projections in the form the file stores them, see `card_packed.rs`,
//! and runs these steps there:
//! - the query factor, the queries and the key row, with their norms, rotation and fp8 round trip,
//! - the attention over the window rows and the picked compressed rows, with the sink,
//! - the turn back of the heads' outputs and the grouped output projection,
//! - the layer's output projection, once the load keeps it on the card.
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
//! A buffer is stamped with the name of the host rows it copies, see `LatentMqaState::stamp`.
//! Every forward names the host state anew, and a copy of a state keeps its name.
//! A decode step on the card appends to its buffers and names them as it names the host state.
//! A decode that finds the host state under another name rebuilds the buffer from it.
//! A restore, a rewind, a restart, a prefill and another sequence's state each come under
//! another name.
//! So a buffer never serves rows the host state does not hold.

use super::card_packed::{Choices, PackedDev};
use super::latent_mqa::{fresh_stamp, phases, Front, LatentMqa, LatentMqaState, FP8_BLOCK};
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::cuda::{
    alloc_f32, fp8_round, gather_ring, pack_download, put_cols, quantize_q8_1_rows_f32_view,
    rms_norm_f32, rope_tokens, slots_attention, sparse_attention_dev, take_cols, upload_f32,
    upload_i32, CudaDevice, GrowDev, RingDev,
};
use crate::tensor::gpu_runtime::driver::{CudaSlice, CudaView};
use crate::tensor::quantized::QKernelMatMul;
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

/// Whether a buffer stamped `stamp` copies rows named in `names`.
///
/// The name 0 copies nothing.
pub(super) fn copies(stamp: Option<u64>, names: &[u64]) -> bool {
    stamp.is_some_and(|s| s != 0 && names.contains(&s))
}

/// Makes `slot` hold `flat`, rows of `width`, and stamps it `name`.
///
/// `prefixes` names rows that `flat` starts with.
/// A buffer stamped with one of them takes the rows it lacks.
/// Any other buffer, or one longer than `flat`, is rebuilt.
/// True when the buffer was rebuilt.
pub(super) fn sync_rows(
    slot: &mut Option<GrowDev>,
    stamp: &mut Option<u64>,
    dev: &CudaDevice,
    width: usize,
    (prefixes, name): (&[u64], u64),
    flat: &[f32],
) -> Result<bool> {
    let rows = flat.len() / width;
    let grows = match slot {
        Some(c) => copies(*stamp, prefixes) && c.len() <= rows,
        None => false,
    };
    *stamp = None;
    if !grows {
        *slot = Some(GrowDev::open(dev, flat, width)?);
    } else if let Some(c) = slot.as_mut() {
        for row in flat.chunks_exact(width).skip(c.len()) {
            let up = upload_f32(dev, "device row", row)?;
            c.append(dev, &up.slice(0..width))?;
        }
    }
    *stamp = Some(name);
    Ok(!grows)
}

/// The ring and the compressed rows on the device, with the names of the rows they copy.
#[derive(Default)]
pub(super) struct Arenas {
    pub(super) ring: Option<RingDev>,
    /// The name of the state whose window the ring holds.
    ring_stamp: Option<u64>,
    /// Tokens the ring stands for.
    ring_tokens: usize,
    pub(super) comp: Option<GrowDev>,
    /// The name of the rows the compressed rows copy.
    comp_stamp: Option<u64>,
    /// Times the ring or the compressed rows were rebuilt from the host.
    #[cfg(test)]
    rebuilt: u64,
}

impl Arenas {
    /// Makes the ring hold the window after a row, which stands for `tokens` tokens seen, and
    /// names it `next`.
    ///
    /// `before` is the host window without the row, named `held`.
    /// `newest` is the row on the card.
    /// A ring that copies `held` takes the row alone.
    /// Any other ring is rebuilt from `before` and the row read back.
    pub(super) fn sync_ring(
        &mut self,
        dev: &CudaDevice,
        (hd, cap): (usize, usize),
        (held, next, tokens): (u64, u64, usize),
        before: &[f32],
        newest: &CudaSlice<f32>,
    ) -> Result<()> {
        let current = copies(self.ring_stamp, &[held]);
        self.ring_stamp = None;
        match (&mut self.ring, current) {
            (Some(ring), true) => ring.append(dev, &newest.slice(0..hd))?,
            _ => {
                #[cfg(test)]
                {
                    self.rebuilt += 1;
                }
                let mut all = before.to_vec();
                all.extend(download(dev, newest, hd)?);
                let kept = all.len().min(cap * hd);
                self.ring = Some(RingDev::open(dev, &all[all.len() - kept..], cap, hd)?);
            }
        }
        self.ring_stamp = Some(next);
        self.ring_tokens = tokens;
        Ok(())
    }

    /// Makes the compressed rows hold `flat`, rows of `hd`, and stamps them `name`.
    ///
    /// `prefixes` names rows that `flat` starts with.
    pub(super) fn sync_comp(
        &mut self,
        dev: &CudaDevice,
        hd: usize,
        (prefixes, name): (&[u64], u64),
        flat: &[f32],
    ) -> Result<()> {
        let names = (prefixes, name);
        let rebuilt = sync_rows(&mut self.comp, &mut self.comp_stamp, dev, hd, names, flat)?;
        #[cfg(test)]
        {
            self.rebuilt += u64::from(rebuilt);
        }
        #[cfg(not(test))]
        let _ = rebuilt;
        Ok(())
    }

    /// Times the ring or the compressed rows were rebuilt from the host.
    #[cfg(test)]
    pub(super) fn rebuilt(&self) -> u64 {
        self.rebuilt
    }

    /// Names `next` the compressed rows, which the card brought to the rows of the host state
    /// named `next`.
    pub(super) fn name_comp(&mut self, next: u64) {
        self.comp_stamp = Some(next);
    }

    /// Drops the names of the ring and the compressed rows, so the next decode rebuilds them.
    #[cfg(test)]
    pub(super) fn forget(&mut self) {
        self.ring_stamp = None;
        self.comp_stamp = None;
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
    /// The card addresses of the groups' blocks, with the host's copy they were uploaded from.
    wo_table: Mutex<Option<(Vec<u64>, CudaSlice<u64>)>>,
    /// The layer's output projection, which takes the groups' outputs to the hidden width.
    ///
    /// `None` leaves it to the host.
    output: Option<PackedDev>,
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
                .map(|g| PackedDev::in_group(&dev, g, choices, m.wo_groups.len()))
                .collect::<Result<Vec<_>>>()?,
            wo_table: Mutex::default(),
            output: None,
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
        self.arenas().ring_tokens
    }

    /// Bytes of the weights held on the card.
    pub(super) fn weight_bytes(&self) -> usize {
        let floats = self.q_norm.len() + self.kv_norm.len() + self.sink.len();
        self.wq_a.bytes()
            + self.wq_b.bytes()
            + self.wkv.bytes()
            + self.wo.iter().map(PackedDev::bytes).sum::<usize>()
            + self.output_bytes()
            + floats * std::mem::size_of::<f32>()
    }

    /// Bytes of the layer's output projection held on the card, 0 when the host holds it.
    pub(super) fn output_bytes(&self) -> usize {
        self.output.as_ref().map_or(0, PackedDev::bytes)
    }

    /// Whether the card takes the outputs through the layer's output projection.
    pub(super) fn projects_output(&self) -> bool {
        self.output.is_some()
    }

    /// Values of a row the card gives back: the hidden width once it holds the layer's output
    /// projection, else the groups' outputs.
    pub(super) fn width(&self, d: &super::latent_mqa::LatentMqaDims) -> usize {
        self.output.as_ref().map_or(d.out_width(), PackedDev::out)
    }

    /// The query factor, the queries and the key rows of `rows` on the card, whose phases are
    /// `cos` and `sin`.
    ///
    /// Nothing comes back to the host.
    pub(super) fn front_dev(
        &self,
        m: &LatentMqa,
        rows: FrontRows<'_>,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<CardFront> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let (hd, rd) = (d.head_dim, d.rope_dim);
        let table = cos.len();
        let mut phases = Vec::with_capacity(3 * table);
        phases.extend_from_slice(cos);
        phases.extend_from_slice(sin);
        phases.extend(sin.iter().map(|&s| 0.0 - s));
        // The rows and the 3 phases tables lie in one buffer.
        let (x, tables, seq) = match rows {
            FrontRows::Host(x2) => {
                let mut up = x2.flatten_all()?.to_vec1::<f32>()?;
                let tables = up.len();
                up.extend_from_slice(&phases);
                (upload_f32(dev, "mixer input", &up)?, tables, x2.dim(0)?)
            }
            FrontRows::Card(row) => {
                let n = row.len();
                let mut x = alloc_f32(dev, "mixer input", n + phases.len())?;
                let stream = dev.stream();
                stream
                    .memcpy_dtod(&row, &mut x.slice_mut(0..n))
                    .map_err(|e| Error::msg(format!("mixer input copy: {e}")))?;
                stream
                    .memcpy_htod(&phases, &mut x.slice_mut(n..n + phases.len()))
                    .map_err(|e| Error::msg(format!("mixer phases upload: {e}")))?;
                (x, n, 1)
            }
        };
        let (cos_v, sin_v) = (
            x.slice(tables..tables + table),
            x.slice(tables + table..tables + 2 * table),
        );

        let q_rank = self.wq_a.out();
        let (qa, raw) = match self.factor_and_key_row(&x, seq)? {
            Some((qa, raw)) => (qa, Some(raw)),
            None => (self.wq_a.apply(&x, seq)?, None),
        };
        let qr = rms_norm_f32(dev, &qa, &self.q_norm, seq, q_rank, d.eps)?;
        let mut q = self.wq_b.apply(&qr, seq)?;
        rope_tokens(dev, &mut q, (seq, d.heads, hd, rd), &cos_v, &sin_v)?;

        let raw = match raw {
            Some(raw) => raw,
            None => self.wkv.apply(&x, seq)?,
        };
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
        let mut since = std::time::Instant::now();
        let card = self.front_dev(m, FrontRows::Host(x2), cos, sin)?;
        mark(dev, "token mixer: front products", &mut since);
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
    ///
    /// A card that holds the layer's output projection runs it next: `[seq, hidden]`.
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
        if !(seq == 1 && self.grouped_row(&o, (p, rank), &mut wide)?) {
            self.grouped_by_columns(&o, seq, (nh * hd, p, rank), &mut wide)?;
        }
        match &self.output {
            Some(output) => output.apply(&wide, seq),
            None => Ok(wide),
        }
    }

    /// The grouped output projection of the `seq` rows `o`, `width` values each, written to
    /// `wide`: group `g` reads columns `g * p..(g + 1) * p` of `o` and writes columns `g * rank..(g + 1) * rank` of
    /// `wide`.
    pub(super) fn grouped_by_columns(
        &self,
        o: &CudaSlice<f32>,
        seq: usize,
        (width, p, rank): (usize, usize, usize),
        wide: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let dev: &CudaDevice = &self.dev;
        let groups = self.wo.len();
        for (g, wo) in self.wo.iter().enumerate() {
            let slice = take_cols(dev, o, (seq, width), (g * p, p))?;
            let y = wo.apply(&slice, seq)?;
            put_cols(dev, &y, wide, (seq, groups * rank), (g * rank, rank))?;
        }
        Ok(())
    }

    /// [`Self::grouped_by_columns`] with each group's product through the quantised matmul's
    /// forward over a copy of its columns.
    #[cfg(test)]
    pub(super) fn grouped_by_forward(
        &self,
        o: &CudaSlice<f32>,
        seq: usize,
        (width, p, rank): (usize, usize, usize),
    ) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let groups = self.wo.len();
        let mut wide = alloc_f32(dev, "grouped output", seq * groups * rank)?;
        for (g, wo) in self.wo.iter().enumerate() {
            let slice = take_cols(dev, o, (seq, width), (g * p, p))?;
            let y = wo.apply_through_forward(&slice, seq)?;
            put_cols(dev, &y, &mut wide, (seq, groups * rank), (g * rank, rank))?;
        }
        Ok(wide)
    }

    /// The query factor and the key row of the rows `x`, each through the quantised matmul's
    /// forward over a copy of the rows.
    #[cfg(test)]
    pub(super) fn factor_and_key_by_forward(
        &self,
        x: &CudaSlice<f32>,
        seq: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        Ok((
            self.wq_a.apply_through_forward(x, seq)?,
            self.wkv.apply_through_forward(x, seq)?,
        ))
    }

    /// The query factor and the key row of the one row `x`, from one q8_1 quantisation of `x`
    /// and one launch of both products.
    ///
    /// Each output has the bits of its own weight's product.
    /// `None` unless `seq` is 1 and both weights take their row through the same one-row
    /// mat-vec.
    pub(super) fn factor_and_key_row(
        &self,
        x: &CudaSlice<f32>,
        seq: usize,
    ) -> Result<Option<(CudaSlice<f32>, CudaSlice<f32>)>> {
        let (Some(ka), Some(kv)) = (self.wq_a.row_kernel(), self.wkv.row_kernel()) else {
            return Ok(None);
        };
        let inp = self.wq_a.inp();
        if seq != 1 || self.wkv.inp() != inp {
            return Ok(None);
        }
        let Some(q81) = ka.quantize_row(&x.slice(0..inp))? else {
            return Ok(None);
        };
        ka.matvec_pair_row(kv, &q81)
    }

    /// The grouped output projection of the one row `o` written to `wide`: group `g` reads the
    /// `p` values of `o` from `g * p` and writes the `rank` values of `wide` from `g * rank`.
    ///
    /// `o` is quantised to q8_1 once.
    /// A group's width is a multiple of the row padding of the q8_1 blocks.
    /// So a group's blocks are the blocks of its own quantisation.
    /// The groups run in one launch where the grouped mat-vec serves them, one launch a group
    /// otherwise.
    /// Each output has the bits of its group's product.
    /// False, with nothing written, unless every group takes its row through the one-row
    /// mat-vec.
    pub(super) fn grouped_row(
        &self,
        o: &CudaSlice<f32>,
        (p, rank): (usize, usize),
        wide: &mut CudaSlice<f32>,
    ) -> Result<bool> {
        self.grouped_row_in(o, (p, rank), wide, true)
    }

    /// Whether the groups' products of 1 row run in one launch.
    #[cfg(test)]
    pub(super) fn groups_in_one_launch(&self) -> bool {
        let kernels: Option<Vec<_>> = self.wo.iter().map(|w| w.row_kernel()).collect();
        kernels.is_some_and(|k| QKernelMatMul::group_addresses(&k).is_some())
    }

    /// The decision of the groups' unit for 1 row, `None` before it is measured.
    #[cfg(test)]
    pub(super) fn group_decision(&self) -> Option<super::card_packed::Decision> {
        self.wo.first()?.decision(1).map(|(_, d)| d)
    }

    /// [`Self::grouped_row`] in one launch a group.
    #[cfg(test)]
    pub(super) fn grouped_row_each(
        &self,
        o: &CudaSlice<f32>,
        (p, rank): (usize, usize),
        wide: &mut CudaSlice<f32>,
    ) -> Result<bool> {
        self.grouped_row_in(o, (p, rank), wide, false)
    }

    /// [`Self::grouped_row`], in one launch when `one_launch` and the grouped mat-vec serves the
    /// groups.
    fn grouped_row_in(
        &self,
        o: &CudaSlice<f32>,
        (p, rank): (usize, usize),
        wide: &mut CudaSlice<f32>,
        one_launch: bool,
    ) -> Result<bool> {
        use crate::tensor::quantized::MATRIX_ROW_PADDING;
        let kernels: Option<Vec<_>> = self
            .wo
            .iter()
            .map(|w| w.row_kernel().filter(|k| k.takes_one_row()))
            .collect();
        let Some(kernels) = kernels else {
            return Ok(false);
        };
        let groups = kernels.len();
        if !p.is_multiple_of(MATRIX_ROW_PADDING)
            || kernels.iter().any(|k| k.shape_kn() != (p, rank))
            || o.len() < groups * p
            || wide.len() < groups * rank
        {
            return Ok(false);
        }
        let q81 = quantize_q8_1_rows_f32_view(&self.dev, &o.slice(0..groups * p), groups * p, 1)?;
        let addresses = one_launch
            .then(|| QKernelMatMul::group_addresses(&kernels))
            .flatten();
        if let Some(addresses) = addresses {
            let mut table = self.wo_table.lock().unwrap_or_else(|e| e.into_inner());
            let (_, on_card) = match table.take() {
                Some((kept, up)) if kept == addresses => table.insert((kept, up)),
                _ => {
                    let up = self
                        .dev
                        .stream()
                        .clone_htod(&addresses)
                        .map_err(|e| Error::msg(format!("group addresses upload: {e}")))?;
                    table.insert((addresses, up))
                }
            };
            kernels[0].matvec_group_rows(
                on_card,
                groups,
                &q81.slice(..),
                &mut wide.slice_mut(0..groups * rank),
            )?;
            return Ok(true);
        }
        let bytes = q81.len() / groups;
        for (g, k) in kernels.iter().enumerate() {
            let blocks = q81.slice(g * bytes..(g + 1) * bytes);
            let mut dst = wide.slice_mut(g * rank..(g + 1) * rank);
            if !k.matvec_row_into(&blocks, &mut dst)? {
                return Err(Error::msg(
                    "a group of the output projection left the mat-vec",
                ));
            }
        }
        Ok(true)
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
        let mut since = std::time::Instant::now();
        let kd = upload_f32(dev, "mixer keys", keys)?;
        let idx = upload_i32(dev, "mixer slots", slots)?;
        mark(dev, "token mixer: attend upload", &mut since);
        let o = slots_attention(
            dev,
            &f.q,
            &kd,
            &self.sink,
            &idx,
            (seq, d.heads, d.head_dim, width),
            m.scale(),
        )?;
        mark(dev, "token mixer: attend slots", &mut since);
        let wide = self.project_dev(m, o, f, seq)?;
        mark(dev, "token mixer: attend projection", &mut since);
        let width = self.width(&d);
        Tensor::from_vec(
            download(dev, &wide, seq * width)?,
            (seq, width),
            &Device::Cpu,
        )
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

    /// The decode step of a window layer over the row `rows`: the output row on the card and the
    /// front, with the ring appended and named `next`.
    fn window_step(
        &self,
        m: &LatentMqa,
        rows: FrontRows<'_>,
        pos: usize,
        state: &LatentMqaState,
        next: u64,
    ) -> Result<(CudaSlice<f32>, CardFront)> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let (cos, sin) = phases(&m.inv_freq, pos..pos + 1);
        let f = self.front_dev(m, rows, &cos, &sin)?;
        let mut arenas = self.arenas();
        arenas.sync_ring(
            dev,
            (d.head_dim, d.window),
            (state.stamp, next, state.len + 1),
            &state.window,
            &f.kv,
        )?;
        let w = arenas.ring.as_ref().map_or(0, |r| r.len());
        let wide = self.attend_decode(m, &f, &arenas, &self.iota, w)?;
        Ok((wide, f))
    }

    /// The decode step of a window layer: the output row, `width` values, with the key row
    /// appended to the host window of `state`.
    ///
    /// The step names the state and the ring anew.
    /// The input row goes up and the output row and the key row come back, in one crossing each.
    pub(super) fn decode_window(
        &self,
        m: &LatentMqa,
        x2: &Tensor,
        pos: usize,
        state: &mut LatentMqaState,
    ) -> Result<Tensor> {
        let hd = m.dims.head_dim;
        let next = fresh_stamp();
        let (wide, f) = self.window_step(m, FrontRows::Host(x2), pos, state, next)?;
        let width = self.width(&m.dims);
        let host = pack_download(&self.dev, &[wide.slice(0..width), f.kv.slice(0..hd)])?;
        keep_key(m, state, &host[width..width + hd], next);
        Tensor::from_vec(host[..width].to_vec(), (1, width), &Device::Cpu)
    }
}

/// Rows a decode step on a card left there for the host state, and how the host reads them.
pub(super) struct Pending {
    dev: Arc<CudaDevice>,
    /// Each buffer with the count of its first values the host reads.
    rows: Vec<(CudaSlice<f32>, usize)>,
    /// How the mixer reads the rows into its state.
    pub(super) keep: Keep,
}

/// How a mixer reads the rows of a [`Pending`] into its state.
pub(super) enum Keep {
    /// The key row of a window layer, whose state takes the name `next`.
    Window { next: u64 },
    /// The rows of a band layer.
    Band(super::card_band::BandKeep),
}

impl Pending {
    /// The rows `rows` on `dev`, each with the count of its first values the host reads, read
    /// as `keep` says.
    pub(super) fn new(
        dev: Arc<CudaDevice>,
        rows: Vec<(CudaSlice<f32>, usize)>,
        keep: Keep,
    ) -> Self {
        Self { dev, rows, keep }
    }

    /// The rows on the card, in the order the host reads them.
    pub(super) fn views(&self) -> Vec<CudaView<'_, f32>> {
        self.rows.iter().map(|(s, n)| s.slice(0..*n)).collect()
    }

    /// The rows read back in one crossing of their own.
    pub(super) fn download(&self) -> Result<Vec<f32>> {
        pack_download(&self.dev, &self.views())
    }
}

/// The rows a card front reads.
pub(super) enum FrontRows<'a> {
    /// The rows `[seq, hidden]` on the host, uploaded with the phases.
    Host(&'a Tensor),
    /// 1 row already on the card.
    Card(CudaView<'a, f32>),
}

/// The key row `key` appended to the host window of `state`, which takes the name `next`.
fn keep_key(m: &LatentMqa, state: &mut LatentMqaState, key: &[f32], next: u64) {
    let d = m.dims;
    state.window.extend_from_slice(key);
    let excess = state.window.len().saturating_sub(d.window * d.head_dim);
    state.window.drain(..excess);
    state.len += 1;
    state.stamp = next;
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

    /// Keeps `output`, the layer's output projection, on the layer's card, in its unit of
    /// `choices`: the card then gives back rows of `output`'s width.
    ///
    /// False, with nothing kept, for a layer on the host or a card shared by a forward in flight.
    pub(super) fn keep_output_on_card(
        &mut self,
        output: &Projection,
        choices: &Choices,
    ) -> Result<bool> {
        let Some(card) = self.card.as_mut().and_then(Arc::get_mut) else {
            return Ok(false);
        };
        card.output = Some(PackedDev::new(&card.dev, output, choices)?);
        Ok(true)
    }

    /// Drops the layer's output projection from its card: the bytes it held.
    ///
    /// `None` when the card does not run it or a forward in flight shares the card.
    pub(super) fn release_output_on_card(&mut self) -> Option<u64> {
        let card = self.card.as_mut().and_then(Arc::get_mut)?;
        card.output.take().map(|p| p.bytes() as u64)
    }

    /// Whether the layer's card runs the layer's output projection.
    pub fn projects_output(&self) -> bool {
        self.card.as_ref().is_some_and(|c| c.projects_output())
    }

    /// The ordinal of the card the layer runs on, `None` on the host.
    pub fn card_ordinal(&self) -> Option<usize> {
        self.card.as_ref().map(|c| c.dev().ordinal())
    }

    /// The decode step of the row `row`, on the layer's card, at `index_pos`: the output row of
    /// the hidden width, left on the card, and the key row, left on the card for the host
    /// window.
    ///
    /// The state is unchanged until [`Self::keep_decoded`] reads the key row into it.
    /// Refused when the card does not run the output projection or the step does not continue
    /// the state.
    /// A failed step leaves no state, as a failed forward does.
    pub(super) fn decode_dev(
        &mut self,
        row: CudaView<'_, f32>,
        index_pos: usize,
    ) -> Result<(CudaSlice<f32>, Pending)> {
        let Some(card) = self.card.clone().filter(|c| c.projects_output()) else {
            return Err(Error::msg("the layer does not decode on a card"));
        };
        if index_pos == 0 || self.len() != index_pos {
            return Err(Error::msg(format!(
                "latent attention holds {} tokens and cannot decode position {index_pos} on its card",
                self.len()
            )));
        }
        let state = self.state.take().unwrap_or_default();
        let next = fresh_stamp();
        let (wide, f) = card.window_step(self, FrontRows::Card(row), index_pos, &state, next)?;
        self.state = Some(state);
        let rows = vec![(f.kv, self.dims.head_dim)];
        let pending = Pending::new(card.dev.clone(), rows, Keep::Window { next });
        Ok((wide, pending))
    }

    /// The key row `rows` of the last decode step on the card appended to the host window, which
    /// takes the step's name.
    pub(super) fn keep_decoded(&mut self, keep: &Keep, rows: &[f32]) -> Result<()> {
        let Keep::Window { next } = keep else {
            return Err(Error::msg("a window layer keeps a window step's rows"));
        };
        if rows.len() != self.dims.head_dim {
            return Err(Error::msg(format!(
                "{} values for a key row of {}",
                rows.len(),
                self.dims.head_dim
            )));
        }
        let mut state = self.state.take().unwrap_or_default();
        keep_key(self, &mut state, rows, *next);
        self.state = Some(state);
        Ok(())
    }

    /// Tokens the layer's window ring on its card stands for, 0 before the first decode step.
    pub fn card_ring_tokens(&self) -> usize {
        self.card.as_ref().map_or(0, |c| c.ring_tokens())
    }

    /// Drops the card's ring and compressed rows: the next decode rebuilds them from the host
    /// state.
    pub(super) fn drop_card_copies(&self) {
        if let Some(c) = &self.card {
            *c.arenas() = Arenas::default();
        }
    }

    /// Drops the names of the card's buffers, so the next decode rebuilds them from the host
    /// state.
    #[cfg(test)]
    pub(super) fn forget_card_copies(&self) {
        if let Some(c) = &self.card {
            c.arenas().forget();
        }
    }

    /// Times the card's ring or compressed rows were rebuilt from the host.
    #[cfg(test)]
    pub(super) fn card_rebuilds(&self) -> u64 {
        self.card.as_ref().map_or(0, |c| c.arenas().rebuilt())
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
        state.stamp = fresh_stamp();
        let slots = window_slots(old, seq, d.window);
        let out = card.attend_rows(self, cf, &keys, &slots, d.window, seq)?;
        self.state = Some(state);
        Ok(out)
    }
}

/// Under the stage profile, waits for the card's work and records the time since `since` under
/// `name`, then restarts `since`.
fn mark(dev: &CudaDevice, name: &'static str, since: &mut std::time::Instant) {
    if crate::inference::stage_prof::stage_prof_enabled() {
        let _ = dev.stream().synchronize();
        crate::inference::stage_prof::prof_record(name, since.elapsed().as_nanos() as u64);
        *since = std::time::Instant::now();
    }
}
