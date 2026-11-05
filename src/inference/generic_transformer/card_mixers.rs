//! The latent attention and compressed band mixers on a card.
//!
//! A mixer keeps its host weights and its host state, which stay the oracle and the source of
//! every snapshot, restore and rewind.
//! On a card it also holds its projections as dense F32 and runs these steps there:
//! - the query factor, the queries and the key row, with their norms, rotation and fp8 round trip,
//! - the attention over the window rows and the picked compressed rows, with the sink,
//! - the turn back of the heads' outputs and the grouped output projection.
//!
//! The compressor, the index keys and the indexer with its top-k stay on the host in a band.
//!
//! The window lives in a ring and the compressed rows in a growing buffer, both in device memory.
//! Each is flat rows that the gather kernel reads by slot, as the decode stream reads them.
//! A prefill gathers from the rows it uploads and leaves the buffers to the next decode.
//!
//! A buffer holds the state it was stamped with, and a decode that finds another stamp rebuilds
//! it from the host state.
//! A restore, a rewind or a restart changes the stamp, so a buffer never serves a state it does
//! not hold.

use super::latent_mqa::{Front, LatentMqa, FP8_BLOCK};
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::cuda::{
    alloc_f32, fp8_round, gather_ring, linear_dev, put_cols, rms_norm_f32, rope_tokens,
    slots_attention, sparse_attention_dev, take_cols, upload_f32, upload_i32, CudaDevice, GrowDev,
    RingDev,
};
use crate::tensor::gpu_runtime::driver::CudaSlice;
use crate::tensor::{Device, Error, Result, Tensor};
use std::sync::{Arc, Mutex};

/// A weight `[out, inp]` held dense F32 in device memory.
struct DenseDev {
    w: CudaSlice<f32>,
    out: usize,
    inp: usize,
}

impl DenseDev {
    /// `p` decoded on the host and uploaded.
    fn new(dev: &CudaDevice, p: &Projection) -> Result<Self> {
        let dims = p.dims();
        let (out, inp) = match dims.as_slice() {
            [out, inp] => (*out, *inp),
            _ => return Err(Error::msg("a card mixer weight is a matrix")),
        };
        Ok(Self {
            w: upload_f32(dev, "mixer weight", &p.dense_f32()?)?,
            out,
            inp,
        })
    }

    /// `x` `[rows, inp]` through the weight: `[rows, out]`.
    fn apply(&self, dev: &CudaDevice, x: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>> {
        linear_dev(dev, x, &self.w, (rows, self.inp, self.out))
    }

    fn bytes(&self) -> usize {
        self.w.len() * std::mem::size_of::<f32>()
    }
}

/// What the device holds of the query side and key rows of one forward.
pub(super) struct CardFront {
    /// The rotated queries `[seq, heads * head_dim]`.
    q: CudaSlice<f32>,
    /// The rotated key rows, rounded to the fp8 grid, `[seq, head_dim]`.
    kv: CudaSlice<f32>,
    /// The phases of the rows, and the sines turned back.
    cos: CudaSlice<f32>,
    back: CudaSlice<f32>,
}

/// The ring and the compressed rows on the device, with the state they were stamped with.
#[derive(Default)]
struct Arenas {
    ring: Option<RingDev>,
    /// The restarts and the tokens the ring holds.
    ring_stamp: Option<(u64, usize)>,
    comp: Option<GrowDev>,
    /// The restarts the compressed rows were built under.
    comp_moves: u64,
}

impl Arenas {
    /// Makes the ring hold `window` (flat rows of `hd`, the newest last), which stands for
    /// `tokens` tokens seen.
    ///
    /// A ring stamped with the tokens before takes the `newest` row alone.
    /// Any other ring is rebuilt from `window`.
    fn sync_ring(
        &mut self,
        dev: &CudaDevice,
        (hd, cap): (usize, usize),
        (moves, tokens): (u64, usize),
        window: &[f32],
        newest: &CudaSlice<f32>,
    ) -> Result<()> {
        let current = self.ring_stamp == Some((moves, tokens.saturating_sub(1)));
        self.ring_stamp = None;
        match (&mut self.ring, current) {
            (Some(ring), true) => ring.append(dev, &newest.slice(0..hd))?,
            _ => self.ring = Some(RingDev::open(dev, window, cap, hd)?),
        }
        self.ring_stamp = Some((moves, tokens));
        Ok(())
    }

    /// Makes the compressed rows hold `flat`, rows of `hd`.
    ///
    /// Rows only grow within one restart, so the buffer takes the rows it lacks.
    /// A buffer of another restart, or longer than `flat`, is rebuilt.
    fn sync_comp(&mut self, dev: &CudaDevice, hd: usize, moves: u64, flat: &[f32]) -> Result<()> {
        let rows = flat.len() / hd;
        let grows = match &self.comp {
            Some(c) => self.comp_moves == moves && c.len() <= rows,
            None => false,
        };
        if !grows {
            self.comp = Some(GrowDev::open(dev, flat, hd)?);
            self.comp_moves = moves;
            return Ok(());
        }
        if let Some(c) = self.comp.as_mut() {
            for row in flat.chunks_exact(hd).skip(c.len()) {
                let up = upload_f32(dev, "compressed row", row)?;
                c.append(dev, &up.slice(0..hd))?;
            }
        }
        Ok(())
    }
}

/// A latent attention layer's weights and buffers on a card.
pub(super) struct CardLatent {
    dev: Arc<CudaDevice>,
    wq_a: DenseDev,
    q_norm: CudaSlice<f32>,
    wq_b: DenseDev,
    wkv: DenseDev,
    kv_norm: CudaSlice<f32>,
    sink: CudaSlice<f32>,
    /// The groups of the output projection.
    wo: Vec<DenseDev>,
    arenas: Mutex<Arenas>,
}

/// `n` values of `s` read back.
fn download(dev: &CudaDevice, s: &CudaSlice<f32>, n: usize) -> Result<Vec<f32>> {
    let mut host = dev
        .stream()
        .clone_dtoh(s)
        .map_err(|e| Error::msg(format!("card mixer download: {e}")))?;
    host.truncate(n);
    Ok(host)
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
    /// The layer `m` on `dev`.
    pub(super) fn new(dev: Arc<CudaDevice>, m: &LatentMqa) -> Result<Self> {
        let vector = |t: &Tensor| -> Result<CudaSlice<f32>> {
            let v = t
                .to_dtype(crate::tensor::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            upload_f32(&dev, "mixer norm", &v)
        };
        Ok(Self {
            wq_a: DenseDev::new(&dev, &m.w.wq_a)?,
            q_norm: vector(&m.w.q_norm)?,
            wq_b: DenseDev::new(&dev, &m.w.wq_b)?,
            wkv: DenseDev::new(&dev, &m.w.wkv)?,
            kv_norm: vector(&m.w.kv_norm)?,
            sink: upload_f32(&dev, "mixer sink", &m.w.sink)?,
            wo: m
                .wo_groups
                .iter()
                .map(|g| DenseDev::new(&dev, g))
                .collect::<Result<Vec<_>>>()?,
            arenas: Mutex::default(),
            dev,
        })
    }

    /// Tokens the window ring on the card stands for, 0 before the first decode step.
    pub(super) fn ring_tokens(&self) -> usize {
        let arenas = self.arenas.lock().unwrap_or_else(|e| e.into_inner());
        arenas.ring_stamp.map_or(0, |(_, tokens)| tokens)
    }

    /// Bytes of the weights held on the card.
    pub(super) fn weight_bytes(&self) -> usize {
        self.wq_a.bytes()
            + self.wq_b.bytes()
            + self.wkv.bytes()
            + self.wo.iter().map(DenseDev::bytes).sum::<usize>()
    }

    /// The query factor, the queries and the key rows of the rows `x2` `[seq, hidden]` on the
    /// card, whose phases are `cos` and `sin`.
    ///
    /// The query factor and the key rows also come back to the host.
    pub(super) fn front(
        &self,
        m: &LatentMqa,
        x2: &Tensor,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Front> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let seq = x2.dim(0)?;
        let (hd, rd) = (d.head_dim, d.rope_dim);
        let xs = x2.flatten_all()?.to_vec1::<f32>()?;
        let x = upload_f32(dev, "mixer input", &xs)?;
        let cos_d = upload_f32(dev, "mixer phases", cos)?;
        let sin_d = upload_f32(dev, "mixer phases", sin)?;
        let back: Vec<f32> = sin.iter().map(|&s| 0.0 - s).collect();
        let back_d = upload_f32(dev, "mixer phases", &back)?;

        let q_rank = self.wq_a.out;
        let qa = self.wq_a.apply(dev, &x, seq)?;
        let qr = rms_norm_f32(dev, &qa, &self.q_norm, seq, q_rank, d.eps)?;
        let mut q = self.wq_b.apply(dev, &qr, seq)?;
        rope_tokens(dev, &mut q, (seq, d.heads, hd, rd), &cos_d, &sin_d)?;

        let raw = self.wkv.apply(dev, &x, seq)?;
        let mut kv = rms_norm_f32(dev, &raw, &self.kv_norm, seq, hd, d.eps)?;
        rope_tokens(dev, &mut kv, (seq, 1, hd, rd), &cos_d, &sin_d)?;
        fp8_round(dev, &mut kv, FP8_BLOCK)?;

        Ok(Front {
            qr: Tensor::from_vec(
                download(dev, &qr, seq * q_rank)?,
                (seq, q_rank),
                &Device::Cpu,
            )?,
            kv: download(dev, &kv, seq * hd)?,
            card: Some(CardFront {
                q,
                kv,
                cos: cos_d,
                back: back_d,
            }),
        })
    }

    /// The heads' outputs `o` `[seq, heads * head_dim]` turned back and through the grouped
    /// output projection: `[seq, o_groups * o_lora_rank]` on the host.
    fn project(
        &self,
        m: &LatentMqa,
        mut o: CudaSlice<f32>,
        f: &CardFront,
        seq: usize,
    ) -> Result<Tensor> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let (hd, rd, nh) = (d.head_dim, d.rope_dim, d.heads);
        rope_tokens(dev, &mut o, (seq, nh, hd, rd), &f.cos, &f.back)?;
        let (groups, rank) = (d.o_groups, d.o_lora_rank);
        let p = nh * hd / groups;
        let mut wide = alloc_f32(dev, "grouped output", seq * groups * rank)?;
        for (g, wo) in self.wo.iter().enumerate() {
            let slice = take_cols(dev, &o, (seq, nh * hd), (g * p, p))?;
            let y = wo.apply(dev, &slice, seq)?;
            put_cols(dev, &y, &mut wide, (seq, groups * rank), (g * rank, rank))?;
        }
        let host = download(dev, &wide, seq * groups * rank)?;
        Tensor::from_vec(host, (seq, groups * rank), &Device::Cpu)
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
        self.project(m, o, f, seq)
    }

    /// The attention of the one decoded query, over the window ring and the compressed rows.
    ///
    /// `window` is the host window after this row, `tokens` the tokens seen with it, and
    /// `moves` the restarts of the host state.
    /// `compressed` is the flat compressed rows of a band, `picks` the compressed slots the
    /// query reaches, already shifted by the window's length.
    pub(super) fn decode(
        &self,
        m: &LatentMqa,
        f: &CardFront,
        (moves, tokens): (u64, usize),
        window: &[f32],
        (compressed, picks): (Option<&[f32]>, &[i32]),
    ) -> Result<Tensor> {
        let dev: &CudaDevice = &self.dev;
        let d = m.dims;
        let hd = d.head_dim;
        let mut arenas = self.arenas.lock().unwrap_or_else(|e| e.into_inner());
        arenas.sync_ring(dev, (hd, d.window), (moves, tokens), window, &f.kv)?;
        match compressed {
            Some(flat) => arenas.sync_comp(dev, hd, moves, flat)?,
            None => arenas.comp = None,
        }
        let Some(ring) = arenas.ring.as_ref() else {
            return Err(Error::msg("the window ring is not on the card"));
        };
        let w = ring.len();
        let mut slots: Vec<i32> = (0..w as i32).collect();
        slots.extend_from_slice(picks);
        let width = slots.len();
        let idx = upload_i32(dev, "mixer slots", &slots)?;
        let keys = gather_ring(dev, ring, arenas.comp.as_ref(), &idx, width)?;
        let o = sparse_attention_dev(
            dev,
            &f.q,
            &keys,
            &self.sink.slice(0..d.heads),
            &idx,
            d.heads,
            hd,
            width,
            m.scale(),
        )?;
        drop(arenas);
        self.project(m, o, f, 1)
    }
}

impl LatentMqa {
    /// Runs the layer's projections and attention on the card `ordinal`.
    ///
    /// The host weights and the host state stay.
    pub fn on_card(&mut self, ordinal: usize) -> Result<()> {
        let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
            return Err(Error::msg("card mixers need a CUDA device"));
        };
        self.card = Some(Arc::new(CardLatent::new(dev, self)?));
        Ok(())
    }

    /// Bytes of weights the layer holds on its card.
    pub fn card_bytes(&self) -> usize {
        self.card.as_ref().map_or(0, |c| c.weight_bytes())
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
        let (cos, sin) = super::latent_mqa::phases(&self.inv_freq, index_pos..index_pos + seq);
        let front = card.front(self, x2, &cos, &sin)?;
        let Some(cf) = front.card.as_ref() else {
            return Err(Error::msg("the card gave no front"));
        };

        let mut state = self.state.take().unwrap_or_default();
        let old = state.window.len() / hd;
        let mut keys = std::mem::take(&mut state.window);
        keys.extend_from_slice(&front.kv);
        let kept = (old + seq).min(d.window) * hd;
        state.window = keys[keys.len() - kept..].to_vec();
        state.len += seq;

        let out = match seq == 1 && index_pos > 0 {
            true => card.decode(
                self,
                cf,
                (self.moves, state.len),
                &state.window,
                (None, &[]),
            )?,
            false => {
                let slots = window_slots(old, seq, d.window);
                card.attend_rows(self, cf, &keys, &slots, d.window, seq)?
            }
        };
        self.state = Some(state);
        Ok(out)
    }
}
