//! Engram: n-gram hash lookups written into the residual stream at a few layers.
//!
//! At each engram layer a position is hashed as the 2-gram .. `max_ngram`-gram ending there, each
//! split over `n_heads` heads; every (n-gram size, head) pair owns a prime-sized bucket range of the
//! layer's table. The rows the hashes fetch are concatenated, projected by `wkv` into one key per
//! hyper-connection copy plus a shared value, and the value is added to each copy under a gate that
//! is the normalised dot product of that copy against its key.
//!
//! The hash is over compressed token ids (tokens that normalise alike share one id), with
//! per-layer multipliers, primes and bucket offsets. All of it is read from the file: a table is
//! unusable without the exact hash that indexed it, so a file that carries the tables without the
//! hash parameters runs without engram. The tables are the largest tensors in the model and are
//! streamed from the mapping one row at a time; a row is one quantised block, so a lookup reads
//! a few dozen bytes.

use super::load::projection;
use super::source::WeightSource;
use crate::inference::offload::projection::Projection;
use crate::tensor::blockscaled::dequant::{e4m3_to_f32, e8m0_to_f32};
use crate::tensor::mapped::MappedBytes;
use crate::tensor::quant_view::rows_view;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::QTensor;
use crate::tensor::quantized::BLOCK_ALIGN;
use crate::tensor::{Device, Error, Result, Tensor};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Below this magnitude the gate's signed square root is taken at the floor, as in training.
const GATE_DOT_FLOOR: f32 = 1e-6;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The hash layout and parameters of every engram layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramConfig {
    pub layer_ids: Vec<usize>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub max_ngram: usize,
    /// Token id -> compressed id, over the whole vocabulary.
    pub token_map: Vec<u32>,
    /// The compressed id an n-gram slot takes before the sequence starts.
    pub pad_id: u32,
    /// Per engram layer, one multiplier per look-back (`max_ngram` of them).
    pub multipliers: Vec<Vec<i64>>,
    /// Per engram layer, the bucket modulus of each (n-gram size, head) column, n-gram major.
    pub primes: Vec<Vec<u64>>,
    /// Per engram layer, where each column's bucket range starts in the table.
    pub offsets: Vec<Vec<u64>>,
}

impl EngramConfig {
    /// Read the layout from the metadata under `{arch}.engram.*`; `None` when the file has no
    /// engram layers or omits any of the hash parameters.
    pub fn read(md: &HashMap<String, Value>, arch: &str) -> Option<Self> {
        let g = |k: &str| md.get(&format!("{arch}.engram.{k}"));
        let u = |k: &str| g(k).and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let arr_u64 = |k: &str| -> Option<Vec<u64>> {
            g(k)?
                .to_vec()
                .ok()?
                .iter()
                .map(|v| v.to_u64().ok())
                .collect()
        };
        let layer_ids: Vec<usize> = arr_u64("layer_ids")?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        if layer_ids.is_empty() {
            return None;
        }
        let (n_heads, head_dim, max_ngram) =
            (u("head_count")?, u("key_length")?, u("max_ngram_size")?);
        let n_cols = (max_ngram.checked_sub(1)?) * n_heads;
        let n = layer_ids.len();
        let split = |v: Vec<u64>, per: usize| -> Option<Vec<Vec<u64>>> {
            (v.len() == n * per).then(|| v.chunks(per).map(|c| c.to_vec()).collect())
        };
        // Written unsigned; each is bounded so that id * multiplier fits a signed 64-bit product.
        let multipliers: Vec<i64> = arr_u64("multipliers")?
            .into_iter()
            .map(i64::try_from)
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        let multipliers = (multipliers.len() == n * max_ngram)
            .then(|| multipliers.chunks(max_ngram).map(|c| c.to_vec()).collect())?;
        let primes = split(arr_u64("primes")?, n_cols)?;
        let offsets = split(arr_u64("offsets")?, n_cols)?;
        let token_map: Vec<u32> = arr_u64("token_map")?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let pad = u("pad_id")?;
        let pad_id = *token_map.get(pad)?;
        Some(Self {
            layer_ids,
            n_heads,
            head_dim,
            max_ngram,
            token_map,
            pad_id,
            multipliers,
            primes,
            offsets,
        })
    }

    /// The layout derived as the reference derives it from the model's arguments
    /// (`inference/config.json`) and the compressed token map: multipliers from the per-layer
    /// seeded generator, bucket primes and offsets from the engram vocabulary size.
    pub fn derive(v: &serde_json::Value, token_map: Vec<u32>) -> Option<Self> {
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let layer_ids: Vec<usize> = v
            .get("engram_layer_ids")?
            .as_array()?
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as usize))
            .collect();
        if layer_ids.is_empty() {
            return None;
        }
        let (n_heads, head_dim, max_ngram) = (
            u("engram_n_heads")?,
            u("engram_head_dim")?,
            u("engram_max_ngram_size")?,
        );
        let vocab = u("engram_vocab_size")? as u64;
        let compressed = u("engram_compressed_vocab_size")?;
        let pad = u("engram_pad_id")?;
        let n_compressed = token_map
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);
        if n_compressed != compressed {
            return None;
        }
        let pad_id = *token_map.get(pad)?;
        let multipliers = super::engram_hash::multipliers(&layer_ids, max_ngram, compressed);
        let (primes, offsets) =
            super::engram_hash::primes_and_offsets(layer_ids.len(), max_ngram, n_heads, vocab);
        Some(Self {
            layer_ids,
            n_heads,
            head_dim,
            max_ngram,
            token_map,
            pad_id,
            multipliers,
            primes,
            offsets,
        })
    }

    /// Hash columns per position.
    pub fn n_cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    /// Compressed id of a token; an id past the map hashes as the pad.
    pub fn compress(&self, token: u32) -> u32 {
        self.token_map
            .get(token as usize)
            .copied()
            .unwrap_or(self.pad_id)
    }

    /// The table rows one position fetches at engram layer `which` (an index into `layer_ids`).
    /// `recent` holds the compressed ids ending at that position, newest last; slots before the
    /// sequence start take the pad. The (i+1)-gram hash is the XOR of the first i+1 multiplied
    /// ids, reduced into each head's bucket range.
    pub fn hash(&self, which: usize, recent: &[u32]) -> Vec<u64> {
        let mult = &self.multipliers[which];
        let mut rolling: i64 = 0;
        let mut out = Vec::with_capacity(self.n_cols());
        for shift in 0..self.max_ngram {
            let id = recent
                .len()
                .checked_sub(1 + shift)
                .map(|i| recent[i])
                .unwrap_or(self.pad_id);
            let product = (id as i64).wrapping_mul(mult[shift]);
            rolling = if shift == 0 {
                product
            } else {
                rolling ^ product
            };
            if shift == 0 {
                continue;
            }
            for h in 0..self.n_heads {
                let col = (shift - 1) * self.n_heads + h;
                let p = self.primes[which][col];
                out.push((rolling as u64) % p + self.offsets[which][col]);
            }
        }
        out
    }
}

/// The table an engram layer looks rows up in: GGUF blocks streamed from the mapping, the
/// released checkpoint's fp8 rows with their ue8m0 scales streamed from theirs, or resident.
pub enum EngramTable {
    Streamed(Arc<QTensor>),
    Fp8Rows {
        bytes: MappedBytes,
        scales: MappedBytes,
        rows: usize,
        row_len: usize,
        block: usize,
    },
    Resident(Tensor),
}

impl EngramTable {
    fn row(&self, r: u64) -> Result<Vec<f32>> {
        let r = r as usize;
        match self {
            Self::Streamed(t) => {
                // A view starts on the block alignment; a row whose byte offset does not is
                // read within the shortest run of rows that does.
                let native = t.native_qtensor();
                let (rows, cols) = (native.dims[0], native.dims[1]);
                let row_bytes = cols / native.dtype.block_size() * native.dtype.type_size();
                let group = BLOCK_ALIGN / gcd(row_bytes, BLOCK_ALIGN);
                let start = r - r % group;
                let count = group.min(rows - start);
                rows_view(t, start, count)?
                    .dequantize(&Device::Cpu)?
                    .narrow(0, r - start, 1)?
                    .flatten_all()?
                    .to_vec1::<f32>()
            }
            Self::Fp8Rows {
                bytes,
                scales,
                row_len,
                block,
                ..
            } => {
                let per_row = row_len / block;
                let row = &bytes.as_slice()[r * row_len..(r + 1) * row_len];
                let sc = &scales.as_slice()[r * per_row..(r + 1) * per_row];
                Ok(row
                    .iter()
                    .enumerate()
                    .map(|(j, &b)| e4m3_to_f32(b) * e8m0_to_f32(sc[j / block]))
                    .collect())
            }
            Self::Resident(t) => t.narrow(0, r, 1)?.flatten_all()?.to_vec1::<f32>(),
        }
    }

    /// Start reading rows `ids` in, where the table is mapped.
    fn will_need(&self, ids: &[u64]) {
        if let Self::Fp8Rows {
            bytes,
            scales,
            row_len,
            block,
            ..
        } = self
        {
            let per_row = row_len / block;
            for &r in ids {
                let r = r as usize;
                bytes.will_need_range(r * row_len, *row_len);
                scales.will_need_range(r * per_row, per_row);
            }
        }
    }

    /// Ask the kernel to read this table at random. A lookup wants a few dozen bytes; read-ahead
    /// around it pulls megabytes per row and pushes the rest of the model out of the page cache.
    pub fn advise_random(&self) {
        if let Self::Fp8Rows { bytes, scales, .. } = self {
            bytes.advise_random();
            scales.advise_random();
        }
    }

    pub fn n_rows(&self) -> usize {
        match self {
            Self::Streamed(t) => t.shape().dims()[0],
            Self::Fp8Rows { rows, .. } => *rows,
            Self::Resident(t) => t.dims()[0],
        }
    }
}

/// One engram layer's weights.
pub struct Engram {
    pub table: EngramTable,
    /// `[dim * (hc_mult + 1), n_cols * head_dim]`: keys for every copy, then the shared value.
    pub wkv: Projection,
    /// `[hc_mult, dim]`, the product of the query and key weights - only ever used as one.
    pub qk_weight: Vec<Vec<f32>>,
    pub dim: usize,
    pub hc_mult: usize,
    pub eps: f32,
}

impl Engram {
    /// Load layer `layer`'s engram from the source.
    pub fn load<S: WeightSource + ?Sized>(
        g: &S,
        layer: usize,
        dim: usize,
        hc_mult: usize,
        eps: f32,
    ) -> Result<Self> {
        let p = format!("layers.{layer}.engram");
        let table = g.engram_table(&format!("{p}.embed.weight"))?;
        let q = g.dense_f32(&format!("{p}.q_weight"))?.to_vec2::<f32>()?;
        let k = g.dense_f32(&format!("{p}.k_weight"))?.to_vec2::<f32>()?;
        let qk_weight = q
            .iter()
            .zip(&k)
            .map(|(qr, kr)| qr.iter().zip(kr).map(|(a, b)| a * b).collect())
            .collect();
        Ok(Self {
            table,
            wkv: projection(g, &format!("{p}.wkv.weight"))?,
            qk_weight,
            dim,
            hc_mult,
            eps,
        })
    }

    /// `x` `[1, s, hc_mult, dim]` in, the same shape out with the gated value added to every copy;
    /// `hashes` holds one row-id list per position.
    pub fn forward(&self, x: &Tensor, hashes: &[Vec<u64>]) -> Result<Tensor> {
        let (dim, hc) = (self.dim, self.hc_mult);
        let s = hashes.len();
        let width = self.wkv.dims()[1];
        let mut gathered = Vec::with_capacity(s * width);
        for ids in hashes {
            self.table.will_need(ids);
        }
        for ids in hashes {
            for &r in ids {
                gathered.extend(self.table.row(r)?);
            }
        }
        if gathered.len() != s * width {
            return Err(Error::msg(format!(
                "engram: {} row values gathered for {s} positions of width {width}",
                gathered.len()
            )));
        }
        let e = Tensor::from_vec(gathered, (s, width), &Device::Cpu)?;
        let kv = self.wkv.apply(&e)?.to_vec2::<f32>()?; // [s, dim * (hc + 1)]
        let mut h = x.flatten_all()?.to_vec1::<f32>()?; // [s * hc * dim]
        let scale = (dim as f32).powf(-0.5);
        for (t, kv_t) in kv.iter().enumerate() {
            let value = &kv_t[hc * dim..];
            for c in 0..hc {
                let key = &kv_t[c * dim..(c + 1) * dim];
                let row = &mut h[(t * hc + c) * dim..(t * hc + c + 1) * dim];
                let w = &self.qk_weight[c];
                let h_ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                let k_ms = key.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                let rstd = (h_ms + self.eps).sqrt().recip() * (k_ms + self.eps).sqrt().recip();
                let dot: f32 = row
                    .iter()
                    .zip(w)
                    .zip(key)
                    .map(|((a, b), c)| a * b * c)
                    .sum::<f32>()
                    * rstd
                    * scale;
                let gate =
                    1.0 / (1.0 + (-dot.abs().max(GATE_DOT_FLOOR).sqrt().copysign(dot)).exp());
                for (o, v) in row.iter_mut().zip(value) {
                    *o += gate * v;
                }
            }
        }
        Tensor::from_vec(h, (1, s, hc, dim), &Device::Cpu)
    }
}

/// The compressed ids a decode stream has seen, enough of them to hash the next position.
#[derive(Default, Clone)]
pub struct EngramState {
    recent: VecDeque<u32>,
}

impl EngramState {
    /// Record `token` and return the ids the hash reads at its position, newest last.
    pub fn push(&mut self, cfg: &EngramConfig, token: u32) -> Vec<u32> {
        self.recent.push_back(cfg.compress(token));
        while self.recent.len() > cfg.max_ngram {
            self.recent.pop_front();
        }
        self.recent.iter().copied().collect()
    }
}

/// The row ids of every position of a prefill, per engram layer index.
pub fn prefill_hashes(cfg: &EngramConfig, which: usize, tokens: &[u32]) -> Vec<Vec<u64>> {
    let mut st = EngramState::default();
    tokens
        .iter()
        .map(|&t| {
            let recent = st.push(cfg, t);
            cfg.hash(which, &recent)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EngramConfig {
        EngramConfig {
            layer_ids: vec![1],
            n_heads: 2,
            head_dim: 4,
            max_ngram: 3,
            token_map: vec![0, 1, 2, 3, 3, 5],
            pad_id: 2,
            multipliers: vec![vec![3, 5, 7]],
            primes: vec![vec![11, 13, 17, 19]],
            offsets: vec![vec![0, 11, 24, 41]],
        }
    }

    /// Every column lands in its own bucket range, and a position hashes the pad for the slots
    /// before the sequence starts: the first position's rows equal those of any position whose
    /// look-back is all pad.
    #[test]
    fn columns_stay_in_their_bucket_ranges_and_the_start_pads() {
        let c = cfg();
        let rows = prefill_hashes(&c, 0, &[5, 1, 3, 4]);
        for r in &rows {
            assert_eq!(r.len(), 4);
            for (col, &id) in r.iter().enumerate() {
                let (o, p) = (c.offsets[0][col], c.primes[0][col]);
                assert!(
                    id >= o && id < o + p,
                    "col {col}: {id} outside [{o}, {})",
                    o + p
                );
            }
        }
        assert_eq!(rows[0], c.hash(0, &[c.pad_id, c.pad_id, c.compress(5)]));
        // Tokens 3 and 4 share a compressed id, so the same history hashes the same.
        assert_eq!(
            c.hash(0, &[1, 3]),
            c.hash(0, &[c.compress(1), c.compress(4)])
        );
        // The 2-gram columns depend on the previous token, the 3-gram ones on the one before.
        assert_ne!(rows[1][..2], rows[3][..2]);
    }

    /// Decoding token by token reads exactly the rows prefill hashes at each position.
    #[test]
    fn decode_hashes_match_prefill() {
        let c = cfg();
        let tokens = [1u32, 5, 0, 3, 2, 4, 1];
        let want = prefill_hashes(&c, 0, &tokens);
        let mut st = EngramState::default();
        for (i, &t) in tokens.iter().enumerate() {
            let recent = st.push(&c, t);
            assert!(recent.len() <= c.max_ngram);
            assert_eq!(c.hash(0, &recent), want[i], "position {i}");
        }
    }
}
