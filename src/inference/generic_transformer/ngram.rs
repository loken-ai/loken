//! A hashed n-gram embedding: each token gathers one row per head from one large table, the
//! row of a head chosen by hashing the token with its predecessors, and the gathered rows are
//! gated into the residual of one layer.
//!
//! The table is only ever read by row, so it stays in the mapping and a token touches
//! `heads * row_bytes` of it.

use super::config::{meta_u32, meta_u32_array, meta_u64_array};
use super::hyper::grouped_rms;
use super::{gguf_file, QMatMul, Result, Tensor};
use crate::tensor::quantized::QTensor;
use crate::tensor::{DType, Device, D};

/// The n-gram embedding's parameters, from the file's `ple.*` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramMeta {
    /// The layer the embedding joins the residual before.
    pub layer: usize,
    /// Longest n-gram hashed; the hashes cover lengths 2 up to this.
    pub ngram_size: usize,
    /// Hash heads of each n-gram length.
    pub heads_per_ngram: usize,
    /// Taps of the dilated convolution; its dilation is `ngram_size`.
    pub conv_kernel: usize,
    /// Token that cuts the n-gram window.
    pub eos_token: u32,
    /// Width of the row one head gathers.
    pub row_dim: usize,
    /// One odd multiplier per position of the window.
    pub multipliers: Vec<u64>,
    /// First table row of each head.
    pub head_offsets: Vec<u64>,
    /// Rows of each head's range.
    pub head_vocab: Vec<u64>,
}

impl NgramMeta {
    /// Heads in all: `ngram_size - 1` lengths of `heads_per_ngram` each.
    pub fn heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// The metadata of `arch`; `None` when the file declares no n-gram layer or the arrays are
    /// shorter than the heads and window need.
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> Option<Self> {
        let layers = meta_u32_array(content, arch, "ple.layers")?;
        let &[layer] = layers.as_slice() else {
            return None;
        };
        let m = Self {
            layer: layer as usize,
            ngram_size: meta_u32(content, arch, "ple.ngram_size")? as usize,
            heads_per_ngram: meta_u32(content, arch, "ple.heads_per_ngram")? as usize,
            conv_kernel: meta_u32(content, arch, "ple.conv_kernel")? as usize,
            eos_token: meta_u32(content, arch, "ple.eos_token_id")?,
            row_dim: meta_u32(content, arch, "embedding_length_per_layer_input")? as usize,
            multipliers: meta_u64_array(content, arch, "ple.layer_multipliers")?,
            head_offsets: meta_u64_array(content, arch, "ple.head_offsets")?,
            head_vocab: meta_u64_array(content, arch, "ple.head_vocab_sizes")?,
        };
        let ok = m.ngram_size >= 2
            && m.heads_per_ngram > 0
            && m.conv_kernel > 0
            && m.row_dim > 0
            && m.multipliers.len() >= m.ngram_size
            && m.head_offsets.len() >= m.heads()
            && m.head_vocab.len() >= m.heads()
            && m.head_vocab[..m.heads()].iter().all(|&v| v > 0);
        ok.then_some(m)
    }

    /// Rows the heads' ranges cover: the least the table must hold.
    pub fn rows(&self) -> u64 {
        (0..self.heads())
            .map(|h| self.head_offsets[h] + self.head_vocab[h])
            .max()
            .unwrap_or(0)
    }
}

/// The last tokens of the sequence, which the hash of the next token reaches back to.
#[derive(Debug, Clone)]
pub struct NgramHistory {
    /// Oldest first; `ngram_size - 1` tokens, the end-of-sequence token where the sequence
    /// has not reached.
    prev: Vec<u32>,
    /// Tokens the history has seen: the position the next forward must start at.
    pub seen: usize,
}

impl NgramHistory {
    pub fn new(meta: &NgramMeta) -> Self {
        Self {
            prev: vec![meta.eos_token; meta.ngram_size - 1],
            seen: 0,
        }
    }

    /// The table rows of `tokens`, `heads` per token, advancing the history past them.
    ///
    /// A token hashes with its predecessors; an end-of-sequence token among them cuts the
    /// window, everything at or before it reading as the end-of-sequence token, but the
    /// token's own end-of-sequence does not cut its own context.
    pub fn rows(&mut self, meta: &NgramMeta, tokens: &[u32]) -> Vec<u32> {
        let heads = meta.heads();
        let back = meta.ngram_size - 1;
        let mut out = Vec::with_capacity(tokens.len() * heads);
        let mut window: Vec<u32> = self.prev.clone();
        window.extend_from_slice(tokens);
        for i in 0..tokens.len() {
            let at = back + i;
            let mut ctx = vec![window[at]];
            let mut cut = false;
            for s in 1..=back {
                let t = window[at - s];
                cut = cut || t == meta.eos_token;
                ctx.push(if cut { meta.eos_token } else { t });
            }
            for n in 2..=meta.ngram_size {
                let mut mixed = u64::from(ctx[0]).wrapping_mul(meta.multipliers[0]);
                for (j, t) in ctx.iter().enumerate().take(n).skip(1) {
                    mixed ^= u64::from(*t).wrapping_mul(meta.multipliers[j]);
                }
                for g in 0..meta.heads_per_ngram {
                    let h = (n - 2) * meta.heads_per_ngram + g;
                    out.push((mixed % meta.head_vocab[h] + meta.head_offsets[h]) as u32);
                }
            }
        }
        let keep = window.len() - back;
        self.prev = window[keep..].to_vec();
        self.seen += tokens.len();
        out
    }

    pub fn reset(&mut self, meta: &NgramMeta) {
        *self = Self::new(meta);
    }
}

/// The table of rows, read in place from the file's mapping.
pub struct NgramTable {
    table: QTensor,
    row_dim: usize,
    row_bytes: usize,
    dtype: crate::tensor::quantized::GgmlDType,
}

impl NgramTable {
    /// `table` has shape `[rows, row_dim]` (the file's `[row_dim, rows]` reversed) and holds
    /// at least `meta.rows()` rows.
    pub fn new(table: QTensor, meta: &NgramMeta) -> Result<Self> {
        let dims = table.shape().dims().to_vec();
        if dims.len() != 2 || dims[1] != meta.row_dim || (dims[0] as u64) < meta.rows() {
            return Err(crate::tensor::Error::msg(format!(
                "n-gram table {dims:?} does not hold {} rows of {}",
                meta.rows(),
                meta.row_dim
            )));
        }
        let dtype = table.dtype();
        let row_bytes = meta.row_dim / dtype.block_size() * dtype.type_size();
        if !meta.row_dim.is_multiple_of(dtype.block_size()) {
            return Err(crate::tensor::Error::msg(
                "n-gram row is not a whole number of blocks",
            ));
        }
        Ok(Self {
            table,
            row_dim: meta.row_dim,
            row_bytes,
            dtype,
        })
    }

    /// Bytes of one row as stored.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The rows `rows`, dequantised and concatenated: `[rows.len() * row_dim]`.
    pub fn gather(&self, rows: &[u32]) -> Result<Vec<f32>> {
        let data = self.table.data()?;
        let mut out = vec![0f32; rows.len() * self.row_dim];
        for (r, y) in rows.iter().zip(out.chunks_exact_mut(self.row_dim)) {
            let at = *r as usize * self.row_bytes;
            let bytes = data.get(at..at + self.row_bytes).ok_or_else(|| {
                crate::tensor::Error::msg(format!("n-gram row {r} is outside the table"))
            })?;
            crate::tensor::quant_cpu::to_float_bytes(self.dtype, bytes, y)?;
        }
        Ok(out)
    }
}

/// The layer that adds the gated n-gram features to every stream of the residual.
pub struct PleBlock {
    key: QMatMul,
    value: QMatMul,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    /// `[channels, taps]` F32.
    conv_w: Tensor,
    taps: usize,
    dilation: usize,
    streams: usize,
    dim: usize,
    eps: f64,
    /// The last `(taps - 1) * dilation` normalised rows, `[history, channels]` F32.
    state: Option<Tensor>,
}

impl PleBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key: QMatMul,
        value: QMatMul,
        norm_key: Tensor,
        norm_query: Tensor,
        norm_conv: Tensor,
        conv_w: Tensor,
        meta: &NgramMeta,
        streams: usize,
        dim: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let flat = |t: Tensor| -> Result<Tensor> {
            t.to_dtype(DType::F32)?.flatten_all()?.to_device(device)
        };
        let conv_w = conv_w
            .to_dtype(DType::F32)?
            .contiguous()?
            .to_device(device)?;
        if conv_w.dims() != [streams * dim, meta.conv_kernel] {
            return Err(crate::tensor::Error::msg(format!(
                "n-gram convolution {:?} does not match {streams} x {dim} channels of {} taps",
                conv_w.dims(),
                meta.conv_kernel
            )));
        }
        Ok(Self {
            key,
            value,
            norm_key: flat(norm_key)?,
            norm_query: flat(norm_query)?,
            norm_conv: flat(norm_conv)?,
            conv_w,
            taps: meta.conv_kernel,
            dilation: meta.ngram_size,
            streams,
            dim,
            eps,
            state: None,
        })
    }

    pub fn reset(&mut self) {
        self.state = None;
    }

    /// The features for the rows of `r` `[1, seq, streams * dim]`, from the gathered
    /// embeddings `emb` `[1, seq, heads * row_dim]`; the caller adds them to `r`. The
    /// convolution carries its history across calls and starts empty at position 0.
    pub fn forward(&mut self, r: &Tensor, emb: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b, seq, wide) = r.dims3()?;
        let (s, d) = (self.streams, self.dim);
        let emb = emb.to_device(&r.device())?.to_dtype(DType::F32)?;
        let key = grouped_rms(&self.key.forward(&emb)?, &self.norm_key, s, d, self.eps)?;
        let value = self.value.forward(&emb)?.reshape((b, seq, 1, d))?;
        let query = grouped_rms(r, &self.norm_query, s, d, self.eps)?;
        let dot =
            (key.reshape((b, seq, s, d))? * query.reshape((b, seq, s, d))?)?.sum(D::Minus1)?;
        let dot = (dot / (d as f64).sqrt())?.unsqueeze(3)?;
        // sign(x) * sqrt(max(|x|, tiny)), then the sigmoid
        let abs = dot.abs()?;
        // max(|x|, tiny) as relu(|x| - tiny) + tiny, and sign(x) as x / |x| (zero at zero)
        let mag = ((abs.clone() - 1e-6)?.relu()? + 1e-6)?.sqrt()?;
        let sign = dot.broadcast_div(&(abs + 1e-30)?)?;
        let gate = crate::tensor::ops::sigmoid(&(sign * mag)?)?;
        let gated = gate.broadcast_mul(&value)?.reshape((b, seq, wide))?;
        let normed = grouped_rms(&gated, &self.norm_conv, s, d, self.eps)?.reshape((seq, wide))?;
        let hist = (self.taps - 1) * self.dilation;
        let past = match (index_pos, self.state.take()) {
            (p, Some(st)) if p > 0 => st,
            _ => Tensor::zeros_on((hist, wide), DType::F32, &r.device())?,
        };
        let padded = Tensor::cat(&[&past, &normed], 0)?;
        let mut acc: Option<Tensor> = None;
        for k in 0..self.taps {
            let rows = padded.narrow(0, k * self.dilation, seq)?;
            let w = self.conv_w.narrow(1, k, 1)?.reshape((1, wide))?;
            let term = rows.broadcast_mul(&w)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        self.state = Some(padded.narrow(0, seq, hist)?.contiguous()?);
        let conv = crate::tensor::ops::silu(&acc.expect("a convolution has taps"))?
            .reshape((b, seq, wide))?;
        gated + conv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{Content, Value, VersionedMagic};
    use std::collections::HashMap;

    fn content(heads: usize) -> Content {
        let mut md = HashMap::new();
        let mut put = |k: &str, v: Value| {
            md.insert(format!("a.{k}"), v);
        };
        put("ple.layers", Value::Array(vec![Value::U32(1)]));
        put("ple.ngram_size", Value::U32(3));
        put("ple.heads_per_ngram", Value::U32(2));
        put("ple.conv_kernel", Value::U32(4));
        put("ple.eos_token_id", Value::U32(7));
        put("embedding_length_per_layer_input", Value::U32(8));
        let arr = |v: Vec<u64>| Value::Array(v.into_iter().map(Value::U64).collect());
        put("ple.layer_multipliers", arr(vec![3, 5, 7]));
        put(
            "ple.head_offsets",
            arr((0..heads as u64).map(|h| h * 100).collect()),
        );
        put("ple.head_vocab_sizes", arr(vec![101; heads]));
        Content {
            magic: VersionedMagic::GgufV3,
            metadata: md,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    fn small_meta() -> NgramMeta {
        let mut m = NgramMeta::from_gguf(&content(4), "a").unwrap();
        m.head_vocab = vec![101, 103, 107, 109];
        m.head_offsets = vec![0, 200, 400, 600];
        m
    }

    #[test]
    fn a_row_is_the_hash_of_the_token_and_its_predecessors() {
        let m = small_meta();
        let mut h = NgramHistory::new(&m);
        let rows = h.rows(&m, &[10, 20]);
        // second token: bigram (20*3 ^ 10*5), trigram (20*3 ^ 10*5 ^ eos*7)
        let bi = (20u64 * 3) ^ (10 * 5);
        let tri = bi ^ (7u64 * 7);
        assert_eq!(rows[4], (bi % 101) as u32);
        assert_eq!(rows[5], (bi % 103 + 200) as u32);
        assert_eq!(rows[6], (tri % 107 + 400) as u32);
        assert_eq!(rows[7], (tri % 109 + 600) as u32);
        assert_eq!(h.seen, 2);
    }

    #[test]
    fn the_history_carries_across_calls_and_an_eos_cuts_the_window() {
        let m = small_meta();
        let mut whole = NgramHistory::new(&m);
        let all = whole.rows(&m, &[5, 6, 7, 7, 9]);
        let mut split = NgramHistory::new(&m);
        let mut parts = split.rows(&m, &[5, 6]);
        parts.extend(split.rows(&m, &[7, 7, 9]));
        assert_eq!(all, parts);
        // after an end-of-sequence token the next one hashes as if the sequence began there
        let mut cut = NgramHistory::new(&m);
        let rows = cut.rows(&m, &[1, 2, 7, 4]);
        let mut fresh = NgramHistory::new(&m);
        let fresh_rows = fresh.rows(&m, &[7, 4]);
        assert_eq!(&rows[3 * 4..], &fresh_rows[4..]);
    }

    #[test]
    fn the_heads_and_rows_follow_the_arrays() {
        let m = NgramMeta::from_gguf(&content(4), "a").unwrap();
        assert_eq!((m.layer, m.heads(), m.rows()), (1, 4, 401));
    }

    #[test]
    fn arrays_shorter_than_the_heads_are_refused() {
        assert_eq!(NgramMeta::from_gguf(&content(3), "a"), None);
    }
}
