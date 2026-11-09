//! The engram kind of the n-gram layer: hashed n-gram rows added to the residual at several
//! layers, beside the single-layer kind of [`super::PleBlock`].
//!
//! A position is hashed as every n-gram from the 2-gram to the `max_ngram`-gram ending there.
//! Each n-gram size is split over `heads` heads.
//! Every (n-gram size, head) column owns a prime-sized bucket range of the layer's table.
//! The hash runs over compressed token ids: tokens that normalise alike share 1 id.
//! Slots before the sequence start take the compressed id of the pad token.
//!
//! The gathered rows are projected by a fused `wkv` into 1 key per stream and 1 shared value.
//! Each stream adds the value under a gate.
//! The gate is the sigmoid of the signed square root of the normalised dot product of the
//! stream, its gate weight and its key.
//! The layer has no convolution.

use crate::tensor::blockscaled::dequant::{e4m3_to_f32, e8m0_to_f32};
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::mapped::MappedBytes;
use crate::tensor::quant_view::rows_view;
use crate::tensor::quantized::gguf_file;
use crate::tensor::quantized::{QTensor, BLOCK_ALIGN};
use crate::tensor::{DType, Device, Error, Result, Tensor};
use std::collections::VecDeque;
use std::sync::Arc;

/// Below this magnitude the gate's signed square root is taken at the floor.
const GATE_DOT_FLOOR: f32 = 1e-6;

/// The metadata keys of one engram layout, each under the architecture prefix.
pub(crate) struct EngramKeys {
    pub layer_ids: &'static str,
    pub heads: &'static str,
    pub head_dim: &'static str,
    pub max_ngram: &'static str,
    pub multipliers: &'static str,
    pub primes: &'static str,
    pub offsets: &'static str,
    pub token_map: &'static str,
    pub pad_id: &'static str,
}

/// The `engram.*` layout: the offsets of a layer count from its own table's first row.
const ENGRAM_KEYS: EngramKeys = EngramKeys {
    layer_ids: "engram.layer_ids",
    heads: "engram.head_count",
    head_dim: "engram.key_length",
    max_ngram: "engram.max_ngram_size",
    multipliers: "engram.multipliers",
    primes: "engram.primes",
    offsets: "engram.offsets",
    token_map: "engram.token_map",
    pad_id: "engram.pad_id",
};

/// The `ple.*` layout: every layer's offsets count from the first row of 1 shared table.
pub(crate) const PLE_KEYS: EngramKeys = EngramKeys {
    layer_ids: "ple.layers",
    heads: "ple.heads_per_ngram",
    head_dim: "embedding_length_per_layer_input",
    max_ngram: "ple.ngram_size",
    multipliers: "ple.layer_multipliers",
    primes: "ple.head_vocab_sizes",
    offsets: "ple.head_offsets",
    token_map: "engram.token_map",
    pad_id: "engram.pad_token_id",
};

/// The hash layout and parameters of every engram layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramMeta {
    /// The layers an engram joins the residual before.
    pub layer_ids: Vec<usize>,
    /// Heads of each n-gram size.
    pub heads: usize,
    /// Width of the row one column gathers.
    pub head_dim: usize,
    /// Longest n-gram hashed.
    pub max_ngram: usize,
    /// Token id to compressed id, over the whole vocabulary.
    pub token_map: Vec<u32>,
    /// The compressed id a slot before the sequence start takes.
    pub pad_id: u32,
    /// Per engram layer, 1 multiplier per look-back, `max_ngram` of them.
    pub multipliers: Vec<Vec<i64>>,
    /// Per engram layer, the bucket modulus of each column, n-gram size major.
    pub primes: Vec<Vec<u64>>,
    /// Per engram layer, the first table row of each column's bucket range.
    pub offsets: Vec<Vec<u64>>,
}

impl EngramMeta {
    /// The layout of `arch` under the `engram.*` keys or the `ple.*` ones.
    /// `None` when the file has no engram layer or omits a hash parameter or the token map.
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> Option<Self> {
        Self::from_metadata(&content.metadata, arch)
    }

    /// The layout of `arch` read from the key table `md` of a file.
    pub fn from_metadata(
        md: &std::collections::HashMap<String, gguf_file::Value>,
        arch: &str,
    ) -> Option<Self> {
        Self::read_keys(md, arch, &ENGRAM_KEYS).or_else(|| Self::read_keys(md, arch, &PLE_KEYS))
    }

    fn read_keys(
        md: &std::collections::HashMap<String, gguf_file::Value>,
        arch: &str,
        keys: &EngramKeys,
    ) -> Option<Self> {
        let g = |k: &str| md.get(&format!("{arch}.{k}"));
        let u = |k: &str| g(k).and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let arr_u64 = |k: &str| -> Option<Vec<u64>> {
            g(k)?
                .to_vec()
                .ok()?
                .iter()
                .map(|v| v.to_u64().ok())
                .collect()
        };
        let layer_ids: Vec<usize> = arr_u64(keys.layer_ids)?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        if layer_ids.is_empty() {
            return None;
        }
        let (heads, head_dim, max_ngram) = (u(keys.heads)?, u(keys.head_dim)?, u(keys.max_ngram)?);
        let n_cols = (max_ngram.checked_sub(1)?) * heads;
        let n = layer_ids.len();
        let split = |v: Vec<u64>, per: usize| -> Option<Vec<Vec<u64>>> {
            (v.len() == n * per).then(|| v.chunks(per).map(|c| c.to_vec()).collect())
        };
        // Written unsigned, each fits a signed product of 64 bits with any compressed id.
        let multipliers: Vec<i64> = arr_u64(keys.multipliers)?
            .into_iter()
            .map(i64::try_from)
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        let multipliers = (multipliers.len() == n * max_ngram)
            .then(|| multipliers.chunks(max_ngram).map(|c| c.to_vec()).collect())?;
        let primes = split(arr_u64(keys.primes)?, n_cols)?;
        let offsets = split(arr_u64(keys.offsets)?, n_cols)?;
        let token_map: Vec<u32> = arr_u64(keys.token_map)?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let pad_id = *token_map.get(u(keys.pad_id)?)?;
        Some(Self {
            layer_ids,
            heads,
            head_dim,
            max_ngram,
            token_map,
            pad_id,
            multipliers,
            primes,
            offsets,
        })
    }

    /// The layout derived from a checkpoint's reference arguments and its compressed token map.
    ///
    /// The multipliers come from the per-layer seeded generator of [`super::engram_hash`].
    /// The bucket primes and offsets come from the engram vocabulary size.
    /// `None` when an argument is missing or the map's id count differs from the arguments'.
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
        let (heads, head_dim, max_ngram) = (
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
            super::engram_hash::primes_and_offsets(layer_ids.len(), max_ngram, heads, vocab);
        Some(Self {
            layer_ids,
            heads,
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
        (self.max_ngram - 1) * self.heads
    }

    /// The index of layer `layer` among the engram layers, `None` for a layer without one.
    pub fn which(&self, layer: usize) -> Option<usize> {
        self.layer_ids.iter().position(|&l| l == layer)
    }

    /// Compressed id of a token.
    /// An id past the map hashes as the pad.
    pub fn compress(&self, token: u32) -> u32 {
        self.token_map
            .get(token as usize)
            .copied()
            .unwrap_or(self.pad_id)
    }

    /// The table rows of 1 position at engram layer `which`, an index into `layer_ids`.
    ///
    /// `recent` holds the compressed ids ending at the position, newest last.
    /// The (i+1)-gram hash is the XOR of the first i+1 ids, each times its multiplier.
    /// Each column reduces the hash into its own bucket range.
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
            for h in 0..self.heads {
                let col = (shift - 1) * self.heads + h;
                let p = self.primes[which][col];
                out.push((rolling as u64) % p + self.offsets[which][col]);
            }
        }
        out
    }
}

/// The compressed ids a sequence has seen, the last `max_ngram` of them, oldest first.
/// Every engram layer of a sequence hashes from the same history.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct EngramHistory {
    recent: VecDeque<u32>,
}

impl EngramHistory {
    /// A history holding exactly `recent`, oldest first.
    pub fn from_recent(recent: impl IntoIterator<Item = u32>) -> Self {
        Self {
            recent: recent.into_iter().collect(),
        }
    }

    /// The compressed ids held, oldest first.
    pub fn recent(&self) -> Vec<u32> {
        self.recent.iter().copied().collect()
    }

    /// Record `token` and return the ids the hash reads at its position, newest last.
    pub fn push(&mut self, meta: &EngramMeta, token: u32) -> Vec<u32> {
        self.recent.push_back(meta.compress(token));
        while self.recent.len() > meta.max_ngram {
            self.recent.pop_front();
        }
        self.recent.iter().copied().collect()
    }

    /// The table rows of every position of `tokens` at engram layer `which`, `tokens`
    /// continuing this history.
    /// The history itself does not move.
    pub fn hashes(&self, meta: &EngramMeta, which: usize, tokens: &[u32]) -> Vec<Vec<u64>> {
        let mut st = self.clone();
        tokens
            .iter()
            .map(|&t| {
                let recent = st.push(meta, t);
                meta.hash(which, &recent)
            })
            .collect()
    }

    /// Move the history past `tokens`.
    pub fn advance(&mut self, meta: &EngramMeta, tokens: &[u32]) {
        for &t in tokens {
            self.push(meta, t);
        }
    }
}

/// The table an engram layer reads rows from.
pub enum EngramTable {
    /// GGUF blocks read in place from the mapping.
    Streamed(Arc<QTensor>),
    /// e4m3 rows with 1 ue8m0 scale per `block` values, read in place from their mappings.
    Fp8Rows {
        bytes: MappedBytes,
        scales: MappedBytes,
        rows: usize,
        row_len: usize,
        block: usize,
    },
    /// A dense table in memory.
    Resident(Tensor),
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl EngramTable {
    /// Row `r`, decoded to f32.
    pub fn row(&self, r: u64) -> Result<Vec<f32>> {
        let r = r as usize;
        match self {
            Self::Streamed(t) => {
                // A view starts on the block alignment.
                // A row off the alignment is read within the shortest run of rows that is on it.
                let native = t.native_qtensor();
                let (rows, cols) = (native.dims[0], native.dims[1]);
                let row_bytes = cols / native.dtype.block_size() * native.dtype.type_size();
                let group = BLOCK_ALIGN / gcd(row_bytes, BLOCK_ALIGN);
                let start = r - r % group;
                let count = group.min(rows - start);
                rows_view(t, start, count)?
                    // CONVERSION: one table row is decoded from the mapped blocks per lookup.
                    .dequantize(&Device::Cpu)?
                    .narrow(0, r - start, 1)?
                    .flatten_all()?
                    // CONVERSION: a BF16 table decodes to BF16 on the host, its row is read as F32.
                    .to_dtype(DType::F32)?
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
            Self::Resident(t) => t
                .narrow(0, r, 1)?
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>(),
        }
    }

    /// Start reading rows `ids` in, where the table is mapped as fp8 rows.
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

    /// Ask the kernel to read the fp8 rows at random: a lookup reads a few dozen bytes.
    pub fn advise_random(&self) {
        if let Self::Fp8Rows { bytes, scales, .. } = self {
            bytes.advise_random();
            scales.advise_random();
        }
    }

    /// Rows the table holds.
    pub fn n_rows(&self) -> usize {
        match self {
            Self::Streamed(t) => t.shape().dims()[0],
            Self::Fp8Rows { rows, .. } => *rows,
            Self::Resident(t) => t.dims()[0],
        }
    }
}

/// One engram layer: its table, the fused key and value projection and the gate weights.
pub struct EngramBlock {
    table: EngramTable,
    /// `[dim * (streams + 1), n_cols * head_dim]`: a key per stream, then the shared value.
    wkv: Projection,
    /// `[streams][dim]`, the product of the query and key weights.
    gate: Vec<Vec<f32>>,
    dim: usize,
    streams: usize,
    eps: f32,
}

impl EngramBlock {
    /// `wkv` is `[dim * (streams + 1), n_cols * head_dim]` and `gate` `[streams, dim]`.
    pub fn new(
        table: EngramTable,
        wkv: Projection,
        gate: &Tensor,
        meta: &EngramMeta,
        streams: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Self> {
        let gate = gate
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let want = [dim * (streams + 1), meta.n_cols() * meta.head_dim];
        let gate_shaped = gate.len() == streams && gate.iter().all(|g| g.len() == dim);
        if wkv.dims() != want || !gate_shaped {
            return Err(Error::msg(format!(
                "engram: projection {:?} and gate of {} rows do not match {streams} streams of {dim}, projection {want:?}",
                wkv.dims(),
                gate.len()
            )));
        }
        Ok(Self {
            table,
            wkv,
            gate,
            dim,
            streams,
            eps,
        })
    }

    /// Bytes of the fused key and value projection as held: what every token of the layer reads.
    pub fn wkv_bytes(&self) -> usize {
        self.wkv.bytes()
    }

    /// The residual `r` `[1, seq, streams * dim]` with the gated value added to every stream.
    /// `hashes` holds the table rows of each position.
    pub fn forward(&self, r: &Tensor, hashes: &[Vec<u64>]) -> Result<Tensor> {
        let (dim, hc) = (self.dim, self.streams);
        let s = hashes.len();
        let width = self.wkv.dims()[1];
        let mut gathered = Vec::with_capacity(s * width);
        for ids in hashes {
            self.table.will_need(ids);
        }
        for ids in hashes {
            for &row in ids {
                gathered.extend(self.table.row(row)?);
            }
        }
        if gathered.len() != s * width {
            return Err(Error::msg(format!(
                "engram: {} row values gathered for {s} positions of width {width}",
                gathered.len()
            )));
        }
        let e = Tensor::from_vec(gathered, (s, width), &Device::Cpu)?;
        // [s, dim * (streams + 1)]
        let kv = self.wkv.apply_here(&e)?.to_vec2::<f32>()?;
        // [s * streams * dim]
        let mut h = r.flatten_all()?.to_vec1::<f32>()?;
        if h.len() != s * hc * dim {
            return Err(Error::msg(format!(
                "engram: a residual of {} values for {s} positions of {hc} streams of {dim}",
                h.len()
            )));
        }
        let scale = (dim as f32).powf(-0.5);
        for (t, kv_t) in kv.iter().enumerate() {
            let value = &kv_t[hc * dim..];
            for c in 0..hc {
                let key = &kv_t[c * dim..(c + 1) * dim];
                let row = &mut h[(t * hc + c) * dim..(t * hc + c + 1) * dim];
                let w = &self.gate[c];
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
        Tensor::from_vec(h, (1, s, hc * dim), &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generic_transformer::wide_sparse_frozen as frozen;
    use crate::tensor::quantized::gguf_file::{Content, Value, VersionedMagic};
    use crate::tensor::quantized::GgmlDType;
    use std::collections::HashMap;

    fn meta() -> EngramMeta {
        EngramMeta {
            layer_ids: vec![1],
            heads: 2,
            head_dim: 4,
            max_ngram: 3,
            token_map: vec![0, 1, 2, 3, 3, 5],
            pad_id: 2,
            multipliers: vec![vec![3, 5, 7]],
            primes: vec![vec![11, 13, 17, 19]],
            offsets: vec![vec![0, 11, 24, 41]],
        }
    }

    /// The fields of `m` in the order the V4.1 configuration held them, as the records hold
    /// them.
    fn fields(m: &EngramMeta) -> String {
        let all = (
            &m.layer_ids,
            m.heads,
            m.head_dim,
            m.max_ngram,
            &m.token_map,
            m.pad_id,
            &m.multipliers,
            &m.primes,
            &m.offsets,
        );
        format!("{all:?}")
    }

    /// Every column lands in its own bucket range, and the first position hashes the pad for
    /// the slots before the sequence starts.
    #[test]
    fn columns_stay_in_their_bucket_ranges_and_the_start_pads() {
        let c = meta();
        let rows = EngramHistory::default().hashes(&c, 0, &[5, 1, 3, 4]);
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

    /// Decoding token by token reads exactly the rows a prompt hashes at each position.
    #[test]
    fn decode_hashes_match_prefill() {
        let c = meta();
        let tokens = [1u32, 5, 0, 3, 2, 4, 1];
        let want = EngramHistory::default().hashes(&c, 0, &tokens);
        let mut st = EngramHistory::default();
        for (i, &t) in tokens.iter().enumerate() {
            let recent = st.push(&c, t);
            assert!(recent.len() <= c.max_ngram);
            assert_eq!(c.hash(0, &recent), want[i], "position {i}");
        }
        // A history advanced past a prefix hashes the rest as the whole prompt does.
        let mut st = EngramHistory::default();
        st.advance(&c, &tokens[..3]);
        assert_eq!(st.hashes(&c, 0, &tokens[3..]), want[3..].to_vec());
    }

    /// 2 layers with their own multipliers: the hashes, the compressed ids past the map and
    /// a continued history equal the recorded V4.1 ones.
    #[test]
    fn the_hashes_equal_the_v41_hashes() {
        let mut m = meta();
        m.layer_ids = vec![1, 4];
        m.multipliers.push(vec![11, 13, 17]);
        m.primes.push(vec![23, 29, 31, 37]);
        m.offsets.push(vec![0, 23, 52, 83]);
        // Token 9 lies past the map and hashes as the pad.
        let tokens = [4u32, 0, 9, 5, 3, 1, 2, 2];
        for which in 0..2 {
            frozen::expect_debug(
                "units",
                &format!("engram hashes {which}"),
                &EngramHistory::default().hashes(&m, which, &tokens),
            );
            let start = EngramHistory::from_recent([3, 1]);
            frozen::expect_debug(
                "units",
                &format!("engram hashes from {which}"),
                &start.hashes(&m, which, &tokens),
            );
        }
    }

    fn content(md: Vec<(&str, Value)>) -> Content {
        Content {
            magic: VersionedMagic::GgufV3,
            metadata: md
                .into_iter()
                .map(|(k, v)| (format!("a.{k}"), v))
                .collect::<HashMap<_, _>>(),
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    fn arr(v: &[u64]) -> Value {
        Value::Array(v.iter().map(|&x| Value::U64(x)).collect())
    }

    /// The layout of 2 layers under both key sets, read as the V4.1 reader read it.
    #[test]
    fn both_key_sets_read_as_the_v41_reader_reads_them() {
        let common = |keys: [&'static str; 9]| {
            content(vec![
                (keys[0], arr(&[1, 4])),
                (keys[1], Value::U32(2)),
                (keys[2], Value::U32(4)),
                (keys[3], Value::U32(3)),
                (keys[4], arr(&[3, 5, 7, 11, 13, 17])),
                (keys[5], arr(&[11, 13, 17, 19, 23, 29, 31, 37])),
                (keys[6], arr(&[0, 11, 24, 41, 0, 23, 52, 83])),
                (keys[7], arr(&[0, 1, 2, 3, 3, 5])),
                (keys[8], Value::U32(4)),
            ])
        };
        let engram = common([
            "engram.layer_ids",
            "engram.head_count",
            "engram.key_length",
            "engram.max_ngram_size",
            "engram.multipliers",
            "engram.primes",
            "engram.offsets",
            "engram.token_map",
            "engram.pad_id",
        ]);
        let ple = common([
            "ple.layers",
            "ple.heads_per_ngram",
            "embedding_length_per_layer_input",
            "ple.ngram_size",
            "ple.layer_multipliers",
            "ple.head_vocab_sizes",
            "ple.head_offsets",
            "engram.token_map",
            "engram.pad_token_id",
        ]);
        for c in [&engram, &ple] {
            let got = EngramMeta::from_gguf(c, "a").expect("a layout");
            let key = format!("engram read {}", c.metadata.len());
            frozen::expect("units", &key, &fields(&got));
            assert_eq!((got.pad_id, got.which(4), got.which(2)), (3, Some(1), None));
        }
        // Without the token map the layout is not read.
        let mut bare = ple;
        bare.metadata.remove("a.engram.token_map");
        assert_eq!(EngramMeta::from_gguf(&bare, "a"), None);
    }

    /// The layout derived from reference arguments equals the recorded V4.1 derivation, and a
    /// map of the wrong size is refused.
    #[test]
    fn the_derived_layout_equals_the_v41_derivation() {
        let args = serde_json::json!({
            "engram_layer_ids": [1, 14],
            "engram_n_heads": 2,
            "engram_head_dim": 4,
            "engram_max_ngram_size": 3,
            "engram_vocab_size": 101,
            "engram_compressed_vocab_size": 5,
            "engram_pad_id": 2,
        });
        let map = vec![0u32, 1, 2, 3, 3, 4];
        let got = EngramMeta::derive(&args, map).expect("a layout");
        frozen::expect("units", "engram derive", &fields(&got));
        assert_eq!(EngramMeta::derive(&args, vec![0, 1, 2]), None);
    }

    /// `n` values in `[-1, 1)` drawn from `seed` by a 64-bit linear congruential generator.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn mapped(bytes: &[u8]) -> MappedBytes {
        let mut m = memmap2::MmapMut::map_anon(bytes.len().max(1)).unwrap();
        m[..bytes.len()].copy_from_slice(bytes);
        MappedBytes::new(Arc::new(m.make_read_only().unwrap()), 0, bytes.len()).unwrap()
    }

    const HC: usize = 4;
    const DIM: usize = 16;
    const EPS: f32 = 1e-6;
    /// A row of Q8_0 holds whole blocks of 32.
    const HEAD_DIM: usize = 32;
    const FP8_BLOCK: usize = 16;

    /// The layout of 1 layer whose table holds `rows` rows of [`HEAD_DIM`].
    fn wide_meta(rows: u64) -> EngramMeta {
        let mut m = meta();
        m.head_dim = HEAD_DIM;
        m.primes = vec![vec![rows / 4 - 1; 4]];
        m.offsets = vec![(0..4).map(|c| c * (rows / 4)).collect()];
        m
    }

    /// The table of storage `kind`: 0 resident, 1 Q8_0 blocks, 2 fp8 rows.
    fn tables(kind: usize, rows: usize) -> EngramTable {
        let values = noise(rows * HEAD_DIM, 77);
        match kind {
            0 => EngramTable::Resident(
                Tensor::from_vec(values, (rows, HEAD_DIM), &Device::Cpu).unwrap(),
            ),
            1 => {
                let t = Tensor::from_vec(values, (rows, HEAD_DIM), &Device::Cpu).unwrap();
                EngramTable::Streamed(Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap()))
            }
            _ => {
                let bytes: Vec<u8> = (0..rows * HEAD_DIM)
                    .map(|i| (i * 37 % 0x7e) as u8)
                    .collect();
                let scales: Vec<u8> = (0..rows * HEAD_DIM / FP8_BLOCK)
                    .map(|i| (120 + i % 9) as u8)
                    .collect();
                EngramTable::Fp8Rows {
                    bytes: mapped(&bytes),
                    scales: mapped(&scales),
                    rows,
                    row_len: HEAD_DIM,
                    block: FP8_BLOCK,
                }
            }
        }
    }

    /// The layer adds the recorded bits the V4.1 engram added, from each table storage, for a
    /// prompt and for a single decoded token after it.
    #[test]
    fn the_block_gives_the_bits_of_the_v41_engram() {
        let rows = 64usize;
        let m = wide_meta(rows as u64);
        let width = m.n_cols() * HEAD_DIM;
        let wkv = Tensor::from_vec(
            noise(DIM * (HC + 1) * width, 5),
            (DIM * (HC + 1), width),
            &Device::Cpu,
        )
        .unwrap();
        let gate = Tensor::from_vec(noise(HC * DIM, 6), (HC, DIM), &Device::Cpu).unwrap();
        let tokens = [1u32, 5, 0, 3, 2, 4, 1];
        for kind in 0..3 {
            let table = tables(kind, rows);
            let n_rows = format!("{}", table.n_rows());
            frozen::expect("units", &format!("engram table {kind} rows"), &n_rows);
            let block = EngramBlock::new(
                table,
                Projection::Dense(wkv.clone()),
                &gate,
                &m,
                HC,
                DIM,
                EPS,
            )
            .unwrap();
            // The prompt, then 1 decoded token from the history the prompt left.
            let mut history = EngramHistory::default();
            for (part, seed) in [(&tokens[..6], 8u64), (&tokens[6..], 9)] {
                let s = part.len();
                let values = noise(s * HC * DIM, seed);
                let r = Tensor::from_vec(values, (1, s, HC * DIM), &Device::Cpu).unwrap();
                let hashes = history.hashes(&m, 0, part);
                let key = |what: &str| format!("engram block {kind} {s} {what}");
                frozen::expect_debug("units", &key("hashes"), &hashes);
                let got = block.forward(&r, &hashes).unwrap();
                assert_eq!(got.dims(), [1, s, HC * DIM]);
                let bits = |t: &Tensor| -> Vec<u32> {
                    t.flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect()
                };
                let out = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                frozen::expect_f32("units", &key("out"), &out);
                assert_ne!(bits(&got), bits(&r), "the layer adds nothing");
                history.advance(&m, part);
            }
        }
    }

    /// A projection that does not hold a key per stream and the value is refused.
    #[test]
    fn a_projection_of_the_wrong_height_is_refused() {
        let m = wide_meta(64);
        let width = m.n_cols() * HEAD_DIM;
        let short =
            Tensor::from_vec(noise(DIM * HC * width, 1), (DIM * HC, width), &Device::Cpu).unwrap();
        let gate = Tensor::from_vec(noise(HC * DIM, 2), (HC, DIM), &Device::Cpu).unwrap();
        let table = tables(0, 64);
        let wkv = Projection::Dense(short);
        assert!(EngramBlock::new(table, wkv, &gate, &m, HC, DIM, EPS).is_err());
    }
}
