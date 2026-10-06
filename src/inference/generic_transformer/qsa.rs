//! Block-sparse attention: a small indexer scores blocks of cached tokens and the attention of
//! the layer reads only the best of them, plus the tokens of the newest, incomplete block.
//!
//! The indexer keeps one key row per token. A complete block of `block` tokens is scored by one
//! pooled key, the mean of its rows after normalisation and rotation at the block's first
//! position; a query keeps the `budget_blocks` best blocks. A context of at most `top_k` tokens
//! keeps every block, so the attention there is dense.

use super::config::{meta_u32, meta_u32_array};
use super::{gguf_file, QMatMul, Result, Tensor};
use crate::tensor::DType;

/// The indexer's dimensions, from the file's `attention.indexer.*` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerMeta {
    /// Query heads of the indexer; the key is shared by all of them.
    pub heads: usize,
    /// Width of an indexer head.
    pub head_dim: usize,
    /// Tokens an attention layer may read beyond the incomplete block.
    pub top_k: usize,
    /// Tokens pooled into one scored block.
    pub block: usize,
}

impl IndexerMeta {
    /// The metadata of `arch`; `None` when the file has no indexer or its block size does not
    /// divide the token budget.
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> Option<Self> {
        let heads = meta_u32(content, arch, "attention.indexer.head_count")? as usize;
        let head_dim = meta_u32(content, arch, "attention.indexer.key_length")? as usize;
        let top_k = meta_u32(content, arch, "attention.indexer.top_k")? as usize;
        let block = meta_u32_array(content, arch, "attention.compress_ratios")?
            .into_iter()
            .find(|&r| r > 0)? as usize;
        (heads > 0 && head_dim > 0 && block > 1 && top_k.is_multiple_of(block) && top_k >= block)
            .then_some(Self {
                heads,
                head_dim,
                top_k,
                block,
            })
    }

    /// Blocks a query keeps.
    pub fn budget_blocks(&self) -> usize {
        self.top_k / self.block
    }
}

/// Half-split rotary embedding over the first `dim` lanes of a row, computed on the host in
/// f32 from the layer's frequency base.
#[derive(Debug, Clone)]
pub struct HostRope {
    inv_freq: Vec<f32>,
}

impl HostRope {
    /// The rotary table of `dim` lanes (even) at frequency base `base`.
    pub fn new(dim: usize, base: f32) -> Self {
        let inv_freq = (0..dim / 2)
            .map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32))
            .collect();
        Self { inv_freq }
    }

    /// Rotate the first `2 * inv_freq.len()` lanes of `row` for position `pos`; the lanes
    /// beyond are untouched.
    pub fn apply(&self, row: &mut [f32], pos: usize) {
        let half = self.inv_freq.len();
        for (i, f) in self.inv_freq.iter().enumerate() {
            let (sin, cos) = (pos as f32 * f).sin_cos();
            let (a, b) = (row[i], row[i + half]);
            row[i] = a * cos - b * sin;
            row[i + half] = b * cos + a * sin;
        }
    }
}

/// RMS-normalise `row` in place and scale it by `gamma`.
pub fn rms_norm_row(row: &mut [f32], gamma: &[f32], eps: f32) {
    let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for (v, g) in row.iter_mut().zip(gamma) {
        *v = *v * inv * g;
    }
}

/// One attention layer's indexer cache: the raw key row of every token and the pooled key of
/// every complete block.
#[derive(Debug, Clone, Default)]
pub struct IndexerCache {
    raw: Vec<f32>,
    pooled: Vec<f32>,
    tokens: usize,
}

impl IndexerCache {
    /// Tokens held.
    pub fn len(&self) -> usize {
        self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Append the raw key rows `rows` (`[n, head_dim]`); each block they complete is pooled:
    /// the mean of its rows, normalised by `k_gamma`, rotated at the position of its first
    /// token.
    pub fn append(
        &mut self,
        rows: &[f32],
        meta: &IndexerMeta,
        k_gamma: &[f32],
        eps: f32,
        rope: &HostRope,
    ) {
        let hd = meta.head_dim;
        for row in rows.chunks_exact(hd) {
            self.raw.extend_from_slice(row);
            self.tokens += 1;
            if self.tokens.is_multiple_of(meta.block) {
                let first = self.tokens - meta.block;
                let mut pooled = vec![0f32; hd];
                for t in first..self.tokens {
                    for (p, v) in pooled.iter_mut().zip(&self.raw[t * hd..(t + 1) * hd]) {
                        *p += v;
                    }
                }
                for p in pooled.iter_mut() {
                    *p /= meta.block as f32;
                }
                rms_norm_row(&mut pooled, k_gamma, eps);
                rope.apply(&mut pooled, first);
                self.pooled.extend_from_slice(&pooled);
            }
        }
    }

    /// The score of each block complete by `pos` for the query heads `q`: the rectified dot
    /// products summed over the heads, scaled by `1 / sqrt(head_dim)`.
    pub fn block_scores(&self, q: &[f32], pos: usize, meta: &IndexerMeta) -> Vec<f32> {
        let hd = meta.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        (0..(pos + 1) / meta.block)
            .map(|b| {
                let key = &self.pooled[b * hd..(b + 1) * hd];
                let score: f32 = q
                    .chunks_exact(hd)
                    .map(|h| h.iter().zip(key).map(|(x, y)| x * y).sum::<f32>().max(0.0))
                    .sum();
                score * scale
            })
            .collect()
    }

    /// The tokens a query at position `pos` attends to, ascending: the best
    /// `budget_blocks` of the blocks complete by `pos`, and every token after them. `q` is
    /// the query's indexer heads `[heads, head_dim]`, already normalised and rotated. A
    /// context that fits the budget keeps every token. Blocks that score alike are taken
    /// lowest first.
    pub fn select(&self, q: &[f32], pos: usize, meta: &IndexerMeta) -> Vec<u32> {
        let visible = pos + 1;
        let complete = visible / meta.block;
        let mut kept: Vec<u32> = Vec::new();
        if complete > 0 {
            let mut scored: Vec<(f32, usize)> = self
                .block_scores(q, pos, meta)
                .into_iter()
                .enumerate()
                .map(|(b, s)| (s, b))
                .collect();
            let take = complete.min(meta.budget_blocks());
            if take < complete {
                scored.select_nth_unstable_by(take - 1, |a, b| {
                    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
                });
                scored.truncate(take);
            }
            let mut blocks: Vec<usize> = scored.into_iter().map(|(_, b)| b).collect();
            blocks.sort_unstable();
            for b in blocks {
                kept.extend((b * meta.block..(b + 1) * meta.block).map(|t| t as u32));
            }
        }
        kept.extend((complete * meta.block..visible).map(|t| t as u32));
        kept
    }
}

/// An attention layer's indexer: its projections, norms and cache.
pub struct SparseAttention {
    pub meta: IndexerMeta,
    q_proj: QMatMul,
    k_proj: QMatMul,
    q_gamma: Vec<f32>,
    k_gamma: Vec<f32>,
    eps: f32,
    rope: HostRope,
    pub cache: IndexerCache,
}

impl SparseAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        meta: IndexerMeta,
        q_proj: QMatMul,
        k_proj: QMatMul,
        q_gamma: Vec<f32>,
        k_gamma: Vec<f32>,
        eps: f32,
        rope_dim: usize,
        rope_base: f32,
    ) -> Self {
        let rope = HostRope::new(rope_dim.min(meta.head_dim), rope_base);
        Self {
            meta,
            q_proj,
            k_proj,
            q_gamma,
            k_gamma,
            eps,
            rope,
            cache: IndexerCache::default(),
        }
    }

    /// Tokens held in the cache: every attended token is observed once.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Whether a context of `total` tokens is read whole: every block fits the budget.
    pub fn dense_for(&self, total: usize) -> bool {
        total < self.meta.top_k + self.meta.block
    }

    /// Take in the rows `x` `[1, seq, hidden]` starting at position `index_pos`: their raw
    /// keys join the cache. When the context they reach no longer fits the budget, the
    /// tokens each row attends to; `None` while attention is dense. The rows are independent
    /// of one another once the keys are in, so they are selected across the host's cores.
    pub fn observe(&mut self, x: &Tensor, index_pos: usize) -> Result<Option<Vec<Vec<u32>>>> {
        use rayon::prelude::*;
        let Some(mut q) = self.take_in(x, index_pos)? else {
            return Ok(None);
        };
        let m = self.meta;
        let row = m.heads * m.head_dim;
        let (cache, rope, gamma, eps) = (&self.cache, &self.rope, &self.q_gamma, self.eps);
        let out = q
            .par_chunks_exact_mut(row)
            .enumerate()
            .map(|(i, rows)| {
                for head in rows.chunks_exact_mut(m.head_dim) {
                    rms_norm_row(head, gamma, eps);
                    rope.apply(head, index_pos + i);
                }
                cache.select(rows, index_pos + i, &m)
            })
            .collect();
        Ok(Some(out))
    }

    /// [`Self::observe`], handing each row's block scores to `on_scores` as `(row, scores)`.
    pub fn observe_scored(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        mut on_scores: impl FnMut(usize, &[f32]),
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let Some(mut q) = self.take_in(x, index_pos)? else {
            return Ok(None);
        };
        let m = self.meta;
        let row = m.heads * m.head_dim;
        let mut out = Vec::with_capacity(q.len() / row);
        for (i, rows) in q.chunks_exact_mut(row).enumerate() {
            for head in rows.chunks_exact_mut(m.head_dim) {
                rms_norm_row(head, &self.q_gamma, self.eps);
                self.rope.apply(head, index_pos + i);
            }
            on_scores(i, &self.cache.block_scores(rows, index_pos + i, &m));
            out.push(self.cache.select(rows, index_pos + i, &m));
        }
        Ok(Some(out))
    }

    /// The rows' keys into the cache; the rows' query heads, unnormalised, when the context
    /// they reach no longer fits the budget.
    fn take_in(&mut self, x: &Tensor, index_pos: usize) -> Result<Option<Vec<f32>>> {
        let (_, seq, _) = x.dims3()?;
        // The projections are dense float weights: they read the activation in F32.
        let x = &x.to_dtype(DType::F32)?;
        let m = self.meta;
        if index_pos == 0 {
            self.cache.clear();
        }
        if self.cache.len() != index_pos {
            return Err(crate::tensor::Error::msg(format!(
                "indexer holds {} tokens but the forward starts at {index_pos}",
                self.cache.len()
            )));
        }
        let host =
            |t: Tensor| -> Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };
        let keys = host(self.k_proj.forward(x)?)?;
        self.cache
            .append(&keys, &m, &self.k_gamma, self.eps, &self.rope);
        if self.dense_for(index_pos + seq) {
            return Ok(None);
        }
        Ok(Some(host(self.q_proj.forward(x)?)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{Content, Value, VersionedMagic};
    use std::collections::HashMap;

    fn content(top_k: u32, ratios: &[u32]) -> Content {
        let mut metadata = HashMap::new();
        for (k, v) in [
            ("attention.indexer.head_count", 4),
            ("attention.indexer.key_length", 128),
            ("attention.indexer.top_k", top_k),
        ] {
            metadata.insert(format!("a.{k}"), Value::U32(v));
        }
        metadata.insert(
            "a.attention.compress_ratios".to_string(),
            Value::Array(ratios.iter().map(|&r| Value::I32(r as i32)).collect()),
        );
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    #[test]
    fn the_block_is_the_first_nonzero_ratio() {
        let m = IndexerMeta::from_gguf(&content(2048, &[0, 0, 0, 4, 0, 4]), "a").unwrap();
        assert_eq!((m.heads, m.head_dim, m.top_k, m.block), (4, 128, 2048, 4));
        assert_eq!(m.budget_blocks(), 512);
    }

    fn meta() -> IndexerMeta {
        IndexerMeta {
            heads: 1,
            head_dim: 2,
            top_k: 8,
            block: 4,
        }
    }

    #[test]
    fn a_context_within_the_budget_keeps_every_token() {
        let m = meta();
        let rope = HostRope::new(2, 10000.0);
        let mut cache = IndexerCache::default();
        let rows: Vec<f32> = (0..2 * 15).map(|i| (i as f32 * 0.3).sin()).collect();
        cache.append(&rows, &m, &[1.0, 1.0], 1e-6, &rope);
        // 15 tokens: three complete blocks and a tail of three, within a budget of two
        // blocks only the best two blocks survive; at 8 tokens everything does.
        let kept = cache.select(&[1.0, 0.5], 7, &m);
        assert_eq!(kept, (0..8).collect::<Vec<u32>>());
        let kept = cache.select(&[1.0, 0.5], 14, &m);
        assert_eq!(kept.len(), 2 * 4 + 3);
        assert_eq!(&kept[kept.len() - 3..], &[12, 13, 14]);
    }

    #[test]
    fn the_pooled_key_is_the_normalised_mean_rotated_at_the_block_start() {
        let m = meta();
        let rope = HostRope::new(2, 10000.0);
        let mut cache = IndexerCache::default();
        let rows = [1.0f32, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0];
        cache.append(&rows, &m, &[1.0, 1.0], 1e-6, &rope);
        // mean (2, 0) -> rms 1.414 -> (1.414, 0) -> rotated at position 0: unchanged.
        assert!((cache.pooled[0] - 2f32.sqrt()).abs() < 1e-5 && cache.pooled[1].abs() < 1e-6);
    }

    #[test]
    fn a_block_that_does_not_divide_the_budget_is_refused() {
        assert_eq!(IndexerMeta::from_gguf(&content(2050, &[0, 4]), "a"), None);
        assert_eq!(IndexerMeta::from_gguf(&content(2048, &[0, 0]), "a"), None);
        assert_eq!(IndexerMeta::from_gguf(&content(2048, &[0, 1]), "a"), None);
    }

    /// The rows selected across the cores are the rows selected one at a time.
    #[test]
    fn selecting_the_rows_across_cores_matches_selecting_them_in_turn() {
        use crate::tensor::quantized::{GgmlDType, QTensor};
        use crate::tensor::Device;
        let (hidden, heads, head_dim, seq) = (16usize, 2usize, 8usize, 96usize);
        let dev = Device::Cpu;
        let weights = |rows: usize, seed: usize| {
            let v: Vec<f32> = (0..rows * hidden)
                .map(|i| (((i * 37 + seed * 11) % 101) as f32 - 50.0) * 0.02)
                .collect();
            let t = Tensor::from_vec(v, (rows, hidden), &dev).unwrap();
            QMatMul::from_qtensor(QTensor::quantize_onto(&t, GgmlDType::F32, &dev).unwrap())
                .unwrap()
        };
        let make = || {
            SparseAttention::new(
                IndexerMeta {
                    heads,
                    head_dim,
                    top_k: 16,
                    block: 4,
                },
                weights(heads * head_dim, 1),
                weights(head_dim, 2),
                vec![1.0; head_dim],
                vec![1.0; head_dim],
                1e-6,
                head_dim,
                10000.0,
            )
        };
        let xs: Vec<f32> = (0..seq * hidden)
            .map(|i| (((i * 53 + 7) % 89) as f32 - 44.0) * 0.03)
            .collect();
        let x = Tensor::from_vec(xs, (1, seq, hidden), &dev).unwrap();
        let (mut across, mut in_turn) = (make(), make());
        let wide = across.observe(&x, 0).unwrap().unwrap();
        let serial = in_turn.observe_scored(&x, 0, |_, _| {}).unwrap().unwrap();
        assert_eq!(wide.len(), seq);
        assert_eq!(wide, serial);
        assert!(
            wide.last().unwrap().len() < seq,
            "the context must be past the budget for the test to select anything"
        );
    }
}
