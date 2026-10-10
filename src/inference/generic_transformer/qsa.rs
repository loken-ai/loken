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

/// One attention layer's indexer cache: the raw key rows of the block being filled and the
/// pooled key of every complete block.
#[derive(Debug, Clone, Default)]
pub struct IndexerCache {
    /// The raw rows of the incomplete block, oldest first.
    raw: Vec<f32>,
    pooled: Vec<f32>,
    /// The pooled keys again, every complete group of [`LANES`] blocks transposed to
    /// `[dim][lane]`, which scores a group with one multiply-add per dimension.
    lanes: Vec<f32>,
    tokens: usize,
}

/// Blocks scored together.
const LANES: usize = 8;

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
                for raw in self.raw.chunks_exact(hd) {
                    for (p, v) in pooled.iter_mut().zip(raw) {
                        *p += v;
                    }
                }
                for p in pooled.iter_mut() {
                    *p /= meta.block as f32;
                }
                rms_norm_row(&mut pooled, k_gamma, eps);
                rope.apply(&mut pooled, first);
                self.raw.clear();
                self.push_pooled(&pooled, hd);
            }
        }
    }

    /// A pooled key joins the blocks; each complete group of [`LANES`] is transposed into
    /// `lanes`.
    fn push_pooled(&mut self, pooled: &[f32], hd: usize) {
        self.pooled.extend_from_slice(pooled);
        let blocks = self.pooled.len() / hd;
        if blocks.is_multiple_of(LANES) {
            let group = &self.pooled[(blocks - LANES) * hd..];
            for d in 0..hd {
                self.lanes
                    .extend((0..LANES).map(|lane| group[lane * hd + d]));
            }
        }
    }

    /// Complete blocks held.
    pub fn blocks(&self, meta: &IndexerMeta) -> usize {
        self.pooled.len() / meta.head_dim
    }

    /// The pooled keys of blocks `from..`, row by row.
    pub fn pooled_from(&self, from: usize, meta: &IndexerMeta) -> &[f32] {
        &self.pooled[from * meta.head_dim..]
    }

    /// The raw rows of the block being filled.
    pub fn tail(&self) -> &[f32] {
        &self.raw
    }

    /// Take over what another holder of the same sequence has beyond this cache: the pooled keys
    /// of the blocks this one lacks (`pooled`, row by row) and the raw rows of the block being
    /// filled, which bring it to `tokens`.
    pub fn extend(&mut self, pooled: &[f32], tail: &[f32], tokens: usize, meta: &IndexerMeta) {
        let hd = meta.head_dim;
        for row in pooled.chunks_exact(hd) {
            self.push_pooled(row, hd);
        }
        self.raw = tail.to_vec();
        self.tokens = tokens;
    }

    /// The score of each block complete by `pos` for the query heads `q`: the rectified dot
    /// products summed over the heads, scaled by `1 / sqrt(head_dim)`. Groups of [`LANES`]
    /// blocks are scored together, each lane summing its dimensions in order as one block
    /// does alone, so the scores are those of [`Self::block_scores_alone`] to the bit.
    pub fn block_scores(&self, q: &[f32], pos: usize, meta: &IndexerMeta) -> Vec<f32> {
        let blocks = (pos + 1) / meta.block;
        let groups = blocks / LANES;
        let mut scores = Vec::with_capacity(blocks);
        let scaled = |scores: &mut Vec<f32>| {
            let scale = 1.0 / (meta.head_dim as f32).sqrt();
            scores.iter_mut().for_each(|s| *s *= scale);
        };
        match meta.heads {
            1 => self.score_groups::<1>(q, meta.head_dim, groups, &mut scores),
            2 => self.score_groups::<2>(q, meta.head_dim, groups, &mut scores),
            3 => self.score_groups::<3>(q, meta.head_dim, groups, &mut scores),
            4 => self.score_groups::<4>(q, meta.head_dim, groups, &mut scores),
            5 => self.score_groups::<5>(q, meta.head_dim, groups, &mut scores),
            6 => self.score_groups::<6>(q, meta.head_dim, groups, &mut scores),
            7 => self.score_groups::<7>(q, meta.head_dim, groups, &mut scores),
            8 => self.score_groups::<8>(q, meta.head_dim, groups, &mut scores),
            _ => {}
        }
        scaled(&mut scores);
        let done = scores.len();
        scores.extend(self.block_scores_alone(q, done, blocks, meta));
        scores
    }

    /// The unscaled scores of the first `groups` groups of blocks for `H` query heads.
    fn score_groups<const H: usize>(
        &self,
        q: &[f32],
        hd: usize,
        groups: usize,
        scores: &mut Vec<f32>,
    ) {
        for group in self.lanes.chunks_exact(hd * LANES).take(groups) {
            let mut dots = [[0f32; LANES]; H];
            for (d, key) in group.chunks_exact(LANES).enumerate() {
                let key: [f32; LANES] = key.try_into().expect("a lane group");
                for (h, dot) in dots.iter_mut().enumerate() {
                    let x = q[h * hd + d];
                    for lane in 0..LANES {
                        dot[lane] += x * key[lane];
                    }
                }
            }
            for lane in 0..LANES {
                scores.push(dots.iter().map(|dot| dot[lane].max(0.0)).sum());
            }
        }
    }

    /// The scores of blocks `from..to`, one block at a time.
    fn block_scores_alone(
        &self,
        q: &[f32],
        from: usize,
        to: usize,
        meta: &IndexerMeta,
    ) -> Vec<f32> {
        let hd = meta.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        (from..to)
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
    /// The cache a recorded decode step keeps on the card.
    #[cfg(feature = "cuda")]
    card: Option<CardIndexer>,
}

#[cfg(feature = "cuda")]
mod card;
#[cfg(feature = "cuda")]
pub use card::{CardIndexer, CardPick};

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
            #[cfg(feature = "cuda")]
            card: None,
        }
    }

    /// The cache as a snapshot of the sequence reads it: the host's, with what a recorded step
    /// made on the card.
    pub fn snapshot_cache(&self) -> Result<IndexerCache> {
        #[cfg(feature = "cuda")]
        return self.current_cache();
        #[cfg(not(feature = "cuda"))]
        Ok(self.cache.clone())
    }

    /// Make `cache` the host's. A restore into the buffers a recorded step lives in
    /// (`in_place`) is of the sequence the card holds, so the card keeps what the cache covers;
    /// any other is of another sequence, and the card forgets its own.
    pub fn restore_cache(&mut self, cache: &IndexerCache, in_place: bool) {
        self.cache = cache.clone();
        #[cfg(feature = "cuda")]
        match in_place {
            true => self.truncate_card(cache.len()),
            false => self.forget_card(),
        }
        #[cfg(not(feature = "cuda"))]
        let _ = in_place;
    }

    /// Forget the sequence.
    pub fn reset(&mut self) {
        self.cache.clear();
        #[cfg(feature = "cuda")]
        self.forget_card();
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
            #[cfg(feature = "cuda")]
            self.forget_card();
        }
        // Tokens a recorded step made on the card join the host's cache first.
        #[cfg(feature = "cuda")]
        self.sync_host_from_card(index_pos)?;
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
        // From here the host holds tokens the card may not.
        #[cfg(feature = "cuda")]
        self.truncate_card(index_pos);
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

    /// What a decoded token's block scoring costs at a long context, in groups and alone. Not a
    /// gate.
    #[test]
    #[ignore = "a measurement: cargo test --release scoring_cost -- --ignored --nocapture"]
    fn scoring_cost() {
        let m = IndexerMeta {
            heads: 4,
            head_dim: 128,
            top_k: 2048,
            block: 4,
        };
        let rope = HostRope::new(128, 10000.0);
        let mut cache = IndexerCache::default();
        let tokens = 12_000;
        let rows: Vec<f32> = (0..tokens * 128)
            .map(|i| ((i * 7919 % 1009) as f32 / 1009.0 - 0.5) * 2.0)
            .collect();
        cache.append(&rows, &m, &[1.0; 128], 1e-6, &rope);
        let q: Vec<f32> = (0..4 * 128)
            .map(|i| ((i * 104729 % 997) as f32 / 997.0 - 0.5) * 3.0)
            .collect();
        let time = |f: &dyn Fn() -> Vec<f32>| {
            let start = std::time::Instant::now();
            let mut sink = 0f32;
            for _ in 0..50 {
                sink += std::hint::black_box(f())[0];
            }
            (start.elapsed().as_secs_f64() / 50.0 * 1e6, sink)
        };
        let blocks = tokens / 4;
        let (grouped, _) = time(&|| cache.block_scores(&q, tokens - 1, &m));
        let (alone, _) = time(&|| cache.block_scores_alone(&q, 0, blocks, &m));
        eprintln!("{blocks} blocks: {grouped:.0} us in groups, {alone:.0} us alone");
    }

    /// Scoring blocks in groups gives, bit for bit, what scoring each alone gives.
    #[test]
    fn scoring_blocks_in_groups_is_scoring_them_alone() {
        let m = IndexerMeta {
            heads: 3,
            head_dim: 16,
            top_k: 64,
            block: 4,
        };
        let rope = HostRope::new(16, 10000.0);
        let mut cache = IndexerCache::default();
        let rows: Vec<f32> = (0..16 * 4 * 21 + 3 * 16)
            .map(|i| ((i * 7919 % 1009) as f32 / 1009.0 - 0.5) * 2.0)
            .collect();
        cache.append(&rows, &m, &[1.0; 16], 1e-6, &rope);
        let q: Vec<f32> = (0..3 * 16)
            .map(|i| ((i * 104729 % 997) as f32 / 997.0 - 0.5) * 3.0)
            .collect();
        for pos in [3usize, 31, 32, 35, 67, 82, 83, 84] {
            let grouped = cache.block_scores(&q, pos, &m);
            let blocks = (pos + 1) / m.block;
            let alone = cache.block_scores_alone(&q, 0, blocks, &m);
            assert_eq!(grouped.len(), blocks);
            for (a, b) in grouped.iter().zip(&alone) {
                assert_eq!(a.to_bits(), b.to_bits(), "pos {pos}");
            }
        }
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
