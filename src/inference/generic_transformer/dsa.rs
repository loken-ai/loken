//! The sparse indexer of a latent attention layer, and the selection it hands to the layers
//! after it.
//!
//! The indexer keeps 1 key row per token. A query projects the layer's query factor to several
//! heads, and scores a cached token by `sum over heads of w_h * relu(q_h . k * d^-1/2)`, where
//! `w_h` is a projection of the layer's input scaled by `heads^-1/2`. The query keeps the
//! `top_k` best-scored tokens among those it may see. Ties go to the lower position.
//!
//! A query that sees at most `top_k` tokens keeps all of them, so the layer attends densely and
//! the scores are not computed; the keys are still cached.
//!
//! A layer that runs no indexer of its own uses the selection of the nearest earlier layer that
//! does. The layers hand it over through a [`SelectionCell`] the model shares.

use super::mla::rotate_pairs;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// The indexer's weights, F32 on the host.
pub struct IndexerWeights {
    /// `[heads * head_dim, q_rank]`: the query heads from the query factor.
    pub wq_b: Vec<f32>,
    /// `[head_dim, hidden]`: the key from the layer's input.
    pub wk: Vec<f32>,
    /// Scale and bias of the layer norm on the key, `[head_dim]` each.
    pub k_norm_w: Vec<f32>,
    pub k_norm_b: Vec<f32>,
    /// `[heads, hidden]`: the weight of each head from the layer's input.
    pub proj: Vec<f32>,
}

/// The tokens each row of a forward attends to, where a layer chose them.
pub struct Selected {
    /// Position of the forward's first row.
    pub first: usize,
    /// Per row, the positions kept in ascending order, or `None` for every position the row
    /// may see.
    pub rows: Vec<Option<Vec<u32>>>,
}

/// Where the layers of a model leave and find the selection of the forward in progress.
#[derive(Default)]
pub struct SelectionCell(Mutex<Option<Arc<Selected>>>);

impl SelectionCell {
    pub fn put(&self, selected: Selected) -> Arc<Selected> {
        let selected = Arc::new(selected);
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(selected.clone());
        selected
    }

    /// The selection of the rows `first..first + rows`, or why a layer cannot use what is held.
    pub fn get(&self, first: usize, rows: usize) -> crate::tensor::Result<Arc<Selected>> {
        match self.0.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            Some(s) if s.first == first && s.rows.len() == rows => Ok(s.clone()),
            Some(s) => Err(crate::tensor::Error::msg(format!(
                "the held selection covers {} rows from position {}, this forward has {rows} from {first}",
                s.rows.len(),
                s.first
            ))),
            None => Err(crate::tensor::Error::msg(
                "a layer that reuses a selection ran before any layer chose one",
            )),
        }
    }
}

/// A layer's indexer.
pub struct Indexer {
    pub(super) w: IndexerWeights,
    pub(super) heads: usize,
    pub(super) head_dim: usize,
    pub(super) top_k: usize,
    pub(super) hidden: usize,
    pub(super) q_rank: usize,
    /// Epsilon of the layer norm on the keys.
    pub(super) norm_eps: f32,
}

/// `out[r] = sum_c w[r * cols + c] * x[c]` for every row.
fn matvec(w: &[f32], cols: usize, x: &[f32]) -> Vec<f32> {
    w.chunks_exact(cols)
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

impl Indexer {
    pub fn new(
        w: IndexerWeights,
        heads: usize,
        head_dim: usize,
        top_k: usize,
        hidden: usize,
        q_rank: usize,
        norm_eps: f32,
    ) -> crate::tensor::Result<Self> {
        let checks = [
            ("wq_b", w.wq_b.len(), heads * head_dim * q_rank),
            ("wk", w.wk.len(), head_dim * hidden),
            ("k_norm weight", w.k_norm_w.len(), head_dim),
            ("k_norm bias", w.k_norm_b.len(), head_dim),
            ("weights projection", w.proj.len(), heads * hidden),
        ];
        for (what, got, want) in checks {
            if got != want {
                return Err(crate::tensor::Error::msg(format!(
                    "indexer: {what} holds {got} values, the header says {want}"
                )));
            }
        }
        Ok(Self {
            w,
            heads,
            head_dim,
            top_k,
            hidden,
            q_rank,
            norm_eps,
        })
    }

    /// The key of 1 row of input `x`: projected, layer-normalised, then rotated on its first
    /// `rope` lanes at `position`.
    pub(super) fn key(&self, x: &[f32], position: usize, inv_freq: &[f32]) -> Vec<f32> {
        let mut k = matvec(&self.w.wk, self.hidden, x);
        let n = k.len() as f32;
        let mean = k.iter().sum::<f32>() / n;
        let var = k.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + self.norm_eps).sqrt();
        for ((v, w), b) in k.iter_mut().zip(&self.w.k_norm_w).zip(&self.w.k_norm_b) {
            *v = (*v - mean) * inv * w + b;
        }
        rotate_pairs(&mut k[..inv_freq.len() * 2], position, inv_freq);
        k
    }

    /// The keys of the rows of `x` `[rows, hidden]` that stand at `first..`.
    pub(super) fn keys(&self, x: &[f32], first: usize, inv_freq: &[f32]) -> Vec<f32> {
        x.par_chunks_exact(self.hidden)
            .enumerate()
            .flat_map_iter(|(t, row)| self.key(row, first + t, inv_freq))
            .collect()
    }

    /// The tokens each row of the forward keeps. `x` `[rows, hidden]` is the layer's input,
    /// `factor` `[rows, q_rank]` its normalised query factor, `keys` every cached key including
    /// the rows' own, which stand at `first..`.
    pub(super) fn select(
        &self,
        x: &[f32],
        factor: &[f32],
        keys: &[f32],
        first: usize,
        inv_freq: &[f32],
    ) -> Vec<Option<Vec<u32>>> {
        let (d, heads) = (self.head_dim, self.heads);
        let score_scale = (d as f32).powf(-0.5);
        let head_scale = (heads as f32).powf(-0.5);
        x.chunks_exact(self.hidden)
            .zip(factor.chunks_exact(self.q_rank))
            .enumerate()
            .map(|(t, (xr, fr))| {
                let pos = first + t;
                if pos < self.top_k {
                    return None;
                }
                let mut q = matvec(&self.w.wq_b, self.q_rank, fr);
                for head in q.chunks_exact_mut(d) {
                    rotate_pairs(&mut head[..inv_freq.len() * 2], pos, inv_freq);
                }
                let weights: Vec<f32> = matvec(&self.w.proj, self.hidden, xr)
                    .into_iter()
                    .map(|w| w * head_scale)
                    .collect();
                let scores: Vec<f32> = keys[..(pos + 1) * d]
                    .par_chunks_exact(d)
                    .map(|k| {
                        q.chunks_exact(d)
                            .zip(&weights)
                            .map(|(qh, w)| {
                                let s = qh.iter().zip(k).map(|(a, b)| a * b).sum::<f32>();
                                w * (s * score_scale).max(0.0)
                            })
                            .sum()
                    })
                    .collect();
                Some(top_k_of(&scores, self.top_k))
            })
            .collect()
    }
}

/// The `k` positions of the highest `scores`, a tie going to the lower position, in ascending
/// order of position.
pub fn top_k_of(scores: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..scores.len() as u32).collect();
    let better = |a: &u32, b: &u32| {
        scores[*b as usize]
            .total_cmp(&scores[*a as usize])
            .then(a.cmp(b))
    };
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, better);
        order.truncate(k);
    }
    order.sort_unstable();
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generic_transformer::mla::pair_frequencies;

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

    const HIDDEN: usize = 12;
    const Q_RANK: usize = 8;
    const HEADS: usize = 3;
    const DIM: usize = 8;

    fn indexer(top_k: usize) -> Indexer {
        Indexer::new(
            IndexerWeights {
                wq_b: values(HEADS * DIM * Q_RANK, 1, 0.6),
                wk: values(DIM * HIDDEN, 2, 0.6),
                k_norm_w: values(DIM, 3, 0.4).iter().map(|v| 1.0 + v).collect(),
                k_norm_b: values(DIM, 4, 0.4),
                proj: values(HEADS * HIDDEN, 5, 0.6),
            },
            HEADS,
            DIM,
            top_k,
            HIDDEN,
            Q_RANK,
            1e-6,
        )
        .unwrap()
    }

    #[test]
    fn the_best_positions_are_kept_and_a_tie_goes_to_the_lower_one() {
        assert_eq!(top_k_of(&[0.1, 0.9, 0.5, 0.7], 2), vec![1, 3]);
        assert_eq!(top_k_of(&[0.0, 0.0, 0.0, 0.0, 0.0], 3), vec![0, 1, 2]);
        assert_eq!(top_k_of(&[0.2, 0.9, 0.9, 0.9], 2), vec![1, 2]);
        assert_eq!(top_k_of(&[0.2, 0.3], 5), vec![0, 1]);
    }

    #[test]
    fn a_key_is_normalised_over_its_width_then_rotated() {
        let ix = indexer(4);
        let inv = pair_frequencies(4, 10000.0);
        let x = values(HIDDEN, 7, 1.0);
        let k = ix.key(&x, 0, &inv);
        // Position 0 rotates by nothing: the key is the layer norm of the projection.
        let raw = matvec(&ix.w.wk, HIDDEN, &x);
        let mean = raw.iter().sum::<f32>() / DIM as f32;
        let var = raw.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / DIM as f32;
        for i in 0..DIM {
            let want = (raw[i] - mean) / (var + 1e-6).sqrt() * ix.w.k_norm_w[i] + ix.w.k_norm_b[i];
            assert!((k[i] - want).abs() < 1e-6, "{i}: {} vs {want}", k[i]);
        }
        // Only the first lanes turn at a later position; the others are as at position 0.
        let later = ix.key(&x, 9, &inv);
        assert_eq!(&later[4..], &k[4..]);
        assert_ne!(&later[..4], &k[..4]);
    }

    /// The score of every visible token written out head by head, and the best kept.
    fn brute_force(
        ix: &Indexer,
        x: &[f32],
        factor: &[f32],
        keys: &[f32],
        inv: &[f32],
        rows: usize,
        first: usize,
    ) -> Vec<Option<Vec<u32>>> {
        (0..rows)
            .map(|t| {
                let pos = first + t;
                if pos < ix.top_k {
                    return None;
                }
                let f = &factor[t * Q_RANK..(t + 1) * Q_RANK];
                let h = &x[t * HIDDEN..(t + 1) * HIDDEN];
                let mut scores = Vec::new();
                for j in 0..=pos {
                    let mut total = 0f32;
                    for head in 0..HEADS {
                        let mut q: Vec<f32> = (0..DIM)
                            .map(|d| {
                                let row = (head * DIM + d) * Q_RANK;
                                (0..Q_RANK).map(|c| ix.w.wq_b[row + c] * f[c]).sum()
                            })
                            .collect();
                        rotate_pairs(&mut q[..4], pos, inv);
                        let dot: f32 = (0..DIM).map(|d| q[d] * keys[j * DIM + d]).sum();
                        let w: f32 = (0..HIDDEN)
                            .map(|c| ix.w.proj[head * HIDDEN + c] * h[c])
                            .sum::<f32>()
                            / (HEADS as f32).sqrt();
                        total += w * (dot / (DIM as f32).sqrt()).max(0.0);
                    }
                    scores.push(total);
                }
                let mut order: Vec<usize> = (0..scores.len()).collect();
                order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
                let mut kept: Vec<u32> = order[..ix.top_k].iter().map(|&i| i as u32).collect();
                kept.sort_unstable();
                Some(kept)
            })
            .collect()
    }

    #[test]
    fn the_selection_is_the_best_scored_tokens_of_those_a_row_may_see() {
        let ix = indexer(4);
        let inv = pair_frequencies(4, 10000.0);
        let rows = 11;
        let x = values(rows * HIDDEN, 11, 1.0);
        let factor = values(rows * Q_RANK, 12, 1.0);
        let keys = ix.keys(&x, 0, &inv);
        let got = ix.select(&x, &factor, &keys, 0, &inv);
        let want = brute_force(&ix, &x, &factor, &keys, &inv, rows, 0);
        assert_eq!(got, want);
        // The first rows see no more than the budget: they keep everything.
        assert!(got[..4].iter().all(Option::is_none));
        assert!(got[4..].iter().all(|r| r.as_ref().unwrap().len() == 4));
        // A row's tokens are causal.
        for (t, row) in got.iter().enumerate() {
            if let Some(row) = row {
                assert!(row.iter().all(|&j| j as usize <= t));
            }
        }
        // The rows of a later chunk keep the tokens they would have kept in 1 pass.
        let tail = ix.select(&x[6 * HIDDEN..], &factor[6 * Q_RANK..], &keys, 6, &inv);
        assert_eq!(tail, got[6..].to_vec());
    }

    #[test]
    fn a_cell_gives_a_forward_the_selection_of_its_own_rows_only() {
        let cell = SelectionCell::default();
        assert!(cell.get(0, 3).is_err());
        cell.put(Selected {
            first: 4,
            rows: vec![None, Some(vec![1, 2])],
        });
        assert_eq!(cell.get(4, 2).unwrap().rows[1], Some(vec![1, 2]));
        assert!(cell.get(4, 1).is_err());
        assert!(cell.get(5, 2).is_err());
    }
}
