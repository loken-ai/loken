//! Latent attention on the host.
//!
//! A layer projects its input to a low-rank query factor and to 1 latent row per token. The
//! latent row is a `kv_lora_rank` vector, normalised, beside a `rope_dim` position key that is
//! rotated. The layer caches that row and nothing else: the keys and values of every head are
//! linear in the latent, `k_nope = W_UK c` and `v = W_UV c`, with both matrices cut from one
//! `kv_b` projection.
//!
//! A query attends in the latent space. Its non-rotated part is carried through `W_UK` once, so a
//! score is `q_abs . c + q_pe . k_pe` against every cached row, and the weighted latents go
//! through `W_UV` once at the end. The result equals attention over the expanded keys and
//! values, and a cached token costs 1 row instead of a key and a value per head.
//!
//! A layer may carry a sparse indexer (`dsa.rs`) that picks, per query, the tokens it attends to,
//! or reuse the pick of an earlier layer; without either it attends to every token it may see.
//!
//! The mixer returns the heads' values side by side, before the layer's output projection.
//! Rotation pairs adjacent lanes of the rotated slice, for the query and the position key alike.

use super::dsa::{Indexer, Selected, SelectionCell};
use super::mla_config::MlaMeta;
use super::{DType, Device, QMatMul, Result, Tensor};
use rayon::prelude::*;
use std::sync::Arc;

/// What a layer keeps of the tokens it has seen: 1 latent row and 1 rotated position key
/// per token, `kv_lora_rank + rope_dim` values each, and 1 indexer key per token on a layer
/// that runs an indexer.
#[derive(Clone, Debug, Default)]
pub struct MlaState {
    /// Tokens held.
    pub len: usize,
    /// `len` rows of `[latent | position key]`.
    pub rows: Vec<f32>,
    /// `len` indexer keys; empty on a layer without an indexer.
    pub index_keys: Vec<f32>,
}

/// The weights of a latent attention layer, F32 where the mixer reads them on the host.
pub struct MlaWeights {
    pub q_a: QMatMul,
    /// Scale of the norm on the query factor, `[q_lora_rank]`.
    pub q_a_norm: Vec<f32>,
    pub q_b: QMatMul,
    /// Projects to the latent row and the position key, `kv_lora_rank + rope_dim` outputs.
    pub kv_a: QMatMul,
    /// Scale of the norm on the latent row, `[kv_lora_rank]`.
    pub kv_a_norm: Vec<f32>,
    /// `[n_head * (nope_dim + v_dim), kv_lora_rank]`: per head, `nope_dim` rows of `W_UK`, then
    /// `v_dim` rows of `W_UV`.
    pub kv_b: Vec<f32>,
}

/// A latent attention layer.
pub struct Mla {
    pub(super) meta: MlaMeta,
    pub(super) n_head: usize,
    w: MlaWeights,
    /// Rotation frequency of each lane pair of the rotated slice.
    inv_freq: Vec<f32>,
    /// Multiplies a score before the softmax.
    scale: f32,
    /// Epsilon of the norms on the query factor and the latent row.
    norm_eps: f32,
    /// The indexer, on a layer that runs one.
    indexer: Option<Indexer>,
    /// Where the model's layers leave and find their selection; present when the model has an
    /// indexer, on the layers that run it and on the ones that reuse its selection.
    selection: Option<Arc<SelectionCell>>,
    pub state: Option<MlaState>,
    /// Times the state was dropped or replaced.
    pub(super) moves: u64,
}

/// `row` scaled to unit root mean square, then by `scale`.
pub fn rms_norm_row(row: &mut [f32], scale: &[f32], eps: f32) {
    let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for (v, s) in row.iter_mut().zip(scale) {
        *v = *v * inv * s;
    }
}

/// The rotation frequencies of `dim` lanes at base `base`.
pub fn pair_frequencies(dim: usize, base: f32) -> Vec<f32> {
    (0..dim / 2)
        .map(|j| 1.0 / base.powf(2.0 * j as f32 / dim as f32))
        .collect()
}

/// Lane pair `(2j, 2j + 1)` of `row` rotated by `position * inv_freq[j]`.
pub fn rotate_pairs(row: &mut [f32], position: usize, inv_freq: &[f32]) {
    for (j, f) in inv_freq.iter().enumerate() {
        let (sin, cos) = (position as f32 * f).sin_cos();
        let (a, b) = (row[2 * j], row[2 * j + 1]);
        row[2 * j] = a * cos - b * sin;
        row[2 * j + 1] = b * cos + a * sin;
    }
}

impl Mla {
    /// The layer over `weights`. `scale` multiplies a score; `rope_base` is the frequency base
    /// of the rotated slice.
    pub fn new(
        meta: MlaMeta,
        n_head: usize,
        w: MlaWeights,
        rope_base: f32,
        scale: f32,
        norm_eps: f32,
    ) -> Result<Self> {
        let bad = |what: &str, got: usize, want: usize| {
            crate::tensor::Error::msg(format!(
                "latent attention: {what} holds {got} values, the header says {want}"
            ))
        };
        let q_rank = meta.q_lora_rank.ok_or_else(|| {
            crate::tensor::Error::msg("latent attention without a query factor is not read")
        })?;
        if w.q_a_norm.len() != q_rank {
            return Err(bad("the query norm", w.q_a_norm.len(), q_rank));
        }
        if w.kv_a_norm.len() != meta.kv_lora_rank {
            return Err(bad("the latent norm", w.kv_a_norm.len(), meta.kv_lora_rank));
        }
        let want = n_head * (meta.nope_dim + meta.v_dim) * meta.kv_lora_rank;
        if w.kv_b.len() != want {
            return Err(bad("kv_b", w.kv_b.len(), want));
        }
        Ok(Self {
            inv_freq: pair_frequencies(meta.rope_dim, rope_base),
            meta,
            n_head,
            w,
            scale,
            norm_eps,
            indexer: None,
            selection: None,
            state: None,
            moves: 0,
        })
    }

    /// The layer in a model with a sparse indexer: it runs `indexer` when it has one, else it
    /// attends to what an earlier layer left in `cell`.
    pub fn with_selection(mut self, indexer: Option<Indexer>, cell: Arc<SelectionCell>) -> Self {
        self.indexer = indexer;
        self.selection = Some(cell);
        self
    }

    /// Values of a cached token's row.
    fn row_width(&self) -> usize {
        self.meta.kv_lora_rank + self.meta.rope_dim
    }

    /// Width of the heads' values side by side.
    pub fn out_width(&self) -> usize {
        self.n_head * self.meta.v_dim
    }

    /// The cell the layer leaves its selection in and finds an earlier layer's, when the model
    /// has an indexer.
    pub(super) fn selection_cell(&self) -> Option<&Arc<SelectionCell>> {
        self.selection.as_ref()
    }

    /// Tokens held.
    pub fn len(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The heads' values for the rows of `x` `[1, seq, hidden]`, which stand at positions
    /// `index_pos..index_pos + seq`. A forward at position 0 starts the cache again; any other
    /// must continue it. The result has the shape `[1, seq, n_head * v_dim]` and `x`'s device
    /// and type.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        if b != 1 {
            return Err(crate::tensor::Error::msg(
                "latent attention keeps one sequence's cache",
            ));
        }
        if index_pos == 0 && self.state.is_some() {
            self.state = None;
            self.moves += 1;
        }
        if self.len() != index_pos {
            return Err(crate::tensor::Error::msg(format!(
                "latent attention holds {} tokens and cannot continue from position {index_pos}",
                self.len()
            )));
        }
        let (device, dtype) = (x.device().clone(), x.dtype());
        let host = x
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let q_rank = self.meta.q_lora_rank.unwrap_or(0);
        let width = self.row_width();
        let qk = self.meta.qk_dim();

        // The query: factor, norm, per-head projection, rotation of each head's rotated slice.
        let mut factor = self
            .w
            .q_a
            .forward(&host)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for row in factor.chunks_exact_mut(q_rank) {
            rms_norm_row(row, &self.w.q_a_norm, self.norm_eps);
        }
        let factor_tensor = Tensor::from_vec(factor.clone(), (1, seq, q_rank), &Device::Cpu)?;
        let mut q = self
            .w
            .q_b
            .forward(&factor_tensor)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (t, token) in q.chunks_exact_mut(self.n_head * qk).enumerate() {
            for head in token.chunks_exact_mut(qk) {
                rotate_pairs(
                    &mut head[self.meta.nope_dim..],
                    index_pos + t,
                    &self.inv_freq,
                );
            }
        }

        // The cache rows: normalised latent, rotated position key.
        let mut rows = self
            .w
            .kv_a
            .forward(&host)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (t, row) in rows.chunks_exact_mut(width).enumerate() {
            let (latent, key) = row.split_at_mut(self.meta.kv_lora_rank);
            rms_norm_row(latent, &self.w.kv_a_norm, self.norm_eps);
            rotate_pairs(key, index_pos + t, &self.inv_freq);
        }
        // A layer that reuses a selection reads it before it changes its cache.
        let reused = match (&self.indexer, &self.selection) {
            (None, Some(cell)) => Some(cell.get(index_pos, seq)?),
            _ => None,
        };
        let mut state = self.state.take().unwrap_or_default();
        state.rows.extend_from_slice(&rows);
        state.len += seq;

        // The tokens each row attends to: this layer's pick, the one an earlier layer left, or
        // every token the row may see.
        let picked = match (&self.indexer, &self.selection) {
            (Some(ix), Some(cell)) => {
                let input = host.flatten_all()?.to_vec1::<f32>()?;
                state
                    .index_keys
                    .extend(ix.keys(&input, index_pos, &self.inv_freq));
                let rows = ix.select(
                    &input,
                    &factor,
                    &state.index_keys,
                    index_pos,
                    &self.inv_freq,
                );
                Some(cell.put(Selected {
                    first: index_pos,
                    rows,
                }))
            }
            _ => reused,
        };

        let mut out = Vec::with_capacity(seq * self.out_width());
        let keys: Vec<u32> = (0..state.len as u32).collect();
        for (t, query) in q.chunks_exact(self.n_head * qk).enumerate() {
            let visible = picked
                .as_ref()
                .and_then(|p| p.rows[t].as_deref())
                .unwrap_or(&keys[..=index_pos + t]);
            out.extend(self.attend(query, &state.rows, visible));
        }
        self.state = Some(state);
        Tensor::from_vec(out, (1, seq, self.out_width()), &Device::Cpu)?
            .to_device(&device)?
            .to_dtype(dtype)
    }

    /// A query's heads against the cached rows named by `keys`: the rotated query `[n_head *
    /// qk_dim]`, the cache `rows`, the positions attended. Returns `[n_head * v_dim]`.
    pub(super) fn attend(&self, query: &[f32], rows: &[f32], keys: &[u32]) -> Vec<f32> {
        let m = &self.meta;
        let (kv, rope, nope, v) = (m.kv_lora_rank, m.rope_dim, m.nope_dim, m.v_dim);
        let (qk, width) = (m.qk_dim(), self.row_width());
        let per_head = nope + v;
        let mut out = vec![0f32; self.n_head * v];
        out.par_chunks_mut(v).enumerate().for_each(|(h, out)| {
            let q = &query[h * qk..(h + 1) * qk];
            let (q_nope, q_pe) = q.split_at(nope);
            let b = &self.w.kv_b[h * per_head * kv..(h + 1) * per_head * kv];
            // The query carried into the latent space: `q_nope . W_UK`.
            let mut q_abs = vec![0f32; kv];
            for (d, qd) in q_nope.iter().enumerate() {
                let w = &b[d * kv..(d + 1) * kv];
                for (a, wc) in q_abs.iter_mut().zip(w) {
                    *a += qd * wc;
                }
            }
            let mut scores: Vec<f32> = keys
                .iter()
                .map(|&j| {
                    let row = &rows[j as usize * width..(j as usize + 1) * width];
                    let dot =
                        |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
                    self.scale * (dot(&q_abs, &row[..kv]) + dot(q_pe, &row[kv..kv + rope]))
                })
                .collect();
            let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - top).exp();
                sum += *s;
            }
            // The weighted latents, then through `W_UV`.
            let mut ctx = vec![0f32; kv];
            for (p, &j) in scores.iter().zip(keys) {
                let row = &rows[j as usize * width..j as usize * width + kv];
                let p = p / sum;
                for (c, r) in ctx.iter_mut().zip(row) {
                    *c += p * r;
                }
            }
            for (d, o) in out.iter_mut().enumerate() {
                let w = &b[(nope + d) * kv..(nope + d + 1) * kv];
                *o = w.iter().zip(&ctx).map(|(a, c)| a * c).sum();
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generic_transformer::dsa::IndexerWeights;
    use crate::tensor::quantized::{GgmlDType, QTensor};

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

    fn projection(out: usize, inp: usize, seed: u64) -> QMatMul {
        let w = Tensor::from_vec(values(out * inp, seed, 0.4), (out, inp), &Device::Cpu).unwrap();
        QMatMul::from_qtensor(QTensor::quantize_onto(&w, GgmlDType::F32, &Device::Cpu).unwrap())
            .unwrap()
    }

    const HIDDEN: usize = 24;
    const HEADS: usize = 3;

    fn meta() -> MlaMeta {
        MlaMeta {
            q_lora_rank: Some(10),
            kv_lora_rank: 12,
            nope_dim: 6,
            rope_dim: 4,
            v_dim: 5,
            index: None,
        }
    }

    fn layer(seed: u64) -> Mla {
        let m = meta();
        let w = MlaWeights {
            q_a: projection(10, HIDDEN, seed),
            q_a_norm: values(10, seed + 1, 0.5).iter().map(|v| 1.0 + v).collect(),
            q_b: projection(HEADS * m.qk_dim(), 10, seed + 2),
            kv_a: projection(m.kv_lora_rank + m.rope_dim, HIDDEN, seed + 3),
            kv_a_norm: values(12, seed + 4, 0.5).iter().map(|v| 1.0 + v).collect(),
            kv_b: values(
                HEADS * (m.nope_dim + m.v_dim) * m.kv_lora_rank,
                seed + 5,
                0.4,
            ),
        };
        Mla::new(
            m.clone(),
            HEADS,
            w,
            8_000_000.0,
            1.0 / (m.qk_dim() as f32).sqrt(),
            1e-6,
        )
        .unwrap()
    }

    fn input(seq: usize, seed: u64) -> Tensor {
        Tensor::from_vec(
            values(seq * HIDDEN, seed, 1.0),
            (1, seq, HIDDEN),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Attention over the expanded keys and values, as the library writes it: every cached
    /// latent through `kv_b` to a key and a value per head, the rotated slice joined to the
    /// key, a softmax over the causal prefix.
    fn expanded_reference(
        mla: &Mla,
        x: &Tensor,
        pick: &dyn Fn(usize) -> Option<Vec<u32>>,
    ) -> Vec<f32> {
        let m = &mla.meta;
        let (kv, rope, nope, v) = (m.kv_lora_rank, m.rope_dim, m.nope_dim, m.v_dim);
        let seq = x.dims()[1];
        let host = flat(x);
        let proj = |p: &QMatMul, rows: &[f32], width: usize| {
            let t = Tensor::from_vec(rows.to_vec(), (1, rows.len() / width, width), &Device::Cpu)
                .unwrap();
            flat(&p.forward(&t).unwrap())
        };
        let mut factor = proj(&mla.w.q_a, &host, HIDDEN);
        for r in factor.chunks_exact_mut(10) {
            rms_norm_row(r, &mla.w.q_a_norm, mla.norm_eps);
        }
        let mut q = proj(&mla.w.q_b, &factor, 10);
        let mut kvr = proj(&mla.w.kv_a, &host, HIDDEN);
        let qk = m.qk_dim();
        for (t, token) in q.chunks_exact_mut(HEADS * qk).enumerate() {
            for head in token.chunks_exact_mut(qk) {
                rotate_pairs(&mut head[nope..], t, &mla.inv_freq);
            }
        }
        let width = kv + rope;
        for (t, row) in kvr.chunks_exact_mut(width).enumerate() {
            let (latent, key) = row.split_at_mut(kv);
            rms_norm_row(latent, &mla.w.kv_a_norm, mla.norm_eps);
            rotate_pairs(key, t, &mla.inv_freq);
        }
        let per_head = nope + v;
        // key[j][h] = [W_UK c_j | k_pe_j], value[j][h] = W_UV c_j
        let expand = |j: usize, h: usize| {
            let c = &kvr[j * width..j * width + kv];
            let row = |d: usize| {
                let w = &mla.w.kv_b[(h * per_head + d) * kv..(h * per_head + d + 1) * kv];
                w.iter().zip(c).map(|(a, b)| a * b).sum::<f32>()
            };
            let mut key: Vec<f32> = (0..nope).map(row).collect();
            key.extend_from_slice(&kvr[j * width + kv..(j + 1) * width]);
            let value: Vec<f32> = (0..v).map(|d| row(nope + d)).collect();
            (key, value)
        };
        let mut out = Vec::new();
        for t in 0..seq {
            for h in 0..HEADS {
                let qh = &q[(t * HEADS + h) * qk..(t * HEADS + h + 1) * qk];
                let seen: Vec<usize> = match pick(t) {
                    Some(kept) => kept.iter().map(|&j| j as usize).collect(),
                    None => (0..=t).collect(),
                };
                let kvs: Vec<_> = seen.iter().map(|&j| expand(j, h)).collect();
                let s: Vec<f32> = kvs
                    .iter()
                    .map(|(k, _)| mla.scale * k.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>())
                    .collect();
                let top = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = s.iter().map(|x| (x - top).exp()).collect();
                let sum: f32 = e.iter().sum();
                for d in 0..v {
                    out.push(
                        e.iter()
                            .zip(&kvs)
                            .map(|(p, (_, val))| p / sum * val[d])
                            .sum(),
                    );
                }
            }
        }
        out
    }

    #[test]
    fn attention_in_the_latent_space_equals_attention_over_the_expanded_keys() {
        let mut mla = layer(1);
        let x = input(7, 9);
        let got = flat(&mla.forward(&x, 0).unwrap());
        let want = expanded_reference(&mla, &x, &|_| None);
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(worst <= 1e-5 * top.max(1.0), "worst {worst} of {top}");
    }

    #[test]
    fn decoding_token_by_token_equals_the_prompt_at_once() {
        let mut mla = layer(2);
        let x = input(6, 3);
        let whole = flat(&mla.forward(&x, 0).unwrap());
        let mut parts = Vec::new();
        for t in 0..6 {
            let row = x.narrow(1, t, 1).unwrap().contiguous().unwrap();
            parts.extend(flat(&mla.forward(&row, t).unwrap()));
        }
        assert_eq!(whole, parts);
        // A split prompt continues the cache.
        let mut split = flat(
            &mla.forward(&x.narrow(1, 0, 4).unwrap().contiguous().unwrap(), 0)
                .unwrap(),
        );
        split.extend(flat(
            &mla.forward(&x.narrow(1, 4, 2).unwrap().contiguous().unwrap(), 4)
                .unwrap(),
        ));
        assert_eq!(whole, split);
    }

    #[test]
    fn the_cache_is_one_latent_row_per_token_and_restarts_at_position_zero() {
        let mut mla = layer(3);
        let x = input(5, 4);
        mla.forward(&x, 0).unwrap();
        let state = mla.state.clone().unwrap();
        assert_eq!(state.len, 5);
        assert_eq!(state.rows.len(), 5 * (12 + 4));
        let again = flat(&mla.forward(&x, 0).unwrap());
        assert_eq!(mla.len(), 5);
        let mut fresh = layer(3);
        assert_eq!(again, flat(&fresh.forward(&x, 0).unwrap()));
        // A forward that skips ahead or goes back is refused: the cache cannot be rewound.
        assert!(mla.forward(&input(1, 5), 7).is_err());
        assert!(mla.forward(&input(1, 5), 2).is_err());
    }

    #[test]
    fn a_wrongly_sized_weight_is_refused() {
        let m = meta();
        let mut w = layer(4).w;
        w.kv_b.pop();
        assert!(Mla::new(m, HEADS, w, 1e4, 1.0, 1e-6).is_err());
    }

    const INDEX_HEADS: usize = 2;
    const INDEX_DIM: usize = 8;

    fn indexer(top_k: usize, seed: u64) -> Indexer {
        let w = IndexerWeights {
            wq_b: values(INDEX_HEADS * INDEX_DIM * 10, seed, 0.6),
            wk: values(INDEX_DIM * HIDDEN, seed + 1, 0.6),
            k_norm_w: values(INDEX_DIM, seed + 2, 0.4)
                .iter()
                .map(|v| 1.0 + v)
                .collect(),
            k_norm_b: values(INDEX_DIM, seed + 3, 0.4),
            proj: values(INDEX_HEADS * HIDDEN, seed + 4, 0.6),
        };
        Indexer::new(w, INDEX_HEADS, INDEX_DIM, top_k, HIDDEN, 10, 1e-6).unwrap()
    }

    /// A layer with an indexer of budget `top_k` over the cell.
    fn sparse_layer(top_k: usize, seed: u64, cell: &Arc<SelectionCell>) -> Mla {
        layer(seed).with_selection(Some(indexer(top_k, seed + 20)), cell.clone())
    }

    /// A budget that covers the context keeps every token: the layer attends as one without an
    /// indexer, whatever the indexer scores.
    #[test]
    fn a_budget_above_the_context_attends_densely() {
        let cell = Arc::new(SelectionCell::default());
        let mut sparse = sparse_layer(64, 5, &cell);
        let mut dense = layer(5);
        let x = input(9, 6);
        assert_eq!(
            flat(&sparse.forward(&x, 0).unwrap()),
            flat(&dense.forward(&x, 0).unwrap())
        );
        assert!(cell.get(0, 9).unwrap().rows.iter().all(Option::is_none));
        let keys = sparse.state.as_ref().unwrap().index_keys.len();
        assert_eq!(keys, 9 * INDEX_DIM, "the keys are cached all the same");
    }

    /// Under a budget below the context, a row attends to the tokens the indexer kept and to
    /// no others: the layer equals attention over exactly those expanded keys.
    #[test]
    fn a_row_attends_to_the_tokens_the_indexer_kept() {
        let cell = Arc::new(SelectionCell::default());
        let mut mla = sparse_layer(3, 7, &cell);
        let x = input(9, 8);
        let got = flat(&mla.forward(&x, 0).unwrap());
        let picked = cell.get(0, 9).unwrap();
        assert!(picked.rows[..3].iter().all(Option::is_none));
        assert!(picked.rows[3..]
            .iter()
            .all(|r| r.as_ref().unwrap().len() == 3));
        let want = expanded_reference(&mla, &x, &|t| picked.rows[t].clone());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 1e-5, "worst {worst}");
        let dense = expanded_reference(&mla, &x, &|_| None);
        let apart = got
            .iter()
            .zip(&dense)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(apart > 1e-3, "the selection changed nothing ({apart})");
    }

    /// The prompt in 1 pass and token by token give the same rows and the same selection:
    /// the indexer's keys continue across forwards.
    #[test]
    fn the_selection_continues_across_forwards() {
        let x = input(9, 9);
        let cell = Arc::new(SelectionCell::default());
        let mut whole = sparse_layer(3, 11, &cell);
        let all = flat(&whole.forward(&x, 0).unwrap());
        let kept = cell.get(0, 9).unwrap();
        let cell = Arc::new(SelectionCell::default());
        let mut steps = sparse_layer(3, 11, &cell);
        let mut parts = Vec::new();
        for t in 0..9 {
            let row = x.narrow(1, t, 1).unwrap().contiguous().unwrap();
            parts.extend(flat(&steps.forward(&row, t).unwrap()));
            assert_eq!(cell.get(t, 1).unwrap().rows[0], kept.rows[t], "row {t}");
        }
        assert_eq!(all, parts);
    }

    /// A layer without an indexer attends to the tokens the layer before it kept, and refuses
    /// rows that layer did not run.
    #[test]
    fn a_shared_layer_reuses_the_selection_of_the_layer_before() {
        let cell = Arc::new(SelectionCell::default());
        let mut full = sparse_layer(3, 13, &cell);
        let mut shared = layer(14).with_selection(None, cell.clone());
        let x = input(9, 15);
        assert!(
            shared.forward(&x, 0).is_err(),
            "no layer chose a selection yet"
        );
        assert_eq!(shared.len(), 0, "a refused forward changes nothing");
        full.forward(&x, 0).unwrap();
        let picked = cell.get(0, 9).unwrap();
        let got = flat(&shared.forward(&x, 0).unwrap());
        let want = expanded_reference(&shared, &x, &|t| picked.rows[t].clone());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 1e-5, "worst {worst}");
        assert!(shared.state.as_ref().unwrap().index_keys.is_empty());
        // Rows the first layer did not run are refused.
        assert!(shared.forward(&input(1, 16), 9).is_err());
    }
}
