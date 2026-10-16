//! Attention over a cache held in ranges of positions, one range a card.
//!
//! Each range answers a query on its own: the scores' maximum, the sum of their exponentials
//! taken from that maximum, and the values weighed by those exponentials. The parts of all the
//! ranges make the attention over the whole cache once each is rescaled to the greatest
//! maximum, so no range ever needs the keys of another.

use crate::tensor::{DType, Device, Result, Tensor, D};

/// What one range of the cache gives a query: per query head and row, the scores' maximum
/// `max` `[b, h, s, 1]`, the sum of their exponentials from it `sum` `[b, h, s, 1]`, and the
/// values weighed by them `out` `[b, h, s, d]`, not divided by the sum.
pub struct Part {
    pub max: Tensor,
    pub sum: Tensor,
    pub out: Tensor,
}

/// The part of the attention of `q` `[b, h, s, d]` over the keys `k` and values `v`
/// `[b, h_kv, t, d]` of one range, F32. The rows of `q` stand at positions `q_pos..q_pos + s`
/// and the keys at `k_pos..k_pos + t`; a key past a row's position is masked out. A row that
/// sees no key of the range gives a maximum of minus infinity and nothing else.
pub fn partial(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    q_pos: usize,
    k_pos: usize,
) -> Result<Part> {
    let (b, h, s, d) = q.dims4()?;
    let (_, h_kv, t, _) = k.dims4()?;
    let group = h / h_kv.max(1);
    let dev = q.device().clone();
    let k = k.to_device(&dev)?.to_dtype(DType::F32)?;
    let v = v.to_device(&dev)?.to_dtype(DType::F32)?;
    // A query head reads the key-value head of its group.
    let (k, v) = if group > 1 {
        (repeat_heads(&k, group)?, repeat_heads(&v, group)?)
    } else {
        (k, v)
    };
    let q = q.to_dtype(DType::F32)?;
    let mut scores = (q.matmul(&k.t()?)? * scale)?; // [b, h, s, t]
    let mask = causal_mask(s, t, q_pos, k_pos, &dev)?;
    if let Some(mask) = mask {
        scores = scores.broadcast_add(&mask)?;
    }
    let max = scores.max_keepdim(D::Minus1)?;
    // A row with every key masked keeps a maximum of minus infinity; its exponentials are
    // taken from zero instead so they come to zero rather than to a NaN.
    let finite_max = max.maximum(&Tensor::new(f32::MIN, &dev)?.broadcast_as(max.shape())?)?;
    let weights = scores.broadcast_sub(&finite_max)?.exp()?;
    let sum = weights.sum_keepdim(D::Minus1)?;
    let out = weights.matmul(&v)?;
    debug_assert_eq!(out.dims(), &[b, h, s, d]);
    Ok(Part {
        max: finite_max,
        sum,
        out,
    })
}

/// The attention the parts of every range make together, `[b, h, s, d]` on `dev`.
pub fn combine(parts: &[Part], dev: &Device) -> Result<Tensor> {
    let first = parts
        .first()
        .ok_or_else(|| crate::tensor::Error::msg("context parallel: no range to combine"))?;
    let on = |t: &Tensor| t.to_device(dev);
    let mut max = on(&first.max)?;
    for p in &parts[1..] {
        max = max.maximum(&on(&p.max)?)?;
    }
    let mut sum: Option<Tensor> = None;
    let mut out: Option<Tensor> = None;
    for p in parts {
        let scale = on(&p.max)?.broadcast_sub(&max)?.exp()?;
        let s = on(&p.sum)?.broadcast_mul(&scale)?;
        let o = on(&p.out)?.broadcast_mul(&scale)?;
        sum = Some(match sum {
            Some(acc) => (acc + s)?,
            None => s,
        });
        out = Some(match out {
            Some(acc) => (acc + o)?,
            None => o,
        });
    }
    let (sum, out) = (
        sum.expect("one part at least"),
        out.expect("one part at least"),
    );
    out.broadcast_div(&sum)
}

/// `x` `[b, h_kv, t, d]` with each head repeated `n` times in place: `[b, h_kv * n, t, d]`.
fn repeat_heads(x: &Tensor, n: usize) -> Result<Tensor> {
    let (b, h_kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, h_kv, n, t, d))?
        .contiguous()?
        .reshape((b, h_kv * n, t, d))
}

/// The additive mask of rows at `q_pos..q_pos + s` over keys at `k_pos..k_pos + t`, `[s, t]`:
/// zero where the key is at or before the row, minus infinity past it. `None` when every key
/// is at or before every row.
fn causal_mask(
    s: usize,
    t: usize,
    q_pos: usize,
    k_pos: usize,
    dev: &Device,
) -> Result<Option<Tensor>> {
    if k_pos + t <= q_pos + 1 {
        return Ok(None);
    }
    let mut m = vec![0f32; s * t];
    for r in 0..s {
        for c in 0..t {
            if k_pos + c > q_pos + r {
                m[r * t + c] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Some(Tensor::from_vec(m, (s, t), dev)?))
}

/// One card's range of a layer's cache: the positions `start..start + len` as F16 keys and
/// values `[1, h_kv, len, d]`, room for `cap` positions.
struct Shard {
    dev: Device,
    cap: usize,
    start: usize,
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl Shard {
    fn len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[2])
    }
}

/// A layer's cache cut into ranges of positions over several cards, filled in order: the first
/// card's range, then the next one's once it is full.
pub struct ContextShards {
    shards: Vec<Shard>,
    /// The most bytes one block of scores may take: a range is read a block of keys at a time.
    scores_bytes: usize,
}

impl ContextShards {
    /// Ranges on `cards`, each with room for its count of positions, in that order; a block of
    /// scores takes at most `scores_bytes`.
    pub fn new(cards: Vec<(Device, usize)>, scores_bytes: usize) -> Self {
        Self {
            scores_bytes,
            shards: cards
                .into_iter()
                .map(|(dev, cap)| Shard {
                    dev,
                    cap,
                    start: 0,
                    k: None,
                    v: None,
                })
                .collect(),
        }
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.shards.iter().map(Shard::len).sum()
    }

    /// Whether no position is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Positions the ranges have room for.
    pub fn capacity(&self) -> usize {
        self.shards.iter().map(|s| s.cap).sum()
    }

    /// Append the rows `k`, `v` `[1, h_kv, n, d]` after the positions held, each on the card
    /// whose range they fall in.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let n = k.dims()[2];
        let mut done = 0;
        let mut at = self.len();
        for i in 0..self.shards.len() {
            if done == n {
                break;
            }
            let start: usize = self.shards[..i].iter().map(|s| s.cap).sum();
            let shard = &mut self.shards[i];
            let room = (start + shard.cap).saturating_sub(at);
            if room == 0 {
                continue;
            }
            let take = room.min(n - done);
            let kr = k
                .narrow(2, done, take)?
                .to_device(&shard.dev)?
                .to_dtype(DType::F16)?;
            let vr = v
                .narrow(2, done, take)?
                .to_device(&shard.dev)?
                .to_dtype(DType::F16)?;
            shard.start = start;
            let grow = |held: Option<Tensor>, rows: Tensor| -> Result<Tensor> {
                match held {
                    Some(h) => Tensor::cat(&[&h, &rows], 2),
                    None => Ok(rows),
                }
            };
            shard.k = Some(grow(shard.k.take(), kr)?);
            shard.v = Some(grow(shard.v.take(), vr)?);
            done += take;
            at += take;
        }
        if done < n {
            crate::tensor::bail!(
                "context parallel: {} positions past the {} the cards hold",
                n - done,
                self.capacity()
            );
        }
        Ok(())
    }

    /// Keep the first `len` positions.
    pub fn trim_to(&mut self, len: usize) -> Result<()> {
        for s in &mut self.shards {
            let keep = len.saturating_sub(s.start).min(s.len());
            if keep == 0 {
                s.k = None;
                s.v = None;
            } else if keep < s.len() {
                s.k = Some(
                    s.k.as_ref()
                        .expect("held")
                        .narrow(2, 0, keep)?
                        .contiguous()?,
                );
                s.v = Some(
                    s.v.as_ref()
                        .expect("held")
                        .narrow(2, 0, keep)?
                        .contiguous()?,
                );
            }
        }
        Ok(())
    }

    /// The attention of `q` `[b, h, s, d]`, its rows at positions `q_pos..`, over every position
    /// held, on `q`'s card.
    pub fn attend(&self, q: &Tensor, scale: f64, q_pos: usize) -> Result<Tensor> {
        let home = q.device().clone();
        let mut parts = Vec::new();
        let (_, h, rows, _) = q.dims4()?;
        // The keys a block of scores holds: the score bytes over a row's scores for every head.
        let block = (self.scores_bytes / (h * rows * 4).max(1)).max(1);
        for s in self.shards.iter().filter(|s| s.len() > 0) {
            let q_here = q.to_device(&s.dev)?;
            let (k, v) = (s.k.as_ref().expect("held"), s.v.as_ref().expect("held"));
            let mut from = 0;
            while from < s.len() {
                let n = block.min(s.len() - from);
                // A block past every row has nothing to give.
                if s.start + from < q_pos + rows {
                    parts.push(partial(
                        &q_here,
                        &k.narrow(2, from, n)?,
                        &v.narrow(2, from, n)?,
                        scale,
                        q_pos,
                        s.start + from,
                    )?);
                }
                from += n;
            }
        }
        combine(&parts, &home)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn t4(n: [usize; 4], seed: u64) -> Tensor {
        Tensor::from_vec(
            values(n.iter().product(), seed),
            (n[0], n[1], n[2], n[3]),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The parts of a cache cut in three ranges make the attention of the whole cache, for a
    /// prompt's rows (masked) and a decoded row, with grouped key-value heads.
    #[test]
    fn ranges_of_the_cache_make_the_attention_of_the_whole() {
        let (b, h, h_kv, d, t) = (1usize, 4usize, 2usize, 8usize, 23usize);
        let k = t4([b, h_kv, t, d], 1);
        let v = t4([b, h_kv, t, d], 2);
        let scale = 1.0 / (d as f64).sqrt();
        for (s, q_pos) in [(5usize, t - 5), (1, t - 1)] {
            let q = t4([b, h, s, d], 3 + s as u64);
            let whole = combine(
                &[partial(&q, &k, &v, scale, q_pos, 0).unwrap()],
                &Device::Cpu,
            )
            .unwrap();
            let cuts = [0usize, 7, 15, t];
            let parts: Vec<Part> = cuts
                .windows(2)
                .map(|w| {
                    partial(
                        &q,
                        &k.narrow(2, w[0], w[1] - w[0]).unwrap(),
                        &v.narrow(2, w[0], w[1] - w[0]).unwrap(),
                        scale,
                        q_pos,
                        w[0],
                    )
                    .unwrap()
                })
                .collect();
            let split = combine(&parts, &Device::Cpu).unwrap();
            let (a, c) = (flat(&whole), flat(&split));
            let worst = a
                .iter()
                .zip(&c)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-5, "{s} rows: max diff {worst}");
            // The one-range result is the softmax attention itself.
            let kr = repeat_heads(&k, h / h_kv).unwrap();
            let vr = repeat_heads(&v, h / h_kv).unwrap();
            let mut sc = (q.matmul(&kr.t().unwrap()).unwrap() * scale).unwrap();
            if let Some(m) = causal_mask(s, t, q_pos, 0, &Device::Cpu).unwrap() {
                sc = sc.broadcast_add(&m).unwrap();
            }
            let reference = crate::tensor::ops::softmax_last_dim(&sc)
                .unwrap()
                .matmul(&vr)
                .unwrap();
            let r = flat(&reference);
            let worst = a
                .iter()
                .zip(&r)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-5,
                "{s} rows against the softmax: max diff {worst}"
            );
        }
    }

    /// A cache appended a prompt then decoded rows at a time across three ranges attends as one
    /// cache does, and a trim keeps the positions it names.
    #[test]
    fn shards_filled_in_order_attend_as_one_cache() {
        let (h, h_kv, d) = (4usize, 2usize, 8usize);
        let dev = Device::Cpu;
        // Blocks of three keys a row: each range is read in several blocks.
        let mut shards = ContextShards::new(
            vec![(dev.clone(), 6), (dev.clone(), 6), (dev.clone(), 20)],
            3 * h * 4,
        );
        let k = t4([1, h_kv, 17, d], 11);
        let v = t4([1, h_kv, 17, d], 12);
        shards
            .append(&k.narrow(2, 0, 9).unwrap(), &v.narrow(2, 0, 9).unwrap())
            .unwrap();
        for p in 9..17 {
            shards
                .append(&k.narrow(2, p, 1).unwrap(), &v.narrow(2, p, 1).unwrap())
                .unwrap();
        }
        assert_eq!(shards.len(), 17);
        let q = t4([1, h, 1, d], 13);
        let scale = 0.3;
        let f16 = |t: &Tensor| t.to_dtype(DType::F16).unwrap();
        let want = combine(
            &[partial(&q, &f16(&k), &f16(&v), scale, 16, 0).unwrap()],
            &dev,
        )
        .unwrap();
        let got = shards.attend(&q, scale, 16).unwrap();
        let worst = flat(&want)
            .iter()
            .zip(flat(&got))
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-5, "max diff {worst}");
        shards.trim_to(8).unwrap();
        assert_eq!(shards.len(), 8);
        let want = combine(
            &[partial(
                &q,
                &f16(&k.narrow(2, 0, 8).unwrap()),
                &f16(&v.narrow(2, 0, 8).unwrap()),
                scale,
                7,
                0,
            )
            .unwrap()],
            &dev,
        )
        .unwrap();
        let got = shards.attend(&q, scale, 7).unwrap();
        let worst = flat(&want)
            .iter()
            .zip(flat(&got))
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-5, "after the trim: max diff {worst}");
        assert!(shards
            .append(&t4([1, h_kv, 30, d], 14), &t4([1, h_kv, 30, d], 15))
            .is_err());
    }

    /// A range every key of which is past the rows gives nothing, and the others are exact.
    #[test]
    fn a_range_past_every_row_adds_nothing() {
        let (b, h, d, t) = (1usize, 2usize, 4usize, 10usize);
        let k = t4([b, h, t, d], 7);
        let v = t4([b, h, t, d], 8);
        let q = t4([b, h, 1, d], 9);
        let scale = 0.5;
        let seen = partial(
            &q,
            &k.narrow(2, 0, 6).unwrap(),
            &v.narrow(2, 0, 6).unwrap(),
            scale,
            5,
            0,
        )
        .unwrap();
        let ahead = partial(
            &q,
            &k.narrow(2, 6, 4).unwrap(),
            &v.narrow(2, 6, 4).unwrap(),
            scale,
            5,
            6,
        )
        .unwrap();
        let both = combine(&[seen, ahead], &Device::Cpu).unwrap();
        let alone = combine(
            &[partial(
                &q,
                &k.narrow(2, 0, 6).unwrap(),
                &v.narrow(2, 0, 6).unwrap(),
                scale,
                5,
                0,
            )
            .unwrap()],
            &Device::Cpu,
        )
        .unwrap();
        let (a, c) = (flat(&both), flat(&alone));
        assert!(a.iter().all(|x| x.is_finite()));
        let worst = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-6, "max diff {worst}");
    }
}

impl super::GenericHeteroTransformer {
    /// Cut the cache of every attention layer that can be cut over `cards`: each layer keeps a
    /// range on its own card first, then one on each other card in their order. A card's room is
    /// its free memory less the work a forward keeps, shared evenly by the layers, at the F16
    /// width of a position. How many layers were cut.
    pub fn attach_context_shards(&mut self, cards: &[Device]) -> usize {
        if cards.len() < 2 {
            return 0;
        }
        let cut: Vec<usize> = (0..self.layers.len())
            .filter(|&i| {
                let l = &self.layers[i];
                l.attends()
                    && l.cos.device().is_cuda()
                    && l.sparse.is_none()
                    && l.attn_sinks.is_none()
                    && l.sliding_window.is_none()
                    && l.tp_shards.is_empty()
                    && !l.populate_dual_kv
            })
            .collect();
        if cut.is_empty() {
            return 0;
        }
        let work = crate::inference::place::runtime_demand::llm_work_reserve(
            self.config.embedding_length.max(self.config.ffn_dim),
            crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS,
        );
        let position: u64 = cut
            .iter()
            .map(|&i| {
                let l = &self.layers[i];
                (2 * l.n_kv_head * l.head_dim * 2) as u64
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let room = |dev: &Device| -> usize {
            let free = crate::inference::place::expert_cache::free_now(dev).saturating_sub(work);
            (free / (position * cut.len() as u64)) as usize
        };
        let rooms: Vec<(Device, usize)> = cards.iter().map(|d| (d.clone(), room(d))).collect();
        for &i in &cut {
            let own = self.layers[i].cos.device();
            let mut order: Vec<(Device, usize)> = rooms
                .iter()
                .filter(|(d, _)| d.same_device(&own))
                .cloned()
                .collect();
            order.extend(rooms.iter().filter(|(d, _)| !d.same_device(&own)).cloned());
            let l = &mut self.layers[i];
            l.context = Some(ContextShards::new(order, work as usize));
            #[cfg(feature = "cuda")]
            {
                l.quant_kv = None;
            }
            l.kv_cache.reset();
        }
        let positions = self.layers[cut[0]]
            .context
            .as_ref()
            .map_or(0, ContextShards::capacity);
        tracing::info!(
            "context parallel: {} attention layers cut over {} cards, room for {positions} positions",
            cut.len(),
            cards.len()
        );
        cut.len()
    }
}
