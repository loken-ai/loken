//! Attention over a cache held in ranges of positions, one range a card.
//!
//! Each range answers a query on its own: the scores' maximum, the sum of their exponentials
//! taken from that maximum, and the values weighed by those exponentials. The parts of all the
//! ranges make the attention over the whole cache once each is rescaled to the greatest
//! maximum, so no range ever needs the keys of another.

use crate::tensor::{DType, Device, Result, Tensor, D};

/// Tensors as large as a block of scores that [`partial`] keeps live at once: the products, the
/// scaled and masked scores, and their exponentials.
const SCORE_TEMPORARIES: usize = 3;

/// Bytes [`partial`] holds for each key of a block, for `rows` query rows of `h` heads over
/// `h_kv` key-value heads of width `d`: the scores and mask, the block's F16 view made whole,
/// its F32 keys and values and the keys transposed.
fn bytes_per_key(h: usize, h_kv: usize, d: usize, rows: usize) -> usize {
    SCORE_TEMPORARIES * h * rows * 4 + rows * 4 + 2 * h_kv * d * (2 + 4) + h_kv * d * 4
}

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
    // The query heads of a group read one key-value head: their rows are stacked over it,
    // `[b, h_kv, group * s, d]`, so the keys and values are read once.
    let q = q.to_dtype(DType::F32)?.reshape((b, h_kv, group * s, d))?;
    let mut scores = (q.matmul(&k.t()?)? * scale)?; // [b, h_kv, group * s, t]
    if let Some(mask) = causal_mask(s, t, q_pos, k_pos, &dev)? {
        scores = scores
            .reshape((b, h_kv, group, s, t))?
            .broadcast_add(&mask)?
            .reshape((b, h_kv, group * s, t))?;
    }
    let max = scores.max_keepdim(D::Minus1)?;
    // A row with every key masked keeps a maximum of minus infinity; its exponentials are
    // taken from zero instead so they come to zero rather than to a NaN.
    let finite_max = max.maximum(&Tensor::new(f32::MIN, &dev)?.broadcast_as(max.shape())?)?;
    let weights = scores.broadcast_sub(&finite_max)?.exp()?;
    let sum = weights.sum_keepdim(D::Minus1)?.reshape((b, h, s, 1))?;
    let out = weights.matmul(&v)?.reshape((b, h, s, d))?;
    let finite_max = finite_max.reshape((b, h, s, 1))?;
    Ok(Part {
        max: finite_max,
        sum,
        out,
    })
}

/// The part of `q` over `k`, `v` as [`partial`] makes it, by the card's fused attention when it
/// takes these shapes.
fn part_of(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    q_pos: usize,
    k_pos: usize,
) -> Result<Part> {
    use crate::inference::kernel::fused::{flash_prefill_gqa_parts, flash_prefill_parts_f16};
    let pos_off = q_pos as i64 - k_pos as i64;
    let fused = match flash_prefill_gqa_parts(q, k, v, scale as f32, pos_off)? {
        Some(p) => Some(p),
        None => flash_prefill_parts_f16(q, k, v, scale as f32, pos_off)?,
    };
    if let Some((max, sum, out)) = fused {
        // A row no key reached keeps a maximum of minus infinity: the least finite value
        // weighs its zero sum without a NaN.
        let max = max.maximum(&Tensor::new(f32::MIN, &max.device())?.broadcast_as(max.shape())?)?;
        return Ok(Part { max, sum, out });
    }
    partial(q, k, v, scale, q_pos, k_pos)
}

/// `acc` and `p` as one part on `dev`: both rescaled to the greater maximum and added.
pub fn merge(acc: Option<Part>, p: Part, dev: &Device) -> Result<Part> {
    let p = Part {
        max: p.max.to_device(dev)?,
        sum: p.sum.to_device(dev)?,
        out: p.out.to_device(dev)?,
    };
    let Some(acc) = acc else {
        return Ok(p);
    };
    let max = acc.max.maximum(&p.max)?;
    let (a, b) = (
        acc.max.broadcast_sub(&max)?.exp()?,
        p.max.broadcast_sub(&max)?.exp()?,
    );
    Ok(Part {
        sum: (acc.sum.broadcast_mul(&a)? + p.sum.broadcast_mul(&b)?)?,
        out: (acc.out.broadcast_mul(&a)? + p.out.broadcast_mul(&b)?)?,
        max,
    })
}

/// The attention a merged part stands for, `[b, h, s, d]`.
pub fn finish(part: &Part) -> Result<Tensor> {
    part.out.broadcast_div(&part.sum)
}

/// The attention the parts of every range make together, `[b, h, s, d]` on `dev`.
#[cfg(test)]
fn combine(parts: &[Part], dev: &Device) -> Result<Tensor> {
    let mut acc = None;
    for p in parts {
        let p = Part {
            max: p.max.clone(),
            sum: p.sum.clone(),
            out: p.out.clone(),
        };
        acc = Some(merge(acc, p, dev)?);
    }
    let acc =
        acc.ok_or_else(|| crate::tensor::Error::msg("context parallel: no range to combine"))?;
    finish(&acc)
}

#[cfg(test)]
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

/// One block of a range: `cap` positions of F16 keys and values `[1, cap, h_kv, d]`,
/// position-major so a run of positions is a view.
struct Seg {
    k: Tensor,
    v: Tensor,
    cap: usize,
}

/// One card's range of a layer's cache: the positions `start..start + len`, held in its blocks
/// in order. A block is written in place and kept from request to request.
struct Shard {
    dev: Device,
    /// The most positions the range takes.
    cap: usize,
    start: usize,
    len: usize,
    segs: Vec<Seg>,
    /// No block can be added: the card would keep less than the ranges leave it.
    closed: bool,
}

impl Shard {
    /// Positions the range's blocks hold.
    fn held(&self) -> usize {
        self.segs.iter().map(|s| s.cap).sum()
    }
}

/// A layer's cache cut into ranges of positions over several cards, filled in order. A request
/// reserves its blocks before its prefill (see [`ContextShards::reserve_on`]); past them, a range
/// grows a block of `seg` positions at a time while its card keeps `keep_free` bytes and no later
/// range has room left, then the next card's range takes the positions that follow.
pub struct ContextShards {
    shards: Vec<Shard>,
    h_kv: usize,
    d: usize,
    /// Positions a block grown past the reserve holds.
    seg: usize,
    /// Bytes each card keeps free beside the ranges.
    keep_free: u64,
    /// The most bytes one block of keys may take: a range is read a block at a time.
    scores_bytes: usize,
}

impl ContextShards {
    /// Ranges of `h_kv` heads of width `d` on `cards`, each holding at most its count of
    /// positions; see the type for `seg` and `keep_free`. A block of scores takes at most
    /// `scores_bytes`.
    pub fn new(
        cards: Vec<(Device, usize)>,
        h_kv: usize,
        d: usize,
        seg: usize,
        keep_free: u64,
        scores_bytes: usize,
    ) -> Self {
        let shards = cards
            .into_iter()
            .map(|(dev, cap)| Shard {
                dev,
                cap,
                start: 0,
                len: 0,
                segs: Vec::new(),
                closed: false,
            })
            .collect();
        Self {
            shards,
            h_kv,
            d,
            seg: seg.max(1),
            keep_free,
            scores_bytes,
        }
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len).sum()
    }

    /// Whether no position is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Positions the blocks of every range hold.
    pub fn held(&self) -> usize {
        self.shards.iter().map(Shard::held).sum()
    }

    /// Bytes a position takes in a range.
    pub fn position_bytes(&self) -> u64 {
        (2 * self.h_kv * self.d * 2) as u64
    }

    /// Bytes each card keeps free beside the ranges.
    pub fn keep_free(&self) -> u64 {
        self.keep_free
    }

    /// The cards of the ranges, the layer's own first.
    pub fn cards(&self) -> Vec<Device> {
        self.shards.iter().map(|s| s.dev.clone()).collect()
    }

    /// Add a block of `positions` to the range on `card`, the `card`-th of [`Self::cards`].
    pub fn reserve_on(&mut self, card: usize, positions: usize) -> Result<()> {
        let (h_kv, d) = (self.h_kv, self.d);
        let shard = &mut self.shards[card];
        let positions = positions.min(shard.cap.saturating_sub(shard.held()));
        if positions == 0 {
            return Ok(());
        }
        shard.segs.push(Seg {
            k: Tensor::zeros_on((1, positions, h_kv, d), DType::F16, &shard.dev)?,
            v: Tensor::zeros_on((1, positions, h_kv, d), DType::F16, &shard.dev)?,
            cap: positions,
        });
        shard.closed = false;
        Ok(())
    }

    /// Append the rows `k`, `v` `[1, h_kv, n, d]` after the positions held, each on the card
    /// whose range they fall in.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (_, h_kv, n, d) = k.dims4()?;
        let (seg, keep_free) = (self.seg, self.keep_free);
        // Whether a range after each one still has room in its blocks: a range grows only when
        // none has.
        let later_room: Vec<bool> = (0..self.shards.len())
            .map(|i| self.shards[i + 1..].iter().any(|s| s.held() > s.len))
            .collect();
        let mut done = 0;
        let mut at = self.len();
        for (i, shard) in self.shards.iter_mut().enumerate() {
            if done == n {
                break;
            }
            if shard.closed {
                continue;
            }
            if shard.len == 0 {
                shard.start = at;
            }
            while done < n {
                if shard.len >= shard.cap {
                    shard.closed = true;
                    break;
                }
                if shard.len == shard.held() {
                    // Only a card's memory is counted: the host's ranges take what they are
                    // given.
                    let bytes = (2 * seg * h_kv * d * 2) as u64;
                    let short = later_room[i]
                        || (shard.dev.is_cuda()
                            && crate::inference::place::expert_cache::free_now(&shard.dev)
                                < bytes.saturating_add(keep_free));
                    if short {
                        if !later_room[i] {
                            tracing::info!(
                                "context parallel: range on {:?} closed at {} positions",
                                shard.dev.location(),
                                shard.len
                            );
                        }
                        shard.closed = true;
                        break;
                    }
                    shard.segs.push(Seg {
                        k: Tensor::zeros_on((1, seg, h_kv, d), DType::F16, &shard.dev)?,
                        v: Tensor::zeros_on((1, seg, h_kv, d), DType::F16, &shard.dev)?,
                        cap: seg,
                    });
                }
                // The block the next position falls in, and its offset there.
                let (mut b, mut off) = (0, shard.len);
                while off >= shard.segs[b].cap {
                    off -= shard.segs[b].cap;
                    b += 1;
                }
                let take = (shard.segs[b].cap - off)
                    .min(n - done)
                    .min(shard.cap - shard.len);
                let rows = |t: &Tensor| -> Result<Tensor> {
                    t.narrow(2, done, take)?
                        .to_device(&shard.dev)?
                        .to_dtype(DType::F16)?
                        .transpose(1, 2)?
                        .contiguous()
                };
                shard.segs[b].k.slice_set(&rows(k)?, 1, off)?;
                shard.segs[b].v.slice_set(&rows(v)?, 1, off)?;
                shard.len += take;
                done += take;
                at += take;
            }
        }
        if done < n {
            crate::tensor::bail!(
                "context parallel: no card has room for {} more positions past {}",
                n - done,
                at
            );
        }
        Ok(())
    }

    /// Keep the first `len` positions; the blocks stay for the positions written next.
    pub fn trim_to(&mut self, len: usize) -> Result<()> {
        for s in &mut self.shards {
            let keep = if s.len == 0 {
                0
            } else {
                len.saturating_sub(s.start).min(s.len)
            };
            if keep < s.len {
                s.closed = false;
            }
            s.len = keep;
        }
        Ok(())
    }

    /// The attention of `q` `[b, h, s, d]`, its rows at positions `q_pos..`, over every position
    /// held, on `q`'s card.
    pub fn attend(&self, q: &Tensor, scale: f64, q_pos: usize) -> Result<Tensor> {
        let home = q.device().clone();
        // The parts are merged as they come: one accumulator, whatever the context's length.
        let mut acc: Option<Part> = None;
        let (_, h, rows, d) = q.dims4()?;
        // The keys a block holds: the bytes a block may take over what each key costs.
        // On a card a block costs its keys and values made whole; the fallback holds its scores.
        let per_key = |dev: &Device| {
            if dev.is_cuda() {
                2 * self.h_kv * d * 2
            } else {
                bytes_per_key(h, self.h_kv, d, rows)
            }
        };
        for s in self.shards.iter().filter(|s| s.len > 0) {
            let q_here = q.to_device(&s.dev)?;
            let block = (self.scores_bytes / per_key(&s.dev)).max(1);
            let mut seg_from = 0;
            for seg in &s.segs {
                if seg_from >= s.len {
                    break;
                }
                let in_seg = (s.len - seg_from).min(seg.cap);
                let mut from = 0;
                while from < in_seg {
                    let n = block.min(in_seg - from);
                    let k_pos = s.start + seg_from + from;
                    // A block past every row has nothing to give.
                    if k_pos < q_pos + rows {
                        // `[1, h_kv, n, d]` views of the block: the card's kernel reads them by
                        // their strides, the fallbacks copy what they need.
                        let heads = |t: &Tensor| t.narrow(1, from, n)?.transpose(1, 2);
                        let part = part_of(
                            &q_here,
                            &heads(&seg.k)?,
                            &heads(&seg.v)?,
                            scale,
                            q_pos,
                            k_pos,
                        )?;
                        acc = Some(merge(acc, part, &home)?);
                    }
                    from += n;
                }
                seg_from += seg.cap;
            }
        }
        let acc = acc
            .ok_or_else(|| crate::tensor::Error::msg("context parallel: no position to attend"))?;
        finish(&acc)
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

    /// The card's fused part matches the portable one: the attention it stands for and the
    /// log-sum-exp of its scores, for a range before every row, one across the diagonal and a
    /// decoded row.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_cards_part_is_the_portable_part() {
        let Ok(card) = Device::new_cuda(0) else {
            return;
        };
        // The fused part runs on the tensor-core cards only; below them the portable part is
        // the only one, and there is nothing to compare it with.
        if !card
            .as_cuda_device()
            .is_ok_and(|d| d.has_ampere_tensor_cores())
        {
            return;
        }
        let (h, h_kv, d) = (8usize, 2usize, 128usize);
        // (query rows, keys, first row's position, first key's position)
        for (s, t, q_pos, k_pos) in [(40, 70, 100, 10), (40, 70, 30, 0), (1, 300, 500, 200)] {
            let q = t4([1, h, s, d], 21).to_device(&card).unwrap();
            // F16 keys and values, as the ranges hold them.
            let f16 = |t: Tensor| t.to_dtype(DType::F16).unwrap().to_device(&card).unwrap();
            let k = f16(t4([1, h_kv, t, d], 22));
            let v = f16(t4([1, h_kv, t, d], 23));
            let scale = 1.0 / (d as f64).sqrt();
            let pos_off = q_pos as i64 - k_pos as i64;
            let ran = crate::inference::kernel::fused::flash_prefill_gqa_parts(
                &q,
                &k,
                &v,
                scale as f32,
                pos_off,
            )
            .unwrap();
            assert!(ran.is_some(), "the card runs the fused part");
            let fused = part_of(&q, &k, &v, scale, q_pos, k_pos).unwrap();
            let portable = partial(&q, &k, &v, scale, q_pos, k_pos).unwrap();
            let worst = |a: &Tensor, b: &Tensor| {
                flat(&a.to_device(&Device::Cpu).unwrap())
                    .iter()
                    .zip(&flat(&b.to_device(&Device::Cpu).unwrap()))
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max)
            };
            let lse = |p: &Part| (p.max.clone() + p.sum.log().unwrap()).unwrap();
            let att = worst(&finish(&fused).unwrap(), &finish(&portable).unwrap());
            let lse = worst(&lse(&fused), &lse(&portable));
            assert!(
                att < 2e-2 && lse < 2e-2,
                "{s}x{t} at {q_pos}/{k_pos}: attention {att}, lse {lse}"
            );
        }
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
        // A reserve of five positions on the first range, blocks of four grown past it, blocks of
        // three keys a row: each range is read in several blocks.
        let mut shards = ContextShards::new(
            vec![(dev.clone(), 6), (dev.clone(), 6), (dev.clone(), 20)],
            h_kv,
            d,
            4,
            0,
            3 * bytes_per_key(h, h_kv, d, 1),
        );
        shards.reserve_on(0, 5).unwrap();
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
    /// range on its own card first, then one on each other card in their order, each growing while
    /// its card keeps the work a forward needs and a block of scores. How many layers were cut.
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
        // Each card keeps the forward's work and a block of scores beside its ranges; a range
        // grows a working window at a time.
        let keep_free = 2 * work;
        let seg = crate::inference::cache::KV_WORKING_WINDOW_TOKENS;
        for &i in &cut {
            let own = self.layers[i].cos.device();
            let mut order: Vec<(Device, usize)> = cards
                .iter()
                .filter(|d| d.same_device(&own))
                .map(|d| (d.clone(), usize::MAX))
                .collect();
            order.extend(
                cards
                    .iter()
                    .filter(|d| !d.same_device(&own))
                    .map(|d| (d.clone(), usize::MAX)),
            );
            let l = &mut self.layers[i];
            // A block of scores takes half the room left beside the forward's work, so the two
            // never meet the card's edge together.
            let (h_kv, d) = (l.n_kv_head, l.head_dim);
            l.context = Some(ContextShards::new(
                order,
                h_kv,
                d,
                seg,
                keep_free,
                (work / 2) as usize,
            ));
            #[cfg(feature = "cuda")]
            {
                l.quant_kv = None;
            }
            l.kv_cache.reset();
        }
        tracing::info!(
            "context parallel: {} attention layers cut over {} cards, each card keeping {:.2} GB beside them",
            cut.len(),
            cards.len(),
            keep_free as f64 / 1e9
        );
        cut.len()
    }

    /// Give every cut layer blocks for `total` positions before a prefill, while nothing else
    /// holds its cards: each card's room past what it keeps free goes in equal shares to the
    /// layers that still need positions, their own card first, then the others.
    pub fn reserve_context(&mut self, total: usize) {
        let cut: Vec<usize> = (0..self.layers.len())
            .filter(|&i| self.layers[i].context.is_some())
            .collect();
        let need = |l: &super::GenericTransformerLayer| {
            l.context
                .as_ref()
                .map_or(0, |c| total.saturating_sub(c.held()))
        };
        let mut needs: Vec<usize> = cut.iter().map(|&i| need(&self.layers[i])).collect();
        if needs.iter().all(|&n| n == 0) {
            return;
        }
        fn ctx(l: &super::GenericTransformerLayer) -> &ContextShards {
            l.context.as_ref().expect("a cut layer")
        }
        let cards = ctx(&self.layers[cut[0]]).cards();
        #[cfg(feature = "cuda")]
        for dev in &cards {
            if let Ok(cd) = dev.as_cuda_device() {
                let _ = cd.0.stream().synchronize();
                crate::tensor::cuda::trim_pool(cd.0.ordinal());
            }
        }
        // The layer's own card first: position 0 of its ranges; then the others.
        for own in [true, false] {
            for dev in &cards {
                let takers: Vec<(usize, usize)> = cut
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| needs[j] > 0)
                    .filter_map(|(j, &i)| {
                        let at = ctx(&self.layers[i])
                            .cards()
                            .iter()
                            .position(|d| d.same_device(dev))?;
                        ((at == 0) == own).then_some((j, at))
                    })
                    .collect();
                let Some(&(j0, _)) = takers.first() else {
                    continue;
                };
                let c = ctx(&self.layers[cut[j0]]);
                let (keep, position) = (c.keep_free(), c.position_bytes());
                let room = if dev.is_cuda() {
                    crate::inference::place::expert_cache::free_now(dev).saturating_sub(keep)
                } else {
                    u64::MAX
                };
                let share = (room / (position * takers.len() as u64)) as usize;
                for (j, at) in takers {
                    let n = needs[j].min(share);
                    if n == 0 {
                        continue;
                    }
                    let l = &mut self.layers[cut[j]];
                    let c = l.context.as_mut().expect("a cut layer");
                    if let Err(e) = c.reserve_on(at, n) {
                        tracing::warn!("context parallel: a reserve of {n} positions failed: {e}");
                        continue;
                    }
                    needs[j] -= n;
                }
            }
        }
        if let Some(&short) = needs.iter().max().filter(|&&n| n > 0) {
            tracing::warn!(
                "context parallel: the cards hold {} of the {total} positions a layer asks for",
                total - short
            );
        }
    }
}
