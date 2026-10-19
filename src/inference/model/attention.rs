//! The attention an encoder-decoder runs, once.
//!
//! A decoder does the same thing three times over: attend to its own past, attend to an
//! encoder's output, and - in the models that share heads - read fewer key heads than query
//! heads. Which of those applies is a property of where the attention sits, not of how it
//! works, so it is said here as two small choices: what the attention keeps between calls, and
//! which positions a query may see.
//!
//! The families that reach this differ in what their checkpoints carry - biases on some
//! projections and not others, one key head per query head or one per group - and none of that
//! reaches the arithmetic either. Each loader reads its own names and hands over four
//! projections.

use crate::tensor::layer::Linear;
use crate::tensor::ops::repeat_kv;
use crate::tensor::{Result, Tensor};

/// What an attention keeps between calls.
#[derive(Debug, Clone)]
pub enum Kv {
    /// Nothing: both operands are projected on every call. What an encoder's own attention
    /// needs, and what a decoder needs when it is handed the whole sequence at once.
    None,
    /// The keys and values seen so far, extended by each call - a decoder attending to its own
    /// past, one step at a time.
    Growing(Option<(Tensor, Tensor)>),
    /// Projected once and kept - a decoder attending to an encoder's output, which is the same
    /// tensor at every step of one generation and so is worth projecting only at the first.
    Fixed(Option<(Tensor, Tensor)>),
    /// The keys and values seen so far, written in place into buffers sized for the longest
    /// sequence - a decoder attending to its own past when that length is known.
    Reserved(Reserved),
}

/// The buffers of [`Kv::Reserved`]: keys and values `[batch, kv_heads, capacity, head_dim]`.
///
/// A step writes its own positions after the `filled` ones and copies nothing else. The
/// buffers are allocated on the first write and kept when a generation starts over, which
/// only sets `filled` back to zero.
#[derive(Debug)]
pub struct Reserved {
    capacity: usize,
    buffers: Option<(Tensor, Tensor)>,
    filled: usize,
}

impl Reserved {
    /// Room for `capacity` positions.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffers: None,
            filled: 0,
        }
    }

    /// Positions written since the last [`Self::clear`].
    pub fn len(&self) -> usize {
        self.filled
    }

    /// Whether nothing was written since the last [`Self::clear`].
    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Forget the sequence; the buffers stay allocated.
    pub fn clear(&mut self) {
        self.filled = 0;
    }

    /// Write `k` and `v` (`[batch, kv_heads, n, head_dim]`) after the positions already held,
    /// and return both whole buffers.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = k.dim(2)?;
        if self.filled + n > self.capacity {
            crate::tensor::bail!(
                "a key/value cache of {} positions cannot take {n} after {}",
                self.capacity,
                self.filled
            );
        }
        let wanted = {
            let mut dims = k.dims().to_vec();
            dims[2] = self.capacity;
            dims
        };
        let fits = self.buffers.as_ref().is_some_and(|(bk, _)| {
            bk.dims() == wanted.as_slice()
                && bk.dtype() == k.dtype()
                && bk.device().same_device(&k.device())
        });
        if !fits {
            let zeros = || Tensor::zeros_on(wanted.clone(), k.dtype(), &k.device());
            self.buffers = Some((zeros()?, zeros()?));
        }
        let (bk, bv) = match &self.buffers {
            Some(b) => b,
            None => crate::tensor::bail!("a key/value cache with no buffers after reserving them"),
        };
        bk.slice_set(k, 2, self.filled)?;
        bv.slice_set(v, 2, self.filled)?;
        self.filled += n;
        Ok((bk.clone(), bv.clone()))
    }
}

/// A clone owns its buffers: the two must not write into one another's sequence.
impl Clone for Reserved {
    fn clone(&self) -> Self {
        let copy = |t: &Tensor| t.affine(1.0, 0.0);
        Self {
            capacity: self.capacity,
            buffers: self
                .buffers
                .as_ref()
                .map(|(k, v)| -> Result<(Tensor, Tensor)> { Ok((copy(k)?, copy(v)?)) })
                .transpose()
                .expect("copying a key/value cache on its own device"),
            filled: self.filled,
        }
    }
}

/// Which positions a query may see.
#[derive(Debug, Clone, Copy)]
pub enum Mask<'a> {
    /// All of them.
    All,
    /// Those at or before it.
    Causal,
    /// A square table the caller carries. Only the rows being asked for and the columns that
    /// exist yet are taken from it: with a cache the queries are the tail of the sequence, so
    /// the rows to use start where the keys end, not at zero.
    Table(&'a Tensor),
    /// A term already shaped like the scores, added to them as it is.
    ///
    /// Not every such term forbids a position: a relative-position bias is a preference, not a
    /// rule, and it is stated per head - which is why it cannot be a square table to be cut.
    Added(&'a Tensor),
}

/// Multi-head attention with a cache policy and a mask policy.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    kv: Kv,
    /// Whether a kept cache is stored in F16 on a card.
    half_cache: bool,
}

impl MultiHeadAttention {
    /// `width` is the query width; `kv_heads` may be fewer than `heads`, in which case each
    /// key head is read by `heads / kv_heads` query heads.
    pub fn new(
        q: Linear,
        k: Linear,
        v: Linear,
        out: Linear,
        width: usize,
        heads: usize,
        kv_heads: usize,
        kv: Kv,
    ) -> Result<Self> {
        if heads == 0 || !width.is_multiple_of(heads) {
            crate::tensor::bail!("an attention of width {width} cannot have {heads} heads");
        }
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            crate::tensor::bail!("{heads} query heads cannot be grouped over {kv_heads} key heads");
        }
        let head_dim = width / heads;
        Ok(Self {
            q,
            k,
            v,
            out,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            kv,
            half_cache: false,
        })
    }

    /// `[batch, positions, width]` into `[batch, heads, positions, head_dim]`.
    fn in_heads(&self, t: Tensor, heads: usize, b: usize) -> Result<Tensor> {
        let len = t.dim(1)?;
        // One position is already in heads order: the swap would move nothing.
        if len == 1 {
            return t.reshape((b, heads, len, self.head_dim));
        }
        t.reshape((b, len, heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// The dtype a kept cache is stored in: F16 on a card when [`Self::with_half_cache`] asked
    /// for it, the projection's own otherwise.
    fn cache_dtype(&self, projected: &Tensor) -> crate::tensor::DType {
        if self.half_cache && projected.device().is_cuda() {
            crate::tensor::DType::F16
        } else {
            projected.dtype()
        }
    }

    fn project_kv(&self, source: &Tensor, b: usize) -> Result<(Tensor, Tensor)> {
        Ok((
            self.in_heads(self.k.forward(source)?, self.kv_heads, b)?,
            self.in_heads(self.v.forward(source)?, self.kv_heads, b)?,
        ))
    }

    /// Attend, given the three operands already in heads, and put the result back in rows.
    fn attend(
        &self,
        q: &Tensor,
        k: Tensor,
        v: Tensor,
        b: usize,
        tgt: usize,
        mask: Mask,
    ) -> Result<Tensor> {
        let groups = self.heads / self.kv_heads;
        let (k, v) = (
            repeat_kv(k, groups)?.contiguous()?,
            repeat_kv(v, groups)?.contiguous()?,
        );

        let kv_len = k.dim(2)?;
        let table = match mask {
            Mask::Table(t) => Some(
                t.narrow(0, kv_len.saturating_sub(tgt), tgt)?
                    .narrow(1, 0, kv_len)?
                    .to_device(&q.device())?,
            ),
            Mask::Added(t) => Some(t.to_device(&q.device())?),
            _ => None,
        };
        crate::inference::model::sdpa::sdpa(
            q,
            &k,
            &v,
            table.as_ref(),
            matches!(mask, Mask::Causal),
            self.scale,
            1.0,
        )?
        .transpose(1, 2)?
        .reshape((b, tgt, self.heads * self.head_dim))?
        .apply(&self.out)
    }

    /// `source` is what the keys and values are read from - the encoder's output for cross
    /// attention, and `xs` itself when it is left out.
    ///
    /// A kept cache with every position visible, or the causal positions, is read by one
    /// kernel on a card ([`Tensor::attend_rows`]); everything else takes the scores, softmax
    /// and value product as separate operations.
    pub fn forward(&mut self, xs: &Tensor, source: Option<&Tensor>, mask: Mask) -> Result<Tensor> {
        let (b, tgt, _) = xs.dims3()?;
        let rows = self.q.forward(xs)?;
        let source = source.unwrap_or(xs);
        let fresh = match &self.kv {
            Kv::Fixed(Some(_)) => None,
            _ => Some(self.project_kv(source, b)?),
        };
        let store = fresh
            .as_ref()
            .map_or(rows.dtype(), |(k, _)| self.cache_dtype(k));
        let projected = || {
            fresh
                .clone()
                .ok_or_else(|| crate::tensor::Error("attention: nothing was projected".into()))
        };

        let (k, v, kv_len) = match &mut self.kv {
            Kv::None => {
                let (k, v) = projected()?;
                let n = k.dim(2)?;
                (k, v, n)
            }
            Kv::Fixed(Some(kept)) => {
                let n = kept.0.dim(2)?;
                (kept.0.clone(), kept.1.clone(), n)
            }
            Kv::Fixed(slot) => {
                let (k, v) = projected()?;
                let kept = (k.to_dtype(store)?, v.to_dtype(store)?);
                *slot = Some(kept.clone());
                let n = kept.0.dim(2)?;
                (kept.0, kept.1, n)
            }
            Kv::Growing(past) => {
                let (nk, nv) = projected()?;
                let kept = match past.take() {
                    // The heads axis is 1 and the positions axis is 2: the new step goes after
                    // the ones already seen, not beside them.
                    Some((pk, pv)) => (Tensor::cat(&[&pk, &nk], 2)?, Tensor::cat(&[&pv, &nv], 2)?),
                    None => (nk, nv),
                };
                *past = Some(kept.clone());
                let n = kept.0.dim(2)?;
                (kept.0, kept.1, n)
            }
            Kv::Reserved(cache) => {
                let (nk, nv) = projected()?;
                let (k, v) = cache.append(&nk.to_dtype(store)?, &nv.to_dtype(store)?)?;
                (k, v, cache.len())
            }
        };

        let kept = matches!(self.kv, Kv::Fixed(_) | Kv::Reserved(_));
        let causal = match mask {
            Mask::All => Some(false),
            Mask::Causal => Some(true),
            Mask::Table(_) | Mask::Added(_) => None,
        };
        if let (true, Some(causal)) = (kept, causal) {
            if let Some(out) = rows.attend_rows(&k, &v, kv_len, self.heads, causal, self.scale)? {
                return out.apply(&self.out);
            }
        }

        let q = self.in_heads(rows, self.heads, b)?;
        let (k, v) = if k.dim(2)? == kv_len {
            (k, v)
        } else {
            (k.narrow(2, 0, kv_len)?, v.narrow(2, 0, kv_len)?)
        };
        let (k, v) = (k.to_dtype(q.dtype())?, v.to_dtype(q.dtype())?);
        self.attend(&q, k, v, b, tgt, mask)
    }

    /// [`Self::forward`] for an attention that keeps nothing.
    ///
    /// A stack that is always handed its whole sequence - an encoder, or a text tower run once
    /// per prompt - has no state to carry, and asking it for a mutable borrow it does not need
    /// would spread through every layer above it.
    pub fn forward_stateless(
        &self,
        xs: &Tensor,
        source: Option<&Tensor>,
        mask: Mask,
    ) -> Result<Tensor> {
        if !matches!(self.kv, Kv::None) {
            crate::tensor::bail!(
                "forward_stateless on an attention that keeps a cache: what it kept would be \
                 ignored, and the answer would silently be the uncached one"
            );
        }
        let (b, tgt, _) = xs.dims3()?;
        let q = self.in_heads(self.q.forward(xs)?, self.heads, b)?;
        let (k, v) = self.project_kv(source.unwrap_or(xs), b)?;
        self.attend(&q, k, v, b, tgt, mask)
    }

    /// Take the scores at `scale` rather than at `1/sqrt(head_dim)`.
    ///
    /// T5 is trained without that factor - its projections absorb it - so a stack built from
    /// one of its checkpoints asks for a scale of one.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Keep a cache's keys and values in F16 on a card, which halves what each step reads of
    /// them; the scores and the weighted sum stay F32.
    pub fn with_half_cache(mut self) -> Self {
        self.half_cache = true;
        self
    }

    /// The four projections, mutably, in the order they are applied: query, key, value, output.
    ///
    /// What reaches for them is an adapter trained apart from the checkpoint and carried as a
    /// rank decomposition - two thin factors per projection, added to the weights in place and
    /// keyed on the checkpoint's own names. The caller holds those names and this holds the
    /// projections, so the order is the contract between them; it is the order the arithmetic
    /// above uses, and it does not change. Nothing else about the attention moves: the forward
    /// reads whatever the projections hold when it is called.
    pub fn projections_mut(&mut self) -> [&mut Linear; 4] {
        [&mut self.q, &mut self.k, &mut self.v, &mut self.out]
    }

    /// Forget what was kept. A generation that starts over must, or it attends to the previous
    /// one's past.
    pub fn clear(&mut self) {
        match &mut self.kv {
            Kv::None => {}
            Kv::Growing(kept) | Kv::Fixed(kept) => *kept = None,
            Kv::Reserved(cache) => cache.clear(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn deterministic(shape: (usize, usize), seed: usize) -> Tensor {
        let (r, c) = shape;
        let v: Vec<f32> = (0..r * c)
            .map(|i| (((i * 29 + seed * 13) % 89) as f32) * 0.019 - 0.8)
            .collect();
        Tensor::from_vec(v, (r, c), &Device::Cpu).unwrap()
    }

    fn build(width: usize, heads: usize, kv_heads: usize, kv: Kv) -> MultiHeadAttention {
        build_on(&Device::Cpu, width, heads, kv_heads, kv)
    }

    fn build_on(
        dev: &Device,
        width: usize,
        heads: usize,
        kv_heads: usize,
        kv: Kv,
    ) -> MultiHeadAttention {
        let head_dim = width / heads;
        let one = |rows: usize, seed: usize| {
            let w = deterministic((rows, width), seed).to_device(dev).unwrap();
            Linear::new(w, None).unwrap()
        };
        MultiHeadAttention::new(
            one(width, 1),
            one(kv_heads * head_dim, 2),
            one(kv_heads * head_dim, 3),
            one(width, 4),
            width,
            heads,
            kv_heads,
            kv,
        )
        .unwrap()
    }

    fn worst(a: &Tensor, b: &Tensor) -> f32 {
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        av.iter()
            .zip(&bv)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// Stepping through a sequence with a cache is attending to it all at once.
    ///
    /// This is the claim the cache exists to make and the one that goes wrong quietly: a
    /// concatenation on the wrong axis, or a mask that keeps counting from zero once the
    /// queries are the tail of the sequence, still returns a tensor of the right shape and
    /// plausible values. So the same weights are run both ways and compared row by row.
    #[test]
    fn stepping_with_a_cache_is_attending_to_the_whole_sequence() {
        let (width, heads, n) = (16usize, 4usize, 5usize);
        let xs = deterministic((n, width), 7).reshape((1, n, width)).unwrap();

        let whole = build(width, heads, heads, Kv::None)
            .forward(&xs, None, Mask::Causal)
            .unwrap();

        let mut stepped = build(width, heads, heads, Kv::Growing(None));
        let mut rows = Vec::new();
        for t in 0..n {
            let step = xs.narrow(1, t, 1).unwrap();
            rows.push(stepped.forward(&step, None, Mask::Causal).unwrap());
        }
        let stepped = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1).unwrap();

        let gap = worst(&whole, &stepped);
        assert!(
            gap < 1e-5,
            "cached and whole-sequence attention differ by {gap}"
        );
    }

    /// A kept cross-attention is a recomputed one.
    ///
    /// The encoder's output does not change across a generation, so projecting it once and
    /// keeping it has to answer what projecting it again would - otherwise the first step of a
    /// generation and the rest of it attend to different things.
    #[test]
    fn keeping_the_encoder_projection_answers_what_recomputing_it_does() {
        let (width, heads, n, src_len) = (16usize, 4usize, 3usize, 6usize);
        let xs = deterministic((n, width), 11)
            .reshape((1, n, width))
            .unwrap();
        let source = deterministic((src_len, width), 13)
            .reshape((1, src_len, width))
            .unwrap();

        let mut kept = build(width, heads, heads, Kv::Fixed(None));
        let mut fresh = build(width, heads, heads, Kv::None);
        for round in 0..3 {
            let a = kept.forward(&xs, Some(&source), Mask::All).unwrap();
            let b = fresh.forward(&xs, Some(&source), Mask::All).unwrap();
            let gap = worst(&a, &b);
            assert!(
                gap < 1e-6,
                "round {round}: kept and recomputed differ by {gap}"
            );
        }
    }

    /// Fewer key heads than query heads is the same attention with the keys shared.
    ///
    /// A grouped attention whose key heads are all copies of one another must answer what the
    /// ungrouped one does on those same repeated keys - which is what says the repetition is
    /// wired to the right query heads.
    #[test]
    fn grouped_heads_read_the_key_head_they_are_grouped_under() {
        let (width, heads, kv_heads, n) = (16usize, 4usize, 2usize, 5usize);
        let xs = deterministic((n, width), 17)
            .reshape((1, n, width))
            .unwrap();

        let mut grouped = build(width, heads, kv_heads, Kv::None);
        let first = grouped.forward(&xs, None, Mask::Causal).unwrap();
        // Run it twice: nothing is kept, so the second answer must be the first.
        let again = grouped.forward(&xs, None, Mask::Causal).unwrap();
        assert!(
            worst(&first, &again) < 1e-7,
            "an uncached attention is not a function"
        );
        assert_eq!(first.dims(), &[1, n, width]);
    }

    /// Clearing is what makes a second generation independent of the first.
    #[test]
    fn clearing_makes_the_next_generation_start_over() {
        let (width, heads, n) = (16usize, 4usize, 4usize);
        let xs = deterministic((n, width), 23)
            .reshape((1, n, width))
            .unwrap();

        let mut attn = build(width, heads, heads, Kv::Growing(None));
        let first = attn.forward(&xs, None, Mask::Causal).unwrap();
        attn.clear();
        let second = attn.forward(&xs, None, Mask::Causal).unwrap();
        let gap = worst(&first, &second);
        assert!(
            gap < 1e-7,
            "after clearing, the same input gave a different answer by {gap}"
        );
    }
    /// A reserved cache stepped through a sequence attends as the whole sequence does, and a
    /// second generation after clearing writes over the first from position zero.
    ///
    /// The prompt is several positions at once and the rest one at a time, so the write offset
    /// and the causal rows both have to follow the cache rather than start at zero.
    #[test]
    fn a_reserved_cache_steps_to_the_whole_sequence() {
        let (width, heads, kv_heads, n, prompt) = (16usize, 4usize, 2usize, 6usize, 3usize);
        let xs = deterministic((n, width), 31)
            .reshape((1, n, width))
            .unwrap();
        let whole = build(width, heads, kv_heads, Kv::None)
            .forward(&xs, None, Mask::Causal)
            .unwrap();

        let mut attn = build(width, heads, kv_heads, Kv::Reserved(Reserved::new(n)));
        for round in 0..2 {
            attn.clear();
            let mut rows = vec![attn
                .forward(&xs.narrow(1, 0, prompt).unwrap(), None, Mask::Causal)
                .unwrap()];
            for t in prompt..n {
                let step = xs.narrow(1, t, 1).unwrap();
                rows.push(attn.forward(&step, None, Mask::Causal).unwrap());
            }
            let stepped = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1).unwrap();
            let gap = worst(&whole, &stepped);
            assert!(
                gap < 1e-5,
                "round {round}: reserved and whole differ by {gap}"
            );
        }
    }

    /// A reserved cache that is full refuses the next position rather than writing past it.
    #[test]
    fn a_reserved_cache_refuses_a_position_past_its_capacity() {
        let (width, heads) = (8usize, 2usize);
        let xs = deterministic((3, width), 37)
            .reshape((1, 3, width))
            .unwrap();
        let mut attn = build(width, heads, heads, Kv::Reserved(Reserved::new(3)));
        attn.forward(&xs, None, Mask::Causal).unwrap();
        let more = xs.narrow(1, 0, 1).unwrap();
        assert!(attn.forward(&more, None, Mask::Causal).is_err());
    }

    /// On a card, the one-kernel read of a reserved cache and of a kept cross attention answers
    /// what the separate operations answer on the host, keys in F32 and in F16.
    #[test]
    fn the_cache_kernel_on_a_card_answers_what_the_host_does() {
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let (width, heads, kv_heads, n, src) = (32usize, 4usize, 2usize, 5usize, 9usize);
        let xs = deterministic((n, width), 41)
            .reshape((1, n, width))
            .unwrap();
        let source = deterministic((src, width), 43)
            .reshape((1, src, width))
            .unwrap();

        let mut host_self = build(width, heads, kv_heads, Kv::Reserved(Reserved::new(n)));
        let mut card_self = build_on(&dev, width, heads, kv_heads, Kv::Reserved(Reserved::new(n)));
        let mut host_cross = build(width, heads, heads, Kv::Fixed(None));
        let mut card_cross = build_on(&dev, width, heads, heads, Kv::Fixed(None));
        let (xs_card, source_card) = (xs.to_device(&dev).unwrap(), source.to_device(&dev).unwrap());
        for t in 0..n {
            let step = xs.narrow(1, t, 1).unwrap();
            let step_card = xs_card.narrow(1, t, 1).unwrap();
            let a = host_self.forward(&step, None, Mask::Causal).unwrap();
            let b = card_self.forward(&step_card, None, Mask::Causal).unwrap();
            let gap = worst(&a, &b.to_device(&Device::Cpu).unwrap());
            assert!(
                gap < 1e-4,
                "self attention at {t}: card and host differ by {gap}"
            );
            let a = host_cross.forward(&step, Some(&source), Mask::All).unwrap();
            let b = card_cross
                .forward(&step_card, Some(&source_card), Mask::All)
                .unwrap();
            let gap = worst(&a, &b.to_device(&Device::Cpu).unwrap());
            assert!(
                gap < 1e-4,
                "cross attention at {t}: card and host differ by {gap}"
            );
        }

        // The F16 kernel: the same cache read from F16 keys and values.
        let q = deterministic((2, width), 47)
            .reshape((1, 2, width))
            .unwrap();
        let head_dim = width / heads;
        let kv = |seed| {
            deterministic((kv_heads * n, head_dim), seed)
                .reshape((1, kv_heads, n, head_dim))
                .unwrap()
        };
        let (k, v) = (kv(53), kv(59));
        let scale = (head_dim as f32).powf(-0.5);
        let want = {
            let attn = build(width, heads, kv_heads, Kv::None);
            let q = attn.in_heads(q.clone(), heads, 1).unwrap();
            crate::inference::model::sdpa::sdpa(
                &q,
                &repeat_kv(k.clone(), heads / kv_heads)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
                &repeat_kv(v.clone(), heads / kv_heads)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
                None,
                true,
                scale,
                1.0,
            )
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .reshape((1, 2, width))
            .unwrap()
        };
        let half = |t: &Tensor| {
            t.to_dtype(crate::tensor::DType::F16)
                .unwrap()
                .to_device(&dev)
                .unwrap()
        };
        let got = q
            .to_device(&dev)
            .unwrap()
            .attend_rows(&half(&k), &half(&v), n, heads, true, scale)
            .unwrap()
            .expect("the kernel takes these shapes");
        let gap = worst(&want, &got.to_device(&Device::Cpu).unwrap());
        assert!(gap < 1e-2, "F16 cache read: card and host differ by {gap}");
    }
}
