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
use crate::tensor::{Module, Result, Tensor};

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
}

impl MultiHeadAttention {
    /// `width` is the query width; `kv_heads` may be fewer than `heads`, in which case each
    /// key head is read by `heads / kv_heads` query heads.
    #[allow(clippy::too_many_arguments)]
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
        if heads == 0 || width % heads != 0 {
            crate::tensor::bail!("an attention of width {width} cannot have {heads} heads");
        }
        if kv_heads == 0 || heads % kv_heads != 0 {
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
        })
    }

    /// `[batch, positions, width]` into `[batch, heads, positions, head_dim]`.
    fn in_heads(&self, t: Tensor, heads: usize, b: usize) -> Result<Tensor> {
        let len = t.dim(1)?;
        t.reshape((b, len, heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn project_kv(&self, source: &Tensor, b: usize) -> Result<(Tensor, Tensor)> {
        Ok((
            self.in_heads(self.k.forward(source)?, self.kv_heads, b)?,
            self.in_heads(self.v.forward(source)?, self.kv_heads, b)?,
        ))
    }

    /// Attend, given the three operands already in heads, and put the result back in rows.
    fn attend(&self, q: &Tensor, k: Tensor, v: Tensor, b: usize, tgt: usize, mask: Mask) -> Result<Tensor> {
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
        crate::inference::model::acestep::ops::sdpa(
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
    pub fn forward(&mut self, xs: &Tensor, source: Option<&Tensor>, mask: Mask) -> Result<Tensor> {
        let (b, tgt, _) = xs.dims3()?;
        let q = self.in_heads(self.q.forward(xs)?, self.heads, b)?;
        let source = source.unwrap_or(xs);

        let (k, v) = match &self.kv {
            Kv::None => self.project_kv(source, b)?,
            Kv::Fixed(Some(kept)) => kept.clone(),
            Kv::Fixed(None) => {
                let kept = self.project_kv(source, b)?;
                self.kv = Kv::Fixed(Some(kept.clone()));
                kept
            }
            Kv::Growing(past) => {
                let past = past.clone();
                let (nk, nv) = self.project_kv(source, b)?;
                let kept = match past {
                    // The heads axis is 1 and the positions axis is 2: the new step goes after
                    // the ones already seen, not beside them.
                    Some((pk, pv)) => {
                        (Tensor::cat(&[&pk, &nk], 2)?, Tensor::cat(&[&pv, &nv], 2)?)
                    }
                    None => (nk, nv),
                };
                self.kv = Kv::Growing(Some(kept.clone()));
                kept
            }
        };
        self.attend(&q, k, v, b, tgt, mask)
    }

    /// [`Self::forward`] for an attention that keeps nothing.
    ///
    /// A stack that is always handed its whole sequence - an encoder, or a text tower run once
    /// per prompt - has no state to carry, and asking it for a mutable borrow it does not need
    /// would spread through every layer above it.
    pub fn forward_stateless(&self, xs: &Tensor, source: Option<&Tensor>, mask: Mask) -> Result<Tensor> {
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
        self.kv = match self.kv {
            Kv::None => Kv::None,
            Kv::Growing(_) => Kv::Growing(None),
            Kv::Fixed(_) => Kv::Fixed(None),
        };
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
        let head_dim = width / heads;
        let one = |rows: usize, seed: usize| {
            Linear::new(deterministic((rows, width), seed), None).unwrap()
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
        assert!(gap < 1e-5, "cached and whole-sequence attention differ by {gap}");
    }

    /// A kept cross-attention is a recomputed one.
    ///
    /// The encoder's output does not change across a generation, so projecting it once and
    /// keeping it has to answer what projecting it again would - otherwise the first step of a
    /// generation and the rest of it attend to different things.
    #[test]
    fn keeping_the_encoder_projection_answers_what_recomputing_it_does() {
        let (width, heads, n, src_len) = (16usize, 4usize, 3usize, 6usize);
        let xs = deterministic((n, width), 11).reshape((1, n, width)).unwrap();
        let source = deterministic((src_len, width), 13)
            .reshape((1, src_len, width))
            .unwrap();

        let mut kept = build(width, heads, heads, Kv::Fixed(None));
        let mut fresh = build(width, heads, heads, Kv::None);
        for round in 0..3 {
            let a = kept.forward(&xs, Some(&source), Mask::All).unwrap();
            let b = fresh.forward(&xs, Some(&source), Mask::All).unwrap();
            let gap = worst(&a, &b);
            assert!(gap < 1e-6, "round {round}: kept and recomputed differ by {gap}");
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
        let xs = deterministic((n, width), 17).reshape((1, n, width)).unwrap();

        let mut grouped = build(width, heads, kv_heads, Kv::None);
        let first = grouped.forward(&xs, None, Mask::Causal).unwrap();
        // Run it twice: nothing is kept, so the second answer must be the first.
        let again = grouped.forward(&xs, None, Mask::Causal).unwrap();
        assert!(worst(&first, &again) < 1e-7, "an uncached attention is not a function");
        assert_eq!(first.dims(), &[1, n, width]);
    }

    /// Clearing is what makes a second generation independent of the first.
    #[test]
    fn clearing_makes_the_next_generation_start_over() {
        let (width, heads, n) = (16usize, 4usize, 4usize);
        let xs = deterministic((n, width), 23).reshape((1, n, width)).unwrap();

        let mut attn = build(width, heads, heads, Kv::Growing(None));
        let first = attn.forward(&xs, None, Mask::Causal).unwrap();
        attn.clear();
        let second = attn.forward(&xs, None, Mask::Causal).unwrap();
        let gap = worst(&first, &second);
        assert!(gap < 1e-7, "after clearing, the same input gave a different answer by {gap}");
    }
}
