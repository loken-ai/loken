//! The indexer's cache and selection on a card, for the decode step a graph records.
//!
//! The cache the host keeps for prefill and for eager steps and the one the card keeps for the
//! recorded step are prefixes of one sequence: whichever holds more has the tail the other lacks,
//! and the lacking one takes it over before it is used. The card's holds the pooled key of every
//! complete block and the raw rows of the block being filled, which is all a later token needs.

use super::*;
use crate::inference::moe_cuda::{
    address_f32, indexer_append, indexer_rows_in, indexer_rows_out, indexer_select, IndexerShape,
    StepPos,
};
use crate::tensor::Device;

/// The card side of one layer's indexer.
pub struct CardIndexer {
    dev: Device,
    shape: IndexerShape,
    /// `[groups, head_dim, 32]` F32: the pooled key of every complete block, 32 blocks to a group
    /// laid out `[dim][lane]`.
    pooled: Tensor,
    /// `[block, head_dim]` F32: the raw rows of the block being filled.
    tail: Tensor,
    /// `[blocks]` F32: scratch for the scores of one query.
    scores: Tensor,
    /// `[top_k + block]` U32 and `[1]` U32: the tokens the last query kept, and how many.
    sel: Tensor,
    count: Tensor,
    /// `[4 * half]` F32: the rotation tables at the first position of the block the step's
    /// token completes, then at the step's position.
    rot: Tensor,
    q_gamma: Tensor,
    k_gamma: Tensor,
    /// Leading tokens whose state the card holds.
    tokens: usize,
    /// The position the next step runs at.
    step_pos: usize,
    /// The rotation tables on their way to `rot`.
    staging: Staging,
}

/// A page-locked buffer the host fills for a small copy to the card that does not stop the host:
/// the copy is queued on the card's stream, and the next fill waits only for that copy to be done.
struct Staging {
    host: *mut f32,
    len: usize,
    copied: Option<crate::tensor::cuda_ext::CudaEvent>,
}

// SAFETY: the page-locked block is owned by this staging alone, freed once on drop, and filled by
// one thread at a time through `&mut self`.
unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

impl Staging {
    fn new(len: usize) -> Result<Self> {
        let host = crate::tensor::cuda_ext::malloc_host(len * std::mem::size_of::<f32>())
            .map_err(crate::tensor::Error::msg)? as *mut f32;
        Ok(Self {
            host,
            len,
            copied: None,
        })
    }

    /// Queue the copy of `data` into `dst` on `dev`'s stream.
    fn send(&mut self, data: &[f32], dst: &Tensor, dev: &Device) -> Result<()> {
        if data.len() != self.len {
            crate::tensor::bail!(
                "a staged copy of {} floats into room for {}",
                data.len(),
                self.len
            );
        }
        if let Some(done) = self.copied.take() {
            done.synchronize()
                .map_err(|e| crate::tensor::Error::msg(format!("staged copy: {e:?}")))?;
        }
        let stream = dev.as_cuda_device()?.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("staged copy: {e:?}")))?;
        // SAFETY: `host` holds `len` floats until the event below has completed, and `dst`
        // is a card tensor of at least that size.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.host, self.len);
            let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                address_f32(dst)?,
                self.host as *const std::ffi::c_void,
                self.len * std::mem::size_of::<f32>(),
                stream.cu_stream(),
            );
            if r != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                crate::tensor::bail!("staged copy to the card failed: {r:?}");
            }
        }
        self.copied = Some(crate::tensor::cuda_ext::record_event(&stream)?);
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if let Some(done) = self.copied.take() {
            let _ = done.synchronize();
        }
        // SAFETY: allocated by `malloc_host` in `new` and freed only here, after its last copy.
        unsafe { crate::tensor::cuda_ext::free_host(self.host as *mut std::ffi::c_void) }
    }
}

/// Bytes of shared memory a launch may use without asking the device for more.
const SMEM_LIMIT: usize = 48 << 10;

impl HostRope {
    /// Rotation pairs.
    pub fn half(&self) -> usize {
        self.inv_freq.len()
    }

    /// The cosines then the sines of the pairs at position `pos`, as [`Self::apply`] computes
    /// them.
    pub fn table(&self, pos: usize) -> Vec<f32> {
        let half = self.inv_freq.len();
        let mut t = vec![0f32; 2 * half];
        for (i, f) in self.inv_freq.iter().enumerate() {
            let (sin, cos) = (pos as f32 * f).sin_cos();
            t[i] = cos;
            t[half + i] = sin;
        }
        t
    }
}

impl SparseAttention {
    /// Whether the card kernels serve this indexer and an attention of heads of `head_dim`: their
    /// working sets fit the shared memory a launch has, and the rows are whole vectors.
    pub fn card_serves(&self, head_dim: usize) -> bool {
        let m = self.meta;
        let room = m.top_k + m.block;
        head_dim.is_multiple_of(8)
            && head_dim <= 1024
            && m.head_dim.is_multiple_of(2)
            && crate::inference::moe_cuda::indexer_select_smem(m.heads, m.head_dim) <= SMEM_LIMIT
            && crate::inference::moe_cuda::selected_attention_smem(room, head_dim) <= SMEM_LIMIT
    }

    /// Room for the kept tokens of a query.
    pub fn card_room(&self) -> usize {
        self.meta.top_k + self.meta.block
    }

    /// Give the indexer its cache on `dev`, with room for `tokens`. Idempotent.
    pub fn attach_card(&mut self, dev: &Device, tokens: usize) -> Result<()> {
        if self.card.is_some() {
            return self.card_room_for(tokens);
        }
        let m = self.meta;
        let hd = m.head_dim;
        let blocks = tokens / m.block + 2;
        let on_card = |v: &[f32]| Tensor::from_slice(v, (hd,), &Device::Cpu)?.to_device(dev);
        self.card = Some(CardIndexer {
            dev: dev.clone(),
            shape: IndexerShape {
                heads: m.heads,
                hd,
                block: m.block,
                budget: m.budget_blocks(),
                half: self.rope.half(),
                eps: self.eps,
            },
            pooled: Tensor::zeros_on((blocks.div_ceil(32), hd, 32), DType::F32, dev)?,
            tail: Tensor::zeros_on((m.block, hd), DType::F32, dev)?,
            scores: Tensor::zeros_on((blocks.div_ceil(32) * 32,), DType::F32, dev)?,
            sel: Tensor::zeros_on((m.top_k + m.block,), DType::U32, dev)?,
            count: Tensor::zeros_on((1,), DType::U32, dev)?,
            rot: Tensor::zeros_on((4 * self.rope.half(),), DType::F32, dev)?,
            q_gamma: on_card(&self.q_gamma)?,
            k_gamma: on_card(&self.k_gamma)?,
            tokens: 0,
            step_pos: 0,
            staging: Staging::new(4 * self.rope.half())?,
        });
        Ok(())
    }

    /// Room on the card for `tokens`: the pooled keys grow with the sequence, and a larger cache
    /// takes over the old one's blocks. The address of the pooled keys moves with it, so a step
    /// recorded against them must end first, which a cache that grew to `tokens` already did.
    pub fn card_room_for(&mut self, tokens: usize) -> Result<()> {
        let m = self.meta;
        let Some(card) = self.card.as_mut() else {
            return Ok(());
        };
        let blocks = tokens / m.block + 2;
        let groups = blocks.div_ceil(32);
        let have = card.pooled.dim(0)?;
        if groups <= have {
            return Ok(());
        }
        let grown = Tensor::zeros_on((groups, m.head_dim, 32), DType::F32, &card.dev)?;
        grown.slice_set(&card.pooled, 0, 0)?;
        card.pooled = grown;
        card.scores = Tensor::zeros_on((groups * 32,), DType::F32, &card.dev)?;
        Ok(())
    }

    /// Make the card hold the first `pos` tokens and the rotation tables of the step at `pos`.
    /// A card that lacks tokens takes them from the host, which must then hold exactly `pos`.
    pub fn card_hold(&mut self, pos: usize) -> Result<()> {
        let m = self.meta;
        let hd = m.head_dim;
        let host_len = self.cache.len();
        let Some(card) = self.card.as_mut() else {
            crate::tensor::bail!("the indexer has no cache on a card");
        };
        if card.tokens < pos {
            if host_len != pos {
                crate::tensor::bail!(
                    "the card's indexer holds {} tokens and the host's {host_len}, the step is at {pos}",
                    card.tokens
                );
            }
            let have = card.tokens / m.block;
            let blocks = self.cache.blocks(&m);
            if blocks > have {
                let rows = self.cache.pooled_from(have, &m);
                let rows = Tensor::from_slice(rows, (blocks - have, hd), &Device::Cpu)?
                    .to_device(&card.dev)?;
                indexer_rows_in(&rows, &card.pooled, have, blocks - have, hd)?;
            }
            let tail = self.cache.tail();
            if !tail.is_empty() {
                let rows = Tensor::from_slice(tail, (tail.len() / hd, hd), &Device::Cpu)?
                    .to_device(&card.dev)?;
                card.tail.slice_set(&rows, 0, 0)?;
            }
            card.tokens = pos;
        }
        // The tables of a token that completes a block read the block's first position.
        let half = self.rope.half();
        let mut tables = if (pos + 1).is_multiple_of(m.block) {
            self.rope.table(pos + 1 - m.block)
        } else {
            vec![0f32; 2 * half]
        };
        tables.extend(self.rope.table(pos));
        let dev = card.dev.clone();
        card.staging.send(&tables, &card.rot, &dev)?;
        card.step_pos = pos;
        Ok(())
    }

    /// The step's token `x` (the sublayer input) joins the card's cache and its query selects
    /// the tokens to read, at `pos`: `sel` and `count` hold the answer.
    pub fn card_step(&mut self, x: &Tensor, pos: StepPos) -> Result<()> {
        let x = x.to_dtype(DType::F32)?;
        let key = self.k_proj.forward(&x)?.contiguous()?;
        let q = self.q_proj.forward(&x)?.contiguous()?;
        let Some(card) = self.card.as_mut() else {
            crate::tensor::bail!("the indexer has no cache on a card");
        };
        indexer_append(
            &key,
            &card.tail,
            &card.pooled,
            &card.k_gamma,
            &card.rot,
            pos,
            &card.shape,
        )?;
        let at_pos = card
            .rot
            .narrow(0, 2 * card.shape.half, 2 * card.shape.half)?;
        indexer_select(
            &q,
            &card.q_gamma,
            &at_pos,
            &card.pooled,
            &card.scores,
            &card.sel,
            &card.count,
            pos,
            &card.shape,
        )?;
        // A step that runs now has made the token on the card; one that is only being recorded
        // makes it when the graph replays, which the engine reports by its token count.
        if !crate::tensor::cuda_ext::capture_active(&card.dev) {
            card.tokens = card.tokens.max(card.step_pos + 1);
        }
        Ok(())
    }

    /// The tokens the last card step kept, and how many.
    pub fn card_selection(&self) -> Result<(&Tensor, &Tensor)> {
        match self.card.as_ref() {
            Some(card) => Ok((&card.sel, &card.count)),
            None => crate::tensor::bail!("the indexer has no cache on a card"),
        }
    }

    /// The engine says the caches hold `len` tokens: the host's last `len - host` of them, if it
    /// lacks any, were made by recorded steps on the card.
    pub fn card_confirm(&mut self, len: usize) {
        let host = self.cache.len();
        if let Some(card) = self.card.as_mut() {
            if host < len && card.tokens >= host {
                card.tokens = len;
            }
        }
    }

    /// The card's tokens are of another sequence now.
    pub(super) fn forget_card(&mut self) {
        if let Some(card) = self.card.as_mut() {
            card.tokens = 0;
        }
    }

    /// The host has taken over the sequence from `len` on.
    pub(super) fn truncate_card(&mut self, len: usize) {
        if let Some(card) = self.card.as_mut() {
            card.tokens = card.tokens.min(len);
        }
    }

    /// The host cache holds what the card holds beyond it, when the card holds exactly `upto`
    /// tokens, the position the host is about to continue from.
    pub(super) fn sync_host_from_card(&mut self, upto: usize) -> Result<()> {
        let m = self.meta;
        let Some(card) = self.card.as_ref() else {
            return Ok(());
        };
        if card.tokens <= self.cache.len() || card.tokens != upto {
            return Ok(());
        }
        let (pooled, tail) = card.beyond(&self.cache, &m)?;
        self.cache.extend(&pooled, &tail, card.tokens, &m);
        Ok(())
    }

    /// The host cache brought up to what the card holds: what a snapshot of the sequence reads.
    pub fn current_cache(&self) -> Result<IndexerCache> {
        let mut cache = self.cache.clone();
        if let Some(card) = self.card.as_ref() {
            if card.tokens > cache.len() {
                let (pooled, tail) = card.beyond(&cache, &self.meta)?;
                cache.extend(&pooled, &tail, card.tokens, &self.meta);
            }
        }
        Ok(cache)
    }
}

impl CardIndexer {
    /// The pooled keys of the blocks `cache` lacks and the raw rows of the block being filled.
    fn beyond(&self, cache: &IndexerCache, meta: &IndexerMeta) -> Result<(Vec<f32>, Vec<f32>)> {
        let host = |t: Tensor| -> Result<Vec<f32>> {
            t.contiguous()?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        let have = cache.blocks(meta);
        let blocks = self.tokens / meta.block;
        let pooled = match blocks > have {
            true => host(indexer_rows_out(
                &self.pooled,
                have,
                blocks - have,
                meta.head_dim,
            )?)?,
            false => Vec::new(),
        };
        let rows = self.tokens % meta.block;
        let tail = match rows > 0 {
            true => host(self.tail.narrow(0, 0, rows)?)?,
            false => Vec::new(),
        };
        Ok((pooled, tail))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QTensor};

    const HIDDEN: usize = 24;

    /// An indexer with float projections, on the host or on `dev`, the same whichever it is.
    fn indexer(meta: IndexerMeta, base: f32, dev: &Device) -> SparseAttention {
        let weights = |rows: usize, seed: usize| {
            let v: Vec<f32> = (0..rows * HIDDEN)
                .map(|i| (((i * 37 + seed * 11) % 101) as f32 - 50.0) * 0.02)
                .collect();
            let t = Tensor::from_vec(v, (rows, HIDDEN), dev).unwrap();
            QMatMul::from_qtensor(QTensor::quantize_onto(&t, GgmlDType::F32, dev).unwrap()).unwrap()
        };
        let gamma = |seed: usize| {
            (0..meta.head_dim)
                .map(|i| 0.5 + ((i * 7 + seed) % 11) as f32 * 0.1)
                .collect::<Vec<f32>>()
        };
        SparseAttention::new(
            meta,
            weights(meta.heads * meta.head_dim, 1),
            weights(meta.head_dim, 2),
            gamma(3),
            gamma(5),
            1e-6,
            meta.head_dim,
            base,
        )
    }

    /// The sublayer input of token `t`; `period` repeats it, so that blocks pool to equal keys.
    fn token_row(t: usize, period: usize, dev: &Device) -> Tensor {
        let t = t % period;
        let v: Vec<f32> = (0..HIDDEN)
            .map(|i| (((i * 53 + t * 29 + 7) % 89) as f32 - 44.0) * 0.03)
            .collect();
        Tensor::from_vec(v, (1, 1, HIDDEN), dev).unwrap()
    }

    fn position(pos: usize, dev: &Device) -> Tensor {
        Tensor::new(&[pos as i32], dev).unwrap()
    }

    /// The tokens the card kept at its last step.
    fn kept_on_card(card: &SparseAttention) -> Vec<u32> {
        let (sel, count) = card.card_selection().unwrap();
        let n = count.to_vec1::<u32>().unwrap()[0] as usize;
        sel.to_vec1::<u32>().unwrap()[..n].to_vec()
    }

    /// Steps `tokens` tokens through the host and through the card, one at a time, and says where
    /// the two keep different tokens.
    fn disagreements(meta: IndexerMeta, base: f32, period: usize, tokens: usize) -> Vec<usize> {
        let card_dev = Device::new_cuda(0).expect("a card for the card test");
        let mut host = indexer(meta, base, &card_dev);
        let mut card = indexer(meta, base, &card_dev);
        // The cache starts small and grows with the sequence, taking its blocks along.
        card.attach_card(&card_dev, 8).unwrap();
        let mut differ = Vec::new();
        for pos in 0..tokens {
            card.card_room_for(pos + 1).unwrap();
            let x = token_row(pos, period, &card_dev);
            let want = match host.observe(&x, pos).unwrap() {
                Some(mut rows) => rows.remove(0),
                None => (0..=pos as u32).collect(),
            };
            card.card_hold(pos).unwrap();
            card.card_step(&x, StepPos::Card(&position(pos, &card_dev)))
                .unwrap();
            if kept_on_card(&card) != want {
                differ.push(pos);
            }
        }
        differ
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_card_keeps_the_tokens_the_host_keeps() {
        let meta = IndexerMeta {
            heads: 2,
            head_dim: 16,
            top_k: 16,
            block: 4,
        };
        assert_eq!(disagreements(meta, 10000.0, 1000, 90), Vec::<usize>::new());
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_card_keeps_the_same_blocks_among_equal_scores() {
        // A frequency base so large that nothing rotates, and a row that repeats every 8
        // tokens: blocks pool to a few distinct keys, so most scores tie.
        let meta = IndexerMeta {
            heads: 3,
            head_dim: 16,
            top_k: 16,
            block: 4,
        };
        assert_eq!(disagreements(meta, 1e30, 8, 90), Vec::<usize>::new());
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_card_keeps_the_tokens_the_host_keeps_at_the_width_of_a_model() {
        let meta = IndexerMeta {
            heads: 4,
            head_dim: 128,
            top_k: 256,
            block: 4,
        };
        assert_eq!(disagreements(meta, 10000.0, 1000, 700), Vec::<usize>::new());
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn a_card_takes_over_the_host_cache_and_the_host_the_card_cache() {
        let meta = IndexerMeta {
            heads: 2,
            head_dim: 16,
            top_k: 16,
            block: 4,
        };
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let mut host = indexer(meta, 10000.0, &dev);
        let mut card = indexer(meta, 10000.0, &dev);
        let prompt = 37;
        for pos in 0..prompt {
            host.observe(&token_row(pos, 1000, &dev), pos).unwrap();
            card.observe(&token_row(pos, 1000, &dev), pos).unwrap();
        }
        // The card lacks the prompt: it takes it from the host and keeps the tokens the host
        // keeps from there on.
        card.attach_card(&dev, 128).unwrap();
        for pos in prompt..prompt + 30 {
            let x = token_row(pos, 1000, &dev);
            let want = host.observe(&x, pos).unwrap().unwrap().remove(0);
            card.card_hold(pos).unwrap();
            card.card_step(&x, StepPos::Host(pos)).unwrap();
            assert_eq!(kept_on_card(&card), want, "position {pos}");
        }
        // The host lacks what the card made: asked for a snapshot, or to continue, it takes it.
        let snapshot = card.snapshot_cache().unwrap();
        assert_eq!(snapshot.len(), prompt + 30);
        assert_eq!(
            snapshot.pooled_from(0, &meta),
            host.cache.pooled_from(0, &meta)
        );
        assert_eq!(snapshot.tail(), host.cache.tail());
        let mut late = indexer(meta, 10000.0, &dev);
        late.attach_card(&dev, 128).unwrap();
        for pos in 0..prompt {
            late.observe(&token_row(pos, 1000, &dev), pos).unwrap();
        }
        for pos in prompt..prompt + 30 {
            late.card_hold(pos).unwrap();
            late.card_step(&token_row(pos, 1000, &dev), StepPos::Host(pos))
                .unwrap();
        }
        let x = token_row(prompt + 30, 1000, &dev);
        let want = host.observe(&x, prompt + 30).unwrap().unwrap().remove(0);
        let got = late.observe(&x, prompt + 30).unwrap().unwrap().remove(0);
        assert_eq!(got, want);
    }
}
