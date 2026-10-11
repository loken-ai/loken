//! The draft head: one more layer of the model's own kind, fed a token's embedding fused with the
//! hidden state before it, that predicts the token after.
//!
//! The head sees pairs `(token at p + 1, hidden at p)` and writes its own cache over the
//! positions, so it keeps no recurrent state: cutting it back is cutting its cache. Its output
//! goes through its own final mixer and the model's output head.

use super::hyper::HyperMixer;
use super::projection::QMatMul;
use super::{GenericHeteroTransformer, GenericTransformerLayer, LayerDevice};
use crate::tensor::layer::RmsNorm;
use crate::tensor::{DType, Device, Result, Tensor, D};

pub(super) struct DraftHead {
    pub(super) layer: GenericTransformerLayer,
    pub(super) layer_dev: LayerDevice,
    pub(super) norm_embedding: RmsNorm,
    /// `[streams * dim]` F32, the scale of the stream-wise norm the hidden state goes through.
    pub(super) hidden_gamma: Tensor,
    pub(super) fc_embedding: QMatMul,
    pub(super) fc_hidden: QMatMul,
    /// Collapses the layer's streams into the row the output head reads.
    pub(super) mixer: HyperMixer,
    pub(super) streams: usize,
    pub(super) dim: usize,
    pub(super) eps: f64,
    /// Positions the head's cache holds.
    pub(super) len: usize,
    /// Positions that hold a pair of the sequence itself, not one the head guessed ahead.
    pub(super) committed: usize,
    /// The last row of the prompt chunk before, whose pair waits for the next chunk's first token.
    pub(super) carried: Option<Tensor>,
    /// The layer's indexer as it stood with `committed` pairs read: what a cut back to that
    /// length returns it to.
    pub(super) indexer_mark: Option<(usize, super::qsa::IndexerCache)>,
    /// The head's guesses as the kernels of a recorded step.
    #[cfg(feature = "cuda")]
    pub(super) step: Option<DraftStep>,
}

/// The head's guesses as the kernels of recorded steps, replayed with the tokens' embeddings,
/// the rows' positions and tables refreshed before each replay: one graph per count of pairs
/// takes the pairs through the layer and leaves the last row; one names the token after the
/// row; one takes a guessed token with the row through the layer and names the next.
#[cfg(feature = "cuda")]
pub(super) struct DraftStep {
    /// The tokens' embeddings, `[1, VERIFY_ROWS, dim]` F32.
    emb: Tensor,
    /// The hidden rows the pairs read, `[1, VERIFY_ROWS, streams * dim]` F32.
    rows_in: Tensor,
    /// The row a guess reads and the row it writes, `[1, 1, streams * dim]` F32.
    hidden: Tensor,
    /// The token named, `[1]` U32, by the card's argmax.
    ids: Tensor,
    /// The softmax mass relative to the top logit, `[1, 1]` F32.
    read: Tensor,
    /// The rows' positions, `[VERIFY_ROWS]` i32.
    pos: Tensor,
    /// The pairs through the layer, by their count, the most recently recorded last: at most
    /// [`DRAFT_PAIRS_GRAPHS`] of them, each reserving memory on the card.
    pairs: Vec<(usize, HeldGraph)>,
    /// The head over the row alone.
    head: Option<HeldGraph>,
    /// The layer over a token and the row, then the head.
    layer: Option<HeldGraph>,
    /// The buffers the graphs were recorded against: the cache's keys and the indexer's births.
    key: Option<(crate::tensor::ops::traits::TensorId, u64)>,
    /// Replays that failed since the load.
    failures: u32,
    /// The recorded guesses are given up on: a capture failed or a replay disagreed.
    blocked: bool,
}

/// Replays that may fail before the head's guesses stay eager.
#[cfg(feature = "cuda")]
const DRAFT_RECORD_FAILURES: u32 = 3;
/// Pairs graphs the head keeps at once: a draft's pairs are the tokens the last verify kept
/// and one more, so a few counts cover nearly every cycle.
#[cfg(feature = "cuda")]
const DRAFT_PAIRS_GRAPHS: usize = 3;

/// A recorded graph the model holds: launched only by the thread that holds the model, on the
/// stream it was captured on, which is what its handles need.
#[cfg(feature = "cuda")]
pub(super) struct HeldGraph(crate::tensor::cuda_ext::CapturedGraph);

#[cfg(feature = "cuda")]
unsafe impl Send for HeldGraph {}
#[cfg(feature = "cuda")]
unsafe impl Sync for HeldGraph {}

/// What a recorded step of the head runs.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum DraftKind {
    /// `n` pairs through the layer; the last row is left for the head.
    Pairs(usize),
    /// The head over the row.
    Head,
    /// A token with the row through the layer, then the head.
    Step,
}

impl DraftHead {
    /// The head's input for `emb` `[1, n, dim]` and `hidden` `[1, n, streams * dim]`: the
    /// embedding normed and projected, the hidden state normed as one row of `streams * dim`
    /// values and each stream projected, the first added to every stream of the second.
    fn fuse(&self, emb: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let (b, n, _) = hidden.dims3()?;
        let e = self.norm_embedding.forward(emb)?;
        let e = self
            .fc_embedding
            .forward(&e.reshape((b * n, self.dim))?)?
            .reshape((b, n, 1, self.dim))?;
        let ms = hidden.sqr()?.mean_keepdim(D::Minus1)?;
        let inv = (ms + self.eps)?.sqrt()?.recip()?;
        let h = hidden
            .broadcast_mul(&inv)?
            .broadcast_mul(&self.hidden_gamma.to_device(&hidden.device())?)?;
        let h = self
            .fc_hidden
            .forward(&h.reshape((b * n * self.streams, self.dim))?)?
            .reshape((b, n, self.streams, self.dim))?;
        h.broadcast_add(&e)?
            .reshape((b, n, self.streams * self.dim))
    }
}

impl DraftHead {
    /// Remember the indexer as it stands: `committed` pairs are read.
    fn mark_indexer(&mut self) {
        let mark = self
            .layer
            .sparse
            .as_ref()
            .and_then(|s| s.snapshot_cache().ok());
        self.indexer_mark = mark.map(|c| (self.committed, c));
    }
}

impl GenericHeteroTransformer {
    /// Whether the model came with a draft head.
    pub fn has_draft_head(&self) -> bool {
        self.draft.is_some()
    }

    /// The expert caches' mailboxes as they stand, one line per layer with one, the draft head's
    /// last: what a wait on the card that lasts is told by.
    pub fn tier_mailboxes(&self) -> Vec<String> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| (format!("layer {i}"), l))
            .chain(
                self.draft
                    .as_ref()
                    .map(|d| ("draft head".to_string(), &d.layer)),
            )
            .filter_map(|(name, l)| {
                let tier = l.moe.as_ref()?.tier.as_ref()?;
                Some(format!("{name}: {}", tier.mailbox_state()?))
            })
            .collect()
    }

    /// Positions the draft head's cache holds.
    pub fn draft_len(&self) -> usize {
        self.draft.as_ref().map_or(0, |d| d.len)
    }

    /// Run the head over the pairs `(tokens[i], hidden[i])` that sit at positions `at..`, and
    /// return the wide rows it wrote, `[1, n, streams * dim]`. `hidden` is the model's own
    /// last-layer state of the position before each token. With `kv_only` the layer stops after
    /// its token mixer: the rows leave their keys and values in its caches, and what is returned
    /// is not the head's output.
    pub fn draft_rows(
        &mut self,
        tokens: &[u32],
        hidden: &Tensor,
        at: usize,
        kv_only: bool,
    ) -> Result<Tensor> {
        let n = tokens.len();
        let (dev, edev) = {
            let d = self.draft.as_ref().ok_or_else(|| {
                crate::tensor::Error::msg("this model has no draft head".to_string())
            })?;
            let dev = match &d.layer_dev {
                LayerDevice::Cuda(i) => self.cuda_devices.get(i).cloned().unwrap_or(Device::Cpu),
                LayerDevice::Cpu => Device::Cpu,
            };
            (dev, self.embeddings.embeddings().device().clone())
        };
        let ids = Tensor::from_vec(tokens.to_vec(), (1, n), &edev)?;
        let mut emb = self.embeddings.forward(&ids)?.to_dtype(DType::F32)?;
        if let Some(scale) = self.config.embed_scale {
            emb = (emb * scale)?;
        }
        let emb = emb.to_device(&dev)?;
        let wide = hidden.elem_count() / n.max(1);
        let hidden = hidden
            .to_dtype(DType::F32)?
            .reshape((1, n, wide))?
            .to_device(&dev)?;
        let mut draft = self.draft.take().expect("checked above");
        let x = draft.fuse(&emb, &hidden);
        let first = self.layers.len();
        // The head's layer stands after the model's own for the pass, so the loop that runs the
        // stack - masks, segment hops, the layer's own forward - runs it unchanged.
        draft.layer.kv_only = kv_only;
        self.layers.push(draft.layer);
        self.layer_devs.push(draft.layer_dev.clone());
        let out = x.and_then(|x| self.run_layers_in(x, None, at, n, first, first + 1));
        let layer_dev = self.layer_devs.pop().expect("pushed above");
        draft.layer = self.layers.pop().expect("pushed above");
        draft.layer.kv_only = false;
        draft.layer_dev = layer_dev;
        draft.len = draft.len.max(at + n);
        self.draft = Some(draft);
        out.map(|(wide, _)| wide)
    }

    /// The logits of the rows `wide` `[1, n, streams * dim]` the head wrote, `[n, vocab]`.
    pub fn draft_logits(&self, wide: &Tensor) -> Result<Tensor> {
        let draft = self
            .draft
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("this model has no draft head".to_string()))?;
        let (b, n, _) = wide.dims3()?;
        let rows = wide.to_device(&draft.mixer.device())?;
        let (normed, _) = draft.mixer.mix(&rows)?;
        let normed = normed.reshape((b * n, draft.dim))?;
        match (&self.output_proj_cuda_qmm, &self.output_proj_dev) {
            (Some(proj), Some(dev)) => proj.forward(&normed.to_device(dev)?),
            _ => self.output_proj.forward(&normed.to_device(&Device::Cpu)?),
        }
    }

    /// Read a prompt chunk `ids` that starts at `index_pos`, whose last-layer rows are `hidden`
    /// `[1, n, streams * dim]`: the pair `(ids[i + 1], hidden[i])` is what the head reads at its
    /// next position, the chunk's last row waiting for the token that follows it.
    pub(super) fn draft_ingest(
        &mut self,
        ids: &[u32],
        hidden: &Tensor,
        index_pos: usize,
    ) -> Result<()> {
        if index_pos == 0 {
            self.draft_reset();
        }
        let c = ids.len();
        let wide = hidden.elem_count() / c.max(1);
        let hidden = hidden.to_dtype(DType::F32)?.reshape((1, c, wide))?;
        let Some(d) = self.draft.as_mut() else {
            return Ok(());
        };
        let carried = d.carried.take();
        let committed = d.committed;
        let head = hidden.narrow(1, 0, c - 1)?;
        let (tokens, rows) = match carried {
            Some(prev) => (
                ids.to_vec(),
                Tensor::cat(&[&prev.to_device(&head.device())?, &head], 1)?,
            ),
            None => (ids[1..].to_vec(), head),
        };
        let last = hidden.narrow(1, c - 1, 1)?;
        if !tokens.is_empty() {
            self.draft_rows(&tokens, &rows, committed, true)?;
        }
        if let Some(d) = self.draft.as_mut() {
            d.committed = committed + tokens.len();
            d.carried = Some(last);
            d.mark_indexer();
        }
        Ok(())
    }

    /// Start the head's history over: the next pairs it is given are the first it holds.
    pub fn draft_reset(&mut self) {
        self.draft_trim(0);
        if let Some(d) = self.draft.as_mut() {
            d.committed = 0;
            d.carried = None;
            d.indexer_mark = None;
        }
    }

    /// Up to `k` tokens that follow the sequence, guessed by the head.
    ///
    /// `tokens` are the tokens the model has been given since the last call plus the one it has
    /// yet to be given, and the rows of `last_feature` the hidden states that went before them:
    /// the pair `(tokens[i], hidden[i])` is what the head reads at its position `committed + i`.
    /// What the head guessed ahead the last time is cut off first, and the pairs are read in one
    /// pass; each further guess reads the one before it with the head's own output as hidden.
    pub fn draft_tokens(&mut self, tokens: &[u32], k: usize, min_odds: f32) -> Result<Vec<u32>> {
        let n = tokens.len();
        let Some(feature) = self.last_feature.take() else {
            return Ok(Vec::new());
        };
        let rows = if feature.rank() == 3 {
            feature.dim(1)?
        } else {
            1
        };
        if n == 0 || k == 0 || rows < n {
            return Ok(Vec::new());
        }
        let committed = self.draft.as_ref().map_or(0, |d| d.committed);
        self.draft_trim(committed);
        let feature = if feature.rank() == 3 {
            feature.narrow(1, 0, n)?
        } else {
            feature
        };
        // Under the stage profile, what the pairs, each guess's layer, its logits and its odds
        // cost, with the card drained at each mark.
        let prof = crate::inference::offload::stage_prof_enabled();
        let mark = |name: &'static str, since: &mut std::time::Instant, dev: &Device| {
            if prof {
                let _ = dev.synchronize();
                crate::inference::offload::prof_record(name, since.elapsed().as_nanos() as u64);
                *since = std::time::Instant::now();
            }
        };
        let dev = feature.device().clone();
        let mut since = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        if let Some(out) = self.draft_tokens_recorded(tokens, &feature, committed, k, min_odds)? {
            mark("draft: recorded", &mut since, &dev);
            return Ok(out);
        }
        let wide = self.draft_rows(tokens, &feature, committed, false)?;
        mark("draft: the pairs", &mut since, &dev);
        if let Some(d) = self.draft.as_mut() {
            d.committed = committed + n;
            d.mark_indexer();
        }
        let mut at = committed + n;
        let mut last = wide.narrow(1, n - 1, 1)?;
        let mut out = Vec::with_capacity(k);
        for step in 0..k {
            let logits = self.draft_logits(&last)?.to_dtype(DType::F32)?;
            mark("draft: logits", &mut since, &dev);
            // The token and the odds the head gives it come back from the card in one read: the
            // odds are the top's share of the softmax, one over the mass relative to the top.
            let top = logits.max_keepdim(1)?;
            let mass = logits.broadcast_sub(&top)?.exp()?.sum_keepdim(1)?;
            let token = logits.argmax_keepdim(1)?.to_dtype(DType::F32)?;
            let read = Tensor::cat(&[&token, &mass], 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let token = read[0] as u32;
            // A row of the verify costs what the policy measured: a token the head gives less
            // than those odds is not worth one, nor are the tokens that would follow it.
            let odds = 1.0 / read[1].max(1.0);
            mark("draft: token and odds", &mut since, &dev);
            if odds < min_odds {
                break;
            }
            out.push(token);
            if step + 1 == k {
                break;
            }
            last = self.draft_rows(&[token], &last, at, false)?;
            mark("draft: a guess's layer", &mut since, &dev);
            at += 1;
        }
        Ok(out)
    }

    /// Cut the head's cache back to `len` positions.
    pub fn draft_trim(&mut self, len: usize) {
        if let Some(d) = self.draft.as_mut() {
            d.layer.trim_kv_to(len);
            if let Some(sparse) = d.layer.sparse.as_mut() {
                // The card keeps its buffers: the recorded guesses read them.
                match &d.indexer_mark {
                    Some((at, cache)) if *at == len => sparse.restore_cache(cache, true),
                    _ if len == 0 => sparse.reset(),
                    _ => {}
                }
                #[cfg(feature = "cuda")]
                sparse.card_cut(len);
            }
            d.len = d.len.min(len);
        }
    }
}

#[cfg(feature = "cuda")]
impl GenericHeteroTransformer {
    /// The card the head's layer runs on, when it runs on one and the head's logits are there.
    fn draft_card(&self) -> Option<Device> {
        let d = self.draft.as_ref()?;
        let dev = d.layer.cos.device().clone();
        let head_there = match (&self.output_proj_cuda_qmm, &self.output_proj_dev) {
            (Some(_), Some(head)) => head.same_device(&dev),
            _ => false,
        };
        (dev.is_cuda() && head_there && d.layer.hyper.is_some() && d.layer.ngram.is_none())
            .then_some(dev)
    }

    /// Ready the buffers of recorded rows at `at`: the embeddings of `tokens` (none for the head
    /// alone), the positions, the layer's tables and its room for the rows. `false` when the
    /// guesses are not recorded.
    fn draft_step_prepare(&mut self, tokens: &[u32], at: usize) -> Result<bool> {
        use crate::inference::engine::decode_step::VERIFY_ROWS;
        let Some(dev) = self.draft_card() else {
            return Ok(false);
        };
        if self
            .draft
            .as_ref()
            .and_then(|d| d.step.as_ref())
            .is_some_and(|s| s.blocked)
        {
            return Ok(false);
        }
        let rows = tokens.len().max(1);
        let cap = self
            .layers
            .first()
            .map(|l| l.kv_cache.max_seq_len_padded())
            .unwrap_or(0);
        if rows > VERIFY_ROWS || at + rows > cap {
            return Ok(false);
        }
        // The workspaces the kernels keep are made before a recording, never inside one.
        self.ensure_graph_workspaces();
        let emb = if tokens.is_empty() {
            None
        } else {
            let edev = self.embeddings.embeddings().device().clone();
            let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &edev)?;
            let mut e = self.embeddings.forward(&ids)?.to_dtype(DType::F32)?;
            if let Some(scale) = self.config.embed_scale {
                e = (e * scale)?;
            }
            Some(e.to_device(&dev)?)
        };
        let d = self.draft.as_mut().expect("checked above");
        let (streams, dim) = (d.streams, d.dim);
        if d.step.is_none() {
            d.step = Some(DraftStep {
                emb: Tensor::zeros_on((1, VERIFY_ROWS, dim), DType::F32, &dev)?,
                rows_in: Tensor::zeros_on((1, VERIFY_ROWS, streams * dim), DType::F32, &dev)?,
                hidden: Tensor::zeros_on((1, 1, streams * dim), DType::F32, &dev)?,
                ids: Tensor::zeros_on(1, DType::U32, &dev)?,
                read: Tensor::zeros_on((1, 1), DType::F32, &dev)?,
                pos: Tensor::zeros_on(VERIFY_ROWS, DType::I32, &dev)?,
                pairs: Vec::new(),
                head: None,
                layer: None,
                key: None,
                failures: 0,
                blocked: false,
            });
        }
        let step = d.step.as_mut().expect("made above");
        if let Some(e) = emb {
            step.emb.slice_set(&e, 1, 0)?;
        }
        let positions: Vec<i32> = (0..VERIFY_ROWS).map(|r| (at + r) as i32).collect();
        step.pos
            .slice_set(&Tensor::new(&positions[..], &dev)?, 0, 0)?;
        d.layer.verify_pos_i32 = Some(step.pos.clone());
        d.layer.verify_keep_ready(&dev)?;
        let (cos, sin) = d.layer.rope_table().rows(at, VERIFY_ROWS, &dev)?;
        match (&d.layer.verify_rope_cos, &d.layer.verify_rope_sin) {
            (Some(c), Some(s)) => {
                c.slice_set(&cos, 0, 0)?;
                s.slice_set(&sin, 0, 0)?;
            }
            _ => {
                d.layer.verify_rope_cos = Some(cos.force_contiguous()?);
                d.layer.verify_rope_sin = Some(sin.force_contiguous()?);
            }
        }
        // The cache holds every position a request can reach, so the recorded kernels keep
        // reading the same buffers; the indexer's room is made the same way.
        if d.layer.kv_cache.max_seq_len_padded() < cap {
            d.layer.kv_cache.reserve(cap)?;
        }
        let births = match d.layer.sparse.as_mut() {
            Some(sparse) => {
                sparse.card_room_for(cap)?;
                sparse.card_hold_rows(at, rows)?;
                sparse.card_births()
            }
            None => 0,
        };
        // Buffers that moved since the recording end it: the graphs are recorded again.
        let keys = d
            .layer
            .kv_cache
            .k_buffer()
            .ok_or_else(|| crate::tensor::Error::msg("the head's cache holds no token"))?
            .id();
        if step.key != Some((keys, births)) {
            step.pairs.clear();
            step.head = None;
            step.layer = None;
            step.key = Some((keys, births));
        }
        Ok(true)
    }

    /// The kernels of a recorded step of `kind`; the token named and the softmax mass are left
    /// in their buffers by the head.
    fn draft_step_forward(&mut self, kind: DraftKind) -> Result<()> {
        let _order = crate::tensor::quantized::DecodeOrder::enter();
        let _in_place = crate::inference::moe_cuda::in_place_state(true);
        let d = self
            .draft
            .as_mut()
            .ok_or_else(|| crate::tensor::Error::msg("this model has no draft head".to_string()))?;
        let (emb, rows_in, hidden, ids, read) = {
            let s = d
                .step
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("the guess was not prepared"))?;
            (
                s.emb.clone(),
                s.rows_in.clone(),
                s.hidden.clone(),
                s.ids.clone(),
                s.read.clone(),
            )
        };
        match kind {
            DraftKind::Pairs(n) => {
                let x = d.fuse(&emb.narrow(1, 0, n)?, &rows_in.narrow(1, 0, n)?)?;
                let out = d.layer.forward_wide_graph_rows(&x, n)?;
                hidden.slice_set(&out.narrow(1, n - 1, 1)?.to_dtype(DType::F32)?, 1, 0)?;
                return Ok(());
            }
            DraftKind::Step => {
                let x = d.fuse(&emb.narrow(1, 0, 1)?, &hidden)?;
                let out = d.layer.forward_wide_graph_rows(&x, 1)?;
                hidden.slice_set(&out.to_dtype(DType::F32)?, 1, 0)?;
            }
            DraftKind::Head => {}
        }
        let xn = d.mixer.normed_on_card(&hidden)?;
        let (mixed, _) = d.mixer.mix_on_card(&xn, false)?;
        let normed = mixed.reshape((1, d.dim))?;
        let logits = match (&self.output_proj_cuda_qmm, &self.output_proj_cuda) {
            (Some(qmm), _) => qmm.forward(&normed)?,
            (None, Some(proj)) => normed.to_dtype(DType::F16)?.matmul_t(proj)?,
            (None, None) => crate::tensor::bail!("a recorded guess needs the card's head"),
        }
        .to_dtype(DType::F32)?;
        let top = logits.max_keepdim(1)?;
        let mass = logits.broadcast_sub(&top)?.exp()?.sum_keepdim(1)?;
        crate::inference::moe_cuda::batched_argmax_into(&logits, &ids)?;
        read.slice_set(&mass, 0, 0)?;
        Ok(())
    }

    /// The token the step named and the head's odds on it.
    fn draft_step_read(&self) -> Result<(u32, f32)> {
        let step = self
            .draft
            .as_ref()
            .and_then(|d| d.step.as_ref())
            .ok_or_else(|| crate::tensor::Error::msg("the guess was not prepared"))?;
        let token = step.ids.to_vec1::<u32>()?[0];
        let mass = step.read.flatten_all()?.to_vec1::<f32>()?[0];
        Ok((token, 1.0 / mass.max(1.0)))
    }

    /// The row the last step wrote, on the host.
    fn draft_step_row(&self) -> Result<Vec<f32>> {
        self.draft
            .as_ref()
            .and_then(|d| d.step.as_ref())
            .ok_or_else(|| crate::tensor::Error::msg("the guess was not prepared"))?
            .hidden
            .flatten_all()?
            .to_vec1::<f32>()
    }

    /// A step of `kind` as a recorded graph: recorded on first use, checked against the eager
    /// run, replayed after. `false` when the graphs are given up on.
    fn draft_step_run(&mut self, kind: DraftKind, at: usize) -> Result<bool> {
        let have = self
            .draft
            .as_ref()
            .and_then(|d| d.step.as_ref())
            .is_some_and(|s| match kind {
                DraftKind::Pairs(n) => s.pairs.iter().any(|(m, _)| *m == n),
                DraftKind::Head => s.head.is_some(),
                DraftKind::Step => s.layer.is_some(),
            });
        if have {
            let step = self
                .draft
                .as_ref()
                .and_then(|d| d.step.as_ref())
                .expect("checked above");
            let g = match kind {
                DraftKind::Pairs(n) => {
                    &step
                        .pairs
                        .iter()
                        .find(|(m, _)| *m == n)
                        .expect("checked above")
                        .1
                }
                DraftKind::Head => step.head.as_ref().expect("checked above"),
                DraftKind::Step => step.layer.as_ref().expect("checked above"),
            };
            if let Err(e) = g.0.launch() {
                let step = self
                    .draft
                    .as_mut()
                    .and_then(|d| d.step.as_mut())
                    .expect("checked above");
                step.failures += 1;
                step.pairs.clear();
                step.head = None;
                step.layer = None;
                step.blocked = step.failures >= DRAFT_RECORD_FAILURES;
                tracing::warn!(
                    "the replay of the draft's {} at pos={at} failed: {e:?}; {}",
                    match kind {
                        DraftKind::Pairs(n) => format!("pairs of {n}"),
                        DraftKind::Head => "head".to_string(),
                        DraftKind::Step => "step".to_string(),
                    },
                    if step.blocked {
                        "the guesses stay eager"
                    } else {
                        "recorded again"
                    }
                );
                return Ok(false);
            }
            return Ok(true);
        }
        let dev = self.draft_card().expect("prepared");
        let stream = dev.as_cuda_device()?.0.stream().clone();
        // The eager run takes the kernels the recording will take, so that what they keep is
        // made before the recording and the two compute alike.
        super::expert_tier::kernels_of_a_recording(true);
        let eager = self.draft_step_forward(kind);
        super::expert_tier::kernels_of_a_recording(false);
        eager?;
        // What the eager run left: the token named, or the row written.
        let want_token = match kind {
            DraftKind::Pairs(_) => None,
            _ => Some(self.draft_step_read()?.0),
        };
        let want_row = match kind {
            DraftKind::Pairs(_) => Some(self.draft_step_row()?),
            _ => None,
        };
        let _ = stream.synchronize();
        let captured = (|| -> Result<Option<crate::tensor::cuda_ext::CapturedGraph>> {
            crate::tensor::cuda_ext::begin_capture(&stream)?;
            self.draft_step_forward(kind)?;
            crate::tensor::cuda_ext::end_capture(&stream)
        })();
        let verdict: std::result::Result<crate::tensor::cuda_ext::CapturedGraph, String> =
            match captured {
                Ok(Some(g)) => {
                    let _ = g.upload();
                    // A graph that reads what it writes back is launched once; the others
                    // twice, as a recording that keeps an allocation of its own cannot be
                    // launched again, and says so here rather than later.
                    let launched = match kind {
                        DraftKind::Step => g.launch(),
                        _ => g.launch().and_then(|()| g.launch()),
                    };
                    match launched {
                        Ok(()) => {
                            let agrees = match (want_token, want_row) {
                                (Some(t), _) => self.draft_step_read().map(|got| got.0 == t),
                                (None, Some(row)) => self.draft_step_row().map(|got| {
                                    crate::inference::engine::decode_step::replay_agrees(&got, &row)
                                }),
                                _ => Ok(false),
                            };
                            match agrees {
                                Ok(true) => Ok(g),
                                Ok(false) => {
                                    Err("the replay does not agree with the eager run".into())
                                }
                                Err(e) => Err(format!("{e}")),
                            }
                        }
                        Err(e) => {
                            let (mut alloc, mut free) = (0usize, 0usize);
                            if let Ok(nodes) = g.nodes(1 << 16) {
                                if let Ok(types) = g.node_types(&nodes) {
                                    for t in types {
                                        match t {
                                            cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => alloc += 1,
                                            cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => free += 1,
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            Err(format!(
                                "launch: {e:?} ({alloc} allocations, {free} frees in the graph)"
                            ))
                        }
                    }
                }
                Ok(None) => Err("the capture recorded nothing".into()),
                Err(e) => {
                    let _ = crate::tensor::cuda_ext::end_capture(&stream);
                    let _ = stream.synchronize();
                    Err(format!("capture: {e}"))
                }
            };
        let step = self
            .draft
            .as_mut()
            .and_then(|d| d.step.as_mut())
            .expect("prepared");
        match verdict {
            Ok(g) => {
                let (mut alloc, mut free) = (0usize, 0usize);
                if let Ok(nodes) = g.nodes(1 << 16) {
                    if let Ok(types) = g.node_types(&nodes) {
                        for t in types {
                            match t {
                                cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => alloc += 1,
                                cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => free += 1,
                                _ => {}
                            }
                        }
                    }
                }
                tracing::info!(
                    "draft {} recorded at pos={at}: {} nodes, {alloc} allocations, {free} frees",
                    match kind {
                        DraftKind::Pairs(n) => format!("pairs of {n}"),
                        DraftKind::Head => "head".to_string(),
                        DraftKind::Step => "step".to_string(),
                    },
                    g.num_nodes().unwrap_or(0)
                );
                match kind {
                    DraftKind::Pairs(n) => {
                        step.pairs.retain(|(m, _)| *m != n);
                        if step.pairs.len() >= DRAFT_PAIRS_GRAPHS {
                            step.pairs.remove(0);
                        }
                        step.pairs.push((n, HeldGraph(g)));
                    }
                    DraftKind::Head => step.head = Some(HeldGraph(g)),
                    DraftKind::Step => step.layer = Some(HeldGraph(g)),
                }
                Ok(true)
            }
            Err(why) => {
                tracing::warn!("draft guesses stay eager: {why}");
                step.blocked = true;
                step.pairs.clear();
                step.head = None;
                step.layer = None;
                Ok(false)
            }
        }
    }

    /// The recorded steps wrote the layer's rows up to `len`: the caches say so.
    fn draft_step_settle(&mut self, len: usize) {
        if let Some(d) = self.draft.as_mut() {
            d.layer.kv_cache.sync_len(len);
            if let Some(sparse) = d.layer.sparse.as_mut() {
                sparse.card_confirm(len);
            }
            d.len = d.len.max(len);
        }
    }

    /// Up to `k` tokens guessed as recorded steps: the pairs `(tokens[i], feature[i])` at
    /// `committed` onward through the layer, then the guesses. `None` when the guesses are not
    /// recorded; the head's bookkeeping is then the caller's.
    fn draft_tokens_recorded(
        &mut self,
        tokens: &[u32],
        feature: &Tensor,
        committed: usize,
        k: usize,
        min_odds: f32,
    ) -> Result<Option<Vec<u32>>> {
        let n = tokens.len();
        if !self.draft_step_prepare(tokens, committed)? {
            return Ok(None);
        }
        {
            let dev = self.draft_card().expect("prepared");
            let wide = feature.elem_count() / n.max(1);
            let rows = feature
                .to_dtype(DType::F32)?
                .reshape((1, n, wide))?
                .to_device(&dev)?;
            let step = self
                .draft
                .as_mut()
                .and_then(|d| d.step.as_mut())
                .expect("prepared");
            step.rows_in.slice_set(&rows, 1, 0)?;
        }
        if !self.draft_step_run(DraftKind::Pairs(n), committed)? {
            return Ok(None);
        }
        let mut at = committed + n;
        self.draft_step_settle(at);
        if let Some(d) = self.draft.as_mut() {
            d.committed = at;
            d.mark_indexer();
        }
        let mut out = Vec::with_capacity(k);
        let mut token: Option<u32> = None;
        for _ in 0..k {
            let kind = match token {
                None => DraftKind::Head,
                Some(t) => {
                    if !self.draft_step_prepare(&[t], at)? {
                        break;
                    }
                    DraftKind::Step
                }
            };
            if !self.draft_step_run(kind, at)? {
                break;
            }
            if kind == DraftKind::Step {
                at += 1;
                self.draft_step_settle(at);
            }
            let (t, odds) = self.draft_step_read()?;
            // A row of the verify costs what the policy measured: a token the head gives less
            // than those odds is not worth one, nor are the tokens that would follow it.
            if odds < min_odds {
                break;
            }
            out.push(t);
            token = Some(t);
        }
        self.draft_step_settle(at);
        Ok(Some(out))
    }
}
