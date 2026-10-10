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

/// The odds the head must give a token for it to be drafted.
const DRAFT_MIN_ODDS: f32 = 0.5;

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
    pub fn draft_tokens(&mut self, tokens: &[u32], k: usize) -> Result<Vec<u32>> {
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
            let logits = self.draft_logits(&last)?;
            mark("draft: logits", &mut since, &dev);
            let token = logits.argmax(1)?.to_vec1::<u32>()?[0];
            // A row of the verify costs a step's worth of the experts it reads: a token the head
            // is not at least even on is not worth one, nor are the tokens that would follow it.
            let odds = crate::tensor::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?
                .max_keepdim(1)?
                .flatten_all()?
                .to_vec1::<f32>()?[0];
            mark("draft: token and odds", &mut since, &dev);
            if odds < DRAFT_MIN_ODDS {
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
                match &d.indexer_mark {
                    Some((at, cache)) if *at == len => sparse.restore_cache(cache, false),
                    _ if len == 0 => sparse.reset(),
                    _ => {}
                }
            }
            d.len = d.len.min(len);
        }
    }
}
