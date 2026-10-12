//! Cutting a recurrent model back to the rows a draft verify kept.
//!
//! A verify runs the drafted rows through every layer as one forward. The KV caches take the
//! rows and are cut back by `trim_kv`; a recurrent mixer and an indexer have advanced past all of
//! them and cannot be cut. So a verify that is asked to record keeps, per layer, the state before
//! the rows and the rows themselves. When the engine cuts the model back to a shorter length,
//! each layer returns to the state it had and takes in only the rows that were kept; when it
//! keeps all of them, the record is dropped with the next forward.

use super::*;

/// What one layer holds from a verify, until the rows kept are known.
pub(super) struct LayerPending {
    /// Position of the first row.
    base: usize,
    /// Rows the verify carried.
    rows: usize,
    /// The input of the token mixer, `[1, rows, hidden]`, once the layer has made it.
    input: Option<Tensor>,
    /// What the n-gram block carried before the rows, with the residual and the table rows it
    /// was given.
    ple: Option<(ngram::PleState, Tensor, Tensor)>,
    /// The recurrent mixer's state before the rows.
    mixer: Option<mixer::MixerState>,
    /// The indexer's cache before the rows.
    indexer: Option<qsa::IndexerCache>,
}

/// What the model holds from a verify: where it started and the tokens it carried.
pub(super) struct Window {
    pub(super) base: usize,
    pub(super) rows: usize,
    /// The n-gram history before the rows: the block's own copy is taken after the model gathered
    /// the rows' table entries, so it already holds them.
    pub(super) ngram: Option<ngram::NgramHistory>,
    pub(super) ids: Vec<u32>,
}

impl GenericTransformerLayer {
    /// Record the state this layer is in before it takes `r`, the residual rows of a verify that
    /// starts at `index_pos`, with `emb` the n-gram rows of the layer that takes them.
    pub(super) fn spec_begin(
        &mut self,
        r: &Tensor,
        emb: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<()> {
        let rows = r.dims3()?.1;
        let mixer = match self.mixer {
            mixer::Mixer::Attention | mixer::Mixer::Absent => None,
            _ => match self.mixer.checkpoint()? {
                Some(state) => Some(state),
                None => crate::tensor::bail!("a recurrent layer with no state to return to"),
            },
        };
        let indexer = self
            .sparse
            .as_ref()
            .map(|s| s.snapshot_cache())
            .transpose()?;
        let ple = match (self.ngram.as_ref(), emb) {
            (Some(ple), Some(emb)) => Some((ple.snapshot()?, r.clone(), emb.clone())),
            _ => None,
        };
        self.spec = Some(LayerPending {
            base: index_pos,
            rows,
            input: None,
            ple,
            mixer,
            indexer,
        });
        Ok(())
    }

    /// The token mixer's input, once the layer has made it.
    pub(super) fn spec_input(&mut self, mixed: &Tensor) {
        if let Some(p) = self.spec.as_mut() {
            p.input = Some(mixed.clone());
        }
    }

    /// Return to the state before the verify's rows and take in the first `kept` of them.
    pub(super) fn spec_settle(&mut self, kept: usize) -> Result<()> {
        let Some(p) = self.spec.take() else {
            return Ok(());
        };
        if kept >= p.rows {
            return Ok(());
        }
        if let (Some((state, r, emb)), Some(ple)) = (&p.ple, self.ngram.as_mut()) {
            ple.restore(state)?;
            ple.forward(&r.narrow(1, 0, kept)?, &emb.narrow(1, 0, kept)?, p.base)?;
        }
        let Some(input) = &p.input else {
            return Ok(());
        };
        let prefix = input.narrow(1, 0, kept)?;
        if let Some(state) = &p.mixer {
            // Where the state lies on a card, and advanced where it lies, the rows as the
            // decoded tokens they stand for: a recorded step reads it there. A state on the
            // host has no buffers to keep.
            if self.mixer.restore_in_place(state).is_err() {
                self.mixer.restore(state)?;
            }
            let _order = crate::tensor::quantized::DecodeOrder::enter();
            #[cfg(feature = "cuda")]
            let _in_place = crate::inference::moe_cuda::in_place_state(true);
            self.mixer.forward(&prefix, p.base)?;
        }
        if let (Some(cache), Some(sparse)) = (&p.indexer, self.sparse.as_mut()) {
            // The cache is of the sequence the card holds: the card keeps what it covers.
            sparse.restore_cache(cache, true);
            sparse.observe(&prefix, p.base)?;
        }
        Ok(())
    }
}

impl GenericHeteroTransformer {
    /// Start recording: the layers keep their state before the rows of the verify about to run.
    pub(super) fn spec_open(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<()> {
        self.spec_clear();
        let ids = input_ids
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        let ngram = self
            .ngram
            .as_ref()
            .and_then(|ng| self.layers[ng.meta.layer].ngram.as_ref())
            .map(|ple| ple.history.clone());
        self.spec_window = Some(Window {
            base: index_pos,
            rows: ids.len(),
            ngram,
            ids,
        });
        self.spec_recording = true;
        for l in &mut self.layers {
            l.spec_record = true;
        }
        Ok(())
    }

    /// Stop recording once the verify's forward has run.
    pub(super) fn spec_close(&mut self, ran: bool) {
        self.spec_recording = false;
        for l in &mut self.layers {
            l.spec_record = false;
        }
        if !ran {
            self.spec_clear();
        }
    }

    /// Forget what a verify recorded: its rows were all kept, or nothing continues from them.
    pub(super) fn spec_clear(&mut self) {
        if self.spec_recording {
            return;
        }
        self.spec_window = None;
        for l in &mut self.layers {
            l.spec = None;
        }
    }

    /// The model is cut back to `new_len` tokens: when that is inside the rows of the verify just
    /// recorded, every layer and the n-gram history return to the state they had and take in
    /// only the rows that stay.
    pub(super) fn spec_settle(&mut self, new_len: usize) {
        let Some(w) = self.spec_window.take() else {
            return;
        };
        if new_len <= w.base || new_len >= w.base + w.rows {
            for l in &mut self.layers {
                l.spec = None;
            }
            return;
        }
        let kept = new_len - w.base;
        for l in &mut self.layers {
            if let Err(e) = l.spec_settle(kept) {
                tracing::warn!("a layer did not return to the rows kept: {e}");
            }
        }
        // The layers' blocks are back at the history before the rows: it moves past the kept ones.
        if let (Some(pre), Some(ng)) = (w.ngram, self.ngram.as_ref()) {
            if let Some(ple) = self.layers[ng.meta.layer].ngram.as_mut() {
                ple.history = pre;
                ple.history.rows(&ng.meta, &w.ids[..kept]);
            }
        }
    }
}
