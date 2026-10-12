//! A draft verify of a few rows run as the kernels of the recorded decode step, so that a graph
//! can record it: one graph per window size, replayed with the rows' embeddings, positions and
//! tables refreshed before each replay, and the layers returned to the rows that were kept once
//! the engine has read the logits.
//!
//! What a replay reads lives in buffers of its own: the rows' embeddings, each card's row
//! positions, each layer's rotation tables, the n-gram rows, and what a layer keeps of its state
//! before the rows (its recurrent state, its convolution's rows, its token mixer's input), so
//! that it can take in the kept rows again from there. The logits of each window size land in
//! a buffer of their own.

use super::*;
use crate::inference::engine::decode_step::VERIFY_ROWS;
use std::collections::HashMap;

/// What a layer keeps of its state before the rows of a verify, in buffers a replay writes
/// again.
pub(super) struct VerifyKeep {
    /// The recurrent mixer's state, in buffers of the state's shape.
    mixer: Option<mixer::MixerState>,
    /// The n-gram convolution's rows.
    ngram: Option<Tensor>,
    /// The residual the layer took, `[1, rows, wide]`, for the n-gram block to read again.
    input: Option<Tensor>,
    /// The token mixer's input, `[1, rows, hidden]`.
    mixed: Option<Tensor>,
    /// Whether the buffers hold the state before the verify in flight: a layer the verify
    /// never reached keeps nothing to return to.
    saved: bool,
}

/// The rows a verify carries, from where, and the n-gram history before them.
pub(crate) struct VerifyWindow {
    pub(super) base: usize,
    pub(super) rows: usize,
    ids: Vec<u32>,
    ngram: Option<ngram::NgramHistory>,
}

/// The buffers of a recorded verify that belong to the model.
#[derive(Default)]
pub(crate) struct VerifyBuffers {
    /// The rows' embeddings, `[1, VERIFY_ROWS, wide]`.
    hidden: Option<Tensor>,
    /// The last layer's rows, `[1, VERIFY_ROWS, wide]`: what the draft head reads.
    out: Option<Tensor>,
    /// The logits of each window size, `[1, rows, vocab]`.
    logits: HashMap<usize, Tensor>,
    /// The rows' positions on each card, `[VERIFY_ROWS]` i32.
    pos_by_card: HashMap<usize, Tensor>,
    /// The rows' hops between consecutive cards, one per boundary.
    hops: Vec<super::graph::GraphHop>,
    /// The verify in flight, from its preparation to its settlement.
    pub(super) window: Option<VerifyWindow>,
}

/// One op of a recorded verify: the layers of a card's segment, or the rows' hop from a
/// card to the next.
#[derive(Clone, Copy)]
enum VerifyOp {
    Run(usize),
    Hop(usize),
}

#[cfg(feature = "cuda")]
impl GenericTransformerLayer {
    /// The buffers this layer keeps its state in before the rows of a verify, made once.
    pub(super) fn verify_keep_ready(&mut self, dev: &Device) -> Result<()> {
        if self.verify_keep.is_some() {
            return Ok(());
        }
        let hc = self
            .hyper
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("a recorded verify runs a wide layer"))?;
        let (s, d) = (hc.attn.streams(), hc.attn.dim());
        let mixed_dtype = super::wide::sublayer_dtype_of(dev);
        let mixer = self.mixer.checkpoint()?;
        let ngram = self
            .ngram
            .as_ref()
            .map(|ple| ple.rows_buffer(dev))
            .transpose()?;
        let input = match self.ngram.is_some() {
            true => Some(Tensor::zeros_on((1, VERIFY_ROWS, s * d), DType::F32, dev)?),
            false => None,
        };
        let mixed = match self.mixer.is_recurrent() {
            true => Some(Tensor::zeros_on((1, VERIFY_ROWS, d), mixed_dtype, dev)?),
            false => None,
        };
        self.verify_keep = Some(VerifyKeep {
            mixer,
            ngram,
            input,
            mixed,
            saved: false,
        });
        Ok(())
    }

    /// The rows of a recorded verify through this layer: the stages of the recorded decode
    /// step over `rows` rows, the state before them kept first.
    pub(super) fn forward_wide_graph_rows(&mut self, x: &Tensor, rows: usize) -> Result<Tensor> {
        use super::hyper::combine_on_card;
        use super::wide::staged;
        let dev = x.device();
        let x = x.to_dtype(DType::F32)?;
        {
            let keep = self
                .verify_keep
                .as_mut()
                .ok_or_else(|| crate::tensor::Error::msg("the verify's rows were not prepared"))?;
            if let (Some(ple), Some(conv)) = (self.ngram.as_ref(), keep.ngram.as_ref()) {
                ple.save_rows_into(conv)?;
            }
            if let Some(input) = keep.input.as_ref() {
                input.slice_set(&x, 1, 0)?;
            }
            if let Some(state) = keep.mixer.as_ref() {
                self.mixer.save_in_place(state)?;
            }
            keep.saved = true;
        }
        let joined;
        let x = match (self.ngram.as_mut(), self.verify_ngram_rows.as_ref()) {
            (Some(ple), Some(table_rows)) => {
                let add = ple.forward(&x, &table_rows.narrow(1, 0, rows)?, 1)?;
                joined = (&x + add)?;
                &joined
            }
            (Some(_), None) => {
                crate::tensor::bail!("the n-gram rows of the verify are not in place")
            }
            _ => &x,
        };
        let (s, d) = {
            let h = self.hyper.as_ref().expect("a wide layer has its mixers");
            (h.attn.streams(), h.attn.dim())
        };
        let (mixed, inj) = staged("mix_token", &dev, false, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            let xn = hc.attn.normed_on_card(x)?;
            hc.attn.mix_on_card(&xn, true)
        })?;
        if let Some(buffer) = self.verify_keep.as_ref().and_then(|k| k.mixed.as_ref()) {
            buffer.slice_set(&mixed.to_dtype(buffer.dtype())?, 1, 0)?;
        }
        let out = staged("token mixer", &dev, false, || match self.mixer {
            mixer::Mixer::Attention => {
                let pos = self.verify_pos_i32.clone().ok_or_else(|| {
                    crate::tensor::Error::msg("the verify's positions are not on the card")
                })?;
                let mut o = match self.attend_rows_recorded(&mixed, &pos, rows)? {
                    Some(o) => o,
                    None => {
                        let mut parts = Vec::with_capacity(rows);
                        for r in 0..rows {
                            let row = mixed.narrow(1, r, 1)?.contiguous()?;
                            parts.push(self.attend_row_recorded(&row, &pos.narrow(0, r, 1)?, r)?);
                        }
                        Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?
                    }
                };
                if let Some(b) = &self.attn_output_bias {
                    o = o.broadcast_add(b)?;
                }
                Ok(o)
            }
            _ => {
                let m = self.mixer.forward(&mixed, 1)?;
                self.attn_output.forward(&m)
            }
        })?;
        let (r, xn) = staged("combine", &dev, false, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            combine_on_card(
                x,
                &out,
                &inj.expect("the token mixer's mixer writes back"),
                Some(&hc.ffn),
                s,
                d,
            )
        })?;
        let (mixed, inj) = staged("mix_ffn", &dev, false, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            hc.ffn
                .mix_on_card(&xn.expect("the norm was asked for"), true)
        })?;
        let out = self.feed_forward(&mixed, &dev)?;
        let (r, _) = staged("combine", &dev, false, || {
            combine_on_card(
                &r,
                &out,
                &inj.expect("the feed-forward's mixer writes back"),
                None,
                s,
                d,
            )
        })?;
        Ok(r)
    }

    /// Return to the state before the verify's `rows` rows at `base`, and take in the first
    /// `kept` of them again.
    fn verify_return(&mut self, base: usize, rows: usize, kept: usize) -> Result<()> {
        let Some(keep) = self.verify_keep.as_mut() else {
            return Ok(());
        };
        let saved = std::mem::replace(&mut keep.saved, false);
        if kept >= rows || !saved {
            if let Some(sparse) = self.sparse.as_mut() {
                sparse.card_cut(base + kept.min(rows));
            }
            return Ok(());
        }
        let keep = self.verify_keep.as_ref().expect("checked above");
        if let Some(state) = keep.mixer.as_ref() {
            self.mixer.restore_in_place(state)?;
        }
        if let (Some(ple), Some(conv)) = (self.ngram.as_mut(), keep.ngram.as_ref()) {
            ple.restore_rows_from(conv)?;
            if let (Some(input), Some(table_rows), true) = (
                keep.input.as_ref(),
                self.verify_ngram_rows.as_ref(),
                kept > 0,
            ) {
                ple.forward(
                    &input.narrow(1, 0, kept)?,
                    &table_rows.narrow(1, 0, kept)?,
                    base,
                )?;
            }
        }
        if let (Some(mixed), true, true) =
            (keep.mixed.as_ref(), self.mixer.is_recurrent(), kept > 0)
        {
            self.mixer.forward(&mixed.narrow(1, 0, kept)?, base)?;
        }
        if let Some(sparse) = self.sparse.as_mut() {
            sparse.card_cut(base + kept);
        }
        Ok(())
    }

    /// Forget the buffers of a recorded verify.
    pub(super) fn clear_verify_state(&mut self) {
        self.verify_pos_i32 = None;
        self.verify_rope_cos = None;
        self.verify_rope_sin = None;
        self.verify_ngram_rows = None;
        self.verify_keep = None;
    }
}

#[cfg(feature = "cuda")]
impl GenericHeteroTransformer {
    /// Why this model records no verify, `None` when it does: every layer has to be recorded,
    /// on one card, with a wide residual and the head on the card.
    pub fn verify_window_block(&self) -> Option<String> {
        if self.config.hyper.is_none() {
            return Some("the residual is not wide".into());
        }
        if self.graph_segments().is_empty() {
            return Some("the layers are not all on cards".into());
        }
        if self.output_hyper.is_none()
            || (self.output_proj_cuda_qmm.is_none() && self.output_proj_cuda.is_none())
        {
            return Some("the head is not on the card".into());
        }
        (0..self.layers.len()).find_map(|i| {
            self.layer_eager_reason(i)
                .map(|why| format!("layer {i}: {why}"))
        })
    }

    /// Rows a recorded verify can carry through this model, `0` when it cannot run one (see
    /// [`Self::verify_window_block`]); every cache of experts has to serve the rows too.
    pub fn verify_window_rows(&self) -> usize {
        if self.verify_window_block().is_some() {
            return 0;
        }
        let mut rows = VERIFY_ROWS;
        for l in &self.layers {
            if let Some(m) = l.moe.as_ref() {
                if let Some(t) = m.tier.as_ref().filter(|t| !t.is_resident()) {
                    rows = rows.min(t.graph_rows());
                }
            }
        }
        rows
    }

    /// Ready the buffers of a verify of `ids` at `pos`: the rows' embeddings, positions and
    /// tables, the n-gram rows, and each layer's room to keep its state. The n-gram history
    /// moves past the rows; [`Self::verify_window_settle`] brings it back to the rows kept.
    pub fn verify_window_prepare(&mut self, pos: usize, ids: &[u32]) -> Result<()> {
        let rows = ids.len();
        let most = self.verify_window_rows();
        if rows == 0 || rows > most {
            crate::tensor::bail!("a recorded verify of {rows} rows, of at most {most}");
        }
        self.spec_clear();
        self.graph_pos = pos;
        self.ensure_graph_workspaces();
        let ngram_before = self
            .ngram
            .as_ref()
            .and_then(|ng| self.layers[ng.meta.layer].ngram.as_ref())
            .map(|ple| ple.history.clone());
        self.verify.window = Some(VerifyWindow {
            base: pos,
            rows,
            ids: ids.to_vec(),
            ngram: ngram_before,
        });
        let card = self
            .layers
            .first()
            .map(|l| l.cos.device().clone())
            .filter(|d| d.is_cuda())
            .ok_or_else(|| crate::tensor::Error::msg("a recorded verify runs on a card"))?;
        // The rows' positions, once per card.
        let positions: Vec<i32> = (0..VERIFY_ROWS).map(|r| (pos + r) as i32).collect();
        if self.graph_state_groups.is_none() {
            self.graph_state_groups = Some(self.compute_graph_state_groups());
        }
        let rope_group = self
            .graph_state_groups
            .as_ref()
            .expect("made above")
            .rope
            .clone();
        let mut filled: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for i in 0..self.layers.len() {
            let dev = self.layers[i].cos.device().clone();
            self.layers[i].verify_keep_ready(&dev)?;
            let crate::tensor::DeviceLocation::Cuda { gpu_id } = dev.location() else {
                crate::tensor::bail!("a recorded verify runs on a card");
            };
            if self.layers[i].sparse.is_some() {
                let shared = match self.verify.pos_by_card.get(&gpu_id) {
                    Some(t) => {
                        t.slice_set(&Tensor::new(&positions[..], &dev)?, 0, 0)?;
                        t.clone()
                    }
                    None => {
                        let t = Tensor::new(&positions[..], &dev)?.force_contiguous()?;
                        self.verify.pos_by_card.insert(gpu_id, t.clone());
                        t
                    }
                };
                self.layers[i].verify_pos_i32 = Some(shared);
                // The tables of the rows, written once by the rope leader of the layer's group,
                // which may be a layer that does not attend.
                let leader = rope_group[i];
                if filled.insert(leader) {
                    let layer = &mut self.layers[leader];
                    let (cos, sin) = layer.rope_table().rows(pos, rows, &dev)?;
                    match (&layer.verify_rope_cos, &layer.verify_rope_sin) {
                        (Some(c), Some(s)) => {
                            c.slice_set(&cos, 0, 0)?;
                            s.slice_set(&sin, 0, 0)?;
                        }
                        _ => {
                            let (c, s) = layer.rope_table().rows(0, VERIFY_ROWS, &dev)?;
                            let (c, s) = (c.force_contiguous()?, s.force_contiguous()?);
                            c.slice_set(&cos, 0, 0)?;
                            s.slice_set(&sin, 0, 0)?;
                            layer.verify_rope_cos = Some(c);
                            layer.verify_rope_sin = Some(s);
                        }
                    }
                }
                // The cache has room for the rows and keeps the buffers a recorded window
                // reads; a cache that outgrew them ends the recorded windows.
                let layer = &mut self.layers[i];
                let cap = layer.kv_cache.max_seq_len_padded();
                match layer.graph_kv_cap {
                    Some(recorded) if recorded != cap || pos + rows > recorded => {
                        crate::tensor::bail!(
                            "graph_state: rows {pos}..{} outgrew the cache a recorded step reads, {recorded} tokens (layer {i})",
                            pos + rows
                        );
                    }
                    Some(_) => {}
                    None => layer.kv_cache.reserve(pos + rows)?,
                }
                let room = layer.kv_cache.max_seq_len_padded();
                if let Some(sparse) = layer.sparse.as_mut() {
                    sparse.card_room_for(room)?;
                    sparse.card_hold_rows(pos, rows)?;
                }
            }
        }
        for i in 0..self.layers.len() {
            if self.layers[i].sparse.is_some() {
                let leader = rope_group[i];
                let (c, s) = (
                    self.layers[leader].verify_rope_cos.clone(),
                    self.layers[leader].verify_rope_sin.clone(),
                );
                self.layers[i].verify_rope_cos = c;
                self.layers[i].verify_rope_sin = s;
            }
        }
        // The n-gram rows of the tokens, and the rows' embeddings.
        let ids_t = Tensor::from_vec(ids.to_vec(), (1, rows), &Device::Cpu)?;
        if let Some(layer) = self.ngram.as_ref().map(|n| n.meta.layer) {
            self.gather_ngram_rows(&ids_t, pos)?;
            let pending = self
                .ngram
                .as_ref()
                .and_then(|n| n.pending.clone())
                .ok_or_else(|| crate::tensor::Error::msg("the n-gram rows were not gathered"))?;
            let held = &mut self.layers[layer];
            let dev = held.cos.device().clone();
            let pending = pending.to_device(&dev)?;
            match held.verify_ngram_rows.as_ref() {
                Some(buffer) => buffer.slice_set(&pending, 1, 0)?,
                None => {
                    let width = pending.dim(2)?;
                    let buffer = Tensor::zeros_on((1, VERIFY_ROWS, width), DType::F32, &dev)?;
                    buffer.slice_set(&pending, 1, 0)?;
                    held.verify_ngram_rows = Some(buffer);
                }
            }
        }
        let emb_device = self.embeddings.embeddings().device().clone();
        let mut hidden = self.embeddings.forward(&ids_t.to_device(&emb_device)?)?;
        if let Some(scale) = self.config.embed_scale {
            hidden = (hidden * scale)?;
        }
        let streams = self.config.hyper.expect("checked above").streams;
        let hidden = super::wide::widen_embedding(&hidden, streams)?.to_device(&card)?;
        let wide = hidden.dim(2)?;
        let segs = self.graph_segments();
        match self.verify.hidden.as_ref() {
            Some(buffer) => buffer.slice_set(&hidden, 1, 0)?,
            None => {
                let buffer = Tensor::zeros_on((1, VERIFY_ROWS, wide), DType::F32, &card)?;
                buffer.slice_set(&hidden, 1, 0)?;
                self.verify.hidden = Some(buffer);
                // The last layer's rows lie on the last card, where the head reads them.
                let last = segs.last().map_or(card.clone(), |(_, d)| d.clone());
                self.verify.out =
                    Some(Tensor::zeros_on((1, VERIFY_ROWS, wide), DType::F32, &last)?);
            }
        }
        if self.verify.hops.len() + 1 != segs.len() {
            self.verify.hops = segs
                .iter()
                .skip(1)
                .map(|(_, d)| super::graph::GraphHop::with_rows(wide, VERIFY_ROWS, d))
                .collect::<Result<_>>()?;
        }
        Ok(())
    }

    /// The ops of a recorded verify, in order: each card's segment of layers, with the rows'
    /// hop before every segment but the first; the head ends the last segment.
    fn verify_plan(&self) -> Vec<VerifyOp> {
        let segs = self.graph_segments().len();
        (0..segs)
            .flat_map(|seg| {
                (seg > 0)
                    .then_some(VerifyOp::Hop(seg - 1))
                    .into_iter()
                    .chain(std::iter::once(VerifyOp::Run(seg)))
            })
            .collect()
    }

    /// For each op of a recorded verify: whether it is recorded in a graph, and the stream
    /// of the card it runs on.
    pub fn verify_step_ops(
        &self,
    ) -> Vec<(bool, std::sync::Arc<crate::tensor::cuda_ext::CudaStream>)> {
        let streams = self.graph_segment_streams();
        self.verify_plan()
            .iter()
            .filter_map(|op| match *op {
                VerifyOp::Run(seg) => streams.get(seg).map(|s| (true, s.clone())),
                VerifyOp::Hop(at) => streams.get(at + 1).map(|s| (false, s.clone())),
            })
            .collect()
    }

    /// Op `op` of the prepared verify over its `rows` rows: a segment runs its layers as the
    /// kernels of the recorded decode step and the last one the head, whose op returns the
    /// logits `[1, rows, vocab]` in the buffer of this window size; a hop carries the rows to
    /// the next card.
    pub fn run_verify_op(&mut self, op: usize, rows: usize) -> Result<Option<Tensor>> {
        let Some(this) = self.verify_plan().into_iter().nth(op) else {
            crate::tensor::bail!("run_verify_op: no op {op}");
        };
        let segs = self.graph_segments();
        match this {
            VerifyOp::Hop(at) => {
                // The next card's stream waits for the copy to the host, then fills its own
                // buffer from it.
                let streams = self.graph_segment_streams();
                if let (Some(a), Some(b)) = (streams.get(at), streams.get(at + 1)) {
                    let ev = crate::tensor::cuda_ext::record_event(a)?;
                    crate::tensor::cuda_ext::wait_event(b, &ev)?;
                }
                self.verify
                    .hops
                    .get(at)
                    .ok_or_else(|| crate::tensor::Error::msg("the verify's hop is not prepared"))?
                    .receive()?;
                Ok(None)
            }
            VerifyOp::Run(seg) => {
                let input = match seg {
                    0 => self
                        .verify
                        .hidden
                        .as_ref()
                        .ok_or_else(|| {
                            crate::tensor::Error::msg("the verify's rows were not prepared")
                        })?
                        .narrow(1, 0, rows)?,
                    _ => self
                        .verify
                        .hops
                        .get(seg - 1)
                        .ok_or_else(|| {
                            crate::tensor::Error::msg("the verify's hop is not prepared")
                        })?
                        .landing(rows)?,
                };
                // The rows are the decode steps they stand for: the mat-vecs take them one by
                // one and the recurrent states advance where they lie.
                let _order = crate::tensor::quantized::DecodeOrder::enter();
                let _in_place = crate::inference::moe_cuda::in_place_state(true);
                let mut h = input;
                for li in segs[seg].0.clone() {
                    h = self.layers[li].forward_wide_graph_rows(&h, rows)?;
                }
                if seg + 1 == segs.len() {
                    return self.verify_head(h, rows).map(Some);
                }
                self.verify
                    .hops
                    .get(seg)
                    .ok_or_else(|| crate::tensor::Error::msg("the verify's hop is not prepared"))?
                    .send(&h)?;
                Ok(None)
            }
        }
    }

    /// The verify's `rows` rows through every layer and the head, as the kernels of the recorded
    /// decode step: what a capture records and a replay runs. The logits, `[1, rows, vocab]`,
    /// are in the buffer of this window size.
    pub fn forward_verify_window(&mut self, rows: usize) -> Result<Tensor> {
        let mut logits = None;
        for op in 0..self.verify_plan().len() {
            if let Some(l) = self.run_verify_op(op, rows)? {
                logits = Some(l);
            }
        }
        logits.ok_or_else(|| crate::tensor::Error::msg("the verify's head did not run"))
    }

    /// The head over the last layer's rows `h`: the logits `[1, rows, vocab]` in the buffer
    /// of this window size.
    fn verify_head(&mut self, h: Tensor, rows: usize) -> Result<Tensor> {
        if let Some(out) = self.verify.out.as_ref() {
            out.slice_set(&h.to_dtype(DType::F32)?, 1, 0)?;
        }
        let oh = self
            .output_hyper
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("a wide residual's head runs on a card"))?;
        let xn = oh.normed_on_card(&h)?;
        let (mixed, _) = oh.mix_on_card(&xn, false)?;
        let normed = mixed.reshape((rows, self.config.embedding_length))?;
        // The quantised head where the card has one, else the float head read in place.
        let mut logits = match (&self.output_proj_cuda_qmm, &self.output_proj_cuda) {
            (Some(qmm), _) => qmm.forward(&normed)?,
            (None, Some(proj)) => normed.to_dtype(DType::F16)?.matmul_t(proj)?,
            (None, None) => crate::tensor::bail!("a recorded verify needs the card's head"),
        };
        logits = self.apply_output_bias(logits)?;
        logits = logits.to_dtype(DType::F32)?;
        if let Some(scale) = self.config.logit_scale {
            logits = (logits / scale)?;
        }
        let logits = match self.config.final_logit_softcapping {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }?;
        let vocab = logits.dim(1)?;
        let logits = logits.reshape((1, rows, vocab))?;
        match self.verify.logits.get(&rows) {
            Some(buffer) => buffer.slice_set(&logits, 0, 0)?,
            None => {
                self.verify.logits.insert(rows, logits.force_contiguous()?);
            }
        }
        Ok(self.verify.logits[&rows].clone())
    }

    /// The verify prepared last has run: the caches hold its rows, of which the first `kept`
    /// stay. Every layer returns to the rows kept, and the draft head is handed the rows.
    pub fn verify_window_settle(&mut self, kept: usize) -> Result<()> {
        let Some(w) = self.verify.window.take() else {
            return Ok(());
        };
        let (base, rows) = (w.base, w.rows);
        self.sync_kv_len_for_graph(base + rows);
        if self.capture_feature {
            self.last_feature = self
                .verify
                .out
                .as_ref()
                .map(|o| o.narrow(1, 0, rows))
                .transpose()?;
        }
        if kept < rows {
            let _order = crate::tensor::quantized::DecodeOrder::enter();
            let _in_place = crate::inference::moe_cuda::in_place_state(true);
            for l in &mut self.layers {
                l.verify_return(base, rows, kept)?;
                l.trim_kv_to(base + kept);
            }
            if let (Some(pre), Some(ng)) = (w.ngram, self.ngram.as_ref()) {
                if let Some(ple) = self.layers[ng.meta.layer].ngram.as_mut() {
                    ple.history = pre;
                    ple.history.rows(&ng.meta, &w.ids[..kept]);
                }
            }
        }
        Ok(())
    }

    /// The logits of the plain step of `token` at `pos`, run eagerly, the model returned to the
    /// state before it: the recurrent states from a checkpoint, the n-gram history as it was;
    /// the cache row the step wrote is written again by what runs at `pos` next.
    pub fn plain_step_held(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let history = self
            .ngram
            .as_ref()
            .and_then(|ng| self.layers[ng.meta.layer].ngram.as_ref())
            .map(|ple| ple.history.clone());
        self.update_graph_state(pos)?;
        let x = Tensor::new(&[token], &Device::Cpu)?.unsqueeze(0)?;
        self.embed_for_graph(&x)?;
        let checkpoint = self.recurrent_checkpoint()?;
        let logits = self.forward_from_hidden();
        self.recurrent_restore(&checkpoint)?;
        if let (Some(history), Some(ng)) = (history, self.ngram.as_ref()) {
            if let Some(ple) = self.layers[ng.meta.layer].ngram.as_mut() {
                ple.history = history;
            }
        }
        logits?.flatten_all()?.to_vec1::<f32>()
    }

    /// Forget the buffers of the recorded verifies: the graphs that read them are gone.
    pub(super) fn clear_verify_state(&mut self) {
        self.verify = VerifyBuffers::default();
        for l in &mut self.layers {
            l.clear_verify_state();
        }
    }
}
