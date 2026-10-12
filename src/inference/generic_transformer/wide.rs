//! A layer of a wide-residual model: the residual is `streams` copies of the hidden state,
//! each sublayer reads it through a mixer and writes back to every stream, and the token mixer
//! of an indexed-attention layer reads only the tokens its indexer keeps.

#[cfg(feature = "cuda")]
use super::hyper::combine_on_card;
use super::hyper::{combine, WideBlock};
use super::*;
#[cfg(feature = "cuda")]
use crate::inference::moe_cuda::StepPos;
use crate::tensor::D;

/// A weight as a projection that accepts the activations the wide residual carries: dense
/// float weights are held as F32 whatever their stored width, quantised ones as they are.
pub(super) fn load_matrix(qt: QTensor) -> Result<QMatMul> {
    use crate::tensor::quantized::GgmlDType::{BF16, F16, F32};
    match qt.dtype() {
        F16 | BF16 => {
            let dev = qt.device().clone();
            // CONVERSION: a float weight is held F32 beside the wide residual's F32 activations.
            let w = qt.dequantize(&dev)?.to_dtype(DType::F32)?;
            QMatMul::from_qtensor(QTensor::quantize_onto(&w, F32, &dev)?)
        }
        _ => QMatMul::from_qtensor(qt),
    }
}

/// Whether the rows are a prefill's: more than the one a decode step carries.
fn prefilling(x: &Tensor) -> bool {
    x.dims().get(1).is_some_and(|&n| n > 1)
}

/// The activation type the sublayers read: what the card kernels take, F32 on the host.
fn sublayer_dtype(x: &Tensor) -> DType {
    sublayer_dtype_of(&x.device())
}

/// Calls of the wide layer at the prompt's start, for the stage probe's names.
static WIDE_PROBE_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Attention layers probed at the prompt's start, in call order.
static ATTN_PROBE_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(super) fn sublayer_dtype_of(dev: &Device) -> DType {
    if dev.is_cpu() {
        DType::F32
    } else {
        DType::F16
    }
}

/// [`load_matrix`] for the tensor-level projection a router uses.
pub(super) fn dense_router(qt: QTensor) -> Result<crate::tensor::quantized::QMatMul> {
    use crate::tensor::quantized::GgmlDType::{BF16, F16, F32};
    match qt.dtype() {
        F16 | BF16 => {
            let dev = qt.device().clone();
            // CONVERSION: a float router weight is held F32, as the wide residual's projections are.
            let w = qt.dequantize(&dev)?.to_dtype(DType::F32)?;
            crate::tensor::quantized::QMatMul::from_qtensor(QTensor::quantize_onto(&w, F32, &dev)?)
        }
        _ => crate::tensor::quantized::QMatMul::from_qtensor(qt),
    }
}

/// Whether a prefill stage drains the card behind it. The frees a stage queues complete before
/// the next stage allocates, which a card with little room needs: the next stage is allocated
/// against the memory the card has, not against what the host has run ahead of. A card with
/// room for the pass has no need, and the host queues stages while the card runs.
static PREFILL_SETTLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set at the start of a prefill pass from the room the cards have for it, and off when the pass
/// ends: the rows of a draft verify are few and queue behind each other without it.
pub fn set_prefill_settle(on: bool) {
    PREFILL_SETTLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Run `f` as the named stage of the wide layer: timed, with the device drained either side so the
/// time is the stage's own, when the stage profile is on. With `settle` the device is drained after
/// the stage, so the frees it queued complete before the next stage allocates: a prefill stage is
/// allocated against the memory the card has, not against what the host has run ahead of.
pub(super) fn staged<R>(
    name: &'static str,
    dev: &crate::tensor::Device,
    settle: bool,
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    if !crate::inference::offload::stage_prof_enabled() {
        let r = f();
        if settle && PREFILL_SETTLE.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = dev.synchronize();
        }
        return r;
    }
    #[cfg(feature = "cuda")]
    if crate::tensor::cuda_ext::capture_active(dev) {
        if let Ok(card) = dev.as_cuda_device() {
            let stream = card.0.stream().clone();
            return crate::inference::offload::mark_stage_in_capture(name, &stream, f);
        }
    }
    let _ = dev.synchronize();
    let started = std::time::Instant::now();
    let r = f();
    let _ = dev.synchronize();
    crate::inference::offload::prof_record(name, started.elapsed().as_nanos() as u64);
    r
}

/// The bytes the live scores of one block of rows of selected attention may take: the scores of
/// every head, the scaled copy, the masked one and the probabilities, at most this at a time.
const SCORE_BUDGET_BYTES: usize = 128 << 20;
/// How many copies of a block's scores are live at once in selected attention.
const SCORE_COPIES: usize = 3;

/// Where the rotation of a token's heads comes from: the layer's tables at a position the host
/// knows, the step's buffers a replay refreshes, or the rows of a verify's tables.
#[cfg(feature = "cuda")]
enum RopeSource {
    Host(usize),
    Step,
    Rows(Tensor, Tensor),
}

/// The tables one row of a recorded verify rotates with: the heads' cosines and sines, and the
/// indexer's.
#[cfg(feature = "cuda")]
struct RowTables {
    cos: Tensor,
    sin: Tensor,
    rot: Tensor,
}

impl GenericTransformerLayer {
    /// The wide block of this layer, `None` on a model with a plain residual.
    pub(super) fn is_wide(&self) -> bool {
        self.hyper.is_some()
    }

    /// One layer: the optional n-gram features, the token mixer behind its mixer, then the
    /// feed-forward behind its own. `ngram` carries the gathered embeddings of the rows when
    /// this layer is the one that takes them.
    pub(super) fn forward_wide(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        ngram: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dev = x.device();
        let settle = prefilling(x);
        let mut r = x.to_dtype(DType::F32)?;
        if self.spec_record {
            self.spec_begin(&r, ngram, index_pos)?;
        }
        if let (Some(ple), Some(emb)) = (self.ngram.as_mut(), ngram) {
            let add = staged("ngram", &dev, settle, || ple.forward(&r, emb, index_pos))?;
            r = (r + add)?;
        }
        #[cfg(feature = "cuda")]
        if dev.is_cuda()
            && self
                .hyper
                .as_ref()
                .is_some_and(|h| h.attn.on_card() && h.ffn.on_card())
        {
            return self.forward_wide_on_card(r, mask, index_pos, &dev);
        }
        let (mixed, inj) = staged("mix_token", &dev, settle, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            let (mixed, inj) = hc.attn.mix(&r)?;
            Ok((mixed.to_dtype(sublayer_dtype(&mixed))?, inj))
        })?;
        let out = self.token_mixer(&mixed, mask, index_pos, &dev)?;
        let inj = inj.expect("the token mixer's mixer writes back");
        r = staged("combine", &dev, settle, || combine(&r, &out, &inj))?;
        if self.kv_only {
            return Ok(r);
        }
        let (mixed, inj) = staged("mix_ffn", &dev, settle, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            let (mixed, inj) = hc.ffn.mix(&r)?;
            Ok((mixed.to_dtype(sublayer_dtype(&mixed))?, inj))
        })?;
        let out = self.feed_forward(&mixed, &dev)?;
        staged("combine", &dev, settle, || {
            combine(
                &r,
                &out,
                &inj.expect("the feed-forward's mixer writes back"),
            )
        })
    }

    /// One layer of the captured decode step: the same stages with the recurrent mixer
    /// advancing its state where it lies and attention reading the graph's position buffers.
    /// The residual is the F32 `[1, 1, streams * dim]` the graph's hidden buffer holds.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_wide_graph(&mut self, x: &Tensor) -> Result<Tensor> {
        let dev = x.device();
        // The n-gram features of the step's token, gathered into their buffer before the replay.
        let joined;
        let x = match (self.ngram.as_mut(), self.graph_ngram_rows.as_ref()) {
            (Some(ple), Some(rows)) => {
                // A decoded token follows others: the convolution reads the rows they left.
                let add = ple.forward(x, rows, 1)?;
                joined = (x + add)?;
                &joined
            }
            (Some(_), None) => crate::tensor::bail!("the n-gram rows of the step are not in place"),
            _ => x,
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
        let out = staged("token mixer", &dev, false, || match self.mixer {
            mixer::Mixer::Attention => {
                let mut o = match self.sparse.is_some() {
                    true => self.forward_sparse_graph(&mixed)?,
                    false => self.forward_attn_graph(&mixed)?,
                };
                if let Some(b) = &self.attn_output_bias {
                    o = o.broadcast_add(b)?;
                }
                Ok(o)
            }
            _ => {
                let m = self.mixer.step_in_place(&mixed)?;
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
        // The feed-forward's router and experts read F32: the mixer writes it, and the
        // experts' sum goes back without a cast either way.
        let (mixed, inj) = staged("mix_ffn", &dev, false, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            hc.ffn
                .mix_on_card(&xn.expect("the norm was asked for"), false)
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

    /// Whether this layer's indexer and attention run on its card: its kernels serve the
    /// layer's shapes and its one cache is the card's half-width one.
    #[cfg(feature = "cuda")]
    pub(super) fn indexed_on_card(&self) -> bool {
        self.sparse.as_ref().is_some_and(|sparse| {
            self.cos.device().is_cuda() && self.kv_f16 && sparse.card_serves(self.head_dim)
        })
    }

    /// The attention of an indexed layer in the decode step a graph records: the indexer takes
    /// the token's key into its cache on the card and keeps the tokens the query selects, the
    /// token's keys and values join the cache at the position the card holds, and one launch
    /// attends over the kept tokens. Nothing in it reads the host.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_sparse_graph(&mut self, x: &Tensor) -> Result<Tensor> {
        let pos_dev = self
            .graph_pos_i32
            .clone()
            .ok_or_else(|| crate::tensor::Error::msg("the step's position is not on the card"))?;
        self.attend_on_card(x, StepPos::Card(&pos_dev), None)
    }

    /// [`Self::forward_sparse_graph`] for a decoded token run now, at `index_pos`, whose
    /// position the host knows: the same launches, and the host's counters moved by it.
    #[cfg(feature = "cuda")]
    fn attend_on_card_eager(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.kv_cache.reserve(index_pos + 1)?;
        let room = self.kv_cache.max_seq_len_padded();
        let sparse = self
            .sparse
            .as_mut()
            .ok_or_else(|| crate::tensor::Error::msg("an indexed layer without its indexer"))?;
        sparse.card_room_for(room)?;
        sparse.card_hold(index_pos)?;
        let out = self.attend_on_card(x, StepPos::Host(index_pos), Some(index_pos))?;
        self.kv_cache.advance_seq_len(1);
        Ok(out)
    }

    /// The launches of an indexed layer's decoded token: indexer, projections, cache write and
    /// attention. `eager_at` is the position of a token run now, whose rotation is taken from the
    /// layer's tables; a recorded step takes it from the buffers its replays refresh.
    #[cfg(feature = "cuda")]
    fn attend_on_card(
        &mut self,
        x: &Tensor,
        pos: StepPos,
        eager_at: Option<usize>,
    ) -> Result<Tensor> {
        self.attend_on_card_with(x, pos, eager_at, None)
    }

    /// [`Self::attend_on_card`] for one row of a recorded verify: its position buffer, and
    /// `row`'s rotation tables of the rows prepared for the verify.
    #[cfg(feature = "cuda")]
    pub(super) fn attend_row_recorded(
        &mut self,
        x: &Tensor,
        pos: &Tensor,
        row: usize,
    ) -> Result<Tensor> {
        let (cos, sin) = match (&self.verify_rope_cos, &self.verify_rope_sin) {
            (Some(c), Some(s)) => (c.narrow(0, row, 1)?, s.narrow(0, row, 1)?),
            _ => crate::tensor::bail!("the rotation tables of the verify's rows are not in place"),
        };
        let rot = self
            .sparse
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("an indexed layer without its indexer"))?
            .rot_row(row)?;
        self.attend_on_card_with(
            x,
            StepPos::Card(pos),
            None,
            Some(RowTables { cos, sin, rot }),
        )
    }

    /// The rows of a recorded verify through an indexed layer's attention, each stage one launch
    /// over the rows with the arithmetic of a row's own step: the projections, the heads' norms
    /// and rotation, every row's indexer key and cache row in order, every row's selection and
    /// attention over what its step would see, the gate and the output projection. `None` when
    /// the rows cannot take the fused head preparation: they then run one by one.
    #[cfg(feature = "cuda")]
    pub(super) fn attend_rows_recorded(
        &mut self,
        x: &Tensor,
        pos: &Tensor,
        row0: usize,
        rows: usize,
    ) -> Result<Option<Tensor>> {
        use crate::inference::moe_cuda::QkShape;
        let hd = self.head_dim;
        if self.qk_norm_is_full() {
            return Ok(None);
        }
        let (qn, kn) = match (&self.attn_q_norm, &self.attn_k_norm) {
            (Some(qn), Some(kn)) => (Some(qn), Some(kn)),
            (None, None) => (None, None),
            _ => return Ok(None),
        };
        let (cos_rows, sin_rows) = match (&self.verify_rope_cos, &self.verify_rope_sin) {
            (Some(c), Some(s)) => (c.clone(), s.clone()),
            _ => crate::tensor::bail!("the rotation tables of the verify's rows are not in place"),
        };
        let rope_dim = match (self.no_rope, self.rope_dim) {
            (true, _) => 0,
            (false, d) if d > 0 && d < hd => d,
            (false, _) => hd,
        };
        let shape = QkShape {
            n_head: self.n_head,
            n_kv: self.n_kv_head,
            head_dim: hd,
            rope_dim,
            interleaved: self.flags.use_rope_i,
            eps: qn.map_or(0.0, |n| n.eps() as f32),
        };
        let (n_head, n_kv) = (self.n_head, self.n_kv_head);
        let (key_all, qi_all) = self
            .sparse
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("an indexed layer without its indexer"))?
            .card_projections(x)?;
        let (q_all, k_all, v_all) = self.project_rows(x)?;
        // `[rows, width]`, rows packed: a view where the projection gave them so.
        let by_row = |t: Tensor| -> Result<Tensor> {
            let width = t.dim(2)?;
            t.contiguous()?.reshape((rows, width))
        };
        let (key_all, qi_all) = (by_row(key_all)?, by_row(qi_all)?);
        let (q_all, k_all, v_all) = (by_row(q_all)?, by_row(k_all)?, by_row(v_all)?);
        let Some((q, k, v)) = crate::inference::moe_cuda::qk_prep_rows(
            &q_all,
            &k_all,
            &v_all,
            qn.map(|n| n.weight()),
            kn.map(|n| n.weight()),
            &cos_rows.narrow(0, row0, rows)?,
            &sin_rows.narrow(0, row0, rows)?,
            shape,
            rows,
        )?
        else {
            return Ok(None);
        };
        // Every row's indexer key and cache row join first, in order; each row then selects
        // and attends over what its own step would: tokens up to its position.
        let (sel, count) = self
            .sparse
            .as_mut()
            .ok_or_else(|| crate::tensor::Error::msg("an indexed layer without its indexer"))?
            .card_steps_projected(&key_all, &qi_all, pos, row0, rows)?;
        let (Some(k_buf), Some(v_buf)) = (self.kv_cache.k_buffer(), self.kv_cache.v_buffer())
        else {
            crate::tensor::bail!("an indexed layer decodes before its cache holds a token");
        };
        let cap = self.kv_cache.max_seq_len_padded();
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt()) as f32;
        crate::inference::moe_cuda::kv_append_pos_rows(
            &k, &v, &k_buf, &v_buf, pos, rows, n_kv, cap, hd,
        )?;
        let y = crate::inference::moe_cuda::selected_attention_steps(
            &q, &k_buf, &v_buf, &sel, &count, rows, n_head, n_kv, cap, hd, scale,
        )?;
        let y = self.gate_attention(x, y)?;
        self.attn_output.forward(&y).map(Some)
    }

    /// [`Self::attend_on_card`] with the rotation tables of `tables` in place of the step's.
    #[cfg(feature = "cuda")]
    fn attend_on_card_with(
        &mut self,
        x: &Tensor,
        pos: StepPos,
        eager_at: Option<usize>,
        tables: Option<RowTables>,
    ) -> Result<Tensor> {
        let sparse = self
            .sparse
            .as_mut()
            .ok_or_else(|| crate::tensor::Error::msg("an indexed layer without its indexer"))?;
        match &tables {
            Some(t) => sparse.card_step_with(x, pos, &t.rot)?,
            None => sparse.card_step(x, pos)?,
        }
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        let rope = match (&tables, eager_at) {
            (Some(t), _) => RopeSource::Rows(t.cos.clone(), t.sin.clone()),
            (None, Some(at)) => RopeSource::Host(at),
            (None, None) => RopeSource::Step,
        };
        let (q, k, v) = match self.prepared_heads(x, &rope)? {
            Some(rows) => rows,
            None => {
                let (q, k, v) = self.project_heads(x)?;
                let (q, k) = match &rope {
                    RopeSource::Host(at) => (
                        self.apply_rotary_emb(&q, *at)?,
                        self.apply_rotary_emb(&k, *at)?,
                    ),
                    RopeSource::Step => (
                        self.apply_rotary_emb_from_buffer(&q)?,
                        self.apply_rotary_emb_from_buffer(&k)?,
                    ),
                    RopeSource::Rows(cos, sin) => (
                        self.apply_rotary_with(&q, cos, sin)?,
                        self.apply_rotary_with(&k, cos, sin)?,
                    ),
                };
                let flat = |t: &Tensor| t.to_dtype(DType::F32)?.contiguous();
                (flat(&q)?, flat(&k)?, flat(&v)?)
            }
        };
        let (Some(k_buf), Some(v_buf)) = (self.kv_cache.k_buffer(), self.kv_cache.v_buffer())
        else {
            crate::tensor::bail!("an indexed layer decodes before its cache holds a token");
        };
        let cap = self.kv_cache.max_seq_len_padded();
        if eager_at.is_none() {
            // The step is recorded against these buffers: it ends, and is recorded again, when
            // the cache grows.
            self.graph_kv_cap = Some(cap);
        }
        crate::inference::moe_cuda::kv_append_pos(&k, &v, &k_buf, &v_buf, pos, n_kv, cap, hd)?;
        let sparse = self
            .sparse
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("an indexed layer without its indexer"))?;
        let (sel, count) = sparse.card_selection()?;
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt()) as f32;
        let y = crate::inference::moe_cuda::selected_attention(
            &q, &k_buf, &v_buf, sel, count, n_head, n_kv, cap, hd, scale,
        )?;
        let y = self.gate_attention(x, y)?;
        self.attn_output.forward(&y)
    }

    /// The token mixer behind its mixer's output: the delta rule through its output
    /// projection, or attention.
    fn token_mixer(
        &mut self,
        mixed: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        let settle = prefilling(mixed);
        let attends = matches!(self.mixer, mixer::Mixer::Attention);
        if self.spec_record {
            self.spec_input(mixed);
        }
        staged(
            if attends { "attention" } else { "delta_rule" },
            dev,
            settle,
            || {
                if attends {
                    self.wide_attention(mixed, mask, index_pos)
                } else {
                    // A decoded token, or the rows of a verify, advance the state where it
                    // lies, as the recorded step does, so the two follow each other on the same
                    // buffers.
                    #[cfg(feature = "cuda")]
                    let _in_place = crate::inference::moe_cuda::in_place_state(
                        index_pos > 0 && (self.spec_record || mixed.dim(1).is_ok_and(|n| n == 1)),
                    );
                    let m = self.mixer.forward(mixed, index_pos)?;
                    self.attn_output.forward(&m)
                }
            },
        )
    }

    /// The routed experts and the shared expert, or the dense feed-forward.
    pub(super) fn feed_forward(&mut self, mixed: &Tensor, dev: &Device) -> Result<Tensor> {
        let settle = prefilling(mixed);
        staged("experts", dev, settle, || match self.moe.as_ref() {
            Some(moe) => moe.forward_rows(
                mixed,
                mixed,
                self.embedding_length_for_moe,
                self.ffn_norm_eps,
            ),
            None => self.forward_ffn(mixed),
        })
    }

    /// The same layer with each elementwise stage of the wide residual one launch, the norm of
    /// the feed-forward's mixer joined to the write-back that ends the token mixer.
    #[cfg(feature = "cuda")]
    fn forward_wide_on_card(
        &mut self,
        r: Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        let f16 = sublayer_dtype_of(dev) == DType::F16;
        let settle = prefilling(&r);
        let (s, d) = {
            let h = self.hyper.as_ref().expect("a wide layer has its mixers");
            (h.attn.streams(), h.attn.dim())
        };
        let (mixed, inj) = staged("mix_token", dev, settle, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            let xn = hc.attn.normed_on_card(&r)?;
            hc.attn.mix_on_card(&xn, f16)
        })?;
        // A probe of the first chunk's stages, layer by layer in call order.
        let probe = if index_pos == 0 {
            super::draft_head::probe_dir().map(|dir| {
                let k = WIDE_PROBE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                (dir, k)
            })
        } else {
            None
        };
        if let Some((dir, k)) = &probe {
            super::draft_head::dump_rows(dir, &format!("w{k}_a_mixed"), &mixed);
        }
        let out = staged("token mixer", dev, settle, || {
            self.token_mixer(&mixed, mask, index_pos, dev)
        })?;
        if let Some((dir, k)) = &probe {
            super::draft_head::dump_rows(dir, &format!("w{k}_b_mixer"), &out);
        }
        let (r, xn) = staged("combine", dev, settle, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            combine_on_card(
                &r,
                &out,
                &inj.expect("the token mixer's mixer writes back"),
                Some(&hc.ffn),
                s,
                d,
            )
        })?;
        if self.kv_only {
            return Ok(r);
        }
        let (mixed, inj) = staged("mix_ffn", dev, settle, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            hc.ffn
                .mix_on_card(&xn.expect("the norm was asked for"), f16)
        })?;
        if let Some((dir, k)) = &probe {
            super::draft_head::dump_rows(dir, &format!("w{k}_c_after_mixer"), &r);
            super::draft_head::dump_rows(dir, &format!("w{k}_d_ffn_in"), &mixed);
        }
        let out = self.feed_forward(&mixed, dev)?;
        if let Some((dir, k)) = &probe {
            super::draft_head::dump_rows(dir, &format!("w{k}_e_ffn_out"), &out);
        }
        let (r, _) = staged("combine", dev, settle, || {
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

    /// Attention over the cache: dense while the context fits the indexer's budget, over
    /// the tokens the indexer keeps for each row after it.
    pub(super) fn wide_attention(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        // The rows of a draft verify, each through the decoded token's launches on the card.
        #[cfg(feature = "cuda")]
        if self.indexed_on_card()
            && x.dim(1)? > 1
            && index_pos > 0
            && self.spec_record
            && self.kv_cache.k_buffer().is_some()
            && self.kv_cache.current_seq_len() == index_pos
        {
            let rows = (0..x.dim(1)?)
                .map(|r| {
                    self.attend_on_card_eager(&x.narrow(1, r, 1)?.contiguous()?, index_pos + r)
                })
                .collect::<Result<Vec<_>>>()?;
            return Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1);
        }
        // A decoded token of a layer whose indexer runs on the card never reaches the host.
        #[cfg(feature = "cuda")]
        if self.indexed_on_card()
            && x.dim(1)? == 1
            && index_pos > 0
            && self.kv_cache.k_buffer().is_some()
            && self.kv_cache.current_seq_len() == index_pos
        {
            return self.attend_on_card_eager(x, index_pos);
        }
        // A prompt's rows over an F16 cache of a shape the kernels serve are selected on the card
        // and attended there, their queries never leaving it.
        #[cfg(feature = "cuda")]
        if x.dim(1)? > 1
            && self.kv_f16
            && self.indexed_on_card()
            && crate::inference::moe_cuda::selected_attention_rows_serves(
                self.head_dim,
                self.n_head,
                self.n_kv_head,
            )
        {
            let sparse = self.sparse.as_mut().expect("an indexed layer");
            return match sparse.observe_on_card(x, index_pos)? {
                super::qsa::CardPick::Kept { sel, count } => {
                    self.attend_selected_card(x, index_pos, &sel, &count)
                }
                super::qsa::CardPick::Dense => self.forward_attn(
                    x,
                    mask,
                    index_pos,
                    None,
                    None,
                    #[cfg(feature = "cuda")]
                    None,
                ),
            };
        }
        let selection = match self.sparse.as_mut() {
            Some(sp) => sp.observe(x, index_pos)?,
            None => None,
        };
        match selection {
            Some(sel) => self.attend_selected(x, index_pos, &sel),
            None => self.forward_attn(
                x,
                mask,
                index_pos,
                None,
                None,
                #[cfg(feature = "cuda")]
                None,
            ),
        }
    }

    /// The projections of `x` `[b, seq, hidden]` split into heads, normalised where the layer
    /// norms them and not yet rotated: `q` `[b, n_head, seq, hd]`, `k` and `v`
    /// `[b, n_kv, seq, hd]`.
    fn project_heads(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, seq, _) = x.dims3()?;
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        let (mut q, mut k, v) = self.project_rows(x)?;
        q = q.reshape((b, seq, n_head, hd))?.transpose(1, 2)?;
        k = k.reshape((b, seq, n_kv, hd))?.transpose(1, 2)?;
        let v = v
            .reshape((b, seq, n_kv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        if let (Some(qn), Some(kn)) = (&self.attn_q_norm, &self.attn_k_norm) {
            q = qn.forward(&q.contiguous()?)?;
            k = kn.forward(&k.contiguous()?)?;
        }
        Ok((q, k, v))
    }

    /// The queries, keys and values of the rows `x` `[1, seq, hidden]` of a prompt chunk at
    /// `index_pos`, as [`Self::attend_selected`] reads them: the queries and keys rotated, the
    /// cache started over at position 0, the keys and values in the cache's width.
    #[cfg(feature = "cuda")]
    fn selected_inputs(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (mut q, mut k, mut v) = self.project_heads(x)?;
        if index_pos == 0 {
            if let Some(dir) = super::draft_head::probe_dir() {
                let k0 = ATTN_PROBE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::draft_head::dump_rows(dir, &format!("attn{k0}_q_normed"), &q);
                super::draft_head::dump_rows(dir, &format!("attn{k0}_k_normed"), &k);
                super::draft_head::dump_rows(dir, &format!("attn{k0}_v"), &v);
            }
        }
        q = self.apply_rotary_emb(&q, index_pos)?;
        k = self.apply_rotary_emb(&k, index_pos)?;
        if index_pos == 0 {
            if let Some(dir) = super::draft_head::probe_dir() {
                let k0 = ATTN_PROBE_CALLS.load(std::sync::atomic::Ordering::Relaxed) - 1;
                super::draft_head::dump_rows(dir, &format!("attn{k0}_q_roped"), &q);
                super::draft_head::dump_rows(dir, &format!("attn{k0}_k_roped"), &k);
            }
            self.kv_cache.reset();
        }
        if self.kv_f16 {
            k = k.to_dtype(DType::F16)?;
            v = v.to_dtype(DType::F16)?;
        }
        Ok((q, k, v))
    }

    /// [`Self::attend_selected`] for a prompt chunk whose kept tokens the card selected:
    /// `sel` `[rows, room]` and `count` `[rows]` are on it.
    #[cfg(feature = "cuda")]
    fn attend_selected_card(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        sel: &Tensor,
        count: &Tensor,
    ) -> Result<Tensor> {
        let (q, k, v) = self.selected_inputs(x, index_pos)?;
        self.attend_rows_kept(x, &q, &k, &v, sel, count)
    }

    /// [`Self::attend_selected`] for a prompt chunk's rows, by one launch of the kernel that
    /// attends each row over its own kept tokens in the F16 cache.
    #[cfg(feature = "cuda")]
    fn attend_selected_rows(
        &mut self,
        x: &Tensor,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        selection: &[Vec<u32>],
    ) -> Result<Tensor> {
        let seq = x.dim(1)?;
        let ncap = selection.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut sel = vec![0u32; seq * ncap];
        let mut count = Vec::with_capacity(seq);
        for (r, row) in selection.iter().enumerate() {
            sel[r * ncap..r * ncap + row.len()].copy_from_slice(row);
            count.push(row.len() as u32);
        }
        let dev = q.device();
        let sel = Tensor::from_vec(sel, (seq, ncap), &dev)?;
        let count = Tensor::from_vec(count, (seq,), &dev)?;
        self.attend_rows_kept(x, q, k, v, &sel, &count)
    }

    /// The rows' keys and values join the cache, and each row attends over the tokens `sel` and
    /// `count` give it.
    #[cfg(feature = "cuda")]
    fn attend_rows_kept(
        &mut self,
        x: &Tensor,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sel: &Tensor,
        count: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        self.kv_cache
            .append_in_place(&k.contiguous()?, &v.contiguous()?)?;
        let (Some(k_buf), Some(v_buf)) = (self.kv_cache.k_buffer(), self.kv_cache.v_buffer())
        else {
            crate::tensor::bail!("selected attention before the cache holds a token");
        };
        let cap = k_buf.dim(2)?;
        let q = q
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((seq, n_head * hd))?;
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt()) as f32;
        let y = crate::inference::moe_cuda::selected_attention_rows(
            &q, &k_buf, &v_buf, sel, count, n_head, n_kv, cap, hd, scale,
        )?
        .reshape((b, seq, n_head * hd))?;
        if let Some(dir) = super::draft_head::probe_dir() {
            if self.kv_cache.current_seq_len() == seq {
                let k0 = ATTN_PROBE_CALLS
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .saturating_sub(1);
                super::draft_head::dump_rows(dir, &format!("attn{k0}_y"), &y);
            }
        }
        let y = self.gate_attention(x, y)?;
        self.attn_output.forward(&y)
    }

    /// The projected rows of `x` `[b, seq, hidden]`: queries, keys and values, as the
    /// projections give them, before any norm or rotation.
    fn project_rows(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        match (self.attn_qkv.as_ref(), self.attn_q.as_ref()) {
            (Some(w), _) => {
                let qkv = w.forward(x)?;
                let (qd, kd) = (n_head * hd, n_kv * hd);
                Ok((
                    qkv.narrow(D::Minus1, 0, qd)?,
                    qkv.narrow(D::Minus1, qd, kd)?,
                    qkv.narrow(D::Minus1, qd + kd, kd)?,
                ))
            }
            (None, Some(wq)) => Ok((
                wq.forward(x)?,
                self.attn_k.as_ref().expect("split k").forward(x)?,
                self.attn_v.as_ref().expect("split v").forward(x)?,
            )),
            _ => crate::tensor::bail!("an attention layer without projections"),
        }
    }

    /// The queries, keys and values of the decoded token `x` `[1, 1, hidden]` as F32 rows
    /// `[1, heads, 1, head_dim]`, normed and rotated by one launch; `None` where the launch does
    /// not serve this layer. `at` is the position of a token run now, whose rotation is taken
    /// from the layer's tables; a recorded step takes it from the buffers its replays refresh.
    #[cfg(feature = "cuda")]
    fn prepared_heads(
        &self,
        x: &Tensor,
        rope: &RopeSource,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        use crate::inference::moe_cuda::{qk_prep, QkShape};
        let hd = self.head_dim;
        if x.dim(1)? != 1 || self.qk_norm_is_full() {
            return Ok(None);
        }
        let rope_dim = match (self.no_rope, self.rope_dim) {
            (true, _) => 0,
            (false, d) if d > 0 && d < hd => d,
            (false, _) => hd,
        };
        let (cos, sin) = match rope {
            RopeSource::Host(index_pos) => match &self.rope_rows {
                Some((cos, sin)) => (cos.to_device(&x.device())?, sin.to_device(&x.device())?),
                None => self.rope_table().rows(*index_pos, 1, &x.device())?,
            },
            RopeSource::Step => match (&self.graph_rope_cos, &self.graph_rope_sin) {
                (Some(cos), Some(sin)) => (cos.clone(), sin.clone()),
                _ => return Ok(None),
            },
            RopeSource::Rows(cos, sin) => (cos.clone(), sin.clone()),
        };
        let (q, k, v) = self.project_rows(x)?;
        let (qn, kn) = match (&self.attn_q_norm, &self.attn_k_norm) {
            (Some(qn), Some(kn)) => (Some(qn), Some(kn)),
            (None, None) => (None, None),
            _ => return Ok(None),
        };
        qk_prep(
            &q,
            &k,
            &v,
            qn.map(|n| n.weight()),
            kn.map(|n| n.weight()),
            &cos,
            &sin,
            QkShape {
                n_head: self.n_head,
                n_kv: self.n_kv_head,
                head_dim: hd,
                rope_dim,
                interleaved: self.flags.use_rope_i,
                eps: qn.map_or(0.0, |n| n.eps() as f32),
            },
        )
    }

    /// Attention in which row `i` reads only the cached tokens `selection[i]`: the
    /// projections, norms and rotation of the dense path, then scores masked to the kept
    /// tokens.
    pub(super) fn attend_selected(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        selection: &[Vec<u32>],
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        let (mut q, mut k, mut v) = self.project_heads(x)?;
        let probe = if index_pos == 0 {
            super::draft_head::probe_dir().map(|dir| {
                (
                    dir,
                    ATTN_PROBE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                )
            })
        } else {
            None
        };
        if let Some((dir, k0)) = &probe {
            super::draft_head::dump_rows(dir, &format!("attn{k0}_q_normed"), &q);
            super::draft_head::dump_rows(dir, &format!("attn{k0}_k_normed"), &k);
            super::draft_head::dump_rows(dir, &format!("attn{k0}_v"), &v);
        }
        q = self.apply_rotary_emb(&q, index_pos)?;
        k = self.apply_rotary_emb(&k, index_pos)?;
        if let Some((dir, k0)) = &probe {
            super::draft_head::dump_rows(dir, &format!("attn{k0}_q_roped"), &q);
            super::draft_head::dump_rows(dir, &format!("attn{k0}_k_roped"), &k);
        }
        if index_pos == 0 {
            self.kv_cache.reset();
        }
        if self.kv_f16 {
            k = k.to_dtype(DType::F16)?;
            v = v.to_dtype(DType::F16)?;
        }
        let rep = n_head / n_kv;
        if let (1, 1, [kept]) = (b, seq, selection) {
            // One token: only the kept positions are read, gathered from the cache where it
            // lies, so nothing of the history is copied or widened.
            self.kv_cache
                .append_in_place(&k.contiguous()?, &v.contiguous()?)?;
            let (Some(k_buf), Some(v_buf)) = (self.kv_cache.k_buffer(), self.kv_cache.v_buffer())
            else {
                crate::tensor::bail!("selected attention before the cache holds a token");
            };
            let at = Tensor::from_vec(kept.clone(), (kept.len(),), &q.device())?;
            let k_kept = k_buf.index_select(&at, 2)?.to_dtype(DType::F32)?;
            let v_kept = v_buf.index_select(&at, 2)?.to_dtype(DType::F32)?;
            let scale = self
                .attention_scale
                .unwrap_or_else(|| 1.0 / (hd as f64).sqrt());
            let q = q
                .to_dtype(DType::F32)?
                .contiguous()?
                .reshape((1, n_kv, rep, hd))?;
            let scores = (q.matmul(&k_kept.transpose(2, 3)?)? * scale)?;
            let probs = crate::tensor::ops::softmax_last_dim(&scores)?;
            let y = probs.matmul(&v_kept)?.reshape((1, n_head, 1, hd))?;
            let y = y.transpose(1, 2)?.reshape((1, 1, n_head * hd))?;
            let y = self.gate_attention(x, y)?;
            return self.attn_output.forward(&y);
        }
        // A prompt's rows over an F16 cache of the shape the kernel serves are attended in one
        // launch, each over its own kept tokens, with no block of scores in between.
        #[cfg(feature = "cuda")]
        if self.kv_f16
            && b == 1
            && seq > 1
            && q.device().is_cuda()
            && crate::inference::moe_cuda::selected_attention_rows_serves(hd, n_head, n_kv)
        {
            return self.attend_selected_rows(x, &q, &k, &v, selection);
        }
        let (k_all, v_all) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
        let kv_len = k_all.dim(2)?;
        // The heads of a group read the same keys and values: they stand as extra rows of one
        // product per group, so the cache is read once and never repeated across its heads.
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt());
        let q = q.to_dtype(DType::F32)?.contiguous()?;
        // The scores of a block of rows at a time: all of them at once is `heads x rows x kv_len`
        // floats, several times over, which a long chunk over a long cache cannot hold. The block
        // is as many rows as keep the live copies of its scores within `SCORE_BUDGET_BYTES`, up to
        // the mixers' own block: a long cache takes short blocks, each of which reads the cache
        // once, which costs little against the product it feeds.
        let live_scores = SCORE_COPIES * n_head * kv_len * std::mem::size_of::<f32>();
        let block = (SCORE_BUDGET_BYTES / live_scores.max(1)).clamp(1, hyper::mixer_block_rows());
        let mut parts = Vec::with_capacity(seq.div_ceil(block));
        // The scores and the sums over the cache's F16 keys and values run on the tensor cores:
        // the operands carry no more than their ten bits, and the sums are F32.
        #[cfg(feature = "cuda")]
        let _tf32 = crate::tensor::cuda::Tf32Scope::enter();
        // The rank of a position in the union of what a block keeps, `u32::MAX` for a position
        // outside it.
        let mut rank = vec![u32::MAX; kv_len];
        for (first, kept) in (0..seq).step_by(block).zip(selection.chunks(block)) {
            let rows = kept.len();
            // The rows of a block keep neighbouring parts of the cache: the product reads the
            // union of what they keep, a fraction of the cache, and each row masks the rest of it.
            let mut union: Vec<u32> = Vec::new();
            for row in kept {
                for &t in row {
                    if rank[t as usize] == u32::MAX {
                        rank[t as usize] = 0;
                        union.push(t);
                    }
                }
            }
            union.sort_unstable();
            for (r, &t) in union.iter().enumerate() {
                rank[t as usize] = r as u32;
            }
            // The width is a multiple of eight: the tensor cores take an aligned product, and
            // the columns added to reach it are masked.
            let width = union.len().next_multiple_of(8);
            let mut bias = vec![f32::MIN / 2.0; rows * width];
            for (row, kept) in bias.chunks_exact_mut(width).zip(kept) {
                for &t in kept {
                    row[rank[t as usize] as usize] = 0.0;
                }
            }
            for &t in &union {
                rank[t as usize] = u32::MAX;
            }
            union.resize(width, 0);
            let bias = Tensor::from_vec(bias, (1, 1, 1, rows, width), &q.device())?;
            let at = Tensor::from_vec(union, (width,), &q.device())?;
            let k_u = k_all
                .index_select(&at, 2)?
                .to_dtype(DType::F32)?
                .transpose(2, 3)?;
            let v_u = v_all.index_select(&at, 2)?.to_dtype(DType::F32)?;
            let q_rows =
                q.narrow(2, first, rows)?
                    .contiguous()?
                    .reshape((b, n_kv, rep * rows, hd))?;
            let scores = (q_rows.matmul(&k_u)? * scale)?.reshape((b, n_kv, rep, rows, width))?;
            let probs = crate::tensor::ops::softmax_last_dim(&scores.broadcast_add(&bias)?)?
                .reshape((b, n_kv, rep * rows, width))?;
            parts.push(probs.matmul(&v_u)?.reshape((b, n_head, rows, hd))?);
        }
        let y = if parts.len() == 1 {
            parts.remove(0)
        } else {
            Tensor::cat(&parts.iter().collect::<Vec<_>>(), 2)?
        };
        let y = y.transpose(1, 2)?.reshape((b, seq, n_head * hd))?;
        let y = self.gate_attention(x, y)?;
        self.attn_output.forward(&y)
    }
}

impl GenericTransformerLayer {
    /// The layer's recurrent state, copied: its mixer's and, on a wide layer, the caches its
    /// indexer and n-gram block carry. `None` for a layer with no state.
    pub(super) fn state_snapshot(&self) -> Result<Option<mixer::MixerState>> {
        let mixer = self.mixer.snapshot()?;
        self.with_wide_state(mixer, |p| p.snapshot())
    }

    /// [`Self::state_snapshot`] for a checkpoint a step may be run twice from.
    pub(super) fn state_checkpoint(&self) -> Result<Option<mixer::MixerState>> {
        let mixer = self.mixer.checkpoint()?;
        self.with_wide_state(mixer, |p| p.snapshot())
    }

    fn with_wide_state(
        &self,
        mixer: Option<mixer::MixerState>,
        ple: impl Fn(&ngram::PleBlock) -> Result<ngram::PleState>,
    ) -> Result<Option<mixer::MixerState>> {
        if self.sparse.is_none() && self.ngram.is_none() {
            return Ok(mixer);
        }
        Ok(Some(mixer::MixerState::Wide(Box::new(mixer::WideState {
            mixer,
            index: self
                .sparse
                .as_ref()
                .map(|s| s.snapshot_cache())
                .transpose()?,
            ngram: self.ngram.as_ref().map(ple).transpose()?,
        }))))
    }

    /// Make `state` the layer's state; the copy stays reusable.
    pub(super) fn state_restore(&mut self, state: &mixer::MixerState) -> Result<()> {
        match state {
            mixer::MixerState::Wide(w) => {
                if let Some(m) = &w.mixer {
                    self.mixer.restore(m)?;
                } else {
                    self.mixer.reset();
                }
                if let (Some(sp), Some(c)) = (self.sparse.as_mut(), w.index.as_ref()) {
                    sp.restore_cache(c, false);
                }
                if let (Some(p), Some(st)) = (self.ngram.as_mut(), w.ngram.as_ref()) {
                    p.restore(st)?;
                }
                Ok(())
            }
            other => self.mixer.restore(other),
        }
    }

    /// [`Self::state_restore`] into the buffers the state lives in.
    pub(super) fn state_restore_in_place(&mut self, state: &mixer::MixerState) -> Result<()> {
        match state {
            mixer::MixerState::Wide(w) => {
                if let Some(m) = &w.mixer {
                    self.mixer.restore_in_place(m)?;
                }
                if let (Some(sp), Some(c)) = (self.sparse.as_mut(), w.index.as_ref()) {
                    sp.restore_cache(c, true);
                }
                if let (Some(p), Some(st)) = (self.ngram.as_mut(), w.ngram.as_ref()) {
                    p.restore(st)?;
                }
                Ok(())
            }
            other => self.mixer.restore_in_place(other),
        }
    }

    /// Forget the sequence: the mixer's state and the caches of a wide layer.
    pub(super) fn state_reset(&mut self) {
        self.mixer.reset();
        if let Some(sp) = self.sparse.as_mut() {
            sp.reset();
        }
        if let Some(p) = self.ngram.as_mut() {
            p.reset();
        }
    }
}

/// The residual a token starts from, one copy of its embedding per stream.
pub(super) fn widen_embedding(hidden: &Tensor, streams: usize) -> Result<Tensor> {
    super::hyper::widen(&hidden.to_dtype(DType::F32)?, streams)
}

#[allow(dead_code)]
fn _wide_block_is_used(_: &WideBlock) {}
