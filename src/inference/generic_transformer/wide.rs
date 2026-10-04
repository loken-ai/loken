//! A layer of a wide-residual model: the residual is `streams` copies of the hidden state,
//! each sublayer reads it through a mixer and writes back to every stream, and the token mixer
//! of an indexed-attention layer reads only the tokens its indexer keeps.

#[cfg(feature = "cuda")]
use super::hyper::combine_on_card;
use super::hyper::{combine, WideBlock};
use super::*;
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

fn sublayer_dtype_of(dev: &Device) -> DType {
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
        if settle {
            let _ = dev.synchronize();
        }
        return r;
    }
    let _ = dev.synchronize();
    let started = std::time::Instant::now();
    let r = f();
    let _ = dev.synchronize();
    crate::inference::offload::prof_record(name, started.elapsed().as_nanos() as u64);
    r
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
        let (s, d) = {
            let h = self.hyper.as_ref().expect("a wide layer has its mixers");
            (h.attn.streams(), h.attn.dim())
        };
        let (mixed, inj) = {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            let xn = hc.attn.normed_on_card(x)?;
            hc.attn.mix_on_card(&xn, true)?
        };
        let out = match self.mixer {
            mixer::Mixer::Attention => {
                let mut o = self.forward_attn_graph(&mixed)?;
                if let Some(b) = &self.attn_output_bias {
                    o = o.broadcast_add(b)?;
                }
                o
            }
            _ => {
                let m = self.mixer.step_in_place(&mixed)?;
                self.attn_output.forward(&m)?
            }
        };
        let (r, xn) = {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            combine_on_card(
                x,
                &out,
                &inj.expect("the token mixer's mixer writes back"),
                Some(&hc.ffn),
                s,
                d,
            )?
        };
        let (mixed, inj) = {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            hc.ffn
                .mix_on_card(&xn.expect("the norm was asked for"), true)?
        };
        let out = self.feed_forward(&mixed, &dev)?;
        let (r, _) = combine_on_card(
            &r,
            &out,
            &inj.expect("the feed-forward's mixer writes back"),
            None,
            s,
            d,
        )?;
        Ok(r)
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
        staged(
            if attends { "attention" } else { "delta_rule" },
            dev,
            settle,
            || {
                if attends {
                    self.wide_attention(mixed, mask, index_pos)
                } else {
                    let m = self.mixer.forward(mixed, index_pos)?;
                    self.attn_output.forward(&m)
                }
            },
        )
    }

    /// The routed experts and the shared expert, or the dense feed-forward.
    fn feed_forward(&mut self, mixed: &Tensor, dev: &Device) -> Result<Tensor> {
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
        let out = self.token_mixer(&mixed, mask, index_pos, dev)?;
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
        let (mixed, inj) = staged("mix_ffn", dev, settle, || {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            hc.ffn
                .mix_on_card(&xn.expect("the norm was asked for"), f16)
        })?;
        let out = self.feed_forward(&mixed, dev)?;
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
        let (mut q, mut k, v) = match (self.attn_qkv.as_ref(), self.attn_q.as_ref()) {
            (Some(w), _) => {
                let qkv = w.forward(x)?;
                let (qd, kd) = (n_head * hd, n_kv * hd);
                (
                    qkv.narrow(D::Minus1, 0, qd)?,
                    qkv.narrow(D::Minus1, qd, kd)?,
                    qkv.narrow(D::Minus1, qd + kd, kd)?,
                )
            }
            (None, Some(wq)) => (
                wq.forward(x)?,
                self.attn_k.as_ref().expect("split k").forward(x)?,
                self.attn_v.as_ref().expect("split v").forward(x)?,
            ),
            _ => crate::tensor::bail!("an attention layer without projections"),
        };
        q = q.reshape((b, seq, n_head, hd))?.transpose(1, 2)?;
        k = k.reshape((b, seq, n_kv, hd))?.transpose(1, 2)?;
        let mut v = v
            .reshape((b, seq, n_kv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        if let (Some(qn), Some(kn)) = (&self.attn_q_norm, &self.attn_k_norm) {
            q = qn.forward(&q.contiguous()?)?;
            k = kn.forward(&k.contiguous()?)?;
        }
        q = self.apply_rotary_emb(&q, index_pos)?;
        k = self.apply_rotary_emb(&k, index_pos)?;
        if index_pos == 0 {
            self.kv_cache.reset();
        }
        if self.kv_f16 {
            k = k.to_dtype(DType::F16)?;
            v = v.to_dtype(DType::F16)?;
        }
        let (k_all, v_all) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
        let kv_len = k_all.dim(2)?;
        let rep = n_head / n_kv;
        let spread = |t: Tensor| -> Result<Tensor> {
            t.to_dtype(DType::F32)?
                .unsqueeze(2)?
                .broadcast_as((b, n_kv, rep, kv_len, hd))?
                .reshape((b, n_head, kv_len, hd))
        };
        let (k_all, v_all) = (spread(k_all)?, spread(v_all)?);
        let mut bias = vec![f32::MIN / 2.0; seq * kv_len];
        for (row, kept) in bias.chunks_exact_mut(kv_len).zip(selection) {
            for t in kept {
                row[*t as usize] = 0.0;
            }
        }
        let bias = Tensor::from_vec(bias, (1, 1, seq, kv_len), &q.device())?;
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt());
        let scores = (q.to_dtype(DType::F32)?.matmul(&k_all.transpose(2, 3)?)? * scale)?;
        let probs = crate::tensor::ops::softmax_last_dim(&scores.broadcast_add(&bias)?)?;
        let y = probs
            .matmul(&v_all)?
            .transpose(1, 2)?
            .reshape((b, seq, n_head * hd))?;
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
            index: self.sparse.as_ref().map(|s| s.cache.clone()),
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
                    sp.cache = c.clone();
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
                    sp.cache = c.clone();
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
            sp.cache.clear();
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
