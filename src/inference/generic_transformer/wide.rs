//! A layer of a wide-residual model: the residual is `streams` copies of the hidden state,
//! each sublayer reads it through a mixer and writes back to every stream, and the token mixer
//! of an indexed-attention layer reads only the tokens its indexer keeps.

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

/// The activation type the sublayers read: what the card kernels take, F32 on the host.
fn sublayer_dtype(x: &Tensor) -> DType {
    if x.device().is_cpu() {
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
        let mut r = x.to_dtype(DType::F32)?;
        if let (Some(ple), Some(emb)) = (self.ngram.as_mut(), ngram) {
            let add = ple.forward(&r, emb, index_pos)?;
            r = (r + add)?;
        }
        let (mixed, inj) = {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            hc.attn.mix(&r)?
        };
        let mixed = mixed.to_dtype(sublayer_dtype(&mixed))?;
        let out = if matches!(self.mixer, mixer::Mixer::Attention) {
            self.wide_attention(&mixed, mask, index_pos)?
        } else {
            let m = self.mixer.forward(&mixed, index_pos)?;
            self.attn_output.forward(&m)?
        };
        let inj = inj.expect("the token mixer's mixer writes back");
        r = combine(&r, &out, &inj)?;
        let (mixed, inj) = {
            let hc = self.hyper.as_ref().expect("a wide layer has its mixers");
            hc.ffn.mix(&r)?
        };
        let mixed = mixed.to_dtype(sublayer_dtype(&mixed))?;
        let out = match self.moe.as_ref() {
            Some(moe) => moe.forward_rows(
                &mixed,
                &mixed,
                self.embedding_length_for_moe,
                self.ffn_norm_eps,
            )?,
            None => self.forward_ffn(&mixed)?,
        };
        combine(
            &r,
            &out,
            &inj.expect("the feed-forward's mixer writes back"),
        )
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
