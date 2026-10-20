//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

/// Turned on by the non-finite-logits gate, and off again by the first report.
///
/// A prefill that ends in NaN says nothing about WHERE the NaN was born, and scanning every
/// layer of every request to find out would tax the path that works. So the failure arms the
/// instrument: the request that fails costs nothing extra, and the NEXT one names the layer.
/// The normal path pays one relaxed atomic load per layer.
pub static SCAN_FOR_NON_FINITE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Report the first layer whose output is not finite, once.
fn note_if_non_finite(stage: &str, x: &Tensor) {
    use std::sync::atomic::Ordering::Relaxed;
    if !SCAN_FOR_NON_FINITE.load(Relaxed) {
        return;
    }
    // One number for the whole tensor: a NaN anywhere poisons the sum, which is all this
    // needs to answer. Read on the host, so it costs a device sync - paid only when armed.
    let sum = x
        .to_dtype(crate::tensor::DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.sum(0))
        .and_then(|t| t.to_scalar_f32());
    match sum {
        Ok(v) if !v.is_finite() => {
            tracing::error!("non-finite hidden state first seen at {stage} (sum {v})");
            SCAN_FOR_NON_FINITE.store(false, Relaxed);
        }
        Ok(_) => {}
        Err(e) => tracing::warn!("non-finite scan could not read the hidden state: {e}"),
    }
}

impl GenericHeteroTransformer {
    /// Read-only view of the layers, which is how anything outside this module reaches a
    /// layer's KV cache after a forward.
    ///
    /// The field stays private: this hands out shared references, so a caller can inspect
    /// what a run produced without being able to reorder the layers or swap a cache out
    /// from under the plan that placed it.
    pub fn layers(&self) -> &[GenericTransformerLayer] {
        &self.layers
    }

    /// Adapter names live on the model right now, in the order they were applied.
    pub fn adapters(&self) -> &[String] {
        &self.adapters
    }

    /// Replace the attached adapter set, in place, without reloading the checkpoint.
    ///
    /// An empty list detaches everything and returns every projection to the file it was read
    /// from. Mutation stays inside this type on purpose: `layers` hands out shared references
    /// so that no caller can reorder the layers or swap a cache from under the plan that
    /// placed them, and an adapter walk needs none of that.
    ///
    /// Applied all-or-nothing. A file that resolves but matches nothing leaves the model on
    /// its base weights and says so, because an adapter on some projections and not others is
    /// worse than one that did not apply.
    pub fn set_adapters(
        &mut self,
        wanted: &[(String, f32)],
    ) -> Result<crate::inference::load::lora::AdapterReport> {
        use crate::inference::load::lora::{transformer_keys, AdapterReport, LoraFile};
        use crate::tensor::Error;

        for layer in self.layers.iter_mut() {
            let mut shards = std::mem::take(&mut layer.tp_shards);
            for part in std::iter::once(&mut *layer).chain(shards.iter_mut()) {
                for p in [
                    part.attn_q.as_mut(),
                    part.attn_k.as_mut(),
                    part.attn_v.as_mut(),
                    part.attn_qkv.as_mut(),
                    part.ffn_gate.as_mut(),
                ]
                .into_iter()
                .flatten()
                {
                    p.clear_lora();
                }
                part.attn_output.clear_lora();
                part.ffn_up.clear_lora();
                part.ffn_down.clear_lora();
            }
            layer.tp_shards = shards;
        }
        self.adapters.clear();
        if wanted.is_empty() {
            return Ok(AdapterReport::default());
        }

        // Read on the host, then move each delta to the card its projection sits on: a split
        // model holds layer 0 and layer 30 on different devices.
        let mut files = Vec::with_capacity(wanted.len());
        for (name, strength) in wanted {
            let path = crate::inference::load::lora::resolve(name).map_err(Error)?;
            let file = LoraFile::load(&path.to_string_lossy(), &Device::Cpu)?;
            if file.is_empty() {
                return Err(Error(format!("adapter {name} holds no usable pair")));
            }
            files.push((name.clone(), *strength, file));
        }

        let mut report = AdapterReport::default();
        let mut matched_entries = 0usize;
        for (name, strength, file) in &files {
            let mut hits = 0usize;
            for (i, layer) in self.layers.iter_mut().enumerate() {
                // Fused QKV is deliberately absent: an adapter names q, k and v separately and
                // splitting a fused weight to take them is a different piece of work. A
                // projection cut by rows across a layer's shares takes its slice of the delta's
                // outputs, one cut by columns its slice of the delta's inputs.
                const SITES: [(&str, bool); 7] = [
                    ("self_attn.q_proj", true),
                    ("self_attn.k_proj", true),
                    ("self_attn.v_proj", true),
                    ("self_attn.o_proj", false),
                    ("mlp.gate_proj", true),
                    ("mlp.up_proj", true),
                    ("mlp.down_proj", false),
                ];
                let ranks = 1 + layer.tp_shards.len();
                for (site, (projection, by_rows)) in SITES.iter().enumerate() {
                    let mut delta = None;
                    for key in transformer_keys(i, projection) {
                        if let Some(d) = file.delta_for_key(&key, *strength)? {
                            delta = Some(d);
                            break;
                        }
                    }
                    let Some(delta) = delta else { continue };
                    let mut shards = std::mem::take(&mut layer.tp_shards);
                    let parts = std::iter::once(&mut *layer).chain(shards.iter_mut());
                    let mut applied = false;
                    for (rank, part) in parts.enumerate() {
                        let slot = match site {
                            0 => part.attn_q.as_mut(),
                            1 => part.attn_k.as_mut(),
                            2 => part.attn_v.as_mut(),
                            3 => Some(&mut part.attn_output),
                            4 => part.ffn_gate.as_mut(),
                            5 => Some(&mut part.ffn_up),
                            _ => Some(&mut part.ffn_down),
                        };
                        let Some(weight) = slot else { continue };
                        let (down, up) = match (ranks, by_rows) {
                            (1, _) => (delta.down.clone(), delta.up.clone()),
                            (_, true) => {
                                let per = delta.up.dim(1)? / ranks;
                                (
                                    delta.down.clone(),
                                    delta.up.narrow(1, rank * per, per)?.contiguous()?,
                                )
                            }
                            (_, false) => {
                                let per = delta.down.dim(0)? / ranks;
                                (
                                    delta.down.narrow(0, rank * per, per)?.contiguous()?,
                                    delta.up.clone(),
                                )
                            }
                        };
                        let device = weight.device();
                        weight.add_lora(crate::tensor::lora::LoraDelta {
                            down: down.to_device(&device)?,
                            up: up.to_device(&device)?,
                            scale: delta.scale,
                        })?;
                        applied = true;
                    }
                    layer.tp_shards = shards;
                    hits += usize::from(applied);
                }
            }
            if hits == 0 {
                let held = self.adapters.clone();
                self.set_adapters(&[])?;
                let _ = held;
                return Err(Error(format!(
                    "adapter {name} matched no projection on this checkpoint"
                )));
            }
            matched_entries += hits;
            report.attached.push(name.clone());
            self.adapters.push(name.clone());
        }
        report.projections = matched_entries;
        report.unmatched = files
            .iter()
            .map(|(_, _, f)| f.entry_count())
            .sum::<usize>()
            .saturating_sub(matched_entries);
        Ok(report)
    }

    /// Build a [seq_q, seq_kv] causal mask where row `i` represents token
    /// at absolute position `index_pos + i` and column `j` represents KV
    /// position `j` (which spans the full history including the new
    /// tokens). For prefill (index_pos=0), seq_kv == seq_q. For
    /// session-resume / PLD verify / spec verify, seq_kv > seq_q.
    pub(super) fn make_mask(
        &mut self,
        seq_q: usize,
        index_pos: usize,
        cuda_idx: Option<usize>,
    ) -> Result<Tensor> {
        self.make_mask_reaching(seq_q, index_pos, cuda_idx, self.sliding_window)
    }

    /// The causal mask of one layer of a period-classified model: a windowed layer sees
    /// its last `window` keys, its own position included, as the reference's sliding
    /// window does.
    pub(super) fn make_layer_mask(
        &mut self,
        seq_q: usize,
        index_pos: usize,
        cuda_idx: Option<usize>,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let reach = window.map(|w| w.saturating_sub(1));
        self.make_mask_reaching(seq_q, index_pos, cuda_idx, reach)
    }

    /// Causal mask that also hides a key more than `sw` positions behind its query.
    fn make_mask_reaching(
        &mut self,
        seq_q: usize,
        index_pos: usize,
        cuda_idx: Option<usize>,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let seq_kv = index_pos + seq_q;
        let cache_key = (seq_q * 65536 + seq_kv, cuda_idx, sw);
        if !self.mask_cache.contains_key(&cache_key) {
            let dev = cuda_idx
                .and_then(|idx| self.cuda_devices.get(&idx))
                .unwrap_or(&Device::Cpu);
            let data: Vec<u8> = (0..seq_q)
                .flat_map(|i| {
                    (0..seq_kv).map(move |j| {
                        let abs_i = i + index_pos;
                        u8::from(j > abs_i || sw.is_some_and(|w| j + w < abs_i))
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&data, (seq_q, seq_kv), dev)?;
            self.mask_cache.insert(cache_key, mask);
        }
        Ok(self.mask_cache[&cache_key].clone())
    }

    /// Real per-device layer distribution for the /api/ps topology view:
    /// consecutive layers on the same device grouped into
    /// `(device_type, device_id, layer_start, layer_end)` segments (ranges
    /// inclusive). Reads the ACTUAL `layer_devs` set at load time from the
    /// HeteroPlan, so a GPU0/GPU1/CPU split surfaces as distinct segments
    /// instead of every layer collapsed onto the primary device (the
    /// long-standing topology bug where the GUI showed the whole model on
    /// one GPU regardless of the real split).
    pub fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        let mut out: Vec<(String, usize, u32, u32)> = Vec::new();
        for (i, ld) in self.layer_devs.iter().enumerate() {
            let (dtype, did) = match ld {
                LayerDevice::Cuda(idx) => ("CUDA", *idx),
                LayerDevice::Cpu => ("CPU", 0usize),
            };
            let i = i as u32;
            match out.last_mut() {
                // Extend the current run when the same device continues.
                Some(last) if last.0 == dtype && last.1 == did => last.3 = i,
                _ => out.push((dtype.to_string(), did, i, i)),
            }
        }
        out
    }

    /// Forward pass: embed -> layers (CUDA+CPU segments) -> norm -> logits.
    /// Standard forward: returns logits for the LAST position only
    /// (shape `[batch, vocab]`). Equivalent to `forward_inner(x, pos, false)`.
    pub fn forward(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, index_pos, /*all_positions=*/ false)
    }

    /// Whether a prompt of `seq_len` rows in chunks of `chunk` goes through the layers better
    /// layer by layer: it takes several chunks, and some layer's experts come from the host in
    /// full for a batch that wide, which a chunk at a time repeats for every chunk.
    pub fn layer_major_pays(&self, seq_len: usize, chunk: usize) -> bool {
        seq_len > chunk.max(1)
            && self.config.shared_kv_layers == 0
            && self.layers.iter().any(|l| {
                l.moe.as_ref().is_some_and(|m| {
                    m.tier
                        .as_ref()
                        .is_some_and(|t| t.stream_pays(seq_len, m.n_experts_used))
                })
            })
    }

    /// A prompt `ids` `[1, seq_len]` from `index_pos` through the layers one layer at a time:
    /// every chunk of `chunk` rows goes through a layer before any goes through the next, so a
    /// layer whose experts live on the host has them brought to the card once for the whole
    /// prompt, where a chunk at a time brings them once per chunk. The chunks' residuals are
    /// kept between layers; the logits of the last row come back, as from `forward`.
    pub fn forward_prefill_layer_major(
        &mut self,
        ids: &Tensor,
        index_pos: usize,
        chunk: usize,
    ) -> Result<Tensor> {
        let (_, seq_len) = ids.dims2()?;
        let chunk = chunk.max(1);
        self.spec_clear();
        // Every chunk's embedding first, in order: the n-gram history follows the tokens.
        let mut spans = Vec::new();
        let mut hidden: Vec<Option<Tensor>> = Vec::new();
        let mut ple: Vec<Option<Vec<Tensor>>> = Vec::new();
        let mut ngram_rows: Vec<Option<Tensor>> = Vec::new();
        let mut off = 0;
        while off < seq_len {
            let n = chunk.min(seq_len - off);
            let (h, p) = self.embed_stage(&ids.narrow(1, off, n)?, index_pos + off)?;
            ngram_rows.push(self.ngram.as_ref().and_then(|g| g.pending.clone()));
            hidden.push(Some(h));
            ple.push(p);
            spans.push((off, n));
            off += n;
        }
        let mut on_cuda = true;
        for layer in 0..self.layers.len() {
            let tier = self.layers[layer].moe.as_ref().and_then(|m| m.tier.clone());
            if let Some(tier) = &tier {
                tier.hold_stacks(true)?;
            }
            let mut outcome = Ok(());
            for (c, &(off, n)) in spans.iter().enumerate() {
                if let (Some(ng), Some(rows)) = (self.ngram.as_mut(), ngram_rows[c].clone()) {
                    ng.pending = Some(rows);
                }
                let h = hidden[c]
                    .take()
                    .expect("a chunk's residual is held between layers");
                match self.run_layers_in(h, ple[c].clone(), index_pos + off, n, layer, layer + 1) {
                    Ok((h, cuda)) => {
                        if index_pos + off == 0 {
                            if let Some(dir) = super::draft_head::probe_dir() {
                                super::draft_head::dump_rows(dir, &format!("layer{layer}"), &h);
                            }
                        }
                        hidden[c] = Some(h);
                        on_cuda = cuda;
                    }
                    Err(e) => {
                        outcome = Err(e);
                        break;
                    }
                }
            }
            if let Some(tier) = &tier {
                tier.hold_stacks(false)?;
            }
            outcome?;
        }
        // The draft head reads the prompt chunk by chunk, in order, from the last layer's rows.
        if self.capture_feature && self.draft.is_some() {
            // The fit probe reads the prompt's first rows whatever the chunking: the chunks
            // within them are gathered, then read once.
            let mut probe_rows: Vec<Tensor> = Vec::new();
            let mut probe_ids: Vec<u32> = Vec::new();
            let span = PROMPT_FIT_SPAN;
            for (c, &(off, n)) in spans.iter().enumerate() {
                let chunk_ids = ids
                    .narrow(1, off, n)?
                    .flatten_all()?
                    .to_dtype(crate::tensor::DType::U32)?
                    .to_vec1::<u32>()?;
                let rows = hidden[c]
                    .as_ref()
                    .expect("a chunk's residual is held after the last layer");
                if index_pos + off < span {
                    probe_rows.push(rows.clone());
                    probe_ids.extend_from_slice(&chunk_ids);
                }
                self.draft_ingest(&chunk_ids, rows, index_pos + off)?;
            }
            if index_pos == 0 && !probe_rows.is_empty() {
                let rows = Tensor::cat(&probe_rows.iter().collect::<Vec<_>>(), 1)?;
                let n = probe_ids.len().min(span);
                self.prompt_fit_probe(
                    &probe_ids[..n],
                    &rows.narrow(1, 0, n.saturating_sub(1).max(1))?,
                )?;
            }
        }
        let last = hidden
            .pop()
            .flatten()
            .ok_or_else(|| crate::tensor::Error::msg("empty prefill".to_string()))?;
        let rows = spans.last().map_or(1, |s| s.1);
        self.output_stage(last, on_cuda, rows, false)
    }

    /// EAGLE: enable/disable stashing the pre-final-norm feature on forward.
    pub fn set_capture_feature(&mut self, on: bool) {
        self.capture_feature = on;
    }
    /// EAGLE: take the feature stashed by the last forward (if capture was on).
    /// Shape [b, hidden] (last-token decode) or [b, seq, hidden] (all_positions verify).
    pub fn take_last_feature(&mut self) -> Option<Tensor> {
        self.last_feature.take()
    }

    /// Embedding: prefill `input_ids` `[1, seq]` and return the LAST token's final-layer
    /// hidden state `[hidden]` (last-token pooling). Reuses the EAGLE feature hook so ANY
    /// llama-arch GGUF works - no separate encoder. KV is reset before and after so the
    /// call is stateless (doesn't disturb a concurrent chat session's cache). Caller
    /// L2-normalizes. Routes through the same loaded model as rerank.
    pub fn embed_last_hidden(&mut self, input_ids: &Tensor) -> Result<Vec<f32>> {
        self.trim_kv(0);
        self.set_capture_feature(true);
        let res = self.forward(input_ids, 0);
        self.set_capture_feature(false);
        self.trim_kv(0);
        res?; // logits ignored; the feature is what we want
        let feat = self
            .take_last_feature()
            .ok_or_else(|| crate::tensor::Error::msg("embed: no captured feature"))?;
        // Apply the model's FINAL norm (the capture is pre-norm). The learned RMSNorm
        // weights de-anisotropize the raw hidden - without it all vectors collapse to
        // ~0.99 cosine and lose discrimination. `output_norm` lives on CPU, so bring the
        // feature to CPU first (a GPU feature + CPU weight yields NaN). Fall back to the
        // pre-norm feature on any mismatch rather than fail the request.
        let feat_cpu = feat.to_device(&crate::tensor::Device::Cpu).unwrap_or(feat);
        let normed = match self.output_norm.forward(&feat_cpu) {
            Ok(n) => n,
            Err(_) => feat_cpu,
        };
        normed
            .flatten_all()?
            .to_dtype(crate::tensor::DType::F32)?
            .to_vec1::<f32>()
    }

    /// EAGLE: target token-embedding lookup (the head's `embed(token)` input).
    /// Moves `ids` to the embedding table's device (CPU for big vocabularies).
    pub fn eagle_embed(&self, ids: &Tensor) -> Result<Tensor> {
        let dev = self.embeddings.embeddings().device().clone();
        self.embeddings.forward(&ids.to_device(&dev)?)
    }

    /// EAGLE: map a PRE-final-norm feature `[n, hidden]` through the target's
    /// final norm + lm-head to logits `[n, vocab]`. This IS the hot draft-sampling
    /// path (called kx per spec cycle), so prefer the GPU quantized lm-head + fused
    /// rms_norm - keeping the feature on-device. The CPU `output_proj` is a fallback
    /// (CPU models / when the cuda lm-head isn't built); a CPU 151936x4096 matmul
    /// per draft is ~7x slower than plain decode, so it must NOT be used on GPU.
    /// The dtype the residual stream is carried in on `dev`.
    ///
    /// THIS IS A DEFECT EXPRESSED AS A FUNCTION, not a settled contract. The
    /// precision depends on WHERE a layer was placed, so a model split across a
    /// card and the host computes some layers in half precision and others in
    /// single, and the answer depends on the split point. Measured: the same
    /// layer differs by ~1e-3 between the two devices, a thousand times more
    /// than the ~1e-6 that reassociation alone would give, and enough for greedy
    /// decoding to pick a different token when two candidates are close.
    ///
    /// The reference carries activations in F32 on every backend
    /// (`ggml_mul_mat` allocates F32 whatever the device) and quantizes them to
    /// Q8_1 at each matmul instead of storing them in half precision, so moving
    /// a layer between devices leaves the stream's type untouched. Adopting that
    /// means teaching the quantized GPU GEMMs to read F32 activations - the
    /// comment at the call site records that they read garbage today - which is
    /// why this returns the device-dependent answer for now instead of a
    /// uniform one.
    fn residual_dtype(dev: &Device) -> DType {
        if dev.is_cpu() {
            DType::F32
        } else {
            DType::F16
        }
    }

    /// The model's final norm, evaluated on whatever device `x` lives on.
    ///
    /// The GPU head used to call the fused RMS kernel unconditionally, which is
    /// the wrong norm for the phi2 family (full LayerNorm, mean subtracted, bias
    /// added). Excluding those models from the GPU head fixed the answer and cost
    /// most of their throughput, because it sent the 51200-wide projection to the
    /// CPU as well. The norm is 2048 values; the projection is a hundred million
    /// multiply-adds - so normalise correctly HERE and keep the projection on the
    /// card.
    fn final_norm_on_device(&self, x: &Tensor) -> Result<Tensor> {
        use crate::inference::generic_transformer::config::WeightedNorm;
        match &self.output_norm {
            WeightedNorm::Rms(_) => {
                let w = self
                    .output_norm_cuda_weight
                    .as_ref()
                    .filter(|w| w.device().same_device(&x.device()))
                    .cloned()
                    .map(Ok)
                    .unwrap_or_else(|| self.output_norm.weight().to_device(&x.device()))?;
                crate::tensor::ops::rms_norm(x, &w, self.config.rms_norm_eps as f32)
            }
            WeightedNorm::Layer(ln) => {
                let dev = x.device();
                let w = ln.weight().to_device(&dev)?.to_dtype(x.dtype())?;
                let mean = x.mean_keepdim(crate::tensor::D::Minus1)?;
                let centered = x.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(crate::tensor::D::Minus1)?;
                let normed = centered.broadcast_div(&(var + ln.eps())?.sqrt()?)?;
                let scaled = normed.broadcast_mul(&w)?;
                match ln.bias() {
                    Some(b) => scaled.broadcast_add(&b.to_device(&dev)?.to_dtype(x.dtype())?),
                    None => Ok(scaled),
                }
            }
            // Post-norm-only architectures pass the residual through unchanged.
            _ => self.output_norm.forward(x),
        }
    }

    /// Adds the per-vocabulary logit bias when the architecture carries one.
    /// EVERY site that produces logits must route through this - the engine has
    /// several, and one that skips it is wrong without failing.
    /// Final norm and output projection for rows computed on a card: both run where the
    /// projection lives, the card-resident quantised head when there is one.
    fn head_on_device(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(crate::tensor::DType::F32)?;
        #[cfg(feature = "gpu")]
        if let (Some(proj), Some(dev)) = (
            self.output_proj_cuda_qmm.as_ref(),
            self.output_proj_dev.as_ref(),
        ) {
            let x = x.to_device(dev)?;
            let f = self.final_norm_on_device(&x)?;
            return self.apply_output_bias(proj.forward(&f)?);
        }
        let f = self.final_norm_on_device(&x)?;
        self.apply_output_bias(self.output_proj.forward(&f)?)
    }

    pub(crate) fn apply_output_bias(&self, logits: Tensor) -> Result<Tensor> {
        let Some(b) = self.output_bias.as_ref() else {
            return Ok(logits);
        };
        let b = b.to_device(&logits.device())?.to_dtype(logits.dtype())?;
        logits.broadcast_add(&b)
    }

    /// The device batched_paged_decode runs on (weights/embeddings device)  -
    /// callers size the paged KV stores on it.
    /// Device where the transformer layers run (where paged KV stores must live).
    /// For GPU models the embedding TABLE often stays on CPU (large vocab) while
    /// the layers are on GPU - so this returns the first layer's weight device,
    /// not the embeddings device.
    pub fn compute_device(&self) -> Device {
        self.layers
            .first()
            .map(|l| l.attn_norm.weight().device().clone())
            .unwrap_or_else(|| self.embeddings.embeddings().device().clone())
    }

    /// Geometry for sizing paged KV stores: (n_kv_head, head_dim, n_layers, vocab).
    pub fn paged_geometry(&self) -> (usize, usize, usize, usize) {
        (
            self.config.n_kv_head,
            self.config.head_dim,
            self.layers.len(),
            self.config.vocab_size,
        )
    }

    /// Why this model can't use the continuous-batch paged path (None = eligible).
    /// Shared by `batched_paged_decode`/`paged_prefill_seq` (the gate) and the
    /// serving layer (to decide whether to route a model through ContinuousServer).
    pub fn cb_unsupported_reason(&self) -> Option<&'static str> {
        let c = &self.config;
        if self.is_recurrent() {
            return Some("recurrent mixer");
        }
        // The batched server prefills token ids; a prompt with embeddings spliced in is not.
        #[cfg(feature = "audio")]
        if self.spliced_audio.is_some() {
            return Some("speech tower");
        }
        if c.embed_scale.is_some() {
            return Some("embed_scale");
        }
        if c.logit_scale.is_some() {
            return Some("logit_scale");
        }
        if c.attention_scale.is_some() {
            return Some("attention_scale");
        }
        if c.residual_scale.is_some() {
            return Some("residual_scale");
        }
        if c.final_logit_softcapping.is_some() {
            return Some("logit softcapping");
        }
        if c.yarn.is_some() {
            return Some("yarn");
        }
        if c.rope_dim.unwrap_or(0) != 0 && c.rope_dim != Some(c.head_dim) {
            return Some("partial rope");
        }
        for l in &self.layers {
            let split = l.attn_q.is_some() && l.attn_k.is_some() && l.attn_v.is_some();
            if !split && l.attn_qkv.is_none() {
                return Some("no Q/K/V");
            }
            if l.ffn_gate.is_none() && c.flags.is_phi2_simple_ffn {
                return Some("Phi2 gate-less FFN");
            }
            if l.ffn_norm.is_none() {
                return Some("no ffn_norm (parallel attn?)");
            }
            // OLMo2 post-norm-only: Identity pre-norm needs the regular forward (the
            // CB/paged executor mis-applies its ones-weight); keep OLMo2 off CB.
            // SmolLM3 NoPE: per-layer RoPE skip isn't wired into the paged/CB rope
            // path - keep it on the serial forward (where apply_rotary_emb gates it).
            if l.no_rope {
                return Some("NoPE per-layer rope (SmolLM3)");
            }
            if matches!(l.attn_norm, WeightedNorm::Identity(_)) {
                return Some("post-norm-only (OLMo2)");
            }
            if l.attn_q_norm.is_some() != l.attn_k_norm.is_some() {
                return Some("asymmetric QK-norm");
            }
            if l.moe.is_some() {
                return Some("MoE");
            }
            if l.attn_sinks.is_some() {
                return Some("attention sinks");
            }
            if l.ple_proj.is_some() {
                return Some("PLE");
            }
        }
        None
    }

    /// True if this model can be served through the continuous-batch worker.
    pub fn cb_eligible(&self) -> bool {
        self.cb_unsupported_reason().is_none()
    }

    pub fn rope_base(&self) -> f32 {
        self.config.rope_freq_base
    }

    /// Whether the residual is wide: parallel streams behind gated mixers.
    pub fn is_wide(&self) -> bool {
        self.output_hyper.is_some()
    }

    /// Width of the residual a layer reads: the model width, times the streams of a wide one.
    pub fn stream_width(&self) -> usize {
        self.config.embedding_length * self.config.hyper.map_or(1, |h| h.streams)
    }

    /// The model width: the residual stream and the embedding.
    pub fn hidden_size(&self) -> usize {
        self.config.embedding_length
    }

    /// Widest FFN intermediate dim across all layers (dense `intermediate_size`
    /// or per-expert `expert_ffn_dim`) - the activation-peak driver used to size
    /// the prefill chunk to fit L3.
    pub fn widest_ffn(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.flags.intermediate_size.max(l.flags.expert_ffn_dim))
            .max()
            .unwrap_or(0)
    }

    /// What the prefill asks of a card per token and the chunk it runs best at.
    ///
    /// A wide residual keeps `streams x dim` f32 per token in the residual and the product a
    /// combine adds, its mixers' temporaries are bounded to a block of rows, and the chunk it
    /// wants is the one that gives every routed expert a full tile of the grouped matmul (see
    /// `hyper::prefill_footprint`).
    pub fn prefill_shape(&self) -> crate::inference::engine::model_backend::PrefillShape {
        let mut shape = self.prefill_shape_with(
            self.layers
                .iter()
                .any(|l| l.moe.as_ref().is_some_and(|m| m.tier.is_some())),
        );
        // A model on cards is sized from the cards it uses, and a model with layers on the host
        // from every card, as before.
        if self
            .layer_devs
            .iter()
            .all(|d| matches!(d, LayerDevice::Cuda(_)))
        {
            shape.cards = self
                .layer_devs
                .iter()
                .filter_map(|d| match d {
                    LayerDevice::Cuda(i) => Some(1u64 << (*i).min(63)),
                    LayerDevice::Cpu => None,
                })
                .fold(0, |a, b| a | b);
        }
        shape
    }

    /// [`Self::prefill_shape`] for a model whose experts do, or do not, run on the card for a wide
    /// batch: they then keep activations live per routed pair, and the widest chunk is the one that
    /// shares the experts' copies over the most rows.
    pub fn prefill_shape_with(
        &self,
        tiered: bool,
    ) -> crate::inference::engine::model_backend::PrefillShape {
        use crate::inference::engine::model_backend::PrefillShape;
        let plain = PrefillShape::from_widest_ffn(self.widest_ffn());
        let experts = self
            .layers
            .iter()
            .find_map(|l| l.moe.as_ref())
            .map(|m| (m.n_experts, m.n_experts_used));
        let expert_ffn = self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref())
            .map(|m| m.expert_ffn_dim)
            .max()
            .unwrap_or(0);
        let base = match self.config.hyper {
            None => plain,
            Some(h) => {
                let fp = super::hyper::prefill_footprint(
                    h.streams,
                    self.config.embedding_length,
                    experts,
                    expert_ffn,
                );
                PrefillShape {
                    bytes_per_token: fp.bytes_per_token.max(plain.bytes_per_token),
                    fixed_bytes: fp.fixed_bytes,
                    target_chunk: fp.target_chunk,
                    share: 2,
                    cards: 0,
                }
            }
        };
        let (true, Some((_, used))) = (tiered, experts) else {
            return base;
        };
        // A wide model's already counts its experts' activations.
        PrefillShape {
            bytes_per_token: base.bytes_per_token
                + if self.config.hyper.is_some() {
                    0
                } else {
                    super::expert_tier::prefill_bytes_per_token(
                        used,
                        self.config.embedding_length,
                        expert_ffn,
                    )
                },
            fixed_bytes: base.fixed_bytes,
            target_chunk: super::hyper::PLANNED_CHUNK_ROWS,
            share: 2,
            cards: 0,
        }
    }

    /// Embed `tokens` -> `[b, hidden]` on the layer-compute device in F16 (the
    /// eager, non-capturable input to `decode_layers_gpu`).
    pub fn embed_batch(&self, tokens: &[u32]) -> Result<Tensor> {
        use crate::tensor::DType;
        let edev = self.embeddings.embeddings().device().clone();
        let dev = self.compute_device();
        let ids = Tensor::from_vec(tokens.to_vec(), &[tokens.len(), 1][..], &edev)?;
        let mut x = self
            .embeddings
            .forward(&ids)?
            .reshape(&[tokens.len(), self.config.embedding_length][..])?
            .to_dtype(DType::F16)?;
        if !x.device().same_device(&dev) {
            x = x.to_device(&dev)?;
        }
        Ok(x)
    }

    /// Continuous-batching batched-paged DECODE forward (one new token per
    /// sequence). `tokens[b]` is sequence b's input token at `positions[b]`; its
    /// new K/V is written to every layer's paged store at `slots[b]`, and it
    /// attends over `block_tables[b]` (`context_lens[b]` tokens). Returns logits
    /// `[B, vocab]`.
    ///
    /// FIRST CUT - dense CPU path only (split Q/K/V, full RoPE, SwiGLU, serial
    /// attention, no QK-norm / PLE / MoE / special scales / YaRN). Anything else
    /// -> `Err` so the caller falls back to the serial engine. The batched
    /// projections (one GEMM over B rows) are the throughput win; RoPE + paged
    /// attention run per-sequence in f32 (the slow parts a fused paged-attention
    /// kernel replaces later). Bit-faithful to the per-seq decode -> batch-invariant
    /// by construction (each sequence reads only its own block table). `stores`
    /// has one [`crate::inference::cache::paged_kv::PagedKvStore`] per layer (shared block tables).
    pub fn batched_paged_decode(
        &self,
        tokens: &[u32],
        positions: &[usize],
        slots: &[usize],
        block_tables: &[Vec<u32>],
        context_lens: &[usize],
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        use crate::tensor::Tensor;
        let c = &self.config;
        // -- variant gate: only the plain dense path is supported here. --
        if let Some(why) = self.cb_unsupported_reason() {
            return Err(crate::tensor::Error::msg(format!(
                "batched_paged_decode: unsupported - {why}"
            )));
        }
        if stores.len() != self.layers.len() {
            return Err(crate::tensor::Error::msg(
                "batched_paged_decode: store/layer count mismatch",
            ));
        }
        let b = tokens.len();
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let base = c.rope_freq_base;
        // Device-agnostic: run on the weights' device (CPU or CUDA). The embedding
        // table may live on CPU even for a GPU model (large vocab) -> x is moved to
        // the layer-compute device right after the lookup.
        let edev = self.embeddings.embeddings().device().clone();
        let dev = self.compute_device(); // layer (matmul) device - may differ from edev
                                         // Compute dtype: F32 on CPU (matches the scalar reference path bit-for-bit),
                                         // F16 on GPU (matches forward()'s on-device kernels; F32 activations through
                                         // GPU quantized GEMMs read garbage).
        let cdt = Self::residual_dtype(&dev);

        // embed on the embeddings device, then move activations to the layer device.
        let ids = Tensor::from_vec(tokens.to_vec(), &[b, 1][..], &edev)?;
        let mut x = self
            .embeddings
            .forward(&ids)?
            .reshape(&[b, c.embedding_length][..])?
            .to_dtype(cdt)?;
        if !x.device().same_device(&dev) {
            x = x.to_device(&dev)?;
        }
        // RoPE cos/sin depend only on positions - compute ONCE, reuse every layer.
        let (rcos, rsin) =
            crate::inference::cache::paged_attention::rope_cos_sin(positions, hd, base, &dev, cdt)?;
        // Ragged attention mask (GPU path, any B) depends only on lens - compute ONCE.
        let ragged_mask = if !dev.is_cpu() {
            let max_ctx = *context_lens.iter().max().unwrap();
            Some(crate::inference::cache::paged_attention::ragged_mask(
                context_lens,
                max_ctx,
                &dev,
                cdt,
            )?)
        } else {
            None
        };
        // Per-seq kv position (= ctx-1) for the fused flash-decode attention kernel
        // (LOKEN_CB_FLASH): one device i32[b], read at kernel-exec time -> no
        // padding-compute waste (each seq attends over its real length), graph-safe.
        // The fused flash-decode is about 10% faster at high batch and is the only
        // capture-safe form, but its float reassociation differs from the cuBLAS chain's:
        // bit-identical on qwen2/qwen3, and on a near-tied-logit model like mistral-nemo or
        // deepcoder it flips a greedy token that was a near-tie. The default keeps strict
        // bit-identity, which is what the parity oracles are written against.
        let cb_flash = false;
        let pos_dev = if cb_flash {
            let pv: Vec<i32> = context_lens.iter().map(|&c| (c as i32) - 1).collect();
            Some(Tensor::from_vec(pv, &[b][..], &dev)?)
        } else {
            None
        };
        // The fully capture-safe paged path: KV write and paged flash-decode both index the
        // store through DEVICE buffers read at kernel-exec time, where index_select and
        // scatter bake host-read indices at capture. Written for a CUDA-graph decode that is
        // not built yet, so nothing selects it.
        let cb_paged = false;
        let paged_dev = if cb_paged {
            let slot_dev = Tensor::from_vec(
                slots.iter().map(|&s| s as i32).collect::<Vec<_>>(),
                &[b][..],
                &dev,
            )?;
            let seq_lens_dev = Tensor::from_vec(
                context_lens.iter().map(|&c| c as i32).collect::<Vec<_>>(),
                &[b][..],
                &dev,
            )?;
            let max_blocks = block_tables.iter().map(|bt| bt.len()).max().unwrap();
            let mut btf = vec![0i32; b * max_blocks];
            for (bi, bt) in block_tables.iter().enumerate() {
                for (j, &blk) in bt.iter().enumerate() {
                    btf[bi * max_blocks + j] = blk as i32;
                }
            }
            let block_table_dev = Tensor::from_vec(btf, &[b, max_blocks][..], &dev)?;
            Some((slot_dev, seq_lens_dev, block_table_dev, max_blocks))
        } else {
            None
        };

        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let hn = layer.attn_norm.forward(&x)?.to_dtype(cdt)?;
            let (qd, kvd) = (nh * hd, nkv * hd);
            let (q, k, v) = if let Some(qkvw) = layer.attn_qkv.as_ref() {
                // fused QKV: one GEMM, then split [q_dim | kv_dim | kv_dim].
                let mut qkv = qkvw.forward(&hn)?;
                if let Some(bias) = layer.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(&bias.to_dtype(qkv.dtype())?)?;
                }
                (
                    qkv.narrow(1, 0, qd)?,
                    qkv.narrow(1, qd, kvd)?,
                    qkv.narrow(1, qd + kvd, kvd)?,
                )
            } else {
                let mut q = layer.attn_q.as_ref().unwrap().forward(&hn)?;
                let mut k = layer.attn_k.as_ref().unwrap().forward(&hn)?;
                let mut v = layer.attn_v.as_ref().unwrap().forward(&hn)?;
                if let Some(bias) = layer.attn_q_bias.as_ref() {
                    q = q.broadcast_add(&bias.to_dtype(q.dtype())?)?;
                }
                if let Some(bias) = layer.attn_k_bias.as_ref() {
                    k = k.broadcast_add(&bias.to_dtype(k.dtype())?)?;
                }
                if let Some(bias) = layer.attn_v_bias.as_ref() {
                    v = v.broadcast_add(&bias.to_dtype(v.dtype())?)?;
                }
                (q, k, v)
            };
            // RoPE q,k on-device, ONE batched write of all B tokens' K/V into this
            // layer's store, then per-seq paged GQA attention.
            // Per-head QK-norm (qwen3) on the [B,heads,hd] view before RoPE - same
            // RmsNorm over head_dim that forward() applies. No-op when absent.
            // OLMo2 full-dim QK-norm: norm the FLAT q/k (whole projection) before the
            // head reshape; Gemma3/Qwen3 norm per-head after. Detect via weight length.
            let qk_full = layer.qk_norm_is_full();
            let qf = q.to_dtype(cdt)?;
            let qf = if qk_full {
                if let Some(qn) = layer.attn_q_norm.as_ref() {
                    qn.forward(&qf.contiguous()?)?
                } else {
                    qf
                }
            } else {
                qf
            };
            let qh = qf.reshape(&[b, nh, hd][..])?;
            let qh = if !qk_full {
                if let Some(qn) = layer.attn_q_norm.as_ref() {
                    qn.forward(&qh.contiguous()?)?
                } else {
                    qh
                }
            } else {
                qh
            };
            let kf = k.to_dtype(cdt)?;
            let kf = if qk_full {
                if let Some(kn) = layer.attn_k_norm.as_ref() {
                    kn.forward(&kf.contiguous()?)?
                } else {
                    kf
                }
            } else {
                kf
            };
            let kh = kf.reshape(&[b, nkv, hd][..])?;
            let kh = if !qk_full {
                if let Some(kn) = layer.attn_k_norm.as_ref() {
                    kn.forward(&kh.contiguous()?)?
                } else {
                    kh
                }
            } else {
                kh
            };
            let q_rot = crate::inference::cache::paged_attention::rope_apply(
                &qh,
                &rcos,
                &rsin,
                self.config.flags.use_rope_i,
            )?
            .reshape(&[b, nh * hd][..])?;
            let k_rot = crate::inference::cache::paged_attention::rope_apply(
                &kh,
                &rcos,
                &rsin,
                self.config.flags.use_rope_i,
            )?
            .reshape(&[b, nkv * hd][..])?;
            let v_f = v.to_dtype(cdt)?;
            if let Some((slot_dev, _, _, _)) = paged_dev.as_ref() {
                crate::inference::kernel::fused::paged_kv_write(
                    &k_rot,
                    &v_f,
                    stores[li].k(),
                    stores[li].v(),
                    slot_dev,
                    b,
                    nkv * hd,
                )?;
            } else {
                stores[li].write_stable(slots, &k_rot, &v_f)?;
            }
            // CPU: scalar attention (bit-identical to forward()). GPU: tensor ops
            // (matmul+softmax run on-device, no host round-trip). The scalar path
            // avoids the tensor-matmul float-reassociation that flips borderline
            // greedy tokens vs the CPU reference; GPU forward() uses the same tensor
            // kernels so the tensor path matches it there.
            let attn = if let Some((_, seq_lens_dev, block_table_dev, max_blocks)) =
                paged_dev.as_ref()
            {
                // Capture-safe paged flash-decode: fused paged gather + attention.
                let qb = q_rot.reshape(&[b, nh, hd][..])?;
                let bs = stores[li].block_size();
                match crate::inference::kernel::fused::paged_flash_decode(
                    &qb,
                    stores[li].k(),
                    stores[li].v(),
                    block_table_dev,
                    seq_lens_dev,
                    scale,
                    b,
                    nh,
                    nkv,
                    hd,
                    bs,
                    *max_blocks,
                )? {
                    Some(o) => o.reshape(&[b, nh * hd][..])?,
                    None => return Err(crate::tensor::Error::msg("paged_flash_decode declined")),
                }
            } else if dev.is_cpu() {
                let qv = q_rot.to_vec2::<f32>()?;
                let mut rows: Vec<f32> = Vec::with_capacity(b * nh * hd);
                for bi in 0..b {
                    let (kg, vg) = stores[li].gather_seq(&block_tables[bi], context_lens[bi])?;
                    let kgv = kg.to_vec2::<f32>()?.concat();
                    let vgv = vg.to_vec2::<f32>()?.concat();
                    let mut out = vec![0f32; nh * hd];
                    let mut s = crate::inference::cache::paged_attention::PagedAttnSeq {
                        q: &qv[bi],
                        k: &kgv,
                        v: &vgv,
                        context_len: context_lens[bi],
                        out: &mut out,
                    };
                    crate::inference::cache::paged_attention::paged_attn_decode_seq(
                        &mut s, nh, nkv, hd, scale,
                    );
                    rows.extend_from_slice(&out);
                }
                Tensor::from_vec(rows, &[b, nh * hd][..], &dev)?
            } else {
                // ONE ragged batched GQA attention over all B sequences (incl. B=1).
                // Handles unequal contexts via padding+mask; does ALL heads in a
                // single batched matmul/softmax (no per-kv-head loop) -> far fewer
                // kernel launches than the per-seq path (the decode is launch-bound
                // at small batch, so launch count is the lever vs the optimized
                // single-stream forward()).
                let max_ctx = *context_lens.iter().max().unwrap();
                let (kb, vb) =
                    stores[li].gather_batch_padded_stable(block_tables, context_lens, max_ctx)?;
                let kb = kb.reshape(&[b, max_ctx, nkv, hd][..])?;
                let vb = vb.reshape(&[b, max_ctx, nkv, hd][..])?;
                let qb = q_rot.reshape(&[b, nh, hd][..])?;
                if let Some(pd) = pos_dev.as_ref() {
                    // Fused flash-decode kernel over the gathered (contiguous, padded)
                    // K/V via strides - per-seq real length from pos_dev, no cuBLAS,
                    // split-K for long ctx. Falls back to the ragged matmul if the
                    // kernel declines (head_dim constraints).
                    let kt = kb.transpose(1, 2)?; // [b, nkv, max_ctx, hd] strided view (hd contiguous)
                    let vt = vb.transpose(1, 2)?;
                    match crate::inference::kernel::fused::flash_decode_devkvlen(
                        &qb, &kt, &vt, pd, None, 0, scale, b, nh, nkv, hd,
                    )? {
                        Some(o) => o.reshape(&[b, nh * hd][..])?,
                        None => crate::inference::cache::paged_attention::paged_attn_decode_ragged_tensor(
                            &qb,
                            &kb,
                            &vb,
                            ragged_mask.as_ref().unwrap(),
                            nh,
                            nkv,
                            hd,
                            scale,
                        )?
                        .reshape(&[b, nh * hd][..])?,
                    }
                } else {
                    let a =
                        crate::inference::cache::paged_attention::paged_attn_decode_ragged_tensor(
                            &qb,
                            &kb,
                            &vb,
                            ragged_mask.as_ref().unwrap(),
                            nh,
                            nkv,
                            hd,
                            scale,
                        )?;
                    a.reshape(&[b, nh * hd][..])?
                }
            };
            let o = attn
                .to_dtype(cdt)
                .and_then(|a| layer.attn_output.forward(&a))?
                .to_dtype(cdt)?;
            x = residual.broadcast_add(&o)?;
            // FFN (SwiGLU) - tensor ops (device-agnostic: on-device on GPU, no host
            // round-trip). act = silu(gate) * up.
            let residual2 = x.clone();
            let hn2 = layer
                .ffn_norm
                .as_ref()
                .unwrap()
                .forward(&x)?
                .to_dtype(cdt)?;
            // gate/up: split weights OR fused gate||up in ffn_up ([B,2.ffn_dim], qwen3/Phi3).
            let (g, u) = if let Some(gate) = layer.ffn_gate.as_ref() {
                (
                    gate.forward(&hn2)?.to_dtype(cdt)?,
                    layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?,
                )
            } else {
                let gu = layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?;
                (
                    gu.narrow(1, 0, c.ffn_dim)?,
                    gu.narrow(1, c.ffn_dim, c.ffn_dim)?,
                )
            };
            let act = if c.flags.use_gelu {
                g.gelu()?.mul(&u)?
            } else {
                g.silu()?.mul(&u)?
            };
            let d = layer.ffn_down.forward(&act)?.to_dtype(cdt)?;
            x = residual2.broadcast_add(&d)?;
            note_if_non_finite(&format!("layer {li}"), &x);
        }
        self.head_on_device(&x)
    }

    /// The capturable GPU decode region: batched layer loop + lm_head, reading ALL
    /// per-step dynamic data from caller-provided tensors at stable addresses
    /// (`x` input hidden, `rcos`/`rsin`, ragged `mask`, gather index `gidx` `[b*cap]`,
    /// write index `widx` `[b,feat]`). No `from_vec` inside -> safe to wrap in a CUDA
    /// graph (the caller refreshes those buffers OUTSIDE capture, then replays).
    /// Returns logits `[b, vocab]`.
    pub fn decode_layers_gpu(
        &self,
        x: &Tensor,
        rcos: &Tensor,
        rsin: &Tensor,
        block_table_dev: &Tensor,
        slot_dev: &Tensor,
        seq_lens_dev: &Tensor,
        b: usize,
        max_blocks: usize,
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        use crate::tensor::DType;
        let c = &self.config;
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        use crate::inference::kernel::fused::fused_rmsnorm_f16;
        let cdt = DType::F16;
        let mut x = x.clone();
        for (li, layer) in self.layers.iter().enumerate() {
            // Time on the calling thread. A layer whose work is enqueued on a device
            // returns before that work is done, so this is what ISSUING the layer costs -
            // which is exactly what separates a layer that dispatches from one that runs
            // on the host, and the gap the endpoint exists to show.
            let _timer = crate::inference::place::layer_perf::LayerTimer::start(li);
            let residual = x.clone();
            // F16-native RmsNorm (no F32 cast kernel -> safe inside CUDA-graph capture,
            // and it dispatches on the device capture stream).
            let hn = fused_rmsnorm_f16(&x, layer.attn_norm.weight(), layer.attn_norm.eps() as f32)?;
            let (qd, kvd) = (nh * hd, nkv * hd);
            let (q, k, v) = if let Some(qkvw) = layer.attn_qkv.as_ref() {
                let mut qkv = qkvw.forward(&hn)?;
                if let Some(bias) = layer.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(&bias.to_dtype(qkv.dtype())?)?;
                }
                (
                    qkv.narrow(1, 0, qd)?,
                    qkv.narrow(1, qd, kvd)?,
                    qkv.narrow(1, qd + kvd, kvd)?,
                )
            } else {
                let mut q = layer.attn_q.as_ref().unwrap().forward(&hn)?;
                let mut k = layer.attn_k.as_ref().unwrap().forward(&hn)?;
                let mut v = layer.attn_v.as_ref().unwrap().forward(&hn)?;
                if let Some(bias) = layer.attn_q_bias.as_ref() {
                    q = q.broadcast_add(&bias.to_dtype(q.dtype())?)?;
                }
                if let Some(bias) = layer.attn_k_bias.as_ref() {
                    k = k.broadcast_add(&bias.to_dtype(k.dtype())?)?;
                }
                if let Some(bias) = layer.attn_v_bias.as_ref() {
                    v = v.broadcast_add(&bias.to_dtype(v.dtype())?)?;
                }
                (q, k, v)
            };
            let qh = q.to_dtype(cdt)?.reshape(&[b, nh, hd][..])?;
            let qh = if let Some(qn) = layer.attn_q_norm.as_ref() {
                fused_rmsnorm_f16(&qh.contiguous()?, qn.weight(), qn.eps() as f32)?
            } else {
                qh
            };
            let kh = k.to_dtype(cdt)?.reshape(&[b, nkv, hd][..])?;
            let kh = if let Some(kn) = layer.attn_k_norm.as_ref() {
                fused_rmsnorm_f16(&kh.contiguous()?, kn.weight(), kn.eps() as f32)?
            } else {
                kh
            };
            let k_rot = crate::inference::cache::paged_attention::rope_apply(
                &kh,
                rcos,
                rsin,
                self.config.flags.use_rope_i,
            )?
            .reshape(&[b, nkv * hd][..])?;
            let q_rot = crate::inference::cache::paged_attention::rope_apply(
                &qh,
                rcos,
                rsin,
                self.config.flags.use_rope_i,
            )?;
            let v_f = v.to_dtype(cdt)?;
            // Fully capture-safe paged write + fused paged flash-decode: both index
            // the paged store via DEVICE buffers (slot/block_table/seq_lens) read at
            // kernel-exec time - no index_select/scatter (which BAKE host-read
            // indices at capture) and no cuBLAS (not graph-replay-safe). This is what
            // makes the captured graph replay correctly.
            crate::inference::kernel::fused::paged_kv_write(
                &k_rot,
                &v_f,
                stores[li].k(),
                stores[li].v(),
                slot_dev,
                b,
                nkv * hd,
            )?;
            let qb = q_rot.reshape(&[b, nh, hd][..])?;
            let bs = stores[li].block_size();
            let a = crate::inference::kernel::fused::paged_flash_decode(
                &qb,
                stores[li].k(),
                stores[li].v(),
                block_table_dev,
                seq_lens_dev,
                scale,
                b,
                nh,
                nkv,
                hd,
                bs,
                max_blocks,
            )?
            .ok_or_else(|| crate::tensor::Error::msg("paged_flash_decode declined"))?
            .reshape(&[b, nh * hd][..])?;
            let o = a
                .to_dtype(cdt)
                .and_then(|a| layer.attn_output.forward(&a))?
                .to_dtype(cdt)?;
            x = residual.broadcast_add(&o)?;
            let residual2 = x.clone();
            let ffn_norm = layer.ffn_norm.as_ref().unwrap();
            let hn2 = fused_rmsnorm_f16(&x, ffn_norm.weight(), ffn_norm.eps() as f32)?;
            let (g, u) = if let Some(gate) = layer.ffn_gate.as_ref() {
                (gate.forward(&hn2)?, layer.ffn_up.forward(&hn2)?)
            } else {
                let gu = layer.ffn_up.forward(&hn2)?;
                (
                    gu.narrow(1, 0, c.ffn_dim)?,
                    gu.narrow(1, c.ffn_dim, c.ffn_dim)?,
                )
            };
            let act = if c.flags.use_gelu {
                g.gelu()?.mul(&u)?
            } else {
                g.silu()?.mul(&u)?
            };
            let d = layer.ffn_down.forward(&act)?;
            x = residual2.broadcast_add(&d)?;
            note_if_non_finite(&format!("layer {li}"), &x);
        }
        // Return the final normalized hidden (GPU). The lm_head (output_proj) is
        // applied OUTSIDE the captured region by `lm_head()` - it can live on CPU
        // for big-vocab models, which would break a GPU-only capture.
        // CRITICAL for CUDA-graph capture: output_norm.weight() often lives on CPU
        // (placed with lm_head/embeddings for big-vocab models). fused_rmsnorm_f16
        // SILENTLY no-ops on a CPU weight (the storage match fails -> no kernel, zero
        // graph nodes -> hidden never written on replay). Use the GPU-resident copy
        // (output_norm_cuda_weight, built at load) so the final norm is a real,
        // capture-safe kernel on the compute device.
        let norm_w = self
            .output_norm_cuda_weight
            .as_ref()
            .filter(|w| w.device().same_device(&x.device()))
            .map(|w| Ok(w.clone()))
            .unwrap_or_else(|| self.output_norm.weight().to_device(&x.device()))?;
        fused_rmsnorm_f16(&x, &norm_w, self.output_norm.eps() as f32)
    }

    /// Apply the lm_head (output projection) to a final hidden `[b, hidden]` ->
    /// logits `[b, vocab]`. Kept out of `decode_layers_gpu` so the captured graph
    /// stays GPU-only even when output_proj lives on CPU.
    pub fn lm_head(&self, f: &Tensor) -> Result<Tensor> {
        // `f` is the POST-final-norm hidden (the caller - decode_layers_gpu /
        // forward_from_hidden - already applied output_norm). Prefer the GPU
        // quantized projection when it's resident: it keeps the feature on-device
        // (no DtoH download), which is both faster per decode step AND mandatory
        // for the CUDA-graph CB path - a CPU output_proj would host-download the
        // graph-produced hidden buffer mid-replay (uninitialized/unready read).
        if let Some(proj) = self.output_proj_cuda_qmm.as_ref() {
            let dev = self
                .output_proj_dev
                .clone()
                .unwrap_or_else(|| f.device().clone());
            let fq = if f.device().same_device(&dev) {
                f.clone()
            } else {
                f.to_device(&dev)?
            };
            return self.apply_output_bias(proj.forward(&fq)?);
        }
        self.apply_output_bias(self.output_proj.forward(f)?)
    }

    /// True when the lm-head runs on GPU (quantized projection resident) - a
    /// prerequisite for the CUDA-graph CB path (no CPU host-download of the
    /// graph-produced hidden buffer during replay).
    /// How many tokens the KV cache actually holds, read from the cache rather
    /// than from a counter kept beside it.
    ///
    /// The decode loop's `pos` is a sampling counter and lags the cache by one
    /// whenever a sampled token is carried forward before being pushed. That is
    /// invisible to a single-position forward - `make_mask(1, pos)` yields
    /// `pos + 1`, which happens to equal the cache - and fatal to a
    /// multi-position one, whose mask then comes out one key short. Adjusting
    /// `pos` by a constant does not work either: the lag is one after an
    /// ordinary step and zero after a speculative verify has already
    /// resynchronised. So callers that need a POSITION ask the cache.
    /// The cards holding this model's layers, fastest-first order not implied, and how many
    /// layers run on the host.
    pub fn placement(&self) -> crate::inference::place::Placement {
        let mut cards = std::collections::BTreeSet::new();
        let mut host_layers = 0;
        for d in &self.layer_devs {
            match d {
                LayerDevice::Cuda(i) => {
                    cards.insert(*i);
                }
                LayerDevice::Cpu => host_layers += 1,
            }
        }
        crate::inference::place::Placement { cards, host_layers }
    }

    /// The routing counts of every layer since the load, zero for a layer without an expert
    /// cache; `None` when no layer has one.
    #[cfg(feature = "gpu")]
    pub fn expert_profile(&self) -> Option<super::expert_profile::ExpertProfile> {
        let experts = self.layers.iter().find_map(|l| {
            l.moe
                .as_ref()?
                .tier
                .as_ref()
                .map(|t| t.routing_counts().len())
        })?;
        let mut counts = Vec::with_capacity(self.layers.len() * experts);
        for layer in &self.layers {
            match layer.moe.as_ref().and_then(|m| m.tier.as_ref()) {
                Some(tier) => counts.extend(tier.routing_counts()),
                None => counts.extend(std::iter::repeat_n(0u32, experts)),
            }
        }
        Some(super::expert_profile::ExpertProfile {
            layers: self.layers.len(),
            experts,
            counts,
        })
    }

    /// Fill every layer's expert cache from `profile`: how many experts were copied. A profile
    /// of another shape is refused, naming both shapes.
    #[cfg(feature = "gpu")]
    pub fn prime_expert_caches(
        &self,
        profile: &super::expert_profile::ExpertProfile,
    ) -> crate::tensor::Result<usize> {
        let experts = self
            .layers
            .iter()
            .find_map(|l| l.moe.as_ref()?.tier.as_ref())
            .map(|t| t.n_experts());
        if profile.layers != self.layers.len() || experts.is_some_and(|n| n != profile.experts) {
            crate::tensor::bail!(
                "the profile counts {} layers of {} experts, the model has {} layers of {} experts",
                profile.layers,
                profile.experts,
                self.layers.len(),
                experts.unwrap_or(0)
            );
        }
        let mut copied = 0;
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(tier) = layer.moe.as_ref().and_then(|m| m.tier.as_ref()) {
                copied += tier.prime(profile.layer(l))?;
            }
        }
        if let Some(h) = &self.hot_locks {
            h.prime(profile);
        }
        Ok(copied)
    }

    /// Routed experts looked up in the layers' expert caches since the load, and how many of
    /// them a card held; `None` when no layer has such a cache.
    pub fn expert_cache_counters(&self) -> Option<(u64, u64)> {
        let mut total = (0u64, 0u64);
        let mut any = false;
        for tier in self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref()?.tier.as_ref())
        {
            let (routed, hits, _) = tier.counters();
            total = (total.0 + routed, total.1 + hits);
            any = true;
        }
        any.then_some(total)
    }

    pub fn kv_len(&self) -> Option<usize> {
        // A recurrent layer keeps no cache: the first attending layer holds the length.
        let l = self
            .layers
            .iter()
            .find(|l| matches!(l.mixer, super::mixer::Mixer::Attention))?;
        let mut n = l.kv_cache.current_seq_len();
        if let Some(c) = l.context.as_ref() {
            n = n.max(c.len());
        }
        #[cfg(feature = "gpu")]
        if let Some(c) = l.quant_kv.as_ref() {
            n = n.max(c.current_seq_len());
        }
        if let Some(c) = l.cpu_q8_kv.as_ref() {
            n = n.max(c.len());
        }
        Some(n)
    }

    /// PREFILL one sequence's whole prompt in ONE forward (vs T single-token
    /// decodes), writing all T positions' K/V into the paged stores at `slots`
    /// and returning the LAST position's logits `[1, vocab]`. Same dense-arch
    /// support + compute dtype as `batched_paged_decode`. This is the TTFT lever
    /// for continuous-batch serving - without it each request pays T serial
    /// memory-bound single-token forwards before its first token.
    pub fn paged_prefill_seq(
        &self,
        tokens: &[u32],
        slots: &[usize],
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        use crate::tensor::Tensor;
        let c = &self.config;
        let t = tokens.len();
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let base = c.rope_freq_base;
        let edev = self.embeddings.embeddings().device().clone();
        let dev = self.compute_device();
        let cdt = Self::residual_dtype(&dev);
        let positions: Vec<usize> = (0..t).collect();

        let ids = Tensor::from_vec(tokens.to_vec(), &[t, 1][..], &edev)?;
        let mut x = self
            .embeddings
            .forward(&ids)?
            .reshape(&[t, c.embedding_length][..])?
            .to_dtype(cdt)?;
        if !x.device().same_device(&dev) {
            x = x.to_device(&dev)?;
        }
        let (rcos, rsin) = crate::inference::cache::paged_attention::rope_cos_sin(
            &positions, hd, base, &dev, cdt,
        )?;

        note_if_non_finite("the embedding, before any layer", &x);
        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let hn = layer.attn_norm.forward(&x)?.to_dtype(cdt)?;
            let (qd, kvd) = (nh * hd, nkv * hd);
            let (q, k, v) = if let Some(qkvw) = layer.attn_qkv.as_ref() {
                let mut qkv = qkvw.forward(&hn)?;
                if let Some(bias) = layer.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(&bias.to_dtype(qkv.dtype())?)?;
                }
                (
                    qkv.narrow(1, 0, qd)?,
                    qkv.narrow(1, qd, kvd)?,
                    qkv.narrow(1, qd + kvd, kvd)?,
                )
            } else {
                let mut q = layer.attn_q.as_ref().unwrap().forward(&hn)?;
                let mut k = layer.attn_k.as_ref().unwrap().forward(&hn)?;
                let mut v = layer.attn_v.as_ref().unwrap().forward(&hn)?;
                if let Some(bias) = layer.attn_q_bias.as_ref() {
                    q = q.broadcast_add(&bias.to_dtype(q.dtype())?)?;
                }
                if let Some(bias) = layer.attn_k_bias.as_ref() {
                    k = k.broadcast_add(&bias.to_dtype(k.dtype())?)?;
                }
                if let Some(bias) = layer.attn_v_bias.as_ref() {
                    v = v.broadcast_add(&bias.to_dtype(v.dtype())?)?;
                }
                (q, k, v)
            };
            let qh = q.to_dtype(cdt)?.reshape(&[t, nh, hd][..])?;
            let qh = if let Some(qn) = layer.attn_q_norm.as_ref() {
                qn.forward(&qh.contiguous()?)?
            } else {
                qh
            };
            let kh = k.to_dtype(cdt)?.reshape(&[t, nkv, hd][..])?;
            let kh = if let Some(kn) = layer.attn_k_norm.as_ref() {
                kn.forward(&kh.contiguous()?)?
            } else {
                kh
            };
            let q_rot = crate::inference::cache::paged_attention::rope_apply(
                &qh,
                &rcos,
                &rsin,
                self.config.flags.use_rope_i,
            )?;
            let k_rot = crate::inference::cache::paged_attention::rope_apply(
                &kh,
                &rcos,
                &rsin,
                self.config.flags.use_rope_i,
            )?;
            let v_f = v.to_dtype(cdt)?.reshape(&[t, nkv, hd][..])?;
            // persist all T positions' K/V for the subsequent decode phase
            stores[li].write(
                slots,
                &k_rot.reshape(&[t, nkv * hd][..])?,
                &v_f.reshape(&[t, nkv * hd][..])?,
            )?;
            let attn = crate::inference::cache::paged_attention::paged_prefill_attn_tensor(
                &q_rot, &k_rot, &v_f, nh, nkv, hd, scale,
            )?
            .reshape(&[t, nh * hd][..])?;
            let o = attn
                .to_dtype(cdt)
                .and_then(|a| layer.attn_output.forward(&a))?
                .to_dtype(cdt)?;
            x = residual.broadcast_add(&o)?;
            let residual2 = x.clone();
            let hn2 = layer
                .ffn_norm
                .as_ref()
                .unwrap()
                .forward(&x)?
                .to_dtype(cdt)?;
            let (g, u) = if let Some(gate) = layer.ffn_gate.as_ref() {
                (
                    gate.forward(&hn2)?.to_dtype(cdt)?,
                    layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?,
                )
            } else {
                let gu = layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?;
                (
                    gu.narrow(1, 0, c.ffn_dim)?,
                    gu.narrow(1, c.ffn_dim, c.ffn_dim)?,
                )
            };
            let act = if c.flags.use_gelu {
                g.gelu()?.mul(&u)?
            } else {
                g.silu()?.mul(&u)?
            };
            let d = layer.ffn_down.forward(&act)?.to_dtype(cdt)?;
            x = residual2.broadcast_add(&d)?;
            note_if_non_finite(&format!("layer {li}"), &x);
        }
        // logits for the last position only
        let last = x.narrow(0, t - 1, 1)?;
        self.head_on_device(&last)
    }

    /// Prefix-cached prefill: compute ONLY the suffix `tokens[cached_len..]`,
    /// reusing the already-resident prefix KV (positions `0..cached_len`, gathered
    /// from each layer's store via `block_table`) for attention instead of
    /// recomputing it. `suffix_slots` are the KV slots for the suffix positions
    /// (length `t - cached_len`). Returns the LAST position's logits - equivalent to
    /// `paged_prefill_seq` over the whole prompt (validated) but skips the prefix's
    /// projections + FFN (the multi-turn / shared-system-prompt win). `cached_len`
    /// must be block-aligned (a whole number of cached blocks).
    pub fn paged_prefill_seq_cached(
        &self,
        tokens: &[u32],
        suffix_slots: &[usize],
        cached_len: usize,
        block_table: &[u32],
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        use crate::tensor::Tensor;
        let c = &self.config;
        let t = tokens.len();
        let tsfx = t - cached_len;
        debug_assert_eq!(suffix_slots.len(), tsfx);
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let base = c.rope_freq_base;
        let edev = self.embeddings.embeddings().device().clone();
        let dev = self.compute_device();
        let cdt = Self::residual_dtype(&dev);
        let positions: Vec<usize> = (cached_len..t).collect();

        let ids = Tensor::from_vec(tokens[cached_len..].to_vec(), &[tsfx, 1][..], &edev)?;
        let mut x = self
            .embeddings
            .forward(&ids)?
            .reshape(&[tsfx, c.embedding_length][..])?
            .to_dtype(cdt)?;
        if !x.device().same_device(&dev) {
            x = x.to_device(&dev)?;
        }
        let (rcos, rsin) = crate::inference::cache::paged_attention::rope_cos_sin(
            &positions, hd, base, &dev, cdt,
        )?;

        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let hn = layer.attn_norm.forward(&x)?.to_dtype(cdt)?;
            let (qd, kvd) = (nh * hd, nkv * hd);
            let (q, k, v) = if let Some(qkvw) = layer.attn_qkv.as_ref() {
                let mut qkv = qkvw.forward(&hn)?;
                if let Some(bias) = layer.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(&bias.to_dtype(qkv.dtype())?)?;
                }
                (
                    qkv.narrow(1, 0, qd)?,
                    qkv.narrow(1, qd, kvd)?,
                    qkv.narrow(1, qd + kvd, kvd)?,
                )
            } else {
                let mut q = layer.attn_q.as_ref().unwrap().forward(&hn)?;
                let mut k = layer.attn_k.as_ref().unwrap().forward(&hn)?;
                let mut v = layer.attn_v.as_ref().unwrap().forward(&hn)?;
                if let Some(bias) = layer.attn_q_bias.as_ref() {
                    q = q.broadcast_add(&bias.to_dtype(q.dtype())?)?;
                }
                if let Some(bias) = layer.attn_k_bias.as_ref() {
                    k = k.broadcast_add(&bias.to_dtype(k.dtype())?)?;
                }
                if let Some(bias) = layer.attn_v_bias.as_ref() {
                    v = v.broadcast_add(&bias.to_dtype(v.dtype())?)?;
                }
                (q, k, v)
            };
            let qh = q.to_dtype(cdt)?.reshape(&[tsfx, nh, hd][..])?;
            let qh = if let Some(qn) = layer.attn_q_norm.as_ref() {
                qn.forward(&qh.contiguous()?)?
            } else {
                qh
            };
            let kh = k.to_dtype(cdt)?.reshape(&[tsfx, nkv, hd][..])?;
            let kh = if let Some(kn) = layer.attn_k_norm.as_ref() {
                kn.forward(&kh.contiguous()?)?
            } else {
                kh
            };
            let q_rot = crate::inference::cache::paged_attention::rope_apply(
                &qh,
                &rcos,
                &rsin,
                self.config.flags.use_rope_i,
            )?;
            let k_rot = crate::inference::cache::paged_attention::rope_apply(
                &kh,
                &rcos,
                &rsin,
                self.config.flags.use_rope_i,
            )?; // [tsfx, nkv, hd]
            let v_f = v.to_dtype(cdt)?.reshape(&[tsfx, nkv, hd][..])?;
            // Persist the SUFFIX K/V; gather the cached PREFIX K/V (already rotated)
            // and prepend it for attention. cached_len==0 = a plain full prefill.
            stores[li].write(
                suffix_slots,
                &k_rot.reshape(&[tsfx, nkv * hd][..])?,
                &v_f.reshape(&[tsfx, nkv * hd][..])?,
            )?;
            let (full_k, full_v) = if cached_len == 0 {
                (k_rot.clone(), v_f.clone())
            } else {
                let (pk, pv) = stores[li].gather_seq(block_table, cached_len)?; // [cached, feat]
                let pk = pk.reshape(&[cached_len, nkv, hd][..])?.to_dtype(cdt)?;
                let pv = pv.reshape(&[cached_len, nkv, hd][..])?.to_dtype(cdt)?;
                (
                    Tensor::cat(&[&pk, &k_rot], 0)?,
                    Tensor::cat(&[&pv, &v_f], 0)?,
                ) // [t, nkv, hd]
            };
            let attn = crate::inference::cache::paged_attention::paged_prefill_attn_suffix(
                &q_rot, &full_k, &full_v, cached_len, nh, nkv, hd, scale,
            )?
            .reshape(&[tsfx, nh * hd][..])?;
            let o = attn
                .to_dtype(cdt)
                .and_then(|a| layer.attn_output.forward(&a))?
                .to_dtype(cdt)?;
            x = residual.broadcast_add(&o)?;
            let residual2 = x.clone();
            let hn2 = layer
                .ffn_norm
                .as_ref()
                .unwrap()
                .forward(&x)?
                .to_dtype(cdt)?;
            let (g, u) = if let Some(gate) = layer.ffn_gate.as_ref() {
                (
                    gate.forward(&hn2)?.to_dtype(cdt)?,
                    layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?,
                )
            } else {
                let gu = layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?;
                (
                    gu.narrow(1, 0, c.ffn_dim)?,
                    gu.narrow(1, c.ffn_dim, c.ffn_dim)?,
                )
            };
            let act = if c.flags.use_gelu {
                g.gelu()?.mul(&u)?
            } else {
                g.silu()?.mul(&u)?
            };
            let d = layer.ffn_down.forward(&act)?.to_dtype(cdt)?;
            x = residual2.broadcast_add(&d)?;
            note_if_non_finite(&format!("layer {li}"), &x);
        }
        let last = x.narrow(0, tsfx - 1, 1)?;
        self.head_on_device(&last)
    }

    /// Multi-position forward: returns logits for EVERY input position
    /// (shape `[batch, seq_len, vocab]`). Used by speculative decoding to
    /// verify K drafts in a single batched forward.
    ///
    /// A draft verify (a few rows continuing a cache) computes every row as the decode step it
    /// stands for, on the host and on a card, so drafting never changes what greedy decoding
    /// emits. The rows still share each weight read where the decode kernels take several.
    pub fn forward_all(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let verify = index_pos > 0 && seq_len <= crate::inference::engine::decode_step::VERIFY_ROWS;
        let recording = verify && self.is_recurrent();
        if recording {
            self.spec_open(input_ids, index_pos)?;
        }
        // The verify advances each state where it lies, as a decoded token does, so a step
        // recorded against those buffers goes on reading them.
        #[cfg(feature = "gpu")]
        let _in_place = recording.then(|| crate::inference::moe_cuda::in_place_state(true));
        // Every verified row runs the decode step it stands for, so greedy output does not
        // depend on how many tokens were drafted.
        let _order = verify.then(crate::tensor::quantized::DecodeOrder::enter);
        let out = self.forward_inner(input_ids, index_pos, /*all_positions=*/ true);
        if recording {
            self.spec_close(out.is_ok());
        }
        out
    }

    /// `forward_all` with the rows batched as a prefill batches them, for the verify
    /// benchmark.
    #[cfg(test)]
    pub(super) fn forward_all_batched(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, index_pos, /*all_positions=*/ true)
    }

    fn forward_inner(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
        all_positions: bool,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        self.spec_clear();
        let (hidden, ple_inputs) = self.embed_stage(input_ids, index_pos)?;
        // A prompt chunk also goes through the draft head, so the head's cache holds the prompt
        // by the time the first token is drafted.
        if self.capture_feature && self.draft.is_some() && seq_len > 1 && !all_positions {
            let ids = input_ids
                .flatten_all()?
                .to_dtype(crate::tensor::DType::U32)?
                .to_vec1::<u32>()?;
            let (hidden, on_cuda) = self.run_layers(hidden, ple_inputs, index_pos, seq_len)?;
            self.draft_ingest(&ids, &hidden, index_pos)?;
            return self.output_stage(hidden, on_cuda, seq_len, all_positions);
        }
        self.forward_layers_and_output(hidden, ple_inputs, index_pos, seq_len, all_positions)
    }

    /// The n-gram table's rows for `input_ids`, which start at `index_pos`, gathered for the
    /// layer that takes them: the history of the sequence advances past the tokens. Nothing for
    /// a model with no n-gram layer.
    pub(super) fn gather_ngram_rows(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<()> {
        let Some(ng) = self.ngram.as_mut() else {
            return Ok(());
        };
        let ple = self.layers[ng.meta.layer]
            .ngram
            .as_mut()
            .expect("the layer the n-gram file names carries its block");
        if index_pos == 0 {
            ple.reset();
        }
        // A forward that failed after its gather runs again from the same position: the
        // history goes back to where that gather found it.
        if ple.history.seen != index_pos {
            if let Some((at, before)) = ng.before.as_ref().filter(|(at, _)| *at == index_pos) {
                debug_assert_eq!(*at, before.seen);
                ple.history = before.clone();
            }
        }
        if ple.history.seen != index_pos {
            // What stands between the history and the forward: the verify windows open.
            let recorded = self.verify.window.as_ref().map_or("none".to_string(), |w| {
                format!("{} rows at {}", w.rows, w.base)
            });
            let eager = self.spec_window.as_ref().map_or("none".to_string(), |w| {
                format!("{} rows at {}", w.rows, w.base)
            });
            crate::tensor::bail!(
                "the n-gram history holds {} tokens but the forward starts at {index_pos} (recorded window {recorded}, eager window {eager})",
                ple.history.seen
            );
        }
        let ids = input_ids
            .flatten_all()?
            .to_dtype(crate::tensor::DType::U32)?
            .to_vec1::<u32>()?;
        ng.before = Some((index_pos, ple.history.clone()));
        let rows = ple.history.rows(&ng.meta, &ids);
        let data = ng.table.gather(&rows)?;
        let width = ng.meta.heads() * ng.meta.row_dim;
        ng.pending = Some(Tensor::from_vec(data, (1, ids.len(), width), &Device::Cpu)?);
        Ok(())
    }

    /// The first stage of a forward: the embeddings of `input_ids` as the residual stream
    /// the layers start from, and the per-layer inputs of a family that has them.
    pub(super) fn embed_stage(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Option<Vec<Tensor>>)> {
        let (_, seq_len) = input_ids.dims2()?;

        // Small-model CPU prefill: route the activation quantize onto the same
        // logical prefill pool as the matmul, so the physical pool stays parked
        // and its cross-pool idle-spin is not paid. A narrow hidden size is the
        // model-level signal the per-matmul shape cannot separate; self-resets
        // to false on decode (seq_len == 1) and on wide-hidden models.
        crate::tensor::quant_cpu::set_small_model_prefill(
            seq_len > 1 && self.config.embedding_length <= 1024,
        );

        // -- Embedding ------------------------------
        // Cast input_ids to the embeddings table's actual device. Today
        // that's always CPU; Path A's load-time change will move small
        // tables to GPU 0 and this cast becomes a no-op (or short device->
        // device copy) for those models.
        let emb_device = self.embeddings.embeddings().device().clone();
        let input_local;
        let input_ids = if input_ids.device().same_device(&emb_device) {
            input_ids
        } else {
            input_local = input_ids.to_device(&emb_device)?;
            &input_local
        };
        let hidden = self.embeddings.forward(input_ids)?;
        // Embedding scaling (Gemma: √embed_len, Granite: explicit value)
        let hidden = if let Some(scale) = self.config.embed_scale {
            (hidden * scale)?
        } else {
            hidden
        };
        note_if_non_finite("the embedding, before any layer", &hidden);

        // A wide residual starts as one copy of the embedding per stream, and the n-gram
        // rows of the tokens are gathered for the layer that takes them.
        let hidden = match self.config.hyper {
            Some(h) => {
                self.gather_ngram_rows(input_ids, index_pos)?;
                super::wide::widen_embedding(&hidden, h.streams)?
            }
            None => hidden,
        };

        // -- Compute PLE per-layer inputs (Gemma4) ------------------------
        let ple_inputs: Option<Vec<Tensor>> = if self.ple_dim > 0 {
            if let (Some(model_proj), Some(proj_norm)) = (&self.ple_model_proj, &self.ple_proj_norm)
            {
                let n_layers = self.layers.len();
                let (b, seq_h, _) = hidden.dims3()?;
                let ple_scale = (self.ple_dim as f64).sqrt();
                // Token embeddings: use Embedding if available, else QTensor per-row dequant.
                // Cast input_ids to PLE table's device.
                let tok_ple = if let Some(embd) = &self.ple_token_embd {
                    let ple_device = embd.embeddings().device().clone();
                    let input_local;
                    let input_ids_ple = if input_ids.device().same_device(&ple_device) {
                        input_ids
                    } else {
                        input_local = input_ids.to_device(&ple_device)?;
                        &input_local
                    };
                    let t = embd.forward(input_ids_ple)?;
                    let t = if t.dtype() != crate::tensor::DType::F32 {
                        t.to_dtype(crate::tensor::DType::F32)?
                    } else {
                        t
                    };
                    Some((t * ple_scale)?)
                } else {
                    None
                };
                // Model projection -> [flat_seq, n_layers * ple_dim]. Move
                // hidden to the device the PLE projection lives on (GPU 0 when
                // available, else CPU); avoids the host round-trip that used
                // to dominate gemma4 prefill.
                let proj_dev = model_proj.device();
                let hidden_on_proj = if hidden.device().same_device(&proj_dev) {
                    hidden.clone()
                } else {
                    hidden.to_device(&proj_dev)?
                };
                let hidden_2d =
                    hidden_on_proj.reshape((b * seq_h, self.config.embedding_length))?;
                let model_ple = model_proj.forward(&hidden_2d)?;
                // Scale by 1/√hidden_size (matching Ollama's Go implementation)
                let model_ple = (model_ple / (self.config.embedding_length as f64).sqrt())?;
                // Reshape to [n_tokens, n_layers, ple_dim] for the RMSNorm (normalizes last dim)
                let model_ple = model_ple.reshape((b * seq_h, n_layers, self.ple_dim))?;
                let model_ple = proj_norm.forward(&model_ple)?;
                // Combine: (model + token) / √2, or just model if no token embedding.
                // tok comes from CPU embeddings; align it with model_ple's device
                // (typically GPU 0 now) before the addition.
                let ple = if let Some(tok) = tok_ple {
                    let tok = tok.reshape((b * seq_h, n_layers, self.ple_dim))?;
                    let tok = if tok.device().same_device(&model_ple.device()) {
                        tok
                    } else {
                        tok.to_device(&model_ple.device())?
                    };
                    // Go code line 249: scale by 1/√2 when combining both sources
                    ((model_ple + tok)? * (1.0 / 2f64.sqrt()))?
                } else {
                    model_ple
                };
                // Split into per-layer tensors: [n_tokens, ple_dim] -> [b, seq, ple_dim]
                let mut per_layer = Vec::with_capacity(n_layers);
                for i in 0..n_layers {
                    per_layer.push(ple.i((.., i, ..))?.reshape((b, seq_h, self.ple_dim))?);
                }
                Some(per_layer)
            } else {
                None
            }
        } else {
            None
        };

        Ok((hidden, ple_inputs))
    }

    /// Run the per-layer loop + final norm + output projection given a
    /// pre-embedded `hidden` state. Factored out of `forward_inner` so the
    /// vision-prefill path (`forward_with_image_embeds`) can splice image
    /// embeddings into `hidden` between the BOS and text token embeddings
    /// before reusing the standard layer + output-proj logic.
    ///
    /// `hidden`: (batch, seq_len, embedding_length) on CPU or a single CUDA
    /// device. The function moves it across segment boundaries as needed.
    /// `ple_inputs`: per-layer Gemma4 PLE tensors, or None for arches that
    /// don't use PLE (phi2, qwen2/3, llama, etc).
    /// `seq_len` must equal `hidden.dim(1)`.
    pub(super) fn forward_layers_and_output(
        &mut self,
        hidden: Tensor,
        ple_inputs: Option<Vec<Tensor>>,
        index_pos: usize,
        seq_len: usize,
        all_positions: bool,
    ) -> Result<Tensor> {
        let (hidden, on_cuda) = self.run_layers(hidden, ple_inputs, index_pos, seq_len)?;
        self.output_stage(hidden, on_cuda, seq_len, all_positions)
    }

    /// The layer loop of a forward: `hidden` through every layer, crossing the placement
    /// segments, returned with whether it ends on a card.
    pub(super) fn run_layers(
        &mut self,
        hidden: Tensor,
        ple_inputs: Option<Vec<Tensor>>,
        index_pos: usize,
        seq_len: usize,
    ) -> Result<(Tensor, bool)> {
        let upto = self.layers.len();
        self.run_layers_in(hidden, ple_inputs, index_pos, seq_len, 0, upto)
    }

    /// `run_layers` over the layers `[from, upto)` alone.
    pub(super) fn run_layers_in(
        &mut self,
        mut hidden: Tensor,
        ple_inputs: Option<Vec<Tensor>>,
        index_pos: usize,
        seq_len: usize,
        from: usize,
        upto: usize,
    ) -> Result<(Tensor, bool)> {
        let mut on_cuda = !hidden.device().is_cpu();

        // -- Layer-by-layer with segment-based transfers -------------------
        let mut seg_start = from;
        let mut current_cuda_idx: Option<usize> =
            if let crate::tensor::DeviceLocation::Cuda { gpu_id } = hidden.device().location() {
                Some(gpu_id)
            } else {
                None
            };
        while seg_start < upto {
            let seg_dev = &self.layer_devs[seg_start];
            let mut seg_end = seg_start + 1;
            while seg_end < upto && &self.layer_devs[seg_end] == seg_dev {
                seg_end += 1;
            }

            let target_cuda_idx = match seg_dev {
                LayerDevice::Cuda(idx) => Some(*idx),
                LayerDevice::Cpu => None,
            };

            // Transfer at segment boundary (CPU↔CUDA or CUDA↔CUDA)
            if target_cuda_idx != current_cuda_idx {
                // The destination must not read partially-written data while the
                // source's kernels are still in flight - without that ordering the
                // output varies between runs (observed on gemma4:26b across GPUs and
                // CPU). The ordering now travels WITH the transfer, inside `to_device`,
                // instead of being restated on both sides here: one declaration, and
                // the CUDA->CUDA case establishes it with an event rather than by
                // stopping the host. Work queued afterwards on the destination stream
                // is already ordered behind the copy, so the trailing sync was only
                // ever symmetry.
                hidden = if let Some(idx) = target_cuda_idx {
                    if let Some(dev) = self.cuda_devices.get(&idx) {
                        hidden.to_device(dev)?
                    } else {
                        hidden
                    }
                } else {
                    hidden.to_device(&Device::Cpu)?
                };
                current_cuda_idx = target_cuda_idx;
                on_cuda = current_cuda_idx.is_some();
            }

            // Compute mask once for segment. seq_kv = index_pos + seq_q so
            // session-resume / PLD-verify paths (where index_pos > 0) get a
            // [seq_q, index_pos+seq_q] mask that broadcasts correctly to
            // attention scores [b, n_head, seq_q, seq_kv]. Prefill at
            // index_pos=0 collapses to [seq_q, seq_q].
            let mask = if seq_len > 1 {
                Some(self.make_mask(seq_len, index_pos, current_cuda_idx)?)
            } else {
                None
            };
            // gemma4 bounded sliding-window attention: SWA layers with a windowed SpecKvCache
            // store only the last `window+chunk` keys, so their prefill K is shorter
            // than the full seq_kv - they need a mask matching the BOUNDED length
            // (swa_bounded_mask_data), not the full `mask`. All windowed SWA layers
            // share the same post-append length (identical append history). Only
            // built when a windowed SWA layer is present in this segment (gemma4 on
            // F16 KV); else None -> all layers use the full `mask` (unchanged).
            let mask_swa = if seq_len > 1 {
                let swa = (seg_start..seg_end).find_map(|i| {
                    let l = &self.layers[i];
                    l.kv_cache.window().filter(|&w| w > 0).map(|w| {
                        (
                            w,
                            l.kv_cache.current_seq_len(),
                            l.kv_cache.max_seq_len_padded(),
                        )
                    })
                });
                match swa {
                    Some((w, cur, cap)) => {
                        // Mirror EXACTLY the windowed SpecKvCache::append the SWA
                        // layers are about to perform, so the mask width equals
                        // the K length the attention will see:
                        //  - the prefill attention resets the cache at index_pos==0
                        //    (first chunk), so the pre-append length is 0 there  -
                        //    NOT the stale current_seq_len left by a prior request.
                        //  - otherwise it's the cache's current length, and the
                        //    slide keeps the last `w` keys before appending.
                        let before = if index_pos == 0 { 0 } else { cur };
                        let post = if before + seq_len > cap && before > w {
                            w + seq_len
                        } else {
                            before + seq_len
                        };
                        let buf_start = (index_pos + seq_len) - post;
                        let dev = current_cuda_idx
                            .and_then(|idx| self.cuda_devices.get(&idx))
                            .unwrap_or(&Device::Cpu);
                        let data = swa_bounded_mask_data(seq_len, index_pos, post, buf_start, w);
                        Some(Tensor::from_slice(&data, (seq_len, post), dev)?)
                    }
                    None => None,
                }
            } else {
                None
            };

            // Shared KV: layers >= first_kv_own compute their own K/V;
            // layers after that reuse K/V from reference layers.
            let n_layers = self.layers.len();
            let shared_kv_n = self.config.shared_kv_layers;
            let first_kv_shared = if shared_kv_n > 0 {
                n_layers - shared_kv_n
            } else {
                n_layers
            };

            // Run all layers in segment
            // THE per-layer loop of the decode path: every request walks the placement
            // segments through here. Timed on the calling thread, so a layer that
            // dispatches to a device and one that runs on the host are told apart.
            for layer_idx in seg_start..seg_end {
                let _timer = crate::inference::place::layer_perf::LayerTimer::start(layer_idx);
                // A wide batch will read this layer's experts and the next one's: they start
                // coming to the card now, behind the work already queued.
                if seq_len > 1 {
                    for l in [layer_idx, layer_idx + 1] {
                        if let Some(moe) = self.layers.get(l).and_then(|x| x.moe.as_ref()) {
                            if let Some(tier) = moe
                                .tier
                                .as_ref()
                                .filter(|t| t.stream_pays(seq_len, moe.n_experts_used))
                            {
                                tier.prefetch()?;
                            }
                        }
                    }
                }
                // PLE: the layer's input moves to the layer's device, host or card.
                let ple_on_dev = match ple_inputs.as_ref().and_then(|inputs| inputs.get(layer_idx))
                {
                    Some(t) => {
                        let layer_dev = if on_cuda {
                            current_cuda_idx
                                .and_then(|idx| self.cuda_devices.get(&idx))
                                .cloned()
                        } else {
                            Some(Device::Cpu)
                        };
                        match layer_dev {
                            Some(d) if !t.device().same_device(&d) => Some(t.to_device(&d)?),
                            _ => Some(t.clone()),
                        }
                    }
                    None => None,
                };

                // The n-gram embeddings go to the one layer that adds them.
                let ple_on_dev = match (
                    self.ngram.as_ref().and_then(|n| n.pending.as_ref()),
                    self.layers[layer_idx].ngram.is_some(),
                ) {
                    (Some(emb), true) => Some(emb.clone()),
                    _ => ple_on_dev,
                };

                // Shared KV: for layers in the shared range, grab cached K/V from a
                // type-matched reference layer (SWA->SWA ref, global->global ref).
                let (shared_kv, ref_idx_opt) =
                    if layer_idx >= first_kv_shared && first_kv_shared > 0 {
                        let layer_hd = self.layers[layer_idx].head_dim;
                        let ref_idx = (0..first_kv_shared)
                            .rev()
                            .find(|&i| self.layers[i].head_dim == layer_hd)
                            .unwrap_or(first_kv_shared.saturating_sub(1));
                        let target_dev = self.layers[layer_idx].cos.device().clone();
                        let ref_cache = &self.layers[ref_idx].kv_cache;
                        // In graph mode (single-token decode + padded mask set on
                        // the receiver), prefer the donor's FULL padded buffer over
                        // the narrowed `current_kv()` view. The receiver's
                        // `padded_standard_attention` adds a [1,1,1,max_kv_padded]
                        // mask onto the scores; with a narrowed K of width
                        // current_seq_len, scores collapse to that width and
                        // broadcast_add throws a shape mismatch.
                        let use_padded = seq_len == 1
                            && self.layers[layer_idx].padded_mask.is_some()
                            && ref_cache.k_buffer().is_some();
                        let kv = if use_padded {
                            let k = ref_cache.k_buffer().unwrap();
                            let v = ref_cache.v_buffer().unwrap();
                            let k = if k.device().same_device(&target_dev) {
                                k
                            } else {
                                k.to_device(&target_dev)?
                            };
                            let v = if v.device().same_device(&target_dev) {
                                v
                            } else {
                                v.to_device(&target_dev)?
                            };
                            Some((k, v))
                        } else if let Some((k, v)) = ref_cache.current_kv() {
                            let k = if k.device().same_device(&target_dev) {
                                k.clone()
                            } else {
                                k.to_device(&target_dev)?
                            };
                            let v = if v.device().same_device(&target_dev) {
                                v.clone()
                            } else {
                                v.to_device(&target_dev)?
                            };
                            Some((k, v))
                        } else {
                            None
                        };
                        (kv, Some(ref_idx))
                    } else {
                        (None, None)
                    };

                // gemma4 bounded-SWA: windowed SpecKvCache layers use the bounded
                // SWA mask (matches their shortened K); all others use the full mask.
                // A period-classified model gives each layer the mask of its own window.
                let own_mask = if seq_len > 1 && self.config.swa_period.is_some() {
                    let window = self.layers[layer_idx].prefill_mask_window;
                    Some(self.make_layer_mask(seq_len, index_pos, current_cuda_idx, window)?)
                } else {
                    None
                };
                let layer_mask = if own_mask.is_some() {
                    own_mask.as_ref()
                } else if self.layers[layer_idx].kv_cache.is_windowed() {
                    mask_swa.as_ref()
                } else {
                    mask.as_ref()
                };
                // Borrow donor's q8 cache (immutable) AND mutate the shared
                // layer simultaneously. split_at_mut gives us the mutable
                // tail (containing layer_idx) plus immutable head (containing
                // donor index, which is always < first_kv_shared <= layer_idx).
                if let Some(rid) = ref_idx_opt {
                    let (head, tail) = self.layers.split_at_mut(layer_idx);
                    #[cfg(feature = "gpu")]
                    let donor_q8 = head[rid].q8_kv();
                    // Lend the shared layer read access to the donor's F16
                    // KV store (gemma4 CPU). The donor (a real layer < layer_idx,
                    // in `head`) already maintains it for its own decode; shared
                    // layers scan it in F16 (half the F32 `shared_kv` bytes, and a
                    // fused single-pass vs a 4-op matmul chain) - measured
                    // ~2.5x faster per global-layer call. split_at_mut proves
                    // rid != layer_idx so the immutable borrow is sound.
                    let donor_f16 = head[rid].cpu_f16_kv.as_ref();
                    let layer = &mut tail[0];
                    hidden = layer.forward(
                        &hidden,
                        layer_mask,
                        index_pos,
                        ple_on_dev.as_ref(),
                        shared_kv,
                        donor_f16,
                        #[cfg(feature = "gpu")]
                        donor_q8,
                    )?;
                } else {
                    let layer = &mut self.layers[layer_idx];
                    hidden = layer.forward(
                        &hidden,
                        layer_mask,
                        index_pos,
                        ple_on_dev.as_ref(),
                        shared_kv,
                        None,
                        #[cfg(feature = "gpu")]
                        None,
                    )?;
                }
                note_if_non_finite(&format!("layer {layer_idx}"), &hidden);
            }

            seg_start = seg_end;
        }
        self.tp_forget()?;
        Ok((hidden, on_cuda))
    }

    /// The last stage of a forward: the final norm and the output projection of `hidden`,
    /// every position's logits when `all_positions`, else the last one's.
    pub(super) fn output_stage(
        &mut self,
        mut hidden: Tensor,
        mut on_cuda: bool,
        seq_len: usize,
        all_positions: bool,
    ) -> Result<Tensor> {
        // -- Final norm + output projection --------------------------
        // The output stage must not read the hidden state while the last layer's GPU is
        // still writing it: the copy would take partially-written data off a running
        // compute stream and produce garbage logits every token (the gemma4:26b
        // multi-GPU+CPU incoherence). Where the read crosses to another GPU, `to_device`
        // now establishes that with an event and no barrier is needed here. What remains
        // is the case with no recorded projection device, where the output stage may pull
        // the hidden to the CPU: that read leaves the device entirely, so it is held here.
        // Same-device output stages are stream-ordered and were never the issue - an
        // unconditional sync per token serializes CPU<->GPU launch-ahead and cost
        // moondream ~46% (434->233 tok/s, caught by a campaign).
        #[cfg(feature = "gpu")]
        if let crate::tensor::DeviceLocation::Cuda { gpu_id } = hidden.device().location() {
            if self.output_proj_dev.is_none() {
                if let Some(src) = self.cuda_devices.get(&gpu_id) {
                    src.synchronize()?;
                }
            }
        }
        // Transfer to the GPU that holds the output projection (if available)
        // Move the hidden state to whichever device holds the output projection.
        // This may be a DIFFERENT GPU than the last layer's (the lm_head can be
        // placed on an idle GPU when the primary is full), so transfer whenever
        // the devices differ - not just when hidden is off-CUDA.
        if let Some(dev) = self.output_proj_dev.as_ref() {
            if !hidden.device().same_device(dev) {
                hidden = hidden.to_device(dev)?;
            }
            on_cuda = true;
        }
        // Final projection. When `all_positions` is true, project every
        // input position (for speculative verification). When false, the
        // standard last-position-only path for single-token decode.
        //
        // For the multi-position case, the reference `matmul` doesn't broadcast
        // 3D x 2D, so we flatten `[1, seq_len, hidden]` -> `[seq_len, hidden]`
        // before the matmul and reshape the result back to 3D.
        if self.capture_feature {
            // EAGLE feature hook: the pre-final-norm last-layer hidden IS
            // the draft-head feature. all_positions -> every position (the verify
            // forward needs them all); else the last token (the draft cycle).
            // F32 to match the head's compute dtype. Off by default -> no-op.
            let f = if all_positions {
                hidden.clone()
            } else {
                hidden.i((.., seq_len - 1, ..))?
            };
            self.last_feature = Some(f.to_dtype(crate::tensor::DType::F32)?);
        }
        let mut logits = if let Some(oh) = self.output_hyper.as_ref() {
            // The final mixer is the output norm: it mixes the streams, then the head reads
            // the result.
            let (b, s, _) = hidden.dims3()?;
            let rows = if all_positions {
                hidden.clone()
            } else {
                hidden.narrow(1, s - 1, 1)?
            };
            let rows = rows.to_device(&oh.device())?;
            let (normed, _) = oh.mix(&rows)?;
            let n = normed.dim(1)?;
            let normed = normed.reshape((b * n, self.config.embedding_length))?;
            let out = match (&self.output_proj_cuda_qmm, &self.output_proj_dev) {
                (Some(proj), Some(dev)) => proj.forward(&normed.to_device(dev)?)?,
                _ => self.output_proj.forward(&normed.to_device(&Device::Cpu)?)?,
            };
            let vocab = out.dim(1)?;
            if all_positions {
                out.reshape((b, n, vocab))?
            } else {
                out
            }
        } else if on_cuda && self.output_proj_cuda_qmm.is_some() {
            // Preferred GPU path: norm in F32 + quantized matmul. The QMatMul
            // dispatches to `mul_mat_vec_q*_K_q8_1` kernels that produce F32
            // output directly, so there are no F16 casts and no cutlass gemm.
            let proj = self.output_proj_cuda_qmm.as_ref().unwrap();
            if all_positions {
                let (b, s, h) = hidden.dims3()?;
                let hidden_2d = hidden
                    .reshape((b * s, h))?
                    .to_dtype(crate::tensor::DType::F32)?;
                let normed = self.final_norm_on_device(&hidden_2d)?;
                let out_2d = proj.forward(&normed)?;
                let vocab = out_2d.dims()[1];
                out_2d.reshape((b, s, vocab))?
            } else {
                // The final norm, then the head: the arithmetic every row of a verify runs.
                let last = hidden.i((.., seq_len - 1, ..))?;
                let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
                let normed = self.final_norm_on_device(&last_f32)?;
                proj.forward(&normed)?
            }
        } else if on_cuda && self.output_proj_cuda.is_some() {
            // Float head: norm, then the F16 matmul against the `[vocab, hidden]` head read
            // in place (a transpose would copy the whole head every call).
            let norm_w = self.output_norm_cuda_weight.as_ref().unwrap();
            let proj = self.output_proj_cuda.as_ref().unwrap();
            if all_positions {
                let (b, s, h) = hidden.dims3()?;
                let hidden_2d = hidden
                    .reshape((b * s, h))?
                    .to_dtype(crate::tensor::DType::F32)?;
                let normed = self.final_norm_on_device(&hidden_2d)?;
                let normed_f16 = normed.to_dtype(crate::tensor::DType::F16)?;
                let out_2d = normed_f16.matmul_t(proj)?;
                let vocab = out_2d.dims()[1];
                out_2d.reshape((b, s, vocab))?
            } else {
                let last = hidden.i((.., seq_len - 1, ..))?;
                let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
                let normed = crate::tensor::ops::rms_norm(
                    &last_f32,
                    norm_w,
                    self.config.rms_norm_eps as f32,
                )?;
                let normed_f16 = normed.to_dtype(crate::tensor::DType::F16)?;
                normed_f16.matmul_t(proj)?
            }
        } else {
            // Fallback: transfer to CPU, norm + matmul on CPU
            if on_cuda {
                hidden = hidden.to_device(&Device::Cpu)?;
                note_if_non_finite("the hidden state downloaded for the CPU head", &hidden);
            }
            let normed = self.output_norm.forward(&hidden)?;
            note_if_non_finite("the final norm, before the head", &normed);
            let src = if all_positions {
                normed
            } else {
                normed.i((.., seq_len - 1, ..))?
            };
            self.output_proj.forward(&src)?
        };
        logits = self.apply_output_bias(logits)?;

        // Ensure F32 dtype before scaling operations (GPU F16 matmul may produce mixed types)
        logits = logits.to_dtype(crate::tensor::DType::F32)?;

        // Logit scaling (Granite: divide by logit_scale)
        if let Some(scale) = self.config.logit_scale {
            logits = (logits / scale)?;
        }

        // Logit softcapping: tanh(logits/cap) * cap (Gemma4: cap=30.0)
        match self.config.final_logit_softcapping {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }
    }

    /// Vision-prefill entry point. Builds the canonical Moondream/LLaVA
    /// embedding sequence `[bos_emb, image_embeds, text_emb]` and runs it
    /// through the transformer at index_pos=0.
    ///
    /// `bos_token`: shape (1, 1) - single BOS token id (usually EOS for
    /// Moondream since it has no separate BOS).
    /// `text_ids`: shape (1, N) - text prompt token ids (without BOS).
    /// `image_embeds`: shape (1, num_patches, embedding_length) - output of
    /// the vision tower + projector. Must match `config.embedding_length`
    /// in the last dim.
    ///
    /// Returns last-token logits, shape (1, vocab). Resets the KV cache as
    /// a side effect of running the transformer at index_pos=0 (caller
    /// must NOT have any pre-existing KV state - this is a fresh prefill).
    pub fn forward_with_image_embeds(
        &mut self,
        bos_token: &Tensor,
        text_ids: &Tensor,
        image_embeds: &Tensor,
    ) -> Result<Tensor> {
        // Embedding table lives on CPU; move all token tensors to CPU
        // before the embedding lookup, then cat with image embeds (also
        // on CPU). The segment loop inside forward_layers_and_output
        // will move hidden onto the model's first GPU on entry.
        // Path A prep: target the embeddings table's actual device
        // (CPU today; future commit moves small tables to GPU 0). bos/text
        // get cast to that device for the lookup; img then matches the
        // resulting emb device so the final cat works on any backend.
        let emb_device = self.embeddings.embeddings().device().clone();
        let to_emb_device = |t: &Tensor| -> Result<Tensor> {
            if t.device().same_device(&emb_device) {
                Ok(t.clone())
            } else {
                t.to_device(&emb_device)
            }
        };
        let bos_local = to_emb_device(bos_token)?;
        let text_local = to_emb_device(text_ids)?;

        let mut bos_emb = self.embeddings.forward(&bos_local)?;
        let mut text_emb = self.embeddings.forward(&text_local)?;
        if let Some(scale) = self.config.embed_scale {
            bos_emb = (bos_emb * scale)?;
            text_emb = (text_emb * scale)?;
        }
        // image_embeds: cast to the same device as bos_emb so the cat below
        // succeeds regardless of where CLIP placed its output.
        let img = if image_embeds.device().same_device(&bos_emb.device()) {
            image_embeds.clone()
        } else {
            image_embeds.to_device(&bos_emb.device())?
        };
        // Match dtype of the text embeddings (F32 default for the substrate's
        // embedding) so cat doesn't fail.
        let img = if img.dtype() != bos_emb.dtype() {
            img.to_dtype(bos_emb.dtype())?
        } else {
            img
        };
        // Last-dim sanity: image embeds must already be projected to the
        // text model's embedding width.
        let img_d = img.dim(crate::tensor::D::Minus1)?;
        if img_d != self.config.embedding_length {
            return Err(crate::tensor::Error::msg(format!(
                "forward_with_image_embeds: image_embeds last dim {} != embedding_length {}",
                img_d, self.config.embedding_length,
            )));
        }
        let hidden = Tensor::cat(&[&bos_emb, &img, &text_emb], 1)?;
        // Vision arches don't use Gemma's PLE per-layer inputs.
        self.forward_spliced_prefill_chunked(hidden)
    }

    /// Chunked prefill over a PRE-EMBEDDED hidden state `[1, seq, emb]`
    /// starting at position 0. The vision/audio splice paths build `hidden`
    /// by concatenating text embeddings with modality embeddings, so they
    /// can't go through the token-id `forward_prefill_chunked`; without
    /// chunking, a long splice (image ≈ 3-4k tokens + question) runs as ONE
    /// monolithic forward whose attention-score transient
    /// (`heads x seq x seq x f32`) OOMs at long context. Mirrors
    /// `forward_prefill_with_chunk`: per-chunk `forward_layers_and_output`
    /// at increasing `index_pos` (first chunk at 0 resets the KV caches,
    /// later chunks append); the last chunk's logits are the next-token
    /// prediction, identical to the monolithic result.
    fn forward_spliced_prefill_chunked(&mut self, hidden: Tensor) -> Result<Tensor> {
        const CHUNK: usize = crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS;
        let seq_len = hidden.dim(1)?;
        if seq_len <= CHUNK {
            return self.forward_layers_and_output(hidden, None, 0, seq_len, false);
        }
        let mut off = 0usize;
        let mut last: Option<Tensor> = None;
        while off < seq_len {
            let n = CHUNK.min(seq_len - off);
            let chunk = hidden.narrow(1, off, n)?;
            last = Some(self.forward_layers_and_output(chunk, None, off, n, false)?);
            off += n;
        }
        last.ok_or_else(|| crate::tensor::Error::msg("empty spliced prefill".to_string()))
    }

    /// Prefill with modality embeds spliced BETWEEN a text prefix and suffix:
    /// `[prefix_ids, embeds, suffix_ids]`. Unlike `forward_with_image_embeds` (which
    /// forces `[bos, embeds, text]`), this lets the caller place the embeds inside a
    /// chat turn - e.g. Ultravox audio: `[<bos>user-header, audio, question+assistant-header]`.
    /// `embeds` must already be `[1, n, embedding_length]` (batch axis present).
    pub fn forward_with_audio_embeds(
        &mut self,
        prefix_ids: &Tensor,
        embeds: &Tensor,
        suffix_ids: &Tensor,
    ) -> Result<Tensor> {
        let emb_device = self.embeddings.embeddings().device().clone();
        let to_emb_device = |t: &Tensor| -> Result<Tensor> {
            if t.device().same_device(&emb_device) {
                Ok(t.clone())
            } else {
                t.to_device(&emb_device)
            }
        };
        // A segment of no tokens contributes no rows.
        let embed = |ids: &Tensor| -> Result<Option<Tensor>> {
            if ids.dim(1)? == 0 {
                return Ok(None);
            }
            let mut e = self.embeddings.forward(&to_emb_device(ids)?)?;
            if let Some(scale) = self.config.embed_scale {
                e = (e * scale)?;
            }
            Ok(Some(e))
        };
        let prefix_emb = embed(prefix_ids)?;
        let suffix_emb = embed(suffix_ids)?;
        let like = prefix_emb.as_ref().or(suffix_emb.as_ref()).ok_or_else(|| {
            crate::tensor::Error::msg(
                "forward_with_audio_embeds: no text around the embeds".to_string(),
            )
        })?;
        let (like_device, like_dtype) = (like.device(), like.dtype());
        let aud = if embeds.device().same_device(&like_device) {
            embeds.clone()
        } else {
            embeds.to_device(&like_device)?
        };
        let aud = if aud.dtype() != like_dtype {
            // CONVERSION: the embeddings join the text rows in the embedding table's dtype.
            aud.to_dtype(like_dtype)?
        } else {
            aud
        };
        let aud_d = aud.dim(crate::tensor::D::Minus1)?;
        if aud_d != self.config.embedding_length {
            return Err(crate::tensor::Error::msg(format!(
                "forward_with_audio_embeds: embeds last dim {} != embedding_length {}",
                aud_d, self.config.embedding_length,
            )));
        }
        let parts: Vec<&Tensor> = prefix_emb
            .iter()
            .chain(std::iter::once(&aud))
            .chain(suffix_emb.iter())
            .collect();
        let hidden = Tensor::cat(&parts, 1)?;
        self.forward_spliced_prefill_chunked(hidden)
    }

    /// Set of CUDA device ordinals the engine should pre-warm custom
    /// kernels for. Includes EVERY detected CUDA device (those that
    /// might receive layers under future placement / lazy-grow), not
    /// just the ones currently holding layers - kernel pre-warm is
    /// cheap and missing a device produces JIT-compile stalls during
    /// the first request that touches it.
    pub fn cuda_device_ordinals(&self) -> std::collections::HashSet<usize> {
        self.cuda_devices.keys().copied().collect()
    }

    /// Set of CUDA device ordinals that ACTUALLY have layers placed on
    /// them. Strictly a subset of `cuda_device_ordinals()`. Use this
    /// when a downstream component (vision tower placement, hetero
    /// planning) needs to know which GPUs are busy with text-model
    /// compute vs which are idle.
    pub fn layer_cuda_ordinals(&self) -> std::collections::HashSet<usize> {
        self.layer_devs
            .iter()
            .filter_map(|d| match d {
                LayerDevice::Cuda(idx) => Some(*idx),
                LayerDevice::Cpu => None,
            })
            .collect()
    }
}

/// Build the `[seq_q, bounded_len]` SWA prefill attention-mask data (1=masked) for
/// a layer whose KV buffer holds `bounded_len` keys at ABSOLUTE positions
/// `[buf_start, buf_start+bounded_len)` (the bounded sliding-window cache).
/// Query local `i` (abs = `index_pos+i`) attends buffer key `j` (abs = `buf_start+j`)
/// iff `abs_j <= abs_i` (causal) AND `abs_j >= abs_i-window` (SWA). The legacy
/// unbounded `make_mask` windowing is exactly the special case `buf_start=0,
/// bounded_len=index_pos+seq_q` (where `j == abs_j`). With buffer = `window+chunk`
/// and slide-keep-last-window, a slid chunk has `buf_start = (index_pos+seq_q) -
/// bounded_len = chunk_start - window`, so every query's full window is in-buffer.
#[cfg_attr(not(test), allow(dead_code))] // wired into make_mask by the sliding-window path
fn swa_bounded_mask_data(
    seq_q: usize,
    index_pos: usize,
    bounded_len: usize,
    buf_start: usize,
    window: usize,
) -> Vec<u8> {
    (0..seq_q)
        .flat_map(|i| {
            (0..bounded_len).map(move |j| {
                let abs_i = i + index_pos;
                let abs_j = buf_start + j;
                u8::from(abs_j > abs_i || abs_j + window < abs_i)
            })
        })
        .collect()
}

#[cfg(test)]
mod swa_mask_tests {
    use super::swa_bounded_mask_data;

    #[test]
    fn bounded_swa_mask_matches_unbounded_when_buf_start_zero() {
        // buf_start=0, bounded_len=index_pos+seq_q -> must equal the legacy
        // windowed make_mask predicate (j == abs_j). Guarantees non-slid /
        // global behavior is unchanged.
        let (seq_q, index_pos, w) = (3usize, 5usize, 4usize);
        let bounded_len = index_pos + seq_q;
        let got = swa_bounded_mask_data(seq_q, index_pos, bounded_len, 0, w);
        let want: Vec<u8> = (0..seq_q)
            .flat_map(|i| {
                (0..bounded_len).map(move |j| {
                    let abs_i = i + index_pos;
                    u8::from(j > abs_i || j + w < abs_i)
                })
            })
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn bounded_swa_mask_windows_correctly_after_slide() {
        // Chunk at index_pos=1536, seq_q=512, window=1024; buffer holds last
        // w+chunk=1536 keys -> buf_start=(1536+512)-1536=512 (abs [512,2048)).
        let (seq_q, index_pos, w, bounded_len) = (512usize, 1536usize, 1024usize, 1536usize);
        let buf_start = (index_pos + seq_q) - bounded_len; // 512
        let m = swa_bounded_mask_data(seq_q, index_pos, bounded_len, buf_start, w);
        let at = |i: usize, j: usize| m[i * bounded_len + j];
        // Query abs 1536 sees the oldest in-window key abs 512 (edge), not a future key.
        assert_eq!(at(0, 0), 0);
        assert_eq!(at(0, 1025), 1); // abs 1537 future -> causal-masked
                                    // Query abs 2047: key abs 1022 older-than-window masked; abs 1023 edge unmasked;
                                    // abs 2047 (current) unmasked.
        assert_eq!(at(511, 510), 1);
        assert_eq!(at(511, 511), 0);
        assert_eq!(at(511, 1535), 0);
    }
}

/// Prompt positions the fit probe reads: the rows just before this many.
const PROMPT_FIT_SPAN: usize = 1024;

impl GenericHeteroTransformer {
    /// The model's fit over the prompt's first rows: its guesses on the last rows before
    /// `PROMPT_FIT_SPAN` against the tokens that followed them, through the output mixer
    /// and the head.
    fn prompt_fit_probe(&self, ids: &[u32], rows: &Tensor) -> Result<()> {
        const ROWS: usize = 32;
        let n = ids.len().min(rows.dim(1)?);
        if n < ROWS + 1 {
            return Ok(());
        }
        let Some(oh) = self.output_hyper.as_ref() else {
            return Ok(());
        };
        let from = n - ROWS - 1;
        let probe = rows.narrow(1, from, ROWS)?.to_device(&oh.device())?;
        let (normed, _) = oh.mix(&probe)?;
        let normed = normed.reshape((ROWS, self.config.embedding_length))?;
        let logits = match (&self.output_proj_cuda_qmm, &self.output_proj_dev) {
            (Some(proj), Some(dev)) => proj.forward(&normed.to_device(dev)?)?,
            _ => self.output_proj.forward(&normed.to_device(&Device::Cpu)?)?,
        };
        let guessed = logits
            .to_dtype(crate::tensor::DType::F32)?
            .argmax_keepdim(1)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let hits = guessed
            .iter()
            .zip(&ids[from + 1..])
            .filter(|(g, t)| g == t)
            .count();
        tracing::info!("target fit over the prompt: {hits} of {ROWS} rows guessed right");
        Ok(())
    }
}
