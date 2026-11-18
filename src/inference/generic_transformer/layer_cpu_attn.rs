//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

/// Bytes the transient scores of one query tile may hold in a multi-row attention that
/// materialises them.
const SCORES_TILE_BYTES: usize = 256 << 20;
/// Shares of a card's free memory one query tile of a materialised attention divides it into:
/// the scores and the copy derived from them (scaled, masked, normalised) live two at a time,
/// and the rest of the layer's step keeps the others.
const SCORES_SHARES_OF_FREE: usize = 4;
/// Copies of a query tile's scores the sink attention keeps alive at once: the scores and the
/// copy derived from them.
const SINK_SCORES_LIVE_COPIES: usize = 2;

/// The additive form of `mask` (1 hides a key) in `dtype`: minus infinity where it hides a key,
/// 0 elsewhere.
/// Added to the scores, it hides the same keys as a select does.
/// It has the mask's shape, so masking a tile of scores makes no copy of the tile's size beside
/// its result.
fn mask_bias(mask: &Tensor, dtype: crate::tensor::DType) -> Result<Tensor> {
    let device = mask.device();
    let fill = |v: f32| -> Result<Tensor> {
        Tensor::new(v, &device)?
            .to_dtype(dtype)?
            .broadcast_as(mask.dims())
    };
    mask.where_cond(&fill(f32::NEG_INFINITY)?, &fill(0.0)?)
}

/// Card bytes the materialised sink attention keeps alive for a prompt of `rows` rows over
/// `keys` keys with `n_head` query heads, at batch 1.
///
/// It is its widest query tile's F32 scores, [`SINK_SCORES_LIVE_COPIES`] times.
/// The tile is the one the attention takes on a card with room to spare.
pub(crate) fn sink_scores_bytes(n_head: usize, rows: usize, keys: usize) -> u64 {
    let per_row = n_head.saturating_mul(keys).max(1) * std::mem::size_of::<f32>();
    let tile = (SCORES_TILE_BYTES / per_row).clamp(1, rows.max(1));
    (tile.saturating_mul(per_row) as u64).saturating_mul(SINK_SCORES_LIVE_COPIES as u64)
}

impl GenericTransformerLayer {
    /// Append the new token(s) `k`/`v` (`[1, n_kv, seq, hd]`, post-RoPE,
    /// pre-F16-append) to the CPU Q8 KV store. Gated to eligible dense GQA
    /// arches on CPU (no V-norm -> excludes gemma4, whose tiny K-norm amplifies
    /// Q8 noise). Any failure disables the store -> transparent F16 fallback.
    pub(super) fn cpu_q8_append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        seq: usize,
        index_pos: usize,
        is_shared: bool,
    ) {
        // gemma4 (V-norm) uses the F16 windowed store (Q8 degenerate at hd512), and so
        // does a sink layer, whose host decode scan folds the sink in; other dense GQA
        // arches use the Q8 store.
        // A layer whose host keeps no F16 store attends over the F-dtype cache.
        let is_gemma = self.attn_v_norm_ones.is_some();
        let f16_store = (is_gemma || self.attn_sinks.is_some()) && self.host_f16_kv;
        // The Q8 store attends over the whole history; a windowed layer keeps the F16 store,
        // and a layer whose configured cache is unquantised keeps no Q8 store.
        let eligible = !is_shared
            && self.alibi_slopes.is_none()
            && (f16_store || (self.host_q8_kv && self.sliding_window.is_none()))
            && self.head_dim.is_multiple_of(32)
            && self.n_kv_head > 0
            && self.n_head.is_multiple_of(self.n_kv_head)
            && matches!(k.device(), crate::tensor::Device::Cpu);
        if !eligible {
            return;
        }
        let (hd, nkv) = (self.head_dim, self.n_kv_head);
        let to_f32 = |t: &Tensor| -> Option<Vec<f32>> {
            t.to_dtype(crate::tensor::DType::F32)
                .ok()
                .and_then(|t| t.flatten_all().ok())
                .and_then(|t| t.to_vec1::<f32>().ok())
        };
        if f16_store {
            if index_pos == 0 {
                self.cpu_f16_kv = None;
            }
            if self.cpu_f16_kv.is_none() {
                let window = self.sliding_window.filter(|&w| w > 0);
                self.cpu_f16_kv = Some(crate::inference::cache::cpu_f16_kv::CpuF16Kv::new(
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                    window,
                ));
            }
            let (kf, vf) = match (to_f32(k), to_f32(v)) {
                (Some(a), Some(b)) if a.len() == nkv * seq * hd && b.len() == nkv * seq * hd => {
                    (a, b)
                }
                _ => {
                    self.cpu_f16_kv = None;
                    return;
                }
            };
            let cache = self.cpu_f16_kv.as_mut().unwrap();
            let mut kt = vec![0f32; nkv * hd];
            let mut vt = vec![0f32; nkv * hd];
            for p in 0..seq {
                for h in 0..nkv {
                    let src = h * seq * hd + p * hd;
                    kt[h * hd..(h + 1) * hd].copy_from_slice(&kf[src..src + hd]);
                    let vs = &vf[src..src + hd];
                    // gemma V-norm (RMSNorm-no-weight, eps 1e-6) - match decode.
                    let inv = if is_gemma {
                        let mut ss = 0f32;
                        for &x in vs {
                            ss += x * x;
                        }
                        1.0 / (ss / hd as f32 + 1e-6).sqrt()
                    } else {
                        1.0
                    };
                    for i in 0..hd {
                        vt[h * hd + i] = vs[i] * inv;
                    }
                }
                if cache.append(&kt, &vt).is_err() {
                    self.cpu_f16_kv = None;
                    return;
                }
            }
            return;
        }
        if index_pos == 0 {
            self.cpu_q8_kv = None;
        } // fresh sequence
        if self.cpu_q8_kv.is_none() {
            self.cpu_q8_kv = Some(crate::inference::cache::cpu_q8_kv::CpuQ8Kv::new(
                self.n_head,
                self.n_kv_head,
                self.head_dim,
            ));
        }
        let (kf, vf) = match (to_f32(k), to_f32(v)) {
            (Some(a), Some(b)) if a.len() == nkv * seq * hd && b.len() == nkv * seq * hd => (a, b),
            _ => {
                self.cpu_q8_kv = None;
                return;
            }
        };
        let cache = self.cpu_q8_kv.as_mut().unwrap();
        if cache.append_prefill(&kf, &vf, seq).is_err() {
            self.cpu_q8_kv = None;
        }
    }

    /// CPU single-token decode attention from the Q8 KV store (half the
    /// KV bytes vs the F16 matmul). Returns `None` (-> F16 fallback) unless
    /// eligible: CPU build, the store is populated, seq==b==1, no mask.
    /// `tanh(scores / cap) * cap` when the layer caps its attention scores, applied after the
    /// scale and before the mask; the scores unchanged otherwise.
    pub(super) fn softcap_scores(&self, att: Tensor) -> Result<Tensor> {
        match self.attn_softcap {
            Some(cap) => {
                let dt = att.dtype();
                (att.to_dtype(DType::F32)?.affine(1.0 / cap, 0.0)?.tanh()? * cap as f64)?
                    .to_dtype(dt)
            }
            None => Ok(att),
        }
    }

    /// The scores `att` `[b, a, c, keys]` with the ALiBi bias added: `a * c` runs over the query
    /// heads, then the queries, the first at absolute position `first`. Unchanged without slopes.
    pub(super) fn alibi_scores(&self, att: Tensor, first: usize) -> Result<Tensor> {
        let Some(slopes) = self.alibi_slopes.as_ref() else {
            return Ok(att);
        };
        let (_, a, c, keys) = att.dims4()?;
        let queries = (a * c) / slopes.len().max(1);
        let mut bias = Vec::with_capacity(a * c * keys);
        for slope in slopes {
            for i in 0..queries {
                let pos = (first + i) as f32;
                bias.extend((0..keys).map(|j| slope * (j as f32 - pos)));
            }
        }
        let bias = Tensor::from_vec(bias, (1, a, c, keys), &att.device())?.to_dtype(att.dtype())?;
        att.broadcast_add(&bias)
    }

    pub(super) fn cpu_q8_attention(
        &self,
        q: &Tensor,
        seq: usize,
        b: usize,
        mask: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        // Compiled in BOTH builds. The canonical binary is the CUDA build
        // run with `--cpu`; without this the whole fused Q8 CPU decode-attention
        // was stubbed to Ok(None) there -> every dense model fell to the O(ctx)
        // standard_attention on CPU (the 2.5K cliff). The impl is device-guarded
        // (only runs when the store is CPU-populated; returns None otherwise).
        {
            if b != 1 || self.attn_softcap.is_some() || self.attn_sinks.is_some() {
                return Ok(None);
            }
            let hd = self.head_dim;
            let scale = self
                .attention_scale
                .unwrap_or_else(|| 1.0 / (hd as f64).sqrt()) as f32;
            //  multi-token spec-decode VERIFY: route the seq>1 masked attention
            // through the SAME Q8 store the single-token decode reads, via
            // `attention_multi`. Two wins: (a) verify becomes Q8-bit-consistent
            // with decode, so PLD/spec re-sampling reproduces the plain-Q8-decode
            // greedy on borderline logits (the F16 standard_attention used a
            // DIFFERENT KV precision than decode, so verified tokens could diverge);
            // (b) it replaces the full-precision F16 scores matmul with the Q8 KV
            // kernel. Gated to SMALL seq (spec-decode K is tiny) so large prefills
            // stay on the F16/flash prefill path, and to `len > seq` (history
            // exists ⟹ a decode-time verify, never an index_pos==0 prefill). The
            // verify mask is the standard causal mask, which `attention_multi`
            // applies internally (token s sees [0 ..= base+s]). gemma (F16 windowed
            // store, Q8 degenerate at hd512) has no `cpu_q8_kv` -> F16 fallback.
            if seq > 1 {
                const MAX_VERIFY_SEQ: usize = 16;
                if mask.is_none() || seq > MAX_VERIFY_SEQ {
                    return Ok(None);
                }
                let cache = match self.cpu_q8_kv.as_ref() {
                    Some(c) if c.len() > seq => c, // base = len-seq > 0 ⟹ verify
                    _ => return Ok(None),
                };
                // q is [1, n_head, seq, hd] (head-major); attention_multi wants
                // token-major [seq, n_head, hd].
                let qtok = q.transpose(1, 2)?.contiguous()?;
                let qf = qtok
                    .to_dtype(crate::tensor::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                if qf.len() != seq * self.n_head * hd {
                    return Ok(None);
                }
                let mut out = vec![0f32; seq * self.n_head * hd];
                cache.attention_multi(&qf, seq, scale, &mut out)?;
                // out is token-major [seq, n_head, hd] -> back to [1, n_head, seq, hd].
                let t =
                    Tensor::from_vec(out, (1, seq, self.n_head, hd), &crate::tensor::Device::Cpu)?
                        .transpose(1, 2)?
                        .contiguous()?;
                return Ok(Some(t));
            }
            if seq != 1 || mask.is_some() {
                return Ok(None);
            }
            // gemma4 (V-norm) uses the F16 windowed store (Q8 degenerate at hd512);
            // it's populated in cpu_q8_append but was previously never read, so
            // gemma4 decode fell through to the F32 fallback matmul whose KV reads
            // (full F32 cache) contend with the weight GEMV at long ctx. Read it
            // here via the F16C fused online-softmax (half the KV bytes, windowed,
            // no scores-tensor materialise). Falls back if not populated.
            if self.cpu_q8_kv.is_none() {
                if let Some(c) = self.cpu_f16_kv.as_ref().filter(|c| !c.is_empty()) {
                    let qf = q
                        .to_dtype(crate::tensor::DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    if qf.len() != self.n_head * hd {
                        return Ok(None);
                    }
                    let mut out = vec![0f32; self.n_head * hd];
                    c.attention(&qf, scale, &mut out)?;
                    let t = Tensor::from_vec(
                        out,
                        (b, self.n_head, 1, hd),
                        &crate::tensor::Device::Cpu,
                    )?;
                    return Ok(Some(t));
                }
                return Ok(None);
            }
            let cache = match self.cpu_q8_kv.as_ref() {
                Some(c) if !c.is_empty() => c,
                _ => return Ok(None),
            };
            let qf = q
                .to_dtype(crate::tensor::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if qf.len() != self.n_head * hd {
                return Ok(None);
            }
            let mut out = vec![0f32; self.n_head * hd];
            // GQA (n_rep>1): grouped-PV flash reads each V row once per kv-head
            // (baseline re-streams V n_repx - 4x at mistral); MHA has no such
            // redundancy so the simpler head-parallel path stays. The knob is a
            // diagnostic A/B lever only (default = grouped ON).
            if self.n_head > self.n_kv_head {
                cache.attention_grouped(&qf, scale, &mut out)?;
            } else {
                cache.attention(&qf, scale, &mut out)?;
            }
            let t = Tensor::from_vec(out, (b, self.n_head, 1, hd), &crate::tensor::Device::Cpu)?;
            Ok(Some(t))
        }
    }

    /// CPU shared-KV-layer decode attention from the DONOR's F16 KV store
    /// (half the KV bytes of the F32 `shared_kv` tensor, and a fused single-pass
    /// online-softmax scan instead of the standard 4-op matmul-chain attention
    /// - reshape.matmul + affine + softmax + matmul, each allocating an
    /// intermediate). gemma4 8B (42 layers, 18 shared) has 3 shared GLOBAL layers
    /// that re-scan the full context every token plus 14 shared SWA layers that
    /// re-scan a 512-window; in aggregate both are a real slice of decode at 2.5K.
    /// The donor (a real layer earlier in the stack) already maintains its F16
    /// store for its own decode - we lend it here. Returns `None` (-> F32
    /// `standard_attention`) unless:
    ///   - opt-out knob unset,
    ///   - CPU build/runtime, single-token decode (seq==b==1), no mask,
    ///   - this is a shared layer,
    ///   - the donor store exists, is populated, and its window EXACTLY matches
    ///     this layer's own window (`None`==global, `Some(w)`==SWA). The equality
    ///     guard makes it impossible for a windowed store to serve a global layer
    ///     (context truncation) or a global store to serve a SWA layer (over-scan).
    /// Byte-for-byte the same math as the donor's own F16 decode (same
    /// `CpuF16Kv::attention`, same scale, same GQA n_rep, same window) -> the F32
    /// `standard_attention` on the already-normed donor K/V within f16 precision
    /// (verified in-process: maxdiff ~1e-5 at hd=512, ctx<=2500).
    pub(super) fn cpu_shared_f16_attention(
        &self,
        q: &Tensor,
        donor_f16: Option<&crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
        is_shared: bool,
        seq: usize,
        b: usize,
        mask: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        // One row is a decode step over the whole store.
        // The rows of a verify each attend the positions up to their own, as their steps did:
        // the donor has already stored every row.
        let verify =
            seq > 1 && mask.is_some() && seq <= crate::inference::engine::decode_step::VERIFY_ROWS;
        if !is_shared || b != 1 || (seq != 1 && !verify) || (seq == 1 && mask.is_some()) {
            return Ok(None);
        }
        if !matches!(q.device(), crate::tensor::Device::Cpu) {
            return Ok(None);
        }
        // This layer's own window (None=global, Some(w)=SWA). Must match the
        // donor store's window exactly (bulletproof vs mismatched donor type).
        let my_window = self.sliding_window.filter(|&w| w > 0);
        let cache = match donor_f16 {
            Some(c) if !c.is_empty() && c.window() == my_window => c,
            _ => return Ok(None),
        };
        let hd = self.head_dim;
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt()) as f32;
        if cache.len() < seq || (verify && cache.len() == seq) {
            return Ok(None);
        }
        let base = cache.len() - seq;
        // `q` is `[1, n_head, seq, hd]`: row `s` takes its own heads.
        let qf = q
            .to_dtype(crate::tensor::DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if qf.len() != self.n_head * seq * hd {
            return Ok(None);
        }
        let mut out = vec![0f32; self.n_head * seq * hd];
        let mut qs = vec![0f32; self.n_head * hd];
        let mut os = vec![0f32; self.n_head * hd];
        for s in 0..seq {
            for h in 0..self.n_head {
                let at = (h * seq + s) * hd;
                qs[h * hd..(h + 1) * hd].copy_from_slice(&qf[at..at + hd]);
            }
            cache.attention_upto(&qs, scale, None, base + s + 1, &mut os)?;
            for h in 0..self.n_head {
                let at = (h * seq + s) * hd;
                out[at..at + hd].copy_from_slice(&os[h * hd..(h + 1) * hd]);
            }
        }
        let t = Tensor::from_vec(out, (b, self.n_head, seq, hd), &crate::tensor::Device::Cpu)?;
        Ok(Some(t))
    }

    /// The rows of a verify through a layer that reads a donor's cache, each attended as the
    /// decode step it stands for: `q` `[1, n_head, rows, d]` over `k` and `v`
    /// `[1, n_kv, base + rows, d]`, row `r` over the first `base + r + 1` rows, the last `w` of
    /// them on a windowed layer.
    ///
    /// The rows of one key-value head share one product of the queries by the keys and one
    /// product of the weights by the values, reading the donor's rows once, in place.
    /// A score is the dot of its query and its key alone, so each row's scores are its step's.
    /// Each row's weights are the softmax of its own scores, the one the step runs.
    /// The weights a row does not read are zero: each output still sums the row's terms in the
    /// step's order from zero, and a zero term leaves the sum as it is.
    fn verify_rows_as_steps(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        base: usize,
        scale: f64,
    ) -> Result<Tensor> {
        use rayon::prelude::*;
        let (_, _, rows, d) = q.dims4()?;
        let n_rep = self.n_head / self.n_kv_head;
        // Row `r` reads the donor's rows `first(r)..base + r + 1`.
        let first = |r: usize| match self.sliding_window {
            Some(w) if w > 0 && base + r + 1 > w => base + r + 1 - w,
            _ => 0,
        };
        let lo = first(0);
        let width = base + rows - lo;
        let kv_dt = k.dtype();
        if let Some(out) = self.verify_rows_f32(q, k, v, base, scale, &first)? {
            return Ok(out);
        }
        let qd = if q.dtype() == kv_dt {
            q.clone()
        } else {
            q.to_dtype(kv_dt)?
        };
        // Each row's query heads, grouped by the key-value head they read, as the step groups
        // them.
        let queries = (0..rows)
            .map(|r| qd.narrow(2, r, 1)?.reshape((1, self.n_kv_head, n_rep, d)))
            .collect::<Result<Vec<_>>>()?;
        let heads = (0..self.n_kv_head)
            .into_par_iter()
            .map(|h| -> Result<Tensor> {
                let kh = k.narrow(1, h, 1)?.narrow(2, lo, width)?;
                let vh = v.narrow(1, h, 1)?.narrow(2, lo, width)?;
                let qh = queries
                    .iter()
                    .map(|q| q.narrow(1, h, 1))
                    .collect::<Result<Vec<_>>>()?;
                let scores = Tensor::cat(&qh.iter().collect::<Vec<_>>(), 2)?.matmul_t(&kh)?;
                let scores = if (scale - 1.0).abs() < 1e-6 {
                    scores
                } else {
                    scores.affine(scale as f32, 0.0)?
                };
                let scores = self
                    .softcap_scores(scores)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                // Each row's weights: the softmax of its own scores, in F32 as the step takes it,
                // and zero past them.
                let mut weights = vec![0f32; rows * n_rep * width];
                for r in 0..rows {
                    let (from, to) = (first(r) - lo, base + r + 1 - lo);
                    for e in 0..n_rep {
                        let at = (r * n_rep + e) * width;
                        let own = &mut weights[at + from..at + to];
                        own.copy_from_slice(&scores[at + from..at + to]);
                        crate::tensor::ops::softmax_row_in_place(own);
                    }
                }
                let weights = Tensor::from_vec(
                    weights,
                    (1, 1, rows * n_rep, width),
                    &crate::tensor::Device::Cpu,
                )?;
                let weights = match kv_dt {
                    DType::F32 => weights,
                    dt => weights.to_dtype(dt)?,
                };
                weights.matmul(&vh)
            })
            .collect::<Result<Vec<_>>>()?;
        let rows_out = (0..rows)
            .map(|r| {
                let parts = heads
                    .iter()
                    .map(|o| o.narrow(2, r * n_rep, n_rep))
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?.reshape((1, self.n_head, 1, d))
            })
            .collect::<Result<Vec<_>>>()?;
        let out = Tensor::cat(&rows_out.iter().collect::<Vec<_>>(), 2)?;
        if out.dtype() == q.dtype() {
            Ok(out)
        } else {
            out.to_dtype(q.dtype())
        }
    }

    /// [`Self::verify_rows_as_steps`] over F32 queries, keys and values on the host, on their
    /// values in place: the step's products, scale and softmax, without a tensor between them.
    /// `None`, with nothing done, for any other operands or a layer that caps its scores.
    ///
    /// Each score is the dot the step's product sums, and is scaled as the step scales it.
    /// Each row's weights are its step's softmax of its own scores, and zero past them.
    /// Each output is the fold over the keys the step's product of weights by values sums.
    fn verify_rows_f32(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        base: usize,
        scale: f64,
        first: &dyn Fn(usize) -> usize,
    ) -> Result<Option<Tensor>> {
        if q.dtype() != DType::F32 || k.dtype() != DType::F32 || self.attn_softcap.is_some() {
            return Ok(None);
        }
        let (_, _, rows, d) = q.dims4()?;
        let n_rep = self.n_head / self.n_kv_head;
        let lo = first(0);
        let width = base + rows - lo;
        let m = rows * n_rep;
        let queries = q.flatten_all()?.to_vec1::<f32>()?;
        // Each row reads the keys `starts[r]..base + r + 1`, counted from `lo`.
        let starts: Vec<usize> = (0..rows).map(|r| first(r) - lo).collect();
        // The donor's rows of each key-value head, read in place.
        let heads = (0..self.n_kv_head)
            .map(|h| -> Result<(Tensor, Tensor)> {
                Ok((
                    k.narrow(1, h, 1)?.narrow(2, lo, width)?,
                    v.narrow(1, h, 1)?.narrow(2, lo, width)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        if heads
            .iter()
            .any(|(kh, vh)| kh.host_f32_in_place().is_none() || vh.host_f32_in_place().is_none())
        {
            return Ok(None);
        }
        if self.sliding_window.is_some_and(|w| w > 0) {
            return self
                .windowed_rows_f32(&queries, &heads, &starts, base - lo, scale, d)
                .map(Some);
        }
        use rayon::prelude::*;
        let outputs: Vec<Vec<f32>> = heads
            .par_iter()
            .enumerate()
            .map(|(h, (kh, vh))| {
                let (keys, values) = (
                    kh.host_f32_in_place().unwrap_or_default(),
                    vh.host_f32_in_place().unwrap_or_default(),
                );
                // The query rows of this key-value head, row-major: row `r`, query head `e`.
                let mut qh = Vec::with_capacity(m * d);
                for r in 0..rows {
                    for e in 0..n_rep {
                        let at = ((h * n_rep + e) * rows + r) * d;
                        qh.extend_from_slice(&queries[at..at + d]);
                    }
                }
                let dots = crate::tensor::dots_by_key_f32(&qh, keys, m, d, width);
                let mut weights = vec![0f32; m * width];
                for (r, &from) in starts.iter().enumerate() {
                    let to = base + r + 1 - lo;
                    for e in 0..n_rep {
                        let i = r * n_rep + e;
                        let own = &mut weights[i * width + from..i * width + to];
                        for (j, w) in own.iter_mut().enumerate() {
                            let s = dots[(from + j) * m + i];
                            // The step scales its scores as `x * scale + 0`, and keeps a scale of 1.
                            *w = if (scale - 1.0).abs() < 1e-6 {
                                s
                            } else {
                                s * scale as f32 + 0.0
                            };
                        }
                        crate::tensor::ops::softmax_row_in_place(own);
                    }
                }
                crate::tensor::gemm_f32_rows(&weights, values, m, width, d)
            })
            .collect();
        let mut out = vec![0f32; self.n_head * rows * d];
        for (h, y) in outputs.iter().enumerate() {
            for r in 0..rows {
                for e in 0..n_rep {
                    let at = ((h * n_rep + e) * rows + r) * d;
                    out[at..at + d].copy_from_slice(&y[(r * n_rep + e) * d..][..d]);
                }
            }
        }
        Ok(Some(Tensor::from_vec(
            out,
            (1, self.n_head, rows, d),
            &crate::tensor::Device::Cpu,
        )?))
    }

    /// The verify rows of a windowed layer in 2 regions of the prefill pool.
    /// The first makes every score: an item is one key of one key-value head, dotted with every
    /// query row that reads the head, as [`crate::tensor::dots_of_key_f32`] sums them.
    /// The second makes every output: an item is one row of the query heads of one key-value head,
    /// each the step's softmax of its own scores and its weighted sum of the values, folded in the
    /// step's order from zero.
    ///
    /// `queries` is `[n_head, rows, d]`; `heads` holds each key-value head's keys and values from
    /// the first key any row reads; row `r` reads keys `starts[r]..held + r + 1` of them.
    fn windowed_rows_f32(
        &self,
        queries: &[f32],
        heads: &[(Tensor, Tensor)],
        starts: &[usize],
        held: usize,
        scale: f64,
        d: usize,
    ) -> Result<Tensor> {
        let rows = starts.len();
        let n_rep = self.n_head / self.n_kv_head;
        let m = rows * n_rep;
        let width = held + rows;
        let slices = heads
            .iter()
            .map(
                |(kh, vh)| match (kh.host_f32_in_place(), vh.host_f32_in_place()) {
                    (Some(keys), Some(values)) => Ok((keys, values)),
                    _ => crate::tensor::bail!("the donor's rows are not F32 on the host"),
                },
            )
            .collect::<Result<Vec<_>>>()?;
        // The query rows of each key-value head, row `r` then query head `e`, laid out for the dots.
        let lanes = (0..self.n_kv_head)
            .map(|h| {
                let mut rows_of_head = Vec::with_capacity(m * d);
                for r in 0..rows {
                    for e in 0..n_rep {
                        let at = ((h * n_rep + e) * rows + r) * d;
                        rows_of_head.extend_from_slice(&queries[at..at + d]);
                    }
                }
                crate::tensor::dot_lanes_f32(&rows_of_head, m, d)
            })
            .collect::<Vec<_>>();
        // Item `h * width + j` holds the dots of key `j` of key-value head `h` with its `m` rows.
        let mut dots = vec![0f32; self.n_kv_head * width * m];
        crate::tensor::quant_cpu::pool_par_chunks_mut_prefill(&mut dots, m, &|item, out| {
            let (h, j) = (item / width, item % width);
            let key = &slices[h].0[j * d..][..d];
            crate::tensor::dots_of_key_f32(&lanes[h], m, d, key, out);
        });
        let scaled = (scale - 1.0).abs() >= 1e-6;
        // Item `h * rows + r` holds the outputs of row `r` of the `n_rep` query heads that read
        // key-value head `h`: each value row is read once for all of them.
        let mut items = vec![0f32; self.n_kv_head * rows * n_rep * d];
        crate::tensor::quant_cpu::pool_par_chunks_mut_prefill(
            &mut items,
            n_rep * d,
            &|item, acc| {
                let (h, r) = (item / rows, item % rows);
                let (from, to) = (starts[r], held + r + 1);
                let span = to - from;
                let mut weights = vec![0f32; n_rep * span];
                for (e, own) in weights.chunks_mut(span).enumerate() {
                    let i = r * n_rep + e;
                    for (j, w) in own.iter_mut().enumerate() {
                        let s = dots[(h * width + from + j) * m + i];
                        // The step scales its scores as `x * scale + 0`, and keeps a scale of 1.
                        *w = if scaled { s * scale as f32 + 0.0 } else { s };
                    }
                    crate::tensor::ops::softmax_row_in_place(own);
                }
                // Each output folds its key terms in order from zero, as the step's product.
                let values = slices[h].1;
                for j in 0..span {
                    let row = &values[(from + j) * d..][..d];
                    for (e, out) in acc.chunks_mut(d).enumerate() {
                        let w = weights[e * span + j];
                        for (o, &x) in out.iter_mut().zip(row) {
                            *o += w * x;
                        }
                    }
                }
            },
        );
        let mut out = vec![0f32; self.n_head * rows * d];
        for h in 0..self.n_kv_head {
            for r in 0..rows {
                for e in 0..n_rep {
                    let at = ((h * n_rep + e) * rows + r) * d;
                    out[at..at + d]
                        .copy_from_slice(&items[((h * rows + r) * n_rep + e) * d..][..d]);
                }
            }
        }
        Tensor::from_vec(out, (1, self.n_head, rows, d), &crate::tensor::Device::Cpu)
    }

    /// CPU prefill flash-attention (online softmax): no `[seq_q x seq_kv]` scores matrix is
    /// materialised. One work item holds the query heads that read the same key/value head,
    /// for a block of queries, and walks the keys a tile at a time: each key tile is read
    /// once for all of those rows, each row keeps a running max, denominator and output.
    /// F32 CPU only; the runtime `q.device()==Cpu` check at the call site gates it.
    /// `sinks`, one logit per query head, join each row's denominator unscaled.
    pub(crate) fn flash_attn_prefill_cpu(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        scale: f32,
        n_rep: usize,
        sinks: Option<&[f32]>,
    ) -> Result<Tensor> {
        let (b, n_head, seq_q, d) = q.dims4()?;
        let (_, n_kv, seq_kv, _) = k.dims4()?;
        let qd = q.flatten_all()?.to_vec1::<f32>()?;
        let kd = k.flatten_all()?.to_vec1::<f32>()?;
        let vd = v.flatten_all()?.to_vec1::<f32>()?;
        // Mask broadcasts to [seq_q, seq_kv] (0 = keep, nonzero = masked). The
        // causal mask is stored u8 - promote to f32 for a uniform read.
        let md = mask
            .flatten_all()?
            .to_dtype(crate::tensor::DType::F32)?
            .to_vec1::<f32>()?;
        // The workers index the mask by row and key: a mask of other keys is refused here.
        if md.len() != seq_q * seq_kv && md.len() != n_head * seq_q * seq_kv {
            crate::tensor::bail!(
                "host attention: a mask of {:?} for {seq_q} rows over {seq_kv} keys",
                mask.dims()
            );
        }
        let mut out = vec![0f32; b * n_head * seq_q * d];
        let job = HostPrefillAttn {
            q: &qd,
            k: &kd,
            v: &vd,
            mask: &md,
            head_independent: md.len() == seq_q * seq_kv,
            seq_q,
            seq_kv,
            d,
            n_rep,
            scale,
            sinks,
        };
        let per_block = (HOST_PREFILL_ROWS_PER_ITEM / n_rep).clamp(1, seq_q.max(1));
        let blocks = seq_q.div_ceil(per_block);
        let dst = HostPrefillOut(out.as_mut_ptr());
        crate::tensor::quant_cpu::prefill_pool_run(n_kv * blocks, &|item| {
            let kvh = item / blocks;
            let q0 = (item % blocks) * per_block;
            let q1 = (q0 + per_block).min(seq_q);
            job.run(kvh, q0, q1, &dst);
        });
        Tensor::from_vec(out, (b, n_head, seq_q, d), &q.device())
    }

    /// Bytes the scores of one query tile may hold on `device`: [`SCORES_TILE_BYTES`], less on
    /// a card whose free memory gives a tile a smaller share.
    fn scores_budget(device: &crate::tensor::Device) -> usize {
        crate::inference::place::vram_manager::free_now(device).map_or(SCORES_TILE_BYTES, |free| {
            SCORES_TILE_BYTES.min(free as usize / SCORES_SHARES_OF_FREE)
        })
    }

    /// Attention of a layer with per-head sinks: `q` `[b, n_head, seq, hd]` against the
    /// first `kv_len` positions of `k`/`v` `[b, n_kv, >= kv_len, hd]`, `mask` `[seq, kv_len]`
    /// u8 (1 hides a key) or `None` at decode. Returns `[b, n_head, seq, hd]` in `q`'s dtype.
    /// A windowed layer's decode reads its last `sliding_window` keys; its prefill window is
    /// in the mask.
    ///
    /// The graph's decode step reads the length from the device, a card decode reads the
    /// buffers in place, the host decode scans the F16 store, the host prefill runs the
    /// online-softmax walk, each with the sink in its denominator; every other shape
    /// materialises the scores.
    pub(super) fn sink_attention(
        &self,
        q: &Tensor,
        k: Tensor,
        v: Tensor,
        kv_len: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let sinks = self
            .attn_sinks
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("sink attention on a layer without sinks"))?;
        let (b, n_head, seq, hd) = q.dims4()?;
        let n_rep = self.n_head / self.n_kv_head.max(1);
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt());
        let window = self.sliding_window.filter(|&w| w > 0);
        let host = matches!(q.device(), crate::tensor::Device::Cpu);
        // A card decode over an F16 cache runs one split-K launch over the whole buffers, the
        // key count read on the device in the graph's step and given here otherwise; the split
        // count follows the buffers' capacity, so both sum alike.
        // A cache at the model's width attends below in its own width.
        #[cfg(feature = "gpu")]
        if seq == 1
            && mask.is_none()
            && q.device().is_cuda()
            && k.dtype() == crate::tensor::DType::F16
        {
            let pos = match self.graph_step() {
                true => self.graph_pos_i32.as_ref(),
                false => None,
            };
            if pos.is_some() || !self.graph_step() {
                let f16 = |t: &Tensor| t.to_dtype(crate::tensor::DType::F16);
                let nsplit = (self.kv_cache.max_seq_len_padded() / F16_DECODE_KEYS_PER_SPLIT)
                    .clamp(F16_DECODE_MIN_SPLITS, F16_DECODE_MAX_SPLITS);
                if let Some(o) = crate::inference::kernel::fused::flash_decode_split_devkvlen(
                    &f16(q)?.reshape((b, n_head, hd))?,
                    &f16(&k)?,
                    &f16(&v)?,
                    pos,
                    kv_len,
                    Some(sinks),
                    window.unwrap_or(0),
                    scale as f32,
                    nsplit,
                    n_head,
                    self.n_kv_head,
                    hd,
                )? {
                    return o.reshape((b, n_head, 1, hd))?.to_dtype(q.dtype());
                }
            }
        }
        let (k, v) = match k.dims()[2] > kv_len {
            true => (k.narrow(2, 0, kv_len)?, v.narrow(2, 0, kv_len)?),
            false => (k, v),
        };
        if seq == 1 && b == 1 && mask.is_none() && host {
            if let (Some(cache), Some(sinks_host)) =
                (self.cpu_f16_kv.as_ref(), self.attn_sinks_host.as_deref())
            {
                if cache.len() == kv_len && cache.window() == window {
                    let qf = q
                        .to_dtype(crate::tensor::DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let mut out = vec![0f32; n_head * hd];
                    cache.attention_sinks(&qf, scale as f32, Some(sinks_host), &mut out)?;
                    return Tensor::from_vec(out, (1, n_head, 1, hd), &q.device())?
                        .to_dtype(q.dtype());
                }
            }
        }
        if host && q.dtype() == crate::tensor::DType::F32 && b == 1 {
            if let (Some(m), Some(sinks_host)) = (mask, self.attn_sinks_host.as_deref()) {
                let f32 = |t: &Tensor| t.to_dtype(crate::tensor::DType::F32);
                return Self::flash_attn_prefill_cpu(
                    q,
                    &f32(&k)?,
                    &f32(&v)?,
                    m,
                    scale as f32,
                    n_rep,
                    Some(sinks_host),
                );
            }
        }
        sink_attention_matmul(q, k, v, mask, Some(sinks), scale, self.n_kv_head, window)
    }

    pub(super) fn standard_attention(
        &self,
        q: &Tensor,
        k: Tensor,
        v: Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let n_rep = self.n_head / self.n_kv_head;
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt());
        // Skip the affine launch when scale==1 (gemma4 hardcodes
        // f_attention_scale=1.0). Saves one launch per attention call.
        let scale_is_one = (scale - 1.0).abs() < 1e-6;

        // The rows of a verify through a layer reading a donor's cache, each as its decode step.
        if let Some(base) = self
            .donor_rows
            .filter(|_| q.device().is_cpu() && q.dims()[0] == 1 && mask.is_none())
        {
            return self.verify_rows_as_steps(q, &k, &v, base, scale);
        }

        // GPU prefill over a Q8 KV history does not reach this function: it runs as
        // `kernel::fused::flash_prefill_gqa` in `attend`. What arrives here with
        // seq_q > 1 is the F-dtype cache, shared-KV layers and the CPU.
        if q.dims()[2] == 1 && n_rep >= 1 && mask.is_none() {
            // Single-token decode fast path: same matmul shape as the
            // general path below but skips the redundant v.contiguous()
            // (the KV cache returns V as a narrow view of the
            // [1, n_kv_head, max_seq_len, d] buffer, which is non-
            // contiguous; contiguous() copies the full slice every
            // layer - ~0.6 ms/tok on qwen3, ~0.3 ms/tok on phi2's
            // smaller hd=64). the reference matmul handles the non-contiguous
            // V directly.
            //
            // For n_rep > 1 (GQA): reshape Q to group heads instead of
            // expanding K/V - avoids the repeat_kv mem-pass.
            // For n_rep == 1 (e.g. phi2/moondream where n_head=n_kv_head):
            // skip the reshape (it'd be a no-op) - just matmul directly.
            //
            // Extension: was gated on `n_rep > 1` only  -
            // missed the n_rep=1 case (phi2). Adding the n_rep==1 path
            // saves one launch per layer per token on phi2 decode.
            let (b, _n_head, _one, d) = q.dims4()?;
            // gemma3 sliding-window attention: SWA layers attend only to the last
            // `w` keys. llama.cpp's SWA KV cache physically retains only `window`
            // entries (n_kv = get_n_kv() for the SWA context) so the score scan is
            // bounded; here the cache holds the full history, so narrow the K/V
            // views to the window before the matmul. This is BOTH the gemma3-trained
            // windowed attention (global layers have sliding_window=None -> full) AND
            // the fix for the long-ctx decode collapse: without it every SWA layer
            // (5 of every 6) re-scans the whole growing context. seq_kv <= window or
            // global layers ⟹ unchanged.
            let (k, v) = match self.sliding_window {
                Some(w) if w > 0 && k.dims()[2] > w => {
                    let start = k.dims()[2] - w;
                    (k.narrow(2, start, w)?, v.narrow(2, start, w)?)
                }
                _ => (k, v),
            };
            // (Measured DEAD x2: fused single-pass decode kernels here REGRESS.
            // A head-only online-softmax kernel REGRESSED 2.5K -29->-34%. And an
            // hd512 split-K flash-decode - coalesced, nsplitxn_head
            // warps, bit-parity exact - also LOSES on gemma4's actual F32 GPU KV:
            // microbench seq_kv=2500 matmul 30µs vs flash 96µs (3.2x, worsening to
            // 4.4x at 3400), e2e decode 49.5 vs 54.7 tok/s (-9.5%). cuBLAS' SGEMM
            // handles the M=n_rep decode gemm efficiently (the tiny-M underfill only
            // hurts F16, which gemma4 never uses); a single-warp-per-split kernel
            // reads hd512 F32 KV at only ~53 GB/s. Global attention is also just
            // ~1.3% of the token (30µsx8 / 18ms) so it is NOT the 2.5K -21% lever.
            // The 3-op matmul path stays.)
            // dtype-agnostic: with an F16 KV cache (gemma4 lever) k/v arrive as
            // F16; cast Q down to match so the q.kᵀ and p.v matmuls read F16 K/V
            // from HBM (half the bytes). No-op / bit-identical when kv is F32.
            let kv_dt = k.dtype();
            let qd = if q.dtype() == kv_dt {
                q.clone()
            } else {
                q.to_dtype(kv_dt)?
            };
            let att = if n_rep > 1 {
                let q_grouped = qd.reshape((b, self.n_kv_head, n_rep, d))?;
                q_grouped.matmul_t(&k)?
            } else {
                qd.matmul_t(&k)?
            };
            let att = if scale_is_one {
                att
            } else {
                att.affine(scale as f32, 0.0)?
            };
            let att = self.softcap_scores(att)?;
            let att = self.alibi_scores(att, k.dims()[2] - 1)?;
            // Softmax in F32 (ollama parity) then back to kv dtype so p.v reads F16.
            let att = if kv_dt == crate::tensor::DType::F32 {
                crate::tensor::ops::softmax_last_dim(&att)?
            } else {
                crate::tensor::ops::softmax_last_dim(&att.to_dtype(crate::tensor::DType::F32)?)?
                    .to_dtype(kv_dt)?
            };
            let out = att.matmul(&v)?;
            let out = if out.dtype() == q.dtype() {
                out
            } else {
                out.to_dtype(q.dtype())?
            };
            if n_rep > 1 {
                Ok(out.reshape((b, self.n_head, 1, d))?)
            } else {
                Ok(out)
            }
        } else {
            // CPU prefill flash-attention: fused online-softmax single K+V walk,
            // no O(seq²) scores matrix. Gated at RUNTIME to CPU+F32+causal-mask+
            // batch-1 (the long-prompt prefill regime) so it fires on the canonical
            // CUDA binary run with `--cpu`; other cases use the naive path below.
            if matches!(q.device(), crate::tensor::Device::Cpu)
                && q.dtype() == crate::tensor::DType::F32
                && q.dims()[0] == 1
                && self.attn_softcap.is_none()
                && self.alibi_slopes.is_none()
            {
                if let Some(m) = mask {
                    return Self::flash_attn_prefill_cpu(q, &k, &v, m, scale as f32, n_rep, None);
                }
            }
            // General prefill path (seq_q > 1), QUERY-TILED. The [b,n_head,seq,seq]
            // scores are NEVER materialized whole: we walk the query rows in tiles so the
            // transient scores are bounded to [b,n_head,tile,seq_kv], then concatenate the
            // per-tile outputs - bit-identical to the single-shot softmax (each query row's
            // softmax is independent of the others). This keeps the FAST cuBLAS batched
            // GEMMs (q.kᵀ and p.v) for EVERY prefill length and can't OOM, so there is ONE
            // path and no size-threshold fork.
            let k = crate::tensor::ops::repeat_kv(k, n_rep)?;
            let v = crate::tensor::ops::repeat_kv(v, n_rep)?.contiguous()?;
            // dtype-agnostic (see decode fast path): F16 KV -> run the matmuls at F16,
            // softmax in F32. Bit-identical when kv is F32 (all other archs).
            let kv_dt = k.dtype();
            let qd = if q.dtype() == kv_dt {
                q.clone()
            } else {
                q.to_dtype(kv_dt)?
            };
            let kt = k.t()?; // [b,n_head,d,seq_kv] view, reused across tiles
            let seq_q = qd.dims()[2];
            let seq_kv = k.dims()[2];
            // Tile so the transient scores stay under a budget REGARDLESS of seq: tile
            // auto-shrinks as seq grows, and on a card it also shrinks to what the card has
            // free. seq_q <= tile ⇒ a single matmul, identical to before.
            let per_row = (qd.dims()[0] * self.n_head * seq_kv).max(1) * 4;
            let tile = (Self::scores_budget(&qd.device()) / per_row).clamp(1, seq_q);
            let attend = |qt: &Tensor, q_off: usize| -> Result<Tensor> {
                let tlen = qt.dims()[2];
                // Each step replaces the scores it read, so no more than two copies of a
                // tile's scores are alive at once.
                let mut att = qt.matmul(&kt)?; // [b,n_head,tlen,seq_kv]
                if !scale_is_one {
                    att = att.affine(scale as f32, 0.0)?;
                }
                att = self.softcap_scores(att)?;
                att = self.alibi_scores(att, seq_kv - seq_q + q_off)?;
                att = match mask {
                    None => att,
                    Some(m) => {
                        // Slice the mask's query rows to this tile (its query axis is the
                        // 2nd-to-last dim); a broadcast query axis (==1) is left as-is.
                        let qdim = m.dims().len() - 2;
                        let mt = if m.dims()[qdim] == 1 {
                            (*m).clone()
                        } else {
                            m.narrow(qdim, q_off, tlen)?
                        };
                        att.broadcast_add(&mask_bias(&mt, kv_dt)?)?
                    }
                };
                att = if kv_dt == crate::tensor::DType::F32 {
                    crate::tensor::ops::softmax_last_dim(&att)?
                } else {
                    crate::tensor::ops::softmax_last_dim(&att.to_dtype(crate::tensor::DType::F32)?)?
                        .to_dtype(kv_dt)?
                };
                att.matmul(&v) // [b,n_head,tlen,d]
            };
            let out = if seq_q <= tile {
                attend(&qd, 0)?
            } else {
                let mut outs = Vec::with_capacity(seq_q.div_ceil(tile));
                let mut off = 0;
                while off < seq_q {
                    let t = (seq_q - off).min(tile);
                    outs.push(attend(&qd.narrow(2, off, t)?, off)?);
                    off += t;
                }
                let refs: Vec<&Tensor> = outs.iter().collect();
                Tensor::cat(&refs, 2)?
            };
            if out.dtype() == q.dtype() {
                Ok(out)
            } else {
                Ok(out.to_dtype(q.dtype())?)
            }
        }
    }
}

/// Keys the host prefill attention scores at a time, for every row of a work item.
const HOST_PREFILL_KEYS_PER_TILE: usize = 32;
/// Output lanes the host prefill attention accumulates at a time over a key tile.
const HOST_PREFILL_LANES: usize = 32;
/// Rows the host prefill attention runs together, so that each element of a key tile and of a
/// value row it loads serves all of them.
const HOST_PREFILL_ROWS_TOGETHER: usize = 2;
/// Query rows (the heads sharing a key/value head, times queries) one work item of the host
/// prefill attention carries over each key tile.
const HOST_PREFILL_ROWS_PER_ITEM: usize = 64;

/// The output of the host prefill attention, written by work items that own disjoint rows.
struct HostPrefillOut(*mut f32);
// SAFETY: each work item writes only the rows of its own key/value head and query block.
unsafe impl Sync for HostPrefillOut {}

/// One host prefill attention call: `q` `[n_head][seq_q][d]`, `k`/`v` `[n_kv][seq_kv][d]`,
/// `mask` `[seq_q][seq_kv]` or `[n_head][seq_q][seq_kv]` (0 keeps a key).
struct HostPrefillAttn<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    mask: &'a [f32],
    head_independent: bool,
    seq_q: usize,
    seq_kv: usize,
    d: usize,
    n_rep: usize,
    scale: f32,
    sinks: Option<&'a [f32]>,
}

impl HostPrefillAttn<'_> {
    /// The rows of key/value head `kvh` for the queries `q0..q1`, at the widest vector unit.
    fn run(&self, kvh: usize, q0: usize, q1: usize, out: &HostPrefillOut) {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx512() {
            // SAFETY: dispatch_avx512 implies the CPU has every feature `run_avx512` enables.
            return unsafe { self.run_avx512(kvh, q0, q1, out) };
        }
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            // SAFETY: dispatch_avx2 implies the CPU has every feature `run_avx2` enables.
            return unsafe { self.run_avx2(kvh, q0, q1, out) };
        }
        self.rows::<{ HostLanes::Scalar as u8 }>(kvh, q0, q1, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn run_avx2(&self, kvh: usize, q0: usize, q1: usize, out: &HostPrefillOut) {
        self.rows::<{ HostLanes::Avx2 as u8 }>(kvh, q0, q1, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
    unsafe fn run_avx512(&self, kvh: usize, q0: usize, q1: usize, out: &HostPrefillOut) {
        self.rows::<{ HostLanes::Avx512 as u8 }>(kvh, q0, q1, out)
    }

    /// Row `r` of a work item is query `q0 + r % nq` of head `kvh * n_rep + r / nq`, with the
    /// products on the vector unit `W` (a [`HostLanes`]). Every product is an element-wise
    /// fused multiply-add in a fixed order, so the vector width does not change the result.
    #[inline(always)]
    fn rows<const W: u8>(&self, kvh: usize, q0: usize, q1: usize, out: &HostPrefillOut) {
        const BK: usize = HOST_PREFILL_KEYS_PER_TILE;
        let (d, nq, seq_kv) = (self.d, q1 - q0, self.seq_kv);
        let rows = self.n_rep * nq;
        let head = |r: usize| kvh * self.n_rep + r / nq;
        let query = |r: usize| q0 + r % nq;
        let mask_row = |r: usize| {
            let qi = query(r);
            if self.head_independent {
                qi * seq_kv
            } else {
                (head(r) * self.seq_q + qi) * seq_kv
            }
        };
        let mut qs = vec![0f32; rows * d];
        for r in 0..rows {
            let at = (head(r) * self.seq_q + query(r)) * d;
            qs[r * d..(r + 1) * d].copy_from_slice(&self.q[at..at + d]);
        }
        let mut acc = vec![0f32; rows * d];
        let mut max = vec![f32::NEG_INFINITY; rows];
        let mut den = vec![0f32; rows];
        if let Some(s) = self.sinks {
            for r in 0..rows {
                max[r] = s[head(r)];
                den[r] = 1.0;
            }
        }
        // The tile's keys transposed, `kt[c * BK + t]` = element `c` of key `t`.
        let mut kt = vec![0f32; d * BK];
        let mut p = vec![0f32; rows * BK];
        let mut kept = vec![false; rows * BK];
        let mut live = vec![false; rows];
        let mut j0 = 0;
        while j0 < seq_kv {
            let nk = BK.min(seq_kv - j0);
            let mut any = false;
            for r in 0..rows {
                let m = &self.mask[mask_row(r) + j0..mask_row(r) + j0 + nk];
                let kr = &mut kept[r * BK..r * BK + nk];
                let mut n = 0;
                for (t, x) in m.iter().enumerate() {
                    kr[t] = *x == 0.0;
                    n += usize::from(kr[t]);
                }
                live[r] = n > 0;
                any |= live[r];
            }
            if !any {
                j0 += nk;
                continue;
            }
            for t in 0..BK {
                let src = (t < nk).then(|| &self.k[(kvh * seq_kv + j0 + t) * d..][..d]);
                for c in 0..d {
                    kt[c * BK + t] = src.map_or(0.0, |s| s[c]);
                }
            }
            let lived: Vec<usize> = (0..rows).filter(|&r| live[r]).collect();
            let mut pairs = lived.chunks_exact(HOST_PREFILL_ROWS_TOGETHER);
            for pair in pairs.by_ref() {
                Self::scores::<W, 2>([pair[0], pair[1]], &qs, &kt, d, &mut p);
            }
            for &r in pairs.remainder() {
                Self::scores::<W, 1>([r], &qs, &kt, d, &mut p);
            }
            for &r in &lived {
                let kr = &kept[r * BK..r * BK + nk];
                let s = &mut p[r * BK..r * BK + nk];
                let mut top = f32::NEG_INFINITY;
                for t in 0..nk {
                    s[t] *= self.scale;
                    if kr[t] && s[t] > top {
                        top = s[t];
                    }
                }
                let m = max[r].max(top);
                let alpha = (max[r] - m).exp();
                crate::inference::kernel::cpu_decode_exec::exp_sub_max_sum(s, m);
                let mut sum = 0f32;
                for t in 0..nk {
                    if !kr[t] {
                        s[t] = 0.0;
                    }
                    sum += s[t];
                }
                den[r] = den[r] * alpha + sum;
                max[r] = m;
                if alpha != 1.0 {
                    for x in acc[r * d..(r + 1) * d].iter_mut() {
                        *x *= alpha;
                    }
                }
            }
            let vt = &self.v[(kvh * seq_kv + j0) * d..(kvh * seq_kv + j0 + nk) * d];
            let mut pairs = lived.chunks_exact(HOST_PREFILL_ROWS_TOGETHER);
            for pair in pairs.by_ref() {
                Self::values::<W, 2>([pair[0], pair[1]], &p, vt, nk, d, &mut acc);
            }
            for &r in pairs.remainder() {
                Self::values::<W, 1>([r], &p, vt, nk, d, &mut acc);
            }
            j0 += nk;
        }
        for r in 0..rows {
            let inv = if den[r] > 0.0 { 1.0 / den[r] } else { 0.0 };
            let at = (head(r) * self.seq_q + query(r)) * d;
            // SAFETY: row `r` of this work item is written by this work item only, and `at + d`
            // is within the output, which holds `n_head * seq_q * d` floats.
            let o = unsafe { std::slice::from_raw_parts_mut(out.0.add(at), d) };
            for c in 0..d {
                o[c] = acc[r * d + c] * inv;
            }
        }
    }

    /// The raw scores of rows `rs` against the key tile `kt` (`kt[c * BK + t]` = element `c` of
    /// key `t`) into their rows of `p`, `BK` per row, on the vector unit `W`.
    #[inline(always)]
    fn scores<const W: u8, const R: usize>(
        rs: [usize; R],
        qs: &[f32],
        kt: &[f32],
        d: usize,
        p: &mut [f32],
    ) {
        const BK: usize = HOST_PREFILL_KEYS_PER_TILE;
        assert!(kt.len() >= d * BK && rs.iter().all(|&r| qs.len() >= (r + 1) * d));
        assert!(rs.iter().all(|&r| p.len() >= (r + 1) * BK));
        #[cfg(target_arch = "x86_64")]
        if W == HostLanes::Avx512 as u8 {
            // SAFETY: `W` is Avx512 only inside `run_avx512`, whose features the kernel enables;
            // the asserts above bound every index it reads and writes.
            return unsafe { host_scores_avx512(rs, qs, kt, d, p) };
        }
        #[cfg(target_arch = "x86_64")]
        if W == HostLanes::Avx2 as u8 {
            // SAFETY: as above, inside `run_avx2`.
            return unsafe { host_scores_avx2(rs, qs, kt, d, p) };
        }
        for &r in &rs {
            let mut s = [0f32; BK];
            for c in 0..d {
                let qc = qs[r * d + c];
                for t in 0..BK {
                    s[t] = qc.mul_add(kt[c * BK + t], s[t]);
                }
            }
            p[r * BK..(r + 1) * BK].copy_from_slice(&s);
        }
    }

    /// Adds the weights of rows `rs` in `p` times the `nk` value rows of `vt` to their rows of
    /// `acc`, `d` per row, on the vector unit `W`.
    #[inline(always)]
    fn values<const W: u8, const R: usize>(
        rs: [usize; R],
        p: &[f32],
        vt: &[f32],
        nk: usize,
        d: usize,
        acc: &mut [f32],
    ) {
        const BK: usize = HOST_PREFILL_KEYS_PER_TILE;
        assert!(nk <= BK && vt.len() >= nk * d);
        assert!(rs
            .iter()
            .all(|&r| p.len() >= (r + 1) * BK && acc.len() >= (r + 1) * d));
        let whole = d - d % HOST_PREFILL_LANES;
        #[cfg(target_arch = "x86_64")]
        if W == HostLanes::Avx512 as u8 {
            // SAFETY: `W` is Avx512 only inside `run_avx512`, whose features the kernel enables;
            // the asserts above bound every index it reads and writes.
            unsafe { host_values_avx512(rs, p, vt, nk, d, whole, acc) };
        }
        #[cfg(target_arch = "x86_64")]
        if W == HostLanes::Avx2 as u8 {
            // SAFETY: as above, inside `run_avx2`.
            unsafe { host_values_avx2(rs, p, vt, nk, d, whole, acc) };
        }
        let from = if W == HostLanes::Scalar as u8 {
            0
        } else {
            whole
        };
        for &r in &rs {
            for t in 0..nk {
                let pt = p[r * BK + t];
                for c in from..d {
                    acc[r * d + c] = pt.mul_add(vt[t * d + c], acc[r * d + c]);
                }
            }
        }
    }
}

/// The vector unit a host prefill attention kernel runs its products on.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum HostLanes {
    Scalar,
    Avx2,
    Avx512,
}

/// The score and value kernels of the host prefill attention at one vector width: `$n` lanes
/// of type `$v`. Each output element is the same chain of fused multiply-adds whatever the
/// width.
macro_rules! host_prefill_kernels {
    ($scores:ident, $values:ident, $feat:literal, $v:ty, $n:literal,
     $zero:ident, $set1:ident, $load:ident, $store:ident, $fmadd:ident) => {
        /// [`HostPrefillAttn::scores`] on this vector width. The caller bounds the indices.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feat)]
        unsafe fn $scores<const R: usize>(
            rs: [usize; R],
            qs: &[f32],
            kt: &[f32],
            d: usize,
            p: &mut [f32],
        ) {
            use std::arch::x86_64::*;
            const BK: usize = HOST_PREFILL_KEYS_PER_TILE;
            const V: usize = BK / $n;
            unsafe {
                let mut s: [[$v; V]; R] = [[$zero(); V]; R];
                for c in 0..d {
                    let kc = kt.as_ptr().add(c * BK);
                    let mut k: [$v; V] = [$zero(); V];
                    for j in 0..V {
                        k[j] = $load(kc.add(j * $n));
                    }
                    for i in 0..R {
                        let q = $set1(*qs.get_unchecked(rs[i] * d + c));
                        for j in 0..V {
                            s[i][j] = $fmadd(q, k[j], s[i][j]);
                        }
                    }
                }
                for i in 0..R {
                    for j in 0..V {
                        $store(p.as_mut_ptr().add(rs[i] * BK + j * $n), s[i][j]);
                    }
                }
            }
        }

        /// [`HostPrefillAttn::values`] on this vector width, over the first `whole` elements
        /// of each row (a multiple of the lanes). The caller bounds the indices.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feat)]
        unsafe fn $values<const R: usize>(
            rs: [usize; R],
            p: &[f32],
            vt: &[f32],
            nk: usize,
            d: usize,
            whole: usize,
            acc: &mut [f32],
        ) {
            use std::arch::x86_64::*;
            const BK: usize = HOST_PREFILL_KEYS_PER_TILE;
            const LANES: usize = HOST_PREFILL_LANES;
            const V: usize = LANES / $n;
            unsafe {
                let mut c0 = 0;
                while c0 < whole {
                    let mut lane: [[$v; V]; R] = [[$zero(); V]; R];
                    for i in 0..R {
                        let a = acc.as_ptr().add(rs[i] * d + c0);
                        for j in 0..V {
                            lane[i][j] = $load(a.add(j * $n));
                        }
                    }
                    for t in 0..nk {
                        let vr = vt.as_ptr().add(t * d + c0);
                        let mut v: [$v; V] = [$zero(); V];
                        for j in 0..V {
                            v[j] = $load(vr.add(j * $n));
                        }
                        for i in 0..R {
                            let pt = $set1(*p.get_unchecked(rs[i] * BK + t));
                            for j in 0..V {
                                lane[i][j] = $fmadd(pt, v[j], lane[i][j]);
                            }
                        }
                    }
                    for i in 0..R {
                        let a = acc.as_mut_ptr().add(rs[i] * d + c0);
                        for j in 0..V {
                            $store(a.add(j * $n), lane[i][j]);
                        }
                    }
                    c0 += LANES;
                }
            }
        }
    };
}

host_prefill_kernels!(
    host_scores_avx2,
    host_values_avx2,
    "avx2,fma,f16c,bmi2",
    __m256,
    8,
    _mm256_setzero_ps,
    _mm256_set1_ps,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_fmadd_ps
);
host_prefill_kernels!(
    host_scores_avx512,
    host_values_avx512,
    "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni",
    __m512,
    16,
    _mm512_setzero_ps,
    _mm512_set1_ps,
    _mm512_loadu_ps,
    _mm512_storeu_ps,
    _mm512_fmadd_ps
);

/// Keys each split of a card decode over an F16 cache covers at full capacity, and the
/// bounds on the split count: enough splits to fill the card at short context, few enough
/// that the combine stays cheap.
pub(super) const F16_DECODE_KEYS_PER_SPLIT: usize = 128;
pub(super) const F16_DECODE_MIN_SPLITS: usize = 32;
pub(super) const F16_DECODE_MAX_SPLITS: usize = 128;

/// Attention over materialised scores: `q` `[b, n_head, seq, hd]`,
/// `k`/`v` `[b, n_kv_head, kv, hd]`, `mask` `[seq, kv]` u8 with 1 where a key is hidden.
/// With no mask and one query row, `window` keeps the last `window` keys.
/// Each head's sink joins its softmax denominator when `sinks` is given.
/// Scores and the softmax run in F32.
/// The query rows are tiled so the scores stay within the budget `standard_attention` uses.
/// Returns `[b, n_head, seq, hd]` in `q`'s dtype.
#[allow(clippy::too_many_arguments)]
pub(super) fn sink_attention_matmul(
    q: &Tensor,
    k: Tensor,
    v: Tensor,
    mask: Option<&Tensor>,
    sinks: Option<&Tensor>,
    scale: f64,
    n_kv_head: usize,
    window: Option<usize>,
) -> Result<Tensor> {
    use crate::tensor::DType;
    let (b, n_head, seq, hd) = q.dims4()?;
    let (k, v) = match window {
        Some(w) if mask.is_none() && seq == 1 && k.dims()[2] > w => {
            let start = k.dims()[2] - w;
            (k.narrow(2, start, w)?, v.narrow(2, start, w)?)
        }
        _ => (k, v),
    };
    let kv_len = k.dims()[2];
    // Query heads sharing one key/value head sit next to each other, so the grouped
    // reshape reads each K/V head once for its whole group.
    let g = n_head / n_kv_head.max(1);
    let kv_dt = k.dtype();
    let kt = k.t()?;
    let v = v.contiguous()?;
    let sinks = sinks.map(|s| s.to_dtype(DType::F32)).transpose()?;
    // Each step drops the scores it read.
    // At most `SINK_SCORES_LIVE_COPIES` copies of a tile's scores are alive at once.
    let attend = |qt: &Tensor, off: usize| -> Result<Tensor> {
        let t = qt.dims()[2];
        let qt = qt.to_dtype(kv_dt)?.contiguous()?;
        let mut scores = qt
            .reshape((b, n_kv_head, g * t, hd))?
            .matmul(&kt)?
            .to_dtype(DType::F32)?
            .reshape((b, n_head, t, kv_len))?;
        if let Some(m) = mask {
            let qdim = m.dims().len() - 2;
            let mt = if m.dims()[qdim] == 1 {
                m.clone()
            } else {
                m.narrow(qdim, off, t)?
            };
            scores = scores.broadcast_add(&mask_bias(&mt, DType::F32)?)?;
        }
        let probs = match sinks.as_ref() {
            Some(sinks) => {
                crate::tensor::ops::softmax_last_dim_with_sinks(&scores, None, sinks, scale)?
            }
            None => crate::tensor::ops::softmax_last_dim(&(&scores * scale)?)?,
        };
        drop(scores);
        let weights = probs.to_dtype(v.dtype())?;
        drop(probs);
        weights
            .reshape((b, n_kv_head, g * t, kv_len))?
            .matmul(&v)?
            .reshape((b, n_head, t, hd))
    };
    let tiled = |tile: usize| -> Result<Tensor> {
        if seq <= tile {
            return attend(q, 0);
        }
        let mut outs = Vec::with_capacity(seq.div_ceil(tile));
        let mut off = 0;
        while off < seq {
            let t = (seq - off).min(tile);
            outs.push(attend(&q.narrow(2, off, t)?, off)?);
            off += t;
        }
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 2)
    };
    let per_row = (b * n_head * kv_len).max(1) * std::mem::size_of::<f32>();
    // A decoded row is one tile.
    // A prompt's tile holds the scores the card's free memory gives it.
    let tile = match seq {
        0 | 1 => 1,
        _ => (GenericTransformerLayer::scores_budget(&q.device()) / per_row).clamp(1, seq),
    };
    // A tile that runs out of card memory runs again at half its rows, down to 1 row.
    let (out, _) = crate::inference::place::recover::shrink_on_oom(
        "sink attention",
        tile,
        |rows| (rows > 1).then_some(rows / 2),
        |_| false,
        tiled,
    )?;
    out.to_dtype(q.dtype())
}

#[cfg(test)]
mod sink_attention_tests {
    use super::*;
    use crate::tensor::Device;

    /// Deterministic values in `[-amp, amp)`.
    fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    /// Scalar attention with sinks: query row `i` at absolute position `base + i` sees the
    /// keys `j <= base + i` no more than `window - 1` behind it, and each head's sink joins
    /// its denominator. `q` `[n_head][seq][hd]`, `k`/`v` `[n_kv][kv][hd]`.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        (n_head, n_kv, seq, kv, hd): (usize, usize, usize, usize, usize),
        base: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Vec<f32> {
        let g = n_head / n_kv;
        let mut out = vec![0f32; n_head * seq * hd];
        for h in 0..n_head {
            let kh = h / g;
            for i in 0..seq {
                let pos = base + i;
                let visible: Vec<usize> = (0..kv)
                    .filter(|&j| j <= pos && window.is_none_or(|w| pos - j < w))
                    .collect();
                let qrow = &q[(h * seq + i) * hd..(h * seq + i + 1) * hd];
                let logits: Vec<f32> = visible
                    .iter()
                    .map(|&j| {
                        let krow = &k[(kh * kv + j) * hd..(kh * kv + j + 1) * hd];
                        qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let m = logits.iter().cloned().fold(sinks[h], f32::max);
                let denom: f32 =
                    logits.iter().map(|l| (l - m).exp()).sum::<f32>() + (sinks[h] - m).exp();
                for (l, &j) in logits.iter().zip(&visible) {
                    let p = (l - m).exp() / denom;
                    let vrow = &v[(kh * kv + j) * hd..(kh * kv + j + 1) * hd];
                    for c in 0..hd {
                        out[(h * seq + i) * hd + c] += p * vrow[c];
                    }
                }
            }
        }
        out
    }

    fn causal_mask(seq: usize, base: usize, window: Option<usize>) -> Tensor {
        let kv = base + seq;
        let data: Vec<u8> = (0..seq)
            .flat_map(|i| {
                (0..kv).map(move |j| {
                    let pos = base + i;
                    u8::from(j > pos || window.is_some_and(|w| j + w <= pos))
                })
            })
            .collect();
        Tensor::from_slice(&data, (seq, kv), &Device::Cpu).unwrap()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    const SHAPE: (usize, usize, usize) = (4, 2, 8); // query heads, key/value heads, head dim

    fn inputs(seq: usize, kv: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (nh, nkv, hd) = SHAPE;
        (
            values(nh * seq * hd, 1, 1.0),
            values(nkv * kv * hd, 2, 1.0),
            values(nkv * kv * hd, 3, 1.0),
            values(nh, 4, 2.0),
        )
    }

    #[test]
    fn the_materialised_prefill_matches_the_scalar_reference() {
        let (nh, nkv, hd) = SHAPE;
        let (seq, base) = (6usize, 3usize);
        let kv = base + seq;
        let (qv, kvv, vv, sv) = inputs(seq, kv);
        let dev = Device::Cpu;
        let scale = 1.0 / (hd as f32).sqrt();
        for window in [None, Some(3)] {
            let q = Tensor::from_vec(qv.clone(), (1, nh, seq, hd), &dev).unwrap();
            let k = Tensor::from_vec(kvv.clone(), (1, nkv, kv, hd), &dev).unwrap();
            let v = Tensor::from_vec(vv.clone(), (1, nkv, kv, hd), &dev).unwrap();
            let sinks = Tensor::from_vec(sv.clone(), (nh,), &dev).unwrap();
            let mask = causal_mask(seq, base, window);
            let got = sink_attention_matmul(
                &q,
                k,
                v,
                Some(&mask),
                Some(&sinks),
                scale as f64,
                nkv,
                window,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let want = reference(
                &qv,
                &kvv,
                &vv,
                &sv,
                (nh, nkv, seq, kv, hd),
                base,
                window,
                scale,
            );
            assert!(max_diff(&got, &want) < 1e-5, "window {window:?}");
        }
    }

    #[test]
    fn the_host_flash_prefill_folds_the_sink_like_the_reference() {
        let (nh, nkv, hd) = SHAPE;
        let (seq, base) = (5usize, 2usize);
        let kv = base + seq;
        let (qv, kvv, vv, sv) = inputs(seq, kv);
        let dev = Device::Cpu;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = Tensor::from_vec(qv.clone(), (1, nh, seq, hd), &dev).unwrap();
        let k = Tensor::from_vec(kvv.clone(), (1, nkv, kv, hd), &dev).unwrap();
        let v = Tensor::from_vec(vv.clone(), (1, nkv, kv, hd), &dev).unwrap();
        let mask = causal_mask(seq, base, Some(4));
        let got = GenericTransformerLayer::flash_attn_prefill_cpu(
            &q,
            &k,
            &v,
            &mask,
            scale,
            nh / nkv,
            Some(&sv),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let want = reference(
            &qv,
            &kvv,
            &vv,
            &sv,
            (nh, nkv, seq, kv, hd),
            base,
            Some(4),
            scale,
        );
        assert!(max_diff(&got, &want) < 1e-5);
    }

    #[test]
    fn a_windowed_decode_attends_the_last_keys_and_the_sink() {
        let (nh, nkv, hd) = SHAPE;
        let kv = 9usize;
        let (qv, kvv, vv, sv) = inputs(1, kv);
        let dev = Device::Cpu;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = Tensor::from_vec(qv.clone(), (1, nh, 1, hd), &dev).unwrap();
        let k = Tensor::from_vec(kvv.clone(), (1, nkv, kv, hd), &dev).unwrap();
        let v = Tensor::from_vec(vv.clone(), (1, nkv, kv, hd), &dev).unwrap();
        let sinks = Tensor::from_vec(sv.clone(), (nh,), &dev).unwrap();
        let got = sink_attention_matmul(&q, k, v, None, Some(&sinks), scale as f64, nkv, Some(3))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = reference(
            &qv,
            &kvv,
            &vv,
            &sv,
            (nh, nkv, 1, kv, hd),
            kv - 1,
            Some(3),
            scale,
        );
        assert!(max_diff(&got, &want) < 1e-5);
    }

    /// Without sinks a prompt and a decoded row read the keys of their window alone.
    #[test]
    fn a_window_without_sinks_attends_the_last_keys_alone() {
        let (nh, nkv, hd) = SHAPE;
        let dev = Device::Cpu;
        let scale = 1.0 / (hd as f32).sqrt();
        let no_sink = vec![f32::NEG_INFINITY; nh];
        for (seq, base) in [(6usize, 3usize), (1, 8)] {
            let kv = base + seq;
            let (qv, kvv, vv, _) = inputs(seq, kv);
            let q = Tensor::from_vec(qv.clone(), (1, nh, seq, hd), &dev).unwrap();
            let k = Tensor::from_vec(kvv.clone(), (1, nkv, kv, hd), &dev).unwrap();
            let v = Tensor::from_vec(vv.clone(), (1, nkv, kv, hd), &dev).unwrap();
            let mask = (seq > 1).then(|| causal_mask(seq, base, Some(3)));
            let got =
                sink_attention_matmul(&q, k, v, mask.as_ref(), None, scale as f64, nkv, Some(3))
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
            let want = reference(
                &qv,
                &kvv,
                &vv,
                &no_sink,
                (nh, nkv, seq, kv, hd),
                base,
                Some(3),
                scale,
            );
            assert!(max_diff(&got, &want) < 1e-5, "{seq} rows after {base} keys");
        }
    }

    /// A short prompt's scores are counted whole.
    /// A long prompt's are counted at the widest tile the attention takes.
    #[test]
    fn the_sink_scores_count_the_widest_tile_twice() {
        let f32_bytes = std::mem::size_of::<f32>();
        let (heads, rows, keys) = (4usize, 6usize, 9usize);
        let short = SINK_SCORES_LIVE_COPIES * heads * rows * keys * f32_bytes;
        assert_eq!(sink_scores_bytes(heads, rows, keys), short as u64);
        let (heads, keys) = (64usize, 32768usize);
        let per_row = heads * keys * f32_bytes;
        let widest = SINK_SCORES_LIVE_COPIES * (SCORES_TILE_BYTES / per_row) * per_row;
        assert_eq!(sink_scores_bytes(heads, 512, keys), widest as u64);
        assert_eq!(sink_scores_bytes(heads, 4096, keys), widest as u64);
    }

    /// The host prefill attention matches the reference across key tiles, query blocks and a
    /// head wider than one accumulator, with and without a window and a sink.
    #[test]
    fn the_host_flash_prefill_matches_the_reference_across_tiles() {
        let (nh, nkv, hd) = (6usize, 2usize, 80usize);
        let (seq, base) = (37usize, 150usize);
        let kv = base + seq;
        let qv = values(nh * seq * hd, 11, 1.0);
        let kvv = values(nkv * kv * hd, 12, 1.0);
        let vv = values(nkv * kv * hd, 13, 1.0);
        let sv = values(nh, 14, 2.0);
        let dev = Device::Cpu;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = Tensor::from_vec(qv.clone(), (1, nh, seq, hd), &dev).unwrap();
        let k = Tensor::from_vec(kvv.clone(), (1, nkv, kv, hd), &dev).unwrap();
        let v = Tensor::from_vec(vv.clone(), (1, nkv, kv, hd), &dev).unwrap();
        for window in [None, Some(70)] {
            for sinks in [None, Some(&sv)] {
                let mask = causal_mask(seq, base, window);
                let got = GenericTransformerLayer::flash_attn_prefill_cpu(
                    &q,
                    &k,
                    &v,
                    &mask,
                    scale,
                    nh / nkv,
                    sinks.map(|s| s.as_slice()),
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                let silent = vec![f32::NEG_INFINITY; nh];
                let want = reference(
                    &qv,
                    &kvv,
                    &vv,
                    sinks.unwrap_or(&silent),
                    (nh, nkv, seq, kv, hd),
                    base,
                    window,
                    scale,
                );
                assert!(max_diff(&got, &want) < 1e-5, "window {window:?}");
            }
        }
    }

    /// The 512-bit host prefill attention returns the bits of the AVX2 one.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_avx512_host_flash_prefill_has_the_bits_of_the_avx2_one() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (nh, nkv, hd) = (8usize, 2usize, 128usize);
        let (seq, base) = (40usize, 100usize);
        let kv = base + seq;
        let qv = values(nh * seq * hd, 21, 1.0);
        let kvv = values(nkv * kv * hd, 22, 1.0);
        let vv = values(nkv * kv * hd, 23, 1.0);
        let md: Vec<f32> = (0..seq)
            .flat_map(|i| (0..kv).map(move |j| if j > base + i { 1.0 } else { 0.0 }))
            .collect();
        let job = HostPrefillAttn {
            q: &qv,
            k: &kvv,
            v: &vv,
            mask: &md,
            head_independent: true,
            seq_q: seq,
            seq_kv: kv,
            d: hd,
            n_rep: nh / nkv,
            scale: 0.1,
            sinks: None,
        };
        let (mut a, mut b) = (vec![0f32; nh * seq * hd], vec![0f32; nh * seq * hd]);
        let (pa, pb) = (
            HostPrefillOut(a.as_mut_ptr()),
            HostPrefillOut(b.as_mut_ptr()),
        );
        for kvh in 0..nkv {
            // SAFETY: the CPU has AVX-512, which implies every feature both kernels enable.
            unsafe {
                job.run_avx2(kvh, 0, seq, &pa);
                job.run_avx512(kvh, 0, seq, &pb);
            }
        }
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    /// Time of one host prefill attention call at a long-context chunk: a prompt chunk of 512
    /// queries against 10 007 keys, 64 query heads over 8 key/value heads of 128.
    #[test]
    #[ignore = "timing, run by hand"]
    fn host_prefill_attention_timing() {
        let (nh, nkv, hd, seq, kv) = (64usize, 8usize, 128usize, 512usize, 10_007usize);
        let dev = Device::Cpu;
        let q = Tensor::from_vec(values(nh * seq * hd, 1, 1.0), (1, nh, seq, hd), &dev).unwrap();
        let k = Tensor::from_vec(values(nkv * kv * hd, 2, 1.0), (1, nkv, kv, hd), &dev).unwrap();
        let v = Tensor::from_vec(values(nkv * kv * hd, 3, 1.0), (1, nkv, kv, hd), &dev).unwrap();
        let mask = causal_mask(seq, kv - seq, None);
        let scale = 1.0 / (hd as f32).sqrt();
        for _ in 0..3 {
            let t = std::time::Instant::now();
            GenericTransformerLayer::flash_attn_prefill_cpu(
                &q,
                &k,
                &v,
                &mask,
                scale,
                nh / nkv,
                None,
            )
            .unwrap();
            eprintln!("host prefill attention: {:?}", t.elapsed());
        }
    }

    #[test]
    fn without_a_sink_the_flash_prefill_is_unchanged() {
        let (nh, nkv, hd) = SHAPE;
        let (seq, base) = (4usize, 0usize);
        let (qv, kvv, vv, _) = inputs(seq, seq);
        let dev = Device::Cpu;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = Tensor::from_vec(qv.clone(), (1, nh, seq, hd), &dev).unwrap();
        let k = Tensor::from_vec(kvv.clone(), (1, nkv, seq, hd), &dev).unwrap();
        let v = Tensor::from_vec(vv.clone(), (1, nkv, seq, hd), &dev).unwrap();
        let mask = causal_mask(seq, base, None);
        let got = GenericTransformerLayer::flash_attn_prefill_cpu(
            &q,
            &k,
            &v,
            &mask,
            scale,
            nh / nkv,
            None,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        // A sink at -inf contributes nothing: the plain softmax.
        let silent = vec![f32::NEG_INFINITY; nh];
        let want = reference(
            &qv,
            &kvv,
            &vv,
            &silent,
            (nh, nkv, seq, seq, hd),
            base,
            None,
            scale,
        );
        assert!(max_diff(&got, &want) < 1e-5);
    }
}
