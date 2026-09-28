//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

/// Bytes the transient scores of one query tile may hold in a multi-row attention that
/// materialises them.
const SCORES_TILE_BYTES: usize = 256 << 20;

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
        let is_gemma = self.attn_v_norm_ones.is_some();
        let f16_store = is_gemma || self.attn_sinks.is_some();
        let eligible = !is_shared
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
        let mut kt = vec![0f32; nkv * hd];
        let mut vt = vec![0f32; nkv * hd];
        for p in 0..seq {
            for h in 0..nkv {
                let src = h * seq * hd + p * hd;
                kt[h * hd..(h + 1) * hd].copy_from_slice(&kf[src..src + hd]);
                vt[h * hd..(h + 1) * hd].copy_from_slice(&vf[src..src + hd]);
            }
            if cache.append(&kt, &vt).is_err() {
                self.cpu_q8_kv = None;
                return;
            }
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
        if !is_shared || b != 1 || seq != 1 || mask.is_some() {
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
        let qf = q
            .to_dtype(crate::tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if qf.len() != self.n_head * hd {
            return Ok(None);
        }
        let mut out = vec![0f32; self.n_head * hd];
        cache.attention(&qf, scale, &mut out)?;
        let t = Tensor::from_vec(out, (b, self.n_head, 1, hd), &crate::tensor::Device::Cpu)?;
        Ok(Some(t))
    }

    /// AVX2 f32 dot (two accumulators for ILP) with a scalar fallback. Used by
    /// the flash-attention inner Q.K product, which Rust won't auto-vectorise
    /// (float reduction order is not reassociated without fast-math).
    // NOT cfg-gated. The canonical binary is the CUDA build run with
    // `--cpu`; gating this `not(feature="cuda")` compiled the fused CPU prefill
    // attention OUT of it (the same trap as the decode-attention stub), leaving
    // CPU prefill on the naive O(seq²) scores path. Runtime device check gates it.
    /// `acc[i] += p * v[i]`, a vector at a time. The value-accumulate of the
    /// attention output; scalar it is one multiply-add per element per key.
    #[inline]
    fn faxpy(p: f32, v: &[f32], acc: &mut [f32]) {
        let n = v.len();
        #[cfg(target_feature = "avx2")]
        unsafe {
            use std::arch::x86_64::*;
            let pv = _mm256_set1_ps(p);
            let mut c = 0;
            while c + 8 <= n {
                let a = _mm256_fmadd_ps(
                    pv,
                    _mm256_loadu_ps(v.as_ptr().add(c)),
                    _mm256_loadu_ps(acc.as_ptr().add(c)),
                );
                _mm256_storeu_ps(acc.as_mut_ptr().add(c), a);
                c += 8;
            }
            while c < n {
                *acc.get_unchecked_mut(c) += p * *v.get_unchecked(c);
                c += 1;
            }
        }
        #[cfg(not(target_feature = "avx2"))]
        for i in 0..n {
            acc[i] += p * v[i];
        }
    }

    fn fdot(a: &[f32], b: &[f32]) -> f32 {
        // The scalar fallback iterates rather than indexes, so the length is the wide
        // path's alone.
        #[cfg(target_feature = "avx2")]
        let n = a.len();
        #[cfg(target_feature = "avx2")]
        unsafe {
            use std::arch::x86_64::*;
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut c = 0;
            while c + 16 <= n {
                acc0 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(c)), _mm256_loadu_ps(pb.add(c)), acc0);
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(c + 8)),
                    _mm256_loadu_ps(pb.add(c + 8)),
                    acc1,
                );
                c += 16;
            }
            let acc = _mm256_add_ps(acc0, acc1);
            let q = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            let q = _mm_hadd_ps(q, q);
            let q = _mm_hadd_ps(q, q);
            let mut sum = _mm_cvtss_f32(q);
            while c < n {
                sum += *pa.add(c) * *pb.add(c);
                c += 1;
            }
            sum
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        }
    }

    /// CPU prefill flash-attention (online softmax, ggml-style): per
    /// (head, query) a single streaming pass over KV maintaining a running max
    /// `m`, denominator `s` and accumulator - NO [seq_q x seq_kv] scores matrix
    /// is materialised and there is no separate softmax pass (the naive 3-step
    /// path writes + re-reads the full O(seq²) scores three times). GQA handled
    /// inline (kv head = h / n_rep), so K/V need not be repeat-expanded. F32 CPU
    /// only; the float reduction order differs from the batched softmax so it is
    /// gated + validated for coherence, like the decode-side fused CPU kernels.
    /// NOT cfg-gated (runs on the canonical CUDA binary under `--cpu`); the
    /// runtime `q.device()==Cpu` check at the call site gates it, no-op on GPU.
    /// `sinks`, one logit per query head, join each row's denominator unscaled.
    fn flash_attn_prefill_cpu(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        scale: f32,
        n_rep: usize,
        sinks: Option<&[f32]>,
    ) -> Result<Tensor> {
        let (b, n_head, seq_q, d) = q.dims4()?;
        let (_, _n_kv, seq_kv, _) = k.dims4()?;
        let qd = q.flatten_all()?.to_vec1::<f32>()?;
        let kd = k.flatten_all()?.to_vec1::<f32>()?;
        let vd = v.flatten_all()?.to_vec1::<f32>()?;
        // Mask broadcasts to [seq_q, seq_kv] (0 = keep, nonzero = masked). The
        // causal mask is stored u8 - promote to f32 for a uniform read.
        let md = mask
            .flatten_all()?
            .to_dtype(crate::tensor::DType::F32)?
            .to_vec1::<f32>()?;
        let head_independent = md.len() == seq_q * seq_kv;
        let mut out = vec![0f32; b * n_head * seq_q * d];
        // Parallelise the per-query-row attention on the shared prefill spin-pool
        // (dynamic atomic chunk-grab -> balances the uneven causal-mask rows) rather
        // than rayon. This keeps the whole prefill forward on ONE pool, so rayon's
        // workers park instead of spin-stealing SMT cycles between the quantized
        // GEMM regions.
        crate::tensor::quant_cpu::pool_par_chunks_mut_prefill(&mut out, d, &|hi, o| {
            // out layout is [b=1, n_head, seq_q, d] row-major -> chunk hi = head*seq_q + qi
            let head = hi / seq_q;
            let qi = hi % seq_q;
            let kvh = head / n_rep;
            let qrow = &qd[(head * seq_q + qi) * d..(head * seq_q + qi + 1) * d];
            let mrow_off = if head_independent {
                qi * seq_kv
            } else {
                (head * seq_q + qi) * seq_kv
            };
            // Two passes over the kept keys: first the scaled dots into a scratch
            // (each dot already vectorised) while tracking the row max; then one
            // vectorised exp sweep with that max known, so the exp, the running-
            // max rescale and the value-accumulate all run a vector at a time
            // instead of one float per key. Only unmasked keys enter the scratch,
            // so masking costs no wasted work.
            let mut sc: Vec<f32> = Vec::with_capacity(seq_kv);
            let mut kept: Vec<usize> = Vec::with_capacity(seq_kv);
            let mut m = f32::NEG_INFINITY;
            for j in 0..seq_kv {
                if md[mrow_off + j] != 0.0 {
                    continue;
                }
                let krow = &kd[(kvh * seq_kv + j) * d..(kvh * seq_kv + j + 1) * d];
                let dot = Self::fdot(qrow, krow) * scale;
                sc.push(dot);
                kept.push(j);
                if dot > m {
                    m = dot;
                }
            }
            // The sink raises the stabilising maximum when it is the largest logit.
            let sink = sinks.map(|s| s[head]);
            if let Some(sl) = sink {
                if sl > m {
                    m = sl;
                }
            }
            // exp(sc - m) in place, returns Σ; the same vectorised primitive the
            // decode/lfm2 softmax scans use.
            let s = if m.is_finite() {
                crate::inference::kernel::cpu_decode_exec::exp_sub_max_sum(&mut sc, m)
                    + sink.map_or(0.0, |sl| (sl - m).exp())
            } else {
                0.0
            };
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            let mut acc = vec![0f32; d];
            for (idx, &j) in kept.iter().enumerate() {
                let p = sc[idx];
                let vrow = &vd[(kvh * seq_kv + j) * d..(kvh * seq_kv + j + 1) * d];
                Self::faxpy(p, vrow, &mut acc);
            }
            for c in 0..d {
                o[c] = acc[c] * inv;
            }
        });
        Tensor::from_vec(out, (b, n_head, seq_q, d), &q.device())
    }

    /// Attention of a layer with per-head sinks: `q` `[b, n_head, seq, hd]` against the
    /// appended `k`/`v` `[b, n_kv, kv, hd]`, `mask` `[seq, kv]` u8 (1 hides a key) or `None`
    /// at decode. Returns `[b, n_head, seq, hd]` in `q`'s dtype. A windowed layer's decode
    /// reads its last `sliding_window` keys; its prefill window is in the mask.
    ///
    /// The host decode scans the F16 store, a card decode runs the F16 flash decode, the
    /// host prefill the online-softmax walk, each with the sink in its denominator; every
    /// other shape materialises the scores.
    pub(super) fn sink_attention(
        &self,
        q: &Tensor,
        k: Tensor,
        v: Tensor,
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
        let kv_len = k.dims()[2];
        let host = matches!(q.device(), crate::tensor::Device::Cpu);
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
        if seq == 1 && mask.is_none() && q.device().is_cuda() {
            let kv_start = match window {
                Some(w) if kv_len > w => kv_len - w,
                _ => 0,
            };
            let f16 = |t: &Tensor| t.to_dtype(crate::tensor::DType::F16);
            if let Some(o) = crate::inference::kernel::fused::gptoss_flash_decode_win(
                &f16(q)?,
                &f16(&k)?,
                &f16(&v)?,
                None,
                Some(sinks),
                scale as f32,
                b,
                n_head,
                self.n_kv_head,
                kv_start,
                kv_len,
                hd,
            )? {
                return o.reshape((b, n_head, 1, hd))?.to_dtype(q.dtype());
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
        sink_attention_matmul(q, k, v, mask, sinks, scale, self.n_kv_head, window)
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
            // Tile so the transient scores stay under a fixed budget REGARDLESS of seq:
            // tile auto-shrinks as seq grows (sized from the shape + a memory ceiling, not
            // a magic switch). seq_q <= tile ⇒ a single matmul, identical to before.
            let per_row = (qd.dims()[0] * self.n_head * seq_kv).max(1) * 4;
            let tile = (SCORES_TILE_BYTES / per_row).clamp(1, seq_q);
            let attend = |qt: &Tensor, q_off: usize| -> Result<Tensor> {
                let tlen = qt.dims()[2];
                let att = qt.matmul(&kt)?; // [b,n_head,tlen,seq_kv]
                let att = if scale_is_one {
                    att
                } else {
                    att.affine(scale as f32, 0.0)?
                };
                let att = self.softcap_scores(att)?;
                let att = match mask {
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
                        let mb = mt.broadcast_as(att.shape())?;
                        let neg = if kv_dt == crate::tensor::DType::F32 {
                            self.neg_inf.clone()
                        } else {
                            self.neg_inf.to_dtype(kv_dt)?
                        };
                        mb.where_cond(&neg.broadcast_as(att.dims())?, &att)?
                    }
                };
                let att = if kv_dt == crate::tensor::DType::F32 {
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

/// Attention with per-head sinks over materialised scores: `q` `[b, n_head, seq, hd]`,
/// `k`/`v` `[b, n_kv_head, kv, hd]`, `mask` `[seq, kv]` u8 with 1 where a key is hidden.
/// With no mask and one query row, `window` keeps the last `window` keys. Scores and the
/// sink softmax run in F32; the query rows are tiled so the scores stay within the
/// budget `standard_attention` uses. Returns `[b, n_head, seq, hd]` in `q`'s dtype.
#[allow(clippy::too_many_arguments)]
pub(super) fn sink_attention_matmul(
    q: &Tensor,
    k: Tensor,
    v: Tensor,
    mask: Option<&Tensor>,
    sinks: &Tensor,
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
    let sinks = sinks.to_dtype(DType::F32)?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &q.device())?;
    let attend = |qt: &Tensor, off: usize| -> Result<Tensor> {
        let t = qt.dims()[2];
        let qt = qt.to_dtype(kv_dt)?.contiguous()?;
        let scores = qt
            .reshape((b, n_kv_head, g * t, hd))?
            .matmul(&kt)?
            .to_dtype(DType::F32)?
            .reshape((b, n_head, t, kv_len))?;
        let scores = match mask {
            None => scores,
            Some(m) => {
                let qdim = m.dims().len() - 2;
                let mt = if m.dims()[qdim] == 1 {
                    m.clone()
                } else {
                    m.narrow(qdim, off, t)?
                };
                mt.broadcast_as(scores.shape())?
                    .where_cond(&neg_inf.broadcast_as(scores.dims())?, &scores)?
            }
        };
        let weights =
            crate::tensor::ops::softmax_last_dim_with_sinks(&scores, None, &sinks, scale)?
                .to_dtype(v.dtype())?;
        weights
            .reshape((b, n_kv_head, g * t, kv_len))?
            .matmul(&v)?
            .reshape((b, n_head, t, hd))
    };
    let per_row = (b * n_head * kv_len).max(1) * std::mem::size_of::<f32>();
    let tile = (SCORES_TILE_BYTES / per_row).clamp(1, seq.max(1));
    let out = if seq <= tile {
        attend(q, 0)?
    } else {
        let mut outs = Vec::with_capacity(seq.div_ceil(tile));
        let mut off = 0;
        while off < seq {
            let t = (seq - off).min(tile);
            outs.push(attend(&q.narrow(2, off, t)?, off)?);
            off += t;
        }
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 2)?
    };
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
            let got =
                sink_attention_matmul(&q, k, v, Some(&mask), &sinks, scale as f64, nkv, window)
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
        let got = sink_attention_matmul(&q, k, v, None, &sinks, scale as f64, nkv, Some(3))
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
