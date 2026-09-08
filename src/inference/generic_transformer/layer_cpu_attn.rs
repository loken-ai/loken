//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

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
        // gemma4 (V-norm) uses the F16 windowed store (Q8 degenerate at hd512);
        // other dense GQA arches use the Q8 store.
        let is_gemma = self.attn_v_norm_ones.is_some();
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
        if is_gemma {
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
                    // gemma V-norm (RMSNorm-no-weight, eps 1e-6) - match decode.
                    let vs = &vf[src..src + hd];
                    let mut ss = 0f32;
                    for &x in vs {
                        ss += x * x;
                    }
                    let inv = 1.0 / (ss / hd as f32 + 1e-6).sqrt();
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
            if b != 1 {
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
    fn flash_attn_prefill_cpu(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        scale: f32,
        n_rep: usize,
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
            // exp(sc - m) in place, returns Σ; the same vectorised primitive the
            // decode/lfm2 softmax scans use.
            let s = if m.is_finite() {
                crate::inference::kernel::cpu_decode_exec::exp_sub_max_sum(&mut sc, m)
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

        // `fused_kernels::flash_prefill_f16` is the lever this path needs and is
        // deliberately NOT wired. On a 6000-token prompt the attention is 79% of the
        // prefill and this path runs 2.7x slower than the reference, so the kernel is
        // where the time is. But it answers differently: same prompt, temperature 0, from
        // a freed machine, three restarts each - this chain gives one continuation and the
        // kernel another, both perfectly reproducible. It is close to this attention and
        // not equal to it, and a faster attention that returns something else is a
        // different feature. Wiring it needs a parity harness against this chain first.
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
            {
                if let Some(m) = mask {
                    return Self::flash_attn_prefill_cpu(q, &k, &v, m, scale as f32, n_rep);
                }
            }
            // General prefill path (seq_q > 1), QUERY-TILED. The [b,n_head,seq,seq]
            // scores are NEVER materialized whole: we walk the query rows in tiles so the
            // transient scores are bounded to [b,n_head,tile,seq_kv], then concatenate the
            // per-tile outputs - bit-identical to the single-shot softmax (each query row's
            // softmax is independent of the others). This keeps the FAST cuBLAS batched
            // GEMMs (q.kᵀ and p.v) for EVERY prefill length and can't OOM, so there is ONE
            // path and no size-threshold fork. (A dedicated fused-SRAM flash-prefill kernel
            // was measured SLOWER than this cuBLAS chain at long seq - its only edge was
            // avoiding the [seq,seq] HBM tensor, which tiling already does at cuBLAS speed.)
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
            let tile = ((256usize << 20) / per_row).clamp(1, seq_q);
            let attend = |qt: &Tensor, q_off: usize| -> Result<Tensor> {
                let tlen = qt.dims()[2];
                let att = qt.matmul(&kt)?; // [b,n_head,tlen,seq_kv]
                let att = if scale_is_one {
                    att
                } else {
                    att.affine(scale as f32, 0.0)?
                };
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
