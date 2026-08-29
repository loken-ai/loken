//! CPU F16 KV cache for dense decode attention - the gemma4 variant.
//!
//! gemma4 can't use the Q8 KV cache ([`crate::inference::cache::cpu_q8_kv::CpuQ8Kv`]): its global
//! HD=512 layers are numerically degenerate under q8 quantisation (verified
//! /). This stores K and V as **f16** (half the bytes of f32, same
//! precision as the existing Tensor F16 KV path, so NO coherence change - just a
//! zero-alloc, windowed decode), and computes attention with an online softmax
//! over an optional **sliding window** (gemma's local/SWA layers cap their KV at
//! `window`; global layers pass `window=None` and read the full context).
//!
//! Per kv-head, K and V are growable `f16` rows (`head_dim` per token). Decode
//! reads them directly with an online-softmax single pass (no materialised
//! scores matrix), GQA handled inline.

use crate::tensor::Result;
use half::f16;

/// `sum_i q[i] * (f32)k[i]` with F16C: convert 8 packed f16 keys -> f32 and FMA,
/// mirroring ggml's `ggml_vec_dot_f16` (AVX2+F16C). `q` is f32, `k` is f16.
/// Lengths equal (head_dim, a multiple of 8 for gemma's 256/512).
#[inline]
pub(crate) fn f16_dot(q: &[f32], k: &[f16]) -> f32 {
    let n = q.len();
    debug_assert_eq!(k.len(), n);
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let pq = q.as_ptr();
        let pk = k.as_ptr() as *const __m128i; // 8xi16 per 128-bit load
                                               // 4 independent accumulators (32 elems/step) for ILP - matches ggml's
                                               // GGML_F16_ARR=4; ~6% faster than 2 accumulators on this AVX2+F16C path.
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut c = 0usize;
        while c + 32 <= n {
            let b = c / 8;
            let k0 = _mm256_cvtph_ps(_mm_loadu_si128(pk.add(b)));
            let k1 = _mm256_cvtph_ps(_mm_loadu_si128(pk.add(b + 1)));
            let k2 = _mm256_cvtph_ps(_mm_loadu_si128(pk.add(b + 2)));
            let k3 = _mm256_cvtph_ps(_mm_loadu_si128(pk.add(b + 3)));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(c)), k0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(c + 8)), k1, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(c + 16)), k2, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(c + 24)), k3, a3);
            c += 32;
        }
        while c + 8 <= n {
            let kf = _mm256_cvtph_ps(_mm_loadu_si128(pk.add(c / 8)));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(c)), kf, a0);
            c += 8;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let q128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let q128 = _mm_hadd_ps(q128, q128);
        let q128 = _mm_hadd_ps(q128, q128);
        let mut sum = _mm_cvtss_f32(q128);
        while c < n {
            sum += *pq.add(c) * (*k.as_ptr().add(c)).to_f32();
            c += 1;
        }
        return sum;
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        let mut s = 0f32;
        for i in 0..n {
            s += q[i] * k[i].to_f32();
        }
        s
    }
}

/// `o[i] += p * (f32)v[i]` with F16C: convert 8 packed f16 values -> f32, FMA the
/// scalar weight `p` into the f32 accumulator `o`. Online-softmax V accumulate.
#[inline]
pub(crate) fn f16_axpy(p: f32, v: &[f16], o: &mut [f32]) {
    let n = v.len();
    debug_assert_eq!(o.len(), n);
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let pv = v.as_ptr() as *const __m128i;
        let po = o.as_mut_ptr();
        let vp = _mm256_set1_ps(p);
        let mut c = 0usize;
        while c + 8 <= n {
            let vf = _mm256_cvtph_ps(_mm_loadu_si128(pv.add(c / 8)));
            let acc = _mm256_fmadd_ps(vp, vf, _mm256_loadu_ps(po.add(c)));
            _mm256_storeu_ps(po.add(c), acc);
            c += 8;
        }
        while c < n {
            *po.add(c) += p * (*v.as_ptr().add(c)).to_f32();
            c += 1;
        }
        return;
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        for i in 0..n {
            o[i] += p * v[i].to_f32();
        }
    }
}

/// Convert `src` f32 -> f16 and push onto `dst` with F16C (`_mm256_cvtps_ph`,
/// round-to-nearest), 8 lanes at a time. Same bits as `f16::from_f32` per value.
#[inline]
fn extend_f16(dst: &mut Vec<f16>, src: &[f32]) {
    let n = src.len();
    dst.reserve(n);
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let ps = src.as_ptr();
        let base = dst.len();
        dst.set_len(base + n);
        let pd = dst.as_mut_ptr().add(base) as *mut __m128i;
        let mut c = 0usize;
        while c + 8 <= n {
            let packed = _mm256_cvtps_ph(_mm256_loadu_ps(ps.add(c)), _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(pd.add(c / 8), packed);
            c += 8;
        }
        while c < n {
            *(dst.as_mut_ptr().add(base + c)) = f16::from_f32(src[c]);
            c += 1;
        }
        return;
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        dst.extend(src.iter().map(|&x| f16::from_f32(x)));
    }
}

#[derive(Clone)]
pub struct CpuF16Kv {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Sliding-window length (local/SWA layers); `None` = global (full ctx).
    window: Option<usize>,
    k: Vec<Vec<f16>>, // [n_kv_head]; grows by head_dim per token
    v: Vec<Vec<f16>>,
    seq: usize,
}

impl CpuF16Kv {
    pub fn new(n_head: usize, n_kv_head: usize, head_dim: usize, window: Option<usize>) -> Self {
        Self {
            n_head,
            n_kv_head,
            head_dim,
            window,
            k: vec![Vec::new(); n_kv_head],
            v: vec![Vec::new(); n_kv_head],
            seq: 0,
        }
    }

    pub fn reset(&mut self) {
        for b in &mut self.k {
            b.clear();
        }
        for b in &mut self.v {
            b.clear();
        }
        self.seq = 0;
    }

    pub fn len(&self) -> usize {
        self.seq
    }
    pub fn is_empty(&self) -> bool {
        self.seq == 0
    }
    /// True for a full-context (global) store; false for a sliding-window (SWA)
    /// store. `None` window ⟹ global.
    pub fn is_global(&self) -> bool {
        self.window.is_none()
    }
    /// The store's sliding-window length (`None` = global). The shared-KV donor
    /// path lends a donor store to a shared layer ONLY when this exactly
    /// equals the shared layer's own window - so a windowed store can never
    /// serve a global (or differently-windowed) layer and silently truncate or
    /// over-scan the context.
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    /// Roll the cache back to `new_len` tokens (speculative-decode reject).
    /// Each token stored `head_dim` f16 per kv-head (windowing is read-time, so
    /// storage holds every token). Without this, multi-token verify appends
    /// drafts here but the rollback only trimmed the F-dtype cache -> stale draft
    /// positions accumulate and corrupt subsequent attention.
    pub fn trim_to(&mut self, new_len: usize) {
        if new_len >= self.seq {
            return;
        }
        if new_len == 0 {
            self.reset();
            return;
        }
        let keep = new_len * self.head_dim;
        for h in 0..self.n_kv_head {
            self.k[h].truncate(keep);
            self.v[h].truncate(keep);
        }
        self.seq = new_len;
    }

    /// Append one token. `k`/`v` are `[n_kv_head * head_dim]` f32.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<()> {
        debug_assert_eq!(k.len(), self.n_kv_head * self.head_dim);
        debug_assert_eq!(v.len(), self.n_kv_head * self.head_dim);
        let hd = self.head_dim;
        for h in 0..self.n_kv_head {
            extend_f16(&mut self.k[h], &k[h * hd..(h + 1) * hd]);
            extend_f16(&mut self.v[h], &v[h * hd..(h + 1) * hd]);
        }
        self.seq += 1;
        Ok(())
    }

    /// Export the full K/V history as flat f16 in `[n_kv_head, seq, head_dim]`
    /// row-major order (each head's rows are stored contiguously already).
    /// Used to rebuild an F-dtype tensor cache on demand (rare multi-token
    /// calls that need history: PLD verify, session suffix prefill).
    pub fn export_kv(&self) -> (Vec<f16>, Vec<f16>) {
        let cap = self.n_kv_head * self.seq * self.head_dim;
        let mut k = Vec::with_capacity(cap);
        let mut v = Vec::with_capacity(cap);
        for h in 0..self.n_kv_head {
            k.extend_from_slice(&self.k[h]);
            v.extend_from_slice(&self.v[h]);
        }
        (k, v)
    }

    /// GQA attention with online softmax over the (optionally windowed) keys.
    /// `q` is `[n_head * head_dim]` f32; writes `out` `[n_head * head_dim]`.
    pub fn attention(&self, q: &[f32], scale: f32, out: &mut [f32]) -> Result<()> {
        self.attention_sinks(q, scale, None, out)
    }

    /// As `attention`, plus gpt-oss learned per-head **attention sinks**: a raw
    /// (unscaled) logit `sinks[qh]` folded into the softmax denominator only  - 
    /// it shifts the stabilizing max and adds `exp(sink-m)` to the denom, but
    /// contributes 0 to the V accumulation (no key/value row). Matches
    /// `softmax_last_dim_with_sinks` / ggml `ggml_soft_max_add_sinks`. `sinks`
    /// is `[n_head]` f32 (per QUERY head); `None` = the plain softmax above.
    pub fn attention_sinks(
        &self,
        q: &[f32],
        scale: f32,
        sinks: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        let (hd, seq, n_rep) = (self.head_dim, self.seq, self.n_head / self.n_kv_head);
        debug_assert_eq!(q.len(), self.n_head * hd);
        debug_assert_eq!(out.len(), self.n_head * hd);
        if let Some(s) = sinks {
            debug_assert_eq!(s.len(), self.n_head);
        }
        // Window: attend to keys [t0, seq). Local layers cap at `window` most
        // recent; global layers see everything (t0=0).
        let t0 = match self.window {
            Some(w) => seq.saturating_sub(w),
            None => 0,
        };
        // GQA-GROUPED scan: parallelise over KV heads, not query heads. The KV
        // head `kvh` is SHARED by `n_rep` query heads; the old per-query-head loop
        // re-streamed that head's K and V from RAM `n_rep`x (each query head on its
        // own core). Reading each K[t]/V[t] ONCE per group and computing the
        // `n_rep` dots/axpys against it cuts the attention memory traffic ~`n_rep`x
        // - attention decode is memory-bound (its compute is <1ms; the cost is the
        // KV read). Bit-identical per query (same dots, same softmax, same axpy).
        // Run the scan on the SAME spin-pool the FFN GEMVs use, not on rayon.
        // Two live pools fight for the cores: a decode token crosses this scan
        // once per attention layer, and each crossing woke rayon's cold workers
        // and work-stole while the gemv-pool's own workers were still spinning  - 
        // `perf` on a decode showed rayon's steal/wait_until_cold/join plus the
        // kernel's schedule/yield paths, and NO attention symbol at all. One pool
        // for the whole layer is ggml's model. Same partition, same math.
        let n = seq - t0;
        crate::tensor::quant_cpu::pool_par_chunks_mut(out, n_rep * hd, &|kvh, o_grp| {
            let krow_all = &self.k[kvh];
            let vrow_all = &self.v[kvh];
            // Per-query scores [r*n + i] + running max; scratch <= n_rep.n f32
            // (<=40KB @2.5K/n_rep=4, L2-resident).
            let mut scores = vec![0f32; n_rep * n];
            let mut m = vec![f32::NEG_INFINITY; n_rep];
            // PASS 1: read K[t] ONCE, dot with all n_rep queries of the group.
            for (i, t) in (t0..seq).enumerate() {
                let kt = &krow_all[t * hd..(t + 1) * hd];
                for r in 0..n_rep {
                    let qh = kvh * n_rep + r;
                    let d = f16_dot(&q[qh * hd..(qh + 1) * hd], kt) * scale;
                    scores[r * n + i] = d;
                    if d > m[r] {
                        m[r] = d;
                    }
                }
            }
            // exp + optional sink, per query.
            let mut s = vec![0f32; n_rep];
            for r in 0..n_rep {
                let sink = sinks.map(|sk| sk[kvh * n_rep + r]);
                if let Some(sk) = sink {
                    if sk > m[r] {
                        m[r] = sk;
                    }
                }
                let sr = crate::inference::kernel::cpu_decode_exec::exp_sub_max_sum(
                    &mut scores[r * n..(r + 1) * n],
                    m[r],
                );
                s[r] = sr + sink.map(|sk| (sk - m[r]).exp()).unwrap_or(0.0);
            }
            for x in o_grp.iter_mut() {
                *x = 0.0;
            }
            // PASS 3: read V[t] ONCE, axpy into all n_rep output accumulators.
            for (i, t) in (t0..seq).enumerate() {
                let vt = &vrow_all[t * hd..(t + 1) * hd];
                for r in 0..n_rep {
                    f16_axpy(scores[r * n + i], vt, &mut o_grp[r * hd..(r + 1) * hd]);
                }
            }
            for r in 0..n_rep {
                let inv = if s[r] > 0.0 { 1.0 / s[r] } else { 0.0 };
                for x in o_grp[r * hd..(r + 1) * hd].iter_mut() {
                    *x *= inv;
                }
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // F16 KV online-softmax attention must match a naive f32 softmax reference
    // (within f16 round-trip tolerance), including the sliding window.
    #[test]
    fn f16_kv_attention_matches_naive() {
        let (nh, nkv, hd) = (4usize, 2usize, 8usize);
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let seq = 20usize;
        let window = Some(6usize);
        let mut kv = CpuF16Kv::new(nh, nkv, hd, window);
        let mut all_k = vec![]; // [seq][nkv*hd]
        let mut all_v = vec![];
        for _ in 0..seq {
            let kt: Vec<f32> = (0..nkv * hd).map(|_| rng()).collect();
            let vt: Vec<f32> = (0..nkv * hd).map(|_| rng()).collect();
            kv.append(&kt, &vt).unwrap();
            all_k.push(kt);
            all_v.push(vt);
        }
        let q: Vec<f32> = (0..nh * hd).map(|_| rng()).collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut got = vec![0f32; nh * hd];
        kv.attention(&q, scale, &mut got).unwrap();

        // naive reference (f16 round-trip K/V to match precision)
        let n_rep = nh / nkv;
        let t0 = seq - window.unwrap();
        let mut want = vec![0f32; nh * hd];
        for qh in 0..nh {
            let kvh = qh / n_rep;
            let qrow = &q[qh * hd..(qh + 1) * hd];
            let mut scores = vec![];
            for t in t0..seq {
                let mut d = 0f32;
                for i in 0..hd {
                    let kf = f16::from_f32(all_k[t][kvh * hd + i]).to_f32();
                    d += qrow[i] * kf;
                }
                scores.push(d * scale);
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|x| (x - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (idx, t) in (t0..seq).enumerate() {
                let p = exps[idx] / sum;
                for i in 0..hd {
                    let vf = f16::from_f32(all_v[t][kvh * hd + i]).to_f32();
                    want[qh * hd + i] += p * vf;
                }
            }
        }
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "i={i} got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    //  shared-KV donor path: a GLOBAL store (window=None) scanned by a
    // shared global layer must match a naive f32 softmax over the FULL context
    // (no windowing), with GQA n_rep>1 and a gemma-like head_dim (mult of 8).
    // This is the exact call the shared-global layer makes into the donor store.
    #[test]
    fn f16_kv_global_gqa_matches_naive() {
        let (nh, nkv, hd) = (8usize, 2usize, 16usize); // n_rep=4, hd%8==0
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let seq = 300usize; // long enough that softmax stability + full-ctx matter
        let mut kv = CpuF16Kv::new(nh, nkv, hd, None); // GLOBAL
        assert!(kv.is_global());
        assert_eq!(kv.window(), None);
        let mut all_k = vec![];
        let mut all_v = vec![];
        for _ in 0..seq {
            let kt: Vec<f32> = (0..nkv * hd).map(|_| rng()).collect();
            let vt: Vec<f32> = (0..nkv * hd).map(|_| rng()).collect();
            kv.append(&kt, &vt).unwrap();
            all_k.push(kt);
            all_v.push(vt);
        }
        let q: Vec<f32> = (0..nh * hd).map(|_| rng()).collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut got = vec![0f32; nh * hd];
        kv.attention(&q, scale, &mut got).unwrap();

        let n_rep = nh / nkv;
        let mut want = vec![0f32; nh * hd];
        for qh in 0..nh {
            let kvh = qh / n_rep;
            let qrow = &q[qh * hd..(qh + 1) * hd];
            let mut scores = vec![];
            for t in 0..seq {
                let mut d = 0f32;
                for i in 0..hd {
                    d += qrow[i] * f16::from_f32(all_k[t][kvh * hd + i]).to_f32();
                }
                scores.push(d * scale);
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|x| (x - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (idx, t) in (0..seq).enumerate() {
                let p = exps[idx] / sum;
                for i in 0..hd {
                    want[qh * hd + i] += p * f16::from_f32(all_v[t][kvh * hd + i]).to_f32();
                }
            }
        }
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 1e-2,
                "i={i} got {} want {}",
                got[i],
                want[i]
            );
        }
    }
}
