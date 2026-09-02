//! CPU Q8 KV cache for dense decode attention.
//!
//! Mirrors llama.cpp's q8_0 KV cache (reference: `tmp/llama.cpp`
//! llama-kv-cache.cpp / ggml-cpu.c): store K and V as `q8_0`, compute the
//! QK^T scores with the SAME quantized `mul_mat` we already have
//! (`matmul_bytes(Q8_0, ...)` -> `vec_dot_q8_0_q8_0`), and the PV product with a
//! dequant-accumulate reduction. Halves the KV bytes streamed at decode
//! (~+8% measured in `decode_sim_bench`) at q8_0 precision - lossy like the
//! CUDA Q8 path, coherence-validated rather than bit-exact.
//!
//! Layout: per kv-head, K and V are growable buffers of `q8_0` blocks, one
//! `head_dim/32`-block row appended per token (row-major by position). Decode
//! reads them directly; no transpose (we use a PV reduction instead of a
//! second `mul_mat`, which would need V stored transposed - strided to append).

use crate::tensor::quant_cpu::{cast_blocks, from_float_bytes, matmul_bytes, BlockQ8_0};
use crate::tensor::quantized::GgmlDType;
use crate::tensor::Result;
use rayon::prelude::*;

const QK: usize = 32; // q8_0 block size

/// PV accumulate for one q8_0 block: `o[i] += scale * qs[i]` (scale = p_t . d_block).
/// AVX2 sign-extends 8 i8 -> i32 -> f32 and FMAs 8 lanes at a time; the per-key
/// accumulation order into `o` is preserved (vectorised across the 32 block dims,
/// not across keys), so it matches the scalar reduction. `o.len() == qs.len()`,
/// a multiple of 8 (QK=32). Mirrors `cpu_f16_kv::f16_axpy` for the q8_0 store  -
/// the decode-attention PV was the scalar long-ctx hotspot (per-token i8->f32).
#[inline]
fn q8_pv_axpy(scale: f32, qs: &[i8], o: &mut [f32]) {
    let n = qs.len();
    debug_assert_eq!(o.len(), n);
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let vs = _mm256_set1_ps(scale);
        let pq = qs.as_ptr();
        let po = o.as_mut_ptr();
        let mut c = 0usize;
        while c + 8 <= n {
            let i8x8 = _mm_loadl_epi64(pq.add(c) as *const __m128i); // 8 i8 (low 64 bits)
            let f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8x8));
            let acc = _mm256_fmadd_ps(f, vs, _mm256_loadu_ps(po.add(c)));
            _mm256_storeu_ps(po.add(c), acc);
            c += 8;
        }
        while c < n {
            *po.add(c) += scale * (*pq.add(c)) as f32;
            c += 1;
        }
        return;
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        for i in 0..n {
            o[i] += scale * qs[i] as f32;
        }
    }
}

pub struct CpuQ8Kv {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    nb: usize,       // head_dim / 32 blocks per token per head
    k: Vec<Vec<u8>>, // [n_kv_head]; grows by `nb` q8_0 blocks per token
    v: Vec<Vec<u8>>, // [n_kv_head]; same
    seq: usize,
}

impl CpuQ8Kv {
    pub fn new(n_head: usize, n_kv_head: usize, head_dim: usize) -> Self {
        assert!(
            head_dim % QK == 0,
            "head_dim {head_dim} must be a multiple of {QK}"
        );
        Self {
            n_head,
            n_kv_head,
            head_dim,
            nb: head_dim / QK,
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

    /// Roll the cache back to `new_len` tokens (speculative-decode reject of
    /// drafted positions). Each token stored `nb` q8_0 blocks/head (34 bytes
    /// each: f16 scale + 32 i8). Without this, multi-token verify (PLD/EAGLE)
    /// appends drafts here but the rollback only trimmed the F-dtype cache, so
    /// stale draft positions accumulate and corrupt subsequent attention.
    pub fn trim_to(&mut self, new_len: usize) {
        if new_len >= self.seq {
            return;
        }
        if new_len == 0 {
            self.reset();
            return;
        }
        let per_tok = self.nb * (2 + QK); // q8_0 block = f16 d (2) + QK i8
        let keep = new_len * per_tok;
        for h in 0..self.n_kv_head {
            self.k[h].truncate(keep);
            self.v[h].truncate(keep);
        }
        self.seq = new_len;
    }

    /// Append one token. `k`/`v` are `[n_kv_head * head_dim]` f32 (per kv-head
    /// contiguous head_dim slice). Quantizes each head's row to q8_0.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<()> {
        debug_assert_eq!(k.len(), self.n_kv_head * self.head_dim);
        debug_assert_eq!(v.len(), self.n_kv_head * self.head_dim);
        let hd = self.head_dim;
        for h in 0..self.n_kv_head {
            let kb = from_float_bytes(GgmlDType::Q8_0, &k[h * hd..(h + 1) * hd])?;
            self.k[h].extend_from_slice(&kb);
            let vb = from_float_bytes(GgmlDType::Q8_0, &v[h * hd..(h + 1) * hd])?;
            self.v[h].extend_from_slice(&vb);
        }
        self.seq += 1;
        Ok(())
    }

    /// Causal multi-token GQA attention over the SAME Q8 store the seq==1 decode
    /// path reads - the building block for CPU speculative-decode verify.
    ///
    /// Spec-decode (PLD/EAGLE) verify feeds `s_count` tokens through one forward;
    /// today the seq>1 path falls to the F16 `standard_attention` (different KV
    /// precision than decode's Q8), so the verified greedy sequence diverges from
    /// plain Q8 decode on borderline logits. Reading verify from this Q8 store
    /// makes verify and decode bit-consistent. The `s_count` query tokens are the
    /// last `s_count` appended positions; query `s` attends causally to
    /// `[0 ..= (seq - s_count) + s]`. `q` is `[s_count * n_head * head_dim]` f32
    /// (token-major); `out` same layout.
    pub fn attention_multi(
        &self,
        q: &[f32],
        s_count: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let (hd, seq, n_rep) = (self.head_dim, self.seq, self.n_head / self.n_kv_head);
        debug_assert_eq!(q.len(), s_count * self.n_head * hd);
        debug_assert_eq!(out.len(), s_count * self.n_head * hd);
        debug_assert!(s_count <= seq, "s_count {s_count} > seq {seq}");
        let base = seq - s_count; // first new-token position

        // scores[kvh] = [s_count * n_rep, seq] : each query token's group . K^T,
        // computed over the full cache; causal validity applied at softmax time.
        let mut scores: Vec<Vec<f32>> = Vec::with_capacity(self.n_kv_head);
        for h in 0..self.n_kv_head {
            let mut s = vec![0f32; s_count * n_rep * seq];
            for st in 0..s_count {
                let qoff = st * self.n_head * hd + h * n_rep * hd;
                let qg = &q[qoff..qoff + n_rep * hd];
                let dst = &mut s[st * n_rep * seq..(st + 1) * n_rep * seq];
                matmul_bytes(GgmlDType::Q8_0, (n_rep, hd, seq), qg, &self.k[h], dst)?;
            }
            scores.push(s);
        }

        // softmax(scale.scores[..=valid]) + PV, parallel over (query token, head).
        out.par_chunks_mut(hd).enumerate().for_each(|(idx, o)| {
            let st = idx / self.n_head;
            let qh = idx % self.n_head;
            let kvh = qh / n_rep;
            let r = qh % n_rep;
            let valid = base + st + 1; // causal: token `st` sees [0 ..= base+st]
            let row = &scores[kvh][(st * n_rep + r) * seq..(st * n_rep + r) * seq + valid];
            let mut p = vec![0f32; valid];
            let mut mx = f32::NEG_INFINITY;
            for (t, &s) in row.iter().enumerate() {
                p[t] = s * scale;
                if p[t] > mx {
                    mx = p[t];
                }
            }
            let sum = crate::inference::kernel::cpu_decode_exec::exp_sub_max_sum(&mut p, mx);
            let inv = 1.0 / sum;
            for x in o.iter_mut() {
                *x = 0.0;
            }
            let vblocks: &[BlockQ8_0] = cast_blocks(&self.v[kvh]).expect("v q8_0 cast");
            for t in 0..valid {
                let pt = p[t] * inv;
                let bbase = t * self.nb;
                for b in 0..self.nb {
                    let blk = &vblocks[bbase + b];
                    let od = b * QK;
                    q8_pv_axpy(pt * blk.d.to_f32(), &blk.qs, &mut o[od..od + QK]);
                }
            }
        });
        Ok(())
    }

    /// GQA attention. `q` is `[n_head * head_dim]` f32; writes `out`
    /// `[n_head * head_dim]` f32. `scale` multiplies the scores pre-softmax.
    pub fn attention(&self, q: &[f32], scale: f32, out: &mut [f32]) -> Result<()> {
        let (hd, seq, n_rep) = (self.head_dim, self.seq, self.n_head / self.n_kv_head);
        debug_assert_eq!(q.len(), self.n_head * hd);
        debug_assert_eq!(out.len(), self.n_head * hd);

        // QK^T per kv-head: scores[n_rep, seq] = q_group[n_rep,hd] . K[seq,hd]^T
        // via the quantized mul_mat (activation q quantized to q8 internally).
        let mut scores: Vec<Vec<f32>> = Vec::with_capacity(self.n_kv_head);
        for h in 0..self.n_kv_head {
            let mut s = vec![0f32; n_rep * seq];
            // q rows for this kv-head's group are heads [h*n_rep .. h*n_rep+n_rep).
            let qg = &q[h * n_rep * hd..(h * n_rep + n_rep) * hd];
            matmul_bytes(GgmlDType::Q8_0, (n_rep, hd, seq), qg, &self.k[h], &mut s)?;
            scores.push(s);
        }

        // softmax(scale.scores) + PV, parallel over query heads.
        out.par_chunks_mut(hd).enumerate().for_each(|(qh, o)| {
            let kvh = qh / n_rep;
            let r = qh % n_rep;
            let row = &scores[kvh][r * seq..(r + 1) * seq];
            // softmax in place (local copy)
            let mut p = vec![0f32; seq];
            let mut mx = f32::NEG_INFINITY;
            for (t, &s) in row.iter().enumerate() {
                p[t] = s * scale;
                if p[t] > mx {
                    mx = p[t];
                }
            }
            let sum = crate::inference::kernel::cpu_decode_exec::exp_sub_max_sum(&mut p, mx);
            let inv = 1.0 / sum;
            // PV: o[hd] = Σ_t p[t] . dequant(V[kvh][t])
            for x in o.iter_mut() {
                *x = 0.0;
            }
            let vblocks: &[BlockQ8_0] = cast_blocks(&self.v[kvh]).expect("v q8_0 cast");
            for t in 0..seq {
                let pt = p[t] * inv;
                let base = t * self.nb;
                for b in 0..self.nb {
                    let blk = &vblocks[base + b];
                    let od = b * QK;
                    q8_pv_axpy(pt * blk.d.to_f32(), &blk.qs, &mut o[od..od + QK]);
                }
            }
        });
        Ok(())
    }

    /// GQA attention, grouped-PV flash variant (#long-ctx CPU decode lever).
    ///
    /// Same math as [`Self::attention`] but restructures the PV reduction to
    /// kill the GQA V-read redundancy: the baseline parallelises PV over the
    /// `n_head` query heads, so each kv-head's V is streamed `n_rep`x (once per
    /// query head in its group). At long ctx that dominates - the QK/softmax/PV
    /// probe measured only ~5 GB/s (STREAM wall ~23), i.e. compute/traffic-bound
    /// well below the memory ceiling. Here PV reads each V row ONCE per group
    /// (the q8 block stays L1-resident across the `n_rep` axpys) and parallelises
    /// over (kv-head x ctx-chunk) so all cores stay busy even though there are
    /// only `n_kv_head` groups. Bit-close to `attention` (softmax identical; PV
    /// is the same accumulation, reassociated across chunks -> maxdiff ~1e-5).
    pub fn attention_grouped(&self, q: &[f32], scale: f32, out: &mut [f32]) -> Result<()> {
        let (hd, seq, n_rep, nkv) = (
            self.head_dim,
            self.seq,
            self.n_head / self.n_kv_head,
            self.n_kv_head,
        );
        debug_assert_eq!(q.len(), self.n_head * hd);
        debug_assert_eq!(out.len(), self.n_head * hd);
        if seq == 0 {
            for x in out.iter_mut() {
                *x = 0.0;
            }
            return Ok(());
        }

        // 1. QK^T per kv-head (K read once per group via the quantized mul_mat).
        let mut scores: Vec<Vec<f32>> = Vec::with_capacity(nkv);
        for h in 0..nkv {
            let mut s = vec![0f32; n_rep * seq];
            let qg = &q[h * n_rep * hd..(h * n_rep + n_rep) * hd];
            matmul_bytes(GgmlDType::Q8_0, (n_rep, hd, seq), qg, &self.k[h], &mut s)?;
            scores.push(s);
        }

        // 2. softmax -> normalised probabilities in place (parallel over groups).
        scores.par_iter_mut().for_each(|s| {
            for r in 0..n_rep {
                let row = &mut s[r * seq..(r + 1) * seq];
                let mut mx = f32::NEG_INFINITY;
                for x in row.iter_mut() {
                    *x *= scale;
                    if *x > mx {
                        mx = *x;
                    }
                }
                let sum = crate::inference::kernel::cpu_decode_exec::exp_sub_max_sum(row, mx);
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for x in row.iter_mut() {
                    *x *= inv;
                }
            }
        });

        // 3. grouped PV: read each V row ONCE per kv-head group, chunked over ctx
        //    to fill cores. Partial acc per (kv-head, chunk), [n_rep*hd].
        let nb = self.nb;
        let nchunk = ((rayon::current_num_threads() + nkv - 1) / nkv).max(1);
        let cs = seq.div_ceil(nchunk);
        let partials: Vec<Vec<f32>> = (0..nkv * nchunk)
            .into_par_iter()
            .map(|idx| {
                let kvh = idx / nchunk;
                let ci = idx % nchunk;
                let t0 = ci * cs;
                let t1 = ((ci + 1) * cs).min(seq);
                let mut acc = vec![0f32; n_rep * hd];
                if t0 >= t1 {
                    return acc;
                }
                let vblocks: &[BlockQ8_0] = cast_blocks(&self.v[kvh]).expect("v q8_0 cast");
                let s = &scores[kvh];
                for t in t0..t1 {
                    let bbase = t * nb;
                    for b in 0..nb {
                        let blk = &vblocks[bbase + b];
                        let d = blk.d.to_f32();
                        let od = b * QK;
                        for r in 0..n_rep {
                            let pt = s[r * seq + t] * d;
                            q8_pv_axpy(pt, &blk.qs, &mut acc[r * hd + od..r * hd + od + QK]);
                        }
                    }
                }
                acc
            })
            .collect();

        // combine chunk partials -> out (trivial: n_head x nchunk x hd adds).
        for qh in 0..self.n_head {
            let kvh = qh / n_rep;
            let r = qh % n_rep;
            let o = &mut out[qh * hd..(qh + 1) * hd];
            for x in o.iter_mut() {
                *x = 0.0;
            }
            for ci in 0..nchunk {
                let acc = &partials[kvh * nchunk + ci][r * hd..(r + 1) * hd];
                for i in 0..hd {
                    o[i] += acc[i];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.123 + seed).sin()).collect()
    }

    // Decode-attention QK/softmax/PV throughput probe. Times a single
    // layer's GQA attention at a realistic 2.5K decode shape and reports the
    // effective KV bandwidth - if it's far below the STREAM wall (~23 GB/s) the
    // QK is overhead-bound (closable), not bandwidth-bound. Run with:
    //   cargo test --release -p loken --lib qk_throughput_probe -- --nocapture --ignored
    #[test]
    #[ignore]
    fn qk_throughput_probe() {
        use std::time::Instant;
        let (nh, nkv, hd, seq) = (32usize, 8usize, 128usize, 2500usize);
        let mut c = CpuQ8Kv::new(nh, nkv, hd);
        for t in 0..seq {
            c.append(&fill(nkv * hd, t as f32), &fill(nkv * hd, t as f32 + 9.0))
                .unwrap();
        }
        let q = fill(nh * hd, 3.0);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; nh * hd];
        // KV bytes read per call: K+V, q8_0 block = 34 bytes / 32 elems.
        let kv_bytes = 2.0 * (nkv as f64) * (seq as f64) * (hd as f64 / 32.0) * 34.0;
        for _ in 0..5 {
            c.attention(&q, scale, &mut out).unwrap();
        } // warm
        let iters = 100;
        let t0 = Instant::now();
        for _ in 0..iters {
            c.attention(&q, scale, &mut out).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let gbps = kv_bytes / (ms / 1000.0) / 1e9;
        println!(
            "QK-probe: 1 layer GQA attn @ seq={seq} nh={nh} nkv={nkv} hd={hd} -> {ms:.3} ms/call, \
             KV={:.2} MB, effective {gbps:.1} GB/s (STREAM wall ~23). 40-layer/token ≈ {:.1} ms",
            kv_bytes / 1e6,
            ms * 40.0
        );
    }

    // A/B the baseline PV (V read n_repx) vs the grouped-PV flash variant at the
    // real mistral-nemo 2.5K shape; report tok/s-equivalent + parity. Run with:
    //   cargo test --release -p loken --lib grouped_ab -- --nocapture --ignored
    #[test]
    #[ignore]
    fn grouped_ab() {
        use std::time::Instant;
        let (nh, nkv, hd, seq) = (32usize, 8usize, 128usize, 2500usize);
        let mut c = CpuQ8Kv::new(nh, nkv, hd);
        for t in 0..seq {
            c.append(&fill(nkv * hd, t as f32), &fill(nkv * hd, t as f32 + 9.0))
                .unwrap();
        }
        let q = fill(nh * hd, 3.0);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut a = vec![0f32; nh * hd];
        let mut b = vec![0f32; nh * hd];
        for _ in 0..5 {
            c.attention(&q, scale, &mut a).unwrap();
            c.attention_grouped(&q, scale, &mut b).unwrap();
        }
        let md = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let iters = 100;
        let t0 = Instant::now();
        for _ in 0..iters {
            c.attention(&q, scale, &mut a).unwrap();
        }
        let ms_base = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let t0 = Instant::now();
        for _ in 0..iters {
            c.attention_grouped(&q, scale, &mut b).unwrap();
        }
        let ms_grp = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!(
            "grouped_ab @ seq={seq} nh={nh} nkv={nkv} hd={hd}: baseline {ms_base:.3} ms/layer | grouped {ms_grp:.3} ms/layer -> {:.2}x | 40L/tok {:.1}->{:.1} ms | maxdiff={md:.2e}",
            ms_base / ms_grp, ms_base * 40.0, ms_grp * 40.0
        );
        assert!(md < 1e-3, "parity: {md}");
    }

    // Parity: grouped-PV variant must match the baseline attention() bit-close
    // across GQA shapes (softmax identical; PV reassociated across chunks).
    #[test]
    fn grouped_matches_attention() {
        for (nh, nkv, hd, seq) in [
            (2usize, 1usize, 32usize, 5usize),
            (8, 2, 64, 130),
            (32, 8, 128, 777),
        ] {
            let mut c = CpuQ8Kv::new(nh, nkv, hd);
            for t in 0..seq {
                c.append(&fill(nkv * hd, t as f32), &fill(nkv * hd, t as f32 + 9.0))
                    .unwrap();
            }
            let q = fill(nh * hd, 3.0);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut a = vec![0f32; nh * hd];
            let mut b = vec![0f32; nh * hd];
            c.attention(&q, scale, &mut a).unwrap();
            c.attention_grouped(&q, scale, &mut b).unwrap();
            let md = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                md < 1e-3,
                "nh={nh} nkv={nkv} hd={hd} seq={seq} maxdiff={md}"
            );
        }
    }

    // attention_multi with s_count==1 must reduce EXACTLY to the validated
    // single-token attention() (same Q8 store, full causal window).
    #[test]
    fn multi_s1_matches_single() {
        let (nh, nkv, hd) = (2usize, 1usize, 32usize);
        let mut c = CpuQ8Kv::new(nh, nkv, hd);
        for t in 0..5 {
            c.append(&fill(nkv * hd, t as f32), &fill(nkv * hd, t as f32 + 9.0))
                .unwrap();
        }
        let q = fill(nh * hd, 3.0);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut a = vec![0f32; nh * hd];
        let mut b = vec![0f32; nh * hd];
        c.attention(&q, scale, &mut a).unwrap();
        c.attention_multi(&q, 1, scale, &mut b).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    // Causality: in a 2-token multi-attention, query 0 (valid window = 1) must
    // be bit-identical to single-token attention over a cache holding ONLY t0  -
    // i.e. it cannot see t1.
    #[test]
    fn multi_query0_is_causal() {
        let (nh, nkv, hd) = (2usize, 1usize, 32usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let (k0, v0) = (fill(nkv * hd, 0.0), fill(nkv * hd, 9.0));
        let (k1, v1) = (fill(nkv * hd, 1.0), fill(nkv * hd, 10.0));
        let q0 = fill(nh * hd, 3.0);
        let q1 = fill(nh * hd, 4.0);

        let mut c1 = CpuQ8Kv::new(nh, nkv, hd);
        c1.append(&k0, &v0).unwrap();
        let mut refq0 = vec![0f32; nh * hd];
        c1.attention(&q0, scale, &mut refq0).unwrap();

        let mut c2 = CpuQ8Kv::new(nh, nkv, hd);
        c2.append(&k0, &v0).unwrap();
        c2.append(&k1, &v1).unwrap();
        let mut qmulti = vec![0f32; 2 * nh * hd];
        qmulti[..nh * hd].copy_from_slice(&q0);
        qmulti[nh * hd..].copy_from_slice(&q1);
        let mut out = vec![0f32; 2 * nh * hd];
        c2.attention_multi(&qmulti, 2, scale, &mut out).unwrap();
        // query 0 occupies the first nh*hd of out.
        for (x, y) in refq0.iter().zip(&out[..nh * hd]) {
            assert!((x - y).abs() < 1e-5, "causal leak: {x} vs {y}");
        }
    }

    // The spec-decode verify shape: a cache with prior HISTORY (base > 0) plus
    // `s_count` new tokens. Each of the last `s_count` queries must equal a
    // single-token attention over a cache truncated to that query's causal window
    // - i.e. attention_multi over [history + drafts] == per-position decode.
    #[test]
    fn multi_with_history_matches_single() {
        let (nh, nkv, hd) = (4usize, 2usize, 32usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let base = 3usize; // prior history tokens
        let s_count = 2usize; // verify tokens
        let total = base + s_count;
        // Build the full cache (history + verify tokens).
        let mut c = CpuQ8Kv::new(nh, nkv, hd);
        for t in 0..total {
            c.append(&fill(nkv * hd, t as f32), &fill(nkv * hd, t as f32 + 9.0))
                .unwrap();
        }
        // The s_count query vectors (positions base..total), token-major.
        let mut qmulti = vec![0f32; s_count * nh * hd];
        for s in 0..s_count {
            let q = fill(nh * hd, (base + s) as f32 + 100.0);
            qmulti[s * nh * hd..(s + 1) * nh * hd].copy_from_slice(&q);
        }
        let mut out = vec![0f32; s_count * nh * hd];
        c.attention_multi(&qmulti, s_count, scale, &mut out)
            .unwrap();
        // Reference: for query s (position base+s), a fresh cache holding exactly
        // tokens [0 ..= base+s], single-token attention with the SAME q vector.
        for s in 0..s_count {
            let mut cref = CpuQ8Kv::new(nh, nkv, hd);
            for t in 0..=(base + s) {
                cref.append(&fill(nkv * hd, t as f32), &fill(nkv * hd, t as f32 + 9.0))
                    .unwrap();
            }
            let q = fill(nh * hd, (base + s) as f32 + 100.0);
            let mut refout = vec![0f32; nh * hd];
            cref.attention(&q, scale, &mut refout).unwrap();
            for (x, y) in refout.iter().zip(&out[s * nh * hd..(s + 1) * nh * hd]) {
                assert!((x - y).abs() < 1e-5, "verify[{s}] mismatch: {x} vs {y}");
            }
        }
    }
}
