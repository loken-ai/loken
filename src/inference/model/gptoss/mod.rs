//! gpt-oss (OPENAI_MOE) support helpers.
//!
//! gpt-oss is a MoE transformer (top-k experts) with sliding-window attention
//! on a subset of layers, YaRN RoPE scaling, MXFP4-quantized experts, and the
//! distinctive **attention sinks**: a learned per-head scalar logit folded into
//! the attention softmax denominator. The sink acts as an always-available
//! attention target that absorbs probability mass (lowering attention to real
//! tokens) but contributes no value - it has no corresponding K/V.
//!
//! This module currently provides the novel numerical primitive (sink-aware
//! softmax). The full multi-device loader (`gptoss_multi.rs`, modeled on
//! `inference/model/qwen3/moe_multi.rs`) composes this with existing MoE / sliding-window /
//! YaRN building blocks; it is built as one complete unit before being wired
//! into the engine dispatch.

use crate::inference::fused_moe::FusedMoeGGUF;
use crate::tensor::layer::Embedding;
use crate::tensor::layer::Linear;
use crate::tensor::layer::RmsNorm;
use crate::tensor::ops::{heads_first, Activation};
use crate::tensor::quantized::{gguf_file, QMatMul, QTensor};
use crate::tensor::KvCache;
use crate::tensor::{DType, Device, IndexOp, Result, Tensor, D};
use std::io::{Read, Seek};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct GptOssConfig {
    pub n_layers: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub embedding_length: usize,
    pub ffn_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// Interleave period for sliding-window vs full-attention layers
    /// (llama.cpp `set_swa_pattern`, HF `layer_types`): layers with
    /// `il % pattern == pattern-1` use FULL attention, the rest the sliding
    /// window; `0` = every layer windowed. gpt-oss ships without the GGUF key
    /// -> default 2 (even layers SWA-128, odd layers full), matching
    /// llama.cpp's `openai-moe` (`swa_period = 2`).
    pub swa_pattern: usize,
    /// Sliding-window size for the windowed-attention layers (full attention
    /// when the running KV length is below this).
    pub sliding_window: usize,
    pub context_length: usize,
    pub rope_freq_base: f32,
    /// YaRN linear scaling factor (1.0 = no scaling).
    pub rope_scaling_factor: f32,
    /// Original (pre-scaling) context the RoPE was trained at.
    pub rope_orig_context: usize,
    pub rms_eps: f64,
}

impl GptOssConfig {
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        let req_u = |k: &str| -> Result<usize> {
            ct.metadata
                .get(&format!("gptoss.{k}"))
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .ok_or_else(|| crate::tensor::Error::msg(format!("gpt-oss: missing gptoss.{k}")))
        };
        let opt_u = |k: &str, d: usize| -> usize {
            ct.metadata
                .get(&format!("gptoss.{k}"))
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let opt_f = |k: &str, d: f32| -> f32 {
            ct.metadata
                .get(&format!("gptoss.{k}"))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(d)
        };
        let embedding_length = req_u("embedding_length")?;
        let n_head = req_u("attention.head_count")?;
        // key_length is the per-head dim; fall back to embedding/head_count.
        let head_dim = opt_u("attention.key_length", embedding_length / n_head.max(1));
        Ok(Self {
            n_layers: req_u("block_count")?,
            n_head,
            n_kv_head: opt_u("attention.head_count_kv", n_head),
            head_dim,
            embedding_length,
            ffn_dim: req_u("feed_forward_length")?,
            n_expert: req_u("expert_count")?,
            n_expert_used: req_u("expert_used_count")?,
            swa_pattern: opt_u("attention.sliding_window_pattern", 2),
            sliding_window: opt_u("attention.sliding_window", 0),
            context_length: opt_u("context_length", 8192),
            rope_freq_base: opt_f("rope.freq_base", 10000.0),
            rope_scaling_factor: opt_f("rope.scaling.factor", 1.0),
            rope_orig_context: opt_u("rope.scaling.original_context_length", 0),
            rms_eps: opt_f("attention.layer_norm_rms_epsilon", 1e-5) as f64,
        })
    }
}

/// Softmax over the last (key) dim with a per-head **sink** logit added to each
/// row's denominator. Matches ggml `ggml_soft_max_add_sinks`.
///
/// - `scores`: attention logits `[b, n_head, q, kv]` (already scaled, masked).
/// - `sinks`:  learned per-head sink logits `[n_head]`.
///
/// Returns weights `[b, n_head, q, kv]` whose rows sum to `< 1` - the missing
/// mass is what the (output-less) sink absorbed.
pub fn softmax_last_dim_with_sinks(
    scores: &Tensor,
    mask: Option<&Tensor>,
    sinks: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    // Fused single-pass path (CPU rowwise, or CUDA single-launch): folds the
    // optional additive causal/window mask, the `scores * scale` affine (the scale
    // applies to scores, the learned sink stays raw), and the whole sink-softmax.
    if let Some(out) =
        crate::inference::kernel::fused::fused_softmax_sinks(scores, mask, sinks, scale as f32)?
    {
        return Ok(out);
    }
    // Fallback: add the mask (if any), then the materialized op chain.
    let scores_owned = match mask {
        Some(m) => Some(scores.broadcast_add(m)?),
        None => None,
    };
    let scores = scores_owned.as_ref().unwrap_or(scores);
    let (_b, n_head, _q, _kv) = scores.dims4()?;
    let scores = (scores * scale)?;
    let scores = &scores;
    // Per-head sink broadcastable to [b, n_head, q, 1].
    let sink = sinks.reshape((1, n_head, 1, 1))?;
    // Numerically-stable max over the key dim, then fold in the sink so the
    // stabilizer covers the extra logit too.
    let row_max = scores.max_keepdim(D::Minus1)?; // [b,h,q,1]
    let m = row_max.broadcast_maximum(&sink)?; // [b,h,q,1]
    let exp_scores = scores.broadcast_sub(&m)?.exp()?; // [b,h,q,kv]
    let exp_sink = sink.broadcast_sub(&m)?.exp()?; // [b,h,q,1]
    let denom = (exp_scores.sum_keepdim(D::Minus1)? + exp_sink)?; // [b,h,q,1]
    exp_scores.broadcast_div(&denom)
}

/// gpt-oss attention: QKV (+bias) -> NEOX RoPE -> KV cache -> scaled scores ->
/// causal + sliding-window mask -> sink-softmax -> Wo (+bias). F-dtype path
/// (correctness first; Q8/Q4 decode kernels are a follow-up).
pub struct GptOssAttn {
    pub wq: crate::tensor::quantized::QMatMul,
    pub wk: crate::tensor::quantized::QMatMul,
    pub wv: crate::tensor::quantized::QMatMul,
    pub wo: crate::tensor::quantized::QMatMul,
    pub bq: Option<Tensor>,
    pub bk: Option<Tensor>,
    pub bv: Option<Tensor>,
    pub bo: Option<Tensor>,
    pub sinks: Tensor, // [n_head]
    pub cos: Tensor,   // [max_seq, head_dim/2]
    pub sin: Tensor,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    /// Pre-allocated in-place KV cache (writes new K/V via slice_set, grows in
    /// chunks). Replaces the concat cache that re-`cat`-ed the whole K/V every
    /// token - that O(kv_len)/token copy made attention the dominant decode
    /// stage (61% per profiling). Sliding-window is enforced by the mask, not
    /// cache storage, so keeping the full cache here matches the prior behavior.
    pub kv_cache: KvCache,
    /// Device-position CUDA-graph decode (gated LOKEN_GPTOSS_DEVPOS). Fixed-max
    /// ring buffers [b,n_kv,kv_max,hd] F16; the write slot lives on device (pos_dev)
    /// so a captured graph advances it on replay. None until first devpos decode.
    pub kbuf: Option<Tensor>,
    pub vbuf: Option<Tensor>,
    pub kv_max: usize,
    pub cos_f16: Tensor, // full [max_seq, head_dim/2] F16 - for device-position rope
    pub sin_f16: Tensor,
    /// CPU decode fused-attention store (F16 KV, AVX2 sink-softmax). Replaces the
    /// materialized [n_head, kv_len] BF16 score-matmul chain in `attend()` - that
    /// path was 66% of gpt-oss CPU decode (profiled). Built lazily on
    /// the first CPU decode token; window = this layer's sliding_window.
    pub cpu_f16_kv: Option<crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
    /// Cached F32 sink logits `[n_head]` for the CPU fused path (avoid re-reading
    /// the tensor every token).
    pub sinks_f32: Option<Vec<f32>>,
}

fn gptoss_devpos_enabled() -> bool {
    false
}

impl GptOssAttn {
    fn proj(
        w: &crate::tensor::quantized::QMatMul,
        b: &Option<Tensor>,
        x: &Tensor,
    ) -> Result<Tensor> {
        let y = w.forward(x)?;
        match b {
            Some(bias) => {
                let bias = bias.to_dtype(y.dtype())?;
                y.broadcast_add(&bias)
            }
            None => Ok(y),
        }
    }

    /// Additive mask `[1,1,seq,kv]` (0 visible, -inf masked) enforcing causality
    /// and, for windowed layers, the sliding window. Returns None when no mask
    /// is needed (decode with full attention).
    fn build_mask(
        &self,
        seq: usize,
        input_pos: usize,
        device: &crate::tensor::Device,
    ) -> Result<Option<Tensor>> {
        if seq == 1 && self.sliding_window.is_none() {
            return Ok(None);
        }
        let kv = input_pos + seq;
        let window = self.sliding_window.unwrap_or(0);
        // On-device mask generation (no host Vec + H2D upload) - the H2D was the
        // sole CUDA-graph replay blocker (a captured transient-host memcpy).
        if let Some(m) =
            crate::inference::kernel::fused::fused_gptoss_mask(seq, kv, input_pos, window, device)?
        {
            return Ok(Some(m));
        }
        // CPU fallback.
        let mut data = vec![0f32; seq * kv];
        for i in 0..seq {
            let qpos = input_pos + i;
            for j in 0..kv {
                let masked = j > qpos
                    || self
                        .sliding_window
                        .is_some_and(|w| qpos.saturating_sub(j) >= w);
                if masked {
                    data[i * kv + j] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Some(Tensor::from_vec(data, (1, 1, seq, kv), device)?))
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        input_pos: usize,
        shared_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let dev = x.device().clone();
        let (q, k, v) = (
            Self::proj(&self.wq, &self.bq, x)?,
            Self::proj(&self.wk, &self.bk, x)?,
            Self::proj(&self.wv, &self.bv, x)?,
        );
        // Three projections, one rule, so only the head count differs between them.
        let q = heads_first(q, self.n_head, self.head_dim)?;
        let k = heads_first(k, self.n_kv_head, self.head_dim)?;
        let v = heads_first(v, self.n_kv_head, self.head_dim)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // CUDA-graph device-position decode (gated LOKEN_GPTOSS_DEVPOS): one
        // device pos scalar drives RoPE + KV-write + attention so a captured graph
        // advances on replay. Bit-identical to the host path (validated for the
        // attention by test_devkvlen_sinks). Mirrors lfm2 (inference/model/lfm2_moe/mod.rs).
        let devpos = (gptoss_devpos_enabled() || gptoss_graph_enabled())
            && dev.is_cuda()
            && q.dtype() == DType::F16
            && self.head_dim == 64
            && input_pos + seq <= self.kv_max;
        // One device pos scalar shared across all layers when the graph machine
        // owns it (so a captured graph advances every layer from one buffer);
        // else a fresh per-layer scalar (non-graph devpos validation).
        let pos_dev = if devpos {
            match shared_pos {
                Some(p) if p.device().location() == dev.location() => Some(p.clone()),
                _ => Some(Tensor::new(&[input_pos as i32], &dev)?),
            }
        } else {
            None
        };
        let (q, k) = if let Some(pd) = pos_dev.as_ref() {
            use crate::inference::kernel::fused::neox_rope_devpos_f16 as rd;
            match (
                rd(&q, &self.cos_f16, &self.sin_f16, pd, self.head_dim)?,
                rd(&k, &self.cos_f16, &self.sin_f16, pd, self.head_dim)?,
            ) {
                (Some(q), Some(k)) => (q, k),
                _ => crate::tensor::bail!("gptoss devpos rope failed"),
            }
        } else {
            // Fused NeoX rope (1 launch per tensor, same kernel as lfm2/
            // nemotron/qwen3.5) off the precomputed F16 tables - the composed
            // ops::rope is ~20 small launches per layer (narrow-cast
            // cos/sin + rotate-half cat/neg/mul/add chains), a pure
            // launch-gap tail at decode. Composed path kept for CPU/non-F16.
            let mut fused: Option<(Tensor, Tensor)> = None;
            if dev.is_cuda() && q.dtype() == DType::F16 {
                let cos16 = self.cos_f16.narrow(0, input_pos, seq)?;
                let sin16 = self.sin_f16.narrow(0, input_pos, seq)?;
                use crate::inference::kernel::fused::neox_rope_f16 as rf;
                if let (Some(qr), Some(kr)) = (
                    rf(&q, &cos16, &sin16, self.head_dim)?,
                    rf(&k, &cos16, &sin16, self.head_dim)?,
                ) {
                    fused = Some((qr, kr));
                }
            }
            match fused {
                Some(qk) => qk,
                None => {
                    let cos = self.cos.narrow(0, input_pos, seq)?.to_dtype(q.dtype())?;
                    let sin = self.sin.narrow(0, input_pos, seq)?.to_dtype(q.dtype())?;
                    (
                        crate::tensor::ops::rope(&q, &cos, &sin)?,
                        crate::tensor::ops::rope(&k, &cos, &sin)?,
                    )
                }
            }
        };
        // CPU decode fast path: fused F16-KV sink-softmax attention (AVX2), the
        // CPU twin of the GPU flash decode. Replaces the materialized
        // [n_head, kv_len] BF16 score-matmul chain in attend() - 66% of gpt-oss
        // CPU decode (profiled). Windowed layers scan only the last
        // `window` keys (read-time). Prefill (seq>1) keeps the matmul path.
        if seq == 1 && b == 1 && dev.is_cpu() {
            if let Some(o) = self.cpu_fused_decode(&q, &k, &v, input_pos, scale as f32)? {
                return Self::proj(&self.wo, &self.bo, &o);
            }
        }
        // devpos KV: ring-buffer write (all seq) + (decode) device-kv_len flash
        // with sinks -> returns. Prefill (seq>1) falls through to populate the
        // contiguous cache too (harmless redundancy) and run the normal attention.
        if let Some(pd) = pos_dev.as_ref() {
            if self.kbuf.is_none() {
                self.kbuf = Some(Tensor::zeros_on(
                    (b, self.n_kv_head, self.kv_max, self.head_dim),
                    DType::F16,
                    &dev,
                )?);
                self.vbuf = Some(Tensor::zeros_on(
                    (b, self.n_kv_head, self.kv_max, self.head_dim),
                    DType::F16,
                    &dev,
                )?);
            }
            let (kbuf, vbuf) = (self.kbuf.as_ref().unwrap(), self.vbuf.as_ref().unwrap());
            crate::inference::kernel::fused::kv_write_at_pos(&k, &v, kbuf, vbuf, pd, self.kv_max)?;
            if seq == 1 {
                if let Some(o) = crate::inference::kernel::fused::flash_decode_devkvlen(
                    &q.reshape((b, self.n_head, self.head_dim))?,
                    kbuf,
                    vbuf,
                    pd,
                    Some(&self.sinks),
                    self.sliding_window.unwrap_or(0),
                    scale as f32,
                    b,
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                )? {
                    let o = o
                        .reshape((b, self.n_head, seq, self.head_dim))?
                        .transpose(1, 2)?
                        .reshape((b, seq, self.n_head * self.head_dim))?;
                    return Self::proj(&self.wo, &self.bo, &o);
                }
            }
        }
        // Decode fast path (seq==1, CUDA, hd==64, F16): append IN PLACE and run
        // the fused sink-softmax flash decode straight off the FULL backing
        // buffers. `append()`'s returned narrows are a device COPY of the whole
        // growing cache per call on this substrate (middle-dim narrow can't be
        // a view) - 2 O(kv_len) copies per layer per token. The flash kernel
        // takes explicit strides + kv_len, so it reads the backing buffers
        // directly. No mask either: at decode a full-attention layer sees every
        // cached position, and a windowed layer's mask is equivalent to scanning
        // only the last `window` positions - done with a kv_start pointer offset
        // (O(window) instead of O(kv_len) + per-token mask alloc/launch).
        if seq == 1 && dev.is_cuda() && self.head_dim == 64 && q.dtype() == DType::F16 {
            self.kv_cache.append_write(&k, &v)?;
            let kv_len = self.kv_cache.current_seq_len();
            let kv_start = match self.sliding_window {
                Some(w) if kv_len > w => kv_len - w,
                _ => 0,
            };
            let kb = self.kv_cache.k_cache().all_data().clone().ok_or_else(|| {
                crate::tensor::Error::msg("gptoss: empty k cache after append".to_string())
            })?;
            let vb = self.kv_cache.v_cache().all_data().clone().ok_or_else(|| {
                crate::tensor::Error::msg("gptoss: empty v cache after append".to_string())
            })?;
            if let Some(o) = crate::inference::kernel::fused::gptoss_flash_decode_win(
                &q,
                &kb,
                &vb,
                None,
                Some(&self.sinks),
                scale as f32,
                b,
                self.n_head,
                self.n_kv_head,
                kv_start,
                kv_len,
                self.head_dim,
            )? {
                let out = o
                    .reshape((b, self.n_head, seq, self.head_dim))?
                    .transpose(1, 2)?
                    .reshape((b, seq, self.n_head * self.head_dim))?;
                return Self::proj(&self.wo, &self.bo, &out);
            }
            // Kernel declined (shouldn't happen under the guards above): fall
            // through to the matmul chain on narrowed copies of the cache.
            let k = self
                .kv_cache
                .k()?
                .ok_or_else(|| crate::tensor::Error::msg("gptoss: empty k cache".to_string()))?;
            let v = self
                .kv_cache
                .v()?
                .ok_or_else(|| crate::tensor::Error::msg("gptoss: empty v cache".to_string()))?;
            return self.attend(&q, &k, &v, seq, input_pos, b, scale, &dev);
        }
        let (k, v) = self.kv_cache.append(&k, &v)?; // k,v: [b, n_kv_head, kv_len, hd]
        self.attend(&q, &k, &v, seq, input_pos, b, scale, &dev)
    }

    /// CPU decode (seq==1): fused F16-KV sink-softmax attention. `q`/`k`/`v` are
    /// the rope'd `[1, n_head|n_kv_head, 1, hd]` tensors. Maintains an F16 KV
    /// store (window = this layer's `sliding_window`), seeded once from the
    /// prefill `kv_cache`, and runs the AVX2 online-softmax scan with the learned
    /// per-head sink. Returns `[1, 1, n_head*hd]` F32->working-dtype, or `None` to
    /// fall through to the matmul path. Also mirrors the current K/V into
    /// `kv_cache` so the next token's seeding and any prefill path stay coherent.
    fn cpu_fused_decode(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        input_pos: usize,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        use crate::inference::cache::cpu_f16_kv::CpuF16Kv;
        let dev = q.device().clone();
        let (nh, nkv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        let to_f32_flat = |t: &Tensor| -> Result<Vec<f32>> {
            t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
        };
        if self.sinks_f32.is_none() {
            self.sinks_f32 = Some(
                self.sinks
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }
        // Fresh sequence -> drop the store (window is this layer's own).
        if input_pos == 0 {
            self.cpu_f16_kv = None;
        }
        if self.cpu_f16_kv.is_none() {
            self.cpu_f16_kv = Some(CpuF16Kv::new(
                nh,
                nkv,
                hd,
                self.sliding_window.filter(|&w| w > 0),
            ));
        }
        // Seed from the prefill kv_cache for positions [store.len(), input_pos):
        // prefill populated kv_cache but not this decode-only store.
        if self.cpu_f16_kv.as_ref().unwrap().len() < input_pos {
            let start = self.cpu_f16_kv.as_ref().unwrap().len();
            if let (Some(kall), Some(vall)) = (self.kv_cache.k()?, self.kv_cache.v()?) {
                let total = kall.dim(2)?; // [1, nkv, total, hd]
                let kv3 = to_f32_flat(&kall)?;
                let vv3 = to_f32_flat(&vall)?;
                let cache = self.cpu_f16_kv.as_mut().unwrap();
                let mut kt = vec![0f32; nkv * hd];
                let mut vt = vec![0f32; nkv * hd];
                for t in start..input_pos.min(total) {
                    for h in 0..nkv {
                        let src = h * total * hd + t * hd;
                        kt[h * hd..(h + 1) * hd].copy_from_slice(&kv3[src..src + hd]);
                        vt[h * hd..(h + 1) * hd].copy_from_slice(&vv3[src..src + hd]);
                    }
                    cache.append(&kt, &vt)?;
                }
            } else {
                return Ok(None); // no history to seed -> fall back
            }
        }
        let (kf, vf) = (to_f32_flat(k)?, to_f32_flat(v)?);
        if kf.len() != nkv * hd || vf.len() != nkv * hd {
            return Ok(None);
        }
        self.cpu_f16_kv.as_mut().unwrap().append(&kf, &vf)?;
        let qf = to_f32_flat(q)?;
        if qf.len() != nh * hd {
            return Ok(None);
        }
        let mut out = vec![0f32; nh * hd];
        let sinks = self.sinks_f32.as_deref();
        self.cpu_f16_kv
            .as_ref()
            .unwrap()
            .attention_sinks(&qf, scale, sinks, &mut out)?;
        // Keep kv_cache consistent for the next token's seed + prefill paths.
        self.kv_cache.append(k, v)?;
        let out_t = Tensor::from_vec(out, (1, 1, nh * hd), &dev)?.to_dtype(q.dtype())?;
        Ok(Some(out_t))
    }

    /// Attention core on materialized K/V `[b, n_kv_head, kv_len, hd]`: fused
    /// flash decode when applicable, else the masked GQA matmul chain, then the
    /// output projection. (Prefill and non-CUDA decode path.)
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq: usize,
        input_pos: usize,
        b: usize,
        scale: f64,
        dev: &crate::tensor::Device,
    ) -> Result<Tensor> {
        let q = q.clone();
        let (mut k, mut v) = (k.clone(), v.clone());
        let mut kv_len = k.dim(2)?;
        // Decode-time sliding window (CPU / non-fused fallback): a windowed
        // layer at seq==1 sees only the last `window` cached positions, and a
        // full layer sees them all - in BOTH cases every remaining position is
        // causally valid, so no additive mask is needed. Narrow K/V to the last
        // `window` rows for windowed layers (O(window) instead of O(kv_len)  -
        // half the gpt-oss layers, window=128, at 2.5K that's ~20x less score
        // work + KV read per token). Mirrors the GPU `kv_start` pointer bump.
        // Prefill (seq>1) keeps the full cache + build_mask.
        let decode_no_mask = seq == 1;
        if decode_no_mask {
            if let Some(w) = self.sliding_window {
                if w > 0 && kv_len > w {
                    let start = kv_len - w;
                    k = k.narrow(2, start, w)?;
                    v = v.narrow(2, start, w)?;
                    kv_len = w;
                }
            }
        }
        // How many query heads share one key/value head: a quotient of two fields this block
        // already holds, so it is taken here rather than stored a second time. The floor is
        // the checkpoint's: nothing forbids a file from declaring no key/value heads at all.
        let g = self.n_head / self.n_kv_head.max(1);
        // GQA without repeat_kv: group the q heads onto their shared kv head so
        // the attention matmuls read k/v ONCE (n_kv_head) instead of n_head
        // copies - `groups`x less KV-cache traffic per token. A tensor-level repeat_kv
        // maps output head H -> kv head H/groups (kv outer, group inner), so the
        // reshape [b, n_head, seq, hd] -> [b, n_kv_head, groups*seq, hd] is the
        // matching grouping. Scores + sink-softmax stay in F32 (the sink term is
        // precision-sensitive); weights cast back to V's dtype for the V matmul.
        // Raw Q.Kᵀ in F32; the 1/sqrt(hd) scale is folded into the sink-softmax
        // kernel (applied to scores, not the learned sink) to drop one affine.
        // Fused flash-decode (seq=1, hd=64): one F16 kernel computes scores +
        // sink-softmax + V - no cuBLAS (makes the forward CUDA-graph-capturable)
        // and no [n_head, kv_len] scores in HBM. Falls back to the matmul chain
        // for prefill / other head dims / CPU.
        let flash = if seq == 1 && dev.is_cuda() && self.head_dim == 64 {
            let mask_flat = match self.build_mask(seq, input_pos, dev)? {
                Some(m) => Some(m.reshape((kv_len,))?),
                None => None,
            };
            crate::inference::kernel::fused::gptoss_flash_decode(
                &q,
                &k,
                &v,
                mask_flat.as_ref(),
                Some(&self.sinks),
                scale as f32,
                b,
                self.n_head,
                self.n_kv_head,
                kv_len,
                self.head_dim,
            )?
        } else {
            None
        };
        let out = if let Some(o) = flash {
            // o: [b, n_head, hd] -> [b, n_head, 1, hd] for the shared epilogue.
            o.reshape((b, self.n_head, seq, self.head_dim))?
        } else {
            let scores = if g > 1 {
                let qg = q.reshape((b, self.n_kv_head, g * seq, self.head_dim))?;
                let s = qg.matmul(&k.transpose(2, 3)?)?.to_dtype(DType::F32)?;
                s.reshape((b, self.n_head, seq, kv_len))?
            } else {
                q.matmul(&k.transpose(2, 3)?)?.to_dtype(DType::F32)?
            };
            // Decode (seq==1) needs no mask after the window narrow above; only
            // prefill (seq>1) builds the causal + sliding-window mask. The mask is
            // folded into the fused softmax rather than added in a separate pass.
            let mask = if decode_no_mask {
                None
            } else {
                self.build_mask(seq, input_pos, dev)?
            };
            let weights = softmax_last_dim_with_sinks(&scores, mask.as_ref(), &self.sinks, scale)?
                .to_dtype(v.dtype())?;
            if g > 1 {
                let wg = weights.reshape((b, self.n_kv_head, g * seq, kv_len))?;
                wg.matmul(&v)?
                    .reshape((b, self.n_head, seq, self.head_dim))?
            } else {
                weights.matmul(&v)?
            }
        };
        let out = out
            .transpose(1, 2)?
            .reshape((b, seq, self.n_head * self.head_dim))?;
        Self::proj(&self.wo, &self.bo, &out)
    }
}

/// One gpt-oss transformer block: pre-norm attention + pre-norm MoE FFN,
/// each with a residual. `device` records which GPU the layer lives on so the
/// forward loop can transfer the hidden state across split boundaries.
pub struct GptOssLayer {
    pub attn_norm: RmsNorm,
    pub attn: GptOssAttn,
    pub ffn_norm: RmsNorm,
    pub moe: crate::inference::fused_moe::FusedMoeGGUF,
    pub device: crate::tensor::Device,
    /// attn_norm / ffn_norm weights (F32) + eps, for the fused F16 RMSNorm
    /// kernels (pre-attn no-residual norm; attn-residual-add + ffn_norm).
    pub attn_norm_w: Tensor,
    pub ffn_norm_w: Tensor,
    pub rms_eps: f64,
}

impl GptOssLayer {
    pub fn forward(
        &mut self,
        x: &Tensor,
        input_pos: usize,
        shared_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        // CPU-capable too: synchronize() is a no-op on CPU, Instant still measures.
        let residual = x;
        // gpt-oss is mixed-dtype: the residual stream / matmul weights are
        // BF16 but the RMSNorm weights are F32. Run each norm in F32 (matches
        // llama.cpp) and cast the result back to the stream dtype for the
        // BF16 attention/MoE matmuls.
        let st = x.dtype();
        // Pre-attn RMSNorm: one fused F16 launch (read F16, reduce in F32, write
        // F16) instead of cast-F32 + rms_norm + cast-F16.
        let h = if x.device().is_cuda() && st == DType::F16 {
            crate::inference::kernel::fused::fused_rmsnorm_f16(
                x,
                &self.attn_norm_w,
                self.rms_eps as f32,
            )?
        } else {
            self.attn_norm
                .forward(&x.to_dtype(DType::F32)?)?
                .to_dtype(st)?
        };
        let h = self.attn.forward(&h, input_pos, shared_pos)?;
        // Fuse the attn-residual add + ffn_norm into one F16 launch: x = residual+h
        // (the kept residual for the MoE), hn = ffn_norm(x). Replaces add + cast +
        // rmsnorm + cast (the MoE folds its own post-FFN residual).
        let (x, hn) = if h.device().is_cuda() && st == DType::F16 {
            crate::inference::kernel::fused::fused_add_rmsnorm_f16(
                &h,
                residual,
                &self.ffn_norm_w,
                self.rms_eps as f32,
            )?
        } else {
            let x = (residual + h)?;
            let hn = self
                .ffn_norm
                .forward(&x.to_dtype(DType::F32)?)?
                .to_dtype(st)?;
            (x, hn)
        };
        let is_prefill = x.dim(1)? > 1;
        // FusedMoeGGUF folds the post-FFN residual into its down-projection.

        self.moe.forward_with_residual(&hn, &x, is_prefill)
    }
}

/// Load a matmul weight tensor as **Q8_0** on `d`. gpt-oss stores its weights
/// as MXFP4 (experts) and BF16 (attn / lm_head). The CUDA MMVQ / MoE-GEMM
/// kernels don't accept MXFP4, and `QMatMul` *dequantizes* BF16/F16 weights to
/// an F32 dense tensor (forcing F32 matmul inputs and a dtype clash with F16
/// activations). Requantizing every float/MXFP4 matmul weight to Q8_0 routes
/// them all through the dtype-agnostic quantized MMVQ path (input quantized to
/// q8_1) and is near-lossless + smaller than BF16. Q8_0 (block 32) is also the
/// only MMVQ block type that divides gpt-oss's 2880 dims (the 256-block
/// K-quants don't). Already-Q8_0/Q4_K/etc. weights load straight to the device.
type Mm<'a> = Option<&'a std::sync::Arc<memmap2::Mmap>>;

/// The quantisation the file stores a tensor in, or `None` when it carries no such tensor.
/// What both loaders below decide on, so the lookup is written once.
fn stored_as(c: &gguf_file::Content, name: &str) -> Option<crate::tensor::quantized::GgmlDType> {
    c.tensor_infos.get(name).map(|i| i.ggml_dtype)
}

fn load_q8<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
    mm: Mm<'_>,
) -> Result<QTensor> {
    use crate::tensor::quantized::GgmlDType;
    // A tensor the file does not carry is treated as needing the requantisation, so the read
    // below is the one that reports it missing, by name.
    let requant = stored_as(c, name).is_none_or(|d| {
        matches!(
            d,
            GgmlDType::MxFp4 | GgmlDType::BF16 | GgmlDType::F16 | GgmlDType::F32
        )
    });
    if requant {
        let cpu = c.tensor(r, name, &Device::Cpu)?;
        let f = cpu.dequantize(&Device::Cpu)?; // -> F32 (CPU)
        return QTensor::quantize_onto(&f, GgmlDType::Q8_0, d);
    }
    // already a quantized MMVQ-supported type - zero-copy view on CPU
    if let (Some(m), true) = (mm, d.is_cpu()) {
        if let Some(qt) = crate::tensor::quant_view::gguf_mmap_view(c, m, name)? {
            return Ok(qt);
        }
    }
    c.tensor(r, name, d)
}
/// Load an expert weight keeping native MXFP4 (the MoE-GEMM now has an MXFP4
/// MMVQ path, case 7). Native MXFP4 halves expert VRAM (~24->~12 GB -> fits one
/// GPU) and halves the per-token expert read-bandwidth vs the Q8_0 requant.
/// Non-MXFP4 experts fall back to the `load_q8` behaviour.
fn load_native<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
    mm: Mm<'_>,
) -> Result<QTensor> {
    use crate::tensor::quantized::GgmlDType;
    let native = stored_as(c, name) == Some(GgmlDType::MxFp4);
    if native {
        // native MXFP4 onto the device - MMVQ case 7; zero-copy view on CPU
        if let (Some(m), true) = (mm, d.is_cpu()) {
            if let Some(qt) = crate::tensor::quant_view::gguf_mmap_view(c, m, name)? {
                return Ok(qt);
            }
        }
        return c.tensor(r, name, d);
    }
    load_q8(c, r, name, d, mm)
}
fn ld_qm<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
    mm: Mm<'_>,
) -> Result<QMatMul> {
    QMatMul::from_qtensor(load_q8(c, r, name, d, mm)?)
}
fn ld_f32<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
) -> Result<Tensor> {
    c.tensor(r, name, d)?.dequantize(d)?.to_dtype(DType::F32)
}
fn ld_f32_opt<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    d: &Device,
) -> Option<Tensor> {
    c.tensor(r, name, d)
        .ok()
        .and_then(|t| t.dequantize(d).ok())
        .and_then(|t| t.to_dtype(DType::F32).ok())
}
fn ld_norm<R: Read + Seek>(
    c: &gguf_file::Content,
    r: &mut R,
    name: &str,
    eps: f64,
    d: &Device,
) -> Result<RmsNorm> {
    RmsNorm::from_qtensor(c.tensor(r, name, d)?, eps)
}

/// gpt-oss (OPENAI_MOE) model - multi-device layer split. The MXFP4 experts
/// are kept native (the MoE-GEMM has an MXFP4 MMVQ path, case 7), so expert
/// VRAM stays ~12 GB (vs ~24 GB if requantized to Q8_0) and decode reads half
/// the expert bytes. Attn / lm_head weights are still Q8_0 (`load_q8`). Layers
/// are assigned to devices in contiguous blocks; the hidden state is transferred
/// across device boundaries in the forward loop. (YaRN long-context RoPE is a
/// follow-up - base RoPE is correct below `rope_orig_context`.)
pub struct GptOssModel {
    embed: Embedding,
    embed_dev: Device,
    layers: Vec<GptOssLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    dtype: DType,
    // CUDA-graph decode state (LOKEN_GPTOSS_GRAPH, single-GPU). Captured once
    // after a few warmup decode tokens, replayed per token. See forward_graph_decode.
    pos_dev: Option<Tensor>, // i32 [1] device position, shared across layers
    input_tok: Option<Tensor>, // u32 [1,1] device input id, written in place pre-replay
    #[cfg(feature = "cuda")]
    graph: Option<crate::tensor::cuda_ext::CudaGraph>,
    out_logits: Option<Tensor>,
    /// Persistent (non-arena) logits buffer the captured forward copies into as its
    /// final op, so each replay deposits logits at a fixed address (the lm_head's
    /// own arena output isn't stable across replays). Sized [vocab] at first warmup.
    logits_buf: Option<Tensor>,
    /// Persistent (non-arena) F32 hidden buffer: each token an EAGER embed lands
    /// here OUTSIDE capture, and the captured forward starts from it (via x_in) so
    /// the embed gather - whose input_tok read froze under capture - is excluded
    /// from the graph. Sized [1,1,hidden] at first capture.
    hidden_buf: Option<Tensor>,
    ws_pinned: bool,
    graph_warmup: usize,
    /// Number of replay-vs-eager validations passed so far. The captured graph
    /// stays on probation (each replay re-checked against an eager forward at
    /// the same token/pos) until GRAPH_VALIDATIONS consecutive matches.
    graph_validations: usize,
    /// Set if validation FAILED (replay != eager) - the captured graph is unsafe on
    /// this build, so permanently fall back to the (correct) eager path. Prevents
    /// ever emitting graph-replay garbage when LOKEN_GPTOSS_GRAPH is enabled.
    graph_disabled: bool,
    /// In-memory exact-prefix snapshot (per-layer KV + prefill logits) for
    /// cross-request prompt reuse. Never persisted. See `snapshot_prefix`.
    prefix_cache: Option<GptOssPrefixCache>,
}

/// One resident exact-prompt snapshot: the per-layer KV at `prompt.len()` plus the
/// prefill logits. In-memory only, never persisted.
struct GptOssPrefixCache {
    prompt: Vec<u32>,
    kv: Vec<crate::tensor::KvCache>,
    logits: Tensor,
}

fn gptoss_graph_enabled() -> bool {
    // CUDA-graph decode is CORRECT on the native substrate but NOT a speedup ->
    // stays OFF by default (same verdict the facade-era A/B reached).
    //
    // Replay-correctness arc (the graph replays bit-identically to eager today):
    //  • frozen embed: the captured token-embedding gather did not re-read the
    //    live in-place input_tok on replay. FIXED by moving the embed OUTSIDE
    //    the captured region: each token an eager embed lands in a persistent
    //    F32 `hidden_buf` (copy_f32_dev) and forward_inner's `x_in` starts the
    //    captured forward from it (eager path passes None -> byte-identical).
    //  • native-substrate replay ILLEGAL_ADDRESS: two captured-H2D/host-bounce
    //    hazards, both fixed at the substrate level  -
    //      (a) permute/broadcast kernels uploaded their dim/stride meta arrays
    //          per call from TEMPORARY host Vecs; a captured H2D memcpy node
    //          re-reads that freed host memory on replay -> garbage strides ->
    //          out-of-bounds gathers. Fixed by `native::cuda::dev_const_i32`
    //          (content-keyed persistent device constants, uploaded once
    //          outside capture).
    //      (b) the MoE routing sort (`sort_last_dim` on CUDA) is a host bounce
    //          (D2H + host sort + H2D) - capture-fatal AND a per-layer decode
    //          pipeline stall. Fixed in fused_moe via the one-warp device
    //          argsort (`moe_cuda::argsort_small_u32`, stable -> bit-identical),
    //          which is also an eager-decode win on its own.
    //    Transient device allocations inside the capture are replay-stable via
    //    the context capture arena (stream.alloc bump-allocates while armed;
    //    overflow forces a grow-and-recapture, so 0 MEM_ALLOC nodes).
    //  • VERDICT - the correct graph replays bit-identically (probation
    //    validates Δlogit = 0.0 vs eager) but decodes a few percent SLOWER
    //    than eager: gpt-oss decode is MoE/bandwidth-bound, not launch-bound,
    //    so the one-launch replay saves less than the graph path's fixed
    //    per-token cost (eager embed outside capture + input/pos/logits
    //    copies). Enabling it would regress decode.
    // The probation self-validation in forward_graph_decode keeps enabling it
    // *safe* (disables -> eager on mismatch, never emits garbage); it stays off
    // because it is slower, not because it is broken. Flag retained for A/Bs.
    false
}

impl GptOssModel {
    pub fn from_gguf<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
        dtype: DType,
        mm: Mm<'_>,
    ) -> Result<Self> {
        let cfg = GptOssConfig::from_gguf(content)?;
        let n_dev = devices.len().max(1);
        let dev_for =
            |layer: usize| -> &Device { &devices[(layer * n_dev / cfg.n_layers).min(n_dev - 1)] };
        let max_seq = cfg.context_length.max(cfg.rope_orig_context).max(8192);
        // Per-device RoPE tables (each layer's rope must live on its device).
        let mut rope: Vec<(Tensor, Tensor)> = Vec::with_capacity(n_dev);
        for d in devices {
            // Base rope - correct below `original_context`; YaRN scaling for longer
            // contexts is a follow-up that rescales the inverse frequencies.
            rope.push(crate::inference::model::rope::precomput_freqs_cis_host(
                cfg.head_dim,
                max_seq,
                cfg.rope_freq_base,
                d,
            )?);
        }

        let embed_dev = devices[0].clone();
        // Embedding table in the working dtype (BF16 = 2 bytes, same as F16  -
        // ~1.1 GB; gpt-oss is BF16-native so its norms/biases are BF16 too).
        let embed_t = content
            .tensor(reader, "token_embd.weight", &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(&embed_dev)?;
        let embed = Embedding::new(embed_t);

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            // The layer OWNS the device it lives on - the forward loop reads it to decide
            // where the hidden state has to travel across a split - and every weight below
            // loads onto that one copy instead of making its own.
            let layer_device = dev_for(i).clone();
            let device = &layer_device;
            let di = (i * n_dev / cfg.n_layers).min(n_dev - 1);
            // The rotation tables this layer reads: one pair at the stream's own width for
            // the fused kernels, one at the table's for the composed fallback. Cast once at
            // load rather than per token, and prepared before the block that holds them.
            let (cos, sin) = &rope[di];
            let (cos_f16, sin_f16) = (cos.to_dtype(DType::F16)?, sin.to_dtype(DType::F16)?);
            let (cos, sin) = (cos.clone(), sin.clone());
            let p = format!("blk.{i}");
            let attn = GptOssAttn {
                wq: ld_qm(content, reader, &format!("{p}.attn_q.weight"), device, mm)?,
                wk: ld_qm(content, reader, &format!("{p}.attn_k.weight"), device, mm)?,
                wv: ld_qm(content, reader, &format!("{p}.attn_v.weight"), device, mm)?,
                wo: ld_qm(content, reader, &format!("{p}.attn_out.weight"), device, mm)?,
                bq: ld_f32_opt(content, reader, &format!("{p}.attn_q.bias"), device),
                bk: ld_f32_opt(content, reader, &format!("{p}.attn_k.bias"), device),
                bv: ld_f32_opt(content, reader, &format!("{p}.attn_v.bias"), device),
                bo: ld_f32_opt(content, reader, &format!("{p}.attn_out.bias"), device),
                sinks: ld_f32(content, reader, &format!("{p}.attn_sinks"), device)?,
                cos,
                sin,
                n_head: cfg.n_head,
                n_kv_head: cfg.n_kv_head,
                head_dim: cfg.head_dim,
                // Interleaved SWA (fixes an every-layer-windowed bug that broke
                // long-context fidelity): with pattern p, layer i%p==p-1 is FULL
                // attention, the others sliding-window (llama.cpp set_swa_pattern
                // dense_first=false; gpt-oss = pattern 2 -> even SWA, odd full).
                sliding_window: (cfg.sliding_window > 0
                    && (cfg.swa_pattern == 0 || i % cfg.swa_pattern < cfg.swa_pattern - 1))
                    .then_some(cfg.sliding_window),
                // dim=2 (seq) cache, pre-allocated 4096 and growing by 4096 if
                // a longer context arrives. Lazily allocated per-layer-device.
                kv_cache: KvCache::new(2, 4096),
                kbuf: None,
                vbuf: None,
                kv_max: cfg.context_length.min(4096).max(256),
                cos_f16,
                sin_f16,
                cpu_f16_kv: None,
                sinks_f32: None,
            };
            let attn_norm = ld_norm(
                content,
                reader,
                &format!("{p}.attn_norm.weight"),
                cfg.rms_eps,
                device,
            )?;
            let attn_norm_w = ld_f32(content, reader, &format!("{p}.attn_norm.weight"), device)?;
            let ffn_norm = ld_norm(
                content,
                reader,
                &format!("{p}.ffn_norm.weight"),
                cfg.rms_eps,
                device,
            )?;
            let ffn_norm_w = ld_f32(content, reader, &format!("{p}.ffn_norm.weight"), device)?;
            let gate_ws = ld_f32(content, reader, &format!("{p}.ffn_gate_inp.weight"), device)?;
            let moe = FusedMoeGGUF {
                gate: Linear::new(gate_ws, None)?,
                gate_experts: Some(Arc::new(load_native(
                    content,
                    reader,
                    &format!("{p}.ffn_gate_exps.weight"),
                    device,
                    mm,
                )?)),
                up_experts: Some(Arc::new(load_native(
                    content,
                    reader,
                    &format!("{p}.ffn_up_exps.weight"),
                    device,
                    mm,
                )?)),
                down_experts: Arc::new(load_native(
                    content,
                    reader,
                    &format!("{p}.ffn_down_exps.weight"),
                    device,
                    mm,
                )?),
                gate_up_experts: None,
                act: Activation::Silu, // unused: swiglu_oai overrides the combine
                swiglu_oai: Some((1.702, 7.0)),
                // gpt-oss biases the router + every expert projection.
                gate_inp_bias: ld_f32_opt(
                    content,
                    reader,
                    &format!("{p}.ffn_gate_inp.bias"),
                    device,
                ),
                gate_exps_bias: ld_f32_opt(
                    content,
                    reader,
                    &format!("{p}.ffn_gate_exps.bias"),
                    device,
                ),
                up_exps_bias: ld_f32_opt(content, reader, &format!("{p}.ffn_up_exps.bias"), device),
                down_exps_bias: ld_f32_opt(
                    content,
                    reader,
                    &format!("{p}.ffn_down_exps.bias"),
                    device,
                ),
                norm_topk_prob: true,
                num_experts_per_tok: cfg.n_expert_used,
                dtype,
                cpu_experts: Arc::new(std::sync::OnceLock::new()),
            };
            if i == 0 || i + 1 == cfg.n_layers {
                tracing::info!("gptoss layer {i} on {:?}", device.location());
            }
            layers.push(GptOssLayer {
                attn_norm,
                attn,
                ffn_norm,
                moe,
                device: layer_device,
                attn_norm_w,
                ffn_norm_w,
                rms_eps: cfg.rms_eps,
            });
        }
        let last_dev = dev_for(cfg.n_layers - 1);
        let norm = ld_norm(content, reader, "output_norm.weight", cfg.rms_eps, last_dev)?;
        // lm_head: separate `output.weight`, else tied to the embedding table.
        let lm_head = match load_q8(content, reader, "output.weight", last_dev, mm) {
            Ok(t) => QMatMul::from_qtensor(t)?,
            Err(_) => ld_qm(content, reader, "token_embd.weight", last_dev, mm)?,
        };
        Ok(Self {
            embed,
            embed_dev,
            layers,
            norm,
            lm_head,
            dtype,
            pos_dev: None,
            input_tok: None,
            #[cfg(feature = "cuda")]
            graph: None,
            out_logits: None,
            logits_buf: None,
            hidden_buf: None,
            ws_pinned: false,
            graph_warmup: 0,
            graph_validations: 0,
            graph_disabled: false,
            prefix_cache: None,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, input_pos: usize) -> Result<Tensor> {
        // CUDA-graph decode is a CUDA-only optimization; CPU build runs eager.
        #[cfg(feature = "cuda")]
        {
            if input_pos == 0 && self.graph.is_some() {
                self.graph = None;
                self.out_logits = None;
                self.graph_warmup = 0;
                self.graph_validations = 0;
                if let Ok(cd) = self.embed_dev.as_cuda_device() {
                    cd.cuda_stream().context().free_capture_arena();
                }
            }
            let (_b, seq) = input_ids.dims2()?;
            let single_gpu = self
                .layers
                .iter()
                .all(|l| l.device.location() == self.embed_dev.location());
            if gptoss_graph_enabled()
                && !self.graph_disabled
                && seq == 1
                && self.embed_dev.is_cuda()
                && single_gpu
            {
                match self.forward_graph_decode(input_ids, input_pos) {
                    Ok(l) => return Ok(l),
                    Err(e) => tracing::warn!("gptoss graph decode failed ({e}); eager fallback"),
                }
            }
        }
        self.forward_inner(input_ids, input_pos, None, None)
    }

    /// Snapshot every layer's KV cache at position `prompt.len()` plus the prefill
    /// logits, for exact-prefix reuse. Call AFTER prefilling `[0, prompt.len())`.
    ///
    /// gpt-oss cannot reuse a prompt via the `trim_kv` prefix mechanism the dense
    /// backends use: that path re-prefills the boundary token through the decode
    /// attention (windowed sink-softmax), which is not numerically identical to the
    /// prefill attention once the prompt exceeds the sliding window - so the first
    /// sampled token, and hence the whole greedy continuation, would diverge from a
    /// cold run. Storing the prefill logits and replaying them sidesteps the decode
    /// path entirely, so a hit is greedy-identical to cold by construction. The KV
    /// is deep-copied (not aliased) so later decode/generation can't corrupt it.
    pub fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> Result<()> {
        let mut kv = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            kv.push(l.attn.kv_cache.deep_copy()?);
        }
        self.prefix_cache = Some(GptOssPrefixCache {
            prompt: prompt.to_vec(),
            kv,
            logits: logits.affine(1.0, 0.0)?,
        });
        Ok(())
    }

    /// On an EXACT prompt match, restore every layer's KV and return the stored
    /// prefill logits (decode resumes at `prompt.len()`). Else `None`. The lazily
    /// built decode stores are dropped so they re-seed from the restored KV, and
    /// any captured decode graph is invalidated - mirroring the fresh-sequence
    /// reset the model performs at `input_pos == 0`.
    pub fn try_restore_prefix(&mut self, prompt: &[u32]) -> Result<Option<Tensor>> {
        match &self.prefix_cache {
            Some(pc) if pc.prompt.as_slice() == prompt => {}
            _ => return Ok(None),
        }
        let pc = self.prefix_cache.take().unwrap();
        for (layer, snap) in self.layers.iter_mut().zip(pc.kv.iter()) {
            layer.attn.kv_cache = snap.deep_copy()?;
            layer.attn.cpu_f16_kv = None;
            layer.attn.kbuf = None;
            layer.attn.vbuf = None;
        }
        #[cfg(feature = "cuda")]
        {
            self.graph = None;
            self.out_logits = None;
            self.graph_warmup = 0;
            self.graph_validations = 0;
            if let Ok(cd) = self.embed_dev.as_cuda_device() {
                cd.cuda_stream().context().free_capture_arena();
            }
        }
        let logits = pc.logits.affine(1.0, 0.0)?;
        self.prefix_cache = Some(pc);
        Ok(Some(logits))
    }

    fn forward_inner(
        &mut self,
        input_ids: &Tensor,
        input_pos: usize,
        shared_pos: Option<&Tensor>,
        x_in: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, seq) = input_ids.dims2()?;
        // New sequence (prefill at pos 0): drop any KV from a prior request so
        // the caches don't accumulate across generations.
        if input_pos == 0 {
            for layer in self.layers.iter_mut() {
                layer.attn.kv_cache.reset();
                layer.attn.kbuf = None;
                layer.attn.vbuf = None;
                // The decode-only F16 store must be dropped on a fresh sequence
                // too: full-attention layers keep every position (no window to
                // roll it out), so a prior request's store would otherwise seed
                // into this one when the new prompt is shorter. The store's own
                // `input_pos == 0` reset never fires during decode (first decode
                // token is at `prompt_len > 0`).
                layer.attn.cpu_f16_kv = None;
            }
        }
        let mut x = match x_in {
            // CUDA-graph: start from a persistent F32 hidden buffer filled by
            // an EAGER embed OUTSIDE the captured region. The captured region's first
            // op is this cast, reading the live hidden_buf (refreshed in place per
            // token) - NOT the embed gather, whose input_tok read froze under capture.
            Some(h) => h.to_dtype(self.dtype)?,
            None => {
                let ids = input_ids.to_device(&self.embed_dev)?;
                let mut x = self.embed.forward(&ids)?;
                if x.dtype() != self.dtype {
                    x = x.to_dtype(self.dtype)?;
                }
                x
            }
        };
        for layer in self.layers.iter_mut() {
            if x.device().location() != layer.device.location() {
                x = x.to_device(&layer.device)?;
            }
            x = layer.forward(&x, input_pos, shared_pos)?;
        }
        let x = self.norm.forward(&x.to_dtype(DType::F32)?)?; // F32 final norm
        let x = x.i((.., seq - 1, ..))?.to_dtype(self.dtype)?; // last token, back to BF16 for lm_head

        self.lm_head.forward(&x)?.to_dtype(DType::F32)
    }

    /// CUDA-graph decode (single-GPU, env-gated). Captured once after a few warmup
    /// decode tokens (capturing right after prefill trips a transition wild-ptr, per
    /// lfm2's bisect), replayed per token. Per token: write input id (D2D) + position
    /// (set_i32 in place) OUTSIDE the captured region, then one graph.launch replaces
    /// the forward's individual kernel launches. Replays bit-identically but measures
    /// slightly slower than eager (see gptoss_graph_enabled). Mirrors lfm2.
    #[cfg(feature = "cuda")]
    fn forward_graph_decode(&mut self, input_ids: &Tensor, input_pos: usize) -> Result<Tensor> {
        use crate::inference::kernel::fused::set_i32_inplace;
        use crate::tensor::cuda_ext;
        let dev = self.embed_dev.clone();
        let cd = dev.as_cuda_device()?;
        let stream = cd.cuda_stream();
        let ctx = stream.context();
        let _ = ctx.bind_to_thread();
        let msg = |s: &str| crate::tensor::Error::msg(s.to_string());
        if self.input_tok.is_none() {
            self.input_tok = Some(Tensor::zeros_on((1usize, 1usize), DType::U32, &dev)?);
        }
        if self.pos_dev.is_none() {
            self.pos_dev = Some(Tensor::zeros_on((1usize,), DType::I32, &dev)?);
        }
        let (input_tok, pos_dev) = (
            self.input_tok.clone().unwrap(),
            self.pos_dev.clone().unwrap(),
        );
        // Advance token + position IN PLACE, outside the captured region: token is
        // copied device->device (no D2H sync); position is a host int -> device.
        let it_src = input_ids.to_device(&dev)?;
        crate::inference::kernel::fused::copy_u32_dev(&it_src, &input_tok)?;
        set_i32_inplace(&pos_dev, input_pos as i32)?;
        // WARMUP: run the first 3 decode tokens UNCAPTURED so the capture is primed
        // by decode-mode state (capturing on the first decode token, primed by the
        // seq>1 prefill, trips a transition wild-pointer -> replay ILLEGAL_ADDRESS).
        if self.graph.is_none() && self.graph_warmup < 3 {
            let logits = self.forward_inner(&input_tok, input_pos, Some(&pos_dev), None)?;
            // Allocate the persistent logits buffer (outside any capture arena, so
            // it's a stable address) sized to the real logits shape, on first use.
            if self.logits_buf.is_none() {
                self.logits_buf = Some(logits.zeros_like()?);
            }
            self.graph_warmup += 1;
            return Ok(logits);
        }
        // Embed OUTSIDE the captured region into a persistent (non-arena) F32
        // hidden_buf - EVERY token (the capture token + every replay). The captured
        // forward starts from this buffer (forward_inner x_in) so its embed gather,
        // whose input_tok read froze under capture, is excluded from the graph;
        // here the LIVE in-place input_tok is embedded eagerly each token. Allocated
        // before begin_capture -> stable address; the in-place copy_f32_dev refreshes
        // it pre-launch so each replay sees the current token's embedding.
        let x_emb_f32 = {
            let ids = input_tok.to_device(&self.embed_dev)?;
            self.embed.forward(&ids)?.to_dtype(DType::F32)?
        };
        if self.hidden_buf.is_none() {
            self.hidden_buf = Some(x_emb_f32.zeros_like()?);
        }
        let hbuf = self.hidden_buf.clone().unwrap();
        crate::inference::kernel::fused::copy_f32_dev(&x_emb_f32, &hbuf)?;
        if self.graph.is_none() {
            if !self.ws_pinned {
                cuda_ext::pin_cublas_workspace(&dev, 64 << 20)?;
                self.ws_pinned = true;
            }
            let free_vram = ctx.mem_get_info().map(|(f, _)| f).unwrap_or(0);
            // Ceiling: leave 1/8 of free VRAM as headroom for the per-token sampling
            // allocations that live outside the arena.
            let ceiling = free_vram.saturating_sub(free_vram / 8).max(1 << 20);
            let it = input_tok.clone();
            let pd = pos_dev.clone();
            let hb = hbuf.clone();
            // Persistent (non-arena) destination for the in-graph logits copy.
            let lbuf = self.logits_buf.clone();
            // Seed at the smallest bump unit and double; the effective size is found
            // by the overflow signal, not chosen.
            let mut arena_bytes = (1usize << 20).min(ceiling);
            let (g, logits) = loop {
                ctx.begin_capture_arena(arena_bytes)
                    .map_err(|e| msg(&format!("arena {e:?}")))?;
                let cap: Result<(cuda_ext::CudaGraph, Tensor)> = (|| {
                    cuda_ext::begin_capture(&stream)?;
                    let logits = self.forward_inner(&it, input_pos, Some(&pd), Some(&hb))?;
                    // Final captured op: copy the lm_head logits (a fresh arena
                    // tensor) into the persistent buffer so each replay deposits
                    // them at a stable address. out_logits then points at that
                    // buffer, not the arena tensor whose identity replay can lose.
                    let out = match lbuf.as_ref() {
                        Some(lb) => {
                            crate::inference::kernel::fused::copy_f32_dev(&logits, lb)?;
                            lb.clone()
                        }
                        None => logits,
                    };
                    let g = cuda_ext::end_capture(&stream)?.ok_or_else(|| msg("no graph"))?;
                    Ok((g, out))
                })();
                let (_peak, of) = ctx.end_capture_arena();
                let (g, logits) = match cap {
                    Ok(x) => x,
                    Err(e) => {
                        ctx.free_capture_arena();
                        return Err(e);
                    }
                };
                if of == 0 {
                    tracing::warn!(
                        "🟦 gptoss GRAPH arena sized to {}MB (free {}MB)",
                        arena_bytes >> 20,
                        free_vram >> 20
                    );
                    break (g, logits);
                }
                // Overflowed: the forward spilled past the arena into real
                // cuMemAllocs (-> MEM_ALLOC nodes that relocate on replay). Discard
                // this graph, grow, and recapture - unless we're already at the
                // VRAM ceiling, in which case bail to the eager path (correct, just
                // unaccelerated).
                drop(g);
                ctx.free_capture_arena();
                if arena_bytes >= ceiling {
                    return Err(msg(&format!(
                        "arena overflow at VRAM ceiling {}MB - eager fallback",
                        ceiling >> 20
                    )));
                }
                arena_bytes = (arena_bytes * 2).min(ceiling);
            };
            g.upload().map_err(|e| msg(&format!("upload {e:?}")))?;
            self.graph = Some(g);
            self.out_logits = Some(logits);
        }
        if let Some(g) = self.graph.as_ref() {
            g.launch().map_err(|e| msg(&format!("launch {e:?}")))?;
        }
        // Probation self-validation: each of the first GRAPH_VALIDATIONS replays
        // MUST match an eager forward at the same (token, pos) - multiple replays
        // are checked because a replay-instability (relocated scratch, stale host
        // source of a captured memcpy) can pass once and corrupt later. Compute
        // the replay argmax FIRST (before the eager re-run can touch any shared
        // buffer), then run eager (its devpos KV write at input_pos is idempotent)
        // and compare. On mismatch permanently disable the graph and return the
        // CORRECT eager logits, so enabling LOKEN_GPTOSS_GRAPH can never emit
        // replay garbage (worst case: no speedup).
        const GRAPH_VALIDATIONS: usize = 3;
        if self.graph_validations < GRAPH_VALIDATIONS && !self.graph_disabled {
            let _ = stream.synchronize();
            // Snapshot the replay logits (clone to host) BEFORE the eager re-run.
            let rep = match self.out_logits.as_ref() {
                Some(r) => Some(r.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?),
                None => None,
            };
            let rep_argmax = rep.as_ref().map(|v| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            });
            let eag = self.forward_inner(&input_tok, input_pos, Some(&pos_dev), None)?;
            let _ = stream.synchronize();
            let eag_argmax = eag.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
            // Quantify the divergence: max |logit diff| tells us tiny FP/scratch
            // noise (argmax flip on near-ties) vs a large KV/routing error.
            if let Some(rv) = rep.as_ref() {
                let ev = eag.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                if rv.len() == ev.len() {
                    let nan = rv.iter().filter(|x| x.is_nan()).count();
                    let maxd = rv
                        .iter()
                        .zip(ev.iter())
                        .filter(|(a, b)| a.is_finite() && b.is_finite())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let rmax = rv
                        .iter()
                        .cloned()
                        .filter(|x| x.is_finite())
                        .fold(f32::MIN, f32::max);
                    tracing::warn!("🟦 gptoss GRAPH validate: rep_argmax={:?} eag_argmax={} max|Δlogit(finite)|={:.4} rep_finite_max={:.3} rep_nan={}/{}",
                        rep_argmax, eag_argmax, maxd, rmax, nan, rv.len());
                }
            }
            let force = false;
            if force || rep_argmax == Some(eag_argmax) {
                self.graph_validations += 1;
            } else {
                tracing::warn!("🟦 gptoss GRAPH replay!=eager (argmax {:?} vs {}) - disabling graph, eager path",
                    rep_argmax, eag_argmax);
                self.graph = None;
                self.graph_disabled = true;
                ctx.free_capture_arena();
                return Ok(eag);
            }
        }
        // No per-token sync: the engine syncs when it reads the logits for sampling,
        // and the next token's input depends on that sample (no race).
        self.out_logits.clone().ok_or_else(|| msg("no out_logits"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    #[test]
    fn sink_softmax_absorbs_mass() {
        // 1 head, q=1, kv=2; scores [1.0, 2.0], sink 0.0.
        // m = max(1,2,0) = 2. exp = [e^-1, e^0] = [0.3679, 1.0]; exp_sink = e^-2 = 0.1353.
        // denom = 0.3679 + 1.0 + 0.1353 = 1.5032.
        // w = [0.2448, 0.6652]; sum = 0.9100 (sink absorbed 0.0900).
        let dev = Device::Cpu;
        // from_vec keeps this substrate-neutral (the native compat shim's
        // IntoTensor doesn't cover 4-deep nested arrays).
        let scores = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 1, 2), &dev).unwrap();
        let sinks = Tensor::new(&[0.0f32], &dev).unwrap();
        let w = softmax_last_dim_with_sinks(&scores, None, &sinks, 1.0).unwrap();
        let v: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        assert!((v[0] - 0.2448).abs() < 1e-3, "w0={}", v[0]);
        assert!((v[1] - 0.6652).abs() < 1e-3, "w1={}", v[1]);
        let sum: f32 = v.iter().sum();
        assert!((sum - 0.9100).abs() < 1e-3, "sum={sum}");
    }

    #[test]
    fn zero_sink_logit_still_absorbs() {
        // A very negative sink ≈ no absorption -> rows ~sum to 1 (≈ plain softmax).
        let dev = Device::Cpu;
        let scores = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 1, 3), &dev).unwrap();
        let sinks = Tensor::new(&[-1e30f32], &dev).unwrap();
        let w = softmax_last_dim_with_sinks(&scores, None, &sinks, 1.0).unwrap();
        let sum: f32 = w
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
    }
}
