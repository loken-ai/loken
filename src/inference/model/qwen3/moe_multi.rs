//! Multi-device Qwen3-MoE (qwen3-coder) loader.
//!
//! Splits transformer layers across multiple CUDA GPUs so models that don't
//! fit on a single GPU (e.g. 30B-A3B ≈ 18.5 GB on 16 GB cards) can run at
//! near-full GPU speed via per-layer placement and cross-device tensor
//! transfer at split boundaries.
//!
//! Reuses the reference `FusedMoeGGUF`, `QuantizedAttention`, and `RmsNorm`
//! implementations - we just control where weights land.
//!
//! Correctness is identical to single-device `GGUFQWenMoE`: each layer runs
//! on its assigned device, and the hidden state is transferred (GPU↔GPU
//! copy) whenever the next layer is on a different GPU.

#![allow(clippy::too_many_arguments)]

use crate::inference::fused_moe::FusedMoeGGUF;
use crate::tensor::layer::Embedding;
use crate::tensor::layer::Linear;
use crate::tensor::layer::RmsNorm;
use crate::tensor::ops::host_f32;
use crate::tensor::ops::Activation;
use crate::tensor::quantized::{gguf_file, QTensor};
use crate::tensor::ConcatKvCache;
use crate::tensor::{DType, Device, IndexOp, Result, Tensor};
use std::io::{Cursor, Read, Seek};
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::Arc;

/// RoPE cos/sin tables for qwen3 attention (native; was
/// the quantized Qwen3 rotary embedding). Precomputes
/// cos/sin and exposes them as Tensors (for the fused inline-RoPE kernels) and
/// flat f32 slices (for zero-alloc per-position decode). `apply` uses the native
/// fused `rope` op.
pub struct RotaryEmbedding {
    sin: crate::tensor::Tensor,
    cos: crate::tensor::Tensor,
    cos_f32: Vec<f32>,
    sin_f32: Vec<f32>,
    half_d: usize,
}

impl RotaryEmbedding {
    pub fn sin_table(&self) -> &crate::tensor::Tensor {
        &self.sin
    }
    pub fn cos_table(&self) -> &crate::tensor::Tensor {
        &self.cos
    }

    pub fn new(
        dtype: crate::tensor::DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &crate::tensor::Device,
    ) -> crate::tensor::Result<Self> {
        // The table is the outer product of the positions with the angular frequencies: one
        // row per position, one column per pair of a head's dimensions.
        //
        // Built in single precision and narrowed only at the end. A position is an integer,
        // and half precision stops representing every integer at 2048: position 2049 becomes
        // 2048 and 2051 becomes 2052, so from there on pairs of neighbouring tokens are given
        // the same rotation and the attention stops being able to tell them apart. What comes
        // out is a model that is fluent and has started to misspell. The sines and cosines
        // themselves live in [-1, 1], where the narrow type has precision to spare, so the
        // cast belongs after the trigonometry rather than before it.
        let angular = crate::inference::model::rope::inverse_frequencies_f64(head_dim, rope_theta);
        let pairs = angular.len();
        let angular = Tensor::from_vec(angular, (1, pairs), dev)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0f32, max_position_embeddings as f32)?
            .reshape((max_position_embeddings, 1))?
            .to_device(dev)?;
        let angles = positions.matmul(&angular)?;
        let (sin_f, cos_f) = (angles.sin()?, angles.cos()?);
        // The host copies are taken before the cast: the CPU decode path has no reason to
        // read back a narrowed number when the wide one is in hand.
        let (cos_f32, sin_f32) = (host_f32(&cos_f)?, host_f32(&sin_f)?);
        Ok(Self {
            cos_f32,
            sin_f32,
            sin: sin_f.to_dtype(dtype)?,
            cos: cos_f.to_dtype(dtype)?,
            half_d: head_dim / 2,
        })
    }

    /// Rotate `q` and `k` - `[b, heads, seq, head_dim]` - by the positions starting at
    /// `offset`: the table's rows for those positions, taken at the tensors' own width.
    pub fn apply(
        &self,
        q: &crate::tensor::Tensor,
        k: &crate::tensor::Tensor,
        offset: usize,
    ) -> crate::tensor::Result<(crate::tensor::Tensor, crate::tensor::Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let rows = |table: &Tensor| -> crate::tensor::Result<Tensor> {
            table.narrow(0, offset, seq_len)?.to_dtype(q.dtype())
        };
        let (cos, sin) = (rows(&self.cos)?, rows(&self.sin)?);
        let rotate = |t: &Tensor| -> crate::tensor::Result<Tensor> {
            crate::tensor::ops::rope(&t.contiguous()?, &cos, &sin)
        };
        Ok((rotate(q)?, rotate(k)?))
    }

    /// One position's row of the table, borrowed: the `head_dim/2` cosines and sines that
    /// position's head pairs turn by, handed over without allocating anything.
    #[inline]
    pub fn cos_sin_at(&self, pos: usize) -> (&[f32], &[f32]) {
        let row = pos * self.half_d..(pos + 1) * self.half_d;
        (&self.cos_f32[row.clone()], &self.sin_f32[row])
    }
}

static Q4_PROF_SCORE_US: AtomicU64 = AtomicU64::new(0);
static Q4_PROF_AFFINE_US: AtomicU64 = AtomicU64::new(0);
static Q4_PROF_SOFTMAX_US: AtomicU64 = AtomicU64::new(0);
static Q4_PROF_OUTPUT_US: AtomicU64 = AtomicU64::new(0);
static Q4_PROF_WO_US: AtomicU64 = AtomicU64::new(0);
static Q4_PROF_LAYER: AtomicUsize = AtomicUsize::new(0);

// ------------------------------------------------------------
// Attention - same math as the reference QuantizedAttention, but all weights live
// on a specific device (attn_q/k/v/o as QMatMul, biases as Tensor, norms as
// RmsNorm). Inputs must already be on the same device as the weights.
// ------------------------------------------------------------

struct Attn {
    /// Fused Q+K+V weight (byte-concat along output dim) - populated
    /// when all three weights share the same quantization. ONE matmul
    /// produces a packed [seq, n_q*head_dim + 2*n_kv*head_dim] output
    /// that we narrow into q/k/v slices. Reduces 3 launches/layer to 1.
    wqkv: Option<crate::tensor::quantized::QMatMul>,
    /// Underlying raw QTensor for the fused QKV - kept separately so
    /// the rms_qmatmul fused kernel (which calls `device_ptr()` on the
    /// QTensor) can be invoked without going through QMatMul.
    /// Fused Q+K weight, used when V's quantization differs from Q/K's
    /// (common in K-quants where V is stored at higher precision  -
    /// e.g. Q4_K_M has Q,K=Q4K and V=Q6K). Reduces 3 launches/layer to
    /// 2. Mutually exclusive with wqkv.
    wqk: Option<crate::tensor::quantized::QMatMul>,
    /// Fused output dim for the Q slice (n_q * head_dim).
    wqkv_q_dim: usize,
    /// Fused output dim for each of K and V (n_kv * head_dim each).
    wqkv_kv_dim: usize,
    wq: Option<crate::tensor::quantized::QMatMul>,
    wk: Option<crate::tensor::quantized::QMatMul>,
    wv: Option<crate::tensor::quantized::QMatMul>,
    wo: crate::tensor::quantized::QMatMul,
    bq: Option<Tensor>,
    bk: Option<Tensor>,
    bv: Option<Tensor>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    num_kv_groups: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    dtype: DType,
    kv_cache: ConcatKvCache,
    /// Optional Q8_0 KV cache - populated alongside `kv_cache` when
    /// kv_quant=Q8. The decode fast path (seq=1) reads from this
    /// cache via the new gemv kernels, bypassing the F-dtype matmul.
    #[cfg(feature = "cuda")]
    q8_kv_cache: Option<crate::inference::cache::q8_kv::Q8KvCache>,
    /// Q4 variant - same role as `q8_kv_cache` but with half the bytes per
    /// element. Only one of {q8_kv_cache, q4_kv_cache} is populated at a
    /// time depending on the load-time `kv_quant` config.
    #[cfg(feature = "cuda")]
    q4_kv_cache: Option<crate::inference::cache::q4_kv::Q4KvCache>,
    /// CPU decode fast path: f16 GQA KV + single-pass online-softmax
    /// attention (`cpu_f16_kv`), replacing the 5-dispatch tensor sdpa and its
    /// per-step intermediates (profiled: attention = 852 µs/layer at ctx≈50 vs
    /// ~300 µs of weight reads - most of qwen3-coder's -14% CPU decode gap).
    /// Mirrors `kv_cache` (which multi-token prefill/PLD-verify still reads);
    /// lazily built on the first CPU forward.
    cpu_f16_kv: Option<crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
    /// q_norm/k_norm weights as flat f32, cached for the fused CPU post-QKV
    /// pass (avoids 2 tensor reads per layer per token).
    qk_norm_f32: Option<(Vec<f32>, Vec<f32>)>,
}

impl Attn {
    fn forward(&mut self, x: &Tensor, input_pos: usize) -> Result<Tensor> {
        self.forward_inner(x, input_pos)
    }

    /// CPU decode phase-2 fusion: pure-Rust mirror of
    /// `moe_cuda::attn_post_qkv_decode` (which gave +5.7% on GPU for the same
    /// reason). One pass over the flat QKV output: per-head rmsnorm
    /// (q_norm/k_norm), non-interleaved RoPE at `pos`, 1/sqrt(hd) folded into
    /// q. Returns (q_scaled [nh*hd], k_roped [nkv*hd], v [nkv*hd]) ready for
    /// `CpuF16Kv` - replacing ~10 small tensor dispatches per layer per token
    /// (narrowx3, reshapex3, rmsnormx2, to_dtypex3, ropex2), the profiled
    /// ~250µs/layer excess vs ollama.
    fn cpu_post_qkv_fused(
        &self,
        qkv: &[f32],
        pos: usize,
        qn_w: &[f32],
        kn_w: &[f32],
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (nh, nkv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        let half = hd / 2;
        let (cos, sin) = self.rotary_emb.cos_sin_at(pos);
        let q_scale = (1.0 / (hd as f64).sqrt()) as f32;
        let mut qo = vec![0f32; nh * hd];
        let mut ko = vec![0f32; nkv * hd];
        // rmsnorm + neox rope + optional scale, one head at a time.
        let do_head = |src: &[f32], w: &[f32], dst: &mut [f32], scale: f32| {
            let ss: f32 = src.iter().map(|&x| x * x).sum();
            let inv = 1.0f32 / (ss / hd as f32 + eps).sqrt();
            for i in 0..half {
                let a = src[i] * inv * w[i];
                let b = src[i + half] * inv * w[i + half];
                dst[i] = (a * cos[i] - b * sin[i]) * scale;
                dst[i + half] = (a * sin[i] + b * cos[i]) * scale;
            }
        };
        for h in 0..nh {
            do_head(
                &qkv[h * hd..(h + 1) * hd],
                qn_w,
                &mut qo[h * hd..(h + 1) * hd],
                q_scale,
            );
        }
        let kbase = nh * hd;
        for h in 0..nkv {
            do_head(
                &qkv[kbase + h * hd..kbase + (h + 1) * hd],
                kn_w,
                &mut ko[h * hd..(h + 1) * hd],
                1.0,
            );
        }
        let vbase = kbase + nkv * hd;
        let vo = qkv[vbase..vbase + nkv * hd].to_vec();
        (qo, ko, vo)
    }

    fn forward_inner(&mut self, x: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (_b, seq, _) = x.dims3()?;
        let in_dtype = x.dtype();

        // FAST PATH: fused QKV matmul + fused post-QKV (q_norm +
        // k_norm + RoPE + cast) for single-token decode on CUDA.
        // Replaces ~5 small ops (q_norm, k_norm, q.to_dtype,
        // k.to_dtype, v.to_dtype, rope.applyx2...) with a single launch.
        // qwen3-coder bench: 88 -> 93 tok/s (+5.7%). Default-on whenever
        // applicable (single-token decode, CUDA, fused wqkv weight,
        // no bias).
        let want_fused_apq = seq == 1
            && x.device().is_cuda()
            && self.wqkv.is_some()
            && self.bq.is_none()
            && self.bk.is_none()
            && self.bv.is_none();

        // CPU decode fast path: fused QKV matmul, then the
        // pure-Rust post-QKV pass (rmsnorm+rope+scale) straight into the f16
        // GQA cache and its single-pass attention. Roped K + V are mirrored
        // into ConcatKvCache so multi-token PLD-verify history stays intact.
        let want_cpu_fused = seq == 1
            && x.device().is_cpu()
            && self.wqkv.is_some()
            && self.bq.is_none()
            && self.bk.is_none()
            && self.bv.is_none();
        if want_cpu_fused {
            let qkv = self.wqkv.as_ref().unwrap().forward(x)?;
            let qkv_v = host_f32(&qkv)?;
            if self.qk_norm_f32.is_none() {
                // Read once per layer, not once per token: the two weights never change.
                self.qk_norm_f32 = Some((
                    host_f32(self.q_norm.weight())?,
                    host_f32(self.k_norm.weight())?,
                ));
            }
            let eps = self.q_norm.eps() as f32;
            let (qn_w, kn_w) = self.qk_norm_f32.as_ref().unwrap();
            let (qf, kf, vf) = self.cpu_post_qkv_fused(&qkv_v, input_pos, qn_w, kn_w, eps);
            // NO per-token ConcatKvCache mirror here - that O(ctx) concat copy
            // cost ~11-15ms/tok at 2.5K ctx. The F-dtype cache is instead
            // resynced FROM the f16 store on demand, in the rare multi-token
            // calls that read history (see attention_after_qkv_inner).
            if self.cpu_f16_kv.is_none() {
                self.cpu_f16_kv = Some(crate::inference::cache::cpu_f16_kv::CpuF16Kv::new(
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                    None,
                ));
            }
            let cache = self.cpu_f16_kv.as_mut().unwrap();
            cache.append(&kf, &vf)?;
            let mut out = vec![0f32; self.n_head * self.head_dim];
            cache.attention(&qf, 1.0, &mut out)?; // q pre-scaled in the fused pass
            let ctx_t = Tensor::from_vec(out, (1, 1, self.n_head * self.head_dim), &x.device())?;
            let ctx_t = if ctx_t.dtype() == in_dtype {
                ctx_t
            } else {
                ctx_t.to_dtype(in_dtype)?
            };
            return self.wo.forward(&ctx_t);
        }

        // The fused fast path produces (q, k, v) already shaped
        // [1, n_head/n_kv_head, 1, head_dim] in self.dtype with q_norm,
        // k_norm, and RoPE already applied. The unfused path produces
        // raw F32 (q, k, v) that still need norm/RoPE/cast.
        let (q, k, v, fused_done) = if want_fused_apq {
            let qkv = self.wqkv.as_ref().unwrap().forward(x)?;
            let total = self.wqkv_q_dim + 2 * self.wqkv_kv_dim;
            let qkv_flat = qkv.reshape((qkv.elem_count() / total, total))?;
            // Pass the FULL cos/sin tables; the kernel uses rope_pos to
            // index the right row. Avoids the narrow -> as_cuda_slice
            // offset-mismatch trap entirely.
            let cos = self.rotary_emb.cos_table().clone();
            let sin = self.rotary_emb.sin_table().clone();
            let q_norm_w = self.q_norm.weight();
            let k_norm_w = self.k_norm.weight();
            let q_norm_w = if q_norm_w.dtype() == DType::F32 {
                q_norm_w.clone()
            } else {
                q_norm_w.to_dtype(DType::F32)?
            };
            let k_norm_w = if k_norm_w.dtype() == DType::F32 {
                k_norm_w.clone()
            } else {
                k_norm_w.to_dtype(DType::F32)?
            };
            let rms_eps = self.q_norm.eps() as f32;
            let cos = if cos.dtype() == self.dtype {
                cos
            } else {
                cos.to_dtype(self.dtype)?
            };
            let sin = if sin.dtype() == self.dtype {
                sin
            } else {
                sin.to_dtype(self.dtype)?
            };
            // Fold attention's 1/sqrt(d) scale into Q at the same time
            // as norm + RoPE - saves the explicit `scores.affine(scale)`
            // launch (Q4 attn_scores still produces unscaled scores; the
            // pre-scaled Q means scores = Q'.K^T = scale x original).
            let q_scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
            // When the Q4 KV cache is active, the downstream score
            // kernel takes Q in F32 - produce it directly here so we
            // don't need a to_dtype(F32) launch on Q. Saves 1 launch
            // per layer per token.
            #[cfg(feature = "cuda")]
            let want_q_f32 = self.q4_kv_cache.is_some();
            #[cfg(not(feature = "cuda"))]
            let want_q_f32 = false;
            let (q_h, k_h, v_h) = if want_q_f32 {
                crate::inference::moe_cuda::attn_post_qkv_decode_qf32(
                    &qkv_flat,
                    &q_norm_w,
                    &k_norm_w,
                    &cos,
                    &sin,
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                    input_pos,
                    rms_eps,
                    q_scale,
                    self.dtype,
                    0,
                )?
            } else {
                crate::inference::moe_cuda::attn_post_qkv_decode(
                    &qkv_flat,
                    &q_norm_w,
                    &k_norm_w,
                    &cos,
                    &sin,
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                    input_pos,
                    rms_eps,
                    q_scale,
                    self.dtype,
                    0,
                )?
            };
            let q = q_h.reshape((1, self.n_head, 1, self.head_dim))?;
            let k = k_h.reshape((1, self.n_kv_head, 1, self.head_dim))?;
            let v = v_h.reshape((1, self.n_kv_head, 1, self.head_dim))?;
            (q, k, v, true)
        } else {
            // Slow path - see the original code below; the let _ here is
            // a placeholder that gets shadowed inside the else branch.
            let dummy = crate::tensor::Tensor::zeros_on((1,), DType::F32, &x.device())?;
            (dummy.clone(), dummy.clone(), dummy, false)
        };

        if fused_done {
            // Skip directly to KV-cache append + attention computation.
            return self.attention_after_qkv(q, k, v, input_pos, in_dtype, seq, true);
        }

        let (q, k, v) = if let Some(wqkv) = self.wqkv.as_ref() {
            // Single fused matmul: input @ [n_q*hd + 2*n_kv*hd, hidden]^T
            // -> [seq, n_q*hd + 2*n_kv*hd]. Narrow into the three slices.
            let qkv = wqkv.forward(x)?;
            let qd = self.wqkv_q_dim;
            let kd = self.wqkv_kv_dim;
            let q = qkv.narrow(crate::tensor::D::Minus1, 0, qd)?;
            let k = qkv.narrow(crate::tensor::D::Minus1, qd, kd)?;
            let v = qkv.narrow(crate::tensor::D::Minus1, qd + kd, kd)?;
            if seq == 1 {
                (q, k, v)
            } else {
                (q.contiguous()?, k.contiguous()?, v.contiguous()?)
            }
        } else if let Some(wqk) = self.wqk.as_ref() {
            // Partial fusion: Q+K packed, V separate (typical with
            // K-quant V at higher precision than Q/K).
            let qk = wqk.forward(x)?;
            let qd = self.wqkv_q_dim;
            let kd = self.wqkv_kv_dim;
            let q = qk.narrow(crate::tensor::D::Minus1, 0, qd)?;
            let k = qk.narrow(crate::tensor::D::Minus1, qd, kd)?;
            let (q, k) = if seq == 1 {
                (q, k)
            } else {
                (q.contiguous()?, k.contiguous()?)
            };
            let v = self.wv.as_ref().unwrap().forward(x)?;
            (q, k, v)
        } else {
            let q = self.wq.as_ref().unwrap().forward(x)?;
            let k = self.wk.as_ref().unwrap().forward(x)?;
            let v = self.wv.as_ref().unwrap().forward(x)?;
            (q, k, v)
        };
        let q = if let Some(b) = &self.bq {
            q.broadcast_add(b)?
        } else {
            q
        };
        let k = if let Some(b) = &self.bk {
            k.broadcast_add(b)?
        } else {
            k
        };
        let v = if let Some(b) = &self.bv {
            v.broadcast_add(b)?
        } else {
            v
        };

        // For decode (seq=1) the transpose+contiguous produces the same data
        // layout as a direct reshape (size-1 dim is a no-op). Skip the copy
        // by reshaping directly into the heads-first layout. Prefill (seq>1)
        // still needs the transpose to reorder head/seq dims.
        let (q, k, v) = if seq == 1 {
            (
                q.reshape((1, self.n_head, 1, self.head_dim))?,
                k.reshape((1, self.n_kv_head, 1, self.head_dim))?,
                v.reshape((1, self.n_kv_head, 1, self.head_dim))?,
            )
        } else {
            (
                q.reshape((1, seq, self.n_head, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
                k.reshape((1, seq, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
                v.reshape((1, seq, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
            )
        };

        // per-head norm on Q, K (qwen3-specific)
        let q_flat = q.reshape((q.elem_count() / self.head_dim, self.head_dim))?;
        let k_flat = k.reshape((k.elem_count() / self.head_dim, self.head_dim))?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((1, self.n_head, seq, self.head_dim))?;
        let k = k_flat.reshape((1, self.n_kv_head, seq, self.head_dim))?;

        // Down to the layer's own width before the rotation and the score matmul: the norms
        // above ran wide, and the three tensors travel together from here on.
        let stream = |t: Tensor| t.to_dtype(self.dtype);
        let (q, k, v) = (stream(q)?, stream(k)?, stream(v)?);
        let (q, k) = self.rotary_emb.apply(&q, &k, input_pos)?;
        // Slow path falls through here to KV-append + attention math.
        // Fused fast path (above) jumps directly to attention_after_qkv.
        // Slow path produces non-pre-scaled Q.
        self.attention_after_qkv(q, k, v, input_pos, in_dtype, seq, false)
    }

    fn attention_after_qkv(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        _input_pos: usize,
        in_dtype: DType,
        seq: usize,
        q_pre_scaled: bool,
    ) -> Result<Tensor> {
        self.attention_after_qkv_inner(q, k, v, _input_pos, in_dtype, seq, q_pre_scaled)
    }

    fn attention_after_qkv_inner(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        _input_pos: usize,
        in_dtype: DType,
        seq: usize,
        q_pre_scaled: bool,
    ) -> Result<Tensor> {
        // The fused CPU decode path skips the per-token ConcatKvCache mirror
        // (O(ctx) concat copy). Multi-token calls that need the F-dtype
        // history (PLD verify <=2048, session suffix prefill) resync it here
        // from the f16 store - a rare one-shot O(ctx) rebuild.
        if seq > 1 && q.device().is_cpu() {
            let need = self
                .cpu_f16_kv
                .as_ref()
                .map(|c| c.len() > self.kv_cache.current_seq_len())
                .unwrap_or(false);
            if need {
                let c = self.cpu_f16_kv.as_ref().unwrap();
                let len = c.len();
                let (kf, vf) = c.export_kv();
                let k_t =
                    Tensor::from_vec(kf, (1, self.n_kv_head, len, self.head_dim), &q.device())?
                        .to_dtype(self.dtype)?;
                let v_t =
                    Tensor::from_vec(vf, (1, self.n_kv_head, len, self.head_dim), &q.device())?
                        .to_dtype(self.dtype)?;
                self.kv_cache = ConcatKvCache::new(2);
                let _ = self.kv_cache.append(&k_t, &v_t)?;
            }
        }
        // Stash the pre-append (new-token-only) K/V for the CPU f16 mirror
        // below - `k`/`v` get shadowed by the FULL history that
        // `kv_cache.append` returns. Tensor clone = refcount bump, no copy.
        let (k_new_cpu, v_new_cpu) = if q.device().is_cpu() {
            (Some(k.clone()), Some(v.clone()))
        } else {
            (None, None)
        };
        #[cfg(feature = "cuda")]
        let (k, v) = if self.q8_kv_cache.is_some() {
            // Consolidated Q8 mode: only Q8 cache holds history. Single-token
            // decode uses the Q8 cache via attn_scores/attn_output below.
            // Multi-token paths (prefill, PLD verify) dequantize the full
            // Q8 history to F-dtype for downstream matmul.
            match self.q8_kv_cache.as_mut().unwrap().append(&k, &v) {
                Ok(()) if seq == 1 => (k, v),
                Ok(()) => match self.q8_kv_cache.as_ref().unwrap().dequantize_kv(self.dtype) {
                    Ok((k_full, v_full)) => (k_full, v_full),
                    Err(e) => {
                        tracing::warn!("Q8 dequantize_kv failed (falling back to F-dtype): {e}");
                        self.q8_kv_cache = None;
                        self.kv_cache.append(&k, &v)?
                    }
                },
                Err(e) => {
                    tracing::warn!("Q8 KV append failed (falling back to F-dtype): {e}");
                    self.q8_kv_cache = None;
                    self.kv_cache.append(&k, &v)?
                }
            }
        } else if self.q4_kv_cache.is_some() {
            // Q4 mode. Single-token decode takes the fused-kernel fast
            // path below (`attn_scores` + `attn_output`), so for seq=1 we
            // only need the append; k, v returned here stay unused.
            // Multi-token paths (prefill, PLD verify batches) still need
            // full F-dtype K, V for attention, so we dequantise the cache.
            // KV-append profiling kept disabled (development diagnostic).
            let profile = std::env::var("GH_PROF").is_ok();
            let t_append = if profile {
                let _ = k.device().synchronize();
                Some(std::time::Instant::now())
            } else {
                None
            };
            match self.q4_kv_cache.as_mut().unwrap().append(&k, &v) {
                Ok(()) if seq == 1 => (k, v),
                Ok(()) => {
                    if let Some(t0) = t_append {
                        let _ = k.device().synchronize();
                        let append_us = t0.elapsed().as_micros();
                        let t1 = std::time::Instant::now();
                        let res = self.q4_kv_cache.as_ref().unwrap().dequantize_kv(self.dtype);
                        let _ = k.device().synchronize();
                        let dequant_us = t1.elapsed().as_micros();
                        match res {
                            Ok((k_full, v_full)) => {
                                tracing::info!(
                                    "Q4-prof seq={} kv_seq={} append_us={} dequant_us={}",
                                    seq,
                                    self.q4_kv_cache.as_ref().unwrap().current_seq_len(),
                                    append_us,
                                    dequant_us,
                                );
                                (k_full, v_full)
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Q4 dequantize_kv failed (falling back to F-dtype): {e}"
                                );
                                self.q4_kv_cache = None;
                                self.kv_cache.append(&k, &v)?
                            }
                        }
                    } else {
                        match self.q4_kv_cache.as_ref().unwrap().dequantize_kv(self.dtype) {
                            Ok((k_full, v_full)) => (k_full, v_full),
                            Err(e) => {
                                tracing::warn!(
                                    "Q4 dequantize_kv failed (falling back to F-dtype): {e}"
                                );
                                self.q4_kv_cache = None;
                                self.kv_cache.append(&k, &v)?
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Q4 KV append failed (falling back to F-dtype): {e}");
                    self.q4_kv_cache = None;
                    self.kv_cache.append(&k, &v)?
                }
            }
        } else {
            self.kv_cache.append(&k, &v)?
        };
        #[cfg(not(feature = "cuda"))]
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // CPU decode fast path: mirror the new K/V into the f16 GQA
        // cache; at seq==1 run the single-pass online-softmax attention and
        // return - replacing the 5-dispatch tensor sdpa below plus the
        // intermediates it materialises each step. Multi-token calls
        // (prefill / PLD verify) only mirror and fall through unchanged.
        if let (Some(kn), Some(vn)) = (k_new_cpu, v_new_cpu) {
            if self.cpu_f16_kv.is_none() {
                self.cpu_f16_kv = Some(crate::inference::cache::cpu_f16_kv::CpuF16Kv::new(
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                    None,
                ));
            }
            let (nkv, hd) = (self.n_kv_head, self.head_dim);
            let cache = self.cpu_f16_kv.as_mut().unwrap();
            // kn/vn are [1, n_kv, seq, hd]; flatten once, then per token
            // regroup to the cache's [nkv*hd] row layout.
            let kf: Vec<f32> = kn.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let vf: Vec<f32> = vn.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let mut kt = vec![0f32; nkv * hd];
            let mut vt = vec![0f32; nkv * hd];
            for t in 0..seq {
                for h in 0..nkv {
                    let src = h * seq * hd + t * hd;
                    kt[h * hd..(h + 1) * hd].copy_from_slice(&kf[src..src + hd]);
                    vt[h * hd..(h + 1) * hd].copy_from_slice(&vf[src..src + hd]);
                }
                cache.append(&kt, &vt)?;
            }
            if seq == 1 {
                let qf: Vec<f32> = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let scale = if q_pre_scaled {
                    1.0f32
                } else {
                    (1.0 / (self.head_dim as f64).sqrt()) as f32
                };
                let mut out = vec![0f32; self.n_head * hd];
                cache.attention(&qf, scale, &mut out)?;
                let ctx_t = Tensor::from_vec(out, (1, 1, self.n_head * hd), &q.device())?;
                let ctx_t = if ctx_t.dtype() == in_dtype {
                    ctx_t
                } else {
                    ctx_t.to_dtype(in_dtype)?
                };
                return self.wo.forward(&ctx_t);
            }
        }

        // Quantized-cache decode fast path: single-token attention via the
        // cache's fused gemv kernels. Triggers when one of q8/q4 is live,
        // seq=1, and we're on CUDA.
        #[cfg(feature = "cuda")]
        if seq == 1 && q.device().is_cuda() {
            if let Some(cache) = self.q8_kv_cache.as_ref() {
                let scores = cache
                    .attn_scores(&q)
                    .map_err(|e| crate::tensor::Error::msg(format!("Q8 attn_scores: {e}")))?;
                let scores = if q_pre_scaled {
                    scores
                } else {
                    let scale = 1.0 / (self.head_dim as f64).sqrt();
                    scores.affine(scale as f32, 0.0)?
                };
                let probs = crate::tensor::ops::softmax_last_dim(&scores)?;
                let ctx = cache
                    .attn_output(&probs)
                    .map_err(|e| crate::tensor::Error::msg(format!("Q8 attn_output: {e}")))?;
                // ctx is [1, n_q_heads, 1, head_dim] in row-major. The seq
                // dim (=1) is in position 2; transposing it to position 1
                // doesn't move data. Reshape directly to skip the copy that
                // transpose+reshape's contiguous() check would force.
                let reshaped = ctx.reshape((1, seq, self.n_head * self.head_dim))?;
                let reshaped = reshaped.to_dtype(in_dtype)?;
                return self.wo.forward(&reshaped);
            }
            if let Some(cache) = self.q4_kv_cache.as_ref() {
                // Fused split-K flash-decode (GQA, hd∈{64,128}): replaces the
                // attn_scores(->HBM scores)+attn_softmax_output 2-kernel chain
                // below with one pass (no HBM scores round-trip, adaptive
                // nsplit). VALIDATED bit-equivalent to the 2-kernel chain
                // (identical greedy output incl. recall) and flips qwen3-coder
                // long from -24% LOSS to +14.5% WIN (88->132 tok/s @ 2664 ctx,
                // vs ollama 115). AUTO-ENABLED for hd∈{64,128} (the qwen3moe Q4
                // path is GQA). On any kernel error falls through to the
                // 2-kernel chain. softmax_scale mirrors the chain's affine:
                // 1/sqrt(hd) iff Q was not pre-scaled. (The diagnostic env
                // override was removed per the no-env-vars rule.)
                let q4_splitk = self.head_dim == 64 || self.head_dim == 128;
                if q4_splitk {
                    let softmax_scale = if q_pre_scaled {
                        1.0f32
                    } else {
                        (1.0 / (self.head_dim as f64).sqrt()) as f32
                    };
                    if let Ok(ctx) = cache.attn_flash_splitk_decode(&q, softmax_scale) {
                        let reshaped = ctx.reshape((1, seq, self.n_head * self.head_dim))?;
                        let reshaped = reshaped.to_dtype(in_dtype)?;
                        return self.wo.forward(&reshaped);
                    }
                }
                let prof_attn = false;
                let dev_ref = q.device();
                let t_score0 = if prof_attn {
                    let _ = dev_ref.synchronize();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let scores = cache
                    .attn_scores(&q)
                    .map_err(|e| crate::tensor::Error::msg(format!("Q4 attn_scores: {e}")))?;
                if let Some(t) = t_score0 {
                    let _ = dev_ref.synchronize();
                    Q4_PROF_SCORE_US.fetch_add(
                        t.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                let t_aff0 = if prof_attn {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let scores = if q_pre_scaled {
                    scores
                } else {
                    let scale = 1.0 / (self.head_dim as f64).sqrt();
                    scores.affine(scale as f32, 0.0)?
                };
                if let Some(t) = t_aff0 {
                    let _ = dev_ref.synchronize();
                    Q4_PROF_AFFINE_US.fetch_add(
                        t.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                // Fused softmax+output kernel for Q4 KV. Each block recomputes
                // softmax stats for its kv-group's q heads - softmax cost
                // scales with seq_kv per-q-head. At ctx <= ~4k FSO is a net
                // win; at ctx >= 12k the redundant softmax recomputation
                // costs ~2x the unfused softmax+output sequence. The
                // crossover is fuzzy in the 4k-12k range, where it's noise.
                // Threshold at 12k to be conservative - only force off when
                // we're clearly in the regime where FSO loses.
                // FSO (attn_softmax_output) auto-enables for ctx <= 12k.
                // At higher ctx the redundant softmax recomputation inside
                // the fused kernel costs ~2x the unfused softmax+output
                // sequence.
                let fso_on = cache.current_seq_len() <= 12288;
                let t_fso0 = if prof_attn {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let ctx = if fso_on {
                    cache.attn_softmax_output(&scores).map_err(|e| {
                        crate::tensor::Error::msg(format!("Q4 attn_softmax_output: {e}"))
                    })?
                } else {
                    let t_sm0 = if prof_attn {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let probs = crate::tensor::ops::softmax_last_dim(&scores)?;
                    if let Some(t) = t_sm0 {
                        let _ = dev_ref.synchronize();
                        Q4_PROF_SOFTMAX_US.fetch_add(
                            t.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    let t_out0 = if prof_attn {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let r = cache
                        .attn_output(&probs)
                        .map_err(|e| crate::tensor::Error::msg(format!("Q4 attn_output: {e}")))?;
                    if let Some(t) = t_out0 {
                        let _ = dev_ref.synchronize();
                        Q4_PROF_OUTPUT_US.fetch_add(
                            t.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    r
                };
                if let Some(t) = t_fso0 {
                    let _ = dev_ref.synchronize();
                    Q4_PROF_OUTPUT_US.fetch_add(
                        t.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                let t_wo0 = if prof_attn {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let reshaped = ctx.reshape((1, seq, self.n_head * self.head_dim))?;
                // Skip the to_dtype cast: the quantized matmul path
                // accepts F32 directly; an F32->F16 cast just for the
                // matmul forces a redundant cast back to F32 inside the
                // q8_1 quantize step. Saves 1 launch per layer per token.
                let reshaped = if reshaped.dtype() == DType::F32 {
                    reshaped
                } else {
                    reshaped.to_dtype(in_dtype)?
                };
                let r = self.wo.forward(&reshaped);
                if let Some(t) = t_wo0 {
                    let _ = dev_ref.synchronize();
                    Q4_PROF_WO_US.fetch_add(
                        t.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    let n = Q4_PROF_LAYER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if n.is_multiple_of(48 * 8) {
                        // Aggregate per ~8 tokens (48 layers/token).
                        let s = Q4_PROF_SCORE_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                        let a = Q4_PROF_AFFINE_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                        let sm = Q4_PROF_SOFTMAX_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                        let o = Q4_PROF_OUTPUT_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                        let w = Q4_PROF_WO_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                        Q4_PROF_LAYER.store(0, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!(
                            "🟧 ATTN_PROF (sum/8tokens, µs): score={} affine={} softmax={} output={} wo={}",
                            s, a, sm, o, w
                        );
                    }
                }
                return r;
            }
        }

        // Decode (seq=1) with GQA: reshape Q to group heads instead of expanding K/V
        // via repeat_kv. This avoids the 2x contiguous copies of K and V.
        let ctx = if seq == 1 && self.num_kv_groups > 1 {
            // q: [1, n_head, 1, d] -> [1, n_kv_head, n_rep, d]
            let q_grouped = q.reshape((1, self.n_kv_head, self.num_kv_groups, self.head_dim))?;
            // att: [1, n_kv_head, n_rep, kv_len]
            let att = q_grouped.matmul_t(&k)?;
            let att = if q_pre_scaled {
                att
            } else {
                let scale = 1.0 / (self.head_dim as f64).sqrt();
                att.affine(scale as f32, 0.0)?
            };
            // mask skipped for seq=1 (causal is trivially satisfied)
            let att = crate::tensor::ops::softmax_last_dim(&att)?;
            // att @ v: [1, n_kv_head, n_rep, d] -> reshape to [1, n_head, 1, d]
            let out = att.matmul(&v)?;
            out.reshape((1, self.n_head, 1, self.head_dim))?
        } else {
            // Multi-token (prefill / PLD verify): causal attention over the key and value
            // heads as they are stored, one band of keys at a time.
            //
            // Query head `h * n_rep + r` reads kv head `h`, so the heads sharing a kv head
            // are already neighbours: viewing Q as `[1, kv_heads, n_rep * seq, d]` puts
            // each group's queries in one matrix and one matmul against the stored K
            // answers all of them. Expanding K and V to one copy per query head instead -
            // what `repeat_kv` does, and it has to materialise them - was the largest
            // allocation in the layer.
            //
            // The band is the reason this is written out rather than left as
            // softmax(QK^T)V: that form holds a score for every query against every key at
            // once, so its memory grows with the context and a long one cannot be served
            // at all. Accumulating band by band, carrying each row's running maximum and
            // sum and rescaling what is already accumulated when the maximum moves, gives
            // the same result out of memory that grows with the band instead.
            //
            // The scale goes on Q rather than on the scores for the same reason: Q is
            // seq x d and the scores are seq x kv_len.
            let n_rep = self.num_kv_groups;
            let hd = self.head_dim;
            let q = if q_pre_scaled {
                q
            } else {
                (q * (1.0 / (hd as f64).sqrt()))?
            };
            let kv_len = k.dim(2)?;
            let qg = q
                .contiguous()?
                .reshape((1, self.n_kv_head, n_rep * seq, hd))?;
            let ctx = crate::tensor::ops::banded_causal_attention(
                &qg,
                &k.contiguous()?,
                &v.contiguous()?,
                seq,
                kv_len.saturating_sub(seq),
            )?
            .reshape((1, self.n_head, seq, hd))?;
            ctx.to_dtype(q.dtype())?
        };
        let reshaped = ctx
            .transpose(1, 2)?
            .reshape((1, seq, self.n_head * self.head_dim))?;
        self.wo.forward(&reshaped.to_dtype(in_dtype)?)
    }
}

// ------------------------------------------------------------
// Per-layer container. Each layer owns its attention + FFN (either dense
// MLP or FusedMoeGGUF) + norms. `device` tracks which GPU the layer lives
// on so the forward loop can transfer hidden state across boundaries.
// ------------------------------------------------------------

enum MoeOrMlp {
    Moe(FusedMoeGGUF),
    Mlp {
        w1: crate::tensor::quantized::QMatMul,
        w2: crate::tensor::quantized::QMatMul,
        w3: crate::tensor::quantized::QMatMul,
    },
}

impl MoeOrMlp {
    /// Forward with the post-MLP residual fused in. For the MoE branch,
    /// this folds the residual add into the down kernel's atomicAdd
    /// reduction (zero extra cost). For the Mlp branch, it adds the
    /// residual after the FFN. Returns mlp_out + residual.
    fn forward_with_residual(
        &self,
        x: &Tensor,
        residual: &Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Moe(m) => m.forward_with_residual(x, residual, is_prefill),
            Self::Mlp { w1, w2, w3 } => {
                let a = w1.forward(x)?;
                let b = w3.forward(x)?;
                let y = w2.forward(&(crate::tensor::ops::silu(&a)? * b)?)?;
                y + residual
            }
        }
    }
}

struct Layer {
    attn: Attn,
    attn_norm: RmsNorm,
    mlp: MoeOrMlp,
    ffn_norm: RmsNorm,
    device: Device,
}

// ------------------------------------------------------------
// Top-level multi-device model.
// ------------------------------------------------------------

pub struct MultiDeviceQwen3MoE {
    tok_embeddings: Embedding,
    embed_device: Device,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: crate::tensor::quantized::QMatMul,
    output_device: Device,
}

fn load_tensor<R: Read + Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
) -> Result<QTensor> {
    content.tensor(reader, name, device)
}

fn load_qmatmul<R: Read + Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
) -> Result<crate::tensor::quantized::QMatMul> {
    let qt = load_tensor(content, reader, name, device)?;
    crate::tensor::quantized::QMatMul::from_qtensor(qt)
}

/// Byte-concat Q, K, V weight tensors along the output dim into a single
/// fused QTensor. The three weights must share the same quantization
/// type and the same hidden (input) dim - output dims may differ (Q is
/// typically n_q_heads x head_dim while K and V are n_kv_heads x
/// head_dim). Returns None if a precondition is violated.
///
/// Resulting layout: rows 0..N_q come from Q, then K, then V. Forward
/// runs ONE matmul on the fused tensor and narrows the [seq, N_q + 2*N_kv]
/// output into the three slices.
fn try_fuse_qkv(wq: &QTensor, wk: &QTensor, wv: &QTensor, device: &Device) -> Option<QTensor> {
    use crate::tensor::quantized::QStorage;
    if wq.dtype() != wk.dtype() || wk.dtype() != wv.dtype() {
        tracing::debug!(
            "try_fuse_qkv: dtype mismatch q={:?} k={:?} v={:?}",
            wq.dtype(),
            wk.dtype(),
            wv.dtype()
        );
        return None;
    }
    let q_shape = wq.shape().dims().to_vec();
    let k_shape = wk.shape().dims().to_vec();
    let v_shape = wv.shape().dims().to_vec();
    if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
        tracing::debug!(
            "try_fuse_qkv: rank!=2 q={:?} k={:?} v={:?}",
            q_shape,
            k_shape,
            v_shape
        );
        return None;
    }
    if q_shape[1] != k_shape[1] || k_shape[1] != v_shape[1] {
        tracing::debug!(
            "try_fuse_qkv: K dim mismatch q={:?} k={:?} v={:?}",
            q_shape,
            k_shape,
            v_shape
        );
        return None;
    }
    let dtype = wq.dtype();
    let hidden = q_shape[1];
    let q_out = q_shape[0];
    let k_out = k_shape[0];
    let v_out = v_shape[0];
    let q_bytes = wq.data().ok()?.into_owned();
    let k_bytes = wk.data().ok()?.into_owned();
    let v_bytes = wv.data().ok()?.into_owned();
    let total = q_bytes.len() + k_bytes.len() + v_bytes.len();
    let n_u32 = total.div_ceil(4);
    let mut buf = vec![0u32; n_u32];
    let buf_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_u32 * 4) };
    buf_bytes[..q_bytes.len()].copy_from_slice(&q_bytes);
    buf_bytes[q_bytes.len()..q_bytes.len() + k_bytes.len()].copy_from_slice(&k_bytes);
    buf_bytes[q_bytes.len() + k_bytes.len()..total].copy_from_slice(&v_bytes);
    let combined: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, total) };
    let storage = QStorage::from_data(std::borrow::Cow::Borrowed(combined), device, dtype).ok()?;
    drop(buf);
    let shape = crate::tensor::Shape::from((q_out + k_out + v_out, hidden));
    QTensor::new(storage, shape).ok()
}

/// Dequantize a QTensor on host and re-quantize at the target dtype on
/// the same device. Used to convert V's typically-higher-precision
/// quantization (Q6K) to match Q/K (Q4K) so all three can be byte-
/// concatted into a single fused QKV tensor. The quality impact is
/// the difference between V at Q6K and V at Q4K - usually <0.5 % on
/// downstream perplexity for K-quant families.
fn requantize_qtensor(
    qt: &QTensor,
    target_dtype: crate::tensor::quantized::GgmlDType,
    device: &Device,
) -> Option<QTensor> {
    if qt.dtype() == target_dtype {
        return None;
    }
    let dequant = qt.dequantize(&Device::Cpu).ok()?;
    QTensor::quantize_onto(&dequant, target_dtype, device).ok()
}

/// Partial Q+K fusion (V left separate). Used when V's quantization
/// differs from Q/K's - common in K-quants (Q4_K_M, Q5_K_M) where V is
/// stored at higher precision (Q6K) for quality. Saves 1 of 3 attention
/// matmul launches per layer.
fn try_fuse_qkv_qk_only(wq: &QTensor, wk: &QTensor, device: &Device) -> Option<QTensor> {
    use crate::tensor::quantized::QStorage;
    if wq.dtype() != wk.dtype() {
        return None;
    }
    let q_shape = wq.shape().dims().to_vec();
    let k_shape = wk.shape().dims().to_vec();
    if q_shape.len() != 2 || k_shape.len() != 2 || q_shape[1] != k_shape[1] {
        return None;
    }
    let dtype = wq.dtype();
    let hidden = q_shape[1];
    let q_out = q_shape[0];
    let k_out = k_shape[0];
    let q_bytes = wq.data().ok()?.into_owned();
    let k_bytes = wk.data().ok()?.into_owned();
    let total = q_bytes.len() + k_bytes.len();
    let n_u32 = total.div_ceil(4);
    let mut buf = vec![0u32; n_u32];
    let buf_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_u32 * 4) };
    buf_bytes[..q_bytes.len()].copy_from_slice(&q_bytes);
    buf_bytes[q_bytes.len()..total].copy_from_slice(&k_bytes);
    let combined: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, total) };
    let storage = QStorage::from_data(std::borrow::Cow::Borrowed(combined), device, dtype).ok()?;
    drop(buf);
    let shape = crate::tensor::Shape::from((q_out + k_out, hidden));
    QTensor::new(storage, shape).ok()
}

fn load_rmsnorm<R: Read + Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    eps: f64,
    device: &Device,
) -> Result<RmsNorm> {
    let qt = load_tensor(content, reader, name, device)?;
    RmsNorm::from_qtensor(qt, eps)
}

impl MultiDeviceQwen3MoE {
    /// Where each layer actually lives, in layer order. `assign_layers` decided
    /// this once at load and each layer owns the device it was placed on - the
    /// same field the forward loop reads to detect a device crossing - so this
    /// reports the placement rather than re-deriving it.
    pub fn layer_device_locations(&self) -> Vec<crate::tensor::DeviceLocation> {
        self.layers.iter().map(|l| l.device.location()).collect()
    }

    /// Load qwen3-moe splitting layers across `devices`. Layer i goes to
    /// `devices[i * devices.len() / n_layers]` (balanced distribution).
    /// The embedding lives on the first device, output_proj on the last.
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
        dtype: DType,
    ) -> Result<Self> {
        Self::from_gguf_with_kv_quant(
            content,
            reader,
            devices,
            dtype,
            crate::inference::engine::llm_engine::KvQuant::Off,
            None,
        )
    }

    /// Compute layer-to-device assignment proportional to per-device weights.
    /// `weights[i]` is the relative throughput score of `devices[i]` (e.g.
    /// SM count x clock). Layers are assigned greedily in proportion: the
    /// device with the largest "remaining capacity" (weight - layers_so_far/total_weight)
    /// gets the next layer. Returns a Vec of length `n_layers` mapping
    /// layer_idx -> device_idx. With uniform weights this collapses to the
    /// previous round-robin behaviour.
    pub fn assign_layers(n_layers: usize, weights: &[f32]) -> Vec<usize> {
        if weights.is_empty() {
            return vec![0; n_layers];
        }
        let n_dev = weights.len();
        let total: f32 = weights.iter().sum::<f32>().max(1e-6);
        let target: Vec<f32> = weights
            .iter()
            .map(|w| w * n_layers as f32 / total)
            .collect();
        let mut assigned = vec![0usize; n_dev];
        let mut out = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            // pick the device whose target - assigned is largest (most behind)
            let pick = (0..n_dev)
                .max_by(|&a, &b| {
                    let da = target[a] - assigned[a] as f32;
                    let db = target[b] - assigned[b] as f32;
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            assigned[pick] += 1;
            out.push(pick);
        }
        out
    }

    pub fn from_gguf_with_kv_quant<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
        dtype: DType,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        Self::from_gguf_full(
            content,
            reader,
            devices,
            None,
            dtype,
            kv_quant,
            max_kv_seq_len,
        )
    }

    /// Full constructor with optional per-device throughput weights for
    /// proportional layer placement. When `weights` is None, falls back to
    /// uniform layers-per-device split.
    pub fn from_gguf_full<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
        weights: Option<&[f32]>,
        dtype: DType,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        if devices.is_empty() {
            crate::tensor::bail!("need at least one device");
        }
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "qwen3moe".to_string());
        let md = |k: &str| {
            content
                .metadata
                .get(k)
                .ok_or_else(|| crate::tensor::Error::msg(format!("missing {k}")))
        };

        let head_count = md(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let head_count_kv = md(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let head_dim = md(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().map(|x| x as usize))
            .unwrap_or_else(|_| {
                let emb = md(&format!("{arch}.embedding_length"))
                    .and_then(crate::tensor::quantized::gguf_file::Value::to_u32)
                    .unwrap_or(1) as usize;
                emb / head_count
            });
        // Read, not used: the `?` is the point. A checkpoint missing this key, or carrying it
        // as the wrong type, is malformed, and failing here names the key instead of failing
        // later on a shape that does not match. The width itself comes from the weights.
        let _embedding_length = md(&format!("{arch}.embedding_length"))?.to_u32()? as usize;
        let context_length = md(&format!("{arch}.context_length"))?.to_u32()? as usize;
        let block_count = md(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let rms_eps = md(&format!("{arch}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;
        let rope_freq = md(&format!("{arch}.rope.freq_base"))
            .and_then(crate::tensor::quantized::gguf_file::Value::to_f32)
            .unwrap_or(10_000.0);
        let num_experts = md(&format!("{arch}.expert_count"))?.to_u32()? as usize;
        let moe_inter = md(&format!("{arch}.expert_feed_forward_length"))?.to_u32()? as usize;
        let num_experts_per_tok = md(&format!("{arch}.expert_used_count"))?.to_u32()? as usize;
        let _ = moe_inter; // carried via weights, not consumed here

        let embed_device = devices[0].clone();
        // Output projection (lm_head) is bandwidth-bound on a 151K-vocab
        // matmul. Place it on the FASTEST device (devices[0] - typically
        // the higher-bandwidth GPU when ranking is set in priority order)
        // and pay the one xs peer-copy per token instead of running the
        // matmul on a slower GPU.
        let output_device = devices[0].clone();

        // Embedding on first device
        let emb_qt = load_tensor(&content, reader, "token_embd.weight", &embed_device)?;
        let emb_tensor = emb_qt.dequantize(&embed_device)?;
        let tok_embeddings = Embedding::new(emb_tensor);

        // Output norm + projection on last device
        let norm = load_rmsnorm(
            &content,
            reader,
            "output_norm.weight",
            rms_eps,
            &output_device,
        )?;
        let output = match load_qmatmul(&content, reader, "output.weight", &output_device) {
            Ok(v) => v,
            Err(_) => load_qmatmul(&content, reader, "token_embd.weight", &output_device)?,
        };

        // Per-device RotaryEmbedding: RoPE cos/sin tables on the layer's device.
        // One rotary_emb per device in `devices` (deduped).
        let mut rotary_per_dev: std::collections::HashMap<String, Arc<RotaryEmbedding>> =
            std::collections::HashMap::new();
        let dev_key = |d: &Device| format!("{:?}", d.location());
        for d in devices.iter() {
            // RotaryEmbedding::new returns Result so use the Vacant-slot
            // entry form (or_insert_with can't propagate ?).
            if let std::collections::hash_map::Entry::Vacant(slot) =
                rotary_per_dev.entry(dev_key(d))
            {
                slot.insert(Arc::new(RotaryEmbedding::new(
                    dtype,
                    head_dim,
                    context_length,
                    rope_freq as f64,
                    d,
                )?));
            }
        }

        // Decide which device each layer goes on. With per-device weights
        // (proportional to throughput), use weighted assignment so the
        // faster GPU gets more layers - total decode time = sum of
        // per-layer times across the pipeline, so concentrating on the
        // faster device reduces it. Without weights, falls back to even
        // round-robin (the previous behaviour).
        let placement: Vec<usize> = if let Some(w) = weights {
            if w.len() == devices.len() {
                Self::assign_layers(block_count, w)
            } else {
                tracing::warn!(
                    "qwen3_moe_multi: weights len {} != devices len {}, falling back to uniform",
                    w.len(),
                    devices.len()
                );
                (0..block_count)
                    .map(|i| (i * devices.len()) / block_count)
                    .collect()
            }
        } else {
            let layers_per_dev = block_count.div_ceil(devices.len());
            (0..block_count)
                .map(|i| (i / layers_per_dev).min(devices.len() - 1))
                .collect()
        };
        // Log the distribution for visibility.
        let mut counts = vec![0usize; devices.len()];
        for &p in &placement {
            counts[p] += 1;
        }
        tracing::info!("qwen3_moe_multi: {block_count} layers placed as {counts:?}");

        // The placement IS the loop: one entry per layer, holding the device that layer's
        // weights load onto, so the index and the card are read from the same place.
        let mut layers = Vec::with_capacity(placement.len());
        for (layer_idx, &dev_idx) in placement.iter().enumerate() {
            // The layer OWNS the device it lives on - the forward loop reads it to decide
            // where the hidden state has to travel - and everything that loads a weight
            // below borrows that one copy instead of making its own.
            let device = devices[dev_idx].clone();
            let dev = &device;
            let prefix = format!("blk.{layer_idx}");
            let rotary = rotary_per_dev.get(&dev_key(dev)).unwrap().clone();

            // Attention weights - load raw QTensors first so we can try
            // byte-concat fusion of Q+K+V along the output dim. When
            // fusion succeeds we drop the originals and the forward
            // path runs ONE matmul instead of three. Saves 2 launches
            // per attention layer per token (96 launches/token on a
            // 48-layer MoE).
            let wq_qt = load_tensor(&content, reader, &format!("{prefix}.attn_q.weight"), dev)?;
            let wk_qt = load_tensor(&content, reader, &format!("{prefix}.attn_k.weight"), dev)?;
            let wv_qt = load_tensor(&content, reader, &format!("{prefix}.attn_v.weight"), dev)?;
            // QKV fusion is always-on: try full Q+K+V pack first, then V-requant
            // fallback, then partial Q+K only. The V-requant tier widens by one
            // quant level (e.g. Q6K->Q4K for K-quant V) to make full fusion
            // possible - saves one matmul launch per layer per token.
            let q_dim = wq_qt.shape().dims()[0];
            let kv_dim = wk_qt.shape().dims()[0];
            let (wqkv, wqk, wq, wk, wv) = {
                // Try full Q+K+V fuse first.
                if let Some(fused) = try_fuse_qkv(&wq_qt, &wk_qt, &wv_qt, dev) {
                    if layer_idx == 0 {
                        tracing::info!(
                            "MoE QKV fusion: full (Q={}+K={}+V={} packed, 1 matmul)",
                            q_dim,
                            kv_dim,
                            kv_dim
                        );
                    }
                    (
                        Some(crate::tensor::quantized::QMatMul::from_qtensor(fused)?),
                        None,
                        None,
                        None,
                        None,
                    )
                } else {
                    // Full QKV pack failed (V dtype differs). Try requant of
                    // V to Q/K's dtype, then retry full fuse. If both fail,
                    // fall back to partial Q+K fuse (2 matmuls). If even Q+K
                    // can't pack, ship 3 separate matmuls.
                    let target_dtype = wq_qt.dtype();
                    if let Some(v_requant) = requantize_qtensor(&wv_qt, target_dtype, dev)
                        .and_then(|v| try_fuse_qkv(&wq_qt, &wk_qt, &v, dev))
                    {
                        if layer_idx == 0 {
                            tracing::info!(
                                "MoE QKV fusion: full (V requantized {:?}->{:?}, 1 matmul)",
                                wv_qt.dtype(),
                                target_dtype
                            );
                        }
                        (
                            Some(crate::tensor::quantized::QMatMul::from_qtensor(v_requant)?),
                            None,
                            None,
                            None,
                            None,
                        )
                    } else if let Some(fused_qk) = try_fuse_qkv_qk_only(&wq_qt, &wk_qt, dev) {
                        if layer_idx == 0 {
                            tracing::info!(
                                "MoE QKV fusion: partial Q+K (V dtype differs, 2 matmuls)"
                            );
                        }
                        (
                            None,
                            Some(crate::tensor::quantized::QMatMul::from_qtensor(fused_qk)?),
                            None,
                            None,
                            Some(crate::tensor::quantized::QMatMul::from_qtensor(wv_qt)?),
                        )
                    } else {
                        if layer_idx == 0 {
                            tracing::info!("MoE QKV fusion: fallback (3 separate matmuls)");
                        }
                        (
                            None,
                            None,
                            Some(crate::tensor::quantized::QMatMul::from_qtensor(wq_qt)?),
                            Some(crate::tensor::quantized::QMatMul::from_qtensor(wk_qt)?),
                            Some(crate::tensor::quantized::QMatMul::from_qtensor(wv_qt)?),
                        )
                    }
                }
            };
            let (wqkv_q_dim, wqkv_kv_dim) = (q_dim, kv_dim);
            let wo = load_qmatmul(
                &content,
                reader,
                &format!("{prefix}.attn_output.weight"),
                dev,
            )?;
            let bq = load_tensor(&content, reader, &format!("{prefix}.attn_q.bias"), dev)
                .ok()
                .map(|qt| qt.dequantize(dev).and_then(|t| t.to_dtype(DType::F32)))
                .transpose()?;
            let bk = load_tensor(&content, reader, &format!("{prefix}.attn_k.bias"), dev)
                .ok()
                .map(|qt| qt.dequantize(dev).and_then(|t| t.to_dtype(DType::F32)))
                .transpose()?;
            let bv = load_tensor(&content, reader, &format!("{prefix}.attn_v.bias"), dev)
                .ok()
                .map(|qt| qt.dequantize(dev).and_then(|t| t.to_dtype(DType::F32)))
                .transpose()?;
            let q_norm = load_rmsnorm(
                &content,
                reader,
                &format!("{prefix}.attn_q_norm.weight"),
                rms_eps,
                dev,
            )?;
            let k_norm = load_rmsnorm(
                &content,
                reader,
                &format!("{prefix}.attn_k_norm.weight"),
                rms_eps,
                dev,
            )?;

            // Size the quantized KV cache. Prefer the caller-supplied cap
            // (engine's `context_length` from config.toml) so we don't
            // eagerly allocate for the model's native 128K/256K context
            // when the user only needs a fraction of it - that alloc alone
            // can OOM the GPU on qwen3-coder. Fall back to GGUF's
            // context_length if no cap is given.
            let max_kv_seq_len = max_kv_seq_len
                .map(|n| n.min(context_length))
                .unwrap_or(context_length);
            #[cfg(feature = "cuda")]
            let q8_kv_cache = if kv_quant == crate::inference::engine::llm_engine::KvQuant::Q8
                && dev.is_cuda()
            {
                match crate::inference::cache::q8_kv::Q8KvCache::new(
                    max_kv_seq_len,
                    head_count_kv,
                    head_dim,
                    dev,
                ) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        tracing::warn!(
                            "MoE layer Q8 KV alloc failed on {:?}: {e}; falling back to F-dtype cache",
                            dev
                        );
                        None
                    }
                }
            } else {
                None
            };
            #[cfg(feature = "cuda")]
            let q4_kv_cache = if kv_quant == crate::inference::engine::llm_engine::KvQuant::Q4
                && dev.is_cuda()
            {
                match crate::inference::cache::q4_kv::Q4KvCache::new(
                    max_kv_seq_len,
                    head_count_kv,
                    head_dim,
                    dev,
                ) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        tracing::warn!(
                            "MoE layer Q4 KV alloc failed on {:?}: {e}; falling back to F-dtype cache",
                            dev
                        );
                        None
                    }
                }
            } else {
                None
            };
            // Extract the underlying QTensor from wqkv for the
            let attn = Attn {
                wqkv,
                wqk,
                wqkv_q_dim,
                wqkv_kv_dim,
                wq,
                wk,
                wv,
                wo,
                bq,
                bk,
                bv,
                q_norm,
                k_norm,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                num_kv_groups: head_count / head_count_kv,
                rotary_emb: rotary,
                dtype,
                kv_cache: ConcatKvCache::new(2),
                #[cfg(feature = "cuda")]
                q8_kv_cache,
                #[cfg(feature = "cuda")]
                q4_kv_cache,
                cpu_f16_kv: None,
                qk_norm_f32: None,
            };

            let attn_norm = load_rmsnorm(
                &content,
                reader,
                &format!("{prefix}.attn_norm.weight"),
                rms_eps,
                dev,
            )?;
            let ffn_norm = load_rmsnorm(
                &content,
                reader,
                &format!("{prefix}.ffn_norm.weight"),
                rms_eps,
                dev,
            )?;

            // MoE or dense MLP
            let mlp = if num_experts > 0 {
                let gate_ws = load_tensor(
                    &content,
                    reader,
                    &format!("{prefix}.ffn_gate_inp.weight"),
                    dev,
                )?
                .dequantize(dev)?
                .to_dtype(DType::F32)?;
                let gate = Linear::new(gate_ws, None)?;
                let gate_experts = Arc::new(load_tensor(
                    &content,
                    reader,
                    &format!("{prefix}.ffn_gate_exps.weight"),
                    dev,
                )?);
                let up_experts = Arc::new(load_tensor(
                    &content,
                    reader,
                    &format!("{prefix}.ffn_up_exps.weight"),
                    dev,
                )?);
                let down_experts = Arc::new(load_tensor(
                    &content,
                    reader,
                    &format!("{prefix}.ffn_down_exps.weight"),
                    dev,
                )?);
                // Load-time gate+up byte-concat fusion was measured to be
                // neutral or worse with the current MMVQ kernel (it does
                // not share input loads across the doubled N dim). Path
                // removed; kept as a placeholder None until a kernel-level
                // gate+up fusion lands.
                let gate_up_experts: Option<Arc<crate::tensor::quantized::QTensor>> = None;
                let (gate_experts_opt, up_experts_opt) = if gate_up_experts.is_some() {
                    (None, None)
                } else {
                    (Some(gate_experts), Some(up_experts))
                };
                if layer_idx == 0 {
                    tracing::info!(
                        "MoE gate+up fusion: {}",
                        if gate_up_experts.is_some() {
                            "ENABLED (originals dropped)"
                        } else {
                            "off (separate matmuls)"
                        }
                    );
                }
                MoeOrMlp::Moe(FusedMoeGGUF {
                    gate,
                    gate_experts: gate_experts_opt,
                    up_experts: up_experts_opt,
                    down_experts,
                    gate_up_experts,
                    act: Activation::Silu,
                    swiglu_oai: None,
                    gate_inp_bias: None,
                    gate_exps_bias: None,
                    up_exps_bias: None,
                    down_exps_bias: None,
                    norm_topk_prob: true,
                    num_experts_per_tok,
                    dtype,
                    cpu_experts: std::sync::Arc::new(std::sync::OnceLock::new()),
                })
            } else {
                let w1 = load_qmatmul(&content, reader, &format!("{prefix}.ffn_gate.weight"), dev)?;
                let w2 = load_qmatmul(&content, reader, &format!("{prefix}.ffn_down.weight"), dev)?;
                let w3 = load_qmatmul(&content, reader, &format!("{prefix}.ffn_up.weight"), dev)?;
                MoeOrMlp::Mlp { w1, w2, w3 }
            };

            layers.push(Layer {
                attn,
                attn_norm,
                mlp,
                ffn_norm,
                device,
            });
        }

        Ok(Self {
            tok_embeddings,
            embed_device,
            layers,
            norm,
            output,
            output_device,
        })
    }

    /// Embed the ids and run every layer, carrying the hidden state across a device boundary
    /// whenever the next layer lives on another card. Returns it on the output device.
    ///
    /// Where the stack ENDS is the caller's: one entry point keeps the last row, the other
    /// keeps them all so a speculative step can verify a whole draft in one pass. Nothing else
    /// separates the two, so the stack itself is written once.
    fn run_layers(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        // x is [b, seq] u32 on any device - move it to the embedding's.
        let x_emb = if x.device().location() == self.embed_device.location() {
            x.clone()
        } else {
            x.to_device(&self.embed_device)?
        };
        let mut xs = self.tok_embeddings.forward(&x_emb)?;
        let (_b, l) = x_emb.dims2()?;

        let mut current_dev = self.embed_device.clone();

        // Per-stage forward profiling. Two things ask for it and both need the same
        // synchronised boundaries, so it is measured once: the environment variable prints
        // a line when the forward ends, and the endpoint accumulates across requests.
        // Feeding only the first left `/api/stage_perf` answering zero for this model
        // rather than saying it could not see it.
        use crate::inference::place::layer_perf::stages;
        let printing = std::env::var("GH_PROF").is_ok();
        let profile = printing || stages::enabled();
        let mut t_attn_norm_us: u128 = 0;
        let mut t_attn_us: u128 = 0;
        let mut t_attn_res_us: u128 = 0;
        let mut t_ffn_norm_us: u128 = 0;
        let mut t_mlp_us: u128 = 0;
        let t_ffn_res_us: u128 = 0;

        for layer in self.layers.iter_mut() {
            // Transfer hidden state to layer's device if we crossed a boundary
            if layer.device.location() != current_dev.location() {
                xs = xs.to_device(&layer.device)?;
                current_dev = layer.device.clone();
            }
            // A prompt is read many tokens at a time, an answer one at a time, and the
            // two take different paths through the attention below. Where a whole-context
            // causal mask used to stand for this, the attention derives what a row may
            // see from its own position: the mask said the same thing in a square the
            // size of the conversation, built on the host and copied to the card on every
            // pass.
            let is_prefill = l != 1;

            let residual = xs.clone();

            let t0 = if profile {
                let _ = layer.device.synchronize();
                Some(std::time::Instant::now())
            } else {
                None
            };
            let x_normed = layer.attn_norm.forward(&xs)?;
            if let Some(t) = t0 {
                let _ = layer.device.synchronize();
                let took = t.elapsed().as_micros();
                t_attn_norm_us += took;
                stages::add(stages::ATTN_NORM, took as u64);
            }
            let t1 = if profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let attn_out = layer.attn.forward(&x_normed, offset)?;
            if let Some(t) = t1 {
                let _ = layer.device.synchronize();
                let took = t.elapsed().as_micros();
                t_attn_us += took;
                stages::add(stages::ATTN, took as u64);
            }

            let t2 = if profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            // Fused (attn_residual + ffn_norm): one kernel writes both
            // xs (= attn_out + residual) and x_normed (= rms_norm(xs)).
            // Saves one launch per layer per token. Only fires on CUDA
            // single-token decode with F32 hidden state.
            let want_arn = l == 1
                && layer.device.is_cuda()
                && xs.dtype() == DType::F32
                && residual.dtype() == DType::F32
                && layer.ffn_norm.weight().dtype() == DType::F32;

            let (xs_new, x_normed) = if want_arn {
                let attn_out_f = if attn_out.dtype() == DType::F32 {
                    attn_out.clone()
                } else {
                    attn_out.to_dtype(DType::F32)?
                };
                let res_f = if residual.dtype() == DType::F32 {
                    residual.clone()
                } else {
                    residual.to_dtype(DType::F32)?
                };
                crate::inference::moe_cuda::add_rms_norm(
                    &attn_out_f,
                    &res_f,
                    layer.ffn_norm.weight(),
                    layer.ffn_norm.eps() as f32,
                )?
            } else {
                let xs_unfused = (attn_out + residual)?;
                let x_n = layer.ffn_norm.forward(&xs_unfused)?;
                (xs_unfused, x_n)
            };
            xs = xs_new;
            if let Some(t) = t2 {
                let _ = layer.device.synchronize();
                let took = t.elapsed().as_micros();
                t_attn_res_us += took;
                stages::add(stages::ATTN_RES, took as u64);
            }

            let residual = xs.clone();
            let t3 = if profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            if let Some(t) = t3 {
                let _ = layer.device.synchronize();
                t_ffn_norm_us += t.elapsed().as_micros();
            }

            // forward_with_residual folds the post-MLP residual add into
            // the down kernel's atomicAdd reduction (when on CUDA
            // decode) - zero extra cost vs the separate broadcast_add.
            let t4 = if profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            xs = layer
                .mlp
                .forward_with_residual(&x_normed, &residual, is_prefill)?;
            if let Some(t) = t4 {
                let _ = layer.device.synchronize();
                let took = t.elapsed().as_micros();
                t_mlp_us += took;
                stages::add(stages::EXPERTS, took as u64);
            }
            // ffn_res launch is now folded into mlp; not separately timed.
        }
        stages::count_call();
        if printing {
            tracing::info!(
                "📊 PROFILE_FWD per-token (sum across {} layers, µs): attn_norm={} attn={} attn_res={} ffn_norm={} mlp={} ffn_res={} TOTAL={}",
                self.layers.len(),
                t_attn_norm_us, t_attn_us, t_attn_res_us,
                t_ffn_norm_us, t_mlp_us, t_ffn_res_us,
                t_attn_norm_us + t_attn_us + t_attn_res_us
                    + t_ffn_norm_us + t_mlp_us + t_ffn_res_us
            );
        }

        // Transfer to output device if different
        if current_dev.location() != self.output_device.location() {
            xs = xs.to_device(&self.output_device)?;
        }
        Ok(xs)
    }

    /// Logits for EVERY input position, not just the last. PLD speculative decoding verifies
    /// its K drafted tokens with one of these.
    pub fn forward_all(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        if offset == 0 {
            for layer in self.layers.iter_mut() {
                layer.attn.kv_cache = ConcatKvCache::new(2);
                #[cfg(feature = "cuda")]
                if let Some(c) = layer.attn.q8_kv_cache.as_mut() {
                    c.reset();
                }
                #[cfg(feature = "cuda")]
                if let Some(c) = layer.attn.q4_kv_cache.as_mut() {
                    c.reset();
                }
            }
        }
        let xs = self.run_layers(x, offset)?;
        let xs = self.norm.forward(&xs)?;
        // Return [b, seq, vocab] - caller indexes into this.
        self.output.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        // Reset KV caches at start of a new generation. `offset == 0` is the
        // engine's signal that we're at token 0 (prefill of a fresh request).
        // Without this the previous request's cache leaks into this one.
        if offset == 0 {
            for layer in self.layers.iter_mut() {
                layer.attn.kv_cache = ConcatKvCache::new(2);
                if let Some(c) = layer.attn.cpu_f16_kv.as_mut() {
                    c.reset();
                }
                #[cfg(feature = "cuda")]
                if let Some(c) = layer.attn.q8_kv_cache.as_mut() {
                    c.reset();
                }
                #[cfg(feature = "cuda")]
                if let Some(c) = layer.attn.q4_kv_cache.as_mut() {
                    c.reset();
                }
            }
        }
        let (_b, l) = x.dims2()?;
        let xs = self.run_layers(x, offset)?;
        let xs = xs.i((.., l - 1..l, ..))?;
        let xs = self.norm.forward(&xs)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }

    /// How many positions the KV cache actually holds.
    ///
    /// Asked rather than remembered: a trim can fail, a cache can be reset by something
    /// that did not write the bookkeeping down, and a prefill that starts past the last
    /// row builds its attention mask for a context the cache does not have.
    pub fn kv_len(&self) -> Option<usize> {
        // Every cache a layer can be using, because only one of them is populated and
        // reading the wrong one reports an empty cache for a full model. The list is the
        // same one `trim_kv` walks, and has to stay that way.
        let layer = self.layers.first()?;
        let mut n = layer.attn.kv_cache.current_seq_len();
        if let Some(c) = layer.attn.cpu_f16_kv.as_ref() {
            n = n.max(c.len());
        }
        #[cfg(feature = "cuda")]
        if let Some(c) = layer.attn.q8_kv_cache.as_ref() {
            n = n.max(c.current_seq_len());
        }
        #[cfg(feature = "cuda")]
        if let Some(c) = layer.attn.q4_kv_cache.as_ref() {
            n = n.max(c.current_seq_len());
        }
        Some(n)
    }

    /// Trim every layer's KV cache to `new_len` valid positions. Used by
    /// session-persistent KV to rewind the cache to the longest common
    /// prefix between the previous and current prompt.
    pub fn trim_kv(&mut self, new_len: usize) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Named rather than dropped: a trim that did not take leaves the cache longer
            // than the offset the caller is about to prefill at, and the mask then covers
            // rows this prompt does not share. The caller checks the length afterwards,
            // and this says which layer to look at.
            if let Err(e) = layer.attn.kv_cache.trim_to(new_len) {
                tracing::warn!("KV trim to {new_len} failed on layer {i}: {e}");
            }
            if let Some(c) = layer.attn.cpu_f16_kv.as_mut() {
                c.trim_to(new_len);
            }
            #[cfg(feature = "cuda")]
            if let Some(c) = layer.attn.q8_kv_cache.as_mut() {
                c.trim_to(new_len);
            }
            #[cfg(feature = "cuda")]
            if let Some(c) = layer.attn.q4_kv_cache.as_mut() {
                // trim_to can fail on a GPU dequant; logged-but-ignored
                // because the cache is small and trim is best-effort.
                if let Err(e) = c.trim_to(new_len) {
                    tracing::warn!("Q4 KV trim_to failed: {e}");
                }
            }
        }
    }

    /// mmap variant of `from_gguf_full` with optional per-device throughput
    /// weights for proportional layer placement.
    pub fn from_gguf_mmap_full(
        content: gguf_file::Content,
        mmap_bytes: &[u8],
        devices: &[Device],
        weights: Option<&[f32]>,
        dtype: DType,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        let mut cursor = Cursor::new(mmap_bytes);
        Self::from_gguf_full(
            content,
            &mut cursor,
            devices,
            weights,
            dtype,
            kv_quant,
            max_kv_seq_len,
        )
    }
}

#[cfg(test)]
mod rope_tests {
    use crate::tensor::{DType, Device};

    /// A position is an integer, and half precision stops representing every integer at
    /// 2048: 2049 lands on 2048 and 2051 on 2052. Building the angle table at that width
    /// gave neighbouring tokens the same rotation, and the model, unable to tell them
    /// apart, stayed fluent and started to misspell.
    #[test]
    fn positions_past_the_half_precision_limit_still_differ() {
        let dev = Device::Cpu;
        let rope = super::RotaryEmbedding::new(DType::F16, 64, 4096, 10_000_000.0, &dev)
            .expect("rope table");
        let half = 32usize;
        let row = |p: usize| -> Vec<f32> { rope.cos_f32[p * half..(p + 1) * half].to_vec() };
        for p in [2049usize, 2051, 3001] {
            assert_ne!(
                row(p),
                row(p - 1),
                "position {p} rotates exactly like the one before it"
            );
        }
    }
}
