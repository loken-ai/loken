//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

impl GenericTransformerLayer {
    /// Update pre-allocated RoPE buffers for a given position.
    /// Call before graph replay so the graph reads correct cos/sin values.
    pub fn update_rope_buffers(&mut self, index_pos: usize) -> Result<()> {
        let cos_row = self.cos.narrow(0, index_pos, 1)?;
        let sin_row = self.sin.narrow(0, index_pos, 1)?;

        // `force_contiguous()` always allocates fresh storage, unlike
        // `contiguous()` which returns a shared-storage handle when the
        // input is already contiguous. Without this copy, subsequent
        // `slice_set` calls would see buffer and source sharing storage
        // and error out.
        //
        // The cos+sin buffers are both initialised together - match
        // both Options once so neither side ever needs `unwrap()` after
        // an `is_none()` (clippy::unnecessary_unwrap).
        match (&self.graph_rope_cos, &self.graph_rope_sin) {
            (Some(cos_buf), Some(sin_buf)) => {
                cos_buf.slice_set(&cos_row, 0, 0)?;
                sin_buf.slice_set(&sin_row, 0, 0)?;
            }
            _ => {
                self.graph_rope_cos = Some(cos_row.force_contiguous()?);
                self.graph_rope_sin = Some(sin_row.force_contiguous()?);
            }
        }
        Ok(())
    }

    /// Apply RoPE using pre-allocated buffers (for graph mode).
    fn apply_rotary_emb_from_buffer(&self, x: &Tensor) -> Result<Tensor> {
        if self.no_rope {
            return Ok(x.clone());
        } // SmolLM3 NoPE layer
        let mut cos = self.graph_rope_cos.as_ref().unwrap().clone();
        let mut sin = self.graph_rope_sin.as_ref().unwrap().clone();
        if !cos.device().same_device(&x.device()) {
            cos = cos.to_device(&x.device())?;
            sin = sin.to_device(&x.device())?;
        }

        let x = x.contiguous()?;
        let apply_rope = if self.flags.use_rope_i {
            crate::tensor::ops::rope_i
        } else {
            crate::tensor::ops::rope
        };
        if self.rope_dim > 0 && self.rope_dim < self.head_dim {
            let x_rot = x
                .narrow(crate::tensor::D::Minus1, 0, self.rope_dim)?
                .contiguous()?;
            let x_pass = x.narrow(
                crate::tensor::D::Minus1,
                self.rope_dim,
                self.head_dim - self.rope_dim,
            )?;
            let x_rot = apply_rope(&x_rot, &cos, &sin)?;
            Tensor::cat(&[&x_rot, &x_pass], crate::tensor::D::Minus1)?.contiguous()
        } else {
            apply_rope(&x, &cos, &sin)
        }
    }

    /// This layer's rotary tables, for a rotation done outside the forward pass.
    pub(crate) fn rope_table(&self) -> crate::inference::model::rope::RopeTable {
        crate::inference::model::rope::RopeTable {
            cos: self.cos.clone(),
            sin: self.sin.clone(),
            freq_factors: self.rope_freq_factors.clone(),
            factored_freqs: self.rope_factored_freqs.clone(),
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            no_rope: self.no_rope,
            interleaved: self.flags.use_rope_i,
        }
    }

    #[inline]
    pub(super) fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        if self.no_rope {
            return Ok(x.clone());
        } // SmolLM3 NoPE layer
        let (_b, _h, seq_len, _d) = x.dims4()?;
        if !self.cos.device().same_device(&x.device()) {
            tracing::warn!(
                "RoPE runtime device mismatch: x on {:?}, cos on {:?} - transferring",
                x.device(),
                self.cos.device()
            );
        }
        let (cos, sin) = self.rope_table().rows(index_pos, seq_len, &x.device())?;
        let apply_rope = if self.flags.use_rope_i {
            crate::tensor::ops::rope_i
        } else {
            crate::tensor::ops::rope
        };
        if self.rope_dim > 0 && self.rope_dim < self.head_dim {
            let x_rot = x
                .narrow(crate::tensor::D::Minus1, 0, self.rope_dim)?
                .contiguous()?;
            let x_pass = x.narrow(
                crate::tensor::D::Minus1,
                self.rope_dim,
                self.head_dim - self.rope_dim,
            )?;
            let x_rot = apply_rope(&x_rot, &cos, &sin)?;
            Tensor::cat(&[&x_rot, &x_pass], crate::tensor::D::Minus1)?.contiguous()
        } else {
            apply_rope(x, &cos, &sin)
        }
    }

    /// Phi2 fast-path: pre-quantize x to Q8_1 once, then run Q/K/V matmuls
    /// against the shared buffer. Saves 2 launches/layer (the 2 redundant
    /// quantize_q8_1 of the same x). Returns Some((q, k, v)) on success,
    /// None on any incompatibility (caller falls back to legacy
    /// QMatMul::forward x 3).
    ///
    /// Conditions: single-token decode + CUDA + all 3 weights are
    /// QMatMul::QTensor variant + dtype is MMVQ-compatible.
    #[cfg(feature = "cuda")]
    fn try_qkv_shared_q8_1(&self, x: &Tensor) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        use crate::inference::quantized_cuda::{mvq_via_pre_quantized_q8_1, quantize_q8_1_pub};
        use crate::tensor::cuda_ext::{self, CudaSlice};
        use crate::tensor::quantized::QStorage;

        let (b, seq, hidden) = x.dims3()?;
        if seq != 1 || b != 1 || !x.device().is_cuda() {
            return Ok(None);
        }
        // All 3 must be crate::tensor::quantized::QMatMul::QTensor (not
        // Tensor/TensorF16 fallbacks). Use the tensor-op path (not our
        // quantized_mistral3::QMatMul shadow).
        let attn_q = self.attn_q.as_ref();
        let attn_k = self.attn_k.as_ref();
        let attn_v = self.attn_v.as_ref();
        let (q_arc, k_arc, v_arc) = match (attn_q, attn_k, attn_v) {
            (Some(qm), Some(km), Some(vm)) => match (qm.qtensor(), km.qtensor(), vm.qtensor()) {
                (Some(qa), Some(ka), Some(va)) => (qa.clone(), ka.clone(), va.clone()),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        // Skip layers with biases - bias-add happens after matmul in the
        // existing path; doing it via this helper would need an extra
        // broadcast_add. Phi2 always has biases (has_attn_output_bias),
        // but the Q/K/V biases are routed via `has_qkv_bias` flag and
        // attn_q_bias/etc. Allow caller to broadcast_add after.
        let _ = (&q_arc, &k_arc, &v_arc); // explicit use

        let dev = x.device().as_cuda_device()?;
        let x_f32 = if x.dtype() == crate::tensor::DType::F32 {
            x.clone()
        } else {
            x.to_dtype(crate::tensor::DType::F32)?
        };
        let x_c = x_f32.contiguous()?;

        // Padded Q8_1 buffer: padded/32 * 36 bytes
        use crate::tensor::quantized::MATRIX_ROW_PADDING;
        let kx_padded = hidden.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        let y_size_bytes = (kx_padded / 32) * 36;
        let mut y_q8_1: CudaSlice<u8> = dev
            .alloc_zeros::<u8>(y_size_bytes)
            .map_err(|e| crate::tensor::Error::msg(format!("alloc y_q8_1: {e}")))?;

        // Pre-quantize x -> y_q8_1 (x is on CUDA - checked above - so the
        // raw-slice borrow can't hit a non-CUDA storage).
        {
            let xs = cuda_ext::f32_slice_of(&x_c)?;
            let x_view = xs.view()?;
            quantize_q8_1_pub(&x_view, &mut y_q8_1, hidden, 1, &dev)
                .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        }

        // For each of q/k/v: get QCudaStorage, run mvq_via_pre_quantized_q8_1
        let q_out_rows = q_arc.shape().dims()[0];
        let k_out_rows = k_arc.shape().dims()[0];
        let v_out_rows = v_arc.shape().dims()[0];

        let q_qstor = match q_arc.storage() {
            QStorage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let q_cuda_st = mvq_via_pre_quantized_q8_1(q_qstor, &y_q8_1, hidden, q_out_rows, 1)
            .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        let _ = q_qstor; // release borrow before next storage()
        let k_qstor = match k_arc.storage() {
            QStorage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let k_cuda_st = mvq_via_pre_quantized_q8_1(k_qstor, &y_q8_1, hidden, k_out_rows, 1)
            .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        let _ = k_qstor;
        let v_qstor = match v_arc.storage() {
            QStorage::Cuda(s) => s,
            _ => return Ok(None),
        };
        let v_cuda_st = mvq_via_pre_quantized_q8_1(v_qstor, &y_q8_1, hidden, v_out_rows, 1)
            .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        let _ = v_qstor;

        Ok(Some((
            cuda_ext::tensor_from_cuda_storage(q_cuda_st, (1usize, 1usize, q_out_rows))?,
            cuda_ext::tensor_from_cuda_storage(k_cuda_st, (1usize, 1usize, k_out_rows))?,
            cuda_ext::tensor_from_cuda_storage(v_cuda_st, (1usize, 1usize, v_out_rows))?,
        )))
    }

    /// Phi2 shared-Q8_1 prelude. Pre-quantizes x_norm to Q8_1 once, then
    /// computes attn_qkv via mvq_via_pre_quantized_q8_1 (skipping the
    /// redundant in-QMatMul quantize). Returns (qkv, q8_1_buf) so the
    /// caller can reuse the buffer for ffn_up. Returns (None, None) on
    /// any incompatibility (caller falls back to QMatMul.forward).
    ///
    /// Saves ~1 launch per layer (the second quantize_q8_1 that ffn_up
    /// would otherwise re-do).
    #[cfg(feature = "cuda")]
    #[allow(clippy::type_complexity)]
    pub(super) fn try_phi2_shared_q8_1_qkv(
        &self,
        x_norm: &Tensor,
    ) -> Result<(
        Option<Tensor>,
        Option<crate::tensor::cuda_ext::CudaSlice<u8>>,
    )> {
        use crate::tensor::cuda_ext;
        use crate::tensor::quantized::QStorage;
        // Use fast_mmvq's plain kernel via `mvq_plain_any_via_shared_q8_1`
        // - dispatches by qstor dtype so it works for Q4_0 (moondream),
        // Q4_K, Q5_K, Q6_K, Q8_0. quantize_q8_1_fast_mmvq_f32 produces
        // the EXACT layout that the plain mvq kernels read (verified by
        // test_ln_qmm_gguf binary, max_diff=0 on real moondream weights).
        use crate::inference::quantized_cuda::{
            mvq_plain_any_via_shared_q8_1 as mvq_via_pre_quantized_q8_1,
            quantize_q8_1_fast_mmvq_f32 as _quantize_q8_1_pub,
        };

        let (b, seq, hidden) = match x_norm.dims3() {
            Ok(d) => d,
            Err(_) => return Ok((None, None)),
        };
        if !(seq == 1 && b == 1 && x_norm.device().is_cuda()) {
            return Ok((None, None));
        }
        if !(self.flags.is_phi2_simple_ffn && self.flags.parallel_attn) {
            return Ok((None, None));
        }
        if !self.flags.fused_qkv {
            return Ok((None, None));
        }
        // Need attn_qkv as QTensor and ffn_up as QTensor - otherwise the
        // mvq path doesn't apply.
        let attn_qkv = match &self.attn_qkv {
            Some(qm) => qm,
            None => return Ok((None, None)),
        };
        let attn_qkv_arc = match attn_qkv.qtensor() {
            Some(q) => q.clone(),
            None => return Ok((None, None)),
        };
        if self.ffn_up.qtensor().is_none() {
            return Ok((None, None));
        }
        // Q4_K mvq fast path requires ncols multiple of 256.
        if !hidden.is_multiple_of(256) {
            return Ok((None, None));
        }
        // Skip when QKV has separate bias add that needs raw x before
        // quantize (phi2 has no qkv_bias; check to be safe).
        if self.flags.has_qkv_bias {
            return Ok((None, None));
        }

        let dev = x_norm.device().as_cuda_device()?;
        let x_f32 = if x_norm.dtype() == crate::tensor::DType::F32 {
            x_norm.clone()
        } else {
            x_norm.to_dtype(crate::tensor::DType::F32)?
        };
        let x_c = x_f32.contiguous()?;

        use crate::tensor::quantized::MATRIX_ROW_PADDING;
        let kx_padded = hidden.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        let y_size_bytes = (kx_padded / 32) * 36;
        let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_bytes)?;

        // Quantize x_norm -> Q8_1 buffer using fast_mmvq's quantize kernel
        // (same layout as what QMatMul.forward / fast_mmvq::try_fwd uses).
        // x_norm is on CUDA (checked above) so the raw-slice borrow holds.
        {
            let xs = cuda_ext::f32_slice_of(&x_c)?;
            let x_view = xs.view()?;
            _quantize_q8_1_pub(&x_view, &mut y_q8_1, hidden, &dev)
                .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        }

        // mvq attn_qkv via fast_mmvq's plain kernel (bypasses buggy IMMA).
        let qkv_out_rows = attn_qkv_arc.shape().dims()[0];
        let qstor = match attn_qkv_arc.storage() {
            QStorage::Cuda(s) => s,
            _ => return Ok((None, None)),
        };
        let qkv_cuda_st = mvq_via_pre_quantized_q8_1(qstor, &y_q8_1, hidden, qkv_out_rows, &dev)
            .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        let _ = qstor;

        let qkv = cuda_ext::tensor_from_cuda_storage(qkv_cuda_st, (1, 1, qkv_out_rows))?;

        Ok((Some(qkv), Some(y_q8_1)))
    }

    /// Whether this layer holds a (CUDA-only) Q8 KV cache. Always false on
    /// non-CUDA builds, where the quantized KV caches don't exist.
    #[cfg(feature = "cuda")]
    #[inline]
    pub(super) fn has_q8_cache(&self) -> bool {
        self.q8_kv_cache.is_some()
    }
    #[cfg(not(feature = "cuda"))]
    #[inline]
    pub(super) fn has_q8_cache(&self) -> bool {
        false
    }
    #[cfg(feature = "cuda")]
    #[inline]
    pub(super) fn has_q4_cache(&self) -> bool {
        self.q4_kv_cache.is_some()
    }
    #[cfg(not(feature = "cuda"))]
    #[inline]
    pub(super) fn has_q4_cache(&self) -> bool {
        false
    }

    /// Run attention. If `shared_kv` is Some, skip K/V computation and use the
    /// provided cached (K, V) tensors from a reference layer (shared KV layers in Gemma4).
    #[inline]
    pub(super) fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        shared_kv: Option<(Tensor, Tensor)>,
        donor_f16: Option<&crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
        #[cfg(feature = "cuda")] shared_kv_q8: Option<&crate::inference::cache::q8_kv::Q8KvCache>,
    ) -> Result<Tensor> {
        // The shared-KV-Q8 fast path (try_forward_attn_shared_q8) is
        // numerically unverified against the standard path - keep the
        // safe legacy fallback. Re-enable when validated.
        #[cfg(feature = "cuda")]
        let _ = shared_kv_q8;
        self.forward_attn_inner(x, mask, index_pos, shared_kv, None, donor_f16)
    }

    /// Variant of forward_attn that takes a pre-computed QKV F32 tensor.
    /// When present, the inner path skips its own attn_qkv.forward call
    /// and uses the supplied qkv. Used by the phi2 shared-Q8_1 fast path
    /// where attn_qkv and ffn_up share a single pre-quantized x_norm.
    pub(super) fn forward_attn_with_qkv(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        shared_kv: Option<(Tensor, Tensor)>,
        pre_computed_qkv: Option<&Tensor>,
        donor_f16: Option<&crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
    ) -> Result<Tensor> {
        self.forward_attn_inner(x, mask, index_pos, shared_kv, pre_computed_qkv, donor_f16)
    }

    fn forward_attn_inner(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        shared_kv: Option<(Tensor, Tensor)>,
        pre_computed_qkv: Option<&Tensor>,
        donor_f16: Option<&crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;

        // --- Fused QKV+norm+RoPE fast path (Qwen3-style models) -----------
        // Single fused launch covers wqkv + q_norm + k_norm + RoPE + cast,
        // replacing ~6 ops with 1. Conditions: single-token decode, CUDA,
        // has fused QKV, neox-style RoPE with full RoPE,
        // no rope_freq_factors, not in graph mode.
        // Routes through Q8 fast attn path if q8_kv_cache active, else
        // through the F-dtype kv_cache + standard_attention path.
        // Two variants: with-norm (qwen3, gemma3 - has_qk_norm) and
        // no-norm (qwen2, llama, mistral - no q_norm/k_norm).
        // Partial RoPE (rope_dim < head_dim, e.g. phi2 with rope_dim=32,
        // head_dim=64) routed through attn_post_qkv_decode_qf32_no_norm_partial_rope.
        // Combined with the dev_pos attn_scores/softmax_output and the Q8
        // KV auto-promote, this gives moondream decode +30 % (273 -> 355
        // tok/s) vs the dispatcher building Q/K/V separately then doing
        // partial RoPE via facade's standard chain.
        let allow_partial_rope = true;
        let rope_ok = self.rope_dim == 0
            || self.rope_dim == self.head_dim
            || (allow_partial_rope
                && self.rope_dim > 0
                && self.rope_dim < self.head_dim
                && self.rope_dim.is_multiple_of(2));
        let want_apq_fast = seq == 1
            && b == 1
            && x.device().is_cuda()
            && self.attn_qkv.is_some()
            && rope_ok
            && self.rope_freq_factors.is_none()
            // NOTE: graph mode is now ALLOWED here (was gated out by
            // graph_rope_cos.is_none()). In graph mode the fused kernel reads
            // RoPE from the per-step-updated graph_rope_cos/sin buffer (index 0)
            // instead of self.cos[index_pos] - see the cos/sin selection below.
            && shared_kv.is_none();
        let want_apq_with_norm =
            self.flags.has_qk_norm && self.attn_q_norm.is_some() && self.attn_k_norm.is_some();
        let want_apq_no_norm = !self.flags.has_qk_norm
            // qwen2-family biases are added as one broadcast_add on the qkv
            // tensor before the fused norm+RoPE+cast kernel.
            && (!self.flags.has_qkv_bias || self.attn_qkv_bias.is_some());
        // For the no-norm case the fast path produces F32 Q/V which the
        // quantized cache consumes directly. The body below only has a
        // Q8 implementation for no-norm - the Q4 + no-norm combination
        // falls through to standard_attention (which already handles
        // Q4 K/V append + dequant correctly) to avoid panicking on the
        // unconditional `q_norm_w.unwrap()` in the F-dtype else branch.
        // GH_APQ enable gate. Previously had a buggy-shape exclusion for
        // GQA hd=128 - that bug was actually a downstream DOUBLE-SCALING
        // issue in attn_softmax_output_graph (fixed at the call site:
        // GH_APQ pre-scales Q by q_scale, so the softmax kernel must
        // receive scale=1.0, not q_scale). The shape exclusion is no
        // longer needed.
        // GH_APQ fused QKV+norm+RoPE+cast - default-on for all supported
        // shapes after the fix (commit e23e7a6) corrected the
        // Q8 dev_pos cur_pos_dev off-by-one that previously hid attention
        // numerical issues for GQA hd=128. Re-enabled unconditionally.
        let disable_apq = false;
        // Graph-mode fused fast-path is validated ONLY for the qwen3 family. Other
        // with_norm arches (e.g. gemma4: SWA + hd256 + special handling) are NOT
        // validated here and were not a loss - keep them on their original path.
        // no_norm is NOT graph-safe at all (crashed qwen2.5:0.5b in capture).
        // Discriminate qwen3-family from gemma4 at layer scope: gemma4 applies a V
        // RMS-norm (attn_v_norm_ones = Some), qwen3/qwen3moe do not (None). Only the
        // latter is validated for the graph-mode fused path.
        let graph_fused_ok = self.graph_rope_cos.is_none() || self.attn_v_norm_ones.is_none();
        let want_apq_fast = want_apq_fast
            && !disable_apq
            && ((want_apq_with_norm && graph_fused_ok)
                || (want_apq_no_norm && self.has_q8_cache() && self.graph_rope_cos.is_none()));

        if want_apq_fast {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "🟩 GH_APQ fast path active (n_head={} n_kv={} hd={} q8_cache={} norm={})",
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                    self.has_q8_cache(),
                    if want_apq_with_norm { "yes" } else { "no" }
                );
            }

            // Fused QKV -> F32 flat buffer.
            // When pre_computed_qkv is supplied by the phi2 shared-Q8_1
            // fast path, skip the QMatMul.forward (saves one quantize +
            // one matmul launch - caller already produced qkv from
            // mvq_via_pre_quantized_q8_1).
            let mut qkv = match pre_computed_qkv {
                Some(pre) => pre.clone(),
                None => self.attn_qkv.as_ref().unwrap().forward(x)?,
            };
            // qwen2-family bias add: applied to qkv before reshape/cast so
            // the kernel sees Q/K/V with biases already folded in.
            if self.flags.has_qkv_bias {
                if let Some(bqkv) = self.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(bqkv)?;
                }
            }
            let total = self.n_head * self.head_dim + 2 * self.n_kv_head * self.head_dim;
            let qkv_flat = qkv.reshape((qkv.elem_count() / total, total))?;
            let qkv_flat = if qkv_flat.dtype() == crate::tensor::DType::F32 {
                qkv_flat
            } else {
                qkv_flat.to_dtype(crate::tensor::DType::F32)?
            };

            // q_norm/k_norm are only needed in the with-norm path; load them lazily.
            let (q_norm_w, k_norm_w, rms_eps) = if want_apq_with_norm {
                let q_norm = self.attn_q_norm.as_ref().unwrap();
                let k_norm = self.attn_k_norm.as_ref().unwrap();
                let qw = q_norm.weight();
                let kw = k_norm.weight();
                let qw = if qw.dtype() == crate::tensor::DType::F32 {
                    qw.clone()
                } else {
                    qw.to_dtype(crate::tensor::DType::F32)?
                };
                let kw = if kw.dtype() == crate::tensor::DType::F32 {
                    kw.clone()
                } else {
                    kw.to_dtype(crate::tensor::DType::F32)?
                };
                (Some(qw), Some(kw), q_norm.eps() as f32)
            } else {
                (None, None, 0.0f32)
            };

            // RoPE source: in graph mode the position is frozen at capture, so the
            // fused kernel must read the per-step-updated single-row graph_rope_cos/sin
            // buffer at index 0 (update_rope_buffers refreshes its contents each step).
            // Outside graph mode, use the full cos/sin table indexed by index_pos.
            let (src_cos, src_sin, kernel_pos) = match (&self.graph_rope_cos, &self.graph_rope_sin)
            {
                (Some(gc), Some(gs)) => (gc.clone(), gs.clone(), 0usize),
                _ => (self.cos.clone(), self.sin.clone(), index_pos),
            };
            let cos = if !src_cos.device().same_device(&x.device()) {
                src_cos.to_device(&x.device())?
            } else {
                src_cos
            };
            let sin = if !src_sin.device().same_device(&x.device()) {
                src_sin.to_device(&x.device())?
            } else {
                src_sin
            };
            let model_dtype = self.cos.dtype();
            let cos = if cos.dtype() == model_dtype {
                cos
            } else {
                cos.to_dtype(model_dtype)?
            };
            let sin = if sin.dtype() == model_dtype {
                sin
            } else {
                sin.to_dtype(model_dtype)?
            };

            let attn_scale = self
                .attention_scale
                .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt());
            let q_scale = attn_scale as f32;
            let rope_style = if self.flags.use_rope_i { 1 } else { 0 };

            let use_q8 = self.has_q8_cache();
            if use_q8 {
                // Q8 KV fast-path is CUDA-only (quantized caches don't exist
                // on non-CUDA builds, where `use_q8` is always false).
                #[cfg(feature = "cuda")]
                {
                    // Q8 path: emit Q & V as F32 (skip downstream casts).
                    //
                    // CRITICAL: the three branches below differ in whether Q gets
                    // pre-scaled by q_scale (1/sqrt(hd)) inside GH_APQ. The
                    // downstream `attn_softmax_output_graph` call multiplies
                    // scores by its `scale` arg - so we must pass q_scale=1.0
                    // when Q is pre-scaled (with_norm + no_norm full_rope) and
                    // pass q_scale when Q is NOT pre-scaled (partial_rope).
                    // Mismatch -> double-scale (with_norm/no_norm-full-rope  -
                    // softmax crushes flat, deepcoder/devstral hd=128 garbage,
                    // fixed by commit dc6f67b) or zero-scale (partial_rope  -
                    // softmax too sharp, moondream emits fragments).
                    let mut q_was_pre_scaled = true;
                    let (q_h, k_h, v_h) = if want_apq_with_norm {
                        crate::inference::moe_cuda::attn_post_qkv_decode_qf32(
                            &qkv_flat,
                            q_norm_w.as_ref().unwrap(),
                            k_norm_w.as_ref().unwrap(),
                            &cos,
                            &sin,
                            self.n_head,
                            self.n_kv_head,
                            self.head_dim,
                            kernel_pos,
                            rms_eps,
                            q_scale,
                            model_dtype,
                            rope_style,
                        )?
                    } else if self.rope_dim > 0 && self.rope_dim < self.head_dim {
                        // Partial RoPE (phi2/phi3: rope_dim=32, head_dim=64).
                        // Passes q_scale=1.0 so Q is NOT pre-scaled - downstream
                        // softmax applies the scale.
                        q_was_pre_scaled = false;
                        let (q, k, v) = crate::inference::moe_cuda::attn_post_qkv_decode_qf32_no_norm_partial_rope(
                        &qkv_flat, &cos, &sin,
                        self.n_head, self.n_kv_head, self.head_dim,
                        self.rope_dim,
                        kernel_pos, 1.0f32, model_dtype, rope_style,
                    )?;
                        (q, k, v)
                    } else {
                        crate::inference::moe_cuda::attn_post_qkv_decode_qf32_no_norm(
                            &qkv_flat,
                            &cos,
                            &sin,
                            self.n_head,
                            self.n_kv_head,
                            self.head_dim,
                            kernel_pos,
                            q_scale,
                            model_dtype,
                            rope_style,
                        )?
                    };
                    let _ = (rms_eps, q_norm_w.as_ref(), k_norm_w.as_ref());
                    let q = q_h.reshape((1, self.n_head, 1, self.head_dim))?;
                    let k = k_h.reshape((1, self.n_kv_head, 1, self.head_dim))?;
                    let mut v = v_h.reshape((1, self.n_kv_head, 1, self.head_dim))?;
                    // gemma4: V also gets RMS-norm-no-weight (matches the
                    // standard_attention path). Use the reference fused single-launch
                    // rms_norm with a unit-weight tensor - replaces the manual
                    // sqr+mean+sqrt+div sequence (~5 launches) with 1.
                    if let Some(ones) = self.attn_v_norm_ones.as_ref() {
                        let v_f32 = if v.dtype() == crate::tensor::DType::F32 {
                            v.clone()
                        } else {
                            v.to_dtype(crate::tensor::DType::F32)?
                        };
                        let v_normed = crate::tensor::ops::rms_norm(&v_f32, ones, 1e-6f32)?;
                        v = if v_normed.dtype() == v.dtype() {
                            v_normed
                        } else {
                            v_normed.to_dtype(v.dtype())?
                        };
                    }

                    let cache = self.q8_kv_cache.as_mut().unwrap();
                    if index_pos == 0 {
                        cache.reset();
                        self.kv_cache.reset();
                    }
                    cache
                        .append(&k, &v)
                        .map_err(|e| crate::tensor::Error::msg(format!("Q8 append: {e}")))?;

                    // BUG FIX: `cache.attn_output` uses the legacy
                    // 1024-thread Q8 kernel which hits CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
                    // at phi2's hd=64 register budget. Switch to
                    // the dev_pos kernel pair (16-warp x 512-thread) used by
                    // try_q8_graph_decode at line 1853. Requires cur_pos_dev to
                    // be primed (host current_seq_len is the slot count BEFORE
                    // append - after the append above, set it to current_seq_len-1).
                    let in_capture = crate::tensor::cuda_ext::capture_active(&q.device());
                    if !in_capture {
                        // Fix: the kernel comment in
                        // `attn_score_q8_inner_dev_pos` at quantized.cu:4279 says
                        // "Host stores current_seq_len BEFORE the append" and
                        // the kernel adds 1. Passing the POST-append value
                        // results in seq_kv being one too large; the score
                        // kernel then reads K[cur_seq] which is uninitialized
                        // cudaMallocAsync memory -> non-deterministic garbage.
                        // The downstream softmax+output kernel reads V[cur_seq]
                        // (also uninitialized) -> cyclic-3 non-determinism in
                        // generated tokens at temperature=0. The prior comment
                        // claimed "off-by-one but benign"; verified NOT benign
                        // via scripts/repro_task26_determinism.py (passes after
                        // this fix).
                        let cur_seq = cache.current_seq_len();
                        cache
                            .update_graph_state(cur_seq.saturating_sub(1))
                            .map_err(|e| {
                                crate::tensor::Error::msg(format!("Q8 prime cur_pos_dev: {e}"))
                            })?;
                    }
                    // Fully-fused 1-kernel path (Q.K + online-softmax + V.attn)
                    // via v3 kernel (16-warp chunk-by-warp design). Numerically
                    // correct but DOES NOT WIN over the 2-kernel chain in
                    // production:
                    //   2-kernel chain: 340.6 tok/s ± 13.3 (FUSED_ATTN=0)
                    //   v3 fused      : 333.9 tok/s ±  5.9 (FUSED_ATTN=1)
                    //   ->  -2 % (within noise)
                    //
                    // The 2-kernel chain has ~4x more warps active concurrently
                    // (attn_scores_graph parallelises over both seq and kv_head)
                    // which compensates for the launch overhead + scores buffer
                    // alloc. v3 fused path is permanently OFF in production  -
                    // 2-kernel chain wins by 2 % on moondream.
                    let try_fused = false;
                    // SPLIT-K flash-decode (diagnostic A/B, gated OFF by default).
                    // MHA (moondream, n_q_per_kv=1): NSPLITxn_q_heads blocks -> good
                    // occupancy, ~+4% (still gated: length-confounded, doesn't flip
                    // the -27% loss alone).
                    // GQA (deepcoder/devstral/qwen3): attn_flash_splitk_decode_graph
                    // now also handles GQA via a KV-bandwidth-sharing kernel (each
                    // (split,h_kv) warp reads its KV head ONCE for all n_q_per_kv
                    // query heads). VALIDATED CORRECT (coherent long-ctx recall) but
                    // MEASURED -9% at long ctx: the grid is NSPLITxn_KV_heads, and
                    // n_kv_heads (8) « n_q_heads (32-40), so only 16x8=128 single-
                    // warp blocks -> underfills ~70 SMs (occupancy-limited), losing
                    // to the 2-kernel chain's wider parallelism. The KV-read saving
                    // doesn't compensate. Untried fix = NSPLIT=64 (->512 blocks) or
                    // multi-warp/shared-mem-KV blocks. NOTE: long context on these GQA
                    // cells is not where this path is losing time.
                    // with RUNTIME-ADAPTIVE nsplit (launcher targets
                    // ~384 blocks regardless of n_kv_heads, vs the old fixed 16x8=128
                    // that underfilled the GPU) the GQA split-K now WINS: deepcoder
                    // long 60.8->68.3 (+12%), short neutral (78 vs 80), coherent.
                    // So AUTO-ENABLE it for GQA (n_q_per_kv>1, hd∈{64,128}). MHA
                    // (moondream, n_q_per_kv=1) stays OFF: re-measured on
                    // the real vision-caption workload (short KV) it is pure noise
                    // (short -0.4%, medium +1%, long +1.8%, all within ±9 stddev)  -
                    // the old "+4%" only appeared at long KV, which captioning never
                    // reaches, so there is nothing for the KV-chunk split to divide.
                    // On any kernel error the match below falls back to the chain.
                    let try_splitk = cache.n_kv_heads() > 0
                        && self.n_head > cache.n_kv_heads()
                        && (self.head_dim == 64 || self.head_dim == 128);
                    // Softmax scale must complement upstream Q pre-scaling:
                    //   q_was_pre_scaled=true  (with_norm, no_norm full_rope) -> scale=1.0
                    //   q_was_pre_scaled=false (partial_rope, moondream)      -> scale=q_scale
                    // Wrong choice double-scales softmax (flat distribution -> garbage
                    // for hd=128 GQA) or zero-scales it (sharp -> fragments for
                    // moondream). See dc6f67b for the original double-scale fix.
                    let softmax_scale = if q_was_pre_scaled { 1.0f32 } else { q_scale };
                    // -- gemma4 windowed-SWA decode ------------------------------
                    // SWA layers (self.sliding_window set) attend only to the last `w`
                    // keys. Same dev_pos 2-kernel chain, but window>0 makes the score +
                    // softmax-output kernels scan ONLY [seq_kv-w, seq_kv) (out-of-window
                    // tokens -INF'd + skipped) - bounds the scan at 512 instead of the
                    // full growing KV (the 2.5K collapse) AND matches gemma4's trained
                    // windowed attention. window=0 everywhere else = bit-identical.
                    let swa_window = self
                        .sliding_window
                        .filter(|&w| w > 0 && cache.current_seq_len() > w);
                    let y_q8 = if let Some(window) = swa_window {
                        let scores = cache.attn_scores_graph(&q, window).map_err(|e| {
                            crate::tensor::Error::msg(format!("Q8 SWA attn_scores_graph: {e}"))
                        })?;
                        cache
                            .attn_softmax_output_graph(&scores, softmax_scale, window)
                            .map_err(|e| {
                                crate::tensor::Error::msg(format!(
                                    "Q8 SWA attn_softmax_output_graph: {e}"
                                ))
                            })?
                    } else if try_splitk {
                        // Query-head-packed tensor-core flash decode. It needs the fragments
                        // Ampere brought, a head dimension of 128, and enough KV to be worth
                        // the deeper grid. Any Err falls through to split-K, which falls
                        // through to the two-kernel chain, so a card that cannot run it loses
                        // nothing.
                        let try_tc = self.head_dim == 128
                            && cache.current_seq_len()
                                >= crate::inference::kernel::flash_decode_tc::TC_MIN_KV
                            && q.device()
                                .as_cuda_device()
                                .map(|d| d.has_ampere_tensor_cores())
                                .unwrap_or(false);
                        let tc_out = if try_tc {
                            match cache.attn_flash_tc_decode_graph(&q, softmax_scale) {
                                Ok(y) => Some(y),
                                Err(e) => {
                                    tracing::warn!(
                                        "attn_flash_tc_decode_graph fallback to split-K: {e}"
                                    );
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        if let Some(y) = tc_out {
                            y
                        } else {
                            match cache.attn_flash_splitk_decode_graph(&q, softmax_scale) {
                                Ok(y) => y,
                                Err(e) => {
                                    tracing::warn!("attn_flash_splitk_decode_graph fallback to 2-kernel chain: {e}");
                                    let scores = cache.attn_scores_graph(&q, 0).map_err(|e| {
                                        crate::tensor::Error::msg(format!(
                                            "Q8 attn_scores_graph: {e}"
                                        ))
                                    })?;
                                    cache
                                        .attn_softmax_output_graph(&scores, softmax_scale, 0)
                                        .map_err(|e| {
                                            crate::tensor::Error::msg(format!(
                                                "Q8 attn_softmax_output_graph: {e}"
                                            ))
                                        })?
                                }
                            }
                        }
                    } else if try_fused {
                        match cache.attn_fused_decode_graph(&q, softmax_scale) {
                            Ok(y) => y,
                            Err(e) => {
                                tracing::warn!(
                                    "attn_fused_decode_graph fallback to 2-kernel chain: {e}"
                                );
                                let scores = cache.attn_scores_graph(&q, 0).map_err(|e| {
                                    crate::tensor::Error::msg(format!("Q8 attn_scores_graph: {e}"))
                                })?;
                                cache
                                    .attn_softmax_output_graph(&scores, softmax_scale, 0)
                                    .map_err(|e| {
                                        crate::tensor::Error::msg(format!(
                                            "Q8 attn_softmax_output_graph: {e}"
                                        ))
                                    })?
                            }
                        }
                    } else {
                        let scores = cache.attn_scores_graph(&q, 0).map_err(|e| {
                            crate::tensor::Error::msg(format!("Q8 attn_scores_graph: {e}"))
                        })?;
                        cache
                            .attn_softmax_output_graph(&scores, softmax_scale, 0)
                            .map_err(|e| {
                                crate::tensor::Error::msg(format!(
                                    "Q8 attn_softmax_output_graph: {e}"
                                ))
                            })?
                    };
                    let attn_out_dim = self.n_head * self.head_dim;
                    // At decode (seq==1) y_q8 is [b, n_head, 1, hd]; transpose(1,2)
                    // then reshape forces a contiguous copy whose order matches a
                    // direct reshape (a size-1 seq dim cannot reorder elements),
                    // so skip it - free view, bit-identical. seq>1 (prefill) still
                    // needs the real transpose.
                    let y = if seq == 1 {
                        y_q8.reshape(&[b, seq, attn_out_dim])?
                    } else {
                        y_q8.transpose(1, 2)?.reshape(&[b, seq, attn_out_dim])?
                    };
                    let _ = mask;
                    return self.attn_output.forward(&y);
                }
            } else {
                // F-dtype path: with-norm variant only. Fused CUDA decode kernel;
                // this whole Q8/fused decode region only activates with a Q8 KV
                // cache, which is CUDA-only, so it's unreachable under --features cpu.
                #[cfg(not(feature = "cuda"))]
                #[allow(clippy::diverging_sub_expression)]
                let (q_h, k_h, v_h): (Tensor, Tensor, Tensor) =
                    unreachable!("fused F-dtype decode (attn_post_qkv_decode) is CUDA-only");
                #[cfg(feature = "cuda")]
                let (q_h, k_h, v_h) = crate::inference::moe_cuda::attn_post_qkv_decode(
                    &qkv_flat,
                    q_norm_w.as_ref().unwrap(),
                    k_norm_w.as_ref().unwrap(),
                    &cos,
                    &sin,
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                    index_pos,
                    rms_eps,
                    q_scale,
                    model_dtype,
                    rope_style,
                )?;
                let q = q_h.reshape((1, self.n_head, 1, self.head_dim))?;
                let k_new = k_h.reshape((1, self.n_kv_head, 1, self.head_dim))?;
                let mut v_new = v_h.reshape((1, self.n_kv_head, 1, self.head_dim))?;
                // gemma4: V also gets RMS-norm-no-weight after extraction.
                if let Some(ones) = self.attn_v_norm_ones.as_ref() {
                    let v_f32 = if v_new.dtype() == crate::tensor::DType::F32 {
                        v_new.clone()
                    } else {
                        v_new.to_dtype(crate::tensor::DType::F32)?
                    };
                    let v_normed = crate::tensor::ops::rms_norm(&v_f32, ones, 1e-6f32)?;
                    v_new = if v_normed.dtype() == v_new.dtype() {
                        v_normed
                    } else {
                        v_normed.to_dtype(v_new.dtype())?
                    };
                }
                if index_pos == 0 {
                    self.kv_cache.reset();
                }
                // gemma4 F16 KV lever: store K/V at F16 so the growing-context
                // score/output matmuls read half the HBM bytes (the dominant
                // long-context decode cost) at ollama-parity precision. Q/norms/
                // RoPE stay F32; only the cached K/V and the two attention
                // matmuls below drop to F16. Keeps the prefill (standard path)
                // and decode (this GH_APQ path) cache dtype consistent.
                let (k_new, v_new) = if self.kv_f16 {
                    let k16 = if k_new.dtype() == crate::tensor::DType::F16 {
                        k_new
                    } else {
                        k_new.to_dtype(crate::tensor::DType::F16)?
                    };
                    let v16 = if v_new.dtype() == crate::tensor::DType::F16 {
                        v_new
                    } else {
                        v_new.to_dtype(crate::tensor::DType::F16)?
                    };
                    (k16, v16)
                } else {
                    (k_new, v_new)
                };
                let (k_full, v_full) = self.kv_cache.append(&k_new, &v_new)?;

                // Pre-scaled attention (q_scale was already folded into Q
                // by attn_post_qkv_decode). For seq=1 GQA, this matches
                // standard_attention's single-token fast path but skips
                // the redundant `affine(scale as f32, 0.0)` call.
                let n_rep = self.n_head / self.n_kv_head;
                let kv_dt = k_full.dtype();
                // Cast Q down to the cache dtype so the matmuls read F16 K/V.
                let qd = if q.dtype() == kv_dt {
                    q.clone()
                } else {
                    q.to_dtype(kv_dt)?
                };
                // Softmax in F32 (ollama parity), back to kv dtype for p.v.
                let sm = |att: &Tensor| -> Result<Tensor> {
                    if kv_dt == crate::tensor::DType::F32 {
                        crate::tensor::ops::softmax_last_dim(att)
                    } else {
                        Ok(crate::tensor::ops::softmax_last_dim(
                            &att.to_dtype(crate::tensor::DType::F32)?,
                        )?
                        .to_dtype(kv_dt)?)
                    }
                };
                let y = if n_rep > 1 {
                    let (bb, _n_head, _one, d) = qd.dims4()?;
                    let q_grouped = qd.reshape((bb, self.n_kv_head, n_rep, d))?;
                    let att = q_grouped.matmul_t(&k_full)?;
                    let att = sm(&att)?;
                    let out = att.matmul(&v_full)?;
                    out.reshape((bb, self.n_head, 1, d))?
                } else {
                    let att = qd.matmul_t(&k_full)?;
                    let att = sm(&att)?;
                    att.matmul(&v_full.contiguous()?)?
                };
                let y = if y.dtype() == q.dtype() {
                    y
                } else {
                    y.to_dtype(q.dtype())?
                };
                let attn_out_dim = self.n_head * self.head_dim;
                let y = y.transpose(1, 2)?.reshape(&[b, seq, attn_out_dim])?;
                let _ = mask; // single-token decode: causal mask trivial
                return self.attn_output.forward(&y);
            }
        }

        // -- Q / K / V projections ------------------------------------------
        // Shared-KV layers (Gemma4 last 18 layers) skip K and V projections
        // entirely - llama.cpp does the same via `has_kv(il) == false`. The
        // donor layer's already-RoPE'd K/V come in via `shared_kv` and are
        // used directly below. Our matmul-heavy K and V projections were
        // wasted work otherwise.
        let is_shared_layer = shared_kv.is_some();
        let kv_dim_dummy = self.n_kv_head * self.head_dim;
        let (mut q, mut k, v) = if self.flags.fused_qkv || self.attn_q.is_none() {
            // Fused QKV: single [b, seq, (n_head + 2*n_kv_head)*head_dim] projection
            let mut qkv = self.attn_qkv.as_ref().unwrap().forward(x)?;
            let q_dim = self.n_head * self.head_dim;
            let kv_dim = self.n_kv_head * self.head_dim;
            // QKV biases (qwen2 family): when all three are present and the
            // shared layer doesn't need to skip K/V, one broadcast_add on the
            // full qkv tensor replaces three per-component broadcast_adds
            // (saves 2 launches/layer). Falls back to the per-component path
            // for shared layers (need to skip K/V bias) or when only Q has a
            // bias.
            if self.flags.has_qkv_bias && !is_shared_layer {
                if let Some(bqkv) = self.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(bqkv)?;
                }
            }
            let mut q = qkv.narrow(crate::tensor::D::Minus1, 0, q_dim)?;
            let mut k = qkv.narrow(crate::tensor::D::Minus1, q_dim, kv_dim)?;
            let mut v = qkv.narrow(crate::tensor::D::Minus1, q_dim + kv_dim, kv_dim)?;
            if self.flags.has_qkv_bias && (self.attn_qkv_bias.is_none() || is_shared_layer) {
                if let (Some(bq), Some(bk), Some(bv)) =
                    (&self.attn_q_bias, &self.attn_k_bias, &self.attn_v_bias)
                {
                    q = q.broadcast_add(bq)?;
                    if !is_shared_layer {
                        k = k.broadcast_add(bk)?;
                        v = v.broadcast_add(bv)?;
                    }
                }
            }
            (q, k, v)
        } else if is_shared_layer {
            // Skip K/V projections entirely - donor's K/V will replace the
            // dummies in the if-let-Some(shared_kv) branch below.
            let mut q = self.attn_q.as_ref().unwrap().forward(x)?;
            if self.flags.has_qkv_bias {
                if let Some(bq) = &self.attn_q_bias {
                    q = q.broadcast_add(bq)?;
                }
            }
            // Allocate tiny dummies that match dtype/device. These get
            // overwritten by `shared_k`/`shared_v` and never feed into a
            // matmul.
            let dummy =
                crate::tensor::Tensor::zeros_on((b, seq, kv_dim_dummy), q.dtype(), &q.device())?;
            (q, dummy.clone(), dummy)
        } else {
            // Phi2 shared-Q8_1 fast path: pre-quantize x once, share across
            // Q/K/V matmuls. Saves 2 launches/layer.
            #[cfg(feature = "cuda")]
            let shared_qkv = self.try_qkv_shared_q8_1(x).ok().flatten();
            #[cfg(not(feature = "cuda"))]
            let shared_qkv: Option<(Tensor, Tensor, Tensor)> = None;

            if let Some((mut q, mut k, mut v)) = shared_qkv {
                // Biases on top of the shared-path outputs (phi2/qwen2 family).
                if self.flags.has_qkv_bias {
                    if let (Some(bq), Some(bk), Some(bv)) =
                        (&self.attn_q_bias, &self.attn_k_bias, &self.attn_v_bias)
                    {
                        q = q.broadcast_add(bq)?;
                        k = k.broadcast_add(bk)?;
                        v = v.broadcast_add(bv)?;
                    }
                }
                (q, k, v)
            } else {
                // Standard split Q/K/V
                let mut q = self.attn_q.as_ref().unwrap().forward(x)?;
                let mut k = self.attn_k.as_ref().unwrap().forward(x)?;
                // Gemma4 26B Global layers: attn_v missing -> reuse k as v
                // (AttentionKEqV pattern, mirrors llama.cpp's `Vcur ?: Kcur`).
                let mut v = match self.attn_v.as_ref() {
                    Some(wv) => wv.forward(x)?,
                    None => k.clone(),
                };
                // Optional biases (Qwen2)
                if self.flags.has_qkv_bias {
                    if let (Some(bq), Some(bk), Some(bv)) =
                        (&self.attn_q_bias, &self.attn_k_bias, &self.attn_v_bias)
                    {
                        q = q.broadcast_add(bq)?;
                        k = k.broadcast_add(bk)?;
                        v = v.broadcast_add(bv)?;
                    }
                }
                (q, k, v)
            }
        };

        // -- Reshape to (b, heads, seq, head_dim) -------------------------
        let n_head = self.n_head;
        let n_kv_head = self.n_kv_head;
        let head_dim = self.head_dim;
        // OLMo2 QK-norm is FULL-DIM (weight length = n_head*head_dim, not head_dim):
        // RMS-norm the whole flat q/k projection BEFORE the head split, unlike the
        // per-head QK-norm of Gemma3/Qwen3 (applied after reshape below).
        let qk_norm_full = self.flags.has_qk_norm
            && self
                .attn_q_norm
                .as_ref()
                .map(|n| n.weight().dims1().map(|d| d != head_dim).unwrap_or(false))
                .unwrap_or(false);
        if qk_norm_full {
            if let (Some(q_norm), Some(k_norm)) = (&self.attn_q_norm, &self.attn_k_norm) {
                q = q_norm.forward(&q.contiguous()?)?;
                if !is_shared_layer {
                    k = k_norm.forward(&k.contiguous()?)?;
                }
            }
        }
        q = q.reshape((b, seq, n_head, head_dim))?.transpose(1, 2)?;
        k = k.reshape((b, seq, n_kv_head, head_dim))?.transpose(1, 2)?;
        let mut v = v
            .reshape((b, seq, n_kv_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // -- Optional QK norms (Gemma3/4) - PER-HEAD (skip if full-dim done above) --
        if self.flags.has_qk_norm && !qk_norm_full {
            if let (Some(q_norm), Some(k_norm)) = (&self.attn_q_norm, &self.attn_k_norm) {
                q = q_norm.forward(&q.contiguous()?)?;
                if !is_shared_layer {
                    k = k_norm.forward(&k.contiguous()?)?;
                }
            }
            // Gemma4: V also gets RMS norm (without learnable weight).
            // Use the reference fused single-launch rms_norm with a unit-weight
            // tensor.
            if !is_shared_layer {
                if let Some(ones) = self.attn_v_norm_ones.as_ref() {
                    let v_f32 = if v.dtype() == crate::tensor::DType::F32 {
                        v.clone()
                    } else {
                        v.to_dtype(crate::tensor::DType::F32)?
                    };
                    let v_normed = crate::tensor::ops::rms_norm(&v_f32, ones, 1e-6f32)?;
                    v = if v_normed.dtype() == v.dtype() {
                        v_normed
                    } else {
                        v_normed.to_dtype(v.dtype())?
                    };
                }
            }
        }

        // -- RoPE ----------------------------------------------------------
        // Graph-mode ops: only for single-token decode (seq_len == 1) when
        // buffers have been primed by update_graph_state. During prefill
        // (seq_len > 1), graph_rope_cos may still be set from a prior
        // generation, but we must NOT use the single-row buffers for it.
        let use_graph_ops = self.graph_rope_cos.is_some() && seq == 1;

        if use_graph_ops {
            q = self.apply_rotary_emb_from_buffer(&q)?;
        } else {
            q = self.apply_rotary_emb(&q, index_pos)?;
        }

        // -- KV: compute or reuse from shared layer ----------------------
        #[cfg(feature = "cuda")]
        let (k, v) = if let Some((shared_k, shared_v)) = shared_kv {
            (shared_k, shared_v)
        } else {
            if use_graph_ops {
                k = self.apply_rotary_emb_from_buffer(&k)?;
            } else {
                k = self.apply_rotary_emb(&k, index_pos)?;
            }
            if index_pos == 0 {
                self.kv_cache.reset();
                #[cfg(feature = "cuda")]
                if let Some(c) = self.q8_kv_cache.as_mut() {
                    c.reset();
                }
                #[cfg(feature = "cuda")]
                if let Some(c) = self.q4_kv_cache.as_mut() {
                    c.reset();
                }
            }
            if use_graph_ops {
                // Q4 graph-safe path: the Q4 cache's append uses the
                // device-pos kernels (kv_residual_scatter_dev_slot +
                // flush_k_residual_q4_dev_pos + q4_v_scatter_bytes_dev_pos)
                // when `cur_pos_dev` is set, so the launch sequence is
                // invariant across graph replays. Attention dispatch below
                // takes the `attn_scores_graph`/`attn_output_graph` path.
                if self.q4_kv_cache.is_some()
                    && self
                        .q4_kv_cache
                        .as_ref()
                        .unwrap()
                        .cur_pos_dev_ptr()
                        .is_some()
                {
                    self.q4_kv_cache
                        .as_mut()
                        .unwrap()
                        .append(&k, &v)
                        .map_err(|e| crate::tensor::Error::msg(format!("Q4 graph append: {e}")))?;
                    // `k, v` here are no-ops downstream - the Q4 graph
                    // decode path reads everything from the cache.
                    (k, v)
                } else if self.q8_kv_cache.is_some()
                    && self
                        .q8_kv_cache
                        .as_ref()
                        .unwrap()
                        .cur_pos_dev_ptr()
                        .is_some()
                {
                    // Q8 graph-safe path: mirror the Q4 branch above. The Q8
                    // cache's append writes K, V at the lazy capacity (4 K
                    // initial); the downstream attn dispatch routes through
                    // `try_q8_graph_decode` which uses cache.attn_scores_graph
                    // / cache.attn_output_graph at the stable max_seq_padded
                    // stride. Without this branch the code falls to the
                    // `kv_cache.append_padded` block below, which allocates
                    // F-dtype padded K,V at user_context_length (~12.8 GB for
                    // deepcoder at 32 K config) - exactly the OOM that
                    // 1e7ac88 hit and got reverted in c1f16cb/e308ab3.
                    self.q8_kv_cache
                        .as_mut()
                        .unwrap()
                        .append(&k, &v)
                        .map_err(|e| crate::tensor::Error::msg(format!("Q8 graph append: {e}")))?;
                    (k, v)
                } else {
                    if self.kv_cache.k_buffer().is_none() {
                        // First graph-mode call: allocate padded buffers
                        let (_, _, _) = self.kv_cache.append_padded(&k, &v)?;
                    } else {
                        let pos_idx = self
                            .graph_kv_pos
                            .as_ref()
                            .ok_or_else(|| crate::tensor::Error::msg("graph_kv_pos not set"))?;
                        let k_buf = self.kv_cache.k_buffer().unwrap();
                        let v_buf = self.kv_cache.v_buffer().unwrap();
                        k_buf.scatter_set(pos_idx, &k, 2)?;
                        v_buf.scatter_set(pos_idx, &v, 2)?;
                        self.kv_cache.advance_seq_len(1);
                    }
                    (
                        self.kv_cache.k_buffer().unwrap(),
                        self.kv_cache.v_buffer().unwrap(),
                    )
                }
            } else if self.q8_kv_cache.is_some() {
                // Consolidated Q8 mode: only Q8 cache holds history. F-dtype
                // SpecKvCache is never populated. Single-token decode is
                // served out of Q8 via try_q8_decode below (no dequant).
                // Multi-token paths (prefill, PLD verify) need full F-dtype
                // K,V to feed flash-attn / standard_attention - we
                // dequantize on the fly.
                //
                // Exception: when this layer is a donor for a shared_kv
                // model (gemma4), shared layers downstream read its F-dtype
                // kv_cache. We populate BOTH caches in that case.
                let append_res = self.q8_kv_cache.as_mut().unwrap().append(&k, &v);
                if self.populate_dual_kv {
                    let _ = self.kv_cache.append(&k, &v)?;
                }
                match append_res {
                    Ok(()) if seq == 1 && !self.populate_dual_kv => {
                        // Decode: attention uses Q8 cache directly; k, v
                        // here are unused downstream.
                        (k, v)
                    }
                    Ok(()) if self.populate_dual_kv => {
                        // Donor layer: return F-dtype cache for downstream
                        // shared_kv reads.

                        self.kv_cache.current_kv().ok_or_else(|| {
                            crate::tensor::Error::msg("dual-populate: kv_cache empty after append")
                        })?
                    }
                    Ok(()) => {
                        // Multi-token: dequantize full Q8 history to the
                        // model dtype for FA / standard_attention.
                        match self.q8_kv_cache.as_ref().unwrap().dequantize_kv(k.dtype()) {
                            Ok((k_full, v_full)) => (k_full, v_full),
                            Err(e) => {
                                tracing::warn!(
                                    "Q8 dequantize_kv failed (falling back to F-dtype): {e}"
                                );
                                self.q8_kv_cache = None;
                                self.kv_cache.append(&k, &v)?
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Q8 KV append failed (falling back to F-dtype): {e}");
                        self.q8_kv_cache = None;
                        self.kv_cache.append(&k, &v)?
                    }
                }
            } else if self.q4_kv_cache.is_some() {
                // Consolidated Q4 KIVI mode (per-channel K, per-token V).
                // Same shape as Q8 path: decode runs against Q4 cache
                // directly; multi-token paths dequantize for FA /
                // standard_attention.
                let append_res = self.q4_kv_cache.as_mut().unwrap().append(&k, &v);
                match append_res {
                    Ok(()) if seq == 1 => (k, v),
                    Ok(()) => match self.q4_kv_cache.as_ref().unwrap().dequantize_kv(k.dtype()) {
                        Ok((k_full, v_full)) => (k_full, v_full),
                        Err(e) => {
                            tracing::warn!(
                                "Q4 dequantize_kv failed (falling back to F-dtype): {e}"
                            );
                            self.q4_kv_cache = None;
                            self.kv_cache.append(&k, &v)?
                        }
                    },
                    Err(e) => {
                        tracing::warn!("Q4 KV append failed (falling back to F-dtype): {e}");
                        self.q4_kv_cache = None;
                        self.kv_cache.append(&k, &v)?
                    }
                }
            } else if self.kv_f16 {
                // gemma4 F16 KV lever: store K/V at F16 so the growing-context
                // score/output matmuls read half the HBM bytes (the dominant
                // long-context decode cost). Projections/norms/RoPE stay F32;
                // only the cached K/V (and the attention matmuls reading them,
                // via the dtype-agnostic standard_attention) drop to F16  -
                // exactly ollama's F16 KV, well above the Q8 K-norm noise floor.
                let k16 = if k.dtype() == crate::tensor::DType::F16 {
                    k
                } else {
                    k.to_dtype(crate::tensor::DType::F16)?
                };
                let v16 = if v.dtype() == crate::tensor::DType::F16 {
                    v
                } else {
                    v.to_dtype(crate::tensor::DType::F16)?
                };
                self.kv_cache.append(&k16, &v16)?
            } else {
                self.cpu_q8_append(&k, &v, seq, index_pos, is_shared_layer);
                self.kv_cache.append(&k, &v)?
            }
        };
        // CPU build: no quantized KV caches - rope, optional reset, then the
        // plain (or graph-buffer) F-dtype append.
        #[cfg(not(feature = "cuda"))]
        let (k, v) = if let Some((shared_k, shared_v)) = shared_kv {
            (shared_k, shared_v)
        } else {
            if use_graph_ops {
                k = self.apply_rotary_emb_from_buffer(&k)?;
            } else {
                k = self.apply_rotary_emb(&k, index_pos)?;
            }
            if index_pos == 0 {
                self.kv_cache.reset();
            }
            if use_graph_ops {
                if self.kv_cache.k_buffer().is_none() {
                    let (_, _, _) = self.kv_cache.append_padded(&k, &v)?;
                } else {
                    let pos_idx = self
                        .graph_kv_pos
                        .as_ref()
                        .ok_or_else(|| crate::tensor::Error::msg("graph_kv_pos not set"))?;
                    let k_buf = self.kv_cache.k_buffer().unwrap();
                    let v_buf = self.kv_cache.v_buffer().unwrap();
                    k_buf.scatter_set(pos_idx, &k, 2)?;
                    v_buf.scatter_set(pos_idx, &v, 2)?;
                    self.kv_cache.advance_seq_len(1);
                }
                (
                    self.kv_cache.k_buffer().unwrap(),
                    self.kv_cache.v_buffer().unwrap(),
                )
            } else {
                // Populate the CPU Q8 KV store from the NEW token(s)
                // (`k`/`v` are k_new/v_new here, before the F16 append merges
                // them). Cheap (only the new token, no full-cache read); the
                // F16 cache below stays the safe fallback.
                self.cpu_q8_append(&k, &v, seq, index_pos, is_shared_layer);
                self.kv_cache.append(&k, &v)?
            }
        };

        // -- Attention ----------------------------------------------------
        // Q8 decode fast path: single-token attention with the Q8 KV cache.
        // Bypasses the F16 `standard_attention` GQA reshape/matmul in favour
        // of the dedicated Q8_0 x Q8_1 gemv (scores) + Q8_0 gemv (output).
        // Gated on: cache populated, no shared_kv override, seq == 1, not in
        // graph mode, on CUDA. Anything else falls through to the F16 path.
        // Shared layers' own q8_kv_cache is empty (they don't compute K/V),
        // so skip the Q8 decode path here - they fall through to standard
        // attention which uses the donor's K/V from `shared_kv`.
        //
        // For `populate_dual_kv` donor layers (gemma4): the Q8 cache exists
        // for downstream consumers, but the Q8 K-quantization noise breaks
        // the donor's own attention math (gemma4 has tiny K-norm weights
        // that amplify Q8 noise into degenerate decoding). Skip Q8 decode
        // for donors and run the F-dtype standard path instead.
        // Q8 graph-mode decode: separate dispatch from the host-int Q8 path,
        // sits ABOVE the existing try_q8_decode below. Active when graph
        // capture has primed cur_pos_dev. Calls attn_scores_graph /
        // attn_output_graph which read seq_kv from cur_pos_dev[0] and emit
        // scores at the fixed max_seq_padded stride - invariant across
        // captured graph replays. The KV append at line 1463-1477 (graph
        // mode) routes through `q8_kv_cache.append` for this branch (lazy
        // 4 K initial capacity) instead of the F-dtype padded path that
        // would allocate user_context_length x n_kv_heads x head_dim x 2
        // x dtype_bytes per layer.
        //
        // dropped the `use_graph_ops` gate FOR PHI2 ONLY so
        // moondream / plain phi2 (blocked on graph capture by // / ) can still route Q8 KV through the dev_pos kernels  -
        // the only Q8 attn kernels that fit phi2's hd=64 register
        // budget. `cur_pos_dev` is lazily primed when None so non-graph
        // callers don't need to know about graph-mode plumbing.
        //
        // For non-phi2 arches (qwen2/3, gemma4, llama, etc.) the
        // dev_pos path is STILL gated on `use_graph_ops` because it
        // does fixed-width 2048-wide attention work even at short
        // seq_kv - deepcoder regressed 14 points when
        // wired without this guard. Those arches use try_q8_decode
        // (host-int, seq_kv-wide) for non-graph and the dev_pos
        // path only inside a captured graph.
        let phi2_dev_pos = self.flags.parallel_attn && self.flags.layer_norm_with_bias;
        #[cfg(feature = "cuda")]
        let try_q8_graph_decode = self.q8_kv_cache.is_some()
            && seq == 1
            && b == 1
            && q.device().is_cuda()
            && !is_shared_layer
            && !self.populate_dual_kv
            && (use_graph_ops || phi2_dev_pos)
            && self.q8_kv_cache.as_ref().unwrap().current_seq_len()
                < self.q8_kv_cache.as_ref().unwrap().max_seq_padded();
        #[cfg(feature = "cuda")]
        if try_q8_graph_decode {
            // Prime cur_pos_dev - UNLESS the current stream is in graph
            // capture mode. `update_graph_state` does a memcpy_htod from
            // a stack-allocated i32; inside a capture, that gets
            // recorded as a graph memcpy node whose src pointer points
            // at the calling stack frame. After the function returns
            // the stack frame is invalid, and subsequent replays copy
            // garbage into cur_pos_dev -> dev_pos kernels read a wild
            // seq_kv -> logits are stale -> sampler emits "!" tokens.
            //
            // The engine primes cur_pos_dev BEFORE begin_capture for
            // graph mode (via the model-level update_graph_state); the
            // in-capture refresh here is redundant and unsafe. For
            // non-graph mode this dispatch must keep refreshing per
            // layer per token (host current_seq_len advances via
            // append() and the dev pos has to stay in lockstep).
            let in_capture = crate::tensor::cuda_ext::capture_active(&q.device());
            if !in_capture {
                // Fix: see the APQ path branch above. Pass the
                // BEFORE-append value (cur_seq - 1) so the score kernel's
                // `seq_kv = cur_pos_dev + 1` exactly matches the cache's
                // filled length. Otherwise the kernel iterates one slot
                // past the cache and reads uninitialized K/V -> garbage.
                let cur_seq = self.q8_kv_cache.as_ref().unwrap().current_seq_len();
                if let Some(cache_mut) = self.q8_kv_cache.as_mut() {
                    cache_mut
                        .update_graph_state(cur_seq.saturating_sub(1))
                        .map_err(|e| {
                            crate::tensor::Error::msg(format!("Q8 prime cur_pos_dev: {e}"))
                        })?;
                }
            }
            let cache = self.q8_kv_cache.as_ref().unwrap();
            let scale = self
                .attention_scale
                .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt());
            // Fused scale + softmax + V.attn in one kernel - saves 3
            // launches per layer per token vs the prior chain (scores,
            // affine, softmax_last_dim, attn_output_graph -> 4 launches;
            // now scores + fused -> 2 launches). For phi2 / moondream
            // that's 72 fewer launches per token across 24 layers.
            let scores = cache
                .attn_scores_graph(&q, 0)
                .map_err(|e| crate::tensor::Error::msg(format!("Q8 attn_scores_graph: {e}")))?;
            let y_q8 = cache
                .attn_softmax_output_graph(&scores, scale as f32, 0)
                .map_err(|e| {
                    crate::tensor::Error::msg(format!("Q8 attn_softmax_output_graph: {e}"))
                })?;
            let attn_out_dim = self.n_head * self.head_dim;
            let y = y_q8.transpose(1, 2)?.reshape(&[b, seq, attn_out_dim])?;
            return self.attn_output.forward(&y);
        }

        #[cfg(feature = "cuda")]
        let try_q8_decode = self.q8_kv_cache.is_some()
            && seq == 1
            && b == 1
            && !use_graph_ops
            && q.device().is_cuda()
            && !is_shared_layer
            && !self.populate_dual_kv;
        #[cfg(feature = "cuda")]
        if try_q8_decode {
            let cache = self.q8_kv_cache.as_ref().unwrap();
            let scale = self
                .attention_scale
                .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt());

            // Q8 decode: 2-launch chain (scores + softmax+output gemv).
            // The Q4-style fused softmax+output kernel was ported to Q8
            // (commit c1c6648) AND its softmax reduction parallelized
            // (this commit), but bench at deepcoder long still showed
            // -7.1 % vs the unfused baseline -6.0 %. Q8's wider block
            // (32 bytes vs Q4's 18) keeps the fused path memory-bound
            // even with the softmax reduction fixed.
            // Fold the 1/sqrt(head_dim) scale into the kernel write
            // (saves the separate affine launch). Bench: ~64 layers x
            // 10 µs/launch = ~0.64 ms/token = ~5% on deepcoder long.
            let scores = cache
                .attn_scores_scaled(&q, scale as f32)
                .map_err(|e| crate::tensor::Error::msg(format!("Q8 attn_scores_scaled: {e}")))?;
            let probs = crate::tensor::ops::softmax_last_dim(&scores)?;
            let y_q8 = cache
                .attn_output(&probs)
                .map_err(|e| crate::tensor::Error::msg(format!("Q8 attn_output: {e}")))?;

            let attn_out_dim = self.n_head * self.head_dim;
            // seq==1 here (try_q8_decode gate), so y_q8 is [1, n_head, 1, hd]
            // and is contiguous. transpose(1,2) then reshape forces a
            // contiguous copy (the ucopy_f32 seen in decode profiles) whose
            // element order is identical to a direct reshape, because a
            // size-1 seq dim cannot reorder elements. Skip the transpose:
            // the reshape is then a free view, bit-identical output.
            let y = y_q8.reshape(&[b, seq, attn_out_dim])?;
            return self.attn_output.forward(&y);
        }

        // Q4 KIVI decode fast path - analogous to Q8 above.
        #[cfg(feature = "cuda")]
        let try_q4_decode = self.q4_kv_cache.is_some()
            && seq == 1
            && b == 1
            && q.device().is_cuda()
            && !is_shared_layer;
        #[cfg(feature = "cuda")]
        if try_q4_decode {
            let cache = self.q4_kv_cache.as_ref().unwrap();
            let scale = self
                .attention_scale
                .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt());
            // Graph mode: use the device-pos variants that read seq_kv
            // from cur_pos_dev[0] and write/read at the fixed
            // max_seq_padded stride. Falls back to the host-int path
            // when graph capture isn't active OR when current_seq_len
            // would exceed the graph-mode seq cap (positions beyond it
            // would be invisibly masked by the score kernel).
            let q4_graph = use_graph_ops
                && cache.cur_pos_dev_ptr().is_some()
                && cache.current_seq_len() < cache.max_seq_padded();
            // Fused flash split-K decode (one pass, no HBM scores round-trip).
            // OPT-IN (LOKEN_Q4_FUSED=1), default OFF. Measured vs the
            // GRAPH 2-kernel chain that the Q4 decode actually uses (which already
            // parallelises over seq AND kv_head): isolated kernel 21.7 µs vs the
            // graph chain's 30 µs at seq_kv=2700 (≈8 µs/layer), but this is only
            // ~3 % of the qwen3 decode token and shows NO measurable end-to-end
            // gain at <=2.5K - attention is NOT the long-context decode bottleneck
            // there (the graph chain is already efficient). Kept as infrastructure:
            // the chain materializes the [n_q,seq_kv] scores buffer in HBM each
            // token, so the fused path's advantage widens at long context (>=8K),
            // where it should start to pay. Same online-softmax math as the chain
            // (the Q8 GQA path already ships this kernel). GQA hd∈{64,128} only;
            // any kernel Err / shape mismatch falls back to the chain.
            // Expected to pay only at long context (>=8K), where the fused form's per-token
            // advantage widens; not yet measured to win, so nothing selects it.
            let q4_fused_ok = false;
            let scale_f32 = scale as f32;
            let fused = if !q4_fused_ok {
                None
            } else if q4_graph {
                match cache.attn_flash_splitk_decode_graph(&q, scale_f32) {
                    Ok(y) => Some(y),
                    Err(e) => {
                        tracing::warn!("Q4 fused graph decode fallback to chain: {e}");
                        None
                    }
                }
            } else {
                match cache.attn_flash_splitk_decode(&q, scale_f32) {
                    Ok(y) => Some(y),
                    Err(e) => {
                        tracing::warn!("Q4 fused decode fallback to chain: {e}");
                        None
                    }
                }
            };
            let y_q4 = if let Some(y) = fused {
                y
            } else if q4_graph {
                let scores = cache
                    .attn_scores_graph(&q)
                    .map_err(|e| crate::tensor::Error::msg(format!("Q4 attn_scores_graph: {e}")))?;
                let scores = scores.affine(scale as f32, 0.0)?;
                let probs = crate::tensor::ops::softmax_last_dim(&scores)?;
                cache
                    .attn_output_graph(&probs)
                    .map_err(|e| crate::tensor::Error::msg(format!("Q4 attn_output_graph: {e}")))?
            } else {
                // Non-graph fallback chain (fused softmax + V-matmul kernel).
                let scores = cache
                    .attn_scores(&q)
                    .map_err(|e| crate::tensor::Error::msg(format!("Q4 attn_scores: {e}")))?;
                let scores = scores.affine(scale as f32, 0.0)?;
                cache.attn_softmax_output(&scores).map_err(|e| {
                    crate::tensor::Error::msg(format!("Q4 attn_softmax_output: {e}"))
                })?
            };

            let attn_out_dim = self.n_head * self.head_dim;
            let y = y_q4.transpose(1, 2)?.reshape(&[b, seq, attn_out_dim])?;
            return self.attn_output.forward(&y);
        }

        let y = if use_graph_ops && self.padded_mask.is_some() {
            let mask = self.padded_mask.as_ref().unwrap().clone();
            self.padded_standard_attention(&q, k, v, &mask)?
        } else {
            // FA-for-decode is automatically enabled when it's profitable:
            //   - head_dim >= 128: matmul+softmax+matmul triple's launch
            //     cost outweighs FA's F16 cast overhead.
            //   - parallel-attn (phi2) decode regardless of head_dim:
            //     those models have ~10 small ops/layer; the fused FA
            //     wins even with HD=64.
            //
            // The external flash-attn fast path was removed with the old
            // substrate: standard_attention is the verified-parity path
            // (the FA kernel was also non-deterministic at temp=0 outside
            // HD∈{128,256}, and a measured LOSS at HD=64).
            // SPLIT-K HD512 F32 flash-decode for gemma4 GLOBAL layers, gated
            // OFF by default (diagnostic LOKEN_HD512_SPLITK=1). grid=
            // (n_q_heads, nsplit); mask is None here so a zeros additive mask is
            // correct (kernel self-masks t>=seq_kv); falls back to
            // standard_attention on any error.
            //
            // VERDICT (measured clean, gate-on vs gate-off, same
            // prompt): the kernel is CORRECT - bit-identical to cuBLAS
            // standard_attention at 565 KV, coherent (fp-argmax-equivalent) at
            // 1453/1664 KV - but it LOSES on speed and the gap WIDENS with KV:
            //   1453 KV: 97.0 vs cuBLAS 104.5 (-7%)
            //   2409 KV: 80.7 vs cuBLAS 89.2 (-9.5%)
            // Two reasons it can't win: (1) this kernel reads K thread-per-
            // position (stride-HD), i.e. UNCOALESCED - worse the more K it
            // scans; (2) more fundamentally, at decode the ONLY thing flash
            // saves over cuBLAS is the tiny scores HBM roundtrip (N.8.4 B) + 2
            // launches (<1% at these tok/s), while the dominant cost is reading
            // K+V (~80 MB/token) which cuBLAS already does near-peak-bandwidth.
            // Unlike the gpt-oss HD64 flash win (attention-LAUNCH-bound on a
            // small model), gemma4 HD512 decode is bandwidth-bound, so attention
            // fusion is the wrong lever. gemma4 long's real lever is the global
            // layers' UN-quantized F-dtype KV cache (Q8 would halve KV
            // bandwidth) - not the attention math; the real lever (Q8 KV for
            // gemma4's global layers) has landed. The F32 hd512 split-K
            // experiment was a confirmed dead-end (reverted twice) and its
            // env-gated call site is removed (no-env-vars rule). The kernel
            // fused_attn_decode_f32_hd512_splitk stays in inference/kernel/fused/mod.rs as
            // uncalled latent infra should the F32-global path ever resurface.
            //
            // CPU dense single-token decode reads the Q8 KV store (half
            // the KV bytes) when eligible; otherwise the F16 path below. The
            // Q8 store is populated above and falls back transparently.
            match self.cpu_q8_attention(&q, seq, b, mask)? {
                Some(y) => y,
                None => match self.cpu_shared_f16_attention(
                    &q,
                    donor_f16,
                    is_shared_layer,
                    seq,
                    b,
                    mask,
                )? {
                    Some(y) => y,
                    None => self.standard_attention(&q, k, v, mask)?,
                },
            }
        };

        // -- Output projection ---------------------------------------------
        let attn_out_dim = self.n_head * self.head_dim;
        let y = y.transpose(1, 2)?.reshape(&[b, seq, attn_out_dim])?;
        // route attn_output projection through stable
        // buffer for graph capture safety (decode only).
        let raw = self.attn_output.forward(&y)?;
        if seq == 1 {
            if let Some(buf) = self.graph_attn_proj_buffer.as_ref() {
                let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                let raw_d = raw.dim(crate::tensor::D::Minus1).unwrap_or(0);
                if buf_d == raw_d && buf_d > 0 {
                    buf.slice_set(&raw, 0, 0)?;
                    return Ok(buf.clone());
                }
            }
        }
        Ok(raw)
    }

    /// Graph-compatible attention block for decode (seq_len == 1).
    ///
    /// Currently delegates to `forward_attn` for correctness validation while
    /// the graph-compatible body (buffer-based RoPE, padded KV cache,
    /// fixed-shape attention) is being developed. Once validated, the body
    /// will be replaced with an implementation whose tensor shapes and device
    /// pointers are invariant across CUDA graph replays.
    pub(super) fn forward_attn_graph(&mut self, x: &Tensor) -> Result<Tensor> {
        // For Q4/Q8-only paths the F-dtype kv_cache is never populated,
        // so its `current_seq_len()` is 0. Passing 0 here would re-enter
        // the `index_pos == 0` reset branch inside `forward_attn` and
        // wipe the populated quantized cache - read seq_len from the
        // actual storage path instead.
        #[cfg(feature = "cuda")]
        let pos = if let Some(c) = self.q4_kv_cache.as_ref() {
            c.current_seq_len()
        } else if let Some(c) = self.q8_kv_cache.as_ref() {
            c.current_seq_len()
        } else {
            self.kv_cache.current_seq_len()
        };
        #[cfg(not(feature = "cuda"))]
        let pos = self.kv_cache.current_seq_len();
        self.forward_attn(
            x,
            None,
            pos,
            None,
            None,
            #[cfg(feature = "cuda")]
            None,
        )
    }

    /// Padded attention for CUDA graph mode.
    /// K/V are the FULL pre-allocated buffers (fixed dimensions).
    /// mask zeros out positions >= valid_seq_len via -inf.
    ///
    /// GQA fast path (n_rep>1, single-token Q): reshape Q to group heads
    /// per KV head instead of expanding K/V via `repeat_kv`. Saves the
    /// per-replay alloc of two `n_rep*` tensors and the cublas-matmul-
    /// with-strided-output overhead - both of which were hot per nsys
    /// on gemma4:latest's 42-layer 8-head/2-kv attention path.
    pub(super) fn padded_standard_attention(
        &mut self,
        q: &Tensor,
        k: Tensor,
        v: Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let n_rep = self.n_head / self.n_kv_head;
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt());

        if n_rep > 1 && q.dims()[2] == 1 {
            // GQA fast path: Q [b, n_head, 1, hd] -> [b, n_kv_head, n_rep, hd]
            //                K [b, n_kv_head, max_kv, hd] (unchanged)
            //                V [b, n_kv_head, max_kv, hd] (unchanged)
            //   att = Q . K^T  -> [b, n_kv_head, n_rep, max_kv]
            //   att += mask    (mask [1,1,1,MAX] broadcasts to [b, n_kv_head, n_rep, MAX])
            //   probs = softmax(att)
            //   out = probs . V -> [b, n_kv_head, n_rep, hd]
            //   reshape -> [b, n_head, 1, hd]
            let (b, _n_head, _one, hd) = q.dims4()?;

            //  HD=512 graph-mode fast path. Replaces
            // the matmul/softmax/matmul chain (which crashes in captured
            // graph due to cuBLAS GEMM workspace state not being tracked
            // - see project_gemma4_graph_ROOT_CAUSE_2026_05_28 - and is
            // unfixable via flash_attn fallback because that kernel
            // HD=512 is broken on sm_120 - see
            // project_fa_hd512_broken_sm120_2026_05_28) with a single
            // custom kernel call. Kernel validated bit-exact to F32
            // reference at 1e-7 precision (see
            // project_fused_attn_decode_f32_hd512_2026_05_28).
            //
            // Conditions:
            //  - HD=512 (only shape the kernel supports)
            //  - b=1 (single batch)
            //  - F32 Q/K/V/mask (kernel is F32-only)
            //  - kv_cache.seq_kv_dev_ptr() is_some (= graph mode is
            //    being prepared and the engine has called update_seq_kv_dev)
            #[cfg(feature = "cuda")]
            let used_fused_hd512: Option<Tensor> = {
                let seq_kv_dev_opt = self.kv_cache.seq_kv_dev_ptr().cloned();
                if hd == 512
                    && b == 1
                    && q.device().is_cuda()
                    && q.dtype() == crate::tensor::DType::F32
                    && k.dtype() == crate::tensor::DType::F32
                    && v.dtype() == crate::tensor::DType::F32
                    && mask.dtype() == crate::tensor::DType::F32
                    && seq_kv_dev_opt.is_some()
                {
                    // Flatten the 4D mask [1,1,1,max_kv] -> 1D [max_kv]
                    // expected by the kernel.
                    let mask_1d = mask.flatten_all()?;
                    let seq_kv_dev = seq_kv_dev_opt.as_ref().unwrap();
                    match crate::inference::kernel::fused::fused_attn_decode_f32_hd512(
                        q,
                        &k,
                        &v,
                        &mask_1d,
                        seq_kv_dev,
                        n_rep,
                        scale as f32,
                    ) {
                        Ok(out) => Some(out),
                        Err(e) => {
                            tracing::warn!(
                                "fused_attn_decode_f32_hd512 fallback: {e}; using matmul chain"
                            );
                            None
                        }
                    }
                } else {
                    None
                }
            };
            #[cfg(not(feature = "cuda"))]
            let used_fused_hd512: Option<Tensor> = None;
            if let Some(out) = used_fused_hd512 {
                // Kernel output is [1, n_q_heads, 1, HD] - already final shape.
                // Push to alive_tensors so the kernel's fresh-alloc output
                // pointer stays valid across graph replays.
                self.graph_alive_tensors.push(out.clone());
                return Ok(out);
            }

            let q_grouped = q.reshape((b, self.n_kv_head, n_rep, hd))?;
            // Fix: split matmul().affine() so the
            // cuBLAS matmul output stays alive across graph replays.
            // The .affine() consumes the matmul output, then drops it;
            // captured affine kernel references the freed address on
            // replay -> ILLEGAL_ADDRESS.
            let raw_matmul = q_grouped.matmul_t(&k)?;
            let raw_att = raw_matmul.affine(scale as f32, 0.0)?;
            self.graph_alive_tensors.push(raw_matmul);
            let att = if let Some(qk_buf) = self.graph_attn_qk_buffer.as_ref() {
                qk_buf.slice_set(&raw_att, 0, 0)?;
                self.graph_alive_tensors.push(raw_att);
                qk_buf.clone()
            } else {
                raw_att
            };
            // route broadcast_add output through stable buffer.
            let raw_mask_added = att.broadcast_add(mask)?;
            let att = if let Some(buf) = self.graph_attn_mask_added_buffer.as_ref() {
                buf.slice_set(&raw_mask_added, 0, 0)?;
                // ROOT CAUSE FIX: keep raw_mask_added alive across
                // captured graph replays (sync cuMemAlloc + Rust drop would
                // otherwise free the captured kernel's output address).
                self.graph_alive_tensors.push(raw_mask_added);
                buf.clone()
            } else {
                raw_mask_added
            };
            // route softmax output through stable buffer.
            let raw_softmax = crate::tensor::ops::softmax_last_dim(&att)?;
            let att = if let Some(buf) = self.graph_attn_softmax_buffer.as_ref() {
                buf.slice_set(&raw_softmax, 0, 0)?;
                self.graph_alive_tensors.push(raw_softmax);
                buf.clone()
            } else {
                raw_softmax
            };
            // NOTE: do NOT `.contiguous()` V - it's a padded buffer at a
            // fixed device pointer; the reference matmul handles the stride
            // directly. Forcing contiguous would alloc+copy every replay.
            // Verified: contiguous() doesn't fix the HD=512
            // graph crash either, so we keep the cheap strided path.
            let raw_out = att.matmul(&v)?;
            let out = if let Some(out_buf) = self.graph_attn_out_buffer.as_ref() {
                out_buf.slice_set(&raw_out, 0, 0)?;
                self.graph_alive_tensors.push(raw_out);
                out_buf.clone()
            } else {
                raw_out
            };
            return out.reshape((b, self.n_head, 1, hd));
        }

        // Multi-token (prefill) or non-GQA fallback: keep repeat_kv path.
        // For single-token decode (q.dims[2] == 1) under graph capture
        // we ALSO route through the persistent QK/out buffers when
        // available - fresh-alloc matmul outputs baked into the
        // captured graph cause CUDA_ERROR_ILLEGAL_ADDRESS at first
        // replay. The GQA fast path above already does
        // this; this branch was the silent gap that left non-GQA
        // F-dtype models (moondream) un-graph-safe.
        let k = crate::tensor::ops::repeat_kv(k, n_rep)?;
        let v = crate::tensor::ops::repeat_kv(v, n_rep)?;
        // Prefill (multi-token): fuse scale+mask+softmax in ONE pass over the
        // scores. F16 models (the CPU fleet) never hit `softmax_last_dim_masked`'s
        // fast path - it requires F32, so an F16 score tensor falls back to the
        // unfused broadcast-add-mask + widen + softmax (several full-tensor passes,
        // the ~20% the F32 fusion note describes but which F16 never received).
        // `softmax_scaled_masked` has an F16 fast path and folds the scale in too,
        // dropping the separate affine pass. It falls back to the same F32 fused
        // form when its shape/dtype guards don't hold, so it is never worse.
        if q.dims()[2] > 1 {
            let raw_att = q.matmul_t(&k)?;
            let att = crate::tensor::ops::softmax_scaled_masked(&raw_att, scale, mask)?
                .to_dtype(v.dtype())?;
            return att.matmul(&v.contiguous()?);
        }
        let raw_att = q.matmul_t(&k)?.affine(scale as f32, 0.0)?;
        let att = if q.dims()[2] == 1 {
            if let Some(qk_buf) = self.graph_attn_qk_buffer.as_ref() {
                qk_buf.slice_set(&raw_att, 0, 0)?;
                qk_buf.clone()
            } else {
                raw_att
            }
        } else {
            raw_att
        };
        // Fused mask+softmax (CPU-F32): avoids materializing the [B,H,Pq,Pk]
        // att+mask tensor per layer - ~20% of CPU prefill at long prompts. GPU
        // path is bit-identical (fused op falls back to broadcast_add+softmax).
        let att = crate::tensor::ops::softmax_last_dim_masked(&att, mask)?;
        let raw_out = att.matmul(&v.contiguous()?)?;
        let out = if q.dims()[2] == 1 {
            if let Some(out_buf) = self.graph_attn_out_buffer.as_ref() {
                out_buf.slice_set(&raw_out, 0, 0)?;
                out_buf.clone()
            } else {
                raw_out
            }
        } else {
            raw_out
        };
        Ok(out)
    }
}
