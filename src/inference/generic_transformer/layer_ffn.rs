//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

impl GenericTransformerLayer {
    /// Phi2 simple FFN minus the final `ffn_down_bias` broadcast_add.
    /// Returns the post-`ffn_down` projection result before the down-bias
    /// is folded in - callers on the parallel-attention fast path defer
    /// that bias into `fused_phi2_residual_merge`. Equivalent to
    /// `forward_ffn(x) - ffn_down_bias` semantically.
    #[inline]
    pub(super) fn forward_ffn_phi2_pre_bias(&self, x: &Tensor) -> Result<Tensor> {
        let up = self.ffn_up.forward(x)?;
        let activated = match &self.ffn_up_bias {
            Some(b) => {
                #[cfg(feature = "cuda")]
                {
                    crate::inference::kernel::fused::fused_bias_gelu_new(&up, b)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    up.broadcast_add(b)?.gelu()?
                }
            }
            None => up.gelu()?,
        };
        self.ffn_down.forward(&activated)
    }

    /// Phi2 simple FFN minus the final `ffn_down_bias` broadcast_add,
    /// using a pre-quantized Q8_1 x_norm buffer for ffn_up. Saves one
    /// quantize launch vs forward_ffn_phi2_pre_bias (which goes through
    /// QMatMul.forward and re-quantizes x_norm).
    #[cfg(feature = "cuda")]
    #[inline]
    pub(super) fn forward_ffn_phi2_pre_bias_from_q8_1(
        &self,
        q8_1_x_norm: &crate::tensor::cuda_ext::CudaSlice<u8>,
        hidden: usize,
    ) -> Result<Tensor> {
        use crate::tensor::quantized::QStorage;
        // Use the generic plain wrapper (dispatches by dtype - works for
        // Q4_0, Q4_K, Q5_K, Q6_K, Q8_0).
        use crate::inference::quantized_cuda::mvq_plain_any_via_shared_q8_1;

        let ffn_up_arc = self.ffn_up.qtensor().ok_or_else(|| {
            crate::tensor::Error::msg("forward_ffn_phi2_pre_bias_from_q8_1: ffn_up not a QTensor")
        })?;
        let intermediate = ffn_up_arc.shape().dims()[0];
        let dev = self
            .ffn_up_bias
            .as_ref()
            .map(|b| b.device())
            .or_else(|| Some(self.attn_norm.weight().device()))
            .and_then(|d| d.as_cuda_device().ok())
            .ok_or_else(|| {
                crate::tensor::Error::msg(
                    "forward_ffn_phi2_pre_bias_from_q8_1: cannot resolve CUDA device",
                )
            })?;
        let qstor = match ffn_up_arc.storage() {
            QStorage::Cuda(s) => s,
            _ => return self.forward_ffn_phi2_pre_bias_fallback_explain("ffn_up not on CUDA"),
        };
        let up_storage =
            mvq_plain_any_via_shared_q8_1(qstor, q8_1_x_norm, hidden, intermediate, &dev)
                .map_err(|e| crate::tensor::Error::msg(format!("mvq ffn_up: {e}")))?;
        let _ = qstor;
        let up =
            crate::tensor::cuda_ext::tensor_from_cuda_storage(up_storage, (1, 1, intermediate))?;
        let activated = match &self.ffn_up_bias {
            Some(b) => crate::inference::kernel::fused::fused_bias_gelu_new(&up, b)?,
            None => up.gelu()?,
        };
        self.ffn_down.forward(&activated)
    }

    /// Launch the ffn_up mvq on the device's alt CUDA stream so it runs
    /// concurrently with the attn chain on the default stream. Returns
    /// (ffn_up_tensor, alt_stream_event_after_mvq). The caller MUST call
    /// `dev.cuda_stream().wait(&event)` before any op (on the default
    /// stream) that reads `ffn_up_tensor`.
    ///
    /// `event_q8_1_ready` is the dependency from the default stream: the
    /// alt stream waits on it so it doesn't read the q8_1 buffer before
    /// it's been populated.
    #[cfg(feature = "cuda")]
    pub(super) fn start_alt_stream_ffn_up_from_q8_1(
        &self,
        q8_1_x_norm: &crate::tensor::cuda_ext::CudaSlice<u8>,
        hidden: usize,
        event_q8_1_ready: &crate::tensor::cuda_ext::CudaEvent,
    ) -> Result<(Tensor, crate::tensor::cuda_ext::CudaEvent)> {
        use crate::inference::quantized_cuda::mvq_plain_any_via_shared_q8_1_on_stream;
        use crate::tensor::quantized::QStorage;

        let ffn_up_arc = self.ffn_up.qtensor().ok_or_else(|| {
            crate::tensor::Error::msg("start_alt_stream_ffn_up_from_q8_1: ffn_up not a QTensor")
        })?;
        let intermediate = ffn_up_arc.shape().dims()[0];
        let dev = self
            .ffn_up_bias
            .as_ref()
            .map(|b| b.device())
            .or_else(|| Some(self.attn_norm.weight().device()))
            .and_then(|d| d.as_cuda_device().ok())
            .ok_or_else(|| {
                crate::tensor::Error::msg(
                    "start_alt_stream_ffn_up_from_q8_1: cannot resolve CUDA device",
                )
            })?;
        let qstor = match ffn_up_arc.storage() {
            QStorage::Cuda(s) => s,
            _ => {
                return Err(crate::tensor::Error::msg(
                    "start_alt_stream_ffn_up_from_q8_1: ffn_up not on CUDA",
                ))
            }
        };
        let alt_stream = dev
            .alt_cuda_stream()
            .map_err(|e| crate::tensor::Error::msg(format!("alt_cuda_stream: {e}")))?;
        // Alt stream waits for the q8_1 quantize to be done on default stream.
        alt_stream
            .wait(event_q8_1_ready)
            .map_err(|e| crate::tensor::Error::msg(format!("alt_stream.wait: {e}")))?;
        let up_storage = mvq_plain_any_via_shared_q8_1_on_stream(
            qstor,
            q8_1_x_norm,
            hidden,
            intermediate,
            &dev,
            &alt_stream,
        )
        .map_err(|e| crate::tensor::Error::msg(format!("mvq ffn_up on alt stream: {e}")))?;
        let _ = qstor;
        // Record event AFTER the alt-stream mvq finishes so the default
        // stream can synchronise before consuming up_storage.
        let event_after = alt_stream
            .record_event(None)
            .map_err(|e| crate::tensor::Error::msg(format!("alt_stream.record_event: {e}")))?;
        let up =
            crate::tensor::cuda_ext::tensor_from_cuda_storage(up_storage, (1, 1, intermediate))?;
        Ok((up, event_after))
    }

    /// Consume an alt-stream-produced `up` tensor: wait on the alt event,
    /// then run bias_gelu + ffn_down on the default stream.
    #[cfg(feature = "cuda")]
    pub(super) fn finish_ffn_phi2_pre_bias_after_alt(
        &self,
        up: &Tensor,
        alt_event: &crate::tensor::cuda_ext::CudaEvent,
    ) -> Result<Tensor> {
        let dev = up.device().as_cuda_device()?;
        // Default stream waits for the alt stream's ffn_up mvq to finish
        // before we read `up`.
        dev.cuda_stream()
            .wait(alt_event)
            .map_err(|e| crate::tensor::Error::msg(format!("cuda_stream.wait(alt_event): {e}")))?;
        let activated = match &self.ffn_up_bias {
            Some(b) => crate::inference::kernel::fused::fused_bias_gelu_new(up, b)?,
            None => up.gelu()?,
        };
        self.ffn_down.forward(&activated)
    }

    /// Full alt-stream FFN: bias-gelu + quantize + ffn_down mvq, all on
    /// the alt CUDA stream. Returns (ffn_pre_tensor, event_after_ffn_pre).
    /// The caller MUST `default_stream.wait(&event)` before any op that
    /// reads ffn_pre.
    #[cfg(feature = "cuda")]
    pub(super) fn finish_ffn_phi2_pre_bias_after_alt_full(
        &self,
        up: &Tensor,
        prior_alt_event: &crate::tensor::cuda_ext::CudaEvent,
        alt_stream: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
    ) -> Result<(Tensor, crate::tensor::cuda_ext::CudaEvent)> {
        use crate::inference::quantized_cuda::{
            mvq_plain_any_via_shared_q8_1_on_stream, quantize_q8_1_fast_mmvq_f32_on_stream,
        };
        use crate::tensor::quantized::QStorage;

        // The alt stream already has the ffn_up mvq queued; `up` lives on
        // its alloc. The prior event marks ffn_up completion - alt stream
        // doesn't need to wait on its own event; just chain ops.
        let _ = prior_alt_event; // kept for future event-graph reasoning
        let dev = up.device().as_cuda_device()?;
        let ffn_up_bias = match &self.ffn_up_bias {
            Some(b) => b,
            None => crate::tensor::bail!("alt-full FFN requires ffn_up_bias"),
        };
        let activated = crate::inference::kernel::fused::fused_bias_gelu_new_on_stream(
            up,
            ffn_up_bias,
            alt_stream,
        )?;

        // Quantize activated -> Q8_1 on alt_stream, then mvq ffn_down.
        let intermediate = activated.dim(crate::tensor::D::Minus1)?;
        let act_c = activated.contiguous()?;
        let act_sl = crate::tensor::cuda_ext::f32_slice_of(&act_c)?;
        let act_view = act_sl.view()?;
        use crate::tensor::quantized::MATRIX_ROW_PADDING;
        let k_padded = intermediate.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        let q8_1_bytes = (k_padded / 32) * 36;
        let mut q8_1_buf = unsafe { alt_stream.alloc::<u8>(q8_1_bytes) }
            .map_err(|e| crate::tensor::Error::msg(format!("alt-full q8_1 alloc: {e}")))?;
        quantize_q8_1_fast_mmvq_f32_on_stream(
            &act_view,
            &mut q8_1_buf,
            intermediate,
            &dev,
            alt_stream,
        )
        .map_err(|e| crate::tensor::Error::msg(format!("alt-full quantize: {e}")))?;
        drop(act_sl);

        let ffn_down_arc = self
            .ffn_down
            .qtensor()
            .ok_or_else(|| crate::tensor::Error::msg("alt-full FFN: ffn_down not a QTensor"))?;
        let down_rows = ffn_down_arc.shape().dims()[0];
        let qstor = match ffn_down_arc.storage() {
            QStorage::Cuda(s) => s,
            _ => crate::tensor::bail!("alt-full FFN: ffn_down not on CUDA"),
        };
        let down_storage = mvq_plain_any_via_shared_q8_1_on_stream(
            qstor,
            &q8_1_buf,
            intermediate,
            down_rows,
            &dev,
            alt_stream,
        )
        .map_err(|e| crate::tensor::Error::msg(format!("alt-full mvq ffn_down: {e}")))?;
        let _ = qstor;

        let event_after = alt_stream
            .record_event(None)
            .map_err(|e| crate::tensor::Error::msg(format!("alt-full record_event: {e}")))?;
        let ffn_pre =
            crate::tensor::cuda_ext::tensor_from_cuda_storage(down_storage, (1, 1, down_rows))?;
        Ok((ffn_pre, event_after))
    }

    #[cfg(feature = "cuda")]
    fn forward_ffn_phi2_pre_bias_fallback_explain(&self, msg: &str) -> Result<Tensor> {
        crate::tensor::bail!("phi2 ffn fast path bail: {msg}");
    }

    /// A projection that routes through the plain IMMA prefill GEMM
    #[inline]
    pub(super) fn proj(&self, qm: &QMatMul, x: &Tensor) -> Result<Tensor> {
        qm.forward(x)
    }

    #[inline]
    pub(super) fn forward_ffn(&mut self, x: &Tensor) -> Result<Tensor> {
        // Phi2 simple FFN: y = down(GELU(up(x) + up_bias)) + down_bias.
        // No gate. Uses tanh-approximated GELU (`gelu_new` in HF) - same
        // approximation as phi2's reference implementation.
        if self.flags.is_phi2_simple_ffn {
            let mut out = self.forward_ffn_phi2_pre_bias(x)?;
            if let Some(b) = &self.ffn_down_bias {
                out = out.broadcast_add(b)?;
            }
            return Ok(out);
        }
        // Fused path: ffn_up holds [gate || up] concatenated along the
        // output dim. Triggered by Phi3 (native GGUF fused) AND by our
        // load-time fusion for qwen3/deepcoder (ffn_gate set to None).
        if self.flags.fused_ffn_gate_up || self.ffn_gate.is_none() {
            let i = self.flags.intermediate_size;
            let raw_up = self.ffn_up.forward(x)?;
            // route ffn_up matmul output through
            // stable buffer when graph mode is in effect (gemma4 etc).
            // The fresh-alloc from QMatMul.forward has CAPTURE-time
            // addr baked into captured graph; goes stale on replay.
            // Decode-only (seq=1) since prefill shape varies.
            let up_states = if x.dim(1).map(|d| d == 1).unwrap_or(false) {
                if let Some(buf) = self.graph_ffn_up_concat_buffer.as_ref() {
                    buf.slice_set(&raw_up, 0, 0)?;
                    buf.clone()
                } else {
                    raw_up
                }
            } else {
                raw_up
            };
            // GELU path: fused_split_gelu_mul reads the packed [B, T, 2N]
            // matmul output directly and emits [B, T, N] in one kernel  -
            // no `narrow -> contiguous -> narrow -> contiguous -> fused_gelu_mul`
            // chain (the two narrow-then-contiguous copies were ~2
            // launches per layer of pure data-shuffle).
            #[cfg(feature = "cuda")]
            let raw_out = if up_states.device().runs_as_card()
                && up_states.dtype() == crate::tensor::DType::F32
                && up_states
                    .dim(crate::tensor::D::Minus1)
                    .map(|d| d == 2 * i)
                    .unwrap_or(false)
            {
                if self.flags.use_gelu {
                    crate::inference::kernel::fused::fused_split_gelu_mul(&up_states)?
                } else {
                    crate::inference::kernel::fused::fused_split_silu_mul(&up_states)?
                }
            } else if !self.flags.use_gelu {
                // CPU (non-cuda device): read the packed [gate | up] projection
                // once and emit silu(gate)*up in one vectorized pass, skipping
                // the host round-trip + scalar exp + intermediate tensor of the
                // separate silu-then-multiply. Off the fast path (strided/odd)
                // fall back to the split-then-op chain.
                if let Some(o) = crate::tensor::ops::split_silu_mul_f32(&up_states)? {
                    o
                } else {
                    let gate = up_states.narrow(crate::tensor::D::Minus1, 0, i)?;
                    let up = up_states.narrow(crate::tensor::D::Minus1, i, i)?;
                    crate::inference::kernel::fused::fused_silu_mul(&gate, &up)?
                }
            } else {
                let gate = up_states.narrow(crate::tensor::D::Minus1, 0, i)?;
                let up = up_states.narrow(crate::tensor::D::Minus1, i, i)?;
                crate::inference::kernel::fused::fused_gelu_mul(&gate, &up)?
            };
            #[cfg(not(feature = "cuda"))]
            let raw_out = {
                // silu.mul: read the packed [gate | up] projection once and emit
                // the activated result in a single vectorized pass, skipping the
                // host round-trip + scalar exp + intermediate tensor of the
                // separate silu-then-multiply. Off the fast path (gelu, non-F32,
                // strided) fall back to the split-then-op chain.
                if !self.flags.use_gelu {
                    if let Some(o) = crate::tensor::ops::split_silu_mul_f32(&up_states)? {
                        o
                    } else {
                        let gate = up_states.narrow(crate::tensor::D::Minus1, 0, i)?;
                        let up = up_states.narrow(crate::tensor::D::Minus1, i, i)?;
                        (up * crate::tensor::ops::silu(&gate)?)?
                    }
                } else {
                    let gate = up_states.narrow(crate::tensor::D::Minus1, 0, i)?;
                    let up = up_states.narrow(crate::tensor::D::Minus1, i, i)?;
                    gate.gelu()?.mul(&up)?
                }
            };
            // route split-gelu/silu_mul output through
            // stable buffer for graph capture.
            let out = if x.dim(1).map(|d| d == 1).unwrap_or(false) {
                if let Some(buf) = self.graph_ffn_activated_buffer.as_ref() {
                    if buf
                        .dim(crate::tensor::D::Minus1)
                        .map(|d| d == i)
                        .unwrap_or(false)
                    {
                        buf.slice_set(&raw_out, 0, 0)?;
                        buf.clone()
                    } else {
                        raw_out
                    }
                } else {
                    raw_out
                }
            } else {
                raw_out
            };
            let raw_down = self.ffn_down.forward(&out)?;
            // route ffn_down output through stable buffer.
            if x.dim(1).map(|d| d == 1).unwrap_or(false) {
                if let Some(buf) = self.graph_ffn_down_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_down.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_down, 0, 0)?;
                        return Ok(buf.clone());
                    }
                }
            }
            Ok(raw_down)
        } else {
            // Dense Q4_K IMMA M=8 silu*mul fast path: when gate+up are
            // both Q4_K QTensors and K % 256 == 0 and input is F32 multi-
            // token (prefill), fuse the three steps (gate matmul + up
            // matmul + silu.mul) into a single IMMA M=8 kernel.
            #[cfg(feature = "cuda")]
            let activated = {
                use crate::tensor::quantized::GgmlDType;
                let use_gelu = self.flags.use_gelu;
                let seq_len = x.dim(1).unwrap_or(1);
                let is_big_multitok = seq_len >= 64;
                let on_cuda = x.device().is_cuda();
                let candidate_qt = if is_big_multitok && on_cuda {
                    let gq = self.ffn_gate.as_ref().unwrap().qtensor();
                    let uq = self.ffn_up.qtensor();
                    //  REFUTED for gate/up too: the default fused path here would
                    // route Q5_K through dense_q5k_imma_m8 (m8 IMMA), but the kernel
                    // A/B shows QMatMul MMQ is 6-8x faster at prefill. Keep Q4_K only
                    // (no fleet models use it) - do NOT add Q5_K (proven regression).
                    match (gq, uq) {
                        (Some(gqt), Some(uqt))
                            if gqt.dtype() == GgmlDType::Q4K
                                && uqt.dtype() == GgmlDType::Q4K
                                && gqt
                                    .shape()
                                    .dims()
                                    .last()
                                    .map(|k| k % 256 == 0)
                                    .unwrap_or(false)
                                && gqt.shape().dims() == uqt.shape().dims() =>
                        {
                            Some((gqt.clone(), uqt.clone()))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                // Same Ampere floor as the MoE path: no code below sm_80, so the dense
                // projection stays on the generic quantised matmul there.
                let candidate_qt = candidate_qt.filter(|_| {
                    x.device()
                        .as_cuda_device()
                        .is_ok_and(|d| d.has_ampere_tensor_cores())
                });
                if let Some((gqt, uqt)) = candidate_qt {
                    // Flatten x to [size_m, K] F32 for the IMMA kernel.
                    let (b, seq, k_in) = x.dims3()?;
                    let x_flat = x.reshape((x.elem_count() / k_in, k_in))?;
                    let x_f32 = if x_flat.dtype() == crate::tensor::DType::F32 {
                        x_flat
                    } else {
                        x_flat.to_dtype(crate::tensor::DType::F32)?
                    };
                    let out = if use_gelu {
                        crate::inference::moe_cuda::dense_q4k_imma_m8_gelu_mul(&x_f32, &gqt, &uqt)?
                    } else {
                        crate::inference::moe_cuda::dense_q4k_imma_m8_silu_mul(&x_f32, &gqt, &uqt)?
                    };
                    // Restore [b, seq, N] shape; cast back to input dtype.
                    let out = out.reshape((b, seq, out.dim(1)?))?;
                    if out.dtype() != x.dtype() {
                        out.to_dtype(x.dtype())?
                    } else {
                        out
                    }
                } else {
                    // Try the fused (up + gate + silu)
                    // single-kernel MMVQ path when:
                    //   - seq == 1 (decode, not prefill)
                    //   - on CUDA
                    //   - activation is SiLU (use_gelu = false)
                    //   - both weights are Q4_K QTensors
                    //   - input is F32
                    // Returns Ok(None) on any mismatch, falling through to
                    // the unfused 3-launch chain.
                    //
                    // Measured: the substrate kernel
                    // commit 651de9e1 shipped a BF16-output variant of
                    // the fused (gate+up+silu) Q4_K kernel. Engine path
                    // widened to BF16 and tested on deepcoder long  -
                    // regressed 78.3->76.6 tok/s (-2.2 %). The BF16
                    // input -> Q8_1 quantize step is measurably slower
                    // than F32 -> Q8_1, and the downstream cast that was
                    // supposed to eat the launch save turned out to be
                    // absent (the reference broadcast_add handles mixed
                    // dtypes without explicit cast). Net negative.
                    // Engine gate kept F32-only; BF16 kernel infrastructure
                    // remains in tree for future BF16
                    // attempts that might compose differently.
                    let fused_si = if on_cuda && !use_gelu && x.dim(1).unwrap_or(0) == 1 {
                        use crate::tensor::quantized::GgmlDType;
                        let gq = self.ffn_gate.as_ref().unwrap().qtensor();
                        let uq = self.ffn_up.qtensor();
                        match (gq, uq) {
                            (Some(gqt), Some(uqt))
                                if gqt.dtype() == GgmlDType::Q4K
                                    && uqt.dtype() == GgmlDType::Q4K
                                    && x.dtype() == crate::tensor::DType::F32 =>
                            {
                                use crate::tensor::quantized::QStorage;
                                let g_stor = gqt.storage();
                                let u_stor = uqt.storage();
                                match (g_stor, u_stor) {
                                    (QStorage::Cuda(gcuda), QStorage::Cuda(ucuda)) => {
                                        let x_storage_layout = x.storage_and_layout();
                                        let (x_storage, x_layout) =
                                            (&*x_storage_layout.0, x_storage_layout.1);
                                        if let crate::tensor::StorageView::Cuda(rhs) = x_storage {
                                            let shape =
                                                crate::tensor::Shape::from_dims(uqt.shape().dims());
                                            crate::tensor::quantized::fast_mmvq::try_fused_silu(
                                                ucuda, gcuda, &shape, rhs, x_layout,
                                            ).ok().flatten()
                                            .and_then(|(storage, out_shape)| {
                                                crate::tensor::cuda_ext::tensor_from_cuda_storage(
                                                    storage, out_shape,
                                                ).ok()
                                            })
                                        } else {
                                            None
                                        }
                                    }
                                    _ => None,
                                }
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    // GELU sibling of try_fused_silu (upstream commit
                    // 651de9e1 / 886da04 + this iteration's launch_..._fused_gelu).
                    // For gemma4 dense decode: gate + up + gelu_mul collapses to
                    // a single fused matmul kernel. Pre-casts BF16->F32 because
                    // the kernel's F32 input + quantize_q8_1_f32 is measurably
                    // faster than the BF16 path (per Phase C measurements).
                    //
                    // Launch accounting (gemma4:latest 30 layers):
                    //   unfused: 2 (gate q+mvq) + 2 (up) + 1 (gelu_mul) + 2 (down)
                    //          = 7 / layer
                    //   fused:   1 (cast) + 2 (fused q+mvq) + 2 (down) = 5 / layer
                    //   save:    2 launches/layer x 30 = 60 launches/token
                    //          ≈ 0.6 ms/token ≈ 4 % at gemma4:latest (147 tok/s).
                    let fused_ge = if fused_si.is_none()
                        && use_gelu
                        && on_cuda
                        && x.dim(1).unwrap_or(0) == 1
                    {
                        use crate::tensor::quantized::GgmlDType;
                        let gq = self.ffn_gate.as_ref().unwrap().qtensor();
                        let uq = self.ffn_up.qtensor();
                        match (gq, uq) {
                            (Some(gqt), Some(uqt))
                                if gqt.dtype() == GgmlDType::Q4K
                                    && uqt.dtype() == GgmlDType::Q4K =>
                            {
                                use crate::tensor::quantized::QStorage;
                                let g_stor = gqt.storage();
                                let u_stor = uqt.storage();
                                match (g_stor, u_stor) {
                                    (QStorage::Cuda(gcuda), QStorage::Cuda(ucuda)) => {
                                        // try_fused_gelu accepts F32 + BF16 natively
                                        // BUT BF16 quantize is measurably slower than
                                        // F32 quantize (Phase C measurement). For
                                        // gemma4:latest medium, the cast-then-F32 path
                                        // benches -1.9 % vs BF16-native -2.6 %.
                                        // Keep upfront cast for now.
                                        let x_f32_owned;
                                        let x_view = if x.dtype() == crate::tensor::DType::F32 {
                                            x
                                        } else {
                                            x_f32_owned =
                                                x.to_dtype(crate::tensor::DType::F32).ok();
                                            match &x_f32_owned {
                                                Some(t) => t,
                                                None => return self.ffn_down.forward(&{
                                                    let gate = self
                                                        .ffn_gate
                                                        .as_ref()
                                                        .unwrap()
                                                        .forward(x)?;
                                                    let up = self.ffn_up.forward(x)?;
                                                    crate::inference::kernel::fused::fused_gelu_mul(
                                                        &gate, &up,
                                                    )?
                                                }),
                                            }
                                        };
                                        let x_storage_layout = x_view.storage_and_layout();
                                        let (x_storage, x_layout) =
                                            (&*x_storage_layout.0, x_storage_layout.1);
                                        if let crate::tensor::StorageView::Cuda(rhs) = x_storage {
                                            let shape =
                                                crate::tensor::Shape::from_dims(uqt.shape().dims());
                                            crate::tensor::quantized::fast_mmvq::try_fused_gelu(
                                                ucuda, gcuda, &shape, rhs, x_layout,
                                            ).ok().flatten()
                                            .and_then(|(storage, out_shape)| {
                                                crate::tensor::cuda_ext::tensor_from_cuda_storage(
                                                    storage, out_shape,
                                                ).ok()
                                            })
                                        } else {
                                            None
                                        }
                                    }
                                    _ => None,
                                }
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    // Path B: route activated output through stable
                    // buffer for gemma4 graph-mode safety. The fused kernels
                    // and the unfused split chain all produce fresh-alloc
                    // outputs that downstream ffn_down.forward reads. Without
                    // routing, captured ffn_down.forward sees stale pointers.
                    let raw_act = if let Some(act) = fused_si {
                        let n = act.dim(crate::tensor::D::Minus1)?;
                        act.reshape((1, 1, n))?
                    } else if let Some(act) = fused_ge {
                        let n = act.dim(crate::tensor::D::Minus1)?;
                        act.reshape((1, 1, n))?
                    } else {
                        let raw_gate = self.ffn_gate.as_ref().unwrap().forward(x)?;
                        let gate = if let Some(buf) = self.graph_ffn_gate_out.as_ref() {
                            buf.slice_set(&raw_gate, 0, 0)?;
                            self.graph_alive_tensors.push(raw_gate);
                            buf.clone()
                        } else {
                            raw_gate
                        };
                        let raw_up = self.ffn_up.forward(x)?;
                        let up = if let Some(buf) = self.graph_ffn_up_buffer.as_ref() {
                            let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                            let raw_d = raw_up.dim(crate::tensor::D::Minus1).unwrap_or(0);
                            if buf_d == raw_d && buf_d > 0 {
                                buf.slice_set(&raw_up, 0, 0)?;
                                self.graph_alive_tensors.push(raw_up);
                                buf.clone()
                            } else {
                                raw_up
                            }
                        } else {
                            raw_up
                        };
                        if use_gelu {
                            crate::inference::kernel::fused::fused_gelu_mul(&gate, &up)?
                        } else {
                            crate::inference::kernel::fused::fused_silu_mul(&gate, &up)?
                        }
                    };
                    if let Some(buf) = self.graph_ffn_activated_buffer.as_ref() {
                        let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        let raw_d = raw_act.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        if buf_d == raw_d && buf_d > 0 {
                            buf.slice_set(&raw_act, 0, 0)?;
                            self.graph_alive_tensors.push(raw_act);
                            buf.clone()
                        } else {
                            raw_act
                        }
                    } else {
                        raw_act
                    }
                }
            };
            #[cfg(not(feature = "cuda"))]
            let activated = {
                let gate = self.ffn_gate.as_ref().unwrap().forward(x)?;
                let up = self.ffn_up.forward(x)?;
                if self.flags.use_gelu {
                    gate.gelu()?.mul(&up)?
                } else {
                    crate::tensor::ops::silu(&gate)?.mul(&up)?
                }
            };
            // Fix: route raw_down through stable
            // buffer + alive_tensors push, mirroring the fused_ffn_gate_up
            // path at line 2934+. Without this, the captured cuBLAS GEMM
            // output pointer goes stale after the function returns ->
            // downstream fused_rmsnorm_then_add reads a freed alloc =
            // ILLEGAL_ADDRESS. Decode-only (seq=1).
            let raw_down = self.ffn_down.forward(&activated)?;
            #[cfg(feature = "cuda")]
            if x.dim(1).map(|d| d == 1).unwrap_or(false) {
                if let Some(buf) = self.graph_ffn_down_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_down.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_down, 0, 0)?;
                        self.graph_alive_tensors.push(raw_down);
                        return Ok(buf.clone());
                    }
                }
            }
            Ok(raw_down)
        }
    }
}
