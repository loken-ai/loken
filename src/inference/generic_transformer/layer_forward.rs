//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

impl GenericTransformerLayer {
    /// Step 1 of split forward: QKV projection + RoPE + KV cache write.
    /// Runs OUTSIDE the CUDA graph (position changes each token).
    /// Returns the RoPE'd Q tensor for use in compute_from_kv.
    pub fn prepare_kv(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let x_norm = self.attn_norm.forward(x)?;

        // QKV projection
        let (mut q, mut k, v) = if self.flags.fused_qkv || self.attn_q.is_none() {
            let mut qkv = self.proj(self.attn_qkv.as_ref().unwrap(), &x_norm)?; // IMMA prefill
            let q_dim = self.n_head * self.head_dim;
            let kv_dim = self.n_kv_head * self.head_dim;
            if self.flags.has_qkv_bias {
                if let Some(bqkv) = self.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(bqkv)?;
                }
            }
            let mut q = qkv.narrow(crate::tensor::D::Minus1, 0, q_dim)?;
            let mut k = qkv.narrow(crate::tensor::D::Minus1, q_dim, kv_dim)?;
            let mut v = qkv.narrow(crate::tensor::D::Minus1, q_dim + kv_dim, kv_dim)?;
            if self.flags.has_qkv_bias && self.attn_qkv_bias.is_none() {
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
            let mut q = self.proj(self.attn_q.as_ref().unwrap(), &x_norm)?; // IMMA prefill
            let mut k = self.proj(self.attn_k.as_ref().unwrap(), &x_norm)?;
            let mut v = self.proj(self.attn_v.as_ref().unwrap(), &x_norm)?;
            // Separate-projection QKV biases (qwen2 family). The fused-qkv
            // branch above applies these; this branch MUST too. Missing it
            // left split-graph-path Q/K/V un-biased -> garbage attention. Only
            // bit qwen2.5:0.5b (separate q/k/v WITH bias, small enough to fit a
            // single GPU -> graph/split path; larger qwen2 spill to CPU/multi-GPU
            // and take the normal path which already biases). Mirrors
            // forward_attn_inner's separate-branch bias add.
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
        };

        // Reshape + RoPE for BOTH Q and K (position-dependent - outside graph).
        // OLMo2 full-dim QK-norm: normalise the flat q/k over the whole projection
        // BEFORE the head split (vs Gemma3/Qwen3 per-head after the reshape).
        let qk_norm_full = self.flags.has_qk_norm
            && self
                .attn_q_norm
                .as_ref()
                .map(|n| {
                    n.weight()
                        .dims1()
                        .map(|d| d != self.head_dim)
                        .unwrap_or(false)
                })
                .unwrap_or(false);
        if qk_norm_full {
            if let (Some(q_norm), Some(k_norm)) = (&self.attn_q_norm, &self.attn_k_norm) {
                q = q_norm.forward(&q.contiguous()?)?;
                k = k_norm.forward(&k.contiguous()?)?;
            }
        }
        q = q
            .reshape((b, seq, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        // FIX: apply q_norm/k_norm BEFORE RoPE.
        // gemma4 has qk_norm; the normal forward_attn_inner standard path
        // does `q = q_norm.forward(&q.contiguous()?)` here. prepare_kv was
        // MISSING it -> split-path Q/K were un-normalized -> wrong attention.
        // (PER-HEAD; skipped when OLMo2 full-dim was applied pre-reshape above.)
        if !qk_norm_full {
            if let (Some(q_norm), Some(k_norm)) = (&self.attn_q_norm, &self.attn_k_norm) {
                q = q_norm.forward(&q.contiguous()?)?;
                k = k_norm.forward(&k.contiguous()?)?;
            }
        }
        // FIX: gemma4 also RMS-norms V with a
        // unit-weight (no learnable scale). forward_attn_inner does this at
        // line 1895-1903; prepare_kv was MISSING it. Donor layers' V must be
        // normed so shared layers reading donor KV get correct values.
        if self.flags.has_qk_norm {
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
        q = self.apply_rotary_emb(&q, index_pos)?;
        k = self.apply_rotary_emb(&k, index_pos)?;

        if index_pos == 0 {
            self.kv_cache.reset();
        }

        // Write to KV cache (NOT captured in graph)
        let (_k_full, _v_full, valid_len) = self.kv_cache.append_padded(&k, &v)?;

        //  update stable device-side seq_kv counter
        // (mirror of Q8 cur_pos_dev). `current_seq_len - 1` matches the
        // convention used by fused_attn_decode_f32_hd512 (kernel adds 1).
        // Runs OUTSIDE capture; the captured kernel reads the updated
        // value via this stable pointer each replay.
        // Skipped silently if device isn't CUDA - kernel won't fire then.
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            if let Ok(cuda_dev) = q.device().as_cuda_device() {
                let cur = self.kv_cache.current_seq_len();
                let pos = cur.saturating_sub(1);
                if let Err(e) = self.kv_cache.update_seq_kv_dev(pos, &cuda_dev) {
                    tracing::warn!("update_seq_kv_dev failed: {e}");
                }
            }
        }

        // Update padded mask in-place
        let max_kv = self.kv_cache.max_seq_len_padded();
        if self.padded_mask.is_none() {
            let mut mask_data = vec![f32::NEG_INFINITY; max_kv];
            // First valid_len entries: visible (mask = 0); rest stays -INF.
            mask_data[..valid_len].fill(0.0);
            self.padded_mask =
                Some(Tensor::new(&mask_data[..], &q.device())?.reshape((1, 1, 1, max_kv))?);
        } else {
            let mask = self.padded_mask.as_ref().unwrap();
            let zero = Tensor::zeros_on((1, 1, 1, 1), crate::tensor::DType::F32, &mask.device())?;
            mask.slice_set(&zero, 3, valid_len - 1)?;
        }

        // Write Q into pre-allocated buffer (fixed pointer for graph replay)
        if self.graph_q_buffer.is_none() {
            self.graph_q_buffer = Some(q.clone());
        } else {
            // Copy Q data into the existing buffer at the same device pointer
            let buf = self.graph_q_buffer.as_ref().unwrap();
            buf.slice_set(&q, 0, 0)?; // overwrite entire buffer
        }

        Ok(self.graph_q_buffer.as_ref().unwrap().clone())
    }

    /// Step 2 of split forward: attention (from pre-computed Q buffer + KV buffer) + FFN.
    /// Runs INSIDE the CUDA graph (all dimensions fixed).
    /// Reads Q from graph_q_buffer (updated by prepare_kv at same device pointer).
    /// Reads K/V from kv_cache buffers (updated by prepare_kv at same pointers).
    /// Reads mask from padded_mask (updated by prepare_kv at same pointer).
    pub fn compute_from_kv(&mut self, x: &Tensor, _q_unused: &Tensor) -> Result<Tensor> {
        self.compute_from_kv_shared(x, _q_unused, None)
    }

    /// Like compute_from_kv but accepts donor K/V buffers for shared-KV
    /// layers (gemma4 8B: shared_kv_layers=18). When `shared_kv` is Some, the
    /// attention reads the donor layer's padded K/V instead of this layer's
    /// own cache - mirroring forward_attn_inner's shared_kv path.
    /// FIX: the split path previously ignored KV sharing,
    /// so the 18 shared layers attended to their own (wrong) K/V -> garbage.
    pub fn compute_from_kv_shared(
        &mut self,
        x: &Tensor,
        _q_unused: &Tensor,
        shared_kv: Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // All inputs are at fixed device pointers - graph reads current data.
        // Extended to match
        // forward_graph's feature set: parallel_attn (phi2), post_attn_norm
        // (gemma4), residual_scale (granite). Q4/Q8 fast paths NOT applied
        // here - those arches stay on the legacy forward_from_hidden path
        // because their KV append already uses graph-safe persistent
        // buffers and doesn't need the split.
        let rs = self.residual_scale;
        let residual = x;

        let q = self.graph_q_buffer.as_ref().unwrap().clone();
        let (k_full, v_full) = match &shared_kv {
            Some((k, v)) => (k.clone(), v.clone()),
            None => (
                self.kv_cache.k_buffer().unwrap(),
                self.kv_cache.v_buffer().unwrap(),
            ),
        };
        let mask = self.padded_mask.as_ref().unwrap().clone();

        let (b, _, _, _d) = q.dims4()?;
        let y = self.padded_standard_attention(&q, k_full, v_full, &mask)?;

        let attn_out_dim = self.n_head * self.head_dim;
        let y = y.transpose(1, 2)?.reshape(&[b, 1, attn_out_dim])?;
        //  push the reshaped y to alive_tensors.
        // The transpose+reshape on a non-contiguous input produces a
        // fresh-alloc which Wo cuBLAS GEMM captures. Without this push,
        // the captured Wo reads from a freed address on replay.
        self.graph_alive_tensors.push(y.clone());
        let raw_attn_out = self.proj(&self.attn_output, &y)?; // IMMA prefill (O projection)
        let mut attn_out = if let Some(buf) = self.graph_attn_proj_buffer.as_ref() {
            buf.slice_set(&raw_attn_out, 0, 0)?;
            self.graph_alive_tensors.push(raw_attn_out);
            buf.clone()
        } else {
            raw_attn_out
        };

        // -- Phi2 / GPT-NeoX parallel-attention path -------------------
        // Mirrors forward_graph parallel_attn branch (lines 2225+).
        if self.flags.parallel_attn {
            #[cfg(feature = "cuda")]
            let phi2_fast = self.flags.is_phi2_simple_ffn
                && rs.is_none()
                && self.attn_output_bias.is_some()
                && self.ffn_down_bias.is_some()
                && attn_out.device().is_cuda()
                && attn_out.dtype() == crate::tensor::DType::F32;
            #[cfg(not(feature = "cuda"))]
            let phi2_fast = false;
            if !phi2_fast {
                if let Some(b) = &self.attn_output_bias {
                    attn_out = attn_out.broadcast_add(b)?;
                }
            }
            if let Some(s) = rs {
                attn_out = (attn_out * s)?;
            }

            // x_norm for parallel FFN = attn_norm output, which was
            // computed and then consumed in prepare_kv. Need to re-derive
            // - graph_q_buffer is the *projected* Q, not x_norm. For
            // parallel_attn we use the residual `x` (graph_hidden_buffer
            // input) re-normed here. attn_norm is graph-safe (LayerNorm).
            let x_norm = self.attn_norm.forward(residual)?;

            #[cfg(feature = "cuda")]
            let out = if phi2_fast {
                // Route phi2 FFN intermediates through stable buffers
                // for graph safety. forward_ffn_phi2_pre_bias
                // calls ffn_up.forward (fresh) + fused_bias_gelu_new
                // (fresh) + ffn_down.forward (fresh). We slice_set each
                // into a per-layer persistent buffer before the next
                // op reads it.
                let up_raw = self.ffn_up.forward(&x_norm)?;
                let up = if let Some(buf) = self.graph_ffn_up_buffer.as_ref() {
                    buf.slice_set(&up_raw, 0, 0)?;
                    buf.clone()
                } else {
                    up_raw
                };
                let activated_raw = match &self.ffn_up_bias {
                    Some(b) => crate::inference::kernel::fused::fused_bias_gelu_new(&up, b)?,
                    None => up.gelu()?,
                };
                let activated = if let Some(buf) = self.graph_ffn_activated_buffer.as_ref() {
                    buf.slice_set(&activated_raw, 0, 0)?;
                    buf.clone()
                } else {
                    activated_raw
                };
                let down_raw = self.ffn_down.forward(&activated)?;
                let ffn_pre = if let Some(buf) = self.graph_ffn_down_buffer.as_ref() {
                    buf.slice_set(&down_raw, 0, 0)?;
                    buf.clone()
                } else {
                    down_raw
                };
                let attn_bias = self.attn_output_bias.as_ref().unwrap();
                let ffn_bias = self.ffn_down_bias.as_ref().unwrap();
                let merge_raw = crate::inference::kernel::fused::fused_phi2_residual_merge(
                    residual, &attn_out, attn_bias, &ffn_pre, ffn_bias,
                )?;
                if let Some(buf) = self.graph_phi2_merge_buffer.as_ref() {
                    buf.slice_set(&merge_raw, 0, 0)?;
                    buf.clone()
                } else {
                    merge_raw
                }
            } else {
                let mut ffn_out = self.forward_ffn(&x_norm)?;
                if let Some(s) = rs {
                    ffn_out = (ffn_out * s)?;
                }
                crate::inference::kernel::fused::fused_add_three(residual, &attn_out, &ffn_out)?
            };
            #[cfg(not(feature = "cuda"))]
            let out = {
                let mut ffn_out = self.forward_ffn(&x_norm)?;
                if let Some(s) = rs {
                    ffn_out = (ffn_out * s)?;
                }
                ((residual + &attn_out)? + ffn_out)?
            };
            return Ok(out);
        }

        // -- Serial path (Llama/Gemma/Qwen) ----------------------------
        if let Some(s) = rs {
            let raw = (attn_out * s)?;
            attn_out = if let Some(buf) = self.graph_attn_scaled_buffer.as_ref() {
                buf.slice_set(&raw, 0, 0)?;
                self.graph_alive_tensors.push(raw);
                buf.clone()
            } else {
                raw
            };
        }

        // Pattern A: fuse post_norm+residual into a single
        // kernel (eliminates the broadcast_add that read fresh-alloc
        // post_norm output). Routes the fused output through
        // graph_post_attn_norm_buffer for downstream stability.
        let raw_x = if let Some(post_norm) = &self.post_attn_norm {
            #[cfg(feature = "cuda")]
            let fused = if attn_out.device().runs_as_card()
                && attn_out.dtype() == crate::tensor::DType::F32
                && residual.dtype() == crate::tensor::DType::F32
                && attn_out.shape() == residual.shape()
            {
                crate::inference::kernel::fused::fused_rmsnorm_then_add(
                    &attn_out,
                    post_norm.weight(),
                    residual,
                    self.ffn_norm_eps,
                )
                .ok()
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let fused: Option<Tensor> = None;
            match fused {
                Some(t) => t,
                None => (post_norm.forward(&attn_out)? + residual)?,
            }
        } else {
            (attn_out + residual)?
        };
        let x = if let Some(buf) = self.graph_post_attn_norm_buffer.as_ref() {
            buf.slice_set(&raw_x, 0, 0)?;
            self.graph_alive_tensors.push(raw_x);
            buf.clone()
        } else {
            raw_x
        };
        let raw_x_norm = self
            .ffn_norm
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("ffn_norm missing"))?
            .forward(&x)?;
        let x_norm = if let Some(buf) = self.graph_x_norm_ffn_buffer.as_ref() {
            buf.slice_set(&raw_x_norm, 0, 0)?;
            self.graph_alive_tensors.push(raw_x_norm);
            buf.clone()
        } else {
            raw_x_norm
        };
        let residual_ffn = &x;
        let mut ffn_out = self.forward_ffn(&x_norm)?;
        if let Some(s) = rs {
            let raw = (ffn_out * s)?;
            ffn_out = if let Some(buf) = self.graph_ffn_down_buffer.as_ref() {
                buf.slice_set(&raw, 0, 0)?;
                self.graph_alive_tensors.push(raw);
                buf.clone()
            } else {
                raw
            };
        }
        // Pattern A: fuse post_ffn_norm + residual.
        let raw_x_out = if let Some(post_norm) = &self.post_ffn_norm {
            #[cfg(feature = "cuda")]
            let fused = if ffn_out.device().runs_as_card()
                && ffn_out.dtype() == crate::tensor::DType::F32
                && residual_ffn.dtype() == crate::tensor::DType::F32
                && ffn_out.shape() == residual_ffn.shape()
            {
                crate::inference::kernel::fused::fused_rmsnorm_then_add(
                    &ffn_out,
                    post_norm.weight(),
                    residual_ffn,
                    self.ffn_norm_eps,
                )
                .ok()
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let fused: Option<Tensor> = None;
            match fused {
                Some(t) => t,
                None => (post_norm.forward(&ffn_out)? + residual_ffn)?,
            }
        } else {
            (ffn_out + residual_ffn)?
        };
        let mut x_out = if let Some(buf) = self.graph_post_ffn_norm_buffer.as_ref() {
            buf.slice_set(&raw_x_out, 0, 0)?;
            self.graph_alive_tensors.push(raw_x_out);
            buf.clone()
        } else {
            raw_x_out
        };

        // PLE block for gemma4 graph-mode SPLIT path.
        // compute_from_kv is the third forward path needing PLE - see
        // commit 805c211 finding. Reads from graph_ple_input_buffer
        // (populated by populate_ple_input_buffers BEFORE begin_capture).
        if let (Some(gate), Some(proj), Some(norm)) =
            (&self.ple_inp_gate, &self.ple_proj, &self.ple_post_norm)
        {
            if let Some(ple_input_buf) = self.graph_ple_input_buffer.as_ref() {
                let raw_gate = gate.forward(&x_out)?;
                let ple_state = if let Some(buf) = self.graph_ple_gate_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_gate.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_gate, 0, 0)?;
                        self.graph_alive_tensors.push(raw_gate);
                        buf.clone()
                    } else {
                        raw_gate
                    }
                } else {
                    raw_gate
                };
                #[cfg(feature = "cuda")]
                let raw_gelu =
                    crate::inference::kernel::fused::fused_gelu_mul(&ple_state, ple_input_buf)?;
                #[cfg(not(feature = "cuda"))]
                let raw_gelu = ple_state.gelu()?.mul(ple_input_buf)?;
                let ple_state = if let Some(buf) = self.graph_ple_gelu_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_gelu.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_gelu, 0, 0)?;
                        self.graph_alive_tensors.push(raw_gelu);
                        buf.clone()
                    } else {
                        raw_gelu
                    }
                } else {
                    raw_gelu
                };
                let raw_proj = proj.forward(&ple_state)?;
                let ple_state = if let Some(buf) = self.graph_ple_proj_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_proj.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_proj, 0, 0)?;
                        self.graph_alive_tensors.push(raw_proj);
                        buf.clone()
                    } else {
                        raw_proj
                    }
                } else {
                    raw_proj
                };
                #[cfg(feature = "cuda")]
                let (raw_out, scale_applied) = {
                    let is_decode = ple_state.dim(1).map(|d| d == 1).unwrap_or(false);
                    let cuda_f32 =
                        is_decode && ple_state.dtype() == DType::F32 && x_out.dtype() == DType::F32;
                    if cuda_f32 {
                        if let Some(scale) = &self.ple_output_scale {
                            if scale.dtype() == DType::F32 {
                                let out = crate::inference::kernel::fused::fused_rmsnorm_add_scale(
                                    &ple_state,
                                    norm.weight(),
                                    &x_out,
                                    scale,
                                    norm.eps() as f32,
                                )?;
                                (out, true)
                            } else {
                                let out = crate::inference::kernel::fused::fused_rmsnorm_then_add(
                                    &ple_state,
                                    norm.weight(),
                                    &x_out,
                                    norm.eps() as f32,
                                )?;
                                (out, false)
                            }
                        } else {
                            let out = crate::inference::kernel::fused::fused_rmsnorm_then_add(
                                &ple_state,
                                norm.weight(),
                                &x_out,
                                norm.eps() as f32,
                            )?;
                            (out, false)
                        }
                    } else {
                        let ple_state = norm.forward(&ple_state)?;
                        ((&x_out + ple_state)?, false)
                    }
                };
                #[cfg(not(feature = "cuda"))]
                let (raw_out, scale_applied): (Tensor, bool) = {
                    let ple_state = norm.forward(&ple_state)?;
                    ((&x_out + ple_state)?, false)
                };
                let routed = if let Some(buf) = self.graph_ple_final_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_out.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_out, 0, 0)?;
                        self.graph_alive_tensors.push(raw_out);
                        buf.clone()
                    } else {
                        raw_out
                    }
                } else {
                    raw_out
                };
                x_out = routed;
                if !scale_applied {
                    if let Some(scale) = &self.ple_output_scale {
                        x_out = x_out.broadcast_mul(scale)?;
                    }
                }
            } else {
                if let Some(scale) = &self.ple_output_scale {
                    x_out = x_out.broadcast_mul(scale)?;
                }
            }
        } else {
            if let Some(scale) = &self.ple_output_scale {
                x_out = x_out.broadcast_mul(scale)?;
            }
        }
        // Fix: push the layer's return value to
        // alive_tensors. It's read by the next layer's compute_from_kv
        // (captured chain) - without a push the address goes stale.
        self.graph_alive_tensors.push(x_out.clone());
        Ok(x_out)
    }

    /// Forward pass with padded attention (for CUDA graph mode).
    /// All operations have fixed dimensions - KV buffer is full padded size.
    pub fn forward_padded(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.forward(
            x,
            None,
            index_pos,
            None,
            None,
            None,
            #[cfg(feature = "cuda")]
            None,
        )
    }

    // FIXED version - was using wrong residual (pre-attn instead of post-attn)
    /// Graph-compatible single-token forward. Same structure as `forward`
    /// but with no `index_pos` (read from graph_rope buffers), no mask (read
    /// from padded_mask), no PLE, no shared KV. Residual + FFN flow mirrors
    /// the normal path. Caller must have refreshed `update_rope_buffers(pos)`
    /// and extended `padded_mask` before calling this.
    pub fn forward_graph(&mut self, x: &Tensor) -> Result<Tensor> {
        let rs = self.residual_scale;

        let residual = x;
        let x_norm = self.attn_norm.forward(x)?;
        let mut attn_out = self.forward_attn_graph(&x_norm)?;

        // Phi2 / GPT-NeoX parallel-attention path. Both attn and FFN
        // consume the same x_norm; their outputs are merged into the
        // residual. Mirrors the non-graph `forward()` branch at L2345+.
        // Returns early so the serial post-attn path below (which
        // requires ffn_norm) is skipped - phi2 has no ffn_norm.
        if self.flags.parallel_attn {
            #[cfg(feature = "cuda")]
            let phi2_fast = self.flags.is_phi2_simple_ffn
                && rs.is_none()
                && self.attn_output_bias.is_some()
                && self.ffn_down_bias.is_some()
                && attn_out.device().is_cuda()
                && attn_out.dtype() == crate::tensor::DType::F32;
            #[cfg(not(feature = "cuda"))]
            let phi2_fast = false;
            if !phi2_fast {
                if let Some(b) = &self.attn_output_bias {
                    attn_out = attn_out.broadcast_add(b)?;
                }
            }
            if let Some(s) = rs {
                attn_out = (attn_out * s)?;
            }

            #[cfg(feature = "cuda")]
            let x = if phi2_fast {
                let ffn_pre = self.forward_ffn_phi2_pre_bias(&x_norm)?;
                let attn_bias = self.attn_output_bias.as_ref().unwrap();
                let ffn_bias = self.ffn_down_bias.as_ref().unwrap();
                crate::inference::kernel::fused::fused_phi2_residual_merge(
                    residual, &attn_out, attn_bias, &ffn_pre, ffn_bias,
                )?
            } else {
                let mut ffn_out = self.forward_ffn(&x_norm)?;
                if let Some(s) = rs {
                    ffn_out = (ffn_out * s)?;
                }
                crate::inference::kernel::fused::fused_add_three(residual, &attn_out, &ffn_out)?
            };
            #[cfg(not(feature = "cuda"))]
            let x = {
                let mut ffn_out = self.forward_ffn(&x_norm)?;
                if let Some(s) = rs {
                    ffn_out = (ffn_out * s)?;
                }
                ((residual + &attn_out)? + ffn_out)?
            };
            return Ok(x);
        }

        if let Some(s) = rs {
            attn_out = (attn_out * s)?;
        }

        // Post-attn residual + FFN norm. The `fused_add_rmsnorm_dual`
        // kernel exists but its `(rows, 1, 1)` grid layout regresses
        // single-token decode (1 SM running vs 2 SMs across the
        // unfused add + rmsnorm). Keep the unfused path here until
        // a multi-block-reduction variant is written.
        let (x, x_norm) = {
            let x = if let Some(post_norm) = &self.post_attn_norm {
                (post_norm.forward(&attn_out)? + residual)?
            } else {
                (attn_out + residual)?
            };
            let x_norm = self
                .ffn_norm
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("ffn_norm missing"))?
                .forward(&x)?;
            (x, x_norm)
        };

        // FFN. The custom fused kernels are graph-safe because their
        // PTX module is pre-warmed at model load - every cached
        // `get_or_load_custom_func` call only does cuModuleGetFunction
        // (driver metadata, not stream-affecting).
        let residual = &x;
        let mut ffn_out = if self.flags.is_phi2_simple_ffn {
            // Phi2: y = down(GELU(up(x))). No gate.
            let up = self.ffn_up.forward(&x_norm)?;
            up.gelu_erf()?
        } else if self.flags.fused_ffn_gate_up || self.ffn_gate.is_none() {
            let i = self.flags.intermediate_size;
            let up_states = self.ffn_up.forward(&x_norm)?;
            // Read the packed [.., 2N] matmul output directly (same as the
            // non-graph path in layer_ffn.rs): the narrow->fused_silu_mul chain
            // materialized both halves via 2 device copies per layer  -
            // nsys (qwen3:0.6b decode) showed them as 2x
            // native_slice_u8 + silu_mul ≈ 2 µs/layer vs 1 fused kernel.
            #[cfg(feature = "cuda")]
            if up_states.device().runs_as_card()
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
            } else {
                let gate = up_states.narrow(crate::tensor::D::Minus1, 0, i)?;
                let up = up_states.narrow(crate::tensor::D::Minus1, i, i)?;
                if self.flags.use_gelu {
                    crate::inference::kernel::fused::fused_gelu_mul(&gate, &up)?
                } else {
                    crate::inference::kernel::fused::fused_silu_mul(&gate, &up)?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let gate = up_states.narrow(crate::tensor::D::Minus1, 0, i)?;
                let up = up_states.narrow(crate::tensor::D::Minus1, i, i)?;
                if self.flags.use_gelu {
                    gate.gelu()?.mul(&up)?
                } else {
                    (up * crate::tensor::ops::silu(&gate)?)?
                }
            }
        } else {
            let gate = self.ffn_gate.as_ref().unwrap().forward(&x_norm)?;
            let up = self.ffn_up.forward(&x_norm)?;
            if self.flags.use_gelu {
                #[cfg(feature = "cuda")]
                if gate.device().runs_as_card() {
                    crate::inference::kernel::fused::fused_gelu_mul(&gate, &up)?
                } else {
                    gate.gelu()?.mul(&up)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    gate.gelu()?.mul(&up)?
                }
            } else {
                #[cfg(feature = "cuda")]
                if gate.device().runs_as_card() {
                    crate::inference::kernel::fused::fused_silu_mul(&gate, &up)?
                } else {
                    crate::tensor::ops::silu(&gate)?.mul(&up)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    crate::tensor::ops::silu(&gate)?.mul(&up)?
                }
            }
        };
        ffn_out = self.ffn_down.forward(&ffn_out)?;
        if let Some(s) = rs {
            ffn_out = (ffn_out * s)?;
        }
        let mut x_out = if let Some(post_norm) = &self.post_ffn_norm {
            (post_norm.forward(&ffn_out)? + residual)?
        } else {
            (ffn_out + residual)?
        };

        // PLE block for gemma4 graph mode. Port from
        // forward() line 3672+ reading PLE input from per-layer
        // graph_ple_input_buffer (populated by populate_ple_input_buffers
        // BEFORE begin_capture). Without this, gemma4 graph mode
        // produces garbage tokens (per a probe).
        if let (Some(gate), Some(proj), Some(norm)) =
            (&self.ple_inp_gate, &self.ple_proj, &self.ple_post_norm)
        {
            if let Some(ple_input_buf) = self.graph_ple_input_buffer.as_ref() {
                // gate.forward -> graph_ple_gate_buffer
                let raw_gate = gate.forward(&x_out)?;
                let ple_state = if let Some(buf) = self.graph_ple_gate_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_gate.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_gate, 0, 0)?;
                        buf.clone()
                    } else {
                        raw_gate
                    }
                } else {
                    raw_gate
                };
                // fused_gelu_mul -> graph_ple_gelu_buffer
                #[cfg(feature = "cuda")]
                let raw_gelu =
                    crate::inference::kernel::fused::fused_gelu_mul(&ple_state, ple_input_buf)?;
                #[cfg(not(feature = "cuda"))]
                let raw_gelu = ple_state.gelu()?.mul(ple_input_buf)?;
                let ple_state = if let Some(buf) = self.graph_ple_gelu_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_gelu.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_gelu, 0, 0)?;
                        buf.clone()
                    } else {
                        raw_gelu
                    }
                } else {
                    raw_gelu
                };
                // proj.forward -> graph_ple_proj_buffer
                let raw_proj = proj.forward(&ple_state)?;
                let ple_state = if let Some(buf) = self.graph_ple_proj_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_proj.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_proj, 0, 0)?;
                        buf.clone()
                    } else {
                        raw_proj
                    }
                } else {
                    raw_proj
                };
                // rmsnorm + add (+ optional scale).
                #[cfg(feature = "cuda")]
                let (raw_out, scale_applied) = {
                    let is_decode = ple_state.dim(1).map(|d| d == 1).unwrap_or(false);
                    let cuda_f32 =
                        is_decode && ple_state.dtype() == DType::F32 && x_out.dtype() == DType::F32;
                    if cuda_f32 {
                        if let Some(scale) = &self.ple_output_scale {
                            if scale.dtype() == DType::F32 {
                                let out = crate::inference::kernel::fused::fused_rmsnorm_add_scale(
                                    &ple_state,
                                    norm.weight(),
                                    &x_out,
                                    scale,
                                    norm.eps() as f32,
                                )?;
                                (out, true)
                            } else {
                                let out = crate::inference::kernel::fused::fused_rmsnorm_then_add(
                                    &ple_state,
                                    norm.weight(),
                                    &x_out,
                                    norm.eps() as f32,
                                )?;
                                (out, false)
                            }
                        } else {
                            let out = crate::inference::kernel::fused::fused_rmsnorm_then_add(
                                &ple_state,
                                norm.weight(),
                                &x_out,
                                norm.eps() as f32,
                            )?;
                            (out, false)
                        }
                    } else {
                        let ple_state = norm.forward(&ple_state)?;
                        ((&x_out + ple_state)?, false)
                    }
                };
                #[cfg(not(feature = "cuda"))]
                let (raw_out, scale_applied): (Tensor, bool) = {
                    let ple_state = norm.forward(&ple_state)?;
                    ((&x_out + ple_state)?, false)
                };
                // Route final PLE output through graph_ple_final_buffer.
                let routed = if let Some(buf) = self.graph_ple_final_buffer.as_ref() {
                    let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    let raw_d = raw_out.dim(crate::tensor::D::Minus1).unwrap_or(0);
                    if buf_d == raw_d && buf_d > 0 {
                        buf.slice_set(&raw_out, 0, 0)?;
                        self.graph_alive_tensors.push(raw_out);
                        buf.clone()
                    } else {
                        raw_out
                    }
                } else {
                    raw_out
                };
                x_out = routed;
                if !scale_applied {
                    if let Some(scale) = &self.ple_output_scale {
                        x_out = x_out.broadcast_mul(scale)?;
                    }
                }
            } else {
                // No PLE input buffer (non-gemma4 or buffer not allocated):
                // skip PLE block but still apply scale if present.
                if let Some(scale) = &self.ple_output_scale {
                    x_out = x_out.broadcast_mul(scale)?;
                }
            }
        } else {
            // No PLE block at all: still honor scale if present.
            if let Some(scale) = &self.ple_output_scale {
                x_out = x_out.broadcast_mul(scale)?;
            }
        }
        Ok(x_out)
    }

    /// Zero-alloc single-token dense-CPU decode of one layer. Updates the
    /// residual stream `x` `[hidden]` in place using `arena` scratch + the Q8 KV
    /// cache. `cos`/`sin` are the position's rope rows `[rope_dim/2]`. Returns
    /// `Ok(false)` (caller falls back to the Tensor `forward`) unless eligible:
    /// CPU, standard dense GQA, neox rope, no fused-qkv-bias quirks we don't
    /// handle, Q8 KV active. Mirrors the Tensor forward's math exactly.
    pub fn decode_layer_cpu(
        &mut self,
        x: &mut [f32],
        arena: &mut crate::inference::kernel::cpu_decode_exec::DecodeArena,
        cos: &[f32],
        sin: &[f32],
        ple_input: Option<&[f32]>,
    ) -> Result<bool> {
        use crate::inference::kernel::cpu_decode_exec as ex;
        let (nh, nkv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        // gemma4 PLE: this layer has per-layer-embedding weights -> it MUST receive
        // its ple_input slice + post-norm, else bail (can't half-apply). The PLE
        // block + output scale run at the end. Non-PLE arches: ple_input None.
        let has_ple = self.ple_inp_gate.is_some();
        if has_ple
            && (ple_input.is_none() || self.ple_proj.is_none() || self.ple_post_norm.is_none())
        {
            return Ok(false);
        }
        // Eligibility: standard dense GQA on CPU with neox rope + Q8 KV; no
        // gemma4 V-norm, no interleaved rope, no residual_scale, no partial rope.
        // gemma4: V-norm + GeGLU + sandwich post-norms + per-layer SWA window +
        // F16 KV (Q8 degenerate at hd512). Handled via the gemma branch below.
        let is_gemma = self.attn_v_norm_ones.is_some();
        // residual_scale (granite) is now supported (applied at both residual adds);
        // MoE is supported for the SIMPLE router-prenormed case (granite/qwen*moe/
        // olmoe) - the attention runs zero-alloc here and the FFN calls moe.forward.
        // gemma4-MoE (NOT prenormed: dense-shared FFN + sandwich norms) still bails.
        let rs = self.residual_scale;
        let moe_prenormed = self
            .moe
            .as_ref()
            .map(|m| m.router_prenormed)
            .unwrap_or(false);
        if self.flags.use_rope_i
            || self.attn_v.is_none() && self.attn_qkv.is_none()
            || self.flags.is_phi2_simple_ffn
            || self.rope_dim != 0 && self.rope_dim != hd
            || x.len() != arena.hidden
            || arena.ffn != self.flags.intermediate_size
            // gemma needs post-norm weights + gelu; if it's gemma-shaped but
            // missing those, bail (don't half-apply the sandwich norm).
            || (is_gemma && (self.post_attn_norm.is_none() || self.post_ffn_norm.is_none() || !self.flags.use_gelu))
            // OLMo2 (post-norm-only): the Identity pre-norm is a passthrough, NOT a
            // real RMS - this zero-alloc executor caches attn_norm.weight() and would
            // mis-apply the [1] ones tensor as a norm. Route it to the regular forward
            // (where WeightedNorm::Identity::forward returns x unchanged, correctly).
            || matches!(self.attn_norm, WeightedNorm::Identity(_))
            // MoE layers: this executor computes a DENSE FFN (ffn_gate/up/down); an
            // MoE layer has no real dense FFN (only expert stacks + a placeholder),
            // so it MUST bail to the regular forward that routes through the experts.
            // Only the SIMPLE router-prenormed MoE is handled here (FFN via
            // moe.forward). gemma4-MoE (not prenormed) still bails.
            || (self.moe.is_some() && !moe_prenormed)
        {
            return Ok(false);
        }
        // Lazily build the Q8 KV store (prefill's cpu_q8_append usually already
        // did, with the same eligibility gate - but be safe for empty-prompt).
        if is_gemma {
            if self.cpu_f16_kv.is_none() {
                // Per-layer window: SWA layers have sliding_window set; global None.
                let window = self.sliding_window.filter(|&w| w > 0);
                self.cpu_f16_kv = Some(crate::inference::cache::cpu_f16_kv::CpuF16Kv::new(
                    nh, nkv, hd, window,
                ));
            }
        } else if self.cpu_q8_kv.is_none() {
            self.cpu_q8_kv = Some(crate::inference::cache::cpu_q8_kv::CpuQ8Kv::new(
                nh, nkv, hd,
            ));
        }
        // Materialise the per-layer constant norm weights/biases once (f32),
        // then reuse every token - removes ~5 `to_vec1` allocs/layer/token of
        // pure repeated conversion of weights that never change.
        if self.cpu_norm_cache.is_none() {
            let q_bias = self
                .attn_q_bias
                .as_ref()
                .map(|b| b.to_vec1::<f32>())
                .transpose()?;
            let k_bias = self
                .attn_k_bias
                .as_ref()
                .map(|b| b.to_vec1::<f32>())
                .transpose()?;
            let v_bias = self
                .attn_v_bias
                .as_ref()
                .map(|b| b.to_vec1::<f32>())
                .transpose()?;
            let q_norm_w = match &self.attn_q_norm {
                Some(qn) => Some((qn.weight().to_vec1::<f32>()?, qn.eps() as f32)),
                None => None,
            };
            let k_norm_w = match &self.attn_k_norm {
                Some(kn) => Some((kn.weight().to_vec1::<f32>()?, kn.eps() as f32)),
                None => None,
            };
            let post_attn_norm_w = match &self.post_attn_norm {
                Some(n) => Some((n.weight().to_vec1::<f32>()?, n.eps() as f32)),
                None => None,
            };
            let post_ffn_norm_w = match &self.post_ffn_norm {
                Some(n) => Some((n.weight().to_vec1::<f32>()?, n.eps() as f32)),
                None => None,
            };
            let ple_post_norm_w = match &self.ple_post_norm {
                Some(n) => Some((n.weight().to_vec1::<f32>()?, n.eps() as f32)),
                None => None,
            };
            let ple_output_scale = self
                .ple_output_scale
                .as_ref()
                .map(|t| {
                    t.to_dtype(crate::tensor::DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()
                })
                .transpose()?;
            self.cpu_norm_cache =
                Some(crate::inference::kernel::cpu_decode_exec::DecodeNormCache {
                    attn_norm_w: self.attn_norm.weight().to_vec1::<f32>()?,
                    attn_eps: self.attn_norm.eps() as f32,
                    ffn_norm_w: self.ffn_norm.as_ref().unwrap().weight().to_vec1::<f32>()?,
                    ffn_eps: self.ffn_norm_eps,
                    q_bias,
                    k_bias,
                    v_bias,
                    q_norm_w,
                    k_norm_w,
                    post_attn_norm_w,
                    post_ffn_norm_w,
                    ple_post_norm_w,
                    ple_output_scale,
                });
        }
        // Owned take: `nc` no longer borrows `self`, so the attention block's
        // `&mut self.cpu_q8_kv` doesn't conflict. Restored after the FFN norm.
        let nc = self.cpu_norm_cache.take().unwrap();
        // -- Attention block ----------------------------------------------
        ex::rms_norm_slice(x, &nc.attn_norm_w, nc.attn_eps, &mut arena.norm);

        // Q/K/V projections (split or fused) into arena.q / .k / .v
        if let Some(qkv) = self.attn_qkv.as_ref() {
            if !qkv.forward_slice_cpu(&arena.norm, &mut arena.qkvfull)? {
                return Ok(false);
            }
            arena.q.copy_from_slice(&arena.qkvfull[0..nh * hd]);
            arena
                .k
                .copy_from_slice(&arena.qkvfull[nh * hd..(nh + nkv) * hd]);
            arena
                .v
                .copy_from_slice(&arena.qkvfull[(nh + nkv) * hd..(nh + 2 * nkv) * hd]);
        } else {
            // q, k and v all read the SAME normed vector. Quantising the
            // activation inside each GEMV would do that work three times, so
            // share one quantisation across the three; the helper returns false
            // without writing when any weight is off the CPU fast path, and the
            // per-weight calls below then run unchanged.
            let (mq, mk, mv) = (
                self.attn_q.as_ref().unwrap(),
                self.attn_k.as_ref().unwrap(),
                self.attn_v.as_ref().unwrap(),
            );
            let shared = {
                let mut outs: [&mut [f32]; 3] = [&mut arena.q, &mut arena.k, &mut arena.v];
                crate::inference::generic_transformer::projection::QMatMul::forward_slice_cpu_shared(
                    &arena.norm,
                    &[mq, mk, mv],
                    &mut outs,
                )?
            };
            if !shared {
                if !mq.forward_slice_cpu(&arena.norm, &mut arena.q)? {
                    return Ok(false);
                }
                if !mk.forward_slice_cpu(&arena.norm, &mut arena.k)? {
                    return Ok(false);
                }
                if !mv.forward_slice_cpu(&arena.norm, &mut arena.v)? {
                    return Ok(false);
                }
            }
        }
        // Optional QKV biases (Qwen2) - from the cached f32 copies.
        if self.flags.has_qkv_bias {
            if let Some(bq) = &nc.q_bias {
                ex::residual_add(&mut arena.q, bq);
            }
            if let Some(bk) = &nc.k_bias {
                ex::residual_add(&mut arena.k, bk);
            }
            if let Some(bv) = &nc.v_bias {
                ex::residual_add(&mut arena.v, bv);
            }
        }
        // Optional QK-norm (Qwen3): per-head RMSNorm over head_dim.
        if let (Some((qw, qe)), Some((kw, ke))) = (&nc.q_norm_w, &nc.k_norm_w) {
            let mut tmp = vec![0f32; hd];
            for h in 0..nh {
                ex::rms_norm_slice(&arena.q[h * hd..(h + 1) * hd], qw, *qe, &mut tmp);
                arena.q[h * hd..(h + 1) * hd].copy_from_slice(&tmp);
            }
            for h in 0..nkv {
                ex::rms_norm_slice(&arena.k[h * hd..(h + 1) * hd], kw, *ke, &mut tmp);
                arena.k[h * hd..(h + 1) * hd].copy_from_slice(&tmp);
            }
        }
        // RoPE (neox) on Q and K.
        ex::rope_neox_slice(&mut arena.q, cos, sin, nh, hd);
        ex::rope_neox_slice(&mut arena.k, cos, sin, nkv, hd);

        // gemma4: V-norm (RMSNorm-no-weight per kv-head, eps 1e-6) before append.
        if is_gemma {
            for h in 0..nkv {
                let vh = &mut arena.v[h * hd..(h + 1) * hd];
                let mut ss = 0f32;
                for &x in vh.iter() {
                    ss += x * x;
                }
                let inv = 1.0 / (ss / hd as f32 + 1e-6).sqrt();
                for x in vh.iter_mut() {
                    *x *= inv;
                }
            }
        }
        // Attention. gemma4 -> F16 windowed KV (Q8 degenerate at hd512); else Q8.
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt()) as f32;
        if is_gemma {
            let cache = self.cpu_f16_kv.as_mut().unwrap();
            cache.append(&arena.k, &arena.v)?;
            cache.attention(&arena.q, scale, &mut arena.attn)?;
        } else {
            let cache = self.cpu_q8_kv.as_mut().unwrap();
            cache.append(&arena.k, &arena.v)?;
            // GQA: grouped-PV flash (V read once/kv-head, not n_repx); MHA keeps
            // the head-parallel path (no V-read redundancy to amortise).
            if self.n_head > self.n_kv_head {
                cache.attention_grouped(&arena.q, scale, &mut arena.attn)?;
            } else {
                cache.attention(&arena.q, scale, &mut arena.attn)?;
            }
        }
        // KV-cache bookkeeping. Donor layers of a shared_kv model (gemma4 8B:
        // every non-shared layer is `populate_dual_kv`) are read DOWNSTREAM by
        // the shared layers via `self.kv_cache.current_kv()` - so the executor
        // must populate the F-dtype kv_cache with the real (RoPE'd, V-normed)
        // K/V, not merely bump the counter. Without this the shared layers read
        // a counter-only cache (stale/empty donor K/V) and their attention
        // collapses to the most recent tokens -> the "capital of the capital of"
        // recent-token duplication that previously gated gemma4 out of this
        // path. Build [1,nkv,1,hd] tensors from the arena slices and append
        // (handles the SWA window + advances the counter). Non-donor layers'
        // kv_cache is never read, so they only need the cheap counter advance
        // and stay zero-alloc (the next request's mask length stays correct).
        if self.populate_dual_kv {
            let kt = crate::tensor::Tensor::from_vec(
                arena.k[..nkv * hd].to_vec(),
                (1, nkv, 1, hd),
                &crate::tensor::Device::Cpu,
            )?;
            let vt = crate::tensor::Tensor::from_vec(
                arena.v[..nkv * hd].to_vec(),
                (1, nkv, 1, hd),
                &crate::tensor::Device::Cpu,
            )?;
            self.kv_cache.append(&kt, &vt)?;
        } else {
            self.kv_cache.advance_seq_len(1);
        }
        // O projection (+ gemma4 post_attn_norm) + residual.
        if !self
            .attn_output
            .forward_slice_cpu(&arena.attn, &mut arena.proj)?
        {
            self.cpu_norm_cache = Some(nc);
            return Ok(false);
        }
        if let Some((w, eps)) = &nc.post_attn_norm_w {
            ex::rms_norm_slice(&arena.proj, w, *eps, &mut arena.norm);
            arena.proj.copy_from_slice(&arena.norm);
        }
        // Granite: scale the attention output by residual_multiplier before the add.
        if let Some(s) = rs {
            let s = s as f32;
            for v in arena.proj.iter_mut() {
                *v *= s;
            }
        }
        ex::residual_add(x, &arena.proj);

        // -- MoE FFN branch (simple router-prenormed MoE) --
        // Attention ran zero-alloc above; route the FFN through the existing
        // moe.forward (router + experts). x is the post-attn residual.
        if self.moe.is_some() {
            let hidden = x.len();
            ex::rms_norm_slice(x, &nc.ffn_norm_w, nc.ffn_eps, &mut arena.norm);
            let dev = crate::tensor::Device::Cpu;
            let x_norm_t = crate::tensor::Tensor::from_vec(
                arena.norm[..hidden].to_vec(),
                (1usize, 1usize, hidden),
                &dev,
            )?;
            let x_t = crate::tensor::Tensor::from_vec(
                x[..hidden].to_vec(),
                (1usize, 1usize, hidden),
                &dev,
            )?;
            let emb = self.embedding_length_for_moe;
            let ffn_eps = self.ffn_norm_eps;
            let m = self
                .moe
                .as_ref()
                .unwrap()
                .forward(&x_norm_t, &x_t, emb, ffn_eps, false)?;
            let mv: Vec<f32> = m
                .flatten_all()?
                .to_dtype(crate::tensor::DType::F32)?
                .to_vec1()?;
            let s = rs.map(|r| r as f32).unwrap_or(1.0);
            for i in 0..hidden {
                x[i] += s * mv[i];
            }
            self.cpu_norm_cache = Some(nc);
            return Ok(true);
        }

        // -- FFN block (SwiGLU) -------------------------------------------
        // Cooperative fast path: norm + gate/up + activation + down + residual in
        // ONE pool region (six pool launches + serial glue otherwise). Same math,
        // same kernels; falls through to the sequential path on any miss.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if self.moe.is_none()
            && nc.post_ffn_norm_w.is_none()
            && !self.flags.use_gelu
            && self.ple_inp_gate.is_none()
        {
            {
                let gate_raw = self.ffn_gate.as_ref().and_then(|g| g.cpu_raw());
                let gate_missing = self.ffn_gate.is_some() && gate_raw.is_none();
                if let (false, Some(u), Some(d)) =
                    (gate_missing, self.ffn_up.cpu_raw(), self.ffn_down.cpu_raw())
                {
                    let g = gate_raw;
                    let (nw, neps) = (&nc.ffn_norm_w, nc.ffn_eps);
                    let norm_fn = |xs: &[f32], out: &mut [f32]| {
                        ex::rms_norm_slice(xs, nw, neps, out);
                    };
                    let act_fn = |gs: &[f32], us: &[f32], out: &mut [f32]| {
                        ex::silu_mul_slice(gs, us, out);
                    };
                    let i = arena.ffn;
                    if crate::tensor::quant_cpu::ffn_swiglu_coop(
                        x,
                        g,
                        u,
                        d,
                        rs.map(|r| r as f32),
                        &norm_fn,
                        &act_fn,
                        &mut arena.norm,
                        &mut arena.gateup[..2 * i],
                        &mut arena.act[..i],
                    )? {
                        self.cpu_norm_cache = Some(nc);
                        return Ok(true);
                    }
                }
            }
        }
        ex::rms_norm_slice(x, &nc.ffn_norm_w, nc.ffn_eps, &mut arena.norm);
        let i = arena.ffn; // intermediate_size
        if let Some(gate_w) = self.ffn_gate.as_ref() {
            // Separate gate/up: write into gateup[0..i] and gateup[i..2i].
            let (g, u) = arena.gateup.split_at_mut(i);
            if !gate_w.forward_slice_cpu(&arena.norm, g)? {
                self.cpu_norm_cache = Some(nc);
                return Ok(false);
            }
            if !self.ffn_up.forward_slice_cpu(&arena.norm, u)? {
                self.cpu_norm_cache = Some(nc);
                return Ok(false);
            }
        } else {
            // Fused: ffn_up outputs [gate‖up] = 2*i.
            if !self
                .ffn_up
                .forward_slice_cpu(&arena.norm, &mut arena.gateup)?
            {
                self.cpu_norm_cache = Some(nc);
                return Ok(false);
            }
        }
        // gemma4 -> GeGLU (gelu(gate)*up); else SwiGLU (silu(gate)*up).
        if self.flags.use_gelu {
            ex::gelu_mul_slice(&arena.gateup[0..i], &arena.gateup[i..2 * i], &mut arena.act);
        } else {
            ex::silu_mul_slice(&arena.gateup[0..i], &arena.gateup[i..2 * i], &mut arena.act);
        }
        if !self
            .ffn_down
            .forward_slice_cpu(&arena.act, &mut arena.proj)?
        {
            self.cpu_norm_cache = Some(nc);
            return Ok(false);
        }
        // gemma4 sandwich: post_ffn_norm on the FFN output BEFORE residual.
        if let Some((w, eps)) = &nc.post_ffn_norm_w {
            ex::rms_norm_slice(&arena.proj, w, *eps, &mut arena.norm);
            arena.proj.copy_from_slice(&arena.norm);
        }
        // Granite: scale the dense-FFN output by residual_multiplier before the add.
        if let Some(s) = rs {
            let s = s as f32;
            for v in arena.proj.iter_mut() {
                *v *= s;
            }
        }
        ex::residual_add(x, &arena.proj);
        // -- gemma4 PLE block (per-layer embedding) + output scale --------------
        // x is the post-FFN residual. PLE: x += rmsnorm(proj . (gelu(gate.x) *
        // ple_input)); then per-layer output scale. Matches the Tensor forward
        // exactly. Zero-alloc (arena.ple lazily sized to ple_dim, proj/norm reused).
        let ple_ok = if let (Some(gate), Some(proj)) =
            (self.ple_inp_gate.as_ref(), self.ple_proj.as_ref())
        {
            if let Some(ple_in) = ple_input {
                let pd = gate.out_dim();
                if arena.ple.len() != pd {
                    arena.ple.resize(pd, 0.0);
                }
                let a = gate.forward_slice_cpu(x, &mut arena.ple)?;
                ex::gelu_mul_inplace(&mut arena.ple, ple_in);
                let b = proj.forward_slice_cpu(&arena.ple, &mut arena.proj)?;
                a && b
            } else {
                true
            }
        } else {
            true
        };
        if !ple_ok {
            return Ok(false);
        }
        if self.ple_inp_gate.is_some() && ple_input.is_some() {
            if let Some((w, eps)) = &nc.ple_post_norm_w {
                ex::rms_norm_slice(&arena.proj, w, *eps, &mut arena.norm);
                arena.proj.copy_from_slice(&arena.norm);
            }
            ex::residual_add(x, &arena.proj);
        }
        if let Some(scale) = &nc.ple_output_scale {
            ex::mul_inplace(x, scale);
        }
        self.cpu_norm_cache = Some(nc);
        Ok(true)
    }

    /// entry: try the zero-alloc decode path for a single CPU F32 token.
    /// Extracts the residual + this position's rope rows, runs `decode_layer_cpu`
    /// on the persistent arena, returns the updated residual as a Tensor.
    /// `Ok(None)` -> caller runs the normal Tensor `forward`.
    fn try_decode_layer_cpu(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        ple_layer_input: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        // Zero-alloc decode executor. Validated coherent + non-crashing
        // (the earlier "segfault" was a self-inflicted pkill; the real bug was
        // the eligibility gate + an F16-kv_cache desync, both fixed).
        const ENABLED: bool = true;
        if !ENABLED {
            return Ok(None);
        }
        let dims = x.dims();
        let reject = if !matches!(x.device(), crate::tensor::Device::Cpu) {
            Some(0)
        } else if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 {
            Some(1)
        } else if x.dtype() != crate::tensor::DType::F32 {
            Some(2)
        } else {
            None
        };
        if crate::inference::place::layer_perf::stages::enabled() {
            crate::inference::place::layer_perf::stages::note_host_path(reject);
        }
        if reject.is_some() {
            return Ok(None);
        }
        let hidden = dims[2];
        // MoE layers have no dense FFN (intermediate_size may be 0); the arena's
        // FFN buffers are unused on the MoE path, so size them to `hidden` as a
        // harmless placeholder. Dense layers still require a real intermediate_size.
        let ffn = if self.flags.intermediate_size == 0 {
            if self.moe.is_some() {
                hidden
            } else {
                return Ok(None);
            }
        } else {
            self.flags.intermediate_size
        };
        let mut xs = x.flatten_all()?.to_vec1::<f32>()?;
        let cos = self
            .cos
            .narrow(0, index_pos, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sin = self
            .sin
            .narrow(0, index_pos, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut arena = self.cpu_decode_arena.take().unwrap_or_else(|| {
            crate::inference::kernel::cpu_decode_exec::DecodeArena::new(
                hidden,
                ffn,
                self.n_head,
                self.n_kv_head,
                self.head_dim,
            )
        });
        // gemma4 PLE: this layer's per-layer-embedding slice [ple_dim] as f32.
        let ple_vec: Option<Vec<f32>> = match ple_layer_input {
            Some(t) => Some(
                t.flatten_all()?
                    .to_dtype(crate::tensor::DType::F32)?
                    .to_vec1::<f32>()?,
            ),
            None => None,
        };
        let used = self.decode_layer_cpu(&mut xs, &mut arena, &cos, &sin, ple_vec.as_deref())?;
        self.cpu_decode_arena = Some(arena);
        if used {
            Ok(Some(Tensor::from_vec(
                xs,
                (1usize, 1usize, hidden),
                &crate::tensor::Device::Cpu,
            )?))
        } else {
            Ok(None)
        }
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        ple_layer_input: Option<&Tensor>,
        shared_kv: Option<(Tensor, Tensor)>,
        donor_f16: Option<&crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
        #[cfg(feature = "cuda")] shared_kv_q8: Option<&crate::inference::cache::q8_kv::Q8KvCache>,
    ) -> Result<Tensor> {
        // Zero-alloc dense-CPU single-token decode fast path (eligible
        // arches only; falls through to the Tensor forward otherwise).
        // The executor handles gemma4 PLE (block + output scale) end-to-end. The
        // prior "recent-token duplication" that gated gemma4 out was NOT a
        // per-component numerical bug (all of norms/qk-norm/rope/V-norm/GeGLU/
        // sandwich-norms/PLE were verified equal) - it was cross-layer KV
        // plumbing: the executor only advanced `self.kv_cache`'s counter, so the
        // shared_kv layers downstream (gemma4 8B has 18) read a counter-only
        // donor cache and collapsed to recent tokens. Fixed in the attention
        // block above (donor layers now append real K/V to self.kv_cache), so
        // gemma4 PLE layers run the executor too. shared_kv layers still take the
        // Tensor path (they reuse the donor's K/V).
        // Runtime CPU-device gate (not compile-time): the fast zero-alloc decode
        // path serves any build whose residual is on CPU - the pure-CPU build AND
        // the CUDA binary run with --cpu / --num-gpu 0 (which otherwise fell back
        // to the slow Tensor forward: measured deepcoder 2.4->2.6 tok/s, -4%->+4%
        // vs ollama). try_decode_layer_cpu re-checks device/dtype/shape and falls
        // through (Ok(None)) when ineligible, so GPU decode is unaffected.
        if matches!(x.device(), crate::tensor::Device::Cpu) && mask.is_none() && shared_kv.is_none()
        {
            if let Some(y) = self.try_decode_layer_cpu(x, index_pos, ple_layer_input)? {
                return Ok(y);
            }
        }
        let rs = self.residual_scale; // Granite: 0.22

        // Per-stage profiling, off unless switched on: the .map() chains below are
        // no-ops when prof_t0 is None. Each mark synchronises, because a stage boundary
        // that does not wait for the device measures the launch and not the work.
        let prof_t0: Option<std::time::Instant> =
            if crate::inference::place::layer_perf::stages::enabled() {
                let _ = x.device().synchronize();
                Some(std::time::Instant::now())
            } else {
                None
            };

        // -- Attention block ------------------------------------------------
        let residual = x;
        let raw_x_norm_attn = self.attn_norm.forward(x)?;
        // route attn_norm output through stable buffer for
        // graph capture (decode only).
        let x_norm = if x.dim(1).map(|d| d == 1).unwrap_or(false) {
            if let Some(buf) = self.graph_x_norm_attn_buffer.as_ref() {
                let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                let raw_d = raw_x_norm_attn.dim(crate::tensor::D::Minus1).unwrap_or(0);
                if buf_d == raw_d && buf_d > 0 {
                    buf.slice_set(&raw_x_norm_attn, 0, 0)?;
                    buf.clone()
                } else {
                    raw_x_norm_attn
                }
            } else {
                raw_x_norm_attn
            }
        } else {
            raw_x_norm_attn
        };
        let prof_t_attn_norm = prof_t0.map(|_| {
            let _ = x.device().synchronize();
            std::time::Instant::now()
        });

        // Phi2 shared-Q8_1 fast path: x_norm feeds BOTH attn_qkv AND ffn_up
        // - pre-quantize x_norm to Q8_1 ONCE and reuse for both matmuls
        // instead of letting each QMatMul.forward re-quantize separately.
        // Saves 1 quantize launch per layer (~1-2 % decode). Currently
        // applies to phi2-class single-token decode on CUDA with Q4_K
        // weights.
        //
        // Wired but DISABLED: produces NaN sampling weights  -
        // suspect a layout/scaling mismatch between standalone
        // quantize_q8_1_pub and what attn_qkv expected from
        // QMatMul.forward. Foundation infrastructure (try_phi2_shared_q8_1_qkv
        // Phi2 shared-Q8_1 fast path: pre-quantize x_norm to Q8_1 ONCE
        // and reuse for both attn_qkv and ffn_up matmuls (saves 1
        // quantize launch/layer = ~1-2% decode). Verified correct on
        // Q4_0 weights via test_ln_qmm_gguf (max_diff=0). Default-on.
        #[cfg(feature = "cuda")]
        let enable_shared_q8_1 = true;
        #[cfg(not(feature = "cuda"))]
        let enable_shared_q8_1 = false;
        #[cfg(feature = "cuda")]
        let (pre_computed_qkv, shared_q8_1_buf) = if enable_shared_q8_1 {
            self.try_phi2_shared_q8_1_qkv(&x_norm)
                .unwrap_or((None, None))
        } else {
            (None, None)
        };
        #[cfg(not(feature = "cuda"))]
        let (pre_computed_qkv, shared_q8_1_buf): (Option<Tensor>, Option<()>) = (None, None);

        // Alt-stream FFN_up overlap: while the attn chain runs on the
        // default stream, the ffn_up mvq can start on the device's alt
        // CUDA stream. They both consume q8_1_buf (which is finished by
        // the time we get here), produce data-independent outputs, and
        // resync at fused_phi2_residual_merge.
        //
        // Phi2 simple-FFN multi-stream (alt_cuda_stream) FFN-up overlap.
        // Default-on for parallel-attn + is_phi2_simple_ffn arches
        // (moondream and similar) - A/B bench measured moondream:latest
        // decode 346 -> 374 tok/s (+8% with FULL variant). Non-phi2 arches
        // unaffected (the inner guard checks parallel_attn && is_phi2_simple_ffn).
        #[cfg(feature = "cuda")]
        let alt_ffn_up: Option<(Tensor, crate::tensor::cuda_ext::CudaEvent)> = {
            let enabled = true;
            if enabled && self.flags.parallel_attn && self.flags.is_phi2_simple_ffn {
                if let Some(buf) = shared_q8_1_buf.as_ref() {
                    // Record event on default stream so alt stream can wait
                    // for the q8_1 quantize to be done.
                    let dev_res = x_norm.device().as_cuda_device();
                    match dev_res {
                        Ok(dev) => {
                            let evt = dev.cuda_stream().record_event(None).ok();
                            match evt {
                                Some(evt_ready) => {
                                    let hidden = x_norm.dim(crate::tensor::D::Minus1)?;
                                    match self
                                        .start_alt_stream_ffn_up_from_q8_1(buf, hidden, &evt_ready)
                                    {
                                        Ok(pair) => Some(pair),
                                        Err(e) => {
                                            tracing::warn!(
                                                "alt-stream ffn_up failed (falling back): {e}"
                                            );
                                            None
                                        }
                                    }
                                }
                                None => None,
                            }
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };
        #[cfg(not(feature = "cuda"))]
        let alt_ffn_up: Option<()> = None;

        let mut attn_out = if pre_computed_qkv.is_some() {
            self.forward_attn_with_qkv(
                &x_norm,
                mask,
                index_pos,
                shared_kv,
                pre_computed_qkv.as_ref(),
                donor_f16,
            )?
        } else {
            self.forward_attn(
                &x_norm,
                mask,
                index_pos,
                shared_kv,
                donor_f16,
                #[cfg(feature = "cuda")]
                shared_kv_q8,
            )?
        };
        // Phi2 fast path: parallel-attn + simple-FFN + both biases on CUDA F32 -> defer
        // both bias adds + residual merge into a single fused kernel. When NOT on
        // the fast path, apply attn_output_bias here as before.
        #[cfg(feature = "cuda")]
        let phi2_residual_fast = self.flags.parallel_attn
            && self.flags.is_phi2_simple_ffn
            && rs.is_none()
            && self.attn_output_bias.is_some()
            && self.ffn_down_bias.is_some()
            && attn_out.device().is_cuda()
            && attn_out.dtype() == crate::tensor::DType::F32;
        #[cfg(not(feature = "cuda"))]
        let phi2_residual_fast = false;
        if !phi2_residual_fast {
            if let Some(b) = &self.attn_output_bias {
                attn_out = attn_out.broadcast_add(b)?;
            }
        }
        if let Some(s) = rs {
            attn_out = (attn_out * s)?;
        }
        let prof_t_attn = prof_t_attn_norm.map(|_| {
            let _ = attn_out.device().synchronize();
            std::time::Instant::now()
        });

        // -- Branch on serial vs parallel attention ------------------------
        // Phi2/GPT-NeoX use parallel attention: attn and ffn both consume
        // the SAME x_norm (computed above), then their outputs are summed
        // into the residual.
        let (mut x, prof_t_attn_res, prof_t_ffn, prof_t_ffn_res) = if self.flags.parallel_attn {
            #[cfg(feature = "cuda")]
            let x = if phi2_residual_fast {
                // 5-input fused merge: residual + (attn + attn_bias) + (ffn_pre + ffn_bias)
                // in one launch. Saves the attn_output_bias broadcast_add (which we
                // skipped above), the ffn_down_bias broadcast_add (which
                // forward_ffn_phi2_pre_bias skips), and collapses the residual sum.
                // If alt-stream FFN_up was kicked off earlier, finish it.
                // The FULL variant keeps the WHOLE FFN chain on alt_stream
                // (gelu + ffn_down) - measured +4% on top of basic alt-FFN
                // for moondream. Default-on permanently.
                let alt_full = true;
                let ffn_pre = match (alt_ffn_up.as_ref(), shared_q8_1_buf.as_ref()) {
                    (Some((up, alt_event)), _) if alt_full => {
                        let dev = up.device().as_cuda_device()?;
                        let alt_stream = dev.alt_cuda_stream().map_err(|e| {
                            crate::tensor::Error::msg(format!("alt_cuda_stream: {e}"))
                        })?;
                        let (pre, final_event) = self.finish_ffn_phi2_pre_bias_after_alt_full(
                            up,
                            alt_event,
                            &alt_stream,
                        )?;
                        // Default stream waits for full-FFN completion
                        // before the residual merge consumes ffn_pre.
                        dev.cuda_stream().wait(&final_event).map_err(|e| {
                            crate::tensor::Error::msg(format!("wait full-alt: {e}"))
                        })?;
                        pre
                    }
                    (Some((up, alt_event)), _) => {
                        self.finish_ffn_phi2_pre_bias_after_alt(up, alt_event)?
                    }
                    (None, Some(buf)) => self.forward_ffn_phi2_pre_bias_from_q8_1(
                        buf,
                        x_norm.dim(crate::tensor::D::Minus1)?,
                    )?,
                    (None, None) => self.forward_ffn_phi2_pre_bias(&x_norm)?,
                };
                let attn_bias = self.attn_output_bias.as_ref().unwrap();
                let ffn_bias = self.ffn_down_bias.as_ref().unwrap();
                crate::inference::kernel::fused::fused_phi2_residual_merge(
                    residual, &attn_out, attn_bias, &ffn_pre, ffn_bias,
                )?
            } else {
                let mut ffn_out = self.forward_ffn(&x_norm)?;
                if let Some(s) = rs {
                    ffn_out = (ffn_out * s)?;
                }
                crate::inference::kernel::fused::fused_add_three(residual, &attn_out, &ffn_out)?
            };
            #[cfg(not(feature = "cuda"))]
            let x = {
                let mut ffn_out = self.forward_ffn(&x_norm)?;
                if let Some(s) = rs {
                    ffn_out = (ffn_out * s)?;
                }
                ((residual + &attn_out)? + ffn_out)?
            };
            let prof_t_attn_res = prof_t_attn.map(|_| {
                let _ = x.device().synchronize();
                std::time::Instant::now()
            });
            let prof_t_ffn = prof_t_attn_res;
            let prof_t_ffn_res = prof_t_ffn;
            (x, prof_t_attn_res, prof_t_ffn, prof_t_ffn_res)
        } else {
            // Serial (Llama/Gemma/Qwen/...): attn -> residual -> ffn_norm -> ffn -> residual.
            // Fuse (attn_out + residual) + ffn_norm into one kernel when possible.
            //
            // Tested: extending fuse to gemma4's post_attn_norm
            // path (apply post_norm first, then fuse rest) REGRESSED long
            // -14.5 pp and medium -7.1 pp. The fused kernel's (rows,1,1)
            // grid layout penalizes the per-token output write path even
            // more than the unfused chain on gemma4's shapes. Kept post_attn_norm
            // out of the fused gate.
            #[cfg(feature = "cuda")]
            let (x, x_norm_ffn) = if self.post_attn_norm.is_none() && self.ffn_norm_weight.is_some()
            {
                let norm_w = self.ffn_norm_weight.as_ref().unwrap();
                crate::inference::kernel::fused::fused_add_rmsnorm_dual(
                    &attn_out,
                    residual,
                    norm_w,
                    self.ffn_norm_eps,
                )?
            } else {
                // An attempt: extended fused kernel for gemma4's
                // post_attn_norm + add + ffn_norm path REGRESSED -6.2pp.
                // Same root cause (-14.5pp): (rows,1,1)
                // grid penalty on per-token decode shapes. Kept the
                // kernel + wrapper in tree (fused_gemma4_post_add_norm)
                // for future graph-mode or batched-prefill use.
                // //
                // A re-attempt: the prior regression at this site
                // was attributed to `Tensor::zeros + zero-fill` overhead
                // (commit ce4cc04). With the kernel switched to
                // `Tensor::empty` (no cudaMemsetAsync) the alloc cost
                // drops to a pool slot lookup, and the 1-launch save
                // should net positive. Decode-only gating mirrors the
                // post_ffn_norm site (line 3566) for the same reason
                // - at prefill seq=T_prefill the buffer could be tens of
                // MB per layer.
                let x = if let Some(post_norm) = &self.post_attn_norm {
                    let is_decode = attn_out.dim(1).map(|d| d == 1).unwrap_or(false);
                    let raw = if is_decode
                        && attn_out.dtype() == DType::F32
                        && residual.dtype() == DType::F32
                    {
                        crate::inference::kernel::fused::fused_rmsnorm_then_add(
                            &attn_out,
                            post_norm.weight(),
                            residual,
                            post_norm.eps() as f32,
                        )?
                    } else {
                        (post_norm.forward(&attn_out)? + residual)?
                    };
                    // route post_attn_norm fuse output through stable buffer.
                    if is_decode {
                        if let Some(buf) = self.graph_post_attn_norm_buffer.as_ref() {
                            let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                            let raw_d = raw.dim(crate::tensor::D::Minus1).unwrap_or(0);
                            if buf_d == raw_d && buf_d > 0 {
                                buf.slice_set(&raw, 0, 0)?;
                                buf.clone()
                            } else {
                                raw
                            }
                        } else {
                            raw
                        }
                    } else {
                        raw
                    }
                } else {
                    (attn_out + residual)?
                };
                let raw_x_norm = self
                    .ffn_norm
                    .as_ref()
                    .ok_or_else(|| {
                        crate::tensor::Error::msg("ffn_norm missing on serial-attn layer")
                    })?
                    .forward(&x)?;
                // route ffn_norm output through stable buffer.
                let x_norm = if x.dim(1).map(|d| d == 1).unwrap_or(false) {
                    if let Some(buf) = self.graph_x_norm_ffn_buffer.as_ref() {
                        let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        let raw_d = raw_x_norm.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        if buf_d == raw_d && buf_d > 0 {
                            buf.slice_set(&raw_x_norm, 0, 0)?;
                            buf.clone()
                        } else {
                            raw_x_norm
                        }
                    } else {
                        raw_x_norm
                    }
                } else {
                    raw_x_norm
                };
                (x, x_norm)
            };
            #[cfg(not(feature = "cuda"))]
            let (x, x_norm_ffn) = {
                let x = if let Some(post_norm) = &self.post_attn_norm {
                    (post_norm.forward(&attn_out)? + residual)?
                } else {
                    (attn_out + residual)?
                };
                let x_norm = self
                    .ffn_norm
                    .as_ref()
                    .ok_or_else(|| {
                        crate::tensor::Error::msg("ffn_norm missing on serial-attn layer")
                    })?
                    .forward(&x)?;
                (x, x_norm)
            };

            let prof_t_attn_res = prof_t_attn.map(|_| {
                let _ = x.device().synchronize();
                std::time::Instant::now()
            });

            let residual_ffn = &x;
            let has_moe = self.moe.is_some();
            let combined = if has_moe {
                let moe_ref = self.moe.as_ref().unwrap();
                if moe_ref.router_prenormed {
                    // Standard MoE (granitemoe): router + experts both read the single
                    // ffn_norm'd hidden; no dense shared FFN, no gemma4 sandwich norms.
                    let is_prefill = x_norm.dim(1).map(|s| s > 1).unwrap_or(false);
                    let mut m = moe_ref.forward(
                        &x_norm_ffn,
                        residual_ffn,
                        self.embedding_length_for_moe,
                        self.ffn_norm_eps,
                        is_prefill,
                    )?;
                    // Granite scales the MoE/FFN output by residual_multiplier before
                    // the residual add (same as the dense else-branch below).
                    if let Some(s) = rs {
                        m = (m * s)?;
                    }
                    m
                } else {
                    // Gemma4-MoE: dense shared FFN + 128-expert MoE, summed.
                    // Per llama.cpp gemma4-iswa.cpp:
                    //   cur_mlp = post_ffw_norm_1(GELU_FFN(ffn_norm(x_attn)))
                    //   moe_in  = pre_ffw_norm_2(x_attn)
                    //   cur_moe = post_ffw_norm_2(MoE(moe_in, gate(rms_norm(x_attn) ...)))
                    //   cur     = cur_mlp + cur_moe
                    let cur_mlp = self.forward_ffn(&x_norm_ffn)?;
                    let moe_in = if let Some(n) = self.pre_ffw_norm_2.as_ref() {
                        n.forward(residual_ffn)?
                    } else {
                        residual_ffn.clone()
                    };
                    let moe = self.moe.as_ref().unwrap();
                    let is_prefill = x_norm.dim(1).map(|s| s > 1).unwrap_or(false);
                    let mut cur_moe = moe.forward(
                        &moe_in,
                        residual_ffn,
                        self.embedding_length_for_moe,
                        self.ffn_norm_eps,
                        is_prefill,
                    )?;
                    if cur_moe.dtype() != cur_mlp.dtype() {
                        cur_moe = cur_moe.to_dtype(cur_mlp.dtype())?;
                    }
                    let cur_mlp = if let Some(n) = self.post_ffw_norm_1.as_ref() {
                        n.forward(&cur_mlp)?
                    } else {
                        cur_mlp
                    };
                    let cur_moe = if let Some(n) = self.post_ffw_norm_2.as_ref() {
                        n.forward(&cur_moe)?
                    } else {
                        cur_moe
                    };
                    (cur_mlp + cur_moe)?
                }
            } else {
                let mut ffn_out = self.forward_ffn(&x_norm_ffn)?;
                if let Some(s) = rs {
                    ffn_out = (ffn_out * s)?;
                }
                ffn_out
            };
            let prof_t_ffn = prof_t_attn_res.map(|_| {
                let _ = combined.device().synchronize();
                std::time::Instant::now()
            });
            let x = if let Some(post_norm) = &self.post_ffn_norm {
                // 1-launch fuse `rmsnorm(combined) + residual_ffn` - gated to
                // **decode shapes (seq=1)**. Prefill's T can reach 1024+,
                // making the new output buffer 10 MB/layer x 30 layers = 300 MB
                // - enough to OOM gemma4:latest on the layer-31 boundary
                // device (verified empirically). At seq=1 the
                // allocation is 10 KB/layer, negligible. Decode-only gating
                // captures the perf target (per project_gemma4_latest_medium_profile)
                // without the prefill memory cost.
                let is_decode = combined.dim(1).map(|d| d == 1).unwrap_or(false);
                #[cfg(feature = "cuda")]
                {
                    if is_decode
                        && combined.dtype() == DType::F32
                        && residual_ffn.dtype() == DType::F32
                    {
                        let raw = crate::inference::kernel::fused::fused_rmsnorm_then_add(
                            &combined,
                            post_norm.weight(),
                            residual_ffn,
                            self.ffn_norm_eps,
                        )?;
                        // route through stable buffer for graph capture.
                        if let Some(buf) = self.graph_post_ffn_norm_buffer.as_ref() {
                            let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                            let raw_d = raw.dim(crate::tensor::D::Minus1).unwrap_or(0);
                            if buf_d == raw_d && buf_d > 0 {
                                buf.slice_set(&raw, 0, 0)?;
                                buf.clone()
                            } else {
                                raw
                            }
                        } else {
                            raw
                        }
                    } else {
                        (post_norm.forward(&combined)? + residual_ffn)?
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = is_decode;
                    (post_norm.forward(&combined)? + residual_ffn)?
                }
            } else {
                (combined + residual_ffn)?
            };
            let prof_t_ffn_res = prof_t_ffn.map(|_| {
                let _ = x.device().synchronize();
                std::time::Instant::now()
            });
            (x, prof_t_attn_res, prof_t_ffn, prof_t_ffn_res)
        };

        // -- PLE block (Gemma4) --------------------------------------------
        // Flow: gate(hidden) -> GELU -> mul(ple_input) -> proj -> norm -> add residual
        if let (Some(gate), Some(proj), Some(norm)) =
            (&self.ple_inp_gate, &self.ple_proj, &self.ple_post_norm)
        {
            if let Some(ple_input) = ple_layer_input {
                let raw_gate = gate.forward(&x)?;
                // Route PLE gate matmul output through stable buffer.
                let ple_state = if x.dim(1).map(|d| d == 1).unwrap_or(false) {
                    if let Some(buf) = self.graph_ple_gate_buffer.as_ref() {
                        let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        let raw_d = raw_gate.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        if buf_d == raw_d && buf_d > 0 {
                            buf.slice_set(&raw_gate, 0, 0)?;
                            buf.clone()
                        } else {
                            raw_gate
                        }
                    } else {
                        raw_gate
                    }
                } else {
                    raw_gate
                };
                // GELU-gated multiplication with per-layer input  -
                // fused single-launch (gelu(tanh) * mul) replaces 2 ops.
                #[cfg(feature = "cuda")]
                let raw_gelu =
                    crate::inference::kernel::fused::fused_gelu_mul(&ple_state, ple_input)?;
                #[cfg(not(feature = "cuda"))]
                let raw_gelu = ple_state.gelu()?.mul(ple_input)?;
                // Route PLE gelu_mul output.
                let ple_state = if x.dim(1).map(|d| d == 1).unwrap_or(false) {
                    if let Some(buf) = self.graph_ple_gelu_buffer.as_ref() {
                        let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        let raw_d = raw_gelu.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        if buf_d == raw_d && buf_d > 0 {
                            buf.slice_set(&raw_gelu, 0, 0)?;
                            buf.clone()
                        } else {
                            raw_gelu
                        }
                    } else {
                        raw_gelu
                    }
                } else {
                    raw_gelu
                };
                let raw_proj = proj.forward(&ple_state)?;
                // Route PLE proj matmul output.
                let ple_state = if x.dim(1).map(|d| d == 1).unwrap_or(false) {
                    if let Some(buf) = self.graph_ple_proj_buffer.as_ref() {
                        let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        let raw_d = raw_proj.dim(crate::tensor::D::Minus1).unwrap_or(0);
                        if buf_d == raw_d && buf_d > 0 {
                            buf.slice_set(&raw_proj, 0, 0)?;
                            buf.clone()
                        } else {
                            raw_proj
                        }
                    } else {
                        raw_proj
                    }
                } else {
                    raw_proj
                };
                // A re-attempt: prior regression here (commit
                // ce4cc04) was attributed to `Tensor::zeros` overhead.
                // With the kernel switched to `Tensor::empty` the alloc
                // cost is a pool slot lookup, so the 1-launch save
                // should net positive. Decode-only gated (same reason
                // as post_ffn_norm at line 3566 - prefill shape would
                // OOM at large T).
                // When both PLE-fuse-conditions AND `ple_output_scale` are
                // present (the common gemma4 case for layers 0..N-1), fold
                // the trailing broadcast_mul into the same kernel pass  -
                // saves one more launch per layer x 30 layers = 30 more
                // launches/token. Otherwise fall back to the rmsnorm+add
                // 2-op fuse (which itself wins over the 3-launch unfused
                // chain).
                #[cfg(feature = "cuda")]
                let (x_after_ple, scale_applied) = {
                    let is_decode = ple_state.dim(1).map(|d| d == 1).unwrap_or(false);
                    let cuda_f32 =
                        is_decode && ple_state.dtype() == DType::F32 && x.dtype() == DType::F32;
                    let (raw_out, applied) = if cuda_f32 {
                        if let Some(scale) = &self.ple_output_scale {
                            if scale.dtype() == DType::F32 {
                                let out = crate::inference::kernel::fused::fused_rmsnorm_add_scale(
                                    &ple_state,
                                    norm.weight(),
                                    &x,
                                    scale,
                                    norm.eps() as f32,
                                )?;
                                (out, true)
                            } else {
                                let out = crate::inference::kernel::fused::fused_rmsnorm_then_add(
                                    &ple_state,
                                    norm.weight(),
                                    &x,
                                    norm.eps() as f32,
                                )?;
                                (out, false)
                            }
                        } else {
                            let out = crate::inference::kernel::fused::fused_rmsnorm_then_add(
                                &ple_state,
                                norm.weight(),
                                &x,
                                norm.eps() as f32,
                            )?;
                            (out, false)
                        }
                    } else {
                        let ple_state = norm.forward(&ple_state)?;
                        ((&x + ple_state)?, false)
                    };
                    // Route PLE final output through stable buffer.
                    let routed = if is_decode {
                        if let Some(buf) = self.graph_ple_final_buffer.as_ref() {
                            let buf_d = buf.dim(crate::tensor::D::Minus1).unwrap_or(0);
                            let raw_d = raw_out.dim(crate::tensor::D::Minus1).unwrap_or(0);
                            if buf_d == raw_d && buf_d > 0 {
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
                    (routed, applied)
                };
                #[cfg(not(feature = "cuda"))]
                let (x_after_ple, scale_applied): (Tensor, bool) = {
                    let ple_state = norm.forward(&ple_state)?;
                    ((&x + ple_state)?, false)
                };
                x = x_after_ple;
                // Layer scale (gemma4 per-layer output scale) is applied
                // here in the PLE-active path. When the fused 3-op kernel
                // ran, scale was already folded in - skip the broadcast_mul.
                if !scale_applied {
                    if let Some(scale) = &self.ple_output_scale {
                        x = x.broadcast_mul(scale)?;
                    }
                }
            } else {
                // PLE inactive (no ple_input): scale still needs applying.
                if let Some(scale) = &self.ple_output_scale {
                    x = x.broadcast_mul(scale)?;
                }
            }
        } else {
            // No PLE block at all (non-gemma4 or gemma4 without PLE): still
            // honor the scale field if present.
            if let Some(scale) = &self.ple_output_scale {
                x = x.broadcast_mul(scale)?;
            }
        }

        if let (Some(t0), Some(t_an), Some(t_a), Some(t_ar), Some(t_f), Some(t_fr)) = (
            prof_t0,
            prof_t_attn_norm,
            prof_t_attn,
            prof_t_attn_res,
            prof_t_ffn,
            prof_t_ffn_res,
        ) {
            use crate::inference::place::layer_perf::stages;
            let attn_norm_us = t_an.duration_since(t0).as_micros() as u64;
            let attn_us = t_a.duration_since(t_an).as_micros() as u64;
            let attn_res_us = t_ar.duration_since(t_a).as_micros() as u64;
            let ffn_us = t_f.duration_since(t_ar).as_micros() as u64;
            let ffn_res_us = t_fr.duration_since(t_f).as_micros() as u64;
            // The router is timed inside the mixture's own forward and reported there;
            // what is left of the block is the expert compute.
            let router_us = self
                .moe
                .as_ref()
                .map(|_| stages::take_router_us())
                .unwrap_or(0);
            stages::add(0, attn_norm_us);
            stages::add(1, attn_us);
            stages::add(2, attn_res_us);
            stages::add(3, router_us);
            stages::add(4, ffn_us.saturating_sub(router_us));
            stages::add(5, ffn_res_us);
            stages::count_call();
            use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
            static AN: AtomicU64 = AtomicU64::new(0);
            static AT: AtomicU64 = AtomicU64::new(0);
            static AR: AtomicU64 = AtomicU64::new(0);
            static FN: AtomicU64 = AtomicU64::new(0);
            static FR: AtomicU64 = AtomicU64::new(0);
            static N: AtomicU64 = AtomicU64::new(0);
            AN.fetch_add(attn_norm_us, Relaxed);
            AT.fetch_add(attn_us, Relaxed);
            AR.fetch_add(attn_res_us, Relaxed);
            FN.fetch_add(ffn_us, Relaxed);
            FR.fetch_add(ffn_res_us, Relaxed);
            let n = N.fetch_add(1, Relaxed) + 1;
            // Print every ~50 layer-calls (≈ 1-2 tokens x 28 layers).
            if n.is_multiple_of(100) {
                tracing::info!("🟦 GH_PROF (sum µs over {} layer-calls): attn_norm={} attn={} attn_res={} ffn={} ffn_res={}",
                    n,
                    AN.load(Relaxed),
                    AT.load(Relaxed),
                    AR.load(Relaxed),
                    FN.load(Relaxed),
                    FR.load(Relaxed),
                );
            }
        }

        Ok(x)
    }
}
