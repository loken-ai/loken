//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

// ------------------------------------------------------------
// Parallel loading helpers
// ------------------------------------------------------------

/// Raw layer tensors loaded from GGUF in parallel (Phase 1).
/// All weight tensors (QTensor); norm weights always on CPU.
struct RawGenericLayer {
    layer_idx: usize,
    device_kind: DeviceKind,
    /// Q output dim (last GGUF dim of attn_q weight), used to derive per-layer head_dim
    q_out_dim: Option<usize>,
    /// Per-layer K output dim, used to derive per-layer n_kv_head when
    /// the model uses an array-valued head_count_kv (Gemma4 26B SWA=8 /
    /// Global=2). Falls back to config.n_kv_head when None.
    k_out_dim: Option<usize>,
    // Split Q/K/V
    attn_q: Option<QTensor>,
    attn_k: Option<QTensor>,
    attn_v: Option<QTensor>,
    // Fused QKV (Phi3)
    attn_qkv: Option<QTensor>,
    // Biases (Qwen2) - CPU QTensors, dequantized in Phase 2
    attn_q_bias: Option<QTensor>,
    attn_k_bias: Option<QTensor>,
    attn_v_bias: Option<QTensor>,
    // QK norms (Gemma3) - CPU QTensors
    attn_q_norm_qt: Option<QTensor>,
    attn_k_norm_qt: Option<QTensor>,
    // Always present
    attn_output: QTensor,
    /// Phi2: attn_output has a bias
    attn_output_bias_qt: Option<QTensor>,
    // Norm weights - always CPU
    attn_norm_qt: Option<QTensor>, // None for post-norm-only archs (OLMo2) -> Identity pre-norm
    /// Phi2/GPT-NeoX: full LayerNorm - bias on attn_norm
    attn_norm_bias_qt: Option<QTensor>,
    post_attn_norm_qt: Option<QTensor>,
    ffn_norm_qt: Option<QTensor>,
    post_ffn_norm_qt: Option<QTensor>,
    // FFN weights
    ffn_gate: Option<QTensor>,
    ffn_up: QTensor,
    /// Phi2: ffn_up has a bias
    ffn_up_bias_qt: Option<QTensor>,
    ffn_down: QTensor,
    /// Phi2: ffn_down has a bias
    ffn_down_bias_qt: Option<QTensor>,
    // PLE (Gemma4)
    ple_inp_gate: Option<QTensor>,
    ple_proj: Option<QTensor>,
    ple_post_norm_qt: Option<QTensor>,
    ple_output_scale_qt: Option<QTensor>,
    // Gemma4-MoE per-layer weights (Some when arch==gemma4 + 26B variant)
    moe_gate_inp: Option<QTensor>,
    moe_gate_inp_scale_qt: Option<QTensor>,
    moe_gate_up_exps: Option<QTensor>,
    moe_down_exps: Option<QTensor>,
    moe_down_exps_scale_qt: Option<QTensor>,
    /// granitemoe/standard MoE: SEPARATE gate/up experts (vs gemma4's fused
    /// `moe_gate_up_exps`). When Some, the build concatenates them into the
    /// fused F32 expert cache and marks the layer as a standard (SiLU,
    /// pre-normed router, no dense shared FFN) MoE.
    moe_gate_exps: Option<QTensor>,
    moe_up_exps: Option<QTensor>,
    pre_ffw_norm_2_qt: Option<QTensor>,
    post_ffw_norm_1_qt: Option<QTensor>,
    post_ffw_norm_2_qt: Option<QTensor>,
}

/// Helper: read a tensor by name from mmap with a Cursor, or return None if absent.
type MmapOwner<'a> = Option<&'a std::sync::Arc<dyn std::any::Any + Send + Sync>>;

fn try_read_tensor(
    tensor_infos: &HashMap<String, gguf_file::TensorInfo>,
    mmap: &[u8],
    tdo: u64,
    name: &str,
    device: &Device,
    owner: MmapOwner,
) -> crate::tensor::Result<Option<QTensor>> {
    match tensor_infos.get(name) {
        Some(ti) => Ok(Some(ti.read_slice_owned(mmap, tdo, device, owner)?)),
        None => Ok(None),
    }
}

fn required_tensor(
    tensor_infos: &HashMap<String, gguf_file::TensorInfo>,
    mmap: &[u8],
    tdo: u64,
    name: &str,
    device: &Device,
    owner: MmapOwner,
) -> crate::tensor::Result<QTensor> {
    tensor_infos
        .get(name)
        .ok_or_else(|| crate::tensor::Error::msg(format!("missing tensor: {name}")))?
        .read_slice_owned(mmap, tdo, device, owner)
}

/// Build a `GenericTransformerLayer` from a raw loaded layer.
fn build_generic_layer(
    raw: RawGenericLayer,
    device: &Device,
    config: &GenericTransformerConfig,
    flags: Arc<GenericLayerFlags>,
    cos: &Tensor,
    sin: &Tensor,
    rope_freq_factors: Option<&Tensor>,
) -> Result<GenericTransformerLayer> {
    let eps = config.rms_norm_eps;
    // Which REPRESENTATION the weights take, not whether a card is touched: on a card
    // a norm scale becomes a dense F32 tensor and its quantised blocks are dropped,
    // and every arm below is a dequantise plus a move. A counted run must land on the
    // same representation or it weighs a model that will not be loaded.
    let on_gpu = device.runs_as_card();

    // Helper: build RmsNorm, placing weights on the layer's device
    let build_norm = |qt: QTensor| -> Result<RmsNorm> {
        if on_gpu {
            let w = qt.dequantize(&Device::Cpu)?.to_device(device)?;
            Ok(RmsNorm::from_tensor(w, eps))
        } else {
            RmsNorm::from_qtensor(qt, eps)
        }
    };

    let build_norm_opt =
        |qt: Option<QTensor>| -> Result<Option<RmsNorm>> { qt.map(build_norm).transpose() };

    // Attention weights. Some GGUFs store specific weights (e.g. qwen3
    // keeps one per-layer tensor at F16) as non-quantized dtypes. That
    // routes them through the reference F32 cublas gemv, which is memory-bound
    // on a path that our fast `mul_mat_vec_q*_K_q8_1` kernel dominates.
    // Requantize them to Q6_K at load time so every per-layer matmul
    // takes the fast path.
    let to_qmatmul = |qt: Option<QTensor>| -> Result<Option<QMatMul>> {
        let Some(qt) = qt else {
            return Ok(None);
        };
        let dtype = qt.dtype();
        let needs_requant = matches!(
            dtype,
            crate::tensor::quantized::GgmlDType::F32
                | crate::tensor::quantized::GgmlDType::F16
                | crate::tensor::quantized::GgmlDType::BF16
        );
        // Requant F16/BF16/F32 weights to Q4K on BOTH cuda and cpu. On GPU it
        // enables the QKV/FFN byte-concat fusions; on CPU it's a decode win in
        // its own right - the reference QMatMul runs a dense F32 GEMM for F16-stored
        // weights, so on a memory-bound CPU those matmuls read 4 B/param instead
        // of Q4K's ~0.6 B. Requanting also makes CPU numerics match GPU (which
        // already requants to Q4K). The byte-concat *fusion* stays GPU-only.
        if needs_requant {
            let shape = qt.shape().clone();
            let dequantized = qt.dequantize(&Device::Cpu)?;
            // Use Q4K (not Q6K) so that all per-layer weights share the
            // same quant type, enabling QKV and FFN gate+up fusions.
            let qt_q4 = QTensor::quantize_onto(
                &dequantized,
                crate::tensor::quantized::GgmlDType::Q4K,
                device,
            )?;
            tracing::debug!("  Requantized {:?} weight from {:?} -> Q4K", shape, dtype);
            Ok(Some(QMatMul::from_qtensor(qt_q4)?))
        } else {
            Ok(Some(QMatMul::from_qtensor(qt)?))
        }
    };
    // QKV fusion: if Q, K, V all exist with the same quantization type
    // (after requant), concatenate raw bytes into [q_out+k_out+v_out, hidden].
    // Saves 2 matmul launches + 2 q8_1 quantize launches per layer.
    //
    // Helper: requant a QTensor to Q4K if it's F16/F32/BF16 (returns QTensor, not QMatMul).
    let requant_qt = |qt: QTensor| -> Result<QTensor> {
        let dtype = qt.dtype();
        let needs = matches!(
            dtype,
            crate::tensor::quantized::GgmlDType::F32
                | crate::tensor::quantized::GgmlDType::F16
                | crate::tensor::quantized::GgmlDType::BF16
        );
        if needs {
            let shape = qt.shape().clone();
            let dequantized = qt.dequantize(&Device::Cpu)?;
            let qt_new = QTensor::quantize_onto(
                &dequantized,
                crate::tensor::quantized::GgmlDType::Q4K,
                device,
            )?;
            tracing::debug!("  Requantized {:?} weight from {:?} -> Q4K", shape, dtype);
            Ok(qt_new)
        } else {
            Ok(qt)
        }
    };
    let (attn_q, attn_k, attn_v, attn_qkv) = if raw.attn_qkv.is_some() || raw.attn_q.is_none() {
        // Already fused in GGUF or missing - pass through
        (
            to_qmatmul(raw.attn_q)?,
            to_qmatmul(raw.attn_k)?,
            to_qmatmul(raw.attn_v)?,
            to_qmatmul(raw.attn_qkv)?,
        )
    } else if raw.attn_v.is_none() {
        // Gemma4 26B Global layers: attn_v missing (AttentionKEqV).
        // Skip QKV byte-concat - Q,K projected separately, V reuses K
        // at forward time.
        (
            Some(QMatMul::from_qtensor(requant_qt(raw.attn_q.unwrap())?)?),
            Some(QMatMul::from_qtensor(requant_qt(raw.attn_k.unwrap())?)?),
            None,
            None,
        )
    } else {
        let q = requant_qt(raw.attn_q.unwrap())?;
        let k = requant_qt(raw.attn_k.unwrap())?;
        let v = requant_qt(raw.attn_v.unwrap())?;
        // V is often stored at HIGHER precision than Q/K (V=Q6K, Q/K=Q4K in
        // most K-quant blobs) - the quantizer picks that on purpose because the
        // attention output is a weighted sum of V rows and V error passes
        // straight through. Requantizing V down to Q/K's dtype to enable the
        // byte-concat is NOT acceptable: dequant->requant Q6K->Q4K measured a
        // 4x weight-error blowup (rel_rms 2% -> 8% vs the f32 oracle on real
        // checkpoints), enough to flip greedy decoding on error-amplifying
        // arches (granite-moe went fully incoherent on every device). Mixed
        // dtypes therefore keep SEPARATE Q/K/V projections; the fused path
        // below only engages when all three already share one dtype, which
        // keeps the fused bytes bit-identical to the checkpoint.
        // QKV byte-concat: combine Q, K, V into one fused matmul tensor.
        // We can do this even when biases exist (qwen2/phi2) - biases get
        // applied after the fused matmul on the split parts (already
        // implemented in the runtime path at lines ~870-892).
        //
        // For mixed-AttentionKEqV models (gemma4 26B), each layer is
        // independently classified by `raw.attn_v.is_none()` upstream
        // (case 2 catches the Global layers). Layers reaching this
        // branch (case 3) have all three weights, so they can fuse
        // safely - the fused-QKV path for SWA layers unblocks the APQ
        // fast-decode kernel which fuses RMSNorm + RoPE + QKV split.
        // Device-neutral: the fused-QKV runtime split (forward_attn_inner,
        // `attn_q.is_none()` branch) is pure narrows and works identically on
        // CPU and CUDA. On CPU it replaces 3 separate q/k/v GEMVs with one
        // [q;k;v] GEMV per decode token (1 activation-quantize + 1 rayon
        // region instead of 3) - the dispatch-overhead lever.
        let can_fuse = q.dtype() == k.dtype()
            && k.dtype() == v.dtype()
            && q.shape().dims().len() == 2
            && q.shape().dims()[1] == k.shape().dims()[1];
        if can_fuse {
            let dtype = q.dtype();
            let hidden = q.shape().dims()[1];
            let q_out = q.shape().dims()[0];
            let k_out = k.shape().dims()[0];
            let v_out = v.shape().dims()[0];
            // Borrow the block bytes (Cow - zero-copy for mmap-view weights) and copy
            // each ONCE straight into the fused buffer. The old `.into_owned()` cloned
            // every weight first (a wasted copy of q+k+v per layer, ~3.6 GB fleet-wide).
            let q_bytes = q.data()?;
            let k_bytes = k.data()?;
            let v_bytes = v.data()?;
            let total = q_bytes.len() + k_bytes.len() + v_bytes.len();
            let n_u32 = total.div_ceil(4);
            let mut buf = vec![0u32; n_u32];
            let buf_bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_u32 * 4) };
            buf_bytes[..q_bytes.len()].copy_from_slice(&q_bytes);
            buf_bytes[q_bytes.len()..q_bytes.len() + k_bytes.len()].copy_from_slice(&k_bytes);
            buf_bytes[q_bytes.len() + k_bytes.len()..total].copy_from_slice(&v_bytes);
            let combined_slice: &[u8] =
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, total) };
            let fused_storage = crate::tensor::quantized::QStorage::from_data(
                std::borrow::Cow::Borrowed(combined_slice),
                device,
                dtype,
            )?;
            drop(buf);
            let fused_shape = crate::tensor::Shape::from((q_out + k_out + v_out, hidden));
            let fused_qt = QTensor::new(fused_storage, fused_shape)?;
            tracing::debug!(
                "  Fused QKV ({:?}): Q[{}]+K[{}]+V[{}] -> [{}x{}]",
                dtype,
                q_out,
                k_out,
                v_out,
                q_out + k_out + v_out,
                hidden
            );
            (None, None, None, Some(QMatMul::from_qtensor(fused_qt)?))
        } else {
            (
                Some(QMatMul::from_qtensor(q)?),
                Some(QMatMul::from_qtensor(k)?),
                Some(QMatMul::from_qtensor(v)?),
                None,
            )
        }
    };

    // Biases: dequantize to F32 Tensor on the layer device
    let load_bias = |qt: Option<QTensor>| -> Result<Option<Tensor>> {
        qt.map(|q| {
            let t = q.dequantize(&Device::Cpu)?;
            if on_gpu {
                t.to_device(device)
            } else {
                Ok(t)
            }
        })
        .transpose()
    };
    let attn_q_bias = load_bias(raw.attn_q_bias)?;
    let attn_k_bias = load_bias(raw.attn_k_bias)?;
    let attn_v_bias = load_bias(raw.attn_v_bias)?;

    // Fused QKV bias: concat of Q||K||V biases for the fused-QKV runtime
    // path (1 broadcast_add per layer instead of 3). All three biases must
    // be present and 1-D for the concat to make sense; fall back to None
    // otherwise (runtime then uses the per-component biases as before).
    let attn_qkv_bias: Option<Tensor> = match (
        attn_q_bias.as_ref(),
        attn_k_bias.as_ref(),
        attn_v_bias.as_ref(),
    ) {
        (Some(bq), Some(bk), Some(bv))
            if bq.dims().len() == 1
                && bk.dims().len() == 1
                && bv.dims().len() == 1
                && bq.dtype() == bk.dtype()
                && bk.dtype() == bv.dtype() =>
        {
            Tensor::cat(&[bq, bk, bv], 0).ok()
        }
        _ => None,
    };

    // QK norms
    let attn_q_norm = build_norm_opt(raw.attn_q_norm_qt)?;
    let attn_k_norm = build_norm_opt(raw.attn_k_norm_qt)?;

    let attn_output = to_qmatmul(Some(raw.attn_output))?.unwrap();

    // FFN gate+up fusion: if both are present, share the same quantization,
    // and the activation is SiLU (not GELU), concatenate their raw quantized
    // bytes into a single [2*intermediate, hidden] QTensor. This halves the
    // matmul launch count (and q8_1 activation quantize count) on the FFN
    // path, which together saves ~0.3 ms/tok on qwen3 decode. The fusion is
    // ZERO-copy on the weight data (no re-quantization) so quality is
    // preserved exactly.
    //
    // The runtime fused path is in `forward_ffn` - it checks
    // `self.ffn_gate.is_none()` to pick the fused branch, so we just null
    // out ffn_gate after fusion.
    let (ffn_gate_final, ffn_up_final) = {
        let gate_opt = raw.ffn_gate;
        let up = raw.ffn_up;
        // Load-time gate+up byte-concat: sets ffn_gate to None and drives
        // forward_ffn through the concat matmul + narrow + fused_silu_mul
        // (or fused_gelu_mul) path. Always-on when dtypes match - the
        // alternative `moe_gemm_gguf_gate_up_silu_mul` runtime fusion only
        // exists for MoE expert weights, so for the dense FFN path the
        // concat is unconditionally faster.
        // Device-neutral: `forward_ffn`'s fused branch (`ffn_gate.is_none()`)
        // does one [gate;up] GEMV then a narrow + silu/gelu.mul split - works
        // identically on CPU and CUDA. On CPU this halves per-token FFN decode
        // dispatch (1 activation-quantize + 1 rayon region instead of 2), the
        // same dispatch-overhead lever as the fused QKV above.
        let can_fuse = gate_opt.as_ref().is_some_and(|g| {
            g.dtype() == up.dtype()
                && matches!(
                    g.dtype(),
                    crate::tensor::quantized::GgmlDType::Q4K
                        | crate::tensor::quantized::GgmlDType::Q5K
                        | crate::tensor::quantized::GgmlDType::Q6K
                        | crate::tensor::quantized::GgmlDType::Q8_0
                        | crate::tensor::quantized::GgmlDType::Q4_0
                )
        });
        if can_fuse {
            let gate = gate_opt.unwrap();
            let gate_shape = gate.shape().dims().to_vec();
            let up_shape = up.shape().dims().to_vec();
            if gate_shape == up_shape && gate_shape.len() == 2 {
                let dtype = gate.dtype();
                let hidden = gate_shape[1];
                let inter = gate_shape[0];
                // Concatenate the raw quantized bytes on CPU, then upload
                // to GPU via QStorage::from_data. The data must be aligned
                // to the block struct's alignment (typically 2 for Q4_K).
                // Vec<u8> from data() only guarantees 1-byte alignment, so
                // allocate as Vec<u32> (4-byte aligned) and copy into it.
                // Borrow (zero-copy for mmap-view weights) + copy once into the fused
                // buffer, instead of cloning gate+up first (the FFN is the largest
                // per-layer weight - the extra copy was ~73 MB/layer).
                let gate_bytes = gate.data()?;
                let up_bytes = up.data()?;
                let total = gate_bytes.len() + up_bytes.len();
                let n_u32 = total.div_ceil(4);
                let mut buf = vec![0u32; n_u32];
                let buf_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_u32 * 4)
                };
                buf_bytes[..gate_bytes.len()].copy_from_slice(&gate_bytes);
                buf_bytes[gate_bytes.len()..total].copy_from_slice(&up_bytes);
                // from_data reads via as_t_slice which needs alignment;
                // buf's u32 backing guarantees >=2-byte alignment.
                let combined_slice: &[u8] =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, total) };
                let fused_storage = crate::tensor::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(combined_slice),
                    device,
                    dtype,
                )?;
                drop(buf); // safe: from_data already copied to GPU
                let fused_shape = crate::tensor::Shape::from((2 * inter, hidden));
                let fused_qt = QTensor::new(fused_storage, fused_shape)?;
                tracing::debug!(
                    "  Fused ffn_gate+ffn_up ({:?}): [{}, {}] -> [{}, {}]",
                    dtype,
                    inter,
                    hidden,
                    2 * inter,
                    hidden
                );
                (None, QMatMul::from_qtensor(fused_qt)?)
            } else {
                // Shape mismatch - fall back to separate.
                (
                    Some(QMatMul::from_qtensor(gate)?),
                    to_qmatmul(Some(up))?.unwrap(),
                )
            }
        } else {
            (to_qmatmul(gate_opt)?, to_qmatmul(Some(up))?.unwrap())
        }
    };
    let ffn_gate = ffn_gate_final;
    let ffn_up = ffn_up_final;
    let ffn_down = to_qmatmul(Some(raw.ffn_down))?.unwrap();

    // Build attn_norm - LayerNorm-with-bias for phi2/gpt-neox, RmsNorm
    // for everyone else. Detected via `attn_norm_bias_qt` presence.
    let attn_norm: WeightedNorm = match raw.attn_norm_qt {
        // Post-norm-only arch (OLMo2): no pre-norm tensor -> identity pre-norm; the
        // held ones tensor is only for weight()/device probes (fused pre-norm path
        // disabled below), and post_attn_norm normalises the sub-layer OUTPUT.
        None => WeightedNorm::Identity(crate::tensor::Tensor::ones(
            (1usize,),
            crate::tensor::DType::F32,
            device,
        )?),
        Some(qt) => {
            if let Some(bias_qt) = raw.attn_norm_bias_qt {
                let weight_t = if on_gpu {
                    qt.dequantize(&Device::Cpu)?.to_device(device)?
                } else {
                    qt.dequantize(device)?
                };
                let bias_t = if on_gpu {
                    bias_qt.dequantize(&Device::Cpu)?.to_device(device)?
                } else {
                    bias_qt.dequantize(device)?
                };
                WeightedNorm::Layer(LayerNorm::new(weight_t, Some(bias_t), eps as f32))
            } else {
                WeightedNorm::Rms(build_norm(qt)?)
            }
        }
    };
    // Phase B: pre-cast attn_norm weight to F32 so the fused
    // rms_norm_then_qmatmul_bf16 kernel can read it directly without
    // a per-call BF16->F32 cast launch (which would offset the launch
    // saved by the fusion). Only built when the layer is eligible.
    let attn_norm_weight_f32: Option<Tensor> =
        if flags.fused_qkv_norm_eligible && !matches!(attn_norm, WeightedNorm::Identity(_)) {
            let w = attn_norm.weight();
            if w.dtype() == crate::tensor::DType::F32 {
                Some(w.clone())
            } else {
                Some(w.to_dtype(crate::tensor::DType::F32)?)
            }
        } else {
            None
        };
    let post_attn_norm = build_norm_opt(raw.post_attn_norm_qt)?;
    // Phi2 biases on output projections (move to layer device)
    let bias_to_device = |qt: Option<QTensor>| -> Result<Option<Tensor>> {
        match qt {
            Some(b) => Ok(Some(if on_gpu {
                b.dequantize(&Device::Cpu)?.to_device(device)?
            } else {
                b.dequantize(device)?
            })),
            None => Ok(None),
        }
    };
    let attn_output_bias = bias_to_device(raw.attn_output_bias_qt)?;
    let ffn_up_bias = bias_to_device(raw.ffn_up_bias_qt)?;
    let ffn_down_bias = bias_to_device(raw.ffn_down_bias_qt)?;
    // Extract raw weight for fused add+rmsnorm kernel before build_norm consumes QTensor
    let ffn_norm_weight_for_fused = match &raw.ffn_norm_qt {
        Some(qt) if on_gpu => Some(qt.dequantize(&Device::Cpu)?.to_device(device)?),
        _ => None,
    };
    let ffn_norm = match raw.ffn_norm_qt {
        Some(qt) => Some(WeightedNorm::Rms(build_norm(qt)?)),
        // OLMo2 post-norm-only: no pre-ffn-norm tensor but post_ffw_norm IS present ->
        // identity pre-norm (the FFN sees raw h; post_ffn_norm normalises its output).
        None if raw.post_ffn_norm_qt.is_some() => Some(WeightedNorm::Identity(
            crate::tensor::Tensor::ones((1usize,), crate::tensor::DType::F32, device)?,
        )),
        None => None,
    };
    let post_ffn_norm = build_norm_opt(raw.post_ffn_norm_qt)?;

    // Ensure cos/sin are on the layer's device (critical for multi-GPU)
    let cos = if cos.device().same_device(device) {
        cos.clone()
    } else {
        tracing::debug!(
            "RoPE device fix: layer {} cos on {:?} -> {:?}",
            raw.layer_idx,
            cos.device(),
            device
        );
        cos.to_device(device)?
    };
    let sin = if sin.device().same_device(device) {
        sin.clone()
    } else {
        tracing::debug!(
            "RoPE device fix: layer {} sin on {:?} -> {:?}",
            raw.layer_idx,
            sin.device(),
            device
        );
        sin.to_device(device)?
    };

    let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

    // Per-layer head_dim: derive from Q weight output dim if available.
    // Gemma4 has per-layer varying head_dim (SWA=256, full=512).
    let head_dim = match raw.q_out_dim {
        Some(q_dim) if q_dim > 0 && config.n_head > 0 => q_dim / config.n_head,
        _ => config.head_dim,
    };
    // Per-layer n_kv_head: derive from K weight output dim. Gemma4 26B
    // has [8,8,8,8,8,2,...] head_count_kv arrays - each layer's K
    // projection encodes its actual GQA factor.
    let n_kv_head_layer = match raw.k_out_dim {
        Some(k_dim) if k_dim > 0 && head_dim > 0 => k_dim / head_dim,
        _ => config.n_kv_head,
    };

    // Partial RoPE: e.g., StableLM uses 16 of 64 dims. 0 = full head_dim.
    let rope_dim = config
        .rope_dim
        .filter(|&d| d > 0 && d < head_dim)
        .unwrap_or(0);

    // Recompute RoPE when head_dim or rope_dim differs from the config-level precomputed tables.
    let effective_rope_dim = if rope_dim > 0 { rope_dim } else { head_dim };
    let (cos, sin) = if head_dim != config.head_dim || rope_dim > 0 {
        let rope_base = if head_dim < config.head_dim {
            // SWA layers (Gemma4) use a different RoPE freq base
            config.rope_freq_base_swa.unwrap_or(config.rope_freq_base)
        } else {
            config.rope_freq_base
        };
        // YaRN applies to full-head-dim layers; SWA/partial layers (head_dim<config)
        // keep plain rope (their lower freq base already shortens their context).
        let yarn = if head_dim < config.head_dim {
            None
        } else {
            config.yarn
        };
        let (c, s) = crate::inference::generic_transformer::rope::precomput_freqs_cis_yarn(
            effective_rope_dim,
            rope_base,
            yarn,
            config
                .effective_max_context()
                .max(crate::inference::generic_transformer::rope::ROPE_TABLE_SEQ_LEN),
            device,
        )?;
        (c, s)
    } else {
        (cos.clone(), sin.clone())
    };
    // (Tried: baking rope_freq_factors into cos/sin at load time to
    // skip per-call recompute. Math should be equivalent but the baked
    // tables produced wrong output for gemma4 - reverted. Likely
    // dtype/precision difference; investigate later.)
    let _rope_freq_factors_baked = false;

    // Build MoE weights when present (Gemma4 26B). All expert tensors
    // live on the layer's target device; the small per-channel scales
    // dequantize to F32 there too.
    //
    // the reference `moe_gemm_gguf` only accepts {Q2K, Q3K, Q4K, Q5K, Q6K, Q8_0}
    // weights. Q4_K_M GGUFs mix Q5_0/Q4_0 on some down projections, which
    // would otherwise abort. Requantize unsupported types up to Q8_0 at load
    // time (small VRAM penalty, kept on CPU then quantize_onto target dev).
    fn coerce_moe_weight(
        qt: crate::tensor::quantized::QTensor,
        device: &Device,
    ) -> Result<crate::tensor::quantized::QTensor> {
        use crate::tensor::quantized::GgmlDType;
        let supported = matches!(
            qt.dtype(),
            GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8_0
        );
        if supported {
            return Ok(qt);
        }
        let dequant = qt.dequantize(&Device::Cpu)?;
        crate::tensor::quantized::QTensor::quantize_onto(&dequant, GgmlDType::Q8_0, device)
    }

    let moe = if let (Some(gi), Some(des)) = (raw.moe_gate_inp, raw.moe_down_exps) {
        // gate||up experts: fused (gemma4-MoE `ffn_gate_up_exps`) OR separate
        // gate/up (standard MoE like granitemoe). For separate, concat the
        // dequantized gate then up experts along the output (ffn) dim into the
        // fused [n_exp, 2*ffn, hidden] layout, then requantize to Q8.
        let (gus, fused_f32, router_prenormed, use_silu, gate_sep, up_sep) =
            match (raw.moe_gate_up_exps, raw.moe_gate_exps, raw.moe_up_exps) {
                (Some(gus_fused), _, _) => (
                    coerce_moe_weight(gus_fused, device)?,
                    None,
                    false,
                    false,
                    None,
                    None,
                ),
                (None, Some(ge), Some(ue)) => {
                    let ge_f = ge.dequantize(&Device::Cpu)?;
                    let ue_f = ue.dequantize(&Device::Cpu)?;
                    let fused = Tensor::cat(&[&ge_f, &ue_f], 1)?; // [n_exp, 2*ffn, hidden] (gate;up)
                    let gus = crate::tensor::quantized::QTensor::quantize_onto(
                        &fused,
                        crate::tensor::quantized::GgmlDType::Q8_0,
                        device,
                    )?;
                    // Keep the separate gate/up experts quantized on-device for the
                    // GPU SiLU kernel; the exact F32 fused feeds the CPU cache.
                    let gate_q = std::sync::Arc::new(coerce_moe_weight(ge, device)?);
                    let up_q = std::sync::Arc::new(coerce_moe_weight(ue, device)?);
                    (gus, Some(fused), true, true, Some(gate_q), Some(up_q))
                }
                _ => {
                    return Err(crate::tensor::Error::msg(
                        "MoE layer has neither fused nor separate gate/up experts",
                    ))
                }
            };
        let gate_inp_scale = match raw.moe_gate_inp_scale_qt {
            Some(qt) => {
                let t = qt.dequantize(&Device::Cpu)?;
                if on_gpu {
                    t.to_device(device)?
                } else {
                    t
                }
            }
            // Default scale = 1.0 if missing
            None => Tensor::ones(config.embedding_length, crate::tensor::DType::F32, device)?,
        };
        // Fold 1/sqrt(n_embd) into gate_inp_scale at load time so the
        // per-token MoE router pipeline drops a kernel launch (was a
        // separate broadcast_mul of a scalar). Mathematically identical;
        // saves one launch per layer per token. (Standard MoE routers are
        // pre-normed -> gate_inp_scale unused, but harmless to build.)
        let inv_sqrt_n_embd = 1.0f64 / (config.embedding_length as f64).sqrt();
        let gate_inp_scale = (gate_inp_scale * inv_sqrt_n_embd)?;
        let down_exps_scale = match raw.moe_down_exps_scale_qt {
            Some(qt) => {
                let t = qt.dequantize(&Device::Cpu)?;
                if on_gpu {
                    t.to_device(device)?
                } else {
                    t
                }
            }
            None => Tensor::ones(flags.n_experts, crate::tensor::DType::F32, device)?,
        };
        let des = coerce_moe_weight(des, device)?;
        // CPU layers: pre-dequantize the expert stacks once into BF16 - the
        // per-call dequant cost (≈3 GB allocation per call) would dominate.
        // BF16 (2 bytes/elem) instead of F32 (4 bytes/elem) halves the RAM
        // footprint per CPU layer (≈1.5 GB instead of 3 GB), keeping the
        // total cache footprint manageable when the heterogeneous planner
        // offloads many layers (e.g., 14 layers x 3 GB = 42 GB -> 21 GB).
        let (gus_f32, des_f32) = if !on_gpu || use_silu {
            let g = match &fused_f32 {
                Some(f) => f.to_dtype(crate::tensor::DType::BF16)?,
                None => gus
                    .dequantize(&Device::Cpu)?
                    .to_dtype(crate::tensor::DType::BF16)?,
            };
            let d = des
                .dequantize(&Device::Cpu)?
                .to_dtype(crate::tensor::DType::BF16)?;
            (Some(std::sync::Arc::new(g)), Some(std::sync::Arc::new(d)))
        } else {
            (None, None)
        };
        Some(MoeWeights {
            gate_inp: crate::tensor::quantized::QMatMul::from_qtensor(gi)?,
            gate_inp_scale,
            gate_up_exps: std::sync::Arc::new(gus),
            down_exps: std::sync::Arc::new(des),
            gate_exps_sep: gate_sep,
            up_exps_sep: up_sep,
            down_exps_scale,
            n_experts: flags.n_experts,
            n_experts_used: flags.n_experts_used,
            expert_ffn_dim: flags.expert_ffn_dim,
            router_prenormed,
            use_silu,
            // Real activation, decoupled from the use_silu layout flag: GELU only for
            // gemma4-MoE (flags.use_gelu, keyed on layer_output_scale.weight).
            gelu: flags.use_gelu,
            // OLMoE uses raw softmax-over-all top-K weights (no renorm); all other
            // MoE arches renormalize the top-K to sum to 1.
            norm_topk: config.arch != "olmoe",
            gate_up_exps_f32: gus_f32,
            down_exps_f32: des_f32,
        })
    } else {
        None
    };
    let pre_ffw_norm_2 = build_norm_opt(raw.pre_ffw_norm_2_qt)?;
    let post_ffw_norm_1 = build_norm_opt(raw.post_ffw_norm_1_qt)?;
    let post_ffw_norm_2 = build_norm_opt(raw.post_ffw_norm_2_qt)?;

    // gemma4: pre-allocate the unit-weight tensor used by the V-RMS-norm
    // pass (head_dim ones, F32). Built on the LAYER's device - CPU or GPU.
    // The V-norm is part of the gemma4 arch and the forward applies it via
    // `crate::tensor::ops::rms_norm` which runs on CPU too, so this must NOT be
    // gated on `on_gpu`: a gemma4 layer placed on CPU (multi-GPU+CPU hybrid
    // OR pure-CPU fallback) would otherwise have `attn_v_norm_ones = None`,
    // silently SKIP the V-norm, and produce garbage attention on that layer
    // (the multi-device incoherence - single-GPU was fine because every
    // layer had it).
    let attn_v_norm_ones = if flags.use_gelu {
        Some(Tensor::ones(head_dim, crate::tensor::DType::F32, device)?)
    } else {
        None
    };

    Ok(GenericTransformerLayer {
        attn_q,
        attn_k,
        attn_v,
        attn_qkv,
        attn_q_bias,
        attn_k_bias,
        attn_v_bias,
        attn_qkv_bias,
        attn_q_norm,
        attn_k_norm,
        attn_v_norm_ones,
        attn_output,
        attn_output_bias,
        // SmolLM3 NoPE: RoPE is skipped on every 4th layer ((idx+1)%4==0).
        no_rope: config.arch == "smollm3" && (raw.layer_idx + 1) % 4 == 0,
        attn_norm,
        attn_norm_weight_f32,
        post_attn_norm,
        ffn_gate,
        ffn_up,
        ffn_up_bias,
        ffn_down,
        ffn_down_bias,
        moe,
        pre_ffw_norm_2,
        post_ffw_norm_1,
        post_ffw_norm_2,
        embedding_length_for_moe: config.embedding_length,
        ffn_norm,
        ffn_norm_weight: ffn_norm_weight_for_fused,
        ffn_norm_eps: eps as f32,
        post_ffn_norm,
        ple_inp_gate: raw.ple_inp_gate.map(QMatMul::from_qtensor).transpose()?,
        ple_proj: raw.ple_proj.map(QMatMul::from_qtensor).transpose()?,
        ple_post_norm: build_norm_opt(raw.ple_post_norm_qt)?,
        ple_output_scale: raw
            .ple_output_scale_qt
            .map(|qt| {
                let t = qt.dequantize(&Device::Cpu)?;
                if on_gpu {
                    t.to_device(device)
                } else {
                    Ok(t)
                }
            })
            .transpose()?,
        // KV cache initial sizing.
        // For Q4/Q8 archs the F16 SpecKvCache buffer is NEVER realized (decode
        // runs on the quantized cache); only `max_seq_len` is read for graph-mode
        // `padded_mask` sizing, which MUST equal effective_max_context - too small
        // fires the "pos >= mask_width" graceful-fallback mid-decode (verified
        // cause of deepcoder/qwen3 long-decode regressions).
        // For kv_quant==Off (gemma4) the buffer IS filled. Pre-allocating
        // effective_max_context (e.g. 32768) per GLOBAL layer overran the
        // planner's KV reserve (sized to the request context) -> CUDA OOM on
        // VRAM-tight 2-GPU variants (gemma4:31b 2.5K). Start at the lazy cap and
        // grow on demand (append doubles) - matching the Q4/Q8 lazy caches and
        // the reserve's planning context; the adaptive prefill + per-layer
        // divert cover any genuine pressure. So the loader spills/grows rather
        // than OOMing.
        kv_cache: {
            use crate::inference::cache::KV_WORKING_WINDOW_TOKENS as LAZY_KV_INITIAL_CAP;
            use crate::inference::engine::llm_engine::KvQuant;
            let full = config.effective_max_context().max(512);
            let init_cap = if matches!(config.kv_quant, KvQuant::Off) {
                full.min(LAZY_KV_INITIAL_CAP)
            } else {
                full
            };
            let mut c = SpecKvCache::new(init_cap);
            // gemma4 bounded sliding-window attention: SWA layers only attend within
            // `sliding_window`, so the cache need keep only the last
            // `window + prefill_chunk` keys - bounding it lets the F16 KV path
            // stay VRAM-cheap at long context (where Q8/Q4 KV degenerated for
            // gemma4's tiny K-norm). Exact: the bounded prefill mask matches.
            // (set_window overrides max_seq_len; init_cap above governs the
            // GLOBAL, non-windowed layers.) Same SWA classification as the
            // per-layer `sliding_window` below.
            let is_swa = match config
                .swa_pattern
                .as_ref()
                .and_then(|p| p.get(raw.layer_idx))
            {
                Some(&s) => s,
                None => head_dim < config.head_dim,
            };
            if is_swa {
                if let Some(w) = config.sliding_window {
                    c.set_window(
                        w,
                        crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS,
                    );
                }
            }
            c
        },
        cpu_q8_kv: None,
        cpu_f16_kv: None,
        cpu_decode_arena: None,
        cpu_norm_cache: None,
        #[cfg(feature = "cuda")]
        q8_kv_cache: {
            use crate::inference::engine::llm_engine::KvQuant;
            // Enable Q8 KV when config asks for it OR when config asks for
            // Q4 but Q4 isn't applicable for this model (no qkv_bias /
            // n_layers > 50). Without this fallback, qwen3 base on
            // kv_quant=q4 ends up on the F-dtype cache and loses the
            // APQ Q8 fast-decode path (~-13% vs Q8).
            // Mirror the q4 block's effective want_q4 (env override + auto)
            // so we never enable both caches simultaneously.
            let q4_effective = flags.has_qkv_bias && config.n_layers <= 50;
            let q4_not_applicable = config.kv_quant == KvQuant::Q4 && !q4_effective;
            let want_q8 = config.kv_quant == KvQuant::Q8 || q4_not_applicable;
            // Per-layer head_dim must be in the kernel-supported set.
            // the old gemma4 HD=512 exclusion ("tiny K-norm
            // weights amplify Q8 noise -> degenerate") was DISPROVEN - gemma's
            // k_norm is a uniform per-layer scalar (≈0.127), which Q8_0's
            // per-block scale absorbs; the degeneracy was the
            // (then-absent) HD=512 dispatch falling back to garbage. The
            // HD=512 Q8 score+output kernels exist (attn_score_q8_0_q8_1_gqa_hd512,
            // attn_output_q8_0_f32_hd512_nq*) and are used by other archs.
            // So gemma4 global (HD=512) layers now go Q8 too - the bandwidth
            // lever for the long-context loss (halve the dominant K+V read).
            let head_dim_supported = matches!(head_dim, 64 | 128 | 256 | 512);
            if want_q8 && device.is_cuda() && head_dim_supported {
                // Pre-allocate for the configured (engine-level) context
                // length so appends never reallocate. Use the PER-LAYER
                // n_kv_head (n_kv_head_layer), not the model-level one
                // from config - for gemma4 26B the model-level value is
                // the worst-case (16) while SWA layers actually have 8.
                // Allocating with the model-level value would over-reserve
                // 2x on every SWA layer and OOM at load.
                let max_seq = config.effective_max_context();
                match crate::inference::cache::q8_kv::Q8KvCache::new(
                    max_seq,
                    n_kv_head_layer,
                    head_dim,
                    device,
                ) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        tracing::warn!(
                            "Q8 KV cache alloc failed for layer (device={:?}): {e}; \
                             decoding will fall back to F16 KV cache",
                            device
                        );
                        None
                    }
                }
            } else {
                None
            }
        },
        #[cfg(feature = "cuda")]
        q4_kv_cache: {
            use crate::inference::engine::llm_engine::KvQuant;
            // Auto-enable Q4 KV cache when GH_APQ will *not* fire - i.e.
            // when has_qkv_bias is true (qwen2, deepcoder) or fused-QKV
            // load-time concat won't happen. APQ's F-dtype path
            // outperforms Q4 decode at short cache lengths, so Q4 only
            // helps the non-APQ slow path (separate Q/K/V matmuls +
            // standard_attention with no fused norm/RoPE).
            //
            // Auto-enable when the model would NOT use APQ:
            //   - has_qkv_bias=true (qwen2-style biases block byte-concat)
            //   - layers <= 50: empirically the Q4 decode overhead amortises
            //     for ~48-layer 14B models (deepcoder) but regresses for
            //     64-layer 32B models (deepseek-r1:32b) under multi-GPU.
            let want_q4 = flags.has_qkv_bias && config.n_layers <= 50;
            if want_q4 && config.kv_quant == KvQuant::Q4 && device.is_cuda() {
                let max_seq = config.effective_max_context();
                match crate::inference::cache::q4_kv::Q4KvCache::new(
                    max_seq,
                    config.n_kv_head,
                    head_dim,
                    device,
                ) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        tracing::warn!(
                            "Q4 KV cache alloc failed for layer (device={:?}): {e}; \
                             decoding will fall back to F-dtype KV cache",
                            device
                        );
                        None
                    }
                }
            } else {
                if config.kv_quant == KvQuant::Q4 && !want_q4 {
                    tracing::debug!(
                        "Q4 KV not applicable for this model (no qkv_bias or \
                         n_layers > 50); using F-dtype cache for this layer"
                    );
                }
                None
            }
        },
        // Donor layer flag: true when this layer is in the non-shared
        // range and the model has shared_kv_layers > 0. Donors must
        // populate F-dtype kv_cache alongside Q8 so shared layers
        // downstream can read their K/V via the standard path.
        populate_dual_kv: {
            let first_kv_shared = if config.shared_kv_layers > 0 {
                config.n_layers.saturating_sub(config.shared_kv_layers)
            } else {
                config.n_layers
            };
            config.shared_kv_layers > 0 && raw.layer_idx < first_kv_shared
        },
        // gemma4 F16 KV lever (kv_quant==Off only - Q8/Q4 caches already halve
        // the read). Opt-out A/B knob `LOKEN_GEMMA4_F32_KV` keeps the F32
        // path for measurement. gemma4 detected by arch; other arches unchanged.
        kv_f16: {
            use crate::inference::engine::llm_engine::KvQuant;
            config.arch == "gemma4"
                && matches!(config.kv_quant, KvQuant::Off)
                && device.is_cuda()
                && true
        },
        padded_mask: None,
        graph_q_buffer: None,
        graph_rope_cos: None,
        graph_rope_sin: None,
        graph_kv_pos: None,
        graph_attn_qk_buffer: None,
        graph_attn_out_buffer: None,
        graph_attn_proj_buffer: None,
        graph_ffn_up_buffer: None,
        graph_ffn_activated_buffer: None,
        graph_ffn_down_buffer: None,
        graph_phi2_merge_buffer: None,
        graph_ffn_up_concat_buffer: None,
        graph_post_ffn_norm_buffer: None,
        graph_ple_gate_buffer: None,
        graph_ple_gelu_buffer: None,
        graph_ple_proj_buffer: None,
        graph_ple_final_buffer: None,
        graph_post_attn_norm_buffer: None,
        graph_x_norm_ffn_buffer: None,
        graph_x_norm_attn_buffer: None,
        graph_ple_input_buffer: None,
        graph_attn_mask_added_buffer: None,
        graph_attn_softmax_buffer: None,
        // Path B schema - 3 NEW buffer fields for gemma4
        // graph-mode unblock. The other 6 buffers are reused from existing
        // graph_post_attn_norm_buffer / graph_x_norm_ffn_buffer /
        // graph_post_ffn_norm_buffer / graph_ffn_up_buffer /
        // graph_ffn_activated_buffer / graph_ffn_down_buffer.
        graph_ffn_gate_out: None,
        graph_x_out_scaled: None,
        graph_attn_scaled_buffer: None,
        graph_alive_tensors: Vec::new(),
        cos,
        sin,
        neg_inf,
        // Proportional RoPE: only for global layers (head_dim matches config).
        rope_freq_factors: if head_dim == config.head_dim {
            rope_freq_factors.map(|t| t.to_device(device)).transpose()?
        } else {
            None
        },
        // Precompute the factored inverse-frequency table for Global
        // layers. Skips a chain of small launches every token.
        rope_factored_freqs: if head_dim == config.head_dim && rope_freq_factors.is_some() {
            let factors = rope_freq_factors.as_ref().unwrap().to_device(device)?;
            let half_dim = effective_rope_dim / 2;
            if factors.elem_count() == half_dim {
                let rope_base = config.rope_freq_base;
                let factors_f32 = if factors.dtype() == crate::tensor::DType::F32 {
                    factors.clone()
                } else {
                    factors.to_dtype(crate::tensor::DType::F32)?
                };
                let inv_factors = factors_f32.recip()?.unsqueeze(0)?;
                let freqs = crate::inference::model::rope::inverse_frequencies(
                    effective_rope_dim,
                    rope_base,
                );
                let freqs = Tensor::new(freqs, device)?.unsqueeze(0)?;
                Some(freqs.broadcast_mul(&inv_factors)?)
            } else {
                None
            }
        } else {
            None
        },
        n_head: config.n_head,
        n_kv_head: n_kv_head_layer,
        head_dim,
        // SWA classification: prefer the explicit per-layer pattern (gemma4
        // `sliding_window_pattern`, authoritative); fall back to the head_dim
        // heuristic (SWA layers have a smaller head than the global config).
        sliding_window: match config
            .swa_pattern
            .as_ref()
            .and_then(|p| p.get(raw.layer_idx))
        {
            Some(&is_swa) => {
                if is_swa {
                    config.sliding_window
                } else {
                    None
                }
            }
            None => {
                if head_dim < config.head_dim {
                    config.sliding_window
                } else {
                    None
                }
            }
        },
        rope_dim,
        attention_scale: config.attention_scale,
        residual_scale: config.residual_scale,
        flags,
    })
}

/// Build one `GenericTransformerLayer` from an AWQ checkpoint. Dense
/// Qwen2/Qwen3/Mistral: separate q/k/v (so the GGUF-byte APQ fast path gates off
/// and `standard_attention` - which works with any `QMatMul` - runs), f16 norms,
/// F32 biases. All CUDA-graph buffers None; KV cache is F16 SpecKvCache
/// (kv_quant=Off for this first cut).
#[allow(clippy::too_many_arguments)]
fn build_awq_layer(
    st: &crate::tensor::safetensors::MmapedSafetensors,
    layer_idx: usize,
    device: &Device,
    config: &GenericTransformerConfig,
    flags: Arc<GenericLayerFlags>,
    cos: &Tensor,
    sin: &Tensor,
    group_size: usize,
) -> Result<GenericTransformerLayer> {
    use crate::inference::generic_transformer::projection::AwqWeight;
    let eps = config.rms_norm_eps;
    let p = format!("model.layers.{layer_idx}");

    // One AWQ projection -> QMatMul::Awq. k = in (qweight rows), n = out (scales cols).
    let load_awq = |proj: &str| -> Result<QMatMul> {
        // Force contiguous, offset-0 device buffers - the AWQ kernels take raw
        // `as_cuda_slice`s and assume a packed layout from the buffer start.
        let qweight = st
            .load(&format!("{p}.{proj}.qweight"), device)?
            .contiguous()?;
        let qzeros = st
            .load(&format!("{p}.{proj}.qzeros"), device)?
            .contiguous()?;
        let scales = st
            .load(&format!("{p}.{proj}.scales"), device)?
            .to_dtype(DType::F16)?
            .contiguous()?;
        let k = qweight.dim(0)?;
        let n = scales.dim(1)?;
        Ok(QMatMul::from_awq(AwqWeight {
            qweight,
            qzeros,
            scales,
            bias: None,
            k,
            n,
            group_size,
            repacked: std::sync::OnceLock::new(),
            marlin: std::sync::OnceLock::new(),
        }))
    };
    // Raw AWQ tensors (for fusion via cat along the output dim).
    let load_raw = |proj: &str| -> Result<(Tensor, Tensor, Tensor)> {
        let qweight = st
            .load(&format!("{p}.{proj}.qweight"), device)?
            .contiguous()?;
        let qzeros = st
            .load(&format!("{p}.{proj}.qzeros"), device)?
            .contiguous()?;
        let scales = st
            .load(&format!("{p}.{proj}.scales"), device)?
            .to_dtype(DType::F16)?
            .contiguous()?;
        Ok((qweight, qzeros, scales))
    };
    // Fuse several projections (same K, same group_size) along the output dim
    // into ONE AwqWeight - qweight/qzeros are [K|K/gs, N/8] (cat dim 1), scales
    // [K/gs, N] (cat dim 1). One GEMV + one launch instead of N.
    let fuse_awq = |projs: &[&str]| -> Result<QMatMul> {
        // Concatenate on the host: there is no i32 concat CUDA kernel (cat on i32
        // device tensors -> "named symbol not found"). Load each shard to CPU, cat,
        // then move the fused tensor to the device once.
        let load_cpu = |proj: &str| -> Result<(Tensor, Tensor, Tensor)> {
            let w = st.load(&format!("{p}.{proj}.qweight"), &Device::Cpu)?;
            let z = st.load(&format!("{p}.{proj}.qzeros"), &Device::Cpu)?;
            let s = st
                .load(&format!("{p}.{proj}.scales"), &Device::Cpu)?
                .to_dtype(DType::F16)?;
            Ok((w, z, s))
        };
        let mut qws = Vec::new();
        let mut qzs = Vec::new();
        let mut scs = Vec::new();
        for pr in projs {
            let (w, z, s) = load_cpu(pr)?;
            qws.push(w);
            qzs.push(z);
            scs.push(s);
        }
        let qweight = Tensor::cat(&qws.iter().collect::<Vec<_>>(), 1)?
            .contiguous()?
            .to_device(device)?;
        let qzeros = Tensor::cat(&qzs.iter().collect::<Vec<_>>(), 1)?
            .contiguous()?
            .to_device(device)?;
        let scales = Tensor::cat(&scs.iter().collect::<Vec<_>>(), 1)?
            .contiguous()?
            .to_device(device)?;
        let k = qweight.dim(0)?;
        let n = scales.dim(1)?;
        Ok(QMatMul::from_awq(AwqWeight {
            qweight,
            qzeros,
            scales,
            bias: None,
            k,
            n,
            group_size,
            repacked: std::sync::OnceLock::new(),
            marlin: std::sync::OnceLock::new(),
        }))
    };
    let _ = &load_raw;
    // Optional bias -> F32 on device.
    let load_bias = |proj: &str| -> Result<Option<Tensor>> {
        match st.load(&format!("{p}.{proj}.bias"), device) {
            Ok(t) => Ok(Some(t.to_dtype(DType::F32)?)),
            Err(_) => Ok(None),
        }
    };
    let load_norm = |name: &str| -> Result<RmsNorm> {
        let w = st
            .load(&format!("{p}.{name}"), device)?
            .to_dtype(DType::F32)?;
        Ok(RmsNorm::from_tensor(w, eps))
    };

    // -- Fused QKV -> routes decode through the APQ fast path (fused norm+RoPE+cast
    //    + Q8 attention), eliminating 2 GEMV launches/layer. q_norm/k_norm
    //    (qwen3) block byte-level fusion in GGUF but here we fuse the F32 GEMV, so
    //    only fuse when there's NO qk-norm (qwen2/mistral); else keep separate.
    let fuse_qkv = !flags.has_qk_norm; // qwen2/mistral: fuse q|k|v -> APQ fast path
    let (attn_q, attn_k, attn_v, attn_qkv, attn_q_bias, attn_k_bias, attn_v_bias, attn_qkv_bias);
    if fuse_qkv {
        attn_qkv = Some(fuse_awq(&[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ])?);
        (attn_q, attn_k, attn_v) = (None, None, None);
        // Fused bias = q||k||v (or None if absent).
        let qb = load_bias("self_attn.q_proj")?;
        attn_qkv_bias = match (
            &qb,
            load_bias("self_attn.k_proj")?,
            load_bias("self_attn.v_proj")?,
        ) {
            (Some(q), Some(k), Some(v)) => Some(Tensor::cat(&[q, &k, &v], 0)?.contiguous()?),
            _ => None,
        };
        (attn_q_bias, attn_k_bias, attn_v_bias) = (None, None, None);
    } else {
        attn_q = Some(load_awq("self_attn.q_proj")?);
        attn_k = Some(load_awq("self_attn.k_proj")?);
        attn_v = Some(load_awq("self_attn.v_proj")?);
        attn_qkv = None;
        attn_qkv_bias = None;
        attn_q_bias = load_bias("self_attn.q_proj")?;
        attn_k_bias = load_bias("self_attn.k_proj")?;
        attn_v_bias = load_bias("self_attn.v_proj")?;
    }
    let attn_output = load_awq("self_attn.o_proj")?;
    let (attn_q_norm, attn_k_norm) = if flags.has_qk_norm {
        (
            Some(load_norm("self_attn.q_norm.weight")?),
            Some(load_norm("self_attn.k_norm.weight")?),
        )
    } else {
        (None, None)
    };
    let ffn_gate: Option<QMatMul> = None;
    let ffn_up = fuse_awq(&["mlp.gate_proj", "mlp.up_proj"])?;
    let ffn_down = load_awq("mlp.down_proj")?;
    let attn_norm = WeightedNorm::Rms(load_norm("input_layernorm.weight")?);
    let ffn_norm = Some(WeightedNorm::Rms(load_norm(
        "post_attention_layernorm.weight",
    )?));

    let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
    let head_dim = config.head_dim;
    let cos = if cos.device().same_device(device) {
        cos.clone()
    } else {
        cos.to_device(device)?
    };
    let sin = if sin.device().same_device(device) {
        sin.clone()
    } else {
        sin.to_device(device)?
    };

    Ok(GenericTransformerLayer {
        attn_q,
        attn_k,
        attn_v,
        attn_qkv,
        attn_q_bias,
        attn_k_bias,
        attn_v_bias,
        attn_qkv_bias,
        attn_q_norm,
        attn_k_norm,
        attn_v_norm_ones: None,
        attn_output,
        attn_output_bias: None,
        attn_norm,
        attn_norm_weight_f32: None,
        post_attn_norm: None,
        ffn_gate,
        ffn_up,
        ffn_up_bias: None,
        ffn_down,
        ffn_down_bias: None,
        moe: None,
        pre_ffw_norm_2: None,
        post_ffw_norm_1: None,
        post_ffw_norm_2: None,
        embedding_length_for_moe: config.embedding_length,
        ffn_norm,
        ffn_norm_weight: None,
        ffn_norm_eps: eps as f32,
        post_ffn_norm: None,
        ple_inp_gate: None,
        ple_proj: None,
        ple_post_norm: None,
        ple_output_scale: None,
        kv_cache: SpecKvCache::new(config.effective_max_context().max(512)),
        cpu_q8_kv: None,
        cpu_f16_kv: None,
        cpu_decode_arena: None,
        cpu_norm_cache: None,
        #[cfg(feature = "cuda")]
        q8_kv_cache: {
            let head_dim_supported = matches!(head_dim, 64 | 128 | 256 | 512);
            if device.is_cuda() && head_dim_supported {
                crate::inference::cache::q8_kv::Q8KvCache::new(
                    config.effective_max_context(),
                    config.n_kv_head,
                    head_dim,
                    device,
                )
                .ok()
            } else {
                None
            }
        },
        #[cfg(feature = "cuda")]
        q4_kv_cache: None,
        populate_dual_kv: false,
        kv_f16: false,
        padded_mask: None,
        graph_q_buffer: None,
        graph_rope_cos: None,
        graph_rope_sin: None,
        graph_kv_pos: None,
        graph_attn_qk_buffer: None,
        graph_attn_out_buffer: None,
        graph_attn_proj_buffer: None,
        graph_ffn_up_buffer: None,
        graph_ffn_activated_buffer: None,
        graph_ffn_down_buffer: None,
        graph_phi2_merge_buffer: None,
        graph_ffn_up_concat_buffer: None,
        graph_post_ffn_norm_buffer: None,
        graph_ple_gate_buffer: None,
        graph_ple_gelu_buffer: None,
        graph_ple_proj_buffer: None,
        graph_ple_final_buffer: None,
        graph_post_attn_norm_buffer: None,
        graph_x_norm_ffn_buffer: None,
        graph_x_norm_attn_buffer: None,
        graph_ple_input_buffer: None,
        graph_attn_mask_added_buffer: None,
        graph_attn_softmax_buffer: None,
        graph_ffn_gate_out: None,
        graph_x_out_scaled: None,
        graph_attn_scaled_buffer: None,
        graph_alive_tensors: Vec::new(),
        cos,
        sin,
        neg_inf,
        rope_freq_factors: None,
        rope_factored_freqs: None,
        n_head: config.n_head,
        n_kv_head: config.n_kv_head,
        head_dim,
        sliding_window: None,
        rope_dim: 0,
        no_rope: false,
        attention_scale: config.attention_scale,
        residual_scale: config.residual_scale,
        flags,
    })
}
impl GenericHeteroTransformer {
    /// Load from GGUF with per-layer device assignment driven by `plan`.
    pub fn from_gguf(
        content: gguf_file::Content,
        mmap_bytes: &[u8],
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
    ) -> Result<Self> {
        Self::from_gguf_with_kv_quant(
            content,
            mmap_bytes,
            cuda_devices,
            plan,
            crate::inference::engine::llm_engine::KvQuant::Off,
            None,
        )
    }

    pub fn from_gguf_with_kv_quant(
        content: gguf_file::Content,
        mmap_bytes: &[u8],
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        use std::io::Cursor;

        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().map(std::string::ToString::to_string))
            .unwrap_or_else(|| "unknown".to_string());

        info!(
            "GenericHeteroTransformer::from_gguf arch={arch}, plan={} segments, kv_quant={:?}",
            plan.segments.len(),
            kv_quant
        );
        let _ldt = std::time::Instant::now();
        macro_rules! ldt { ($($a:tt)*) => { info!("[load] {:>7.2}s  {}", _ldt.elapsed().as_secs_f32(), format!($($a)*)); } }

        let mut cfg = GenericTransformerConfig::from_gguf(&content, &arch)?;
        cfg.kv_quant = kv_quant;
        // Cap the per-layer Q8/Q4 KV allocation at the user's configured
        // context_length when supplied - otherwise GGUF's native
        // (often 128K/256K) eagerly allocates gigabytes that may OOM.
        if let Some(n) = max_kv_seq_len {
            cfg.max_q8_seq_len = n;
        }
        let config = Arc::new(cfg);
        let flags = Arc::new(config.flags.clone());
        let n_layers = config.n_layers;

        // -- 1. Load embeddings --------------------------
        // Small embed tables (< 1 GB after F32 dequant) live on GPU 0 to
        // eliminate the CPU->GPU sync per decode token (~10-25% wall
        // closure on small models). Larger tables stay on CPU.
        //
        // 3 GB threshold was tested - caused deepcoder OOM
        // (model 8 GB + emb 2.9 GB + KV + activations > 14 GB on 17 GB GPU
        // when activation peak fires). 1 GB stays safe.
        //
        // Threshold examples:
        //   moondream:latest:   50K x 2048 x 4 =  392 MB -> GPU 0
        //   qwen3-coder:30b:   151K x 2048 x 4 = 1.2 GB -> CPU
        //   gemma4:latest:     262K x 2560 x 4 = 2.5 GB -> CPU
        //   devstral:24b:      131K x 5120 x 4 = 2.5 GB -> CPU
        //   deepcoder:14b:     152K x 5120 x 4 = 2.9 GB -> CPU
        //   qwen3:8b:          151K x 4096 x 4 = 2.4 GB -> CPU
        //
        // The cast logic in all 5 embed-lookup callsites
        // (forward_inner, forward_graph, embed_for_graph, prepare_all_kv,
        // forward_with_image_embeds) reads `embeddings.embeddings().device()`
        // and casts input_ids accordingly, so this load-time decision is
        // the ONLY behavioral switch.
        let mut cursor = Cursor::new(mmap_bytes);
        let emb_qt = content.tensor(&mut cursor, "token_embd.weight", &Device::Cpu)?;
        let emb_table_bytes = (config.vocab_size as u64) * (config.embedding_length as u64) * 4; // F32 after dequantize
        const EMB_GPU_MAX_BYTES: u64 = 1024 * 1024 * 1024; // 1 GB threshold (3 GB OOMs on 17 GB GPU)
        let emb_device = if emb_table_bytes < EMB_GPU_MAX_BYTES {
            cuda_devices.get(&0).cloned().unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };
        let emb_tensor = emb_qt.dequantize(&emb_device)?;
        let embeddings = Embedding::new(emb_tensor);
        info!(
            "✅ Loaded embeddings: vocab={}, dim={} ({} MB on {:?})",
            config.vocab_size,
            config.embedding_length,
            emb_table_bytes / (1024 * 1024),
            emb_device
        );
        ldt!(
            "embeddings loaded ({} MB, {:?})",
            emb_table_bytes / (1024 * 1024),
            emb_device
        );

        // -- 2. Load output norm + output projection on CPU ----------------
        let norm_qt = content.tensor(&mut cursor, "output_norm.weight", &Device::Cpu)?;
        let norm_weight_cpu = norm_qt.dequantize(&Device::Cpu)?;
        // Phi2/GPT-NeoX use full LayerNorm: `output_norm.bias` exists.
        let output_norm_bias_cpu = content
            .tensor(&mut cursor, "output_norm.bias", &Device::Cpu)
            .ok()
            .and_then(|qt| qt.dequantize(&Device::Cpu).ok());
        let output_norm = match &output_norm_bias_cpu {
            Some(bias) => WeightedNorm::Layer(LayerNorm::new(
                norm_weight_cpu.clone(),
                Some(bias.clone()),
                config.rms_norm_eps as f32,
            )),
            None => WeightedNorm::Rms(RmsNorm::from_tensor(
                norm_weight_cpu.clone(),
                config.rms_norm_eps,
            )),
        };

        // Per-vocabulary logit bias. The reference adds it as the last step of the
        // phi2 graph; without it the residual stream stays exact and the top of the
        // distribution is still reordered, which reads as a coherent answer to a
        // different question.
        let output_bias = content
            .tensor(&mut cursor, "output.bias", &Device::Cpu)
            .ok()
            .and_then(|qt| qt.dequantize(&Device::Cpu).ok());
        let (output_qt, output_weight_name) =
            match content.tensor(&mut cursor, "output.weight", &Device::Cpu) {
                Ok(qt) => (qt, "output.weight"),
                Err(_) => {
                    // Tied embeddings (weight sharing)
                    (
                        content.tensor(&mut cursor, "token_embd.weight", &Device::Cpu)?,
                        "token_embd.weight",
                    )
                }
            };
        // Dequantize before wrapping (needed for CUDA path later)
        let output_weight_f32 = output_qt.dequantize(&Device::Cpu)?;
        let output_proj = crate::tensor::quantized::QMatMul::from_qtensor(output_qt)?;
        info!("✅ Loaded output norm and projection on CPU");

        // -- 2b. Load PLE globals (Gemma4) - optional ---------------------
        let ple_dim = meta_u32(&content, &config.arch, "embedding_length_per_layer_input")
            .unwrap_or(0) as usize;
        let ple_token_embd: Option<Embedding> = None;
        let mut ple_token_embd_bf16: Option<(Tensor, usize)> = None;
        let load_ple_token_embd = true;
        if ple_dim > 0 && load_ple_token_embd {
            let tensor_name = "per_layer_token_embd.weight";
            if let Some(ti) = content.tensor_infos.get(tensor_name) {
                let total_ple_dim = n_layers * ple_dim;
                let elem_count: usize = ti.shape.dims().iter().product();
                let _expected_q4_bytes = elem_count / 32 * 18;
                let expected_bf16_bytes = elem_count * 2;
                let offset = ti.offset as usize;
                let tdo = content.tensor_data_offset as usize;
                let _start = tdo + offset;

                {
                    // Load as BF16 tensor via facade's normal path, keep in BF16 to save RAM
                    // (dequantize -> F32 would be 9.4GB, BF16 stays at 4.7GB)
                    tracing::info!(
                        "PLE: Loading per_layer_token_embd ({:.1}GB BF16)...",
                        expected_bf16_bytes as f64 / 1e9
                    );
                    match content.tensor(&mut cursor, tensor_name, &Device::Cpu) {
                        Ok(qt) => {
                            // dequantize BF16 -> gets F32 tensor, then convert back to BF16 for Embedding
                            match qt.dequantize(&Device::Cpu) {
                                Ok(t) => {
                                    // Store as BF16 to halve memory
                                    let t_bf16 =
                                        t.to_dtype(crate::tensor::DType::BF16).unwrap_or(t);
                                    tracing::info!(
                                        "PLE: ✅ Token embedding {:?} {:?}",
                                        t_bf16.shape(),
                                        t_bf16.dtype()
                                    );
                                    ple_token_embd_bf16 = Some((t_bf16, total_ple_dim));
                                }
                                Err(e) => warn!("PLE dequantize failed (needs ~9.4GB): {e}"),
                            }
                        }
                        Err(e) => warn!("PLE tensor read failed: {e}"),
                    }
                }
            }
        }
        // Load PLE projection on GPU 0 when available - every prefill runs
        // a [batch*seq, hidden] @ [hidden, n_layers*ple_dim] matmul through
        // `ple_model_proj.forward(hidden_2d)`. On a CPU device that becomes
        // a 28+ GFLOPS dequantize-then-matmul at 1k-token prefill, dominating
        // the whole forward and forcing a synchronous device->host copy of
        // the residual stream every prompt. Loading on CUDA flips it to the
        // tensor-core fast_mmq path. Fall back to CPU only when no GPU.
        let ple_dev = cuda_devices.get(&0).cloned().unwrap_or(Device::Cpu);
        let ple_model_proj = if ple_dim > 0 {
            content
                .tensor(&mut cursor, "per_layer_model_proj.weight", &ple_dev)
                .ok()
                .map(crate::tensor::quantized::QMatMul::from_qtensor)
                .transpose()?
        } else {
            None
        };
        let ple_proj_norm = if ple_dim > 0 {
            content
                .tensor(&mut cursor, "per_layer_proj_norm.weight", &ple_dev)
                .ok()
                .map(|qt| RmsNorm::from_qtensor(qt, config.rms_norm_eps))
                .transpose()?
        } else {
            None
        };
        if ple_dim > 0 {
            info!(
                "✅ Loaded PLE globals: ple_dim={}, token_embd={}, model_proj={}, norm={}",
                ple_dim,
                ple_token_embd.is_some(),
                ple_model_proj.is_some(),
                ple_proj_norm.is_some()
            );
        }

        // -- 2c. Load rope_freqs (Gemma4 proportional RoPE for global layers) --
        let rope_freq_factors = content
            .tensor(&mut cursor, "rope_freqs.weight", &Device::Cpu)
            .ok()
            .and_then(|qt| qt.dequantize(&Device::Cpu).ok());
        if rope_freq_factors.is_some() {
            info!("✅ Loaded rope_freqs.weight for proportional RoPE");
        }

        // -- 3. Precompute RoPE tables per device --------------------------
        // Use rope_dim for partial RoPE models; else full head_dim
        let rope_precompute_dim = config
            .rope_dim
            .filter(|&d| d > 0 && d < config.head_dim)
            .unwrap_or(config.head_dim);
        // Size the RoPE table to the model's KV window (never smaller than the default
        // 16384) so a prompt filling the whole context can't overflow it at prefill.
        let rope_table_len = config
            .effective_max_context()
            .max(crate::inference::generic_transformer::rope::ROPE_TABLE_SEQ_LEN);
        let (cos_cpu, sin_cpu) =
            crate::inference::generic_transformer::rope::precomput_freqs_cis_yarn(
                rope_precompute_dim,
                config.rope_freq_base,
                config.yarn,
                rope_table_len,
                &Device::Cpu,
            )?;
        let mut rope_cache: HashMap<String, (Tensor, Tensor)> = HashMap::new();
        rope_cache.insert("cpu".into(), (cos_cpu.clone(), sin_cpu.clone()));

        // Precompute RoPE for EVERY available CUDA device, not just the ones
        // the plan currently targets. The adaptive NVML loader (Phase 1
        // below) diverts overflow layers to whichever GPU has the most free
        // VRAM when the plan's primary fills up - that target may not appear
        // in `plan.segments` at all (e.g. a pack-first single-GPU plan whose
        // tail spills to GPU 1). If its RoPE table is missing, the diverted
        // layer dies with "RoPE cache miss: cuda_N" and the whole load falls
        // back to CPU. RoPE tables are tiny (cos/sin over ctx x rope_dim), so
        // materialising them on every detected GPU is cheap insurance that
        // keeps the optimistic-placement + divert-on-pressure path working
        // without any pre-reserved placement margin.
        // Only the cards the PLAN can reach get a RoPE table. On a single-GPU plan the
        // other probed cards hold no layer and can never receive one (the adaptive divert
        // needs a multi-GPU plan), yet the two cached tensors pinned their contexts -
        // ~130 MB and ~10 W per idle card, forever, on every small model.
        let planned_cuda: std::collections::HashSet<usize> = plan
            .segments
            .iter()
            .filter_map(|seg| match seg.kind {
                DeviceKind::Cuda(i) => Some(i),
                _ => None,
            })
            .collect();
        let divert_eligible = planned_cuda.len() > 1;
        for (&idx, dev) in cuda_devices.iter() {
            if !divert_eligible && !planned_cuda.contains(&idx) {
                continue;
            }
            // Vacant-slot form for the same reason as 8d4a4c8 / ea9f8e7:
            // the body uses ?, so or_insert_with can't propagate it.
            if let std::collections::hash_map::Entry::Vacant(slot) =
                rope_cache.entry(format!("cuda_{idx}"))
            {
                let cos = cos_cpu.to_device(dev)?;
                let sin = sin_cpu.to_device(dev)?;
                slot.insert((cos, sin));
            }
        }

        // Wait for those uploads before anything reads them.
        //
        // The host-to-device copy is issued on the stream and returns before it lands, while
        // `cos_cpu`/`sin_cpu` are ordinary host buffers that die at the end of this function.
        // Nothing here forced the copy to finish first, so the table a layer reads could be
        // whatever the allocator had already handed to someone else - and every layer's
        // attention reads it, which is why the damage showed as a hidden state that was NaN
        // from the first block rather than as an error.
        //
        // A machine that copies faster than it loads never loses that race, which is why this
        // stayed invisible where it was developed and appeared on a slower one. Paid once per
        // load, not on any hot path.
        #[cfg(feature = "cuda")]
        for dev in cuda_devices.values() {
            if let Ok(cd) = dev.as_cuda_device() {
                cd.synchronize()?;
            }
        }

        #[cfg(feature = "cuda")]
        for (idx, dev) in cuda_devices.iter() {
            if let Ok(cd) = dev.as_cuda_device() {
                if let Ok((free, _)) = cd.cuda_stream().context().mem_get_info() {
                    info!(
                        "  [mem] after RoPE precompute: CUDA:{idx} free {:.0} MB",
                        free as f64 / 1e6
                    );
                }
            }
        }
        // -- 4. Flatten plan to per-layer device assignment ----------------
        let layer_devices: Vec<(usize, DeviceKind)> = plan
            .segments
            .iter()
            .flat_map(|seg| (seg.layer_start..seg.layer_end).map(move |i| (i, seg.kind)))
            .collect();
        assert_eq!(
            layer_devices.len(),
            n_layers,
            "plan covers {} layers but model has {}",
            layer_devices.len(),
            n_layers
        );

        // Host-RAM admission. The planner already measures MemAvailable, subtracts
        // its safety headroom and carries the result in the CPU segment's
        // `free_memory_bytes` - and nothing read it. The field had three writers
        // and no reader, so a spill was bounded by nothing: a model too large for
        // the cards planned as much host memory as it liked and the machine went
        // to swap, which is the failure this was written to prevent.
        //
        // Refusing here is the point. A load that cannot fit must fail while it is
        // still a load; once the pages are dirtied the box is already unusable and
        // no later cascade can take them back.
        {
            let cpu_layers = layer_devices
                .iter()
                .filter(|(_, k)| matches!(k, DeviceKind::Cpu))
                .count();
            if cpu_layers > 0 {
                let per_layer = (mmap_bytes.len() as u64) / (n_layers.max(1) as u64);
                let want = per_layer.saturating_mul(cpu_layers as u64);
                let budget = plan
                    .segments
                    .iter()
                    .find(|s| matches!(s.kind, DeviceKind::Cpu))
                    .map(|s| s.free_memory_bytes)
                    .unwrap_or(0);
                if !host_spill_fits(want, budget) {
                    return Err(crate::tensor::Error::msg(format!(
                        "host memory: this placement spills {} layers to the host, about {:.1} GB, \
                         and only {:.1} GB is available after the safety margin. Refusing rather \
                         than swapping the machine.",
                        cpu_layers, want as f64 / 1e9, budget as f64 / 1e9)));
                }
                info!(
                    "🧠 host spill admitted: {} layers, {:.1} GB of {:.1} GB available",
                    cpu_layers,
                    want as f64 / 1e9,
                    budget as f64 / 1e9
                );
            }
        }

        // -- 5. Phase 1: GGUF tensor loading -----------------------------
        // Multi-GPU: sequential loading (CUDA contexts require one GPU per thread)
        // Single-GPU: parallel via rayon (safe with one CUDA context)
        let multi_gpu = cuda_devices.len() > 1;
        info!(
            "📋 Loading {} layers {} (multi_gpu={})",
            n_layers,
            if multi_gpu {
                "sequentially"
            } else {
                "in parallel"
            },
            multi_gpu
        );
        let tdo = content.tensor_data_offset;
        let f = &flags;
        let ti = &content.tensor_infos;
        // When set (engine path), weights load as zero-copy mmap views (Arc pins it).
        let mmap_owner: MmapOwner = content.mmap_owner.as_ref();

        let load_one = |idx: usize, dk: DeviceKind| -> crate::tensor::Result<RawGenericLayer> {
            let prefix = format!("blk.{idx}");
            let weight_dev = match dk {
                DeviceKind::Cuda(gpu_idx) => cuda_devices.get(&gpu_idx).unwrap_or(&Device::Cpu),
                _ => &Device::Cpu,
            };

            macro_rules! req {
                ($name:expr) => {
                    required_tensor(ti, mmap_bytes, tdo, $name, weight_dev, mmap_owner)
                };
            }
            // Biases and norms always on CPU
            macro_rules! opt_cpu {
                ($name:expr) => {
                    try_read_tensor(ti, mmap_bytes, tdo, $name, &Device::Cpu, mmap_owner)
                };
            }
            macro_rules! req_cpu {
                ($name:expr) => {
                    required_tensor(ti, mmap_bytes, tdo, $name, &Device::Cpu, mmap_owner)
                };
            }

            // Attention
            let (attn_q, attn_k, attn_v, attn_qkv) = if f.fused_qkv {
                let qkv = req!(&format!("{prefix}.attn_qkv.weight"))?;
                (None, None, None, Some(qkv))
            } else {
                let q = req!(&format!("{prefix}.attn_q.weight"))?;
                let k = req!(&format!("{prefix}.attn_k.weight"))?;
                // Gemma4 26B Global layers omit attn_v and reuse attn_k
                // as V (AttentionKEqV pattern). Use opt-load for V.
                let v = try_read_tensor(
                    ti,
                    mmap_bytes,
                    tdo,
                    &format!("{prefix}.attn_v.weight"),
                    weight_dev,
                    mmap_owner,
                )?;
                (Some(q), Some(k), v, None)
            };

            // Detect per-layer Q output dim from GGUF tensor shape.
            // In GGUF, QMatMul output dim = last dim of stored shape.
            // Gemma4 has per-layer head_dim (SWA=256, full=512).
            // Q output dim: the loader reverses GGUF dims, so the FIRST dim
            // in the reference Shape is the output dim (last GGUF dim).
            let q_out_dim = if !f.fused_qkv {
                ti.get(&format!("{prefix}.attn_q.weight"))
                    .map(|info| info.shape.dims()[0])
            } else {
                None // fused QKV uses config head_dim
            };
            let k_out_dim = if !f.fused_qkv {
                ti.get(&format!("{prefix}.attn_k.weight"))
                    .map(|info| info.shape.dims()[0])
            } else {
                None
            };

            let attn_output = req!(&format!("{prefix}.attn_output.weight"))?;
            // Phi2: attn_output bias
            let attn_output_bias_qt = if f.has_attn_output_bias {
                opt_cpu!(&format!("{prefix}.attn_output.bias"))?
            } else {
                None
            };

            // Optional biases (Qwen2) - always CPU (small tensors)
            let (attn_q_bias, attn_k_bias, attn_v_bias) = if f.has_qkv_bias {
                let bq = opt_cpu!(&format!("{prefix}.attn_q.bias"))?;
                let bk = opt_cpu!(&format!("{prefix}.attn_k.bias"))?;
                let bv = opt_cpu!(&format!("{prefix}.attn_v.bias"))?;
                (bq, bk, bv)
            } else {
                (None, None, None)
            };

            // QK norms (Gemma3) - always CPU
            let (attn_q_norm_qt, attn_k_norm_qt) = if f.has_qk_norm {
                let qn = opt_cpu!(&format!("{prefix}.attn_q_norm.weight"))?;
                let kn = opt_cpu!(&format!("{prefix}.attn_k_norm.weight"))?;
                (qn, kn)
            } else {
                (None, None)
            };

            // Norms - always CPU. attn_norm is OPTIONAL: post-norm-only archs
            // (OLMo2) have no pre-norm -> None here -> Identity pre-norm at build.
            let attn_norm_qt = opt_cpu!(&format!("{prefix}.attn_norm.weight"))?;
            let attn_norm_bias_qt = if f.layer_norm_with_bias {
                opt_cpu!(&format!("{prefix}.attn_norm.bias"))?
            } else {
                None
            };
            let post_attn_norm_qt = if f.has_post_attn_norm {
                opt_cpu!(&format!("{prefix}.post_attention_norm.weight"))?
            } else {
                None
            };
            let ffn_norm_qt = if f.parallel_attn {
                None
            } else if f.has_post_ffn_norm {
                // OLMo2 post-norm-only: pre-ffn-norm absent -> Identity at build.
                opt_cpu!(&format!("{prefix}.ffn_norm.weight"))?
            } else {
                Some(req_cpu!(&format!("{prefix}.ffn_norm.weight"))?)
            };
            let post_ffn_norm_qt = if f.has_post_ffn_norm {
                opt_cpu!(&format!("{prefix}.post_ffw_norm.weight"))?
            } else {
                None
            };

            // FFN
            let (ffn_gate, ffn_up, ffn_down) =
                if f.is_moe && !ti.contains_key(&format!("{prefix}.ffn_up.weight")) {
                    // Standard MoE (granitemoe): the experts ARE the FFN - there is
                    // no dense/shared FFN (unlike gemma4-MoE). ffn_up/ffn_down are
                    // never forwarded (layer-forward branches to MoE-only), so load
                    // tiny Q8 placeholders to satisfy the non-Option struct fields.
                    let dummy = || -> crate::tensor::Result<QTensor> {
                        let z = Tensor::zeros_on(
                            (32usize, 32usize),
                            crate::tensor::DType::F32,
                            weight_dev,
                        )?;
                        QTensor::quantize_onto(
                            &z,
                            crate::tensor::quantized::GgmlDType::Q8_0,
                            weight_dev,
                        )
                    };
                    (None, dummy()?, dummy()?)
                } else if f.fused_ffn_gate_up {
                    // Phi3: no separate gate
                    let up = req!(&format!("{prefix}.ffn_up.weight"))?;
                    let down = req!(&format!("{prefix}.ffn_down.weight"))?;
                    (None, up, down)
                } else {
                    let gate = req!(&format!("{prefix}.ffn_gate.weight"))?;
                    let up = req!(&format!("{prefix}.ffn_up.weight"))?;
                    let down = req!(&format!("{prefix}.ffn_down.weight"))?;
                    (Some(gate), up, down)
                };
            // Phi2: ffn biases
            let (ffn_up_bias_qt, ffn_down_bias_qt) = if f.has_ffn_bias {
                let up_b = opt_cpu!(&format!("{prefix}.ffn_up.bias"))?;
                let down_b = opt_cpu!(&format!("{prefix}.ffn_down.bias"))?;
                (up_b, down_b)
            } else {
                (None, None)
            };

            // PLE tensors (Gemma4) - on layer's target device
            macro_rules! opt_dev {
                ($name:expr) => {
                    try_read_tensor(ti, mmap_bytes, tdo, $name, weight_dev, mmap_owner)
                };
            }
            let ple_inp_gate = opt_dev!(&format!("{prefix}.inp_gate.weight"))?;
            let ple_proj = opt_dev!(&format!("{prefix}.proj.weight"))?;
            let ple_post_norm_qt = opt_cpu!(&format!("{prefix}.post_norm.weight"))?;
            let ple_output_scale_qt = opt_cpu!(&format!("{prefix}.layer_output_scale.weight"))?;

            // Gemma4-MoE per-layer weights
            let moe_gate_inp = if f.is_moe {
                opt_dev!(&format!("{prefix}.ffn_gate_inp.weight"))?
            } else {
                None
            };
            let moe_gate_inp_scale_qt = if f.is_moe {
                opt_cpu!(&format!("{prefix}.ffn_gate_inp.scale"))?
            } else {
                None
            };
            let moe_gate_up_exps = if f.is_moe {
                opt_dev!(&format!("{prefix}.ffn_gate_up_exps.weight"))?
            } else {
                None
            };
            let moe_down_exps = if f.is_moe {
                opt_dev!(&format!("{prefix}.ffn_down_exps.weight"))?
            } else {
                None
            };
            let moe_down_exps_scale_qt = if f.is_moe {
                opt_cpu!(&format!("{prefix}.ffn_down_exps.scale"))?
            } else {
                None
            };
            // granitemoe/standard MoE: SEPARATE gate & up experts (gemma4 fuses
            // them into ffn_gate_up_exps, read above -> None here).
            let moe_gate_exps = if f.is_moe {
                opt_dev!(&format!("{prefix}.ffn_gate_exps.weight"))?
            } else {
                None
            };
            let moe_up_exps = if f.is_moe {
                opt_dev!(&format!("{prefix}.ffn_up_exps.weight"))?
            } else {
                None
            };
            let pre_ffw_norm_2_qt = if f.has_pre_ffw_norm_2 {
                opt_cpu!(&format!("{prefix}.pre_ffw_norm_2.weight"))?
            } else {
                None
            };
            let post_ffw_norm_1_qt = if f.has_post_ffw_norm_split {
                opt_cpu!(&format!("{prefix}.post_ffw_norm_1.weight"))?
            } else {
                None
            };
            let post_ffw_norm_2_qt = if f.has_post_ffw_norm_split {
                opt_cpu!(&format!("{prefix}.post_ffw_norm_2.weight"))?
            } else {
                None
            };

            Ok(RawGenericLayer {
                layer_idx: idx,
                device_kind: dk,
                q_out_dim,
                k_out_dim,
                attn_q,
                attn_k,
                attn_v,
                attn_qkv,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_q_norm_qt,
                attn_k_norm_qt,
                attn_output,
                attn_output_bias_qt,
                attn_norm_qt,
                attn_norm_bias_qt,
                post_attn_norm_qt,
                ffn_norm_qt,
                post_ffn_norm_qt,
                ffn_gate,
                ffn_up,
                ffn_up_bias_qt,
                ffn_down,
                ffn_down_bias_qt,
                ple_inp_gate,
                ple_proj,
                ple_post_norm_qt,
                ple_output_scale_qt,
                moe_gate_inp,
                moe_gate_inp_scale_qt,
                moe_gate_up_exps,
                moe_down_exps,
                moe_down_exps_scale_qt,
                moe_gate_exps,
                moe_up_exps,
                pre_ffw_norm_2_qt,
                post_ffw_norm_1_qt,
                post_ffw_norm_2_qt,
            })
        };

        // Adaptive load: when more than one CUDA device is available the
        // loader iterates layers sequentially and queries NVML free VRAM
        // after each layer. If the plan's target GPU drops below
        // the model-derived activation reserve, subsequent layers route to
        // another CUDA device (whichever has the most free) - or to CPU
        // if all GPUs are tight. This catches the
        // qwen3-class case (5.2 GB GGUF -> 15.7 GB resident on a 17 GB
        // card) without dropping + reloading: we observe pressure on
        // the actual device as it builds up, and divert the next layer
        // instead of fighting OOM at first prefill.
        //
        // Single-GPU systems keep the original parallel-load fast path
        // (no adaptive routing is possible with only one GPU). Multi-GPU
        // systems pay a sequential-load cost but gain crash-free
        // placement; the load is dominated by tensor copy time anyway.
        // Per-GPU activation reserve for the adaptive divert - DERIVED from
        // the model, not a flat magic constant. Prefill currently feeds the
        // prompt unchunked, so the peak GPU working set scales with the
        // prefill length x the widest per-layer intermediate
        // (max(hidden, ffn_dim)), in f32, with ~2 simultaneous live buffers
        // (the up-projection output and its activation copy).
        //
        // The prefill length is bounded by the SAME working window the KV
        // cache actually pre-allocates - `q8_kv_cache::INITIAL_CAPACITY_TOKENS`
        // (the lazy-KV initial capacity, 4 K) - not the model's full GGUF
        // context. Using the full 32 K/256 K context here would reserve many
        // GB for a worst-case prefill that the lazy KV cache itself doesn't
        // even allocate up front, needlessly spilling weights to CPU. Tying
        // the activation window to the KV window keeps the two consistent and
        // the reserve realistic (devstral: 4 K x 32768 x 8 ≈ 1.1 GB, not the
        // 8.6 GB a 32 K window would demand). The integer factors (4 = f32
        // bytes, 2 = live buffers) are concrete buffer-count estimates, not
        // tuned constants.
        //
        // NOTE: a prefill longer than this window can still exceed the
        // reserve - the proper structural fix is to CHUNK prefill (bounding
        // the activation to a fixed batch, as llama.cpp/ollama do), after
        // which this reserve shrinks to the chunk size and the divert needs
        // only KV headroom. Until then this matches the lazy KV cache's own
        // risk posture (it, too, grows past 4 K lazily and can OOM).
        // Prefill is forwarded in PREFILL_CHUNK_TOKENS-token chunks, so the
        // per-forward activation PEAK is bounded by chunkxwidest - NOT the full KV
        // window. Size the reserve from the chunk. (It used to use the 4096-token
        // KV window, which over-reserved ~8x - ~1.07 GB for devstral - and pushed
        // large output projections like a 131072-vocab lm_head off the primary GPU
        // onto the slow second card. The chunk-sized reserve (~134 MB) leaves room
        // for the lm_head on GPU0, matching ollama's single-GPU placement.)
        let activation_window = config
            .effective_max_context()
            .min(crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS);
        let activation_floor_bytes: u64 = {
            let widest = config.embedding_length.max(config.ffn_dim) as u64;
            (activation_window as u64)
                .saturating_mul(widest)
                .saturating_mul(4)
                .saturating_mul(2)
        };
        info!("  Adaptive load: per-GPU activation reserve = {:.2} GB (derived: min(ctx {}, prefill-chunk {}) x max(hidden {}, ffn {}) x 4 x 2)",
            activation_floor_bytes as f64 / 1e9,
            config.effective_max_context(), crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS,
            config.embedding_length, config.ffn_dim);
        // Adaptive per-layer NVML diversion only earns its sequential cost when the
        // plan ACTUALLY spans >1 GPU (spillover is possible). When every layer targets
        // the SAME GPU (the pack-first common case, even on a 2-GPU box), the single
        // CUDA context makes the parallel rayon load safe AND ~2x faster - take it.
        let distinct_gpus: std::collections::HashSet<usize> = layer_devices
            .iter()
            .filter_map(|(_, dk)| {
                if let DeviceKind::Cuda(i) = dk {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect();
        let adaptive = cuda_devices.len() > 1 && distinct_gpus.len() > 1;
        if cuda_devices.len() > 1 && distinct_gpus.len() <= 1 {
            info!("  Load: plan targets a single GPU ({:?}) -> parallel load (skipping adaptive sequential)",
                distinct_gpus.iter().next());
        }
        let raw_layers: Vec<RawGenericLayer> = if adaptive {
            // Sequential adaptive: load layer N, observe NVML, decide
            // device for layer N+1. nvml init failure falls back to
            // strict plan-following (same as the previous behaviour).
            let nvml_handle = nvml_wrapper::Nvml::init().ok();
            let mut current_dk_override: Option<DeviceKind> = None;
            let mut diverted_count: usize = 0;
            let mut result: Vec<RawGenericLayer> = Vec::with_capacity(layer_devices.len());
            for (idx, original_dk) in layer_devices {
                let dk = current_dk_override.unwrap_or(original_dk);
                let layer = load_one(idx, dk)?;
                // Only check + adapt when the current target is CUDA.
                // CPU layers don't consume VRAM; no diversion needed.
                if let (DeviceKind::Cuda(gpu_idx), Some(ref nvml)) = (dk, &nvml_handle) {
                    if let Ok(dev) = nvml.device_by_index(gpu_idx as u32) {
                        if let Ok(mem) = dev.memory_info() {
                            if mem.free < activation_floor_bytes {
                                // Find the CUDA device with the most free
                                // VRAM that's NOT the one we just filled.
                                // PLACEMENT-EXEMPT: a CORRECTION to a plan already being
                                // followed, not the plan. The card the planner chose has
                                // just been observed with less free VRAM than the load
                                // assumed - another process took it - so the remaining
                                // layers have to go somewhere that exists NOW. Throughput
                                // ranking already decided the order; what is left to ask
                                // is which of the other cards still has room.
                                let best_other = cuda_devices
                                    .keys()
                                    .filter(|&&k| k != gpu_idx)
                                    .filter_map(|&k| {
                                        nvml.device_by_index(k as u32)
                                            .ok()
                                            .and_then(|d| d.memory_info().ok())
                                            .map(|m| (k, m.free))
                                    })
                                    .max_by_key(|&(_, free)| free);
                                let new_target = match best_other {
                                    Some((k, free)) if free >= activation_floor_bytes => {
                                        DeviceKind::Cuda(k)
                                    }
                                    _ => DeviceKind::Cpu, // all GPUs tight
                                };
                                if current_dk_override != Some(new_target) {
                                    diverted_count = 0;
                                    info!(
                                        "⚖️  GPU {} dropped to {:.2} GB free at layer {} - diverting remaining layers to {:?}",
                                        gpu_idx,
                                        mem.free as f64 / 1e9,
                                        idx,
                                        new_target,
                                    );
                                    current_dk_override = Some(new_target);
                                }
                            }
                        }
                    }
                }
                if current_dk_override.is_some() {
                    diverted_count += 1;
                }
                result.push(layer);
            }
            if diverted_count > 0 {
                info!(
                    "⚖️  Adaptive load diverted {} layers off the plan's primary GPU",
                    diverted_count
                );
            }
            result
        } else if distinct_gpus.len() > 1 {
            // Defensive: a genuine >1-GPU plan normally takes the adaptive branch above
            // (adaptive ⟺ cuda_devices>1 && distinct_gpus>1). If it ever reaches here,
            // load sequentially (plan-strict) - safe cross-context, unchanged behaviour.
            layer_devices
                .into_iter()
                .map(|(idx, dk)| load_one(idx, dk))
                .collect::<Result<Vec<_>>>()?
        } else {
            // Single target GPU (or CPU): one context -> parallel rayon over all layers.
            // The win: a 2-GPU box whose plan packs onto ONE GPU no longer pays the
            // adaptive sequential cost (Phase 1 ~1.8x; Phase 2 below is parallel too).
            layer_devices
                .into_par_iter()
                .map(|(idx, dk)| load_one(idx, dk))
                .collect::<Result<Vec<_>>>()?
        };

        info!("✅ Phase 1: {} layers loaded", raw_layers.len());
        ldt!(
            "Phase 1 (raw GGUF tensor load -> device) done: {} layers",
            raw_layers.len()
        );
        // Tightest headroom left on a card THIS MODEL WILL BUILD ON once the raw
        // weights are resident. Cards the plan does not target are reported but
        // excluded: another model resident elsewhere would otherwise dictate this
        // model's worker count, which is neither its constraint nor its business.
        let mut post_phase1_free = u64::MAX;
        #[cfg(feature = "cuda")]
        for (idx, dev) in cuda_devices.iter() {
            if let Ok(cd) = dev.as_cuda_device() {
                if let Ok((free, _)) = cd.cuda_stream().context().mem_get_info() {
                    let targeted = distinct_gpus.contains(idx);
                    info!(
                        "  [mem] after Phase1 raw-weight load: CUDA:{idx} free {:.0} MB{}",
                        free as f64 / 1e6,
                        if targeted { "" } else { " (not in this plan)" }
                    );
                    if targeted {
                        post_phase1_free = post_phase1_free.min(free as u64);
                    }
                }
            }
        }

        // -- 6. Phase 2: Build per-device layers ----------------------------
        // build_generic_layer's per-layer dequant->Q4K requant is independent and
        // touches only its layer's tensors on that layer's GPU context. When all
        // layers share ONE GPU context (single-target, the common case) this is the
        // same safety profile as the parallel Phase 1 -> build them in parallel (this
        // was the dominant load cost). For a genuine multi-GPU split (mixed contexts)
        // a rayon worker could stripe across both devices, so build sequentially.
        let build_one = |raw: RawGenericLayer| -> Result<(GenericTransformerLayer, LayerDevice)> {
            let (device, dev_enum) = match raw.device_kind {
                DeviceKind::Cuda(idx) => (
                    cuda_devices.get(&idx).unwrap_or(&Device::Cpu),
                    LayerDevice::Cuda(idx),
                ),
                _ => (&Device::Cpu, LayerDevice::Cpu),
            };
            let rope_key = match raw.device_kind {
                DeviceKind::Cuda(idx) => format!("cuda_{idx}"),
                _ => "cpu".to_string(),
            };
            let (cos, sin) = rope_cache
                .get(&rope_key)
                .ok_or_else(|| crate::tensor::Error::msg(format!("RoPE cache miss: {rope_key}")))?;
            let layer_idx = raw.layer_idx;
            let layer = build_generic_layer(
                raw,
                device,
                &config,
                flags.clone(),
                cos,
                sin,
                rope_freq_factors.as_ref(),
            )?;
            // The layer's bytes are on the card now; the file pages that carried them are
            // dead weight, and on a spilled model they are what pushes the host layers'
            // working set into swap before decode even starts.
            if matches!(dev_enum, LayerDevice::Cuda(_)) {
                content.release_layer_pages(layer_idx);
            }
            Ok((layer, dev_enum))
        };
        let built: Vec<(GenericTransformerLayer, LayerDevice)> = if distinct_gpus.len() <= 1 {
            // Each worker dequantizes a weight to F32 before requantizing it, so
            // its transient peak is set by the widest 2-D weight in a layer. The
            // raw weights are still resident here, so on a card sized to hold the
            // whole model the remaining headroom fits only a few of those F32
            // buffers - running one worker per core overflows it and the load
            // falls back to a split across cards it did not need. Derive the
            // worker count from the headroom actually measured above.
            // A worker holds the F32 dequant AND the requantized result it
            // writes, while the raw source stays resident until the layer is
            // done - so budget for both, and keep one worker's slack free so
            // the last allocation of a round still has somewhere to land.
            let per_worker = (config.embedding_length as u64)
                .saturating_mul(config.ffn_dim as u64)
                .saturating_mul(std::mem::size_of::<f32>() as u64 + 1);
            let workers = if per_worker == 0 || post_phase1_free == u64::MAX {
                rayon::current_num_threads()
            } else {
                ((post_phase1_free.saturating_sub(per_worker) / per_worker) as usize)
                    .clamp(1, rayon::current_num_threads())
            };
            info!(
                "  Phase 2: {} worker(s) (headroom {:.0} MB / {:.0} MB per worker)",
                workers,
                post_phase1_free as f64 / 1e6,
                per_worker as f64 / 1e6
            );
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(|e| crate::tensor::Error::msg(format!("Phase 2 pool: {e}")))?;
            pool.install(|| {
                raw_layers
                    .into_par_iter()
                    .map(build_one)
                    .collect::<Result<Vec<_>>>()
            })?
        } else {
            raw_layers
                .into_iter()
                .map(build_one)
                .collect::<Result<Vec<_>>>()?
        };
        let (layers, layer_devs): (Vec<GenericTransformerLayer>, Vec<LayerDevice>) =
            built.into_iter().unzip();

        info!(
            "✅ Phase 2: {} layers built, cuda_devices={}",
            layers.len(),
            cuda_devices.len()
        );
        ldt!("Phase 2 (build per-device layers) done");

        // Sync CUDA streams after parallel loading (non-default stream requires explicit sync)
        for dev in cuda_devices.values() {
            #[cfg(feature = "cuda")]
            if let Ok(cd) = dev.as_cuda_device() {
                let _ = cd.cuda_stream().synchronize();
            }
            #[cfg(not(feature = "cuda"))]
            let _ = dev;
        }
        #[cfg(feature = "cuda")]
        for (idx, dev) in cuda_devices.iter() {
            if let Ok(cd) = dev.as_cuda_device() {
                if let Ok((free, total)) = cd.cuda_stream().context().mem_get_info() {
                    info!(
                        "  [mem] after Phase2 layer-build: CUDA:{idx} free {:.0} MB / {:.0} MB",
                        free as f64 / 1e6,
                        total as f64 / 1e6
                    );
                }
            }
        }

        let sliding_window = config.sliding_window;

        // Dequantize output projection to GPU (F16) whenever CUDA is available.
        // Even for split models, the output proj (~256MB F16) fits in GPU memory
        // and avoids the expensive CPU matmul (32K x 4096).
        // Use the last CUDA device (handles final layers, closest to output).
        let last_cuda_idx = layer_devs.iter().rev().find_map(|d| {
            if let LayerDevice::Cuda(idx) = d {
                Some(*idx)
            } else {
                None
            }
        });
        // Candidate devices for the output projection, in priority order: the
        // last layer's GPU first (closest - no cross-GPU hop), then any OTHER
        // CUDA device. The fallback to a SECOND GPU is what saves large-vocab
        // models on a tight primary GPU: devstral (vocab 131072, 24B) fills
        // GPU0, so its lm_head OOM'd there and fell back to a CPU lm_head  -
        // stalling decode to ~35-48% GPU util (-60% vs ollama). An idle GPU1
        // hosts the ~0.5 GB quantized projection instead; the per-token hidden
        // [hidden] cross-GPU hop is negligible vs a CPU 5120x131072 matmul.
        let mut cand_idxs: Vec<usize> = Vec::new();
        if let Some(li) = last_cuda_idx {
            cand_idxs.push(li);
        }
        let mut others: Vec<usize> = cuda_devices
            .keys()
            .cloned()
            .filter(|i| Some(*i) != last_cuda_idx)
            .collect();
        others.sort();
        cand_idxs.extend(others);

        let mut output_proj_cuda: Option<Tensor> = None;
        let mut output_norm_cuda_weight: Option<Tensor> = None;
        let mut output_proj_cuda_qmm: Option<crate::tensor::quantized::QMatMul> = None;
        let mut output_proj_dev: Option<Device> = None;
        for (ci, idx) in cand_idxs.iter().enumerate() {
            let cuda_dev = match cuda_devices.get(idx) {
                Some(d) => d,
                None => continue,
            };
            // The QUANTIZED projection is both smaller (~quant size vs a 1.3 GB
            // F16 dequant) and the FASTER decode path (mul_mat_vec_q* emits F32
            // directly, ~650 µs/tok win on qwen3). Try it; skip the OOM-prone
            // F16 dequant unless this is the primary GPU (graph mode needs F16).
            let norm_w = match norm_weight_cpu.to_device(cuda_dev) {
                Ok(w) => w,
                Err(_) => continue,
            };
            #[cfg(feature = "cuda")]
            if let Ok(cd) = cuda_dev.as_cuda_device() {
                if let Ok((free, total)) = cd.cuda_stream().context().mem_get_info() {
                    info!("  Output projection: CUDA:{idx} free {:.0} MB / {:.0} MB before lm_head load",
                        free as f64 / 1e6, total as f64 / 1e6);
                }
            }
            let t_q = std::time::Instant::now();
            let qmm = content
                .tensor(&mut cursor, output_weight_name, cuda_dev)
                .ok()
                .and_then(|qt| crate::tensor::quantized::QMatMul::from_qtensor(qt).ok());
            if qmm.is_none() {
                if ci == 0 {
                    warn!("  Output projection OOM on primary GPU {idx}; trying other GPUs before CPU");
                }
                continue;
            }
            // The lm_head fitting at LOAD isn't enough - the first prefill needs the
            // cublas workspace (~512 MB) + activation buffers on the SAME GPU. If placing
            // the lm_head here leaves less than that, the request later OOMs at prefill
            // (CUBLAS_STATUS_ALLOC_FAILED -> failed request) even though load "succeeded".
            // So require the prefill reserve to remain free; otherwise drop this placement
            // and fall to the next GPU / CPU (slow lm_head but a WORKING request - strictly
            // better than a non-deterministic prefill OOM). Mirrors the planner's
            // gpu_runtime_reserve (inference/engine/llm_engine/mod.rs).
            #[cfg(feature = "cuda")]
            if let Ok(cd) = cuda_dev.as_cuda_device() {
                if let Ok((free_after, _)) = cd.cuda_stream().context().mem_get_info() {
                    const CUBLAS_WS_BYTES: u64 = 512 * 1024 * 1024;
                    let widest = config.embedding_length.max(config.ffn_dim) as u64;
                    let prefill_act = (crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS
                        as u64)
                        .saturating_mul(widest)
                        .saturating_mul(4)
                        .saturating_mul(2);
                    let need = CUBLAS_WS_BYTES.saturating_add(prefill_act);
                    if (free_after as u64) < need {
                        warn!("  lm_head fits CUDA:{idx} but leaves {:.0} MB < {:.0} MB prefill reserve \
                               (cublas+activation) - skipping so the first prefill won't OOM",
                            free_after as f64 / 1e6, need as f64 / 1e6);
                        drop(qmm);
                        continue;
                    }
                }
            }
            let f16 = if ci == 0 {
                output_weight_f32
                    .to_dtype(crate::tensor::DType::F16)
                    .and_then(|t| t.to_device(cuda_dev))
                    .ok()
            } else {
                None
            };
            info!(
                "  Output projection on CUDA:{idx} as QMatMul ({:.0}ms){}",
                t_q.elapsed().as_secs_f64() * 1000.0,
                if ci > 0 {
                    " [fallback to idle GPU - primary was full]"
                } else {
                    ""
                }
            );
            output_proj_cuda = f16;
            output_norm_cuda_weight = Some(norm_w);
            output_proj_cuda_qmm = qmm;
            output_proj_dev = Some(cuda_dev.clone());
            break;
        }
        if output_proj_cuda_qmm.is_none() && output_proj_cuda.is_none() {
            warn!("  GPU output projection failed on all CUDA devices; using CPU fallback");
        }

        // Pre-allocate graph_logits_buffer at model load. CRITICAL for the
        // split-path graph mode (phi2/moondream) and a small safety win
        // for working arches: this alloc happens BEFORE the first
        // update_graph_state call that flips the cudaMallocAsync pool into
        // graph-tracked mode. By pre-allocating in the standard pool, the
        // address can't be pool-reorganized when captured free_async nodes
        // execute on replay - that reorganization is the root cause.
        let graph_logits_buffer = if let Some(cuda_dev) = output_proj_dev.as_ref() {
            let vocab = config.vocab_size;
            // Shape (1, vocab) - matches what `last.matmul(proj.t())` and
            // `qmm.forward(normed)` produce in both forward_from_hidden
            // and compute_all_from_kv. Wrong shape would silently fall
            // through to the lazy alloc path at warmup.
            match Tensor::zeros_on((1, vocab), crate::tensor::DType::F32, cuda_dev) {
                Ok(t) => Some(t),
                Err(e) => {
                    warn!(
                        "Pre-alloc graph_logits_buffer failed ({}): {e}",
                        config.arch
                    );
                    None
                }
            }
        } else {
            None
        };
        ldt!("output projection (lm_head) + norm done - model ready");

        // Keep a handle only for the cards this model actually landed on. The
        // map above lists every CANDIDATE, because the load needs them all:
        // the per-layer divert picks its escape card from these keys, and the
        // out-of-memory retry re-plans across them. Once placement is settled
        // those extra handles buy nothing and are not free - a handle holds
        // the card's CUDA context alive, and a context that holds no layer
        // still costs its allocator workspace and keeps the card out of its
        // idle power state. On a model that fits one card, that is a second
        // GPU awake for the whole session.
        //
        // It also removes an arbitrary choice: several fallbacks reach for
        // `cuda_devices.values().next()`, and a map ordered by nothing could
        // hand them the card holding none of the model.
        let retained: HashMap<usize, Device> = {
            let mut keep: std::collections::HashSet<usize> = layer_devs
                .iter()
                .filter_map(|d| match d {
                    LayerDevice::Cuda(i) => Some(*i),
                    _ => None,
                })
                .collect();
            let emb_dev = embeddings.embeddings().device();
            for d in [output_proj_dev.as_ref(), Some(&emb_dev)]
                .into_iter()
                .flatten()
            {
                if let Device::Cuda(cd) = d {
                    keep.insert(cd.ordinal());
                }
            }
            cuda_devices
                .iter()
                .filter(|(idx, _)| keep.contains(idx))
                .map(|(idx, d)| (*idx, d.clone()))
                .collect()
        };
        for (idx, d) in cuda_devices
            .iter()
            .filter(|(i, _)| !retained.contains_key(i))
        {
            if let Device::Cuda(cd) = d {
                tracing::debug!(
                    "cuda ctx: dropping idle gpu{idx} handle, strong_count={} (>1 means another \
                     holder keeps its context alive)",
                    std::sync::Arc::strong_count(cd)
                );
            }
        }
        if retained.len() < cuda_devices.len() {
            // The sentence names the model because placement is its subject, not because any
            // of it is read. PRIVACY-OK: device ordinals and a count of them.
            info!(
                "  Placement: releasing {} idle CUDA context(s) - the model occupies GPU {:?}",
                cuda_devices.len() - retained.len(),
                {
                    let mut v: Vec<_> = retained.keys().copied().collect();
                    v.sort_unstable();
                    v
                }
            );
        }

        let model = Self {
            adapters: Vec::new(),
            config,
            embeddings,
            layers,
            layer_devs,
            output_norm,
            output_proj,
            output_bias,
            capture_feature: false,
            last_feature: None,
            output_proj_cuda,
            output_norm_cuda_weight,
            output_proj_cuda_qmm,
            output_proj_dev,
            cuda_devices: retained,
            sliding_window,
            mask_cache: HashMap::new(),
            ple_token_embd,
            ple_token_embd_bf16,
            ple_model_proj,
            ple_proj_norm,
            ple_dim,
            graph_hidden_buffer: None,
            graph_logits_buffer,
            graph_alive_tensors_model: Vec::new(),
            graph_state_groups: None,
            workspace_pre_grown: false,
            kv_snapshots: Vec::new(),
            kv_snapshot_tick: 0,
        };

        // A SiLU MoE runs its prefill experts on the host, and that loop reads its weights
        // through a repacked layout derived from the stacks rather than stored with them. It
        // is built on first use, which is inside the first request - for a 16-layer, 64-expert
        // model, seconds of it. Build it here instead, on a thread of its own: the load does
        // not wait for it, so a model answers exactly as early as before, and a request that
        // arrives while it is still running waits on the stack being built and is handed that
        // result rather than repacking the same stack beside it.
        let host_prefill_stacks: Vec<_> = model
            .layers
            .iter()
            .filter_map(|layer| layer.moe.as_ref())
            .flat_map(|moe| moe.host_prefill_stacks())
            .collect();
        if !host_prefill_stacks.is_empty() {
            #[cfg(feature = "cuda")]
            use crate::inference::moe_cpu as host_moe;
            #[cfg(not(feature = "cuda"))]
            use crate::inference::moe_cuda as host_moe;
            std::thread::spawn(move || {
                use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
                let started = std::time::Instant::now();
                let count = host_prefill_stacks.len();
                // The stacks are independent of each other, and repacking them one after
                // another takes longer than the rest of the load has left to run - long
                // enough that a question asked the moment the model reports ready still waits
                // on the tail of it. Spread them so the warm finishes inside the load it
                // overlaps rather than past the end of it.
                let workers = std::thread::available_parallelism()
                    .map(|n| n.get().min(8))
                    .unwrap_or(4);
                let next = AtomicUsize::new(0);
                let failed = AtomicBool::new(false);
                std::thread::scope(|scope| {
                    for _ in 0..workers {
                        scope.spawn(|| loop {
                            let Some(stack) =
                                host_prefill_stacks.get(next.fetch_add(1, Ordering::Relaxed))
                            else {
                                break;
                            };
                            if let Err(e) = host_moe::prewarm_expert_repack(stack) {
                                // The first prefill builds whatever is missing, so a failure
                                // here costs time and never an answer - said once for the
                                // warm, not once per worker and not once per stack.
                                if !failed.swap(true, Ordering::Relaxed) {
                                    tracing::warn!("expert repack warm stopped: {e}");
                                }
                                break;
                            }
                        });
                    }
                });
                if !failed.load(Ordering::Relaxed) {
                    tracing::info!(
                        "  Expert repack warm: {count} stacks on {workers} threads in {:.2}s - \
                         off the request path",
                        started.elapsed().as_secs_f64()
                    );
                }
            });
        }
        Ok(model.release_graph_only_lm_head())
    }

    /// Build a `GenericHeteroTransformer` from an HF AutoAWQ checkpoint.
    /// Dense Qwen2/Qwen3/Mistral only, single-GPU. The projection `QMatMul`s
    /// become `QMatMul::Awq` (the 483 GB/s GEMV at decode, dequant->cuBLAS at
    /// prefill); everything else (attention, RoPE, KV, forward) is the generic
    /// path unchanged. `kv_quant=Off` (F16 SpecKvCache) for this first cut - Q8
    /// KV is a later optimization.
    pub fn from_awq_safetensors(
        dir: &str,
        shards: &[String],
        cuda_devices: &HashMap<usize, Device>,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        use crate::inference::load::awq_loader as awql;

        let merr = |e: anyhow::Error| crate::tensor::Error::msg(e.to_string());

        let cfgj = awql::parse_config_json(dir).map_err(merr)?;
        let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(shards)? };

        // Detect optional tensors from the real checkpoint so flag detection matches.
        let has_q_bias = st
            .load("model.layers.0.self_attn.q_proj.bias", &Device::Cpu)
            .is_ok();
        let has_qk_norm = st
            .load("model.layers.0.self_attn.q_norm.weight", &Device::Cpu)
            .is_ok();
        let present = awql::AwqTensorPresence {
            has_q_bias,
            has_qk_norm,
        };
        let content = awql::synth_gguf_content(&cfgj, present);
        let arch = cfgj.gguf_arch.clone();
        let mut config = GenericTransformerConfig::from_gguf(&content, &arch)?;
        // Q8 KV - the GGUF qwen2/qwen3 incremental-decode path (prefill appends +
        // dequantizes for the scan; seq=1 decode reads the Q8 cache via the fused
        // decode kernels). Populated consistently across prefill and decode.
        config.kv_quant = crate::inference::engine::llm_engine::KvQuant::Q8;
        if let Some(m) = max_kv_seq_len {
            config.max_q8_seq_len = m;
        }
        let config = Arc::new(config);
        let group_size = cfgj.group_size;

        // Device choice follows the fleet placement rule: iterate GPUs
        // FASTEST-FIRST (vram_manager::probe ranks by compute throughput,
        // never by index) and take the first whose CURRENT free VRAM fits the
        // checkpoint weights plus a prefill-activation reserve. The previous
        // lowest-index pick loaded a 13 GB checkpoint onto a GPU already
        // hosting an image model while the other card sat 100% free - the
        // load "succeeded" with ~1 GB of slack and every prefill then OOM'd
        // on activations regardless of chunk size.
        let weight_bytes: u64 = shards
            .iter()
            .filter_map(|s| std::fs::metadata(s).ok().map(|m| m.len()))
            .sum();
        // Activation reserve: per-chunk transients (attention scores + FFN
        // intermediates + Marlin/dp4a repack scratch) - sized generously so a
        // "fits" verdict leaves real prefill headroom.
        const AWQ_ACT_RESERVE: u64 = 3 * 1024 * 1024 * 1024;
        let want = weight_bytes + AWQ_ACT_RESERVE;
        let mut chosen: Option<(usize, Device)> = None;
        #[cfg(feature = "cuda")]
        for (idx, free, _) in crate::inference::place::vram_manager::probe(0) {
            if let Some(d) = cuda_devices.get(&idx) {
                if free >= want {
                    chosen = Some((idx, d.clone()));
                    break;
                }
                if chosen.is_none() {
                    // Remember the fastest card as a fallback; replaced by the
                    // first card that actually FITS.
                    chosen = Some((idx, d.clone()));
                }
            }
        }
        let (device_ordinal, device) = match chosen {
            Some((idx, d)) => (idx, d),
            None => (0, Device::Cpu),
        };
        if device.is_cuda() {
            info!(
                "AWQ placement: GPU{} (weights {:.1} GB + {:.1} GB activation reserve)",
                device_ordinal,
                weight_bytes as f64 / 1e9,
                AWQ_ACT_RESERVE as f64 / 1e9
            );
        }
        info!(
            "AWQ load: {} arch={} layers={} on {:?} (gs={})",
            cfgj.hf_arch, arch, config.n_layers, device, group_size
        );

        // Prewarm the fused decode kernels onto a CLEAN device - BEFORE loading
        // the ~336 AWQ weight tensors. Lazily loading the "loken_fused" module
        // after the AWQ weights are resident fails with CUDA "named symbol not
        // found" (a facade module-cache quirk after many custom-module loads);
        // prewarming first sidesteps it.
        #[cfg(feature = "cuda")]
        if let Ok(cd) = device.as_cuda_device() {
            let _ = crate::inference::kernel::fused::prewarm_fused_kernels(&cd);
        }

        // RoPE tables (full head_dim, YaRN if configured), sized to the KV window.
        let (cos, sin) = crate::inference::generic_transformer::rope::precomput_freqs_cis_yarn(
            config.head_dim,
            config.rope_freq_base,
            config.yarn,
            config
                .effective_max_context()
                .max(crate::inference::generic_transformer::rope::ROPE_TABLE_SEQ_LEN),
            &device,
        )?;

        let flags = Arc::new(config.flags.clone());
        let mut layers = Vec::with_capacity(config.n_layers);
        let mut layer_devs = Vec::with_capacity(config.n_layers);
        for li in 0..config.n_layers {
            layers.push(build_awq_layer(
                &st,
                li,
                &device,
                &config,
                flags.clone(),
                &cos,
                &sin,
                group_size,
            )?);
            layer_devs.push(if device.is_cuda() {
                LayerDevice::Cuda(device_ordinal)
            } else {
                LayerDevice::Cpu
            });
        }

        // Embeddings (CPU - large vocab table). Loaded as F32.
        let emb = st
            .load("model.embed_tokens.weight", &Device::Cpu)?
            .to_dtype(DType::F32)?;
        let embeddings = Embedding::new(emb);

        // Final norm - CPU RmsNorm + a GPU copy for the GPU final-norm path.
        let norm_w = st
            .load("model.norm.weight", &Device::Cpu)?
            .to_dtype(DType::F32)?;
        let output_norm =
            WeightedNorm::Rms(RmsNorm::from_tensor(norm_w.clone(), config.rms_norm_eps));
        let output_norm_cuda_weight = if device.runs_as_card() {
            Some(norm_w.to_device(&device)?)
        } else {
            None
        };

        // LM head: lm_head.weight, or embed_tokens if tied. F16 on GPU for the
        // cutlass path; Q8_0 on CPU as the required fallback `output_proj`.
        let lm_cpu_f32 = match st.load("lm_head.weight", &Device::Cpu) {
            Ok(t) => t.to_dtype(DType::F32)?,
            Err(_) => st
                .load("model.embed_tokens.weight", &Device::Cpu)?
                .to_dtype(DType::F32)?,
        };
        let output_proj = crate::tensor::quantized::QMatMul::from_qtensor(
            crate::tensor::quantized::QTensor::quantize(
                &lm_cpu_f32,
                crate::tensor::quantized::GgmlDType::Q8_0,
            )?,
        )?;
        // A SECOND copy of the head, dense F16, beside the quantised one - the largest
        // thing this branch decides, and a move rather than a launch.
        let (output_proj_cuda, output_proj_dev) = if device.runs_as_card() {
            (
                Some(lm_cpu_f32.to_dtype(DType::F16)?.to_device(&device)?),
                Some(device.clone()),
            )
        } else {
            (None, None)
        };

        let sliding_window = config.sliding_window;
        Ok(Self {
            adapters: Vec::new(),
            config,
            embeddings,
            layers,
            layer_devs,
            output_norm,
            output_proj,
            // The AWQ safetensors path does not read a logit bias today; the
            // checkpoints wired through it carry none. A phi2-family AWQ model
            // would need it read here, exactly as the GGUF path does.
            output_bias: None,
            capture_feature: false,
            last_feature: None,
            output_proj_cuda,
            output_norm_cuda_weight,
            output_proj_cuda_qmm: None,
            output_proj_dev,
            cuda_devices: cuda_devices.clone(),
            sliding_window,
            mask_cache: HashMap::new(),
            ple_token_embd: None,
            ple_token_embd_bf16: None,
            ple_model_proj: None,
            ple_proj_norm: None,
            ple_dim: 0,
            graph_hidden_buffer: None,
            graph_logits_buffer: None,
            graph_alive_tensors_model: Vec::new(),
            graph_state_groups: None,
            workspace_pre_grown: false,
            kv_snapshots: Vec::new(),
            kv_snapshot_tick: 0,
        })
    }
}

/// Whether a host spill of `want` bytes fits the budget the planner measured.
///
/// Separated from the loader so the REFUSING side can be tested: exercising it
/// for real means filling host RAM, which is the failure this prevents. A guard
/// nobody has seen say no is a guard nobody knows anything about - this one sat
/// unread for months while the machine swapped.
///
/// A budget of zero means the planner supplied none (a plan with no CPU segment,
/// or one predating this field being carried); admit rather than block a load on
/// a number that was never measured.
fn host_spill_fits(want: u64, budget: u64) -> bool {
    budget == 0 || want <= budget
}

#[cfg(test)]
mod host_spill_admission_tests {
    use super::host_spill_fits;

    #[test]
    fn refuses_a_spill_larger_than_the_budget() {
        assert!(!host_spill_fits(30_000_000_000, 12_000_000_000));
    }

    #[test]
    fn admits_a_spill_that_fits() {
        // The real figures measured for deepseek-r1:70b on this box.
        assert!(host_spill_fits(13_800_000_000, 55_700_000_000));
    }

    #[test]
    fn admits_when_the_planner_supplied_no_budget() {
        // Zero means unmeasured, not "no room": blocking on it would refuse every
        // plan built without a CPU segment.
        assert!(host_spill_fits(99_000_000_000, 0));
    }

    #[test]
    fn the_boundary_admits_and_one_byte_past_it_refuses() {
        assert!(host_spill_fits(1_000, 1_000));
        assert!(!host_spill_fits(1_001, 1_000));
    }
}
