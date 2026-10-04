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
    /// Per-head attention-sink logits `[n_head]`.
    attn_sinks_qt: Option<QTensor>,
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
    /// Router bias `[n_experts]` and per-expert projection biases `[n_experts, width]`.
    moe_gate_inp_bias_qt: Option<QTensor>,
    moe_gate_exps_bias_qt: Option<QTensor>,
    moe_up_exps_bias_qt: Option<QTensor>,
    moe_down_exps_bias_qt: Option<QTensor>,
    pre_ffw_norm_2_qt: Option<QTensor>,
    post_ffw_norm_1_qt: Option<QTensor>,
    post_ffw_norm_2_qt: Option<QTensor>,
    /// Sigmoid router selection bias `[n_experts]`.
    moe_exp_probs_b_qt: Option<QTensor>,
    /// The recurrent mixer's own tensors; its output projection is `attn_output`.
    mixer: RawMixer,
    /// The block has a feed-forward sublayer.
    has_ffn: bool,
    /// Attention output gate, split from the Q projection.
    attn_gate: Option<QTensor>,
    /// Shared expert: gate (SwiGLU) or none (squared ReLU), up, down, and the scalar gate
    /// `[hidden]` scaling its output.
    shexp_gate: Option<QTensor>,
    shexp_up: Option<QTensor>,
    shexp_down: Option<QTensor>,
    shexp_scalar_gate_qt: Option<QTensor>,
    /// The wide residual's, indexer's and n-gram layer's tensors, when the block has them.
    wide: RawWide,
}

/// The tensors of a wide-residual block beyond a plain one's.
#[derive(Default)]
struct RawWide {
    attn_mixer: Option<RawMixerWeights>,
    ffn_mixer: Option<RawMixerWeights>,
    indexer: Option<RawIndexer>,
    ngram: Option<RawNgram>,
}

struct RawMixerWeights {
    norm: QTensor,
    down: QTensor,
    up: QTensor,
    inject: Option<QTensor>,
}

struct RawIndexer {
    q_proj: QTensor,
    k_proj: QTensor,
    q_norm: QTensor,
    k_norm: QTensor,
}

struct RawNgram {
    key: QTensor,
    value: QTensor,
    norm_key: QTensor,
    norm_query: QTensor,
    norm_conv: QTensor,
    conv: QTensor,
}

/// A block's mixer tensors, as read.
enum RawMixer {
    Attention,
    ShortConv {
        in_proj: QTensor,
        conv: QTensor,
    },
    DeltaNet {
        /// `attn_qkv | attn_gate | ssm_alpha | ssm_beta`, or `attn_qkv | attn_gate | ssm_ba`.
        proj: Vec<QTensor>,
        conv_w: QTensor,
        a: QTensor,
        dt_bias: QTensor,
        norm: QTensor,
    },
    Ssm {
        in_proj: QTensor,
        conv_w: QTensor,
        conv_b: Option<QTensor>,
        a: QTensor,
        d: QTensor,
        dt_bias: QTensor,
        norm: QTensor,
    },
    Absent,
}

/// Where a model's tensors are read from: one mapped file, or the parts of a split set.
///
/// A loader asks by name. A whole file answers from its own directory; a split set answers from
/// the part that holds the tensor, since each part's offsets are relative to its own mapping.
/// Either way the read is a zero-copy view of the mapping - the engine's path pins it - which is
/// what keeps a large model's load off the host heap.
struct TensorSource<'a> {
    parts: Vec<TensorPart<'a>>,
    /// Which part holds a tensor. Empty for one file: every tensor is in it.
    owner: HashMap<String, usize>,
}

struct TensorPart<'a> {
    content: &'a gguf_file::Content,
    bytes: &'a [u8],
    /// The mapping a view pins, when there is one; `None` copies (dev bins, tests).
    owner: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

impl<'a> TensorSource<'a> {
    /// One mapped file: every tensor is in it.
    fn one(content: &'a gguf_file::Content, bytes: &'a [u8]) -> Self {
        Self {
            owner: HashMap::new(),
            parts: vec![TensorPart {
                content,
                bytes,
                owner: content.mmap_owner.clone(),
            }],
        }
    }

    /// The parts of a split set, in order: metadata from the first, tensor bytes from whichever
    /// part carries them.
    fn of_split(split: &'a crate::tensor::quantized::gguf_source::SplitGguf) -> Self {
        let mut owner = HashMap::new();
        let mut parts = Vec::new();
        for part in split.parts() {
            let index = parts.len();
            for name in part.content.tensor_infos.keys() {
                owner.insert(name.clone(), index);
            }
            let mapped: std::sync::Arc<dyn std::any::Any + Send + Sync> = part.mmap.clone();
            parts.push(TensorPart {
                content: &part.content,
                bytes: &part.mmap[..],
                owner: Some(mapped),
            });
        }
        Self { parts, owner }
    }

    /// The part a tensor's bytes are in.
    fn part_of(&self, name: &str) -> Option<&TensorPart<'a>> {
        match self.owner.get(name) {
            Some(&i) => self.parts.get(i),
            None => self.parts.first(),
        }
    }

    /// The first part's header and bytes: what a reader of its own - a family's vision tower -
    /// takes, since it reads by a cursor over one mapping.
    fn first(&self) -> (&'a gguf_file::Content, &'a [u8]) {
        match self.parts.first() {
            Some(p) => (p.content, p.bytes),
            None => unreachable!("a source is built with at least one part"),
        }
    }

    /// `name` on `device`, `None` when no part carries it.
    fn read_opt(&self, name: &str, device: &Device) -> crate::tensor::Result<Option<QTensor>> {
        let Some(part) = self.part_of(name) else {
            return Ok(None);
        };
        let Some(info) = part.content.tensor_infos.get(name) else {
            return Ok(None);
        };
        Ok(Some(info.read_slice_owned(
            part.bytes,
            part.content.tensor_data_offset,
            device,
            part.owner.as_ref(),
        )?))
    }

    /// `name` on `device`; a name no part carries is an error that says so.
    fn read(&self, name: &str, device: &Device) -> crate::tensor::Result<QTensor> {
        self.read_opt(name, device)?
            .ok_or_else(|| crate::tensor::Error::msg(format!("missing tensor: {name}")))
    }

    /// The bytes the model is stored in, across every part: what a host spill is measured against.
    fn total_bytes(&self) -> u64 {
        self.parts.iter().map(|p| p.bytes.len() as u64).sum()
    }

    /// Drop layer `layer`'s pages: its weights are on a card now. Per part, since a set's
    /// offsets are relative to the mapping they sit in.
    fn release_layer_pages(&self, layer: usize) {
        for part in &self.parts {
            if let Some(mmap) = part
                .owner
                .as_ref()
                .and_then(|o| o.downcast_ref::<memmap2::Mmap>())
            {
                part.content.release_layer_pages_in(mmap, layer);
            }
        }
    }
}

/// A quantised weight built on `device` from its block bytes, `rows` x `cols`.
fn qtensor_from_bytes(
    bytes: &[u8],
    dtype: crate::tensor::quantized::GgmlDType,
    rows: usize,
    cols: usize,
    device: &Device,
) -> crate::tensor::Result<QTensor> {
    // The storage reads the bytes as blocks, so they sit in a word-aligned buffer.
    let mut buf = vec![0u32; bytes.len().div_ceil(4)];
    let aligned: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 4) };
    aligned[..bytes.len()].copy_from_slice(bytes);
    // CONVERSION: the q projection is split into its q and gate halves, each a weight of its own.
    let storage = crate::tensor::quantized::QStorage::from_data(
        std::borrow::Cow::Borrowed(&aligned[..bytes.len()]),
        device,
        dtype,
    )?;
    QTensor::new(storage, crate::tensor::Shape::from((rows, cols)))
}

/// The rows of `a` in `row_bytes`-sized rows.
fn row_bytes(qt: &QTensor) -> usize {
    let cols = qt.shape().dims()[1];
    cols / qt.dtype().block_size() * qt.dtype().type_size()
}

/// Splits a Q projection whose rows hold, per head, the head's Q rows followed by as many
/// output-gate rows into the Q projection and the gate projection, each `[n_head *
/// head_dim, hidden]` in the stored format.
fn split_q_gate(
    q: QTensor,
    n_head: usize,
    device: &Device,
) -> crate::tensor::Result<(QTensor, QTensor)> {
    let (rows, cols) = (q.shape().dims()[0], q.shape().dims()[1]);
    if n_head == 0 || rows % (2 * n_head) != 0 {
        return Err(crate::tensor::Error::msg(format!(
            "a Q projection of {rows} rows does not interleave Q and a gate over {n_head} heads"
        )));
    }
    let head_bytes = rows / (2 * n_head) * row_bytes(&q);
    let bytes = q.data()?;
    let (mut qb, mut gb) = (
        Vec::with_capacity(bytes.len() / 2),
        Vec::with_capacity(bytes.len() / 2),
    );
    for h in 0..n_head {
        let at = 2 * h * head_bytes;
        qb.extend_from_slice(&bytes[at..at + head_bytes]);
        gb.extend_from_slice(&bytes[at + head_bytes..at + 2 * head_bytes]);
    }
    let dtype = q.dtype();
    Ok((
        qtensor_from_bytes(&qb, dtype, rows / 2, cols, device)?,
        qtensor_from_bytes(&gb, dtype, rows / 2, cols, device)?,
    ))
}

/// The weights `parts` stacked by rows into one, when they share a format the expert
/// kernels read and an input width; `None` otherwise.
fn concat_rows(parts: &[QTensor], device: &Device) -> crate::tensor::Result<Option<QTensor>> {
    let Some(first) = parts.first() else {
        return Ok(None);
    };
    let (dtype, cols) = (first.dtype(), first.shape().dims()[1]);
    let same = parts
        .iter()
        .all(|p| p.dtype() == dtype && p.shape().dims().len() == 2 && p.shape().dims()[1] == cols);
    if !same || !crate::inference::moe_cuda::gemm::expert_kernels_serve(dtype) {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    let mut rows = 0;
    for p in parts {
        bytes.extend_from_slice(&p.data()?);
        rows += p.shape().dims()[0];
    }
    qtensor_from_bytes(&bytes, dtype, rows, cols, device).map(Some)
}

/// Build a `GenericTransformerLayer` from a raw loaded layer.
/// The rotary tables the layers of one load share: layers with the same rotary width, base
/// and scaling on the same device read one table, built once. A table spans the whole
/// context the model declares, so one per layer is memory the plan never counted.
#[derive(Default)]
pub(crate) struct LayerRopes(
    std::sync::Mutex<HashMap<(String, usize, u32, String, usize), (Tensor, Tensor)>>,
);

impl LayerRopes {
    fn table(
        &self,
        dim: usize,
        base: f32,
        yarn: Option<crate::inference::model::rope::YarnParams>,
        len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let key = (
            format!("{:?}", device.location()),
            dim,
            base.to_bits(),
            format!("{yarn:?}"),
            len,
        );
        let mut tables = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = tables.get(&key) {
            return Ok(t.clone());
        }
        let t = crate::inference::generic_transformer::rope::precomput_freqs_cis_yarn(
            dim, base, yarn, len, device,
        )?;
        tables.insert(key, t.clone());
        Ok(t)
    }
}

/// A float projection rebuilt as the block format `target` on `device`, and the ledger entry
/// that says so. The block-quant kernels read it at a fraction of the float bytes and the
/// fused projections need one format across a layer.
fn requant_float_weight(
    qt: QTensor,
    target: crate::tensor::quantized::GgmlDType,
    device: &Device,
    class: &'static str,
) -> Result<QTensor> {
    let from = qt.dtype();
    let elems = qt.shape().elem_count() as u64;
    let before = elems * qt.dtype().type_size() as u64 / qt.dtype().block_size() as u64;
    // CONVERSION: the block-quant kernels need a block format; the float weight is rebuilt.
    let dequantized = qt.dequantize(&Device::Cpu)?;
    // CONVERSION: the block-quant kernels need a block format; the float weight is rebuilt.
    let out = QTensor::quantize_onto(&dequantized, target, device)?;
    crate::tensor::conversion::record_kept(
        class,
        from.gguf_name(),
        target.gguf_name(),
        before,
        elems * target.type_size() as u64 / target.block_size() as u64,
        "float projection rebuilt as a block format for the block-quant kernels and fused projections",
    );
    Ok(out)
}

/// Whether the expert kernels of a device read a stack of `dtype` as the file stores it: the
/// card kernels on a card, the host dot kernels on the host. A stack for which this is false
/// is rebuilt in a format they read.
pub(crate) fn moe_stack_served_as_stored(
    dtype: crate::tensor::quantized::GgmlDType,
    on_card: bool,
) -> bool {
    if on_card {
        crate::inference::moe_cuda::gemm::expert_kernels_serve(dtype)
    } else {
        crate::inference::moe_cpu::cpu_qmatmul_supported(dtype)
    }
}

fn build_generic_layer(
    raw: RawGenericLayer,
    device: &Device,
    config: &GenericTransformerConfig,
    flags: Arc<GenericLayerFlags>,
    cos: &Tensor,
    sin: &Tensor,
    ropes: &LayerRopes,
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
            // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
            let w = qt.dequantize(&Device::Cpu)?.to_device(device)?;
            Ok(RmsNorm::from_tensor(w, eps))
        } else {
            RmsNorm::from_qtensor(qt, eps)
        }
    };

    let build_norm_opt =
        |qt: Option<QTensor>| -> Result<Option<RmsNorm>> { qt.map(build_norm).transpose() };

    // The block format float projections are requantised to: Q4K, unless a float weight's
    // rows are not whole Q4K blocks, in which case the layer's float weights all take Q8_0
    // so the byte-concat fusions below still see one format.
    let requant_dtype = {
        use crate::tensor::quantized::GgmlDType;
        let is_float = |qt: &QTensor| {
            matches!(
                qt.dtype(),
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
            )
        };
        let whole_q4k_rows = [
            raw.attn_q.as_ref(),
            raw.attn_k.as_ref(),
            raw.attn_v.as_ref(),
            raw.attn_qkv.as_ref(),
            Some(&raw.attn_output),
            raw.ffn_gate.as_ref(),
            Some(&raw.ffn_up),
            Some(&raw.ffn_down),
        ]
        .into_iter()
        .flatten()
        .filter(|qt| is_float(qt))
        .all(|qt| {
            qt.shape()
                .dims()
                .last()
                .is_some_and(|k| k.is_multiple_of(GgmlDType::Q4K.block_size()))
        });
        if whole_q4k_rows {
            GgmlDType::Q4K
        } else {
            GgmlDType::Q8_0
        }
    };

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
            // One block format for every per-layer weight, so that the QKV and FFN
            // gate+up fusions apply.
            let qt_q4 = requant_float_weight(qt, requant_dtype, device, "projection")?;
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
            requant_float_weight(qt, requant_dtype, device, "projection")
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
            // CONVERSION: q, k and v are copied into one fused weight, one GEMV per token.
            let fused_storage = crate::tensor::quantized::QStorage::from_data(
                std::borrow::Cow::Borrowed(combined_slice),
                device,
                dtype,
            )?;
            drop(buf);
            // A card holds the fused blocks alone; only a host layer keeps an owned copy.
            if !on_gpu {
                crate::tensor::conversion::record_copy(
                    "projection",
                    dtype.gguf_name(),
                    total as u64,
                    "q, k and v fused into one weight, one GEMV per token",
                );
            }
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
            // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
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
                // CONVERSION: gate and up are copied into one fused weight, one GEMV per token.
                let fused_storage = crate::tensor::quantized::QStorage::from_data(
                    std::borrow::Cow::Borrowed(combined_slice),
                    device,
                    dtype,
                )?;
                drop(buf); // safe: from_data already copied to GPU
                           // A card holds the fused blocks alone; only a host layer keeps an owned copy.
                if !on_gpu {
                    crate::tensor::conversion::record_copy(
                        "projection",
                        dtype.gguf_name(),
                        total as u64,
                        "gate and up fused into one weight, one GEMV per token",
                    );
                }
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
                    // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
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
                // CONVERSION: the fused norm kernel reads the scale as F32.
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
                // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
                b.dequantize(&Device::Cpu)?.to_device(device)?
            } else {
                b.dequantize(device)?
            })),
            None => Ok(None),
        }
    };
    let attn_output_bias = bias_to_device(raw.attn_output_bias_qt)?;
    let attn_sinks_host = raw
        .attn_sinks_qt
        .as_ref()
        // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
        .map(|qt| qt.dequantize(&Device::Cpu)?.to_vec1::<f32>())
        .transpose()?;
    let attn_sinks = bias_to_device(raw.attn_sinks_qt)?;
    let ffn_up_bias = bias_to_device(raw.ffn_up_bias_qt)?;
    let ffn_down_bias = bias_to_device(raw.ffn_down_bias_qt)?;
    // Extract raw weight for fused add+rmsnorm kernel before build_norm consumes QTensor
    let ffn_norm_weight_for_fused = match &raw.ffn_norm_qt {
        // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
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
    let rope_base = if head_dim < config.head_dim {
        // SWA layers (Gemma4) use a different RoPE freq base
        config.rope_freq_base_swa.unwrap_or(config.rope_freq_base)
    } else {
        config.rope_freq_base
    };
    let (cos, sin) = if head_dim != config.head_dim || rope_dim > 0 {
        // YaRN applies to full-head-dim layers; SWA/partial layers (head_dim<config)
        // keep plain rope (their lower freq base already shortens their context).
        let yarn = if head_dim < config.head_dim {
            None
        } else {
            config.yarn
        };
        ropes.table(
            effective_rope_dim,
            rope_base,
            yarn,
            config
                .effective_max_context()
                .max(crate::inference::generic_transformer::rope::ROPE_TABLE_SEQ_LEN),
            device,
        )?
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
    // An expert stack is used in the format the file stores it in whenever a kernel on the
    // device it lives on reads that format. Only a format no kernel there reads is rebuilt,
    // as Q8_0.
    fn coerce_moe_weight(
        qt: crate::tensor::quantized::QTensor,
        device: &Device,
    ) -> Result<crate::tensor::quantized::QTensor> {
        use crate::tensor::quantized::GgmlDType;
        // A stack the plan keeps in host memory stays there whatever device its layer runs on.
        let home = qt.device().clone();
        let device = if home.runs_as_card() { device } else { &home };
        if moe_stack_served_as_stored(qt.dtype(), device.runs_as_card()) {
            return Ok(qt);
        }
        let before = qt.shape().elem_count() as u64 * qt.dtype().type_size() as u64
            / qt.dtype().block_size() as u64;
        let from = qt.dtype();
        // CONVERSION: no expert kernel on this device reads the stored format.
        let dequant = qt.dequantize(&Device::Cpu)?;
        // CONVERSION: no expert kernel on this device reads the stored format.
        let out =
            crate::tensor::quantized::QTensor::quantize_onto(&dequant, GgmlDType::Q8_0, device)?;
        let after = out.shape().elem_count() as u64 * GgmlDType::Q8_0.type_size() as u64
            / GgmlDType::Q8_0.block_size() as u64;
        crate::tensor::conversion::record_kept(
            "experts",
            from.gguf_name(),
            GgmlDType::Q8_0.gguf_name(),
            before,
            after,
            "no expert kernel reads the stored format on this device",
        );
        Ok(out)
    }

    let moe = if let (Some(gi), Some(des)) = (raw.moe_gate_inp, raw.moe_down_exps) {
        // gate||up experts: fused (gemma4-MoE `ffn_gate_up_exps`) OR separate
        // gate/up (standard MoE like granitemoe). For separate, concat the
        // dequantized gate then up experts along the output (ffn) dim into the
        // fused [n_exp, 2*ffn, hidden] layout, then requantize to Q8.
        let (gus, fused_f32, router_prenormed, use_silu, gate_sep, up_sep) =
            match (raw.moe_gate_up_exps, raw.moe_gate_exps, raw.moe_up_exps) {
                // Clamped-SwiGLU experts read gate and up apart on every device, so no
                // fused stack or host cache is built for them.
                (None, Some(ge), Some(ue)) if config.expert_swiglu_clamp.is_some() => (
                    None,
                    None,
                    true,
                    true,
                    Some(std::sync::Arc::new(coerce_moe_weight(ge, device)?)),
                    Some(std::sync::Arc::new(coerce_moe_weight(ue, device)?)),
                ),
                // Gateless experts: the up stack alone, squared ReLU after it.
                (None, None, Some(ue)) => (
                    None,
                    None,
                    true,
                    false,
                    None,
                    Some(std::sync::Arc::new(coerce_moe_weight(ue, device)?)),
                ),
                (Some(gus_fused), _, _) => (
                    Some(coerce_moe_weight(gus_fused, device)?),
                    None,
                    false,
                    false,
                    None,
                    None,
                ),
                // A SiLU mixture reads gate and up apart on every path, card and host, in
                // their stored format: nothing else is built for it.
                (None, Some(ge), Some(ue)) if !flags.use_gelu => (
                    None,
                    None,
                    true,
                    true,
                    Some(std::sync::Arc::new(coerce_moe_weight(ge, device)?)),
                    Some(std::sync::Arc::new(coerce_moe_weight(ue, device)?)),
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
                    (Some(gus), Some(fused), true, true, Some(gate_q), Some(up_q))
                }
                _ => {
                    return Err(crate::tensor::Error::msg(
                        "MoE layer has neither fused nor separate gate/up experts",
                    ))
                }
            };
        let gate_inp_scale = match raw.moe_gate_inp_scale_qt {
            Some(qt) => {
                // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
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
                // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
                let t = qt.dequantize(&Device::Cpu)?;
                Some(if on_gpu { t.to_device(device)? } else { t })
            }
            None => None,
        };
        let decode_slots = if on_gpu {
            let ids: Vec<u32> = (0..flags.n_experts_used as u32).collect();
            Some(Tensor::from_vec(ids, (flags.n_experts_used,), &Device::Cpu)?.to_device(device)?)
        } else {
            None
        };
        let des = coerce_moe_weight(des, device)?;
        // A stack in host memory is read in place by the host expert loop, which decodes only
        // the experts a call routes to. A stack on a card has no host blocks to slice, and the
        // layer's host prefill reads a BF16 cache built here, declared in the ledger.
        let stacks_on_card =
            gus.as_ref().is_some_and(|g| g.device().runs_as_card()) || des.device().runs_as_card();
        let (gus_f32, des_f32) = if stacks_on_card && use_silu && gus.is_some() {
            // CONVERSION: a card layer's host prefill reads its experts dense; no host blocks exist to slice.
            let g = match (&fused_f32, &gus) {
                (Some(f), _) => f.to_dtype(crate::tensor::DType::BF16)?,
                (None, Some(gus)) => gus
                    .dequantize(&Device::Cpu)?
                    .to_dtype(crate::tensor::DType::BF16)?,
                (None, None) => unreachable!("guarded by gus.is_some()"),
            };
            // CONVERSION: a card layer's host prefill reads its experts dense; no host blocks exist to slice.
            let d = des
                .dequantize(&Device::Cpu)?
                .to_dtype(crate::tensor::DType::BF16)?;
            crate::tensor::conversion::record_kept(
                "experts",
                des.dtype().gguf_name(),
                "bf16",
                (des.shape().elem_count() as u64 * des.dtype().type_size() as u64)
                    / des.dtype().block_size() as u64,
                (g.elem_count() + d.elem_count()) as u64 * 2,
                "host prefill of a card layer reads a dense expert cache",
            );
            (Some(std::sync::Arc::new(g)), Some(std::sync::Arc::new(d)))
        } else {
            (None, None)
        };
        let swiglu_clamp = config.expert_swiglu_clamp.filter(|_| gate_sep.is_some());
        let up_only = gus.is_none() && gate_sep.is_none() && up_sep.is_some();
        let shared_expert = match (raw.shexp_up, raw.shexp_down) {
            (Some(up), Some(down)) => Some(SharedExpert {
                gate: to_qmatmul(raw.shexp_gate)?,
                up: to_qmatmul(Some(up))?.unwrap(),
                down: to_qmatmul(Some(down))?.unwrap(),
                scalar_gate: load_bias(raw.shexp_scalar_gate_qt)?
                    .map(|w| {
                        let n = w.elem_count();
                        // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
                        w.to_dtype(crate::tensor::DType::F32)?.reshape((n, 1))
                    })
                    .transpose()?,
            }),
            _ => None,
        };
        let gate_inp_bias = load_bias(raw.moe_gate_inp_bias_qt)?;
        let gate_exps_bias = load_bias(raw.moe_gate_exps_bias_qt)?;
        let up_exps_bias = load_bias(raw.moe_up_exps_bias_qt)?;
        let down_exps_bias = load_bias(raw.moe_down_exps_bias_qt)?;
        let biased = gate_inp_bias.is_some()
            || gate_exps_bias.is_some()
            || up_exps_bias.is_some()
            || down_exps_bias.is_some();
        if biased && (swiglu_clamp.is_none() || !router_prenormed) {
            return Err(crate::tensor::Error::msg(
                "MoE layer carries router or expert biases, which only the pre-normed \
                 clamped-SwiGLU expert block applies",
            ));
        }
        Some(MoeWeights {
            gate_inp: crate::tensor::quantized::QMatMul::from_qtensor(gi)?,
            gate_inp_scale,
            gate_inp_bias,
            gate_exps_bias,
            up_exps_bias,
            down_exps_bias,
            swiglu_clamp,
            gate_up_exps: gus.map(std::sync::Arc::new),
            down_exps: std::sync::Arc::new(des),
            gate_exps_sep: gate_sep,
            up_exps_sep: up_sep,
            down_exps_scale,
            decode_slots,
            n_experts: flags.n_experts,
            n_experts_used: flags.n_experts_used,
            expert_ffn_dim: flags.expert_ffn_dim,
            router_prenormed,
            use_silu,
            // Real activation, decoupled from the use_silu layout flag.
            gelu: flags.use_gelu,
            // Whether the chosen experts' softmax weights are rescaled to sum to one.
            norm_topk: config.expert_topk_norm,
            sigmoid_routing: config
                .expert_sigmoid_gating
                .then(|| -> Result<SigmoidRouting> {
                    Ok(SigmoidRouting {
                        selection_bias: load_bias(raw.moe_exp_probs_b_qt)?
                            // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
                            .map(|b| b.to_dtype(crate::tensor::DType::F32))
                            .transpose()?,
                        normalize: config.expert_weights_norm,
                        scale: config.expert_weights_scale,
                    })
                })
                .transpose()?,
            up_relu2: up_only,
            shared_expert,
            gate_up_exps_f32: gus_f32,
            down_exps_f32: des_f32,
        })
    } else {
        None
    };
    let pre_ffw_norm_2 = build_norm_opt(raw.pre_ffw_norm_2_qt)?;
    let post_ffw_norm_1 = build_norm_opt(raw.post_ffw_norm_1_qt)?;
    let post_ffw_norm_2 = build_norm_opt(raw.post_ffw_norm_2_qt)?;

    let mixer = match raw.mixer {
        RawMixer::Attention => super::mixer::Mixer::Attention,
        RawMixer::ShortConv { in_proj, conv } => {
            super::mixer::Mixer::ShortConv(super::mixer::ShortConv::new(
                to_qmatmul(Some(in_proj))?.unwrap(),
                // CONVERSION: a recurrent mixer holds its small convolution kernel F32 for the host recurrence.
                conv.dequantize(&Device::Cpu)?,
                device,
            )?)
        }
        RawMixer::Ssm {
            in_proj,
            conv_w,
            conv_b,
            a,
            d,
            dt_bias,
            norm,
        } => {
            let meta = config.ssm.ok_or_else(|| {
                crate::tensor::Error::msg("ssm layer in a file without ssm.inner_size")
            })?;
            // CONVERSION: a recurrent mixer holds its small weights F32 for the host recurrence.
            let host = |qt: QTensor| qt.dequantize(&Device::Cpu);
            let a = host(a)?;
            let dims = super::mixer::Mamba2Dims {
                d_inner: meta.inner_size,
                d_state: meta.state_size,
                ngroups: meta.group_count.max(1),
                d_conv: meta.conv_kernel,
                nheads: a.elem_count(),
            };
            let conv_b = match conv_b {
                Some(b) => host(b)?,
                None => Tensor::zeros(dims.d_xbc(), crate::tensor::DType::F32)?,
            };
            super::mixer::Mixer::Ssm(super::mixer::Mamba2::new(
                super::mixer::Mamba2Weights {
                    in_proj: to_qmatmul(Some(in_proj))?.unwrap(),
                    conv_w: host(conv_w)?,
                    conv_b,
                    a,
                    d: host(d)?,
                    dt_bias: host(dt_bias)?,
                    norm_w: host(norm)?,
                },
                dims,
                eps,
                device,
            )?)
        }
        RawMixer::DeltaNet {
            proj,
            conv_w,
            a,
            dt_bias,
            norm,
        } => {
            let meta = config.ssm.ok_or_else(|| {
                crate::tensor::Error::msg("delta-rule layer in a file without ssm.inner_size")
            })?;
            let dims = super::mixer::DeltaDims {
                k_heads: meta.group_count.max(1),
                v_heads: meta.time_step_rank,
                k_dim: meta.state_size,
                v_dim: meta.inner_size / meta.time_step_rank.max(1),
                conv_kernel: meta.conv_kernel,
                v_heads_tiled: meta.v_heads_tiled,
                gate_sigmoid: config.delta_output_gate_sigmoid,
            };
            let packed_ba = proj.len() == 3;
            let proj = match concat_rows(&proj, device)? {
                Some(fused) if !packed_ba => {
                    super::mixer::DeltaProj::Fused(QMatMul::from_qtensor(fused)?)
                }
                _ => {
                    let mut it = proj.into_iter();
                    let mut next = || -> Result<QMatMul> {
                        to_qmatmul(it.next()).and_then(|q| {
                            q.ok_or_else(|| {
                                crate::tensor::Error::msg("delta-rule projection missing")
                            })
                        })
                    };
                    let (qkv, z) = (next()?, next()?);
                    match packed_ba {
                        true => super::mixer::DeltaProj::Ba {
                            qkv,
                            z,
                            ba: next()?,
                        },
                        false => super::mixer::DeltaProj::Split {
                            qkv,
                            z,
                            alpha: next()?,
                            beta: next()?,
                        },
                    }
                }
            };
            // CONVERSION: a recurrent mixer holds its small weights F32 for the host recurrence.
            let host = |qt: QTensor| qt.dequantize(&Device::Cpu);
            super::mixer::Mixer::DeltaNet(super::mixer::GatedDeltaNet::new(
                proj,
                host(conv_w)?,
                host(a)?,
                host(dt_bias)?,
                host(norm)?,
                dims,
                eps,
                device,
            )?)
        }
        RawMixer::Absent => super::mixer::Mixer::Absent,
    };
    let attends = matches!(mixer, super::mixer::Mixer::Attention);

    // gemma4: pre-allocate the unit-weight tensor used by the V-RMS-norm
    // pass (head_dim ones, F32). Built on the LAYER's device - CPU or GPU.
    // The V-norm is part of the gemma4 arch and the forward applies it via
    // `crate::tensor::ops::rms_norm` which runs on CPU too, so this must NOT be
    // gated on `on_gpu`: a gemma4 layer placed on CPU (multi-GPU+CPU hybrid
    // OR pure-CPU fallback) would otherwise have `attn_v_norm_ones = None`,
    // silently SKIP the V-norm, and produce garbage attention on that layer
    // (the multi-device incoherence - single-GPU was fine because every
    // layer had it).
    let attn_v_norm_ones = if flags.value_rms_norm {
        Some(Tensor::ones(head_dim, crate::tensor::DType::F32, device)?)
    } else {
        None
    };

    // SWA classification: the explicit per-layer pattern (gemma4 `sliding_window_pattern`)
    // when the file has one, else the architecture's period, else the head_dim heuristic
    // (SWA layers have a smaller head than the global config).
    let is_swa = match config
        .swa_pattern
        .as_ref()
        .and_then(|p| p.get(raw.layer_idx))
    {
        Some(&s) => s,
        None => match config.swa_period {
            Some(period) => windowed_by_period(period, raw.layer_idx),
            None => head_dim < config.head_dim,
        },
    };
    let layer_window = config.sliding_window.filter(|_| is_swa);
    let has_sinks = attn_sinks.is_some();

    let (hyper, sparse, ngram) = {
        let wide = raw.wide;
        let dim = config.embedding_length;
        let build_mixer =
            |w: RawMixerWeights, streams: usize| -> Result<super::hyper::HyperMixer> {
                // CONVERSION: a mixer's norm scale is a small vector held dense on the layer's device, whatever the file stores.
                let gamma = w.norm.dequantize(&Device::Cpu)?.to_device(device)?;
                super::hyper::HyperMixer::new(
                    gamma,
                    super::wide::load_matrix(w.down)?,
                    super::wide::load_matrix(w.up)?,
                    w.inject.map(super::wide::load_matrix).transpose()?,
                    streams,
                    dim,
                    eps,
                )
            };
        let hyper = match (config.hyper, wide.attn_mixer, wide.ffn_mixer) {
            (Some(h), Some(a), Some(f)) => Some(super::hyper::WideBlock {
                attn: build_mixer(a, h.streams)?,
                ffn: build_mixer(f, h.streams)?,
            }),
            _ => None,
        };
        // CONVERSION: the indexer's norm vectors are read to host F32, whatever the file stores.
        let host = |qt: QTensor| -> Result<Vec<f32>> {
            qt.dequantize(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()
        };
        let sparse = match (config.indexer, wide.indexer) {
            (Some(meta), Some(ix)) => Some(super::qsa::SparseAttention::new(
                meta,
                super::wide::load_matrix(ix.q_proj)?,
                super::wide::load_matrix(ix.k_proj)?,
                host(ix.q_norm)?,
                host(ix.k_norm)?,
                eps as f32,
                config.rope_dim.filter(|&d| d > 0).unwrap_or(meta.head_dim),
                config.rope_freq_base,
            )),
            _ => None,
        };
        let ngram = match (&config.ngram, config.hyper, wide.ngram) {
            (Some(meta), Some(h), Some(n)) => {
                // CONVERSION: the n-gram block's norm vectors are small and held dense, whatever the file stores.
                let dense = |qt: QTensor| -> Result<Tensor> { qt.dequantize(&Device::Cpu) };
                Some(super::ngram::PleBlock::new(
                    super::wide::load_matrix(n.key)?,
                    super::wide::load_matrix(n.value)?,
                    dense(n.norm_key)?,
                    dense(n.norm_query)?,
                    dense(n.norm_conv)?,
                    dense(n.conv)?,
                    meta,
                    h.streams,
                    dim,
                    eps,
                    device,
                )?)
            }
            _ => None,
        };
        (hyper, sparse, ngram)
    };

    let mut layer = GenericTransformerLayer {
        mixer,
        has_ffn: raw.has_ffn,
        attn_gate: to_qmatmul(raw.attn_gate)?,
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
        attn_sinks,
        attn_sinks_host,
        no_rope: crate::inference::model_facts::no_rope_layer_period(&config.arch)
            .is_some_and(|period| (raw.layer_idx + 1).is_multiple_of(period)),
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
                // CONVERSION: a per-layer gate vector is held F32.
                let t = qt.dequantize(&Device::Cpu)?;
                if on_gpu {
                    t.to_device(device)
                } else {
                    Ok(t)
                }
            })
            .transpose()?,
        // KV cache initial sizing: the working window, doubled on demand up to the model's
        // context. The dense cache is filled by F-dtype layers and by the fused attention of
        // quantised layers alike, so allocating the whole context up front puts gigabytes
        // per layer on the card for a window nobody has used. A captured graph bound to the
        // old width is dropped and recorded again when the cache grows, and a growth that
        // runs out of card memory is recovered by the placement ladder.
        kv_cache: {
            use crate::inference::cache::KV_WORKING_WINDOW_TOKENS as LAZY_KV_INITIAL_CAP;
            let init_cap = config
                .effective_max_context()
                .max(512)
                .min(LAZY_KV_INITIAL_CAP);
            let mut c = SpecKvCache::new(init_cap);
            // gemma4 bounded sliding-window attention: SWA layers only attend within
            // `sliding_window`, so the cache need keep only the last
            // `window + prefill_chunk` keys - bounding it lets the F16 KV path
            // stay VRAM-cheap at long context (where Q8/Q4 KV degenerated for
            // gemma4's tiny K-norm). Exact: the bounded prefill mask matches.
            // (set_window overrides max_seq_len; init_cap above governs the
            // GLOBAL, non-windowed layers.) Same SWA classification as the
            // per-layer `sliding_window` below.
            // A period-classified model keeps every position: its layers window at
            // attention time.
            if layer_window.is_some() && config.swa_period.is_none() {
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
        quant_kv: {
            use crate::inference::cache::quant_kv::QuantKv;
            use crate::inference::engine::llm_engine::KvQuant;
            // The layer's own head shape must be one the quantised kernels are instantiated
            // for. The cache is sized with the PER-LAYER n_kv_head: gemma4 26B's model-level
            // value is the worst case (16) while its SWA layers have 8, and the model-level
            // value would over-reserve every SWA layer twice and OOM at load.
            let shape_supported = crate::inference::cache::kv_core::kv_shape_supported(
                head_dim,
                config.n_head / n_kv_head_layer.max(1),
            );
            let max_seq = config.effective_max_context();
            // The quantised attention kernels have no sink term: a sink layer keeps its
            // history in the F-dtype cache.
            let built = if device.is_cuda() && shape_supported && !has_sinks && attends {
                match config.kv_quant {
                    KvQuant::Q8 => Some(
                        crate::inference::cache::q8_kv::Q8KvCache::new(
                            max_seq,
                            n_kv_head_layer,
                            head_dim,
                            device,
                        )
                        .map(QuantKv::Q8),
                    ),
                    KvQuant::Q4 => Some(
                        crate::inference::cache::q4_kv::Q4KvCache::new(
                            max_seq,
                            n_kv_head_layer,
                            head_dim,
                            device,
                        )
                        .map(QuantKv::Q4),
                    ),
                    KvQuant::Off => None,
                }
            } else {
                None
            };
            match built {
                Some(Ok(c)) => Some(c),
                Some(Err(e)) => {
                    tracing::warn!(
                        "{:?} KV cache alloc failed for layer (device={:?}): {e}; \
                         decoding will fall back to the F-dtype KV cache",
                        config.kv_quant,
                        device
                    );
                    None
                }
                None => None,
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
        // An unquantised cache on a card stores F16 when the architecture keeps its cache
        // unquantised: the half-width read stands in for the quantised one it cannot take.
        // A sink layer stores F16 too: its card decode attention reads F16 K/V.
        kv_f16: {
            use crate::inference::engine::llm_engine::KvQuant;
            (crate::inference::model_facts::facts_or_default(&config.arch).quantized_kv_degrades
                || has_sinks)
                && matches!(config.kv_quant, KvQuant::Off)
                && device.is_cuda()
        },
        padded_mask: None,
        graph_q_buffer: None,
        graph_rope_cos: None,
        graph_rope_sin: None,
        graph_kv_pos: None,
        graph_pos_i32: None,
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
                let factors_f32 = if factors.dtype() == crate::tensor::DType::F32 {
                    factors.clone()
                } else {
                    // CONVERSION: rope factors are held F32, one table per layer kind.
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
        sliding_window: layer_window,
        prefill_mask_window: if config.swa_period.is_some() {
            layer_window
        } else {
            config.sliding_window
        },
        rope_dim,
        rope_rows: None,
        rope_base,
        attention_scale: config.attention_scale,
        attn_softcap: config.attn_logit_softcapping.map(|c| c as f32),
        residual_scale: config.residual_scale,
        flags,
        hyper,
        sparse,
        ngram,
    };
    // A layer that selects tokens reads its history from the one cache the selection reads.
    #[cfg(feature = "cuda")]
    if layer.sparse.is_some() {
        layer.quant_kv = None;
    }
    Ok(layer)
}

/// Build one `GenericTransformerLayer` from an AWQ checkpoint. Dense
/// Qwen2/Qwen3/Mistral: separate q/k/v (so the GGUF-byte APQ fast path gates off
/// and `standard_attention` - which works with any `QMatMul` - runs), f16 norms,
/// F32 biases. All CUDA-graph buffers None; KV cache is F16 SpecKvCache
/// (kv_quant=Off for this first cut).
fn build_awq_layer(
    st: &crate::tensor::safetensors::MmapedSafetensors,
    layer_idx: usize,
    device: &Device,
    config: &GenericTransformerConfig,
    flags: Arc<GenericLayerFlags>,
    cos: &Tensor,
    sin: &Tensor,
    group_size: usize,
    keep_free: usize,
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
            // CONVERSION: AWQ scales are held F16, the width the AWQ kernels read.
            .to_dtype(DType::F16)?
            .contiguous()?;
        let k = qweight.dim(0)?;
        let n = scales.dim(1)?;
        Ok(QMatMul::from_awq(
            AwqWeight {
                qweight,
                qzeros,
                scales,
                bias: None,
                k,
                n,
                group_size,
                repacked: std::sync::OnceLock::new(),
                marlin: std::sync::OnceLock::new(),
            },
            keep_free,
        ))
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
            // CONVERSION: AWQ scales are held F16, the width the AWQ kernels read.
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
                // CONVERSION: AWQ scales are held F16, the width the AWQ kernels read.
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
        Ok(QMatMul::from_awq(
            AwqWeight {
                qweight,
                qzeros,
                scales,
                bias: None,
                k,
                n,
                group_size,
                repacked: std::sync::OnceLock::new(),
                marlin: std::sync::OnceLock::new(),
            },
            keep_free,
        ))
    };
    let _ = &load_raw;
    // Optional bias -> F32 on device.
    let load_bias = |proj: &str| -> Result<Option<Tensor>> {
        match st.load(&format!("{p}.{proj}.bias"), device) {
            // CONVERSION: a bias is a small vector held F32.
            Ok(t) => Ok(Some(t.to_dtype(DType::F32)?)),
            Err(_) => Ok(None),
        }
    };
    let load_norm = |name: &str| -> Result<RmsNorm> {
        let w = st
            .load(&format!("{p}.{name}"), device)?
            // CONVERSION: a norm scale is a small vector held F32.
            .to_dtype(DType::F32)?;
        Ok(RmsNorm::from_tensor(w, eps))
    };

    // -- Fused QKV -> routes decode through the APQ fast path (fused norm+RoPE+cast
    //    + Q8 attention), eliminating 2 GEMV launches/layer. The per-head q/k norms are
    //    applied to the fused projection by that path, so a model with them fuses too.
    let fuse_qkv = true;
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
        mixer: super::mixer::Mixer::Attention,
        has_ffn: true,
        attn_gate: None,
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
        attn_sinks: None,
        attn_sinks_host: None,
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
        kv_cache: SpecKvCache::new(
            config
                .effective_max_context()
                .max(512)
                .min(crate::inference::cache::KV_WORKING_WINDOW_TOKENS),
        ),
        cpu_q8_kv: None,
        cpu_f16_kv: None,
        cpu_decode_arena: None,
        cpu_norm_cache: None,
        #[cfg(feature = "cuda")]
        quant_kv: {
            let head_dim_supported = crate::inference::cache::kv_core::kv_shape_supported(
                head_dim,
                config.n_head / config.n_kv_head.max(1),
            );
            if device.is_cuda() && head_dim_supported {
                crate::inference::cache::q8_kv::Q8KvCache::new(
                    config.effective_max_context(),
                    config.n_kv_head,
                    head_dim,
                    device,
                )
                .ok()
                .map(crate::inference::cache::quant_kv::QuantKv::Q8)
            } else {
                None
            }
        },
        populate_dual_kv: false,
        kv_f16: false,
        padded_mask: None,
        graph_q_buffer: None,
        graph_rope_cos: None,
        graph_rope_sin: None,
        graph_kv_pos: None,
        graph_pos_i32: None,
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
        prefill_mask_window: config.sliding_window,
        rope_dim: 0,
        rope_rows: None,
        rope_base: config.rope_freq_base,
        no_rope: false,
        attention_scale: config.attention_scale,
        attn_softcap: config.attn_logit_softcapping.map(|c| c as f32),
        residual_scale: config.residual_scale,
        flags,
        hyper: None,
        sparse: None,
        ngram: None,
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

    /// [`Self::from_gguf_placed`] with the embedding table on the plan's card when the plan
    /// runs on one card, else on the host.
    pub fn from_gguf_with_kv_quant(
        content: gguf_file::Content,
        mmap_bytes: &[u8],
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
    ) -> Result<Self> {
        Self::from_gguf_placed(
            content,
            mmap_bytes,
            cuda_devices,
            plan,
            kv_quant,
            max_kv_seq_len,
            crate::inference::place::ladder::on_one_card(plan),
        )
    }

    /// Load from GGUF with every layer on the device `plan` gives it, the output head on the
    /// last card a layer runs on, and the embedding table on the first layer's card when
    /// `stem_on_card` says the plan holds room for it there, else on the host. Nothing is
    /// placed by reading the cards: a card that runs out of memory fails the load, and the
    /// caller's ladder places it again.
    pub fn from_gguf_placed(
        content: gguf_file::Content,
        mmap_bytes: &[u8],
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
        stem_on_card: bool,
    ) -> Result<Self> {
        // One file: a part of a set is refused rather than built from the blocks this file
        // happens to carry. A set is read as one model through `from_split_gguf`.
        super::config::refuse_a_split_part(&content)?;
        let source = TensorSource::one(&content, mmap_bytes);
        Self::from_gguf_source(
            &content,
            source,
            cuda_devices,
            plan,
            kv_quant,
            max_kv_seq_len,
            stem_on_card,
        )
    }

    /// A split set read as one model: the header is the union of the parts and every tensor is
    /// read from the part that carries it, so a set loads as the merged file would.
    pub fn from_split_gguf(
        split: &crate::tensor::quantized::gguf_source::SplitGguf,
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
        stem_on_card: bool,
    ) -> Result<Self> {
        let content = split.merged_content();
        let source = TensorSource::of_split(split);
        Self::from_gguf_source(
            &content,
            source,
            cuda_devices,
            plan,
            kv_quant,
            max_kv_seq_len,
            stem_on_card,
        )
    }

    /// The load itself: `content` is the header the model is read from - one file's, or a set's
    /// union - and `source` says where each tensor's bytes are.
    fn from_gguf_source(
        content: &gguf_file::Content,
        source: TensorSource<'_>,
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
        max_kv_seq_len: Option<usize>,
        _stem_on_card: bool,
    ) -> Result<Self> {
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

        let mut cfg = GenericTransformerConfig::from_gguf(content, &arch)?;
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
        // The table sits on the first layer's card when the plan made room for it there:
        // every token's lookup then stays on the card instead of crossing from the host.
        // The embed-lookup callsites read `embeddings.device()` and cast the
        // ids to it, so this is the only switch.
        let first_card = plan.segments.first().and_then(|seg| match seg.kind {
            DeviceKind::Cuda(idx) => cuda_devices.get(&idx).cloned(),
            _ => None,
        });
        let emb_qt = source.read("token_embd.weight", &Device::Cpu)?;
        // The table stays in the file's format and each lookup decodes the rows it needs:
        // resident cost is the mapping, and the residual stream is still F32.
        let emb_table_bytes = emb_qt.shape().elem_count() as u64
            * emb_qt.dtype().type_size() as u64
            / emb_qt.dtype().block_size() as u64;
        let emb_device = Device::Cpu;
        let embeddings = Embedding::from_stored(Arc::new(emb_qt))?;
        info!(
            "✅ Loaded embeddings: vocab={}, dim={} ({} MB stored, decoded per lookup on {:?})",
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
        // LFM2 files name the final norm `token_embd_norm`; it is read under that name only
        // when the file carries no `output_norm`.
        // A wide-residual file has none: its final mixer is the output norm, so the plain one
        // is a placeholder the forward never reads.
        let norm_weight_cpu =
            if config.hyper.is_some() && !content.tensor_infos.contains_key("output_norm.weight") {
                Tensor::ones(
                    config.embedding_length,
                    crate::tensor::DType::F32,
                    &Device::Cpu,
                )?
            } else {
                let norm_qt = match source.read("output_norm.weight", &Device::Cpu) {
                    Ok(qt) => qt,
                    Err(e) => match content.tensor_infos.contains_key("token_embd_norm.weight") {
                        true => source.read("token_embd_norm.weight", &Device::Cpu)?,
                        false => return Err(e),
                    },
                };
                // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
                norm_qt.dequantize(&Device::Cpu)?
            };
        // Phi2/GPT-NeoX use full LayerNorm: `output_norm.bias` exists.
        let output_norm_bias_cpu = source
            .read_opt("output_norm.bias", &Device::Cpu)
            .ok()
            .flatten()
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
        let output_bias = source
            .read_opt("output.bias", &Device::Cpu)
            .ok()
            .flatten()
            // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
            .and_then(|qt| qt.dequantize(&Device::Cpu).ok());
        let (output_qt, output_weight_name) = match source.read("output.weight", &Device::Cpu) {
            Ok(qt) => (qt, "output.weight"),
            Err(_) => {
                // Tied embeddings (weight sharing)
                (
                    source.read("token_embd.weight", &Device::Cpu)?,
                    "token_embd.weight",
                )
            }
        };
        let output_proj = crate::tensor::quantized::QMatMul::from_qtensor(output_qt)?;
        info!("✅ Loaded output norm and projection on CPU");

        // -- 2b. Load PLE globals (Gemma4) - optional ---------------------
        // Per-layer input embeddings are a model-wide table plus a projection. A file whose
        // `per_layer_token_embd` is a hashed gather table has no projection, and is read by
        // row elsewhere: it must not be expanded here.
        let ple_dim = if content
            .tensor_infos
            .contains_key("per_layer_model_proj.weight")
        {
            meta_u32(content, &config.arch, "embedding_length_per_layer_input").unwrap_or(0)
                as usize
        } else {
            0
        };
        // The per-layer token table is `vocab x (layers * ple_dim)`: the largest tensor of a
        // model that has one. It stays in the file's format and each lookup decodes its rows.
        let ple_token_embd: Option<Embedding> = if ple_dim > 0 {
            match source.read_opt("per_layer_token_embd.weight", &Device::Cpu) {
                Ok(None) => None,
                Ok(Some(qt)) => match Embedding::from_stored(Arc::new(qt)) {
                    Ok(e) => {
                        tracing::info!("PLE: per-layer token table kept in the stored format");
                        Some(e)
                    }
                    Err(e) => {
                        warn!("PLE token table unusable: {e}");
                        None
                    }
                },
                Err(e) => {
                    warn!("PLE tensor read failed: {e}");
                    None
                }
            }
        } else {
            None
        };
        // Load PLE projection on the first layer's card when there is one - every prefill runs
        // a [batch*seq, hidden] @ [hidden, n_layers*ple_dim] matmul through
        // `ple_model_proj.forward(hidden_2d)`. On a CPU device that becomes
        // a 28+ GFLOPS dequantize-then-matmul at 1k-token prefill, dominating
        // the whole forward and forcing a synchronous device->host copy of
        // the residual stream every prompt. Loading on CUDA flips it to the
        // tensor-core fast_mmq path. Fall back to CPU only when no GPU.
        let ple_dev = first_card.clone().unwrap_or(Device::Cpu);
        let ple_model_proj = if ple_dim > 0 {
            source
                .read_opt("per_layer_model_proj.weight", &ple_dev)
                .ok()
                .flatten()
                .map(crate::tensor::quantized::QMatMul::from_qtensor)
                .transpose()?
        } else {
            None
        };
        let ple_proj_norm = if ple_dim > 0 {
            source
                .read_opt("per_layer_proj_norm.weight", &ple_dev)
                .ok()
                .flatten()
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
        let rope_freq_factors = source
            .read_opt("rope_freqs.weight", &Device::Cpu)
            .ok()
            .flatten()
            // CONVERSION: norm scales, biases and gates are small vectors, held F32 whatever the file stores.
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

        // A RoPE table on each card the plan runs a layer on, and on no other: a cached
        // table pins its card's context.
        let planned_cuda: std::collections::HashSet<usize> = plan
            .segments
            .iter()
            .filter_map(|seg| match seg.kind {
                DeviceKind::Cuda(i) => Some(i),
                _ => None,
            })
            .collect();
        for (&idx, dev) in cuda_devices.iter() {
            if !planned_cuda.contains(&idx) {
                continue;
            }
            // Vacant-slot form: the body uses ?, so or_insert_with can't propagate it.
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

        crate::inference::place::device_probe::debug_vram_by_card("after RoPE precompute");
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
                let per_layer = (source.total_bytes()) / (n_layers.max(1) as u64);
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
        let f = &flags;
        let ti = &content.tensor_infos;

        let load_one = |idx: usize, dk: DeviceKind| -> crate::tensor::Result<RawGenericLayer> {
            let prefix = format!("blk.{idx}");
            let weight_dev = match dk {
                DeviceKind::Cuda(gpu_idx) => cuda_devices.get(&gpu_idx).unwrap_or(&Device::Cpu),
                _ => &Device::Cpu,
            };

            macro_rules! req {
                ($name:expr) => {
                    source.read($name, weight_dev)
                };
            }
            // Biases and norms always on CPU
            macro_rules! opt_cpu {
                ($name:expr) => {
                    source.read_opt($name, &Device::Cpu)
                };
            }
            macro_rules! req_cpu {
                ($name:expr) => {
                    source.read($name, &Device::Cpu)
                };
            }
            // A tiny Q8 weight for a projection field the layer never runs.
            let placeholder = || -> crate::tensor::Result<QTensor> {
                let z =
                    Tensor::zeros_on((32usize, 32usize), crate::tensor::DType::F32, weight_dev)?;
                // CONVERSION: a 32 by 32 placeholder for a projection field the layer never runs.
                QTensor::quantize_onto(&z, crate::tensor::quantized::GgmlDType::Q8_0, weight_dev)
            };

            let mixer_kind = super::mixer::layer_mixer_kind(content, idx);
            let attends = mixer_kind == super::mixer::MixerKind::Attention;
            let has_ffn = super::mixer::layer_has_ffn(content, idx);
            if ti.contains_key(&format!("{prefix}.ffn_latent_down.weight")) {
                return Err(crate::tensor::Error::msg(format!(
                    "{prefix}: routed experts behind a latent projection are not supported"
                )));
            }
            // Attention
            let mut attn_gate = None;
            let (attn_q, attn_k, attn_v, attn_qkv) = if !attends {
                (None, None, None, None)
            } else if f.fused_qkv {
                let qkv = req!(&format!("{prefix}.attn_qkv.weight"))?;
                (None, None, None, Some(qkv))
            } else {
                let q = req!(&format!("{prefix}.attn_q.weight"))?;
                let q = match config.attn_output_gate {
                    true => {
                        let (q, gate) = split_q_gate(q, config.n_head, weight_dev)?;
                        attn_gate = Some(gate);
                        q
                    }
                    false => q,
                };
                let k = req!(&format!("{prefix}.attn_k.weight"))?;
                // Gemma4 26B Global layers omit attn_v and reuse attn_k
                // as V (AttentionKEqV pattern). Use opt-load for V.
                let v = source.read_opt(&format!("{prefix}.attn_v.weight"), weight_dev)?;
                (Some(q), Some(k), v, None)
            };

            // Detect per-layer Q output dim from GGUF tensor shape.
            // In GGUF, QMatMul output dim = last dim of stored shape.
            // Gemma4 has per-layer head_dim (SWA=256, full=512).
            // Q output dim: the loader reverses GGUF dims, so the FIRST dim
            // in the reference Shape is the output dim (last GGUF dim).
            let q_out_dim = if attends && !f.fused_qkv {
                ti.get(&format!("{prefix}.attn_q.weight"))
                    .map(|info| info.shape.dims()[0] / if attn_gate.is_some() { 2 } else { 1 })
            } else {
                None // fused QKV uses config head_dim
            };
            let k_out_dim = if attends && !f.fused_qkv {
                ti.get(&format!("{prefix}.attn_k.weight"))
                    .map(|info| info.shape.dims()[0])
            } else {
                None
            };

            let attn_output_stem = attn_output_name(content);
            let (attn_output, mixer) = match mixer_kind {
                super::mixer::MixerKind::Attention => (
                    req!(&format!("{prefix}.{attn_output_stem}.weight"))?,
                    RawMixer::Attention,
                ),
                super::mixer::MixerKind::ShortConv => (
                    req!(&format!("{prefix}.shortconv.out_proj.weight"))?,
                    RawMixer::ShortConv {
                        in_proj: req!(&format!("{prefix}.shortconv.in_proj.weight"))?,
                        conv: req_cpu!(&format!("{prefix}.shortconv.conv.weight"))?,
                    },
                ),
                super::mixer::MixerKind::DeltaNet => {
                    let tail = match ti.contains_key(&format!("{prefix}.ssm_ba.weight")) {
                        true => vec!["ssm_ba"],
                        false => vec!["ssm_alpha", "ssm_beta"],
                    };
                    let mut proj = Vec::with_capacity(4);
                    for stem in ["attn_qkv", "attn_gate"].into_iter().chain(tail) {
                        proj.push(req!(&format!("{prefix}.{stem}.weight"))?);
                    }
                    // The dt bias is stored bare or as a `.bias`.
                    let dt_bias = match ti.contains_key(&format!("{prefix}.ssm_dt")) {
                        true => req_cpu!(&format!("{prefix}.ssm_dt"))?,
                        false => req_cpu!(&format!("{prefix}.ssm_dt.bias"))?,
                    };
                    (
                        req!(&format!("{prefix}.ssm_out.weight"))?,
                        RawMixer::DeltaNet {
                            proj,
                            conv_w: req_cpu!(&format!("{prefix}.ssm_conv1d.weight"))?,
                            a: req_cpu!(&format!("{prefix}.ssm_a"))?,
                            dt_bias,
                            norm: req_cpu!(&format!("{prefix}.ssm_norm.weight"))?,
                        },
                    )
                }
                super::mixer::MixerKind::Ssm => (
                    req!(&format!("{prefix}.ssm_out.weight"))?,
                    RawMixer::Ssm {
                        in_proj: req!(&format!("{prefix}.ssm_in.weight"))?,
                        conv_w: req_cpu!(&format!("{prefix}.ssm_conv1d.weight"))?,
                        conv_b: opt_cpu!(&format!("{prefix}.ssm_conv1d.bias"))?,
                        a: req_cpu!(&format!("{prefix}.ssm_a"))?,
                        d: req_cpu!(&format!("{prefix}.ssm_d"))?,
                        dt_bias: req_cpu!(&format!("{prefix}.ssm_dt.bias"))?,
                        norm: req_cpu!(&format!("{prefix}.ssm_norm.weight"))?,
                    },
                ),
                // A feed-forward block alone: its output projection is never read, so a tiny
                // placeholder stands in for the non-optional field.
                super::mixer::MixerKind::Absent => (placeholder()?, RawMixer::Absent),
            };
            // Phi2: attn_output bias
            let attn_output_bias_qt = if attends && f.has_attn_output_bias {
                opt_cpu!(&format!("{prefix}.{attn_output_stem}.bias"))?
            } else {
                None
            };
            let attn_sinks_qt = if attends && f.has_attn_sinks {
                let name = attn_sinks_name(content, idx).ok_or_else(|| {
                    crate::tensor::Error::msg(format!("missing tensor: {prefix}.attn_sinks"))
                })?;
                Some(req_cpu!(&name)?)
            } else {
                None
            };

            // Optional biases (Qwen2) - always CPU (small tensors)
            let (attn_q_bias, attn_k_bias, attn_v_bias) = if attends && f.has_qkv_bias {
                let bq = opt_cpu!(&format!("{prefix}.attn_q.bias"))?;
                let bk = opt_cpu!(&format!("{prefix}.attn_k.bias"))?;
                let bv = opt_cpu!(&format!("{prefix}.attn_v.bias"))?;
                (bq, bk, bv)
            } else {
                (None, None, None)
            };

            // QK norms (Gemma3) - always CPU
            let (attn_q_norm_qt, attn_k_norm_qt) = if attends && f.has_qk_norm {
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
            let ffn_norm_qt = if !has_ffn {
                None
            } else if mixer_kind == super::mixer::MixerKind::Absent {
                // A block that is a feed-forward alone normalises its input with its one norm.
                Some(req_cpu!(&format!("{prefix}.attn_norm.weight"))?)
            } else if f.parallel_attn || config.hyper.is_some() {
                None
            } else if f.has_post_ffn_norm {
                // OLMo2 post-norm-only: pre-ffn-norm absent -> Identity at build.
                opt_cpu!(&format!("{prefix}.ffn_norm.weight"))?
            } else {
                Some(req_cpu!(&format!(
                    "{prefix}.{}.weight",
                    config.ffn_norm_stem
                ))?)
            };
            let post_ffn_norm_qt = if f.has_post_ffn_norm {
                opt_cpu!(&format!("{prefix}.post_ffw_norm.weight"))?
            } else {
                None
            };

            // FFN
            let (ffn_gate, ffn_up, ffn_down) = if !has_ffn {
                (None, placeholder()?, placeholder()?)
            } else if f.is_moe && !ti.contains_key(&format!("{prefix}.ffn_up.weight")) {
                // Standard MoE (granitemoe): the experts ARE the FFN - there is
                // no dense/shared FFN (unlike gemma4-MoE). ffn_up/ffn_down are
                // never forwarded (layer-forward branches to MoE-only), so load
                // tiny Q8 placeholders to satisfy the non-Option struct fields.
                (None, placeholder()?, placeholder()?)
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
                    source.read_opt($name, weight_dev)
                };
            }
            // Routed expert stacks: in host memory for a layer the plan keeps its experts
            // there, else with the layer.
            let expert_dev = match plan.host_experts_from {
                Some(from) if idx >= from => &Device::Cpu,
                _ => weight_dev,
            };
            macro_rules! opt_exp {
                ($name:expr) => {
                    source.read_opt($name, expert_dev)
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
                opt_exp!(&format!("{prefix}.ffn_gate_up_exps.weight"))?
            } else {
                None
            };
            let moe_down_exps = if f.is_moe {
                opt_exp!(&format!("{prefix}.ffn_down_exps.weight"))?
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
                opt_exp!(&format!("{prefix}.ffn_gate_exps.weight"))?
            } else {
                None
            };
            let moe_up_exps = if f.is_moe {
                opt_exp!(&format!("{prefix}.ffn_up_exps.weight"))?
            } else {
                None
            };
            let moe_exp_probs_b_qt = if f.is_moe {
                opt_cpu!(&format!("{prefix}.exp_probs_b.bias"))?
            } else {
                None
            };
            let (
                moe_gate_inp_bias_qt,
                moe_gate_exps_bias_qt,
                moe_up_exps_bias_qt,
                moe_down_exps_bias_qt,
            ) = if f.is_moe {
                (
                    opt_cpu!(&format!("{prefix}.ffn_gate_inp.bias"))?,
                    opt_cpu!(&format!("{prefix}.ffn_gate_exps.bias"))?,
                    opt_cpu!(&format!("{prefix}.ffn_up_exps.bias"))?,
                    opt_cpu!(&format!("{prefix}.ffn_down_exps.bias"))?,
                )
            } else {
                (None, None, None, None)
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

            let wide = match config.hyper {
                None => RawWide::default(),
                Some(_) => {
                    let mixer_weights =
                        |stem: &str, inject: bool| -> crate::tensor::Result<RawMixerWeights> {
                            Ok(RawMixerWeights {
                                norm: req_cpu!(&format!("{prefix}.{stem}_norm.weight"))?,
                                down: req!(&format!("{prefix}.{stem}_down.weight"))?,
                                up: req!(&format!("{prefix}.{stem}_up.weight"))?,
                                inject: if inject {
                                    Some(req!(&format!("{prefix}.{stem}_inject.weight"))?)
                                } else {
                                    None
                                },
                            })
                        };
                    let indexer = match (attends, config.indexer) {
                        (true, Some(_)) => Some(RawIndexer {
                            q_proj: req!(&format!("{prefix}.indexer.q_proj.weight"))?,
                            k_proj: req!(&format!("{prefix}.indexer.k_proj.weight"))?,
                            q_norm: req_cpu!(&format!("{prefix}.indexer.q_norm.weight"))?,
                            k_norm: req_cpu!(&format!("{prefix}.indexer.k_norm.weight"))?,
                        }),
                        _ => None,
                    };
                    let ngram = match &config.ngram {
                        Some(n) if n.layer == idx => Some(RawNgram {
                            key: req!(&format!("{prefix}.ple_key.weight"))?,
                            value: req!(&format!("{prefix}.ple_value.weight"))?,
                            norm_key: req_cpu!(&format!("{prefix}.ple_norm_key.weight"))?,
                            norm_query: req_cpu!(&format!("{prefix}.ple_norm_query.weight"))?,
                            norm_conv: req_cpu!(&format!("{prefix}.ple_norm_conv.weight"))?,
                            conv: req_cpu!(&format!("{prefix}.ple_conv1d.weight"))?,
                        }),
                        _ => None,
                    };
                    RawWide {
                        attn_mixer: Some(mixer_weights("hc_attn", true)?),
                        ffn_mixer: Some(mixer_weights("hc_ffn", true)?),
                        indexer,
                        ngram,
                    }
                }
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
                attn_sinks_qt,
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
                moe_gate_inp_bias_qt,
                moe_gate_exps_bias_qt,
                moe_up_exps_bias_qt,
                moe_down_exps_bias_qt,
                pre_ffw_norm_2_qt,
                post_ffw_norm_1_qt,
                post_ffw_norm_2_qt,
                moe_exp_probs_b_qt,
                mixer,
                has_ffn,
                attn_gate,
                shexp_gate: opt_dev!(&format!("{prefix}.ffn_gate_shexp.weight"))?,
                shexp_up: opt_dev!(&format!("{prefix}.ffn_up_shexp.weight"))?,
                shexp_down: opt_dev!(&format!("{prefix}.ffn_down_shexp.weight"))?,
                shexp_scalar_gate_qt: opt_cpu!(&format!("{prefix}.ffn_gate_inp_shexp.weight"))?,
                wide,
            })
        };

        // Every layer goes to the device the plan gives it. A plan that spans more than one
        // card loads sequentially, one context at a time; a plan on one card, or on the host,
        // loads in parallel over one context.
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
        let raw_layers: Vec<RawGenericLayer> = if distinct_gpus.len() > 1 {
            layer_devices
                .into_iter()
                .map(|(idx, dk)| load_one(idx, dk))
                .collect::<Result<Vec<_>>>()?
        } else {
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
        // Tightest room left on a card this model builds on once the raw weights are
        // resident; cards the plan does not use are not this model's constraint.
        let post_phase1_free = distinct_gpus
            .iter()
            .filter_map(|idx| cuda_devices.get(idx))
            .filter_map(crate::inference::place::room::free_now)
            .min()
            .unwrap_or(u64::MAX);
        crate::inference::place::device_probe::debug_vram_by_card("after Phase1 raw-weight load");

        // -- 6. Phase 2: Build per-device layers ----------------------------
        // build_generic_layer's per-layer dequant->Q4K requant is independent and
        // touches only its layer's tensors on that layer's GPU context. When all
        // layers share ONE GPU context (single-target, the common case) this is the
        // same safety profile as the parallel Phase 1 -> build them in parallel (this
        // was the dominant load cost). For a genuine multi-GPU split (mixed contexts)
        // a rayon worker could stripe across both devices, so build sequentially.
        let layer_ropes = LayerRopes::default();
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
                &layer_ropes,
                rope_freq_factors.as_ref(),
            )?;
            // The layer's bytes are on the card now; the file pages that carried them are
            // dead weight, and on a spilled model they are what pushes the host layers'
            // working set into swap before decode even starts.
            if matches!(dev_enum, LayerDevice::Cuda(_)) {
                source.release_layer_pages(layer_idx);
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
        crate::inference::place::device_probe::debug_vram_by_card("after Phase2 layer-build");

        let sliding_window = config.sliding_window;

        // The output projection sits on the last card a layer runs on, in its stored format:
        // the plan counted it there. A card that runs out of memory fails the load, so the
        // ladder places the model again; any other failure leaves the head on the host.
        let last_cuda_idx = layer_devs.iter().rev().find_map(|d| {
            if let LayerDevice::Cuda(idx) = d {
                Some(*idx)
            } else {
                None
            }
        });
        let mut output_proj_cuda: Option<Tensor> = None;
        let mut output_norm_cuda_weight: Option<Tensor> = None;
        let mut output_proj_cuda_qmm: Option<crate::tensor::quantized::QMatMul> = None;
        let mut output_proj_dev: Option<Device> = None;
        if let Some((idx, cuda_dev)) =
            last_cuda_idx.and_then(|idx| cuda_devices.get(&idx).map(|d| (idx, d)))
        {
            let t_q = std::time::Instant::now();
            // A quantised head runs as a quantised matmul; a float head (F32, F16, BF16) is
            // held once, in F16, for the dense matmul the card paths take without one. The
            // float head is widened on the host, so the card receives only the F16 copy.
            let placed = norm_weight_cpu.to_device(cuda_dev).and_then(|norm_w| {
                let dense = content
                    .tensor_infos
                    .get(output_weight_name)
                    .is_some_and(|info| {
                        matches!(
                            info.ggml_dtype,
                            crate::tensor::quantized::GgmlDType::F32
                                | crate::tensor::quantized::GgmlDType::F16
                                | crate::tensor::quantized::GgmlDType::BF16
                        )
                    });
                let head = if dense {
                    let qt = source.read(output_weight_name, &Device::Cpu)?;
                    // CONVERSION: a float head is held once, in F16, for the dense matmul the card paths take.
                    (None, Some(qt.dequantize_f16(cuda_dev)?))
                } else {
                    let qt = source.read(output_weight_name, cuda_dev)?;
                    (
                        Some(crate::tensor::quantized::QMatMul::from_qtensor(qt)?),
                        None,
                    )
                };
                Ok((norm_w, head))
            });
            match placed {
                Ok((norm_w, (qmm, dense))) => {
                    info!(
                        "  Output projection on CUDA:{idx} ({:.0}ms)",
                        t_q.elapsed().as_secs_f64() * 1000.0
                    );
                    output_norm_cuda_weight = Some(norm_w);
                    output_proj_cuda_qmm = qmm;
                    output_proj_cuda = dense;
                    output_proj_dev = Some(cuda_dev.clone());
                }
                Err(e) if crate::inference::place::ladder::is_out_of_memory(&e) => return Err(e),
                Err(e) => warn!("  Output projection on CUDA:{idx} failed ({e}); using the host"),
            }
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

        // Keep a handle only for the cards this model landed on. A caller may hand in
        // handles for cards the plan does not use; they buy nothing and are not free - a
        // handle holds the card's CUDA context alive, with its allocator workspace, and
        // keeps the card out of its idle power state.
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
            let emb_dev = embeddings.device();
            for d in [output_proj_dev.as_ref(), Some(&emb_dev)]
                .into_iter()
                .flatten()
            {
                #[cfg(feature = "cuda")]
                if let Device::Cuda(cd) = d {
                    keep.insert(cd.ordinal());
                }
                #[cfg(not(feature = "cuda"))]
                let _ = d;
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
            #[cfg(not(feature = "cuda"))]
            let _ = (idx, d);
            #[cfg(feature = "cuda")]
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

        // The vision tower of a family whose images splice into the text stream, on the card
        // of the first layer; the tower runs on a card only.
        let spliced_vision = match (
            splices_vision_tower(content, &config.arch),
            content.tensor_infos.contains_key("v.patch_embed.weight"),
            layer_devs.first(),
        ) {
            (true, true, Some(LayerDevice::Cuda(idx))) => {
                let token = meta_u32(content, &config.arch, "vision.image_token_id")
                    .or_else(|| meta_u32(content, &config.arch, "image_token_id"));
                match (token, cuda_devices.get(idx)) {
                    (Some(image_token), Some(card)) => {
                        // The tower reads through a cursor over one mapping, so it takes the
                        // first part's header and bytes. A set whose `v.*` tensors live in
                        // another part fails here and is served as text only.
                        let (tower_content, tower_bytes) = source.first();
                        let mut cursor = std::io::Cursor::new(tower_bytes);
                        let tower =
                            crate::inference::model::qwen35::vision::Qwen35Vision::from_gguf(
                                tower_content,
                                &mut cursor,
                                config.embedding_length,
                                card,
                            );
                        match tower {
                            Ok(tower) => Some(super::splice::SplicedVision { tower, image_token }),
                            Err(e) => {
                                warn!("vision tower not loaded ({e}); text only");
                                None
                            }
                        }
                    }
                    _ => {
                        warn!(
                            "the file names no image token or card for its vision tower; text only"
                        );
                        None
                    }
                }
            }
            _ => None,
        };

        // A wide-residual model has no output norm: its final mixer does that work, on the
        // device the head runs on.
        let output_hyper = match config.hyper {
            Some(h) => {
                let dev = output_proj_dev.clone().unwrap_or(Device::Cpu);
                // CONVERSION: the output mixer's norm scale is a small vector held dense on the head's device.
                let gamma = source
                    .read("output_hc_norm.weight", &Device::Cpu)?
                    .dequantize(&Device::Cpu)?
                    .to_device(&dev)?;
                let down = source.read("output_hc_down.weight", &dev)?;
                let up = source.read("output_hc_up.weight", &dev)?;
                Some(super::hyper::HyperMixer::new(
                    gamma,
                    super::wide::load_matrix(down)?,
                    super::wide::load_matrix(up)?,
                    None,
                    h.streams,
                    config.embedding_length,
                    config.rms_norm_eps,
                )?)
            }
            None => None,
        };
        let ngram_model = match &config.ngram {
            Some(meta) if config.hyper.is_some() => {
                let table = source.read("per_layer_token_embd.weight", &Device::Cpu)?;
                Some(NgramModel {
                    table: super::ngram::NgramTable::new(table, meta)?,
                    meta: meta.clone(),
                    pending: None,
                })
            }
            _ => None,
        };
        let mut model = Self {
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
            ple_model_proj,
            ple_proj_norm,
            ple_dim,
            graph_hidden_buffer: None,
            #[cfg(feature = "cuda")]
            graph_hops: Vec::new(),
            graph_logits_buffer,
            graph_alive_tensors_model: Vec::new(),
            graph_state_groups: None,
            workspace_pre_grown: false,
            kv_snapshots: Vec::new(),
            kv_snapshot_tick: 0,
            hybrid_cache: None,
            spliced_vision,
            #[cfg(feature = "audio")]
            spliced_audio: None,
            host_experts_hot: None,
            output_hyper,
            ngram: ngram_model,
        };

        // A SiLU MoE layer on the host runs its experts through a repacked layout derived
        // from the stacks rather than stored with them. It is built on first use, which is
        // inside the first request - for a 16-layer, 64-expert model, seconds of it. Build it
        // here instead, on a thread of its own: the load does not wait for it, so a model
        // answers exactly as early as before, and a request that arrives while it is still
        // running waits on the stack being built and is handed that result rather than
        // repacking the same stack beside it. A layer on a card never reads that layout.
        // A layer on a card whose routed experts the plan left in host memory reads them in
        // place: the host holds them because it has no room for a repacked second copy.
        {
            #[cfg(feature = "cuda")]
            use crate::inference::moe_cpu as host_moe;
            #[cfg(not(feature = "cuda"))]
            use crate::inference::moe_cuda as host_moe;
            for (layer, dev) in model.layers.iter().zip(model.layer_devs.iter()) {
                let Some(moe) = layer.moe.as_ref().filter(|m| m.experts_on_host()) else {
                    continue;
                };
                if matches!(dev, LayerDevice::Cuda(_)) {
                    for stack in moe.host_prefill_stacks() {
                        host_moe::read_in_place(&stack);
                    }
                }
            }
        }
        model.host_experts_hot = model
            .layers
            .iter()
            .any(|l| l.moe.as_ref().is_some_and(|m| m.experts_on_host()))
            .then(crate::tensor::quant_cpu::gemv_pool::hold_workers_hot);
        let host_prefill_stacks: Vec<_> = model
            .layers
            .iter()
            .zip(model.layer_devs.iter())
            .filter(|(_, dev)| matches!(dev, LayerDevice::Cpu))
            .filter_map(|(layer, _)| layer.moe.as_ref())
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
                        scope.spawn(|| {
                            while let Some(stack) =
                                host_prefill_stacks.get(next.fetch_add(1, Ordering::Relaxed))
                            {
                                if let Err(e) = host_moe::prewarm_expert_repack(stack) {
                                    // The first prefill builds whatever is missing, so a failure
                                    // here costs time and never an answer - said once for the
                                    // warm, not once per worker and not once per stack.
                                    if !failed.swap(true, Ordering::Relaxed) {
                                        tracing::warn!("expert repack warm stopped: {e}");
                                    }
                                    break;
                                }
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
        Ok(model)
    }

    /// Build a `GenericHeteroTransformer` from an HF AutoAWQ checkpoint.
    /// Dense Qwen2/Qwen3/Mistral only, each layer on the device `plan` gives it. The
    /// projection `QMatMul`s
    /// become `QMatMul::Awq` (the 483 GB/s GEMV at decode, dequant->cuBLAS at
    /// prefill); everything else (attention, RoPE, KV, forward) is the generic
    /// path unchanged.
    pub fn from_awq_safetensors(
        dir: &str,
        shards: &[String],
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        kv_quant: crate::inference::engine::llm_engine::KvQuant,
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
        config.kv_quant = kv_quant;
        if let Some(m) = max_kv_seq_len {
            config.max_q8_seq_len = m;
        }
        // The checkpoint's Q and K rows are in their original order, which pairs each rotary
        // lane with the one half a head away. Adjacent pairing belongs to GGUF files, whose
        // converter reorders the rows of the architectures that use it.
        config.flags.use_rope_i = false;
        let config = Arc::new(config);
        let group_size = cfgj.group_size;

        // Layer sizes as stored: what each layer puts on its device.
        let mut layer_bytes = vec![0u64; config.n_layers];
        for (name, bytes) in st.byte_sizes() {
            if let Some(li) = name
                .strip_prefix("model.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
            {
                if let Some(b) = layer_bytes.get_mut(li) {
                    *b += bytes as u64;
                }
            }
        }
        let head_name = if st.get("lm_head.weight").is_ok() {
            "lm_head.weight"
        } else {
            "model.embed_tokens.weight"
        };
        // Each layer on the device its plan segment names.
        let layer_card: Vec<Option<usize>> = plan
            .segments
            .iter()
            .flat_map(|seg| {
                let card = match seg.kind {
                    DeviceKind::Cuda(idx) if cuda_devices.contains_key(&idx) => Some(idx),
                    _ => None,
                };
                (seg.layer_start..seg.layer_end).map(move |_| card)
            })
            .collect();
        if layer_card.len() != config.n_layers {
            return Err(crate::tensor::Error::msg(format!(
                "the plan covers {} layers and the checkpoint has {}",
                layer_card.len(),
                config.n_layers
            )));
        }
        let device_of = |card: Option<usize>| match card {
            Some(idx) => (cuda_devices[&idx].clone(), LayerDevice::Cuda(idx)),
            None => (Device::Cpu, LayerDevice::Cpu),
        };
        // The head goes with the last layer.
        let (device, _) = device_of(layer_card.last().copied().flatten());
        let mut used: Vec<usize> = layer_card.iter().flatten().copied().collect();
        used.dedup();
        info!(
            "AWQ load: {} arch={} layers={} (gs={})",
            cfgj.hf_arch, arch, config.n_layers, group_size
        );

        // Prewarm the fused decode kernels onto a CLEAN device - BEFORE loading
        // the ~336 AWQ weight tensors. Lazily loading the "loken_fused" module
        // after the AWQ weights are resident fails with CUDA "named symbol not
        // found" (a facade module-cache quirk after many custom-module loads);
        // prewarming first sidesteps it.
        #[cfg(feature = "cuda")]
        for idx in &used {
            if let Ok(cd) = cuda_devices[idx].as_cuda_device() {
                let _ = crate::inference::kernel::fused::prewarm_fused_kernels(&cd);
            }
        }

        // RoPE tables (full head_dim, YaRN if configured), sized to the KV window, one per
        // device a layer runs on.
        let mut rope: HashMap<Option<usize>, (Tensor, Tensor)> = HashMap::new();
        for &card in &layer_card {
            if let std::collections::hash_map::Entry::Vacant(slot) = rope.entry(card) {
                let (dev, _) = device_of(card);
                slot.insert(
                    crate::inference::generic_transformer::rope::precomput_freqs_cis_yarn(
                        config.head_dim,
                        config.rope_freq_base,
                        config.yarn,
                        config
                            .effective_max_context()
                            .max(crate::inference::generic_transformer::rope::ROPE_TABLE_SEQ_LEN),
                        &dev,
                    )?,
                );
            }
        }

        // LM head: lm_head.weight, or embed_tokens if tied. F16 on the last layer's card for
        // the cutlass path; Q8_0 on CPU as the required fallback `output_proj`. Placed before
        // the layers, so the room each layer's second Marlin layout keeps free need not
        // count it.
        let lm_cpu_f32 = st.load(head_name, &Device::Cpu)?.to_dtype(DType::F32)?;
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

        let flags = Arc::new(config.flags.clone());
        // What each layer's cache takes at the size it starts at, K and V in f16, and what a
        // request's work keeps free on a card.
        let kv_per_layer = (2 * config.n_kv_head * config.head_dim * 2) as u64
            * config
                .effective_max_context()
                .min(crate::inference::cache::KV_WORKING_WINDOW_TOKENS) as u64;
        let work = crate::inference::place::runtime_demand::llm_work_reserve(
            config.embedding_length.max(config.ffn_dim),
            crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS,
        );
        let mut layers = Vec::with_capacity(config.n_layers);
        let mut layer_devs = Vec::with_capacity(config.n_layers);
        for li in 0..config.n_layers {
            let card = layer_card[li];
            let (dev, dev_enum) = device_of(card);
            let (cos, sin) = &rope[&card];
            // What the card must keep beside this layer's second Marlin layout: the layers
            // after it, the caches of all its layers and the work.
            let still_to_load: u64 = (li + 1..config.n_layers)
                .filter(|&l| layer_card[l] == card)
                .map(|l| layer_bytes[l])
                .sum();
            let caches = kv_per_layer * layer_card.iter().filter(|&&c| c == card).count() as u64;
            let keep_free = still_to_load + caches + work;
            layers.push(build_awq_layer(
                &st,
                li,
                &dev,
                &config,
                flags.clone(),
                cos,
                sin,
                group_size,
                keep_free as usize,
            )?);
            layer_devs.push(dev_enum);
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
            ple_model_proj: None,
            ple_proj_norm: None,
            ple_dim: 0,
            graph_hidden_buffer: None,
            #[cfg(feature = "cuda")]
            graph_hops: Vec::new(),
            graph_logits_buffer: None,
            graph_alive_tensors_model: Vec::new(),
            graph_state_groups: None,
            workspace_pre_grown: false,
            kv_snapshots: Vec::new(),
            kv_snapshot_tick: 0,
            hybrid_cache: None,
            spliced_vision: None,
            #[cfg(feature = "audio")]
            spliced_audio: None,
            host_experts_hot: None,
            output_hyper: None,
            ngram: None,
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

#[cfg(test)]
mod split_tests {
    use super::*;

    #[test]
    fn a_q_projection_splits_into_q_and_gate_per_head() {
        let (heads, hd, hidden) = (3usize, 2usize, 4usize);
        let rows = 2 * heads * hd;
        // Row r holds the value r in every column.
        let data: Vec<f32> = (0..rows).flat_map(|r| vec![r as f32; hidden]).collect();
        let w = Tensor::from_vec(data, (rows, hidden), &Device::Cpu).unwrap();
        let qt = QTensor::quantize_onto(&w, crate::tensor::quantized::GgmlDType::F32, &Device::Cpu)
            .unwrap();
        let (q, gate) = split_q_gate(qt, heads, &Device::Cpu).unwrap();
        let first_col = |t: QTensor| -> Vec<f32> {
            let d = t.dequantize(&Device::Cpu).unwrap();
            assert_eq!(d.dims(), [heads * hd, hidden]);
            d.to_vec2::<f32>().unwrap().iter().map(|r| r[0]).collect()
        };
        assert_eq!(first_col(q), vec![0., 1., 4., 5., 8., 9.]);
        assert_eq!(first_col(gate), vec![2., 3., 6., 7., 10., 11.]);
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::Value;
    use crate::tensor::quantized::gguf_source::SplitGguf;
    use crate::tensor::quantized::GgmlDType;

    const LAYERS: usize = 2;
    const EMBD: usize = 8;
    const FFN: usize = 16;
    const VOCAB: usize = 11;

    /// A tiny dense model: `[out, in]` weights of a deterministic pattern, F32.
    fn entries() -> Vec<GgufEntry> {
        let mut seed = 7u32;
        let mut entry = |name: String, dims: Vec<usize>| {
            let n: usize = dims.iter().product();
            let data = (0..n)
                .flat_map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((seed >> 16) as f32 / 65536.0 - 0.5) * 0.4).to_le_bytes()
                })
                .collect();
            GgufEntry {
                name,
                dims,
                dtype: GgmlDType::F32,
                data,
            }
        };
        let mut out = vec![
            entry("token_embd.weight".into(), vec![VOCAB, EMBD]),
            entry("output.weight".into(), vec![VOCAB, EMBD]),
        ];
        let norm = |name: String| GgufEntry {
            name,
            dims: vec![EMBD],
            dtype: GgmlDType::F32,
            data: [1.0f32.to_le_bytes(); EMBD].concat(),
        };
        out.push(norm("output_norm.weight".into()));
        for il in 0..LAYERS {
            out.push(norm(format!("blk.{il}.attn_norm.weight")));
            out.push(norm(format!("blk.{il}.ffn_norm.weight")));
            for (n, d) in [
                ("attn_q", [EMBD, EMBD]),
                ("attn_k", [EMBD, EMBD]),
                ("attn_v", [EMBD, EMBD]),
                ("attn_output", [EMBD, EMBD]),
                ("ffn_gate", [FFN, EMBD]),
                ("ffn_up", [FFN, EMBD]),
                ("ffn_down", [EMBD, FFN]),
            ] {
                out.push(entry(format!("blk.{il}.{n}.weight"), d.to_vec()));
            }
        }
        out
    }

    fn metadata() -> Vec<(String, Value)> {
        let u = |k: &str, v: u32| (format!("llama.{k}"), Value::U32(v));
        vec![
            ("general.architecture".into(), Value::String("llama".into())),
            u("block_count", LAYERS as u32),
            u("embedding_length", EMBD as u32),
            u("feed_forward_length", FFN as u32),
            u("attention.head_count", 2),
            u("attention.head_count_kv", 2),
            u("context_length", 64),
            (
                "llama.attention.layer_norm_rms_epsilon".into(),
                Value::F32(1e-5),
            ),
            ("llama.rope.freq_base".into(), Value::F32(10000.0)),
        ]
    }

    fn logits(mut model: GenericHeteroTransformer) -> Vec<f32> {
        let ids = Tensor::from_vec(vec![1u32, 4, 2, 9], (1usize, 4usize), &Device::Cpu).unwrap();
        model
            .forward(&ids, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// Embeddings that are the model's own rows for some tokens, spliced over a placeholder,
    /// give the logits of those tokens read as text, and the next decode position is the
    /// length of the sequence they make.
    #[test]
    fn spliced_embeddings_stand_for_the_tokens_they_replace() {
        let dir = std::env::temp_dir().join(format!("loken-splice-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.gguf");
        write_gguf_with_metadata(&path, &metadata(), &entries()).unwrap();
        let plan = HeteroPlan::calculate_with_kv(LAYERS, 1 << 20, &[], &[], 1.0, 0);
        let none = HashMap::new();
        let load = || {
            let g = crate::tensor::quantized::gguf_file::open_mapped(&path).unwrap();
            let bytes = g.mmap.clone();
            GenericHeteroTransformer::from_gguf(g.content, &bytes[..], &none, &plan).unwrap()
        };
        let want = logits(load());

        let mut model = load();
        let rows = model
            .embeddings
            .forward(&Tensor::from_vec(vec![4u32, 2], (1usize, 2usize), &Device::Cpu).unwrap())
            .unwrap();
        // 1, [placeholder], 9 with the placeholder standing for 4, 2.
        let prompt = [1u32, 0, 9];
        let (got, next) = model.forward_spliced_audio(&prompt, 1, &rows).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            next, 4,
            "the sequence is the prompt less the placeholder plus the rows"
        );
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-5, "spliced logits differ by {worst}");
    }

    /// A model written as two parts, blocks divided between them, loads to the logits of the
    /// same model in one file: which part a tensor sits in does not matter. A part opened alone
    /// is refused.
    #[test]
    fn a_split_set_loads_like_one_file() {
        let dir = std::env::temp_dir().join(format!("loken-source-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let plan = HeteroPlan::calculate_with_kv(LAYERS, 1 << 20, &[], &[], 1.0, 0);
        let none = HashMap::new();

        let single = dir.join("one.gguf");
        write_gguf_with_metadata(&single, &metadata(), &entries()).unwrap();
        let g = crate::tensor::quantized::gguf_file::open_mapped(&single).unwrap();
        let bytes = g.mmap.clone();
        let want = logits(
            GenericHeteroTransformer::from_gguf(g.content, &bytes[..], &none, &plan).unwrap(),
        );

        let (first, second): (Vec<_>, Vec<_>) = entries()
            .into_iter()
            .partition(|e| !e.name.starts_with("blk.1."));
        let with = |no: u32| {
            let mut m = metadata();
            m.push(("split.count".to_string(), Value::U32(2)));
            m.push(("split.no".to_string(), Value::U32(no)));
            m
        };
        let (p1, p2) = (
            dir.join("set-00001-of-00002.gguf"),
            dir.join("set-00002-of-00002.gguf"),
        );
        write_gguf_with_metadata(&p1, &with(0), &first).unwrap();
        write_gguf_with_metadata(&p2, &with(1), &second).unwrap();

        let part = crate::tensor::quantized::gguf_file::open_mapped(&p1).unwrap();
        let part_bytes = part.mmap.clone();
        let refused =
            GenericHeteroTransformer::from_gguf(part.content, &part_bytes[..], &none, &plan);
        assert!(
            refused
                .err()
                .is_some_and(|e| e.to_string().contains("part 1 of a 2-part")),
            "a part alone must not build an incomplete model"
        );

        let set = SplitGguf::open(&p1).unwrap();
        let got = logits(
            GenericHeteroTransformer::from_split_gguf(
                &set,
                &none,
                &plan,
                crate::inference::engine::llm_engine::KvQuant::Off,
                None,
                false,
            )
            .unwrap(),
        );
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got, want, "the split set changed the logits");
    }
}
