//! Generic Quantized Transformer - Multi-Device (CUDA + CPU).
//!
//! A single implementation covering Qwen2, Qwen3, Gemma3, Phi3, and any other
//! GGUF-format transformer decoder that uses the standard `blk.{i}.*` tensor naming.
//!
//! Architectural variants detected automatically from GGUF content:
//! - **Qwen2**: Split Q/K/V + attention biases (`attn_q.bias` etc.)
//! - **Qwen3**: Split Q/K/V, standard SwiGLU
//! - **Gemma3**: Split Q/K/V + QK norms + post-attention/FFN norms
//! - **Phi3**: Fused `attn_qkv.weight` + fused gate/up in `ffn_up.weight`
//!
//! Layer distribution across CUDA + CPU is driven by `HeteroPlan`.
//! OpenCL (Arc) is out of scope here - handled by the Mistral3-specific path.

// The graph-mode init-then-update idiom (see padded_mask, graph_q_buffer,
// graph_logits_buffer, graph_hidden_buffer, q4_kv_cache, output_proj_*) checks
// `Option::is_some()` / `is_none()` and then `unwrap()`s in the matching
// branch. clippy::unnecessary_unwrap can't reason across the if-condition;
// each site is logically safe. Refactoring the ~20 sites to `match` /
// `if let Some(_)` would touch attention/projection paths in the hot
// decode loop with no behaviour change. Allow the lint for the whole file.
#![allow(clippy::unnecessary_unwrap)]
/// The quantised projection every layer multiplies through.
pub mod projection;

/// Rotary position embeddings and the table a decode reads them from.
/// Rotary tables are not this transformer's own: the image families want them too, so they
/// live under `model` and are named here for the paths that already say `rope::`.
pub use crate::inference::model::rope;

pub(crate) use std::collections::HashMap;
pub(crate) use std::sync::Arc;

pub(crate) use crate::tensor::layer::RmsNorm;
pub(crate) use crate::tensor::layer::{Embedding, LayerNorm};
pub(crate) use crate::tensor::quantized::{gguf_file, QTensor};
pub(crate) use crate::tensor::{DType, Device, IndexOp, Result, Tensor};

mod config;
mod graph;
pub mod hetero;
mod layer_attn;
mod layer_cpu_attn;
mod layer_ffn;
mod layer_forward;
mod loader;
mod moe_weights;
pub use config::*;
pub use moe_weights::*;

pub(crate) use rayon::prelude::*;
pub(crate) use tracing::{info, warn};

pub(crate) use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
pub(crate) use crate::inference::serve::spec_kv_cache::SpecKvCache;
pub(crate) use projection::QMatMul;

/// One transformer layer with all optional architectural variants.
pub struct GenericTransformerLayer {
    // Attention - split Q/K/V (most models)
    pub attn_q: Option<QMatMul>,
    pub attn_k: Option<QMatMul>,
    pub attn_v: Option<QMatMul>,
    // Attention - fused QKV (Phi3)
    pub attn_qkv: Option<QMatMul>,
    // Optional biases (Qwen2): dequantized F32 tensors
    pub attn_q_bias: Option<Tensor>,
    pub attn_k_bias: Option<Tensor>,
    pub attn_v_bias: Option<Tensor>,
    /// Optional fused QKV bias: concatenation of attn_q_bias || attn_k_bias ||
    /// attn_v_bias, populated at load-time when all three exist and the
    /// fused-QKV matmul path is active. Lets the runtime do a single
    /// broadcast_add on the [q_dim+2*kv_dim] qkv tensor instead of three
    /// per-tensor broadcast_adds. Saves 2 launches/layer/token on qwen2-style
    /// models. Always tracks the same dtype/device as the per-component biases.
    pub attn_qkv_bias: Option<Tensor>,
    // Optional QK norms (Gemma3)
    pub attn_q_norm: Option<RmsNorm>,
    pub attn_k_norm: Option<RmsNorm>,
    /// Gemma4: V-RMS-norm uses a unit weight tensor (head_dim ones, F32).
    /// Cached so the per-call `rms_norm(v, ones, eps)` fires the tensor-op
    /// fused single-launch op instead of the manual sqr+mean+sqrt+div
    /// (~5 launches/layer/token).
    pub attn_v_norm_ones: Option<Tensor>,
    // Always present
    pub attn_output: QMatMul,
    /// Optional bias on attn_output projection (phi2)
    pub attn_output_bias: Option<Tensor>,
    pub attn_norm: WeightedNorm,
    /// Phase B: F32-cast of attn_norm.weight, pre-built
    /// at load time when fused_qkv_norm_eligible. Avoids the per-
    /// call BF16->F32 cast launch that would otherwise offset the
    /// launch saved by the fused norm+QKV path. None when
    /// fused_qkv_norm is not enabled for this layer.
    pub attn_norm_weight_f32: Option<Tensor>,
    pub post_attn_norm: Option<RmsNorm>, // Gemma3
    pub ffn_gate: Option<QMatMul>,       // None when fused with ffn_up (Phi3)
    pub ffn_up: QMatMul,
    pub ffn_up_bias: Option<Tensor>, // Phi2: ffn_up has .bias
    pub ffn_down: QMatMul,
    /// Gemma4-MoE only: every layer has BOTH the dense FFN above AND
    /// an MoE block. When `Some`, run shared dense + MoE and add.
    pub moe: Option<MoeWeights>,
    /// Pre-norm before the MoE block (Gemma4 26B has it as a separate
    /// `pre_ffw_norm_2` tensor on top of the standard `ffn_norm`).
    pub pre_ffw_norm_2: Option<RmsNorm>,
    /// Post-norm after the SHARED dense FFN before adding MoE output.
    pub post_ffw_norm_1: Option<RmsNorm>,
    /// Post-norm after the MoE FFN before adding to dense output.
    pub post_ffw_norm_2: Option<RmsNorm>,
    /// Cached embedding length (config.embedding_length) - used by the
    /// gemma4-MoE router which scales `rms_norm(attn_out) * 1/sqrt(n_embd)`
    /// before the gate matmul.
    pub embedding_length_for_moe: usize,
    pub ffn_down_bias: Option<Tensor>, // Phi2: ffn_down has .bias
    /// `Some` for serial-attention models (llama, qwen, gemma...).
    /// `None` for parallel-attention models (phi2/phi/gpt-neox-style),
    /// where the same input layer norm feeds both attn and ffn branches.
    pub ffn_norm: Option<WeightedNorm>,
    /// Raw FFN norm weight tensor (for fused add+rmsnorm kernel)
    pub ffn_norm_weight: Option<Tensor>,
    pub ffn_norm_eps: f32,
    pub post_ffn_norm: Option<RmsNorm>, // Gemma3
    // PLE - Per-Layer Embeddings (Gemma4)
    pub ple_inp_gate: Option<QMatMul>, // hidden_dim -> ple_dim
    pub ple_proj: Option<QMatMul>,     // ple_dim -> hidden_dim
    pub ple_post_norm: Option<RmsNorm>,
    pub ple_output_scale: Option<Tensor>, // [hidden_dim] per-element scale (altup_correct_scale)
    // KV cache + RoPE
    pub kv_cache: SpecKvCache,
    /// Optional Q8_0-quantized KV cache, populated on every append alongside
    /// the F-dtype cache when `config.kv_quant == Q8`. Used by the decode
    /// path (seq=1) to run attention against Q8 K/V and halve KV VRAM.
    /// Prefill still uses the F16 `kv_cache`.
    #[cfg(feature = "cuda")]
    pub q8_kv_cache: Option<crate::inference::cache::q8_kv::Q8KvCache>,
    /// CPU-only Q8 KV cache: halves KV streaming bytes at dense decode.
    /// Populated alongside the F16 `kv_cache` (which stays the safe fallback);
    /// read only at single-token CPU decode for eligible dense arches.
    pub cpu_q8_kv: Option<crate::inference::cache::cpu_q8_kv::CpuQ8Kv>,
    /// gemma4 F16 windowed KV (Q8 degenerate at hd512). Per-layer window.
    pub cpu_f16_kv: Option<crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
    /// Persistent zero-alloc decode scratch (lazily built, reused/token).
    pub cpu_decode_arena: Option<crate::inference::kernel::cpu_decode_exec::DecodeArena>,
    /// Per-layer constant norm weights/biases as f32 (lazily built once,
    /// reused every token - removes ~5 `to_vec1` allocs/layer/token).
    pub cpu_norm_cache: Option<crate::inference::kernel::cpu_decode_exec::DecodeNormCache>,
    /// Optional Q4_0-quantized KV cache (KIVI-style, per-channel K)  - 
    /// populated when `config.kv_quant == Q4` and device is CUDA. Halves
    /// KV vs Q8 / quarters vs F16. Decode path uses
    /// `attn_scores`+`attn_output` directly; prefill / multi-token paths
    /// dequantize on the fly.
    #[cfg(feature = "cuda")]
    pub q4_kv_cache: Option<crate::inference::cache::q4_kv::Q4KvCache>,
    /// True for layers in the donor range of a shared_kv model (gemma4).
    /// When set, the layer ALSO populates F-dtype `kv_cache` alongside
    /// the Q8 cache, so shared layers can read donor's F-dtype K/V via
    /// the standard attention path. Without this, consolidated-Q8 mode
    /// would leave F-dtype empty and shared layers would fail.
    pub populate_dual_kv: bool,
    /// Store the F-dtype KV cache in F16 rather than the F32 working dtype.
    /// Halves the dominant K+V decode read (the long-context lever) at
    /// ollama-parity precision - F16's 10-bit mantissa is far above the Q8/Q4
    /// noise floor that degenerates gemma4's tiny K-norm, so it stays coherent
    /// where quantized KV does not. Set for gemma4 at `kv_quant==Off`; other
    /// arches keep F32 (their winning cells regress under nothing). The decode/
    /// prefill append casts k/v to F16 and `standard_attention` runs the score/
    /// output matmuls at the cache dtype (dtype-agnostic; no-op when F32).
    pub kv_f16: bool,
    /// Pre-allocated padded attention mask for CUDA graph mode.
    pub padded_mask: Option<Tensor>,
    /// Pre-allocated Q tensor for CUDA graph mode (fixed device pointer).
    pub graph_q_buffer: Option<Tensor>,
    /// Pre-allocated RoPE cos/sin buffers for graph mode (fixed pointers).
    /// Updated via slice_set before each graph replay.
    pub graph_rope_cos: Option<Tensor>,
    pub graph_rope_sin: Option<Tensor>,
    /// 1-element i64 GPU tensor holding the current KV write position.
    /// Updated outside the graph; scatter_set reads it at replay time.
    pub graph_kv_pos: Option<Tensor>,
    /// Persistent attention-score buffer for the GQA fast path under
    /// graph mode. Shape `[1, n_kv_head, n_rep, max_kv]` F32. Hypothesis:
    /// the fresh per-replay alloc that the reference matmul returns is the
    /// HD=512 graph crash trigger; routing through this stable buffer
    /// via slice_set should remove the dependency on the graph memory
    /// pool's per-replay remapping.
    pub graph_attn_qk_buffer: Option<Tensor>,
    /// Persistent attention-output buffer for the GQA fast path.
    /// Shape `[1, n_kv_head, n_rep, head_dim]` F32.
    pub graph_attn_out_buffer: Option<Tensor>,
    /// Persistent attn_output projection result.
    /// Shape `[1, 1, hidden_size]` F32. Routes `self.attn_output.forward(&y)`
    /// through a stable device pointer for graph capture.
    pub graph_attn_proj_buffer: Option<Tensor>,
    /// Persistent ffn_up output (phi2 simple FFN path).
    /// Shape `[1, 1, intermediate_size]` F32.
    pub graph_ffn_up_buffer: Option<Tensor>,
    /// Persistent fused_bias_gelu_new output (phi2 simple FFN path).
    /// Shape `[1, 1, intermediate_size]` F32.
    pub graph_ffn_activated_buffer: Option<Tensor>,
    /// Persistent ffn_down output (phi2 simple FFN path, pre-bias).
    /// Shape `[1, 1, hidden_size]` F32.
    pub graph_ffn_down_buffer: Option<Tensor>,
    /// Persistent fused_phi2_residual_merge final output.
    /// Shape `[1, 1, hidden_size]` F32.
    pub graph_phi2_merge_buffer: Option<Tensor>,
    /// Persistent ffn_up concat output for gemma4-style FFN with
    /// load-time-fused gate||up. Shape `[1, 1, 2*intermediate_size]` F32.
    /// routes the ffn_up matmul output through a stable
    /// device pointer for graph capture. The fresh-alloc from
    /// `self.ffn_up.forward(x)` has CAPTURE-time addr baked into the
    /// captured graph and goes stale on replay (same root cause as
    /// phi2's ). Sized to 2N because the concat weight has shape
    /// [2*intermediate, hidden].
    pub graph_ffn_up_concat_buffer: Option<Tensor>,
    /// Persistent post_ffn_norm + add output.
    /// Shape `[1, 1, hidden_size]` F32. Routes the
    /// `fused_rmsnorm_then_add` output at the gemma4 post_ffn_norm
    /// site (forward_inner line ~3651) through a stable device pointer.
    /// Without this, the per-layer fresh-alloc inside that kernel
    /// produces a wild pointer in the captured graph on replay.
    pub graph_post_ffn_norm_buffer: Option<Tensor>,
    /// Persistent PLE gate output. Shape `[1, 1, ple_dim]`
    /// F32. Routes `ple_inp_gate.forward(&x)` at the gemma4 PLE block.
    pub graph_ple_gate_buffer: Option<Tensor>,
    /// Persistent PLE gelu_mul output. Same shape as
    /// graph_ple_gate_buffer. Routes `fused_gelu_mul(...)` output.
    pub graph_ple_gelu_buffer: Option<Tensor>,
    /// Persistent PLE proj output. Shape
    /// `[1, 1, hidden_size]` F32. Routes `ple_proj.forward(...)` output.
    pub graph_ple_proj_buffer: Option<Tensor>,
    /// Persistent PLE final output. Shape
    /// `[1, 1, hidden_size]` F32. Routes
    /// `fused_rmsnorm_add_scale` / `fused_rmsnorm_then_add` final output.
    pub graph_ple_final_buffer: Option<Tensor>,
    /// Persistent post_attn_norm + add output. Shape
    /// `[1, 1, hidden_size]` F32. Routes the fused_rmsnorm_then_add
    /// at the post_attn_norm site (line ~3595).
    pub graph_post_attn_norm_buffer: Option<Tensor>,
    /// Persistent ffn_norm.forward output. Shape
    /// `[1, 1, hidden_size]` F32. Routes the RmsNorm.forward output
    /// at the serial-attn ffn_norm site (line ~3606).
    pub graph_x_norm_ffn_buffer: Option<Tensor>,
    /// Persistent attn_norm.forward output. Shape
    /// `[1, 1, hidden_size]` F32. Routes the RmsNorm.forward output
    /// at the start of forward_inner (line ~3375).
    pub graph_x_norm_attn_buffer: Option<Tensor>,
    /// Per-layer PLE input slot for graph-mode forward.
    /// Populated by the engine BEFORE begin_capture each token via
    /// `populate_ple_input_buffers(input_ids)`; the captured graph
    /// reads from this stable per-layer buffer rather than receiving
    /// PLE input as a function parameter (which would be passing
    /// data through capture). Shape `[1, 1, ple_dim=256]` F32 for
    /// gemma4. None for non-gemma4 / non-PLE layers.
    pub graph_ple_input_buffer: Option<Tensor>,
    /// Persistent buffer for `att.broadcast_add(mask)` output in the
    /// GQA fast path of padded_standard_attention (line 2471).
    /// Same shape as graph_attn_qk_buffer: [1, n_kv, n_rep, max_kv] F32.
    /// Attention-internal wiring.
    pub graph_attn_mask_added_buffer: Option<Tensor>,
    /// Persistent buffer for `softmax_last_dim(att)` output in the
    /// GQA fast path (line 2472). Same shape as graph_attn_qk_buffer.
    pub graph_attn_softmax_buffer: Option<Tensor>,
    // -- Path B (gemma4 graph mode unblock) ---------------
    // 6 existing buffers reused (post_attn_norm, x_norm_ffn, post_ffn_norm,
    // ffn_up, ffn_activated, ffn_down). Only 3 genuinely new fields needed
    // for compute_from_kv's serial path:
    pub graph_ffn_gate_out: Option<Tensor>, // split gate output (gemma4)
    pub graph_x_out_scaled: Option<Tensor>, // PLE-skip path scale output
    pub graph_attn_scaled_buffer: Option<Tensor>, // attn_out * residual_scale
    // -- ROOT CAUSE FIX -------------------------------------
    // Keep fresh-alloc captured-region tensors alive across graph
    // replays. The substrate uses sync `cuMemAlloc` for graph capture
    // (cuMemAllocAsync blocks cuStreamBeginCapture). Sync-allocated
    // tensors normally drop at end of Rust scope -> cuMemFree fires
    // OUTSIDE the captured graph -> captured kernel launch args point
    // to freed memory on replay -> ILLEGAL_ADDRESS.
    //
    // Fix: push every captured-region fresh-alloc Tensor into this
    // Vec during the SINGLE capture run. The Vec persists for the
    // captured graph's lifetime (cleared via invalidate_graph_state).
    // Tensor::clone is Arc-clone (cheap, no copy) - multiple holders
    // share the same storage; cuMemFree only fires when ALL Arcs drop.
    //
    // Active only when one of the graph_*_buffer fields is Some
    // (= graph mode is being prepared). Cleared by
    // invalidate_graph_state. Empty + Vec::with_capacity(0) when
    // unused - zero VRAM cost for non-graph models.
    pub graph_alive_tensors: Vec<Tensor>,
    pub cos: Tensor,
    pub sin: Tensor,
    pub neg_inf: Tensor,
    /// Gemma4 global layers: per-dimension RoPE frequency factors (proportional RoPE)
    pub rope_freq_factors: Option<Tensor>,
    /// Pre-computed inverse-frequency table with `rope_freq_factors`
    /// already applied: shape [1, head_dim/2], values
    /// `(1 / rope_base^(2i/d)) / freq_factors[i]`. Set when
    /// `rope_freq_factors` is set. Skips the per-token Tensor::new +
    /// recip + broadcast_mul launches.
    pub rope_factored_freqs: Option<Tensor>,
    // Config (shared across layers)
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// gemma4 SWA per-layer window: Some(w) ⟹ decode attends only to the last w
    /// keys (the windowed-SWA lever); None ⟹ global/full attention.
    pub sliding_window: Option<usize>,
    /// Partial RoPE: only apply to first `rope_dim` of `head_dim` dims (0 = full)
    pub rope_dim: usize,
    /// NoPE (SmolLM3): this layer skips RoPE entirely (RoPE-free layer).
    pub no_rope: bool,
    /// Custom attention scale (None = use 1/√head_dim)
    pub attention_scale: Option<f64>,
    /// Residual connection multiplier (None = 1.0)
    pub residual_scale: Option<f64>,
    pub flags: Arc<GenericLayerFlags>,
}

impl std::fmt::Debug for GenericTransformerLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GenericTransformerLayer(n_head={}, n_kv_head={}, head_dim={})",
            self.n_head, self.n_kv_head, self.head_dim
        )
    }
}

// ------------------------------------------------------------
// Top-level: GenericHeteroTransformer
// ------------------------------------------------------------

/// Whether a layer lives on CUDA or CPU.
#[derive(Debug, Clone, PartialEq)]
enum LayerDevice {
    Cuda(usize),
    Cpu,
}

/// Full model with layers distributed across CUDA + CPU via `HeteroPlan`.
pub struct GenericHeteroTransformer {
    config: Arc<GenericTransformerConfig>,
    embeddings: Embedding, // GPU 0 if table < 1 GB, else CPU
    layers: Vec<GenericTransformerLayer>,
    /// Adapters attached after the load, in the order they were applied. Empty is the
    /// checkpoint as it was read.
    adapters: Vec<String>,
    layer_devs: Vec<LayerDevice>, // parallel to `layers`
    output_norm: WeightedNorm,    // CPU or CUDA - RmsNorm or LayerNorm-with-bias
    output_proj: crate::tensor::quantized::QMatMul, // CPU fallback
    /// Per-vocabulary bias added to the logits (`output.bias`). Present on the
    /// phi2/GPT-NeoX family, absent on most others. It shifts each vocabulary
    /// entry independently, so dropping it leaves the residual stream exact and
    /// still reorders the top of the distribution: the model answers fluent text
    /// unrelated to the prompt. Applied by `apply_output_bias` at EVERY site that
    /// produces logits - there is more than one, and a site that forgets it is
    /// silently wrong.
    output_bias: Option<Tensor>,
    /// EAGLE feature hook: when `capture_feature` is set, `forward` stashes
    /// the pre-final-norm last-layer hidden (the draft-head feature) here. Off by
    /// default -> production forward is unaffected.
    capture_feature: bool,
    last_feature: Option<Tensor>,
    /// Dequantized output projection on CUDA (F16) - legacy cutlass path.
    output_proj_cuda: Option<Tensor>,
    /// Quantized output projection on CUDA (Q4_K/Q6_K). When present, used
    /// in preference to `output_proj_cuda` on the primary forward path to
    /// skip the F16 cutlass gemm (1.45 ms/tok on qwen3) in favor of the
    /// same `mul_mat_vec_q*_K_q8_1` kernel used by per-layer matmuls.
    output_proj_cuda_qmm: Option<crate::tensor::quantized::QMatMul>,
    /// Device the CUDA output projection actually lives on. May differ from the
    /// last layer's GPU: when that GPU is full (e.g. a 24B model + KV leaves no
    /// room for a 131072-vocab lm_head), the projection is placed on another idle
    /// GPU rather than catastrophically falling back to a CPU lm_head.
    output_proj_dev: Option<Device>,
    /// RmsNorm weight on CUDA for GPU-side norm
    output_norm_cuda_weight: Option<Tensor>,
    cuda_devices: HashMap<usize, Device>,
    sliding_window: Option<usize>,
    mask_cache: HashMap<(usize, Option<usize>), Tensor>, // (seq_len, cuda_gpu_idx_or_none)
    // PLE globals (Gemma4) - all CPU
    ple_token_embd: Option<Embedding>, // small vocab or None
    /// BF16 PLE token embedding for large-vocab models (Gemma4: stays in BF16 to save memory)
    ple_token_embd_bf16: Option<(Tensor, usize)>, // (bf16_tensor, total_ple_dim)
    ple_model_proj: Option<crate::tensor::quantized::QMatMul>, // hidden -> n_layers * ple_dim
    ple_proj_norm: Option<RmsNorm>,
    ple_dim: usize, // per-layer PLE dimension (e.g. 256)
    /// Pre-allocated hidden state buffer for CUDA graph replay.
    graph_hidden_buffer: Option<Tensor>,
    /// Pre-allocated logits output buffer for graph replay. The graph's last
    /// op copies logits here so the output is at a stable device pointer
    /// (CUDA graph instantiation remaps internal allocations).
    graph_logits_buffer: Option<Tensor>,
    /// ROOT CAUSE FIX: model-level fresh-alloc tensor lifetime
    /// extension for `compute_all_from_kv`'s LM head path. Sister of the
    /// per-layer `graph_alive_tensors` - but covers tensors created in the
    /// model's post-layer output projection chain (last F32 cast, rms_norm,
    /// F16 cast, matmul output). Without these the captured graph still
    /// crashes with ILLEGAL_ADDRESS on first replay. Cleared by
    /// invalidate_graph_state.
    pub graph_alive_tensors_model: Vec<Tensor>,
    /// Cached layer-consolidation groups for `update_graph_state`. Derived
    /// once from immutable layer config - no need to rebuild HashMaps
    /// every token. Indexed by layer; value = leader layer index for
    /// each kind of state buffer (mask / kv_pos / rope).
    graph_state_groups: Option<GraphStateGroups>,
    ///  tracks whether fast_mmvq workspace has been
    /// pre-grown to max model size. Set on first update_graph_state call.
    workspace_pre_grown: bool,
}

#[derive(Debug, Clone)]
struct GraphStateGroups {
    mask: Vec<usize>,
    kv_pos: Vec<usize>,
    rope: Vec<usize>,
}

impl std::fmt::Debug for GenericHeteroTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GenericHeteroTransformer(arch={}, layers={})",
            self.config.arch,
            self.layers.len()
        )
    }
}

unsafe impl Send for GenericHeteroTransformer {}

