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
// graph_logits_buffer, graph_hidden_buffer, quant_kv, output_proj_*) checks
// `Option::is_some()` / `is_none()` and then `unwrap()`s in the matching
// branch. clippy::unnecessary_unwrap can't reason across the if-condition;
// each site is logically safe. Refactoring the ~20 sites to `match` /
// `if let Some(_)` would touch attention/projection paths in the hot
// decode loop with no behaviour change. Allow the lint for the whole file.
#![allow(clippy::unnecessary_unwrap)]
/// The quantised projection every layer multiplies through.
pub mod projection;

/// The disk cache of a recurrent hybrid's decode state.
pub(crate) mod state_cache;

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

#[cfg(all(test, feature = "gpu"))]
mod card_host_parity;
#[cfg(test)]
mod wide_tests;

mod config;
#[cfg(test)]
mod conversion_tests;
mod dense;
mod dense_layouts;
mod draft_head;
pub use draft_head::set_probe_dir;
mod context_parallel;
#[cfg(test)]
mod conversation_tests;
pub mod expert_profile;
#[cfg(feature = "gpu")]
pub mod expert_tier;
#[cfg(not(feature = "gpu"))]
#[path = "expert_tier_host.rs"]
pub mod expert_tier;
mod graph;
pub mod hetero;
#[cfg(test)]
mod hf_reference_tests;
pub mod hyper;
mod layer_attn;
mod layer_cpu_attn;
mod layer_ffn;
mod layer_forward;
mod layer_rows_cpu;
mod loader;
#[cfg(test)]
mod loading_tests;
pub mod mixer;
mod moe_weights;
pub mod ngram;
mod opencl_layers;
#[cfg(all(test, feature = "opencl"))]
mod opencl_parity;
pub mod qsa;
mod segment;
mod speculate;
mod splice;
#[cfg(test)]
mod state_cache_tests;
#[cfg(test)]
pub(crate) mod tiny_model;
mod tp_forward;
#[cfg(all(test, feature = "gpu"))]
mod tp_tests;
#[cfg(all(test, feature = "cuda", feature = "rocm"))]
mod vendor_split_parity;
#[cfg(all(test, feature = "gpu"))]
mod verify_decode_parity;
#[cfg(test)]
mod verify_rows_cpu_tests;
mod verify_window;
mod vulkan_layers;
#[cfg(all(test, feature = "vulkan"))]
mod vulkan_parity;
mod wide;
pub use config::*;
pub use dense::{text_bytes, DenseCheckpoint};
pub(crate) use loader::tp_file_refusal;
pub use moe_weights::*;
pub use segment::LayerRates;
pub use wide::set_prefill_settle;

pub(crate) use rayon::prelude::*;
pub(crate) use tracing::{info, warn};

pub(crate) use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
pub(crate) use crate::inference::serve::spec_kv_cache::SpecKvCache;
pub(crate) use projection::QMatMul;

/// One transformer layer with all optional architectural variants.
pub struct GenericTransformerLayer {
    /// What mixes the tokens: the attention fields below, or a recurrent operator whose
    /// output projection is `attn_output`.
    pub mixer: mixer::Mixer,
    /// The layer has a feed-forward sublayer; a hybrid block may be a mixer alone.
    pub has_ffn: bool,
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
    /// Output gate: the attention output is multiplied by `sigmoid(attn_gate(x))` before
    /// the output projection. Split at load from a Q projection that interleaves Q and the
    /// gate per head.
    pub attn_gate: Option<QMatMul>,
    // Always present
    pub attn_output: QMatMul,
    /// Optional bias on attn_output projection (phi2)
    pub attn_output_bias: Option<Tensor>,
    /// Per-head attention-sink logits `[n_head]` F32 on the layer's device. Each joins its
    /// head's softmax denominator and contributes no value, so a row's weights sum below 1.
    pub attn_sinks: Option<Tensor>,
    /// The same logits on the host, read by the host attention kernels.
    pub attn_sinks_host: Option<Vec<f32>>,
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
    /// The shares of this layer's heads and feed-forward on the other cards of a tensor-parallel
    /// group, each an ordinary layer of its own width; this layer holds the first share and sums
    /// theirs into its own. Empty when the layer runs whole on its device.
    pub tp_shards: Vec<GenericTransformerLayer>,
    /// The group's communicators when this layer holds shares on other cards and NCCL opens.
    pub tp_comms: Option<std::sync::Arc<TpComms>>,
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
    /// The layer's cache cut into ranges of positions over several cards, when the context is
    /// split: it then holds the history in place of `kv_cache` and the quantised caches.
    pub context: Option<context_parallel::ContextShards>,
    /// The quantised KV cache (Q8_0 or Q4_0 KIVI) when `config.kv_quant` asks for one on a
    /// CUDA layer. It then holds the history: decode attends over it directly and
    /// multi-token paths dequantise it; the F-dtype `kv_cache` is only kept for donor
    /// layers (`populate_dual_kv`).
    #[cfg(feature = "gpu")]
    pub quant_kv: Option<crate::inference::cache::quant_kv::QuantKv>,
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
    /// 1-element i32 on the layer's card holding the position the graph step writes; the
    /// device-length sink attention reads it at replay time.
    pub graph_pos_i32: Option<Tensor>,
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
    /// The rows of the n-gram table the step's token gathers, `[1, 1, heads * row_dim]` F32 on
    /// the card, written before each replay: what the layer that adds them reads.
    pub graph_ngram_rows: Option<Tensor>,
    /// What a draft verify left of this layer's state, until the rows kept are known.
    spec: Option<speculate::LayerPending>,
    /// Whether the forward running now is a verify that records [`Self::spec`].
    spec_record: bool,
    /// Whether the rows running now are read only for the keys and values they leave in the
    /// caches: the layer stops after its token mixer, and its output is not used.
    kv_only: bool,
    /// The capacity of the key and value buffers a recorded indexed layer was last run against:
    /// the step ends, and is recorded again, when the cache outgrows or replaces them.
    pub graph_kv_cap: Option<usize>,
    /// The buffers of a recorded verify's rows: the positions of the rows, their rotation
    /// tables, the n-gram rows of the layer that adds them, and what the layer keeps to return
    /// to the rows that were kept.
    pub verify_pos_i32: Option<Tensor>,
    pub verify_rope_cos: Option<Tensor>,
    pub verify_rope_sin: Option<Tensor>,
    pub verify_ngram_rows: Option<Tensor>,
    verify_keep: Option<verify_window::VerifyKeep>,
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
    /// Window of the causal mask a multi-token forward receives: the model-wide
    /// `attention.sliding_window` applied to every layer, or this layer's own window on a
    /// period-classified model.
    pub prefill_mask_window: Option<usize>,
    /// Partial RoPE: only apply to first `rope_dim` of `head_dim` dims (0 = full)
    pub rope_dim: usize,
    /// Rotary rows `[seq, rope_dim / 2]` the next forward rotates by instead of the table
    /// rows at its position: set around a spliced prefill whose positions are not the
    /// sequence's.
    pub rope_rows: Option<(Tensor, Tensor)>,
    /// Rotary base this layer's frequencies are powers of: the SWA base on a
    /// sliding-window layer that has one, the model's `rope.freq_base` otherwise.
    pub rope_base: f32,
    /// NoPE (SmolLM3): this layer skips RoPE entirely (RoPE-free layer).
    pub no_rope: bool,
    /// Custom attention scale (None = use 1/√head_dim)
    pub attention_scale: Option<f64>,
    /// Attention scores pass through tanh(s/cap)*cap before the mask (Gemma2). Only the
    /// unfused attention applies it; the fused and quantised paths step aside for it.
    pub attn_softcap: Option<f32>,
    /// ALiBi slopes, one per query head: head `h` adds `slope * (key - query)` to its scores
    /// before the mask. Only the unfused host attention applies them; the other paths step
    /// aside, the load offers such a model no card (`host_only_reason`), and a card's
    /// attention refuses the layer.
    pub alibi_slopes: Option<Vec<f32>>,
    /// Residual connection multiplier (None = 1.0)
    pub residual_scale: Option<f64>,
    pub flags: Arc<GenericLayerFlags>,
    /// The mixers of the wide residual, on a model that has one.
    pub hyper: Option<hyper::WideBlock>,
    /// The block-sparse attention indexer, on an attention layer of a model with one.
    pub sparse: Option<qsa::SparseAttention>,
    /// The gated n-gram features this layer adds to the residual before it mixes.
    pub ngram: Option<ngram::PleBlock>,
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
    /// The threads that serve the recorded decode of the cached layers; they stop before the
    /// layers they hold go.
    tier_workers: Vec<expert_tier::TierWorker>,
    /// The hot experts locked in host memory, when the tiered layers do not fit it whole.
    #[cfg(feature = "gpu")]
    hot_locks: Option<Arc<expert_tier::HotLocks>>,
    /// The graph generation of the graphs the engine keeps for the next request.
    kept_graphs_at: Option<u64>,
    config: Arc<GenericTransformerConfig>,
    embeddings: Embedding, // GPU 0 if table < 1 GB, else CPU
    /// The learned table of absolute positions `[positions, dim]`, F32 on the host, added to
    /// the token embeddings of a model that has one.
    position_embd: Option<Tensor>,
    /// The stored embedding table on each card of the plan, when its format is gathered there:
    /// a token's row is decoded on the card that needs it instead of on the host.
    embed_cards: Vec<Arc<crate::tensor::quantized::QTensor>>,
    layers: Vec<GenericTransformerLayer>,
    /// The rows of the draft verify last recorded, until the model is cut back or moves on.
    spec_window: Option<speculate::Window>,
    /// Whether a verify is recording now.
    spec_recording: bool,
    /// The place in the stack of `layers[0]`: zero unless this model holds one node's share of
    /// a stack other nodes run the rest of.
    first_layer: usize,
    /// Adapters attached after the load, in the order they were applied. Empty is the
    /// checkpoint as it was read.
    adapters: Vec<String>,
    layer_devs: Vec<LayerDevice>, // parallel to `layers`
    /// The layers the plan gives a Vulkan device, which run there instead of on their host
    /// build.
    vulkan: vulkan_layers::VulkanLayers,
    /// The layers the plan gives an OpenCL device, which run there likewise.
    opencl: opencl_layers::OpenClLayers,
    output_norm: WeightedNorm, // CPU or CUDA - RmsNorm or LayerNorm-with-bias
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
    /// The output projection's share of the bytes a decoded token reads: what reading only a
    /// prefix of its rows saves, against what the tokens past the prefix cost.
    head_share: f32,
    /// Device the CUDA output projection actually lives on. May differ from the
    /// last layer's GPU: when that GPU is full (e.g. a 24B model + KV leaves no
    /// room for a 131072-vocab lm_head), the projection is placed on another idle
    /// GPU rather than catastrophically falling back to a CPU lm_head.
    output_proj_dev: Option<Device>,
    /// RmsNorm weight on CUDA for GPU-side norm
    output_norm_cuda_weight: Option<Tensor>,
    cuda_devices: HashMap<usize, Device>,
    sliding_window: Option<usize>,
    /// Causal masks by (query and key lengths, card or host, window).
    mask_cache: HashMap<(usize, Option<usize>, Option<usize>), Tensor>,
    // PLE globals (Gemma4) - all CPU
    /// The per-layer token table, in the file's format, decoded per lookup.
    ple_token_embd: Option<Embedding>,
    ple_model_proj: Option<crate::tensor::quantized::QMatMul>, // hidden -> n_layers * ple_dim
    ple_proj_norm: Option<RmsNorm>,
    ple_dim: usize, // per-layer PLE dimension (e.g. 256)
    /// Pre-allocated hidden state buffer for CUDA graph replay.
    graph_hidden_buffer: Option<Tensor>,
    /// The first card's working copy of the hidden state: the step's ops read the embedding in
    /// `graph_hidden_buffer` and hand the state on through this one, so a step run again starts
    /// from the embedding.
    graph_work_buffer: Option<Tensor>,
    /// The graph step's hidden-state hops between consecutive cards, one per boundary.
    #[cfg(feature = "gpu")]
    graph_hops: Vec<graph::GraphHop>,
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
    /// The recordings that passed every check once with this model: bit 0 the decode step,
    /// bit `r` the verify of `r` rows. A later recording of one of them keeps the check of its
    /// replay and drops the others.
    pub(crate) graph_proven: u64,
    /// Bumped by every explicit drop of the graph state and every move of the expert slots;
    /// [`Self::graph_generation`] adds the layers' buffer moves to it.
    pub(crate) graph_generation: u64,
    /// A number no other model built in this process carries: graphs kept for one model are
    /// never replayed over another.
    pub(crate) graph_instance: u64,
    /// Cached layer-consolidation groups for `update_graph_state`. Derived
    /// once from immutable layer config - no need to rebuild HashMaps
    /// every token. Indexed by layer; value = leader layer index for
    /// each kind of state buffer (mask / kv_pos / rope).
    graph_state_groups: Option<GraphStateGroups>,
    ///  tracks whether fast_mmvq workspace has been
    /// pre-grown to max model size. Set on first update_graph_state call.
    workspace_pre_grown: bool,
    /// The position of the token the decode step in flight carries: what an eagerly run layer
    /// of a captured step reads where a captured one reads the position buffers.
    graph_pos: usize,
    /// The buffers of the recorded draft verifies.
    verify: verify_window::VerifyBuffers,
    /// The last layer's row of the recorded step, `[1, 1, wide]`, where its replays leave it.
    graph_feature: Option<Tensor>,
    /// Resident sequences copied aside after a request (`snapshot_kv`), most recently
    /// used last by `tick`; empty unless the configuration asks for them.
    kv_snapshots: Vec<KvSnapshot>,
    kv_snapshot_tick: u64,
    /// How long a snapshot is kept after it was last used; `None` keeps it until memory is needed.
    kv_lease: Option<std::time::Duration>,
    /// The token count up to which the prompt being read is the same in the next turn of its
    /// conversation: the state there is kept for that turn.
    prefix_boundary: Option<usize>,
    /// Tokens the last prefix restore covered.
    prefix_covered: usize,
    /// The most a request's decode took of a card beyond its start, in bytes.
    decode_reserve: u64,
    /// The disk cache of this model's decode state, when the operator opened a store and the
    /// model carries a recurrent state.
    hybrid_cache: Option<state_cache::HybridStateCache>,
    /// A vision tower whose embeddings are spliced over a placeholder run.
    spliced_vision: Option<splice::SplicedVision>,
    /// A speech tower whose embeddings are spliced over the placeholder of a prompt.
    #[cfg(feature = "audio")]
    spliced_audio: Option<splice::SplicedAudio>,
    /// Held while some layer keeps its routed experts in host memory: the decode pool's
    /// workers then spin through the card work between host layers instead of parking.
    host_experts_hot: Option<crate::tensor::quant_cpu::gemv_pool::HotGuard>,
    /// The mixer that stands for the final norm of a wide-residual model.
    output_hyper: Option<hyper::HyperMixer>,
    /// The n-gram embedding table, the tokens it hashes behind, and the rows gathered for
    /// the forward in progress.
    ngram: Option<NgramModel>,
    /// The draft head the checkpoint shipped beside its layers, if the model came with one.
    draft: Option<draft_head::DraftHead>,
}

/// The model-level part of the n-gram embedding; the token history lives with its layer.
pub struct NgramModel {
    pub meta: ngram::NgramMeta,
    pub table: ngram::NgramTable,
    /// The gathered embeddings `[1, seq, heads * row_dim]` of the forward in progress.
    pub pending: Option<Tensor>,
    /// The history as it stood before the last gather, with the position that gather started
    /// at: what a forward run again from there, after a failure, starts from.
    pub before: Option<(usize, ngram::NgramHistory)>,
}

/// A resident sequence copied aside: its tokens and every layer's KV, taken after a
/// request so a later one can resume from it instead of prefilling.
pub struct KvSnapshot {
    pub tokens: Vec<u32>,
    pub kv_len: usize,
    pub(crate) layers: Vec<LayerKvSnapshot>,
    pub(crate) tick: u64,
    /// The last row's logits of the forward that ended at `kv_len`: a recurrent model's
    /// prefix snapshot resumes decoding from them.
    pub(crate) logits: Option<Tensor>,
    /// The draft head's state at `kv_len`, when the head had read the same prefix.
    pub(crate) draft: Option<state_cache::DraftHeld>,
    /// When the snapshot was taken or last restored: the lease runs from here.
    pub(crate) used: std::time::Instant,
    /// Times a later prompt restored this conversation's snapshot, the lineage of the ones it
    /// replaced included: a conversation that came back is kept over one that never did.
    pub(crate) hits: u32,
}

impl KvSnapshot {
    /// What decides which snapshot goes first when memory is needed: the least recently used,
    /// after every one that never came back.
    pub(crate) fn rank(&self) -> u64 {
        const CAME_BACK: u64 = 1 << 40;
        self.tick + if self.hits > 0 { CAME_BACK } else { 0 }
    }

    /// Device memory this snapshot holds across its layers (host copies are not counted).
    pub fn device_bytes(&self) -> u64 {
        self.layers.iter().map(LayerKvSnapshot::device_bytes).sum()
    }
}

/// A quantised cache's packed bytes copied aside; uninhabited on a build without CUDA.
#[cfg(feature = "gpu")]
pub(crate) type QuantSnap = crate::inference::cache::quant_kv::QuantKvSnapshot;

#[cfg(not(feature = "gpu"))]
pub(crate) enum QuantSnap {}

#[cfg(not(feature = "gpu"))]
impl QuantSnap {
    pub fn len(&self) -> usize {
        match *self {}
    }
    pub fn device_bytes(&self) -> u64 {
        match *self {}
    }
    pub fn host_rows(&self, _from: usize, _to: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        match *self {}
    }
}

pub(crate) struct LayerKvSnapshot {
    pub(crate) spec: Option<crate::inference::serve::spec_kv_cache::SpecKvSnapshot>,
    /// A quantised cache's raw codes and scales, in place of `spec`.
    pub(crate) quant: Option<QuantSnap>,
    pub(crate) cpu_f16: Option<crate::inference::cache::cpu_f16_kv::CpuF16Kv>,
    /// A recurrent mixer's state at the snapshot's length.
    pub(crate) mixer: Option<mixer::MixerState>,
    /// The snapshots of the layer's tensor-parallel shares, in their order.
    pub(crate) shards: Vec<LayerKvSnapshot>,
}

impl LayerKvSnapshot {
    /// Device memory this layer's snapshot and its shares' hold.
    fn device_bytes(&self) -> u64 {
        self.spec.as_ref().map_or(0, |s| s.device_bytes())
            + self.quant.as_ref().map_or(0, |q| q.device_bytes())
            + self.mixer.as_ref().map_or(0, |m| m.device_bytes())
            + self.shards.iter().map(Self::device_bytes).sum::<u64>()
    }

    /// Positions the layer's KV snapshot holds, from whichever form it has; `None` for a
    /// layer that keeps no KV of its own.
    pub(crate) fn kv_len(&self) -> Option<usize> {
        match (&self.spec, &self.quant, &self.cpu_f16) {
            (Some(spec), _, _) => Some(spec.len()),
            (None, Some(q), _) => Some(q.len()),
            (None, None, Some(c)) => Some(c.len()),
            (None, None, None) => None,
        }
    }

    /// Tokens `[from, to)` of the layer's KV snapshot as host f32 rows, K then V; `None`
    /// for a layer that keeps no KV of its own.
    pub(crate) fn host_rows(
        &self,
        from: usize,
        to: usize,
    ) -> crate::tensor::Result<Option<(Vec<f32>, Vec<f32>)>> {
        match (&self.spec, &self.quant, &self.cpu_f16) {
            (Some(spec), _, _) => spec.host_rows(from, to).map(Some),
            (None, Some(q), _) => q
                .host_rows(from, to)
                .map(Some)
                .map_err(|e| crate::tensor::Error::msg(e.to_string())),
            (None, None, Some(c)) => Ok(Some(c.host_rows(from, to))),
            (None, None, None) => Ok(None),
        }
    }
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

/// The NCCL communicators of a tensor-parallel group, one per card on that card's compute
/// stream, rank 0 on the layers' own card. They are used only by the thread that runs the
/// model's forward, which holds the model mutably.
pub struct TpComms {
    #[cfg(feature = "gpu")]
    comms: Vec<crate::tensor::gpu_runtime::nccl::safe::Comm>,
    /// The hidden state a split layer last returned on rank 0, and its copies on the other
    /// ranks: the next split layer starts from them instead of sending its input out.
    #[cfg(feature = "gpu")]
    replicas: std::sync::Mutex<Option<(Tensor, Vec<Tensor>)>>,
}

unsafe impl Send for TpComms {}
unsafe impl Sync for TpComms {}

#[cfg(feature = "gpu")]
impl TpComms {
    /// Communicators over `devices`, the layers' own card first; `None` when NCCL does not
    /// open or refuses the group.
    pub(crate) fn new(devices: &[Device]) -> Option<Self> {
        if !crate::tensor::cuda::runtime::nccl_present() {
            tracing::info!(
                "tensor parallelism moves its activations through copies: NCCL does not open"
            );
            return None;
        }
        let streams = devices
            .iter()
            .map(|d| d.as_cuda_device().ok().map(|c| c.cuda_stream()))
            .collect::<Option<Vec<_>>>()?;
        match crate::tensor::gpu_runtime::nccl::safe::Comm::from_devices(streams) {
            Ok(comms) => Some(Self {
                comms,
                replicas: std::sync::Mutex::new(None),
            }),
            Err(e) => {
                tracing::warn!("tensor parallelism moves its activations through copies: NCCL refused the group: {e:?}");
                None
            }
        }
    }

    /// Whether the group's ranks are `x`'s card and the cards of `ranks - 1` shares, and `x` a
    /// contiguous F32 tensor the collectives take as it is.
    fn takes(&self, x: &Tensor, ranks: usize) -> bool {
        self.comms.len() == ranks && x.dtype() == DType::F32 && x.is_contiguous()
    }

    /// `x`, on rank 0's card, sent to the cards of `to`, ranks 1 onwards. The other ranks'
    /// streams wait for rank 0's first, which joins them to a step rank 0 is recording.
    fn send_out(&self, x: &Tensor, to: &[Device]) -> Result<Vec<Tensor>> {
        use crate::tensor::gpu_runtime::nccl::safe::{group_end, group_start};
        let nccl = |e: crate::tensor::gpu_runtime::nccl::result::NcclError| {
            crate::tensor::Error::msg(format!("nccl: {e:?}"))
        };
        let ready = crate::tensor::cuda_ext::record_event(&self.comms[0].stream())?;
        for c in &self.comms[1..] {
            crate::tensor::cuda_ext::wait_event(&c.stream(), &ready)?;
        }
        let src = crate::tensor::cuda_ext::f32_slice_of(x)?;
        let view = src.view()?;
        let n = x.elem_count();
        let mut outs = self.comms[1..]
            .iter()
            .map(|c| unsafe { c.stream().alloc::<f32>(n) })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| crate::tensor::Error::msg(format!("{e:?}")))?;
        // The group is closed on every path: a group left open on a thread defers every later
        // collective of that thread.
        group_start().map_err(nccl)?;
        let issued = outs.iter_mut().enumerate().try_for_each(|(rank, out)| {
            self.comms[0].send(&view, rank as i32 + 1)?;
            self.comms[rank + 1].recv(out, 0).map(|_| ())
        });
        let ended = group_end();
        issued.map_err(nccl)?;
        ended.map_err(nccl)?;
        outs.into_iter()
            .zip(to)
            .map(|(out, dev)| {
                crate::tensor::cuda_ext::tensor_from_f32_slice(out, x.shape().clone(), dev)
            })
            .collect()
    }

    /// Every rank's `parts[rank]` summed onto every rank's card. Each part is a contiguous F32
    /// tensor on its rank's card.
    pub(crate) fn all_reduce(&self, parts: &[Tensor]) -> Result<Vec<Tensor>> {
        use crate::tensor::gpu_runtime::nccl::safe::{group_end, group_start, ReduceOp};
        let nccl = |e: crate::tensor::gpu_runtime::nccl::result::NcclError| {
            crate::tensor::Error::msg(format!("nccl: {e:?}"))
        };
        let slices = parts
            .iter()
            .map(crate::tensor::cuda_ext::f32_slice_of)
            .collect::<Result<Vec<_>>>()?;
        let views = slices
            .iter()
            .map(|s| s.view())
            .collect::<Result<Vec<_>>>()?;
        let mut outs = self
            .comms
            .iter()
            .zip(parts)
            .map(|(c, p)| unsafe { c.stream().alloc::<f32>(p.elem_count()) })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| crate::tensor::Error::msg(format!("{e:?}")))?;
        group_start().map_err(nccl)?;
        let issued = self
            .comms
            .iter()
            .zip(&views)
            .zip(outs.iter_mut())
            .try_for_each(|((comm, view), out)| {
                comm.all_reduce(view, out, &ReduceOp::Sum).map(|_| ())
            });
        let ended = group_end();
        issued.map_err(nccl)?;
        ended.map_err(nccl)?;
        outs.into_iter()
            .zip(parts)
            .map(|(out, p)| {
                crate::tensor::cuda_ext::tensor_from_f32_slice(out, p.shape().clone(), &p.device())
            })
            .collect()
    }

    /// `x` on every rank: the copies a split layer kept when `x` is what it returned on rank 0,
    /// else `x` sent out to the cards of `to`, ranks 1 onwards.
    pub(super) fn replicas_of(&self, x: &Tensor, to: &[Device]) -> Result<Vec<Tensor>> {
        let kept = self
            .replicas
            .lock()
            .map_err(|_| crate::tensor::Error::msg("tensor-parallel replicas poisoned"))?;
        if let Some((own, copies)) = kept.as_ref() {
            if own.shares_storage(x) && own.dims() == x.dims() {
                return Ok(std::iter::once(x.clone())
                    .chain(copies.iter().cloned())
                    .collect());
            }
        }
        drop(kept);
        let copies = self.send_out(x, to)?;
        Ok(std::iter::once(x.clone()).chain(copies).collect())
    }

    /// Keeps `all`, one hidden state per rank, for the next split layer.
    pub(super) fn keep(&self, mut all: Vec<Tensor>) -> Result<()> {
        let mut kept = self
            .replicas
            .lock()
            .map_err(|_| crate::tensor::Error::msg("tensor-parallel replicas poisoned"))?;
        let own = all.remove(0);
        *kept = Some((own, all));
        Ok(())
    }

    /// Drops the kept copies and has rank 0's stream wait for every other rank's, which ends
    /// their part of a step rank 0 is recording.
    pub(super) fn forget(&self) -> Result<()> {
        if let Ok(mut kept) = self.replicas.lock() {
            kept.take();
        }
        self.join()
    }

    /// Rank 0's stream waits for the work issued so far on every other rank's.
    pub(super) fn join(&self) -> Result<()> {
        for c in &self.comms[1..] {
            let done = crate::tensor::cuda_ext::record_event(&c.stream())?;
            crate::tensor::cuda_ext::wait_event(&self.comms[0].stream(), &done)?;
        }
        Ok(())
    }

    /// `own`, on rank 0's card, plus the partials of ranks 1 onwards, summed on rank 0's card.
    /// Rank 0's stream then waits for the others, which ends their part of a recorded step.
    fn sum_in(&self, own: &Tensor, parts: &[Tensor]) -> Result<Tensor> {
        use crate::tensor::gpu_runtime::nccl::safe::{group_end, group_start, ReduceOp};
        let nccl = |e: crate::tensor::gpu_runtime::nccl::result::NcclError| {
            crate::tensor::Error::msg(format!("nccl: {e:?}"))
        };
        let slices = std::iter::once(own)
            .chain(parts)
            .map(crate::tensor::cuda_ext::f32_slice_of)
            .collect::<Result<Vec<_>>>()?;
        let views = slices
            .iter()
            .map(|s| s.view())
            .collect::<Result<Vec<_>>>()?;
        let mut out = unsafe { self.comms[0].stream().alloc::<f32>(own.elem_count()) }
            .map_err(|e| crate::tensor::Error::msg(format!("{e:?}")))?;
        group_start().map_err(nccl)?;
        let issued = self.comms[0]
            .reduce(&views[0], Some(&mut out), &ReduceOp::Sum, 0)
            .and_then(|_| {
                self.comms[1..]
                    .iter()
                    .zip(&views[1..])
                    .try_for_each(|(comm, view)| {
                        comm.reduce(
                            view,
                            None::<&mut crate::tensor::gpu_runtime::driver::CudaSlice<f32>>,
                            &ReduceOp::Sum,
                            0,
                        )
                        .map(|_| ())
                    })
            });
        let ended = group_end();
        issued.map_err(nccl)?;
        ended.map_err(nccl)?;
        self.join()?;
        crate::tensor::cuda_ext::tensor_from_f32_slice(out, own.shape().clone(), &own.device())
    }
}

/// `x` on the card of every share in `shards`: sent through the group's communicators when they
/// take it, copied otherwise.
pub(super) fn tp_send_out(
    x: &Tensor,
    shards: &[GenericTransformerLayer],
    comms: Option<&TpComms>,
) -> Result<Vec<Tensor>> {
    #[cfg(feature = "gpu")]
    if let Some(c) = comms.filter(|c| c.takes(x, shards.len() + 1)) {
        let to: Vec<Device> = shards.iter().map(|s| s.cos.device()).collect();
        return c.send_out(x, &to);
    }
    let _ = comms;
    shards
        .iter()
        .map(|s| x.to_device(&s.cos.device()))
        .collect()
}

/// `own` plus every partial output of a tensor-parallel layer's other shares, on `own`'s card:
/// reduced through the group's communicators when they take them, copied and added otherwise.
pub(super) fn tp_sum(own: Tensor, parts: Vec<Tensor>, comms: Option<&TpComms>) -> Result<Tensor> {
    #[cfg(feature = "gpu")]
    if let Some(c) = comms.filter(|c| {
        c.takes(&own, parts.len() + 1) && parts.iter().all(|p| c.takes(p, parts.len() + 1))
    }) {
        return c.sum_in(&own, &parts);
    }
    let _ = comms;
    let dev = own.device();
    parts
        .into_iter()
        .try_fold(own, |sum, part| sum + part.to_device(&dev)?)
}
