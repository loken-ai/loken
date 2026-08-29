//! MoE and fused-FFN dispatch: the host side of the kernels NVCC builds into `libloken_imma.a`.
//!
//! Every function here does the same three things - read the addresses out of the tensors and
//! quantised tensors it was handed, launch one `extern "C"` kernel on the device's compute
//! stream, and wrap what the kernel wrote back into a tensor. Routing, the expert GEMMs and the
//! fused attention epilogue each get their own file below.

use crate::inference::kernel::cuda_launchers;
use crate::tensor::cuda_ext::{tensor_from_cuda_storage, CudaStorage, DevicePtr};
use crate::tensor::{DType, Result, StorageView, Tensor};
use core::ffi::c_void;

cuda_launchers! {
    loken_topk_softmax(
        logits: *const f32, weights: *mut f32, ids: *mut u32, n_rows: i32, n_experts: i32,
        n_expert_used: i32, with_norm: i32);
}

mod routing;
pub use routing::*;
mod norm;
pub use norm::*;
pub(crate) mod gemm;

/// The two expert-GEMM paths judged against each other; nothing else covers the prefill one.
#[cfg(test)]
mod prefill_parity;

/// The IMMA down projection and the fused SiLU gate‖up, judged against a host reduction over
/// the dequantised weights - a reference neither of them can agree with by accident.
#[cfg(test)]
mod imma_reference;
pub use gemm::*;
mod dense;
pub use dense::*;

cuda_launchers! {
    loken_attn_post_qkv_decode(qkv: *const f32, q_norm_w: *const f32, k_norm_w: *const f32,
        rope_cos: *const c_void, rope_sin: *const c_void,
        q_out: *mut c_void, k_out: *mut c_void,
        v_out: *mut c_void, n_q: i32, n_kv: i32, hd: i32, rope_pos: i32,
        rms_eps: f32, q_scale: f32, dtype: i32, rope_style: i32);
    loken_attn_post_qkv_decode_qf32(
        qkv: *const f32, q_norm_w: *const f32, k_norm_w: *const f32,
        rope_cos: *const c_void, rope_sin: *const c_void,
        q_out: *mut f32, k_out: *mut c_void, v_out: *mut f32, n_q: i32, n_kv: i32,
        hd: i32, rope_pos: i32, rms_eps: f32, q_scale: f32, dtype: i32, rope_style: i32);
    loken_attn_post_qkv_decode_qf32_no_norm(
        qkv: *const f32, rope_cos: *const c_void,
        rope_sin: *const c_void, q_out: *mut f32, k_out: *mut c_void,
        v_out: *mut f32, n_q: i32, n_kv: i32, hd: i32, rope_pos: i32, q_scale: f32, dtype: i32,
        rope_style: i32);
    loken_attn_post_qkv_decode_qf32_no_norm_partial_rope(
        qkv: *const f32, rope_cos: *const c_void,
        rope_sin: *const c_void, q_out: *mut f32, k_out: *mut c_void,
        v_out: *mut f32, n_q: i32, n_kv: i32, hd: i32, rope_dim: i32, rope_pos: i32,
        q_scale: f32, dtype: i32, rope_style: i32);
}
cuda_launchers! {
    loken_kv_residual_scatter_f16(
        src: *const c_void, dst: *mut c_void, n_kv: i32, head_dim: i32,
        slot: i32);
    loken_kv_residual_scatter_f16_dev_slot(
        src: *const c_void, dst: *mut c_void,
        slot_dev: *const c_void, n_kv: i32, head_dim: i32);
    loken_q4_v_scatter_bytes_dev_pos(
        src: *const c_void, dst: *mut c_void,
        pos_dev: *const c_void, token_bytes: i32);
    loken_flush_k_residual_q4_dev_pos(
        residual: *const c_void, k_blocks: *mut c_void,
        pos_dev: *const c_void, n_kv: i32, head_dim: i32, max_seq_blocks: i32);
    loken_rms_quantize_q8_1_bf16(
        x: *const c_void, w_norm: *const f32, y_q8_1: *mut c_void,
        hidden: i32, rms_eps: f32);
    loken_gated_delta_net(
        q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta: *const f32,
        state_in: *const f32, o_out: *mut f32, state_out: *mut f32, h: i32, n_tokens: i32,
        s_v: i32, sq1: i64, sq2: i64, sv1: i64, sv2: i64, sg1: i64, sg2: i64, scale: f32);
    loken_fused_conv_silu(
        qkv: *const f32, conv_state: *const f32, w: *const f32, out: *mut f32,
        new_conv_state: *mut f32, c: i32, seq: i32, k: i32);
    loken_fused_conv_silu_f16in(
        qkv: *const c_void, conv_state: *const f32, w: *const f32, out: *mut f32,
        new_conv_state: *mut f32, c: i32, seq: i32, k: i32);
    loken_deltanet_gate_f16in(
        alpha: *const c_void, beta_in: *const c_void, a_log: *const f32,
        dt_bias: *const f32, g_pre: *mut f32, beta: *mut f32, n: i32, h: i32);
    loken_zgate_rmsnorm(
        o: *const f32, z: *const f32, norm_w: *const f32, y: *mut f32, n: i32, d: i32,
        eps: f32);
    loken_zgate_rmsnorm_f16io(o: *const f32, z: *const c_void, norm_w: *const f32,
        y: *mut c_void, n: i32, d: i32, eps: f32);
    loken_deltanet_gate(
        alpha: *const f32, beta_in: *const f32, a_log: *const f32, dt_bias: *const f32,
        g_pre: *mut f32, beta: *mut f32, n: i32, h: i32);
    loken_l2norm_gqa(
        x: *const f32, y: *mut f32, seq: i32, kg: i32, kd: i32, rep: i32, eps: f32);
    loken_head_rmsnorm(x: *const f32, w: *const f32, y: *mut f32, n: i32, d: i32, eps: f32);
    loken_gate_gemv_f32(xs: *const c_void, gate_w: *const c_void,
        logits: *mut c_void, hidden: i32, n_rows: i32, n_experts: i32);
    loken_gate_topk_softmax(xs: *const c_void, gate_w: *const c_void,
        weights: *mut c_void, ids: *mut c_void, hidden: i32, n_rows: i32,
        n_experts: i32, n_expert_used: i32, with_norm: i32);
    loken_gate_topk_sigmoid(xs: *const c_void, gate_w: *const c_void,
        bias: *const c_void, weights: *mut c_void,
        ids: *mut c_void, hidden: i32, n_rows: i32, n_experts: i32,
        n_expert_used: i32, with_norm: i32, scale: f32);
    loken_topk_sigmoid_post(logits: *const c_void, bias: *const c_void,
        weights: *mut c_void, ids: *mut c_void, n_rows: i32,
        n_experts: i32, n_expert_used: i32, with_norm: i32, scale: f32);
    loken_argsort_small_u32(input: *const c_void, sorted: *mut c_void,
        index: *mut c_void, m: i32);
    loken_lfm2_shortconv_f16io(
        bcx: *const c_void, state_in: *const c_void,
        conv_w: *const c_void, y_out: *mut c_void,
        state_out: *mut c_void, b: i32, d: i32, l: i32);
}
cuda_launchers! {
    loken_moe_gemm_gguf_gate_up_gelu_mul_concat(
        input: *const f32, gate_up_weights: *const c_void,
        sorted_token_ids: *const i32, expert_ids: *const i32, output: *mut c_void,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, gguf_dtype: i32);
    loken_moe_gemm_gguf_gate_up_silu_mul(
        inputs: *const f32, gate_weights: *const c_void,
        up_weights: *const c_void, sorted_token_ids: *const i32,
        expert_ids: *const i32, outputs: *mut f32, num_experts: i32, topk: i32, size_m: i32,
        size_n: i32, size_k: i32, quant_type: i32);
    loken_moe_gemm_gguf_gate_up_swiglu_oai(
        inputs: *const f32, gate_weights: *const c_void,
        up_weights: *const c_void, sorted_token_ids: *const i32,
        expert_ids: *const i32, gate_bias: *const f32, up_bias: *const f32, outputs: *mut f32,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, quant_type: i32,
        alpha: f32, limit: f32);
}
cuda_launchers! {
    loken_cast_init_f32_from_dtype(
        dst: *mut f32, src: *const c_void, n: i32, dtype: i32);
    loken_moe_gemm_gguf_down_reduce(
        input: *const f32, weights: *const c_void, sorted_token_ids: *const i32,
        expert_ids: *const i32, topk_weights: *const f32, output: *mut f32, num_experts: i32,
        topk: i32, size_m: i32, size_n: i32, size_k: i32, gguf_dtype: i32,
        down_bias: *const f32);
}
cuda_launchers! {
    loken_moe_q4k_imma_m8_gate_up(
        gate_up_weights: *const c_void, q8_1: *const c_void,
        sorted_token_ids: *const i32, expert_ids: *const i32, output: *mut f32,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32);
    loken_moe_q4k_imma_m8_down(
        weights: *const c_void, q8_1: *const c_void,
        sorted_token_ids: *const i32, expert_ids: *const i32, topk_weights: *const f32,
        output: *mut f32, num_experts: i32, topk: i32, size_m: i32, hidden: i32, size_k: i32);
    loken_dense_q4k_imma_m8_actmul(
        gate: *const c_void, up: *const c_void,
        q8_1: *const c_void, output: *mut f32, size_m: i32, size_n: i32,
        size_k: i32, use_gelu: i32);
    loken_dense_q4k_imma_m8_plain(
        w: *const c_void, q8_1: *const c_void, output: *mut f32,
        size_m: i32, size_n: i32, size_k: i32);
    // Q5_K siblings (block_q5_K = block_q4_K + qh 5th-bit plane).
    loken_dense_q5k_imma_m8_plain(
        w: *const c_void, q8_1: *const c_void, output: *mut f32,
        size_m: i32, size_n: i32, size_k: i32);
    loken_dense_q5k_imma_m8_actmul(
        gate: *const c_void, up: *const c_void,
        q8_1: *const c_void, output: *mut f32, size_m: i32, size_n: i32,
        size_k: i32, use_gelu: i32);
}
cuda_launchers! {
    loken_rms_quantize_q8_1(
        x: *const f32, w_norm: *const f32, y_q8_1: *mut c_void, hidden: i32,
        rms_eps: f32);
}
cuda_launchers! {
    loken_moe_gemm_gguf(
        input: *const f32, weights: *const c_void, sorted_token_ids: *const i32,
        expert_ids: *const i32, topk_weights: *const f32, output: *mut c_void,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, gguf_dtype: i32);
    loken_moe_gemm_gguf_prefill(
        input: *const c_void, weights: *const u8, sorted_token_ids: *const i32,
        expert_ids: *const i32, topk_weights: *const f32, output: *mut c_void,
        num_experts: i32, topk: i32, size_m: i32, size_n: i32, size_k: i32, input_dtype: i32,
        gguf_dtype: i32);
}
cuda_launchers! {
    loken_batched_argmax_f16(
        logits: *const std::ffi::c_void, out: *mut u32, n_rows: i32, vocab: i32);
    loken_batched_argmax_f32(
        logits: *const std::ffi::c_void, out: *mut u32, n_rows: i32, vocab: i32);
}
cuda_launchers! {
    loken_batched_topk_denom_f16(
        logits: *mut std::ffi::c_void, out_logit: *mut f32, out_idx: *mut u32,
        out_stats: *mut f32, n_rows: i32, vocab: i32, k: i32, inv_temp: f32);
    loken_batched_topk_denom_f32(
        logits: *mut std::ffi::c_void, out_logit: *mut f32, out_idx: *mut u32,
        out_stats: *mut f32, n_rows: i32, vocab: i32, k: i32, inv_temp: f32);
    loken_repeat_penalty_f16(
        logits: *mut std::ffi::c_void, rows: *const i32, toks: *const u32, rp: f32, n: i32,
        vocab: i32);
    loken_repeat_penalty_f32(
        logits: *mut std::ffi::c_void, rows: *const i32, toks: *const u32, rp: f32, n: i32,
        vocab: i32);
}
cuda_launchers! {
    loken_add_rms_norm(a: *const f32, b: *const f32, gamma: *const f32, xs_out: *mut f32,
        normed_out: *mut f32, hidden: i32, eps: f32);
}
