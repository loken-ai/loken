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
    loken_stable_sort_u32(keys: *const u32, sorted: *mut u32, order: *mut u32, n: i32);
}

mod routing;
pub use routing::*;
mod norm;
pub use norm::*;
pub(crate) mod gemm;
mod mmq_experts;
pub use mmq_experts::*;

/// The expert-batched tiled matmul against the decode kernels and the float answer.
#[cfg(test)]
mod mmq_parity;

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
    loken_attn_post_qkv_decode_qf32_split(
        q_in: *const f32, k_in: *const f32, v_in: *const f32, q_norm_w: *const f32,
        k_norm_w: *const f32, rope_cos: *const c_void, rope_sin: *const c_void,
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
    loken_gdn_prep(qkv: *const c_void, window: *mut f32, w: *const f32, alpha: *const c_void,
        beta_in: *const c_void, a_log: *const f32, dt_bias: *const f32, q_out: *mut f32,
        k_out: *mut f32, v_out: *mut f32, g_out: *mut f32, beta_out: *mut f32,
        new_window: *mut f32, kg: i32, kd: i32, vg: i32, vd: i32, k: i32, tiled: i32,
        eps: f32, in_f16: i32);
    loken_shexp_swiglu(gate: *const c_void, up: *const c_void, out: *mut f32, n: i32,
        gate_f16: i32, up_f16: i32);
    loken_shexp_tail(routed: *const f32, down: *const c_void, x: *const f32, w: *const f32,
        out: *mut f32, n: i32, down_f16: i32);
    loken_qk_prep(q: *const c_void, k: *const c_void, v: *const c_void, qw: *const f32,
        kw: *const f32, cos: *const f32, sin: *const f32, qo: *mut f32, ko: *mut f32,
        vo: *mut f32, n_head: i32, n_kv: i32, hd: i32, rope_dim: i32, interleaved: i32, eps: f32,
        in_f16: i32);
    loken_deltanet_gate_f16in(
        alpha: *const c_void, beta_in: *const c_void, a_log: *const f32,
        dt_bias: *const f32, g_pre: *mut f32, beta: *mut f32, n: i32, h: i32);
    loken_zgate_rmsnorm(
        o: *const f32, z: *const f32, norm_w: *const f32, y: *mut f32, n: i32, d: i32,
        eps: f32, sigmoid_gate: i32);
    loken_zgate_rmsnorm_f16io(o: *const f32, z: *const c_void, norm_w: *const f32,
        y: *mut c_void, n: i32, d: i32, eps: f32, sigmoid_gate: i32);
    loken_indexer_append(key: *const f32, tail: *mut f32, pooled: *mut f32, gamma: *const f32,
        rot: *const f32, pos_dev: *const i32, pos_host: i32, hd: i32, block: i32, half: i32,
        eps: f32);
    loken_indexer_select(q: *const f32, qgamma: *const f32, rot: *const f32, pooled: *const f32,
        scores: *mut f32, sel: *mut u32, count: *mut u32, pos_dev: *const i32, pos_host: i32,
        heads: i32, hd: i32, block: i32, budget: i32, half: i32, eps: f32, scale: f32,
        max_blocks: i32, rows: i32, rot_stride: i32, sel_stride: i32);
    loken_indexer_rows_in(rows: *const f32, pooled: *mut f32, first: i32, n: i32, hd: i32);
    loken_indexer_rows_out(pooled: *const f32, rows: *mut f32, first: i32, n: i32, hd: i32);
    loken_kv_append_pos(k_new: *const f32, v_new: *const f32, kbuf: *mut c_void,
        vbuf: *mut c_void, pos_dev: *const i32, pos_host: i32, n_kv: i32, cap: i32, hd: i32);
    loken_selected_attn(q: *const f32, kbuf: *const c_void, vbuf: *const c_void,
        sel: *const u32, count: *const u32, out: *mut f32, part_o: *mut f32, part_ml: *mut f32,
        n_head: i32, n_kv: i32, cap: i32, hd: i32, ncap: i32, splits: i32, scale: f32);
    loken_selected_attn_rows(q: *const f32, kbuf: *const c_void, vbuf: *const c_void,
        sel: *const u32, count: *const u32, out: *mut f32, rows: i32, n_head: i32, n_kv: i32,
        cap: i32, ncap: i32, scale: f32);
    loken_hc_norm(r: *const f32, gamma: *const f32, xn: *mut f32, groups: i32, s: i32, d: i32,
        eps: f32);
    loken_hc_low(down: *const f32, low: *mut f32, n: i32, s: i32);
    loken_hc_mean_gate(xn: *const f32, up: *const f32, mixed: *mut c_void, rows: i32, s: i32,
        d: i32, out_f16: i32);
    loken_hc_down_inject(xn: *const f32, down: *const c_void, inject: *const c_void, low: *mut f32,
        inj: *mut f32, rows: i32, k: i32, r: i32, i: i32, s: i32, half_w: i32,
        down_scale: *const f32, inject_scale: *const f32);
    loken_hc_up_mean(xn: *const f32, up: *const c_void, low: *const f32, mixed: *mut c_void,
        rows: i32, s: i32, d: i32, r: i32, out_f16: i32, half_w: i32, up_scale: *const f32);
    loken_hc_dequant(q: *const c_void, scale: *const f32, out: *mut c_void, n: i32, k: i32, out_f32: i32);
    loken_hc_combine(r: *const f32, out: *const c_void, inject: *const f32, r_new: *mut f32,
        xn: *mut f32, gamma: *const f32, groups: i32, s: i32, d: i32, eps: f32,
        out_f16: i32);
    loken_deltanet_gate(
        alpha: *const f32, beta_in: *const f32, a_log: *const f32, dt_bias: *const f32,
        g_pre: *mut f32, beta: *mut f32, n: i32, h: i32);
    loken_l2norm_gqa(
        x: *const f32, y: *mut f32, seq: i32, kg: i32, kd: i32, rep: i32, tiled: i32,
        eps: f32);
    loken_head_rmsnorm(x: *const f32, w: *const f32, y: *mut f32, n: i32, d: i32, eps: f32);
    loken_gate_gemv_f32(xs: *const c_void, gate_w: *const c_void,
        logits: *mut c_void, hidden: i32, n_rows: i32, n_experts: i32);
    loken_gate_topk_softmax(xs: *const c_void, gate_w: *const c_void,
        weights: *mut c_void, ids: *mut c_void, hidden: i32, n_rows: i32,
        n_experts: i32, n_expert_used: i32, with_norm: i32);
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
    loken_moe_gemm_gguf_gate_up_silu_mul_two(
        inputs: *const f32, gate_weights: *const c_void,
        up_weights: *const c_void, alt_gate: *const c_void, alt_up: *const c_void,
        alt_from: i32, sorted_token_ids: *const i32,
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
    loken_tier_plan(
        ids: *const u32, weights: *const f32, x: *const f32, table: *const i32,
        counter: *mut u32, slot_out: *mut u32, weight_out: *mut f32, mail_ids: *mut u32,
        mail_weights: *mut f32, mail_slots: *mut u32, mail_x: *mut f32, mail_rows: *mut u32,
        mail_request: *mut u32, rows: i32, k: i32, hidden: i32);
    loken_tier_wait(counter: *const u32, reply: *mut u32, abort_word: *mut u32);
    loken_tier_merge(card: *const f32, product: *const f32, counter: *const u32, out: *mut f32, n: i32);
}
/// The launchers of a cached layer's decode recorded in a graph (see `expert_tier::graph`).
///
/// # Safety
/// Every pointer is a card's address, or a page-locked mapping's, of the extent the kernel reads
/// and writes; the stream is the card's.
#[allow(clippy::too_many_arguments)]
pub(crate) mod tier_graph {
    pub unsafe fn plan(
        ids: *const u32,
        weights: *const f32,
        x: *const f32,
        table: *const i32,
        counter: *mut u32,
        slot_out: *mut u32,
        weight_out: *mut f32,
        mail_ids: *mut u32,
        mail_weights: *mut f32,
        mail_slots: *mut u32,
        mail_x: *mut f32,
        mail_rows: *mut u32,
        mail_request: *mut u32,
        rows: i32,
        k: i32,
        hidden: i32,
        stream: i64,
    ) {
        super::loken_tier_plan(
            ids,
            weights,
            x,
            table,
            counter,
            slot_out,
            weight_out,
            mail_ids,
            mail_weights,
            mail_slots,
            mail_x,
            mail_rows,
            mail_request,
            rows,
            k,
            hidden,
            stream,
        );
    }
    /// Most routed pairs one plan launch serves.
    pub const PLAN_PAIRS: usize = 256;
    pub unsafe fn wait(counter: *const u32, reply: *mut u32, abort_word: *mut u32, stream: i64) {
        super::loken_tier_wait(counter, reply, abort_word, stream);
    }
    pub unsafe fn merge(
        card: *const f32,
        product: *const f32,
        counter: *const u32,
        out: *mut f32,
        n: i32,
        stream: i64,
    ) {
        super::loken_tier_merge(card, product, counter, out, n, stream);
    }
}
cuda_launchers! {
    loken_cast_init_f32_from_dtype(
        dst: *mut f32, src: *const c_void, n: i32, dtype: i32);
    loken_moe_gemm_gguf_down_reduce_two(
        input: *const f32, weights: *const c_void, alt_weights: *const c_void, alt_from: i32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32, topk_weights: *const f32, output: *mut f32, num_experts: i32,
        topk: i32, size_m: i32, size_n: i32, size_k: i32, gguf_dtype: i32,
        down_bias: *const f32, add_after: *const f32);
    loken_moe_gemm_gguf_down_reduce(
        input: *const f32, weights: *const c_void, sorted_token_ids: *const i32,
        expert_ids: *const i32, topk_weights: *const f32, output: *mut f32, num_experts: i32,
        topk: i32, size_m: i32, size_n: i32, size_k: i32, gguf_dtype: i32,
        down_bias: *const f32, add_after: *const f32);
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
