//! Custom fused CUDA kernels for inference acceleration.
//!
//! Reduces CUDA kernel launch count by fusing element-wise operations.
//! Kernels compiled at runtime via NVRTC (no nvcc/build.rs needed).

use crate::tensor::cuda_ext::PushKernelArg;
use crate::tensor::{DType, Result, Tensor};
use core::ffi::c_void;
use std::sync::OnceLock;

cuda_launchers! {
    loken_fused_rmsnorm_f16(
        x: *const c_void, res: *const c_void, weight: *const f32,
        norm_out: *mut c_void, sum_out: *mut c_void, rows: i32,
        cols: i32, eps: f32);
    loken_fused_rmsnorm_f16_out_f32(
        x: *const c_void, weight: *const f32, norm_out: *mut c_void,
        rows: i32, cols: i32, eps: f32);
    loken_silu_mul_f16(
        g: *const c_void, u: *const c_void, out: *mut c_void,
        n: i64);
    loken_fused_shexp_out(
        routed: *const f32, down: *const c_void, gate_logit: *const f32,
        out: *mut f32, n_tokens: i32, hidden: i32);
    loken_relu2_f16(x: *const c_void, out: *mut c_void, n: i64);
    loken_softplus_bias(dt: *const f32, bias: *const f32, out: *mut f32, rows: i32, cols: i32);
    loken_add_to_f16(
        a: *const f32, b: *const c_void, out: *mut c_void, n: i64);
    loken_neox_rope_f16(x: *const c_void, cos: *const c_void,
        sin: *const c_void, out: *mut c_void, outer: i32, seq: i32,
        hd: i32, rope_dim: i32);
    loken_neox_rope_devpos_f16(x: *const c_void, cos_full: *const c_void,
        sin_full: *const c_void, pos_dev: *const i32, out: *mut c_void,
        outer: i32, seq: i32, hd: i32, rope_dim: i32);
    loken_paged_rope_f16(x: *const c_void, cos: *const c_void,
        sin: *const c_void, out: *mut c_void, outer: i32, heads: i32,
        hd: i32, rope_dim: i32, interleaved: i32);
    gptoss_flash_decode_f16(
        q: *const c_void, k: *const c_void, v: *const c_void,
        mask: *const f32, sinks: *const f32, out: *mut c_void, batch: i32,
        n_head: i32, n_kv: i32, kv_len: i32, head_dim: i32, scale: f32, k_batch_stride: i64,
        k_head_stride: i64, k_pos_stride: i64, v_batch_stride: i64, v_head_stride: i64,
        v_pos_stride: i64);
    gptoss_flash_decode_split_f16(
        q: *const c_void, k: *const c_void, v: *const c_void,
        mask: *const f32, sinks: *const f32, out: *mut c_void, part_m: *mut f32,
        part_l: *mut f32, part_acc: *mut f32, batch: i32, n_head: i32, n_kv: i32, kv_len: i32,
        nsplit: i32, head_dim: i32, scale: f32, k_batch_stride: i64, k_head_stride: i64,
        k_pos_stride: i64, v_batch_stride: i64, v_head_stride: i64, v_pos_stride: i64);
    // Multi-warp token-per-lane split-K decode (flash_decode_tiled_probe.cu).
    // Probed x1.39 vs the single-warp split-K on lfm2-class hd64 GQA at kv 2.5K
    // (a go/no-go probe); no batch/mask/sinks support - the Rust gate below
    // restricts it to that exact regime.
    tp_mw_flash_decode_split(
        q: *const c_void, k: *const c_void, v: *const c_void,
        part_m: *mut f32, part_l: *mut f32, part_acc: *mut f32, out: *mut c_void,
        n_head: i32, n_kv: i32, kv_len: i32, nsplit: i32, head_dim: i32, scale: f32,
        k_head_stride: i64, k_pos_stride: i64, v_head_stride: i64, v_pos_stride: i64);
    loken_flash_decode_f16(
        q: *const c_void, k: *const c_void, v: *const c_void,
        mask: *const f32, sinks: *const f32, out: *mut c_void, batch: i32,
        n_head: i32, n_kv: i32, kv_len: i32, head_dim: i32, scale: f32, k_batch_stride: i64,
        k_head_stride: i64, k_pos_stride: i64, v_batch_stride: i64, v_head_stride: i64,
        v_pos_stride: i64);
    loken_flash_decode_split_f16(
        q: *const c_void, k: *const c_void, v: *const c_void,
        mask: *const f32, sinks: *const f32, out: *mut c_void, part_m: *mut f32,
        part_l: *mut f32, part_acc: *mut f32, batch: i32, n_head: i32, n_kv: i32, kv_len: i32,
        nsplit: i32, head_dim: i32, scale: f32, k_batch_stride: i64, k_head_stride: i64,
        k_pos_stride: i64, v_batch_stride: i64, v_head_stride: i64, v_pos_stride: i64);
    loken_kv_write_at_pos_f16(k_new: *const c_void, v_new: *const c_void,
        k_buf: *mut c_void, v_buf: *mut c_void, pos_dev: *const i32,
        b: i32, n_kv: i32, seq: i32, hd: i32, kv_max: i32);
    loken_embed_gather_f16(
        table: *const c_void, tok: *const u32, out: *mut c_void,
        d_model: i32);
    loken_copy_u32(src: *const u32, dst: *mut u32);
    loken_copy_f32(src: *const f32, dst: *mut f32, n: i32);
    loken_set_i32(p: *mut i32, v: i32);
    #[link_name = "flash_dit_bf16_hd64"]
    flash_dit_bf16_hd64_raw(
        q: *const c_void, k: *const c_void, v: *const c_void,
        o: *mut c_void, batch: i32, n_head: i32, seq_q: i32, seq_kv: i32,
        scale: f32, frame_tokens: i32, grid_w: i32);
    #[link_name = "flash_dit_bf16_hd128"]
    flash_dit_bf16_hd128_raw(
        q: *const c_void, k: *const c_void, v: *const c_void,
        o: *mut c_void, batch: i32, n_head: i32, seq_q: i32, seq_kv: i32,
        scale: f32, frame_tokens: i32, grid_w: i32);
    #[link_name = "flash_prefill_f16"]
    flash_prefill_f16_raw(
        q: *const c_void, k: *const c_void, v: *const c_void,
        o: *mut c_void, batch: i32, n_head: i32, n_kv: i32, seq_q: i32, seq_kv: i32,
        head_dim: i32, scale: f32, window: i32);
    #[link_name = "flash_prefill_split_f16"]
    flash_prefill_split_f16_raw(
        q: *const c_void, k: *const c_void, v: *const c_void,
        partials: *mut f32, o: *mut c_void, batch: i32, n_head: i32, n_kv: i32,
        seq_q: i32, seq_kv: i32, head_dim: i32, nsplit: i32, scale: f32, window: i32);
    loken_flash_decode_devkvlen_f16(
        q: *const c_void, k: *const c_void, v: *const c_void,
        pos_dev: *const i32, out: *mut c_void, sinks: *const f32, batch: i32,
        n_head: i32, n_kv: i32, head_dim: i32, scale: f32, window: i32, k_batch_stride: i64,
        k_head_stride: i64, k_pos_stride: i64, v_batch_stride: i64, v_head_stride: i64,
        v_pos_stride: i64);
    loken_paged_flash_decode_f16(q: *const c_void, kp: *const c_void,
        vp: *const c_void, block_table: *const i32, seq_lens: *const i32,
        out: *mut c_void, batch: i32, n_head: i32, n_kv: i32, head_dim: i32,
        scale: f32, block_size: i32, max_blocks: i32, feat: i32);
    loken_paged_kv_write_f16(k_new: *const c_void, v_new: *const c_void,
        kp: *mut c_void, vp: *mut c_void, slot_dev: *const i32,
        batch: i32, feat: i32);
    loken_paged_flash_decode_split_f16(
        q: *const c_void, kp: *const c_void,
        vp: *const c_void, block_table: *const i32, seq_lens: *const i32,
        out: *mut c_void, part_m: *mut f32, part_l: *mut f32, part_acc: *mut f32,
        batch: i32, n_head: i32, n_kv: i32, nsplit: i32, head_dim: i32, scale: f32,
        block_size: i32, max_blocks: i32, feat: i32);
}

mod attention;
pub use attention::*;
mod buffers;
pub use buffers::*;

#[cfg(all(test, feature = "cuda"))]
mod cuda_numerics_tests;
#[cfg(all(test, feature = "cuda"))]
mod flash_prefill_parity;
