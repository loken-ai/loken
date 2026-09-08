//! Query-head-packed tensor-core flash-DECODE for GQA over a Q8_0 KV cache
//! (lever #3 of docs/RESEARCH_LEVERS_2026_06.md).
//!
//! Packs a GQA group's query heads (n_q_per_kv) AND the optional speculative
//! verify tokens (qlen) into the MMA M-dimension and streams each KV tile ONCE
//! through the tensor cores (nvcuda::wmma m16n8k16 HMMA on sm_120). The cuda-core
//! split-K path (`attn_flash_splitk_q8_gqa_decode_dev_pos`) reduces each QK dot
//! with a warp-shuffle PER query row PER token; here one HMMA produces the whole
//! [M,16]-token score tile and a second does P.V.
//!
//! The kernel lives in `cuda/flashdecode_tc/flashdecode_tc.cu`, compiled by
//! `build.rs::compile_flashdecode_tc_kernels` with real sm_120a SASS.
//!
//! Inputs match the Q8KvCache layout exactly: `k_q8`/`v_q8` are
//! `block_q8_0[t, n_kv, head_dim/32]`; `seq_kv_dev` is the device int with
//! count = `*seq_kv_dev + 1` (graph-replay-safe). Q rows are f16, flat
//! `[n_q_heads*qlen, head_dim]`; the kernel addresses each kv-head's packed rows
//! by the global row `h_kv*Mrows + r` and guards padding internally.
//!
//! Gated OFF by default (`LOKEN_FLASH_TC` env + a seq_kv-length floor); the
//! dispatch site in inference/generic_transformer/ falls back to split-K otherwise.

use crate::tensor::cuda_ext::{CudaStorage, RawCudaDevice as CudaDevice};
use anyhow::Result;
use core::ffi::c_void;
use cudarc::driver::{CudaSlice, DevicePtr};
use half::f16;

extern "C" {
    fn flashdecode_tc_hd128(
        k_blocks: *const c_void,
        v_blocks: *const c_void,
        q: *const c_void,
        partials: *mut f32,
        seq_kv_dev: *const i32,
        out: *mut c_void,
        n_kv: i32,
        n_q_per_kv: i32,
        qlen: i32,
        nsplit: i32,
        scale: f32,
        stream: i64,
    );
}

const MAX_NSPLIT: usize = 256;

/// Adaptive split count over the KV axis (mirrors the cuda-core split-K
/// launcher: keep each split's KV chunk small for memory-level parallelism, but
/// bounded so the per-call partial scratch is a stable size for mempool reuse).
fn pick_nsplit(seq_kv: usize) -> usize {
    seq_kv.div_ceil(24).clamp(8, MAX_NSPLIT).min(seq_kv.max(1))
}

/// Tensor-core flash-decode for one attention layer's Q8 KV cache.
///
/// `q_f16`: device f16 `[n_q_heads*qlen, head_dim]` row-major (the model's Q for
/// this layer; for qlen=1 it is the plain `[n_q_heads, head_dim]` decode Q).
/// Returns f16 `[n_q_heads*qlen, head_dim]` attention output (same row order).
pub fn flash_decode_tc_q8_gqa(
    k_q8: &CudaSlice<u8>,
    v_q8: &CudaSlice<u8>,
    q_f16: &CudaSlice<f16>,
    seq_kv_dev: &CudaSlice<i32>,
    head_dim: usize,
    n_kv_heads: usize,
    n_q_per_kv: usize,
    qlen: usize,
    seq_kv_hint: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if head_dim != 128 {
        anyhow::bail!("flash_decode_tc_q8_gqa: head_dim {head_dim} unsupported (only 128)");
    }
    let mrows = n_q_per_kv * qlen;
    if mrows == 0 || mrows > 16 {
        anyhow::bail!("flash_decode_tc_q8_gqa: Mrows {mrows} (n_q_per_kv*qlen) not in 1..=16");
    }
    let n_q_heads = n_kv_heads * n_q_per_kv;
    let total_rows = n_q_heads * qlen;
    if q_f16.len() != total_rows * head_dim {
        anyhow::bail!(
            "flash_decode_tc_q8_gqa: Q has {} elems, expected {} ({}*{})",
            q_f16.len(),
            total_rows * head_dim,
            total_rows,
            head_dim
        );
    }
    let nsplit = pick_nsplit(seq_kv_hint);
    debug_assert!(nsplit <= MAX_NSPLIT);

    // partials [total_rows, MAX_NSPLIT, head_dim+2] f32 (allocated at the CONSTANT
    // cap so the per-decode alloc size never changes as nsplit grows with seq_kv;
    // the kernel indexes only the live `nsplit` prefix).
    let partials = unsafe { dev.alloc::<f32>(total_rows * MAX_NSPLIT * (head_dim + 2))? };
    let out = unsafe { dev.alloc::<f16>(total_rows * head_dim)? };

    let stream = dev.cuda_stream().cu_stream() as i64;
    let kp = k_q8.device_ptr(k_q8.stream()).0 as *const c_void;
    let vp = v_q8.device_ptr(v_q8.stream()).0 as *const c_void;
    let qp = q_f16.device_ptr(q_f16.stream()).0 as *const c_void;
    let pp = partials.device_ptr(partials.stream()).0 as *mut f32;
    let sp = seq_kv_dev.device_ptr(seq_kv_dev.stream()).0 as *const i32;
    let op = out.device_ptr(out.stream()).0 as *mut c_void;

    unsafe {
        flashdecode_tc_hd128(
            kp,
            vp,
            qp,
            pp,
            sp,
            op,
            n_kv_heads as i32,
            n_q_per_kv as i32,
            qlen as i32,
            nsplit as i32,
            scale,
            stream,
        );
    }
    drop(partials);
    Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
}

/// The shortest KV this path is worth taking. Below it, split-K's wider grid fills the device
/// better than the tensor cores' deeper one; the crossover was measured at 512.
pub const TC_MIN_KV: usize = 512;
