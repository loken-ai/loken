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
//! Selected for head_dim 128 on Ampere tensor cores once seq_kv reaches `TC_MIN_KV`; the
//! dispatch site in inference/generic_transformer/ falls back to split-K otherwise.

use crate::tensor::cuda_ext::{CudaStorage, RawCudaDevice as CudaDevice};
use crate::tensor::gpu_runtime::driver::{CudaSlice, DevicePtr};
use anyhow::Result;
use core::ffi::c_void;
use half::f16;

crate::tensor::gpu_runtime::aot_kernels! {
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
            grid_splits: i32,
            chunk: i32,
            lo: i32,
            hi: i32,
            scale: f32,
            stream: i64,
        );
    }
}

const MAX_NSPLIT: usize = 256;

/// Adaptive split count over the KV axis (mirrors the cuda-core split-K
/// launcher: keep each split's KV chunk small for memory-level parallelism, but
/// bounded so the per-call partial scratch is a stable size for mempool reuse).
fn pick_nsplit(seq_kv: usize) -> usize {
    seq_kv
        .div_ceil(SPLIT_CHUNK)
        .clamp(MIN_NSPLIT, MAX_NSPLIT)
        .min(seq_kv.max(1))
}

/// Keys per split.
const SPLIT_CHUNK: usize = 24;
const MIN_NSPLIT: usize = 8;

/// Tensor-core flash-decode for one attention layer's Q8 KV cache.
///
/// `q_f16`: device f16 `[n_q_heads*qlen, head_dim]` row-major (the model's Q for
/// this layer; for qlen=1 it is the plain `[n_q_heads, head_dim]` decode Q).
/// Returns f16 `[n_q_heads*qlen, head_dim]` attention output (same row order).
///
/// The kernels take their split count from the length they read on the device
/// (`pick_nsplit`), so an eager step and a graph replay at the same length split alike.
/// `seq_kv_hint` is the longest length the launch may serve: it sizes the grid.
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
    if !dev.has_f16_matrix_cores() {
        anyhow::bail!("flash_decode_tc_q8_gqa: the card has no f16 matrix instructions");
    }
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
            SPLIT_CHUNK as i32,
            MIN_NSPLIT as i32,
            MAX_NSPLIT as i32,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn pseudo(i: usize, seed: usize) -> f32 {
        let h = (i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 2001;
        h as f32 / 1000.0 - 1.0
    }

    /// The kernel's answer against a host reference on a card with f16 matrix instructions; a
    /// card without them declines, so its caller takes split-K.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn flashdecode_tc_matches_a_host_reference() {
        let dev = Device::new_cuda(0)
            .expect("a card for the card test")
            .as_cuda_device()
            .unwrap();
        let (hd, n_kv, n_q_per_kv, qlen, seq_kv) = (128usize, 2usize, 4usize, 2usize, 600usize);
        let hdb = hd / 32;
        let mrows = n_q_per_kv * qlen;
        let rows = n_kv * mrows;
        // block_q8_0 [t, n_kv, hd / 32]: an f16 scale and 32 signed bytes.
        let quantised = |seed: usize| -> (Vec<u8>, Vec<f32>) {
            let (mut bytes, mut values) = (Vec::new(), vec![0f32; seq_kv * n_kv * hd]);
            for blk in 0..seq_kv * n_kv * hdb {
                let d = f16::from_f32(0.01 + 0.005 * pseudo(blk, seed).abs());
                bytes.extend_from_slice(&d.to_bits().to_le_bytes());
                for e in 0..32 {
                    let qv = (pseudo(blk * 32 + e, seed + 1) * 127.0) as i8;
                    bytes.push(qv as u8);
                    values[blk * 32 + e] = qv as f32 * d.to_f32();
                }
            }
            (bytes, values)
        };
        let (kb, k) = quantised(2);
        let (vb, v) = quantised(4);
        let q: Vec<f16> = (0..rows * hd)
            .map(|i| f16::from_f32(pseudo(i, 6)))
            .collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let k_dev = dev.memcpy_stod(&kb).unwrap();
        let v_dev = dev.memcpy_stod(&vb).unwrap();
        let q_dev = dev.memcpy_stod(&q).unwrap();
        let pos_dev = dev.memcpy_stod(&[seq_kv as i32 - 1]).unwrap();
        let got = flash_decode_tc_q8_gqa(
            &k_dev, &v_dev, &q_dev, &pos_dev, hd, n_kv, n_q_per_kv, qlen, seq_kv, scale, &dev,
        );
        if !dev.has_f16_matrix_cores() {
            assert!(
                got.is_err(),
                "a card without f16 fragments declines the kernel"
            );
            return;
        }
        let got = dev
            .memcpy_dtov(got.unwrap().as_cuda_slice::<f16>().unwrap())
            .unwrap();
        let mut worst = 0f32;
        for g in 0..rows {
            let h = g / mrows;
            let at = |x: &[f32], t: usize, d: usize| x[(t * n_kv + h) * hd + d];
            let s: Vec<f64> = (0..seq_kv)
                .map(|t| {
                    (0..hd)
                        .map(|d| q[g * hd + d].to_f32() as f64 * at(&k, t, d) as f64)
                        .sum::<f64>()
                        * scale as f64
                })
                .collect();
            let m = s.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let total: f64 = e.iter().sum();
            for d in 0..hd {
                let want = (0..seq_kv).map(|t| e[t] * at(&v, t, d) as f64).sum::<f64>() / total;
                worst = worst.max((got[g * hd + d].to_f32() - want as f32).abs());
            }
        }
        assert!(worst < 2e-2, "differs by {worst}");
    }
}
