//! Causal GQA prefill attention over an F16 KV history (`cuda/flash_prefill_gqa.cu`).

use crate::tensor::{DType, Result, Tensor};
use core::ffi::c_void;

extern "C" {
    fn loken_flash_prefill_gqa_f16kv(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut c_void,
        n_head: i32,
        n_kv: i32,
        seq_q: i32,
        seq_kv: i32,
        head_dim: i32,
        kv_tok_stride: i64,
        kv_head_stride: i64,
        scale: f32,
        window: i32,
        stream: i64,
    ) -> i32;
}

/// Head sizes `flash_prefill_gqa` has a kernel for.
pub const FLASH_PREFILL_GQA_HEAD_DIMS: [usize; 3] = [64, 128, 256];

/// Elements per 16-byte `cp.async` run of F16: every K/V row start must be a multiple.
const F16_PER_CP_ASYNC: usize = 16 / std::mem::size_of::<half::f16>();

/// Causal GQA prefill attention over an F16 KV history.
///
/// `q` is `[1, n_head, seq_q, hd]` (any dtype, cast to contiguous F32). `k` and `v` are F16
/// `[1, n_kv, seq_kv, hd]` views sharing one layout whose last axis is contiguous; the head
/// and token strides are read from it, so a permuted `[seq, n_kv, hd]` store needs no copy.
/// Query token `i` sits at position `i + seq_kv - seq_q`; `window > 0` also hides keys more
/// than `window` positions behind it. Returns `[1, seq_q, n_head * hd]` F32, the layout the
/// output projection reads, or `None` when the shape, dtype or device has no kernel.
pub fn flash_prefill_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    window: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    if !q.device().is_cuda() || k.dtype() != DType::F16 || v.dtype() != DType::F16 {
        return Ok(None);
    }
    let (b, n_head, seq_q, hd) = q.dims4()?;
    let (bk, n_kv, seq_kv, hdk) = k.dims4()?;
    if b != 1
        || bk != 1
        || hd != hdk
        || v.dims() != k.dims()
        || !FLASH_PREFILL_GQA_HEAD_DIMS.contains(&hd)
        || n_kv == 0
        || !n_head.is_multiple_of(n_kv)
        || seq_q == 0
        || seq_kv < seq_q
        || i32::try_from(seq_kv).is_err()
    {
        return Ok(None);
    }
    let q = q.to_dtype(DType::F32)?.contiguous()?;
    let out = unsafe { Tensor::empty((1, seq_q, n_head * hd), DType::F32, &q.device())? };
    let dev = q.device().as_cuda_device()?;
    // The launcher uses the runtime API, which launches on the thread's current device:
    // make it the one owning the stream (tensor-parallel ranks alternate devices).
    dev.cuda_stream()
        .context()
        .bind_to_thread()
        .map_err(|e| crate::tensor::Error::msg(format!("flash_prefill_gqa bind: {e}")))?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    let (kst, vst) = (kl.stride(), vl.stride());
    if kst != vst || kst[3] != 1 {
        return Ok(None);
    }
    let (head_stride, tok_stride) = (kst[1], kst[2]);
    if [
        head_stride,
        tok_stride,
        kl.start_offset(),
        vl.start_offset(),
    ]
    .iter()
    .any(|s| s % F16_PER_CP_ASYNC != 0)
    {
        return Ok(None);
    }
    let (C(qc), C(kc), C(vc), C(oc)) = (&*qs, &*ks, &*vs, &*os) else {
        return Ok(None);
    };
    let qsl = qc.as_cuda_slice::<f32>()?;
    let ksl = kc.as_cuda_slice::<half::f16>()?;
    let vsl = vc.as_cuda_slice::<half::f16>()?;
    let osl = oc.as_cuda_slice::<f32>()?;
    let qp = (qsl.device_ptr(qsl.stream()).0 + (ql.start_offset() * 4) as u64) as *const c_void;
    let kp = (ksl.device_ptr(ksl.stream()).0 + (kl.start_offset() * 2) as u64) as *const c_void;
    let vp = (vsl.device_ptr(vsl.stream()).0 + (vl.start_offset() * 2) as u64) as *const c_void;
    let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
    let rc = unsafe {
        loken_flash_prefill_gqa_f16kv(
            qp,
            kp,
            vp,
            op,
            n_head as i32,
            n_kv as i32,
            seq_q as i32,
            seq_kv as i32,
            hd as i32,
            tok_stride as i64,
            head_stride as i64,
            scale,
            i32::try_from(window).unwrap_or(i32::MAX),
            stream,
        )
    };
    drop((qs, ks, vs, os));
    match rc {
        0 => Ok(Some(out)),
        // Unsupported head size or a device below sm_80: the caller keeps its own path.
        1 | 2 => Ok(None),
        e => Err(crate::tensor::Error::msg(format!(
            "flash_prefill_gqa launch failed ({e})"
        ))),
    }
}
