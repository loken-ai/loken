//! Causal GQA prefill attention over an F16 KV history (`cuda/flash_prefill_gqa.cu`).

use crate::tensor::{DType, Result, Tensor};
use core::ffi::c_void;

crate::tensor::gpu_runtime::aot_kernels! {
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
        fn loken_flash_prefill_gqa_parts_f16kv(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            o: *mut c_void,
            ml: *mut f32,
            n_head: i32,
            n_kv: i32,
            seq_q: i32,
            seq_kv: i32,
            head_dim: i32,
            kv_tok_stride: i64,
            kv_head_stride: i64,
            scale: f32,
            pos_off: i32,
            stream: i64,
        ) -> i32;
        fn loken_flash_prefill_gqa_carry_f16kv(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            o: *mut c_void,
            c: *mut f32,
            n_head: i32,
            n_kv: i32,
            seq_q: i32,
            seq_kv: i32,
            head_dim: i32,
            kv_tok_stride: i64,
            kv_head_stride: i64,
            scale: f32,
            pos_off: i32,
            carry: i32,
            stream: i64,
        ) -> i32;
        fn loken_flash_prefill_gqa_keys_per_tile() -> i32;
        fn loken_flash_prefill_gqa_carry_floats(head_dim: i32) -> i32;
        fn loken_flash_prefill_gqa_varlen_f16kv(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            o: *mut c_void,
            starts: *const i32,
            lens: *const i32,
            n_seq: i32,
            n_head: i32,
            n_kv: i32,
            head_dim: i32,
            q_head_stride: i64,
            kv_tok_stride: i64,
            kv_head_stride: i64,
            scale: f32,
            stream: i64,
        ) -> i32;
        fn loken_flash_prefill_gqa_varlen_f16kv_f16out(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            o: *mut c_void,
            starts: *const i32,
            lens: *const i32,
            n_seq: i32,
            n_head: i32,
            n_kv: i32,
            head_dim: i32,
            q_head_stride: i64,
            kv_tok_stride: i64,
            kv_head_stride: i64,
            scale: f32,
            stream: i64,
        ) -> i32;
    }
}

/// Head sizes `flash_prefill_gqa` has a kernel for.
pub const FLASH_PREFILL_GQA_HEAD_DIMS: [usize; 3] = [64, 128, 256];

/// Whether the flash prefill kernels run on this device: an NVIDIA card, or a card with bf16
/// matrix instructions.
pub fn flash_prefill_gqa_card(dev: &crate::tensor::Device) -> Result<bool> {
    if !dev.is_cuda() {
        return Ok(false);
    }
    Ok(dev.as_cuda_device()?.has_bf16_matrix_cores())
}

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
    Ok(run(q, k, v, scale, window, None)?.map(|(out, _)| out))
}

/// The attention of `q` `[1, n_head, seq_q, hd]` over `k`, `v` as [`flash_prefill_gqa`] reads
/// them, the keys' positions starting `pos_off` before the first query token's (token i attends
/// key j iff `j <= i + pos_off`), left for a caller to merge with other ranges: each row's
/// maximum score and sum of exponentials, `[1, n_head, seq_q, 1]`, and its weighted values
/// `[1, n_head, seq_q, hd]`, all F32. A row no key reached has a maximum of minus infinity.
pub fn flash_prefill_gqa_parts(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    pos_off: i64,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let Ok(pos_off) = i32::try_from(pos_off) else {
        return Ok(None);
    };
    let Some((out, Some(ml))) = run(q, k, v, scale, 0, Some(pos_off))? else {
        return Ok(None);
    };
    let (_, n_head, seq_q, hd) = q.dims4()?;
    let heads_first = |t: Tensor, w: usize| -> Result<Tensor> {
        t.reshape((1, seq_q, n_head, w))?
            .transpose(1, 2)?
            .contiguous()
    };
    let ml = heads_first(ml, 2)?;
    let (max, sum) = (ml.narrow(3, 0, 1)?, ml.narrow(3, 1, 1)?);
    let out = heads_first(out, hd)?.broadcast_mul(&sum)?;
    Ok(Some((max, sum, out)))
}

/// Keys the flash prefill kernel stages per tile.
/// A walk of [`flash_prefill_gqa_carry`] calls cuts the history at multiples of it.
pub fn flash_prefill_gqa_keys_per_tile() -> usize {
    let keys = unsafe { loken_flash_prefill_gqa_keys_per_tile() };
    usize::try_from(keys).unwrap_or(1).max(1)
}

/// F32 values one query row of one head carries between calls of [`flash_prefill_gqa_carry`].
pub fn flash_prefill_gqa_carry_floats(head_dim: usize) -> usize {
    let floats = unsafe { loken_flash_prefill_gqa_carry_floats(head_dim as i32) };
    usize::try_from(floats).unwrap_or(0)
}

/// One call of a walk over a history in ranges of keys, in order.
///
/// `q` is contiguous F32 `[1, n_head, seq_q, hd]`.
/// `k` and `v` hold the range's keys as [`flash_prefill_gqa`] reads them.
/// Token i attends key j of the range iff `j <= i + pos_off`.
/// `carry` is F32 with [`flash_prefill_gqa_carry_floats`] values per row and head.
/// With `resume` the rows start from the state in `carry`, else afresh.
/// With `out` the call is the last: it writes the output `[1, seq_q, n_head * hd]` F32 there.
/// Without it the call leaves the rows' state in `carry`.
/// Every call but the last covers a multiple of [`flash_prefill_gqa_keys_per_tile`] keys.
/// The walk then gives the bits one [`flash_prefill_gqa`] call over the whole history gives.
/// False when the shape, dtype or device has no kernel.
#[allow(clippy::too_many_arguments)]
pub fn flash_prefill_gqa_carry(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    pos_off: i64,
    carry: &Tensor,
    out: Option<&Tensor>,
    resume: bool,
) -> Result<bool> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    let Ok(pos_off) = i32::try_from(pos_off) else {
        return Ok(false);
    };
    if !q.device().is_cuda()
        || !(q.device().is_nvidia() || q.device().as_cuda_device()?.has_bf16_matrix_cores())
        || q.dtype() != DType::F32
        || !q.is_contiguous()
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || carry.dtype() != DType::F32
        || !carry.is_contiguous()
    {
        return Ok(false);
    }
    let (b, n_head, seq_q, hd) = q.dims4()?;
    let (bk, n_kv, seq_kv, hdk) = k.dims4()?;
    let rows = seq_q * n_head;
    if b != 1
        || bk != 1
        || hd != hdk
        || v.dims() != k.dims()
        || !FLASH_PREFILL_GQA_HEAD_DIMS.contains(&hd)
        || n_kv == 0
        || !n_head.is_multiple_of(n_kv)
        || seq_q == 0
        || seq_kv == 0
        || i32::try_from(seq_kv).is_err()
        || carry.elem_count() != rows * flash_prefill_gqa_carry_floats(hd)
        || out.is_some_and(|o| {
            o.dtype() != DType::F32 || !o.is_contiguous() || o.elem_count() != rows * hd
        })
    {
        return Ok(false);
    }
    let dev = q.device().as_cuda_device()?;
    dev.cuda_stream()
        .context()
        .bind_to_thread()
        .map_err(|e| crate::tensor::Error::msg(format!("flash_prefill_gqa bind: {e}")))?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (cs, cl) = carry.storage_and_layout();
    let (kst, vst) = (kl.stride(), vl.stride());
    if kst != vst || kst[3] != 1 {
        return Ok(false);
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
        return Ok(false);
    }
    let (C(qc), C(kc), C(vc), C(cc)) = (&*qs, &*ks, &*vs, &*cs) else {
        return Ok(false);
    };
    let f32_ptr = |c: &crate::tensor::cuda_ext::CudaStorage, offset: usize| -> Result<u64> {
        let s = c.as_cuda_slice::<f32>()?;
        Ok(s.device_ptr(s.stream()).0 + (offset * 4) as u64)
    };
    let ksl = kc.as_cuda_slice::<half::f16>()?;
    let vsl = vc.as_cuda_slice::<half::f16>()?;
    let qp = f32_ptr(qc, ql.start_offset())? as *const c_void;
    let kp = (ksl.device_ptr(ksl.stream()).0 + (kl.start_offset() * 2) as u64) as *const c_void;
    let vp = (vsl.device_ptr(vsl.stream()).0 + (vl.start_offset() * 2) as u64) as *const c_void;
    let cp = f32_ptr(cc, cl.start_offset())? as *mut f32;
    // A call that writes no output is handed the carry as its output pointer, which it never
    // writes through.
    let out_storage = out.map(|o| o.storage_and_layout());
    let op = match &out_storage {
        Some((os, ol)) => {
            let C(oc) = &**os else {
                return Ok(false);
            };
            f32_ptr(oc, ol.start_offset())? as *mut c_void
        }
        None => cp as *mut c_void,
    };
    let mode = i32::from(resume) | (i32::from(out.is_none()) << 1);
    let rc = unsafe {
        loken_flash_prefill_gqa_carry_f16kv(
            qp,
            kp,
            vp,
            op,
            cp,
            n_head as i32,
            n_kv as i32,
            seq_q as i32,
            seq_kv as i32,
            hd as i32,
            tok_stride as i64,
            head_stride as i64,
            scale,
            pos_off,
            mode,
            stream,
        )
    };
    drop((qs, ks, vs, cs, out_storage));
    match rc {
        0 => Ok(true),
        // Unsupported head size or device: the caller keeps its own path.
        1 | 2 => Ok(false),
        e => Err(crate::tensor::Error::msg(format!(
            "flash_prefill_gqa launch failed ({e})"
        ))),
    }
}

/// Causal GQA prefill attention of several sequences laid end to end, each over its own tokens
/// only: sequence `i` holds `lens[i]` tokens after the tokens of the ones before it. `q` is
/// `[tokens, n_head, hd]` in any dtype, `k` and `v` contiguous F16 `[tokens, n_kv, hd]`.
/// Returns `[tokens, n_head * hd]` F32, the layout the output projection reads, or `None` when
/// the shape, dtype or device has no kernel.
pub fn flash_prefill_gqa_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lens: &[usize],
    scale: f32,
) -> Result<Option<Tensor>> {
    if !q.device().is_cuda() {
        return Ok(None);
    }
    let (t, n_head, hd) = q.dims3()?;
    if !varlen_fits(q, k, v, lens, t, n_head, hd)? {
        return Ok(None);
    }
    // Heads first, so each sequence's queries of one head are one run.
    let qh = q.to_dtype(DType::F32)?.transpose(0, 1)?.contiguous()?;
    varlen_launch(&qh, k, v, lens, scale, false)
}

/// [`flash_prefill_gqa_varlen`] over a `q` already in the kernel's layout, F32
/// `[n_head, tokens, hd]` contiguous, with the result stored in F16 `[tokens, n_head * hd]`:
/// each value is the F32 result rounded to nearest even, as a cast after the F32 call gives it.
pub fn flash_prefill_gqa_varlen_head_major(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lens: &[usize],
    scale: f32,
) -> Result<Option<Tensor>> {
    if !q.device().is_cuda() || q.dtype() != DType::F32 || !q.is_contiguous() {
        return Ok(None);
    }
    let (n_head, t, hd) = q.dims3()?;
    if !varlen_fits(q, k, v, lens, t, n_head, hd)? {
        return Ok(None);
    }
    varlen_launch(q, k, v, lens, scale, true)
}

/// Whether the varlen kernel takes these shapes: `q` holds `t` tokens of `n_head` heads of `hd`
/// values, `k` and `v` are contiguous F16 `[t, n_kv, hd]` from an offset the 16-byte loads
/// keep aligned, and `lens` split `t` into non-empty runs.
fn varlen_fits(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lens: &[usize],
    t: usize,
    n_head: usize,
    hd: usize,
) -> Result<bool> {
    if !flash_prefill_gqa_card(&q.device())? || k.dtype() != DType::F16 || v.dtype() != DType::F16 {
        return Ok(false);
    }
    let (tk, n_kv, hdk) = k.dims3()?;
    let feat = n_kv * hd;
    if tk != t
        || hd != hdk
        || v.dims() != k.dims()
        || !FLASH_PREFILL_GQA_HEAD_DIMS.contains(&hd)
        || n_kv == 0
        || !n_head.is_multiple_of(n_kv)
        || lens.contains(&0)
        || lens.iter().sum::<usize>() != t
        || i32::try_from(t).is_err()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !feat.is_multiple_of(F16_PER_CP_ASYNC)
    {
        return Ok(false);
    }
    let (_, kl) = k.storage_and_layout();
    let (_, vl) = v.storage_and_layout();
    Ok([kl.start_offset(), vl.start_offset()]
        .iter()
        .all(|s| s % F16_PER_CP_ASYNC == 0))
}

/// The varlen launch over a `qh` F32 `[n_head, t, hd]` that [`varlen_fits`] accepted, with the
/// output in F32 or, with `out_f16`, in F16. `None` for an unsupported head size or a device
/// below sm_80.
fn varlen_launch(
    qh: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lens: &[usize],
    scale: f32,
    out_f16: bool,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    let (n_head, t, hd) = qh.dims3()?;
    let (_, n_kv, _) = k.dims3()?;
    let feat = n_kv * hd;
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let out_dtype = if out_f16 { DType::F16 } else { DType::F32 };
    let out = unsafe { Tensor::empty((t, n_head * hd), out_dtype, &qh.device())? };
    let dev = qh.device().as_cuda_device()?;
    dev.cuda_stream()
        .context()
        .bind_to_thread()
        .map_err(|e| crate::tensor::Error::msg(format!("flash_prefill_gqa bind: {e}")))?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (qs, ql) = qh.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    let (C(qc), C(kc), C(vc), C(oc)) = (&*qs, &*ks, &*vs, &*os) else {
        return Ok(None);
    };
    let qsl = qc.as_cuda_slice::<f32>()?;
    let ksl = kc.as_cuda_slice::<half::f16>()?;
    let vsl = vc.as_cuda_slice::<half::f16>()?;
    let op = if out_f16 {
        let osl = oc.as_cuda_slice::<half::f16>()?;
        osl.device_ptr(osl.stream()).0 as *mut c_void
    } else {
        let osl = oc.as_cuda_slice::<f32>()?;
        osl.device_ptr(osl.stream()).0 as *mut c_void
    };
    let qp = (qsl.device_ptr(qsl.stream()).0 + (ql.start_offset() * 4) as u64) as *const c_void;
    let kp = (ksl.device_ptr(ksl.stream()).0 + (kl.start_offset() * 2) as u64) as *const c_void;
    let vp = (vsl.device_ptr(vsl.stream()).0 + (vl.start_offset() * 2) as u64) as *const c_void;
    let starts: Vec<i32> = lens
        .iter()
        .scan(0usize, |at, &n| {
            let start = *at;
            *at += n;
            Some(start as i32)
        })
        .collect();
    let lens32: Vec<i32> = lens.iter().map(|&n| n as i32).collect();
    let launch = if out_f16 {
        loken_flash_prefill_gqa_varlen_f16kv_f16out
    } else {
        loken_flash_prefill_gqa_varlen_f16kv
    };
    let rc = unsafe {
        launch(
            qp,
            kp,
            vp,
            op,
            starts.as_ptr(),
            lens32.as_ptr(),
            lens.len() as i32,
            n_head as i32,
            n_kv as i32,
            hd as i32,
            t as i64,
            feat as i64,
            hd as i64,
            scale,
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

/// The kernel's output `[1, seq_q, n_head * hd]`, and with `parts` (the keys' offset) each
/// row's maximum and sum `[seq_q * n_head * 2]`.
fn run(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    window: usize,
    parts: Option<i32>,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    // The kernel has an NVIDIA build and an AMD build for cards with matrix instructions; any
    // other card keeps the caller's path.
    if !q.device().is_cuda()
        || !(q.device().is_nvidia() || q.device().as_cuda_device()?.has_bf16_matrix_cores())
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
    {
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
        || (parts.is_none() && seq_kv < seq_q)
        || i32::try_from(seq_kv).is_err()
    {
        return Ok(None);
    }
    let q = q.to_dtype(DType::F32)?.contiguous()?;
    let out = unsafe { Tensor::empty((1, seq_q, n_head * hd), DType::F32, &q.device())? };
    let ml = match parts {
        Some(_) => Some(unsafe { Tensor::empty((seq_q * n_head * 2,), DType::F32, &q.device())? }),
        None => None,
    };
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
    let ml_storage = ml.as_ref().map(|t| t.storage_and_layout());
    let mlp = match &ml_storage {
        Some((ms, _)) => {
            let C(mc) = &**ms else {
                return Ok(None);
            };
            let msl = mc.as_cuda_slice::<f32>()?;
            msl.device_ptr(msl.stream()).0 as *mut f32
        }
        None => std::ptr::null_mut(),
    };
    let rc = unsafe {
        if let Some(pos_off) = parts {
            loken_flash_prefill_gqa_parts_f16kv(
                qp,
                kp,
                vp,
                op,
                mlp,
                n_head as i32,
                n_kv as i32,
                seq_q as i32,
                seq_kv as i32,
                hd as i32,
                tok_stride as i64,
                head_stride as i64,
                scale,
                pos_off,
                stream,
            )
        } else {
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
        }
    };
    drop((qs, ks, vs, os, ml_storage));
    match rc {
        0 => Ok(Some((out, ml))),
        // Unsupported head size or a device below sm_80: the caller keeps its own path.
        1 | 2 => Ok(None),
        e => Err(crate::tensor::Error::msg(format!(
            "flash_prefill_gqa launch failed ({e})"
        ))),
    }
}
