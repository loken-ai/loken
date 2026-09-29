//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Device-to-device copy of ALL elements of `src` (F32) into `dst` (F32). Used as
/// the final op of the captured gpt-oss decode forward: the lm_head logits land in
/// a fresh arena tensor whose contents a captured graph won't keep stable across
/// replays; copying into a persistent `dst` INSIDE the captured region makes every
/// replay deposit the logits at the fixed address the host samples from.
pub fn copy_f32_dev(src: &Tensor, dst: &Tensor) -> Result<()> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    if !src.device().is_cuda() {
        return Ok(());
    }
    let src = src.flatten_all()?.contiguous()?;
    let n = src.elem_count() as i32;
    let dev = src.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (ss, sl) = src.storage_and_layout();
    let (ds, dl) = dst.storage_and_layout();
    if let (C(sc), C(dc)) = (&*ss, &*ds) {
        let ssl = sc.as_cuda_slice::<f32>()?;
        let dsl = dc.as_cuda_slice::<f32>()?;
        let sp = (ssl.device_ptr(ssl.stream()).0 + (sl.start_offset() * 4) as u64) as *const f32;
        let dp = (dsl.device_ptr(dsl.stream()).0 + (dl.start_offset() * 4) as u64) as *mut f32;
        unsafe {
            loken_copy_f32(sp, dp, n, stream);
        }
    }
    Ok(())
}

/// Device-kv_len flash-decode (seq=1, GQA): kv_len = *pos_dev + 1 read on device,
/// over the FULL ring buffer kbuf/vbuf [b,n_kv,kv_max,hd] - so a captured graph
/// attends the correct growing count on replay. q [b,n_head,hd] F16. No sinks/
/// mask. Returns [b,n_head,hd] F16, or None unless CUDA + F16 + hd ok.
pub fn flash_decode_devkvlen(
    q: &Tensor,
    kbuf: &Tensor,
    vbuf: &Tensor,
    pos_dev: &Tensor,
    sinks: Option<&Tensor>,
    window: usize,
    scale: f32,
    batch: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !q.device().is_cuda()
        || !head_dim.is_multiple_of(32)
        || !(32..=256).contains(&head_dim)
        || q.dtype() != DType::F16
        || kbuf.dtype() != DType::F16
        || vbuf.dtype() != DType::F16
    {
        return Ok(None);
    }
    // Per-head sink logits (gpt-oss); must be contiguous F32 [n_head]. Held alive
    // for the launch. lfm2 passes None -> null (empty-seeded softmax).
    let sinks_c = match sinks {
        Some(s) => Some(s.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };
    let q = q.contiguous()?;
    let kst_lay = kbuf.layout();
    let kst = kst_lay.stride();
    let vst_lay = vbuf.layout();
    let vst = vst_lay.stride();
    if kst[3] != 1 || vst[3] != 1 {
        return Ok(None);
    }
    let out = unsafe { Tensor::empty((batch, n_head, head_dim), DType::F16, &q.device())? };
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = kbuf.storage_and_layout();
    let (vs, vl) = vbuf.storage_and_layout();
    let (ps, pl) = pos_dev.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    let sink_sl = sinks_c.as_ref().map(|s| s.storage_and_layout());
    if let (C(qc), C(kc), C(vc), C(pc), C(oc)) = (&*qs, &*ks, &*vs, &*ps, &*os) {
        let qsl = qc.as_cuda_slice::<half::f16>()?;
        let ksl = kc.as_cuda_slice::<half::f16>()?;
        let vsl = vc.as_cuda_slice::<half::f16>()?;
        let psl = pc.as_cuda_slice::<i32>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let qp = (qsl.device_ptr(qsl.stream()).0 + (ql.start_offset() * 2) as u64) as *const c_void;
        let kp = (ksl.device_ptr(ksl.stream()).0 + (kl.start_offset() * 2) as u64) as *const c_void;
        let vp = (vsl.device_ptr(vsl.stream()).0 + (vl.start_offset() * 2) as u64) as *const c_void;
        let pp = (psl.device_ptr(psl.stream()).0 + (pl.start_offset() * 4) as u64) as *const i32;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        let sp: *const f32 = match &sink_sl {
            Some((sg, sl)) => match &**sg {
                C(sc) => {
                    let ssl = sc.as_cuda_slice::<f32>()?;
                    (ssl.device_ptr(ssl.stream()).0 + (sl.start_offset() * 4) as u64) as *const f32
                }
                _ => std::ptr::null(),
            },
            None => std::ptr::null(),
        };
        unsafe {
            loken_flash_decode_devkvlen_f16(
                qp,
                kp,
                vp,
                pp,
                op,
                sp,
                batch as i32,
                n_head as i32,
                n_kv as i32,
                head_dim as i32,
                scale,
                window as i32,
                kst[0] as i64,
                kst[1] as i64,
                kst[2] as i64,
                vst[0] as i64,
                vst[1] as i64,
                vst[2] as i64,
                stream,
            );
        }
    }
    drop(qs);
    drop(ks);
    drop(vs);
    drop(ps);
    drop(os);
    Ok(Some(out))
}

/// Split-K flash decode (seq 1, GQA) over whole KV buffers `kbuf`/`vbuf` `[b, n_kv, cap,
/// hd]` F16 (hd contiguous): the key count is `*pos_dev + 1` read on the device when
/// `pos_dev` (`[1]` i32) is given, `kv_len` otherwise, so a captured step and an eager one
/// run the same launch. `window` > 0 keeps the last `window` keys; `sinks` `[n_head]`
/// joins each head's denominator. `nsplit` fixes the partial count and must not change
/// under a recorded graph. Returns `[b, n_head, hd]` F16, or `None` off a card or for a
/// head size the kernel does not serve.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_split_devkvlen(
    q: &Tensor,
    kbuf: &Tensor,
    vbuf: &Tensor,
    pos_dev: Option<&Tensor>,
    kv_len: usize,
    sinks: Option<&Tensor>,
    window: usize,
    scale: f32,
    nsplit: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !q.device().is_cuda()
        || !head_dim.is_multiple_of(32)
        || !(32..=256).contains(&head_dim)
        || q.dtype() != DType::F16
        || kbuf.dtype() != DType::F16
        || vbuf.dtype() != DType::F16
        || nsplit == 0
    {
        return Ok(None);
    }
    let batch = q.elem_count() / (n_head * head_dim);
    let q = q.contiguous()?;
    let kst_lay = kbuf.layout();
    let kst = kst_lay.stride();
    let vst_lay = vbuf.layout();
    let vst = vst_lay.stride();
    if kst.len() != 4 || vst.len() != 4 || kst[3] != 1 || vst[3] != 1 {
        return Ok(None);
    }
    let sinks_c = match sinks {
        Some(s) => Some(s.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };
    let dev = q.device();
    let out = unsafe { Tensor::empty((batch, n_head, head_dim), DType::F16, &dev)? };
    let parts = batch * n_head * nsplit;
    let part_m = unsafe { Tensor::empty(parts, DType::F32, &dev)? };
    let part_l = unsafe { Tensor::empty(parts, DType::F32, &dev)? };
    let part_acc = unsafe { Tensor::empty(parts * head_dim, DType::F32, &dev)? };
    let stream = dev.as_cuda_device()?.cuda_stream().cu_stream() as i64;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = kbuf.storage_and_layout();
    let (vs, vl) = vbuf.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    let (pms, _) = part_m.storage_and_layout();
    let (pls, _) = part_l.storage_and_layout();
    let (pas, _) = part_acc.storage_and_layout();
    let pos_sl = pos_dev.map(|p| p.storage_and_layout());
    let sink_sl = sinks_c.as_ref().map(|s| s.storage_and_layout());
    let (C(qc), C(kc), C(vc), C(oc), C(pmc), C(plc), C(pac)) =
        (&*qs, &*ks, &*vs, &*os, &*pms, &*pls, &*pas)
    else {
        return Ok(None);
    };
    let f16p = |c: &crate::tensor::kernel_ffi::CudaStorage, off: usize| -> Result<u64> {
        let s = c.as_cuda_slice::<half::f16>()?;
        Ok(s.device_ptr(s.stream()).0 + (off * 2) as u64)
    };
    let f32p = |c: &crate::tensor::kernel_ffi::CudaStorage, off: usize| -> Result<u64> {
        let s = c.as_cuda_slice::<f32>()?;
        Ok(s.device_ptr(s.stream()).0 + (off * 4) as u64)
    };
    let pp: *const i32 = match &pos_sl {
        Some((ps, pl)) => match &**ps {
            C(pc) => {
                let s = pc.as_cuda_slice::<i32>()?;
                (s.device_ptr(s.stream()).0 + (pl.start_offset() * 4) as u64) as *const i32
            }
            _ => return Ok(None),
        },
        None => std::ptr::null(),
    };
    let sp: *const f32 = match &sink_sl {
        Some((sg, sl)) => match &**sg {
            C(sc) => f32p(sc, sl.start_offset())? as *const f32,
            _ => std::ptr::null(),
        },
        None => std::ptr::null(),
    };
    unsafe {
        loken_flash_decode_split_devkvlen_f16(
            f16p(qc, ql.start_offset())? as *const c_void,
            f16p(kc, kl.start_offset())? as *const c_void,
            f16p(vc, vl.start_offset())? as *const c_void,
            pp,
            kv_len as i32,
            sp,
            f16p(oc, 0)? as *mut c_void,
            f32p(pmc, 0)? as *mut f32,
            f32p(plc, 0)? as *mut f32,
            f32p(pac, 0)? as *mut f32,
            batch as i32,
            n_head as i32,
            n_kv as i32,
            nsplit as i32,
            head_dim as i32,
            scale,
            window as i32,
            kst[0] as i64,
            kst[1] as i64,
            kst[2] as i64,
            vst[0] as i64,
            vst[1] as i64,
            vst[2] as i64,
            stream,
        );
    }
    drop((qs, ks, vs, os, pms, pls, pas, pos_sl, sink_sl));
    Ok(Some(out))
}
