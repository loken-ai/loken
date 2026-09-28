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
