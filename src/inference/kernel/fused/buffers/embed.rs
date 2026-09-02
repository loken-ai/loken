//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Capture-safe embedding gather: out = table[*tok] reading the token id from a
/// device buffer at kernel-exec time (a tensor-level index_select bakes the index at
/// graph capture). table [vocab, d_model] F16, tok u32 [..1]. Returns [d_model]
/// F16, or None (caller falls back) unless CUDA + F16.
pub fn embed_gather_f16(table: &Tensor, tok: &Tensor, d_model: usize) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !table.device().is_cuda() || table.dtype() != DType::F16 {
        return Ok(None);
    }
    let table = table.contiguous()?;
    let tok = tok.flatten_all()?.contiguous()?;
    let out = unsafe { Tensor::empty((d_model,), DType::F16, &table.device())? };
    let dev = table.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (ts, tl) = table.storage_and_layout();
    let (ks, kl) = tok.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(tc), C(kc), C(oc)) = (&*ts, &*ks, &*os) {
        let tsl = tc.as_cuda_slice::<half::f16>()?;
        let ksl = kc.as_cuda_slice::<u32>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let tp = (tsl.device_ptr(tsl.stream()).0 + (tl.start_offset() * 2) as u64) as *const c_void;
        let kp = (ksl.device_ptr(ksl.stream()).0 + (kl.start_offset() * 4) as u64) as *const u32;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            loken_embed_gather_f16(tp, kp, op, d_model as i32, stream);
        }
    }
    drop(ts);
    drop(ks);
    drop(os);
    Ok(Some(out))
}

/// Write a host i32 into a device [1] i32 tensor IN PLACE (one tiny kernel; the
/// value is a kernel arg, so launching it OUTSIDE a captured graph region advances
/// the buffer between replays without baking the value into the graph).
pub fn set_i32_inplace(t: &Tensor, val: i32) -> Result<()> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    if !t.device().is_cuda() {
        return Ok(());
    }
    let dev = t.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (s, l) = t.storage_and_layout();
    if let C(c) = &*s {
        let sl = c.as_cuda_slice::<i32>()?;
        let p = (sl.device_ptr(sl.stream()).0 + (l.start_offset() * 4) as u64) as *mut i32;
        unsafe {
            loken_set_i32(p, val, stream);
        }
    }
    Ok(())
}

/// Device-to-device copy of one u32 (the new token id) from `src` into `dst`  -
/// no host roundtrip/sync. Both u32 device tensors (uses element 0 of each).
pub fn copy_u32_dev(src: &Tensor, dst: &Tensor) -> Result<()> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    if !src.device().is_cuda() {
        return Ok(());
    }
    let src = src.flatten_all()?.contiguous()?;
    let dev = src.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (ss, sl) = src.storage_and_layout();
    let (ds, dl) = dst.storage_and_layout();
    if let (C(sc), C(dc)) = (&*ss, &*ds) {
        let ssl = sc.as_cuda_slice::<u32>()?;
        let dsl = dc.as_cuda_slice::<u32>()?;
        let sp = (ssl.device_ptr(ssl.stream()).0 + (sl.start_offset() * 4) as u64) as *const u32;
        let dp = (dsl.device_ptr(dsl.stream()).0 + (dl.start_offset() * 4) as u64) as *mut u32;
        unsafe {
            loken_copy_u32(sp, dp, stream);
        }
    }
    Ok(())
}

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
#[allow(clippy::too_many_arguments)]
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
        || head_dim % 32 != 0
        || head_dim < 32
        || head_dim > 256
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

/// Device-position KV-cache write: scatter k_new/v_new [b,n_kv,hd] F16 into the
/// persistent ring buffers kbuf/vbuf [b,n_kv,kv_max,hd] F16 at slot `*pos_dev`
/// (i32 [1] on device), IN PLACE. The keystone for CUDA-graph decode - the write
/// position is device-resident so a captured graph advances it on replay. Errors
/// (returns Ok with no-op) off the CUDA F16 path; bit-exact (a plain copy).
pub fn kv_write_at_pos(
    k_new: &Tensor,
    v_new: &Tensor,
    kbuf: &Tensor,
    vbuf: &Tensor,
    pos_dev: &Tensor,
    kv_max: usize,
) -> Result<()> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !k_new.device().is_cuda() || k_new.dtype() != DType::F16 {
        return Ok(());
    }
    let (b, n_kv, seq, hd) = k_new.dims4()?; // [b, n_kv, seq, hd]
    let k_new = k_new.contiguous()?;
    let v_new = v_new.contiguous()?;
    let dev = k_new.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (kns, knl) = k_new.storage_and_layout();
    let (vns, vnl) = v_new.storage_and_layout();
    let (kbs, _) = kbuf.storage_and_layout();
    let (vbs, _) = vbuf.storage_and_layout();
    let (ps, pl) = pos_dev.storage_and_layout();
    if let (C(knc), C(vnc), C(kbc), C(vbc), C(pc)) = (&*kns, &*vns, &*kbs, &*vbs, &*ps) {
        let knsl = knc.as_cuda_slice::<half::f16>()?;
        let vnsl = vnc.as_cuda_slice::<half::f16>()?;
        let kbsl = kbc.as_cuda_slice::<half::f16>()?;
        let vbsl = vbc.as_cuda_slice::<half::f16>()?;
        let psl = pc.as_cuda_slice::<i32>()?;
        let knp =
            (knsl.device_ptr(knsl.stream()).0 + (knl.start_offset() * 2) as u64) as *const c_void;
        let vnp =
            (vnsl.device_ptr(vnsl.stream()).0 + (vnl.start_offset() * 2) as u64) as *const c_void;
        let kbp = kbsl.device_ptr(kbsl.stream()).0 as *mut c_void;
        let vbp = vbsl.device_ptr(vbsl.stream()).0 as *mut c_void;
        let pp = (psl.device_ptr(psl.stream()).0 + (pl.start_offset() * 4) as u64) as *const i32;
        unsafe {
            loken_kv_write_at_pos_f16(
                knp,
                vnp,
                kbp,
                vbp,
                pp,
                b as i32,
                n_kv as i32,
                seq as i32,
                hd as i32,
                kv_max as i32,
                stream,
            );
        }
    }
    Ok(())
}
