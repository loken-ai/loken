//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Head-dim-generalized flash-decode (seq=1, GQA), head_dim a multiple of 32 and
/// <= 256 (each warp lane owns head_dim/32 dims). Same contract as
/// `gptoss_flash_decode`; `sinks` optional (None -> empty-seeded softmax). Returns
/// `[b, n_head, hd]` F16, or None (caller falls back) unless CUDA + F16 + hd ok.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sinks: Option<&Tensor>,
    scale: f32,
    batch: usize,
    n_head: usize,
    n_kv: usize,
    kv_len: usize,
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
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
    {
        return Ok(None);
    }
    let q = q.contiguous()?;
    let kst_lay = k.layout();
    let kst = kst_lay.stride();
    let vst_lay = v.layout();
    let vst = vst_lay.stride();
    if kst[3] != 1 || vst[3] != 1 {
        return Ok(None);
    }
    let out = unsafe { Tensor::empty((batch, n_head, head_dim), DType::F16, &q.device())? };
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    let sinks_c = match sinks {
        Some(s) => Some(s.contiguous()?),
        None => None,
    };
    let mask_c = match mask {
        Some(m) => Some(m.contiguous()?),
        None => None,
    };
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    let ss = sinks_c.as_ref().map(|s| s.storage_and_layout());
    let ms = mask_c.as_ref().map(|m| m.storage_and_layout());

    if let (C(qc), C(kc), C(vc), C(oc)) = (&*qs, &*ks, &*vs, &*os) {
        let qsl = qc.as_cuda_slice::<half::f16>()?;
        let ksl = kc.as_cuda_slice::<half::f16>()?;
        let vsl = vc.as_cuda_slice::<half::f16>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let qp = (qsl.device_ptr(qsl.stream()).0 + (ql.start_offset() * 2) as u64) as *const c_void;
        let kp = (ksl.device_ptr(ksl.stream()).0 + (kl.start_offset() * 2) as u64) as *const c_void;
        let vp = (vsl.device_ptr(vsl.stream()).0 + (vl.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        let mut sp: *const f32 = std::ptr::null();
        if let Some((sstore, _)) = &ss {
            if let C(sc) = &**sstore {
                let ssl = sc.as_cuda_slice::<f32>()?;
                sp = ssl.device_ptr(ssl.stream()).0 as *const f32;
            }
        }
        let mut mp: *const f32 = std::ptr::null();
        if let Some((mstore, mlayout)) = &ms {
            if let C(mc) = &**mstore {
                let msl = mc.as_cuda_slice::<f32>()?;
                mp = (msl.device_ptr(msl.stream()).0 + (mlayout.start_offset() * 4) as u64)
                    as *const f32;
            }
        }
        // Split-K when kv is large: the serial one-warp/head scan leaves the GPU
        // idle (qwen3.5 launch-bound). Tiered nsplit (held to {32,64,128} so the F32
        // scratch is one of 3 fixed sizes - same mempool-friendliness as the gpt-oss
        // path). Mathematically identical to the serial kernel (log-sum-exp merge).
        let nsplit: usize = if kv_len >= 3072 {
            128
        } else if kv_len >= 1024 {
            64
        } else if kv_len >= 384 {
            32
        } else {
            1
        };
        if nsplit > 1 {
            let part_m =
                unsafe { Tensor::empty((batch, n_head, nsplit), DType::F32, &q.device())? };
            let part_l =
                unsafe { Tensor::empty((batch, n_head, nsplit), DType::F32, &q.device())? };
            let part_acc = unsafe {
                Tensor::empty((batch, n_head, nsplit, head_dim), DType::F32, &q.device())?
            };
            let (pms, _) = part_m.storage_and_layout();
            let (pls, _) = part_l.storage_and_layout();
            let (pas, _) = part_acc.storage_and_layout();
            if let (C(pmc), C(plc), C(pac)) = (&*pms, &*pls, &*pas) {
                let pmp = pmc
                    .as_cuda_slice::<f32>()?
                    .device_ptr(pmc.as_cuda_slice::<f32>()?.stream())
                    .0 as *mut f32;
                let plp = plc
                    .as_cuda_slice::<f32>()?
                    .device_ptr(plc.as_cuda_slice::<f32>()?.stream())
                    .0 as *mut f32;
                let pap = pac
                    .as_cuda_slice::<f32>()?
                    .device_ptr(pac.as_cuda_slice::<f32>()?.stream())
                    .0 as *mut f32;
                unsafe {
                    loken_flash_decode_split_f16(
                        qp,
                        kp,
                        vp,
                        mp,
                        sp,
                        op,
                        pmp,
                        plp,
                        pap,
                        batch as i32,
                        n_head as i32,
                        n_kv as i32,
                        kv_len as i32,
                        nsplit as i32,
                        head_dim as i32,
                        scale,
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
            drop(pms);
            drop(pls);
            drop(pas);
        } else {
            unsafe {
                loken_flash_decode_f16(
                    qp,
                    kp,
                    vp,
                    mp,
                    sp,
                    op,
                    batch as i32,
                    n_head as i32,
                    n_kv as i32,
                    kv_len as i32,
                    head_dim as i32,
                    scale,
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
    }
    drop(qs);
    drop(ks);
    drop(vs);
    drop(ss);
    drop(os);
    drop(ms);
    let _ = &sinks_c;
    Ok(Some(out))
}

/// out = silu(f32(g)) . f32(u) in one F16 launch (F32-internal silu, bit-identical
/// to the tensor-op silu(g.to_f32).u.to_f32 reference). g/u: same-shape F16 contiguous.
/// Returns None (caller falls back) unless CUDA + both F16.
pub fn silu_mul_f16(g: &Tensor, u: &Tensor) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !g.device().is_cuda() || g.dtype() != DType::F16 || u.dtype() != DType::F16 {
        return Ok(None);
    }
    let g = g.contiguous()?;
    let u = u.contiguous()?;
    let n = g.elem_count() as i64;
    let out = unsafe { Tensor::empty(g.shape(), DType::F16, &g.device())? };
    let dev = g.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (gs, gl) = g.storage_and_layout();
    let (us, ul) = u.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(gc), C(uc), C(oc)) = (&*gs, &*us, &*os) {
        let gsl = gc.as_cuda_slice::<half::f16>()?;
        let usl = uc.as_cuda_slice::<half::f16>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let gp = (gsl.device_ptr(gsl.stream()).0 + (gl.start_offset() * 2) as u64) as *const c_void;
        let up = (usl.device_ptr(usl.stream()).0 + (ul.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            loken_silu_mul_f16(gp, up, op, n, stream);
        }
    }
    drop(gs);
    drop(us);
    drop(os);
    Ok(Some(out))
}

/// Partial NEOX RoPE in one launch (was narrow+contig+rope+cat+contig). x
/// [b,nh,seq,hd] F16, cos/sin [seq, rope_dim/2] F16. Rotates first `rope_dim`
/// dims, passes through the rest. Bit-identical to the reference f16 rope. Returns
/// [b,nh,seq,hd] F16, or None (caller falls back) unless CUDA + all F16.
pub fn neox_rope_f16(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rope_dim: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !x.device().is_cuda()
        || x.dtype() != DType::F16
        || cos.dtype() != DType::F16
        || sin.dtype() != DType::F16
    {
        return Ok(None);
    }
    let (b, nh, seq, hd) = x.dims4()?;
    let x = x.contiguous()?;
    // cos/sin: no `contiguous()` - native tensors are always packed-with-offset
    // (this substrate has no strided views) and the FFI extraction below carries
    // the view offset, so packing a narrow view here would be a pure extra copy
    // launch per table per call (2/layer/token on the lfm2 decode path).
    let outer = (b * nh * seq) as i32;
    let out = unsafe { Tensor::empty((b, nh, seq, hd), DType::F16, &x.device())? };
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (xs, xl) = x.storage_and_layout();
    let (cs, cl) = cos.storage_and_layout();
    let (ss, sl) = sin.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(xc), C(cc), C(sc), C(oc)) = (&*xs, &*cs, &*ss, &*os) {
        let xsl = xc.as_cuda_slice::<half::f16>()?;
        let csl = cc.as_cuda_slice::<half::f16>()?;
        let ssl = sc.as_cuda_slice::<half::f16>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let xp = (xsl.device_ptr(xsl.stream()).0 + (xl.start_offset() * 2) as u64) as *const c_void;
        let cp = (csl.device_ptr(csl.stream()).0 + (cl.start_offset() * 2) as u64) as *const c_void;
        let sp = (ssl.device_ptr(ssl.stream()).0 + (sl.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            loken_neox_rope_f16(
                xp,
                cp,
                sp,
                op,
                outer,
                seq as i32,
                hd as i32,
                rope_dim as i32,
                stream,
            );
        }
    }
    drop(xs);
    drop(cs);
    drop(ss);
    drop(os);
    Ok(Some(out))
}

/// Paged-decode RoPE (F16), replacing the ~8-op tensor `paged_attention::rope_apply`
/// in the CB decode with ONE kernel. `x`: `[B, heads, hd]` (one position/sequence-row);
/// `cos`/`sin`: `[B, 1, half]` (per-BATCH position). `interleaved` selects GPT-J
/// (Llama/Mistral) vs NeoX pairing - MUST match the model's `flags.use_rope_i`.
/// Returns `None` for non-CUDA/non-F16 (caller falls back to the tensor path).
pub fn paged_rope_f16(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    interleaved: bool,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !x.device().is_cuda()
        || x.dtype() != DType::F16
        || cos.dtype() != DType::F16
        || sin.dtype() != DType::F16
    {
        return Ok(None);
    }
    let dims = x.dims();
    if dims.len() != 3 {
        return Ok(None);
    }
    let (b, heads, hd) = (dims[0], dims[1], dims[2]);
    let half = *cos.dims().last().unwrap_or(&0);
    let rope_dim = half * 2;
    if rope_dim == 0 || rope_dim > hd {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let outer = (b * heads) as i32;
    let out = unsafe { Tensor::empty((b, heads, hd), DType::F16, &x.device())? };
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (xs, xl) = x.storage_and_layout();
    let (cs, cl) = cos.storage_and_layout();
    let (ss, sl) = sin.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(xc), C(cc), C(sc), C(oc)) = (&*xs, &*cs, &*ss, &*os) {
        let xsl = xc.as_cuda_slice::<half::f16>()?;
        let csl = cc.as_cuda_slice::<half::f16>()?;
        let ssl = sc.as_cuda_slice::<half::f16>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let xp = (xsl.device_ptr(xsl.stream()).0 + (xl.start_offset() * 2) as u64) as *const c_void;
        let cp = (csl.device_ptr(csl.stream()).0 + (cl.start_offset() * 2) as u64) as *const c_void;
        let sp = (ssl.device_ptr(ssl.stream()).0 + (sl.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            loken_paged_rope_f16(
                xp,
                cp,
                sp,
                op,
                outer,
                heads as i32,
                hd as i32,
                rope_dim as i32,
                interleaved as i32,
                stream,
            );
        }
    }
    drop(xs);
    drop(cs);
    drop(ss);
    drop(os);
    Ok(Some(out))
}

/// Device-position partial-NEOX rope: like `neox_rope_f16` but cos/sin are the
/// FULL [max_seq, rope_dim/2] F16 tables and the position is read from pos_dev
/// (i32 [1]) - for CUDA-graph replay. x [b,nh,seq,hd] F16. Returns [b,nh,seq,hd]
/// F16, or None unless CUDA + all F16.
pub fn neox_rope_devpos_f16(
    x: &Tensor,
    cos_full: &Tensor,
    sin_full: &Tensor,
    pos_dev: &Tensor,
    rope_dim: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !x.device().is_cuda()
        || x.dtype() != DType::F16
        || cos_full.dtype() != DType::F16
        || sin_full.dtype() != DType::F16
    {
        return Ok(None);
    }
    let (b, nh, seq, hd) = x.dims4()?;
    let x = x.contiguous()?;
    let cos_full = cos_full.contiguous()?;
    let sin_full = sin_full.contiguous()?;
    let outer = (b * nh * seq) as i32;
    let out = unsafe { Tensor::empty((b, nh, seq, hd), DType::F16, &x.device())? };
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (xs, xl) = x.storage_and_layout();
    let (cs, cl) = cos_full.storage_and_layout();
    let (ss, sl) = sin_full.storage_and_layout();
    let (ps, pl) = pos_dev.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(xc), C(cc), C(sc), C(pc), C(oc)) = (&*xs, &*cs, &*ss, &*ps, &*os) {
        let xsl = xc.as_cuda_slice::<half::f16>()?;
        let csl = cc.as_cuda_slice::<half::f16>()?;
        let ssl = sc.as_cuda_slice::<half::f16>()?;
        let psl = pc.as_cuda_slice::<i32>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let xp = (xsl.device_ptr(xsl.stream()).0 + (xl.start_offset() * 2) as u64) as *const c_void;
        let cp = (csl.device_ptr(csl.stream()).0 + (cl.start_offset() * 2) as u64) as *const c_void;
        let sp = (ssl.device_ptr(ssl.stream()).0 + (sl.start_offset() * 2) as u64) as *const c_void;
        let pp = (psl.device_ptr(psl.stream()).0 + (pl.start_offset() * 4) as u64) as *const i32;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            loken_neox_rope_devpos_f16(
                xp,
                cp,
                sp,
                pp,
                op,
                outer,
                seq as i32,
                hd as i32,
                rope_dim as i32,
                stream,
            );
        }
    }
    drop(xs);
    drop(cs);
    drop(ss);
    drop(ps);
    drop(os);
    Ok(Some(out))
}

/// out_f16 = f16(a_f32 + f32(b_f16)) in one launch (F32 add -> bit-identical to
/// a.broadcast_add(b.to_f32()).to_f16()). a F32, b F16, same shape. Returns F16,
/// or None (caller falls back) unless CUDA + a F32 + b F16.
pub fn add_to_f16(a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !a.device().is_cuda() || a.dtype() != DType::F32 || b.dtype() != DType::F16 {
        return Ok(None);
    }
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let n = a.elem_count() as i64;
    let out = unsafe { Tensor::empty(a.shape(), DType::F16, &a.device())? };
    let dev = a.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (as_, al) = a.storage_and_layout();
    let (bs, bl) = b.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(ac), C(bc), C(oc)) = (&*as_, &*bs, &*os) {
        let asl = ac.as_cuda_slice::<f32>()?;
        let bsl = bc.as_cuda_slice::<half::f16>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let ap = (asl.device_ptr(asl.stream()).0 + (al.start_offset() * 4) as u64) as *const f32;
        let bp = (bsl.device_ptr(bsl.stream()).0 + (bl.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            loken_add_to_f16(ap, bp, op, n, stream);
        }
    }
    drop(as_);
    drop(bs);
    drop(os);
    Ok(Some(out))
}

/// out = softplus(dt + bias) = log(exp(dt+bias)+1), one F32 launch (bit-identical
/// to the reference ((dt.broadcast_add(bias)).exp()+1).log()). dt [rows,cols] F32, bias
/// [cols] F32. Returns None (caller falls back) unless CUDA + F32.
pub fn softplus_bias(dt: &Tensor, bias: &Tensor) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    if !dt.device().is_cuda() || dt.dtype() != DType::F32 || bias.dtype() != DType::F32 {
        return Ok(None);
    }
    let dt = dt.contiguous()?;
    let bias = bias.flatten_all()?.contiguous()?;
    let dims = dt.dims();
    let cols = *dims.last().unwrap();
    let rows = (dt.elem_count() / cols.max(1)) as i32;
    let out = unsafe { Tensor::empty(dt.shape(), DType::F32, &dt.device())? };
    let dev = dt.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (ds, dl) = dt.storage_and_layout();
    let (bs, bl) = bias.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(dc), C(bc), C(oc)) = (&*ds, &*bs, &*os) {
        let dsl = dc.as_cuda_slice::<f32>()?;
        let bsl = bc.as_cuda_slice::<f32>()?;
        let osl = oc.as_cuda_slice::<f32>()?;
        let dp = (dsl.device_ptr(dsl.stream()).0 + (dl.start_offset() * 4) as u64) as *const f32;
        let bp = (bsl.device_ptr(bsl.stream()).0 + (bl.start_offset() * 4) as u64) as *const f32;
        let op = osl.device_ptr(osl.stream()).0 as *mut f32;
        unsafe {
            loken_softplus_bias(dp, bp, op, rows, cols as i32, stream);
        }
    }
    drop(ds);
    drop(bs);
    drop(os);
    Ok(Some(out))
}
