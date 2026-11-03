//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Head-dim-generalized flash-decode (seq=1, GQA), head_dim a multiple of 32 and
/// <= 256 (each warp lane owns head_dim/32 dims). Same contract as
/// `gptoss_flash_decode_win` at `kv_start` 0; `sinks` optional (None -> empty-seeded softmax). Returns
/// `[b, n_head, hd]` F16, or None (caller falls back) unless CUDA + F16 + hd ok.
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
        || !head_dim.is_multiple_of(32)
        || !(32..=256).contains(&head_dim)
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

/// out = f32(y) . sigmoid(f32(g)) in one F16 launch, the sigmoid in F32 as the tensor ops
/// compute it. g and y: the same shape, F16. `None` unless on a card with both F16; nothing
/// has run then.
pub fn sigmoid_mul_f16(g: &Tensor, y: &Tensor) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !g.device().is_cuda()
        || g.dtype() != DType::F16
        || y.dtype() != DType::F16
        || g.elem_count() != y.elem_count()
    {
        return Ok(None);
    }
    let g = g.contiguous()?;
    let y = y.contiguous()?;
    let n = g.elem_count() as i64;
    let out = unsafe { Tensor::empty(y.shape(), DType::F16, &g.device())? };
    let dev = g.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (gs, gl) = g.storage_and_layout();
    let (ys, yl) = y.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(gc), C(yc), C(oc)) = (&*gs, &*ys, &*os) {
        let gsl = gc.as_cuda_slice::<half::f16>()?;
        let ysl = yc.as_cuda_slice::<half::f16>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let gp = (gsl.device_ptr(gsl.stream()).0 + (gl.start_offset() * 2) as u64) as *const c_void;
        let yp = (ysl.device_ptr(ysl.stream()).0 + (yl.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            loken_sigmoid_mul_f16(gp, yp, op, n, stream);
        }
    }
    drop(gs);
    drop(ys);
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

/// The input stage of a prefill attention layer in one F16 launch: the head norms of q and k
/// when `norms` holds them, the rotation of q and k, and the write of k and v into the paged
/// store `kp`/`vp` at `slot_dev`.
///
/// `srcs` are q, k and v, each a `[rows, width]` tensor and the column its group starts at, so
/// the three can be views of one fused projection. `slot_dev` is `[rows]` I32 and `slots` holds
/// the same values on the host. `kp`/`vp` are `[num_slots, n_kv * hd]` and start at offset 0. `cos`/`sin` are
/// `[rows, 1, hd/2]`. Returns q `[rows, n_head, hd]`, k and v `[rows, n_kv, hd]`, rotated and
/// normed as the separate launches give them. Returns `None` when the card, a dtype, a width, a
/// store shape, a slot or the shared memory does not fit, and the caller runs the separate
/// launches. The first `None` of the process is noted at debug level with its reason.
pub fn prefill_rope_kv_write_f16(
    srcs: [(&Tensor, usize); 3],
    norms: Option<[(&Tensor, f32); 2]>,
    cos: &Tensor,
    sin: &Tensor,
    interleaved: bool,
    slot_dev: &Tensor,
    slots: &[i32],
    kp: &Tensor,
    vp: &Tensor,
    n_head: usize,
    n_kv: usize,
    hd: usize,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::kernel_ffi as kf;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    let rows = srcs[0].0.dims().first().copied().unwrap_or(0);
    let widths = [n_head * hd, n_kv * hd, n_kv * hd];
    let card = srcs[0].0.device();
    let srcs_fit = srcs.iter().zip(widths).all(|((t, col), w)| {
        t.device().same_device(&card)
            && t.dtype() == DType::F16
            && t.dims().len() == 2
            && t.dims()[0] == rows
            && col + w <= t.dims()[1]
    });
    let norms_fit = norms.is_none_or(|ns| {
        ns.iter().all(|(w, _)| {
            w.device().same_device(&card) && w.dtype() == DType::F32 && w.elem_count() == hd
        })
    });
    let on_card = [cos, sin, slot_dev, kp, vp]
        .iter()
        .all(|t| t.device().same_device(&card));
    // The store of the layer: `[num_slots, n_kv * hd]` for both k and v.
    let store_slots = kp.dims().first().copied().unwrap_or(0);
    let stores_fit = [kp, vp]
        .iter()
        .all(|t| t.dims() == [store_slots, n_kv * hd]);
    let slots_fit = slots.len() == rows
        && slots
            .iter()
            .all(|&s| usize::try_from(s).is_ok_and(|s| s < store_slots));
    // The block size of the kernel, and the shared memory it asks for: the head as f32 and one
    // partial sum per thread.
    let block = hd.min(256).next_power_of_two();
    let misfit = if rows == 0 || !card.is_cuda() || !on_card {
        Some("device")
    } else if hd == 0 || !hd.is_multiple_of(2) || !srcs_fit {
        Some("shape")
    } else if cos.dtype() != DType::F16
        || sin.dtype() != DType::F16
        || slot_dev.dtype() != DType::I32
        || kp.dtype() != DType::F16
        || vp.dtype() != DType::F16
        || !norms_fit
    {
        Some("dtype")
    } else if cos.elem_count() != rows * (hd / 2)
        || sin.elem_count() != rows * (hd / 2)
        || slot_dev.elem_count() != rows
        || !stores_fit
    {
        Some("shape")
    } else if !slots_fit {
        Some("slot out of range")
    } else if !fused_norm_fits(srcs[0].0, hd + block) {
        Some("shared memory")
    } else {
        None
    };
    if let Some(reason) = misfit {
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| {
            tracing::debug!("fused prefill input stage declined ({reason}), separate launches run");
        });
        return Ok(None);
    }
    let q_out = unsafe { Tensor::empty((rows, n_head, hd), DType::F16, &card)? };
    let k_out = unsafe { Tensor::empty((rows, n_kv, hd), DType::F16, &card)? };
    let v_out = unsafe { Tensor::empty((rows, n_kv, hd), DType::F16, &card)? };
    let dev = card.as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (qs, ql) = srcs[0].0.storage_and_layout();
    let (ks, kl) = srcs[1].0.storage_and_layout();
    let (vs, vl) = srcs[2].0.storage_and_layout();
    let (cs, cl) = cos.storage_and_layout();
    let (ss, sl) = sin.storage_and_layout();
    let (sds, sdl) = slot_dev.storage_and_layout();
    let (kps, kpl) = kp.storage_and_layout();
    let (vps, vpl) = vp.storage_and_layout();
    let (qos, _) = q_out.storage_and_layout();
    let (kos, _) = k_out.storage_and_layout();
    let (vos, _) = v_out.storage_and_layout();
    let qw = norms.map(|ns| ns[0].0.storage_and_layout());
    let kw = norms.map(|ns| ns[1].0.storage_and_layout());
    let weight_ptr = |w: &Option<(kf::StorageHandle, kf::Layout)>| -> *const f32 {
        match w {
            Some((handle, layout)) => match &**handle {
                C(wc) => wc.as_cuda_slice::<f32>().map_or(core::ptr::null(), |s| {
                    (s.device_ptr(s.stream()).0 + (layout.start_offset() * 4) as u64) as *const f32
                }),
                _ => core::ptr::null(),
            },
            None => core::ptr::null(),
        }
    };
    let (q_eps, k_eps) = norms.map_or((0.0, 0.0), |ns| (ns[0].1, ns[1].1));
    if norms.is_some() && (weight_ptr(&qw).is_null() || weight_ptr(&kw).is_null()) {
        return Ok(None);
    }
    debug_assert_eq!(kpl.start_offset(), 0);
    debug_assert_eq!(vpl.start_offset(), 0);
    let (C(qc), C(kc), C(vc), C(cc), C(sc), C(sdc), C(kpc), C(vpc), C(qoc), C(koc), C(voc)) = (
        &*qs, &*ks, &*vs, &*cs, &*ss, &*sds, &*kps, &*vps, &*qos, &*kos, &*vos,
    ) else {
        return Ok(None);
    };
    let f16_at = |c: &kf::CudaStorage, offset: usize| -> Result<u64> {
        let s = c.as_cuda_slice::<half::f16>()?;
        Ok(s.device_ptr(s.stream()).0 + (offset * 2) as u64)
    };
    let qp = f16_at(qc, ql.start_offset() + srcs[0].1)? as *const c_void;
    let kp_in = f16_at(kc, kl.start_offset() + srcs[1].1)? as *const c_void;
    let vp_in = f16_at(vc, vl.start_offset() + srcs[2].1)? as *const c_void;
    let cp = f16_at(cc, cl.start_offset())? as *const c_void;
    let sp = f16_at(sc, sl.start_offset())? as *const c_void;
    let kpp = f16_at(kpc, 0)? as *mut c_void;
    let vpp = f16_at(vpc, 0)? as *mut c_void;
    let qop = f16_at(qoc, 0)? as *mut c_void;
    let kop = f16_at(koc, 0)? as *mut c_void;
    let vop = f16_at(voc, 0)? as *mut c_void;
    let sdsl = sdc.as_cuda_slice::<i32>()?;
    let sdp = (sdsl.device_ptr(sdsl.stream()).0 + (sdl.start_offset() * 4) as u64) as *const i32;
    unsafe {
        loken_prefill_qk_rope_kv_f16(
            qp,
            kp_in,
            vp_in,
            srcs[0].0.dims()[1] as i64,
            srcs[1].0.dims()[1] as i64,
            srcs[2].0.dims()[1] as i64,
            weight_ptr(&qw),
            weight_ptr(&kw),
            q_eps,
            k_eps,
            cp,
            sp,
            qop,
            kop,
            vop,
            kpp,
            vpp,
            sdp,
            rows as i32,
            n_head as i32,
            n_kv as i32,
            hd as i32,
            interleaved as i32,
            stream,
        );
    }
    drop((qs, ks, vs, cs, ss, sds, kps, vps, qos, kos, vos, qw, kw));
    Ok(Some((q_out, k_out, v_out)))
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

#[cfg(test)]
mod sigmoid_mul_tests {
    use super::*;
    use crate::tensor::Device;

    /// The one launch gives the gate applied as the tensor ops apply it.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_gated_rows_match_the_tensor_ops() {
        let Ok(card) = crate::tensor::cuda::CudaDevice::new(0).map(Device::Cuda) else {
            return;
        };
        let n = 6144;
        let wave = |k: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32) * k).sin() * 3.0).collect() };
        let t = |v: Vec<f32>| {
            Tensor::from_vec(v, (1, 1, n), &card)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
        };
        let (g, y) = (t(wave(0.013)), t(wave(0.029)));
        let got = sigmoid_mul_f16(&g, &y)
            .unwrap()
            .expect("the launch takes these");
        let want = (crate::tensor::ops::sigmoid(&g.to_dtype(DType::F32).unwrap()).unwrap()
            * y.to_dtype(DType::F32).unwrap())
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
        let flat = |t: &Tensor| -> Vec<f32> {
            t.to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let gap = flat(&got)
            .iter()
            .zip(flat(&want))
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(gap < 2e-3, "gap {gap}");
    }
}
