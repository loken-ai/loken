//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Capture-safe PAGED flash-decode: Q `[b,n_head,hd]` attends over a flat paged KV
/// store `kp/vp` `[num_slots, feat]` (feat=n_kv*hd) using per-seq `block_table`
/// `[b,max_blocks]` + `seq_lens` `[b]` (both i32, read on-device -> graph-safe).
/// Returns `[b,n_head,hd]` f16. The paged gather + attention are fused (no
/// index_select, which would bake its index at capture).
#[allow(clippy::too_many_arguments)]
pub fn paged_flash_decode(
    q: &Tensor,
    kp: &Tensor,
    vp: &Tensor,
    block_table: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    batch: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !q.device().is_cuda()
        || head_dim % 32 != 0
        || head_dim < 32
        || head_dim > 256
        || q.dtype() != DType::F16
        || kp.dtype() != DType::F16
    {
        return Ok(None);
    }
    let feat = n_kv * head_dim;
    let q = q.contiguous()?;
    let out = unsafe { Tensor::empty((batch, n_head, head_dim), DType::F16, &q.device())? };
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    // KV-split: split each (head,batch) KV scan across `nsplit` z-blocks so a single
    // low-batch decode FILLS the GPU instead of running just n_head warps serially over
    // the whole sequence (the batch=1 long-context collapse). `nsplit` depends only on
    // batchxn_head (constant per CUDA-graph capture); the per-split KV range is derived
    // from on-device `seq_lens` at exec time, so it stays correct as the sequence grows
    // across graph replays. Well-filled grids (large batch) keep `nsplit==1` -> the plain
    // serial kernel, no combine overhead.
    // A/B: 512->1024 won (+6-14%); 1024->2048 NEUTRAL (mistral-nemo
    // CB N=1/8/16 = 88/388/535 vs 90/390/527, all ±2% noise) - 1024 is the
    // occupancy sweet spot; extra splits' combine overhead offsets the fill.
    // The residual N=8 vs-vLLM gap is in the mmq GEMM / attention kernel, NOT here.
    const TARGET_WARPS: usize = 1024;
    const MAX_NSPLIT: usize = 64;
    let lanes = (batch * n_head).max(1);
    let nsplit = TARGET_WARPS.div_ceil(lanes).clamp(1, MAX_NSPLIT);
    let parts = if nsplit > 1 {
        let n = batch * n_head * nsplit;
        let pm = unsafe { Tensor::empty((n,), DType::F32, &q.device())? };
        let pl = unsafe { Tensor::empty((n,), DType::F32, &q.device())? };
        let pa = unsafe { Tensor::empty((n * head_dim,), DType::F32, &q.device())? };
        Some((pm, pl, pa))
    } else {
        None
    };
    let (qs, ql) = q.storage_and_layout();
    let (ks, _) = kp.storage_and_layout();
    let (vs, _) = vp.storage_and_layout();
    let (bts, btl) = block_table.storage_and_layout();
    let (sls, sll) = seq_lens.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(qc), C(kc), C(vc), C(btc), C(slc), C(oc)) = (&*qs, &*ks, &*vs, &*bts, &*sls, &*os) {
        let qsl = qc.as_cuda_slice::<half::f16>()?;
        let ksl = kc.as_cuda_slice::<half::f16>()?;
        let vsl = vc.as_cuda_slice::<half::f16>()?;
        let btsl = btc.as_cuda_slice::<i32>()?;
        let slsl = slc.as_cuda_slice::<i32>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let qp = (qsl.device_ptr(qsl.stream()).0 + (ql.start_offset() * 2) as u64) as *const c_void;
        let kpp = ksl.device_ptr(ksl.stream()).0 as *const c_void;
        let vpp = vsl.device_ptr(vsl.stream()).0 as *const c_void;
        let btp =
            (btsl.device_ptr(btsl.stream()).0 + (btl.start_offset() * 4) as u64) as *const i32;
        let slp =
            (slsl.device_ptr(slsl.stream()).0 + (sll.start_offset() * 4) as u64) as *const i32;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        if let Some((pm, pl, pa)) = parts.as_ref() {
            let (pms, _) = pm.storage_and_layout();
            let (pls, _) = pl.storage_and_layout();
            let (pas, _) = pa.storage_and_layout();
            if let (C(pmc), C(plc), C(pac)) = (&*pms, &*pls, &*pas) {
                let pmsl = pmc.as_cuda_slice::<f32>()?;
                let plsl = plc.as_cuda_slice::<f32>()?;
                let pasl = pac.as_cuda_slice::<f32>()?;
                let pmp = pmsl.device_ptr(pmsl.stream()).0 as *mut f32;
                let plp = plsl.device_ptr(plsl.stream()).0 as *mut f32;
                let pap = pasl.device_ptr(pasl.stream()).0 as *mut f32;
                unsafe {
                    loken_paged_flash_decode_split_f16(
                        qp,
                        kpp,
                        vpp,
                        btp,
                        slp,
                        op,
                        pmp,
                        plp,
                        pap,
                        batch as i32,
                        n_head as i32,
                        n_kv as i32,
                        nsplit as i32,
                        head_dim as i32,
                        scale,
                        block_size as i32,
                        max_blocks as i32,
                        feat as i32,
                        stream,
                    );
                }
                return Ok(Some(out));
            }
        }
        unsafe {
            loken_paged_flash_decode_f16(
                qp,
                kpp,
                vpp,
                btp,
                slp,
                op,
                batch as i32,
                n_head as i32,
                n_kv as i32,
                head_dim as i32,
                scale,
                block_size as i32,
                max_blocks as i32,
                feat as i32,
                stream,
            );
        }
        return Ok(Some(out));
    }
    Ok(None)
}

/// Split count over the KV axis for flash-prefill (batch==1). The plain grid has
/// `nqblk*n_head` warps (~1024 at seq_q=512); at late chunks each warp serially
/// scans the whole KV, starving the GPU. Split so total blocks fill the device
/// (~target warps), but keep each split's KV chunk >= a few WMMA tiles so the
/// per-split fixed cost (Q stage + combine) stays amortized.
pub(super) fn pick_prefill_nsplit(n_head: usize, seq_q: usize, seq_kv: usize) -> usize {
    const TARGET_WARPS: usize = 8192;
    const MAX_NSPLIT: usize = 32;
    let nqblk = seq_q.div_ceil(16);
    let base_warps = (nqblk * n_head).max(1);
    let by_fill = TARGET_WARPS.div_ceil(base_warps);
    // Don't over-split short KV: keep >= 256 kv tokens (16 tiles) per split.
    let by_work = (seq_kv / 256).max(1);
    by_fill.min(by_work).clamp(1, MAX_NSPLIT)
}

/// Fused FlashAttention-2-style PREFILL for GQA over F16 K/V (seq_q > 1, causal).
///
/// Replaces the `q.kᵀ -> softmax -> att.v` chain that materializes the
/// `[n_head, seq_q, seq_kv]` scores in HBM (the O(seq²) long-context prefill
/// loss). Tiles QxK, online-softmax in SRAM, tensor-core mma.sync for both GEMMs;
/// GQA + causal (+ optional sliding-window) mask handled analytically in-kernel
/// Non-causal BF16 flash attention for a diffusion transformer's self-attention.
///
/// `q`/`k`/`v`: `[b, n_head, seq, head_dim]`, BF16 or F32 on CUDA, head_dim 64 or 128, no
/// GQA and no mask. Returns `[b, n_head, seq_q, head_dim]` in the caller's dtype, or `None`
/// when the shapes or the device are outside what the kernel covers - the caller then runs
/// its ordinary tiled path.
///
/// The point is what it does NOT do: the tiled path writes a `[tile, seq]` score slab per
/// head to HBM, reads it back for the softmax, writes it again and reads it a third time.
/// Here that slab never leaves shared memory. Measured on this workload, attention is 82%
/// of a 14B block and runs at 31 TFLOP/s where the same card's BF16 GEMMs reach 86, so the
/// traffic is the cost and removing it is the whole win.
pub fn flash_dit_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    // Tokens per video frame, for the radial mask. 0 = dense (every tile computed).
    frame_tokens: usize,
    // Patches across one frame, so the mask can measure distance in two dimensions rather
    // than along the raster index - the difference between a neighbourhood and a full-width
    // strip, and at a 64-patch-wide frame that is most of the cost.
    grid_w: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !q.device().is_cuda() {
        return Ok(None);
    }
    let out_dtype = q.dtype();
    let (b, nh, seq_q, hd) = q.dims4()?;
    let (bk, nhk, seq_kv, hdk) = k.dims4()?;
    if hd != hdk || (hd != 64 && hd != 128) || nh != nhk || bk != b || seq_q == 0 || seq_kv == 0 {
        return Ok(None);
    }
    if k.dims4()? != v.dims4()? {
        return Ok(None);
    }
    // This kernel ships SASS only for the architectures whose shared memory fits its
    // tile, and a missing kernel image is an ASYNC error the raw launcher cannot
    // report. The first call on each device is synchronized and checked; a card the
    // image does not cover is remembered and every later call takes the fallback.
    static DIT_LAUNCH_OK: OnceLock<std::sync::Mutex<std::collections::HashMap<usize, bool>>> =
        OnceLock::new();
    let ordinal = q.device().as_cuda_device()?.ordinal();
    let verdict = DIT_LAUNCH_OK
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&ordinal)
        .copied();
    if verdict == Some(false) {
        return Ok(None);
    }
    let q = q.to_dtype(DType::BF16)?.contiguous()?;
    let k = k.to_dtype(DType::BF16)?.contiguous()?;
    let v = v.to_dtype(DType::BF16)?.contiguous()?;
    let out = unsafe { Tensor::empty((b, nh, seq_q, hd), DType::BF16, &q.device())? };
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(qc), C(kc), C(vc), C(oc)) = (&*qs, &*ks, &*vs, &*os) {
        let qsl = qc.as_cuda_slice::<half::bf16>()?;
        let ksl = kc.as_cuda_slice::<half::bf16>()?;
        let vsl = vc.as_cuda_slice::<half::bf16>()?;
        let osl = oc.as_cuda_slice::<half::bf16>()?;
        let qp = (qsl.device_ptr(qsl.stream()).0 + (ql.start_offset() * 2) as u64) as *const c_void;
        let kp = (ksl.device_ptr(ksl.stream()).0 + (kl.start_offset() * 2) as u64) as *const c_void;
        let vp = (vsl.device_ptr(vsl.stream()).0 + (vl.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            if hd == 64 {
                flash_dit_bf16_hd64_raw(
                    qp,
                    kp,
                    vp,
                    op,
                    b as i32,
                    nh as i32,
                    seq_q as i32,
                    seq_kv as i32,
                    scale,
                    frame_tokens as i32,
                    grid_w as i32,
                    stream,
                );
            } else {
                flash_dit_bf16_hd128_raw(
                    qp,
                    kp,
                    vp,
                    op,
                    b as i32,
                    nh as i32,
                    seq_q as i32,
                    seq_kv as i32,
                    scale,
                    frame_tokens as i32,
                    grid_w as i32,
                    stream,
                );
            }
        }
        if verdict.is_none() {
            let landed = dev.synchronize().is_ok();
            DIT_LAUNCH_OK
                .get_or_init(Default::default)
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(ordinal, landed);
            if !landed {
                tracing::warn!(
                    "flash_dit_bf16: first launch on gpu {ordinal} failed - this card has \
                     no image for the kernel; using the ordinary attention path"
                );
                return Ok(None);
            }
        }
        return Ok(Some(out.to_dtype(out_dtype)?));
    }
    Ok(None)
}

/// (no repeat_kv, no mask tensor read). See cuda/flash_prefill_f16.cu.
///
/// `q`: `[b, n_head, seq_q, hd]`; `k`/`v`: `[b, n_kv, seq_kv, hd]` (all F16). The
/// analytic mask reproduces `make_mask`: query row i (abs pos i + seq_kv - seq_q)
/// attends key j iff `j <= abs_i && (window <= 0 || j + window >= abs_i)`. Returns
/// `[b, n_head, seq_q, hd]` F16, or `None` (-> caller's chain fallback) when the
/// shapes/dtype/device/head_dim are unsupported.
#[allow(clippy::too_many_arguments)]
pub fn flash_prefill_f16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_head: usize,
    n_kv: usize,
    scale: f32,
    window: i32,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !q.device().is_cuda() {
        return Ok(None);
    }
    let out_dtype = q.dtype();
    let (b, nh, seq_q, hd) = q.dims4()?;
    let (bk, nkv, seq_kv, hdk) = k.dims4()?;
    if hd != hdk
        || (hd != 64 && hd != 128)
        || nh != n_head
        || nkv != n_kv
        || bk != b
        || n_kv == 0
        || n_head % n_kv != 0
        || seq_q < 1
        || seq_kv < seq_q
    {
        return Ok(None);
    }
    // Tensor-core WMMA needs F16 inputs. GPU prefill runs the attention in F32,
    // so cast Q/K/V to F16 (O(seq.hd), cheap vs the avoided O(seq²) scores) with
    // F32 online-softmax accumulation in-kernel - the standard flash-attention
    // precision. Output is cast back to the caller's dtype. Contiguous as the
    // kernel indexes [b,n_head,seq_q,hd] / [b,n_kv,seq_kv,hd] row-major.
    let q = q.to_dtype(DType::F16)?.contiguous()?;
    let k = k.to_dtype(DType::F16)?.contiguous()?;
    let v = v.to_dtype(DType::F16)?.contiguous()?;
    let out = unsafe { Tensor::empty((b, n_head, seq_q, hd), DType::F16, &q.device())? };
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(qc), C(kc), C(vc), C(oc)) = (&*qs, &*ks, &*vs, &*os) {
        let qsl = qc.as_cuda_slice::<half::f16>()?;
        let ksl = kc.as_cuda_slice::<half::f16>()?;
        let vsl = vc.as_cuda_slice::<half::f16>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let qp = (qsl.device_ptr(qsl.stream()).0 + (ql.start_offset() * 2) as u64) as *const c_void;
        let kp = (ksl.device_ptr(ksl.stream()).0 + (kl.start_offset() * 2) as u64) as *const c_void;
        let vp = (vsl.device_ptr(vsl.stream()).0 + (vl.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        // Split-K over the KV axis for batch==1 when the KV scan is long enough to
        // starve the plain (nqblk*n_head)-warp grid - nsplitx more blocks fill the
        // GPU at late prefill chunks (seq_q=512 fixed but seq_kv -> full prompt).
        let nsplit = if b == 1 {
            pick_prefill_nsplit(n_head, seq_q, seq_kv)
        } else {
            1
        };
        if nsplit > 1 {
            let n_rows = n_head * seq_q; // combine touches exactly these rows (batch==1)
            let parts =
                unsafe { Tensor::empty((n_rows * nsplit * (hd + 2),), DType::F32, &q.device())? };
            let (pas, _) = parts.storage_and_layout();
            if let C(pac) = &*pas {
                let pasl = pac.as_cuda_slice::<f32>()?;
                let pap = pasl.device_ptr(pasl.stream()).0 as *mut f32;
                unsafe {
                    flash_prefill_split_f16_raw(
                        qp,
                        kp,
                        vp,
                        pap,
                        op,
                        b as i32,
                        n_head as i32,
                        n_kv as i32,
                        seq_q as i32,
                        seq_kv as i32,
                        hd as i32,
                        nsplit as i32,
                        scale,
                        window,
                        stream,
                    );
                }
                return Ok(Some(out.to_dtype(out_dtype)?));
            }
        }
        unsafe {
            flash_prefill_f16_raw(
                qp,
                kp,
                vp,
                op,
                b as i32,
                n_head as i32,
                n_kv as i32,
                seq_q as i32,
                seq_kv as i32,
                hd as i32,
                scale,
                window,
                stream,
            );
        }
        return Ok(Some(out.to_dtype(out_dtype)?));
    }
    Ok(None)
}

/// Capture-safe PAGED KV write: scatter B new tokens' K/V (`[b,feat]`) into the
/// paged store `kp/vp` at `slot_dev[bi]` (slot read on-device -> graph-safe).
pub fn paged_kv_write(
    k_new: &Tensor,
    v_new: &Tensor,
    kp: &Tensor,
    vp: &Tensor,
    slot_dev: &Tensor,
    batch: usize,
    feat: usize,
) -> Result<()> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !k_new.device().is_cuda() || k_new.dtype() != DType::F16 {
        return Ok(());
    }
    let k_new = k_new.contiguous()?;
    let v_new = v_new.contiguous()?;
    let dev = k_new.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (kns, knl) = k_new.storage_and_layout();
    let (vns, vnl) = v_new.storage_and_layout();
    let (kps, _) = kp.storage_and_layout();
    let (vps, _) = vp.storage_and_layout();
    let (sds, sdl) = slot_dev.storage_and_layout();
    if let (C(knc), C(vnc), C(kpc), C(vpc), C(sdc)) = (&*kns, &*vns, &*kps, &*vps, &*sds) {
        let knsl = knc.as_cuda_slice::<half::f16>()?;
        let vnsl = vnc.as_cuda_slice::<half::f16>()?;
        let kpsl = kpc.as_cuda_slice::<half::f16>()?;
        let vpsl = vpc.as_cuda_slice::<half::f16>()?;
        let sdsl = sdc.as_cuda_slice::<i32>()?;
        let knp =
            (knsl.device_ptr(knsl.stream()).0 + (knl.start_offset() * 2) as u64) as *const c_void;
        let vnp =
            (vnsl.device_ptr(vnsl.stream()).0 + (vnl.start_offset() * 2) as u64) as *const c_void;
        let kpp = kpsl.device_ptr(kpsl.stream()).0 as *mut c_void;
        let vpp = vpsl.device_ptr(vpsl.stream()).0 as *mut c_void;
        let sdp =
            (sdsl.device_ptr(sdsl.stream()).0 + (sdl.start_offset() * 4) as u64) as *const i32;
        unsafe {
            loken_paged_kv_write_f16(knp, vnp, kpp, vpp, sdp, batch as i32, feat as i32, stream);
        }
    }
    Ok(())
}

/// gpt-oss flash-attention decode (seq=1) with per-head sinks + GQA, in one F16
/// launch (no cuBLAS, no scores in HBM). `q`: [b, n_head, 1, hd] F16; `k`/`v`:
/// [b, n_kv, kv_len, hd] F16 (may be the strided KV-cache narrow); `mask`:
/// `[kv_len]` F32 additive or None; `sinks`: [n_head] F32. Returns `[b, n_head, hd]`
/// F16. Returns `None` (caller falls back) unless CUDA + hd==64 + contiguous hd.
#[allow(clippy::too_many_arguments)]
pub fn gptoss_flash_decode(
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
    gptoss_flash_decode_win(
        q, k, v, mask, sinks, scale, batch, n_head, n_kv, 0, kv_len, head_dim,
    )
}

/// `gptoss_flash_decode` with a `kv_start` offset: attends positions
/// `[kv_start, kv_len)` of the K/V buffers by bumping the base pointers
/// `kv_start` rows - no narrow (a middle-dim narrow is a device COPY of the
/// whole cache on this substrate) and no additive mask. This is how the
/// gpt-oss decode enforces its sliding window (window w ⇒ positions
/// `[kv_len-w, kv_len)` visible): the masked-out prefix contributes exactly
/// 0 to the softmax, so skipping the scan is mathematically identical and
/// O(w) instead of O(kv_len). `mask`, if given, is `[kv_len-kv_start]` F32
/// covering only the scanned rows.
#[allow(clippy::too_many_arguments)]
pub fn gptoss_flash_decode_win(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sinks: Option<&Tensor>,
    scale: f32,
    batch: usize,
    n_head: usize,
    n_kv: usize,
    kv_start: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !q.device().is_cuda()
        || head_dim != 64
        || q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || kv_start >= kv_len
    {
        return Ok(None);
    }
    // The scanned span (kernel-visible kv length). All kernel calls below use
    // this; the base-pointer bump below skips the first kv_start rows.
    let kv_len = kv_len - kv_start;
    let q = q.contiguous()?;
    // k,v keep their (possibly strided) layout; require contiguous innermost (hd) dim.
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
        // kv_start rows skipped via base-pointer bump (row stride = kst/vst[2]).
        let kp = (ksl.device_ptr(ksl.stream()).0
            + ((kl.start_offset() + kv_start * kst[2]) * 2) as u64)
            as *const c_void;
        let vp = (vsl.device_ptr(vsl.stream()).0
            + ((vl.start_offset() + kv_start * vst[2]) * 2) as u64)
            as *const c_void;
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
        // Split-K Flash-Decoding: the single-warp kernel scans the whole kv_len
        // serially (one warp/head -> GPU idle). Fresh nsys (lfm2 decode)
        // measured the serial kernel at 110 µs avg vs the split-K variant at 24 µs
        // (4.6x) across the kv 15-1024 band - i.e. the serial path is slower at ALL
        // but tiny kv, because parallelism (not bandwidth) is the bound. The previous
        // kv>=1024 gate was set conservatively before that profile existed; lower it to
        // 256 so the bulk of a hybrid-MoE decode (lfm2/gpt-oss attn layers) runs the
        // parallel path. Below 256 the serial warp is cheap enough that the split's
        // combine + 2 scratch allocs are not worth it, so it stays on the proven path.
        // nsplit is still held CONSTANT (32) so the per-call F32 scratch is a fixed
        // size - the async mempool reuses one block instead of fragmenting on kv_len.
        // TIERED split count (fresh nsys, gpt-oss long-ctx decode): each split
        // is ONE warp scanning split_len=ceil(kv/nsplit) positions serially, so per-call
        // cost is ∝ split_len. At kv 3.5K the fixed nsplit=32 left ~110 positions/warp =
        // 80 µs/layer - the context-growing part of the decode. More splits -> less serial
        // scan/warp. The kernel uses `nsplit` as the partial-buffer STRIDE, so the scratch
        // must be allocated at exactly `nsplit` (a CAP-sized buffer with a smaller nsplit
        // would alias heads). Tiers {32,64,128} keep that to 3 distinct alloc sizes that a
        // monotonically-growing kv crosses only twice per decode - no mempool fragmentation
        // churn (the original reason nsplit was held constant). kv 256-1024 stays at 32, so
        // the validated lfm2-medium win is bit-identical. Split-K is bit-exact at any tier.
        // Low gate >=64 (was >=256): a fresh nsys (gpt-oss long-ctx decode)
        // measured the serial warp at 27.5 µs/layer for kv=128
        // (the gpt-oss sliding-window span) vs 13+3.3 µs for split-K at
        // kv≈430 - the serial scan is the slower path well below the old 256
        // gate, consistent with the earlier lfm2 nsys (serial slower at ALL
        // but tiny kv). Same nsplit=32 ⇒ same scratch size as the 256 tier
        // (no new mempool bucket).
        // Multi-warp token-per-lane variant (a go/no-go probe): at hd64
        // batch=1 no-sinks/no-mask (lfm2-class; gpt-oss always carries sinks ->
        // stays on the proven path) and kv >= 1024, 4 warps cooperate per
        // (head,split) with half2 loads + smem merge - x1.39 kernel-level over
        // the single-warp split-K at kv 2.5K. nsplit ≈ kv/512 rounded to a power
        // of two (the probe's operating point), so the scratch has few distinct
        // sizes and, like the tiers below, a growing kv crosses each once.
        let use_mw = batch == 1 && sinks_c.is_none() && mask_c.is_none() && kv_len >= 1024;
        let nsplit = if use_mw {
            if kv_len >= 8192 {
                16
            } else if kv_len >= 4096 {
                8
            } else if kv_len >= 2048 {
                4
            } else {
                2
            }
        } else if kv_len >= 3072 {
            128
        } else if kv_len >= 1024 {
            64
        } else if kv_len >= 64 {
            32
        } else {
            1
        };
        if nsplit > 1 {
            // F32 scratch at exactly `nsplit` (matches the kernel's stride).
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
                let pmsl = pmc.as_cuda_slice::<f32>()?;
                let plsl = plc.as_cuda_slice::<f32>()?;
                let pasl = pac.as_cuda_slice::<f32>()?;
                let pmp = pmsl.device_ptr(pmsl.stream()).0 as *mut f32;
                let plp = plsl.device_ptr(plsl.stream()).0 as *mut f32;
                let pap = pasl.device_ptr(pasl.stream()).0 as *mut f32;
                unsafe {
                    if use_mw {
                        // batch==1 -> batch strides are irrelevant; the probe kernel
                        // takes (head, pos) strides only and q contiguous [n_head, hd].
                        tp_mw_flash_decode_split(
                            qp,
                            kp,
                            vp,
                            pmp,
                            plp,
                            pap,
                            op,
                            n_head as i32,
                            n_kv as i32,
                            kv_len as i32,
                            nsplit as i32,
                            head_dim as i32,
                            scale,
                            kst[1] as i64,
                            kst[2] as i64,
                            vst[1] as i64,
                            vst[2] as i64,
                            stream,
                        );
                    } else {
                        gptoss_flash_decode_split_f16(
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
            }
            drop(pms);
            drop(pls);
            drop(pas);
        } else {
            unsafe {
                gptoss_flash_decode_f16(
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
