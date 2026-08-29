//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Mamba2 causal depthwise conv1d + bias + SiLU + state-shift, one launch.
/// x [b, D] F32; state_in [b, D, L]; conv_w [D, L]; conv_b [D]. Returns
/// (y=silu(conv) [b, D], state_out [b, D, L]). CUDA only.
pub fn fused_causal_conv1d_silu(
    x: &Tensor,
    state_in: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    d: usize,
    l: usize,
) -> Result<(Tensor, Tensor)> {
    if !x.device().is_cuda() {
        crate::tensor::bail!("fused_causal_conv1d_silu requires CUDA");
    }
    let bsz = x.dim(0)?;
    let x = x.contiguous()?;
    let state_in = state_in.contiguous()?;
    let conv_w = conv_w.contiguous()?;
    let conv_b = conv_b.contiguous()?;
    let y = unsafe { Tensor::empty((bsz, d), DType::F32, &x.device())? };
    let state_out = unsafe { Tensor::empty((bsz, d, l), DType::F32, &x.device())? };

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_causal_conv1d_silu_f32", "loken_fused", ptx)?;

    let (x_s, x_l) = x.storage_and_layout();
    let (si_s, si_l) = state_in.storage_and_layout();
    let (cw_s, cw_l) = conv_w.storage_and_layout();
    let (cb_s, cb_l) = conv_b.storage_and_layout();
    let (y_s, _) = y.storage_and_layout();
    let (so_s, _) = state_out.storage_and_layout();
    use crate::tensor::StorageView::Cuda as C;
    if let (C(xx), C(si), C(cw), C(cb), C(yy), C(so)) =
        (&*x_s, &*si_s, &*cw_s, &*cb_s, &*y_s, &*so_s)
    {
        let x_v = xx.as_cuda_slice::<f32>()?.slice(x_l.start_offset()..);
        let si_v = si.as_cuda_slice::<f32>()?.slice(si_l.start_offset()..);
        let cw_v = cw.as_cuda_slice::<f32>()?.slice(cw_l.start_offset()..);
        let cb_v = cb.as_cuda_slice::<f32>()?.slice(cb_l.start_offset()..);
        let y_sl = yy.as_cuda_slice::<f32>()?;
        let so_sl = so.as_cuda_slice::<f32>()?;
        let n = (bsz * d) as u32;
        let (block, grid) = adaptive_grid_1d(n, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (bi, di, li) = (bsz as i32, d as i32, l as i32);
        let mut builder = func.builder();
        builder.arg(&x_v);
        builder.arg(&si_v);
        builder.arg(&cw_v);
        builder.arg(&cb_v);
        builder.arg(y_sl);
        builder.arg(so_sl);
        builder.arg(&bi);
        builder.arg(&di);
        builder.arg(&li);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_causal_conv1d_silu: {e}")))?;
    }
    drop(x_s);
    drop(si_s);
    drop(cw_s);
    drop(cb_s);
    drop(y_s);
    drop(so_s);
    Ok((y, state_out))
}

/// Mamba2 single-token SSM recurrence fused into one launch. Inputs (all F32):
/// x_c [b, nh*hd], b_/c_ [b, ngroups*ds], dt [b, nh], a [nh], h_in [b,nh,hd,ds].
/// Returns (y [b,nh,hd], h_out [b,nh,hd,ds]). CUDA only.
pub fn fused_mamba2_ssm_step(
    x_c: &Tensor,
    b_: &Tensor,
    c_: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    h_in: &Tensor,
    nh: usize,
    hd: usize,
    ds: usize,
    ngroups: usize,
) -> Result<(Tensor, Tensor)> {
    if !x_c.device().is_cuda() {
        crate::tensor::bail!("fused_mamba2_ssm_step requires CUDA");
    }
    let bsz = x_c.dim(0)?;
    let x_c = x_c.contiguous()?;
    let b_ = b_.contiguous()?;
    let c_ = c_.contiguous()?;
    let dt = dt.contiguous()?;
    let a = a.contiguous()?;
    let h_in = h_in.contiguous()?;
    let y = unsafe { Tensor::empty((bsz, nh, hd), DType::F32, &x_c.device())? };
    let h_out = unsafe { Tensor::empty((bsz, nh, hd, ds), DType::F32, &x_c.device())? };

    let cuda_dev = x_c.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_mamba2_ssm_step_f32", "loken_fused", ptx)?;

    let (xc_s, xc_l) = x_c.storage_and_layout();
    let (b_s, b_l) = b_.storage_and_layout();
    let (c_s, c_l) = c_.storage_and_layout();
    let (dt_s, dt_l) = dt.storage_and_layout();
    let (a_s, a_l) = a.storage_and_layout();
    let (h_s, h_l) = h_in.storage_and_layout();
    let (y_s, _) = y.storage_and_layout();
    let (ho_s, _) = h_out.storage_and_layout();
    use crate::tensor::StorageView::Cuda as C;
    if let (C(xc), C(bb), C(cc), C(dtt), C(aa), C(hh), C(yy), C(hoo)) =
        (&*xc_s, &*b_s, &*c_s, &*dt_s, &*a_s, &*h_s, &*y_s, &*ho_s)
    {
        let xc_v = xc.as_cuda_slice::<f32>()?.slice(xc_l.start_offset()..);
        let bb_v = bb.as_cuda_slice::<f32>()?.slice(b_l.start_offset()..);
        let cc_v = cc.as_cuda_slice::<f32>()?.slice(c_l.start_offset()..);
        let dt_v = dtt.as_cuda_slice::<f32>()?.slice(dt_l.start_offset()..);
        let a_v = aa.as_cuda_slice::<f32>()?.slice(a_l.start_offset()..);
        let h_v = hh.as_cuda_slice::<f32>()?.slice(h_l.start_offset()..);
        let y_sl = yy.as_cuda_slice::<f32>()?;
        let ho_sl = hoo.as_cuda_slice::<f32>()?;
        let n = (bsz * nh * hd) as u32;
        let (block, grid) = adaptive_grid_1d(n, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (bi, nhi, hdi, dsi, ngi) =
            (bsz as i32, nh as i32, hd as i32, ds as i32, ngroups as i32);
        let mut builder = func.builder();
        builder.arg(&xc_v);
        builder.arg(&bb_v);
        builder.arg(&cc_v);
        builder.arg(&dt_v);
        builder.arg(&a_v);
        builder.arg(&h_v);
        builder.arg(y_sl);
        builder.arg(ho_sl);
        builder.arg(&bi);
        builder.arg(&nhi);
        builder.arg(&hdi);
        builder.arg(&dsi);
        builder.arg(&ngi);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_mamba2_ssm_step: {e}")))?;
    }
    drop(xc_s);
    drop(b_s);
    drop(c_s);
    drop(dt_s);
    drop(a_s);
    drop(h_s);
    drop(y_s);
    drop(ho_s);
    Ok((y, h_out))
}

pub fn fused_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if !gate.device().runs_as_card() {
        return crate::tensor::ops::silu(gate)?.mul(up);
    }

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let n = gate.elem_count();

    // The kernel writes one buffer where the chain above writes two, so which arm a
    // counted run takes decides what it reports. It takes this one, and stops here -
    // before the handle, which a counting device cannot give.
    if gate.device().is_dry() {
        return Tensor::dry(&gate.device(), DType::F32, gate.shape().clone());
    }

    let out = unsafe { Tensor::empty(gate.shape(), DType::F32, &gate.device())? };

    let cuda_dev = gate.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_silu_mul_f32", "loken_fused", ptx)?;

    let (g_store, g_layout) = gate.storage_and_layout();
    let (u_store, u_layout) = up.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(g),
        crate::tensor::StorageView::Cuda(u),
        crate::tensor::StorageView::Cuda(o),
    ) = (&*g_store, &*u_store, &*o_store)
    {
        let g_slice = g.as_cuda_slice::<f32>()?;
        let u_slice = u.as_cuda_slice::<f32>()?;
        let o_slice = o.as_cuda_slice::<f32>()?;
        let g_view = g_slice.slice(g_layout.start_offset()..);
        let u_view = u_slice.slice(u_layout.start_offset()..);

        let (block, grid) = adaptive_grid_1d(n as u32, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let mut builder = func.builder();
        builder.arg(&g_view);
        builder.arg(&u_view);
        builder.arg(o_slice);
        builder.arg(&n_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_silu_mul: {e}")))?;
    }

    drop(g_store);
    drop(u_store);
    drop(o_store);
    Ok(out)
}

/// Launch a 1-input -> 1-output elementwise F32 kernel (signature
/// `(const float* x, float* out, int n)`). Shared by the native silu/sigmoid
/// ops. Caller guarantees CUDA F32.
pub(super) fn launch_elementwise_f32(x: &Tensor, kernel: &str, label: &str) -> Result<Tensor> {
    let x = x.contiguous()?;
    let n = x.elem_count();
    let out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };
    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func(kernel, "loken_fused", ptx)?;
    let (x_store, x_layout) = x.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();
    if let (crate::tensor::StorageView::Cuda(xs), crate::tensor::StorageView::Cuda(os)) =
        (&*x_store, &*o_store)
    {
        let x_slice = xs.as_cuda_slice::<f32>()?;
        let o_slice = os.as_cuda_slice::<f32>()?;
        let x_view = x_slice.slice(x_layout.start_offset()..);
        let (block, grid) = adaptive_grid_1d(n as u32, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(o_slice);
        builder.arg(&n_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("{label}: {e}")))?;
    }
    drop(x_store);
    drop(o_store);
    Ok(out)
}

/// SiLU(x) = x / (1+exp(-x)), CUDA F32. Native replacement for crate::tensor::ops::silu.
pub fn fused_silu_f32(x: &Tensor) -> Result<Tensor> {
    launch_elementwise_f32(x, "silu_f32", "fused_silu_f32")
}
/// Sigmoid(x) = 1/(1+exp(-x)), CUDA F32. Native replacement for crate::tensor::ops::sigmoid.
pub fn fused_sigmoid_f32(x: &Tensor) -> Result<Tensor> {
    launch_elementwise_f32(x, "sigmoid_f32", "fused_sigmoid_f32")
}

/// Launch a RoPE kernel `(x, cos, sin, out, n_rows, d, seq)`. x:[b,h,s,d] F32,
/// cos/sin:[s,d/2] F32. Shared by native rope/rope_i.
pub(super) fn launch_rope_f32(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    kernel: &str,
    label: &str,
) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let n_rows = b * h * s;
    // No `.contiguous()` on x/cos/sin: native tensors are packed or contiguous
    // narrow views, and the views' offsets are applied below via
    // `layout.start_offset()` - zero-copy.
    let out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };
    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func(kernel, "loken_fused", ptx)?;
    let (x_store, x_layout) = x.storage_and_layout();
    let (c_store, c_layout) = cos.storage_and_layout();
    let (s_store, s_layout) = sin.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();
    if let (
        crate::tensor::StorageView::Cuda(xs),
        crate::tensor::StorageView::Cuda(cs),
        crate::tensor::StorageView::Cuda(ss),
        crate::tensor::StorageView::Cuda(os),
    ) = (&*x_store, &*c_store, &*s_store, &*o_store)
    {
        let x_view = xs.as_cuda_slice::<f32>()?.slice(x_layout.start_offset()..);
        let c_view = cs.as_cuda_slice::<f32>()?.slice(c_layout.start_offset()..);
        let s_view = ss.as_cuda_slice::<f32>()?.slice(s_layout.start_offset()..);
        let o_slice = os.as_cuda_slice::<f32>()?;
        let half = (d / 2) as u32;
        let block = 256u32.min(half).max(1).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nr, dd, sq) = (n_rows as i32, d as i32, s as i32);
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(&c_view);
        builder.arg(&s_view);
        builder.arg(o_slice);
        builder.arg(&nr);
        builder.arg(&dd);
        builder.arg(&sq);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("{label}: {e}")))?;
    }
    drop(x_store);
    drop(c_store);
    drop(s_store);
    drop(o_store);
    Ok(out)
}
/// NeoX (non-interleaved) RoPE, CUDA F32. Native crate::tensor::ops::rope.
pub fn fused_rope_neox_f32(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    launch_rope_f32(x, cos, sin, "rope_neox_f32", "fused_rope_neox")
}
/// Interleaved RoPE, CUDA F32. Native crate::tensor::ops::rope_i.
pub fn fused_rope_interleaved_f32(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    launch_rope_f32(
        x,
        cos,
        sin,
        "rope_interleaved_f32",
        "fused_rope_interleaved",
    )
}

/// Fused GELU(tanh-approx)(gate) * up - replaces 2 kernel launches with 1.
/// Mirrors `fused_silu_mul` but with gemma4's `gelu_pytorch_tanh` activation.
/// Both inputs must be F32 contiguous tensors of the same shape.
pub fn fused_gelu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    // CPU F32: single-pass `gelu_tanh(gate) * up`. The Tensor-path `.gelu()` is a
    // multi-op tanh-approx chain (~6 elementwise ops, each a fresh [ffn] alloc)
    // followed by a separate `mul` - ~7 allocs/layer of churn. SiLU models avoid
    // this via the fused `fused_silu_f32`; gelu models (gemma*) had no CPU-fused
    // equivalent and paid the chain every layer. Match the exact gelu-tanh
    // form so output is bit-for-bit equal to `gate.gelu().mul(up)`.
    if !gate.device().runs_as_card() {
        if gate.dtype() == DType::F32 && up.dtype() == DType::F32 && gate.shape() == up.shape() {
            use rayon::prelude::*;
            let g = gate.flatten_all()?.to_vec1::<f32>()?;
            let u = up.flatten_all()?.to_vec1::<f32>()?;
            // EXACT native gelu form (tensor/tensor/mod.rs): 0.5*v*(1 + tanh(0.797_884_56*(v + 0.044715*v³)))
            // - same op order so the fused result is bit-for-bit `gate.gelu().mul(up)`.
            let mut out = vec![0f32; g.len()];
            out.par_iter_mut()
                .zip(g.par_iter().zip(u.par_iter()))
                .for_each(|(o, (&v, &uu))| {
                    let gelu =
                        0.5 * v * (1.0 + (0.797_884_56_f32 * (v + 0.044715 * v * v * v)).tanh());
                    *o = gelu * uu;
                });
            return Tensor::from_vec(out, gate.shape(), &gate.device());
        }
        return gate.gelu()?.mul(up);
    }
    if gate.dtype() != DType::F32 || up.dtype() != DType::F32 {
        return gate.gelu()?.mul(up);
    }
    if gate.shape() != up.shape() {
        return gate.gelu()?.mul(up);
    }

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let n = gate.elem_count();
    // One buffer where the chain above writes two.
    if gate.device().is_dry() {
        return Tensor::dry(&gate.device(), DType::F32, gate.shape().clone());
    }
    let out = unsafe { Tensor::empty(gate.shape(), DType::F32, &gate.device())? };

    let cuda_dev = gate.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_gelu_mul_f32", "loken_fused", ptx)?;

    let (g_store, g_layout) = gate.storage_and_layout();
    let (u_store, u_layout) = up.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(g),
        crate::tensor::StorageView::Cuda(u),
        crate::tensor::StorageView::Cuda(o),
    ) = (&*g_store, &*u_store, &*o_store)
    {
        let g_slice = g.as_cuda_slice::<f32>()?;
        let u_slice = u.as_cuda_slice::<f32>()?;
        let o_slice = o.as_cuda_slice::<f32>()?;
        let g_view = g_slice.slice(g_layout.start_offset()..);
        let u_view = u_slice.slice(u_layout.start_offset()..);

        let (block, grid) = adaptive_grid_1d(n as u32, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let mut builder = func.builder();
        builder.arg(&g_view);
        builder.arg(&u_view);
        builder.arg(o_slice);
        builder.arg(&n_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_gelu_mul: {e}")))?;
    }

    drop(g_store);
    drop(u_store);
    drop(o_store);
    Ok(out)
}

/// Fused split + GELU(tanh-approx) + mul on a packed [M, 2N] gate||up
/// tensor - replaces narrow + contiguous + narrow + contiguous + gelu +
/// mul (~6 kernel launches) with one elementwise kernel.
///
/// Used by the gemma4-MoE forward path on the moe_gemm_gguf gate||up
/// output. Activation is the tanh-approximation GELU
/// (`gelu_pytorch_tanh`), matching what gemma4 uses.
pub fn fused_split_gelu_mul(gu: &Tensor) -> Result<Tensor> {
    let dims = gu.dims();
    if dims.len() < 2 {
        crate::tensor::bail!("fused_split_gelu_mul: input must be >= 2D, got {:?}", dims);
    }
    let two_n = *dims.last().unwrap();
    if !two_n.is_multiple_of(2) {
        crate::tensor::bail!(
            "fused_split_gelu_mul: last dim must be even (got {})",
            two_n
        );
    }
    let n = two_n / 2;
    let leading: usize = dims[..dims.len() - 1].iter().product();

    if !gu.device().runs_as_card() {
        let g = gu.narrow(crate::tensor::D::Minus1, 0, n)?.contiguous()?;
        let u = gu.narrow(crate::tensor::D::Minus1, n, n)?.contiguous()?;
        return (u * g.gelu()?)?.contiguous();
    }

    let gu_c = gu.contiguous()?;
    let mut out_shape = dims.to_vec();
    *out_shape.last_mut().unwrap() = n;
    // One buffer here against the two narrow copies plus a product above - the
    // difference this arm exists for is the difference a reserve has to report.
    if gu.device().is_dry() {
        return Tensor::dry(&gu.device(), DType::F32, out_shape);
    }
    let out = unsafe { Tensor::empty(out_shape.as_slice(), DType::F32, &gu.device())? };

    let cuda_dev = gu.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_split_gelu_mul_f32", "loken_fused", ptx)?;

    let (gu_store, gu_layout) = gu_c.storage_and_layout();
    let (out_store, _) = out.storage_and_layout();

    if let (crate::tensor::StorageView::Cuda(g), crate::tensor::StorageView::Cuda(o)) =
        (&*gu_store, &*out_store)
    {
        let g_slice = g.as_cuda_slice::<f32>()?;
        let o_slice = o.as_cuda_slice::<f32>()?;
        let g_view = g_slice.slice(gu_layout.start_offset()..);

        let total = (leading * n) as u32;
        let (block, grid) = adaptive_grid_1d(total, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let m_i32 = leading as i32;
        let n_i32 = n as i32;
        let mut builder = func.builder();
        builder.arg(&g_view);
        builder.arg(o_slice);
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_split_gelu_mul: {e}")))?;
    }

    drop(gu_store);
    drop(out_store);
    Ok(out)
}

/// SiLU sister of `fused_split_gelu_mul`. Operates on a packed
/// [..., 2N] gate||up tensor and emits [..., N] = silu(gate) * up in
/// one kernel - replaces narrow + (maybe contiguous) + narrow +
/// (maybe contiguous) + fused_silu_mul.
///
/// For decode (T=1) the narrow result is already contiguous so the
/// net launch count is unchanged. For prefill (T>1) the two
/// contiguous() copies are skipped.
pub fn fused_split_silu_mul(gu: &Tensor) -> Result<Tensor> {
    let dims = gu.dims();
    if dims.len() < 2 {
        crate::tensor::bail!("fused_split_silu_mul: input must be >= 2D, got {:?}", dims);
    }
    let two_n = *dims.last().unwrap();
    if !two_n.is_multiple_of(2) {
        crate::tensor::bail!(
            "fused_split_silu_mul: last dim must be even (got {})",
            two_n
        );
    }
    let n = two_n / 2;
    let leading: usize = dims[..dims.len() - 1].iter().product();

    if !gu.device().runs_as_card() {
        let g = gu.narrow(crate::tensor::D::Minus1, 0, n)?.contiguous()?;
        let u = gu.narrow(crate::tensor::D::Minus1, n, n)?.contiguous()?;
        return (u * crate::tensor::ops::silu(&g)?)?.contiguous();
    }

    let gu_c = gu.contiguous()?;
    let mut out_shape = dims.to_vec();
    *out_shape.last_mut().unwrap() = n;
    // One buffer here against the two narrow copies plus a product above - the
    // difference this arm exists for is the difference a reserve has to report.
    if gu.device().is_dry() {
        return Tensor::dry(&gu.device(), DType::F32, out_shape);
    }
    let out = unsafe { Tensor::empty(out_shape.as_slice(), DType::F32, &gu.device())? };

    let cuda_dev = gu.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_split_silu_mul_f32", "loken_fused", ptx)?;

    let (gu_store, gu_layout) = gu_c.storage_and_layout();
    let (out_store, _) = out.storage_and_layout();

    if let (crate::tensor::StorageView::Cuda(g), crate::tensor::StorageView::Cuda(o)) =
        (&*gu_store, &*out_store)
    {
        let g_slice = g.as_cuda_slice::<f32>()?;
        let o_slice = o.as_cuda_slice::<f32>()?;
        let g_view = g_slice.slice(gu_layout.start_offset()..);

        let total = (leading * n) as u32;
        let (block, grid) = adaptive_grid_1d(total, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let m_i32 = leading as i32;
        let n_i32 = n as i32;
        let mut builder = func.builder();
        builder.arg(&g_view);
        builder.arg(o_slice);
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_split_silu_mul: {e}")))?;
    }

    drop(gu_store);
    drop(out_store);
    Ok(out)
}

/// Fused (x + residual) + RmsNorm with dual output - replaces 2 kernel launches with 1.
/// Returns (sum, norm) where sum = x + residual, norm = rmsnorm(sum, weight, eps).
/// Gemma4 fused: y=rmsnorm(attn_out, w_post); x=y+residual; x_norm=rmsnorm(x, w_ffn).
/// Returns (x, x_norm). Replaces 3 launches (post_norm + add + ffn_norm) with 1.
/// All tensors F32 contiguous on CUDA; weights 1-D [cols].
pub fn fused_gemma4_post_add_norm(
    attn_out: &Tensor,
    residual: &Tensor,
    w_post: &Tensor,
    w_ffn: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    if !attn_out.device().runs_as_card() {
        let y = crate::tensor::ops::rms_norm(attn_out, w_post, eps)?;
        let x = (&y + residual)?;
        let x_norm = crate::tensor::ops::rms_norm(&x, w_ffn, eps)?;
        return Ok((x, x_norm));
    }

    let attn_out = attn_out.contiguous()?;
    let residual = residual.contiguous()?;
    let w_post = w_post.contiguous()?;
    let w_ffn = w_ffn.contiguous()?;

    let total_elems = attn_out.elem_count();
    let cols = *attn_out.dims().last().unwrap_or(&1);
    let rows = total_elems / cols;

    // Two buffers here against the three the chain above writes.
    if attn_out.device().is_dry() {
        let d = attn_out.device();
        return Ok((
            Tensor::dry(&d, DType::F32, attn_out.shape().clone())?,
            Tensor::dry(&d, DType::F32, attn_out.shape().clone())?,
        ));
    }
    let x_out = unsafe { Tensor::empty(attn_out.shape(), DType::F32, &attn_out.device())? };
    let x_norm_out = unsafe { Tensor::empty(attn_out.shape(), DType::F32, &attn_out.device())? };

    let cuda_dev = attn_out.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_gemma4_post_add_norm_f32", "loken_fused", ptx)?;

    let (a_store, a_layout) = attn_out.storage_and_layout();
    let (r_store, r_layout) = residual.storage_and_layout();
    let (wp_store, _) = w_post.storage_and_layout();
    let (wf_store, _) = w_ffn.storage_and_layout();
    let (xo_store, _) = x_out.storage_and_layout();
    let (xn_store, _) = x_norm_out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(a_s),
        crate::tensor::StorageView::Cuda(r_s),
        crate::tensor::StorageView::Cuda(wp_s),
        crate::tensor::StorageView::Cuda(wf_s),
        crate::tensor::StorageView::Cuda(xo_s),
        crate::tensor::StorageView::Cuda(xn_s),
    ) = (
        &*a_store, &*r_store, &*wp_store, &*wf_store, &*xo_store, &*xn_store,
    ) {
        let a_slice = a_s.as_cuda_slice::<f32>()?;
        let r_slice = r_s.as_cuda_slice::<f32>()?;
        let wp_slice = wp_s.as_cuda_slice::<f32>()?;
        let wf_slice = wf_s.as_cuda_slice::<f32>()?;
        let xo_slice = xo_s.as_cuda_slice::<f32>()?;
        let xn_slice = xn_s.as_cuda_slice::<f32>()?;
        let a_view = a_slice.slice(a_layout.start_offset()..);
        let r_view = r_slice.slice(r_layout.start_offset()..);

        let block = 256u32.min(cols as u32).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };

        let cols_i32 = cols as i32;
        let mut builder = func.builder();
        builder.arg(&a_view);
        builder.arg(&r_view);
        builder.arg(wp_slice);
        builder.arg(wf_slice);
        builder.arg(xo_slice);
        builder.arg(xn_slice);
        builder.arg(&eps);
        builder.arg(&cols_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_gemma4_post_add_norm: {e}")))?;
    }
    drop(a_store);
    drop(r_store);
    drop(wp_store);
    drop(wf_store);
    drop(xo_store);
    drop(xn_store);
    Ok((x_out, x_norm_out))
}

pub fn fused_add_rmsnorm_dual(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    if !x.device().runs_as_card() {
        // One-pass CPU fusion (add + rmsnorm in a single pooled pass) when the
        // inputs are full-contiguous F32 - saves a pool.run barrier and a full
        // read+write pass over the [rows, hidden] intermediate on the prefill
        // critical path. Bit-identical; falls back to the two-pass chain otherwise.
        if let Some(res) = crate::tensor::ops::fused_add_rmsnorm_f32(x, residual, weight, eps)? {
            return Ok(res);
        }
        let sum = (x + residual)?;
        let norm = crate::tensor::ops::rms_norm(&sum, weight, eps)?;
        return Ok((sum, norm));
    }

    let x = x.contiguous()?;
    let residual = residual.contiguous()?;

    let total_elems = x.elem_count();
    let cols = *x.dims().last().unwrap_or(&1);
    let rows = total_elems / cols;

    // Two buffers, which is what the two-pass chain above also leaves - but they are
    // written in one pass over the intermediate, and a counted run must take the arm
    // that will actually run.
    if x.device().is_dry() {
        let d = x.device();
        return Ok((
            Tensor::dry(&d, DType::F32, x.shape().clone())?,
            Tensor::dry(&d, DType::F32, x.shape().clone())?,
        ));
    }
    let sum_out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };
    let norm_out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_add_rmsnorm_dual_f32", "loken_fused", ptx)?;

    let (x_store, x_layout) = x.storage_and_layout();
    let (r_store, r_layout) = residual.storage_and_layout();
    let (w_store, _) = weight.storage_and_layout();
    let (s_store, _) = sum_out.storage_and_layout();
    let (n_store, _) = norm_out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(xs),
        crate::tensor::StorageView::Cuda(rs),
        crate::tensor::StorageView::Cuda(ws),
        crate::tensor::StorageView::Cuda(ss),
        crate::tensor::StorageView::Cuda(ns),
    ) = (&*x_store, &*r_store, &*w_store, &*s_store, &*n_store)
    {
        let x_slice = xs.as_cuda_slice::<f32>()?;
        let r_slice = rs.as_cuda_slice::<f32>()?;
        let w_slice = ws.as_cuda_slice::<f32>()?;
        let s_slice = ss.as_cuda_slice::<f32>()?;
        let n_slice = ns.as_cuda_slice::<f32>()?;
        let x_view = x_slice.slice(x_layout.start_offset()..);
        let r_view = r_slice.slice(r_layout.start_offset()..);

        let block = 256u32.min(cols as u32).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };

        let cols_i32 = cols as i32;
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(&r_view);
        builder.arg(w_slice);
        builder.arg(s_slice);
        builder.arg(n_slice);
        builder.arg(&eps);
        builder.arg(&cols_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_add_rmsnorm_dual: {e}")))?;
    }

    drop(x_store);
    drop(r_store);
    drop(w_store);
    drop(s_store);
    drop(n_store);
    Ok((sum_out, norm_out))
}

/// Fused `rmsnorm(x, weight) + residual` in 1 launch.
///
/// Gemma4 lays a `post_attn_norm` / `post_ffn_norm` on top of every
/// transformer layer, then adds the residual - 2 launches per call.
/// Over 30 layers that's 60 launches per token on each pattern; saving
/// 30 launches/token closes ~3 tok/s on gemma4:latest medium (the
/// flip margin per project_gemma4_latest_medium_profile_2026_05_22).
///
/// Falls back to the unfused `rms_norm(x, w, eps) + residual` chain on
/// CPU / non-F32 inputs. All CUDA tensors must be F32 contiguous.
pub fn fused_rmsnorm_then_add(
    x: &Tensor,
    weight: &Tensor,
    residual: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if !x.device().runs_as_card() {
        let norm = crate::tensor::ops::rms_norm(x, weight, eps)?;
        return Ok((norm + residual)?);
    }
    if x.dtype() != DType::F32 || residual.dtype() != DType::F32 || weight.dtype() != DType::F32 {
        let norm = crate::tensor::ops::rms_norm(x, weight, eps)?;
        return Ok((norm + residual)?);
    }
    if x.shape() != residual.shape() {
        let norm = crate::tensor::ops::rms_norm(x, weight, eps)?;
        return Ok(norm.broadcast_add(residual)?);
    }

    let x = x.contiguous()?;
    let residual = residual.contiguous()?;

    let total_elems = x.elem_count();
    let cols = *x.dims().last().unwrap_or(&1);
    let rows = total_elems / cols;

    // Uninitialized alloc - every output element is written by the kernel
    // body (out[offset + i] = ... loop covers i ∈ [0, cols) for every row).
    // Skipping cudaMemsetAsync removes the second launch that previously
    // made the fuse net-negative at post_attn_norm and PLE post_norm
    // sites - the regressions were measured at both, in the generic transformer.
    // One buffer where the norm-then-add chain above writes two.
    if x.device().is_dry() {
        return Tensor::dry(&x.device(), DType::F32, x.shape().clone());
    }
    let out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_rmsnorm_then_add_f32", "loken_fused", ptx)?;

    let (x_store, x_layout) = x.storage_and_layout();
    let (r_store, r_layout) = residual.storage_and_layout();
    let (w_store, _) = weight.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(xs),
        crate::tensor::StorageView::Cuda(rs),
        crate::tensor::StorageView::Cuda(ws),
        crate::tensor::StorageView::Cuda(os),
    ) = (&*x_store, &*r_store, &*w_store, &*o_store)
    {
        let x_slice = xs.as_cuda_slice::<f32>()?;
        let r_slice = rs.as_cuda_slice::<f32>()?;
        let w_slice = ws.as_cuda_slice::<f32>()?;
        let o_slice = os.as_cuda_slice::<f32>()?;
        let x_view = x_slice.slice(x_layout.start_offset()..);
        let r_view = r_slice.slice(r_layout.start_offset()..);

        let block = 256u32.min(cols as u32).next_power_of_two();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };

        let cols_i32 = cols as i32;
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(w_slice);
        builder.arg(&r_view);
        builder.arg(o_slice);
        builder.arg(&eps);
        builder.arg(&cols_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_rmsnorm_then_add: {e}")))?;
    }

    drop(x_store);
    drop(r_store);
    drop(w_store);
    drop(o_store);
    Ok(out)
}

/// Standalone RMSNorm on the CUDA F32 path: `out = x / sqrt(mean(x^2)+eps) * weight`.
/// Native replacement for `crate::tensor::ops::rms_norm`.
/// Strict preconditions (caller dispatches the fallback otherwise): CUDA, F32
/// x + weight, weight.dims() == [last(x)]. Returns a fresh F32 tensor.
pub fn fused_rmsnorm_f32(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let cols = *x.dims().last().unwrap_or(&1);
    let rows = x.elem_count() / cols;
    let out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    // Wide-block path for the few-rows (decode / small-batch verify) case:
    // with one block per row, a 256-thread block under-fills the SM and the
    // narrow kernel reads the row from global memory twice - measured ~3x
    // the per-row latency of the staged wide variant. Large-rows launches
    // keep the narrow kernel (one small block per row scales across SMs).
    // Both kernels share the exact same accumulation order -> bit-identical.
    let wide = rows <= 64 && cols >= 256 && cols * 4 + 1024 <= 48 * 1024;
    let kname = if wide {
        "fused_rmsnorm_wide_f32"
    } else {
        "fused_rmsnorm_f32"
    };
    let func = cuda_dev.get_or_load_custom_func(kname, "loken_fused", ptx)?;

    let (x_store, x_layout) = x.storage_and_layout();
    let (w_store, _) = weight.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();
    if let (
        crate::tensor::StorageView::Cuda(xs),
        crate::tensor::StorageView::Cuda(ws),
        crate::tensor::StorageView::Cuda(os),
    ) = (&*x_store, &*w_store, &*o_store)
    {
        let x_slice = xs.as_cuda_slice::<f32>()?;
        let w_slice = ws.as_cuda_slice::<f32>()?;
        let o_slice = os.as_cuda_slice::<f32>()?;
        let x_view = x_slice.slice(x_layout.start_offset()..);
        let cfg = if wide {
            crate::tensor::cuda_ext::LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: ((cols as u32).next_power_of_two().clamp(256, 1024), 1, 1),
                shared_mem_bytes: (cols as u32) * 4,
            }
        } else {
            let block = 256u32.min(cols as u32).max(1).next_power_of_two();
            crate::tensor::cuda_ext::LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: block * 4,
            }
        };
        let cols_i32 = cols as i32;
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(w_slice);
        builder.arg(o_slice);
        builder.arg(&eps);
        builder.arg(&cols_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_rmsnorm_f32: {e}")))?;
    }
    drop(x_store);
    drop(w_store);
    drop(o_store);
    Ok(out)
}
