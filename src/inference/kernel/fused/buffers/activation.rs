//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// out = relu(f32(x))² in one F16 launch (F32-internal -> bit-identical to the
/// the x.to_f32().relu().sqr().to_f16() chain). x F16 -> out F16. Returns None
/// (caller falls back) unless CUDA + F16.
pub fn relu2_f16(x: &Tensor) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !x.device().is_cuda() || x.dtype() != DType::F16 {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let n = x.elem_count() as i64;
    let out = unsafe { Tensor::empty(x.shape(), DType::F16, &x.device())? };
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (xs, xl) = x.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(xc), C(oc)) = (&*xs, &*os) {
        let xsl = xc.as_cuda_slice::<half::f16>()?;
        let osl = oc.as_cuda_slice::<half::f16>()?;
        let xp = (xsl.device_ptr(xsl.stream()).0 + (xl.start_offset() * 2) as u64) as *const c_void;
        let op = osl.device_ptr(osl.stream()).0 as *mut c_void;
        unsafe {
            loken_relu2_f16(xp, op, n, stream);
        }
    }
    drop(xs);
    drop(os);
    Ok(Some(out))
}

/// MoE shared-expert output epilogue: out = routed + f32(down).sigmoid(gate_logit)
/// in one launch (sigmoid + f16->f32 cast + per-token broadcast-mul + add). All F32
/// -> bit-identical. routed F32 [n,hidden], down F16 [n,hidden], gate_logit F32
/// [n] (or [n,1], pre-sigmoid). Returns F32 [n,hidden], or None (caller falls back).
pub fn fused_shexp_out(
    routed: &Tensor,
    down: &Tensor,
    gate_logit: &Tensor,
) -> Result<Option<Tensor>> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !routed.device().is_cuda() || routed.dtype() != DType::F32 || down.dtype() != DType::F16 {
        return Ok(None);
    }
    let (n_tokens, hidden) = routed.dims2()?;
    let routed = routed.contiguous()?;
    let down = down.contiguous()?;
    let gate_logit = gate_logit.flatten_all()?.contiguous()?; // [n]
    let out = unsafe { Tensor::empty((n_tokens, hidden), DType::F32, &routed.device())? };
    let dev = routed.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (rs, rl) = routed.storage_and_layout();
    let (ds, dl) = down.storage_and_layout();
    let (gs, gl) = gate_logit.storage_and_layout();
    let (os, _) = out.storage_and_layout();
    if let (C(rc), C(dc), C(gc), C(oc)) = (&*rs, &*ds, &*gs, &*os) {
        let rsl = rc.as_cuda_slice::<f32>()?;
        let dsl = dc.as_cuda_slice::<half::f16>()?;
        let gsl = gc.as_cuda_slice::<f32>()?;
        let osl = oc.as_cuda_slice::<f32>()?;
        let rp = (rsl.device_ptr(rsl.stream()).0 + (rl.start_offset() * 4) as u64) as *const f32;
        let dp = (dsl.device_ptr(dsl.stream()).0 + (dl.start_offset() * 2) as u64) as *const c_void;
        let gp = (gsl.device_ptr(gsl.stream()).0 + (gl.start_offset() * 4) as u64) as *const f32;
        let op = osl.device_ptr(osl.stream()).0 as *mut f32;
        unsafe {
            loken_fused_shexp_out(rp, dp, gp, op, n_tokens as i32, hidden as i32, stream);
        }
    }
    drop(rs);
    drop(ds);
    drop(gs);
    drop(os);
    Ok(Some(out))
}

/// nvcc-compiled F16 fused residual-add + RMSNorm (cuda/fused_norm_f16.cu). The
/// NVRTC FUSED_CUDA_SRC path is F32-only, so this uses the build.rs/FFI route.
/// x/residual [.., cols] F16; weight [cols] F32. Returns (sum, norm) both F16,
/// where sum = f16(x+residual) (the kept residual) and norm = rmsnorm(sum).weight.
pub fn fused_add_rmsnorm_f16(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !x.device().is_cuda() {
        let sum = (x + residual)?;
        let norm = crate::tensor::ops::rms_norm(&sum.to_dtype(DType::F32)?, weight, eps)?
            .to_dtype(x.dtype())?;
        return Ok((sum, norm));
    }
    let x = x.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let cols = *x.dims().last().unwrap();
    let rows = (x.elem_count() / cols) as i32;
    let norm_out = unsafe { Tensor::empty(x.shape(), DType::F16, &x.device())? };
    let sum_out = unsafe { Tensor::empty(x.shape(), DType::F16, &x.device())? };
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    let (xs, xl) = x.storage_and_layout();
    let (rs, rl) = residual.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (ns, _) = norm_out.storage_and_layout();
    let (ss, _) = sum_out.storage_and_layout();
    if let (C(xc), C(rc), C(wc), C(nc), C(sc)) = (&*xs, &*rs, &*ws, &*ns, &*ss) {
        let xsl = xc.as_cuda_slice::<half::f16>()?;
        let rsl = rc.as_cuda_slice::<half::f16>()?;
        let wsl = wc.as_cuda_slice::<f32>()?;
        let nsl = nc.as_cuda_slice::<half::f16>()?;
        let ssl = sc.as_cuda_slice::<half::f16>()?;
        let xp = (xsl.device_ptr(xsl.stream()).0 + (xl.start_offset() * 2) as u64) as *const c_void;
        let rp = (rsl.device_ptr(rsl.stream()).0 + (rl.start_offset() * 2) as u64) as *const c_void;
        let wp = (wsl.device_ptr(wsl.stream()).0 + (wl.start_offset() * 4) as u64) as *const f32;
        let np = nsl.device_ptr(nsl.stream()).0 as *mut c_void;
        let sp = ssl.device_ptr(ssl.stream()).0 as *mut c_void;
        unsafe {
            loken_fused_rmsnorm_f16(xp, rp, wp, np, sp, rows, cols as i32, eps, stream);
        }
    }
    drop(xs);
    drop(rs);
    drop(ws);
    drop(ns);
    drop(ss);
    Ok((sum_out, norm_out))
}

/// nvcc-compiled F16 RMSNorm with NO residual (the `res == nullptr` path of the
/// same kernel). Replaces cast-F32 + rms_norm + cast-F16 (3 launches) with one
/// for pre-attn / pre-FFN norms that have no residual to fold. x [.., cols] F16;
/// weight [cols] F32. Returns norm = rmsnorm(x).weight in F16.
pub fn fused_rmsnorm_f16(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !x.device().is_cuda() {
        return crate::tensor::ops::rms_norm(&x.to_dtype(DType::F32)?, weight, eps)?
            .to_dtype(x.dtype());
    }
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let cols = *x.dims().last().unwrap();
    let rows = (x.elem_count() / cols) as i32;
    let norm_out = unsafe { Tensor::empty(x.shape(), DType::F16, &x.device())? };
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (ns, _) = norm_out.storage_and_layout();
    if let (C(xc), C(wc), C(nc)) = (&*xs, &*ws, &*ns) {
        let xsl = xc.as_cuda_slice::<half::f16>()?;
        let wsl = wc.as_cuda_slice::<f32>()?;
        let nsl = nc.as_cuda_slice::<half::f16>()?;
        let xp = (xsl.device_ptr(xsl.stream()).0 + (xl.start_offset() * 2) as u64) as *const c_void;
        let wp = (wsl.device_ptr(wsl.stream()).0 + (wl.start_offset() * 4) as u64) as *const f32;
        let np = nsl.device_ptr(nsl.stream()).0 as *mut c_void;
        // res = null, sum_out = null -> kernel reads x directly, writes only norm.
        unsafe {
            loken_fused_rmsnorm_f16(
                xp,
                core::ptr::null(),
                wp,
                np,
                core::ptr::null_mut(),
                rows,
                cols as i32,
                eps,
                stream,
            );
        }
    }

    drop(xs);
    drop(ws);
    drop(ns);
    Ok(norm_out)
}

/// Fused F16 RMSNorm with F32 output = `fused_rmsnorm_f16(x,..)` followed by
/// `.to_dtype(F32)`, in ONE launch (the kernel rounds through F16 before the
/// F32 store, so it is BIT-IDENTICAL to that two-launch chain). For F32-input
/// consumers of a norm over F16 activations (lfm2 MoE FFN).
pub fn fused_rmsnorm_f16_out_f32(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use crate::tensor::cuda_ext::DevicePtr;
    use crate::tensor::StorageView::Cuda as C;
    use core::ffi::c_void;
    if !x.device().is_cuda() || x.dtype() != DType::F16 {
        return crate::tensor::ops::rms_norm(&x.to_dtype(DType::F32)?, weight, eps)?
            .to_dtype(x.dtype())?
            .to_dtype(DType::F32);
    }
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let cols = *x.dims().last().unwrap();
    let rows = (x.elem_count() / cols) as i32;
    let norm_out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (ns, _) = norm_out.storage_and_layout();
    if let (C(xc), C(wc), C(nc)) = (&*xs, &*ws, &*ns) {
        let xsl = xc.as_cuda_slice::<half::f16>()?;
        let wsl = wc.as_cuda_slice::<f32>()?;
        let nsl = nc.as_cuda_slice::<f32>()?;
        let xp = (xsl.device_ptr(xsl.stream()).0 + (xl.start_offset() * 2) as u64) as *const c_void;
        let wp = (wsl.device_ptr(wsl.stream()).0 + (wl.start_offset() * 4) as u64) as *const f32;
        let np = nsl.device_ptr(nsl.stream()).0 as *mut c_void;
        unsafe {
            loken_fused_rmsnorm_f16_out_f32(xp, wp, np, rows, cols as i32, eps, stream);
        }
    }
    drop(xs);
    drop(ws);
    drop(ns);
    Ok(norm_out)
}

/// Adaptive 1D grid sizing for element-wise kernels (1 thread -> 1 element).
/// When the natural grid is below the SM count, halve block_dim and double
/// grid_dim until SMs are saturated. Caps: block_dim >= 64 (2 warps), max
/// 4x expansion. Source: SGLang Blackwell NVFP4 MoE blog adaptive launcher.
pub(super) fn adaptive_grid_1d(n: u32, sm_count: u32) -> (u32, u32) {
    const MIN_BLOCK: u32 = 64;
    const MAX_EXPANSION: u32 = 4;
    let mut block = 256u32;
    let mut grid = n.div_ceil(block);
    let mut expansion = 1u32;
    while grid < sm_count && block > MIN_BLOCK && expansion < MAX_EXPANSION {
        block /= 2;
        grid = n.div_ceil(block);
        expansion *= 2;
    }
    (block, grid)
}

/// CUDA source for fused kernels
/// The production fused-kernel CUDA source (native tensor substrate binds the
/// same kernels - see tensor::cuda tests).
pub(crate) fn fused_cuda_src() -> &'static str {
    FUSED_CUDA_SRC
}

pub(super) const FUSED_CUDA_SRC: &str = include_str!("../../../cuda/fused.cu");

/// Compiled PTX per NVRTC target architecture. On a mixed machine each card must load
/// PTX compiled for its own capability - one card's code JIT'd onto another is the
/// silent-garbage class the quantized module already guards against.
pub(super) static COMPILED_PTX: OnceLock<
    std::sync::Mutex<std::collections::HashMap<&'static str, &'static str>>,
> = OnceLock::new();

/// Pre-warm all custom fused kernels on the given CUDA device: compile
/// the PTX module (via NVRTC) and pre-load each kernel function. After
/// this, subsequent `get_or_load_custom_func` calls hit the cached path
/// (no driver-level module load, no NVRTC). Critical for CUDA graph
/// capture safety - JIT/module-load during capture can break it.
///
/// Idempotent. Safe to call multiple times.
pub fn prewarm_fused_kernels(cuda_dev: &crate::tensor::cuda_ext::RawCudaDevice) -> Result<()> {
    let ptx = get_ptx(&cuda_dev)?;
    // Touching get_or_load_custom_func once per kernel name forces the
    // module into the device's custom_modules cache.
    let names = [
        "fused_silu_mul_f32",
        "fused_gelu_mul_f32",
        "fused_split_gelu_mul_f32",
        "fused_split_silu_mul_f32",
        "fused_add_rmsnorm_dual_f32",
        "fused_rmsnorm_wide_f32",
        "fused_add_rmsnorm_dual_wide_f32",
        "fused_dual_rmsnorm_add_f32",
        "fused_gemma4_post_add_norm_f32",
        "fused_rmsnorm_then_add_f32",
        "fused_rmsnorm_add_scale_f32",
        "fused_penalty_argmax_f32",
        "fused_penalty_argmax_f32_u32_out",
        "fused_penalty_argmax_block_f32",
        "fused_penalty_argmax_final_f32",
        "fused_penalty_argmax_final_f32_u32_out",
        "fused_add_three_f32",
        "fused_bias_gelu_new_f32",
        "fused_phi2_residual_merge_f32",
        "fused_attn_decode_f32_hd512",
        "fused_softmax_sinks_f32",
        "fused_gptoss_mask_f32",
    ];
    for name in names {
        // Errors are non-fatal: a missing symbol just means this kernel
        // isn't in the current PTX build, which is fine - we'll fall
        // back to non-fused paths at use site.
        let _ = cuda_dev.get_or_load_custom_func(name, "loken_fused", ptx);
    }
    Ok(())
}

pub(super) fn get_ptx(cuda_dev: &crate::tensor::cuda_ext::RawCudaDevice) -> Result<&'static str> {
    // Same arch rule as every other NVRTC site: target THIS card, never generic PTX,
    // and never a sibling card's PTX on a mixed machine.
    let arch = crate::inference::quantized_cuda::nvrtc_arch_of(cuda_dev)
        .or_else(crate::inference::quantized_cuda::nvrtc_arch)
        .unwrap_or("generic");
    let map = COMPILED_PTX.get_or_init(Default::default);
    let mut g = map.lock().unwrap_or_else(|e| e.into_inner());
    let ptx: &'static str = match g.get(arch) {
        Some(p) => p,
        None => {
            let opts = cudarc::nvrtc::CompileOptions {
                arch: (arch != "generic").then_some(arch),
                ..Default::default()
            };
            let src = match cudarc::nvrtc::compile_ptx_with_opts(FUSED_CUDA_SRC, opts) {
                Ok(ptx) => ptx.to_src(),
                Err(e) => {
                    tracing::error!("NVRTC compile failed for {arch}: {e}");
                    String::new()
                }
            };
            let leaked: &'static str = Box::leak(src.into_boxed_str());
            g.insert(arch, leaked);
            leaked
        }
    };
    if ptx.is_empty() {
        return Err(crate::tensor::Error::msg(
            "Fused CUDA kernels failed to compile".to_string(),
        ));
    }
    Ok(ptx)
}

/// Fused three-way elementwise add: returns `a + b + c` in one launch
/// (vs the default `(a + b)? + c` which uses 2 launches and allocates an
/// intermediate tensor). Falls back to the unfused chain if any input
/// isn't F32 or isn't on CUDA, or if shapes disagree. Used by the
/// parallel-attn (phi2 / gpt-neox) forward where the residual +
/// attn_out + ffn_out merge happens once per layer per token.
pub fn fused_add_three(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    let supported = a.device().is_cuda()
        && b.device().is_cuda()
        && c.device().is_cuda()
        && a.dtype() == DType::F32
        && b.dtype() == DType::F32
        && c.dtype() == DType::F32
        && a.shape() == b.shape()
        && a.shape() == c.shape();
    if !supported {
        return (a + b)? + c;
    }

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = c.contiguous()?;
    let n = a.elem_count();
    let out = unsafe { Tensor::empty(a.shape(), DType::F32, &a.device())? };

    let cuda_dev = a.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_add_three_f32", "loken_fused", ptx)?;

    let (a_store, a_layout) = a.storage_and_layout();
    let (b_store, b_layout) = b.storage_and_layout();
    let (c_store, c_layout) = c.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(a_s),
        crate::tensor::StorageView::Cuda(b_s),
        crate::tensor::StorageView::Cuda(c_s),
        crate::tensor::StorageView::Cuda(o_s),
    ) = (&*a_store, &*b_store, &*c_store, &*o_store)
    {
        let a_slice = a_s.as_cuda_slice::<f32>()?;
        let b_slice = b_s.as_cuda_slice::<f32>()?;
        let c_slice = c_s.as_cuda_slice::<f32>()?;
        let o_slice = o_s.as_cuda_slice::<f32>()?;
        let a_view = a_slice.slice(a_layout.start_offset()..);
        let b_view = b_slice.slice(b_layout.start_offset()..);
        let c_view = c_slice.slice(c_layout.start_offset()..);

        let (block, grid) = adaptive_grid_1d(n as u32, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let mut builder = func.builder();
        builder.arg(&a_view);
        builder.arg(&b_view);
        builder.arg(&c_view);
        builder.arg(o_slice);
        builder.arg(&n_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_add_three: {e}")))?;
    }

    drop(a_store);
    drop(b_store);
    drop(c_store);
    drop(o_store);
    Ok(out)
}

/// Fused `gelu_new(x + bias)` - replaces `x.broadcast_add(bias)?.gelu()?`
/// (2 launches + 1 intermediate alloc) with one fused elementwise launch.
/// `bias` must be a 1-D F32 tensor whose length equals the trailing dim
/// of `x`. Falls back to the unfused chain for non-CUDA / non-F32 inputs
/// or any shape disagreement.
///
/// Used by the phi2 simple-FFN path on the ffn_up output where the
/// up_bias broadcast-add is followed immediately by gelu before the
/// down projection.
pub fn fused_bias_gelu_new(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let x_dims = x.dims();
    let supported = x.device().is_cuda()
        && bias.device().is_cuda()
        && x.dtype() == DType::F32
        && bias.dtype() == DType::F32
        && bias.dims().len() == 1
        && x_dims.last().copied() == Some(bias.dims()[0]);
    if !supported {
        return x.broadcast_add(bias)?.gelu();
    }

    let x = x.contiguous()?;
    let bias = bias.contiguous()?;
    let n = x.elem_count();
    let bias_dim = bias.dims()[0];

    let out = unsafe { Tensor::empty(x.shape(), DType::F32, &x.device())? };

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_bias_gelu_new_f32", "loken_fused", ptx)?;

    let (x_store, x_layout) = x.storage_and_layout();
    let (b_store, b_layout) = bias.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(x_s),
        crate::tensor::StorageView::Cuda(b_s),
        crate::tensor::StorageView::Cuda(o_s),
    ) = (&*x_store, &*b_store, &*o_store)
    {
        let x_slice = x_s.as_cuda_slice::<f32>()?;
        let b_slice = b_s.as_cuda_slice::<f32>()?;
        let o_slice = o_s.as_cuda_slice::<f32>()?;
        let x_view = x_slice.slice(x_layout.start_offset()..);
        let b_view = b_slice.slice(b_layout.start_offset()..);

        let (block, grid) = adaptive_grid_1d(n as u32, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let bias_dim_i32 = bias_dim as i32;
        let mut builder = func.builder();
        builder.arg(&x_view);
        builder.arg(&b_view);
        builder.arg(o_slice);
        builder.arg(&n_i32);
        builder.arg(&bias_dim_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_bias_gelu_new: {e}")))?;
    }

    drop(x_store);
    drop(b_store);
    drop(o_store);
    Ok(out)
}

/// Stream-parameterized variant of `fused_bias_gelu_new`. Launches the
/// fused elementwise kernel on the supplied stream and allocates the
/// output on that stream too. Used for the alt-stream FFN_up->gelu chain
/// where the caller is overlapping FFN with the attn chain.
///
/// All preconditions are the same as `fused_bias_gelu_new`. No fallback
/// - the caller is responsible for only invoking this on shapes that
/// match.
#[cfg(feature = "cuda")]
pub fn fused_bias_gelu_new_on_stream(
    x: &Tensor,
    bias: &Tensor,
    stream: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
) -> Result<Tensor> {
    if !(x.device().is_cuda()
        && bias.device().is_cuda()
        && x.dtype() == DType::F32
        && bias.dtype() == DType::F32
        && bias.dims().len() == 1
        && x.dims().last().copied() == Some(bias.dims()[0]))
    {
        crate::tensor::bail!("fused_bias_gelu_new_on_stream: precondition mismatch");
    }
    let x = x.contiguous()?;
    let bias = bias.contiguous()?;
    let n = x.elem_count();
    let bias_dim = bias.dims()[0];

    let cuda_dev = x.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_bias_gelu_new_f32", "loken_fused", ptx)?;

    let (x_store, x_layout) = x.storage_and_layout();
    let (b_store, b_layout) = bias.storage_and_layout();
    let (x_s, b_s) = match (&*x_store, &*b_store) {
        (crate::tensor::StorageView::Cuda(xs), crate::tensor::StorageView::Cuda(bs)) => (xs, bs),
        _ => crate::tensor::bail!("fused_bias_gelu_new_on_stream: inputs not on CUDA"),
    };
    let x_view = x_s.as_cuda_slice::<f32>()?.slice(x_layout.start_offset()..);
    let b_view = b_s.as_cuda_slice::<f32>()?.slice(b_layout.start_offset()..);

    // Allocate output on the supplied stream (NOT default).
    let out_slice = unsafe { stream.alloc::<f32>(n) }
        .map_err(|e| crate::tensor::Error::msg(format!("alt-stream alloc: {e}")))?;

    let (block, grid) = adaptive_grid_1d(n as u32, cuda_dev.multiprocessor_count() as u32);
    let cfg = crate::tensor::cuda_ext::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i32 = n as i32;
    let bias_dim_i32 = bias_dim as i32;
    // Launch on the supplied stream via the bare CudaFunction.
    let mut builder = stream.launch_builder(func.cuda_function());
    builder.arg(&x_view);
    builder.arg(&b_view);
    builder.arg(&out_slice);
    builder.arg(&n_i32);
    builder.arg(&bias_dim_i32);
    unsafe { builder.launch(cfg) }.map_err(|e| {
        crate::tensor::Error::msg(format!("fused_bias_gelu_new_on_stream launch: {e}"))
    })?;
    drop(x_store);
    drop(b_store);

    let out_storage =
        crate::tensor::cuda_ext::CudaStorage::wrap_cuda_slice(out_slice, cuda_dev.clone());
    crate::tensor::cuda_ext::tensor_from_cuda_storage(out_storage, x.shape().clone())
}

/// Phi2 / GPT-NeoX parallel-attention residual merge with both biases
/// folded into the same launch:
///   out = residual + (attn_out + attn_bias) + (ffn_out + ffn_bias)
///
/// Replaces the 3-launch chain
/// `attn_out.broadcast_add(attn_bias)? .broadcast_add(ffn_out)? .broadcast_add(ffn_bias)? + residual`
/// (or its `fused_add_three` + 2 broadcast_adds variant) with a single
/// elementwise launch. `attn_bias` and `ffn_bias` are 1-D F32 tensors of
/// length equal to the trailing dim of `residual`. Falls back to the
/// unfused broadcast_add chain for any shape / dtype / device mismatch.
pub fn fused_phi2_residual_merge(
    residual: &Tensor,
    attn_out: &Tensor,
    attn_bias: &Tensor,
    ffn_out: &Tensor,
    ffn_bias: &Tensor,
) -> Result<Tensor> {
    let r_dims = residual.dims();
    let trailing = r_dims.last().copied().unwrap_or(0);
    let supported = residual.device().is_cuda()
        && attn_out.device().is_cuda()
        && attn_bias.device().is_cuda()
        && ffn_out.device().is_cuda()
        && ffn_bias.device().is_cuda()
        && residual.dtype() == DType::F32
        && attn_out.dtype() == DType::F32
        && attn_bias.dtype() == DType::F32
        && ffn_out.dtype() == DType::F32
        && ffn_bias.dtype() == DType::F32
        && residual.shape() == attn_out.shape()
        && residual.shape() == ffn_out.shape()
        && attn_bias.dims().len() == 1
        && ffn_bias.dims().len() == 1
        && attn_bias.dims()[0] == trailing
        && ffn_bias.dims()[0] == trailing;
    if !supported {
        let s1 = attn_out.broadcast_add(attn_bias)?;
        let s2 = ffn_out.broadcast_add(ffn_bias)?;
        return (residual + &s1)? + &s2;
    }

    let residual = residual.contiguous()?;
    let attn_out = attn_out.contiguous()?;
    let attn_bias = attn_bias.contiguous()?;
    let ffn_out = ffn_out.contiguous()?;
    let ffn_bias = ffn_bias.contiguous()?;
    let n = residual.elem_count();
    let d = trailing;

    let out = unsafe { Tensor::empty(residual.shape(), DType::F32, &residual.device())? };

    let cuda_dev = residual.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_phi2_residual_merge_f32", "loken_fused", ptx)?;

    let (r_store, r_layout) = residual.storage_and_layout();
    let (a_store, a_layout) = attn_out.storage_and_layout();
    let (ab_store, ab_layout) = attn_bias.storage_and_layout();
    let (f_store, f_layout) = ffn_out.storage_and_layout();
    let (fb_store, fb_layout) = ffn_bias.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(r_s),
        crate::tensor::StorageView::Cuda(a_s),
        crate::tensor::StorageView::Cuda(ab_s),
        crate::tensor::StorageView::Cuda(f_s),
        crate::tensor::StorageView::Cuda(fb_s),
        crate::tensor::StorageView::Cuda(o_s),
    ) = (
        &*r_store, &*a_store, &*ab_store, &*f_store, &*fb_store, &*o_store,
    ) {
        let r_slice = r_s.as_cuda_slice::<f32>()?;
        let a_slice = a_s.as_cuda_slice::<f32>()?;
        let ab_slice = ab_s.as_cuda_slice::<f32>()?;
        let f_slice = f_s.as_cuda_slice::<f32>()?;
        let fb_slice = fb_s.as_cuda_slice::<f32>()?;
        let o_slice = o_s.as_cuda_slice::<f32>()?;
        let r_view = r_slice.slice(r_layout.start_offset()..);
        let a_view = a_slice.slice(a_layout.start_offset()..);
        let ab_view = ab_slice.slice(ab_layout.start_offset()..);
        let f_view = f_slice.slice(f_layout.start_offset()..);
        let fb_view = fb_slice.slice(fb_layout.start_offset()..);

        let (block, grid) = adaptive_grid_1d(n as u32, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let d_i32 = d as i32;
        let mut builder = func.builder();
        builder.arg(&r_view);
        builder.arg(&a_view);
        builder.arg(&ab_view);
        builder.arg(&f_view);
        builder.arg(&fb_view);
        builder.arg(o_slice);
        builder.arg(&n_i32);
        builder.arg(&d_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_phi2_residual_merge: {e}")))?;
    }

    drop(r_store);
    drop(a_store);
    drop(ab_store);
    drop(f_store);
    drop(fb_store);
    drop(o_store);
    Ok(out)
}

/// Fused SiLU(gate) * up - replaces 2 kernel launches with 1.
/// Fused OAI clamped-SwiGLU (gpt-oss): out = (min(gate,limit).σ(α.min(gate,limit))).(1+clamp(up,±limit)).
/// One elementwise launch in place of swiglu_oai_combine's ~7 ops. CPU falls
/// back to the explicit op sequence (kept bit-identical to the kernel formula).
pub fn fused_swiglu_oai(gate: &Tensor, up: &Tensor, alpha: f64, limit: f64) -> Result<Tensor> {
    if !gate.device().is_cuda() {
        let lim = limit as f32;
        // Scalar clamp via relu folds (native Tensor has no scalar `clamp`):
        //   min(t, hi) = hi - relu(hi - t);  max(t, lo) = relu(t - lo) + lo.
        // gate is clamped above only (lower bound is effectively -inf), up to ±lim.
        let min_hi = |t: &Tensor, hi: f32| -> Result<Tensor> {
            t.affine(1.0, -hi)?.relu()?.affine(-1.0, hi)
        };
        let max_lo =
            |t: &Tensor, lo: f32| -> Result<Tensor> { t.affine(1.0, -lo)?.relu()?.affine(1.0, lo) };
        let x = min_hi(gate, lim)?;
        let g = max_lo(&min_hi(up, lim)?, -lim)?;
        let sig = crate::tensor::ops::sigmoid(&x.affine(alpha as f32, 0.0)?)?;
        let glu = x.broadcast_mul(&sig)?;
        return glu.broadcast_mul(&g.affine(1.0, 1.0)?);
    }

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let n = gate.elem_count();

    let out = unsafe { Tensor::empty(gate.shape(), DType::F32, &gate.device())? };

    let cuda_dev = gate.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_swiglu_oai_f32", "loken_fused", ptx)?;

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
        let alpha_f = alpha as f32;
        let limit_f = limit as f32;
        let mut builder = func.builder();
        builder.arg(&g_view);
        builder.arg(&u_view);
        builder.arg(o_slice);
        builder.arg(&n_i32);
        builder.arg(&alpha_f);
        builder.arg(&limit_f);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_swiglu_oai: {e}")))?;
    }

    drop(g_store);
    drop(u_store);
    drop(o_store);
    Ok(out)
}

/// fused_swiglu_oai with per-row gate/up bias folded in. Biases must already be
/// gathered to the same [M,N] shape as gate/up. Both-present takes the fused
/// 4-input kernel; neither falls back to fused_swiglu_oai; one-present adds that
/// bias then calls fused_swiglu_oai (rare - gpt-oss always has both).
pub fn fused_swiglu_oai_bias(
    gate: &Tensor,
    up: &Tensor,
    gbias: Option<&Tensor>,
    ubias: Option<&Tensor>,
    alpha: f64,
    limit: f64,
) -> Result<Tensor> {
    let (gb, ub) = match (gbias, ubias) {
        (Some(gb), Some(ub)) => (gb, ub),
        (None, None) => return fused_swiglu_oai(gate, up, alpha, limit),
        (Some(gb), None) => return fused_swiglu_oai(&gate.broadcast_add(gb)?, up, alpha, limit),
        (None, Some(ub)) => return fused_swiglu_oai(gate, &up.broadcast_add(ub)?, alpha, limit),
    };
    if !gate.device().is_cuda() {
        return fused_swiglu_oai(
            &gate.broadcast_add(gb)?,
            &up.broadcast_add(ub)?,
            alpha,
            limit,
        );
    }

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let gb = gb.contiguous()?;
    let ub = ub.contiguous()?;
    let n = gate.elem_count();
    let out = unsafe { Tensor::empty(gate.shape(), DType::F32, &gate.device())? };

    let cuda_dev = gate.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_swiglu_oai_bias_f32", "loken_fused", ptx)?;

    let (g_store, g_layout) = gate.storage_and_layout();
    let (u_store, u_layout) = up.storage_and_layout();
    let (gb_store, gb_layout) = gb.storage_and_layout();
    let (ub_store, ub_layout) = ub.storage_and_layout();
    let (o_store, _) = out.storage_and_layout();

    if let (
        crate::tensor::StorageView::Cuda(g),
        crate::tensor::StorageView::Cuda(u),
        crate::tensor::StorageView::Cuda(gbs),
        crate::tensor::StorageView::Cuda(ubs),
        crate::tensor::StorageView::Cuda(o),
    ) = (&*g_store, &*u_store, &*gb_store, &*ub_store, &*o_store)
    {
        let g_view = g.as_cuda_slice::<f32>()?.slice(g_layout.start_offset()..);
        let u_view = u.as_cuda_slice::<f32>()?.slice(u_layout.start_offset()..);
        let gb_view = gbs
            .as_cuda_slice::<f32>()?
            .slice(gb_layout.start_offset()..);
        let ub_view = ubs
            .as_cuda_slice::<f32>()?
            .slice(ub_layout.start_offset()..);
        let o_slice = o.as_cuda_slice::<f32>()?;

        let (block, grid) = adaptive_grid_1d(n as u32, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let alpha_f = alpha as f32;
        let limit_f = limit as f32;
        let mut builder = func.builder();
        builder.arg(&g_view);
        builder.arg(&u_view);
        builder.arg(&gb_view);
        builder.arg(&ub_view);
        builder.arg(o_slice);
        builder.arg(&n_i32);
        builder.arg(&alpha_f);
        builder.arg(&limit_f);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_swiglu_oai_bias: {e}")))?;
    }
    drop(g_store);
    drop(u_store);
    drop(gb_store);
    drop(ub_store);
    drop(o_store);
    Ok(out)
}

/// LFM2 gated short-conv inner, fused into one launch. Inputs: bcx [b, 3*D]
/// (in_proj output, F32), state_in [b, D, L-1], conv_w [D, L]. Returns
/// (y [b, D], state_out [b, D, L-1]). CUDA only (lfm2 is a CUDA arch).
pub fn fused_lfm2_shortconv(
    bcx: &Tensor,
    state_in: &Tensor,
    conv_w: &Tensor,
    d_model: usize,
    l_cache: usize,
) -> Result<(Tensor, Tensor)> {
    if !bcx.device().is_cuda() {
        crate::tensor::bail!("fused_lfm2_shortconv requires CUDA");
    }
    let b = bcx.dim(0)?;
    let lm1 = l_cache - 1;
    let bcx = bcx.contiguous()?;
    let state_in = state_in.contiguous()?;
    let conv_w = conv_w.contiguous()?;
    let y = unsafe { Tensor::empty((b, d_model), DType::F32, &bcx.device())? };
    let state_out = unsafe { Tensor::empty((b, d_model, lm1), DType::F32, &bcx.device())? };

    let cuda_dev = bcx.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func = cuda_dev.get_or_load_custom_func("fused_lfm2_shortconv_f32", "loken_fused", ptx)?;

    let (bcx_st, bcx_l) = bcx.storage_and_layout();
    let (si_st, si_l) = state_in.storage_and_layout();
    let (cw_st, cw_l) = conv_w.storage_and_layout();
    let (y_st, _) = y.storage_and_layout();
    let (so_st, _) = state_out.storage_and_layout();
    if let (
        crate::tensor::StorageView::Cuda(bc),
        crate::tensor::StorageView::Cuda(si),
        crate::tensor::StorageView::Cuda(cw),
        crate::tensor::StorageView::Cuda(yo),
        crate::tensor::StorageView::Cuda(so),
    ) = (&*bcx_st, &*si_st, &*cw_st, &*y_st, &*so_st)
    {
        let bc_v = bc.as_cuda_slice::<f32>()?.slice(bcx_l.start_offset()..);
        let si_v = si.as_cuda_slice::<f32>()?.slice(si_l.start_offset()..);
        let cw_v = cw.as_cuda_slice::<f32>()?.slice(cw_l.start_offset()..);
        let yo_s = yo.as_cuda_slice::<f32>()?;
        let so_s = so.as_cuda_slice::<f32>()?;
        let n = (b * d_model) as u32;
        let (block, grid) = adaptive_grid_1d(n, cuda_dev.multiprocessor_count() as u32);
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (bi, di, li) = (b as i32, d_model as i32, l_cache as i32);
        let mut builder = func.builder();
        builder.arg(&bc_v);
        builder.arg(&si_v);
        builder.arg(&cw_v);
        builder.arg(yo_s);
        builder.arg(so_s);
        builder.arg(&bi);
        builder.arg(&di);
        builder.arg(&li);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_lfm2_shortconv: {e}")))?;
    }
    drop(bcx_st);
    drop(si_st);
    drop(cw_st);
    drop(y_st);
    drop(so_st);
    Ok((y, state_out))
}

/// Mamba2 post-SSM D-skip + SiLU(z) gate + group-wise gated RMSNorm, fused.
/// y_ssm/x_c/z [b, d_inner] F32; dvec [nh]; norm_w [ngroups, gs]. Returns out
/// [b, d_inner]. One block per (b, group). CUDA only.
pub fn fused_mamba2_gate_gnorm(
    y_ssm: &Tensor,
    x_c: &Tensor,
    dvec: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    bsz: usize,
    d_inner: usize,
    hd: usize,
    ngroups: usize,
    eps: f64,
) -> Result<Tensor> {
    if !y_ssm.device().is_cuda() {
        crate::tensor::bail!("fused_mamba2_gate_gnorm requires CUDA");
    }
    let y_ssm = y_ssm.contiguous()?;
    let x_c = x_c.contiguous()?;
    let dvec = dvec.contiguous()?;
    let z = z.contiguous()?;
    let norm_w = norm_w.contiguous()?;
    let out = unsafe { Tensor::empty((bsz, d_inner), DType::F32, &y_ssm.device())? };

    let cuda_dev = y_ssm.device().as_cuda_device()?;
    let ptx = get_ptx(&cuda_dev)?;
    let func =
        cuda_dev.get_or_load_custom_func("fused_mamba2_gate_gnorm_f32", "loken_fused", ptx)?;

    let (ys_s, ys_l) = y_ssm.storage_and_layout();
    let (xc_s, xc_l) = x_c.storage_and_layout();
    let (d_s, d_l) = dvec.storage_and_layout();
    let (z_s, z_l) = z.storage_and_layout();
    let (nw_s, nw_l) = norm_w.storage_and_layout();
    let (o_s, _) = out.storage_and_layout();
    use crate::tensor::StorageView::Cuda as C;
    if let (C(ys), C(xc), C(dd), C(zz), C(nw), C(oo)) =
        (&*ys_s, &*xc_s, &*d_s, &*z_s, &*nw_s, &*o_s)
    {
        let ys_v = ys.as_cuda_slice::<f32>()?.slice(ys_l.start_offset()..);
        let xc_v = xc.as_cuda_slice::<f32>()?.slice(xc_l.start_offset()..);
        let d_v = dd.as_cuda_slice::<f32>()?.slice(d_l.start_offset()..);
        let z_v = zz.as_cuda_slice::<f32>()?.slice(z_l.start_offset()..);
        let nw_v = nw.as_cuda_slice::<f32>()?.slice(nw_l.start_offset()..);
        let o_sl = oo.as_cuda_slice::<f32>()?;
        let block: u32 = 256;
        let grid: u32 = (bsz * ngroups) as u32;
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        let (bi, dii, hdi, ngi, epsf) = (
            bsz as i32,
            d_inner as i32,
            hd as i32,
            ngroups as i32,
            eps as f32,
        );
        let mut builder = func.builder();
        builder.arg(&ys_v);
        builder.arg(&xc_v);
        builder.arg(&d_v);
        builder.arg(&z_v);
        builder.arg(&nw_v);
        builder.arg(o_sl);
        builder.arg(&bi);
        builder.arg(&dii);
        builder.arg(&hdi);
        builder.arg(&ngi);
        builder.arg(&epsf);
        unsafe { builder.launch(cfg) }
            .map_err(|e| crate::tensor::Error::msg(format!("fused_mamba2_gate_gnorm: {e}")))?;
    }
    drop(ys_s);
    drop(xc_s);
    drop(d_s);
    drop(z_s);
    drop(nw_s);
    drop(o_s);
    Ok(out)
}
