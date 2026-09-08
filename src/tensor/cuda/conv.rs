//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Direct transposed 1-D conv for one batch slab (dilation 1, groups 1).
pub fn convt1d_f32(
    dev: &CudaDevice,
    x: &cudarc::driver::CudaView<f32>,
    w: &CudaSlice<f32>,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    l_out: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_convt1d_f32")?;
    let stream = dev.stream();
    let n = c_out * l_out;
    let out = with_oom_retry(dev, "convt", || stream.alloc_zeros::<f32>(n))?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let args: Vec<i32> = vec![
        c_in as i32,
        c_out as i32,
        l_in as i32,
        l_out as i32,
        k as i32,
        stride as i32,
        padding as i32,
    ];
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(w);
    b.arg(&out);
    for v in &args {
        b.arg(v);
    }
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("convt launch: {e}")))?;
    Ok(out)
}

/// GroupNorm on [b, c, spatial]: one block per (batch, group) slab.
pub fn group_norm_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    b: usize,
    num_groups: usize,
    cg: usize,
    spatial: usize,
    eps: f32,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_group_norm_f32")?;
    let stream = dev.stream();
    let n = b * num_groups * cg * spatial;
    let out = with_oom_retry(dev, "group_norm", || stream.alloc_zeros::<f32>(n))?;
    let cfg = LaunchConfig {
        grid_dim: ((b * num_groups) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (g_i, cg_i, sp_i) = (num_groups as i32, cg as i32, spatial as i32);
    let mut bd = stream.launch_builder(&func);
    bd.arg(x);
    bd.arg(w);
    bd.arg(bias);
    bd.arg(&out);
    bd.arg(&g_i);
    bd.arg(&cg_i);
    bd.arg(&sp_i);
    bd.arg(&eps);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("group_norm launch: {e}")))?;
    Ok(out)
}

/// Generic permute (transpose is a 2-axis swap). `odims` = output dims,
/// `in_strides_perm[ax]` = input stride of the axis that lands at output `ax`.
pub fn permute_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    odims: &[i32],
    in_strides_perm: &[i32],
    n: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_permute_f32")?;
    let stream = dev.stream();
    // Cached device constants (NOT per-call uploads): shape-stable contents,
    // and a captured H2D from a temporary host slice breaks graph replay.
    let d_odims = dev_const_i32(dev, odims)?;
    let d_strides = dev_const_i32(dev, in_strides_perm)?;
    let out = with_oom_retry(dev, "permute", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rank_i, n_i) = (odims.len() as i32, n as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&*d_odims);
    b.arg(&*d_strides);
    b.arg(&rank_i);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("permute launch: {e}")))?;
    Ok(out)
}

/// Width-generic permute/broadcast gather over any storage variant:
/// same index math as [`permute_f32`], element width chosen by
/// dtype. Returns a storage of the SAME variant.
pub fn permute_storage(
    dev: &CudaDevice,
    src: &CudaStorage,
    odims: &[i32],
    in_strides_perm: &[i32],
    n: usize,
) -> Result<CudaStorage> {
    let stream = dev.stream();
    // Cached device constants (NOT per-call uploads): shape-stable contents,
    // and a captured H2D from a temporary host slice breaks graph replay.
    let d_odims = dev_const_i32(dev, odims)?;
    let d_strides = dev_const_i32(dev, in_strides_perm)?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rank_i, n_i) = (odims.len() as i32, n as i32);
    macro_rules! run {
        ($slice:expr, $t:ty, $variant:ident, $kernel:expr) => {{
            let func = dev.elementwise_fn($kernel)?;
            let out = with_oom_retry(dev, "permute", || unsafe { stream.alloc::<$t>(n) })?;
            let mut b = stream.launch_builder(&func);
            b.arg($slice);
            b.arg(&out);
            b.arg(&*d_odims);
            b.arg(&*d_strides);
            b.arg(&rank_i);
            b.arg(&n_i);
            unsafe { b.launch(cfg) }.map_err(|e| Error(format!("permute launch: {e}")))?;
            Ok(CudaStorage::$variant(out))
        }};
    }
    match src {
        CudaStorage::F32(s) => run!(s, f32, F32, "native_permute_w32"),
        CudaStorage::U32(s) => run!(s, u32, U32, "native_permute_w32"),
        CudaStorage::I32(s) => run!(s, i32, I32, "native_permute_w32"),
        CudaStorage::F16(s) => run!(s, half::f16, F16, "native_permute_w16"),
        CudaStorage::BF16(s) => run!(s, half::bf16, BF16, "native_permute_w16"),
        CudaStorage::I16(s) => run!(s, i16, I16, "native_permute_w16"),
        CudaStorage::I64(s) => run!(s, i64, I64, "native_permute_w64"),
        CudaStorage::U8(s) => run!(s, u8, U8, "native_permute_w8"),
    }
}

/// RoPE on [bh, seq, d] with [seq, d/2] cos/sin tables.
pub fn rope_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    bh: usize,
    seq: usize,
    d: usize,
    interleaved: bool,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_rope_f32")?;
    let stream = dev.stream();
    let n = bh * seq * d;
    let out = with_oom_retry(dev, "rope", || unsafe { stream.alloc::<f32>(n) })?;
    let pairs = bh * seq * (d / 2);
    let cfg = LaunchConfig {
        grid_dim: (pairs.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bh_i, seq_i, d_i) = (bh as i32, seq as i32, d as i32);
    let inter_i = interleaved as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(cos);
    b.arg(sin);
    b.arg(&out);
    b.arg(&bh_i);
    b.arg(&seq_i);
    b.arg(&d_i);
    b.arg(&inter_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("rope launch: {e}")))?;
    Ok(out)
}

// -- cuBLAS math-mode plumbing --------------------------
//
// Mirrors the reference per-dtype reduced-precision GEMM flags EXACTLY
// default OFF (f32 gemm = COMPUTE_32F,
// f16 gemm = f32 accumulation, bf16 gemm = COMPUTE_32F); when the engine
// enables them (cuda_ext::set_gemm_reduced_precision(true) at model load),
// f32 -> COMPUTE_32F_FAST_TF32, f16 -> COMPUTE_16F, bf16 -> COMPUTE_32F_FAST_16BF.
// Every gemm goes through gemm_strided_batched_ex with
// CUBLAS_GEMM_DEFAULT_TENSOR_OP - so the substrate picks
// the same cuBLAS kernels per-dtype as the facade for any flag state.

pub(super) static MM_F32_REDUCED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
pub(super) static MM_F16_REDUCED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
pub(super) static MM_BF16_REDUCED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// API-compat: allow TF32 reductions in f32 GEMMs.
pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED.store(b, std::sync::atomic::Ordering::Relaxed)
}
