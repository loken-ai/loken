//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

pub fn unary_f32(
    dev: &CudaDevice,
    name: &str,
    x: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    launch_elementwise(dev, name, &[x], n)
}

/// Unary f16 op computed in half precision.
pub fn unary_f16(
    dev: &CudaDevice,
    name: &str,
    x: &CudaSlice<half::f16>,
    n: usize,
) -> Result<CudaSlice<half::f16>> {
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "kernel out", || unsafe {
        stream.alloc::<half::f16>(n)
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("{name} launch: {e}")))?;
    Ok(out)
}

/// Binary f16 op computed in half precision.
pub fn binary_f16(
    dev: &CudaDevice,
    name: &str,
    a: &CudaSlice<half::f16>,
    b: &CudaSlice<half::f16>,
    n: usize,
) -> Result<CudaSlice<half::f16>> {
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "kernel out", || unsafe {
        stream.alloc::<half::f16>(n)
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut bd = stream.launch_builder(&func);
    bd.arg(a);
    bd.arg(b);
    bd.arg(&out);
    bd.arg(&n_i);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("{name} launch: {e}")))?;
    Ok(out)
}

/// `x * alpha + beta` in HALF precision (the reference f16 affine semantics).
pub fn affine_f16(
    dev: &CudaDevice,
    x: &CudaSlice<half::f16>,
    alpha: f32,
    beta: f32,
    n: usize,
) -> Result<CudaSlice<half::f16>> {
    let func = dev.elementwise_fn("native_affine_f16")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "affine f16", || unsafe {
        stream.alloc::<half::f16>(n)
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&alpha);
    b.arg(&beta);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("affine f16 launch: {e}")))?;
    Ok(out)
}

/// Binary bf16 op (f32 compute, one rounding - see the kernel comment).
pub fn binary_bf16(
    dev: &CudaDevice,
    name: &str,
    a: &CudaSlice<half::bf16>,
    b: &CudaSlice<half::bf16>,
    n: usize,
) -> Result<CudaSlice<half::bf16>> {
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "kernel out", || unsafe {
        stream.alloc::<half::bf16>(n)
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut bd = stream.launch_builder(&func);
    bd.arg(a);
    bd.arg(b);
    bd.arg(&out);
    bd.arg(&n_i);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("{name} launch: {e}")))?;
    Ok(out)
}

/// `x * alpha + beta` in bf16 (per-op rounding like the reference bf16 affine).
pub fn affine_bf16(
    dev: &CudaDevice,
    x: &CudaSlice<half::bf16>,
    alpha: f32,
    beta: f32,
    n: usize,
) -> Result<CudaSlice<half::bf16>> {
    let func = dev.elementwise_fn("native_affine_bf16")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "affine bf16", || unsafe {
        stream.alloc::<half::bf16>(n)
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&alpha);
    b.arg(&beta);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("affine bf16 launch: {e}")))?;
    Ok(out)
}

/// `x * alpha` elementwise.
pub fn scale_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    alpha: f32,
    n: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_scale_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "scale", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&alpha);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("scale launch: {e}")))?;
    Ok(out)
}

/// Tail-aligned broadcast binary (rhs repeats over lhs's leading dims).
pub fn broadcast_tail_f32(
    dev: &CudaDevice,
    name: &str,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    n: usize,
    bn: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "kernel out", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, bn_i) = (n as i32, bn as i32);
    let mut bd = stream.launch_builder(&func);
    bd.arg(a);
    bd.arg(b);
    bd.arg(&out);
    bd.arg(&n_i);
    bd.arg(&bn_i);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("{name} launch: {e}")))?;
    Ok(out)
}

/// Tail-aligned broadcast binary in HALF (rhs repeats over lhs's leading
/// dims) - the reference f16 broadcast arithmetic without the f32-cast detour.
pub fn broadcast_tail_f16(
    dev: &CudaDevice,
    name: &str,
    a: &CudaSlice<half::f16>,
    b: &CudaSlice<half::f16>,
    n: usize,
    bn: usize,
) -> Result<CudaSlice<half::f16>> {
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "kernel out", || unsafe {
        stream.alloc::<half::f16>(n)
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, bn_i) = (n as i32, bn as i32);
    let mut bd = stream.launch_builder(&func);
    bd.arg(a);
    bd.arg(b);
    bd.arg(&out);
    bd.arg(&n_i);
    bd.arg(&bn_i);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("{name} launch: {e}")))?;
    Ok(out)
}

/// Tail-aligned broadcast binary in bf16 (f32 compute, one rounding).
pub fn broadcast_tail_bf16(
    dev: &CudaDevice,
    name: &str,
    a: &CudaSlice<half::bf16>,
    b: &CudaSlice<half::bf16>,
    n: usize,
    bn: usize,
) -> Result<CudaSlice<half::bf16>> {
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "kernel out", || unsafe {
        stream.alloc::<half::bf16>(n)
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, bn_i) = (n as i32, bn as i32);
    let mut bd = stream.launch_builder(&func);
    bd.arg(a);
    bd.arg(b);
    bd.arg(&out);
    bd.arg(&n_i);
    bd.arg(&bn_i);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("{name} launch: {e}")))?;
    Ok(out)
}

/// General N-d broadcast binary (`op`: 0=add, 1=mul). `odims` = broadcast
/// output dims; `a_strides`/`b_strides` = per-axis effective strides into the
/// operands (0 on broadcast axes). One small meta upload per call.
pub fn broadcast_nd_f32(
    dev: &CudaDevice,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    odims: &[i32],
    a_strides: &[i32],
    b_strides: &[i32],
    n: usize,
    op: i32,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_bbin_nd_f32")?;
    let stream = dev.stream();
    let mut meta = Vec::with_capacity(odims.len() * 3);
    meta.extend_from_slice(odims);
    meta.extend_from_slice(a_strides);
    meta.extend_from_slice(b_strides);
    // Cached device constant (NOT a per-call upload): shape-stable contents,
    // and a captured H2D from a temporary host Vec breaks graph replay.
    let d_meta = dev_const_i32(dev, &meta)?;
    // the kernel writes every output element - no zero-fill needed
    let out = with_oom_retry(dev, "broadcast_nd", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rank_i, n_l) = (odims.len() as i32, n as i64);
    let mut bd = stream.launch_builder(&func);
    bd.arg(a);
    bd.arg(b);
    bd.arg(&out);
    bd.arg(&*d_meta);
    bd.arg(&rank_i);
    bd.arg(&n_l);
    bd.arg(&op);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("broadcast_nd launch: {e}")))?;
    Ok(out)
}

/// LayerNorm over the last dim on a [rows, cols] device buffer (one block per
/// row). `bias = None` runs the weight-only form.
pub fn layer_norm_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_layer_norm_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "layer_norm", || unsafe {
        stream.alloc::<f32>(rows * cols)
    })?;
    let block = row_block(cols);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 8, // two f32 accumulator arrays
    };
    let cols_i = cols as i32;
    let has_bias = bias.is_some() as i32;
    let b_slice = bias.unwrap_or(w); // dummy alias when absent (kernel skips it)
    let mut bd = stream.launch_builder(&func);
    bd.arg(x);
    bd.arg(w);
    bd.arg(b_slice);
    bd.arg(&out);
    bd.arg(&cols_i);
    bd.arg(&eps);
    bd.arg(&has_bias);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("layer_norm launch: {e}")))?;
    Ok(out)
}

/// Copy one cat operand (`src` = [outer, src_row] elements) into `dst` whose
/// rows are `dst_stride` apart, starting at element offset `dst_off`.
pub fn cat_copy_f32(
    dev: &CudaDevice,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    n: usize,
    src_row: usize,
    dst_stride: usize,
    dst_off: usize,
) -> Result<()> {
    let func = dev.elementwise_fn("native_cat_copy_f32")?;
    let stream = dev.stream();
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_l = n as i64;
    let (sr_i, ds_i, do_i) = (src_row as i32, dst_stride as i32, dst_off as i32);
    let mut bd = stream.launch_builder(&func);
    bd.arg(src);
    bd.arg(&*dst);
    bd.arg(&n_l);
    bd.arg(&sr_i);
    bd.arg(&ds_i);
    bd.arg(&do_i);
    unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("cat_copy launch: {e}")))?;
    Ok(())
}

/// Width-generic cat copy: same row/stride math as [`cat_copy_f32`], with
/// counts in ELEMENTS of the storage dtype, dispatched to the kernel of the
/// element width (cat is pure data movement).
pub fn cat_copy_storage(
    dev: &CudaDevice,
    src: &CudaStorage,
    dst: &mut CudaStorage,
    n: usize,
    src_row: usize,
    dst_stride: usize,
    dst_off: usize,
) -> Result<()> {
    let kernel = match src.dtype().size_in_bytes() {
        1 => "native_cat_copy_w8",
        2 => "native_cat_copy_w16",
        4 => "native_cat_copy_f32",
        8 => "native_cat_copy_w64",
        w => return Err(Error(format!("cat_copy_storage: unsupported width {w}"))),
    };
    let func = dev.elementwise_fn(kernel)?;
    let stream = dev.stream();
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_l = n as i64;
    let (sr_i, ds_i, do_i) = (src_row as i32, dst_stride as i32, dst_off as i32);
    macro_rules! launch {
        ($s:expr, $d:expr) => {{
            let mut bd = stream.launch_builder(&func);
            bd.arg($s);
            bd.arg(&*$d);
            bd.arg(&n_l);
            bd.arg(&sr_i);
            bd.arg(&ds_i);
            bd.arg(&do_i);
            unsafe { bd.launch(cfg) }.map_err(|e| Error(format!("cat_copy launch: {e}")))?;
        }};
    }
    match (src, dst) {
        (CudaStorage::U8(s), CudaStorage::U8(d)) => launch!(s, d),
        (CudaStorage::U32(s), CudaStorage::U32(d)) => launch!(s, d),
        (CudaStorage::I16(s), CudaStorage::I16(d)) => launch!(s, d),
        (CudaStorage::I32(s), CudaStorage::I32(d)) => launch!(s, d),
        (CudaStorage::I64(s), CudaStorage::I64(d)) => launch!(s, d),
        (CudaStorage::F16(s), CudaStorage::F16(d)) => launch!(s, d),
        (CudaStorage::BF16(s), CudaStorage::BF16(d)) => launch!(s, d),
        (CudaStorage::F32(s), CudaStorage::F32(d)) => launch!(s, d),
        _ => return Err(Error("cat_copy_storage: src/dst dtype mismatch".into())),
    }
    Ok(())
}

/// `x * alpha + beta` elementwise.
pub fn affine_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    alpha: f32,
    beta: f32,
    n: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_affine_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "affine", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&alpha);
    b.arg(&beta);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("affine launch: {e}")))?;
    Ok(out)
}

/// Per-channel bias add on [b, c, spatial]-shaped data.
pub fn bias_chw_f32(
    dev: &CudaDevice,
    a: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    n: usize,
    spatial: usize,
    c: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_bias_chw_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "bias_chw", || stream.alloc_zeros::<f32>(n))?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, sp_i, c_i) = (n as i32, spatial as i32, c as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(a);
    b.arg(bias);
    b.arg(&out);
    b.arg(&n_i);
    b.arg(&sp_i);
    b.arg(&c_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("bias_chw launch: {e}")))?;
    Ok(out)
}

/// Nearest-neighbor 2-D upsample on [bc, h, w] -> [bc, oh, ow].
pub fn upsample2d_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    bc: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_upsample2d_f32")?;
    let stream = dev.stream();
    let n = bc * oh * ow;
    let out = with_oom_retry(dev, "upsample", || stream.alloc_zeros::<f32>(n))?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bc_i, h_i, w_i, oh_i, ow_i) = (bc as i32, h as i32, w as i32, oh as i32, ow as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&bc_i);
    b.arg(&h_i);
    b.arg(&w_i);
    b.arg(&oh_i);
    b.arg(&ow_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("upsample launch: {e}")))?;
    Ok(out)
}

/// Bilinear 2-D resample on [bc, h, w] -> [bc, oh, ow], half-pixel centres.
pub fn upsample_bilinear2d_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    bc: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_upsample_bilinear2d_f32")?;
    let stream = dev.stream();
    let n = bc * oh * ow;
    let out = with_oom_retry(dev, "upsample_bilinear", || stream.alloc_zeros::<f32>(n))?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bc_i, h_i, w_i, oh_i, ow_i) = (bc as i32, h as i32, w as i32, oh as i32, ow as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&bc_i);
    b.arg(&h_i);
    b.arg(&w_i);
    b.arg(&oh_i);
    b.arg(&ow_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("bilinear launch: {e}")))?;
    Ok(out)
}

/// Zero-pad one dim: [outer, d_in, inner] -> [outer, d_in+left+right, inner].
pub fn pad_dim_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    outer: usize,
    d_in: usize,
    left: usize,
    right: usize,
    inner: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_pad_dim_f32")?;
    let stream = dev.stream();
    let d_out = d_in + left + right;
    let n = outer * d_out * inner;
    let out = with_oom_retry(dev, "pad", || stream.alloc_zeros::<f32>(n))?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (o_i, di_i, do_i, l_i, in_i) = (
        outer as i32,
        d_in as i32,
        d_out as i32,
        left as i32,
        inner as i32,
    );
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&o_i);
    b.arg(&di_i);
    b.arg(&do_i);
    b.arg(&l_i);
    b.arg(&in_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("pad launch: {e}")))?;
    Ok(out)
}

/// Fused snake activation on [b, c, l].
pub fn snake1d_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    alpha: &CudaSlice<f32>,
    inv_alpha: &CudaSlice<f32>,
    n: usize,
    l: usize,
    c: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_snake1d_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "snake", || stream.alloc_zeros::<f32>(n))?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_l = n as i64;
    let (l_i, c_i) = (l as i32, c as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(alpha);
    b.arg(inv_alpha);
    b.arg(&out);
    b.arg(&n_l);
    b.arg(&l_i);
    b.arg(&c_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("snake launch: {e}")))?;
    Ok(out)
}
