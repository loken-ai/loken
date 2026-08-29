//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Device-side narrow of a contiguous tensor along one dim, dtype-agnostic
/// (byte copy): rows = product of dims before+at the dim, the narrowed span
/// becomes the new row payload. Returns a storage of the SAME variant.
pub fn narrow_storage(
    dev: &CudaDevice,
    src: &CudaStorage,
    n_rows: usize,
    src_row_bytes: usize,
    off_bytes: usize,
    row_bytes: usize,
) -> Result<CudaStorage> {
    let func = dev.elementwise_fn("native_slice_u8")?;
    let stream = dev.stream();
    let total = n_rows * row_bytes;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (offb, srb, rb, nr) = (
        off_bytes as i64,
        src_row_bytes as i64,
        row_bytes as i64,
        n_rows as i64,
    );
    macro_rules! run {
        ($slice:expr, $t:ty, $variant:ident) => {{
            let elem = std::mem::size_of::<$t>();
            let out = with_oom_retry(dev, "narrow", || unsafe {
                stream.alloc::<$t>(total / elem)
            })?;
            // The kernel reads both buffers as byte pointers; passing the
            // typed slices is fine at the ABI level (a pointer either way).
            let mut b = stream.launch_builder(&func);
            b.arg($slice);
            b.arg(&out);
            b.arg(&offb);
            b.arg(&srb);
            b.arg(&rb);
            b.arg(&nr);
            unsafe { b.launch(cfg) }.map_err(|e| Error(format!("narrow launch: {e}")))?;
            Ok(CudaStorage::$variant(out))
        }};
    }
    match src {
        CudaStorage::F32(s) => run!(s, f32, F32),
        CudaStorage::F16(s) => run!(s, half::f16, F16),
        CudaStorage::BF16(s) => run!(s, half::bf16, BF16),
        CudaStorage::U32(s) => run!(s, u32, U32),
        CudaStorage::I16(s) => run!(s, i16, I16),
        CudaStorage::I32(s) => run!(s, i32, I32),
        CudaStorage::I64(s) => run!(s, i64, I64),
        CudaStorage::U8(s) => run!(s, u8, U8),
    }
}

/// In-place `slice_set`: copy `src` (viewed as `[n_rows, src_row_bytes]`)
/// into `dst` rows of `dst_row_bytes` at byte offset `off_bytes`
/// (dtype-agnostic, same-variant storages). The kernel writes through the
/// dst slice in place - the production fused kernels mutate KV buffers
/// through shared refs the same way.
pub fn slice_set_storage(
    dev: &CudaDevice,
    dst: &CudaStorage,
    src: &CudaStorage,
    n_rows: usize,
    dst_row_bytes: usize,
    src_row_bytes: usize,
    off_bytes: usize,
) -> Result<()> {
    let total = n_rows * src_row_bytes;
    if total == 0 {
        return Ok(());
    }
    let func = dev.elementwise_fn("native_slice_set_u8")?;
    let stream = dev.stream();
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (offb, drb, srb, nr) = (
        off_bytes as i64,
        dst_row_bytes as i64,
        src_row_bytes as i64,
        n_rows as i64,
    );
    macro_rules! run {
        ($s:expr, $d:expr) => {{
            // byte-pointer kernel; typed slices are fine at the ABI level
            let mut b = stream.launch_builder(&func);
            b.arg($s);
            b.arg($d);
            b.arg(&offb);
            b.arg(&drb);
            b.arg(&srb);
            b.arg(&nr);
            unsafe { b.launch(cfg) }.map_err(|e| Error(format!("slice_set launch: {e}")))?;
            Ok(())
        }};
    }
    match (src, dst) {
        (CudaStorage::F32(s), CudaStorage::F32(d)) => run!(s, d),
        (CudaStorage::F16(s), CudaStorage::F16(d)) => run!(s, d),
        (CudaStorage::BF16(s), CudaStorage::BF16(d)) => run!(s, d),
        (CudaStorage::U32(s), CudaStorage::U32(d)) => run!(s, d),
        (CudaStorage::I32(s), CudaStorage::I32(d)) => run!(s, d),
        (CudaStorage::I64(s), CudaStorage::I64(d)) => run!(s, d),
        (CudaStorage::U8(s), CudaStorage::U8(d)) => run!(s, d),
        _ => Err(Error("slice_set: dtype mismatch".into())),
    }
}

/// In-place `scatter_set` along one dim: dst/src viewed as
/// `[outer, {dst_d|src_d}, inner]`, idx (u32/i64) has src's element layout.
/// Dtype-agnostic byte copy (the graph KV write runs f16/f32 through this).
pub fn scatter_set_storage(
    dev: &CudaDevice,
    dst: &CudaStorage,
    src: &CudaStorage,
    idx: &CudaStorage,
    inner: usize,
    src_d: usize,
    dst_d: usize,
) -> Result<()> {
    let n = src.len();
    if n == 0 {
        return Ok(());
    }
    if dst.dtype() != src.dtype() {
        return Err(Error(format!(
            "scatter_set: dtype mismatch {} vs {}",
            dst.dtype(),
            src.dtype()
        )));
    }
    let name = match idx {
        CudaStorage::U32(_) => "native_scatter_set_u32",
        CudaStorage::I64(_) => "native_scatter_set_i64",
        other => {
            return Err(Error(format!(
                "scatter_set: indexes must be u32/i64, got {}",
                other.dtype()
            )))
        }
    };
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_l = n as i64;
    let (inner_i, srcd_i, dstd_i) = (inner as i32, src_d as i32, dst_d as i32);
    let es_i = src.dtype().size_in_bytes() as i32;
    macro_rules! go {
        ($s:expr, $d:expr, $ix:expr) => {{
            let mut b = stream.launch_builder(&func);
            b.arg($s);
            b.arg($d);
            b.arg($ix);
            b.arg(&n_l);
            b.arg(&inner_i);
            b.arg(&srcd_i);
            b.arg(&dstd_i);
            b.arg(&es_i);
            unsafe { b.launch(cfg) }.map_err(|e| Error(format!("scatter_set launch: {e}")))?;
            Ok(())
        }};
    }
    macro_rules! pairs {
        ($ix:expr) => {
            match (src, dst) {
                (CudaStorage::F32(s), CudaStorage::F32(d)) => go!(s, d, $ix),
                (CudaStorage::F16(s), CudaStorage::F16(d)) => go!(s, d, $ix),
                (CudaStorage::BF16(s), CudaStorage::BF16(d)) => go!(s, d, $ix),
                (CudaStorage::U32(s), CudaStorage::U32(d)) => go!(s, d, $ix),
                (CudaStorage::I64(s), CudaStorage::I64(d)) => go!(s, d, $ix),
                (CudaStorage::U8(s), CudaStorage::U8(d)) => go!(s, d, $ix),
                _ => Err(Error("scatter_set: dtype mismatch".into())),
            }
        };
    }
    match idx {
        CudaStorage::U32(ix) => pairs!(ix),
        CudaStorage::I64(ix) => pairs!(ix),
        _ => unreachable!(),
    }
}

/// `index_select` along one dim: out `[outer, idx_len, inner]` from src
/// `[outer, src_d, inner]` with a 1-D `[idx_len]` index vector (the
/// token-embedding lookup form). Dtype-agnostic byte copy, ids u32/i64.
pub fn index_select_storage(
    dev: &CudaDevice,
    src: &CudaStorage,
    ids: &CudaStorage,
    n_out: usize,
    inner: usize,
    idx_len: usize,
    src_d: usize,
) -> Result<CudaStorage> {
    let name = match ids {
        CudaStorage::U32(_) => "native_index_select_u32",
        CudaStorage::I64(_) => "native_index_select_i64",
        other => {
            return Err(Error(format!(
                "index_select: indexes must be u32/i64, got {}",
                other.dtype()
            )))
        }
    };
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let cfg = LaunchConfig {
        grid_dim: (n_out.max(1).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_l = n_out as i64;
    let (inner_i, idxl_i, srcd_i) = (inner as i32, idx_len as i32, src_d as i32);
    let es_i = src.dtype().size_in_bytes() as i32;
    macro_rules! go {
        ($s:expr, $ix:expr, $t:ty, $variant:ident) => {{
            let out = with_oom_retry(dev, "index_select", || unsafe { stream.alloc::<$t>(n_out) })?;
            if n_out > 0 {
                let mut b = stream.launch_builder(&func);
                b.arg($s);
                b.arg($ix);
                b.arg(&out);
                b.arg(&n_l);
                b.arg(&inner_i);
                b.arg(&idxl_i);
                b.arg(&srcd_i);
                b.arg(&es_i);
                unsafe { b.launch(cfg) }.map_err(|e| Error(format!("index_select launch: {e}")))?;
            }
            Ok(CudaStorage::$variant(out))
        }};
    }
    macro_rules! values {
        ($ix:expr) => {
            match src {
                CudaStorage::F32(s) => go!(s, $ix, f32, F32),
                CudaStorage::F16(s) => go!(s, $ix, half::f16, F16),
                CudaStorage::BF16(s) => go!(s, $ix, half::bf16, BF16),
                CudaStorage::U32(s) => go!(s, $ix, u32, U32),
                CudaStorage::I16(s) => go!(s, $ix, i16, I16),
                CudaStorage::I32(s) => go!(s, $ix, i32, I32),
                CudaStorage::I64(s) => go!(s, $ix, i64, I64),
                CudaStorage::U8(s) => go!(s, $ix, u8, U8),
            }
        };
    }
    match ids {
        CudaStorage::U32(ix) => values!(ix),
        CudaStorage::I64(ix) => values!(ix),
        _ => unreachable!(),
    }
}

/// Gather along one dim (non-dim dims equal): out (idx's element layout,
/// src's dtype). Dtype-agnostic byte copy, idx u32/i64.
pub fn gather_storage(
    dev: &CudaDevice,
    src: &CudaStorage,
    idx: &CudaStorage,
    n_out: usize,
    inner: usize,
    idx_d: usize,
    src_d: usize,
) -> Result<CudaStorage> {
    let name = match idx {
        CudaStorage::U32(_) => "native_gather_u32",
        CudaStorage::I64(_) => "native_gather_i64",
        other => {
            return Err(Error(format!(
                "gather: indexes must be u32/i64, got {}",
                other.dtype()
            )))
        }
    };
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let cfg = LaunchConfig {
        grid_dim: (n_out.max(1).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_l = n_out as i64;
    let (inner_i, idxd_i, srcd_i) = (inner as i32, idx_d as i32, src_d as i32);
    let es_i = src.dtype().size_in_bytes() as i32;
    macro_rules! go {
        ($s:expr, $ix:expr, $t:ty, $variant:ident) => {{
            let out = with_oom_retry(dev, "gather", || unsafe { stream.alloc::<$t>(n_out) })?;
            if n_out > 0 {
                let mut b = stream.launch_builder(&func);
                b.arg($s);
                b.arg($ix);
                b.arg(&out);
                b.arg(&n_l);
                b.arg(&inner_i);
                b.arg(&idxd_i);
                b.arg(&srcd_i);
                b.arg(&es_i);
                unsafe { b.launch(cfg) }.map_err(|e| Error(format!("gather launch: {e}")))?;
            }
            Ok(CudaStorage::$variant(out))
        }};
    }
    macro_rules! values {
        ($ix:expr) => {
            match src {
                CudaStorage::F32(s) => go!(s, $ix, f32, F32),
                CudaStorage::F16(s) => go!(s, $ix, half::f16, F16),
                CudaStorage::BF16(s) => go!(s, $ix, half::bf16, BF16),
                CudaStorage::U32(s) => go!(s, $ix, u32, U32),
                CudaStorage::I16(s) => go!(s, $ix, i16, I16),
                CudaStorage::I32(s) => go!(s, $ix, i32, I32),
                CudaStorage::I64(s) => go!(s, $ix, i64, I64),
                CudaStorage::U8(s) => go!(s, $ix, u8, U8),
            }
        };
    }
    match idx {
        CudaStorage::U32(ix) => values!(ix),
        CudaStorage::I64(ix) => values!(ix),
        _ => unreachable!(),
    }
}

/// Element select on f32 values; condition u8/u32 (non-zero = true).
pub fn where_f32(
    dev: &CudaDevice,
    cond: &CudaStorage,
    t: &CudaSlice<f32>,
    f: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    let name = match cond {
        CudaStorage::U8(_) => "native_where_u8_f32",
        CudaStorage::U32(_) => "native_where_u32_f32",
        other => {
            return Err(Error(format!(
                "where_cond: condition must be u8/u32, got {}",
                other.dtype()
            )))
        }
    };
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "where", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.max(1).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    macro_rules! go {
        ($c:expr) => {{
            let mut b = stream.launch_builder(&func);
            b.arg($c);
            b.arg(t);
            b.arg(f);
            b.arg(&out);
            b.arg(&n_i);
            unsafe { b.launch(cfg) }.map_err(|e| Error(format!("where launch: {e}")))?;
        }};
    }
    match cond {
        CudaStorage::U8(c) => go!(c),
        CudaStorage::U32(c) => go!(c),
        _ => unreachable!(),
    }
    Ok(out)
}

/// `x^e` elementwise.
pub fn pow_f32(dev: &CudaDevice, x: &CudaSlice<f32>, e: f32, n: usize) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_pow_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "pow", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&e);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|er| Error(format!("pow launch: {er}")))?;
    Ok(out)
}

/// Reduce F32 storage over the middle axis of a logical [outer, red, inner] view, on-device.
/// `opcode` 0=sum, 1=max; `scale` post-multiplies each output (1.0 for sum/max, 1/red for mean).
pub fn reduce_dim_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    outer: usize,
    red: usize,
    inner: usize,
    opcode: i32,
    scale: f32,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_reduce_f32")?;
    let stream = dev.stream();
    let n = outer * inner;
    let out = with_oom_retry(dev, "reduce", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (outer_i, red_i, inner_i) = (outer as i32, red as i32, inner as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&outer_i);
    b.arg(&red_i);
    b.arg(&inner_i);
    b.arg(&opcode);
    b.arg(&scale);
    unsafe { b.launch(cfg) }.map_err(|er| Error(format!("reduce launch: {er}")))?;
    Ok(out)
}
