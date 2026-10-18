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
    let stream = dev.stream();
    let total = n_rows * row_bytes;
    macro_rules! run {
        ($t:ty, $variant:ident) => {{
            let elem = std::mem::size_of::<$t>();
            let out = with_oom_retry(dev, "narrow", || unsafe {
                stream.alloc::<$t>(total / elem)
            })?;
            let out = CudaStorage::$variant(out);
            copy_rows(
                dev,
                &out,
                src,
                n_rows,
                row_bytes,
                row_bytes,
                src_row_bytes,
                0,
                off_bytes,
            )
            .map_err(|e| Error(format!("narrow: {e}{}", super::device::stack_note())))?;
            Ok(out)
        }};
    }
    match src {
        CudaStorage::F32(_) => run!(f32, F32),
        CudaStorage::F16(_) => run!(half::f16, F16),
        CudaStorage::BF16(_) => run!(half::bf16, BF16),
        CudaStorage::U32(_) => run!(u32, U32),
        CudaStorage::I16(_) => run!(i16, I16),
        CudaStorage::I32(_) => run!(i32, I32),
        CudaStorage::I64(_) => run!(i64, I64),
        CudaStorage::U8(_) => run!(u8, U8),
    }
}

/// In-place `slice_set`: copy `src` (from byte `src_off_bytes`, viewed as
/// `[n_rows, src_row_bytes]`) into `dst` rows of `dst_row_bytes` at byte offset `off_bytes`
/// (dtype-agnostic, same-variant storages). The copy writes through the dst slice in place -
/// the production fused kernels mutate KV buffers through shared refs the same way. The two
/// ranges do not overlap.
#[allow(clippy::too_many_arguments)]
pub fn slice_set_storage(
    dev: &CudaDevice,
    dst: &CudaStorage,
    src: &CudaStorage,
    n_rows: usize,
    dst_row_bytes: usize,
    src_row_bytes: usize,
    off_bytes: usize,
    src_off_bytes: usize,
) -> Result<()> {
    copy_rows(
        dev,
        dst,
        src,
        n_rows,
        src_row_bytes,
        dst_row_bytes,
        src_row_bytes,
        off_bytes,
        src_off_bytes,
    )
}

/// `n_rows` rows of `width` bytes from `src` (rows `src_pitch` apart from byte `src_off`)
/// into `dst` (rows `dst_pitch` apart from byte `dst_off`), same-variant storages: one
/// device-to-device copy at the card's bandwidth, recorded as a copy node inside a graph.
#[allow(clippy::too_many_arguments)]
fn copy_rows(
    dev: &CudaDevice,
    dst: &CudaStorage,
    src: &CudaStorage,
    n_rows: usize,
    width: usize,
    dst_pitch: usize,
    src_pitch: usize,
    dst_off: usize,
    src_off: usize,
) -> Result<()> {
    use crate::tensor::gpu_runtime::driver::{sys, DevicePtr};
    if n_rows == 0 || width == 0 {
        return Ok(());
    }
    let stream = dev.stream();
    // The copy is a driver call on this card: its context is the thread's current one.
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| Error(format!("device copy: {e:?}")))?;
    macro_rules! ptr {
        ($s:expr) => {{
            let (p, _g) = $s.device_ptr(&stream);
            p
        }};
    }
    let (sp, dp) = match (src, dst) {
        (CudaStorage::F32(s), CudaStorage::F32(d)) => (ptr!(s), ptr!(d)),
        (CudaStorage::F16(s), CudaStorage::F16(d)) => (ptr!(s), ptr!(d)),
        (CudaStorage::BF16(s), CudaStorage::BF16(d)) => (ptr!(s), ptr!(d)),
        (CudaStorage::U32(s), CudaStorage::U32(d)) => (ptr!(s), ptr!(d)),
        (CudaStorage::I16(s), CudaStorage::I16(d)) => (ptr!(s), ptr!(d)),
        (CudaStorage::I32(s), CudaStorage::I32(d)) => (ptr!(s), ptr!(d)),
        (CudaStorage::I64(s), CudaStorage::I64(d)) => (ptr!(s), ptr!(d)),
        (CudaStorage::U8(s), CudaStorage::U8(d)) => (ptr!(s), ptr!(d)),
        _ => return Err(Error("copy: dtype mismatch".into())),
    };
    let (sp, dp) = (sp + src_off as u64, dp + dst_off as u64);
    let r = if n_rows == 1 || (dst_pitch == width && src_pitch == width) {
        unsafe { sys::cuMemcpyDtoDAsync_v2(dp, sp, n_rows * width, stream.cu_stream()) }
    } else {
        let desc = sys::CUDA_MEMCPY2D_st {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: sp,
            srcArray: std::ptr::null_mut(),
            srcPitch: src_pitch,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dp,
            dstArray: std::ptr::null_mut(),
            dstPitch: dst_pitch,
            WidthInBytes: width,
            Height: n_rows,
        };
        unsafe { sys::cuMemcpy2DAsync_v2(&desc, stream.cu_stream()) }
    };
    if r != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(Error(format!("device copy: {r:?}")));
    }
    Ok(())
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

/// Most outputs of a reduction that gets a block each, and the shortest axis it does so for.
const REDUCE_BLOCK_OUTPUTS: usize = 64;
const REDUCE_BLOCK_AXIS: usize = 4096;

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
    let n = outer * inner;
    // Few outputs over a long axis: a block per output, where one thread per output would walk
    // the whole axis alone.
    let blocked = n <= REDUCE_BLOCK_OUTPUTS && red >= REDUCE_BLOCK_AXIS;
    let func = dev.elementwise_fn(if blocked {
        "native_reduce_block_f32"
    } else {
        "native_reduce_f32"
    })?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "reduce", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: if blocked {
            (n as u32, 1, 1)
        } else {
            (n.div_ceil(256) as u32, 1, 1)
        },
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

#[cfg(test)]
mod reduce_tests {
    use crate::tensor::{Device, Tensor};

    /// A long axis reduced by a block per output gives the serial kernel's sum and max.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn a_long_axis_reduces_to_the_same_sum_and_max() {
        let Ok(card) = crate::tensor::cuda::CudaDevice::new(0).map(Device::Cuda) else {
            return;
        };
        let n = 248_320;
        let host: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin()).collect();
        let x = Tensor::from_vec(host.clone(), (1, n), &card).unwrap();
        let sum = x.sum_keepdim(1).unwrap().to_vec2::<f32>().unwrap()[0][0];
        let max = x.max_keepdim(1).unwrap().to_vec2::<f32>().unwrap()[0][0];
        let want_sum: f64 = host.iter().map(|v| *v as f64).sum();
        let want_max = host.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            (sum as f64 - want_sum).abs() < 1e-2 * want_sum.abs().max(1.0),
            "{sum} {want_sum}"
        );
        assert_eq!(max, want_max);
    }
}

#[cfg(test)]
mod copy_tests {
    use crate::tensor::{DType, Device, Tensor};

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// A write into rows, a write from a narrowed source, and a narrow of a middle dim give
    /// what the host's copies give, on either card.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn slice_set_and_narrow_copy_the_rows_they_name() {
        for ordinal in 0..2 {
            let Ok(dev) = Device::new_cuda(ordinal) else {
                continue;
            };
            let v: Vec<f32> = (0..2 * 5 * 3).map(|i| i as f32).collect();
            let dst = Tensor::zeros_on((2, 5, 3), DType::F32, &dev).unwrap();
            let src = Tensor::from_vec(v[..12].to_vec(), (2, 2, 3), &dev).unwrap();
            dst.slice_set(&src, 1, 2).unwrap();
            let got = host(&dst);
            let mut want = vec![0f32; 30];
            for o in 0..2 {
                for j in 0..6 {
                    want[o * 15 + 6 + j] = v[o * 6 + j];
                }
            }
            assert_eq!(got, want, "rows written at an offset on card {ordinal}");
            let whole = Tensor::from_vec(v.clone(), (10, 3), &dev).unwrap();
            let into = Tensor::zeros_on((2, 3), DType::F32, &dev).unwrap();
            into.slice_set(&whole.narrow(0, 4, 2).unwrap(), 0, 0)
                .unwrap();
            assert_eq!(
                host(&into),
                v[12..18].to_vec(),
                "a narrowed source on card {ordinal}"
            );
            let mid = Tensor::from_vec(v.clone(), (2, 5, 3), &dev)
                .unwrap()
                .narrow(1, 1, 3)
                .unwrap();
            let mut want = Vec::new();
            for o in 0..2 {
                want.extend_from_slice(&v[o * 15 + 3..o * 15 + 12]);
            }
            assert_eq!(
                host(&mid.contiguous().unwrap()),
                want,
                "a middle-dim narrow on card {ordinal}"
            );
        }
    }
}
