//! A compensated quantisation held on the card from the first block to the last.
//!
//! The correction of `inference::load::compensate`, with every matrix kept on the device in column
//! order, so a block of columns is a contiguous stretch of each: the weights, the calibration rows,
//! those rows solved against their moment, and the correction carried from block to block. A block
//! is copied, corrected by one product, encoded by the format's kernel - which also writes what the
//! blocks decode to - and its error carried on by three products. Only the block's own corner of
//! the inverse moment leaves the card: `invert` turns it into the inverse it needs, on the host,
//! where a matrix one block wide costs nothing.

use super::*;
use crate::tensor::quantized::GgmlDType;
use cudarc::cublas::sys;
use cudarc::driver::{DevicePtr, DevicePtrMut};

/// Values per block of the formats this carries: one block per row of a block of columns.
const BLOCK: usize = 256;

/// Calibration rows and the rows solved against their moment, on a card, `n` by `cols` column-major.
pub struct CardFactor {
    n: usize,
    cols: usize,
    lambda: f32,
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
}

/// `m` row-major, `rows` by `cols`, in column order.
fn to_columns(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = vec![0f32; m.len()];
    out.par_chunks_mut(rows)
        .enumerate()
        .for_each(|(c, column)| {
            for (r, v) in column.iter_mut().enumerate() {
                *v = m[r * cols + c];
            }
        });
    out
}

/// Upload a factor: `x` the calibration rows and `y` those rows solved against their moment, both
/// `n` by `cols` row-major, and the ridge the moment was taken with.
pub fn card_factor(
    dev: &CudaDevice,
    x: &[f32],
    y: &[f32],
    (n, cols): (usize, usize),
    lambda: f32,
) -> Result<CardFactor> {
    if x.len() != n * cols || y.len() != n * cols {
        return Err(Error(format!(
            "card factor: {} and {} values for {n} rows of {cols}",
            x.len(),
            y.len()
        )));
    }
    let stream = dev.stream();
    let upload = |m: &[f32]| {
        stream
            .clone_htod(&to_columns(m, n, cols))
            .map_err(|e| alloc_err("card factor upload", e))
    };
    Ok(CardFactor {
        n,
        cols,
        lambda,
        x: upload(x)?,
        y: upload(y)?,
    })
}

fn status(ctx: &str, s: sys::cublasStatus_t) -> Result<()> {
    if s == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error(format!("cublas {ctx}: {s:?}")))
    }
}

/// `c = alpha op(a) op(b) + beta c`, column-major, through raw device pointers.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm(
    handle: sys::cublasHandle_t,
    (ta, tb): (bool, bool),
    (m, n, k): (usize, usize, usize),
    a: (*const f32, usize),
    b: (*const f32, usize),
    beta: f32,
    c: *mut f32,
) -> Result<()> {
    use sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
    let op = |t: bool| if t { CUBLAS_OP_T } else { CUBLAS_OP_N };
    let alpha = 1.0f32;
    status(
        "gemm",
        sys::cublasSgemm_v2(
            handle,
            op(ta),
            op(tb),
            m as i32,
            n as i32,
            k as i32,
            &alpha,
            a.0,
            a.1 as i32,
            b.0,
            b.1 as i32,
            &beta,
            c,
            m as i32,
        ),
    )
}

/// Quantise `w` (`rows` by `cols`, row-major) to `dtype` - iq2_xxs, or q2_K weighted by
/// `importance` - carrying each block's error onto the columns after it through `factor`. `invert`
/// takes a block's `X_B^T Y_B`, `BLOCK` by `BLOCK` row-major, and returns the inverse of that
/// block's corner of the moment, or `None` when it is not positive, which leaves the block's error
/// where it is. Returns the blocks as the file stores them, row by row.
pub fn quantize_compensated_on_card(
    dev: &CudaDevice,
    dtype: GgmlDType,
    w: &[f32],
    (rows, cols): (usize, usize),
    importance: &[f32],
    factor: &CardFactor,
    invert: &dyn Fn(&[f32], f32) -> Option<Vec<f32>>,
) -> Result<Vec<u8>> {
    let (kernel, block_bytes) = match dtype {
        GgmlDType::Iq2Xxs => ("requant_iq2_xxs", 66usize),
        GgmlDType::Q2K => ("requant_q2_k_guided", 84usize),
        other => {
            return Err(Error(format!(
                "{other:?} has no kernel that carries a correction"
            )))
        }
    };
    if cols % BLOCK != 0
        || w.len() != rows * cols
        || importance.len() != cols
        || factor.cols != cols
    {
        return Err(Error(format!(
            "card correction: {rows} by {cols}, {} weights, {} importances, a factor of {}",
            w.len(),
            importance.len(),
            factor.cols
        )));
    }
    let (n, lambda) = (factor.n, factor.lambda);
    let per_row = cols / BLOCK;
    let stream = dev.stream();
    let alloc =
        |len: usize| with_oom_retry(dev, "card correction", || stream.alloc_zeros::<f32>(len));
    let weights = stream
        .clone_htod(&to_columns(w, rows, cols))
        .map_err(|e| alloc_err("card correction upload", e))?;
    let mut carried = alloc(rows * n)?;
    let mut block = alloc(rows * BLOCK)?;
    let mut got = alloc(rows * BLOCK)?;
    let mut lost = alloc(rows * BLOCK)?;
    let mut corner = alloc(BLOCK * BLOCK)?;
    let mut inverse = alloc(BLOCK * BLOCK)?;
    let mut imp = alloc(BLOCK)?;
    let bytes = with_oom_retry(dev, "card correction", || unsafe {
        stream.alloc::<u8>(rows * block_bytes)
    })?;
    let grid = if dtype == GgmlDType::Iq2Xxs {
        let g: Vec<f32> = crate::tensor::quant_cpu::IQ2XXS_GRID
            .iter()
            .flat_map(|p| p.to_le_bytes().map(|b| b as f32))
            .collect();
        Some(
            stream
                .clone_htod(&g)
                .map_err(|e| alloc_err("iq2_xxs grid upload", e))?,
        )
    } else {
        None
    };
    let func = dev.quantized_fn(kernel)?;
    let blas = dev.blas()?;
    let handle = *blas.handle();
    let mut out = vec![0u8; rows * per_row * block_bytes];
    let mut host_corner = vec![0f32; BLOCK * BLOCK];

    for b in 0..per_row {
        let a = b * BLOCK;
        {
            let (w_ptr, _gw) = weights.device_ptr(&stream);
            let (c_ptr, _gc) = carried.device_ptr(&stream);
            let (y_ptr, _gy) = factor.y.device_ptr(&stream);
            let (blk, _gb) = block.device_ptr_mut(&stream);
            let (w_ptr, c_ptr, y_ptr, blk) = (
                w_ptr as *const f32,
                c_ptr as *const f32,
                y_ptr as *const f32,
                blk as *mut f32,
            );
            unsafe {
                status(
                    "copy",
                    sys::cublasScopy_v2(
                        handle,
                        (rows * BLOCK) as i32,
                        w_ptr.add(a * rows),
                        1,
                        blk,
                        1,
                    ),
                )?;
                // block = W_B + C Y_B
                gemm(
                    handle,
                    (false, false),
                    (rows, BLOCK, n),
                    (c_ptr, rows),
                    (y_ptr.add(a * n), n),
                    1.0,
                    blk,
                )?;
            }
        }
        stream
            .memcpy_htod(&importance[a..a + BLOCK], &mut imp)
            .map_err(|e| alloc_err("importance upload", e))?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (block_step, value_step, nblocks, one, yes) =
            (1i64, rows as i64, rows as i32, 1i32, 1i32);
        let mut launch = stream.launch_builder(&func);
        launch.arg(&block);
        launch.arg(&block_step);
        launch.arg(&value_step);
        launch.arg(&imp);
        if let Some(g) = &grid {
            launch.arg(g);
        }
        launch.arg(&bytes);
        launch.arg(&got);
        launch.arg(&yes);
        launch.arg(&nblocks);
        launch.arg(&one);
        if grid.is_some() {
            launch.arg(&yes);
        }
        unsafe { launch.launch(cfg) }.map_err(|e| Error(format!("{kernel} launch: {e}")))?;
        let encoded = stream
            .clone_dtoh(&bytes)
            .map_err(|e| Error(format!("{kernel} download: {e}")))?;
        for r in 0..rows {
            let at = (r * per_row + b) * block_bytes;
            out[at..at + block_bytes]
                .copy_from_slice(&encoded[r * block_bytes..(r + 1) * block_bytes]);
        }
        if b + 1 == per_row {
            break;
        }

        {
            let (x_ptr, _gx) = factor.x.device_ptr(&stream);
            let (y_ptr, _gy) = factor.y.device_ptr(&stream);
            let (h, _gh) = corner.device_ptr_mut(&stream);
            unsafe {
                // H = X_B^T Y_B
                gemm(
                    handle,
                    (true, false),
                    (BLOCK, BLOCK, n),
                    ((x_ptr as *const f32).add(a * n), n),
                    ((y_ptr as *const f32).add(a * n), n),
                    0.0,
                    h as *mut f32,
                )?;
            }
        }
        stream
            .memcpy_dtoh(&corner, &mut host_corner)
            .map_err(|e| Error(format!("corner download: {e}")))?;
        // Column-major on the card, row-major for the host: the transpose.
        let row_major: Vec<f32> = (0..BLOCK * BLOCK)
            .map(|i| host_corner[(i % BLOCK) * BLOCK + i / BLOCK])
            .collect();
        let Some(inv) = invert(&row_major, lambda) else {
            continue;
        };
        stream
            .memcpy_htod(&to_columns(&inv, BLOCK, BLOCK), &mut inverse)
            .map_err(|e| alloc_err("inverse upload", e))?;
        {
            let (blk, _gb) = block.device_ptr(&stream);
            let (g, _gg) = got.device_ptr_mut(&stream);
            let (inv_ptr, _gi) = inverse.device_ptr(&stream);
            let (l, _gl) = lost.device_ptr_mut(&stream);
            let (x_ptr, _gx) = factor.x.device_ptr(&stream);
            let (c, _gc) = carried.device_ptr_mut(&stream);
            let len = (rows * BLOCK) as i32;
            let (minus, scale) = (-1.0f32, 1.0f32 / lambda);
            let one = 1.0f32;
            unsafe {
                // got = (block - got) / lambda: what the block lost.
                status(
                    "scal",
                    sys::cublasSscal_v2(handle, len, &minus, g as *mut f32, 1),
                )?;
                status(
                    "axpy",
                    sys::cublasSaxpy_v2(handle, len, &one, blk as *const f32, 1, g as *mut f32, 1),
                )?;
                status(
                    "scal",
                    sys::cublasSscal_v2(handle, len, &scale, g as *mut f32, 1),
                )?;
                // lost = got inverse; carried += lost X_B^T
                gemm(
                    handle,
                    (false, false),
                    (rows, BLOCK, BLOCK),
                    (g as *const f32, rows),
                    (inv_ptr as *const f32, BLOCK),
                    0.0,
                    l as *mut f32,
                )?;
                gemm(
                    handle,
                    (false, true),
                    (rows, n, BLOCK),
                    (l as *const f32, rows),
                    ((x_ptr as *const f32).add(a * n), n),
                    1.0,
                    c as *mut f32,
                )?;
            }
        }
    }
    Ok(out)
}
