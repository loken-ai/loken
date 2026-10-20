//! A compensated quantisation held on the card from the first block to the last.
//!
//! The correction of `inference::load::compensate`, with every matrix kept on the device in column
//! order, so a block of columns is a contiguous stretch of each: the weights, the calibration rows,
//! those rows solved against their moment, and the correction carried from block to block. A block
//! is copied, corrected by one product, encoded by the format's kernel - which also writes what the
//! blocks decode to - and its error carried on by three products. The inverse each block needs
//! depends on the calibration rows alone: it is formed once per factor, its corner on the card and
//! its inverse on the host, so a projection runs on the card without waiting on the host.

use super::*;
use crate::inference::load::compensate::stage::{time, Stage};
use crate::tensor::gpu_runtime::cublas::sys;
use crate::tensor::gpu_runtime::driver::DevicePtr;
use crate::tensor::quantized::GgmlDType;

/// Values per block of the formats this carries: one block per row of a block of columns.
const BLOCK: usize = 256;

/// Calibration rows and the rows solved against their moment, on a card, `n` by `cols` column-major,
/// with what the correction needs from them alone: for every block of columns but the last, the
/// inverse of its corner of the moment. The buffers may be longer than what they hold.
pub struct CardFactor {
    n: usize,
    cols: usize,
    lambda: f32,
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
    /// One `BLOCK` by `BLOCK` inverse a block, column order, laid end to end.
    inverses: Option<CudaSlice<f32>>,
    /// Whether a block's corner was positive; a block whose corner was not keeps its error.
    positive: Vec<bool>,
}

/// The card buffers a worker keeps from one projection to the next. Allocating or freeing on a
/// card waits for all the work queued on it, from every stream, so a worker grows these and never
/// gives them back while it runs. They belong to the card of the first call that uses them.
#[derive(Default)]
pub struct Scratch {
    ordinal: Option<usize>,
    weights: Option<CudaSlice<f32>>,
    packed: Option<CudaSlice<u8>>,
    scales: Option<CudaSlice<f32>>,
    carried: Option<CudaSlice<f32>>,
    block: Option<CudaSlice<f32>>,
    got: Option<CudaSlice<f32>>,
    lost: Option<CudaSlice<f32>>,
    imp: Option<CudaSlice<f32>>,
    bytes: Option<CudaSlice<u8>>,
    corner: Option<CudaSlice<f32>>,
    x: Option<CudaSlice<f32>>,
    y: Option<CudaSlice<f32>>,
    inverses: Option<CudaSlice<f32>>,
    grid: Option<CudaSlice<f32>>,
    book: Option<CudaSlice<i32>>,
    pinned: Option<Pinned>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every buffer dropped when `dev` is not the card they were made on.
    fn on(&mut self, dev: &CudaDevice) {
        if self.ordinal != Some(dev.ordinal()) {
            *self = Self {
                ordinal: Some(dev.ordinal()),
                ..Self::default()
            };
        }
    }

    /// A factor's buffers, kept for the next factor built here.
    pub fn keep(&mut self, factor: CardFactor) {
        self.x = Some(factor.x);
        self.y = Some(factor.y);
        if factor.inverses.is_some() {
            self.inverses = factor.inverses;
        }
    }
}

/// The buffer in `slot` when it holds `len` elements, a new one otherwise.
fn take<T: crate::tensor::gpu_runtime::driver::DeviceRepr>(
    slot: &mut Option<CudaSlice<T>>,
    dev: &CudaDevice,
    len: usize,
) -> Result<CudaSlice<T>> {
    match slot.take() {
        Some(s) if s.len() >= len => Ok(s),
        old => {
            drop(old);
            with_oom_retry(dev, "card correction", || unsafe {
                dev.stream().alloc::<T>(len.max(1))
            })
        }
    }
}

/// Page-locked host memory uploads are staged through, so a copy to the card runs beside the
/// card's work instead of holding the host until it is done. Filled from the start after each
/// wait on the stream; freed behind the copies still reading it.
struct Pinned {
    ptr: *mut std::ffi::c_void,
    cap: usize,
    used: usize,
    stream: Arc<CudaStream>,
}

// Safety: the buffer is owned by one scratch, used by one thread at a time.
unsafe impl Send for Pinned {}

impl Drop for Pinned {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        // Safety: allocated by `malloc_host`, and no copy reads it any more.
        let _ = unsafe { crate::tensor::gpu_runtime::driver::result::free_host(self.ptr) };
    }
}

/// The bytes a staging buffer starts with; it doubles when an upload does not fit.
const PINNED_START: usize = 16 << 20;

/// `src` into the start of `dst`, staged through `pinned`: the copy is queued on the stream and the
/// call returns before it runs.
fn put<T: crate::tensor::gpu_runtime::driver::DeviceRepr>(
    pinned: &mut Option<Pinned>,
    dev: &CudaDevice,
    src: &[T],
    dst: &mut CudaSlice<T>,
    ctx: &str,
) -> Result<()> {
    let stream = dev.stream();
    let bytes = std::mem::size_of_val(src);
    if bytes == 0 {
        return Ok(());
    }
    let align = 256;
    let at = pinned
        .as_ref()
        .map_or(0, |p| p.used.div_ceil(align) * align);
    if pinned.as_ref().is_none_or(|p| at + bytes > p.cap) {
        // The copies already queued from this buffer finish before it is refilled or freed.
        stream
            .synchronize()
            .map_err(|e| Error(format!("{ctx}: {e}")))?;
        let cap = pinned.as_ref().map_or(PINNED_START, |p| p.cap);
        if let Some(p) = pinned.as_mut().filter(|p| p.cap >= bytes) {
            p.used = 0;
        } else {
            *pinned = None;
            let cap = (cap * 2).max(bytes.next_power_of_two());
            stream
                .context()
                .bind_to_thread()
                .map_err(|e| Error(format!("{ctx}: {e}")))?;
            let ptr = unsafe { crate::tensor::gpu_runtime::driver::result::malloc_host(cap, 0) }
                .map_err(|e| Error(format!("{ctx}: page-locked buffer: {e}")))?;
            *pinned = Some(Pinned {
                ptr,
                cap,
                used: 0,
                stream: stream.clone(),
            });
        }
    }
    let p = pinned.as_mut().expect("a staging buffer");
    let at = p.used.div_ceil(align) * align;
    // Safety: [at, at + bytes) lies within the buffer, and no queued copy reads it: copies read
    // the stretch before `used`, and the buffer is refilled only after the stream is waited on.
    let staged = unsafe {
        let dst = (p.ptr as *mut u8).add(at);
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst, bytes);
        std::slice::from_raw_parts(dst as *const T, src.len())
    };
    p.used = at + bytes;
    stream
        .memcpy_htod(staged, &mut dst.slice_mut(0..src.len()))
        .map_err(|e| alloc_err(ctx, e))
}

/// The staging buffer marked free: the stream has been waited on since its copies were queued.
fn waited(pinned: &mut Option<Pinned>) {
    if let Some(p) = pinned {
        p.used = 0;
    }
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

/// A block inverter: a block's `X_B^T Y_B`, `BLOCK` by `BLOCK` row-major, and the ridge, to the
/// inverse of that block's corner of the moment, or `None` when it is not positive.
pub type Invert<'a> = dyn Fn(&[f32], f32) -> Option<Vec<f32>> + Sync + 'a;

/// Upload a factor: `x` the calibration rows and `y` those rows solved against their moment, both
/// `n` by `cols` row-major, and the ridge the moment was taken with. Every block's corner of the
/// moment is formed on the card and inverted by `invert` on the host, the blocks across the rayon
/// pool, once for every projection the factor serves.
pub fn card_factor(
    dev: &CudaDevice,
    x: &[f32],
    y: &[f32],
    shape: (usize, usize),
    lambda: f32,
    invert: &Invert,
) -> Result<CardFactor> {
    card_factor_in(dev, x, y, shape, lambda, invert, &mut Scratch::new())
}

/// `card_factor` in buffers from `scratch`; `Scratch::keep` takes them back.
pub fn card_factor_in(
    dev: &CudaDevice,
    x: &[f32],
    y: &[f32],
    (n, cols): (usize, usize),
    lambda: f32,
    invert: &Invert,
    scratch: &mut Scratch,
) -> Result<CardFactor> {
    use rayon::prelude::*;
    if x.len() != n * cols || y.len() != n * cols || cols % BLOCK != 0 {
        return Err(Error(format!(
            "card factor: {} and {} values for {n} rows of {cols}",
            x.len(),
            y.len()
        )));
    }
    scratch.on(dev);
    let stream = dev.stream();
    let upload = |m: &[f32],
                  slot: &mut Option<CudaSlice<f32>>,
                  pinned: &mut Option<Pinned>|
     -> Result<CudaSlice<f32>> {
        let mut buf = take(slot, dev, n * cols)?;
        time(Stage::Upload, || {
            put(
                pinned,
                dev,
                &to_columns(m, n, cols),
                &mut buf,
                "card factor upload",
            )
        })?;
        Ok(buf)
    };
    let x = upload(x, &mut scratch.x, &mut scratch.pinned)?;
    let y = upload(y, &mut scratch.y, &mut scratch.pinned)?;
    let corners = cols / BLOCK - 1;
    if corners == 0 {
        return Ok(CardFactor {
            n,
            cols,
            lambda,
            x,
            y,
            inverses: None,
            positive: Vec::new(),
        });
    }
    let square = BLOCK * BLOCK;
    let corner = take(&mut scratch.corner, dev, corners * square)?;
    let blas = dev.blas()?;
    let handle = *blas.handle();
    let host = time(Stage::Compensate, || -> Result<Vec<f32>> {
        {
            let (x_ptr, _gx) = x.device_ptr(stream);
            let (y_ptr, _gy) = y.device_ptr(stream);
            let (h, _gh) = corner.device_ptr(stream);
            for b in 0..corners {
                let a = b * BLOCK;
                unsafe {
                    // H = X_B^T Y_B
                    gemm(
                        handle,
                        (true, false),
                        (BLOCK, BLOCK, n),
                        ((x_ptr as *const f32).add(a * n), n),
                        ((y_ptr as *const f32).add(a * n), n),
                        0.0,
                        (h as *mut f32).add(b * square),
                    )?;
                }
            }
        }
        stream
            .clone_dtoh(&corner.slice(0..corners * square))
            .map_err(|e| Error(format!("corner download: {e}")))
    });
    scratch.corner = Some(corner);
    let host = host?;
    waited(&mut scratch.pinned);
    let inverted: Vec<Option<Vec<f32>>> = time(Stage::Inverse, || {
        host.par_chunks(square)
            .map(|c| {
                // Column-major on the card, row-major for the host: the transpose.
                let row_major: Vec<f32> = (0..square)
                    .map(|i| c[(i % BLOCK) * BLOCK + i / BLOCK])
                    .collect();
                invert(&row_major, lambda).map(|inv| to_columns(&inv, BLOCK, BLOCK))
            })
            .collect()
    });
    let positive: Vec<bool> = inverted.iter().map(Option::is_some).collect();
    let mut laid = vec![0f32; corners * square];
    for (slot, inv) in laid.chunks_mut(square).zip(&inverted) {
        if let Some(inv) = inv {
            slot.copy_from_slice(inv);
        }
    }
    let mut inverses = take(&mut scratch.inverses, dev, corners * square)?;
    time(Stage::Upload, || {
        put(
            &mut scratch.pinned,
            dev,
            &laid,
            &mut inverses,
            "inverse upload",
        )
    })?;
    Ok(CardFactor {
        n,
        cols,
        lambda,
        x,
        y,
        inverses: Some(inverses),
        positive,
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

/// Quantise `w` (`rows` by `cols`, row-major) to `dtype` - iq2_xxs, iq3_xxs, q2_K or q3_K, each
/// weighted by `importance` - carrying each block's error onto the columns after it through
/// `factor`. Returns the blocks as the file stores them, row by row.
pub fn quantize_compensated_on_card(
    dev: &CudaDevice,
    dtype: GgmlDType,
    w: &[f32],
    (rows, cols): (usize, usize),
    importance: &[f32],
    factor: &CardFactor,
) -> Result<Vec<u8>> {
    if w.len() != rows * cols {
        return Err(Error(format!(
            "card correction: {} weights for {rows} by {cols}",
            w.len()
        )));
    }
    let columns = time(Stage::Upload, || to_columns(w, rows, cols));
    quantize_columns_on_card(
        dev,
        dtype,
        &columns,
        (rows, cols),
        importance,
        factor,
        &mut Scratch::new(),
    )
}

/// `quantize_compensated_on_card` with `w` in column order, value `(r, c)` at `c * rows + r`, in
/// buffers from `scratch`. The card runs every block without waiting on the host, and the blocks
/// come back in one copy.
pub fn quantize_columns_on_card(
    dev: &CudaDevice,
    dtype: GgmlDType,
    w: &[f32],
    (rows, cols): (usize, usize),
    importance: &[f32],
    factor: &CardFactor,
    scratch: &mut Scratch,
) -> Result<Vec<u8>> {
    if w.len() != rows * cols {
        return Err(Error(format!(
            "card correction: {} weights for {rows} by {cols}",
            w.len()
        )));
    }
    scratch.on(dev);
    let mut weights = take(&mut scratch.weights, dev, rows * cols)?;
    time(Stage::Upload, || {
        put(
            &mut scratch.pinned,
            dev,
            w,
            &mut weights,
            "card correction upload",
        )
    })?;
    let out = quantize_on_card(
        dev,
        dtype,
        &weights,
        (rows, cols),
        importance,
        factor,
        scratch,
    );
    scratch.weights = Some(weights);
    out
}

/// `quantize_columns_on_card` from a block-scaled fp4 weight as the checkpoint stores it:
/// `packed` `rows` by `cols / 2` bytes of e2m1 pairs, `scales` one decoded power of two per `block`
/// values of a row. The card decodes it into column order itself, so only the packed bytes cross.
#[allow(clippy::too_many_arguments)]
pub fn quantize_fp4_on_card(
    dev: &CudaDevice,
    dtype: GgmlDType,
    packed: &[u8],
    scales: &[f32],
    block: usize,
    (rows, cols): (usize, usize),
    importance: &[f32],
    factor: &CardFactor,
    scratch: &mut Scratch,
) -> Result<Vec<u8>> {
    if block == 0 || cols % 2 != 0 || cols % block != 0 {
        return Err(Error(format!(
            "card fp4: {cols} columns do not tile by {block}"
        )));
    }
    if packed.len() != rows * cols / 2 || scales.len() != rows * cols / block {
        return Err(Error(format!(
            "card fp4: {} bytes and {} scales for {rows} by {cols}",
            packed.len(),
            scales.len()
        )));
    }
    scratch.on(dev);
    let stream = dev.stream();
    let mut bytes = take(&mut scratch.packed, dev, packed.len())?;
    let mut scale = take(&mut scratch.scales, dev, scales.len())?;
    let mut weights = take(&mut scratch.weights, dev, rows * cols)?;
    let decoded = (|| -> Result<()> {
        time(Stage::Upload, || -> Result<()> {
            put(
                &mut scratch.pinned,
                dev,
                packed,
                &mut bytes,
                "card fp4 upload",
            )?;
            put(
                &mut scratch.pinned,
                dev,
                scales,
                &mut scale,
                "card fp4 scale upload",
            )
        })?;
        let func = dev.quantized_fn("fp4_to_columns")?;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((rows * cols).div_ceil(threads as usize) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, c, b) = (rows as i32, cols as i32, block as i32);
        let mut launch = stream.launch_builder(&func);
        launch.arg(&bytes);
        launch.arg(&scale);
        launch.arg(&mut weights);
        launch.arg(&r);
        launch.arg(&c);
        launch.arg(&b);
        unsafe { launch.launch(cfg) }
            .map(|_| ())
            .map_err(|e| Error(format!("fp4_to_columns launch: {e}")))
    })();
    let out = decoded.and_then(|()| {
        quantize_on_card(
            dev,
            dtype,
            &weights,
            (rows, cols),
            importance,
            factor,
            scratch,
        )
    });
    scratch.packed = Some(bytes);
    scratch.scales = Some(scale);
    scratch.weights = Some(weights);
    out
}

/// The correction over `weights`, the first `rows * cols` values of the buffer, on the card in
/// column order.
fn quantize_on_card(
    dev: &CudaDevice,
    dtype: GgmlDType,
    weights: &CudaSlice<f32>,
    (rows, cols): (usize, usize),
    importance: &[f32],
    factor: &CardFactor,
    scratch: &mut Scratch,
) -> Result<Vec<u8>> {
    use crate::tensor::quant_cpu::{BlockIq2Xxs, BlockIq3Xxs, BlockQ2K, BlockQ3K};
    let (kernel, block_bytes, threads) = match dtype {
        GgmlDType::Iq2Xxs => ("requant_iq2_xxs", std::mem::size_of::<BlockIq2Xxs>(), 32u32),
        GgmlDType::Iq3Xxs => (
            "requant_iq3_xxs",
            std::mem::size_of::<BlockIq3Xxs>(),
            super::quant::IQ3_XXS_THREADS,
        ),
        GgmlDType::Q2K => ("requant_q2_k_guided", std::mem::size_of::<BlockQ2K>(), 32),
        GgmlDType::Q3K => ("requant_q3_k_guided", std::mem::size_of::<BlockQ3K>(), 32),
        other => {
            return Err(Error(format!(
                "{other:?} has no kernel that carries a correction"
            )))
        }
    };
    if cols % BLOCK != 0
        || weights.len() < rows * cols
        || importance.len() != cols
        || factor.cols != cols
    {
        return Err(Error(format!(
            "card correction: {rows} by {cols}, {} weights, {} importances, a factor of {}",
            weights.len(),
            importance.len(),
            factor.cols
        )));
    }
    let (n, lambda) = (factor.n, factor.lambda);
    let per_row = cols / BLOCK;
    let stream = dev.stream();
    if dtype == GgmlDType::Iq2Xxs && scratch.grid.is_none() {
        let g: Vec<f32> = crate::tensor::quant_cpu::IQ2XXS_GRID
            .iter()
            .flat_map(|p| p.to_le_bytes().map(|b| b as f32))
            .collect();
        scratch.grid = Some(
            stream
                .clone_htod(&g)
                .map_err(|e| alloc_err("iq2_xxs grid upload", e))?,
        );
    }
    if dtype == GgmlDType::Iq3Xxs && scratch.book.is_none() {
        scratch.book = Some(
            stream
                .clone_htod(&crate::tensor::quant_cpu::iq3_xxs_codebook())
                .map_err(|e| alloc_err("iq3_xxs codebook upload", e))?,
        );
    }
    let mut imp = take(&mut scratch.imp, dev, cols)?;
    let mut carried = take(&mut scratch.carried, dev, rows * n)?;
    let block = take(&mut scratch.block, dev, rows * BLOCK)?;
    let got = take(&mut scratch.got, dev, rows * BLOCK)?;
    let lost = take(&mut scratch.lost, dev, rows * BLOCK)?;
    // Block `b` of every row, the rows end to end, block after block.
    let mut bytes = take(&mut scratch.bytes, dev, per_row * rows * block_bytes)?;
    let mut run = || -> Result<Vec<u8>> {
        time(Stage::Upload, || {
            put(
                &mut scratch.pinned,
                dev,
                importance,
                &mut imp,
                "importance upload",
            )
        })?;
        stream
            .memset_zeros(&mut carried.slice_mut(0..rows * n))
            .map_err(|e| alloc_err("card correction", e))?;
        let grid = scratch.grid.as_ref().filter(|_| dtype == GgmlDType::Iq2Xxs);
        let book = scratch.book.as_ref().filter(|_| dtype == GgmlDType::Iq3Xxs);
        // The grid formats' kernels end on a flag that says the importance is there.
        let flags_importance = grid.is_some() || book.is_some();
        let func = dev.quantized_fn(kernel)?;
        let blas = dev.blas()?;
        let handle = *blas.handle();

        for b in 0..per_row {
            let a = b * BLOCK;
            {
                let (w_ptr, _gw) = weights.device_ptr(stream);
                let (c_ptr, _gc) = carried.device_ptr(stream);
                let (y_ptr, _gy) = factor.y.device_ptr(stream);
                let (blk, _gb) = block.device_ptr(stream);
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
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let (block_step, value_step, nblocks, one, yes) =
                (1i64, rows as i64, rows as i32, 1i32, 1i32);
            let imp_b = imp.slice(a..a + BLOCK);
            let mut out_b = bytes.slice_mut(b * rows * block_bytes..(b + 1) * rows * block_bytes);
            let mut launch = stream.launch_builder(&func);
            launch.arg(&block);
            launch.arg(&block_step);
            launch.arg(&value_step);
            launch.arg(&imp_b);
            if let Some(g) = grid {
                launch.arg(g);
            }
            if let Some(b) = book {
                launch.arg(b);
            }
            launch.arg(&mut out_b);
            launch.arg(&got);
            launch.arg(&yes);
            launch.arg(&nblocks);
            launch.arg(&one);
            if flags_importance {
                launch.arg(&yes);
            }
            unsafe { launch.launch(cfg) }.map_err(|e| Error(format!("{kernel} launch: {e}")))?;
            if b + 1 == per_row {
                break;
            }
            if !factor.positive[b] {
                continue;
            }
            let Some(inverses) = &factor.inverses else {
                continue;
            };
            {
                let (blk, _gb) = block.device_ptr(stream);
                let (g, _gg) = got.device_ptr(stream);
                let (inv_ptr, _gi) = inverses.device_ptr(stream);
                let (l, _gl) = lost.device_ptr(stream);
                let (x_ptr, _gx) = factor.x.device_ptr(stream);
                let (c, _gc) = carried.device_ptr(stream);
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
                        sys::cublasSaxpy_v2(
                            handle,
                            len,
                            &one,
                            blk as *const f32,
                            1,
                            g as *mut f32,
                            1,
                        ),
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
                        ((inv_ptr as *const f32).add(b * BLOCK * BLOCK), BLOCK),
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
        let encoded = time(Stage::Encode, || {
            stream
                .clone_dtoh(&bytes.slice(0..per_row * rows * block_bytes))
                .map_err(|e| Error(format!("{kernel} download: {e}")))
        })?;
        waited(&mut scratch.pinned);
        // Block-major from the card, row-major for the file.
        let mut out = vec![0u8; rows * per_row * block_bytes];
        for (r, row) in out.chunks_mut(per_row * block_bytes).enumerate() {
            for (b, slot) in row.chunks_mut(block_bytes).enumerate() {
                let at = (b * rows + r) * block_bytes;
                slot.copy_from_slice(&encoded[at..at + block_bytes]);
            }
        }
        Ok(out)
    };
    let out = run();
    scratch.imp = Some(imp);
    scratch.carried = Some(carried);
    scratch.block = Some(block);
    scratch.got = Some(got);
    scratch.lost = Some(lost);
    scratch.bytes = Some(bytes);
    out
}
