//! Quantise a matrix so that what one block of columns loses, the columns after it absorb.
//!
//! A projection is judged by what it does to activations, so the quantity to minimise is the error
//! at the output, not at the weights. Against the second moment of the calibration rows, the error
//! a block leaves is partly cancelled by the columns that come after it, and the correction is the
//! one least squares gives - the family of methods that quantise a column at a time and repair the
//! rest.
//!
//! The second moment is never formed. Calibration rows are few beside columns - tens of rows
//! against thousands of columns - so every step goes through the small matrix those rows span: one
//! factorisation the size of the row count, and updates whose cost follows the row count rather
//! than the column count.
//!
//! What comes out is the block quantiser's own format, unchanged: this decides what each block is
//! handed, not how it is encoded.

use crate::tensor::quant_cpu::BlockFormat;
use crate::tensor::{Error, Result};

/// The rows a projection saw, and how strongly to hold the fit back where they say little.
pub struct Calibration<'a> {
    /// `n` rows of `cols` values, row-major: the inputs of this projection over a calibration text.
    pub rows: &'a [f32],
    pub n: usize,
    /// The ridge on the second moment, as a multiple of its mean diagonal. Rows are far fewer than
    /// columns, so without it the correction follows directions the calibration never exercised.
    pub damping: f32,
}

/// In-place Cholesky of a symmetric positive matrix, lower triangle. False when it is not positive.
/// Each column's entries below the diagonal depend only on the columns before it, so they are
/// filled across the rayon pool, a row at a time.
fn cholesky(a: &mut [f32], n: usize) -> bool {
    use rayon::prelude::*;
    for j in 0..n {
        let (done, rest) = a.split_at_mut((j + 1) * n);
        let row_j = &mut done[j * n..];
        let d = row_j[j] - row_j[..j].iter().map(|v| v * v).sum::<f32>();
        if d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        row_j[j] = d;
        let row_j = &done[j * n..j * n + j];
        // At most one task per core per column: a small matrix costs less than the tasks would.
        let per_task = ((n - j - 1) / rayon::current_num_threads()).max(1);
        rest.par_chunks_mut(n)
            .with_min_len(per_task)
            .for_each(|row_i| {
                let s = row_i[j]
                    - row_i[..j]
                        .iter()
                        .zip(row_j)
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                row_i[j] = s / d;
            });
    }
    true
}

/// Solve `L L^T z = b` for every column of `b` (`b` is `n` by `width`, row-major), in place. The
/// columns are independent systems, solved across the rayon pool, each in the order a single
/// column would be.
fn cholesky_solve(l: &[f32], n: usize, b: &mut [f32], width: usize) {
    use rayon::prelude::*;
    let mut columns = vec![0f32; n * width];
    for i in 0..n {
        for c in 0..width {
            columns[c * n + i] = b[i * width + c];
        }
    }
    let per_task = (width / rayon::current_num_threads()).max(1);
    columns
        .par_chunks_mut(n)
        .with_min_len(per_task)
        .for_each(|z| {
            for i in 0..n {
                let row = &l[i * n..i * n + i];
                let s = z[i] - row.iter().zip(&z[..i]).map(|(f, v)| f * v).sum::<f32>();
                z[i] = s / l[i * n + i];
            }
            for i in (0..n).rev() {
                z[i] /= l[i * n + i];
                let v = z[i];
                for k in 0..i {
                    z[k] -= l[i * n + k] * v;
                }
            }
        });
    for i in 0..n {
        for c in 0..width {
            b[i * width + c] = columns[c * n + i];
        }
    }
}

/// The correction's products on the CPU, in the form `quantize_factored_with` takes.
pub fn cpu_product(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    matmul(a, b, out, m, k, n);
    Ok(())
}

/// `a` (rows by inner) times `b` (inner by cols), into `out` (rows by cols).
fn matmul(a: &[f32], b: &[f32], out: &mut [f32], rows: usize, inner: usize, cols: usize) {
    use rayon::prelude::*;
    debug_assert_eq!(
        (a.len(), b.len(), out.len()),
        (rows * inner, inner * cols, rows * cols)
    );
    out.par_chunks_mut(cols).enumerate().for_each(|(i, row)| {
        row.fill(0.0);
        for k in 0..inner {
            let f = a[i * inner + k];
            if f == 0.0 {
                continue;
            }
            let brow = &b[k * cols..k * cols + cols];
            for (o, &v) in row.iter_mut().zip(brow) {
                *o += f * v;
            }
        }
    });
}

/// Quantise `values` (`rows` by `cols`) into `out`, rows taken in chunks across the rayon pool.
/// Every format's blocks tile a row, so chunks of whole rows are independent, and the importance a
/// row's blocks take is the same whichever chunk it lands in.
pub fn quantize_rows<T: BlockFormat>(
    values: &[f32],
    cols: usize,
    importance: Option<&[f32]>,
    out: &mut [T],
) {
    use rayon::prelude::*;
    const ROWS_PER_TASK: usize = 32;
    let per_row = cols / T::BLOCK_LEN;
    out.par_chunks_mut(ROWS_PER_TASK * per_row)
        .zip(values.par_chunks(ROWS_PER_TASK * cols))
        .for_each(|(blocks, chunk)| match importance {
            Some(imp) => T::quantize_guided(chunk, blocks, imp, cols),
            None => T::quantize(chunk, blocks),
        });
}

/// Quantise `w` (`rows` by `cols`, row-major) block of columns by block of columns, each block's
/// error carried onto the columns after it. `importance` is one weight per column, as the block
/// quantiser takes it.
pub fn quantize_compensated<T: BlockFormat>(
    w: &[f32],
    rows: usize,
    cols: usize,
    cal: &Calibration,
    importance: Option<&[f32]>,
    out: &mut [T],
) -> Result<()> {
    let on_cpu = |values: &[f32], width: usize, imp: Option<&[f32]>, blocks: &mut [T]| {
        quantize_rows(values, width, imp, blocks);
        Ok(())
    };
    quantize_compensated_with(w, rows, cols, cal, importance, out, &on_cpu)
}

/// The inverse of a block's corner of the inverse moment, `(I - X_B^T Y_B) / lambda`, from
/// `corner = X_B^T Y_B` (`width` by `width`, row-major); `None` when it is not positive.
pub fn block_inverse(corner: &[f32], lambda: f32) -> Option<Vec<f32>> {
    let width = (corner.len() as f64).sqrt() as usize;
    if width * width != corner.len() {
        return None;
    }
    let mut h: Vec<f32> = corner.iter().map(|v| -v / lambda).collect();
    for i in 0..width {
        h[i * width + i] += 1.0 / lambda;
    }
    if !cholesky(&mut h, width) {
        return None;
    }
    let mut inverse = vec![0f32; width * width];
    for i in 0..width {
        inverse[i * width + i] = 1.0;
    }
    cholesky_solve(&h, width, &mut inverse, width);
    Some(inverse)
}

/// The part of the correction that depends on the calibration rows alone: the ridge, and the rows
/// solved against their own moment. Projections that share a record share it, so it is computed
/// once for all of them.
pub struct Factor {
    n: usize,
    cols: usize,
    lambda: f32,
    /// `m^-1 X`, `n` by `cols`, with `m = lambda I + X X^T`.
    y: Vec<f32>,
}

impl Factor {
    /// The ridge the moment was taken with.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// The calibration rows solved against their moment, `n` by `cols` row-major.
    pub fn solved(&self) -> &[f32] {
        &self.y
    }

    pub fn new(cal: &Calibration, cols: usize) -> Result<Self> {
        use rayon::prelude::*;
        let n = cal.n;
        if n == 0 || cols == 0 || cal.rows.len() != n * cols {
            return Err(Error(format!(
                "compensate: {} calibration values are not {n} rows of {cols}",
                cal.rows.len()
            )));
        }
        let x = cal.rows;
        // The ridge, from the mean diagonal of the second moment - computed from the rows, not from
        // the matrix they would form.
        let diag: f64 = x.par_iter().map(|&v| (v as f64) * (v as f64)).sum();
        let lambda = (cal.damping as f64 * diag / cols as f64).max(1e-12) as f32;

        // m = lambda I + X X^T, its lower triangle filled a row at a time across the pool.
        let mut m = vec![0f32; n * n];
        m.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let xi = &x[i * cols..(i + 1) * cols];
            for (j, slot) in row[..=i].iter_mut().enumerate() {
                *slot = xi
                    .iter()
                    .zip(&x[j * cols..(j + 1) * cols])
                    .map(|(a, b)| a * b)
                    .sum();
            }
            row[i] += lambda;
        });
        for i in 0..n {
            for j in 0..i {
                m[j * n + i] = m[i * n + j];
            }
        }
        if !cholesky(&mut m, n) {
            return Err(Error(
                "compensate: the calibration rows give no positive moment".into(),
            ));
        }
        let mut y = x.to_vec();
        cholesky_solve(&m, n, &mut y, cols);
        Ok(Self { n, cols, lambda, y })
    }
}

/// A block quantiser `quantize_compensated_with` hands each block of columns to: `values` are
/// `rows` by `width`, `importance` the block's column weights, and the blocks are written in place.
pub type BlockQuantiser<'a, T> = dyn Fn(&[f32], usize, Option<&[f32]>, &mut [T]) -> Result<()> + 'a;

/// A matrix product the correction hands its two large products to: `a` (`rows` by `inner`) times
/// `b` (`inner` by `cols`), row-major, into `out`.
pub type MatMul<'a> = dyn Fn(&[f32], &[f32], &mut [f32], (usize, usize, usize)) -> Result<()> + 'a;

/// `quantize_compensated`, with each block of columns quantised by `quantise` - an accelerator's
/// kernel, say. The correction around it does not change.
pub fn quantize_compensated_with<T: BlockFormat>(
    w: &[f32],
    rows: usize,
    cols: usize,
    cal: &Calibration,
    importance: Option<&[f32]>,
    out: &mut [T],
    quantise: &BlockQuantiser<T>,
) -> Result<()> {
    let factor = Factor::new(cal, cols)?;
    quantize_factored_with(
        w,
        rows,
        cols,
        cal,
        &factor,
        importance,
        out,
        quantise,
        &cpu_product,
    )
}

/// `quantize_compensated_with` against a `Factor` already computed from `cal`, its two large
/// products per block through `product`.
#[allow(clippy::too_many_arguments)]
pub fn quantize_factored_with<T: BlockFormat>(
    w: &[f32],
    rows: usize,
    cols: usize,
    cal: &Calibration,
    factor: &Factor,
    importance: Option<&[f32]>,
    out: &mut [T],
    quantise: &BlockQuantiser<T>,
    product: &MatMul,
) -> Result<()> {
    let width = T::BLOCK_LEN;
    if cols % width != 0 || w.len() != rows * cols {
        return Err(Error(format!(
            "compensate: {rows} by {cols} does not tile into blocks of {width}"
        )));
    }
    if factor.cols != cols || factor.n != cal.n || cal.rows.len() != cal.n * cols {
        return Err(Error(format!(
            "compensate: a factor of {} rows of {} does not fit {} rows of {cols}",
            factor.n, factor.cols, cal.n
        )));
    }
    let (n, x, y, lambda) = (cal.n, cal.rows, &factor.y, factor.lambda);

    let per_row = cols / width;
    // The correction every earlier block owes the columns still to come, in the space the rows
    // span: `carried` times the solved rows gives it for any block. Applying it lazily, a block at
    // a time, is exact - the correction is linear - and costs a block's width instead of the tail.
    let mut carried = vec![0f32; rows * n];
    let mut block = vec![0f32; rows * width];
    let mut quantised = vec![T::zeros(); rows];
    let mut got = vec![0f32; rows * width];
    let mut y_block = vec![0f32; n * width];
    let mut x_block = vec![0f32; n * width];

    for a in (0..cols).step_by(width) {
        let b = a + width;
        for j in 0..n {
            y_block[j * width..(j + 1) * width].copy_from_slice(&y[j * cols + a..j * cols + b]);
            x_block[j * width..(j + 1) * width].copy_from_slice(&x[j * cols + a..j * cols + b]);
        }
        product(&carried, &y_block, &mut block, (rows, n, width))?;
        for r in 0..rows {
            for (c, v) in block[r * width..(r + 1) * width].iter_mut().enumerate() {
                *v += w[r * cols + a + c];
            }
        }
        quantise(
            &block,
            width,
            importance.map(|imp| &imp[a..b]),
            &mut quantised,
        )?;
        T::dequantize(&quantised, &mut got);
        for (r, q) in quantised.iter().enumerate() {
            out[r * per_row + a / width] = q.clone();
        }
        if b == cols {
            break;
        }

        // The block's own corner of the inverse moment, through the small matrix the rows span:
        // (I - X_B^T Y_B) / lambda.
        let mut corner = vec![0f32; width * width];
        let mut x_t = vec![0f32; width * n];
        for k in 0..n {
            for i in 0..width {
                x_t[i * n + k] = x_block[k * width + i];
            }
        }
        product(&x_t, &y_block, &mut corner, (width, n, width))?;
        let Some(inverse) = block_inverse(&corner, lambda) else {
            continue;
        };
        let mut m_small = vec![0f32; width * n];
        product(&inverse, &x_t, &mut m_small, (width, width, n))?;
        // What this block lost, carried through m into the rows' space: k = e m / lambda.
        let mut e = vec![0f32; rows * width];
        for r in 0..rows {
            for c in 0..width {
                e[r * width + c] = (block[r * width + c] - got[r * width + c]) / lambda;
            }
        }
        let mut k = vec![0f32; rows * n];
        product(&e, &m_small, &mut k, (rows, width, n))?;
        for (acc, v) in carried.iter_mut().zip(&k) {
            *acc += v;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu::BlockIq2Xxs;

    /// Held on a card from end to end, the correction leaves the output error the CPU leaves, for
    /// both formats it carries, on rows fewer and more numerous than a block is wide.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_card_carries_the_error_as_the_cpu_does() {
        use crate::tensor::cuda::{card_factor, quantize_compensated_on_card, CudaDevice};
        use crate::tensor::quant_cpu::{to_float_bytes, BlockQ2K};
        use crate::tensor::quantized::GgmlDType;
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card correction is NOT covered by this run");
            return;
        };
        let stream = |seed: u64, len: usize, scale: f32| -> Vec<f32> {
            let mut v = seed | 1;
            (0..len)
                .map(|_| {
                    v ^= v << 13;
                    v ^= v >> 7;
                    v ^= v << 17;
                    ((v >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * scale
                })
                .collect()
        };
        let (rows, cols) = (40usize, 1024usize);
        for n in [24usize, 300] {
            let w = stream(3, rows * cols, 0.02);
            let x = stream(5 + n as u64, n * cols, 0.5);
            let importance: Vec<f32> = (0..cols)
                .map(|c| (0..n).map(|r| x[r * cols + c].powi(2)).sum::<f32>() / n as f32)
                .collect();
            let output_error = |got: &[f32]| -> f64 {
                let (mut num, mut den) = (0f64, 0f64);
                for i in 0..n {
                    for r in 0..rows {
                        let (mut a, mut b) = (0f64, 0f64);
                        for c in 0..cols {
                            a += (x[i * cols + c] * got[r * cols + c]) as f64;
                            b += (x[i * cols + c] * w[r * cols + c]) as f64;
                        }
                        num += (a - b).powi(2);
                        den += b * b;
                    }
                }
                (num / den).sqrt()
            };
            let cal = Calibration {
                rows: &x,
                n,
                damping: 1.0,
            };
            let factor = Factor::new(&cal, cols).unwrap();
            let on_card =
                card_factor(&dev, &x, factor.solved(), (n, cols), factor.lambda()).unwrap();
            for dtype in [GgmlDType::Iq2Xxs, GgmlDType::Q2K] {
                let decode = |bytes: &[u8]| {
                    let mut got = vec![0f32; rows * cols];
                    to_float_bytes(dtype, bytes, &mut got).unwrap();
                    got
                };
                let cpu: Vec<u8> = if dtype == GgmlDType::Q2K {
                    let mut blocks = vec![BlockQ2K::zeros(); rows * cols / 256];
                    quantize_compensated(&w, rows, cols, &cal, Some(&importance), &mut blocks)
                        .unwrap();
                    unsafe {
                        std::slice::from_raw_parts(
                            blocks.as_ptr() as *const u8,
                            std::mem::size_of_val(&blocks[..]),
                        )
                    }
                    .to_vec()
                } else {
                    let mut blocks = vec![BlockIq2Xxs::zeros(); rows * cols / 256];
                    quantize_compensated(&w, rows, cols, &cal, Some(&importance), &mut blocks)
                        .unwrap();
                    unsafe {
                        std::slice::from_raw_parts(
                            blocks.as_ptr() as *const u8,
                            std::mem::size_of_val(&blocks[..]),
                        )
                    }
                    .to_vec()
                };
                let card = quantize_compensated_on_card(
                    &dev,
                    dtype,
                    &w,
                    (rows, cols),
                    &importance,
                    &on_card,
                    &block_inverse,
                )
                .unwrap();
                let (e_cpu, e_card) = (output_error(&decode(&cpu)), output_error(&decode(&card)));
                assert!(
                    e_card <= e_cpu * 1.02 + 1e-6,
                    "{dtype:?}, {n} rows: card {e_card} against cpu {e_cpu}"
                );
            }
        }
    }
}
