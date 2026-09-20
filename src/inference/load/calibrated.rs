//! Quantise one tensor with what a calibration run recorded for it.
//!
//! A two-dimensional weight is one projection. A three-dimensional one is a stack of them - the
//! routed experts of a layer - and each gets its own record when the run saw it, the whole stack's
//! otherwise. What a record allows decides the path: enough sampled rows buy the error correction
//! across columns, a record without them still weighs the fit by importance, and no record falls
//! back on the plain quantiser. The output is the format's own blocks either way.

use super::calibration::Seen;
use super::compensate::{
    cpu_product, quantize_factored_with, quantize_rows, Calibration, Factor, MatMul,
};
use crate::tensor::quant_cpu::{
    BlockFormat, BlockIq2Xxs, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K,
};
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::collections::HashMap;

/// How much a record has to hold before a path is taken.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    /// Sampled rows a projection needs before its error is carried across columns. Below it the
    /// second moment says too little about how its columns move together.
    pub min_rows: usize,
    /// The ridge the correction uses, as a multiple of the moment's mean diagonal.
    pub damping: f32,
}

/// Records by projection name, as `calibration::read` returns them: importance, rows, rows kept.
pub type Records = HashMap<String, (Vec<f32>, Vec<f32>, usize)>;

/// The formats that take an importance: every one this dispatches to.
pub fn guided(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Iq2Xxs
    )
}

fn bytes_of<T>(blocks: &[T]) -> Vec<u8> {
    // Safety: every block format is `repr(C)` plain data, which is what the file stores.
    unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, std::mem::size_of_val(blocks))
    }
    .to_vec()
}

/// What an accelerator lends a projection's quantisation: the correction's large products, and the
/// encoding of the formats it has a kernel for.
pub struct Kernels<'a> {
    pub product: &'a MatMul<'a>,
    /// Blocks of `dtype` for `values` (`width` per row) weighed by `importance`, as the format's
    /// bytes; `None` for a format it has no kernel for.
    pub encode: &'a Encode<'a>,
}

pub type Encode<'a> =
    dyn Fn(GgmlDType, &[f32], usize, Option<&[f32]>) -> Option<Result<Vec<u8>>> + 'a;

/// Quantise one projection (`rows` by `cols`) to `dtype` with `record`. `compensate` false keeps a
/// record's rows out of the fit, for a record pooled from other projections, whose rows describe
/// what those saw rather than this one.
pub fn quantize_projection(
    dtype: GgmlDType,
    w: &[f32],
    rows: usize,
    cols: usize,
    record: Option<&(Vec<f32>, Vec<f32>, usize)>,
    compensate: bool,
    recipe: Recipe,
) -> Result<Vec<u8>> {
    quantize_projection_on(
        dtype,
        w,
        (rows, cols),
        record,
        compensate,
        recipe,
        None,
        None,
    )
}

/// `quantize_projection` with `kernels` when an accelerator lends them, and `factor` when the
/// record's has already been computed - projections that share a record share it.
#[allow(clippy::too_many_arguments)]
pub fn quantize_projection_on(
    dtype: GgmlDType,
    w: &[f32],
    (rows, cols): (usize, usize),
    record: Option<&(Vec<f32>, Vec<f32>, usize)>,
    compensate: bool,
    recipe: Recipe,
    factor: Option<&Factor>,
    kernels: Option<&Kernels>,
) -> Result<Vec<u8>> {
    let importance_only;
    let record = match record {
        Some((imp, _, _)) if !compensate => {
            importance_only = (imp.clone(), Vec::new(), 0);
            Some(&importance_only)
        }
        other => other,
    };
    let at = Placement {
        rows,
        cols,
        recipe,
        factor,
        kernels,
    };
    match dtype {
        GgmlDType::Q2K => one::<BlockQ2K>(w, record, &at),
        GgmlDType::Q3K => one::<BlockQ3K>(w, record, &at),
        GgmlDType::Q4K => one::<BlockQ4K>(w, record, &at),
        GgmlDType::Q5K => one::<BlockQ5K>(w, record, &at),
        GgmlDType::Q6K => one::<BlockQ6K>(w, record, &at),
        GgmlDType::Iq2Xxs => one::<BlockIq2Xxs>(w, record, &at),
        other => Err(Error(format!(
            "{other:?} takes no importance; quantise it plainly"
        ))),
    }
}

struct Placement<'a> {
    rows: usize,
    cols: usize,
    recipe: Recipe,
    factor: Option<&'a Factor>,
    kernels: Option<&'a Kernels<'a>>,
}

/// One projection: `w` is `at.rows` by `at.cols`.
fn one<T: BlockFormat>(
    w: &[f32],
    record: Option<&(Vec<f32>, Vec<f32>, usize)>,
    at: &Placement,
) -> Result<Vec<u8>> {
    let (rows, cols) = (at.rows, at.cols);
    let mut blocks = vec![T::zeros(); w.len() / T::BLOCK_LEN];
    let encode =
        |values: &[f32], width: usize, imp: Option<&[f32]>, blocks: &mut [T]| -> Result<()> {
            if let Some(bytes) = at
                .kernels
                .and_then(|k| (k.encode)(T::DTYPE, values, width, imp))
            {
                let bytes = bytes?;
                if bytes.len() != std::mem::size_of_val(blocks) {
                    return Err(Error(format!(
                        "{:?}: the kernel wrote a different block count",
                        T::DTYPE
                    )));
                }
                // Safety: the kernel writes whole blocks of the format, whose layout `T` is.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        blocks.as_mut_ptr() as *mut u8,
                        bytes.len(),
                    )
                };
            } else {
                quantize_rows(values, width, imp, blocks);
            }
            Ok(())
        };
    match record {
        Some((importance, sample, kept)) if *kept >= at.recipe.min_rows => {
            let cal = Calibration {
                rows: sample,
                n: *kept,
                damping: at.recipe.damping,
            };
            let own;
            let factor = match at.factor {
                Some(f) => f,
                None => {
                    own = Factor::new(&cal, cols)?;
                    &own
                }
            };
            let product: &MatMul = at.kernels.map_or(&cpu_product, |k| k.product);
            quantize_factored_with(
                w,
                rows,
                cols,
                &cal,
                factor,
                Some(importance),
                &mut blocks,
                &encode,
                product,
            )?;
        }
        Some((importance, _, _)) => encode(w, cols, Some(importance), &mut blocks)?,
        None => encode(w, cols, None, &mut blocks)?,
    }
    Ok(bytes_of(&blocks))
}

/// Quantise `values` (a weight of shape `dims`, row-major) to `dtype`, using what `records` hold for
/// `name`. A stack's slices are looked up as `{name}.{index}`, and fall back on `{name}`.
pub fn quantize_tensor(
    name: &str,
    dims: &[usize],
    values: &[f32],
    dtype: GgmlDType,
    records: &Records,
    recipe: Recipe,
) -> Result<Vec<u8>> {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let (stack, rows, cols) = match *dims {
        [rows, cols] => (1, rows, cols),
        [stack, rows, cols] => (stack, rows, cols),
        _ => {
            return Err(Error(format!(
                "{name}: a calibrated tensor is a projection or a stack of them, not {dims:?}"
            )))
        }
    };
    if values.len() != stack * rows * cols {
        return Err(Error(format!(
            "{name}: {} values for {dims:?}",
            values.len()
        )));
    }
    let at = Placement {
        rows,
        cols,
        recipe,
        factor: None,
        kernels: None,
    };
    let mut out = Vec::new();
    for s in 0..stack {
        let slice = &values[s * rows * cols..(s + 1) * rows * cols];
        let record = if dims.len() == 3 {
            records
                .get(&format!("{base}.{s}"))
                .or_else(|| records.get(base))
        } else {
            records.get(base)
        };
        let bytes = match dtype {
            GgmlDType::Q2K => one::<BlockQ2K>(slice, record, &at)?,
            GgmlDType::Q3K => one::<BlockQ3K>(slice, record, &at)?,
            GgmlDType::Q4K => one::<BlockQ4K>(slice, record, &at)?,
            GgmlDType::Q5K => one::<BlockQ5K>(slice, record, &at)?,
            GgmlDType::Q6K => one::<BlockQ6K>(slice, record, &at)?,
            GgmlDType::Iq2Xxs => one::<BlockIq2Xxs>(slice, record, &at)?,
            other => {
                return Err(Error(format!(
                    "{name}: {other:?} takes no importance; quantise it plainly"
                )))
            }
        };
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

/// A record in the form `quantize_tensor` reads, from what a run observed. For callers that hold
/// the observations rather than a file.
pub fn record_of(seen: &Seen) -> (Vec<f32>, Vec<f32>, usize) {
    (seen.importance(), seen.rows.clone(), seen.kept())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu::to_float_bytes;

    /// A deterministic stream, so the test needs no crate and draws the same values every run.
    fn stream(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// The error a reader would see: the bytes read back as blocks, row by row, applied to the
    /// calibration rows and compared with the original matrix - nothing from the path that wrote
    /// them is trusted. A stack whose slices were laid out in the wrong order fails here.
    #[test]
    fn a_stack_reads_back_as_its_own_slices() {
        let (stack, rows, cols, n) = (2usize, 6usize, 512usize, 24usize);
        let w = stream(11, stack * rows * cols, 0.02);
        let mut records = Records::new();
        let mut xs = Vec::new();
        for s in 0..stack {
            let x = stream(100 + s as u64, n * cols, 0.5);
            let mut squares = vec![0f32; cols];
            for r in 0..n {
                for c in 0..cols {
                    squares[c] += x[r * cols + c] * x[r * cols + c] / n as f32;
                }
            }
            records.insert(format!("blk.3.ffn_gate_exps.{s}"), (squares, x.clone(), n));
            xs.push(x);
        }
        let recipe = Recipe {
            min_rows: 8,
            damping: 1.0,
        };
        let bytes = quantize_tensor(
            "blk.3.ffn_gate_exps.weight",
            &[stack, rows, cols],
            &w,
            GgmlDType::Iq2Xxs,
            &records,
            recipe,
        )
        .unwrap();
        let plain = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Iq2Xxs, &w).unwrap();

        let per = bytes.len() / stack;
        let output_error = |got: &[f32], orig: &[f32], x: &[f32]| -> f64 {
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..n {
                for r in 0..rows {
                    let (mut a, mut b) = (0f64, 0f64);
                    for c in 0..cols {
                        a += (x[i * cols + c] * got[r * cols + c]) as f64;
                        b += (x[i * cols + c] * orig[r * cols + c]) as f64;
                    }
                    num += (a - b).powi(2);
                    den += b * b;
                }
            }
            (num / den).sqrt()
        };
        for s in 0..stack {
            let orig = &w[s * rows * cols..(s + 1) * rows * cols];
            let mut ours = vec![0f32; rows * cols];
            to_float_bytes(GgmlDType::Iq2Xxs, &bytes[s * per..(s + 1) * per], &mut ours).unwrap();
            let mut base = vec![0f32; rows * cols];
            to_float_bytes(GgmlDType::Iq2Xxs, &plain[s * per..(s + 1) * per], &mut base).unwrap();
            let (e_ours, e_plain) = (
                output_error(&ours, orig, &xs[s]),
                output_error(&base, orig, &xs[s]),
            );
            assert!(
                e_ours < 0.9,
                "slice {s}: output error {e_ours}, the bytes do not read back as the slice"
            );
            assert!(
                e_ours < e_plain,
                "slice {s}: calibrated {e_ours} against plain {e_plain}"
            );
        }
    }
}
