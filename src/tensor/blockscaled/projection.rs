//! A weight in whatever storage the file gives it, applied in place.
//!
//! A projection is `[out, in]`: dense f32, GGUF blocks the CPU dot engine reads in place, or
//! block-scaled fp8/fp4 and bf16 read in place from the mapping. Its product runs on the host
//! cores. A mapped weight is never dequantised or copied: its bytes are the working set.

use super::bf16::Bf16Weight;
use super::fp4::Fp4Weight;
use super::fp8::Fp8Weight;
use crate::tensor::quantized::QTensor;
use crate::tensor::{DType, Device, Result, Tensor};
use std::sync::Arc;

/// One projection of an expert, `[out, in]`: dense f32, GGUF blocks the CPU dot engine reads in
/// place, or the released checkpoint's block-scaled fp4 read in place - a mapped expert is never
/// dequantised, its bytes are the working set.
#[derive(Clone)]
pub enum Projection {
    Dense(Tensor),
    Quant(Arc<QTensor>),
    Fp4(Arc<Fp4Weight>),
    Fp8(Arc<Fp8Weight>),
    Bf16(Arc<Bf16Weight>),
}

impl Projection {
    /// `[out, in]`.
    pub fn dims(&self) -> Vec<usize> {
        match self {
            Self::Dense(w) => w.dims().to_vec(),
            Self::Quant(q) => q.shape().dims().to_vec(),
            Self::Fp4(f) => vec![f.out, f.inp],
            Self::Fp8(f) => vec![f.out, f.inp],
            Self::Bf16(f) => vec![f.out, f.inp],
        }
    }

    /// The GGUF quant format this projection is stored in, when it has one. `None` for a dense or
    /// block-scaled (fp4/fp8/bf16) weight, whose cost the miss bench does not model.
    pub fn quant_dtype(&self) -> Option<crate::tensor::quantized::GgmlDType> {
        match self {
            Self::Quant(q) => Some(q.dtype()),
            _ => None,
        }
    }

    /// Rows `[start, start + count)` as a projection of their own, in place where the storage
    /// allows: a grouped projection's group.
    pub fn rows(&self, start: usize, count: usize) -> Result<Self> {
        Ok(match self {
            Self::Dense(w) => Self::Dense(w.narrow(0, start, count)?.contiguous()?),
            Self::Quant(q) => Self::Quant(Arc::new(crate::tensor::quant_view::rows_view(
                q, start, count,
            )?)),
            Self::Fp8(f) => Self::Fp8(Arc::new(f.rows(start, count)?)),
            Self::Fp4(_) | Self::Bf16(_) => {
                return Err(crate::tensor::Error::msg(
                    "projection: no row range view for this storage",
                ));
            }
        })
    }

    /// The weight decoded to `[out, in]` f32, row-major, on the host.
    pub fn dense_f32(&self) -> Result<Vec<f32>> {
        Ok(match self {
            Self::Dense(w) => w
                .to_dtype(DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?,
            // CONVERSION: the caller asks for the decoded values.
            Self::Quant(q) => q
                .dequantize(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            Self::Fp8(f) => f.dense_f32(),
            Self::Fp4(f) => f.dense_f32(),
            Self::Bf16(f) => f.dense_f32(),
        })
    }

    /// The same weight as dense F32 values: what a product with no weight quantisation reads.
    #[cfg(all(test, feature = "gpu"))]
    pub fn dense_form(&self) -> Result<Self> {
        let dims = self.dims();
        Ok(Self::Dense(Tensor::from_vec(
            self.dense_f32()?,
            (dims[0], dims[1]),
            &Device::Cpu,
        )?))
    }

    /// The bytes behind the projection, as stored.
    pub fn bytes(&self) -> usize {
        match self {
            Self::Dense(w) => w.elem_count() * w.dtype().size_in_bytes(),
            Self::Quant(q) => q.elem_count() / q.dtype().block_size() * q.dtype().type_size(),
            Self::Fp4(f) => f.nibbles.len() + f.scales.len(),
            Self::Fp8(f) => f.bytes.len() + f.scales.len(),
            Self::Bf16(f) => f.bytes.len(),
        }
    }

    /// Drop the storage from the page cache, where the storage is mapped: for a projection
    /// that will not be read again soon.
    pub fn dont_need(&self) {
        match self {
            Self::Fp4(f) => {
                f.nibbles.dont_need();
                f.scales.dont_need();
            }
            Self::Fp8(f) => {
                f.bytes.dont_need();
                f.scales.dont_need();
            }
            Self::Bf16(f) => f.bytes.dont_need(),
            // A quantised view knows its bytes but not its file, so the pages are reclaimed
            // through the mapping: clean file pages leave the page cache.
            #[cfg(target_os = "linux")]
            Self::Quant(q) => {
                if let Ok(bytes) = q.data() {
                    crate::tensor::mapped::advise_bytes(&bytes, libc::MADV_PAGEOUT, true);
                }
            }
            #[cfg(not(target_os = "linux"))]
            Self::Quant(_) => {}
            Self::Dense(_) => {}
        }
    }

    /// Drop this process's mapping entries for the storage of a mapped quantised projection,
    /// without any claim on the page cache: the step before advice to the file itself.
    pub fn unmap_pages(&self) {
        #[cfg(unix)]
        if let Self::Quant(q) = self {
            if let Ok(bytes) = q.data() {
                crate::tensor::mapped::advise_bytes(&bytes, libc::MADV_DONTNEED, true);
            }
        }
    }

    /// The address and length of the storage, for a mapped quantised projection.
    pub fn storage_span(&self) -> Option<(usize, usize)> {
        match self {
            Self::Quant(q) => q.data().ok().map(|b| (b.as_ptr() as usize, b.len())),
            _ => None,
        }
    }

    /// The address and length of every byte range the storage reads in place.
    ///
    /// Empty for a dense weight, which holds its own values.
    pub fn mapped_spans(&self) -> Vec<(usize, usize)> {
        let of = |b: &crate::tensor::mapped::MappedBytes| (b.as_slice().as_ptr() as usize, b.len());
        match self {
            Self::Quant(_) => self.storage_span().into_iter().collect(),
            Self::Fp4(f) => vec![of(&f.nibbles), of(&f.scales)],
            Self::Fp8(f) => vec![of(&f.bytes), of(&f.scales)],
            Self::Bf16(f) => vec![of(&f.bytes)],
            Self::Dense(_) => Vec::new(),
        }
    }

    /// How many of the storage's pages are resident in memory, and how many pages there are:
    /// the fact behind a slow token, read from the kernel rather than inferred.
    #[cfg(target_os = "linux")]
    pub fn resident_pages(&self) -> (usize, usize) {
        let bytes = match self {
            Self::Quant(q) => match q.data() {
                Ok(b) => b,
                Err(_) => return (0, 0),
            },
            _ => return (0, 0),
        };
        // Safety: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as usize;
        let start = bytes.as_ptr() as usize / page * page;
        let len = bytes.as_ptr() as usize + bytes.len() - start;
        let pages = len.div_ceil(page);
        let mut vec = vec![0u8; pages];
        // Safety: [start, start + len) lies within the mapped pages `bytes` occupies, and the
        // vector holds one byte per page as mincore requires.
        let rc = unsafe { libc::mincore(start as *mut libc::c_void, len, vec.as_mut_ptr()) };
        if rc != 0 {
            return (0, pages);
        }
        (vec.iter().filter(|&&v| v & 1 == 1).count(), pages)
    }

    /// Start reading the storage in ahead of `apply`, where the storage is mapped.
    pub fn will_need(&self) {
        match self {
            Self::Fp4(f) => {
                f.nibbles.will_need();
                f.scales.will_need();
            }
            Self::Fp8(f) => {
                f.bytes.will_need();
                f.scales.will_need();
            }
            Self::Bf16(f) => f.bytes.will_need(),
            #[cfg(unix)]
            Self::Quant(q) => {
                if let Ok(bytes) = q.data() {
                    crate::tensor::mapped::advise_bytes(&bytes, libc::MADV_WILLNEED, false);
                }
            }
            #[cfg(not(unix))]
            Self::Quant(_) => {}
            Self::Dense(_) => {}
        }
    }

    /// `x` [n, in] -> [n, out] on this thread's cores.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        self.apply_here(x)
    }

    /// `x` [n, in] -> [n, out] on this thread's cores.
    pub fn apply_here(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            // A single row is a matrix-vector product: each weight row against the input, across
            // the cores, with nothing transposed or copied.
            Self::Dense(w) if x.elem_count() == w.dim(1)? => {
                use rayon::prelude::*;
                let (out, inp) = w.dims2()?;
                let xs = x.flatten_all()?.to_vec1::<f32>()?;
                let contiguous;
                let wv: &[f32] = match w.cpu_f32_data() {
                    Ok(v) => v,
                    Err(_) => {
                        contiguous = w.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                        &contiguous
                    }
                };
                let y: Vec<f32> = (0..out)
                    .into_par_iter()
                    .map(|o| {
                        wv[o * inp..(o + 1) * inp]
                            .iter()
                            .zip(&xs)
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();
                let mut dims = x.dims().to_vec();
                *dims.last_mut().unwrap() = out;
                Tensor::from_vec(y, dims, &Device::Cpu)
            }
            Self::Dense(w) => x.matmul(&w.t()?),
            Self::Quant(q) => q.native_qmm()?.forward(x),
            Self::Fp4(f) => f.apply(x),
            Self::Fp8(f) => f.apply(x),
            Self::Bf16(f) => f.apply(x),
        }
    }

    /// The `n` rows `x` `[n, in]` row-major, each as [`Self::apply_here`] takes a single row:
    /// `[n, out]` row-major, on this thread's cores.
    ///
    /// A dense output is the fold from -0, in input order, of the rounded products.
    /// Other storages apply their product to the rows at once.
    pub fn apply_rows_alone(&self, x: &[f32], n: usize) -> Result<Vec<f32>> {
        let inp = self.dims().get(1).copied().unwrap_or(0);
        if x.len() != n * inp {
            return Err(crate::tensor::Error::msg(format!(
                "{} values for {n} rows of {inp}",
                x.len()
            )));
        }
        match self {
            Self::Dense(w) => {
                let out = w.dim(0)?;
                let contiguous;
                let wv: &[f32] = match w.cpu_f32_data() {
                    Ok(v) => v,
                    Err(_) => {
                        contiguous = w.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                        &contiguous
                    }
                };
                Ok(fold_rows(wv, out, inp, x, n))
            }
            _ => {
                let xt = Tensor::from_vec(x.to_vec(), (n, inp), &Device::Cpu)?;
                self.apply_here(&xt)?.flatten_all()?.to_vec1::<f32>()
            }
        }
    }
}

/// Rows a task of [`fold_rows`] takes together, a lane each.
const FOLD_LANES: usize = 16;

/// `x` `[n, inp]` through `w` `[out, inp]`, both row-major: `[n, out]` row-major.
///
/// Each output is the fold from -0, in input order, of the rounded products of its weight row
/// and its row.
/// A task takes [`FOLD_LANES`] rows, input-major, so each lane's sum is its own row's fold.
fn fold_rows(w: &[f32], out: usize, inp: usize, x: &[f32], n: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut y = vec![0f32; n * out];
    if out == 0 {
        return y;
    }
    y.par_chunks_mut(FOLD_LANES * out)
        .enumerate()
        .for_each(|(b, yb)| {
            let r0 = b * FOLD_LANES;
            let lanes = yb.len() / out;
            // Lane `j` of input `k` at `k * FOLD_LANES + j`, the missing lanes zero.
            let mut xt = vec![0f32; inp * FOLD_LANES];
            for j in 0..lanes {
                let row = &x[(r0 + j) * inp..(r0 + j + 1) * inp];
                for (k, &v) in row.iter().enumerate() {
                    xt[k * FOLD_LANES + j] = v;
                }
            }
            let sums: Vec<[f32; FOLD_LANES]> = (0..out)
                .into_par_iter()
                .map(|o| {
                    let mut acc = [-0.0f32; FOLD_LANES];
                    let wr = &w[o * inp..(o + 1) * inp];
                    for (xk, &wk) in xt.chunks_exact(FOLD_LANES).zip(wr) {
                        for (a, &v) in acc.iter_mut().zip(xk) {
                            *a += wk * v;
                        }
                    }
                    acc
                })
                .collect();
            for (o, acc) in sums.iter().enumerate() {
                for (j, &v) in acc.iter().take(lanes).enumerate() {
                    yb[j * out + o] = v;
                }
            }
        });
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows applied together give, bit for bit, each row applied alone.
    #[test]
    fn rows_applied_together_give_each_row_alone() {
        for (out, inp, n) in [(37usize, 1000usize, 1usize), (384, 513, 35), (8, 31, 16)] {
            let values: Vec<f32> = (0..out * inp)
                .map(|i| ((i as f32 * 0.731).sin() * (1.0 + (i % 17) as f32)) * 0.01)
                .collect();
            let w = Projection::Dense(Tensor::from_vec(values, (out, inp), &Device::Cpu).unwrap());
            let x: Vec<f32> = (0..n * inp)
                .map(|i| ((i * 13 + 7) as f32 * 0.29).cos() * (1 + i % 5) as f32)
                .collect();
            let together = w.apply_rows_alone(&x, n).unwrap();
            for r in 0..n {
                let row =
                    Tensor::from_vec(x[r * inp..(r + 1) * inp].to_vec(), (1, inp), &Device::Cpu)
                        .unwrap();
                let alone = w.apply_here(&row).unwrap().flatten_all().unwrap();
                let alone: Vec<u32> = alone
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let got: Vec<u32> = together[r * out..(r + 1) * out]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(got, alone, "[{out}, {inp}] row {r} of {n}");
            }
        }
    }
}
