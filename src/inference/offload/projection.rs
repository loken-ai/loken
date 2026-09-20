//! A weight in whatever storage the file gives it, applied in place.
//!
//! A projection is `[out, in]`: dense f32, GGUF blocks the CPU dot engine reads in place, or
//! block-scaled fp8/fp4 and bf16 read in place from the mapping. Its product goes to the
//! thread's offload when one takes it, and runs here otherwise. A mapped weight is never
//! dequantised or copied: its bytes are the working set.

use crate::tensor::blockscaled::bf16::Bf16Weight;
use crate::tensor::blockscaled::fp4::Fp4Weight;
use crate::tensor::blockscaled::fp8::Fp8Weight;
use crate::tensor::quantized::QTensor;
use crate::tensor::{Device, Result, Tensor};
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
            // through the mapping: clean file pages leave the page cache. Left in place, a
            // prompt's sweep over every expert evicted the kept ones by plain recency, and the
            // tier kept names of experts whose bytes were gone.
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

    /// `x` [n, in] -> [n, out].
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(offload) = super::current() {
            let xs = x.flatten_all()?.to_vec1::<f32>()?;
            if let Some(y) = offload.projection(self, &xs) {
                let mut dims = x.dims().to_vec();
                if let Some(last) = dims.last_mut() {
                    *last = self.dims()[0];
                }
                return Tensor::from_vec(y?, dims, &Device::Cpu);
            }
        }
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
}
