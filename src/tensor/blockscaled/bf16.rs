//! A bf16 projection read in place: the released checkpoint's head and gate weights.
//!
//! A `[out, in]` bf16 weight is a row-major run of 16-bit values; the product with an
//! activation runs a row at a time straight from the mapping, each value widened to f32 by
//! placing its bits in the high half.

use crate::tensor::mapped::MappedBytes;
use crate::tensor::{Device, Error, Result, Tensor};
use rayon::prelude::*;

/// The dot of `len` bf16 values (as bytes) with `len` f32 activations, `len` a multiple of
/// eight: each value widened by placing its bits in the high half, then fused multiply-adds.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn row_dot_avx2(bytes: *const u8, x: *const f32, len: usize) -> f32 {
    use core::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < len {
        let h = _mm256_cvtepu16_epi32(_mm_loadu_si128(bytes.add(2 * i) as *const __m128i));
        let w = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(h));
        acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x.add(i)), acc);
        i += 8;
    }
    let hi = _mm256_extractf128_ps::<1>(acc);
    let lo = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps::<0b01>(s2, s2));
    _mm_cvtss_f32(s1)
}

#[inline(always)]
fn row_dot(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        if x.len().is_multiple_of(8) {
            // Safety: `row` holds two bytes per activation element.
            return unsafe { row_dot_avx2(row.as_ptr(), x.as_ptr(), x.len()) };
        }
    }
    row.chunks_exact(2)
        .zip(x)
        .map(|(c, &v)| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) * v)
        .sum()
}

/// One `[out, in]` bf16 weight in its storage.
#[derive(Clone)]
pub struct Bf16Weight {
    pub bytes: MappedBytes,
    pub out: usize,
    pub inp: usize,
}

impl Bf16Weight {
    pub fn new(bytes: MappedBytes, out: usize, inp: usize) -> Result<Self> {
        if bytes.len() != out * inp * 2 {
            return Err(Error::msg(format!(
                "bf16 weight: {} bytes for [{out}, {inp}]",
                bytes.len()
            )));
        }
        Ok(Self { bytes, out, inp })
    }

    /// `x` [n, in] f32 -> [n, out] f32.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims
            .last()
            .ok_or_else(|| Error::msg("bf16 apply: rank-0 input"))?;
        if k != self.inp {
            return Err(Error::msg(format!(
                "bf16 apply: input {k} != weight in {}",
                self.inp
            )));
        }
        let n = x.elem_count() / k;
        let xs = x.flatten_all()?.to_vec1::<f32>()?;
        let bytes = self.bytes.as_slice();
        let inp = self.inp;
        let mut y = vec![0f32; n * self.out];
        y.par_chunks_mut(n).enumerate().for_each(|(o, col)| {
            let row = &bytes[o * inp * 2..(o + 1) * inp * 2];
            for (t, out) in col.iter_mut().enumerate() {
                *out = row_dot(row, &xs[t * inp..(t + 1) * inp]);
            }
        });
        let mut yt = vec![0f32; n * self.out];
        for o in 0..self.out {
            for t in 0..n {
                yt[t * self.out + o] = y[o * n + t];
            }
        }
        let mut odims = dims;
        *odims.last_mut().unwrap() = self.out;
        Tensor::from_vec(yt, odims, &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// The in-place product equals the dense matmul over the widened weight.
    #[test]
    fn in_place_product_matches_dense_matmul() {
        let (out, inp, n) = (40usize, 96usize, 3usize);
        let vals: Vec<f32> = (0..out * inp).map(|i| ((i as f32) * 0.11).sin()).collect();
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let widened: Vec<f32> = vals
            .iter()
            .map(|v| f32::from_bits(v.to_bits() & 0xFFFF_0000))
            .collect();
        let mut m = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
        m.copy_from_slice(&bytes);
        let mapped =
            MappedBytes::new(Arc::new(m.make_read_only().unwrap()), 0, bytes.len()).unwrap();
        let w = Tensor::from_vec(widened, (out, inp), &Device::Cpu).unwrap();
        let x = Tensor::from_vec(
            (0..n * inp).map(|i| ((i as f32) * 0.07).cos()).collect(),
            (n, inp),
            &Device::Cpu,
        )
        .unwrap();
        let want = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got = Bf16Weight::new(mapped, out, inp)
            .unwrap()
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{g} vs {w}");
        }
    }
}
