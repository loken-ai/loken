//! A block-scaled fp8 projection read in place: the released checkpoint's attention and shared
//! expert weights.
//!
//! A `[out, in]` weight is stored as e4m3 bytes row-major with one ue8m0 scale per
//! `block x block` tile (`[out / block, in / block]`). The product with an activation runs a row
//! at a time, one tile-column of `block` inputs at a time - bytes decoded through a 256-entry
//! table, the partial sum scaled once per tile - straight from the mapping. A row range of the
//! weight is a view of the same bytes, so a grouped projection reads its group in place too.

use super::dequant::{e4m3_to_f32, e8m0_to_f32};
use crate::tensor::mapped::MappedBytes;
use crate::tensor::{Device, Error, Result, Tensor};
use rayon::prelude::*;

/// The dot of `len` e4m3 bytes with `len` f32 activations, `len` a multiple of eight: eight
/// bytes at a time, each widened to its f32 bits by integer arithmetic - sign to bit 31, the
/// exponent rebiased from 7 to 127, the mantissa shifted up - and, where the exponent is zero,
/// replaced by the subnormal's value (the mantissa times 2^-9); then fused multiply-adds over
/// eight lanes.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn tile_dot_avx2(bytes: *const u8, x: *const f32, len: usize) -> f32 {
    use core::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let sub_scale = _mm256_set1_ps(2f32.powi(-9));
    let mut i = 0;
    while i < len {
        let b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(bytes.add(i) as *const __m128i));
        let sign = _mm256_slli_epi32::<24>(_mm256_and_si256(b, _mm256_set1_epi32(0x80)));
        let exp = _mm256_and_si256(_mm256_srli_epi32::<3>(b), _mm256_set1_epi32(0xF));
        let mant = _mm256_and_si256(b, _mm256_set1_epi32(7));
        let normal = _mm256_or_si256(
            _mm256_or_si256(
                sign,
                _mm256_slli_epi32::<23>(_mm256_add_epi32(exp, _mm256_set1_epi32(120))),
            ),
            _mm256_slli_epi32::<20>(mant),
        );
        let sub = _mm256_or_si256(
            sign,
            _mm256_castps_si256(_mm256_mul_ps(_mm256_cvtepi32_ps(mant), sub_scale)),
        );
        let is_sub = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
        let w = _mm256_castsi256_ps(_mm256_blendv_epi8(normal, sub, is_sub));
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
fn tile_dot(bytes: &[u8], x: &[f32], table: &[f32; 256]) -> f32 {
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        if bytes.len().is_multiple_of(8) {
            // Safety: both slices hold `len` elements, a multiple of eight.
            return unsafe { tile_dot_avx2(bytes.as_ptr(), x.as_ptr(), bytes.len()) };
        }
    }
    bytes
        .iter()
        .zip(x)
        .map(|(&b, &v)| table[b as usize] * v)
        .sum()
}

/// One `[out, in]` fp8 weight in its storage.
#[derive(Clone)]
pub struct Fp8Weight {
    pub bytes: MappedBytes,
    pub scales: MappedBytes,
    pub out: usize,
    pub inp: usize,
    pub block: usize,
    /// First row of `bytes` this view covers, in rows of the whole tensor: the scales are
    /// addressed in the whole tensor's tiles.
    row0: usize,
    /// Rows of the whole tensor, for the scale layout.
    rows_total: usize,
}

impl Fp8Weight {
    /// This view's bytes: its rows of the whole weight.
    pub fn view_bytes(&self) -> &[u8] {
        &self.bytes.as_slice()[self.row0 * self.inp..(self.row0 + self.out) * self.inp]
    }

    /// The first row of the whole weight this view covers.
    pub fn row0(&self) -> usize {
        self.row0
    }

    pub fn new(
        bytes: MappedBytes,
        scales: MappedBytes,
        out: usize,
        inp: usize,
        block: usize,
    ) -> Result<Self> {
        if bytes.len() != out * inp {
            return Err(Error::msg(format!(
                "fp8 weight: {} bytes for [{out}, {inp}]",
                bytes.len()
            )));
        }
        let tiles = out.div_ceil(block) * inp.div_ceil(block);
        if scales.len() != tiles {
            return Err(Error::msg(format!(
                "fp8 weight: {} scales for [{out}, {inp}] in {block}x{block} tiles ({tiles} wanted)",
                scales.len()
            )));
        }
        Ok(Self {
            bytes,
            scales,
            out,
            inp,
            block,
            row0: 0,
            rows_total: out,
        })
    }

    /// Rows `[start, start + count)` as a view, on a tile boundary.
    pub fn rows(&self, start: usize, count: usize) -> Result<Self> {
        if !start.is_multiple_of(self.block) || start + count > self.out {
            return Err(Error::msg(format!(
                "fp8 rows: {start}+{count} of {} must start on a {}-row tile",
                self.out, self.block
            )));
        }
        Ok(Self {
            bytes: self.bytes.clone(),
            scales: self.scales.clone(),
            out: count,
            inp: self.inp,
            block: self.block,
            row0: self.row0 + start,
            rows_total: self.rows_total,
        })
    }

    /// `x` [n, in] f32 -> [n, out] f32.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims
            .last()
            .ok_or_else(|| Error::msg("fp8 apply: rank-0 input"))?;
        if k != self.inp {
            return Err(Error::msg(format!(
                "fp8 apply: input {k} != weight in {}",
                self.inp
            )));
        }
        let n = x.elem_count() / k;
        let xs = x.flatten_all()?.to_vec1::<f32>()?;
        let table: [f32; 256] = std::array::from_fn(|i| e4m3_to_f32(i as u8));
        let bytes = self.bytes.as_slice();
        let sc = self.scales.as_slice();
        let (inp, block) = (self.inp, self.block);
        let tiles_per_row = inp.div_ceil(block);
        let _ = self.rows_total;
        let mut y = vec![0f32; n * self.out];
        y.par_chunks_mut(n).enumerate().for_each(|(o, col)| {
            let row_abs = self.row0 + o;
            let row = &bytes[row_abs * inp..(row_abs + 1) * inp];
            let scales =
                &sc[(row_abs / block) * tiles_per_row..(row_abs / block + 1) * tiles_per_row];
            for (t, out) in col.iter_mut().enumerate() {
                let xt = &xs[t * inp..(t + 1) * inp];
                let mut acc = 0f32;
                for (b, &s) in scales.iter().enumerate() {
                    let lo = b * block;
                    let hi = (lo + block).min(inp);
                    acc += tile_dot(&row[lo..hi], &xt[lo..hi], &table) * e8m0_to_f32(s);
                }
                *out = acc;
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
    use crate::tensor::blockscaled::dequant::dequant_fp8_blocked;
    use std::sync::Arc;

    fn mapped(bytes: &[u8]) -> MappedBytes {
        let mut m = memmap2::MmapMut::map_anon(bytes.len().max(1)).unwrap();
        m[..bytes.len()].copy_from_slice(bytes);
        MappedBytes::new(Arc::new(m.make_read_only().unwrap()), 0, bytes.len()).unwrap()
    }

    /// The vector byte-to-f32 widening agrees with the decoder table on every e4m3 byte that
    /// is a number.
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    #[test]
    fn vector_widening_matches_the_table() {
        let bytes: Vec<u8> = (0..=255u8).filter(|b| b & 0x7F != 0x7F).collect();
        let n = bytes.len() / 8 * 8;
        for (i, &b) in bytes[..n].iter().enumerate() {
            // One-hot activation picks lane i's widened value out of the dot.
            let mut x = vec![0f32; n];
            x[i] = 1.0;
            let got = unsafe { tile_dot_avx2(bytes.as_ptr(), x.as_ptr(), n) };
            // Equal as numbers: the dot's sum turns a widened -0 into +0.
            assert!(
                got == e4m3_to_f32(b),
                "byte {b:#04x}: {got} vs {}",
                e4m3_to_f32(b)
            );
        }
    }

    /// The in-place product equals the dense matmul over the dequantised weight, and a row
    /// range of it equals the same rows of that product.
    #[test]
    fn in_place_product_and_row_range_match_dequantised_matmul() {
        let (out, inp, block, n) = (96usize, 160usize, 32usize, 3usize);
        // Bytes that avoid the NaN pattern (S1111.111).
        let bytes: Vec<u8> = (0..out * inp)
            .map(|i| ((i * 29 + 7) % 251) as u8)
            .map(|b| if b & 0x7F == 0x7F { 0x40 } else { b })
            .collect();
        let scales: Vec<u8> = (0..(out / block) * (inp / block))
            .map(|i| (118 + (i % 11)) as u8)
            .collect();
        let scales_f: Vec<f32> = scales.iter().map(|&s| e8m0_to_f32(s)).collect();
        let dense = dequant_fp8_blocked(&bytes, &scales_f, out, inp, block);
        let w = Tensor::from_vec(dense, (out, inp), &Device::Cpu).unwrap();
        let x = Tensor::from_vec(
            (0..n * inp).map(|i| ((i as f32) * 0.31).cos()).collect(),
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
        let fw = Fp8Weight::new(mapped(&bytes), mapped(&scales), out, inp, block).unwrap();
        let got = fw
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{g} vs {w}");
        }
        let sub = fw.rows(32, 32).unwrap();
        let got_sub = sub
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for t in 0..n {
            for o in 0..32 {
                let (g, w) = (got_sub[t * 32 + o], want[t * out + 32 + o]);
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "row range: {g} vs {w}"
                );
            }
        }
        assert!(fw.rows(5, 32).is_err());
    }
}
