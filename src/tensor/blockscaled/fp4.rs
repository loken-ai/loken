//! A block-scaled fp4 projection read in place: the released checkpoint's routed experts.
//!
//! A row of `[out, in]` is stored as `in / 2` bytes of e2m1 nibbles (element `2j` in the low
//! nibble, `2j + 1` in the high) and `in / block` ue8m0 scales, one per `block` elements along
//! the input. The activation is quantised to 8 bits per block of inputs (the block's largest
//! magnitude at 127), its elements reordered as the nibbles are stored - the low nibbles'
//! elements first, the high nibbles' next - so a 16-byte chunk of the row and 32 bytes of the
//! activation pair up lane for lane. Each block's products are summed exactly as integers (the
//! e2m1 values doubled, so they are integers too) and scaled once, straight from the mapping:
//! an expert costs the bytes it occupies and nothing is dequantised or copied. The AVX2 path
//! and the scalar one compute the same integers and the same f32 sequence, so they agree bit
//! for bit.

use super::dequant::e8m0_to_f32;
use crate::tensor::mapped::MappedBytes;
use crate::tensor::{Device, Error, Result, Tensor};
use rayon::prelude::*;

/// The e2m1 values, doubled to integers, by nibble (bit 3 the sign).
const E2M1_X2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// The width of one nibble chunk in elements: 16 bytes of the row, 32 elements.
const CHUNK: usize = 32;

/// One `[out, in]` fp4 weight in its storage.
pub struct Fp4Weight {
    pub nibbles: MappedBytes,
    pub scales: MappedBytes,
    pub out: usize,
    pub inp: usize,
    pub block: usize,
}

/// Activation rows quantised per block: `q[t][i]` in i8 with the block's elements reordered as
/// the nibbles are stored, `d[t][b]` the block's scale.
struct Q8Rows {
    q: Vec<i8>,
    d: Vec<f32>,
}

impl Q8Rows {
    fn quantize(xs: &[f32], n: usize, inp: usize, block: usize) -> Self {
        let per_row = inp / block;
        let mut q = vec![0i8; n * inp];
        let mut d = vec![0f32; n * per_row];
        for t in 0..n {
            let x = &xs[t * inp..(t + 1) * inp];
            for b in 0..per_row {
                let xb = &x[b * block..(b + 1) * block];
                let amax = xb.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let scale = amax / 127.0;
                let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
                d[t * per_row + b] = scale;
                let qb = &mut q[t * inp + b * block..t * inp + (b + 1) * block];
                for c in 0..block / CHUNK {
                    for j in 0..CHUNK / 2 {
                        let even = (xb[c * CHUNK + 2 * j] * inv).round().clamp(-127.0, 127.0) as i8;
                        let odd = (xb[c * CHUNK + 2 * j + 1] * inv)
                            .round()
                            .clamp(-127.0, 127.0) as i8;
                        qb[c * CHUNK + j] = even;
                        qb[c * CHUNK + CHUNK / 2 + j] = odd;
                    }
                }
            }
        }
        Self { q, d }
    }
}

/// The integer dot of one 16-byte chunk of nibbles with 32 reordered i8 activations: the path
/// without AVX2, and what the vector path is checked against.
#[cfg_attr(all(target_feature = "avx2", target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
fn chunk_dot_scalar(bytes: &[u8], x: &[i8]) -> i32 {
    let mut s = 0i32;
    for j in 0..CHUNK / 2 {
        let b = bytes[j];
        s += E2M1_X2[(b & 0x0F) as usize] as i32 * x[j] as i32;
        s += E2M1_X2[(b >> 4) as usize] as i32 * x[CHUNK / 2 + j] as i32;
    }
    s
}

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn chunk_dot_avx2(bytes: *const u8, x: *const i8) -> i32 {
    use core::arch::x86_64::*;
    let raw = _mm_loadu_si128(bytes as *const __m128i);
    let m4 = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(raw, m4);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), m4);
    let nib = _mm256_set_m128i(hi, lo);
    let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(E2M1_X2.as_ptr() as *const __m128i));
    let w = _mm256_shuffle_epi8(lut, nib);
    let xv = _mm256_loadu_si256(x as *const __m256i);
    // maddubs takes an unsigned and a signed operand: |x| against w carrying x's sign.
    let ax = _mm256_sign_epi8(xv, xv);
    let sw = _mm256_sign_epi8(w, xv);
    let dot16 = _mm256_maddubs_epi16(ax, sw);
    let dot32 = _mm256_madd_epi16(_mm256_set1_epi16(1), dot16);
    let lo128 = _mm256_castsi256_si128(dot32);
    let hi128 = _mm256_extracti128_si256::<1>(dot32);
    let s4 = _mm_add_epi32(lo128, hi128);
    let s2 = _mm_add_epi32(s4, _mm_shuffle_epi32::<0b01_00_11_10>(s4));
    let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32::<0b10_11_00_01>(s2));
    _mm_cvtsi128_si32(s1)
}

#[inline(always)]
fn chunk_dot(bytes: &[u8], x: &[i8]) -> i32 {
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        // Safety: both slices are at least one chunk long, checked by the callers' layout.
        unsafe { chunk_dot_avx2(bytes.as_ptr(), x.as_ptr()) }
    }
    #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
    {
        chunk_dot_scalar(bytes, x)
    }
}

impl Fp4Weight {
    pub fn new(nibbles: MappedBytes, scales: MappedBytes, out: usize, inp: usize) -> Result<Self> {
        if nibbles.len() != out * inp / 2 || !inp.is_multiple_of(2) {
            return Err(Error::msg(format!(
                "fp4 weight: {} nibble bytes for [{out}, {inp}]",
                nibbles.len()
            )));
        }
        if scales.is_empty()
            || !scales.len().is_multiple_of(out)
            || !inp.is_multiple_of(scales.len() / out)
        {
            return Err(Error::msg(format!(
                "fp4 weight: {} scales for [{out}, {inp}]",
                scales.len()
            )));
        }
        let block = inp / (scales.len() / out);
        if !block.is_multiple_of(CHUNK) {
            return Err(Error::msg(format!(
                "fp4 weight: a block of {block} inputs is not a whole number of {CHUNK}-element chunks"
            )));
        }
        Ok(Self {
            nibbles,
            scales,
            out,
            inp,
            block,
        })
    }

    /// `x` [n, in] f32 -> [n, out] f32.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims
            .last()
            .ok_or_else(|| Error::msg("fp4 apply: rank-0 input"))?;
        if k != self.inp {
            return Err(Error::msg(format!(
                "fp4 apply: input {k} != weight in {}",
                self.inp
            )));
        }
        let n = x.elem_count() / k;
        let xs = x.flatten_all()?.to_vec1::<f32>()?;
        let act = Q8Rows::quantize(&xs, n, self.inp, self.block);
        let y = self.apply_q8(&act, n);
        let mut odims = dims;
        *odims.last_mut().unwrap() = self.out;
        Tensor::from_vec(y, odims, &Device::Cpu)
    }

    /// The product over already-quantised activations, `[n, out]` row-major.
    fn apply_q8(&self, act: &Q8Rows, n: usize) -> Vec<f32> {
        let nib = self.nibbles.as_slice();
        let sc = self.scales.as_slice();
        let (inp, block, half) = (self.inp, self.block, self.inp / 2);
        let per_row = inp / block;
        let chunks = block / CHUNK;
        let mut y = vec![0f32; n * self.out];
        // Output rows are independent: one weight row read once for every activation row.
        y.par_chunks_mut(n).enumerate().for_each(|(o, col)| {
            let row = &nib[o * half..(o + 1) * half];
            let scales = &sc[o * per_row..(o + 1) * per_row];
            for (t, out) in col.iter_mut().enumerate() {
                let xq = &act.q[t * inp..(t + 1) * inp];
                let xd = &act.d[t * per_row..(t + 1) * per_row];
                let mut acc = 0f32;
                for (b, &s) in scales.iter().enumerate() {
                    let mut sum = 0i32;
                    for c in 0..chunks {
                        let at = b * block + c * CHUNK;
                        sum += chunk_dot(&row[at / 2..at / 2 + CHUNK / 2], &xq[at..at + CHUNK]);
                    }
                    acc += sum as f32 * (xd[b] * e8m0_to_f32(s) * 0.5);
                }
                *out = acc;
            }
        });
        // y is [out, n]; the caller wants [n, out].
        let mut yt = vec![0f32; n * self.out];
        for o in 0..self.out {
            for t in 0..n {
                yt[t * self.out + o] = y[o * n + t];
            }
        }
        yt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::blockscaled::dequant::dequant_fp4_blocked;
    use std::sync::Arc;

    fn mapped(bytes: &[u8]) -> MappedBytes {
        let mut m = memmap2::MmapMut::map_anon(bytes.len().max(1)).unwrap();
        m[..bytes.len()].copy_from_slice(bytes);
        MappedBytes::new(Arc::new(m.make_read_only().unwrap()), 0, bytes.len()).unwrap()
    }

    fn fixture(out: usize, inp: usize, block: usize, n: usize) -> (Vec<u8>, Vec<u8>, Tensor) {
        let nibbles: Vec<u8> = (0..out * inp / 2)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect();
        let scales: Vec<u8> = (0..out * inp / block)
            .map(|i| (120 + (i % 9)) as u8)
            .collect();
        let x = Tensor::from_vec(
            (0..n * inp).map(|i| ((i as f32) * 0.37).sin()).collect(),
            (n, inp),
            &Device::Cpu,
        )
        .unwrap();
        (nibbles, scales, x)
    }

    /// The in-place product equals the dense matmul over the dequantised weight to within the
    /// activation's 8-bit quantisation.
    #[test]
    fn in_place_product_matches_dequantised_matmul() {
        let (out, inp, block, n) = (24usize, 128usize, 32usize, 3usize);
        let (nibbles, scales, x) = fixture(out, inp, block, n);
        let scales_f: Vec<f32> = scales.iter().map(|&s| e8m0_to_f32(s)).collect();
        let dense = dequant_fp4_blocked(&nibbles, &scales_f, out, inp, block);
        let w = Tensor::from_vec(dense, (out, inp), &Device::Cpu).unwrap();
        let want = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let fw = Fp4Weight::new(mapped(&nibbles), mapped(&scales), out, inp).unwrap();
        assert_eq!(fw.block, block);
        let got = fw
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs()));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-2 * scale, "{g} vs {w} (scale {scale})");
        }
    }

    /// On a card the weight is dequantised and applied at full precision: the dense matmul over the
    /// CPU's dequantisation, to rounding.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_card_applies_the_dequantised_weight() {
        use crate::tensor::cuda::{gpu_fp4_linear, CudaDevice};
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card fp4 product is NOT covered by this run");
            return;
        };
        let (out, inp, block, n) = (24usize, 128usize, 32usize, 3usize);
        let (nibbles, mut scales, x) = fixture(out, inp, block, n);
        scales[5] = 0xFF;
        let scales_f: Vec<f32> = scales.iter().map(|&s| e8m0_to_f32(s)).collect();
        let dense = dequant_fp4_blocked(&nibbles, &scales_f, out, inp, block);
        let w = Tensor::from_vec(dense, (out, inp), &Device::Cpu).unwrap();
        let want = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let xs = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = gpu_fp4_linear(&dev, &nibbles, &scales, (out, inp), &xs).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            if w.is_nan() {
                assert!(g.is_nan(), "{g} where the CPU gives NaN");
            } else {
                assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{g} vs {w}");
            }
        }
    }

    /// A block wider than one chunk is a whole number of chunks, and a block that is not is
    /// refused.
    #[test]
    fn wider_blocks_and_odd_blocks() {
        let (out, inp, block, n) = (8usize, 256usize, 64usize, 2usize);
        let (nibbles, scales, x) = fixture(out, inp, block, n);
        let fw = Fp4Weight::new(mapped(&nibbles), mapped(&scales), out, inp).unwrap();
        assert_eq!(fw.block, 64);
        assert!(fw
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        let (nibbles, scales, _) = fixture(8, 96, 48, 1);
        assert!(Fp4Weight::new(mapped(&nibbles), mapped(&scales), 8, 96).is_err());
    }

    /// The vector chunk dot equals the scalar one exactly.
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    #[test]
    fn vector_chunk_dot_equals_scalar() {
        let bytes: Vec<u8> = (0..16u32).map(|i| ((i * 97 + 3) % 256) as u8).collect();
        let x: Vec<i8> = (0..32i32).map(|i| ((i * 53) % 255 - 127) as i8).collect();
        let want = chunk_dot_scalar(&bytes, &x);
        let got = unsafe { chunk_dot_avx2(bytes.as_ptr(), x.as_ptr()) };
        assert_eq!(got, want);
    }
}
