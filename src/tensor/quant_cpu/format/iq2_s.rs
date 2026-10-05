//! iq2_s - 256 values in 82 bytes: ten-bit codes into a 1024-entry grid of eight magnitudes (the
//! top two bits in a side array), a sign bit per value, and a four-bit scale per sixteen values.
//! Read-only.

use super::iq_grids::IQ2S_GRID;
use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq2S {
    pub(crate) d: f16,
    /// Codes (the low eight bits) of the first half; the signs follow in the second half.
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIq2S>() == 82);

impl BlockFormat for BlockIq2S {
    const DTYPE: GgmlDType = GgmlDType::Iq2S;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; QK_K / 4],
            qh: [0; QK_K / 32],
            scales: [0; QK_K / 32],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (x, out) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            let d = x.d.to_f32();
            let signs = &x.qs[QK_K / 8..];
            for (ib, sub) in out.chunks_exact_mut(32).enumerate() {
                let db = [
                    d * (0.5 + (x.scales[ib] & 0x0F) as f32) * 0.25,
                    d * (0.5 + (x.scales[ib] >> 4) as f32) * 0.25,
                ];
                for l in 0..4 {
                    let code = x.qs[4 * ib + l] as usize
                        | ((((x.qh[ib] as u32) << (8 - 2 * l)) & 0x300) as usize);
                    let grid = IQ2S_GRID[code].to_le_bytes();
                    let sign = signs[4 * ib + l];
                    for j in 0..8 {
                        let s = if sign & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        sub[8 * l + j] = db[l / 2] * grid[j] as f32 * s;
                    }
                }
            }
        }
    }

    fn quantize(_: &[f32], _: &mut [Self]) {
        panic!("iq2_s is read, not written");
    }

    /// Runs of thirty-two weights carry two scales, one for each half: each half is its own integer
    /// dot against the activation's codes.
    #[allow(unreachable_code)]
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return unsafe {
            use std::arch::x86_64::*;
            let mut total = _mm256_setzero_ps();
            for (x, y) in xs.iter().zip(ys) {
                let signs_all = &x.qs[QK_K / 8..];
                let d = x.d.to_f32();
                let mut block = _mm256_setzero_ps();
                for ib in 0..QK_K / 32 {
                    // Code `l` is its byte with bits `2l` and `2l + 1` of the run's high byte above it.
                    let low = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(
                        x.qs[4 * ib..4 * ib + 4].try_into().unwrap(),
                    )));
                    let high = _mm_slli_epi32(
                        _mm_and_si128(
                            _mm_srlv_epi32(
                                _mm_set1_epi32(x.qh[ib] as i32),
                                _mm_setr_epi32(0, 2, 4, 6),
                            ),
                            _mm_set1_epi32(3),
                        ),
                        8,
                    );
                    let mags = _mm256_i32gather_epi64(
                        IQ2S_GRID.as_ptr() as *const i64,
                        _mm_or_si128(low, high),
                        8,
                    );
                    let signs =
                        u32::from_le_bytes(signs_all[4 * ib..4 * ib + 4].try_into().unwrap());
                    let sums = super::iq_read::grid_run_sums(mags, signs, &y.qs[32 * ib..]);
                    // Two scales: the first sixteen values' lanes under one, the rest under the other.
                    let (lo, hi) = (x.scales[ib] & 0x0F, x.scales[ib] >> 4);
                    let scale = _mm256_set_m128(
                        _mm_set1_ps(d * (0.5 + hi as f32) * 0.25),
                        _mm_set1_ps(d * (0.5 + lo as f32) * 0.25),
                    );
                    block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), scale, block);
                }
                total = _mm256_fmadd_ps(block, _mm256_set1_ps(y.d), total);
            }
            super::iq_read::sum_ps(total)
        };
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let signs = &x.qs[QK_K / 8..];
                let d = x.d.to_f32();
                let mut block = 0f32;
                for ib in 0..QK_K / 32 {
                    let mut w = [0i8; 32];
                    for l in 0..4 {
                        let code = x.qs[4 * ib + l] as usize
                            | ((((x.qh[ib] as u32) << (8 - 2 * l)) & 0x300) as usize);
                        let grid = IQ2S_GRID[code].to_le_bytes();
                        let sign = signs[4 * ib + l];
                        for j in 0..8 {
                            w[8 * l + j] = if sign & KMASK_IQ2XS[j] != 0 {
                                -(grid[j] as i8)
                            } else {
                                grid[j] as i8
                            };
                        }
                    }
                    let a = &y.qs[32 * ib..32 * ib + 32];
                    let half = |r: std::ops::Range<usize>| -> i32 {
                        w[r.clone()]
                            .iter()
                            .zip(&a[r])
                            .map(|(&w, &a)| w as i32 * a as i32)
                            .sum()
                    };
                    let (lo, hi) = (x.scales[ib] & 0x0F, x.scales[ib] >> 4);
                    block += d * (0.5 + lo as f32) * 0.25 * half(0..16) as f32
                        + d * (0.5 + hi as f32) * 0.25 * half(16..32) as f32;
                }
                block * y.d
            })
            .sum()
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        super::iq_read::dot_by_decoding(xs, ys)
    }
}
