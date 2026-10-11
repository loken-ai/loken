//! iq3_xxs - 256 values in 98 bytes: codes into a 256-entry grid of four magnitudes, a 7-bit sign
//! pattern per eight values, and a four-bit scale per thirty-two. Read-only.

use super::iq_grids::IQ3XXS_GRID;
use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq3Xxs {
    pub(crate) d: f16,
    pub(crate) qs: [u8; 3 * QK_K / 8],
}
const _: () = assert!(std::mem::size_of::<BlockIq3Xxs>() == 98);

impl BlockFormat for BlockIq3Xxs {
    const DTYPE: GgmlDType = GgmlDType::Iq3Xxs;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K32;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; 3 * QK_K / 8],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (x, out) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            let d = x.d.to_f32();
            let (codes, tail) = x.qs.split_at(QK_K / 4);
            for (ib, sub) in out.chunks_exact_mut(32).enumerate() {
                let word = u32::from_le_bytes(tail[4 * ib..4 * ib + 4].try_into().unwrap());
                let db = d * (0.5 + (word >> 28) as f32) * 0.5;
                let q = &codes[8 * ib..8 * ib + 8];
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((word >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[q[2 * l] as usize].to_le_bytes();
                    let g2 = IQ3XXS_GRID[q[2 * l + 1] as usize].to_le_bytes();
                    for j in 0..4 {
                        let s1 = if signs & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        sub[8 * l + j] = db * g1[j] as f32 * s1;
                        sub[8 * l + j + 4] = db * g2[j] as f32 * s2;
                    }
                }
            }
        }
    }

    fn quantize(_: &[f32], _: &mut [Self]) {
        panic!("iq3_xxs is read, not written");
    }

    #[allow(unreachable_code)]
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return unsafe {
            use std::arch::x86_64::*;
            // A run's sum stays integer under its odd scale until the run ends, and takes the
            // run's activation scale then.
            let mut total = _mm256_setzero_ps();
            for (x, y) in xs.iter().zip(ys) {
                let d = x.d.to_f32() * 0.25;
                for ib in 0..QK_K / 32 {
                    // The magnitudes are read one by one: a gather is slow on a processor whose
                    // microcode blocks the leak of its data.
                    let codes = u64::from_le_bytes(x.qs[8 * ib..8 * ib + 8].try_into().unwrap());
                    let grid = |l: u32| IQ3XXS_GRID[((codes >> (8 * l)) & 0xFF) as usize] as i32;
                    let pair = |l: u32| (grid(l + 1) as u32 as i64) << 32 | grid(l) as u32 as i64;
                    let mags = _mm256_set_m128i(
                        _mm_set_epi64x(pair(6), pair(4)),
                        _mm_set_epi64x(pair(2), pair(0)),
                    );
                    let tail = &x.qs[QK_K / 4 + 4 * ib..QK_K / 4 + 4 * ib + 4];
                    let word = u32::from_le_bytes(tail.try_into().unwrap());
                    let signs = (0..4).fold(0u32, |acc, l| {
                        acc | (KSIGNS_IQ2XS[((word >> (7 * l)) & 127) as usize] as u32) << (8 * l)
                    });
                    let pairs = super::iq_read::grid_run_pairs(mags, signs, &y.qs[32 * ib..]);
                    let scale = _mm256_set1_epi16(2 * (word >> 28) as i16 + 1);
                    total = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, scale)),
                        _mm256_set1_ps(d * y.d[ib].to_f32()),
                        total,
                    );
                }
            }
            super::iq_read::sum_ps(total)
        };
        super::iq_read::dot_runs(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        super::iq_read::dot_by_decoding(xs, ys)
    }
}

impl super::iq_read::Runs for BlockIq3Xxs {
    fn run_scale(&self, ib: usize) -> f32 {
        let tail = &self.qs[QK_K / 4 + 4 * ib..QK_K / 4 + 4 * ib + 4];
        let word = u32::from_le_bytes(tail.try_into().unwrap());
        self.d.to_f32() * (0.5 + (word >> 28) as f32) * 0.5
    }

    fn run_values(&self, ib: usize) -> [i8; 32] {
        let codes = &self.qs[8 * ib..8 * ib + 8];
        let tail = &self.qs[QK_K / 4 + 4 * ib..QK_K / 4 + 4 * ib + 4];
        let word = u32::from_le_bytes(tail.try_into().unwrap());
        let mut w = [0i8; 32];
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((word >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[codes[2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[codes[2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                w[8 * l + j] = if signs & KMASK_IQ2XS[j] != 0 {
                    -(g1[j] as i8)
                } else {
                    g1[j] as i8
                };
                w[8 * l + j + 4] = if signs & KMASK_IQ2XS[j + 4] != 0 {
                    -(g2[j] as i8)
                } else {
                    g2[j] as i8
                };
            }
        }
        w
    }
}
