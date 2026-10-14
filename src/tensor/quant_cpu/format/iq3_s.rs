//! iq3_s - 256 values in 110 bytes: nine-bit codes into a 512-entry grid of four magnitudes (the
//! ninth bit in a side array), an explicit sign bit per value, and a four-bit scale per sixty-four.
//! Read-only.

use super::iq_grids::IQ3S_GRID;
use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq3S {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) signs: [u8; QK_K / 8],
    pub(crate) scales: [u8; QK_K / 64],
}
const _: () = assert!(std::mem::size_of::<BlockIq3S>() == 110);

impl BlockFormat for BlockIq3S {
    const DTYPE: GgmlDType = GgmlDType::Iq3S;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; QK_K / 4],
            qh: [0; QK_K / 32],
            signs: [0; QK_K / 8],
            scales: [0; QK_K / 64],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (x, out) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            let d = x.d.to_f32();
            // Two sub-blocks of thirty-two per step: one scale nibble each.
            for (pair, chunk) in out.chunks_exact_mut(64).enumerate() {
                let low = d * (1 + 2 * (x.scales[pair] & 0x0F) as i32) as f32;
                let high = d * (1 + 2 * (x.scales[pair] >> 4) as i32) as f32;
                for half in 0..2 {
                    let db = if half == 0 { low } else { high };
                    let qh = x.qh[2 * pair + half] as u32;
                    let q = &x.qs[16 * pair + 8 * half..16 * pair + 8 * half + 8];
                    let signs = &x.signs[8 * pair + 4 * half..8 * pair + 4 * half + 4];
                    let sub = &mut chunk[32 * half..32 * half + 32];
                    for l in 0..4 {
                        let i1 = q[2 * l] as usize | (((qh << (8 - 2 * l)) & 256) as usize);
                        let i2 = q[2 * l + 1] as usize | (((qh << (7 - 2 * l)) & 256) as usize);
                        let (g1, g2) = (IQ3S_GRID[i1].to_le_bytes(), IQ3S_GRID[i2].to_le_bytes());
                        for j in 0..4 {
                            let s1 = if signs[l] & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let s2 = if signs[l] & KMASK_IQ2XS[j + 4] != 0 {
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
    }

    fn quantize(_: &[f32], _: &mut [Self]) {
        panic!("iq3_s is read, not written");
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            // SAFETY: dispatch_avx2 implies the CPU has every feature `dot_avx2` enables.
            return unsafe { dot_avx2(xs, ys) };
        }
        super::iq_read::dot_runs(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        super::iq_read::dot_by_decoding(xs, ys)
    }
}

impl super::iq_read::Runs for BlockIq3S {
    fn run_scale(&self, ib: usize) -> f32 {
        let scale = if ib.is_multiple_of(2) {
            self.scales[ib / 2] & 0x0F
        } else {
            self.scales[ib / 2] >> 4
        };
        self.d.to_f32() * (1 + 2 * scale as i32) as f32
    }

    fn run_values(&self, ib: usize) -> [i8; 32] {
        let qh = self.qh[ib] as u32;
        let q = &self.qs[8 * ib..8 * ib + 8];
        let signs = &self.signs[4 * ib..4 * ib + 4];
        let mut w = [0i8; 32];
        for l in 0..4 {
            let i1 = q[2 * l] as usize | (((qh << (8 - 2 * l)) & 256) as usize);
            let i2 = q[2 * l + 1] as usize | (((qh << (7 - 2 * l)) & 256) as usize);
            let (g1, g2) = (IQ3S_GRID[i1].to_le_bytes(), IQ3S_GRID[i2].to_le_bytes());
            for j in 0..4 {
                w[8 * l + j] = if signs[l] & KMASK_IQ2XS[j] != 0 {
                    -(g1[j] as i8)
                } else {
                    g1[j] as i8
                };
                w[8 * l + j + 4] = if signs[l] & KMASK_IQ2XS[j + 4] != 0 {
                    -(g2[j] as i8)
                } else {
                    g2[j] as i8
                };
            }
        }
        w
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn dot_avx2(xs: &[BlockIq3S], ys: &[<BlockIq3S as BlockFormat>::ActivationBlock]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let mut total = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys) {
            let mut block = _mm256_setzero_ps();
            for ib in 0..QK_K / 32 {
                // Code `c` of a run is its byte with bit `c` of the run's high-bit byte as the
                // ninth bit.
                let low = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    x.qs[8 * ib..].as_ptr() as *const __m128i
                ));
                let high = _mm256_slli_epi32(
                    _mm256_and_si256(
                        _mm256_srlv_epi32(
                            _mm256_set1_epi32(x.qh[ib] as i32),
                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                        ),
                        _mm256_set1_epi32(1),
                    ),
                    8,
                );
                let mags = _mm256_i32gather_epi32(
                    IQ3S_GRID.as_ptr() as *const i32,
                    _mm256_or_si256(low, high),
                    4,
                );
                let signs = u32::from_le_bytes(x.signs[4 * ib..4 * ib + 4].try_into().unwrap());
                let sums = super::iq_read::grid_run_sums(mags, signs, &y.qs[32 * ib..]);
                let scale = _mm256_set1_ps(super::iq_read::Runs::run_scale(x, ib));
                block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), scale, block);
            }
            total = _mm256_fmadd_ps(block, _mm256_set1_ps(y.d), total);
        }
        super::iq_read::sum_ps(total)
    }
}
