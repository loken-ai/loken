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
    type ActivationBlock = BlockQ8K32;

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

    /// [`Self::dot`] over several activations, each run's magnitudes, signs and scale decoded
    /// once.
    fn dot_many(xs: &[Self], ys: &[&[Self::ActivationBlock]], out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            // SAFETY: dispatch_avx2 implies the CPU has every feature `dot_many_avx2` enables.
            return unsafe { dot_many_avx2(xs, ys, out) };
        }
        for (o, y) in out.iter_mut().zip(ys) {
            *o = Self::dot(xs, y);
        }
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
        // A run's sum stays integer under its odd scale until the run ends, and takes the
        // run's activation scale then.
        let mut total = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys) {
            let d = x.d.to_f32();
            for ib in 0..QK_K / 32 {
                // Code `c` of a run is its byte with bit `c` of the run's high-bit byte as the
                // ninth bit. The magnitudes are read one by one: a gather is slow on a
                // processor whose microcode blocks the leak of its data.
                let codes = u64::from_le_bytes(x.qs[8 * ib..8 * ib + 8].try_into().unwrap());
                let high = x.qh[ib] as u32;
                let grid = |l: u32| {
                    IQ3S_GRID
                        [(((codes >> (8 * l)) as u32 & 0xFF) | (((high >> l) & 1) << 8)) as usize]
                        as i32
                };
                let pair = |l: u32| (grid(l + 1) as u32 as i64) << 32 | grid(l) as u32 as i64;
                let mags = _mm256_set_m128i(
                    _mm_set_epi64x(pair(6), pair(4)),
                    _mm_set_epi64x(pair(2), pair(0)),
                );
                let signs = u32::from_le_bytes(x.signs[4 * ib..4 * ib + 4].try_into().unwrap());
                let pairs = super::iq_read::grid_run_pairs(mags, signs, &y.qs[32 * ib..]);
                let nibble = if ib % 2 == 0 {
                    x.scales[ib / 2] & 0x0F
                } else {
                    x.scales[ib / 2] >> 4
                };
                let scale = _mm256_set1_epi16(2 * nibble as i16 + 1);
                total = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, scale)),
                    _mm256_set1_ps(d * y.d[ib].to_f32()),
                    total,
                );
            }
        }
        super::iq_read::sum_ps(total)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn dot_many_avx2(
    xs: &[BlockIq3S],
    ys: &[&[<BlockIq3S as BlockFormat>::ActivationBlock]],
    out: &mut [f32],
) {
    unsafe {
        use std::arch::x86_64::*;
        super::iq_read::dot_many_runs(
            xs,
            ys,
            out,
            |x: &BlockIq3S, ib| {
                let codes = u64::from_le_bytes(x.qs[8 * ib..8 * ib + 8].try_into().unwrap());
                let high = x.qh[ib] as u32;
                let grid = |l: u32| {
                    IQ3S_GRID
                        [(((codes >> (8 * l)) as u32 & 0xFF) | (((high >> l) & 1) << 8)) as usize]
                        as i32
                };
                let pair = |l: u32| (grid(l + 1) as u32 as i64) << 32 | grid(l) as u32 as i64;
                let mags = _mm256_set_m128i(
                    _mm_set_epi64x(pair(6), pair(4)),
                    _mm_set_epi64x(pair(2), pair(0)),
                );
                let signs = u32::from_le_bytes(x.signs[4 * ib..4 * ib + 4].try_into().unwrap());
                let nibble = if ib % 2 == 0 {
                    x.scales[ib / 2] & 0x0F
                } else {
                    x.scales[ib / 2] >> 4
                };
                let scale = _mm256_set1_epi16(2 * nibble as i16 + 1);
                (
                    (mags, super::iq_read::run_negate(signs), scale),
                    x.d.to_f32(),
                )
            },
            |(mags, negate, scale), acts| {
                _mm256_madd_epi16(super::iq_read::run_pairs(mags, negate, acts), scale)
            },
        )
    }
}
