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

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx512() {
            // SAFETY: dispatch_avx512 implies the CPU has every feature `dot_avx512` enables.
            return unsafe { dot_avx512(xs, ys) };
        }
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
        if crate::cpu::dispatch_avx512() {
            // SAFETY: dispatch_avx512 implies the CPU has every feature `dot_many_avx512` enables.
            return unsafe { dot_many_avx512(xs, ys, out) };
        }
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

/// The run-by-run dot the grid kernel replaced, kept as its reference.
#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn reference_dot_avx2(
    xs: &[BlockIq3Xxs],
    ys: &[<BlockIq3Xxs as BlockFormat>::ActivationBlock],
) -> f32 {
    unsafe {
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
    }
}

/// The run-by-run dot over several activations the grid kernel replaced, kept as its reference.
#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn reference_dot_many_avx2(
    xs: &[BlockIq3Xxs],
    ys: &[&[<BlockIq3Xxs as BlockFormat>::ActivationBlock]],
    out: &mut [f32],
) {
    unsafe {
        use std::arch::x86_64::*;
        super::iq_read::dot_many_runs(
            xs,
            ys,
            out,
            |x: &BlockIq3Xxs, ib| {
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
                let scale = _mm256_set1_epi16(2 * (word >> 28) as i16 + 1);
                (
                    (mags, super::iq_read::run_negate(signs), scale),
                    x.d.to_f32() * 0.25,
                )
            },
            |(mags, negate, scale), acts| {
                _mm256_madd_epi16(super::iq_read::run_pairs(mags, negate, acts), scale)
            },
        )
    }
}

/// What the runs of an iq3_xxs block share: each run's 32 sign bits, the eighth of each group
/// of eight being the parity of the seven stored, and each run's odd scale in both i16 halves
/// of a word.
#[cfg(target_arch = "x86_64")]
pub(super) struct Iq3XxsRuns {
    signs: [u32; QK_K / 32],
    scales: [u32; QK_K / 32],
}

#[cfg(target_arch = "x86_64")]
impl super::iq_grid_rows::GridRun for BlockIq3Xxs {
    type Block = Iq3XxsRuns;

    #[inline(always)]
    fn run_unit(&self) -> f32 {
        self.d.to_f32() * 0.25
    }

    #[inline(always)]
    unsafe fn block(&self) -> Iq3XxsRuns {
        unsafe {
            use std::arch::x86_64::*;
            // Word `ib` of the tail: four seven-bit sign groups, then the run's four-bit scale.
            let tails = _mm256_loadu_si256(self.qs.as_ptr().add(QK_K / 4) as *const __m256i);
            let odd = _mm256_or_si256(
                _mm256_slli_epi32(_mm256_srli_epi32(tails, 28), 1),
                _mm256_set1_epi32(1),
            );
            let odd = _mm256_or_si256(odd, _mm256_slli_epi32(odd, 16));
            // Group `l` moved to byte `l`.
            let group = |l: i32, shifted: __m256i| {
                _mm256_and_si256(shifted, _mm256_set1_epi32(0x7F << (8 * l)))
            };
            let groups = _mm256_or_si256(
                _mm256_or_si256(group(0, tails), group(1, _mm256_slli_epi32(tails, 1))),
                _mm256_or_si256(
                    group(2, _mm256_slli_epi32(tails, 2)),
                    group(3, _mm256_slli_epi32(tails, 3)),
                ),
            );
            // Bit 0 of each byte becomes the parity of its byte.
            let mut parity = _mm256_xor_si256(groups, _mm256_srli_epi32(groups, 4));
            parity = _mm256_xor_si256(parity, _mm256_srli_epi32(parity, 2));
            parity = _mm256_xor_si256(parity, _mm256_srli_epi32(parity, 1));
            let parity = _mm256_and_si256(parity, _mm256_set1_epi8(1));
            let signs = _mm256_or_si256(groups, _mm256_slli_epi32(parity, 7));
            let mut runs = Iq3XxsRuns {
                signs: [0; QK_K / 32],
                scales: [0; QK_K / 32],
            };
            _mm256_storeu_si256(runs.signs.as_mut_ptr() as *mut __m256i, signs);
            _mm256_storeu_si256(runs.scales.as_mut_ptr() as *mut __m256i, odd);
            runs
        }
    }

    #[inline(always)]
    unsafe fn grid_run<const IB: usize>(
        &self,
        block: &Iq3XxsRuns,
    ) -> (
        std::arch::x86_64::__m256i,
        std::arch::x86_64::__m256i,
        std::arch::x86_64::__m256i,
    ) {
        unsafe {
            let mags =
                super::iq_grid_rows::table_u32_by_bytes::<IB>(&IQ3XXS_GRID, self.qs.as_ptr());
            (
                mags,
                super::iq_grid_rows::run_negate_spread(super::iq_grid_rows::broadcast_u32::<IB>(
                    block.signs.as_ptr(),
                )),
                super::iq_grid_rows::broadcast_u32::<IB>(block.scales.as_ptr()),
            )
        }
    }

    #[inline(always)]
    fn sign_words(&self, block: &Iq3XxsRuns) -> *const u32 {
        block.signs.as_ptr()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn dot_avx2(xs: &[BlockIq3Xxs], ys: &[BlockQ8K32]) -> f32 {
    unsafe { super::iq_grid_rows::dot_grid(xs, ys) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn dot_many_avx2(xs: &[BlockIq3Xxs], ys: &[&[BlockQ8K32]], out: &mut [f32]) {
    unsafe { super::iq_grid_rows::dot_many_grid(xs, ys, out) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq")]
unsafe fn dot_avx512(xs: &[BlockIq3Xxs], ys: &[BlockQ8K32]) -> f32 {
    unsafe { super::iq_grid_rows::dot_grid_avx512(xs, ys) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq")]
pub(super) unsafe fn dot_many_avx512(xs: &[BlockIq3Xxs], ys: &[&[BlockQ8K32]], out: &mut [f32]) {
    unsafe { super::iq_grid_rows::dot_many_grid_avx512(xs, ys, out) }
}
