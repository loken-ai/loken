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
    type ActivationBlock = BlockQ8K32;

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
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            // SAFETY: dispatch_avx2 implies the CPU has every feature `dot_avx2` enables.
            return unsafe { dot_avx2(xs, ys) };
        }
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
                    block += (d * (0.5 + lo as f32) * 0.25 * half(0..16) as f32
                        + d * (0.5 + hi as f32) * 0.25 * half(16..32) as f32)
                        * y.d[ib].to_f32();
                }
                block
            })
            .sum()
    }

    /// [`Self::dot`] over several activations, each run's magnitudes, signs and scales decoded
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

/// The run-by-run dot the grid kernel replaced, kept as its reference.
#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn reference_dot_avx2(
    xs: &[BlockIq2S],
    ys: &[<BlockIq2S as BlockFormat>::ActivationBlock],
) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let mut total = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys) {
            let signs_all = &x.qs[QK_K / 8..];
            // A run's sum stays integer under its odd scales until the run ends, and takes
            // the run's activation scale then.
            let d = x.d.to_f32() * 0.125;
            for ib in 0..QK_K / 32 {
                // Code `l` is its byte with bits `2l` and `2l + 1` of the run's high byte above it.
                // The magnitudes are read one by one: a gather is slow on a processor whose
                // microcode blocks the leak of its data.
                let codes = u32::from_le_bytes(x.qs[4 * ib..4 * ib + 4].try_into().unwrap());
                let high = (x.qh[ib] as u32) << 8;
                let grid = |l: u32| {
                    IQ2S_GRID[(((codes >> (8 * l)) & 0xFF) | ((high >> (2 * l)) & 0x300)) as usize]
                };
                let mags = _mm256_set_m128i(
                    _mm_set_epi64x(grid(3) as i64, grid(2) as i64),
                    _mm_set_epi64x(grid(1) as i64, grid(0) as i64),
                );
                let signs = u32::from_le_bytes(signs_all[4 * ib..4 * ib + 4].try_into().unwrap());
                let pairs = super::iq_read::grid_run_pairs(mags, signs, &y.qs[32 * ib..]);
                // Two scales: the first sixteen values' pairs under one, the rest under the other.
                let (lo, hi) = (x.scales[ib] & 0x0F, x.scales[ib] >> 4);
                let scale = _mm256_set_m128i(
                    _mm_set1_epi16(2 * hi as i16 + 1),
                    _mm_set1_epi16(2 * lo as i16 + 1),
                );
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
    xs: &[BlockIq2S],
    ys: &[&[<BlockIq2S as BlockFormat>::ActivationBlock]],
    out: &mut [f32],
) {
    unsafe {
        use std::arch::x86_64::*;
        super::iq_read::dot_many_runs(
            xs,
            ys,
            out,
            |x: &BlockIq2S, ib| {
                let codes = u32::from_le_bytes(x.qs[4 * ib..4 * ib + 4].try_into().unwrap());
                let high = (x.qh[ib] as u32) << 8;
                let grid = |l: u32| {
                    IQ2S_GRID[(((codes >> (8 * l)) & 0xFF) | ((high >> (2 * l)) & 0x300)) as usize]
                };
                let mags = _mm256_set_m128i(
                    _mm_set_epi64x(grid(3) as i64, grid(2) as i64),
                    _mm_set_epi64x(grid(1) as i64, grid(0) as i64),
                );
                let signs_all = &x.qs[QK_K / 8..];
                let signs = u32::from_le_bytes(signs_all[4 * ib..4 * ib + 4].try_into().unwrap());
                let (lo, hi) = (x.scales[ib] & 0x0F, x.scales[ib] >> 4);
                let scale = _mm256_set_m128i(
                    _mm_set1_epi16(2 * hi as i16 + 1),
                    _mm_set1_epi16(2 * lo as i16 + 1),
                );
                (
                    (mags, super::iq_read::run_negate(signs), scale),
                    x.d.to_f32() * 0.125,
                )
            },
            |(mags, negate, scale), acts| {
                _mm256_madd_epi16(super::iq_read::run_pairs(mags, negate, acts), scale)
            },
        )
    }
}

/// What the runs of an iq2_s block share: the ten-bit grid code of each group of eight, and the
/// runs' odd scales, the first halves' in the low 128 bits and the second halves' in the high.
#[cfg(target_arch = "x86_64")]
pub(super) struct Iq2SRuns {
    codes: [u16; QK_K / 8],
    scales: std::arch::x86_64::__m256i,
}

#[cfg(target_arch = "x86_64")]
impl super::iq_grid_rows::GridRun for BlockIq2S {
    type Block = Iq2SRuns;

    #[inline(always)]
    fn run_unit(&self) -> f32 {
        self.d.to_f32() * 0.125
    }

    #[inline(always)]
    unsafe fn block(&self) -> Iq2SRuns {
        unsafe {
            use std::arch::x86_64::*;
            let raw = _mm_loadl_epi64(self.scales.as_ptr() as *const __m128i);
            let wide = _mm_cvtepu8_epi16(raw);
            let halves = _mm256_set_m128i(
                _mm_srli_epi16(wide, 4),
                _mm_and_si128(wide, _mm_set1_epi16(0x0F)),
            );
            let scales = _mm256_or_si256(_mm256_add_epi16(halves, halves), _mm256_set1_epi16(1));
            // Code `c` is byte `c` with bits `2 (c % 4)` and `2 (c % 4) + 1` of high byte `c / 4`
            // above it: the high byte, spread to each of its four codes, multiplied into place.
            let high = _mm256_set1_epi64x(i64::from_le_bytes(self.qh));
            let lift = _mm256_set1_epi64x(256 | 64 << 16 | 16 << 32 | 4 << 48);
            let codes = |first: usize| {
                let low = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                    self.qs.as_ptr().add(16 * first) as *const __m128i
                ));
                // Word `c` of the half takes high byte `4 first + c / 4`: bytes `2 first + {0, 1}`
                // in the low 128 bits and `2 first + {2, 3}` in the high.
                let (a, b, c, d) = (
                    (4 * first) as i8,
                    (4 * first + 1) as i8,
                    (4 * first + 2) as i8,
                    (4 * first + 3) as i8,
                );
                let z = -128i8;
                let spread = _mm256_shuffle_epi8(
                    high,
                    _mm256_setr_epi8(
                        a, z, a, z, a, z, a, z, b, z, b, z, b, z, b, z, c, z, c, z, c, z, c, z, d,
                        z, d, z, d, z, d, z,
                    ),
                );
                let top =
                    _mm256_and_si256(_mm256_mullo_epi16(spread, lift), _mm256_set1_epi16(0x300));
                _mm256_or_si256(low, top)
            };
            let mut runs = Iq2SRuns {
                codes: [0; QK_K / 8],
                scales,
            };
            _mm256_storeu_si256(runs.codes.as_mut_ptr() as *mut __m256i, codes(0));
            _mm256_storeu_si256(runs.codes.as_mut_ptr().add(16) as *mut __m256i, codes(1));
            runs
        }
    }

    #[inline(always)]
    unsafe fn grid_run<const IB: usize>(
        &self,
        block: &Iq2SRuns,
    ) -> (
        std::arch::x86_64::__m256i,
        std::arch::x86_64::__m256i,
        std::arch::x86_64::__m256i,
    ) {
        unsafe {
            let mags =
                super::iq_grid_rows::table_u64_by_words::<IB>(&IQ2S_GRID, block.codes.as_ptr());
            let signs = self.qs.as_ptr().add(QK_K / 8) as *const u32;
            let signs = super::iq_grid_rows::broadcast_u32::<IB>(signs);
            (
                mags,
                super::iq_grid_rows::run_negate_spread(signs),
                super::iq_grid_rows::run_scale_lanes(block.scales, IB),
            )
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn dot_avx2(xs: &[BlockIq2S], ys: &[BlockQ8K32]) -> f32 {
    unsafe { super::iq_grid_rows::dot_grid(xs, ys) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn dot_many_avx2(xs: &[BlockIq2S], ys: &[&[BlockQ8K32]], out: &mut [f32]) {
    unsafe { super::iq_grid_rows::dot_many_grid(xs, ys, out) }
}
