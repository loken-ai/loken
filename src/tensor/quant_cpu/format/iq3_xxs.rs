//! iq3_xxs - 256 values in 98 bytes: codes into a 256-entry grid of four magnitudes, a 7-bit sign
//! pattern per eight values, and a four-bit scale per thirty-two.

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

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        encode(xs, ys, None, 0);
    }

    fn quantize_guided(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        encode(xs, ys, Some(imatrix_weights), n_per_row);
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

/// Levels a grid coordinate takes: the value is `2 * level + 1` steps of the sub-block scale.
const LEVELS: i32 = 8;

/// What the encoder needs of the grid: each point's four levels, the point a pattern of four levels
/// names when the grid holds it, and for every pattern it does not hold, the points nearest it.
struct Codebook {
    levels: [[u8; 4]; 256],
    index: Vec<i16>,
    near: Vec<Vec<u8>>,
}

/// Four levels as one index, three bits each.
fn pattern(levels: &[u8]) -> usize {
    levels
        .iter()
        .enumerate()
        .fold(0, |p, (i, &l)| p | (l as usize) << (3 * i))
}

fn codebook() -> &'static Codebook {
    static BOOK: std::sync::OnceLock<Codebook> = std::sync::OnceLock::new();
    BOOK.get_or_init(|| {
        // A coordinate's byte is 4 * (2 * level + 1), the top level rounded up to 62.
        let levels: [[u8; 4]; 256] =
            std::array::from_fn(|n| IQ3XXS_GRID[n].to_le_bytes().map(|b| (b - 4) / 8));
        let patterns = 1usize << (3 * 4);
        let mut index = vec![-1i16; patterns];
        for (n, l) in levels.iter().enumerate() {
            index[pattern(l)] = n as i16;
        }
        // A pattern off the grid is searched among the points at the two least distances from it.
        let near = (0..patterns)
            .map(|p| {
                if index[p] >= 0 {
                    return Vec::new();
                }
                let at: [i32; 4] = std::array::from_fn(|i| ((p >> (3 * i)) & 7) as i32);
                let mut by_distance: Vec<(i32, u8)> = levels
                    .iter()
                    .enumerate()
                    .map(|(n, l)| {
                        let d = (0..4).map(|i| (l[i] as i32 - at[i]).pow(2)).sum();
                        (d, n as u8)
                    })
                    .collect();
                by_distance.sort_unstable();
                let mut distinct = 0;
                let mut last = -1;
                by_distance
                    .into_iter()
                    .take_while(|&(d, _)| {
                        if d != last {
                            distinct += 1;
                            last = d;
                        }
                        distinct <= 2
                    })
                    .map(|(_, n)| n)
                    .collect()
            })
            .collect();
        Codebook {
            levels,
            index,
            near,
        }
    })
}

/// The encoder's codebook as one table, for a card that runs the same search: the point each
/// pattern of four levels names or -1, an offset per pattern into the lists of near points and one
/// past the last, each point's four levels, each point's four stored magnitudes, then the lists.
pub fn iq3_xxs_codebook() -> Vec<i32> {
    let book = codebook();
    let mut out: Vec<i32> = book.index.iter().map(|&i| i as i32).collect();
    let mut at = 0i32;
    for near in &book.near {
        out.push(at);
        at += near.len() as i32;
    }
    out.push(at);
    out.extend(book.levels.iter().flatten().map(|&l| l as i32));
    out.extend(
        IQ3XXS_GRID
            .iter()
            .flat_map(|g| g.to_le_bytes())
            .map(|b| b as i32),
    );
    out.extend(book.near.iter().flatten().map(|&n| n as i32));
    out
}

/// Quantise whole blocks, optionally against an importance matrix: one weight per column of a row
/// `n_per_row` long.
///
/// The search is ggml's `quantize_row_iq3_xxs_impl` (MIT). Each sub-block of thirty-two values
/// tries thirty-one scales around its largest magnitude; at each, every value rounds to a level,
/// a group of four whose levels the grid does not hold takes the nearest point that it does, and
/// the scale is refitted by least squares. The scale with the best fit wins, and the off-grid
/// groups are placed again at it. A value weighs its importance times its distance from the
/// block's spread, or its square without an importance.
fn encode(xs: &[f32], ys: &mut [BlockIq3Xxs], imatrix: Option<&[f32]>, n_per_row: usize) {
    use rayon::prelude::*;
    let book = codebook();
    ys.par_iter_mut()
        .zip(xs.par_chunks_exact(QK_K))
        .enumerate()
        .for_each(|(idx, (block, x))| {
            let row = imatrix.map(|m| {
                let base = (idx % (n_per_row / QK_K)) * QK_K;
                &m[base..base + QK_K]
            });
            encode_block(book, x, row, block);
        });
}

/// The point among `candidates` that leaves the group `x` the least error at `scale`, each value's
/// error weighed by `w`.
fn nearest_point(candidates: &[u8], book: &Codebook, x: &[f32], w: &[f32], scale: f32) -> usize {
    let (mut at, mut best) = (candidates[0] as usize, f32::INFINITY);
    for &n in candidates {
        let l = &book.levels[n as usize];
        let err: f32 = (0..4)
            .map(|i| {
                let d = scale * (2 * l[i] as i32 + 1) as f32 - x[i];
                w[i] * d * d
            })
            .sum();
        if err < best {
            (at, best) = (n as usize, err);
        }
    }
    at
}

/// The four levels nearest `x` at `inverse` (one over the scale), and whether the grid holds them.
fn round_group(x: &[f32], inverse: f32) -> [u8; 4] {
    std::array::from_fn(|i| nearest_int(0.5 * (inverse * x[i] - 1.0)).clamp(0, LEVELS - 1) as u8)
}

/// The least-squares fit of levels `l` to `x` under weights `w`: `(sum w x q, sum w q q)` with
/// `q = 2 l + 1`.
fn fit(w: &[f32; 32], x: &[f32; 32], l: &[u8; 32]) -> (f32, f32) {
    let (mut qx, mut q2) = (0f32, 0f32);
    for i in 0..32 {
        let q = (2 * l[i] as i32 + 1) as f32;
        qx += w[i] * x[i] * q;
        q2 += w[i] * q * q;
    }
    (qx, q2)
}

fn encode_block(book: &Codebook, x: &[f32], row: Option<&[f32]>, block: &mut BlockIq3Xxs) {
    /// The scale search: steps either side of the largest magnitude's own scale, a fifth of a
    /// level apart.
    const STEPS: i32 = 15;
    const STEP: f32 = 0.2;
    /// The largest level's value, in steps of the scale.
    const TOP: f32 = (2 * LEVELS - 1) as f32;
    /// A sub-block below this magnitude is stored as zeros.
    const EMPTY: f32 = 1e-8;
    /// Sub-block scales are four bits; the block scale is the largest over this many steps.
    const SCALE_STEPS: f32 = 31.0;
    /// The block scale is stored slightly larger than the step it was quantised against, which
    /// lowers the error of the rounded sub-block scales.
    const SCALE_LIFT: f32 = 1.0125;
    const SUBS: usize = QK_K / 32;

    let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;
    let mut scales = [0f32; SUBS];
    let mut words = [0u32; SUBS];
    block.qs.fill(0);
    for sub in 0..SUBS {
        let xb = &x[32 * sub..32 * sub + 32];
        let weight: [f32; 32] = std::array::from_fn(|i| match row {
            Some(r) => r[32 * sub + i] * (sigma2 + xb[i] * xb[i]).sqrt(),
            None => xb[i] * xb[i],
        });
        let root = weight.map(f32::sqrt);

        // Magnitudes, with each group of eight given an even count of negatives: the format stores
        // seven signs and the eighth by parity, so an odd group flips the value that costs least.
        let mut xval = [0f32; 32];
        let mut signs = [0u32; 4];
        for (k, sign) in signs.iter_mut().enumerate() {
            let mut s = 0u32;
            for i in 0..8 {
                let v = xb[8 * k + i];
                xval[8 * k + i] = v.abs();
                if v < 0.0 {
                    s |= 1 << i;
                }
            }
            if s.count_ones() % 2 == 1 {
                let cost = |i: usize| weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                let cheapest = (0..8).min_by(|&a, &b| cost(a).total_cmp(&cost(b))).unwrap();
                xval[8 * k + cheapest] = -xval[8 * k + cheapest];
                s ^= 1 << cheapest;
            }
            *sign = s & 127;
        }
        let max = xval.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max < EMPTY {
            continue;
        }

        let mut best = 0f32;
        let mut scale = max / TOP;
        let mut l = [0u8; 32];
        let mut on_grid = [true; 8];
        for step in -STEPS..=STEPS {
            let inverse = (TOP + step as f32 * STEP) / max;
            let mut trial = [0u8; 32];
            let mut held = [true; 8];
            for k in 0..8 {
                let group = round_group(&xval[4 * k..4 * k + 4], inverse);
                let p = pattern(&group);
                let n = if book.index[p] >= 0 {
                    book.index[p] as usize
                } else {
                    held[k] = false;
                    nearest_point(
                        &book.near[p],
                        book,
                        &xval[4 * k..4 * k + 4],
                        &root[4 * k..4 * k + 4],
                        1.0 / inverse,
                    )
                };
                trial[4 * k..4 * k + 4].copy_from_slice(&book.levels[n]);
            }
            let (qx, q2) = fit(&weight, &xval, &trial);
            if q2 > 0.0 && qx * qx > best * q2 {
                scale = qx / q2;
                best = scale * qx;
                l = trial;
                on_grid = held;
            }
        }

        // The groups that left the grid were placed for their trial scale; place them again for
        // the one chosen, and refit.
        if on_grid.contains(&false) && scale > 0.0 {
            let inverse = 1.0 / scale;
            for k in (0..8).filter(|&k| !on_grid[k]) {
                let group = round_group(&xval[4 * k..4 * k + 4], inverse);
                let p = pattern(&group);
                let n = if book.index[p] >= 0 {
                    book.index[p] as usize
                } else {
                    nearest_point(
                        &book.near[p],
                        book,
                        &xval[4 * k..4 * k + 4],
                        &root[4 * k..4 * k + 4],
                        scale,
                    )
                };
                l[4 * k..4 * k + 4].copy_from_slice(&book.levels[n]);
            }
            let (qx, q2) = fit(&weight, &xval, &l);
            if q2 > 0.0 {
                scale = qx / q2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in signs.iter_mut() {
                *s = !*s & 127;
            }
        }

        for k in 0..8 {
            block.qs[8 * sub + k] = book.index[pattern(&l[4 * k..4 * k + 4])] as u8;
        }
        words[sub] = signs[0] | signs[1] << 7 | signs[2] << 14 | signs[3] << 21;
        scales[sub] = scale;
    }

    let max_scale = scales.iter().copied().fold(0f32, f32::max);
    if max_scale == 0.0 {
        block.d = f16::ZERO;
        block.qs.fill(0);
        return;
    }
    let d = max_scale / SCALE_STEPS;
    block.d = f16::from_f32(d * SCALE_LIFT);
    for (sub, word) in words.iter_mut().enumerate() {
        let level = nearest_int(0.5 * (scales[sub] / d - 1.0)).clamp(0, 15) as u32;
        *word |= level << 28;
        block.qs[QK_K / 4 + 4 * sub..QK_K / 4 + 4 * sub + 4].copy_from_slice(&word.to_le_bytes());
    }
}

#[cfg(test)]
mod encode_tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        let path = format!(
            "{}/tests/vectors/iq3xxs_encode.json",
            env!("CARGO_MANIFEST_DIR")
        );
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn floats(v: &serde_json::Value, key: &str) -> Vec<f32> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    /// Every grid point is a distinct pattern of four levels, so the encoder's codes name exactly
    /// the points it chose.
    #[test]
    fn every_grid_point_is_its_own_pattern() {
        let book = codebook();
        for (n, l) in book.levels.iter().enumerate() {
            assert_eq!(book.index[pattern(l)], n as i16);
        }
        assert_eq!(book.index.iter().filter(|&&i| i >= 0).count(), 256);
        assert!(book
            .index
            .iter()
            .zip(&book.near)
            .all(|(&i, near)| (i >= 0) == near.is_empty()));
    }

    /// The blocks ggml wrote for the fixture's rows read back as gguf-py reads them, and the
    /// blocks this encoder writes keep the format's invariants: four-bit scales, codes inside the
    /// grid, a scale that reconstructs the row.
    #[test]
    fn blocks_read_back() {
        let v = fixture();
        let hex = v["blocks"].as_str().unwrap();
        let bytes: Vec<u8> = (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        let expected = floats(&v, "expected");
        let mut got = vec![0f32; expected.len()];
        crate::tensor::quant_cpu::to_float_bytes(GgmlDType::Iq3Xxs, &bytes, &mut got).unwrap();
        assert_eq!(got, expected);

        let x = floats(&v, "x");
        let ours = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Iq3Xxs, &x).unwrap();
        assert_eq!(ours.len(), bytes.len());
        let mut back = vec![0f32; x.len()];
        crate::tensor::quant_cpu::to_float_bytes(GgmlDType::Iq3Xxs, &ours, &mut back).unwrap();
        let (num, den) = x.iter().zip(&back).fold((0f64, 0f64), |(n, d), (&a, &b)| {
            (n + (a as f64 - b as f64).powi(2), d + (a as f64).powi(2))
        });
        assert!(
            (num / den).sqrt() < 0.4,
            "the blocks do not read back as the row"
        );
    }

    /// What the encoder leaves against what ggml's leaves on the same rows and importance.
    #[test]
    fn encodes_as_well_as_ggml() {
        let v = fixture();
        let cols = v["cols"].as_u64().unwrap() as usize;
        let (x, importance) = (floats(&v, "x"), floats(&v, "importance"));
        let weighted = |got: &[f32], w: &dyn Fn(usize) -> f32| {
            let (mut num, mut den) = (0f64, 0f64);
            for (i, (g, t)) in got.iter().zip(&x).enumerate() {
                let wi = w(i % cols) as f64;
                num += wi * (*g as f64 - *t as f64).powi(2);
                den += wi * (*t as f64).powi(2);
            }
            (num / den).sqrt() as f32
        };
        let mut blocks = vec![BlockIq3Xxs::zeros(); x.len() / QK_K];
        let mut got = vec![0f32; x.len()];

        BlockIq3Xxs::quantize_guided(&x, &mut blocks, &importance, cols);
        BlockIq3Xxs::dequantize(&blocks, &mut got);
        let ours = weighted(&got, &|i| importance[i]);
        let theirs = v["weighted_error"].as_f64().unwrap() as f32;
        eprintln!("iq3_xxs importance-guided: {ours} against ggml's {theirs}");
        assert!(
            ours <= theirs * 1.02,
            "importance-guided: {ours} against ggml's {theirs}"
        );

        BlockIq3Xxs::quantize(&x, &mut blocks);
        BlockIq3Xxs::dequantize(&blocks, &mut got);
        let ours = weighted(&got, &|_| 1.0);
        let theirs = v["unweighted_error"].as_f64().unwrap() as f32;
        eprintln!("iq3_xxs plain: {ours} against ggml's {theirs}");
        assert!(
            ours <= theirs * 1.02,
            "plain: {ours} against ggml's {theirs}"
        );
    }
}
