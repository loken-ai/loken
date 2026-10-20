//! iq3_s - 256 values in 110 bytes: nine-bit codes into a 512-entry grid of four magnitudes (the
//! ninth bit in a side array), an explicit sign bit per value, and a four-bit scale per thirty-two.

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

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        encode(xs, ys, None, 0);
    }

    fn quantize_guided(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        encode(xs, ys, Some(imatrix_weights), n_per_row);
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

/// Levels a grid coordinate takes: the value is `2 * level + 1` steps of the run's scale.
const LEVELS: u8 = 8;

pub(super) fn codebook() -> &'static super::iq_codebook::Codebook<4> {
    static BOOK: std::sync::OnceLock<super::iq_codebook::Codebook<4>> = std::sync::OnceLock::new();
    BOOK.get_or_init(|| {
        let bytes: Vec<[u8; 4]> = IQ3S_GRID.iter().map(|g| g.to_le_bytes()).collect();
        let levels = bytes.iter().map(|b| b.map(|v| (v - 1) / 2)).collect();
        super::iq_codebook::Codebook::new(bytes, levels, 3, LEVELS, 3)
    })
}

/// The encoder's codebook as one table, for a card that runs the same search.
pub fn iq3_s_codebook() -> Vec<i32> {
    codebook().table()
}

/// Quantise whole blocks, optionally against an importance matrix: one weight per column of a row
/// `n_per_row` long.
///
/// The search is ggml's `quantize_row_iq3_s_impl` (MIT). Each run of thirty-two values tries
/// nineteen scales around its largest magnitude; at each, every value rounds to a level, a group
/// of four whose levels the grid does not hold takes the nearest point it does, and the scale is
/// refitted by least squares. The best fit wins; when a group of it left the grid, every group is
/// placed again at the scale chosen. A value weighs its importance times its distance from the
/// block's spread, or its square without an importance.
fn encode(xs: &[f32], ys: &mut [BlockIq3S], imatrix: Option<&[f32]>, n_per_row: usize) {
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

/// `(sum w x q, sum w q q)` for levels `l`, `q = 2 l + 1`.
fn fit(w: &[f32], x: &[f32], l: &[u8]) -> (f32, f32) {
    let (mut qx, mut q2) = (0f32, 0f32);
    for i in 0..w.len() {
        let q = (2 * l[i] as i32 + 1) as f32;
        qx += w[i] * x[i] * q;
        q2 += w[i] * q * q;
    }
    (qx, q2)
}

fn encode_block(
    book: &super::iq_codebook::Codebook<4>,
    x: &[f32],
    row: Option<&[f32]>,
    block: &mut BlockIq3S,
) {
    /// Scales tried either side of the largest magnitude's own, a fifth of a level apart.
    const STEPS: i32 = 9;
    const STEP: f32 = 0.2;
    /// The largest level's value, in steps of the scale.
    const TOP: f32 = (2 * LEVELS - 1) as f32;
    /// Run scales are four bits; the block scale is the largest over this many steps.
    const SCALE_STEPS: f32 = 31.0;
    /// The block scale is stored larger than the step the runs were quantised against.
    const SCALE_LIFT: f32 = 1.033;
    const RUN: usize = 32;
    const RUNS: usize = QK_K / RUN;

    *block = BlockIq3S::zeros();
    let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;
    let mut scales = [0f32; RUNS];
    for run in 0..RUNS {
        let xb = &x[RUN * run..RUN * run + RUN];
        let weight: [f32; RUN] = std::array::from_fn(|i| match row {
            Some(r) => r[RUN * run + i] * (sigma2 + xb[i] * xb[i]).sqrt(),
            None => xb[i] * xb[i],
        });
        let root = weight.map(f32::sqrt);
        let xval = xb.iter().map(|v| v.abs()).collect::<Vec<_>>();
        let mut signs = [0u8; RUN / 8];
        for (k, s) in signs.iter_mut().enumerate() {
            for i in 0..8 {
                if xb[8 * k + i] < 0.0 {
                    *s |= 1 << i;
                }
            }
        }
        let max = xval.iter().copied().fold(xval[0], f32::max);
        if max == 0.0 {
            continue;
        }

        let mut best = 0f32;
        let mut scale = max / TOP;
        let mut l = [0u8; RUN];
        let mut on_grid = [false; RUN / 4];
        for step in -STEPS..=STEPS {
            let inverse = (TOP + step as f32 * STEP) / max;
            let this = 1.0 / inverse;
            let mut trial = [0u8; RUN];
            let mut held = [true; RUN / 4];
            for k in 0..RUN / 4 {
                let at = 4 * k..4 * k + 4;
                let (n, h) = book.place(
                    &xval[at.clone()],
                    &root[at.clone()],
                    (inverse, this),
                    LEVELS - 1,
                );
                held[k] = h;
                trial[at].copy_from_slice(&book.levels[n]);
            }
            let (qx, q2) = fit(&weight, &xval, &trial);
            if q2 > 0.0 && qx * qx > best * q2 {
                scale = qx / q2;
                best = scale * qx;
                l = trial;
                on_grid = held;
            }
        }

        // Some group left the grid at the scale chosen: place every group again at it, and refit.
        if on_grid.contains(&false) && scale > 0.0 {
            let inverse = 1.0 / scale;
            for k in 0..RUN / 4 {
                let at = 4 * k..4 * k + 4;
                let (n, _) = book.place(
                    &xval[at.clone()],
                    &root[at.clone()],
                    (inverse, scale),
                    LEVELS - 1,
                );
                l[at].copy_from_slice(&book.levels[n]);
            }
            let (qx, q2) = fit(&weight, &xval, &l);
            if q2 > 0.0 {
                scale = qx / q2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in signs.iter_mut() {
                *s = !*s;
            }
        }

        for k in 0..RUN / 4 {
            let n = book.index[book.pattern(&l[4 * k..4 * k + 4])] as usize;
            block.qs[8 * run + k] = n as u8;
            block.qh[run] |= ((n >> 8) as u8) << k;
        }
        block.signs[4 * run..4 * run + 4].copy_from_slice(&signs);
        scales[run] = scale;
    }

    let max_scale = scales.iter().copied().fold(0f32, f32::max);
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / SCALE_STEPS;
    block.d = f16::from_f32(d * SCALE_LIFT);
    let inverse = 1.0 / d;
    for (pair, out) in block.scales.iter_mut().enumerate() {
        let level = |s: f32| nearest_int(0.5 * (inverse * s - 1.0)).clamp(0, 15) as u8;
        *out = level(scales[2 * pair]) | level(scales[2 * pair + 1]) << 4;
    }
}
