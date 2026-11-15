//! The exact dots of the run formats in AVX2: the steps of [`ExactDot`] for a format that reads
//! q8_K32 blocks.
//!
//! Each run's integer sum is computed in eight i32 lanes and the lanes of the block's runs are
//! reduced together, run `r` in lane `r`.
//! The fold is the scalar one: each run's `(ws * s) * da` in its own float lane, then the terms
//! added from -0 in run order into the block's share, and the shares added in block order.
//! Integer sums are exact in any order, so the result is the scalar dot's to the bit.

use super::iq_grids::IQ3S_GRID;
use super::q8_k32::Q8K32_RUNS;
use super::*;
use std::arch::x86_64::*;

/// The sums of the eight i32 lanes of each of `v`, the sum of `v[r]` in lane `r`.
#[inline(always)]
unsafe fn run_sums(v: [__m256i; Q8K32_RUNS]) -> __m256i {
    // Within each 128-bit half: lane `j` of `quad` the half's sum of `v[j]` (and of `v[4 + j]`).
    let low = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
    let high = _mm256_hadd_epi32(_mm256_hadd_epi32(v[4], v[5]), _mm256_hadd_epi32(v[6], v[7]));
    _mm256_add_epi32(
        _mm256_permute2x128_si256::<0x20>(low, high),
        _mm256_permute2x128_si256::<0x31>(low, high),
    )
}

/// A block's share: the fold from -0, in run order, of the first `runs` terms
/// `(ws[r] * s[r]) * da[r]`, with `s` the run sums and `da` the activation's run scales.
#[inline(always)]
unsafe fn block_share(ws: __m256, sums: __m256i, da: &[f16; Q8K32_RUNS], runs: usize) -> f32 {
    let da = _mm256_cvtph_ps(_mm_loadu_si128(da.as_ptr() as *const __m128i));
    let terms = _mm256_mul_ps(_mm256_mul_ps(ws, _mm256_cvtepi32_ps(sums)), da);
    let mut t = [0f32; Q8K32_RUNS];
    _mm256_storeu_ps(t.as_mut_ptr(), terms);
    let mut share = -0.0f32;
    for &term in &t[..runs] {
        share += term;
    }
    share
}

/// Eight i32 lanes whose sum is the dot of the signed bytes `w` and `a`.
#[inline(always)]
unsafe fn signed_lanes(w: __m256i, a: __m256i) -> __m256i {
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w)),
        _mm256_set1_epi16(1),
    )
}

/// The 32 activation codes of run `r` of `y`.
#[inline(always)]
unsafe fn run_acts(y: &BlockQ8K32, r: usize) -> __m256i {
    _mm256_loadu_si256(y.qs.as_ptr().add(32 * r) as *const __m256i)
}

/// Run `IB` of grid block `x` as eight i32 lanes whose sum is the run's integer sum.
#[inline(always)]
unsafe fn grid_lanes<B: super::iq_grid_rows::GridRun, const IB: usize>(
    x: &B,
    block: &B::Block,
    y: &BlockQ8K32,
) -> __m256i {
    let (mags, negate, scale) = x.grid_run::<IB>(block);
    let pairs = _mm256_maddubs_epi16(mags, _mm256_sign_epi8(run_acts(y, IB), negate));
    _mm256_madd_epi16(pairs, scale)
}

/// The exact dot of a grid format.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn dot_grid<B: super::iq_grid_rows::GridRun>(xs: &[B], ys: &[BlockQ8K32]) -> f32 {
    const _: () = assert!(Q8K32_RUNS == 8);
    let mut total = -0.0f32;
    for (x, y) in xs.iter().zip(ys) {
        let block = x.block();
        let sums = run_sums([
            grid_lanes::<B, 0>(x, &block, y),
            grid_lanes::<B, 1>(x, &block, y),
            grid_lanes::<B, 2>(x, &block, y),
            grid_lanes::<B, 3>(x, &block, y),
            grid_lanes::<B, 4>(x, &block, y),
            grid_lanes::<B, 5>(x, &block, y),
            grid_lanes::<B, 6>(x, &block, y),
            grid_lanes::<B, 7>(x, &block, y),
        ]);
        total += block_share(_mm256_set1_ps(x.run_unit()), sums, &y.d, Q8K32_RUNS);
    }
    total
}

/// The exact dot of IQ3_S.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn dot_iq3_s(xs: &[BlockIq3S], ys: &[BlockQ8K32]) -> f32 {
    let mut total = -0.0f32;
    for (x, y) in xs.iter().zip(ys) {
        let mut lanes = [_mm256_setzero_si256(); Q8K32_RUNS];
        for (ib, lane) in lanes.iter_mut().enumerate() {
            // Code `c` of a run is its byte with bit `c` of the run's high-bit byte as the ninth
            // bit.
            let codes = u64::from_le_bytes(x.qs[8 * ib..8 * ib + 8].try_into().unwrap());
            let high = x.qh[ib] as u32;
            let grid = |l: u32| {
                IQ3S_GRID[(((codes >> (8 * l)) as u32 & 0xFF) | (((high >> l) & 1) << 8)) as usize]
                    as i32
            };
            let pair = |l: u32| (grid(l + 1) as u32 as i64) << 32 | grid(l) as u32 as i64;
            let mags = _mm256_set_m128i(
                _mm_set_epi64x(pair(6), pair(4)),
                _mm_set_epi64x(pair(2), pair(0)),
            );
            let signs = u32::from_le_bytes(x.signs[4 * ib..4 * ib + 4].try_into().unwrap());
            let pairs = super::iq_read::grid_run_pairs(mags, signs, &y.qs[32 * ib..]);
            let nibble = (x.scales[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            *lane = _mm256_madd_epi16(pairs, _mm256_set1_epi16(2 * nibble as i16 + 1));
        }
        total += block_share(
            _mm256_set1_ps(x.d.to_f32()),
            run_sums(lanes),
            &y.d,
            Q8K32_RUNS,
        );
    }
    total
}

/// The exact dot of a format of 32-value blocks, block `r` of each 8 against run `r` of an
/// activation block, fewer in a row's last block: `scale` gives a block's scale, `weights` its
/// 32 signed weights.
#[inline(always)]
unsafe fn dot_blocks32<X>(
    xs: &[X],
    ys: &[BlockQ8K32],
    scale: impl Fn(&X) -> f16,
    weights: impl Fn(&X) -> __m256i,
) -> f32 {
    let mut total = -0.0f32;
    for (x8, y) in xs.chunks(Q8K32_RUNS).zip(ys) {
        let mut lanes = [_mm256_setzero_si256(); Q8K32_RUNS];
        let mut ws = [f16::ZERO; Q8K32_RUNS];
        for (r, x) in x8.iter().enumerate() {
            lanes[r] = signed_lanes(weights(x), run_acts(y, r));
            ws[r] = scale(x);
        }
        let ws = _mm256_cvtph_ps(_mm_loadu_si128(ws.as_ptr() as *const __m128i));
        total += block_share(ws, run_sums(lanes), &y.d, x8.len());
    }
    total
}

/// The exact dot of IQ4_NL.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn dot_iq4_nl(xs: &[BlockIq4Nl], ys: &[BlockQ8K32]) -> f32 {
    dot_blocks32(
        xs,
        ys,
        |x| x.d,
        |x| super::iq_read::iq4_run_weights(&x.qs).0,
    )
}

/// The exact dot of Q8_0.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn dot_q8_0(xs: &[BlockQ8_0], ys: &[BlockQ8K32]) -> f32 {
    dot_blocks32(
        xs,
        ys,
        |x| x.d,
        |x| _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i),
    )
}

/// The exact dot of Q5_0.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn dot_q5_0(xs: &[BlockQ5_0], ys: &[BlockQ8K32]) -> f32 {
    use crate::tensor::quant_cpu::avx::{mask_from_bits, split_nibbles};
    // A 5-bit code less its bias of 16 is its nibble with 0xF0 above it where the fifth bit is
    // clear.
    let fill = _mm256_set1_epi8(0xF0u8 as i8);
    dot_blocks32(
        xs,
        ys,
        |x| x.d,
        |x| {
            _mm256_or_si256(
                split_nibbles(x.qs.as_ptr()),
                _mm256_andnot_si256(mask_from_bits(&x.qh), fill),
            )
        },
    )
}

/// The exact dot of Q2_0: runs `2b` and `2b + 1` of an activation block are the halves of
/// block `b` of its 4, fewer in a row's last block.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn dot_q2_0(xs: &[BlockQ2_0], ys: &[BlockQ8K32]) -> f32 {
    let (ones8, ones16) = (_mm256_set1_epi8(1), _mm256_set1_epi16(1));
    let mut total = -0.0f32;
    for (x4, y) in xs.chunks(QK_K / QK2_0).zip(ys) {
        let mut lanes = [_mm256_setzero_si256(); Q8K32_RUNS];
        let mut ws = [f16::ZERO; Q8K32_RUNS];
        for (b, x) in x4.iter().enumerate() {
            let codes = super::q2_0::unpack_codes(x);
            for h in 0..2 {
                let r = 2 * b + h;
                let a = run_acts(y, r);
                // The codes 0 to 3 against the activations, less the activations' own sum: the
                // levels `code - 1`.
                let coded = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_set_m128i(codes[2 * h + 1], codes[2 * h]), a),
                    ones16,
                );
                let offset = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, a), ones16);
                lanes[r] = _mm256_sub_epi32(coded, offset);
                ws[r] = x.d;
            }
        }
        let ws = _mm256_cvtph_ps(_mm_loadu_si128(ws.as_ptr() as *const __m128i));
        total += block_share(ws, run_sums(lanes), &y.d, 2 * x4.len());
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` blocks of `T` from random bytes, each block's leading f16 scale a small finite value.
    fn blocks<T: BlockFormat>(n: usize, seed: u64) -> Vec<T> {
        let size = std::mem::size_of::<T>();
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        (0..n)
            .map(|_| {
                let mut bytes: Vec<u8> = (0..size).map(|_| next() as u8).collect();
                let d = f16::from_f32((next() % 2001) as f32 / 1000.0 - 1.0) * f16::from_f32(0.01);
                bytes[..2].copy_from_slice(&d.to_le_bytes());
                // Safety: every format here is plain bytes after its scale.
                unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const T) }
            })
            .collect()
    }

    /// A row of `len` values as q8_K32 blocks, the last block padded with zeros.
    fn activations(len: usize, seed: u64) -> Vec<BlockQ8K32> {
        let mut x: Vec<f32> = (0..len)
            .map(|i| (((i as u64 * 2_654_435_761 + seed) % 2001) as f32 / 1000.0 - 1.0) * 3.0)
            .collect();
        x.resize(len.div_ceil(QK_K) * QK_K, 0.0);
        let mut y = vec![BlockQ8K32::zeros(); x.len() / QK_K];
        BlockQ8K32::quantize(&x, &mut y);
        y
    }

    /// The vector dot of `T` against its scalar steps on random rows of each width, to the bit.
    fn same_bits<T: ExactDot<Act = BlockQ8K32>>(
        widths: &[usize],
        scalar: fn(&[T], &[BlockQ8K32]) -> f32,
    ) {
        if !crate::cpu::dispatch_avx2() {
            return;
        }
        for &len in widths {
            for seed in 0..64u64 {
                let x = blocks::<T>(len / T::BLOCK_LEN, seed * 7 + len as u64);
                let y = activations(len, seed);
                let (fast, slow) = (T::dot_exact(&x, &y), scalar(&x, &y));
                assert_eq!(
                    fast.to_bits(),
                    slow.to_bits(),
                    "{:?} at {len}, seed {seed}: {fast} against {slow}",
                    T::DTYPE
                );
            }
        }
    }

    /// Widths of whole activation blocks, and of a last block holding 4 runs of its 8.
    const RUN_WIDTHS: [usize; 2] = [2560, 640];
    /// Widths of whole activation blocks, for formats of 256-value blocks.
    const BLOCK_WIDTHS: [usize; 2] = [2560, 512];

    #[test]
    fn the_vector_exact_dots_give_the_scalar_bits() {
        same_bits::<BlockIq2S>(&BLOCK_WIDTHS, super::super::iq2_s::dot_exact_scalar);
        same_bits::<BlockIq3Xxs>(&BLOCK_WIDTHS, super::super::iq3_xxs::dot_exact_scalar);
        same_bits::<BlockIq3S>(&BLOCK_WIDTHS, super::super::iq3_s::dot_exact_scalar);
        same_bits::<BlockIq4Nl>(&RUN_WIDTHS, super::super::iq4_nl::dot_exact_scalar);
        same_bits::<BlockQ2_0>(&RUN_WIDTHS, super::super::q2_0::dot_exact_scalar);
        same_bits::<BlockQ8_0>(&RUN_WIDTHS, super::super::q8_0::dot_exact_scalar);
        same_bits::<BlockQ5_0>(&RUN_WIDTHS, super::super::q5_0::dot_exact_scalar);
    }
}
