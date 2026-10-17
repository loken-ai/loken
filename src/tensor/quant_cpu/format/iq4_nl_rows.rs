//! The iq4_nl dot against one to `MANY` activations at once: each block is decoded once and
//! dotted against every activation, the integer dots of eight blocks reduced together into one
//! vector per activation.
//!
//! Each activation's result is the block-by-block dot to the bit: per block, the exact integer
//! dot as a float, times the block's scale, times the activation block's scale, added to the
//! activation's total in block order.

#[cfg(target_arch = "x86_64")]
use super::*;

/// Blocks whose integer dots one reduction yields, one per float lane.
#[cfg(target_arch = "x86_64")]
const GROUP: usize = 8;

/// Block `bi`'s integer dot against activation `y`, as eight i32 lanes whose sum is the dot,
/// from the block's signed weights `w` and their magnitudes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn block_lanes(
    (w, magnitude): (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i),
    y: &BlockQ8_0,
) -> std::arch::x86_64::__m256i {
    unsafe {
        use std::arch::x86_64::*;
        let a = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
        // Unsigned times signed: the magnitudes against the activations carrying the sign.
        let pairs = _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(a, w));
        _mm256_madd_epi16(pairs, _mm256_set1_epi16(1))
    }
}

/// Blocks `k` and `k + 1` of a group interleaved, per activation: lane `2i` the sum of lanes
/// `2i` and `2i + 1` of the first block, lane `2i + 1` that of the second. Shifts and blends,
/// off the shuffle port.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pair_lanes<const N: usize>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]; N],
    base: usize,
    n: usize,
    k: usize,
) -> [std::arch::x86_64::__m256i; N] {
    unsafe {
        use std::arch::x86_64::*;
        let (b0, b1) = (base + k.min(n - 1), base + (k + 1).min(n - 1));
        let w0 = super::iq_read::iq4_run_weights(&xs[b0].qs);
        let w1 = super::iq_read::iq4_run_weights(&xs[b1].qs);
        let mut out = [_mm256_setzero_si256(); N];
        for t in 0..N {
            let (a, b) = (block_lanes(w0, &ys[t][b0]), block_lanes(w1, &ys[t][b1]));
            out[t] = _mm256_add_epi32(
                _mm256_blend_epi32::<0xAA>(a, b),
                _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(a), _mm256_slli_epi64::<32>(b)),
            );
        }
        out
    }
}

/// Blocks `k` to `k + 3` of a group, per activation: lane `j` of each 128-bit half the half's
/// sum for block `k + j`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn quad_lanes<const N: usize>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]; N],
    base: usize,
    n: usize,
    k: usize,
) -> [std::arch::x86_64::__m256i; N] {
    unsafe {
        use std::arch::x86_64::*;
        let a = pair_lanes(xs, ys, base, n, k);
        let b = pair_lanes(xs, ys, base, n, k + 2);
        let mut out = [_mm256_setzero_si256(); N];
        for t in 0..N {
            out[t] = _mm256_add_epi32(
                _mm256_unpacklo_epi64(a[t], b[t]),
                _mm256_unpackhi_epi64(a[t], b[t]),
            );
        }
        out
    }
}

/// The integer dots of blocks `base + k` against each activation, block `base + k` in lane
/// `k`. Lanes from `n` on repeat block `base + n - 1`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn group_dots<const N: usize>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]; N],
    base: usize,
    n: usize,
) -> [std::arch::x86_64::__m256i; N] {
    unsafe {
        use std::arch::x86_64::*;
        let lo = quad_lanes(xs, ys, base, n, 0);
        let hi = quad_lanes(xs, ys, base, n, 4);
        let mut out = [_mm256_setzero_si256(); N];
        for t in 0..N {
            out[t] = _mm256_add_epi32(
                _mm256_blend_epi32::<0xF0>(lo[t], hi[t]),
                _mm256_permute2x128_si256::<0x21>(lo[t], hi[t]),
            );
        }
        out
    }
}

/// Blocks `k` and `k + 1` of a group in 512-bit registers, per activation: sixteen i32 lanes,
/// the first eight summing to the first block's dot and the last eight to the second's, the
/// products summed by VNNI.
///
/// # Safety
/// The caller runs with AVX512F, AVX512BW, AVX512VL and AVX512VNNI enabled.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pair_lanes_avx512<const N: usize>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]; N],
    base: usize,
    n: usize,
    k: usize,
) -> [std::arch::x86_64::__m512i; N] {
    unsafe {
        use std::arch::x86_64::*;
        let table = _mm512_broadcast_i32x4(_mm_loadu_si128(
            super::iq4_nl::LEVELS.as_ptr() as *const __m128i
        ));
        let (b0, b1) = (base + k.min(n - 1), base + (k + 1).min(n - 1));
        let packed = _mm256_set_m128i(
            _mm_loadu_si128(xs[b1].qs.as_ptr() as *const __m128i),
            _mm_loadu_si128(xs[b0].qs.as_ptr() as *const __m128i),
        );
        // Each block's bytes twice: low nibbles in its first quarter, high in its second.
        let spread = _mm512_castsi256_si512(packed);
        let spread = _mm512_shuffle_i64x2::<0x50>(spread, spread);
        let nibbles = _mm512_and_si512(
            _mm512_mask_srli_epi16(spread, 0xFF00_FF00, spread, 4),
            _mm512_set1_epi8(0x0F),
        );
        let w = _mm512_shuffle_epi8(table, nibbles);
        let magnitude = _mm512_abs_epi8(w);
        let negative = _mm512_movepi8_mask(w);
        let mut out = [_mm512_setzero_si512(); N];
        for t in 0..N {
            let a = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_loadu_si256(ys[t][b0].qs.as_ptr() as *const __m256i)),
                _mm256_loadu_si256(ys[t][b1].qs.as_ptr() as *const __m256i),
            );
            // 0 - a under a negative weight: the byte negation of `_mm256_sign_epi8`.
            let signed = _mm512_mask_sub_epi8(a, negative, _mm512_setzero_si512(), a);
            out[t] = _mm512_dpbusd_epi32(_mm512_setzero_si512(), magnitude, signed);
        }
        out
    }
}

/// Pairs `k / 2` and `k / 2 + 1` of a group in 512-bit registers, per activation: within each
/// 128-bit quarter, lane 0 and 1 the quarter's partial sums for the first pair, lanes 2 and 3
/// for the second.
///
/// # Safety
/// As [`pair_lanes_avx512`].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn half_lanes_avx512<const N: usize>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]; N],
    base: usize,
    n: usize,
    k: usize,
) -> [std::arch::x86_64::__m512i; N] {
    unsafe {
        use std::arch::x86_64::*;
        let a = pair_lanes_avx512(xs, ys, base, n, k);
        let b = pair_lanes_avx512(xs, ys, base, n, k + 2);
        let mut out = [_mm512_setzero_si512(); N];
        for t in 0..N {
            out[t] = _mm512_add_epi32(
                _mm512_unpacklo_epi32(a[t], b[t]),
                _mm512_unpackhi_epi32(a[t], b[t]),
            );
        }
        out
    }
}

/// [`group_dots`] in 512-bit registers, two blocks to a register.
///
/// # Safety
/// As [`pair_lanes_avx512`].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn group_dots_avx512<const N: usize>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]; N],
    base: usize,
    n: usize,
) -> [std::arch::x86_64::__m256i; N] {
    unsafe {
        use std::arch::x86_64::*;
        let h0 = half_lanes_avx512(xs, ys, base, n, 0);
        let h1 = half_lanes_avx512(xs, ys, base, n, 4);
        let mut out = [_mm256_setzero_si256(); N];
        for t in 0..N {
            // Quarter `q`, lane `p`: the quarter's sum for pair `p`.
            let quarters = _mm512_add_epi32(
                _mm512_unpacklo_epi64(h0[t], h1[t]),
                _mm512_unpackhi_epi64(h0[t], h1[t]),
            );
            // Quarters 0 + 1 hold blocks 0, 2, 4, 6 and quarters 2 + 3 blocks 1, 3, 5, 7.
            let sums = _mm512_add_epi32(
                _mm512_shuffle_i64x2::<0x08>(quarters, quarters),
                _mm512_shuffle_i64x2::<0x0D>(quarters, quarters),
            );
            out[t] = _mm256_permutevar8x32_epi32(
                _mm512_castsi512_si256(sums),
                _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7),
            );
        }
        out
    }
}

/// The f16 scales `d[base + k]` as floats in lane `k`, lanes from `n` on repeating the last;
/// `d` points at the first block's scale and `stride` is the block size in bytes.
///
/// Four scales to a 64-bit word, two words to the vector: a vector load of eight separate
/// stores would stall, and lane inserts would load the shuffle port.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn group_scales(
    d: *const u8,
    stride: usize,
    base: usize,
    n: usize,
) -> std::arch::x86_64::__m256 {
    unsafe {
        use std::arch::x86_64::*;
        const _: () = assert!(GROUP == 8);
        let mut words = [0u64; 2];
        for (w, word) in words.iter_mut().enumerate() {
            for j in 0..4 {
                let bi = base + (4 * w + j).min(n - 1);
                let bits = (d.add(bi * stride) as *const u16).read_unaligned();
                *word |= (bits as u64) << (16 * j);
            }
        }
        _mm256_cvtph_ps(_mm_set_epi64x(words[1] as i64, words[0] as i64))
    }
}

/// Blocks `base..base + n` against every activation, added to their totals in block order.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn group<const WIDE: bool, const N: usize>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]; N],
    base: usize,
    n: usize,
    totals: &mut [f32; N],
) {
    unsafe {
        use std::arch::x86_64::*;
        let dots = if WIDE {
            group_dots_avx512(xs, ys, base, n)
        } else {
            group_dots(xs, ys, base, n)
        };
        let dx = group_scales(
            &xs[0].d as *const f16 as *const u8,
            std::mem::size_of::<BlockIq4Nl>(),
            base,
            n,
        );
        for t in 0..N {
            let dy = group_scales(
                &ys[t][0].d as *const f16 as *const u8,
                std::mem::size_of::<BlockQ8_0>(),
                base,
                n,
            );
            let terms = _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(dots[t]), dx), dy);
            let terms: [f32; GROUP] = std::mem::transmute(terms);
            for &term in &terms[..n] {
                totals[t] += term;
            }
        }
    }
}

/// `out[t] = dot(xs, ys[t])` for exactly `N` activations, in 512-bit registers when `WIDE`.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled, and AVX512F, AVX512BW, AVX512VL and
/// AVX512VNNI too when `WIDE`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn dot_rows<const WIDE: bool, const N: usize>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]],
    out: &mut [f32],
) {
    unsafe {
        let blocks = ys.iter().fold(xs.len(), |n, y| n.min(y.len()));
        let xs = &xs[..blocks];
        let ys: [&[BlockQ8_0]; N] = std::array::from_fn(|t| &ys[t][..blocks]);
        let mut totals = [0f32; N];
        let full = blocks / GROUP * GROUP;
        for base in (0..full).step_by(GROUP) {
            group::<WIDE, N>(xs, &ys, base, GROUP, &mut totals);
        }
        if full < blocks {
            group::<WIDE, N>(xs, &ys, full, blocks - full, &mut totals);
        }
        out[..N].copy_from_slice(&totals);
    }
}

/// `out[t] = dot(xs, ys[t])` for every activation, `MANY` at a time.
///
/// # Safety
/// As [`dot_rows`].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn dot_many<const WIDE: bool>(
    xs: &[BlockIq4Nl],
    ys: &[&[BlockQ8_0]],
    out: &mut [f32],
) {
    unsafe {
        let many = super::iq_read::MANY;
        for (ys, out) in ys.chunks(many).zip(out.chunks_mut(many)) {
            match ys.len() {
                1 => dot_rows::<WIDE, 1>(xs, ys, out),
                2 => dot_rows::<WIDE, 2>(xs, ys, out),
                3 => dot_rows::<WIDE, 3>(xs, ys, out),
                4 => dot_rows::<WIDE, 4>(xs, ys, out),
                5 => dot_rows::<WIDE, 5>(xs, ys, out),
                6 => dot_rows::<WIDE, 6>(xs, ys, out),
                7 => dot_rows::<WIDE, 7>(xs, ys, out),
                _ => dot_rows::<WIDE, 8>(xs, ys, out),
            }
        }
    }
}
