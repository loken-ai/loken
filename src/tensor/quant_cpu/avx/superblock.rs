//! AVX2 dot products for the k-quants: 256 values under one pair of block
//! scales, with per-sub-block scales quantised in turn.
//!
//! Two things separate these from the 32-value formats. The codes no longer fit
//! one register, so each kernel walks its super-block in halves or quarters;
//! and the scale is no longer constant over the block, so the products have to
//! meet a per-sub-block scale before they are summed. That second step is the
//! shape of every kernel here: `maddubs` leaves i16 pairs, `madd` against the
//! spread scale both applies it and widens to i32, and only the block scale is
//! left for the end.
//!
//! The offset formats (q2_K, q4_K, q5_K) never subtract their minimum from a
//! code. `m` is constant over a sub-block, so `Σ m.a` over that sub-block is
//! `m` times a sum the activation already carries - q8_K's `bsums`, one i16 per
//! sixteen values. The whole minimum term is therefore one `madd` per block,
//! outside the loop. q6_K does the same with its fixed -32 zero point.
//!
//! Bit shifts are written as literals throughout. The intrinsics require an
//! immediate, and the alternative - a shift count in a register - costs a port
//! in the innermost loop of the decode path.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{
    join_halves, lanes_sum, scale_over_16_pairs, scale_pair_over_8_bytes, scale_pair_over_8_pairs,
};
use crate::tensor::quant_cpu::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K};

const LOW4: u32 = 0x0f0f_0f0f;
const HIGH2: u32 = 0x0303_0303;

/// Four bytes of `packed`, little-endian, as one word.
#[inline(always)]
fn word(packed: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([packed[i], packed[i + 1], packed[i + 2], packed[i + 3]])
}

/// A six-bit field split across two bytes: four bits in place, two more waiting
/// in a spare nibble.
#[inline(always)]
fn rejoin(nibble: u32, spare: u32) -> u32 {
    nibble | ((spare & HIGH2) << 4)
}

/// The sixteen i16 scales of a 16-value-sub-block format, as two registers each
/// holding one half twice - the form `shuffle_epi8` needs, which selects within
/// a 128-bit lane and so must see the same eight scales in both.
#[inline(always)]
unsafe fn scales_by_half(bytes: __m128i) -> [__m256i; 2] {
    let widened = _mm256_cvtepi8_epi16(bytes);
    let low = _mm256_extracti128_si256(widened, 0);
    let high = _mm256_extracti128_si256(widened, 1);
    [join_halves(low, low), join_halves(high, high)]
}

/// A sub-block's products, scaled and widened to i32.
#[inline(always)]
unsafe fn scaled(scales: __m256i, quarter: usize, products: __m256i) -> __m256i {
    _mm256_madd_epi16(
        _mm256_shuffle_epi8(scales, scale_pair_over_8_pairs(quarter)),
        products,
    )
}

/// Two bits per value, sixteen to a sub-block, with a 4-bit scale and a 4-bit
/// minimum sharing one byte. Both halves of the super-block read the same
/// 32 bytes of codes at shifts advancing by two.
#[inline(always)]
pub(crate) fn vec_dot_q2k_q8k(xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    unsafe {
        let two_bits = _mm256_set1_epi8(3);
        let nibble = _mm_set1_epi8(0xF);
        let mut acc = _mm256_setzero_ps();

        for (x, y) in xs.iter().zip(ys) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();
            let mut codes = x.qs.as_ptr();
            let mut acts = y.qs.as_ptr();

            let packed = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let scale_codes = _mm_and_si128(packed, nibble);
            let min_codes = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble);

            // The minimum term, whole, before the loop: sixteen values per
            // sub-block is exactly one `bsums` entry.
            let from_mins = _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(min_codes),
                _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i),
            );
            acc = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&dmin),
                _mm256_cvtepi32_ps(from_mins),
                acc,
            );

            let mut total = _mm256_setzero_si256();
            for scales in scales_by_half(scale_codes) {
                let packed_codes = _mm256_loadu_si256(codes as *const __m256i);
                codes = codes.add(32);

                let mut next = || {
                    let a = _mm256_loadu_si256(acts as *const __m256i);
                    acts = acts.add(32);
                    a
                };
                let (a0, a1, a2, a3) = (next(), next(), next(), next());

                let code = |shift_done: __m256i| _mm256_and_si256(shift_done, two_bits);
                let p0 = scaled(scales, 0, _mm256_maddubs_epi16(code(packed_codes), a0));
                let p1 = scaled(
                    scales,
                    1,
                    _mm256_maddubs_epi16(code(_mm256_srli_epi16(packed_codes, 2)), a1),
                );
                let p2 = scaled(
                    scales,
                    2,
                    _mm256_maddubs_epi16(code(_mm256_srli_epi16(packed_codes, 4)), a2),
                );
                let p3 = scaled(
                    scales,
                    3,
                    _mm256_maddubs_epi16(code(_mm256_srli_epi16(packed_codes, 6)), a3),
                );

                total = _mm256_add_epi32(
                    total,
                    _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)),
                );
            }
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(total), acc);
        }
        lanes_sum(acc)
    }
}

/// Three bits per value: two in `qs`, the third in a `hmask` bitplane, and
/// sixteen signed 6-bit scales biased by 32.
///
/// The third bit carries -4 when CLEAR, so the kernel builds it INVERTED  - 
/// `hmask` complemented gives a 0-or-1 that, shifted left by two, is the amount
/// to subtract. Low part and high part go through `maddubs` separately and the
/// difference is taken in i16, which keeps both operands unsigned where the
/// instruction requires it.
#[inline(always)]
pub(crate) fn vec_dot_q3k_q8k(xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
    unsafe {
        let two_bits = _mm256_set1_epi8(3);
        let ones = _mm256_set1_epi8(1);
        let bias = _mm_set1_epi8(32);
        let mut acc = _mm256_setzero_ps();

        for (x, y) in xs.iter().zip(ys) {
            let d = y.d * x.d.to_f32();
            let mut codes = x.qs.as_ptr();
            let mut acts = y.qs.as_ptr();

            // Twelve bytes hold sixteen 6-bit scales the way q4_K packs its
            // eight, except all sixteen are scales and each is biased by 32.
            let (low, high, spare) = (word(&x.scales, 0), word(&x.scales, 4), word(&x.scales, 8));
            let scale_bytes = _mm_sub_epi8(
                _mm_set_epi32(
                    rejoin((high >> 4) & LOW4, spare >> 6) as i32,
                    rejoin((low >> 4) & LOW4, spare >> 4) as i32,
                    rejoin(high & LOW4, spare >> 2) as i32,
                    rejoin(low & LOW4, spare) as i32,
                ),
                bias,
            );
            let third_bits = _mm256_loadu_si256(x.hmask.as_ptr() as *const __m256i);

            let mut total = _mm256_setzero_si256();
            for (half, scales) in scales_by_half(scale_bytes).into_iter().enumerate() {
                let packed_codes = _mm256_loadu_si256(codes as *const __m256i);
                codes = codes.add(32);

                let mut next = || {
                    let a = _mm256_loadu_si256(acts as *const __m256i);
                    acts = acts.add(32);
                    a
                };
                let (a0, a1, a2, a3) = (next(), next(), next(), next());

                // Bit 4.half + q of the plane, inverted, at position two.
                macro_rules! borrow {
                    ($bit:literal) => {
                        _mm256_slli_epi16(
                            _mm256_srli_epi16(
                                _mm256_andnot_si256(third_bits, _mm256_slli_epi16(ones, $bit)),
                                $bit,
                            ),
                            2,
                        )
                    };
                }
                let borrows = if half == 0 {
                    [borrow!(0), borrow!(1), borrow!(2), borrow!(3)]
                } else {
                    [borrow!(4), borrow!(5), borrow!(6), borrow!(7)]
                };

                let low_two = [
                    _mm256_and_si256(packed_codes, two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(packed_codes, 2), two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(packed_codes, 4), two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(packed_codes, 6), two_bits),
                ];

                let mut quarter = [_mm256_setzero_si256(); 4];
                for (q, (a, (lo, bo))) in [a0, a1, a2, a3]
                    .into_iter()
                    .zip(low_two.into_iter().zip(borrows))
                    .enumerate()
                {
                    let value =
                        _mm256_sub_epi16(_mm256_maddubs_epi16(lo, a), _mm256_maddubs_epi16(bo, a));
                    quarter[q] = scaled(scales, q, value);
                }
                total = _mm256_add_epi32(
                    total,
                    _mm256_add_epi32(
                        _mm256_add_epi32(quarter[0], quarter[1]),
                        _mm256_add_epi32(quarter[2], quarter[3]),
                    ),
                );
            }
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(total), acc);
        }
        lanes_sum(acc)
    }
}

/// The minimum term of q4_K and q5_K: a 32-value sub-block spans two `bsums`
/// entries, so those are summed pairwise first, after which it is one `madd`.
#[inline(always)]
unsafe fn offset_term(scales_and_mins: __m256i, bsums: &[i16; QK_K / 16]) -> __m128i {
    let loaded = _mm256_loadu_si256(bsums.as_ptr() as *const __m256i);
    let per_sub_block = _mm_hadd_epi16(
        _mm256_extracti128_si256(loaded, 0),
        _mm256_extracti128_si256(loaded, 1),
    );
    _mm_madd_epi16(_mm256_extracti128_si256(scales_and_mins, 1), per_sub_block)
}

/// The eight scales and eight minimums of q4_K/q5_K, widened to i16 in one
/// register: scales in the low half, minimums in the high.
#[inline(always)]
unsafe fn scales_and_mins(packed: &[u8; 12]) -> __m256i {
    let u = crate::tensor::quant_cpu::repack_q4k::sub_scales_and_mins(packed);
    _mm256_cvtepu8_epi16(_mm_set_epi32(
        u[3] as i32,
        u[2] as i32,
        u[1] as i32,
        u[0] as i32,
    ))
}

/// Four bits per value, thirty-two to a sub-block, scales and minimums packed
/// six bits each.
#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    unsafe {
        let nibble = _mm256_set1_epi8(0xF);
        let mut acc = _mm256_setzero_ps();
        let mut acc_min = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();
            let mut codes = x.qs.as_ptr();
            let mut acts = y.qs.as_ptr();

            let both = scales_and_mins(&x.scales);
            let from_mins = offset_term(both, &y.bsums);
            acc_min = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(from_mins), acc_min);

            let sc = _mm256_extracti128_si256(both, 0);
            let scales = join_halves(sc, sc);

            let mut total = _mm256_setzero_si256();
            for j in 0..QK_K / 64 {
                let packed_codes = _mm256_loadu_si256(codes as *const __m256i);
                codes = codes.add(32);

                let low = _mm256_and_si256(packed_codes, nibble);
                let a_low = _mm256_loadu_si256(acts as *const __m256i);
                acts = acts.add(32);
                total = _mm256_add_epi32(
                    total,
                    _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales, scale_over_16_pairs(2 * j)),
                        _mm256_maddubs_epi16(low, a_low),
                    ),
                );

                let high = _mm256_and_si256(_mm256_srli_epi16(packed_codes, 4), nibble);
                let a_high = _mm256_loadu_si256(acts as *const __m256i);
                acts = acts.add(32);
                total = _mm256_add_epi32(
                    total,
                    _mm256_madd_epi16(
                        _mm256_shuffle_epi8(scales, scale_over_16_pairs(2 * j + 1)),
                        _mm256_maddubs_epi16(high, a_high),
                    ),
                );
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(total), acc);
        }

        let pairs = _mm_add_ps(acc_min, _mm_movehl_ps(acc_min, acc_min));
        let min_total = _mm_add_ss(pairs, _mm_movehdup_ps(pairs));
        lanes_sum(acc) + _mm_cvtss_f32(min_total)
    }
}

/// q4_K plus a fifth bit per value, held in a `qh` bitplane consumed one bit at
/// a time as the loop walks the super-block.
#[inline(always)]
pub(crate) fn vec_dot_q5k_q8k(xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    unsafe {
        let nibble = _mm256_set1_epi8(0xF);
        let ones = _mm256_set1_epi8(1);
        let zero = _mm_setzero_si128();
        let mut acc = _mm256_setzero_ps();
        let mut from_mins_total = 0f32;

        for (x, y) in xs.iter().zip(ys) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();
            let mut codes = x.qs.as_ptr();
            let mut acts = y.qs.as_ptr();

            let both = scales_and_mins(&x.scales);
            // Folded to a scalar rather than accumulated in a register: q5_K's
            // loop already holds the bit plane and its walking mask.
            let folded = _mm_hadd_epi32(_mm_hadd_epi32(offset_term(both, &y.bsums), zero), zero);
            from_mins_total += dmin * _mm_extract_epi32(folded, 0) as f32;

            let sc = _mm256_extracti128_si256(both, 0);
            let scales = join_halves(sc, sc);

            let fifth_bits = _mm256_loadu_si256(x.qh.as_ptr() as *const __m256i);
            let mut bit = ones;

            let mut total = _mm256_setzero_si256();
            for j in 0..QK_K / 64 {
                let packed_codes = _mm256_loadu_si256(codes as *const __m256i);
                codes = codes.add(32);

                // Bits 2j and 2j+1 of the plane, each brought down to lane zero
                // and put back at position four, where a fifth bit belongs.
                macro_rules! fifth {
                    ($shift:literal) => {{
                        let taken = _mm256_srli_epi16(_mm256_and_si256(fifth_bits, bit), $shift);
                        bit = _mm256_slli_epi16(bit, 1);
                        _mm256_slli_epi16(taken, 4)
                    }};
                }
                let (top_low, top_high) = match j {
                    0 => (fifth!(0), fifth!(1)),
                    1 => (fifth!(2), fifth!(3)),
                    2 => (fifth!(4), fifth!(5)),
                    3 => (fifth!(6), fifth!(7)),
                    _ => unreachable!(),
                };

                let low = _mm256_add_epi8(_mm256_and_si256(packed_codes, nibble), top_low);
                let a_low = _mm256_loadu_si256(acts as *const __m256i);
                acts = acts.add(32);

                let high = _mm256_add_epi8(
                    _mm256_and_si256(_mm256_srli_epi16(packed_codes, 4), nibble),
                    top_high,
                );
                let a_high = _mm256_loadu_si256(acts as *const __m256i);
                acts = acts.add(32);

                total = _mm256_add_epi32(
                    total,
                    _mm256_add_epi32(
                        _mm256_madd_epi16(
                            _mm256_shuffle_epi8(scales, scale_over_16_pairs(2 * j)),
                            _mm256_maddubs_epi16(low, a_low),
                        ),
                        _mm256_madd_epi16(
                            _mm256_shuffle_epi8(scales, scale_over_16_pairs(2 * j + 1)),
                            _mm256_maddubs_epi16(high, a_high),
                        ),
                    ),
                );
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(total), acc);
        }
        lanes_sum(acc) + from_mins_total
    }
}

/// Six bits per value - four in `ql`, two in `qh` - sixteen to a sub-block,
/// with i8 scales and a fixed -32 zero point.
///
/// That zero point is constant, so `Σ -32.a.scale` over a sub-block is
/// `-32.scale` times one `bsums` entry: one `madd` and a shift by five for the
/// whole block, instead of a subtract on every product in the loop.
#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    unsafe {
        let nibble = _mm256_set1_epi8(0xF);
        let two_bits = _mm256_set1_epi8(3);
        let mut acc = _mm256_setzero_ps();

        for (x, y) in xs.iter().zip(ys) {
            let d = y.d * x.d.to_f32();
            let mut low_ptr = x.ql.as_ptr();
            let mut high_ptr = x.qh.as_ptr();
            let mut acts = y.qs.as_ptr();

            let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let zero_point = _mm256_slli_epi32(
                _mm256_madd_epi16(
                    _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i),
                    _mm256_cvtepi8_epi16(scales),
                ),
                5,
            );

            let mut total = _mm256_setzero_si256();
            for j in 0..QK_K / 128 {
                let low_a = _mm256_loadu_si256(low_ptr as *const __m256i);
                low_ptr = low_ptr.add(32);
                let low_b = _mm256_loadu_si256(low_ptr as *const __m256i);
                low_ptr = low_ptr.add(32);
                let high_bits = _mm256_loadu_si256(high_ptr as *const __m256i);
                high_ptr = high_ptr.add(32);

                // Quarter q takes bits 2q..2q+2 of the high plane and puts them
                // at position four, above the nibble they complete.
                let codes = [
                    _mm256_or_si256(
                        _mm256_and_si256(low_a, nibble),
                        _mm256_slli_epi16(_mm256_and_si256(high_bits, two_bits), 4),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(low_b, nibble),
                        _mm256_slli_epi16(
                            _mm256_and_si256(_mm256_srli_epi16(high_bits, 2), two_bits),
                            4,
                        ),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(low_a, 4), nibble),
                        _mm256_slli_epi16(
                            _mm256_and_si256(_mm256_srli_epi16(high_bits, 4), two_bits),
                            4,
                        ),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(low_b, 4), nibble),
                        _mm256_slli_epi16(
                            _mm256_and_si256(_mm256_srli_epi16(high_bits, 6), two_bits),
                            4,
                        ),
                    ),
                ];

                for (q, code) in codes.into_iter().enumerate() {
                    let scale = _mm_shuffle_epi8(scales, scale_pair_over_8_bytes(4 * j + q));
                    let a = _mm256_loadu_si256(acts as *const __m256i);
                    acts = acts.add(32);
                    total = _mm256_add_epi32(
                        total,
                        _mm256_madd_epi16(
                            _mm256_cvtepi8_epi16(scale),
                            _mm256_maddubs_epi16(code, a),
                        ),
                    );
                }
            }
            let total = _mm256_sub_epi32(total, zero_point);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(total), acc);
        }
        lanes_sum(acc)
    }
}

/// Eight bits per value with one f32 block scale - the activation format
/// against itself, which is what a matmul reduces to when both sides are
/// already q8_K.
#[inline(always)]
pub(crate) fn vec_dot_q8k_q8k(xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys) {
            let mut total = _mm256_setzero_si256();
            for j in (0..QK_K).step_by(32) {
                let a = _mm256_loadu_si256(x.qs.as_ptr().add(j) as *const __m256i);
                let b = _mm256_loadu_si256(y.qs.as_ptr().add(j) as *const __m256i);
                // Widened to i16 before the product: eight bits against eight
                // bits reaches 2^14 per pair, which `maddubs` would saturate.
                total = _mm256_add_epi32(
                    total,
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 0)),
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 0)),
                    ),
                );
                total = _mm256_add_epi32(
                    total,
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1)),
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1)),
                    ),
                );
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(x.d * y.d), _mm256_cvtepi32_ps(total), acc);
        }
        lanes_sum(acc)
    }
}
