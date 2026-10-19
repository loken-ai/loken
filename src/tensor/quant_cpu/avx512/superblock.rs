//! AVX-512 dot products for the k-quants.

use core::arch::x86_64::*;

use super::require_avx512;
use crate::tensor::quant_cpu::avx::lanes_sum;
use crate::tensor::quant_cpu::avx::superblock::{offset_term, scales_and_mins};
use crate::tensor::quant_cpu::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K};

/// Selector that puts scale `a` over lanes 0..8 and scale `b` over lanes 8..16.
#[inline(always)]
unsafe fn scale_pair(a: i32, b: i32) -> __m512i {
    _mm512_set_epi32(b, b, b, b, b, b, b, b, a, a, a, a, a, a, a, a)
}

/// Four bits per value, thirty-two to a sub-block, scales and minimums packed six bits each.
///
/// A 64-byte load holds two 32-byte runs, each a pair of sub-blocks (low nibbles, high nibbles),
/// so one `vpdpbusd` covers two sub-blocks and the loop runs twice.
pub(crate) fn vec_dot_q4k_q8k(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    require_avx512();
    // SAFETY: `require_avx512` just confirmed the CPU has every feature the kernel enables.
    unsafe { vec_dot_q4k_q8k_avx512(xs, ys) }
}

#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn vec_dot_q4k_q8k_avx512(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    let nibble = _mm512_set1_epi8(0xF);
    let mut acc = _mm256_setzero_ps();
    let mut acc_min = _mm_setzero_ps();

    for (x, y) in xs.iter().zip(ys) {
        let d = y.d * crate::tensor::quant_cpu::avx::f16c_to_f32(x.d);
        let dmin = -y.d * crate::tensor::quant_cpu::avx::f16c_to_f32(x.dmin);

        let both = scales_and_mins(&x.scales);
        let from_mins = offset_term(both, &y.bsums);
        acc_min = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(from_mins), acc_min);

        let mut sc = [0i32; 8];
        _mm256_storeu_si256(
            sc.as_mut_ptr() as *mut __m256i,
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<0>(both)),
        );

        let qs = x.qs.as_ptr();
        let acts = y.qs.as_ptr();
        let mut total = _mm512_setzero_si512();
        for half in 0..2 {
            let packed = _mm512_loadu_si512(qs.add(64 * half) as *const _);
            let a_first = _mm512_loadu_si512(acts.add(128 * half) as *const _);
            let a_second = _mm512_loadu_si512(acts.add(128 * half + 64) as *const _);
            // 128-bit lanes: [first.0 first.1 second.0 second.1] and [first.2 first.3 second.2
            // second.3] - the activations of the two low-nibble runs, then of the two high ones.
            let a_low = _mm512_shuffle_i64x2::<0b01_00_01_00>(a_first, a_second);
            let a_high = _mm512_shuffle_i64x2::<0b11_10_11_10>(a_first, a_second);

            let low = _mm512_and_si512(packed, nibble);
            let high = _mm512_and_si512(_mm512_srli_epi16::<4>(packed), nibble);

            let b = 4 * half;
            let dot_low = _mm512_dpbusd_epi32(_mm512_setzero_si512(), low, a_low);
            let dot_high = _mm512_dpbusd_epi32(_mm512_setzero_si512(), high, a_high);
            total = _mm512_add_epi32(
                total,
                _mm512_mullo_epi32(dot_low, scale_pair(sc[b], sc[b + 2])),
            );
            total = _mm512_add_epi32(
                total,
                _mm512_mullo_epi32(dot_high, scale_pair(sc[b + 1], sc[b + 3])),
            );
        }
        // Lanes 8..16 hold the same positions of the other run: adding them in i32 is exact and
        // leaves the lane layout of the AVX2 kernel.
        let total = _mm256_add_epi32(
            _mm512_castsi512_si256(total),
            _mm512_extracti64x4_epi64::<1>(total),
        );
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(total), acc);
    }

    let pairs = _mm_add_ps(acc_min, _mm_movehl_ps(acc_min, acc_min));
    let min_total = _mm_add_ss(pairs, _mm_movehdup_ps(pairs));
    lanes_sum(acc) + _mm_cvtss_f32(min_total)
}

/// The two 256-bit halves of a 512-bit i32 vector added: each half holds the sums of one 32-byte
/// run, and the AVX2 kernels add those runs lane by lane.
#[inline(always)]
unsafe fn fold(v: __m512i) -> __m256i {
    _mm256_add_epi32(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64::<1>(v))
}

/// q4_K plus a fifth bit per value. A zmm holds a sub-block pair: the low nibbles of 32 bytes in
/// the low half, the high nibbles in the high half, each with its own fifth bit.
pub(crate) fn vec_dot_q5k_q8k(xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    require_avx512();
    // SAFETY: `require_avx512` just confirmed the CPU has every feature the kernel enables.
    unsafe { vec_dot_q5k_q8k_avx512(xs, ys) }
}

#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn vec_dot_q5k_q8k_avx512(xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    let nibble = _mm512_set1_epi8(0xF);
    let one = _mm512_set1_epi8(1);
    let nib_shift = _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(_mm256_set1_epi16(0)),
        _mm256_set1_epi16(4),
    );
    let zero = _mm_setzero_si128();
    let mut acc = _mm256_setzero_ps();
    let mut from_mins_total = 0f32;

    for (x, y) in xs.iter().zip(ys) {
        let d = y.d * crate::tensor::quant_cpu::avx::f16c_to_f32(x.d);
        let dmin = -y.d * crate::tensor::quant_cpu::avx::f16c_to_f32(x.dmin);

        let both = scales_and_mins(&x.scales);
        let folded = _mm_hadd_epi32(_mm_hadd_epi32(offset_term(both, &y.bsums), zero), zero);
        from_mins_total += dmin * _mm_extract_epi32::<0>(folded) as f32;

        let mut sc = [0i32; 8];
        _mm256_storeu_si256(
            sc.as_mut_ptr() as *mut __m256i,
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<0>(both)),
        );

        let fifth_bits =
            _mm512_broadcast_i64x4(_mm256_loadu_si256(x.qh.as_ptr() as *const __m256i));
        let mut total = _mm512_setzero_si512();
        for j in 0..4 {
            let raw = _mm512_broadcast_i64x4(_mm256_loadu_si256(
                x.qs.as_ptr().add(32 * j) as *const __m256i
            ));
            let plane = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set1_epi16(2 * j as i16)),
                _mm256_set1_epi16((2 * j + 1) as i16),
            );
            let top =
                _mm512_slli_epi16::<4>(_mm512_and_si512(_mm512_srlv_epi16(fifth_bits, plane), one));
            let codes = _mm512_add_epi8(
                _mm512_and_si512(_mm512_srlv_epi16(raw, nib_shift), nibble),
                top,
            );
            let acts = _mm512_loadu_si512(y.qs.as_ptr().add(64 * j) as *const _);
            let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, acts);
            total = _mm512_add_epi32(
                total,
                _mm512_mullo_epi32(dot, scale_pair(sc[2 * j], sc[2 * j + 1])),
            );
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(fold(total)), acc);
    }
    lanes_sum(acc) + from_mins_total
}

/// Six bits per value, sixteen to a scale, with the fixed zero point of 32 folded once per block.
pub(crate) fn vec_dot_q6k_q8k(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    require_avx512();
    // SAFETY: `require_avx512` just confirmed the CPU has every feature the kernel enables.
    unsafe { vec_dot_q6k_q8k_avx512(xs, ys) }
}

#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn vec_dot_q6k_q8k_avx512(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    let m4 = _mm512_set1_epi8(0xF);
    let m2 = _mm512_set1_epi8(3);
    let shift_01 = _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(_mm256_set1_epi16(0)),
        _mm256_set1_epi16(2),
    );
    let shift_23 = _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(_mm256_set1_epi16(4)),
        _mm256_set1_epi16(6),
    );
    // Byte i of a four-scale word becomes four lanes: one scale over each 4-lane group.
    let spread = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    let mut acc = _mm256_setzero_ps();

    for (x, y) in xs.iter().zip(ys) {
        let d = y.d * crate::tensor::quant_cpu::avx::f16c_to_f32(x.d);
        let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
        let zero_point = _mm256_slli_epi32::<5>(_mm256_madd_epi16(
            _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i),
            _mm256_cvtepi8_epi16(scales),
        ));
        let mut total = _mm512_setzero_si512();
        for step in 0..2 {
            let ql = _mm512_loadu_si512(x.ql.as_ptr().add(64 * step) as *const _);
            let qh = _mm512_broadcast_i64x4(_mm256_loadu_si256(
                x.qh.as_ptr().add(32 * step) as *const __m256i
            ));
            let h_01 =
                _mm512_slli_epi16::<4>(_mm512_and_si512(_mm512_srlv_epi16(qh, shift_01), m2));
            let h_23 =
                _mm512_slli_epi16::<4>(_mm512_and_si512(_mm512_srlv_epi16(qh, shift_23), m2));
            let pair = [
                _mm512_or_si512(_mm512_and_si512(ql, m4), h_01),
                _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16::<4>(ql), m4), h_23),
            ];
            for (p, codes) in pair.into_iter().enumerate() {
                let g = 8 * step + 4 * p;
                let word = (x.scales.as_ptr().add(g) as *const i32).read_unaligned();
                let sv = _mm512_cvtepi8_epi32(_mm_shuffle_epi8(_mm_cvtsi32_si128(word), spread));
                let acts = _mm512_loadu_si512(y.qs.as_ptr().add(128 * step + 64 * p) as *const _);
                let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, acts);
                total = _mm512_add_epi32(total, _mm512_mullo_epi32(dot, sv));
            }
        }
        let total = _mm256_sub_epi32(fold(total), zero_point);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(total), acc);
    }
    lanes_sum(acc)
}

/// Both operands eight-bit: widened to i16 for `vpmaddwd`: a byte product can reach 2^14.
pub(crate) fn vec_dot_q8k_q8k(xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    require_avx512();
    // SAFETY: `require_avx512` just confirmed the CPU has every feature the kernel enables.
    unsafe { vec_dot_q8k_q8k_avx512(xs, ys) }
}

#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn vec_dot_q8k_q8k_avx512(xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for (x, y) in xs.iter().zip(ys) {
        let mut total = _mm512_setzero_si512();
        for j in (0..256).step_by(32) {
            let a =
                _mm512_cvtepi8_epi16(_mm256_loadu_si256(x.qs.as_ptr().add(j) as *const __m256i));
            let b =
                _mm512_cvtepi8_epi16(_mm256_loadu_si256(y.qs.as_ptr().add(j) as *const __m256i));
            total = _mm512_add_epi32(total, _mm512_madd_epi16(a, b));
        }
        acc = _mm256_fmadd_ps(
            _mm256_set1_ps(x.d * y.d),
            _mm256_cvtepi32_ps(fold(total)),
            acc,
        );
    }
    lanes_sum(acc)
}

/// A pair of 16-value scale groups x four lanes: the four scale bytes at `scales[4 * pair..]`
/// as i32, each over the four i32 lanes of its group.
#[inline(always)]
unsafe fn group_scales(scales: *const i8, pair: usize) -> __m512i {
    let spread = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    let word = (scales.add(4 * pair) as *const i32).read_unaligned();
    _mm512_cvtepi8_epi32(_mm_shuffle_epi8(_mm_cvtsi32_si128(word), spread))
}

/// `[lo | hi]` with a 16-bit shift count in each 256-bit half.
#[inline(always)]
unsafe fn shifts(lo: i16, hi: i16) -> __m512i {
    _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(_mm256_set1_epi16(lo)),
        _mm256_set1_epi16(hi),
    )
}

/// Two bits per value, sixteen to a sub-block, a 4-bit scale and minimum to each.
pub(crate) fn vec_dot_q2k_q8k(xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    require_avx512();
    // SAFETY: `require_avx512` just confirmed the CPU has every feature the kernel enables.
    unsafe { vec_dot_q2k_q8k_avx512(xs, ys) }
}

#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn vec_dot_q2k_q8k_avx512(xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    let two_bits = _mm512_set1_epi8(3);
    let nibble = _mm_set1_epi8(0xF);
    let pair_shift = [shifts(0, 2), shifts(4, 6)];
    let mut acc = -0.0f32;

    for (x, y) in xs.iter().zip(ys) {
        let packed = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
        let scale_codes = _mm_and_si128(packed, nibble);
        let min_codes = _mm_and_si128(_mm_srli_epi16::<4>(packed), nibble);
        let from_mins = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(min_codes),
            _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i),
        );

        let mut scales = [0i8; 16];
        _mm_storeu_si128(scales.as_mut_ptr() as *mut __m128i, scale_codes);
        let mut total = _mm512_setzero_si512();
        for half in 0..2 {
            let raw = _mm512_broadcast_i64x4(_mm256_loadu_si256(
                x.qs.as_ptr().add(32 * half) as *const __m256i
            ));
            for p in 0..2 {
                let codes = _mm512_and_si512(_mm512_srlv_epi16(raw, pair_shift[p]), two_bits);
                let acts = _mm512_loadu_si512(y.qs.as_ptr().add(128 * half + 64 * p) as *const _);
                let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, acts);
                total = _mm512_add_epi32(
                    total,
                    _mm512_mullo_epi32(dot, group_scales(scales.as_ptr(), 2 * half + p)),
                );
            }
        }
        acc += crate::tensor::quant_cpu::format::q2_k::combine(
            x,
            y.d,
            hsum_i32_8(fold(total)),
            hsum_i32_8(from_mins),
        );
    }
    acc
}

/// The eight lanes of `v` added.
#[inline(always)]
unsafe fn hsum_i32_8(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x4E>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0xB1>(s));
    _mm_cvtsi128_si32(s)
}

/// Three bits per value: two in `qs`, the third in `hmask`, which carries -4 when clear.
pub(crate) fn vec_dot_q3k_q8k(xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
    require_avx512();
    // SAFETY: `require_avx512` just confirmed the CPU has every feature the kernel enables.
    unsafe { vec_dot_q3k_q8k_avx512(xs, ys) }
}

#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn vec_dot_q3k_q8k_avx512(xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
    let two_bits = _mm512_set1_epi8(3);
    let ones = _mm512_set1_epi8(1);
    let pair_shift = [shifts(0, 2), shifts(4, 6)];
    let mut acc = _mm256_setzero_ps();

    for (x, y) in xs.iter().zip(ys) {
        let d = y.d * crate::tensor::quant_cpu::avx::f16c_to_f32(x.d);
        let scales = crate::tensor::quant_cpu::avx::superblock::q3k_scales(&x.scales);
        let third = _mm512_broadcast_i64x4(_mm256_loadu_si256(x.hmask.as_ptr() as *const __m256i));
        let mut total = _mm512_setzero_si512();
        for half in 0..2 {
            let raw = _mm512_broadcast_i64x4(_mm256_loadu_si256(
                x.qs.as_ptr().add(32 * half) as *const __m256i
            ));
            for p in 0..2 {
                let low = _mm512_and_si512(_mm512_srlv_epi16(raw, pair_shift[p]), two_bits);
                // Bit 4 * half + 2 * p + {0, 1} of the plane, inverted, at position two.
                let bit = shifts((4 * half + 2 * p) as i16, (4 * half + 2 * p + 1) as i16);
                let borrow = _mm512_slli_epi16::<2>(_mm512_srlv_epi16(
                    _mm512_andnot_si512(third, _mm512_sllv_epi16(ones, bit)),
                    bit,
                ));
                let acts = _mm512_loadu_si512(y.qs.as_ptr().add(128 * half + 64 * p) as *const _);
                let value = _mm512_sub_epi32(
                    _mm512_dpbusd_epi32(_mm512_setzero_si512(), low, acts),
                    _mm512_dpbusd_epi32(_mm512_setzero_si512(), borrow, acts),
                );
                total = _mm512_add_epi32(
                    total,
                    _mm512_mullo_epi32(value, group_scales(scales.as_ptr(), 2 * half + p)),
                );
            }
        }
        acc = _mm256_fmadd_ps(
            _mm256_broadcast_ss(&d),
            _mm256_cvtepi32_ps(fold(total)),
            acc,
        );
    }
    lanes_sum(acc)
}
