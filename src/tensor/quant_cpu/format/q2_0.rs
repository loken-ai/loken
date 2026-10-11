//! q2_0 - sixty-four values, a scale and two-bit codes: `y = (q - 1) . d` for codes 0 to 3, so the
//! levels are -1, 0, +1 and +2 times the scale. Four codes share a byte, lowest first.

use super::q8_64::QK8_64_HALF;
use super::*;

pub const QK2_0: usize = 64;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK2_0 / 4],
}
const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == 2 + QK2_0 / 4);

impl BlockFormat for BlockQ2_0 {
    const DTYPE: GgmlDType = GgmlDType::Q2_0;
    const BLOCK_LEN: usize = QK2_0;
    type ActivationBlock = BlockQ8_64;

    /// Every code at the level of zero.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0x55; QK2_0 / 4],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            for (j, out) in y.iter_mut().enumerate() {
                let code = (block.qs[j / 4] >> (2 * (j % 4))) & 3;
                *out = (code as i32 - 1) as f32 * d;
            }
        }
    }

    /// The scale puts the extreme on the end of the level range it falls in; each value takes
    /// the nearest of the four levels.
    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            let amax = x
                .iter()
                .copied()
                .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
            let d = if amax == 0.0 {
                0.0
            } else if amax > 0.0 {
                amax / 2.0
            } else {
                -amax
            };
            block.d = f16::from_f32(d);
            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            block.qs = [0; QK2_0 / 4];
            for (j, v) in x.iter().enumerate() {
                let level = (v * inv).round().clamp(-1.0, 2.0) as i32;
                block.qs[j / 4] |= ((level + 1) as u8) << (2 * (j % 4));
            }
        }
    }

    /// Each block's integer sum is exact, and the blocks join in order as the scalar loop joins
    /// them, so the result is the scalar's to the bit.
    #[allow(unreachable_code)]
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return unsafe { dot_blocks(xs, ys) };
        Self::dot_scalar(xs, ys)
    }

    /// [`Self::dot`] over several activations, each block's codes unpacked once.
    #[allow(unreachable_code)]
    fn dot_many(xs: &[Self], ys: &[&[Self::ActivationBlock]], out: &mut [f32]) {
        #[cfg(target_feature = "avx2")]
        return unsafe { dot_blocks_many(xs, ys, out) };
        for (o, y) in out.iter_mut().zip(ys) {
            *o = Self::dot(xs, y);
        }
    }

    /// The two halves of the activation block carry a scale each, as the card's do: each half's
    /// integer dot is scaled on its own and joins the total in turn.
    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        let mut total = 0f32;
        for (x, y) in xs.iter().zip(ys) {
            let d = x.d.to_f32();
            for h in 0..2 {
                let at = h * QK8_64_HALF;
                let dot: i32 = (at..at + QK8_64_HALF)
                    .map(|j| {
                        let code = ((x.qs[j / 4] >> (2 * (j % 4))) & 3) as i32;
                        (code - 1) * y.qs[j] as i32
                    })
                    .sum();
                total += dot as f32 * d * y.d[h].to_f32();
            }
        }
        total
    }
}

/// The dot of whole blocks with the four codes of each byte unpacked into the activation's order
/// and multiplied as unsigned codes by signed bytes; the offset of one level is taken off with
/// the activation's own sum.
#[cfg(target_feature = "avx2")]
unsafe fn dot_blocks(xs: &[BlockQ2_0], ys: &[BlockQ8_64]) -> f32 {
    use std::arch::x86_64::*;
    let mask = _mm_set1_epi8(3);
    let ones8 = _mm_set1_epi8(1);
    let ones16 = _mm_set1_epi16(1);
    let mut total = 0f32;
    for (x, y) in xs.iter().zip(ys) {
        let bytes = _mm_loadu_si128(x.qs.as_ptr() as *const __m128i);
        let plane0 = _mm_and_si128(bytes, mask);
        let plane1 = _mm_and_si128(_mm_srli_epi16(bytes, 2), mask);
        let plane2 = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        let plane3 = _mm_and_si128(_mm_srli_epi16(bytes, 6), mask);
        // Code `4i + k` is plane `k` of byte `i`: interleave the planes back into that order.
        let low01 = _mm_unpacklo_epi8(plane0, plane1);
        let high01 = _mm_unpackhi_epi8(plane0, plane1);
        let low23 = _mm_unpacklo_epi8(plane2, plane3);
        let high23 = _mm_unpackhi_epi8(plane2, plane3);
        let codes = [
            _mm_unpacklo_epi16(low01, low23),
            _mm_unpackhi_epi16(low01, low23),
            _mm_unpacklo_epi16(high01, high23),
            _mm_unpackhi_epi16(high01, high23),
        ];
        let acts = y.qs.as_ptr() as *const __m128i;
        let horizontal = |v: __m128i| {
            let v = _mm_hadd_epi32(v, v);
            _mm_cvtsi128_si32(_mm_hadd_epi32(v, v))
        };
        // Sixteen codes to a group, two groups to a half: each half's dot takes its own scale.
        let d = x.d.to_f32();
        for h in 0..2 {
            let mut sum = _mm_setzero_si128();
            let mut act_sum = _mm_setzero_si128();
            for (i, code) in codes[2 * h..2 * h + 2].iter().enumerate() {
                let a = _mm_loadu_si128(acts.add(2 * h + i));
                sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_maddubs_epi16(*code, a), ones16));
                act_sum =
                    _mm_add_epi32(act_sum, _mm_madd_epi16(_mm_maddubs_epi16(ones8, a), ones16));
            }
            let dot = horizontal(sum) - horizontal(act_sum);
            total += dot as f32 * d * y.d[h].to_f32();
        }
    }
    total
}

/// The four planes of a block's codes interleaved back into the activation's order, sixteen codes
/// to a group.
#[cfg(target_feature = "avx2")]
#[inline]
unsafe fn unpack_codes(x: &BlockQ2_0) -> [std::arch::x86_64::__m128i; 4] {
    use std::arch::x86_64::*;
    let mask = _mm_set1_epi8(3);
    let bytes = _mm_loadu_si128(x.qs.as_ptr() as *const __m128i);
    let plane0 = _mm_and_si128(bytes, mask);
    let plane1 = _mm_and_si128(_mm_srli_epi16(bytes, 2), mask);
    let plane2 = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    let plane3 = _mm_and_si128(_mm_srli_epi16(bytes, 6), mask);
    let low01 = _mm_unpacklo_epi8(plane0, plane1);
    let high01 = _mm_unpackhi_epi8(plane0, plane1);
    let low23 = _mm_unpacklo_epi8(plane2, plane3);
    let high23 = _mm_unpackhi_epi8(plane2, plane3);
    [
        _mm_unpacklo_epi16(low01, low23),
        _mm_unpackhi_epi16(low01, low23),
        _mm_unpacklo_epi16(high01, high23),
        _mm_unpackhi_epi16(high01, high23),
    ]
}

/// [`dot_blocks`] over several activations: each block's codes are unpacked once and dotted
/// against every activation in the arithmetic of [`dot_blocks`].
#[cfg(target_feature = "avx2")]
unsafe fn dot_blocks_many(xs: &[BlockQ2_0], ys: &[&[BlockQ8_64]], out: &mut [f32]) {
    use std::arch::x86_64::*;
    const MANY: usize = 8;
    let ones8 = _mm_set1_epi8(1);
    let ones16 = _mm_set1_epi16(1);
    let horizontal = |v: __m128i| {
        let v = _mm_hadd_epi32(v, v);
        _mm_cvtsi128_si32(_mm_hadd_epi32(v, v))
    };
    for (ys, out) in ys.chunks(MANY).zip(out.chunks_mut(MANY)) {
        let mut totals = [0f32; MANY];
        for (bi, x) in xs.iter().enumerate() {
            let codes = unpack_codes(x);
            let d = x.d.to_f32();
            for h in 0..2 {
                for (total, y) in totals.iter_mut().zip(ys) {
                    let y = &y[bi];
                    let acts = y.qs.as_ptr() as *const __m128i;
                    let mut sum = _mm_setzero_si128();
                    let mut act_sum = _mm_setzero_si128();
                    for (i, code) in codes[2 * h..2 * h + 2].iter().enumerate() {
                        let a = _mm_loadu_si128(acts.add(2 * h + i));
                        sum =
                            _mm_add_epi32(sum, _mm_madd_epi16(_mm_maddubs_epi16(*code, a), ones16));
                        act_sum = _mm_add_epi32(
                            act_sum,
                            _mm_madd_epi16(_mm_maddubs_epi16(ones8, a), ones16),
                        );
                    }
                    let dot = horizontal(sum) - horizontal(act_sum);
                    *total += dot as f32 * d * y.d[h].to_f32();
                }
            }
        }
        out.copy_from_slice(&totals[..out.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vector dot gives the scalar dot to the bit, whatever the codes and the activations.
    #[test]
    fn the_vector_dot_is_the_scalar_dot() {
        let mut state = 12345u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        };
        for blocks in [1usize, 2, 5, 40] {
            let mut xs = vec![BlockQ2_0::zeros(); blocks];
            let mut ys = vec![BlockQ8_64::zeros(); blocks];
            for x in &mut xs {
                x.d = f16::from_f32(0.01 * (next() as f32 + 1.0));
                x.qs.iter_mut().for_each(|b| *b = next());
            }
            for y in &mut ys {
                y.d = [
                    f16::from_f32(0.003 * (next() as f32 + 1.0)),
                    f16::from_f32(0.003 * (next() as f32 + 1.0)),
                ];
                y.qs.iter_mut().for_each(|q| *q = next() as i8);
            }
            // The extremes of the activation codes, where a saturating add would show.
            ys[0].qs[0] = -128;
            ys[0].qs[1] = 127;
            let want = BlockQ2_0::dot_scalar(&xs, &ys);
            let got = BlockQ2_0::dot(&xs, &ys);
            assert_eq!(got.to_bits(), want.to_bits(), "{blocks} blocks");
        }
    }
}
