//! The AVX2 dot products, and the vocabulary they share.
//!
//! A quantised dot product is the same four steps in every format: widen the
//! packed codes to bytes, multiply them against the activation's int8 codes,
//! widen the products far enough not to overflow, and scale. AVX2 gives that
//! shape one awkward constraint - `maddubs` takes one UNSIGNED and one SIGNED
//! operand - which is why the codes are kept unsigned as long as possible and
//! the zero point is folded in afterwards, per format, out of the inner loop.
//!
//! [`block32`] holds the formats whose block is 32 values and carries its own
//! scale; [`superblock`] the k-quants, 256 values under one pair of scales with
//! sub-block scales quantised in turn.
//!
//! Every kernel here is answerable to its scalar twin: [`parity`] runs the two
//! against each other for all twelve formats, and the scalar side is itself
//! pinned to ggml's output by `oracle_parity`.

// The index arithmetic in these kernels IS the block layout, and the `unsafe fn`s
// wrap intrinsics whose contract is the intrinsic's. Named rather than
// `clippy::all` so anything else here still gets reported.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    clippy::type_complexity,
    clippy::redundant_closure
)]

pub(crate) mod block32;
pub(crate) mod superblock;

#[cfg(test)]
mod parity;

pub(crate) use block32::{
    vec_dot_mxfp4_q8_0, vec_dot_q4_0_q8_0, vec_dot_q4_1_q8_1, vec_dot_q5_0_q8_0, vec_dot_q5_1_q8_1,
    vec_dot_q8_0_q8_0, vec_dot_q8_1_q8_1,
};
pub(crate) use superblock::{
    vec_dot_q2k_q8k, vec_dot_q3k_q8k, vec_dot_q4k_q8k, vec_dot_q5k_q8k, vec_dot_q6k_q8k,
    vec_dot_q8k_q8k,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// -- the four steps --

/// Adjacent i16 lanes summed into i32, then to f32. `madd` against ones is the
/// only single-instruction pairwise horizontal add AVX2 offers.
#[inline(always)]
pub(crate) unsafe fn widen_pairs(x: __m256i) -> __m256 {
    _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), x))
}

/// `Σ` over 32 byte-products where the left side is unsigned and the right
/// signed - the operand asymmetry `maddubs` imposes.
#[inline(always)]
pub(crate) unsafe fn dot_unsigned_signed(unsigned: __m256i, signed: __m256i) -> __m256 {
    widen_pairs(_mm256_maddubs_epi16(unsigned, signed))
}

/// `Σ` over 32 byte-products with both sides signed. The sign of `x` is moved
/// onto `y` and stripped from `x`, which leaves a product identical to the
/// signed one with an operand `maddubs` accepts.
#[inline(always)]
pub(crate) unsafe fn dot_signed(x: __m256i, y: __m256i) -> __m256 {
    dot_unsigned_signed(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x))
}

/// The eight f32 lanes down to one.
#[inline(always)]
pub(crate) unsafe fn lanes_sum(x: __m256) -> f32 {
    let halves = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    _mm_cvtss_f32(_mm_add_ss(pairs, _mm_movehdup_ps(pairs)))
}

/// 16 bytes of packed nibbles into 32 codes: the low nibbles land in lane 0 and
/// the high ones in lane 1, which is the order the block stores its values in.
#[inline(always)]
pub(crate) unsafe fn split_nibbles(packed: *const u8) -> __m256i {
    let bytes = _mm_loadu_si128(packed as *const __m128i);
    let both =
        _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(bytes), _mm_srli_epi16(bytes, 4));
    _mm256_and_si256(_mm256_set1_epi8(0xF), both)
}

/// 32 packed bits into 32 bytes of 0xFF or 0x00: byte k is set iff bit k is.
/// Each byte selects its own bit with a shuffle, then compares against it.
#[inline(always)]
pub(crate) unsafe fn mask_from_bits(bits: &[u8; 4]) -> __m256i {
    let spread = _mm256_set_epi64x(
        0x0303030303030303,
        0x0202020202020202,
        0x0101010101010101,
        0x0000000000000000,
    );
    let one_bit_per_byte = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
    let byte_of = _mm256_shuffle_epi8(_mm256_set1_epi32(u32::from_le_bytes(*bits) as i32), spread);
    _mm256_cmpeq_epi8(
        _mm256_and_si256(byte_of, one_bit_per_byte),
        one_bit_per_byte,
    )
}

/// Two 128-bit halves into one 256-bit register, `high` above `low`.
#[inline(always)]
pub(crate) unsafe fn join_halves(high: __m128i, low: __m128i) -> __m256i {
    _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1)
}

// -- spreading a sub-block scale over the values it governs --
//
// A k-quant's products come out of `maddubs` already paired, and each pair has
// to meet the scale of the sub-block it belongs to. These build the shuffle
// masks that put a scale in front of every product it applies to. They are
// computed, not tabulated: the pattern IS the sub-block size, and a literal
// table would state it a second time in a form nothing checks.

/// Selector `i` puts i8 scales `2i` and `2i+1` in front of eight bytes each  - 
/// q6_K, whose sub-block is 16 values and whose scales are still i8.
#[inline(always)]
pub(crate) unsafe fn scale_pair_over_8_bytes(i: usize) -> __m128i {
    const SELECT: [u8; 128] = {
        let mut s = [0u8; 128];
        let mut j = 0;
        while j < 128 {
            s[j] = (j / 8) as u8;
            j += 1;
        }
        s
    };
    _mm_loadu_si128((SELECT.as_ptr() as *const __m128i).add(i))
}

/// Selector `i` puts i16 scale `i` in front of all sixteen products of a
/// 32-value sub-block - q4_K and q5_K, whose scales are widened to i16 first.
#[inline(always)]
pub(crate) unsafe fn scale_over_16_pairs(i: usize) -> __m256i {
    const SELECT: [u8; 256] = {
        let mut s = [0u8; 256];
        let mut j = 0;
        while j < 256 {
            s[j] = (2 * (j / 32) + j % 2) as u8;
            j += 1;
        }
        s
    };
    _mm256_loadu_si256((SELECT.as_ptr() as *const __m256i).add(i))
}

/// Selector `i` puts i16 scales `2i` and `2i+1` in front of eight products
/// each - the 16-value sub-blocks of q2_K and q3_K.
#[inline(always)]
pub(crate) unsafe fn scale_pair_over_8_pairs(i: usize) -> __m256i {
    const SELECT: [u8; 128] = {
        let mut s = [0u8; 128];
        let mut j = 0;
        while j < 128 {
            s[j] = (2 * (j / 16) + j % 2) as u8;
            j += 1;
        }
        s
    };
    _mm256_loadu_si256((SELECT.as_ptr() as *const __m256i).add(i))
}
