//! The IQ2_XXS weight block against a q8_K activation block, vectorised.
//!
//! A sub-block of 32 weights is four grid entries of eight small unsigned values, each entry
//! with an eight-sign pattern, all under one scale. Its product with 32 signed activations is an
//! integer: the signs move onto the activations, the grid values multiply them pairwise into 16
//! bits, and the pairs sum into 32 bits. The sub-block scale and the two block scales apply once
//! to that integer, so no weight is ever widened to a float.

use super::*;

/// The sign of each of the eight values of an entry, by sign selector: -1 where the pattern's
/// bit is set.
#[allow(dead_code)]
const SIGNS: [[i8; 8]; 128] = {
    let mut t = [[1i8; 8]; 128];
    let mut i = 0;
    while i < 128 {
        let mut j = 0;
        while j < 8 {
            if KSIGNS_IQ2XS[i] & KMASK_IQ2XS[j] != 0 {
                t[i][j] = -1;
            }
            j += 1;
        }
        i += 1;
    }
    t
};

/// The eight signs of an entry packed little-endian into a u64: byte `j` is `0xff` where the
/// pattern negates, `0x01` where it keeps. The same values `SIGNS` holds, in the layout a SIMD
/// lane loads directly, so the AVX2 path skips building a `[i8; 32]` array before the dot.
const SIGNS_U64: [u64; 128] = {
    let mut t = [0u64; 128];
    let mut i = 0;
    while i < 128 {
        let mut v = 0u64;
        let mut j = 0;
        while j < 8 {
            let b: u64 = if KSIGNS_IQ2XS[i] & KMASK_IQ2XS[j] != 0 {
                0xff
            } else {
                0x01
            };
            v |= b << (8 * j);
            j += 1;
        }
        t[i] = v;
        i += 1;
    }
    t
};

/// One sub-block's integer dot, its grid entries and sign patterns loaded straight from the
/// tables into the SIMD lanes: `IQ2XXS_GRID` is already eight packed bytes per entry and
/// `SIGNS_U64` mirrors it, so no `[u8; 32]`/`[i8; 32]` is assembled and copied first. Byte-for-byte
/// the same product as `sub_block` + `sub_block_dot` on little-endian x86.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sub_block_dot_fused_avx2(q: &[u16], act: *const i8) -> i32 {
    use core::arch::x86_64::*;
    let lo = q[0] as u32 | ((q[1] as u32) << 16);
    let hi = q[2] as u32 | ((q[3] as u32) << 16);
    let g = _mm256_set_epi64x(
        IQ2XXS_GRID[((lo >> 24) & 0xff) as usize] as i64,
        IQ2XXS_GRID[((lo >> 16) & 0xff) as usize] as i64,
        IQ2XXS_GRID[((lo >> 8) & 0xff) as usize] as i64,
        IQ2XXS_GRID[(lo & 0xff) as usize] as i64,
    );
    let s = _mm256_set_epi64x(
        SIGNS_U64[((hi >> 21) & 127) as usize] as i64,
        SIGNS_U64[((hi >> 14) & 127) as usize] as i64,
        SIGNS_U64[((hi >> 7) & 127) as usize] as i64,
        SIGNS_U64[(hi & 127) as usize] as i64,
    );
    let a = _mm256_sign_epi8(_mm256_loadu_si256(act as *const __m256i), s);
    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(g, a), _mm256_set1_epi16(1));
    let lo128 = _mm256_castsi256_si128(p);
    let hi128 = _mm256_extracti128_si256::<1>(p);
    let s4 = _mm_add_epi32(lo128, hi128);
    let s2 = _mm_add_epi32(s4, _mm_shuffle_epi32::<0b01_00_11_10>(s4));
    let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32::<0b10_11_00_01>(s2));
    _mm_cvtsi128_si32(s1)
}

/// The grid values and signs of one sub-block, in the order its 32 weights run.
#[allow(dead_code)]
#[inline(always)]
fn sub_block(q: &[u16]) -> ([u8; 32], [i8; 32], f32) {
    let lo = q[0] as u32 | ((q[1] as u32) << 16);
    let hi = q[2] as u32 | ((q[3] as u32) << 16);
    let mut grid = [0u8; 32];
    let mut signs = [0i8; 32];
    for l in 0..4 {
        grid[8 * l..8 * l + 8]
            .copy_from_slice(&IQ2XXS_GRID[((lo >> (8 * l)) & 0xff) as usize].to_le_bytes());
        signs[8 * l..8 * l + 8].copy_from_slice(&SIGNS[((hi >> (7 * l)) & 127) as usize]);
    }
    (grid, signs, (0.5 + (hi >> 28) as f32) * 0.25)
}

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sub_block_dot_avx2(grid: &[u8; 32], signs: &[i8; 32], act: *const i8) -> i32 {
    use core::arch::x86_64::*;
    let g = _mm256_loadu_si256(grid.as_ptr() as *const __m256i);
    let s = _mm256_loadu_si256(signs.as_ptr() as *const __m256i);
    let a = _mm256_sign_epi8(_mm256_loadu_si256(act as *const __m256i), s);
    // The grid values are small, so a pair of products fits the 16-bit lanes.
    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(g, a), _mm256_set1_epi16(1));
    let lo = _mm256_castsi256_si128(p);
    let hi = _mm256_extracti128_si256::<1>(p);
    let s4 = _mm_add_epi32(lo, hi);
    let s2 = _mm_add_epi32(s4, _mm_shuffle_epi32::<0b01_00_11_10>(s4));
    let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32::<0b10_11_00_01>(s2));
    _mm_cvtsi128_si32(s1)
}

#[allow(dead_code)]
#[inline(always)]
fn sub_block_dot(grid: &[u8; 32], signs: &[i8; 32], act: &[i8]) -> i32 {
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        // Safety: `act` holds the sub-block's 32 activations.
        unsafe { sub_block_dot_avx2(grid, signs, act.as_ptr()) }
    }
    #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
    {
        grid.iter()
            .zip(signs)
            .zip(act)
            .map(|((&g, &s), &a)| g as i32 * s as i32 * a as i32)
            .sum()
    }
}

/// The dot product of IQ2_XXS weight blocks with q8_K activation blocks.
pub(crate) fn dot(xs: &[BlockIq2Xxs], ys: &[BlockQ8K]) -> f32 {
    xs.iter()
        .zip(ys)
        .map(|(x, y)| {
            let mut acc = 0f32;
            for ib32 in 0..QK_K / 32 {
                let q = &x.qs[4 * ib32..4 * ib32 + 4];
                let act = &y.qs[32 * ib32..32 * ib32 + 32];
                // The scale rides in the top four bits of the sign half of the sub-block.
                let hi = q[2] as u32 | ((q[3] as u32) << 16);
                let scale = (0.5 + (hi >> 28) as f32) * 0.25;
                #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
                // Safety: `act` is the sub-block's 32 activations.
                let sum = unsafe { sub_block_dot_fused_avx2(q, act.as_ptr()) };
                #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
                let sum = {
                    let (grid, signs, _) = sub_block(q);
                    sub_block_dot(&grid, &signs, act)
                };
                acc += scale * sum as f32;
            }
            x.d.to_f32() * y.d * acc
        })
        .sum()
}
