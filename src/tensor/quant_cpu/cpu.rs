//! The dot products of the unquantised carriers: `f32`, `f16`, `bf16`.
//!
//! Thirty-two values per iteration in four live accumulators. Four rather than
//! one because a single accumulator serialises on the latency of the add; four
//! independent chains keep the port busy, and thirty-two f32 is two cache lines.
//!
//! The three differ in exactly two places - how sixteen bytes become eight
//! f32s, and the order the four accumulators are folded in. That second one is
//! not a free choice: float addition does not associate, so the order IS part
//! of the answer, and each carrier keeps the one it is pinned to.
//!
//! Whatever the carrier, the tail past the last whole thirty-two runs scalar,
//! and every one of these WRITES its result rather than adding to it.

#![allow(clippy::missing_safety_doc)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
mod vectorised {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use half::{bf16, f16};

    /// f32 lanes in one AVX register.
    const LANES: usize = 8;
    /// Independent accumulator chains held live across the loop.
    const CHAINS: usize = 4;
    /// Values consumed per iteration.
    const STRIDE: usize = LANES * CHAINS;

    /// The eight lanes of one register to one f32.
    #[inline(always)]
    unsafe fn across_lanes(v: __m256) -> f32 {
        let halves = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let pairs = _mm_hadd_ps(halves, halves);
        _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
    }

    /// `(a0 + a1) + (a2 + a3)` - adjacent chains first.
    #[inline(always)]
    unsafe fn fold_adjacent(a: [__m256; CHAINS]) -> f32 {
        across_lanes(_mm256_add_ps(
            _mm256_add_ps(a[0], a[1]),
            _mm256_add_ps(a[2], a[3]),
        ))
    }

    /// `(a0 + a2) + (a1 + a3)` - chains half the array apart first.
    ///
    /// A different order from [`fold_adjacent`], and a different answer in the
    /// low bits. Both are in use because both are pinned: the f32 carrier is
    /// checked against a reference that folds one way and the narrow carriers
    /// against one that folds the other. Collapsing them to a single order
    /// would move results that other tests hold fixed.
    #[inline(always)]
    unsafe fn fold_halving(a: [__m256; CHAINS]) -> f32 {
        across_lanes(_mm256_add_ps(
            _mm256_add_ps(a[0], a[2]),
            _mm256_add_ps(a[1], a[3]),
        ))
    }

    /// Eight f32 straight from memory.
    #[inline(always)]
    unsafe fn load_f32(p: *const f32) -> __m256 {
        _mm256_loadu_ps(p)
    }

    /// Eight f16. `f16c` decodes them in one instruction; without it, one at a
    /// time through the same conversion the scalar tail uses.
    #[cfg(target_feature = "f16c")]
    #[inline(always)]
    unsafe fn load_f16(p: *const f16) -> __m256 {
        _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
    }
    #[cfg(not(target_feature = "f16c"))]
    #[inline(always)]
    unsafe fn load_f16(p: *const f16) -> __m256 {
        let mut tmp = [0.0f32; LANES];
        for (i, out) in tmp.iter_mut().enumerate() {
            *out = (*p.add(i)).to_f32();
        }
        _mm256_loadu_ps(tmp.as_ptr())
    }

    /// Eight bf16. A bf16 IS the top half of an f32, so widening is a
    /// zero-extend and a shift - exact, and needing no converter instruction.
    ///
    /// This read `_mm256_cvtph_ps`, which decodes f16: the two formats share a
    /// width and nothing else, and 1.0 came back as 1.875.
    #[inline(always)]
    unsafe fn load_bf16(p: *const bf16) -> __m256 {
        let raw = _mm_loadu_si128(p as *const __m128i);
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16))
    }

    /// One carrier's dot product: the whole-stride part vectorised, the
    /// remainder scalar, written to `c`.
    macro_rules! carrier {
        ($name:ident, $ty:ty, $load:ident, $fold:ident, $widen:expr) => {
            #[inline(always)]
            pub unsafe fn $name(a: *const $ty, b: *const $ty, c: *mut f32, k: usize) {
                let whole = k & !(STRIDE - 1);
                let mut chain = [_mm256_setzero_ps(); CHAINS];
                for i in (0..whole).step_by(STRIDE) {
                    for (j, acc) in chain.iter_mut().enumerate() {
                        let x = $load(a.add(i + j * LANES));
                        let y = $load(b.add(i + j * LANES));
                        *acc = _mm256_add_ps(_mm256_mul_ps(x, y), *acc);
                    }
                }
                let widen: fn($ty) -> f32 = $widen;
                let mut sum = $fold(chain);
                for i in whole..k {
                    sum += widen(*a.add(i)) * widen(*b.add(i));
                }
                *c = sum;
            }
        };
    }

    carrier!(vec_dot_f32, f32, load_f32, fold_adjacent, |v| v);
    carrier!(vec_dot_f16, f16, load_f16, fold_halving, |v: f16| v
        .to_f32());
    carrier!(vec_dot_bf16, bf16, load_bf16, fold_halving, |v: bf16| v
        .to_f32());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
pub(super) use vectorised::{vec_dot_bf16, vec_dot_f16, vec_dot_f32};

/// No AVX2: the same contract, one value at a time.
///
/// `*c` is WRITTEN, as above. The f32 arm of this used to `+=` into it without
/// setting it first - a different function from the one it stands in for, which
/// survived only because every caller happened to pass a zero.
#[cfg(not(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
)))]
mod scalar_only {
    use super::{bf16, f16};

    macro_rules! carrier {
        ($name:ident, $ty:ty, $widen:expr) => {
            pub unsafe fn $name(a: *const $ty, b: *const $ty, c: *mut f32, k: usize) {
                let widen: fn($ty) -> f32 = $widen;
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += widen(*a.add(i)) * widen(*b.add(i));
                }
                *c = sum;
            }
        };
    }

    carrier!(vec_dot_f32, f32, |v| v);
    carrier!(vec_dot_f16, f16, |v: f16| v.to_f32());
    carrier!(vec_dot_bf16, bf16, |v: bf16| v.to_f32());
}

#[cfg(not(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
)))]
pub(super) use scalar_only::{vec_dot_bf16, vec_dot_f16, vec_dot_f32};
