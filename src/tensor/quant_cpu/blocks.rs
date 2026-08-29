//! Block geometry, shared by every format.
//!
//! What is left here after each format took its own struct: the element counts. `QK_K` is the
//! k-quant super-block, the `QK*_*` the legacy block sizes - facts about the FILE FORMAT that
//! several formats quote, which is why they stay in one place rather than being repeated.

use super::*;

// ===========================================================================
// (only the module paths changed: avx is a submodule here, float dot
//  lives in the local `cpu` module, Result/Error are the native ones)
// ===========================================================================

// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;
pub const QK_MXFP4: usize = 32;

/// OCP MXFP4 E2M1 code -> value LUT (ggml `kvalues_mxfp4`). These are the
/// representable magnitudes x2 (the x0.5 is folded into the E8M0 scale via
/// `e8m0_to_fp32_half`), so dequant = `KVALUES_MXFP4[code] * e8m0_to_fp32_half(e)`.
pub(crate) const KVALUES_MXFP4: [i8; 16] =
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// ggml `ggml_e8m0_to_fp32_half`: 0.5 . 2^(e-127), matching the KVALUES_MXFP4
/// (x2) convention. Denormal patterns for e<2, normalized exponent otherwise.
#[inline]
pub(crate) fn e8m0_to_fp32_half(e: u8) -> f32 {
    let bits: u32 = if e < 2 {
        0x0020_0000u32 << e
    } else {
        (e as u32 - 1) << 23
    };
    f32::from_bits(bits)
}

pub trait BlockFormat: Sized + Clone + Send + Sync {
    const DTYPE: GgmlDType;
    const BLOCK_LEN: usize;
    const COPIES_VERBATIM: bool = false;
    type ActivationBlock: BlockFormat;

    /// The block a pre-allocated buffer is filled with before anything is written into it.
    ///
    /// Each format states its own rather than inheriting one: "all bytes zero" is a claim
    /// about a type's layout that only that type can make.
    fn zeros() -> Self;
    fn dequantize(xs: &[Self], ys: &mut [f32]);
    fn quantize(xs: &[f32], ys: &mut [Self]);
    fn quantize_guided(_xs: &[f32], _ys: &mut [Self], _imatrix_weights: &[f32], _n_per_row: usize) {
        panic!("`quantize_guided` is unimplemented for {:?}", Self::DTYPE);
    }

    fn copy_verbatim(_xs: &[f32], _ys: &mut [Self]) {}

    /// Dot product used as a building block for quantized mat-mul.
    /// n is the number of elements to be considered.
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32;

    /// Generic implementation of the dot product without simd optimizations.
    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32;
}

/// The eight `(scale, min)` pairs a q4_K or q5_K block packs into twelve bytes.
///
/// Sixteen six-bit values, ninety-six bits, twelve bytes - and the packing is not uniform.
/// The first four scales occupy the low six bits of bytes 0..4, the first four mins the low
/// six bits of bytes 4..8. That leaves two spare bits above each of those eight, and four
/// bytes (8..12) untouched; the last four scales and mins are split across both, taking
/// their low nibble from bytes 8..12 and their high two bits from the room left above their
/// own first-four counterpart.
///
/// Unpacked once into a table rather than fetched inside the decode loop: which bits hold a
/// scale is a property of the BLOCK, not of the sub-block being decoded.
pub(crate) fn kquant_scales_and_mins(packed: &[u8; K_SCALE_SIZE]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for i in 0..4 {
        scales[i] = packed[i] & 0x3F;
        mins[i] = packed[i + 4] & 0x3F;
    }
    for i in 4..8 {
        let low = packed[i + 4];
        scales[i] = (low & 0x0F) | ((packed[i - 4] >> 6) << 4);
        mins[i] = (low >> 4) | ((packed[i] >> 6) << 4);
    }
    (scales, mins)
}

/// The inverse of [`kquant_scales_and_mins`]: eight six-bit scales and eight six-bit minimums
/// folded into twelve bytes.
///
/// The first four of each pair sit whole in their own byte; the last four are split, their low
/// nibbles sharing bytes 8..12 and their top two bits riding in the spare high bits of the
/// bytes the first four occupy. q4_K and q5_K use the identical arrangement, which is why this
/// is written once rather than in each of their encoders.
pub(crate) fn pack_kquant_scales_and_mins(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; K_SCALE_SIZE] {
    let mut packed = [0u8; K_SCALE_SIZE];
    for i in 0..4 {
        packed[i] = scales[i] & 0x3F;
        packed[i + 4] = mins[i] & 0x3F;
    }
    for i in 4..8 {
        packed[i + 4] = (scales[i] & 0x0F) | ((mins[i] & 0x0F) << 4);
        packed[i - 4] |= (scales[i] >> 4) << 6;
        packed[i] |= (mins[i] >> 4) << 6;
    }
    packed
}

/// Each block of `ys` paired with the `T::BLOCK_LEN` values it will hold.
///
/// The pairing is by `chunks_exact`, so a caller whose lengths disagree gets the shorter of
/// the two silently - hence the assertion, which says which way round the mismatch is.
#[inline]
pub(super) fn blocks_with_input<'a, 'b, T: BlockFormat>(
    xs: &'b [f32],
    ys: &'a mut [T],
) -> impl Iterator<Item = (&'a mut T, &'b [f32])> {
    debug_assert_eq!(
        xs.len() / T::BLOCK_LEN,
        ys.len(),
        "quantize {:?}: {} values fill {} blocks, but {} were given",
        T::DTYPE,
        xs.len(),
        xs.len() / T::BLOCK_LEN,
        ys.len(),
    );
    ys.iter_mut().zip(xs.chunks_exact(T::BLOCK_LEN))
}

/// Each block of `xs` paired with the `T::BLOCK_LEN` values it expands into.
#[inline]
pub(super) fn blocks_with_output<'a, 'b, T: BlockFormat>(
    xs: &'a [T],
    ys: &'b mut [f32],
) -> impl Iterator<Item = (&'a T, &'b mut [f32])> {
    debug_assert_eq!(
        xs.len() * T::BLOCK_LEN,
        ys.len(),
        "dequantize {:?}: {} blocks expand to {} values, but room for {} was given",
        T::DTYPE,
        xs.len(),
        xs.len() * T::BLOCK_LEN,
        ys.len(),
    );
    xs.iter().zip(ys.chunks_exact_mut(T::BLOCK_LEN))
}
