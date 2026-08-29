//! The unquantised types, as block formats of one value.
//!
//! `f32`, `f16` and `bf16` implement the same trait as the GGUF blocks so a matmul can take
//! any weight dtype without branching on it. Their "block" is a single value, they have no
//! scale to fit, and `COPIES_VERBATIM` says the loader may move the bytes without going through
//! f32 at all.
//!
//! The dot products come from `cpu`, which is where this crate's vectorised float kernels are.

use super::*;

/// The three impls differ in one thing each: how a value converts to and from f32. Written
/// once rather than three times, because identical bodies are how one of them ends up quietly
/// out of step with the others.
///
/// The `copy` arm is f32's, where the conversion is the identity and the slices move whole;
/// the `convert` arm is for the narrower types, whose `half` conversions are element-wise.
macro_rules! plain_float {
    ($ty:ty, $dtype:ident, $zero:expr, $dot:path, copy) => {
        plain_float!(@common $ty, $dtype, $zero, $dot);
        impl PlainFloatConv for $ty {
            fn write(xs: &[f32], ys: &mut [Self]) {
                ys.copy_from_slice(xs);
            }
            fn read(xs: &[Self], ys: &mut [f32]) {
                ys.copy_from_slice(xs);
            }
        }
    };
    ($ty:ty, $dtype:ident, $zero:expr, $dot:path, convert) => {
        plain_float!(@common $ty, $dtype, $zero, $dot);
        impl PlainFloatConv for $ty {
            fn write(xs: &[f32], ys: &mut [Self]) {
                ys.convert_from_f32_slice(xs);
            }
            fn read(xs: &[Self], ys: &mut [f32]) {
                xs.convert_to_f32_slice(ys);
            }
        }
    };
    (@common $ty:ty, $dtype:ident, $zero:expr, $dot:path) => {
        impl BlockFormat for $ty {
            const DTYPE: GgmlDType = GgmlDType::$dtype;
            const BLOCK_LEN: usize = 1;
            const COPIES_VERBATIM: bool = true;
            type ActivationBlock = $ty;

            fn zeros() -> Self {
                $zero
            }

            fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
                Self::dot_scalar(xs, ys)
            }

            fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
                let n = xs.len().min(ys.len());
                let mut acc = 0f32;
                // The kernel takes raw pointers and a length; `n` is derived from the slices
                // themselves, so there is no way for a caller to disagree with it.
                unsafe { $dot(xs.as_ptr(), ys.as_ptr(), &mut acc, n) };
                acc
            }

            fn quantize(xs: &[f32], ys: &mut [Self]) {
                debug_assert_eq!(xs.len(), ys.len(), "quantize: {} into {}", xs.len(), ys.len());
                <Self as PlainFloatConv>::write(xs, ys);
            }

            fn dequantize(xs: &[Self], ys: &mut [f32]) {
                debug_assert_eq!(xs.len(), ys.len(), "dequantize: {} into {}", xs.len(), ys.len());
                <Self as PlainFloatConv>::read(xs, ys);
            }

            fn copy_verbatim(xs: &[f32], ys: &mut [Self]) {
                Self::quantize(xs, ys)
            }
        }
    };
}

/// The one thing that differs between the three. Private: it exists to keep the length checks
/// and the dot product in one place above, not to be a public capability.
trait PlainFloatConv: Sized {
    fn write(xs: &[f32], ys: &mut [Self]);
    fn read(xs: &[Self], ys: &mut [f32]);
}

plain_float!(f32, F32, 0.0, cpu::vec_dot_f32, copy);
plain_float!(f16, F16, f16::ZERO, cpu::vec_dot_f16, convert);
plain_float!(bf16, BF16, bf16::ZERO, cpu::vec_dot_bf16, convert);
