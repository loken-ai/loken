//! q8_K - the activation side of every k-quant dot product.
//!
//! `y = q . d` over a 256-value block, with `d` kept in f32 rather than f16: this format is
//! never written to a file, it is what a row of activations is quantised into just before a
//! matmul, so the two bytes are not worth the rounding.
//!
//! It also carries `bsums`, the sum of each sixteen codes. The formats that reconstruct as
//! `q.d + m` need `Σ activations` per sub-block to apply their minimum, and computing it here
//! once - where the codes are already in registers - spares every one of them a second pass.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 4 + QK_K + QK_K / 8);

/// The widest code, and what the block's extreme value is mapped onto.
const Q8_K_MAX: f32 = 127.0;

impl BlockFormat for BlockQ8K {
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            for (out, &q) in y.iter_mut().zip(&block.qs) {
                *out = q as f32 * block.d;
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            // The extreme keeps its sign and is mapped onto -127, not +127: the codes reach
            // -128, and starting from the negative end is what keeps the AVX2 unpacking of the
            // k-quant kernels free of a sign fixup.
            let extreme = x
                .iter()
                .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            if extreme == 0.0 {
                *block = Self::zeros();
                continue;
            }
            let inv = -Q8_K_MAX / extreme;
            block.d = 1.0 / inv;
            for (q, &v) in block.qs.iter_mut().zip(x) {
                // Only the upper end needs clamping: the extreme lands exactly on -127, but a
                // value of the opposite sign can round past +127.
                *q = nearest_int(inv * v).min(Q8_K_MAX as i32) as i8;
            }
            for (sum, sub) in block.bsums.iter_mut().zip(block.qs.chunks_exact(16)) {
                *sum = sub.iter().map(|&q| q as i16).sum();
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_q8k_q8k(xs, ys);

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        // Both sides are eight-bit codes, so 256 products stay well inside i32 and the two
        // scales come out of the sum.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let dot: i32 =
                    x.qs.iter()
                        .zip(&y.qs)
                        .map(|(&a, &b)| a as i32 * b as i32)
                        .sum();
                dot as f32 * x.d * y.d
            })
            .sum()
    }
}
