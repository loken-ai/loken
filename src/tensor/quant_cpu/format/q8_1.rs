//! q8_1 - the activation side of the formats that carry a minimum.
//!
//! `y = q . d` over thirty-two values, plus a second scale `s = d . Σq`. That sum exists
//! because a weight format reconstructing as `q.d + m` needs `m . Σ activations` to finish its
//! dot product, and computing it here - once, where the codes are already in registers  -
//! spares q4_1, q5_1 and their kernels a second pass over the block.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub(crate) d: f16,
    pub(crate) s: f16,
    pub(crate) qs: [i8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 4 + QK8_1);

/// The widest code; the block's largest magnitude lands here.
const Q8_1_MAX: f32 = 127.0;

impl BlockFormat for BlockQ8_1 {
    const DTYPE: GgmlDType = GgmlDType::Q8_1;
    const BLOCK_LEN: usize = QK8_1;
    type ActivationBlock = BlockQ8_1;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            s: f16::ZERO,
            qs: [0; QK8_1],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            for (out, &q) in y.iter_mut().zip(&block.qs) {
                *out = q as f32 * d;
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = amax / Q8_1_MAX;
            block.d = f16::from_f32(d);

            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut sum = 0i32;
            for (q, &v) in block.qs.iter_mut().zip(x) {
                *q = (v * inv).round() as i8;
                sum += *q as i32;
            }
            // Stored scaled, so a consumer applies its own minimum with one multiply.
            block.s = f16::from_f32(sum as f32 * d);
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::super::avx::vec_dot_q8_1_q8_1(xs, ys);

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        // Nothing here reads `s`: both sides are plain `q.d`, and the second scale only means
        // something when one side carries a minimum. It is the weight format that spends it.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let dot: i32 =
                    x.qs.iter()
                        .zip(&y.qs)
                        .map(|(&a, &b)| a as i32 * b as i32)
                        .sum();
                dot as f32 * x.d.to_f32() * y.d.to_f32()
            })
            .sum()
    }
}
