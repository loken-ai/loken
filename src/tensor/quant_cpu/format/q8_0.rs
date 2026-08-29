//! q8_0 - thirty-two values, one scale, eight bits each.
//!
//! The simplest of the GGUF block formats and the one every other format's activations are
//! quantised into before a dot product: `y = q . d`, with `q` signed and `d` an `f16` shared
//! by the block. There is no minimum and no sub-block, so an encoder's only decision is the
//! scale, and the scale is forced - the block's largest magnitude has to land on the widest
//! code that exists.
//!
//! `oracle_parity.rs` pins the dequantiser bit-for-bit against ggml's scalar output and
//! measures the encoder's reconstruction against ggml's on the same input.

use super::*;

/// `d . q[i]`, thirty-two values to the block.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}
// The GGUF layout is the struct layout: two bytes of scale then the codes, no padding.
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 2 + QK8_0);

/// The widest code the format has. `i8` reaches -128, but the range is kept symmetric so a
/// value and its negation quantise to opposite codes.
const Q8_0_MAX: f32 = 127.0;

impl BlockFormat for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLOCK_LEN: usize = QK8_0;
    type ActivationBlock = BlockQ8_0;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; QK8_0],
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
            // The scale is not searched: with one scale, no offset and a symmetric code range,
            // any scale smaller than this clips the extreme value and any larger one wastes
            // codes. Where the k-quants have a choice to make, this format has none.
            let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = amax / Q8_0_MAX;
            block.d = f16::from_f32(d);

            // Reciprocal once, not a division per value - and zero when the block is all zero,
            // so the multiply below yields zeros rather than NaN.
            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            for (q, &v) in block.qs.iter_mut().zip(x) {
                *q = (v * inv).round() as i8;
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_q8_0_q8_0(xs, ys);

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        // Both sides share the format, so the codes multiply as integers and the two scales
        // come out of the sum - one f32 rounding per block instead of one per product.
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
