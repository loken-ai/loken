//! The activation side of q2_0: sixty-four values, one scale, eight bits each.
//!
//! q8_0 holds thirty-two values, and the dot engine pairs a weight block with exactly one
//! activation block of its own width, so a weight block of sixty-four needs an activation block
//! of sixty-four. This format is never written to a file.

use super::*;

pub const QK8_64: usize = 64;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_64 {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK8_64],
}

impl BlockFormat for BlockQ8_64 {
    /// Stands in for the format's own id, which a file never carries: the block is reached by
    /// the weight format that asks for it, not by a type id.
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLOCK_LEN: usize = QK8_64;
    type ActivationBlock = BlockQ8_64;

    fn zeros() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK8_64],
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
            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            block.d = d;
            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            for (q, v) in block.qs.iter_mut().zip(x) {
                *q = (v * inv).round() as i8;
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                x.d * y.d
                    * x.qs
                        .iter()
                        .zip(&y.qs)
                        .map(|(a, b)| *a as i32 * *b as i32)
                        .sum::<i32>() as f32
            })
            .sum()
    }
}
