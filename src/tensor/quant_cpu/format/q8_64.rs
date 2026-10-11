//! The activation side of q2_0: sixty-four values as the two halves the card quantises them in,
//! thirty-two values to a half with a scale of its own, eight bits each.
//!
//! q8_0 holds thirty-two values, and the dot engine pairs a weight block with exactly one
//! activation block of its own width, so a weight block of sixty-four needs an activation block
//! of sixty-four. Its halves are quantised as q8_0 blocks are, which is how the card quantises
//! the same row: an expert computed on either side then gives the same values. This format is
//! never written to a file.

use super::*;

pub const QK8_64: usize = 64;
/// Values a half holds, with a scale of its own.
pub const QK8_64_HALF: usize = QK8_64 / 2;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_64 {
    pub(crate) d: [f16; 2],
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
            d: [f16::ZERO; 2],
            qs: [0; QK8_64],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            for (h, half) in y.chunks_mut(QK8_64_HALF).enumerate() {
                let d = block.d[h].to_f32();
                for (out, &q) in half.iter_mut().zip(&block.qs[h * QK8_64_HALF..]) {
                    *out = q as f32 * d;
                }
            }
        }
    }

    /// Each half as a q8_0 block: the scale puts the extreme at the end of the code range.
    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            for (h, half) in x.chunks(QK8_64_HALF).enumerate() {
                let amax = half.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                block.d[h] = f16::from_f32(d);
                let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
                for (q, v) in block.qs[h * QK8_64_HALF..(h + 1) * QK8_64_HALF]
                    .iter_mut()
                    .zip(half)
                {
                    *q = (v * inv).round() as i8;
                }
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
                (0..2)
                    .map(|h| {
                        let at = h * QK8_64_HALF;
                        let dot: i32 = x.qs[at..at + QK8_64_HALF]
                            .iter()
                            .zip(&y.qs[at..at + QK8_64_HALF])
                            .map(|(a, b)| *a as i32 * *b as i32)
                            .sum();
                        dot as f32 * x.d[h].to_f32() * y.d[h].to_f32()
                    })
                    .sum::<f32>()
            })
            .sum()
    }
}
