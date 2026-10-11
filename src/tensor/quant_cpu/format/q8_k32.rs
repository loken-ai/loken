//! The activation side of the codebook formats: two hundred and fifty-six values in the eight
//! runs the card quantises them in, thirty-two values to a run with a scale of its own, eight
//! bits each.
//!
//! The card dots these formats against q8_1 blocks of thirty-two, each under its own scale, and
//! a weight block of `QK_K` values pairs with one activation block of its width: an activation
//! block of `QK_K` values with a scale per run is what makes an expert computed on either side
//! give the same values. This format is never written to a file.

use super::*;

/// Runs a block holds, each with a scale of its own.
pub const Q8K32_RUNS: usize = QK_K / 32;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8K32 {
    pub(crate) d: [f16; Q8K32_RUNS],
    pub(crate) qs: [i8; QK_K],
}

impl BlockFormat for BlockQ8K32 {
    /// Stands in for the format's own id, which a file never carries: the block is reached by
    /// the weight format that asks for it, not by a type id.
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K32;

    fn zeros() -> Self {
        Self {
            d: [f16::ZERO; Q8K32_RUNS],
            qs: [0; QK_K],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            for (ib, run) in y.chunks_mut(32).enumerate() {
                let d = block.d[ib].to_f32();
                for (out, &q) in run.iter_mut().zip(&block.qs[32 * ib..]) {
                    *out = q as f32 * d;
                }
            }
        }
    }

    /// Each run as a q8_0 block: the scale puts the extreme at the end of the code range.
    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            for (ib, run) in x.chunks(32).enumerate() {
                let amax = run.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                block.d[ib] = f16::from_f32(d);
                let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
                for (q, v) in block.qs[32 * ib..32 * ib + 32].iter_mut().zip(run) {
                    *q = (v * inv).round() as i8;
                }
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        let mut total = 0f32;
        for (x, y) in xs.iter().zip(ys) {
            for ib in 0..Q8K32_RUNS {
                let at = 32 * ib;
                let dot: i32 = x.qs[at..at + 32]
                    .iter()
                    .zip(&y.qs[at..at + 32])
                    .map(|(a, b)| *a as i32 * *b as i32)
                    .sum();
                total += dot as f32 * x.d[ib].to_f32() * y.d[ib].to_f32();
            }
        }
        total
    }
}
