//! TQ2_0: 256 ternary values under one f16 scale, 2 bits a value.
//! A code `q` reads `(q - 1) * d`.
//! Byte `m` of each 32-byte group holds values `m`, `m + 32`, `m + 64` and `m + 96` of the
//! group's 128, from its low bits up.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockTq2_0 {
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
}
const _: () = assert!(std::mem::size_of::<BlockTq2_0>() == 66);

impl BlockTq2_0 {
    /// The 256 values of the block as -1, 0 or 1, in order.
    #[inline]
    fn trits(&self) -> [i8; QK_K] {
        let mut out = [0i8; QK_K];
        for (g, group) in self.qs.chunks_exact(32).enumerate() {
            for l in 0..4 {
                for (m, &q) in group.iter().enumerate() {
                    out[128 * g + 32 * l + m] = ((q >> (2 * l)) & 3) as i8 - 1;
                }
            }
        }
        out
    }
}

impl BlockFormat for BlockTq2_0 {
    const DTYPE: GgmlDType = GgmlDType::Tq2_0;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            qs: [0; QK_K / 4],
            d: f16::ZERO,
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            for (out, t) in y.iter_mut().zip(block.trits()) {
                *out = t as f32 * d;
            }
        }
    }

    /// The scale is the largest magnitude.
    /// Each value takes the nearest of -1, 0 and 1 at that scale.
    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            let d = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            block.d = f16::from_f32(d);
            for (g, group) in block.qs.chunks_exact_mut(32).enumerate() {
                for (m, q) in group.iter_mut().enumerate() {
                    *q = (0..4).fold(0u8, |acc, n| {
                        let t = ((x[128 * g + 32 * n + m] * id).round() as i32 + 1) as u8;
                        acc | ((t & 3) << (2 * n))
                    });
                }
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    /// One integer sum a block, taken under the activation's scale times the block's scale.
    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        let mut total = 0f32;
        for (x, y) in xs.iter().zip(ys) {
            let sum: i32 = x
                .trits()
                .iter()
                .zip(&y.qs)
                .map(|(&t, &a)| t as i32 * a as i32)
                .sum();
            total += sum as f32 * (y.d * x.d.to_f32());
        }
        total
    }
}
