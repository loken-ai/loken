//! TQ1_0: 256 ternary values under one f16 scale, 5 values a byte.
//! A byte holds 5 trits as a fraction of 243 scaled to 256, the first trit the most significant.
//! Bytes 0..32 hold values 0..160, trit `n` of byte `m` being value `32 n + m`.
//! Bytes 32..48 hold values 160..240, trit `n` of byte `m` being value `160 + 16 n + m`.
//! The 4 `qh` bytes hold values 240..256 as 4 trits each, trit `n` of byte `j` being value
//! `240 + 4 n + j`.

use super::*;

/// Bytes of 5 trits a block holds.
const TQ1_QS: usize = (QK_K - 4 * QK_K / 64) / 5;
/// Bytes of 4 trits a block holds.
const TQ1_QH: usize = QK_K / 64;
/// Bytes of 5 trits read in groups of 32.
const TQ1_QS_WIDE: usize = TQ1_QS - TQ1_QS % 32;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockTq1_0 {
    pub(crate) qs: [u8; TQ1_QS],
    pub(crate) qh: [u8; TQ1_QH],
    pub(crate) d: f16,
}
const _: () = assert!(std::mem::size_of::<BlockTq1_0>() == 54);

/// Trit `n` of a packed byte, as -1, 0 or 1.
#[inline]
fn trit(q: u8, n: usize) -> i8 {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let shifted = q.wrapping_mul(POW3[n]);
    ((shifted as u16 * 3) >> 8) as i8 - 1
}

/// `count` trits (0, 1 or 2) as one byte: their base-3 number, padded to 5 trits, scaled from
/// 243 to 256 and rounded up.
#[inline]
fn pack(trits: impl Iterator<Item = u8>, count: usize) -> u8 {
    let mut q = trits.fold(0u16, |acc, t| acc * 3 + t as u16);
    for _ in count..5 {
        q *= 3;
    }
    (q * 256).div_ceil(243) as u8
}

impl BlockTq1_0 {
    /// The 256 values of the block as -1, 0 or 1, in order.
    #[inline]
    fn trits(&self) -> [i8; QK_K] {
        let mut out = [0i8; QK_K];
        let mut at = 0;
        for group in self.qs[..TQ1_QS_WIDE].chunks_exact(32) {
            for n in 0..5 {
                for &q in group {
                    out[at] = trit(q, n);
                    at += 1;
                }
            }
        }
        for group in self.qs[TQ1_QS_WIDE..].chunks_exact(16) {
            for n in 0..5 {
                for &q in group {
                    out[at] = trit(q, n);
                    at += 1;
                }
            }
        }
        for n in 0..4 {
            for &q in &self.qh {
                out[at] = trit(q, n);
                at += 1;
            }
        }
        out
    }
}

impl BlockFormat for BlockTq1_0 {
    const DTYPE: GgmlDType = GgmlDType::Tq1_0;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            qs: [0; TQ1_QS],
            qh: [0; TQ1_QH],
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
            let code = |i: usize| ((x[i] * id).round() as i32 + 1) as u8;
            for (m, q) in block.qs[..TQ1_QS_WIDE].iter_mut().enumerate() {
                *q = pack((0..5).map(|n| code(m + 32 * n)), 5);
            }
            let base = 5 * TQ1_QS_WIDE;
            for (m, q) in block.qs[TQ1_QS_WIDE..].iter_mut().enumerate() {
                *q = pack((0..5).map(|n| code(base + m + 16 * n)), 5);
            }
            let base = 5 * TQ1_QS;
            for (j, q) in block.qh.iter_mut().enumerate() {
                *q = pack((0..4).map(|n| code(base + j + TQ1_QH * n)), 4);
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    /// One integer sum a block, taken under the block's scale times the activation's scale.
    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        let mut total = 0f32;
        for (x, y) in xs.iter().zip(ys) {
            let sum: i32 = x
                .trits()
                .iter()
                .zip(&y.qs)
                .map(|(&t, &a)| t as i32 * a as i32)
                .sum();
            total += sum as f32 * (x.d.to_f32() * y.d);
        }
        total
    }
}
