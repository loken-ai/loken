//! q5_0 - thirty-two values, one scale, five bits each.
//!
//! `y = (q - 16) . d`, the same shape as q4_0 with one more bit of code. The fifth bit does
//! not fit in the nibble array, so it lives in a separate 32-bit word where bit `v` belongs to
//! value `v`: the code for value `v` is `qs[v mod 16]`'s nibble with that bit stuck on top.
//!
//! The nibble packing is q4_0's: byte `j` holds value `j` low and value `j + 16` high, so the
//! two halves of a block are interleaved rather than consecutive.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 2 + 4 + QK5_0 / 2);

/// Code `c` means `c - Q5_0_BIAS`; five bits reach 0..31, so -16..15.
const Q5_0_BIAS: i32 = 16;

impl BlockFormat for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLOCK_LEN: usize = QK5_0;
    type ActivationBlock = BlockQ8_0;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qh: [0; 4],
            qs: [0; QK5_0 / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        const HALF: usize = QK5_0 / 2;
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            let qh = u32::from_le_bytes(block.qh);
            let top = |v: usize| ((qh >> v) & 1) << 4;
            let (low, high) = y.split_at_mut(HALF);
            for (j, ((lo, hi), &packed)) in low.iter_mut().zip(high).zip(&block.qs).enumerate() {
                let a = ((packed & 0x0F) as u32 | top(j)) as i32 - Q5_0_BIAS;
                let b = ((packed >> 4) as u32 | top(j + HALF)) as i32 - Q5_0_BIAS;
                *lo = a as f32 * d;
                *hi = b as f32 * d;
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        const HALF: usize = QK5_0 / 2;
        for (block, x) in blocks_with_input(xs, ys) {
            // As in q4_0: the extreme keeps its sign because only -16 is representable.
            let extreme = x
                .iter()
                .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            let d = extreme / -(Q5_0_BIAS as f32);
            block.d = f16::from_f32(d);

            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            let code = |v: f32| ((v * inv + Q5_0_BIAS as f32).round() as i32).clamp(0, 31) as u32;
            let mut qh = 0u32;
            let (low, high) = x.split_at(HALF);
            for (j, ((&lo, &hi), packed)) in
                low.iter().zip(high).zip(block.qs.iter_mut()).enumerate()
            {
                let (a, b) = (code(lo), code(hi));
                *packed = (a & 0x0F) as u8 | (((b & 0x0F) as u8) << 4);
                qh |= (a >> 4) << j;
                qh |= (b >> 4) << (j + HALF);
            }
            block.qh = qh.to_le_bytes();
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_q5_0_q8_0(xs, ys);

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const HALF: usize = QK5_0 / 2;
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let qh = u32::from_le_bytes(x.qh);
                let top = |v: usize| ((qh >> v) & 1) << 4;
                let (low, high) = y.qs.split_at(HALF);
                let dot: i32 =
                    x.qs.iter()
                        .zip(low)
                        .zip(high)
                        .enumerate()
                        .map(|(j, ((&packed, &a), &b))| {
                            let p = ((packed & 0x0F) as u32 | top(j)) as i32 - Q5_0_BIAS;
                            let q = ((packed >> 4) as u32 | top(j + HALF)) as i32 - Q5_0_BIAS;
                            p * a as i32 + q * b as i32
                        })
                        .sum();
                dot as f32 * x.d.to_f32() * y.d.to_f32()
            })
            .sum()
    }
}
