//! q5_1 - thirty-two values, a scale and a minimum, five bits each.
//!
//! `y = q . d + m`: q4_1's shape with one more bit of code, or q5_0's with an explicit minimum
//! instead of a fixed bias. The fifth bit lives apart in a 32-bit word where bit `v` belongs
//! to value `v`, and the nibble array uses the same split as q4_0 - byte `j` holds value `j`
//! low and value `j + 16` high.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 4 + 4 + QK5_1 / 2);

/// The largest code five bits reach; the interval `[min, max]` is divided into this many steps.
const Q5_1_MAX: f32 = 31.0;

impl BlockFormat for BlockQ5_1 {
    const DTYPE: GgmlDType = GgmlDType::Q5_1;
    const BLOCK_LEN: usize = QK5_1;
    type ActivationBlock = BlockQ8_1;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            m: f16::ZERO,
            qh: [0; 4],
            qs: [0; QK5_1 / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        const HALF: usize = QK5_1 / 2;
        for (block, y) in blocks_with_output(xs, ys) {
            let (d, m) = (block.d.to_f32(), block.m.to_f32());
            let qh = u32::from_le_bytes(block.qh);
            let top = |v: usize| ((qh >> v) & 1) << 4;
            let (low, high) = y.split_at_mut(HALF);
            for (j, ((lo, hi), &packed)) in low.iter_mut().zip(high).zip(&block.qs).enumerate() {
                *lo = ((packed & 0x0F) as u32 | top(j)) as f32 * d + m;
                *hi = ((packed >> 4) as u32 | top(j + HALF)) as f32 * d + m;
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        const HALF: usize = QK5_1 / 2;
        for (block, x) in blocks_with_input(xs, ys) {
            let min = x.iter().copied().fold(f32::INFINITY, f32::min);
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let d = (max - min) / Q5_1_MAX;
            block.d = f16::from_f32(d);
            block.m = f16::from_f32(min);

            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            let code = |v: f32| (((v - min) * inv).round() as i32).clamp(0, 31) as u32;
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
        return super::super::avx::vec_dot_q5_1_q8_1(xs, ys);

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const HALF: usize = QK5_1 / 2;
        // As in q4_1: the minimum is constant over the block, so it multiplies the activation's
        // stored sum rather than each value.
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
                            let p = ((packed & 0x0F) as u32 | top(j)) as i32;
                            let q = ((packed >> 4) as u32 | top(j + HALF)) as i32;
                            p * a as i32 + q * b as i32
                        })
                        .sum();
                dot as f32 * x.d.to_f32() * y.d.to_f32() + x.m.to_f32() * y.s.to_f32()
            })
            .sum()
    }
}
