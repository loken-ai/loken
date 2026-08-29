//! q4_1 - thirty-two values, a scale and a minimum, four bits each.
//!
//! `y = q . d + m`. Where q4_0 centres the codes on zero and pays for it with a bias, this
//! one carries the block's minimum explicitly, so the sixteen codes span `[min, max]` however
//! that interval sits. It costs two more bytes per block and reconstructs a one-sided block  - 
//! the output of a ReLU, say - far better, because none of its range is spent below zero.
//!
//! Same nibble packing as q4_0: byte `j` holds value `j` low and value `j + 16` high.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 4 + QK4_1 / 2);

/// The largest code four bits reach; the interval is divided into this many steps.
const Q4_1_MAX: f32 = 15.0;

impl BlockFormat for BlockQ4_1 {
    const DTYPE: GgmlDType = GgmlDType::Q4_1;
    const BLOCK_LEN: usize = QK4_1;
    type ActivationBlock = BlockQ8_1;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            m: f16::ZERO,
            qs: [0; QK4_1 / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        const HALF: usize = QK4_1 / 2;
        for (block, y) in blocks_with_output(xs, ys) {
            let (d, m) = (block.d.to_f32(), block.m.to_f32());
            let (low, high) = y.split_at_mut(HALF);
            for ((lo, hi), &packed) in low.iter_mut().zip(high).zip(&block.qs) {
                *lo = (packed & 0x0F) as f32 * d + m;
                *hi = (packed >> 4) as f32 * d + m;
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        const HALF: usize = QK4_1 / 2;
        for (block, x) in blocks_with_input(xs, ys) {
            let min = x.iter().copied().fold(f32::INFINITY, f32::min);
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let d = (max - min) / Q4_1_MAX;
            block.d = f16::from_f32(d);
            block.m = f16::from_f32(min);

            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            let code = |v: f32| (((v - min) * inv).round() as i32).clamp(0, 15) as u8;
            let (low, high) = x.split_at(HALF);
            for ((&lo, &hi), packed) in low.iter().zip(high).zip(block.qs.iter_mut()) {
                *packed = code(lo) | (code(hi) << 4);
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::super::avx::vec_dot_q4_1_q8_1(xs, ys);

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const HALF: usize = QK4_1 / 2;
        // The minimum is constant across the block, so `Σ (q.d + m).a = d.Σqa + m.Σa`, and
        // `Σa` scaled is exactly what the activation carries in `s`. That is why the q8_1
        // format has a second field at all.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let (low, high) = y.qs.split_at(HALF);
                let dot: i32 =
                    x.qs.iter()
                        .zip(low)
                        .zip(high)
                        .map(|((&packed, &a), &b)| {
                            (packed & 0x0F) as i32 * a as i32 + (packed >> 4) as i32 * b as i32
                        })
                        .sum();
                dot as f32 * x.d.to_f32() * y.d.to_f32() + x.m.to_f32() * y.s.to_f32()
            })
            .sum()
    }
}
