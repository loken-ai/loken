//! q4_0 - thirty-two values, one scale, four bits each.
//!
//! `y = (q - 8) . d`. The bias of 8 is what makes an unsigned nibble signed: codes run 0..15
//! and represent -8..7, so the scale is set from the block's extreme NEGATIVE value rather
//! than its largest magnitude - -8 is representable and +8 is not.
//!
//! The two nibbles of a byte are NOT adjacent values. Byte `j` carries value `j` in its low
//! nibble and value `j + 16` in its high one, so a dot product can unpack sixteen pairs at a
//! time. `oracle_parity.rs` pins the dequantiser bit-for-bit against ggml's scalar output.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 2 + QK4_0 / 2);

/// What a nibble stands for: code `c` means `c - Q4_0_BIAS`.
const Q4_0_BIAS: i32 = 8;

impl BlockFormat for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLOCK_LEN: usize = QK4_0;
    type ActivationBlock = BlockQ8_0;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; QK4_0 / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        const HALF: usize = QK4_0 / 2;
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            let (low, high) = y.split_at_mut(HALF);
            for ((lo, hi), &packed) in low.iter_mut().zip(high).zip(&block.qs) {
                *lo = ((packed & 0x0F) as i32 - Q4_0_BIAS) as f32 * d;
                *hi = ((packed >> 4) as i32 - Q4_0_BIAS) as f32 * d;
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        const HALF: usize = QK4_0 / 2;
        for (block, x) in blocks_with_input(xs, ys) {
            // The extreme value, keeping its SIGN: it has to land on -8, the one end of the
            // range that exists. Taking the magnitude instead would let a block whose extreme
            // is positive map it to +8, which no code represents.
            let extreme = x
                .iter()
                .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            let d = extreme / -(Q4_0_BIAS as f32);
            block.d = f16::from_f32(d);

            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            let code = |v: f32| ((v * inv + Q4_0_BIAS as f32).round() as i32).clamp(0, 15) as u8;
            let (low, high) = x.split_at(HALF);
            for ((&lo, &hi), packed) in low.iter().zip(high).zip(block.qs.iter_mut()) {
                *packed = code(lo) | (code(hi) << 4);
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_q4_0_q8_0(xs, ys);

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const HALF: usize = QK4_0 / 2;
        // The activation shares the block size but not the packing: its codes are sequential,
        // so the nibble at byte `j` pairs with `qs[j]` and its neighbour with `qs[j + 16]`.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let (low, high) = y.qs.split_at(HALF);
                let dot: i32 =
                    x.qs.iter()
                        .zip(low)
                        .zip(high)
                        .map(|((&packed, &a), &b)| {
                            let lo = (packed & 0x0F) as i32 - Q4_0_BIAS;
                            let hi = (packed >> 4) as i32 - Q4_0_BIAS;
                            lo * a as i32 + hi * b as i32
                        })
                        .sum();
                dot as f32 * x.d.to_f32() * y.d.to_f32()
            })
            .sum()
    }
}
