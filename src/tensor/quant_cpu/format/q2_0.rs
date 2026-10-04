//! q2_0 - sixty-four values, a scale and two-bit codes: `y = (q - 1) . d` for codes 0 to 3, so the
//! levels are -1, 0, +1 and +2 times the scale. Four codes share a byte, lowest first.

use super::*;

pub const QK2_0: usize = 64;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK2_0 / 4],
}
const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == 2 + QK2_0 / 4);

impl BlockFormat for BlockQ2_0 {
    const DTYPE: GgmlDType = GgmlDType::Q2_0;
    const BLOCK_LEN: usize = QK2_0;
    type ActivationBlock = BlockQ8_64;

    /// Every code at the level of zero.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0x55; QK2_0 / 4],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            for (j, out) in y.iter_mut().enumerate() {
                let code = (block.qs[j / 4] >> (2 * (j % 4))) & 3;
                *out = (code as i32 - 1) as f32 * d;
            }
        }
    }

    /// The scale puts the extreme on the end of the level range it falls in; each value takes
    /// the nearest of the four levels.
    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            let amax = x
                .iter()
                .copied()
                .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
            let d = if amax == 0.0 {
                0.0
            } else if amax > 0.0 {
                amax / 2.0
            } else {
                -amax
            };
            block.d = f16::from_f32(d);
            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            block.qs = [0; QK2_0 / 4];
            for (j, v) in x.iter().enumerate() {
                let level = (v * inv).round().clamp(-1.0, 2.0) as i32;
                block.qs[j / 4] |= ((level + 1) as u8) << (2 * (j % 4));
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
                let dot: i32 = (0..QK2_0)
                    .map(|j| {
                        let code = ((x.qs[j / 4] >> (2 * (j % 4))) & 3) as i32;
                        (code - 1) * y.qs[j] as i32
                    })
                    .sum();
                dot as f32 * x.d.to_f32() * y.d
            })
            .sum()
    }
}
