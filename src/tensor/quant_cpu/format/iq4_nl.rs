//! iq4_nl - thirty-two values, a scale, four-bit codes into a non-linear table.
//!
//! `y = d . table[q]`: the sixteen levels are packed towards zero, where weights concentrate.
//! The nibble array splits as q4_0 does: byte `j` holds value `j` low and value `j + 16` high.

use super::*;

pub const QK4_NL: usize = 32;

/// The sixteen levels a code names.
pub(super) const LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq4Nl {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_NL / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIq4Nl>() == 2 + QK4_NL / 2);

/// The code whose level is nearest `v` after division by the scale.
fn nearest(v: f32) -> u8 {
    let mut best = 0u8;
    let mut err = f32::INFINITY;
    for (code, &level) in LEVELS.iter().enumerate() {
        let e = (v - level as f32).abs();
        if e < err {
            err = e;
            best = code as u8;
        }
    }
    best
}

impl BlockFormat for BlockIq4Nl {
    const DTYPE: GgmlDType = GgmlDType::Iq4Nl;
    const BLOCK_LEN: usize = QK4_NL;
    type ActivationBlock = BlockQ8_0;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0x88; QK4_NL / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        const HALF: usize = QK4_NL / 2;
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            let (low, high) = y.split_at_mut(HALF);
            for ((lo, hi), &packed) in low.iter_mut().zip(high).zip(&block.qs) {
                *lo = d * LEVELS[(packed & 0x0F) as usize] as f32;
                *hi = d * LEVELS[(packed >> 4) as usize] as f32;
            }
        }
    }

    /// The scale puts the block's largest magnitude on the widest level of its sign, then
    /// every value takes the nearest level.
    fn quantize(xs: &[f32], ys: &mut [Self]) {
        const HALF: usize = QK4_NL / 2;
        for (block, x) in blocks_with_input(xs, ys) {
            let amax = x
                .iter()
                .copied()
                .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
            let d = if amax == 0.0 {
                0.0
            } else {
                amax / if amax < 0.0 { -127.0 } else { 113.0 }
            };
            block.d = f16::from_f32(d);
            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            let (low, high) = x.split_at(HALF);
            for ((&lo, &hi), packed) in low.iter().zip(high).zip(block.qs.iter_mut()) {
                *packed = nearest(lo * inv) | (nearest(hi * inv) << 4);
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const HALF: usize = QK4_NL / 2;
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let (low, high) = y.qs.split_at(HALF);
                let dot: i32 =
                    x.qs.iter()
                        .zip(low)
                        .zip(high)
                        .map(|((&packed, &a), &b)| {
                            LEVELS[(packed & 0x0F) as usize] as i32 * a as i32
                                + LEVELS[(packed >> 4) as usize] as i32 * b as i32
                        })
                        .sum();
                dot as f32 * x.d.to_f32() * y.d.to_f32()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_round_trips_within_the_level_spacing() {
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let mut b = [BlockIq4Nl::zeros()];
        BlockIq4Nl::quantize(&x, &mut b);
        let mut y = vec![0f32; 32];
        BlockIq4Nl::dequantize(&b, &mut y);
        let d = b[0].d.to_f32();
        for (a, r) in x.iter().zip(&y) {
            assert!((a - r).abs() <= d * 12.0, "{a} vs {r}");
        }
    }

    #[test]
    fn a_known_block_decodes_by_its_table() {
        // Byte j holds value j low and value j + 16 high.
        let mut qs = [0u8; 16];
        qs[0] = 0x80; // value 0 -> code 0 (-127), value 16 -> code 8 (1)
        qs[15] = 0xF7; // value 15 -> code 7 (-10), value 31 -> code 15 (113)
        let b = [BlockIq4Nl {
            d: f16::from_f32(2.0),
            qs,
        }];
        let mut y = vec![0f32; 32];
        BlockIq4Nl::dequantize(&b, &mut y);
        assert_eq!((y[0], y[16], y[15], y[31]), (-254.0, 2.0, -20.0, 226.0));
    }

    #[test]
    fn the_dot_equals_the_dequantised_product() {
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.21).cos()).collect();
        let a: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.13).sin()).collect();
        let mut w = [BlockIq4Nl::zeros()];
        BlockIq4Nl::quantize(&x, &mut w);
        let mut act = [BlockQ8_0::zeros()];
        BlockQ8_0::quantize(&a, &mut act);
        let mut wd = vec![0f32; 32];
        let mut ad = vec![0f32; 32];
        BlockIq4Nl::dequantize(&w, &mut wd);
        BlockQ8_0::dequantize(&act, &mut ad);
        let want: f32 = wd.iter().zip(&ad).map(|(p, q)| p * q).sum();
        let got = BlockIq4Nl::dot(&w, &act);
        assert!(
            (want - got).abs() < 1e-3 * want.abs().max(1.0),
            "{want} vs {got}"
        );
    }
}
