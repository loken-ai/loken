//! NVFP4: 64 values in 4 runs of 16.
//! Each run holds 16 E2M1 codes under one unsigned E4M3 scale.
//! A byte holds value `j` of its run in its low nibble and value `j + 8` in its high nibble.

use super::*;

pub const QK_NVFP4: usize = 64;
/// Values under one scale.
pub const QK_NVFP4_SUB: usize = 16;
/// Runs a block holds.
const NVFP4_RUNS: usize = QK_NVFP4 / QK_NVFP4_SUB;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockNvFp4 {
    /// One unsigned E4M3 scale per run.
    pub(crate) d: [u8; NVFP4_RUNS],
    pub(crate) qs: [u8; QK_NVFP4 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockNvFp4>() == 36);

/// An unsigned E4M3 scale as an f32, halved to match the doubled codes of `KVALUES_MXFP4`.
/// The codes 0 and 0x7F read as zero.
/// The top bit is not read.
#[inline]
pub(crate) fn ue4m3_to_fp32_half(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 {
        man * pow2(-9)
    } else {
        (1.0 + man / 8.0) * pow2(exp - 7)
    };
    raw * 0.5
}

/// `2^e` for an exponent inside the normal range of an f32.
#[inline]
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// The unsigned E4M3 code of a positive value, rounding the mantissa half up.
/// A value that is not positive gives 0.
/// A value past the largest code saturates at 0x7E.
pub(crate) fn fp32_to_ue4m3(x: f32) -> u8 {
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    let x = x.min(448.0);
    let bits = x.to_bits();
    let mut exp = ((bits >> 23) & 0xFF) as i32 - 127 + 7;
    if exp <= 0 {
        // Subnormal: the code is the value in steps of 2^-9.
        let man = ((x * 512.0 + 0.5) as i32).min(7);
        return if man < 1 { 0 } else { man as u8 };
    }
    if exp >= 15 {
        return 0x7E;
    }
    let mut man = ((bits >> 20) & 0x7) as i32 + ((bits >> 19) & 1) as i32;
    if man > 7 {
        man = 0;
        exp += 1;
        if exp >= 15 {
            return 0x7E;
        }
    }
    ((exp << 3) | man) as u8
}

/// The E2M1 code whose value at scale `d` is nearest `x`, the first one on a tie.
#[inline]
fn nearest_code(x: f32, d: f32) -> u8 {
    let mut best = 0usize;
    let mut best_err = (KVALUES_MXFP4[0] as f32 * d - x).abs();
    for (i, &kv) in KVALUES_MXFP4.iter().enumerate().skip(1) {
        let err = (kv as f32 * d - x).abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best as u8
}

impl BlockFormat for BlockNvFp4 {
    const DTYPE: GgmlDType = GgmlDType::NvFp4;
    const BLOCK_LEN: usize = QK_NVFP4;
    type ActivationBlock = BlockQ8_64;

    fn zeros() -> Self {
        Self {
            d: [0; NVFP4_RUNS],
            qs: [0; QK_NVFP4 / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        const HALF: usize = QK_NVFP4_SUB / 2;
        for (block, y) in blocks_with_output(xs, ys) {
            for (s, run) in y.chunks_exact_mut(QK_NVFP4_SUB).enumerate() {
                let d = ue4m3_to_fp32_half(block.d[s]);
                for j in 0..HALF {
                    let q = block.qs[s * HALF + j];
                    run[j] = KVALUES_MXFP4[(q & 0x0F) as usize] as f32 * d;
                    run[j + HALF] = KVALUES_MXFP4[(q >> 4) as usize] as f32 * d;
                }
            }
        }
    }

    /// Each run's scale maps its largest magnitude onto the largest code, 6.
    /// Each value then takes its nearest code at the rounded scale.
    fn quantize(xs: &[f32], ys: &mut [Self]) {
        const HALF: usize = QK_NVFP4_SUB / 2;
        for (block, x) in blocks_with_input(xs, ys) {
            for (s, run) in x.chunks_exact(QK_NVFP4_SUB).enumerate() {
                let amax = run
                    .iter()
                    .fold(0f32, |m, &v| if m < v.abs() { v.abs() } else { m });
                let code = fp32_to_ue4m3(amax / 6.0);
                block.d[s] = code;
                let d = ue4m3_to_fp32_half(code);
                for j in 0..HALF {
                    block.qs[s * HALF + j] =
                        nearest_code(run[j], d) | (nearest_code(run[j + HALF], d) << 4);
                }
            }
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    /// Run `s` pairs with half `s / 2` of the activation block, at offset `16 * (s % 2)`.
    /// Each run's integer sum is taken under its scale and the half's scale.
    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const HALF: usize = QK_NVFP4_SUB / 2;
        let mut total = 0f32;
        for (x, y) in xs.iter().zip(ys) {
            for s in 0..NVFP4_RUNS {
                let d = ue4m3_to_fp32_half(x.d[s]);
                let half = s / 2;
                let at = half * QK8_64_HALF + (s % 2) * QK_NVFP4_SUB;
                let dy = y.d[half].to_f32();
                let (mut lo, mut hi) = (0i32, 0i32);
                for j in 0..HALF {
                    let q = x.qs[s * HALF + j];
                    lo += y.qs[at + j] as i32 * KVALUES_MXFP4[(q & 0x0F) as usize] as i32;
                    hi += y.qs[at + j + HALF] as i32 * KVALUES_MXFP4[(q >> 4) as usize] as i32;
                }
                total += dy * d * (lo + hi) as f32;
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every code below the top exponent decodes to a positive value that encodes to it again.
    /// The encoder saturates the top exponent at 0x7E.
    #[test]
    fn every_scale_code_round_trips() {
        for code in 1u8..0x78 {
            let v = ue4m3_to_fp32_half(code) * 2.0;
            assert!(v > 0.0, "code {code:#x}");
            assert_eq!(fp32_to_ue4m3(v), code, "code {code:#x} reads {v}");
        }
        assert_eq!(ue4m3_to_fp32_half(0), 0.0);
        assert_eq!(ue4m3_to_fp32_half(0x7F), 0.0);
        assert_eq!(fp32_to_ue4m3(1e9), 0x7E);
        assert_eq!(fp32_to_ue4m3(-1.0), 0);
    }
}
