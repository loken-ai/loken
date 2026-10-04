//! iq3_xxs - 256 values in 98 bytes: codes into a 256-entry grid of four magnitudes, a 7-bit sign
//! pattern per eight values, and a four-bit scale per thirty-two. Read-only.

use super::iq_grids::IQ3XXS_GRID;
use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq3Xxs {
    pub(crate) d: f16,
    pub(crate) qs: [u8; 3 * QK_K / 8],
}
const _: () = assert!(std::mem::size_of::<BlockIq3Xxs>() == 98);

impl BlockFormat for BlockIq3Xxs {
    const DTYPE: GgmlDType = GgmlDType::Iq3Xxs;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; 3 * QK_K / 8],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (x, out) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            let d = x.d.to_f32();
            let (codes, tail) = x.qs.split_at(QK_K / 4);
            for (ib, sub) in out.chunks_exact_mut(32).enumerate() {
                let word = u32::from_le_bytes(tail[4 * ib..4 * ib + 4].try_into().unwrap());
                let db = d * (0.5 + (word >> 28) as f32) * 0.5;
                let q = &codes[8 * ib..8 * ib + 8];
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((word >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[q[2 * l] as usize].to_le_bytes();
                    let g2 = IQ3XXS_GRID[q[2 * l + 1] as usize].to_le_bytes();
                    for j in 0..4 {
                        let s1 = if signs & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        sub[8 * l + j] = db * g1[j] as f32 * s1;
                        sub[8 * l + j + 4] = db * g2[j] as f32 * s2;
                    }
                }
            }
        }
    }

    fn quantize(_: &[f32], _: &mut [Self]) {
        panic!("iq3_xxs is read, not written");
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        super::iq_read::dot_by_decoding(xs, ys)
    }
}
