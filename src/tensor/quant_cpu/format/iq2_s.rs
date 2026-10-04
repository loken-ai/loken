//! iq2_s - 256 values in 82 bytes: ten-bit codes into a 1024-entry grid of eight magnitudes (the
//! top two bits in a side array), a sign bit per value, and a four-bit scale per sixteen values.
//! Read-only.

use super::iq_grids::IQ2S_GRID;
use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq2S {
    pub(crate) d: f16,
    /// Codes (the low eight bits) of the first half; the signs follow in the second half.
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIq2S>() == 82);

impl BlockFormat for BlockIq2S {
    const DTYPE: GgmlDType = GgmlDType::Iq2S;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; QK_K / 4],
            qh: [0; QK_K / 32],
            scales: [0; QK_K / 32],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (x, out) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            let d = x.d.to_f32();
            let signs = &x.qs[QK_K / 8..];
            for (ib, sub) in out.chunks_exact_mut(32).enumerate() {
                let db = [
                    d * (0.5 + (x.scales[ib] & 0x0F) as f32) * 0.25,
                    d * (0.5 + (x.scales[ib] >> 4) as f32) * 0.25,
                ];
                for l in 0..4 {
                    let code = x.qs[4 * ib + l] as usize
                        | ((((x.qh[ib] as u32) << (8 - 2 * l)) & 0x300) as usize);
                    let grid = IQ2S_GRID[code].to_le_bytes();
                    let sign = signs[4 * ib + l];
                    for j in 0..8 {
                        let s = if sign & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        sub[8 * l + j] = db[l / 2] * grid[j] as f32 * s;
                    }
                }
            }
        }
    }

    fn quantize(_: &[f32], _: &mut [Self]) {
        panic!("iq2_s is read, not written");
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        super::iq_read::dot_by_decoding(xs, ys)
    }
}
