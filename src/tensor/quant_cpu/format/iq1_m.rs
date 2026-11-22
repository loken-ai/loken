//! IQ1_M: 256 values in 8 runs of 32, each run 4 groups of 8.
//! A group names one of the 2048 points of `IQ1S_GRID`, whose values are -1, 0 or 1.
//! A value reads `d * (2 k + 1) * (g + delta)`.
//! `d` is the block's f16 scale, its 4 nibbles stored in the top nibbles of the 4 scale words.
//! `k` is a 3-bit scale per 16 values: 4 to a 16-bit scale word, from its low bits up.
//! `delta` is plus or minus `IQ1S_DELTA`, its sign chosen per group.
//!
//! `qs` holds the low 8 bits of each group's index.
//! Each `qh` byte holds 2 groups: the high 3 bits of the index and the sign of `delta` above
//! them, the first group in the low nibble.

use super::iq1_s::{
    best_point, fit_span, grid_point, run_scale_code, IQ1S_DELTA, IQ1_GROUP, IQ1_SCALE_TOP,
};
use super::*;

/// The 16-bit words the run scales and the block scale share.
const IQ1M_WORDS: usize = QK_K / 64;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq1M {
    pub(crate) qs: [u8; QK_K / 8],
    pub(crate) qh: [u8; QK_K / 16],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIq1M>() == 56);

impl BlockIq1M {
    /// The scale words, read little-endian.
    #[inline]
    fn words(&self) -> [u16; IQ1M_WORDS] {
        std::array::from_fn(|w| u16::from_le_bytes([self.scales[2 * w], self.scales[2 * w + 1]]))
    }

    /// The block's f16 scale, gathered from the top nibble of each scale word.
    #[inline]
    fn block_scale(sc: &[u16; IQ1M_WORDS]) -> f32 {
        let bits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        f16::from_bits(bits).to_f32()
    }

    /// The odd multipliers of run `ib`'s two halves.
    #[inline]
    fn run_multipliers(sc: &[u16; IQ1M_WORDS], ib: usize) -> [i32; 2] {
        let word = sc[ib / 2] >> (6 * (ib % 2));
        [2 * (word & 7) as i32 + 1, 2 * ((word >> 3) & 7) as i32 + 1]
    }

    /// Group `l` of run `ib`: its grid index and whether its shift is negative.
    #[inline]
    fn group(&self, ib: usize, l: usize) -> (usize, bool) {
        let h = self.qh[2 * ib + l / 2] >> (4 * (l % 2));
        let index = self.qs[4 * ib + l] as usize | ((h & 7) as usize) << 8;
        (index, h & 0x08 != 0)
    }
}

impl BlockFormat for BlockIq1M {
    const DTYPE: GgmlDType = GgmlDType::Iq1M;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            qs: [0; QK_K / 8],
            qh: [0; QK_K / 16],
            scales: [0; QK_K / 32],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let sc = block.words();
            let d = Self::block_scale(&sc);
            for (ib, run) in y.chunks_exact_mut(32).enumerate() {
                let dl = Self::run_multipliers(&sc, ib).map(|m| d * m as f32);
                for (l, group) in run.chunks_exact_mut(IQ1_GROUP).enumerate() {
                    let (index, negative) = block.group(ib, l);
                    let delta = if negative { -IQ1S_DELTA } else { IQ1S_DELTA };
                    for (out, g) in group.iter_mut().zip(grid_point(index)) {
                        *out = dl[l / 2] * (g as f32 + delta);
                    }
                }
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            encode_block(x, block);
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    /// The grid sums and the signed activation sums of each half stay integers under the half's
    /// odd multiplier.
    /// The block's two scales and the shift come in once at the end of the block.
    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        let mut total = 0f32;
        for (x, y) in xs.iter().zip(ys) {
            let sc = x.words();
            let (mut sumi1, mut sumi2) = (0i32, 0i32);
            for ib in 0..QK_K / 32 {
                let (mut sum1, mut sum2) = ([0i32; 2], [0i32; 2]);
                for l in 0..4 {
                    let (index, negative) = x.group(ib, l);
                    let acts = &y.qs[32 * ib + IQ1_GROUP * l..32 * ib + IQ1_GROUP * (l + 1)];
                    let (mut lsum1, mut lsum2) = (0i32, 0i32);
                    for (g, &a) in grid_point(index).iter().zip(acts) {
                        lsum1 += a as i32 * *g as i32;
                        lsum2 += a as i32;
                    }
                    sum1[l / 2] += lsum1;
                    sum2[l / 2] += if negative { -lsum2 } else { lsum2 };
                }
                let [ls1, ls2] = Self::run_multipliers(&sc, ib);
                sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
                sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            }
            let d = Self::block_scale(&sc);
            total += d * y.d * (sumi1 as f32 + IQ1S_DELTA * sumi2 as f32);
        }
        total
    }
}

/// Encode one block: each half run's scale is fitted with its points and each group's shift
/// sign, the block scale takes the largest half scale onto the top multiplier, and each half
/// picks its points and signs again at its scale as stored.
fn encode_block(x: &[f32], block: &mut BlockIq1M) {
    const EITHER: &[f32] = &[IQ1S_DELTA, -IQ1S_DELTA];
    const HALF: usize = 16;
    let mut scales = [0f32; QK_K / HALF];
    for (h, half) in x.chunks_exact(HALF).enumerate() {
        scales[h] = fit_span(half, &[EITHER]).0;
    }
    let top = scales.iter().fold(0f32, |m, &s| m.max(s));
    let d = f16::from_f32(top / IQ1_SCALE_TOP);
    let df = d.to_f32();
    let mut sc = [0u16; IQ1M_WORDS];
    block.qh = [0; QK_K / 16];
    for (h, half) in x.chunks_exact(HALF).enumerate() {
        let k = run_scale_code(scales[h], df);
        let (ib, side) = (h / 2, h % 2);
        sc[ib / 2] |= k << (6 * (ib % 2) + 3 * side);
        let dl = df * (2 * k + 1) as f32;
        for (g, group) in half.chunks_exact(IQ1_GROUP).enumerate() {
            let l = 2 * side + g;
            let (n, delta, _) = best_point(group, dl, EITHER);
            block.qs[4 * ib + l] = (n & 0xFF) as u8;
            let nibble = ((n >> 8) as u8 & 7) | if delta < 0.0 { 0x08 } else { 0 };
            block.qh[2 * ib + l / 2] |= nibble << (4 * (l % 2));
        }
    }
    let bits = d.to_bits();
    for (w, word) in sc.iter_mut().enumerate() {
        *word |= ((bits >> (4 * w)) & 0xF) << 12;
    }
    for (w, word) in sc.iter().enumerate() {
        block.scales[2 * w..2 * w + 2].copy_from_slice(&word.to_le_bytes());
    }
}
