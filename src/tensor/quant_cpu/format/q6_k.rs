//! q6_k: block layout, dequantiser and quantiser together.
//!
//! Its vectorised kernel is still in `super::avx`, which shares primitives across
//! formats; it follows once those are factored out.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ6K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

/// Six bits reach 0..63 and stand for -32..31; the file stores the code biased.
const Q6_K_BIAS: i32 = 32;

impl BlockFormat for BlockQ6K {
    const DTYPE: GgmlDType = GgmlDType::Q6K;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            ql: [0; QK_K / 2],
            qh: [0; QK_K / 4],
            scales: [0; QK_K / 16],
            d: f16::ZERO,
        }
    }

    #[allow(unreachable_code)]
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_q6k_q8k(xs, ys);

        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        // One scale per sixteen values, so the products are gathered per sub-block as exact
        // integers and scaled once. Accumulating in f32 as they come would round 256 times per
        // block for no gain: six-bit codes against an eight-bit activation, summed sixteen
        // times, cannot leave i32.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let mut per_sub = [0i32; QK_K / 16];
                for half in 0..2 {
                    let ql = &x.ql[64 * half..];
                    let qh = &x.qh[32 * half..];
                    let acts = &y.qs[128 * half..];
                    for g in 0..4 {
                        let (span, nibble) = (32 * (g % 2), if g < 2 { 0 } else { 4 });
                        for l in 0..32 {
                            let low = (ql[span + l] >> nibble) & 0x0F;
                            let high = (qh[l] >> (2 * g)) & 0x03;
                            let q = (low | (high << 4)) as i32 - Q6_K_BIAS;
                            per_sub[8 * half + 2 * g + l / 16] += q * acts[32 * g + l] as i32;
                        }
                    }
                }
                let dot: f32 = per_sub
                    .iter()
                    .zip(&x.scales)
                    .map(|(&sum, &scale)| sum as f32 * scale as f32)
                    .sum();
                dot * x.d.to_f32() * y.d
            })
            .sum()
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        encode(xs, ys, None, 0);
    }

    fn quantize_guided(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        encode(xs, ys, Some(imatrix_weights), n_per_row);
    }
    /// `y = d * scale[s] * (q - 32)`, six bits per value over sixteen sub-blocks of sixteen.
    ///
    /// The six bits are split: the low four live in `ql`, two values to a byte, and the top two
    /// in `qh`, four values to a byte. Within a half of 128 values the four groups of 32 read
    /// the same 32 bytes of `qh` at a shift that advances with the group, and alternate between
    /// the two 32-byte spans of `ql` and between its low and high nibbles. The stored value is
    /// BIASED: the file holds the signed number plus 32.
    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            for half in 0..2 {
                let ql = &block.ql[64 * half..];
                let qh = &block.qh[32 * half..];
                let scales = &block.scales[8 * half..];
                let out = &mut y[128 * half..];
                for g in 0..4 {
                    let span = 32 * (g % 2);
                    let nibble = if g < 2 { 0 } else { 4 };
                    for l in 0..32 {
                        let low = (ql[span + l] >> nibble) & 0x0F;
                        let high = (qh[l] >> (2 * g)) & 0x03;
                        let q = (low | (high << 4)) as i32 - Q6_K_BIAS;
                        out[32 * g + l] = d * scales[2 * g + l / 16] as f32 * q as f32;
                    }
                }
            }
        }
    }
}

/// Quantise whole blocks, optionally against an importance matrix.
///
/// The plain and imatrix encoders differ in one thing only - whether each group of sixteen is
/// fitted against measured importances or against the values' own magnitudes - so they are one
/// function. `imatrix` holds `n_per_row` weights per row, which a block indexes by its position
/// within its row; without it `fit_signed_scale` falls back to weighting by `x²`.
fn encode(xs: &[f32], ys: &mut [BlockQ6K], imatrix: Option<&[f32]>, n_per_row: usize) {
    for (idx, (block, x)) in blocks_with_input(xs, ys).enumerate() {
        let row = imatrix.map(|m| {
            let base = (idx % (n_per_row / QK_K)) * QK_K;
            &m[base..base + QK_K]
        });
        let mut levels = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut fallback = [0f32; 16];

        // Sixteen groups of sixteen, each fitted on its own; one block scale then has to span
        // all sixteen results, which is what the second quantisation below does.
        for (g, ((scale, xg), lg)) in scales
            .iter_mut()
            .zip(x.chunks_exact(16))
            .zip(levels.chunks_exact_mut(16))
            .enumerate()
        {
            if row.is_none() {
                magnitude_weights(xg, &mut fallback);
            }
            let w = row.map_or(&fallback[..], |r| &r[16 * g..16 * (g + 1)]);
            *scale = fit_signed_scale(32, xg, lg, w);
        }

        // The group scales are themselves quantised, to eight bits against the largest of them.
        let max = scales
            .iter()
            .fold(0f32, |m, &s| if s.abs() > m.abs() { s } else { m });
        if max == 0.0 {
            block.d = f16::from_f32(0.0);
            block.scales.fill(0);
            block.ql.fill(0);
            block.qh.fill(0);
            continue;
        }
        let iscale = -128.0 / max;
        block.d = f16::from_f32(1.0 / iscale);
        for (out, &scale) in block.scales.iter_mut().zip(scales.iter()) {
            *out = nearest_int(iscale * scale).min(127) as i8;
        }

        // Re-derive every level from the scale as it was ROUNDED into the block, not as it was
        // fitted: the encoder has to quantise against the number the decoder will actually read,
        // and those two differ by the rounding just applied.
        for ((&scale, xg), lg) in block
            .scales
            .iter()
            .zip(x.chunks_exact(16))
            .zip(levels.chunks_exact_mut(16))
        {
            let d = block.d.to_f32() * scale as f32;
            if d == 0.0 {
                continue;
            }
            for (out, &xi) in lg.iter_mut().zip(xg) {
                *out = (nearest_int(xi / d).clamp(-Q6_K_BIAS, Q6_K_BIAS - 1) + Q6_K_BIAS) as i8;
            }
        }

        // Pack six bits per value: low four into `ql`, top two into `qh`, over halves of 128.
        // Within a half the four groups of 32 share `qh` at advancing shifts and alternate
        // between the two 32-byte spans of `ql` and between its nibbles - the layout `dequantize`
        // documents, written here in the same order it is read there.
        let (lo, hi) = (|v: i8| (v as u8) & 0x0F, |v: i8| (v as u8) >> 4);
        for ((ql, qh), half) in block
            .ql
            .chunks_exact_mut(64)
            .zip(block.qh.chunks_exact_mut(32))
            .zip(levels.chunks_exact(128))
        {
            for i in 0..32 {
                let (a, b, c, e) = (half[i], half[i + 32], half[i + 64], half[i + 96]);
                ql[i] = lo(a) | (lo(c) << 4);
                ql[i + 32] = lo(b) | (lo(e) << 4);
                qh[i] = hi(a) | (hi(b) << 2) | (hi(c) << 4) | (hi(e) << 6);
            }
        }
    }
}
