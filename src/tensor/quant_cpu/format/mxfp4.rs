//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockMxFp4 {
    pub(crate) e: u8,                  // E8M0 per-block scale
    pub(crate) qs: [u8; QK_MXFP4 / 2], // 32 x 4-bit E2M1 codes, packed
}
const _: () = assert!(std::mem::size_of::<BlockMxFp4>() == 17);

impl BlockFormat for BlockMxFp4 {
    const DTYPE: GgmlDType = GgmlDType::MxFp4;
    const BLOCK_LEN: usize = QK_MXFP4;
    type ActivationBlock = BlockQ8_0;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            e: 0,
            qs: [0; QK_MXFP4 / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_MXFP4));
        let nb = k / QK_MXFP4;
        for i in 0..nb {
            let d = e8m0_to_fp32_half(xs[i].e);
            // Interleaved like Q4_0: low nibbles -> first half, high -> second.
            for j in 0..QK_MXFP4 / 2 {
                let b = xs[i].qs[j];
                ys[i * QK_MXFP4 + j] = KVALUES_MXFP4[(b & 0x0F) as usize] as f32 * d;
                ys[i * QK_MXFP4 + j + QK_MXFP4 / 2] = KVALUES_MXFP4[(b >> 4) as usize] as f32 * d;
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        debug_assert!(xs.len().is_multiple_of(QK_MXFP4));
        for (block, xb) in ys.iter_mut().zip(xs.chunks_exact(QK_MXFP4)) {
            let amax = xb.iter().fold(0f32, |m, &v| m.max(v.abs()));
            if amax == 0.0 {
                block.e = 0;
                block.qs = [0u8; QK_MXFP4 / 2];
                continue;
            }
            // A block's only free parameter is its E8M0 exponent, and there is no formula
            // that picks the best one. The OCP rule - `floor(log2 amax) - 2`, which is what
            // ggml applies - always CLIPS: it caps the block at 1.5.2^floor(log2 amax) while
            // amax runs up to 2^(floor+1), so a third of the range can be lost on one value.
            // The next exponent up never clips, at the price of a step twice as coarse. The
            // one below is coarser still on the largest values but resolves the small ones,
            // which wins on a block that is near-zero except for one outlier.
            //
            // Which of the three is best is a property of the block, not of the format, so
            // measure instead of choosing: encode with each and keep the lowest squared
            // error. Encoding is offline, and the cost buys back what the OCP rule clips.
            let base = amax.log2().floor() as i32 - 2 + 127;
            let mut best = (f32::INFINITY, 0u8, [0u8; QK_MXFP4 / 2]);
            for candidate in (base - 1)..=(base + 1) {
                let e = match u8::try_from(candidate) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let d = e8m0_to_fp32_half(e);
                if d == 0.0 {
                    continue;
                }
                let (mut qs, mut err) = ([0u8; QK_MXFP4 / 2], 0f32);
                for (j, packed) in qs.iter_mut().enumerate() {
                    let (lo, hi) = (xb[j], xb[j + QK_MXFP4 / 2]);
                    let (clo, chi) = (quantize_mxfp4_nibble(lo / d), quantize_mxfp4_nibble(hi / d));
                    *packed = clo | (chi << 4);
                    err += (KVALUES_MXFP4[clo as usize] as f32 * d - lo).powi(2)
                        + (KVALUES_MXFP4[chi as usize] as f32 * d - hi).powi(2);
                }
                if err < best.0 {
                    best = (err, e, qs);
                }
            }
            block.e = best.1;
            block.qs = best.2;
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_mxfp4_q8_0(xs, ys);

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const HALF: usize = QK_MXFP4 / 2;
        // The weight is interleaved - byte `j` carries value `j` low and `j + 16` high - while
        // the q8_0 activation is sequential, so the two halves of its block pair with the two
        // nibbles rather than with consecutive bytes.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let scale = e8m0_to_fp32_half(x.e) * y.d.to_f32();
                let (low, high) = y.qs.split_at(HALF);
                let dot: i32 =
                    x.qs.iter()
                        .zip(low)
                        .zip(high)
                        .map(|((&packed, &a), &b)| {
                            KVALUES_MXFP4[(packed & 0x0F) as usize] as i32 * a as i32
                                + KVALUES_MXFP4[(packed >> 4) as usize] as i32 * b as i32
                        })
                        .sum();
                dot as f32 * scale
            })
            .sum()
    }
}

/// Nearest MXFP4 4-bit code for a value already divided by the block scale.
#[inline]
pub(super) fn quantize_mxfp4_nibble(x: f32) -> u8 {
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (i, &kv) in KVALUES_MXFP4.iter().enumerate() {
        let err = (kv as f32 - x).abs();
        if err < best_err {
            best_err = err;
            best = i;
        }
    }
    best as u8
}
