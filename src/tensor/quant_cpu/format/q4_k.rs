//! q4_K: 256 values, eight sub-blocks of thirty-two, four bits each.
//!
//! Layout, dequantiser and quantiser together, because they are one statement about the
//! format: a sub-block count changed in one of them and not the others is a silent defect.
//!
//! Still elsewhere: the AVX2 `vec_dot_q4k_q8k`, in `super::avx`. It leans on primitives
//! shared with the other formats' kernels, so it follows once those are factored out - moving
//! it alone would duplicate them.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

impl BlockFormat for BlockQ4K {
    const DTYPE: GgmlDType = GgmlDType::Q4K;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            dmin: f16::ZERO,
            scales: [0; K_SCALE_SIZE],
            qs: [0; QK_K / 2],
        }
    }

    #[allow(unreachable_code)]
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_q4k_q8k(xs, ys);

        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const SUB: usize = 32;
        // `y = d.sc[s].q - dmin.m[s]`, so over a sub-block the minimum multiplies the SUM of
        // the activations rather than each of them - and that sum is already in `bsums`, two
        // entries of sixteen per thirty-two-value sub-block. Nothing here re-adds it.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let (scales, mins) = kquant_scales_and_mins(&x.scales);
                let (mut scaled, mut offset) = (0f32, 0f32);
                for s in 0..QK_K / SUB {
                    let group = &x.qs[(s / 2) * SUB..(s / 2) * SUB + SUB];
                    let shift = if s % 2 == 0 { 0 } else { 4 };
                    let acts = &y.qs[s * SUB..(s + 1) * SUB];
                    let dot: i32 = group
                        .iter()
                        .zip(acts)
                        .map(|(&packed, &a)| ((packed >> shift) & 0x0F) as i32 * a as i32)
                        .sum();
                    scaled += scales[s] as f32 * dot as f32;
                    offset +=
                        mins[s] as f32 * (y.bsums[2 * s] as i32 + y.bsums[2 * s + 1] as i32) as f32;
                }
                y.d * (x.d.to_f32() * scaled - x.dmin.to_f32() * offset)
            })
            .sum()
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        encode(xs, ys, None, 0);
    }

    fn quantize_guided(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        encode(xs, ys, Some(imatrix_weights), n_per_row);
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        const SUB: usize = 32;
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            let (scales, mins) = kquant_scales_and_mins(&block.scales);
            for s in 0..QK_K / SUB {
                let group = &block.qs[(s / 2) * SUB..(s / 2) * SUB + SUB];
                let shift = if s % 2 == 0 { 0 } else { 4 };
                let scale = d * scales[s] as f32;
                let offset = dmin * mins[s] as f32;
                for (out, packed) in y[s * SUB..(s + 1) * SUB].iter_mut().zip(group) {
                    *out = scale * ((packed >> shift) & 0x0F) as f32 - offset;
                }
            }
        }
    }
}

/// Quantise whole blocks, optionally against an importance matrix.
///
/// Two stages. Each thirty-two-value sub-block is fitted on its own to a `(scale, minimum)`
/// pair; those sixteen numbers are then themselves quantised to six bits each, which is what
/// `d` and `dmin` scale. The levels are re-derived afterwards from the scales as they were
/// ROUNDED into the block, not as they were fitted - the encoder has to quantise against the
/// number the decoder will read.
///
/// The importance matrix changes BOTH stages, not just the first: it weighs each value by
/// `imatrix . sqrt(σ² + x²)`, and the sixteen scales are then fitted by weighted least squares
/// against how much weight each sub-block carries, instead of being divided by the largest of
/// them. Losing the second of those is invisible to a test that only exercises `quantize`.
fn encode(xs: &[f32], ys: &mut [BlockQ4K], imatrix: Option<&[f32]>, n_per_row: usize) {
    const SUB: usize = 32;
    const GROUPS: usize = QK_K / SUB;
    /// Six bits for a scale and six for a minimum.
    const LEVELS: u8 = 63;

    for (idx, (block, x)) in blocks_with_input(xs, ys).enumerate() {
        let row = imatrix.map(|m| {
            let base = (idx % (n_per_row / QK_K)) * QK_K;
            &m[base..base + QK_K]
        });
        // The variance floor keeps a near-zero value from being weighed at zero and dropping
        // out of the fit entirely.
        let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;

        let mut scales = [0f32; GROUPS];
        let mut mins = [0f32; GROUPS];
        let mut carried = [0f32; GROUPS];
        let mut w = [0f32; SUB];

        for (g, (xg, (scale, min))) in x
            .chunks_exact(SUB)
            .zip(scales.iter_mut().zip(mins.iter_mut()))
            .enumerate()
        {
            match row {
                Some(r) => {
                    for (out, (&iw, &v)) in w.iter_mut().zip(r[SUB * g..].iter().zip(xg)) {
                        *out = iw * (sigma2 + v * v).sqrt();
                    }
                }
                None => magnitude_weights(xg, &mut w),
            }
            carried[g] = w.iter().sum();
            (*scale, *min) = fit_scale_and_offset(15, xg, &w);
        }

        let (mut ls, mut lm) = ([0u8; GROUPS], [0u8; GROUPS]);
        let (d, dmin) = if row.is_some() {
            (
                fit_unsigned_scale(LEVELS, &scales, &mut ls, &carried),
                fit_unsigned_scale(LEVELS, &mins, &mut lm, &carried),
            )
        } else {
            // Both families are non-negative by construction, so the largest of each is what
            // the six bits have to reach.
            let max_scale = scales.iter().fold(0.0f32, |m, &v| v.max(m));
            let max_min = mins.iter().fold(0.0f32, |m, &v| v.max(m));
            let to_level = |v: f32, max: f32| {
                let inv = if max > 0.0 { LEVELS as f32 / max } else { 0.0 };
                nearest_int(inv * v).min(LEVELS as i32) as u8
            };
            for g in 0..GROUPS {
                ls[g] = to_level(scales[g], max_scale);
                lm[g] = to_level(mins[g], max_min);
            }
            (max_scale / LEVELS as f32, max_min / LEVELS as f32)
        };
        block.scales = pack_kquant_scales_and_mins(&ls, &lm);
        block.d = f16::from_f32(d);
        block.dmin = f16::from_f32(dmin);

        let (scales, mins) = kquant_scales_and_mins(&block.scales);
        let mut levels = [0u8; QK_K];
        for (g, (xg, lg)) in x
            .chunks_exact(SUB)
            .zip(levels.chunks_exact_mut(SUB))
            .enumerate()
        {
            let d = block.d.to_f32() * scales[g] as f32;
            if d == 0.0 {
                continue;
            }
            let m = block.dmin.to_f32() * mins[g] as f32;
            for (out, &v) in lg.iter_mut().zip(xg) {
                *out = nearest_int((v + m) / d).clamp(0, 15) as u8;
            }
        }

        // Two sub-blocks share a byte span: the even one in the low nibbles, the odd in the
        // high - the same split `dequantize` reads back.
        for (packed, pair) in block
            .qs
            .chunks_exact_mut(SUB)
            .zip(levels.chunks_exact(2 * SUB))
        {
            let (low, high) = pair.split_at(SUB);
            for ((out, &a), &b) in packed.iter_mut().zip(low).zip(high) {
                *out = a | (b << 4);
            }
        }
    }
}
