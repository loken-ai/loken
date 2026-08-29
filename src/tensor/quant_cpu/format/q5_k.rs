//! q5_k: block layout, dequantiser and quantiser together.
//!
//! Its vectorised kernel is still in `super::avx`, which shares primitives across
//! formats; it follows once those are factored out.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qh: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () =
    assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5K>());
impl BlockFormat for BlockQ5K {
    const DTYPE: GgmlDType = GgmlDType::Q5K;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            dmin: f16::ZERO,
            scales: [0; K_SCALE_SIZE],
            qh: [0; QK_K / 8],
            qs: [0; QK_K / 2],
        }
    }

    #[allow(unreachable_code)]
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_q5k_q8k(xs, ys);

        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const SUB: usize = 32;
        // As in q4_K: the minimum is constant over a sub-block, so it multiplies the sum of
        // the activations that `bsums` already holds - two sixteen-value entries per
        // thirty-two-value sub-block.
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
                        .enumerate()
                        .map(|(i, (&packed, &a))| {
                            let fifth = (x.qh[i] >> s) & 1;
                            (((packed >> shift) & 0x0F) | (fifth << 4)) as i32 * a as i32
                        })
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
                for (i, (out, packed)) in
                    y[s * SUB..(s + 1) * SUB].iter_mut().zip(group).enumerate()
                {
                    let fifth = (block.qh[i] >> s) & 1;
                    let q = ((packed >> shift) & 0x0F) | (fifth << 4);
                    *out = scale * q as f32 - offset;
                }
            }
        }
    }
}

/// Quantise whole blocks, optionally against an importance matrix.
///
/// q4_K's encoder with one more bit per value. The two stages, and what the importance matrix
/// changes in each of them, are the same; see `q4_k::encode` for why the second stage matters
/// and how it was once lost. What differs is only where the fifth bit goes: `qh` holds thirty-
/// two bytes read by every sub-block, bit `s` of byte `i` belonging to sub-block `s`.
fn encode(xs: &[f32], ys: &mut [BlockQ5K], imatrix: Option<&[f32]>, n_per_row: usize) {
    const SUB: usize = 32;
    const GROUPS: usize = QK_K / SUB;
    /// Six bits for a scale and six for a minimum.
    const LEVELS: u8 = 63;

    for (idx, (block, x)) in blocks_with_input(xs, ys).enumerate() {
        let row = imatrix.map(|m| {
            let base = (idx % (n_per_row / QK_K)) * QK_K;
            &m[base..base + QK_K]
        });
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
            (*scale, *min) = fit_scale_and_offset(31, xg, &w);
        }

        let (mut ls, mut lm) = ([0u8; GROUPS], [0u8; GROUPS]);
        let (d, dmin) = if row.is_some() {
            (
                fit_unsigned_scale(LEVELS, &scales, &mut ls, &carried),
                fit_unsigned_scale(LEVELS, &mins, &mut lm, &carried),
            )
        } else {
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
                *out = nearest_int((v + m) / d).clamp(0, 31) as u8;
            }
        }

        block.qs = [0; QK_K / 2];
        block.qh = [0; QK_K / 8];
        for (s, sub) in levels.chunks_exact(SUB).enumerate() {
            let span = (s / 2) * SUB;
            let shift = if s % 2 == 0 { 0 } else { 4 };
            for (i, &code) in sub.iter().enumerate() {
                block.qs[span + i] |= (code & 0x0F) << shift;
                block.qh[i] |= (code >> 4) << s;
            }
        }
    }
}
