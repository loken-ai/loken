//! q3_k: block layout, dequantiser and quantiser together.
//!
//! Its vectorised kernel is still in `super::avx`, which shares primitives across
//! formats; it follows once those are factored out.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3K {
    pub(crate) hmask: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) scales: [u8; 12],
    pub(crate) d: f16,
}
const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

/// The sixteen six-bit scales a q3_K block packs into twelve bytes, already un-biased.
///
/// Four bits come from a nibble of bytes 0..8 and two from a two-bit lane of bytes 8..12, so
/// each scale is assembled from two places. The scale is SIGNED with a bias of 32, which is
/// why this returns `i8` and subtracts here rather than at every use.
///
/// The C does this with four lines of 32-bit masking over `0x0f0f0f0f` and `0x03030303`, then
/// reinterprets the buffer as `i8` through a raw pointer. Written per scale it is the same
/// arithmetic and says which bits it is fetching.
fn q3k_scales(packed: &[u8; 12]) -> [i8; 16] {
    let mut out = [0i8; 16];
    for (i, slot) in out.iter_mut().enumerate() {
        let (group, j) = (i / 4, i % 4);
        let byte = packed[j + 4 * (group % 2)];
        let low = if group < 2 { byte & 0x0F } else { byte >> 4 };
        let high = (packed[8 + j] >> (2 * group)) & 0x03;
        *slot = ((low | (high << 4)) as i8) - 32;
    }
    out
}

/// The inverse of [`q3k_scales`]: sixteen six-bit levels, each already biased into 0..63,
/// folded into twelve bytes.
///
/// Four bits of each go into a nibble of bytes 0..8 - the first eight low, the next eight high
/// - and the remaining two into a two-bit lane of bytes 8..12.
fn pack_q3k_scales(levels: &[i32; 16]) -> [u8; 12] {
    let mut packed = [0u8; 12];
    for (i, &level) in levels.iter().enumerate() {
        if i < 8 {
            packed[i] |= (level & 0x0F) as u8;
        } else {
            packed[i - 8] |= ((level & 0x0F) << 4) as u8;
        }
        packed[8 + i % 4] |= ((level >> 4) << (2 * (i / 4))) as u8;
    }
    packed
}

impl BlockFormat for BlockQ3K {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            hmask: [0; QK_K / 8],
            qs: [0; QK_K / 4],
            scales: [0; 12],
            d: f16::ZERO,
        }
    }

    #[allow(unreachable_code)]
    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return avx::vec_dot_q3k_q8k(xs, ys);

        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        // The same walk as `dequantize`, gathering per sub-block as exact integers: three-bit
        // codes against an eight-bit activation, sixteen at a time, stay far inside i32.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let scales = q3k_scales(&x.scales);
                let mut dot = 0f32;
                for (s, &sc) in scales.iter().enumerate() {
                    let (chunk, t) = (s / 8, s % 8);
                    let (lane, half) = (t / 2, t % 2);
                    let qs = &x.qs[32 * chunk + 16 * half..][..16];
                    let hmask = &x.hmask[16 * half..][..16];
                    let bit = 1u8 << (4 * chunk + lane);
                    let acts = &y.qs[128 * chunk + 32 * lane + 16 * half..][..16];
                    let sum: i32 = qs
                        .iter()
                        .zip(acts)
                        .enumerate()
                        .map(|(k, (q, &a))| {
                            let low = ((q >> (2 * lane)) & 0x03) as i32;
                            let third = if hmask[k] & bit == 0 { -4 } else { 0 };
                            (low + third) * a as i32
                        })
                        .sum();
                    dot += sc as f32 * sum as f32;
                }
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

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            let scales = q3k_scales(&block.scales);
            for (s, &sc) in scales.iter().enumerate() {
                let (chunk, t) = (s / 8, s % 8);
                let (lane, half) = (t / 2, t % 2);
                let scale = d * sc as f32;
                let qs = &block.qs[32 * chunk + 16 * half..][..16];
                let hmask = &block.hmask[16 * half..][..16];
                let bit = 1u8 << (4 * chunk + lane);
                let out = &mut y[128 * chunk + 32 * lane + 16 * half..][..16];
                for (k, (o, q)) in out.iter_mut().zip(qs).enumerate() {
                    let low = ((q >> (2 * lane)) & 0x03) as i8;
                    let third = if hmask[k] & bit == 0 { -4 } else { 0 };
                    *o = scale * (low + third) as f32;
                }
            }
        }
    }
}

/// Quantise whole blocks, optionally against an importance matrix.
///
/// Two stages, as in q4_K: each group of sixteen is fitted to a scale, then the sixteen scales
/// are themselves quantised to six signed bits, which is what `d` scales. There is no minimum
/// here - q3_K's codes are centred - so only one family is fitted rather than two.
///
/// The importance matrix changes both stages: it weighs each value by `imatrix . sqrt(σ² + x²)`
/// and fits the sixteen scales by weighted least squares against what each group carries,
/// instead of dividing them by the largest.
fn encode(xs: &[f32], ys: &mut [BlockQ3K], imatrix: Option<&[f32]>, n_per_row: usize) {
    const SUB: usize = 16;
    const GROUPS: usize = QK_K / SUB;
    /// Codes run -4..3, so the fit is asked for four levels either side of zero.
    const CODE_MAX: i32 = 4;
    /// Scales are six bits signed, biased by this on the way into the file.
    const SCALE_BIAS: i32 = 32;

    for (idx, (block, x)) in blocks_with_input(xs, ys).enumerate() {
        let row = imatrix.map(|m| {
            let base = (idx % (n_per_row / QK_K)) * QK_K;
            &m[base..base + QK_K]
        });
        let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;

        let mut scales = [0f32; GROUPS];
        let mut carried = [0f32; GROUPS];
        let mut w = [0f32; SUB];
        // The fitted levels are discarded: they are re-derived below from the scales as they
        // end up ROUNDED into the block, which is what the decoder will read.
        let mut fitted = [0i8; SUB];

        for (g, (xg, scale)) in x.chunks_exact(SUB).zip(scales.iter_mut()).enumerate() {
            match row {
                Some(r) => {
                    for (out, (&iw, &v)) in w.iter_mut().zip(r[SUB * g..].iter().zip(xg)) {
                        *out = iw * (sigma2 + v * v).sqrt();
                    }
                }
                None => magnitude_weights(xg, &mut w),
            }
            carried[g] = w.iter().sum();
            *scale = fit_signed_scale(CODE_MAX, xg, &mut fitted, &w);
        }

        let mut levels = [0i32; GROUPS];
        block.d = f16::from_f32(match row {
            Some(_) => {
                let mut ls = [0i8; GROUPS];
                let d = fit_signed_scale(SCALE_BIAS, &scales, &mut ls, &carried);
                for (out, &l) in levels.iter_mut().zip(&ls) {
                    *out = l as i32;
                }
                d
            }
            None => {
                // The extreme keeps its sign: scales may be negative, and -32 is the end of
                // the range that exists.
                let extreme = scales
                    .iter()
                    .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
                if extreme == 0.0 {
                    0.0
                } else {
                    let inv = -(SCALE_BIAS as f32) / extreme;
                    for (out, &s) in levels.iter_mut().zip(&scales) {
                        *out = nearest_int(inv * s).clamp(-SCALE_BIAS, SCALE_BIAS - 1) + SCALE_BIAS;
                    }
                    1.0 / inv
                }
            }
        });
        block.scales = pack_q3k_scales(&levels);

        let scales = q3k_scales(&block.scales);
        let mut codes = [0i8; QK_K];
        for (g, (xg, cg)) in x
            .chunks_exact(SUB)
            .zip(codes.chunks_exact_mut(SUB))
            .enumerate()
        {
            let d = block.d.to_f32() * scales[g] as f32;
            if d == 0.0 {
                continue;
            }
            for (out, &v) in cg.iter_mut().zip(xg) {
                *out = (nearest_int(v / d).clamp(-CODE_MAX, CODE_MAX - 1) + CODE_MAX) as i8;
            }
        }

        // The biased code needs three bits. Its top one goes to `hmask`, where value `v` owns
        // bit `v / 32` of byte `v % 32`; the low two stay in `qs`, four values to a byte.
        block.hmask.fill(0);
        for (v, code) in codes.iter_mut().enumerate() {
            if *code >= CODE_MAX as i8 {
                block.hmask[v % 32] |= 1u8 << (v / 32);
                *code -= CODE_MAX as i8;
            }
        }
        for (half, quad) in codes.chunks_exact(128).enumerate() {
            for l in 0..32 {
                block.qs[32 * half + l] =
                    (quad[l] | (quad[l + 32] << 2) | (quad[l + 64] << 4) | (quad[l + 96] << 6))
                        as u8;
            }
        }
    }
}
