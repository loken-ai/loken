//! q2_K - 256 values at two bits each, sixteen to a sub-block.
//!
//! `y = d . sc[s] . q - dmin . m[s]`, with `q` an unsigned two-bit code and each sub-block's
//! scale and minimum sharing one byte, four bits apiece. The two f16 at the end scale those
//! nibbles, so the block spends thirty-two bits of scale on sixteen sub-blocks - proportionally
//! far more than the wider formats, which is what makes two bits per value usable at all.
//!
//! Within a half of 128 values the eight sub-blocks read the same 32 bytes of `qs` at shifts
//! that advance by two, alternating between its two sixteen-byte spans.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2K {
    pub(crate) scales: [u8; QK_K / 16],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
    pub(crate) dmin: f16,
}
const _: () = assert!(std::mem::size_of::<BlockQ2K>() == QK_K / 16 + QK_K / 4 + 4);

/// Sixteen values share a scale and a minimum.
const SUB: usize = 16;
const GROUPS: usize = QK_K / SUB;
/// Four bits for a scale and four for a minimum, in one byte.
const LEVELS: u8 = 15;

/// Where sub-block `s` reads its codes: the byte its value `l` lives in, and the shift.
fn placement(s: usize) -> (usize, u32) {
    let (half, t) = (s / 8, s % 8);
    (32 * half + SUB * (t % 2), 2 * (t / 2) as u32)
}

impl BlockFormat for BlockQ2K {
    const DTYPE: GgmlDType = GgmlDType::Q2K;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            scales: [0; QK_K / 16],
            qs: [0; QK_K / 4],
            d: f16::ZERO,
            dmin: f16::ZERO,
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let (d, dmin) = (block.d.to_f32(), block.dmin.to_f32());
            for (s, &packed) in block.scales.iter().enumerate() {
                let (base, shift) = placement(s);
                let scale = d * (packed & 0x0F) as f32;
                let offset = dmin * (packed >> 4) as f32;
                let out = &mut y[SUB * s..SUB * (s + 1)];
                for (o, &q) in out.iter_mut().zip(&block.qs[base..base + SUB]) {
                    *o = scale * ((q >> shift) & 0x03) as f32 - offset;
                }
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        encode(xs, ys, None, 0);
    }

    fn quantize_guided(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        encode(xs, ys, Some(imatrix_weights), n_per_row);
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx512() {
            return avx512::vec_dot_q2k_q8k(xs, ys);
        }
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            return avx::vec_dot_q2k_q8k(xs, ys);
        }

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        // The minimum is constant over a sub-block, so it multiplies the sum of the sixteen
        // activations - which is exactly one `bsums` entry, sixteen being this format's
        // sub-block size as well.
        let mut total = -0.0f32;
        for (x, y) in xs.iter().zip(ys) {
            let (mut scaled, mut offset) = (0i32, 0i32);
            for (s, &packed) in x.scales.iter().enumerate() {
                let (base, shift) = placement(s);
                let acts = &y.qs[SUB * s..SUB * (s + 1)];
                let dot: i32 = x.qs[base..base + SUB]
                    .iter()
                    .zip(acts)
                    .map(|(&q, &a)| ((q >> shift) & 0x03) as i32 * a as i32)
                    .sum();
                scaled += (packed & 0x0F) as i32 * dot;
                offset += (packed >> 4) as i32 * y.bsums[s] as i32;
            }
            total += combine(x, y.d, scaled, offset);
        }
        total
    }
}

/// One block's share of a dot product from its two integer sums: `scaled`, the codes against the
/// activations under each sub-block's scale, and `offset`, the activation sums under each
/// sub-block's minimum. Every kernel ends a block here and adds the shares in block order from
/// -0, so the product is the same bits whichever kernel formed the sums, a card's included.
#[inline(always)]
pub(crate) fn combine(x: &BlockQ2K, y_d: f32, scaled: i32, offset: i32) -> f32 {
    y_d * (x.d.to_f32() * scaled as f32 - x.dmin.to_f32() * offset as f32)
}

/// Quantise whole blocks, optionally against an importance matrix.
///
/// The two stages of q4_K, at sixteen values per sub-block instead of thirty-two and four bits
/// of scale instead of six. The variance floor differs too - `Σx²/256` here against twice that
/// elsewhere - which is a property of this format's fit, not an oversight.
fn encode(xs: &[f32], ys: &mut [BlockQ2K], imatrix: Option<&[f32]>, n_per_row: usize) {
    for (idx, (block, x)) in blocks_with_input(xs, ys).enumerate() {
        let row = imatrix.map(|m| {
            let base = (idx % (n_per_row / QK_K)) * QK_K;
            &m[base..base + QK_K]
        });
        let sigma2 = x.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;

        let mut scales = [0f32; GROUPS];
        let mut mins = [0f32; GROUPS];
        let mut carried = [0f32; GROUPS];
        let mut w = [0f32; QK_K];

        for (g, (xg, wg)) in x.chunks_exact(SUB).zip(w.chunks_exact_mut(SUB)).enumerate() {
            match row {
                Some(r) => {
                    for (out, (&iw, &v)) in wg.iter_mut().zip(r[SUB * g..].iter().zip(xg)) {
                        *out = iw * (sigma2 + v * v).sqrt();
                    }
                }
                None => magnitude_weights(xg, wg),
            }
            carried[g] = wg.iter().sum();
        }
        let fitted = fit_scales_and_offsets::<GROUPS>(3, x, &w);
        for (g, (scale, min)) in fitted.into_iter().enumerate() {
            (scales[g], mins[g]) = (scale, min);
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
        for (packed, (&s, &m)) in block.scales.iter_mut().zip(ls.iter().zip(&lm)) {
            *packed = s | (m << 4);
        }
        block.d = f16::from_f32(d);
        block.dmin = f16::from_f32(dmin);

        block.qs.fill(0);
        for (s, &packed) in block.scales.iter().enumerate() {
            let scale = block.d.to_f32() * (packed & 0x0F) as f32;
            if scale == 0.0 {
                continue;
            }
            let offset = block.dmin.to_f32() * (packed >> 4) as f32;
            let (base, shift) = placement(s);
            for (i, &v) in x[SUB * s..SUB * (s + 1)].iter().enumerate() {
                let code = nearest_int((v + offset) / scale).clamp(0, 3) as u8;
                block.qs[base + i] |= code << shift;
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    /// Every kernel's dot is the scalar dot bit for bit: the integer sums of a block in any order,
    /// then `combine` and the blocks added in order. A card's kernel ends a block the same way, so
    /// an expert's product does not depend on where it is formed.
    #[test]
    fn the_vector_dots_are_the_scalar_dot_bit_for_bit() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        for blocks in [1usize, 3, 9] {
            let w: Vec<f32> = (0..blocks * QK_K).map(|_| next() * 0.1).collect();
            let x: Vec<f32> = (0..blocks * QK_K).map(|_| next() * 8.0).collect();
            let mut wq = vec![BlockQ2K::zeros(); blocks];
            BlockQ2K::quantize(&w, &mut wq);
            let mut xq = vec![BlockQ8K::zeros(); blocks];
            BlockQ8K::quantize(&x, &mut xq);
            let want = BlockQ2K::dot_scalar(&wq, &xq).to_bits();
            if crate::cpu::hardware_has_avx2() {
                let got = crate::tensor::quant_cpu::avx::superblock::vec_dot_q2k_q8k(&wq, &xq);
                assert_eq!(got.to_bits(), want, "avx2, {blocks} blocks");
            }
            if crate::cpu::hardware_has_avx512() {
                let got = crate::tensor::quant_cpu::avx512::superblock::vec_dot_q2k_q8k(&wq, &xq);
                assert_eq!(got.to_bits(), want, "avx512, {blocks} blocks");
            }
        }
    }
}
