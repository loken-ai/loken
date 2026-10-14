//! q8_K - the activation side of every k-quant dot product.
//!
//! `y = q . d` over a 256-value block, with `d` kept in f32 rather than f16: this format is
//! never written to a file, it is what a row of activations is quantised into just before a
//! matmul, so the two bytes are not worth the rounding.
//!
//! It also carries `bsums`, the sum of each sixteen codes. The formats that reconstruct as
//! `q.d + m` need `Σ activations` per sub-block to apply their minimum, and computing it here
//! once - where the codes are already in registers - spares every one of them a second pass.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 4 + QK_K + QK_K / 8);

/// The widest code, and what the block's extreme value is mapped onto.
const Q8_K_MAX: f32 = 127.0;

impl BlockFormat for BlockQ8K {
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            for (out, &q) in y.iter_mut().zip(&block.qs) {
                *out = q as f32 * block.d;
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx512() {
            // SAFETY: dispatch_avx512 implies the CPU has every feature `quantize_avx512` enables.
            return unsafe { quantize_avx512(xs, ys) };
        }
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            // SAFETY: dispatch_avx2 implies the CPU has every feature `quantize_avx2` enables.
            return unsafe { quantize_avx2(xs, ys) };
        }
        quantize_blocks(xs, ys)
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx512() {
            return avx512::vec_dot_q8k_q8k(xs, ys);
        }
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            return avx::vec_dot_q8k_q8k(xs, ys);
        }

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        // Both sides are eight-bit codes, so 256 products stay well inside i32 and the two
        // scales come out of the sum.
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let dot: i32 =
                    x.qs.iter()
                        .zip(&y.qs)
                        .map(|(&a, &b)| a as i32 * b as i32)
                        .sum();
                dot as f32 * x.d * y.d
            })
            .sum()
    }
}

/// The activation quantiser. Its loops vectorise to the target features of the function that
/// inlines it.
#[inline(always)]
fn quantize_blocks(xs: &[f32], ys: &mut [BlockQ8K]) {
    for (block, x) in blocks_with_input(xs, ys) {
        // The extreme keeps its sign and is mapped onto -127, not +127: the codes reach
        // -128, and starting from the negative end is what keeps the AVX2 unpacking of the
        // k-quant kernels free of a sign fixup.
        let extreme = x
            .iter()
            .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        if extreme == 0.0 {
            *block = BlockQ8K::zeros();
            continue;
        }
        let inv = -Q8_K_MAX / extreme;
        block.d = 1.0 / inv;
        for (q, &v) in block.qs.iter_mut().zip(x) {
            // Only the upper end needs clamping: the extreme lands exactly on -127, but a
            // value of the opposite sign can round past +127.
            *q = nearest_int(inv * v).min(Q8_K_MAX as i32) as i8;
        }
        for (sum, sub) in block.bsums.iter_mut().zip(block.qs.chunks_exact(16)) {
            *sum = sub.iter().map(|&q| q as i16).sum();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn quantize_avx2(xs: &[f32], ys: &mut [BlockQ8K]) {
    quantize_blocks(xs, ys)
}

/// The same loops, left to vectorise over 512-bit registers. Every step is per element (the
/// extreme is a max, which does not depend on order), so the codes are those of the other builds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn quantize_avx512(xs: &[f32], ys: &mut [BlockQ8K]) {
    quantize_blocks(xs, ys)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn the_avx512_quantiser_has_the_codes_of_the_avx2_one() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0
        };
        for blocks in [1usize, 2, 5, 16] {
            let mut xs: Vec<f32> = (0..blocks * QK_K).map(|_| next()).collect();
            // A block of zeros, and one whose extreme is positive and lopsided.
            xs[..QK_K].fill(0.0);
            if blocks > 1 {
                xs[QK_K + 3] = 500.0;
            }
            let (mut a, mut b) = (
                vec![BlockQ8K::zeros(); blocks],
                vec![BlockQ8K::zeros(); blocks],
            );
            unsafe {
                quantize_avx2(&xs, &mut a);
                quantize_avx512(&xs, &mut b);
            }
            assert_eq!(a, b, "{blocks} blocks");
        }
    }
}
