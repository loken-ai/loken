//! Every AVX-512 kernel against the scalar definition and against its AVX2 twin.
//!
//! The AVX2 kernels are pinned to the scalar ones by `avx::parity`; here the 512-bit kernel must
//! agree with the scalar definition to the same tolerance and return the AVX2 kernel's exact bits;
//! it folds onto the same lane layout before any float operation.

use super::*;
use crate::tensor::quant_cpu::avx::parity::{close, operands};
use crate::tensor::quant_cpu::*;

fn agrees<T: BlockFormat>(
    name: &str,
    kernel: fn(&[T], &[T::ActivationBlock]) -> f32,
    twin: fn(&[T], &[T::ActivationBlock]) -> f32,
) {
    for seed in [1u32, 17, 4321, 99_991] {
        for blocks in [1usize, 2, 3, 7, 16] {
            let (w, a) = operands::<T>(seed, blocks);
            let case = format!("seed {seed}, {blocks} blocks");
            let got = kernel(&w, &a);
            close(name, &case, got, T::dot_scalar(&w, &a));
            assert_eq!(
                got.to_bits(),
                twin(&w, &a).to_bits(),
                "{name} ({case}): not the bits of the AVX2 kernel"
            );
        }
    }
}

#[test]
fn every_avx512_kernel_agrees_with_the_scalar_and_avx2_definitions() {
    if !crate::cpu::hardware_has_avx512() {
        return;
    }
    agrees::<BlockQ4K>("q4_K", vec_dot_q4k_q8k, avx::vec_dot_q4k_q8k);
    agrees::<BlockQ2K>("q2_K", vec_dot_q2k_q8k, avx::vec_dot_q2k_q8k);
    agrees::<BlockQ3K>("q3_K", vec_dot_q3k_q8k, avx::vec_dot_q3k_q8k);
    agrees::<BlockQ5K>("q5_K", vec_dot_q5k_q8k, avx::vec_dot_q5k_q8k);
    agrees::<BlockQ6K>("q6_K", vec_dot_q6k_q8k, avx::vec_dot_q6k_q8k);
    agrees::<BlockQ8K>("q8_K", vec_dot_q8k_q8k, avx::vec_dot_q8k_q8k);
    agrees::<BlockQ4_0>("q4_0", block32::vec_dot_q4_0_q8_0, avx::vec_dot_q4_0_q8_0);
    agrees::<BlockQ5_0>("q5_0", block32::vec_dot_q5_0_q8_0, avx::vec_dot_q5_0_q8_0);
    agrees::<BlockQ8_0>("q8_0", block32::vec_dot_q8_0_q8_0, avx::vec_dot_q8_0_q8_0);
    agrees::<BlockMxFp4>(
        "mxfp4",
        block32::vec_dot_mxfp4_q8_0,
        avx::vec_dot_mxfp4_q8_0,
    );
    agrees::<BlockQ4_1>("q4_1", block32::vec_dot_q4_1_q8_1, avx::vec_dot_q4_1_q8_1);
    agrees::<BlockQ5_1>("q5_1", block32::vec_dot_q5_1_q8_1, avx::vec_dot_q5_1_q8_1);
    agrees::<BlockQ8_1>("q8_1", block32::vec_dot_q8_1_q8_1, avx::vec_dot_q8_1_q8_1);
}

/// Edge blocks: zero scales, saturated codes and activations, where an exact integer path and a
/// saturating one part ways.
#[test]
fn q4k_agrees_on_saturated_blocks() {
    if !crate::cpu::hardware_has_avx512() {
        return;
    }
    let mut w = vec![BlockQ4K::zeros(); 3];
    let mut a = vec![BlockQ8K::zeros(); 3];
    for (i, blk) in w.iter_mut().enumerate() {
        blk.d = half::f16::from_f32(1.0);
        blk.dmin = half::f16::from_f32(0.5);
        blk.scales = [if i == 0 { 0 } else { 0xFF }; 12];
        blk.qs = [if i == 2 { 0x00 } else { 0xFF }; 128];
    }
    for (i, blk) in a.iter_mut().enumerate() {
        blk.d = 1.0;
        blk.qs = [if i == 1 { -128 } else { 127 }; 256];
        for (j, s) in blk.bsums.iter_mut().enumerate() {
            *s = blk.qs[16 * j..16 * j + 16].iter().map(|&v| v as i16).sum();
        }
    }
    assert_eq!(
        vec_dot_q4k_q8k(&w, &a).to_bits(),
        avx::vec_dot_q4k_q8k(&w, &a).to_bits()
    );
    close(
        "q4_K saturated",
        "3 blocks",
        vec_dot_q4k_q8k(&w, &a),
        BlockQ4K::dot_scalar(&w, &a),
    );
}
