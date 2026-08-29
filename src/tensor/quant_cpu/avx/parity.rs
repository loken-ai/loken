//! Every vectorised dot product against the scalar one it replaces.
//!
//! `dot` reaches an AVX2 kernel where one exists and falls through to
//! `dot_scalar` otherwise. Only the scalar path is pinned against ggml by
//! `oracle_parity`, so it is the definition here, and a kernel that disagrees
//! with it is a wrong answer that runs fast - which no other test would notice.
//!
//! Each format is checked twice: the kernel called BY NAME, so the comparison
//! cannot degenerate into scalar-against-scalar on a machine without AVX2, and
//! the `dot` dispatch, because a kernel the dispatch never reaches is the other
//! way this goes wrong.

use super::block32::*;
use super::superblock::*;
use crate::tensor::quant_cpu::*;

/// A deterministic signal with both signs and a wide dynamic range: a block
/// whose values all sit on one side, or within one octave, hides the clipping
/// and scale-search behaviour the kernels have to agree about.
fn signal(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            let unit = (h >> 8) as f32 / (1 << 24) as f32 - 0.5;
            // Spread over four octaves so some blocks clip and some do not.
            unit * (1 << (i % 5)) as f32
        })
        .collect()
}

/// The quantised operands of one case: `blocks` blocks of weights and the
/// matching activations.
fn operands<T: BlockFormat>(seed: u32, blocks: usize) -> (Vec<T>, Vec<T::ActivationBlock>) {
    let n = blocks * T::BLOCK_LEN;
    let mut weights = vec![T::zeros(); blocks];
    T::quantize(&signal(seed, n), &mut weights);

    let mut acts = vec![T::ActivationBlock::zeros(); n / T::ActivationBlock::BLOCK_LEN];
    T::ActivationBlock::quantize(&signal(seed ^ 0x5bf0_3635, n), &mut acts);
    (weights, acts)
}

/// Relative, because the two paths sum in different orders; the point is that
/// they compute the same function, not the same rounding.
fn close(name: &str, case: &str, fast: f32, slow: f32) {
    let tolerance = 1e-3 * slow.abs().max(1.0);
    assert!(
        (fast - slow).abs() <= tolerance,
        "{name} ({case}): {fast} disagrees with the scalar definition {slow} \
         (tolerance {tolerance:e})"
    );
}

/// The kernel, by name, against the scalar definition.
fn kernel_agrees<T: BlockFormat>(name: &str, kernel: fn(&[T], &[T::ActivationBlock]) -> f32) {
    for seed in [1u32, 17, 4321] {
        for blocks in [1usize, 2, 7] {
            let (w, a) = operands::<T>(seed, blocks);
            close(
                name,
                &format!("seed {seed}, {blocks} blocks"),
                kernel(&w, &a),
                T::dot_scalar(&w, &a),
            );
        }
    }
}

/// The dispatch, against the scalar definition - a kernel silently not reached
/// is the other way this goes wrong.
fn dispatch_agrees<T: BlockFormat>(name: &str) {
    for seed in [1u32, 17, 4321] {
        for blocks in [1usize, 2, 7] {
            let (w, a) = operands::<T>(seed, blocks);
            close(
                name,
                &format!("dispatch, seed {seed}, {blocks} blocks"),
                T::dot(&w, &a),
                T::dot_scalar(&w, &a),
            );
        }
    }
}

#[test]
#[cfg(target_feature = "avx2")]
fn every_avx2_kernel_agrees_with_its_scalar_definition() {
    kernel_agrees::<BlockQ4_0>("q4_0", vec_dot_q4_0_q8_0);
    kernel_agrees::<BlockQ4_1>("q4_1", vec_dot_q4_1_q8_1);
    kernel_agrees::<BlockQ5_0>("q5_0", vec_dot_q5_0_q8_0);
    kernel_agrees::<BlockQ5_1>("q5_1", vec_dot_q5_1_q8_1);
    kernel_agrees::<BlockQ8_0>("q8_0", vec_dot_q8_0_q8_0);
    kernel_agrees::<BlockQ8_1>("q8_1", vec_dot_q8_1_q8_1);
    kernel_agrees::<BlockMxFp4>("mxfp4", vec_dot_mxfp4_q8_0);
    kernel_agrees::<BlockQ2K>("q2_K", vec_dot_q2k_q8k);
    kernel_agrees::<BlockQ3K>("q3_K", vec_dot_q3k_q8k);
    kernel_agrees::<BlockQ4K>("q4_K", vec_dot_q4k_q8k);
    kernel_agrees::<BlockQ5K>("q5_K", vec_dot_q5k_q8k);
    kernel_agrees::<BlockQ6K>("q6_K", vec_dot_q6k_q8k);
    kernel_agrees::<BlockQ8K>("q8_K", vec_dot_q8k_q8k);
}

#[test]
fn every_dispatch_agrees_with_its_scalar_definition() {
    dispatch_agrees::<BlockQ4_0>("q4_0");
    dispatch_agrees::<BlockQ4_1>("q4_1");
    dispatch_agrees::<BlockQ5_0>("q5_0");
    dispatch_agrees::<BlockQ5_1>("q5_1");
    dispatch_agrees::<BlockQ8_0>("q8_0");
    dispatch_agrees::<BlockQ8_1>("q8_1");
    dispatch_agrees::<BlockMxFp4>("mxfp4");
    dispatch_agrees::<BlockQ2K>("q2_K");
    dispatch_agrees::<BlockQ3K>("q3_K");
    dispatch_agrees::<BlockQ4K>("q4_K");
    dispatch_agrees::<BlockQ5K>("q5_K");
    dispatch_agrees::<BlockQ6K>("q6_K");
    dispatch_agrees::<BlockQ8K>("q8_K");
}
