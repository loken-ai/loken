//! The AVX-512 dot products, answerable to the AVX2 ones in [`super::avx`].
//!
//! Same function names and signatures, so `dot` chooses between the two tiers in one place. The
//! integer sums are exact (`vpdpbusd` accumulates in i32 with no saturation), and each kernel folds
//! its 512-bit lanes back onto the 256-bit lane layout of its AVX2 twin before the first float
//! operation, so the two return the same bits.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::missing_safety_doc
)]

pub(crate) mod block32;
pub(crate) mod float;
pub(crate) mod repack_inline;
pub(crate) mod repack_q4k;
pub(crate) mod repack_q5k;
pub(crate) mod repack_q6k;
pub(crate) mod repack_x8;
pub(crate) mod superblock;

#[cfg(test)]
mod parity;

pub(crate) use superblock::{
    vec_dot_q2k_q8k, vec_dot_q3k_q8k, vec_dot_q4k_q8k, vec_dot_q5k_q8k, vec_dot_q6k_q8k,
    vec_dot_q8k_q8k,
};

/// Guard in front of every AVX-512 `#[target_feature]` entry point of this module.
#[inline]
pub(crate) fn require_avx512() {
    assert!(
        crate::cpu::hardware_has_avx512(),
        "an AVX-512 kernel was called on a CPU without avx512f/bw/vl/dq/vnni"
    );
}
