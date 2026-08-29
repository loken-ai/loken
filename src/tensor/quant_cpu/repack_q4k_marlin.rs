//! Marlin-style repack for GGUF Q4_K weights (decode GEMV bandwidth lever).
//!
//! vLLM's AWQ-Marlin beats our `mmvq` Q4_K decode by ~34% on the same GPU
//! (deepcoder 2.5K: 59.5 vs 79.7 tok/s, sm_120). The kernel-efficiency part of
//! that gap (mmvq ~72% DRAM bandwidth) comes from two things in the GGUF Q4_K
//! block (`d:f16, dmin:f16, scales:[u8;12] (6-bit packed), qs:[u8;128]`):
//!   1. the 8 sub-block scales+mins are 6-bit-packed -> a serial unpack per block;
//!   2. `d`/`dmin` are interleaved with the bulk `qs`.
//!
//! This module repacks a Q4_K weight `[N, K]` (row-major super-blocks of 256)
//! into separate, aligned arrays so a GEMV kernel reads vectorizable, coalesced
//! loads and skips the in-kernel scale unpack:
//!   - `qs`     : the 4-bit quants, unchanged byte layout (128 B / super-block);
//!   - `scales` : pre-unpacked 6-bit sub-block scales (8 per super-block);
//!   - `mins`   : pre-unpacked 6-bit sub-block mins   (8 per super-block);
//!   - `d`,`dmin`: the per-super-block f16 scales, separated.
//!
//! This is the foundation for the (gated, default-off) Marlin Q4_K decode
//! kernel; it is quality-preserving (same weights, lossless reorder). The
//! coalescing reorder of `qs` and the CUDA kernel are later increments.

// Transcribed kernels: the index arithmetic IS the layout, the argument lists are the
// reference's, and the `unsafe fn`s wrap intrinsics whose contract is the intrinsic's.
// Named rather than `clippy::all` so anything else here still gets reported.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    clippy::type_complexity,
    clippy::redundant_closure
)]

// The 12-byte six-bit scale/min packing is q4_K's, whatever the target layout,
// so it is read through the one statement of it.
use super::repack_q4k::sub_scales_and_mins_bytes;
use half::f16;

/// Weights per super-block. The count is a fact about the file format that several formats
/// quote, so it is stated once beside the other block geometry and re-exported here.
pub use super::QK_K;

/// Bytes in one GGUF Q4_K super-block (`d,dmin: 2xf16`, `scales: 12`, `qs: 128`).
pub const Q4K_BLOCK_BYTES: usize = 144;
/// Sub-blocks per super-block (each 32 weights, one 6-bit scale + min).
pub const SUBBLOCKS: usize = 8;

/// Repacked Q4_K weight, decode-GEMV friendly.
pub struct RepackedQ4K {
    pub n: usize,
    pub k: usize,
    /// `[N * K/2]` - 4-bit quants (unchanged byte layout for now).
    pub qs: Vec<u8>,
    /// `[N * K/256 * 8]` - pre-unpacked 6-bit sub-block scales.
    pub scales: Vec<u8>,
    /// `[N * K/256 * 8]` - pre-unpacked 6-bit sub-block mins.
    pub mins: Vec<u8>,
    /// `[N * K/256]` - per-super-block d.
    pub d: Vec<f16>,
    /// `[N * K/256]` - per-super-block dmin.
    pub dmin: Vec<f16>,
}

/// Repack raw GGUF Q4_K block bytes (`[N, K]`, row-major super-blocks) into
/// `RepackedQ4K`. `data.len()` must be `N * (K/256) * 144`.
pub fn repack_q4k(data: &[u8], n: usize, k: usize) -> RepackedQ4K {
    assert!(k.is_multiple_of(QK_K), "K={k} not a multiple of {QK_K}");
    let blocks_per_row = k / QK_K;
    let nblocks = n * blocks_per_row;
    assert_eq!(
        data.len(),
        nblocks * Q4K_BLOCK_BYTES,
        "Q4_K byte count mismatch"
    );

    let mut qs = vec![0u8; n * k / 2];
    let mut scales = vec![0u8; nblocks * SUBBLOCKS];
    let mut mins = vec![0u8; nblocks * SUBBLOCKS];
    let mut d = vec![f16::ZERO; nblocks];
    let mut dmin = vec![f16::ZERO; nblocks];

    for b in 0..nblocks {
        let base = b * Q4K_BLOCK_BYTES;
        let blk = &data[base..base + Q4K_BLOCK_BYTES];
        d[b] = f16::from_le_bytes([blk[0], blk[1]]);
        dmin[b] = f16::from_le_bytes([blk[2], blk[3]]);
        // The twelve packed bytes give the eight scales then the eight mins;
        // this layout keeps the two roles in separate arrays, so split there.
        let sm = sub_scales_and_mins_bytes(&blk[4..16]);
        let (sc, mn) = sm.split_at(SUBBLOCKS);
        scales[b * SUBBLOCKS..(b + 1) * SUBBLOCKS].copy_from_slice(sc);
        mins[b * SUBBLOCKS..(b + 1) * SUBBLOCKS].copy_from_slice(mn);
        // qs: 128 bytes per block, contiguous.
        qs[b * (QK_K / 2)..(b + 1) * (QK_K / 2)].copy_from_slice(&blk[16..144]);
    }
    RepackedQ4K {
        n,
        k,
        qs,
        scales,
        mins,
        d,
        dmin,
    }
}

/// Dequantize from the repacked form (reference for the kernel + correctness
/// test). Nibble layout matches `BlockQ4K::dequantize`: per 32 qs-bytes,
/// low nibbles fill one 32-wide sub-block, then high nibbles the next.
/// `w = d * scale[sb] * nibble - dmin * min[sb]`.
pub fn dequant_repacked(r: &RepackedQ4K) -> Vec<f32> {
    let blocks_per_row = r.k / QK_K;
    let mut out = vec![0f32; r.n * r.k];
    for b in 0..(r.n * blocks_per_row) {
        let row = b / blocks_per_row;
        let bcol = b % blocks_per_row;
        let d = r.d[b].to_f32();
        let dmin = r.dmin[b].to_f32();
        let qs = &r.qs[b * (QK_K / 2)..(b + 1) * (QK_K / 2)];
        let out_base = row * r.k + bcol * QK_K;
        // 4 groups of 32 qs-bytes -> 8 sub-blocks (low then high nibble).
        for g in 0..(QK_K / 64) {
            let qg = &qs[g * 32..g * 32 + 32];
            for (sb_local, hi) in [(0usize, false), (1usize, true)] {
                let sb = 2 * g + sb_local;
                let sc = r.scales[b * SUBBLOCKS + sb] as f32;
                let mn = r.mins[b * SUBBLOCKS + sb] as f32;
                let d1 = d * sc;
                let m1 = dmin * mn;
                let dst = out_base + sb * 32;
                for l in 0..32 {
                    let nib = if hi {
                        (qg[l] >> 4) as f32
                    } else {
                        (qg[l] & 0x0F) as f32
                    };
                    out[dst + l] = d1 * nib - m1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QTensor};
    use crate::tensor::{Device, Tensor};

    /// The repack must reproduce the reference own Q4_K dequantization (lossless
    /// reorder + pre-unpacked scales). Validates the 6-bit unpack, nibble
    /// layout, and dequant formula against the reference oracle.
    #[test]
    fn repack_matches_reference_dequant() {
        let dev = Device::Cpu;
        let (n, k) = (16usize, 512usize); // 2 super-blocks per row
                                          // deterministic pseudo-random weights
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let w: Vec<f32> = (0..n * k).map(|_| next() * 0.7).collect();
        let t = Tensor::from_vec(w, (n, k), &dev).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
        let reference = qt
            .dequantize(&dev)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let data = qt.data().unwrap();
        let r = repack_q4k(&data, n, k);
        let mine = dequant_repacked(&r);

        assert_eq!(mine.len(), reference.len());
        let mut max_abs = 0f32;
        for (a, b) in mine.iter().zip(reference.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(max_abs < 1e-3, "repack dequant mismatch, max_abs={max_abs}");
    }
}
