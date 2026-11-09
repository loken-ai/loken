//! Rows of latent attention state written as the grid codes they hold, and read back to the
//! same f32 bits.
//!
//! Every value of these rows is a point of a small grid times a block scale:
//! - a window key row is e4m3 with a power-of-two scale per block of [`FP8_BLOCK`],
//! - a compressed row is e2m1 with an f32 scale per block of [`COMPRESSED_BLOCK`],
//! - an index key is e2m1 with a power-of-two scale per block of [`INDEX_BLOCK`].
//!
//! Each `pack_*` unpacks what it packed and compares every bit.
//! It gives `None` for values that are not on the grid it was asked for, so a caller falls back
//! to raw f32 and the round trip stays exact whatever the values are.

use super::compressed_band::{COMPRESSED_BLOCK, FP4_GRID, FP4_MAX, INDEX_BLOCK};
use super::latent_mqa::{pow2_ceil, FP8_BLOCK, FP8_E4M3_MAX};

/// Mantissa bits of e4m3.
const E4M3_MANTISSA_BITS: i32 = 3;

/// Exponent bias of e4m3.
const E4M3_BIAS: i32 = 7;

/// Finite non-negative e4m3 codes: every 7-bit code but the last, which is NaN.
const E4M3_CODES: usize = 127;

/// The sign of an e4m3 code.
const FP8_SIGN: u8 = 0x80;

/// The sign of an e2m1 nibble.
const FP4_SIGN: u8 = 8;

/// The grid index of an e2m1 nibble.
const FP4_INDEX: u8 = 7;

/// Neighbours, in units in the last place, of a candidate f32 scale that a compressed block is
/// tried at, nearest first.
const SCALE_ULPS: [i32; 5] = [0, 1, -1, 2, -2];

/// The finite non-negative e4m3 values by code.
fn e4m3_table() -> &'static [f32; E4M3_CODES] {
    static T: std::sync::OnceLock<[f32; E4M3_CODES]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; E4M3_CODES];
        let steps = (1 << E4M3_MANTISSA_BITS) as f32;
        for (code, v) in t.iter_mut().enumerate() {
            let e = (code >> E4M3_MANTISSA_BITS) as i32;
            let m = (code & ((1 << E4M3_MANTISSA_BITS) - 1)) as f32;
            *v = if e == 0 {
                m * 2f32.powi(1 - E4M3_BIAS - E4M3_MANTISSA_BITS)
            } else {
                (1.0 + m / steps) * 2f32.powi(e - E4M3_BIAS)
            };
        }
        t
    })
}

/// The exponent of a power-of-two `scale` that an i8 holds.
fn exponent_of(scale: f32) -> Option<i8> {
    let e = scale.log2();
    (e.fract() == 0.0 && (f32::from(i8::MIN)..=f32::from(i8::MAX)).contains(&e)).then_some(e as i8)
}

/// Bytes [`pack_fp8`] writes for `n` values: per block an exponent byte and a code per value.
pub(super) fn fp8_len(n: usize) -> usize {
    n / FP8_BLOCK * (FP8_BLOCK + 1)
}

/// Bytes [`pack_fp4`] writes for `n` values in blocks of `block`: per block a scale and a
/// nibble per value.
///
/// The scale is an exponent byte with `pow2`, the f32 bits otherwise.
pub(super) fn fp4_len(n: usize, block: usize, pow2: bool) -> usize {
    let scale = if pow2 { 1 } else { std::mem::size_of::<f32>() };
    n / block * (block / 2 + scale)
}

/// e4m3 values with a power-of-two scale per block of [`FP8_BLOCK`].
pub(super) fn pack_fp8(values: &[f32]) -> Option<Vec<u8>> {
    if !values.len().is_multiple_of(FP8_BLOCK) {
        return None;
    }
    let table = e4m3_table();
    let mut out = Vec::with_capacity(fp8_len(values.len()));
    for block in values.chunks_exact(FP8_BLOCK) {
        let amax = block.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if !amax.is_finite() {
            return None;
        }
        let scale = if amax == 0.0 {
            1.0
        } else {
            pow2_ceil(amax / FP8_E4M3_MAX)
        };
        out.push(exponent_of(scale)? as u8);
        for &v in block {
            let mag = v.abs() / scale;
            let code = table.partition_point(|&t| t < mag);
            if code >= table.len() || table[code] != mag {
                return None;
            }
            out.push(code as u8 | if v.is_sign_negative() { FP8_SIGN } else { 0 });
        }
    }
    let back = unpack_fp8(&out, values.len());
    same_bits(&back, values).then_some(out)
}

/// The values [`pack_fp8`] wrote. `bytes` is `fp8_len(n)` long.
pub(super) fn unpack_fp8(bytes: &[u8], n: usize) -> Vec<f32> {
    let table = e4m3_table();
    let mut out = Vec::with_capacity(n);
    for block in bytes.chunks_exact(FP8_BLOCK + 1) {
        let scale = f32::from(block[0] as i8).exp2();
        for &c in &block[1..] {
            let m = table[usize::from(c & !FP8_SIGN).min(E4M3_CODES - 1)] * scale;
            out.push(if c & FP8_SIGN != 0 { -m } else { m });
        }
    }
    out
}

/// e2m1 values in blocks of `block`, an even count: a scale then 1 nibble per value.
///
/// The scale is an exponent byte with `pow2`, the f32 bits otherwise.
pub(super) fn pack_fp4(values: &[f32], block: usize, pow2: bool) -> Option<Vec<u8>> {
    if block == 0 || !block.is_multiple_of(2) || !values.len().is_multiple_of(block) {
        return None;
    }
    let mut out = Vec::with_capacity(fp4_len(values.len(), block, pow2));
    for blk in values.chunks_exact(block) {
        let mut best_mag = 0f32;
        for &v in blk {
            if !v.is_finite() {
                return None;
            }
            best_mag = best_mag.max(v.abs());
        }
        let scale = if best_mag == 0.0 {
            Some(if pow2 { 1.0 } else { 0.0 })
        } else if pow2 {
            // The scale that puts the largest value on the grid's largest, or 1 binade off
            // where the largest rounded down.
            let s = pow2_ceil(best_mag / FP4_MAX);
            [s, s * 2.0, s * 0.5]
                .into_iter()
                .find(|&c| block_fits(blk, c))
        } else {
            // The largest value is some grid point times the scale: each is tried as that point.
            FP4_GRID[1..].iter().rev().find_map(|&g| {
                let s = best_mag / g;
                SCALE_ULPS
                    .into_iter()
                    .map(|d| f32::from_bits((s.to_bits() as i64 + d as i64) as u32))
                    .find(|&c| c.is_finite() && block_fits(blk, c))
            })
        };
        let scale = scale?;
        if pow2 {
            out.push(exponent_of(scale)? as u8);
        } else {
            out.extend_from_slice(&scale.to_bits().to_le_bytes());
        }
        let nib = |v: f32| -> u8 {
            let idx = FP4_GRID
                .iter()
                .position(|&g| (g.copysign(v) * scale).to_bits() == v.to_bits())
                .unwrap_or(0) as u8;
            idx | if v.is_sign_negative() { FP4_SIGN } else { 0 }
        };
        for pair in blk.chunks_exact(2) {
            out.push(nib(pair[0]) | nib(pair[1]) << 4);
        }
    }
    let back = unpack_fp4(&out, values.len(), block, pow2);
    same_bits(&back, values).then_some(out)
}

/// Whether every value of `blk` is a grid point times `scale`, the sign of a zero included.
fn block_fits(blk: &[f32], scale: f32) -> bool {
    blk.iter().all(|&v| {
        FP4_GRID
            .iter()
            .any(|&g| (g.copysign(v) * scale).to_bits() == v.to_bits())
    })
}

/// The values [`pack_fp4`] wrote.
pub(super) fn unpack_fp4(bytes: &[u8], n: usize, block: usize, pow2: bool) -> Vec<f32> {
    let head = if pow2 { 1 } else { std::mem::size_of::<f32>() };
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(head + block / 2) {
        let scale = if pow2 {
            f32::from(chunk[0] as i8).exp2()
        } else {
            f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        };
        for &b in &chunk[head..] {
            for nib in [b & 0x0f, b >> 4] {
                let m = FP4_GRID[usize::from(nib & FP4_INDEX)] * scale;
                out.push(if nib & FP4_SIGN != 0 { -m } else { m });
            }
        }
    }
    out
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The grid a kind of latent attention row lies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Grid {
    /// A window key row.
    Fp8,
    /// A compressed row.
    Fp4Scaled,
    /// An index key.
    Fp4Pow2,
}

impl Grid {
    /// `values` packed, `None` when they are not on the grid.
    pub(super) fn pack(self, values: &[f32]) -> Option<Vec<u8>> {
        match self {
            Grid::Fp8 => pack_fp8(values),
            Grid::Fp4Scaled => pack_fp4(values, COMPRESSED_BLOCK, false),
            Grid::Fp4Pow2 => pack_fp4(values, INDEX_BLOCK, true),
        }
    }

    /// Packed bytes of `n` values, `None` when `n` is not a whole number of blocks.
    pub(super) fn packed_len(self, n: usize) -> Option<usize> {
        let (block, len) = match self {
            Grid::Fp8 => (FP8_BLOCK, fp8_len(n)),
            Grid::Fp4Scaled => (COMPRESSED_BLOCK, fp4_len(n, COMPRESSED_BLOCK, false)),
            Grid::Fp4Pow2 => (INDEX_BLOCK, fp4_len(n, INDEX_BLOCK, true)),
        };
        n.is_multiple_of(block).then_some(len)
    }

    /// The `n` values of `bytes`, which are `packed_len(n)` long.
    pub(super) fn unpack(self, bytes: &[u8], n: usize) -> Vec<f32> {
        match self {
            Grid::Fp8 => unpack_fp8(bytes, n),
            Grid::Fp4Scaled => unpack_fp4(bytes, n, COMPRESSED_BLOCK, false),
            Grid::Fp4Pow2 => unpack_fp4(bytes, n, INDEX_BLOCK, true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generic_transformer::compressed_band::fp4_round_trip;
    use crate::inference::generic_transformer::latent_mqa::fp8_round_trip;
    use crate::inference::generic_transformer::wide_sparse_frozen as frozen;

    /// Values with a spread of magnitudes, zeros of both signs and a few exact grid points.
    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                match i % 37 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => u * 1e-7,
                    3 => u * 3000.0,
                    _ => u * 4.0 * (1.0 + (i % 7) as f32),
                }
            })
            .collect()
    }

    /// The rows the mixers round, packed, come back to the bit, in the bytes the V4.1 packing
    /// wrote.
    #[test]
    fn the_quantisers_outputs_pack_and_come_back_to_the_bit() {
        for seed in 0..40 {
            let raw = data(64 * 24, seed);
            let mut q8 = raw.clone();
            fp8_round_trip(&mut q8, FP8_BLOCK);
            let p = pack_fp8(&q8).expect("fp8 rows are on the grid");
            assert_eq!(p.len(), fp8_len(q8.len()));
            assert!(same_bits(&unpack_fp8(&p, q8.len()), &q8), "fp8 seed {seed}");
            frozen::expect_bytes("units", &format!("pack fp8 {seed}"), &p);
            for (block, pow2) in [(COMPRESSED_BLOCK, false), (INDEX_BLOCK, true)] {
                let mut q4 = raw.clone();
                fp4_round_trip(&mut q4, block, pow2);
                let p = pack_fp4(&q4, block, pow2)
                    .unwrap_or_else(|| panic!("fp4 block {block} pow2 {pow2} seed {seed}"));
                assert_eq!(p.len(), fp4_len(q4.len(), block, pow2));
                assert!(same_bits(&unpack_fp4(&p, q4.len(), block, pow2), &q4));
                let key = format!("pack fp4 {seed} {block} {pow2}");
                frozen::expect_bytes("units", &key, &p);
            }
        }
    }

    #[test]
    fn values_off_the_grid_are_refused_not_rounded() {
        let raw = data(64 * 4, 3);
        assert!(pack_fp8(&raw).is_none());
        assert!(pack_fp4(&raw, COMPRESSED_BLOCK, false).is_none());
        assert!(pack_fp4(&raw, INDEX_BLOCK, true).is_none());
        assert!(pack_fp8(&raw[..FP8_BLOCK - 1]).is_none());
        assert!(pack_fp4(&[f32::NAN; COMPRESSED_BLOCK], COMPRESSED_BLOCK, false).is_none());
    }

    #[test]
    fn a_packed_row_costs_a_fraction_of_the_floats() {
        assert_eq!(fp8_len(512), 528);
        assert_eq!(fp4_len(512, 16, false), 384);
        assert_eq!(fp4_len(128, 32, true), 68);
        for n in [0, 32, 512] {
            let lens = (fp8_len(n), fp4_len(n, 16, false), fp4_len(n, 32, true));
            frozen::expect_debug("units", &format!("pack lens {n}"), &lens);
        }
    }

    /// A grid's length is refused for a count that is not a whole number of blocks.
    #[test]
    fn a_grid_reads_only_whole_blocks() {
        assert_eq!(Grid::Fp8.packed_len(FP8_BLOCK), Some(fp8_len(FP8_BLOCK)));
        assert_eq!(Grid::Fp8.packed_len(FP8_BLOCK + 1), None);
        assert_eq!(Grid::Fp4Scaled.packed_len(COMPRESSED_BLOCK - 1), None);
        assert_eq!(Grid::Fp4Pow2.packed_len(0), Some(0));
    }
}
