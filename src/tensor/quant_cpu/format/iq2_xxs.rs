//! IQ2_XXS: 2.0625 bits per weight in 256-element blocks. Each 32-element sub-block holds four
//! codebook entries of eight values (an index into a 256-entry grid of E8-lattice points), a 7-bit
//! sign pattern per entry, and a 4-bit scale; the block carries one f16 scale. Read-only: loken
//! reads these files and does not produce them.
//!
//! The grid, sign and mask tables are the format's published codebook - the bytes any correct
//! reader holds.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq2Xxs {
    pub(crate) d: f16,
    pub(crate) qs: [u16; QK_K / 8],
}
const _: () = assert!(std::mem::size_of::<BlockIq2Xxs>() == 66);

impl BlockFormat for BlockIq2Xxs {
    const DTYPE: GgmlDType = GgmlDType::Iq2Xxs;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; QK_K / 8],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        debug_assert!(ys.len().is_multiple_of(QK_K));
        for (x, out) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            let d = x.d.to_f32();
            for (ib32, sub) in out.chunks_exact_mut(32).enumerate() {
                // Four u16 per sub-block, read as two little-endian u32: the low word holds the
                // four grid indices, one byte each; the high word packs four 7-bit sign selectors
                // and, in its top nibble, the sub-block scale.
                let q = &x.qs[4 * ib32..4 * ib32 + 4];
                let lo = q[0] as u32 | ((q[1] as u32) << 16);
                let hi = q[2] as u32 | ((q[3] as u32) << 16);
                let db = d * (0.5 + (hi >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let grid = IQ2XXS_GRID[((lo >> (8 * l)) & 0xff) as usize].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((hi >> (7 * l)) & 127) as usize];
                    for j in 0..8 {
                        let sign = if signs & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        sub[8 * l + j] = db * grid[j] as f32 * sign;
                    }
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
        super::iq2_xxs_dot::dot(xs, ys)
    }

    /// Dequantise the weight block and dot it with the q8_K activation (`d * qs`): the reference
    /// the integer path is checked against.
    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        let mut w = [0f32; QK_K];
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                Self::dequantize(std::slice::from_ref(x), &mut w);
                let s: f32 = w.iter().zip(y.qs.iter()).map(|(&a, &q)| a * q as f32).sum();
                y.d * s
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the encoder leaves, against what the prototype this format's recipe was gated on
    /// leaves on the same weights: the same search, so the same error to within rounding. The
    /// fixture carries the weights, the importance and the prototype's own reconstruction.
    #[test]
    fn encodes_as_well_as_the_recipe_it_was_gated_on() {
        let path = format!(
            "{}/tests/vectors/iq2xxs_encode.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let floats = |key: &str| -> Vec<f32> {
            v[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let (rows, cols) = (
            v["rows"].as_u64().unwrap() as usize,
            v["cols"].as_u64().unwrap() as usize,
        );
        let (x, importance) = (floats("x"), floats("importance"));
        assert_eq!(x.len(), rows * cols);

        let weighted = |got: &[f32], w: &dyn Fn(usize) -> f32| {
            let (mut num, mut den) = (0f64, 0f64);
            for (i, (g, t)) in got.iter().zip(&x).enumerate() {
                let wi = w(i % cols) as f64;
                num += wi * (*g as f64 - *t as f64).powi(2);
                den += wi * (*t as f64).powi(2);
            }
            (num / den).sqrt() as f32
        };
        let mut blocks = vec![BlockIq2Xxs::zeros(); x.len() / QK_K];
        let mut got = vec![0f32; x.len()];

        BlockIq2Xxs::quantize_guided(&x, &mut blocks, &importance, cols);
        BlockIq2Xxs::dequantize(&blocks, &mut got);
        let ours = weighted(&got, &|i| importance[i]);
        let theirs = v["weighted_error"].as_f64().unwrap() as f32;
        assert!(
            ours <= theirs * 1.02,
            "importance-guided: {ours} against the recipe's {theirs}"
        );

        BlockIq2Xxs::quantize(&x, &mut blocks);
        BlockIq2Xxs::dequantize(&blocks, &mut got);
        let ours = weighted(&got, &|_| 1.0);
        let theirs = v["unweighted_error"].as_f64().unwrap() as f32;
        assert!(
            ours <= theirs * 1.02,
            "plain: {ours} against the recipe's {theirs}"
        );

        // Every sub-block scale is four bits, and every sign pattern has even parity, or the file
        // would dequantise to something else on another reader.
        for b in &blocks {
            for sub in 0..QK_K / 32 {
                let hi = b.qs[4 * sub + 2] as u32 | ((b.qs[4 * sub + 3] as u32) << 16);
                assert!(hi >> 28 <= 15);
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((hi >> (7 * l)) & 127) as usize];
                    assert_eq!(signs.count_ones() % 2, 0);
                }
            }
        }
    }
}

/// Quantise whole blocks, optionally against an importance matrix.
///
/// A sub-block of thirty-two values is four groups of eight, and every group takes one point of the
/// grid with one sign pattern. The scale those four groups share is searched over a range around
/// the sub-block's largest magnitude, each candidate scored by the error the best grid point would
/// leave; the winning scales then go through the block's f16 scale and four bits apiece, and the
/// grid points are picked again against what the format can actually represent.
///
/// The sign of one value per group is free: the format stores seven signs and fixes the eighth by
/// parity, so a group with an odd count of negatives has to flip one, and it flips the value whose
/// weighted magnitude costs least.
fn encode(xs: &[f32], ys: &mut [BlockIq2Xxs], imatrix: Option<&[f32]>, n_per_row: usize) {
    use rayon::prelude::*;
    let grid = grid_values();
    // Blocks are independent: each takes its own values and, with an importance, the weights of
    // the columns it covers.
    ys.par_iter_mut()
        .zip(xs.par_chunks_exact(QK_K))
        .enumerate()
        .for_each(|(idx, (block, x))| {
            let row = imatrix.map(|m| {
                let base = (idx % (n_per_row / QK_K)) * QK_K;
                &m[base..base + QK_K]
            });
            encode_block(&grid, x, row, block);
        });
}

/// The grid's points as floats, in index order.
fn grid_values() -> Vec<[f32; 8]> {
    IQ2XXS_GRID
        .iter()
        .map(|g| {
            let b = g.to_le_bytes();
            std::array::from_fn(|i| b[i] as f32)
        })
        .collect()
}

/// The least error any grid point leaves a group at `scale`. Only the value is needed while the
/// scale is searched, so the loop over the points is a plain minimum, which vectorises.
#[inline]
fn least_error(vv: f32, dot: &[f32], gg: &[f32], scale: f32) -> f32 {
    let (a, b) = (2.0 * scale, scale * scale);
    let mut best = f32::INFINITY;
    for (d, q) in dot.iter().zip(gg) {
        let e = b * q - a * d;
        best = if e < best { e } else { best };
    }
    vv + best
}

/// The grid point leaving a group the least error at `scale`.
#[inline]
fn best_point(dot: &[f32], gg: &[f32], scale: f32) -> usize {
    let (a, b) = (2.0 * scale, scale * scale);
    let (mut at, mut best) = (0usize, f32::INFINITY);
    for (n, (d, q)) in dot.iter().zip(gg).enumerate() {
        let e = b * q - a * d;
        if e < best {
            (at, best) = (n, e);
        }
    }
    at
}

fn encode_block(grid: &[[f32; 8]], x: &[f32], row: Option<&[f32]>, block: &mut BlockIq2Xxs) {
    /// Candidate scales, as a fraction of what the largest magnitude would need.
    const STEPS: usize = 15;
    const LOW: f32 = 0.45;
    const HIGH: f32 = 1.25;
    /// The largest value the grid holds, which the scale is measured against.
    const GRID_MAX: f32 = 43.0;
    /// Sub-block scales are four bits under the block's f16 scale, a quarter of it per step.
    const LEVELS: f32 = 15.0;
    const GROUPS: usize = QK_K / 8;

    // Per group: the magnitudes, their signs after the parity fix, and the weights.
    let mut v = [[0f32; 8]; GROUPS];
    let mut neg = [[false; 8]; GROUPS];
    let mut w = [[0f32; 8]; GROUPS];
    for g in 0..GROUPS {
        let (mut worst, mut worst_at) = (f32::INFINITY, 0usize);
        for k in 0..8 {
            let value = x[8 * g + k];
            v[g][k] = value.abs();
            neg[g][k] = value < 0.0;
            w[g][k] = row.map_or(1.0, |r| r[8 * g + k]);
            let cost = w[g][k] * v[g][k];
            if cost < worst {
                (worst, worst_at) = (cost, k);
            }
        }
        if neg[g].iter().filter(|&&n| n).count() % 2 == 1 {
            neg[g][worst_at] = !neg[g][worst_at];
        }
    }

    // What each grid point would cost a group, at a scale of one: the two terms of
    // |scale * point - value|^2 that depend on the point.
    let mut dot = vec![0f32; GROUPS * 256];
    let mut gg = vec![0f32; GROUPS * 256];
    let mut vv = [0f32; GROUPS];
    for g in 0..GROUPS {
        let (wv, ww): ([f32; 8], [f32; 8]) = (
            std::array::from_fn(|k| w[g][k] * v[g][k]),
            std::array::from_fn(|k| w[g][k]),
        );
        vv[g] = (0..8).map(|k| wv[k] * v[g][k]).sum();
        for (n, point) in grid.iter().enumerate() {
            let (mut d, mut q) = (0f32, 0f32);
            for k in 0..8 {
                d += wv[k] * point[k];
                q += ww[k] * point[k] * point[k];
            }
            dot[256 * g + n] = d;
            gg[256 * g + n] = q;
        }
    }
    let at = |g: usize| (&dot[256 * g..256 * (g + 1)], &gg[256 * g..256 * (g + 1)]);

    // One scale per sub-block, searched over the range its own largest magnitude sets.
    let mut scales = [0f32; QK_K / 32];
    for (sub, scale) in scales.iter_mut().enumerate() {
        let amax = (0..4).flat_map(|g| v[4 * sub + g]).fold(0f32, f32::max) + 1e-30;
        let mut best = f32::INFINITY;
        for step in 0..STEPS {
            let f = LOW + (HIGH - LOW) * step as f32 / (STEPS - 1) as f32;
            let candidate = amax * f / GRID_MAX;
            let err: f32 = (0..4)
                .map(|g| {
                    let (d, q) = at(4 * sub + g);
                    least_error(vv[4 * sub + g], d, q, candidate)
                })
                .sum();
            if err < best {
                (best, *scale) = (err, candidate);
            }
        }
    }

    let d = f16::from_f32(scales.iter().fold(0f32, |m, &s| s.max(m)) / ((0.5 + LEVELS) * 0.25));
    block.d = d;
    block.qs.fill(0);
    let d = d.to_f32();
    for (sub, &scale) in scales.iter().enumerate() {
        let level = if d > 0.0 {
            nearest_int(scale / (d * 0.25) - 0.5).clamp(0, LEVELS as i32) as u32
        } else {
            0
        };
        let quantised = d * (0.5 + level as f32) * 0.25;
        let (mut lo, mut hi) = (0u32, level << 28);
        for l in 0..4 {
            let g = 4 * sub + l;
            let (dg, qg) = at(g);
            lo |= (best_point(dg, qg, quantised) as u32) << (8 * l);
            let mut signs = 0u8;
            for k in 0..8 {
                if neg[g][k] {
                    signs |= KMASK_IQ2XS[k];
                }
            }
            hi |= ((signs & 127) as u32) << (7 * l);
        }
        block.qs[4 * sub] = lo as u16;
        block.qs[4 * sub + 1] = (lo >> 16) as u16;
        block.qs[4 * sub + 2] = hi as u16;
        block.qs[4 * sub + 3] = (hi >> 16) as u16;
    }
}

pub(crate) const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

pub(crate) const KSIGNS_IQ2XS: [u8; 128] = [
    0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, 0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f,
    0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, 0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f,
    0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, 0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf,
    0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, 0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f,
    0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, 0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf,
    0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, 0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f,
    0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, 0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f,
    0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, 0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff,
];

pub(crate) const KMASK_IQ2XS: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];
