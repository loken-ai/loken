//! IQ1_S: 256 values in 8 runs of 32, each run 4 groups of 8.
//! A group names one of the 2048 points of `IQ1S_GRID`, whose values are -1, 0 or 1.
//! A value reads `d * (2 k + 1) * (g + delta)`.
//! `d` is the block's f16 scale and `k` the run's 3-bit scale.
//! `delta` is plus or minus `IQ1S_DELTA`, its sign shared by the run.
//!
//! `qs` holds the low 8 bits of each group's index.
//! `qh` holds per run: the high 3 bits of group `l`'s index at bit `3 l`, `k` at bit 12 and the
//! sign of `delta` at bit 15.

use super::iq1_grid::IQ1S_GRID;
use super::*;

/// The shift every value of a group carries, as a fraction of the run's scale.
pub(super) const IQ1S_DELTA: f32 = 0.125;
/// Values a group of the codebook holds.
pub(super) const IQ1_GROUP: usize = 8;
/// The largest odd multiplier a 3-bit run scale reaches, `2 * 7 + 1`.
pub(super) const IQ1_SCALE_TOP: f32 = 15.0;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq1S {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 8],
    pub(crate) qh: [u16; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIq1S>() == 50);

/// The 8 values of grid point `index`, each -1, 0 or 1.
#[inline]
pub(super) fn grid_point(index: usize) -> [i8; IQ1_GROUP] {
    IQ1S_GRID[index].to_le_bytes().map(|b| b as i8)
}

impl BlockFormat for BlockIq1S {
    const DTYPE: GgmlDType = GgmlDType::Iq1S;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0; QK_K / 8],
            qh: [0; QK_K / 32],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            for (ib, run) in y.chunks_exact_mut(32).enumerate() {
                let qh = block.qh[ib];
                let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                let delta = if qh & 0x8000 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                };
                for (l, group) in run.chunks_exact_mut(IQ1_GROUP).enumerate() {
                    let index =
                        block.qs[4 * ib + l] as usize | (((qh >> (3 * l)) & 7) as usize) << 8;
                    for (out, g) in group.iter_mut().zip(grid_point(index)) {
                        *out = dl * (g as f32 + delta);
                    }
                }
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in blocks_with_input(xs, ys) {
            encode_block(x, block);
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        Self::dot_scalar(xs, ys)
    }

    /// The grid sums and the shift sums of a block stay integers under each run's odd
    /// multiplier.
    /// The block's two scales and the shift come in once at the end of the block.
    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        let mut total = 0f32;
        for (x, y) in xs.iter().zip(ys) {
            let (mut sumi, mut sumi1) = (0i32, 0i32);
            for ib in 0..QK_K / 32 {
                let qh = x.qh[ib];
                let ls = 2 * ((qh >> 12) & 7) as i32 + 1;
                let delta = if qh & 0x8000 != 0 { -1 } else { 1 };
                let mut lsum = 0i32;
                for l in 0..4 {
                    let index = x.qs[4 * ib + l] as usize | (((qh >> (3 * l)) & 7) as usize) << 8;
                    let acts = &y.qs[32 * ib + IQ1_GROUP * l..];
                    for (g, &a) in grid_point(index).iter().zip(acts) {
                        lsum += *g as i32 * a as i32;
                    }
                }
                sumi += ls * lsum;
                sumi1 += ls * delta * (y.bsums[2 * ib] as i32 + y.bsums[2 * ib + 1] as i32);
            }
            total += x.d.to_f32() * y.d * (sumi as f32 + IQ1S_DELTA * sumi1 as f32);
        }
        total
    }
}

/// The 2048 grid points as floats, in index order.
fn grid_values() -> &'static [[f32; IQ1_GROUP]] {
    static GRID: std::sync::OnceLock<Vec<[f32; IQ1_GROUP]>> = std::sync::OnceLock::new();
    GRID.get_or_init(|| {
        (0..IQ1S_GRID.len())
            .map(|i| grid_point(i).map(|g| g as f32))
            .collect()
    })
}

/// The grid point and shift sign that leave a group of 8 values the least squared error at
/// `scale`, with that error.
/// `signs` lists the shift signs the group may take, each `IQ1S_DELTA` or `-IQ1S_DELTA`.
pub(super) fn best_point(x: &[f32], scale: f32, signs: &[f32]) -> (usize, f32, f32) {
    let mut best = (0usize, signs[0], f32::INFINITY);
    for &delta in signs {
        for (n, g) in grid_values().iter().enumerate() {
            let err: f32 = (0..IQ1_GROUP)
                .map(|i| {
                    let e = x[i] - scale * (g[i] + delta);
                    e * e
                })
                .sum();
            if err < best.2 {
                best = (n, delta, err);
            }
        }
    }
    best
}

/// The scale that fits `x` best to the groups it was given, each a grid point and a shift:
/// `sum(x v) / sum(v v)` over the values `v = g + delta` the groups read.
pub(super) fn fitted_scale(x: &[f32], picks: &[(usize, f32)]) -> f32 {
    let (mut xv, mut vv) = (0f32, 0f32);
    for (group, &(n, delta)) in x.chunks_exact(IQ1_GROUP).zip(picks) {
        for (xi, g) in group.iter().zip(grid_values()[n]) {
            let v = g + delta;
            xv += xi * v;
            vv += v * v;
        }
    }
    if vv > 0.0 {
        (xv / vv).max(0.0)
    } else {
        0.0
    }
}

/// The scale of a span of whole groups, and each group's point and shift.
/// Points and scale are fitted in turn from the scale that maps the largest magnitude onto
/// `1 + IQ1S_DELTA`, until the error stops falling.
/// `sign_sets` lists, for each choice of shift signs the span allows, the signs each group may
/// take.
pub(super) fn fit_span(x: &[f32], sign_sets: &[&[f32]]) -> (f32, Vec<(usize, f32)>) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let groups = x.len() / IQ1_GROUP;
    let mut best = (0f32, vec![(0usize, IQ1S_DELTA); groups], f32::INFINITY);
    if amax == 0.0 {
        return (best.0, best.1);
    }
    for signs in sign_sets {
        let mut scale = amax / (1.0 + IQ1S_DELTA);
        let mut err = f32::INFINITY;
        loop {
            let mut picks = Vec::with_capacity(groups);
            let mut total = 0f32;
            for group in x.chunks_exact(IQ1_GROUP) {
                let (n, delta, e) = best_point(group, scale, signs);
                picks.push((n, delta));
                total += e;
            }
            if total >= err {
                break;
            }
            err = total;
            if err < best.2 {
                best = (scale, picks.clone(), err);
            }
            scale = fitted_scale(x, &picks);
        }
    }
    (best.0, best.1)
}

/// The odd multiplier `2 k + 1` of the block scale `d` nearest `scale`, as `k`.
pub(super) fn run_scale_code(scale: f32, d: f32) -> u16 {
    if d <= 0.0 {
        return 0;
    }
    ((scale / d - 1.0) / 2.0).round().clamp(0.0, 7.0) as u16
}

/// Encode one block: each run's shift sign and scale are fitted with its points, the block
/// scale takes the largest run scale onto the top multiplier, and each run picks its points
/// again at its scale as stored.
fn encode_block(x: &[f32], block: &mut BlockIq1S) {
    const POS: &[f32] = &[IQ1S_DELTA];
    const NEG: &[f32] = &[-IQ1S_DELTA];
    let mut scales = [0f32; QK_K / 32];
    let mut signs = [IQ1S_DELTA; QK_K / 32];
    for (ib, run) in x.chunks_exact(32).enumerate() {
        let (scale, picks) = fit_span(run, &[POS, NEG]);
        scales[ib] = scale;
        signs[ib] = picks[0].1;
    }
    let top = scales.iter().fold(0f32, |m, &s| m.max(s));
    block.d = f16::from_f32(top / IQ1_SCALE_TOP);
    let d = block.d.to_f32();
    for (ib, run) in x.chunks_exact(32).enumerate() {
        let k = run_scale_code(scales[ib], d);
        let dl = d * (2 * k + 1) as f32;
        let sign = [signs[ib]];
        let mut qh = (k << 12) | if signs[ib] < 0.0 { 0x8000 } else { 0 };
        for (l, group) in run.chunks_exact(IQ1_GROUP).enumerate() {
            let (n, _, _) = best_point(group, dl, &sign);
            block.qs[4 * ib + l] = (n & 0xFF) as u8;
            qh |= ((n >> 8) as u16 & 7) << (3 * l);
        }
        block.qh[ib] = qh;
    }
}
