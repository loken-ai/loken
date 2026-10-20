//! The importance-weighted fits of the thirty-two value formats: q4_0 and q5_0, one signed scale a
//! block, and q4_1 and q5_1, a scale and an offset.
//!
//! As in ggml's `quantize_row_q4_0_impl` and its kin (MIT): a value weighs its importance times
//! its distance from the mean square of its row, and the block's scale is the least-squares fit
//! of its levels under those weights. The values are then levelled against the scale as stored.

use super::*;

/// The weight of each value of `row` against `importance`, one importance per column.
fn row_weights(row: &[f32], importance: &[f32], out: &mut [f32]) {
    let mut sum = 0f32;
    for v in row {
        sum += v * v;
    }
    let sigma2 = sum / row.len() as f32;
    for ((o, &v), &q) in out.iter_mut().zip(row).zip(importance) {
        *o = q * (sigma2 + v * v).sqrt();
    }
}

/// The steps of a wider sweep than `fit_signed_scale` takes: a tenth of a level apart, from half
/// a level to nine tenths either side of the extreme's own.
const WIDE: [i32; 10] = [-9, -8, -7, -6, -5, 5, 6, 7, 8, 9];

/// `fit_signed_scale` then the scales a wider sweep reaches, kept where they fit better: levels
/// in `[-nmax, nmax - 1]` written biased by `nmax` into `ls`, the scale returned.
pub(super) fn fit_signed_wide(nmax: i32, x: &[f32], ls: &mut [i8], w: &[f32]) -> f32 {
    let mut scale = fit_signed_scale(nmax, x, ls, w);
    let extreme = x
        .iter()
        .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
    if extreme == 0.0 {
        return scale;
    }
    let sums = |level: &dyn Fn(usize) -> i32| {
        let (mut wxl, mut wl2) = (0f32, 0f32);
        for i in 0..x.len() {
            let l = level(i) as f32;
            wxl += w[i] * x[i] * l;
            wl2 += w[i] * l * l;
        }
        (wxl, wl2)
    };
    let (wxl, wl2) = sums(&|i| ls[i] as i32 - nmax);
    let mut best = if wl2 > 0.0 { wxl * wxl / wl2 } else { 0.0 };
    for step in WIDE {
        let inverse = -(nmax as f32 + 0.1 * step as f32) / extreme;
        let level = |i: usize| nearest_int(inverse * x[i]).clamp(-nmax, nmax - 1);
        let (wxl, wl2) = sums(&level);
        if wl2 > 0.0 && wxl * wxl > best * wl2 {
            for (i, l) in ls.iter_mut().enumerate() {
                *l = (level(i) + nmax) as i8;
            }
            scale = wxl / wl2;
            best = scale * wxl;
        }
    }
    scale
}

/// Each block of thirty-two values of `xs` (rows `n_per_row` long) as a signed scale and codes
/// `0..2 nmax`, code `c` meaning `c - nmax` steps of the scale.
pub(super) fn signed_blocks(
    xs: &[f32],
    importance: &[f32],
    n_per_row: usize,
    nmax: i32,
) -> Vec<(f16, [u8; 32])> {
    let mut w = vec![0f32; n_per_row];
    let mut out = Vec::with_capacity(xs.len() / 32);
    for row in xs.chunks_exact(n_per_row) {
        row_weights(row, importance, &mut w);
        for (xb, wb) in row.chunks_exact(32).zip(w.chunks_exact(32)) {
            let mut ls = [0i8; 32];
            let d = f16::from_f32(fit_signed_wide(nmax, xb, &mut ls, wb));
            let step = d.to_f32();
            let inverse = if step != 0.0 { 1.0 / step } else { 0.0 };
            let codes = std::array::from_fn(|i| {
                (nearest_int(inverse * xb[i]).clamp(-nmax, nmax - 1) + nmax) as u8
            });
            out.push((d, codes));
        }
    }
    out
}

/// Each block of thirty-two values of `xs` (rows `n_per_row` long) as a scale, an offset and codes
/// `0..=nmax`: the value is `scale * code + offset`.
pub(super) fn affine_blocks(
    xs: &[f32],
    importance: &[f32],
    n_per_row: usize,
    nmax: i32,
) -> Vec<(f16, f16, [u8; 32])> {
    let mut w = vec![0f32; n_per_row];
    let mut out = Vec::with_capacity(xs.len() / 32);
    for row in xs.chunks_exact(n_per_row) {
        row_weights(row, importance, &mut w);
        for (xb, wb) in row.chunks_exact(32).zip(w.chunks_exact(32)) {
            let [(scale, below)] = fit_scales_and_offsets::<1>(nmax, xb, wb);
            let (d, m) = (f16::from_f32(scale), f16::from_f32(-below));
            let (step, offset) = (d.to_f32(), m.to_f32());
            let inverse = if step != 0.0 { 1.0 / step } else { 0.0 };
            let codes = std::array::from_fn(|i| {
                nearest_int(inverse * (xb[i] - offset)).clamp(0, nmax) as u8
            });
            out.push((d, m, codes));
        }
    }
    out
}
