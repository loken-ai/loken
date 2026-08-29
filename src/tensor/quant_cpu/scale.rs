//! Choosing the scale a group of values is quantised against.
//!
//! Every block format reduces to the same question: given `n` values and a fixed set of
//! levels, which scale loses the least? A plain `max / nmax` answers it only if the extreme
//! value is worth as much as the rest, and it usually is not - so each search here starts
//! from that answer and then spends its budget finding out how far off it was.
//!
//! Two kinds of search, and one weighting:
//!
//!   [`fit_signed_scale`]   signed levels, no offset - q3_K's and q6_K's sub-blocks
//!   [`fit_scale_and_offset`] unsigned levels with an offset - q2_K, q4_K, q5_K
//!   [`fit_unsigned_scale`]   the SECOND stage: the scales themselves, quantised in turn
//!   [`magnitude_weights`] what a value counts for when no importance matrix says

use super::*;

pub(super) fn nearest_int(v: f32) -> i32 {
    v.round() as i32
}

/// The scale a fixed set of levels implies, and the objective it reaches.
///
/// Minimising `Σ w(x - d.l)²` over `d` alone gives `d = Σwxl / Σwl²`; substituting that back
/// leaves `(Σwxl)² / Σwl²` to be maximised. Both sums are kept so two candidate level sets can
/// be compared by cross-multiplying rather than dividing.
#[derive(Clone, Copy)]
struct Fit {
    wxl: f32,
    wl2: f32,
}

impl Fit {
    fn scale(self) -> f32 {
        self.wxl / self.wl2
    }

    fn objective(self) -> f32 {
        self.scale() * self.wxl
    }
}

/// Pick the scale `d` for a group quantised to levels in `[-nmax, nmax - 1]`, and write the
/// levels into `ls` biased by `+nmax` so they are non-negative.
///
/// The range is asymmetric - `-nmax` exists and `+nmax` does not - so the search starts by
/// mapping the group's extreme onto the end that exists, then refines. Each refinement is
/// guarded to commit only when the objective rises, so none of them can make the fit worse.
///
/// `weights` is the importance matrix when the caller has one and `magnitude_weights`
/// otherwise.
pub(super) fn fit_signed_scale(nmax: i32, x: &[f32], ls: &mut [i8], weights: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), ls.len());
    debug_assert!(weights.len() >= x.len());

    let extreme = x
        .iter()
        .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
    if extreme == 0.0 {
        ls.fill(0);
        return 0.0;
    }

    let level = |inv: f32, v: f32| nearest_int(inv * v).clamp(-nmax, nmax - 1);
    let score = |inv: f32| {
        x.iter()
            .zip(weights)
            .fold(Fit { wxl: 0.0, wl2: 0.0 }, |f, (&v, &w)| {
                let l = level(inv, v) as f32;
                Fit {
                    wxl: f.wxl + w * v * l,
                    wl2: f.wl2 + w * l * l,
                }
            })
    };
    let commit = |inv: f32, ls: &mut [i8]| {
        for (&v, out) in x.iter().zip(ls.iter_mut()) {
            *out = (level(inv, v) + nmax) as i8;
        }
        score(inv)
    };

    let mut fit = commit(-(nmax as f32) / extreme, ls);
    let mut best = fit.objective();

    // Re-derive every level from the scale the last pass implied, up to three times. It stops
    // as soon as nothing moves, which is the common case after one.
    for _ in 0..3 {
        let inv = 1.0 / fit.scale();
        let unchanged = x
            .iter()
            .zip(ls.iter())
            .all(|(&v, &l)| level(inv, v) + nmax == l as i32);
        let candidate = score(inv);
        if unchanged
            || candidate.wl2 == 0.0
            || candidate.wxl * candidate.wxl <= best * candidate.wl2
        {
            break;
        }
        fit = commit(inv, ls);
        best = fit.objective();
    }

    // Move one level at a time: take a value out of the fit, ask where it would best go back
    // in, keep the move only if the objective rises.
    for _ in 0..5 {
        let mut moved = 0;
        for i in 0..x.len() {
            let (v, w) = (x[i], weights[i]);
            let l = ls[i] as i32 - nmax;
            let mut wxl = fit.wxl - w * v * l as f32;
            if wxl <= 0.0 {
                continue;
            }
            let mut wl2 = fit.wl2 - w * (l * l) as f32;
            let to = nearest_int(v * wl2 / wxl).clamp(-nmax, nmax - 1);
            if to == l {
                continue;
            }
            wxl += w * v * to as f32;
            wl2 += w * (to * to) as f32;
            // Cross-multiplied: no division, and no guard needed on `best`.
            if wl2 > 0.0 && wxl * wxl * fit.wl2 > fit.wxl * fit.wxl * wl2 {
                ls[i] = (to + nmax) as i8;
                fit = Fit { wxl, wl2 };
                best = fit.objective();
                moved += 1;
            }
        }
        if moved == 0 {
            break;
        }
    }

    // Nothing above revisits the assumption that the extreme lands exactly on `-nmax`.
    // Stretching a few percent either way clips that one value in exchange for resolving the
    // rest, which is often the better trade - so try eight of those and keep any that wins.
    for step in (-4..=4).filter(|&s| s != 0) {
        let inv = -(nmax as f32 + 0.1 * step as f32) / extreme;
        let candidate = score(inv);
        if candidate.wl2 > 0.0 && candidate.wxl * candidate.wxl > best * candidate.wl2 {
            fit = commit(inv, ls);
            best = fit.objective();
        }
    }
    fit.scale()
}

/// The grid of scales the search below sweeps, as a multiplier on `nmax`: from `nmax - 0.9`
/// to `nmax + 0.9` in steps of `0.05`.
///
/// A scale wider than the extreme value demands buys resolution on everything else at the
/// price of clipping that one value, and which side of that trade wins is a property of the
/// data, not of the format. Thirty-seven candidates are cheap enough to score them all.
const GRID_MIN: f32 = -0.9;
const GRID_STEP: f32 = 0.05;
const GRID_STEPS: usize = 36;

/// Fit `x` to levels `0..=nmax` under an affine reconstruction `scale.l + min`, and return
/// `(scale, -min)` - the caller stores the negated minimum, which is how the k-quant block
/// formats spell it.
///
/// `weights` is the importance matrix when the caller has one and `magnitude_weights`
/// otherwise. The scoring is weighted squared error throughout.
///
/// NOT rewritten with the rest of this file, deliberately. Two of its properties are easy to
/// miss and both change the result: the minimum a candidate settles on is CARRIED into the
/// next candidate's levels, and the grid is rescaled by the new span each time - so this walks
/// rather than sampling a fixed grid, and the order of the steps is part of the answer. A
/// reimplementation honouring both still moved q2_K's importance-matrix ratio from 0.780 to
/// 0.792, and shipping a degradation that is not understood is worse than keeping the shape
/// that is measured.
pub(super) fn fit_scale_and_offset(nmax: i32, x: &[f32], weights: &[f32]) -> (f32, f32) {
    let n = x.len();
    let mut levels: [u8; 32] = [0; 32];

    let mut offset = x[0];
    let mut top = x[0];
    let mut w_total = weights[0];
    let mut wx_total = w_total * x[0];

    for i in 1..n {
        if x[i] < offset {
            offset = x[i];
        }
        if x[i] > top {
            top = x[i];
        }
        let w = weights[i];
        w_total += w;
        wx_total += w * x[i];
    }

    if offset > 0.0 {
        offset = 0.0;
    }

    if top <= offset {
        return (0.0, -offset);
    }

    let mut per_unit = nmax as f32 / (top - offset);
    let mut scale = 1.0 / per_unit;
    let mut best_error = 0.0;

    for i in 0..n {
        let level = nearest_int(per_unit * (x[i] - offset)).clamp(0, nmax) as u8;
        let diff = scale * (level as f32) + offset - x[i];
        let w = weights[i];
        best_error += w * diff * diff;
    }

    for step in 0..=GRID_STEPS {
        per_unit = (GRID_MIN + GRID_STEP * step as f32 + nmax as f32) / (top - offset);
        let (mut wl, mut wl2, mut wxl) = (0.0, 0.0, 0.0);

        for i in 0..n {
            let level = nearest_int(per_unit * (x[i] - offset)).clamp(0, nmax) as u8;
            levels[i] = level;
            let w = weights[i];
            wl += w * level as f32;
            wl2 += w * (level as f32).powi(2);
            wxl += w * level as f32 * x[i];
        }

        let denom = w_total * wl2 - wl * wl;
        if denom > 0.0 {
            let mut candidate_scale = (w_total * wxl - wx_total * wl) / denom;
            let mut candidate_offset = (wl2 * wx_total - wl * wxl) / denom;

            if candidate_offset > 0.0 {
                candidate_offset = 0.0;
                candidate_scale = wxl / wl2;
            }

            let mut error = 0.0;
            for i in 0..n {
                let diff = candidate_scale * (levels[i] as f32) + candidate_offset - x[i];
                let w = weights[i];
                error += w * diff * diff;
            }

            if error < best_error {
                best_error = error;
                scale = candidate_scale;
                offset = candidate_offset;
            }
        }
    }

    (scale, -offset)
}

/// Fit non-negative `x` to levels `0..=nmax` with no offset, writing the levels into `l` and
/// returning the scale.
///
/// This is the second stage of a k-quant block: the per-sub-block scales and minimums that
/// `fit_scale_and_offset` produced are themselves quantised, against `quant_weights` - the weight
/// each sub-block carries, so a scale that governs many large values is fitted more tightly
/// than one that governs noise.
pub(super) fn fit_unsigned_scale(nmax: u8, x: &[f32], l: &mut [u8], quant_weights: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), l.len());
    debug_assert_eq!(x.len(), quant_weights.len());

    let max = x.iter().copied().fold(0.0, f32::max);
    if max == 0.0 {
        l.fill(0);
        return 0.0;
    }

    let code = |inv: f32, v: f32| (nearest_int(inv * v) as u8).min(nmax);
    let error = |inv: f32| -> f32 {
        let scale = 1.0 / inv;
        x.iter()
            .zip(quant_weights)
            .map(|(&v, &w)| {
                let diff = v - scale * code(inv, v) as f32;
                w * diff * diff
            })
            .sum()
    };

    // Start with the largest value on the widest code, then try eight scales around it - the
    // same trade as elsewhere between clipping one value and resolving the rest.
    let mut inv = nmax as f32 / max;
    let mut best = error(inv);
    for step in (-4..=4).filter(|&s| s != 0) {
        let candidate = (0.1 * step as f32 + nmax as f32) / max;
        let e = error(candidate);
        if e < best {
            best = e;
            inv = candidate;
        }
    }

    let mut fit = Fit { wxl: 0.0, wl2: 0.0 };
    for ((&v, out), &w) in x.iter().zip(l.iter_mut()).zip(quant_weights) {
        let c = code(inv, v) as f32;
        *out = c as u8;
        fit.wxl += w * v * c;
        fit.wl2 += w * c * c;
    }

    // Then the same one-at-a-time refinement as `fit_signed_scale`, over unsigned codes.
    for _ in 0..5 {
        let mut moved = 0;
        for i in 0..x.len() {
            let (v, w) = (x[i], quant_weights[i]);
            let c = l[i] as f32;
            let mut wxl = fit.wxl - w * v * c;
            let mut wl2 = fit.wl2 - w * c * c;
            if wxl <= 0.0 || wl2 <= 0.0 {
                continue;
            }
            let to = (nearest_int(v * wl2 / wxl) as u8).min(nmax);
            if to == l[i] {
                continue;
            }
            wxl += w * v * to as f32;
            wl2 += w * (to as f32) * (to as f32);
            if wxl * wxl * fit.wl2 > fit.wxl * fit.wxl * wl2 {
                l[i] = to;
                fit = Fit { wxl, wl2 };
                moved += 1;
            }
        }
        if moved == 0 {
            break;
        }
    }
    fit.scale()
}

/// How much each value in a group counts toward the fit, when no importance matrix says.
///
/// `rms(x) + |x|`: linear in magnitude rather than quadratic, and shifted by the group's RMS so
/// it never reaches zero. Weighting by `x²` - what a plain least-squares fit does implicitly -
/// lets a group's small values fall out of the fit altogether, and those are precisely the ones
/// the scale was NOT chosen for, so they are the ones that need a say in it.
pub(super) fn magnitude_weights(x: &[f32], out: &mut [f32]) {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    for (o, &v) in out.iter_mut().zip(x) {
        *o = rms + v.abs();
    }
}
