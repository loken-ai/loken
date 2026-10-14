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
//!   [`fit_scales_and_offsets`] unsigned levels with an offset - q2_K, q4_K, q5_K
//!   [`fit_unsigned_scale`]   the SECOND stage: the scales themselves, quantised in turn
//!   [`magnitude_weights`] what a value counts for when no importance matrix says

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

/// The longest sub-block a k-quant fits.
const MAX_SUB: usize = 32;

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
///
/// `G` sub-blocks of `x.len() / G` values are fitted side by side, each to its own `(scale,
/// -min)`: every sub-block runs exactly the arithmetic a fit of it alone runs, in the same order,
/// so the result is the same to the bit; the sub-blocks are the lanes the arithmetic is
/// vectorised over.
pub(super) fn fit_scales_and_offsets<const G: usize>(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
) -> [(f32, f32); G] {
    let n = x.len() / G;
    assert!(n > 0 && n <= MAX_SUB && x.len() == G * n && weights.len() == x.len());
    // Value `i` of every sub-block side by side.
    let mut xt = [[0f32; G]; MAX_SUB];
    let mut wt = [[0f32; G]; MAX_SUB];
    for g in 0..G {
        for i in 0..n {
            xt[i][g] = x[g * n + i];
            wt[i][g] = weights[g * n + i];
        }
    }
    let (xt, wt) = (&xt[..n], &wt[..n]);
    let top_level = nmax as f32;
    let level = |v: f32| (v.round() as i32).clamp(0, nmax) as f32;

    let mut offset = xt[0];
    let mut top = xt[0];
    let mut w_total = wt[0];
    let mut wx_total = [0f32; G];
    for g in 0..G {
        wx_total[g] = w_total[g] * xt[0][g];
    }
    for i in 1..n {
        for g in 0..G {
            let v = xt[i][g];
            if v < offset[g] {
                offset[g] = v;
            }
            if v > top[g] {
                top[g] = v;
            }
            w_total[g] += wt[i][g];
            wx_total[g] += wt[i][g] * v;
        }
    }
    let mut flat = [false; G];
    for g in 0..G {
        if offset[g] > 0.0 {
            offset[g] = 0.0;
        }
        flat[g] = top[g] <= offset[g];
    }

    let mut per_unit = [0f32; G];
    let mut scale = [0f32; G];
    let mut best_error = [0f32; G];
    for g in 0..G {
        per_unit[g] = top_level / (top[g] - offset[g]);
        scale[g] = 1.0 / per_unit[g];
    }
    for i in 0..n {
        for g in 0..G {
            let diff =
                scale[g] * level(per_unit[g] * (xt[i][g] - offset[g])) + offset[g] - xt[i][g];
            best_error[g] += wt[i][g] * diff * diff;
        }
    }

    let mut levels = [[0f32; G]; MAX_SUB];
    for step in 0..=GRID_STEPS {
        for g in 0..G {
            per_unit[g] = (GRID_MIN + GRID_STEP * step as f32 + top_level) / (top[g] - offset[g]);
        }
        let (mut wl, mut wl2, mut wxl) = ([0f32; G], [0f32; G], [0f32; G]);
        for i in 0..n {
            for g in 0..G {
                let l = level(per_unit[g] * (xt[i][g] - offset[g]));
                levels[i][g] = l;
                let w = wt[i][g];
                wl[g] += w * l;
                wl2[g] += w * l.powi(2);
                wxl[g] += w * l * xt[i][g];
            }
        }
        let mut candidate_scale = [0f32; G];
        let mut candidate_offset = [0f32; G];
        let mut fits = [false; G];
        for g in 0..G {
            let denom = w_total[g] * wl2[g] - wl[g] * wl[g];
            fits[g] = denom > 0.0;
            candidate_scale[g] = (w_total[g] * wxl[g] - wx_total[g] * wl[g]) / denom;
            candidate_offset[g] = (wl2[g] * wx_total[g] - wl[g] * wxl[g]) / denom;
            if candidate_offset[g] > 0.0 {
                candidate_offset[g] = 0.0;
                candidate_scale[g] = wxl[g] / wl2[g];
            }
        }
        let mut error = [0f32; G];
        for i in 0..n {
            for g in 0..G {
                let diff = candidate_scale[g] * levels[i][g] + candidate_offset[g] - xt[i][g];
                error[g] += wt[i][g] * diff * diff;
            }
        }
        for g in 0..G {
            if fits[g] && !flat[g] && error[g] < best_error[g] {
                best_error[g] = error[g];
                scale[g] = candidate_scale[g];
                offset[g] = candidate_offset[g];
            }
        }
    }

    let mut out = [(0f32, 0f32); G];
    for g in 0..G {
        out[g] = if flat[g] {
            (0.0, -offset[g])
        } else {
            (scale[g], -offset[g])
        };
    }
    out
}

/// Fit non-negative `x` to levels `0..=nmax` with no offset, writing the levels into `l` and
/// returning the scale.
///
/// This is the second stage of a k-quant block: the per-sub-block scales and minimums that
/// `fit_scales_and_offsets` produced are themselves quantised, against `quant_weights` - the weight
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
