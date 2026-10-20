//! The host-side arithmetic around the network: the causal per-patch frame a row is read in,
//! the whole-row frame of a covariate known over the horizon, the monotone quantile fan, the
//! frames a decoded block is re-read in, and the target encoding of categorical covariates.
//!
//! Statistics accumulate in `f64` and are handed on in `f32`; the transforms run in `f32`.

/// A running weight, mean and sum of squared deviations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Moments {
    pub count: f64,
    pub mean: f64,
    pub m2: f64,
}

impl Moments {
    /// The moments of `self` and `other` together. Two empty states give an empty state.
    pub fn merge(self, other: Self) -> Self {
        let count = self.count + other.count;
        if count == 0.0 {
            return Self::default();
        }
        let delta = other.mean - self.mean;
        Self {
            count,
            mean: self.mean + delta * other.count / count,
            m2: self.m2 + other.m2 + delta * delta * self.count * other.count / count,
        }
    }

    /// The moments of the observed values of a run of points.
    pub fn of(values: &[f32], observed: &[bool]) -> Self {
        let count = observed.iter().filter(|o| **o).count() as f64;
        if count == 0.0 {
            return Self::default();
        }
        let seen = || values.iter().zip(observed).filter(|(_, o)| **o);
        let mean = seen().map(|(v, _)| f64::from(*v)).sum::<f64>() / count;
        let m2 = seen().map(|(v, _)| (f64::from(*v) - mean).powi(2)).sum();
        Self { count, mean, m2 }
    }

    /// The spread with Bessel's correction, floored at `minimum`.
    pub fn scale(&self, minimum: f64) -> f64 {
        (self.m2 / (self.count - 1.0).max(1.0))
            .max(0.0)
            .sqrt()
            .max(minimum)
    }
}

/// The causal mean and spread at every point of a row: point `t` reads the observed values
/// at `0..=t`. Bessel's correction, the spread floored at `minimum` after the root.
pub fn causal_statistics(values: &[f32], observed: &[bool], minimum: f64) -> (Vec<f32>, Vec<f32>) {
    let (mut cum_w, mut cum_x, mut m2, mut prev_mean) = (0f64, 0f64, 0f64, 0f64);
    let mut means = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len());
    for (v, o) in values.iter().zip(observed) {
        let (x, w) = (f64::from(*v), f64::from(u8::from(*o)));
        cum_w += w;
        cum_x += w * x;
        let denominator = cum_w.max(1.0);
        let mean = cum_x / denominator;
        m2 += (x - prev_mean) * (x - mean) * w;
        prev_mean = mean;
        let variance = m2 / (denominator - 1.0).max(1.0);
        means.push(mean as f32);
        scales.push(variance.sqrt().max(minimum) as f32);
    }
    (means, scales)
}

/// The frame of each patch of a row: the causal statistics at the patch's last point.
pub fn patch_frames(
    values: &[f32],
    observed: &[bool],
    patch: usize,
    minimum: f64,
) -> (Vec<f32>, Vec<f32>) {
    let (means, scales) = causal_statistics(values, observed, minimum);
    let last = |s: &[f32]| s.chunks(patch).map(|c| c[c.len() - 1]).collect::<Vec<_>>();
    (last(&means), last(&scales))
}

/// A row's whole-row frame, for a covariate known over the horizon: population spread over
/// every observed point. A row whose observed values are all 0 or 1 keeps the identity frame
/// and is flagged as an indicator.
pub fn row_frame(values: &[f32], observed: &[bool], minimum: f64) -> (f32, f32, bool) {
    let any = observed.iter().any(|o| *o);
    let indicator = any
        && values
            .iter()
            .zip(observed)
            .all(|(v, o)| !*o || *v == 0.0 || *v == 1.0);
    if indicator {
        return (0.0, 1.0, true);
    }
    let n = (observed.iter().filter(|o| **o).count() as f64).max(1.0);
    let seen = || values.iter().zip(observed).filter(|(_, o)| **o);
    let loc = seen().map(|(v, _)| f64::from(*v)).sum::<f64>() / n;
    let var = seen()
        .map(|(v, _)| (f64::from(*v) - loc).powi(2))
        .sum::<f64>()
        / n;
    (loc as f32, var.sqrt().max(minimum) as f32, false)
}

/// `(v - loc) / scale`, squashed by `asinh` when `squash`, on an observed point; 0 elsewhere.
pub fn normalize(v: f32, loc: f32, scale: f32, observed: bool, squash: bool) -> f32 {
    if !observed {
        return 0.0;
    }
    let s = (v - loc) / scale.max(f32::EPSILON);
    if squash {
        s.asinh()
    } else {
        s
    }
}

/// `softplus` with the usual linear tail past 20.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Non-crossing quantiles from one step's raw head outputs, one per level, in place. The
/// median channel is free; each other level sits `width * gap * softplus(raw)` further out
/// than the next level inward, `gap` being the probability between the two.
pub fn monotone(raw: &mut [f32], levels: &[f32], median: usize, width: f32) {
    let gap = |i: usize| {
        if i > median {
            levels[i] - levels[i - 1]
        } else {
            levels[i + 1] - levels[i]
        }
    };
    let inc: Vec<f32> = (0..raw.len())
        .map(|i| {
            if i == median {
                0.0
            } else {
                width * gap(i) * softplus(raw[i])
            }
        })
        .collect();
    let m = raw[median];
    let mut acc = 0f32;
    for i in median + 1..raw.len() {
        acc += inc[i];
        raw[i] = (m + acc).clamp(-f32::MAX, f32::MAX);
    }
    acc = 0.0;
    for i in (0..median).rev() {
        acc += inc[i];
        raw[i] = (m - acc).clamp(-f32::MAX, f32::MAX);
    }
}

/// Infinities of one step's quantiles mapped to the nearest finite value; a step without a
/// finite value becomes 0. NaN is left as it is.
pub fn clamp_nonfinite(q: &mut [f32]) {
    let finite = || q.iter().copied().filter(|v| v.is_finite());
    let hi = finite().fold(f32::NEG_INFINITY, f32::max);
    let lo = finite().fold(f32::INFINITY, f32::min);
    for v in q.iter_mut() {
        if *v == f32::INFINITY {
            *v = hi;
        } else if *v == f32::NEG_INFINITY {
            *v = lo;
        }
        if v.is_infinite() {
            *v = 0.0;
        }
    }
}

/// The frame each patch of a decoded block is re-read in: the statistics of the observed
/// context and of the block's medians before it. `sinh_median[h]` is patch `h`'s median in
/// the last context patch's frame with the squash undone. Returns `(loc, scale)` per patch.
pub fn advancing_frames(
    context: &[f32],
    observed: &[bool],
    patch: usize,
    sinh_median: &[Vec<f64>],
    minimum: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut state = context
        .chunks(patch)
        .zip(observed.chunks(patch))
        .map(|(v, o)| Moments::of(v, o))
        .fold(Moments::default(), Moments::merge);
    // Medians are clamped relative to the context mean, so a large level survives.
    let offset = state.mean;
    let clip = f64::from(f32::MAX).sqrt() / 1e3;
    let (mut locs, mut scales) = (Vec::new(), Vec::new());
    for med in sinh_median {
        let scale = state.scale(minimum);
        locs.push(state.mean);
        scales.push(scale);
        let folded: Vec<f64> = med
            .iter()
            .map(|m| (m * scale + state.mean - offset).clamp(-clip, clip) + offset)
            .collect();
        let mean = folded.iter().sum::<f64>() / folded.len() as f64;
        let m2 = folded.iter().map(|f| (f - mean).powi(2)).sum();
        state = state.merge(Moments {
            count: folded.len() as f64,
            mean,
            m2,
        });
    }
    (locs, scales)
}

/// Smoothed mean-target encoding of one categorical covariate of a series: category `c`
/// becomes `(smooth * mean + sum_c) / (smooth + count_c)` over the observed target. A missing
/// code is a category of its own; a code the context never shows becomes the series mean.
/// Returns the context's codes encoded, and the horizon's when given.
pub fn target_encode(
    target: &[f64],
    past: &[f64],
    future: Option<&[f64]>,
    smooth: f64,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let seen = || target.iter().filter(|t| t.is_finite());
    let count = seen().count() as f64;
    let mean = if count > 0.0 {
        seen().sum::<f64>() / count
    } else {
        0.0
    };
    // Per category (NaN as its own): sum and count of the observed target.
    let key = |c: f64| if c.is_nan() { None } else { Some(c.to_bits()) };
    let mut table: std::collections::HashMap<Option<u64>, (f64, f64)> = Default::default();
    for (t, c) in target.iter().zip(past) {
        let e = table.entry(key(*c)).or_default();
        if t.is_finite() {
            e.0 += t;
            e.1 += 1.0;
        }
    }
    let lookup = |c: f64| -> f32 {
        match table.get(&key(c)) {
            Some((s, n)) => ((smooth * mean + s) / (smooth + n)) as f32,
            None if c.is_nan() => ((smooth * mean) / smooth) as f32,
            None => mean as f32,
        }
    };
    let enc_past = past.iter().map(|c| lookup(*c)).collect();
    let enc_future = future.map(|f| f.iter().map(|c| lookup(*c)).collect());
    (enc_past, enc_future)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_statistics_read_only_the_past() {
        let v = [1.0, 3.0, 0.0, 5.0];
        let o = [true, true, false, true];
        let (m, s) = causal_statistics(&v, &o, 1e-6);
        assert_eq!(m[0], 1.0);
        assert_eq!(m[1], 2.0);
        assert_eq!(m[2], 2.0);
        assert!((m[3] - 3.0).abs() < 1e-6);
        assert!((s[1] - 2f32.sqrt()).abs() < 1e-6);
        assert!((s[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn merged_moments_match_one_pass() {
        let v = [1.0f32, 2.0, 4.0, 8.0, 3.0];
        let all = [true; 5];
        let whole = Moments::of(&v, &all);
        let split = Moments::of(&v[..2], &all[..2]).merge(Moments::of(&v[2..], &all[2..]));
        assert!((whole.mean - split.mean).abs() < 1e-12);
        assert!((whole.m2 - split.m2).abs() < 1e-9);
    }

    #[test]
    fn monotone_quantiles_do_not_cross() {
        let levels = [0.1, 0.5, 0.9];
        let mut raw = [-3.0, 2.0, -5.0];
        monotone(&mut raw, &levels, 1, 3.0);
        assert!(raw[0] < raw[1] && raw[1] < raw[2], "{raw:?}");
        assert_eq!(raw[1], 2.0);
    }

    #[test]
    fn indicator_rows_keep_the_identity_frame() {
        assert_eq!(
            row_frame(&[0.0, 1.0, 1.0], &[true; 3], 1e-6),
            (0.0, 1.0, true)
        );
        let (loc, scale, flag) = row_frame(&[1.0, 3.0], &[true; 2], 1e-6);
        assert!(!flag && loc == 2.0 && scale == 1.0);
    }

    #[test]
    fn unseen_categories_become_the_series_mean() {
        let target = [1.0, 3.0, 5.0, f64::NAN];
        let codes = [0.0, 0.0, 1.0, 1.0];
        let (past, future) = target_encode(&target, &codes, Some(&[2.0, 1.0]), 1.0);
        assert!((past[0] - (3.0 + 4.0) / 3.0).abs() < 1e-6);
        assert!((past[2] - (3.0 + 5.0) / 2.0).abs() < 1e-6);
        let future = future.unwrap();
        assert_eq!(future[0], 3.0);
        assert_eq!(future[1], past[2]);
    }
}
