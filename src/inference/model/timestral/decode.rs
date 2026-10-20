//! A forecast around the network: the request's rows assembled, decoded block by block, each
//! block re-read in its advancing frame, capped, averaged with the forecast of the mirrored
//! series and clipped at zero for a series that never goes below it.

use super::{frame, Network, Packed, Role};
use crate::inference::model::chronos2::Forecast;
use crate::tensor;

/// A forecast request: one target series and its covariates. A non-finite value is a
/// missing point.
#[derive(Debug, Clone, Default)]
pub struct Request {
    pub series: Vec<f32>,
    pub horizon: usize,
    /// Covariates known over the context: each as long as `series`.
    pub past: Vec<Vec<f32>>,
    /// Covariates known over the context and the horizon: each `series` plus `horizon` long.
    pub known: Vec<Vec<f32>>,
    /// Category codes known over the context; NaN is a missing code.
    pub categorical_past: Vec<Vec<f64>>,
    /// Category codes known over the context and the horizon.
    pub categorical_known: Vec<Vec<f64>>,
}

impl Request {
    pub fn univariate(series: Vec<f32>, horizon: usize) -> Self {
        Self {
            series,
            horizon,
            ..Self::default()
        }
    }

    pub fn has_covariates(&self) -> bool {
        !(self.past.is_empty()
            && self.known.is_empty()
            && self.categorical_past.is_empty()
            && self.categorical_known.is_empty())
    }

    pub fn has_known(&self) -> bool {
        !(self.known.is_empty() && self.categorical_known.is_empty())
    }
}

/// Clamp a forecast to the observed range widened by this many spreads of the last patch.
const CAP_SPREADS: f32 = 1e4;
/// A fed-back median at or past this magnitude is fed back as 0.
const EXTREME: f32 = 1e30;
/// The weight of the series mean in a category's mean.
const ENCODING_SMOOTH: f64 = 1.0;
/// How far a level and its mirror may sum away from 1 and still be read as a pair.
const MIRROR_TOLERANCE: f32 = 1e-6;

/// One pass's rows before packing: the target first, then past and known covariates.
struct Rows {
    values: Vec<Vec<f32>>,
    observed: Vec<Vec<bool>>,
    roles: Vec<Role>,
    /// The horizon of each known covariate row, `None` on the others.
    future: Vec<Option<(Vec<f32>, Vec<bool>)>>,
}

/// The network and how it decodes.
#[derive(Debug, Clone)]
pub struct Forecaster {
    net: Network,
    /// Patches decoded per block.
    per_shot: usize,
}

/// Unobserved points carry the last observed value before them; leading ones carry the
/// first point's value, which is 0 when it is unobserved.
fn forward_fill(values: &[f32], observed: &[bool]) -> Vec<f32> {
    let mut last = 0usize;
    (0..values.len())
        .map(|t| {
            if observed[t] {
                last = t;
            }
            let v = values[last];
            if v.is_nan() {
                0.0
            } else {
                v
            }
        })
        .collect()
}

/// Values with missing points zeroed, and the observed mask.
fn split_missing(row: &[f32]) -> (Vec<f32>, Vec<bool>) {
    let observed: Vec<bool> = row.iter().map(|v| !v.is_nan()).collect();
    (
        row.iter()
            .map(|v| if v.is_nan() { 0.0 } else { *v })
            .collect(),
        observed,
    )
}

impl Forecaster {
    pub fn new(net: Network) -> Self {
        let per_shot = net.config().block_size.max(1);
        Self { net, per_shot }
    }

    pub fn network(&self) -> &Network {
        &self.net
    }

    /// The quantile levels the head answers.
    pub fn levels(&self) -> &[f32] {
        &self.net.config().quantile_knots
    }

    /// Check what a request asks of this checkpoint.
    pub fn check(&self, req: &Request) -> tensor::Result<()> {
        let c = self.net.config();
        let t = req.series.len();
        if req.horizon == 0 || req.horizon > c.max_horizon() {
            tensor::bail!("horizon {} is outside 1..={}", req.horizon, c.max_horizon());
        }
        if !req.series.iter().any(|v| v.is_finite()) {
            tensor::bail!("the series holds no observed value");
        }
        if req.has_covariates() && !c.covariate_enabled {
            tensor::bail!("this checkpoint was not trained with covariates");
        }
        if req.has_known() && !c.supports_known_covariates() {
            tensor::bail!("this checkpoint cannot read covariates known over the horizon");
        }
        let past = req.past.iter().map(Vec::len);
        let cat_past = req.categorical_past.iter().map(Vec::len);
        if past.chain(cat_past).any(|l| l != t) {
            tensor::bail!("a past covariate must be as long as the series, {t} points");
        }
        let known = req.known.iter().map(Vec::len);
        let cat_known = req.categorical_known.iter().map(Vec::len);
        if known.chain(cat_known).any(|l| l != t + req.horizon) {
            tensor::bail!(
                "a known covariate must cover the series and the horizon, {} points",
                t + req.horizon
            );
        }
        Ok(())
    }

    /// Forecast `req.horizon` steps after `req.series`.
    pub fn forecast(&self, req: &Request) -> tensor::Result<Forecast> {
        self.check(req)?;
        let c = self.net.config();
        let levels = c.quantile_knots.clone();
        let q = levels.len();
        let mask: Vec<bool> = req.series.iter().map(|v| v.is_finite()).collect();
        let inputs: Vec<f32> = req
            .series
            .iter()
            .map(|v| if v.is_finite() { *v } else { 0.0 })
            .collect();

        // Categorical covariates become numbers, encoded against the series.
        let target = self.encoding_target(&inputs, &mask);
        let encode = |codes: &Vec<f64>| -> Vec<f32> {
            let t = target.len();
            let future = (codes.len() > t).then(|| &codes[t..]);
            let (past, fut) = frame::target_encode(&target, &codes[..t], future, ENCODING_SMOOTH);
            let mut row: Vec<f32> = past
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    // A missing code before the target starts is padding.
                    if target[i].is_nan() && codes[i].is_nan() {
                        f32::NAN
                    } else {
                        *v
                    }
                })
                .collect();
            row.extend(fut.unwrap_or_default());
            row
        };
        let mut past = req.past.clone();
        past.extend(req.categorical_past.iter().map(encode));
        let mut known = req.known.clone();
        known.extend(req.categorical_known.iter().map(encode));

        let negate = |rows: &[Vec<f32>]| -> Vec<Vec<f32>> {
            rows.iter()
                .map(|r| r.iter().map(|v| -v).collect())
                .collect()
        };
        let neg_inputs: Vec<f32> = inputs.iter().map(|v| -v).collect();
        let mirror = self.symmetric(&levels);
        let mut passes = vec![self.assemble(&inputs, &mask, req.horizon, &past, &known)];
        if mirror {
            passes.push(self.assemble(
                &neg_inputs,
                &mask,
                req.horizon,
                &negate(&past),
                &negate(&known),
            ));
        }
        let per_pass = passes[0].values.len();
        let raw = self.decode(&passes, req.horizon)?;

        // The target row of each pass, capped to the range its context saw.
        let h = req.horizon;
        let mut outs = Vec::with_capacity(passes.len());
        for (i, pass) in passes.iter().enumerate() {
            let mut out = raw[i * per_pass * h * q..][..h * q].to_vec();
            self.cap(&mut out, &pass.values[0], &pass.observed[0]);
            if out.iter().any(|v| v.is_nan()) {
                tensor::bail!("the forecast holds NaN values; the input is degenerate");
            }
            outs.push(out);
        }
        let mut out = outs[0].clone();
        if mirror {
            let m = &outs[1];
            for t in 0..h {
                for l in 0..q {
                    out[t * q + l] = 0.5 * (out[t * q + l] + -m[t * q + (q - 1 - l)]);
                }
            }
        }
        let observed_any = mask.iter().any(|m| *m);
        let nonnegative = inputs.iter().zip(&mask).all(|(v, m)| !*m || *v >= 0.0);
        if observed_any && nonnegative {
            out.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        let values = (0..q)
            .map(|l| (0..h).map(|t| out[t * q + l]).collect())
            .collect();
        Ok(Forecast { levels, values })
    }

    /// Whether every level has its mirror `1 - level`, the condition of averaging with the
    /// forecast of the negated series.
    fn symmetric(&self, levels: &[f32]) -> bool {
        levels
            .iter()
            .zip(levels.iter().rev())
            .all(|(lo, hi)| (lo + hi - 1.0).abs() < MIRROR_TOLERANCE)
    }

    /// The target categories are encoded against: the forward-filled series from its first
    /// observation, over the last context length; NaN elsewhere.
    fn encoding_target(&self, inputs: &[f32], mask: &[bool]) -> Vec<f64> {
        let t = inputs.len();
        let window = t.saturating_sub(self.net.config().context_length);
        let filled = forward_fill(inputs, mask);
        let mut started = false;
        (0..t)
            .map(|i| {
                started |= mask[i];
                if started && i >= window {
                    f64::from(filled[i])
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// One pass's rows: the target forward-filled, then each past covariate, then each known
    /// covariate's context with its horizon set aside; cut to the context length and
    /// left-padded to whole patches.
    fn assemble(
        &self,
        inputs: &[f32],
        mask: &[bool],
        horizon: usize,
        past: &[Vec<f32>],
        known: &[Vec<f32>],
    ) -> Rows {
        let c = self.net.config();
        let ps = c.patch_size;
        let t = inputs.len();
        let mut rows = Rows {
            values: vec![forward_fill(inputs, mask)],
            observed: vec![mask.to_vec()],
            roles: vec![Role::Target],
            future: vec![None],
        };
        for p in past {
            let (v, o) = split_missing(p);
            rows.values.push(v);
            rows.observed.push(o);
            rows.roles.push(Role::Past);
            rows.future.push(None);
        }
        for k in known {
            let (v, o) = split_missing(&k[..t]);
            rows.values.push(v);
            rows.observed.push(o);
            rows.roles.push(Role::Known);
            rows.future.push(Some(split_missing(&k[t..t + horizon])));
        }
        let keep = t.min(c.context_length);
        let pad = keep.next_multiple_of(ps) - keep;
        for (v, o) in rows.values.iter_mut().zip(rows.observed.iter_mut()) {
            let mut nv = vec![0f32; pad];
            nv.extend_from_slice(&v[t - keep..]);
            let mut no = vec![false; pad];
            no.extend_from_slice(&o[t - keep..]);
            *v = nv;
            *o = no;
        }
        rows
    }

    /// Decode every pass together, one group per pass, block by block. Returns per pass and
    /// row `horizon` steps of quantiles, flattened `[pass][row][step][level]`.
    fn decode(&self, passes: &[Rows], horizon: usize) -> tensor::Result<Vec<f32>> {
        let c = self.net.config();
        let ps = c.patch_size;
        let q = c.quantile_knots.len();
        let median = c.median_index().unwrap_or(0);
        let with_covariates = passes[0].roles.iter().any(|r| *r != Role::Target);
        let roles: Vec<Role> = passes.iter().flat_map(|p| p.roles.clone()).collect();
        let group: Vec<usize> = passes
            .iter()
            .enumerate()
            .flat_map(|(i, p)| std::iter::repeat_n(i, p.roles.len()))
            .collect();
        let rows = roles.len();
        let mut context: Vec<Vec<f32>> = passes.iter().flat_map(|p| p.values.clone()).collect();
        let mut mask: Vec<Vec<bool>> = passes.iter().flat_map(|p| p.observed.clone()).collect();
        let span = horizon.div_ceil(ps) * ps;
        let futures: Vec<(Vec<f32>, Vec<bool>)> = passes
            .iter()
            .flat_map(|p| p.future.clone())
            .map(|f| {
                let (mut v, mut o) = f.unwrap_or_default();
                v.resize(span, 0.0);
                o.resize(span, false);
                (v, o)
            })
            .collect();
        let is_target = |r: usize| !with_covariates || roles[r] == Role::Target;

        let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(span * q); rows];
        let (mut remaining, mut offset) = (horizon.div_ceil(ps), 0usize);
        while remaining > 0 {
            let block = remaining.min(self.per_shot);
            let blen = block * ps;
            let len = context[0].len();
            let known_future = with_covariates.then(|| {
                let mut v = Vec::with_capacity(rows * blen);
                let mut o = Vec::with_capacity(rows * blen);
                for (fv, fo) in &futures {
                    v.extend_from_slice(&fv[offset..offset + blen]);
                    o.extend_from_slice(&fo[offset..offset + blen]);
                }
                (v, o)
            });
            let packed = Packed {
                rows,
                len,
                values: context.concat(),
                observed: mask.concat(),
                group: group.clone(),
                roles: with_covariates.then(|| roles.clone()),
                known_future,
            };
            let mut quantiles = self.net.predict_block(&packed, block, None)?;
            for r in (0..rows).filter(|r| is_target(*r)) {
                let at = r * blen * q;
                self.refine(
                    &mut quantiles[at..at + blen * q],
                    &context[r],
                    &mask[r],
                    block,
                );
            }
            for r in 0..rows {
                out[r].extend_from_slice(&quantiles[r * blen * q..][..blen * q]);
            }
            remaining -= block;
            if remaining == 0 {
                break;
            }
            // The context grows by the block: a target by its median, a known covariate by
            // its real horizon, a past covariate by a blank.
            for r in 0..rows {
                let (v, o): (Vec<f32>, Vec<bool>) = match (with_covariates, roles[r]) {
                    (true, Role::Known) => (
                        futures[r].0[offset..offset + blen].to_vec(),
                        futures[r].1[offset..offset + blen].to_vec(),
                    ),
                    (true, Role::Past) => (vec![0.0; blen], vec![false; blen]),
                    _ => (
                        (0..blen)
                            .map(|t| {
                                let m = quantiles[(r * blen + t) * q + median];
                                if m.is_finite() && m.abs() < EXTREME {
                                    m
                                } else {
                                    0.0
                                }
                            })
                            .collect(),
                        vec![true; blen],
                    ),
                };
                context[r].extend(v);
                mask[r].extend(o);
                let keep = (c.context_length / ps * ps).max(ps);
                if context[r].len() > c.context_length {
                    let cut = context[r].len() - keep;
                    context[r].drain(..cut);
                    mask[r].drain(..cut);
                }
            }
            offset += blen;
        }
        Ok(out
            .into_iter()
            .flat_map(|row| row.into_iter().take(horizon * q))
            .collect())
    }

    /// Move one decoded block of a row from the frame of its last context patch to the
    /// advancing frame: horizon patch `j` reads the statistics of the context and of the
    /// block's medians before it. The first patch keeps its frame.
    fn refine(&self, q: &mut [f32], context: &[f32], mask: &[bool], patches: usize) {
        if patches <= 1 {
            return;
        }
        let c = self.net.config();
        let (ps, levels) = (c.patch_size, c.quantile_knots.len());
        let median = c.median_index().unwrap_or(0);
        let (means, scales) = frame::causal_statistics(context, mask, c.minimum_scale);
        let (loc0, scale0) = (means[means.len() - 1], scales[scales.len() - 1]);
        let sinh_median: Vec<Vec<f64>> = (0..patches)
            .map(|j| {
                (0..ps)
                    .map(|t| {
                        let m = f64::from(q[(j * ps + t) * levels + median]);
                        (m - f64::from(loc0)) / f64::from(scale0)
                    })
                    .collect()
            })
            .collect();
        let (locs, scales) =
            frame::advancing_frames(context, mask, ps, &sinh_median, c.minimum_scale);
        for j in 1..patches {
            let (loc, scale) = (locs[j] as f32, scales[j] as f32);
            for t in 0..ps {
                let step = &mut q[(j * ps + t) * levels..][..levels];
                for v in step.iter_mut() {
                    *v = (*v - loc0) / scale0 * scale + loc;
                }
                frame::clamp_nonfinite(step);
            }
        }
    }

    /// Clamp a target's quantiles `[step][level]` to its observed range widened by
    /// [`CAP_SPREADS`] spreads of its last point.
    fn cap(&self, q: &mut [f32], values: &[f32], observed: &[bool]) {
        let c = self.net.config();
        let seen = || {
            values
                .iter()
                .zip(observed)
                .filter(|(_, o)| **o)
                .map(|(v, _)| *v)
        };
        let hi = seen().fold(f32::NEG_INFINITY, f32::max);
        let lo = seen().fold(f32::INFINITY, f32::min);
        let (hi, lo) = if hi.is_finite() { (hi, lo) } else { (0.0, 0.0) };
        let (_, scales) = frame::causal_statistics(values, observed, c.minimum_scale);
        let spread = scales[scales.len() - 1] * CAP_SPREADS;
        q.iter_mut()
            .for_each(|v| *v = v.clamp(lo - spread, hi + spread));
    }
}
