//! Drawing a code from a row of logits.
//!
//! The talker's row first goes through the penalty, the suppressed ids and the minimum before
//! the end of speech.
//! Both rows then go through the temperature, the top-k and the top-p before the draw.

use super::config::{Sampling, SpeechIds};
use crate::tensor::{Error, Result};
use rand::distr::Distribution;

/// Frames an utterance holds before the end of speech may close it.
///
/// The reference generation passes this as its `min_new_tokens`.
/// No file of the checkpoint states it.
pub const MIN_FRAMES_BEFORE_EOS: usize = 2;

/// `logits` with each code of `drawn` penalised once: a positive logit divided by `penalty`,
/// a negative one multiplied.
pub fn penalise(logits: &mut [f32], drawn: &[u32], penalty: f32) {
    let mut seen = std::collections::HashSet::new();
    for &id in drawn {
        if !seen.insert(id) {
            continue;
        }
        if let Some(v) = logits.get_mut(id as usize) {
            *v = if *v < 0.0 { *v * penalty } else { *v / penalty };
        }
    }
}

/// The talker's `logits` of frame `frame` made ready for the draw.
///
/// The codes of group 0 already drawn are penalised.
/// The suppressed ids are banned.
/// The end of speech is banned before [`MIN_FRAMES_BEFORE_EOS`] frames.
pub fn talker_logits(
    logits: &mut [f32],
    ids: &SpeechIds,
    frame: usize,
    drawn: &[u32],
    penalty: f32,
) {
    penalise(logits, drawn, penalty);
    for id in ids.suppressed() {
        if let Some(v) = logits.get_mut(id as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
    if frame < MIN_FRAMES_BEFORE_EOS {
        if let Some(v) = logits.get_mut(ids.codec_eos as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// The index of the largest of `v`, the first of equal ones.
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// A code drawn from `logits` as `s` says, with `rng`.
pub fn draw(logits: &[f32], s: &Sampling, rng: &mut rand::rngs::StdRng) -> Result<u32> {
    if !s.sample {
        return Ok(argmax(logits));
    }
    let mut v: Vec<f32> = logits.iter().map(|x| x / s.temperature).collect();
    if s.top_k > 0 && s.top_k < v.len() {
        let mut sorted = v.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let kth = sorted[s.top_k - 1];
        for x in v.iter_mut().filter(|x| **x < kth) {
            *x = f32::NEG_INFINITY;
        }
    }
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(Error::msg("no code is left to draw"));
    }
    let mut p: Vec<f64> = v.iter().map(|x| ((x - max) as f64).exp()).collect();
    let total: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= total);
    if s.top_p < 1.0 {
        // The least probable codes go while the share they hold together stays within
        // `1 - top_p`.
        // The most probable code always stays.
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|a, b| p[*a].total_cmp(&p[*b]));
        let mut gone = 0.0;
        for &i in &order[..order.len() - 1] {
            gone += p[i];
            if gone > 1.0 - s.top_p as f64 {
                break;
            }
            p[i] = 0.0;
        }
    }
    let pick = rand::distr::weighted::WeightedIndex::new(&p).map_err(Error::msg)?;
    Ok(pick.sample(rng) as u32)
}
