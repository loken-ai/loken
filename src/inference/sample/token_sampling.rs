//! Choosing the next token from a row of logits.
//!
//! Every mode here is the same two steps: decide which tokens are still candidates, then draw
//! one of them in proportion to its probability. Greedy is the degenerate case where the
//! candidate set is one token and there is nothing to draw.
//!
//! The draw is host-side by nature - a weighted choice needs the whole probability vector in
//! one place - so the logits come back from the device first, once per token.

use crate::tensor::Device;
use crate::tensor::{Error, Result, Tensor};
use rand::distr::Distribution;
use rand::SeedableRng;

/// How the candidates are chosen.
///
/// `temperature` divides the logits before they become probabilities: below one it sharpens
/// the distribution toward the leader, above one it flattens it.
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    /// The most probable token, always. No draw, so no seed dependence.
    ArgMax,
    /// Every token is a candidate.
    All { temperature: f64 },
    /// The `k` most probable tokens.
    TopK { k: usize, temperature: f64 },
    /// The shortest set of leading tokens whose probabilities reach `p` - the nucleus.
    TopP { p: f64, temperature: f64 },
    /// The nucleus, taken within the `k` most probable rather than within the whole vocabulary.
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

/// The `k` most probable tokens, in no particular order among themselves.
///
/// Ranking a whole vocabulary to keep a few of it is work for nothing: a partition puts the
/// `k` largest on one side and stops there.
fn most_probable(prs: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..prs.len()).collect();
    let (kept, _, _) = order.select_nth_unstable_by(k, |&i, &j| prs[j].total_cmp(&prs[i]));
    kept.to_vec()
}

/// Zero everything outside the nucleus.
///
/// Walking from most to least probable, the shortest prefix whose mass reaches `p` stays and
/// the tail goes. The prefix is allowed to overshoot `p` - the token that crosses the
/// threshold is inside it, or a `p` below the leader's own probability would keep nothing.
fn keep_nucleus(prs: &mut [f32], p: f32) {
    let mut order: Vec<usize> = (0..prs.len()).collect();
    order.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));
    let mut mass = 0.0f32;
    for i in order {
        if mass >= p {
            prs[i] = 0.0;
        } else {
            mass += prs[i];
        }
    }
}

impl LogitsProcessor {
    /// The seed is the whole state: two processors given the same one and the same mode draw
    /// the same tokens, which is what makes a generation reproducible.
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            sampling,
        }
    }

    /// The server's own spelling: a temperature indistinguishable from zero means greedy, and
    /// a `top_p` without a `top_k` means nucleus over the whole vocabulary.
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        // Dividing by a temperature this small overflows the logits rather than sharpening
        // them, so the limit is taken instead: below the threshold, greedy.
        let temperature = match temperature {
            Some(t) if t < 1e-7 => None,
            given => given,
        };
        let sampling = match (temperature, top_p) {
            (None, _) => Sampling::ArgMax,
            (Some(temperature), None) => Sampling::All { temperature },
            (Some(temperature), Some(p)) => Sampling::TopP { p, temperature },
        };
        Self::from_sampling(seed, sampling)
    }

    /// Draw one token in proportion to its weight. The weights need not sum to one.
    fn draw(&mut self, weights: &[f32]) -> Result<usize> {
        let distribution =
            rand::distr::weighted::WeightedIndex::new(weights).map_err(Error::msg)?;
        Ok(distribution.sample(&mut self.rng))
    }

    /// Draw from a restricted candidate set and answer with the vocabulary index.
    fn draw_among(&mut self, prs: &[f32], candidates: &[usize]) -> Result<u32> {
        let weights: Vec<f32> = candidates.iter().map(|&i| prs[i]).collect();
        let drawn = self.draw(&weights)?;
        Ok(candidates[drawn] as u32)
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_device(&Device::Cpu)?;
        let probabilities = |temperature: f64| -> Result<Vec<f32>> {
            let scaled = logits.scale((1.0 / temperature) as f32)?;
            Ok(scaled.softmax_last_dim()?.to_vec_f32())
        };

        match &self.sampling {
            Sampling::ArgMax => {
                // Vectorised, and first-max-wins like the scalar loop it replaces - greedy has
                // to be exact, and it runs once per token over the whole vocabulary.
                Ok(crate::inference::kernel::cpu_decode_exec::argmax_f32(
                    &logits.to_vec_f32(),
                ))
            }
            Sampling::All { temperature } => {
                let prs = probabilities(*temperature)?;
                Ok(self.draw(&prs)? as u32)
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = probabilities(*temperature)?;
                // A nucleus that covers everything, or nothing, is no restriction at all.
                if *p > 0.0 && *p < 1.0 {
                    keep_nucleus(&mut prs, *p as f32);
                }
                Ok(self.draw(&prs)? as u32)
            }
            Sampling::TopK { k, temperature } => {
                let prs = probabilities(*temperature)?;
                if *k >= prs.len() {
                    return Ok(self.draw(&prs)? as u32);
                }
                let candidates = most_probable(&prs, *k);
                self.draw_among(&prs, &candidates)
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = probabilities(*temperature)?;
                if *k >= prs.len() {
                    if *p > 0.0 && *p < 1.0 {
                        keep_nucleus(&mut prs, *p as f32);
                    }
                    return Ok(self.draw(&prs)? as u32);
                }
                let candidates = most_probable(&prs, *k);
                // The nucleus is taken inside what `k` left, so the threshold is compared
                // against that set's own mass rather than against one.
                let mut kept: Vec<f32> = candidates.iter().map(|&i| prs[i]).collect();
                let mass: f32 = kept.iter().sum();
                let p = *p as f32;
                if p > 0.0 && p < mass {
                    keep_nucleus(&mut kept, p);
                }
                let drawn = self.draw(&kept)?;
                Ok(candidates[drawn] as u32)
            }
        }
    }
}
