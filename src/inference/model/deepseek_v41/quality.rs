//! A teacher-forced quality instrument for settings that give up exactness. One fixed token
//! sequence is run through the model once exactly, recording at each position the top-20
//! log-probabilities; the same sequence is then run under each setting, and the setting is
//! scored against the exact run: how often its top choice is the exact one, the divergence of
//! its distribution from the exact one over the recorded top-20, and the log-probability it gives
//! the exact run's choice. A prefill over the whole sequence is the teacher-forced pass: every
//! position's logits come from the true tokens before it.
//!
//! The runs are drivers the caller writes against the real checkpoint (no
//! file or model name lives here).

/// Positions' top-20 of an exact run, and the scores of a run against it.
pub struct Reference {
    pub ids: Vec<[u32; TOP]>,
    pub logprobs: Vec<[f64; TOP]>,
}

pub const TOP: usize = 20;

/// The log-softmax of one row of logits.
pub fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    row.iter().map(|&x| x as f64 - lse).collect()
}

/// The `TOP` most likely tokens of a row of log-probabilities, most likely first.
pub fn top(lp: &[f64]) -> ([u32; TOP], [f64; TOP]) {
    let mut idx: Vec<u32> = (0..lp.len() as u32).collect();
    idx.select_nth_unstable_by(TOP, |&a, &b| lp[b as usize].total_cmp(&lp[a as usize]));
    let mut head: Vec<u32> = idx[..TOP].to_vec();
    head.sort_by(|&a, &b| lp[b as usize].total_cmp(&lp[a as usize]));
    let mut ids = [0u32; TOP];
    let mut vals = [0f64; TOP];
    for (i, &t) in head.iter().enumerate() {
        ids[i] = t;
        vals[i] = lp[t as usize];
    }
    (ids, vals)
}

/// KL(exact || variant) over the exact run's top-20, both renormalised over those tokens.
pub fn kl_over_top(exact: &[f64; TOP], variant_at_ids: &[f64; TOP]) -> f64 {
    let norm = |v: &[f64; TOP]| -> Vec<f64> {
        let s: f64 = v.iter().map(|x| x.exp()).sum();
        v.iter().map(|x| x.exp() / s).collect()
    };
    let (p, q) = (norm(exact), norm(variant_at_ids));
    p.iter()
        .zip(&q)
        .map(|(p, q)| {
            if *p > 0.0 {
                p * (p / q.max(1e-300)).ln()
            } else {
                0.0
            }
        })
        .sum()
}

#[derive(Debug, Default)]
pub struct Score {
    pub positions: usize,
    pub top1_agreement: f64,
    pub mean_kl: f64,
    pub mean_logprob_of_exact_choice: f64,
    pub mean_nll_of_true_next: f64,
}

/// Score `logits` (one row per position) against `reference`; `next` is the true token that
/// follows each position, `None` for the last.
pub fn score(logits: &[Vec<f32>], reference: &Reference, next: &[Option<u32>]) -> Score {
    let mut s = Score::default();
    let mut nll_n = 0usize;
    for (i, row) in logits.iter().enumerate() {
        let lp = log_softmax(row);
        let ids = &reference.ids[i];
        let at: [f64; TOP] = std::array::from_fn(|j| lp[ids[j] as usize]);
        let var_top = lp
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(t, _)| t as u32);
        s.positions += 1;
        s.top1_agreement += (var_top == ids[0]) as usize as f64;
        s.mean_kl += kl_over_top(&reference.logprobs[i], &at);
        s.mean_logprob_of_exact_choice += at[0];
        if let Some(t) = next[i] {
            s.mean_nll_of_true_next -= lp[t as usize];
            nll_n += 1;
        }
    }
    let n = s.positions.max(1) as f64;
    s.top1_agreement /= n;
    s.mean_kl /= n;
    s.mean_logprob_of_exact_choice /= n;
    s.mean_nll_of_true_next /= nll_n.max(1) as f64;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measures on rows built by hand: a run against itself scores perfectly, and a variant
    /// that flips the top two tokens loses agreement and gains divergence.
    #[test]
    fn the_scores_read_what_they_should() {
        let vocab = 64usize;
        let row = |a: usize, b: usize| -> Vec<f32> {
            let mut r = vec![0f32; vocab];
            r[a] = 3.0;
            r[b] = 2.5;
            r
        };
        let rows = vec![row(5, 9), row(7, 11)];
        let mut reference = Reference {
            ids: Vec::new(),
            logprobs: Vec::new(),
        };
        for r in &rows {
            let (ids, vals) = top(&log_softmax(r));
            reference.ids.push(ids);
            reference.logprobs.push(vals);
        }
        let next = [Some(5), None];
        let same = score(&rows, &reference, &next);
        assert_eq!(same.top1_agreement, 1.0);
        assert!(same.mean_kl.abs() < 1e-12);
        let flipped = vec![row(9, 5), row(7, 11)];
        let f = score(&flipped, &reference, &next);
        assert_eq!(f.top1_agreement, 0.5);
        assert!(f.mean_kl > 0.0);
        assert!(f.mean_logprob_of_exact_choice < same.mean_logprob_of_exact_choice);
    }
}
