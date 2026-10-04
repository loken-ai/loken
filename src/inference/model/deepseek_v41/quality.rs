//! A teacher-forced quality instrument for settings that give up exactness. One fixed token
//! sequence is run through the model once exactly, recording at each position the top-20
//! log-probabilities; the same sequence is then run under each setting, and the setting is
//! scored against the exact run: how often its top choice is the exact one, the divergence of
//! its distribution from the exact one over the recorded top-20, and the log-probability it gives
//! the exact run's choice. A prefill over the whole sequence is the teacher-forced pass: every
//! position's logits come from the true tokens before it.
//!
//! The runs are by hand, on the real checkpoint, with everything named by the environment (no
//! file or model name lives here); see the `#[ignore]`d tests.

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

    /// `name:tokens=file+file;...`: each text's name, its token budget and the files it is
    /// made of, read in order.
    fn parse_texts(spec: &str) -> Vec<(String, usize, String)> {
        spec.split(';')
            .filter(|t| !t.trim().is_empty())
            .map(|t| {
                let (head, files) = t.split_once('=').expect("name:tokens=files");
                let (name, budget) = head.split_once(':').expect("name:tokens");
                let text: String = files
                    .split('+')
                    .map(|f| std::fs::read_to_string(f.trim()).expect("a readable text file"))
                    .collect::<Vec<_>>()
                    .join("\n");
                (
                    name.to_string(),
                    budget.parse().expect("a token budget"),
                    text,
                )
            })
            .collect()
    }

    fn tokens_of(
        source: &crate::tensor::quantized::gguf_source::SplitGguf,
        text: &str,
        budget: usize,
    ) -> Vec<u32> {
        use crate::inference::model::deepseek_v41::convert::TOKENIZER_KEY;
        use crate::tensor::quantized::gguf_source::GgufMeta;
        use std::str::FromStr;
        let md = source.metadata();
        let json = md[TOKENIZER_KEY].to_string().unwrap().clone();
        let tok = tokenizers::Tokenizer::from_str(&json).unwrap();
        let bos = md["tokenizer.ggml.bos_token_id"].to_u32().unwrap();
        let mut ids = vec![bos];
        ids.extend(tok.encode(text, false).unwrap().get_ids().iter().copied());
        ids.truncate(budget);
        ids
    }

    /// Settings are `exact`, `keep:<fraction>`, `drop:<mass>`, `miss_sub:<share>` and
    /// `miss_drop:<share>`.
    fn apply(
        model: &DeepseekV41Model,
        setting: &str,
        resident: &[std::collections::HashSet<usize>],
    ) {
        model.set_resident_override(None);
        let (kind, value) = setting.split_once(':').unwrap_or((setting, "0"));
        let v: f32 = value.parse().unwrap();
        match kind {
            "exact" => model.set_expert_truncation(0.0, 0.0, 0.0, false),
            "keep" => model.set_expert_truncation(v, 0.0, 0.0, false),
            "drop" => model.set_expert_truncation(0.0, v, 0.0, false),
            "miss_sub" | "miss_drop" => {
                model.set_resident_override(Some(resident));
                model.set_expert_truncation(0.0, 0.0, v, kind == "miss_drop");
            }
            other => panic!("unknown setting {other}"),
        }
    }

    use super::super::model::DeepseekV41Model;
    use super::super::DeepseekV41Config;

    fn load(
        path: &str,
    ) -> (
        crate::tensor::quantized::gguf_source::SplitGguf,
        DeepseekV41Config,
        DeepseekV41Model,
    ) {
        let source =
            crate::tensor::quantized::gguf_source::SplitGguf::open(std::path::Path::new(path))
                .unwrap();
        let cfg = DeepseekV41Config::from_meta(&source).unwrap();
        let model = DeepseekV41Model::load(&source, &cfg).unwrap();
        (source, cfg, model)
    }

    /// The real-model run: `LOKEN_DSV41_GGUF` the file, `LOKEN_QUALITY_TEXTS` the texts
    /// (`name:tokens=file+file;...`), `LOKEN_QUALITY_SETTINGS` the settings to score (`exact` is
    /// always run first), `LOKEN_QUALITY_RESIDENT` how many of each layer's prior-ranked experts a
    /// `miss_*` setting treats as resident. One line per text and setting on stderr.
    #[test]
    #[ignore = "needs the real checkpoint and texts"]
    fn score_settings_against_the_exact_run() {
        let path = std::env::var("LOKEN_DSV41_GGUF").expect("LOKEN_DSV41_GGUF");
        let texts =
            parse_texts(&std::env::var("LOKEN_QUALITY_TEXTS").expect("LOKEN_QUALITY_TEXTS"));
        let settings: Vec<String> = std::env::var("LOKEN_QUALITY_SETTINGS")
            .unwrap_or_default()
            .split(';')
            .filter(|s| !s.is_empty() && *s != "exact")
            .map(String::from)
            .collect();
        let resident_n: usize = std::env::var("LOKEN_QUALITY_RESIDENT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let (source, cfg, model) = load(&path);
        let resident: Vec<std::collections::HashSet<usize>> = cfg
            .hot_experts
            .iter()
            .map(|l| l.iter().copied().take(resident_n).collect())
            .collect();
        for (name, budget, text) in texts {
            let ids = tokens_of(&source, &text, budget);
            let next: Vec<Option<u32>> = (0..ids.len()).map(|i| ids.get(i + 1).copied()).collect();
            let rows = |model: &DeepseekV41Model| -> Vec<Vec<f32>> {
                model
                    .forward_prefill(&ids)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap()
            };
            apply(&model, "exact", &resident);
            let exact_rows = rows(&model);
            let mut reference = Reference {
                ids: Vec::new(),
                logprobs: Vec::new(),
            };
            for r in &exact_rows {
                let (i, v) = top(&log_softmax(r));
                reference.ids.push(i);
                reference.logprobs.push(v);
            }
            let s = score(&exact_rows, &reference, &next);
            eprintln!(
                "QUALITY text={name} tokens={} setting=exact top1=1.0000 kl=0.00000 lp_exact_choice={:.4} nll_true={:.4}",
                ids.len(),
                s.mean_logprob_of_exact_choice,
                s.mean_nll_of_true_next
            );
            drop(exact_rows);
            for setting in &settings {
                apply(&model, setting, &resident);
                let s = score(&rows(&model), &reference, &next);
                eprintln!(
                    "QUALITY text={name} tokens={} setting={setting} top1={:.4} kl={:.5} lp_exact_choice={:.4} nll_true={:.4}",
                    s.positions,
                    s.top1_agreement,
                    s.mean_kl,
                    s.mean_logprob_of_exact_choice,
                    s.mean_nll_of_true_next
                );
            }
        }
    }

    /// Rank each layer's experts by how often a calibration text routes to them and write that
    /// as a routing prior beside the model: `LOKEN_DSV41_GGUF`, `LOKEN_QUALITY_TEXTS` (one text),
    /// `LOKEN_QUALITY_PRIOR_OUT` the file to write.
    #[test]
    #[ignore = "needs the real checkpoint and a calibration text"]
    fn write_a_prior_from_a_text() {
        let path = std::env::var("LOKEN_DSV41_GGUF").expect("LOKEN_DSV41_GGUF");
        let out = std::env::var("LOKEN_QUALITY_PRIOR_OUT").expect("LOKEN_QUALITY_PRIOR_OUT");
        let texts =
            parse_texts(&std::env::var("LOKEN_QUALITY_TEXTS").expect("LOKEN_QUALITY_TEXTS"));
        let (_, budget, text) = &texts[0];
        let (source, _cfg, model) = load(&path);
        let ids = tokens_of(&source, text, *budget);
        model.forward_prefill(&ids).unwrap();
        let lists: Vec<Vec<u32>> = model
            .expert_usage()
            .iter()
            .map(|counts| {
                let mut order: Vec<u32> = (0..counts.len() as u32).collect();
                order.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]));
                order
            })
            .collect();
        crate::inference::model::deepseek_v41::sibling::write_prior(
            std::path::Path::new(&out),
            &lists,
        )
        .unwrap();
        eprintln!("prior written to {out}: {} layers", lists.len());
    }
}
