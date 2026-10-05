//! Choosing how much exactness to give up for speed, from a quality level. The level is turned
//! into a budget: the largest mean divergence (KL, in nats) accepted between the model's
//! distribution and the exact one, derived from the model's own numerical noise (see
//! [`noise`]); a zero budget keeps routing exact. A short bundled text is run teacher-forced once exactly and
//! once under each candidate setting, each scored by the teacher-forced harness; the candidate that
//! saves the most disk reads while staying within the budget is chosen. The measurements are cached
//! per model and plan, so they are taken once and a different budget reuses them.
//!
//! Candidates are derived from the number of experts a token routes to, `K`: keeping `K-1` and
//! `K-2` of them, and replacing a light expert that sits in no tier when its weight share is
//! under `0.5/K`, `0.75/K` or `1/K` (the weight share of an average slot is `1/K`).

use super::model::DeepseekV41Model;
use super::moe::Truncation;
use super::quality::{log_softmax, score, top, Reference, TOP};
use crate::inference::engine::llm_engine::QualityLevel;
use crate::tensor::Result;
use serde::{Deserialize, Serialize};

/// What `fast` accepts, as a multiple of the model's own numerical noise: the divergence between
/// two exact runs that differ only in numerics the engine does not control. `balanced` accepts
/// one times the noise.
pub const FAST_NOISE_MULTIPLE: f64 = 3.0;

/// The budget a level accepts for a model whose exact runs diverge by `noise` (mean KL, nats);
/// `None` when the level asks for no loss, or when the noise is unknown and nothing can be said
/// to stay within it.
pub fn budget_for(level: QualityLevel, noise: Option<f64>) -> Option<f64> {
    let noise = noise.filter(|n| n.is_finite() && *n >= 0.0)?;
    match level {
        QualityLevel::Exact => None,
        QualityLevel::Balanced => Some(noise),
        QualityLevel::Fast => Some(noise * FAST_NOISE_MULTIPLE),
    }
}

/// The model's own numerical noise: the mean KL of the exact run on `other` against the exact run
/// on `run` (the card's kernels against the host's, the pair the engine cannot choose between).
pub fn noise(
    model: &DeepseekV41Model,
    ids: &[u32],
    run: &dyn Fn(&DeepseekV41Model, &[u32]) -> Result<Vec<Vec<f32>>>,
    other: &dyn Fn(&DeepseekV41Model, &[u32]) -> Result<Vec<Vec<f32>>>,
) -> Result<f64> {
    model.set_resident_override(None);
    model.set_truncation(Truncation::default());
    let first = run(model, ids)?;
    let second = other(model, ids)?;
    let mut reference = Reference {
        ids: Vec::new(),
        logprobs: Vec::new(),
    };
    for row in &first {
        let (i, v) = top(&log_softmax(row));
        reference.ids.push(i);
        reference.logprobs.push(v);
    }
    let next: Vec<Option<u32>> = (0..ids.len()).map(|i| ids.get(i + 1).copied()).collect();
    Ok(score(&second, &reference, &next).mean_kl)
}

fn noise_file() -> Option<std::path::PathBuf> {
    Some(dirs::cache_dir()?.join("loken").join("quality-noise.json"))
}

/// The cached noise for `key`, if one was written.
pub fn cached_noise(key: &str) -> Option<f64> {
    let text = std::fs::read_to_string(noise_file()?).ok()?;
    let mut all: std::collections::HashMap<String, f64> = serde_json::from_str(&text).ok()?;
    all.remove(key)
}

/// Keep `noise` under `key`, beside the figures of other models and plans.
pub fn store_noise(key: &str, noise: f64) {
    let Some(path) = noise_file() else { return };
    let mut all: std::collections::HashMap<String, f64> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    all.insert(key.to_string(), noise);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(text) = serde_json::to_string_pretty(&all) {
        let _ = std::fs::write(path, text);
    }
}

/// Apply `level`: the noise of this model and plan from the cache, else measured against `other`
/// (and cached), turned into a budget and spent by [`apply_budget`]. Without a second device kind
/// to compare (`other` is `None`) the noise cannot be measured and routing stays exact, with the
/// reason logged.
pub fn apply_level(
    model: &DeepseekV41Model,
    level: QualityLevel,
    key: &str,
    ids: &[u32],
    resident: &[std::collections::HashSet<usize>],
    run: &dyn Fn(&DeepseekV41Model, &[u32]) -> Result<Vec<Vec<f32>>>,
    other: Option<&dyn Fn(&DeepseekV41Model, &[u32]) -> Result<Vec<Vec<f32>>>>,
) -> Result<Option<Measured>> {
    if level == QualityLevel::Exact {
        return Ok(None);
    }
    let measured = match (cached_noise(key), other) {
        (Some(n), _) => Some(n),
        (None, Some(other)) => {
            let n = noise(model, ids, run, other)?;
            store_noise(key, n);
            Some(n)
        }
        (None, None) => None,
    };
    let Some(budget) = budget_for(level, measured) else {
        tracing::info!(
            "quality {level:?}: the model's numerical noise cannot be measured here (no card and \
             host to compare); routing stays exact"
        );
        return Ok(None);
    };
    tracing::info!(
        "quality {level:?}: the model's own run-to-run noise is {:.4} nats; budget {budget:.4}",
        measured.unwrap_or(0.0)
    );
    apply_budget(model, budget, key, ids, resident, run)
}

/// The text the calibration runs, bundled with the engine: some code and some prose, long enough
/// to route a few hundred tokens and short enough to run in the time a model takes to load.
pub const CALIBRATION_TEXT: &str = include_str!("calibration_text.txt");

/// The bundled text as token ids, with the file's own tokenizer and start token; `None` when the
/// file carries no tokenizer.
pub fn calibration_ids(
    source: &impl crate::tensor::quantized::gguf_source::GgufMeta,
) -> Option<Vec<u32>> {
    use std::str::FromStr;
    let md = source.metadata();
    let json = md
        .get(super::convert::TOKENIZER_KEY)?
        .to_string()
        .ok()?
        .clone();
    let tok = tokenizers::Tokenizer::from_str(&json).ok()?;
    let bos = md.get("tokenizer.ggml.bos_token_id")?.to_u32().ok()?;
    let mut ids = vec![bos];
    ids.extend(
        tok.encode(CALIBRATION_TEXT, false)
            .ok()?
            .get_ids()
            .iter()
            .copied(),
    );
    Some(ids)
}

/// One candidate setting and what running the calibration text under it measured.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Measured {
    pub keep: usize,
    pub miss_below: f32,
    /// Mean KL(exact || setting) over the recorded top-20, nats.
    pub kl: f64,
    pub top1: f64,
    /// Disk reads saved per token: slots in no tier that the setting removed or replaced.
    pub saved_per_token: f64,
}

impl Measured {
    pub fn truncation(&self) -> Truncation {
        Truncation {
            keep: self.keep,
            miss_below: self.miss_below,
            ..Default::default()
        }
    }
}

/// The candidate settings for a model whose tokens route to `k` experts.
pub fn candidates(k: usize) -> Vec<Truncation> {
    let mut out = Vec::new();
    for keep in [k.saturating_sub(1), k.saturating_sub(2)] {
        if keep >= 1 && keep < k {
            out.push(Truncation {
                keep,
                ..Default::default()
            });
        }
    }
    for share in [0.5f32, 0.75, 1.0] {
        out.push(Truncation {
            miss_below: share / k.max(1) as f32,
            ..Default::default()
        });
    }
    out
}

/// The candidate with the most disk reads saved among those within `budget`, if any.
pub fn pick(table: &[Measured], budget: f64) -> Option<&Measured> {
    table
        .iter()
        .filter(|m| m.kl <= budget && m.saved_per_token > 0.0)
        .max_by(|a, b| a.saved_per_token.total_cmp(&b.saved_per_token))
}

/// Run the calibration text exactly and under each candidate, returning the table. `run` gives the
/// logits of every position of a token sequence, on whatever the model runs on.
pub fn measure(
    model: &DeepseekV41Model,
    ids: &[u32],
    resident: &[std::collections::HashSet<usize>],
    run: &dyn Fn(&DeepseekV41Model, &[u32]) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<Measured>> {
    model.set_resident_override(None);
    model.set_truncation(Truncation::default());
    let exact = run(model, ids)?;
    let mut reference = Reference {
        ids: Vec::new(),
        logprobs: Vec::new(),
    };
    for row in &exact {
        let (i, v) = top(&log_softmax(row));
        reference.ids.push(i);
        reference.logprobs.push(v);
    }
    let next: Vec<Option<u32>> = (0..ids.len()).map(|i| ids.get(i + 1).copied()).collect();
    let mut table = Vec::new();
    for t in candidates(model.n_activated()) {
        // The experts a miss setting treats as resident are the plan's: the ones on the cards
        // and the ones the kept tier will hold, as there is no tier to fill during a prefill.
        model.set_resident_override(Some(resident));
        model.set_truncation(t);
        let _ = model.take_saved();
        let rows = run(model, ids)?;
        let saved = model.take_saved();
        let s = score(&rows, &reference, &next);
        table.push(Measured {
            keep: t.keep,
            miss_below: t.miss_below,
            kl: s.mean_kl,
            top1: s.top1_agreement,
            saved_per_token: saved as f64 / ids.len().max(1) as f64,
        });
    }
    model.set_resident_override(None);
    model.set_truncation(Truncation::default());
    let _ = TOP;
    Ok(table)
}

fn cache_file() -> Option<std::path::PathBuf> {
    Some(
        dirs::cache_dir()?
            .join("loken")
            .join("quality-calibration.json"),
    )
}

/// The cached table for `key`, if one was written.
pub fn cached(key: &str) -> Option<Vec<Measured>> {
    let text = std::fs::read_to_string(cache_file()?).ok()?;
    let mut all: std::collections::HashMap<String, Vec<Measured>> =
        serde_json::from_str(&text).ok()?;
    all.remove(key)
}

/// Keep `table` under `key`, beside the tables of other models and plans.
pub fn store(key: &str, table: &[Measured]) {
    let Some(path) = cache_file() else { return };
    let mut all: std::collections::HashMap<String, Vec<Measured>> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    all.insert(key.to_string(), table.to_vec());
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(text) = serde_json::to_string_pretty(&all) {
        let _ = std::fs::write(path, text);
    }
}

/// Choose and apply the setting for `budget`: from the cached table of `key`, else by measuring
/// (and caching). Logs the setting and what it costs. A zero budget changes nothing.
pub fn apply_budget(
    model: &DeepseekV41Model,
    budget: f64,
    key: &str,
    ids: &[u32],
    resident: &[std::collections::HashSet<usize>],
    run: &dyn Fn(&DeepseekV41Model, &[u32]) -> Result<Vec<Vec<f32>>>,
) -> Result<Option<Measured>> {
    if budget <= 0.0 {
        return Ok(None);
    }
    let table = match cached(key) {
        Some(t) => {
            tracing::info!(
                "quality calibration: table for this model and plan read from the cache"
            );
            t
        }
        None => {
            let started = std::time::Instant::now();
            let t = measure(model, ids, resident, run)?;
            tracing::info!(
                "quality calibration: {} candidates measured on {} tokens in {:.1}s",
                t.len(),
                ids.len(),
                started.elapsed().as_secs_f32()
            );
            store(key, &t);
            t
        }
    };
    let chosen = pick(&table, budget).cloned();
    match &chosen {
        Some(m) => {
            tracing::info!(
                "quality budget {budget}: keep {} experts, replace a missed expert under share {:.3}; measured KL {:.4}, top-1 {:.3}, {:.2} disk reads saved per token",
                if m.keep == 0 { model.n_activated() } else { m.keep },
                m.miss_below,
                m.kl,
                m.top1,
                m.saved_per_token
            );
            model.set_truncation(m.truncation());
        }
        None => tracing::info!("quality budget {budget}: no candidate fits; routing stays exact"),
    }
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_is_a_multiple_of_the_models_noise() {
        assert_eq!(budget_for(QualityLevel::Exact, Some(0.02)), None);
        assert_eq!(budget_for(QualityLevel::Balanced, Some(0.02)), Some(0.02));
        assert_eq!(
            budget_for(QualityLevel::Fast, Some(0.02)),
            Some(0.02 * FAST_NOISE_MULTIPLE)
        );
        assert_eq!(budget_for(QualityLevel::Balanced, None), None);
        assert_eq!(budget_for(QualityLevel::Fast, Some(f64::NAN)), None);
    }

    fn m(keep: usize, miss: f32, kl: f64, saved: f64) -> Measured {
        Measured {
            keep,
            miss_below: miss,
            kl,
            top1: 0.9,
            saved_per_token: saved,
        }
    }

    /// The candidates follow the expert count, and the pick is the biggest saving within budget.
    #[test]
    fn candidates_follow_the_expert_count_and_the_pick_follows_the_budget() {
        let c = candidates(6);
        assert_eq!(
            c.iter()
                .filter(|t| t.keep > 0)
                .map(|t| t.keep)
                .collect::<Vec<_>>(),
            [5, 4]
        );
        assert!((c.last().unwrap().miss_below - 1.0 / 6.0).abs() < 1e-6);
        let table = vec![
            m(5, 0.0, 0.015, 4.0),
            m(4, 0.0, 0.035, 8.0),
            m(0, 0.12, 0.016, 9.0),
        ];
        assert!(pick(&table, 0.0).is_none(), "a zero budget fits nothing");
        assert_eq!(pick(&table, 0.02).unwrap().miss_below, 0.12);
        assert_eq!(pick(&table, 0.04).unwrap().saved_per_token, 9.0);
        assert!(pick(&table, 0.01).is_none());
    }

    /// A table survives the cache.
    #[test]
    fn the_table_round_trips() {
        let t = vec![m(5, 0.0, 0.015, 4.0)];
        let text = serde_json::to_string(&t).unwrap();
        let back: Vec<Measured> = serde_json::from_str(&text).unwrap();
        assert_eq!(back, t);
    }
}
