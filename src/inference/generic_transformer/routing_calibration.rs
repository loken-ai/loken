//! How many routed experts a token keeps at each quality level, measured on the model.
//!
//! A level is turned into a budget: the largest mean divergence (KL, in nats) accepted between
//! the model's distribution and the exact one.
//! The budget is a multiple of the model's own numerical noise (see [`noise`]).
//! A zero budget keeps routing exact.
//!
//! A text is run teacher-forced once exactly and once under each candidate truncation.
//! The engine's quality instrument scores each run against the exact one.
//! The candidate that saves the most expert reads within the budget is chosen.
//! The measurements are cached per model and plan, so a different budget reuses them.
//!
//! The candidates follow the number of experts a token routes to, `K`:
//! - keeping `K-1` and `K-2` of them,
//! - replacing a light expert that is not in memory when its weight share is under `0.5/K`,
//!   `0.75/K` or `1/K`. The weight share of an average slot is `1/K`.

use super::moe_weights::Truncation;
use super::GenericHeteroTransformer;
use crate::inference::engine::llm_engine::quality::{score_rows, Accum, Dump};
use crate::inference::engine::llm_engine::QualityLevel;
use crate::tensor::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// What `fast` accepts, as a multiple of the model's own numerical noise.
///
/// `balanced` accepts 1 times the noise.
pub const FAST_NOISE_MULTIPLE: f64 = 3.0;

/// Shares of an average slot's weight under which a missed expert is replaced, 1 candidate
/// each.
const MISS_SHARES: [f32; 3] = [0.5, 0.75, 1.0];

/// The logits of every position of a token sequence, one row each, from the model as it is
/// set.
pub type Run<'a> = dyn FnMut(&mut GenericHeteroTransformer, &[u32]) -> Result<Vec<Vec<f32>>> + 'a;

/// The budget a level accepts for a model whose exact runs diverge by `noise` (mean KL, nats).
///
/// `None` when the level asks for no loss, or when the noise is unknown.
pub fn budget_for(level: QualityLevel, noise: Option<f64>) -> Option<f64> {
    let noise = noise.filter(|n| n.is_finite() && *n >= 0.0)?;
    match level {
        QualityLevel::Exact => None,
        QualityLevel::Balanced => Some(noise),
        QualityLevel::Fast => Some(noise * FAST_NOISE_MULTIPLE),
    }
}

/// `rows` as 1 row-major slice and its width.
///
/// Refused unless there is 1 row per token of `ids`, all of 1 width.
fn flat_rows(rows: &[Vec<f32>], ids: &[u32]) -> Result<(Vec<f32>, usize)> {
    let vocab = rows.first().map_or(0, Vec::len);
    if rows.len() != ids.len() || vocab == 0 || rows.iter().any(|r| r.len() != vocab) {
        crate::tensor::bail!(
            "a calibration run gave {} rows for {} tokens, not all of 1 width",
            rows.len(),
            ids.len()
        );
    }
    Ok((rows.concat(), vocab))
}

/// An empty dump of `ids`, which the instrument fills.
fn dump_of(ids: &[u32]) -> Dump {
    Dump {
        tokens: ids.to_vec(),
        ids: Vec::new(),
        lp: Vec::new(),
        true_lp: Vec::new(),
    }
}

/// The exact run's top-20 at every position of `ids`, which other runs are scored against.
fn reference(rows: &[Vec<f32>], ids: &[u32]) -> Result<Dump> {
    let (flat, vocab) = flat_rows(rows, ids)?;
    let mut dump = dump_of(ids);
    score_rows(&flat, vocab, 0, ids, None, &mut Accum::default(), &mut dump);
    Ok(dump)
}

/// The scores of `rows` against `reference`.
fn against(rows: &[Vec<f32>], ids: &[u32], reference: &Dump) -> Result<Accum> {
    let (flat, vocab) = flat_rows(rows, ids)?;
    let mut acc = Accum::default();
    score_rows(
        &flat,
        vocab,
        0,
        ids,
        Some(reference),
        &mut acc,
        &mut dump_of(ids),
    );
    Ok(acc)
}

/// The model's own numerical noise: the mean KL of the exact run `other` against the exact
/// run `run`.
///
/// The 2 runs differ only in numerics the engine does not choose between, as a card's
/// kernels and the host's.
pub fn noise(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    run: &mut Run<'_>,
    other: &mut Run<'_>,
) -> Result<f64> {
    model.set_resident_override(None);
    model.set_truncation(Truncation::default());
    let first = run(model, ids)?;
    let second = other(model, ids)?;
    let reference = reference(&first, ids)?;
    Ok(against(&second, ids, &reference)?.kl().unwrap_or(0.0))
}

/// One candidate truncation and what running the calibration text under it measured.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Measured {
    pub keep: usize,
    pub miss_below: f32,
    /// Mean KL(exact || setting) over the exact run's top-20, nats.
    pub kl: f64,
    pub top1: f64,
    /// Expert reads saved per token: slots of experts not in memory that the setting removed
    /// or replaced.
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

/// The candidate truncations for a model whose tokens route to `k` experts.
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
    for share in MISS_SHARES {
        out.push(Truncation {
            miss_below: share / k.max(1) as f32,
            ..Default::default()
        });
    }
    out
}

/// The candidate with the most reads saved among those within `budget`, if any.
pub fn pick(table: &[Measured], budget: f64) -> Option<&Measured> {
    table
        .iter()
        .filter(|m| m.kl <= budget && m.saved_per_token > 0.0)
        .max_by(|a, b| a.saved_per_token.total_cmp(&b.saved_per_token))
}

/// Runs `ids` exactly and under each candidate, and returns the table.
///
/// `resident` holds, per layer, the experts the plan keeps in memory.
/// The model routes exactly again when this returns, measured or not.
pub fn measure(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    resident: &[HashSet<usize>],
    run: &mut Run<'_>,
) -> Result<Vec<Measured>> {
    model.set_resident_override(None);
    model.set_truncation(Truncation::default());
    let measured = measure_each(model, ids, resident, run);
    model.set_resident_override(None);
    model.set_truncation(Truncation::default());
    measured
}

/// The table of [`measure`], each candidate left set on the model after its run.
fn measure_each(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    resident: &[HashSet<usize>],
    run: &mut Run<'_>,
) -> Result<Vec<Measured>> {
    let exact = run(model, ids)?;
    let reference = reference(&exact, ids)?;
    let mut table = Vec::new();
    for t in candidates(model.routed_per_token()) {
        model.set_resident_override(Some(resident));
        model.set_truncation(t);
        let _ = model.take_saved();
        let rows = run(model, ids)?;
        let saved = model.take_saved();
        let scores = against(&rows, ids, &reference)?;
        table.push(Measured {
            keep: t.keep,
            miss_below: t.miss_below,
            kl: scores.kl().unwrap_or(0.0),
            top1: scores.top1().unwrap_or(0.0),
            saved_per_token: saved as f64 / ids.len().max(1) as f64,
        });
    }
    Ok(table)
}

/// The file the noise of every model and plan is kept in.
fn noise_file() -> Option<std::path::PathBuf> {
    Some(dirs::cache_dir()?.join("loken").join("quality-noise.json"))
}

/// The file the table of every model and plan is kept in.
fn table_file() -> Option<std::path::PathBuf> {
    Some(
        dirs::cache_dir()?
            .join("loken")
            .join("quality-calibration.json"),
    )
}

/// The value kept under `key` in the map of `path`.
fn read_kept<T: serde::de::DeserializeOwned>(
    path: Option<std::path::PathBuf>,
    key: &str,
) -> Option<T> {
    let text = std::fs::read_to_string(path?).ok()?;
    let mut all: HashMap<String, T> = serde_json::from_str(&text).ok()?;
    all.remove(key)
}

/// Keeps `value` under `key` in the map of `path`, beside the values of other keys.
fn keep<T: Serialize + serde::de::DeserializeOwned>(
    path: Option<std::path::PathBuf>,
    key: &str,
    value: T,
) {
    let Some(path) = path else { return };
    let mut all: HashMap<String, T> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    all.insert(key.to_string(), value);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(text) = serde_json::to_string_pretty(&all) {
        let _ = std::fs::write(path, text);
    }
}

/// Chooses and applies the truncation for `budget`: from the cached table of `key`, else
/// measured and cached.
///
/// A zero budget changes nothing and runs nothing.
pub fn apply_budget(
    model: &mut GenericHeteroTransformer,
    budget: f64,
    key: &str,
    ids: &[u32],
    resident: &[HashSet<usize>],
    run: &mut Run<'_>,
) -> Result<Option<Measured>> {
    if budget <= 0.0 {
        return Ok(None);
    }
    let table = match read_kept::<Vec<Measured>>(table_file(), key) {
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
            keep(table_file(), key, t.clone());
            t
        }
    };
    let chosen = pick(&table, budget).cloned();
    match &chosen {
        Some(m) => {
            tracing::info!(
                "quality budget {budget}: keep {} experts, replace a missed expert under share {:.3}; measured KL {:.4}, top-1 {:.3}, {:.2} expert reads saved per token",
                if m.keep == 0 { model.routed_per_token() } else { m.keep },
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

/// Applies `level`: the noise of this model and plan from the cache, else measured against
/// `other` and cached, turned into a budget and spent by [`apply_budget`].
///
/// Without a second run to compare, `other` `None`, the noise is unknown and routing stays
/// exact.
pub fn apply_level(
    model: &mut GenericHeteroTransformer,
    level: QualityLevel,
    key: &str,
    ids: &[u32],
    resident: &[HashSet<usize>],
    run: &mut Run<'_>,
    other: Option<&mut Run<'_>>,
) -> Result<Option<Measured>> {
    if level == QualityLevel::Exact {
        return Ok(None);
    }
    let measured = match (read_kept::<f64>(noise_file(), key), other) {
        (Some(n), _) => Some(n),
        (None, Some(other)) => {
            let n = noise(model, ids, run, other)?;
            keep(noise_file(), key, n);
            Some(n)
        }
        (None, None) => None,
    };
    let Some(budget) = budget_for(level, measured) else {
        tracing::info!(
            "quality {level:?}: the model's numerical noise cannot be measured here (no second \
             run to compare); routing stays exact"
        );
        return Ok(None);
    };
    tracing::info!(
        "quality {level:?}: the model's own run-to-run noise is {:.4} nats; budget {budget:.4}",
        measured.unwrap_or(0.0)
    );
    apply_budget(model, budget, key, ids, resident, run)
}

/// Sets the truncation `level` allows on a loaded model, measured on `ids`.
///
/// `identity` names the model file, and the table is kept under it.
/// No expert is counted in memory: the layers read every routed expert from its mapping.
/// The noise needs a second run to compare with, which a model on 1 device kind does not have:
/// routing then stays exact.
/// A model without routed experts is left as loaded.
pub fn calibrate_loaded(
    model: &mut GenericHeteroTransformer,
    level: QualityLevel,
    identity: &str,
    ids: &[u32],
) -> Result<Option<Measured>> {
    if level == QualityLevel::Exact || model.routed_per_token() == 0 {
        return Ok(None);
    }
    let resident = vec![HashSet::new(); model.segment_layers()];
    let key = format!("{identity}|resident:0");
    let mut prefill = |m: &mut GenericHeteroTransformer, toks: &[u32]| -> Result<Vec<Vec<f32>>> {
        let x = crate::tensor::Tensor::from_vec(
            toks.to_vec(),
            (1, toks.len()),
            &crate::tensor::Device::Cpu,
        )?;
        let all = m.forward_all(&x, 0)?.flatten_all()?.to_vec1::<f32>()?;
        let vocab = all.len() / toks.len().max(1);
        Ok(all.chunks(vocab.max(1)).map(<[f32]>::to_vec).collect())
    };
    apply_level(model, level, &key, ids, &resident, &mut prefill, None)
}

/// The text the calibration runs: some code and some prose, long enough to route a few
/// hundred tokens.
pub const CALIBRATION_TEXT: &str = include_str!("calibration_text.txt");

/// [`CALIBRATION_TEXT`] as token ids of `tokenizer`, after `bos` when the model has one.
pub fn calibration_ids(tokenizer: &tokenizers::Tokenizer, bos: Option<u32>) -> Option<Vec<u32>> {
    let encoded = tokenizer.encode(CALIBRATION_TEXT, false).ok()?;
    Some(
        bos.into_iter()
            .chain(encoded.get_ids().iter().copied())
            .collect(),
    )
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

    /// The candidates follow the expert count, and the pick is the biggest saving within
    /// budget.
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

    /// A table survives the cache, and reads as the V4.1 calibration's table.
    #[test]
    fn the_table_round_trips_and_reads_as_the_v41_table() {
        use crate::inference::model::deepseek_v41::calibrate;
        let t = vec![m(5, 0.0, 0.015, 4.0)];
        let text = serde_json::to_string(&t).unwrap();
        let back: Vec<Measured> = serde_json::from_str(&text).unwrap();
        assert_eq!(back, t);
        let theirs: Vec<calibrate::Measured> = serde_json::from_str(&text).unwrap();
        assert_eq!(serde_json::to_string(&theirs).unwrap(), text);
    }

    /// The candidates and the budgets are the V4.1 calibration's for every expert count.
    #[test]
    fn the_candidates_and_budgets_are_the_v41_ones() {
        use crate::inference::model::deepseek_v41::calibrate;
        for k in 0..10 {
            let ours: Vec<(usize, u32)> = candidates(k)
                .iter()
                .map(|t| (t.keep, t.miss_below.to_bits()))
                .collect();
            let theirs: Vec<(usize, u32)> = calibrate::candidates(k)
                .iter()
                .map(|t| (t.keep, t.miss_below.to_bits()))
                .collect();
            assert_eq!(ours, theirs, "{k} experts");
        }
        for level in [
            QualityLevel::Exact,
            QualityLevel::Balanced,
            QualityLevel::Fast,
        ] {
            for noise in [None, Some(0.0), Some(0.013), Some(f64::INFINITY)] {
                assert_eq!(
                    budget_for(level, noise),
                    calibrate::budget_for(level, noise),
                    "{level:?} {noise:?}"
                );
            }
        }
    }
}
