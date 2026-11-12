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
//! Its divergence on each half of the text must also stay within the budget.
//! The measurements are cached per model and plan, so a different budget reuses them.
//!
//! A load that finds no measure in the cache routes exactly and leaves a [`Pending`]
//! calibration.
//! The engine runs it between requests, 1 run at a time.
//! Each run keeps what it measured in the cache, so a later load resumes where it stopped.
//! The truncation is set when the last run ends.
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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// The larger of the mean KLs of the 2 halves of `rows` against `reference`.
fn worst_half(rows: &[Vec<f32>], ids: &[u32], reference: &Dump) -> Result<f64> {
    let (flat, vocab) = flat_rows(rows, ids)?;
    let half = ids.len() / 2;
    let mut worst = 0f64;
    for (first, last) in [(0, half), (half, ids.len())] {
        let mut acc = Accum::default();
        score_rows(
            &flat[first * vocab..last * vocab],
            vocab,
            first,
            ids,
            Some(reference),
            &mut acc,
            &mut dump_of(ids),
        );
        worst = worst.max(acc.kl().unwrap_or(0.0));
    }
    Ok(worst)
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
    /// The larger of the mean KLs of the 2 halves of the text, `None` when not measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kl_worst_half: Option<f64>,
}

impl Measured {
    /// Whether the setting saves reads, and its divergence over the text and over each half
    /// of it stays within `budget`.
    pub fn fits(&self, budget: f64) -> bool {
        self.kl <= budget
            && self.kl_worst_half.is_none_or(|w| w <= budget)
            && self.saved_per_token > 0.0
    }

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

/// [`pick`] among the candidates whose divergence on each half of the text also stays within
/// `budget`.
///
/// The half that did not decide a candidate checks it.
pub fn pick_held(table: &[Measured], budget: f64) -> Option<&Measured> {
    table
        .iter()
        .filter(|m| m.fits(budget))
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
    let all = candidates(model.routed_per_token());
    measure_these(model, ids, resident, run, &all)
}

/// [`measure`] under the truncations `these`.
pub fn measure_these(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    resident: &[HashSet<usize>],
    run: &mut Run<'_>,
    these: &[Truncation],
) -> Result<Vec<Measured>> {
    model.set_resident_override(None);
    model.set_truncation(Truncation::default());
    let measured = measure_each(model, ids, resident, run, these);
    model.set_resident_override(None);
    model.set_truncation(Truncation::default());
    measured
}

/// The table of [`measure_these`], each truncation left set on the model after its run.
fn measure_each(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    resident: &[HashSet<usize>],
    run: &mut Run<'_>,
    these: &[Truncation],
) -> Result<Vec<Measured>> {
    let exact = run(model, ids)?;
    let reference = reference(&exact, ids)?;
    these
        .iter()
        .map(|&t| measure_one(model, ids, resident, run, t, &reference))
        .collect()
}

/// Runs `ids` under the truncation `t`, left set on the model, and scores it against
/// `reference`.
fn measure_one(
    model: &mut GenericHeteroTransformer,
    ids: &[u32],
    resident: &[HashSet<usize>],
    run: &mut Run<'_>,
    t: Truncation,
    reference: &Dump,
) -> Result<Measured> {
    model.set_resident_override(Some(resident));
    model.set_truncation(t);
    let _ = model.take_saved();
    let rows = run(model, ids)?;
    let saved = model.take_saved();
    let scores = against(&rows, ids, reference)?;
    Ok(Measured {
        keep: t.keep,
        miss_below: t.miss_below,
        kl: scores.kl().unwrap_or(0.0),
        top1: scores.top1().unwrap_or(0.0),
        saved_per_token: saved as f64 / ids.len().max(1) as f64,
        kl_worst_half: Some(worst_half(&rows, ids, reference)?),
    })
}

/// The directory the calibrations of this user are kept in.
pub fn cache_dir() -> Option<PathBuf> {
    Some(dirs::cache_dir()?.join("loken"))
}

/// The file the noise of every model and plan is kept in, under `dir`.
fn noise_file(dir: Option<&Path>) -> Option<PathBuf> {
    Some(dir?.join("quality-noise.json"))
}

/// The file the table of every model and plan is kept in, under `dir`.
fn table_file(dir: Option<&Path>) -> Option<PathBuf> {
    Some(dir?.join("quality-calibration.json"))
}

/// The file the progress of every unfinished calibration is kept in, under `dir`.
fn steps_file(dir: Option<&Path>) -> Option<PathBuf> {
    Some(dir?.join("quality-calibration-steps.json"))
}

/// The file the exact run of the unfinished calibration `key` is kept in, under `dir`.
fn reference_file(dir: Option<&Path>, key: &str) -> Option<PathBuf> {
    use sha2::{Digest, Sha256};
    let digest: String = Sha256::digest(key.as_bytes())[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Some(
        dir?.join("quality-calibration-steps")
            .join(format!("{digest}.lkq")),
    )
}

/// The value kept under `key` in the map of `path`.
fn read_kept<T: serde::de::DeserializeOwned>(path: Option<PathBuf>, key: &str) -> Option<T> {
    let text = std::fs::read_to_string(path?).ok()?;
    let mut all: HashMap<String, T> = serde_json::from_str(&text).ok()?;
    all.remove(key)
}

/// Keeps `value` under `key` in the map of `path`, beside the values of other keys.
///
/// `None` removes `key`.
fn keep<T: Serialize + serde::de::DeserializeOwned>(
    path: Option<PathBuf>,
    key: &str,
    value: Option<T>,
) {
    let Some(path) = path else { return };
    let mut all: HashMap<String, T> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    match value {
        Some(value) => {
            all.insert(key.to_string(), value);
        }
        None => {
            all.remove(key);
        }
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(text) = serde_json::to_string_pretty(&all) {
        let _ = std::fs::write(path, text);
    }
}

/// Sets on `model` the candidate of `table` that `budget` allows, and returns it.
///
/// Routing stays exact when none fits.
fn install(
    model: &mut GenericHeteroTransformer,
    budget: f64,
    table: &[Measured],
) -> Option<Measured> {
    let chosen = pick_held(table, budget).cloned();
    match &chosen {
        Some(m) => {
            tracing::info!(
                "quality budget {budget}: keep {} experts, replace a missed expert under share {:.3}; measured KL {:.4}, {:.4} on the worse half, top-1 {:.3}, {:.2} expert reads saved per token",
                if m.keep == 0 { model.routed_per_token() } else { m.keep },
                m.miss_below,
                m.kl,
                m.kl_worst_half.unwrap_or(m.kl),
                m.top1,
                m.saved_per_token
            );
            model.set_truncation(m.truncation());
        }
        None => tracing::info!("quality budget {budget}: no candidate fits; routing stays exact"),
    }
    chosen
}

/// [`install`] under the budget `level` gives a model whose noise is `noise`.
fn install_level(
    model: &mut GenericHeteroTransformer,
    level: QualityLevel,
    noise: Option<f64>,
    table: &[Measured],
) -> Option<Measured> {
    let budget = budget_for(level, noise).filter(|b| *b > 0.0)?;
    tracing::info!(
        "quality {level:?}: the model's own run-to-run noise is {:.4} nats; budget {budget:.4}",
        noise.unwrap_or(0.0)
    );
    install(model, budget, table)
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
    let dir = cache_dir();
    let table = match read_kept::<Vec<Measured>>(table_file(dir.as_deref()), key) {
        Some(t) => t,
        None => {
            let t = measure(model, ids, resident, run)?;
            keep(table_file(dir.as_deref()), key, Some(t.clone()));
            t
        }
    };
    Ok(install(model, budget, &table))
}

/// The ids the loads of this process give their calibrations.
static PENDING_IDS: AtomicU64 = AtomicU64::new(0);

/// A routing calibration a load left to run between requests, 1 run at a time.
///
/// The model routes exactly until its last run ends.
/// Each run keeps what it measured under the cache directory, so a later load resumes it.
#[derive(Clone, Debug)]
pub struct Pending {
    level: QualityLevel,
    key: String,
    ids: Vec<u32>,
    id: u64,
}

/// What an unfinished calibration keeps: the experts it counts in memory and the candidates
/// measured so far.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
struct Progress {
    resident: Vec<Vec<usize>>,
    measured: Vec<Measured>,
}

/// What a step of a [`Pending`] calibration did.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// A run measured and kept, and more remain.
    Measured,
    /// The run gave way to a request and kept nothing.
    Yielded,
    /// The last run ended, and the candidate it allows is set, if any.
    Done(Option<Measured>),
}

/// Sets the truncation `level` allows on a loaded model, from a measure on `ids` the cache
/// holds.
///
/// Without such a measure, the model routes exactly and keeps a [`Pending`] calibration.
///
/// The experts in memory are those the cards and the host tier keep when the load ends.
///
/// `identity` names the model file.
/// The table and the noise are kept under it with the digest of the tokens and the count of
/// experts in memory.
/// A load reuses the measure of the largest count not above its own, see [`covering`].
/// An unfinished calibration resumes the same way, with the experts it counted.
///
/// The noise is the prefill of `ids` on the host against the same tokens decoded 1 at a time.
/// A decoded row runs through the card's kernels where a card keeps the mixers.
/// Without such a card the 2 runs share the host's kernels, and routing stays exact.
/// A model without routed experts is left as loaded.
pub fn calibrate_loaded(
    model: &mut GenericHeteroTransformer,
    level: QualityLevel,
    identity: &str,
    ids: &[u32],
) -> Result<Option<Measured>> {
    let second = model.mixes_on_cards();
    let resident = model.held_experts();
    let dir = cache_dir();
    prepare(
        model,
        level,
        identity,
        ids,
        &resident,
        dir.as_deref(),
        second,
    )
}

/// [`calibrate_loaded`] with `resident` the experts in memory, and the measures kept under
/// `dir`.
///
/// `second` says whether a second run can measure the noise.
pub fn prepare(
    model: &mut GenericHeteroTransformer,
    level: QualityLevel,
    identity: &str,
    ids: &[u32],
    resident: &[HashSet<usize>],
    dir: Option<&Path>,
    second: bool,
) -> Result<Option<Measured>> {
    model.set_routing_pending(None);
    if level == QualityLevel::Exact || model.routed_per_token() == 0 {
        return Ok(None);
    }
    let held_now = resident;
    let held = held_now.iter().map(HashSet::len).sum::<usize>();
    let base = format!("{identity}|text:{}", text_digest(ids));
    let noises = kept_keys(noise_file(dir));
    let complete = kept_keys(table_file(dir))
        .into_iter()
        .filter(|k| noises.contains(k));
    if let Some((measured, key)) = covering(complete, &base, held) {
        tracing::info!(
            "quality calibration: {held} routed experts in memory, covered by the measure of {measured}"
        );
        let noise = read_kept::<f64>(noise_file(dir), &key);
        let table = read_kept::<Vec<Measured>>(table_file(dir), &key).unwrap_or_default();
        tracing::info!("quality calibration: table for this model and plan read from the cache");
        return Ok(install_level(model, level, noise, &table));
    }
    let resumed = covering(kept_keys(steps_file(dir)).into_iter(), &base, held)
        .and_then(|(_, key)| Some((read_kept::<Progress>(steps_file(dir), &key)?, key)));
    let (progress, key) = match resumed {
        Some((progress, key)) => {
            tracing::info!(
                "quality calibration: {held} routed experts in memory, resuming the calibration of {key} after {} candidates",
                progress.measured.len()
            );
            (progress, key)
        }
        None => {
            tracing::info!("quality calibration: {held} routed experts in memory");
            let resident = held_now
                .iter()
                .map(|set| {
                    let mut experts: Vec<usize> = set.iter().copied().collect();
                    experts.sort_unstable();
                    experts
                })
                .collect();
            let progress = Progress {
                resident,
                measured: Vec::new(),
            };
            (progress, format!("{base}|held:{held}"))
        }
    };
    if !second && read_kept::<f64>(noise_file(dir), &key).is_none() {
        tracing::info!(
            "quality {level:?}: the model's numerical noise cannot be measured here (no second \
             run to compare); routing stays exact"
        );
        return Ok(None);
    }
    keep(steps_file(dir), &key, Some(progress));
    tracing::info!(
        "quality {level:?}: routing stays exact until the calibration ends between requests"
    );
    model.set_routing_pending(Some(Pending {
        level,
        key,
        ids: ids.to_vec(),
        id: PENDING_IDS.fetch_add(1, Ordering::Relaxed),
    }));
    Ok(None)
}

impl Pending {
    /// Which load left this calibration.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Runs the next run of the calibration on `model`, and keeps what it measured under
    /// `dir`.
    ///
    /// The runs are, in order: the exact prefill, the second run of the noise, then each
    /// candidate.
    /// `run` prefills the text, and `other` is the second run of the noise.
    /// `other` gives way to a request by failing while `yielded` says so.
    /// The model routes exactly when a step returns, but for the candidate the last one sets.
    pub fn step(
        &self,
        model: &mut GenericHeteroTransformer,
        dir: Option<&Path>,
        run: &mut Run<'_>,
        other: &mut Run<'_>,
        yielded: &dyn Fn() -> bool,
    ) -> Result<Step> {
        model.set_resident_override(None);
        model.set_truncation(Truncation::default());
        let stepped = self.next_run(model, dir, run, other, yielded);
        if !matches!(stepped, Ok(Step::Done(_))) {
            model.set_resident_override(None);
            model.set_truncation(Truncation::default());
        }
        stepped
    }

    /// The body of [`Self::step`], the model routing exactly when it starts.
    fn next_run(
        &self,
        model: &mut GenericHeteroTransformer,
        dir: Option<&Path>,
        run: &mut Run<'_>,
        other: &mut Run<'_>,
        yielded: &dyn Fn() -> bool,
    ) -> Result<Step> {
        let ids = &self.ids;
        let Some(mut progress) = read_kept::<Progress>(steps_file(dir), &self.key) else {
            crate::tensor::bail!("the calibration {} is no longer in the cache", self.key);
        };
        let kept_run = reference_file(dir, &self.key);
        let exact = kept_run
            .as_ref()
            .and_then(|p| std::fs::File::open(p).ok())
            .and_then(|f| Dump::read(std::io::BufReader::new(f)).ok())
            .filter(|d| d.tokens == *ids);
        let Some(exact) = exact else {
            let rows = run(model, ids)?;
            let dump = reference(&rows, ids)?;
            if let Some(path) = &kept_run {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let mut bytes = Vec::new();
                dump.write(&mut bytes)?;
                std::fs::write(path, bytes)?;
            }
            return Ok(Step::Measured);
        };
        let noise = match read_kept::<f64>(noise_file(dir), &self.key) {
            Some(n) => n,
            None => {
                let rows = match other(model, ids) {
                    Ok(rows) => rows,
                    Err(_) if yielded() => return Ok(Step::Yielded),
                    Err(e) => return Err(e),
                };
                let n = against(&rows, ids, &exact)?.kl().unwrap_or(0.0);
                keep(noise_file(dir), &self.key, Some(n));
                return Ok(Step::Measured);
            }
        };
        let all = candidates(model.routed_per_token());
        if let Some(&t) = all.get(progress.measured.len()) {
            let resident: Vec<HashSet<usize>> = progress
                .resident
                .iter()
                .map(|experts| experts.iter().copied().collect())
                .collect();
            progress
                .measured
                .push(measure_one(model, ids, &resident, run, t, &exact)?);
            keep(steps_file(dir), &self.key, Some(progress.clone()));
            if progress.measured.len() < all.len() {
                return Ok(Step::Measured);
            }
        }
        model.set_resident_override(None);
        model.set_truncation(Truncation::default());
        keep(table_file(dir), &self.key, Some(progress.measured.clone()));
        keep::<Progress>(steps_file(dir), &self.key, None);
        if let Some(path) = kept_run {
            let _ = std::fs::remove_file(path);
        }
        tracing::info!(
            "quality calibration: {} candidates measured on {} tokens between requests",
            progress.measured.len(),
            ids.len()
        );
        Ok(Step::Done(install_level(
            model,
            self.level,
            Some(noise),
            &progress.measured,
        )))
    }
}

/// [`decode_run`] that gives way between 2 tokens once `yielded` says so, by failing.
pub fn decode_run_yielding(
    m: &mut GenericHeteroTransformer,
    toks: &[u32],
    yielded: &dyn Fn() -> bool,
) -> Result<Vec<Vec<f32>>> {
    toks.iter()
        .enumerate()
        .map(|(pos, &t)| {
            if yielded() {
                crate::tensor::bail!("the calibration gave way to a request");
            }
            let x = crate::tensor::Tensor::from_vec(vec![t], (1, 1), &crate::tensor::Device::Cpu)?;
            m.forward(&x, pos)?.flatten_all()?.to_vec1::<f32>()
        })
        .collect()
}

/// The keys of the map of `path`, none when it cannot be read.
fn kept_keys(path: Option<std::path::PathBuf>) -> HashSet<String> {
    path.and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<HashMap<String, serde_json::Value>>(&t).ok())
        .map(|all| all.into_keys().collect())
        .unwrap_or_default()
}

/// Among `keys`, the one of `base` measured with the most experts in memory that is not
/// above `held`, with that count.
///
/// The cards and the host tier start from the hottest experts of each layer.
/// A measure with fewer in memory counted a part of what they keep now.
/// Every expert it counted is in memory now, so it gave away at least the slots the router
/// gives away now.
fn covering(
    keys: impl Iterator<Item = String>,
    base: &str,
    held: usize,
) -> Option<(usize, String)> {
    let prefix = format!("{base}|held:");
    keys.filter_map(|k| {
        let n = k.strip_prefix(&prefix)?.parse::<usize>().ok()?;
        (n <= held).then_some((n, k))
    })
    .max_by_key(|(n, _)| *n)
}

/// The first 16 hex digits of the SHA-256 of `ids`, which names the calibration text in a key.
fn text_digest(ids: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The logits of every position of `toks`, prefilled from position 0.
pub fn prefill_run(m: &mut GenericHeteroTransformer, toks: &[u32]) -> Result<Vec<Vec<f32>>> {
    let x = crate::tensor::Tensor::from_vec(
        toks.to_vec(),
        (1, toks.len()),
        &crate::tensor::Device::Cpu,
    )?;
    let all = m.forward_all(&x, 0)?.flatten_all()?.to_vec1::<f32>()?;
    let vocab = all.len() / toks.len().max(1);
    Ok(all.chunks(vocab.max(1)).map(<[f32]>::to_vec).collect())
}

/// The logits of every position of `toks`, decoded 1 at a time from position 0.
pub fn decode_run(m: &mut GenericHeteroTransformer, toks: &[u32]) -> Result<Vec<Vec<f32>>> {
    decode_run_yielding(m, toks, &|| false)
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
            kl_worst_half: None,
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

    /// A candidate whose divergence on a half of the text passes the budget is not picked,
    /// whatever its mean over the text.
    #[test]
    fn a_candidate_must_fit_on_each_half_of_the_text() {
        let mut uneven = m(0, 0.12, 0.015, 9.0);
        uneven.kl_worst_half = Some(0.025);
        let mut even = m(5, 0.0, 0.016, 4.0);
        even.kl_worst_half = Some(0.019);
        let table = vec![uneven.clone(), even.clone()];
        assert_eq!(pick(&table, 0.02), Some(&uneven));
        assert_eq!(pick_held(&table, 0.02), Some(&even));
        assert_eq!(pick_held(&table, 0.018), None);
        assert_eq!(pick_held(&table, 0.03), Some(&uneven));
    }

    /// A load takes the measure of the most experts in memory not above the count it holds,
    /// under its own file and text only.
    #[test]
    fn a_load_takes_the_measure_its_cards_cover() {
        let keys = [
            "f|text:a|held:500",
            "f|text:a|held:528",
            "f|text:a|held:600",
            "f|text:b|held:520",
            "f|text:a|resident:5248",
            "g|text:a|held:100",
        ]
        .map(String::from);
        let at = |cards| covering(keys.iter().cloned(), "f|text:a", cards);
        assert_eq!(at(528), Some((528, "f|text:a|held:528".into())));
        assert_eq!(at(560), Some((528, "f|text:a|held:528".into())));
        assert_eq!(at(700), Some((600, "f|text:a|held:600".into())));
        assert_eq!(at(510), Some((500, "f|text:a|held:500".into())));
        assert_eq!(at(499), None);
    }

    /// A table survives the cache, and is written as the V4.1 calibration wrote its table.
    #[test]
    fn the_table_round_trips_and_reads_as_the_v41_table() {
        use crate::inference::generic_transformer::wide_sparse_frozen as frozen;
        let t = vec![m(5, 0.0, 0.015, 4.0)];
        let text = serde_json::to_string(&t).unwrap();
        let back: Vec<Measured> = serde_json::from_str(&text).unwrap();
        assert_eq!(back, t);
        frozen::expect("units", "calibration table json", &text);
    }

    /// The candidates and the budgets are the V4.1 calibration's for every expert count.
    #[test]
    fn the_candidates_and_budgets_are_the_v41_ones() {
        use crate::inference::generic_transformer::wide_sparse_frozen as frozen;
        for k in 0..10 {
            let ours: Vec<(usize, u32)> = candidates(k)
                .iter()
                .map(|t| (t.keep, t.miss_below.to_bits()))
                .collect();
            frozen::expect_debug("units", &format!("calibration candidates {k}"), &ours);
        }
        for level in [
            QualityLevel::Exact,
            QualityLevel::Balanced,
            QualityLevel::Fast,
        ] {
            for noise in [None, Some(0.0), Some(0.013), Some(f64::INFINITY)] {
                let key = format!("calibration budget {level:?} {noise:?}");
                frozen::expect_debug("units", &key, &budget_for(level, noise));
            }
        }
    }
}
