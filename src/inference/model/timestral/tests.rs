//! Parity with the reference library (1.0.0) on a toy checkpoint it wrote: the embedding, every
//! layer and the block's quantiles on the inputs it recorded, then the whole forecast.

use super::*;
use crate::tensor::safetensors_io::MmapedSafetensors;
use std::path::{Path, PathBuf};

fn toy() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/timestral/toy")
}

pub(crate) struct Vectors(MmapedSafetensors);

impl Vectors {
    pub(crate) fn open(file: &Path) -> Self {
        Self(unsafe { MmapedSafetensors::new(file) }.unwrap())
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.0.load(name, &Device::Cpu).is_ok()
    }

    fn tensor(&self, name: &str) -> Tensor {
        self.0.load(name, &Device::Cpu).unwrap()
    }

    pub(crate) fn values(&self, name: &str) -> Vec<f32> {
        self.tensor(name)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    pub(crate) fn rows(&self, name: &str) -> Vec<Vec<f32>> {
        let t = self.tensor(name);
        let (r, l) = t.dims2().unwrap();
        let v = self.values(name);
        (0..r).map(|i| v[i * l..][..l].to_vec()).collect()
    }
}

/// The largest absolute difference relative to the largest reference magnitude.
fn relative(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "lengths differ");
    let diff = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    diff / want.iter().map(|v| v.abs()).fold(1e-6, f32::max)
}

/// Hidden states agree to this share of their largest magnitude.
const HIDDEN_TOLERANCE: f32 = 1e-4;
/// Quantiles agree to this share of their largest magnitude.
const QUANTILE_TOLERANCE: f32 = 1e-4;

fn network(dir: &Path) -> Forecaster {
    load_dir(dir, &Device::Cpu).unwrap()
}

fn packed(v: &Vectors, prefix: &str) -> (Packed, usize) {
    let inputs = v.tensor(&format!("{prefix}.block.inputs"));
    let (rows, len) = inputs.dims2().unwrap();
    let flag = |name: &str| -> Vec<bool> { v.values(name).iter().map(|x| *x != 0.0).collect() };
    let roles_name = format!("{prefix}.block.role_ids");
    let roles = v.has(&roles_name).then(|| {
        v.values(&roles_name)
            .iter()
            .map(|r| match *r as u32 {
                0 => Role::Target,
                1 => Role::Past,
                _ => Role::Known,
            })
            .collect()
    });
    let kf = format!("{prefix}.block.known_future");
    let known_future = v
        .has(&kf)
        .then(|| (v.values(&kf), flag(&format!("{kf}_mask"))));
    let h = v.values(&format!("{prefix}.block.horizon_patches"))[0] as usize;
    let p = Packed {
        rows,
        len,
        values: v.values(&format!("{prefix}.block.inputs")),
        observed: flag(&format!("{prefix}.block.observed")),
        group: v
            .values(&format!("{prefix}.block.group_ids"))
            .iter()
            .map(|g| *g as usize)
            .collect(),
        roles,
        known_future,
    };
    (p, h)
}

#[test]
fn every_layer_matches_the_reference() {
    let dir = toy();
    let f = network(&dir);
    let v = Vectors::open(&dir.join("vectors.safetensors"));
    for prefix in ["plain", "cov"] {
        let (p, h) = packed(&v, prefix);
        let mut trace = Trace::new();
        let q = f.network().predict_block(&p, h, Some(&mut trace)).unwrap();
        for (name, got) in &trace {
            let want = v.tensor(&format!("{prefix}.block.{name}"));
            assert_eq!(got.dims(), want.dims(), "{prefix} {name}");
            let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let err = relative(&got, &want.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            eprintln!("{prefix} {name}: {err:.1e}");
            assert!(err < HIDDEN_TOLERANCE, "{prefix} {name}: {err:e}");
        }
        let err = relative(&q, &v.values(&format!("{prefix}.block.quantiles")));
        assert!(
            err < QUANTILE_TOLERANCE,
            "{prefix} block quantiles: {err:e}"
        );
    }
}

#[test]
fn whole_forecasts_match_the_reference() {
    let dir = toy();
    let f = network(&dir);
    let v = Vectors::open(&dir.join("vectors.safetensors"));
    let horizon = 20;
    let codes = |name: &str| -> Vec<Vec<f64>> {
        v.rows(name)
            .into_iter()
            .map(|r| r.into_iter().map(f64::from).collect())
            .collect()
    };
    let requests = [
        (
            "plain",
            Request::univariate(v.values("plain.series"), horizon),
        ),
        (
            "cov",
            Request {
                series: v.values("cov.series"),
                horizon,
                past: v.rows("cov.past"),
                known: v.rows("cov.known"),
                categorical_past: codes("cov.cat_past"),
                categorical_known: codes("cov.cat_known"),
            },
        ),
    ];
    for (prefix, req) in requests {
        let got = f.forecast(&req).unwrap();
        let want = v.values(&format!("{prefix}.forecast"));
        let q = got.levels.len();
        let flat: Vec<f32> = (0..horizon)
            .flat_map(|t| (0..q).map(move |l| (t, l)))
            .map(|(t, l)| got.values[l][t])
            .collect();
        let err = relative(&flat, &want);
        eprintln!("{prefix} forecast: {err:.1e}");
        assert!(err < QUANTILE_TOLERANCE, "{prefix} forecast: {err:e}");
    }
}

#[test]
fn the_schedule_says_whether_known_covariates_reach_a_target() {
    let text = std::fs::read_to_string(toy().join("config.json")).unwrap();
    let mut cfg: Config = serde_json::from_str(&text).unwrap();
    assert!(cfg.supports_known_covariates());
    cfg.block_schedule = vec!["time".into(), "time".into()];
    assert!(!cfg.supports_known_covariates());
    cfg.block_schedule = vec!["variate".into(), "time".into()];
    assert!(!cfg.supports_known_covariates());
}

/// The fixed set the reference forecast with every published checkpoint.
pub(crate) fn fixed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/timestral")
}

/// The series of the fixed set.
pub(crate) const FIXED_SET: [&str; 8] = [
    "hourly",
    "weekly",
    "walk",
    "counts",
    "gappy",
    "air",
    "covariates",
    "long",
];

/// Quantiles of a downloaded checkpoint agree with the reference's to this share of their
/// largest magnitude on each series.
const CHECKPOINT_TOLERANCE: f32 = 5e-4;

/// Downloaded checkpoints of this network, found by what their files declare.
fn local_checkpoints() -> Vec<PathBuf> {
    let roots = [
        crate::config::Config::load_test()
            .get_hf_models_dir()
            .join("hub"),
        crate::config::Config::default_hf_models_dir(),
    ];
    let mut found: Vec<PathBuf> = roots
        .iter()
        .filter_map(|hub| std::fs::read_dir(hub).ok())
        .flatten()
        .flatten()
        .flat_map(|repo| {
            std::fs::read_dir(repo.path().join("snapshots"))
                .into_iter()
                .flatten()
        })
        .flatten()
        .filter_map(|snap| snap.path().canonicalize().ok())
        .filter(|dir| {
            crate::inference::media_facts::read(dir)
                .is_some_and(|f| f.kind == crate::inference::media_facts::MediaKind::Forecast)
                && read_config(dir).is_ok()
        })
        .collect();
    found.sort();
    found.dedup();
    found
}

/// The key the reference filed a checkpoint's forecasts under: its width and depth.
fn key_of(cfg: &Config) -> String {
    format!("d{}.l{}", cfg.d_model, cfg.block_schedule.len())
}

/// The request of a series of the fixed set, and its held-out continuation and period.
pub(crate) fn fixed_request(v: &Vectors, name: &str) -> (Request, Vec<f32>, usize) {
    let meta = v.values(&format!("{name}.meta"));
    let (horizon, period) = (meta[0] as usize, meta[1] as usize);
    let rows = |part: &str| {
        let n = format!("{name}.{part}");
        if v.has(&n) {
            v.rows(&n)
        } else {
            Vec::new()
        }
    };
    let req = Request {
        series: v.values(&format!("{name}.history")),
        horizon,
        past: rows("past"),
        known: rows("known"),
        ..Request::default()
    };
    (req, v.values(&format!("{name}.actual")), period)
}

/// The last season repeated, a missing point taken from a season before it.
pub(crate) fn seasonal_naive(history: &[f32], period: usize, horizon: usize) -> Vec<f32> {
    let mut filled = history.to_vec();
    for t in 0..filled.len() {
        if !filled[t].is_finite() {
            filled[t] = if t >= period { filled[t - period] } else { 0.0 };
        }
    }
    let season = &filled[filled.len() - period..];
    (0..horizon).map(|h| season[h % period]).collect()
}

/// The mean absolute step of the seasonal naive forecast over the history: the scale both
/// MASE and the scaled CRPS divide by.
pub(crate) fn naive_scale(history: &[f32], period: usize) -> f32 {
    let steps: Vec<f32> = (period..history.len())
        .map(|t| (history[t] - history[t - period]).abs())
        .filter(|d| d.is_finite())
        .collect();
    (steps.iter().sum::<f32>() / steps.len().max(1) as f32).max(f32::EPSILON)
}

/// The CRPS of a quantile forecast, as twice the mean pinball loss over its levels.
fn crps(levels: &[f32], values: &[Vec<f32>], actual: &[f32]) -> f32 {
    let mut sum = 0f32;
    for (l, q) in levels.iter().zip(values) {
        for (f, y) in q.iter().zip(actual) {
            let d = y - f;
            sum += 2.0 * if d >= 0.0 { l * d } else { (l - 1.0) * d };
        }
    }
    sum / (levels.len() * actual.len()) as f32
}

pub(crate) fn mae(f: &[f32], y: &[f32]) -> f32 {
    f.iter().zip(y).map(|(a, b)| (a - b).abs()).sum::<f32>() / y.len() as f32
}

#[test]
fn checkpoints_match_the_reference_and_beat_seasonal_naive() {
    let dirs = local_checkpoints();
    if dirs.is_empty() {
        eprintln!("no checkpoint of this network downloaded; skipped");
        return;
    }
    let v = Vectors::open(&fixed().join("fixed.safetensors"));
    for dir in dirs {
        let cfg = read_config(&dir).unwrap();
        let key = key_of(&cfg);
        if !v.has(&format!("{key}.hourly.quantiles")) {
            eprintln!("{key}: no reference forecasts recorded; skipped");
            continue;
        }
        let f = network(&dir);
        let (mut worst, mut ms) = (0f32, 0f64);
        let (mut mase, mut mase_naive, mut crps_model, mut crps_naive) = (0f32, 0f32, 0f32, 0f32);
        for name in FIXED_SET {
            let (req, actual, period) = fixed_request(&v, name);
            let t0 = std::time::Instant::now();
            let got = f.forecast(&req).unwrap();
            ms += t0.elapsed().as_secs_f64() * 1e3;
            let want = v.rows(&format!("{key}.{name}.quantiles"));
            let flat: Vec<f32> = (0..req.horizon)
                .flat_map(|t| got.values.iter().map(move |q| q[t]))
                .collect();
            let err = relative(&flat, &want.concat());
            assert!(err < CHECKPOINT_TOLERANCE, "{key} {name}: {err:e}");
            worst = worst.max(err);

            let scale = naive_scale(&req.series, period);
            let naive = seasonal_naive(&req.series, period, req.horizon);
            mase += mae(got.median(), &actual) / scale;
            mase_naive += mae(&naive, &actual) / scale;
            crps_model += crps(&got.levels, &got.values, &actual) / scale;
            crps_naive += mae(&naive, &actual) / scale;
        }
        let n = FIXED_SET.len() as f32;
        let (mase, mase_naive) = (mase / n, mase_naive / n);
        let (crps_model, crps_naive) = (crps_model / n, crps_naive / n);
        eprintln!(
            "{key}: worst {worst:.1e}, {ms:.0} ms, MASE {mase:.3} (seasonal naive \
             {mase_naive:.3}), scaled CRPS {crps_model:.3} (seasonal naive {crps_naive:.3})"
        );
        assert!(mase < mase_naive, "{key}: MASE {mase} >= {mase_naive}");
        assert!(
            crps_model < crps_naive,
            "{key}: CRPS {crps_model} >= {crps_naive}"
        );
    }
}
