//! The resident time series forecaster: loaded on first request through the placement ladder,
//! kept until another checkpoint is asked for.

use crate::inference::model::chronos2::{self, Forecast};
use crate::inference::model::timestral;
use crate::tensor::Device;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub use crate::inference::model::timestral::Request;

/// Bytes per element of the weights and activations once loaded.
const F32_BYTES: u64 = 4;

/// A loaded forecaster, of whichever network its configuration describes.
pub enum Forecaster {
    /// An encoder over one series, without covariates.
    Encoder(chronos2::Forecaster),
    /// A patched transformer across the series of a group, with covariates.
    Patched(timestral::Forecaster),
}

impl std::fmt::Debug for Forecaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Forecaster::Encoder(_) => "Forecaster::Encoder",
            Forecaster::Patched(_) => "Forecaster::Patched",
        })
    }
}

impl Forecaster {
    /// The quantile levels the head answers.
    pub fn levels(&self) -> &[f32] {
        match self {
            Forecaster::Encoder(m) => m.levels(),
            Forecaster::Patched(m) => m.levels(),
        }
    }

    /// Forecast what `req` asks.
    pub fn forecast(&self, req: &Request) -> anyhow::Result<Forecast> {
        match self {
            Forecaster::Encoder(m) => {
                if req.has_covariates() {
                    anyhow::bail!("this forecaster reads one series without covariates");
                }
                Ok(m.forecast(&req.series, req.horizon)?)
            }
            Forecaster::Patched(m) => Ok(m.forecast(req)?),
        }
    }
}

/// What a forecaster checkpoint's configuration bounds a request by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_horizon: usize,
    pub covariates: bool,
    pub known_covariates: bool,
}

/// The limits of the checkpoint in `dir`.
pub fn limits(dir: &Path) -> anyhow::Result<Limits> {
    if let Ok(cfg) = timestral::read_config(dir) {
        return Ok(Limits {
            max_horizon: cfg.max_horizon(),
            covariates: cfg.covariate_enabled,
            known_covariates: cfg.supports_known_covariates(),
        });
    }
    let cfg = chronos2::read_config(dir)?;
    Ok(Limits {
        max_horizon: cfg.max_horizon(),
        covariates: false,
        known_covariates: false,
    })
}

/// One forecaster at a time, keyed by its checkpoint directory.
#[derive(Default)]
pub struct ForecastEngine {
    resident: Mutex<Option<(PathBuf, Arc<Forecaster>)>>,
}

impl ForecastEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// The forecaster of the checkpoint in `dir`, loaded on the cards the ladder offers now.
    pub fn forecaster(&self, dir: &Path) -> anyhow::Result<Arc<Forecaster>> {
        let mut slot = self.resident.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((held, model)) = slot.as_ref() {
            if held == dir {
                return Ok(model.clone());
            }
        }
        *slot = None;
        let (need, reserve) = demand(dir)?;
        let cards = crate::inference::place::ladder::cards(reserve);
        let (model, device) = load_on(dir, need, &cards)?;
        tracing::info!("forecaster {} on {device:?}", dir.display());
        let model = Arc::new(model);
        *slot = Some((dir.to_path_buf(), model.clone()));
        Ok(model)
    }

    /// Forecast what `req` asks with the checkpoint in `dir`.
    pub fn forecast(&self, dir: &Path, req: &Request) -> anyhow::Result<Forecast> {
        self.forecaster(dir)?.forecast(req)
    }

    /// Drop the resident forecaster. Whether one was resident.
    pub fn unload(&self) -> bool {
        self.resident
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .is_some()
    }
}

/// What the checkpoint in `dir` holds resident, and what one forecast at its longest context
/// keeps free beside it.
pub fn demand(dir: &Path) -> anyhow::Result<(u64, u64)> {
    let files = chronos2::weight_files(dir)?;
    let (weights, _) =
        crate::inference::place::runtime_demand::safetensors_f32_split(&files, |_| true)
            .map_err(|e| anyhow::anyhow!("size forecaster weights: {e}"))?;
    // Rows one attention spans, its width, heads and feed-forward width, and how many rows
    // a forecast runs at once.
    let (tokens, dim, heads, ffn, batch) = match timestral::read_config(dir) {
        Ok(cfg) => {
            // The context's patches and a horizon block, for the series and its mirror.
            let tokens = cfg.max_tokens() / 2;
            (tokens, cfg.d_model, cfg.num_heads, cfg.mlp_hidden_dim, 2)
        }
        Err(_) => {
            let cfg = chronos2::read_config(dir)?;
            (cfg.max_rows(), cfg.d_model, cfg.num_heads, cfg.d_ff, 1)
        }
    };
    let reserve = crate::inference::place::runtime_demand::transformer_work_reserve(
        weights,
        tokens * batch,
        dim,
        heads,
        ffn,
    );
    // The scores of one attention over every row, per head.
    let rows = tokens as u64;
    let scores = rows * rows * heads as u64 * F32_BYTES * batch as u64;
    Ok((weights, reserve.saturating_add(scores)))
}

/// Build the forecaster in `dir` through the ladder, on the first of `cards` offering `need`
/// bytes, else on the host.
pub fn load_on(
    dir: &Path,
    need: u64,
    cards: &[(usize, u64)],
) -> anyhow::Result<(Forecaster, Device)> {
    let patched = timestral::read_config(dir).is_ok();
    crate::inference::place::ladder::load_whole_on(
        "Forecaster",
        need,
        need,
        cards,
        None,
        |device| {
            Ok(if patched {
                Forecaster::Patched(timestral::load_dir(dir, device)?)
            } else {
                Forecaster::Encoder(chronos2::load_dir(dir, device)?)
            })
        },
    )
}

/// Mean absolute error of `forecast` against `actual`, over their common length.
pub fn mae(forecast: &[f32], actual: &[f32]) -> f32 {
    let n = forecast.len().min(actual.len()).max(1);
    forecast
        .iter()
        .zip(actual)
        .map(|(f, a)| (f - a).abs())
        .sum::<f32>()
        / n as f32
}

/// The last observed value, repeated over `horizon`.
pub fn naive(series: &[f32], horizon: usize) -> Vec<f32> {
    let last = series
        .iter()
        .rev()
        .copied()
        .find(|v| v.is_finite())
        .unwrap_or(0.0);
    vec![last; horizon]
}

/// The last `period` values, repeated over `horizon`.
pub fn seasonal_naive(series: &[f32], period: usize, horizon: usize) -> Vec<f32> {
    let period = period.clamp(1, series.len().max(1));
    let season = &series[series.len().saturating_sub(period)..];
    (0..horizon).map(|h| season[h % season.len()]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A daily cycle of `period` steps on a slow trend, with deterministic noise.
    fn seasonal(len: usize, period: usize) -> Vec<f32> {
        let mut state: u32 = 0x9e37_79b9;
        (0..len)
            .map(|t| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                let phase = 2.0 * std::f32::consts::PI * t as f32 / period as f32;
                10.0 + 0.02 * t as f32 + 3.0 * phase.sin() + 1.5 * (2.0 * phase).cos() + noise
            })
            .collect()
    }

    #[test]
    fn baselines_repeat_what_they_should() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(naive(&s, 3), vec![5.0; 3]);
        assert_eq!(seasonal_naive(&s, 2, 5), vec![4.0, 5.0, 4.0, 5.0, 4.0]);
        assert_eq!(mae(&[1.0, 3.0], &[2.0, 1.0]), 1.5);
    }

    /// A downloaded forecaster, found by what its files declare.
    fn local_forecaster() -> Option<PathBuf> {
        let hub = crate::config::Config::default_hf_models_dir();
        std::fs::read_dir(&hub)
            .ok()?
            .flatten()
            .flat_map(|repo| {
                std::fs::read_dir(repo.path().join("snapshots"))
                    .into_iter()
                    .flatten()
            })
            .flatten()
            .map(|snap| snap.path())
            .find(|dir| {
                crate::inference::media_facts::read(dir)
                    .is_some_and(|f| f.kind == crate::inference::media_facts::MediaKind::Forecast)
            })
    }

    #[test]
    fn forecast_beats_naive_baselines_on_a_seasonal_series() {
        let Some(dir) = local_forecaster() else {
            eprintln!("no forecaster checkpoint downloaded; skipped");
            return;
        };
        let (period, horizon) = (24, 48);
        let full = seasonal(20 * period + horizon, period);
        let (history, actual) = full.split_at(full.len() - horizon);
        let (need, _) = demand(&dir).unwrap();
        // No card offered: the ladder builds it on the host.
        let (model, device) = load_on(&dir, need, &[]).unwrap();
        assert!(device.is_cpu());
        let f = model
            .forecast(&Request::univariate(history.to_vec(), horizon))
            .unwrap();
        assert_eq!(f.values.len(), f.levels.len());
        assert!(f.values.iter().all(|v| v.len() == horizon));
        // Quantiles are ordered at every step.
        for h in 0..horizon {
            let col: Vec<f32> = f.values.iter().map(|v| v[h]).collect();
            assert!(
                col.windows(2).all(|w| w[0] <= w[1] + 1e-3),
                "step {h}: {col:?}"
            );
        }
        let model_mae = mae(f.median(), actual);
        let naive_mae = mae(&naive(history, horizon), actual);
        let seasonal_mae = mae(&seasonal_naive(history, period, horizon), actual);
        eprintln!(
            "forecast MAE {model_mae:.4}, naive {naive_mae:.4}, seasonal naive {seasonal_mae:.4}"
        );
        assert!(model_mae < naive_mae, "{model_mae} >= naive {naive_mae}");
        assert!(
            model_mae < seasonal_mae,
            "{model_mae} >= seasonal {seasonal_mae}"
        );
    }
}
