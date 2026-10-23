//! The fixed set of the Timestral gate, scored the same way: the median forecast of a
//! downloaded checkpoint must beat the seasonal naive forecast on mean MASE.

use super::*;
use crate::inference::model::timestral::tests::{
    fixed, fixed_request, mae, naive_scale, seasonal_naive, Vectors, FIXED_SET,
};
use std::path::PathBuf;

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
        .filter(|dir| read_config(dir).is_ok())
        .collect();
    found.sort();
    found.dedup();
    found
}

/// The forecast of `horizon` steps, chunked at the checkpoint's longest single pass: each
/// chunk's median extends the series the next chunk is forecast from.
fn forecast(f: &Forecaster, series: &[f32], horizon: usize) -> Forecast {
    let cap = f.config().max_horizon();
    let (mut levels, mut values) = (Vec::new(), Vec::new());
    let mut extended = series.to_vec();
    let mut left = horizon;
    while left > 0 {
        let h = left.min(cap);
        let got = f.forecast(&extended, h).unwrap();
        let median = got.median().to_vec();
        if levels.is_empty() {
            levels = got.levels.clone();
            values = vec![Vec::new(); got.levels.len()];
        }
        for (v, q) in values.iter_mut().zip(&got.values) {
            v.extend(q);
        }
        extended.extend(&median);
        left -= h;
    }
    Forecast { levels, values }
}

#[test]
fn chronos2_beats_seasonal_naive() {
    let dirs = local_checkpoints();
    if dirs.is_empty() {
        eprintln!("no checkpoint of this network downloaded; skipped");
        return;
    }
    let v = Vectors::open(&fixed().join("fixed.safetensors"));
    for dir in dirs {
        let cfg = read_config(&dir).unwrap();
        let key = format!("d{}.l{}", cfg.d_model, cfg.num_layers);
        let f = load_dir(&dir, &Device::Cpu).unwrap();
        let (mut mase, mut mase_naive, mut ms) = (0f32, 0f32, 0f64);
        for name in FIXED_SET {
            let (req, actual, period) = fixed_request(&v, name);
            let t0 = std::time::Instant::now();
            let got = forecast(&f, &req.series, req.horizon);
            ms += t0.elapsed().as_secs_f64() * 1e3;
            let scale = naive_scale(&req.series, period);
            let naive = seasonal_naive(&req.series, period, req.horizon);
            mase += mae(got.median(), &actual) / scale;
            mase_naive += mae(&naive, &actual) / scale;
        }
        let n = FIXED_SET.len() as f32;
        let (mase, mase_naive) = (mase / n, mase_naive / n);
        eprintln!("{key}: {ms:.0} ms, MASE {mase:.3} (seasonal naive {mase_naive:.3})");
        assert!(mase < mase_naive, "{key}: MASE {mase} >= {mase_naive}");
    }
}
