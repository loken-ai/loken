//! `POST /v1/forecast`: a time series in, the quantiles of its continuation out.

use super::*;
use axum::response::IntoResponse;

/// The values at `level`, read from the checkpoint's levels: the level itself when it has
/// it, else the straight line between the two around it. `None` outside their range.
fn at_level(levels: &[f32], values: &[Vec<f32>], level: f32) -> Option<Vec<f32>> {
    const SAME: f32 = 1e-6;
    if let Some(i) = levels.iter().position(|l| (l - level).abs() <= SAME) {
        return Some(values[i].clone());
    }
    let hi = levels.iter().position(|l| *l > level)?;
    let lo = hi.checked_sub(1)?;
    let w = (level - levels[lo]) / (levels[hi] - levels[lo]);
    Some(
        values[lo]
            .iter()
            .zip(&values[hi])
            .map(|(a, b)| a + w * (b - a))
            .collect(),
    )
}

/// The series of a request: numbers, with `null` as a missing point.
fn series_of(body: &serde_json::Value) -> Option<Vec<f32>> {
    body.get("series")?
        .as_array()?
        .iter()
        .map(|v| match v {
            serde_json::Value::Null => Some(f32::NAN),
            v => v.as_f64().map(|x| x as f32),
        })
        .collect()
}

pub(crate) async fn forecast(
    State(state): State<APIServer>,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    let err = |code: StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    let model = text_field(&body, "model");
    if let Err(e) = validate_model_id(&model) {
        return e.into_response();
    }
    let Some(series) = series_of(&body) else {
        return err(
            StatusCode::BAD_REQUEST,
            "`series` must be an array of numbers, `null` for a missing point".into(),
        );
    };
    let Some(horizon) = body
        .get("horizon")
        .and_then(serde_json::Value::as_u64)
        .filter(|h| *h > 0)
    else {
        return err(
            StatusCode::BAD_REQUEST,
            "`horizon` must be a positive integer".into(),
        );
    };
    let wanted: Option<Vec<f32>> = match body.get("quantiles") {
        None | Some(serde_json::Value::Null) => None,
        Some(q) => match q
            .as_array()
            .and_then(|a| a.iter().map(|v| v.as_f64().map(|x| x as f32)).collect())
        {
            Some(levels) => Some(levels),
            None => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "`quantiles` must be an array of levels between 0 and 1".into(),
                )
            }
        },
    };

    let hf_dir = state.huggingface_models_dir.clone();
    let Some((id, dir)) = media_catalogue::resolve(&hf_dir, &model) else {
        return err(
            StatusCode::NOT_FOUND,
            format!("Model '{model}' not found. Pull it first with POST /api/pull."),
        );
    };
    let is_forecaster = crate::inference::media_facts::read(&dir)
        .is_some_and(|f| f.kind == crate::inference::media_facts::MediaKind::Forecast);
    if !is_forecaster {
        return err(
            StatusCode::BAD_REQUEST,
            format!("Model '{id}' is not a time series forecaster"),
        );
    }

    let max_horizon = match crate::inference::model::chronos2::read_config(&dir) {
        Ok(cfg) => cfg.max_horizon(),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    };
    if horizon as usize > max_horizon {
        return err(
            StatusCode::BAD_REQUEST,
            format!("`horizon` {horizon} is past the {max_horizon} steps this model answers"),
        );
    }
    if !series.iter().any(|v| v.is_finite()) {
        return err(
            StatusCode::BAD_REQUEST,
            "`series` holds no observed value".into(),
        );
    }

    let engine = state.forecast_engine.clone();
    let t0 = std::time::Instant::now();
    let run =
        tokio::task::spawn_blocking(move || engine.forecast(&dir, &series, horizon as usize)).await;
    let f = match run {
        Ok(Ok(f)) => f,
        Ok(Err(e)) => {
            crate::inference::engine::log_engine_error("forecast", "forecast", &e);
            return err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"));
        }
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("forecast: {e}")),
    };

    let levels = wanted.unwrap_or_else(|| f.levels.clone());
    let mut quantiles = Vec::with_capacity(levels.len());
    for level in levels {
        match at_level(&f.levels, &f.values, level) {
            Some(values) => quantiles.push(serde_json::json!({ "level": level, "values": values })),
            None => {
                return err(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "quantile {level} is outside the levels this model answers, {} to {}",
                        f.levels.first().copied().unwrap_or(0.0),
                        f.levels.last().copied().unwrap_or(1.0)
                    ),
                )
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    info!("Forecast: model={id} horizon={horizon} {ms:.0}ms");
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "model": id,
            "object": "forecast",
            "horizon": horizon,
            "median": f.median(),
            "quantiles": quantiles,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_between_are_interpolated() {
        let levels = [0.1, 0.5, 0.9];
        let values = vec![vec![0.0], vec![4.0], vec![8.0]];
        assert_eq!(at_level(&levels, &values, 0.5), Some(vec![4.0]));
        let mid = at_level(&levels, &values, 0.3).unwrap();
        assert!((mid[0] - 2.0).abs() < 1e-5, "{mid:?}");
        assert_eq!(at_level(&levels, &values, 0.95), None);
        assert_eq!(at_level(&levels, &values, 0.05), None);
    }

    #[test]
    fn series_reads_null_as_missing() {
        let s = series_of(&serde_json::json!({"series": [1, null, 2.5]})).unwrap();
        assert_eq!(s[0], 1.0);
        assert!(s[1].is_nan());
        assert!(series_of(&serde_json::json!({"series": ["x"]})).is_none());
    }
}
