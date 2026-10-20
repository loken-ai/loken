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

/// The rows of a covariate field: an array of rows, or an object of named rows. Absent is
/// no row. `cell` reads one point; `None` from it rejects the field.
fn rows_of<T>(
    body: &serde_json::Value,
    field: &str,
    cell: impl Fn(&serde_json::Value) -> Option<T>,
) -> Result<Vec<Vec<T>>, String> {
    let rows: Vec<&serde_json::Value> = match body.get(field) {
        None | Some(serde_json::Value::Null) => return Ok(Vec::new()),
        Some(serde_json::Value::Array(rows)) => rows.iter().collect(),
        Some(serde_json::Value::Object(rows)) => rows.values().collect(),
        Some(_) => {
            return Err(format!(
                "`{field}` must be an array of rows or an object of rows"
            ))
        }
    };
    rows.into_iter()
        .map(|row| {
            row.as_array()
                .and_then(|r| r.iter().map(&cell).collect::<Option<Vec<T>>>())
                .ok_or_else(|| format!("`{field}` holds a row that is not an array of its points"))
        })
        .collect()
}

/// A numeric point: a number, `null` for a missing one.
fn number(v: &serde_json::Value) -> Option<f32> {
    match v {
        serde_json::Value::Null => Some(f32::NAN),
        v => v.as_f64().map(|x| x as f32),
    }
}

/// Category rows as codes: each distinct value of a row its own code, `null` missing.
fn categories(rows: Vec<Vec<serde_json::Value>>) -> Vec<Vec<f64>> {
    rows.into_iter()
        .map(|row| {
            let mut seen: Vec<String> = Vec::new();
            row.iter()
                .map(|v| {
                    if v.is_null() {
                        return f64::NAN;
                    }
                    let key = v.to_string();
                    let code = seen.iter().position(|k| *k == key).unwrap_or_else(|| {
                        seen.push(key);
                        seen.len() - 1
                    });
                    code as f64
                })
                .collect()
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

    let limits = match crate::inference::engine::forecast_engine::limits(&dir) {
        Ok(l) => l,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    };
    let max_horizon = limits.max_horizon;
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
    let scalar = |v: &serde_json::Value| (!v.is_array() && !v.is_object()).then(|| v.clone());
    let fields = (
        rows_of(&body, "past_covariates", number),
        rows_of(&body, "known_covariates", number),
        rows_of(&body, "categorical_past_covariates", scalar),
        rows_of(&body, "categorical_known_covariates", scalar),
    );
    let (past, known, cat_past, cat_known) = match fields {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, categories(c), categories(d)),
        (Err(e), ..) | (_, Err(e), ..) | (_, _, Err(e), _) | (.., Err(e)) => {
            return err(StatusCode::BAD_REQUEST, e)
        }
    };
    let req = crate::inference::engine::forecast_engine::Request {
        series,
        horizon: horizon as usize,
        past,
        known,
        categorical_past: cat_past,
        categorical_known: cat_known,
    };
    if req.has_covariates() && !limits.covariates {
        return err(
            StatusCode::BAD_REQUEST,
            format!("Model '{id}' reads one series without covariates"),
        );
    }
    if req.has_known() && !limits.known_covariates {
        return err(
            StatusCode::BAD_REQUEST,
            format!("Model '{id}' cannot read covariates known over the horizon"),
        );
    }
    let t = req.series.len();
    let short = |rows: &[Vec<f32>], want: usize| rows.iter().any(|r| r.len() != want);
    let short_codes = |rows: &[Vec<f64>], want: usize| rows.iter().any(|r| r.len() != want);
    if short(&req.past, t) || short_codes(&req.categorical_past, t) {
        return err(
            StatusCode::BAD_REQUEST,
            format!("a past covariate must be as long as `series`, {t} points"),
        );
    }
    let span = t + req.horizon;
    if short(&req.known, span) || short_codes(&req.categorical_known, span) {
        return err(
            StatusCode::BAD_REQUEST,
            format!("a known covariate must cover `series` and the horizon, {span} points"),
        );
    }

    let engine = state.forecast_engine.clone();
    let t0 = std::time::Instant::now();
    let run = tokio::task::spawn_blocking(move || engine.forecast(&dir, &req)).await;
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
    fn covariates_read_as_rows_named_or_not() {
        let body = serde_json::json!({
            "past_covariates": [[1, null, 2]],
            "known_covariates": {"price": [1, 2], "promo": [0, 1]},
            "categorical_past_covariates": [["a", "b", "a", null]],
        });
        let past = rows_of(&body, "past_covariates", number).unwrap();
        assert!(past[0][1].is_nan() && past[0][2] == 2.0);
        assert_eq!(rows_of(&body, "known_covariates", number).unwrap().len(), 2);
        let scalar = |v: &serde_json::Value| Some(v.clone());
        let codes = categories(rows_of(&body, "categorical_past_covariates", scalar).unwrap());
        assert_eq!(&codes[0][..3], &[0.0, 1.0, 0.0]);
        assert!(codes[0][3].is_nan());
        assert!(rows_of(&body, "absent", number).unwrap().is_empty());
        assert!(rows_of(&serde_json::json!({"x": [["a"]]}), "x", number).is_err());
    }

    #[test]
    fn series_reads_null_as_missing() {
        let s = series_of(&serde_json::json!({"series": [1, null, 2.5]})).unwrap();
        assert_eq!(s[0], 1.0);
        assert!(s[1].is_nan());
        assert!(series_of(&serde_json::json!({"series": ["x"]})).is_none());
    }
}
