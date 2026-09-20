//! Calibrating and converting a released checkpoint, as detached work of the daemon.
//!
//! Both jobs take hours and hold the memory and the accelerators while they run, which is why they
//! belong to the daemon rather than to a process beside it: the daemon is what knows the store, and
//! what a client that disconnects must not cancel. Each request starts the work in a detached task
//! and streams its progress as NDJSON, the way `/api/create` does for a requantisation; a client that
//! goes away stops receiving, and the work still finishes.
//!
//! `POST /api/calibrate` serves the checkpoint over the texts the request carries and keeps the
//! record under the store, by name. `POST /api/convert` writes a GGUF from a checkpoint and a record,
//! and names it as a model in the store, servable at once.

use super::*;
use crate::inference::load::calibrated::Recipe;
use crate::inference::load::{calibration, requantize::parse_dtype};
use crate::inference::model::deepseek_v41::{
    convert, safetensors_source::SafeTensorsSource, source::WeightSource,
};
use crate::tensor::quantized::GgmlDType;
use std::path::PathBuf;
use std::sync::Mutex;

#[derive(Debug, Deserialize)]
pub(crate) struct CalibrateRequest {
    /// The released checkpoint, by its Hugging Face id.
    pub model: String,
    /// The record's name, for `/api/convert` to find it by.
    pub name: String,
    /// The texts to calibrate on; excerpts are taken from each in turn.
    pub texts: Vec<String>,
    /// Tokens per excerpt.
    pub excerpt: Option<usize>,
    /// Tokens in all.
    pub tokens: Option<usize>,
    /// Rows each projection keeps for the error correction; by default every row, while they fit.
    pub rows: Option<usize>,
    #[serde(default = "default_true")]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ConvertRequest {
    /// The released checkpoint, by its Hugging Face id.
    pub model: String,
    /// The record `/api/calibrate` wrote, by name.
    pub calibration: String,
    /// The model to create, `name:tag`.
    pub name: String,
    pub gate_up: Option<String>,
    pub down: Option<String>,
    /// Every large weight a token always reads: attention, shared expert, embeddings, head.
    pub always_read: Option<String>,
    pub min_rows: Option<usize>,
    pub damping: Option<f32>,
    /// `first:last`, for a partial file that checks a recipe first.
    pub layers: Option<String>,
    #[serde(default = "default_true")]
    pub stream: bool,
}

fn default_true() -> bool {
    true
}

/// Where calibration records live: beside the store's blobs, by name.
fn calibrations_dir(state: &APIServer) -> PathBuf {
    let blobs = state.model_manager.ollama().blobs_dir();
    blobs.parent().unwrap_or(&blobs).join("calibrations")
}

/// The released checkpoint `model` names, which this conversion knows how to read.
async fn checkpoint(state: &APIServer, model: &str) -> Result<PathBuf, ApiError> {
    validate_model_id(model)?;
    let dir = state
        .model_manager
        .resolve_path(model, "huggingface")
        .await
        .map_err(|_| ApiError::NotFound(format!("no checkpoint {model} in the store")))?;
    if !dir.join("inference").join("config.json").is_file() {
        return Err(ApiError::Validation(format!(
            "{model}: calibrate and convert read a released DeepSeek V4.1 checkpoint, and this is not one"
        )));
    }
    Ok(dir)
}

enum Msg {
    Progress(usize, usize, String),
    Done(String),
    Err(String),
}

/// Run `job` detached from the request, streaming what it reports. `job` gets a progress sink and
/// returns the status its last line carries.
fn detached<F>(stream: bool, first: String, job: F) -> Response
where
    F: FnOnce(&(dyn Fn(usize, usize, &str) + Sync)) -> std::result::Result<String, String>
        + Send
        + 'static,
{
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Msg>();
    tokio::spawn(async move {
        let worker = tx.clone();
        let res = tokio::task::spawn_blocking(move || {
            let sink = Mutex::new(worker.clone());
            let progress = move |done: usize, total: usize, what: &str| {
                let _ = sink
                    .lock()
                    .unwrap()
                    .send(Msg::Progress(done, total, what.to_string()));
            };
            job(&progress)
        })
        .await;
        let _ = tx.send(match res {
            Ok(Ok(status)) => Msg::Done(status),
            Ok(Err(e)) => Msg::Err(e),
            Err(join) => Msg::Err(format!("task: {join}")),
        });
    });
    let line = |v: serde_json::Value| axum::body::Bytes::from(v.to_string() + "\n");
    if stream {
        let body = async_stream::stream! {
            yield Ok::<_, std::io::Error>(line(serde_json::json!({ "status": first })));
            while let Some(m) = rx.recv().await {
                yield Ok(line(match m {
                    Msg::Progress(done, total, what) => serde_json::json!({
                        "status": what, "completed": done, "total": total,
                    }),
                    Msg::Done(status) => serde_json::json!({ "status": status }),
                    Msg::Err(e) => serde_json::json!({ "status": "error", "error": e }),
                }));
            }
        };
        return Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(axum::body::Body::from_stream(body))
            .unwrap();
    }
    let body = async_stream::stream! {
        let mut last = serde_json::json!({ "status": "error", "error": "the task gave no result" });
        while let Some(m) = rx.recv().await {
            match m {
                Msg::Done(status) => last = serde_json::json!({ "status": status }),
                Msg::Err(e) => last = serde_json::json!({ "status": "error", "error": e }),
                Msg::Progress(..) => {}
            }
        }
        yield Ok::<_, std::io::Error>(line(last));
    };
    Response::builder()
        .header("content-type", "application/json")
        .body(axum::body::Body::from_stream(body))
        .unwrap()
}

pub(crate) async fn calibrate_model(
    State(state): State<APIServer>,
    Json(request): Json<CalibrateRequest>,
) -> Result<Response, ApiError> {
    let dir = checkpoint(&state, &request.model).await?;
    validate_model_id(&request.name)?;
    if request.texts.iter().all(|t| t.trim().is_empty()) {
        return Err(ApiError::Validation(
            "calibrate: no text to calibrate on".into(),
        ));
    }
    let records = calibrations_dir(&state);
    std::fs::create_dir_all(&records)
        .map_err(|e| ApiError::Internal(format!("calibrations dir: {e}")))?;
    let out = records.join(format!("{}.gguf", request.name.replace([':', '/'], "_")));
    let first = format!("calibrating {} as '{}'", request.model, request.name);
    Ok(detached(request.stream, first, move |progress| {
        let src = SafeTensorsSource::open(&dir).map_err(|e| e.to_string())?;
        let cfg = convert::released_config(&src, &dir).map_err(|e| e.to_string())?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| e.to_string())?;
        let streams = request
            .texts
            .iter()
            .map(|t| {
                tok.encode(t.as_str(), false)
                    .map(|ids| ids.get_ids().to_vec())
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let length = request.excerpt.unwrap_or(200);
        let chosen = convert::excerpts(&streams, length, request.tokens.unwrap_or(2200));
        if chosen.is_empty() {
            return Err(format!(
                "calibrate: no text holds an excerpt of {length} tokens"
            ));
        }
        let inter = src
            .shape("layers.0.ffn.experts.0.w2.weight")
            .map(|s| s[1])
            .ok_or("calibrate: the checkpoint has no routed expert")?;
        let tokens: usize = chosen.iter().map(Vec::len).sum();
        let rows = convert::rows_to_keep(&cfg, inter, tokens, request.rows);
        let record = convert::calibrate(&src, &cfg, &chosen, rows, &|done, total| {
            progress(done, total, "calibrating layers")
        })
        .map_err(|e| e.to_string())?;
        let note = format!(
            "{} excerpts of {length} tokens from {} texts",
            chosen.len(),
            request.texts.len()
        );
        record.write(&out, &note).map_err(|e| e.to_string())?;
        Ok("success".to_string())
    }))
}

pub(crate) async fn convert_model(
    State(state): State<APIServer>,
    Json(request): Json<ConvertRequest>,
) -> Result<Response, ApiError> {
    let dir = checkpoint(&state, &request.model).await?;
    validate_model_id(&request.name)?;
    validate_model_id(&request.calibration)?;
    let record = calibrations_dir(&state).join(format!(
        "{}.gguf",
        request.calibration.replace([':', '/'], "_")
    ));
    if !record.is_file() {
        return Err(ApiError::NotFound(format!(
            "no calibration '{}'",
            request.calibration
        )));
    }
    let format = |name: Option<&str>, default: GgmlDType| -> Result<GgmlDType, ApiError> {
        name.map_or(Ok(default), |n| {
            parse_dtype(n).ok_or_else(|| ApiError::Validation(format!("unknown format {n}")))
        })
    };
    let defaults = convert::Formats::default();
    let formats = convert::Formats {
        gate_up: format(request.gate_up.as_deref(), defaults.gate_up)?,
        down: format(request.down.as_deref(), defaults.down)?,
        always_read: format(request.always_read.as_deref(), defaults.always_read)?,
    };
    let recipe = Recipe {
        min_rows: request.min_rows.unwrap_or(8),
        damping: request.damping.unwrap_or(1.0),
    };
    let layers = match request.layers.as_deref() {
        None => None,
        Some(span) => {
            let parsed = span
                .split_once(':')
                .and_then(|(a, b)| Some(a.parse::<usize>().ok()?..b.parse::<usize>().ok()?));
            Some(parsed.ok_or_else(|| ApiError::Validation("layers takes first:last".into()))?)
        }
    };
    let manager = state.model_manager.clone();
    let blobs = manager.ollama().blobs_dir();
    std::fs::create_dir_all(&blobs).map_err(|e| ApiError::Internal(format!("blobs dir: {e}")))?;
    let work = blobs.join(format!(
        ".convert-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let first = format!("converting {} into '{}'", request.model, request.name);
    let name = request.name.clone();
    Ok(detached(request.stream, first, move |progress| {
        let run = || -> std::result::Result<(), String> {
            let src = SafeTensorsSource::open(&dir).map_err(|e| e.to_string())?;
            let cfg = convert::released_config(&src, &dir).map_err(|e| e.to_string())?;
            let records = calibration::read(&record).map_err(|e| e.to_string())?;
            let layers = layers.unwrap_or(0..cfg.n_layers);
            let tokenizer = convert::TokenizerFiles::read(&dir).map_err(|e| e.to_string())?;
            let digest = convert::convert(
                &src,
                &cfg,
                &records,
                &work,
                formats,
                recipe,
                layers,
                Some(&tokenizer),
                &|done, total, _name| progress(done, total, "converting tensors"),
            )
            .map_err(|e| e.to_string())?;
            progress(0, 0, "writing manifest");
            ollama::store_gguf_with_digest(manager.ollama(), &work, &digest, &name, [None; 4])
        };
        let outcome = run();
        if outcome.is_err() {
            let _ = std::fs::remove_file(&work);
            let _ = std::fs::remove_file(work.with_extension("gguf.partial"));
        }
        outcome.map(|()| "success".to_string())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request names what it needs and takes the documented defaults for the rest.
    #[test]
    fn requests_take_their_defaults() {
        let c: CalibrateRequest = serde_json::from_str(
            r#"{"model":"deepseek-ai/DeepSeek-V4.1-Flash","name":"docs","texts":["a text"]}"#,
        )
        .unwrap();
        assert!(c.stream && c.tokens.is_none() && c.rows.is_none());
        let v: ConvertRequest = serde_json::from_str(
            r#"{"model":"deepseek-ai/DeepSeek-V4.1-Flash","calibration":"docs","name":"dsv41:iq2","stream":false}"#,
        )
        .unwrap();
        assert!(!v.stream && v.gate_up.is_none() && v.layers.is_none());
    }
}
