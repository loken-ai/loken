//! HTTP request handlers for the API server
//!
//! This module contains the implementation of HTTP handlers for model
//! management and chat completion endpoints.
//! Supports Ollama-compatible API format.

use axum::{
    extract::{rejection::JsonRejection, FromRequest, Path, Query, Request, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, warn};

use crate::api::types::*;
use crate::gpu::{GPUManagerImpl, GPUManagerInterface};
#[cfg(feature = "audio")]
use crate::inference::engine::audio_engine::{
    AudioEngine, AudioTranscribeParams, TranscribeResult, WhisperTask,
};
#[cfg(feature = "image")]
use crate::inference::engine::image_engine::{
    ImageEngine, ImageGenParams, ImageStreamEvent, LoadingProgress as ImageEngineLoadingProgress,
};
use crate::inference::engine::llm_engine::{GenerationParams, InferenceConfig, LlmEngine};
#[cfg(feature = "audio")]
use crate::inference::engine::tts_engine::{LoadWatch, TtsResult, TtsSynthParams};
use crate::inference::load::model_manager::ModelManager;
use crate::stats::StatsMonitor;
use serde::{Deserialize, Serialize};

/// Safely serialize to JSON with error logging instead of panic
macro_rules! to_json_string {
    ($obj:expr) => {
        match serde_json::to_string(&$obj) {
            Ok(json) => json,
            Err(e) => {
                error!("JSON serialization error: {}", e);
                r#"{"error":"internal error"}"#.to_string()
            }
        }
    };
}

mod anthropic_api;
#[cfg(feature = "audio")]
mod audio;
#[cfg(feature = "image")]
pub(crate) mod media;
#[cfg(feature = "metrics")]
mod metrics;
mod model_manager;
mod ollama;
mod openai;
mod prompt_format;
#[cfg(feature = "audio")]
mod separate;
mod system;

pub(crate) use anthropic_api::*;
#[cfg(feature = "audio")]
pub(crate) use audio::*;
#[cfg(feature = "image")]
pub(crate) use media::*;
// Re-exported at crate-visibility above; these also need to reach the render CLI, which
// classifies, resolves and LOADS a checkpoint exactly as the handlers do rather than
// keeping a dispatch of its own.
#[cfg(feature = "image")]
pub use media::{
    image_family, image_family_loader, image_model_defaults, load_image_family,
    requested_checkpoint, ImageLoader, ImageModelDefaults,
};
pub(crate) use model_manager::*;
pub(crate) use ollama::*;
pub(crate) use openai::*;
pub(crate) use prompt_format::*;
pub(crate) use system::*;
// `parse_keep_alive` is re-exported `pub` from api/mod.rs.
pub use model_manager::parse_keep_alive;

/// Normalize model ID for Ollama models (append :latest if no tag)
/// - "llama3" -> "llama3:latest"
/// - "llama3:8b" -> "llama3:8b" (already has tag)
/// - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" -> unchanged (HuggingFace)
fn normalize_model_id(model_id: &str) -> String {
    // Skip if it has a path separator (HuggingFace) or already has a tag
    if model_id.contains('/') || model_id.contains(':') {
        return model_id.to_string();
    }
    // Ollama model without tag - append :latest
    format!("{}:latest", model_id)
}

/// Clamp a finite f64 into `[lo, hi]`. NaN / ±Infinity collapse to
/// `default` rather than propagating into the engine - `f64::clamp`
/// returns NaN for NaN inputs (per IEEE 754 + Rust spec), which would
/// then poison guidance / strength / temperature values downstream.
/// Used by the image + tts endpoints where the field is optional and
/// a sensible fallback already exists.
fn clamp_finite_f64(v: f64, lo: f64, hi: f64, default: f64) -> f64 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        default.clamp(lo, hi)
    }
}

/// f32 sibling of `clamp_finite_f64` - same semantics, same rationale.
/// Used by whisper temperature parsing.
fn clamp_finite_f32(v: f32, lo: f32, hi: f32, default: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        default.clamp(lo, hi)
    }
}

/// Cap for OpenAI's `user` field (also the Ollama `options.session_id`
/// extension). The value flows directly into the KV cache HashMap as
/// the per-session key - without a cap, a single client request can
/// pin an arbitrary-sized String into the engine pool. 256 chars
/// matches OpenAI's documented user-id ceiling. Empty/None is fine
/// (no session reuse for that turn).
fn validate_user_id(user: Option<&str>) -> Result<(), ApiError> {
    let Some(u) = user else {
        return Ok(());
    };
    // Empty string ≡ None for our purposes (no session is keyed off it
    // anyway). Reject non-empty oversized values.
    if u.is_empty() {
        return Ok(());
    }
    if u.len() > 256 {
        return Err(ApiError::Validation(format!(
            "`user` is {} chars; cap at 256 (OpenAI's documented user-id ceiling)",
            u.len()
        )));
    }
    // Reject control characters (NUL, newlines, etc.). The user_id flows
    // into structured-log lines and HashMap keys; an embedded `\n` would
    // break log parsing for downstream collectors, and a NUL would wedge
    // some terminals on display. Printable + space is the only safe set.
    if u.chars().any(|c| c.is_control()) {
        return Err(ApiError::Validation(
            "`user` contains control characters (NUL/newline/etc.); ASCII-printable only".into(),
        ));
    }
    Ok(())
}

/// Called at the entry of every handler that takes a model name from
/// request body / URL path before it reaches the model_manager. Keep
/// it strict but tolerant of normal Ollama / HF naming
/// (`<publisher>/<model>:<tag>`).
fn validate_model_id(model_id: &str) -> Result<(), ApiError> {
    // `trim().is_empty()` already covers the bare `is_empty()` case.
    if model_id.trim().is_empty() {
        return Err(ApiError::Validation("`model` must not be empty".into()));
    }
    if model_id.len() > 256 {
        return Err(ApiError::Validation(format!(
            "`model` is {} chars; cap at 256",
            model_id.len()
        )));
    }
    // Reject path-traversal patterns. We accept normal slashes
    // (`publisher/model`) but not `..`, backslash, NUL, control
    // characters, leading whitespace, or leading dots in any segment.
    if model_id.contains('\0')
        || model_id.contains('\\')
        || model_id
            .chars()
            .any(|c| c.is_control() || c == ' ' || c == '\t')
        || model_id.contains("..")
        || model_id.starts_with('/')
        || model_id.starts_with('.')
    {
        return Err(ApiError::Validation(format!(
            "`model` '{model_id}' contains invalid characters"
        )));
    }
    Ok(())
}

/// Estimate token count from text
/// Uses a reasonable approximation: 1 token ≈ 1.3 words on average
/// This is a common heuristic used when exact tokenization isn't available
/// For more accurate counts, use a proper tokenizer based on model architecture
fn estimate_token_count(text: &str) -> u64 {
    // Count words (tokens separated by whitespace)
    let word_count = text.split_whitespace().count();

    // Apply heuristic: average word is about 1.3 tokens
    // This is a reasonable approximation for most models
    // More sophisticated tokenizers can be used for better accuracy
    ((word_count as f64 * 1.3).ceil()) as u64
}

/// Server state containing model manager and inference engine
/// One origin for every elapsed-time reading the detector sees, fixed at first use.
///
/// The detector compares silences; taking each reading from a fresh `Instant` would make
/// every peer look as though it had just spoken.
static CLUSTER_EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

#[derive(Clone)]
pub struct APIServer {
    ollama_models_dir: String,
    huggingface_models_dir: String,
    model_manager: Arc<ModelManager>,
    /// The catalogue published to peers, with the moment it was read. Listing walks the model
    /// directories, and gossip asks every round while the answer changes when someone pulls a
    /// model - so it is remembered rather than recomputed.
    servable: Arc<RwLock<Option<(std::time::Instant, Vec<String>)>>>,
    engines: Arc<RwLock<Vec<LoadedModelEntry>>>,
    gpu_manager: Option<Arc<GPUManagerImpl>>,
    stats_monitor: Arc<StatsMonitor>,
    default_keep_alive: i64,
    default_inference_config: InferenceConfig,
    #[cfg(feature = "image")]
    image_engine: Arc<ImageEngine>,
    #[cfg(feature = "audio")]
    audio_engine: Arc<AudioEngine>,
    #[cfg(feature = "audio")]
    tts_engine: Arc<crate::inference::engine::TtsEngine>,
    /// Priority-aware gate in front of `engine.generate*()` calls. Only one
    /// generate runs at a time (matches the model lock); waiters are
    /// ordered by priority so FIM editor completions skip ahead of queued
    /// chats. Does not preempt mid-decode.
    request_gate: Arc<crate::api::gate::RequestGate>,
    /// Per-model load locks. Prevents two concurrent /api/generate (or
    /// /api/chat) empty-prompt LOAD requests for the same model from
    /// both calling engine.load_model() in parallel - without this, both
    /// requests pass the "is loaded?" check, both spawn their own engine,
    /// and the engines vec ends up with two entries for the same model
    /// (each holding its own ~20 GB of weight tensors).
    loading_locks: Arc<RwLock<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    /// ONE media job at a time (image / video / audio generation, including its
    /// model load). Media engines each want most of a card; running two at once
    /// cannot fit by construction, and doing it anyway produced the OOM storm that
    /// took the image path down (a Flux load racing a Qwen-Image VAE decode: both
    /// cards squeezed, every fallback exhausted, a view materialization panicking
    /// inside a worker). LLM generations are NOT gated here - they compete for VRAM
    /// through the reclaim protocol instead, so chat stays responsive during a render.
    media_gate: Arc<tokio::sync::Mutex<()>>,
    /// API keys accepted when authentication is required. Empty + required = every
    /// request is refused, which is the safe direction for a misconfiguration.
    auth: Arc<AuthGate>,
    /// The cluster this node belongs to, or `None` when it runs alone.
    ///
    /// An Option rather than an always-present object with an empty peer list: a node that is
    /// not clustered must take exactly the code path it took before any of this existed, and
    /// `None` makes that structural instead of a condition somebody can forget to write.
    cluster: Option<Arc<crate::distributed::cluster::Cluster>>,
}

/// Relay a request to a peer and stream its answer back as it arrives.
///
/// Streamed, not buffered, and that is the whole point: a generation takes seconds to minutes,
/// and `bytes().await` would hold every token until the last one was produced. A client asking
/// for a stream would see nothing, then everything - which is not a slow cluster, it is a
/// broken one, and only a forwarded STREAMING request shows it.
///
/// Status and content type are passed through as received. A proxy that reinterprets turns a
/// peer's clean error into a local one and hides which node actually failed - on a cluster the
/// first question after a bad response is always which machine produced it.
/// The bytes that mark a finished generation on the wire. Matched literally: text the model
/// produces cannot contain it, because a quote inside a JSON string arrives escaped.
const TERMINAL_MARKER: &[u8] = b"\"done\":true";

pub(crate) async fn forward_to_peer(
    url: &str,
    body: &serde_json::Value,
) -> Result<axum::response::Response, ApiError> {
    use axum::response::IntoResponse;
    let (status, headers, resp) = crate::distributed::cluster_runtime::forward_generate(url, body)
        .await
        .map_err(ApiError::Internal)?;
    let code = axum::http::StatusCode::from_u16(status.as_u16())
        .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    // Each chunk is handed on the moment it arrives. What the relay must add is an ending:
    // when the serving node dies mid-generation the body simply stops, with no error and no
    // terminal chunk, and a client reading to end-of-stream cannot tell that from a finished
    // answer. Measured on two machines - a killed peer cost 1118 chunks and no complaint.
    //
    // So the stream is watched for its own terminator and given one if it never arrives. The
    // check is a byte match on the terminal marker rather than a parse: generated text cannot
    // forge it, since any such text is escaped on its way into the JSON string.
    let saw_end = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watching = saw_end.clone();
    // A transport error has to END this stream, not travel down it. Forwarded, it aborts the
    // body where it stands and nothing chained after is ever polled - which is exactly how the
    // first version of this closing chunk came to be silently dropped.
    let upstream = futures::StreamExt::take_while(resp.bytes_stream(), |item| {
        futures::future::ready(item.is_ok())
    });
    let stream = futures::StreamExt::map(upstream, move |chunk| {
        if let Ok(bytes) = &chunk {
            if bytes
                .windows(TERMINAL_MARKER.len())
                .any(|w| w == TERMINAL_MARKER)
            {
                watching.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        chunk
    });
    // Runs only after the body above has ended, which is what lets it read the flag.
    let closing = futures::StreamExt::filter_map(
        futures::stream::once(async move { saw_end.load(std::sync::atomic::Ordering::Relaxed) }),
        |ended| async move {
            if ended {
                return None;
            }
            tracing::warn!("cluster: the peer serving this request stopped before finishing it");
            Some(Ok::<_, reqwest::Error>(axum::body::Bytes::from(
                serde_json::json!({
                    "response": "",
                    "done": true,
                    "done_reason": "peer_failed",
                    "error": "the node serving this request stopped before finishing it",
                })
                .to_string()
                    + "\n",
            )))
        },
    );
    let mut response =
        axum::body::Body::from_stream(futures::StreamExt::chain(stream, closing)).into_response();
    *response.status_mut() = code;
    if let Ok(v) = axum::http::HeaderValue::from_str(&content_type) {
        response
            .headers_mut()
            .insert(axum::http::header::CONTENT_TYPE, v);
    }
    Ok(response)
}

/// What the auth middleware needs, resolved once at startup.
///
/// Kept as plain accepted-key hashes rather than the full AuthManager because that is
/// all a bearer check needs, and a smaller surface is easier to be sure about. Keys are
/// compared by SHA-256 digest in constant time, so neither the config value nor a
/// timing difference leaks through.
pub(crate) struct AuthGate {
    required: bool,
    key_hashes: Vec<[u8; 32]>,
    /// Origins a browser may call from once the API is credentialed.
    allowed_origins: Vec<String>,
    /// Per-client throttle. `None` = unlimited, which stays the default: a local
    /// install is one user, and a limit there only gets in the way.
    limiter: Option<Arc<crate::api::rate_limiter::RateLimiter>>,
}

impl AuthGate {
    fn new(
        required: bool,
        keys: &[String],
        allowed_origins: &[String],
        per_minute: usize,
        burst: usize,
    ) -> Self {
        use sha2::{Digest, Sha256};
        let key_hashes = keys
            .iter()
            .map(|k| {
                let mut h = Sha256::new();
                h.update(k.trim().as_bytes());
                let d = h.finalize();
                let mut out = [0u8; 32];
                out.copy_from_slice(&d);
                out
            })
            .collect();
        // A token bucket, not a fixed window: the refill rate is the sustained
        // allowance and the capacity is how much of it may be spent at once, so a UI
        // opening several calls together is not punished while a loop still converges
        // to the configured rate.
        let limiter = (per_minute > 0).then(|| {
            Arc::new(crate::api::rate_limiter::RateLimiter::new(
                crate::api::rate_limiter::RateLimitConfig {
                    max_requests: per_minute,
                    window: std::time::Duration::from_secs(60),
                    burst_capacity: burst.max(1),
                    refill_rate: per_minute as f64 / 60.0,
                },
            ))
        });
        Self {
            required,
            key_hashes,
            allowed_origins: allowed_origins.to_vec(),
            limiter,
        }
    }

    /// Whether `presented` is one of the accepted keys.
    ///
    /// Constant time in the number of keys AND in the comparison: a short-circuiting
    /// `==` on a secret is a timing oracle, and the fact that this server is usually
    /// local is not a reason to write one.
    fn accepts(&self, presented: &str) -> bool {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(presented.trim().as_bytes());
        let d = h.finalize();
        let mut ok = false;
        for k in &self.key_hashes {
            let mut diff = 0u8;
            for (a, b) in k.iter().zip(d.iter()) {
                diff |= a ^ b;
            }
            ok |= diff == 0;
        }
        ok
    }
}

/// Which origins CORS admits.
///
/// Unauthenticated: anything, because the wildcard protects nothing an attacker could
/// not do from a script anyway. Authenticated: only the configured list, because the
/// API now carries credentials and a wildcard would let any page a user visits spend
/// their key. An empty list under auth means no browser origin at all - use a proxy.
fn cors_origin(auth: &AuthGate) -> tower_http::cors::AllowOrigin {
    if !auth.required {
        return tower_http::cors::AllowOrigin::any();
    }
    let list: Vec<axum::http::HeaderValue> = auth
        .allowed_origins
        .iter()
        .filter_map(|o| o.parse().ok())
        .collect();
    tower_http::cors::AllowOrigin::list(list)
}

/// Endpoints an orchestrator must reach without a credential and without a quota.
fn is_health_path(path: &str) -> bool {
    matches!(path, "/health" | "/healthz" | "/api/health")
}

/// Apply the per-client rate limit, if one is configured.
///
/// The client is the API KEY when there is one, not the address: several people behind
/// one NAT are separate clients, and one key used from several addresses is one client.
/// Falling back to the address for anonymous traffic keeps an open install from being
/// throttled globally by a single caller.
async fn throttled(
    auth: &AuthGate,
    client: &str,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Some(limiter) = auth.limiter.as_ref() else {
        return next.run(req).await;
    };
    // Health probes are exempt, as they are from authentication. An orchestrator polls
    // liveness on a fixed schedule it does not coordinate with anyone; throttling that
    // turns a busy server into an apparently DEAD one, and restarting a loaded server
    // because it was briefly popular is worse than the load that caused it. Observed
    // while testing this: /health answered 429.
    if is_health_path(req.uri().path()) {
        return next.run(req).await;
    }
    if limiter.check_rate_limit(client).await {
        return next.run(req).await;
    }
    // 429 with Retry-After: a client that is told to back off can, and one that is
    // simply refused will hammer.
    (
        axum::http::StatusCode::TOO_MANY_REQUESTS,
        [(axum::http::header::RETRY_AFTER, "1")],
        axum::Json(serde_json::json!({
            "error": {
                "message": "rate limit exceeded",
                "type": "rate_limit_error",
                "code": "too_many_requests"
            }
        })),
    )
        .into_response()
}

/// Reject requests without a valid API key when authentication is required.
///
/// Health endpoints stay open: an orchestrator has to be able to probe liveness without
/// holding a credential, and they expose nothing.
async fn require_api_key(
    axum::extract::State(auth): axum::extract::State<Arc<AuthGate>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !auth.required {
        // Still throttled when a rate is configured: an open API is exactly the one
        // that needs a ceiling, and tying the limit to authentication would leave the
        // more exposed case unprotected.
        return throttled(&auth, "anonymous", req, next).await;
    }
    if is_health_path(req.uri().path()) {
        return next.run(req).await;
    }
    // `Authorization: Bearer <key>` is what the OpenAI clients already send; `X-API-Key`
    // is accepted so a plain curl does not need the ceremony.
    // Owned, so the request can be moved into the next layer afterwards.
    let presented: Option<String> = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .or_else(|| req.headers().get("x-api-key").and_then(|v| v.to_str().ok()))
        .map(str::to_string);
    match presented {
        Some(key) if auth.accepts(&key) => throttled(&auth, &key, req, next).await,
        _ => (
            axum::http::StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({
                "error": {
                    "message": "missing or invalid API key",
                    "type": "invalid_request_error",
                    "code": "unauthorized"
                }
            })),
        )
            .into_response(),
    }
}

impl APIServer {
    /// The cluster handle, if this node joined one.
    pub(crate) fn cluster_handle(&self) -> Option<&Arc<crate::distributed::cluster::Cluster>> {
        self.cluster.as_ref()
    }

    /// The monotonic origin the failure detector measures from.
    pub fn cluster_started(&self) -> std::time::Instant {
        *CLUSTER_EPOCH.get_or_init(std::time::Instant::now)
    }

    /// Join a cluster. Called once at startup when peers are configured or discovered.
    pub fn attach_cluster(&mut self, cluster: Arc<crate::distributed::cluster::Cluster>) {
        self.cluster = Some(cluster);
    }

    /// How much of `prompt` this node holds for `model`, in tokens.
    ///
    /// Only the engine holding the model can answer: the tokeniser is its own, and so is the
    /// cache. `None` from the engine means not resident, which the caller reads as none.
    pub(crate) async fn cached_prompt_tokens(&self, model: &str, prompt: &str) -> Option<usize> {
        // The template first: the session holds the tokens the generation path prefilled, and
        // that path templates the prompt unless the caller asked for raw. Comparing against the
        // bare text diverges at token zero and answers zero forever.
        let templated = super::handlers::ollama::templated_prompt(self, model, prompt).await;
        let engines = self.engines.read().await;
        let entry = engines.iter().find(|e| e.model_id == model)?;
        entry.engine.cached_prompt_tokens(model, &templated).await
    }

    /// What this node publishes to its peers.
    ///
    /// `prefix_blocks` is empty for now and that is deliberate rather than forgotten: the
    /// hashes routing needs are the ones `paged_kv` computes over TOKEN blocks, and inventing
    /// them here from the prompt text would publish a cache that does not exist - routing
    /// would then send requests to a node for a prefix it never held. Until that is wired,
    /// selection weighs load, residency and measured rates, which is already the larger part.
    /// The catalogue this node could load from, as opposed to what it currently holds.
    ///
    /// Cached, because gossip asks for it every round while the answer changes when someone
    /// pulls a model - minutes apart at best. Listing walks the model directories, and doing
    /// that once a second on every node would be a filesystem load the cluster imposes on
    /// itself for nothing.
    /// `None` means the catalogue could not be read - published as "did not say" rather than
    /// as "nothing", so a transient directory error cannot quietly remove this node from every
    /// routing decision in the cluster.
    pub async fn servable_models(&self) -> Option<Vec<String>> {
        const REFRESH: std::time::Duration = std::time::Duration::from_secs(60);
        {
            let seen = self.servable.read().await;
            if let Some((at, ref list)) = *seen {
                if at.elapsed() < REFRESH {
                    return Some(list.clone());
                }
            }
        }
        let list: Vec<String> = match self.model_manager.list_models().await {
            // `id`, not `name`: a request names a model the way the API names it, tag
            // included. `name` is a display label with the tag dropped, so every quantisation
            // of one model collapses to the same string and matches no request at all - a
            // catalogue that looks full and answers nothing.
            Ok(models) => models.into_iter().map(|m| m.id).collect(),
            Err(e) => {
                tracing::warn!("cluster: cannot list local models ({e}); publishing no catalogue");
                return None;
            }
        };
        *self.servable.write().await = Some((std::time::Instant::now(), list.clone()));
        Some(list)
    }

    /// What this node publishes about itself, for gossip and for its own place in the table.
    ///
    /// One function, because the node has to appear in its own routing table exactly as peers
    /// see it. Two builders would let the local view and the published view drift, and the
    /// symptom - a node judging itself by different facts than its peers use - is invisible
    /// until a request lands somewhere absurd.
    pub async fn local_node_state(&self) -> crate::distributed::membership::NodeState {
        let r = self.cluster_report().await;
        r.split().1
    }

    pub async fn cluster_report(&self) -> crate::distributed::cluster_runtime::PeerReport {
        let models: Vec<String> = self
            .engines
            .read()
            .await
            .iter()
            .map(|e| e.model_id.clone())
            .collect();
        let serves = self.servable_models().await;
        let node_id = self
            .cluster
            .as_ref()
            .map(|c| c.config.node_id.clone())
            .unwrap_or_default();
        // Published per model: this node's speed on one model predicts nothing about
        // another. The flat fields below stay as a fallback for a peer that has no entry.
        let by_model = crate::distributed::rate_meter::all();
        let rates = by_model.values().copied().fold(
            crate::distributed::rate_meter::Rates::default(),
            |acc, r| if r.samples > acc.samples { r } else { acc },
        );
        crate::distributed::cluster_runtime::PeerReport {
            node_id,
            models,
            serves,
            // Load as the gate sees it: what is running plus what is waiting, against the
            // gate's own capacity. Measured rather than guessed, and it is the same figure
            // that decides admission here, so a peer reading it sees what this node feels.
            // Load as the gate sees it: running plus waiting, against what this node can
            // actually take. It is the same figure that decides admission here, so a peer
            // reading it sees what this node feels rather than a count it cannot interpret.
            load: {
                let g = self.request_gate.snapshot().await;
                let busy = (g.in_flight + g.queue_depth) as f32;
                (busy / self.request_gate.capacity().max(1) as f32).min(1.0)
            },
            // The raw queue alongside the fraction: routing prices the WAIT as rounds of
            // service, and only a count over a width can say how many rounds there are.
            // The generate path never crosses the admission gate, so the gate alone reads
            // idle under any load; the meter's own counter is where generations actually run.
            busy: {
                let g = self.request_gate.snapshot().await;
                (g.in_flight + g.queue_depth) as u32 + crate::distributed::rate_meter::in_flight()
            },
            lanes: self.request_gate.capacity() as u32,
            // Probed once at startup and held by the manager, so publishing costs a lock
            // rather than an NVML pass per gossip round.
            devices: self
                .gpu_manager
                .as_ref()
                .map(|g| {
                    use crate::gpu::GPUManagerInterface;
                    g.get_devices().iter().map(|d| d.name()).collect()
                })
                .unwrap_or_default(),
            prefix_blocks: Vec::new(),
            // Measured on this node's own work. Zero until something has run here, which a
            // peer reads as unknown and prices pessimistically - a node has to earn its
            // reputation rather than be granted one from a nameplate.
            rates_by_model: by_model
                .into_iter()
                .map(|(m, r)| {
                    (
                        m,
                        (
                            r.prefill_tok_per_s,
                            r.decode_tok_per_s,
                            r.model_load_s,
                            r.agg_tok_per_s,
                        ),
                    )
                })
                .collect(),
            prefill_tok_per_s: rates.prefill_tok_per_s,
            decode_tok_per_s: rates.decode_tok_per_s,
            model_load_s: rates.model_load_s,
        }
    }

    /// Apply the configured authentication policy.
    ///
    /// Separate from construction because the server is built before the config is
    /// resolved; calling it is what turns the setting into behaviour, and forgetting to
    /// call it leaves auth OFF - which is why the binary logs what it applied.
    pub fn configure_auth(
        &mut self,
        required: bool,
        keys: &[String],
        origins: &[String],
        rate_per_minute: usize,
        rate_burst: usize,
    ) {
        self.auth = Arc::new(AuthGate::new(
            required,
            keys,
            origins,
            rate_per_minute,
            rate_burst,
        ));
    }

    /// Create a new API server with explicit Ollama and HuggingFace directories
    pub fn new(ollama_models_dir: String, huggingface_models_dir: String) -> Self {
        Self::with_config(
            ollama_models_dir,
            huggingface_models_dir,
            InferenceConfig::default(),
        )
    }

    /// Create server with custom inference config
    pub fn with_config(
        ollama_models_dir: String,
        huggingface_models_dir: String,
        inference_config: InferenceConfig,
    ) -> Self {
        Self::with_config_and_keep_alive(
            ollama_models_dir,
            huggingface_models_dir,
            inference_config,
            get_default_keep_alive(),
        )
    }

    /// Create server with custom inference config and keep_alive
    pub fn with_config_and_keep_alive(
        ollama_models_dir: String,
        huggingface_models_dir: String,
        inference_config: InferenceConfig,
        keep_alive_minutes: i64,
    ) -> Self {
        let model_manager = Arc::new(ModelManager::new(
            &ollama_models_dir,
            &huggingface_models_dir,
        ));

        // Initialize GPU manager and detect GPUs
        let gpu_manager = Arc::new(GPUManagerImpl::new());
        info!("GPU manager initialized");

        // Initialize stats monitor
        let stats_monitor = Arc::new(StatsMonitor::new(Some(gpu_manager.clone())));

        // Get default keep_alive from env or use default
        let default_keep_alive = get_default_keep_alive();
        info!("Default keep_alive: {} minutes", default_keep_alive);

        let request_gate = crate::api::gate::RequestGate::new(1);
        let server = Self {
            ollama_models_dir,
            huggingface_models_dir,
            model_manager,
            servable: Arc::new(RwLock::new(None)),
            engines: Arc::new(RwLock::new(Vec::new())),
            gpu_manager: Some(gpu_manager),
            stats_monitor,
            default_keep_alive: keep_alive_minutes,
            default_inference_config: inference_config,
            #[cfg(feature = "image")]
            image_engine: Arc::new(ImageEngine::new()),
            #[cfg(feature = "audio")]
            audio_engine: Arc::new(AudioEngine::new()),
            #[cfg(feature = "audio")]
            tts_engine: Arc::new(crate::inference::engine::TtsEngine::new()),
            request_gate,
            loading_locks: Arc::new(RwLock::new(std::collections::HashMap::new())),
            media_gate: Arc::new(tokio::sync::Mutex::new(())),
            // Defaults to off; `configure_auth` applies the config at startup.
            auth: Arc::new(AuthGate::new(false, &[], &[], 0, 10)),
            // Off until the binary joins one; `attach_cluster` is what turns it on.
            cluster: None,
        };
        // Register every engine's idle resident with the central VRAM authority. The
        // hooks fire ONLY under the pressure protocol (a hot component that would otherwise
        // land on CPU), LRU-ordered; a request touching an engine bumps it to MRU.
        //
        // Driven by `ResidentEngine::ALL` - the same walk `unload_model` performs. Two
        // hand-written lists let an engine be reclaimable under pressure and unreachable on
        // request at the same time: `keep_alive:0` against a resident voice model would
        // answer "unload" and free nothing, while a video render evicted it without trouble.
        //
        // The LLM hook reclaims the LEAST-recently-used entry, one per call, so the
        // pressure loop re-probes between evictions and stops as soon as a card fits.
        // Before it existed, an image generation could never evict a chat model left
        // resident by its keep_alive - the "edit -> enhance -> edit" failure, where the
        // second edit OOM'd on the MMQ workspace with an idle LLM squatting the card.
        // The face networks are likewise not an engine but a cache, and were once
        // unreclaimable statics: five of them accumulated on one card and the server
        // retried OOMs in matmul with no way to free them short of a restart.
        for kind in ResidentEngine::ALL {
            let kind = *kind;
            let releaser = server.clone();
            let holder = server.clone();
            crate::inference::place::vram_manager::register_reclaimer(
                kind.reclaimer_name(),
                Box::new(move || {
                    let s = releaser.clone();
                    Box::pin(async move { s.release_resident(kind).await })
                }),
                Box::new(move || {
                    let s = holder.clone();
                    Box::pin(async move { s.resident_bytes(kind).await })
                }),
            );
        }
        server
    }

    /// Serialize one media job (load + generate) against every other media job.
    /// Held for the whole request; LLM generations are deliberately not gated.
    pub(crate) async fn media_lock(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.media_gate.lock().await
    }

    /// Acquire the per-model load lock. Held for the duration of a single
    /// load attempt so two concurrent requests for the same model don't
    /// each spawn their own engine.
    async fn loading_lock(&self, model_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.loading_locks.write().await;
        locks
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    /// Idempotent load. If the model is already in `engines`, returns Ok
    /// without doing work. Otherwise serialises with any in-flight load
    /// of the same model via `loading_lock`, double-checks after acquiring
    /// the lock, and only then calls engine.load_model() + pushes to
    /// engines. Caller-supplied `keep_alive_minutes` controls the
    /// expiration timer.
    async fn ensure_loaded(&self, model_id: &str, keep_alive_minutes: i64) -> Result<(), String> {
        // Fast path: already loaded.
        if self.get_engine(model_id).await.is_ok() {
            return Ok(());
        }
        let lock = self.loading_lock(model_id).await;
        let _g = lock.lock().await;
        // Re-check under the lock - another concurrent request may have
        // already loaded it.
        if self.get_engine(model_id).await.is_ok() {
            return Ok(());
        }
        // Auto-evict LRU models if we'd exceed the loaded-models cap.
        // Generic across all architectures: any prior engine with its
        // weights still on GPU steals VRAM from the new model and may
        // force the next HeteroPlan into CPU fallback.
        self.evict_to_make_room(model_id).await;

        let config = self.config_for_model(model_id);
        // Pressure protocol: idle MEDIA residents (image/video/TTS engines) hold
        // VRAM this LLM may need. Same rule as the media loads in reverse -
        // hetero placement first, reclaim only when the model would not fit a
        // GPU otherwise. Sized from the checkpoint file (resident ~ file size
        // for GGUF).
        let hot = self.model_blob_size(model_id, &config);
        if hot > 0 {
            crate::inference::place::vram_manager::ensure_gpu_headroom("llm", hot, 0).await;
        }
        // A LOAD THAT RUNS OUT OF MEMORY IS NOT A MISSING MODEL.
        //
        // There was no retry here: a chat request that arrived while a render held the
        // cards failed outright, and the handler turned that into a 404 - "could not be
        // loaded" - for a model sitting on disk. Observed under exactly the load this
        // server is for: `cuda OOM in QCudaStorage upload on GPU0` during the load, 404
        // to the caller 91 seconds later.
        //
        // So it escalates instead, the same way the image path does: each notch shrinks
        // every card's usable budget, which moves layers off the GPU rather than
        // re-packing the starved one, and the last notch places on the host, which
        // always fits. Slower is a result; a 404 is not.
        // START FROM NO PRESSURE. The degrade level is a PER-REQUEST escalation - its own
        // documentation says it is reset "at the START of a render invocation so each run
        // begins at the pack-first no-pressure path" - but it lives in a process-wide
        // atomic that only ever counts up. So a notch raised by some earlier failure,
        // possibly in another subsystem entirely, was still shrinking every card by 4 GB
        // per notch when this placement ran, and the model went to the host with VRAM
        // sitting free. Reported exactly that way: "on a de la VRAM disponible mais le
        // traitement reste sur CPU".
        //
        // Resetting here rather than after a success is the difference between an
        // escalation and a memory of an old failure.
        #[cfg(feature = "audio")]
        crate::inference::place::vram_manager::vram_degrade_reset();
        const LOAD_ATTEMPTS: usize = 3;
        let mut engine = Arc::new(LlmEngine::with_config(self.config_for_model(model_id)));
        let mut last_err = String::new();
        for attempt in 1..=LOAD_ATTEMPTS {
            match engine.load_model().await {
                Ok(()) => {
                    last_err.clear();
                    break;
                }
                Err(e) => {
                    last_err = format!("{e}");
                    let out_of_memory = last_err.to_lowercase().contains("out of memory")
                        || last_err.to_lowercase().contains("oom");
                    if !out_of_memory || attempt == LOAD_ATTEMPTS {
                        break;
                    }
                    let level = crate::inference::place::vram_manager::vram_degrade();
                    tracing::warn!(
                        "llm: {model_id} ran out of VRAM loading ({last_err}) - pressure now \
                         {level}, re-planning further off the GPU rather than failing"
                    );
                    engine = Arc::new(LlmEngine::with_config(self.config_for_model(model_id)));
                }
            }
        }
        if !last_err.is_empty() {
            return Err(last_err);
        }
        let expire_handle = if keep_alive_minutes > 0 {
            Some(self.schedule_expiration(model_id.to_string(), keep_alive_minutes))
        } else {
            None
        };
        let mut engines = self.engines.write().await;
        // Final dedup: in case the lock was bypassed (e.g. an admin path),
        // refuse to push if an entry already exists.
        if engines.iter().any(|e| e.model_id == model_id) {
            // Drop our extra engine before returning to free its resources.
            // Engine's Drop impl handles teardown.
            return Ok(());
        }
        let mut entry =
            LoadedModelEntry::new(model_id.to_string(), engine, Some(keep_alive_minutes));
        entry.expire_handle = expire_handle;
        engines.push(entry);
        Ok(())
    }

    /// Read the chat template blob from an Ollama manifest for a given model.
    /// The description a distributor packaged with the model: format, family, size and
    /// quantisation, as it publishes them.
    fn read_manifest_config(&self, model_id: &str) -> Option<serde_json::Value> {
        let (name, tag) = match model_id.split_once(':') {
            Some((n, t)) => (n.to_lowercase(), t),
            None => (model_id.to_lowercase(), "latest"),
        };
        let ollama = crate::inference::load::ollama_manager::OllamaManager::new(PathBuf::from(
            &self.ollama_models_dir,
        ));
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(ollama.get_model_path(&name, tag)?).ok()?,
        )
        .ok()?;
        let digest = manifest.get("config")?.get("digest")?.as_str()?;
        let blob = PathBuf::from(&self.ollama_models_dir)
            .join("blobs")
            .join(digest.replace(':', "-"));
        serde_json::from_str(&std::fs::read_to_string(blob).ok()?).ok()
    }

    /// A blob from the model's manifest, by layer kind (`model`, `params`, `license`, `template`).
    fn manifest_layer_path(&self, model_id: &str, kind: &str) -> Option<PathBuf> {
        let (name, tag) = match model_id.split_once(':') {
            Some((n, t)) => (n.to_lowercase(), t),
            None => (model_id.to_lowercase(), "latest"),
        };
        let ollama = crate::inference::load::ollama_manager::OllamaManager::new(PathBuf::from(
            &self.ollama_models_dir,
        ));
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(ollama.get_model_path(&name, tag)?).ok()?,
        )
        .ok()?;
        for layer in manifest.get("layers")?.as_array()? {
            let media = layer
                .get("mediaType")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if media.rsplit('.').next() == Some(kind) {
                let digest = layer.get("digest").and_then(|v| v.as_str())?;
                return Some(
                    PathBuf::from(&self.ollama_models_dir)
                        .join("blobs")
                        .join(digest.replace(':', "-")),
                );
            }
        }
        None
    }

    fn read_manifest_layer(&self, model_id: &str, kind: &str) -> Option<String> {
        std::fs::read_to_string(self.manifest_layer_path(model_id, kind)?).ok()
    }

    /// The generation defaults a model ships with, as the `PARAMETER` lines of a Modelfile.
    fn read_model_parameters(&self, model_id: &str) -> Option<String> {
        let params: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&self.read_manifest_layer(model_id, "params")?).ok()?;
        let mut lines = Vec::new();
        // A repeated key is one line each, which is how `stop` carries several sequences.
        let mut push = |k: &str, v: &serde_json::Value| {
            let rendered = match v {
                serde_json::Value::String(s) => format!("{s:?}"),
                other => other.to_string(),
            };
            lines.push(format!("{k:<31}{rendered}"));
        };
        for (k, v) in &params {
            match v {
                serde_json::Value::Array(xs) => xs.iter().for_each(|x| push(k, x)),
                other => push(k, other),
            }
        }
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    fn read_chat_template(&self, model_id: &str) -> Option<String> {
        let (model_name, tag) = if model_id.contains(':') {
            let parts: Vec<&str> = model_id.split(':').collect();
            (
                parts[0].to_lowercase(),
                parts.get(1).copied().unwrap_or("latest"),
            )
        } else {
            (model_id.to_lowercase(), "latest")
        };
        let ollama = crate::inference::load::ollama_manager::OllamaManager::new(PathBuf::from(
            &self.ollama_models_dir,
        ));
        let manifest_path = ollama.get_model_path(&model_name, tag)?;
        let content = std::fs::read_to_string(&manifest_path).ok()?;
        let manifest: serde_json::Value = serde_json::from_str(&content).ok()?;
        let layers = manifest.get("layers")?.as_array()?;
        let mut model_blob: Option<String> = None;
        let mut packaged: Option<String> = None;
        for layer in layers {
            let media_type = layer
                .get("mediaType")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if media_type.contains("template") {
                if let Some(digest) = layer.get("digest").and_then(|v| v.as_str()) {
                    let blob_path = PathBuf::from(&self.ollama_models_dir)
                        .join("blobs")
                        .join(digest.replace(':', "-"));
                    packaged = std::fs::read_to_string(&blob_path).ok();
                }
            }
            if media_type.contains("model") {
                model_blob = layer
                    .get("digest")
                    .and_then(|v| v.as_str())
                    .map(|d| d.replace(':', "-"));
            }
        }
        // No template layer. The weights file carries the authoritative one under
        // `tokenizer.chat_template`; read it rather than letting the caller guess
        // from the model name. The reference resolves this with a built-in
        // per-architecture renderer, so a name-derived guess diverges from it: on
        // the qwen3-next family it wrapped a plain completion prompt as a chat
        // turn, and the model replied that the user's message looked truncated
        // instead of continuing the text.
        // Prefer the weights file's own Jinja template over the distributor's
        // wrapper. Both describe the same format, but the packaged one is a Go
        // template we can only approximate by pattern-matching, while the Jinja
        // one we evaluate exactly - and an approximated chat format makes a model
        // close its turn after a dozen tokens.
        let from_gguf = (|| {
            let blob_path = PathBuf::from(&self.ollama_models_dir)
                .join("blobs")
                .join(model_blob.clone()?);
            let content = crate::tensor::quantized::gguf_file::open_header(&blob_path).ok()?;
            content
                .metadata
                .get("tokenizer.chat_template")?
                .to_string()
                .ok()
                .cloned()
        })();
        match from_gguf {
            Some(j) if j.contains("{%") => Some(j),
            other => packaged.or(other),
        }
    }

    /// True when the model accepts image input (has a vision projector)  -
    /// Stage C capability probe, done WITHOUT loading the model. Ollama
    /// packages the vision tower as a separate manifest layer
    /// (`application/vnd.ollama.image.projector`), so a manifest scan is an
    /// exact, load-free signal for all Ollama vision models (moondream,
    /// llava, ...). Falls back to a tight name heuristic for direct-path HF
    /// GGUF whose sibling `mmproj*.gguf` the Ollama manifest can't see
    /// (pixtral, qwen-vl, ...).
    fn model_has_vision(&self, model_id: &str) -> bool {
        let (model_name, tag) = if model_id.contains(':') {
            let parts: Vec<&str> = model_id.split(':').collect();
            (
                parts[0].to_lowercase(),
                parts.get(1).copied().unwrap_or("latest"),
            )
        } else {
            (model_id.to_lowercase(), "latest")
        };
        let ollama = crate::inference::load::ollama_manager::OllamaManager::new(PathBuf::from(
            &self.ollama_models_dir,
        ));
        if let Some(manifest_path) = ollama.get_model_path(&model_name, tag) {
            if let Ok(content) = std::fs::read_to_string(&manifest_path) {
                if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
                        if layers.iter().any(|layer| {
                            layer
                                .get("mediaType")
                                .and_then(|v| v.as_str())
                                .is_some_and(|mt| mt.contains("projector"))
                        }) {
                            return true;
                        }
                    }
                }
            }
        }
        is_vision_model_by_name(&model_name)
    }

    /// Create an InferenceConfig for a model, resolving the correct models_dir.
    /// Checks HuggingFace dir first (for flat GGUF downloads), then Ollama dir.
    fn config_for_model(&self, model_id: &str) -> InferenceConfig {
        let mut config = self.default_inference_config.clone();
        config.model_id = model_id.to_string();

        // Check if this model exists in the HuggingFace directory
        let hf_manager = crate::inference::load::huggingface_manager::HuggingFaceManager::new(
            PathBuf::from(&self.huggingface_models_dir),
        );
        if let Some(path) = hf_manager.get_model_path(model_id) {
            if path.is_file() {
                // Flat GGUF file - point models_dir directly at the file
                config.models_dir = Some(path);
            } else {
                // Cache-format directory
                config.models_dir = Some(path);
            }
        } else {
            // Default to Ollama models dir
            config.models_dir = Some(PathBuf::from(&self.ollama_models_dir));
        }

        config
    }

    fn model_blob_size(
        &self,
        model_id: &str,
        config: &crate::inference::engine::llm_engine::InferenceConfig,
    ) -> u64 {
        if let Some(dir) = &config.models_dir {
            if dir.is_file() {
                return std::fs::metadata(dir).map(|m| m.len()).unwrap_or(0);
            }
        }
        let (name, tag) = match model_id.split_once(':') {
            Some((n, t)) => (n.to_string(), t.to_string()),
            None => (model_id.to_string(), "latest".to_string()),
        };
        let ollama = crate::inference::load::ollama_manager::OllamaManager::new(PathBuf::from(
            &self.ollama_models_dir,
        ));
        ollama
            .get_model_path(&name, &tag)
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Start periodic stats logging
    pub fn start_stats_monitoring(&self) {
        let engines = self.engines.clone();
        let get_model_info: Arc<dyn Fn() -> (usize, u64) + Send + Sync> = Arc::new(move || {
            // Use try_read to avoid blocking, default to 0 if lock not available
            match engines.try_read() {
                Ok(engines) => {
                    let count = engines.len();
                    let size: u64 = engines
                        .iter()
                        .map(|e| e.engine.model_size_nonblocking())
                        .sum();
                    (count, size)
                }
                Err(_) => (0, 0),
            }
        });

        self.stats_monitor
            .start_periodic_logging(Duration::from_secs(10), get_model_info);
    }

    /// Create router with state
    /// Is any request actually being SERVED right now?
    ///
    /// A shutdown has to wait for work, not for sockets. `with_graceful_shutdown` waits
    /// for connections to close, and a client holding one open - an idle keep-alive, a
    /// stream nobody ended - holds it forever: the signal arrives, the drain begins, and
    /// the process never exits. Seen directly - SIGTERM logged, the heartbeat still
    /// running minutes later, and SIGKILL the only way out, which throws away the clean
    /// unload the graceful path exists to give.
    ///
    /// Both gates count: a generation holds the request gate, a render holds the media
    /// one, and either is work a stop should let finish.
    pub async fn work_in_flight(&self) -> bool {
        if self.media_gate.try_lock().is_err() {
            return true;
        }
        self.request_gate.snapshot().await.in_flight > 0
    }

    pub fn create_router(&self) -> axum::Router {
        // Capture process start the first time the router is built so
        // /health can report uptime. Lock-free after this set.
        let _ = SERVER_START.set(std::time::Instant::now());
        let router = axum::Router::new()
            // Root endpoints
            .route("/", axum::routing::get(ollama_root).head(ollama_head_root))
            // Ollama-compatible endpoints
            .route("/api/tags", axum::routing::get(ollama_list_models))
            // Stopping a render, by identifier rather than by hoping the connection is
            // noticed. See `inference::serve::cancel::registry` for why the transport is not a
            // dependable signal.
            // What a render will be, and what it will cost, BEFORE it starts.
            // (video planner route folded into the video group below)
            .route("/v1/renders", axum::routing::get(list_renders))
            .route(
                "/v1/renders/{id}/cancel",
                axum::routing::post(cancel_render),
            )
            .route("/api/pull", axum::routing::post(ollama_pull_model))
            .route("/api/delete", axum::routing::delete(ollama_delete_model))
            .route("/api/show", axum::routing::post(ollama_show_model))
            .route("/api/chat", axum::routing::post(ollama_chat))
            .route("/api/generate", axum::routing::post(ollama_generate))
            // (voice route folded into the audio group below)
            // .route("/api/unload", axum::routing::post(ollama_unload_model))
            .route("/api/ps", axum::routing::get(ollama_ps))
            .route("/api/inflight", axum::routing::get(inflight_status))
            .route("/api/draft/attach", axum::routing::post(draft_attach))
            .route("/api/draft/detach", axum::routing::post(draft_detach))
            .route("/api/draft/status", axum::routing::get(draft_status))
            .route("/api/copy", axum::routing::post(ollama_copy_model))
            .route("/api/embed", axum::routing::post(ollama_embed))
            .route("/api/embeddings", axum::routing::post(ollama_embed))
            // OpenAI-shaped embeddings - same engine plumbing as
            // /api/embed, response reshaped to the `{object:"list",
            // data:[{embedding,index,object:"embedding"}],usage:{...}}`
            // envelope SDKs expect.
            .route("/v1/embeddings", axum::routing::post(openai_embeddings))
            .route("/v1/rerank", axum::routing::post(openai_rerank))
            .route("/rerank", axum::routing::post(openai_rerank))
            .route("/api/create", axum::routing::post(ollama_create_model))
            .route("/api/push", axum::routing::post(ollama_push_model))
            .route(
                "/api/blobs/{digest}",
                axum::routing::head(ollama_head_blob).post(ollama_create_blob),
            )
            // Custom endpoints
            .route("/api/models/loaded", axum::routing::get(list_loaded_models))
            .route("/api/models/validate", axum::routing::post(validate_model))
            .route("/api/models/repair", axum::routing::post(repair_model))
            .route("/api/swap", axum::routing::post(swap_model_handler))
            .route("/api/layers/swap", axum::routing::post(swap_layers_handler))
            // OpenAI-compatible endpoints (for compatibility)
            .route("/api/models", axum::routing::get(list_models))
            // OpenAI-compatible model index. Lists locally available
            // LLM checkpoints plus the multimodal models the server
            // can spin up (whisper ASR, parler-tts, flux/z-image).
            .route("/v1/models", axum::routing::get(openai_list_models))
            // OpenAI's "retrieve model" - single-entry lookup by id.
            // SDKs call this to confirm a model is available before
            // sending a generation.
            .route(
                "/v1/models/{model_id}",
                axum::routing::get(openai_retrieve_model),
            )
            .route(
                "/api/chat/completions",
                axum::routing::post(chat_completion),
            )
            .route("/v1/chat/completions", axum::routing::post(chat_completion))
            // Anthropic Messages API shim - lets Claude Code and other
            // Anthropic-protocol clients use loken natively. Same
            // engine + tool-calling plumbing as the OpenAI path, request/
            // response translated to/from Anthropic content blocks.
            .route("/v1/messages", axum::routing::post(anthropic_messages))
            // Legacy text-completion endpoint (OpenAI 0.27-era). Many
            // older SDKs (LangChain default, llamaindex) still call it.
            // Internally just a thin shim: prompt passes through raw
            // (no chat-template wrapping), response is reshaped to the
            // `text_completion` object form with `choices[*].text`
            // instead of `choices[*].message.content`.
            .route("/v1/completions", axum::routing::post(text_completions))
            // (Removed: /v1/images/cache/{filename}. Generated images no
            // longer touch the filesystem - privacy fix. Callers using
            // response_format=url now get the same b64_json payload as
            // response_format=b64_json; the OpenAI SDK accepts both.)
            .route("/health", axum::routing::get(health_check))
            .route("/api/version", axum::routing::get(ollama_version))
            // Distributed inference endpoints
            .route("/api/distributed/devices", axum::routing::get(list_devices))
            .route("/api/cluster/state", axum::routing::get(cluster_state))
            .route("/api/cluster/prefix", axum::routing::post(cluster_prefix))
            .route("/api/cluster/peers", axum::routing::get(cluster_peers))
            .route(
                "/api/distributed/stats",
                axum::routing::get(distributed_stats),
            )
            .route(
                "/api/distributed/recommend",
                axum::routing::get(recommended_model),
            )
            .route(
                "/api/layer_perf",
                axum::routing::get(layer_performance_endpoint),
            )
            .route(
                "/api/stage_perf",
                axum::routing::get(stage_performance_endpoint),
            )
            // Multi-device endpoints
            .route(
                "/api/multi-device/status",
                axum::routing::get(get_multi_device_status),
            )
            // Fallback 404 handler with logging
            .fallback(handle_404)
            // Defense-in-depth: convert ANY handler panic into a clean 500 instead
            // of dropping the connection (client sees a reset, no response). Innermost
            // layer so it wraps the handlers directly; the outer layers then format
            // the 500 normally, so a single malformed request costs its own caller a
            // 500 and no one else anything.
            //
            // This layer catches by UNWINDING, so it is worth exactly nothing if the
            // profile aborts on panic - which the release profile did, for months,
            // while this comment claimed the opposite. The gate below keeps the two
            // in agreement.
            .layer(CatchPanicLayer::new())
            // Global request timeout. It bounds every non-streaming handler and the
            // time-to-first-byte of streaming ones - streaming BODIES are produced after
            // the response is returned, so a long SSE generation is not cut off by it.
            //
            // This exists to release a WEDGED connection, not to decide how long
            // legitimate work may take, and at 600 s it was doing the second: a video is
            // minutes of denoise and a long one is many segments of it, so the ceiling sat
            // below the honest cost of what was being asked for. Set far above any real
            // render; a request still hanging after this has stopped being a render.
            .layer(tower_http::timeout::TimeoutLayer::with_status_code(
                axum::http::StatusCode::REQUEST_TIMEOUT,
                std::time::Duration::from_secs(6 * 3600),
            ))
            // Audio + image multipart uploads dwarf axum's default 2 MB body limit. This
            // has to stay clear of the sum of the per-asset caps, because one request can
            // carry several: an edit sends a source image, a mask and a reference face,
            // and a montage sends a whole set. Sizing it to a single asset is what made
            // the server refuse ordinary material.
            .layer(axum::extract::DefaultBodyLimit::max(4 * 1024 * 1024 * 1024))
            // AUTHENTICATION. Applied to everything the router serves except the
            // health endpoints, which a load balancer has to reach unauthenticated.
            // Off unless the config turns it on, so an existing local install is
            // untouched; on, it is the difference between "anyone who can route a
            // packet here can delete my models" and not.
            .layer(axum::middleware::from_fn_with_state(
                self.auth.clone(),
                require_api_key,
            ))
            // CORS. Wide open ONLY while the API is unauthenticated, where it protects
            // nothing anyway - the browser is not the attacker's only option and
            // same-origin would not stop a script that can simply use fetch from a
            // server. Once auth is on the API is CREDENTIALED, and a wildcard origin
            // there invites any page the user visits to spend their key, so origins
            // must be named.
            .layer(
                CorsLayer::new()
                    .allow_origin(cors_origin(&self.auth))
                    .allow_methods(Any)
                    .allow_headers(Any)
                    // Without explicit expose_headers, browsers can't
                    // read response headers like `x-request-id` or
                    // `server-timing` from JS - only the safelisted
                    // CORS-default ones. We expose everything since the
                    // server has no auth and no sensitive header data.
                    .expose_headers(Any)
                    // Cache preflight for 1 hour. Browsers re-send
                    // OPTIONS on every request without this, which
                    // doubles the request count from JS clients. 1h
                    // is the Chromium ceiling - Firefox caps at 24h
                    // but ignores larger values gracefully.
                    .max_age(std::time::Duration::from_secs(3600)),
            )
            // Gzip / brotli / zstd response compression when the
            // client supports it (Accept-Encoding header). High-value
            // for /v1/models (3KB+ JSON), /v1/embeddings (4KB+ b64
            // string), large chat responses, and SRT/VTT subtitle
            // dumps. Streaming SSE responses are skipped automatically
            // (CompressionLayer detects text/event-stream).
            .layer(CompressionLayer::new())
            // Mint an `x-request-id` per request and propagate it to
            // the response. OpenAI returns `X-Request-Id: req_xyz`;
            // SDKs surface it for retry / log correlation. If the
            // client supplied one, SetRequestId leaves it intact.
            //
            // Tower layers compose inside-out: the LAST `.layer()` is
            // the innermost wrapper. To get Set-then-Propagate on the
            // request side, declare them in *reverse* - Propagate
            // outer, Set inner - so on request flow Set runs first
            // (mints the ID), then Propagate snapshots it for the
            // response.
            .layer(PropagateRequestIdLayer::x_request_id())
            .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
            .layer(TraceLayer::new_for_http());

        // The scrape endpoint, when the feature asks for it. Folded in as a rebinding for the
        // same reason as the media families below: a chained builder cannot carry a cfg on one
        // link, and a build without it answers 404 rather than existing and refusing.
        #[cfg(feature = "metrics")]
        let router = router.route("/metrics", axum::routing::get(metrics::metrics));

        // Audio routes. A chained builder cannot carry a cfg on one link,
        // so the family folds in as a rebinding: absent from a build without it,
        // and the endpoint then answers 404 rather than existing and failing.
        #[cfg(feature = "audio")]
        let router = router
            .route("/voice", axum::routing::post(voice_handler))
            // Unified multimodal conversation: auto-routes a turn to chat / vision /
            // image-gen / TTS and returns the assistant turn to append (Stage A).
            .route("/conversation", axum::routing::post(conversation_handler))
            // OpenAI-compatible whisper ASR. Multipart form: `file` (WAV
            // bytes, 16 kHz mono required), `model` (defaults to
            // openai/whisper-small), `language`, `temperature`.
            .route(
                "/v1/audio/transcriptions",
                axum::routing::post(audio_transcriptions),
            )
            .route(
                "/v1/audio/translations",
                axum::routing::post(audio_translations),
            )
            // OpenAI-compatible TTS. JSON body: { "model": "...",
            //   "input": "text to speak", "voice": "alloy|echo|...",
            //   "response_format": "wav"|"pcm", "speed": 1.0 }
            // Returns mono audio bytes at the loaded checkpoint's native
            // rate (mini-v1: 44.1 kHz, large-v1: 24 kHz).
            .route("/v1/audio/speech", axum::routing::post(audio_speech))
            // Music source separation (Mel-Band RoFormer). Multipart: `file` = the
            // mix in any supported container, `stems` = vocals|instrumental|both.
            // Returns base64 WAVs at 44.1 kHz stereo; the two stems sum back to the
            // input exactly, because the accompaniment IS the mix minus the vocal.
            .route(
                "/v1/audio/separate",
                axum::routing::post(separate::audio_separate),
            )
            // List the supported TTS voice presets and the description
            // each one maps to. Helps clients discover voices without
            // reading source.
            .route("/v1/audio/voices", axum::routing::get(audio_voices))
            // Generative audio (text->sound). JSON body:
            //   { "model": "ezaudio", "prompt": "...", "seconds": 5,
            //     "steps": 50, "cfg": 3.0, "seed": 0 }
            // Returns data[0].b64_json = WAV base64. `ezaudio` -> text->SFX
            // (OpenSound EzAudio, MIT). Closes the CLI-only gap for media gen.
            .route(
                "/v1/audio/generations",
                axum::routing::post(audio_generations),
            );

        // Image routes. A chained builder cannot carry a cfg on one link,
        // so the family folds in as a rebinding: absent from a build without it,
        // and the endpoint then answers 404 rather than existing and failing.
        #[cfg(feature = "image")]
        let router = router
            // OpenAI-compatible image generation. JSON body:
            //   { "model": "...", "prompt": "...", "n": 1,
            //     "size": "1024x1024", "response_format": "b64_json" }
            // Default model resolves to Flux schnell GGUF (Z-Image when the
            // requested name contains "z-image").
            .route("/v1/loras", axum::routing::get(list_loras))
            .route(
                "/v1/images/generations",
                axum::routing::post(images_generations),
            )
            // OpenAI-compatible image edit (img2img). Multipart form:
            //   `image` (PNG/JPEG bytes), `prompt`, `model`, `n`, `size`,
            //   `response_format`, plus extension fields `strength`/`seed`.
            .route("/v1/images/edits", axum::routing::post(images_edits))
            // OpenAI-compatible image variations. Multipart form:
            //   `image` (PNG/JPEG), optional `model`, `n`, `size`,
            //   `response_format`, plus extension fields `strength`/`seed`.
            // Internally a no-prompt img2img run.
            .route(
                "/v1/images/variations",
                axum::routing::post(images_variations),
            );

        // Video routes. A chained builder cannot carry a cfg on one link,
        // so the family folds in as a rebinding: absent from a build without it,
        // and the endpoint then answers 404 rather than existing and failing.
        #[cfg(feature = "video")]
        let router = router
            .route("/api/video/plan", axum::routing::post(plan_video))
            // Generative video (text->video, Wan) returned as an animated GIF. JSON body:
            //   { "model": "wan", "prompt": "...", "frames": 17, "size": "256x256",
            //     "steps": 20, "cfg": 6.0, "seed": 0 }. Heavy (minutes/clip).
            .route(
                "/v1/video/generations",
                axum::routing::post(video_generations),
            );

        // The state binds LAST: every family group above is added to a Router that still
        // carries it, and binding turns the type into the one the caller expects.
        router.with_state(self.clone())
    }
}

/// Tight name heuristic for known vision-LLM families - the fallback for
/// direct-path HF GGUF (sibling mmproj) that `model_has_vision`'s manifest
/// scan can't see. Kept conservative (specific family tokens, not a bare
/// "vision" substring) to avoid tagging a plain text model as image-capable.
fn is_vision_model_by_name(model_name: &str) -> bool {
    let l = model_name.to_lowercase();
    l.contains("moondream")
        || l.contains("llava")
        || l.contains("bakllava")
        || l.contains("pixtral")
        || l.contains("minicpm-v")
        || l.contains("minicpm-o")
        || l.contains("-vl")           // qwen2-vl / qwen2.5-vl / qwen3-vl
        || l.contains("qwen-vl")
        || l.contains("internvl")
        || l.contains("llama3.2-vision")
        || l.contains("llama-3.2-vision")
        || l.contains("granite3.2-vision")
        || l.contains("gemma3-vision")
        || l.contains("smolvlm")
}

/// Models that ship in the HF cache as components of a larger pipeline
/// (T5 / CLIP text+vision encoders for Flux), or as encoder-only
/// architectures (whisper ASR, parler TTS) that have their own purpose-
/// built endpoints. Routing any of these to /api/chat or /api/generate
/// would either panic the tensor-op decoder (no decoder layers to load)
/// or silently produce garbage. Returning a clear 400 instead keeps the
/// user from staring at a vague "no images" / panic backtrace.
///
/// Returns `Some(suggestion)` when the model should be rejected with
/// the suggested alternative endpoint, `None` when it's free to proceed
/// down the text-decoder path.
fn non_chat_pipeline_component(model_name: &str) -> Option<&'static str> {
    let lower = model_name.to_lowercase();
    // CLIP / T5: used as text+vision encoders by Flux. The Flux loader
    // pulls them on its own - direct use isn't supported.
    if lower.contains("clip-vit") || lower.contains("clip_vit") || lower.contains("/clip-") {
        return Some(
            "CLIP is a text+image encoder, not a chat model. \
             Select a Flux / Z-Image model instead - it pulls CLIP \
             automatically as a component.",
        );
    }
    if lower.contains("t5-v1_1") || lower.contains("/t5-") || lower.contains("t5xxl") {
        return Some(
            "T5 is a text encoder, not a chat model. \
             Select a Flux model instead - it pulls T5 automatically.",
        );
    }
    // (Whisper is NOT rejected here on the /api/chat path - it's routed
    // to handle_chat_asr when the user attaches an audio file via
    // images[]. On the OpenAI-shape /v1/chat/completions + /v1/
    // completions paths the chat_completion / text_completions
    // handlers still 400 with a pointer to /v1/audio/transcriptions
    // because there's no input-bytes field there to carry the audio.)
    // (Parler / TTS are NOT rejected here - they're routed to
    // handle_chat_tts in the /api/chat + /api/generate handlers so the
    // GUI's "type text, get audio back" UX works directly through the
    // chat tab, exactly like image-gen models are routed through
    // handle_image_generation.)
    // mt5-tokenizers: tokenizer-only repo.
    if lower.contains("mt5-tokenizers") {
        return Some(
            "This is a tokenizer-only repo, not a model. \
             It can't serve chat requests on its own.",
        );
    }
    None
}

// ============================================================================
// Error Types
// ============================================================================

/// How long a request may wait for the single in-flight inference slot before the
/// server answers 503 instead of queueing forever behind a wedged generation.
const GATE_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// Acquire the request gate with a bounded wait; `Err(ApiError::Busy)` (-> 503) on timeout.
async fn acquire_gate(
    state: &APIServer,
    priority: crate::api::gate::Priority,
    model: String,
    endpoint: &'static str,
) -> Result<crate::api::gate::RequestGuard, ApiError> {
    state
        .request_gate
        .acquire_with_info_timeout(priority, model, endpoint, GATE_WAIT)
        .await
        .ok_or_else(|| {
            ApiError::Busy(format!(
                "no inference slot became available within {}s ({endpoint}); retry later",
                GATE_WAIT.as_secs()
            ))
        })
}

/// API error types
#[derive(thiserror::Error, Debug)]
pub enum ApiError {
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Server busy: {0}")]
    Busy(String),
}

/// Fallback 404 handler with logging
/// The discoverability list surfaced inside 404 error envelopes. Kept
/// as a pure helper so a unit test can assert against the same source
/// of truth without spinning up axum.
fn available_endpoints_list() -> serde_json::Value {
    serde_json::json!([
        // Ollama-style
        "GET /api/tags",
        "POST /api/pull",
        "DELETE /api/delete",
        "POST /api/show",
        "POST /api/chat",
        "POST /api/generate",
        "GET /api/ps",
        "POST /api/copy",
        "POST /api/embed",
        "POST /api/embeddings",
        "POST /api/create",
        "POST /api/push",
        "HEAD,POST /api/blobs/{digest}",
        "GET /api/models",
        "GET /api/models/loaded",
        "POST /api/models/validate",
        "POST /api/models/repair",
        "POST /api/swap",
        "POST /api/layers/swap",
        "POST /api/chat/completions",
        "GET /api/inflight",
        "GET /api/layer_perf",
        "POST /api/draft/attach",
        "POST /api/draft/detach",
        "GET /api/draft/status",
        // Distributed / multi-device
        "GET /api/distributed/devices",
        "GET /api/distributed/stats",
        "GET /api/distributed/recommend",
        "POST /api/multi-device/plan",
        "POST /api/multi-device/configure",
        "GET /api/multi-device/status",
        // OpenAI-style
        "GET /v1/models",
        "GET /v1/models/{model_id}",
        "POST /v1/chat/completions",
        "POST /v1/completions",
        "POST /v1/embeddings",
        "POST /v1/audio/transcriptions",
        "POST /v1/audio/translations",
        "POST /v1/audio/speech",
        "GET /v1/audio/voices",
        "POST /v1/images/generations",
        "POST /v1/images/edits",
        "POST /v1/images/variations",
        // Misc
        "GET /health",
        "GET /api/version"
    ])
}

async fn handle_404(uri: axum::http::Uri) -> impl IntoResponse {
    warn!("⚠️  404 Not Found: {} - endpoint does not exist", uri);

    let mut body = openai_error_body(StatusCode::NOT_FOUND, format!("Endpoint not found: {uri}"));
    // Augment the OpenAI error envelope with a discoverability hint
    // listing both the Ollama-shaped /api/* surface and the OpenAI-
    // shaped /v1/* surface this server exposes.
    if let Some(err) = body.get_mut("error").and_then(|v| v.as_object_mut()) {
        err.insert(
            "available_endpoints".to_string(),
            available_endpoints_list(),
        );
    }

    (StatusCode::NOT_FOUND, Json(body))
}

/// Drop-in replacement for `axum::Json` that rewrites deserialize
/// rejections into the OpenAI error envelope. The default `Json<T>`
/// extractor responds with 422 + a plain-text body - SDKs that match
/// on `error.message` can't see the failure.
pub struct OpenAIJson<T>(pub T);

impl<S, T> FromRequest<S> for OpenAIJson<T>
where
    S: Send + Sync,
    T: serde::de::DeserializeOwned,
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(body)) => Ok(Self(body)),
            Err(rej) => Err(ApiError::Validation(rej.body_text())),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ApiError::Validation(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::Busy(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
        };

        // Use the OpenAI error envelope shape so SDKs that match on
        // `error.message` / `error.type` (most of them) can parse our
        // failures consistently with their primary backend's.
        let body = Json(openai_error_body(status, error_message));

        (status, body).into_response()
    }
}

/// Validation helper. Strips the validator crate's verbose serialized
/// `{value, min, max}` JSON form and surfaces a human-readable
/// `"<field> must be in <range>"` instead, so clients see a useful
/// `error.message` rather than the raw debug repr.
pub fn validate_request<T: validator::Validate>(request: &T) -> Result<(), ApiError> {
    request
        .validate()
        .map_err(|e| ApiError::Validation(humanize_validation_error(&e)))
}

fn humanize_validation_error(errs: &validator::ValidationErrors) -> String {
    let mut parts: Vec<String> = Vec::new();
    collect_humanized(errs, "", &mut parts);
    if parts.is_empty() {
        "request validation failed".to_string()
    } else {
        parts.join("; ")
    }
}

/// Walks the validator's `ValidationErrors` tree, including nested
/// struct + list (`#[validate(nested)]`) errors, producing one
/// human-readable message per failure. `prefix` carries the
/// dotted path to the current scope (e.g. `messages[0].role`).
fn collect_humanized(errs: &validator::ValidationErrors, prefix: &str, parts: &mut Vec<String>) {
    use validator::ValidationErrorsKind;
    for (field, kind) in errs.errors() {
        let path = if prefix.is_empty() {
            field.to_string()
        } else {
            format!("{prefix}.{field}")
        };
        match kind {
            ValidationErrorsKind::Field(field_errs) => {
                for err in field_errs {
                    parts.push(humanize_one(&path, err));
                }
            }
            ValidationErrorsKind::Struct(nested) => {
                collect_humanized(nested, &path, parts);
            }
            ValidationErrorsKind::List(map) => {
                for (idx, nested) in map.iter() {
                    let indexed = format!("{path}[{idx}]");
                    collect_humanized(nested, &indexed, parts);
                }
            }
        }
    }
}

/// Map one validator error onto a human-readable string. Pulled out
/// of the walker so both flat and nested errors share the same
/// formatting logic.
fn humanize_one(field: &str, err: &validator::ValidationError) -> String {
    let code = err.code.as_ref();
    match code {
        "range" => {
            let min = err.params.get("min").map(std::string::ToString::to_string);
            let max = err.params.get("max").map(std::string::ToString::to_string);
            let val = err
                .params
                .get("value")
                .map(std::string::ToString::to_string);
            match (min, max, val) {
                (Some(lo), Some(hi), Some(v)) => {
                    format!("{field} must be in [{lo}, {hi}]; got {v}")
                }
                (Some(lo), None, Some(v)) => format!("{field} must be >= {lo}; got {v}"),
                (None, Some(hi), Some(v)) => format!("{field} must be <= {hi}; got {v}"),
                _ => format!("{field} out of range"),
            }
        }
        "length" => {
            let min = err.params.get("min").map(std::string::ToString::to_string);
            let max = err.params.get("max").map(std::string::ToString::to_string);
            let val = err
                .params
                .get("value")
                .and_then(|v| {
                    v.as_str()
                        .map(|s| s.chars().count())
                        .or_else(|| v.as_array().map(std::vec::Vec::len))
                })
                .map(|n| n.to_string());
            match (min.as_deref(), max.as_deref(), val.as_deref()) {
                (Some("1"), None, Some("0")) => format!("{field} must not be empty"),
                (Some(lo), Some(hi), Some(v)) => {
                    format!("{field} length must be in [{lo}, {hi}]; got {v}")
                }
                (Some(lo), None, Some(v)) => format!("{field} length must be >= {lo}; got {v}"),
                (None, Some(hi), Some(v)) => format!("{field} length must be <= {hi}; got {v}"),
                _ => format!("{field} length out of range"),
            }
        }
        other => format!("{field}: {other}"),
    }
}

#[cfg(test)]
mod panic_isolation_gate {
    /// The router's panic net only works if panics UNWIND.
    ///
    /// `panic = "abort"` in the release profile made `CatchPanicLayer` dead code: abort
    /// terminates the process without unwinding, so a panic in any handler - one bad
    /// index, one `unwrap` on a malformed request - killed the server and every other
    /// client's in-flight generation with it. Nothing pointed at it, because the layer
    /// was right there in the router and the comment beside it said it was covered.
    ///
    /// Read the config rather than test the behaviour: a test cannot observe its own
    /// profile's panic strategy, and the setting is the thing that regresses.
    #[test]
    fn the_release_profile_must_unwind_so_a_panic_stays_in_its_request() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .find(|p| p.join(".cargo/config.toml").exists())
            .expect("the workspace .cargo/config.toml must be findable from the crate");
        let cfg = std::fs::read_to_string(root.join(".cargo/config.toml")).unwrap();
        // Only look inside the release profile: a `panic` setting elsewhere is not this.
        let release = cfg
            .split("[profile.")
            .find(|s| s.starts_with("release]"))
            .expect("a [profile.release] section");
        let offending = release
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim())
            .find(|l| l.starts_with("panic") && l.contains("abort"));
        assert!(
            offending.is_none(),
            "the release profile sets `{}`, which disables the router's CatchPanicLayer \
             entirely - one panicking request would take the whole server down",
            offending.unwrap_or_default()
        );
    }
}

#[cfg(test)]
mod tests;

/// Renders in flight, by identifier.
async fn list_renders() -> impl IntoResponse {
    let ids = crate::inference::serve::cancel::registry::in_flight();
    Json(serde_json::json!({ "renders": ids }))
}

/// Stop the render under `id`.
///
/// `cancelled` is false when nothing is registered under it - already finished, or never
/// there. The caller cannot tell those apart and does not need to: either way there is
/// nothing left to stop, so this is a 200 rather than a 404.
async fn cancel_render(axum::extract::Path(id): axum::extract::Path<String>) -> impl IntoResponse {
    let hit = crate::inference::serve::cancel::registry::cancel(&id);
    Json(serde_json::json!({ "id": id, "cancelled": hit }))
}

/// What a clip of this length at this quality will be, and roughly how long it will take.
///
/// Exists so nobody has to answer for resolution, frame count, step count and checkpoint in
/// order to get a sensible render - and so the wait is stated before it is spent rather than
/// discovered. The estimate covers the DENOISE only, and says so in `basis`: loading and
/// decoding are real time too, and quietly folding a guess for them into one number would
/// make the whole thing less trustworthy, not more.
#[cfg(feature = "video")]
async fn plan_video(Json(req): Json<serde_json::Value>) -> impl IntoResponse {
    use crate::inference::place::video_plan::{plan, Quality};
    let seconds = req.get("seconds").and_then(|x| x.as_f64()).unwrap_or(5.0) as f32;
    let quality = match req
        .get("quality")
        .and_then(|x| x.as_str())
        .unwrap_or("standard")
    {
        "draft" => Quality::Draft,
        "fine" => Quality::Fine,
        _ => Quality::Standard,
    };
    // Which checkpoint runs changes the cost by more than two to one, so it is asked for
    // rather than assumed.
    let model = req.get("model").and_then(|x| x.as_str()).unwrap_or("");
    let wide = model.contains("14b");
    let chosen = plan(seconds, quality, wide);
    // An interface that shows its OWN controls is not asking to be planned for, it is
    // asking what its settings cost. Same arithmetic either way, so what a caller is shown
    // cannot drift from what the planner used.
    let num = |k: &str| req.get(k).and_then(|x| x.as_u64()).map(|v| v as usize);
    let (w, h) = (
        num("width").unwrap_or(chosen.width),
        num("height").unwrap_or(chosen.height),
    );
    let (frames, steps) = (
        num("frames").unwrap_or(chosen.frames),
        num("steps").unwrap_or(chosen.steps),
    );
    let (passes, est) = crate::inference::place::video_plan::estimate(w, h, frames, steps, wide);
    Json(serde_json::json!({
        "width": w, "height": h, "frames": frames, "steps": steps,
        "seconds": frames as f32 / 16.0,
        "passes": passes,
        "estimated_seconds": est,
        "basis": chosen.basis,
    }))
}

/// Whether this build carries a family, asked where a request is classified.
///
/// A request naming an image model on a text-only build must be told the model is unknown,
/// which is the truth, rather than dispatched to a handler that is not there. These shims
/// keep that decision at the classification site instead of scattering `cfg` through every
/// branch that follows it.
pub(crate) mod family {
    #[cfg(feature = "image")]
    pub(crate) fn is_image(model: &str) -> bool {
        super::media::is_image_gen_model(model)
    }
    #[cfg(not(feature = "image"))]
    pub(crate) fn is_image(_model: &str) -> bool {
        false
    }

    #[cfg(feature = "audio")]
    pub(crate) fn is_tts(model: &str) -> bool {
        super::is_tts_model(model)
    }
    #[cfg(not(feature = "audio"))]
    pub(crate) fn is_tts(_model: &str) -> bool {
        false
    }

    #[cfg(feature = "audio")]
    pub(crate) fn is_asr(model: &str) -> bool {
        super::is_asr_model(model)
    }
    #[cfg(not(feature = "audio"))]
    pub(crate) fn is_asr(_model: &str) -> bool {
        false
    }
}

#[cfg(test)]
mod relay_terminator_tests {
    /// The relay decides a forwarded stream ended by matching bytes, not by parsing every
    /// chunk. That is only sound if generated text cannot produce those bytes - and it cannot,
    /// because a quote inside a JSON string is escaped on the way in. If this ever fails, a
    /// model could end its own stream early by talking about the protocol.
    #[test]
    fn generated_text_cannot_forge_the_terminal_marker() {
        let hostile = serde_json::json!({
            "response": r#"the field is "done":true when finished"#,
            "done": false,
        })
        .to_string();
        assert!(
            !hostile
                .as_bytes()
                .windows(super::TERMINAL_MARKER.len())
                .any(|w| w == super::TERMINAL_MARKER),
            "text describing the marker was taken for the marker: {hostile}"
        );
        // And the real thing still matches, or the relay would append a second ending to
        // every finished answer.
        let genuine = serde_json::json!({ "response": "", "done": true }).to_string();
        assert!(genuine
            .as_bytes()
            .windows(super::TERMINAL_MARKER.len())
            .any(|w| w == super::TERMINAL_MARKER));
    }
}
