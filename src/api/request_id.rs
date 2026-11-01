//! The identifier a client gives a request in `X-Request-Id`.
//!
//! The server mints one when the client sends none, so every request has one.
//! A render registers under it, and the client stops the render by naming it.
//! Only an identifier of letters, digits, dot, underscore and dash, 64 characters at most,
//! names a render. The task-local is dropped when a response stream is polled, so a handler
//! reads it before it returns the stream.

tokio::task_local! {
    static CURRENT: Option<String>;
}

const MAX_LEN: usize = 64;

/// Whether `id` may name a render.
pub fn valid(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// The valid identifier of the request being served.
pub fn current() -> Option<String> {
    CURRENT.try_with(Clone::clone).ok().flatten()
}

/// A fresh identifier for a render of a request that named none.
///
/// Monotonic within the process, which is all the registry needs: it exists so a client can
/// name the work it started and stop it, not to be meaningful across restarts.
pub fn next_render_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    format!("r{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

/// The identifier a render of the request being served registers under: the client's
/// `X-Request-Id` when it is valid, else a fresh one.
pub fn render_id() -> String {
    current().unwrap_or_else(next_render_id)
}

/// Serves each request with its identifier published, when it is valid.
pub async fn scope_request_id(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| valid(v))
        .map(str::to_string);
    CURRENT.scope(id, next.run(req)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name of letters, digits, dot, underscore and dash up to the limit is valid.
    /// Anything else is not.
    #[test]
    fn only_a_plain_short_name_is_valid() {
        assert!(valid("req-7f.a_b"));
        assert!(valid(&"a".repeat(MAX_LEN)));
        assert!(!valid(&"a".repeat(MAX_LEN + 1)));
        assert!(!valid(""));
        assert!(!valid("a b"));
        assert!(!valid("a/b"));
        assert!(!valid("é"));
    }

    /// The middleware publishes a valid header and nothing for an invalid or missing one.
    #[tokio::test]
    async fn the_middleware_publishes_a_valid_identifier() {
        use tower::ServiceExt;
        let app = || {
            axum::Router::new()
                .route(
                    "/probe",
                    axum::routing::get(|| async { format!("{:?}", current()) }),
                )
                .layer(axum::middleware::from_fn(scope_request_id))
        };
        let ask = |id: Option<&'static str>| async move {
            let mut request = axum::http::Request::builder().uri("/probe");
            if let Some(id) = id {
                request = request.header("x-request-id", id);
            }
            let response = app()
                .oneshot(request.body(axum::body::Body::empty()).unwrap())
                .await
                .unwrap();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 10)
                .await
                .unwrap();
            String::from_utf8_lossy(&bytes).to_string()
        };
        assert_eq!(ask(Some("req-1")).await, "Some(\"req-1\")");
        assert_eq!(ask(Some("bad id")).await, "None");
        assert_eq!(ask(None).await, "None");
    }

    /// A render that carries an identifier is stopped by cancelling that identifier.
    /// The loop on the blocking thread polls the token the request published.
    /// A request with no identifier gets a fresh one and is not reachable by a client name.
    #[tokio::test]
    async fn a_render_stops_when_its_request_id_is_cancelled() {
        use crate::inference::serve::cancel::{registry, scoped, CancelToken};
        use tower::ServiceExt;
        async fn work() -> String {
            let token = CancelToken::new();
            let id = current().unwrap_or_else(|| "none".to_string());
            let _registered = registry::Entry::new(
                crate::inference::engine::cache_scope::CacheScope::current(),
                &id,
                &token,
            );
            tokio::task::spawn_blocking(move || {
                let _stop = scoped::publish(&token);
                for _ in 0..400 {
                    if scoped::bail().is_err() {
                        return "stopped";
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                "finished"
            })
            .await
            .unwrap()
            .to_string()
        }
        async fn cancel(axum::extract::Path(id): axum::extract::Path<String>) -> String {
            registry::cancel(&id).to_string()
        }
        let app = axum::Router::new()
            .route("/work", axum::routing::get(work))
            .route("/cancel/{id}", axum::routing::post(cancel))
            .layer(axum::middleware::from_fn(scope_request_id));
        let ask = |app: axum::Router,
                   method: &'static str,
                   uri: &'static str,
                   id: Option<&'static str>| async move {
            let mut request = axum::http::Request::builder().method(method).uri(uri);
            if let Some(id) = id {
                request = request.header("x-request-id", id);
            }
            let response = app
                .oneshot(request.body(axum::body::Body::empty()).unwrap())
                .await
                .unwrap();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 10)
                .await
                .unwrap();
            String::from_utf8_lossy(&bytes).to_string()
        };
        let running = tokio::spawn(ask(app.clone(), "GET", "/work", Some("job-1")));
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            ask(app.clone(), "POST", "/cancel/other", None).await,
            "false"
        );
        assert_eq!(ask(app, "POST", "/cancel/job-1", None).await, "true");
        assert_eq!(running.await.unwrap(), "stopped");
    }
}
