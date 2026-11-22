//! Copying a model from a peer that holds it.
//!
//! A peer serves the manifest of each Ollama model it lists, and the blobs those manifests name.
//! A blob is served with Range support, so a copy that was cut resumes where it stopped.
//! A node copies a model blob by blob, checks each against its digest, then writes the manifest.

use anyhow::{anyhow, Result};

use super::cluster::PeerView;
use crate::inference::load::ollama_manager::{OllamaManager, OllamaManifest, ProgressCallback};

/// The peer route that answers the manifest of a model the peer lists.
pub const MANIFEST_ROUTE: &str = "/api/cluster/manifest";

/// The peer route that serves a blob by its digest, as `<route>/<digest>`.
pub const BLOB_ROUTE: &str = "/api/cluster/blobs";

/// The endpoints of the live peers whose catalogue lists `model_id`.
///
/// The peer with the shortest measured round trip comes first.
/// A peer without a measured round trip comes after those with one.
/// Peers that tie keep the order of the peer view.
pub fn holders(peers: &[PeerView], model_id: &str) -> Vec<String> {
    let mut found: Vec<&PeerView> = peers
        .iter()
        .filter(|p| {
            !p.is_self
                && p.alive
                && p.endpoint.is_some()
                && p.state
                    .serves
                    .as_ref()
                    .is_some_and(|listed| listed.iter().any(|m| m == model_id))
        })
        .collect();
    found.sort_by(|a, b| {
        let rtt = |p: &PeerView| p.rtt_ms.unwrap_or(f64::INFINITY);
        rtt(a).total_cmp(&rtt(b))
    });
    found
        .into_iter()
        .filter_map(|p| p.endpoint.clone())
        .collect()
}

/// `name:tag` of an Ollama model id, `latest` when no tag is given.
fn name_and_tag(model_id: &str) -> (&str, &str) {
    match model_id.rsplit_once(':') {
        Some((name, tag)) if !tag.contains('/') => (name, tag),
        _ => (model_id, "latest"),
    }
}

/// The manifest of `model_id` when `store` lists it.
pub fn listed_manifest(store: &OllamaManager, model_id: &str) -> Option<serde_json::Value> {
    let (name, tag) = name_and_tag(model_id);
    let path = store.get_model_path(name, tag)?;
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// Answers `request` with the blob `digest`, when a model `store` lists names it.
///
/// A Range header is honoured: the answer is `206` with the bytes asked for.
/// A digest that is not `sha256:` and 64 hex digits, or that no listed manifest names,
/// answers `404`.
pub async fn serve_blob(
    store: &OllamaManager,
    digest: &str,
    request: axum::extract::Request,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use tower::ServiceExt;
    let Some(path) = store.listed_blob(digest) else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            "no model this node lists holds this blob",
        )
            .into_response();
    };
    match tower_http::services::ServeFile::new(path)
        .oneshot(request)
        .await
    {
        Ok(answer) => answer.map(axum::body::Body::new),
        Err(never) => match never {},
    }
}

/// Copies the model `name:tag` from the peer at `peer` into `store`.
///
/// The peer answers the manifest, then each blob the store lacks.
/// Each blob is checked against its digest before it enters the store.
/// The manifest is written once every blob is in.
pub async fn copy_from_peer(
    store: &OllamaManager,
    name: &str,
    tag: &str,
    peer: &str,
    progress: Option<ProgressCallback>,
) -> Result<()> {
    let base = peer.trim_end_matches('/');
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}{MANIFEST_ROUTE}"))
        .json(&serde_json::json!({ "model": format!("{name}:{tag}") }))
        .send()
        .await
        .map_err(|e| anyhow!("{peer}: manifest of {name}:{tag} not fetched: {e}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!(
            "{peer}: manifest of {name}:{tag} answered {}",
            resp.status()
        ));
    }
    let manifest: OllamaManifest = resp
        .json()
        .await
        .map_err(|e| anyhow!("{peer}: manifest of {name}:{tag} unreadable: {e}"))?;
    let blob_url = |digest: &str| format!("{base}{BLOB_ROUTE}/{digest}");
    store
        .store_pulled(name, tag, &manifest, &client, blob_url, progress)
        .await
}

/// Copies the model `name:tag` into `store` from the first of `peers` that completes it.
///
/// A peer that fails leaves the next one to try.
/// A blob cut short stays as a partial file, which the next peer resumes.
/// A blob with the wrong digest is deleted, and the next peer sends it again.
/// Returns the peer the copy came from.
pub async fn copy_from_peers(
    store: &OllamaManager,
    name: &str,
    tag: &str,
    peers: &[String],
    progress: Option<ProgressCallback>,
) -> Result<String> {
    let mut failures = Vec::new();
    for peer in peers {
        match copy_from_peer(store, name, tag, peer, progress.clone()).await {
            Ok(()) => {
                tracing::info!("cluster: {name}:{tag} copied from {peer}");
                return Ok(peer.clone());
            }
            Err(e) => {
                tracing::warn!("cluster: copy of {name}:{tag} from {peer} failed: {e}");
                failures.push(e.to_string());
            }
        }
    }
    Err(anyhow!(
        "no peer completed the copy of {name}:{tag}: {}",
        if failures.is_empty() {
            "no peer lists it".to_string()
        } else {
            failures.join("; ")
        }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::membership::NodeState;
    use sha2::Digest;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    const NAME: &str = "tiny";
    const TAG: &str = "1b";

    fn digest_of(bytes: &[u8]) -> String {
        format!("sha256:{}", hex::encode(sha2::Sha256::digest(bytes)))
    }

    /// Weights large enough to arrive in several chunks.
    fn weights() -> Vec<u8> {
        (0..u32::from(u16::MAX) * 4)
            .map(|i| (i % 251) as u8)
            .collect()
    }

    /// Writes `NAME:TAG` in the store at `dir`: a config blob and a weights blob holding
    /// `stored`, named as the manifest names them, `digest_of(weights)` for the weights.
    fn put_model(dir: &Path, weights: &[u8], stored: &[u8]) -> (String, String) {
        let config = br#"{"model_format":"gguf"}"#.to_vec();
        let (config_digest, weights_digest) = (digest_of(&config), digest_of(weights));
        let blobs = dir.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join(config_digest.replace(':', "-")), &config).unwrap();
        std::fs::write(blobs.join(weights_digest.replace(':', "-")), stored).unwrap();
        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "config": {"digest": config_digest, "size": config.len()},
            "layers": [{
                "mediaType": "application/vnd.ollama.image.model",
                "digest": weights_digest,
                "size": weights.len(),
            }],
        });
        OllamaManager::with_dirs(vec![dir.to_path_buf()])
            .write_manifest(NAME, TAG, &manifest)
            .unwrap();
        (config_digest, weights_digest)
    }

    fn scratch(tag: &str) -> PathBuf {
        crate::inference::load::model_dirs::tests::scratch(&format!("model-copy-{tag}"))
    }

    /// Serves `app` on a loopback port and answers its URL.
    async fn serve(app: axum::Router) -> String {
        let port = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", port.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(port, app).await.unwrap() });
        url
    }

    /// A node serving the Ollama store at `dir` with its full router.
    async fn peer(dir: &Path, hf: &Path) -> String {
        let state = crate::api::handlers::APIServer::new(
            dir.to_string_lossy().into_owned(),
            hf.to_string_lossy().into_owned(),
        );
        serve(state.create_router()).await
    }

    fn partials(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir.join("blobs"))
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "partial"))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// A node copies a model from a peer: each blob arrives whole under its digest and the
    /// manifest lists the model.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_is_copied_from_a_peer_with_its_digests() {
        let (held, empty, hf) = (scratch("held"), scratch("empty"), scratch("hf"));
        let weights = weights();
        let (config_digest, weights_digest) = put_model(&held, &weights, &weights);
        let url = peer(&held, &hf).await;
        let node = OllamaManager::with_dirs(vec![empty.clone()]);

        let from = copy_from_peers(&node, NAME, TAG, std::slice::from_ref(&url), None)
            .await
            .unwrap();

        assert_eq!(from, url);
        let blob = |d: &str| empty.join("blobs").join(d.replace(':', "-"));
        assert_eq!(std::fs::read(blob(&weights_digest)).unwrap(), weights);
        assert!(blob(&config_digest).is_file());
        assert!(node.get_model_path(NAME, TAG).is_some());
        assert_eq!(
            listed_manifest(&node, &format!("{NAME}:{TAG}")),
            listed_manifest(&OllamaManager::with_dirs(vec![held.clone()]), "tiny:1b")
        );
        assert!(partials(&empty).is_empty());
        for d in [held, empty, hf] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// A copy whose connection drops in the middle of a blob asks for the rest with a Range
    /// request and completes the blob.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_cut_copy_resumes_with_a_range_request() {
        let (held, empty) = (scratch("cut-held"), scratch("cut-empty"));
        let weights = weights();
        let (_, weights_digest) = put_model(&held, &weights, &weights);
        let store = Arc::new(OllamaManager::with_dirs(vec![held.clone()]));
        let ranges: Arc<Mutex<Vec<Option<String>>>> = Arc::default();
        let cuts = Arc::new(AtomicUsize::new(0));
        let half = weights.len() / 2;

        let manifest_store = store.clone();
        let manifest = move |axum::Json(req): axum::Json<serde_json::Value>| {
            let store = manifest_store.clone();
            async move {
                let model = req["model"].as_str().unwrap_or_default().to_string();
                axum::Json(listed_manifest(&store, &model).unwrap())
            }
        };
        let (blob_store, seen, cut, first_half) = (
            store.clone(),
            ranges.clone(),
            cuts.clone(),
            weights[..half].to_vec(),
        );
        let blob = move |axum::extract::Path(digest): axum::extract::Path<String>,
                         request: axum::extract::Request| {
            let (store, seen, cut, first_half) = (
                blob_store.clone(),
                seen.clone(),
                cut.clone(),
                first_half.clone(),
            );
            let target = weights_digest.clone();
            async move {
                if digest == target {
                    seen.lock().unwrap().push(
                        request
                            .headers()
                            .get(axum::http::header::RANGE)
                            .map(|v| v.to_str().unwrap().to_string()),
                    );
                }
                if digest != target || cut.fetch_add(1, Ordering::SeqCst) > 0 {
                    return serve_blob(&store, &digest, request).await;
                }
                // The first answer for the weights sends half the bytes, then drops.
                let total = first_half.len() * 2;
                let body = async_stream::stream! {
                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(first_half));
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    yield Err(std::io::Error::other("connection cut"));
                };
                axum::response::Response::builder()
                    .header(axum::http::header::CONTENT_LENGTH, total)
                    .body(axum::body::Body::from_stream(body))
                    .unwrap()
            }
        };
        let app = axum::Router::new()
            .route(MANIFEST_ROUTE, axum::routing::post(manifest))
            .route(
                &format!("{BLOB_ROUTE}/{{digest}}"),
                axum::routing::get(blob),
            );
        let url = serve(app).await;
        let node = OllamaManager::with_dirs(vec![empty.clone()]);

        copy_from_peers(&node, NAME, TAG, &[url], None)
            .await
            .unwrap();

        let seen = ranges.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "the weights were asked {} times", seen.len());
        assert_eq!(seen[0], None);
        let resumed_at: usize = seen[1]
            .as_deref()
            .and_then(|r| r.strip_prefix("bytes="))
            .and_then(|r| r.strip_suffix('-'))
            .and_then(|r| r.parse().ok())
            .expect("the second request asks a range from an offset");
        assert!(resumed_at > 0 && resumed_at <= half);
        let blob = empty
            .join("blobs")
            .join(digest_of(&weights).replace(':', "-"));
        assert_eq!(std::fs::read(blob).unwrap(), weights);
        assert!(partials(&empty).is_empty());
        for d in [held, empty] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// A blob whose bytes do not match its digest is deleted and fails the copy from that
    /// peer: the model does not enter the store, and the next peer completes it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_blob_with_the_wrong_digest_is_rejected() {
        let (bad, good, empty, hf) = (
            scratch("bad"),
            scratch("good"),
            scratch("mismatch-empty"),
            scratch("mismatch-hf"),
        );
        let weights = weights();
        let mut damaged = weights.clone();
        damaged[weights.len() / 3] ^= 1;
        let (_, weights_digest) = put_model(&bad, &weights, &damaged);
        put_model(&good, &weights, &weights);
        let (bad_url, good_url) = (peer(&bad, &hf).await, peer(&good, &hf).await);
        let node = OllamaManager::with_dirs(vec![empty.clone()]);
        let blob = empty.join("blobs").join(weights_digest.replace(':', "-"));

        let refused = copy_from_peers(&node, NAME, TAG, std::slice::from_ref(&bad_url), None).await;
        assert!(refused.is_err());
        assert!(!blob.exists());
        assert!(partials(&empty).is_empty());
        assert!(node.get_model_path(NAME, TAG).is_none());

        let from = copy_from_peers(&node, NAME, TAG, &[bad_url, good_url.clone()], None)
            .await
            .unwrap();
        assert_eq!(from, good_url);
        assert_eq!(std::fs::read(&blob).unwrap(), weights);
        assert!(node.get_model_path(NAME, TAG).is_some());
        for d in [bad, good, empty, hf] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// The blob route serves the blobs of listed models, a range of them on request, and
    /// refuses a path, an unlisted file and a model it does not list.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_blob_route_serves_only_listed_blobs() {
        let (held, hf) = (scratch("route-held"), scratch("route-hf"));
        let weights = weights();
        let (_, weights_digest) = put_model(&held, &weights, &weights);
        let stray = b"not named by any manifest".to_vec();
        std::fs::write(
            held.join("blobs").join(digest_of(&stray).replace(':', "-")),
            &stray,
        )
        .unwrap();
        std::fs::write(held.join("secret"), b"outside the blobs").unwrap();
        let url = peer(&held, &hf).await;
        let client = reqwest::Client::new();
        let get = |path: String| client.get(format!("{url}{BLOB_ROUTE}/{path}"));

        let whole = get(weights_digest.clone()).send().await.unwrap();
        assert_eq!(whole.status(), reqwest::StatusCode::OK);
        assert_eq!(whole.bytes().await.unwrap().as_ref(), weights.as_slice());
        let tail = get(weights_digest.clone())
            .header(reqwest::header::RANGE, "bytes=10-")
            .send()
            .await
            .unwrap();
        assert_eq!(tail.status(), reqwest::StatusCode::PARTIAL_CONTENT);
        assert_eq!(tail.bytes().await.unwrap().as_ref(), &weights[10..]);

        for refused in [
            "..%2Fsecret".to_string(),
            "sha256-..%2F..%2Fsecret".to_string(),
            "sha256:..%2Fsecret".to_string(),
            "%2Fetc%2Fpasswd".to_string(),
            digest_of(&stray),
            weights_digest.replace(':', "-"),
        ] {
            let answer = get(refused.clone()).send().await.unwrap();
            assert!(
                !answer.status().is_success(),
                "{refused} answered {}",
                answer.status()
            );
        }

        let manifest = |model: &str| {
            client
                .post(format!("{url}{MANIFEST_ROUTE}"))
                .json(&serde_json::json!({ "model": model }))
                .send()
        };
        assert!(manifest("tiny:1b").await.unwrap().status().is_success());
        assert_eq!(
            manifest("absent:1b").await.unwrap().status(),
            reqwest::StatusCode::NOT_FOUND
        );
        assert!(!manifest("../tiny:1b").await.unwrap().status().is_success());
        for d in [held, hf] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    fn view(id: &str, alive: bool, rtt_ms: Option<f64>, serves: Option<&[&str]>) -> PeerView {
        PeerView {
            node_id: id.to_string(),
            endpoint: Some(format!("http://{id}")),
            alive,
            phi: None,
            rtt_ms,
            is_self: false,
            state: NodeState {
                serves: serves.map(|s| s.iter().map(|m| m.to_string()).collect()),
                ..NodeState::default()
            },
        }
    }

    /// The holders are the live peers listing the model, the shortest round trip first, the
    /// peers without a round trip last in their own order.
    #[test]
    fn holders_are_the_live_peers_listing_the_model_nearest_first() {
        let model = "tiny:1b";
        let mut me = view("me", true, Some(0.0), Some(&[model]));
        me.is_self = true;
        let peers = [
            me,
            view("a", true, None, Some(&[model])),
            view("b", true, Some(9.0), Some(&[model])),
            view("c", true, Some(2.0), Some(&[model, "other:1b"])),
            view("dead", false, Some(1.0), Some(&[model])),
            view("other", true, Some(1.0), Some(&["other:1b"])),
            view("silent", true, Some(1.0), None),
            view("d", true, None, Some(&[model])),
        ];
        assert_eq!(
            holders(&peers, model),
            ["http://c", "http://b", "http://a", "http://d"]
        );
        assert!(holders(&peers, "absent:1b").is_empty());
    }
}
