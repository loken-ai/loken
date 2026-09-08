//! The catalogue of the whole cluster, as an entry point lists it.
//!
//! A request naming a model this node does not hold is forwarded to a peer that does, so the
//! list a client reads has to name those models too, or the client never asks. Each peer is
//! asked for its own list, and the answer is merged under the peer's id.

use super::APIServer;
use crate::api::types::{OllamaListModelsResponse, OllamaModel};
use crate::distributed::cluster::FORWARDED_HEADER;

/// How long a peer has to answer for its catalogue: a first listing reads every header of a
/// large store. A peer that takes longer is left out of this answer rather than delaying it;
/// the next call asks again.
const PEER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// The models alive peers hold, each carrying the id of the node that holds it. Empty when
/// the node is not clustered, and when the request came from a peer asking for this node's own
/// list, so two nodes never ask each other in a loop.
pub(crate) async fn peer_models(
    state: &APIServer,
    headers: &axum::http::HeaderMap,
) -> Vec<OllamaModel> {
    if headers.contains_key(FORWARDED_HEADER) {
        return Vec::new();
    }
    let Some(cluster) = state.cluster_handle() else {
        return Vec::new();
    };
    let now = crate::distributed::cluster_runtime::now_ms(state.cluster_started());
    let peers: Vec<(String, String)> = cluster
        .peer_view(now)
        .into_iter()
        .filter(|p| !p.is_self && p.alive)
        .filter_map(|p| p.endpoint.map(|e| (p.node_id, e)))
        .collect();
    if peers.is_empty() {
        return Vec::new();
    }
    let client = match reqwest::Client::builder().timeout(PEER_TIMEOUT).build() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let asks = peers.into_iter().map(|(node, endpoint)| {
        let client = client.clone();
        async move {
            let url = format!("{}/api/tags", endpoint.trim_end_matches('/'));
            let answer = client
                .get(&url)
                .header(FORWARDED_HEADER, "1")
                .send()
                .await
                .ok()?
                .json::<OllamaListModelsResponse>()
                .await
                .ok()?;
            Some((node, answer.models))
        }
    });
    let answers = futures::future::join_all(asks).await;
    let mut out = Vec::new();
    for (node, models) in answers.into_iter().flatten() {
        for mut m in models {
            m.node = Some(node.clone());
            out.push(m);
        }
    }
    out
}

/// The local list followed by what the peers hold and this node does not, in name order
/// among the peers' entries. A model held here is listed once, as local, whatever the peers
/// also hold.
pub(crate) fn merge(local: Vec<OllamaModel>, peers: Vec<OllamaModel>) -> Vec<OllamaModel> {
    let mut seen: std::collections::HashSet<String> =
        local.iter().map(|m| m.name.clone()).collect();
    let mut remote: Vec<OllamaModel> = peers
        .into_iter()
        .filter(|m| seen.insert(m.name.clone()))
        .collect();
    remote.sort_by(|a, b| a.name.cmp(&b.name));
    let mut out = local;
    out.extend(remote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, node: Option<&str>) -> OllamaModel {
        let mut m = OllamaModel::new(name.to_string(), 1, "2026-01-01T00:00:00Z".to_string());
        m.node = node.map(str::to_string);
        m
    }

    /// A model this node holds is listed once and as local; a peer's other models follow,
    /// each under the node that holds it, and a model two peers hold appears once.
    #[test]
    fn a_peer_adds_only_what_this_node_lacks() {
        let local = vec![entry("qwen3:0.6b", None)];
        let peers = vec![
            entry("qwen3:0.6b", Some("laptop")),
            entry("qwen3:8b", Some("desktop")),
            entry("llama3.2:1b", Some("laptop")),
            entry("qwen3:8b", Some("laptop")),
        ];
        let merged = merge(local, peers);
        let names: Vec<(&str, Option<&str>)> = merged
            .iter()
            .map(|m| (m.name.as_str(), m.node.as_deref()))
            .collect();
        assert_eq!(
            names,
            vec![
                ("qwen3:0.6b", None),
                ("llama3.2:1b", Some("laptop")),
                ("qwen3:8b", Some("desktop")),
            ]
        );
    }
}
