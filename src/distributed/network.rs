//! Network communication for distributed inference
//!
//! Provides client and server components for layer-to-layer communication
// Written and reached by nothing yet; docs/STATUS.md lists it under that heading.
#![allow(dead_code)]

use crate::distributed::protocol::{
    DeviceInfo, Heartbeat, LayerRequest, LayerResponse, ServerRegistration,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Default port for distributed inference
pub const DEFAULT_PORT: u16 = 8765;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server ID (unique identifier)
    pub server_id: String,
    /// Listen address (e.g., "0.0.0.0:8765")
    pub listen_addr: String,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Heartbeat interval in seconds
    pub heartbeat_interval_secs: u64,
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server_id: uuid::Uuid::new_v4().to_string(),
            listen_addr: format!("0.0.0.0:{}", DEFAULT_PORT),
            max_connections: 100,
            heartbeat_interval_secs: 30,
            request_timeout_ms: 30000,
        }
    }
}

impl ServerConfig {
    /// Create a new server configuration
    pub fn new(server_id: String, port: u16) -> Self {
        Self {
            server_id,
            listen_addr: format!("0.0.0.0:{}", port),
            ..Default::default()
        }
    }
}

/// Network server for handling layer requests
pub struct NetworkServer {
    config: ServerConfig,
    clients: Arc<RwLock<HashMap<String, ClientInfo>>>,
    request_handler: Option<RequestHandler>,
}

/// Client connection information
#[derive(Debug, Clone)]
struct ClientInfo {
    addr: SocketAddr,
    connected_at: Instant,
    last_heartbeat: Instant,
    load: f32,
}

/// Request handler type
type RequestHandler = Arc<dyn Fn(LayerRequest) -> Result<LayerResponse> + Send + Sync>;

impl NetworkServer {
    /// Create a new network server
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            clients: Arc::new(RwLock::new(HashMap::new())),
            request_handler: None,
        }
    }

    /// Start the server (simplified - would use axum or similar)
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting distributed inference server on {}",
            self.config.listen_addr
        );
        info!("Server ID: {}", self.config.server_id);

        // In a real implementation, this would start an HTTP/WebSocket server
        // For now, we'll just log the configuration

        Ok(())
    }

    /// Get server registration info
    pub fn registration(&self, devices: Vec<DeviceInfo>) -> ServerRegistration {
        let max_memory = devices.iter().map(|d| d.memory_bytes).sum();
        ServerRegistration {
            server_id: self.config.server_id.clone(),
            endpoint: self.config.listen_addr.clone(),
            devices,
            max_memory,
            priority: 50,
        }
    }

    /// Get number of connected clients
    pub async fn client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    /// Handle incoming heartbeat
    pub async fn handle_heartbeat(&self, heartbeat: Heartbeat) {
        let mut clients = self.clients.write().await;
        if let Some(client) = clients.get_mut(&heartbeat.server_id) {
            client.last_heartbeat = Instant::now();
            client.load = heartbeat.load;
        }
    }

    /// Clean up stale clients (no heartbeat for > 2x interval)
    pub async fn cleanup_stale_clients(&self) {
        let threshold = Duration::from_secs(self.config.heartbeat_interval_secs * 2);
        let mut clients = self.clients.write().await;

        let stale: Vec<String> = clients
            .iter()
            .filter(|(_, info)| info.last_heartbeat.elapsed() > threshold)
            .map(|(id, _)| id.clone())
            .collect();

        for id in stale {
            info!("Removing stale client: {}", id);
            clients.remove(&id);
        }
    }
}

/// Network client for connecting to remote servers
#[derive(Clone)]
pub struct NetworkClient {
    /// HTTP client for requests
    http_client: reqwest::Client,
    /// Server endpoint
    endpoint: String,
    /// Connection timeout
    timeout: Duration,
    /// Request latency tracking
    latency_stats: Arc<tokio::sync::Mutex<LatencyStats>>,
}

/// Latency statistics
#[derive(Debug, Default)]
struct LatencyStats {
    total_requests: u64,
    total_latency_ms: f64,
    max_latency_ms: f64,
    min_latency_ms: f64,
}

impl NetworkClient {
    /// Create a new network client
    pub fn new(endpoint: String) -> Self {
        Self {
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            endpoint,
            timeout: Duration::from_secs(30),
            latency_stats: Arc::new(tokio::sync::Mutex::new(LatencyStats::default())),
        }
    }

    /// Send a layer request to the server
    pub async fn send_layer_request(&mut self, request: LayerRequest) -> Result<LayerResponse> {
        let start = Instant::now();

        let url = format!("http://{}/layer/compute", self.endpoint);

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;

        let layer_response: LayerResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse response: {}", e))?;

        // Update latency stats
        let latency_ms = start.elapsed().as_millis() as f64;
        let mut stats = self.latency_stats.lock().await;
        stats.total_requests += 1;
        stats.total_latency_ms += latency_ms;
        stats.max_latency_ms = stats.max_latency_ms.max(latency_ms);
        if stats.min_latency_ms == 0.0 {
            stats.min_latency_ms = latency_ms;
        } else {
            stats.min_latency_ms = stats.min_latency_ms.min(latency_ms);
        }

        Ok(layer_response)
    }

    /// Register this client with a server
    pub async fn register(&self, registration: ServerRegistration) -> Result<()> {
        let url = format!("http://{}/register", self.endpoint);

        let response = self
            .http_client
            .post(&url)
            .json(&registration)
            .send()
            .await
            .map_err(|e| anyhow!("Registration failed: {}", e))?;

        if response.status().is_success() {
            info!("Successfully registered with server at {}", self.endpoint);
            Ok(())
        } else {
            Err(anyhow!(
                "Registration failed with status: {}",
                response.status()
            ))
        }
    }

    /// Send heartbeat to server
    pub async fn send_heartbeat(&self, heartbeat: Heartbeat) -> Result<()> {
        let url = format!("http://{}/heartbeat", self.endpoint);

        let response = self
            .http_client
            .post(&url)
            .json(&heartbeat)
            .send()
            .await
            .map_err(|e| anyhow!("Heartbeat failed: {}", e))?;

        if response.status().is_success() {
            Ok(())
        } else {
            warn!("Heartbeat failed with status: {}", response.status());
            Ok(()) // Non-fatal
        }
    }

    /// Get average latency
    pub async fn average_latency_ms(&self) -> f64 {
        let stats = self.latency_stats.lock().await;
        if stats.total_requests == 0 {
            0.0
        } else {
            stats.total_latency_ms / stats.total_requests as f64
        }
    }

    /// Get latency statistics
    pub async fn latency_stats(&self) -> (f64, f64, f64) {
        let stats = self.latency_stats.lock().await;
        (
            stats.min_latency_ms,
            if stats.total_requests == 0 {
                0.0
            } else {
                stats.total_latency_ms / stats.total_requests as f64
            },
            stats.max_latency_ms,
        )
    }

    /// Ping the server to check connectivity
    pub async fn ping(&self) -> Result<Duration> {
        let start = Instant::now();
        let url = format!("http://{}/ping", self.endpoint);

        let response = self
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| anyhow!("Ping failed: {}", e))?;

        if response.status().is_success() {
            Ok(start.elapsed())
        } else {
            Err(anyhow!("Ping failed with status: {}", response.status()))
        }
    }
}

/// Connection pool for multiple servers
pub struct ConnectionPool {
    connections: Arc<RwLock<HashMap<String, NetworkClient>>>,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a connection to the pool
    pub async fn add_connection(&self, server_id: String, endpoint: String) {
        let client = NetworkClient::new(endpoint);
        self.connections.write().await.insert(server_id, client);
    }

    /// Get a connection from the pool
    pub async fn get_connection(&self, server_id: &str) -> Option<NetworkClient> {
        self.connections.read().await.get(server_id).cloned()
    }

    /// Remove a connection from the pool
    pub async fn remove_connection(&self, server_id: &str) {
        self.connections.write().await.remove(server_id);
    }

    /// Get all connected server IDs
    pub async fn connected_servers(&self) -> Vec<String> {
        self.connections.read().await.keys().cloned().collect()
    }

    /// Send request to a specific server
    pub async fn send_request(
        &self,
        server_id: &str,
        request: LayerRequest,
    ) -> Result<LayerResponse> {
        let mut connections = self.connections.write().await;

        let client = connections
            .get_mut(server_id)
            .ok_or_else(|| anyhow!("Server {} not found in connection pool", server_id))?;

        client.send_layer_request(request).await
    }

    /// Broadcast heartbeat to all servers
    pub async fn broadcast_heartbeat(&self, heartbeat: Heartbeat) {
        let connections = self.connections.read().await;

        for (server_id, client) in connections.iter() {
            let mut heartbeat = heartbeat.clone();
            heartbeat.server_id = server_id.clone();

            if let Err(e) = client.send_heartbeat(heartbeat).await {
                warn!("Failed to send heartbeat to {}: {}", server_id, e);
            }
        }
    }
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert!(!config.server_id.is_empty());
        assert!(config.listen_addr.contains(&DEFAULT_PORT.to_string()));
    }

    #[test]
    fn test_server_config_custom() {
        let config = ServerConfig::new("test-server".to_string(), 9000);
        assert_eq!(config.server_id, "test-server");
        assert!(config.listen_addr.contains("9000"));
    }

    #[test]
    fn test_network_client_creation() {
        let client = NetworkClient::new("localhost:8765".to_string());
        assert_eq!(client.endpoint, "localhost:8765");
    }

    #[tokio::test]
    async fn test_connection_pool() {
        let pool = ConnectionPool::new();

        pool.add_connection("server1".to_string(), "localhost:8765".to_string())
            .await;

        let servers = pool.connected_servers().await;
        assert_eq!(servers.len(), 1);
        assert!(servers.contains(&"server1".to_string()));

        pool.remove_connection("server1").await;

        let servers = pool.connected_servers().await;
        assert!(servers.is_empty());
    }
}
