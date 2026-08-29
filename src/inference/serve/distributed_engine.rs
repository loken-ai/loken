//! Distributed inference engine
//!
//! Combines multiple devices and servers for distributed LLM inference

use crate::distributed::layer_scheduler::ModelInfo;
use crate::distributed::protocol::DistributionPlan;
use crate::distributed::{DeviceManager, LayerScheduler, NetworkClient, NetworkServer};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Execution context for a distributed inference session
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Session ID
    pub session_id: String,
    /// Model being used
    pub model_id: String,
    /// Distribution plan
    pub plan: DistributionPlan,
    /// Current position in sequence
    pub position: usize,
    /// KV cache handles per device
    pub cache_handles: HashMap<String, usize>,
}

/// Distributed inference engine
pub struct DistributedEngine {
    /// Device manager for local devices
    device_manager: Arc<RwLock<DeviceManager>>,
    /// Layer scheduler for distribution planning
    scheduler: Arc<RwLock<Option<LayerScheduler>>>,
    /// Current distribution plan
    plan: Arc<RwLock<Option<DistributionPlan>>>,
    /// Network server (if acting as coordinator)
    // distributed coordinator scaffold; not in production path
    server: Option<NetworkServer>,
    /// Network clients for remote servers
    clients: Vec<NetworkClient>,
    /// Active execution contexts
    contexts: Arc<RwLock<HashMap<String, ExecutionContext>>>,
}

impl DistributedEngine {
    /// Create a new distributed engine
    pub fn new() -> Self {
        Self {
            device_manager: Arc::new(RwLock::new(DeviceManager::new())),
            scheduler: Arc::new(RwLock::new(None)),
            plan: Arc::new(RwLock::new(None)),
            server: None,
            clients: Vec::new(),
            contexts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the engine with device detection
    pub async fn initialize(&mut self) -> Result<()> {
        // Detect local devices
        {
            let mut dm = self.device_manager.write().await;
            dm.detect_devices()?;
        }

        // Create scheduler with device manager
        {
            let _dm = self.device_manager.read().await;
            // We need to clone the device manager for the scheduler
            // For now, create a new one - this should be refactored
        }

        info!("Distributed engine initialized");
        Ok(())
    }

    /// Plan distribution for a model
    pub async fn plan_model(&mut self, model_id: &str) -> Result<DistributionPlan> {
        let model = ModelInfo::from_model_id(model_id);

        let scheduler = self.scheduler.write().await;
        if scheduler.is_none() {
            let _dm = self.device_manager.read().await;
            // Create a new scheduler (temporary - needs proper device manager sharing)
            // For now, return a basic plan
        }

        // Create a basic plan for now
        let plan = DistributionPlan::new(model.model_id.clone(), model.num_layers);

        *self.plan.write().await = Some(plan.clone());
        Ok(plan)
    }

    /// Get available devices
    pub async fn get_devices(&self) -> Vec<String> {
        let dm = self.device_manager.read().await;
        dm.devices()
            .iter()
            .map(|d| format!("{}: {} ({:.1} GB)", d.device_type, d.name, d.memory_gb()))
            .collect()
    }

    /// Get recommended model size for current hardware
    pub async fn recommended_model_size(&self) -> &'static str {
        let dm = self.device_manager.read().await;
        let total_memory = dm.total_compute_memory();

        match total_memory {
            m if m >= 80 * 1024 * 1024 * 1024 => "70B (requires 80GB+)",
            m if m >= 24 * 1024 * 1024 * 1024 => "13B-14B (requires 24GB+)",
            m if m >= 12 * 1024 * 1024 * 1024 => "7B-8B (requires 12GB+)",
            m if m >= 6 * 1024 * 1024 * 1024 => "3B (requires 6GB+)",
            _ => "Use 4-bit quantized models or CPU offloading",
        }
    }

    /// Check if a model can fit in available memory
    pub async fn can_fit_model(&self, model_id: &str) -> bool {
        let model = ModelInfo::from_model_id(model_id);
        let dm = self.device_manager.read().await;

        let total_available: u64 = dm
            .devices()
            .iter()
            .map(crate::distributed::device_manager::ComputeDevice::available_memory_for_model)
            .sum();

        total_available >= model.total_memory()
    }

    /// Get memory summary
    pub async fn memory_summary(&self) -> String {
        let dm = self.device_manager.read().await;
        let total = dm.total_memory();
        let compute = dm.total_compute_memory();

        format!(
            "Total memory: {:.1} GB | GPU memory: {:.1} GB | Devices: {}",
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            compute as f64 / (1024.0 * 1024.0 * 1024.0),
            dm.device_count()
        )
    }

    /// Create a new execution context for a session
    pub async fn create_context(&self, model_id: &str, plan: DistributionPlan) -> Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();

        let context = ExecutionContext {
            session_id: session_id.clone(),
            model_id: model_id.to_string(),
            plan,
            position: 0,
            cache_handles: HashMap::new(),
        };

        let mut contexts = self.contexts.write().await;
        contexts.insert(session_id.clone(), context);

        info!("Created execution context: {}", session_id);
        Ok(session_id)
    }

    /// Get execution context by ID
    pub async fn get_context(&self, session_id: &str) -> Option<ExecutionContext> {
        let contexts = self.contexts.read().await;
        contexts.get(session_id).cloned()
    }

    /// Remove an execution context
    pub async fn remove_context(&self, session_id: &str) -> Option<ExecutionContext> {
        let mut contexts = self.contexts.write().await;
        contexts.remove(session_id)
    }

    /// Execute distributed forward pass for a single token
    ///
    /// This is the core method that:
    /// 1. Routes tokens to the correct devices based on the distribution plan
    /// 2. Executes layers in sequence across devices
    /// 3. Handles tensor transfers between devices/servers
    pub async fn forward_distributed(
        &self,
        session_id: &str,
        input_tokens: &[u32],
    ) -> Result<Vec<f32>> {
        // Get the execution context
        let context = self
            .get_context(session_id)
            .await
            .ok_or_else(|| anyhow!("Session {} not found", session_id))?;

        let plan = &context.plan;

        // Process through each shard in order
        let mut current_activation: Option<Vec<f32>> = None;

        for shard in &plan.shards {
            let server_id = &shard.server_id;
            let layers = &shard.layers;

            if server_id == "local" {
                // Execute locally
                current_activation = Some(
                    self.execute_local_layers(&current_activation, layers, input_tokens)
                        .await?,
                );
            } else {
                // Execute remotely
                current_activation = Some(
                    self.execute_remote_layers(&current_activation, layers, server_id)
                        .await?,
                );
            }
        }

        // Return the final logits
        current_activation.ok_or_else(|| anyhow!("No output from distributed forward pass"))
    }

    /// Execute layers on local device.
    ///
    /// Stub - would call into the inference engine for the assigned layer
    /// range. Returning all-zero logits silently let callers ship code
    /// that "worked" but generated noise; bail loudly until the
    /// real local-execution path is wired so misuse fails at integration
    /// time rather than at deploy time.
    async fn execute_local_layers(
        &self,
        _input: &Option<Vec<f32>>,
        _layers: &[u32],
        _tokens: &[u32],
    ) -> Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "execute_local_layers: not implemented (call LlmEngine::generate / generate_stream directly)"
        ))
    }

    /// Execute layers on remote server.
    ///
    /// Stub - would serialize the tensor, send to remote server,
    /// deserialize response. Same loud-bail rationale as
    /// `execute_local_layers`: returning zero-vector placeholders
    /// would let any distributed-inference caller integrate against
    /// the method and ship code that silently produced noise.
    async fn execute_remote_layers(
        &self,
        _input: &Option<Vec<f32>>,
        _layers: &[u32],
        _server_id: &str,
    ) -> Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "execute_remote_layers: cross-host distributed inference not implemented"
        ))
    }

    /// Generate tokens with distributed inference
    pub async fn generate_distributed(
        &self,
        session_id: &str,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        let mut current_tokens = prompt_tokens.to_vec();

        for _ in 0..max_tokens {
            // Run forward pass
            let logits = self
                .forward_distributed(session_id, &current_tokens)
                .await?;

            // Sample next token
            let next_token = self.sample_token(&logits, temperature)?;

            generated.push(next_token);
            current_tokens.push(next_token);

            // Check for EOS (simplified)
            if next_token == 2 {
                // Common EOS token ID
                break;
            }
        }

        Ok(generated)
    }

    /// Sample next token from logits
    fn sample_token(&self, logits: &[f32], temperature: f32) -> Result<u32> {
        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

        // Simple softmax
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&l| (l - max_logit).exp()).sum();

        let probs: Vec<f32> = scaled
            .iter()
            .map(|&l| ((l - max_logit).exp()) / exp_sum)
            .collect();

        // Sample. rand 0.9 renamed thread_rng() -> rng() and moved
        // gen() onto a new RngExt trait as random().
        use rand::RngExt;
        let mut rng = rand::rng();
        let r: f32 = rng.random();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return Ok(i as u32);
            }
        }

        // Fallback to argmax
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(max_idx as u32)
    }

    /// Get statistics about the distributed system
    pub async fn get_stats(&self) -> DistributedStats {
        let dm = self.device_manager.read().await;
        let contexts = self.contexts.read().await;

        DistributedStats {
            total_devices: dm.device_count(),
            total_memory_gb: dm.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            active_sessions: contexts.len(),
            remote_servers: self.clients.len(),
        }
    }
}

/// Statistics about the distributed inference system
#[derive(Debug, Clone)]
pub struct DistributedStats {
    pub total_devices: usize,
    pub total_memory_gb: f64,
    pub active_sessions: usize,
    pub remote_servers: usize,
}

impl Default for DistributedEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = DistributedEngine::new();
        assert!(engine.server.is_none());
        assert!(engine.clients.is_empty());
    }

    #[tokio::test]
    async fn test_engine_initialize() {
        let mut engine = DistributedEngine::new();
        let result = engine.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_recommended_model_size() {
        let engine = DistributedEngine::new();
        let size = engine.recommended_model_size().await;
        assert!(!size.is_empty());
    }
}
