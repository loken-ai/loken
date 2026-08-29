//! Network protocol for distributed inference
//!
//! Defines the message format for layer-to-layer communication across servers

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for an inference request
pub type RequestId = uuid::Uuid;

/// Tensor serialization format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMessage {
    /// Tensor shape (dimensions)
    pub shape: Vec<usize>,
    /// Data type (F32, F16, BF16, etc.)
    pub dtype: String,
    /// Raw tensor data as bytes
    pub data: Vec<u8>,
}

impl TensorMessage {
    /// Create a new tensor message from raw data
    pub fn new(shape: Vec<usize>, dtype: String, data: Vec<u8>) -> Self {
        Self { shape, dtype, data }
    }

    /// Create from a facade Tensor
    #[cfg(feature = "cuda")]
    pub fn from_tensor(tensor: &crate::tensor::Tensor) -> Result<Self, crate::tensor::Error> {
        let shape = tensor.dims().to_vec();
        let dtype = format!("{:?}", tensor.dtype());

        // Flatten and convert to bytes
        let data = match tensor.dtype() {
            crate::tensor::DType::F32 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<f32> = flat.to_vec1()?;
                data.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            crate::tensor::DType::F16 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<half::f16> = flat.to_vec1()?;
                data.iter()
                    .flat_map(|f: &half::f16| f.to_le_bytes())
                    .collect()
            }
            crate::tensor::DType::BF16 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<half::bf16> = flat.to_vec1()?;
                data.iter()
                    .flat_map(|f: &half::bf16| f.to_le_bytes())
                    .collect()
            }
            crate::tensor::DType::U8 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<u8> = flat.to_vec1()?;
                data
            }
            crate::tensor::DType::U32 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<u32> = flat.to_vec1()?;
                data.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            crate::tensor::DType::I32 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<i32> = flat.to_vec1()?;
                data.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            crate::tensor::DType::I64 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<i64> = flat.to_vec1()?;
                data.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            crate::tensor::DType::I16 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<i16> = flat.to_vec1()?;
                data.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            crate::tensor::DType::F64 => {
                let flat = tensor.flatten_all()?;
                let data: Vec<f64> = flat.to_vec1()?;
                data.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
        };

        Ok(Self { shape, dtype, data })
    }

    /// Get the number of elements in this tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get the size in MB
    pub fn size_mb(&self) -> f32 {
        self.data.len() as f32 / (1024.0 * 1024.0)
    }
}

/// Request to compute a layer forward pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRequest {
    /// Unique request ID
    pub request_id: RequestId,
    /// Layer ID to compute
    pub layer_id: u32,
    /// Input activation tensor
    pub activation: TensorMessage,
    /// Current position in sequence (for KV cache)
    pub position: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl LayerRequest {
    /// Create a new layer request
    pub fn new(layer_id: u32, activation: TensorMessage, position: usize) -> Self {
        Self {
            request_id: RequestId::new_v4(),
            layer_id,
            activation,
            position,
            metadata: HashMap::new(),
        }
    }
}

/// Response from a layer forward pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResponse {
    /// Request ID this response corresponds to
    pub request_id: RequestId,
    /// Layer ID that was computed
    pub layer_id: u32,
    /// Output activation tensor
    pub activation: TensorMessage,
    /// Device metrics for this computation
    pub metrics: DeviceMetrics,
    /// Whether computation was successful
    pub success: bool,
    /// Error message if not successful
    pub error: Option<String>,
}

impl LayerResponse {
    /// Create a successful response
    pub fn success(
        request_id: RequestId,
        layer_id: u32,
        activation: TensorMessage,
        metrics: DeviceMetrics,
    ) -> Self {
        Self {
            request_id,
            layer_id,
            activation,
            metrics,
            success: true,
            error: None,
        }
    }

    /// Create an error response
    pub fn error(request_id: RequestId, layer_id: u32, error: String) -> Self {
        Self {
            request_id,
            layer_id,
            activation: TensorMessage::new(vec![], "F32".to_string(), vec![]),
            metrics: DeviceMetrics::default(),
            success: false,
            error: Some(error),
        }
    }
}

/// Device metrics for a computation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceMetrics {
    /// Time to compute in milliseconds
    pub compute_time_ms: f32,
    /// Time to transfer input data in milliseconds
    pub input_transfer_ms: f32,
    /// Time to transfer output data in milliseconds
    pub output_transfer_ms: f32,
    /// Memory used in bytes
    pub memory_used: u64,
    /// GPU utilization during computation (0.0 to 1.0)
    pub gpu_utilization: f32,
    /// Device type that computed this
    pub device_type: String,
    /// Device name
    pub device_name: String,
}

/// Model shard assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAssignment {
    /// Server ID
    pub server_id: String,
    /// Layers assigned to this server
    pub layers: Vec<u32>,
    /// Total memory required for these layers
    pub memory_required: u64,
    /// Estimated latency per token
    pub estimated_latency_ms: f32,
}

/// Full model distribution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionPlan {
    /// Model ID
    pub model_id: String,
    /// Total number of layers
    pub total_layers: u32,
    /// Shard assignments per server
    pub shards: Vec<ShardAssignment>,
    /// Total estimated latency
    pub total_latency_ms: f32,
    /// Whether this plan is valid
    pub valid: bool,
}

impl DistributionPlan {
    /// Create a new distribution plan
    pub fn new(model_id: String, total_layers: u32) -> Self {
        Self {
            model_id,
            total_layers,
            shards: Vec::new(),
            total_latency_ms: 0.0,
            valid: true,
        }
    }

    /// Add a shard to the plan
    pub fn add_shard(&mut self, shard: ShardAssignment) {
        self.total_latency_ms += shard.estimated_latency_ms;
        self.shards.push(shard);
    }

    /// Get the server responsible for a layer
    pub fn server_for_layer(&self, layer_id: u32) -> Option<&str> {
        self.shards
            .iter()
            .find(|s| s.layers.contains(&layer_id))
            .map(|s| s.server_id.as_str())
    }

    /// Validate the plan
    pub fn validate(&mut self) -> bool {
        // Check that all layers are assigned
        let mut assigned_layers: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for shard in &self.shards {
            for layer in &shard.layers {
                if !assigned_layers.insert(*layer) {
                    // Layer was already assigned
                    self.valid = false;
                    return false;
                }
            }
        }

        // Check that all layers from 0 to total_layers-1 are assigned
        for layer in 0..self.total_layers {
            if !assigned_layers.contains(&layer) {
                self.valid = false;
                return false;
            }
        }

        self.valid = true;
        true
    }
}

/// Server registration message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerRegistration {
    /// Unique server ID
    pub server_id: String,
    /// Server endpoint (host:port)
    pub endpoint: String,
    /// Available devices on this server
    pub devices: Vec<DeviceInfo>,
    /// Maximum memory available
    pub max_memory: u64,
    /// Server priority (higher = preferred)
    pub priority: u8,
}

impl ServerRegistration {
    /// Create a new server registration
    pub fn new(server_id: String, endpoint: String) -> Self {
        Self {
            server_id,
            endpoint,
            devices: Vec::new(),
            max_memory: 0,
            priority: 50,
        }
    }
}

/// Device information for registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device ID (local to server)
    pub id: usize,
    /// Device type
    pub device_type: String,
    /// Device name
    pub name: String,
    /// Available memory in bytes
    pub memory_bytes: u64,
    /// Priority for layer assignment
    pub priority: u8,
}

impl DeviceInfo {}

/// Heartbeat message for server health monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    /// Server ID
    pub server_id: String,
    /// Timestamp of heartbeat
    pub timestamp: u64,
    /// Current load (0.0 to 1.0)
    pub load: f32,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f32,
    /// Number of active requests
    pub active_requests: usize,
}

/// Error response for network operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkError {
    /// Error code
    pub code: u32,
    /// Error message
    pub message: String,
    /// Whether this error is recoverable
    pub recoverable: bool,
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for NetworkError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_message_creation() {
        let tensor = TensorMessage::new(vec![2, 3], "F32".to_string(), vec![0u8; 24]);
        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(tensor.size_bytes(), 24);
    }

    #[test]
    fn test_layer_request_creation() {
        let tensor = TensorMessage::new(vec![1, 512], "F32".to_string(), vec![0u8; 2048]);
        let request = LayerRequest::new(5, tensor, 10);

        assert_eq!(request.layer_id, 5);
        assert_eq!(request.position, 10);
        assert!(!request.request_id.is_nil());
    }

    #[test]
    fn test_distribution_plan() {
        let mut plan = DistributionPlan::new("test-model".to_string(), 32);

        let shard1 = ShardAssignment {
            server_id: "server1".to_string(),
            layers: (0..16).collect(),
            memory_required: 4_000_000_000,
            estimated_latency_ms: 10.0,
        };

        let shard2 = ShardAssignment {
            server_id: "server2".to_string(),
            layers: (16..32).collect(),
            memory_required: 4_000_000_000,
            estimated_latency_ms: 12.0,
        };

        plan.add_shard(shard1);
        plan.add_shard(shard2);

        assert!(plan.validate());
        assert_eq!(plan.total_latency_ms, 22.0);
        assert_eq!(plan.server_for_layer(5), Some("server1"));
        assert_eq!(plan.server_for_layer(20), Some("server2"));
    }

    #[test]
    fn distribution_plan_validate_catches_duplicate_layer_assignment() {
        // If two shards both claim the same layer, the plan is
        // ambiguous - calling forward on that layer would race
        // between the two servers. Must mark invalid.
        let mut plan = DistributionPlan::new("dup-model".to_string(), 4);
        plan.add_shard(ShardAssignment {
            server_id: "a".into(),
            layers: vec![0, 1, 2],
            memory_required: 0,
            estimated_latency_ms: 0.0,
        });
        plan.add_shard(ShardAssignment {
            server_id: "b".into(),
            layers: vec![2, 3], // layer 2 conflicts with shard a
            memory_required: 0,
            estimated_latency_ms: 0.0,
        });
        assert!(!plan.validate(), "duplicate layer 2 must invalidate");
        assert!(!plan.valid);
    }

    #[test]
    fn distribution_plan_validate_catches_gap_in_coverage() {
        // Layers 0..total_layers must all be covered; a gap means
        // the inference loop can't route through that layer. Must
        // mark invalid.
        let mut plan = DistributionPlan::new("gap-model".to_string(), 5);
        plan.add_shard(ShardAssignment {
            server_id: "a".into(),
            layers: vec![0, 1],
            memory_required: 0,
            estimated_latency_ms: 0.0,
        });
        plan.add_shard(ShardAssignment {
            server_id: "b".into(),
            layers: vec![3, 4], // layer 2 missing
            memory_required: 0,
            estimated_latency_ms: 0.0,
        });
        assert!(!plan.validate(), "missing layer 2 must invalidate");
    }

    #[test]
    fn distribution_plan_server_for_layer_returns_none_for_unassigned() {
        // Out-of-range layer id or simply unassigned -> None (caller
        // surfaces a clear "layer not routable" error rather than a
        // silent default).
        let mut plan = DistributionPlan::new("m".to_string(), 4);
        plan.add_shard(ShardAssignment {
            server_id: "x".into(),
            layers: vec![0, 1],
            memory_required: 0,
            estimated_latency_ms: 0.0,
        });
        assert_eq!(plan.server_for_layer(0), Some("x"));
        assert_eq!(plan.server_for_layer(99), None, "out-of-range layer -> None");
        assert_eq!(plan.server_for_layer(3), None, "unassigned layer -> None");
    }

    #[test]
    fn layer_response_error_marks_failure_with_empty_activation() {
        // Error responses must (a) flag success=false, (b) carry the
        // error text, (c) include a dummy zero-length activation so
        // deserialization doesn't blow up on the receiver. Pin all
        // three - a refactor that "cleaned up" the empty activation
        // (e.g. made it Option) would be a wire-incompatible break.
        let req_id = RequestId::new_v4();
        let resp = LayerResponse::error(req_id, 7, "oom".to_string());
        assert_eq!(resp.request_id, req_id);
        assert_eq!(resp.layer_id, 7);
        assert!(!resp.success);
        assert_eq!(resp.error.as_deref(), Some("oom"));
        // Empty activation - but valid TensorMessage shape.
        assert!(resp.activation.shape.is_empty());
        assert!(resp.activation.data.is_empty());
        assert_eq!(resp.activation.dtype, "F32");
    }

    #[test]
    fn layer_response_success_carries_full_activation_and_metrics() {
        // Symmetry with the error case: success path preserves the
        // activation + metrics verbatim.
        let req_id = RequestId::new_v4();
        let act = TensorMessage::new(vec![1, 4], "F32".to_string(), vec![0u8; 16]);
        let m = DeviceMetrics {
            compute_time_ms: 12.5,
            device_type: "CUDA".into(),
            ..Default::default()
        };
        let resp = LayerResponse::success(req_id, 3, act, m);
        assert!(resp.success);
        assert!(resp.error.is_none());
        assert_eq!(resp.layer_id, 3);
        assert_eq!(resp.activation.size_bytes(), 16);
        assert_eq!(resp.metrics.compute_time_ms, 12.5);
        assert_eq!(resp.metrics.device_type, "CUDA");
    }

    #[test]
    fn tensor_message_size_mb_matches_data_bytes_over_1mb() {
        // 1 MiB exactly -> 1.0 MB (the helper uses binary MB =
        // 1024 * 1024, matching nvidia-smi convention).
        let t = TensorMessage::new(vec![1024 * 1024], "U8".into(), vec![0u8; 1024 * 1024]);
        assert!((t.size_mb() - 1.0).abs() < 1e-4);
        // 5 MiB.
        let big = TensorMessage::new(
            vec![5 * 1024 * 1024],
            "U8".into(),
            vec![0u8; 5 * 1024 * 1024],
        );
        assert!((big.size_mb() - 5.0).abs() < 1e-4);
        // Empty -> 0 MB, no NaN/Inf.
        let empty = TensorMessage::new(vec![], "F32".into(), vec![]);
        assert_eq!(empty.size_mb(), 0.0);
        assert!(empty.size_mb().is_finite());
    }
}
