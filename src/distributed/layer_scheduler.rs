//! Layer scheduler for distributed inference
//!
//! Implements optimal layer assignment across heterogeneous devices and servers

use crate::distributed::device_manager::{ComputeDevice, DeviceManager, DeviceType};
use crate::distributed::protocol::{DeviceInfo, DistributionPlan};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{info, warn};

/// Memory per layer for different model sizes (approximate, FP16)
const MEMORY_PER_LAYER_7B_FP16: u64 = 256 * 1024 * 1024; // ~256 MB per layer
const MEMORY_PER_LAYER_13B_FP16: u64 = 512 * 1024 * 1024; // ~512 MB per layer
const MEMORY_PER_LAYER_70B_FP16: u64 = 2 * 1024 * 1024 * 1024; // ~2 GB per layer

/// Quantization type for memory estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// Full precision (FP16/BF16) - 2 bytes per parameter
    FP16,
    /// 8-bit quantization - 1 byte per parameter
    Q8,
    /// 6-bit quantization - 0.75 bytes per parameter
    Q6,
    /// 5-bit quantization - 0.625 bytes per parameter
    Q5,
    /// 4-bit quantization - 0.5 bytes per parameter
    Q4,
    /// 3-bit quantization - 0.375 bytes per parameter
    Q3,
    /// 2-bit quantization - 0.25 bytes per parameter
    Q2,
}

impl QuantizationType {
    /// Get bytes per parameter for this quantization
    pub fn bytes_per_param(&self) -> f32 {
        match self {
            QuantizationType::FP16 => 2.0,
            QuantizationType::Q8 => 1.0,
            QuantizationType::Q6 => 0.75,
            QuantizationType::Q5 => 0.625,
            QuantizationType::Q4 => 0.5,
            QuantizationType::Q3 => 0.375,
            QuantizationType::Q2 => 0.25,
        }
    }

    /// Parse from GGUF-style quantization name
    pub fn from_gguf_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();
        if name_lower.contains("q2") || name_lower.contains("q2_k") {
            QuantizationType::Q2
        } else if name_lower.contains("q3") || name_lower.contains("q3_k") {
            QuantizationType::Q3
        } else if name_lower.contains("q4") || name_lower.contains("q4_k") {
            QuantizationType::Q4
        } else if name_lower.contains("q5") || name_lower.contains("q5_k") {
            QuantizationType::Q5
        } else if name_lower.contains("q6") || name_lower.contains("q6_k") {
            QuantizationType::Q6
        } else if name_lower.contains("q8") || name_lower.contains("q8_0") {
            QuantizationType::Q8
        } else {
            QuantizationType::FP16
        }
    }
}

/// Model layer information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model ID
    pub model_id: String,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Memory per layer in bytes
    pub memory_per_layer: u64,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
}

/// Everything a [`ModelInfo`] needs beyond the id it was asked about: the geometry a family
/// is published with, and what one of its layers weighs at FP16 before a quantisation is
/// applied to it.
#[derive(Clone, Copy)]
struct Family {
    num_layers: u32,
    memory_per_layer_fp16: u64,
    hidden_size: usize,
    num_heads: usize,
}

/// The 7B/8B shape, and what an id nobody recognises is estimated as.
const SEVEN_B: Family = Family {
    num_layers: 32,
    memory_per_layer_fp16: MEMORY_PER_LAYER_7B_FP16,
    hidden_size: 4096,
    num_heads: 32,
};

/// The families told apart by what their names contain, MOST SPECIFIC FIRST - the order is
/// part of the rule, not a presentation of it. A name carries every substring in it:
/// `devstral-small-24b` carries `24b`, and `llama-13b` carries `3b`, so a table consulted in
/// any other order answers a question the caller did not ask.
///
/// A 24B checkpoint is 40 layers, and a Q4 publication of one is ~15 GB on disk - which is
/// where the ~1.5 GB an FP16 layer of it weighs comes from.
const FAMILIES: &[(&[&str], Family)] = &[
    (
        &["devstral", "24b"],
        Family {
            num_layers: 40,
            memory_per_layer_fp16: 1536 * 1024 * 1024,
            hidden_size: 5120,
            num_heads: 32,
        },
    ),
    (&["7b", "8b"], SEVEN_B),
    (
        &["13b", "14b"],
        Family {
            num_layers: 40,
            memory_per_layer_fp16: MEMORY_PER_LAYER_13B_FP16,
            hidden_size: 5120,
            num_heads: 40,
        },
    ),
    (
        &["70b"],
        Family {
            num_layers: 80,
            memory_per_layer_fp16: MEMORY_PER_LAYER_70B_FP16,
            hidden_size: 8192,
            num_heads: 64,
        },
    ),
    (
        &["3b"],
        Family {
            num_layers: 28,
            memory_per_layer_fp16: MEMORY_PER_LAYER_7B_FP16 / 2,
            hidden_size: 3072,
            num_heads: 24,
        },
    ),
];

impl ModelInfo {
    /// Create model info for known architectures (FP16 by default)
    pub fn from_model_id(model_id: &str) -> Self {
        Self::from_model_id_with_quant(model_id, QuantizationType::FP16)
    }

    /// Create model info for known architectures with specific quantization
    pub fn from_model_id_with_quant(model_id: &str, quant: QuantizationType) -> Self {
        let model_lower = model_id.to_lowercase();
        let quant_factor = quant.bytes_per_param() / 2.0; // Ratio vs FP16
        let family = FAMILIES
            .iter()
            .find(|(names, _)| names.iter().any(|n| model_lower.contains(n)))
            .map_or(SEVEN_B, |(_, f)| *f);

        Self {
            model_id: model_id.to_string(),
            num_layers: family.num_layers,
            memory_per_layer: (family.memory_per_layer_fp16 as f32 * quant_factor) as u64,
            hidden_size: family.hidden_size,
            num_heads: family.num_heads,
        }
    }

    /// Estimate total model memory (weights only, no KV cache)
    pub fn total_memory(&self) -> u64 {
        self.memory_per_layer * self.num_layers as u64
    }

    /// Estimate KV cache memory per token
    pub fn kv_cache_per_token(&self) -> u64 {
        // 2 * num_layers * num_heads * head_dim * 2 bytes (F16)
        let head_dim = self.hidden_size / self.num_heads;
        2 * self.num_layers as u64 * self.num_heads as u64 * head_dim as u64 * 2
    }
}

/// Layer assignment to a device
#[derive(Debug, Clone)]
pub struct LayerAssignment {
    /// Device ID
    pub device_id: usize,
    /// Server ID (for remote devices)
    pub server_id: Option<String>,
    /// Layer indices assigned to this device
    pub layers: Vec<u32>,
    /// Memory required for these layers
    pub memory_required: u64,
    /// Estimated latency per token
    pub estimated_latency_ms: f32,
}

/// Model shard for network distribution
#[derive(Debug, Clone)]
pub struct ModelShard {
    /// Shard ID
    pub shard_id: u32,
    /// Server ID hosting this shard
    pub server_id: String,
    /// Device on the server
    pub device_id: usize,
    /// Layer range (start, end)
    pub layer_range: (u32, u32),
    /// Memory required
    pub memory_required: u64,
}

/// Layer scheduler for optimal distribution
pub struct LayerScheduler {
    /// Device manager
    device_manager: DeviceManager,
    /// Registered remote servers
    remote_servers: HashMap<String, ServerInfo>,
    /// Current distribution plan
    current_plan: Option<DistributionPlan>,
}

/// Remote server information
#[derive(Debug, Clone)]
struct ServerInfo {
    server_id: String,
    endpoint: String,
    devices: Vec<DeviceInfo>,
    latency_ms: f32,
}

impl LayerScheduler {
    /// Create a new layer scheduler
    pub fn new(device_manager: DeviceManager) -> Self {
        Self {
            device_manager,
            remote_servers: HashMap::new(),
            current_plan: None,
        }
    }

    /// Collect all devices (local and remote), filtering unavailable ones
    fn collect_all_devices(&self) -> Vec<(ComputeDevice, Option<String>)> {
        let mut devices = Vec::new();

        // Add local devices that are available
        for device in self.device_manager.devices() {
            // Skip devices that are not available
            if !device.availability.is_available() {
                info!(
                    "Skipping unavailable device: {} #{} ({}) - {:?}",
                    device.device_type,
                    device.id,
                    device.name,
                    device.availability.reason()
                );
                continue;
            }
            devices.push((device.clone(), None));
        }

        // Add remote devices
        for (server_id, server) in &self.remote_servers {
            for device_info in &server.devices {
                let device = ComputeDevice::new(
                    device_info.id,
                    match device_info.device_type.as_str() {
                        "CUDA" => DeviceType::Cuda,
                        "Intel Arc" => DeviceType::IntelArc,
                        "Intel Xe" => DeviceType::IntelXe,
                        "Metal" => DeviceType::Metal,
                        _ => DeviceType::Remote,
                    },
                    device_info.name.clone(),
                    device_info.memory_bytes,
                );
                // Skip remote devices that are unavailable
                if !device.availability.is_available() {
                    continue;
                }
                devices.push((device, Some(server_id.clone())));
            }
        }

        // Sort by priority (highest first)
        devices.sort_by(|a, b| b.0.priority.cmp(&a.0.priority));

        devices
    }

    /// Assign layers to devices based on available memory.
    ///
    /// The layers of a stack all weigh `model.memory_per_layer` and the devices come with a
    /// preference - the accelerators ahead of the hosts, each group by priority - so what is
    /// being asked here is the question the repository answers in one place: given a list of
    /// weights and a list of budgets, which budget takes which element. Consecutive layers
    /// therefore land on one device for as long as it holds them, which is what keeps a
    /// cross-device transfer to one per boundary, and a layer no device holds comes back
    /// unplaced instead of being quietly dropped.
    fn assign_layers_by_memory(
        &self,
        model: &ModelInfo,
        devices: &[(ComputeDevice, Option<String>)],
    ) -> Result<Vec<LayerAssignment>> {
        let total_layers = model.num_layers;

        // The order below IS the preference: accelerators first, then hosts, each group with
        // the highest priority leading. The sort is stable, so devices that rank the same keep
        // the order the caller collected them in.
        let mut offered: Vec<&(ComputeDevice, Option<String>)> = devices
            .iter()
            .filter(|(d, _)| d.availability.is_available())
            .collect();
        offered.sort_by_key(|(d, _)| {
            (
                d.device_type == DeviceType::Cpu,
                std::cmp::Reverse(d.priority),
            )
        });

        let weights = vec![model.memory_per_layer; total_layers as usize];
        let budgets: Vec<u64> = offered
            .iter()
            .map(|(d, _)| d.available_memory_for_model())
            .collect();
        let placed = crate::inference::place::plan::place(&weights, &budgets);

        // Consecutive layers on one device are one assignment. A layer that fits nowhere ends
        // the walk: the layers all weigh the same, so nothing after it could fit either.
        let mut runs: Vec<(usize, u32, u32)> = Vec::new(); // (offered index, first layer, count)
        for (layer, slot) in placed.iter().enumerate() {
            let Some(slot) = *slot else { break };
            match runs.last_mut() {
                Some(run) if run.0 == slot => run.2 += 1,
                _ => runs.push((slot, layer as u32, 1)),
            }
        }

        if let Some(&(_, first, _)) = runs
            .iter()
            .find(|&&(slot, _, _)| offered[slot].0.device_type == DeviceType::Cpu)
        {
            info!(
                "GPU memory insufficient for all layers. Offloading layers {}-{} to CPU",
                first,
                total_layers - 1
            );
        }

        let assignments: Vec<LayerAssignment> = runs
            .iter()
            .map(|&(slot, first, count)| {
                let (device, server_id) = offered[slot];
                let base_latency = self.estimate_layer_latency(&device.device_type);
                let network_latency = server_id
                    .as_ref()
                    .and_then(|s| self.remote_servers.get(s))
                    .map_or(0.0, |s| s.latency_ms);

                info!(
                    "Assigned layers {}-{} to {} ({:.1} GB)",
                    first,
                    first + count - 1,
                    device.name,
                    device.memory_gb()
                );

                LayerAssignment {
                    device_id: device.id,
                    server_id: server_id.clone(),
                    layers: (first..first + count).collect(),
                    memory_required: count as u64 * model.memory_per_layer,
                    estimated_latency_ms: (base_latency * count as f32) + network_latency,
                }
            })
            .collect();

        let assigned: u32 = runs.iter().map(|&(_, _, count)| count).sum();
        if assigned < total_layers {
            warn!(
                "No device holds a layer of {}: {} bytes each",
                model.model_id, model.memory_per_layer
            );
            return Err(anyhow!(
                "Insufficient total memory across all devices. Assigned {}/{} layers",
                assigned,
                total_layers
            ));
        }

        // Log distribution summary
        self.log_layer_distribution(model, &assignments);

        Ok(assignments)
    }

    /// Log a summary of layer distribution across devices
    fn log_layer_distribution(&self, model: &ModelInfo, assignments: &[LayerAssignment]) {
        use crate::distributed::device_manager::{log_distribution_plan, DeviceDistribution};

        let distributions: Vec<DeviceDistribution> = assignments
            .iter()
            .map(|a| {
                let device = self.device_manager.get_device(a.device_id);
                let device_type = device.map(|d| d.device_type).unwrap_or(DeviceType::Cpu);
                let device_name = device
                    .map(|d| format!("{} #{}", d.device_type, d.id))
                    .unwrap_or_else(|| format!("Device {}", a.device_id));

                DeviceDistribution {
                    device_name,
                    server_id: a.server_id.clone(),
                    layer_range: (
                        *a.layers.first().unwrap_or(&0),
                        *a.layers.last().unwrap_or(&0),
                    ),
                    memory_bytes: a.memory_required,
                    device_type,
                }
            })
            .collect();

        log_distribution_plan(&model.model_id, model.num_layers, &distributions);
    }

    /// Estimate latency per layer for a device type
    fn estimate_layer_latency(&self, device_type: &DeviceType) -> f32 {
        // Approximate latency in milliseconds per layer
        match device_type {
            DeviceType::Cuda => 0.5,     // Fast GPU inference
            DeviceType::IntelArc => 1.0, // Good GPU performance
            DeviceType::Metal => 0.8,    // Apple Silicon
            DeviceType::IntelXe => 3.0,  // Integrated GPU
            DeviceType::Remote => 2.0,   // Network overhead
            DeviceType::Cpu => 10.0,     // CPU fallback
        }
    }

    /// Log the distribution plan
    fn log_plan(&self, plan: &DistributionPlan) {
        info!("Distribution plan for {}:", plan.model_id);
        for shard in &plan.shards {
            info!(
                "  Server '{}': layers {:?} ({:.2} GB, {:.1}ms)",
                shard.server_id,
                shard.layers.first().copied()..=shard.layers.last().copied(),
                shard.memory_required as f64 / (1024.0 * 1024.0 * 1024.0),
                shard.estimated_latency_ms
            );
        }
        info!(
            "  Total estimated latency: {:.1}ms/token",
            plan.total_latency_ms
        );
    }

    /// Get the current distribution plan
    pub fn current_plan(&self) -> Option<&DistributionPlan> {
        self.current_plan.as_ref()
    }

    /// Check if a model can fit in available memory
    pub fn can_fit_model(&self, model: &ModelInfo) -> bool {
        let mut total_memory: u64 = self
            .device_manager
            .devices()
            .iter()
            .map(super::device_manager::ComputeDevice::available_memory_for_model)
            .sum();

        // Add remote server memory
        for server in self.remote_servers.values() {
            for device in &server.devices {
                total_memory += device.memory_bytes / 5 * 4; // 80% usable
            }
        }

        total_memory >= model.total_memory()
    }

    /// Get recommended model size for current hardware
    pub fn recommended_model_size(&self) -> &'static str {
        let total_memory: u64 = self.device_manager.total_compute_memory();

        match total_memory {
            m if m >= 80 * 1024 * 1024 * 1024 => "70B",
            m if m >= 24 * 1024 * 1024 * 1024 => "13B-14B",
            m if m >= 12 * 1024 * 1024 * 1024 => "7B-8B",
            m if m >= 6 * 1024 * 1024 * 1024 => "3B",
            _ => "Use quantized models or CPU offloading",
        }
    }

    /// Get device manager reference
    pub fn device_manager(&self) -> &DeviceManager {
        &self.device_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::device_manager::{DeviceAvailability, UnavailableReason};

    #[test]
    fn test_model_info_7b() {
        let model = ModelInfo::from_model_id("llama-7b");
        assert_eq!(model.num_layers, 32);
        assert_eq!(model.hidden_size, 4096);
    }

    #[test]
    fn test_model_info_70b() {
        let model = ModelInfo::from_model_id("llama-70b");
        assert_eq!(model.num_layers, 80);
        assert_eq!(model.hidden_size, 8192);
    }

    #[test]
    fn test_memory_estimation() {
        let model = ModelInfo::from_model_id("llama-7b");
        let total = model.total_memory();
        assert!(total > 0);

        let kv_per_token = model.kv_cache_per_token();
        assert!(kv_per_token > 0);
    }

    #[test]
    fn test_scheduler_creation() {
        let dm = DeviceManager::new();
        let scheduler = LayerScheduler::new(dm);

        assert!(scheduler.current_plan().is_none());
    }

    #[test]
    fn test_devstral_layer_count() {
        // Devstral-small-24b has 40 layers
        let model = ModelInfo::from_model_id("devstral-small-24b");
        assert_eq!(model.num_layers, 40);
    }

    #[test]
    fn quantization_bytes_per_param_matches_ggml_definitions() {
        // GGML's per-quant bytes-per-param ratios - drift here would
        // mis-size every distributed-inference memory budget, leading
        // to false OOM rejections OR overcommit + actual OOM at load.
        assert_eq!(QuantizationType::FP16.bytes_per_param(), 2.0);
        assert_eq!(QuantizationType::Q8.bytes_per_param(), 1.0);
        assert_eq!(QuantizationType::Q6.bytes_per_param(), 0.75);
        assert_eq!(QuantizationType::Q5.bytes_per_param(), 0.625);
        assert_eq!(QuantizationType::Q4.bytes_per_param(), 0.5);
        assert_eq!(QuantizationType::Q3.bytes_per_param(), 0.375);
        assert_eq!(QuantizationType::Q2.bytes_per_param(), 0.25);
    }

    #[test]
    fn quantization_from_gguf_name_recognises_each_tier() {
        // Bare aliases.
        assert_eq!(QuantizationType::from_gguf_name("q2"), QuantizationType::Q2);
        assert_eq!(QuantizationType::from_gguf_name("q3"), QuantizationType::Q3);
        assert_eq!(QuantizationType::from_gguf_name("q4"), QuantizationType::Q4);
        assert_eq!(QuantizationType::from_gguf_name("q5"), QuantizationType::Q5);
        assert_eq!(QuantizationType::from_gguf_name("q6"), QuantizationType::Q6);
        assert_eq!(QuantizationType::from_gguf_name("q8"), QuantizationType::Q8);
        // GGUF-style suffixed variants (Q4_K, Q8_0).
        assert_eq!(
            QuantizationType::from_gguf_name("q4_k_m"),
            QuantizationType::Q4
        );
        assert_eq!(
            QuantizationType::from_gguf_name("q8_0"),
            QuantizationType::Q8
        );
        assert_eq!(
            QuantizationType::from_gguf_name("q6_k"),
            QuantizationType::Q6
        );
        // Case-insensitive.
        assert_eq!(
            QuantizationType::from_gguf_name("Q4_K_M"),
            QuantizationType::Q4
        );
        assert_eq!(
            QuantizationType::from_gguf_name("Q8_0"),
            QuantizationType::Q8
        );
    }

    #[test]
    fn quantization_from_gguf_name_falls_back_to_fp16_on_unknown() {
        // Unknown / empty -> FP16 (safe upper bound, won't undersize the
        // memory estimate). A regression that fell back to Q4 instead
        // would silently undercount memory by 4x and OOM at load.
        assert_eq!(QuantizationType::from_gguf_name(""), QuantizationType::FP16);
        assert_eq!(
            QuantizationType::from_gguf_name("fp16"),
            QuantizationType::FP16
        );
        assert_eq!(
            QuantizationType::from_gguf_name("f32"),
            QuantizationType::FP16
        );
        assert_eq!(
            QuantizationType::from_gguf_name("garbage"),
            QuantizationType::FP16
        );
    }

    #[test]
    fn quantization_from_gguf_name_priority_matches_search_order() {
        // The match is sequential: q2 -> q3 -> q4 -> q5 -> q6 -> q8. So a
        // name like "q2_then_q4" (contrived) hits Q2 first because the
        // q2 check fires before q4. Pin this so a future refactor to
        // a HashMap doesn't silently change priority.
        // (Real-world this matters for the suffixed-name lookup  -
        // "q2_k" must hit Q2, not fall through to anything else.)
        assert_eq!(
            QuantizationType::from_gguf_name("q2_k"),
            QuantizationType::Q2,
            "q2 lookup must not leak through to higher quants"
        );
        assert_eq!(
            QuantizationType::from_gguf_name("q3_k_l"),
            QuantizationType::Q3,
            "q3 lookup must not leak through to higher quants"
        );
    }

    #[test]
    fn model_info_quantization_scales_memory_per_layer() {
        // Same model, different quants: memory_per_layer must scale
        // linearly with bytes_per_param relative to FP16.
        let fp16 = ModelInfo::from_model_id_with_quant("llama-7b", QuantizationType::FP16);
        let q4 = ModelInfo::from_model_id_with_quant("llama-7b", QuantizationType::Q4);
        let q8 = ModelInfo::from_model_id_with_quant("llama-7b", QuantizationType::Q8);
        // Layer counts identical (same architecture).
        assert_eq!(fp16.num_layers, q4.num_layers);
        assert_eq!(fp16.num_layers, q8.num_layers);
        // Q4 is 1/4 the memory of FP16 (0.5 / 2.0), Q8 is 1/2.
        let ratio_q4 = q4.memory_per_layer as f64 / fp16.memory_per_layer as f64;
        let ratio_q8 = q8.memory_per_layer as f64 / fp16.memory_per_layer as f64;
        assert!(
            (ratio_q4 - 0.25).abs() < 0.001,
            "Q4/FP16 ratio = {ratio_q4}, want 0.25"
        );
        assert!(
            (ratio_q8 - 0.5).abs() < 0.001,
            "Q8/FP16 ratio = {ratio_q8}, want 0.50"
        );
    }

    // ---------------------------------------------------------------------
    // The placement judge.
    //
    // `assign_layers_by_memory` decides which device holds which layer. A change
    // there is not observable from a type check: the same signature returns a
    // different plan, and a different plan is either an out-of-memory at load or
    // layers silently moved to the host. The two functions below state the rule
    // twice - once as the two-pass device walk, once as it is implemented now  -
    // and `placements_agree_over_the_case_table` asserts the two answers are the
    // same object over a table of cases chosen so that each degree of freedom of
    // the rule (device count, budget, layer count, ordering) is exercised.
    // ---------------------------------------------------------------------

    /// The device-placement rule expressed as an explicit two-pass walk: the
    /// accelerators in priority order take `available / memory_per_layer`
    /// consecutive layers each, then the hosts take whatever is left, and a stack
    /// that does not finish is an error naming how far it got.
    ///
    /// This is the judge, not production code. It is compared against
    /// [`LayerScheduler::assign_layers_by_memory`]; nothing else calls it.
    /// Logging is omitted - it does not enter the returned plan.
    fn reference_assign_layers_by_memory(
        scheduler: &LayerScheduler,
        model: &ModelInfo,
        devices: &[(ComputeDevice, Option<String>)],
    ) -> Result<Vec<LayerAssignment>> {
        let total_layers = model.num_layers;

        let mut gpu_devices: Vec<_> = devices
            .iter()
            .filter(|(d, _)| d.device_type != DeviceType::Cpu && d.availability.is_available())
            .collect();
        let mut cpu_devices: Vec<_> = devices
            .iter()
            .filter(|(d, _)| d.device_type == DeviceType::Cpu && d.availability.is_available())
            .collect();

        gpu_devices.sort_by(|a, b| b.0.priority.cmp(&a.0.priority));
        cpu_devices.sort_by(|a, b| b.0.priority.cmp(&a.0.priority));

        /// One pass over a pool of devices, each taking as many consecutive
        /// layers as its budget divided by the per-layer weight allows.
        fn take(
            scheduler: &LayerScheduler,
            model: &ModelInfo,
            pool: &[&(ComputeDevice, Option<String>)],
            current_layer: &mut u32,
            assignments: &mut Vec<LayerAssignment>,
        ) {
            let total_layers = model.num_layers;
            for (device, server_id) in pool {
                if *current_layer >= total_layers {
                    break;
                }
                let available_memory = device.available_memory_for_model();
                let layers_can_fit = (available_memory / model.memory_per_layer) as usize;
                if layers_can_fit == 0 {
                    continue;
                }
                let layers_remaining = total_layers - *current_layer;
                let layers_to_assign = std::cmp::min(layers_can_fit, layers_remaining as usize);
                let layer_range: Vec<u32> =
                    (*current_layer..*current_layer + layers_to_assign as u32).collect();

                let base_latency = scheduler.estimate_layer_latency(&device.device_type);
                let network_latency = server_id
                    .as_ref()
                    .and_then(|s| scheduler.remote_servers.get(s))
                    .map_or(0.0, |s| s.latency_ms);

                assignments.push(LayerAssignment {
                    device_id: device.id,
                    server_id: server_id.clone(),
                    layers: layer_range,
                    memory_required: layers_to_assign as u64 * model.memory_per_layer,
                    estimated_latency_ms: (base_latency * layers_to_assign as f32)
                        + network_latency,
                });
                *current_layer += layers_to_assign as u32;
            }
        }

        let mut assignments = Vec::new();
        let mut current_layer: u32 = 0;

        take(
            scheduler,
            model,
            &gpu_devices,
            &mut current_layer,
            &mut assignments,
        );
        if current_layer < total_layers {
            take(
                scheduler,
                model,
                &cpu_devices,
                &mut current_layer,
                &mut assignments,
            );
        }

        if current_layer < total_layers {
            return Err(anyhow!(
                "Insufficient total memory across all devices. Assigned {}/{} layers",
                current_layer,
                total_layers
            ));
        }

        Ok(assignments)
    }

    /// A plan reduced to what a caller can observe, with the latency compared by
    /// its bits so two f32 that differ in the last place are not read as equal.
    type Observed = Vec<(usize, Option<String>, Vec<u32>, u64, u32)>;

    fn observe(plan: &Result<Vec<LayerAssignment>>) -> std::result::Result<Observed, String> {
        match plan {
            Ok(assignments) => Ok(assignments
                .iter()
                .map(|a| {
                    (
                        a.device_id,
                        a.server_id.clone(),
                        a.layers.clone(),
                        a.memory_required,
                        a.estimated_latency_ms.to_bits(),
                    )
                })
                .collect()),
            Err(e) => Err(e.to_string()),
        }
    }

    /// One 256 MiB layer - the FP16 weight of a 7B block, and the unit every
    /// budget in the case table below is expressed in.
    const L: u64 = MEMORY_PER_LAYER_7B_FP16;

    /// A model of `num_layers` blocks weighing `memory_per_layer` each. The
    /// geometry fields play no part in placement.
    fn model_of(num_layers: u32, memory_per_layer: u64) -> ModelInfo {
        ModelInfo {
            model_id: "judge".to_string(),
            num_layers,
            memory_per_layer,
            hidden_size: 4096,
            num_heads: 32,
        }
    }

    /// A device holding `layers` of `per_layer` bytes and not one more.
    /// `available_memory_for_model` keeps 20% back, so the card is sized 5/4 of
    /// the budget wanted; the assertion is what makes a "fits exactly" row of the
    /// table an actual boundary rather than an approximate one.
    fn card(
        id: usize,
        device_type: DeviceType,
        layers: u64,
        per_layer: u64,
    ) -> (ComputeDevice, Option<String>) {
        let d = raw_card(id, device_type, layers * per_layer * 5 / 4);
        assert_eq!(
            d.0.available_memory_for_model() / per_layer,
            layers,
            "budget for device {id} is not the intended {layers}-layer boundary"
        );
        d
    }

    /// A device sized in bytes directly, for budgets that are not a whole number
    /// of layers. Availability is pinned Available: which backends this build
    /// compiled is not the rule under test.
    fn raw_card(
        id: usize,
        device_type: DeviceType,
        memory_bytes: u64,
    ) -> (ComputeDevice, Option<String>) {
        let mut d = ComputeDevice::new(
            id,
            device_type,
            format!("{device_type} #{id}"),
            memory_bytes,
        );
        d.availability = DeviceAvailability::Available;
        d.available = true;
        (d, None)
    }

    #[test]
    fn placements_agree_over_the_case_table() {
        // A scheduler with one registered remote server, so the row that places
        // layers behind the network gets a non-zero per-shard latency term.
        let mut scheduler = LayerScheduler::new(DeviceManager::new());
        scheduler.remote_servers.insert(
            "srv-1".to_string(),
            ServerInfo {
                server_id: "srv-1".to_string(),
                endpoint: "http://127.0.0.1:9000".to_string(),
                devices: Vec::new(),
                latency_ms: 12.5,
            },
        );

        // A card that the rule must not offer at all.
        let mut absent = card(9, DeviceType::Cuda, 8, L);
        absent.0.availability = DeviceAvailability::Unavailable {
            reason: UnavailableReason::DeviceBusy,
            suggestion: "held by another process".to_string(),
        };
        absent.0.available = false;

        // A host that outranks every accelerator on priority alone: the rule
        // orders by kind first, so it must still be served last.
        let mut loud_host = card(7, DeviceType::Cpu, 8, L);
        loud_host.0.priority = 250;

        // A card reached over the network, carrying its server id.
        let mut remote = card(5, DeviceType::Remote, 8, L);
        remote.1 = Some("srv-1".to_string());

        let cases: Vec<(&str, ModelInfo, Vec<(ComputeDevice, Option<String>)>)> = vec![
            (
                "one card, room to spare",
                model_of(32, L),
                vec![card(0, DeviceType::Cuda, 64, L)],
            ),
            (
                "one card holding exactly the stack",
                model_of(32, L),
                vec![card(0, DeviceType::Cuda, 32, L)],
            ),
            (
                "one card one layer short, host takes the tail",
                model_of(32, L),
                vec![
                    card(0, DeviceType::Cuda, 31, L),
                    card(1, DeviceType::Cpu, 8, L),
                ],
            ),
            (
                "one card holding no layer at all",
                model_of(32, L),
                vec![raw_card(0, DeviceType::Cuda, L / 2)],
            ),
            (
                "card holds none, host holds all",
                model_of(32, L),
                vec![
                    raw_card(0, DeviceType::Cuda, L / 2),
                    card(1, DeviceType::Cpu, 32, L),
                ],
            ),
            (
                "nothing holds anything",
                model_of(4, L),
                vec![
                    raw_card(0, DeviceType::Cuda, 0),
                    raw_card(1, DeviceType::Cpu, 0),
                ],
            ),
            ("no device offered", model_of(4, L), vec![]),
            (
                "two cards, layer count divides the count",
                model_of(32, L),
                vec![
                    card(0, DeviceType::Cuda, 16, L),
                    card(1, DeviceType::Cuda, 16, L),
                ],
            ),
            (
                "two unequal cards, boundary off the halfway mark",
                model_of(32, L),
                vec![
                    card(0, DeviceType::Cuda, 5, L),
                    card(1, DeviceType::Cuda, 27, L),
                ],
            ),
            (
                "three equal cards, 30 layers divides by three",
                model_of(30, L),
                vec![
                    card(0, DeviceType::Cuda, 10, L),
                    card(1, DeviceType::Cuda, 10, L),
                    card(2, DeviceType::Cuda, 10, L),
                ],
            ),
            (
                "three equal cards, 32 layers does not divide by three",
                model_of(32, L),
                vec![
                    card(0, DeviceType::Cuda, 10, L),
                    card(1, DeviceType::Cuda, 10, L),
                    card(2, DeviceType::Cuda, 10, L),
                ],
            ),
            (
                "three cards, the middle one holds nothing",
                model_of(12, L),
                vec![
                    card(0, DeviceType::Cuda, 5, L),
                    raw_card(1, DeviceType::Cuda, L / 2),
                    card(2, DeviceType::Cuda, 20, L),
                ],
            ),
            (
                "offered in reverse order of preference",
                model_of(20, L),
                vec![
                    card(0, DeviceType::Cpu, 20, L),
                    card(1, DeviceType::Remote, 20, L),
                    card(2, DeviceType::IntelXe, 20, L),
                    card(3, DeviceType::Metal, 20, L),
                    card(4, DeviceType::Cuda, 20, L),
                ],
            ),
            (
                "an unavailable card is not offered",
                model_of(8, L),
                vec![absent, card(1, DeviceType::Cuda, 8, L)],
            ),
            (
                "a high-priority host still comes after the accelerators",
                model_of(8, L),
                vec![loud_host, card(1, DeviceType::Cuda, 8, L)],
            ),
            (
                "a remote shard carries its server latency",
                model_of(8, L),
                vec![remote],
            ),
            (
                "spill across two cards then the host, exact at each split",
                model_of(10, L),
                vec![
                    card(0, DeviceType::Cuda, 4, L),
                    card(1, DeviceType::Cuda, 3, L),
                    card(2, DeviceType::Cpu, 3, L),
                ],
            ),
            (
                "equal priority keeps the order the caller collected",
                model_of(20, L),
                vec![
                    card(1, DeviceType::Cuda, 3, L),
                    card(0, DeviceType::Cuda, 30, L),
                ],
            ),
            (
                "a stack of no layers",
                model_of(0, L),
                vec![card(0, DeviceType::Cuda, 8, L)],
            ),
            (
                "the whole model fits on the first of two cards",
                model_of(3, L),
                vec![
                    card(0, DeviceType::Cuda, 100, L),
                    card(1, DeviceType::Cuda, 100, L),
                ],
            ),
            (
                "a real checkpoint over two unequal cards",
                ModelInfo::from_model_id_with_quant("devstral-small-24b", QuantizationType::Q4),
                vec![
                    card(
                        0,
                        DeviceType::Cuda,
                        13,
                        ModelInfo::from_model_id_with_quant(
                            "devstral-small-24b",
                            QuantizationType::Q4,
                        )
                        .memory_per_layer,
                    ),
                    card(
                        1,
                        DeviceType::Cuda,
                        40,
                        ModelInfo::from_model_id_with_quant(
                            "devstral-small-24b",
                            QuantizationType::Q4,
                        )
                        .memory_per_layer,
                    ),
                ],
            ),
        ];

        for (name, model, devices) in &cases {
            let want = observe(&reference_assign_layers_by_memory(
                &scheduler, model, devices,
            ));
            let got = observe(&scheduler.assign_layers_by_memory(model, devices));
            assert_eq!(want, got, "placement differs on case: {name}");
        }
    }

    /// The family table is consulted in order and matches on substrings, so a name
    /// carries every substring in it: `llama-13b` carries `3b`, `devstral-small-24b`
    /// carries `24b`. This pins which row each name reaches.
    fn reference_model_info(model_id: &str, quant: QuantizationType) -> (u32, u64, usize, usize) {
        let model_lower = model_id.to_lowercase();
        let quant_factor = quant.bytes_per_param() / 2.0;

        if model_lower.contains("devstral") {
            let memory_per_layer_fp16 = 1536 * 1024 * 1024;
            (
                40,
                (memory_per_layer_fp16 as f32 * quant_factor) as u64,
                5120,
                32,
            )
        } else if model_lower.contains("24b") {
            let memory_per_layer_fp16 = 1536 * 1024 * 1024;
            (
                40,
                (memory_per_layer_fp16 as f32 * quant_factor) as u64,
                5120,
                32,
            )
        } else if model_lower.contains("7b") || model_lower.contains("8b") {
            (
                32,
                (MEMORY_PER_LAYER_7B_FP16 as f32 * quant_factor) as u64,
                4096,
                32,
            )
        } else if model_lower.contains("13b") || model_lower.contains("14b") {
            (
                40,
                (MEMORY_PER_LAYER_13B_FP16 as f32 * quant_factor) as u64,
                5120,
                40,
            )
        } else if model_lower.contains("70b") {
            (
                80,
                (MEMORY_PER_LAYER_70B_FP16 as f32 * quant_factor) as u64,
                8192,
                64,
            )
        } else if model_lower.contains("3b") {
            (
                28,
                (MEMORY_PER_LAYER_7B_FP16 as f32 * quant_factor / 2.0) as u64,
                3072,
                24,
            )
        } else {
            (
                32,
                (MEMORY_PER_LAYER_7B_FP16 as f32 * quant_factor) as u64,
                4096,
                32,
            )
        }
    }

    #[test]
    fn family_lookup_agrees_on_the_names_that_share_a_prefix() {
        let names = [
            // The pair the ordering exists for: `13b` must win over the `3b` it
            // contains, and `14b` likewise.
            "llama-3b",
            "llama-13b",
            "llama-14b",
            "Llama-13B",
            // `8b` is tested before `13b`, so a 13B published in 8-bit reads as
            // the 7B/8B row. Pinned because it is the surprising one.
            "llama-13b-8bit",
            // `70b` does not contain `7b` - the substring is `70b`.
            "llama-70b",
            "llama-3-70b",
            "llama-7b",
            "llama-3.1-8b",
            // `devstral` and a bare `24b` reach the same geometry.
            "devstral-small-24b",
            "mistral-24b",
            "Devstral-Small-2507",
            // An MoE tag whose expert size ends in `3b`.
            "qwen3-30b-a3b",
            // Names carrying no family token at all.
            "yi-34b",
            "phi-3-mini-4k",
            "",
            "gpt-oss-20b",
        ];
        let quants = [
            QuantizationType::FP16,
            QuantizationType::Q8,
            QuantizationType::Q6,
            QuantizationType::Q5,
            QuantizationType::Q4,
            QuantizationType::Q3,
            QuantizationType::Q2,
        ];

        for name in names {
            for quant in quants {
                let want = reference_model_info(name, quant);
                let info = ModelInfo::from_model_id_with_quant(name, quant);
                let got = (
                    info.num_layers,
                    info.memory_per_layer,
                    info.hidden_size,
                    info.num_heads,
                );
                assert_eq!(want, got, "family lookup differs for {name:?} at {quant:?}");
                assert_eq!(info.model_id, name);
            }
        }
    }
}
