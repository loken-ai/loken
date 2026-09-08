//! Heterogeneous device management for distributed inference
//!
//! Supports NVIDIA GPUs (CUDA), Intel Arc (oneAPI/SYCL), and CPU backends
//! with automatic device detection and memory-aware layer assignment.

use anyhow::Result;

#[cfg(feature = "cuda")]
use anyhow::anyhow;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, warn};

/// Horizontal line for logging
const HLINE: &str = "------------------------------------------------------------";
const SLINE: &str = "------------------------------------------------------------";

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeviceType {
    /// NVIDIA GPU via CUDA
    Cuda,
    /// Intel Arc/iGPU via oneAPI/SYCL
    IntelArc,
    /// Intel integrated graphics (Xe)
    IntelXe,
    /// Apple Silicon via Metal
    Metal,
    /// CPU fallback
    Cpu,
    /// Remote network device
    Remote,
}

/// Reason why a device is unavailable
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnavailableReason {
    /// GPU driver not installed
    DriverNotInstalled,
    /// Backend feature not compiled in (e.g., CUDA not enabled)
    BackendNotCompiled,
    /// Hardware not supported by available backends
    BackendNotSupported,
    /// Not enough memory for any layers
    InsufficientMemory,
    /// Device is being used by another process
    DeviceBusy,
    /// Permission denied to access device
    PermissionDenied,
    /// Failed to initialize device
    InitializationFailed,
    /// Runtime library not found (e.g., oneAPI for Intel Arc)
    RuntimeNotFound,
}

impl std::fmt::Display for UnavailableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnavailableReason::DriverNotInstalled => write!(f, "GPU driver not installed"),
            UnavailableReason::BackendNotCompiled => write!(f, "Backend not compiled in"),
            UnavailableReason::BackendNotSupported => {
                write!(f, "Hardware not supported by available backends")
            }
            UnavailableReason::InsufficientMemory => write!(f, "Insufficient memory"),
            UnavailableReason::DeviceBusy => write!(f, "Device busy"),
            UnavailableReason::PermissionDenied => write!(f, "Permission denied"),
            UnavailableReason::InitializationFailed => write!(f, "Initialization failed"),
            UnavailableReason::RuntimeNotFound => write!(f, "Runtime not found"),
        }
    }
}

/// Device availability status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceAvailability {
    /// Device is available for inference
    Available,
    /// Device is not available with reason and suggestion
    Unavailable {
        reason: UnavailableReason,
        suggestion: String,
    },
}

impl DeviceAvailability {
    /// Check if device is available
    pub fn is_available(&self) -> bool {
        matches!(self, DeviceAvailability::Available)
    }

    /// Get the reason if unavailable
    pub fn reason(&self) -> Option<&UnavailableReason> {
        match self {
            DeviceAvailability::Unavailable { reason, .. } => Some(reason),
            _ => None,
        }
    }

    /// Get the suggestion if unavailable
    pub fn suggestion(&self) -> Option<&str> {
        match self {
            DeviceAvailability::Unavailable { suggestion, .. } => Some(suggestion),
            _ => None,
        }
    }
}

impl std::fmt::Display for DeviceAvailability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceAvailability::Available => write!(f, "✅ AVAILABLE"),
            DeviceAvailability::Unavailable { reason, suggestion } => {
                write!(f, "⚠️ UNAVAILABLE: {} ({})", reason, suggestion)
            }
        }
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cuda => write!(f, "CUDA"),
            DeviceType::IntelArc => write!(f, "Intel Arc"),
            DeviceType::IntelXe => write!(f, "Intel Xe"),
            DeviceType::Metal => write!(f, "Metal"),
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::Remote => write!(f, "Remote"),
        }
    }
}

/// Compute device information
#[derive(Debug, Clone)]
pub struct ComputeDevice {
    /// Unique device identifier
    pub id: usize,
    /// Device type
    pub device_type: DeviceType,
    /// Human-readable device name
    pub name: String,
    /// Available memory in bytes
    pub memory_bytes: u64,
    /// Compute capability or architecture version
    pub compute_capability: Option<String>,
    /// Current memory utilization (0.0 to 1.0)
    pub memory_utilization: f32,
    /// Current compute utilization (0.0 to 1.0)
    pub compute_utilization: f32,
    /// Memory bandwidth in GB/s
    pub bandwidth_gbps: f32,
    /// TFLOPS for FP16
    pub tflops_fp16: f32,
    /// Whether this device is available for compute
    pub available: bool,
    /// Priority for layer assignment (higher = preferred)
    pub priority: u8,
    /// Availability status with reason if unavailable
    pub availability: DeviceAvailability,
}

impl ComputeDevice {
    /// Create a new compute device
    pub fn new(id: usize, device_type: DeviceType, name: String, memory_bytes: u64) -> Self {
        let availability = Self::check_device_availability(&device_type);
        let available = availability.is_available();

        Self {
            id,
            device_type,
            name,
            memory_bytes,
            compute_capability: None,
            memory_utilization: 0.0,
            compute_utilization: 0.0,
            bandwidth_gbps: 0.0,
            tflops_fp16: 0.0,
            available,
            priority: Self::calculate_priority(&device_type),
            availability,
        }
    }

    /// Check device availability based on device type and compiled features
    fn check_device_availability(device_type: &DeviceType) -> DeviceAvailability {
        match device_type {
            DeviceType::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA feature is compiled in - check if runtime is available
                    // This will be verified during actual device creation
                    DeviceAvailability::Available
                }
                #[cfg(not(feature = "cuda"))]
                {
                    DeviceAvailability::Unavailable {
                        reason: UnavailableReason::BackendNotCompiled,
                        suggestion: "Build with --features cuda to enable CUDA support".to_string(),
                    }
                }
            }

            DeviceType::IntelArc | DeviceType::IntelXe => {
                // Found, and nothing here dispatches to it: an Arc card is reachable through
                // the OpenCL backend, not through a SYCL one, because there is no SYCL one.
                DeviceAvailability::Unavailable {
                    reason: UnavailableReason::BackendNotSupported,
                    suggestion:
                        "No SYCL backend in this build; an Arc card is served through OpenCL."
                            .to_string(),
                }
            }

            DeviceType::Metal => {
                #[cfg(all(feature = "cuda", target_os = "macos"))]
                {
                    // Metal, on macOS
                    DeviceAvailability::Available
                }
                #[cfg(not(all(feature = "cuda", target_os = "macos")))]
                {
                    DeviceAvailability::Unavailable {
                        reason: UnavailableReason::BackendNotSupported,
                        suggestion: "Metal requires a macOS build with the metal feature"
                            .to_string(),
                    }
                }
            }

            DeviceType::Cpu => {
                // CPU is always available as fallback
                DeviceAvailability::Available
            }

            DeviceType::Remote => {
                // Remote devices availability is determined by connection status
                DeviceAvailability::Available
            }
        }
    }

    /// Calculate device priority for layer assignment
    fn calculate_priority(device_type: &DeviceType) -> u8 {
        match device_type {
            DeviceType::Cuda => 100,    // NVIDIA GPU - highest priority
            DeviceType::IntelArc => 80, // Intel Arc - good performance
            DeviceType::Metal => 80,    // Apple Silicon - good performance
            DeviceType::IntelXe => 60,  // Intel iGPU - moderate
            DeviceType::Remote => 40,   // Network - network latency
            DeviceType::Cpu => 20,      // CPU - lowest priority
        }
    }

    /// Get available memory in GB
    pub fn memory_gb(&self) -> f32 {
        self.memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get available memory in MB
    pub fn memory_mb(&self) -> f32 {
        self.memory_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Calculate memory available for model layers
    pub fn available_memory_for_model(&self) -> u64 {
        // Reserve 20% for overhead (KV cache, activations, etc.)
        let overhead = 0.2;
        let available =
            self.memory_bytes as f64 * (1.0 - overhead - self.memory_utilization as f64);
        available as u64
    }

    /// Estimate max layers this device can handle
    pub fn max_layers(&self, bytes_per_layer: u64) -> usize {
        let available = self.available_memory_for_model();
        (available / bytes_per_layer) as usize
    }

    /// Get device status summary
    pub fn status_summary(&self) -> String {
        format!(
            "[{}] {} {} - {:.1}GB ({}% utilized) - Priority {}",
            self.device_type,
            self.id,
            self.name,
            self.memory_gb(),
            (self.memory_utilization * 100.0) as u8,
            self.priority
        )
    }

    /// Check if device is suitable for a given model size
    pub fn can_fit_model(&self, model_size_bytes: u64) -> bool {
        self.available_memory_for_model() >= model_size_bytes
    }
}

/// Device manager for heterogeneous compute
pub struct DeviceManager {
    devices: Vec<ComputeDevice>,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
        }
    }

    /// Detect all available devices
    pub fn detect_devices(&mut self) -> Result<()> {
        info!("Detecting compute devices...");

        // Detect CUDA devices
        self.detect_cuda_devices()?;

        // Detect Intel devices (Arc, Xe)
        self.detect_intel_devices()?;

        // Always add CPU as fallback
        self.add_cpu_device();

        info!("Detected {} compute device(s)", self.devices.len());
        for device in &self.devices {
            info!("  {}", device.status_summary());
        }

        Ok(())
    }

    /// Detect NVIDIA CUDA devices
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices(&mut self) -> Result<()> {
        use crate::tensor::Device;

        // `--cpu`: skip CUDA enumeration entirely so only the CPU device is
        // registered and the placement plan keeps every layer on the CPU.
        if crate::gpu::force_cpu() {
            info!("--cpu: skipping CUDA device detection (CPU-only placement)");
            return Ok(());
        }

        // Try to enumerate CUDA devices
        for i in 0..8 {
            // Check up to 8 devices
            if let Ok(_device) = Device::new_cuda(i) {
                // Get device properties via NVML if available
                let name = self
                    .get_cuda_device_name(i)
                    .unwrap_or_else(|_| format!("CUDA Device {}", i));
                let memory = self.get_cuda_device_memory(i).unwrap_or(0);

                let device = ComputeDevice::new(self.devices.len(), DeviceType::Cuda, name, memory);

                info!(
                    "Found CUDA device: {} ({:.1} GB)",
                    device.name,
                    device.memory_gb()
                );
                self.devices.push(device);
            } else {
                break; // No more CUDA devices
            }
        }

        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn detect_cuda_devices(&mut self) -> Result<()> {
        Ok(())
    }

    /// Get CUDA device name
    #[cfg(feature = "cuda")]
    fn get_cuda_device_name(&self, _index: usize) -> Result<String> {
        // Use NVML to get device name
        use nvml_wrapper::Nvml;
        let nvml = Nvml::init().map_err(|e| anyhow!("NVML init failed: {}", e))?;
        let device = nvml
            .device_by_index(_index as u32)
            .map_err(|e| anyhow!("NVML device access failed: {}", e))?;
        device
            .name()
            .map_err(|e| anyhow!("NVML name failed: {}", e))
    }

    #[cfg(not(feature = "cuda"))]
    fn get_cuda_device_name(&self, _index: usize) -> Result<String> {
        Ok(format!("CUDA Device {}", _index))
    }

    /// Get CUDA device memory
    #[cfg(feature = "cuda")]
    fn get_cuda_device_memory(&self, _index: usize) -> Result<u64> {
        use nvml_wrapper::Nvml;
        let nvml = Nvml::init().map_err(|e| anyhow!("NVML init failed: {}", e))?;
        let device = nvml
            .device_by_index(_index as u32)
            .map_err(|e| anyhow!("NVML device access failed: {}", e))?;
        let info = device
            .memory_info()
            .map_err(|e| anyhow!("NVML memory info failed: {}", e))?;
        Ok(info.total)
    }

    #[cfg(not(feature = "cuda"))]
    fn get_cuda_device_memory(&self, _index: usize) -> Result<u64> {
        Ok(0)
    }

    /// Detect Intel Arc and integrated GPUs
    fn detect_intel_devices(&mut self) -> Result<()> {
        // Check for Intel Arc discrete GPU
        #[cfg(target_os = "windows")]
        {
            if let Some(info) = self.get_intel_arc_info_windows()? {
                let device = ComputeDevice::new(
                    self.devices.len(),
                    DeviceType::IntelArc,
                    info.0, // Use actual name from WMI
                    info.1, // Use actual memory from WMI
                );
                info!(
                    "Found Intel Arc GPU: {} ({:.1} GB)",
                    device.name,
                    device.memory_gb()
                );
                self.devices.push(device);
                return Ok(()); // Only add one Intel Arc device
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            if self.has_intel_arc_gpu()? {
                let device = ComputeDevice::new(
                    self.devices.len(),
                    DeviceType::IntelArc,
                    "Intel Arc GPU".to_string(),
                    self.get_intel_arc_memory()?,
                );
                info!("Found Intel Arc GPU: {:.1} GB", device.memory_gb());
                self.devices.push(device);
            }
        }

        // Check for Intel Xe integrated graphics (Meteor Lake, etc.)
        if self.has_intel_xe_gpu()? {
            let device = ComputeDevice::new(
                self.devices.len(),
                DeviceType::IntelXe,
                "Intel Xe iGPU".to_string(),
                self.get_intel_xe_memory()?,
            );
            info!(
                "Found Intel Xe iGPU (shared memory): {:.1} GB",
                device.memory_gb()
            );
            self.devices.push(device);
        }

        Ok(())
    }

    /// Check if Intel Arc discrete GPU is present
    fn has_intel_arc_gpu(&self) -> Result<bool> {
        // Check Windows registry or sysfs for Intel Arc
        #[cfg(target_os = "windows")]
        {
            let info = self.get_intel_arc_info_windows()?;
            Ok(info.is_some())
        }

        #[cfg(not(target_os = "windows"))]
        {
            // Check /sys/class/drm for Intel Arc
            if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.join("device/vendor").exists() {
                        if let Ok(vendor) = std::fs::read_to_string(path.join("device/vendor")) {
                            if vendor.contains("8086") {
                                // Intel vendor ID
                                if let Ok(device) =
                                    std::fs::read_to_string(path.join("device/device"))
                                {
                                    // Intel Arc has specific device IDs (0x56A0+)
                                    if let Ok(dev_id) = u32::from_str_radix(
                                        device.trim().trim_start_matches("0x"),
                                        16,
                                    ) {
                                        if (0x56A0..=0x56FF).contains(&dev_id) {
                                            return Ok(true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(false)
        }
    }

    /// Get Intel Arc GPU info on Windows (name and VRAM)
    #[cfg(target_os = "windows")]
    fn get_intel_arc_info_windows(&self) -> Result<Option<(String, u64)>> {
        use std::process::Command;

        // Use WMIC to get video controller info
        let output = Command::new("wmic")
            .args(&["path", "win32_VideoController", "get", "name,AdapterRAM"])
            .output()?;

        let output_str = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = output_str.lines().collect();

        // Parse the output - format is:
        // Name                                          AdapterRAM
        // Intel(R) Arc(tm) A770 Graphics                17179869184
        for line in lines.iter().skip(1) {
            // Skip header
            let line = line.trim();
            if line.contains("Arc") || line.contains("Intel Arc") {
                // Split by whitespace and find the number at the end
                let parts: Vec<&str> = line.split_whitespace().collect();

                // Find the GPU name (everything except the last number)
                let name_parts: Vec<&str> = parts
                    .iter()
                    .filter(|p| !p.chars().all(|c| c.is_numeric()))
                    .cloned()
                    .collect();
                let name = name_parts.join(" ");

                // Find the memory (last numeric field)
                let memory = parts
                    .iter()
                    .rev()
                    .find_map(|p| p.parse::<u64>().ok())
                    .unwrap_or(8 * 1024 * 1024 * 1024); // Default to 8GB

                info!(
                    "Detected Intel Arc GPU via WMIC: {} ({:.1} GB)",
                    name,
                    memory as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                return Ok(Some((name, memory)));
            }
        }

        Ok(None)
    }

    /// Get Intel Arc GPU memory
    fn get_intel_arc_memory(&self) -> Result<u64> {
        #[cfg(target_os = "windows")]
        {
            // Try to get actual memory via WMIC
            if let Some(info) = self.get_intel_arc_info_windows()? {
                return Ok(info.1);
            }
        }

        // Fallback: Try to detect based on model name
        #[cfg(target_os = "windows")]
        {
            use std::process::Command;
            let output = Command::new("wmic")
                .args(&["path", "win32_VideoController", "get", "name"])
                .output()?;

            let output_str = String::from_utf8_lossy(&output.stdout);
            let name_lower = output_str.to_lowercase();

            // A770 is 16GB, A750 is 8GB, A580 is 8GB, A380 is 6GB
            if name_lower.contains("a770") {
                info!("Detected Intel Arc A770 (16 GB VRAM)");
                return Ok(16 * 1024 * 1024 * 1024);
            } else if name_lower.contains("a750") || name_lower.contains("a580") {
                info!("Detected Intel Arc A750/A580 (8 GB VRAM)");
                return Ok(8 * 1024 * 1024 * 1024);
            } else if name_lower.contains("a380") || name_lower.contains("a310") {
                info!("Detected Intel Arc A380/A310 (6 GB VRAM)");
                return Ok(6 * 1024 * 1024 * 1024);
            }
        }

        // Default to 8GB
        Ok(8 * 1024 * 1024 * 1024)
    }

    /// Get Intel Arc GPU name
    #[cfg(target_os = "windows")]
    fn get_intel_arc_name(&self) -> Result<String> {
        if let Some(info) = self.get_intel_arc_info_windows()? {
            return Ok(info.0);
        }
        Ok("Intel Arc GPU".to_string())
    }

    /// Check for Intel Xe integrated graphics (Meteor Lake, etc.)
    fn has_intel_xe_gpu(&self) -> Result<bool> {
        #[cfg(target_os = "windows")]
        {
            use std::process::Command;
            let output = Command::new("wmic")
                .args(&["path", "win32_VideoController", "get", "name"])
                .output()?;

            let output_str = String::from_utf8_lossy(&output.stdout);
            // Check for Intel Iris Xe or Intel Graphics (Core Ultra series)
            Ok(output_str.contains("Intel")
                && (output_str.contains("Iris")
                    || output_str.contains("UHD")
                    || output_str.contains("Graphics")))
        }

        #[cfg(not(target_os = "windows"))]
        Ok(false)
    }

    /// Get Intel Xe memory (shared system memory)
    fn get_intel_xe_memory(&self) -> Result<u64> {
        // Intel Xe uses shared system memory
        // Use half of available system RAM as estimate
        let sys_mem = self.get_system_memory();
        Ok(sys_mem / 2)
    }

    /// Get system memory
    fn get_system_memory(&self) -> u64 {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        sys.total_memory()
    }

    /// Add CPU as fallback device
    fn add_cpu_device(&mut self) {
        let sys_mem = self.get_system_memory();
        let device = ComputeDevice::new(
            self.devices.len(),
            DeviceType::Cpu,
            format!("CPU ({} cores)", num_cpus::get()),
            sys_mem,
        );
        info!(
            "Added CPU device with {:.1} GB system memory",
            device.memory_gb()
        );
        self.devices.push(device);
    }

    /// Get total available compute memory
    pub fn total_compute_memory(&self) -> u64 {
        self.devices
            .iter()
            .filter(|d| d.device_type != DeviceType::Cpu)
            .map(|d| d.memory_bytes)
            .sum()
    }

    /// Get total memory across all devices
    pub fn total_memory(&self) -> u64 {
        self.devices.iter().map(|d| d.memory_bytes).sum()
    }

    /// Get device count
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get device by ID
    pub fn get_device(&self, id: usize) -> Option<&ComputeDevice> {
        self.devices.get(id)
    }

    /// Get all devices
    pub fn devices(&self) -> &[ComputeDevice] {
        &self.devices
    }
}

/// Remote server information
#[derive(Debug, Clone)]
pub struct RemoteServer {
    /// Unique server identifier
    pub server_id: String,
    /// Network endpoint (host:port)
    pub endpoint: String,
    /// Average latency in milliseconds
    pub latency_ms: f64,
    /// Devices on this remote server
    pub devices: Vec<ComputeDevice>,
    /// Server status
    pub status: RemoteServerStatus,
    /// When this server was registered
    pub registered_at: Instant,
    /// Last heartbeat received
    pub last_heartbeat: Instant,
}

/// Remote server status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteServerStatus {
    /// Server is connected and healthy
    Online,
    /// Server is connected but slow (high latency)
    Degraded,
    /// Server is disconnected
    Offline,
}

impl std::fmt::Display for RemoteServerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemoteServerStatus::Online => write!(f, "✅ ONLINE"),
            RemoteServerStatus::Degraded => write!(f, "⚠️ DEGRADED"),
            RemoteServerStatus::Offline => write!(f, "❌ OFFLINE"),
        }
    }
}

/// Layer assignment for distributed inference
#[derive(Debug, Clone)]
pub struct LayerAssignment {
    /// Device ID -> assigned layers
    pub device_layers: HashMap<String, Vec<u32>>,
    /// Total layers in the model
    pub total_layers: u32,
    /// When this assignment was made
    pub assigned_at: Instant,
}

/// Loaded model information for display
#[derive(Debug, Clone)]
pub struct LoadedModelInfo {
    /// Model ID/name
    pub model_id: String,
    /// Device the model is loaded on
    pub device: String,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Number of layers (if known)
    pub num_layers: Option<u32>,
}

impl LoadedModelInfo {
    /// Create a new loaded model info
    pub fn new(model_id: String, device: String, size_bytes: u64) -> Self {
        Self {
            model_id,
            device,
            size_bytes,
            num_layers: None,
        }
    }

    /// Set the number of layers
    pub fn with_layers(mut self, num_layers: u32) -> Self {
        self.num_layers = Some(num_layers);
        self
    }

    /// Get size in GB
    pub fn size_gb(&self) -> f32 {
        self.size_bytes as f32 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Complete hardware topology (local + remote)
#[derive(Debug)]
pub struct HardwareTopology {
    /// Local devices
    pub local_devices: Vec<ComputeDevice>,
    /// Remote servers with their devices
    pub remote_servers: HashMap<String, RemoteServer>,
    /// Current layer assignment (if any)
    pub layer_assignment: Option<LayerAssignment>,
    /// When this topology was last updated
    pub last_updated: Instant,
}

impl HardwareTopology {
    /// Create a new empty topology
    pub fn new() -> Self {
        Self {
            local_devices: Vec::new(),
            remote_servers: HashMap::new(),
            layer_assignment: None,
            last_updated: Instant::now(),
        }
    }

    /// Create topology from device manager
    pub fn from_device_manager(dm: &DeviceManager) -> Self {
        Self {
            local_devices: dm.devices().to_vec(),
            remote_servers: HashMap::new(),
            layer_assignment: None,
            last_updated: Instant::now(),
        }
    }

    /// Get total memory across all devices
    pub fn total_memory(&self) -> u64 {
        let local: u64 = self.local_devices.iter().map(|d| d.memory_bytes).sum();
        let remote: u64 = self
            .remote_servers
            .values()
            .flat_map(|s| s.devices.iter())
            .map(|d| d.memory_bytes)
            .sum();
        local + remote
    }

    /// Check if we have remote servers
    pub fn has_remote_servers(&self) -> bool {
        !self.remote_servers.is_empty()
    }

    /// Log a comprehensive summary of the hardware topology
    pub fn log_summary(&self) {
        info!("{}", HLINE);
        info!("                    HARDWARE DISCOVERY");
        info!("{}", HLINE);
        info!("");

        // Feature status
        info!("COMPILED FEATURES:");
        info!(
            "  CUDA support:    {}",
            if cfg!(feature = "cuda") {
                "✅ Enabled"
            } else {
                "❌ Not compiled"
            }
        );
        info!("");

        // Header
        info!("DEVICE                    MEMORY      STATUS");
        info!("{}", SLINE);

        // Local devices - sorted by priority
        let mut local_sorted = self.local_devices.clone();
        local_sorted.sort_by_key(|d| std::cmp::Reverse(d.priority));

        // Count available/unavailable
        let mut available_count = 0;
        let mut available_memory: f32 = 0.0;

        for device in &local_sorted {
            let status_str = match &device.availability {
                DeviceAvailability::Available => {
                    available_count += 1;
                    available_memory += device.memory_gb();
                    "✅ AVAILABLE".to_string()
                }
                DeviceAvailability::Unavailable { reason, .. } => {
                    format!("⚠️ UNAVAILABLE ({})", reason)
                }
            };

            info!(
                "[LOCAL] {:<14} {:>6.1} GB   {}    {}",
                format!("{} #{}", device.device_type, device.id),
                device.memory_gb(),
                status_str,
                device.name
            );

            // Show suggestion for unavailable devices
            if let DeviceAvailability::Unavailable { suggestion, .. } = &device.availability {
                info!("        -- Suggestion: {}", suggestion);
            }
        }

        // Remote devices - grouped by server
        for (server_id, server) in &self.remote_servers {
            for device in &server.devices {
                let status_str = match server.status {
                    RemoteServerStatus::Online => {
                        available_count += 1;
                        available_memory += device.memory_gb();
                        "✅ ONLINE".to_string()
                    }
                    RemoteServerStatus::Degraded => "⚠️ DEGRADED".to_string(),
                    RemoteServerStatus::Offline => "❌ OFFLINE".to_string(),
                };
                info!(
                    "[{}] {:<14} {:>6.1} GB   {}    {}",
                    truncate_server_id(server_id),
                    format!("{} #{}", device.device_type, device.id),
                    device.memory_gb(),
                    status_str,
                    device.name
                );
            }
        }

        info!("{}", SLINE);

        // Summary
        let total_memory: f32 = self
            .local_devices
            .iter()
            .map(ComputeDevice::memory_gb)
            .sum();
        let remote_memory: f32 = self
            .remote_servers
            .values()
            .flat_map(|s| s.devices.iter())
            .map(ComputeDevice::memory_gb)
            .sum();

        info!(
            "TOTAL MEMORY:             {:>6.1} GB",
            total_memory + remote_memory
        );
        info!(
            "USABLE MEMORY:            {:>6.1} GB ({} device(s) available)",
            available_memory, available_count
        );
        info!("");

        // Inference mode
        let inference_mode = if self.has_remote_servers() {
            "DISTRIBUTED"
        } else {
            "SINGLE-NODE"
        };
        info!("Inference Mode: {}", inference_mode);

        // Show which devices will be used
        info!("");
        info!("INFERENCE DEVICES (in priority order):");
        let usable_devices: Vec<_> = local_sorted
            .iter()
            .filter(|d| d.availability.is_available())
            .collect();

        if usable_devices.is_empty() {
            warn!("  ⚠️ No devices available for inference!");
        } else {
            for (i, device) in usable_devices.iter().enumerate() {
                info!(
                    "  {}. {} #{} ({:.1} GB) - {}",
                    i + 1,
                    device.device_type,
                    device.id,
                    device.memory_gb(),
                    device.name
                );
            }
        }

        // Layer assignment info
        if let Some(ref assignment) = self.layer_assignment {
            info!("");
            info!("LOADED MODEL DISTRIBUTION");
            info!("{}", SLINE);
            for (device_id, layers) in &assignment.device_layers {
                if layers.len() <= 4 {
                    info!("  {} -> {:?}", device_id, layers);
                } else {
                    info!(
                        "  {} -> layers [{}, {}, ..., {}, {}] ({} total)",
                        device_id,
                        layers[0],
                        layers[1],
                        layers[layers.len() - 2],
                        layers[layers.len() - 1],
                        layers.len()
                    );
                }
            }
        }

        info!("{}", HLINE);
    }
}

impl Default for HardwareTopology {
    fn default() -> Self {
        Self::new()
    }
}

/// Truncate server ID for display (take first part before hyphen or limit to 8 chars)
fn truncate_server_id(server_id: &str) -> &str {
    // Pick the target byte cap: dash position when present and <= 8,
    // else the 8-byte cap, else the full string. Then walk back to
    // the nearest char boundary so multibyte UTF-8 input from a
    // potentially-untrusted peer doesn't slice mid-codepoint and
    // panic. (server_id is wire-deserialised JSON in protocol.rs;
    // most are ASCII UUIDs but the type is unconstrained `String`.)
    let target_end = match server_id.find('-') {
        Some(idx) if idx <= 8 => idx,
        _ => 8.min(server_id.len()),
    };
    let mut safe_end = target_end;
    while safe_end > 0 && !server_id.is_char_boundary(safe_end) {
        safe_end -= 1;
    }
    &server_id[..safe_end]
}

/// Types of topology change events
#[derive(Debug, Clone, Copy)]
pub enum TopologyEventType {
    DeviceAdded,
    DeviceRemoved,
    ServerJoined,
    ServerLeft,
    LayerRebalance,
    DegradedLatency,
}

/// Log loaded models in a clean format
pub fn log_loaded_models(models: &[LoadedModelInfo]) {
    if models.is_empty() {
        return;
    }

    info!("");
    info!("LOADED MODELS");
    info!("{}", SLINE);

    for model in models {
        let layers_str = model
            .num_layers
            .map(|n| format!("{} layers", n))
            .unwrap_or_else(|| "? layers".to_string());

        info!(
            "📦 {} ({:.1} GB, {}) -> {}",
            model.model_id,
            model.size_gb(),
            layers_str,
            model.device
        );
    }

    // Total
    let total_size: f32 = models.iter().map(LoadedModelInfo::size_gb).sum();
    info!("{}", SLINE);
    info!("TOTAL: {} models ({:.1} GB)", models.len(), total_size);
}

/// Distribution assignment for a single device
#[derive(Debug, Clone)]
pub struct DeviceDistribution {
    /// Device name (e.g., "CUDA #0", "CPU")
    pub device_name: String,
    /// Server ID (None for local)
    pub server_id: Option<String>,
    /// Layer range (start, end inclusive)
    pub layer_range: (u32, u32),
    /// Memory required in bytes
    pub memory_bytes: u64,
    /// Device type
    pub device_type: DeviceType,
}

/// Log a model distribution plan
pub fn log_distribution_plan(
    model_id: &str,
    num_layers: u32,
    distributions: &[DeviceDistribution],
) {
    info!("");
    info!(
        "MODEL DISTRIBUTION PLAN: {} ({} layers)",
        model_id, num_layers
    );
    info!("{}", SLINE);

    for dist in distributions {
        let location = match dist.server_id.as_deref() {
            Some(server) => format!("[{}] {}", server, dist.device_name),
            None => format!("[LOCAL] {}", dist.device_name),
        };

        let memory_gb = dist.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let layers_str = if dist.layer_range.0 == dist.layer_range.1 {
            format!("layer {}", dist.layer_range.0)
        } else {
            format!("layers {}-{}", dist.layer_range.0, dist.layer_range.1)
        };

        info!("{:<25} {:<15} {:>5.1} GB", location, layers_str, memory_gb);
    }

    info!("{}", SLINE);
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_priority() {
        // Test that priority values are correct
        let cuda = ComputeDevice::new(0, DeviceType::Cuda, "CUDA".to_string(), 0);
        let arc = ComputeDevice::new(1, DeviceType::IntelArc, "Arc".to_string(), 0);
        let xe = ComputeDevice::new(2, DeviceType::IntelXe, "Xe".to_string(), 0);
        let cpu = ComputeDevice::new(3, DeviceType::Cpu, "CPU".to_string(), 0);

        assert!(cuda.priority > arc.priority);
        assert!(arc.priority > xe.priority);
        assert!(xe.priority > cpu.priority);
    }

    #[test]
    fn test_compute_device_creation() {
        let device = ComputeDevice::new(
            0,
            DeviceType::Cuda,
            "Test GPU 0".to_string(),
            24 * 1024 * 1024 * 1024,
        );

        assert_eq!(device.id, 0);
        assert_eq!(device.device_type, DeviceType::Cuda);
        assert_eq!(device.memory_gb(), 24.0);
        assert_eq!(device.priority, 100);
    }

    #[test]
    fn truncate_server_id_uses_dash_boundary_or_eight_char_cap() {
        // Standard case: hyphenated short prefix -> truncate at the dash.
        assert_eq!(truncate_server_id("srv-12345-abc"), "srv");
        assert_eq!(truncate_server_id("node-1"), "node");
        assert_eq!(truncate_server_id("a-b"), "a");
        // Hyphen past the 8-byte cap -> fall back to 8-byte cap.
        assert_eq!(truncate_server_id("longserveridname-tail"), "longserv");
        // No hyphen, longer than 8 -> 8-byte cap.
        assert_eq!(truncate_server_id("verylongservernameattheend"), "verylong");
        // No hyphen, <= 8 -> return as-is.
        assert_eq!(truncate_server_id("short"), "short");
        assert_eq!(truncate_server_id("exactly8"), "exactly8");
        // Boundary: dash exactly at position 8 -> take all 8 chars.
        assert_eq!(truncate_server_id("eightchr-tail"), "eightchr");
        // Empty string -> empty.
        assert_eq!(truncate_server_id(""), "");
    }

    #[test]
    fn truncate_server_id_handles_dash_at_position_zero() {
        // Leading hyphen: idx=0, idx <= 8 -> take &[..0] = "".
        // Pin so a future refactor that uses `>` instead of `<=` doesn't
        // shift this case to the 8-cap path (would return "-tail-of"
        // instead of "").
        assert_eq!(truncate_server_id("-leading-dash"), "");
    }

    #[test]
    fn truncate_server_id_does_not_panic_on_multibyte_utf8() {
        // server_id is wire-deserialised JSON from a remote peer.
        // A malicious peer sending multibyte UTF-8 must not crash
        // the server with a slice-at-non-char-boundary panic.
        //
        // "1234567ñ-tail": ñ occupies bytes 7-8 (the 0xC3 0xB1
        // continuation pair). find('-') returns 9 (past the 8 cap),
        // so target_end falls back to 8 - which lands MID-codepoint
        // on the ñ's continuation byte. The walk-back-to-char-
        // boundary step must clamp to 7 (just before the ñ starts)
        // rather than panic.
        let out = truncate_server_id("1234567ñ-tail");
        assert_eq!(out, "1234567");
        // Multibyte before the 8 cap entirely -> take all of it up
        // to the boundary nearest 8.
        // "ñabcdefgh-tail": ñ at bytes 0-1, then 8 ASCII chars.
        // target_end=8 -> "ñabcdef" (7 chars, 8 bytes - fits exactly).
        let out = truncate_server_id("ñabcdefgh-tail");
        assert_eq!(out, "ñabcdef");
        // Pure multibyte: each char = 3 bytes. "あいうえお" (5x3=15B).
        // No dash, len>8 -> cap at 8 -> walk back to byte 6 (after 2
        // full chars). Result = "あい".
        let out = truncate_server_id("あいうえお");
        assert_eq!(out, "あい");
        // Empty multibyte case shouldn't panic either.
        assert_eq!(truncate_server_id("ñ"), "ñ");
    }

    #[test]
    fn test_available_memory_calculation() {
        let mut device = ComputeDevice::new(
            0,
            DeviceType::Cuda,
            "Test GPU".to_string(),
            10 * 1024 * 1024 * 1024, // 10 GB
        );

        // With 0% utilization, should have ~80% available (20% reserved)
        let available = device.available_memory_for_model();
        assert!(available > 7 * 1024 * 1024 * 1024);
        assert!(available < 9 * 1024 * 1024 * 1024);

        // With 50% utilization
        device.memory_utilization = 0.5;
        let available = device.available_memory_for_model();
        assert!(available < 4 * 1024 * 1024 * 1024);
    }
}
