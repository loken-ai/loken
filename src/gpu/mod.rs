//! GPU management functionality
//!
//! This module provides GPU detection and management capabilities for
//! the inference engine.

use std::sync::Arc;

mod device;
mod manager;

pub use device::Device;
pub use manager::GPUManagerImpl;

// Re-export helper functions for live NVML queries
pub use manager::{
    get_live_clock_info, get_live_fan_info, get_live_power_info, get_live_temperature,
    get_live_utilization,
};
// `--cpu` force-CPU switch (set before any GPUManagerImpl is constructed).
pub use manager::{force_cpu, set_force_cpu};

/// GPU memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total: u64,       // Total memory in bytes
    pub used: u64,        // Used memory in bytes
    pub free: u64,        // Free memory in bytes
    pub utilization: f32, // Memory utilization 0.0 to 1.0
}

/// GPU utilization information
#[derive(Debug, Clone)]
pub struct UtilizationInfo {
    pub gpu: f32,     // GPU utilization 0.0 to 1.0
    pub memory: f32,  // Memory utilization 0.0 to 1.0
    pub encoder: f32, // Encoder utilization 0.0 to 1.0
    pub decoder: f32, // Decoder utilization 0.0 to 1.0
}

/// GPU temperature information
#[derive(Debug, Clone)]
pub struct TemperatureInfo {
    pub gpu: f32,     // GPU temperature in Celsius
    pub memory: f32,  // Memory temperature in Celsius
    pub hotspot: f32, // Hotspot temperature in Celsius
}

/// GPU power information
#[derive(Debug, Clone)]
pub struct PowerInfo {
    pub power: f32,       // Current power draw in watts
    pub limit: f32,       // Power limit in watts
    pub utilization: f32, // Power utilization 0.0 to 1.0
}

/// GPU fan information
#[derive(Debug, Clone)]
pub struct FanInfo {
    pub speed: u32,         // Fan speed in RPM
    pub speed_percent: u32, // Fan speed percentage (0-100)
    pub fan_count: u32,     // Number of fans
}

/// GPU clock information
#[derive(Debug, Clone)]
pub struct ClockInfo {
    pub graphics_clock: u32,     // Graphics clock in MHz
    pub memory_clock: u32,       // Memory clock in MHz
    pub sm_clock: u32,           // SM (Streaming Multiprocessor) clock in MHz
    pub video_clock: u32,        // Video encoder/decoder clock in MHz
    pub max_graphics_clock: u32, // Max graphics clock in MHz
    pub max_memory_clock: u32,   // Max memory clock in MHz
}

/// GPU performance state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceState {
    P0, // Maximum performance
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9, // Minimum performance
    P10,
    P11,
    P12,
    Unknown,
}

/// GPU error types
#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("GPU not found: {0}")]
    NotFound(String),
    #[error("GPU initialization failed: {0}")]
    Initialization(String),
    #[error("GPU operation failed: {0}")]
    Operation(String),
    #[error("GPU not supported")]
    NotSupported,
    #[error("GPU detection failed: {0}")]
    Detection(String),
}

/// GPU device interface
pub trait GPUDevice: Send + Sync + 'static {
    /// Get device name
    fn name(&self) -> String;

    /// Get device memory information
    fn memory_info(&self) -> Result<MemoryInfo, GPUError>;

    /// Get device utilization
    fn utilization(&self) -> Result<UtilizationInfo, GPUError>;

    /// Get device temperature
    fn temperature(&self) -> Result<TemperatureInfo, GPUError>;

    /// Get device power information
    fn power_info(&self) -> Result<PowerInfo, GPUError>;

    /// Get device fan information
    fn fan_info(&self) -> Result<FanInfo, GPUError>;

    /// Get device clock information
    fn clock_info(&self) -> Result<ClockInfo, GPUError>;

    /// Get device performance state
    fn performance_state(&self) -> Result<PerformanceState, GPUError>;

    /// Check if device is available for inference
    fn is_available(&self) -> bool;

    /// Get device compute capability
    fn compute_capability(&self) -> (u32, u32);

    /// Get device driver version
    fn driver_version(&self) -> String;
}

/// GPU manager interface
#[async_trait::async_trait]
pub trait GPUManagerInterface: Send + Sync + 'static {
    /// Detect available GPUs
    async fn detect_gpus(&self) -> Result<(), GPUError>;

    /// Get all available devices
    fn get_devices(&self) -> Vec<Arc<dyn GPUDevice>>;

    /// Get device by ID
    fn get_device(&self, id: u32) -> Option<Arc<dyn GPUDevice>>;

    /// Get best available device for inference
    fn get_best_device(&self) -> Option<Arc<dyn GPUDevice>>;

    /// Get device with most free memory
    fn get_device_with_most_memory(&self) -> Option<Arc<dyn GPUDevice>>;

    /// Get device utilization summary
    fn get_utilization_summary(&self) -> Vec<(u32, f32)>;

    /// Check if any GPU is available
    fn has_gpus(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_info_creation() {
        let memory = MemoryInfo {
            total: 8 * 1024 * 1024 * 1024, // 8GB
            used: 2 * 1024 * 1024 * 1024,  // 2GB
            free: 6 * 1024 * 1024 * 1024,  // 6GB
            utilization: 0.25,
        };

        assert_eq!(memory.total, 8 * 1024 * 1024 * 1024);
        assert_eq!(memory.utilization, 0.25);
    }

    #[test]
    fn test_performance_state() {
        // Round-trip: each variant compares equal to itself and not to others.
        // Was a series of `match x { Variant => assert!(true), _ => assert!(false, ...) }`
        // blocks (clippy::assertions_on_constants) - `assert_eq!`/`assert_ne!`
        // expresses the same intent in two lines per variant instead of four.
        assert_eq!(PerformanceState::P0, PerformanceState::P0);
        assert_ne!(PerformanceState::P0, PerformanceState::P9);

        assert_eq!(PerformanceState::P9, PerformanceState::P9);
        assert_ne!(PerformanceState::P9, PerformanceState::Unknown);

        assert_eq!(PerformanceState::Unknown, PerformanceState::Unknown);
        assert_ne!(PerformanceState::Unknown, PerformanceState::P0);
    }
}
