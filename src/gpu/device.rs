//! GPU device implementation
//!
//! This module provides the implementation of the GPU device interface.

use std::sync::Arc;

/// GPU device implementation
#[derive(Debug, Clone)]
pub struct Device {
    pub id: u32,
    pub name: String,
    pub memory: u64,                           // in bytes
    pub utilization: f32,                      // 0.0 to 1.0
    pub nvml: Option<Arc<nvml_wrapper::Nvml>>, // NVML reference for live queries
}

impl Device {
    /// Create a new GPU device
    pub fn new(id: u32, name: String, memory: u64, utilization: f32) -> Self {
        Self {
            id,
            name,
            memory,
            utilization,
            nvml: None, // Default to None for backward compatibility
        }
    }

    /// Get device memory in GB
    pub fn memory_gb(&self) -> f32 {
        self.memory as f32 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get device memory in MB
    pub fn memory_mb(&self) -> f32 {
        self.memory as f32 / (1024.0 * 1024.0)
    }

    /// Check if device has sufficient memory
    pub fn has_sufficient_memory(&self, required_mb: u64) -> bool {
        self.memory >= required_mb * 1024 * 1024
    }

    /// Get device status summary
    pub fn status_summary(&self) -> String {
        format!(
            "Device {}: {} ({} GB, {}% utilization)",
            self.id,
            self.name,
            self.memory_gb(),
            self.utilization * 100.0
        )
    }
}

impl Default for Device {
    fn default() -> Self {
        Self {
            id: 0,
            name: "Unknown GPU".to_string(),
            memory: 0,
            utilization: 0.0,
            nvml: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let device = Device::new(
            0,
            "Test GPU 0".to_string(),
            10 * 1024 * 1024 * 1024, // 10GB
            0.3,
        );

        assert_eq!(device.id, 0);
        assert_eq!(device.name, "Test GPU 0");
        assert_eq!(device.memory, 10 * 1024 * 1024 * 1024);
        assert_eq!(device.utilization, 0.3);
    }

    #[test]
    fn test_device_memory_calculations() {
        let device = Device::new(
            0,
            "Test GPU".to_string(),
            8 * 1024 * 1024 * 1024, // 8GB
            0.5,
        );

        assert_eq!(device.memory_gb(), 8.0);
        assert_eq!(device.memory_mb(), 8192.0);
    }

    #[test]
    fn test_sufficient_memory() {
        let device = Device::new(
            0,
            "Test GPU".to_string(),
            8 * 1024 * 1024 * 1024, // 8GB
            0.5,
        );

        assert!(device.has_sufficient_memory(4096)); // 4GB required
        assert!(!device.has_sufficient_memory(12288)); // 12GB required
    }

    #[test]
    fn test_status_summary() {
        let device = Device::new(
            0,
            "Test GPU 0".to_string(),
            10 * 1024 * 1024 * 1024, // 10GB
            0.3,
        );

        let summary = device.status_summary();
        assert!(summary.contains("Device 0"));
        assert!(summary.contains("Test GPU 0"));
        assert!(summary.contains("10")); // memory in GB
        assert!(summary.contains("30")); // utilization percentage
    }
}
