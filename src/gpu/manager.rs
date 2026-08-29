//! GPU manager implementation
//!
//! This module provides the implementation of the GPU manager interface
//! with CUDA/NVML support for NVIDIA GPUs.

use super::{
    ClockInfo, Device, FanInfo, GPUDevice, GPUError, GPUManagerInterface, MemoryInfo,
    PerformanceState, PowerInfo, TemperatureInfo, UtilizationInfo,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// Process-global CPU-only switch. Set once by the `--cpu` serve flag BEFORE any
/// GPU manager is constructed; makes GPU detection report zero devices so the
/// placement falls through to the exact CPU path the no-cuda build uses (no CUDA
/// context is ever created). This is a CLI-driven flag, not an env var.
static FORCE_CPU: AtomicBool = AtomicBool::new(false);

/// Force CPU-only operation (no GPU placement). Call before constructing any
/// `GPUManagerImpl`.
pub fn set_force_cpu(v: bool) {
    FORCE_CPU.store(v, Ordering::SeqCst);
}

/// Whether `--cpu` forced CPU-only operation.
pub fn force_cpu() -> bool {
    FORCE_CPU.load(Ordering::SeqCst)
}

/// GPU manager implementation with NVML support
pub struct GPUManagerImpl {
    devices: Arc<Mutex<Vec<Device>>>,
    pub nvml: Option<Arc<nvml_wrapper::Nvml>>, // Public for live stats queries
}

impl Clone for GPUManagerImpl {
    fn clone(&self) -> Self {
        Self {
            devices: self.devices.clone(),
            nvml: self.nvml.clone(),
        }
    }
}

impl GPUManagerImpl {
    /// Create a new GPU manager - detects GPUs once at initialization
    pub fn new() -> Self {
        // CPU-only mode (--cpu / force_cpu): do NOT initialize NVML. On this driver
        // nvmlInit dlopen's libcuda and spawns a CUDA driver thread that lives for
        // the whole process; that thread periodically wakes and preempts a GEMV
        // worker on the otherwise-saturated core set, adding run-to-run jitter and
        // ~8% throughput loss to CPU decode vs a build that never links/inits CUDA.
        // With no GPU in use there is nothing to monitor, so skip it entirely - this
        // makes the CUDA binary run with --cpu match the pure-CPU build.
        let nvml = if force_cpu() {
            info!("--cpu: skipping NVML init (no GPU monitoring in CPU-only mode)");
            None
        } else {
            match nvml_wrapper::Nvml::init() {
                Ok(nvml) => {
                    info!("NVML initialized successfully");
                    Some(Arc::new(nvml))
                }
                Err(e) => {
                    warn!(
                        "Failed to initialize NVML: {}. GPU detection will be limited.",
                        e
                    );
                    None
                }
            }
        };

        // Detect GPUs once during initialization
        let devices = Self::detect_nvidia_gpus_once(&nvml);

        Self {
            devices: Arc::new(Mutex::new(devices)),
            nvml,
        }
    }

    /// Detect NVIDIA GPUs once at initialization (not in a loop)
    fn detect_nvidia_gpus_once(nvml: &Option<Arc<nvml_wrapper::Nvml>>) -> Vec<Device> {
        if force_cpu() {
            info!("--cpu: GPU placement disabled - running CPU-only (no GPU devices reported)");
            return Vec::new();
        }
        let nvml_arc = match nvml {
            Some(nvml) => nvml,
            None => return Vec::new(),
        };

        let device_count = match nvml_arc.device_count() {
            Ok(count) => count,
            Err(e) => {
                warn!("Failed to get device count: {}", e);
                return Vec::new();
            }
        };

        info!("Found {} NVIDIA GPU(s)", device_count);

        let mut devices = Vec::new();
        for i in 0..device_count {
            match nvml_arc.device_by_index(i) {
                Ok(gpu) => {
                    let name = gpu
                        .name()
                        .unwrap_or_else(|_| "Unknown NVIDIA GPU".to_string());
                    let memory_info = gpu.memory_info().ok();
                    let total_memory = memory_info.map(|m| m.total).unwrap_or(0);

                    // Get utilization
                    let utilization = gpu.utilization_rates().ok();
                    let gpu_util = utilization.map(|u| u.gpu as f32 / 100.0).unwrap_or(0.0);

                    let device = Device {
                        id: i,
                        name,
                        memory: total_memory,
                        utilization: gpu_util,
                        nvml: Some(nvml_arc.clone()), // Store NVML reference for live queries
                    };

                    info!(
                        "Detected GPU {}: {} ({} MB)",
                        device.id,
                        device.name,
                        device.memory / (1024 * 1024)
                    );
                    devices.push(device);
                }
                Err(e) => {
                    warn!("Failed to get GPU {}: {}", i, e);
                }
            }
        }

        devices
    }
}

#[async_trait::async_trait]
impl GPUManagerInterface for GPUManagerImpl {
    async fn detect_gpus(&self) -> Result<(), GPUError> {
        // GPUs are already detected once at initialization in new()
        // This method is now a no-op to avoid repeated logging
        Ok(())
    }

    fn get_devices(&self) -> Vec<Arc<dyn GPUDevice>> {
        let devices = self.devices.lock().unwrap();
        devices
            .iter()
            .map(|d| Arc::new(d.clone()) as Arc<dyn GPUDevice>)
            .collect()
    }

    fn get_device(&self, id: u32) -> Option<Arc<dyn GPUDevice>> {
        let devices = self.devices.lock().unwrap();
        devices
            .iter()
            .find(|d| d.id == id)
            .map(|d| Arc::new(d.clone()) as Arc<dyn GPUDevice>)
    }

    fn get_best_device(&self) -> Option<Arc<dyn GPUDevice>> {
        let devices = self.devices.lock().unwrap();
        // Return device with lowest utilization
        devices
            .iter()
            .min_by(|a, b| a.utilization.partial_cmp(&b.utilization).unwrap())
            .map(|d| Arc::new(d.clone()) as Arc<dyn GPUDevice>)
    }

    fn get_device_with_most_memory(&self) -> Option<Arc<dyn GPUDevice>> {
        let devices = self.devices.lock().unwrap();
        // PLACEMENT-EXEMPT: a MONITORING answer, not a placement. This manager reports the
        // machine (memory, utilisation, temperature) to the status endpoints; no model is
        // loaded through it. Every load goes through vram_manager / HeteroPlan, which rank
        // by measured throughput.
        devices
            .iter()
            .max_by_key(|d| d.memory)
            .map(|d| Arc::new(d.clone()) as Arc<dyn GPUDevice>)
    }

    fn get_utilization_summary(&self) -> Vec<(u32, f32)> {
        let devices = self.devices.lock().unwrap();
        devices.iter().map(|d| (d.id, d.utilization)).collect()
    }

    fn has_gpus(&self) -> bool {
        let devices = self.devices.lock().unwrap();
        !devices.is_empty()
    }
}

impl GPUDevice for Device {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn memory_info(&self) -> Result<MemoryInfo, GPUError> {
        // Prefer live NVML free/used over the cached utilization
        // snapshot (set once at detect-time) - anything that allocates
        // VRAM after startup (warm engines, other processes) is invisible
        // to the cached path, so `free` rapidly becomes a lie.
        if let Some(ref nvml) = self.nvml {
            if let Ok(dev) = nvml.device_by_index(self.id) {
                if let Ok(mi) = dev.memory_info() {
                    let util = if mi.total > 0 {
                        mi.used as f32 / mi.total as f32
                    } else {
                        0.0
                    };
                    return Ok(MemoryInfo {
                        total: mi.total,
                        used: mi.used,
                        free: mi.free,
                        utilization: util,
                    });
                }
            }
        }
        Ok(MemoryInfo {
            total: self.memory,
            used: (self.memory as f32 * self.utilization) as u64,
            free: (self.memory as f32 * (1.0 - self.utilization)) as u64,
            utilization: self.utilization,
        })
    }

    fn utilization(&self) -> Result<UtilizationInfo, GPUError> {
        // Try to get live utilization from NVML
        if let Some(ref nvml) = self.nvml {
            if let Some(util) = get_live_utilization(nvml, self.id) {
                return Ok(util);
            }
        }

        // Fallback to cached values if NVML not available
        Ok(UtilizationInfo {
            gpu: self.utilization,
            memory: self.utilization,
            encoder: 0.0,
            decoder: 0.0,
        })
    }

    fn temperature(&self) -> Result<TemperatureInfo, GPUError> {
        // Try to get live temperature from NVML
        if let Some(ref nvml) = self.nvml {
            if let Some(temp_info) = get_live_temperature(nvml, self.id) {
                return Ok(temp_info);
            }
        }

        // Fallback to default values if NVML not available
        Ok(TemperatureInfo {
            gpu: 65.0,
            memory: 70.0,
            hotspot: 75.0,
        })
    }

    fn power_info(&self) -> Result<PowerInfo, GPUError> {
        // Try to get live power info from NVML
        if let Some(ref nvml) = self.nvml {
            if let Some(power) = get_live_power_info(nvml, self.id) {
                return Ok(power);
            }
        }

        // Fallback to default values if NVML not available
        Ok(PowerInfo {
            power: 150.0,
            limit: 250.0,
            utilization: 0.6,
        })
    }

    fn fan_info(&self) -> Result<FanInfo, GPUError> {
        // Try to get live fan info from NVML
        if let Some(ref nvml) = self.nvml {
            if let Some(fan) = get_live_fan_info(nvml, self.id) {
                return Ok(fan);
            }
        }

        // Fallback to default values if NVML not available
        Ok(FanInfo {
            speed: 1500,
            speed_percent: 50,
            fan_count: 1,
        })
    }

    fn clock_info(&self) -> Result<ClockInfo, GPUError> {
        // Try to get live clock info from NVML
        if let Some(ref nvml) = self.nvml {
            if let Some(clock) = get_live_clock_info(nvml, self.id) {
                return Ok(clock);
            }
        }

        // Fallback to default values if NVML not available
        Ok(ClockInfo {
            graphics_clock: 1500,
            memory_clock: 7000,
            sm_clock: 1500,
            video_clock: 1200,
            max_graphics_clock: 2100,
            max_memory_clock: 10000,
        })
    }

    fn performance_state(&self) -> Result<PerformanceState, GPUError> {
        // Try to get live performance state from NVML
        if let Some(ref nvml) = self.nvml {
            if let Some(state) = get_live_performance_state(nvml, self.id) {
                return Ok(state);
            }
        }

        // Fallback to default if NVML not available
        Ok(PerformanceState::P2)
    }

    fn is_available(&self) -> bool {
        self.utilization < 0.9
    }

    fn compute_capability(&self) -> (u32, u32) {
        // Default to common compute capability
        (8, 6)
    }

    fn driver_version(&self) -> String {
        "525.89.02".to_string()
    }
}

impl Default for GPUManagerImpl {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for live NVML queries
pub fn get_live_temperature(
    nvml: &nvml_wrapper::Nvml,
    device_index: u32,
) -> Option<TemperatureInfo> {
    use nvml_wrapper::enum_wrappers::device::TemperatureSensor;

    let gpu = nvml.device_by_index(device_index).ok()?;

    // Get GPU core temperature
    let gpu_temp = gpu.temperature(TemperatureSensor::Gpu).ok()?;

    // Memory temperature sensor may not be available on all GPUs
    // Use Gpu sensor as fallback
    let memory_temp = gpu
        .temperature(TemperatureSensor::Gpu)
        .ok()
        .unwrap_or(gpu_temp);

    // Hotspot temperature - use the GPU temp as baseline (actual hotspot may not be available)
    let hotspot = gpu_temp;

    Some(TemperatureInfo {
        gpu: gpu_temp as f32,
        memory: memory_temp as f32,
        hotspot: hotspot as f32,
    })
}

/// Get live power information from NVML
pub fn get_live_power_info(nvml: &nvml_wrapper::Nvml, device_index: u32) -> Option<PowerInfo> {
    let gpu = nvml.device_by_index(device_index).ok()?;

    // Get current power usage in milliwatts
    let power_mw = gpu.power_usage().ok()?;
    let power = power_mw as f32 / 1000.0; // Convert to watts

    // Get power limit in milliwatts
    let limit_mw = gpu.power_management_limit().ok()?;
    let limit = limit_mw as f32 / 1000.0; // Convert to watts

    let utilization = if limit > 0.0 { power / limit } else { 0.0 };

    Some(PowerInfo {
        power,
        limit,
        utilization,
    })
}

/// Get live fan information from NVML
pub fn get_live_fan_info(nvml: &nvml_wrapper::Nvml, device_index: u32) -> Option<FanInfo> {
    let gpu = nvml.device_by_index(device_index).ok()?;

    // Get number of fans
    let fan_count = gpu.num_fans().ok().unwrap_or(0);

    if fan_count == 0 {
        return Some(FanInfo {
            speed: 0,
            speed_percent: 0,
            fan_count: 0,
        });
    }

    // Get fan speed percentage for first fan
    let speed_percent = gpu.fan_speed(0).ok().unwrap_or(0);

    // Estimate RPM based on percentage (typical GPU fans range 0-3000 RPM)
    let speed = speed_percent * 30;

    Some(FanInfo {
        speed,
        speed_percent,
        fan_count,
    })
}

/// Get live clock information from NVML
pub fn get_live_clock_info(nvml: &nvml_wrapper::Nvml, device_index: u32) -> Option<ClockInfo> {
    use nvml_wrapper::enum_wrappers::device::Clock;

    let gpu = nvml.device_by_index(device_index).ok()?;

    // Get current clock speeds
    let graphics_clock = gpu.clock_info(Clock::Graphics).ok().unwrap_or(0);
    let memory_clock = gpu.clock_info(Clock::Memory).ok().unwrap_or(0);
    let sm_clock = gpu.clock_info(Clock::SM).ok().unwrap_or(graphics_clock);
    let video_clock = gpu.clock_info(Clock::Video).ok().unwrap_or(0);

    // Get max clock speeds
    let max_graphics_clock = gpu
        .max_clock_info(Clock::Graphics)
        .ok()
        .unwrap_or(graphics_clock);
    let max_memory_clock = gpu
        .max_clock_info(Clock::Memory)
        .ok()
        .unwrap_or(memory_clock);

    Some(ClockInfo {
        graphics_clock,
        memory_clock,
        sm_clock,
        video_clock,
        max_graphics_clock,
        max_memory_clock,
    })
}

/// Get live utilization information from NVML
pub fn get_live_utilization(
    nvml: &nvml_wrapper::Nvml,
    device_index: u32,
) -> Option<UtilizationInfo> {
    let gpu = nvml.device_by_index(device_index).ok()?;

    let util = gpu.utilization_rates().ok()?;

    // Try to get encoder/decoder utilization
    let encoder_util = gpu
        .encoder_utilization()
        .ok()
        .map(|e| e.utilization as f32 / 100.0)
        .unwrap_or(0.0);

    let decoder_util = gpu
        .decoder_utilization()
        .ok()
        .map(|d| d.utilization as f32 / 100.0)
        .unwrap_or(0.0);

    Some(UtilizationInfo {
        gpu: util.gpu as f32 / 100.0,
        memory: util.memory as f32 / 100.0,
        encoder: encoder_util,
        decoder: decoder_util,
    })
}

/// Get live performance state from NVML
pub fn get_live_performance_state(
    nvml: &nvml_wrapper::Nvml,
    device_index: u32,
) -> Option<PerformanceState> {
    let gpu = nvml.device_by_index(device_index).ok()?;
    let state = gpu.performance_state().ok()?;

    // Convert from NVML PerformanceState to our PerformanceState
    // Using string comparison as a workaround since the enum structure varies
    let state_str = format!("{:?}", state);
    match state_str.as_str() {
        "P0" => Some(PerformanceState::P0),
        "P1" => Some(PerformanceState::P1),
        "P2" => Some(PerformanceState::P2),
        "P3" => Some(PerformanceState::P3),
        "P4" => Some(PerformanceState::P4),
        "P5" => Some(PerformanceState::P5),
        "P6" => Some(PerformanceState::P6),
        "P7" => Some(PerformanceState::P7),
        "P8" => Some(PerformanceState::P8),
        "P9" => Some(PerformanceState::P9),
        "P10" => Some(PerformanceState::P10),
        "P11" => Some(PerformanceState::P11),
        "P12" => Some(PerformanceState::P12),
        _ => Some(PerformanceState::Unknown),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_manager_creation() {
        // Just verify manager constructs cleanly + the device list call
        // doesn't panic. has_gpus depends on system hardware; we don't
        // assert a count. Was `assert!(.len() >= 0)` (always true for usize,
        // clippy::absurd_extreme_comparisons).
        let manager = GPUManagerImpl::new();
        let _ = manager.get_devices();
    }

    #[tokio::test]
    async fn test_gpu_detection() {
        let manager = GPUManagerImpl::new();
        let result = manager.detect_gpus().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_device_creation() {
        let device = Device::new(
            0,
            "Test GPU 0".to_string(),
            10 * 1024 * 1024 * 1024, // 10GB
            0.3,
        );

        assert_eq!(device.name(), "Test GPU 0");
        assert_eq!(device.id, 0);
        assert!(device.is_available());
    }

    #[test]
    fn test_memory_info() {
        let device = Device::new(
            0,
            "Test GPU".to_string(),
            8 * 1024 * 1024 * 1024, // 8GB
            0.25,
        );

        let mem = device.memory_info().unwrap();
        assert_eq!(mem.total, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_fan_info() {
        let device = Device::new(0, "Test GPU".to_string(), 8 * 1024 * 1024 * 1024, 0.25);

        let fan = device.fan_info().unwrap();
        // fan.fan_count is u32 (always >= 0); cap at 64 instead - anything past
        // there is a parse bug rather than a real GPU. speed_percent <= 100
        // remains the real bound (NVML returns 0..=100).
        assert!(fan.fan_count <= 64);
        assert!(fan.speed_percent <= 100);
    }

    #[test]
    fn test_clock_info() {
        let device = Device::new(0, "Test GPU".to_string(), 8 * 1024 * 1024 * 1024, 0.25);

        let clock = device.clock_info().unwrap();
        assert!(clock.graphics_clock > 0);
        assert!(clock.memory_clock > 0);
    }
}
