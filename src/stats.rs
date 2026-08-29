//! System statistics monitoring
//!
//! Periodically logs memory, GPU, and model statistics

use crate::gpu::GPUManagerInterface;
use crate::gpu::{
    get_live_clock_info, get_live_power_info, get_live_temperature, get_live_utilization,
};
use std::sync::Arc;
use std::time::Duration;
use sysinfo::System;
use tokio::sync::RwLock;
use tracing::info;

/// Per-GPU statistics
#[derive(Debug, Clone)]
pub struct GpuStats {
    pub index: u32,
    pub name: String,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub utilization: Option<f32>,
    pub temp: Option<f32>,
    pub power: Option<f32>,
    pub power_limit: Option<f32>,
    pub graphics_clock: Option<u32>,
}

/// System statistics
#[derive(Debug, Clone)]
pub struct SystemStats {
    pub cpu_usage: f32,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub memory_percent: f32,
    // Per-GPU stats. The earlier struct also carried 21 single-GPU
    // legacy fields (gpu_memory_used_gb, gpu_temp, etc.) "for backward
    // compat with GUI" - the GUI actually consumes the full Vec via
    // `gpus[i].*`, so the legacy mirror was pure dead-write state and
    // has been dropped.
    pub gpus: Vec<GpuStats>,
    // Model info
    pub loaded_models: usize,
    pub total_model_size_gb: f64,
}

impl Default for SystemStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemStats {
    pub fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_used_gb: 0.0,
            memory_total_gb: 0.0,
            memory_percent: 0.0,
            gpus: Vec::new(),
            loaded_models: 0,
            total_model_size_gb: 0.0,
        }
    }
}

/// Statistics monitor
pub struct StatsMonitor {
    system: Arc<RwLock<System>>,
    gpu_manager: Option<Arc<crate::gpu::GPUManagerImpl>>,
}

impl StatsMonitor {
    /// Create a new stats monitor
    pub fn new(gpu_manager: Option<Arc<crate::gpu::GPUManagerImpl>>) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self {
            system: Arc::new(RwLock::new(system)),
            gpu_manager,
        }
    }

    /// Get current system statistics
    pub async fn get_stats(&self, loaded_models: usize, total_model_size: u64) -> SystemStats {
        let mut stats = SystemStats::new();

        // Refresh system info. Only CPU% + memory are reported, so refresh JUST
        // those - NOT refresh_all(), which also re-scans every process, disk,
        // network and component in /proc each tick. On a memory-bandwidth-bound
        // CPU decode that full system scan is a periodic stall that adds
        // run-to-run jitter for no benefit (we never read process/disk stats).
        {
            let mut sys = self.system.write().await;
            sys.refresh_cpu_usage();
            sys.refresh_memory();

            stats.cpu_usage = sys.global_cpu_usage();

            let memory_used = sys.used_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
            let memory_total = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

            stats.memory_used_gb = memory_used;
            stats.memory_total_gb = memory_total;
            stats.memory_percent = (memory_used / memory_total * 100.0) as f32;
        }

        // Get GPU stats directly from NVML for live updates.
        // Skipped in CPU-only mode (--cpu / force_cpu): with no GPU in use the
        // periodic NVML driver queries (util/temp/power/clock per device) are
        // pure overhead - and on a memory-bandwidth-bound CPU decode they hold
        // the driver lock long enough to periodically stall the GEMV worker
        // pool, costing ~8% throughput + adding run-to-run jitter (measured:
        // CPU decode 4.7->5.1 tok/s once gated). No GPU stats to report anyway.
        if !crate::gpu::force_cpu() {
            if let Some(ref gpu_manager) = self.gpu_manager {
                let _ = gpu_manager.detect_gpus().await;

                if let Some(nvml) = &gpu_manager.nvml {
                    let device_count = nvml.device_count().unwrap_or(0);
                    for i in 0..device_count {
                        if let Ok(gpu) = nvml.device_by_index(i) {
                            let name = gpu.name().unwrap_or_else(|_| format!("GPU {}", i));
                            let (mem_used, mem_total) = gpu
                                .memory_info()
                                .map(|m| (m.used as f64 / 1e9, m.total as f64 / 1e9))
                                .unwrap_or((0.0, 0.0));
                            let util = get_live_utilization(nvml, i).map(|u| u.gpu);
                            let temp = get_live_temperature(nvml, i).map(|t| t.gpu);
                            let (power, power_limit) = get_live_power_info(nvml, i)
                                .map(|p| (Some(p.power), Some(p.limit)))
                                .unwrap_or((None, None));
                            let clock = get_live_clock_info(nvml, i).map(|c| c.graphics_clock);

                            stats.gpus.push(GpuStats {
                                index: i,
                                name,
                                memory_used_gb: mem_used,
                                memory_total_gb: mem_total,
                                utilization: util,
                                temp,
                                power,
                                power_limit,
                                graphics_clock: clock,
                            });
                        }
                    }
                }
            }
        } // !force_cpu

        stats.loaded_models = loaded_models;
        stats.total_model_size_gb = total_model_size as f64 / (1024.0 * 1024.0 * 1024.0);

        stats
    }

    /// Log statistics
    pub async fn log_stats(&self, loaded_models: usize, total_model_size: u64) {
        let stats = self.get_stats(loaded_models, total_model_size).await;

        // Build per-GPU info strings
        let gpu_info = if stats.gpus.is_empty() {
            " | GPU: Not detected".to_string()
        } else {
            let mut gpu_parts: Vec<String> = Vec::new();
            for g in &stats.gpus {
                let pct = if g.memory_total_gb > 0.0 {
                    g.memory_used_gb / g.memory_total_gb * 100.0
                } else {
                    0.0
                };
                let mut parts = vec![format!(
                    "GPU{} {}: {:.1}/{:.1}GB ({:.0}%)",
                    g.index, g.name, g.memory_used_gb, g.memory_total_gb, pct
                )];
                if let Some(util) = g.utilization {
                    parts.push(format!("{:.0}%", util * 100.0));
                }
                if let Some(temp) = g.temp {
                    parts.push(format!("{}C", temp as i32));
                }
                if let (Some(power), Some(limit)) = (g.power, g.power_limit) {
                    parts.push(format!("{:.0}/{:.0}W", power, limit));
                }
                if let Some(clock) = g.graphics_clock {
                    parts.push(format!("{}MHz", clock));
                }
                gpu_parts.push(parts.join(" "));
            }
            format!(" | {}", gpu_parts.join(" | "))
        };

        info!(
            "Stats | CPU: {:.1}% | RAM: {:.1}/{:.1}GB ({:.0}%){} | Models: {} ({:.2}GB)",
            stats.cpu_usage,
            stats.memory_used_gb,
            stats.memory_total_gb,
            stats.memory_percent,
            gpu_info,
            stats.loaded_models,
            stats.total_model_size_gb
        );
    }

    /// Start periodic stats logging
    pub fn start_periodic_logging(
        &self,
        interval: Duration,
        get_model_info: Arc<dyn Fn() -> (usize, u64) + Send + Sync>,
    ) -> tokio::task::JoinHandle<()> {
        // Clone the monitor to use in the spawned task
        let monitor = Arc::new(Self {
            system: self.system.clone(),
            gpu_manager: self.gpu_manager.clone(),
        });

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                let (model_count, model_size) = get_model_info();
                monitor.log_stats(model_count, model_size).await;
            }
        })
    }
}

impl Default for StatsMonitor {
    fn default() -> Self {
        Self::new(None)
    }
}
