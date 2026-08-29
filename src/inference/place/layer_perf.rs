//! Layer Performance Tracking
//!
//! Tracks per-layer inference performance metrics for real-time monitoring.

use std::sync::Arc;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Performance metrics for a single layer
#[derive(Debug, Clone)]
pub struct LayerPerformance {
    /// Layer index
    pub layer_idx: usize,
    /// Device type (CPU or GPU)
    pub device_type: String,
    /// Model name this layer belongs to (for multi-model display)
    pub model_name: String,
    /// Total time spent in this layer across all tokens
    pub total_duration: Duration,
    /// Number of tokens processed
    pub token_count: usize,
    /// Average time per token in milliseconds
    pub avg_ms_per_token: f64,
    /// Tokens per second
    pub tokens_per_second: f64,
    /// Number of early exits triggered at this layer boundary
    pub early_exit_count: usize,
}

impl LayerPerformance {
    pub fn new(layer_idx: usize, device_type: String) -> Self {
        Self::with_model(layer_idx, device_type, String::new())
    }

    pub fn with_model(layer_idx: usize, device_type: String, model_name: String) -> Self {
        Self {
            layer_idx,
            device_type,
            model_name,
            total_duration: Duration::ZERO,
            token_count: 0,
            avg_ms_per_token: 0.0,
            tokens_per_second: 0.0,
            early_exit_count: 0,
        }
    }

    /// Record a new timing sample
    pub fn record(&mut self, duration: Duration) {
        self.total_duration += duration;
        self.token_count += 1;

        // Recalculate averages
        let total_ms = self.total_duration.as_secs_f64() * 1000.0;
        self.avg_ms_per_token = if self.token_count > 0 {
            total_ms / self.token_count as f64
        } else {
            0.0
        };

        self.tokens_per_second = if total_ms > 0.0 {
            (self.token_count as f64 * 1000.0) / total_ms
        } else {
            0.0
        };
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.total_duration = Duration::ZERO;
        self.token_count = 0;
        self.avg_ms_per_token = 0.0;
        self.tokens_per_second = 0.0;
        self.early_exit_count = 0;
    }
}

/// Global performance tracker for all layers
#[derive(Debug, Clone, Default)]
pub struct LayerPerformanceTracker {
    layers: Arc<RwLock<Vec<LayerPerformance>>>,
}

impl LayerPerformanceTracker {
    /// Create a new tracker
    pub fn new() -> Self {
        Self {
            layers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize tracker with layer count and device types
    pub fn initialize(&self, layer_count: usize, layers_on_gpu: usize) {
        self.initialize_with_device(layer_count, layers_on_gpu, 0);
    }

    /// Initialize tracker for a named model (replaces only that model's layers, preserves others)
    pub fn initialize_model(&self, model_name: &str, layer_count: usize, layers_on_gpu: usize) {
        self.initialize_model_with_device(model_name, layer_count, layers_on_gpu, 0);
    }

    /// Initialize tracker with layer count, device types, and GPU device index
    pub fn initialize_with_device(
        &self,
        layer_count: usize,
        layers_on_gpu: usize,
        gpu_index: usize,
    ) {
        let mut layers = self.layers.write().unwrap();
        layers.clear();

        for i in 0..layer_count {
            let device_type = if i < layers_on_gpu {
                format!("GPU #{}", gpu_index)
            } else {
                "CPU".to_string()
            };
            layers.push(LayerPerformance::new(i, device_type));
        }
    }

    /// Initialize tracker for a named model with device index
    pub fn initialize_model_with_device(
        &self,
        model_name: &str,
        layer_count: usize,
        layers_on_gpu: usize,
        gpu_index: usize,
    ) {
        let mut layers = self.layers.write().unwrap();
        // Remove old entries for this model
        layers.retain(|l| l.model_name != model_name);

        for i in 0..layer_count {
            let device_type = if i < layers_on_gpu {
                format!("GPU #{}", gpu_index)
            } else {
                "CPU".to_string()
            };
            layers.push(LayerPerformance::with_model(
                i,
                device_type,
                model_name.to_string(),
            ));
        }
    }

    /// Initialize tracker with heterogeneous segments for a named model
    #[cfg(feature = "opencl")]
    pub fn initialize_heterogeneous_model(
        &self,
        model_name: &str,
        segments: &[crate::inference::place::layer_executor::HeteroSegment],
    ) {
        let mut layers = self.layers.write().unwrap();
        // Remove old entries for this model
        layers.retain(|l| l.model_name != model_name);

        for seg in segments {
            let device_label = match seg.kind {
                crate::inference::place::layer_executor::DeviceKind::Cuda(idx) => {
                    format!("CUDA #{}", idx)
                }
                crate::inference::place::layer_executor::DeviceKind::OpenCL(idx) => {
                    format!("Arc #{}", idx)
                }
                crate::inference::place::layer_executor::DeviceKind::Cpu => "CPU".to_string(),
            };

            for layer_idx in seg.layer_start..seg.layer_end {
                layers.push(LayerPerformance::with_model(
                    layer_idx,
                    device_label.clone(),
                    model_name.to_string(),
                ));
            }
        }
    }

    /// Record timing for a specific layer (unnamed model - matches first with layer_idx)
    pub fn record_layer(&self, layer_idx: usize, duration: Duration) {
        self.record_layer_for("", layer_idx, duration);
    }

    /// Record timing for a specific layer of a named model
    pub fn record_layer_for(&self, model_name: &str, layer_idx: usize, duration: Duration) {
        if let Ok(mut layers) = self.layers.write() {
            if let Some(layer) = layers
                .iter_mut()
                .find(|l| l.model_name == model_name && l.layer_idx == layer_idx)
            {
                layer.record(duration);
            }
        }
    }

    /// Get all layer performance metrics (sorted by tokens/sec descending)
    pub fn get_metrics_sorted(&self) -> Vec<LayerPerformance> {
        if let Ok(layers) = self.layers.read() {
            let mut metrics = layers.clone();
            // Sort by tokens per second, descending (fastest first)
            metrics.sort_by(|a, b| {
                b.tokens_per_second
                    .partial_cmp(&a.tokens_per_second)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            metrics
        } else {
            Vec::new()
        }
    }

    /// Get layer performance metrics in original order
    pub fn get_metrics(&self) -> Vec<LayerPerformance> {
        if let Ok(layers) = self.layers.read() {
            layers.clone()
        } else {
            Vec::new()
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        if let Ok(mut layers) = self.layers.write() {
            for layer in layers.iter_mut() {
                layer.reset();
            }
        }
    }
}

/// Global static tracker instance
static GLOBAL_TRACKER: once_cell::sync::Lazy<LayerPerformanceTracker> =
    once_cell::sync::Lazy::new(LayerPerformanceTracker::new);

/// Get the global performance tracker
/// Where a layer's time goes, in the stages a decode step is actually made of.
///
/// Separate from the per-layer tracker because the question it answers is different: the
/// layer view says WHICH layer is slow, this one says WHAT inside a layer is slow. A cost
/// spread evenly over every layer is invisible to the first and obvious to the second.
///
/// Switched at runtime rather than read from the environment: the gate sits on the decode
/// path, evaluated once per layer per token, and a lookup there is paid by every model
/// whether or not anyone is profiling.
pub mod stages {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

    /// `experts` is the whole expert block; `gate_up` and `down` are the two matrix
    /// products inside it, so they sum to it rather than adding to the total.
    pub const NAMES: [&str; 10] = [
        "attn_norm",
        "attn",
        "attn_res",
        "router",
        "experts",
        "ffn_res",
        "gate_up",
        "down",
        "expert_call",
        "topk",
    ];
    /// The stages that partition a layer-call. The two inside `experts` are excluded, or
    /// the shares would add up to more than the whole.
    pub const PARTITION: usize = 6;
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static SUMS: [AtomicU64; 10] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    static CALLS: AtomicU64 = AtomicU64::new(0);

    /// One relaxed load. Cheap enough to sit on the decode path.
    #[inline]
    pub fn enabled() -> bool {
        ENABLED.load(Relaxed)
    }

    pub fn set_enabled(on: bool) {
        ENABLED.store(on, Relaxed);
    }

    /// Zero the sums, so a profile describes one model rather than everything the
    /// process has run since it started.
    pub fn reset() {
        for s in &SUMS {
            s.store(0, Relaxed);
        }
        CALLS.store(0, Relaxed);
        TOPK_FAST.store(0, Relaxed);
        TOPK_SLOW.store(0, Relaxed);
        HOST_FAST.store(0, Relaxed);
        for c in &HOST_REJECT {
            c.store(0, Relaxed);
        }
    }

    #[inline]
    pub fn add(stage: usize, micros: u64) {
        SUMS[stage].fetch_add(micros, Relaxed);
    }

    #[inline]
    pub fn count_call() {
        CALLS.fetch_add(1, Relaxed);
    }

    static TOPK_FAST: AtomicU64 = AtomicU64::new(0);
    static TOPK_SLOW: AtomicU64 = AtomicU64::new(0);

    /// Which top-k selection ran. A kernel whose precondition holds and which is still
    /// not taken is the difference between reading a guard and knowing it fired.
    #[inline]
    pub fn note_topk_path(fast: bool) {
        if fast {
            TOPK_FAST.fetch_add(1, Relaxed);
        } else {
            TOPK_SLOW.fetch_add(1, Relaxed);
        }
    }

    static HOST_FAST: AtomicU64 = AtomicU64::new(0);
    static HOST_REJECT: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    pub const HOST_REJECT_NAMES: [&str; 3] = ["device", "shape", "dtype"];

    /// Why a layer did not take the zero-allocation host executor.
    ///
    /// The guard has three clauses and a model that misses it misses all of them equally
    /// from outside. Counting the cause is the difference between knowing and guessing.
    #[inline]
    pub fn note_host_path(reject: Option<usize>) {
        match reject {
            None => {
                HOST_FAST.fetch_add(1, Relaxed);
            }
            Some(i) => {
                HOST_REJECT[i].fetch_add(1, Relaxed);
            }
        }
    }

    pub fn host_paths() -> (u64, Vec<(&'static str, u64)>) {
        (
            HOST_FAST.load(Relaxed),
            HOST_REJECT_NAMES
                .iter()
                .copied()
                .zip(HOST_REJECT.iter().map(|c| c.load(Relaxed)))
                .collect(),
        )
    }

    pub fn topk_paths() -> (u64, u64) {
        (TOPK_FAST.load(Relaxed), TOPK_SLOW.load(Relaxed))
    }

    /// Microseconds the router spent on the layer-call being closed, and reset.
    ///
    /// The mixture times its own routing because that is where the boundary is; the
    /// caller subtracts it from the block total to get the expert compute. Thread-local
    /// rather than shared: two requests must not read each other's routing.
    pub fn take_router_us() -> u64 {
        ROUTER_US.with(|c| c.replace(0))
    }

    /// Record the routing time of the layer-call in progress.
    pub fn note_router_us(micros: u64) {
        ROUTER_US.with(|c| c.set(c.get() + micros));
    }

    thread_local! {
        static ROUTER_US: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// (stage name, total microseconds), plus the number of layer-calls they cover.
    pub fn snapshot() -> (Vec<(&'static str, u64)>, u64) {
        (
            NAMES
                .iter()
                .copied()
                .zip(SUMS.iter().map(|s| s.load(Relaxed)))
                .collect(),
            CALLS.load(Relaxed),
        )
    }
}

pub fn global_tracker() -> &'static LayerPerformanceTracker {
    &GLOBAL_TRACKER
}

/// RAII helper for timing a layer.
///
/// Timing sits inside the per-layer loop, so it runs once per layer per token. Reading
/// the clock twice and then taking a write lock to find the layer by name costs more
/// than the smallest layers themselves, which turns the profiler into part of what it
/// measures. `start` is therefore `None` unless profiling was asked for - the same
/// switch every other probe on this path already consults.
pub struct LayerTimer {
    layer_idx: usize,
    model_name: &'static str,
    start: Option<Instant>,
}

impl LayerTimer {
    /// Start timing a layer (unnamed model - backwards compatible)
    pub fn start(layer_idx: usize) -> Self {
        Self::start_for(layer_idx, "")
    }

    /// Start timing a layer for a specific model
    pub fn start_for(layer_idx: usize, model_name: &'static str) -> Self {
        Self {
            layer_idx,
            model_name,
            start: stages::enabled().then(Instant::now),
        }
    }
}

impl Drop for LayerTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            global_tracker().record_layer_for(self.model_name, self.layer_idx, start.elapsed());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct layers must keep distinct totals. The endpoint was answering the same
    /// figure for every layer of a model, which reads as a measurement and is not one:
    /// this pins whether the tracker or its caller is responsible.
    #[test]
    fn each_layer_keeps_its_own_total() {
        let t = LayerPerformanceTracker::new();
        t.initialize_with_device(4, 4, 0);
        for (i, ms) in [(0usize, 1u64), (1, 2), (2, 4), (3, 8)] {
            t.record_layer(i, Duration::from_millis(ms));
        }
        let m = t.get_metrics();
        let totals: Vec<u128> = m.iter().map(|l| l.total_duration.as_millis()).collect();
        assert_eq!(
            totals,
            vec![1, 2, 4, 8],
            "the tracker collapsed distinct layers: {totals:?}"
        );
        assert_eq!(
            m.iter().map(|l| l.token_count).collect::<Vec<_>>(),
            vec![1, 1, 1, 1]
        );
    }

    #[test]
    fn test_distribute_layer_timing() {
        // Create a fresh tracker (don't use global to avoid test interference)
        let tracker = LayerPerformanceTracker::new();
        let num_layers = 40;
        let gpu_layers = 22;

        // Initialize with 40 layers, 22 GPU
        tracker.initialize_with_device(num_layers, gpu_layers, 0);

        // Simulate 9 forward passes recorded to layer 0
        for _ in 0..9 {
            tracker.record_layer(0, Duration::from_millis(300));
        }

        // Verify layer 0 accumulated correctly
        let metrics = tracker.get_metrics();
        let l0 = metrics.iter().find(|l| l.layer_idx == 0).unwrap();
        assert_eq!(l0.token_count, 9);
        assert!(l0.total_duration >= Duration::from_millis(2700));
        assert!(l0.tokens_per_second > 0.0);

        // Verify layer 1 has no data
        let l1 = metrics.iter().find(|l| l.layer_idx == 1).unwrap();
        assert_eq!(l1.token_count, 0);

        // Now simulate distribute_layer_timing logic
        let layers = tracker.get_metrics();
        let layer_0 = layers.iter().find(|l| l.layer_idx == 0).unwrap();
        let total_duration = layer_0.total_duration;
        let token_count = layer_0.token_count;
        let per_layer_duration = total_duration / num_layers as u32;

        tracker.reset();
        for layer_idx in 0..num_layers {
            for _ in 0..token_count {
                tracker.record_layer(layer_idx, per_layer_duration);
            }
        }

        // Verify ALL layers now have data
        let metrics = tracker.get_metrics();
        for layer_idx in 0..num_layers {
            let l = metrics.iter().find(|l| l.layer_idx == layer_idx).unwrap();
            assert_eq!(l.token_count, 9, "Layer {} should have 9 tokens", layer_idx);
            assert!(
                l.tokens_per_second > 0.0,
                "Layer {} should have >0 tok/s",
                layer_idx
            );
        }

        // Verify device types preserved after reset
        let l0 = metrics.iter().find(|l| l.layer_idx == 0).unwrap();
        assert_eq!(l0.device_type, "GPU #0");
        let l21 = metrics.iter().find(|l| l.layer_idx == 21).unwrap();
        assert_eq!(l21.device_type, "GPU #0");
        let l22 = metrics.iter().find(|l| l.layer_idx == 22).unwrap();
        assert_eq!(l22.device_type, "CPU");
        let l39 = metrics.iter().find(|l| l.layer_idx == 39).unwrap();
        assert_eq!(l39.device_type, "CPU");

        // Verify sorted returns all 40 layers
        let sorted = tracker.get_metrics_sorted();
        assert_eq!(sorted.len(), 40);
        // All should have same tok/s since distributed evenly
        let first_tps = sorted[0].tokens_per_second;
        let last_tps = sorted[39].tokens_per_second;
        assert!(
            (first_tps - last_tps).abs() < 0.01,
            "All layers should have equal tok/s but got first={:.2} last={:.2}",
            first_tps,
            last_tps
        );
    }
}
