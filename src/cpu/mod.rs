//! CPU topology detection and core affinity management
//!
//! This module provides CPU topology detection (P-cores vs E-cores on hybrid CPUs)
//! and thread affinity management for optimal performance.

pub mod level;
pub mod topology;

pub use level::{
    cpu_level, dispatch_avx2, dispatch_avx512, hardware_has_avx2,
    hardware_has_avx512, hardware_level, set_level_cap, supported_levels, CpuLevel,
};
pub use topology::{
    detect_cpu_topology, set_thread_affinity, simd_capabilities, CoreType, CpuTopology,
};
