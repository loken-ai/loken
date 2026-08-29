//! CPU topology detection and core affinity management
//!
//! This module provides CPU topology detection (P-cores vs E-cores on hybrid CPUs)
//! and thread affinity management for optimal performance.

pub mod topology;

pub use topology::{
    detect_cpu_topology, set_thread_affinity, simd_capabilities, CoreType, CpuTopology,
};
