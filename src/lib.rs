//! `loken` - OpenAI- and Ollama-compatible HTTP inference server.
//!
//! This library crate is consumed by the `server` binary (and several
//! `test_*` binaries) under `src/bin/`. The HTTP surface lives in [`api`];
//! model loading and per-architecture inference lives in [`inference`];
//! device + system telemetry in [`gpu`] / [`stats`] / [`cpu`]; multi-host
//! distributed work in [`distributed`].
//!
//! The default build pulls in CUDA + OpenCL via the corresponding cargo
//! features. The `cpu` feature carves out a CPU-only build.

// Public modules
pub mod api;
pub mod cli;
pub mod config;
pub mod cpu;
pub mod distributed;
// Per-request energy measurement (CPU RAPL + GPU NVML) - same core the bench uses,
// here wired into the request path for systematic energy reporting.
pub mod energy;
pub mod energy_report;
pub mod gpu;
pub mod inference;
mod notice_gate;
pub mod privacy;
pub mod stats;
// Tensor substrate (in-crate native implementation).
pub mod tensor;

// Re-export commonly used types
pub use inference::InferenceEngine;
