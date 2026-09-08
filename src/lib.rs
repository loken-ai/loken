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

// Tensor shapes are written `[b, h, d]` throughout the documentation, which rustdoc would
// read as links; the crate's own items are documented from wherever they are reachable.
#![allow(
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
// The host build compiles the shapes of the device path, so what the device path alone
// reads is left in place rather than gated site by site. The default build denies all of it.
#![cfg_attr(
    not(feature = "cuda"),
    allow(
        unused_variables,
        unused_imports,
        unused_mut,
        unused_assignments,
        unused_macros,
        dead_code,
        unreachable_code
    )
)]
// Shapes this code takes on purpose, so the lint gate can deny everything else: kernel
// launchers and forwards take every dimension as an argument, hot loops index several
// arrays by one position, enums carry a resident tensor beside a small variant, a
// clamp on a bound that may be NaN must not panic, reference constants are written at
// their published precision, and a test spells its formula out in full.
#![allow(
    // Named by a clippy this tree is not always built with; the older one must not complain
    // about the name, and `as_chunks` changes nothing the chunked loops rely on.
    unknown_lints,
    clippy::chunks_exact_to_as_chunks,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_range_contains,
    clippy::items_after_test_module,
    clippy::field_reassign_with_default,
    clippy::assertions_on_constants,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::large_enum_variant,
    clippy::result_large_err,
    clippy::needless_range_loop,
    clippy::explicit_counter_loop,
    clippy::manual_clamp,
    clippy::manual_checked_ops,
    clippy::module_inception,
    clippy::new_ret_no_self,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

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
