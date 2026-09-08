//! Deciding WHERE a component runs, and what that costs. The heterogeneous
//! plan, the VRAM budgets it is checked against, and the device probes under both.
//!
//! Callers name a module through this directory - `crate::inference::place::<module>` - so the path says which
//! part of the system a file belongs to, which is the whole reason the directory exists.

pub mod audio_demand;
pub mod device_probe;
pub mod dry_plan;
pub mod fit_report;
pub mod layer_executor;
pub mod layer_perf;
pub mod model_request;
#[cfg(all(test, feature = "image"))]
pub mod placement_invariants;
pub mod plan;
pub mod runtime_demand;
#[cfg(feature = "video")]
pub mod video_plan;
pub mod vram_manager;
