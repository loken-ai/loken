//! Distributed inference module
//!
//! This module provides network-distributed LLM inference across multiple servers
//! and heterogeneous devices (NVIDIA GPU, Intel Arc, CPU).

pub mod cluster;
pub mod cluster_runtime;
pub mod cut_plan;
pub mod device_manager;
pub mod discovery;
pub mod evidence;
pub mod kv_tiers;
pub mod layer_scheduler;
pub mod link_matrix;
pub mod membership;
pub mod network;
pub mod protocol;
pub mod rate_meter;
pub mod replay;
pub mod routing;
pub mod wire;
pub mod wire_link;
pub mod wire_mux;

pub use device_manager::{
    log_distribution_plan, log_loaded_models, ComputeDevice, DeviceAvailability,
    DeviceDistribution, DeviceManager, DeviceType, HardwareTopology, LoadedModelInfo, RemoteServer,
    RemoteServerStatus, TopologyEventType, UnavailableReason,
};
pub use layer_scheduler::{LayerAssignment, LayerScheduler, ModelInfo, ModelShard};
pub use network::{NetworkClient, NetworkServer, ServerConfig};
pub use protocol::{DeviceInfo, LayerRequest, LayerResponse, ServerRegistration, TensorMessage};

/// Which of these modules anything outside itself actually reaches.
///
/// Seven of them are reachable only from their own tests: nothing in the serving path
/// constructs a `LinkMatrix`, asks `cut_plan` for a plan, or opens a `WireLink`. A test suite
/// makes a module look alive - `link_matrix` has four tests and its `derive_costs` is called
/// by two of them, so `cost()` returns `Some` there and `None` everywhere else.
///
/// They are kept rather than deleted: they are the groundwork of the cluster-routing work,
/// not abandoned code. What they must not do is look wired. This gate names them, and fails
/// in BOTH directions - a new module that nothing reaches, or one of these finally reached  - 
/// so the list stays a statement about today rather than a comment that was true once.
#[cfg(test)]
mod wiring_gate {
    /// One distinctive public name per module: generic ones (`plan`, `Tier`, `Segment`)
    /// collide with unrelated code and would report a module as wired when it is not.
    const REACHED_BY: [(&str, &str); 7] = [
        ("cut_plan", "CutPlan"),
        ("evidence", "RequestEvidence"),
        ("kv_tiers", "TieredKv"),
        ("link_matrix", "LinkMatrix"),
        ("replay", "ResumePoint"),
        ("wire_link", "WireLink"),
        ("wire_mux", "MuxLink"),
    ];

    fn sources(dir: &std::path::Path, out: &mut Vec<(std::path::PathBuf, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                sources(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    // Comment lines dropped: a module's header may DISCUSS another
                    // ("`WireLink` serialises, this one does not") without reaching it,
                    // and a mention is not a call.
                    let code: String = text
                        .lines()
                        .filter(|l| !l.trim_start().starts_with("//"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    out.push((path, code));
                }
            }
        }
    }

    #[test]
    fn the_modules_nothing_reaches_are_the_ones_recorded() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        sources(&root, &mut files);
        assert!(
            files.len() > 100,
            "found only {} sources - this walk is looking in the wrong place",
            files.len()
        );

        let mut wired = Vec::new();
        for (module, marker) in REACHED_BY {
            let own = format!("distributed/{module}.rs");
            // This file is skipped too: the table below names all seven, so a gate that
            // searched it would find every module reaching itself through its own record.
            if files.iter().any(|(p, t)| {
                !p.ends_with(&own) && !p.ends_with("distributed/mod.rs") && t.contains(marker)
            }) {
                wired.push(module);
            }
        }

        assert!(
            wired.is_empty(),
            "these are recorded as reachable only from their own tests, and something now \
             reaches them: {wired:?}. That is good news - take them off the list in this \
             module's header and out of REACHED_BY."
        );
    }
}
