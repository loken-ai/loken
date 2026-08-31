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
/// Nine of them are reachable only from their own tests: nothing in the serving path
/// constructs a `LinkMatrix`, asks `cut_plan` for a plan, or opens a `WireLink`. A test suite
/// makes a module look alive - `link_matrix` has four tests and its `derive_costs` is called
/// by two of them, so `cost()` returns `Some` there and `None` everywhere else.
///
/// They are kept rather than deleted: they are the groundwork of the cluster-routing work,
/// not abandoned code. What they must not do is look wired.
///
/// The table below states, for every module here, whether anything outside it reaches it, and
/// the gate fails in BOTH directions: one of these finally reached, or one that was reached
/// falling silent. The second direction is what makes the list a statement about today - a
/// module can be orphaned by a deletion somewhere else entirely, and nothing else would say
/// so.
#[cfg(test)]
mod wiring_gate {
    /// Every module here, one distinctive public name for it, and whether the serving path
    /// reaches it. Generic names (`plan`, `Tier`, `Segment`) collide with unrelated code and
    /// would report a module as wired when it is not.
    ///
    /// One table, not two: a module that appears in neither list is the case a pair of tables
    /// lets through.
    const MODULES: [(&str, &str, bool); 19] = [
        ("cluster", "distributed::cluster::", true),
        ("cluster_runtime", "cluster_runtime::", true),
        ("cut_plan", "CutPlan", false),
        ("device_manager", "DeviceManager", true),
        ("discovery", "discovery::bind_discovery", true),
        ("evidence", "RequestEvidence", false),
        ("kv_tiers", "TieredKv", false),
        ("layer_scheduler", "LayerScheduler", false),
        ("link_matrix", "LinkMatrix", false),
        ("membership", "membership::", true),
        ("network", "NetworkClient", false),
        ("protocol", "protocol::", true),
        ("rate_meter", "rate_meter::", true),
        ("replay", "ResumePoint", false),
        ("routing", "routing::", true),
        ("wire", "WireFrame", false),
        ("wire_link", "WireLink", false),
        ("wire_mux", "MuxLink", false),
        ("mod", "", true),
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

        // Every module in this directory is in the table, or the table is not about today.
        let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/distributed");
        let mut on_disk: Vec<String> = std::fs::read_dir(&here)
            .expect("read distributed/")
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                (p.extension().is_some_and(|x| x == "rs"))
                    .then(|| p.file_stem()?.to_str().map(str::to_string))
                    .flatten()
            })
            .collect();
        on_disk.sort();
        let mut listed: Vec<String> = MODULES.iter().map(|(m, _, _)| m.to_string()).collect();
        listed.sort();
        assert_eq!(on_disk, listed, "the table and the directory disagree");

        let mut wrong = Vec::new();
        for (module, marker, expected) in MODULES {
            if marker.is_empty() {
                continue;
            }
            let own = format!("distributed/{module}.rs");
            // This file is skipped too: the table names every module, so a gate that searched
            // it would find each one reaching itself through its own record.
            let reached = files.iter().any(|(p, t)| {
                !p.ends_with(&own) && !p.ends_with("distributed/mod.rs") && t.contains(marker)
            });
            if reached != expected {
                wrong.push(format!(
                    "{module}: recorded as {}, is {}",
                    if expected { "reached" } else { "unreached" },
                    if reached { "reached" } else { "unreached" }
                ));
            }
        }

        assert!(
            wrong.is_empty(),
            "the record no longer describes the code. A module that became reachable is good \
             news - flip its flag. One that fell silent was orphaned by a change elsewhere, \
             and is the case this direction exists for:\n{}",
            wrong.join("\n")
        );
    }
}
