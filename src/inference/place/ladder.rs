//! The one recovery ladder: every component the server loads is placed by the planner and,
//! when a placement runs out of card memory, placed again on the next rung.
//!
//! The rungs, in order:
//!  1. the plan the demand asks for: whole on the fastest card that holds it, else the blocks
//!     spread over the cards with the head on the last one, the rest on the host;
//!  2. an even split over every card, when the first plan left one out or packed one too full;
//!  3. the cards at a shrinking share of their free memory, the host taking the rest;
//!  4. the host alone.
//!
//! An out-of-memory failure moves to the next rung. Any other failure goes straight to the
//! host rung, and if that fails too the first error is returned. A cancelled load stops.

use super::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::Device;

/// Shares of each card's free memory the host-assisted rung offers, largest first.
const HOST_ASSISTED_SHARES_PCT: [u64; 4] = [70, 50, 35, 20];

/// What a component asks of the cards.
#[derive(Debug, Clone, Copy, Default)]
pub struct Demand {
    /// Repeating blocks the plan divides.
    pub layers: usize,
    /// Bytes of all the blocks together.
    pub block_bytes: u64,
    /// Bytes of the output head, which sits on the card of the last block.
    pub head_bytes: u64,
    /// Cache bytes each block adds on its device.
    pub kv_bytes_per_layer: u64,
    /// Bytes each card keeps free for the work a request does.
    pub runtime_reserve: u64,
}

impl Demand {
    /// Bytes one card needs to hold the whole component.
    pub fn whole(&self) -> u64 {
        self.block_bytes
            .saturating_add(self.head_bytes)
            .saturating_add(self.kv_bytes_per_layer.saturating_mul(self.layers as u64))
            .saturating_add(self.runtime_reserve)
    }

    /// The first rung: whole on the first card that holds it, else spread with the head.
    pub fn plan(&self, cards: &[(usize, u64)]) -> HeteroPlan {
        match cards.iter().find(|(_, free)| *free >= self.whole()) {
            Some(&(idx, _)) => HeteroPlan::forced_gpu(self.layers, self.layers, idx),
            None => HeteroPlan::calculate_holding_head(
                self.layers,
                self.block_bytes,
                self.head_bytes,
                cards,
                self.kv_bytes_per_layer,
                self.runtime_reserve,
            ),
        }
    }
}

/// Whether an error is the card running out of memory, whatever layer raised it.
pub fn is_out_of_memory<E: std::fmt::Display + ?Sized>(err: &E) -> bool {
    let s = err.to_string().to_ascii_lowercase();
    s.contains("out of memory")
        || s.contains("out_of_memory")
        || s.contains("[oom]")
        // cuBLAS reports its workspace and handle allocations with its own status.
        || s.contains("alloc_failed")
        || s.contains("alloc failed")
}

fn cards_used(plan: &HeteroPlan) -> usize {
    plan.segments
        .iter()
        .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
        .count()
}

/// Every rung after the first, in the order they are tried.
fn later_rungs(demand: &Demand, cards: &[(usize, u64)]) -> Vec<(String, HeteroPlan)> {
    let mut rungs = Vec::new();
    if cards.len() > 1 {
        rungs.push((
            format!("an even split over {} cards", cards.len()),
            HeteroPlan::split_across_cuda(demand.layers, cards),
        ));
    }
    for pct in HOST_ASSISTED_SHARES_PCT {
        let offered: Vec<(usize, u64)> = cards
            .iter()
            .map(|&(idx, free)| (idx, free.saturating_mul(pct) / 100))
            .filter(|(_, free)| *free > 0)
            .collect();
        if offered.is_empty() {
            break;
        }
        let plan = HeteroPlan::calculate_with_kv_reserve(
            demand.layers,
            demand.block_bytes,
            &offered,
            &[],
            1.0,
            demand.kv_bytes_per_layer,
            demand.runtime_reserve,
        );
        let on_host = plan
            .segments
            .iter()
            .any(|s| matches!(s.kind, DeviceKind::Cpu));
        if cards_used(&plan) > 0 && on_host {
            rungs.push((format!("the cards at {pct}% with the host"), plan));
        }
    }
    rungs.push((
        "the host".to_string(),
        HeteroPlan::calculate(demand.layers, demand.block_bytes, &[], &[], 1.0),
    ));
    rungs
}

/// Build `what` on the plan its demand asks for, and on the next rung each time a build
/// runs out of card memory. `cards` are ranked fastest first with the bytes each offers.
pub fn load_placed<T, E: std::fmt::Display>(
    what: &str,
    demand: &Demand,
    cards: &[(usize, u64)],
    cancelled: impl Fn() -> bool,
    mut build: impl FnMut(&HeteroPlan) -> Result<T, E>,
) -> Result<T, E> {
    let first = match build(&demand.plan(cards)) {
        Ok(m) => return Ok(m),
        Err(e) => e,
    };
    if cancelled() {
        return Err(first);
    }
    let mut rungs = later_rungs(demand, cards);
    if !is_out_of_memory(&first) {
        tracing::warn!("{what}: placement failed ({first}); trying the host");
        rungs.retain(|(name, _)| name == "the host");
    }
    let mut last_oom = is_out_of_memory(&first);
    for (name, plan) in rungs {
        if !last_oom && name != "the host" {
            continue;
        }
        release_pools();
        tracing::warn!("{what}: retrying on {name}");
        match build(&plan) {
            Ok(m) => {
                tracing::info!("{what}: placed on {name}");
                return Ok(m);
            }
            Err(e) => {
                if cancelled() {
                    return Err(e);
                }
                tracing::warn!("{what}: {name} failed ({e})");
                last_oom = is_out_of_memory(&e);
            }
        }
    }
    Err(first)
}

/// Build a component of `layers` blocks and `bytes` resident bytes, with nothing beside the
/// blocks, through the ladder. A cancelled load stops.
pub fn load_blocks<T>(
    layers: usize,
    bytes: u64,
    cards: &[(usize, u64)],
    what: &str,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    build: impl Fn(&HeteroPlan) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let demand = Demand {
        layers,
        block_bytes: bytes,
        ..Demand::default()
    };
    load_placed(
        what,
        &demand,
        cards,
        || cancel.is_some_and(crate::inference::serve::cancel::CancelToken::is_cancelled),
        |plan| build(plan).map_err(|e| anyhow::anyhow!("{e:#}")),
    )
}

/// The cards a placement may use now, fastest first, with the bytes each offers under the
/// pressure the fleet is in.
pub fn offered_cards() -> Vec<(usize, u64)> {
    super::vram_manager::probe_under_pressure(0)
        .into_iter()
        .map(|(idx, free, _)| (idx, free))
        .collect()
}

/// Whether the load running on this thread was cancelled.
fn scoped_cancelled() -> bool {
    crate::inference::serve::cancel::scoped::bail().is_err()
}

/// The device a plan's segment runs on; a card that does not open is the host.
pub fn device_of(kind: DeviceKind) -> Device {
    match kind {
        DeviceKind::Cuda(idx) => Device::new_cuda(idx).unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    }
}

/// The device each of `layers` blocks runs on under `plan`.
pub fn layer_devices(plan: &HeteroPlan, layers: usize) -> Vec<Device> {
    (0..layers)
        .map(|l| {
            plan.segments
                .iter()
                .find(|s| l >= s.layer_start && l < s.layer_end)
                .map(|s| device_of(s.kind))
                .unwrap_or(Device::Cpu)
        })
        .collect()
}

/// The device for what sits beside the blocks: the first card the plan uses, else the host.
pub fn primary_device(plan: &HeteroPlan) -> Device {
    plan.segments
        .iter()
        .find(|s| matches!(s.kind, DeviceKind::Cuda(_)))
        .map(|s| device_of(s.kind))
        .unwrap_or(Device::Cpu)
}

/// Whether a plan runs on one card with nothing on the host.
pub fn on_one_card(plan: &HeteroPlan) -> bool {
    let on_host = plan
        .segments
        .iter()
        .any(|s| matches!(s.kind, DeviceKind::Cpu));
    cards_used(plan) == 1 && !on_host
}

/// Build `what` through the ladder on the cards as they are now. A cancelled load stops.
pub fn load<T, E: std::fmt::Display>(
    what: &str,
    demand: &Demand,
    build: impl FnMut(&HeteroPlan) -> Result<T, E>,
) -> Result<T, E> {
    load_placed(what, demand, &offered_cards(), scoped_cancelled, build)
}

/// The demand of a component that runs whole on one device: `bytes` resident and
/// `reserve` kept free beside them for its work.
fn whole_demand(bytes: u64, reserve: u64) -> Demand {
    Demand {
        layers: 1,
        block_bytes: bytes,
        runtime_reserve: reserve,
        ..Demand::default()
    }
}

/// Build a component that runs whole on one device through the ladder: the fastest card
/// that holds it, else the host.
pub fn load_whole<T, E: std::fmt::Display>(
    what: &str,
    bytes: u64,
    reserve: u64,
    mut build: impl FnMut(&Device) -> Result<T, E>,
) -> Result<T, E> {
    load(what, &whole_demand(bytes, reserve), |plan| {
        build(&primary_device(plan))
    })
}

/// [`load_whole`] for a component that runs once per request: with more than one card it
/// takes the slowest card that holds it, leaving the fastest to what runs every step.
pub fn load_aside<T, E: std::fmt::Display>(
    what: &str,
    bytes: u64,
    reserve: u64,
    mut build: impl FnMut(&Device) -> Result<T, E>,
) -> Result<T, E> {
    let mut cards = offered_cards();
    cards.reverse();
    load_placed(
        what,
        &whole_demand(bytes, reserve),
        &cards,
        scoped_cancelled,
        |plan| build(&primary_device(plan)),
    )
}

/// Run one unit of a render; when it runs out of card memory, raise the pressure level so
/// its next placement leaves each card more room, and run it again. The unit reloads what
/// it places so the new level takes effect. Stops once the level that places on the host
/// has been tried, and at the first failure that is not out of memory.
pub fn rerun_under_pressure<T, E: std::fmt::Display>(
    what: &str,
    mut unit: impl FnMut() -> Result<T, E>,
) -> Result<T, String> {
    loop {
        match unit() {
            Ok(v) => return Ok(v),
            Err(e) if is_out_of_memory(&e) && !super::vram_manager::vram_force_cpu() => {
                release_pools();
                let level = super::vram_manager::vram_degrade();
                tracing::warn!("{what}: out of card memory; retrying at pressure level {level}");
            }
            Err(e) => return Err(format!("{what}: {e}")),
        }
    }
}

/// Run `work` at `size`, halving the size each time a card runs out of memory while it
/// stays above `floor`. For work whose transient memory follows a size it chooses, such
/// as a decode chunk or a capture arena.
pub fn halve_until_fits<T, E: std::fmt::Display>(
    what: &str,
    size: usize,
    floor: usize,
    mut work: impl FnMut(usize) -> Result<T, E>,
) -> Result<T, E> {
    let mut size = size.max(floor);
    loop {
        match work(size) {
            Err(e) if is_out_of_memory(&e) && size / 2 >= floor && size > floor => {
                size /= 2;
                release_pools();
                tracing::warn!("{what}: out of card memory; retrying at {size}");
            }
            done => return done,
        }
    }
}

/// Hands the pools' cached blocks back before the next rung measures the cards.
fn release_pools() {
    #[cfg(feature = "cuda")]
    crate::inference::engine::llm_engine::trim_cuda_pools();
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn demand(layers: usize, block_gb: u64) -> Demand {
        Demand {
            layers,
            block_bytes: block_gb * GB,
            ..Demand::default()
        }
    }

    fn kinds(plan: &HeteroPlan) -> Vec<String> {
        plan.segments
            .iter()
            .map(|s| format!("{}", s.kind))
            .collect()
    }

    #[test]
    fn a_component_that_fits_one_card_is_built_once_on_it() {
        let mut tries = 0;
        let got: Result<Vec<String>, String> = load_placed(
            "t",
            &demand(10, 4),
            &[(3, 8 * GB), (5, 8 * GB)],
            || false,
            |plan| {
                tries += 1;
                Ok(kinds(plan))
            },
        );
        assert_eq!(tries, 1);
        assert_eq!(got.unwrap(), vec!["CUDA(3)".to_string()]);
    }

    #[test]
    fn out_of_memory_walks_down_to_the_host() {
        let mut seen = Vec::new();
        let got: Result<(), String> = load_placed(
            "t",
            &demand(10, 4),
            &[(0, 8 * GB), (1, 8 * GB)],
            || false,
            |plan| {
                seen.push(kinds(plan));
                if plan
                    .segments
                    .iter()
                    .all(|s| matches!(s.kind, DeviceKind::Cpu))
                {
                    Ok(())
                } else {
                    Err("cuda upload: CUDA_ERROR_OUT_OF_MEMORY".to_string())
                }
            },
        );
        assert!(got.is_ok());
        assert!(seen.len() > 2, "{seen:?}");
        assert!(seen.last().unwrap().iter().all(|k| k == "CPU"), "{seen:?}");
    }

    #[test]
    fn another_failure_goes_straight_to_the_host() {
        let mut seen = 0;
        let got: Result<(), String> = load_placed(
            "t",
            &demand(10, 4),
            &[(0, 8 * GB), (1, 8 * GB)],
            || false,
            |plan| {
                seen += 1;
                if plan
                    .segments
                    .iter()
                    .all(|s| matches!(s.kind, DeviceKind::Cpu))
                {
                    Err("missing tensor".to_string())
                } else {
                    Err("unsupported kernel".to_string())
                }
            },
        );
        assert_eq!(seen, 2);
        assert_eq!(got.unwrap_err(), "unsupported kernel");
    }

    #[test]
    fn a_cancelled_load_stops_at_once() {
        let mut seen = 0;
        let got: Result<(), String> = load_placed(
            "t",
            &demand(10, 4),
            &[(0, 8 * GB)],
            || true,
            |_| {
                seen += 1;
                Err("out of memory".to_string())
            },
        );
        assert_eq!(seen, 1);
        assert!(got.is_err());
    }

    #[test]
    fn the_head_counts_toward_fitting_one_card() {
        let d = Demand {
            layers: 4,
            block_bytes: 6 * GB,
            head_bytes: 3 * GB,
            ..Demand::default()
        };
        assert_eq!(d.whole(), 9 * GB);
        assert_eq!(kinds(&d.plan(&[(0, 10 * GB)])), vec!["CUDA(0)".to_string()]);
        // Blocks that fit a card alone do not make it the home of the whole component.
        assert!(d.whole() > 8 * GB && d.block_bytes < 8 * GB);
    }

    #[test]
    fn a_size_halves_after_each_exhaustion_down_to_its_floor() {
        let mut tried = Vec::new();
        let got: Result<usize, String> = halve_until_fits("t", 64, 16, |size| {
            tried.push(size);
            if size > 16 {
                Err("[oom] alloc".to_string())
            } else {
                Ok(size)
            }
        });
        assert_eq!(got.unwrap(), 16);
        assert_eq!(tried, vec![64, 32, 16]);
        let mut n = 0;
        let got: Result<(), String> = halve_until_fits("t", 64, 16, |_| {
            n += 1;
            Err("bad shape".to_string())
        });
        assert!(got.is_err());
        assert_eq!(n, 1);
    }

    #[test]
    fn a_component_asking_for_more_than_any_card_runs_on_the_host() {
        let d = whole_demand(8 * GB, GB);
        let plan = d.plan(&[(0, 4 * GB)]);
        assert!(matches!(primary_device(&plan), Device::Cpu));
        assert!(!on_one_card(&plan));
        assert!(layer_devices(&plan, 1)
            .iter()
            .all(|d| matches!(d, Device::Cpu)));
        assert_eq!(
            kinds(&d.plan(&[(0, 4 * GB), (1, 10 * GB)])),
            vec!["CUDA(1)".to_string()]
        );
    }
}
