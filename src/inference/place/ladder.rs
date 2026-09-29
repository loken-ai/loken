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
//! A rung that plans what an earlier rung already tried is skipped.
//!
//! Host-side accelerators, when a caller offers them, take the blocks a rung leaves on the
//! host before the host does.

use super::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};
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

    /// The first rung for a component whose stem, the embedders and the way back out, is
    /// built on the card its first block runs on, and whose work runs comfortably with
    /// `runtime_reserve` free and still runs with `floor` free: whole on the first card that
    /// holds it and the stem at the floor, else the blocks spread at the full reserve with
    /// the stem held on the card the spread starts on.
    pub fn plan_with_stem(
        &self,
        cards: &[(usize, u64)],
        stem_bytes: u64,
        floor: u64,
    ) -> HeteroPlan {
        let at_floor = Demand {
            runtime_reserve: floor.min(self.runtime_reserve),
            ..*self
        };
        let whole = at_floor.whole().saturating_add(stem_bytes);
        if let Some(&(idx, _)) = cards.iter().find(|(_, free)| *free >= whole) {
            return HeteroPlan::forced_gpu(self.layers, self.layers, idx);
        }
        let spread = |holder: Option<usize>| {
            let offered: Vec<(usize, u64)> = cards
                .iter()
                .map(|&(idx, free)| {
                    if Some(idx) == holder {
                        (idx, free.saturating_sub(stem_bytes))
                    } else {
                        (idx, free)
                    }
                })
                .collect();
            HeteroPlan::calculate_holding_head(
                self.layers,
                self.block_bytes,
                self.head_bytes,
                &offered,
                self.kv_bytes_per_layer,
                self.runtime_reserve,
            )
        };
        let mut holder = cards.first().map(|(idx, _)| *idx);
        let mut plan = spread(holder);
        for _ in 0..cards.len() {
            let starts_on = first_card(&plan);
            if starts_on.is_none() || starts_on == holder {
                break;
            }
            holder = starts_on;
            plan = spread(holder);
        }
        plan
    }
}

impl Demand {
    /// The first rung for a component whose stem of `stem_bytes` may sit on the card its
    /// first block runs on: planned with the stem there when that costs the blocks nothing,
    /// every block on the device it takes without the stem, else planned without it.
    /// Returns the plan and whether it holds the stem.
    pub fn plan_stem_if_free(&self, cards: &[(usize, u64)], stem_bytes: u64) -> (HeteroPlan, bool) {
        let without = self.plan(cards);
        let with = self.plan_with_stem(cards, stem_bytes, self.runtime_reserve);
        // A block the stem pushes to another card, a slower one or the host, costs every
        // step; the stem off the card costs a row lookup per token.
        let free = first_card(&with).is_some()
            && crate::inference::place::dry_plan::same_plan(&with, &without);
        if free {
            (with, true)
        } else {
            (without, false)
        }
    }
}

impl Demand {
    /// [`Demand::plan_with_stem`] for a component whose one-shot companion of `companion`
    /// bytes sits on a card the blocks leave alone, or on the host. Whole on one card
    /// when a card holds it; else the blocks are spread with each card in turn keeping
    /// the companion's room, and with none keeping it, and the spread with the fewest
    /// blocks on the host, then the fewest cards, then the most blocks on the fastest
    /// card wins.
    pub fn plan_with_stem_beside(
        &self,
        cards: &[(usize, u64)],
        stem_bytes: u64,
        floor: u64,
        companion: u64,
    ) -> HeteroPlan {
        let plain = self.plan_with_stem(cards, stem_bytes, floor);
        if plain.segments.len() == 1 && first_card(&plain).is_some() {
            return plain;
        }
        let fastest = cards.first().map(|(idx, _)| *idx);
        let rank = |plan: &HeteroPlan| {
            let host: usize = plan
                .segments
                .iter()
                .filter(|s| !matches!(s.kind, DeviceKind::Cuda(_)))
                .map(HeteroSegment::num_layers)
                .sum();
            let on_fastest: usize = plan
                .segments
                .iter()
                .filter(|s| Some(s.kind) == fastest.map(DeviceKind::Cuda))
                .map(HeteroSegment::num_layers)
                .sum();
            (host, cards_of(plan).len(), std::cmp::Reverse(on_fastest))
        };
        let mut best = plain;
        for &(keeper, _) in cards {
            let offered: Vec<(usize, u64)> = cards
                .iter()
                .map(|&(idx, free)| {
                    if idx == keeper {
                        (idx, free.saturating_sub(companion))
                    } else {
                        (idx, free)
                    }
                })
                .collect();
            let plan = self.plan_with_stem(&offered, stem_bytes, floor);
            if rank(&plan) < rank(&best) {
                best = plan;
            }
        }
        best
    }
}

/// The card a plan's first block runs on, when it runs on a card.
fn first_card(plan: &HeteroPlan) -> Option<usize> {
    plan.segments.first().and_then(|s| match s.kind {
        DeviceKind::Cuda(idx) => Some(idx),
        _ => None,
    })
}

/// The cards a component may be placed on now, fastest first, with the bytes each offers
/// once `reserve` is kept free on it and the pressure the fleet is under is taken off.
/// Empty when there is no card, or when the pressure has sent every load to the host.
pub fn cards(reserve: u64) -> Vec<(usize, u64)> {
    super::vram_manager::probe_under_pressure(reserve)
        .into_iter()
        .map(|(idx, free, _)| (idx, free))
        .collect()
}

/// [`cards`] as `owner` sees them while it runs: net of what other owners declared they
/// are about to allocate, and read without handing the pools back, so it may be asked
/// between the steps of a render.
pub fn cards_for(owner: &str, reserve: u64) -> Vec<(usize, u64)> {
    super::device_probe::probe_cuda_gpus_for(owner)
        .into_iter()
        .map(|g| (g.index, g.stable_free.saturating_sub(reserve)))
        .filter(|(_, free)| *free > 0)
        .collect()
}

/// The host-side accelerators, with the bytes each offers once `reserve` is kept free on
/// it. Empty in a build without them.
pub fn accelerators(reserve: u64) -> Vec<(usize, u64)> {
    #[cfg(feature = "opencl")]
    {
        crate::inference::kernel::opencl::enumerate_opencl_devices()
            .into_iter()
            .enumerate()
            .map(|(i, id)| {
                let mem = crate::inference::kernel::opencl::get_opencl_device_memory(id);
                (i, mem.saturating_sub(reserve))
            })
            .filter(|(_, free)| *free > 0)
            .collect()
    }
    #[cfg(not(feature = "opencl"))]
    {
        let _ = reserve;
        Vec::new()
    }
}

/// Free bytes on the card `device` runs on, with the pools handed back first; `None` for
/// the host or a card that does not report.
pub fn free_on_card(device: &Device) -> Option<u64> {
    let idx = super::plan::ordinal(device)?;
    super::vram_manager::probe(0)
        .into_iter()
        .find(|(i, _, _)| *i == idx)
        .map(|(_, free, _)| free)
}

/// The fastest card as a device, for a caller that only names where it runs.
pub fn fastest() -> Option<Device> {
    // PLACEMENT-EXEMPT: nothing is loaded onto this device by being named here; the
    // components a caller builds are placed through the ladder.
    super::vram_manager::probe(0)
        .into_iter()
        .next()
        .map(|(_, _, device)| device)
}

/// The cards for a one-shot component that must not compete with a hot one: the cards
/// `hot` does not use first, then the hot cards offering what is left once `keep` stays
/// free on them. A hot card with nothing left to offer is not listed.
pub fn aside(cards: &[(usize, u64)], hot: &[usize], keep: u64) -> Vec<(usize, u64)> {
    let cold = cards.iter().filter(|(idx, _)| !hot.contains(idx)).copied();
    let shared = cards
        .iter()
        .filter(|(idx, _)| hot.contains(idx))
        .map(|&(idx, free)| (idx, free.saturating_sub(keep)))
        .filter(|(_, free)| *free > 0);
    cold.chain(shared).collect()
}

/// `cards` with `card` offered first, when it is one of them.
pub fn prefer(cards: &[(usize, u64)], card: Option<usize>) -> Vec<(usize, u64)> {
    let mut ranked = cards.to_vec();
    if let Some(at) = card.and_then(|c| ranked.iter().position(|(idx, _)| *idx == c)) {
        let first = ranked.remove(at);
        ranked.insert(0, first);
    }
    ranked
}

/// The CUDA cards a plan runs on, in the plan's order.
pub fn cards_of(plan: &HeteroPlan) -> Vec<usize> {
    let mut out = Vec::new();
    for s in &plan.segments {
        if let DeviceKind::Cuda(idx) = s.kind {
            if !out.contains(&idx) {
                out.push(idx);
            }
        }
    }
    out
}

/// A tensor device for a plan's device: the card's shared stream-bound handle, or the host.
pub fn open(kind: DeviceKind) -> anyhow::Result<Device> {
    match kind {
        DeviceKind::Cpu => Ok(Device::Cpu),
        DeviceKind::Cuda(idx) => crate::tensor::cuda_ext::new_device_with_stream(idx)
            .map(|d| d.native_device())
            .map_err(|e| anyhow::anyhow!("CUDA:{idx}: {e}")),
        DeviceKind::OpenCL(_) => Err(anyhow::anyhow!("{kind} has no tensor device")),
    }
}

/// Pressure rungs a request walks before its placement reaches the host: one per share of the
/// cards the host-assisted rung offers.
pub const PRESSURE_RUNGS: u64 = HOST_ASSISTED_SHARES_PCT.len() as u64;

/// Whether an error is the card running out of memory, whatever layer raised it. The one
/// classifier: every caller that branches on memory pressure asks this.
pub fn is_out_of_memory<E: std::fmt::Display + ?Sized>(err: &E) -> bool {
    let s = err.to_string().to_ascii_lowercase();
    s.contains("out of memory")
        || s.contains("out_of_memory")
        // The substrate's `[oom]` marker, and the word on its own.
        || s.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| w == "oom")
        // cuBLAS and cuDNN report their workspace and handle allocations with a status of
        // their own.
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
fn later_rungs(
    demand: &Demand,
    cards: &[(usize, u64)],
    accelerators: &[(usize, u64)],
) -> Vec<(String, HeteroPlan)> {
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
        .into_iter()
        .map(|(name, plan)| (name, beside_host(plan, demand, accelerators)))
        .collect()
}

/// `plan` with the blocks it leaves on the host offered to `accelerators` first.
fn beside_host(plan: HeteroPlan, demand: &Demand, accelerators: &[(usize, u64)]) -> HeteroPlan {
    if accelerators.is_empty() || demand.layers == 0 {
        return plan;
    }
    let per_block = demand.block_bytes / demand.layers as u64;
    let mut segments = Vec::with_capacity(plan.segments.len());
    for seg in plan.segments {
        if !matches!(seg.kind, DeviceKind::Cpu) {
            segments.push(seg);
            continue;
        }
        let blocks = seg.num_layers();
        let host = HeteroPlan::calculate(
            blocks,
            per_block.saturating_mul(blocks as u64),
            &[],
            accelerators,
            1.0,
        );
        segments.extend(host.segments.into_iter().map(|s| HeteroSegment {
            layer_start: s.layer_start + seg.layer_start,
            layer_end: s.layer_end + seg.layer_start,
            ..s
        }));
    }
    HeteroPlan {
        segments,
        total_layers: plan.total_layers,
    }
}

/// Build `what` on the plan its demand asks for, and on the next rung each time a build
/// runs out of card memory. `cards` are ranked fastest first with the bytes each offers.
pub fn load_placed<T, E: std::fmt::Display>(
    what: &str,
    demand: &Demand,
    cards: &[(usize, u64)],
    cancelled: impl Fn() -> bool,
    build: impl FnMut(&HeteroPlan) -> Result<T, E>,
) -> Result<T, E> {
    load_planned(
        what,
        demand.plan(cards),
        demand,
        cards,
        &[],
        cancelled,
        build,
    )
}

/// [`load_placed`] from a first plan the planner made for this component, with
/// `accelerators` taking the blocks each rung leaves on the host before the host does.
pub fn load_planned<T, E: std::fmt::Display>(
    what: &str,
    first_plan: HeteroPlan,
    demand: &Demand,
    cards: &[(usize, u64)],
    accelerators: &[(usize, u64)],
    cancelled: impl Fn() -> bool,
    mut build: impl FnMut(&HeteroPlan) -> Result<T, E>,
) -> Result<T, E> {
    let first_plan = beside_host(first_plan, demand, accelerators);
    let first = match build(&first_plan) {
        Ok(m) => return Ok(m),
        Err(e) => e,
    };
    if cancelled() {
        return Err(first);
    }
    let mut rungs = later_rungs(demand, cards, accelerators);
    if !is_out_of_memory(&first) {
        tracing::warn!("{what}: placement failed ({first}); trying the host");
        rungs.retain(|(name, _)| name == "the host");
    }
    let mut tried = vec![first_plan];
    let mut last_oom = is_out_of_memory(&first);
    for (name, plan) in rungs {
        if !last_oom && name != "the host" {
            continue;
        }
        if tried.iter().any(|t| super::dry_plan::same_plan(t, &plan)) {
            continue;
        }
        tried.push(plan.clone());
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

/// [`load_blocks`] on the cards as they are now, each keeping `reserve` free for the work a
/// request runs beside the blocks. The pressure the fleet is under is taken off.
pub fn load_blocks_reserving<T>(
    layers: usize,
    bytes: u64,
    reserve: u64,
    what: &str,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    build: impl Fn(&HeteroPlan) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    load_blocks(layers, bytes, &cards(reserve), what, cancel, build)
}

/// The cards a whole component is offered, in the order it takes them: those offering
/// `want`, then those offering only `need`. A card offering less is not listed.
pub fn whole_order(cards: &[(usize, u64)], need: u64, want: u64) -> Vec<(usize, u64)> {
    let want = want.max(need);
    cards
        .iter()
        .filter(|(_, free)| *free >= want)
        .chain(
            cards
                .iter()
                .filter(|(_, free)| *free >= need && *free < want),
        )
        .copied()
        .collect()
}

/// Build a component that runs whole on one device, among `cards`: on the first offering
/// `want`, else on the first offering `need`, else on the host. A card that runs out of memory
/// hands the component to the next rung. Returns the component and the device it was
/// built on.
pub fn load_whole_on<T>(
    what: &str,
    need: u64,
    want: u64,
    cards: &[(usize, u64)],
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    mut build: impl FnMut(&Device) -> anyhow::Result<T>,
) -> anyhow::Result<(T, Device)> {
    let ranked = whole_order(cards, need, want);
    let demand = Demand {
        layers: 1,
        block_bytes: need,
        ..Demand::default()
    };
    let mut built_on = Device::Cpu;
    let built = load_placed(
        what,
        &demand,
        &ranked,
        || cancel.is_some_and(crate::inference::serve::cancel::CancelToken::is_cancelled),
        |plan| {
            let kind = plan.segments.first().map_or(DeviceKind::Cpu, |s| s.kind);
            let device = open(kind)?;
            let built = build(&device).map_err(|e| anyhow::anyhow!("{e:#}"))?;
            built_on = device;
            Ok::<T, anyhow::Error>(built)
        },
    )?;
    Ok((built, built_on))
}

/// Run work that sits whole on one device and takes `need` bytes of card memory there: on
/// each card of `cards` offering `need`, in their order, the next one when a card runs out of
/// memory, then on the host when `host` is set. Any other failure skips the remaining cards.
/// `None` when no card offered `need` and the host was not allowed.
pub fn run_whole_on<T, E: std::fmt::Display>(
    what: &str,
    need: u64,
    cards: &[(usize, u64)],
    host: bool,
    mut work: impl FnMut(&Device) -> Result<T, E>,
) -> Option<Result<T, E>> {
    let mut last = None;
    for &(idx, _) in cards.iter().filter(|(_, free)| *free >= need) {
        let device = device_of(DeviceKind::Cuda(idx));
        if device.is_cpu() {
            continue;
        }
        match work(&device) {
            Ok(v) => return Some(Ok(v)),
            Err(e) => {
                let oom = is_out_of_memory(&e);
                tracing::warn!("{what}: CUDA:{idx} failed ({e})");
                last = Some(Err(e));
                if !oom {
                    break;
                }
                release_pools();
            }
        }
    }
    if host {
        return Some(work(&Device::Cpu));
    }
    last
}

/// Build a component of uneven stages, `weights` bytes each in forward order, packed onto the
/// cards as they are now with `reserve` kept free on each for the work a request does there.
/// `build` receives the device of every stage. A build that runs out of card memory is packed
/// again at each shrinking share of the cards the host-assisted rung offers, the host taking
/// what no card holds, and last on the host alone. Any other failure goes straight to the
/// host, and if that fails too the first error is returned. A cancelled load stops.
pub fn load_stages<T, E: std::fmt::Display>(
    what: &str,
    weights: &[u64],
    reserve: u64,
    mut build: impl FnMut(&[Device]) -> Result<T, E>,
) -> Result<T, E> {
    let cards = cards(reserve);
    let devices: Vec<Device> = cards
        .iter()
        .map(|(idx, _)| device_of(DeviceKind::Cuda(*idx)))
        .collect();
    let pack = |pct: u64| -> Vec<Device> {
        let budgets: Vec<u64> = cards
            .iter()
            .map(|(_, free)| free.saturating_mul(pct) / 100)
            .collect();
        super::plan::place(weights, &budgets)
            .into_iter()
            .map(|slot| {
                slot.and_then(|d| devices.get(d).cloned())
                    .unwrap_or(Device::Cpu)
            })
            .collect()
    };
    let host = vec![Device::Cpu; weights.len()];
    let mut rungs: Vec<(String, Vec<Device>)> = HOST_ASSISTED_SHARES_PCT
        .iter()
        .map(|pct| (format!("the cards at {pct}% with the host"), pack(*pct)))
        .collect();
    rungs.push(("the host".to_string(), host));
    let locations = |d: &[Device]| d.iter().map(Device::location).collect::<Vec<_>>();
    let first_plan = pack(100);
    let first = match build(&first_plan) {
        Ok(m) => return Ok(m),
        Err(e) => e,
    };
    if scoped_cancelled() {
        return Err(first);
    }
    if !is_out_of_memory(&first) {
        tracing::warn!("{what}: placement failed ({first}); trying the host");
        rungs.retain(|(name, _)| name == "the host");
    }
    let mut tried = vec![locations(&first_plan)];
    let mut last_oom = is_out_of_memory(&first);
    for (name, plan) in rungs {
        if (!last_oom && name != "the host") || tried.contains(&locations(&plan)) {
            continue;
        }
        tried.push(locations(&plan));
        release_pools();
        tracing::warn!("{what}: retrying on {name}");
        match build(&plan) {
            Ok(m) => {
                tracing::info!("{what}: placed on {name}");
                return Ok(m);
            }
            Err(e) => {
                if scoped_cancelled() {
                    return Err(e);
                }
                tracing::warn!("{what}: {name} failed ({e})");
                last_oom = is_out_of_memory(&e);
            }
        }
    }
    Err(first)
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
    load_placed(what, demand, &cards(0), scoped_cancelled, build)
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
    let mut cards = cards(0);
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
pub(crate) fn release_pools() {
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
    fn every_spelling_of_out_of_memory_is_recognised() {
        for e in [
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")",
            "boom: Cuda_Error_OUT_OF_MEMORY at line 17",
            "alloc failed: out of memory",
            "[oom] kernel out on card 1",
            "cuda OOM in QCudaStorage upload",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "cudnn: CUDNN_STATUS_ALLOC_FAILED",
            "workspace alloc failed",
        ] {
            assert!(is_out_of_memory(e), "{e}");
        }
    }

    #[test]
    fn other_failures_are_not_out_of_memory() {
        for e in [
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS)",
            "shape mismatch: expected (4, 64), got (4, 32)",
            "zoom and cool hold the letters, not the word",
            "missing tensor blk.0.attn_q.weight",
        ] {
            assert!(!is_out_of_memory(e), "{e}");
        }
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
    fn a_whole_component_prefers_the_card_that_runs_it_best() {
        let cards = [(0, 3 * GB), (1, 5 * GB), (2, 9 * GB)];
        let order: Vec<usize> = whole_order(&cards, 4 * GB, 8 * GB)
            .iter()
            .map(|c| c.0)
            .collect();
        assert_eq!(order, vec![2, 1]);
        assert!(whole_order(&cards, 10 * GB, 10 * GB).is_empty());
        let order: Vec<usize> = prefer(&cards, Some(1)).iter().map(|c| c.0).collect();
        assert_eq!(order, vec![1, 0, 2]);
    }

    #[test]
    fn a_one_shot_component_is_offered_the_hot_cards_last() {
        let cards = [(0, 10 * GB), (1, 6 * GB)];
        assert_eq!(aside(&cards, &[0], 4 * GB), vec![(1, 6 * GB), (0, 6 * GB)]);
        assert_eq!(aside(&cards, &[0], u64::MAX), vec![(1, 6 * GB)]);
    }

    #[test]
    fn the_stem_is_held_on_the_card_the_spread_starts_on() {
        let d = demand(10, 10);
        // Whole when the blocks, the stem and the floor fit one card.
        assert_eq!(
            kinds(&d.plan_with_stem(&[(4, 13 * GB)], 2 * GB, 0)),
            vec!["CUDA(4)".to_string()]
        );
        // Spread otherwise, with fewer blocks on the card carrying the stem.
        let plan = d.plan_with_stem(&[(0, 8 * GB), (1, 8 * GB)], 2 * GB, 0);
        let on = |k: DeviceKind| -> usize {
            plan.segments
                .iter()
                .filter(|s| s.kind == k)
                .map(HeteroSegment::num_layers)
                .sum()
        };
        assert_eq!(on(DeviceKind::Cuda(0)) + on(DeviceKind::Cuda(1)), 10);
        assert!(
            on(DeviceKind::Cuda(0)) < on(DeviceKind::Cuda(1)),
            "{:?}",
            kinds(&plan)
        );
    }

    #[test]
    fn a_stem_goes_on_the_card_only_when_it_costs_the_blocks_nothing() {
        let d = Demand {
            runtime_reserve: GB,
            ..demand(10, 8)
        };
        let (plan, stem) = d.plan_stem_if_free(&[(0, 12 * GB)], 2 * GB);
        assert!(stem);
        assert_eq!(kinds(&plan), vec!["CUDA(0)".to_string()]);
        let (plan, stem) = d.plan_stem_if_free(&[(0, 10 * GB)], 6 * GB);
        assert!(!stem);
        assert_eq!(kinds(&plan), vec!["CUDA(0)".to_string()]);
    }

    #[test]
    fn host_blocks_go_to_an_accelerator_first() {
        let d = demand(10, 10);
        let plan = beside_host(HeteroPlan::forced_gpu(10, 4, 0), &d, &[(0, 100 * GB)]);
        assert_eq!(
            kinds(&plan),
            vec!["CUDA(0)".to_string(), "opencl(0)".to_string()]
        );
        assert_eq!(plan.segments[1].layer_start, 4);
        assert_eq!(plan.segments[1].layer_end, 10);
    }

    #[test]
    fn a_rung_that_repeats_a_tried_plan_is_skipped() {
        let mut seen = Vec::new();
        let _: Result<(), String> = load_placed(
            "t",
            &demand(1, 1),
            &[(0, 8 * GB)],
            || false,
            |plan| {
                seen.push(kinds(plan));
                Err("out of memory".to_string())
            },
        );
        assert_eq!(
            seen,
            vec![vec!["CUDA(0)".to_string()], vec!["CPU".to_string()]]
        );
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
    fn whole_work_no_card_offers_room_for_runs_on_the_host_only_when_allowed() {
        let cards = [(0, GB)];
        let ran: Option<Result<bool, String>> =
            run_whole_on("t", 2 * GB, &cards, true, |d| Ok(d.is_cpu()));
        assert_eq!(ran, Some(Ok(true)));
        let ran: Option<Result<bool, String>> =
            run_whole_on("t", 2 * GB, &cards, false, |d| Ok(d.is_cpu()));
        assert_eq!(ran, None);
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
