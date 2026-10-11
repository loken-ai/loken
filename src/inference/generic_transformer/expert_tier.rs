//! A layer's routed experts kept in host memory, with an adaptive cache of them on a card.
//!
//! The experts stay in the file's mapping. The card holds `slots` of them in stacks of the same
//! format and shape as the host's, so the expert kernels that read a stack on a card read a slot
//! as they read an expert: the routed ids are rewritten to slot numbers. A routed expert that
//! has no slot is computed on the host from the mapping, beside the card's own launch, and its
//! product joins the card's. Every routed expert's use is counted, and a missing expert that
//! the counts show to be hotter than the coldest slot not in use this token is copied into it
//! by a worker, which never holds up a token.
//!
//! A missing expert is run in one of two places, split by the rates measured on the layer: it is
//! copied from the host into the card's transit room and run there with the others, or computed
//! on the host. The two proceed together and the split leaves neither waiting on the other.
//!
//! A card with room for every expert of its layers holds them all: the layer is resident, and the
//! tier is the stacks the kernels read, with no host in the token.
//!
//! The counts decay: when `n_experts` tokens have routed through a layer, each count halves, so
//! the cache follows the routing of the text in hand rather than the routing since the load.

use super::moe_weights::{ExpertRouting, MoeWeights};
use crate::tensor::quantized::{GgmlDType, QCudaStorage, QHostTensor, QTensor};
use crate::tensor::{Device, Result, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// What the tier knows of each expert and slot, under one lock: every access is one token's
/// bookkeeping, short next to the launches around it.
struct TierState {
    /// Slot holding each expert, `-1` when it has none.
    slot_of: Vec<i32>,
    /// Expert in each slot; `-1` for a slot never filled and `-2` for one being filled.
    owner: Vec<i32>,
    /// Decayed count of the times each expert was routed to.
    heat: Vec<f32>,
    /// The times each expert was routed to since the load, undecayed: the run's profile.
    counts: Vec<u32>,
    /// Tokens since the counts last halved.
    since_decay: usize,
    /// The card's copy of `slot_of`, while a recorded decode reads it.
    table: Option<Arc<graph::SlotTable>>,
}

impl TierState {
    /// `rows` more tokens routed: the counts cool every `ADAPT_EVERY_TOKENS` of them.
    fn tokens_routed(&mut self, rows: usize) {
        self.since_decay += rows;
        if self.since_decay >= ADAPT_EVERY_TOKENS {
            self.since_decay = 0;
            self.heat.iter_mut().for_each(|h| *h *= ADAPT_DECAY);
        }
    }

    /// Say that `expert` is in `slot`, or in none (`-1`), to the card as well.
    fn set_slot(&mut self, expert: usize, slot: i32) {
        self.slot_of[expert] = slot;
        if let Some(t) = &self.table {
            t.set(expert, slot);
        }
    }
}

mod graph;
pub use graph::TierWorker;

/// Set by the tests that run the kernels of a recorded decode without recording them.
static GRAPH_WITHOUT_CAPTURE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Run the kernels of a recorded decode without recording them while `on`: what a recording
/// is checked against, and what makes the buffers they keep before the recording.
pub(crate) fn kernels_of_a_recording(on: bool) {
    GRAPH_WITHOUT_CAPTURE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The hit rate from which a layer's extra slots are not given back to it after a prefill:
/// of ten routed experts it misses half of one.
const REFILL_BELOW_HIT_RATE: f64 = 0.95;

/// How the cache follows the routing. An expert's heat counts the tokens that routed to it, and
/// every `ADAPT_EVERY_TOKENS` tokens the counts are multiplied by `ADAPT_DECAY`. A missing expert
/// takes a slot only once its heat reaches `ADAPT_MIN_HEAT` and exceeds the coldest slot's by
/// `ADAPT_MARGIN`, the pairs with the largest gain first, at most `ADAPT_SWAPS_A_REQUEST` for
/// each request a layer serves: a one-off route never moves an expert the decode keeps using.
const ADAPT_EVERY_TOKENS: usize = 256;
const ADAPT_DECAY: f32 = 0.5;
const ADAPT_MIN_HEAT: f32 = 2.0;
const ADAPT_MARGIN: f32 = 1.5;
const ADAPT_SWAPS_A_REQUEST: usize = 2;
/// Copies into the transit room timed before the share it takes of a request's missing experts
/// is fixed for the run.
const TRANSIT_PACE_SAMPLES: u64 = 16;

/// One layer's cache.
pub struct LayerTier {
    dev: Device,
    /// Gate, up and down stacks as the file stores them, `[n_experts, ..]`.
    host: [Arc<QTensor>; 3],
    /// The same stacks on the card, `[slots, ..]`: the slots that stay.
    card: [QTensor; 3],
    /// How many slots `card` has. The slots numbered from there on are in `extra`.
    base_slots: usize,
    /// The slots a prefill can take the card's memory back from: the stacks that hold them, or
    /// `None` while they are lent. A lent slot holds nothing (owner `-3`).
    extra: Mutex<Option<[QTensor; 3]>>,
    /// Bytes of one expert in each stack.
    expert_bytes: [usize; 3],
    n_experts: usize,
    state: Mutex<TierState>,
    hits: AtomicU64,
    routed: AtomicU64,
    promotions: AtomicU64,
    /// The layer that reports the cache's hit rate.
    leader: std::sync::atomic::AtomicBool,
    /// What a wide batch of rows reads instead: the whole layer, on the card for the one launch.
    stage: Option<Stage>,
    /// The cache holds every expert, expert `e` in slot `e`: nothing is ever missing.
    resident: bool,
    /// While a prompt holds the layer's stacks: the routing of its chunks so far and the stream
    /// they ran on, ranked into the cache once, when the hold ends.
    prompt_ids: Mutex<Option<(Arc<cudarc::driver::CudaStream>, Vec<Tensor>)>>,
    /// The stream this layer's copies into the cache run on.
    copy: Arc<cudarc::driver::CudaStream>,
    /// A copy into the cache is queued or running.
    promoting: std::sync::atomic::AtomicBool,
    /// Microseconds a decoded token took here, smoothed: what a batch's rows cost one by one.
    decode_us: AtomicU64,
    /// What a decoded token moves, made on the first.
    scratch: Mutex<Option<Scratch>>,
    /// Microseconds (an `f64`'s bits, `0` before a measure) the copy of one missing expert to the
    /// transit room takes, and the host's computation of one.
    upload_us: AtomicU64,
    host_us: AtomicU64,
    /// Samples the two paces were measured over, and the share of the missing experts the
    /// transit room takes once they are, in eighths: set once, so that a request's split no
    /// longer follows the clock and a run gives the text it gave.
    paced: AtomicU64,
    transit_eighths: AtomicU64,
    /// Missing experts copied to the transit room for the token that wanted them.
    uploaded: AtomicU64,
    /// The layer's side of a decode recorded in a graph, once enabled.
    graph: std::sync::OnceLock<graph::GraphSide>,
}

/// What one decoded token moves between the card and the host, in page-locked memory and in
/// buffers on the card that stay where they are: the copies are asynchronous and the token waits
/// once, for the last of those that come back.
struct Scratch {
    pinned: *mut u8,
    /// Floats of the hidden row, and the routed experts per token, that the buffers are sized for.
    hidden: usize,
    k: usize,
    /// On the card: the slot of each routed pair (a slot past the base ones is in the extra
    /// stacks), its weight if the card runs it, and the product the host sends back.
    slots: Tensor,
    weights: Tensor,
    product: Tensor,
    /// The same for the experts run from the transit room: their ids there and their weights.
    transit_ids: Tensor,
    transit_weights: Tensor,
    /// Recorded around the copies into the transit room, and how many experts they carried:
    /// read on the layer's next token, when they are long done.
    upload_start: cudarc::driver::CudaEvent,
    upload_end: cudarc::driver::CudaEvent,
    in_flight: std::sync::atomic::AtomicUsize,
}

// Safety: the page-locked block is used by one decode at a time, under the tier's lock.
unsafe impl Send for Scratch {}

/// The views of a decode's page-locked buffer.
struct Pinned<'a> {
    x: &'a mut [f32],
    ids: &'a mut [u32],
    weights: &'a mut [f32],
    product: &'a mut [f32],
    slots: &'a mut [u32],
    slot_weights: &'a mut [f32],
    transit_ids: &'a mut [u32],
    transit_weights: &'a mut [f32],
}

impl Scratch {
    fn new(dev: &Device, hidden: usize, k: usize) -> Result<Self> {
        dev.as_cuda_device()?
            .0
            .context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        let bytes = 4 * (2 * hidden + 6 * k);
        let pinned = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
            .map_err(|e| crate::tensor::Error::msg(format!("pinned scratch: {e}")))?
            as *mut u8;
        let host = Device::Cpu;
        Ok(Self {
            pinned,
            hidden,
            k,
            slots: Tensor::from_vec(vec![0u32; k], (k,), &host)?.to_device(dev)?,
            weights: Tensor::from_vec(vec![0f32; k], (1, k), &host)?.to_device(dev)?,
            product: Tensor::from_vec(vec![0f32; hidden], (1, hidden), &host)?.to_device(dev)?,
            transit_ids: Tensor::from_vec(vec![0u32; k], (k,), &host)?.to_device(dev)?,
            transit_weights: Tensor::from_vec(vec![0f32; k], (1, k), &host)?.to_device(dev)?,
            upload_start: timed_event(dev)?,
            upload_end: timed_event(dev)?,
            in_flight: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// The row the card sends the host, the choice's ids and weights, then what goes back:
    /// the product, the slots and the weights the card runs, the transit ids and weights. The
    /// eight views are disjoint ranges of the pinned buffer, which only the one thread running
    /// the layer touches.
    #[allow(clippy::mut_from_ref)]
    fn parts(&self) -> Pinned<'_> {
        let (h, k) = (self.hidden, self.k);
        unsafe {
            let f = self.pinned as *mut f32;
            let u = self.pinned as *mut u32;
            Pinned {
                x: std::slice::from_raw_parts_mut(f, h),
                ids: std::slice::from_raw_parts_mut(u.add(h), k),
                weights: std::slice::from_raw_parts_mut(f.add(h + k), k),
                product: std::slice::from_raw_parts_mut(f.add(h + 2 * k), h),
                slots: std::slice::from_raw_parts_mut(u.add(2 * h + 2 * k), k),
                slot_weights: std::slice::from_raw_parts_mut(f.add(2 * h + 3 * k), k),
                transit_ids: std::slice::from_raw_parts_mut(u.add(2 * h + 4 * k), k),
                transit_weights: std::slice::from_raw_parts_mut(f.add(2 * h + 5 * k), k),
            }
        }
    }
}

/// An event that keeps the time it was reached at.
fn timed_event(dev: &Device) -> Result<cudarc::driver::CudaEvent> {
    dev.as_cuda_device()?
        .0
        .context()
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| crate::tensor::Error::msg(format!("event: {e:?}")))
}

impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe { cudarc::driver::result::free_host(self.pinned as *mut std::ffi::c_void) }.ok();
    }
}

/// The card's address of the first element of `t`, which a copy reads or writes.
fn address_of<T: crate::tensor::kernel_ffi::CudaDType>(t: &Tensor) -> Result<u64> {
    use cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        crate::tensor::StorageView::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            Ok(slice.device_ptr(slice.stream()).0
                + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
        }
        _ => crate::tensor::bail!("an expert tier buffer is on the card"),
    }
}

/// Wall time between marks along a call, recorded under the stage profile.
struct Laps(Option<std::time::Instant>);

impl Laps {
    fn new() -> Self {
        Self(crate::inference::offload::stage_prof_enabled().then(std::time::Instant::now))
    }

    /// The time since the last mark is the stage `name`'s.
    fn mark(&mut self, name: &'static str) {
        if let Some(t) = self.0 {
            crate::inference::offload::prof_record(name, t.elapsed().as_nanos() as u64);
            self.0 = Some(std::time::Instant::now());
        }
    }
}

/// The card's room for a layer's experts to pass through, and the copies that fill it.
pub struct CardStager {
    dev: Device,
    /// Bytes per microsecond the host sends to this card, measured when the stager is made.
    bandwidth: f64,
    copy: Arc<cudarc::driver::CudaStream>,
    /// Bytes of the widest stack of each kind: what the arenas of a wide batch hold.
    widest: [usize; 3],
    /// How many experts the transit room holds.
    transit: usize,
    /// The transit room: two sets of arenas, each a stack of `transit` experts, that consecutive
    /// layers take turns over.
    buffers: [StageBuffer; 2],
    /// The room a wide batch fills with a layer's whole stacks, from the memory the caches lend
    /// while a prefill runs.
    full: Mutex<Option<[FullBuffer; 2]>>,
}

/// One of two sets of arenas, a stack each, that consecutive layers take turns over.
struct StageBuffer {
    arenas: [Arc<cudarc::driver::CudaSlice<u8>>; 3],
}

/// [`StageBuffer`] wide enough for the whole stacks of the widest layer.
struct FullBuffer {
    arenas: [Arc<cudarc::driver::CudaSlice<u8>>; 3],
    /// Recorded when the layer that read the buffer last has queued all its kernels: the next
    /// fill waits for it.
    free: Mutex<Option<cudarc::driver::CudaEvent>>,
}

/// A layer's view of its buffer: the three stacks as tensors over the arenas, and the copies in
/// flight into them.
struct Stage {
    stager: Arc<CardStager>,
    buffer: usize,
    /// The transit room as stacks of `transit` experts.
    compact: [QTensor; 3],
    /// The whole stacks of the layer in the room of a wide batch, while there is one.
    full: Mutex<Option<Arc<[QTensor; 3]>>>,
    /// The fill's end, when one has been queued and not yet read.
    ready: Mutex<Option<cudarc::driver::CudaEvent>>,
    /// Host ranges, per stack, that are page-locked: copies from them are asynchronous.
    pinned: [Option<(usize, usize)>; 3],
    /// The stacks are kept for several forwards in a row (see `hold_stacks`).
    held: std::sync::atomic::AtomicBool,
    /// While held: the fill has landed and the compute stream has waited for it.
    filled: std::sync::atomic::AtomicBool,
}

/// The stacks' dims with the expert axis replaced by `n`.
fn with_experts(host: &QTensor, n: usize) -> Vec<usize> {
    let mut dims = host.shape().dims().to_vec();
    dims[0] = n;
    dims
}

/// The first `n` experts of `host` on `dev`: a view of the host's own bytes, so the host holds no
/// second copy of what the card was filled from.
fn first_experts(host: &Arc<QTensor>, n: usize, dev: &Device) -> Result<QTensor> {
    let inner = host.native_qtensor();
    let bytes = inner.data();
    let per_expert = bytes.len() / host.shape().dims()[0];
    let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(inner.clone());
    let view = unsafe {
        QHostTensor::view(
            owner,
            bytes.as_ptr(),
            0,
            n * per_expert,
            host.dtype(),
            with_experts(host, n),
        )
    }
    .map_err(|e| crate::tensor::Error::msg(e.0))?;
    QTensor::from_native(Arc::new(view), dev)
}

/// The first `n` experts of `host` as a view of the host's own bytes, and their size.
fn first_view(host: &Arc<QTensor>, n: usize) -> Result<(Arc<QHostTensor>, usize)> {
    let inner = host.native_qtensor();
    let bytes = inner.data();
    let per_expert = bytes.len() / host.shape().dims()[0];
    let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(inner.clone());
    let view = unsafe {
        QHostTensor::view(
            owner,
            bytes.as_ptr(),
            0,
            n * per_expert,
            host.dtype(),
            with_experts(host, n),
        )
    }
    .map_err(|e| crate::tensor::Error::msg(e.0))?;
    Ok((Arc::new(view), n * per_expert))
}

/// [`first_experts`] with the card's copy in the memory pool of what is lent, so that freeing it
/// returns its memory to the card.
fn lent_experts(host: &Arc<QTensor>, n: usize, dev: &Device) -> Result<QTensor> {
    let (view, len) = first_view(host, n)?;
    let cd = dev.as_cuda_device()?;
    let storage = QCudaStorage::zeroed_in_lent_pool(&cd, len, host.dtype())
        .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
    Ok(QTensor::over_card(view, dev, storage))
}

impl super::GenericHeteroTransformer {
    /// Give every layer whose experts the plan left on the host a cache of them on its card:
    /// `slots` takes what each card has left of its budget, `free_before` being what the card
    /// had free before the load.
    pub(super) fn attach_expert_tiers(
        &mut self,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        free_before: &std::collections::HashMap<usize, u64>,
    ) -> Result<()> {
        use super::LayerDevice;
        use crate::inference::place::{expert_cache, layer_executor::DeviceKind};
        // Locked pages cannot be reclaimed: the cards share what the host can give.
        let mut pin_left = expert_cache::host_pin_budget();
        let mut ordinals: Vec<_> = cuda_devices.keys().copied().collect();
        ordinals.sort_unstable();
        for ordinal in ordinals {
            let dev = &cuda_devices[&ordinal];
            let tiered: Vec<usize> = (0..self.layers.len())
                .filter(|&l| self.layer_devs[l] == LayerDevice::Cuda(ordinal))
                .filter(|&l| {
                    self.layers[l]
                        .moe
                        .as_ref()
                        .is_some_and(|m| m.experts_on_host() && serves(m))
                })
                .collect();
            let Some(&first) = tiered.first() else {
                continue;
            };
            // Room for the layers' experts to pass through: two layers at once, each as wide
            // as the widest stack of any layer.
            let mut widest = [0usize; 3];
            for &l in &tiered {
                let m = self.layers[l]
                    .moe
                    .as_ref()
                    .expect("a tiered layer has experts");
                let stacks = [
                    m.gate_exps_sep.as_ref().expect("separate stacks"),
                    m.up_exps_sep.as_ref().expect("separate stacks"),
                    &m.down_exps,
                ];
                for i in 0..3 {
                    widest[i] = widest[i].max(stacks[i].native_qtensor().data().len());
                }
            }
            // One slot in every one of the card's layers: the layers' formats differ, so what a
            // slot costs is the sum of what it costs in each.
            let bytes: usize = tiered
                .iter()
                .map(|&l| {
                    expert_bytes(
                        self.layers[l]
                            .moe
                            .as_ref()
                            .expect("a tiered layer has experts"),
                    )
                })
                .sum();
            // The draft head's layer, when it runs on this card with its experts on the host,
            // takes a cache of them too: a slot costs its share in that layer as well.
            let draft_bytes = self
                .draft
                .as_ref()
                .filter(|d| d.layer_dev == LayerDevice::Cuda(ordinal))
                .and_then(|d| d.layer.moe.as_ref())
                .filter(|m| m.experts_on_host() && serves(m))
                .map(expert_bytes);
            let bytes = bytes + draft_bytes.unwrap_or(0);
            let budget = plan
                .segments
                .iter()
                .filter(|s| s.kind == DeviceKind::Cuda(ordinal))
                .map(|s| s.free_memory_bytes)
                .max()
                .unwrap_or(0);
            let before = free_before.get(&ordinal).copied().unwrap_or(0);
            // Room for every expert of every layer beside the plan's own reserve: the layers
            // are resident, and need neither a copy room nor a share of the card for one.
            let widest_mixture = tiered
                .iter()
                .filter_map(|&l| self.layers[l].moe.as_ref().map(|m| m.n_experts))
                .max()
                .unwrap_or(0);
            if expert_cache::slots_per_layer(budget, before, expert_cache::free_now(dev), 1, bytes)
                >= widest_mixture
            {
                for &l in &tiered {
                    let moe = self.layers[l]
                        .moe
                        .as_mut()
                        .expect("a tiered layer has experts");
                    moe.tier = Some(Arc::new(LayerTier::all_on_card(moe, dev)?));
                }
                tracing::info!(
                    "expert tier on card {ordinal}: {} layers resident, every expert on the card, {:.2} GB",
                    tiered.len(),
                    (widest_mixture * bytes) as f64 / 1e9
                );
                continue;
            }
            // What the wide batches the experts now run on the card for keep live: the chunk the
            // prefill is to run at. The plan already leaves a reserve free for the work of a
            // forward, so the cache gives up only what the chunk needs beyond it.
            let shape = self.prefill_shape_with(true);
            let chunk_bytes = (shape.bytes_per_token as u64)
                .saturating_mul(shape.target_chunk as u64)
                .saturating_add(shape.fixed_bytes as u64);
            // The rooms of the wide batches are lent by the cache while a prefill runs, so the
            // base slots leave room for the transit room that stays and for the room those
            // batches fill, which is taken from the lent slots.
            let stager = CardStager::new(dev, widest, widest_mixture)?;
            let held: u64 = stager.full_bytes()
                + stager.transit_bytes()
                + chunk_bytes.saturating_sub(before.saturating_sub(budget));
            let slots = expert_cache::slots_per_layer(
                budget.saturating_sub(held),
                before,
                expert_cache::free_now(dev),
                1,
                bytes,
            );
            if slots == 0 {
                continue;
            }
            // The room a prefill takes from the card is lent to the cache while it is not
            // running: the plan's reserve less the work a decode keeps, as more slots that a
            // prefill gives back.
            let work_floor = crate::inference::place::runtime_demand::CUBLAS_WORKSPACE_BYTES
                .saturating_add(shape.fixed_bytes as u64);
            let lent = before
                .saturating_sub(budget)
                .saturating_sub(work_floor)
                .saturating_add(stager.full_bytes());
            let extra = (lent / bytes.max(1) as u64) as usize;
            let mut locked = 0usize;
            for (rank, &l) in tiered.iter().enumerate() {
                let moe = self.layers[l]
                    .moe
                    .as_mut()
                    .expect("a tiered layer has experts");
                let mut tier = LayerTier::new(moe, dev, slots)?.with_extra(extra)?;
                let layer_bytes = (tier.expert_bytes.iter().sum::<usize>() * tier.n_experts) as u64;
                let pin = layer_bytes <= pin_left;
                pin_left = pin_left.saturating_sub(if pin { layer_bytes } else { 0 });
                tier.stage = Some(Stage::new(&stager, rank % 2, &tier.host, pin)?);
                locked += usize::from(
                    tier.stage
                        .as_ref()
                        .is_some_and(|s| s.pinned.iter().all(Option::is_some)),
                );
                if l == first {
                    tier.leader.store(true, Ordering::Relaxed);
                }
                moe.tier = Some(Arc::new(tier));
            }
            // The head's layer reads its experts through the cache a token at a time and the
            // rows of a draft together; a prompt's rows take the host's grouped path, so the
            // layer has no transit room.
            if draft_bytes.is_some() {
                if let Some(moe) = self.draft.as_mut().and_then(|d| d.layer.moe.as_mut()) {
                    let tier = LayerTier::new(moe, dev, slots)?;
                    tier.enable_graph(
                        tier.host[0].shape().dims()[2],
                        moe.n_experts_used,
                        crate::inference::engine::decode_step::VERIFY_ROWS,
                    )?;
                    moe.tier = Some(Arc::new(tier));
                }
            }
            // The layers' decoded tokens run as kernels alone, served by one thread of the host.
            let arcs: Vec<Arc<LayerTier>> = tiered
                .iter()
                .filter_map(|&l| self.layers[l].moe.as_ref().and_then(|m| m.tier.clone()))
                .chain(
                    self.draft
                        .as_ref()
                        .filter(|d| d.layer_dev == LayerDevice::Cuda(ordinal))
                        .and_then(|d| d.layer.moe.as_ref().and_then(|m| m.tier.clone())),
                )
                .collect();
            for &l in &tiered {
                let moe = self.layers[l]
                    .moe
                    .as_ref()
                    .expect("a tiered layer has experts");
                if let Some(t) = &moe.tier {
                    let hidden = t.host[0].shape().dims()[2];
                    t.enable_graph(
                        hidden,
                        moe.n_experts_used,
                        crate::inference::engine::decode_step::VERIFY_ROWS,
                    )?;
                }
            }
            self.tier_workers.push(TierWorker::start(dev, arcs)?);
            tracing::info!(
                "expert tier on card {ordinal}: {} layers{} ({locked} with locked host pages, {:.1} GB/s to the card, {} experts in the transit room), {} experts each cached of {} ({extra} of them lent to a prefill), {:.2} GB",
                tiered.len(),
                if draft_bytes.is_some() { " and the draft head" } else { "" },
                stager.bandwidth * 1e-3,
                stager.transit,
                (slots + extra).min(self.layers[first].moe.as_ref().map_or(0, |m| m.n_experts)),
                self.layers[first].moe.as_ref().map_or(0, |m| m.n_experts),
                ((slots + extra) * bytes) as f64 / 1e9
            );
        }
        Ok(())
    }
}

impl super::GenericHeteroTransformer {
    /// Take the memory the caches lent to a prefill of `tokens` rows back to the cards, when the
    /// prefill needs more than they have free: every layer's extra slots are freed. How many
    /// layers lent.
    pub fn lend_expert_cache(&mut self, tokens: usize) -> usize {
        let shape = self.prefill_shape();
        if shape.target_chunk == 0 {
            return 0;
        }
        let rows = tokens.min(shape.target_chunk);
        // A prompt that routes more pairs than a layer has experts reads nearly all of them: it
        // fills a room with the layer's whole stacks, which the card has to find.
        let wide = self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref())
            .any(|m| m.tier.is_some() && tokens * m.n_experts_used >= m.n_experts);
        let room = if wide {
            self.layers
                .iter()
                .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
                .filter_map(|t| t.stager())
                .map(|s| s.full_bytes())
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let need = ((rows * shape.bytes_per_token + shape.fixed_bytes) as u64)
            .saturating_mul(shape.share.max(1) as u64)
            .saturating_add(room);
        let free = crate::inference::place::vram_manager::tightest_free_on_cards(shape.cards);
        let lent = if free >= need {
            0
        } else {
            self.layers
                .iter()
                .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
                .filter(|t| t.lend_extra())
                .count()
        };
        if lent > 0 {
            crate::inference::place::ladder::release_pools();
            crate::inference::place::vram_manager::residency_changed();
        }
        if wide {
            for tier in self
                .layers
                .iter()
                .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
            {
                if let Err(e) = tier.attach_full() {
                    tracing::warn!("a wide batch's room did not fit: {e}");
                    break;
                }
            }
        }
        lent
    }

    /// Give the caches their extra slots again, filled with the experts the prefill found
    /// hottest. How many experts were copied.
    pub fn return_expert_cache(&mut self) -> Result<usize> {
        let mut copied = 0;
        let mut stagers: Vec<Arc<CardStager>> = Vec::new();
        for tier in self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
        {
            tier.detach_full();
            if let Some(s) = tier.stager() {
                if !stagers.iter().any(|t| Arc::ptr_eq(t, &s)) {
                    stagers.push(s);
                }
            }
        }
        for stager in &stagers {
            stager.drop_full();
        }
        // What the prefill left in the pools goes back to the card before the slots take theirs.
        crate::inference::place::ladder::release_pools();
        for tier in self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
        {
            // A layer that cannot take its slots back runs on without them.
            match tier.restore_extra() {
                Ok(n) => copied += n,
                Err(e) => tracing::warn!("expert tier: the extra slots did not come back: {e}"),
            }
        }
        crate::inference::place::vram_manager::residency_changed();
        Ok(copied)
    }
}

impl LayerTier {
    /// This tier with room to read a wide batch's experts whole: `buffer` of `stager`, the
    /// host stacks page-locked when `pin`.
    pub fn with_stage(
        mut self,
        stager: &Arc<CardStager>,
        buffer: usize,
        pin: bool,
    ) -> Result<Self> {
        self.stage = Some(Stage::new(stager, buffer, &self.host, pin)?);
        self.attach_full()?;
        Ok(self)
    }

    /// The layer's whole stacks over the room of a wide batch, which is made if it is not.
    pub fn attach_full(&self) -> Result<()> {
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        stage.stager.ensure_full()?;
        let cd = self.dev.as_cuda_device()?;
        let full = stage.stager.full.lock().unwrap_or_else(|e| e.into_inner());
        let arenas = &full.as_ref().expect("the room was made")[stage.buffer].arenas;
        let mk = |i: usize| {
            let inner = self.host[i].native_qtensor().clone();
            let len = inner.data().len();
            let storage = QCudaStorage::sharing(arenas[i].clone(), len, self.host[i].dtype(), &cd);
            QTensor::over_card(inner, &self.dev, storage)
        };
        *stage.full.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Arc::new([mk(0), mk(1), mk(2)]));
        Ok(())
    }

    /// Let go of the layer's view of the room of a wide batch.
    pub fn detach_full(&self) {
        if let Some(stage) = self.stage.as_ref() {
            *stage.full.lock().unwrap_or_else(|e| e.into_inner()) = None;
            *stage.ready.lock().unwrap_or_else(|e| e.into_inner()) = None;
            stage.filled.store(false, Ordering::Release);
            stage.held.store(false, Ordering::Release);
        }
    }

    /// The rooms of the wide batches, made for the card of every layer that has one.
    pub fn stager(&self) -> Option<Arc<CardStager>> {
        self.stage.as_ref().map(|s| s.stager.clone())
    }
}

pub use super::moe_weights::prefill_bytes_per_token;

/// What a layer's tier can serve: separate gate and up stacks read by a SiLU, no biases, and a
/// format both the card's expert kernels and the host's dot read.
pub fn serves(moe: &MoeWeights) -> bool {
    let (Some(gate), Some(up)) = (moe.gate_exps_sep.as_ref(), moe.up_exps_sep.as_ref()) else {
        return false;
    };
    let kernels = |t: &QTensor| {
        crate::inference::moe_cuda::gemm::expert_kernels_serve(t.dtype())
            && crate::inference::moe_cpu::cpu_qmatmul_supported(t.dtype())
    };
    moe.use_silu
        && !moe.gelu
        && !moe.up_relu2
        && moe.swiglu_clamp.is_none()
        && moe.gate_exps_bias.is_none()
        && moe.up_exps_bias.is_none()
        && moe.down_exps_bias.is_none()
        && moe.decode_slots.is_some()
        && gate.dtype() == up.dtype()
        && kernels(gate)
        && kernels(&moe.down_exps)
}

/// Whether the routed experts of the file `content` read as a tier serves them: in every layer
/// that has experts, separate gate and up stacks of one format, no biases, and a format that the
/// card's kernels and the host's dot both read.
pub fn header_serves(content: &crate::tensor::quantized::gguf_file::Content) -> bool {
    let kernels = |d: GgmlDType| {
        crate::inference::moe_cuda::gemm::expert_kernels_serve(d)
            && crate::inference::moe_cpu::cpu_qmatmul_supported(d)
    };
    let mut layers = 0;
    for (name, gate) in &content.tensor_infos {
        let Some(prefix) = name.strip_suffix(".ffn_gate_exps.weight") else {
            continue;
        };
        let (up, down) = (
            content
                .tensor_infos
                .get(&format!("{prefix}.ffn_up_exps.weight")),
            content
                .tensor_infos
                .get(&format!("{prefix}.ffn_down_exps.weight")),
        );
        let biased = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .any(|n| {
                content
                    .tensor_infos
                    .contains_key(&format!("{prefix}.{n}.bias"))
            });
        let (Some(up), Some(down)) = (up, down) else {
            return false;
        };
        if biased
            || up.ggml_dtype != gate.ggml_dtype
            || !kernels(gate.ggml_dtype)
            || !kernels(down.ggml_dtype)
        {
            return false;
        }
        layers += 1;
    }
    layers > 0
}

/// Bytes of one expert of `moe` across its three stacks.
pub fn expert_bytes(moe: &MoeWeights) -> usize {
    let stacks = [
        moe.gate_exps_sep.as_ref(),
        moe.up_exps_sep.as_ref(),
        Some(&moe.down_exps),
    ];
    stacks
        .iter()
        .flatten()
        .map(|t| t.native_qtensor().data().len() / t.shape().dims()[0].max(1))
        .sum()
}

impl LayerTier {
    /// A cache of `slots` experts on `dev` for the layer `moe`, filled with its first experts.
    pub fn new(moe: &MoeWeights, dev: &Device, slots: usize) -> Result<Self> {
        let slots = slots.min(moe.n_experts);
        let host = [
            moe.gate_exps_sep
                .clone()
                .expect("a tier serves separate gate and up stacks"),
            moe.up_exps_sep
                .clone()
                .expect("a tier serves separate gate and up stacks"),
            moe.down_exps.clone(),
        ];
        let card = [
            first_experts(&host[0], slots, dev)?,
            first_experts(&host[1], slots, dev)?,
            first_experts(&host[2], slots, dev)?,
        ];
        let expert_bytes = [0, 1, 2]
            .map(|i| host[i].native_qtensor().data().len() / host[i].shape().dims()[0].max(1));
        let mut slot_of = vec![-1i32; moe.n_experts];
        let mut owner = vec![-1i32; slots];
        for s in 0..slots {
            slot_of[s] = s as i32;
            owner[s] = s as i32;
        }
        Ok(Self {
            dev: dev.clone(),
            host,
            card,
            base_slots: slots,
            extra: Mutex::new(None),
            expert_bytes,
            n_experts: moe.n_experts,
            state: Mutex::new(TierState {
                slot_of,
                owner,
                heat: vec![0.0; moe.n_experts],
                counts: vec![0; moe.n_experts],
                since_decay: 0,
                table: None,
            }),
            hits: AtomicU64::new(0),
            routed: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            leader: std::sync::atomic::AtomicBool::new(false),
            stage: None,
            resident: false,
            copy: dev
                .as_cuda_device()?
                .0
                .context()
                .new_stream()
                .map_err(|e| crate::tensor::Error::msg(format!("copy stream: {e:?}")))?,
            promoting: std::sync::atomic::AtomicBool::new(false),
            decode_us: AtomicU64::new(0),
            scratch: Mutex::new(None),
            prompt_ids: Mutex::new(None),
            upload_us: AtomicU64::new(0),
            host_us: AtomicU64::new(0),
            paced: AtomicU64::new(0),
            transit_eighths: AtomicU64::new(0),
            uploaded: AtomicU64::new(0),
            graph: std::sync::OnceLock::new(),
        })
    }

    /// A layer whose every expert is on the card `dev`: no copy room, no host.
    pub fn all_on_card(moe: &MoeWeights, dev: &Device) -> Result<Self> {
        let mut tier = Self::new(moe, dev, moe.n_experts)?;
        tier.resident = true;
        Ok(tier)
    }

    /// Whether every expert is on the card.
    pub fn is_resident(&self) -> bool {
        self.resident
    }

    /// The cached stacks, in expert order when the tier is resident.
    pub(super) fn card_stacks(&self) -> &[QTensor; 3] {
        &self.card
    }

    /// This tier with `extra` more slots that a prefill can take the card's memory back from.
    pub fn with_extra(mut self, extra: usize) -> Result<Self> {
        let n = extra.min(self.n_experts.saturating_sub(self.base_slots));
        if n == 0 {
            return Ok(self);
        }
        *self.extra.get_mut().unwrap_or_else(|e| e.into_inner()) = Some([
            lent_experts(&self.host[0], n, &self.dev)?,
            lent_experts(&self.host[1], n, &self.dev)?,
            lent_experts(&self.host[2], n, &self.dev)?,
        ]);
        let st = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        st.owner.resize(self.base_slots + n, -1);
        Ok(self)
    }

    /// Whether `rows` rows read the layer's experts through the cache, as decoded tokens would:
    /// always for a layer with a transit room, else while they read fewer experts than the
    /// layer has (a wider batch is served from the host, grouped by expert).
    pub fn rows_through_cache(&self, rows: usize, used: usize) -> bool {
        self.stage.is_some() || rows * used < self.n_experts
    }

    /// How many slots are past the base ones, lent or not.
    pub fn extra_slots(&self) -> usize {
        self.slots() - self.base_slots
    }

    /// The card's address of slot `slot` of stack `i`, in the base stacks or the extra ones.
    fn slot_addr(&self, i: usize, slot: usize) -> Result<u64> {
        let eb = self.expert_bytes[i];
        if slot < self.base_slots {
            return Ok(self.card[i].device_ptr()? as u64 + (slot * eb) as u64);
        }
        let extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
        match extra.as_ref() {
            Some(stacks) => {
                Ok(stacks[i].device_ptr()? as u64 + ((slot - self.base_slots) * eb) as u64)
            }
            None => crate::tensor::bail!("slot {slot} is lent"),
        }
    }

    /// Give the memory of the extra slots back to the card: the experts they held leave the
    /// cache and the stacks are freed. Whether there was anything to give.
    pub fn lend_extra(&self) -> bool {
        let mut extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
        if extra.is_none() {
            return false;
        }
        self.settle();
        if let Ok(cd) = self.dev.as_cuda_device() {
            let _ = cd.0.stream().synchronize();
        }
        let _ = self.copy.synchronize();
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for s in self.base_slots..st.owner.len() {
                if st.owner[s] >= 0 {
                    let e = st.owner[s] as usize;
                    st.set_slot(e, -1);
                }
                st.owner[s] = -3;
            }
        }
        *extra = None;
        if let Ok(cd) = self.dev.as_cuda_device() {
            let _ = cd.0.stream().synchronize();
            crate::tensor::quantized::trim_lent_pool(cd.0.ordinal());
        }
        true
    }

    /// Take the extra slots' memory back and fill them, from the cache's copier thread, with the
    /// hottest experts the base slots do not hold: the slots take no expert until its copy has
    /// landed, and nothing else takes them meanwhile. A card with less room than the slots took
    /// gets as many of them as fit. How many experts are on their way.
    pub fn restore_extra(self: &Arc<Self>) -> Result<usize> {
        let n = self.extra_slots();
        // A layer that already finds nearly every expert on the card gains nothing from more
        // slots, and a prefill takes their memory again.
        let (routed, hits, _) = self.counters();
        if routed > 0 && hits as f64 >= REFILL_BELOW_HIT_RATE * routed as f64 {
            return Ok(0);
        }
        {
            let mut extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
            if extra.is_some() || n == 0 {
                return Ok(0);
            }
            let mut fit = n;
            let stacks = loop {
                let made = (|| -> Result<[QTensor; 3]> {
                    Ok([
                        lent_experts(&self.host[0], fit, &self.dev)?,
                        lent_experts(&self.host[1], fit, &self.dev)?,
                        lent_experts(&self.host[2], fit, &self.dev)?,
                    ])
                })();
                match made {
                    Ok(stacks) => break Some(stacks),
                    Err(e) => {
                        if let Ok(cd) = self.dev.as_cuda_device() {
                            let _ = cd.0.stream().synchronize();
                            crate::tensor::quantized::trim_lent_pool(cd.0.ordinal());
                        }
                        if fit <= 1 {
                            tracing::warn!("expert tier: no room for the extra slots: {e}");
                            break None;
                        }
                        fit /= 2;
                    }
                }
            };
            let Some(stacks) = stacks else {
                return Ok(0);
            };
            *extra = Some(stacks);
            drop(extra);
            return self.fill_extra(fit);
        }
    }

    /// Fill the first `n` extra slots, now with memory behind them, from the hottest experts the
    /// base slots do not hold. How many experts are on their way.
    fn fill_extra(self: &Arc<Self>, n: usize) -> Result<usize> {
        let wanted: Vec<usize> = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for s in self.base_slots..self.base_slots + n {
                st.owner[s] = -2;
            }
            let mut order: Vec<usize> = (0..self.n_experts)
                .filter(|&e| st.slot_of[e] < 0 && st.heat[e] > 0.0)
                .collect();
            order.sort_by(|&a, &b| st.heat[b].total_cmp(&st.heat[a]));
            order.truncate(n);
            order
        };
        self.settle();
        self.promoting.store(true, Ordering::Release);
        let tier = self.clone();
        let count = wanted.len();
        copier(self.dev.as_cuda_device().map_or(0, |d| d.0.ordinal()))
            .send(Box::new(move || {
                let landed = (|| -> Result<()> {
                    for (j, &e) in wanted.iter().enumerate() {
                        tier.copy_in_async(e, tier.base_slots + j)?;
                    }
                    tier.copy
                        .synchronize()
                        .map_err(|e| crate::tensor::Error::msg(format!("slot copy: {e:?}")))
                })();
                let mut st = tier.state.lock().unwrap_or_else(|e| e.into_inner());
                for s in tier.base_slots..tier.base_slots + n {
                    st.owner[s] = -1;
                }
                match landed {
                    Ok(()) => {
                        for (j, &e) in wanted.iter().enumerate() {
                            let slot = tier.base_slots + j;
                            st.owner[slot] = e as i32;
                            st.set_slot(e, slot as i32);
                        }
                    }
                    Err(e) => tracing::warn!("expert tier: refilling the extra slots: {e}"),
                }
                drop(st);
                tier.promoting.store(false, Ordering::Release);
            }))
            .ok();
        Ok(count)
    }

    /// Wait for the copy in flight, if any, to land.
    pub fn settle(&self) {
        while self.promoting.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }

    /// Experts the card holds.
    pub fn slots(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .owner
            .len()
    }

    /// Whether `expert` is on the card now.
    pub fn resident(&self, expert: usize) -> bool {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).slot_of[expert] >= 0
    }

    /// (routed experts, of which the card held, copies made) since the load.
    /// The times each expert was routed to since the load.
    pub fn routing_counts(&self) -> Vec<u32> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .counts
            .clone()
    }

    /// Fill the slots with the experts `counts` ranks highest, hottest first, copied from the
    /// host stacks now, and give each expert the heat its rank earns, so that the decode's
    /// first routes do not move what the profile placed. How many experts were copied.
    /// How many experts the layer routes to.
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn prime(&self, counts: &[u32]) -> Result<usize> {
        if counts.len() != self.n_experts || self.resident {
            return Ok(0);
        }
        let top = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
        let mut ranked: Vec<usize> = (0..self.n_experts).collect();
        ranked.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
        let moves: Vec<(usize, usize)> = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for (h, &c) in st.heat.iter_mut().zip(counts) {
                *h = h.max(ADAPT_MIN_HEAT * c as f32 / top);
            }
            let slots = st.owner.len();
            let wanted: Vec<usize> = ranked.into_iter().take(slots).collect();
            // A wanted expert already in a slot keeps it; the other slots take the rest. A lent
            // slot (owner -3) holds nothing until it comes back, and takes no expert.
            let mut moves = Vec::new();
            let mut free: Vec<usize> = (0..slots)
                .filter(|&s| st.owner[s] != -3)
                .filter(|&s| st.owner[s] < 0 || !wanted.contains(&(st.owner[s] as usize)))
                .collect();
            for &e in &wanted {
                if st.slot_of[e] >= 0 {
                    continue;
                }
                let Some(slot) = free.pop() else { break };
                let old = st.owner[slot];
                if old >= 0 {
                    st.set_slot(old as usize, -1);
                }
                st.owner[slot] = -2;
                moves.push((e, slot));
            }
            moves
        };
        // The stacks were allocated on the card's stream moments ago: the copy stream writes
        // them only once that allocation has run.
        self.dev.as_cuda_device()?.0.synchronize()?;
        for &(e, slot) in &moves {
            self.copy_in_async(e, slot)?;
        }
        self.copy
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("slot copy: {e:?}")))?;
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for &(e, slot) in &moves {
            st.owner[slot] = e as i32;
            st.set_slot(e, slot as i32);
        }
        Ok(moves.len())
    }

    pub fn counters(&self) -> (u64, u64, u64) {
        (
            self.routed.load(Ordering::Relaxed),
            self.hits.load(Ordering::Relaxed),
            self.promotions.load(Ordering::Relaxed),
        )
    }

    /// One decoded token's expert block: `xs` `[1, hidden]` and the router's choice `ids`,
    /// `weights` `[1, k]`, all on the card. Returns `[1, hidden]` on the card.
    pub(super) fn decode(
        self: &Arc<Self>,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let started = std::time::Instant::now();
        // A recorded decode takes the layer through the card's kernels alone.
        if self.records(&xs.device()) {
            return self.decode_graph(moe, xs, ids, weights);
        }
        let out = self.decode_token(moe, xs, ids, weights);
        let us = started.elapsed().as_micros() as u64;
        let before = self.decode_us.load(Ordering::Relaxed);
        // A weighted mean that forgets a token in sixteen.
        self.decode_us.store(
            if before == 0 {
                us
            } else {
                (before * 15 + us) / 16
            },
            Ordering::Relaxed,
        );
        out
    }

    fn decode_token(
        self: &Arc<Self>,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let k = moe.n_experts_used;
        let host = Device::Cpu;
        let dev = xs.device().clone();
        let hidden = xs.elem_count();
        let cd = dev.as_cuda_device()?;
        let stream = cd.0.stream().clone();
        let mut guard = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(Scratch::new(&dev, hidden, k)?);
        }
        let scratch = guard.as_ref().expect("made above");
        let mut laps = Laps::new();
        // The last token's copies into the transit room are done: what they took sets the pace
        // this one's are split by.
        let carried = scratch.in_flight.swap(0, Ordering::Relaxed);
        if carried > 0 {
            if let Ok(ms) = scratch.upload_start.elapsed_ms(&scratch.upload_end) {
                blend(&self.upload_us, ms as f64 * 1e3 / carried as f64);
            }
        }
        let Pinned {
            x: h_x,
            ids: h_ids,
            weights: h_w,
            product: h_out,
            slots: h_slots,
            slot_weights: h_hw,
            transit_ids: h_tids,
            transit_weights: h_tw,
        } = scratch.parts();
        // The hidden row and the router's choice come to the host together, behind the work that
        // made them, and the token waits once.
        let x_card = xs.contiguous()?;
        let ids_card = ids.to_dtype(crate::tensor::DType::U32)?.contiguous()?;
        let w_card = weights.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
        cd.0.context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        for (dst, src, bytes) in [
            (
                h_x.as_mut_ptr() as *mut std::ffi::c_void,
                address_of::<f32>(&x_card)?,
                4 * hidden,
            ),
            (
                h_ids.as_mut_ptr() as *mut std::ffi::c_void,
                address_of::<u32>(&ids_card)?,
                4 * k,
            ),
            (
                h_w.as_mut_ptr() as *mut std::ffi::c_void,
                address_of::<f32>(&w_card)?,
                4 * k,
            ),
        ] {
            driver("choice to host", unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(dst, src, bytes, stream.cu_stream())
            })?;
        }
        let prof = crate::inference::offload::stage_prof_enabled();
        let t_wait = prof.then(std::time::Instant::now);
        stream
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("choice to host: {e:?}")))?;
        if let Some(t) = t_wait {
            crate::inference::offload::prof_record(
                "tier wait for the choice",
                t.elapsed().as_nanos() as u64,
            );
        }
        laps.mark("tier token: choice wait");
        let (ids_h, w_h): (Vec<u32>, Vec<f32>) = (h_ids.to_vec(), h_w.to_vec());
        // Partition by what the card holds now, and count the use.
        let mut slot_ids = vec![0u32; k];
        let mut hit_weights = vec![0f32; k];
        let (mut miss_ids, mut miss_weights) = (Vec::new(), Vec::new());
        let mut in_use = vec![false; self.slots()];
        let mut is_hit = vec![false; k];
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for (i, (&e, &w)) in ids_h.iter().zip(&w_h).enumerate() {
                st.heat[e as usize] += 1.0;
                st.counts[e as usize] += 1;
                match st.slot_of[e as usize] {
                    s if s >= 0 => {
                        slot_ids[i] = s as u32;
                        hit_weights[i] = w;
                        in_use[s as usize] = true;
                        is_hit[i] = true;
                    }
                    _ => {
                        miss_ids.push(e);
                        miss_weights.push(w);
                    }
                }
            }
            st.tokens_routed(1);
        }
        let hits = k - miss_ids.len();
        let routed = self.routed.fetch_add(k as u64, Ordering::Relaxed) + k as u64;
        self.hits.fetch_add(hits as u64, Ordering::Relaxed);
        // The leader says how the cache is doing, once per window of tokens as wide as the
        // experts it chooses among.
        let window = (self.n_experts * k) as u64;
        if self.leader.load(Ordering::Relaxed) && routed % window < k as u64 {
            let (_, h, p) = self.counters();
            tracing::info!(
                "expert tier layer sample: {:.1}% of {} routed experts on the card, {} copies, {} to transit, {:.0} us copy {:.0} us host per expert, {} us a token in the layer",
                100.0 * h as f64 / routed as f64,
                routed,
                p,
                self.uploaded.load(Ordering::Relaxed),
                f64::from_bits(self.upload_us.load(Ordering::Relaxed)),
                f64::from_bits(self.host_us.load(Ordering::Relaxed)),
                self.decode_us.load(Ordering::Relaxed),
            );
        }
        // The misses the transit room runs, and those the host does.
        let to_transit = self.transit_share(miss_ids.len());
        let (transit_ids, host_ids) = miss_ids.split_at(to_transit);
        let (transit_w, host_w) = miss_weights.split_at(to_transit);
        laps.mark("tier token: partition");

        // The card's share is launched first, and runs while the host does its own. A routed pair
        // whose expert is missing reads a slot that is in use and weighs zero.
        let card_out = if hits > 0 {
            let filler = (0..k)
                .find(|&i| is_hit[i])
                .map(|i| slot_ids[i])
                .unwrap_or(0);
            for i in 0..k {
                if !is_hit[i] {
                    slot_ids[i] = filler;
                }
            }
            h_slots.copy_from_slice(&slot_ids);
            h_hw.copy_from_slice(&hit_weights);
            for (dst, src, bytes) in [
                (
                    address_of::<u32>(&scratch.slots)?,
                    h_slots.as_ptr() as *const std::ffi::c_void,
                    4 * k,
                ),
                (
                    address_of::<f32>(&scratch.weights)?,
                    h_hw.as_ptr() as *const std::ffi::c_void,
                    4 * k,
                ),
            ] {
                driver("slots to card", unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, src, bytes, stream.cu_stream())
                })?;
            }
            let routing = ExpertRouting {
                slots: moe.decode_slots.clone().expect("a tier layer is on a card"),
                experts: scratch.slots.clone(),
                topk: k,
            };
            // The slots past the base ones are in the extra stacks: one launch reads both.
            let extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
            let more = extra.as_ref();
            let h = routing.gate_up_silu_over(
                xs,
                &self.card[0],
                &self.card[1],
                more.map(|m| (&m[0], &m[1])),
            )?;
            Some(routing.down_reduce_over(
                &h,
                &self.card[2],
                more.map(|m| &m[2]),
                &scratch.weights,
                1,
            )?)
        } else {
            None
        };
        laps.mark("tier token: card launch");
        // The copies into the transit room start now, and the host's share is computed while they
        // run.
        let transit_landed = if transit_ids.is_empty() {
            false
        } else {
            self.copy_to_transit(scratch, transit_ids)?;
            true
        };
        laps.mark("tier token: transit copy");
        let t_host = std::time::Instant::now();
        let host_out = if host_ids.is_empty() {
            false
        } else {
            let m = host_ids.len();
            let x = Tensor::from_vec(h_x.to_vec(), (1, hidden), &host)?;
            let routing =
                ExpertRouting::group(&Tensor::from_vec(host_ids.to_vec(), (1, m), &host)?, m)?;
            let h = routing.host_gate_up_silu(&x, &self.host[0], &self.host[1])?;
            let w = Tensor::from_vec(host_w.to_vec(), (1, m), &host)?;
            let product = routing.host_down_reduce(&h, &self.host[2], &w, 1, None)?;
            h_out.copy_from_slice(&product.flatten_all()?.to_vec1::<f32>()?);
            driver("product to card", unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    address_of::<f32>(&scratch.product)?,
                    h_out.as_ptr() as *const std::ffi::c_void,
                    4 * hidden,
                    stream.cu_stream(),
                )
            })?;
            blend(
                &self.host_us,
                t_host.elapsed().as_secs_f64() * 1e6 / m as f64,
            );
            true
        };
        if prof && host_out {
            crate::inference::offload::prof_record(
                "tier host experts",
                t_host.elapsed().as_nanos() as u64,
            );
        }
        laps.mark("tier token: host experts");
        // The experts in the transit room run once the copies have landed.
        let transit_out = if transit_landed {
            let stage = self.stage.as_ref().expect("a copy to transit has a stage");
            // The experts sit in the room in the order they were copied; the pairs left over
            // read the first with no weight.
            for i in 0..k {
                h_tids[i] = if i < transit_ids.len() { i as u32 } else { 0 };
                h_tw[i] = transit_w.get(i).copied().unwrap_or(0.0);
            }
            for (dst, src, bytes) in [
                (
                    address_of::<u32>(&scratch.transit_ids)?,
                    h_tids.as_ptr() as *const std::ffi::c_void,
                    4 * k,
                ),
                (
                    address_of::<f32>(&scratch.transit_weights)?,
                    h_tw.as_ptr() as *const std::ffi::c_void,
                    4 * k,
                ),
            ] {
                driver("transit to card", unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, src, bytes, stream.cu_stream())
                })?;
            }
            stream
                .wait(&scratch.upload_end)
                .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
            let routing = ExpertRouting {
                slots: moe.decode_slots.clone().expect("a tier layer is on a card"),
                experts: scratch.transit_ids.clone(),
                topk: k,
            };
            let h = routing.gate_up_silu(xs, &stage.compact[0], &stage.compact[1])?;
            Some(routing.down_reduce(
                &h,
                &stage.compact[2],
                &scratch.transit_weights,
                1,
                false,
                None,
                None,
            )?)
        } else {
            None
        };
        laps.mark("tier token: transit launch");
        // Fresh tensors: the buffers on the card are this layer's, written again by the next token.
        let mut out: Option<Tensor> = None;
        for part in [card_out, transit_out] {
            out = match (out, part) {
                (Some(a), Some(b)) => Some(
                    (a.to_dtype(crate::tensor::DType::F32)?
                        + b.to_dtype(crate::tensor::DType::F32)?)?,
                ),
                (a, b) => a.or(b),
            };
        }
        let out = match (out, host_out) {
            (Some(c), true) => (c.to_dtype(crate::tensor::DType::F32)? + &scratch.product)?,
            (Some(c), false) => c,
            (None, true) => scratch.product.affine(1.0, 0.0)?,
            (None, false) => unreachable!("a token routes to at least one expert"),
        };
        drop(guard);

        // An expert that came to the card for this token stays in a slot where it is hotter than
        // what the slot holds, without a second copy over the link; the host's misses go to the
        // worker.
        self.keep_in_transit(transit_ids, false, &mut in_use, &stream)?;
        self.promote(host_ids, &in_use);
        laps.mark("tier token: finish");
        Ok(out)
    }

    /// The expert blocks of several tokens at once, as a draft verify routes them: `xs`
    /// `[rows, hidden]` and the router's choices `ids`, `weights` `[rows, k]`, all on the card.
    /// One wait for the choices serves every row; the experts some row misses are copied into
    /// the transit room once whichever rows use them; the card runs its share of every row in
    /// one launch pair, the transit room's in another, and the host the rest row by row.
    /// Returns `[rows, hidden]` on the card.
    pub(super) fn decode_rows(
        self: &Arc<Self>,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let k = moe.n_experts_used;
        let host = Device::Cpu;
        let dev = xs.device().clone();
        let (rows, hidden) = xs.dims2()?;
        let cd = dev.as_cuda_device()?;
        let stream = cd.0.stream().clone();
        let mut guard = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(Scratch::new(&dev, hidden, k)?);
        }
        let scratch = guard.as_ref().expect("made above");
        let mut laps = Laps::new();
        let carried = scratch.in_flight.swap(0, Ordering::Relaxed);
        if carried > 0 {
            if let Ok(ms) = scratch.upload_start.elapsed_ms(&scratch.upload_end) {
                blend(&self.upload_us, ms as f64 * 1e3 / carried as f64);
            }
        }
        // The choices of every row come to the host together, behind the work that made them.
        let ids_h: Vec<u32> = ids
            .to_dtype(crate::tensor::DType::U32)?
            .to_device(&host)?
            .flatten_all()?
            .to_vec1()?;
        let w_h: Vec<f32> = weights
            .to_dtype(crate::tensor::DType::F32)?
            .to_device(&host)?
            .flatten_all()?
            .to_vec1()?;
        laps.mark("tier rows: choice wait");
        // Partition every routed pair by what the card holds now, and count the use.
        let mut slot_ids = vec![0u32; rows * k];
        let mut hit_weights = vec![0f32; rows * k];
        let mut is_hit = vec![false; rows * k];
        let mut in_use = vec![false; self.slots()];
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for (p, (&e, &w)) in ids_h.iter().zip(&w_h).enumerate() {
                st.heat[e as usize] += 1.0;
                st.counts[e as usize] += 1;
                let s = st.slot_of[e as usize];
                if s >= 0 {
                    slot_ids[p] = s as u32;
                    hit_weights[p] = w;
                    in_use[s as usize] = true;
                    is_hit[p] = true;
                }
            }
            st.tokens_routed(rows);
        }
        let pairs = rows * k;
        let hits = is_hit.iter().filter(|&&h| h).count();
        self.routed.fetch_add(pairs as u64, Ordering::Relaxed);
        self.hits.fetch_add(hits as u64, Ordering::Relaxed);
        // The experts some pair misses, the most used first: the transit room takes as many as
        // the copy keeps pace with, the host computes the pairs of the others.
        let mut users: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for p in 0..pairs {
            if !is_hit[p] {
                *users.entry(ids_h[p]).or_insert(0) += 1;
            }
        }
        let mut missing: Vec<u32> = users.keys().copied().collect();
        missing.sort_by(|a, b| users[b].cmp(&users[a]).then(a.cmp(b)));
        let to_transit = self.transit_share(missing.len());
        let transit_experts: Vec<u32> = missing[..to_transit].to_vec();
        laps.mark("tier rows: partition");

        // The card's share is launched first, and runs while the host does its own.
        let card_out = if hits > 0 {
            let filler = (0..pairs)
                .find(|&p| is_hit[p])
                .map(|p| slot_ids[p])
                .unwrap_or(0);
            for p in 0..pairs {
                if !is_hit[p] {
                    slot_ids[p] = filler;
                }
            }
            // The slots past the base ones are in the extra stacks, launched apart. A pair a stack
            // does not serve reads a slot of that stack which this call reads, and weighs zero.
            let first_extra = self.base_slots as u32;
            let hit_slots: Vec<(usize, u32)> = (0..pairs)
                .filter(|&p| is_hit[p])
                .map(|p| (p, slot_ids[p]))
                .collect();
            let base_filler = hit_slots.iter().find(|h| h.1 < first_extra).map(|h| h.1);
            let extra_filler = hit_slots
                .iter()
                .find(|h| h.1 >= first_extra)
                .map(|h| h.1 - first_extra);
            let (base_hits, extra_hits) = (
                hit_slots.iter().filter(|h| h.1 < first_extra).count(),
                hit_slots.iter().filter(|h| h.1 >= first_extra).count(),
            );
            slot_ids = vec![base_filler.unwrap_or(0); pairs];
            let mut extra_ids = vec![extra_filler.unwrap_or(0); pairs];
            let mut extra_w = vec![0f32; pairs];
            let mut base_w = vec![0f32; pairs];
            for &(p, s) in &hit_slots {
                if s < first_extra {
                    slot_ids[p] = s;
                    base_w[p] = hit_weights[p];
                } else {
                    extra_ids[p] = s - first_extra;
                    extra_w[p] = hit_weights[p];
                }
            }
            hit_weights = base_w;
            let base_out = if base_hits > 0 {
                let choice = Tensor::from_vec(slot_ids, (rows, k), &host)?.to_device(&dev)?;
                let w = Tensor::from_vec(hit_weights, (rows, k), &host)?.to_device(&dev)?;
                Some(moe.forward_cuda(
                    xs,
                    &choice,
                    &w,
                    rows,
                    false,
                    None,
                    None,
                    Some(&self.card),
                )?)
            } else {
                None
            };
            let extra_out = if extra_hits > 0 {
                let extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
                let stacks = extra
                    .as_ref()
                    .expect("an extra slot that holds an expert is not lent");
                let choice = Tensor::from_vec(extra_ids, (rows, k), &host)?.to_device(&dev)?;
                let w = Tensor::from_vec(extra_w, (rows, k), &host)?.to_device(&dev)?;
                Some(moe.forward_cuda(xs, &choice, &w, rows, false, None, None, Some(stacks))?)
            } else {
                None
            };
            match (base_out, extra_out) {
                (Some(a), Some(b)) => Some(
                    (a.to_dtype(crate::tensor::DType::F32)?
                        + b.to_dtype(crate::tensor::DType::F32)?)?,
                ),
                (a, b) => a.or(b),
            }
        } else {
            None
        };

        laps.mark("tier rows: card launch");
        // The copies into the transit room start now, and the host's share is computed while
        // they run.
        if !transit_experts.is_empty() {
            self.copy_to_transit(scratch, &transit_experts)?;
        }
        laps.mark("tier rows: transit copy");
        let t_host = std::time::Instant::now();
        // The host computes each expert it was given once, for every row that routes to it, so
        // the blocks are read from memory once however many rows use them.
        let host_experts: Vec<u32> = missing[to_transit..].to_vec();
        let host_out = if host_experts.is_empty() {
            None
        } else {
            let mut product = vec![0f32; rows * hidden];
            let x_host: Vec<f32> = xs
                .to_device(&host)?
                .to_dtype(crate::tensor::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            // Every pair the host takes is one row of one grouped call, whatever its expert: the
            // pool is woken twice however many experts the rows need.
            let mine: Vec<usize> = (0..pairs)
                .filter(|&p| !is_hit[p] && host_experts.contains(&ids_h[p]))
                .collect();
            let m = mine.len();
            let mut x = Vec::with_capacity(m * hidden);
            for &p in &mine {
                let r = p / k;
                x.extend_from_slice(&x_host[r * hidden..(r + 1) * hidden]);
            }
            let x = Tensor::from_vec(x, (m, hidden), &host)?;
            let choice = Tensor::from_vec(mine.iter().map(|&p| ids_h[p]).collect(), (m, 1), &host)?;
            let routing = ExpertRouting::group(&choice, 1)?;
            let h = routing.host_gate_up_silu(&x, &self.host[0], &self.host[1])?;
            let w = Tensor::from_vec(mine.iter().map(|&p| w_h[p]).collect(), (m, 1), &host)?;
            let out = routing
                .host_down_reduce(&h, &self.host[2], &w, m, None)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (j, &p) in mine.iter().enumerate() {
                let r = p / k;
                for (acc, v) in product[r * hidden..(r + 1) * hidden]
                    .iter_mut()
                    .zip(&out[j * hidden..(j + 1) * hidden])
                {
                    *acc += v;
                }
            }
            blend(
                &self.host_us,
                t_host.elapsed().as_secs_f64() * 1e6 / host_experts.len() as f64,
            );
            Some(Tensor::from_vec(product, (rows, hidden), &host)?.to_device(&dev)?)
        };
        laps.mark("tier rows: host experts");

        // The experts in the transit room run once the copies have landed.
        let transit_out = if transit_experts.is_empty() {
            None
        } else {
            let stage = self.stage.as_ref().expect("a copy to transit has a stage");
            stream
                .wait(&scratch.upload_end)
                .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
            let mut t_ids = vec![0u32; pairs];
            let mut t_w = vec![0f32; pairs];
            for p in 0..pairs {
                if is_hit[p] {
                    continue;
                }
                if let Some(at) = transit_experts.iter().position(|&e| e == ids_h[p]) {
                    t_ids[p] = at as u32;
                    t_w[p] = w_h[p];
                }
            }
            let choice = Tensor::from_vec(t_ids, (rows, k), &host)?.to_device(&dev)?;
            let w = Tensor::from_vec(t_w, (rows, k), &host)?.to_device(&dev)?;
            Some(moe.forward_cuda(
                xs,
                &choice,
                &w,
                rows,
                false,
                None,
                None,
                Some(&stage.compact),
            )?)
        };
        laps.mark("tier rows: transit launch");
        let mut out: Option<Tensor> = None;
        for part in [card_out, transit_out, host_out] {
            out = match (out, part) {
                (Some(a), Some(b)) => Some(
                    (a.to_dtype(crate::tensor::DType::F32)?
                        + b.to_dtype(crate::tensor::DType::F32)?)?,
                ),
                (a, b) => a.or(b),
            };
        }
        let out = out.expect("a token routes to at least one expert");
        drop(guard);
        self.keep_in_transit(&transit_experts, false, &mut in_use, &stream)?;
        self.promote_up_to(&host_experts, &in_use, rows.max(ADAPT_SWAPS_A_REQUEST));
        laps.mark("tier rows: finish");
        Ok(out)
    }

    /// A prompt's routing through this layer, `ids` `[rows, k]`, warms the cache before the first
    /// token is decoded: the experts the prompt used are ranked into the heat, below what a
    /// decoded token's use earns, and those of them that are now in the transit room, hottest first,
    /// take the slots of the colder ones by copies on the card. The layer's whole stack is in the
    /// room while a wide batch runs through it, so the copies cost no transfer over the link.
    pub(super) fn keep_hottest_of_prompt(
        &self,
        ids: &Tensor,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        if self.stage.is_none() || self.resident {
            return Ok(());
        }
        // A prompt that holds the layer's stacks goes through it in several chunks: their routing
        // is ranked together when the hold ends, one read of the ids back to the host for the
        // layer where a read for each chunk would drain the card each time.
        if self
            .stage
            .as_ref()
            .is_some_and(|s| s.held.load(Ordering::Acquire))
        {
            let mut pending = self.prompt_ids.lock().unwrap_or_else(|e| e.into_inner());
            match pending.as_mut() {
                Some((_, all)) => all.push(ids.clone()),
                None => *pending = Some((stream.clone(), vec![ids.clone()])),
            }
            return Ok(());
        }
        self.keep_hottest_now(ids, stream)
    }

    /// Rank the routing `ids` of a prompt into the cache from the experts now in the transit room.
    fn keep_hottest_now(
        &self,
        ids: &Tensor,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let ids_h: Vec<u32> = ids
            .to_dtype(crate::tensor::DType::U32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;
        let mut count = vec![0f32; self.n_experts];
        for &e in &ids_h {
            count[e as usize] += 1.0;
        }
        // The prompt ranks the experts but must not outweigh what the decode finds it needs: the
        // hottest expert of the batch starts at the heat of one use, and a heat the decode has
        // already earned is kept.
        let top = count.iter().copied().fold(0f32, f32::max).max(1.0);
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for (h, c) in st.heat.iter_mut().zip(&count) {
                *h = h.max(c / top);
            }
        }
        // Only experts the cache lacks can take a slot: one it holds is already in the cache.
        let present: Vec<u32> = {
            let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            (0..self.n_experts as u32)
                .filter(|&e| count[e as usize] > 0.0 && st.slot_of[e as usize] < 0)
                .collect()
        };
        let mut in_use = vec![false; self.slots()];
        self.keep_in_transit(&present, true, &mut in_use, stream)
    }

    /// Move the experts of `experts` that are in the transit room, hottest first, into the slots
    /// that hold something colder and are not read by this token, as copies on the card behind the
    /// token's own work: a kernel that reads a slot later is queued after its copy. `in_use` marks
    /// the slots this token reads, and is extended with the slots filled here. The experts are in
    /// the order they were copied in the transit room, or each at its own place in the room of a
    /// wide batch (`by_id`).
    fn keep_in_transit(
        &self,
        experts: &[u32],
        by_id: bool,
        in_use: &mut [bool],
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        let full;
        let arenas = if by_id {
            full = stage.stager.full.lock().unwrap_or_else(|e| e.into_inner());
            match full.as_ref() {
                Some(f) => &f[stage.buffer].arenas,
                None => return Ok(()),
            }
        } else {
            &stage.stager.buffers[stage.buffer].arenas
        };
        let mut order: Vec<(u32, usize)> = experts
            .iter()
            .enumerate()
            .map(|(j, &e)| (e, if by_id { e as usize } else { j }))
            .collect();
        let mut moves = Vec::new();
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            order.sort_by(|a, b| st.heat[b.0 as usize].total_cmp(&st.heat[a.0 as usize]));
            for &(e, at) in &order {
                let victim = (0..st.owner.len())
                    .filter(|&slot| !in_use[slot] && st.owner[slot] >= -1)
                    .min_by(|&a, &b| slot_heat(&st, a).total_cmp(&slot_heat(&st, b)));
                let Some(victim) = victim else {
                    break;
                };
                if st.heat[e as usize] <= slot_heat(&st, victim) {
                    break;
                }
                let old = st.owner[victim];
                if old >= 0 {
                    st.set_slot(old as usize, -1);
                }
                st.owner[victim] = e as i32;
                st.set_slot(e as usize, victim as i32);
                in_use[victim] = true;
                moves.push((at, victim));
            }
        }
        for (at, slot) in moves.iter().copied() {
            for i in 0..3 {
                let eb = self.expert_bytes[i];
                let arena = &arenas[i];
                let from = arena.device_ptr(arena.stream()).0 + (at * eb) as u64;
                let to = self.slot_addr(i, slot)?;
                driver("transit to slot", unsafe {
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(to, from, eb, stream.cu_stream())
                })?;
            }
            self.promotions.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// How many of `missing` experts are copied into the card's transit room for the token, the
    /// rest being computed on the host: the split at which the two finish together, from the
    /// time a copy and a host computation have taken on this layer. The host does at least one
    /// of several until its time is known.
    fn transit_share(&self, missing: usize) -> usize {
        if missing == 0 || self.stage.is_none() {
            return 0;
        }
        let room = self.stage.as_ref().map_or(0, |s| s.stager.transit);
        let eighths = self.transit_eighths.load(Ordering::Relaxed);
        if eighths > 0 {
            return ((missing as u64 * eighths + 4) / 8).min(room as u64) as usize;
        }
        let host = f64::from_bits(self.host_us.load(Ordering::Relaxed));
        let mut copy = f64::from_bits(self.upload_us.load(Ordering::Relaxed));
        if copy == 0.0 {
            let rate = self.stage.as_ref().map_or(0.0, |s| s.stager.bandwidth);
            if rate <= 0.0 {
                return 0;
            }
            copy = self.expert_bytes.iter().sum::<usize>() as f64 / rate;
        }
        if host == 0.0 {
            return (if missing > 1 { missing - 1 } else { missing }).min(room);
        }
        // The paces measured enough, the share is fixed: the copies and the products finish
        // together when the room takes host / (host + copy) of the experts.
        if self.paced.load(Ordering::Relaxed) >= TRANSIT_PACE_SAMPLES {
            let share = (8.0 * host / (host + copy)).round().clamp(1.0, 7.0) as u64;
            self.transit_eighths.store(share, Ordering::Relaxed);
            return ((missing as u64 * share + 4) / 8).min(room as u64) as usize;
        }
        let finish = |n: usize| (n as f64 * copy).max((missing - n) as f64 * host);
        (0..=missing.min(room))
            .min_by(|&a, &b| finish(a).total_cmp(&finish(b)))
            .unwrap_or(0)
    }

    /// Queue the copies of `experts` from the host into the transit room, one after the other
    /// from its start, and the events around them.
    fn copy_to_transit(&self, scratch: &Scratch, experts: &[u32]) -> Result<()> {
        self.copy_to_transit_between(&scratch.upload_start, &scratch.upload_end, experts)?;
        scratch.in_flight.store(experts.len(), Ordering::Relaxed);
        Ok(())
    }

    /// [`Self::copy_to_transit`] with the events that bracket the copies.
    pub(super) fn copy_to_transit_between(
        &self,
        start: &cudarc::driver::CudaEvent,
        end: &cudarc::driver::CudaEvent,
        experts: &[u32],
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let stage = self.stage.as_ref().expect("a copy to transit has a stage");
        let copy = &stage.stager.copy;
        let buffer = &stage.stager.buffers[stage.buffer];
        // The choice was waited for, so the layers that read this buffer before are done.
        start
            .record(copy)
            .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?;
        for i in 0..3 {
            let eb = self.expert_bytes[i];
            let arena = &buffer.arenas[i];
            let base = arena.device_ptr(arena.stream()).0;
            let data = self.host[i].native_qtensor().data();
            for (at, &e) in experts.iter().enumerate() {
                let e = e as usize;
                self.upload(i, data, e * eb, eb, base + (at * eb) as u64, copy)?;
            }
        }
        end.record(copy)
            .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?;
        self.uploaded
            .fetch_add(experts.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Of `missing` experts, how many the transit room takes for this request, the rest being
    /// the host's: [`Self::transit_share`] where the layer has a room.
    pub(super) fn transit_share_of(&self, missing: usize) -> usize {
        self.transit_share(missing)
    }

    /// The transit room's stacks, where the layer has one.
    pub(super) fn transit_stacks(&self) -> Option<&[QTensor; 3]> {
        self.stage.as_ref().map(|s| &s.compact)
    }

    /// Note what one copy into the transit room took, in microseconds.
    pub(super) fn note_upload_us(&self, us: f64) {
        blend(&self.upload_us, us);
        self.paced.fetch_add(1, Ordering::Relaxed);
    }

    /// Hand the worker the hottest missing experts that earn a slot over the coldest slots this
    /// token did not read. The copy runs beside the tokens that follow: an expert is routed to
    /// the card only once it has landed.
    fn promote(self: &Arc<Self>, missing: &[u32], in_use: &[bool]) {
        self.promote_up_to(missing, in_use, ADAPT_SWAPS_A_REQUEST);
    }

    /// [`Self::promote`] for up to `most` experts at once: the hottest missing ones, each hotter
    /// than the coldest slot left, copied in one job. A request of several rows misses as many
    /// experts as rows, and the cache learns at that pace.
    pub(super) fn promote_up_to(self: &Arc<Self>, missing: &[u32], in_use: &[bool], most: usize) {
        if most == 0 || self.promoting.swap(true, Ordering::AcqRel) {
            return;
        }
        let chosen: Vec<(usize, usize)> = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let mut hot: Vec<u32> = missing
                .iter()
                .copied()
                .filter(|&e| st.heat[e as usize] >= ADAPT_MIN_HEAT)
                .collect();
            hot.sort_unstable();
            hot.dedup();
            hot.sort_by(|&a, &b| st.heat[b as usize].total_cmp(&st.heat[a as usize]));
            let mut victims: Vec<usize> = (0..st.owner.len())
                .filter(|&s| !in_use[s] && st.owner[s] >= -1)
                .collect();
            victims.sort_by(|&a, &b| slot_heat(&st, a).total_cmp(&slot_heat(&st, b)));
            // The hottest candidate against the coldest slot, then the next pair: the gains only
            // fall from there, so the first pair short of the margin ends it.
            let mut pairs = Vec::new();
            for (&expert, &victim) in hot.iter().zip(&victims).take(most) {
                if st.heat[expert as usize] < slot_heat(&st, victim) + ADAPT_MARGIN {
                    break;
                }
                let old = st.owner[victim];
                if old >= 0 {
                    st.set_slot(old as usize, -1);
                }
                st.owner[victim] = -2;
                pairs.push((expert as usize, victim));
            }
            pairs
        };
        if chosen.is_empty() {
            self.promoting.store(false, Ordering::Release);
            return;
        }
        let tier = self.clone();
        copier(self.dev.as_cuda_device().map_or(0, |d| d.0.ordinal()))
            .send(Box::new(move || {
                let landed = (|| -> Result<()> {
                    for &(expert, slot) in &chosen {
                        tier.copy_in_async(expert, slot)?;
                    }
                    tier.copy
                        .synchronize()
                        .map_err(|e| crate::tensor::Error::msg(format!("slot copy: {e:?}")))
                })();
                let mut st = tier.state.lock().unwrap_or_else(|e| e.into_inner());
                for &(expert, slot) in &chosen {
                    match &landed {
                        Ok(()) => {
                            st.owner[slot] = expert as i32;
                            st.set_slot(expert, slot as i32);
                            tier.promotions.fetch_add(1, Ordering::Relaxed);
                            graph::note_promotion();
                        }
                        Err(e) => {
                            tracing::warn!(
                                "expert tier: copying expert {expert} to slot {slot}: {e}"
                            );
                            st.owner[slot] = -1;
                        }
                    }
                }
                drop(st);
                tier.promoting.store(false, Ordering::Release);
            }))
            .ok();
    }

    /// Expert `expert` of the host stacks into slot `slot` of the card's, queued on the copy
    /// stream: the caller waits for it.
    fn copy_in_async(&self, expert: usize, slot: usize) -> Result<()> {
        let cd = self.dev.as_cuda_device()?;
        cd.0.context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        for i in 0..3 {
            let eb = self.expert_bytes[i];
            let data = self.host[i].native_qtensor().data();
            let dst = self.slot_addr(i, slot)?;
            self.upload(i, data, expert * eb, eb, dst, &self.copy)?;
        }
        Ok(())
    }
}

/// The thread of a card that runs the copies into its caches, one at a time.
fn copier(ordinal: usize) -> std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>> {
    use std::collections::HashMap;
    use std::sync::{mpsc, OnceLock};
    static WORKERS: OnceLock<Mutex<HashMap<usize, mpsc::Sender<Box<dyn FnOnce() + Send>>>>> =
        OnceLock::new();
    let mut workers = WORKERS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    workers
        .entry(ordinal)
        .or_insert_with(|| {
            let (tx, rx) = mpsc::channel::<Box<dyn FnOnce() + Send>>();
            std::thread::Builder::new()
                .name(format!("expert-copy-{ordinal}"))
                .spawn(move || {
                    for job in rx {
                        job();
                    }
                })
                .expect("a thread for the expert copies");
            tx
        })
        .clone()
}

/// Fold `sample` into the running figure in `cell` (an `f64`'s bits), which a first sample sets.
fn blend(cell: &AtomicU64, sample: f64) {
    let old = f64::from_bits(cell.load(Ordering::Relaxed));
    let next = if old == 0.0 {
        sample
    } else {
        (old * 7.0 + sample) / 8.0
    };
    cell.store(next.to_bits(), Ordering::Relaxed);
}

/// CUDA driver result as a tensor error.
fn driver(what: &str, r: cudarc::driver::sys::cudaError_enum) -> Result<()> {
    if r == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(crate::tensor::Error::msg(format!("{what}: {r:?}")))
    }
}

impl CardStager {
    /// Room on `dev` for experts to pass through: two sets of arenas for the transit room, each
    /// holding the experts the link can carry while the host computes the rest of a token's
    /// misses, of the `experts` a layer has, `widest` bytes per stack for the widest layer. The
    /// rooms a wide batch fills with whole stacks are made when a prefill runs, see
    /// [`Self::ensure_full`].
    pub fn new(dev: &Device, widest: [usize; 3], experts: usize) -> Result<Arc<Self>> {
        let cd = dev.as_cuda_device()?;
        let copy =
            cd.0.context()
                .new_stream()
                .map_err(|e| crate::tensor::Error::msg(format!("copy stream: {e:?}")))?;
        let experts = experts.max(1);
        let per_expert = widest.map(|w| w.div_ceil(experts));
        let bandwidth = {
            let bytes = widest[0].min(1 << 26);
            let probe = QCudaStorage::arena(&cd, bytes)
                .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
            measure_upload_bandwidth(dev, &probe, bytes)?
        };
        let transit = transit_experts(experts, bandwidth, host_read_bandwidth(widest[0]));
        let buffer = || -> Result<StageBuffer> {
            let arena = |i: usize| {
                QCudaStorage::arena(&cd, per_expert[i] * transit)
                    .map_err(|e| crate::tensor::Error::msg(e.to_string()))
            };
            Ok(StageBuffer {
                arenas: [arena(0)?, arena(1)?, arena(2)?],
            })
        };
        let buffers = [buffer()?, buffer()?];
        Ok(Arc::new(Self {
            dev: dev.clone(),
            bandwidth,
            copy,
            widest,
            transit,
            buffers,
            full: Mutex::new(None),
        }))
    }

    /// Bytes the transit room takes on the card.
    pub fn transit_bytes(&self) -> u64 {
        self.buffers
            .iter()
            .flat_map(|b| b.arenas.iter())
            .map(|a| a.len() as u64)
            .sum()
    }

    /// Bytes the rooms of a wide batch take on the card.
    pub fn full_bytes(&self) -> u64 {
        2 * self.widest.iter().sum::<usize>() as u64
    }

    /// Make the rooms a wide batch fills with a layer's whole stacks, from the memory the caches
    /// lend. Nothing is done while they exist.
    pub fn ensure_full(&self) -> Result<()> {
        let mut full = self.full.lock().unwrap_or_else(|e| e.into_inner());
        if full.is_some() {
            return Ok(());
        }
        let cd = self.dev.as_cuda_device()?;
        let one = || -> Result<FullBuffer> {
            let arena = |i: usize| {
                QCudaStorage::arena_in_lent_pool(&cd, self.widest[i])
                    .map_err(|e| crate::tensor::Error::msg(e.to_string()))
            };
            Ok(FullBuffer {
                arenas: [arena(0)?, arena(1)?, arena(2)?],
                free: Mutex::new(None),
            })
        };
        *full = Some([one()?, one()?]);
        // The arenas were zeroed in the order of the compute stream: the copies into them wait
        // for that.
        cd.0.stream()
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("compute stream: {e:?}")))?;
        Ok(())
    }

    /// Give the card the rooms of a wide batch back, once the copies into them have landed.
    pub fn drop_full(&self) {
        let _ = self.copy.synchronize();
        let dropped = self
            .full
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .is_some();
        if dropped {
            if let Ok(cd) = self.dev.as_cuda_device() {
                let _ = cd.0.stream().synchronize();
                crate::tensor::quantized::trim_lent_pool(cd.0.ordinal());
            }
        }
    }
}

/// How many of a layer's `experts` the transit room holds: a token's misses are split between
/// the link and the host so that both finish together, which sends the link the share
/// `link / (link + host)` of them, the two rates being bytes per microsecond.
fn transit_experts(experts: usize, link: f64, host: f64) -> usize {
    if link <= 0.0 || host <= 0.0 {
        return experts;
    }
    let share = link / (link + host);
    ((experts as f64 * share).ceil() as usize).clamp(1, experts)
}

/// Bytes per microsecond the host's threads read a stack of `bytes` bytes at, as the host
/// computation of an expert does: the best of a few reads of a block of zeros that is not in any
/// cache. Zero when it could not be measured.
fn host_read_bandwidth(bytes: usize) -> f64 {
    use rayon::prelude::*;
    let len = bytes.clamp(1 << 20, 1 << 28);
    let block = vec![1u8; len];
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let sum: u64 = block
            .par_chunks(1 << 16)
            .map(|c| c.iter().map(|&b| b as u64).sum::<u64>())
            .sum();
        let us = t.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(sum);
        best = best.min(us);
    }
    if best.is_finite() && best > 0.0 {
        len as f64 / best
    } else {
        0.0
    }
}

/// Bytes per microsecond that page-locked host memory reaches the card at: a copy into `arena`,
/// the best of a few.
fn measure_upload_bandwidth(
    dev: &Device,
    arena: &Arc<cudarc::driver::CudaSlice<u8>>,
    arena_bytes: usize,
) -> Result<f64> {
    use cudarc::driver::DevicePtr;
    let cd = dev.as_cuda_device()?;
    cd.0.context()
        .bind_to_thread()
        .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
    let bytes = arena_bytes.min(64 << 20);
    let host = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
        .map_err(|e| crate::tensor::Error::msg(format!("pinned probe: {e}")))?;
    let dst = arena.device_ptr(arena.stream()).0;
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let r = unsafe { cudarc::driver::sys::cuMemcpyHtoD_v2(dst, host, bytes) };
        let us = t.elapsed().as_secs_f64() * 1e6;
        if r == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            best = best.min(us);
        }
    }
    unsafe { cudarc::driver::result::free_host(host) }.ok();
    Ok(if best.is_finite() && best > 0.0 {
        bytes as f64 / best
    } else {
        0.0
    })
}

/// Page-lock `[ptr, ptr + len)` as far as whole pages allow, and give the locked range.
fn lock_pages(ptr: usize, len: usize) -> Option<(usize, usize)> {
    let page = crate::tensor::mapped::page_size()?;
    let lo = ptr.div_ceil(page) * page;
    let hi = (ptr + len) / page * page;
    if hi <= lo {
        return None;
    }
    // Read-only: the mapping is the file's.
    const READ_ONLY: u32 = 8;
    let r = unsafe {
        cudarc::driver::sys::cuMemHostRegister_v2(lo as *mut std::ffi::c_void, hi - lo, READ_ONLY)
    };
    (r == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS).then_some((lo, hi))
}

impl Stage {
    fn new(
        stager: &Arc<CardStager>,
        buffer: usize,
        host: &[Arc<QTensor>; 3],
        pin: bool,
    ) -> Result<Self> {
        let cd = stager.dev.as_cuda_device()?;
        cd.0.context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        let mk = |i: usize| -> Result<QTensor> {
            let (inner, len) = first_view(&host[i], stager.transit)?;
            let storage = QCudaStorage::sharing(
                stager.buffers[buffer].arenas[i].clone(),
                len,
                host[i].dtype(),
                &cd,
            );
            Ok(QTensor::over_card(inner, &stager.dev, storage))
        };
        let pinned = [0, 1, 2].map(|i| {
            if !pin {
                return None;
            }
            let data = host[i].native_qtensor().data();
            lock_pages(data.as_ptr() as usize, data.len())
        });
        Ok(Self {
            stager: stager.clone(),
            buffer,
            compact: [mk(0)?, mk(1)?, mk(2)?],
            full: Mutex::new(None),
            ready: Mutex::new(None),
            pinned,
            held: std::sync::atomic::AtomicBool::new(false),
            filled: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        for r in self.pinned.iter().flatten() {
            unsafe { cudarc::driver::sys::cuMemHostUnregister(r.0 as *mut std::ffi::c_void) };
        }
    }
}

impl LayerTier {
    /// Whether a batch of `rows` pays for the whole layer's experts to come to the card: when
    /// the rows' routed pairs outnumber the experts, nearly every expert is used.
    ///
    /// The rows run one by one through the cache where that is cheaper: the copy of what the card
    /// does not hold, at the rate the host reaches the card, against the rows at the pace tokens
    /// have been decoded here. Before any token was decoded the copy is taken.
    pub fn stream_pays(&self, rows: usize, used: usize) -> bool {
        let Some(stage) = self.stage.as_ref() else {
            return false;
        };
        if rows * used < self.n_experts
            || stage
                .full
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_none()
        {
            return false;
        }
        let per_token = self.decode_us.load(Ordering::Relaxed) as f64;
        let held = self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .slot_of
            .iter()
            .filter(|&&s| s >= 0)
            .count();
        let missing = (self.n_experts - held) * self.expert_bytes.iter().sum::<usize>();
        let copy_us = missing as f64 / stage.stager.bandwidth.max(1e-9);
        per_token == 0.0 || (rows as f64) * per_token > copy_us
    }

    /// Start the copies that fill this layer's buffer with every expert: the ones the card
    /// holds from the card, the others from the host's mapping. Nothing waits for them here.
    /// A fill already queued and not yet read stands.
    pub fn prefetch(&self) -> Result<()> {
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        // A prompt holding the stacks of this layer for its later chunks has them on the card
        // already: a second fill would carry the layer over the link again into the buffer the
        // chunks are reading.
        if stage.held.load(Ordering::Acquire) && stage.filled.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut ready = stage.ready.lock().unwrap_or_else(|e| e.into_inner());
        if ready.is_some() {
            return Ok(());
        }
        let cd = self.dev.as_cuda_device()?;
        cd.0.context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        let copy = &stage.stager.copy;
        let full = stage.stager.full.lock().unwrap_or_else(|e| e.into_inner());
        let Some(full) = full.as_ref() else {
            crate::tensor::bail!("a wide batch's fill has no room");
        };
        let buffer = &full[stage.buffer];
        if let Some(free) = buffer.free.lock().unwrap_or_else(|e| e.into_inner()).take() {
            copy.wait(&free)
                .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
        }
        let slot_of = self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .slot_of
            .clone();
        use cudarc::driver::DevicePtr;
        for i in 0..3 {
            let eb = self.expert_bytes[i];
            let arena = &buffer.arenas[i];
            let base = arena.device_ptr(arena.stream()).0;
            let data = self.host[i].native_qtensor().data();
            // The kernels read whole blocks past the end of a stack, against activations padded
            // with zeros: that tail is zero, as every blob's is, and not what the layer before
            // left in the arena - a scale in it that is not finite would poison the zeros.
            driver("stage tail", unsafe {
                cudarc::driver::sys::cuMemsetD8Async(
                    base + data.len() as u64,
                    0,
                    crate::tensor::quantized::BLOB_TAIL_PAD_BYTES,
                    copy.cu_stream(),
                )
            })?;
            let mut e = 0;
            while e < self.n_experts {
                if slot_of[e] >= 0 {
                    driver("slot to stage", unsafe {
                        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            base + (e * eb) as u64,
                            self.slot_addr(i, slot_of[e] as usize)?,
                            eb,
                            copy.cu_stream(),
                        )
                    })?;
                    e += 1;
                    continue;
                }
                // The run of experts the card does not hold, in one copy.
                let first = e;
                while e < self.n_experts && slot_of[e] < 0 {
                    e += 1;
                }
                self.upload(
                    i,
                    data,
                    first * eb,
                    (e - first) * eb,
                    base + (first * eb) as u64,
                    copy,
                )?;
            }
        }
        *ready = Some(
            copy.record_event(None)
                .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?,
        );
        Ok(())
    }

    /// `len` bytes of stack `i` from `start` to the card at `dst`: from the page-locked part
    /// of the range as it stands, the rest through the driver's own staging.
    fn upload(
        &self,
        i: usize,
        data: &[u8],
        start: usize,
        len: usize,
        dst: u64,
        copy: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let (a, b) = (
            data.as_ptr() as usize + start,
            data.as_ptr() as usize + start + len,
        );
        let pinned = self.stage.as_ref().and_then(|s| s.pinned[i]);
        let (lo, hi) = pinned.map_or((b, b), |(lo, hi)| (lo.clamp(a, b), hi.clamp(a, b)));
        let (lo, hi) = if lo < hi { (lo, hi) } else { (b, b) };
        for (from, to) in [(a, lo), (lo, hi), (hi, b)] {
            if to > from {
                driver("expert to stage", unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        dst + (from - a) as u64,
                        from as *const std::ffi::c_void,
                        to - from,
                        copy.cu_stream(),
                    )
                })?;
            }
        }
        Ok(())
    }

    /// The layer's experts as stacks on the card, once the fill has landed: the compute stream
    /// waits for it. The fill is started here if nothing started it.
    pub(super) fn stacks(&self) -> Result<Arc<[QTensor; 3]>> {
        let stage = self.stage.as_ref().expect("a staged layer");
        let Some(stacks) = stage.full.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
            crate::tensor::bail!("a wide batch's fill has no room");
        };
        // Held and already filled: the compute stream waited for the fill at the first forward,
        // and the next ones queue behind that wait.
        if stage.held.load(Ordering::Acquire) && stage.filled.load(Ordering::Acquire) {
            return Ok(stacks);
        }
        self.prefetch()?;
        let event = stage
            .ready
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("the fill was queued");
        let cd = self.dev.as_cuda_device()?;
        cd.0.stream()
            .wait(&event)
            .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
        if stage.held.load(Ordering::Acquire) {
            stage.filled.store(true, Ordering::Release);
        }
        Ok(stacks)
    }

    /// Keep the layer's stacks on the card across the forwards that follow, for a prompt that
    /// goes through the layer in several chunks: the experts come once, not once per chunk.
    /// `false` ends the hold and hands the buffer on, as the last forward's release would.
    pub fn hold_stacks(&self, on: bool) -> Result<()> {
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        if on {
            stage.filled.store(false, Ordering::Release);
            stage.held.store(true, Ordering::Release);
            return Ok(());
        }
        let pending = self
            .prompt_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some((stream, chunks)) = pending {
            // Ranking the prompt is a convenience: a pass that failed, or a card out of memory,
            // must still end the hold and hand the buffer on.
            let refs: Vec<&Tensor> = chunks.iter().collect();
            let ranked = Tensor::cat(&refs, 0).and_then(|all| self.keep_hottest_now(&all, &stream));
            if let Err(e) = ranked {
                tracing::warn!("the cache is not warmed from the prompt: {e}");
            }
        }
        stage.held.store(false, Ordering::Release);
        if stage.filled.swap(false, Ordering::AcqRel) {
            self.release_stacks()?;
        }
        Ok(())
    }

    /// Forget the fills and holds of the layers' passes through the buffers. A pass cut short
    /// leaves a fill queued for a layer that never ran, and layers share two buffers, so the
    /// next pass would read another layer's experts as its own. The copies already issued
    /// are waited for first, so nothing writes a buffer after it is handed on.
    pub(super) fn reset_stage(&self) -> Result<()> {
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        stage
            .stager
            .copy
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("copy stream: {e:?}")))?;
        *stage.ready.lock().unwrap_or_else(|e| e.into_inner()) = None;
        if let Some(full) = stage
            .stager
            .full
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            *full[stage.buffer]
                .free
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = None;
        }
        Ok(())
    }

    /// The layer's kernels are queued: the buffer of a wide batch is free once they have run.
    pub(super) fn release_stacks(&self) -> Result<()> {
        let stage = self.stage.as_ref().expect("a staged layer");
        if stage.held.load(Ordering::Acquire) {
            return Ok(());
        }
        let full = stage.stager.full.lock().unwrap_or_else(|e| e.into_inner());
        let Some(full) = full.as_ref() else {
            return Ok(());
        };
        let cd = self.dev.as_cuda_device()?;
        let event =
            cd.0.stream()
                .record_event(None)
                .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?;
        *full[stage.buffer]
            .free
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(event);
        Ok(())
    }
}

/// Heat of the expert a slot holds; an empty slot is the coldest there is.
fn slot_heat(st: &TierState, slot: usize) -> f32 {
    match st.owner[slot] {
        e if e >= 0 => st.heat[e as usize],
        _ => -1.0,
    }
}

/// The dtype of a layer's stacks, for a log line.
pub fn stack_dtype(moe: &MoeWeights) -> Option<GgmlDType> {
    moe.gate_exps_sep.as_ref().map(|t| t.dtype())
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::tensor::quantized::QMatMul;
    use crate::tensor::DType;

    fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u64 + 1)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(40503));
                (((h >> 8) % 2001) as f32 / 1000.0 - 1.0) * scale
            })
            .collect()
    }

    /// Fill the arenas of `stager` with 0xFF, which read as f16 is not a number.
    /// The memory pool of what is lent is the card's, shared by the tests of a process: a test that
    /// measures what lending gives back runs while no other holds a room in it.
    static LENT_POOL: Mutex<()> = Mutex::new(());

    fn poison(stager: &CardStager) {
        use cudarc::driver::DevicePtr;
        let cd = stager.dev.as_cuda_device().unwrap();
        cd.0.context().bind_to_thread().unwrap();
        for buffer in &stager.buffers {
            for arena in &buffer.arenas {
                let base = arena.device_ptr(arena.stream()).0;
                let r = unsafe { cudarc::driver::sys::cuMemsetD8_v2(base, 0xFF, arena.len()) };
                assert_eq!(r, cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS);
            }
        }
        cd.0.synchronize().unwrap();
    }

    /// A softmax-routed SiLU layer of `experts` experts on `dev` for its router and slots, the
    /// experts themselves on the host.
    fn layer(dev: &Device, experts: usize, used: usize, hidden: usize, ffn: usize) -> MoeWeights {
        layer_of(dev, experts, used, hidden, ffn, [GgmlDType::Q8_0; 2])
    }

    /// [`layer`] with the formats of the gate and up stacks and of the down stack chosen.
    fn layer_of(
        dev: &Device,
        experts: usize,
        used: usize,
        hidden: usize,
        ffn: usize,
        [gate_up, down]: [GgmlDType; 2],
    ) -> MoeWeights {
        let host = Device::Cpu;
        let stack = |seed: u64, rows: usize, cols: usize, dtype: GgmlDType| {
            // A format only read from a file takes any bytes with a small, finite scale in
            // front of each block; the others are quantised from values.
            if matches!(
                dtype,
                GgmlDType::Iq2S
                    | GgmlDType::Iq3Xxs
                    | GgmlDType::Iq3S
                    | GgmlDType::Iq4Xs
                    | GgmlDType::Iq2Xxs
                    | GgmlDType::Iq4Nl
            ) {
                let n = experts * rows * cols;
                let block_bytes = dtype.type_size();
                let blocks = n / dtype.block_size();
                let mut state = 0x9E37_79B9u32 ^ seed as u32;
                let bytes: Vec<u8> = (0..blocks * block_bytes)
                    .map(|i| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        match i % block_bytes {
                            0 => 0x00,
                            1 => 0x2C,
                            _ => (state >> 24) as u8,
                        }
                    })
                    .collect();
                let qhost = crate::tensor::quantized::QHostTensor::from_bytes(
                    &bytes,
                    dtype,
                    vec![experts, rows, cols],
                )
                .unwrap();
                return QTensor::from_native(Arc::new(qhost), &host).unwrap();
            }
            let w = Tensor::from_vec(
                values(experts * rows * cols, seed, 0.5),
                (experts, rows, cols),
                &host,
            )
            .unwrap();
            QTensor::quantize_onto(&w, dtype, &host).unwrap()
        };
        MoeWeights {
            gate_inp: QMatMul::Tensor(
                Tensor::from_vec(values(experts * hidden, 7, 0.5), (experts, hidden), dev).unwrap(),
            ),
            gate_inp_scale: Tensor::ones(hidden, DType::F32, dev).unwrap(),
            gate_inp_bias: None,
            gate_exps_bias: None,
            up_exps_bias: None,
            down_exps_bias: None,
            swiglu_clamp: None,
            gate_up_exps: None,
            down_exps: Arc::new(stack(3, hidden, ffn, down)),
            gate_exps_sep: Some(Arc::new(stack(1, ffn, hidden, gate_up))),
            up_exps_sep: Some(Arc::new(stack(2, ffn, hidden, gate_up))),
            down_exps_scale: None,
            decode_slots: dev.runs_as_card().then(|| {
                Tensor::from_vec((0..used as u32).collect::<Vec<_>>(), (used,), &host)
                    .unwrap()
                    .to_device(dev)
                    .unwrap()
            }),
            n_experts: experts,
            n_experts_used: used,
            expert_ffn_dim: ffn,
            router_prenormed: true,
            use_silu: true,
            gelu: false,
            norm_topk: true,
            sigmoid_routing: None,
            up_relu2: false,
            shared_expert: None,
            gate_up_exps_f32: None,
            tier: None,
            down_exps_f32: None,
        }
    }

    /// A decoded token through the tier gives what the host's expert loop gives, whatever the
    /// cache holds: nothing hot, some of it, then the hot experts after the worker has copied them.
    #[test]
    fn a_token_through_the_tier_matches_the_host_whatever_the_cache_holds() {
        a_token_through_the_tier(false);
    }

    /// A token through a tier with extra slots gives what the host gives while the extra slots
    /// hold experts, after they are lent, and after they come back filled.
    #[test]
    fn a_token_through_the_tier_matches_the_host_while_the_extra_slots_are_lent_and_returned() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 512usize, 512usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let (base, extra) = (4usize, 8usize);
        let built = LayerTier::new(&tiered, &card, base)
            .unwrap()
            .with_extra(extra)
            .unwrap();
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        assert_eq!(tier.slots(), base + extra);
        let inputs: Vec<Vec<f32>> = (0..3).map(|s| values(hidden, 100 + s, 1.0)).collect();
        let mut token = 0usize;
        let mut run = |count: usize| {
            for _ in 0..count {
                let x = &inputs[token % inputs.len()];
                token += 1;
                let on_host = Tensor::from_vec(x.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
                let on_card = Tensor::from_vec(x.clone(), (1, 1, hidden), &card).unwrap();
                let want = reference
                    .forward(&on_host, &on_host, hidden, 1e-5, false)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let got = tiered
                    .forward(&on_card, &on_card, hidden, 1e-5, false)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                tier.settle();
                let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 2e-2 * peak,
                    "token {token}: max diff {worst} of {peak}"
                );
            }
        };
        run(60);
        let held = |tier: &LayerTier| {
            let st = tier.state.lock().unwrap();
            (tier.base_slots..st.owner.len())
                .filter(|&s| st.owner[s] >= 0)
                .count()
        };
        assert!(held(&tier) > 0, "no expert reached the extra slots");
        let free = || {
            let _ = card.as_cuda_device().unwrap().0.stream().synchronize();
            crate::inference::place::ladder::release_pools();
            crate::inference::place::expert_cache::free_now(&card)
        };
        let before = free();
        assert!(tier.lend_extra());
        let returned = free().saturating_sub(before);
        let lent_bytes = (extra * tier.expert_bytes.iter().sum::<usize>()) as u64;
        assert!(
            returned >= lent_bytes / 2,
            "lending {lent_bytes} bytes gave the card {returned}"
        );
        assert_eq!(held(&tier), 0);
        run(30);
        assert!(!tier.lend_extra(), "lending twice");
        let filled = tier.restore_extra().unwrap();
        tier.settle();
        assert!(filled > 0, "nothing came back");
        assert_eq!(held(&tier), filled);
        run(30);
    }

    /// The same with room for the misses to pass through the card: some run there, the others
    /// on the host, and the split follows the rates the layer measures.
    #[test]
    fn a_token_through_the_transit_room_matches_the_host() {
        a_token_through_the_tier(true);
    }

    fn a_token_through_the_tier(transit: bool) {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let slots = 6;
        assert!(serves(&tiered));
        let mut built = LayerTier::new(&tiered, &card, slots).unwrap();
        if transit {
            let widest = [0, 1, 2].map(|i| {
                [
                    tiered.gate_exps_sep.as_ref().unwrap(),
                    tiered.up_exps_sep.as_ref().unwrap(),
                    &tiered.down_exps,
                ][i]
                    .native_qtensor()
                    .data()
                    .len()
            });
            let stager = CardStager::new(&card, widest, experts).unwrap();
            poison(&stager);
            built = built.with_stage(&stager, 0, false).unwrap();
        }
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        assert_eq!(tier.slots(), slots);
        // A few distinct inputs, repeated: their experts become the hot set.
        let inputs: Vec<Vec<f32>> = (0..3).map(|s| values(hidden, 100 + s, 1.0)).collect();
        let mut hit_rate_first = None;
        for token in 0..60 {
            let x = &inputs[token % inputs.len()];
            let on_host = Tensor::from_vec(x.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x.clone(), (1, 1, hidden), &card).unwrap();
            let want = reference
                .forward(&on_host, &on_host, hidden, 1e-5, false)
                .unwrap();
            let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got = tiered
                .forward(&on_card, &on_card, hidden, 1e-5, false)
                .unwrap();
            let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            tier.settle();
            let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 2e-2 * peak,
                "token {token}: max diff {worst} of {peak}"
            );
            if token == inputs.len() - 1 {
                let (routed, hits, _) = tier.counters();
                hit_rate_first = Some(hits as f64 / routed as f64);
            }
        }
        let (routed, hits, promoted) = tier.counters();
        if transit {
            assert!(
                tier.uploaded.load(Ordering::Relaxed) > 0,
                "no miss went through the transit room"
            );
        }
        assert!(promoted > 0, "the hot experts were never copied in");
        assert!(
            hits as f64 / routed as f64 > hit_rate_first.unwrap(),
            "the cache did not learn: {hits} of {routed}"
        );
    }

    /// A token through the kernels a recorded decode is made of, the host answering from its
    /// thread, gives what the host's expert loop gives it, and the cache learns as it does.
    #[test]
    fn a_token_through_the_recorded_decode_matches_the_host() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let slots = 6;
        let built = LayerTier::new(&tiered, &card, slots).unwrap();
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(hidden, used, 4).unwrap();
        let worker = TierWorker::start(&card, vec![tier.clone()]).unwrap();
        GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
        let inputs: Vec<Vec<f32>> = (0..3).map(|s| values(hidden, 100 + s, 1.0)).collect();
        let mut first = None;
        for token in 0..60 {
            let x = &inputs[token % inputs.len()];
            let on_host = Tensor::from_vec(x.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x.clone(), (1, 1, hidden), &card).unwrap();
            let want = reference
                .forward(&on_host, &on_host, hidden, 1e-5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = tiered
                .forward(&on_card, &on_card, hidden, 1e-5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            tier.settle();
            let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 2e-2 * peak,
                "token {token}: max diff {worst} of {peak}"
            );
            if token == inputs.len() - 1 {
                let (routed, hits, _) = tier.counters();
                first = Some(hits as f64 / routed as f64);
            }
        }
        GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
        let (routed, hits, promoted) = tier.counters();
        assert!(promoted > 0, "the hot experts were never copied in");
        assert!(
            hits as f64 / routed as f64 > first.unwrap(),
            "the cache did not learn: {hits} of {routed}"
        );
        drop(worker);
    }

    /// The rows of a draft verify as the kernels of the recorded decode, the host answering
    /// through the mailbox and the lent slots in play, give what the host's expert loop gives
    /// each row.
    #[test]
    fn the_rows_of_a_verify_through_the_recorded_decode_match_the_host() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let built = LayerTier::new(&tiered, &card, 6)
            .unwrap()
            .with_extra(8)
            .unwrap();
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(hidden, used, 6).unwrap();
        let worker = TierWorker::start(&card, vec![tier.clone()]).unwrap();
        GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
        let _order = crate::tensor::quantized::DecodeOrder::enter();
        for round in 0..40usize {
            let rows = 1 + round % 6;
            let x = values(rows * hidden, 100 + (round % 4) as u64, 1.0);
            let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
            let want = reference
                .forward_rows(&on_host, &on_host, hidden, 1e-5)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = tiered
                .forward_rows(&on_card, &on_card, hidden, 1e-5)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            tier.settle();
            for r in 0..rows {
                let (w, g) = (
                    &want[r * hidden..(r + 1) * hidden],
                    &got[r * hidden..(r + 1) * hidden],
                );
                let peak = w.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let worst = g
                    .iter()
                    .zip(w)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 2e-2 * peak,
                    "round {round}, row {r} of {rows}: max diff {worst} of {peak}"
                );
            }
        }
        GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
        let (routed, hits, promoted) = tier.counters();
        assert!(promoted > 0, "the hot experts were never copied in");
        assert!(hits > 0 && hits < routed, "{hits} of {routed} on the card");
        drop(worker);
    }

    /// The rows of a draft verify as the kernels of the recorded decode, with a transit room
    /// taking a share of each request's missing experts and the card running them after its
    /// wait, give what the host's expert loop gives each row.
    #[test]
    fn the_rows_of_a_verify_through_the_recorded_decode_and_the_transit_room_match_the_host() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let widest = [0, 1, 2].map(|i| {
            [
                tiered.gate_exps_sep.as_ref().unwrap(),
                tiered.up_exps_sep.as_ref().unwrap(),
                &tiered.down_exps,
            ][i]
                .native_qtensor()
                .data()
                .len()
        });
        let stager = CardStager::new(&card, widest, experts).unwrap();
        let built = LayerTier::new(&tiered, &card, 6)
            .unwrap()
            .with_stage(&stager, 0, false)
            .unwrap();
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(hidden, used, 6).unwrap();
        let worker = TierWorker::start(&card, vec![tier.clone()]).unwrap();
        GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
        let _order = crate::tensor::quantized::DecodeOrder::enter();
        for round in 0..48usize {
            let rows = 1 + round % 6;
            let x = values(rows * hidden, 100 + (round % 4) as u64, 1.0);
            let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
            let want = reference
                .forward_rows(&on_host, &on_host, hidden, 1e-5)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = tiered
                .forward_rows(&on_card, &on_card, hidden, 1e-5)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            tier.settle();
            for r in 0..rows {
                let (w, g) = (
                    &want[r * hidden..(r + 1) * hidden],
                    &got[r * hidden..(r + 1) * hidden],
                );
                let peak = w.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let worst = g
                    .iter()
                    .zip(w)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 2e-2 * peak,
                    "round {round}, row {r} of {rows}: max diff {worst} of {peak}"
                );
            }
        }
        GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
        let (routed, hits, _) = tier.counters();
        assert!(hits > 0 && hits < routed, "{hits} of {routed} on the card");
        assert!(
            tier.uploaded.load(Ordering::Relaxed) > 0,
            "no missing expert went through the transit room"
        );
        drop(worker);
    }

    /// The rows of a draft verify through the tier together give what the host's expert loop
    /// gives them, whatever the cache holds and however the misses are split.
    #[test]
    fn the_rows_of_a_verify_through_the_tier_match_the_host() {
        rows_through_the_tier(false);
    }

    /// The same with the transit room taking some of the misses.
    #[test]
    fn the_rows_of_a_verify_through_the_transit_room_match_the_host() {
        rows_through_the_tier(true);
    }

    fn rows_through_the_tier(transit: bool) {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let slots = 6;
        let mut built = LayerTier::new(&tiered, &card, slots).unwrap();
        if transit {
            let widest = [0, 1, 2].map(|i| {
                [
                    tiered.gate_exps_sep.as_ref().unwrap(),
                    tiered.up_exps_sep.as_ref().unwrap(),
                    &tiered.down_exps,
                ][i]
                    .native_qtensor()
                    .data()
                    .len()
            });
            let stager = CardStager::new(&card, widest, experts).unwrap();
            built = built.with_stage(&stager, 0, false).unwrap();
        }
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        let _order = crate::tensor::quantized::DecodeOrder::enter();
        for round in 0..40usize {
            let rows = 1 + round % 6;
            let x = values(rows * hidden, 100 + (round % 4) as u64, 1.0);
            let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
            let want = reference
                .forward_rows(&on_host, &on_host, hidden, 1e-5)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = tiered
                .forward_rows(&on_card, &on_card, hidden, 1e-5)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            tier.settle();
            let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 2e-2 * peak,
                "round {round}, {rows} rows: max diff {worst} of {peak}"
            );
        }
        let (routed, hits, promoted) = tier.counters();
        assert!(promoted > 0, "the hot experts were never copied in");
        assert!(hits > 0 && hits < routed, "{hits} of {routed} on the card");
    }

    /// A token and the rows of a verify through a layer's experts on the card give what the
    /// host's loop gives, in the format the served models hold their layers' experts in and in
    /// the one the tests build theirs in: the two paths compute every missing expert of a
    /// decode, so a difference between them is a difference between two computations of the
    /// same step.
    #[test]
    fn the_card_and_the_host_agree_on_an_expert_in_each_served_format() {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 256usize, 512usize);
        for formats in [
            [GgmlDType::Q2_0, GgmlDType::Q2_0],
            [GgmlDType::Q8_0, GgmlDType::Q8_0],
            [GgmlDType::Iq2S, GgmlDType::Iq4Nl],
            [GgmlDType::Iq3Xxs, GgmlDType::Iq4Nl],
            [GgmlDType::Iq3S, GgmlDType::Q2_0],
            [GgmlDType::Iq4Xs, GgmlDType::Q8_0],
        ] {
            let reference = layer_of(&Device::Cpu, experts, used, hidden, ffn, formats);
            let mut resident = layer_of(&card, experts, used, hidden, ffn, formats);
            resident.tier = Some(Arc::new(LayerTier::all_on_card(&resident, &card).unwrap()));
            let _order = crate::tensor::quantized::DecodeOrder::enter();
            for rows in [1usize, 3, 7] {
                let x = values(rows * hidden, 7 + rows as u64, 1.0);
                let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
                let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
                let want = reference
                    .forward_rows(&on_host, &on_host, hidden, 1e-5)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let got = resident
                    .forward_rows(&on_card, &on_card, hidden, 1e-5)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for r in 0..rows {
                    let (w, g) = (
                        &want[r * hidden..(r + 1) * hidden],
                        &got[r * hidden..(r + 1) * hidden],
                    );
                    let distance = crate::inference::engine::decode_step::replay_distance(g, w);
                    eprintln!("{formats:?} rows {rows} row {r}: relative distance {distance:.5}");
                    assert!(
                        distance <= 1e-2,
                        "{formats:?}, {rows} rows, row {r}: the card is {distance:.4} of the host's norm away"
                    );
                }
            }
        }
    }

    /// A layer with every expert on the card runs as a loaded one does: a token and wide batches
    /// give what the host's loop gives.
    #[test]
    fn a_resident_layer_matches_the_host() {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut resident = layer(&card, experts, used, hidden, ffn);
        resident.tier = Some(Arc::new(LayerTier::all_on_card(&resident, &card).unwrap()));
        assert!(resident.tier.as_ref().unwrap().is_resident());
        for (rows, prefill) in [(1usize, false), (24, true), (700, true)] {
            let x = values(rows * hidden, 31, 1.0);
            let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
            let want = reference
                .forward(&on_host, &on_host, hidden, 1e-5, prefill)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = resident
                .forward(&on_card, &on_card, hidden, 1e-5, prefill)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 2e-2 * peak,
                "rows {rows}: max diff {worst} of {peak}"
            );
        }
    }

    /// A wide batch through the tier gives what the host's loop gives, with the layer's experts
    /// copied to the card from the host and from the cache, and with the host range locked or not.
    #[test]
    fn a_wide_batch_through_the_tier_matches_the_host() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        for (pin, rows) in [(false, 24usize), (true, 24), (false, 1520), (true, 3000)] {
            let mut tiered = layer(&card, experts, used, hidden, ffn);
            let widest = [0, 1, 2].map(|i| {
                let t = [
                    tiered.gate_exps_sep.as_ref().unwrap(),
                    tiered.up_exps_sep.as_ref().unwrap(),
                    &tiered.down_exps,
                ][i];
                t.native_qtensor().data().len()
            });
            let stager = CardStager::new(&card, widest, experts).unwrap();
            // What an earlier layer left in the arenas: bytes whose f16 scales are not finite.
            poison(&stager);
            // Some experts on the card, the rest to come from the host.
            let tier = LayerTier::new(&tiered, &card, 9)
                .unwrap()
                .with_stage(&stager, 0, pin)
                .unwrap();
            assert!(tier.stream_pays(rows, used));
            tiered.tier = Some(Arc::new(tier));
            let x = values(rows * hidden, 55, 1.0);
            let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
            let want = reference
                .forward(&on_host, &on_host, hidden, 1e-5, true)
                .unwrap();
            let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            // Twice, since the second fill comes after the first buffer was released.
            for _ in 0..2 {
                let got = tiered
                    .forward(&on_card, &on_card, hidden, 1e-5, true)
                    .unwrap();
                let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 2e-2 * peak,
                    "pin {pin}: max diff {worst} of {peak}"
                );
            }
        }
    }

    /// A wide batch warms the cache with the experts it used: afterwards every slot holds one,
    /// and a decoded token reads the right blocks from them.
    #[test]
    fn a_wide_batch_warms_the_cache_with_the_experts_it_used() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let widest = [0, 1, 2].map(|i| {
            [
                tiered.gate_exps_sep.as_ref().unwrap(),
                tiered.up_exps_sep.as_ref().unwrap(),
                &tiered.down_exps,
            ][i]
                .native_qtensor()
                .data()
                .len()
        });
        let stager = CardStager::new(&card, widest, experts).unwrap();
        let slots = 6;
        let tier = LayerTier::new(&tiered, &card, slots)
            .unwrap()
            .with_stage(&stager, 0, false)
            .unwrap();
        let rows = 1520usize;
        assert!(tier.stream_pays(rows, used));
        let before = tier.counters().2;
        tiered.tier = Some(Arc::new(tier));
        let tier = tiered.tier.clone().unwrap();
        let x = values(rows * hidden, 55, 1.0);
        let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
        tiered
            .forward(&on_card, &on_card, hidden, 1e-5, true)
            .unwrap();
        tier.settle();
        let held = (0..experts).filter(|&e| tier.resident(e)).count();
        assert_eq!(held, slots, "every slot holds an expert");
        assert!(
            tier.counters().2 > before,
            "no expert the batch used took the place of a colder one"
        );
        // A token whose experts are in the cache gives what the host gives.
        for token in 0..8usize {
            let row = values(hidden, 200 + token as u64, 1.0);
            let on_host = Tensor::from_vec(row.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(row, (1, 1, hidden), &card).unwrap();
            let want = reference
                .forward(&on_host, &on_host, hidden, 1e-5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = tiered
                .forward(&on_card, &on_card, hidden, 1e-5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 2e-2 * peak,
                "token {token}: max diff {worst} of {peak}"
            );
        }
        let (_, hits, _) = tier.counters();
        assert!(hits > 0, "the warmed cache served no expert");
    }

    /// The same, in the formats of a real file and at the row counts where the expert-batched
    /// kernels take over from the sorted launches.
    #[test]
    fn a_wide_batch_matches_the_host_in_k_quants_across_the_batching_threshold() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        // The dimensions of a real file: a hidden width that is not a multiple of the expert width.
        let (experts, used, hidden, ffn) = (32usize, 8usize, 2560usize, 640usize);
        for formats in [
            [GgmlDType::Q4K, GgmlDType::Q5_0],
            [GgmlDType::Q4K, GgmlDType::Q8_0],
        ] {
            let reference = layer_of(&Device::Cpu, experts, used, hidden, ffn, formats);
            let mut tiered = layer_of(&card, experts, used, hidden, ffn, formats);
            let widest = [0, 1, 2].map(|i| {
                let t = [
                    tiered.gate_exps_sep.as_ref().unwrap(),
                    tiered.up_exps_sep.as_ref().unwrap(),
                    &tiered.down_exps,
                ][i];
                t.native_qtensor().data().len()
            });
            let stager = CardStager::new(&card, widest, experts).unwrap();
            poison(&stager);
            let tier = LayerTier::new(&tiered, &card, 10)
                .unwrap()
                .with_stage(&stager, 0, false)
                .unwrap();
            tiered.tier = Some(Arc::new(tier));
            for rows in [100usize, 700, 900, 1520] {
                let x = values(rows * hidden, 77, 1.0);
                let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
                let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
                let want = reference
                    .forward(&on_host, &on_host, hidden, 1e-5, true)
                    .unwrap();
                let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let got = tiered
                    .forward(&on_card, &on_card, hidden, 1e-5, true)
                    .unwrap();
                let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "{formats:?} rows {rows}: non-finite output"
                );
                let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 3e-2 * peak,
                    "{formats:?} rows {rows}: max diff {worst} of {peak}"
                );
            }
        }
    }
}
