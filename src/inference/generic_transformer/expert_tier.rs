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
//! The counts decay within a few tens of tokens (`ADAPT_DECAY` every `ADAPT_EVERY_TOKENS`), so
//! the cache follows the routing of the text in hand rather than that of the prompt or the load.

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

mod direct;
mod graph;
mod hot;
pub(crate) use graph::mapped;
pub use graph::TierWorker;
pub use hot::HotLocks;

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
const ADAPT_EVERY_TOKENS: usize = 10;
const ADAPT_DECAY: f32 = 0.7;
const ADAPT_MIN_HEAT: f32 = 2.0;
const ADAPT_MARGIN: f32 = 1.5;
const ADAPT_SWAPS_A_REQUEST: usize = 2;
/// Copies into the transit room timed before the share it takes of a request's missing experts
/// is fixed for the run.
const TRANSIT_PACE_SAMPLES: u64 = 16;

/// A tier's table on the card of its extra stacks' addresses: three `u64`, at an address that
/// stays for the tier's life.
struct ExtraTable {
    slice: Mutex<crate::tensor::gpu_runtime::driver::CudaSlice<u64>>,
    addr: u64,
}

impl ExtraTable {
    /// A table of nulls on `dev`.
    fn new(dev: &Device) -> Result<Self> {
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        let cd = dev.as_cuda_device()?;
        let slice = cd.alloc_zeros::<u64>(3)?;
        let addr = slice.device_ptr(slice.stream()).0;
        Ok(Self {
            slice: Mutex::new(slice),
            addr,
        })
    }

    /// Copies `addrs` into the table from the host, done before this returns: never while a
    /// graph is being recorded.
    fn write(&self, dev: &Device, addrs: [u64; 3]) -> Result<()> {
        let cd = dev.as_cuda_device()?;
        let mut slice = self.slice.lock().unwrap_or_else(|e| e.into_inner());
        cd.memcpy_htod(&addrs[..], &mut *slice)?;
        cd.0.stream()
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("extra slots' table: {e:?}")))
    }
}

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
    /// On the card, once the tier has extra slots: the extra stacks' addresses, gate, up and
    /// down, each null while the slots are lent. Written whenever the stacks are lent or come
    /// back, never moved: a recorded launch reads the extra stacks through it.
    extra_table: Option<ExtraTable>,
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
    prompt_ids: Mutex<
        Option<(
            Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
            Vec<Tensor>,
        )>,
    >,
    /// The stream this layer's copies into the cache run on.
    copy: Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    /// A copy into the cache is queued or running.
    promoting: std::sync::atomic::AtomicBool,
    /// Microseconds a decoded token took here, smoothed: what a batch's rows cost one by one.
    decode_us: AtomicU64,
    /// What a decoded token moves, made on the first.
    scratch: Mutex<Option<Scratch>>,
    /// Microseconds (an `f64`'s bits, `0` before a measure) the copy of one missing expert to the
    /// transit room takes, and the host's computation of one, for a request of one row and for
    /// a request of several (`pace_class`): the host's share of a row grows with the rows that
    /// share an expert, the copy's does not.
    upload_us: [AtomicU64; 2],
    host_us: [AtomicU64; 2],
    /// Samples the two paces were measured over, and the share of the missing experts the
    /// transit room takes once they are, in eighths, for each class of request: set once, so
    /// that a request's split no longer follows the clock and a run gives the text it gave.
    paced: [AtomicU64; 2],
    transit_eighths: [AtomicU64; 2],
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
    upload_start: crate::tensor::gpu_runtime::driver::CudaEvent,
    upload_end: crate::tensor::gpu_runtime::driver::CudaEvent,
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
        let pinned = unsafe { crate::tensor::gpu_runtime::driver::result::malloc_host(bytes, 0) }
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
fn timed_event(dev: &Device) -> Result<crate::tensor::gpu_runtime::driver::CudaEvent> {
    dev.as_cuda_device()?
        .0
        .context()
        .new_event(Some(
            crate::tensor::gpu_runtime::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
        ))
        .map_err(|e| crate::tensor::Error::msg(format!("event: {e:?}")))
}

impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe {
            crate::tensor::gpu_runtime::driver::result::free_host(
                self.pinned as *mut std::ffi::c_void,
            )
        }
        .ok();
    }
}

/// The card's address of the first element of `t`, which a copy reads or writes.
pub(super) fn address_of<T: crate::tensor::kernel_ffi::CudaDType>(t: &Tensor) -> Result<u64> {
    use crate::tensor::gpu_runtime::driver::DevicePtr;
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
    copy: Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
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
    /// The most a decode has taken of the card between a prefill's end and the next prefill,
    /// measured; `None` until one has run. The extra slots come back only within what it leaves.
    decode_need: Mutex<Option<u64>>,
    /// The card's free memory once the extra slots came back, when a prefill ended, and what
    /// the card's memory pool held in use then.
    free_after_return: Mutex<Option<(u64, u64)>>,
    /// A prefill lent the slots since then: what the card lost meanwhile is the prefill's.
    lent_since_return: std::sync::atomic::AtomicBool,
    /// What a decode that recorded no graph was seen to take, apart from [`Self::decode_need`].
    replay_need: Mutex<Option<u64>>,
    /// A graph was recorded since the slots came back.
    recorded_since_return: std::sync::atomic::AtomicBool,
}

/// One of two sets of arenas, a stack each, that consecutive layers take turns over.
struct StageBuffer {
    arenas: [Arc<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>; 3],
}

/// [`StageBuffer`] wide enough for the whole stacks of the widest layer.
struct FullBuffer {
    arenas: [Arc<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>; 3],
    /// Recorded when the layer that read the buffer last has queued all its kernels: the next
    /// fill waits for it.
    free: Mutex<Option<crate::tensor::gpu_runtime::driver::CudaEvent>>,
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
    ready: Mutex<Option<crate::tensor::gpu_runtime::driver::CudaEvent>>,
    /// Host ranges, per stack, that are page-locked: copies from them are asynchronous.
    pinned: [Option<(usize, usize)>; 3],
    /// The direct reader for the parts of an upload the page cache does not hold.
    #[cfg(target_os = "linux")]
    bounce: Mutex<BounceState>,
    /// The stacks are kept for several forwards in a row (see `hold_stacks`).
    held: std::sync::atomic::AtomicBool,
    /// While held: the fill has landed and the compute stream has waited for it.
    filled: std::sync::atomic::AtomicBool,
}

/// A stage's direct reader: not opened yet, open, or given up after a failure.
#[cfg(target_os = "linux")]
enum BounceState {
    Closed,
    Open(direct::Bounce),
    Off,
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
    experts_range(host, 0, n, dev)
}

/// The experts `first..first + n` of `host` on `dev`: a view of the host's own bytes, so the host
/// holds no copy.
pub(super) fn experts_range(
    host: &Arc<QTensor>,
    first: usize,
    n: usize,
    dev: &Device,
) -> Result<QTensor> {
    let inner = host.native_qtensor();
    let bytes = inner.data();
    let per_expert = bytes.len() / host.shape().dims()[0];
    let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(inner.clone());
    let view = unsafe {
        QHostTensor::view(
            owner,
            bytes.as_ptr(),
            first * per_expert,
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
    /// The room each card has left for experts after the load: the plan's budget for it, less
    /// what the load took of it since `free_before` and less the draft head's experts when it
    /// keeps them there.
    fn expert_room(
        &self,
        ordinal: usize,
        dev: &Device,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        free_before: &std::collections::HashMap<usize, u64>,
        draft_resident: bool,
    ) -> u64 {
        use crate::inference::place::{expert_cache, layer_executor::DeviceKind};
        let budget = plan
            .segments
            .iter()
            .filter(|s| s.kind == DeviceKind::Cuda(ordinal))
            .map(|s| s.free_memory_bytes)
            .max()
            .unwrap_or(0);
        let draft = self
            .draft
            .as_ref()
            .filter(|_| draft_resident)
            .filter(|d| d.layer_dev == super::LayerDevice::Cuda(ordinal))
            .and_then(|d| d.layer.moe.as_ref())
            .filter(|m| m.experts_on_host() && serves(m))
            .map_or(0, |m| (expert_bytes(m) * m.n_experts) as u64);
        let before = free_before.get(&ordinal).copied().unwrap_or(0);
        let used = before.saturating_sub(expert_cache::free_now(dev));
        budget.saturating_sub(draft).saturating_sub(used)
    }

    /// Before any card caches the experts the plan left on the host: when the cards together
    /// have room for every one of them, each such layer's experts go to the cards whole, split
    /// across them where that is predicted faster than its own card alone, else on its own
    /// card. The layers this does not place are cached as before.
    fn split_host_experts(
        &mut self,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        free_before: &std::collections::HashMap<usize, u64>,
        draft_resident: bool,
    ) -> Result<()> {
        use super::LayerDevice;
        use crate::inference::generic_transformer::moe_weights::ExpertShard;
        use crate::inference::place::expert_parallel::{split_experts, ExpertLayer, ShardCard};
        if cuda_devices.len() < 2 {
            return Ok(());
        }
        let candidates: Vec<(usize, usize)> = (0..self.layers.len())
            .filter_map(|l| match self.layer_devs[l] {
                LayerDevice::Cuda(o) => Some((l, o)),
                _ => None,
            })
            .filter(|&(l, _)| {
                self.layers[l].moe.as_ref().is_some_and(|m| {
                    m.experts_on_host() && serves(m) && m.shards.is_empty() && m.tier.is_none()
                })
            })
            .collect();
        if candidates.is_empty() {
            return Ok(());
        }
        let mut ordinals: Vec<usize> = cuda_devices.keys().copied().collect();
        ordinals.sort_unstable();
        let mut room: std::collections::HashMap<usize, u64> = ordinals
            .iter()
            .map(|&o| {
                (
                    o,
                    self.expert_room(o, &cuda_devices[&o], plan, free_before, draft_resident),
                )
            })
            .collect();
        let need: u64 = candidates
            .iter()
            .filter_map(|&(l, _)| self.layers[l].moe.as_ref())
            .map(|m| (expert_bytes(m) * m.n_experts) as u64)
            .sum();
        if need > room.values().sum::<u64>() {
            return Ok(());
        }
        let read = crate::inference::place::expert_parallel::card_read_rates();
        for (l, owner) in candidates {
            let moe = self.layers[l]
                .moe
                .as_ref()
                .expect("a candidate has experts");
            let layer = ExpertLayer {
                n_experts: moe.n_experts,
                used: moe.n_experts_used,
                expert_bytes: expert_bytes(moe) as u64,
                row_bytes: (moe.down_exps.shape().dims()[1] * std::mem::size_of::<f32>()) as u64,
            };
            let mut cards = vec![ShardCard {
                index: owner,
                read_bytes_per_us: read.get(&owner).copied().unwrap_or(0.0),
                link_us: 0.0,
                link_bytes_per_us: f64::INFINITY,
                room: room[&owner],
            }];
            for &o in ordinals.iter().filter(|&&o| o != owner) {
                if let Some((us, rate)) =
                    crate::inference::place::device_probe::card_pair_link(owner, o)
                {
                    cards.push(ShardCard {
                        index: o,
                        read_bytes_per_us: read.get(&o).copied().unwrap_or(0.0),
                        link_us: us,
                        link_bytes_per_us: rate,
                        room: room[&o],
                    });
                }
            }
            if cards.iter().any(|c| c.read_bytes_per_us <= 0.0) {
                continue;
            }
            let all = layer.n_experts as u64 * layer.expert_bytes;
            match split_experts(&layer, &cards, 1) {
                Some(counts) => {
                    let mut shards = Vec::new();
                    let mut first = 0usize;
                    for (card, &count) in cards.iter().zip(&counts).filter(|(_, &c)| c > 0) {
                        shards.push(ExpertShard::from_host(
                            moe,
                            first,
                            count,
                            &cuda_devices[&card.index],
                        )?);
                        *room.get_mut(&card.index).expect("a known card") -=
                            count as u64 * layer.expert_bytes;
                        first += count;
                    }
                    tracing::info!(
                        "expert parallel: layer {l} split over cards {:?} as {:?} experts",
                        cards.iter().map(|c| c.index).collect::<Vec<_>>(),
                        counts
                    );
                    self.layers[l]
                        .moe
                        .as_mut()
                        .expect("a candidate has experts")
                        .shards = shards;
                }
                None if room[&owner] >= all => {
                    let dev = &cuda_devices[&owner];
                    let moe = self.layers[l]
                        .moe
                        .as_mut()
                        .expect("a candidate has experts");
                    moe.tier = Some(Arc::new(LayerTier::all_on_card(moe, dev)?));
                    *room.get_mut(&owner).expect("a known card") -= all;
                }
                None => {}
            }
        }
        Ok(())
    }

    /// Give every layer whose experts the plan left on the host a cache of them on its card:
    /// `slots` takes what each card has left of its budget, `free_before` being what the card
    /// had free before the load.
    pub(super) fn attach_expert_tiers(
        &mut self,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        free_before: &std::collections::HashMap<usize, u64>,
        draft_resident: bool,
    ) -> Result<()> {
        use super::LayerDevice;
        use crate::inference::place::{expert_cache, layer_executor::DeviceKind};
        self.split_host_experts(cuda_devices, plan, free_before, draft_resident)?;
        // Locked pages cannot be reclaimed: the cards share what the host can give.
        let mut pin_left = expert_cache::host_pin_budget();
        // When the budget cannot lock every tiered layer whole, it is spent on the most routed
        // experts of all of them instead (`HotLocks`). A card whose layers turn out resident
        // takes none of it; the sum counts its layers anyway.
        let tiered_bytes: u64 = self
            .layers
            .iter()
            .enumerate()
            .filter(|(l, _)| matches!(self.layer_devs[*l], LayerDevice::Cuda(o) if cuda_devices.contains_key(&o)))
            .filter_map(|(_, layer)| layer.moe.as_ref())
            .filter(|m| m.experts_on_host() && serves(m) && m.shards.is_empty() && m.tier.is_none())
            .map(|m| (expert_bytes(m) * m.n_experts) as u64)
            .sum();
        let per_expert = cfg!(target_os = "linux") && tiered_bytes > pin_left;
        let mut hot_tiers: Vec<(usize, Arc<LayerTier>)> = Vec::new();
        let mut ordinals: Vec<_> = cuda_devices.keys().copied().collect();
        ordinals.sort_unstable();
        for ordinal in ordinals {
            let dev = &cuda_devices[&ordinal];
            let tiered: Vec<usize> = (0..self.layers.len())
                .filter(|&l| self.layer_devs[l] == LayerDevice::Cuda(ordinal))
                .filter(|&l| {
                    self.layers[l].moe.as_ref().is_some_and(|m| {
                        m.experts_on_host() && serves(m) && m.shards.is_empty() && m.tier.is_none()
                    })
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
            // The draft head's layer, when the drafts run and it runs on this card with its
            // experts on the host, keeps every one of its experts on the card: a draft that
            // missed one would wait for the host at each guess. Its stacks come off the budget
            // before the slots.
            let draft_bytes = self
                .draft
                .as_ref()
                .filter(|_| draft_resident)
                .filter(|d| d.layer_dev == LayerDevice::Cuda(ordinal))
                .and_then(|d| d.layer.moe.as_ref())
                .filter(|m| m.experts_on_host() && serves(m))
                .map(|m| expert_bytes(m) * m.n_experts);
            let budget = plan
                .segments
                .iter()
                .filter(|s| s.kind == DeviceKind::Cuda(ordinal))
                .map(|s| s.free_memory_bytes)
                .max()
                .unwrap_or(0);
            // What the slots may take of the budget: the head's stacks are taken off it first.
            // The room lent by a prefill is counted against the whole budget, as the plan is.
            let slots_budget = budget.saturating_sub(draft_bytes.unwrap_or(0) as u64);
            let before = free_before.get(&ordinal).copied().unwrap_or(0);
            // Room for every expert of every layer beside the plan's own reserve: the layers
            // are resident, and need neither a copy room nor a share of the card for one.
            let widest_mixture = tiered
                .iter()
                .filter_map(|&l| self.layers[l].moe.as_ref().map(|m| m.n_experts))
                .max()
                .unwrap_or(0);
            if expert_cache::slots_per_layer(
                slots_budget,
                before,
                expert_cache::free_now(dev),
                1,
                bytes,
            ) >= widest_mixture
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
            // The card's free memory as the load left it, before the stager takes its rooms:
            // those are counted once, in what the slots leave free.
            let free_at_tier = expert_cache::free_now(dev);
            let stager = CardStager::new(dev, widest, widest_mixture)?;
            let held: u64 = stager.full_bytes()
                + stager.transit_bytes()
                + chunk_bytes.saturating_sub(before.saturating_sub(budget));
            let slots = expert_cache::slots_per_layer(
                slots_budget.saturating_sub(held),
                before,
                free_at_tier,
                1,
                bytes,
            );
            if slots == 0 {
                tracing::warn!(
                    "expert tier on card {ordinal}: no room for a cache slot ({:.2} GB budget, {:.2} GB taken by the load, {:.2} GB held for the stager and a prefill chunk); the experts are served from the host",
                    slots_budget as f64 / 1e9,
                    before.saturating_sub(free_at_tier) as f64 / 1e9,
                    held as f64 / 1e9
                );
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
                let pin = !per_expert && layer_bytes <= pin_left;
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
                let tier = Arc::new(tier);
                if per_expert {
                    hot_tiers.push((l, tier.clone()));
                }
                moe.tier = Some(tier);
            }
            // The head's layer runs as one loaded on the card: every expert there, no host
            // answer to wait for at a guess.
            if draft_bytes.is_some() {
                if let Some(moe) = self.draft.as_mut().and_then(|d| d.layer.moe.as_mut()) {
                    moe.tier = Some(Arc::new(LayerTier::all_on_card(moe, dev)?));
                }
            }
            // The layers' decoded tokens run as kernels alone, served by one thread of the host.
            let arcs: Vec<(String, Arc<LayerTier>)> = tiered
                .iter()
                .filter_map(|&l| {
                    let tier = self.layers[l].moe.as_ref()?.tier.clone()?;
                    Some((format!("layer {l}"), tier))
                })
                .chain(
                    self.draft
                        .as_ref()
                        .filter(|d| d.layer_dev == LayerDevice::Cuda(ordinal))
                        .and_then(|d| d.layer.moe.as_ref().and_then(|m| m.tier.clone()))
                        .map(|t| ("the draft head".to_string(), t)),
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
        if !hot_tiers.is_empty() {
            self.hot_locks = HotLocks::new(hot_tiers, pin_left).map(Arc::new);
        }
        Ok(())
    }

    /// Move the locks of the hot experts to what the tiers route most now, off this thread.
    pub fn rerank_hot_experts(&self) {
        if let Some(h) = &self.hot_locks {
            h.rerank();
        }
    }
}

impl super::GenericHeteroTransformer {
    /// The generation of the graphs the engine keeps for the next request, `None` for none.
    pub fn note_graphs_kept(&mut self, at: Option<u64>) {
        self.kept_graphs_at = at;
    }

    /// A graph was recorded: the decode's room is measured as a recording one's.
    pub fn note_graphs_recorded(&self) {
        for s in self.card_stagers() {
            s.note_recorded();
        }
    }

    /// Take the memory the caches lent to a prefill of `tokens` rows back to the cards, when the
    /// prefill needs more than they have free: every layer's extra slots are freed. How many
    /// layers lent.
    pub fn lend_expert_cache(&mut self, tokens: usize) -> usize {
        for stager in self.card_stagers() {
            stager.note_lend();
        }
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
        // The prompt's own prefix is restored by now: the snapshots of other prompts give the
        // chunk their room before the expert slots do.
        let free = if free < need {
            self.drop_kv_snapshots_for(need - free);
            crate::inference::place::vram_manager::tightest_free_on_cards(shape.cards)
        } else {
            free
        };
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

    /// The stagers of the cards the layers' caches are on, each once.
    fn card_stagers(&self) -> Vec<Arc<CardStager>> {
        let mut out: Vec<Arc<CardStager>> = Vec::new();
        for s in self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
            .filter_map(|t| t.stager())
        {
            if !out.iter().any(|t| Arc::ptr_eq(t, &s)) {
                out.push(s);
            }
        }
        out
    }

    /// Give the caches their extra slots again, filled with the experts the prefill found
    /// hottest, within what the decode that follows was seen to need. How many experts were
    /// copied.
    pub fn return_expert_cache(&mut self) -> Result<usize> {
        for stager in self.card_stagers() {
            stager.note_decode();
        }
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
        let mut moved = false;
        // What each card's slots may take back, spent layer by layer.
        // The graphs kept for the next decode are still current: it records nothing.
        let replay = self.kept_graphs_at == Some(self.graph_generation());
        let mut room: Vec<(Arc<CardStager>, u64)> = self
            .card_stagers()
            .into_iter()
            .map(|s| {
                let r = s.room_for_extra(replay);
                (s, r)
            })
            .collect();
        for tier in self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
        {
            let Some(left) = tier
                .stager()
                .and_then(|s| room.iter_mut().find(|(t, _)| Arc::ptr_eq(t, &s)))
                .map(|(_, r)| r)
            else {
                continue;
            };
            let had = tier.has_extra();
            let slot = tier.expert_bytes.iter().sum::<usize>().max(1) as u64;
            let asked = (*left / slot).min(tier.extra_slots() as u64);
            // A layer that cannot take its slots back runs on without them.
            match tier.restore_extra_within(*left) {
                Ok(n) => copied += n,
                Err(e) => tracing::warn!("expert tier: the extra slots did not come back: {e}"),
            }
            if tier.has_extra() && !had {
                *left = left.saturating_sub(slot * asked);
            }
            moved |= had != tier.has_extra();
        }
        for (stager, _) in &room {
            stager.note_return();
        }
        let (held, of): (usize, usize) = self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
            .map(|t| {
                (
                    if t.has_extra() { t.extra_slots() } else { 0 },
                    t.extra_slots(),
                )
            })
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        if moved {
            tracing::info!("expert tier: the extra slots back in {held} of {of} layer-slots");
        }
        crate::inference::place::vram_manager::residency_changed();
        Ok(copied)
    }
}

impl LayerTier {
    /// Whether the extra slots hold their memory now, rather than lending it.
    pub fn has_extra(&self) -> bool {
        self.extra
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

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
            extra_table: None,
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
            upload_us: [AtomicU64::new(0), AtomicU64::new(0)],
            host_us: [AtomicU64::new(0), AtomicU64::new(0)],
            paced: [AtomicU64::new(0), AtomicU64::new(0)],
            transit_eighths: [AtomicU64::new(0), AtomicU64::new(0)],
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
    /// The extra slots start lent: they take their memory once a decode has shown what it
    /// leaves (see [`CardStager::room_for_extra`]).
    pub fn with_extra(mut self, extra: usize) -> Result<Self> {
        let n = extra.min(self.n_experts.saturating_sub(self.base_slots));
        if n == 0 {
            return Ok(self);
        }
        let st = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        st.owner.resize(self.base_slots + n, -3);
        self.extra_table = Some(ExtraTable::new(&self.dev)?);
        Ok(self)
    }

    /// The card's address of the extra stacks' entry `stack` (gate 0, up 1, down 2) in the
    /// tier's table, `None` for a tier without extra slots.
    pub(super) fn extra_table_at(&self, stack: usize) -> Option<u64> {
        self.extra_table
            .as_ref()
            .map(|t| t.addr + (stack * std::mem::size_of::<u64>()) as u64)
    }

    /// Writes the addresses of `stacks`, or nulls, into the tier's table on the card.
    fn write_extra_table(&self, stacks: Option<&[QTensor; 3]>) -> Result<()> {
        let Some(t) = self.extra_table.as_ref() else {
            return Ok(());
        };
        let addrs = match stacks {
            Some(s) => [
                s[0].device_ptr()? as u64,
                s[1].device_ptr()? as u64,
                s[2].device_ptr()? as u64,
            ],
            None => [0u64; 3],
        };
        t.write(&self.dev, addrs)
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
        if self
            .extra
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none()
        {
            return false;
        }
        // The copies on their way look their slots up under the stacks' lock: they land before
        // the lock is taken to free the stacks, and one queued after it finds the slots lent.
        self.settle();
        let mut extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
        if extra.is_none() {
            return false;
        }
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
        if let Err(e) = self.write_extra_table(None) {
            tracing::warn!("expert tier: the extra slots' table: {e}");
        }
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
        self.restore_extra_within(u64::MAX)
    }

    /// [`Self::restore_extra`] taking at most `bytes` of the card: as many slots as fit in them.
    pub fn restore_extra_within(self: &Arc<Self>, bytes: u64) -> Result<usize> {
        let slot = self.expert_bytes.iter().sum::<usize>().max(1) as u64;
        let n = self
            .extra_slots()
            .min((bytes / slot).min(usize::MAX as u64) as usize);
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
            self.write_extra_table(Some(&stacks))?;
            *extra = Some(stacks);
            drop(extra);
            self.fill_extra(fit)
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
        self.health_check()?;
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
                blend(&self.upload_us[0], ms as f64 * 1e3 / carried as f64);
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
                crate::tensor::gpu_runtime::driver::sys::cuMemcpyDtoHAsync_v2(
                    dst,
                    src,
                    bytes,
                    stream.cu_stream(),
                )
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
                f64::from_bits(self.upload_us[0].load(Ordering::Relaxed)),
                f64::from_bits(self.host_us[0].load(Ordering::Relaxed)),
                self.decode_us.load(Ordering::Relaxed),
            );
        }
        // The misses the transit room runs, and those the host does.
        let to_transit = self.transit_share(miss_ids.len(), 1);
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
                    crate::tensor::gpu_runtime::driver::sys::cuMemcpyHtoDAsync_v2(
                        dst,
                        src,
                        bytes,
                        stream.cu_stream(),
                    )
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
                more.map(|m| crate::inference::moe_cuda::MoreExperts::Stacks((&m[0], &m[1]))),
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
                crate::tensor::gpu_runtime::driver::sys::cuMemcpyHtoDAsync_v2(
                    address_of::<f32>(&scratch.product)?,
                    h_out.as_ptr() as *const std::ffi::c_void,
                    4 * hidden,
                    stream.cu_stream(),
                )
            })?;
            blend(
                &self.host_us[0],
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
                    crate::tensor::gpu_runtime::driver::sys::cuMemcpyHtoDAsync_v2(
                        dst,
                        src,
                        bytes,
                        stream.cu_stream(),
                    )
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
        self.health_check()?;
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
                blend(
                    &self.upload_us[pace_class(rows)],
                    ms as f64 * 1e3 / carried as f64,
                );
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
        let to_transit = self.transit_share(missing.len(), rows);
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
                &self.host_us[pace_class(rows)],
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
        stream: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
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
        stream: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
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
        stream: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<()> {
        use crate::tensor::gpu_runtime::driver::DevicePtr;
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
                    crate::tensor::gpu_runtime::driver::sys::cuMemcpyDtoDAsync_v2(
                        to,
                        from,
                        eb,
                        stream.cu_stream(),
                    )
                })?;
            }
            self.promotions.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// How many of `missing` experts are copied into the card's transit room for the token, the
    /// rest being computed on the host: the split at which the two finish together, from the
    /// time copies and host computations take on this layer lately. The host does at least one
    /// of several until its time is known.
    fn transit_share(&self, missing: usize, rows: usize) -> usize {
        if missing == 0 || self.stage.is_none() {
            return 0;
        }
        let class = pace_class(rows);
        let room = self.stage.as_ref().map_or(0, |s| s.stager.transit);
        let host = f64::from_bits(self.host_us[class].load(Ordering::Relaxed));
        let mut copy = f64::from_bits(self.upload_us[class].load(Ordering::Relaxed));
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
        // Once the paces are measured, the copies and the products finish together when the room
        // takes host / (host + copy) of the experts, from the paces as they stand now: they move
        // with the host's load and the bus.
        if self.paced[class].load(Ordering::Relaxed) >= TRANSIT_PACE_SAMPLES {
            let share = (8.0 * host / (host + copy)).round().clamp(1.0, 7.0) as u64;
            self.transit_eighths[class].store(share, Ordering::Relaxed);
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
        start: &crate::tensor::gpu_runtime::driver::CudaEvent,
        end: &crate::tensor::gpu_runtime::driver::CudaEvent,
        experts: &[u32],
    ) -> Result<()> {
        use crate::tensor::gpu_runtime::driver::DevicePtr;
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
                self.upload_mapped(i, data, e * eb, eb, base + (at * eb) as u64, copy)?;
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
    pub(super) fn transit_share_of(&self, missing: usize, rows: usize) -> usize {
        self.transit_share(missing, rows)
    }

    /// The transit room's stacks, where the layer has one.
    pub(super) fn transit_stacks(&self) -> Option<&[QTensor; 3]> {
        self.stage.as_ref().map(|s| &s.compact)
    }

    /// Note what one copy into the transit room took, in microseconds, for a request of `rows`.
    pub(super) fn note_upload_us(&self, us: f64, rows: usize) {
        let class = pace_class(rows);
        blend(&self.upload_us[class], us);
        self.paced[class].fetch_add(1, Ordering::Relaxed);
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
            self.upload_mapped(i, data, expert * eb, eb, dst, &self.copy)?;
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
/// Which paces a request of `rows` rows is measured and split by: one row, or several.
fn pace_class(rows: usize) -> usize {
    usize::from(rows > 1)
}

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
fn driver(what: &str, r: crate::tensor::gpu_runtime::driver::sys::cudaError_enum) -> Result<()> {
    if r == crate::tensor::gpu_runtime::driver::sys::cudaError_enum::CUDA_SUCCESS {
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
            decode_need: Mutex::new(None),
            free_after_return: Mutex::new(None),
            lent_since_return: std::sync::atomic::AtomicBool::new(false),
            replay_need: Mutex::new(None),
            recorded_since_return: std::sync::atomic::AtomicBool::new(false),
        }))
    }

    /// Records what the decode since the last return took of the card: the free memory then,
    /// less the free memory now, its transients still in the pools. Called as the slots are to
    /// come back; a span a prefill lent in is the prefill's, and is not counted.
    pub fn note_decode(&self) {
        let then = self
            .free_after_return
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let lent = self.lent_since_return.swap(false, Ordering::AcqRel);
        let Some((then, pool_then)) = then.filter(|_| !lent) else {
            return;
        };
        // What the card lost for good, or what the pool held at the decode's peak beyond what it
        // held at the return, whichever is more: transients go back to the pool, not the card.
        let lost = then.saturating_sub(crate::inference::place::expert_cache::free_now(&self.dev));
        let peak = self
            .dev
            .as_cuda_device()
            .map_or(0, |d| crate::tensor::cuda::pool_peak(d.0.ordinal()))
            .saturating_sub(pool_then);
        let took = lost.max(peak);
        // A decode that recorded its graphs takes their buffers and the recording's transients;
        // one that only replayed graphs kept from before takes what a replay needs.
        let recorded = self.recorded_since_return.swap(false, Ordering::AcqRel);
        let cell = if recorded {
            &self.decode_need
        } else {
            &self.replay_need
        };
        let mut need = cell.lock().unwrap_or_else(|e| e.into_inner());
        if need.is_none_or(|n| took > n) {
            tracing::info!(
                "expert tier: a decode {} took {:.2} GB of the card; the extra slots leave that free",
                if recorded { "that recorded its graphs" } else { "that replayed kept graphs" },
                took as f64 / 1e9
            );
            *need = Some(took);
        }
    }

    /// Marks the span since the last return as one where a graph was recorded.
    pub fn note_recorded(&self) {
        self.recorded_since_return.store(true, Ordering::Release);
    }

    /// The bytes the extra slots may take back now: the card's free memory less what a decode
    /// was seen to take, a replaying one's when `replay` (the decode replays kept graphs and
    /// one was measured), nothing before a decode has been measured. A replay also takes back
    /// what the kept graphs reserved at their peak and gave back to the card since.
    pub fn room_for_extra(&self, replay: bool) -> u64 {
        let replayed = *self.replay_need.lock().unwrap_or_else(|e| e.into_inner());
        let recorded = *self.decode_need.lock().unwrap_or_else(|e| e.into_inner());
        let need = match replay {
            true => replayed
                .map(|n| n + self.graph_reserve_given_back())
                .or(recorded),
            false => recorded,
        };
        match need {
            Some(need) => {
                crate::inference::place::expert_cache::free_now(&self.dev).saturating_sub(need)
            }
            None => 0,
        }
    }

    /// The graph memory the card's recorded graphs reserved at their peak and no longer hold.
    fn graph_reserve_given_back(&self) -> u64 {
        let [_, reserved, _, peak] = self
            .dev
            .as_cuda_device()
            .map_or([0; 4], |d| crate::tensor::cuda::graph_mem(d.0.ordinal()));
        peak.saturating_sub(reserved)
    }

    /// Marks the span since the last return as a prefill's.
    pub fn note_lend(&self) {
        self.lent_since_return.store(true, Ordering::Release);
    }

    /// Records the card's free memory as a prefill ends and its decode is about to run.
    pub fn note_return(&self) {
        let pool = self
            .dev
            .as_cuda_device()
            .map_or(0, |d| crate::tensor::cuda::pool_mark(d.0.ordinal()));
        *self
            .free_after_return
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some((
            crate::inference::place::expert_cache::free_now(&self.dev),
            pool,
        ));
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
    arena: &Arc<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>,
    arena_bytes: usize,
) -> Result<f64> {
    use crate::tensor::gpu_runtime::driver::DevicePtr;
    let cd = dev.as_cuda_device()?;
    cd.0.context()
        .bind_to_thread()
        .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
    let bytes = arena_bytes.min(64 << 20);
    let host = unsafe { crate::tensor::gpu_runtime::driver::result::malloc_host(bytes, 0) }
        .map_err(|e| crate::tensor::Error::msg(format!("pinned probe: {e}")))?;
    let dst = arena.device_ptr(arena.stream()).0;
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let r =
            unsafe { crate::tensor::gpu_runtime::driver::sys::cuMemcpyHtoD_v2(dst, host, bytes) };
        let us = t.elapsed().as_secs_f64() * 1e6;
        if r == crate::tensor::gpu_runtime::driver::sys::cudaError_enum::CUDA_SUCCESS {
            best = best.min(us);
        }
    }
    unsafe { crate::tensor::gpu_runtime::driver::result::free_host(host) }.ok();
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
        crate::tensor::gpu_runtime::driver::sys::cuMemHostRegister_v2(
            lo as *mut std::ffi::c_void,
            hi - lo,
            READ_ONLY,
        )
    };
    (r == crate::tensor::gpu_runtime::driver::sys::cudaError_enum::CUDA_SUCCESS).then_some((lo, hi))
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
            #[cfg(target_os = "linux")]
            bounce: Mutex::new(BounceState::Closed),
            held: std::sync::atomic::AtomicBool::new(false),
            filled: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        for r in self.pinned.iter().flatten() {
            unsafe {
                crate::tensor::gpu_runtime::driver::sys::cuMemHostUnregister(
                    r.0 as *mut std::ffi::c_void,
                )
            };
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
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        for i in 0..3 {
            let eb = self.expert_bytes[i];
            let arena = &buffer.arenas[i];
            let base = arena.device_ptr(arena.stream()).0;
            let data = self.host[i].native_qtensor().data();
            // The kernels read whole blocks past the end of a stack, against activations padded
            // with zeros: that tail is zero, as every blob's is, and not what the layer before
            // left in the arena - a scale in it that is not finite would poison the zeros.
            driver("stage tail", unsafe {
                crate::tensor::gpu_runtime::driver::sys::cuMemsetD8Async(
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
                        crate::tensor::gpu_runtime::driver::sys::cuMemcpyDtoDAsync_v2(
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
        copy: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<()> {
        let (a, b) = (
            data.as_ptr() as usize + start,
            data.as_ptr() as usize + start + len,
        );
        let pinned = self.stage.as_ref().and_then(|s| s.pinned[i]);
        let (lo, hi) = pinned.map_or((b, b), |(lo, hi)| (lo.clamp(a, b), hi.clamp(a, b)));
        let (lo, hi) = if lo < hi { (lo, hi) } else { (b, b) };
        // The locked part is copied from the mapping; the rest goes through `upload_unlocked`.
        self.upload_unlocked(a, lo, a, dst, copy)?;
        self.copy_mapped(lo, hi, a, dst, copy)?;
        self.upload_unlocked(hi, b, a, dst, copy)
    }

    /// Copy `len` bytes of `data` from `start` to the card at `dst`, from the mapping alone: the
    /// copies of a decode, which go out between the card's kernels and take no buffer of their own.
    fn upload_mapped(
        &self,
        i: usize,
        data: &[u8],
        start: usize,
        len: usize,
        dst: u64,
        copy: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<()> {
        let (a, b) = (
            data.as_ptr() as usize + start,
            data.as_ptr() as usize + start + len,
        );
        // Cut at the edges of the stack's locked pages, as `upload` does.
        let pinned = self.stage.as_ref().and_then(|s| s.pinned[i]);
        let (lo, hi) = pinned.map_or((b, b), |(lo, hi)| (lo.clamp(a, b), hi.clamp(a, b)));
        let (lo, hi) = if lo < hi { (lo, hi) } else { (b, b) };
        self.copy_mapped(a, lo, a, dst, copy)?;
        self.copy_mapped(lo, hi, a, dst, copy)?;
        self.copy_mapped(hi, b, a, dst, copy)
    }

    /// Copy the mapped bytes `[from, to)` to the card, where `a` is the address `dst` stands for.
    fn copy_mapped(
        &self,
        from: usize,
        to: usize,
        a: usize,
        dst: u64,
        copy: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<()> {
        if to <= from {
            return Ok(());
        }
        // One copy for each piece within or between the experts held in host memory.
        hot::for_each_piece(from, to, |lo, hi| {
            driver("expert to stage", unsafe {
                crate::tensor::gpu_runtime::driver::sys::cuMemcpyHtoDAsync_v2(
                    dst + (lo - a) as u64,
                    lo as *const std::ffi::c_void,
                    hi - lo,
                    copy.cu_stream(),
                )
            })
        })
    }

    /// Copy the mapped bytes `[from, to)` of a layer that is not page-locked: the parts the page
    /// cache holds from the mapping, the others by direct reads that leave the cache alone.
    fn upload_unlocked(
        &self,
        from: usize,
        to: usize,
        a: usize,
        dst: u64,
        copy: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<()> {
        if to <= from {
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        if let Some(runs) = crate::tensor::mapped::page_size()
            .and_then(|page| direct::residency(from, to - from, page))
        {
            for (f, t, resident) in runs {
                if resident {
                    self.copy_mapped(f, t, a, dst, copy)?;
                } else {
                    self.upload_direct(f, t, a, dst, copy)?;
                }
            }
            return Ok(());
        }
        self.copy_mapped(from, to, a, dst, copy)
    }

    /// The mapped bytes `[from, to)` through the stage's direct reader; what it cannot read comes
    /// from the mapping, after one warning.
    #[cfg(target_os = "linux")]
    fn upload_direct(
        &self,
        from: usize,
        to: usize,
        a: usize,
        dst: u64,
        copy: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<()> {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let Some(stage) = self.stage.as_ref() else {
            return self.copy_mapped(from, to, a, dst, copy);
        };
        let mut done = 0;
        let mut failure = None;
        if let Some((file, off)) = crate::tensor::mapped::file_span(from, to - from) {
            let mut state = stage.bounce.lock().unwrap_or_else(|e| e.into_inner());
            if let BounceState::Open(b) = &*state {
                if !b.reads(&file) {
                    *state = BounceState::Closed;
                }
            }
            if matches!(*state, BounceState::Closed) {
                // A chunk carries at most the largest expert, so a run is read in a few chunks.
                let unit = self.expert_bytes.iter().copied().max().unwrap_or(0);
                let page = crate::tensor::mapped::page_size().unwrap_or(4096);
                *state = match direct::Bounce::open(&file, unit.max(page), page) {
                    Ok(b) => BounceState::Open(b),
                    Err(e) => {
                        failure = Some(e);
                        BounceState::Off
                    }
                };
            }
            if let BounceState::Open(b) = &mut *state {
                let (n, err) = b.upload(off, to - from, dst + (from - a) as u64, copy);
                done = n;
                if err.is_some() {
                    *state = BounceState::Off;
                }
                failure = err;
            }
        }
        if let Some(e) = failure {
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!("direct expert reads off, copying from the mapping: {e}");
            }
        }
        self.copy_mapped(from + done, to, a, dst, copy)
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
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        let cd = stager.dev.as_cuda_device().unwrap();
        cd.0.context().bind_to_thread().unwrap();
        for buffer in &stager.buffers {
            for arena in &buffer.arenas {
                let base = arena.device_ptr(arena.stream()).0;
                let r = unsafe {
                    crate::tensor::gpu_runtime::driver::sys::cuMemsetD8_v2(base, 0xFF, arena.len())
                };
                assert_eq!(
                    r,
                    crate::tensor::gpu_runtime::driver::sys::cudaError_enum::CUDA_SUCCESS
                );
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
        // The stacks are quantised once a shape and kept: the host's and the card's layers of a
        // test read the same ones, which a large shape makes worth sharing.
        type Key = (usize, usize, usize, u64, GgmlDType);
        static STACKS: std::sync::Mutex<Vec<(Key, Arc<QTensor>)>> =
            std::sync::Mutex::new(Vec::new());
        let stack = |seed: u64, rows: usize, cols: usize, dtype: GgmlDType| -> Arc<QTensor> {
            let key = (experts, rows, cols, seed, dtype);
            let mut kept = STACKS.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((_, t)) = kept.iter().find(|(k, _)| *k == key) {
                return t.clone();
            }
            let made = Arc::new(quantised_stack(experts, seed, rows, cols, dtype));
            kept.push((key, made.clone()));
            made
        };
        fn quantised_stack(
            experts: usize,
            seed: u64,
            rows: usize,
            cols: usize,
            dtype: GgmlDType,
        ) -> QTensor {
            let host = Device::Cpu;
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
        }
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
            down_exps: stack(3, hidden, ffn, down),
            gate_exps_sep: Some(stack(1, ffn, hidden, gate_up)),
            up_exps_sep: Some(stack(2, ffn, hidden, gate_up)),
            down_exps_scale: None,
            decode_slots: dev.runs_as_card().then(|| {
                Tensor::from_vec((0..used as u32).collect::<Vec<_>>(), (used,), &host)
                    .unwrap()
                    .to_device(dev)
                    .unwrap()
            }),
            n_experts: experts,
            shards: Vec::new(),
            local: None,
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

    /// Experts split by id range across two cards give the rows of one card holding them all: a
    /// decoded token, a few rows and a prefill wide enough for the expert-batched kernels.
    #[test]
    #[ignore = "needs two idle CUDA cards"]
    fn experts_split_across_two_cards_give_the_rows_of_one_card() {
        use crate::inference::generic_transformer::moe_weights::ExpertShard;
        let (Ok(first), Ok(second)) = (Device::new_cuda(0), Device::new_cuda(1)) else {
            eprintln!("skipping: two usable CUDA devices are needed");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 512usize, 512usize);
        let mut one = layer(&first, experts, used, hidden, ffn);
        one.shards = vec![ExpertShard::from_host(&one, 0, experts, &first).unwrap()];
        let mut two = layer(&first, experts, used, hidden, ffn);
        let half = experts / 2;
        two.shards = vec![
            ExpertShard::from_host(&two, 0, half, &first).unwrap(),
            ExpertShard::from_host(&two, half, experts - half, &second).unwrap(),
        ];
        for (rows, prefill) in [(1usize, false), (5, false), (160, true)] {
            let x = Tensor::from_vec(
                values(rows * hidden, rows as u64, 1.0),
                (1, rows, hidden),
                &first,
            )
            .unwrap();
            let run = |moe: &MoeWeights| -> Vec<f32> {
                moe.forward(&x, &x, hidden, 1e-5, prefill)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            };
            let (want, got) = (run(&one), run(&two));
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 1e-5 * scale,
                "{rows} rows: off by {worst} of {scale}"
            );
        }
    }

    /// The cards of a split layer, each holding a range of the experts and weighing the other
    /// cards' pairs zero, sum to the rows of one card holding them all: a decoded token, a few
    /// rows and a prefill wide enough for the expert-batched kernels.
    #[test]
    #[ignore = "needs two idle CUDA cards"]
    fn the_cards_of_a_split_layer_sum_to_the_rows_of_one_card() {
        use crate::inference::generic_transformer::moe_weights::{ExpertShard, LocalExperts};
        let (Ok(first), Ok(second)) = (Device::new_cuda(0), Device::new_cuda(1)) else {
            eprintln!("skipping: two usable CUDA devices are needed");
            return;
        };
        let (experts, used, hidden, ffn) = (8usize, 2usize, 64usize, 64usize);
        let mut one = layer(&first, experts, used, hidden, ffn);
        one.shards = vec![ExpertShard::from_host(&one, 0, experts, &first).unwrap()];
        let part = |dev: &Device, from: usize, count: usize| {
            let mut m = layer(dev, experts, used, hidden, ffn);
            let range = |s: &Arc<QTensor>| Arc::new(experts_range(s, from, count, dev).unwrap());
            m.gate_exps_sep = m.gate_exps_sep.as_ref().map(range);
            m.up_exps_sep = m.up_exps_sep.as_ref().map(range);
            m.down_exps = range(&m.down_exps);
            m.n_experts = count;
            m.local = Some(LocalExperts {
                first: from,
                total: experts,
            });
            m
        };
        // Uneven ranges, as cards of different speeds take.
        let cut = experts * 2 / 3;
        let parts = [
            (part(&first, 0, cut), &first),
            (part(&second, cut, experts - cut), &second),
        ];
        for (rows, prefill) in [
            (1usize, false),
            (2, false),
            (5, false),
            (2, true),
            (6, true),
            (160, true),
        ] {
            let x = Tensor::from_vec(
                values(rows * hidden, rows as u64, 1.0),
                (1, rows, hidden),
                &first,
            )
            .unwrap();
            let flat = |t: Tensor| -> Vec<f32> {
                t.to_device(&first)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            };
            let want = flat(one.forward(&x, &x, hidden, 1e-5, prefill).unwrap());
            let mut got = vec![0f32; want.len()];
            for (moe, dev) in &parts {
                let x = x.to_device(dev).unwrap();
                for (g, v) in got
                    .iter_mut()
                    .zip(flat(moe.forward(&x, &x, hidden, 1e-5, prefill).unwrap()))
                {
                    *g += v;
                }
            }
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 1e-5 * scale,
                "{rows} rows: off by {worst} of {scale}"
            );
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
        assert!(
            !tier.has_extra(),
            "the extra slots took memory before a decode was measured"
        );
        tier.restore_extra().unwrap();
        tier.settle();
        assert!(tier.has_extra());
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
        recorded_decode_matches_the_host((32, 4, 64, 64), [GgmlDType::Q8_0; 2], 6);
    }

    /// The same in the k-quants and the legacy formats of a served file, at its widths.
    #[test]
    fn a_token_through_the_recorded_decode_matches_the_host_in_k_quants() {
        for formats in [
            [GgmlDType::Q4K, GgmlDType::Q5_0],
            [GgmlDType::Q4K, GgmlDType::Q8_0],
        ] {
            recorded_decode_matches_the_host((32, 8, 2560, 640), formats, 12);
        }
    }

    fn recorded_decode_matches_the_host(
        (experts, used, hidden, ffn): (usize, usize, usize, usize),
        formats: [GgmlDType; 2],
        slots: usize,
    ) {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let reference = layer_of(&Device::Cpu, experts, used, hidden, ffn, formats);
        let mut tiered = layer_of(&card, experts, used, hidden, ffn, formats);
        let built = LayerTier::new(&tiered, &card, slots).unwrap();
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(hidden, used, 4).unwrap();
        let worker = TierWorker::start(&card, vec![("layer 0".into(), tier.clone())]).unwrap();
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
                "{formats:?} token {token}: max diff {worst} of {peak}"
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

    /// A worker that never answers a recorded token's request releases the card within the
    /// watchdog's bound: the token's output stands untrusted, the next token fails with the
    /// stall, and the layer serves again once the worker answers.
    #[test]
    fn a_stalled_worker_fails_the_token_within_the_bound() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        tiered.tier = Some(Arc::new(LayerTier::new(&tiered, &card, 2).unwrap()));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(hidden, used, 1).unwrap();
        let named = vec![("layer 0".to_string(), tier.clone())];
        let worker = TierWorker::start(&card, named.clone()).unwrap();
        GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
        let x = Tensor::from_vec(values(hidden, 7, 1.0), (1, 1, hidden), &card).unwrap();
        let run = || {
            tiered
                .forward(&x, &x, hidden, 1e-5, false)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
        };
        run().unwrap();
        let bound = std::time::Duration::from_nanos(worker.stall_bound(&named).unwrap());
        graph::STALL_FOR_TEST.store(true, Ordering::Release);
        let started = std::time::Instant::now();
        let released = run();
        let took = started.elapsed();
        let failure = worker.failure();
        let next = run();
        graph::STALL_FOR_TEST.store(false, Ordering::Release);
        let deadline = std::time::Instant::now() + 4 * bound + std::time::Duration::from_secs(1);
        while tier.health_check().is_err() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let again = run();
        GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
        drop(worker);
        assert!(released.is_ok(), "the released token's own launches ran");
        assert!(
            took < 2 * bound + std::time::Duration::from_secs(1),
            "the card waited {took:?} past a bound of {bound:?}"
        );
        assert!(
            failure.as_deref().is_some_and(|f| f.contains("layer 0")),
            "the stall names its layer: {failure:?}"
        );
        assert!(next.is_err(), "a token after the stall fails");
        let again = again.expect("the layer serves again once the worker answers");
        assert!(again.iter().all(|v| v.is_finite()));
    }

    /// A decoded token recorded in a graph while the extra slots hold experts replays what the
    /// same kernels give run eagerly, to the bit, after the extra slots were lent and came back
    /// at other addresses: the recording reads the extra stacks through the tier's table.
    #[test]
    fn a_recorded_token_follows_the_extra_slots_when_they_come_back_elsewhere() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let (base, extra) = (2usize, 8usize);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let built = LayerTier::new(&tiered, &card, base)
            .unwrap()
            .with_extra(extra)
            .unwrap();
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(hidden, used, 1).unwrap();
        let worker = TierWorker::start(&card, vec![("layer 0".into(), tier.clone())]).unwrap();
        let stream = card.as_cuda_device().unwrap().0.stream().clone();
        let x = Tensor::from_vec(values(hidden, 7, 1.0), (1, 1, hidden), &card).unwrap();
        let held = |tier: &LayerTier| {
            let st = tier.state.lock().unwrap();
            (tier.base_slots..st.owner.len())
                .filter(|&s| st.owner[s] >= 0)
                .count()
        };
        let reset_counters = |tier: &LayerTier| {
            tier.routed.store(0, Ordering::Relaxed);
            tier.hits.store(0, Ordering::Relaxed);
        };
        let eager = |tiered: &MoeWeights| -> Vec<f32> {
            GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
            let out = tiered.forward(&x, &x, hidden, 1e-5, false).unwrap();
            GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
            let out = out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            tier.settle();
            out
        };
        // The token's experts heat up and fill the base slots, then the extra slots take the
        // others: every expert the token routes to is on the card, so nothing moves after.
        for _ in 0..8 {
            eager(&tiered);
        }
        reset_counters(&tier);
        tier.restore_extra().unwrap();
        tier.settle();
        assert!(held(&tier) > 0, "no expert reached the extra slots");
        let want = eager(&tiered);
        assert_eq!(want, eager(&tiered), "the eager token is not stable");
        let into = Tensor::zeros_on((1, 1, hidden), DType::F32, &card).unwrap();
        crate::tensor::cuda_ext::begin_capture(&stream).unwrap();
        let ran = tiered
            .forward(&x, &x, hidden, 1e-5, false)
            .and_then(|out| into.slice_set(&out.to_dtype(DType::F32)?, 0, 0));
        let graph = crate::tensor::cuda_ext::end_capture(&stream);
        ran.unwrap();
        let graph = graph.unwrap().expect("a recorded token");
        let replay = || -> Vec<f32> {
            graph.launch().unwrap();
            stream.synchronize().unwrap();
            tier.settle();
            into.flatten_all().unwrap().to_vec1().unwrap()
        };
        assert_eq!(replay(), want, "the replay differs from the eager run");
        let table = || {
            let extra = tier.extra.lock().unwrap();
            extra.as_ref().map(|s| s[0].device_ptr().unwrap() as u64)
        };
        let before = table().expect("the extra slots hold memory");
        // Stacks held where the lent ones were, so that the slots come back elsewhere.
        let mut blockers = Vec::new();
        let mut moved = false;
        for _ in 0..4 {
            assert!(tier.lend_extra());
            blockers.push([0, 1, 2].map(|i| lent_experts(&tier.host[i], extra, &card).unwrap()));
            reset_counters(&tier);
            tier.restore_extra().unwrap();
            tier.settle();
            if table().is_some_and(|a| a != before) {
                moved = true;
                break;
            }
        }
        assert!(moved, "the extra slots came back where they were");
        assert!(held(&tier) > 0, "no expert came back to the extra slots");
        let want = eager(&tiered);
        assert_eq!(
            replay(),
            want,
            "the replay after the slots moved differs from the eager run"
        );
        drop(blockers);
        drop(worker);
    }

    /// The extra slots lent and given back leave the model's graph generation where it was:
    /// the recorded graphs read them through the tier's table, which never moves.
    #[test]
    fn lending_and_restoring_the_extra_slots_leave_the_graph_generation() {
        use crate::inference::generic_transformer::tiny_model;
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let path = tiny_model::write_moe("tier-generation", 1);
        let mut model = tiny_model::load(&path, 1);
        tiny_model::remove(&path);
        let moe = layer(&card, 32, 4, 64, 64);
        let tier = Arc::new(
            LayerTier::new(&moe, &card, 2)
                .unwrap()
                .with_extra(8)
                .unwrap(),
        );
        model.layers[0].moe.as_mut().expect("a mixture layer").tier = Some(tier.clone());
        let at = model.graph_generation();
        let table = tier
            .extra_table_at(0)
            .expect("a tier with extra slots has a table");
        tier.restore_extra().unwrap();
        tier.settle();
        assert!(tier.has_extra());
        assert!(tier.lend_extra());
        model.return_expert_cache().unwrap();
        tier.restore_extra().unwrap();
        tier.settle();
        assert!(tier.has_extra());
        assert_eq!(tier.extra_table_at(0), Some(table), "the table moved");
        assert_eq!(model.graph_generation(), at, "the generation moved");
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
        let worker = TierWorker::start(&card, vec![("layer 0".into(), tier.clone())]).unwrap();
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
    /// A request every expert of which is cached is not waited for, so the late pairs the
    /// host handed for an earlier request must not be read again: the card clears their
    /// count as it plans.
    #[test]
    fn a_cached_request_reads_no_late_pairs_of_the_request_before() {
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
        tier.enable_graph(hidden, used, 4).unwrap();
        let worker = TierWorker::start(&card, vec![("layer 0".into(), tier.clone())]).unwrap();
        GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
        let x = values(hidden, 100, 1.0);
        let on_host = Tensor::from_vec(x.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
        let on_card = Tensor::from_vec(x, (1, 1, hidden), &card).unwrap();
        let want = reference
            .forward(&on_host, &on_host, hidden, 1e-5, false)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // The same token until its experts are all cached, then once more with late pairs
        // left in the mailbox as a request with misses leaves them.
        let mut cached = false;
        for _ in 0..12 {
            let (routed, hits, _) = tier.counters();
            let _ = tiered
                .forward(&on_card, &on_card, hidden, 1e-5, false)
                .unwrap();
            tier.settle();
            let (routed_after, hits_after, _) = tier.counters();
            if routed_after - routed == hits_after - hits {
                cached = true;
                break;
            }
        }
        assert!(cached, "the token's experts never all came to the card");
        tier.hand_late_for_test(&[(0, 0, 1.0), (1, 0, 1.0)]);
        let got = tiered
            .forward(&on_card, &on_card, hidden, 1e-5, false)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        tier.settle();
        GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
        let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 2e-2 * peak, "max diff {worst} of {peak}");
        drop(worker);
    }

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
        let worker = TierWorker::start(&card, vec![("layer 0".into(), tier.clone())]).unwrap();
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
        resident_layer_matches_the_host((32, 4, 64, 64), [GgmlDType::Q8_0; 2]);
    }

    /// The same in the k-quants and the legacy formats of a served file, at its widths.
    #[test]
    fn a_resident_layer_matches_the_host_in_k_quants() {
        for formats in [
            [GgmlDType::Q4K, GgmlDType::Q5_0],
            [GgmlDType::Q4K, GgmlDType::Q8_0],
        ] {
            resident_layer_matches_the_host((32, 8, 2560, 640), formats);
        }
    }

    fn resident_layer_matches_the_host(
        (experts, used, hidden, ffn): (usize, usize, usize, usize),
        formats: [GgmlDType; 2],
    ) {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let reference = layer_of(&Device::Cpu, experts, used, hidden, ffn, formats);
        let mut resident = layer_of(&card, experts, used, hidden, ffn, formats);
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
                worst.is_finite() && worst <= 2e-2 * peak,
                "{formats:?} rows {rows}: max diff {worst} of {peak}"
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
                let on_card = Tensor::from_vec(x.clone(), (1, rows, hidden), &card).unwrap();
                // Rows route and multiply alone, so the host answers for the first and last rows
                // of the batch only, while the card runs the whole batch.
                let sample: Vec<usize> = (0..rows).filter(|&r| r < 32 || r + 32 >= rows).collect();
                let picked: Vec<f32> = sample
                    .iter()
                    .flat_map(|&r| x[r * hidden..(r + 1) * hidden].iter().copied())
                    .collect();
                let on_host =
                    Tensor::from_vec(picked, (1, sample.len(), hidden), &Device::Cpu).unwrap();
                let want = reference
                    .forward(&on_host, &on_host, hidden, 1e-5, true)
                    .unwrap();
                let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let got = tiered
                    .forward(&on_card, &on_card, hidden, 1e-5, true)
                    .unwrap();
                let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let got: Vec<f32> = sample
                    .iter()
                    .flat_map(|&r| got[r * hidden..(r + 1) * hidden].iter().copied())
                    .collect();
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
