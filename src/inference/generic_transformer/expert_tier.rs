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
//! The split holds for the formats with exact kernels, where a pair gives the same bits on the
//! card and on the host. A layer in another format runs every missing expert through the
//! transit room, with the kernels its slots run: a pair's row is the same bits wherever its
//! expert sits, and a row's pairs are summed in the router's order.
//!
//! A card with room for every expert of its layers holds them all: the layer is resident, and the
//! tier is the stacks the kernels read, with no host in the token.
//!
//! A wide batch runs through the expert-batched kernels, which read the slots where they are.
//! The experts its rows route to and the card misses are copied into the room of a wide batch,
//! and nothing else is.
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
    /// The times each expert was routed to, undecayed, from the profile read at the load on:
    /// what the next save writes.
    counts: Vec<u32>,
    /// Tokens since the counts last halved.
    since_decay: usize,
    /// The card's copy of `slot_of`, while a recorded decode reads it.
    table: Option<Arc<graph::SlotTable>>,
    /// Each expert's route count when a promotion gave it the slot it holds.
    /// `None` once the expert left that slot, and for an expert placed any other way.
    admitted: Vec<Option<u32>>,
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
        self.admitted[expert] = None;
        if let Some(t) = &self.table {
            t.set(expert, slot);
        }
    }
}

pub(crate) mod direct;
mod exact;
#[cfg(feature = "microbench")]
pub use exact::bench::{ExactPairs, HostPairs, Projection};
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
    /// Experts a row routes to.
    per_row: usize,
    state: Mutex<TierState>,
    hits: AtomicU64,
    routed: AtomicU64,
    promotions: AtomicU64,
    /// The layer that reports the cache's hit rate.
    leader: std::sync::atomic::AtomicBool,
    /// The rooms a missing expert passes through: the transit room of a decode, and the room of
    /// a wide batch.
    stage: Option<Stage>,
    /// The cache holds every expert, expert `e` in slot `e`: nothing is ever missing.
    resident: bool,
    /// While a prompt holds the layer's room: the routed experts of its chunks so far and the
    /// stream they ran on, ranked into the cache once, when the hold ends.
    prompt_ids: Mutex<
        Option<(
            Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
            Vec<u32>,
        )>,
    >,
    /// Experts a wide batch copied into the room of a wide batch, since the load.
    to_room: AtomicU64,
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
    /// The history of this layer's one-row requests: what they shared, what a guess of the
    /// next one got right, and the lead the layer before leaves.
    ahead: Mutex<super::expert_ahead::Ahead>,
    /// The layer the worker serves after this one, which this layer's request may fetch for.
    ahead_next: std::sync::OnceLock<std::sync::Weak<LayerTier>>,
    /// Experts fetched for this layer before its request, and those its request used.
    fetched_ahead: AtomicU64,
    used_ahead: AtomicU64,
    /// The layer's side of a decode recorded in a graph, once enabled.
    graph: std::sync::OnceLock<graph::GraphSide>,
    /// The layer's formats and widths when the card, the transit room and the host compute its
    /// experts in the same steps; `None` for the formats without an exact kernel.
    exact: Option<exact::Exact>,
    /// The card's admission by measured payback, shared by its cached layers.
    admission: std::sync::OnceLock<Arc<super::expert_admission::Admission>>,
    /// The card's read-ahead of the experts its layers are expected to miss, shared by its
    /// cached layers. Set only where a host tier keeps part of the experts.
    readahead: std::sync::OnceLock<Arc<super::expert_readahead::ReadAhead>>,
    /// The last cached layer of the card: a one-row request here ends the token's span.
    closer: std::sync::atomic::AtomicBool,
    /// The level gives up the same bits for every cache: a layer without exact kernels splits
    /// its missing experts between the transit room and the host by their paces.
    splits_misses: std::sync::atomic::AtomicBool,
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
        Self(crate::inference::stage_prof::stage_prof_enabled().then(std::time::Instant::now))
    }

    /// The time since the last mark is the stage `name`'s.
    fn mark(&mut self, name: &'static str) {
        if let Some(t) = self.0 {
            crate::inference::stage_prof::prof_record(name, t.elapsed().as_nanos() as u64);
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
    /// What each buffer holds, for the layer it was filled for.
    held: [Mutex<super::expert_ahead::Held>; 2],
    /// The room of a wide batch, wide enough for a layer's whole stacks, which holds the experts
    /// its rows route to and the card misses: from the memory the caches lend while a prefill
    /// runs.
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

/// [`StageBuffer`] wide enough for the whole stacks of the widest layer: the room of a wide
/// batch, which holds the experts its rows route to and the card misses.
struct FullBuffer {
    arenas: [Arc<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>; 3],
    /// Recorded when the layer that read the buffer last has queued all its kernels: the next
    /// copy into the buffer waits for it.
    free: Mutex<Option<crate::tensor::gpu_runtime::driver::CudaEvent>>,
}

/// A layer's view of its buffer: the three stacks as tensors over the arenas, and the copies in
/// flight into them.
struct Stage {
    stager: Arc<CardStager>,
    buffer: usize,
    /// The transit room as stacks of `transit` experts.
    compact: [QTensor; 3],
    /// The room of a wide batch as stacks of the layer's shape, while there is one.
    full: Mutex<Option<Arc<[QTensor; 3]>>>,
    /// The experts in the room of a wide batch, in the order of their places there.
    /// Empty while the room holds nothing of this layer.
    placed: Mutex<Vec<u32>>,
    /// Host ranges, per stack, that are page-locked: copies from them are asynchronous.
    pinned: [Option<(usize, usize)>; 3],
    /// The direct reader for the parts of an upload the page cache does not hold.
    #[cfg(target_os = "linux")]
    bounce: Mutex<BounceState>,
    /// The room keeps its experts for several forwards in a row (see `hold_stacks`).
    held: std::sync::atomic::AtomicBool,
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
        // The read-ahead's reader count, measured once for every card.
        let mut readers: Option<usize> = None;
        let mut ordinals: Vec<_> = cuda_devices.keys().copied().collect();
        ordinals.sort_unstable();
        for ordinal in ordinals {
            let dev = &cuda_devices[&ordinal];
            let mut streams = CopyStreams::new(dev);
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
                    moe.tier = Some(Arc::new(LayerTier::all_on_card_on(
                        moe,
                        dev,
                        streams.take()?,
                    )?));
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
            // A layer without exact kernels runs every missing expert of a token through the
            // transit room below the fast level: the room holds a token's pairs.
            let least = tiered
                .iter()
                .filter_map(|&l| self.layers[l].moe.as_ref())
                .filter(|m| {
                    let stacks = [
                        m.gate_exps_sep.clone().expect("separate stacks"),
                        m.up_exps_sep.clone().expect("separate stacks"),
                        m.down_exps.clone(),
                    ];
                    exact::Exact::of(&stacks, m.swiglu_limit.unwrap_or(0.0)).is_none()
                })
                .map(|m| m.n_experts_used)
                .max()
                .unwrap_or(0);
            let stager = CardStager::holding(dev, widest, widest_mixture, least)?;
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
            // One admission for the card's cached layers: a token crosses each of them once.
            let admission = Arc::new(super::expert_admission::Admission::new(tiered.len()));
            let mut readahead: Option<Arc<super::expert_readahead::ReadAhead>> = None;
            for (rank, &l) in tiered.iter().enumerate() {
                let moe = self.layers[l]
                    .moe
                    .as_mut()
                    .expect("a tiered layer has experts");
                let mut tier =
                    LayerTier::new_on(moe, dev, slots, streams.take()?)?.with_extra(extra)?;
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
                if tiered.last() == Some(&l) {
                    tier.closer.store(true, Ordering::Relaxed);
                }
                let _ = tier.admission.set(admission.clone());
                // Where a host tier keeps part of the experts, the card reads ahead from the
                // disk the experts its layers are expected to miss.
                if per_expert {
                    let ra = readahead.get_or_insert_with(|| {
                        let n = *readers.get_or_insert_with(|| read_ahead_readers(&tier));
                        Arc::new(super::expert_readahead::ReadAhead::new(n, tiered.len()))
                    });
                    let _ = tier.readahead.set(ra.clone());
                }
                let tier = Arc::new(tier);
                if per_expert {
                    hot_tiers.push((l, tier.clone()));
                }
                moe.tier = Some(tier);
                // The buffers the layer's copies freed go back to the kernel past the burst
                // headroom.
                crate::inference::place::heap::release_retained();
            }
            if let Some(ra) = &readahead {
                tracing::info!(
                    "expert tier on card {ordinal}: {} threads read the experts the next layer is expected to miss from the disk, while a token runs faster with them",
                    ra.readers()
                );
            }
            // The head's layer runs as one loaded on the card: every expert there, no host
            // answer to wait for at a guess.
            if draft_bytes.is_some() {
                if let Some(moe) = self.draft.as_mut().and_then(|d| d.layer.moe.as_mut()) {
                    moe.tier = Some(Arc::new(LayerTier::all_on_card_on(
                        moe,
                        dev,
                        streams.take()?,
                    )?));
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
                    t.enable_graph(moe, crate::inference::engine::decode_step::VERIFY_ROWS)?;
                }
            }
            self.tier_workers.push(TierWorker::start(dev, arcs)?);
            tracing::info!(
                "expert tier on card {ordinal}: {} layers{} ({locked} with locked host pages, {:.1} GB/s to the card, {} experts in the transit room, {} copy streams), {} experts each cached of {} ({extra} of them lent to a prefill), {:.2} GB",
                tiered.len(),
                if draft_bytes.is_some() { " and the draft head" } else { "" },
                stager.bandwidth * 1e-3,
                stager.transit,
                streams.len(),
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
    /// Wait for the locks of the hot experts a re-rank is moving, before a request runs.
    pub fn settle_hot_experts(&self) {
        if let Some(h) = &self.hot_locks {
            h.settle();
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

    /// What the caches of the experts may give up at the level `quality`: below the fast level a
    /// layer without exact kernels runs every routed pair through the card's kernels, so that
    /// its output is the same bits whatever its cache holds.
    pub fn set_expert_quality(
        &mut self,
        quality: crate::inference::engine::llm_engine::QualityLevel,
    ) {
        use crate::inference::engine::llm_engine::QualityLevel;
        let splits = quality != QualityLevel::Exact;
        for tier in self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
        {
            tier.splits_misses.store(splits, Ordering::Relaxed);
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
        // A prompt that routes more pairs than a layer has experts runs through the expert-batched
        // kernels: the experts it misses come to the room of a wide batch, which the card has to
        // find.
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
        // The room of a wide batch is taken from what the card has free, and a chunk's
        // activations come out of what is left. A card that cannot give both keeps the room
        // unmade: the prompt then reads chunk by chunk, each bringing the experts it routes
        // to, where a room that left no memory for a chunk would fail whatever the chunk's size.
        let free_now = crate::inference::place::vram_manager::tightest_free_on_cards(shape.cards);
        let smallest_chunk = ((crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS
            * shape.bytes_per_token
            + shape.fixed_bytes) as u64)
            .saturating_mul(shape.share.max(1) as u64);
        let made = self
            .card_stagers()
            .iter()
            .all(|s| s.full.lock().unwrap_or_else(|e| e.into_inner()).is_some());
        let wide = wide && (made || free_now >= room.saturating_add(smallest_chunk));
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
        // A model with no expert tier lent nothing: the pools and the residency stay as they are.
        if !self
            .layers
            .iter()
            .any(|l| l.moe.as_ref().is_some_and(|m| m.tier.is_some()))
        {
            return Ok(0);
        }
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

    /// The layer's stacks over the room of a wide batch, which is made if it is not.
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
            stage
                .placed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
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
///
/// A SwiGLU bounded by `swiglu_limit` is served only in a format the exact kernels read.
/// The card, the transit room and the host then all run it through those kernels.
pub fn serves(moe: &MoeWeights) -> bool {
    let (Some(gate), Some(up)) = (moe.gate_exps_sep.as_ref(), moe.up_exps_sep.as_ref()) else {
        return false;
    };
    let kernels = |t: &QTensor| {
        crate::inference::moe_cuda::gemm::expert_kernels_serve(t.dtype())
            && crate::inference::moe_cpu::cpu_qmatmul_supported(t.dtype())
    };
    let bound_served = moe.swiglu_limit.is_none_or(|limit| {
        exact::Exact::of(&[gate.clone(), up.clone(), moe.down_exps.clone()], limit).is_some()
    });
    bound_served
        && moe.use_silu
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

/// Reader threads for the read-ahead of `tier`'s experts: the fewest that reach the rate of the
/// device its file is on, measured now over the host's cores, else the experts a row routes to.
/// A copy stream of its own on the card `dev`.
fn new_copy_stream(dev: &Device) -> Result<Arc<crate::tensor::gpu_runtime::driver::CudaStream>> {
    dev.as_cuda_device()?
        .0
        .context()
        .new_stream()
        .map_err(|e| crate::tensor::Error::msg(format!("copy stream: {e:?}")))
}

/// The copy streams the tiers of one card copy their experts on.
///
/// A stream takes host memory of its own in some runtimes.
/// The first one measures it: the process's anonymous memory before and after.
/// A layer gets a stream of its own while the host's room holds one more.
/// Past that, the layers take the card's streams in turn.
struct CopyStreams {
    dev: Device,
    streams: Vec<Arc<crate::tensor::gpu_runtime::driver::CudaStream>>,
    cost: u64,
    next: usize,
}

impl CopyStreams {
    fn new(dev: &Device) -> Self {
        Self {
            dev: dev.clone(),
            streams: Vec::new(),
            cost: 0,
            next: 0,
        }
    }

    /// The stream the next layer copies on.
    fn take(&mut self) -> Result<Arc<crate::tensor::gpu_runtime::driver::CudaStream>> {
        use crate::inference::place::layer_executor::{
            host_spill_budget_bytes, process_anon_bytes,
        };
        if self.streams.is_empty() {
            let before = process_anon_bytes();
            let stream = new_copy_stream(&self.dev)?;
            self.cost = match (before, process_anon_bytes()) {
                (Some(b), Some(a)) => a.saturating_sub(b),
                _ => 0,
            };
            self.streams.push(stream.clone());
            return Ok(stream);
        }
        if own_stream_fits(self.cost, host_spill_budget_bytes()) {
            let stream = new_copy_stream(&self.dev)?;
            self.streams.push(stream.clone());
            return Ok(stream);
        }
        let stream = self.streams[self.next % self.streams.len()].clone();
        self.next += 1;
        Ok(stream)
    }

    /// The streams made, for the log.
    fn len(&self) -> usize {
        self.streams.len()
    }
}

/// Whether one more stream of `cost` host bytes fits in the host's `room`.
fn own_stream_fits(cost: u64, room: u64) -> bool {
    cost <= room
}

fn read_ahead_readers(tier: &LayerTier) -> usize {
    let data = tier.host[0].native_qtensor().data();
    let unit = tier.expert_bytes.iter().sum::<usize>();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    // Each reader the probe tries holds 1 expert in host memory.
    // Its rounds double the readers up to the cores, so they hold at most twice the cores' worth.
    // A host without that room takes the count without a probe.
    let room = {
        use crate::inference::place::layer_executor::{
            host_burst_headroom_bytes, host_spill_budget_bytes,
        };
        host_spill_budget_bytes().min(host_burst_headroom_bytes())
    };
    if (2 * cores as u64).saturating_mul(unit as u64) > room {
        return tier.per_row;
    }
    crate::tensor::mapped::file_span(data.as_ptr() as usize, data.len())
        .and_then(|(file, _)| super::expert_readahead::probe_readers(&file, unit, cores))
        .unwrap_or(tier.per_row)
}

impl LayerTier {
    /// A cache of `slots` experts on `dev` for the layer `moe`, filled with its first experts,
    /// copied on a stream of its own.
    pub fn new(moe: &MoeWeights, dev: &Device, slots: usize) -> Result<Self> {
        Self::new_on(moe, dev, slots, new_copy_stream(dev)?)
    }

    /// [`Self::new`] with its copies on `copy`, a stream other layers may share.
    pub fn new_on(
        moe: &MoeWeights,
        dev: &Device,
        slots: usize,
        copy: Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<Self> {
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
        let exact = exact::Exact::of(&host, moe.swiglu_limit.unwrap_or(0.0));
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
            per_row: moe.n_experts_used,
            state: Mutex::new(TierState {
                slot_of,
                owner,
                heat: vec![0.0; moe.n_experts],
                counts: vec![0; moe.n_experts],
                since_decay: 0,
                table: None,
                admitted: vec![None; moe.n_experts],
            }),
            hits: AtomicU64::new(0),
            routed: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            leader: std::sync::atomic::AtomicBool::new(false),
            stage: None,
            resident: false,
            copy,
            promoting: std::sync::atomic::AtomicBool::new(false),
            decode_us: AtomicU64::new(0),
            scratch: Mutex::new(None),
            prompt_ids: Mutex::new(None),
            to_room: AtomicU64::new(0),
            upload_us: [AtomicU64::new(0), AtomicU64::new(0)],
            host_us: [AtomicU64::new(0), AtomicU64::new(0)],
            paced: [AtomicU64::new(0), AtomicU64::new(0)],
            transit_eighths: [AtomicU64::new(0), AtomicU64::new(0)],
            uploaded: AtomicU64::new(0),
            ahead: Default::default(),
            ahead_next: std::sync::OnceLock::new(),
            fetched_ahead: AtomicU64::new(0),
            used_ahead: AtomicU64::new(0),
            graph: std::sync::OnceLock::new(),
            exact,
            admission: std::sync::OnceLock::new(),
            readahead: std::sync::OnceLock::new(),
            closer: std::sync::atomic::AtomicBool::new(false),
            splits_misses: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// A layer whose every expert is on the card `dev`: no copy room, no host.
    pub fn all_on_card(moe: &MoeWeights, dev: &Device) -> Result<Self> {
        Self::all_on_card_on(moe, dev, new_copy_stream(dev)?)
    }

    /// [`Self::all_on_card`] with its copies on `copy`, a stream other layers may share.
    pub fn all_on_card_on(
        moe: &MoeWeights,
        dev: &Device,
        copy: Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<Self> {
        let mut tier = Self::new_on(moe, dev, moe.n_experts, copy)?;
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

    /// Adds to `secs[m]` the time this layer spends on the experts that `m` rows route to and
    /// the card does not hold: the expected count from the routing heat and what is resident,
    /// at the pace a missing expert is brought in (the copy and the host's own product run
    /// side by side). Returns whether the layer streams experts at all.
    pub fn add_miss_secs(&self, secs: &mut [f64]) -> bool {
        let Some(stage) = self.stage.as_ref() else {
            return false;
        };
        if self.resident {
            return false;
        }
        let pace = |cells: &[AtomicU64; 2]| {
            [1, 0]
                .iter()
                .map(|&c| f64::from_bits(cells[c].load(Ordering::Relaxed)))
                .find(|&v| v > 0.0)
        };
        let copy = pace(&self.upload_us).unwrap_or_else(|| {
            let rate = stage.stager.bandwidth;
            if rate > 0.0 {
                self.expert_bytes.iter().sum::<usize>() as f64 / rate
            } else {
                0.0
            }
        });
        let per_expert_us = match pace(&self.host_us) {
            Some(host) if copy > 0.0 => copy * host / (copy + host),
            _ => copy,
        };
        if per_expert_us <= 0.0 {
            return true;
        }
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for (m, cell) in secs.iter_mut().enumerate() {
            let missing = super::expert_ahead::missing_after(m, self.per_row, &st.heat, |e| {
                st.slot_of[e] >= 0
            });
            *cell += missing * per_expert_us * 1e-6;
        }
        true
    }

    /// How many experts the layer routes to.
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Fill the slots with the experts `counts` ranks highest, hottest first, copied from the
    /// host stacks now, and give each expert the heat its rank earns, so that the decode's
    /// first routes do not move what the profile placed. The profile's counts are added to the
    /// layer's, so the next save carries them. How many experts were copied.
    pub fn prime(&self, counts: &[u32]) -> Result<usize> {
        if counts.len() != self.n_experts {
            return Ok(0);
        }
        super::expert_profile::add_counts(
            &mut self.state.lock().unwrap_or_else(|e| e.into_inner()).counts,
            counts,
        );
        if self.resident {
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
        let prof = crate::inference::stage_prof::stage_prof_enabled();
        let t_wait = prof.then(std::time::Instant::now);
        stream
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("choice to host: {e:?}")))?;
        if let Some(t) = t_wait {
            crate::inference::stage_prof::prof_record(
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
                super::expert_profile::count_route(&mut st.counts, e as usize);
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
        self.note_served(&ids_h, hits, &miss_ids);
        // The leader says how the cache is doing, once per window of tokens as wide as the
        // experts it chooses among.
        let window = (self.n_experts * k) as u64;
        if self.leader.load(Ordering::Relaxed) && routed % window < k as u64 {
            let (_, h, p) = self.counters();
            tracing::info!(
                "expert tier layer sample: {:.1}% of {} routed experts on the card, {} copies, {} to transit, {:.0} us copy {:.0} us host per expert, {} us a token in the layer, {:.0} routes take a slot",
                100.0 * h as f64 / routed as f64,
                routed,
                p,
                self.uploaded.load(Ordering::Relaxed),
                f64::from_bits(self.upload_us[0].load(Ordering::Relaxed)),
                f64::from_bits(self.host_us[0].load(Ordering::Relaxed)),
                self.decode_us.load(Ordering::Relaxed),
                self.heat_to_take_a_slot(),
            );
        }
        self.mark_token();
        self.read_next_ahead();
        // The misses the transit room runs, and those the host does.
        let to_transit = self.transit_share(miss_ids.len(), 1);
        let (transit_ids, host_ids) = miss_ids.split_at(to_transit);
        let (transit_w, host_w) = miss_weights.split_at(to_transit);
        laps.mark("tier token: partition");
        if let Some(exact) = self.exact {
            let slots: Vec<Option<u32>> =
                (0..k).map(|i| is_hit[i].then_some(slot_ids[i])).collect();
            let out = self.exact_eager(
                &exact,
                scratch,
                xs,
                h_x,
                &w_card,
                exact::Partition {
                    ids: &ids_h,
                    weights: &w_h,
                    slots: &slots,
                    transit: transit_ids,
                },
                &mut laps,
            )?;
            drop(guard);
            self.keep_in_transit(transit_ids, None, &mut in_use, &stream)?;
            self.promote(host_ids, &in_use);
            laps.mark("tier token: finish");
            return Ok(out);
        }

        // Without exact kernels and below the fast level, every pair runs on the card, its
        // missing experts through transit. A layer without a transit room runs them all on the
        // host, and the cache still learns from the misses.
        if !self.splits_misses.load(Ordering::Relaxed) && self.stage.is_none() {
            drop(guard);
            let x = xs.to_device(&host)?;
            let out = moe.forward_cpu(&x, &ids_card, &w_card, 1, hidden)?;
            self.promote(&miss_ids, &in_use);
            laps.mark("tier token: finish");
            return out.to_device(&dev);
        }
        if !self.splits_misses.load(Ordering::Relaxed) {
            let slots: Vec<Option<u32>> =
                (0..k).map(|i| is_hit[i].then_some(slot_ids[i])).collect();
            let (out, rounds) =
                self.pairs_on_card(moe, scratch, xs, &ids_h, &w_card, &slots, &mut laps)?;
            drop(guard);
            self.keep_rounds(&rounds, &mut in_use, &stream, ADAPT_SWAPS_A_REQUEST)?;
            laps.mark("tier token: finish");
            return Ok(out);
        }

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
            crate::inference::stage_prof::prof_record(
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
        self.keep_in_transit(transit_ids, None, &mut in_use, &stream)?;
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
                super::expert_profile::count_route(&mut st.counts, e as usize);
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
        let missed: Vec<u32> = (0..pairs)
            .filter(|&p| !is_hit[p])
            .map(|p| ids_h[p])
            .collect();
        self.note_served(&ids_h, hits, &missed);
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
        if let Some(exact) = self.exact {
            let slots: Vec<Option<u32>> = (0..pairs)
                .map(|p| is_hit[p].then_some(slot_ids[p]))
                .collect();
            let x_host: Vec<f32> = xs
                .to_device(&host)?
                .to_dtype(crate::tensor::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let w_card = weights.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
            let out = self.exact_eager(
                &exact,
                scratch,
                xs,
                &x_host,
                &w_card,
                exact::Partition {
                    ids: &ids_h,
                    weights: &w_h,
                    slots: &slots,
                    transit: &transit_experts,
                },
                &mut laps,
            )?;
            drop(guard);
            let host_experts: Vec<u32> = missing[to_transit..].to_vec();
            self.keep_in_transit(&transit_experts, None, &mut in_use, &stream)?;
            self.promote_up_to(&host_experts, &in_use, rows.max(ADAPT_SWAPS_A_REQUEST));
            laps.mark("tier rows: finish");
            return Ok(out);
        }
        // Without exact kernels and below the fast level, every pair runs as a decoded token's
        // does: on the card, its missing experts through transit. A layer without a transit room
        // runs them all on the host.
        if !self.splits_misses.load(Ordering::Relaxed) {
            let w_card = weights.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
            if self.stage.is_none() {
                drop(guard);
                let ids_card = ids.to_dtype(crate::tensor::DType::U32)?.contiguous()?;
                let out =
                    moe.forward_cpu(&xs.to_device(&host)?, &ids_card, &w_card, rows, hidden)?;
                self.promote_up_to(&missing, &in_use, rows.max(ADAPT_SWAPS_A_REQUEST));
                laps.mark("tier rows: finish");
                return out.to_device(&dev);
            }
            let slots: Vec<Option<u32>> = (0..pairs)
                .map(|p| is_hit[p].then_some(slot_ids[p]))
                .collect();
            let (out, rounds) =
                self.pairs_on_card(moe, scratch, xs, &ids_h, &w_card, &slots, &mut laps)?;
            drop(guard);
            self.keep_rounds(
                &rounds,
                &mut in_use,
                &stream,
                rows.max(ADAPT_SWAPS_A_REQUEST),
            )?;
            laps.mark("tier rows: finish");
            return Ok(out);
        }

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
        self.keep_in_transit(&transit_experts, None, &mut in_use, &stream)?;
        self.promote_up_to(&host_experts, &in_use, rows.max(ADAPT_SWAPS_A_REQUEST));
        laps.mark("tier rows: finish");
        Ok(out)
    }

    /// Rank the routed experts `ids` of a prompt into the cache from the room of a wide batch,
    /// which holds the experts of `placed` in that order: the experts the prompt used are ranked
    /// into the heat, below what a decoded token's use earns, and those of them the room holds,
    /// hottest first, take the slots of the colder ones by copies on the card.
    fn keep_hottest_now(
        &self,
        ids: &[u32],
        placed: &[u32],
        stream: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<()> {
        let mut count = vec![0f32; self.n_experts];
        for &e in ids {
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
        self.keep_in_transit(&present, Some(placed), &mut in_use, stream)
    }

    /// Move the experts of `experts` that are in a room, hottest first, into the slots that hold
    /// something colder and are not read by this token, as copies on the card behind the token's
    /// own work: a kernel that reads a slot later is queued after its copy. `in_use` marks the
    /// slots this token reads, and is extended with the slots filled here.
    ///
    /// Without `room`, the experts are in the transit room in the order they were copied there.
    /// With `room`, they are in the room of a wide batch at the place of each in `room`, and one
    /// that has none is not moved.
    fn keep_in_transit(
        &self,
        experts: &[u32],
        room: Option<&[u32]>,
        in_use: &mut [bool],
        stream: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    ) -> Result<()> {
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        let full;
        let arenas = if room.is_some() {
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
            .filter_map(|(j, &e)| match room {
                Some(placed) => placed.iter().position(|&p| p == e).map(|at| (e, at)),
                None => Some((e, j)),
            })
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

    /// The expert block of the rows `xs` whose pairs route to `ids` (router order) at the weights
    /// `w_card` (on the card, in pair order), for a layer whose formats have no exact kernels.
    ///
    /// Every pair's row is computed on the card by the same kernels: from its slot `slots[p]`,
    /// or, for a missing expert, from the transit room.
    /// The missing experts pass through the room in rounds of what it holds.
    /// Each row's pairs are summed in the router's order.
    ///
    /// Returns the rows `[rows, hidden]` and the experts of each round, in their room order.
    #[allow(clippy::too_many_arguments)]
    fn pairs_on_card(
        &self,
        moe: &MoeWeights,
        scratch: &Scratch,
        xs: &Tensor,
        ids: &[u32],
        w_card: &Tensor,
        slots: &[Option<u32>],
        laps: &mut Laps,
    ) -> Result<(Tensor, Vec<Vec<u32>>)> {
        use crate::inference::moe_cuda::MoreExperts;
        let stage = self.stage.as_ref().expect("a layer with a transit room");
        let host = Device::Cpu;
        let dev = xs.device().clone();
        let stream = dev.as_cuda_device()?.0.stream().clone();
        let k = moe.n_experts_used;
        let pairs = ids.len();
        let rows = pairs / k.max(1);
        let hidden = xs.dim(crate::tensor::D::Minus1)?;
        let x = xs.reshape((rows, hidden))?;
        let w = w_card.flatten_all()?;
        let on_card = |v: Vec<u32>| Tensor::from_vec(v, pairs, &host)?.to_device(&dev);
        // Pair p reads row p / k and writes row p: every pair is computed on its own.
        let order = on_card((0..pairs as u32).collect())?;
        let card_rows = if slots.iter().any(Option::is_some) {
            let routing = ExpertRouting {
                slots: order.clone(),
                experts: on_card(slots.iter().map(|s| s.unwrap_or(u32::MAX)).collect())?,
                topk: k,
            };
            let extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
            let more = extra.as_ref();
            let h = routing.gate_up_silu_over(
                &x,
                &self.card[0],
                &self.card[1],
                more.map(|m| MoreExperts::Stacks((&m[0], &m[1]))),
            )?;
            Some(routing.down_pairs_over(
                &h,
                &self.card[2],
                more.map(|m| MoreExperts::Stacks(&m[2])),
                &w,
            )?)
        } else {
            None
        };
        laps.mark("tier pairs: card launch");
        let mut missing: Vec<u32> = (0..pairs)
            .filter(|&p| slots[p].is_none())
            .map(|p| ids[p])
            .collect();
        missing.sort_unstable();
        missing.dedup();
        // Where each pair's row is read from: the card's rows (all ones), or a row of the rounds'.
        let mut src = vec![u32::MAX; pairs];
        let mut rounds = Vec::new();
        let mut late = Vec::new();
        for (r, round) in missing.chunks(stage.stager.transit.max(1)).enumerate() {
            // The room's next copies wait for the kernels of the round before to have read it.
            if r > 0 {
                let read = stream
                    .record_event(None)
                    .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?;
                stage
                    .stager
                    .copy
                    .wait(&read)
                    .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
            }
            self.copy_to_transit(scratch, round)?;
            stream
                .wait(&scratch.upload_end)
                .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
            let places: Vec<u32> = (0..pairs)
                .map(|p| match slots[p] {
                    Some(_) => u32::MAX,
                    None => round
                        .iter()
                        .position(|&e| e == ids[p])
                        .map_or(u32::MAX, |at| at as u32),
                })
                .collect();
            for (p, &at) in places.iter().enumerate() {
                if at != u32::MAX {
                    src[p] = (r * pairs + p) as u32;
                }
            }
            let routing = ExpertRouting {
                slots: order.clone(),
                experts: on_card(places)?,
                topk: k,
            };
            let h = routing.gate_up_silu(&x, &stage.compact[0], &stage.compact[1])?;
            late.push(routing.down_pairs_over(&h, &stage.compact[2], None, &w)?);
            rounds.push(round.to_vec());
        }
        laps.mark("tier pairs: transit");
        let late = match late.len() {
            0 => None,
            1 => late.pop(),
            _ => Some(Tensor::cat(&late.iter().collect::<Vec<_>>(), 0)?),
        };
        let first = card_rows
            .as_ref()
            .or(late.as_ref())
            .expect("a row routes to at least one expert");
        let src = on_card(src)?;
        let out = Tensor::zeros_on(rows * hidden, crate::tensor::DType::F32, &dev)?;
        exact::sum(
            &dev,
            address_of::<f32>(first)?,
            late.as_ref().map_or(Ok(0), address_of::<f32>)?,
            address_of::<u32>(&src)?,
            address_of::<f32>(&out)?,
            rows,
            k,
            hidden,
        )?;
        // The buffers the launches read live until the launches are queued behind them.
        drop((card_rows, late, src, order));
        Ok((out.reshape((rows, hidden))?, rounds))
    }

    /// After a request whose missing experts passed through the transit room in `rounds`: those
    /// of the last round, still in the room, keep a slot where they are hotter than what it holds;
    /// up to `most` of those of the rounds before go to the worker.
    fn keep_rounds(
        self: &Arc<Self>,
        rounds: &[Vec<u32>],
        in_use: &mut [bool],
        stream: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
        most: usize,
    ) -> Result<()> {
        let Some((last, before)) = rounds.split_last() else {
            return Ok(());
        };
        self.keep_in_transit(last, None, in_use, stream)?;
        self.promote_up_to(&before.concat(), in_use, most);
        Ok(())
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
        let stage = self.stage.as_ref().expect("a copy to transit has a stage");
        // The choice was waited for, so the layers that read this buffer before are done.
        start
            .record(&stage.stager.copy)
            .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?;
        let puts: Vec<(u32, usize)> = experts.iter().copied().zip(0..).collect();
        self.queue_copies(&puts)?;
        stage.stager.held[stage.buffer]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        end.record(&stage.stager.copy)
            .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))
    }

    /// Queue the copies of experts into places of this layer's transit buffer: each entry of
    /// `puts` is an expert and its place.
    pub(super) fn queue_copies(&self, puts: &[(u32, usize)]) -> Result<()> {
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        let stage = self.stage.as_ref().expect("a copy to transit has a stage");
        let copy = &stage.stager.copy;
        let buffer = &stage.stager.buffers[stage.buffer];
        for i in 0..3 {
            let eb = self.expert_bytes[i];
            let arena = &buffer.arenas[i];
            let base = arena.device_ptr(arena.stream()).0;
            let data = self.host[i].native_qtensor().data();
            for &(e, at) in puts {
                self.upload_mapped(i, data, e as usize * eb, eb, base + (at * eb) as u64, copy)?;
            }
        }
        self.uploaded
            .fetch_add(puts.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Of `missing` experts, how many the transit room takes for this request, the rest being
    /// the host's: [`Self::transit_share`] where the layer has a room.
    pub(super) fn transit_share_of(&self, missing: usize, rows: usize) -> usize {
        self.transit_share(missing, rows)
    }

    /// Places the transit room has, and the time to copy one expert into it in microseconds:
    /// the measured pace, else the card's measured rate. Zero where there is no room.
    pub(super) fn transit_room(&self) -> (usize, f64) {
        let Some(stage) = self.stage.as_ref() else {
            return (0, 0.0);
        };
        let measured = f64::from_bits(self.upload_us[0].load(Ordering::Relaxed));
        let rate = stage.stager.bandwidth;
        let copy = if measured > 0.0 {
            measured
        } else if rate > 0.0 {
            self.expert_bytes.iter().sum::<usize>() as f64 / rate
        } else {
            0.0
        };
        (stage.stager.transit, copy)
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

    /// Nanoseconds the host takes for one expert of a one-row request, measured on this layer,
    /// `0` before a measure.
    fn host_expert_ns(&self) -> u64 {
        super::expert_admission::ns_of_us(f64::from_bits(self.host_us[0].load(Ordering::Relaxed)))
    }

    /// The heat a missing expert needs to take a slot: the routes that repay its upload, as the
    /// card's admission measures them, never below `ADAPT_MIN_HEAT`.
    fn heat_to_take_a_slot(&self) -> f32 {
        super::expert_admission::heat_to_take_a_slot(
            self.admission.get().map(|a| a.as_ref()),
            self.host_expert_ns(),
            ADAPT_MIN_HEAT,
        )
    }

    /// The bytes of `expert` in the three host stacks, gate, up and down.
    fn expert_spans(&self, expert: usize) -> Vec<super::expert_readahead::Span> {
        (0..3)
            .map(|i| {
                let base = self.host[i].native_qtensor().data().as_ptr() as usize;
                let eb = self.expert_bytes[i];
                (base + expert * eb, eb)
            })
            .collect()
    }

    /// A one-row request reached this layer: the card's first cached layer starts the token's
    /// span and its last ends it, which times the read-ahead's trials.
    fn mark_token(&self) {
        let Some(ra) = self.readahead.get() else {
            return;
        };
        let now = std::time::Instant::now();
        if self.closer.load(Ordering::Relaxed) {
            ra.token_ends(now);
        }
        if self.leader.load(Ordering::Relaxed) {
            ra.token_starts(now);
        }
    }

    /// Notes in the process's expert counters the pairs of `ids` this tier served: `hits` from
    /// its slots, and the host's reads of the experts of the pairs in `missed`.
    /// An expert locked in host memory is not a read from the file.
    pub(super) fn note_served(&self, ids: &[u32], hits: usize, missed: &[u32]) {
        let counters = crate::inference::expert_counters::global();
        if !counters.on() {
            return;
        }
        let layer = self as *const Self as usize;
        counters.routed(layer, ids.iter().map(|&e| e as usize));
        counters.served(hits as u64, missed.len() as u64);
        let mut read: Vec<u32> = missed.to_vec();
        read.sort_unstable();
        read.dedup();
        let bytes: usize = read
            .iter()
            .flat_map(|&e| self.expert_spans(e as usize))
            .filter(|&(at, len)| !hot::overlaps_registered(at, at + len))
            .map(|(_, len)| len)
            .sum();
        counters.read(false, bytes as u64);
    }

    /// Queue the disk reads of the experts the next layer is expected to miss, while the
    /// read-ahead runs: its guess from its last request, of the experts one row routes to.
    /// An expert locked in host memory is not read.
    fn read_next_ahead(&self) {
        let Some(ra) = self.readahead.get().filter(|ra| ra.reads()) else {
            return;
        };
        let Some(next) = self.ahead_next.get().and_then(std::sync::Weak::upgrade) else {
            return;
        };
        let guess = {
            let st = next.state.lock().unwrap_or_else(|e| e.into_inner());
            next.ahead
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .guess(|e| st.slot_of[e as usize] >= 0, next.per_row)
        };
        for e in guess {
            let spans = next.expert_spans(e as usize);
            if !spans
                .iter()
                .any(|&(at, len)| hot::overlaps_registered(at, at + len))
            {
                let bytes: usize = spans.iter().map(|&(_, len)| len).sum();
                crate::inference::expert_counters::global().read(true, bytes as u64);
                ra.read(spans);
            }
        }
    }

    /// Drop from the page cache the pages of `used`, experts the card missed and the host read,
    /// while a tuned host tier exists and the host's memory is short.
    /// An expert locked in host memory keeps its pages.
    fn drop_after_use(&self, used: &[u32]) {
        let Some(ra) = self.readahead.get() else {
            return;
        };
        if used.is_empty() || !super::expert_host_tune::drops_unkept_pages() {
            return;
        }
        for &e in used {
            let spans = self.expert_spans(e as usize);
            if !spans
                .iter()
                .any(|&(at, len)| hot::overlaps_registered(at, at + len))
            {
                ra.drop_pages(spans);
            }
        }
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
    ///
    /// Every request ends here with the experts the card missed.
    /// Those not promoted have their pages dropped where memory is short (`drop_after_use`).
    pub(super) fn promote_up_to(self: &Arc<Self>, missing: &[u32], in_use: &[bool], most: usize) {
        let promoted = self.promote_some(missing, in_use, most);
        let left: Vec<u32> = missing
            .iter()
            .copied()
            .filter(|&e| !promoted.contains(&(e as usize)))
            .collect();
        self.drop_after_use(&left);
    }

    /// [`Self::promote_up_to`] without the drop: the experts handed to the copy worker.
    fn promote_some(self: &Arc<Self>, missing: &[u32], in_use: &[bool], most: usize) -> Vec<usize> {
        if most == 0 || self.promoting.swap(true, Ordering::AcqRel) {
            return Vec::new();
        }
        let host_ns = self.host_expert_ns();
        let bar = self.heat_to_take_a_slot();
        let chosen: Vec<(usize, usize)> = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let mut hot: Vec<u32> = missing
                .iter()
                .copied()
                .filter(|&e| st.heat[e as usize] >= bar)
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
                    let old = old as usize;
                    // A promoted expert leaves: what it earned there moves the admission's bias.
                    if let (Some(at), Some(a)) = (st.admitted[old], self.admission.get()) {
                        a.settle(u64::from(st.counts[old].saturating_sub(at)), host_ns);
                    }
                    st.set_slot(old, -1);
                }
                st.owner[victim] = -2;
                pairs.push((expert as usize, victim));
            }
            pairs
        };
        if chosen.is_empty() {
            self.promoting.store(false, Ordering::Release);
            return Vec::new();
        }
        let promoted: Vec<usize> = chosen.iter().map(|&(expert, _)| expert).collect();
        let tier = self.clone();
        copier(self.dev.as_cuda_device().map_or(0, |d| d.0.ordinal()))
            .send(Box::new(move || {
                let started = std::time::Instant::now();
                let landed = (|| -> Result<()> {
                    for &(expert, slot) in &chosen {
                        tier.copy_in_async(expert, slot)?;
                    }
                    tier.copy
                        .synchronize()
                        .map_err(|e| crate::tensor::Error::msg(format!("slot copy: {e:?}")))
                })();
                // The job copied its experts one after the other: each took its share.
                let upload_ns = started.elapsed().as_nanos() as u64 / chosen.len() as u64;
                crate::inference::stage_prof::prof_record("tier slot upload", upload_ns);
                let mut st = tier.state.lock().unwrap_or_else(|e| e.into_inner());
                for &(expert, slot) in &chosen {
                    match &landed {
                        Ok(()) => {
                            st.owner[slot] = expert as i32;
                            st.set_slot(expert, slot as i32);
                            let count = st.counts[expert];
                            st.admitted[expert] = Some(count);
                            if let Some(a) = tier.admission.get() {
                                a.note_upload(upload_ns);
                            }
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
        promoted
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
    /// rooms of a wide batch are made when a prefill runs, see
    /// [`Self::ensure_full`].
    pub fn new(dev: &Device, widest: [usize; 3], experts: usize) -> Result<Arc<Self>> {
        Self::holding(dev, widest, experts, 0)
    }

    /// [`Self::new`] with a transit room of at least `least` experts: the pairs of a token of a
    /// layer that runs every missing expert there.
    pub fn holding(
        dev: &Device,
        widest: [usize; 3],
        experts: usize,
        least: usize,
    ) -> Result<Arc<Self>> {
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
        let transit = transit_experts(experts, bandwidth, host_read_bandwidth(widest[0]))
            .max(least)
            .min(experts);
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
            held: Default::default(),
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

    /// Make the rooms of a wide batch, each wide enough for a layer's whole stacks, from the
    /// memory the caches lend. Nothing is done while they exist.
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
/// The host bytes a rate probe of `want` bytes takes: all of them while they fit in the burst
/// headroom and the host's room now, the smallest probe otherwise.
fn probe_bytes(want: usize) -> usize {
    use crate::inference::place::layer_executor::{
        host_burst_headroom_bytes, host_spill_budget_bytes,
    };
    probe_size(
        want,
        host_spill_budget_bytes().min(host_burst_headroom_bytes()),
    )
}

/// [`probe_bytes`] for a host with `room` bytes to give a probe.
fn probe_size(want: usize, room: u64) -> usize {
    const SMALLEST: usize = 1 << 20;
    if want as u64 <= room {
        want.max(SMALLEST)
    } else {
        SMALLEST
    }
}

fn host_read_bandwidth(bytes: usize) -> f64 {
    use rayon::prelude::*;
    let len = probe_bytes(bytes.min(1 << 28));
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
    let bytes = probe_bytes(arena_bytes.min(64 << 20)).min(arena_bytes);
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
            placed: Mutex::new(Vec::new()),
            pinned,
            #[cfg(target_os = "linux")]
            bounce: Mutex::new(BounceState::Closed),
            held: std::sync::atomic::AtomicBool::new(false),
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
    /// Whether a batch of `rows` runs through the expert-batched kernels, which read the layer's
    /// experts where the card holds them: the room of a wide batch is there for the experts the
    /// card misses, the layer's formats are ones those kernels read, and the rows' routed pairs
    /// reach the count from which they pay. The batch's shape alone decides.
    pub fn reads_in_place(&self, rows: usize, used: usize) -> bool {
        use crate::inference::moe_cuda::{moe_mmq_pays, moe_mmq_serves};
        let Some(stage) = self.stage.as_ref() else {
            return false;
        };
        if self.resident
            || stage
                .full
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_none()
        {
            return false;
        }
        // The tile kernels have no bounded SwiGLU.
        if self.exact.is_some_and(|e| e.limit != 0.0) {
            return false;
        }
        let reads = |t: &QTensor| moe_mmq_serves(t.dtype(), t.shape().dims()[2]);
        reads(&self.host[0]) && reads(&self.host[2]) && moe_mmq_pays(rows * used, self.n_experts)
    }

    /// A wide batch's expert block through the expert-batched kernels: `xs` `[rows, hidden]`
    /// and the router's choices `ids`, `weights` `[rows, k]`, all on the card. Returns
    /// `[rows, hidden]` on the card.
    ///
    /// The kernels read each expert where the card holds it: the base slots, the extra slots,
    /// and the room of a wide batch for the experts the rows route to and the card misses.
    /// Only those are copied, each once while the room is held.
    /// A row gives the bits it gives on a layer whose every expert is on the card.
    pub(super) fn wide_rows(
        self: &Arc<Self>,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        use crate::inference::moe_cuda::{Glu, MmqRouting, Region};
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        self.health_check()?;
        let stage = self
            .stage
            .as_ref()
            .expect("a wide batch reads a staged layer");
        let host = Device::Cpu;
        let dev = xs.device().clone();
        let cd = dev.as_cuda_device()?;
        let stream = cd.0.stream().clone();
        let k = moe.n_experts_used;
        // The layer's one wait: the choices of every row come to the host.
        let ids_h: Vec<u32> = ids
            .to_dtype(crate::tensor::DType::U32)?
            .to_device(&host)?
            .flatten_all()?
            .to_vec1()?;
        let rows = ids_h.len() / k.max(1);
        let Some(full) = stage.full.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
            crate::tensor::bail!("a wide batch's room is not there");
        };
        let extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
        let extra_n = extra.as_ref().map_or(0, |m| m[0].shape().dims()[0]);
        let first_room = self.base_slots + extra_n;
        let mut placed = stage.placed.lock().unwrap_or_else(|e| e.into_inner());
        let had = placed.len();
        // Each pair's place: the slot of its expert, or the expert's place in the room.
        // An expert on its way into a slot has none yet, and the slot it takes is read by none.
        let places: Vec<u32> = {
            let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let mut room_at = vec![u32::MAX; self.n_experts];
            for (j, &e) in placed.iter().enumerate() {
                room_at[e as usize] = j as u32;
            }
            let mut missing: Vec<u32> = ids_h
                .iter()
                .copied()
                .filter(|&e| st.slot_of[e as usize] < 0 && room_at[e as usize] == u32::MAX)
                .collect();
            missing.sort_unstable();
            missing.dedup();
            for &e in &missing {
                room_at[e as usize] = placed.len() as u32;
                placed.push(e);
            }
            ids_h
                .iter()
                .map(|&e| match st.slot_of[e as usize] {
                    s if s >= 0 => s as u32,
                    _ => first_room as u32 + room_at[e as usize],
                })
                .collect()
        };
        // The experts new to the room come from the host, behind the last reader of the room.
        if placed.len() > had {
            let bufs = stage.stager.full.lock().unwrap_or_else(|e| e.into_inner());
            let Some(bufs) = bufs.as_ref() else {
                crate::tensor::bail!("a wide batch's room is not there");
            };
            let buffer = &bufs[stage.buffer];
            let copy = &stage.stager.copy;
            cd.0.context()
                .bind_to_thread()
                .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
            if had == 0 {
                if let Some(free) = buffer.free.lock().unwrap_or_else(|e| e.into_inner()).take() {
                    copy.wait(&free)
                        .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
                }
            }
            let new = &placed[had..];
            for i in 0..3 {
                let eb = self.expert_bytes[i];
                let arena = &buffer.arenas[i];
                let base = arena.device_ptr(arena.stream()).0;
                let data = self.host[i].native_qtensor().data();
                // A run of consecutive experts is one copy.
                let mut j = 0;
                while j < new.len() {
                    let mut end = j + 1;
                    while end < new.len() && new[end] == new[end - 1] + 1 {
                        end += 1;
                    }
                    self.upload(
                        i,
                        data,
                        new[j] as usize * eb,
                        (end - j) * eb,
                        base + ((had + j) * eb) as u64,
                        copy,
                    )?;
                    j = end;
                }
                // The kernels read whole blocks past the room's last expert, against activations
                // padded with zeros: that tail is zero, and not what the layer before left there.
                driver("room tail", unsafe {
                    crate::tensor::gpu_runtime::driver::sys::cuMemsetD8Async(
                        base + (placed.len() * eb) as u64,
                        0,
                        crate::tensor::quantized::BLOB_TAIL_PAD_BYTES,
                        copy.cu_stream(),
                    )
                })?;
            }
            self.to_room
                .fetch_add((placed.len() - had) as u64, Ordering::Relaxed);
            let landed = copy
                .record_event(None)
                .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?;
            stream
                .wait(&landed)
                .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
        }
        let choices = Tensor::from_vec(places, (rows, k), &host)?.to_device(&dev)?;
        let plan = MmqRouting::from_choices(&choices, first_room + placed.len())?;
        let regions = |i: usize, up: bool| -> Vec<Region<'_>> {
            let mut out = vec![Region {
                stack: &self.card[i],
                second: up.then_some(&self.card[1]),
                first: 0,
                experts: self.base_slots,
            }];
            if let Some(m) = extra.as_ref() {
                out.push(Region {
                    stack: &m[i],
                    second: up.then_some(&m[1]),
                    first: self.base_slots,
                    experts: extra_n,
                });
            }
            out.push(Region {
                stack: &full[i],
                second: up.then_some(&full[1]),
                first: first_room,
                experts: placed.len(),
            });
            out.retain(|r| r.experts > 0);
            out
        };
        let h = plan.gate_up_over(xs, &regions(0, true), Glu::Silu)?;
        let out = plan.down_fold_over(&h, &regions(2, false), weights, None, None)?;
        drop(extra);
        if stage.held.load(Ordering::Acquire) {
            // A held room ranks the prompt's routing once, when the hold ends.
            let mut pending = self.prompt_ids.lock().unwrap_or_else(|e| e.into_inner());
            match pending.as_mut() {
                Some((_, all)) => all.extend_from_slice(&ids_h),
                None => *pending = Some((stream.clone(), ids_h)),
            }
        } else {
            if let Err(e) = self.keep_hottest_now(&ids_h, &placed, &stream) {
                tracing::warn!("the cache is not warmed from the prompt: {e}");
            }
            self.release_room(&mut placed)?;
        }
        Ok(out)
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
                let (n, err) = b.upload(off, to - from, dst + (from - a) as u64, copy, false);
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

    /// Keep the experts a prompt copies into the room of a wide batch across the forwards that
    /// follow, for a prompt that goes through the layer in several chunks: an expert comes once,
    /// not once per chunk. `false` ends the hold, ranks the prompt's routing into the cache and
    /// hands the room on.
    pub fn hold_stacks(&self, on: bool) -> Result<()> {
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        if on {
            stage.held.store(true, Ordering::Release);
            return Ok(());
        }
        let pending = self
            .prompt_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let mut placed = stage.placed.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((stream, ids)) = pending {
            // Ranking the prompt is a convenience: a pass that failed, or a card out of memory,
            // must still end the hold and hand the room on.
            if let Err(e) = self.keep_hottest_now(&ids, &placed, &stream) {
                tracing::warn!("the cache is not warmed from the prompt: {e}");
            }
        }
        stage.held.store(false, Ordering::Release);
        self.release_room(&mut placed)
    }

    /// Forget what the room of a wide batch holds of this layer. A pass cut short leaves experts
    /// there, and layers share two rooms, so the next pass would read another layer's experts as
    /// its own. The copies already issued are waited for first, so nothing writes a room after it
    /// is handed on.
    pub(super) fn reset_stage(&self) -> Result<()> {
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        stage
            .stager
            .copy
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("copy stream: {e:?}")))?;
        stage
            .placed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
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

    /// The layer's kernels that read the room of a wide batch are queued: the room is free once
    /// they have run, and holds nothing of this layer from here on.
    fn release_room(&self, placed: &mut Vec<u32>) -> Result<()> {
        if placed.is_empty() {
            return Ok(());
        }
        placed.clear();
        let stage = self.stage.as_ref().expect("a staged layer");
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

    /// Experts wide batches copied into the room of a wide batch since the load.
    pub fn copied_to_room(&self) -> u64 {
        self.to_room.load(Ordering::Relaxed)
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
#[cfg(feature = "gpu")]
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

    /// A SwiGLU bound below the gate and up products of the test layers: the clamp moves some.
    const BOUNDED_SWIGLU_LIMIT: f32 = 0.5;

    /// A bounded SwiGLU is served only in a layer the exact kernels read.
    #[test]
    fn a_bounded_swiglu_is_served_only_through_the_exact_kernels() {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let mut plain = layer_of(&card, 8, 2, 512, 256, [GgmlDType::Q5_1; 2]);
        assert!(serves(&plain), "a Q5_1 layer is served unbounded");
        plain.swiglu_limit = Some(BOUNDED_SWIGLU_LIMIT);
        assert!(!serves(&plain), "Q5_1 has no exact kernel");
        let mut k_quants = layer_of(&card, 8, 2, 512, 256, [GgmlDType::Q4K, GgmlDType::Q6K]);
        k_quants.swiglu_limit = Some(BOUNDED_SWIGLU_LIMIT);
        assert!(serves(&k_quants));
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
            // CONVERSION: the test builds the expert stack in the dtype under test.
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
            swiglu_limit: None,
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
            block_scaled: None,
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
            let stager = CardStager::holding(&card, widest, experts, used).unwrap();
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
        tier.enable_graph(&tiered, 4).unwrap();
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

    /// Two layers on one card, served by one worker, run the same tokens with the next layer's
    /// experts fetched ahead and without: both match the host, and the fetched experts are used.
    #[test]
    fn experts_fetched_a_layer_ahead_give_the_same_tokens() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let refs = [
            layer(&Device::Cpu, experts, used, hidden, ffn),
            layer(&Device::Cpu, experts, used, hidden, ffn),
        ];
        let x = values(hidden, 100, 1.0);
        let on_host = Tensor::from_vec(x.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
        let through = |layers: &[MoeWeights; 2], x: &Tensor| {
            let y = layers[0].forward(x, x, hidden, 1e-5, false).unwrap();
            layers[1]
                .forward(&y, &y, hidden, 1e-5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let want = through(&refs, &on_host);
        let run = |forced: u8| {
            let mut layers = [
                layer(&card, experts, used, hidden, ffn),
                layer(&card, experts, used, hidden, ffn),
            ];
            let widest = [0, 1, 2].map(|i| {
                [
                    layers[0].gate_exps_sep.as_ref().unwrap(),
                    layers[0].up_exps_sep.as_ref().unwrap(),
                    &layers[0].down_exps,
                ][i]
                    .native_qtensor()
                    .data()
                    .len()
            });
            let stager = CardStager::holding(&card, widest, experts, used).unwrap();
            poison(&stager);
            let mut named = Vec::new();
            for (i, l) in layers.iter_mut().enumerate() {
                let built = LayerTier::new(l, &card, 2)
                    .unwrap()
                    .with_stage(&stager, i, false)
                    .unwrap();
                l.tier = Some(Arc::new(built));
                let tier = l.tier.clone().unwrap();
                tier.enable_graph(l, 1).unwrap();
                named.push((format!("layer {i}"), tier));
            }
            let worker = TierWorker::start(&card, named.clone()).unwrap();
            GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
            graph::AHEAD_FOR_TEST.store(forced, Ordering::Relaxed);
            let on_card = Tensor::from_vec(x.clone(), (1, 1, hidden), &card).unwrap();
            let mut got = Vec::new();
            for _ in 0..40 {
                got.push(through(&layers, &on_card));
                named.iter().for_each(|(_, t)| t.settle());
            }
            graph::AHEAD_FOR_TEST.store(0, Ordering::Relaxed);
            GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
            drop(worker);
            (got, named[1].1.fetched_ahead())
        };
        let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let close = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
                <= 2e-2 * peak
        };
        let (without, (none_fetched, _)) = run(1);
        let (with, (fetched, used_ahead)) = run(2);
        assert_eq!(none_fetched, 0, "nothing is fetched when it is off");
        assert!(fetched > 0, "nothing was fetched ahead");
        assert!(used_ahead > 0, "no fetched expert was used");
        for (token, (a, b)) in without.iter().zip(&with).enumerate() {
            assert!(close(a, &want), "token {token} without, off the host");
            assert!(close(b, &want), "token {token} with, off the host");
            assert!(close(a, b), "token {token}: with and without differ");
        }
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
        tier.enable_graph(&tiered, 1).unwrap();
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
        let bound = std::time::Duration::from_nanos(worker.stall_bound(&named));
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

    /// A kernel another thread loads for the first time while the card waits on the worker
    /// holds every call and copy to the card until the card is idle. The worker still answers
    /// a request whose experts go through the transit room: the host computes them past the
    /// copies' bound, and the token is the host's.
    #[test]
    fn a_kernel_loaded_while_the_card_waits_leaves_the_request_answered() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        // A format without exact kernels, whose misses all go to transit below the fast level,
        // and one with them.
        kernel_loaded_while_the_card_waits(&card, (32, 4, 64, 64), [GgmlDType::Q5_1; 2], false);
        kernel_loaded_while_the_card_waits(
            &card,
            (32, 4, 512, 256),
            [GgmlDType::Q4K, GgmlDType::Q6K],
            true,
        );
    }

    /// The test above over a layer of `formats`, which takes the exact kernels when `exact`.
    fn kernel_loaded_while_the_card_waits(
        card: &Device,
        (experts, used, hidden, ffn): (usize, usize, usize, usize),
        formats: [GgmlDType; 2],
        exact: bool,
    ) {
        use crate::tensor::gpu_runtime::driver::{LaunchConfig, PushKernelArg};
        let card = card.clone();
        let reference = layer_of(&Device::Cpu, experts, used, hidden, ffn, formats);
        let mut tiered = layer_of(&card, experts, used, hidden, ffn, formats);
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
        let stager = CardStager::holding(&card, widest, experts, used).unwrap();
        let built = LayerTier::new(&tiered, &card, 2)
            .unwrap()
            .with_stage(&stager, 0, false)
            .unwrap();
        assert_eq!(
            built.exact.is_some(),
            exact,
            "{formats:?}: the exact kernels"
        );
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(&tiered, 1).unwrap();
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
        // A first token, answered, then paces that send every missing expert to the transit
        // room, and a room that holds none of them.
        tiered
            .forward(&on_card, &on_card, hidden, 1e-5, false)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap();
        tier.wait_answered();
        tier.settle();
        for class in 0..2 {
            tier.host_us[class].store(1e3f64.to_bits(), Ordering::Relaxed);
            tier.upload_us[class].store(1f64.to_bits(), Ordering::Relaxed);
        }
        for held in &stager.held {
            held.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
        // A module this process never loaded, and the stream it runs on, made while the card
        // is idle.
        let ptx = crate::tensor::gpu_runtime::nvrtc::compile_ptx(
            "extern \"C\" __global__ void loken_tier_test_touch(float *x, int n) { \
             if (threadIdx.x < n) x[threadIdx.x] += 1.0f; }",
        )
        .unwrap();
        let ctx = card.as_cuda_device().unwrap().0.context().clone();
        let side = ctx.new_stream().unwrap();
        // The launch rounds its threads up to a whole block: the kernel touches these alone.
        let touched_len = 32i32;
        let touched = side.alloc_zeros::<f32>(touched_len as usize).unwrap();
        graph::STALL_FOR_TEST.store(true, Ordering::Release);
        let pending = tiered
            .forward(&on_card, &on_card, hidden, 1e-5, false)
            .unwrap();
        tier.wait_posted();
        let (started, starting) = std::sync::mpsc::channel();
        let loader = std::thread::spawn(move || -> std::result::Result<(), String> {
            ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
            started.send(()).map_err(|e| e.to_string())?;
            let module = ctx.load_module(ptx).map_err(|e| format!("{e:?}"))?;
            let f = module
                .load_function("loken_tier_test_touch")
                .map_err(|e| format!("{e:?}"))?;
            let mut launch = side.launch_builder(&f);
            launch.arg(&touched);
            launch.arg(&touched_len);
            unsafe { launch.launch(LaunchConfig::for_num_elems(touched_len as u32)) }
                .map_err(|e| format!("{e:?}"))?;
            side.synchronize().map_err(|e| format!("{e:?}"))
        });
        starting.recv().unwrap();
        // The loader is in its load when the worker reads the request.
        std::thread::sleep(std::time::Duration::from_millis(50));
        graph::STALL_FOR_TEST.store(false, Ordering::Release);
        let got = pending
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap();
        let loaded = loader.join().unwrap();
        tier.settle();
        GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
        let failure = worker.failure();
        drop(worker);
        assert!(
            failure.is_none(),
            "{formats:?}: the worker stalled: {failure:?}"
        );
        loaded.expect("the kernel loaded and ran once the card was idle");
        let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 2e-2 * peak,
            "{formats:?}: max diff {worst} of {peak}"
        );
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
        tier.enable_graph(&tiered, 1).unwrap();
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
        tier.enable_graph(&tiered, 6).unwrap();
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
            // A stall releases the card's wait, and the rows then read a product that was not
            // answered: the mismatch is the stall's, so the stall is what fails the test.
            assert!(
                worker.failure().is_none(),
                "round {round}: {:?}",
                worker.failure()
            );
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
        let stager = CardStager::holding(&card, widest, experts, used).unwrap();
        let built = LayerTier::new(&tiered, &card, 6)
            .unwrap()
            .with_stage(&stager, 0, false)
            .unwrap();
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(&tiered, 4).unwrap();
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
            // The card has posted the request, and the worker has counted and answered it.
            card.synchronize().unwrap();
            tier.wait_answered();
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
        let stager = CardStager::holding(&card, widest, experts, used).unwrap();
        let built = LayerTier::new(&tiered, &card, 6)
            .unwrap()
            .with_stage(&stager, 0, false)
            .unwrap();
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        tier.enable_graph(&tiered, 6).unwrap();
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
            // A stall releases the card's wait, and the rows then read a product that was not
            // answered: the mismatch is the stall's, so the stall is what fails the test.
            assert!(
                worker.failure().is_none(),
                "round {round}: {:?}",
                worker.failure()
            );
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
            let stager = CardStager::holding(&card, widest, experts, used).unwrap();
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
            [GgmlDType::Iq2Xxs, GgmlDType::Q2_0],
            [GgmlDType::Iq2Xxs, GgmlDType::Iq2Xxs],
        ]
        .into_iter()
        .flat_map(|f| [(f, None), (f, Some(BOUNDED_SWIGLU_LIMIT))])
        {
            let (formats, limit) = formats;
            // A bounded SwiGLU over IQ2_XXS experts runs on the host.
            if limit.is_some() && formats.contains(&GgmlDType::Iq2Xxs) {
                continue;
            }
            let mut reference = layer_of(&Device::Cpu, experts, used, hidden, ffn, formats);
            reference.swiglu_limit = limit;
            let mut resident = layer_of(&card, experts, used, hidden, ffn, formats);
            resident.swiglu_limit = limit;
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
                    eprintln!(
                        "{formats:?} bound {limit:?} rows {rows} row {r}: relative distance {distance:.5}"
                    );
                    // IQ2_XXS reads its activations as 8 bit blocks on the card.
                    let tolerance = if formats.contains(&GgmlDType::Iq2Xxs) {
                        2e-2
                    } else {
                        1e-2
                    };
                    assert!(
                        distance <= tolerance,
                        "{formats:?} bound {limit:?}, {rows} rows, row {r}: the card is {distance:.4} of the host's norm away"
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
            let stager = CardStager::holding(&card, widest, experts, used).unwrap();
            // What an earlier layer left in the arenas: bytes whose f16 scales are not finite.
            poison(&stager);
            // Some experts on the card, the rest to come from the host.
            let tier = LayerTier::new(&tiered, &card, 9)
                .unwrap()
                .with_stage(&stager, 0, pin)
                .unwrap();
            assert!(tier.reads_in_place(rows, used));
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

    /// A layer of a served file, its slots lent to a prompt and returned, then decoding token
    /// after token: every decoded row gives the host's, for every format the down stack may be
    /// stored in, its missing experts coming through the transit room or every expert held.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn decoded_rows_after_a_prompt_and_a_return_match_the_host() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden) = (32usize, 8usize, 2560usize);
        // Inputs at the scale a normed hidden state has against these weights: the sums a
        // quantised activation block carries stay within half precision.
        const SCALE: f32 = 0.1;
        // A decoded row's distance from the host: the rounding of the 8-bit activation and of the
        // 3- and 5-bit superblocks.
        const DECODED: f32 = 3e-2;
        // A prompt's rows go through the tiled kernels, whose activation quantiser rounds more.
        const PROMPT: f32 = 5e-2;
        let mut failures = Vec::new();
        for down in EXPERT_QUANT_CODES_FOR_TESTS {
            // The depth of a down row is a whole number of the format's blocks.
            let ffn = if down.block_size() == 256 { 768 } else { 640 };
            let formats = [GgmlDType::Q4K, down];
            let quantised = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (
                    layer_of(&Device::Cpu, experts, used, hidden, ffn, formats),
                    layer_of(&card, experts, used, hidden, ffn, formats),
                )
            }));
            let Ok((reference, mut tiered)) = quantised else {
                eprintln!("{down:?}: not quantised on the host");
                continue;
            };
            for base in [6usize, experts] {
                let stager = stager_for(&tiered, &card, experts);
                poison(&stager);
                let mut built = LayerTier::new(&tiered, &card, base).unwrap();
                if base < experts {
                    built = built.with_extra(8).unwrap();
                }
                tiered.tier = Some(Arc::new(built.with_stage(&stager, 0, false).unwrap()));
                let tier = tiered.tier.clone().unwrap();
                if base < experts {
                    tier.restore_extra().unwrap();
                }
                tier.settle();
                let measure = |got: &[f32], want: &[f32]| -> (f32, usize) {
                    let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    let bad = got.iter().filter(|v| !v.is_finite()).count();
                    let d = got
                        .iter()
                        .zip(want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    (d / peak, bad)
                };
                let decoded = |seed: u64| -> (f32, usize) {
                    let x = values(hidden, seed, SCALE);
                    let on_host =
                        Tensor::from_vec(x.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
                    let on_card = Tensor::from_vec(x, (1, 1, hidden), &card).unwrap();
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
                    measure(&got, &want)
                };
                let fold = |a: (f32, usize), b: (f32, usize)| (a.0.max(b.0), a.1 + b.1);
                let decode_before = (0..3).map(|t| decoded(200 + t)).fold((0f32, 0usize), fold);
                if base < experts {
                    assert!(tier.lend_extra());
                }
                let rows = 109;
                let x = values(rows * hidden, 55, SCALE);
                let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
                let want = reference
                    .forward(&on_host, &on_host, hidden, 1e-5, true)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let prompt = measure(&prompt_rows(&tiered, &card, &x, hidden), &want);
                if base < experts {
                    tier.restore_extra().unwrap();
                }
                tier.settle();
                let after = (0..4).map(|t| decoded(300 + t)).fold((0f32, 0usize), fold);
                let (worst, not_finite) = fold(decode_before, after);
                eprintln!(
                    "{down:?} slots {base}: decoded before {:.4} ({} not finite), prompt {:.4} \
                     ({} not finite), decoded in all {worst:.4} ({not_finite} not finite)",
                    decode_before.0, decode_before.1, prompt.0, prompt.1
                );
                if worst > DECODED || not_finite > 0 || prompt.0 > PROMPT || prompt.1 > 0 {
                    failures.push(format!(
                        "{down:?} with {base} slots: decoded rows off by {worst} with \
                         {not_finite} values not finite, prompt off by {} with {} not finite",
                        prompt.0, prompt.1
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The formats the expert kernels serve, in the order the kernels number them.
    const EXPERT_QUANT_CODES_FOR_TESTS: [GgmlDType; 16] = [
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q5_0,
        GgmlDType::MxFp4,
        GgmlDType::Q5_1,
        GgmlDType::Iq4Nl,
        GgmlDType::Iq4Xs,
        GgmlDType::Iq3Xxs,
        GgmlDType::Iq3S,
        GgmlDType::Iq2S,
        GgmlDType::Q2_0,
        GgmlDType::Iq2Xxs,
    ];

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
        let stager = CardStager::holding(&card, widest, experts, used).unwrap();
        let slots = 6;
        let tier = LayerTier::new(&tiered, &card, slots)
            .unwrap()
            .with_stage(&stager, 0, false)
            .unwrap();
        let rows = 1520usize;
        assert!(tier.reads_in_place(rows, used));
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
            let stager = CardStager::holding(&card, widest, experts, used).unwrap();
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

    /// Bytes whose f16 scales are not finite in the rooms of a wide batch: what another layer
    /// may have left there.
    fn poison_rooms(stager: &CardStager) {
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        let cd = stager.dev.as_cuda_device().unwrap();
        cd.0.context().bind_to_thread().unwrap();
        let full = stager.full.lock().unwrap();
        for buffer in full.as_ref().expect("the rooms of a wide batch are made") {
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
    }

    /// The bytes of each stack of `moe` and the stager of a card for a layer of its shape.
    fn stager_for(moe: &MoeWeights, card: &Device, experts: usize) -> Arc<CardStager> {
        let widest = [0, 1, 2].map(|i| {
            [
                moe.gate_exps_sep.as_ref().unwrap(),
                moe.up_exps_sep.as_ref().unwrap(),
                &moe.down_exps,
            ][i]
                .native_qtensor()
                .data()
                .len()
        });
        CardStager::holding(card, widest, experts, moe.n_experts_used).unwrap()
    }

    /// The flattened output of `moe` for the rows `x` of a prompt.
    fn prompt_rows(moe: &MoeWeights, card: &Device, x: &[f32], hidden: usize) -> Vec<f32> {
        let rows = x.len() / hidden;
        let on_card = Tensor::from_vec(x.to_vec(), (1, rows, hidden), card).unwrap();
        moe.forward(&on_card, &on_card, hidden, 1e-5, true)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// How a test cache holds its experts: base slots, extra slots, whether the extra slots are
    /// lent, and whether a profile primed the slots.
    #[derive(Clone, Copy, Debug)]
    struct Holding {
        base: usize,
        extra: usize,
        lent: bool,
        primed: bool,
    }

    /// A tier over `moe` holding as `holding` says, its room of a wide batch poisoned.
    ///
    /// A primed tier holds the experts a profile ranks highest, spread over the stack, each at the
    /// heat the profile gives it.
    /// An unprimed one holds the first experts, with no heat.
    fn holding_tier(
        moe: &MoeWeights,
        card: &Device,
        experts: usize,
        holding: Holding,
    ) -> Arc<LayerTier> {
        let stager = stager_for(moe, card, experts);
        let mut built = LayerTier::new(moe, card, holding.base).unwrap();
        if holding.extra > 0 {
            built = built.with_extra(holding.extra).unwrap();
        }
        let tier = Arc::new(built.with_stage(&stager, 0, false).unwrap());
        if holding.primed {
            let counts: Vec<u32> = (0..experts)
                .map(|e| ((e * 7) % experts + 1) as u32)
                .collect();
            tier.prime(&counts).unwrap();
        }
        if holding.extra > 0 {
            tier.restore_extra().unwrap();
            tier.settle();
            assert!(tier.has_extra());
            if holding.lent {
                assert!(tier.lend_extra());
            }
        }
        poison_rooms(&stager);
        tier
    }

    /// A wide batch reads the experts the cache holds where they are and copies only those its
    /// rows route to and the card misses: each row gives the bits a layer whose every expert is
    /// on the card gives, whatever the cache holds, with the extra slots held or lent, and with
    /// what another layer left in the room.
    #[test]
    fn a_wide_batch_reads_the_cache_in_place_with_the_bits_of_a_resident_layer() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let experts = 32usize;
        for ((used, hidden, ffn), formats, rows) in [
            (
                (4usize, 64usize, 64usize),
                [GgmlDType::Q8_0; 2],
                vec![24usize, 700, 1520],
            ),
            (
                (8, 512, 256),
                [GgmlDType::Q4K, GgmlDType::Q6K],
                vec![100, 700],
            ),
            (
                (8, 512, 256),
                [GgmlDType::Q4K, GgmlDType::Q5_0],
                vec![100, 700],
            ),
        ] {
            let mut resident = layer_of(&card, experts, used, hidden, ffn, formats);
            resident.tier = Some(Arc::new(LayerTier::all_on_card(&resident, &card).unwrap()));
            for holding in [
                Holding {
                    base: 9,
                    extra: 0,
                    lent: false,
                    primed: true,
                },
                Holding {
                    base: 9,
                    extra: 0,
                    lent: false,
                    primed: false,
                },
                Holding {
                    base: 6,
                    extra: 8,
                    lent: false,
                    primed: true,
                },
                Holding {
                    base: 6,
                    extra: 8,
                    lent: true,
                    primed: true,
                },
                Holding {
                    base: experts,
                    extra: 0,
                    lent: false,
                    primed: true,
                },
            ] {
                let mut tiered = layer_of(&card, experts, used, hidden, ffn, formats);
                let tier = holding_tier(&tiered, &card, experts, holding);
                tiered.tier = Some(tier.clone());
                for &n in &rows {
                    assert!(tier.reads_in_place(n, used), "{formats:?} rows {n}");
                    let x = values(n * hidden, 91 + n as u64, 1.0);
                    let want = prompt_rows(&resident, &card, &x, hidden);
                    let before = tier.copied_to_room();
                    let held = (0..experts).filter(|&e| tier.resident(e)).count();
                    let got = prompt_rows(&tiered, &card, &x, hidden);
                    assert!(
                        got == want,
                        "{formats:?} {holding:?} rows {n}: not the bits of a resident layer"
                    );
                    let copied = tier.copied_to_room() - before;
                    assert!(
                        copied as usize <= experts - held,
                        "{formats:?} {holding:?} rows {n}: {copied} copies"
                    );
                    if holding.base == experts {
                        assert_eq!(copied, 0, "a cache holding every expert copied some");
                    }
                }
            }
        }
    }

    /// A prompt in several chunks through a layer that holds its room, as a prompt read layer by
    /// layer goes: every chunk gives the bits of a resident layer, the room holds exactly the
    /// experts the chunks route to and the card misses, each copied once, and the end of the hold
    /// hands the room on. With slots colder than the prompt makes its experts, the end of the
    /// hold moves experts of the room into them.
    #[test]
    fn a_held_room_copies_each_missing_expert_once_across_the_chunks() {
        held_room_across_chunks(
            Holding {
                base: 6,
                extra: 0,
                lent: false,
                primed: false,
            },
            true,
        );
    }

    /// The same with every slot hotter than the prompt can make an expert: the end of the hold
    /// moves nothing into the slots, and still hands the room on.
    #[test]
    fn a_held_room_leaves_slots_hotter_than_the_prompt() {
        held_room_across_chunks(
            Holding {
                base: 6,
                extra: 8,
                lent: false,
                primed: true,
            },
            false,
        );
    }

    /// Three chunks of a prompt through a held room over a cache holding as `holding` says;
    /// `moves` says whether the slots are colder than what the prompt makes its experts.
    fn held_room_across_chunks(holding: Holding, moves: bool) {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 8usize, 512usize, 256usize);
        let formats = [GgmlDType::Q4K, GgmlDType::Q6K];
        let mut resident = layer_of(&card, experts, used, hidden, ffn, formats);
        resident.tier = Some(Arc::new(LayerTier::all_on_card(&resident, &card).unwrap()));
        let mut tiered = layer_of(&card, experts, used, hidden, ffn, formats);
        let tier = holding_tier(&tiered, &card, experts, holding);
        tiered.tier = Some(tier.clone());
        let missing_before: Vec<bool> = (0..experts).map(|e| !tier.resident(e)).collect();
        let heat_before = tier.state.lock().unwrap().heat.clone();
        let owners_before = tier.state.lock().unwrap().owner.clone();
        let (copied_before, promoted_before) = (tier.copied_to_room(), tier.counters().2);
        tier.hold_stacks(true).unwrap();
        for chunk in 0..3u64 {
            let x = values(100 * hidden, 300 + chunk, 1.0);
            let want = prompt_rows(&resident, &card, &x, hidden);
            let got = prompt_rows(&tiered, &card, &x, hidden);
            assert!(
                got == want,
                "{holding:?} chunk {chunk}: not the bits of a resident layer"
            );
        }
        let stage = tier.stage.as_ref().unwrap();
        let placed = stage.placed.lock().unwrap().clone();
        let routed: Vec<u32> = tier
            .prompt_ids
            .lock()
            .unwrap()
            .as_ref()
            .map(|(_, ids)| ids.clone())
            .expect("the held chunks keep their routing");
        let mut want_placed: Vec<u32> = routed
            .iter()
            .copied()
            .filter(|&e| missing_before[e as usize])
            .collect();
        want_placed.sort_unstable();
        want_placed.dedup();
        assert!(
            !want_placed.is_empty(),
            "{holding:?}: the prompt missed no expert"
        );
        // The end of the hold ranks each routed expert at the share of its uses in the hottest
        // one's, keeping a hotter heat. An expert of the room takes a slot only when it ends
        // hotter than the coldest slot.
        let mut uses = vec![0f32; experts];
        for &e in &routed {
            uses[e as usize] += 1.0;
        }
        let top = uses.iter().copied().fold(1f32, f32::max);
        let after: Vec<f32> = (0..experts)
            .map(|e| heat_before[e].max(uses[e] / top))
            .collect();
        let coldest_slot = owners_before
            .iter()
            .filter(|&&o| o >= 0)
            .map(|&o| after[o as usize])
            .fold(f32::INFINITY, f32::min);
        let hottest_placed = placed
            .iter()
            .map(|&e| after[e as usize])
            .fold(0f32, f32::max);
        assert_eq!(
            hottest_placed > coldest_slot,
            moves,
            "{holding:?}: the coldest slot ends at {coldest_slot}, the room's hottest at {hottest_placed}"
        );
        let mut sorted = placed.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted, want_placed,
            "{holding:?}: the room holds other experts than the missed ones"
        );
        assert_eq!(
            tier.copied_to_room() - copied_before,
            placed.len() as u64,
            "{holding:?}: an expert came to the room more than once"
        );
        tier.hold_stacks(false).unwrap();
        assert!(
            stage.placed.lock().unwrap().is_empty(),
            "{holding:?}: the room was not handed on"
        );
        let promoted = tier.counters().2 - promoted_before;
        if moves {
            assert!(
                promoted > 0,
                "no expert of the room took the slot of a colder one"
            );
            let now_held: Vec<u32> = placed
                .iter()
                .copied()
                .filter(|&e| tier.resident(e as usize))
                .collect();
            assert_eq!(
                now_held.len() as u64,
                promoted,
                "the experts moved into slots are not those of the room"
            );
        } else {
            assert_eq!(promoted, 0, "an expert took the slot of a hotter one");
            assert_eq!(
                tier.state.lock().unwrap().owner,
                owners_before,
                "the slots changed hands"
            );
        }
        // The cache the hold left serves the next prompt with the same bits.
        let x = values(100 * hidden, 400, 1.0);
        assert!(
            prompt_rows(&tiered, &card, &x, hidden) == prompt_rows(&resident, &card, &x, hidden),
            "{holding:?}: the next prompt is not the bits of a resident layer"
        );
    }

    /// A routed pair's row is the same bits whether its expert sits in a slot of the cache, in the
    /// transit room or on the host, in each format the exact kernels serve, through an eager
    /// decode and a recorded one, for a token and for the rows of a verify: the output of a
    /// layer does not change with what the cache holds or how the misses are split.
    #[test]
    fn the_card_the_transit_room_and_the_host_give_the_same_bits() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let formats = [
            [GgmlDType::Q4K, GgmlDType::Q6K],
            [GgmlDType::Q4K, GgmlDType::Q4K],
            [GgmlDType::Q6K, GgmlDType::Q6K],
            [GgmlDType::Q2K, GgmlDType::Q2K],
            [GgmlDType::Q2K, GgmlDType::Q4K],
            [GgmlDType::Iq2S, GgmlDType::Iq2S],
            [GgmlDType::Iq3Xxs, GgmlDType::Iq3Xxs],
            [GgmlDType::Iq3S, GgmlDType::Iq3S],
            [GgmlDType::Iq4Nl, GgmlDType::Iq4Nl],
            [GgmlDType::Q2_0, GgmlDType::Q2_0],
            [GgmlDType::Q8_0, GgmlDType::Q8_0],
            [GgmlDType::Q5_0, GgmlDType::Q5_0],
        ];
        same_bits_wherever_pairs_run(&card, (16, 4, 512, 256), &formats);
    }

    /// The same at the widths of a released file whose expert width is whole runs of 32 values
    /// and not whole blocks of 256, its down projections in 32-value formats.
    #[test]
    fn the_same_bits_wherever_pairs_run_at_an_expert_width_of_whole_runs() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let formats = [
            [GgmlDType::Q4K, GgmlDType::Q8_0],
            [GgmlDType::Q4K, GgmlDType::Q5_0],
        ];
        same_bits_wherever_pairs_run(&card, (16, 4, 2560, 640), &formats);
    }

    /// A routed pair's row is the same bits whether its expert sits in a slot of the cache, in the
    /// transit room or on the host, through an eager decode and a recorded one, for a token and
    /// for the rows of a verify, in a layer of `shape` (experts, experts a row routes to, hidden
    /// width, expert width) in each of `formats` (gate and up, down), unbounded and bounded.
    fn same_bits_wherever_pairs_run(
        card: &Device,
        (experts, used, hidden, ffn): (usize, usize, usize, usize),
        formats: &[[GgmlDType; 2]],
    ) {
        let host = Device::Cpu;
        for (formats, limit) in formats
            .iter()
            .copied()
            .flat_map(|f| [(f, 0.0f32), (f, BOUNDED_SWIGLU_LIMIT)])
        {
            let bounded = |mut m: MoeWeights| {
                m.swiglu_limit = (limit > 0.0).then_some(limit);
                m
            };
            let moe = bounded(layer_of(card, experts, used, hidden, ffn, formats));
            assert!(serves(&moe), "{formats:?} bounded at {limit}");
            let stacks = [
                moe.gate_exps_sep.clone().unwrap(),
                moe.up_exps_sep.clone().unwrap(),
                moe.down_exps.clone(),
            ];
            let exact = exact::Exact::of(&stacks, limit).expect("a format with exact kernels");
            let widest = [0, 1, 2].map(|i| stacks[i].native_qtensor().data().len());
            let stager = CardStager::holding(card, widest, experts, used).unwrap();
            for rows in [1usize, 3] {
                let ids: Vec<u32> = (0..rows)
                    .flat_map(|r| [1u32, 6, 9, 14].map(|e| (e + r as u32) % experts as u32))
                    .collect();
                let w: Vec<f32> = (0..rows * used).map(|i| 0.1 + 0.05 * i as f32).collect();
                let x = values(rows * hidden, 31 + rows as u64, 1.0);
                let pairs: Vec<(usize, u32, f32)> =
                    (0..rows * used).map(|p| (p / used, ids[p], w[p])).collect();
                let want = exact::sum_in_order(
                    &exact::host_pairs(&exact, &stacks, &x, &pairs).unwrap(),
                    used,
                    hidden,
                );
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                // All missing, half, none; the misses split from mostly the host's to mostly the
                // transit room's.
                for slots in [1usize, 8, experts] {
                    for eighths in [1u64, 4, 7] {
                        for recorded in [false, true] {
                            let mut tiered =
                                bounded(layer_of(card, experts, used, hidden, ffn, formats));
                            // The stacks page-locked: a copy from pageable memory waits for a
                            // card that waits on the worker.
                            let built = LayerTier::new(&tiered, card, slots)
                                .unwrap()
                                .with_stage(&stager, 0, true)
                                .unwrap();
                            // Paces in milliseconds, for the watchdog's bound to stand well past
                            // a first copy.
                            for class in 0..2 {
                                built.host_us[class]
                                    .store((1e3 * eighths as f64).to_bits(), Ordering::Relaxed);
                                built.upload_us[class].store(
                                    (1e3 * (8 - eighths) as f64).to_bits(),
                                    Ordering::Relaxed,
                                );
                                built.paced[class].store(TRANSIT_PACE_SAMPLES, Ordering::Relaxed);
                            }
                            tiered.tier = Some(Arc::new(built));
                            let tier = tiered.tier.clone().unwrap();
                            let worker = if recorded {
                                tier.enable_graph(&tiered, 4).unwrap();
                                let w =
                                    TierWorker::start(card, vec![("layer".into(), tier.clone())])
                                        .unwrap();
                                GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
                                Some(w)
                            } else {
                                None
                            };
                            let xs = Tensor::from_vec(x.clone(), (rows, hidden), card).unwrap();
                            let ids_t = Tensor::from_vec(ids.clone(), (rows, used), &host)
                                .unwrap()
                                .to_device(card)
                                .unwrap();
                            let w_t = Tensor::from_vec(w.clone(), (rows, used), &host)
                                .unwrap()
                                .to_device(card)
                                .unwrap();
                            let got = match (recorded, rows) {
                                (_, 1) => tier.decode(&tiered, &xs, &ids_t, &w_t),
                                (true, _) => tier.decode_rows_graph(&tiered, &xs, &ids_t, &w_t),
                                (false, _) => tier.decode_rows(&tiered, &xs, &ids_t, &w_t),
                            }
                            .unwrap();
                            // The card waits on the worker, whose copies from pageable memory
                            // queue behind a read of the card's from the same: the stream is
                            // waited for first.
                            card.as_cuda_device()
                                .unwrap()
                                .0
                                .stream()
                                .synchronize()
                                .unwrap();
                            let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
                            let failure = worker.as_ref().and_then(|w| w.failure());
                            assert!(failure.is_none(), "{failure:?}");
                            // The tier is dropped here, once the copies into its slots let go of
                            // it: freeing its page-locked memory waits for the card, which must
                            // not be waiting on the next tier's worker then.
                            tier.settle();
                            drop(worker);
                            while Arc::strong_count(&tier) > 2 {
                                std::thread::yield_now();
                            }
                            drop(tiered);
                            drop(tier);
                            assert!(
                                bits(&got) == bits(&want),
                                "{formats:?} bounded at {limit}, {rows} rows, {slots} slots, \
                                 {eighths}/8 to transit, recorded {recorded}: the layer's output \
                                 moved with the cache"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The shape of the layers the bits-by-place tests build: experts, experts a row routes
    /// to, hidden width, expert width.
    const PLACE_SHAPE: (usize, usize, usize, usize) = (16, 4, 256, 512);

    /// Where a test request finds its experts: `slots` slots holding the first experts, the
    /// misses split between the transit room and the host in `eighths` of eight to the room
    /// where the level splits them (`splits`), a transit room of `room` experts (`0`: as the
    /// card's rates size it), and the recorded decode when `recorded`.
    #[derive(Clone, Copy, Debug)]
    struct Place {
        slots: usize,
        eighths: u64,
        recorded: bool,
        room: usize,
        splits: bool,
    }

    impl Place {
        /// `slots` slots, the misses even between the room and the host, eager, below fast.
        fn slots(slots: usize) -> Self {
            Self {
                slots,
                eighths: 4,
                recorded: false,
                room: 0,
                splits: false,
            }
        }
    }

    /// The rows `x` routed to `ids` with weights `w`, through a fresh tier over a layer of
    /// `formats` placed as `place` says. `run` makes the request; the output comes back as bits.
    #[allow(clippy::too_many_arguments)]
    fn bits_through_tier(
        card: &Device,
        formats: [GgmlDType; 2],
        place: Place,
        x: &[f32],
        ids: &[u32],
        w: &[f32],
        run: impl Fn(&Arc<LayerTier>, &MoeWeights, &Tensor, &Tensor, &Tensor) -> Vec<f32>,
    ) -> Vec<u32> {
        let host = Device::Cpu;
        let (experts, used, hidden, ffn) = PLACE_SHAPE;
        let rows = x.len() / hidden;
        let mut tiered = layer_of(card, experts, used, hidden, ffn, formats);
        assert!(
            serves(&tiered),
            "{formats:?}: the tier does not serve the layer"
        );
        let widest = [
            tiered.gate_exps_sep.as_ref().unwrap(),
            tiered.up_exps_sep.as_ref().unwrap(),
            &tiered.down_exps,
        ]
        .map(|t| t.native_qtensor().data().len());
        // The room holds the request's pairs: a token's, as the load sizes it for these formats,
        // and those of the rows of a verify recorded whole.
        let mut stager = CardStager::holding(card, widest, experts, rows * used).unwrap();
        assert!(
            stager.transit >= (rows * used).min(experts),
            "the room holds fewer than the request's pairs"
        );
        if place.room > 0 {
            Arc::get_mut(&mut stager).unwrap().transit = place.room;
        }
        // The stacks page-locked: a copy from pageable memory waits for a card that waits on the
        // worker.
        let built = LayerTier::new(&tiered, card, place.slots)
            .unwrap()
            .with_stage(&stager, 0, true)
            .unwrap();
        let eighths = place.eighths;
        for class in 0..2 {
            built.host_us[class].store((1e3 * eighths as f64).to_bits(), Ordering::Relaxed);
            built.upload_us[class].store((1e3 * (8 - eighths) as f64).to_bits(), Ordering::Relaxed);
            built.paced[class].store(TRANSIT_PACE_SAMPLES, Ordering::Relaxed);
        }
        built.splits_misses.store(place.splits, Ordering::Relaxed);
        tiered.tier = Some(Arc::new(built));
        let tier = tiered.tier.clone().unwrap();
        let worker = if place.recorded {
            tier.enable_graph(&tiered, 4).unwrap();
            // A room short of a token's pairs leaves the token to the eager decode.
            assert_eq!(
                tier.graph_rows() > 0,
                place.room == 0 || place.room >= used || place.splits || tier.exact.is_some(),
                "{place:?}: the recorded decode runs where the room holds a token"
            );
            let w = TierWorker::start(card, vec![("layer".into(), tier.clone())]).unwrap();
            GRAPH_WITHOUT_CAPTURE.store(true, Ordering::Relaxed);
            Some(w)
        } else {
            None
        };
        let xs = Tensor::from_vec(x.to_vec(), (rows, hidden), card).unwrap();
        let ids_t = Tensor::from_vec(ids.to_vec(), (rows, used), &host)
            .unwrap()
            .to_device(card)
            .unwrap();
        let w_t = Tensor::from_vec(w.to_vec(), (rows, used), &host)
            .unwrap()
            .to_device(card)
            .unwrap();
        let got = run(&tier, &tiered, &xs, &ids_t, &w_t);
        GRAPH_WITHOUT_CAPTURE.store(false, Ordering::Relaxed);
        let failure = worker.as_ref().and_then(|w| w.failure());
        assert!(failure.is_none(), "{failure:?}");
        tier.settle();
        drop(worker);
        while Arc::strong_count(&tier) > 2 {
            std::thread::yield_now();
        }
        drop(tiered);
        drop(tier);
        got.iter().map(|f| f.to_bits()).collect()
    }

    /// The output of one tier request on the card, as host floats.
    fn request_rows(card: &Device, request: Result<Tensor>) -> Vec<f32> {
        let out = request.unwrap();
        card.as_cuda_device()
            .unwrap()
            .0
            .stream()
            .synchronize()
            .unwrap();
        out.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// One token whose routed pairs all hit the cache, whose one missing pair goes by the paces
    /// to the host or to the transit room, and whose 2 missing pairs pass a room of 1 expert in
    /// 2 rounds, through the eager decode or the recorded one as `recorded` says: the places
    /// where the output's bits differ from those of the eager token that hits every pair.
    fn pair_places_that_move_the_bits(
        card: &Device,
        formats: [GgmlDType; 2],
        recorded: bool,
    ) -> Vec<String> {
        let (experts, used, hidden, _) = PLACE_SHAPE;
        let ids: Vec<u32> = vec![1, 6, 9, 14];
        let w: Vec<f32> = (0..used).map(|i| 0.1 + 0.05 * i as f32).collect();
        let x = values(hidden, 77, 1.0);
        let token =
            |tier: &Arc<LayerTier>, moe: &MoeWeights, xs: &Tensor, ids: &Tensor, w: &Tensor| {
                request_rows(card, tier.decode(moe, xs, ids, w))
            };
        // The first experts are in the slots: with all but the last 2 there, expert 14 misses;
        // with 8 there, experts 9 and 14 do.
        let cases = [
            ("every pair hits", Place::slots(experts)),
            (
                "1 missing pair the paces give the host",
                Place {
                    eighths: 1,
                    ..Place::slots(experts - 2)
                },
            ),
            (
                "1 missing pair the paces give the transit room",
                Place {
                    eighths: 7,
                    ..Place::slots(experts - 2)
                },
            ),
            (
                "2 missing pairs through a room of 1",
                Place {
                    eighths: 7,
                    room: 1,
                    ..Place::slots(8)
                },
            ),
        ];
        let want = bits_through_tier(card, formats, Place::slots(experts), &x, &ids, &w, token);
        let mut moved = Vec::new();
        for (what, place) in cases {
            let place = Place { recorded, ..place };
            if bits_through_tier(card, formats, place, &x, &ids, &w, token) != want {
                moved.push(format!("{what}, recorded {recorded}"));
            }
        }
        moved
    }

    /// The 3 rows of a verify through the eager tier and through the recorded one, against the
    /// same 3 rows as 3 plain recorded steps: the verifies whose bits differ from the steps'.
    /// The rows route to experts no other row uses, 2 of them missing from the slots.
    fn verifies_that_differ_from_the_steps(card: &Device, formats: [GgmlDType; 2]) -> Vec<String> {
        let (experts, used, hidden, _) = PLACE_SHAPE;
        let rows = 3usize;
        let ids: Vec<u32> = (0..rows)
            .flat_map(|r| [1u32, 6, 9, 14].map(|e| (e + r as u32) % experts as u32))
            .collect();
        let w: Vec<f32> = (0..rows * used).map(|i| 0.1 + 0.05 * i as f32).collect();
        let x = values(rows * hidden, 91, 1.0);
        let slots = experts - 4;
        let steps =
            |tier: &Arc<LayerTier>, moe: &MoeWeights, xs: &Tensor, ids: &Tensor, w: &Tensor| {
                (0..rows)
                    .flat_map(|r| {
                        let one = |t: &Tensor| t.narrow(0, r, 1).unwrap().contiguous().unwrap();
                        request_rows(card, tier.decode(moe, &one(xs), &one(ids), &one(w)))
                    })
                    .collect::<Vec<f32>>()
            };
        let recorded = Place {
            recorded: true,
            ..Place::slots(slots)
        };
        let want = bits_through_tier(card, formats, recorded, &x, &ids, &w, steps);
        let mut differ = Vec::new();
        let eager = bits_through_tier(
            card,
            formats,
            Place::slots(slots),
            &x,
            &ids,
            &w,
            |tier, moe, xs, ids, w| request_rows(card, tier.decode_rows(moe, xs, ids, w)),
        );
        if eager != want {
            differ.push("the eager verify".to_string());
        }
        let recorded = bits_through_tier(
            card,
            formats,
            recorded,
            &x,
            &ids,
            &w,
            |tier, moe, xs, ids, w| request_rows(card, tier.decode_rows_graph(moe, xs, ids, w)),
        );
        if recorded != want {
            differ.push("the recorded verify".to_string());
        }
        differ
    }

    /// Expert formats with exact kernels: gate and up, down.
    const FORMATS_WITH_EXACT_KERNELS: [[GgmlDType; 2]; 9] = [
        [GgmlDType::Q4K, GgmlDType::Q6K],
        [GgmlDType::Iq2S, GgmlDType::Q6K],
        [GgmlDType::Iq3Xxs, GgmlDType::Q6K],
        [GgmlDType::Iq3S, GgmlDType::Q6K],
        [GgmlDType::Iq2S, GgmlDType::Iq4Nl],
        [GgmlDType::Iq3S, GgmlDType::Iq4Nl],
        [GgmlDType::Iq3Xxs, GgmlDType::Q2_0],
        [GgmlDType::Q4K, GgmlDType::Q8_0],
        [GgmlDType::Q4K, GgmlDType::Q5_0],
    ];

    /// Expert formats the tier serves without exact kernels, gate and up, down: a gate, up or
    /// down stack in a format they do not read sends the whole layer to the fallback.
    const FORMATS_WITHOUT_EXACT_KERNELS: [[GgmlDType; 2]; 5] = [
        [GgmlDType::Q5K, GgmlDType::Q5K],
        [GgmlDType::Q3K, GgmlDType::Q5_1],
        [GgmlDType::Iq4Xs, GgmlDType::Q5_1],
        [GgmlDType::Q5_1, GgmlDType::Q5_1],
        [GgmlDType::Q4K, GgmlDType::Q5K],
    ];

    /// Whether a layer of `formats` at the test shape takes the exact kernels: a test meant for
    /// one side fails when its layer takes the other.
    fn takes_exact_kernels(formats: [GgmlDType; 2]) -> bool {
        let (experts, used, hidden, ffn) = PLACE_SHAPE;
        let moe = layer_of(&Device::Cpu, experts, used, hidden, ffn, formats);
        let stacks = [
            moe.gate_exps_sep.clone().unwrap(),
            moe.up_exps_sep.clone().unwrap(),
            moe.down_exps.clone(),
        ];
        exact::Exact::of(&stacks, 0.0).is_some()
    }

    /// The places of [`pair_places_that_move_the_bits`] that move the bits, for every format of
    /// `formats`, through the eager decode or the recorded one.
    fn places_that_move_the_bits(formats: &[[GgmlDType; 2]], recorded: bool) -> Vec<String> {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return Vec::new();
        };
        formats
            .iter()
            .filter_map(|&f| {
                let moved = pair_places_that_move_the_bits(&card, f, recorded);
                (!moved.is_empty()).then(|| format!("{f:?}: {moved:?}"))
            })
            .collect()
    }

    /// A routed pair gives the same bits wherever it runs, in the formats with exact kernels:
    /// a slot, the host or the transit room, eager or recorded.
    #[test]
    fn a_pair_gives_the_same_bits_wherever_it_runs_in_formats_with_exact_kernels() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        assert!(FORMATS_WITH_EXACT_KERNELS
            .iter()
            .all(|&f| takes_exact_kernels(f)));
        let mut moved = places_that_move_the_bits(&FORMATS_WITH_EXACT_KERNELS, false);
        moved.extend(places_that_move_the_bits(&FORMATS_WITH_EXACT_KERNELS, true));
        assert!(moved.is_empty(), "the bits moved with the place: {moved:?}");
    }

    /// The same in the formats without exact kernels, through the eager decode: a missing pair
    /// runs through the transit room with the kernels of the slots, never on the host.
    #[test]
    fn a_pair_gives_the_same_bits_wherever_it_runs_eagerly_in_formats_without_exact_kernels() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!FORMATS_WITHOUT_EXACT_KERNELS
            .iter()
            .any(|&f| takes_exact_kernels(f)));
        let moved = places_that_move_the_bits(&FORMATS_WITHOUT_EXACT_KERNELS, false);
        assert!(
            moved.is_empty(),
            "the bits moved with the place:\n{}",
            moved.join("\n")
        );
    }

    /// The same through the recorded decode.
    #[test]
    fn a_pair_gives_the_same_bits_wherever_it_runs_recorded_in_formats_without_exact_kernels() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!FORMATS_WITHOUT_EXACT_KERNELS
            .iter()
            .any(|&f| takes_exact_kernels(f)));
        let moved = places_that_move_the_bits(&FORMATS_WITHOUT_EXACT_KERNELS, true);
        assert!(
            moved.is_empty(),
            "the bits moved with the place:\n{}",
            moved.join("\n")
        );
    }

    /// The host time and the transit copies of one token whose missing pairs the paces give the
    /// host, at a level that splits them or not: `(host time before, after, copies to transit)`.
    fn host_and_transit_of_a_token(splits: bool) -> Option<(u64, u64, u64)> {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return None;
        };
        let (experts, used, hidden, _) = PLACE_SHAPE;
        let ids: Vec<u32> = vec![1, 6, 9, 14];
        let w: Vec<f32> = (0..used).map(|i| 0.1 + 0.05 * i as f32).collect();
        let x = values(hidden, 77, 1.0);
        let seen = std::cell::Cell::new((0u64, 0u64, 0u64));
        let place = Place {
            eighths: 1,
            splits,
            ..Place::slots(experts - 8)
        };
        let formats = FORMATS_WITHOUT_EXACT_KERNELS[0];
        assert!(!takes_exact_kernels(formats));
        bits_through_tier(
            &card,
            formats,
            place,
            &x,
            &ids,
            &w,
            |tier, moe, xs, ids, w| {
                let host = || tier.host_us[0].load(Ordering::Relaxed);
                let (before, copied) = (host(), tier.uploaded.load(Ordering::Relaxed));
                let out = request_rows(&card, tier.decode(moe, xs, ids, w));
                let copied = tier.uploaded.load(Ordering::Relaxed) - copied;
                seen.set((before, host(), copied));
                out
            },
        );
        Some(seen.get())
    }

    /// Below the fast level a layer without exact kernels runs no pair on the host, whatever
    /// the paces say: its 2 missing experts go through the transit room.
    #[test]
    fn a_layer_without_exact_kernels_runs_no_pair_on_the_host_below_fast() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Some((before, after, copied)) = host_and_transit_of_a_token(false) else {
            return;
        };
        assert_eq!(before, after, "the host computed a pair");
        assert_eq!(copied, 2, "{copied} experts went through the transit room");
    }

    /// At the fast level the paces split the missing experts: the host takes its share.
    #[test]
    fn a_layer_without_exact_kernels_splits_its_misses_at_fast() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Some((before, after, copied)) = host_and_transit_of_a_token(true) else {
            return;
        };
        assert_ne!(before, after, "the host computed no pair");
        assert!(copied < 2, "{copied} experts went through the transit room");
    }

    /// A verify of 3 rows gives the bits of the same rows as 3 plain recorded steps, in the
    /// formats with exact kernels.
    #[test]
    fn a_verify_gives_the_bits_of_plain_steps_in_formats_with_exact_kernels() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        for formats in FORMATS_WITH_EXACT_KERNELS {
            let differ = verifies_that_differ_from_the_steps(&card, formats);
            assert!(
                differ.is_empty(),
                "{formats:?}: not the bits of the steps: {differ:?}"
            );
        }
    }

    /// The same in the formats without exact kernels.
    #[test]
    fn a_verify_gives_the_bits_of_plain_steps_in_formats_without_exact_kernels() {
        let _pool = LENT_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        assert!(!FORMATS_WITHOUT_EXACT_KERNELS
            .iter()
            .any(|&f| takes_exact_kernels(f)));
        for formats in FORMATS_WITHOUT_EXACT_KERNELS {
            let differ = verifies_that_differ_from_the_steps(&card, formats);
            assert!(
                differ.is_empty(),
                "{formats:?}: not the bits of the steps: {differ:?}"
            );
        }
    }
}

#[cfg(test)]
mod probe_size_tests {
    /// A probe takes its bytes while the room holds them, and the smallest probe otherwise.
    #[test]
    fn a_probe_fits_the_room_or_takes_the_smallest_size() {
        assert_eq!(super::probe_size(64 << 20, 1 << 30), 64 << 20);
        assert_eq!(super::probe_size(64 << 20, 32 << 20), 1 << 20);
        assert_eq!(super::probe_size(4096, 1 << 30), 1 << 20);
        assert_eq!(super::probe_size(64 << 20, 0), 1 << 20);
    }
}

#[cfg(test)]
mod copy_stream_tests {
    /// A layer takes a stream of its own while the room holds what a stream costs.
    #[test]
    fn a_layer_takes_its_own_stream_while_the_room_holds_one() {
        assert!(super::own_stream_fits(8 << 20, 1 << 30));
        assert!(!super::own_stream_fits(8 << 20, 4 << 20));
        // A stream that costs nothing measurable is always taken.
        assert!(super::own_stream_fits(0, 0));
    }
}
