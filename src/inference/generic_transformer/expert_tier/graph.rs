//! A cached layer's decoded token, or the rows of a draft verify, as kernels alone, so that a
//! graph can record them.
//!
//! The layer's missing experts are computed by a thread of the host. The card hands it a
//! mailbox in page-locked memory (the router's choice of each row, the slots the cache served,
//! the rows' input), raises a request word, waits on a reply word with a kernel, and adds the
//! product the host left in the mailbox. Neither side returns to the code that launched it
//! between two kernels, which is what a recorded graph needs and what an eager decode pays a
//! round trip for.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::tensor::gpu_runtime::driver::sys::{self, CUresult};

use super::*;

fn check(what: &str, r: CUresult) -> Result<()> {
    if r == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(crate::tensor::Error::msg(format!(
            "tier mailbox {what}: {r:?}"
        )))
    }
}

/// `bytes` of page-locked memory the card addresses, zeroed: the host's pointer and the card's.
pub(crate) fn mapped(dev: &Device, bytes: usize) -> Result<(*mut u8, u64)> {
    dev.as_cuda_device()?
        .0
        .context()
        .bind_to_thread()
        .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
    let mut host = std::ptr::null_mut();
    check("alloc", unsafe {
        sys::cuMemHostAlloc(
            &mut host,
            bytes,
            sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
        )
    })?;
    unsafe { std::ptr::write_bytes(host as *mut u8, 0, bytes) };
    let mut device = 0;
    check("map", unsafe {
        sys::cuMemHostGetDevicePointer_v2(&mut device, host, 0)
    })?;
    Ok((host as *mut u8, device))
}

/// The cache's table for the card: the slot each expert holds, or a negative number.
pub(super) struct SlotTable {
    host: *mut i32,
    device: u64,
    n: usize,
}

unsafe impl Send for SlotTable {}
unsafe impl Sync for SlotTable {}

impl SlotTable {
    fn new(dev: &Device, slot_of: &[i32]) -> Result<Self> {
        let (host, device) = mapped(dev, 4 * slot_of.len().max(1))?;
        let table = Self {
            host: host as *mut i32,
            device,
            n: slot_of.len(),
        };
        for (e, &s) in slot_of.iter().enumerate() {
            table.set(e, s);
        }
        Ok(table)
    }

    pub(super) fn set(&self, expert: usize, slot: i32) {
        debug_assert!(expert < self.n);
        unsafe { std::ptr::write_volatile(self.host.add(expert), slot) };
    }
}

impl Drop for SlotTable {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

/// What the card posts for a layer's request and the host answers: room for `rows_max` rows.
pub(super) struct Mailbox {
    host: *mut u8,
    device: u64,
    k: usize,
    hidden: usize,
    rows_max: usize,
    ids: usize,
    weights: usize,
    slots: usize,
    x: usize,
    product: usize,
    /// The late pairs: each one's index among the request's pairs, its expert's place in the
    /// transit room, and its weight.
    late_pairs: usize,
    late_experts: usize,
    late_weights: usize,
}

unsafe impl Send for Mailbox {}
unsafe impl Sync for Mailbox {}

/// Offsets of the request, reply, abort and rows words, each on a line of its own: the card
/// writes the request and the rows, the host the other two.
const REQUEST: usize = 0;
const REPLY: usize = 64;
const ABORT: usize = 128;
const ROWS: usize = 192;
/// The number of late pairs: routed pairs whose expert the host copied into the transit room
/// for this request, which the card computes from there once the host has answered.
const LIVE: usize = 256;
const BODY: usize = 320;

fn line(bytes: usize) -> usize {
    bytes.div_ceil(64) * 64
}

/// The request as the host reads it, with the choice of each of its rows.
pub(super) struct Request {
    pub seq: u32,
    pub rows: usize,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    /// The slot the cache served each expert from, `u32::MAX` for a missing one.
    pub slots: Vec<u32>,
    pub x: Vec<f32>,
}

impl Mailbox {
    fn new(dev: &Device, k: usize, hidden: usize, rows_max: usize) -> Result<Self> {
        let pairs = rows_max * k;
        let ids = BODY;
        let weights = ids + line(4 * pairs);
        let slots = weights + line(4 * pairs);
        let x = slots + line(4 * pairs);
        let product = x + line(4 * hidden * rows_max);
        let late_pairs = product + line(4 * hidden * rows_max);
        let late_experts = late_pairs + line(4 * pairs);
        let late_weights = late_experts + line(4 * pairs);
        let (host, device) = mapped(dev, late_weights + line(4 * pairs))?;
        Ok(Self {
            host,
            device,
            k,
            hidden,
            rows_max,
            ids,
            weights,
            slots,
            x,
            product,
            late_pairs,
            late_experts,
            late_weights,
        })
    }

    fn word(&self, at: usize) -> *mut u32 {
        unsafe { self.host.add(at) as *mut u32 }
    }

    fn dptr(&self, at: usize) -> u64 {
        self.device + at as u64
    }

    /// The request number the card last posted.
    pub(super) fn request(&self) -> u32 {
        unsafe { std::ptr::read_volatile(self.word(REQUEST)) }
    }

    /// The request, reply, abort and rows words, as they stand.
    fn words(&self) -> [u32; 4] {
        [REQUEST, REPLY, ABORT, ROWS].map(|at| unsafe { std::ptr::read_volatile(self.word(at)) })
    }

    /// The posted request, once its number is read: the card writes the body before the number.
    pub(super) fn read(&self) -> Request {
        loop {
            let seq = self.request();
            std::sync::atomic::fence(Ordering::Acquire);
            let rows = (unsafe { std::ptr::read_volatile(self.word(ROWS)) } as usize)
                .clamp(1, self.rows_max);
            let pairs = rows * self.k;
            let at = |off: usize| unsafe { self.host.add(off) };
            let ids = unsafe { std::slice::from_raw_parts(at(self.ids) as *const u32, pairs) };
            let weights =
                unsafe { std::slice::from_raw_parts(at(self.weights) as *const f32, pairs) };
            let slots = unsafe { std::slice::from_raw_parts(at(self.slots) as *const u32, pairs) };
            let x =
                unsafe { std::slice::from_raw_parts(at(self.x) as *const f32, self.hidden * rows) };
            let read = Request {
                seq,
                rows,
                ids: ids.to_vec(),
                weights: weights.to_vec(),
                slots: slots.to_vec(),
                x: x.to_vec(),
            };
            // A request posted over this one while it was read: read the newer.
            std::sync::atomic::fence(Ordering::Acquire);
            if self.request() == seq {
                return read;
            }
        }
    }

    /// Leave `product`, one row per row of the request, for the card and raise the reply word
    /// to `seq`.
    /// How many values of the last request's rows and of the host's product are not finite,
    /// over the rows the request named: `(rows, x not finite, product not finite)`.
    pub(super) fn finite_report(&self) -> (usize, usize, usize) {
        let rows =
            (unsafe { std::ptr::read_volatile(self.word(ROWS)) } as usize).clamp(1, self.rows_max);
        let count = |off: usize, n: usize| unsafe {
            std::slice::from_raw_parts(self.host.add(off) as *const f32, n)
                .iter()
                .filter(|v| !v.is_finite())
                .count()
        };
        (
            rows,
            count(self.x, self.hidden * rows),
            count(self.product, self.hidden * rows),
        )
    }

    pub(super) fn answer(&self, product: &[f32], seq: u32) {
        debug_assert!(product.len() <= self.hidden * self.rows_max);
        unsafe {
            std::ptr::copy_nonoverlapping(
                product.as_ptr(),
                self.host.add(self.product) as *mut f32,
                product.len(),
            );
        }
        std::sync::atomic::fence(Ordering::Release);
        unsafe { std::ptr::write_volatile(self.word(REPLY), seq) };
    }

    /// The late pairs of the request being answered, before its answer: `(pair, place, weight)`
    /// each, the place being the expert's in the transit room. The kernels read the pairs and
    /// their places in the order given, and a pair's weight at the pair's own index.
    pub(super) fn hand_late(&self, late: &[(u32, u32, f32)]) {
        debug_assert!(late.len() <= self.rows_max * self.k);
        unsafe {
            let pairs = self.host.add(self.late_pairs) as *mut u32;
            let experts = self.host.add(self.late_experts) as *mut u32;
            let weights = self.host.add(self.late_weights) as *mut f32;
            for (i, &(pair, place, weight)) in late.iter().enumerate() {
                pairs.add(i).write(pair);
                experts.add(i).write(place);
                weights.add(pair as usize).write(weight);
            }
            std::ptr::write_volatile(self.word(LIVE), late.len() as u32);
        }
    }

    /// Raise the reply word to `seq`, for a request that needs no product.
    pub(super) fn raise(&self, seq: u32) {
        std::sync::atomic::fence(Ordering::Release);
        unsafe { std::ptr::write_volatile(self.word(REPLY), seq) };
    }

    /// Release a card waiting for an answer that is not coming.
    pub(super) fn abort(&self) {
        unsafe { std::ptr::write_volatile(self.word(ABORT), 1) };
    }

    /// Let the card wait for its answers again.
    fn clear_abort(&self) {
        unsafe { std::ptr::write_volatile(self.word(ABORT), 0) };
    }
}

impl Drop for Mailbox {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

/// What the requests of every layer cost the host, since the load.
struct WorkerStats {
    requests: std::sync::atomic::AtomicU64,
    ns: std::sync::atomic::AtomicU64,
    /// Requests with a missing expert, the experts they were missing, and the nanoseconds their
    /// products took.
    missed: std::sync::atomic::AtomicU64,
    missing: std::sync::atomic::AtomicU64,
    product_ns: std::sync::atomic::AtomicU64,
    /// Of the products' nanoseconds: readying the rows and the routing, the gate and up
    /// projections, the down projection and the sum per row.
    prep_ns: std::sync::atomic::AtomicU64,
    gate_up_ns: std::sync::atomic::AtomicU64,
    down_ns: std::sync::atomic::AtomicU64,
    /// Experts copied into a slot.
    promotions: std::sync::atomic::AtomicU64,
    /// Missing experts the transit room took.
    transit: std::sync::atomic::AtomicU64,
    /// Requests of one row and of several, and those of each that missed.
    rows_one: std::sync::atomic::AtomicU64,
    rows_many: std::sync::atomic::AtomicU64,
    missed_one: std::sync::atomic::AtomicU64,
    missed_many: std::sync::atomic::AtomicU64,
}

/// One more expert landed in a slot of some layer.
pub(super) fn note_promotion() {
    WORKER.promotions.fetch_add(1, Ordering::Relaxed);
}

static WORKER: WorkerStats = WorkerStats {
    requests: std::sync::atomic::AtomicU64::new(0),
    ns: std::sync::atomic::AtomicU64::new(0),
    missed: std::sync::atomic::AtomicU64::new(0),
    missing: std::sync::atomic::AtomicU64::new(0),
    product_ns: std::sync::atomic::AtomicU64::new(0),
    prep_ns: std::sync::atomic::AtomicU64::new(0),
    gate_up_ns: std::sync::atomic::AtomicU64::new(0),
    down_ns: std::sync::atomic::AtomicU64::new(0),
    promotions: std::sync::atomic::AtomicU64::new(0),
    transit: std::sync::atomic::AtomicU64::new(0),
    rows_one: std::sync::atomic::AtomicU64::new(0),
    rows_many: std::sync::atomic::AtomicU64::new(0),
    missed_one: std::sync::atomic::AtomicU64::new(0),
    missed_many: std::sync::atomic::AtomicU64::new(0),
};

/// The steps of a request the worker names in its heartbeat, by index.
const STEPS: [&str; 8] = [
    "waiting for a request",
    "reading a request",
    "counting the routing",
    "copying experts into the transit room",
    "computing the missing experts on the host",
    "waiting for the transit copies to land",
    "answering",
    "queueing promotions",
];
const IDLE: usize = 0;
const READING: usize = 1;
const COUNTING: usize = 2;
const TO_TRANSIT: usize = 3;
const ON_HOST: usize = 4;
const LANDING: usize = 5;
const ANSWERING: usize = 6;
const PROMOTING: usize = 7;

/// How many times the longest request the worker is known to take a request may go unanswered,
/// with the worker making no progress, before the worker counts as stalled.
const STALL_MARGIN: u64 = 64;

/// How many times the watchdog looks within one stall bound.
const LOOKS_A_BOUND: u32 = 4;

/// The heartbeat of a card's tier worker and what stands when it stalled or failed: the
/// worker writes the beat, its watchdog and the decode read it.
pub(super) struct Health {
    epoch: std::time::Instant,
    /// Nanoseconds from `epoch` to the worker's last step.
    beat: AtomicU64,
    /// The worker's layer, as its place in the list it serves, and its step, at the last beat.
    tier: AtomicUsize,
    step: AtomicUsize,
    /// Nanoseconds the longest request the worker answered took.
    longest: AtomicU64,
    /// Why the card's mailboxes were released, and the beat at that time.
    failure: Mutex<Option<(String, u64)>>,
    /// The worker thread returned: what it released stays released.
    exited: AtomicBool,
    /// Each recorded layer's place in the worker's list, its mailbox and the request the
    /// worker last answered there.
    mails: Vec<(usize, Arc<Mailbox>, Arc<AtomicU32>)>,
    /// The watchdog's thread, woken once the worker has a measured request.
    watchdog: OnceLock<std::thread::Thread>,
}

impl Health {
    fn new(mails: Vec<(usize, Arc<Mailbox>, Arc<AtomicU32>)>) -> Self {
        Self {
            epoch: std::time::Instant::now(),
            beat: AtomicU64::new(0),
            tier: AtomicUsize::new(0),
            step: AtomicUsize::new(IDLE),
            longest: AtomicU64::new(0),
            failure: Mutex::new(None),
            exited: AtomicBool::new(false),
            mails,
            watchdog: OnceLock::new(),
        }
    }

    fn now(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    /// The worker is at `step` of the request of its layer `tier`.
    fn beat(&self, tier: usize, step: usize) {
        self.tier.store(tier, Ordering::Relaxed);
        self.step.store(step, Ordering::Relaxed);
        self.beat.store(self.now(), Ordering::Release);
    }

    /// A request of `ns` was answered.
    fn answered(&self, ns: u64) {
        if self.longest.fetch_max(ns, Ordering::Relaxed) == 0 {
            if let Some(t) = self.watchdog.get() {
                t.unpark();
            }
        }
    }

    /// The first layer with a request posted and not answered: its place in the worker's list,
    /// its mailbox and the request's number.
    fn pending(&self) -> Option<(usize, &Mailbox, u32)> {
        self.mails.iter().find_map(|(i, m, done)| {
            let seq = m.request();
            (seq != done.load(Ordering::Relaxed)).then_some((*i, &**m, seq))
        })
    }

    /// Release every waiting card and keep `why` for the decode, unless a failure stands.
    fn fail(&self, why: String) {
        let mut f = self.failure.lock().unwrap_or_else(|e| e.into_inner());
        if f.is_none() {
            *f = Some((why, self.beat.load(Ordering::Acquire)));
        }
        for (_, m, _) in &self.mails {
            m.abort();
        }
    }

    /// Why the card's mailboxes stand released, if they do.
    pub(super) fn failure(&self) -> Option<String> {
        self.failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|(why, _)| why.clone())
    }

    /// Before a token runs through the card's layers: an error while a failure stands. A stall
    /// is lifted once the worker has moved on since and answered every request posted: the
    /// card's stream is waited for, so no wait of the failed step reads the cleared word.
    pub(super) fn check(&self, dev: &Device) -> Result<()> {
        let mut f = self.failure.lock().unwrap_or_else(|e| e.into_inner());
        let Some((why, at)) = f.as_ref() else {
            return Ok(());
        };
        let moved = self.beat.load(Ordering::Acquire) > *at;
        if self.exited.load(Ordering::Acquire)
            || !moved
            || self.pending().is_some()
            || crate::tensor::cuda_ext::capture_active(dev)
        {
            crate::tensor::bail!("expert tier: {why}");
        }
        dev.as_cuda_device()?
            .0
            .stream()
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("expert tier: {e:?}")))?;
        for (_, m, _) in &self.mails {
            m.clear_abort();
        }
        tracing::warn!("expert tier: the worker serves again after: {why}");
        *f = None;
        Ok(())
    }

    /// Nanoseconds a request may go unanswered while the worker does not move: [`STALL_MARGIN`]
    /// times the longest request it answered or the longest any of `tiers` takes at the paces
    /// measured. None before anything is measured.
    fn bound(&self, tiers: &[(String, Arc<LayerTier>)]) -> Option<u64> {
        let paced = tiers
            .iter()
            .map(|(_, t)| t.longest_request_ns())
            .fold(0, u64::max);
        let base = self.longest.load(Ordering::Relaxed).max(paced);
        (base > 0).then(|| base.saturating_mul(STALL_MARGIN))
    }
}

/// A layer's decoded rows between the post of their request and the wait for the answer.
pub(crate) struct GraphPending {
    card: Tensor,
    /// The rows' input, which the late experts read.
    x: Tensor,
    rows: usize,
    k: usize,
    hidden: usize,
    dev: Device,
}

/// A layer's side of the graph decode: the mailbox, the buffers the kernels write on the card,
/// the rows they have room for, and the number of the last request the host answered.
pub(super) struct GraphSide {
    mail: Arc<Mailbox>,
    counter: Tensor,
    slots: Tensor,
    weights: Tensor,
    rows_max: usize,
    done: Arc<AtomicU32>,
    /// The health of the worker that answers this layer, once one is started.
    health: OnceLock<Arc<Health>>,
    /// Around the copies of a request's late experts into the transit room.
    upload_start: crate::tensor::gpu_runtime::driver::CudaEvent,
    upload_end: crate::tensor::gpu_runtime::driver::CudaEvent,
}

impl LayerTier {
    /// The mailbox as it stands, for a wait that lasts: the request, reply, abort and rows
    /// words, what the worker served, and the card's counter read over the copy stream.
    pub fn mailbox_state(&self) -> Option<String> {
        let g = self.graph.get()?;
        let [request, reply, abort, rows] = g.mail.words();
        let done = g.done.load(Ordering::Relaxed);
        let counter = (|| -> Result<[u32; 2]> {
            use crate::tensor::gpu_runtime::driver::DevicePtr;
            let (storage, layout) = g.counter.storage_and_layout();
            let crate::tensor::StorageView::Cuda(c) = &*storage else {
                crate::tensor::bail!("the counter is on the card");
            };
            let slice = c.as_cuda_slice::<u32>()?;
            let src = slice.device_ptr(slice.stream()).0 + (layout.start_offset() * 4) as u64;
            let mut out = [0u32; 2];
            unsafe {
                crate::tensor::gpu_runtime::driver::result::memcpy_dtoh_async(
                    &mut out,
                    src,
                    self.copy.cu_stream(),
                )
                .map_err(|e| crate::tensor::Error::msg(format!("counter read: {e:?}")))?;
            }
            self.copy
                .synchronize()
                .map_err(|e| crate::tensor::Error::msg(format!("counter read: {e:?}")))?;
            Ok(out)
        })();
        let (fr, fx, fp) = g.mail.finite_report();
        Some(format!(
            "request {request} reply {reply} abort {abort} rows {rows} served {done} counter {:?}, last rows {fr}: x not finite {fx}, product not finite {fp}",
            counter.ok()
        ))
    }

    /// Nanoseconds the widest request of this layer takes when every pair misses, at the slower
    /// of the paces measured for a host product and a copy of an expert, the copy at the card's
    /// measured rate while no copy is timed. Zero before any is known.
    fn longest_request_ns(&self) -> u64 {
        let Some(g) = self.graph.get() else {
            return 0;
        };
        let mut us = [&self.host_us, &self.upload_us]
            .iter()
            .flat_map(|c| c.iter())
            .map(|c| f64::from_bits(c.load(Ordering::Relaxed)))
            .fold(0f64, f64::max);
        if let Some(rate) = self
            .stage
            .as_ref()
            .map(|s| s.stager.bandwidth)
            .filter(|&r| r > 0.0)
        {
            us = us.max(self.expert_bytes.iter().sum::<usize>() as f64 / rate);
        }
        (us * 1e3 * (g.rows_max * g.mail.k) as f64) as u64
    }

    /// An error while the worker serving this layer's recorded decode stands stalled or failed.
    pub(super) fn health_check(&self) -> Result<()> {
        match self.graph.get().and_then(|g| g.health.get()) {
            Some(h) => h.check(&self.dev),
            None => Ok(()),
        }
    }

    /// Let this layer's decoded token, and the rows of a verify up to `rows_max`, run as kernels
    /// alone, served by a [`TierWorker`].
    pub(super) fn enable_graph(&self, hidden: usize, k: usize, rows_max: usize) -> Result<()> {
        // The plan kernel looks the choices up with one thread per routed pair.
        let rows_max = rows_max.max(1);
        if self.graph.get().is_some()
            || self.resident
            || rows_max * k > crate::inference::moe_cuda::tier_graph::PLAN_PAIRS
        {
            return Ok(());
        }
        let host = Device::Cpu;
        let dev = &self.dev;
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let table = Arc::new(SlotTable::new(dev, &st.slot_of)?);
        st.table = Some(table.clone());
        drop(st);
        let pairs = rows_max * k;
        let side = GraphSide {
            mail: Arc::new(Mailbox::new(dev, k, hidden, rows_max)?),
            counter: Tensor::from_vec(vec![0u32; 2], (2,), &host)?.to_device(dev)?,
            slots: Tensor::from_vec(vec![0u32; pairs], (pairs,), &host)?.to_device(dev)?,
            weights: Tensor::from_vec(vec![0f32; pairs], (1, pairs), &host)?.to_device(dev)?,
            rows_max,
            done: Arc::new(AtomicU32::new(0)),
            health: OnceLock::new(),
            upload_start: super::timed_event(dev)?,
            upload_end: super::timed_event(dev)?,
        };
        let _ = self.graph.set(side);
        Ok(())
    }

    /// Rows of a verify a recorded decode can run through this layer at once.
    pub(crate) fn graph_rows(&self) -> usize {
        self.graph.get().map_or(0, |g| g.rows_max)
    }

    /// Whether a token on `dev` goes through the kernels of a recorded decode: one is being
    /// recorded, or a test runs them without.
    pub(crate) fn records(&self, dev: &Device) -> bool {
        self.graph_ready()
            && (crate::tensor::cuda_ext::capture_active(dev)
                || GRAPH_WITHOUT_CAPTURE.load(Ordering::Relaxed))
    }

    /// Whether a recorded decode can run this layer.
    pub(crate) fn graph_ready(&self) -> bool {
        self.graph.get().is_some()
    }

    /// One decoded token's expert block as kernels alone: `xs` `[1, hidden]` and the router's
    /// choice `ids`, `weights` `[1, k]`, all on the card. The host's share comes back through the
    /// mailbox. Returns `[1, hidden]` on the card.
    pub(crate) fn decode_graph(
        &self,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let pending = self.graph_begin(moe, xs, ids, weights)?;
        self.graph_end(pending)
    }

    /// The expert blocks of the rows of a draft verify as kernels alone: `xs` `[rows, hidden]`
    /// and the router's choices `ids`, `weights` `[rows, k]`, all on the card. Returns
    /// `[rows, hidden]` on the card.
    pub(crate) fn decode_rows_graph(
        &self,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let pending = self.graph_begin(moe, xs, ids, weights)?;
        self.graph_end(pending)
    }

    /// The first half of [`Self::decode_graph`]: the request is posted and the card's own experts
    /// run. What the card has to do that does not depend on the host's answer can be queued
    /// before [`Self::graph_end`] waits for it. `xs` is `[rows, hidden]`, the choice
    /// `[rows, k]`.
    /// Hand the mailbox late pairs as the host would, for a test of a request that must
    /// not read them.
    #[cfg(test)]
    pub(super) fn hand_late_for_test(&self, late: &[(u32, u32, f32)]) {
        if let Some(g) = self.graph.get() {
            g.mail.hand_late(late);
        }
    }

    pub(crate) fn graph_begin(
        &self,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<GraphPending> {
        use crate::inference::moe_cuda;
        let g = self.graph.get().expect("a graph layer is enabled");
        self.health_check()?;
        let k = moe.n_experts_used;
        let dev = xs.device().clone();
        let cd = dev.as_cuda_device()?;
        let stream = cd.0.stream().cu_stream() as i64;
        let x = xs.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
        let rows = ids.elem_count() / k.max(1);
        let hidden = x.elem_count() / rows.max(1);
        if rows == 0 || rows > g.rows_max {
            crate::tensor::bail!(
                "a recorded expert block of {rows} rows, with room for {}",
                g.rows_max
            );
        }
        let ids = ids.to_dtype(crate::tensor::DType::U32)?.contiguous()?;
        let w = weights.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
        let table = {
            let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.table
                .as_ref()
                .expect("a graph layer has its table")
                .device
        };
        unsafe {
            moe_cuda::tier_graph::plan(
                address_of::<u32>(&ids)? as *const u32,
                address_of::<f32>(&w)? as *const f32,
                address_of::<f32>(&x)? as *const f32,
                table as *const i32,
                address_of::<u32>(&g.counter)? as *mut u32,
                address_of::<u32>(&g.slots)? as *mut u32,
                address_of::<f32>(&g.weights)? as *mut f32,
                g.mail.dptr(g.mail.ids) as *mut u32,
                g.mail.dptr(g.mail.weights) as *mut f32,
                g.mail.dptr(g.mail.slots) as *mut u32,
                g.mail.dptr(g.mail.x) as *mut f32,
                g.mail.dptr(ROWS) as *mut u32,
                g.mail.dptr(REQUEST) as *mut u32,
                g.mail.dptr(LIVE) as *mut u32,
                rows as i32,
                k as i32,
                hidden as i32,
                stream,
            );
        }
        let pairs = rows * k;
        let slots = g.slots.narrow(0, 0, pairs)?;
        let weights = g.weights.narrow(1, 0, pairs)?;
        // One token's pairs are read as the router listed them; the rows of a verify are
        // grouped by slot, as the batched kernels read them.
        let routing = if rows == 1 {
            ExpertRouting {
                slots: moe.decode_slots.clone().expect("a tier layer is on a card"),
                experts: slots,
                topk: k,
            }
        } else {
            ExpertRouting::group(&slots.reshape((rows, k))?, k)?
        };
        // The extra stacks are read through the tier's table, so that a recorded launch follows
        // them when they are lent and come back elsewhere.
        let slots = self.extra_slots();
        let gate_up = self
            .extra_table_at(0)
            .map(|table| moe_cuda::MoreExperts::Table { table, slots });
        let h = routing.gate_up_silu_over(&x, &self.card[0], &self.card[1], gate_up)?;
        // Each pair's row, at its weight: the merge sums a row's pairs with the late ones and
        // the host's product.
        let down = self
            .extra_table_at(2)
            .map(|table| moe_cuda::MoreExperts::Table { table, slots });
        let card = routing.down_pairs_over(&h, &self.card[2], down, &weights)?;
        Ok(GraphPending {
            card,
            x,
            rows,
            k,
            hidden,
            dev,
        })
    }

    /// The second half: wait for the host's answer and add its product to the card's.
    pub(crate) fn graph_end(&self, pending: GraphPending) -> Result<Tensor> {
        use crate::inference::moe_cuda;
        let g = self.graph.get().expect("a graph layer is enabled");
        let GraphPending {
            card,
            x,
            rows,
            k,
            hidden,
            dev,
        } = pending;
        let cd = dev.as_cuda_device()?;
        let stream = cd.0.stream().cu_stream() as i64;
        let out = unsafe { cd.alloc::<f32>(rows * hidden) }?;
        unsafe {
            moe_cuda::tier_graph::wait(
                address_of::<u32>(&g.counter)? as *const u32,
                g.mail.dptr(REPLY) as *mut u32,
                g.mail.dptr(ABORT) as *mut u32,
                stream,
            );
        }
        // The late pairs, from the transit room the host filled while the card waited: the
        // launches run the count the host wrote, nothing when it wrote none.
        let late = match self.transit_stacks() {
            Some(stacks) => {
                let h = moe_cuda::moe_gemm_gguf_gate_up_silu_mul_live(
                    &x,
                    &stacks[0],
                    &stacks[1],
                    g.mail.dptr(g.mail.late_pairs),
                    g.mail.dptr(g.mail.late_experts),
                    g.mail.dptr(LIVE),
                    rows * k,
                    k,
                )?;
                Some(moe_cuda::moe_gemm_gguf_down_pairs_live(
                    &h,
                    &stacks[2],
                    g.mail.dptr(g.mail.late_pairs),
                    g.mail.dptr(g.mail.late_experts),
                    g.mail.dptr(g.mail.late_weights),
                    g.mail.dptr(LIVE),
                    k,
                )?)
            }
            None => None,
        };
        unsafe {
            use crate::tensor::gpu_runtime::driver::DevicePtr;
            let late_ptr = match &late {
                Some(t) => address_of::<f32>(t)? as *const f32,
                None => std::ptr::null(),
            };
            moe_cuda::tier_graph::merge(
                address_of::<f32>(&card)? as *const f32,
                late_ptr,
                g.mail.dptr(g.mail.product) as *const f32,
                address_of::<u32>(&g.counter)? as *const u32,
                out.device_ptr(out.stream()).0 as *mut f32,
                (rows * hidden) as i32,
                k as i32,
                hidden as i32,
                stream,
            );
        }
        crate::tensor::cuda_ext::tensor_from_f32_slice(out, (rows, hidden), &dev)
    }

    /// The host's computation of the experts a request's rows found missing: `pairs` names
    /// `(row, expert, weight)` for each, `x` holds `rows` rows of `hidden`. One grouped call
    /// computes every pair; the answer is one row per row of the request.
    fn host_product(
        &self,
        x: &[f32],
        rows: usize,
        pairs: &[(usize, u32, f32)],
    ) -> Result<Vec<f32>> {
        let host = Device::Cpu;
        let hidden = x.len() / rows.max(1);
        let m = pairs.len();
        let t = std::time::Instant::now();
        let mut rows_in = Vec::with_capacity(m * hidden);
        for &(r, _, _) in pairs {
            rows_in.extend_from_slice(&x[r * hidden..(r + 1) * hidden]);
        }
        let xs = Tensor::from_vec(rows_in, (m, hidden), &host)?;
        let ids: Vec<u32> = pairs.iter().map(|p| p.1).collect();
        let routing = ExpertRouting::group(&Tensor::from_vec(ids, (m, 1), &host)?, 1)?;
        let w = Tensor::from_vec(
            pairs.iter().map(|p| p.2).collect::<Vec<f32>>(),
            (m, 1),
            &host,
        )?;
        WORKER
            .prep_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t = std::time::Instant::now();
        let h = routing.host_gate_up_silu(&xs, &self.host[0], &self.host[1])?;
        WORKER
            .gate_up_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t = std::time::Instant::now();
        let out = routing
            .host_down_reduce(&h, &self.host[2], &w, m, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        WORKER
            .down_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let mut product = vec![0f32; rows * hidden];
        for (j, &(r, _, _)) in pairs.iter().enumerate() {
            for (acc, v) in product[r * hidden..(r + 1) * hidden]
                .iter_mut()
                .zip(&out[j * hidden..(j + 1) * hidden])
            {
                *acc += v;
            }
        }
        Ok(product)
    }

    /// Answer the layer's request if the card posted one since the last answer. Whether it did.
    /// `beat` tells the worker's health which step it is at.
    pub(super) fn serve_graph_request(self: &Arc<Self>, beat: &dyn Fn(usize)) -> Result<bool> {
        let Some(g) = self.graph.get() else {
            return Ok(false);
        };
        if g.mail.request() == g.done.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let started = std::time::Instant::now();
        let mut laps = Laps::new();
        beat(READING);
        #[cfg(test)]
        while STALL_FOR_TEST.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let req = g.mail.read();
        laps.mark("tier worker: read");
        let pairs = req.ids.len();
        let k = pairs / req.rows.max(1);
        // The card read these slots for this request; a missing expert's pair read the first
        // of them at weight zero, so none of them is a victim.
        let mut in_use = vec![false; self.slots()];
        let mut missing: Vec<u32> = Vec::new();
        let mut miss_pairs: Vec<(usize, u32, f32)> = Vec::new();
        beat(COUNTING);
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for i in 0..pairs {
                let e = req.ids[i] as usize;
                st.heat[e] += 1.0;
                st.counts[e] += 1;
                if req.slots[i] == u32::MAX {
                    if !missing.contains(&req.ids[i]) {
                        missing.push(req.ids[i]);
                    }
                    miss_pairs.push((i / k, req.ids[i], req.weights[i]));
                } else {
                    in_use[req.slots[i] as usize] = true;
                }
            }
            if let Some(first) = req.slots.iter().find(|&&s| s != u32::MAX) {
                in_use[*first as usize] = true;
            } else if !in_use.is_empty() {
                in_use[0] = true;
            }
            st.tokens_routed(req.rows);
        }
        self.routed.fetch_add(pairs as u64, Ordering::Relaxed);
        self.hits
            .fetch_add((pairs - miss_pairs.len()) as u64, Ordering::Relaxed);
        laps.mark("tier worker: bookkeeping");
        // The card waits for the product only when an expert was missing. Of the missing
        // experts, a share goes to the transit room by copy while the host computes the rest:
        // the card runs the copied ones once the answer says they landed.
        if miss_pairs.is_empty() {
            g.mail.hand_late(&[]);
            g.mail.raise(req.seq);
        } else {
            let to_transit = self.transit_share_of(missing.len(), req.rows);
            let (transit, host_experts) = missing.split_at(to_transit);
            if !transit.is_empty() {
                beat(TO_TRANSIT);
                self.copy_to_transit_between(&g.upload_start, &g.upload_end, transit)?;
            }
            let host_pairs: Vec<(usize, u32, f32)> = miss_pairs
                .iter()
                .copied()
                .filter(|p| host_experts.contains(&p.1))
                .collect();
            let late: Vec<(u32, u32, f32)> = miss_pairs
                .iter()
                .filter_map(|&(row, e, w)| {
                    let place = transit.iter().position(|&t| t == e)?;
                    let pair = req
                        .ids
                        .iter()
                        .enumerate()
                        .position(|(i, &id)| i / k == row && id == e)?;
                    Some((pair as u32, place as u32, w))
                })
                .collect();
            let t = std::time::Instant::now();
            beat(ON_HOST);
            let product = if host_pairs.is_empty() {
                vec![0f32; req.rows * req.x.len() / req.rows.max(1)]
            } else {
                self.host_product(&req.x, req.rows, &host_pairs)?
            };
            if !host_experts.is_empty() {
                blend(
                    &self.host_us[super::pace_class(req.rows)],
                    t.elapsed().as_secs_f64() * 1e6 / host_experts.len() as f64,
                );
            }
            laps.mark("tier worker: host product");
            WORKER
                .product_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            WORKER
                .missing
                .fetch_add(missing.len() as u64, Ordering::Relaxed);
            if !transit.is_empty() {
                beat(LANDING);
                g.upload_end
                    .synchronize()
                    .map_err(|e| crate::tensor::Error::msg(format!("transit copy: {e:?}")))?;
                if let Ok(ms) = g.upload_start.elapsed_ms(&g.upload_end) {
                    self.note_upload_us(ms as f64 * 1e3 / transit.len() as f64, req.rows);
                }
                WORKER
                    .transit
                    .fetch_add(transit.len() as u64, Ordering::Relaxed);
            }
            laps.mark("tier worker: transit landed");
            beat(ANSWERING);
            g.mail.hand_late(&late);
            g.mail.answer(&product, req.seq);
        }
        g.done.store(req.seq, Ordering::Relaxed);
        laps.mark("tier worker: answer");
        beat(PROMOTING);
        self.promote_up_to(&missing, &in_use, super::ADAPT_SWAPS_A_REQUEST);
        laps.mark("tier worker: promote");
        // The leader says what a request costs the host, once per window of requests.
        let us = started.elapsed().as_nanos() as u64;
        let n = WORKER.requests.fetch_add(1, Ordering::Relaxed) + 1;
        let total = WORKER.ns.fetch_add(us, Ordering::Relaxed) + us;
        let (rows_of, missed_of) = if req.rows > 1 {
            (&WORKER.rows_many, &WORKER.missed_many)
        } else {
            (&WORKER.rows_one, &WORKER.missed_one)
        };
        rows_of.fetch_add(1, Ordering::Relaxed);
        if !miss_pairs.is_empty() {
            WORKER.missed.fetch_add(1, Ordering::Relaxed);
            missed_of.fetch_add(1, Ordering::Relaxed);
        }
        if n.is_multiple_of(4096) {
            let missed = WORKER.missed.load(Ordering::Relaxed).max(1);
            let pct = |m: &std::sync::atomic::AtomicU64, r: &std::sync::atomic::AtomicU64| {
                100.0 * m.load(Ordering::Relaxed) as f64 / r.load(Ordering::Relaxed).max(1) as f64
            };
            tracing::info!(
                "expert tier worker: {:.0}% of one-row requests and {:.0}% of windows missed",
                pct(&WORKER.missed_one, &WORKER.rows_one),
                pct(&WORKER.missed_many, &WORKER.rows_many)
            );
            tracing::info!(
                "expert tier worker: {} us a request, {:.0}% had a missing expert: {:.1} missing ({:.2} to transit), product {} us (ready {}, gate and up {}, down {}), {:.2} promotions a request; transit share {}/8 (one row) {}/8 (rows), host {:.0}/{:.0} us an expert, copy {:.0}/{:.0} us",
                total / n / 1000,
                100.0 * missed as f64 / n as f64,
                WORKER.missing.load(Ordering::Relaxed) as f64 / missed as f64,
                WORKER.transit.load(Ordering::Relaxed) as f64 / missed as f64,
                WORKER.product_ns.load(Ordering::Relaxed) / missed / 1000,
                WORKER.prep_ns.load(Ordering::Relaxed) / missed / 1000,
                WORKER.gate_up_ns.load(Ordering::Relaxed) / missed / 1000,
                WORKER.down_ns.load(Ordering::Relaxed) / missed / 1000,
                WORKER.promotions.load(Ordering::Relaxed) as f64 / n as f64,
                self.transit_eighths[0].load(Ordering::Relaxed),
                self.transit_eighths[1].load(Ordering::Relaxed),
                f64::from_bits(self.host_us[0].load(Ordering::Relaxed)),
                f64::from_bits(self.host_us[1].load(Ordering::Relaxed)),
                f64::from_bits(self.upload_us[0].load(Ordering::Relaxed)),
                f64::from_bits(self.upload_us[1].load(Ordering::Relaxed)),
            );
        }
        Ok(true)
    }
}

/// Holds the tier worker of a test at the start of its next request until cleared.
#[cfg(test)]
pub(super) static STALL_FOR_TEST: AtomicBool = AtomicBool::new(false);

/// The thread that answers the requests of the graph layers of one card, and the watchdog that
/// releases the card when a request goes unanswered while the worker does not move.
pub struct TierWorker {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    watchdog: Option<std::thread::JoinHandle<()>>,
    tiers: Vec<Arc<LayerTier>>,
    health: Arc<Health>,
    dev: Device,
}

impl TierWorker {
    /// Serve `tiers`, each named for the logs.
    pub(super) fn start(dev: &Device, tiers: Vec<(String, Arc<LayerTier>)>) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let ctx = dev.as_cuda_device()?.0.context().clone();
        let mails = tiers
            .iter()
            .enumerate()
            .filter_map(|(i, (_, t))| Some((i, t.graph.get()?)))
            .map(|(i, g)| (i, g.mail.clone(), g.done.clone()))
            .collect();
        let health = Arc::new(Health::new(mails));
        for (_, t) in &tiers {
            if let Some(g) = t.graph.get() {
                let _ = g.health.set(health.clone());
            }
        }
        let mine: Vec<Arc<LayerTier>> = tiers.iter().map(|(_, t)| t.clone()).collect();
        let served = mine.clone();
        let (flag, beats) = (stop.clone(), health.clone());
        let thread = std::thread::Builder::new()
            .name("tier-worker".into())
            .spawn(move || {
                let _ = ctx.bind_to_thread();
                serve_loop(&served, &flag, &beats);
            })
            .map_err(|e| crate::tensor::Error::msg(format!("tier worker: {e}")))?;
        let (flag, watched) = (stop.clone(), health.clone());
        let watchdog = std::thread::Builder::new()
            .name("tier-watchdog".into())
            .spawn(move || watch(&tiers, &flag, &watched))
            .map_err(|e| crate::tensor::Error::msg(format!("tier watchdog: {e}")))?;
        let _ = health.watchdog.set(watchdog.thread().clone());
        Ok(Self {
            stop,
            thread: Some(thread),
            watchdog: Some(watchdog),
            tiers: mine,
            health,
            dev: dev.clone(),
        })
    }

    /// Before a token runs through this card's layers: an error while the worker stands
    /// stalled or failed, see [`Health::check`].
    pub fn check(&self) -> Result<()> {
        self.health.check(&self.dev)
    }

    /// Why this card's mailboxes stand released, if they do.
    pub fn failure(&self) -> Option<String> {
        self.health.failure()
    }

    /// The stall bound as it stands.
    #[cfg(test)]
    pub(super) fn stall_bound(&self, tiers: &[(String, Arc<LayerTier>)]) -> Option<u64> {
        self.health.bound(tiers)
    }
}

/// Answer requests until told to stop: spinning while requests keep coming, which the card
/// waits on, and polling at a gentler pace once they stop. A failure releases every waiting
/// card and stands for the decode to read.
fn serve_loop(tiers: &[Arc<LayerTier>], stop: &AtomicBool, health: &Health) {
    let mut last = std::time::Instant::now();
    // Seconds between requests while they come, smoothed: the spin lasts a few of them.
    let mut gap = 0.0f64;
    while !stop.load(Ordering::Relaxed) {
        let mut served = false;
        for (i, t) in tiers.iter().enumerate() {
            health.beat(i, IDLE);
            let started = std::time::Instant::now();
            match t.serve_graph_request(&|step| health.beat(i, step)) {
                Ok(s) => {
                    if s {
                        health.answered(started.elapsed().as_nanos() as u64);
                    }
                    served |= s;
                }
                Err(e) => {
                    tracing::error!("tier worker: {e}");
                    health.exited.store(true, Ordering::Release);
                    health.fail(format!("the tier worker failed: {e}"));
                    return;
                }
            }
        }
        let now = std::time::Instant::now();
        if served {
            let dt = now.duration_since(last).as_secs_f64();
            gap = if gap == 0.0 {
                dt
            } else {
                gap + (dt - gap) * 0.125
            };
            last = now;
        } else if now.duration_since(last).as_secs_f64() > 8.0 * gap {
            std::thread::sleep(std::time::Duration::from_micros(50));
        } else {
            std::hint::spin_loop();
        }
    }
}

/// Watch the worker of `tiers`: a request left unanswered while the worker has not moved for
/// longer than [`Health::bound`] releases every card waiting on the worker, and the decode
/// fails on it. The watchdog sleeps until the worker has a measured request.
fn watch(tiers: &[(String, Arc<LayerTier>)], stop: &AtomicBool, health: &Health) {
    while !stop.load(Ordering::Acquire) {
        let Some(bound) = health.bound(tiers) else {
            std::thread::park();
            continue;
        };
        std::thread::park_timeout(std::time::Duration::from_nanos(
            bound / u64::from(LOOKS_A_BOUND),
        ));
        if stop.load(Ordering::Acquire) || health.failure().is_some() {
            continue;
        }
        let still = health
            .now()
            .saturating_sub(health.beat.load(Ordering::Acquire));
        if still < bound {
            continue;
        }
        let Some((at, mail, seq)) = health.pending() else {
            continue;
        };
        let name = |i: usize| tiers.get(i).map_or("a layer", |(n, _)| n.as_str());
        let worker_at = health.tier.load(Ordering::Relaxed);
        let step = STEPS[health.step.load(Ordering::Relaxed).min(STEPS.len() - 1)];
        let [request, reply, abort, rows] = mail.words();
        let why = format!(
            "the tier worker stalled: {} has request {seq} unanswered while the worker has been {step} for {} for {:.3} s, past the bound of {:.3} s (request {request} reply {reply} abort {abort} rows {rows})",
            name(at),
            name(worker_at),
            still as f64 * 1e-9,
            bound as f64 * 1e-9,
        );
        tracing::error!("expert tier: {why}; the card's mailboxes are released");
        health.fail(why);
    }
}

impl Drop for TierWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for t in &self.tiers {
            if let Some(g) = t.graph.get() {
                g.mail.abort();
            }
        }
        if let Some(h) = self.watchdog.take() {
            h.thread().unpark();
            let _ = h.join();
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}
