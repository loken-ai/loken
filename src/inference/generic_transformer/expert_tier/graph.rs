//! A cached layer's decoded token as kernels alone, so that a graph can record it.
//!
//! The layer's missing experts are computed by a thread of the host. The card hands it a
//! mailbox in page-locked memory (the router's choice, the slots the cache served, the layer's
//! input), raises a request word, waits on a reply word with a kernel, and adds the product the
//! host left in the mailbox. Neither side returns to the code that launched it between two
//! kernels, which is what a recorded graph needs and what an eager decode pays a round trip for.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use cudarc::driver::sys::{self, CUresult};

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
fn mapped(dev: &Device, bytes: usize) -> Result<(*mut u8, u64)> {
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

/// What the card posts for a layer's request and the host answers.
pub(super) struct Mailbox {
    host: *mut u8,
    device: u64,
    k: usize,
    hidden: usize,
    ids: usize,
    weights: usize,
    slots: usize,
    x: usize,
    product: usize,
}

unsafe impl Send for Mailbox {}
unsafe impl Sync for Mailbox {}

/// Offsets of the request, reply and abort words, each on a line of its own: the card writes the
/// first and the host the other two.
const REQUEST: usize = 0;
const REPLY: usize = 64;
const ABORT: usize = 128;
const BODY: usize = 192;

fn line(bytes: usize) -> usize {
    bytes.div_ceil(64) * 64
}

/// The request as the host reads it, with the choice it carries.
pub(super) struct Request {
    pub seq: u32,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    /// The slot the cache served each expert from, `u32::MAX` for a missing one.
    pub slots: Vec<u32>,
    pub x: Vec<f32>,
}

impl Mailbox {
    fn new(dev: &Device, k: usize, hidden: usize) -> Result<Self> {
        let ids = BODY;
        let weights = ids + line(4 * k);
        let slots = weights + line(4 * k);
        let x = slots + line(4 * k);
        let product = x + line(4 * hidden);
        let (host, device) = mapped(dev, product + line(4 * hidden))?;
        Ok(Self {
            host,
            device,
            k,
            hidden,
            ids,
            weights,
            slots,
            x,
            product,
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

    /// The posted request, once its number is read: the card writes the body before the number.
    pub(super) fn read(&self) -> Request {
        loop {
            let seq = self.request();
            std::sync::atomic::fence(Ordering::Acquire);
            let at = |off: usize| unsafe { self.host.add(off) };
            let ids = unsafe { std::slice::from_raw_parts(at(self.ids) as *const u32, self.k) };
            let weights =
                unsafe { std::slice::from_raw_parts(at(self.weights) as *const f32, self.k) };
            let slots = unsafe { std::slice::from_raw_parts(at(self.slots) as *const u32, self.k) };
            let x = unsafe { std::slice::from_raw_parts(at(self.x) as *const f32, self.hidden) };
            let read = Request {
                seq,
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

    /// Leave `product` for the card and raise the reply word to `seq`.
    pub(super) fn answer(&self, product: &[f32], seq: u32) {
        debug_assert_eq!(product.len(), self.hidden);
        unsafe {
            std::ptr::copy_nonoverlapping(
                product.as_ptr(),
                self.host.add(self.product) as *mut f32,
                self.hidden,
            );
        }
        std::sync::atomic::fence(Ordering::Release);
        unsafe { std::ptr::write_volatile(self.word(REPLY), seq) };
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
}

static WORKER: WorkerStats = WorkerStats {
    requests: std::sync::atomic::AtomicU64::new(0),
    ns: std::sync::atomic::AtomicU64::new(0),
    missed: std::sync::atomic::AtomicU64::new(0),
    missing: std::sync::atomic::AtomicU64::new(0),
    product_ns: std::sync::atomic::AtomicU64::new(0),
};

/// A layer's decoded token between the post of its request and the wait for the answer.
pub(crate) struct GraphPending {
    card: Tensor,
    hidden: usize,
    dev: Device,
}

/// A layer's side of the graph decode: the mailbox, the buffers the kernels write on the card,
/// and the number of the last request the host answered.
pub(super) struct GraphSide {
    mail: Mailbox,
    counter: Tensor,
    slots: Tensor,
    weights: Tensor,
    done: AtomicU32,
}

impl LayerTier {
    /// Let this layer's decoded token run as kernels alone, served by a [`TierWorker`].
    pub(super) fn enable_graph(&self, hidden: usize, k: usize) -> Result<()> {
        // The plan kernel looks a choice up with one thread per expert of a warp.
        if self.graph.get().is_some() || self.resident || k > 32 {
            return Ok(());
        }
        let host = Device::Cpu;
        let dev = &self.dev;
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let table = Arc::new(SlotTable::new(dev, &st.slot_of)?);
        st.table = Some(table.clone());
        drop(st);
        let side = GraphSide {
            mail: Mailbox::new(dev, k, hidden)?,
            counter: Tensor::from_vec(vec![0u32; 2], (2,), &host)?.to_device(dev)?,
            slots: Tensor::from_vec(vec![0u32; k], (k,), &host)?.to_device(dev)?,
            weights: Tensor::from_vec(vec![0f32; k], (1, k), &host)?.to_device(dev)?,
            done: AtomicU32::new(0),
        };
        let _ = self.graph.set(side);
        Ok(())
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
    pub(super) fn decode_graph(
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
    /// before [`Self::graph_end`] waits for it.
    pub(crate) fn graph_begin(
        &self,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<GraphPending> {
        use crate::inference::moe_cuda;
        let g = self.graph.get().expect("a graph layer is enabled");
        let k = moe.n_experts_used;
        let hidden = xs.elem_count();
        let dev = xs.device().clone();
        let cd = dev.as_cuda_device()?;
        let stream = cd.0.stream().cu_stream() as i64;
        let x = xs.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
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
                g.mail.dptr(REQUEST) as *mut u32,
                k as i32,
                hidden as i32,
                stream,
            );
        }
        let routing = ExpertRouting {
            slots: moe.decode_slots.clone().expect("a tier layer is on a card"),
            experts: g.slots.clone(),
            topk: k,
        };
        let extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
        let more = extra.as_ref();
        let h = routing.gate_up_silu_over(
            &x,
            &self.card[0],
            &self.card[1],
            more.map(|m| (&m[0], &m[1])),
        )?;
        let card = routing.down_reduce_over(&h, &self.card[2], more.map(|m| &m[2]), &g.weights)?;
        drop(extra);
        Ok(GraphPending { card, hidden, dev })
    }

    /// The second half: wait for the host's answer and add its product to the card's.
    pub(crate) fn graph_end(&self, pending: GraphPending) -> Result<Tensor> {
        use crate::inference::moe_cuda;
        let g = self.graph.get().expect("a graph layer is enabled");
        let GraphPending { card, hidden, dev } = pending;
        let cd = dev.as_cuda_device()?;
        let stream = cd.0.stream().cu_stream() as i64;
        let out = unsafe { cd.alloc::<f32>(hidden) }?;
        unsafe {
            moe_cuda::tier_graph::wait(
                address_of::<u32>(&g.counter)? as *const u32,
                g.mail.dptr(REPLY) as *mut u32,
                g.mail.dptr(ABORT) as *mut u32,
                stream,
            );
            use cudarc::driver::DevicePtr;
            moe_cuda::tier_graph::merge(
                address_of::<f32>(&card)? as *const f32,
                g.mail.dptr(g.mail.product) as *const f32,
                address_of::<u32>(&g.counter)? as *const u32,
                out.device_ptr(out.stream()).0 as *mut f32,
                hidden as i32,
                stream,
            );
        }
        crate::tensor::cuda_ext::tensor_from_f32_slice(out, (1, hidden), &dev)
    }

    /// The host's computation of the experts a request found missing: `x` through `ids`, scaled
    /// by `weights`, as one row.
    fn host_product(&self, x: &[f32], ids: &[u32], weights: &[f32]) -> Result<Vec<f32>> {
        let host = Device::Cpu;
        let m = ids.len();
        let x = Tensor::from_vec(x.to_vec(), (1, x.len()), &host)?;
        let routing = ExpertRouting::group(&Tensor::from_vec(ids.to_vec(), (1, m), &host)?, m)?;
        let h = routing.host_gate_up_silu(&x, &self.host[0], &self.host[1])?;
        let w = Tensor::from_vec(weights.to_vec(), (1, m), &host)?;
        let product = routing.host_down_reduce(&h, &self.host[2], &w, 1, None)?;
        product.flatten_all()?.to_vec1::<f32>()
    }

    /// Answer the layer's request if the card posted one since the last answer. Whether it did.
    pub(super) fn serve_graph_request(self: &Arc<Self>) -> Result<bool> {
        let Some(g) = self.graph.get() else {
            return Ok(false);
        };
        if g.mail.request() == g.done.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let started = std::time::Instant::now();
        let mut laps = Laps::new();
        let req = g.mail.read();
        laps.mark("tier worker: read");
        let k = req.ids.len();
        // The card read these slots for this request; a missing expert's pair read the first
        // of them at weight zero, so none of them is a victim.
        let mut in_use = vec![false; self.slots()];
        let mut missing = Vec::new();
        let (mut miss_ids, mut miss_w) = (Vec::new(), Vec::new());
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for i in 0..k {
                let e = req.ids[i] as usize;
                st.heat[e] += 1.0;
                if req.slots[i] == u32::MAX {
                    missing.push(req.ids[i]);
                    miss_ids.push(req.ids[i]);
                    miss_w.push(req.weights[i]);
                } else {
                    in_use[req.slots[i] as usize] = true;
                }
            }
            if let Some(first) = req.slots.iter().find(|&&s| s != u32::MAX) {
                in_use[*first as usize] = true;
            } else if !in_use.is_empty() {
                in_use[0] = true;
            }
            st.since_decay += 1;
            if st.since_decay >= self.n_experts {
                st.since_decay = 0;
                st.heat.iter_mut().for_each(|h| *h *= 0.5);
            }
        }
        self.routed.fetch_add(k as u64, Ordering::Relaxed);
        self.hits
            .fetch_add((k - miss_ids.len()) as u64, Ordering::Relaxed);
        laps.mark("tier worker: bookkeeping");
        // The card waits for the product only when an expert was missing.
        if miss_ids.is_empty() {
            g.mail.raise(req.seq);
        } else {
            let t = std::time::Instant::now();
            let product = self.host_product(&req.x, &miss_ids, &miss_w)?;
            blend(
                &self.host_us,
                t.elapsed().as_secs_f64() * 1e6 / miss_ids.len() as f64,
            );
            laps.mark("tier worker: host product");
            WORKER
                .product_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            WORKER
                .missing
                .fetch_add(miss_ids.len() as u64, Ordering::Relaxed);
            g.mail.answer(&product, req.seq);
        }
        g.done.store(req.seq, Ordering::Relaxed);
        laps.mark("tier worker: answer");
        self.promote(&missing, &in_use);
        laps.mark("tier worker: promote");
        // The leader says what a request costs the host, once per window of requests.
        let us = started.elapsed().as_nanos() as u64;
        let n = WORKER.requests.fetch_add(1, Ordering::Relaxed) + 1;
        let total = WORKER.ns.fetch_add(us, Ordering::Relaxed) + us;
        if !miss_ids.is_empty() {
            WORKER.missed.fetch_add(1, Ordering::Relaxed);
        }
        if self.leader.load(Ordering::Relaxed) && n % 4096 == 0 {
            let missed = WORKER.missed.load(Ordering::Relaxed).max(1);
            tracing::info!(
                "expert tier worker: {} us a request, {:.0}% had a missing expert: {:.1} missing, product {} us",
                total / n / 1000,
                100.0 * missed as f64 / n as f64,
                WORKER.missing.load(Ordering::Relaxed) as f64 / missed as f64,
                WORKER.product_ns.load(Ordering::Relaxed) / missed / 1000,
            );
        }
        Ok(true)
    }
}

/// The thread that answers the requests of the graph layers of one card.
pub struct TierWorker {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    tiers: Vec<Arc<LayerTier>>,
}

impl TierWorker {
    pub(super) fn start(dev: &Device, tiers: Vec<Arc<LayerTier>>) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let ctx = dev.as_cuda_device()?.0.context().clone();
        let mine = tiers.clone();
        let flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("tier-worker".into())
            .spawn(move || {
                let _ = ctx.bind_to_thread();
                serve_loop(&mine, &flag);
            })
            .map_err(|e| crate::tensor::Error::msg(format!("tier worker: {e}")))?;
        Ok(Self {
            stop,
            thread: Some(thread),
            tiers,
        })
    }
}

/// Answer requests until told to stop: spinning while requests keep coming, which the card
/// waits on, and polling at a gentler pace once they stop. A failure releases every waiting card.
fn serve_loop(tiers: &[Arc<LayerTier>], stop: &AtomicBool) {
    let mut last = std::time::Instant::now();
    // Seconds between requests while they come, smoothed: the spin lasts a few of them.
    let mut gap = 0.0f64;
    while !stop.load(Ordering::Relaxed) {
        let mut served = false;
        for t in tiers {
            match t.serve_graph_request() {
                Ok(s) => served |= s,
                Err(e) => {
                    tracing::error!("tier worker: {e}");
                    for t in tiers {
                        if let Some(g) = t.graph.get() {
                            g.mail.abort();
                        }
                    }
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

impl Drop for TierWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in &self.tiers {
            if let Some(g) = t.graph.get() {
                g.mail.abort();
            }
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}
