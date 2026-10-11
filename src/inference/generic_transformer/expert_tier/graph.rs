//! A cached layer's decoded token, or the rows of a draft verify, as kernels alone, so that a
//! graph can record them.
//!
//! The layer's missing experts are computed by a thread of the host. The card hands it a
//! mailbox in page-locked memory (the router's choice of each row, the slots the cache served,
//! the rows' input), raises a request word, waits on a reply word with a kernel, and adds the
//! product the host left in the mailbox. Neither side returns to the code that launched it
//! between two kernels, which is what a recorded graph needs and what an eager decode pays a
//! round trip for.

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
}

unsafe impl Send for Mailbox {}
unsafe impl Sync for Mailbox {}

/// Offsets of the request, reply, abort and rows words, each on a line of its own: the card
/// writes the request and the rows, the host the other two.
const REQUEST: usize = 0;
const REPLY: usize = 64;
const ABORT: usize = 128;
const ROWS: usize = 192;
const BODY: usize = 256;

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
        let (host, device) = mapped(dev, product + line(4 * hidden * rows_max))?;
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
    /// Of the products' nanoseconds: readying the rows and the routing, the gate and up
    /// projections, the down projection and the sum per row.
    prep_ns: std::sync::atomic::AtomicU64,
    gate_up_ns: std::sync::atomic::AtomicU64,
    down_ns: std::sync::atomic::AtomicU64,
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
};

/// A layer's decoded rows between the post of their request and the wait for the answer.
pub(crate) struct GraphPending {
    card: Tensor,
    rows: usize,
    hidden: usize,
    dev: Device,
}

/// A layer's side of the graph decode: the mailbox, the buffers the kernels write on the card,
/// the rows they have room for, and the number of the last request the host answered.
pub(super) struct GraphSide {
    mail: Mailbox,
    counter: Tensor,
    slots: Tensor,
    weights: Tensor,
    rows_max: usize,
    done: AtomicU32,
}

impl LayerTier {
    /// The mailbox as it stands, for a wait that lasts: the request, reply, abort and rows
    /// words, what the worker served, and the card's counter read over the copy stream.
    pub fn mailbox_state(&self) -> Option<String> {
        let g = self.graph.get()?;
        let [request, reply, abort, rows] = g.mail.words();
        let done = g.done.load(Ordering::Relaxed);
        let counter = (|| -> Result<[u32; 2]> {
            use cudarc::driver::DevicePtr;
            let (storage, layout) = g.counter.storage_and_layout();
            let crate::tensor::StorageView::Cuda(c) = &*storage else {
                crate::tensor::bail!("the counter is on the card");
            };
            let slice = c.as_cuda_slice::<u32>()?;
            let src = slice.device_ptr(slice.stream()).0 + (layout.start_offset() * 4) as u64;
            let mut out = [0u32; 2];
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(&mut out, src, self.copy.cu_stream())
                    .map_err(|e| crate::tensor::Error::msg(format!("counter read: {e:?}")))?;
            }
            self.copy
                .synchronize()
                .map_err(|e| crate::tensor::Error::msg(format!("counter read: {e:?}")))?;
            Ok(out)
        })();
        Some(format!(
            "request {request} reply {reply} abort {abort} rows {rows} served {done} counter {:?}",
            counter.ok()
        ))
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
            mail: Mailbox::new(dev, k, hidden, rows_max)?,
            counter: Tensor::from_vec(vec![0u32; 2], (2,), &host)?.to_device(dev)?,
            slots: Tensor::from_vec(vec![0u32; pairs], (pairs,), &host)?.to_device(dev)?,
            weights: Tensor::from_vec(vec![0f32; pairs], (1, pairs), &host)?.to_device(dev)?,
            rows_max,
            done: AtomicU32::new(0),
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
        let extra = self.extra.lock().unwrap_or_else(|e| e.into_inner());
        let more = extra.as_ref();
        let h = routing.gate_up_silu_over(
            &x,
            &self.card[0],
            &self.card[1],
            more.map(|m| (&m[0], &m[1])),
        )?;
        let card =
            routing.down_reduce_over(&h, &self.card[2], more.map(|m| &m[2]), &weights, rows)?;
        drop(extra);
        Ok(GraphPending {
            card,
            rows,
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
            rows,
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
            use cudarc::driver::DevicePtr;
            moe_cuda::tier_graph::merge(
                address_of::<f32>(&card)? as *const f32,
                g.mail.dptr(g.mail.product) as *const f32,
                address_of::<u32>(&g.counter)? as *const u32,
                out.device_ptr(out.stream()).0 as *mut f32,
                (rows * hidden) as i32,
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
        let pairs = req.ids.len();
        let k = pairs / req.rows.max(1);
        // The card read these slots for this request; a missing expert's pair read the first
        // of them at weight zero, so none of them is a victim.
        let mut in_use = vec![false; self.slots()];
        let mut missing: Vec<u32> = Vec::new();
        let mut miss_pairs: Vec<(usize, u32, f32)> = Vec::new();
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for i in 0..pairs {
                let e = req.ids[i] as usize;
                st.heat[e] += 1.0;
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
            st.since_decay += req.rows;
            if st.since_decay >= self.n_experts {
                st.since_decay = 0;
                st.heat.iter_mut().for_each(|h| *h *= 0.5);
            }
        }
        self.routed.fetch_add(pairs as u64, Ordering::Relaxed);
        self.hits
            .fetch_add((pairs - miss_pairs.len()) as u64, Ordering::Relaxed);
        laps.mark("tier worker: bookkeeping");
        // The card waits for the product only when an expert was missing.
        if miss_pairs.is_empty() {
            g.mail.raise(req.seq);
        } else {
            let t = std::time::Instant::now();
            let product = self.host_product(&req.x, req.rows, &miss_pairs)?;
            blend(
                &self.host_us,
                t.elapsed().as_secs_f64() * 1e6 / missing.len() as f64,
            );
            laps.mark("tier worker: host product");
            WORKER
                .product_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            WORKER
                .missing
                .fetch_add(missing.len() as u64, Ordering::Relaxed);
            g.mail.answer(&product, req.seq);
        }
        g.done.store(req.seq, Ordering::Relaxed);
        laps.mark("tier worker: answer");
        self.promote_up_to(&missing, &in_use, req.rows);
        laps.mark("tier worker: promote");
        // The leader says what a request costs the host, once per window of requests.
        let us = started.elapsed().as_nanos() as u64;
        let n = WORKER.requests.fetch_add(1, Ordering::Relaxed) + 1;
        let total = WORKER.ns.fetch_add(us, Ordering::Relaxed) + us;
        if !miss_pairs.is_empty() {
            WORKER.missed.fetch_add(1, Ordering::Relaxed);
        }
        if n % 4096 == 0 {
            let missed = WORKER.missed.load(Ordering::Relaxed).max(1);
            tracing::info!(
                "expert tier worker: {} us a request, {:.0}% had a missing expert: {:.1} missing, product {} us (ready {}, gate and up {}, down {})",
                total / n / 1000,
                100.0 * missed as f64 / n as f64,
                WORKER.missing.load(Ordering::Relaxed) as f64 / missed as f64,
                WORKER.product_ns.load(Ordering::Relaxed) / missed / 1000,
                WORKER.prep_ns.load(Ordering::Relaxed) / missed / 1000,
                WORKER.gate_up_ns.load(Ordering::Relaxed) / missed / 1000,
                WORKER.down_ns.load(Ordering::Relaxed) / missed / 1000,
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
