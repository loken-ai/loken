//! A hot tier over the routed experts, so a low-memory model keeps only the working set resident.
//!
//! A 552B MoE routes a handful of its hundreds of experts per token, so holding every expert in
//! memory is the cost that does not fit. The store keeps them behind an `ExpertLoader` (the seam a
//! disk-backed, quantised source plugs into): an expert is a set of views into the mapping,
//! built on demand and read in place. Routing has locality - a third of a token's experts were
//! the previous token's, two thirds were among the last sixteen tokens' - and the page cache is
//! where the hot ones live; what the store decides is which ones. A bounded number of the most
//! recently routed experts per layer are kept - the least recently used gives way, a prompt's
//! batch counting each expert at the last position that routed to it - and every other
//! expert is dropped from the page cache as soon as it has been computed, so streaming a cold
//! expert never evicts a kept one. On a traced generation a frequency count halved every
//! `capacity` requests missed twice as often as recency. Nothing is copied: no memory of its
//! own, nothing to swap.
//! `get` returns the same weights either way, so the tier changes time, never the output.

use super::experts::Expert;
use super::projection::Projection;
use crate::tensor::quant_view::expert_view;
use crate::tensor::quantized::QTensor;
use crate::tensor::{Result, Tensor};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A measurement of how many distinct experts each layer routes to over a run: the working set,
/// against which the card and host tiers are sized. Off until asked (LOKEN_WS at start, or the
/// enable call). Keyed by the layer's store address.
static WORKING_SET: Mutex<Option<HashMap<usize, HashMap<usize, u64>>>> = Mutex::new(None);

pub fn ws_enable() {
    *WORKING_SET.lock().unwrap_or_else(|e| e.into_inner()) = Some(HashMap::new());
}

pub fn ws_record(layer: usize, expert: usize) {
    if let Some(m) = WORKING_SET
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
    {
        *m.entry(layer).or_default().entry(expert).or_insert(0) += 1;
    }
}

/// The fraction of activations the `cap` most-frequent experts of each layer cover: what a card
/// holding the hottest `cap` would answer instead of the host. Averaged over the layers.
pub fn ws_coverage(cap: usize) -> f64 {
    let g = WORKING_SET.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = g.as_ref() else { return 0.0 };
    if m.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for counts in m.values() {
        let total: u64 = counts.values().sum();
        if total == 0 {
            continue;
        }
        let mut v: Vec<u64> = counts.values().copied().collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        let top: u64 = v.iter().take(cap).sum();
        sum += top as f64 / total as f64;
    }
    sum / m.len() as f64
}

/// The distinct-expert count of each layer seen so far, largest first, and the reset that starts a
/// fresh window.
pub fn ws_report() -> Vec<usize> {
    let g = WORKING_SET.lock().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<usize> = g
        .as_ref()
        .map(|m| m.values().map(HashMap::len).collect())
        .unwrap_or_default();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

pub fn ws_reset() {
    if let Some(m) = WORKING_SET
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
    {
        m.clear();
    }
}

/// Builds one routed expert's weights on demand. The resident model materialises every expert up
/// front; a low-memory model builds each from its backing source only when needed.
pub trait ExpertLoader: Send + Sync {
    fn build(&self, id: usize) -> Result<Expert>;
    fn count(&self) -> usize;
    /// Drop an expert's bytes from memory once it is not to be kept: by default through its
    /// projections' own mapping advice.
    fn release(&self, expert: &Expert) {
        expert.dont_need();
    }
}

/// Tier accounting, for observability and to let the gate assert eviction actually happened.
#[derive(Default, Clone, Copy, Debug)]
pub struct StoreStats {
    /// Requests for a kept expert.
    pub hits: u64,
    /// Requests for an expert not kept.
    pub misses: u64,
    pub evictions: u64,
}

struct CacheInner {
    kept: HashMap<usize, Arc<Expert>>,
    /// Requests per expert since the store was built: observability only.
    freq: HashMap<usize, u64>,
    /// The clock at which each expert was last routed to, and the clock itself: one tick per
    /// token seen, a batch advancing it by its length.
    last_use: HashMap<usize, u64>,
    clock: u64,
    stats: StoreStats,
}

/// The routed experts of one layer behind their loader, keeping `capacity` of them in the page
/// cache (0: none), the least recently routed to giving way.
pub struct ExpertStore {
    loader: Box<dyn ExpertLoader>,
    capacity: AtomicUsize,
    inner: Mutex<CacheInner>,
}

impl ExpertStore {
    pub fn new(loader: Box<dyn ExpertLoader>, capacity: usize) -> Self {
        Self {
            loader,
            capacity: AtomicUsize::new(capacity),
            inner: Mutex::new(CacheInner {
                kept: HashMap::new(),
                freq: HashMap::new(),
                last_use: HashMap::new(),
                clock: 0,
                stats: StoreStats::default(),
            }),
        }
    }

    pub fn count(&self) -> usize {
        self.loader.count()
    }

    pub fn stats(&self) -> StoreStats {
        self.inner.lock().unwrap().stats
    }

    /// Whether expert `id` is in the kept tier now.
    pub fn is_kept(&self, id: usize) -> bool {
        self.inner.lock().unwrap().kept.contains_key(&id)
    }

    /// The fraction of pages resident in memory over the kept experts and over the others:
    /// whether the tier's promise holds, read from the kernel.
    #[cfg(target_os = "linux")]
    pub fn residency(&self) -> Result<(f64, f64)> {
        let kept: Vec<(usize, Arc<Expert>)> = {
            let g = self.inner.lock().unwrap();
            g.kept.iter().map(|(&k, v)| (k, v.clone())).collect()
        };
        let mut sum = [(0usize, 0usize); 2];
        for id in 0..self.loader.count() {
            let (slot, expert) = match kept.iter().find(|(k, _)| *k == id) {
                Some((_, e)) => (0, e.clone()),
                None => (1, Arc::new(self.loader.build(id)?)),
            };
            let (r, n) = expert.resident_pages();
            sum[slot].0 += r;
            sum[slot].1 += n;
        }
        let frac = |(r, n): (usize, usize)| if n == 0 { 0.0 } else { r as f64 / n as f64 };
        Ok((frac(sum[0]), frac(sum[1])))
    }

    /// Per expert, whether it is kept and how many of its pages are resident: the raw facts
    /// behind `residency`.
    #[cfg(target_os = "linux")]
    pub fn residency_by_expert(&self) -> Result<Vec<(bool, usize, usize)>> {
        let kept: Vec<usize> = self.inner.lock().unwrap().kept.keys().copied().collect();
        (0..self.loader.count())
            .map(|id| {
                let (r, n) = self.loader.build(id)?.resident_pages();
                Ok((kept.contains(&id), r, n))
            })
            .collect()
    }

    /// The kept experts with the clock each was last routed to at: for probes.
    pub fn kept(&self) -> Vec<(usize, u64)> {
        let g = self.inner.lock().unwrap();
        let mut v: Vec<(usize, u64)> = g
            .kept
            .keys()
            .map(|&k| (k, g.last_use.get(&k).copied().unwrap_or(0)))
            .collect();
        v.sort();
        v
    }

    /// Build expert `id`'s views without touching the tier: for probes.
    pub fn view(&self, id: usize) -> Result<Expert> {
        self.loader.build(id)
    }

    /// How many times each expert was asked for since the store was built, by expert id.
    pub fn usage(&self) -> Vec<u64> {
        let g = self.inner.lock().unwrap();
        (0..self.loader.count())
            .map(|e| g.freq.get(&e).copied().unwrap_or(0))
            .collect()
    }

    /// The bytes one expert occupies as stored.
    pub fn expert_bytes(&self) -> Result<usize> {
        Ok(self.loader.build(0)?.bytes())
    }

    /// Resize the kept set. Entries past the new size are released least-frequently-used
    /// first, now rather than on the next request, so the memory comes back at once.
    pub fn set_capacity(&self, capacity: usize) {
        self.capacity.store(capacity, Ordering::Relaxed);
        let mut g = self.inner.lock().unwrap();
        while g.kept.len() > capacity {
            match Self::coldest(&g, None) {
                Some(v) => self.evict(&mut g, v),
                None => break,
            }
        }
    }

    fn coldest(g: &CacheInner, except: Option<usize>) -> Option<usize> {
        g.kept
            .keys()
            .filter(|&&k| Some(k) != except)
            .min_by_key(|&&k| g.last_use.get(&k).copied().unwrap_or(0))
            .copied()
    }

    fn evict(&self, g: &mut CacheInner, id: usize) {
        if let Some(e) = g.kept.remove(&id) {
            self.loader.release(&e);
            g.stats.evictions += 1;
        }
    }

    /// Note `uses` for each expert of a batch of `span` tokens: its request count, and the
    /// clock at its last position in the batch, so recency within a prompt is kept.
    fn note_uses(&self, uses: &[(usize, u64, u64)], span: u64) {
        let mut g = self.inner.lock().unwrap();
        let clock = g.clock;
        for &(id, n, last) in uses {
            *g.freq.entry(id).or_insert(0) += n;
            g.last_use.insert(id, clock + last);
        }
        g.clock += span;
    }

    /// The kept expert `id`, or a view built for this use alone: no request is counted and
    /// nothing is admitted. The compute loop reads through this after `prefetch` decided the
    /// tier; admitting here too, in the loop's own order, put the last ids read in the tier
    /// in place of the most recently routed to.
    pub fn fetch(&self, id: usize) -> Result<Arc<Expert>> {
        self.lookup(id, false)
    }

    fn lookup(&self, id: usize, admit: bool) -> Result<Arc<Expert>> {
        let capacity = self.capacity.load(Ordering::Relaxed);
        {
            let mut g = self.inner.lock().unwrap();
            if let Some(e) = g.kept.get(&id).cloned() {
                g.stats.hits += 1;
                return Ok(e);
            }
            g.stats.misses += 1;
        }
        // Build outside the lock; a concurrent build of the same id is harmless (same value).
        let expert = Arc::new(self.loader.build(id)?);
        if capacity == 0 || !admit {
            return Ok(expert);
        }
        // Kept: the least recently used kept expert is dropped from the page cache for it.
        let mut g = self.inner.lock().unwrap();
        while g.kept.len() >= capacity {
            match Self::coldest(&g, Some(id)) {
                Some(v) => self.evict(&mut g, v),
                None => break,
            }
        }
        g.kept.insert(id, expert.clone());
        Ok(expert)
    }

    /// One request for expert `id` by one token: noted, then fetched and kept.
    pub fn get(&self, id: usize) -> Result<Arc<Expert>> {
        self.note_uses(&[(id, 1, 0)], 1);
        self.lookup(id, true)
    }

    /// Called once `expert` (id `id`) has been computed: one that is not kept is dropped from
    /// the page cache now, so its bytes do not stay in the way of the kept ones.
    pub fn release(&self, id: usize, expert: &Expert) {
        let kept = self.inner.lock().unwrap().kept.contains_key(&id);
        if !kept && self.capacity.load(Ordering::Relaxed) > 0 {
            self.loader.release(expert);
        }
    }

    /// Reclaim the pages of every expert not kept, asked for or not. A batch reads the experts
    /// of a layer nearly in file order, and the kernel's read-ahead pulls in the neighbours of
    /// each one read; those pages were never asked for, so no release ever dropped them, and
    /// they grew to evict the kept experts of the layers read before.
    pub fn sweep(&self) -> Result<()> {
        if self.capacity.load(Ordering::Relaxed) == 0 {
            return Ok(());
        }
        let kept: Vec<usize> = self.inner.lock().unwrap().kept.keys().copied().collect();
        for id in 0..self.loader.count() {
            if !kept.contains(&id) {
                self.loader.release(&self.loader.build(id)?);
            }
        }
        Ok(())
    }

    /// Warm the tier with the experts a batch of `span` tokens routes to - each with the
    /// number of tokens it serves and the last position among them - before the compute loop
    /// reads them.
    pub fn prefetch(&self, uses: &[(usize, u64, u64)], span: u64) -> Result<()> {
        self.note_uses(uses, span);
        // Most recent last, so what a full tier evicts while admitting the batch is the batch's
        // own oldest, not its newest.
        let mut order: Vec<(usize, u64, u64)> = uses.to_vec();
        order.sort_by_key(|&(_, _, last)| last);
        for &(id, _, _) in &order {
            self.lookup(id, true)?;
        }
        Ok(())
    }
}

/// The routed experts of one MoE, either all resident or streamed through a hot cache. Both answer
/// `get` with the same weights, so the MoE forward is written once against this.
pub enum ExpertSet {
    Resident(Vec<Arc<Expert>>),
    Streamed(ExpertStore),
}

impl ExpertSet {
    pub fn count(&self) -> usize {
        match self {
            ExpertSet::Resident(v) => v.len(),
            ExpertSet::Streamed(s) => s.count(),
        }
    }

    pub fn get(&self, id: usize) -> Result<Arc<Expert>> {
        match self {
            ExpertSet::Resident(v) => Ok(v[id].clone()),
            ExpertSet::Streamed(s) => s.get(id),
        }
    }

    /// `get` without counting a request: for the compute loop, once `prefetch` has counted the
    /// batch.
    pub fn fetch(&self, id: usize) -> Result<Arc<Expert>> {
        match self {
            ExpertSet::Resident(v) => Ok(v[id].clone()),
            ExpertSet::Streamed(s) => s.fetch(id),
        }
    }

    /// Warm a streamed set with the experts a batch of `span` tokens will use, each with its
    /// token count and last position; a no-op when resident.
    pub fn prefetch(&self, uses: &[(usize, u64, u64)], span: u64) -> Result<()> {
        match self {
            ExpertSet::Resident(_) => Ok(()),
            ExpertSet::Streamed(s) => s.prefetch(uses, span),
        }
    }

    /// Reclaim the pages of every expert a streamed set does not keep; a no-op when resident.
    pub fn sweep(&self) -> Result<()> {
        match self {
            ExpertSet::Resident(_) => Ok(()),
            ExpertSet::Streamed(s) => s.sweep(),
        }
    }

    /// Hand an expert back once computed; a streamed set drops one it does not keep from the
    /// page cache.
    pub fn release(&self, id: usize, expert: &Expert) {
        if let ExpertSet::Streamed(s) = self {
            s.release(id, expert);
        }
    }

    /// The bytes one routed expert occupies as stored.
    pub fn expert_bytes(&self) -> Result<usize> {
        match self {
            ExpertSet::Resident(v) => Ok(v.first().map(|e| e.bytes()).unwrap_or(0)),
            ExpertSet::Streamed(s) => s.expert_bytes(),
        }
    }

    /// Per-expert request counts of a streamed set; empty when resident.
    pub fn usage(&self) -> Vec<u64> {
        match self {
            ExpertSet::Resident(_) => Vec::new(),
            ExpertSet::Streamed(s) => s.usage(),
        }
    }

    /// Resize a streamed set's hot cache; a no-op when resident.
    pub fn set_capacity(&self, capacity: usize) {
        if let ExpertSet::Streamed(s) = self {
            s.set_capacity(capacity);
        }
    }
}

/// An `ExpertLoader` over experts already held in memory, handing back a clone on each build. It
/// backs the machinery gate (a small cache over these must match the resident forward) and is the
/// shape a disk-backed, dequant-on-read loader will replace to actually shrink the footprint.
pub struct VecExpertLoader {
    experts: Vec<Arc<Expert>>,
}

impl VecExpertLoader {
    pub fn new(experts: Vec<Arc<Expert>>) -> Self {
        Self { experts }
    }
}

impl ExpertLoader for VecExpertLoader {
    fn build(&self, id: usize) -> Result<Expert> {
        let e = &self.experts[id];
        Ok(Expert {
            w1: e.w1.clone(),
            w2: e.w2.clone(),
            w3: e.w3.clone(),
        })
    }

    fn count(&self) -> usize {
        self.experts.len()
    }
}

/// An `ExpertLoader` over the stacked, block-quantised expert tensors held in the mmap'd GGUF. Each
/// `[E, R, C]` stack is expert-major, so expert `e` is one contiguous byte range viewed zero-copy
/// (`expert_view`) and read in place by the quantised dot engine: nothing is dequantised, the
/// experts a prompt routes to live in the page cache and an evicted one costs a build of three
/// views. The forward differs from a dense one only by the engine's activation quantisation.
pub struct QuantExpertLoader {
    gate: Arc<QTensor>, // [E, inter, dim]
    up: Arc<QTensor>,   // [E, inter, dim]
    down: Arc<QTensor>, // [E, dim, inter]
    n_experts: usize,
    /// For each stack (gate, up, down): its file and the address its mapping starts at, when
    /// the stack is a mapped file. A released expert's pages are then dropped by file range,
    /// which reaches pages read-ahead brought in that this process never mapped; advice on
    /// the mapping alone walks the page tables and cannot see them.
    anchors: [Option<(Arc<std::fs::File>, usize)>; 3],
}

impl QuantExpertLoader {
    pub fn new(gate: Arc<QTensor>, up: Arc<QTensor>, down: Arc<QTensor>, n_experts: usize) -> Self {
        // The stacks keep the kernel's read-ahead: each routed expert is asked for in full before
        // it computes, and a request served page by page runs at a small fraction of the disk.
        Self {
            gate,
            up,
            down,
            n_experts,
            anchors: [None, None, None],
        }
    }

    /// Name the file behind each stack, in (gate, up, down) order.
    pub fn anchored(mut self, anchors: [Option<(Arc<std::fs::File>, usize)>; 3]) -> Self {
        self.anchors = anchors;
        self
    }
}

impl ExpertLoader for QuantExpertLoader {
    fn build(&self, id: usize) -> Result<Expert> {
        let one = |stack: &Arc<QTensor>| -> Result<Projection> {
            Ok(Projection::Quant(Arc::new(expert_view(stack, id)?)))
        };
        Ok(Expert {
            w1: one(&self.gate)?,
            w2: one(&self.down)?,
            w3: one(&self.up)?,
        })
    }

    fn count(&self) -> usize {
        self.n_experts
    }

    fn release(&self, expert: &Expert) {
        let stacks = [
            (&expert.w1, &self.anchors[0]),
            (&expert.w3, &self.anchors[1]),
            (&expert.w2, &self.anchors[2]),
        ];
        for (p, anchor) in stacks {
            match (anchor, p.storage_span()) {
                (Some((file, base)), Some((addr, len))) if addr >= *base => {
                    // This mapping's entries first: the file's pages are not dropped while a
                    // process maps them.
                    p.unmap_pages();
                    #[cfg(target_os = "linux")]
                    {
                        use std::os::fd::AsRawFd;
                        // Safety: a valid descriptor and a range inside the file; advice only.
                        unsafe {
                            libc::posix_fadvise(
                                file.as_raw_fd(),
                                (addr - base) as libc::off_t,
                                len as libc::off_t,
                                libc::POSIX_FADV_DONTNEED,
                            );
                        }
                    }
                }
                _ => p.dont_need(),
            }
        }
    }
}

/// An `ExpertLoader` over stacks already dequantised to f32, sliced per expert on build. The path
/// a stack takes when its dtype has no in-place view; every expert is resident from the start.
pub struct DenseStackLoader {
    gate: Tensor, // [E, inter, dim]
    up: Tensor,   // [E, inter, dim]
    down: Tensor, // [E, dim, inter]
    n_experts: usize,
}

impl DenseStackLoader {
    pub fn new(gate: Tensor, up: Tensor, down: Tensor, n_experts: usize) -> Self {
        Self {
            gate,
            up,
            down,
            n_experts,
        }
    }
}

impl ExpertLoader for DenseStackLoader {
    fn build(&self, id: usize) -> Result<Expert> {
        let slice = |t: &Tensor| -> Result<Tensor> { t.narrow(0, id, 1)?.squeeze(0)?.contiguous() };
        Ok(Expert::dense(
            slice(&self.gate)?,
            slice(&self.down)?,
            slice(&self.up)?,
        ))
    }

    fn count(&self) -> usize {
        self.n_experts
    }
}
