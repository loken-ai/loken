//! The most routed experts of every tiered layer held in host memory, when the pin budget cannot
//! hold the layers whole: their pages in the file mapping are locked with `mlock`, ranked across
//! layers by heat, and moved between requests.

use super::LayerTier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A half-open byte range `[lo, hi)` of the address space.
pub(crate) type Range = (usize, usize);

/// The most bytes of whole pages one expert can make resident: `ceil(b / page) + 1` pages in each
/// stack of `b` bytes an expert, whatever its alignment.
pub(crate) fn pair_cost(expert_bytes: &[usize; 3], page: usize) -> u64 {
    expert_bytes
        .iter()
        .map(|&b| ((b.div_ceil(page) + 1) * page) as u64)
        .sum()
}

/// The (layer, expert) pairs to hold: the hottest first across every layer, each taken while its
/// layer's `cost` fits in what `budget` has left. A pair with no heat is never taken. Sorted by
/// layer, then expert.
pub(crate) fn rank(heat: &[Vec<f32>], cost: &[u64], budget: u64) -> Vec<(usize, usize)> {
    let mut pairs: Vec<(usize, usize)> = heat
        .iter()
        .enumerate()
        .flat_map(|(l, h)| {
            h.iter()
                .enumerate()
                .filter(|&(_, &x)| x > 0.0)
                .map(move |(e, _)| (l, e))
        })
        .collect();
    pairs.sort_by(|a, b| heat[b.0][b.1].total_cmp(&heat[a.0][a.1]).then(a.cmp(b)));
    let mut left = budget;
    let mut hot: Vec<(usize, usize)> = pairs
        .into_iter()
        .filter(|&(l, _)| {
            let fits = cost[l] <= left;
            if fits {
                left -= cost[l];
            }
            fits
        })
        .collect();
    hot.sort_unstable();
    hot
}

/// The pages under `experts` (sorted, distinct) of a stack whose expert `e` is the bytes
/// `[base + e * bytes, base + (e + 1) * bytes)`: one range for each run of consecutive experts,
/// widened to whole pages.
pub(crate) fn cover(base: usize, bytes: usize, experts: &[usize], page: usize) -> Vec<Range> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < experts.len() {
        let mut j = i + 1;
        while j < experts.len() && experts[j] == experts[j - 1] + 1 {
            j += 1;
        }
        let lo = base + experts[i] * bytes;
        let hi = base + (experts[j - 1] + 1) * bytes;
        if hi > lo {
            out.push((lo / page * page, hi.div_ceil(page) * page));
        }
        i = j;
    }
    out
}

/// `ranges` sorted, the ranges that overlap or touch made one: no byte lies in two of them.
pub(crate) fn merge(mut ranges: Vec<Range>) -> Vec<Range> {
    ranges.retain(|r| r.1 > r.0);
    ranges.sort_unstable();
    let mut out: Vec<Range> = Vec::with_capacity(ranges.len());
    for (lo, hi) in ranges {
        match out.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}

/// What of `a` lies outside `b`, both merged: merged as well.
pub(crate) fn subtract(a: &[Range], b: &[Range]) -> Vec<Range> {
    let mut out = Vec::new();
    let mut j = 0;
    for &(lo, hi) in a {
        while j < b.len() && b[j].1 <= lo {
            j += 1;
        }
        let mut cur = lo;
        let mut k = j;
        while k < b.len() && b[k].0 < hi {
            if b[k].0 > cur {
                out.push((cur, b[k].0));
            }
            cur = cur.max(b[k].1);
            k += 1;
        }
        if cur < hi {
            out.push((cur, hi));
        }
    }
    out
}

/// Moving the registered ranges `old` to the merged `target`: the ranges released, each whole,
/// and the ranges registered; a range that changes shape is in both.
pub(crate) fn reregister(old: &[Range], target: &[Range]) -> (Vec<Range>, Vec<Range>) {
    (
        old.iter()
            .filter(|r| !target.contains(r))
            .copied()
            .collect(),
        target
            .iter()
            .filter(|r| !old.contains(r))
            .copied()
            .collect(),
    )
}

/// Bytes the merged `ranges` span.
fn span(ranges: &[Range]) -> u64 {
    ranges.iter().map(|r| (r.1 - r.0) as u64).sum()
}

/// The ranges registered now and those on their way in or out, sorted and disjoint. A copy from
/// the mapping reads it to cut itself at their edges, for the driver refuses a copy that spans a
/// registered range and memory outside it. A range is listed before its registration starts and
/// unlisted after its release ends, and the write lock is held only to edit the list: the
/// driver's calls run without it, since a release waits for the card, and a card waiting on the
/// tier worker waits on a copy that reads this list.
static REGISTERED: std::sync::RwLock<Vec<Range>> = std::sync::RwLock::new(Vec::new());

/// Calls `f` on the pieces of `[from, to)` cut at the edges of the registered ranges, in order,
/// while no range is registered or released.
pub(crate) fn for_each_piece(
    from: usize,
    to: usize,
    mut f: impl FnMut(usize, usize) -> crate::tensor::Result<()>,
) -> crate::tensor::Result<()> {
    let reg = REGISTERED.read().unwrap_or_else(|e| e.into_inner());
    let mut at = from;
    let first = reg.partition_point(|r| r.1 <= from);
    for r in reg[first..].iter().take_while(|r| r.0 < to) {
        let (lo, hi) = (r.0.max(at), r.1.min(to));
        if lo > at {
            f(at, lo)?;
        }
        f(lo, hi)?;
        at = hi;
    }
    if at < to {
        f(at, to)?;
    }
    Ok(())
}

/// Locks `r` in host memory: registered with the card's driver, which pins the pages outside
/// the process's locked-memory limit and lets copies from them run as transfers.
#[cfg(feature = "gpu")]
fn lock(r: Range) -> bool {
    {
        let mut reg = REGISTERED.write().unwrap_or_else(|e| e.into_inner());
        let at = reg.partition_point(|x| x.0 < r.0);
        reg.insert(at, r);
    }
    // Read-only, the mapping being the file's; portable, every card's context seeing it pinned.
    const PORTABLE: u32 = 1;
    const READ_ONLY: u32 = 8;
    // Safety: the range lies in mappings the tiers keep alive; registration only pins it.
    let rc = unsafe {
        crate::tensor::gpu_runtime::driver::sys::cuMemHostRegister_v2(
            r.0 as *mut std::ffi::c_void,
            r.1 - r.0,
            PORTABLE | READ_ONLY,
        )
    };
    let ok = rc == crate::tensor::gpu_runtime::driver::sys::cudaError_enum::CUDA_SUCCESS;
    if !ok {
        REGISTERED
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|x| *x != r);
    }
    ok
}

/// Releases a range [`lock`] registered, by the address it was registered at.
#[cfg(feature = "gpu")]
fn unlock(r: Range) {
    // Safety: `r` was registered by `lock` and is released once.
    let _ = unsafe {
        crate::tensor::gpu_runtime::driver::sys::cuMemHostUnregister(r.0 as *mut std::ffi::c_void)
    };
    REGISTERED
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|x| *x != r);
}

#[cfg(not(feature = "gpu"))]
fn lock(_: Range) -> bool {
    false
}

#[cfg(not(feature = "gpu"))]
fn unlock(_: Range) {}

/// What is locked now.
struct Locked {
    /// The locked pages, merged.
    ranges: Vec<Range>,
    /// The (tier, expert) pairs whose bytes in all three stacks lie in `ranges`.
    hot: Vec<(usize, usize)>,
    /// Each tier's routing counts at the last re-rank.
    counts: Vec<Vec<u32>>,
    /// An mlock failed: nothing more is locked.
    off: bool,
}

/// The hot experts of the tiered layers, held in host memory within `budget` bytes.
pub struct HotLocks {
    /// The tiers, with the model layer each serves.
    tiers: Vec<(usize, Arc<LayerTier>)>,
    budget: u64,
    page: usize,
    state: Mutex<Locked>,
    /// A re-rank is applying its locks: the next one is skipped.
    busy: AtomicBool,
}

impl HotLocks {
    /// Nothing locked yet over `tiers` and `budget`; `None` when the page size is unknown.
    pub(crate) fn new(tiers: Vec<(usize, Arc<LayerTier>)>, budget: u64) -> Option<Self> {
        let page = crate::tensor::mapped::page_size()?;
        let counts = tiers.iter().map(|(_, t)| t.routing_counts()).collect();
        tracing::info!(
            "hot experts: {} layers too large to lock whole in {:.2} GB of host memory; their most routed experts are locked instead, once a profile or a request ranks them",
            tiers.len(),
            budget as f64 / 1e9
        );
        Some(Self {
            tiers,
            budget,
            page,
            state: Mutex::new(Locked {
                ranges: Vec::new(),
                hot: Vec::new(),
                counts,
                off: false,
            }),
            busy: AtomicBool::new(false),
        })
    }

    fn costs(&self) -> Vec<u64> {
        self.tiers
            .iter()
            .map(|(_, t)| pair_cost(&t.expert_bytes, self.page))
            .collect()
    }

    /// The pages under each pair's bytes in the three stacks of its tier, one list a pair.
    fn covers(&self, tier: usize, experts: &[usize]) -> Vec<Range> {
        let t = &self.tiers[tier].1;
        (0..3)
            .flat_map(|i| {
                let base = t.host[i].native_qtensor().data().as_ptr() as usize;
                cover(base, t.expert_bytes[i], experts, self.page)
            })
            .collect()
    }

    /// The merged pages under every pair of `hot` (sorted by tier).
    fn target(&self, hot: &[(usize, usize)]) -> Vec<Range> {
        let mut all = Vec::new();
        for tier in 0..self.tiers.len() {
            let experts: Vec<usize> = hot.iter().filter(|p| p.0 == tier).map(|p| p.1).collect();
            if !experts.is_empty() {
                all.extend(self.covers(tier, &experts));
            }
        }
        merge(all)
    }

    /// Rank by the counts of `profile` and lock the result, off the caller's thread.
    pub(crate) fn prime(self: &Arc<Self>, profile: &super::super::expert_profile::ExpertProfile) {
        let weights: Vec<Vec<f32>> = self
            .tiers
            .iter()
            .map(|(l, t)| {
                let c = profile.layer(*l);
                if c.len() == t.n_experts {
                    c.iter().map(|&x| x as f32).collect()
                } else {
                    vec![0.0; t.n_experts]
                }
            })
            .collect();
        self.start(weights, None);
    }

    /// At a request's end: rank by the tiers' routing counts since the load, steadier than the
    /// cache's decayed heat, and move the locks, off the caller's thread.
    /// A request that routed nothing through the tiers moves nothing.
    pub(crate) fn rerank(self: &Arc<Self>) {
        if self.busy.load(Ordering::Acquire) {
            return;
        }
        let (covered, routed, weights) = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let now: Vec<Vec<u32>> = self.tiers.iter().map(|(_, t)| t.routing_counts()).collect();
            let delta = |l: usize, e: usize| {
                u64::from(now[l][e].saturating_sub(st.counts[l].get(e).copied().unwrap_or(0)))
            };
            let routed: u64 = (0..now.len())
                .map(|l| (0..now[l].len()).map(|e| delta(l, e)).sum::<u64>())
                .sum();
            let covered: u64 = st.hot.iter().map(|&(l, e)| delta(l, e)).sum();
            let weights: Vec<Vec<f32>> = now
                .iter()
                .map(|v| v.iter().map(|&c| c as f32).collect())
                .collect();
            st.counts = now;
            (covered, routed, weights)
        };
        if routed == 0 {
            return;
        }
        self.start(weights, Some((covered, routed)));
    }

    /// Rank `weights` and apply the result on a thread of its own, unless one is applying.
    fn start(self: &Arc<Self>, weights: Vec<Vec<f32>>, traffic: Option<(u64, u64)>) {
        if self.busy.swap(true, Ordering::AcqRel) {
            return;
        }
        let hot = rank(&weights, &self.costs(), self.budget);
        let me = self.clone();
        let spawned = std::thread::Builder::new()
            .name("hot-experts".into())
            .spawn(move || {
                me.apply(hot, traffic);
                me.busy.store(false, Ordering::Release);
            });
        if let Err(e) = spawned {
            tracing::warn!("hot experts not moved: {e}");
            self.busy.store(false, Ordering::Release);
        }
    }

    /// Release the registered ranges `hot` no longer covers as they are, register the ranges
    /// of `hot` not yet registered, and log the held set. Registrations are released whole, by
    /// the address they were made at, so a range that changes shape is registered again.
    fn apply(&self, hot: Vec<(usize, usize)>, traffic: Option<(u64, u64)>) {
        // The driver registers memory for the thread's current context: a card's, bound here.
        #[cfg(feature = "gpu")]
        if let Some(cd) = self
            .tiers
            .first()
            .and_then(|(_, t)| t.dev.as_cuda_device().ok())
        {
            let _ = cd.0.context().bind_to_thread();
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let target = self.target(&hot);
        let (gone, added) = reregister(&st.ranges, &target);
        gone.iter().for_each(|&r| unlock(r));
        let mut ranges: Vec<Range> = st
            .ranges
            .iter()
            .filter(|r| !gone.contains(r))
            .copied()
            .collect();
        if !st.off {
            for r in added {
                if !lock(r) {
                    tracing::warn!(
                        "hot experts: the driver refused to lock {:.1} MB; no more experts are locked in host memory",
                        (r.1 - r.0) as f64 / 1e6
                    );
                    st.off = true;
                    break;
                }
                ranges.push(r);
            }
        }
        ranges.sort_unstable();
        st.ranges = ranges;
        st.hot = hot
            .into_iter()
            .filter(|&(l, e)| subtract(&merge(self.covers(l, &[e])), &st.ranges).is_empty())
            .collect();
        let share = traffic.map_or(String::new(), |(covered, routed)| {
            format!(
                "; they held {:.1}% of the {routed} routed experts of the last request",
                100.0 * covered as f64 / routed.max(1) as f64
            )
        });
        tracing::info!(
            "hot experts: {} locked in host memory over {} layers, {:.2} GB of {:.2} GB{share}",
            st.hot.len(),
            self.tiers.len(),
            span(&st.ranges) as f64 / 1e9,
            self.budget as f64 / 1e9
        );
    }
}

impl Drop for HotLocks {
    fn drop(&mut self) {
        let st = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        st.ranges.iter().for_each(|&r| unlock(r));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hottest_pairs_of_every_layer_share_the_budget() {
        let heat = vec![vec![5.0, 0.0, 1.0], vec![4.0, 3.0, 0.5]];
        let cost = vec![10, 20];
        // 5.0 (10), 4.0 (20), 3.0 (20 > 15 left: skipped), 1.0 (10), 0.5 (20: skipped).
        assert_eq!(rank(&heat, &cost, 45), vec![(0, 0), (0, 2), (1, 0)]);
        assert_eq!(rank(&heat, &cost, 9), vec![]);
        // No heat, nothing taken, however wide the budget.
        assert!(rank(&[vec![0.0; 4]], &[1], u64::MAX).is_empty());
    }

    #[test]
    fn a_pair_never_costs_less_than_its_pages() {
        let page = 4096;
        for bytes in [1, 4095, 4096, 4097, 10_000] {
            let cost = pair_cost(&[bytes; 3], page);
            for base in [0, 1, 4095, 4096, 8191] {
                let pages = span(&cover(base, bytes, &[0], page)) * 3;
                assert!(pages <= cost, "{bytes} bytes at {base}");
            }
        }
    }

    #[test]
    fn a_cover_holds_every_byte_and_no_page_twice() {
        let page = 4096;
        let (base, bytes) = (1000, 5000);
        let experts = [0, 1, 2, 5, 6, 8, 10];
        let runs = cover(base, bytes, &experts, page);
        // Four runs, each on whole pages.
        assert_eq!(runs.len(), 4);
        assert!(runs.iter().all(|r| r.0 % page == 0 && r.1 % page == 0));
        let merged = merge(runs.clone());
        for &e in &experts {
            let (lo, hi) = (base + e * bytes, base + (e + 1) * bytes);
            assert!(merged.iter().any(|r| r.0 <= lo && hi <= r.1), "expert {e}");
        }
        for w in merged.windows(2) {
            assert!(w[0].1 < w[1].0, "pages shared by {w:?}");
        }
        // Experts 8 and 10 end and start on the same page: one range.
        assert_eq!(merged.len(), 3);
        assert_eq!(
            span(&merged),
            span(&merge(runs.into_iter().rev().collect()))
        );
    }

    #[test]
    fn a_copy_is_cut_at_the_edges_of_the_registered_ranges() {
        // Addresses no mapping holds, so no other test's ranges meet them.
        let base = usize::MAX / 2;
        let reg = [(base + 10, base + 20), (base + 30, base + 40)];
        REGISTERED.write().unwrap().extend(reg);
        let pieces = |from: usize, to: usize| {
            let mut out = Vec::new();
            for_each_piece(base + from, base + to, |lo, hi| {
                out.push((lo - base, hi - base));
                Ok(())
            })
            .unwrap();
            out
        };
        let all = pieces(0, 50);
        let mid = pieces(15, 35);
        let inside = pieces(21, 29);
        REGISTERED.write().unwrap().retain(|r| !reg.contains(r));
        assert_eq!(all, vec![(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]);
        assert_eq!(mid, vec![(15, 20), (20, 30), (30, 35)]);
        assert_eq!(inside, vec![(21, 29)]);
    }

    #[test]
    fn a_rerank_releases_whole_ranges_and_registers_new_ones() {
        let old = vec![(0, 4), (8, 12)];
        let new = vec![(0, 4), (8, 14), (16, 20)];
        let (gone, added) = reregister(&old, &new);
        // A range that grew is released and registered again; one unchanged stays.
        assert_eq!(gone, vec![(8, 12)]);
        assert_eq!(added, vec![(8, 14), (16, 20)]);
        let mut after: Vec<Range> = old.iter().filter(|r| !gone.contains(r)).copied().collect();
        after.extend(added);
        after.sort_unstable();
        assert_eq!(after, new);
        assert_eq!(reregister(&new, &new), (vec![], vec![]));
    }
}
