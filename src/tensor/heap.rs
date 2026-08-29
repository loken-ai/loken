//! What an ALLOCATOR holds for a sequence of allocations, as against what the
//! sequence keeps alive.
//!
//! WHY THIS EXISTS. A dry run counts the live set (see [`super::dry`]) and that is a
//! true number about the program. It is not the number the card reports, and the
//! difference is not a margin: it is the cost of an allocator. A block is placed
//! somewhere, a later free leaves a hole, the next block does not fit the hole, and
//! the memory the driver holds walks away from the memory the program holds. That
//! gap is a property of the ORDER of the events, so no high-water mark can carry it -
//! it has to be replayed.
//!
//! THE ALLOCATOR THIS MODELS, and how its shape was established by probing it. Every
//! device allocation in this substrate goes through the vendored cudarc's
//! `CudaStream::alloc`, which issues `cuMemAllocAsync` whenever the device has a
//! memory pool - so the allocator is the driver's STREAM-ORDERED POOL, and what a
//! card reports is that pool's reservation. Probed, it behaves like one heap:
//!
//! * requests are packed next to each other, aligned to a small boundary, with no
//!   size classes - a kilobyte block and a megabyte block share a chunk;
//! * the backing is reserved in CHUNKS far larger than a request, and a chunk comes
//!   back to the card only when everything on it is free;
//! * holes coalesce and are reused by any request that fits, and a request that fits
//!   no hole grows the reservation by another chunk.
//!
//! Both parameters are PROPERTIES OF THE DEVICE AND ITS DRIVER, not of this model,
//! which is why they are passed in rather than written down: the chunk is what the
//! pool reserves for one small allocation (`CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT`
//! against `USED_MEM_CURRENT`), the alignment is the boundary the driver rounds a
//! request up to. A card, a driver or a platform with different ones gives different
//! answers from the same trace, which is the point.
//!
//! WHAT IT DOES NOT MODEL, and both matter. Anything allocated below the tensor layer
//! - a cuBLAS workspace, a kernel's own scratch, the context and its modules - is not
//! in the trace, so it is not in the answer; the caller adds what it knows sits
//! outside. And the trace is SERIALIZED: it says what the pool holds when every free
//! has been retired before the next allocation is made. A render whose stream is
//! running behind frees later than it allocates, and holds more.

use std::collections::BTreeMap;

use super::dry::AllocEvent;

/// What the replay found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeapCost {
    /// The high-water mark of what the program held: the dry run's own figure.
    pub live_peak: u64,
    /// The high-water mark of what the allocator held - pages with anything live on
    /// them. This is the figure a card reports.
    pub held_peak: u64,
    /// What the allocator held at the moment the LIVE set peaked. Below `held_peak`
    /// whenever the allocator's worst moment is not the program's worst moment.
    pub held_at_live_peak: u64,
    /// Bytes lost to rounding each request up to the allocator's alignment.
    pub alignment_waste: u64,
    /// What the program and the allocator held once the trace ran out. A forward
    /// hands these two to the step that follows it.
    pub live_at_end: u64,
    pub held_at_end: u64,
    /// How many events were replayed, and how many of them were allocations.
    pub events: u64,
    pub allocations: u64,
}

impl HeapCost {
    /// Room the allocator is holding that the program is not using.
    ///
    /// WHAT IT IS FOR. A trace replayed after this one is served out of this room
    /// before the pool asks the card for anything more, so it is the amount by which
    /// replaying that later trace ON ITS OWN over-states what THIS MODEL says the two
    /// of them cost together. That is a property of the model, and the test beside it
    /// proves that property.
    ///
    /// IT IS NOT AN ERROR BAR AGAINST A CARD, and it was briefly written up as one.
    /// The series that appeared to support it - an over-statement of 0.26, 0.16 and
    /// 0.05 GB at three geometries - was arithmetic on a request's peak MINUS a
    /// weights figure this same model had produced. A bound derived from the model it
    /// bounds does not bound anything, and the peaks it leaned on move by ~0.3 GB
    /// between two identical renders. Measured instead by bracketing the denoise loop
    /// directly, which needs no figure from here, the model reads 0.21, 0.31 and
    /// 0.41 GB LOW at those same geometries - the opposite sign.
    ///
    /// So the real error is not bounded by this, and its sign is what a card says, not
    /// what a model says about itself. What the difference is made of is a separate
    /// question with its own evidence: the replay agrees with the card to within 2%
    /// ON THE SAME TRACE, so what is missing is in the trace - the forward allocates
    /// through paths a walk of the tensor layer does not see.
    pub fn slack(&self) -> u64 {
        self.held_at_end.saturating_sub(self.live_at_end)
    }
}

/// Where an allocator puts a block when several holes would take it.
///
/// NOT a detail: it decides how much of a hole is left over, and a leftover too small
/// for the next request is memory the card keeps holding. The two policies disagree by
/// hundreds of megabytes on a real forward, so which one the driver uses is something
/// to establish against the card rather than to assume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// The first hole in address order that takes it.
    First,
    /// The tightest hole that takes it - leaves the large holes intact for the large
    /// requests that follow.
    Best,
}

/// Replay a trace against a first-fit heap reserving in `page`-sized chunks with
/// `align`-byte alignment, and report what it would have held.
///
/// `page` and `align` come from the device (see the module note); passing a page of
/// one and an alignment of one replays with no reservation quantum at all, which
/// measures the pure placement cost.
pub fn replay(trace: &[AllocEvent], page: u64, align: u64) -> HeapCost {
    replay_with(trace, page, align, Fit::First)
}

/// [`replay`] with the placement policy named.
pub fn replay_with(trace: &[AllocEvent], page: u64, align: u64, fit: Fit) -> HeapCost {
    let mut heap = Heap::new(page, align, fit);
    let mut out = HeapCost {
        events: trace.len() as u64,
        ..Default::default()
    };
    // The live total is the DRY RUN's total - raw bytes, not what the allocator
    // rounded them to - so the two figures being compared measure the same thing.
    let mut raw: BTreeMap<u64, u64> = BTreeMap::new();
    let mut live = 0u64;
    for ev in trace {
        match *ev {
            AllocEvent::Alloc { id, bytes } => {
                out.allocations += 1;
                let rounded = round_up(bytes, align);
                out.alignment_waste += rounded - bytes;
                heap.alloc(id, rounded);
                raw.insert(id, bytes);
                live += bytes;
                if live > out.live_peak {
                    out.live_peak = live;
                    out.held_at_live_peak = heap.held();
                }
                out.held_peak = out.held_peak.max(heap.held());
            }
            AllocEvent::Free { id } => {
                heap.free(id);
                live -= raw.remove(&id).unwrap_or(0).min(live);
            }
        }
    }
    out.live_at_end = live;
    out.held_at_end = heap.held();
    out
}

fn round_up(v: u64, to: u64) -> u64 {
    if to <= 1 {
        return v;
    }
    v.div_ceil(to) * to
}

/// A first-fit heap that remembers which pages have something live on them.
struct Heap {
    page: u64,
    align: u64,
    fit: Fit,
    /// Live blocks by offset - a BTreeMap so the gaps between them can be walked in
    /// address order, which is what first-fit means.
    blocks: BTreeMap<u64, u64>,
    /// Offset of each live block, by the id it was allocated under.
    at: BTreeMap<u64, u64>,
    /// How many live blocks touch each page. A page with none comes back to the card.
    pages: BTreeMap<u64, u32>,
    /// One past the highest byte ever handed out.
    top: u64,
    held_pages: u64,
    /// The size the block was rounded to, for the live total.
    sizes: BTreeMap<u64, u64>,
}

impl Heap {
    fn new(page: u64, align: u64, fit: Fit) -> Self {
        Self {
            page: page.max(1),
            align: align.max(1),
            fit,
            blocks: BTreeMap::new(),
            at: BTreeMap::new(),
            pages: BTreeMap::new(),
            top: 0,
            held_pages: 0,
            sizes: BTreeMap::new(),
        }
    }

    fn held(&self) -> u64 {
        self.held_pages * self.page
    }

    /// Place `size` bytes in a gap that fits, by the heap's policy, extending it if
    /// no gap does.
    fn alloc(&mut self, id: u64, size: u64) {
        let mut at: Option<u64> = None;
        let mut best = u64::MAX;
        let mut cursor = 0u64;
        for (&off, &len) in self.blocks.iter() {
            let gap = off - cursor;
            if gap >= size {
                match self.fit {
                    Fit::First => {
                        at = Some(cursor);
                        break;
                    }
                    Fit::Best => {
                        if gap < best {
                            best = gap;
                            at = Some(cursor);
                        }
                    }
                }
            }
            cursor = off + len;
        }
        let off = at.unwrap_or_else(|| {
            let o = round_up(self.top, self.align);
            self.top = o + size;
            o
        });
        if off + size > self.top {
            self.top = off + size;
        }
        self.blocks.insert(off, size);
        self.at.insert(id, off);
        self.sizes.insert(id, size);
        self.touch(off, size, 1);
    }

    /// Take the block back, returning the size it occupied.
    fn free(&mut self, id: u64) -> u64 {
        let Some(off) = self.at.remove(&id) else {
            return 0;
        };
        let size = self.sizes.remove(&id).unwrap_or(0);
        self.blocks.remove(&off);
        self.touch(off, size, -1);
        size
    }

    /// Count a block in or out of every page it touches.
    fn touch(&mut self, off: u64, size: u64, by: i32) {
        if size == 0 {
            return;
        }
        let first = off / self.page;
        let last = (off + size - 1) / self.page;
        for p in first..=last {
            let e = self.pages.entry(p).or_insert(0);
            if by > 0 {
                if *e == 0 {
                    self.held_pages += 1;
                }
                *e += 1;
            } else {
                *e = e.saturating_sub(1);
                if *e == 0 {
                    self.pages.remove(&p);
                    self.held_pages = self.held_pages.saturating_sub(1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 2 << 20;
    const ALIGN: u64 = 512;

    fn alloc(id: u64, bytes: u64) -> AllocEvent {
        AllocEvent::Alloc { id, bytes }
    }
    fn free(id: u64) -> AllocEvent {
        AllocEvent::Free { id }
    }

    /// The page is the unit the card gives up, so one byte costs a page - which is
    /// the whole reason a live-set total under-reports what a card shows.
    #[test]
    fn one_byte_costs_a_page() {
        let c = replay(&[alloc(0, 1)], PAGE, ALIGN);
        assert_eq!(c.live_peak, 1);
        assert_eq!(c.held_peak, PAGE);
    }

    /// Blocks that pack into one page cost one page - no size classes, which is what
    /// the driver was measured to do.
    #[test]
    fn small_blocks_share_a_page() {
        let evs: Vec<_> = (0..4096).map(|i| alloc(i, 512)).collect();
        let c = replay(&evs, PAGE, ALIGN);
        assert_eq!(c.live_peak, PAGE, "4096 x 512 B is exactly a page");
        assert_eq!(c.held_peak, PAGE, "they did not share");
    }

    /// A block freed and re-allocated at the same size costs nothing more: the hole
    /// is reused, which is why a chain of temporaries is not a sum.
    #[test]
    fn a_hole_is_reused_by_the_same_size() {
        let evs = vec![alloc(0, PAGE), alloc(1, PAGE), free(0), alloc(2, PAGE)];
        let c = replay(&evs, PAGE, ALIGN);
        assert_eq!(c.held_peak, 2 * PAGE);
    }

    /// THE PROPERTY THIS MODULE EXISTS FOR. Blocks that outlive their neighbours
    /// leave holes nothing fits, and the allocator holds pages the program does not.
    /// The live set says one thing and the card says another, from the same trace.
    /// THE PROPERTY THIS MODULE EXISTS FOR. Small blocks that outlive the big ones
    /// they were interleaved with pin a page each, and the card goes on holding those
    /// pages for a live set that is now kilobytes. Nothing about the live set - not
    /// its total, not its peak - can express this, which is why the trace is replayed.
    #[test]
    fn scattered_survivors_pin_pages_the_live_set_does_not_explain() {
        let mut evs = Vec::new();
        // A small block and a big one, thirty-two times: the shape of a forward that
        // keeps a norm's output and throws an activation away.
        for i in 0..32 {
            evs.push(alloc(2 * i, 512));
            evs.push(alloc(2 * i + 1, PAGE));
        }
        for i in 0..32 {
            evs.push(free(2 * i + 1));
        }
        let c = replay(&evs, PAGE, ALIGN);
        assert_eq!(c.live_at_end, 32 * 512, "only the small blocks are left");
        assert!(
            c.held_at_end >= 32 * PAGE,
            "the card gave back pages the survivors are still sitting on: held {} for {} live",
            c.held_at_end,
            c.live_at_end
        );
        assert!(
            c.held_at_end / c.live_at_end > 1000,
            "the allocator holds {}x the live set",
            c.held_at_end / c.live_at_end
        );
    }

    /// The two policies are not the same allocator, and a trace can tell them apart.
    ///
    /// A small hole and a large one, then a request that fits both: first-fit takes
    /// the large one and leaves a remnant too small for the request after it, so the
    /// heap grows; best-fit takes the small one and the large request still fits.
    /// Which one a driver uses is therefore worth establishing rather than assuming -
    /// hence the parameter.
    #[test]
    fn first_fit_and_best_fit_disagree_on_the_same_trace() {
        // A wide hole first, a narrow one after it, and small survivors between them
        // so nothing coalesces. The requests then arrive narrow-then-wide: first-fit
        // spends the wide hole on the narrow request and has nowhere to put the wide
        // one, so the heap grows; best-fit keeps the wide hole for the wide request.
        let pin = PAGE / 8;
        let evs = vec![
            alloc(0, pin),
            alloc(1, 4 * PAGE), // becomes the wide hole
            alloc(2, pin),
            alloc(3, 2 * PAGE), // becomes the narrow hole
            alloc(4, pin),
            free(1),
            free(3),
            alloc(5, 2 * PAGE),
            alloc(6, 4 * PAGE),
        ];
        let first = replay_with(&evs, PAGE, ALIGN, Fit::First);
        let best = replay_with(&evs, PAGE, ALIGN, Fit::Best);
        assert!(
            best.held_peak < first.held_peak,
            "first-fit held {} and best-fit held {}",
            first.held_peak,
            best.held_peak
        );
        assert_eq!(
            best.live_peak, first.live_peak,
            "the live set is the program's, not the heap's"
        );
    }

    /// The leftover room is what a later trace is served out of, and it is the reason
    /// replaying that trace alone over-states what a card gives up for it.
    #[test]
    fn the_leftovers_of_one_trace_serve_the_next_one() {
        // A trace that ends holding a little in a lot of pages.
        let mut evs = Vec::new();
        for i in 0..8 {
            evs.push(alloc(2 * i, PAGE / 8));
            evs.push(alloc(2 * i + 1, PAGE));
        }
        for i in 0..8 {
            evs.push(free(2 * i + 1));
        }
        let first = replay(&evs, PAGE, ALIGN);
        assert!(first.slack() > 0, "the allocator held nothing spare");

        // The same second trace, replayed on its own and replayed after the first.
        // Adding the two answers is what over-states: the second trace never asks the
        // card for what the first one is already holding spare.
        let second: Vec<AllocEvent> = (0..6).map(|i| alloc(100 + i, PAGE / 4)).collect();
        let alone = replay(&second, PAGE, ALIGN);
        let together = replay(&[evs, second].concat(), PAGE, ALIGN);
        let naive = first.held_at_end + alone.held_peak;
        assert!(
            together.held_peak < naive,
            "the two answers added ({naive}) did not over-state the truth ({})",
            together.held_peak
        );
        assert!(
            naive - together.held_peak <= first.slack(),
            "the over-statement escaped the bound the leftovers set"
        );
    }

    /// The alignment is counted where it lands: on every request, not on the total.
    #[test]
    fn alignment_waste_is_per_request() {
        let evs = vec![alloc(0, 1), alloc(1, 1), alloc(2, 1)];
        let c = replay(&evs, PAGE, ALIGN);
        assert_eq!(c.alignment_waste, 3 * (ALIGN - 1));
    }

    /// With no page and no alignment the model reduces to the live set, so the two
    /// figures can be compared against each other with the allocator switched off.
    #[test]
    fn without_a_page_the_model_is_the_live_set() {
        let evs = vec![alloc(0, 1000), alloc(1, 2000), free(0), alloc(2, 500)];
        let c = replay(&evs, 1, 1);
        assert_eq!(c.held_peak, c.live_peak);
    }
}
