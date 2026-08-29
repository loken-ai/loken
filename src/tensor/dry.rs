//! A device that runs the forward and allocates nothing.
//!
//! WHY THIS EXISTS. Every placement decision needs one number: how much memory a
//! forward of THIS request will need beyond the weights. Until now that number came
//! from a formula written next to the model - `4 * dim` where the family builds its
//! feed-forward at `8/3 * dim`, a cast copy nobody counted, a norm that widens
//! internally. The formula and the forward are two descriptions of one computation,
//! and they drift, silently, in whichever direction the next request happens to
//! expose.
//!
//! There is only one description that cannot drift: the forward itself. So run it -
//! the real code, the real shapes, the real ops - on a device that owns no memory and
//! only keeps a running total. Nothing is allocated, no kernel launches, no card is
//! touched; what comes back is the high-water mark of everything the forward would
//! have held at once.
//!
//! This is `ggml_gallocr_reserve`'s measure pass, arrived at from the other side.
//! llama.cpp can walk a graph because its forward IS a graph object; ours is
//! imperative Rust, so the equivalent of "walk the graph without allocating" is
//! "execute the code with an allocator that counts".
//!
//! BUFFER REUSE IS NOT OPTIONAL, and it is the part that comes for free here. A sum
//! of every allocation is not a peak - a denoiser that runs forty blocks would report
//! forty blocks' worth of scratch, and an over-stated reserve is not caution, it is
//! blocks pushed onto the host. ggml has to reconstruct tensor lifetimes explicitly
//! (`n_children`/`n_views` counters) to know when a buffer may be recycled. Here the
//! lifetime IS the Rust one: a tensor's storage is an `Arc`, the last drop runs
//! [`DryStorage::drop`], and the total goes back down at exactly the point the real
//! allocator would have freed. The occupancy profile is therefore the same profile,
//! by construction, not by a second model of it.
//!
//! WHAT IT IS NOT. The count is the LIVE SET at its high-water mark. A real driver
//! also holds fragmentation, its own suballocator's rounding, and whatever a kernel
//! allocates internally below the tensor layer. So this is a lower bound on what the
//! card must have free, exact about the part it describes and silent about the rest -
//! which is why it is reported alongside the reserve in use rather than replacing it
//! until the two have been compared on a real render.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::DType;

/// One thing the forward did to memory, in the order it did it.
///
/// The identity matters as much as the size: an allocator's cost depends on WHICH
/// block came back, not just on how many bytes did, so a trace of sizes alone cannot
/// be replayed against a heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocEvent {
    Alloc { id: u64, bytes: u64 },
    Free { id: u64 },
}

/// A place to run a forward that owns no memory.
///
/// Instances are independent: two dry runs in flight - two requests being planned at
/// once - never see each other's totals. That is deliberate and it is why this is a
/// device rather than a mode flag on the CUDA one: `CudaDevice::get` hands the SAME
/// handle to every request in the process, so a "count instead of allocate" flag on it
/// would turn a live render into a no-op and hand the user a black image.
#[derive(Debug, Default)]
pub struct DryDevice {
    live: AtomicU64,
    peak: AtomicU64,
    allocations: AtomicU64,
    /// Times the forward asked for data that a dry tensor does not have.
    ///
    /// It is not automatically wrong - a positional-id table read back to build the
    /// rotary tables produces the same SHAPES whatever the values are - but it is the
    /// one way a dry run can silently diverge from the real one (a slice whose length
    /// depends on a value, a branch on a norm), so the caller is told and can refuse
    /// the answer.
    blind_reads: AtomicU64,
    /// Whether the events are being kept, checked without taking the lock so a dry
    /// run that does not want them pays one relaxed load per allocation.
    recording: AtomicBool,
    /// The events, in order, when [`DryDevice::record_trace`] asked for them.
    ///
    /// WHY A TRACE AT ALL, when the peak is already counted. The peak is the live
    /// set; what a card gives up is what its ALLOCATOR holds, which depends on the
    /// order the blocks arrived and left. That is a property of the sequence, not of
    /// its high-water mark, so it cannot be recovered from any single number - it has
    /// to be replayed. Off by default: a placement decision does not need it.
    trace: Mutex<Vec<AllocEvent>>,
}

impl DryDevice {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The high-water mark of everything held at once, in bytes.
    pub fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// What is still held right now - zero once the forward's tensors are dropped.
    pub fn live_bytes(&self) -> u64 {
        self.live.load(Ordering::Relaxed)
    }

    /// How many device allocations the forward performed.
    pub fn allocations(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Whether the forward read values that a dry run cannot provide.
    pub fn read_absent_data(&self) -> bool {
        self.blind_reads.load(Ordering::Relaxed) > 0
    }

    pub(crate) fn note_blind_read(&self) {
        self.blind_reads.fetch_add(1, Ordering::Relaxed);
    }

    /// Start a new window: the high-water mark restarts from what is held right
    /// now, so what follows is measured against this moment rather than against
    /// everything that came before it.
    ///
    /// The same shape as the driver's own `USED_MEM_HIGH` reset, and for the same
    /// reason: a peak is only meaningful relative to a baseline, and the baseline a
    /// placement cares about is "the weights are resident, now run one forward".
    pub fn open_window(&self) -> u64 {
        let live = self.live.load(Ordering::Relaxed);
        self.peak.store(live, Ordering::Relaxed);
        live
    }

    /// Keep the event sequence from here on, so an allocator can be replayed against
    /// it. Costs one push per allocation and is off unless asked for.
    pub fn record_trace(&self) {
        self.recording.store(true, Ordering::Relaxed);
    }

    /// The events recorded so far, oldest first.
    pub fn trace(&self) -> Vec<AllocEvent> {
        self.trace.lock().map(|t| t.clone()).unwrap_or_default()
    }

    /// The identity given to the allocation, so its free can be paired with it.
    fn charge(&self, bytes: u64) -> u64 {
        let now = self.live.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak.fetch_max(now, Ordering::Relaxed);
        let id = self.allocations.fetch_add(1, Ordering::Relaxed);
        if self.recording.load(Ordering::Relaxed) {
            if let Ok(mut t) = self.trace.lock() {
                t.push(AllocEvent::Alloc { id, bytes });
            }
        }
        id
    }

    fn release(&self, bytes: u64, id: u64) {
        self.live.fetch_sub(bytes, Ordering::Relaxed);
        if self.recording.load(Ordering::Relaxed) {
            if let Ok(mut t) = self.trace.lock() {
                t.push(AllocEvent::Free { id });
            }
        }
    }
}

/// A tensor's memory, as a number.
///
/// Holds no buffer: the whole point is that the shape and the dtype are the entire
/// truth about what a tensor costs, and both are known the moment the op that produces
/// it decides its output shape.
#[derive(Debug)]
pub struct DryStorage {
    dtype: DType,
    elems: usize,
    dev: Arc<DryDevice>,
    /// Which allocation this was, so the trace can pair its free with it.
    id: u64,
}

impl DryStorage {
    pub fn new(dev: Arc<DryDevice>, dtype: DType, elems: usize) -> Self {
        let id = dev.charge(bytes_of(dtype, elems));
        Self {
            dtype,
            elems,
            dev,
            id,
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.elems
    }

    pub fn is_empty(&self) -> bool {
        self.elems == 0
    }

    pub fn device(&self) -> &Arc<DryDevice> {
        &self.dev
    }

    pub fn bytes(&self) -> u64 {
        bytes_of(self.dtype, self.elems)
    }
}

/// The frees. Without this the count is a sum of allocations rather than a peak, and
/// a thirty-block stack reports thirty times its scratch.
impl Drop for DryStorage {
    fn drop(&mut self) {
        self.dev.release(bytes_of(self.dtype, self.elems), self.id);
    }
}

fn bytes_of(dtype: DType, elems: usize) -> u64 {
    elems as u64 * dtype.size_in_bytes() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The peak follows what is HELD, not what has been allocated.
    ///
    /// This is the property that separates a measure pass from a sum. Two buffers
    /// allocated one after the other, the first dropped before the second exists,
    /// cost one buffer - which is what the real allocator does when it hands the
    /// freed block straight back.
    #[test]
    fn a_freed_tensor_gives_its_room_back() {
        let dev = DryDevice::new();
        {
            let _a = DryStorage::new(dev.clone(), DType::F32, 1_000);
        }
        let _b = DryStorage::new(dev.clone(), DType::F32, 1_000);
        assert_eq!(
            dev.peak_bytes(),
            4_000,
            "the peak counted both, so it is a sum"
        );
        assert_eq!(dev.live_bytes(), 4_000);
        assert_eq!(dev.allocations(), 2);
    }

    /// ...and two buffers alive at once cost both.
    #[test]
    fn overlapping_tensors_are_both_charged() {
        let dev = DryDevice::new();
        let _a = DryStorage::new(dev.clone(), DType::F32, 1_000);
        let _b = DryStorage::new(dev.clone(), DType::F32, 1_000);
        assert_eq!(dev.peak_bytes(), 8_000);
    }

    /// The dtype is half the answer: the same shape at BF16 costs half what it does
    /// at F32, and a demand that assumes one width for a forward that runs the other
    /// is wrong by a factor of two - which at these sizes is a whole card.
    #[test]
    fn the_dtype_decides_half_the_cost() {
        let dev = DryDevice::new();
        let _a = DryStorage::new(dev.clone(), DType::BF16, 1_000);
        assert_eq!(dev.peak_bytes(), 2_000);
    }

    /// Dry runs are independent: one request being planned cannot read another's
    /// total.
    #[test]
    fn two_dry_runs_do_not_see_each_other() {
        let a = DryDevice::new();
        let b = DryDevice::new();
        let _x = DryStorage::new(a.clone(), DType::F32, 1_000);
        assert_eq!(b.peak_bytes(), 0);
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::tensor::{Device, Tensor};

    /// THE gate on this whole mechanism.
    ///
    /// A dry run is only worth anything if it creates the same tensors the real
    /// forward does. Each op's dry arm is handed the output shape the op has already
    /// computed for its real path - so they cannot disagree by construction - and
    /// this is what keeps that true as the ops change: every op below is run twice
    /// on the same inputs, once computing and once counting, and the two results
    /// must have the same shape and the same dtype. A dry arm that starts deriving
    /// its own shape shows up here rather than as a reserve that is quietly wrong.
    #[test]
    fn every_counted_op_produces_the_shape_the_real_one_does() {
        let dry = Device::dry();
        let mk = |shape: &[usize], dtype: DType| -> (Tensor, Tensor) {
            let cpu = Tensor::zeros(shape.to_vec(), dtype).unwrap();
            let d = Tensor::dry(&dry, dtype, shape.to_vec()).unwrap();
            (cpu, d)
        };
        let same = |name: &str, a: &Tensor, b: &Tensor| {
            assert_eq!(a.dims(), b.dims(), "{name}: shape");
            assert_eq!(a.dtype(), b.dtype(), "{name}: dtype");
        };

        let (a_c, a_d) = mk(&[2, 3, 4], DType::F32);
        let (b_c, b_d) = mk(&[2, 4, 5], DType::F32);
        same(
            "matmul",
            &a_c.matmul(&b_c).unwrap(),
            &a_d.matmul(&b_d).unwrap(),
        );

        let (t_c, t_d) = mk(&[2, 5, 4], DType::F32);
        same(
            "matmul_t",
            &a_c.matmul_t(&t_c).unwrap(),
            &a_d.matmul_t(&t_d).unwrap(),
        );

        let (r_c, r_d) = mk(&[1, 1, 4], DType::F32);
        same(
            "broadcast_add",
            &a_c.broadcast_add(&r_c).unwrap(),
            &a_d.broadcast_add(&r_d).unwrap(),
        );
        same(
            "broadcast_mul",
            &a_c.broadcast_mul(&r_c).unwrap(),
            &a_d.broadcast_mul(&r_d).unwrap(),
        );

        let (s_c, s_d) = mk(&[2, 3, 4], DType::F32);
        same("add", &(&a_c + &s_c).unwrap(), &(&a_d + &s_d).unwrap());
        same(
            "to_dtype",
            &a_c.to_dtype(DType::BF16).unwrap(),
            &a_d.to_dtype(DType::BF16).unwrap(),
        );
        same("silu", &a_c.silu().unwrap(), &a_d.silu().unwrap());
        same(
            "softmax",
            &a_c.softmax_last_dim().unwrap(),
            &a_d.softmax_last_dim().unwrap(),
        );
        same(
            "affine",
            &a_c.affine(2.0, 1.0).unwrap(),
            &a_d.affine(2.0, 1.0).unwrap(),
        );
        same("scale", &a_c.scale(2.0).unwrap(), &a_d.scale(2.0).unwrap());

        let (w_c, w_d) = mk(&[4], DType::F32);
        same(
            "rms_norm",
            &a_c.rms_norm(&w_c, 1e-6).unwrap(),
            &a_d.rms_norm(&w_d, 1e-6).unwrap(),
        );
        same(
            "layer_norm",
            &a_c.layer_norm(&w_c, None, 1e-6).unwrap(),
            &a_d.layer_norm(&w_d, None, 1e-6).unwrap(),
        );

        same(
            "cat",
            &Tensor::cat(&[&a_c, &s_c], 1).unwrap(),
            &Tensor::cat(&[&a_d, &s_d], 1).unwrap(),
        );

        // A gather whose INDICES are real host values and whose table is counted -
        // the shape of the rotary tables' lookup, and the one case where a dry run
        // reads values it does not have.
        let ids = Tensor::from_vec_u32(vec![0u32, 1, 1], 3).unwrap();
        let (tab_c, tab_d) = mk(&[4, 6], DType::F32);
        same(
            "index_select",
            &tab_c.index_select(&ids, 0).unwrap(),
            &tab_d.index_select(&ids, 0).unwrap(),
        );

        // Views that materialise: the transposed-then-packed copies attention holds.
        same(
            "transpose+contiguous",
            &a_c.transpose(1, 2).unwrap().contiguous().unwrap(),
            &a_d.transpose(1, 2).unwrap().contiguous().unwrap(),
        );

        // The reductions. Every one of these was missing its counted arm while this
        // test looked complete, and a language model reaches them in its first layer -
        // so the omission showed up as a forward that refused rather than as a red
        // test, which is the failure this list exists to prevent.
        same("sum", &a_c.sum(1).unwrap(), &a_d.sum(1).unwrap());
        same(
            "max_keepdim",
            &a_c.max_keepdim(2).unwrap(),
            &a_d.max_keepdim(2).unwrap(),
        );

        // Rotary embedding: the shape and the dtype come back, the angle does not.
        let (x4_c, x4_d) = mk(&[1, 2, 3, 4], DType::F32);
        let (tab_cos_c, tab_cos_d) = mk(&[3, 2], DType::F32);
        same(
            "rope",
            &x4_c.rope(&tab_cos_c, &tab_cos_c).unwrap(),
            &x4_d.rope(&tab_cos_d, &tab_cos_d).unwrap(),
        );

        // A mask grown to the head count, and the select that applies it.
        let (m_c, m_d) = mk(&[1, 1, 3, 4], DType::F32);
        same(
            "broadcast_as",
            &m_c.broadcast_as(vec![1usize, 2, 3, 4]).unwrap(),
            &m_d.broadcast_as(vec![1usize, 2, 3, 4]).unwrap(),
        );
        let cond_c = Tensor::zeros(vec![2usize, 3, 4], DType::U8).unwrap();
        let cond_d = Tensor::dry(&dry, DType::U8, vec![2usize, 3, 4]).unwrap();
        same(
            "where_cond",
            &cond_c.where_cond(&a_c, &s_c).unwrap(),
            &cond_d.where_cond(&a_d, &s_d).unwrap(),
        );

        // The expert gather.
        let gi_c = Tensor::zeros(vec![2usize, 1, 4], DType::U32).unwrap();
        let gi_d = Tensor::dry(&dry, DType::U32, vec![2usize, 1, 4]).unwrap();
        same(
            "gather",
            &a_c.gather(&gi_c, 1).unwrap(),
            &a_d.gather(&gi_d, 1).unwrap(),
        );
        same(
            "narrow+contiguous",
            &a_c.narrow(1, 1, 2).unwrap().contiguous().unwrap(),
            &a_d.narrow(1, 1, 2).unwrap().contiguous().unwrap(),
        );
    }

    /// A counted tensor costs exactly its shape times its dtype - and this
    /// substrate's transpose is a COPY, not a view, which the count reflects and no
    /// formula beside the model ever did.
    ///
    /// The attention swaps q, k and v to head-major on every block. Three copies of
    /// the projections per block, per step, that the analytic reserve has to be told
    /// about by hand - and was, eventually, under the name "casts", with a
    /// multiplier chosen to match. Here it is simply what the code does.
    #[test]
    fn a_reshape_costs_nothing_and_a_transpose_costs_a_copy() {
        let dry = Device::dry();
        let led = dry.dry_ledger().unwrap().clone();
        let one = 8 * 16 * 4;
        let x = Tensor::dry(&dry, DType::F32, (8, 16)).unwrap();
        assert_eq!(led.live_bytes(), one);
        let r = x.reshape((16, 8)).unwrap();
        assert_eq!(led.live_bytes(), one, "a reshape allocated something");
        let t = x.transpose(0, 1).unwrap();
        assert_eq!(
            led.live_bytes(),
            2 * one,
            "the transpose copy was not charged"
        );
        drop(r);
        drop(t);
        drop(x);
        assert_eq!(
            led.live_bytes(),
            0,
            "the forward's tensors did not give their room back"
        );
        assert_eq!(led.peak_bytes(), 2 * one);
    }

    /// The peak of a chain is what is HELD at once, not the sum of the chain.
    ///
    /// Ten casts in a row, each dropping the previous, cost one cast - and the
    /// analytic reserves this replaces have no way to know that. Over a denoiser's
    /// forty blocks the difference between the sum and the peak is the difference
    /// between a model that fits a card and one whose blocks go to the host.
    #[test]
    fn a_chain_of_temporaries_costs_one_temporary() {
        let dry = Device::dry();
        let led = dry.dry_ledger().unwrap().clone();
        let x = Tensor::dry(&dry, DType::F32, (256, 256)).unwrap();
        let one = 256 * 256 * 4;
        let mut t = x.clone();
        for _ in 0..10 {
            t = t.silu().unwrap();
        }
        // The input, the tensor in hand, and the one being produced from it.
        assert!(
            led.peak_bytes() <= 3 * one,
            "peak {} is a sum of the chain, not its high-water mark",
            led.peak_bytes()
        );
        assert!(led.peak_bytes() >= 2 * one);
    }

    /// The two device predicates differ on the counting device and NOWHERE else.
    ///
    /// This is what makes a branch safe to move between them. `runs_as_card` asks which
    /// ARRANGEMENT to run; `is_cuda` asks whether a card is about to be touched. On a
    /// card and on the host they agree, so a site migrated from one to the other
    /// resolves identically everywhere a model actually runs, and changes only what a
    /// counted run walks into.
    ///
    /// The card arm is not opened here - doing so would latch a context for the life of
    /// the process, which a unit test has no business doing - and it does not need to
    /// be: both predicates answer a bare `true` for it, from the same shape of match,
    /// which is the one case a reader can settle without running anything.
    #[test]
    fn the_two_device_predicates_agree_everywhere_but_the_ledger() {
        let host = Device::Cpu;
        assert_eq!(
            host.runs_as_card(),
            host.is_cuda(),
            "the predicates disagree on the host, so a migrated branch would move there too"
        );
        assert!(!host.runs_as_card());
        let counted = Device::dry();
        assert!(
            counted.runs_as_card(),
            "a counted run must take the arrangement a card gets, or it weighs another model"
        );
        assert!(
            !counted.is_cuda(),
            "a counted run must never look like hardware, or an upload would be attempted"
        );
    }

    /// A write into a buffer that is already counted is a write, not an allocation.
    ///
    /// The key-value cache is filled this way, block after block into room that was
    /// taken once - so a dry run that could not perform the write could not reach the
    /// second token of any language model, and the whole family stayed unmeasurable
    /// while the mechanism looked complete.
    #[test]
    fn an_in_place_write_into_a_counted_buffer_costs_nothing() {
        let dry = Device::dry();
        let led = dry.dry_ledger().unwrap().clone();
        let dst = Tensor::dry(&dry, DType::F32, (2usize, 8usize)).unwrap();
        let src = Tensor::dry(&dry, DType::F32, (2usize, 3usize)).unwrap();
        let held = led.live_bytes();
        dst.slice_set(&src, 1, 2).unwrap();
        assert_eq!(led.live_bytes(), held, "an in-place write allocated");
        // Every check the real write makes still runs: a write past the end of the
        // destination is refused here exactly as it is on a card.
        assert!(
            dst.slice_set(&src, 1, 7).is_err(),
            "a write past the end of the destination was accepted"
        );
        // And two ledgers are two devices.
        let other = Device::dry();
        let far = Tensor::dry(&other, DType::F32, (2usize, 3usize)).unwrap();
        assert!(
            dst.slice_set(&far, 1, 2).is_err(),
            "a write between two ledgers was accepted"
        );
    }

    /// A dry tensor must not escape into a real computation.
    #[test]
    fn a_counted_tensor_refuses_to_be_read() {
        let dry = Device::dry();
        let x = Tensor::dry(&dry, DType::F32, (4, 4)).unwrap();
        assert!(
            x.to_device(&Device::Cpu).is_err(),
            "a dry tensor moved to the host"
        );
    }
}
