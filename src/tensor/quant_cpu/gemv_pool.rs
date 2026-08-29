// Transcribed kernels: the index arithmetic IS the layout, the argument lists are the
// reference's, and the `unsafe fn`s wrap intrinsics whose contract is the intrinsic's.
// Named rather than `clippy::all` so anything else here still gets reported.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    clippy::type_complexity,
    clippy::redundant_closure
)]

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// One published job: workers grab chunk indices with `next` and call the
/// borrowed work closure for every index they win. `done` counts finished
/// chunks; the publisher returns only once `done == n_chunks`, so the
/// closure (which lives on the publisher's stack) is only ever invoked
/// while that frame is alive - a straggler that arrives after completion
/// sees `next >= n_chunks` and never touches the pointer.
pub(super) struct Task {
    pub(super) next: AtomicUsize,
    pub(super) done: AtomicUsize,
    pub(super) n_chunks: usize,
    pub(super) func: *const (dyn Fn(usize) + Sync),
}
// SAFETY: `func` is only dereferenced for grabbed chunk indices, and the
// publisher keeps the referent alive until every grabbed chunk is done
// (see `run`). The atomics serialize the chunk handoff.
unsafe impl Send for Task {}
unsafe impl Sync for Task {}

impl Task {
    fn execute(&self) {
        loop {
            let c = self.next.fetch_add(1, Ordering::Relaxed);
            if c >= self.n_chunks {
                return;
            }
            // SAFETY: chunk grabbed => publisher frame alive (see above).
            unsafe { (*self.func)(c) };
            self.done.fetch_add(1, Ordering::Release);
        }
    }
}

pub(super) struct Shared {
    /// Bumped once per published job; workers spin on it.
    pub(super) generation: AtomicU64,
    /// Raw view of the current job. Workers load it lock-free after they see
    /// the generation change (the publisher stores it Release before the
    /// Release generation bump); kept alive by `Pool::retired` for two
    /// generations so a straggler's post-completion `next` access is safe.
    pub(super) task_ptr: AtomicPtr<Task>,
    /// Count of workers currently parked. The publisher reads it to skip the
    /// wake mutex entirely while every worker is spinning (the decode case).
    pub(super) parked_count: AtomicUsize,
    /// Count of workers currently inside the `task_ptr` load + `execute()`
    /// window (i.e. potentially holding a raw `Task` pointer). The publisher
    /// drains this to zero before letting a task fall out of the retired ring
    /// and be dropped - otherwise a worker descheduled past two publishes
    /// (common under decode's rapid back-to-back matmuls + SMT oversubscription)
    /// could dereference a freed `Task`, a use-after-free SIGSEGV. Bumped
    /// BEFORE the `task_ptr` load so the publisher's zero-wait cannot race
    /// ahead of a worker that is about to grab the pointer.
    pub(super) in_execute: AtomicUsize,
    /// Condvar companion mutex - taken only on the park/wake slow path.
    pub(super) park_mx: Mutex<()>,
    pub(super) wake: Condvar,
    /// Serializes publishers; competing callers run their job inline.
    pub(super) publish: Mutex<()>,
}

pub struct Pool {
    pub(super) shared: Arc<Shared>,
    pub threads: usize,
    /// Epoch reclamation: the last two published tasks. A worker may touch
    /// `task.next` for one atomic op AFTER bumping `done` (which releases the
    /// publisher's straggler wait), so the task must outlive that window.
    /// Keeping two generations alive covers it with margin. Only the
    /// lock-winning publisher touches this (guarded by `publish`), so the
    /// mutex is uncontended.
    pub(super) retired: Mutex<[Option<Arc<Task>>; 2]>,
}

/// Spin iterations before a worker parks. Short: it bridges back-to-back
/// matmuls, while a long spin steals execution resources from the
/// publisher's serial sections (SMT siblings) and from other thread
/// pools' regions (rayon's attention GEMM / MoE expert GEMVs) - both measured
/// net losses. Re-tuned 1024->512: parking sooner cuts the cross-pool contention
/// with rayon (attention `gemm_f16w`) - net win on both GEMM-bound (granite
/// 1.35->1.29x vs ollama) and small (qwen3:0.6b 1.61->1.59x) prefill; 256 parked
/// too eagerly and regressed the small path.
const SPIN_ITERS: u32 = 1 << 9;

thread_local! {
    static IS_POOL_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn worker_loop(shared: Arc<Shared>) {
    IS_POOL_WORKER.with(|f| f.set(true));
    let mut seen = 0u64;
    loop {
        // Spin first, then park.
        let mut spins = 0u32;
        loop {
            let g = shared.generation.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            spins += 1;
            if spins < SPIN_ITERS {
                std::hint::spin_loop();
            } else {
                let guard = shared.park_mx.lock().unwrap_or_else(|e| e.into_inner());
                // SeqCst pairs with the publisher's SeqCst generation bump +
                // parked_count load: their total order guarantees either the
                // publisher sees this `+1` (and notifies under `park_mx`), or
                // we see the new generation below and don't sleep - no lost
                // wakeup despite the publisher's lock-free fast-path skip.
                shared.parked_count.fetch_add(1, Ordering::SeqCst);
                if shared.generation.load(Ordering::SeqCst) == seen {
                    let _g = shared.wake.wait(guard).unwrap();
                } else {
                    drop(guard);
                }
                shared.parked_count.fetch_sub(1, Ordering::SeqCst);
                spins = 0;
            }
        }
        // Mark this worker as potentially holding a raw `Task` pointer BEFORE
        // loading it, so a publisher draining `in_execute` to zero before a
        // task is dropped cannot race ahead of us (SeqCst gives the total
        // order between this bump and the publisher's zero-check).
        shared.in_execute.fetch_add(1, Ordering::SeqCst);
        // Lock-free: the publisher stored `task_ptr` (Release) before the
        // Release bump of the generation we just observed (Acquire), so this
        // is that job's task.
        let tp = shared.task_ptr.load(Ordering::Acquire);
        if !tp.is_null() {
            // SAFETY: the publisher will not drop this task (out of the
            // retired ring) until `in_execute` drains to zero - which our
            // `fetch_add` above keeps non-zero across this whole `execute()`
            // - so `tp` is live for the entire dereference, even if we were
            // descheduled past several intervening publishes.
            unsafe { (*tp).execute() };
        }
        shared.in_execute.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Pool {
    fn new() -> Self {
        // One worker per PHYSICAL core. Decode GEMV is DRAM-bandwidth-bound:
        // a second worker on an SMT sibling adds almost no throughput (the
        // core is already stalled on memory) but spins/contends at full
        // package power for the whole token - pure energy waste. Pinning to
        // the physical core count keeps decode tok/s at parity while cutting
        // package energy, and leaves the sibling logical CPUs free to absorb
        // background/runtime threads (so no worker is preempted mid-chunk).
        // Re-tested on the four dense models that lose on CPU: doubling the
        // workers to the logical count moved none of them (-2.8/-4.2/-5.4%
        // against -3.5/-4.0/-6.0%, inside run-to-run spread) while the energy
        // advantage held. Decode is not thread-limited here, so the physical
        // count keeps the energy saving at no throughput cost, and the gap the
        // campaign shows is inside the GEMV rather than in how it is spread.
        Self::new_sized(num_cpus::get_physical().max(1))
    }

    fn new_sized(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            task_ptr: AtomicPtr::new(std::ptr::null_mut()),
            parked_count: AtomicUsize::new(0),
            in_execute: AtomicUsize::new(0),
            park_mx: Mutex::new(()),
            wake: Condvar::new(),
            publish: Mutex::new(()),
        });
        // Workers are left UNPINNED on purpose: with the pool sized to the
        // physical-core count, the scheduler already spreads workers one per
        // core (avoiding SMT siblings while cores are free) and can briefly
        // migrate/idle a worker stalled on memory so its core drops P-state.
        // Hard-pinning one worker per core measured worse package energy at
        // equal tok/s - the cores stay hot. Count is the lever, not affinity.
        for _ in 1..threads {
            let s = shared.clone();
            std::thread::Builder::new()
                .name("qgemv".into())
                .spawn(move || worker_loop(s))
                .expect("spawn qgemv worker");
        }
        Pool {
            shared,
            threads,
            retired: Mutex::new([None, None]),
        }
    }

    /// Run `f(chunk_idx)` for every index in `0..n_chunks` across the
    /// pool. The publisher participates. A call from a pool worker runs
    /// inline serial (no nested publish). A call that loses the publish
    /// lock to another publisher (e.g. the concurrent per-expert GEMVs an
    /// MoE arch fans out across rayon) falls back to rayon so it keeps the
    /// work-stealing parallelism it had before this pool, rather than
    /// collapsing to one thread; only the lock-winning publisher drives
    /// the spin-pool.
    pub fn run(&self, n_chunks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_chunks <= 1 || self.threads == 1 || IS_POOL_WORKER.with(|w| w.get()) {
            for c in 0..n_chunks {
                f(c);
            }
            return;
        }
        // Called from inside a rayon worker (an MoE arch fans its active
        // experts out across rayon), OR while another publisher already
        // holds the spin-pool: keep rayon's work-stealing parallelism
        // instead of driving the spin-pool. The spin-pool's persistent
        // spinners otherwise contend with the concurrent rayon expert
        // GEMVs for cores, whereas a sequential single-publisher decode
        // loop keeps the hot pool to itself.
        if rayon::current_thread_index().is_some() {
            use rayon::prelude::*;
            (0..n_chunks).into_par_iter().for_each(|c| f(c));
            return;
        }
        let Ok(_publish) = self.shared.publish.try_lock() else {
            use rayon::prelude::*;
            (0..n_chunks).into_par_iter().for_each(|c| f(c));
            return;
        };
        let task = Arc::new(Task {
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            n_chunks,
            // Erase the closure's lifetime: `run` blocks until every
            // grabbed chunk completed, and stragglers never dereference.
            func: unsafe {
                std::mem::transmute::<*const (dyn Fn(usize) + Sync), *const (dyn Fn(usize) + Sync)>(
                    f as *const _,
                )
            },
        });
        // Publish lock-free: store the task pointer, then bump the generation
        // (SeqCst - see the worker park path). Workers spinning on the
        // generation pick it up with one atomic load, no mutex / Arc clone.
        self.shared
            .task_ptr
            .store(Arc::as_ptr(&task) as *mut Task, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        // Wake parked workers only if any are parked (the common decode case
        // has all workers spinning -> skip the mutex entirely). SeqCst load
        // pairs with the worker's SeqCst parked_count bump.
        if self.shared.parked_count.load(Ordering::SeqCst) > 0 {
            let _g = self
                .shared
                .park_mx
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            self.shared.wake.notify_all();
        }
        task.execute();
        // Wait for stragglers' in-flight chunks (at most one chunk each);
        // yield if one got descheduled so it can finish.
        let mut spins = 0u32;
        while task.done.load(Ordering::Acquire) < n_chunks {
            spins += 1;
            if spins < SPIN_ITERS {
                std::hint::spin_loop();
            } else {
                spins = 0;
                std::thread::yield_now();
            }
        }
        // Before retiring, drain `in_execute` to zero: no worker may be
        // inside the `task_ptr` load + `execute()` window. The 2-deep ring
        // alone is NOT sufficient - a worker descheduled past two publishes
        // (decode fires hundreds of matmuls back-to-back, and the spin-pool
        // is oversubscribed by the rayon expert pool) would otherwise hold a
        // raw pointer to the task about to be dropped and fault on the next
        // `next.fetch_add`. Waiting for quiescence makes the lifetime exact.
        // Cheap in practice: the `done == n_chunks` wait above already means
        // every worker has finished its chunks and is returning from
        // `execute()`, so this almost always reads zero on the first load.
        {
            let mut spins = 0u32;
            while self.shared.in_execute.load(Ordering::SeqCst) != 0 {
                spins += 1;
                if spins < SPIN_ITERS {
                    std::hint::spin_loop();
                } else {
                    spins = 0;
                    std::thread::yield_now();
                }
            }
        }
        // Epoch reclamation: retire `task` into the 2-deep ring; the task
        // from two generations ago is dropped here. `task_ptr` still points at
        // `task` until the next publish overwrites it, and `task` stays alive
        // in the ring until then - so a worker that observes the current
        // generation always loads a live task.
        {
            let mut ret = self.retired.lock().unwrap_or_else(|e| e.into_inner());
            ret[1] = ret[0].take();
            ret[0] = Some(task);
        }
    }
}

static POOL: OnceLock<Pool> = OnceLock::new();

pub fn pool() -> &'static Pool {
    POOL.get_or_init(Pool::new)
}

static PREFILL_POOL: OnceLock<Pool> = OnceLock::new();

/// Pool sized to LOGICAL cores (SMT siblings included), for the compute-bound
/// tiled prefill GEMMs on the fat K-quant weight matrices. Those hide memory
/// latency behind ALU work, so a second worker per core lifts throughput  - 
/// unlike the bandwidth-bound decode GEMV, which the physical-sized [`pool`]
/// serves (SMT there only burns package power). Prefill and decode run in
/// separate phases, so the idle pool's workers park; only one is hot at a time.
pub fn prefill_pool() -> &'static Pool {
    PREFILL_POOL.get_or_init(|| {
        let logical = num_cpus::get().max(1);
        // Never smaller than the decode pool; if the machine reports no SMT,
        // this is the same size and simply mirrors it.
        Pool::new_sized(logical.max(num_cpus::get_physical().max(1)))
    })
}
