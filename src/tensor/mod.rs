//! LOKEN's tensor substrate: the CPU block-quantisation engine, CUDA storage
//! and streams over the vendored cudarc, GGUF and safetensors IO, the layer
//! library, and the kernel boundary (`kernel_ffi`: storage projection plus the
//! CUDA device and launch handles).
//!
//! There is one implementation and one path to it. Every type is spelled one way here: two
//! spellings for the same thing let three pairs of DIFFERENT types share a name unnoticed.

mod cpu_storage;
#[cfg(feature = "cuda")]
pub mod cuda;
mod device;
pub mod dry;
mod dtype;
pub mod gguf_write;
pub mod heap;
pub mod kernel_ffi;
pub mod lora;
pub mod marlin;
pub mod ops;
/// Bit-exact parity of the quantised block formats against ggml's own scalar
/// quantisers, which is the only judge the ported engine cannot grade itself with.
#[cfg(test)]
mod oracle_parity;
pub mod pth;
pub mod quant_cpu;
/// GGUF/quantized tensor types (QTensor/QMatMul/QStorage/GgmlDType/gguf_file/...),
/// re-exported as `crate::tensor::quantized`.
pub mod quantized;
pub mod safetensors_io;
/// A linear projection over a quantised weight - one declaration for every family
/// that needs one, rather than a copy per model file.
mod shape;
mod tensor;

pub use cpu_storage::CpuStorage;
pub use device::DeviceLocation;
pub use device::{Device, Storage};
pub use dtype::DType;
pub use kernel_ffi::StorageView;
pub use ops::traits::{IndexOp, Module, TensorId, TensorIndexer};
pub use shape::{Dim, Shape, D};
pub use tensor::Tensor;

/// Host-bounce profiler:
/// env-gated counters on every correctness-first CPU fallback so the decode
/// path's top offenders can be ranked and given device kernels. Zero
/// overhead when `NATIVE_BOUNCE` is unset; with `NATIVE_BOUNCE=1` each
/// instrumented site records (count, wall-time) keyed by its `#[track_caller]`
/// location, and a summary prints at process exit.
pub mod bounce {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use std::sync::atomic::{AtomicU64, Ordering};

    /// Always-on counter of pressure host-bounces: an op that could not allocate on
    /// its device and ran on the CPU instead. One relaxed increment next to a PCIe
    /// round trip costs nothing, and it turns a SILENT performance cliff into a
    /// number a caller can report - twice now a decode that had quietly fallen to
    /// this path was only found by reading the log by hand.
    static PRESSURE_BOUNCES: AtomicU64 = AtomicU64::new(0);

    /// Record one pressure bounce (called from the op fallbacks, not from the
    /// correctness-driven CPU paths, which are not a degradation).
    pub fn note_pressure_bounce() {
        PRESSURE_BOUNCES.fetch_add(1, Ordering::Relaxed);
    }

    /// Monotonic count since process start; take the difference around a request.
    pub fn pressure_bounces() -> u64 {
        PRESSURE_BOUNCES.load(Ordering::Relaxed)
    }

    fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("NATIVE_BOUNCE").is_ok_and(|v| !v.is_empty() && v != "0"))
    }

    type Stats = Mutex<HashMap<String, (u64, f64)>>;
    fn stats() -> &'static Stats {
        static S: OnceLock<Stats> = OnceLock::new();
        S.get_or_init(|| {
            unsafe { libc::atexit(report_atexit) };
            Mutex::new(HashMap::new())
        })
    }

    extern "C" fn report_atexit() {
        report();
    }

    /// RAII timer for one fallback execution.
    pub struct Bounce {
        tag: String,
        start: Instant,
    }

    /// Start timing a fallback at the caller's source location.
    /// `kind` distinguishes fallback classes sharing a location
    /// (host_bounce / host_map / f16_via_f32 / host_cat / to_host).
    #[track_caller]
    pub fn start(kind: &str) -> Option<Bounce> {
        if !enabled() {
            return None;
        }
        let loc = std::panic::Location::caller();
        Some(Bounce {
            tag: format!("{kind} @ {}:{}", loc.file(), loc.line()),
            start: Instant::now(),
        })
    }

    impl Drop for Bounce {
        fn drop(&mut self) {
            let dt = self.start.elapsed().as_secs_f64();
            let mut g = stats().lock().unwrap_or_else(|e| e.into_inner());
            let e = g.entry(std::mem::take(&mut self.tag)).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += dt;
        }
    }

    /// Print the per-site totals sorted by accumulated wall time.
    pub fn report() {
        if !enabled() {
            return;
        }
        let g = stats().lock().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            eprintln!("[native_bounce] no host bounces recorded");
            return;
        }
        let mut rows: Vec<_> = g.iter().collect();
        rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        let (mut tc, mut tt) = (0u64, 0f64);
        eprintln!("[native_bounce] -- host-fallback profile --");
        for (tag, (count, secs)) in &rows {
            eprintln!("[native_bounce] {secs:>9.4}s {count:>8}x  {tag}");
            tc += count;
            tt += secs;
        }
        eprintln!("[native_bounce] total {tt:.4}s over {tc} bounces");
    }
}

/// Error type for the native substrate. Message-based like the current one  - 
/// op kernels attach context at the call site.
#[derive(Debug)]
pub struct Error(pub String);

impl Error {
    /// True when this error originated from a CUDA out-of-memory condition
    /// (cuMemAlloc / CUDA_ERROR_OUT_OF_MEMORY). Device-allocation failures format
    /// the driver error whose Display always carries the `CUDA_ERROR_OUT_OF_MEMORY`
    /// name and the `out of memory` description, and the alloc sites additionally
    /// tag a stable `[oom]` marker. Lets callers branch on memory pressure
    /// specifically (degrade footprint / fall back to CPU) instead of treating every
    /// error as fatal - and instead of blanket-catching ALL errors as OOM.
    pub fn is_oom(&self) -> bool {
        let s = self.0.to_ascii_lowercase();
        s.contains("[oom]") || s.contains("out of memory") || s.contains("out_of_memory")
            // cuBLAS / cuDNN report OOM as *_STATUS_ALLOC_FAILED (no "out of memory"
            // text) - a handle/workspace alloc that lost VRAM to a concurrent consumer.
            // Treat those as memory pressure so the render degrades instead of crashing.
            || s.contains("alloc_failed") || s.contains("alloc failed")
    }

    /// The one constructor: anything printable becomes the message. Op kernels
    /// attach their context at the call site, so there is nothing to chain.
    pub fn msg<M: std::fmt::Display>(m: M) -> Self {
        Self(m.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(format!("io error: {e}"))
    }
}
#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self(format!("cuda driver error: {e}"))
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! tensor_bail {
    ($($arg:tt)*) => {
        return Err($crate::tensor::Error(format!($($arg)*)))
    };
}

/// The random stream behind `Tensor::randn` / `randn_like`.
///
/// A generation's `seed` is the user's promise that the same request returns the
/// same image. It was not kept: `Device::set_seed` was a no-op and the noise came
/// from an OS-seeded RNG, so a Flux render with a fixed seed produced a different
/// image every time (three 512^2 one-step renders differed by a mean of 15-44/255).
///
/// The stream is PER THREAD. A process-wide one is reproducible only while nothing
/// else draws: two concurrent renders would steal each other's values and neither
/// would replay. Each generation runs its noise draw on the same blocking thread
/// that installs the seed, so a thread-local stream is both reproducible and safe
/// under concurrency. Without a seed installed the thread draws from the OS, so a
/// caller that wants genuine noise still gets it.
pub mod rng {
    use std::cell::Cell;

    thread_local! {
        /// `None` = OS randomness; `Some(state)` = deterministic xorshift64*.
        static STREAM: Cell<Option<u64>> = const { Cell::new(None) };
    }

    /// Make subsequent draws ON THIS THREAD reproducible from `seed`.
    pub fn set_global_seed(seed: u64) {
        // Avoid the xorshift fixed point at 0 and decorrelate adjacent seeds.
        STREAM.with(|s| s.set(Some(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)));
    }

    /// Return this thread to OS randomness.
    pub fn clear_global_seed() {
        STREAM.with(|s| s.set(None));
    }

    /// Uniform in [0, 1).
    pub fn next_f32() -> f32 {
        STREAM.with(|s| match s.get() {
            None => {
                use rand::RngExt;
                rand::rng().random::<f32>()
            }
            Some(mut x) => {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                s.set(Some(x));
                (x >> 40) as f32 / (1u64 << 24) as f32
            }
        })
    }
}

#[cfg(test)]
mod rng_tests {
    use super::rng;

    /// Each test owns its thread's stream, so these run in parallel without
    /// stealing each other's draws - the failure mode a process-wide stream had.
    #[test]
    fn the_same_seed_replays_the_same_stream() {
        rng::set_global_seed(7);
        let a: Vec<f32> = (0..64).map(|_| rng::next_f32()).collect();
        rng::set_global_seed(7);
        let b: Vec<f32> = (0..64).map(|_| rng::next_f32()).collect();
        assert_eq!(
            a, b,
            "a fixed seed must replay exactly - this is the promise `seed` makes"
        );
        rng::clear_global_seed();
    }

    #[test]
    fn different_seeds_give_different_streams() {
        rng::set_global_seed(7);
        let a: Vec<f32> = (0..64).map(|_| rng::next_f32()).collect();
        rng::set_global_seed(8);
        let b: Vec<f32> = (0..64).map(|_| rng::next_f32()).collect();
        assert_ne!(a, b);
        rng::clear_global_seed();
    }

    #[test]
    fn draws_stay_in_the_unit_interval() {
        rng::set_global_seed(12345);
        for _ in 0..10_000 {
            let x = rng::next_f32();
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
        }
        rng::clear_global_seed();
    }

    #[test]
    fn the_stream_is_not_degenerate() {
        rng::set_global_seed(1);
        let v: Vec<f32> = (0..4096).map(|_| rng::next_f32()).collect();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        // A uniform stream averages 0.5; a stuck or trivially-correlated one would not.
        assert!((mean - 0.5).abs() < 0.03, "mean {mean} is not uniform");
        let distinct = v
            .iter()
            .map(|x| x.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct > 4000,
            "only {distinct} distinct values in 4096 draws"
        );
        rng::clear_global_seed();
    }
}

/// The CUDA boundary: substrate-neutral slice/stream/event/graph-capture
/// helpers so inference files get raw-CUDA access from one namespace.
/// The layer library: what a model file loads its weights into.
pub mod layer;

/// Key/value caches: what a decode step appends to and reads back.
pub mod kv_cache;

mod varbuilder;

/// The dot-joined tensor path both weight loaders walk.
pub(crate) mod prefix;

pub use kv_cache::{Cache, ConcatKvCache, KvCache};
pub use varbuilder::VarBuilder;

pub mod cuda_ext;

/// Zero-copy quantized CPU views: QTensors sharing a parent's block
/// data (per-expert slices) or an mmap'd GGUF instead of heap copies.
pub mod quant_view;

pub use quantized::gguf_file;
pub use safetensors_io as safetensors;

/// `bail!` - `tensor_bail` (expands to a `tensor::Error`).
pub use crate::tensor_bail as bail;
