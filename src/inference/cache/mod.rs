//! Everything that remembers between tokens: the KV caches (paged, quantised,
//! host-side) and the checkpoint cache.
//!
//! Callers name a module through this directory - `crate::inference::cache::<module>` - so the path says which
//! part of the system a file belongs to, which is the whole reason the directory exists.

/// The window a quantised KV cache is built for, and the window the placement budget
/// plans against - one number, because those two must not drift apart.
///
/// A growing cache starts here and doubles on the first append past it, up to the context
/// the configuration allows. The budget plans for THIS rather than for the model's full GGUF
/// context, and that is deliberate: planning the worst case reserved many gigabytes the cache
/// never allocates, which measured as an unnecessary two-card split on a model that fits one,
/// costing about a fifth of its decode rate.
///
/// It was written out separately in the two caches, in the loader and in the planner. Four
/// copies of one policy is four chances for the allocation and the budget to disagree, and
/// that disagreement is either an out-of-memory or a placement nobody asked for.
pub const KV_WORKING_WINDOW_TOKENS: usize = 4096;

pub mod cpu_f16_kv;
pub mod cpu_q8_kv;
pub mod hf;
pub mod kv_disk;
pub mod paged_attention;
pub mod paged_kv;
#[cfg(feature = "cuda")]
pub mod q4_kv;
#[cfg(feature = "cuda")]
pub mod q8_kv;
pub mod qvb;
/// A 3-bit KV scheme that exists on the host only - no kernel, and no cache type the
/// loader can select. It is here for the measurement in `turboquant::measure`, which is
/// what decides whether the device path gets written at all.
pub mod turboquant;
