//! Serving a request: the scheduler, continuous batching, cancellation,
//! speculative decoding and the tensor-parallel decode path.
//!
//! Callers name a module through this directory - `crate::inference::serve::<module>` - so the path says which
//! part of the system a file belongs to, which is the whole reason the directory exists.

pub mod batched_forward;
pub mod cancel;
pub mod continuous_batch;
pub mod continuous_serve;
pub mod eagle;
pub mod pipeline;
pub mod progress;
pub mod prompt_lookup;
pub mod scheduler;
pub mod spec_kv_cache;
pub mod speculative_config;
#[cfg(feature = "cuda")]
pub mod tp_decode;
#[cfg(feature = "cuda")]
pub mod tp_model;
