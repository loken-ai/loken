//! The driver over both runtimes: raw calls under their CUDA names, their `Result` forms, and
//! the owning wrappers.

pub mod result;
mod safe;
pub mod sys;

pub use result::DriverError;
pub use safe::*;
