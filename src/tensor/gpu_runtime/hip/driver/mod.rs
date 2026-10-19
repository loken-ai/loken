//! The driver: raw calls under their CUDA names, their `Result` forms, and the owning wrappers.

mod capture_hold;
pub mod result;
mod safe;
pub mod sys;
mod unpinnable;

pub use result::DriverError;
pub use safe::*;
