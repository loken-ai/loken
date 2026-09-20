//! Weights read in place in the formats a released checkpoint ships: e4m3 fp8 with a scale per
//! tile, e2m1 fp4 with a scale per block of inputs, and bf16. Each is a `[out, in]` weight whose
//! product with an activation runs straight from the mapping.

pub mod bf16;
pub mod dequant;
pub mod fp4;
pub mod fp8;
