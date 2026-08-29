//! One file per quantisation format: layout, dequantiser and quantiser together.
//!
//! A format's block layout, its dequantiser and its quantiser are one thing: change the
//! layout and the other two are wrong. Keeping them in one file puts that consequence in the
//! diff instead of three files away.

use super::*;

pub mod mxfp4;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod q8_1;
pub mod q8_k;

pub use mxfp4::*;
pub use q2_k::*;
pub use q3_k::*;
pub use q4_0::*;
pub use q4_1::*;
pub use q4_k::*;
pub use q5_0::*;
pub use q5_1::*;
pub use q5_k::*;
pub use q6_k::*;
pub use q8_0::*;
pub use q8_1::*;
pub use q8_k::*;
