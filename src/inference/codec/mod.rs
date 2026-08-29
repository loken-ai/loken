//! Audio analysis and synthesis front-ends: STFT, mel filterbanks, the DAC codec.
//! Shared by every audio family rather than owned by one.
//!
//! `crate::inference::codec::<part>`

#[cfg(feature = "audio")]
pub mod dac;
#[cfg(feature = "audio")]
pub mod melband;
#[cfg(feature = "audio")]
pub mod stft;
