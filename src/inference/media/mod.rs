//! Reading and writing the container formats a request arrives in or leaves as.
//!
//! `crate::inference::media::<part>`

#[cfg(feature = "audio")]
pub mod audio_io;
#[cfg(feature = "image")]
pub mod gif;
pub mod image_processor;
#[cfg(feature = "midi")]
pub mod midi;
