//! Image + video API family: /v1/images generations/edits/variations,
//! /v1/videos, and their validation + response helpers.

use super::*;
use crate::inference::place::model_request::ModelRequest;

mod discover;
pub use discover::*;
mod family;
pub use family::*;
mod generate;
pub use generate::*;
mod request;
pub use request::*;
mod video;
pub use video::*;
mod respond;
pub use respond::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod header_discovery_tests;
