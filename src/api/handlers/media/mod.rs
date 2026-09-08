//! Image + video API family: /v1/images generations/edits/variations,
//! /v1/videos, and their validation + response helpers.

use super::*;
use crate::inference::place::model_request::ModelRequest;

mod discover;
pub use discover::*;
mod family;
pub use family::*;
mod generate;
pub(crate) use generate::*;
mod request;
pub(crate) use request::*;
mod video;
pub(crate) use video::*;
mod respond;
pub(crate) use respond::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod header_discovery_tests;
