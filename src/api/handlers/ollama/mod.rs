//! Ollama-compatible /api/* handlers: tags/pull/delete/show/create/push/
//! copy/blobs/ps/version, chat + generate, and model load/unload endpoints.

use super::*;

mod models;
pub(crate) use models::*;
mod options;
pub(crate) use options::*;
mod chat;
mod load;
pub(crate) use chat::*;
mod embed;
pub(crate) use embed::*;

#[cfg(test)]
mod tests;
