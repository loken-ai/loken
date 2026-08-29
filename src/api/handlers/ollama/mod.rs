//! Ollama-compatible /api/* handlers: tags/pull/delete/show/create/push/
//! copy/blobs/ps/version, chat + generate, and model load/unload endpoints.

use super::*;

mod models;
pub use models::*;
mod options;
pub use options::*;
mod chat;
mod load;
pub use chat::*;
mod embed;
pub use embed::*;

#[cfg(test)]
mod tests;
