//! The path a weight is addressed by.
//!
//! A checkpoint names its tensors with dot-separated segments, and a loader walks that
//! naming one segment at a time - `attn`, then `q_proj`, then `weight`. The joining rule
//! belongs to the NAMES, not to what is stored under them, so the dense and the quantized
//! loader share it: a tensor name is the checkpoint's specification, and there is exactly
//! one way to spell it.

/// A dot-joined tensor path, empty at the root.
#[derive(Clone, Debug)]
pub struct Prefix(String);

impl Prefix {
    /// The root of a checkpoint: nothing to prepend, so a leaf name is used bare.
    pub fn root() -> Self {
        Self(String::new())
    }

    /// The full name of a leaf under this prefix. At the root that is the leaf name
    /// itself - never a leading dot, which no checkpoint writes.
    pub fn path(&self, name: &str) -> String {
        if self.0.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.0, name)
        }
    }

    /// Descend one segment. Naming a submodule and naming a tensor are the same
    /// concatenation, so this is [`Prefix::path`] under another name.
    pub fn join<S: ToString>(&self, segment: S) -> Self {
        Self(self.path(&segment.to_string()))
    }

    /// This prefix as the checkpoint spells it, e.g. `down_blocks.1.attentions.0`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
