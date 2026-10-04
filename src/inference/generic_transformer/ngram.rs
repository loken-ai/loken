//! A hashed n-gram embedding: each token gathers one row per head from one large table, the
//! row of a head chosen by hashing the token with its predecessors, and the gathered rows are
//! gated into the residual of one layer.
//!
//! The table is only ever read by row, so it stays in the mapping and a token touches
//! `heads * row_bytes` of it.

use super::config::{meta_u32, meta_u32_array, meta_u64_array};
use super::gguf_file;

/// The n-gram embedding's parameters, from the file's `ple.*` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramMeta {
    /// The layer the embedding joins the residual before.
    pub layer: usize,
    /// Longest n-gram hashed; the hashes cover lengths 2 up to this.
    pub ngram_size: usize,
    /// Hash heads of each n-gram length.
    pub heads_per_ngram: usize,
    /// Taps of the dilated convolution; its dilation is `ngram_size`.
    pub conv_kernel: usize,
    /// Token that cuts the n-gram window.
    pub eos_token: u32,
    /// Width of the row one head gathers.
    pub row_dim: usize,
    /// One odd multiplier per position of the window.
    pub multipliers: Vec<u64>,
    /// First table row of each head.
    pub head_offsets: Vec<u64>,
    /// Rows of each head's range.
    pub head_vocab: Vec<u64>,
}

impl NgramMeta {
    /// Heads in all: `ngram_size - 1` lengths of `heads_per_ngram` each.
    pub fn heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// The metadata of `arch`; `None` when the file declares no n-gram layer or the arrays are
    /// shorter than the heads and window need.
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> Option<Self> {
        let layers = meta_u32_array(content, arch, "ple.layers")?;
        let &[layer] = layers.as_slice() else {
            return None;
        };
        let m = Self {
            layer: layer as usize,
            ngram_size: meta_u32(content, arch, "ple.ngram_size")? as usize,
            heads_per_ngram: meta_u32(content, arch, "ple.heads_per_ngram")? as usize,
            conv_kernel: meta_u32(content, arch, "ple.conv_kernel")? as usize,
            eos_token: meta_u32(content, arch, "ple.eos_token_id")?,
            row_dim: meta_u32(content, arch, "embedding_length_per_layer_input")? as usize,
            multipliers: meta_u64_array(content, arch, "ple.layer_multipliers")?,
            head_offsets: meta_u64_array(content, arch, "ple.head_offsets")?,
            head_vocab: meta_u64_array(content, arch, "ple.head_vocab_sizes")?,
        };
        let ok = m.ngram_size >= 2
            && m.heads_per_ngram > 0
            && m.conv_kernel > 0
            && m.row_dim > 0
            && m.multipliers.len() >= m.ngram_size
            && m.head_offsets.len() >= m.heads()
            && m.head_vocab.len() >= m.heads()
            && m.head_vocab[..m.heads()].iter().all(|&v| v > 0);
        ok.then_some(m)
    }

    /// Rows the heads' ranges cover: the least the table must hold.
    pub fn rows(&self) -> u64 {
        (0..self.heads())
            .map(|h| self.head_offsets[h] + self.head_vocab[h])
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{Content, Value, VersionedMagic};
    use std::collections::HashMap;

    fn content(heads: usize) -> Content {
        let mut md = HashMap::new();
        let mut put = |k: &str, v: Value| {
            md.insert(format!("a.{k}"), v);
        };
        put("ple.layers", Value::Array(vec![Value::U32(1)]));
        put("ple.ngram_size", Value::U32(3));
        put("ple.heads_per_ngram", Value::U32(2));
        put("ple.conv_kernel", Value::U32(4));
        put("ple.eos_token_id", Value::U32(7));
        put("embedding_length_per_layer_input", Value::U32(8));
        let arr = |v: Vec<u64>| Value::Array(v.into_iter().map(Value::U64).collect());
        put("ple.layer_multipliers", arr(vec![3, 5, 7]));
        put(
            "ple.head_offsets",
            arr((0..heads as u64).map(|h| h * 100).collect()),
        );
        put("ple.head_vocab_sizes", arr(vec![101; heads]));
        Content {
            magic: VersionedMagic::GgufV3,
            metadata: md,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    #[test]
    fn the_heads_and_rows_follow_the_arrays() {
        let m = NgramMeta::from_gguf(&content(4), "a").unwrap();
        assert_eq!((m.layer, m.heads(), m.rows()), (1, 4, 401));
    }

    #[test]
    fn arrays_shorter_than_the_heads_are_refused() {
        assert_eq!(NgramMeta::from_gguf(&content(3), "a"), None);
    }
}
