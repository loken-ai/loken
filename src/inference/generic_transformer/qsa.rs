//! Block-sparse attention: a small indexer scores blocks of cached tokens and the attention of
//! the layer reads only the best of them, plus the tokens of the newest, incomplete block.
//!
//! The indexer keeps one key row per token. A complete block of `block` tokens is scored by one
//! pooled key, the mean of its rows after normalisation and rotation at the block's first
//! position; a query keeps the `budget_blocks` best blocks. A context of at most `top_k` tokens
//! keeps every block, so the attention there is dense.

use super::config::{meta_u32, meta_u32_array};
use super::gguf_file;

/// The indexer's dimensions, from the file's `attention.indexer.*` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerMeta {
    /// Query heads of the indexer; the key is shared by all of them.
    pub heads: usize,
    /// Width of an indexer head.
    pub head_dim: usize,
    /// Tokens an attention layer may read beyond the incomplete block.
    pub top_k: usize,
    /// Tokens pooled into one scored block.
    pub block: usize,
}

impl IndexerMeta {
    /// The metadata of `arch`; `None` when the file has no indexer or its block size does not
    /// divide the token budget.
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> Option<Self> {
        let heads = meta_u32(content, arch, "attention.indexer.head_count")? as usize;
        let head_dim = meta_u32(content, arch, "attention.indexer.key_length")? as usize;
        let top_k = meta_u32(content, arch, "attention.indexer.top_k")? as usize;
        let block = meta_u32_array(content, arch, "attention.compress_ratios")?
            .into_iter()
            .find(|&r| r > 0)? as usize;
        (heads > 0 && head_dim > 0 && block > 1 && top_k.is_multiple_of(block) && top_k >= block)
            .then_some(Self {
                heads,
                head_dim,
                top_k,
                block,
            })
    }

    /// Blocks a query keeps.
    pub fn budget_blocks(&self) -> usize {
        self.top_k / self.block
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{Content, Value, VersionedMagic};
    use std::collections::HashMap;

    fn content(top_k: u32, ratios: &[u32]) -> Content {
        let mut metadata = HashMap::new();
        for (k, v) in [
            ("attention.indexer.head_count", 4),
            ("attention.indexer.key_length", 128),
            ("attention.indexer.top_k", top_k),
        ] {
            metadata.insert(format!("a.{k}"), Value::U32(v));
        }
        metadata.insert(
            "a.attention.compress_ratios".to_string(),
            Value::Array(ratios.iter().map(|&r| Value::I32(r as i32)).collect()),
        );
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    #[test]
    fn the_block_is_the_first_nonzero_ratio() {
        let m = IndexerMeta::from_gguf(&content(2048, &[0, 0, 0, 4, 0, 4]), "a").unwrap();
        assert_eq!((m.heads, m.head_dim, m.top_k, m.block), (4, 128, 2048, 4));
        assert_eq!(m.budget_blocks(), 512);
    }

    #[test]
    fn a_block_that_does_not_divide_the_budget_is_refused() {
        assert_eq!(IndexerMeta::from_gguf(&content(2050, &[0, 4]), "a"), None);
        assert_eq!(IndexerMeta::from_gguf(&content(2048, &[0, 0]), "a"), None);
        assert_eq!(IndexerMeta::from_gguf(&content(2048, &[0, 1]), "a"), None);
    }
}
