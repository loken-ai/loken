//! The wide residual: the stream a layer reads and writes is `streams` parallel copies of the
//! hidden state, mixed down to one input by a data-dependent gate before each sublayer and
//! written back with a per-stream weight.
//!
//! A block of such a model carries no norm of its own: the grouped RMS norm inside the mixer
//! is its input norm, and the final mixer is the output norm.

use super::config::meta_u32;
use super::gguf_file;

/// The wide residual's dimensions, from the file's `hyper_connection.*` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperMeta {
    /// Parallel copies of the hidden state.
    pub streams: usize,
    /// Rank of the mixer's low-rank gate projection.
    pub rank: usize,
}

impl HyperMeta {
    /// The metadata of `arch`, `None` when the file has no wide residual or declares fewer
    /// than two streams, which have nothing to mix.
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> Option<Self> {
        let streams = meta_u32(content, arch, "hyper_connection.count")? as usize;
        let rank = meta_u32(content, arch, "hyper_connection.low_rank")? as usize;
        (streams > 1 && rank > 0).then_some(Self { streams, rank })
    }

    /// Width of the residual stream for a hidden state of `embedding_length`.
    pub fn width(&self, embedding_length: usize) -> usize {
        self.streams * embedding_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{Content, Value, VersionedMagic};
    use std::collections::HashMap;

    fn content(meta: &[(&str, u32)]) -> Content {
        let mut metadata = HashMap::new();
        for (k, v) in meta {
            metadata.insert(format!("a.{k}"), Value::U32(*v));
        }
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    #[test]
    fn a_file_without_streams_has_no_wide_residual() {
        assert_eq!(HyperMeta::from_gguf(&content(&[]), "a"), None);
        let one = content(&[
            ("hyper_connection.count", 1),
            ("hyper_connection.low_rank", 8),
        ]);
        assert_eq!(HyperMeta::from_gguf(&one, "a"), None);
    }

    #[test]
    fn the_streams_and_rank_are_read() {
        let c = content(&[
            ("hyper_connection.count", 4),
            ("hyper_connection.low_rank", 320),
        ]);
        let m = HyperMeta::from_gguf(&c, "a").unwrap();
        assert_eq!((m.streams, m.rank, m.width(2560)), (4, 320, 10240));
    }
}
