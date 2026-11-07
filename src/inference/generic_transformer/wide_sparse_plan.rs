//! What a Sinkhorn wide-residual file asks of a card.
//!
//! Only the token mixers of a layer run on a card.
//! The residual mixing, the feed-forward, the routed experts, the engram tables and the head stay
//! on the host, read in place from the mapping.
//! The plan therefore counts the mixers' bytes as the blocks, and the largest mixer projection
//! as the staging of a load.

use super::wide_sparse_config::WideSparseMeta;
use crate::tensor::quantized::gguf_file;

/// Mixer tensors a card holds in the packed form the file stores, by their stem.
const PROJECTIONS: [&str; 6] = [
    "attn_q_a",
    "attn_q_b",
    "attn_kv",
    "attn_output_a",
    "indexer.attn_q_b",
    "attn_o_a",
];

/// Mixer tensors a card holds as F32 values, by their stem.
const DENSE: [&str; 13] = [
    "attn_q_a_norm",
    "attn_kv_a_norm",
    "attn_kv_norm",
    "attn_sinks",
    "attn_sink",
    "attn_compressor_kv",
    "attn_compressor_gate",
    "attn_compressor_norm",
    "indexer.proj",
    "indexer.attn_k",
    "indexer.k_norm",
    "attn_indexer_weights",
    "attn_indexer_k",
];

/// The bytes a wide-residual file puts on a card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardMixerBytes {
    /// Bytes each layer's token mixer holds on its card.
    pub per_layer: Vec<u64>,
    /// Bytes of the largest single tensor a load uploads.
    pub largest_upload: u64,
}

impl CardMixerBytes {
    /// Bytes of every layer's mixer together.
    pub fn total(&self) -> u64 {
        self.per_layer.iter().sum()
    }
}

/// The stem of a tensor name of block `il`, without the block prefix and the `.weight` suffix.
fn stem(name: &str) -> Option<(usize, &str)> {
    let (il, tail) = name.strip_prefix("blk.")?.split_once('.')?;
    Some((
        il.parse().ok()?,
        tail.strip_suffix(".weight").unwrap_or(tail),
    ))
}

/// What the cards hold of `content` when it describes a wide residual, else `None`.
///
/// A projection counts the bytes the file stores, a dense weight counts 4 bytes a value.
pub fn card_mixer_bytes(content: &gguf_file::Content) -> Option<CardMixerBytes> {
    if !WideSparseMeta::claims(content) {
        return None;
    }
    let layers = WideSparseMeta::from_file(content).ok()?.n_layers;
    let mut per_layer = vec![0u64; layers];
    let mut largest_upload = 0u64;
    for (name, info) in &content.tensor_infos {
        let Some((il, stem)) = stem(name) else {
            continue;
        };
        let Some(slot) = per_layer.get_mut(il) else {
            continue;
        };
        let bytes = if PROJECTIONS.contains(&stem) {
            info.size_in_bytes() as u64
        } else if DENSE.contains(&stem) {
            (info.elem_count() * std::mem::size_of::<f32>()) as u64
        } else {
            continue;
        };
        *slot += bytes;
        largest_upload = largest_upload.max(bytes);
    }
    Some(CardMixerBytes {
        per_layer,
        largest_upload,
    })
}
