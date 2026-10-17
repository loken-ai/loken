//! A layer's quantised KV cache, whichever precision it stores.
//!
//! The lifecycle (length, trim, reset, device position, graph state) is the same for every
//! precision and is answered here once; the attention kernels differ, so a caller that needs
//! one reaches the precision through `q8()` / `q4()`.

use super::q4_kv::Q4KvCache;
use super::q8_kv::Q8KvCache;
use crate::tensor::cuda_ext::CudaSlice;
use crate::tensor::{DType, Tensor};
use anyhow::Result;

pub enum QuantKv {
    Q8(Q8KvCache),
    Q4(Q4KvCache),
}

impl QuantKv {
    pub fn q8(&self) -> Option<&Q8KvCache> {
        match self {
            QuantKv::Q8(c) => Some(c),
            QuantKv::Q4(_) => None,
        }
    }

    pub fn q8_mut(&mut self) -> Option<&mut Q8KvCache> {
        match self {
            QuantKv::Q8(c) => Some(c),
            QuantKv::Q4(_) => None,
        }
    }

    pub fn q4(&self) -> Option<&Q4KvCache> {
        match self {
            QuantKv::Q4(c) => Some(c),
            QuantKv::Q8(_) => None,
        }
    }

    pub fn q4_mut(&mut self) -> Option<&mut Q4KvCache> {
        match self {
            QuantKv::Q4(c) => Some(c),
            QuantKv::Q8(_) => None,
        }
    }

    pub fn current_seq_len(&self) -> usize {
        match self {
            QuantKv::Q8(c) => c.current_seq_len(),
            QuantKv::Q4(c) => c.current_seq_len(),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        match self {
            QuantKv::Q8(c) => c.append(k, v),
            QuantKv::Q4(c) => c.append(k, v),
        }
    }

    /// The whole history as `[1, n_kv, len, head_dim]` K and V in `dtype`.
    pub fn dequantize_kv(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        match self {
            QuantKv::Q8(c) => c.dequantize_kv(dtype),
            QuantKv::Q4(c) => c.dequantize_kv(dtype),
        }
    }

    /// Shortens to `new_len`. A trim the precision cannot honour exactly empties the cache,
    /// which is always correct and only loses the reuse.
    pub fn trim_to(&mut self, new_len: usize) {
        match self {
            QuantKv::Q8(c) => c.trim_to(new_len),
            QuantKv::Q4(c) => {
                if let Err(e) = c.trim_to(new_len) {
                    tracing::warn!("Q4 KV trim_to({new_len}) failed: {e}; resetting cache");
                    c.reset();
                }
            }
        }
    }

    pub fn reset(&mut self) {
        match self {
            QuantKv::Q8(c) => c.reset(),
            QuantKv::Q4(c) => c.reset(),
        }
    }

    pub fn update_graph_state(&mut self, pos: usize) -> Result<()> {
        match self {
            QuantKv::Q8(c) => c.update_graph_state(pos),
            QuantKv::Q4(c) => c.update_graph_state(pos),
        }
    }

    pub fn cur_pos_dev_ptr(&self) -> Option<&CudaSlice<i32>> {
        match self {
            QuantKv::Q8(c) => c.cur_pos_dev_ptr(),
            QuantKv::Q4(c) => c.cur_pos_dev_ptr(),
        }
    }

    pub fn mark_graph_captured(&mut self) {
        match self {
            QuantKv::Q8(c) => c.mark_graph_captured(),
            QuantKv::Q4(c) => c.mark_graph_captured(),
        }
    }

    pub fn clear_graph_captured(&mut self) {
        match self {
            QuantKv::Q8(c) => c.clear_graph_captured(),
            QuantKv::Q4(c) => c.clear_graph_captured(),
        }
    }

    pub fn resume_graph_captured(&mut self) {
        match self {
            QuantKv::Q8(c) => c.resume_graph_captured(),
            QuantKv::Q4(c) => c.resume_graph_captured(),
        }
    }

    pub fn suspend_graph_captured(&mut self) {
        match self {
            QuantKv::Q8(c) => c.suspend_graph_captured(),
            QuantKv::Q4(c) => c.suspend_graph_captured(),
        }
    }

    /// Times the buffers were made anew: a graph recorded against them is stale once this moves.
    pub fn buffer_moves(&self) -> u64 {
        match self {
            QuantKv::Q8(c) => c.buffer_moves(),
            QuantKv::Q4(c) => c.buffer_moves(),
        }
    }

    pub fn graph_seq_limit(&self) -> Option<usize> {
        match self {
            QuantKv::Q8(c) => c.graph_seq_limit(),
            QuantKv::Q4(c) => c.graph_seq_limit(),
        }
    }

    pub fn set_seq_len_for_graph(&mut self, len: usize) {
        match self {
            QuantKv::Q8(c) => c.set_seq_len_for_graph(len),
            QuantKv::Q4(c) => c.set_seq_len_for_graph(len),
        }
    }

    pub fn max_seq_padded(&self) -> usize {
        match self {
            QuantKv::Q8(c) => c.max_seq_padded(),
            QuantKv::Q4(c) => c.max_seq_padded(),
        }
    }
}
