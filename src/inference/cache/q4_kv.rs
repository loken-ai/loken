//! Q4_0 KV cache, KIVI-style.
//!
//! Fixes session 1's accuracy problem by moving K to **per-channel**
//! quantization. Naive Q4_0 of K failed because K has systematic
//! per-channel outliers - a single block spanning 32 head_dim values of
//! one token mixes large-magnitude channels with small ones, so the
//! block scale is dominated by outliers and most values lose resolution.
//!
//! StorageView
//! -------
//! **K**: packed as `[seq_block, n_kv_heads, head_dim]` of Q4_0 blocks.
//! Each 18-byte block covers **32 consecutive seq positions** of one
//! (head, channel), so the scale is calibrated per-channel per-32-token
//! window. This matches KIVI's per-channel K scheme with an additional
//! seq-axis grouping for block-quantize compatibility.
//!
//! The current partial seq-block (up to 31 positions not yet flushed)
//! lives in an F16 residual buffer `[n_kv_heads, head_dim, 32]`. When
//! seq hits a multiple of 32 we quantize the residual to Q4_0 and memcpy
//! it into the main K blob.
//!
//! **V**: unchanged from session 1 - `[seq_tokens, n_kv_heads, head_dim]`
//! with Q4_0 blocks spanning 32 consecutive head_dim values of one
//! (token, head). This is already per-token (with head_dim sub-grouping)
//! and matches KIVI's V quant scheme closely enough.
//!
//! Trim semantics
//! --------------
//! Trimming back into a previously-flushed K block dequantizes that
//! block's content into the residual buffer, so subsequent appends see
//! consistent state. Cost: one block dequant per trim event - small.

use crate::inference::quantized_cuda::dequantize_q4_0_blob_f16;
use crate::tensor::cuda_ext::{
    tensor_from_cuda_storage, CudaSlice, CudaStorage, DevicePtr, RawCudaDevice as CudaDevice,
};
use crate::tensor::quantized::{GgmlDType, QCudaStorage};
use crate::tensor::{DType, Device, IndexOp, StorageView, Tensor};
use anyhow::{anyhow, Result};
use half::f16;

/// A view settled into the one run of numbers a kernel walks, in the width that kernel takes.
///
/// Every device pointer this cache hands out is one run: the kernels are given a raw pointer
/// and their own strides, so the shape a tensor arrives in does not reach them. A decode step
/// comes as `[1, heads, 1, width]` and a bulk append as a slice with its axes permuted so each
/// contiguous 32 values is one block - different shapes, and the run of values in reading
/// order is the same thing in both, which is why one function answers for both.
///
/// The width is the caller's, because the kernel reached for decides it: the query, score and
/// probability kernels work in F32, the residual scatter in F16.
fn settled(t: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(t.contiguous()?.to_dtype(dtype)?.flatten_all()?)
}

/// Values per Q4_0 block, and the bytes one occupies - read from the dtype table rather than
/// restated here.
///
/// A cache that held a second opinion about where a block ends would still run: the sizes
/// would still multiply out, the copies would still land inside the buffers, and every scale
/// written would sit a slot away from the values it scales. There is one declaration of what
/// a Q4_0 block is, and this is a reader of it.
fn q4_0_block_size() -> usize {
    GgmlDType::Q4_0.block_size()
}

fn q4_0_type_size() -> usize {
    GgmlDType::Q4_0.type_size()
}

/// Initial allocated capacity in tokens for lazy growth. See Q8KvCache
/// for rationale - at 4 K we cover the bench + typical chat without
/// reserving worst-case 32 K up front. Past this, `append` grows the
/// k_q4_blocks + v_q4 buffers (doubling, clamped to max_seq_len).
///
/// IMPORTANT graph-mode constraint: once `cur_pos_dev.is_some()` (graph
/// has been primed via update_graph_state), `grow_to` returns an error
/// - the captured graph bakes the buffer base pointer and the
/// max_seq_blocks stride into kernel launches, both of which change on
/// realloc. In practice this only matters for >4 K prompts; the engine
/// would need to invalidate_graph_state() before retrying.
use super::KV_WORKING_WINDOW_TOKENS as INITIAL_CAPACITY_TOKENS;

/// KIVI-style Q4_0 KV cache.
pub struct Q4KvCache {
    /// Packed K blocks `[seq_block, n_kv, head_dim]` - 18 bytes per cell.
    /// Block (sb, h, c) holds the Q4_0 quantisation of K values at
    /// (head=h, channel=c) for seq positions [sb*32, sb*32+32).
    k_q4_blocks: CudaSlice<u8>,
    /// F16 residual for the current partial seq-block. Layout
    /// `[n_kv, head_dim, 32]`. Slots `[0, current_seq_len % 32)` hold
    /// valid K values for positions
    /// `[current_block_base, current_seq_len)`.
    k_residual: CudaSlice<f16>,
    /// Reused staging for the residual->Q4 quantize step.
    k_residual_staging: QCudaStorage,

    /// V storage: packed as `[seq_tokens, n_kv, head_dim]` Q4_0 blocks.
    /// Block (t, h, c_start) holds the quant of V[t, h, c_start..c_start+32].
    v_q4: CudaSlice<u8>,
    v_staging_decode: QCudaStorage,

    current_seq_len: usize,
    /// Currently allocated capacity in tokens, rounded up to a block
    /// boundary by `current_capacity_blocks`. Lazy - grows on append.
    current_capacity_tokens: usize,
    /// Currently allocated number of Q4 blocks for k_q4_blocks. Equal
    /// to `current_capacity_tokens.div_ceil(BLOCK_SIZE)`.
    current_capacity_blocks: usize,
    /// Hard cap on capacity - appends past it error out.
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: Device,
    cuda_device: CudaDevice,

    /// Single-element i32 device tensor holding the current write
    /// position (= current_seq_len). Host-side `update_graph_state(pos)`
    /// writes new values into this slot OUTSIDE the captured CUDA-graph
    /// region; the device-pos variants of the append kernels read it on
    /// every replay. None until the first call to `update_graph_state`
    /// - in non-graph mode this stays None and the host-int kernel
    /// variants are used.
    cur_pos_dev: Option<CudaSlice<i32>>,
}

impl Q4KvCache {
    pub fn new(
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        if !head_dim.is_multiple_of(q4_0_block_size()) {
            return Err(anyhow!(
                "Q4KvCache: head_dim {head_dim} must be a multiple of {}",
                q4_0_block_size()
            ));
        }
        let cuda_device = device
            .as_cuda_device()
            .map_err(|_| anyhow!("Q4KvCache: CUDA device required"))?;

        // Round up to a multiple of 32 so every slot has a home in
        // k_q4_blocks. Initial capacity is INITIAL_CAPACITY_TOKENS
        // (clamped to max_seq_len); `grow_to` doubles it on demand up
        // to max_seq_len.
        let current_capacity_tokens = INITIAL_CAPACITY_TOKENS
            .min(max_seq_len)
            .max(q4_0_block_size());
        let current_capacity_blocks = current_capacity_tokens.div_ceil(q4_0_block_size());

        // K blocks: one 18-byte Q4_0 block per (seq_block, head, channel).
        let k_bytes = current_capacity_blocks * n_kv_heads * head_dim * q4_0_type_size();
        let k_q4_blocks = cuda_device.alloc_zeros::<u8>(k_bytes)?;

        // K residual: F16 staging for the current partial seq-block.
        // Independent of seq_len; fixed at one block's worth.
        let k_residual_elems = n_kv_heads * head_dim * q4_0_block_size();
        let k_residual = cuda_device.alloc_zeros::<f16>(k_residual_elems)?;

        let k_residual_staging =
            QCudaStorage::zeros(&cuda_device, k_residual_elems, GgmlDType::Q4_0)?;

        // V: packed as [seq_tokens, n_kv, head_dim] Q4_0 blocks.
        let v_bytes = current_capacity_tokens
            * n_kv_heads
            * (head_dim / q4_0_block_size())
            * q4_0_type_size();
        let v_q4 = cuda_device.alloc_zeros::<u8>(v_bytes)?;
        let v_decode_elems = n_kv_heads * head_dim;
        let v_staging_decode = QCudaStorage::zeros(&cuda_device, v_decode_elems, GgmlDType::Q4_0)?;

        Ok(Self {
            k_q4_blocks,
            k_residual,
            k_residual_staging,
            v_q4,
            v_staging_decode,
            current_seq_len: 0,
            current_capacity_tokens,
            current_capacity_blocks,
            max_seq_len,
            n_kv_heads,
            head_dim,
            device: device.clone(),
            cuda_device,
            cur_pos_dev: None,
        })
    }

    /// Grow the K/V buffers to hold at least `new_cap` tokens. Doubles
    /// the current capacity (clamped to `max_seq_len`).
    ///
    /// **Fails if a graph has been captured** (`cur_pos_dev.is_some()`).
    /// The captured graph bakes both the buffer base pointer (which
    /// changes on realloc) and `current_capacity_blocks` as the
    /// stride into kernel launches. Re-capturing requires the engine
    /// to call `invalidate_graph_state()` first; we surface the error
    /// up so the engine can take that action.
    fn grow_to(&mut self, new_cap_tokens: usize) -> Result<()> {
        if self.cur_pos_dev.is_some() {
            return Err(anyhow!(
                "Q4KvCache::grow_to: cannot grow after CUDA graph has been captured \
                 (current_capacity={}, requested={}, max_seq_len={}). The captured graph \
                 bakes the buffer pointer + stride into kernel launches; engine must call \
                 invalidate_graph_state() before retrying.",
                self.current_capacity_tokens,
                new_cap_tokens,
                self.max_seq_len
            ));
        }
        let target_tokens = new_cap_tokens
            .max(self.current_capacity_tokens * 2)
            .min(self.max_seq_len);
        if target_tokens <= self.current_capacity_tokens {
            return Ok(());
        }
        // Try the amortized geometric target first; on a GPU OOM, retry with the
        // EXACT requested capacity (`new_cap_tokens`) before propagating - keeps a
        // near-full GPU growing one chunk at a time. A genuine OOM at the minimal
        // size still propagates to the engine's OOM-adaptive prefill (smaller chunk).
        let is_oom = |e: &anyhow::Error| {
            let s = e.to_string().to_ascii_lowercase();
            s.contains("out of memory") || s.contains("cuda_error_out_of_memory")
        };
        let alloc_for = |toks: usize| {
            let blocks = toks.div_ceil(q4_0_block_size());
            let k_bytes = blocks * self.n_kv_heads * self.head_dim * q4_0_type_size();
            let v_bytes =
                toks * self.n_kv_heads * (self.head_dim / q4_0_block_size()) * q4_0_type_size();
            let k = self.cuda_device.alloc_zeros::<u8>(k_bytes)?;
            let v = self.cuda_device.alloc_zeros::<u8>(v_bytes)?;
            Ok::<_, anyhow::Error>((toks, blocks, k, v))
        };
        let (target_tokens, target_blocks, mut new_k, mut new_v) = match alloc_for(target_tokens) {
            Ok(t) => t,
            Err(e) if is_oom(&e) && new_cap_tokens < target_tokens => {
                let minimal = new_cap_tokens
                    .max(self.current_capacity_tokens + 1)
                    .min(self.max_seq_len);
                tracing::warn!(
                    "Q4KvCache::grow_to: doubled target {target_tokens} OOM'd ({e}); retrying minimal capacity {minimal}"
                );
                alloc_for(minimal)?
            }
            Err(e) => return Err(e),
        };

        if self.current_seq_len > 0 {
            // Copy used prefix of K blocks: blocks [0, current_seq_blocks),
            // where current_seq_blocks covers everything actually written
            // (including the currently-in-flight partial block if any).
            let current_seq_blocks = self.current_seq_len.div_ceil(q4_0_block_size());
            let used_k = current_seq_blocks * self.n_kv_heads * self.head_dim * q4_0_type_size();
            let used_v = self.current_seq_len
                * self.n_kv_heads
                * (self.head_dim / q4_0_block_size())
                * q4_0_type_size();
            {
                let src_k = self.k_q4_blocks.slice(0..used_k);
                let mut dst_k = new_k.slice_mut(0..used_k);
                self.cuda_device
                    .memcpy_dtod(&src_k, &mut dst_k)
                    .map_err(|e| {
                        anyhow!("Q4KvCache::grow_to: k_q4_blocks memcpy_dtod failed: {e}")
                    })?;
            }
            {
                let src_v = self.v_q4.slice(0..used_v);
                let mut dst_v = new_v.slice_mut(0..used_v);
                self.cuda_device
                    .memcpy_dtod(&src_v, &mut dst_v)
                    .map_err(|e| anyhow!("Q4KvCache::grow_to: v_q4 memcpy_dtod failed: {e}"))?;
            }
        }

        tracing::debug!(
            "Q4KvCache: grew capacity {} -> {} tokens ({} -> {} blocks)",
            self.current_capacity_tokens,
            target_tokens,
            self.current_capacity_blocks,
            target_blocks,
        );
        self.k_q4_blocks = new_k;
        self.v_q4 = new_v;
        self.current_capacity_tokens = target_tokens;
        self.current_capacity_blocks = target_blocks;
        Ok(())
    }

    /// Push the current write position into a device-side i32 tensor.
    /// Required for CUDA graph capture of Q4 KV append - the captured
    /// kernels read their offsets from this device pointer, while the
    /// host updates it OUTSIDE the captured region each token via
    /// `update_graph_state(pos)`.
    ///
    /// Allocated lazily on first call. Subsequent calls do a small
    /// host-to-device copy (4 bytes) to refresh the position.
    pub fn update_graph_state(&mut self, pos: usize) -> Result<()> {
        let val: [i32; 1] = [pos as i32];
        // First call: allocate the device slot. Subsequent calls: memcpy into
        // the same pointer so graph-replay sees a fixed device address.
        match self.cur_pos_dev.as_mut() {
            Some(dst) => {
                self.cuda_device
                    .cuda_stream()
                    .memcpy_htod(val.as_slice(), dst)
                    .map_err(|e| anyhow!("Q4KvCache::update_graph_state copy: {e}"))?;
            }
            None => {
                let slice: CudaSlice<i32> = self
                    .cuda_device
                    .cuda_stream()
                    .clone_htod(val.as_slice())
                    .map_err(|e| anyhow!("Q4KvCache::update_graph_state alloc: {e}"))?;
                self.cur_pos_dev = Some(slice);
            }
        }
        Ok(())
    }

    /// Returns the device pointer to the current-position i32 tensor,
    /// or None if `update_graph_state` has never been called. Append
    /// paths use this to pick between the host-int and device-pos
    /// kernel variants.
    pub fn cur_pos_dev_ptr(&self) -> Option<&CudaSlice<i32>> {
        self.cur_pos_dev.as_ref()
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        // Drop the device-pos tensor so prefill goes back to the host-int
        // append path. A stale `cur_pos_dev` from a previous request
        // would make every prefill token's append read the LAST decode's
        // pos (= slot from prior turn), overwriting the residual at one
        // wrong slot. The engine re-primes via `update_graph_state(pos)`
        // before the first decode call of the next graph cycle.
        // Don't bother zeroing buffers - `dequantize_kv` only reads up to
        // current_seq_len and residual slots are overwritten before read.
        self.cur_pos_dev = None;
    }

    /// Trim to `new_len`. If the trim lands inside a previously-flushed
    /// K block, dequantize that block back into the residual so the next
    /// append has consistent state.
    pub fn trim_to(&mut self, new_len: usize) -> Result<()> {
        let new_len = new_len.min(self.current_seq_len);
        if new_len == self.current_seq_len {
            return Ok(());
        }

        let old_block = self.current_seq_len.div_ceil(q4_0_block_size()); // number of flushed blocks
        let new_block = new_len / q4_0_block_size();
        let new_slot = new_len % q4_0_block_size();

        // If trim reached back into a block that was already flushed
        // (new_block is strictly less than what would have been the
        // residual's block index, i.e. we lost our current partial data),
        // repopulate residual from that flushed block.
        if new_slot != 0 && new_block < old_block {
            self.dequant_block_to_residual(new_block)?;
        }

        self.current_seq_len = new_len;
        Ok(())
    }

    /// Append K and V for one or more tokens. For prefill (n_new ≫ 32 with
    /// the cache aligned to a 32-boundary) the work is batched into one
    /// quantize+memcpy per K and one per V. The trailing partial block
    /// (and any unaligned-start case) falls back to the per-token path.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let k_dims = k_new.dims();
        if k_dims.len() != 4
            || k_dims[0] != 1
            || k_dims[1] != self.n_kv_heads
            || k_dims[3] != self.head_dim
        {
            return Err(anyhow!(
                "Q4KvCache::append: K shape {:?} does not match [1,{},*,{}]",
                k_dims,
                self.n_kv_heads,
                self.head_dim
            ));
        }
        if v_new.dims() != k_dims {
            return Err(anyhow!(
                "Q4KvCache::append: V shape {:?} does not match K shape {:?}",
                v_new.dims(),
                k_dims
            ));
        }
        let n_new = k_dims[2];
        if self.current_seq_len + n_new > self.max_seq_len {
            return Err(anyhow!(
                "Q4KvCache::append: would exceed max_seq_len {} (have {}, adding {})",
                self.max_seq_len,
                self.current_seq_len,
                n_new
            ));
        }

        // Lazy growth: enlarge buffers if the append would exceed
        // current capacity. Fails cleanly if a CUDA graph has been
        // captured (the captured stride + base pointer would be stale).
        if self.current_seq_len + n_new > self.current_capacity_tokens {
            self.grow_to(self.current_seq_len + n_new)?;
        }

        // Bulk fast path: when we start aligned to a 32-boundary and have at
        // least one full block worth of new tokens. Quantizes all
        // n_full_blocks*32 positions in a single launch each for K and V.
        let mut start = 0usize;
        if self.current_seq_len.is_multiple_of(q4_0_block_size()) && n_new >= q4_0_block_size() {
            let n_full_blocks = n_new / q4_0_block_size();
            let bulk_seq = n_full_blocks * q4_0_block_size();
            self.append_bulk(k_new, v_new, bulk_seq)?;
            start = bulk_seq;
        }

        for i in start..n_new {
            let k_i = k_new.i((.., .., i..i + 1, ..))?;
            let v_i = v_new.i((.., .., i..i + 1, ..))?;
            self.append_single_k(&k_i)?;
            self.append_single_v(&v_i)?;
            self.current_seq_len += 1;
            // Host-side flush is skipped in graph mode - the conditional
            // dev-pos kernel launched inside `append_single_k` handles the
            // 32-window closing on the device side. This keeps the captured
            // launch list identical across replays.
            if self.cur_pos_dev.is_none() && self.current_seq_len.is_multiple_of(q4_0_block_size())
            {
                self.flush_k_residual_to_blocks(self.current_seq_len / q4_0_block_size() - 1)?;
            }
        }
        Ok(())
    }

    /// Bulk-append `bulk_seq` (multiple of 32) tokens worth of K and V at
    /// the current 32-aligned cache boundary. Single quantize + memcpy
    /// each side. Caller must guarantee `current_seq_len % 32 == 0`.
    fn append_bulk(&mut self, k_new: &Tensor, v_new: &Tensor, bulk_seq: usize) -> Result<()> {
        debug_assert!(bulk_seq.is_multiple_of(q4_0_block_size()));
        debug_assert!(self.current_seq_len.is_multiple_of(q4_0_block_size()));
        let n_full_blocks = bulk_seq / q4_0_block_size();

        // ----- K bulk -----
        // Layout per the cache spec: block (sb, h, c) holds 32 seq values
        // of K[h][seq=sb*32..sb*32+32][c]. Reshape source [1, n_kv, bulk_seq, head_dim]
        // -> [n_full_blocks, n_kv, head_dim, 32] so each contiguous 32-element
        // row that quantize sees corresponds to one (sb, h, c) block.
        let k_blocks = k_new
            .i((.., .., 0..bulk_seq, ..))?
            .squeeze(0)? // [n_kv, bulk_seq, head_dim]
            .permute((0, 2, 1))? // [n_kv, head_dim, bulk_seq]
            .reshape((
                self.n_kv_heads,
                self.head_dim,
                n_full_blocks,
                q4_0_block_size(),
            ))?
            .permute((2, 0, 1, 3))?; // [n_full_blocks, n_kv, head_dim, 32]
        let k_view = settled(&k_blocks, DType::F32)?;
        let k_total_elems = n_full_blocks * self.n_kv_heads * self.head_dim * q4_0_block_size();
        let k_total_bytes = n_full_blocks * self.n_kv_heads * self.head_dim * q4_0_type_size();

        let mut k_staging = QCudaStorage::zeros(&self.cuda_device, k_total_elems, GgmlDType::Q4_0)?;
        {
            let (s, l) = k_view.storage_and_layout();
            let cuda = match &*s {
                StorageView::Cuda(c) => c,
                _ => return Err(anyhow!("Q4KvCache::append_bulk: K must be CUDA")),
            };
            k_staging
                .quantize_with_layout(cuda, l)
                .map_err(|e| anyhow!("Q4KvCache::append_bulk: K quantize: {e}"))?;
        }
        let src_k = k_staging
            .data_slice_for_copy(k_total_bytes)
            .map_err(|e| anyhow!("Q4KvCache::append_bulk: K data_slice: {e}"))?;
        let k_block_start = self.current_seq_len / q4_0_block_size();
        let k_dst_off = k_block_start * self.n_kv_heads * self.head_dim * q4_0_type_size();
        let mut k_dst = self
            .k_q4_blocks
            .slice_mut(k_dst_off..k_dst_off + k_total_bytes);
        self.cuda_device
            .memcpy_dtod(&src_k, &mut k_dst)
            .map_err(|e| anyhow!("Q4KvCache::append_bulk: K memcpy: {e}"))?;

        // ----- V bulk -----
        // Layout: block (t, h, c_block) covers V[t][h][c_block*32..+32].
        // Reshape source [1, n_kv, bulk_seq, head_dim] -> [bulk_seq, n_kv, head_dim]
        // contiguous; quantize sees rows of 32 head_dim elements per (t, h, c_block).
        let v_blocks = v_new
            .i((.., .., 0..bulk_seq, ..))?
            .permute((0, 2, 1, 3))? // [1, bulk_seq, n_kv, head_dim]
            .squeeze(0)?; // [bulk_seq, n_kv, head_dim]
        let v_view = settled(&v_blocks, DType::F32)?;
        let v_total_elems = bulk_seq * self.n_kv_heads * self.head_dim;
        let token_bytes = self.n_kv_heads * (self.head_dim / q4_0_block_size()) * q4_0_type_size();
        let v_total_bytes = bulk_seq * token_bytes;

        let mut v_staging = QCudaStorage::zeros(&self.cuda_device, v_total_elems, GgmlDType::Q4_0)?;
        {
            let (s, l) = v_view.storage_and_layout();
            let cuda = match &*s {
                StorageView::Cuda(c) => c,
                _ => return Err(anyhow!("Q4KvCache::append_bulk: V must be CUDA")),
            };
            v_staging
                .quantize_with_layout(cuda, l)
                .map_err(|e| anyhow!("Q4KvCache::append_bulk: V quantize: {e}"))?;
        }
        let src_v = v_staging
            .data_slice_for_copy(v_total_bytes)
            .map_err(|e| anyhow!("Q4KvCache::append_bulk: V data_slice: {e}"))?;
        let v_dst_off = self.current_seq_len * token_bytes;
        let mut v_dst = self.v_q4.slice_mut(v_dst_off..v_dst_off + v_total_bytes);
        self.cuda_device
            .memcpy_dtod(&src_v, &mut v_dst)
            .map_err(|e| anyhow!("Q4KvCache::append_bulk: V memcpy: {e}"))?;

        self.current_seq_len += bulk_seq;
        Ok(())
    }

    /// Write one token's K into the residual buffer at `slot`.
    ///
    /// When `cur_pos_dev` is set (graph-capture mode), the slot index is
    /// read from the device pointer - host-int variants would freeze at
    /// capture time and replay would overwrite the same slot every
    /// token. The two kernel variants produce numerically identical
    /// output for a given `current_seq_len`; the choice is purely
    /// about graph-capture compatibility.
    fn append_single_k(&mut self, k: &Tensor) -> Result<()> {
        // k: [1, n_kv, 1, head_dim] -> one F16 run of [n_kv * head_dim], which is what the
        // residual scatter reads.
        let k_flat = settled(k, DType::F16)?;
        let (k_storage_guard, _) = k_flat.storage_and_layout();
        let k_cuda = match &*k_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q4KvCache: K must be on CUDA")),
        };
        let k_slice = k_cuda.as_cuda_slice::<f16>()?;
        let stream = self.cuda_device.cuda_stream();
        let src_ptr = k_slice.device_ptr(&stream).0 as *const core::ffi::c_void;
        let dst_ptr = self.k_residual.device_ptr(&stream).0 as *mut core::ffi::c_void;
        let stream_h = stream.cu_stream() as i64;
        unsafe {
            if let Some(pos_dev) = self.cur_pos_dev.as_ref() {
                let slot_ptr = pos_dev.device_ptr(&stream).0 as *const core::ffi::c_void;
                crate::inference::moe_cuda::kv_residual_scatter_f16_dev_slot_raw(
                    src_ptr,
                    dst_ptr,
                    slot_ptr,
                    self.n_kv_heads as i32,
                    self.head_dim as i32,
                    stream_h,
                );
                // Graph-mode conditional flush: launched every token so the
                // launch is recorded by CUDA graph capture, but writes only
                // when `pos & 31 == 31` (the just-scattered slot closes a
                // 32-token window). Quantises the F16 residual into the
                // Q4_0 block at `block_idx = pos >> 5` in k_q4_blocks.
                let residual_ptr =
                    self.k_residual.device_ptr(&stream).0 as *const core::ffi::c_void;
                let blocks_ptr = self.k_q4_blocks.device_ptr(&stream).0 as *mut core::ffi::c_void;
                // Use current_capacity_blocks, NOT max_seq_blocks: the
                // kernel strides through k_q4_blocks using this value,
                // and the buffer is sized for current_capacity_blocks
                // (lazy growth). At graph-capture time the stride is
                // baked in - grow_to refuses to run while a graph is
                // captured (cur_pos_dev.is_some()), so this stays valid
                // across replays.
                crate::inference::moe_cuda::flush_k_residual_q4_dev_pos_raw(
                    residual_ptr,
                    blocks_ptr,
                    slot_ptr,
                    self.n_kv_heads as i32,
                    self.head_dim as i32,
                    self.current_capacity_blocks as i32,
                    stream_h,
                );
            } else {
                let slot = self.current_seq_len % q4_0_block_size();
                crate::inference::moe_cuda::kv_residual_scatter_f16_raw(
                    src_ptr,
                    dst_ptr,
                    self.n_kv_heads as i32,
                    self.head_dim as i32,
                    slot as i32,
                    stream_h,
                );
            }
        }
        Ok(())
    }

    /// Quantize the residual buffer to Q4_0 blocks and memcpy them into
    /// k_q4_blocks at the slot for `block_idx`.
    fn flush_k_residual_to_blocks(&mut self, block_idx: usize) -> Result<()> {
        let residual_elems = self.n_kv_heads * self.head_dim * q4_0_block_size();
        let blocks_per_block_idx = self.n_kv_heads * self.head_dim;
        let bytes_per_block_idx = blocks_per_block_idx * q4_0_type_size();

        // Wrap residual as a flat [n_kv * head_dim * 32] F16 tensor for
        // quantize_with_layout.
        let residual_view = self.k_residual.slice(..residual_elems);
        let residual_owned = self
            .cuda_device
            .cuda_stream()
            .clone_dtod(&residual_view)
            .map_err(|e| anyhow!("Q4KvCache: flush clone_dtod: {e}"))?;
        let residual_storage =
            CudaStorage::wrap_cuda_slice(residual_owned, self.cuda_device.clone());
        let residual_tensor = tensor_from_cuda_storage(residual_storage, (residual_elems,))?;
        let (res_storage_guard, res_layout) = residual_tensor.storage_and_layout();
        let res_cuda = match &*res_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q4KvCache: residual must be CUDA for quantize")),
        };

        // Quantize into the staging blob.
        self.k_residual_staging
            .quantize_with_layout(res_cuda, res_layout)
            .map_err(|e| anyhow!("Q4KvCache: residual quantize: {e}"))?;
        let src_slice = self
            .k_residual_staging
            .data_slice_for_copy(bytes_per_block_idx)
            .map_err(|e| anyhow!("Q4KvCache: residual data_slice: {e}"))?;

        let dst_byte_offset = block_idx * bytes_per_block_idx;
        let mut dst_slice = self
            .k_q4_blocks
            .slice_mut(dst_byte_offset..dst_byte_offset + bytes_per_block_idx);
        self.cuda_device
            .memcpy_dtod(&src_slice, &mut dst_slice)
            .map_err(|e| anyhow!("Q4KvCache: K block memcpy: {e}"))?;
        Ok(())
    }

    /// Dequantize K block `block_idx` back into the residual buffer. Used
    /// on trim when the new_len lands inside a previously-flushed block.
    fn dequant_block_to_residual(&mut self, block_idx: usize) -> Result<()> {
        let blocks_per_block_idx = self.n_kv_heads * self.head_dim;
        let bytes_per_block_idx = blocks_per_block_idx * q4_0_type_size();
        let elem_count = blocks_per_block_idx * q4_0_block_size();

        // Allocate owned CudaSlice<u8> with just this block_idx's data
        // (dequantize_q4_0_blob_f16 wants a contiguous CudaSlice<u8>).
        let src_offset = block_idx * bytes_per_block_idx;
        let src_view = self
            .k_q4_blocks
            .slice(src_offset..src_offset + bytes_per_block_idx);
        let mut src_owned = self.cuda_device.alloc_zeros::<u8>(bytes_per_block_idx)?;
        self.cuda_device
            .memcpy_dtod(&src_view, &mut src_owned)
            .map_err(|e| anyhow!("Q4KvCache: trim block copy: {e}"))?;

        let dq_storage = dequantize_q4_0_blob_f16(&src_owned, elem_count, &self.cuda_device)
            .map_err(|e| anyhow!("Q4KvCache: trim block dequant: {e}"))?;

        // dq_storage is a CudaStorage holding f16 values. Extract the
        // slice and memcpy into self.k_residual.
        let dq_slice = dq_storage.as_cuda_slice::<f16>()?;
        self.cuda_device
            .memcpy_dtod(&dq_slice.slice(..elem_count), &mut self.k_residual)
            .map_err(|e| anyhow!("Q4KvCache: trim residual restore: {e}"))?;
        Ok(())
    }

    /// Append one V token - same behaviour as the session-1 Q4KvCache
    /// V path (packed per-token with 32-channel sub-groups).
    fn append_single_v(&mut self, v: &Tensor) -> Result<()> {
        // v: [1, n_kv, 1, head_dim] F32 (from attn_post_qkv_decode_qf32).
        // The seq=1 dim makes the permute (0,2,1,3) -> [1,1,n_kv,head_dim]
        // a no-op in memory order; we can flatten v directly to
        // [n_kv*head_dim] and skip the permute+contiguous launch.
        let cast = if v.dtype() == DType::F32 {
            v.clone()
        } else {
            v.to_dtype(DType::F32)?
        };
        let laid_out = cast.flatten_all()?;
        let (src_storage_guard, src_layout) = laid_out.storage_and_layout();
        let src_cuda = match &*src_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q4KvCache: V must be on CUDA")),
        };

        let token_bytes = self.n_kv_heads * (self.head_dim / q4_0_block_size()) * q4_0_type_size();

        self.v_staging_decode
            .quantize_with_layout(src_cuda, src_layout)
            .map_err(|e| anyhow!("Q4KvCache: V quantize: {e}"))?;
        let src_slice = self
            .v_staging_decode
            .data_slice_for_copy(token_bytes)
            .map_err(|e| anyhow!("Q4KvCache: V data_slice: {e}"))?;

        // Graph-mode: use the device-pos byte scatter kernel so the
        // destination offset is computed from `cur_pos_dev[0]` at
        // replay time. Non-graph mode keeps the host-offset `memcpy_dtod`
        // - slightly cheaper because it bypasses kernel launch overhead.
        if let Some(pos_dev) = self.cur_pos_dev.as_ref() {
            let stream = self.cuda_device.cuda_stream();
            let src_ptr = src_slice.device_ptr(&stream).0 as *const core::ffi::c_void;
            let dst_ptr = self.v_q4.device_ptr(&stream).0 as *mut core::ffi::c_void;
            let pos_ptr = pos_dev.device_ptr(&stream).0 as *const core::ffi::c_void;
            let stream_h = stream.cu_stream() as i64;
            unsafe {
                crate::inference::moe_cuda::q4_v_scatter_bytes_dev_pos_raw(
                    src_ptr,
                    dst_ptr,
                    pos_ptr,
                    token_bytes as i32,
                    stream_h,
                );
            }
        } else {
            let dst_byte_offset = self.current_seq_len * token_bytes;
            let mut dst_slice = self
                .v_q4
                .slice_mut(dst_byte_offset..dst_byte_offset + token_bytes);
            self.cuda_device
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow!("Q4KvCache: V memcpy: {e}"))?;
        }
        Ok(())
    }

    /// Dequantize the full K and V caches into
    /// `[1, n_kv_heads, current_seq_len, head_dim]` F-dtype tensors.
    pub fn dequantize_kv(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        let seq = self.current_seq_len;
        if seq == 0 {
            return Err(anyhow!("Q4KvCache::dequantize_kv: cache is empty"));
        }

        let full_seq_blocks = seq / q4_0_block_size();
        let partial_slots = seq % q4_0_block_size();
        let full_seq = full_seq_blocks * q4_0_block_size();

        // -- K dequant ------------------------------------------------
        // Blocks live in [seq_block, n_kv, head_dim] order, each block is
        // 32 seq positions of one (head, channel). Dequantize all flushed
        // blocks, then permute to [full_seq, n_kv, head_dim].
        let k_full = if full_seq > 0 {
            let elem_count = full_seq_blocks * self.n_kv_heads * self.head_dim * q4_0_block_size();
            // dequantize_q4_0_blob_f16 takes a CudaSlice<u8>; use the prefix
            // of k_q4_blocks covering just the flushed blocks.
            let bytes_needed = full_seq_blocks * self.n_kv_heads * self.head_dim * q4_0_type_size();
            let mut owned = self.cuda_device.alloc_zeros::<u8>(bytes_needed)?;
            self.cuda_device
                .memcpy_dtod(&self.k_q4_blocks.slice(..bytes_needed), &mut owned)
                .map_err(|e| anyhow!("Q4KvCache: K dequant copy: {e}"))?;
            let dq = dequantize_q4_0_blob_f16(&owned, elem_count, &self.cuda_device)
                .map_err(|e| anyhow!("Q4KvCache: K dequant: {e}"))?;

            // Layout after dequant is [seq_block, n_kv, head_dim, 32].
            // Permute to [seq_block, 32, n_kv, head_dim] -> reshape to
            // [full_seq, n_kv, head_dim].
            let t = tensor_from_cuda_storage(
                dq,
                (
                    full_seq_blocks,
                    self.n_kv_heads,
                    self.head_dim,
                    q4_0_block_size(),
                ),
            )?;
            let t = t.permute((0, 3, 1, 2))?.contiguous()?; // [sb, 32, n_kv, hd]
            t.reshape((full_seq, self.n_kv_heads, self.head_dim))?
        } else {
            // Empty - placeholder shape; we'll skip concat when full_seq == 0.
            Tensor::zeros_on(
                (0, self.n_kv_heads, self.head_dim),
                DType::F16,
                &self.device,
            )?
        };

        // Add the residual partial block (if any). Residual is
        // [n_kv, head_dim, 32] F16; slice first `partial_slots` seq slots
        // and permute to [partial_slots, n_kv, head_dim].
        let k = if partial_slots > 0 {
            let residual_elems = self.n_kv_heads * self.head_dim * q4_0_block_size();
            let mut owned = self.cuda_device.alloc_zeros::<f16>(residual_elems)?;
            self.cuda_device
                .memcpy_dtod(&self.k_residual.slice(..residual_elems), &mut owned)
                .map_err(|e| anyhow!("Q4KvCache: K residual copy: {e}"))?;
            let residual_storage = CudaStorage::wrap_cuda_slice(owned, self.cuda_device.clone());
            let res_tensor = tensor_from_cuda_storage(
                residual_storage,
                (self.n_kv_heads, self.head_dim, q4_0_block_size()),
            )?;
            // [n_kv, head_dim, 32] -> narrow to [n_kv, head_dim, partial_slots]
            // -> permute to [partial_slots, n_kv, head_dim].
            let partial = res_tensor
                .narrow(2, 0, partial_slots)?
                .permute((2, 0, 1))?
                .contiguous()?;

            if full_seq > 0 {
                Tensor::cat(&[&k_full, &partial], 0)?
            } else {
                partial
            }
        } else if full_seq > 0 {
            k_full
        } else {
            return Err(anyhow!("Q4KvCache::dequantize_kv: empty K reconstruction"));
        };

        // [seq, n_kv, head_dim] -> [1, n_kv, seq, head_dim]
        let k = k
            .unsqueeze(0)? // [1, seq, n_kv, head_dim]
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        // -- V dequant ------------------------------------------------
        // V layout is the session-1 [seq, n_kv, head_dim] Q4_0 with blocks
        // grouped along head_dim. dequantize_q4_0_blob_f16 emits F16 in the
        // same element order, so reshape + permute reaches the same
        // [1, n_kv, seq, head_dim] shape.
        let v_elem_count = seq * self.n_kv_heads * self.head_dim;
        let v_bytes =
            seq * self.n_kv_heads * (self.head_dim / q4_0_block_size()) * q4_0_type_size();
        let mut v_owned = self.cuda_device.alloc_zeros::<u8>(v_bytes)?;
        self.cuda_device
            .memcpy_dtod(&self.v_q4.slice(..v_bytes), &mut v_owned)
            .map_err(|e| anyhow!("Q4KvCache: V dequant copy: {e}"))?;
        let v_dq = dequantize_q4_0_blob_f16(&v_owned, v_elem_count, &self.cuda_device)
            .map_err(|e| anyhow!("Q4KvCache: V dequant: {e}"))?;
        let v = tensor_from_cuda_storage(v_dq, (1, seq, self.n_kv_heads, self.head_dim))?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        let k = if k.dtype() != dtype {
            k.to_dtype(dtype)?
        } else {
            k
        };
        let v = if v.dtype() != dtype {
            v.to_dtype(dtype)?
        } else {
            v
        };
        Ok((k, v))
    }

    /// Fused Q4_0 attention scores. One kernel launch covers the entire
    /// `current_seq_len` - the kernel reads the flushed blocks from
    /// `k_q4_blocks` and the partial block directly from the F16 residual
    /// buffer. No host-side residual matmul or concat is needed.
    ///
    /// `q`: shape `[1, n_q_heads, 1, head_dim]`. Returns scores
    /// `[1, n_q_heads, 1, current_seq_len]` f32.
    pub fn attn_scores(&self, q: &Tensor) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_score_q4_0_f32_kivi;

        let seq = self.current_seq_len;
        if seq == 0 {
            return Err(anyhow!("Q4KvCache::attn_scores: cache is empty"));
        }
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q4KvCache::attn_scores: Q shape {:?} not [1,n_q,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q4KvCache::attn_scores: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads,
                self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;

        let q_flat = settled(q, DType::F32)?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q4KvCache::attn_scores: Q must be on CUDA")),
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);

        let scores_storage = attn_score_q4_0_f32_kivi(
            &self.k_q4_blocks,
            &self.k_residual,
            &q_view,
            self.head_dim,
            seq,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q4KvCache: q4 score kernel: {e}"))?;

        let scores = tensor_from_cuda_storage(scores_storage, (1, n_q_heads, 1, seq))?;
        Ok(scores)
    }

    /// Fused Q4_0 attention output (probs @ V). Analogous to Q8's
    /// `attn_output`: feeds the full V cache (including the not-yet-flushed
    /// tail token) directly to the fused kernel without dequantising.
    ///
    /// `probs` shape: `[1, n_q_heads, 1, current_seq_len]` f32. Returns
    /// context `[1, n_q_heads, 1, head_dim]` f32.
    /// Fused softmax + Q4_0 attn output: takes raw scores (post Q@K^T,
    /// post-scale), applies softmax internally, and accumulates into the
    /// V multiply. One kernel instead of softmax_last_dim + attn_output.
    pub fn attn_softmax_output(&self, scores: &Tensor) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_softmax_output_q4_0_f32_gqa;

        let seq = self.current_seq_len;
        if seq == 0 {
            return Err(anyhow!("Q4KvCache::attn_softmax_output: cache is empty"));
        }
        let dims = scores.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != seq {
            return Err(anyhow!(
                "Q4KvCache::attn_softmax_output: scores shape {:?} not [1,n_q,1,{}]",
                dims,
                seq
            ));
        }
        let n_q_heads = dims[1];
        let n_q_per_kv = n_q_heads / self.n_kv_heads;

        let scores_flat = settled(scores, DType::F32)?;
        let s_storage_guard = scores_flat.storage_and_layout().0;
        let s_cuda = match &*s_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q4KvCache::attn_softmax_output: scores must be CUDA"
                ))
            }
        };
        let s_view = s_cuda.as_cuda_slice::<f32>()?.slice(..);

        let out = attn_softmax_output_q4_0_f32_gqa(
            &self.v_q4,
            &s_view,
            self.head_dim,
            seq,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q4KvCache: q4 softmax+output kernel: {e}"))?;
        let out_tensor = tensor_from_cuda_storage(out, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    /// Fused split-K flash-decode: replaces the `attn_scores` (-> HBM scores) +
    /// `attn_softmax_output` 2-kernel chain with ONE pass (no HBM scores
    /// round-trip, adaptive nsplit for occupancy). `q` is `[1,n_q,1,head_dim]`,
    /// `scale` is the softmax scale (apply iff Q was not pre-scaled). Returns
    /// context `[1,n_q,1,head_dim]` f32. hd∈{64,128}, GQA only.
    pub fn attn_flash_splitk_decode(&self, q: &Tensor, scale: f32) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_flash_splitk_q4_gqa_decode;
        let seq = self.current_seq_len;
        if seq == 0 {
            return Err(anyhow!(
                "Q4KvCache::attn_flash_splitk_decode: cache is empty"
            ));
        }
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q4KvCache::attn_flash_splitk_decode: Q shape {:?} not [1,n_q,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q4KvCache::attn_flash_splitk_decode: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads,
                self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        let q_flat = settled(q, DType::F32)?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q4KvCache::attn_flash_splitk_decode: Q must be on CUDA"
                ))
            }
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);
        let out = attn_flash_splitk_q4_gqa_decode(
            &self.k_q4_blocks,
            &self.k_residual,
            &self.v_q4,
            &q_view,
            self.head_dim,
            self.n_kv_heads,
            n_q_per_kv,
            seq,
            scale,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q4KvCache: q4 flash-splitk kernel: {e}"))?;
        let out_tensor = tensor_from_cuda_storage(out, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    /// Graph-capturable fused flash split-K decode (device-position seq_kv).
    /// Same result as the `attn_scores`+softmax+`attn_output` chain but in one
    /// fused pass (no HBM scores round-trip), reading seq_kv from `cur_pos_dev`.
    /// Mirrors `Q8KvCache::attn_flash_splitk_decode_graph`. Q `[1,n_q,1,hd]` ->
    /// `[1,n_q,1,hd]` f32.
    pub fn attn_flash_splitk_decode_graph(&self, q: &Tensor, scale: f32) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_flash_splitk_q4_gqa_decode_dev_pos;
        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!("Q4KvCache::attn_flash_splitk_decode_graph: update_graph_state not called")
        })?;
        let seq = self.current_seq_len;
        if seq == 0 {
            return Err(anyhow!(
                "Q4KvCache::attn_flash_splitk_decode_graph: cache is empty"
            ));
        }
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q4KvCache::attn_flash_splitk_decode_graph: Q shape {:?} not [1,n_q,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q4KvCache::attn_flash_splitk_decode_graph: n_q_heads {} not multiple of n_kv_heads {}",
                n_q_heads, self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        let q_flat = settled(q, DType::F32)?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q4KvCache::attn_flash_splitk_decode_graph: Q must be on CUDA"
                ))
            }
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);
        let out = attn_flash_splitk_q4_gqa_decode_dev_pos(
            &self.k_q4_blocks,
            &self.k_residual,
            &self.v_q4,
            &q_view,
            pos_dev,
            self.head_dim,
            self.n_kv_heads,
            n_q_per_kv,
            seq,
            scale,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q4KvCache: q4 flash-splitk dev_pos kernel: {e}"))?;
        let out_tensor = tensor_from_cuda_storage(out, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    pub fn attn_output(&self, probs: &Tensor) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_output_q4_0_f32_gqa;

        let seq = self.current_seq_len;
        if seq == 0 {
            return Err(anyhow!("Q4KvCache::attn_output: cache is empty"));
        }
        let dims = probs.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != seq {
            return Err(anyhow!(
                "Q4KvCache::attn_output: probs shape {:?} not [1,n_q,1,{}]",
                dims,
                seq
            ));
        }
        let n_q_heads = dims[1];
        let n_q_per_kv = n_q_heads / self.n_kv_heads;

        let probs_flat = settled(probs, DType::F32)?;
        let p_storage_guard = probs_flat.storage_and_layout().0;
        let p_cuda = match &*p_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q4KvCache::attn_output: probs must be on CUDA")),
        };
        let p_view = p_cuda.as_cuda_slice::<f32>()?.slice(..);

        let out = attn_output_q4_0_f32_gqa(
            &self.v_q4,
            &p_view,
            self.head_dim,
            seq,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q4KvCache: q4 output kernel: {e}"))?;
        let out_tensor = tensor_from_cuda_storage(out, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    /// Max sequence length rounded up to a multiple of 32 - the padded
    /// stride used by the graph-safe attention kernels. Positions in
    /// `[current_seq_len, max_seq_padded)` are masked via -INFINITY by
    /// the score kernel so the downstream softmax produces zero there.
    ///
    /// **Capped at 2048** for graph mode: at the model's full context
    /// (32K+), the captured kernel grid schedules max_seq_padded/32
    /// blocks per layer per token. The vast majority of those blocks
    /// early-exit (write -INF) but the scheduling cost dominates decode
    /// latency. Capping here trades graph-mode reach for ~3x speedup at
    /// short contexts. Callers MUST check `current_seq_len < max_seq_padded`
    /// before invoking the graph-safe attention path.
    pub fn max_seq_padded(&self) -> usize {
        // 2048 covers typical chat decode lengths (500-1500 tok) without
        // hitting the graph-mode `pos >= mask_width` fallback that
        // invalidates the captured graph mid-request. The attention
        // score kernel iterates `max_seq_padded` positions per token,
        // so larger cap = more per-token work - but in practice the
        // 4x kernel cost is dwarfed by the savings from staying in
        // graph mode (vs 1 cuGraphLaunch + falling to ~14xN_layers
        // per-token kernel launches once the cap is hit).
        // Verified: at 1000-tok decode the cap=512 fallback
        // regresses deepcoder -3% -> -11% vs Ollama, qwen3 +0% -> -5%.
        const CAP: usize = 2048;
        // Use current_capacity_tokens (rounded up to block boundary)
        // instead of max_seq_len: the buffer is sized for current
        // capacity, and the graph attn kernel must not stride past it.
        // Still clamp to CAP to avoid the per-token grid overhead at
        // full 32 K context.
        let full = self.current_capacity_blocks * q4_0_block_size();
        full.min(CAP)
    }

    /// Graph-safe attention scores. Mirrors `attn_scores` but reads
    /// `current_seq_len` from `cur_pos_dev` on the GPU (set by
    /// `update_graph_state`) and writes scores at the fixed
    /// `[n_q_heads, max_seq_padded]` stride. Positions >= current_seq_len
    /// are written as -INFINITY so a subsequent `softmax_last_dim` over
    /// the padded width zeroes them out.
    ///
    /// `q`: shape `[1, n_q_heads, 1, head_dim]`. Returns scores
    /// `[1, n_q_heads, 1, max_seq_padded]` f32. The caller MUST apply
    /// softmax across the FULL `max_seq_padded` axis (not `seq`).
    pub fn attn_scores_graph(&self, q: &Tensor) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_score_q4_0_f32_kivi_dev_pos;

        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!("Q4KvCache::attn_scores_graph: update_graph_state has not been called")
        })?;
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q4KvCache::attn_scores_graph: Q shape {:?} not [1,n_q,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q4KvCache::attn_scores_graph: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads,
                self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        let max_seq_padded = self.max_seq_padded();

        let q_flat = settled(q, DType::F32)?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q4KvCache::attn_scores_graph: Q must be on CUDA")),
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);

        let scores_storage = attn_score_q4_0_f32_kivi_dev_pos(
            &self.k_q4_blocks,
            &self.k_residual,
            &q_view,
            pos_dev,
            self.head_dim,
            max_seq_padded,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q4KvCache: q4 dev-pos score kernel: {e}"))?;

        let scores = tensor_from_cuda_storage(scores_storage, (1, n_q_heads, 1, max_seq_padded))?;
        Ok(scores)
    }

    /// Graph-safe attention output. Mirrors `attn_output` but reads
    /// `current_seq_len` from `cur_pos_dev` and expects probs at the
    /// fixed `max_seq_padded` stride. Probs at positions >= current_seq_len
    /// must be zero (the standard softmax-on-padded-scores does this).
    ///
    /// `probs`: shape `[1, n_q_heads, 1, max_seq_padded]` f32. Returns
    /// context `[1, n_q_heads, 1, head_dim]` f32.
    pub fn attn_output_graph(&self, probs: &Tensor) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_output_q4_0_f32_dev_pos;

        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!("Q4KvCache::attn_output_graph: update_graph_state has not been called")
        })?;
        let max_seq_padded = self.max_seq_padded();
        let dims = probs.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != max_seq_padded {
            return Err(anyhow!(
                "Q4KvCache::attn_output_graph: probs shape {:?} not [1,n_q,1,{}]",
                dims,
                max_seq_padded
            ));
        }
        let n_q_heads = dims[1];
        let n_q_per_kv = n_q_heads / self.n_kv_heads;

        let probs_flat = settled(probs, DType::F32)?;
        let p_storage_guard = probs_flat.storage_and_layout().0;
        let p_cuda = match &*p_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q4KvCache::attn_output_graph: probs must be on CUDA"
                ))
            }
        };
        let p_view = p_cuda.as_cuda_slice::<f32>()?.slice(..);

        let out = attn_output_q4_0_f32_dev_pos(
            &self.v_q4,
            &p_view,
            pos_dev,
            self.head_dim,
            max_seq_padded,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q4KvCache: q4 dev-pos output kernel: {e}"))?;
        let out_tensor = tensor_from_cuda_storage(out, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }
}
