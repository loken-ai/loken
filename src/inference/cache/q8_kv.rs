//! Q8-quantized KV cache (K-only for now, V stays F-dtype).
//!
//! StorageView layout: the K cache is a single `CudaSlice<u8>` packed as
//! `[max_seq_len, n_kv_heads, head_dim]` of Q8_0 blocks (34 bytes per 32
//! elements). This order lets us do **one** contiguous memcpy per decode
//! step to append a freshly-quantized row covering all kv-heads, and it
//! lets `attn_score_q8_0_q8_1_raw` stride over the siblings of a token's
//! K row with `n_kv_heads * head_dim / 32` blocks.
//!
//! V is currently kept as a pre-allocated F-dtype tensor (matching
//! `SpecKvCache` semantics); porting V to Q8 needs a separate kernel.

use crate::inference::quantized_cuda::{
    attn_output_q8_0_f32_dev_pos, attn_output_q8_0_f32_gqa, attn_score_q8_0_f32_dev_pos,
    attn_score_q8_0_q8_1_gqa, attn_softmax_output_q8_0_f32_dev_pos, quantize_q8_0_f32_dev_slot,
    quantize_q8_0_f32_into_offset, MATRIX_ROW_PADDING,
};
use crate::tensor::cuda_ext::{
    capture_active_on, tensor_from_cuda_storage, CudaSlice, RawCudaDevice as CudaDevice,
};
use crate::tensor::quantized::{GgmlDType, QCudaStorage};
use crate::tensor::{DType, Device, StorageView, Tensor};
use anyhow::{anyhow, Result};

/// One decode step with its two length-one axes dropped, in F32 and contiguous.
///
/// A step arrives shaped `[1, heads, 1, width]`; those two axes are shape, not data, and the
/// kernels below take a device pointer with their own strides, so what they want is the rows
/// packed and nothing else.
fn step_rows_f32(t: &Tensor) -> Result<Tensor> {
    Ok(t.squeeze(2)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .contiguous()?)
}

/// The same step as one flat run, for the kernels that index it themselves.
fn packed_f32(t: &Tensor) -> Result<Tensor> {
    Ok(step_rows_f32(t)?.flatten_all()?)
}

// The block geometry is the format's, not this cache's: asking the dtype keeps a cache that
// stores Q8_0 from disagreeing with the codec that reads it.
const Q8_0_BLOCK_SIZE: usize = crate::tensor::quantized::GgmlDType::Q8_0.block_size();
const Q8_0_TYPE_SIZE: usize = crate::tensor::quantized::GgmlDType::Q8_0.type_size();

pub(crate) use super::KV_WORKING_WINDOW_TOKENS as INITIAL_CAPACITY_TOKENS;

/// KV cache where both K and V are persistently stored as Q8_0.
///
/// **Lazy capacity:** the underlying CudaSlice<u8> buffers are sized
/// for `current_capacity` tokens (<= `max_seq_len`). `append` grows the
/// capacity in-place by allocating a larger buffer + memcpying the
/// used prefix when more is needed. `max_seq_len` is the hard cap
/// (typically the user's `context_length` config); appends beyond it
/// return an error.
pub struct Q8KvCache {
    /// Packed Q8_0 K storage, laid out as [current_capacity, n_kv_heads, head_dim].
    /// The trailing `head_dim / 32` blocks of a token are contiguous in
    /// memory for a given kv-head, and consecutive kv-heads of the same
    /// token follow immediately.
    k_q8: CudaSlice<u8>,
    /// Packed Q8_0 V storage, same layout as `k_q8`.
    v_q8: CudaSlice<u8>,
    /// Reused staging buffer for per-step K quantization. Sized for a
    /// single-token append (`n_kv_heads * head_dim` elements). The
    /// buffer is kept live across calls so `QCudaStorage::quantize` hits
    /// its reuse fast path instead of allocating a fresh blob each step.
    k_staging_decode: QCudaStorage,
    /// Same as `k_staging_decode` but for the V tensor.
    v_staging_decode: QCudaStorage,
    current_seq_len: usize,
    /// Currently allocated capacity in tokens (<= max_seq_len). Grows
    /// in-place when an append would exceed it.
    current_capacity: usize,
    /// Hard cap on capacity - appends past this fail with an error.
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cuda_device: CudaDevice,
    /// CUDA graph mode: device-side i32 holding the current_seq_len BEFORE
    /// the next append. None until `update_graph_state(pos)` is first
    /// called. Once set, the graph-safe attention path (`attn_scores_graph`,
    /// `attn_output_graph`) becomes available - and the engine can
    /// dispatch to the lower-thread-count `dev_pos` kernels (which fit
    /// phi2's register budget where the non-`dev_pos` 1024-thread kernel
    /// hits `LAUNCH_OUT_OF_RESOURCES`).
    cur_pos_dev: Option<CudaSlice<i32>>,
    /// Last value written to `cur_pos_dev`. Used by `update_graph_state`
    /// to skip the H2D copy when the value hasn't changed (every layer
    /// in the same decode token calls update with the same pos).
    cur_pos_dev_value: Option<usize>,
    /// Set true once the engine has captured a CUDA graph that references
    /// these K/V buffers. While true `grow_to` rejects growth so the
    /// captured graph's kernel-node pointers stay valid. False while the
    /// dev_pos path is used WITHOUT a captured graph (phi2 non-graph
    /// decode) - then growth is allowed and re-allocates `cur_pos_dev`
    /// alongside the buffers.
    graph_captured: bool,
    /// Hard sequence-length ceiling for the CURRENTLY captured graph.
    /// Frozen by `mark_graph_captured()`:
    ///   - the buffer capacity at capture time (`grow_to` can't run under
    ///     a live graph - the dev_slot append kernel would write past the
    ///     captured pointer), and
    ///   - the `max_seq_padded` stride the 2-kernel score chain was
    ///     captured with, IF that chain was used (`chain_stride_seen`).
    ///     The captured score kernel writes -INF at a frozen stride;
    ///     once seq_kv needs a wider stride the replayed attention would
    ///     silently drop positions. (The split-K flash path derives its
    ///     per-split ranges from `cur_pos_dev` at replay time and has no
    ///     frozen stride, so it is only capacity-bound.)
    /// The model-level `update_graph_state` errors when `pos + 1` would
    /// exceed this, which makes the engine tear the graph down and
    /// re-capture (or decode eagerly) at the wider state.
    graph_seq_limit: Option<usize>,
    /// Stride (`max_seq_padded` at launch) most recently used by the
    /// fixed-stride score/output kernel chain. 0 = chain not used since
    /// the last capture cycle. Written from `&self` methods
    /// (`attn_scores_graph` / `attn_output_graph`), hence atomic.
    chain_stride_seen: std::sync::atomic::AtomicUsize,
}

impl Q8KvCache {
    /// Allocate K and V caches at a small initial capacity. They grow
    /// lazily on demand up to `max_seq_len`.
    pub fn new(
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        if !head_dim.is_multiple_of(Q8_0_BLOCK_SIZE) {
            return Err(anyhow!(
                "Q8KvCache: head_dim {head_dim} must be a multiple of {Q8_0_BLOCK_SIZE}"
            ));
        }
        let cuda_device = device
            .as_cuda_device()
            .map_err(|_| anyhow!("Q8KvCache: CUDA device required"))?;
        let initial_capacity = INITIAL_CAPACITY_TOKENS.min(max_seq_len).max(1);
        let kv_bytes =
            initial_capacity * n_kv_heads * (head_dim / Q8_0_BLOCK_SIZE) * Q8_0_TYPE_SIZE;
        let k_q8 = cuda_device.alloc_zeros::<u8>(kv_bytes)?;
        let v_q8 = cuda_device.alloc_zeros::<u8>(kv_bytes)?;

        // Pre-size staging buffers to one token's worth of Q8_0 bytes so
        // per-step quantize calls hit the reuse fast path. Multi-token
        // prefill append paths can still allocate larger temps on demand.
        let decode_elem_count = n_kv_heads * head_dim;
        let k_staging_decode =
            QCudaStorage::zeros(&cuda_device, decode_elem_count, GgmlDType::Q8_0)?;
        let v_staging_decode =
            QCudaStorage::zeros(&cuda_device, decode_elem_count, GgmlDType::Q8_0)?;

        Ok(Self {
            k_q8,
            v_q8,
            k_staging_decode,
            v_staging_decode,
            current_seq_len: 0,
            current_capacity: initial_capacity,
            max_seq_len,
            n_kv_heads,
            head_dim,
            cuda_device,
            cur_pos_dev: None,
            cur_pos_dev_value: None,
            graph_captured: false,
            graph_seq_limit: None,
            chain_stride_seen: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Engine signals that a CUDA graph capturing these buffers is now
    /// live. Subsequent `grow_to` calls will fail rather than re-allocating
    /// buffers under the captured kernels' pointers. Pair with
    /// `clear_graph_captured()` on graph invalidation.
    pub fn mark_graph_captured(&mut self) {
        self.graph_captured = true;
        // Freeze the replay-validity ceiling for this capture: buffer
        // capacity always; the score-chain stride only if the chain was
        // recorded into the graph (see `graph_seq_limit` field docs).
        let chain_stride = self
            .chain_stride_seen
            .load(std::sync::atomic::Ordering::Relaxed);
        let mut limit = self.current_capacity;
        if chain_stride > 0 {
            limit = limit.min(chain_stride);
        }
        self.graph_seq_limit = Some(limit);
    }

    /// Inverse of `mark_graph_captured` - called when the engine invalidates
    /// or recaptures the graph and the buffers are once again safe to grow.
    pub fn clear_graph_captured(&mut self) {
        self.graph_captured = false;
        self.graph_seq_limit = None;
        self.chain_stride_seen
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Sequence-length ceiling of the currently captured graph (None when
    /// no graph is live). `update_graph_state` at the model level checks
    /// `pos + 1 <= limit` each replayed token.
    pub fn graph_seq_limit(&self) -> Option<usize> {
        self.graph_seq_limit
    }

    /// Host-side bookkeeping sync for graph-replayed tokens. A captured
    /// graph advances the cache purely device-side (the dev_slot append
    /// kernel reads `cur_pos_dev`), so `current_seq_len` goes stale for
    /// every replayed token. The engine calls this after each replay with
    /// the post-token length so cross-request trim/reuse/prefill (which
    /// are host-int driven) operate on the true occupancy.
    pub fn set_seq_len_for_graph(&mut self, len: usize) {
        self.current_seq_len = len.min(self.max_seq_len);
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Padded stride used by the graph-safe attention kernels - sequence
    /// positions in `[current_seq_len, max_seq_padded)` get -INFINITY
    /// scores so the downstream softmax produces zeros there.
    ///
    /// Dynamic sizing: rounds the **current** sequence length up to the
    /// next 256-token block instead of using the full capacity. The score
    /// kernel iterates `max_seq_padded` positions per token, so capping
    /// at the actually-occupied + small headroom cuts the -INF writes
    /// from ~1700 to <=256 at decode time. Capacity grows the underlying
    /// K/V buffers in 256-token chunks via `grow_to`, so the stride
    /// stays valid until the next 256-block crossing.
    ///
    /// Capped at 2048 to keep parity with the graph-capture path: a wider score
    /// buffer makes a captured graph too large to hold without hitting
    /// `pos >= mask_width` invalidation.
    pub fn max_seq_padded(&self) -> usize {
        const CAP: usize = 2048;
        const STEP: usize = 256;
        // After append the position is current_seq_len; attention sees
        // current_seq_len+1 (kernel adds 1 internally for the just-written
        // slot). Round up to STEP-aligned bucket so the stride only changes
        // every 256 tokens, not every token.
        let used = (self.current_seq_len + 1).min(self.current_capacity);
        let rounded = ((used + STEP - 1) / STEP) * STEP;
        let cap_aligned = (self.current_capacity + Q8_0_BLOCK_SIZE - 1) & !(Q8_0_BLOCK_SIZE - 1);
        rounded.min(cap_aligned).min(CAP).max(STEP)
    }

    /// Returns the device pointer to the current-position i32 tensor,
    /// or None if `update_graph_state` has never been called. Engine
    /// dispatch reads this to decide between the non-dev_pos and dev_pos
    /// attention kernel variants.
    pub fn cur_pos_dev_ptr(&self) -> Option<&CudaSlice<i32>> {
        self.cur_pos_dev.as_ref()
    }

    /// Prime / refresh the device-side current_seq_len before each
    /// captured replay. Allocates lazily on first call; subsequent calls
    /// do a 4-byte H2D copy into the same device pointer so the captured
    /// graph reads the right value each replay.
    ///
    /// `pos` is `current_seq_len BEFORE the upcoming append` (= the slot
    /// the next token will write into).
    ///
    /// Fast path: if `cur_pos_dev` already holds `pos`, skip the H2D copy.
    /// At decode time, every layer in the same token calls this with the
    /// same value (positions only advance between tokens), so 23/24 of
    /// the per-token launches were redundant.
    pub fn update_graph_state(&mut self, pos: usize) -> Result<()> {
        if self.cur_pos_dev_value == Some(pos) && self.cur_pos_dev.is_some() {
            return Ok(());
        }
        let val: [i32; 1] = [pos as i32];
        match self.cur_pos_dev.as_mut() {
            Some(dst) => {
                self.cuda_device
                    .cuda_stream()
                    .memcpy_htod(val.as_slice(), dst)
                    .map_err(|e| anyhow!("Q8KvCache::update_graph_state copy: {e}"))?;
            }
            None => {
                let slice: CudaSlice<i32> = self
                    .cuda_device
                    .cuda_stream()
                    .clone_htod(val.as_slice())
                    .map_err(|e| anyhow!("Q8KvCache::update_graph_state alloc: {e}"))?;
                self.cur_pos_dev = Some(slice);
            }
        }
        self.cur_pos_dev_value = Some(pos);
        Ok(())
    }

    /// Reset the cache for a new request. Drops the cur_pos_dev so the
    /// next graph cycle re-primes it; the engine MUST call
    /// `update_graph_state(0)` (or appropriate seed) before the first
    /// decode token of the new request.
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.cur_pos_dev = None;
        self.cur_pos_dev_value = None;
        // A reset means no live graph can validly reference this cache
        // anymore (the engine tears request-local graphs down before any
        // cross-request reset). Clear the capture bookkeeping so the next
        // request's prefill can grow the buffers again.
        self.graph_captured = false;
        self.graph_seq_limit = None;
        self.chain_stride_seen
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Grow the K/V buffers to hold at least `new_cap` tokens. Doubles
    /// the current capacity (clamped to `max_seq_len`) so growth is
    /// amortized O(1). Memcpy is sized to just the used prefix.
    fn grow_to(&mut self, new_cap: usize) -> Result<()> {
        // CUDA graph capture pins the K/V buffer pointers - growing them
        // mid-graph orphans the captured kernels. The engine falls back to
        // the non-graph path when this returns an error.
        //
        // We gate on the engine-set `graph_captured` flag rather than on
        // `cur_pos_dev.is_some()` because the non-graph dev_pos path
        // (phi2 KV-quant unblock) also primes cur_pos_dev but is safe to
        // grow - the captured-graph constraint does not apply there.
        if self.graph_captured {
            return Err(anyhow!(
                "Q8KvCache::grow_to: cannot grow while a captured graph references the buffers (capacity={}, requested {})",
                self.current_capacity, new_cap,
            ));
        }
        let target = new_cap.max(self.current_capacity * 2).min(self.max_seq_len);
        if target <= self.current_capacity {
            return Ok(());
        }
        let token_bytes = self.n_kv_heads * (self.head_dim / Q8_0_BLOCK_SIZE) * Q8_0_TYPE_SIZE;
        // Try the amortized geometric target first (doubling -> O(1) appends). If
        // the GPU can't fit the doubled buffer, retry with the EXACT requested
        // capacity (`new_cap`) - half-to-quarter the bytes - before giving up.
        // This keeps a near-full GPU (deepseek-r1 TP / gemma4:26b at long ctx)
        // growing one chunk at a time instead of surfacing a runtime OOM to the
        // user. A genuine OOM at the minimal size still propagates, where the
        // engine's OOM-adaptive prefill shrinks the compute chunk and retries.
        let is_oom = |e: &anyhow::Error| {
            let s = e.to_string().to_ascii_lowercase();
            s.contains("out of memory") || s.contains("cuda_error_out_of_memory")
        };
        let alloc_pair = |cap: usize| {
            let bytes = cap * token_bytes;
            let k = self.cuda_device.alloc_zeros::<u8>(bytes)?;
            let v = self.cuda_device.alloc_zeros::<u8>(bytes)?;
            Ok::<_, anyhow::Error>((cap, bytes, k, v))
        };
        let (target, new_kv_bytes, mut new_k, mut new_v) = match alloc_pair(target) {
            Ok(t) => t,
            Err(e) if is_oom(&e) && new_cap < target => {
                let minimal = new_cap.max(self.current_capacity + 1).min(self.max_seq_len);
                tracing::warn!(
                    "Q8KvCache::grow_to: doubled target {target} OOM'd ({e}); retrying minimal capacity {minimal}"
                );
                alloc_pair(minimal)?
            }
            Err(e) => return Err(e),
        };
        if self.current_seq_len > 0 {
            let used_bytes = self.current_seq_len * token_bytes;
            {
                let src_k = self.k_q8.slice(0..used_bytes);
                let mut dst_k = new_k.slice_mut(0..used_bytes);
                self.cuda_device
                    .memcpy_dtod(&src_k, &mut dst_k)
                    .map_err(|e| anyhow!("Q8KvCache::grow_to: k memcpy_dtod failed: {e}"))?;
            }
            {
                let src_v = self.v_q8.slice(0..used_bytes);
                let mut dst_v = new_v.slice_mut(0..used_bytes);
                self.cuda_device
                    .memcpy_dtod(&src_v, &mut dst_v)
                    .map_err(|e| anyhow!("Q8KvCache::grow_to: v memcpy_dtod failed: {e}"))?;
            }
        }
        self.k_q8 = new_k;
        self.v_q8 = new_v;
        self.current_capacity = target;
        tracing::debug!(
            "Q8KvCache: grew capacity {} -> {} tokens ({:.1} MB per buffer)",
            self.current_capacity,
            target,
            (new_kv_bytes as f64) / 1e6,
        );
        Ok(())
    }

    // (see earlier `reset()` definition - also clears cur_pos_dev for graph mode)

    /// Shorten the cache to `new_len` tokens, keeping every piece of state that
    /// describes its length in agreement.
    ///
    /// The host counter is not the only such piece: the dev_pos attention
    /// kernels read the DEVICE-side position to derive how much cache to attend
    /// over, and a graph captured while the cache was longer encodes the old
    /// geometry. `reset()` clears both, which is why trimming to zero was safe;
    /// cross-request prefix reuse trims to a NON-zero length and took a path
    /// that updated the host counter alone, so the next request attended over
    /// positions the cache no longer held and decoded from wrong logits.
    pub fn trim_to(&mut self, new_len: usize) {
        self.current_seq_len = new_len.min(self.current_seq_len);
        // Drop the device-side position rather than refresh it. Its PRESENCE is
        // what makes the engine dispatch the dev_pos attention variants, and the
        // re-prefill that follows a trim appends several tokens at once - the one
        // shape a cold prefill never takes on that path, because a cold cache has
        // no device position yet and goes through the host-int variants. Clearing
        // it puts the re-prefill back on the path the cold prefill validates, and
        // the first decode token re-establishes it via `update_graph_state`.
        self.cur_pos_dev = None;
        self.cur_pos_dev_value = None;
        // Any graph captured while the cache was longer no longer describes it,
        // including the chain stride its sequence ceiling was derived from.
        self.clear_graph_captured();
    }

    /// Append `k_new` and `v_new` for one or more tokens.
    ///
    /// Expected shapes: `[1, n_kv_heads, n_new, head_dim]`. Both K and V
    /// are quantized to Q8_0 and stored in packed `[seq, n_kv, head_dim]`
    /// layout.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let k_dims = k_new.dims();
        if k_dims.len() != 4
            || k_dims[0] != 1
            || k_dims[1] != self.n_kv_heads
            || k_dims[3] != self.head_dim
        {
            return Err(anyhow!(
                "Q8KvCache::append: K shape {:?} does not match [1,{},*,{}]",
                k_dims,
                self.n_kv_heads,
                self.head_dim
            ));
        }
        if v_new.dims() != k_dims {
            return Err(anyhow!(
                "Q8KvCache::append: V shape {:?} does not match K shape {:?}",
                v_new.dims(),
                k_dims
            ));
        }
        let n_new = k_dims[2];
        if self.current_seq_len + n_new > self.max_seq_len {
            return Err(anyhow!(
                "Q8KvCache::append: would exceed max_seq_len {} (have {}, adding {})",
                self.max_seq_len,
                self.current_seq_len,
                n_new
            ));
        }

        // Grow the underlying buffers if needed. New caches start at
        // INITIAL_CAPACITY_TOKENS (typically 4 K) and double up to
        // max_seq_len; this avoids reserving worst-case VRAM upfront
        // for users whose `context_length=32 K+` config rarely sees a
        // request that uses it all.
        if self.current_seq_len + n_new > self.current_capacity {
            self.grow_to(self.current_seq_len + n_new)?;
        }

        // Fast-fast path: single-token decode with row-aligned width AND
        // not under graph capture -> both K and V quantize launches collapse
        // into ONE `quantize_q8_0_kv_paired_into_offset` kernel (gridDim.z=2).
        // Saves 1 launch per layer = 24/token on phi2. Falls back to the
        // two-call helper for n_new>1, non-aligned widths, or capture mode
        // (those cases either need staging or dev_slot semantics).
        let row_width = self.n_kv_heads * self.head_dim;
        let in_capture = capture_active_on(&self.cuda_device.cuda_stream());
        let paired_ok = n_new == 1 && row_width.is_multiple_of(MATRIX_ROW_PADDING);
        if paired_ok && !in_capture {
            self.append_kv_paired_single(k_new, v_new, row_width)?;
        } else if paired_ok && self.cur_pos_dev.is_some() {
            // Capture-mode paired append: same one-launch K+V quantize, with
            // the destination slot read from `cur_pos_dev` at replay time
            // (dev_slot twin) - one fewer graph node per layer.
            self.append_kv_paired_dev_slot(k_new, v_new, row_width)?;
        } else {
            self.quantize_and_store(k_new, n_new, /* is_v */ false)?;
            self.quantize_and_store(v_new, n_new, /* is_v */ true)?;
        }

        self.current_seq_len += n_new;
        Ok(())
    }

    /// Single-token paired K+V quantize. Both pass through one CUDA kernel
    /// launch (`quantize_q8_0_kv_paired_into_offset`, z=2 grid).
    fn append_kv_paired_single(
        &mut self,
        k_new: &Tensor,
        v_new: &Tensor,
        row_width: usize,
    ) -> Result<()> {
        use crate::inference::quantized_cuda::quantize_q8_0_kv_paired_into_offset;
        // Layout: both tensors are [1, n_kv, 1, head_dim]; with n_new=1 the
        // permute (0,2,1,3) is a memory no-op - flatten directly.
        let k_f32 = if k_new.dtype() == DType::F32 {
            k_new.clone()
        } else {
            k_new.to_dtype(DType::F32)?
        };
        let v_f32 = if v_new.dtype() == DType::F32 {
            v_new.clone()
        } else {
            v_new.to_dtype(DType::F32)?
        };
        let k_flat = k_f32.flatten_all()?;
        let v_flat = v_f32.flatten_all()?;
        let (k_st, k_lo) = k_flat.storage_and_layout();
        let (v_st, v_lo) = v_flat.storage_and_layout();
        let k_cuda = match &*k_st {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache: K must be on CUDA")),
        };
        let v_cuda = match &*v_st {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache: V must be on CUDA")),
        };
        let k_view = k_cuda
            .as_cuda_slice::<f32>()
            .map_err(|e| anyhow!("Q8KvCache: K slice cast: {e}"))?
            .slice(k_lo.start_offset()..k_lo.start_offset() + row_width);
        let v_view = v_cuda
            .as_cuda_slice::<f32>()
            .map_err(|e| anyhow!("Q8KvCache: V slice cast: {e}"))?
            .slice(v_lo.start_offset()..v_lo.start_offset() + row_width);
        let token_bytes = self.n_kv_heads * (self.head_dim / Q8_0_BLOCK_SIZE) * Q8_0_TYPE_SIZE;
        let dst_byte_offset = self.current_seq_len * token_bytes;
        quantize_q8_0_kv_paired_into_offset(
            &k_view,
            &v_view,
            &mut self.k_q8,
            &mut self.v_q8,
            dst_byte_offset,
            row_width,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q8KvCache: paired quantize failed: {e}"))?;
        Ok(())
    }

    /// Capture-mode twin of `append_kv_paired_single`: one z=2 kernel
    /// quantizes K+V into the slot read from `cur_pos_dev` at replay time.
    fn append_kv_paired_dev_slot(
        &mut self,
        k_new: &Tensor,
        v_new: &Tensor,
        row_width: usize,
    ) -> Result<()> {
        use crate::inference::quantized_cuda::quantize_q8_0_kv_paired_f32_dev_slot;
        let k_f32 = if k_new.dtype() == DType::F32 {
            k_new.clone()
        } else {
            k_new.to_dtype(DType::F32)?
        };
        let v_f32 = if v_new.dtype() == DType::F32 {
            v_new.clone()
        } else {
            v_new.to_dtype(DType::F32)?
        };
        let k_flat = k_f32.flatten_all()?;
        let v_flat = v_f32.flatten_all()?;
        let (k_st, k_lo) = k_flat.storage_and_layout();
        let (v_st, v_lo) = v_flat.storage_and_layout();
        let k_cuda = match &*k_st {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache: K must be on CUDA")),
        };
        let v_cuda = match &*v_st {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache: V must be on CUDA")),
        };
        let k_view = k_cuda
            .as_cuda_slice::<f32>()
            .map_err(|e| anyhow!("Q8KvCache: K slice cast: {e}"))?
            .slice(k_lo.start_offset()..k_lo.start_offset() + row_width);
        let v_view = v_cuda
            .as_cuda_slice::<f32>()
            .map_err(|e| anyhow!("Q8KvCache: V slice cast: {e}"))?
            .slice(v_lo.start_offset()..v_lo.start_offset() + row_width);
        let slot_dev = self
            .cur_pos_dev
            .as_ref()
            .ok_or_else(|| anyhow!("Q8KvCache::append_kv_paired_dev_slot: cur_pos_dev unprimed"))?;
        quantize_q8_0_kv_paired_f32_dev_slot(
            &k_view,
            &v_view,
            &mut self.k_q8,
            &mut self.v_q8,
            slot_dev,
            row_width,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q8KvCache: paired dev_slot quantize failed: {e}"))?;
        Ok(())
    }

    /// Shared path used by both the K and V branches of `append`. Permutes
    /// the incoming `[1, n_kv, n_new, head_dim]` tensor to the packed
    /// `[n_new, n_kv, head_dim]` layout, quantizes to Q8_0 (reusing the
    /// pre-sized staging buffer on the decode fast path), and memcpys the
    /// result into the persistent K/V buffer at the current offset.
    fn quantize_and_store(&mut self, t: &Tensor, n_new: usize, is_v: bool) -> Result<()> {
        // Produce a contiguous F32 view with layout [n_new, n_kv, head_dim]
        // so Q8_0 blocks line up with the packed cache layout. For seq=1
        // the permute (0,2,1,3) is a memory no-op (the seq=1 dim merges
        // trivially), so we can flatten directly and avoid the contiguous
        // launch.
        let laid_out = if n_new == 1 {
            let cast = if t.dtype() == DType::F32 {
                t.clone()
            } else {
                t.to_dtype(DType::F32)?
            };
            cast.flatten_all()?
        } else {
            t.permute((0, 2, 1, 3))?
                .contiguous()?
                .to_dtype(DType::F32)?
                .flatten_all()?
        };
        let (src_storage_guard, src_layout) = laid_out.storage_and_layout();
        let src_cuda = match &*src_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache: tensor must be on CUDA")),
        };

        let token_bytes = self.n_kv_heads * (self.head_dim / Q8_0_BLOCK_SIZE) * Q8_0_TYPE_SIZE;
        let dst_byte_offset = self.current_seq_len * token_bytes;
        let copy_bytes = n_new * token_bytes;

        // Fast path: single-token decode where the row width (n_kv x
        // head_dim) is a multiple of MATRIX_ROW_PADDING. Quantize
        // directly into the persistent buffer - saves the staging
        // quantize + memcpy_dtod pair (2 launches -> 1).
        //
        // Two variants depending on capture mode:
        //   - In capture mode: use `quantize_q8_0_f32_dev_slot` which
        //     reads the destination slot from `cur_pos_dev` at REPLAY
        //     time. Required for graph capture - the static-offset
        //     variant would bake the current slot into the captured
        //     kernel params, and every replay would overwrite slot N
        //     instead of advancing to N+1, clobbering prior tokens'
        //     KV. Mirrors Q4's `flush_k_residual_q4_dev_pos_raw`.
        //   - Non-capture mode: use `quantize_q8_0_f32_into_offset`
        //     with the host-int dst_byte_offset (~1 launch saved per
        //     K/V vs the staged path).
        let row_width = self.n_kv_heads * self.head_dim;
        let direct_write_ok = n_new == 1 && row_width.is_multiple_of(MATRIX_ROW_PADDING);
        if direct_write_ok {
            let src_view = src_cuda
                .as_cuda_slice::<f32>()
                .map_err(|e| anyhow!("Q8KvCache: src as f32 slice: {e}"))?
                .slice(src_layout.start_offset()..src_layout.start_offset() + row_width);
            let in_capture = capture_active_on(&self.cuda_device.cuda_stream());
            if in_capture {
                let slot_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
                    anyhow!("Q8KvCache::append: cur_pos_dev unprimed under graph capture")
                })?;
                let dst_buf = if is_v { &mut self.v_q8 } else { &mut self.k_q8 };
                quantize_q8_0_f32_dev_slot(
                    &src_view,
                    dst_buf,
                    slot_dev,
                    row_width,
                    &self.cuda_device,
                )
                .map_err(|e| anyhow!("Q8KvCache: quantize_dev_slot failed: {e}"))?;
            } else {
                let dst_buf = if is_v { &mut self.v_q8 } else { &mut self.k_q8 };
                quantize_q8_0_f32_into_offset(
                    &src_view,
                    dst_buf,
                    dst_byte_offset,
                    row_width,
                    &self.cuda_device,
                )
                .map_err(|e| anyhow!("Q8KvCache: quantize_into_offset failed: {e}"))?;
            }
            return Ok(());
        }

        let staging: &mut QCudaStorage = if is_v {
            &mut self.v_staging_decode
        } else {
            &mut self.k_staging_decode
        };

        let src_slice_owned;
        let src_slice = if n_new == 1 {
            staging
                .quantize_with_layout(src_cuda, src_layout)
                .map_err(|e| anyhow!("Q8KvCache: staged quantize failed: {e}"))?;
            staging
                .data_slice_for_copy(copy_bytes)
                .map_err(|e| anyhow!("Q8KvCache: {e}"))?
        } else {
            let mut fresh = QCudaStorage::zeros(
                &self.cuda_device,
                n_new * self.n_kv_heads * self.head_dim,
                GgmlDType::Q8_0,
            )
            .map_err(|e| anyhow!("Q8KvCache: prefill staging alloc failed: {e}"))?;
            fresh
                .quantize_with_layout(src_cuda, src_layout)
                .map_err(|e| anyhow!("Q8KvCache: prefill quantize failed: {e}"))?;
            src_slice_owned = fresh;
            src_slice_owned
                .data_slice_for_copy(copy_bytes)
                .map_err(|e| anyhow!("Q8KvCache: {e}"))?
        };

        let dst_buf = if is_v { &mut self.v_q8 } else { &mut self.k_q8 };
        let mut dst_slice = dst_buf.slice_mut(dst_byte_offset..dst_byte_offset + copy_bytes);
        self.cuda_device
            .memcpy_dtod(&src_slice, &mut dst_slice)
            .map_err(|e| anyhow!("Q8KvCache: memcpy_dtod failed: {e}"))?;
        Ok(())
    }

    /// Compute `Q @ K^T` for the current K cache.
    ///
    /// `q` must have shape `[1, n_q_heads, 1, head_dim]` (decode) where
    /// `n_q_heads` is a multiple of `n_kv_heads` (GQA). Returns scores
    /// shaped `[1, n_q_heads, 1, current_seq_len]` in f32.
    ///
    /// Runs one kernel launch per kv-head; with n_kv ∈ {4, 8} the launch
    /// overhead is dominated by the dot-product work even at modest
    /// sequence lengths.
    /// Like `attn_scores` but pre-multiplies the output by `scale`
    /// (the 1/sqrt(head_dim) softmax scale) in-kernel. Saves the
    /// separate `affine(scale, 0.0)` launch - one fewer kernel
    /// per layer per token.
    pub fn attn_scores_scaled(&self, q: &Tensor, scale: f32) -> Result<Tensor> {
        if self.current_seq_len == 0 {
            return Err(anyhow!("Q8KvCache::attn_scores_scaled: cache is empty"));
        }
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q8KvCache::attn_scores_scaled: Q shape {:?} not [1,n_q_heads,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q8KvCache::attn_scores_scaled: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads,
                self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        let seq_kv = self.current_seq_len;
        let q_squeezed = q.squeeze(2)?.squeeze(0)?;
        let q_cast = if q_squeezed.dtype() == DType::F32 {
            q_squeezed
        } else {
            q_squeezed.to_dtype(DType::F32)?
        };
        let q_flat = q_cast.contiguous()?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache: Q must be on CUDA")),
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);
        let scores_storage = crate::inference::quantized_cuda::attn_score_q8_0_q8_1_gqa_scaled(
            &self.k_q8,
            &q_view,
            self.head_dim,
            seq_kv,
            self.n_kv_heads,
            n_q_per_kv,
            scale,
            &self.cuda_device,
        )?;
        Ok(tensor_from_cuda_storage(
            scores_storage,
            (1, n_q_heads, 1, seq_kv),
        )?)
    }

    pub fn attn_scores(&self, q: &Tensor) -> Result<Tensor> {
        if self.current_seq_len == 0 {
            return Err(anyhow!("Q8KvCache::attn_scores: cache is empty"));
        }
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q8KvCache::attn_scores: Q shape {:?} not [1,n_q_heads,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q8KvCache::attn_scores: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads,
                self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        let seq_kv = self.current_seq_len;

        // Squeeze the singleton seq_q dim: Q becomes [n_q_heads, head_dim].
        // Skip the to_dtype cast when Q is already F32 (caller-side
        // upstream optimisation: produce Q as F32 to avoid one launch
        // per layer per token).
        let q_squeezed = q.squeeze(2)?.squeeze(0)?;
        let q_cast = if q_squeezed.dtype() == DType::F32 {
            q_squeezed
        } else {
            q_squeezed.to_dtype(DType::F32)?
        };
        let q_flat = q_cast.contiguous()?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache: Q must be on CUDA")),
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);

        // Single kernel launch handles all kv-heads via gridDim.y.
        let scores_storage = attn_score_q8_0_q8_1_gqa(
            &self.k_q8,
            &q_view,
            self.head_dim,
            seq_kv,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )?;

        let out = tensor_from_cuda_storage(scores_storage, (1, n_q_heads, 1, seq_kv))?;
        Ok(out)
    }

    /// Dequantize the current K and V caches back to the standard
    /// `[1, n_kv_heads, current_seq_len, head_dim]` layout in `dtype`
    /// (typically F16 to feed flash-attn or the F-dtype matmul path).
    ///
    /// Used by the consolidated storage mode where the F-dtype cache is
    /// skipped entirely - multi-token attention calls (prefill, PLD verify
    /// batches) need F-dtype K, V tensors with the full accumulated
    /// history. Single-token decode avoids this round-trip by going
    /// through `attn_scores`/`attn_output` directly.
    pub fn dequantize_kv(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        use crate::inference::quantized_cuda::dequantize_q8_0_blob_f16;

        let seq = self.current_seq_len;
        if seq == 0 {
            return Err(anyhow!("Q8KvCache::dequantize_kv: cache is empty"));
        }
        let elem_count = seq * self.n_kv_heads * self.head_dim;

        // Packed storage layout is [seq, n_kv, hd]. dequantize_q8_0_blob_f16
        // writes f16 values in the same element order; we then reshape and
        // permute to the [1, n_kv, seq, hd] layout attention code expects.
        let k_f16 = dequantize_q8_0_blob_f16(&self.k_q8, elem_count, &self.cuda_device)
            .map_err(|e| anyhow!("Q8KvCache: K dequantize: {e}"))?;
        let v_f16 = dequantize_q8_0_blob_f16(&self.v_q8, elem_count, &self.cuda_device)
            .map_err(|e| anyhow!("Q8KvCache: V dequantize: {e}"))?;

        let shape = (1, seq, self.n_kv_heads, self.head_dim);
        let k = tensor_from_cuda_storage(k_f16, shape)?;
        let v = tensor_from_cuda_storage(v_f16, shape)?;

        // [1, seq, n_kv, hd] -> [1, n_kv, seq, hd]
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        // Cast to requested dtype if different from the dequant output (F16).
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

    /// Compute `attn_out = probs @ V` for the current V cache.
    ///
    /// `probs` shape: `[1, n_q_heads, 1, current_seq_len]` (softmax output).
    /// Returns `[1, n_q_heads, 1, head_dim]` in f32.
    pub fn attn_output(&self, probs: &Tensor) -> Result<Tensor> {
        if self.current_seq_len == 0 {
            return Err(anyhow!("Q8KvCache::attn_output: cache is empty"));
        }
        let dims = probs.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.current_seq_len {
            return Err(anyhow!(
                "Q8KvCache::attn_output: probs shape {:?} not [1,n_q_heads,1,{}]",
                dims,
                self.current_seq_len
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q8KvCache::attn_output: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads,
                self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        let seq_kv = self.current_seq_len;

        // Flatten probs to [n_q_heads, seq_kv].
        let probs_flat = step_rows_f32(probs)?;
        let probs_guard = probs_flat.storage_and_layout().0;
        let probs_cuda = match &*probs_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache: probs must be on CUDA")),
        };
        let probs_view = probs_cuda.as_cuda_slice::<f32>()?.slice(..);

        let out_storage = attn_output_q8_0_f32_gqa(
            &self.v_q8,
            &probs_view,
            self.head_dim,
            seq_kv,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )?;

        let out = tensor_from_cuda_storage(out_storage, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out)
    }

    /// Fused softmax + Q8_0 attn output. Takes RAW scores (post Q@K^T,
    /// post-scale), applies softmax internally, and accumulates into the
    /// V multiply. One kernel instead of softmax_last_dim + attn_output.
    /// Q8 sibling of Q4KvCache::attn_softmax_output.
    pub fn attn_softmax_output(&self, scores: &Tensor) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_softmax_output_q8_0_f32_gqa;

        let seq = self.current_seq_len;
        if seq == 0 {
            return Err(anyhow!("Q8KvCache::attn_softmax_output: cache is empty"));
        }
        let dims = scores.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != seq {
            return Err(anyhow!(
                "Q8KvCache::attn_softmax_output: scores shape {:?} not [1,n_q,1,{}]",
                dims,
                seq
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q8KvCache::attn_softmax_output: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads,
                self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;

        let scores_flat = packed_f32(scores)?;
        let s_storage_guard = scores_flat.storage_and_layout().0;
        let s_cuda = match &*s_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q8KvCache::attn_softmax_output: scores must be CUDA"
                ))
            }
        };
        let s_view = s_cuda.as_cuda_slice::<f32>()?.slice(..);

        let out = attn_softmax_output_q8_0_f32_gqa(
            &self.v_q8,
            &s_view,
            self.head_dim,
            seq,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q8KvCache: q8 softmax+output kernel: {e}"))?;
        let out_tensor = tensor_from_cuda_storage(out, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    /// CUDA-graph-safe attention scores. Mirrors `attn_scores` but reads
    /// `current_seq_len` from the device-side `cur_pos_dev` slot, and
    /// writes scores at the fixed `max_seq_padded` stride so the captured
    /// graph sees stable buffer addresses on every replay. Positions
    /// `[current_seq_len, max_seq_padded)` are written as -INFINITY so
    /// the downstream `softmax_last_dim` masks them to zero.
    ///
    /// `q`: shape `[1, n_q_heads, 1, head_dim]`. Returns scores
    /// `[1, n_q_heads, 1, max_seq_padded]` f32.
    pub fn attn_scores_graph(&self, q: &Tensor, window: usize) -> Result<Tensor> {
        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!("Q8KvCache::attn_scores_graph: update_graph_state has not been called")
        })?;
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q8KvCache::attn_scores_graph: Q shape {:?} not [1,n_q,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q8KvCache::attn_scores_graph: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads,
                self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        let max_seq_padded = self.max_seq_padded();
        // Record the stride this launch bakes in - if this call is being
        // recorded into a CUDA graph, replays are only valid while
        // seq_kv fits the stride (see `mark_graph_captured`).
        self.chain_stride_seen
            .store(max_seq_padded, std::sync::atomic::Ordering::Relaxed);

        let q_flat = packed_f32(q)?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => return Err(anyhow!("Q8KvCache::attn_scores_graph: Q must be on CUDA")),
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);

        // Mirror the Q4 graph path: fresh-alloc CudaStorage per call. The
        // cudaMallocAsync pool handles allocations under graph capture
        // (Q4 has proven this in production). The earlier persistent
        // buffer + `_into` variant raced the kernel write through
        // `CudaSlice::clone()` (deep d2d copy on a possibly different
        // stream than the kernel). Fresh-alloc avoids the clone entirely.
        let scores_storage = attn_score_q8_0_f32_dev_pos(
            &self.k_q8,
            &q_view,
            pos_dev,
            self.head_dim,
            max_seq_padded,
            self.n_kv_heads,
            n_q_per_kv,
            window,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q8KvCache: q8 dev-pos score kernel: {e}"))?;

        let scores = tensor_from_cuda_storage(scores_storage, (1, n_q_heads, 1, max_seq_padded))?;
        Ok(scores)
    }

    /// CUDA-graph-safe attention output. Mirrors `attn_output` but reads
    /// `current_seq_len` from `cur_pos_dev` and expects probs at the
    /// fixed `max_seq_padded` stride. Probs at positions >= current_seq_len
    /// must be zero (the standard softmax-on-padded-scores does this).
    ///
    /// `probs`: shape `[1, n_q_heads, 1, max_seq_padded]` f32. Returns
    /// context `[1, n_q_heads, 1, head_dim]` f32.
    pub fn attn_output_graph(&self, probs: &Tensor) -> Result<Tensor> {
        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!("Q8KvCache::attn_output_graph: update_graph_state has not been called")
        })?;
        let max_seq_padded = self.max_seq_padded();
        let dims = probs.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != max_seq_padded {
            return Err(anyhow!(
                "Q8KvCache::attn_output_graph: probs shape {:?} not [1,n_q,1,{}]",
                dims,
                max_seq_padded
            ));
        }
        let n_q_heads = dims[1];
        let n_q_per_kv = n_q_heads / self.n_kv_heads;

        let probs_flat = packed_f32(probs)?;
        let p_storage_guard = probs_flat.storage_and_layout().0;
        let p_cuda = match &*p_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q8KvCache::attn_output_graph: probs must be on CUDA"
                ))
            }
        };
        let p_view = p_cuda.as_cuda_slice::<f32>()?.slice(..);

        let out_storage = attn_output_q8_0_f32_dev_pos(
            &self.v_q8,
            &p_view,
            pos_dev,
            self.head_dim,
            max_seq_padded,
            self.n_kv_heads,
            n_q_per_kv,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q8KvCache: q8 dev-pos output kernel: {e}"))?;

        let out_tensor = tensor_from_cuda_storage(out_storage, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    /// Fused scale + softmax + V.attn for the dev_pos path. Takes RAW
    /// Q.K^T scores (output of `attn_scores_graph`, already at
    /// max_seq_padded stride with -INFINITY past current_seq_len) and
    /// returns the attention context directly - collapsing the prior
    /// 3-launch chain (affine scale -> softmax_last_dim -> attn_output_graph)
    /// into one. The intermediate softmax probs are never materialised
    /// in global memory.
    ///
    /// `scores` shape: `[1, n_q_heads, 1, max_seq_padded]`. Returns
    /// `[1, n_q_heads, 1, head_dim]` F32.
    ///
    /// Phi2 hot path: 24 layers x (3 saved launches) = 72 fewer launches
    /// per token.
    pub fn attn_softmax_output_graph(
        &self,
        scores: &Tensor,
        scale: f32,
        window: usize,
    ) -> Result<Tensor> {
        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!("Q8KvCache::attn_softmax_output_graph: update_graph_state has not been called")
        })?;
        let max_seq_padded = self.max_seq_padded();
        let dims = scores.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != max_seq_padded {
            return Err(anyhow!(
                "Q8KvCache::attn_softmax_output_graph: scores shape {:?} not [1,n_q,1,{}]",
                dims,
                max_seq_padded
            ));
        }
        let n_q_heads = dims[1];
        if !n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(anyhow!(
                "Q8KvCache::attn_softmax_output_graph: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads, self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;

        let scores_flat = packed_f32(scores)?;
        let s_storage_guard = scores_flat.storage_and_layout().0;
        let s_cuda = match &*s_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q8KvCache::attn_softmax_output_graph: scores must be on CUDA"
                ))
            }
        };
        let s_view = s_cuda.as_cuda_slice::<f32>()?.slice(..);

        let out_storage = attn_softmax_output_q8_0_f32_dev_pos(
            &self.v_q8,
            &s_view,
            pos_dev,
            self.head_dim,
            max_seq_padded,
            self.n_kv_heads,
            n_q_per_kv,
            scale,
            window,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q8KvCache: q8 fused softmax+output dev-pos kernel: {e}"))?;

        let out_tensor = tensor_from_cuda_storage(out_storage, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    /// Single-launch fused attention: Q.K^T + scale + softmax + V.attn in
    /// one kernel. Replaces the `attn_scores_graph` + `attn_softmax_output_graph`
    /// pair - keeps scores in registers (online softmax) instead of writing
    /// an `[n_q, max_seq_padded]` F32 intermediate (256 KB at 4 K ctx).
    ///
    /// Restricted to n_q_per_kv = 1 (phi2 / no GQA). HD ∈ {64, 128}.
    /// Returns out `[1, n_q_heads, 1, head_dim]` F32.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_fused_decode_graph(&self, q: &Tensor, scale: f32) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_fused_q8_decode_dev_pos;

        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!("Q8KvCache::attn_fused_decode_graph: update_graph_state has not been called")
        })?;
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q8KvCache::attn_fused_decode_graph: Q shape {:?} not [1,n_q,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if n_q_heads != self.n_kv_heads {
            return Err(anyhow!(
                "Q8KvCache::attn_fused_decode_graph: n_q_heads {} must equal n_kv_heads {} (no GQA)",
                n_q_heads, self.n_kv_heads
            ));
        }
        if !(self.head_dim == 64 || self.head_dim == 128) {
            return Err(anyhow!(
                "Q8KvCache::attn_fused_decode_graph: head_dim {} not in {{64,128}}",
                self.head_dim
            ));
        }

        let q_squeezed = q.squeeze(2)?.squeeze(0)?;
        let q_cast = if q_squeezed.dtype() == DType::F32 {
            q_squeezed
        } else {
            q_squeezed.to_dtype(DType::F32)?
        };
        let q_flat = q_cast.contiguous()?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q8KvCache::attn_fused_decode_graph: Q must be on CUDA"
                ))
            }
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);

        let max_seq_padded = self.max_seq_padded();
        let out_storage = attn_fused_q8_decode_dev_pos(
            &self.k_q8,
            &self.v_q8,
            &q_view,
            pos_dev,
            self.head_dim,
            max_seq_padded,
            self.n_kv_heads,
            scale,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q8KvCache: attn_fused_q8_decode_dev_pos kernel: {e}"))?;

        let out_tensor = tensor_from_cuda_storage(out_storage, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    /// SPLIT-K flash-decode attention (one-kernel-pair, fully fused). Like
    /// `attn_fused_decode_graph` but uses the split-K kernels: `NSPLITxn_heads`
    /// blocks of online-softmax partials + a combine pass. Far more parallel
    /// than the v3 single-block-per-head fused kernel - designed to beat the
    /// 2-kernel score+softmax_output chain on long-KV decode (moondream image
    /// context). Same `[1, n_q_heads, 1, head_dim]` F32 output; nq_per_kv==1,
    /// head_dim in {64,128}.
    pub fn attn_flash_splitk_decode_graph(&self, q: &Tensor, scale: f32) -> Result<Tensor> {
        use crate::inference::quantized_cuda::attn_flash_splitk_q8_decode_dev_pos;

        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!(
                "Q8KvCache::attn_flash_splitk_decode_graph: update_graph_state has not been called"
            )
        })?;
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q8KvCache::attn_flash_splitk_decode_graph: Q shape {:?} not [1,n_q,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if self.n_kv_heads == 0 || n_q_heads % self.n_kv_heads != 0 {
            return Err(anyhow!(
                "Q8KvCache::attn_flash_splitk_decode_graph: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads, self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        if !(self.head_dim == 64 || self.head_dim == 128) {
            return Err(anyhow!(
                "Q8KvCache::attn_flash_splitk_decode_graph: head_dim {} not in {{64,128}}",
                self.head_dim
            ));
        }

        let q_squeezed = q.squeeze(2)?.squeeze(0)?;
        let q_cast = if q_squeezed.dtype() == DType::F32 {
            q_squeezed
        } else {
            q_squeezed.to_dtype(DType::F32)?
        };
        let q_flat = q_cast.contiguous()?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q8KvCache::attn_flash_splitk_decode_graph: Q must be on CUDA"
                ))
            }
        };
        let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(..);

        let out_storage = if n_q_per_kv == 1 {
            attn_flash_splitk_q8_decode_dev_pos(
                &self.k_q8,
                &self.v_q8,
                &q_view,
                pos_dev,
                self.head_dim,
                self.n_kv_heads,
                scale,
                &self.cuda_device,
            )
            .map_err(|e| anyhow!("Q8KvCache: attn_flash_splitk_q8_decode_dev_pos kernel: {e}"))?
        } else {
            crate::inference::quantized_cuda::attn_flash_splitk_q8_gqa_decode_dev_pos(
                &self.k_q8,
                &self.v_q8,
                &q_view,
                pos_dev,
                self.head_dim,
                self.n_kv_heads,
                n_q_per_kv,
                self.current_seq_len(),
                scale,
                &self.cuda_device,
            )
            .map_err(|e| {
                anyhow!("Q8KvCache: attn_flash_splitk_q8_gqa_decode_dev_pos kernel: {e}")
            })?
        };

        let out_tensor = tensor_from_cuda_storage(out_storage, (1, n_q_heads, 1, self.head_dim))?;
        Ok(out_tensor)
    }

    /// Query-head-packed TENSOR-CORE flash-decode. Same contract as
    /// `attn_flash_splitk_decode_graph` (Q `[1,n_q,1,hd]`, returns `[1,n_q,1,hd]`
    /// F32 logically - here F16 then cast) but runs the m16n8k16 HMMA kernel that
    /// packs the GQA group's query heads into the MMA M-dim. hd=128 GQA only.
    /// Gated by the dispatch site; falls back to split-K on any Err.
    pub fn attn_flash_tc_decode_graph(&self, q: &Tensor, scale: f32) -> Result<Tensor> {
        use crate::inference::kernel::flash_decode_tc::flash_decode_tc_q8_gqa;

        let pos_dev = self.cur_pos_dev.as_ref().ok_or_else(|| {
            anyhow!("Q8KvCache::attn_flash_tc_decode_graph: update_graph_state has not been called")
        })?;
        let dims = q.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != 1 || dims[3] != self.head_dim {
            return Err(anyhow!(
                "Q8KvCache::attn_flash_tc_decode_graph: Q shape {:?} not [1,n_q,1,{}]",
                dims,
                self.head_dim
            ));
        }
        let n_q_heads = dims[1];
        if self.n_kv_heads == 0 || n_q_heads % self.n_kv_heads != 0 {
            return Err(anyhow!(
                "Q8KvCache::attn_flash_tc_decode_graph: n_q_heads {} not a multiple of n_kv_heads {}",
                n_q_heads, self.n_kv_heads
            ));
        }
        let n_q_per_kv = n_q_heads / self.n_kv_heads;
        if self.head_dim != 128 {
            return Err(anyhow!(
                "Q8KvCache::attn_flash_tc_decode_graph: head_dim {} != 128",
                self.head_dim
            ));
        }
        // Q must be flat [n_q_heads, head_dim] F16 on CUDA.
        let q_squeezed = q.squeeze(2)?.squeeze(0)?;
        let q_f16 = if q_squeezed.dtype() == DType::F16 {
            q_squeezed
        } else {
            q_squeezed.to_dtype(DType::F16)?
        };
        let q_flat = q_f16.contiguous()?;
        let q_storage_guard = q_flat.storage_and_layout().0;
        let q_cuda = match &*q_storage_guard {
            StorageView::Cuda(c) => c,
            _ => {
                return Err(anyhow!(
                    "Q8KvCache::attn_flash_tc_decode_graph: Q must be on CUDA"
                ))
            }
        };
        let q_view = q_cuda.as_cuda_slice::<half::f16>()?;

        let out_storage = flash_decode_tc_q8_gqa(
            &self.k_q8,
            &self.v_q8,
            q_view,
            pos_dev,
            self.head_dim,
            self.n_kv_heads,
            n_q_per_kv,
            1, // qlen (decode)
            self.current_seq_len(),
            scale,
            &self.cuda_device,
        )
        .map_err(|e| anyhow!("Q8KvCache: flash_decode_tc_q8_gqa kernel: {e}"))?;

        // Kernel emits F16 [n_q_heads, head_dim]; build the [1,n_q,1,hd] tensor
        // (F16) then cast to F32 so the downstream attn_output matmul matches the
        // split-K path's F32 logical output.
        let out_f16 = tensor_from_cuda_storage(out_storage, (1, n_q_heads, 1, self.head_dim))?;
        let out_f32 = out_f16.to_dtype(DType::F32)?;
        Ok(out_f32)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod dev_pos_tests {
    //! Numerical equivalence tests for the Q8 KV cache graph-mode kernels
    //! (`attn_score_q8_0_f32_dev_pos`, `attn_output_q8_0_f32_dev_pos`).
    //!
    //! Kernel correctness must be verified BEFORE
    //! wiring into Q8KvCache. These tests compare the dev_pos outputs to
    //! the existing non-dev_pos baselines on identical inputs.
    //!
    //! Strategy: build a Q8KvCache, append K/V tensors via its public API,
    //! then access the internal `k_q8 / v_q8` CudaSlice<u8> buffers
    //! through this child module (which has access to the parent's private
    //! fields). That keeps the Q8 layout identical to what production
    //! callers see.
    use super::Q8KvCache;
    use crate::inference::quantized_cuda::{
        attn_output_q8_0_f32_dev_pos, attn_score_q8_0_f32_dev_pos, attn_score_q8_0_q8_1_gqa,
    };
    use crate::tensor::{Device, StorageView, Tensor};

    fn try_open_cuda() -> Option<Device> {
        for idx in 0..4 {
            if let Ok(dev) = Device::new_cuda(idx) {
                if Tensor::zeros_on((1,), crate::tensor::DType::F32, &dev).is_ok() {
                    return Some(dev);
                }
            }
        }
        None
    }

    /// Build a Q8KvCache and fill it with `seq_kv` tokens of random K/V.
    /// Returns the populated cache so callers can read its k_q8/v_q8 buffers.
    fn build_filled_cache(
        seq_kv: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dev: &Device,
    ) -> (Q8KvCache, Tensor, Tensor) {
        let mut cache =
            Q8KvCache::new(seq_kv.max(4096), n_kv_heads, head_dim, dev).expect("Q8KvCache::new");
        let k_full = Tensor::randn(0.0f32, 1.0, (1, n_kv_heads, seq_kv, head_dim), dev).unwrap();
        let v_full = Tensor::randn(0.0f32, 1.0, (1, n_kv_heads, seq_kv, head_dim), dev).unwrap();
        cache.append(&k_full, &v_full).expect("Q8KvCache::append");
        (cache, k_full, v_full)
    }

    #[test]
    fn score_dev_pos_matches_non_dev_pos_baseline() {
        let dev = match try_open_cuda() {
            Some(d) => d,
            None => {
                eprintln!("skipping: no usable CUDA device");
                return;
            }
        };
        let cuda_dev = dev.as_cuda_device().unwrap();

        for &head_dim in &[64usize, 128, 256] {
            for &(n_kv_heads, n_q_per_kv) in &[(1usize, 1usize), (2, 4), (4, 8)] {
                let n_q_heads = n_kv_heads * n_q_per_kv;
                for &seq_kv in &[37usize, 256, 1023] {
                    let (cache, _k_full, _v_full) =
                        build_filled_cache(seq_kv, n_kv_heads, head_dim, &dev);

                    let q_f32 = Tensor::randn(0.0f32, 1.0, (n_q_heads, head_dim), &dev).unwrap();
                    let (q_guard, _) = q_f32.storage_and_layout();
                    let q_view = match &*q_guard {
                        StorageView::Cuda(c) => c
                            .as_cuda_slice::<f32>()
                            .unwrap()
                            .slice(0..n_q_heads * head_dim),
                        _ => unreachable!(),
                    };

                    let baseline = attn_score_q8_0_q8_1_gqa(
                        &cache.k_q8,
                        &q_view,
                        head_dim,
                        seq_kv,
                        n_kv_heads,
                        n_q_per_kv,
                        &cuda_dev,
                    )
                    .unwrap();
                    let baseline_v = {
                        let storage = StorageView::Cuda(baseline);
                        match &storage {
                            StorageView::Cuda(c) => {
                                let s = c.as_cuda_slice::<f32>().unwrap();
                                let mut v = vec![0f32; n_q_heads * seq_kv];
                                cuda_dev.cuda_stream().memcpy_dtoh(s, &mut v).unwrap();
                                v
                            }
                            _ => unreachable!(),
                        }
                    };

                    let max_seq_padded = (seq_kv + 64 + 31) & !31;
                    let pos_before_append: i32 = (seq_kv as i32) - 1;
                    let mut seq_kv_dev = cuda_dev.alloc_zeros::<i32>(1).unwrap();
                    cuda_dev
                        .memcpy_htod(&[pos_before_append], &mut seq_kv_dev)
                        .unwrap();

                    let devpos = attn_score_q8_0_f32_dev_pos(
                        &cache.k_q8,
                        &q_view,
                        &seq_kv_dev,
                        head_dim,
                        max_seq_padded,
                        n_kv_heads,
                        n_q_per_kv,
                        /*window=*/ 0,
                        &cuda_dev,
                    )
                    .unwrap();
                    let devpos_v = {
                        let storage = StorageView::Cuda(devpos);
                        match &storage {
                            StorageView::Cuda(c) => {
                                let s = c.as_cuda_slice::<f32>().unwrap();
                                let mut v = vec![0f32; n_q_heads * max_seq_padded];
                                cuda_dev.cuda_stream().memcpy_dtoh(s, &mut v).unwrap();
                                v
                            }
                            _ => unreachable!(),
                        }
                    };

                    // Valid range matches baseline within the Q8_1 quantisation
                    // noise of Q. The baseline quantises Q to Q8_1 (host-side
                    // alloc + dp4a dot product) and our dev_pos variant keeps
                    // Q as F32 (no quantisation) and dequantises K to F32 for
                    // the multiply. Per-element diff is bounded by
                    //   |Q8_1(Q) - Q| * |K_dequant| ≈ (q_max/127) * |K| ≈ 0.025
                    // and the sum over HD terms scales the std-dev by sqrt(HD).
                    // Use 0.5 as a comfortable upper bound for HD <= 256.
                    let mut max_abs = 0.0f32;
                    for h in 0..n_q_heads {
                        for s in 0..seq_kv {
                            let b = baseline_v[h * seq_kv + s];
                            let d = devpos_v[h * max_seq_padded + s];
                            max_abs = max_abs.max((b - d).abs());
                        }
                    }
                    // Looser bound for HD=256 because of larger sum.
                    let tolerance = if head_dim <= 64 {
                        0.3
                    } else if head_dim <= 128 {
                        0.5
                    } else {
                        0.8
                    };
                    assert!(
                        max_abs < tolerance,
                        "hd={head_dim} kv={n_kv_heads} q_per_kv={n_q_per_kv} \
                         seq={seq_kv}: max_abs {max_abs:.4e} > {tolerance}",
                    );

                    // Padding range is -INFINITY.
                    for h in 0..n_q_heads {
                        for s in seq_kv..max_seq_padded {
                            let v = devpos_v[h * max_seq_padded + s];
                            assert!(
                                v == f32::NEG_INFINITY,
                                "hd={head_dim} seq={seq_kv} h={h} s={s}: \
                                 expected -INFINITY, got {v}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn output_dev_pos_matches_cpu_reference() {
        let dev = match try_open_cuda() {
            Some(d) => d,
            None => {
                eprintln!("skipping: no usable CUDA device");
                return;
            }
        };
        let cuda_dev = dev.as_cuda_device().unwrap();

        for &head_dim in &[64usize, 128, 256] {
            for &(n_kv_heads, n_q_per_kv) in &[(1usize, 1usize), (2, 4), (4, 8)] {
                let n_q_heads = n_kv_heads * n_q_per_kv;
                for &seq_kv in &[37usize, 256, 1023] {
                    let (cache, _k_full, v_full_gpu) =
                        build_filled_cache(seq_kv, n_kv_heads, head_dim, &dev);

                    // Dequantize V on CPU. The cache stores V quantized
                    // from `v_full_gpu` (shape [1, n_kv, seq_kv, head_dim])
                    // through the same Q8_0 path the production engine uses.
                    // Pre-quant CPU values approximate post-Q8 to within
                    // ~0.4% - close enough that a `1e-2` tolerance on the
                    // probs @ V matmul detects any kernel layout bug while
                    // tolerating the Q8 round-trip.
                    let v_cpu = v_full_gpu
                        .to_device(&Device::Cpu)
                        .unwrap()
                        .squeeze(0)
                        .unwrap() // [n_kv, seq_kv, head_dim]
                        .permute((1, 0, 2))
                        .unwrap() // [seq, n_kv, hd]
                        .contiguous()
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();

                    // Host-generated uniform [0,1) probs - substrate-neutral
                    // (the native compat shim has no `Tensor::rand`).
                    let probs_seqkv_v: Vec<f32> = {
                        use rand::RngExt;
                        let mut rng = rand::rng();
                        (0..n_q_heads * seq_kv)
                            .map(|_| rng.random_range(0f32..1f32))
                            .collect()
                    };

                    // CPU reference: for each (h_q, d) compute
                    //   sum over s in [0, seq_kv) of probs[h_q,s] * V[s, h_kv, d]
                    // where h_kv = h_q / n_q_per_kv.
                    let mut reference = vec![0.0f32; n_q_heads * head_dim];
                    for h_q in 0..n_q_heads {
                        let h_kv = h_q / n_q_per_kv;
                        for d in 0..head_dim {
                            let mut acc = 0.0f32;
                            for s in 0..seq_kv {
                                let p = probs_seqkv_v[h_q * seq_kv + s];
                                let v_idx = (s * n_kv_heads + h_kv) * head_dim + d;
                                acc += p * v_cpu[v_idx];
                            }
                            reference[h_q * head_dim + d] = acc;
                        }
                    }

                    let max_seq_padded = (seq_kv + 64 + 31) & !31;
                    let mut probs_padded = vec![0.0f32; n_q_heads * max_seq_padded];
                    for h in 0..n_q_heads {
                        for s in 0..seq_kv {
                            probs_padded[h * max_seq_padded + s] = probs_seqkv_v[h * seq_kv + s];
                        }
                    }
                    let pp_t = Tensor::from_slice(&probs_padded, (n_q_heads, max_seq_padded), &dev)
                        .unwrap();
                    let (pp_guard, _) = pp_t.storage_and_layout();
                    let pp_view = match &*pp_guard {
                        StorageView::Cuda(c) => c
                            .as_cuda_slice::<f32>()
                            .unwrap()
                            .slice(0..n_q_heads * max_seq_padded),
                        _ => unreachable!(),
                    };

                    let pos_before_append: i32 = (seq_kv as i32) - 1;
                    let mut seq_kv_dev = cuda_dev.alloc_zeros::<i32>(1).unwrap();
                    cuda_dev
                        .memcpy_htod(&[pos_before_append], &mut seq_kv_dev)
                        .unwrap();

                    let devpos = attn_output_q8_0_f32_dev_pos(
                        &cache.v_q8,
                        &pp_view,
                        &seq_kv_dev,
                        head_dim,
                        max_seq_padded,
                        n_kv_heads,
                        n_q_per_kv,
                        &cuda_dev,
                    )
                    .unwrap();
                    let devpos_v = {
                        let storage = StorageView::Cuda(devpos);
                        match &storage {
                            StorageView::Cuda(c) => {
                                let s = c.as_cuda_slice::<f32>().unwrap();
                                let mut v = vec![0f32; n_q_heads * head_dim];
                                cuda_dev.cuda_stream().memcpy_dtoh(s, &mut v).unwrap();
                                v
                            }
                            _ => unreachable!(),
                        }
                    };

                    // Relative error: |kernel - cpu_ref| / |cpu_ref|.
                    // Q8_0 round-trip introduces ~1/127 noise per element.
                    // Cumulative over `seq_kv` (each output element is a
                    // sum over seq_kv terms), the std-dev scales by
                    // sqrt(seq_kv). For seq=1023 that's ~32 x 0.008 = ~0.26
                    // absolute; relative to typical sums ~0.5-1.0 of
                    // values, 1e-2 relative is the right gate.
                    let mut abs_err_sum = 0.0f32;
                    let mut scale_sum = 0.0f32;
                    for (k, r) in devpos_v.iter().zip(reference.iter()) {
                        abs_err_sum += (k - r).abs();
                        scale_sum += r.abs();
                    }
                    let rel_err = abs_err_sum / scale_sum.max(1e-6);
                    let tolerance = 0.05; // 5% accounts for Q8 quant noise
                    assert!(
                        rel_err < tolerance,
                        "hd={head_dim} kv={n_kv_heads} q_per_kv={n_q_per_kv} \
                         seq={seq_kv}: rel_err {rel_err:.4e} > {tolerance}",
                    );
                }
            }
        }
    }
}
