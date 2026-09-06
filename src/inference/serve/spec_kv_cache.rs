//! Pre-allocated KV Cache
//!
//! A pre-allocated KV cache with O(1) append using slice_set.
//! Supports trim_to for KV cache management.

#[cfg(feature = "cuda")]
use crate::tensor::cuda_ext::CudaSlice;
use crate::tensor::{DType, Device, Result, Tensor};

/// A pre-allocated, trimmable KV cache.
///
/// Stores K and V tensors with shape [batch, n_kv_heads, seq_len, head_dim].
/// Pre-allocates for max_seq_len and uses in-place writes (slice_set) for O(1) append.
/// Trimming via `trim_to()` just updates the sequence length counter (O(1)).
#[derive(Debug)]
/// A sequence's K and V copied out of a `SpecKvCache`: a real copy, so the live
/// buffers can move on.
pub struct SpecKvSnapshot {
    k: Tensor,
    v: Tensor,
    len: usize,
}

impl SpecKvSnapshot {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The element dtype of the stored K/V, so a disk manifest can record it and a cold
    /// import rebuild tensors the cache will accept.
    pub fn dtype(&self) -> DType {
        self.k.dtype()
    }
    /// Device memory the stored K and V occupy.
    pub fn device_bytes(&self) -> u64 {
        let per = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        per(&self.k) + per(&self.v)
    }
    /// The stored K and V, `[1, n_kv, len, head_dim]`, for re-appending into a cache.
    pub fn kv(&self) -> (&Tensor, &Tensor) {
        (&self.k, &self.v)
    }
    /// A snapshot holding already-built device tensors `[1, n_kv, len, head_dim]`, as a
    /// quantised cache hands back when dequantised.
    pub fn from_kv_tensors(k: Tensor, v: Tensor, len: usize) -> Self {
        Self { k, v, len }
    }

    /// Tokens `[from, to)` as host f32 rows, token-major `[n, n_kv, head_dim]`, K then V.
    pub fn host_rows(&self, from: usize, to: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = to - from;
        let rows = |t: &Tensor| -> Result<Vec<f32>> {
            let t = if t.dims().len() == 4 {
                t.narrow(2, from, n)?.permute((0, 2, 1, 3))?
            } else {
                t.narrow(1, from, n)?
            };
            t.contiguous()?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        Ok((rows(&self.k)?, rows(&self.v)?))
    }

    /// A snapshot built from host rows, shaped and placed like `like`'s buffers.
    pub fn from_host_rows(
        like: &Tensor,
        k: &[f32],
        v: &[f32],
        n: usize,
        n_kv: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let build = |rows: &[f32]| -> Result<Tensor> {
            let t = Tensor::from_slice(rows, (1, n, n_kv, head_dim), &Device::Cpu)?;
            let t = if like.dims().len() == 4 {
                t.permute((0, 2, 1, 3))?.contiguous()?
            } else {
                t.reshape((1, n, n_kv * head_dim))?
            };
            t.to_dtype(like.dtype())?.to_device(&like.device())
        };
        Ok(Self {
            k: build(k)?,
            v: build(v)?,
            len: n,
        })
    }
}

pub struct SpecKvCache {
    k: Option<Tensor>,      // Pre-allocated [1, n_kv_heads, max_seq_len, head_dim]
    v: Option<Tensor>,      // Pre-allocated [1, n_kv_heads, max_seq_len, head_dim]
    current_seq_len: usize, // Number of valid entries
    max_seq_len: usize,     // Current allocation size
    dim: usize,             // Concatenation axis (2 for 4D, 1 for 3D)
    /// Bounded sliding-window KV (gemma4 SWA layers): when Some(w), the cache
    /// retains only the last `w` keys/values - it slides (keeps last `w`) before
    /// any append that would exceed the buffer, instead of growing. The buffer is
    /// sized `w + max_prefill_chunk` so a whole prefill chunk's queries still see
    /// their full window (`w >= chunk`). None = unbounded (default; all non-SWA
    /// layers and non-gemma models -> append is bit-identical to before).
    window: Option<usize>,
    /// Stable device-side `i32` holding `current_seq_len - 1`. Used by
    /// fused_attn_decode_f32_hd512 (and similar dev_pos kernels) so the
    /// captured graph reads the live position from device memory each
    /// replay instead of baking in a host int at capture time. Allocated
    /// lazily; updated via memcpy_htod once per token outside capture
    /// (sister of `Q8KvCache::cur_pos_dev`).
    #[cfg(feature = "cuda")]
    seq_kv_dev: Option<CudaSlice<i32>>,
    #[cfg(feature = "cuda")]
    seq_kv_dev_value: Option<usize>,
}

impl Clone for SpecKvCache {
    fn clone(&self) -> Self {
        Self {
            k: self.k.clone(),
            v: self.v.clone(),
            current_seq_len: self.current_seq_len,
            max_seq_len: self.max_seq_len,
            dim: self.dim,
            window: self.window,
            // seq_kv_dev is per-instance device state; clone resets it
            // since the new clone shouldn't share the same captured-graph
            // pointer dependency.
            #[cfg(feature = "cuda")]
            seq_kv_dev: None,
            #[cfg(feature = "cuda")]
            seq_kv_dev_value: None,
        }
    }
}

impl SpecKvCache {
    /// Create a new empty KV cache with initial allocation size.
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k: None,
            v: None,
            current_seq_len: 0,
            max_seq_len,
            dim: 2, // Default: 4D tensors, seq at axis 2
            window: None,
            #[cfg(feature = "cuda")]
            seq_kv_dev: None,
            #[cfg(feature = "cuda")]
            seq_kv_dev_value: None,
        }
    }

    /// Bound this cache to a sliding window of `w` keys (gemma4 SWA layers).
    /// Sizes the buffer to `w + chunk` so a prefill chunk still sees its full
    /// window; append then slides (keeps last `w`) instead of growing. Must be
    /// set before the first append (sizes the lazy allocation).
    pub fn set_window(&mut self, w: usize, max_chunk: usize) {
        self.window = Some(w);
        self.max_seq_len = w + max_chunk;
    }

    /// Reset the KV cache (clears all entries).
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.current_seq_len = 0;
        // Don't drop seq_kv_dev - its device address must stay stable
        // for any captured graph that references it. The next update
        // will overwrite the value via memcpy_htod.
        #[cfg(feature = "cuda")]
        {
            self.seq_kv_dev_value = None;
        }
    }

    /// Pre-allocate (lazily) the stable device i32 buffer holding the
    /// current seq_kv position (= current_seq_len - 1, matching the Q8
    /// dev_pos convention where the kernel adds 1). Updates the value
    /// to `pos` via memcpy_htod. Returns a reference to the stable
    /// device slice so the caller can pass it to the captured kernel.
    ///
    /// Fast path: if the stored value already equals `pos`, skip the
    /// memcpy. Per-layer in a homogeneous model, every layer in the
    /// same token calls with the same value - saves N-1 redundant
    /// 4-byte H2D copies per token.
    #[cfg(feature = "cuda")]
    pub fn update_seq_kv_dev(
        &mut self,
        pos: usize,
        dev: &crate::tensor::cuda_ext::RawCudaDevice,
    ) -> Result<&CudaSlice<i32>> {
        if self.seq_kv_dev_value == Some(pos) && self.seq_kv_dev.is_some() {
            return Ok(self.seq_kv_dev.as_ref().unwrap());
        }
        let val: [i32; 1] = [pos as i32];
        let stream = dev.cuda_stream();
        match self.seq_kv_dev.as_mut() {
            Some(dst) => {
                stream.memcpy_htod(val.as_slice(), dst).map_err(|e| {
                    crate::tensor::Error::msg(format!("SpecKvCache::update_seq_kv_dev memcpy: {e}"))
                })?;
            }
            None => {
                let slice: CudaSlice<i32> = stream.clone_htod(val.as_slice()).map_err(|e| {
                    crate::tensor::Error::msg(format!("SpecKvCache::update_seq_kv_dev alloc: {e}"))
                })?;
                self.seq_kv_dev = Some(slice);
            }
        }
        self.seq_kv_dev_value = Some(pos);
        Ok(self.seq_kv_dev.as_ref().unwrap())
    }

    /// Returns the stable device i32 slice if it has been allocated.
    /// Engine-side gates check `is_some` to decide whether to use the
    /// device-pos-driven kernel path.
    #[cfg(feature = "cuda")]
    pub fn seq_kv_dev_ptr(&self) -> Option<&CudaSlice<i32>> {
        self.seq_kv_dev.as_ref()
    }

    /// Get the current sequence length.
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// True when this cache is bounded to a sliding window (gemma4 SWA layers via
    /// set_window). The prefill attention uses this to gate the bounded SWA mask.
    pub fn is_windowed(&self) -> bool {
        self.window.is_some()
    }

    /// The sliding window, if bounded (gemma4 SWA layers).
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    /// Advance the sequence length counter without writing data.
    /// Used when the KV write is done via scatter_set outside the cache.
    pub fn advance_seq_len(&mut self, n: usize) {
        self.current_seq_len += n;
    }

    /// Get the current full K and V tensors (narrowed to current_seq_len).
    /// Returns None if the cache is empty.
    pub fn current_kv(&self) -> Option<(Tensor, Tensor)> {
        if self.current_seq_len == 0 {
            return None;
        }
        let k = self.k.as_ref()?;
        let v = self.v.as_ref()?;
        let k = k.narrow(self.dim, 0, self.current_seq_len).ok()?;
        let v = v.narrow(self.dim, 0, self.current_seq_len).ok()?;
        Some((k, v))
    }

    /// Context shift: drops positions `[from - discard, from)` and pulls `[from, len)`
    /// down by `discard`, passing the moved keys through `rotate` so their phase matches
    /// the new positions. Refused on a windowed cache, where after a slide the buffer
    /// index is no longer the position. An empty cache is left alone: the layer keeps
    /// its KV in another store.
    pub fn shift_tail(
        &mut self,
        from: usize,
        discard: usize,
        rotate: &mut dyn FnMut(&Tensor) -> Result<Tensor>,
    ) -> Result<()> {
        let len = self.current_seq_len;
        if len == 0 {
            return Ok(());
        }
        if self.window.is_some() {
            return Err(crate::tensor::Error::msg(
                "KvCache.shift_tail: windowed cache".to_string(),
            ));
        }
        if discard == 0 || from > len || discard > from {
            return Err(crate::tensor::Error::msg(format!(
                "KvCache.shift_tail: from {from} discard {discard} len {len}"
            )));
        }
        let n = len - from;
        if n > 0 {
            let dim = self.dim;
            let kb = self.k.as_mut().unwrap();
            let vb = self.v.as_mut().unwrap();
            let k_tail = kb.narrow(dim, from, n)?.contiguous()?;
            let v_tail = vb.narrow(dim, from, n)?.contiguous()?;
            let k_tail = rotate(&k_tail)?;
            kb.slice_set(&k_tail, dim, from - discard)?;
            vb.slice_set(&v_tail, dim, from - discard)?;
        }
        self.current_seq_len = len - discard;
        Ok(())
    }

    /// Copies the `[0, len)` entries out. `None` on an empty cache: the layer keeps its
    /// KV in another store, and there is nothing to bring back.
    pub fn snapshot(&self) -> Result<Option<SpecKvSnapshot>> {
        let len = self.current_seq_len;
        let (Some(k), Some(v)) = (self.k.as_ref(), self.v.as_ref()) else {
            return Ok(None);
        };
        if len == 0 {
            return Ok(None);
        }
        let dim = self.dim;
        let k = k.narrow(dim, 0, len)?.contiguous()?.affine(1.0, 0.0)?;
        let v = v.narrow(dim, 0, len)?.contiguous()?.affine(1.0, 0.0)?;
        Ok(Some(SpecKvSnapshot { k, v, len }))
    }

    /// Makes a snapshot the whole content of the cache.
    pub fn restore(&mut self, snap: &SpecKvSnapshot) -> Result<()> {
        self.current_seq_len = 0;
        self.append(&snap.k, &snap.v).map(|_| ())
    }

    /// Trim the KV cache to a specific sequence length.
    /// This is O(1): only updates current_seq_len counter.
    pub fn trim_to(&mut self, seq_len: usize) {
        self.current_seq_len = seq_len.min(self.current_seq_len);
    }

    /// EAGLE-2 tree-KV COMPACTION. After a tree verify writes M+1 tokens
    /// at scattered slots, keep the first `prefix_len` slots untouched, then
    /// GATHER the K/V at `source_slots` (the accepted root->leaf path nodes, which
    /// sit at non-contiguous BFS slots >= prefix_len) and write them CONTIGUOUSLY
    /// at `prefix_len..`. Sets `current_seq_len = prefix_len + source_slots.len()`.
    /// Replaces the re-decode forward (the accepted path's tree-RoPE is already
    /// path-contiguous, so the gathered K/V are valid as-is). O(path_len), not
    /// O(seq). `source_slots` are absolute slot indices into the current buffer.
    pub fn compact_tail(&mut self, prefix_len: usize, source_slots: &[u32]) -> Result<()> {
        if source_slots.is_empty() {
            self.current_seq_len = prefix_len.min(self.current_seq_len);
            return Ok(());
        }
        let dim = self.dim;
        let (kb, vb) = match (self.k.as_ref(), self.v.as_ref()) {
            (Some(k), Some(v)) => (k, v),
            _ => return Err(crate::tensor::Error::msg("compact_tail: empty cache")),
        };
        let idx = Tensor::from_vec(source_slots.to_vec(), (source_slots.len(),), &kb.device())?;
        // Gather first (into fresh tensors) so the subsequent in-place write can't
        // alias the source region.
        let gk = kb.index_select(&idx, dim)?.contiguous()?;
        let gv = vb.index_select(&idx, dim)?.contiguous()?;
        self.k.as_mut().unwrap().slice_set(&gk, dim, prefix_len)?;
        self.v.as_mut().unwrap().slice_set(&gv, dim, prefix_len)?;
        self.current_seq_len = prefix_len + source_slots.len();
        Ok(())
    }

    /// Append new K and V tensors and return the full accumulated K, V.
    ///
    /// Uses pre-allocated buffers with slice_set for O(1) append (no copy of existing data).
    ///
    /// # Returns
    /// (full_k, full_v) narrowed views of shape [batch, n_kv_heads, current_seq_len+n_new, head_dim]
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = k_new.dims();
        let (dim, n_new) = if dims.len() == 4 {
            (2, dims[2]) // seq_len at index 2 for 4D
        } else if dims.len() == 3 {
            (1, dims[1]) // seq_len at index 1 for 3D
        } else {
            let msg = format!(
                "KvCache.append: k_new has unexpected shape {:?}, expected 3D or 4D",
                dims
            );
            return Err(crate::tensor::Error::msg(msg));
        };
        self.dim = dim;

        // Initialize pre-allocated buffer on first call
        if self.k.is_none() {
            let mut shape = dims.to_vec();
            shape[dim] = self.max_seq_len;
            let k_buf = Tensor::zeros_on(&shape[..], k_new.dtype(), &k_new.device())?;
            let v_buf = Tensor::zeros_on(&shape[..], v_new.dtype(), &v_new.device())?;
            self.k = Some(k_buf);
            self.v = Some(v_buf);
        }

        // Bounded sliding-window (gemma4 SWA layers): instead of growing past the
        // buffer, SLIDE - keep the last `w` keys then append. Buffer is sized
        // `w + max_chunk`, so after sliding (current=w) a chunk of <=max_chunk fits
        // without growing, and every query in the appended chunk still sees its full
        // `w`-key window (w >= chunk). Overlap-safe: narrow->contiguous (a copy) before
        // the in-place write at offset 0. No-op when window is None (default).
        if let Some(w) = self.window {
            if self.current_seq_len + n_new > self.max_seq_len && self.current_seq_len > w {
                let kb = self.k.as_mut().unwrap();
                let vb = self.v.as_mut().unwrap();
                let start = self.current_seq_len - w;
                let gk = kb.narrow(dim, start, w)?.contiguous()?;
                let gv = vb.narrow(dim, start, w)?.contiguous()?;
                kb.slice_set(&gk, dim, 0)?;
                vb.slice_set(&gv, dim, 0)?;
                self.current_seq_len = w;
            }
        }

        // Grow if needed (double the buffer)
        while self.current_seq_len + n_new > self.max_seq_len {
            let grow = self.max_seq_len.max(n_new);
            self.max_seq_len += grow;

            let k_old = self.k.take().unwrap();
            let v_old = self.v.take().unwrap();
            let mut shape = k_old.dims().to_vec();
            shape[dim] = grow;
            let k_ext = Tensor::zeros_on(&shape[..], k_old.dtype(), &k_old.device())?;
            let v_ext = Tensor::zeros_on(&shape[..], v_old.dtype(), &v_old.device())?;
            self.k = Some(Tensor::cat(&[&k_old, &k_ext], dim)?);
            self.v = Some(Tensor::cat(&[&v_old, &v_ext], dim)?);
        }

        // In-place write: copy new tokens into pre-allocated buffer at current_seq_len offset
        let k_buf = self.k.as_mut().unwrap();
        let v_buf = self.v.as_mut().unwrap();
        k_buf.slice_set(k_new, dim, self.current_seq_len)?;
        v_buf.slice_set(v_new, dim, self.current_seq_len)?;

        self.current_seq_len += n_new;

        // Return narrowed views (O(1), no copy)
        let k_out = k_buf.narrow(dim, 0, self.current_seq_len)?;
        let v_out = v_buf.narrow(dim, 0, self.current_seq_len)?;

        Ok((k_out, v_out))
    }

    /// Get the full K buffer (for graph mode - fixed pointer).
    pub fn k_buffer(&self) -> Option<Tensor> {
        self.k.clone()
    }

    /// Get the full V buffer (for graph mode - fixed pointer).
    pub fn v_buffer(&self) -> Option<Tensor> {
        self.v.clone()
    }

    /// Get the current padded buffer size.
    pub fn max_seq_len_padded(&self) -> usize {
        self.max_seq_len
    }

    /// Append and return the FULL pre-allocated buffers (not narrowed).
    /// Used for CUDA graph capture - all operations have fixed dimensions.
    /// The caller must apply a causal mask to ignore positions >= current_seq_len.
    pub fn append_padded(
        &mut self,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<(Tensor, Tensor, usize)> {
        let dims = k_new.dims();
        let (dim, n_new) = if dims.len() == 4 {
            (2, dims[2])
        } else if dims.len() == 3 {
            (1, dims[1])
        } else {
            return Err(crate::tensor::Error::msg(format!(
                "KvCache: unexpected shape {:?}",
                dims
            )));
        };
        self.dim = dim;

        // Initialize buffer on first call
        if self.k.is_none() {
            let mut shape = dims.to_vec();
            shape[dim] = self.max_seq_len;
            self.k = Some(Tensor::zeros_on(
                &shape[..],
                k_new.dtype(),
                &k_new.device(),
            )?);
            self.v = Some(Tensor::zeros_on(
                &shape[..],
                v_new.dtype(),
                &v_new.device(),
            )?);
        }

        // Grow if needed
        while self.current_seq_len + n_new > self.max_seq_len {
            let grow = self.max_seq_len.max(n_new);
            self.max_seq_len += grow;
            let k_old = self.k.take().unwrap();
            let v_old = self.v.take().unwrap();
            let mut shape = k_old.dims().to_vec();
            shape[dim] = grow;
            let k_ext = Tensor::zeros_on(&shape[..], k_old.dtype(), &k_old.device())?;
            let v_ext = Tensor::zeros_on(&shape[..], v_old.dtype(), &v_old.device())?;
            self.k = Some(Tensor::cat(&[&k_old, &k_ext], dim)?);
            self.v = Some(Tensor::cat(&[&v_old, &v_ext], dim)?);
        }

        let k_buf = self.k.as_mut().unwrap();
        let v_buf = self.v.as_mut().unwrap();
        k_buf.slice_set(k_new, dim, self.current_seq_len)?;
        v_buf.slice_set(v_new, dim, self.current_seq_len)?;
        self.current_seq_len += n_new;

        // Return FULL buffers (fixed dimensions for graph capture)
        Ok((k_buf.clone(), v_buf.clone(), self.current_seq_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Device};

    fn make_4d(dev: &Device, n_new: usize) -> Tensor {
        // Shape: (batch=1, n_kv_heads=2, seq_len=n_new, head_dim=4)
        Tensor::zeros_on(&[1, 2, n_new, 4], DType::F32, dev).unwrap()
    }

    /// [1,2,n,4] where every element of token i (0-based within the chunk,
    /// offset by `base`) equals (base+i) - a position-identifiable value.
    fn make_4d_pos(dev: &Device, base: usize, n: usize) -> Tensor {
        let mut data = Vec::with_capacity(2 * n * 4);
        for _h in 0..2 {
            for i in 0..n {
                for _d in 0..4 {
                    data.push((base + i) as f32);
                }
            }
        }
        Tensor::from_vec(data, (1, 2, n, 4), dev).unwrap()
    }

    #[test]
    fn windowed_cache_slides_keeping_last_window() {
        let dev = Device::Cpu;
        let mut c = SpecKvCache::new(100);
        c.set_window(4, 2); // window=4, chunk=2 -> buffer 6
                            // Append positions 0..8 in chunks of 2.
        for chunk in 0..4 {
            let k = make_4d_pos(&dev, chunk * 2, 2);
            c.append(&k, &k).unwrap();
        }
        // After 8 tokens with window 4: buffer holds the last `current` tokens
        // time-ordered; the latest query's window = last 4 = positions [4,5,6,7].
        let (k, _) = c.current_kv().unwrap();
        let cur = c.current_seq_len();
        // last 4 entries along seq axis (dim 2)
        let last4 = k.narrow(2, cur - 4, 4).unwrap();
        // head 0, dim 0 of each of the 4 tokens
        let vals: Vec<f32> = last4
            .narrow(1, 0, 1)
            .unwrap()
            .narrow(3, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            vals,
            vec![4.0, 5.0, 6.0, 7.0],
            "windowed cache must retain the last `window` positions in order"
        );
    }

    #[test]
    fn unwindowed_cache_keeps_everything() {
        let dev = Device::Cpu;
        let mut c = SpecKvCache::new(4); // small, will grow
        for chunk in 0..4 {
            let k = make_4d_pos(&dev, chunk * 2, 2);
            c.append(&k, &k).unwrap();
        }
        // No window -> keeps all 8 positions (grows), bit-identical to old behavior.
        assert_eq!(c.current_seq_len(), 8);
        let (k, _) = c.current_kv().unwrap();
        let vals: Vec<f32> = k
            .narrow(1, 0, 1)
            .unwrap()
            .narrow(3, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(vals, (0..8).map(|x| x as f32).collect::<Vec<_>>());
    }

    #[test]
    fn new_cache_is_empty_with_no_buffers() {
        let c = SpecKvCache::new(128);
        assert_eq!(c.current_seq_len(), 0);
        assert!(c.current_kv().is_none(), "empty cache should yield no view");
        assert!(c.k_buffer().is_none(), "no buffer until first append");
        assert!(c.v_buffer().is_none());
        assert_eq!(c.max_seq_len_padded(), 128);
    }

    #[test]
    fn advance_seq_len_increments_counter_without_touching_buffers() {
        // advance_seq_len is used when the K/V write happens via an
        // out-of-band scatter_set (cuda_sampling path). The buffers
        // stay None until a real append, but the counter must still
        // tick - pin this so a refactor doesn't accidentally couple
        // the two paths.
        let mut c = SpecKvCache::new(128);
        c.advance_seq_len(5);
        assert_eq!(c.current_seq_len(), 5);
        c.advance_seq_len(3);
        assert_eq!(c.current_seq_len(), 8);
        // Buffers still uninitialised.
        assert!(c.k_buffer().is_none());
    }

    #[test]
    fn trim_to_saturates_at_current_seq_len_never_grows() {
        // trim_to is O(1) - it just lowers the counter. Critically,
        // passing a value LARGER than current_seq_len must NOT grow
        // the counter (which would expose uninitialised buffer slots
        // as valid KV - silent corruption).
        let mut c = SpecKvCache::new(128);
        c.advance_seq_len(10);
        c.trim_to(20);
        assert_eq!(c.current_seq_len(), 10, "trim_to must saturate, not grow");
        c.trim_to(7);
        assert_eq!(c.current_seq_len(), 7);
        c.trim_to(0);
        assert_eq!(c.current_seq_len(), 0);
    }

    #[test]
    fn reset_clears_buffers_and_counter() {
        let mut c = SpecKvCache::new(64);
        let k = make_4d(&Device::Cpu, 3);
        let v = make_4d(&Device::Cpu, 3);
        c.append(&k, &v).expect("append");
        assert_eq!(c.current_seq_len(), 3);
        assert!(c.k_buffer().is_some());

        c.reset();
        assert_eq!(c.current_seq_len(), 0);
        assert!(c.k_buffer().is_none(), "reset must drop buffers");
        assert!(c.v_buffer().is_none());
        assert!(c.current_kv().is_none(), "no view after reset");
        // Cap stays at the pre-reset size - reset doesn't shrink the
        // allocator's growth state. (max_seq_len_padded grows on
        // append; reset only clears the data, not the size hint.)
    }

    #[test]
    fn append_accumulates_seq_len_across_calls() {
        let mut c = SpecKvCache::new(64);
        let _ = c
            .append(&make_4d(&Device::Cpu, 2), &make_4d(&Device::Cpu, 2))
            .unwrap();
        assert_eq!(c.current_seq_len(), 2);
        let _ = c
            .append(&make_4d(&Device::Cpu, 5), &make_4d(&Device::Cpu, 5))
            .unwrap();
        assert_eq!(c.current_seq_len(), 7, "second append accumulates");
        // current_kv returns a view of exactly current_seq_len slots.
        let (k, v) = c.current_kv().expect("nonempty");
        assert_eq!(k.dims(), &[1, 2, 7, 4]);
        assert_eq!(v.dims(), &[1, 2, 7, 4]);
    }

    #[test]
    fn append_grows_buffer_when_exceeded() {
        // Cap at 4, append 3 then another 3 -> must grow.
        let mut c = SpecKvCache::new(4);
        c.append(&make_4d(&Device::Cpu, 3), &make_4d(&Device::Cpu, 3))
            .unwrap();
        assert_eq!(c.max_seq_len_padded(), 4);
        // 3 + 3 = 6 > 4 -> triggers doubling. Doubling rule: grow by
        // max(max_seq_len, n_new) = max(4, 3) = 4 -> new cap = 8.
        c.append(&make_4d(&Device::Cpu, 3), &make_4d(&Device::Cpu, 3))
            .unwrap();
        assert_eq!(c.current_seq_len(), 6);
        assert!(c.max_seq_len_padded() >= 6, "buffer must grow to fit");
    }

    #[test]
    fn append_rejects_non_3d_or_4d_shapes() {
        let mut c = SpecKvCache::new(8);
        // 2D tensor - neither 3D nor 4D, must error rather than alias
        // dim wrong (would silently corrupt cache shape).
        let bad = Tensor::zeros_on(&[2, 4], DType::F32, &Device::Cpu).unwrap();
        assert!(c.append(&bad, &bad).is_err());
        // 5D also rejected.
        let bad5 = Tensor::zeros_on(&[1, 1, 2, 4, 4], DType::F32, &Device::Cpu).unwrap();
        assert!(c.append(&bad5, &bad5).is_err());
    }

    #[test]
    fn current_kv_returns_narrowed_view_not_full_buffer() {
        // After 2 appends totalling 3 tokens with cap = 16, current_kv
        // must return a (..., 3, ...) view - not the full padded
        // (..., 16, ...) buffer.
        let mut c = SpecKvCache::new(16);
        c.append(&make_4d(&Device::Cpu, 1), &make_4d(&Device::Cpu, 1))
            .unwrap();
        c.append(&make_4d(&Device::Cpu, 2), &make_4d(&Device::Cpu, 2))
            .unwrap();
        let (k, _) = c.current_kv().unwrap();
        assert_eq!(k.dims()[2], 3, "view must narrow to current_seq_len");
        // The full buffer is still 16.
        assert_eq!(c.k_buffer().unwrap().dims()[2], 16);
    }
}
