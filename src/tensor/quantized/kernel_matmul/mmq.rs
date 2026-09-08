//! Driving the tiled quantised matmul.
//!
//! Unlike the mat-vec path this one quantises the activation into a scratch buffer, so it
//! needs a workspace it may fail to get - and the caller must be able to do without it.

use super::*;

impl QKernelMatMul {
    /// Batched quantized matmul through the production MMQ launchers
    /// (the same compiled kernels the current serving path uses - FFI boundary,
    /// raw pointers + stream, like the MoE family).
    #[cfg(feature = "cuda")]
    pub(super) fn mmq_forward(
        &self,
        dev: &std::sync::Arc<crate::tensor::cuda::CudaDevice>,
        blob: &cudarc::driver::CudaSlice<u8>,
        x_slice: &cudarc::driver::CudaSlice<f32>,
        rows: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        use cudarc::driver::DevicePtr;
        const QK8_1: usize = 32;
        const BLOCK_Q8_1_MMQ_SIZE: usize = 4 * QK8_1 + 4 * 4; // 128 qs + 16 scale bytes
        let pad = |p: usize, q: usize| p.div_ceil(q) * q;

        let stream = dev.stream();
        let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;

        // Activation workspace: row padding must hold deterministic zeros
        // (the quantize step writes only the valid K columns; mul_mat_q
        // reads the full padded tile) - hence alloc_zeros via the grow-only
        // per-device slot.
        let k_padded = pad(pad(self.k, 512), 4 * QK8_1);
        let blocks_per_row = k_padded / (4 * QK8_1);
        let ws_bytes = rows * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE + 128 * BLOCK_Q8_1_MMQ_SIZE;
        const FIXUP_BYTES: usize = 256 * 128 * 128 * 4; // stream-k fixup

        let ws_map = MMQ_WORKSPACES.get_or_init(|| std::sync::Mutex::new(Default::default()));
        let mut ws_guard = ws_map.lock().unwrap_or_else(|e| e.into_inner());
        let slot = ws_guard.entry(dev.ordinal()).or_default();
        if slot.main.as_ref().map(|s| s.len()) < Some(ws_bytes) || slot.main.is_none() {
            // RELEASE THE OLD ONE FIRST. `slot.main = Some(alloc(..))` evaluates the
            // allocation before the assignment, so the previous (smaller) buffer was
            // still live while the bigger one was being carved out - double peak at
            // exactly the moment memory is tightest.
            slot.main = None;
            // AND GO THROUGH THE RECLAIM CASCADE, like the `mmq out` allocation twenty
            // lines below always did. These two did not, so a workspace that missed by a
            // few MB failed the whole render with
            // `mmq workspace: CUDA_ERROR_OUT_OF_MEMORY` while its neighbour would have
            // trimmed the pools and retried. Safe under the guard we hold: the reclaim
            // try_locks this same map and skips a contended device, which is the locking
            // invariant it documents.
            slot.main = Some(crate::tensor::cuda::with_oom_retry(
                dev,
                "mmq workspace",
                || stream.alloc_zeros::<u8>(ws_bytes),
            )?);
        }
        if slot.fixup.is_none() {
            slot.fixup = Some(crate::tensor::cuda::with_oom_retry(
                dev,
                "mmq fixup workspace",
                || stream.alloc_zeros::<u8>(FIXUP_BYTES),
            )?);
        }
        // Queue behind whoever used these buffers last. Releasing the mutex when this
        // function returns orders the host, and the host is finished here - the kernels it
        // enqueued have not run. A second stream taking the workspace next would quantise
        // its own activation over the blocks this stream's matmul has yet to read, which is
        // exactly what it did: about one prefill in ten came back with whole columns of
        // another caller's numbers, and only when two GPU users ran at once. Ordering the
        // reuse on the device costs one event and stops the host nowhere.
        if let Some(ev) = slot.last_use.as_ref() {
            stream
                .wait(ev)
                .map_err(|e| Error(format!("mmq workspace ordering: {e}")))?;
        }

        let ws = slot.main.as_ref().unwrap();
        let fixup = slot.fixup.as_ref().unwrap();

        let info = dev.mmq_device_info()?;
        let out = crate::tensor::cuda::with_oom_retry(dev, "mmq out", || unsafe {
            stream.alloc::<f32>(self.n * rows)
        })?;

        let x_ptr = x_slice.device_ptr(stream).0 as *const std::ffi::c_void;
        let ws_ptr = ws.device_ptr(stream).0 as *mut std::ffi::c_void;
        let fixup_ptr = fixup.device_ptr(stream).0 as *mut std::ffi::c_void;
        let w_ptr = blob.device_ptr(stream).0 as *const std::ffi::c_void;
        let out_ptr = out.device_ptr(stream).0 as *mut std::ffi::c_void;

        unsafe {
            let quantize = mmq_quantize_launcher(self.dtype)?;
            quantize(
                x_ptr,
                std::ptr::null(),
                ws_ptr,
                0,
                self.k as i64,
                self.k as i64,
                0,
                0,
                k_padded as i64,
                rows as i64,
                1,
                1,
                stream_ptr,
            );
            let mmq = mmq_launcher(self.dtype)?;
            mmq(
                fixup_ptr,
                w_ptr,
                ws_ptr as *const std::ffi::c_void,
                out_ptr,
                self.k as i64,
                self.n as i64,
                rows as i64,
                (self.k / mmq_qk(self.dtype)) as i64,
                self.n as i64,
                info.cc,
                info.nsm,
                info.smpbo,
                info.warp_size,
                stream_ptr,
            );
        }

        // Publish our own mark before the mutex goes: from here the buffers are ours until
        // these two kernels retire, and the next taker waits on this.
        let ev = match slot.last_use.take() {
            Some(e) => e,
            None => dev.new_event()?,
        };
        ev.record(stream)
            .map_err(|e| Error(format!("mmq workspace mark: {e}")))?;
        slot.last_use = Some(ev);

        Ok(out)
    }
}
