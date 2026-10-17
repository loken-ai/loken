//! Reads of expert bytes that bypass the page cache, for the parts of a prefill's upload whose
//! pages the cache does not hold: a direct read into a page-locked bounce buffer, then a copy to
//! the card, so a layer streamed once does not evict the experts the decode reads next.

#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::sync::Arc;

/// The span `[aligned_off, aligned_off + aligned_len)`, aligned to `align` (a power of two), that
/// covers the file bytes `[off, off + len)`, and the position of `off` inside that span.
pub(super) fn aligned_cover(off: u64, len: usize, align: usize) -> (u64, usize, usize) {
    let a = align as u64;
    let lo = off / a * a;
    let hi = (off + len as u64).div_ceil(a) * a;
    (lo, (hi - lo) as usize, (off - lo) as usize)
}

/// The bytes `[start, start + len)` of memory as maximal runs `(from, to, resident)`, given
/// `resident`, one `mincore` byte per page from the page holding `start`.
pub(super) fn split_by_residency(
    start: usize,
    len: usize,
    page: usize,
    resident: &[u8],
) -> Vec<(usize, usize, bool)> {
    let base = start / page * page;
    let end = start + len;
    let mut runs: Vec<(usize, usize, bool)> = Vec::new();
    for (k, r) in resident.iter().enumerate() {
        let from = (base + k * page).max(start);
        let to = (base + (k + 1) * page).min(end);
        if to <= from {
            continue;
        }
        let here = r & 1 != 0;
        match runs.last_mut() {
            Some(last) if last.2 == here => last.1 = to,
            _ => runs.push((from, to, here)),
        }
    }
    runs
}

/// Which bytes of `[start, start + len)` the page cache holds, or `None` when that is unknown.
#[cfg(target_os = "linux")]
pub(super) fn residency(
    start: usize,
    len: usize,
    page: usize,
) -> Option<Vec<(usize, usize, bool)>> {
    let base = start / page * page;
    let span = (start + len).div_ceil(page) * page - base;
    let mut vec = vec![0u8; span / page];
    // Safety: `vec` holds one byte per page of the range, which lies in a live mapping.
    let r = unsafe { libc::mincore(base as *mut libc::c_void, span, vec.as_mut_ptr() as *mut _) };
    (r == 0).then(|| split_by_residency(start, len, page, &vec))
}

/// A stage's direct reader: an `O_DIRECT` handle on the weights' file and two page-locked bounce
/// buffers used in turn, each guarded by the event of the last copy out of it.
#[cfg(target_os = "linux")]
pub(super) struct Bounce {
    source: Arc<File>,
    direct: File,
    bufs: [*mut u8; 2],
    /// Bytes of file a chunk carries; a chunk's aligned cover fits in one buffer.
    payload: usize,
    page: usize,
    events: [Option<cudarc::driver::CudaEvent>; 2],
    next: usize,
}

// Safety: the buffers are owned by the reader and only touched through `&mut self`.
#[cfg(target_os = "linux")]
unsafe impl Send for Bounce {}

#[cfg(target_os = "linux")]
impl Bounce {
    /// A reader for `source` whose chunks carry up to `payload` bytes (the largest run's unit).
    pub(super) fn open(source: &Arc<File>, payload: usize, page: usize) -> Result<Self, String> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;
        let direct = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(format!("/proc/self/fd/{}", source.as_raw_fd()))
            .map_err(|e| format!("open for direct reads: {e}"))?;
        let payload = payload.div_ceil(page) * page;
        let cap = payload + 2 * page;
        let mut bufs = [std::ptr::null_mut::<u8>(); 2];
        for i in 0..2 {
            match unsafe { cudarc::driver::result::malloc_host(cap, 0) } {
                Ok(p) => bufs[i] = p as *mut u8,
                Err(e) => {
                    for p in bufs.iter().filter(|p| !p.is_null()) {
                        unsafe { cudarc::driver::result::free_host(*p as *mut _) }.ok();
                    }
                    return Err(format!("bounce buffer: {e}"));
                }
            }
        }
        Ok(Self {
            source: source.clone(),
            direct,
            bufs,
            payload,
            page,
            events: [None, None],
            next: 0,
        })
    }

    /// Whether this reader reads `file`.
    pub(super) fn reads(&self, file: &Arc<File>) -> bool {
        Arc::ptr_eq(&self.source, file)
    }

    /// Read `len` file bytes from `file_off` and queue their copy to `dst` on `stream`. Returns
    /// the bytes queued and the error that stopped the rest, if any.
    pub(super) fn upload(
        &mut self,
        file_off: u64,
        len: usize,
        dst: u64,
        stream: &cudarc::driver::CudaStream,
    ) -> (usize, Option<String>) {
        use std::os::unix::fs::FileExt;
        let mut done = 0;
        while done < len {
            let n = (len - done).min(self.payload);
            let (lo, cover, skip) = aligned_cover(file_off + done as u64, n, self.page);
            let i = self.next;
            self.next ^= 1;
            // The last copy out of this buffer has to land before it is overwritten.
            if let Some(ev) = self.events[i].take() {
                if let Err(e) = ev.synchronize() {
                    return (done, Some(format!("bounce wait: {e:?}")));
                }
            }
            let buf = unsafe { std::slice::from_raw_parts_mut(self.bufs[i], cover) };
            let mut got = 0;
            while got < skip + n {
                match self.direct.read_at(&mut buf[got..], lo + got as u64) {
                    Ok(0) => return (done, Some("direct read past the end".to_string())),
                    Ok(k) => got += k,
                    Err(e) => return (done, Some(format!("direct read: {e}"))),
                }
            }
            let r = unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    dst + done as u64,
                    self.bufs[i].add(skip) as *const std::ffi::c_void,
                    n,
                    stream.cu_stream(),
                )
            };
            if r != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return (done, Some(format!("bounce to stage: {r:?}")));
            }
            match stream.record_event(None) {
                Ok(ev) => self.events[i] = Some(ev),
                Err(e) => return (done + n, Some(format!("bounce record: {e:?}"))),
            }
            done += n;
        }
        (done, None)
    }
}

#[cfg(target_os = "linux")]
impl Drop for Bounce {
    fn drop(&mut self) {
        for ev in self.events.iter().flatten() {
            ev.synchronize().ok();
        }
        for p in self.bufs {
            unsafe { cudarc::driver::result::free_host(p as *mut std::ffi::c_void) }.ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cover_is_aligned_and_holds_the_range() {
        let (lo, len, skip) = aligned_cover(4096 * 3 + 100, 5000, 4096);
        assert_eq!((lo, len, skip), (4096 * 3, 4096 * 2, 100));
        assert!(skip + 5000 <= len);
        assert_eq!(aligned_cover(8192, 4096, 4096), (8192, 4096, 0));
        assert_eq!(aligned_cover(8191, 2, 4096), (4096, 8192, 4095));
    }

    #[test]
    fn residency_splits_into_runs() {
        let page = 4096;
        let start = page + 10;
        let end = 5 * page + 20;
        let v = [1u8, 0, 0, 1, 1];
        let runs = split_by_residency(start, end - start, page, &v);
        assert_eq!(
            runs,
            vec![
                (start, 2 * page, true),
                (2 * page, 4 * page, false),
                (4 * page, end, true)
            ]
        );
        let all = split_by_residency(start, end - start, page, &[1, 1, 1, 1, 1]);
        assert_eq!(all, vec![(start, end, true)]);
    }
}
