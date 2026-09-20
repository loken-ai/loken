//! A byte range of a mapped file, and the page-cache advice a reader gives about it.

use crate::tensor::{Error, Result};
use std::sync::Arc;

/// Give the kernel `advice` about the pages behind `bytes`: the pages wholly inside the slice when
/// `inward`, every page the slice touches otherwise. Advice only; a refusal changes nothing.
#[cfg(unix)]
pub fn advise_bytes(bytes: &[u8], advice: libc::c_int, inward: bool) {
    // Safety: sysconf has no preconditions.
    let page = match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        p if p > 0 => p as usize,
        _ => return,
    };
    let start = bytes.as_ptr() as usize;
    let end = start + bytes.len();
    let (lo, hi) = if inward {
        (start.div_ceil(page) * page, end / page * page)
    } else {
        (start / page * page, end.div_ceil(page) * page)
    };
    if hi > lo {
        // Safety: [lo, hi) lies within the pages `bytes` occupies, which are mapped.
        unsafe { libc::madvise(lo as *mut libc::c_void, hi - lo, advice) };
    }
}

/// A byte range of a mapped file, pinned by the mapping.
#[derive(Clone)]
pub struct MappedBytes {
    map: Arc<memmap2::Mmap>,
    /// The mapped file, when known: what page-cache advice is addressed to.
    file: Option<Arc<std::fs::File>>,
    offset: usize,
    len: usize,
}

impl MappedBytes {
    pub fn new(map: Arc<memmap2::Mmap>, offset: usize, len: usize) -> Result<Self> {
        Self::in_file(map, None, offset, len)
    }

    /// A range of `map`, which maps `file` from its start.
    pub fn in_file(
        map: Arc<memmap2::Mmap>,
        file: Option<Arc<std::fs::File>>,
        offset: usize,
        len: usize,
    ) -> Result<Self> {
        if offset + len > map.len() {
            return Err(Error::msg(format!(
                "mapped bytes: range {offset}+{len} exceeds the mapping of {}",
                map.len()
            )));
        }
        Ok(Self {
            map,
            file,
            offset,
            len,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.map[self.offset..self.offset + self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Ask the kernel to start reading this range in, so a read that follows finds it paged
    /// in rather than faulting page by page. Advice only: a refusal changes nothing.
    pub fn will_need(&self) {
        self.will_need_range(0, self.len);
    }

    /// `will_need` over `[start, start + len)` of this range.
    pub fn will_need_range(&self, start: usize, len: usize) {
        if start + len <= self.len {
            let _ = self
                .map
                .advise_range(memmap2::Advice::WillNeed, self.offset + start, len);
        }
    }

    /// Tell the kernel these bytes will not be read again soon: the range is unmapped from
    /// this process and dropped from the page cache, so what is read next does not evict
    /// something worth keeping. A later read faults the bytes back in from the file.
    /// Tell the kernel these bytes are read at random: read-ahead around a row nobody asked for
    /// pulls megabytes per lookup and evicts what the model is using.
    pub fn advise_random(&self) {
        advise_bytes(self.as_slice(), libc::MADV_RANDOM, true);
    }

    pub fn dont_need(&self) {
        // Safety: the mapping is read-only over a file that is not written while served, so
        // discarding its pages loses nothing; they are read back from the file on demand.
        let _ = unsafe {
            self.map.unchecked_advise_range(
                memmap2::UncheckedAdvice::DontNeed,
                self.offset,
                self.len,
            )
        };
        // Dropping the pages from the page cache itself, beyond this mapping, is a Linux
        // advice; elsewhere the unmapping above leaves them to the kernel's own reclaim.
        #[cfg(target_os = "linux")]
        if let Some(f) = &self.file {
            use std::os::fd::AsRawFd;
            // Safety: a valid descriptor and a range inside the file; advice only.
            unsafe {
                libc::posix_fadvise(
                    f.as_raw_fd(),
                    self.offset as libc::off_t,
                    self.len as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
        }
    }
}
