//! A byte range of a mapped file, and the page-cache advice a reader gives about it.

use crate::tensor::{Error, Result};
use std::sync::Arc;

/// The size of a memory page, `None` when the system does not say.
#[cfg(unix)]
pub fn page_size() -> Option<usize> {
    // Safety: sysconf has no preconditions.
    match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        p if p > 0 => Some(p as usize),
        _ => None,
    }
}

/// Give the kernel `advice` about the pages behind `bytes`: the pages wholly inside the slice when
/// `inward`, every page the slice touches otherwise. Advice only; a refusal changes nothing.
#[cfg(unix)]
pub fn advise_bytes(bytes: &[u8], advice: libc::c_int, inward: bool) {
    advise_span(bytes.as_ptr() as usize, bytes.len(), advice, inward);
}

/// `advise_bytes` over `len` bytes at address `start`, which must lie within one mapping.
#[cfg(unix)]
pub fn advise_span(start: usize, len: usize, advice: libc::c_int, inward: bool) {
    let Some(page) = page_size() else {
        return;
    };
    let end = start + len;
    let (lo, hi) = if inward {
        (start.div_ceil(page) * page, end / page * page)
    } else {
        (start / page * page, end.div_ceil(page) * page)
    };
    if hi <= lo {
        return;
    }
    // The kernel reads one read-ahead window per WILLNEED request and drops the rest of the
    // span, so a longer span is asked for a window at a time.
    let step = if advice == libc::MADV_WILLNEED {
        read_ahead_window(lo, hi - lo).map_or(hi - lo, |w| w.div_ceil(page).max(1) * page)
    } else {
        hi - lo
    };
    let mut at = lo;
    while at < hi {
        let n = step.min(hi - at);
        // Safety: [at, at + n) lies within [lo, hi), the pages the span occupies, which are mapped.
        unsafe { libc::madvise(at as *mut libc::c_void, n, advice) };
        at += n;
    }
}

/// A live mapping and the file behind it.
struct MappedFile {
    map: std::sync::Weak<memmap2::Mmap>,
    file: Arc<std::fs::File>,
    /// The bytes one read-ahead request on the file's device reads, `None` when the system
    /// does not say.
    #[cfg(unix)]
    window: Option<usize>,
}

impl MappedFile {
    /// The offset in the file of `len` bytes at `start`, when they lie within this mapping.
    fn offset_of(&self, start: usize, len: usize) -> Option<u64> {
        let m = self.map.upgrade()?;
        let base = m.as_ptr() as usize;
        (start >= base && start + len <= base + m.len()).then(|| (start - base) as u64)
    }
}

/// The mapped files by address: each live mapping with the file behind it, so bytes read through
/// a mapping can be read from the file instead (see `file_span`).
static FILES: std::sync::Mutex<Vec<MappedFile>> = std::sync::Mutex::new(Vec::new());

/// The most bytes one read-ahead request reads on the block device that holds `file`: the
/// larger of the device's read-ahead window and its largest single request, as the kernel caps
/// a WILLNEED request. `None` when the device's queue limits cannot be read.
#[cfg(target_os = "linux")]
fn device_read_ahead(file: &std::fs::File) -> Option<usize> {
    use std::os::unix::fs::MetadataExt;
    let dev = file.metadata().ok()?.dev();
    // A partition has no queue of its own; its disk's queue is the directory above.
    let base = format!("/sys/dev/block/{}:{}", libc::major(dev), libc::minor(dev));
    let kb = |name: &str| -> Option<usize> {
        [
            format!("{base}/queue/{name}"),
            format!("{base}/../queue/{name}"),
        ]
        .iter()
        .find_map(|p| {
            std::fs::read_to_string(p)
                .ok()?
                .trim()
                .parse::<usize>()
                .ok()
        })
    };
    let window = kb("read_ahead_kb").max(kb("max_sectors_kb"))?;
    (window > 0).then_some(window * 1024)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn device_read_ahead(_file: &std::fs::File) -> Option<usize> {
    None
}

/// The read-ahead window of the file mapped at `start`, `len` bytes, when the bytes lie within
/// a live mapping registered with `register_file` and its device says.
#[cfg(unix)]
fn read_ahead_window(start: usize, len: usize) -> Option<usize> {
    let g = FILES.lock().unwrap_or_else(|e| e.into_inner());
    g.iter()
        .find_map(|f| f.offset_of(start, len).map(|_| f.window))?
}

/// Record that `map` maps `file`. The record holds the mapping weakly: once it is unmapped the
/// record can answer nothing, so an address a later mapping reuses is never taken for this file.
pub fn register_file(map: &Arc<memmap2::Mmap>, file: &Arc<std::fs::File>) {
    let mut g = FILES.lock().unwrap_or_else(|e| e.into_inner());
    g.retain(|f| f.map.strong_count() > 0);
    g.push(MappedFile {
        map: Arc::downgrade(map),
        file: file.clone(),
        #[cfg(unix)]
        window: device_read_ahead(file),
    });
}

/// The file and the offset in it of `len` bytes at `start`, when they lie within a live mapping
/// registered with `register_file`.
pub fn file_span(start: usize, len: usize) -> Option<(Arc<std::fs::File>, u64)> {
    let g = FILES.lock().unwrap_or_else(|e| e.into_inner());
    g.iter()
        .find_map(|f| f.offset_of(start, len).map(|off| (f.file.clone(), off)))
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
            #[cfg(unix)]
            advise_bytes(
                &self.as_slice()[start..start + len],
                libc::MADV_WILLNEED,
                false,
            );
            #[cfg(not(unix))]
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
