//! Host memory the HIP runtime cannot page-lock: mappings without write access, which the amdgpu
//! driver refuses to pin. The runtime page-locks a pageable source before copying it when the
//! copy is larger than its pinning threshold, and falls back to staging when that fails; a copy
//! from such memory is therefore cut into pieces of at most the threshold, which the runtime
//! stages without any attempt to pin them.

use std::sync::RwLock;

/// The largest pageable copy the HIP runtime stages without page-locking it first: the default
/// of its `GPU_PINNED_MIN_XFER_SIZE`, 1 MiB.
pub(super) const PIECE: usize = 1 << 20;

/// The process's mappings as `/proc/self/maps` listed them, with the address space size they
/// were read at: `(start, end, writable)`, sorted by address.
struct Maps {
    vm_pages: Option<u64>,
    areas: Vec<(usize, usize, bool)>,
}

static MAPS: RwLock<Option<Maps>> = RwLock::new(None);

/// The size of the address space in pages, which changes whenever a mapping comes or goes.
fn vm_pages() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    statm.split_whitespace().next()?.parse().ok()
}

fn read_maps(vm_pages: Option<u64>) -> Maps {
    let text = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let areas = text
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let (lo, hi) = fields.next()?.split_once('-')?;
            let perms = fields.next()?;
            Some((
                usize::from_str_radix(lo, 16).ok()?,
                usize::from_str_radix(hi, 16).ok()?,
                perms.as_bytes().get(1) == Some(&b'w'),
            ))
        })
        .collect();
    Maps { vm_pages, areas }
}

impl Maps {
    /// Whether any mapping without write access meets `[at, at + len)`.
    fn read_only(&self, at: usize, len: usize) -> bool {
        let end = at.saturating_add(len);
        let first = self.areas.partition_point(|a| a.1 <= at);
        self.areas[first..]
            .iter()
            .take_while(|a| a.0 < end)
            .any(|a| !a.2)
    }
}

/// Whether `[at, at + len)` meets memory the runtime cannot page-lock. The mappings are read
/// again whenever the address space changed size since they were last read.
pub(super) fn contains(at: usize, len: usize) -> bool {
    let now = vm_pages();
    {
        let maps = MAPS.read().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = maps.as_ref().filter(|m| m.vm_pages == now && now.is_some()) {
            return m.read_only(at, len);
        }
    }
    let fresh = read_maps(now);
    let answer = fresh.read_only(at, len);
    *MAPS.write().unwrap_or_else(|e| e.into_inner()) = Some(fresh);
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_only_mapping_is_unpinnable_and_the_heap_is_not() {
        let page = crate::tensor::mapped::page_size().unwrap();
        let map = memmap2::MmapOptions::new()
            .len(4 * page)
            .map_anon()
            .unwrap()
            .make_read_only()
            .unwrap();
        assert!(contains(map.as_ptr() as usize + page, page));
        let heap = vec![1u8; 4 * page];
        assert!(!contains(heap.as_ptr() as usize, heap.len()));
    }
}
