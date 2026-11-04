//! Sleep mode: an idle model's weights wait in host memory instead of on disk.
//!
//! A sleeping model holds an image of the files its load reads: each file mapped, every page
//! read in and locked when the system lets the process lock it.
//! The card memory is released as for an unload.
//! The next load maps the same files and finds their pages in memory, so waking reads no disk.
//!
//! The image is taken only when the files fit the host budget that every host spill is admitted
//! against. A model that does not fit is unloaded.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Whether an image of `image_bytes` fits the host.
///
/// `budget` is the host budget now, and `held_unpinned` the bytes that other images hold without
/// the lock: their pages already count as available, so they must fit beside the new one.
/// Locked images are already out of the budget.
pub fn fits(image_bytes: u64, budget: u64, held_unpinned: u64) -> bool {
    image_bytes
        .checked_add(held_unpinned)
        .is_some_and(|need| need <= budget)
}

/// The files of a sleeping model, resident in host memory.
pub struct SleepImage {
    maps: Vec<memmap2::Mmap>,
    pinned: bool,
}

impl SleepImage {
    /// Map each of `paths`, read every page in and lock them in memory when the system allows.
    ///
    /// An empty file holds no page and is left out.
    pub fn capture(paths: &[PathBuf]) -> std::io::Result<Self> {
        let mut maps = Vec::with_capacity(paths.len());
        let mut pinned = true;
        for path in paths {
            let file = std::fs::File::open(path)?;
            if file.metadata()?.len() == 0 {
                continue;
            }
            // Safety: the map is read only, and a checkpoint file is not written while it is served.
            let map = unsafe { memmap2::MmapOptions::new().populate().map(&file)? };
            #[cfg(unix)]
            // Safety: the range is the whole live mapping.
            let locked = unsafe { libc::mlock(map.as_ptr().cast(), map.len()) } == 0;
            #[cfg(not(unix))]
            let locked = false;
            pinned &= locked;
            maps.push(map);
        }
        Ok(Self { maps, pinned })
    }

    /// Bytes the image holds.
    pub fn bytes(&self) -> u64 {
        self.maps.iter().map(|m| m.len() as u64).sum()
    }

    /// Whether the pages of every file are locked in memory.
    pub fn pinned(&self) -> bool {
        self.pinned
    }
}

/// The sleeping models by id.
#[derive(Default)]
pub struct Sleepers {
    images: Mutex<HashMap<String, SleepImage>>,
}

impl Sleepers {
    fn images(&self) -> std::sync::MutexGuard<'_, HashMap<String, SleepImage>> {
        self.images.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Bytes held by images whose pages are not locked.
    fn held_unpinned(&self) -> u64 {
        self.images()
            .values()
            .filter(|i| !i.pinned)
            .map(SleepImage::bytes)
            .sum()
    }

    /// Put the model `id` to sleep on the files its load reads.
    ///
    /// Returns the bytes held and whether they are locked.
    /// Returns `None` when the files do not fit the host or one cannot be read: the caller
    /// unloads the model.
    pub fn sleep(&self, id: &str, files: &[PathBuf]) -> Option<(u64, bool)> {
        let mut size = 0u64;
        for file in files {
            size = size.checked_add(std::fs::metadata(file).ok()?.len())?;
        }
        let budget = crate::inference::place::layer_executor::host_spill_budget_bytes();
        if !fits(size, budget, self.held_unpinned()) {
            return None;
        }
        let image = SleepImage::capture(files).ok()?;
        let held = (image.bytes(), image.pinned());
        self.images().insert(id.to_string(), image);
        Some(held)
    }

    /// Whether the model `id` sleeps.
    pub fn sleeping(&self, id: &str) -> bool {
        self.images().contains_key(id)
    }

    /// Let the image of the model `id` go. Returns whether one was held.
    pub fn release(&self, id: &str) -> bool {
        self.images().remove(id).is_some()
    }
}

/// Bytes of the file at `path` whose pages sit in host memory.
#[cfg(all(test, unix))]
pub(crate) fn resident_bytes(path: &std::path::Path) -> u64 {
    let file = std::fs::File::open(path).unwrap();
    if file.metadata().unwrap().len() == 0 {
        return 0;
    }
    // Safety: a read only map, queried and dropped here.
    let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
    // Safety: the page size is a constant of the system.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let mut pages = vec![0u8; map.len().div_ceil(page)];
    // Safety: the vector has a byte per page of the live map.
    let rc = unsafe { libc::mincore(map.as_ptr() as *mut _, map.len(), pages.as_mut_ptr()) };
    assert_eq!(rc, 0, "mincore on {}", path.display());
    let held = pages.iter().filter(|p| **p & 1 == 1).count() as u64 * page as u64;
    held.min(map.len() as u64)
}

/// Write the file at `path` back to disk and ask the system to drop its pages from memory.
///
/// Pages that a map holds or that are locked stay.
#[cfg(all(test, unix))]
pub(crate) fn forget(path: &std::path::Path) {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::File::open(path).unwrap();
    file.sync_all().unwrap();
    // Safety: advice on an open descriptor.
    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_image_that_fits_the_budget_sleeps() {
        assert!(fits(10, 10, 0));
        assert!(fits(4, 10, 6));
    }

    #[test]
    fn an_image_beyond_the_budget_does_not_sleep() {
        assert!(!fits(11, 10, 0));
        assert!(!fits(5, 10, 6));
        assert!(!fits(u64::MAX, u64::MAX, 1));
    }

    #[test]
    fn files_are_held_and_released() {
        let dir = std::env::temp_dir().join(format!("loken-sleep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let files = [dir.join("a.gguf"), dir.join("b.gguf"), dir.join("empty")];
        std::fs::write(&files[0], vec![7u8; 100_000]).unwrap();
        std::fs::write(&files[1], vec![9u8; 30_000]).unwrap();
        std::fs::write(&files[2], b"").unwrap();
        let sleepers = Sleepers::default();
        let (bytes, _) = sleepers.sleep("m", &files).expect("small files fit");
        assert_eq!(bytes, 130_000);
        assert!(sleepers.sleeping("m"));
        assert!(sleepers.release("m"));
        assert!(!sleepers.sleeping("m"));
        assert!(sleepers.sleep("m", &[dir.join("absent")]).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
