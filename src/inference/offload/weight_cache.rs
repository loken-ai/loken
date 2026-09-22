//! An ephemeral on-disk cache for weight bytes streamed from a peer.
//!
//! A node that runs a pipeline stage whose weights it does not hold fetches them from the node
//! that does. The fetch crosses the network once; after that the bytes live in a local file and
//! the cost is a disk read. The cache is ephemeral: a scratch area, bounded in size, that belongs
//! to one run and is removed when the run ends. It is not a copy of the model - it holds only the
//! tensors this stage actually touched, drops the least recently used when it fills, and leaves
//! nothing behind. That distinction is the whole point: a stage runs against a peer's weights
//! without ever holding the peer's file.
//!
//! The budget is what makes it safe on a node far smaller than the model. A stage's resident set
//! is a fraction of its layers' weights at any moment; the cache holds that working set on disk
//! and evicts the rest, so a 8 GB node can run a stage of a model whose weights are measured in
//! hundreds of gigabytes, paying a re-fetch only for a tensor it evicted and then needed again.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// One cached tensor: the file that holds it, its size, and the clock tick it was last read at.
struct Entry {
    file: PathBuf,
    size: u64,
    last: u64,
}

struct CacheInner {
    /// Key -> where its bytes are and how fresh they are.
    entries: HashMap<String, Entry>,
    /// One tick per access, so the least recently used is the smallest `last`.
    clock: u64,
    /// Bytes currently on disk, the sum of every entry's size.
    used: u64,
}

/// A bounded, ephemeral directory of streamed weight bytes.
pub struct WeightCache {
    dir: PathBuf,
    capacity: u64,
    /// Whether dropping the cache removes its directory. An `ephemeral` cache owns its directory;
    /// one opened `at` a caller's directory does not, so a caller can inspect what was cached.
    own_dir: bool,
    inner: Mutex<CacheInner>,
}

/// Unique enough that two caches on one node, even at the same nanosecond, get different
/// directories: the pid, a per-process counter, and the wall clock.
fn unique_suffix() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{n}-{nanos}", std::process::id())
}

/// FNV-1a over the key, so a tensor name becomes a filesystem-safe filename without a path
/// separator or a length limit leaking through. A cache collision would return the wrong bytes,
/// so the 64-bit space is the safety margin; for a stage's few thousand tensors it is ample.
fn key_to_filename(key: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

impl WeightCache {
    /// A fresh ephemeral cache under `parent`, holding at most `capacity` bytes. The directory is
    /// created now and removed when the cache is dropped. `parent` comes from configuration - the
    /// node's scratch disk - never a hardcoded path.
    pub fn ephemeral(parent: &Path, capacity: u64) -> io::Result<Self> {
        let dir = parent.join(format!("loken-weights-{}", unique_suffix()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            capacity,
            own_dir: true,
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                clock: 0,
                used: 0,
            }),
        })
    }

    /// A cache over an existing directory the caller owns; dropping the cache leaves the directory
    /// in place. For tests and for a run that wants to keep what it cached.
    pub fn at(dir: PathBuf, capacity: u64) -> io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            capacity,
            own_dir: false,
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                clock: 0,
                used: 0,
            }),
        })
    }

    /// The directory the cache writes into.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Bytes currently held on disk.
    pub fn bytes_used(&self) -> u64 {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).used
    }

    /// How many tensors are cached.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .len()
    }

    /// Whether nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The file holding `key`, if cached, marking it freshly used. `None` is a miss the caller
    /// answers by fetching and then `store`-ing.
    pub fn path(&self, key: &str) -> Option<PathBuf> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.clock += 1;
        let now = inner.clock;
        match inner.entries.get_mut(key) {
            Some(e) => {
                e.last = now;
                Some(e.file.clone())
            }
            None => None,
        }
    }

    /// The bytes of `key`, if cached, marking it freshly used.
    pub fn load(&self, key: &str) -> io::Result<Option<Vec<u8>>> {
        match self.path(key) {
            Some(p) => Ok(Some(std::fs::read(p)?)),
            None => Ok(None),
        }
    }

    /// Write `bytes` under `key`, evicting the least recently used until they fit, and return the
    /// file they now live in. Storing a key already present replaces it. A single tensor larger
    /// than the whole budget is still stored - evicting everything else cannot make room, and
    /// refusing it would stall the stage that needs it - so the budget is a target the working set
    /// respects, not a hard ceiling one oversized tensor can breach.
    pub fn store(&self, key: &str, bytes: &[u8]) -> io::Result<PathBuf> {
        self.store_writer(key, |f| {
            f.write_all(bytes)?;
            Ok(bytes.len() as u64)
        })
    }

    /// Store the tensor a `write` closure produces, streaming its bytes straight to the cache file
    /// rather than holding them whole in memory, and returning the file. The closure writes to a
    /// temporary file and returns the byte count; the file is then renamed into place and recorded.
    /// The write is not fsynced: this is a scratch cache, and a byte lost to a crash is re-fetched,
    /// not corrupted state - fsyncing gigabytes per tensor would dominate a stage's load.
    pub fn store_writer(
        &self,
        key: &str,
        write: impl FnOnce(&mut std::fs::File) -> io::Result<u64>,
    ) -> io::Result<PathBuf> {
        let file = self.dir.join(key_to_filename(key));
        let tmp = self.dir.join(format!("{}.tmp", key_to_filename(key)));
        let size = {
            let mut f = std::fs::File::create(&tmp)?;
            let n = write(&mut f)?;
            f.flush()?;
            n
        };
        std::fs::rename(&tmp, &file)?;
        self.record(key, file.clone(), size);
        Ok(file)
    }

    /// Record a stored file under `key`, evicting the least recently used until it fits the budget.
    fn record(&self, key: &str, file: PathBuf, size: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.clock += 1;
        let now = inner.clock;

        // Replacing an existing key: drop its old size before counting the new one.
        if let Some(old) = inner.entries.remove(key) {
            inner.used = inner.used.saturating_sub(old.size);
        }

        // Evict the least recently used until the newcomer fits, but never evict the key we are
        // about to insert and never loop forever once the cache is empty.
        while inner.used + size > self.capacity && !inner.entries.is_empty() {
            let victim = inner
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last)
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    if let Some(e) = inner.entries.remove(&k) {
                        inner.used = inner.used.saturating_sub(e.size);
                        let _ = std::fs::remove_file(&e.file);
                    }
                }
                None => break,
            }
        }

        inner.used += size;
        inner.entries.insert(
            key.to_string(),
            Entry {
                file,
                size,
                last: now,
            },
        );
    }
}

impl Drop for WeightCache {
    fn drop(&mut self) {
        // Ephemeral: the directory is this run's alone, so removing it is the promise that nothing
        // is left behind. A cache opened `at` a caller's directory leaves it untouched.
        if self.own_dir {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> PathBuf {
        std::env::temp_dir().join(format!("loken-wc-test-{}", unique_suffix()))
    }

    #[test]
    fn stores_and_reads_back_the_bytes() {
        let cache = WeightCache::ephemeral(&scratch(), 1 << 20).unwrap();
        cache.store("layers.3.attn.wq_a", &[1, 2, 3, 4]).unwrap();
        assert_eq!(
            cache.load("layers.3.attn.wq_a").unwrap().unwrap(),
            vec![1, 2, 3, 4]
        );
        assert!(cache.load("absent").unwrap().is_none());
        assert_eq!(cache.bytes_used(), 4);
    }

    #[test]
    fn the_budget_evicts_the_least_recently_used() {
        // Room for two 100-byte tensors, not three.
        let cache = WeightCache::ephemeral(&scratch(), 250).unwrap();
        let blob = vec![0u8; 100];
        cache.store("a", &blob).unwrap();
        cache.store("b", &blob).unwrap();
        // Touch a, so b is now the least recently used.
        assert!(cache.path("a").is_some());
        cache.store("c", &blob).unwrap();
        assert!(
            cache.load("a").unwrap().is_some(),
            "a was touched, must survive"
        );
        assert!(
            cache.load("b").unwrap().is_none(),
            "b was coldest, must be evicted"
        );
        assert!(cache.load("c").unwrap().is_some());
        assert!(cache.bytes_used() <= 250);
    }

    #[test]
    fn replacing_a_key_does_not_grow_the_cache() {
        let cache = WeightCache::ephemeral(&scratch(), 1 << 20).unwrap();
        cache.store("k", &[0u8; 100]).unwrap();
        cache.store("k", &[0u8; 100]).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes_used(), 100);
    }

    #[test]
    fn an_oversized_tensor_is_still_cached() {
        // One tensor larger than the whole budget: evicting everything cannot make room, and the
        // stage still needs it, so it is stored rather than refused.
        let cache = WeightCache::ephemeral(&scratch(), 50).unwrap();
        cache.store("big", &[0u8; 200]).unwrap();
        assert!(cache.load("big").unwrap().is_some());
    }

    #[test]
    fn dropping_an_ephemeral_cache_removes_its_directory() {
        let cache = WeightCache::ephemeral(&scratch(), 1 << 20).unwrap();
        let dir = cache.dir().to_path_buf();
        cache.store("k", &[1, 2, 3]).unwrap();
        assert!(dir.exists());
        drop(cache);
        assert!(!dir.exists(), "an ephemeral cache leaves nothing behind");
    }
}
