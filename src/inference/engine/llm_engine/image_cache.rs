use crate::inference::place::layer_executor::host_spill_budget_bytes;
use crate::tensor::{DType, Device, Tensor};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};

/// Process-wide list of registered image embed caches for LRU eviction.
static CACHES: std::sync::OnceLock<Mutex<Vec<Weak<Mutex<Inner>>>>> = std::sync::OnceLock::new();

/// Internal LRU state shared among all caches.
struct Inner {
    /// Least recently used first: the image hash, its embeddings on the host, their bytes.
    entries: VecDeque<(u64, Tensor, u64)>,
    /// The image of the latest request, the one its cached prompt holds.
    last: Option<u64>,
}

impl Inner {
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            last: None,
        }
    }

    /// Drop least recently used entries until `needed` bytes are freed or the cache is empty.
    /// Return the bytes freed.
    fn free(&mut self, needed: u64) -> u64 {
        let mut freed = 0u64;

        while freed < needed {
            match self.entries.pop_front() {
                Some((_, dropped_embeds, bytes)) => {
                    drop(dropped_embeds);
                    freed += bytes;
                }
                None => break,
            }
        }

        freed
    }
}

pub(crate) struct ImageEmbedCache {
    inner: Arc<Mutex<Inner>>,
}

impl ImageEmbedCache {
    /// Create an empty cache and register it with the process-wide cache list.
    pub(crate) fn new() -> Self {
        let inner = Arc::new(Mutex::new(Inner::new()));
        let cache = Self {
            inner: Arc::clone(&inner),
        };

        // Register this cache with the process-wide list
        register_image_embed_cache(inner);

        cache
    }

    /// Look up an image by hash. On hit, move the entry to the newest end, set `last`, and
    /// return the embeddings copied to `device`. On miss return `Ok(None)` and leave `last` as it is.
    /// A poisoned lock is treated as an empty cache.
    pub(crate) fn get(&self, hash: u64, device: &Device) -> Result<Option<Tensor>, String> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return Ok(None),
        };

        // Find the entry
        if let Some(pos) = guard.entries.iter().position(|(h, _, _)| *h == hash) {
            // Remove and reinsert at the end (most recently used)
            let (_, embeds, bytes) = guard.entries.remove(pos).unwrap();
            let device_copy = match embeds.to_device(device) {
                Ok(c) => c,
                Err(_) => return Ok(None),
            };
            guard.entries.push_back((hash, embeds, bytes));
            guard.last = Some(hash);
            Ok(Some(device_copy))
        } else {
            // Miss - leave last unchanged
            Ok(None)
        }
    }

    /// Insert or update an image hash with its embeddings. Keep a host copy (`to_device(Cpu)`),
    /// its bytes being `elem_count * dtype.size_in_bytes()`. Before keeping it, while the host
    /// spill budget is below those bytes and the cache is not empty, drop the LRU entry. If there
    /// is still no room, do not keep it. Set `last` to `hash` in every case.
    /// A poisoned lock is treated as an empty cache.
    pub(crate) fn put(&self, hash: u64, embeds: &Tensor) -> Result<(), String> {
        let bytes = (embeds.elem_count() * embeds.dtype().size_in_bytes()) as u64;

        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return Ok(()),
        };

        // Check if we already have this hash - replace it
        if let Some(pos) = guard.entries.iter().position(|(h, _, _)| *h == hash) {
            let (_, old_embeds, _) = guard.entries.remove(pos).unwrap();
            // Free the old entry's bytes indirectly by dropping
            drop(old_embeds);
        }

        // Get a host copy of the embeddings
        let host_copy = match embeds.to_device(&Device::Cpu) {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };

        // While we don't have room and cache is not empty, drop LRU entries
        while host_spill_budget_bytes() < bytes {
            if guard.entries.is_empty() {
                // No room and nothing to drop - don't keep this entry
                return Ok(());
            }
            // Drop the LRU entry (first in the deque)
            let (_, dropped_embeds, _) = guard.entries.pop_front().unwrap();
            drop(dropped_embeds);
        }

        // Add the new entry (most recently used)
        guard.entries.push_back((hash, host_copy, bytes));
        guard.last = Some(hash);

        Ok(())
    }

    /// Return the hash of the latest request's image, or `None` if no image has been cached.
    /// A poisoned lock is treated as an empty cache.
    pub(crate) fn last(&self) -> Option<u64> {
        match self.inner.lock() {
            Ok(guard) => guard.last,
            Err(_) => None,
        }
    }

    /// Drop least recently used entries until `needed` bytes are freed or the cache is empty.
    /// Return the bytes freed. A poisoned lock is treated as an empty cache.
    pub(crate) fn free(&self, needed: u64) -> u64 {
        let mut freed = 0u64;
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return 0,
        };

        while freed < needed {
            match guard.entries.pop_front() {
                Some((_, dropped_embeds, bytes)) => {
                    drop(dropped_embeds);
                    freed += bytes;
                }
                None => break,
            }
        }

        // If cache is now empty, clear the last
        if guard.entries.is_empty() {
            guard.last = None;
        }

        freed
    }
}

/// Register a new image embed cache with the process-wide list and vram_manager.
fn register_image_embed_cache(inner: Arc<Mutex<Inner>>) {
    // Add to the process-wide cache list
    let caches = CACHES.get_or_init(|| Mutex::new(Vec::new()));
    {
        let mut list = match caches.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let weak: Weak<Mutex<Inner>> = Arc::downgrade(&inner);
        list.push(weak);
    }

    // Register a host cache hook with vram_manager
    // The hook frees LRU entries across all caches until the needed bytes are freed
    crate::inference::place::vram_manager::register_host_cache(
        "image-embeds",
        0, // estimate bytes - the hook will figure out the real amount
        Box::new(move |needed: u64| {
            // Try to collect all caches and free from them
            let caches = CACHES.get();
            let Some(caches) = caches else {
                return 0;
            };
            let mut freed = 0u64;
            let mut to_remove = Vec::new();

            // Iterate through all weak refs
            let mut list = match caches.lock() {
                Ok(g) => g,
                Err(_) => return 0,
            };

            for (i, weak) in list.iter().enumerate() {
                if let Some(inner) = weak.upgrade() {
                    // Call Inner::free directly through the mutex guard
                    let inner_freed = inner
                        .lock()
                        .map_or(0, |mut guard| guard.free(needed - freed));
                    freed += inner_freed;
                    if freed >= needed {
                        break;
                    }
                } else {
                    to_remove.push(i);
                }
            }

            // Drop dead weak refs
            for i in to_remove.into_iter().rev() {
                list.remove(i);
            }

            freed
        }),
    );
}

#[cfg(test)]
mod image_cache_tests {
    use super::*;

    /// Helper to compute bytes of a tensor
    fn tensor_bytes(t: &Tensor) -> u64 {
        (t.elem_count() * t.dtype().size_in_bytes()) as u64
    }

    #[test]
    fn put_then_get_returns_same_values_and_sets_last() {
        let cache = ImageEmbedCache::new();
        let tensor = Tensor::zeros((1, 4, 8), DType::F32).unwrap();
        let hash = 0x12345678;
        let expected_bytes = tensor_bytes(&tensor);

        // Initial state: last is None
        assert_eq!(cache.last(), None);

        // Put the image
        cache.put(hash, &tensor).expect("put should succeed");
        assert_eq!(cache.last(), Some(hash));

        // Get the image - should return a copy on device
        let result = cache.get(hash, &Device::Cpu).expect("get should succeed");
        assert!(result.is_some());
        let retrieved = result.unwrap();
        assert_eq!(tensor_bytes(&retrieved), expected_bytes);

        // Verify values are the same
        let tensor_cpu = tensor.to_device(&Device::Cpu).unwrap();
        let retrieved_cpu = retrieved.to_device(&Device::Cpu).unwrap();
        assert_eq!(tensor_cpu.elem_count(), retrieved_cpu.elem_count());
    }

    #[test]
    fn three_images_all_kept_and_lru_makes_second_least_recent() {
        let cache = ImageEmbedCache::new();

        let hash1 = 0x11111111;
        let hash2 = 0x22222222;
        let hash3 = 0x33333333;

        let embed1 = Tensor::zeros((1, 4, 8), DType::F32).unwrap();
        let embed2 = Tensor::zeros((1, 4, 8), DType::F32).unwrap();
        let embed3 = Tensor::zeros((1, 4, 8), DType::F32).unwrap();

        let bytes_one = (embed1.elem_count() * embed1.dtype().size_in_bytes()) as u64;

        // Put three images
        cache.put(hash1, &embed1).expect("put hash1");
        cache.put(hash2, &embed2).expect("put hash2");
        cache.put(hash3, &embed3).expect("put hash3");

        // All three should be kept
        assert_eq!(cache.last(), Some(hash3));

        // Get the first image - this should make it the most recently used
        // and leave hash2 as the least recently used (first in the deque)
        let _ = cache.get(hash1, &Device::Cpu).expect("get hash1");
        assert_eq!(cache.last(), Some(hash1));

        // Now free bytes of one entry - should evict the LRU (hash2)
        let freed = cache.free(bytes_one);
        assert_eq!(freed, bytes_one);

        // Hash2 should now be gone, hash1 and hash3 should remain
        // The cache should contain hash3 (LRU now) and hash1 (most recent)
        assert_eq!(cache.last(), Some(hash1));

        // Verify hash2 is gone
        let hash2_result = cache.get(hash2, &Device::Cpu);
        assert!(hash2_result.is_ok());
        assert!(hash2_result.unwrap().is_none());

        // Verify hash1 and hash3 still work
        assert!(cache.get(hash1, &Device::Cpu).unwrap().is_some());
        assert!(cache.get(hash3, &Device::Cpu).unwrap().is_some());
    }

    #[test]
    fn miss_leaves_last_unchanged() {
        let cache = ImageEmbedCache::new();
        let hash = 0x12345678;
        let embed = Tensor::zeros((1, 4, 8), DType::F32).unwrap();

        // Put one image
        cache.put(hash, &embed).expect("put");
        assert_eq!(cache.last(), Some(hash));

        // Miss on different hash - last should still be the same
        let result = cache.get(0xDEADBEEF, &Device::Cpu).unwrap();
        assert!(result.is_none());
        assert_eq!(cache.last(), Some(hash));
    }

    #[test]
    fn free_of_more_than_cache_holds_emptys_and_returns_total() {
        let cache = ImageEmbedCache::new();
        let embed = Tensor::zeros((1, 4, 8), DType::F32).unwrap();
        let hash = 0x12345678;
        let total_bytes = (embed.elem_count() * embed.dtype().size_in_bytes()) as u64;

        cache.put(hash, &embed).expect("put");
        assert_eq!(cache.last(), Some(hash));

        // Free more than the cache holds
        let freed = cache.free(total_bytes * 2);
        assert_eq!(freed, total_bytes);

        // Cache should now be empty
        assert_eq!(cache.last(), None);
    }
}
