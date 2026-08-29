//! Small in-memory LRU keyed by `String`. Used by the image-gen
//! pipelines (Flux + Z-Image) to cache prompt text-encoder outputs
//! and img2img VAE-encoded init latents, and by the TTS engine to
//! cache the T5-encoded voice description across repeated synth
//! calls. Pulled into a shared module so the same idiom doesn't
//! drift across engines.

/// Small LRU keyed by `String`. Lookup is linear (cap is tiny on
/// purpose); hits bump to the front; inserts evict from the back
/// when over `cap`. Values stay generic: callers pick what to
/// store per slot (a tensor pair, a tensor + token count, etc.).
///
/// Memory bound is `cap x sizeof(V)` - `V` typically holds Arc-
/// backed `Tensor`s so per-slot cost is one storage handle plus
/// a few bytes of metadata, not the full backing buffer.
pub(crate) struct PromptTextCache<V: Clone> {
    entries: Vec<(String, V)>,
    cap: usize,
}

/// Default cache capacity for image-gen prompt text encodings.
/// Four covers the common "compare two phrasings" / "tweak one
/// word and flip back" pattern without keeping a long tail of
/// stale embeddings warm.
pub(crate) const PROMPT_TEXT_CACHE_CAP: usize = 4;

/// Default cache capacity for img2img VAE-encoded init latents.
/// Two covers "upload an image, try this prompt, undo, try that
/// prompt" - the common img2img flow. Latents are ~256 KB at
/// 1024x1024 so per-slot RAM is cheap, but two is enough since
/// users rarely flip between three or more source images in a row.
pub(crate) const IMG2IMG_LATENT_CACHE_CAP: usize = 2;

/// Default cache capacity for TTS voice-description encodings.
/// Most users repeat the same voice across many synthesis calls
/// (a podcast script, a chat conversation), so even one slot
/// covers the common path. Four leaves room for A/B comparing
/// alternative voice descriptions without thrashing.
pub(crate) const TTS_DESCRIPTION_CACHE_CAP: usize = 4;

impl<V: Clone> PromptTextCache<V> {
    pub fn new(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
            cap,
        }
    }

    /// Linear scan over `entries` (capacity is tiny). On hit, moves
    /// the entry to the front (MRU) and returns a clone of its value.
    pub fn get(&mut self, key: &str) -> Option<V> {
        let pos = self.entries.iter().position(|(k, _)| k == key)?;
        if pos != 0 {
            let entry = self.entries.remove(pos);
            self.entries.insert(0, entry);
        }
        Some(self.entries[0].1.clone())
    }

    /// Insert at the front. Existing entry with the same key is
    /// replaced (no duplicates). Evicts from the back until the
    /// occupancy fits `cap`.
    pub fn insert(&mut self, key: String, value: V) {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
            self.entries.remove(pos);
        }
        self.entries.insert(0, (key, value));
        while self.entries.len() > self.cap {
            self.entries.pop();
        }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub fn keys_front_to_back(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }
}

/// Build a stable cache key from the base64 input image plus the
/// target output dimensions. Different sizes hit different VAE
/// encoder graphs so they MUST occupy separate cache slots. We hash
/// the base64 string rather than store it so the cache never holds
/// the user's image bytes verbatim - only a fixed-width digest.
pub(crate) fn vae_latent_cache_key(input_b64: &str, height: usize, width: usize) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input_b64.hash(&mut hasher);
    let h = hasher.finish();
    format!("{h:016x}-{width}x{height}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- PromptTextCache ---------------------------------------------
    // Capacity-bounded LRU for any per-prompt encoder output. Tests
    // pin behavior so the perf gain (skip a 50-200 ms text-encoder
    // pass on prompt flip) doesn't regress to a single-slot cache
    // or grow unbounded.

    #[test]
    fn prompt_cache_miss_then_hit_returns_inserted_value() {
        let mut c: PromptTextCache<u32> = PromptTextCache::new(4);
        assert!(c.get("a").is_none(), "fresh cache misses");
        c.insert("a".to_string(), 7);
        assert_eq!(c.get("a"), Some(7), "same key returns inserted value");
    }

    #[test]
    fn prompt_cache_back_and_forth_hits_both_sides() {
        // The core A/B-flip use case: alternating between two prompts
        // must keep both warm so neither pays the encoder cost twice.
        let mut c: PromptTextCache<u32> = PromptTextCache::new(4);
        c.insert("a".to_string(), 1);
        c.insert("b".to_string(), 2);
        // Flip a -> b -> a -> b: every lookup should hit.
        assert_eq!(c.get("a"), Some(1));
        assert_eq!(c.get("b"), Some(2));
        assert_eq!(c.get("a"), Some(1));
        assert_eq!(c.get("b"), Some(2));
        assert_eq!(c.len(), 2, "no spurious eviction during alternation");
    }

    #[test]
    fn prompt_cache_get_bumps_entry_to_front() {
        // MRU semantics: a get() must move the hit to the front so
        // it survives subsequent eviction at the back.
        let mut c: PromptTextCache<u32> = PromptTextCache::new(3);
        c.insert("a".to_string(), 1);
        c.insert("b".to_string(), 2);
        c.insert("c".to_string(), 3);
        // After 3 inserts, order is [c, b, a]. Touching "a" must bump
        // it to the front so the next eviction drops "b" not "a".
        let _ = c.get("a");
        assert_eq!(c.keys_front_to_back(), vec!["a", "c", "b"]);
    }

    #[test]
    fn prompt_cache_evicts_oldest_when_over_capacity() {
        // Insert past capacity - the least-recently-used entry at the
        // tail must drop, not a middle entry.
        let mut c: PromptTextCache<u32> = PromptTextCache::new(2);
        c.insert("a".to_string(), 1);
        c.insert("b".to_string(), 2);
        c.insert("c".to_string(), 3);
        assert_eq!(c.len(), 2, "capacity bounded");
        assert_eq!(c.keys_front_to_back(), vec!["c", "b"], "tail entry evicted");
        assert!(c.get("a").is_none(), "evicted entry no longer reachable");
    }

    #[test]
    fn prompt_cache_reinsert_replaces_existing_value() {
        // Same key twice = updated value, position bumped to front,
        // no duplicate entries.
        let mut c: PromptTextCache<u32> = PromptTextCache::new(4);
        c.insert("a".to_string(), 1);
        c.insert("b".to_string(), 2);
        c.insert("a".to_string(), 99);
        assert_eq!(c.len(), 2, "no duplicate entries on re-insert");
        assert_eq!(c.get("a"), Some(99), "value updated to latest insert");
        assert_eq!(c.keys_front_to_back(), vec!["a", "b"]);
    }

    #[test]
    fn prompt_cache_capacity_one_behaves_like_single_slot() {
        // Capacity-1 special case: matches the prior single-slot
        // behavior - every new prompt evicts the previous.
        let mut c: PromptTextCache<u32> = PromptTextCache::new(1);
        c.insert("a".to_string(), 1);
        c.insert("b".to_string(), 2);
        assert!(
            c.get("a").is_none(),
            "single-slot drops prior on new insert"
        );
        assert_eq!(c.get("b"), Some(2));
    }

    // -- vae_latent_cache_key ----------------------------------------
    // The img2img latent cache keys are content-addressed (b64 hash)
    // + dimension-tagged. Tests pin: identical inputs collide, any
    // input change diverges. Keeps the cache from serving stale
    // latents at the wrong shape or for a different image.

    #[test]
    fn vae_key_same_input_same_size_yields_same_key() {
        // Cache MUST hit when the user re-submits the exact same
        // input image at the same target dimensions.
        let k1 = vae_latent_cache_key("aGVsbG8=", 1024, 1024);
        let k2 = vae_latent_cache_key("aGVsbG8=", 1024, 1024);
        assert_eq!(k1, k2);
    }

    #[test]
    fn vae_key_different_input_diverges() {
        // Different b64 -> different hash -> different cache slot.
        // Otherwise a different user image would serve another
        // user's cached latent.
        let k1 = vae_latent_cache_key("aGVsbG8=", 1024, 1024);
        let k2 = vae_latent_cache_key("d29ybGQ=", 1024, 1024);
        assert_ne!(k1, k2);
    }

    #[test]
    fn vae_key_different_dimensions_diverge() {
        // Same image at 512x512 cannot serve 1024x1024 - the cached
        // latent has the wrong tensor shape, would crash at the noise
        // mix. Key must split on size.
        let same_img = "aGVsbG8=";
        let k_small = vae_latent_cache_key(same_img, 512, 512);
        let k_large = vae_latent_cache_key(same_img, 1024, 1024);
        assert_ne!(k_small, k_large);
    }

    #[test]
    fn vae_key_swapped_h_w_diverge() {
        // 1024x768 and 768x1024 differ - portrait vs landscape
        // produce different latent shapes.
        let same_img = "aGVsbG8=";
        let k_landscape = vae_latent_cache_key(same_img, 768, 1024);
        let k_portrait = vae_latent_cache_key(same_img, 1024, 768);
        assert_ne!(k_landscape, k_portrait);
    }
}
