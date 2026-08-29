//! Paged KV block allocator - the foundation for continuous batching / paged
//! attention (vLLM-lineage), the throughput lever vs vLLM under concurrent load.
//!
//! The model today is batch=1, single-sequence (one contiguous KV cache per
//! request, one in-flight request per engine). Continuous batching needs MANY
//! sequences sharing one engine's KV memory, each growing independently - which
//! a single contiguous cache can't do without over-allocating to max-context per
//! slot. Paged attention solves it like an OS virtual-memory manager: KV is a
//! pool of fixed-size physical BLOCKS (`block_size` tokens each); each sequence
//! holds a BLOCK TABLE (its logical->physical block list), and blocks are
//! allocated on demand and freed on completion. No fragmentation, no per-slot
//! max-context waste -> many more concurrent sequences fit in the same VRAM.
//!
//! This module owns ONLY the index bookkeeping (which physical block backs each
//! logical position, and the per-token `slot_mapping` the write/attention
//! kernels consume). The actual KV tensor - one `[num_blocks, block_size, n_kv,
//! head_dim]` buffer - and the paged-attention kernel are layered on top later;
//! keeping the allocator pure makes it deterministic and unit-testable with no
//! model, no GPU, no heat. Mirrors vLLM's BlockSpaceManager / block table.

use std::collections::HashMap;

/// Opaque per-sequence handle (the engine's request id maps to one of these).
pub type SeqId = u64;

/// Allocation outcome - the caller (scheduler) uses `OutOfBlocks` to decide
/// whether to admit a request this step or queue it (continuous batching admits
/// only what fits, so a hot decode step never half-allocates).
#[derive(Debug, PartialEq, Eq)]
pub enum AllocErr {
    OutOfBlocks { needed: usize, free: usize },
    UnknownSeq(SeqId),
}

/// One sequence's paged state: its ordered physical block list + how many
/// tokens currently occupy the last block (so we know when to grow).
#[derive(Debug, Clone, Default)]
struct SeqBlocks {
    blocks: Vec<u32>, // logical block i -> physical block blocks[i]
    len: usize,       // tokens currently stored
    /// Prefix-cache hashes this sequence holds a ref on - one per LEADING full
    /// prompt block that is cache-managed (acquired from the cache OR computed-then-
    /// inserted). `held_hashes.len()` is the count of leading cache-managed blocks;
    /// on free, each is `release`d (the physical block returns to `free` only when
    /// the last ref drops). The remaining blocks are seq-owned (partial prompt tail
    /// + decode growth) and freed directly. Empty unless prefix caching is enabled.
    held_hashes: Vec<u64>,
}

/// Fixed-size-block KV allocator. All sizes are in BLOCKS except `len`/token
/// counts. Single-threaded by design (driven by the engine's scheduler loop);
/// wrap in the engine's existing lock if shared.
pub struct PagedKvAllocator {
    block_size: usize,
    num_blocks: usize,
    free: Vec<u32>, // free physical block ids (LIFO; warm-cache reuse)
    seqs: HashMap<SeqId, SeqBlocks>,
    /// Optional automatic prefix cache (vLLM-style). When `Some`, a new sequence's
    /// shared block-aligned prompt prefix reuses already-computed KV blocks instead
    /// of recomputing them. `None` (default) = the plain single-owner allocator,
    /// byte-for-byte the original behavior.
    prefix: Option<PrefixCache>,
}

impl PagedKvAllocator {
    /// `num_blocks` physical blocks of `block_size` tokens each. Both > 0.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(num_blocks > 0, "num_blocks must be > 0");
        // Hand out low ids first (LIFO free list -> a just-freed block is reused
        // next, keeping recently-touched physical pages warm in cache).
        let free = (0..num_blocks as u32).rev().collect();
        Self {
            block_size,
            num_blocks,
            free,
            seqs: HashMap::new(),
            prefix: None,
        }
    }

    /// Same pool, but with automatic prefix caching enabled.
    pub fn new_with_prefix(num_blocks: usize, block_size: usize) -> Self {
        let mut a = Self::new(num_blocks, block_size);
        a.prefix = Some(PrefixCache::new(block_size));
        a
    }

    pub fn prefix_enabled(&self) -> bool {
        self.prefix.is_some()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }
    pub fn used_blocks(&self) -> usize {
        self.num_blocks - self.free.len()
    }

    /// Blocks needed to hold `tokens` tokens.
    #[inline]
    fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size)
    }

    /// Can a sequence currently `seq_len` tokens grow by `add` more without
    /// running out of physical blocks? (Scheduler admission test - never starts a
    /// decode step it can't finish.)
    pub fn can_append(&self, seq_len: usize, add: usize) -> bool {
        let have = self.blocks_for(seq_len);
        let need = self.blocks_for(seq_len + add);
        // Idle (refs==0) cached prefix blocks are reclaimable on demand, so they
        // count toward capacity alongside the truly-free blocks.
        let reclaimable = self.free.len() + self.prefix.as_ref().map_or(0, |p| p.evictable());
        need - have <= reclaimable
    }

    /// Make at least `need` blocks free, evicting idle (refs==0) cached prefix
    /// blocks LRU-first if the free pool is short. No-op without prefix caching.
    fn ensure_free(&mut self, need: usize) {
        while self.free.len() < need {
            let evicted = match self.prefix.as_mut() {
                Some(pc) => pc.evict_one(),
                None => None,
            };
            match evicted {
                Some(b) => self.free.push(b),
                None => break,
            }
        }
    }

    /// Register a new sequence and reserve blocks for its `prompt_tokens` prefill.
    /// Idempotent-unsafe: the `seq` must not already exist.
    pub fn allocate(&mut self, seq: SeqId, prompt_tokens: usize) -> Result<(), AllocErr> {
        debug_assert!(!self.seqs.contains_key(&seq), "seq {seq} already allocated");
        let need = self.blocks_for(prompt_tokens);
        self.ensure_free(need);
        if need > self.free.len() {
            return Err(AllocErr::OutOfBlocks {
                needed: need,
                free: self.free.len(),
            });
        }
        let mut blocks = Vec::with_capacity(need);
        for _ in 0..need {
            blocks.push(self.free.pop().expect("free checked above"));
        }
        self.seqs.insert(
            seq,
            SeqBlocks {
                blocks,
                len: prompt_tokens,
                held_hashes: Vec::new(),
            },
        );
        Ok(())
    }

    /// Prefix-cache-aware allocation for a new sequence's `prompt` prefill. Reuses
    /// the physical blocks of the longest cached block-aligned prompt prefix (no
    /// recompute, no fresh blocks consumed for it) and allocates fresh blocks only
    /// for the uncached suffix. Returns the number of CACHED prompt tokens - the
    /// position the prefill forward should START at (it must compute only
    /// `prompt[cached_tokens..]`, attending over the full block table). Requires the
    /// allocator to have prefix caching enabled (`new_with_prefix`); otherwise falls
    /// back to a plain `allocate` and returns 0. The fresh suffix blocks are
    /// published to the cache later via [`commit_prefix`] (after prefill computes
    /// their KV). The caller MUST `commit_prefix` then eventually `free`.
    pub fn allocate_with_prefix(&mut self, seq: SeqId, prompt: &[u32]) -> Result<usize, AllocErr> {
        debug_assert!(!self.seqs.contains_key(&seq), "seq {seq} already allocated");
        let prompt_tokens = prompt.len();
        if self.prefix.is_none() {
            self.allocate(seq, prompt_tokens)?;
            return Ok(0);
        }
        // Acquire the longest contiguous cached prefix (refs incremented in the
        // cache). Borrow of `self.prefix` is scoped to this statement.
        let (mut cached_blocks, mut held) = self.prefix.as_mut().unwrap().acquire_prefix(prompt);
        // NEVER cache the ENTIRE prompt: the prefill must compute at least the last
        // token to produce the next-token logits. If the cached prefix covers every
        // token (block-aligned prompt fully cached), drop the last cached block so a
        // >=1-token suffix remains - otherwise tsfx==0 -> an empty-tensor forward
        // (CUDA cast INVALID_VALUE that kills the worker).
        if cached_blocks.len() * self.block_size >= prompt_tokens {
            if let Some(h) = held.pop() {
                cached_blocks.pop();
                self.prefix.as_mut().unwrap().release(h);
            }
        }
        let cached_tokens = cached_blocks.len() * self.block_size;
        let total_blocks = prompt_tokens.div_ceil(self.block_size);
        let fresh_need = total_blocks - cached_blocks.len();
        // Make room for the suffix, evicting idle cached blocks if the pool is short.
        self.ensure_free(fresh_need);
        if fresh_need > self.free.len() {
            for h in &held {
                self.prefix.as_mut().unwrap().release(*h);
            } // roll back
            return Err(AllocErr::OutOfBlocks {
                needed: fresh_need,
                free: self.free.len(),
            });
        }
        let mut blocks = cached_blocks;
        for _ in 0..fresh_need {
            blocks.push(self.free.pop().expect("ensure_free covered this"));
        }
        self.seqs.insert(
            seq,
            SeqBlocks {
                blocks,
                len: prompt_tokens,
                held_hashes: held,
            },
        );
        Ok(cached_tokens)
    }

    /// Publish a sequence's freshly-computed FULL prompt blocks to the prefix cache
    /// (call once after prefill has written their KV). For each leading full prompt
    /// block not already cache-managed, `try_insert` it under its prefix hash so
    /// later requests can reuse it. On an insert race (another seq cached the same
    /// prefix first), adopt the winner's physical block, return our duplicate to the
    /// free pool, and rewrite our block table - so all sharers converge on one block.
    pub fn commit_prefix(&mut self, seq: SeqId, prompt: &[u32]) {
        let Some(pc) = self.prefix.as_mut() else {
            return;
        };
        let Some(s) = self.seqs.get_mut(&seq) else {
            return;
        };
        let hashes = block_hashes(prompt, self.block_size);
        let already = s.held_hashes.len(); // leading blocks already cache-managed
        for (bi, &h) in hashes.iter().enumerate().skip(already) {
            let phys = s.blocks[bi];
            if pc.try_insert(h, phys) {
                s.held_hashes.push(h);
            } else {
                // Lost the race: adopt the single cached winner, free our duplicate.
                let winner = pc.acquire_one(h).expect("insert lost -> entry present");
                self.free.push(phys);
                s.blocks[bi] = winner;
                s.held_hashes.push(h);
            }
        }
    }

    /// Append `add` decoded tokens to `seq`, growing its block table on block
    /// boundaries. Returns the per-token `slot_mapping` (flat KV slot index =
    /// physical_block * block_size + offset) for the appended tokens, in order  - 
    /// exactly what the KV-write kernel needs.
    pub fn append(&mut self, seq: SeqId, add: usize) -> Result<Vec<usize>, AllocErr> {
        // Pre-check growth so we never half-grow (scheduler relies on atomicity).
        let cur_len = self.seqs.get(&seq).ok_or(AllocErr::UnknownSeq(seq))?.len;
        let have = self.blocks_for(cur_len);
        let need_total = self.blocks_for(cur_len + add);
        let grow = need_total - have;
        self.ensure_free(grow);
        if grow > self.free.len() {
            return Err(AllocErr::OutOfBlocks {
                needed: grow,
                free: self.free.len(),
            });
        }
        for _ in 0..grow {
            let b = self.free.pop().expect("free checked above");
            self.seqs.get_mut(&seq).unwrap().blocks.push(b);
        }
        let s = self.seqs.get_mut(&seq).unwrap();
        let mut slots = Vec::with_capacity(add);
        for i in 0..add {
            let pos = cur_len + i;
            let logical = pos / self.block_size;
            let offset = pos % self.block_size;
            let phys = s.blocks[logical] as usize;
            slots.push(phys * self.block_size + offset);
        }
        s.len += add;
        Ok(slots)
    }

    /// The sequence's block table (logical->physical), what paged attention reads
    /// to gather KV across non-contiguous blocks.
    pub fn block_table(&self, seq: SeqId) -> Option<&[u32]> {
        self.seqs.get(&seq).map(|s| s.blocks.as_slice())
    }

    pub fn seq_len(&self, seq: SeqId) -> Option<usize> {
        self.seqs.get(&seq).map(|s| s.len)
    }

    /// Slot mapping (physical KV slot per token) for logical positions `from..to`
    /// of `seq` - what the prefill KV-write kernel needs for a suffix-only prefill
    /// (the suffix occupies the fresh blocks after the cached prefix).
    pub fn slots_for_range(&self, seq: SeqId, from: usize, to: usize) -> Vec<usize> {
        let s = self.seqs.get(&seq).expect("seq exists");
        (from..to)
            .map(|pos| {
                let blk = s.blocks[pos / self.block_size] as usize;
                blk * self.block_size + pos % self.block_size
            })
            .collect()
    }

    /// Roll a sequence back to `new_len` tokens (speculative-decode reject),
    /// freeing any blocks that are no longer needed. Returns blocks freed.
    pub fn trim(&mut self, seq: SeqId, new_len: usize) -> Result<usize, AllocErr> {
        let s = self.seqs.get_mut(&seq).ok_or(AllocErr::UnknownSeq(seq))?;
        if new_len >= s.len {
            return Ok(0);
        }
        let keep = new_len.div_ceil(self.block_size);
        let mut freed = 0;
        while s.blocks.len() > keep {
            self.free.push(s.blocks.pop().unwrap());
            freed += 1;
        }
        s.len = new_len;
        Ok(freed)
    }

    /// Release all of a finished sequence's blocks back to the pool. Cache-managed
    /// leading blocks (held_hashes) are released through the prefix cache and return
    /// to `free` only when their LAST reference drops (another live sequence sharing
    /// the prefix keeps them resident); the rest are seq-owned and freed directly.
    /// With prefix caching disabled (held_hashes empty) this is the original
    /// "free everything" behavior, byte-for-byte.
    pub fn free(&mut self, seq: SeqId) -> Result<usize, AllocErr> {
        let s = self.seqs.remove(&seq).ok_or(AllocErr::UnknownSeq(seq))?;
        let n = s.blocks.len();
        let cache_managed = s.held_hashes.len();
        // Drop this seq's live refs on the cache-managed leading blocks - they STAY
        // RESIDENT in the cache (reclaimed only by eviction under pressure), so a
        // later request with the same prefix reuses them. Only the seq-owned blocks
        // (partial prompt tail + decode growth) go straight back to the free pool.
        if let Some(pc) = self.prefix.as_mut() {
            for h in &s.held_hashes {
                pc.release(*h);
            }
        }
        self.free.extend(s.blocks[cache_managed..].iter().copied());
        Ok(n)
    }
}

/// Block-aligned, prefix-AWARE content hashes for a token sequence: `hashes[i]`
/// depends on tokens `[0 ..= block i]` (it folds in the previous block's hash), so
/// two prompts sharing their first k full blocks produce identical first-k hashes
/// and diverge at the first differing block - exactly what prefix caching needs.
/// Only FULL blocks are hashed (a partial trailing block isn't cacheable). Mirrors
/// vLLM's prefix-caching block hash. FNV-1a over the token ids + the parent hash.
pub fn block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
    let nfull = tokens.len() / block_size;
    let mut hashes = Vec::with_capacity(nfull);
    let mut parent: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis (root prefix)
    for b in 0..nfull {
        let mut h = parent;
        h ^= 0x9e37_79b9_7f4a_7c15; // separate the parent fold from token folds
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        for &t in &tokens[b * block_size..(b + 1) * block_size] {
            h ^= t as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
        }
        hashes.push(h);
        parent = h;
    }
    hashes
}

/// Content-addressed cache of already-computed KV blocks for shared prompt
/// prefixes (vLLM-style automatic prefix caching). Keyed by [`block_hashes`], so a
/// new request reuses the physical blocks of its longest cached block-aligned
/// prefix instead of recomputing that KV - the fix for multi-turn chat / shared
/// system prompts re-prefilling every turn. Ref-counts each cached block so it is
/// only returned to the allocator's free pool once no live sequence (and no cache
/// entry) references it. PURE bookkeeping (no model/GPU); unit-tested.
/// the structure + matching/refcount logic; the allocator/prefill wiring lands next.
#[derive(Default)]
pub struct PrefixCache {
    block_size: usize,
    map: HashMap<u64, CachedBlock>,
    tick: u64, // monotonic clock for LRU eviction
}

struct CachedBlock {
    block: u32,
    refs: u32, // live sequences currently using this block (0 = cached but idle)
    tick: u64, // last access, for LRU eviction
}

impl PrefixCache {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            map: HashMap::new(),
            tick: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Longest cached block-aligned prefix of `tokens`: the physical blocks (in
    /// order) plus the hashes they were matched on. Each returned block's refcount
    /// is incremented - the caller MUST later [`release`](Self::release) each
    /// returned hash (on sequence free / trim) to avoid leaking the block. Matching
    /// is CONTIGUOUS: it stops at the first uncached block (a hole can't be reused).
    pub fn acquire_prefix(&mut self, tokens: &[u32]) -> (Vec<u32>, Vec<u64>) {
        let hashes = block_hashes(tokens, self.block_size);
        let mut blocks = Vec::new();
        let mut held = Vec::new();
        for h in hashes {
            let tick = self.tick;
            match self.map.get_mut(&h) {
                Some(cb) => {
                    cb.refs += 1;
                    cb.tick = tick;
                    blocks.push(cb.block);
                    held.push(h);
                    self.tick += 1;
                }
                None => break,
            }
        }
        (blocks, held)
    }

    /// Take ONE live ref on the cached block for `hash` (e.g. after losing an
    /// insert race - adopt the winner). Returns its physical block, or None if not
    /// cached. Caller must later `release(hash)`.
    pub fn acquire_one(&mut self, hash: u64) -> Option<u32> {
        let tick = self.tick;
        self.map.get_mut(&hash).map(|cb| {
            cb.refs += 1;
            cb.tick = tick;
            self.tick += 1;
            cb.block
        })
    }

    /// Number of cached blocks with no live user (refs==0) - evictable to reclaim
    /// free blocks under memory pressure.
    pub fn evictable(&self) -> usize {
        self.map.values().filter(|c| c.refs == 0).count()
    }

    /// Evict the least-recently-used IDLE (refs==0) cached block, returning its
    /// physical block id for the caller to return to the free pool. None if every
    /// cached block is still in use. This is what keeps a cached prefix RESIDENT
    /// across sequential requests (chat turns) yet bounded under memory pressure.
    pub fn evict_one(&mut self) -> Option<u32> {
        let victim = self
            .map
            .iter()
            .filter(|(_, c)| c.refs == 0)
            .min_by_key(|(_, c)| c.tick)
            .map(|(&h, _)| h)?;
        self.map.remove(&victim).map(|c| c.block)
    }

    /// Publish a freshly-computed physical `block` under its prefix `hash`, taking
    /// one reference for the producing sequence. Returns true if inserted; false if
    /// another sequence already cached this prefix (a race) - in that case the
    /// caller should release its now-duplicate block back to the allocator and
    /// instead `acquire` the winning one.
    pub fn try_insert(&mut self, hash: u64, block: u32) -> bool {
        use std::collections::hash_map::Entry;
        let tick = self.tick;
        match self.map.entry(hash) {
            Entry::Occupied(_) => false,
            Entry::Vacant(v) => {
                v.insert(CachedBlock {
                    block,
                    refs: 1,
                    tick,
                });
                self.tick += 1;
                true
            }
        }
    }

    /// Drop one live-user reference. The block STAYS CACHED (resident for later
    /// requests) when refs reach 0 - it is reclaimed only by [`evict_one`] under
    /// memory pressure. This persistence is what makes sequential chat turns reuse
    /// a prefix. Never returns a block (eviction is the allocator's job).
    pub fn release(&mut self, hash: u64) {
        if let Some(cb) = self.map.get_mut(&hash) {
            cb.refs = cb.refs.saturating_sub(1);
        }
    }
}

/// The physical KV tensor the allocator's indices address: a flat
/// `[num_blocks * block_size, n_kv_head * head_dim]` buffer for K and one for V.
/// `write` scatters new tokens' KV into their `slot_mapping` rows (from
/// [`PagedKvAllocator::append`]); `gather_seq` collects a sequence's KV
/// contiguously via its block table (logical pos `p` -> physical slot
/// `block_table[p/block_size]*block_size + p%block_size`) so the existing
/// attention can consume it. Device/dtype-agnostic (CPU for tests, CUDA in prod).
pub struct PagedKvStore {
    k: crate::tensor::Tensor,
    v: crate::tensor::Tensor,
    block_size: usize,
    feat: usize, // n_kv_head * head_dim
    // Persistent gather-index scratch (one per store), grown on demand and
    // updated in place each step. A stable device address is the prerequisite
    // for CUDA-graph capture (a fresh `from_vec` per step bakes capture-time
    // indices into the graph -> wrong on replay). Holds u32 slot ids [B*ctx].
    gather_idx: std::cell::RefCell<Option<crate::tensor::Tensor>>,
    // Persistent write-index scratch (the B new tokens' scatter slots [B,feat]).
    write_idx: std::cell::RefCell<Option<crate::tensor::Tensor>>,
}

impl PagedKvStore {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        n_kv_head: usize,
        head_dim: usize,
        dtype: crate::tensor::DType,
        device: &crate::tensor::Device,
    ) -> crate::tensor::Result<Self> {
        let feat = n_kv_head * head_dim;
        let slots = num_blocks * block_size;
        Ok(Self {
            k: crate::tensor::Tensor::zeros_on(&[slots, feat][..], dtype, device)?,
            v: crate::tensor::Tensor::zeros_on(&[slots, feat][..], dtype, device)?,
            block_size,
            feat,
            gather_idx: std::cell::RefCell::new(None),
            write_idx: std::cell::RefCell::new(None),
        })
    }

    /// Like [`write`] but the scatter-index tensor lives in a PERSISTENT per-store
    /// buffer (stable device address) refreshed in place - the write form a
    /// CUDA-graph replay needs. Numerically identical to `write`.
    pub fn write_stable(
        &self,
        slots: &[usize],
        k_new: &crate::tensor::Tensor,
        v_new: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<()> {
        let n = slots.len();
        let mut idx = Vec::with_capacity(n * self.feat);
        for &s in slots {
            for _ in 0..self.feat {
                idx.push(s as u32);
            }
        }
        let fresh = crate::tensor::Tensor::from_vec(idx, &[n, self.feat][..], &k_new.device())?;
        {
            let mut slot = self.write_idx.borrow_mut();
            let need_new = slot
                .as_ref()
                .map(|t| t.dims() != [n, self.feat])
                .unwrap_or(true);
            if need_new {
                *slot = Some(fresh.clone());
            } else {
                slot.as_ref().unwrap().slice_set(&fresh, 0, 0)?;
            }
        }
        let idx_ref = self.write_idx.borrow();
        let ids = idx_ref.as_ref().unwrap();
        self.k.scatter_set(ids, k_new, 0)?;
        self.v.scatter_set(ids, v_new, 0)?;
        Ok(())
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn feat(&self) -> usize {
        self.feat
    }
    /// The flat paged K/V store tensors `[num_slots, feat]` - for the capture-safe
    /// paged flash-decode / KV-write kernels that index them via device buffers.
    pub fn k(&self) -> &crate::tensor::Tensor {
        &self.k
    }
    pub fn v(&self) -> &crate::tensor::Tensor {
        &self.v
    }

    /// Scatter `n_tok` new tokens' K/V (each `[n_tok, feat]`) into `slots`.
    pub fn write(
        &self,
        slots: &[usize],
        k_new: &crate::tensor::Tensor,
        v_new: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<()> {
        let n = slots.len();
        debug_assert_eq!(k_new.dims(), &[n, self.feat]);
        // index tensor [n, feat] with row i == slots[i] (scatter_set wants
        // indexes shaped like source; the column dim is identity).
        let mut idx = Vec::with_capacity(n * self.feat);
        for &s in slots {
            for _ in 0..self.feat {
                idx.push(s as u32);
            }
        }
        let idx = crate::tensor::Tensor::from_vec(idx, &[n, self.feat][..], &k_new.device())?;
        self.k.scatter_set(&idx, k_new, 0)?;
        self.v.scatter_set(&idx, v_new, 0)?;
        Ok(())
    }

    /// Gather a sequence's first `len` tokens' K and V into contiguous
    /// `[len, feat]` tensors via its block table.
    pub fn gather_seq(
        &self,
        block_table: &[u32],
        len: usize,
    ) -> crate::tensor::Result<(crate::tensor::Tensor, crate::tensor::Tensor)> {
        let mut slot_ids = Vec::with_capacity(len);
        for p in 0..len {
            let blk = block_table[p / self.block_size] as usize;
            slot_ids.push((blk * self.block_size + p % self.block_size) as u32);
        }
        let ids = crate::tensor::Tensor::from_vec(slot_ids, &[len][..], &self.k.device())?;
        Ok((self.k.index_select(&ids, 0)?, self.v.index_select(&ids, 0)?))
    }

    /// Ragged batched gather: B sequences of DIFFERENT lengths `lens`, padded to
    /// `max_len`, in ONE index_select per K/V -> `[B, max_len, feat]`. Rows past a
    /// sequence's length map to slot 0 (garbage) and MUST be masked out by the
    /// caller's attention. This is the real-serving gather: continuous-batch
    /// sequences desync by a token during the prefill ramp, so contexts differ.
    pub fn gather_batch_padded(
        &self,
        block_tables: &[Vec<u32>],
        lens: &[usize],
        max_len: usize,
    ) -> crate::tensor::Result<(crate::tensor::Tensor, crate::tensor::Tensor)> {
        let b = block_tables.len();
        let mut slot_ids = Vec::with_capacity(b * max_len);
        for (bi, bt) in block_tables.iter().enumerate() {
            let len = lens[bi];
            for p in 0..max_len {
                if p < len {
                    let blk = bt[p / self.block_size] as usize;
                    slot_ids.push((blk * self.block_size + p % self.block_size) as u32);
                } else {
                    slot_ids.push(0); // padding row, masked in attention
                }
            }
        }
        let ids = crate::tensor::Tensor::from_vec(slot_ids, &[b * max_len][..], &self.k.device())?;
        let kg = self
            .k
            .index_select(&ids, 0)?
            .reshape(&[b, max_len, self.feat][..])?;
        let vg = self
            .v
            .index_select(&ids, 0)?
            .reshape(&[b, max_len, self.feat][..])?;
        Ok((kg, vg))
    }

    /// Like [`gather_batch_padded`] but the slot-index tensor lives in a PERSISTENT
    /// per-store buffer (stable device address), updated in place each step rather
    /// than freshly `from_vec`'d. This is the gather form a CUDA-graph replay needs:
    /// the captured `index_select` reads a fixed address whose contents we refresh
    /// (via `slice_set`) BEFORE each replay. Numerically identical to the non-stable
    /// form. The buffer is (re)allocated only when `b*max_len` changes (bucketed
    /// shapes keep it stable within a bucket).
    pub fn gather_batch_padded_stable(
        &self,
        block_tables: &[Vec<u32>],
        lens: &[usize],
        max_len: usize,
    ) -> crate::tensor::Result<(crate::tensor::Tensor, crate::tensor::Tensor)> {
        let b = block_tables.len();
        let n = b * max_len;
        let mut slot_ids = Vec::with_capacity(n);
        for (bi, bt) in block_tables.iter().enumerate() {
            for p in 0..max_len {
                if p < lens[bi] {
                    let blk = bt[p / self.block_size] as usize;
                    slot_ids.push((blk * self.block_size + p % self.block_size) as u32);
                } else {
                    slot_ids.push(0);
                }
            }
        }
        let fresh = crate::tensor::Tensor::from_vec(slot_ids, &[n][..], &self.k.device())?;
        // refresh the persistent buffer's contents at a stable address
        {
            let mut slot = self.gather_idx.borrow_mut();
            let need_new = slot.as_ref().map(|t| t.dims() != [n]).unwrap_or(true);
            if need_new {
                *slot = Some(fresh.clone());
            } else {
                slot.as_ref().unwrap().slice_set(&fresh, 0, 0)?;
            }
        }
        let idx_ref = self.gather_idx.borrow();
        let ids = idx_ref.as_ref().unwrap();
        let kg = self
            .k
            .index_select(ids, 0)?
            .reshape(&[b, max_len, self.feat][..])?;
        let vg = self
            .v
            .index_select(ids, 0)?
            .reshape(&[b, max_len, self.feat][..])?;
        Ok((kg, vg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_hashes_match_shared_prefix_and_diverge_after() {
        let bs = 4;
        let a = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 103];
        let b = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 200, 201, 202, 203]; // shares first 2 blocks
        let ha = block_hashes(&a, bs);
        let hb = block_hashes(&b, bs);
        assert_eq!(ha.len(), 3);
        assert_eq!(ha[0], hb[0], "block 0 identical content -> same hash");
        assert_eq!(ha[1], hb[1], "block 1 identical content -> same hash");
        assert_ne!(ha[2], hb[2], "block 2 diverges -> different hash");
        // A differing FIRST block must change ALL downstream hashes (prefix-aware).
        let c = vec![9u32, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 103];
        let hc = block_hashes(&c, bs);
        assert_ne!(ha[0], hc[0]);
        assert_ne!(
            ha[1], hc[1],
            "same block-1 tokens but different prefix -> different hash"
        );
    }

    #[test]
    fn block_hashes_only_full_blocks() {
        assert_eq!(block_hashes(&[1, 2, 3], 4).len(), 0); // partial block not hashed
        assert_eq!(block_hashes(&[1, 2, 3, 4, 5], 4).len(), 1); // 1 full + partial
    }

    #[test]
    fn prefix_cache_acquire_matches_contiguous_prefix() {
        let mut pc = PrefixCache::new(4);
        let toks = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let hashes = block_hashes(&toks, 4);
        // cache blocks 0 and 1 (physical 50, 51); leave block 2 uncached
        assert!(pc.try_insert(hashes[0], 50));
        assert!(pc.try_insert(hashes[1], 51));
        let (blocks, held) = pc.acquire_prefix(&toks);
        assert_eq!(blocks, vec![50, 51], "longest contiguous cached prefix");
        assert_eq!(held.len(), 2);
        // a hole stops the match: cache block 2 but NOT block 1 -> only block 0 matches
        let mut pc2 = PrefixCache::new(4);
        assert!(pc2.try_insert(hashes[0], 50));
        assert!(pc2.try_insert(hashes[2], 52));
        let (blocks2, _) = pc2.acquire_prefix(&toks);
        assert_eq!(
            blocks2,
            vec![50],
            "non-contiguous cache -> prefix stops at the hole"
        );
    }

    #[test]
    fn prefix_cache_persists_until_evicted() {
        let mut pc = PrefixCache::new(4);
        let h = block_hashes(&[1, 2, 3, 4], 4)[0];
        let toks = vec![1u32, 2, 3, 4];
        assert!(pc.try_insert(h, 70)); // producer, refs=1
        assert!(!pc.try_insert(h, 99)); // race: already present
        let _ = pc.acquire_prefix(&toks); // refs=2
        pc.release(h); // refs=1 (still in use)
        assert_eq!(pc.evictable(), 0);
        pc.release(h); // refs=0 - but STAYS cached (persistent!)
        assert!(!pc.is_empty(), "idle prefix stays resident for later reuse");
        assert_eq!(pc.evictable(), 1);
        // THE FIX: a later (sequential) request reuses it after the producer finished.
        assert_eq!(
            pc.acquire_prefix(&toks).0,
            vec![70],
            "reused after producer released"
        );
        pc.release(h); // refs=0 again
                       // Reclaimed only under memory pressure, LRU-first.
        assert_eq!(pc.evict_one(), Some(70));
        assert!(pc.is_empty());
        assert_eq!(pc.evict_one(), None);
    }

    #[test]
    fn allocate_with_prefix_shares_blocks_and_frees_without_leak() {
        let bs = 4;
        let mut a = PagedKvAllocator::new_with_prefix(16, bs);
        let base = a.free_blocks();
        // seq 1: cold cache -> allocates all 3 prompt blocks, then publishes them.
        let p1 = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        assert_eq!(
            a.allocate_with_prefix(1, &p1).unwrap(),
            0,
            "cold cache caches nothing"
        );
        assert_eq!(a.free_blocks(), base - 3);
        a.commit_prefix(1, &p1);
        // seq 2: shares the first 2 blocks (8 tokens), diverges in block 3.
        let p2 = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97, 96];
        assert_eq!(
            a.allocate_with_prefix(2, &p2).unwrap(),
            8,
            "first 2 cached blocks -> start prefill at 8"
        );
        assert_eq!(
            a.free_blocks(),
            base - 3 - 1,
            "only 1 fresh block for seq2's divergent block"
        );
        a.commit_prefix(2, &p2);
        // 4 distinct physical blocks were allocated (seq1's 3 + seq2's 1 fresh).
        a.free(1).unwrap();
        a.free(2).unwrap();
        // PERSISTENCE: all 4 cached prompt blocks STAY resident (idle) for reuse by a
        // future request - only seq-owned blocks return. None freed on finish.
        assert_eq!(
            a.free_blocks(),
            base - 4,
            "cached prefix blocks persist (not freed on finish)"
        );
        assert_eq!(
            a.prefix.as_ref().unwrap().evictable(),
            4,
            "4 idle cached blocks reclaimable"
        );
        // A request that needs the whole pool evicts the idle cached blocks -> no leak.
        assert!(
            a.can_append(0, base * 4),
            "evictable cached blocks count toward capacity"
        );
        a.allocate(9, base * 4).unwrap(); // base*4 tokens = `base` blocks = everything
        assert_eq!(a.free_blocks(), 0);
        a.free(9).unwrap();
        assert_eq!(
            a.free_blocks(),
            base,
            "eviction reclaimed the cached blocks - no leak"
        );
    }

    #[test]
    fn allocate_with_prefix_never_caches_the_whole_prompt() {
        // Regression: a block-aligned, fully-cached prompt must still leave a >=1-token
        // suffix to compute (else tsfx==0 -> empty-tensor forward kills the worker).
        let mut a = PagedKvAllocator::new_with_prefix(16, 4);
        let p = vec![1u32, 2, 3, 4, 5, 6, 7, 8]; // 8 tokens = 2 full blocks, block-aligned
        a.allocate_with_prefix(1, &p).unwrap();
        a.commit_prefix(1, &p);
        a.free(1).unwrap();
        let cached = a.allocate_with_prefix(2, &p).unwrap(); // both blocks cached
        assert!(
            cached < p.len(),
            "must not cache the entire prompt (need >=1 token for logits)"
        );
        assert_eq!(
            cached, 4,
            "drop the last cached block -> 1 cached, 1 recomputed"
        );
    }

    #[test]
    fn allocate_with_prefix_is_plain_allocate_when_disabled() {
        let mut a = PagedKvAllocator::new(16, 4); // prefix disabled
        let p = vec![1u32, 2, 3, 4, 5, 6, 7];
        assert_eq!(
            a.allocate_with_prefix(1, &p).unwrap(),
            0,
            "no caching -> 0 cached tokens"
        );
        assert_eq!(
            a.free_blocks(),
            14,
            "2 blocks for 7 tokens, same as plain allocate"
        );
        a.commit_prefix(1, &p); // no-op when disabled
        a.free(1).unwrap();
        assert_eq!(a.free_blocks(), 16);
    }

    #[test]
    fn store_write_gather_roundtrip() {
        use crate::tensor::{DType, Device, Tensor};
        // 4 blocks x 2 tokens, n_kv=1, hd=3 -> feat=3.
        let store = PagedKvStore::new(4, 2, 1, 3, DType::F32, &Device::Cpu).unwrap();
        let mut a = PagedKvAllocator::new(4, 2);
        a.allocate(7, 0).unwrap();
        // append 3 tokens -> slots from the allocator (may span 2 blocks)
        let slots = a.append(7, 3).unwrap();
        // distinct per-token K/V values
        let k = Tensor::from_vec(
            vec![1f32, 1., 1., 2., 2., 2., 3., 3., 3.],
            &[3, 3][..],
            &Device::Cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            vec![10f32, 10., 10., 20., 20., 20., 30., 30., 30.],
            &[3, 3][..],
            &Device::Cpu,
        )
        .unwrap();
        store.write(&slots, &k, &v).unwrap();
        // gather back via the block table -> must equal what we wrote, in order
        let bt = a.block_table(7).unwrap();
        let (kg, vg) = store.gather_seq(bt, 3).unwrap();
        assert_eq!(
            kg.to_vec2::<f32>().unwrap(),
            vec![vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]]
        );
        assert_eq!(
            vg.to_vec2::<f32>().unwrap(),
            vec![
                vec![10., 10., 10.],
                vec![20., 20., 20.],
                vec![30., 30., 30.]
            ]
        );
    }

    #[test]
    fn prompt_alloc_rounds_up_to_blocks() {
        let mut a = PagedKvAllocator::new(16, 8);
        a.allocate(1, 20).unwrap(); // 20 tokens -> ceil(20/8)=3 blocks
        assert_eq!(a.block_table(1).unwrap().len(), 3);
        assert_eq!(a.used_blocks(), 3);
        assert_eq!(a.free_blocks(), 13);
        assert_eq!(a.seq_len(1), Some(20));
    }

    #[test]
    fn append_grows_only_on_block_boundary() {
        let mut a = PagedKvAllocator::new(16, 4);
        a.allocate(1, 4).unwrap(); // exactly 1 block, full
        assert_eq!(a.block_table(1).unwrap().len(), 1);
        let slots = a.append(1, 1).unwrap(); // token 4 -> new block
        assert_eq!(a.block_table(1).unwrap().len(), 2);
        assert_eq!(slots.len(), 1);
        // token 4 lives in logical block 1, offset 0 -> phys*4 + 0
        let phys = a.block_table(1).unwrap()[1] as usize;
        assert_eq!(slots[0], phys * 4);
        // tokens 5,6,7 fill block 1, no new block
        a.append(1, 3).unwrap();
        assert_eq!(a.block_table(1).unwrap().len(), 2);
    }

    #[test]
    fn slot_mapping_is_contiguous_within_a_block() {
        let mut a = PagedKvAllocator::new(8, 4);
        a.allocate(1, 0).unwrap();
        let slots = a.append(1, 4).unwrap(); // fills block 0
        let phys = a.block_table(1).unwrap()[0] as usize;
        assert_eq!(
            slots,
            vec![phys * 4, phys * 4 + 1, phys * 4 + 2, phys * 4 + 3]
        );
    }

    #[test]
    fn out_of_blocks_is_reported_not_panicked() {
        let mut a = PagedKvAllocator::new(2, 4); // 2 blocks = 8 tokens
        a.allocate(1, 8).unwrap();
        assert_eq!(a.free_blocks(), 0);
        assert!(!a.can_append(8, 1));
        assert_eq!(
            a.append(1, 1),
            Err(AllocErr::OutOfBlocks { needed: 1, free: 0 })
        );
        // a brand-new sequence also can't be admitted
        assert_eq!(
            a.allocate(2, 1),
            Err(AllocErr::OutOfBlocks { needed: 1, free: 0 })
        );
    }

    #[test]
    fn free_returns_blocks_and_enables_reuse() {
        let mut a = PagedKvAllocator::new(4, 4);
        a.allocate(1, 16).unwrap(); // all 4 blocks
        assert_eq!(a.free_blocks(), 0);
        assert_eq!(a.free(1).unwrap(), 4);
        assert_eq!(a.free_blocks(), 4);
        // a new sequence reuses the freed blocks
        a.allocate(2, 16).unwrap();
        assert_eq!(a.used_blocks(), 4);
    }

    #[test]
    fn trim_frees_unneeded_blocks() {
        let mut a = PagedKvAllocator::new(16, 4);
        a.allocate(1, 16).unwrap(); // 4 blocks
        let freed = a.trim(1, 5).unwrap(); // 5 tokens -> ceil(5/4)=2 blocks -> free 2
        assert_eq!(freed, 2);
        assert_eq!(a.block_table(1).unwrap().len(), 2);
        assert_eq!(a.seq_len(1), Some(5));
        assert_eq!(a.free_blocks(), 14);
    }

    #[test]
    fn many_sequences_share_the_pool_without_fragmentation() {
        let mut a = PagedKvAllocator::new(10, 4); // 40 tokens of capacity
        for s in 0..10 {
            a.allocate(s, 4).unwrap();
        } // 10 seqs x 1 block
        assert_eq!(a.free_blocks(), 0);
        a.free(3).unwrap();
        a.free(7).unwrap();
        assert_eq!(a.free_blocks(), 2);
        // two freed (non-adjacent) blocks fully satisfy a new 2-block sequence  - 
        // no contiguity needed (the whole point of paging).
        a.allocate(99, 5).unwrap();
        assert_eq!(a.free_blocks(), 0);
    }

    #[test]
    fn unknown_seq_errors() {
        let mut a = PagedKvAllocator::new(4, 4);
        assert_eq!(a.append(42, 1), Err(AllocErr::UnknownSeq(42)));
        assert_eq!(a.free(42), Err(AllocErr::UnknownSeq(42)));
    }
}
