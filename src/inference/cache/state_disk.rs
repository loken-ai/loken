//! A disk store of decode-state snapshots, found by the longest prefix of a prompt they cover.
//!
//! A snapshot is a small manifest plus references to content-addressed blocks. The manifest is an
//! opaque blob the caller encodes: the part of the state that belongs to its exact position. A
//! block holds what a block of `block_tokens` tokens added to the state and is named by the chain
//! hash of the tokens up to its end, so every snapshot of one conversation, and every snapshot of
//! conversations that share a prefix, names the same block files and stores each once.
//!
//! The chain: each block of tokens is hashed with its parent's hash and the scope (the identity of
//! the model and numerics); a manifest's key hashes the last full block's chain value with the
//! length and the tokens after that block. A lookup hashes the prompt once and probes one key per
//! distinct stored length, longest first, so it finds the longest stored prefix without the store
//! ever holding a token.
//!
//! Privacy: file names and the index hold lengths, hashes, byte counts, use counts and times; the
//! blobs are whatever the caller encodes, KV tensors and not text. The store is off unless the
//! operator gives it a directory, and the files are prompt-derived state.
//!
//! Durability: every file is written beside its name, synced, then renamed in, so a reader sees a
//! whole file or none. A manifest is written only after every block it names is on disk, and a
//! block a put is about to reference is counted before it is written, so eviction cannot take it
//! away between the two. At open, a stray or foreign file is removed, a manifest whose block is
//! missing is dropped, and a block no manifest names is removed; an unreadable index is dropped
//! with a log line and rebuilt from the names.
//!
//! Eviction: when the manifests plus the blocks they reference pass the budget the least recently
//! used manifests go first, and a block goes with its last reference. In each family of manifests
//! - those sharing a first block - the one used most, the longest among equals, is pinned: it goes
//! last, so a long prefix that conversations keep coming back to outlives the one-off tails around
//! it. The manifest just written is never the victim.

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The hash a stored snapshot or block is found by.
pub type Key = [u8; 16];

const SUFFIX: &str = "snap";
const BLOCK_SUFFIX: &str = "blk";
const BLOCK_DIR: &str = "blocks";
const INDEX_FILE: &str = "index.json";
const GEOMETRY_FILE: &str = "geometry.json";
const TMP_PREFIX: &str = ".tmp-";
const MANIFEST_MAGIC: &[u8; 8] = b"LKSMAN01";

/// One stored snapshot: its manifest file and the blocks it references.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub key: Key,
    /// Tokens the snapshot's state covers.
    pub len: usize,
    family: u64,
    /// Size of the manifest file.
    pub bytes: u64,
    last_used: u64,
    hits: u64,
    /// The blocks the snapshot references, in token order.
    pub blocks: Vec<Key>,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct IndexFile {
    /// Key (hex) to `[last_used, hits]`.
    uses: HashMap<String, (u64, u64)>,
}

struct Inner {
    entries: HashMap<Key, Entry>,
    /// Manifests (and puts in flight) referencing each block, and each block's size on disk.
    refs: HashMap<Key, usize>,
    block_bytes: HashMap<Key, u64>,
}

pub struct StateStore {
    dir: PathBuf,
    /// The files of a store kept in host memory, by path, none of them on disk; `None` for a
    /// store in its directory.
    memory: Option<Mutex<HashMap<PathBuf, Arc<Vec<u8>>>>>,
    /// Bytes the operator allows; 0 derives the limit from the filesystem's free space.
    budget: u64,
    /// Tokens per block the operator fixed for every scope; 0 derives one per scope.
    block_tokens: usize,
    /// The block size settled for each scope, kept in the directory so a restart finds the chain
    /// it wrote.
    geometry: Mutex<HashMap<[u8; 32], usize>>,
    inner: Mutex<Inner>,
    tmp_seq: AtomicU64,
    tuning: super::state_tune::Tuning,
    /// Bytes of writes waiting, and the turn they take on the disk one at a time.
    pending: Arc<AtomicU64>,
    turn: Arc<Mutex<()>>,
}

/// A write admitted to the queue. Dropping it releases its bytes.
pub struct WriteTicket {
    pending: Arc<AtomicU64>,
    turn: Arc<Mutex<()>>,
    bytes: u64,
}

impl WriteTicket {
    /// Waits for the disk's turn, then runs `write`.
    pub fn run<T>(&self, write: impl FnOnce() -> T) -> T {
        let _turn = self.turn.lock().unwrap_or_else(|e| e.into_inner());
        write()
    }
}

impl Drop for WriteTicket {
    fn drop(&mut self) {
        self.pending.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// The identity of a model's numerics from the parts that define them, in order: a part that
/// changes (a file digest, a numerics setting) gives another identity.
pub fn identity(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"loken state identity");
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p);
    }
    h.finalize().into()
}

/// The hashes of a token sequence under one scope: a chain over its full blocks, from which the
/// key at any length up to its own is read.
pub struct PrefixKeys<'a> {
    tokens: &'a [u32],
    block_tokens: usize,
    root: Key,
    chain: Vec<Key>,
}

fn short(digest: impl AsRef<[u8]>) -> Key {
    digest.as_ref()[..16].try_into().unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

fn unhex(s: &str) -> Option<Key> {
    if s.len() != 32 {
        return None;
    }
    let mut key = [0u8; 16];
    for (i, b) in key.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(key)
}

fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

impl<'a> PrefixKeys<'a> {
    fn new(scope: &[u8; 32], block_tokens: usize, tokens: &'a [u32]) -> Self {
        let root = short(
            Sha256::new()
                .chain_update(scope)
                .chain_update(b"root")
                .finalize(),
        );
        let mut prev = root;
        let chain = tokens
            .chunks_exact(block_tokens)
            .map(|block| {
                let mut h = Sha256::new().chain_update(prev);
                for t in block {
                    h.update(t.to_le_bytes());
                }
                prev = short(h.finalize());
                prev
            })
            .collect();
        Self {
            tokens,
            block_tokens,
            root,
            chain,
        }
    }

    /// The key of the first `len` tokens and the family it belongs to, `None` past the end.
    fn at(&self, len: usize) -> Option<(Key, u64)> {
        if len > self.tokens.len() || len == 0 {
            return None;
        }
        let full = len / self.block_tokens;
        let parent = if full == 0 {
            self.root
        } else {
            self.chain[full - 1]
        };
        let mut h = Sha256::new()
            .chain_update(parent)
            .chain_update((len as u64).to_le_bytes());
        for t in &self.tokens[full * self.block_tokens..len] {
            h.update(t.to_le_bytes());
        }
        let key = short(h.finalize());
        let family_of = self.chain.first().copied().unwrap_or(key);
        Some((key, u64::from_le_bytes(family_of[..8].try_into().unwrap())))
    }

    /// The hash of the block ending at token `(b + 1) * block_tokens`, `None` past the sequence.
    pub fn block_hash(&self, b: usize) -> Option<Key> {
        self.chain.get(b).copied()
    }

    /// Tokens the sequence has.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

fn file_name(len: usize, family: u64, key: &Key) -> String {
    format!("{len:012}.{family:016x}.{}.{SUFFIX}", hex(key))
}

fn parse_name(name: &str) -> Option<(usize, u64, Key)> {
    let stem = name.strip_suffix(&format!(".{SUFFIX}"))?;
    let mut parts = stem.split('.');
    let len = parts.next()?.parse().ok()?;
    let family = u64::from_str_radix(parts.next()?, 16).ok()?;
    let key = unhex(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some((len, family, key))
}

/// The blocks a manifest file names, from its header, and where the caller's payload starts.
fn read_header(bytes: &[u8]) -> Option<(Vec<Key>, usize)> {
    if bytes.len() < 12 || &bytes[..8] != MANIFEST_MAGIC {
        return None;
    }
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let end = n.checked_mul(16)?.checked_add(12)?;
    let hashes = bytes.get(12..end)?;
    Some((
        hashes
            .chunks_exact(16)
            .map(|c| c.try_into().unwrap())
            .collect(),
        end,
    ))
}

impl StateStore {
    /// Open (creating) the store in `dir`. `budget` is in bytes, 0 to derive it from the free
    /// space of the filesystem; `block_tokens` the block size of the hash chain for every scope,
    /// 0 to settle one per scope (see [`Self::settle`]). A block size must not change for a store
    /// in use.
    pub fn open(dir: PathBuf, budget: u64, block_tokens: usize) -> Result<Self> {
        use std::io::Read;
        std::fs::create_dir_all(dir.join(BLOCK_DIR)).with_context(|| dir.display().to_string())?;
        let index: IndexFile = match std::fs::read(dir.join(INDEX_FILE)) {
            Ok(b) => serde_json::from_slice(&b).unwrap_or_else(|e| {
                tracing::warn!("state cache: index unreadable ({e}); rebuilt from the file names");
                IndexFile::default()
            }),
            Err(_) => IndexFile::default(),
        };
        let mut block_bytes: HashMap<Key, u64> = HashMap::new();
        for item in std::fs::read_dir(dir.join(BLOCK_DIR))? {
            let item = item?;
            let name = item.file_name().to_string_lossy().into_owned();
            let hash = name
                .strip_suffix(&format!(".{BLOCK_SUFFIX}"))
                .and_then(unhex);
            match hash {
                Some(h) => {
                    block_bytes.insert(h, item.metadata()?.len());
                }
                None => {
                    let _ = std::fs::remove_file(item.path());
                }
            }
        }
        let mut entries = HashMap::new();
        let mut refs: HashMap<Key, usize> = HashMap::new();
        for item in std::fs::read_dir(&dir)? {
            let item = item?;
            let name = item.file_name().to_string_lossy().into_owned();
            if name == INDEX_FILE || name == BLOCK_DIR || name == GEOMETRY_FILE {
                continue;
            }
            if name.starts_with(TMP_PREFIX) {
                let _ = std::fs::remove_file(item.path());
                continue;
            }
            let Some((len, family, key)) = parse_name(&name) else {
                tracing::warn!("state cache: {name} is not a snapshot of ours; removed");
                let _ = std::fs::remove_file(item.path());
                continue;
            };
            let mut head = Vec::new();
            let blocks = std::fs::File::open(item.path())
                .and_then(|f| {
                    // The header is 12 bytes and the hashes after it; read as many as it says.
                    let mut f = f.take(12);
                    f.read_to_end(&mut head)?;
                    let n = head
                        .get(8..12)
                        .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
                    let mut f = f.into_inner().take(n as u64 * 16);
                    f.read_to_end(&mut head)
                })
                .ok()
                .and_then(|_| read_header(&head))
                .map(|(b, _)| b);
            let Some(blocks) = blocks.filter(|b| b.iter().all(|h| block_bytes.contains_key(h)))
            else {
                tracing::warn!("state cache: {name} is damaged or names a missing block; removed");
                let _ = std::fs::remove_file(item.path());
                continue;
            };
            let meta = item.metadata()?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            let (last_used, hits) = index.uses.get(&hex(&key)).copied().unwrap_or((mtime, 0));
            for b in &blocks {
                *refs.entry(*b).or_default() += 1;
            }
            entries.insert(
                key,
                Entry {
                    key,
                    len,
                    family,
                    bytes: meta.len(),
                    last_used,
                    hits,
                    blocks,
                },
            );
        }
        // A block no manifest names was an interrupted write.
        block_bytes.retain(|h, _| {
            let keep = refs.contains_key(h);
            if !keep {
                let _ = std::fs::remove_file(
                    dir.join(BLOCK_DIR)
                        .join(format!("{}.{BLOCK_SUFFIX}", hex(h))),
                );
            }
            keep
        });
        let geometry: HashMap<[u8; 32], usize> = std::fs::read(dir.join(GEOMETRY_FILE))
            .ok()
            .and_then(|b| serde_json::from_slice::<HashMap<String, usize>>(&b).ok())
            .map(|m| {
                m.into_iter()
                    .filter_map(|(k, v)| Some((unhex32(&k)?, v)))
                    .filter(|(_, v)| *v > 0)
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            dir,
            memory: None,
            budget,
            block_tokens,
            geometry: Mutex::new(geometry),
            inner: Mutex::new(Inner {
                entries,
                refs,
                block_bytes,
            }),
            tmp_seq: AtomicU64::new(0),
            tuning: Default::default(),
            pending: Arc::new(AtomicU64::new(0)),
            turn: Arc::new(Mutex::new(())),
        })
    }

    /// A store in host memory: what a store on disk keeps, held only while the process lives and
    /// never written. `budget` is in bytes, 0 to derive it from the host memory a component may
    /// take; `block_tokens` as [`Self::open`] takes it.
    pub fn in_memory(budget: u64, block_tokens: usize) -> Self {
        Self {
            dir: PathBuf::from("state-cache-in-memory"),
            memory: Some(Mutex::new(HashMap::new())),
            budget,
            block_tokens,
            geometry: Mutex::new(HashMap::new()),
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                refs: HashMap::new(),
                block_bytes: HashMap::new(),
            }),
            tmp_seq: AtomicU64::new(0),
            tuning: Default::default(),
            pending: Arc::new(AtomicU64::new(0)),
            turn: Arc::new(Mutex::new(())),
        }
    }

    /// Whether the store is held in host memory rather than in a directory.
    pub fn is_in_memory(&self) -> bool {
        self.memory.is_some()
    }

    /// The bytes of the file at `path`.
    fn load(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        match &self.memory {
            Some(files) => files
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(path)
                .map(|b| b.as_ref().clone())
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound)),
            None => std::fs::read(path),
        }
    }

    /// Removes the file at `path`, if there is one.
    fn unlink(&self, path: &Path) {
        match &self.memory {
            Some(files) => {
                files.lock().unwrap_or_else(|e| e.into_inner()).remove(path);
            }
            None => {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    /// The block size the operator fixed, 0 when each scope settles its own.
    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    /// The block size of the chain under `scope`: the operator's when one was fixed, else the one
    /// settled for it. Panics when neither exists: a cache settles its scope before it hashes.
    pub fn block_tokens_of(&self, scope: &[u8; 32]) -> usize {
        if self.block_tokens > 0 {
            return self.block_tokens;
        }
        *self
            .geometry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(scope)
            .expect("state cache: the scope's block size was not settled")
    }

    /// Settles the block size of `scope`: the operator's when fixed, the one kept from an earlier
    /// run when there is one, else `derive` computes it from the store's measurements and it is
    /// kept.
    pub fn settle(&self, scope: &[u8; 32], derive: impl FnOnce(&Self) -> usize) -> usize {
        if self.block_tokens > 0 {
            return self.block_tokens;
        }
        if let Some(&b) = self
            .geometry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(scope)
        {
            return b;
        }
        let b = derive(self).max(1);
        let json = {
            let mut g = self.geometry.lock().unwrap_or_else(|e| e.into_inner());
            g.insert(*scope, b);
            serde_json::to_vec(
                &g.iter()
                    .map(|(k, v)| (hex(k), *v))
                    .collect::<HashMap<String, usize>>(),
            )
        };
        if let Ok(json) = json {
            let _ = self.write_file(&self.dir.join(GEOMETRY_FILE), &json);
        }
        b
    }

    /// The directory the store lives in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// What the engine and the disk have been measured to do.
    pub fn tuning(&self) -> &super::state_tune::Tuning {
        &self.tuning
    }

    /// Bytes the store may hold with `held` on disk now: the operator's budget, else what is held
    /// and what the filesystem still offers less room for one more state of the largest size
    /// written. `None` when the filesystem cannot say.
    fn limit(&self, held: u64) -> Option<u64> {
        if self.budget > 0 {
            return Some(self.budget);
        }
        let free = match self.memory {
            Some(_) => crate::inference::place::layer_executor::host_spill_budget_bytes(),
            None => super::state_tune::free_space(&self.dir)?,
        };
        Some(super::state_tune::derived_budget(
            held,
            free,
            self.tuning.get().largest_state,
        ))
    }

    /// Bytes the store could hold now: its limit, unbounded when the filesystem cannot say.
    pub fn room(&self) -> u64 {
        self.limit(self.held_bytes()).unwrap_or(u64::MAX)
    }

    /// What restoring `entry` reads: its manifest and its blocks.
    pub fn entry_bytes(&self, entry: &Entry) -> u64 {
        let g = self.lock();
        entry.bytes
            + entry
                .blocks
                .iter()
                .map(|b| g.block_bytes.get(b).copied().unwrap_or(0))
                .sum::<u64>()
    }

    /// The bytes of the manifests and blocks on disk.
    fn held_bytes(&self) -> u64 {
        self.stats().1
    }

    /// Admits a write of `bytes` to the queue when the disk drains what waits within
    /// `window_s`, the compute time of the prompt the state belongs to.
    pub fn admit(&self, bytes: u64, window_s: Option<f64>) -> Option<WriteTicket> {
        let pending = self.pending.load(Ordering::Relaxed);
        let w = self.tuning.get().write.per_sec();
        if !super::state_tune::write_may_queue(pending, bytes, w, window_s) {
            return None;
        }
        self.pending.fetch_add(bytes, Ordering::Relaxed);
        Some(WriteTicket {
            pending: self.pending.clone(),
            turn: self.turn.clone(),
            bytes,
        })
    }

    /// The hashes of `tokens` under `scope`.
    pub fn keys<'a>(&self, scope: &[u8; 32], tokens: &'a [u32]) -> PrefixKeys<'a> {
        PrefixKeys::new(scope, self.block_tokens_of(scope), tokens)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn path_of(&self, e: &Entry) -> PathBuf {
        self.dir.join(file_name(e.len, e.family, &e.key))
    }

    fn block_path(&self, h: &Key) -> PathBuf {
        self.dir
            .join(BLOCK_DIR)
            .join(format!("{}.{BLOCK_SUFFIX}", hex(h)))
    }

    /// The stored snapshots that are prefixes of the sequence `keys` hashes and no longer than
    /// `max_len`, longest first.
    pub fn candidates(&self, keys: &PrefixKeys, max_len: usize) -> Vec<Entry> {
        let g = self.lock();
        let mut lens: Vec<usize> = g
            .entries
            .values()
            .map(|e| e.len)
            .filter(|&l| l <= max_len)
            .collect();
        lens.sort_unstable_by(|a, b| b.cmp(a));
        lens.dedup();
        lens.into_iter()
            .filter_map(|l| keys.at(l))
            .filter_map(|(key, _)| g.entries.get(&key).cloned())
            .collect()
    }

    /// The manifest payload of a stored snapshot, counted as a use. A file that cannot be read is
    /// dropped.
    pub fn read(&self, entry: &Entry) -> Result<Vec<u8>> {
        let path = self.path_of(entry);
        let bytes = self.load(&path).map_err(anyhow::Error::from).and_then(|b| {
            let (_, start) = read_header(&b).ok_or_else(|| anyhow!("manifest header"))?;
            Ok(b[start..].to_vec())
        });
        match bytes {
            Ok(bytes) => {
                {
                    let mut g = self.lock();
                    if let Some(e) = g.entries.get_mut(&entry.key) {
                        e.last_used = now_nanos();
                        e.hits += 1;
                    }
                }
                self.save_index();
                Ok(bytes)
            }
            Err(e) => {
                self.forget(entry);
                Err(anyhow!("{}: {e}", path.display()))
            }
        }
    }

    /// The bytes of a block a manifest references. A block that cannot be read is an error the
    /// caller answers by forgetting the snapshot that needs it.
    pub fn read_block(&self, hash: &Key) -> Result<Vec<u8>> {
        let path = self.block_path(hash);
        self.load(&path)
            .map_err(|e| anyhow!("{}: {e}", path.display()))
    }

    /// Remove a snapshot that proved unusable, and the blocks only it referenced.
    pub fn forget(&self, entry: &Entry) {
        self.unlink(&self.path_of(entry));
        self.drop_entry(&entry.key);
        self.save_index();
    }

    /// Remove an entry from the index and release its blocks, deleting those no one else names.
    fn drop_entry(&self, key: &Key) {
        let freed: Vec<Key> = {
            let mut g = self.lock();
            let Some(e) = g.entries.remove(key) else {
                return;
            };
            let mut freed = Vec::new();
            for b in e.blocks {
                if let Some(n) = g.refs.get_mut(&b) {
                    *n -= 1;
                    if *n == 0 {
                        g.refs.remove(&b);
                        g.block_bytes.remove(&b);
                        freed.push(b);
                    }
                }
            }
            freed
        };
        for b in freed {
            self.unlink(&self.block_path(&b));
        }
    }

    /// The first of the first `nblocks` blocks of the sequence `keys` hashes that is not on disk,
    /// `nblocks` when all are. A caller that needs only the missing blocks' bytes starts there.
    pub fn first_missing_block(&self, keys: &PrefixKeys, nblocks: usize) -> usize {
        let g = self.lock();
        (0..nblocks)
            .find(|&b| {
                keys.block_hash(b)
                    .is_none_or(|h| !g.block_bytes.contains_key(&h))
            })
            .unwrap_or(nblocks)
    }

    /// Store the state after the first `len` tokens of the sequence `keys` hashes: `manifest` is
    /// the caller's payload and the state also references the first `nblocks` blocks of the
    /// sequence, whose bytes `block(b)` supplies for those not on disk already. Does nothing when
    /// the state is stored already, or when its manifest alone exceeds the budget.
    pub fn put(
        &self,
        keys: &PrefixKeys,
        len: usize,
        manifest: &[u8],
        nblocks: usize,
        mut block: impl FnMut(usize) -> Vec<u8>,
    ) -> Result<()> {
        let Some((key, family)) = keys.at(len) else {
            return Err(anyhow!("state cache: length {len} is outside the sequence"));
        };
        let hashes: Vec<Key> = (0..nblocks)
            .map(|b| keys.block_hash(b))
            .collect::<Option<_>>()
            .filter(|_| nblocks * keys.block_tokens <= len)
            .ok_or_else(|| anyhow!("state cache: {nblocks} blocks are outside {len} tokens"))?;
        let started = std::time::Instant::now();
        let mut written = 0u64;
        if let Some(limit) = self.limit(self.held_bytes()) {
            if manifest.len() as u64 > limit {
                tracing::info!(
                    "state cache: a {} MiB manifest exceeds the {} MiB budget; not stored",
                    manifest.len() >> 20,
                    limit >> 20
                );
                return Ok(());
            }
        }
        // Count the references first, so the blocks cannot be evicted between their write and the
        // manifest that names them.
        {
            let mut g = self.lock();
            if let Some(e) = g.entries.get_mut(&key) {
                e.last_used = now_nanos();
                drop(g);
                self.save_index();
                return Ok(());
            }
            for h in &hashes {
                *g.refs.entry(*h).or_default() += 1;
            }
        }
        let release = |this: &Self| {
            let mut freed = Vec::new();
            {
                let mut g = this.lock();
                for h in &hashes {
                    if let Some(n) = g.refs.get_mut(h) {
                        *n -= 1;
                        if *n == 0 {
                            g.refs.remove(h);
                            g.block_bytes.remove(h);
                            freed.push(*h);
                        }
                    }
                }
            }
            for h in freed {
                this.unlink(&this.block_path(&h));
            }
        };
        for (b, h) in hashes.iter().enumerate() {
            if self.lock().block_bytes.contains_key(h) {
                continue;
            }
            let bytes = block(b);
            if let Err(e) = self.write_file(&self.block_path(h), &bytes) {
                release(self);
                return Err(e);
            }
            written += bytes.len() as u64;
            self.lock().block_bytes.insert(*h, bytes.len() as u64);
        }
        let mut file = Vec::with_capacity(12 + hashes.len() * 16 + manifest.len());
        file.extend_from_slice(MANIFEST_MAGIC);
        file.extend_from_slice(&(hashes.len() as u32).to_le_bytes());
        for h in &hashes {
            file.extend_from_slice(h);
        }
        file.extend_from_slice(manifest);
        let entry = Entry {
            key,
            len,
            family,
            bytes: file.len() as u64,
            last_used: now_nanos(),
            hits: 0,
            blocks: hashes.clone(),
        };
        if let Err(e) = self.write_file(&self.path_of(&entry), &file) {
            release(self);
            return Err(e);
        }
        written += file.len() as u64;
        let whole = file.len() as u64 + {
            let g = self.lock();
            hashes
                .iter()
                .map(|h| g.block_bytes.get(h).copied().unwrap_or(0))
                .sum::<u64>()
        };
        self.lock().entries.insert(key, entry);
        self.tuning.stored(whole, manifest.len() as u64);
        self.tuning
            .write_rate(written, started.elapsed().as_secs_f64());
        self.evict(&key);
        self.save_index();
        Ok(())
    }

    /// `bytes` in a new file at `path`: written beside it, synced, renamed in.
    fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        if let Some(files) = &self.memory {
            files
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(path.to_path_buf(), Arc::new(bytes.to_vec()));
            return Ok(());
        }
        let tmp = self.dir.join(format!(
            "{TMP_PREFIX}{}-{}",
            std::process::id(),
            self.tmp_seq.fetch_add(1, Ordering::Relaxed)
        ));
        write_whole(&tmp, bytes).with_context(|| tmp.display().to_string())?;
        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(anyhow!("{}: {e}", path.display()));
        }
        if let Some(parent) = path.parent() {
            if let Ok(d) = std::fs::File::open(parent) {
                let _ = d.sync_all();
            }
        }
        Ok(())
    }

    /// Drop manifests, least recently used first and pinned ones last, until the manifests and the
    /// blocks they reference fit the budget. `keep` is never dropped.
    fn evict(&self, keep: &Key) {
        let mut victims = Vec::new();
        let limit;
        {
            let g = self.lock();
            let mut total: u64 = g.entries.values().map(|e| e.bytes).sum::<u64>()
                + g.block_bytes
                    .iter()
                    .filter(|(h, _)| g.refs.contains_key(*h))
                    .map(|(_, n)| *n)
                    .sum::<u64>();
            limit = match self.limit(total) {
                Some(l) => l,
                None => return,
            };
            if total <= limit {
                return;
            }
            let mut pinned: HashMap<u64, Key> = HashMap::new();
            for e in g.entries.values().filter(|e| e.hits > 0) {
                let best = pinned.entry(e.family).or_insert(e.key);
                let cur = &g.entries[best];
                if (e.hits, e.len) > (cur.hits, cur.len) {
                    *best = e.key;
                }
            }
            let is_pinned = |e: &Entry| pinned.get(&e.family) == Some(&e.key);
            let mut order: Vec<&Entry> = g.entries.values().filter(|e| &e.key != keep).collect();
            order.sort_by_key(|e| (is_pinned(e), e.last_used));
            // A block is freed with its last reference: count what each victim really returns.
            let mut left: HashMap<Key, usize> = g.refs.clone();
            for e in order {
                if total <= limit {
                    break;
                }
                total -= e.bytes;
                for b in &e.blocks {
                    if let Some(n) = left.get_mut(b) {
                        *n -= 1;
                        if *n == 0 {
                            total -= g.block_bytes.get(b).copied().unwrap_or(0);
                        }
                    }
                }
                victims.push(e.clone());
            }
        }
        for v in victims {
            tracing::info!(
                "state cache: evicted a {}-token snapshot ({} KiB manifest)",
                v.len,
                v.bytes >> 10
            );
            self.unlink(&self.path_of(&v));
            self.drop_entry(&v.key);
        }
    }

    fn save_index(&self) {
        let index = IndexFile {
            uses: self
                .lock()
                .entries
                .values()
                .map(|e| (hex(&e.key), (e.last_used, e.hits)))
                .collect(),
        };
        let Ok(json) = serde_json::to_vec(&index) else {
            return;
        };
        let _ = self.write_file(&self.dir.join(INDEX_FILE), &json);
    }

    /// Snapshots held and the bytes they and their blocks take.
    pub fn stats(&self) -> (usize, u64) {
        let g = self.lock();
        let blocks: u64 = g
            .block_bytes
            .iter()
            .filter(|(h, _)| g.refs.contains_key(*h))
            .map(|(_, n)| *n)
            .sum();
        (
            g.entries.len(),
            g.entries.values().map(|e| e.bytes).sum::<u64>() + blocks,
        )
    }

    /// The token counts of the snapshots held, ascending.
    pub fn lengths(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.lock().entries.values().map(|e| e.len).collect();
        v.sort_unstable();
        v
    }

    /// Blocks on disk, for tests and status.
    pub fn block_count(&self) -> usize {
        self.lock().block_bytes.len()
    }
}

/// `bytes` in a new file at `path`, synced and handed back to the kernel, so a snapshot never
/// pushes the weights out of the page cache.
fn write_whole(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)?;
    super::kv_disk::release_written(&f);
    Ok(())
}

/// Bytes read from the head of each model file for its digest: in a GGUF, the metadata and the
/// tensor table.
pub const DIGEST_HEAD_BYTES: usize = 4 << 20;

/// The identity of a set of model files: each file's length, modification time and head, so a
/// file requantised or replaced gives another digest without its weights being read.
#[derive(Default)]
pub struct FilesDigest(Sha256);

impl FilesDigest {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one file, from its metadata and the first bytes it holds (at most
    /// `DIGEST_HEAD_BYTES` are used).
    pub fn file(&mut self, meta: &std::fs::Metadata, head: &[u8]) {
        self.0.update(meta.len().to_le_bytes());
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        self.0.update(mtime.to_le_bytes());
        self.0.update(&head[..head.len().min(DIGEST_HEAD_BYTES)]);
    }

    /// Add the file at `path`.
    pub fn path(&mut self, path: &Path) -> std::io::Result<()> {
        use std::io::Read;
        let mut head = Vec::new();
        std::fs::File::open(path)?
            .take(DIGEST_HEAD_BYTES as u64)
            .read_to_end(&mut head)?;
        self.file(&std::fs::metadata(path)?, &head);
        Ok(())
    }

    pub fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCOPE: [u8; 32] = [7; 32];

    fn dir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("loken-state-store-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn seq(n: u32, salt: u32) -> Vec<u32> {
        (0..n).map(|i| i * 31 + salt).collect()
    }

    /// A put with `len / block` blocks of 10 bytes each.
    fn put(store: &StateStore, t: &[u32], len: usize, manifest: &[u8]) {
        let nb = len / store.block_tokens();
        store
            .put(&store.keys(&SCOPE, t), len, manifest, nb, |_| vec![1u8; 10])
            .unwrap();
    }

    #[test]
    fn the_longest_stored_prefix_of_a_prompt_is_found() {
        let d = dir("longest");
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        let base = seq(40, 1);
        put(&store, &base, 13, b"short");
        put(&store, &base, 37, b"long");
        // Storing a state that is stored already changes nothing.
        put(&store, &base, 37, b"long");
        assert_eq!(store.stats().0, 2);
        let mut other = base.clone();
        other[20] ^= 1;
        put(&store, &other, 40, b"diverged");

        let mut prompt = base.clone();
        prompt.extend(seq(10, 99));
        let keys = store.keys(&SCOPE, &prompt);
        let found = store.candidates(&keys, prompt.len());
        assert_eq!(
            found.iter().map(|e| e.len).collect::<Vec<_>>(),
            vec![37, 13]
        );
        assert_eq!(store.read(&found[0]).unwrap(), b"long");
        // A cap below the longest picks the next, and a prompt that diverges early the first only.
        assert_eq!(store.candidates(&keys, 36)[0].len, 13);
        let mut early = base.clone();
        early[5] ^= 1;
        assert!(store.candidates(&store.keys(&SCOPE, &early), 40).is_empty());
        // Another scope shares nothing.
        assert!(store
            .candidates(&store.keys(&[8; 32], &prompt), 50)
            .is_empty());
        // A restart finds the same entries.
        let again = StateStore::open(d.clone(), 0, 8).unwrap();
        assert_eq!(again.candidates(&again.keys(&SCOPE, &prompt), 50).len(), 2);
        let _ = std::fs::remove_dir_all(d);
    }

    /// A store in host memory finds, reads and forgets states as one in a directory does, and
    /// writes nothing.
    #[test]
    fn a_store_in_memory_keeps_its_states_without_a_file() {
        let store = StateStore::in_memory(1 << 20, 8);
        assert!(store.is_in_memory());
        let t = seq(64, 5);
        put(&store, &t, 24, b"m1");
        put(&store, &t, 61, b"m3");
        assert_eq!(store.stats().0, 2);
        assert_eq!(store.block_count(), 7);
        let keys = store.keys(&SCOPE, &t);
        let found = store.candidates(&keys, 64);
        assert_eq!(
            found.iter().map(|e| e.len).collect::<Vec<_>>(),
            vec![61, 24]
        );
        assert_eq!(store.read(&found[0]).unwrap(), b"m3");
        assert_eq!(
            store.read_block(&found[0].blocks[6]).unwrap(),
            vec![1u8; 10]
        );
        store.forget(&found[0]);
        assert_eq!(store.stats().0, 1);
        assert_eq!(store.block_count(), 3);
        assert!(!store.dir().exists(), "a store in memory has no directory");
    }

    #[test]
    fn snapshots_of_one_conversation_store_their_shared_blocks_once() {
        let d = dir("shared");
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        let t = seq(64, 5);
        put(&store, &t, 24, b"m1");
        assert_eq!(store.block_count(), 3);
        put(&store, &t, 40, b"m2");
        put(&store, &t, 61, b"m3");
        // Three manifests, blocks 0..7 written once each: 7 blocks, not 3 + 5 + 7.
        assert_eq!(store.block_count(), 7);
        let keys = store.keys(&SCOPE, &t);
        let e = store.candidates(&keys, 64).remove(0);
        assert_eq!(e.len, 61);
        assert_eq!(e.blocks.len(), 7);
        assert_eq!(store.read_block(&e.blocks[6]).unwrap(), vec![1u8; 10]);
        // Forgetting the longest frees only the blocks no other manifest names.
        store.forget(&e);
        assert_eq!(store.block_count(), 5);
        let again = StateStore::open(d.clone(), 0, 8).unwrap();
        assert_eq!(again.stats().0, 2);
        assert_eq!(again.block_count(), 5);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn the_least_recently_used_goes_first_and_the_pinned_prefix_last() {
        let d = dir("evict");
        // Each snapshot is a 10-byte manifest plus 12 bytes of header; none has blocks.
        let store = StateStore::open(d.clone(), 3 * 22, 8).unwrap();
        let a = seq(7, 1);
        let b = seq(7, 5000);
        let c = seq(7, 9000);
        let blob = [0u8; 10];
        let (ka, kb, kc) = (
            store.keys(&SCOPE, &a),
            store.keys(&SCOPE, &b),
            store.keys(&SCOPE, &c),
        );
        put(&store, &a, 7, &blob);
        put(&store, &b, 7, &blob);
        put(&store, &c, 7, &blob);
        // `a` is the oldest but was restored: pinned. The next write drops `b`, not `a`.
        let ea = store.candidates(&ka, 7).remove(0);
        store.read(&ea).unwrap();
        let e = seq(7, 13000);
        put(&store, &e, 7, &blob);
        assert_eq!(store.stats().0, 3);
        assert!(
            !store.candidates(&ka, 7).is_empty(),
            "the pinned entry stays"
        );
        assert!(
            store.candidates(&kb, 7).is_empty(),
            "the oldest unpinned goes"
        );
        assert!(!store.candidates(&kc, 7).is_empty());
        // A manifest larger than the whole budget is not stored.
        let big = seq(7, 20000);
        put(&store, &big, 7, &[0u8; 80]);
        assert!(store.candidates(&store.keys(&SCOPE, &big), 7).is_empty());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn eviction_counts_a_block_with_its_last_reference() {
        let d = dir("evictblocks");
        // Blocks are 10 bytes, manifests 12 + 16 per block + payload. Budget for two of the three.
        let store = StateStore::open(d.clone(), 200, 8).unwrap();
        let t = seq(48, 3);
        put(&store, &t, 16, b"aa"); // 2 blocks: 20 + 12 + 32 + 2 = 66 bytes
        put(&store, &t, 24, b"bb"); // + 1 block + 62 -> 138
        assert_eq!(store.stats().0, 2);
        put(&store, &t, 32, b"cc"); // + 1 block + 78 -> 226, over 200: the oldest goes, leaving 180
        assert_eq!(store.stats().0, 2);
        assert!(store.stats().1 <= 200);
        assert!(store.candidates(&store.keys(&SCOPE, &t), 16).is_empty());
        assert_eq!(store.candidates(&store.keys(&SCOPE, &t), 48)[0].len, 32);
        // Blocks 0..4 are all still named by the survivors.
        assert_eq!(store.block_count(), 4);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_half_written_block_or_manifest_is_never_referenced() {
        let d = dir("crash");
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        let t = seq(32, 9);
        put(&store, &t, 16, b"ok");
        drop(store);
        // An interrupted write leaves a temp file, a block no manifest names, and a manifest that
        // names a block that is gone.
        std::fs::write(d.join(format!("{TMP_PREFIX}9-9")), b"half").unwrap();
        let orphan = d
            .join(BLOCK_DIR)
            .join(format!("{}.{BLOCK_SUFFIX}", hex(&[9u8; 16])));
        std::fs::write(&orphan, b"orphan").unwrap();
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        assert!(!orphan.exists());
        assert!(!d.join(format!("{TMP_PREFIX}9-9")).exists());
        let keys = store.keys(&SCOPE, &t);
        let e = store.candidates(&keys, 32).remove(0);
        std::fs::remove_file(store.block_path(&e.blocks[1])).unwrap();
        drop(store);
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        assert_eq!(
            store.stats().0,
            0,
            "a manifest with a missing block is dropped"
        );
        assert_eq!(store.block_count(), 0);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_damaged_directory_opens_and_is_tidied() {
        let d = dir("damaged");
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        let t = seq(16, 3);
        put(&store, &t, 16, b"state");
        drop(store);
        std::fs::write(d.join(INDEX_FILE), b"{ not json").unwrap();
        std::fs::write(d.join("stray.txt"), b"x").unwrap();
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        assert_eq!(store.stats().0, 1);
        assert!(!d.join("stray.txt").exists());
        // A file that vanished is dropped on the read that finds it gone.
        let keys = store.keys(&SCOPE, &t);
        let e = store.candidates(&keys, 16).remove(0);
        std::fs::remove_file(store.path_of(&e)).unwrap();
        assert!(store.read(&e).is_err());
        assert!(store.candidates(&keys, 16).is_empty());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_model_file_that_changes_gives_another_digest() {
        let d = dir("digest");
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("model.gguf");
        std::fs::write(&f, b"header-one and the weights").unwrap();
        let digest = |p: &Path| {
            let mut h = FilesDigest::new();
            h.path(p).unwrap();
            h.finish()
        };
        let a = digest(&f);
        assert_eq!(a, digest(&f));
        std::fs::write(&f, b"header-two and the weights").unwrap();
        assert_ne!(a, digest(&f));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn names_and_the_index_hold_hashes_and_numbers_only() {
        let d = dir("names");
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        let t = seq(20, 123_456);
        put(&store, &t, 20, b"state");
        let mut text = String::new();
        for item in std::fs::read_dir(&d).unwrap() {
            let p = item.unwrap().path();
            text.push_str(&p.file_name().unwrap().to_string_lossy());
            if p.file_name().is_some_and(|n| n == INDEX_FILE) {
                text.push_str(&std::fs::read_to_string(&p).unwrap());
            }
        }
        for tok in &t {
            assert!(!text.contains(&tok.to_string()), "token {tok} appears");
        }
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_scopes_block_size_is_settled_once_and_survives_a_restart() {
        let d = dir("settle");
        let (a, b) = ([1u8; 32], [2u8; 32]);
        let store = StateStore::open(d.clone(), 0, 0).unwrap();
        assert_eq!(store.settle(&a, |_| 32), 32);
        assert_eq!(store.settle(&a, |_| 99), 32, "settled: not derived again");
        assert_eq!(
            store.settle(&b, |_| 99),
            99,
            "another scope derives its own"
        );
        drop(store);
        let again = StateStore::open(d.clone(), 0, 0).unwrap();
        assert_eq!(again.block_tokens_of(&a), 32);
        assert_eq!(again.block_tokens_of(&b), 99);
        // An operator's block size wins for every scope.
        let fixed = StateStore::open(dir("settle-fixed"), 0, 16).unwrap();
        assert_eq!(fixed.settle(&a, |_| 99), 16);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_write_is_admitted_on_an_empty_queue_and_refused_behind_a_slow_one() {
        let d = dir("admit");
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        let first = store.admit(1 << 20, None).expect("an empty queue admits");
        assert!(
            store.admit(1 << 20, None).is_none(),
            "unmeasured: no queueing"
        );
        drop(first);
        assert!(
            store.admit(1 << 20, None).is_some(),
            "released bytes leave the queue"
        );
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn the_budget_follows_the_free_space_unless_the_operator_set_one() {
        let d = dir("limit");
        let derived = StateStore::open(d.clone(), 0, 8).unwrap();
        let free = crate::inference::cache::state_tune::free_space(&d).unwrap();
        let room = derived.room();
        assert!(room > free / 2, "room {room} of {free} free");
        let set = StateStore::open(dir("limit-set"), 1 << 20, 8).unwrap();
        assert_eq!(set.room(), 1 << 20);
        let _ = std::fs::remove_dir_all(d);
    }
}
