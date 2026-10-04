//! A disk store of decode-state snapshots, found by the longest prefix of a prompt they cover.
//!
//! A snapshot is an opaque blob that stands for the state of a sequence after its first `len`
//! tokens. The store keys it by a chain of hashes over the tokens, so a prompt finds the longest
//! stored prefix of itself without the store ever holding a token: each block of `block_tokens`
//! tokens is hashed with its parent's hash and the scope (the identity of the model and numerics),
//! and the entry's key hashes the last full block's chain value with the length and the tokens
//! after that block. Two sequences that share a prefix share the chain over it, so a lookup is one
//! pass over the prompt and one map probe per distinct stored length.
//!
//! Privacy: file names and the index hold lengths, hashes, byte counts, use counts and times; the
//! blobs are whatever the caller encodes, KV tensors and not text. The store is off unless the
//! operator gives it a directory, and the files are prompt-derived state.
//!
//! Durability: every file is written beside its name, synced, then renamed in, so a reader sees a
//! whole file or none; the index is the same. A file the index does not know is taken up from its
//! name, one whose name is not ours is removed, and an unreadable index is dropped with a log
//! line and rebuilt from the names.
//!
//! Eviction: when the stored bytes pass the budget the least recently used entries go first. In
//! each family of entries - those sharing a first block - the entry used most, the longest
//! among equals, is pinned: it goes last, so a long prefix that conversations keep coming back
//! to outlives the one-off tails around it. The entry just written is never the victim.

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The hash a stored snapshot is found by.
pub type Key = [u8; 16];

const SUFFIX: &str = "snap";
const INDEX_FILE: &str = "index.json";
const TMP_PREFIX: &str = ".tmp-";

/// One stored snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub key: Key,
    /// Tokens the snapshot's state covers.
    pub len: usize,
    family: u64,
    pub bytes: u64,
    last_used: u64,
    hits: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct IndexFile {
    /// Key (hex) to `[last_used, hits]`.
    uses: HashMap<String, (u64, u64)>,
}

struct Inner {
    entries: HashMap<Key, Entry>,
}

pub struct StateStore {
    dir: PathBuf,
    budget: u64,
    block_tokens: usize,
    inner: Mutex<Inner>,
    tmp_seq: AtomicU64,
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
    let key_hex = parts.next()?;
    if parts.next().is_some() || key_hex.len() != 32 {
        return None;
    }
    let mut key = [0u8; 16];
    for (i, b) in key.iter_mut().enumerate() {
        *b = u8::from_str_radix(&key_hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some((len, family, key))
}

impl StateStore {
    /// Open (creating) the store in `dir`. `budget` is in bytes, 0 for no limit; `block_tokens`
    /// the block size of the hash chain, which must not change for a store in use.
    pub fn open(dir: PathBuf, budget: u64, block_tokens: usize) -> Result<Self> {
        if block_tokens == 0 {
            return Err(anyhow!(
                "state cache: block size must be at least one token"
            ));
        }
        std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
        let index: IndexFile = match std::fs::read(dir.join(INDEX_FILE)) {
            Ok(b) => serde_json::from_slice(&b).unwrap_or_else(|e| {
                tracing::warn!("state cache: index unreadable ({e}); rebuilt from the file names");
                IndexFile::default()
            }),
            Err(_) => IndexFile::default(),
        };
        let mut entries = HashMap::new();
        for item in std::fs::read_dir(&dir)? {
            let item = item?;
            let name = item.file_name().to_string_lossy().into_owned();
            if name == INDEX_FILE {
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
            let meta = item.metadata()?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            let (last_used, hits) = index.uses.get(&hex(&key)).copied().unwrap_or((mtime, 0));
            entries.insert(
                key,
                Entry {
                    key,
                    len,
                    family,
                    bytes: meta.len(),
                    last_used,
                    hits,
                },
            );
        }
        Ok(Self {
            dir,
            budget,
            block_tokens,
            inner: Mutex::new(Inner { entries }),
            tmp_seq: AtomicU64::new(0),
        })
    }

    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    /// The hashes of `tokens` under `scope`.
    pub fn keys<'a>(&self, scope: &[u8; 32], tokens: &'a [u32]) -> PrefixKeys<'a> {
        PrefixKeys::new(scope, self.block_tokens, tokens)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn path_of(&self, e: &Entry) -> PathBuf {
        self.dir.join(file_name(e.len, e.family, &e.key))
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

    /// The bytes of a stored snapshot, counted as a use. A file that cannot be read is dropped.
    pub fn read(&self, entry: &Entry) -> Result<Vec<u8>> {
        let path = self.path_of(entry);
        match std::fs::read(&path) {
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

    /// Remove a snapshot that proved unusable.
    pub fn forget(&self, entry: &Entry) {
        let _ = std::fs::remove_file(self.path_of(entry));
        self.lock().entries.remove(&entry.key);
        self.save_index();
    }

    /// Store `bytes` as the state after the first `len` tokens of the sequence `keys` hashes.
    /// Does nothing when that state is stored already, or when it alone exceeds the budget.
    pub fn put(&self, keys: &PrefixKeys, len: usize, bytes: &[u8]) -> Result<()> {
        let Some((key, family)) = keys.at(len) else {
            return Err(anyhow!("state cache: length {len} is outside the sequence"));
        };
        if self.budget > 0 && bytes.len() as u64 > self.budget {
            tracing::info!(
                "state cache: a {} MiB snapshot exceeds the {} MiB budget; not stored",
                bytes.len() >> 20,
                self.budget >> 20
            );
            return Ok(());
        }
        let known = {
            let mut g = self.lock();
            g.entries.get_mut(&key).map(|e| e.last_used = now_nanos())
        };
        if known.is_some() {
            self.save_index();
            return Ok(());
        }
        let entry = Entry {
            key,
            len,
            family,
            bytes: bytes.len() as u64,
            last_used: now_nanos(),
            hits: 0,
        };
        let path = self.path_of(&entry);
        let tmp = self.dir.join(format!(
            "{TMP_PREFIX}{}-{}",
            std::process::id(),
            self.tmp_seq.fetch_add(1, Ordering::Relaxed)
        ));
        write_whole(&tmp, bytes).with_context(|| tmp.display().to_string())?;
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(anyhow!("{}: {e}", path.display()));
        }
        if let Ok(d) = std::fs::File::open(&self.dir) {
            let _ = d.sync_all();
        }
        self.lock().entries.insert(key, entry);
        self.evict(&key);
        self.save_index();
        Ok(())
    }

    /// Drop entries, least recently used first and pinned ones last, until the store fits its
    /// budget. `keep` is never dropped.
    fn evict(&self, keep: &Key) {
        if self.budget == 0 {
            return;
        }
        let mut victims = Vec::new();
        {
            let g = self.lock();
            let mut total: u64 = g.entries.values().map(|e| e.bytes).sum();
            if total <= self.budget {
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
            for e in order {
                if total <= self.budget {
                    break;
                }
                total -= e.bytes;
                victims.push(e.clone());
            }
        }
        for v in victims {
            tracing::info!(
                "state cache: evicted a {}-token snapshot ({} MiB)",
                v.len,
                v.bytes >> 20
            );
            let _ = std::fs::remove_file(self.path_of(&v));
            self.lock().entries.remove(&v.key);
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
        let tmp = self.dir.join(format!(
            "{TMP_PREFIX}index-{}-{}",
            std::process::id(),
            self.tmp_seq.fetch_add(1, Ordering::Relaxed)
        ));
        if write_whole(&tmp, &json).is_ok()
            && std::fs::rename(&tmp, self.dir.join(INDEX_FILE)).is_err()
        {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// Snapshots held and the bytes they take.
    pub fn stats(&self) -> (usize, u64) {
        let g = self.lock();
        (g.entries.len(), g.entries.values().map(|e| e.bytes).sum())
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

    #[test]
    fn the_longest_stored_prefix_of_a_prompt_is_found() {
        let d = dir("longest");
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        let base = seq(40, 1);
        store.put(&store.keys(&SCOPE, &base), 13, b"short").unwrap();
        store.put(&store.keys(&SCOPE, &base), 37, b"long").unwrap();
        // Storing a state that is stored already changes nothing.
        store.put(&store.keys(&SCOPE, &base), 37, b"long").unwrap();
        assert_eq!(store.stats().0, 2);
        let mut other = base.clone();
        other[20] ^= 1;
        store
            .put(&store.keys(&SCOPE, &other), 40, b"diverged")
            .unwrap();

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

    #[test]
    fn the_least_recently_used_goes_first_and_the_pinned_prefix_last() {
        let d = dir("evict");
        let store = StateStore::open(d.clone(), 30, 8).unwrap();
        let blob = [0u8; 10];
        let a = seq(24, 1);
        let b = seq(24, 5000);
        let c = seq(24, 9000);
        let (ka, kb, kc) = (
            store.keys(&SCOPE, &a),
            store.keys(&SCOPE, &b),
            store.keys(&SCOPE, &c),
        );
        store.put(&ka, 24, &blob).unwrap();
        store.put(&kb, 24, &blob).unwrap();
        store.put(&kc, 24, &blob).unwrap();
        // `a` is the oldest but was restored: pinned. The next write drops `b`, not `a`.
        let ea = store.candidates(&ka, 24).remove(0);
        store.read(&ea).unwrap();
        let e = seq(24, 13000);
        store.put(&store.keys(&SCOPE, &e), 24, &blob).unwrap();
        assert_eq!(store.stats().0, 3);
        assert!(
            !store.candidates(&ka, 24).is_empty(),
            "the pinned entry stays"
        );
        assert!(
            store.candidates(&kb, 24).is_empty(),
            "the oldest unpinned goes"
        );
        assert!(!store.candidates(&kc, 24).is_empty());
        // A snapshot larger than the whole budget is not stored.
        let big = seq(24, 20000);
        store
            .put(&store.keys(&SCOPE, &big), 24, &[0u8; 31])
            .unwrap();
        assert!(store.candidates(&store.keys(&SCOPE, &big), 24).is_empty());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_damaged_directory_opens_and_is_tidied() {
        let d = dir("damaged");
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        let t = seq(16, 3);
        store.put(&store.keys(&SCOPE, &t), 16, b"state").unwrap();
        drop(store);
        std::fs::write(d.join(INDEX_FILE), b"{ not json").unwrap();
        std::fs::write(d.join("stray.txt"), b"x").unwrap();
        std::fs::write(d.join(format!("{TMP_PREFIX}1-1")), b"half").unwrap();
        let store = StateStore::open(d.clone(), 0, 8).unwrap();
        assert_eq!(store.stats().0, 1);
        assert!(!d.join("stray.txt").exists());
        assert!(!d.join(format!("{TMP_PREFIX}1-1")).exists());
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
        store.put(&store.keys(&SCOPE, &t), 20, b"state").unwrap();
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
}
