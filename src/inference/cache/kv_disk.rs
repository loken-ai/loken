//! The disk tier under the KV snapshots: a resident sequence written as blocks of
//! `block_tokens` tokens, each block one safetensors file holding every layer's K and V
//! quantised to Q8_0, addressed by a hash chained over the tokens so that two
//! conversations sharing a prefix share its blocks, and one manifest per sequence naming
//! its blocks. Writes happen after a response; reads bring back only the blocks a prompt
//! covers.

use anyhow::{anyhow, Context, Result};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One layer's rows for a range of tokens, token-major `[n_tok, n_kv, head_dim]` f32,
/// K then V; `None` for a layer that keeps no KV of its own.
pub type LayerRows = Option<(Vec<f32>, Vec<f32>)>;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub model: String,
    pub layout: u64,
    pub tokens: Vec<u32>,
    /// Tokens covered by `blocks`: a multiple of the block size, never past the KV.
    pub covered: usize,
    pub blocks: Vec<u64>,
    pub bytes: u64,
    pub last_used: u64,
    /// KV element dtype code (0 = F16, 1 = F32, 2 = BF16). A cold import rebuilds tensors
    /// the model's cache accepts; the dtype is only known while a model is warm, so it is
    /// recorded here. Absent in manifests written before this field: default 0 (F16).
    #[serde(default)]
    pub kv_dtype: u8,
    /// Hash of the side blob holding this snapshot's windowed layers, when it has any.
    /// A windowed layer's rows depend on the whole sequence, not on a token block, so
    /// they cannot share the content-addressed chain; the blob is per-manifest and reused
    /// only when the whole prefix matches. Absent in older manifests.
    #[serde(default)]
    pub window_blob: Option<u64>,
}

pub struct KvDiskStore {
    dir: PathBuf,
    budget: u64,
    block_tokens: usize,
    /// Persists run on their own threads; one at a time per store, or the garbage
    /// collector of one would sweep the blocks another is still writing.
    persist_lock: Mutex<()>,
    index: Mutex<Vec<Manifest>>,
}

fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed ^ 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The chain of block hashes for `tokens` under one model and layout: block `b` hashes
/// block `b - 1` and its own tokens, so equal prefixes give equal chains.
pub fn block_chain(model: &str, layout: u64, tokens: &[u32], block_tokens: usize) -> Vec<u64> {
    let mut prev = fnv1a(layout, model.as_bytes());
    tokens
        .chunks_exact(block_tokens)
        .map(|blk| {
            let mut bytes = Vec::with_capacity(8 + blk.len() * 4);
            bytes.extend_from_slice(&prev.to_le_bytes());
            for t in blk {
                bytes.extend_from_slice(&t.to_le_bytes());
            }
            prev = fnv1a(prev, &bytes);
            prev
        })
        .collect()
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Written pages are handed back to the kernel once they are on disk, so a snapshot
/// never pushes the weights out of the page cache.
fn release_written(file: &std::fs::File) {
    let _ = file.sync_data();
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

impl KvDiskStore {
    /// Opens (creating) the store and reads its manifests. `budget` in bytes, 0 for
    /// no limit; `block_tokens` the block size every manifest in the store uses.
    pub fn open(dir: PathBuf, budget: u64, block_tokens: usize) -> Result<Self> {
        std::fs::create_dir_all(dir.join("blocks"))?;
        std::fs::create_dir_all(dir.join("manifests"))?;
        let mut index = Vec::new();
        for entry in std::fs::read_dir(dir.join("manifests"))? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match std::fs::read(&path)
                .map_err(anyhow::Error::from)
                .and_then(|b| serde_json::from_slice::<Manifest>(&b).map_err(Into::into))
            {
                Ok(m) => index.push(m),
                Err(e) => {
                    tracing::warn!(
                        "kv disk: manifest {} unreadable ({e}); removed",
                        path.display()
                    );
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        Ok(Self {
            dir,
            budget,
            block_tokens,
            persist_lock: Mutex::new(()),
            index: Mutex::new(index),
        })
    }

    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    fn block_path(&self, hash: u64) -> PathBuf {
        self.dir
            .join("blocks")
            .join(format!("{hash:016x}.safetensors"))
    }

    fn manifest_path(&self, m: &Manifest) -> PathBuf {
        let last = m.blocks.last().copied().unwrap_or(0);
        self.dir.join("manifests").join(format!("{last:016x}.json"))
    }

    /// The manifest sharing the longest block-aligned prefix with `prompt`, as
    /// `(manifest index, covered tokens)`.
    pub fn best(&self, model: &str, layout: u64, prompt: &[u32]) -> Option<(usize, usize)> {
        let chain = block_chain(model, layout, prompt, self.block_tokens);
        let index = self.index.lock().ok()?;
        index
            .iter()
            .enumerate()
            .filter(|(_, m)| m.model == model && m.layout == layout)
            .map(|(i, m)| {
                let shared = m
                    .blocks
                    .iter()
                    .zip(&chain)
                    .take_while(|(a, b)| a == b)
                    .count();
                (i, shared * self.block_tokens)
            })
            .filter(|(_, covered)| *covered > 0)
            .max_by_key(|(_, covered)| *covered)
    }

    /// The tokens a manifest covers, to seed the resident entry on a restore.
    /// The KV dtype code recorded for a manifest.
    pub fn manifest_kv_dtype(&self, index: usize) -> u8 {
        self.index
            .lock()
            .ok()
            .and_then(|g| g.get(index).map(|m| m.kv_dtype))
            .unwrap_or(0)
    }

    /// The block-aligned prefix a persist of `kv_len` tokens would cover.
    pub fn covered_len(&self, tokens_len: usize, kv_len: usize) -> usize {
        kv_len.min(tokens_len) / self.block_tokens * self.block_tokens
    }

    /// The blocks of a prefix not yet on disk, as `(index, from, to)` spans, so a caller
    /// can copy only those out before writing. A chain shares its head with any earlier
    /// persist of the same conversation, which is why most turns leave one block to do.
    pub fn missing_blocks(
        &self,
        model: &str,
        layout: u64,
        tokens: &[u32],
        covered: usize,
    ) -> Vec<(usize, usize, usize)> {
        block_chain(
            model,
            layout,
            &tokens[..covered.min(tokens.len())],
            self.block_tokens,
        )
        .iter()
        .enumerate()
        .filter(|(_, h)| std::fs::metadata(self.block_path(**h)).is_err())
        .map(|(b, _)| (b, b * self.block_tokens, (b + 1) * self.block_tokens))
        .collect()
    }

    /// The token prefix length a manifest covers.
    pub fn manifest_covered(&self, index: usize) -> usize {
        self.index
            .lock()
            .ok()
            .and_then(|g| g.get(index).map(|m| m.covered))
            .unwrap_or(0)
    }

    /// Whether a manifest carries a windowed-layer side blob (reusable only whole).
    pub fn manifest_windowed(&self, index: usize) -> bool {
        self.index
            .lock()
            .ok()
            .and_then(|g| g.get(index).map(|m| m.window_blob.is_some()))
            .unwrap_or(false)
    }

    pub fn manifest_tokens(&self, index: usize) -> Option<(Vec<u32>, usize)> {
        let mut g = self.index.lock().ok()?;
        let m = g.get_mut(index)?;
        m.last_used = now_secs();
        Some((m.tokens.clone(), m.covered))
    }

    /// Writes the full blocks of a sequence that are not on disk yet, then its
    /// manifest; `rows(from, to)` supplies every layer's rows for one block.
    #[allow(clippy::too_many_arguments)]
    pub fn persist(
        &self,
        model: &str,
        layout: u64,
        tokens: &[u32],
        kv_len: usize,
        kv_dtype: u8,
        window: &[LayerRows],
        mut rows: impl FnMut(usize, usize) -> Result<Vec<LayerRows>>,
    ) -> Result<()> {
        let _serial = self
            .persist_lock
            .lock()
            .map_err(|_| anyhow!("kv disk persist lock poisoned"))?;
        let covered = kv_len.min(tokens.len()) / self.block_tokens * self.block_tokens;
        if covered == 0 {
            return Ok(());
        }
        let chain = block_chain(model, layout, &tokens[..covered], self.block_tokens);
        let mut bytes = 0u64;
        for (b, hash) in chain.iter().enumerate() {
            let path = self.block_path(*hash);
            if let Ok(meta) = std::fs::metadata(&path) {
                bytes += meta.len();
                continue;
            }
            let from = b * self.block_tokens;
            let layers = rows(from, from + self.block_tokens)?;
            bytes += self.write_block(&path, &layers, kv_dtype)?;
        }
        // Windowed layers, if any, go in a per-manifest side blob keyed off the prefix.
        let window_blob = if window.iter().any(Option::is_some) {
            let wh = fnv1a(
                layout ^ 0x77,
                &tokens[..covered.min(tokens.len())]
                    .iter()
                    .flat_map(|t| t.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
            let path = self.block_path(wh);
            if std::fs::metadata(&path).is_err() {
                bytes += self.write_block(&path, window, kv_dtype)?;
            }
            Some(wh)
        } else {
            None
        };
        let manifest = Manifest {
            model: model.to_string(),
            layout,
            tokens: tokens[..covered].to_vec(),
            covered,
            blocks: chain,
            bytes,
            last_used: now_secs(),
            kv_dtype,
            window_blob,
        };
        let path = self.manifest_path(&manifest);
        let json = serde_json::to_vec(&manifest)?;
        std::fs::write(&path, json).with_context(|| path.display().to_string())?;
        {
            let mut g = self
                .index
                .lock()
                .map_err(|_| anyhow!("kv disk index poisoned"))?;
            // A manifest whose blocks this one extends is the same conversation, one turn on.
            g.retain(|m| {
                !(m.model == manifest.model
                    && m.layout == manifest.layout
                    && manifest.blocks.starts_with(&m.blocks)
                    && self.manifest_path(m) != path)
            });
            g.retain(|m| self.manifest_path(m) != path);
            g.push(manifest);
        }
        self.collect_garbage()
    }

    fn write_block(&self, path: &Path, layers: &[LayerRows], dtype_code: u8) -> Result<u64> {
        let data = encode_block(layers, dtype_code)?;
        let tmp = path.with_extension("tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&data)?;
            release_written(&f);
        }
        std::fs::rename(&tmp, path)?;
        Ok(std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
    }

    /// Reads `blocks` leading blocks of a manifest back as rows per layer, concatenated
    /// along the token axis. A block that is missing or damaged fails the whole load and
    /// is removed with its manifests, so the next attempt prefills.
    pub fn load(&self, index: usize, blocks: usize, n_layers: usize) -> Result<Vec<LayerRows>> {
        let chain: Vec<u64> = {
            let g = self
                .index
                .lock()
                .map_err(|_| anyhow!("kv disk index poisoned"))?;
            g.get(index)
                .ok_or_else(|| anyhow!("kv disk: manifest {index} gone"))?
                .blocks
                .iter()
                .take(blocks)
                .copied()
                .collect()
        };
        let window_blob = {
            let g = self
                .index
                .lock()
                .map_err(|_| anyhow!("kv disk index poisoned"))?;
            g.get(index).and_then(|m| m.window_blob)
        };
        let mut out: Vec<LayerRows> = (0..n_layers).map(|_| None).collect();
        for hash in &chain {
            let path = self.block_path(*hash);
            match self.read_block(&path, &mut out, n_layers) {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!(
                        "kv disk: block {} unreadable ({e}); dropped",
                        path.display()
                    );
                    self.forget_block(*hash);
                    return Err(e);
                }
            }
        }
        // Windowed layers come from the side blob, filling the slots the blocks left empty.
        if let Some(wh) = window_blob {
            let path = self.block_path(wh);
            self.read_block(&path, &mut out, n_layers)?;
        }
        Ok(out)
    }

    fn read_block(&self, path: &Path, out: &mut [LayerRows], n_layers: usize) -> Result<()> {
        let file = std::fs::File::open(path)?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&map).map_err(|e| anyhow!("{e}"))?;
        for (l, slot) in out.iter_mut().enumerate().take(n_layers) {
            let (Ok(k), Ok(v)) = (st.tensor(&format!("k.{l}")), st.tensor(&format!("v.{l}")))
            else {
                continue;
            };
            let (kf, vf) = (block_rows(&k), block_rows(&v));
            match slot {
                Some((ok, ov)) => {
                    ok.extend_from_slice(&kf);
                    ov.extend_from_slice(&vf);
                }
                slot => *slot = Some((kf, vf)),
            }
        }
        Ok(())
    }

    fn forget_block(&self, hash: u64) {
        let _ = std::fs::remove_file(self.block_path(hash));
        if let Ok(mut g) = self.index.lock() {
            let (gone, kept): (Vec<_>, Vec<_>) =
                g.drain(..).partition(|m| m.blocks.contains(&hash));
            *g = kept;
            for m in gone {
                let _ = std::fs::remove_file(self.manifest_path(&m));
            }
        }
    }

    /// Drops the least recently used manifests until the store fits its budget, then
    /// every block no manifest names.
    fn collect_garbage(&self) -> Result<()> {
        let mut g = self
            .index
            .lock()
            .map_err(|_| anyhow!("kv disk index poisoned"))?;
        if self.budget > 0 {
            g.sort_by_key(|m| std::cmp::Reverse(m.last_used));
            let mut named: HashSet<u64> = HashSet::new();
            let mut total = 0u64;
            let mut keep = Vec::new();
            for m in g.drain(..) {
                let mut fresh = 0u64;
                for h in m.blocks.iter().chain(m.window_blob.iter()) {
                    if named.insert(*h) {
                        fresh += std::fs::metadata(self.block_path(*h))
                            .map(|x| x.len())
                            .unwrap_or(0);
                    }
                }
                if total + fresh > self.budget && !keep.is_empty() {
                    let _ = std::fs::remove_file(self.manifest_path(&m));
                    continue;
                }
                total += fresh;
                keep.push(m);
            }
            *g = keep;
        }
        let mut named: HashSet<u64> = g.iter().flat_map(|m| m.blocks.iter().copied()).collect();
        named.extend(g.iter().filter_map(|m| m.window_blob));
        for entry in std::fs::read_dir(self.dir.join("blocks"))? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
                continue; // a block still being written by a persist
            }
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            match u64::from_str_radix(stem, 16) {
                Ok(h) if named.contains(&h) => {}
                _ => {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        Ok(())
    }

    /// Manifests and bytes on disk, for the status endpoints.
    pub fn stats(&self) -> (usize, u64) {
        let g = self.index.lock().map(|g| g.clone()).unwrap_or_default();
        let named: HashMap<u64, u64> = g
            .iter()
            .flat_map(|m| m.blocks.iter().copied())
            .map(|h| {
                (
                    h,
                    std::fs::metadata(self.block_path(h))
                        .map(|x| x.len())
                        .unwrap_or(0),
                )
            })
            .collect();
        (g.len(), named.values().sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_agree_on_a_shared_prefix_and_part_after_it() {
        let a: Vec<u32> = (0..600).collect();
        let mut b = a[..300].to_vec();
        b.extend(1000..1300);
        let ca = block_chain("m", 7, &a, 100);
        let cb = block_chain("m", 7, &b, 100);
        assert_eq!(ca.len(), 6);
        assert_eq!(&ca[..3], &cb[..3]);
        assert_ne!(ca[3], cb[3]);
        assert_ne!(block_chain("other", 7, &a, 100)[0], ca[0]);
    }

    #[test]
    fn a_block_round_trips_and_the_store_finds_its_prefix() {
        let dir = std::env::temp_dir().join(format!("loken-kv-disk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = KvDiskStore::open(dir.clone(), 0, 4).unwrap();
        let tokens: Vec<u32> = (0..10).collect();
        let n_kv = 1;
        let hd = 32;
        let row =
            |t: usize| -> Vec<f32> { (0..n_kv * hd).map(|i| (t * 100 + i) as f32 / 7.0).collect() };
        store
            .persist("m", 1, &tokens, tokens.len(), 1, &[], |from, to| {
                let mut k = Vec::new();
                let mut v = Vec::new();
                for t in from..to {
                    k.extend(row(t));
                    v.extend(row(t + 1000));
                }
                Ok(vec![Some((k, v)), None])
            })
            .unwrap();
        // 10 tokens, blocks of 4: two blocks cover 8 tokens.
        let prompt: Vec<u32> = (0..9).collect();
        let (idx, covered) = store.best("m", 1, &prompt).unwrap();
        assert_eq!(covered, 8);
        let rows = store.load(idx, 2, 2).unwrap();
        assert!(rows[1].is_none());
        let (k, _v) = rows[0].as_ref().unwrap();
        assert_eq!(k.len(), 8 * n_kv * hd);
        let want = row(5);
        assert_eq!(
            &k[5 * hd..6 * hd],
            &want[..],
            "rows must come back bit-exact"
        );
        assert!(store.best("m", 2, &prompt).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f32_and_f16_rows_round_trip_exactly() {
        // F32 rows (dtype code 1) come back bit-for-bit; F16 rows (code 0) come back
        // exactly when the values are F16-representable, as a dequantised cache's are.
        let dir = std::env::temp_dir().join(format!("loken-kv-exact-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = KvDiskStore::open(dir.clone(), 0, 4).unwrap();
        let hd = 32usize;
        let k: Vec<f32> = (0..4 * hd).map(|i| (i as f32) * 0.31830988 - 3.7).collect();
        let v: Vec<f32> = (0..4 * hd).map(|i| 1.0 / (i as f32 + 2.0)).collect();
        let path = store.block_path(42);
        store
            .write_block(&path, &[Some((k.clone(), v.clone()))], 1)
            .unwrap();
        let mut out: Vec<LayerRows> = vec![None];
        store.read_block(&path, &mut out, 1).unwrap();
        let (rk, rv) = out[0].as_ref().unwrap();
        assert_eq!(rk, &k, "F32 K not bit-exact");
        assert_eq!(rv, &v, "F32 V not bit-exact");

        // F16 path: half-representable values survive exactly.
        let kf16: Vec<f32> = (0..4 * hd)
            .map(|i| half::f16::from_f32(i as f32 * 0.5 - 8.0).to_f32())
            .collect();
        let p16 = store.block_path(43);
        store
            .write_block(&p16, &[Some((kf16.clone(), kf16.clone()))], 0)
            .unwrap();
        let mut o16: Vec<LayerRows> = vec![None];
        store.read_block(&p16, &mut o16, 1).unwrap();
        assert_eq!(
            &o16[0].as_ref().unwrap().0,
            &kf16,
            "F16 not exact for representable values"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A block: each layer's rows at their native dtype (F16 for a dtype code of 0, F32
/// otherwise), which is exact for the values a snapshot holds, as a plain safetensors
/// file the reader maps. Not compressed: F16 KV rows are close to incompressible, and
/// deflating them measured a tenth of the write rate for a tenth of the space.
fn encode_block(layers: &[LayerRows], dtype_code: u8) -> Result<Vec<u8>> {
    use safetensors::Dtype;
    let f16 = dtype_code == 0;
    let mut owned: Vec<(String, Vec<u8>, usize)> = Vec::new();
    for (l, rows) in layers.iter().enumerate() {
        let Some((k, v)) = rows else { continue };
        for (name, data) in [("k", k), ("v", v)] {
            let bytes = if f16 {
                let halves: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
                let mut b = Vec::with_capacity(halves.len() * 2);
                for h in halves {
                    b.extend_from_slice(&h.to_le_bytes());
                }
                b
            } else {
                let mut b = Vec::with_capacity(data.len() * 4);
                for &x in data {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                b
            };
            owned.push((format!("{name}.{l}"), bytes, data.len()));
        }
    }
    let dt = if f16 { Dtype::F16 } else { Dtype::F32 };
    let views: Vec<(String, safetensors::tensor::TensorView)> = owned
        .iter()
        .map(|(name, b, n)| {
            let view = safetensors::tensor::TensorView::new(dt, vec![*n], b)
                .map_err(|e| anyhow!("kv disk: {e}"))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize(views, None).map_err(|e| anyhow!("kv disk: {e}"))
}

/// A layer's tensor of a block back to f32 rows.
fn block_rows(view: &safetensors::tensor::TensorView) -> Vec<f32> {
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}
