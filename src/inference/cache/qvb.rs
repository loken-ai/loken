//! Host-RAM cache of parsed quantized checkpoints (`QVarBuilder`s), so a model SWAP does not
//! re-read gigabytes from disk. The first load of a GGUF (or a converted fp8 sidecar) parses
//! it once into host block-quantized tensors; the cache keeps that `QVarBuilder` (device set
//! to CPU - the host blobs are device-independent, placement happens per layer at build time)
//! keyed by absolute path + mtime. A later load of the same file is an `Arc` clone: the swap
//! cost collapses to the GPU upload.
//!
//! Memory safety comes from the vram_manager HOST-cache registry: every entry registers a
//! reclaim hook, so RAM pressure (a CPU-spill plan, another model's load) drops cached
//! checkpoints biggest-first instead of strangling the requester. Insertion itself reclaims
//! when free RAM would drop too low, so filling the cache can never push the host into swap.

use crate::tensor::quantized::QVarBuilder;
use crate::tensor::Result;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

struct Entry {
    vb: QVarBuilder,
    bytes: u64,
    mtime: SystemTime,
}

static CACHE: OnceLock<Mutex<HashMap<String, Entry>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, Entry>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// How much host RAM to leave free after an insertion, below which other cached checkpoints
/// are reclaimed first - the OS page cache and any later CPU-spill plan need the room.
///
/// A share of the machine rather than a figure. It was 8 GiB flat, which is half of a 16 GB
/// host and a rounding error on a 256 GB one: the same number cannot mean "comfortable" on
/// both. An eighth of total RAM, and no floor or cap - those were two more hand-written sizes,
/// and a floor at 2 GiB reserves an eighth of an 8 GB machine twice over.
fn insert_headroom() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory() / 8
}

fn file_mtime(path: &std::path::Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn available_ram() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// `QVarBuilder::from_gguf` behind the host cache. Same result, but a repeat load of an
/// unchanged file returns the already-parsed tensors instead of re-reading the disk.
pub fn from_gguf_cached<P: AsRef<std::path::Path>>(
    path: P,
    device: &crate::tensor::Device,
) -> Result<QVarBuilder> {
    let path = path.as_ref();
    let key = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_string();
    let mtime = file_mtime(path);
    if let Some(m) = mtime {
        if let Some(e) = cache().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            if e.mtime == m {
                tracing::info!("qvb_cache: host-cache hit for {key}");
                return Ok(e.vb.clone_for_device(device));
            }
        }
    }
    let vb = QVarBuilder::from_gguf(path, &crate::tensor::Device::Cpu)?;
    let Some(m) = mtime.or_else(|| file_mtime(path)) else {
        // Unstat-able file: serve uncached rather than caching something unverifiable.
        return Ok(vb.clone_for_device(device));
    };
    insert(key, vb.clone(), m);
    Ok(vb.clone_for_device(device))
}

/// Cache an externally built `QVarBuilder` (the fp8 bridge's in-memory conversion) under the
/// source checkpoint's identity.
pub fn insert_for_path(path: &std::path::Path, vb: &QVarBuilder) {
    let key = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_string();
    if let Some(m) = file_mtime(path) {
        insert(key, vb.clone_for_device(&crate::tensor::Device::Cpu), m);
    }
}

fn insert(key: String, vb: QVarBuilder, mtime: SystemTime) {
    let bytes = vb.host_bytes();
    // Never let the cache be the reason the host runs out of RAM: reclaim other cached
    // checkpoints (and any other registered host caches) before this entry lands.
    let avail = available_ram();
    let headroom = insert_headroom();
    if avail < bytes + headroom {
        let freed =
            crate::inference::place::vram_manager::reclaim_host_ram(bytes + headroom - avail);
        if available_ram() < bytes + headroom / 2 {
            tracing::info!(
                "qvb_cache: not caching {key} ({:.1} GB) - host RAM too tight (freed {:.1} GB)",
                bytes as f64 / 1e9,
                freed as f64 / 1e9
            );
            return;
        }
    }
    tracing::info!(
        "qvb_cache: cached {key} ({:.1} GB host)",
        bytes as f64 / 1e9
    );
    let hook_key = key.clone();
    cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, Entry { vb, bytes, mtime });
    // Per-entry registration so the registry's biggest-first ordering sees real sizes. The
    // leaked name is bounded by the number of DISTINCT checkpoint paths ever cached.
    let name: &'static str = Box::leak(format!("qvb:{hook_key}").into_boxed_str());
    crate::inference::place::vram_manager::register_host_cache(
        name,
        bytes,
        Box::new(move || {
            let Ok(mut map) = cache().try_lock() else {
                return 0;
            };
            map.remove(&hook_key).map_or(0, |e| e.bytes)
        }),
    );
}

/// Test-only visibility: number of cached checkpoints.
#[cfg(test)]
pub fn len() -> usize {
    cache().lock().unwrap_or_else(|e| e.into_inner()).len()
}
