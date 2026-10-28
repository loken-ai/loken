use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

use super::model_dirs;

/// Progress callback for download tracking (bytes_downloaded, total_bytes)
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Ollama Manager for native Ollama model directory structure, over one directory or several
/// read as one catalogue. A model's manifest and every blob it names live in one directory.
pub struct OllamaManager {
    /// The directories in the order configured; the first receives models built here.
    dirs: Vec<PathBuf>,
}

impl OllamaManager {
    /// A manager on `models_dir`. When it is one of the configured Ollama directories, the
    /// manager reads every configured one.
    pub fn new(models_dir: PathBuf) -> Self {
        Self::with_dirs(model_dirs::catalogue_of(
            models_dir,
            crate::config::ollama_models_dirs(),
        ))
    }

    /// A manager reading `dirs` as one catalogue; the first receives models built here. No
    /// directory at all is the default one.
    pub fn with_dirs(dirs: Vec<PathBuf>) -> Self {
        let dirs = if dirs.is_empty() {
            vec![crate::config::Config::default_ollama_models_dir()]
        } else {
            dirs
        };
        Self { dirs }
    }

    /// The directories in the order a model held by several is read.
    fn read_dirs(&self) -> Vec<PathBuf> {
        model_dirs::read_order(&self.dirs)
    }

    /// The directory models built here are written to.
    fn write_dir(&self) -> &Path {
        &self.dirs[0]
    }

    /// The directory `name:tag` is read from, with its blobs, when one holds it. A model held
    /// by several is reported once, with the reason its copy is read.
    pub fn model_root(&self, name: &str, tag: &str) -> Option<PathBuf> {
        let holders: Vec<PathBuf> = self
            .read_dirs()
            .into_iter()
            .filter(|d| Self::find_manifest_in(d, name, tag).is_some())
            .collect();
        model_dirs::report_copies(&format!("{name}:{tag}"), &holders);
        holders.into_iter().next()
    }

    /// The directory holding the manifest of `name:tag` at its canonical path, else the one
    /// models are written to.
    fn root_of(&self, name: &str, tag: &str) -> PathBuf {
        self.read_dirs()
            .into_iter()
            .find(|d| Self::manifest_path_in(d, name, tag).is_file())
            .unwrap_or_else(|| self.write_dir().to_path_buf())
    }

    /// List all Ollama models across the directories. A model held by several directories is
    /// listed once, from the directory it is read from.
    pub fn list_models(&self) -> Result<Vec<OllamaModelEntry>> {
        let mut models: Vec<OllamaModelEntry> = Vec::new();
        let mut copies: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
        for dir in self.read_dirs() {
            let found = match Self::list_models_in(&dir) {
                Ok(found) => found,
                Err(e) if self.dirs.len() == 1 => return Err(e),
                Err(e) => {
                    warn!("Ollama directory {} skipped: {e}", dir.display());
                    continue;
                }
            };
            for model in found {
                let holders = copies.entry(model.id.clone()).or_default();
                holders.push(dir.clone());
                if holders.len() == 1 {
                    models.push(model);
                }
            }
        }
        for (id, holders) in &copies {
            model_dirs::report_copies(id, holders);
        }
        Ok(models)
    }

    /// The Ollama models of one directory.
    fn list_models_in(models_dir: &Path) -> Result<Vec<OllamaModelEntry>> {
        let registry_dir = models_dir.join("manifests").join("registry.ollama.ai");

        if !registry_dir.exists() {
            debug!(
                "Ollama registry directory does not exist: {:?}",
                registry_dir
            );
            return Ok(vec![]);
        }

        debug!("Scanning Ollama registry at: {:?}", registry_dir);
        let mut models = vec![];

        // Iterate through all publishers (library, bjoernb, etc.)
        match std::fs::read_dir(&registry_dir) {
            Ok(entries) => {
                for publisher_entry in entries {
                    let publisher_path = match publisher_entry {
                        Ok(e) => e.path(),
                        Err(e) => {
                            debug!("Error reading publisher entry: {}", e);
                            continue;
                        }
                    };

                    if !publisher_path.is_dir() {
                        debug!("Skipping non-directory: {:?}", publisher_path);
                        continue;
                    }

                    let publisher = match publisher_path.file_name() {
                        Some(name) => name.to_string_lossy().to_string(),
                        None => {
                            debug!("Could not get publisher name from: {:?}", publisher_path);
                            continue;
                        }
                    };

                    debug!("Scanning publisher: {}", publisher);

                    // Iterate through models in this publisher
                    match std::fs::read_dir(&publisher_path) {
                        Ok(model_entries) => {
                            for model_entry in model_entries {
                                let model_path = match model_entry {
                                    Ok(e) => e.path(),
                                    Err(e) => {
                                        debug!("Error reading model entry in {}: {}", publisher, e);
                                        continue;
                                    }
                                };

                                if !model_path.is_dir() {
                                    debug!(
                                        "Skipping non-directory in {}: {:?}",
                                        publisher, model_path
                                    );
                                    continue;
                                }

                                let model_name = match model_path.file_name() {
                                    Some(name) => name.to_string_lossy().to_string(),
                                    None => {
                                        debug!("Could not get model name from: {:?}", model_path);
                                        continue;
                                    }
                                };

                                // Construct full model ID (publisher/model or just model for library)
                                let full_model_id = if publisher == "library" {
                                    model_name.clone()
                                } else {
                                    format!("{}/{}", publisher, model_name)
                                };

                                debug!("Found model: {}", full_model_id);

                                // Read all tags for this model
                                match std::fs::read_dir(&model_path) {
                                    Ok(tag_entries) => {
                                        for tag_entry in tag_entries {
                                            let tag_path = match tag_entry {
                                                Ok(e) => e.path(),
                                                Err(e) => {
                                                    debug!(
                                                        "Error reading tag entry for {}: {}",
                                                        full_model_id, e
                                                    );
                                                    continue;
                                                }
                                            };

                                            if !tag_path.is_file() {
                                                debug!(
                                                    "Skipping non-file tag in {}: {:?}",
                                                    full_model_id, tag_path
                                                );
                                                continue;
                                            }

                                            let tag = match tag_path.file_name() {
                                                Some(name) => name.to_string_lossy().to_string(),
                                                None => {
                                                    debug!(
                                                        "Could not get tag name from: {:?}",
                                                        tag_path
                                                    );
                                                    continue;
                                                }
                                            };

                                            debug!(
                                                "Reading manifest for {}:{} at {:?}",
                                                full_model_id, tag, tag_path
                                            );

                                            // Read manifest to get size and digest
                                            match Self::read_manifest(&tag_path) {
                                                Ok(manifest) => {
                                                    // Calculate total size: sum of config + all layers
                                                    let mut total_size =
                                                        manifest.config.size as u64;
                                                    let layers_count =
                                                        if let Some(ref layers) = manifest.layers {
                                                            for layer in layers {
                                                                total_size += layer.size as u64;
                                                            }
                                                            layers.len()
                                                        } else {
                                                            0
                                                        };

                                                    debug!("Successfully parsed manifest for {}:{} (size: {}, layers: {}, digest: {})",
                                                        full_model_id, tag, total_size, layers_count, manifest.config.digest);

                                                    models.push(OllamaModelEntry {
                                                        id: format!("{}:{}", full_model_id, tag),
                                                        name: full_model_id.clone(),
                                                        tag: tag.clone(),
                                                        size: total_size,
                                                        digest: manifest.config.digest.clone(),
                                                        downloaded_at: chrono::Utc::now()
                                                            .to_rfc3339(),
                                                        files: vec![], // Ollama doesn't track individual files
                                                        source: "ollama".to_string(),
                                                    });
                                                }
                                                Err(e) => {
                                                    debug!(
                                                        "Failed to read manifest for {}:{}: {}",
                                                        full_model_id, tag, e
                                                    );
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        debug!("Failed to read tags for {}: {}", full_model_id, e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            debug!("Failed to read models in publisher {}: {}", publisher, e);
                        }
                    }
                }
            }
            Err(e) => {
                debug!("Failed to read registry directory: {}", e);
                return Err(anyhow!("Failed to read Ollama registry: {}", e));
            }
        }

        debug!("Total models discovered: {}", models.len());
        Ok(models)
    }

    /// Get the path to a model's manifest, in the directory it is read from.
    pub fn get_model_path(&self, name: &str, tag: &str) -> Option<PathBuf> {
        let root = self.model_root(name, tag)?;
        Self::find_manifest_in(&root, name, tag)
    }

    /// The manifest of `name:tag` in one directory: its own path, else, for a bare name, under
    /// any publisher.
    fn find_manifest_in(models_dir: &Path, name: &str, tag: &str) -> Option<PathBuf> {
        let registry_dir = models_dir.join("manifests").join("registry.ollama.ai");
        debug!(
            "get_model_path('{}:{}') - registry: {}",
            name,
            tag,
            registry_dir.display()
        );

        // Try as full path (publisher/model or just model)
        if name.contains('/') {
            // Full path like "bjoernb/claude-haiku-4-5"
            let parts: Vec<&str> = name.split('/').collect();
            let mut manifest_path = registry_dir.clone();
            for part in parts {
                manifest_path = manifest_path.join(part);
            }
            manifest_path = manifest_path.join(tag);
            debug!("  -> Checking (multi-pub): {}", manifest_path.display());
            if manifest_path.exists() {
                debug!("    ✓ FOUND");
                return Some(manifest_path);
            }
        } else {
            // Try library first
            let manifest_path = registry_dir.join("library").join(name).join(tag);
            debug!("  -> Checking (library): {}", manifest_path.display());
            if manifest_path.exists() {
                debug!("    ✓ FOUND");
                return Some(manifest_path);
            }

            // Then search all publishers
            debug!("  -> Scanning publishers: {}", registry_dir.display());
            if let Ok(entries) = std::fs::read_dir(&registry_dir) {
                for entry in entries.flatten() {
                    let pub_path = entry.path();
                    if !pub_path.is_dir() {
                        continue;
                    }
                    let pub_name = pub_path.file_name().unwrap_or_default().to_string_lossy();
                    let manifest_path = pub_path.join(name).join(tag);
                    debug!(
                        "      -> {}/{}: {}",
                        pub_name,
                        name,
                        manifest_path.display()
                    );
                    if manifest_path.exists() {
                        debug!("        ✓ FOUND in '{}'", pub_name);
                        return Some(manifest_path);
                    }
                }
                debug!("    ✗ Not found in any publisher");
            } else {
                debug!("    ✗ Cannot read registry directory!");
            }
        }

        debug!("  ✗ Model not found");
        None
    }

    /// Pull a model from Ollama registry (async HTTP download)
    pub async fn pull_model(
        &self,
        name: &str,
        tag: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<()> {
        info!("🔽 Pulling Ollama model: {}:{}", name, tag);

        // Namespaced models (e.g., "x/flux2-klein") use their namespace directly;
        // non-namespaced models (e.g., "llama3") go under "library/"
        let registry_namespace = if name.contains('/') {
            name.to_string()
        } else {
            format!("library/{}", name)
        };

        // Fetch manifest from registry
        let manifest_url = format!(
            "https://registry.ollama.ai/v2/{}/manifests/{}",
            registry_namespace, tag
        );

        debug!("Fetching manifest from: {}", manifest_url);

        let client = reqwest::Client::new();
        let resp = client
            .get(&manifest_url)
            .header(
                "Accept",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
            .send()
            .await
            .map_err(|e| anyhow!("Failed to fetch manifest: {}", e))?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Failed to fetch manifest: {} {}",
                resp.status(),
                resp.status().canonical_reason().unwrap_or("Unknown")
            ));
        }

        let manifest: OllamaManifest = resp
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse manifest: {}", e))?;

        let layers_count = manifest
            .layers
            .as_ref()
            .map(std::vec::Vec::len)
            .unwrap_or(0);
        info!("   ✓ Got manifest with {} layers", layers_count);

        // Download all blobs (config + layers)
        let mut all_blobs = vec![manifest.config.clone()];
        if let Some(ref layers) = manifest.layers {
            all_blobs.extend(layers.clone());
        }

        let total_size: u64 = all_blobs.iter().map(|b| b.size as u64).sum();
        let dest = self.destination(name, tag, &all_blobs)?;
        info!("   ✓ {}:{} goes to {}", name, tag, dest.display());
        let blobs_dir = dest.join("blobs");
        let manifest_path = Self::manifest_path_in(&dest, name, tag);
        std::fs::create_dir_all(&blobs_dir)?;
        if let Some(dir) = manifest_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut downloaded_size: u64 = 0;

        for (i, blob) in all_blobs.iter().enumerate() {
            let blob_filename = blob.digest.replace(':', "-"); // Convert sha256:xxx to sha256-xxx
            let blob_path = blobs_dir.join(&blob_filename);

            // Check if blob already exists
            if blob_path.exists() {
                info!("   ✓ Blob {} already cached", blob_filename);
                downloaded_size += blob.size as u64;
                if let Some(ref cb) = progress {
                    cb(downloaded_size, total_size);
                }
                continue;
            }

            // Download blob
            let blob_url = format!(
                "https://registry.ollama.ai/v2/{}/blobs/{}",
                registry_namespace, blob.digest
            );

            info!("   ⬇ Downloading blob {} ({} bytes)...", i + 1, blob.size);

            let mut blob_resp = client
                .get(&blob_url)
                .send()
                .await
                .map_err(|e| anyhow!("Failed to download blob: {}", e))?;

            if !blob_resp.status().is_success() {
                return Err(anyhow!(
                    "Failed to download blob {}: {}",
                    blob.digest,
                    blob_resp.status()
                ));
            }

            // Streamed to a partial file as it arrives, hashed on the way, and renamed into
            // the store only once the digest matches: a blob is never held whole in memory,
            // and an interrupted or corrupted download never lands under its final name.
            let partial_path = blobs_dir.join(format!("{blob_filename}.partial"));
            let mut file = std::fs::File::create(&partial_path)
                .map_err(|e| anyhow!("Failed to create {}: {e}", partial_path.display()))?;
            let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
            let base = downloaded_size;
            let mut received: u64 = 0;
            // One report per thousandth of the whole pull keeps a progress bar live without a
            // frame per network chunk.
            let report_step = (total_size / 1000).max(1);
            let mut reported = base;
            while let Some(chunk) = blob_resp
                .chunk()
                .await
                .map_err(|e| anyhow!("Failed to read blob {}: {e}", blob.digest))?
            {
                std::io::Write::write_all(&mut file, &chunk)
                    .map_err(|e| anyhow!("Failed to write {}: {e}", partial_path.display()))?;
                sha2::Digest::update(&mut hasher, &chunk);
                received += chunk.len() as u64;
                if base + received - reported >= report_step {
                    reported = base + received;
                    if let Some(ref cb) = progress {
                        cb(reported, total_size);
                    }
                }
            }
            std::io::Write::flush(&mut file)
                .map_err(|e| anyhow!("Failed to flush {}: {e}", partial_path.display()))?;
            drop(file);
            let digest = format!("sha256:{}", hex::encode(sha2::Digest::finalize(hasher)));
            if digest != blob.digest {
                let _ = std::fs::remove_file(&partial_path);
                return Err(anyhow!(
                    "Blob {} arrived with digest {digest}; discarded",
                    blob.digest
                ));
            }
            std::fs::rename(&partial_path, &blob_path)
                .map_err(|e| anyhow!("Failed to store {}: {e}", blob_path.display()))?;

            downloaded_size += blob.size as u64;
            if let Some(ref cb) = progress {
                cb(downloaded_size, total_size);
            }

            info!("   ✓ Downloaded {}", blob_filename);
        }

        // Write manifest to disk
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, manifest_json)?;
        self.drop_other_copies(&dest, name, tag);

        if let Some(ref cb) = progress {
            cb(total_size, total_size);
        }

        info!("   ✅ Successfully pulled {}:{}", name, tag);
        Ok(())
    }

    /// Where the manifest of `name:tag` lives inside `models_dir`: `library/<name>/<tag>` for
    /// a bare name, the name's own path for `namespace/name`.
    fn manifest_path_in(models_dir: &Path, name: &str, tag: &str) -> PathBuf {
        let mut path = models_dir.join("manifests").join("registry.ollama.ai");
        if name.contains('/') {
            for part in name.split('/') {
                path = path.join(part);
            }
        } else {
            path = path.join("library").join(name);
        }
        path.join(tag)
    }

    /// Where the manifest of `name:tag` lives: in the directory holding it, else in the one
    /// models are written to.
    pub fn manifest_path(&self, name: &str, tag: &str) -> PathBuf {
        Self::manifest_path_in(&self.root_of(name, tag), name, tag)
    }

    /// The directory a pull of `name:tag` with `blobs` lands on: the one holding its manifest
    /// or its largest blob when it has room for the rest, else the fastest-reading one with
    /// room for it all.
    fn destination(&self, name: &str, tag: &str, blobs: &[OllamaDescriptor]) -> Result<PathBuf> {
        let file = |b: &OllamaDescriptor| b.digest.replace(':', "-");
        let size: u64 = blobs.iter().map(|b| b.size as u64).sum();
        let largest = blobs.iter().max_by_key(|b| b.size).map(file);
        let facts = model_dirs::facts(&self.dirs, |dir| {
            let blobs_dir = dir.join("blobs");
            let held: u64 = blobs
                .iter()
                .filter(|&b| blobs_dir.join(file(b)).is_file())
                .map(|b| b.size as u64)
                .sum();
            let holds = Self::manifest_path_in(dir, name, tag).is_file()
                || largest
                    .as_ref()
                    .is_some_and(|f| blobs_dir.join(f).is_file());
            (held, holds)
        });
        let i = model_dirs::choose_destination(&facts, size).map_err(|e| anyhow!(e))?;
        Ok(facts[i].dir.clone())
    }

    /// Removes the manifest of `name:tag` from every directory but `keep`: the copy just
    /// written replaces it. The blobs are left to `prune_blobs`.
    fn drop_other_copies(&self, keep: &Path, name: &str, tag: &str) {
        for dir in self.dirs.iter().filter(|d| d.as_path() != keep) {
            let stale = Self::manifest_path_in(dir, name, tag);
            if !stale.is_file() {
                continue;
            }
            match std::fs::remove_file(&stale) {
                Ok(()) => info!(
                    "{name}:{tag}: the copy in {} replaced by the one in {}",
                    dir.display(),
                    keep.display()
                ),
                Err(e) => warn!("{name}:{tag}: {} not removed: {e}", stale.display()),
            }
        }
    }

    /// The blob directory models built here write to, where every layer lives under its
    /// digest: the first configured directory's.
    pub fn blobs_dir(&self) -> PathBuf {
        self.write_dir().join("blobs")
    }

    /// The blob of `digest` beside a manifest: in the `blobs` directory of the store holding
    /// the manifest.
    pub fn blob_for(manifest: &Path, digest: &str) -> Option<PathBuf> {
        let root = manifest
            .ancestors()
            .find(|p| p.file_name().is_some_and(|n| n == "manifests"))?
            .parent()?;
        Some(root.join("blobs").join(digest.replace(':', "-")))
    }

    /// Writes `content` as a blob and returns its `sha256:<hex>` digest; a blob that is
    /// already there is left as it is.
    pub fn write_blob(&self, content: &[u8]) -> Result<(String, usize)> {
        use sha2::{Digest, Sha256};
        let hex = format!("{:x}", Sha256::digest(content));
        let path = self.blobs_dir().join(format!("sha256-{hex}"));
        if !path.exists() {
            std::fs::create_dir_all(self.blobs_dir())?;
            std::fs::write(&path, content)?;
        }
        Ok((format!("sha256:{hex}"), content.len()))
    }

    /// Writes a manifest for `name:tag` from its JSON, in the first directory holding every
    /// blob it names: the one models are written to, then the others in read order. A copy
    /// of `name:tag` in another directory is replaced.
    pub fn write_manifest(
        &self,
        name: &str,
        tag: &str,
        manifest: &serde_json::Value,
    ) -> Result<()> {
        let digests = manifest_digests(manifest);
        let write_dir = self.write_dir().to_path_buf();
        let mut candidates = vec![write_dir.clone()];
        candidates.extend(self.read_dirs().into_iter().filter(|d| *d != write_dir));
        let dir = candidates
            .into_iter()
            .find(|d| {
                digests
                    .iter()
                    .all(|g| d.join("blobs").join(g.replace(':', "-")).is_file())
            })
            .unwrap_or(write_dir);
        let path = Self::manifest_path_in(&dir, name, tag);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(manifest)?)?;
        self.drop_other_copies(&dir, name, tag);
        Ok(())
    }

    /// The manifest of `name:tag` as JSON, when there is one.
    pub fn read_manifest_json(&self, name: &str, tag: &str) -> Option<serde_json::Value> {
        let path = self.manifest_path(name, tag);
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }

    /// Copies a model under another name: one manifest more in the directory holding the
    /// model, the blobs shared.
    pub fn copy_model(&self, name: &str, tag: &str, to_name: &str, to_tag: &str) -> Result<()> {
        let root = self.root_of(name, tag);
        let from = Self::manifest_path_in(&root, name, tag);
        if !from.exists() {
            return Err(anyhow!("Model not found: {}:{}", name, tag));
        }
        let to = Self::manifest_path_in(&root, to_name, to_tag);
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::copy(&from, &to)?;
        self.drop_other_copies(&root, to_name, to_tag);
        info!("Copied {}:{} to {}:{}", name, tag, to_name, to_tag);
        Ok(())
    }

    /// Removes, in every directory, each blob no manifest of that directory names any more;
    /// returns how many.
    pub fn prune_blobs(&self) -> Result<usize> {
        let mut removed = 0usize;
        for dir in &self.dirs {
            if dir.join("blobs").is_dir() {
                removed += Self::prune_blobs_in(dir)?;
            }
        }
        Ok(removed)
    }

    /// Removes every blob of `models_dir` no manifest of it names any more; returns how many.
    fn prune_blobs_in(models_dir: &Path) -> Result<usize> {
        let manifests = models_dir.join("manifests");
        let mut named: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut stack = vec![manifests];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(m) = Self::read_manifest(&path) {
                    named.insert(m.config.digest.replace(':', "-"));
                    for layer in m.layers.unwrap_or_default() {
                        named.insert(layer.digest.replace(':', "-"));
                    }
                }
            }
        }
        let mut removed = 0usize;
        for entry in std::fs::read_dir(models_dir.join("blobs"))?.flatten() {
            let path = entry.path();
            let Some(file) = path.file_name().and_then(|f| f.to_str()) else {
                continue;
            };
            if file.starts_with("sha256-") && !named.contains(file) {
                std::fs::remove_file(&path)?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Delete a model: its manifest in every directory holding one.
    pub fn delete_model(&self, name: &str, tag: &str) -> Result<()> {
        info!("Deleting Ollama model: {}:{}", name, tag);

        let mut deleted = false;
        for dir in &self.dirs {
            let manifest_path = Self::manifest_path_in(dir, name, tag);
            if !manifest_path.exists() {
                continue;
            }
            std::fs::remove_file(&manifest_path)?;
            deleted = true;

            // Clean up empty directories
            if let Some(parent) = manifest_path.parent() {
                if let Ok(entries) = std::fs::read_dir(parent) {
                    if entries.count() == 0 {
                        let _ = std::fs::remove_dir(parent);
                    }
                }
            }
            info!(
                "Successfully deleted {}:{} from {}",
                name,
                tag,
                dir.display()
            );
        }
        if deleted {
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}:{}", name, tag))
        }
    }

    /// Read and parse a manifest file
    fn read_manifest(path: &Path) -> Result<OllamaManifest> {
        let content = std::fs::read_to_string(path)?;
        debug!(
            "Manifest file read: {} bytes at {}",
            content.len(),
            path.display()
        );
        match serde_json::from_str::<OllamaManifest>(&content) {
            Ok(manifest) => {
                let layers_count = manifest
                    .layers
                    .as_ref()
                    .map(std::vec::Vec::len)
                    .unwrap_or(0);
                debug!("Successfully parsed manifest with {} layers", layers_count);
                Ok(manifest)
            }
            Err(e) => {
                debug!("JSON parse error: {}", e);
                Err(anyhow!("Failed to parse manifest JSON: {}", e))
            }
        }
    }

    /// The blob holding the model's weights, named by its manifest's model layer, in the
    /// directory holding the manifest.
    pub fn weights_blob(&self, name: &str, tag: &str) -> Option<PathBuf> {
        let root = self.root_of(name, tag);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(Self::manifest_path_in(&root, name, tag)).ok()?)
                .ok()?;
        let digest = manifest
            .get("layers")?
            .as_array()?
            .iter()
            .find(|l| {
                l.get("mediaType").and_then(|m| m.as_str())
                    == Some("application/vnd.ollama.image.model")
            })?
            .get("digest")?
            .as_str()?
            .replace(':', "-");
        Some(root.join("blobs").join(digest))
    }

    /// Bytes of the weights a load reads: the manifest's model and projector layers.
    pub fn weights_size(&self, name: &str, tag: &str) -> Option<u64> {
        let manifest = self.read_manifest_json(name, tag)?;
        let layers = manifest.get("layers")?.as_array()?;
        let bytes = layers
            .iter()
            .filter(|l| {
                l.get("mediaType")
                    .and_then(|m| m.as_str())
                    .is_some_and(|m| {
                        m == "application/vnd.ollama.image.model"
                            || m == "application/vnd.ollama.image.projector"
                    })
            })
            .filter_map(|l| l.get("size").and_then(|s| s.as_u64()))
            .sum();
        Some(bytes)
    }

    /// Get model size in bytes
    pub fn get_model_size(&self, name: &str, tag: &str) -> Result<u64> {
        if let Some(_path) = self.get_model_path(name, tag) {
            if let Ok(models) = self.list_models() {
                for model in models {
                    if model.name == name && model.tag == tag {
                        return Ok(model.size);
                    }
                }
            }
        }
        Ok(0)
    }
}

/// Every digest a manifest names: its config and its layers.
fn manifest_digests(manifest: &serde_json::Value) -> Vec<String> {
    let config = manifest
        .get("config")
        .and_then(|c| c.get("digest"))
        .and_then(|d| d.as_str());
    let layers = manifest
        .get("layers")
        .and_then(|l| l.as_array())
        .into_iter()
        .flatten()
        .filter_map(|l| l.get("digest").and_then(|d| d.as_str()));
    config
        .into_iter()
        .chain(layers)
        .map(str::to_string)
        .collect()
}

/// Ollama manifest structure (Docker V2 format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaManifest {
    #[serde(default, rename = "schemaVersion")]
    pub schema_version: Option<u32>,
    #[serde(default, rename = "mediaType")]
    pub media_type: Option<String>,
    pub config: OllamaDescriptor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<OllamaDescriptor>>,
}

/// Descriptor for a layer or config blob
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaDescriptor {
    #[serde(default, rename = "mediaType")]
    pub media_type: Option<String>,
    pub digest: String,
    pub size: usize,
}

/// Ollama model entry (list response)
#[derive(Debug, Clone)]
pub struct OllamaModelEntry {
    pub id: String, // "name:tag" format
    pub name: String,
    pub tag: String,
    pub size: u64,
    pub digest: String,
    pub downloaded_at: String,
    pub files: Vec<String>,
    pub source: String, // "ollama"
}

/// Ollama model metadata (for compatibility)
pub struct OllamaModelMetadata {
    pub name: String,
    pub size_in_gb: f32,
    pub description: String,
}

#[derive(Debug)]
pub enum OllamaError {
    ModelNotFound(String),
    InvalidModelPath(PathBuf),
    IoError(std::io::Error),
}

impl From<std::io::Error> for OllamaError {
    fn from(err: std::io::Error) -> Self {
        OllamaError::IoError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weights_are_the_model_and_projector_layers() {
        let dir = std::env::temp_dir().join(format!("loken_weights_size_{}", std::process::id()));
        let m = OllamaManager::new(dir.clone());
        let path = m.manifest_path("tiny", "1b");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            r#"{"layers":[
                {"mediaType":"application/vnd.ollama.image.model","digest":"a","size":1000},
                {"mediaType":"application/vnd.ollama.image.projector","digest":"b","size":200},
                {"mediaType":"application/vnd.ollama.image.params","digest":"c","size":30}
            ]}"#,
        )
        .unwrap();
        assert_eq!(m.weights_size("tiny", "1b"), Some(1200));
        assert_eq!(m.weights_size("absent", "1b"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Writes a manifest of `name:tag` in `dir` whose config is `config` and whose model layer
    /// is a blob of `weights` bytes, with both blobs.
    fn put_model(dir: &Path, name: &str, tag: &str, config: &str, weights: &str) {
        let blobs = dir.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join(format!("sha256-{config}")), b"{}").unwrap();
        std::fs::write(blobs.join(format!("sha256-{weights}")), b"weights").unwrap();
        let path = OllamaManager::manifest_path_in(dir, name, tag);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            format!(
                r#"{{"config":{{"digest":"sha256:{config}","size":2}},
                    "layers":[{{"mediaType":"application/vnd.ollama.image.model",
                                "digest":"sha256:{weights}","size":7}}]}}"#
            ),
        )
        .unwrap();
    }

    /// Two directories form one catalogue: each model is listed once, a model held by both is
    /// read from the first listed when their disks read as fast, and its blobs from the
    /// directory holding its manifest.
    #[test]
    fn two_directories_form_one_catalogue() {
        let scratch = super::model_dirs::tests::scratch;
        let (a, b) = (scratch("ollama-a"), scratch("ollama-b"));
        put_model(&a, "only-a", "1b", "ca", "wa");
        put_model(&b, "only-b", "1b", "cb", "wb");
        put_model(&a, "both", "1b", "c1", "w1");
        put_model(&b, "both", "1b", "c2", "w2");
        let m = OllamaManager::with_dirs(vec![a.clone(), b.clone()]);
        let mut listed: Vec<(String, String)> = m
            .list_models()
            .unwrap()
            .into_iter()
            .map(|e| (e.id, e.digest))
            .collect();
        listed.sort();
        assert_eq!(
            listed,
            [
                ("both:1b".to_string(), "sha256:c1".to_string()),
                ("only-a:1b".to_string(), "sha256:ca".to_string()),
                ("only-b:1b".to_string(), "sha256:cb".to_string()),
            ]
        );
        assert_eq!(m.model_root("both", "1b"), Some(a.clone()));
        assert_eq!(m.model_root("only-b", "1b"), Some(b.clone()));
        assert_eq!(m.model_root("absent", "1b"), None);
        assert_eq!(
            m.weights_blob("only-b", "1b"),
            Some(b.join("blobs").join("sha256-wb"))
        );
        // The other order reads the other copy.
        let reversed = OllamaManager::with_dirs(vec![b.clone(), a.clone()]);
        assert_eq!(
            reversed.weights_blob("both", "1b"),
            Some(b.join("blobs").join("sha256-w2"))
        );
        let manifest = m.get_model_path("only-b", "1b").unwrap();
        assert_eq!(
            OllamaManager::blob_for(&manifest, "sha256:wb"),
            Some(b.join("blobs").join("sha256-wb"))
        );
        // Deleting a model removes every copy.
        m.delete_model("both", "1b").unwrap();
        assert_eq!(m.model_root("both", "1b"), None);
        assert!(m.delete_model("both", "1b").is_err());
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
    }

    /// A pull lands where the model's weights already are, so that a blob two tags share stays
    /// on one disk; a model held nowhere lands on the first listed of equally fast disks.
    #[test]
    fn a_pull_lands_beside_the_blobs_it_shares() {
        let scratch = super::model_dirs::tests::scratch;
        let (a, b) = (scratch("pull-a"), scratch("pull-b"));
        put_model(&b, "base", "1b", "cfg", "shared");
        let m = OllamaManager::with_dirs(vec![a.clone(), b.clone()]);
        let blob = |digest: &str, size: usize| OllamaDescriptor {
            media_type: None,
            digest: digest.to_string(),
            size,
        };
        let new_tag = [blob("sha256:newcfg", 2), blob("sha256:shared", 7)];
        assert_eq!(m.destination("base", "alias", &new_tag).unwrap(), b);
        let fresh = [blob("sha256:x", 2), blob("sha256:y", 7)];
        assert_eq!(m.destination("other", "1b", &fresh).unwrap(), a);
        // No directory has room for a model larger than any disk.
        let huge = [blob("sha256:z", usize::MAX)];
        assert!(m.destination("huge", "1b", &huge).is_err());
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
    }

    /// A manifest is written where every blob it names is, and replaces a copy elsewhere; a
    /// copy stays beside its source's blobs.
    #[test]
    fn a_manifest_goes_where_its_blobs_are() {
        let scratch = super::model_dirs::tests::scratch;
        let (a, b) = (scratch("write-a"), scratch("write-b"));
        put_model(&b, "src", "1b", "c", "w");
        put_model(&a, "made", "1b", "old", "old-w");
        let m = OllamaManager::with_dirs(vec![a.clone(), b.clone()]);
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(OllamaManager::manifest_path_in(&b, "src", "1b")).unwrap(),
        )
        .unwrap();
        m.write_manifest("made", "1b", &manifest).unwrap();
        assert!(OllamaManager::manifest_path_in(&b, "made", "1b").is_file());
        assert!(!OllamaManager::manifest_path_in(&a, "made", "1b").exists());
        m.copy_model("src", "1b", "copy", "1b").unwrap();
        assert!(OllamaManager::manifest_path_in(&b, "copy", "1b").is_file());
        // Blobs built here go to the first directory.
        assert_eq!(m.blobs_dir(), a.join("blobs"));
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
    }
}
