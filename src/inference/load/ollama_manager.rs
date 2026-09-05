use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info};

/// Progress callback for download tracking (bytes_downloaded, total_bytes)
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Ollama Manager for native Ollama model directory structure
pub struct OllamaManager {
    models_dir: PathBuf,
}

impl OllamaManager {
    /// Create a new Ollama manager pointing to a models directory
    pub fn new(models_dir: PathBuf) -> Self {
        Self { models_dir }
    }

    /// List all Ollama models in the directory
    pub fn list_models(&self) -> Result<Vec<OllamaModelEntry>> {
        let registry_dir = self.models_dir.join("manifests").join("registry.ollama.ai");

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
                                            match self.read_manifest(&tag_path) {
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

    /// Get the path to a model's manifest
    pub fn get_model_path(&self, name: &str, tag: &str) -> Option<PathBuf> {
        let registry_dir = self.models_dir.join("manifests").join("registry.ollama.ai");
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

        // Ensure directories exist
        let blobs_dir = self.models_dir.join("blobs");
        // Namespaced models (e.g., "x/flux2-klein") use their namespace directly;
        // non-namespaced models (e.g., "llama3") go under "library/"
        let (registry_namespace, manifests_dir) = if name.contains('/') {
            // e.g., "x/flux2-klein" -> namespace "x", model "flux2-klein"
            let parts: Vec<&str> = name.splitn(2, '/').collect();
            let ns = format!("{}/{}", parts[0], parts[1]);
            let dir = self
                .models_dir
                .join("manifests")
                .join("registry.ollama.ai")
                .join(parts[0])
                .join(parts[1]);
            (ns, dir)
        } else {
            let ns = format!("library/{}", name);
            let dir = self
                .models_dir
                .join("manifests")
                .join("registry.ollama.ai")
                .join("library")
                .join(name);
            (ns, dir)
        };

        std::fs::create_dir_all(&blobs_dir)?;
        std::fs::create_dir_all(&manifests_dir)?;

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

            let blob_resp = client
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

            // Download entire blob
            let bytes = blob_resp
                .bytes()
                .await
                .map_err(|e| anyhow!("Failed to read blob: {}", e))?;

            // Write to file
            std::fs::write(&blob_path, bytes.as_ref())
                .map_err(|e| anyhow!("Failed to write blob file: {}", e))?;

            downloaded_size += blob.size as u64;

            // Report progress
            if let Some(ref cb) = progress {
                cb(downloaded_size, total_size);
            }

            info!("   ✓ Downloaded {}", blob_filename);
        }

        // Write manifest to disk
        let manifest_path = manifests_dir.join(tag);
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, manifest_json)?;

        if let Some(ref cb) = progress {
            cb(total_size, total_size);
        }

        info!("   ✅ Successfully pulled {}:{}", name, tag);
        Ok(())
    }

    /// Where the manifest of `name:tag` lives: `library/<name>/<tag>` for a bare name,
    /// the name's own path for `namespace/name`.
    pub fn manifest_path(&self, name: &str, tag: &str) -> PathBuf {
        let mut path = self
            .models_dir
            .join("manifests")
            .join("registry.ollama.ai");
        if name.contains('/') {
            for part in name.split('/') {
                path = path.join(part);
            }
        } else {
            path = path.join("library").join(name);
        }
        path.join(tag)
    }

    /// The blob directory, where every layer lives under its digest.
    pub fn blobs_dir(&self) -> PathBuf {
        self.models_dir.join("blobs")
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

    /// Writes a manifest for `name:tag` from its JSON.
    pub fn write_manifest(&self, name: &str, tag: &str, manifest: &serde_json::Value) -> Result<()> {
        let path = self.manifest_path(name, tag);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(manifest)?)?;
        Ok(())
    }

    /// The manifest of `name:tag` as JSON, when there is one.
    pub fn read_manifest_json(&self, name: &str, tag: &str) -> Option<serde_json::Value> {
        let path = self.manifest_path(name, tag);
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }

    /// Copies a model under another name: one manifest more, the blobs shared.
    pub fn copy_model(&self, name: &str, tag: &str, to_name: &str, to_tag: &str) -> Result<()> {
        let from = self.manifest_path(name, tag);
        if !from.exists() {
            return Err(anyhow!("Model not found: {}:{}", name, tag));
        }
        let to = self.manifest_path(to_name, to_tag);
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::copy(&from, &to)?;
        info!("Copied {}:{} to {}:{}", name, tag, to_name, to_tag);
        Ok(())
    }

    /// Removes every blob no manifest names any more; returns how many.
    pub fn prune_blobs(&self) -> Result<usize> {
        let manifests = self.models_dir.join("manifests");
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
                } else if let Ok(m) = self.read_manifest(&path) {
                    named.insert(m.config.digest.replace(':', "-"));
                    for layer in m.layers.unwrap_or_default() {
                        named.insert(layer.digest.replace(':', "-"));
                    }
                }
            }
        }
        let mut removed = 0usize;
        for entry in std::fs::read_dir(self.models_dir.join("blobs"))?.flatten() {
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

    /// Delete a model
    pub fn delete_model(&self, name: &str, tag: &str) -> Result<()> {
        info!("Deleting Ollama model: {}:{}", name, tag);

        let manifest_path = self.manifest_path(name, tag);

        if manifest_path.exists() {
            std::fs::remove_file(&manifest_path)?;

            // Clean up empty directories
            if let Ok(entries) = std::fs::read_dir(manifest_path.parent().unwrap()) {
                if entries.count() == 0 {
                    let _ = std::fs::remove_dir(manifest_path.parent().unwrap());
                }
            }

            info!("Successfully deleted {}:{}", name, tag);
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}:{}", name, tag))
        }
    }

    /// Read and parse a manifest file
    fn read_manifest(&self, path: &PathBuf) -> Result<OllamaManifest> {
        let content = std::fs::read_to_string(path)?;
        debug!(
            "Manifest file content (first 200 chars): {}",
            &content.chars().take(200).collect::<String>()
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
