use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info};

/// Progress callback for download tracking (bytes_downloaded, total_bytes)
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// HuggingFace Manager for native HF cache directory structure
pub struct HuggingFaceManager {
    models_dir: PathBuf,
    hf_token: Option<String>,
}

/// If `dir/hub` is itself a directory, treat that as the cache root.
/// One HF API client, its cache rooted in the CONFIGURED models directory.
///
/// hf_hub's default follows the process ENVIRONMENT, so a server launched without
/// HF_HOME quietly resolves ~/.cache and whatever stale artifact lives there - a
/// truncated May conversion of T5 broke every FLUX render exactly that way while the
/// good copy's location was in the config all along. `None` keeps the environment
/// default for callers that have no configured directory yet.
pub fn hf_api(
    hf_models_dir: Option<&str>,
) -> Result<hf_hub::api::sync::Api, hf_hub::api::sync::ApiError> {
    let mut b = hf_hub::api::sync::ApiBuilder::new();
    if let Some(dir) = hf_models_dir {
        // A models directory that does not exist is not one. This argument has been handed a
        // MODEL NAME by a caller whose parameters were in the other order, and because a
        // relative path is a perfectly good path, the download created it and fetched thirteen
        // gigabytes into the source tree next to weights that were already on disk. Nothing
        // failed, so nothing was reported. Refusing here catches that for every caller at once,
        // where a per-call review would have to be right every time.
        let root = std::path::Path::new(dir);
        debug_assert!(
            root.is_dir(),
            "hf_api: {dir:?} is not a directory - a model name reached the models-directory \
             argument, and downloading would create it"
        );
        if !root.is_dir() {
            tracing::error!(
                "hf_api: models directory {dir:?} does not exist; downloads would create it. \
                 Check the caller's argument order - this takes a directory, not a model name."
            );
        }
        b = b.with_cache_dir(root.join("hub"));
    }
    if let Ok(token) = std::env::var("HF_TOKEN") {
        b = b.with_token(Some(token));
    }
    b.build()
}

/// Matches HuggingFace's `HF_HOME` convention where `$HF_HOME/hub` is the
/// per-repo cache and `$HF_HOME/xet` etc. live alongside it.
fn resolve_hub_dir(dir: PathBuf) -> PathBuf {
    let hub = dir.join("hub");
    if hub.is_dir() {
        return hub;
    }
    dir
}

/// Pick the commit SHA to expose for a model directory. HF's canonical
/// branch lives at `refs/main`; some repos (PR previews, alternative
/// branches) only have `refs/refs/pr/<n>` or other custom paths. As a
/// final fallback, when refs are missing entirely but there's exactly
/// one snapshot directory whose name is a 40-char git SHA, use that  - 
/// covers the case where the refs subtree was lost in a file move but
/// snapshots survived.
fn pick_revision(model_dir: &std::path::Path) -> Option<String> {
    let refs = model_dir.join("refs");
    // 1. Prefer refs/main
    let main = refs.join("main");
    if let Ok(s) = std::fs::read_to_string(&main) {
        return Some(s.trim().to_string());
    }
    // 2. Walk refs/ recursively for any text file containing a SHA
    if refs.is_dir() {
        let mut stack = vec![refs];
        while let Some(d) = stack.pop() {
            if let Ok(it) = std::fs::read_dir(&d) {
                for e in it.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        stack.push(p);
                    } else if let Ok(s) = std::fs::read_to_string(&p) {
                        let s = s.trim();
                        if s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit()) {
                            return Some(s.to_string());
                        }
                    }
                }
            }
        }
    }
    // 3. Fall back to inferring from snapshots/ subdir names
    let snapshots = model_dir.join("snapshots");
    if let Ok(it) = std::fs::read_dir(&snapshots) {
        for e in it.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.len() == 40 && name.chars().all(|c| c.is_ascii_hexdigit()) {
                return Some(name);
            }
        }
    }
    None
}

impl HuggingFaceManager {
    /// Create a new HuggingFace manager pointing to a models cache directory.
    ///
    /// Accepts either the canonical HF cache root (`<dir>/models--*`) or
    /// an `HF_HOME`-style parent (`<dir>/hub/models--*`). When `<dir>/hub`
    /// exists we transparently descend into it so users can set
    /// `huggingface_models_dir = "/path/to/HF_HOME"` in config.toml
    /// without having to remember to append `/hub`.
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            models_dir: resolve_hub_dir(models_dir),
            hf_token: None,
        }
    }

    /// Create a new manager with HuggingFace token for private repos
    pub fn with_token(models_dir: PathBuf, token: String) -> Self {
        Self {
            models_dir: resolve_hub_dir(models_dir),
            hf_token: Some(token),
        }
    }

    /// List all HuggingFace models in the cache directory
    /// Supports two layouts:
    /// 1. HF cache format: models--user--repo/snapshots/revision/
    /// 2. Flat local-dir format: directory with GGUF files directly (from `hf download --local-dir`)
    pub fn list_models(&self) -> Result<Vec<HFModelEntry>> {
        if !self.models_dir.exists() {
            debug!(
                "HuggingFace cache directory does not exist: {:?}",
                self.models_dir
            );
            return Ok(vec![]);
        }

        let mut models = vec![];
        let mut found_cache_format = false;

        for entry in std::fs::read_dir(&self.models_dir)? {
            let path = entry?.path();
            if !path.is_dir() {
                continue;
            }

            let dir_name = match path.file_name() {
                Some(name) => name.to_string_lossy().to_string(),
                None => continue,
            };

            // HF cache format: models--user--repo
            if !dir_name.starts_with("models--") {
                continue;
            }
            found_cache_format = true;

            let model_id = dir_name
                .strip_prefix("models--")
                .unwrap()
                .replace("--", "/");

            // Resolve commit SHA - prefer refs/main, then any 40-char
            // SHA elsewhere under refs/ (covers PR previews like t5's
            // refs/refs/pr/2), and finally the snapshots/ subdir name
            // as a last-resort fallback.
            let Some(revision) = pick_revision(&path) else {
                debug!("No commit SHA discoverable for model: {}", model_id);
                continue;
            };

            // Scan snapshots directory to get file list and size
            let snapshots_dir = path.join("snapshots").join(&revision);
            let (files, total_size) = if snapshots_dir.exists() {
                self.scan_snapshots(&snapshots_dir).unwrap_or((vec![], 0))
            } else {
                (vec![], 0)
            };

            models.push(HFModelEntry {
                id: model_id.clone(),
                name: model_id.clone(),
                model_id,
                revision,
                size: total_size,
                files,
                downloaded_at: chrono::Utc::now().to_rfc3339(),
                source: "huggingface".to_string(),
            });
        }

        // If no models--* dirs found, scan for flat local-dir layout (GGUF files directly in dir)
        if !found_cache_format {
            models.extend(self.scan_flat_directory()?);
        }

        Ok(models)
    }

    /// Scan a flat directory for GGUF model files (from `hf download --local-dir`)
    /// Each GGUF file becomes a separate model entry
    fn scan_flat_directory(&self) -> Result<Vec<HFModelEntry>> {
        let mut models = vec![];

        // Try to extract repo name from .cache metadata or README
        let repo_name = self.detect_repo_name().unwrap_or_default();

        for entry in std::fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let filename = match path.file_name() {
                Some(name) => name.to_string_lossy().to_string(),
                None => continue,
            };

            if !filename.to_lowercase().ends_with(".gguf") {
                continue;
            }

            let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

            let modified_at = std::fs::metadata(&path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| {
                    chrono::DateTime::<chrono::Utc>::from(std::time::UNIX_EPOCH + d).to_rfc3339()
                })
                .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

            // Model name: strip .gguf extension
            let stem = filename.strip_suffix(".gguf").unwrap_or(&filename);
            let display_name = if repo_name.is_empty() {
                stem.to_string()
            } else {
                format!("{}/{}", repo_name, stem)
            };

            // Use filename (without extension) as the model ID for loading
            models.push(HFModelEntry {
                id: display_name.clone(),
                name: display_name,
                model_id: filename.clone(),
                revision: String::new(),
                size: file_size,
                files: vec![filename],
                downloaded_at: modified_at,
                source: "huggingface".to_string(),
            });
        }

        if !models.is_empty() {
            info!(
                "Found {} GGUF models in flat HF directory {:?}",
                models.len(),
                self.models_dir
            );
        }

        Ok(models)
    }

    /// Try to detect the HuggingFace repo name from .cache metadata or README
    fn detect_repo_name(&self) -> Option<String> {
        // Check .cache/huggingface/download/*.metadata for commit hashes
        // The README.md often has the repo info in YAML frontmatter
        let readme = self.models_dir.join("README.md");
        if readme.exists() {
            if let Ok(content) = std::fs::read_to_string(&readme) {
                // Look for "repo_url:" or "base_model:" or the HF URL pattern
                for line in content.lines().take(30) {
                    // Pattern: "  - unsloth/FLUX.1-schnell-GGUF" or similar
                    if let Some(repo) = line.trim().strip_prefix("- ") {
                        let repo = repo.trim();
                        if repo.contains('/') && !repo.contains(' ') && !repo.starts_with("http") {
                            return Some(repo.to_string());
                        }
                    }
                }
            }
        }
        None
    }

    /// Get the path to a model's files
    /// Supports:
    /// 1. HF cache format: returns snapshots/revision/ directory
    /// 2. Flat local-dir: returns path to the GGUF file directly
    /// 3. model_id can be a filename like "flux1-schnell-Q4_K_S.gguf"
    pub fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
        // Try HF cache format first: models--user--repo/snapshots/revision/
        let dir_name = format!("models--{}", model_id.replace('/', "--"));
        let model_dir = self.models_dir.join(&dir_name);

        let refs_main = model_dir.join("refs").join("main");
        if refs_main.exists() {
            if let Ok(revision) = std::fs::read_to_string(&refs_main) {
                let revision = revision.trim();
                let snapshot_dir = model_dir.join("snapshots").join(revision);
                if snapshot_dir.exists() {
                    return Some(snapshot_dir);
                }
            }
        }

        // Try flat layout: model_id is a GGUF filename
        let gguf_path = if model_id.ends_with(".gguf") {
            self.models_dir.join(model_id)
        } else {
            self.models_dir.join(format!("{}.gguf", model_id))
        };
        if gguf_path.exists() {
            return Some(gguf_path);
        }

        // Try matching by stem (e.g. "repo/stem" -> "stem.gguf")
        if model_id.contains('/') {
            if let Some(stem) = model_id.rsplit('/').next() {
                let gguf_path = self.models_dir.join(format!("{}.gguf", stem));
                if gguf_path.exists() {
                    return Some(gguf_path);
                }
            }
        }

        None
    }

    /// Pull a model from HuggingFace (async HTTP download)
    pub async fn pull_model(
        &self,
        model_id: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<()> {
        info!("🔽 Pulling HuggingFace model: {}", model_id);

        // Validate model_id format
        if !model_id.contains('/') {
            return Err(anyhow!(
                "Invalid HuggingFace model ID: {} (must be user/repo format)",
                model_id
            ));
        }

        let dir_name = format!("models--{}", model_id.replace('/', "--"));
        let model_dir = self.models_dir.join(&dir_name);

        // Ensure directories exist
        std::fs::create_dir_all(&model_dir)?;
        std::fs::create_dir_all(model_dir.join("blobs"))?;
        std::fs::create_dir_all(model_dir.join("refs"))?;
        std::fs::create_dir_all(model_dir.join("snapshots"))?;

        let client = reqwest::Client::new();

        // Fetch model info from HuggingFace API
        let api_url = format!("https://huggingface.co/api/models/{}", model_id);
        info!("   📋 Fetching model info from {}", api_url);

        let mut req = client.get(&api_url);

        // Add authentication if token is available
        if let Some(token) = &self.hf_token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| anyhow!("Failed to fetch model info: {}", e))?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Failed to fetch model info: {} (check model ID and permissions)",
                resp.status()
            ));
        }

        #[derive(Deserialize)]
        struct HFModelInfo {
            #[serde(default)]
            sha: Option<String>,
            #[serde(default)]
            siblings: Vec<HFSibling>,
        }

        #[derive(Deserialize)]
        struct HFSibling {
            rfilename: String,
            #[serde(default)]
            size: Option<u64>,
        }

        let model_info: HFModelInfo = resp
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse model info: {}", e))?;

        let revision = model_info.sha.unwrap_or_else(|| "main".to_string());
        info!("   ✓ Model revision: {}", revision);

        // Filter files to download (skip unnecessary files)
        let files_to_download: Vec<&HFSibling> = model_info
            .siblings
            .iter()
            .filter(|f| {
                let name = f.rfilename.to_lowercase();
                // Download model files and configs
                name.ends_with(".safetensors")
                    || name.ends_with(".gguf")
                    || name.ends_with(".bin")
                    || name.ends_with("config.json")
                    || name.contains("tokenizer")
                    || name.ends_with(".json") && !name.ends_with(".gitattributes")
            })
            .collect();

        let total_size: u64 = files_to_download.iter().filter_map(|f| f.size).sum();

        info!(
            "   ✓ Will download {} files (~{} MB)",
            files_to_download.len(),
            total_size / (1024 * 1024)
        );

        let snapshots_dir = model_dir.join("snapshots").join(&revision);
        std::fs::create_dir_all(&snapshots_dir)?;

        let mut downloaded_size: u64 = 0;

        // Download each file
        for (i, file) in files_to_download.iter().enumerate() {
            let file_url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                model_id, revision, file.rfilename
            );

            info!("   ⬇ Downloading file {} ({})...", i + 1, file.rfilename);

            let mut req = client.get(&file_url);
            if let Some(token) = &self.hf_token {
                req = req.header("Authorization", format!("Bearer {}", token));
            }

            let file_resp = req
                .send()
                .await
                .map_err(|e| anyhow!("Failed to download file {}: {}", file.rfilename, e))?;

            if !file_resp.status().is_success() {
                return Err(anyhow!(
                    "Failed to download {}: {}",
                    file.rfilename,
                    file_resp.status()
                ));
            }

            // Download file bytes
            let bytes = file_resp
                .bytes()
                .await
                .map_err(|e| anyhow!("Failed to read file {}: {}", file.rfilename, e))?;

            // Create file in snapshots directory
            let file_path = snapshots_dir.join(&file.rfilename);

            // Create parent directories if needed
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            std::fs::write(&file_path, bytes.as_ref())
                .map_err(|e| anyhow!("Failed to write file {}: {}", file.rfilename, e))?;

            let file_size = bytes.len() as u64;
            downloaded_size += file_size;

            // Report progress
            if let Some(ref cb) = progress {
                cb(downloaded_size, total_size);
            }

            info!("   ✓ Downloaded {} ({} bytes)", file.rfilename, file_size);
        }

        // Write refs/main with revision hash
        let refs_main = model_dir.join("refs").join("main");
        std::fs::write(&refs_main, &revision)?;

        if let Some(ref cb) = progress {
            cb(total_size, total_size);
        }

        info!(
            "   ✅ Successfully pulled {} (revision: {})",
            model_id, revision
        );
        Ok(())
    }

    /// Delete a model
    pub fn delete_model(&self, model_id: &str) -> Result<()> {
        info!("Deleting HuggingFace model: {}", model_id);

        let dir_name = format!("models--{}", model_id.replace('/', "--"));
        let model_dir = self.models_dir.join(&dir_name);

        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir)?;
            info!("Successfully deleted {}", model_id);
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", model_id))
        }
    }

    /// Scan a snapshots directory to get files and total size
    fn scan_snapshots(&self, snapshots_dir: &Path) -> Result<(Vec<String>, u64)> {
        // Walk the snapshot tree recursively. Models that store weights
        // under subdirectories - Z-Image-Turbo (transformer/, text_encoder/,
        // vae/, tokenizer/, scheduler/), the unsloth GGUF repo (BF16/), and
        // anything else following diffusers' subfolder layout - would
        // otherwise report a phantom size of 0 MB because the top-level
        // listing only sees the subdir entries and skips them.
        //
        // Filenames returned include their snapshot-relative path
        // ('transformer/diffusion_pytorch_model-00001-of-00003.safetensors')
        // so the GUI / API consumers can identify components, not just
        // bare basenames that collide across subdirs.
        let mut files = vec![];
        let mut total_size = 0u64;

        fn walk(
            dir: &std::path::Path,
            rel_prefix: &str,
            files: &mut Vec<String>,
            total_size: &mut u64,
        ) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name_str = name.to_string_lossy().to_string();
                let rel = if rel_prefix.is_empty() {
                    name_str
                } else {
                    format!("{rel_prefix}/{name_str}")
                };
                let Ok(metadata) = std::fs::metadata(&path) else {
                    continue;
                };
                if metadata.is_dir() {
                    walk(&path, &rel, files, total_size);
                } else if metadata.is_file() {
                    *total_size += metadata.len();
                    files.push(rel);
                }
            }
        }

        walk(snapshots_dir, "", &mut files, &mut total_size);
        Ok((files, total_size))
    }

    /// Get model size in bytes
    pub fn get_model_size(&self, model_id: &str) -> Result<u64> {
        if let Some(_path) = self.get_model_path(model_id) {
            if let Ok(models) = self.list_models() {
                for model in models {
                    if model.model_id == model_id {
                        return Ok(model.size);
                    }
                }
            }
        }
        Ok(0)
    }
}

/// HuggingFace model entry (list response)
#[derive(Debug, Clone)]
pub struct HFModelEntry {
    pub id: String,   // Same as model_id
    pub name: String, // Display name
    pub model_id: String,
    pub revision: String,
    pub size: u64,
    pub files: Vec<String>,
    pub downloaded_at: String,
    pub source: String, // "huggingface"
}

/// HuggingFace model metadata (for compatibility)
pub struct HuggingFaceModelMetadata {
    pub name: String,
    pub size_in_gb: f32,
    pub description: String,
}

#[derive(Debug)]
pub enum HuggingFaceError {
    ModelNotFound(String),
    InvalidModelPath(PathBuf),
    IoError(std::io::Error),
}

impl From<std::io::Error> for HuggingFaceError {
    fn from(err: std::io::Error) -> Self {
        HuggingFaceError::IoError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Minimal tempdir: under /tmp/loken-hf-tests/<pid>-<n>/, cleaned on drop.
    struct Tmp(PathBuf);
    impl Tmp {
        fn new() -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join("loken-hf-tests")
                .join(format!("{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            Tmp(p)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write(p: &std::path::Path, s: &str) {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, s).unwrap();
    }

    #[test]
    fn resolve_hub_dir_descends_into_hub_subdir() {
        // HF_HOME-style layout: <root>/hub/ exists alongside <root>/xet/.
        let tmp = Tmp::new();
        let root = tmp.path().to_path_buf();
        fs::create_dir_all(root.join("hub")).unwrap();
        fs::create_dir_all(root.join("xet")).unwrap();
        assert_eq!(resolve_hub_dir(root.clone()), root.join("hub"));
        // When already pointing at the hub, no further descent.
        let hub_direct = root.join("hub");
        assert_eq!(resolve_hub_dir(hub_direct.clone()), hub_direct);
    }

    #[test]
    fn pick_revision_prefers_refs_main() {
        let tmp = Tmp::new();
        let model = tmp.path().join("models--foo--bar");
        write(
            &model.join("refs").join("main"),
            "0123456789abcdef0123456789abcdef01234567\n",
        );
        assert_eq!(
            pick_revision(&model).as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567"),
        );
    }

    #[test]
    fn pick_revision_falls_back_to_nested_pr_ref() {
        // Mirrors google/t5-v1_1-xxl which uses refs/refs/pr/2 instead
        // of refs/main - must still surface that SHA.
        let tmp = Tmp::new();
        let model = tmp.path().join("models--google--t5-v1_1-xxl");
        write(
            &model.join("refs").join("refs").join("pr").join("2"),
            "3db68a3ef122daf6e605701de53f766d671c19aa",
        );
        assert_eq!(
            pick_revision(&model).as_deref(),
            Some("3db68a3ef122daf6e605701de53f766d671c19aa"),
        );
    }

    #[test]
    fn pick_revision_falls_back_to_snapshot_dir_name() {
        // refs/ tree entirely missing - infer from the snapshot dir.
        let tmp = Tmp::new();
        let model = tmp.path().join("models--example--model");
        let sha = "abcdef0123456789abcdef0123456789abcdef01";
        fs::create_dir_all(model.join("snapshots").join(sha)).unwrap();
        assert_eq!(pick_revision(&model).as_deref(), Some(sha));
    }

    #[test]
    fn pick_revision_none_when_nothing_present() {
        let tmp = Tmp::new();
        let model = tmp.path().join("models--nothing");
        fs::create_dir_all(&model).unwrap();
        assert!(pick_revision(&model).is_none());
    }

    #[test]
    fn scan_snapshots_recurses_into_subdirectories() {
        // Z-Image-style nested layout: weights live under transformer/,
        // text_encoder/, etc. A non-recursive walk reports 0 bytes and
        // hides the real model size from /api/tags.
        let tmp = Tmp::new();
        let snap = tmp.path().join("snapshots").join("abcdef");
        write(&snap.join("model_index.json"), "{}");
        write(
            &snap.join("transformer").join("model-00001.safetensors"),
            &"x".repeat(100),
        );
        write(
            &snap.join("transformer").join("model-00002.safetensors"),
            &"x".repeat(200),
        );
        write(
            &snap.join("text_encoder").join("model.safetensors"),
            &"y".repeat(50),
        );
        write(&snap.join("vae").join("config.json"), "{}");

        let mgr = HuggingFaceManager::new(tmp.path().to_path_buf());
        let (files, total) = mgr.scan_snapshots(&snap).unwrap();

        // 5 files total: model_index.json + 2 transformer + text_encoder + vae/config
        assert_eq!(files.len(), 5);
        assert_eq!(total, 100 + 200 + 50 + 2 + 2);

        // Filenames carry the snapshot-relative subpath so callers can
        // distinguish 'transformer/model-00001.safetensors' from
        // 'text_encoder/model.safetensors'.
        assert!(files
            .iter()
            .any(|f| f == "transformer/model-00001.safetensors"));
        assert!(files.iter().any(|f| f == "text_encoder/model.safetensors"));
        assert!(files.iter().any(|f| f == "vae/config.json"));
        assert!(files.iter().any(|f| f == "model_index.json"));
    }

    #[test]
    fn list_models_handles_hf_home_layout_with_pr_ref() {
        let tmp = Tmp::new();
        let hub = tmp.path().join("hub");
        let model = hub.join("models--google--t5-v1_1-xxl");
        let sha = "3db68a3ef122daf6e605701de53f766d671c19aa";
        write(&model.join("refs").join("refs").join("pr").join("2"), sha);
        write(&model.join("snapshots").join(sha).join("config.json"), "{}");

        // Pass HF_HOME - the manager should descend into hub/ on its own.
        let mgr = HuggingFaceManager::new(tmp.path().to_path_buf());
        let models = mgr.list_models().expect("list should succeed");
        assert_eq!(models.len(), 1, "should find the t5 model");
        assert_eq!(models[0].model_id, "google/t5-v1_1-xxl");
        assert_eq!(models[0].revision, sha);
    }
}
