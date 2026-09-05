//! Model manager for downloading and loading models
//!
//! This module provides functionality to pull models from multiple sources:
//! - HuggingFace Hub (SafeTensors format)
//! - Ollama Registry (GGUF format)
//! - Direct URLs
//!
//! Storage approach:
//! - Models stored directly in ollama_models_dir or huggingface_models_dir based on source
//! - No manifest or hash-based storage system
//! - Direct directory-based resolution

use crate::inference::load::huggingface_manager::HuggingFaceManager;
use crate::inference::load::ollama_manager::OllamaManager;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// Progress callback for download tracking (bytes_downloaded, total_bytes)
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Model metadata stored in each model directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Original model ID (e.g., "llama3:latest" or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    pub id: String,
    /// Short name for display
    pub name: String,
    /// Size in bytes
    pub size: u64,
    /// Download timestamp
    pub downloaded_at: String,
    /// List of files in the model directory
    pub files: Vec<String>,
    /// Source: "huggingface", "ollama", or "direct"
    pub source: String,
    /// SHA256 digest of the model file (for Ollama compatibility)
    #[serde(default)]
    pub digest: String,
}

/// Model validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationResult {
    /// Whether the model exists
    pub exists: bool,
    /// Whether a GGUF file was found
    pub has_gguf: bool,
    /// Whether a SafeTensors file was found
    pub has_safetensors: bool,
    /// Total size in bytes
    pub total_size: u64,
    /// List of files found
    pub files: Vec<String>,
    /// Whether the model is valid (has model weights)
    pub is_valid: bool,
    /// Human-readable message
    pub message: String,
}

/// Model source type
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    HuggingFace,
    Ollama,
    DirectUrl,
    Local,
}

/// Result of computing model digest and metadata
#[derive(Debug, Clone)]
struct ModelDigestInfo {
    digest: String,
    modified_at: String,
}

/// Get digest and modification time from Ollama manifest file
fn get_manifest_digest_info(manifest_path: &PathBuf) -> Option<ModelDigestInfo> {
    // Read manifest file (JSON)
    if let Ok(content) = std::fs::read_to_string(manifest_path) {
        if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
            // Extract config digest
            if let Some(config_digest) = manifest
                .get("config")
                .and_then(|c| c.get("digest"))
                .and_then(|d| d.as_str())
            {
                // Get manifest file modification time
                let modified_at = std::fs::metadata(manifest_path)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        chrono::DateTime::<chrono::Utc>::from(std::time::UNIX_EPOCH + d)
                            .to_rfc3339()
                    })
                    .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

                return Some(ModelDigestInfo {
                    digest: config_digest.to_string(),
                    modified_at,
                });
            }
        }
    }
    None
}

/// Compute SHA256 digest and get modification time of a model file
/// Returns digest in "sha256:xxxxxxxx" format and ISO8601 modified time
fn compute_model_digest_info(model_dir: &PathBuf) -> ModelDigestInfo {
    // For Ollama models (manifest path), try to read digest from manifest JSON
    if model_dir.ends_with("manifests") || model_dir.to_string_lossy().contains("manifests") {
        if let Some(digest_info) = get_manifest_digest_info(model_dir) {
            return digest_info;
        }
    }

    // Fallback: use manifest file mtime
    let modified_at = std::fs::metadata(model_dir)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| chrono::DateTime::<chrono::Utc>::from(std::time::UNIX_EPOCH + d).to_rfc3339())
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

    ModelDigestInfo {
        digest: "sha256:0000000000000000000000000000000000000000".to_string(),
        modified_at,
    }
}

/// Model manager for downloading and managing models
/// Supports Ollama and HuggingFace sources only
pub struct ModelManager {
    ollama_manager: OllamaManager,
    huggingface_manager: HuggingFaceManager,
}

impl ModelManager {
    /// Create a new model manager with separate directories for different sources
    pub fn new(
        ollama_models_dir: impl Into<PathBuf>,
        huggingface_models_dir: impl Into<PathBuf>,
    ) -> Self {
        let ollama_dir = ollama_models_dir.into();
        let hf_dir = huggingface_models_dir.into();

        let ollama_manager = OllamaManager::new(ollama_dir);
        let huggingface_manager = HuggingFaceManager::new(hf_dir);

        Self {
            ollama_manager,
            huggingface_manager,
        }
    }

    /// Get model path for Ollama model
    /// Note: Use source-specific manager methods for explicit control
    /// Only Ollama models are supported in this method
    pub fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
        // This method requires knowing the source - prefer using managers directly
        // Only supports Ollama models
        self.ollama_manager.get_model_path(model_id, "latest")
    }

    /// Resolve ID to storage path - direct directory resolution without manifest
    /// Note: Source must be determined before calling this
    /// Only supports "ollama" and "huggingface" sources
    pub async fn resolve_path(&self, model_id: &str, source: &str) -> Result<PathBuf> {
        match source {
            "ollama" => {
                // Parse as name:tag
                let parts: Vec<&str> = model_id.split(':').collect();
                let name = parts.first().copied().unwrap_or(model_id);
                let tag = parts.get(1).copied().unwrap_or("latest");
                let path = self
                    .ollama_manager
                    .get_model_path(name, tag)
                    .ok_or_else(|| anyhow::anyhow!("Ollama model not found: {}", model_id))?;
                Ok(path)
            }
            "huggingface" => {
                let path = self
                    .huggingface_manager
                    .get_model_path(model_id)
                    .ok_or_else(|| anyhow::anyhow!("HuggingFace model not found: {}", model_id))?;
                Ok(path)
            }
            other => Err(anyhow::anyhow!(
                "Unsupported model source: '{}'. Supported sources: 'ollama', 'huggingface'",
                other
            )),
        }
    }

    /// Check if model exists at the given path (without manifest)
    pub async fn is_model_downloaded(&self, model_id: &str) -> bool {
        if let Some(path) = self.get_model_path(model_id) {
            path.exists()
        } else {
            false
        }
    }

    /// List models from both Ollama and HuggingFace directories
    pub async fn list_models(&self) -> Result<Vec<ModelMetadata>> {
        let mut models = Vec::new();

        // List Ollama models
        if let Ok(ollama_models) = self.ollama_manager.list_models() {
            for model in ollama_models {
                // Compute digest and modification time from the model directory using name:tag
                let digest_info = self
                    .ollama_manager
                    .get_model_path(&model.name, &model.tag)
                    .map(|p| compute_model_digest_info(&p))
                    .unwrap_or_else(|| ModelDigestInfo {
                        digest: "sha256:0000000000000000000000000000000000000000".to_string(),
                        modified_at: chrono::Utc::now().to_rfc3339(),
                    });

                models.push(ModelMetadata {
                    id: model.id,
                    name: model.name,
                    size: model.size,
                    downloaded_at: digest_info.modified_at,
                    files: model.files,
                    source: model.source,
                    digest: digest_info.digest,
                });
            }
        }

        // List HuggingFace models
        if let Ok(hf_models) = self.huggingface_manager.list_models() {
            for model in hf_models {
                // Compute digest and modification time from the model directory using model_id
                let digest_info = self
                    .huggingface_manager
                    .get_model_path(&model.model_id)
                    .map(|p| compute_model_digest_info(&p))
                    .unwrap_or_else(|| ModelDigestInfo {
                        digest: "sha256:0000000000000000000000000000000000000000".to_string(),
                        modified_at: chrono::Utc::now().to_rfc3339(),
                    });

                models.push(ModelMetadata {
                    id: model.id,
                    name: model.name,
                    size: model.size,
                    downloaded_at: digest_info.modified_at,
                    files: model.files,
                    source: model.source,
                    digest: digest_info.digest,
                });
            }
        }

        Ok(models)
    }

    /// Validate a model - check if it has the required files
    pub async fn validate_model(&self, model_id: &str) -> Result<ModelValidationResult> {
        let path = match self.get_model_path(model_id) {
            Some(p) => p,
            None => {
                return Ok(ModelValidationResult {
                    exists: false,
                    has_gguf: false,
                    has_safetensors: false,
                    total_size: 0,
                    files: vec![],
                    is_valid: false,
                    message: "Model directory does not exist".to_string(),
                });
            }
        };

        if !path.exists() {
            return Ok(ModelValidationResult {
                exists: false,
                has_gguf: false,
                has_safetensors: false,
                total_size: 0,
                files: vec![],
                is_valid: false,
                message: "Model directory does not exist".to_string(),
            });
        }

        // List files and check for model files
        let mut files = Vec::new();
        let mut total_size = 0u64;
        let mut has_gguf = false;
        let mut has_safetensors = false;

        if let Ok(entries) = std::fs::read_dir(&path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name == "metadata.json" {
                        continue;
                    }
                    if let Ok(meta) = std::fs::metadata(&path) {
                        total_size += meta.len();
                        files.push(format!("{} ({} bytes)", name, meta.len()));

                        // A weights file is megabytes at least; anything smaller with
                        // the right extension is a placeholder, a lock file or a
                        // truncated download, and counting it as a model makes the
                        // directory look loadable when it is not.
                        // not-a-vram-size: a floor on a FILE's size, not memory reserved.
                        const PLAUSIBLE_WEIGHTS_FILE: u64 = 1_000_000;
                        if name.ends_with(".gguf") && meta.len() > PLAUSIBLE_WEIGHTS_FILE {
                            has_gguf = true;
                        }
                        if name.ends_with(".safetensors") && meta.len() > PLAUSIBLE_WEIGHTS_FILE {
                            has_safetensors = true;
                        }
                    }
                }
            }
        }

        let is_valid = has_gguf || has_safetensors;
        let message = if is_valid {
            format!("Model is valid ({} bytes total)", total_size)
        } else {
            format!(
                "Model is broken - no valid model files found (only {} bytes)",
                total_size
            )
        };

        Ok(ModelValidationResult {
            exists: true,
            has_gguf,
            has_safetensors,
            total_size,
            files,
            is_valid,
            message,
        })
    }

    /// Pull a model with explicit source
    pub async fn pull_model_with_source(
        &self,
        model_id: &str,
        source: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<ModelMetadata> {
        info!("Pulling {} model: {}", source, model_id);

        match source {
            "ollama" => {
                // Parse ollama model_id as "name:tag"
                let parts: Vec<&str> = model_id.split(':').collect();
                let name = parts.first().copied().unwrap_or(model_id);
                let tag = parts.get(1).copied().unwrap_or("latest");

                self.ollama_manager.pull_model(name, tag, progress).await?;

                // Compute digest and modification time from the newly downloaded model
                let digest_info = self
                    .ollama_manager
                    .get_model_path(name, tag)
                    .map(|p| compute_model_digest_info(&p))
                    .unwrap_or_else(|| ModelDigestInfo {
                        digest: "sha256:0000000000000000000000000000000000000000".to_string(),
                        modified_at: chrono::Utc::now().to_rfc3339(),
                    });

                Ok(ModelMetadata {
                    id: model_id.to_string(),
                    name: format!("{}:{}", name, tag),
                    size: 0,
                    downloaded_at: digest_info.modified_at,
                    files: vec![],
                    source: "ollama".to_string(),
                    digest: digest_info.digest,
                })
            }
            "huggingface" => {
                self.huggingface_manager
                    .pull_model(model_id, progress)
                    .await?;

                // Compute digest and modification time from the newly downloaded model
                let digest_info = self
                    .huggingface_manager
                    .get_model_path(model_id)
                    .map(|p| compute_model_digest_info(&p))
                    .unwrap_or_else(|| ModelDigestInfo {
                        digest: "sha256:0000000000000000000000000000000000000000".to_string(),
                        modified_at: chrono::Utc::now().to_rfc3339(),
                    });

                Ok(ModelMetadata {
                    id: model_id.to_string(),
                    name: model_id.to_string(),
                    size: 0,
                    downloaded_at: digest_info.modified_at,
                    files: vec![],
                    source: "huggingface".to_string(),
                    digest: digest_info.digest,
                })
            }
            _ => {
                anyhow::bail!("Unknown source: {}", source)
            }
        }
    }

    /// Removes a model from the store: its manifest and the blobs nothing else names
    /// for an Ollama model, its directory for a Hugging Face one.
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        let (name, tag) = split_name_tag(model_id);
        match self.ollama_manager.delete_model(name, tag) {
            Ok(()) => {
                let pruned = self.ollama_manager.prune_blobs()?;
                info!("Deleted {model_id}: {pruned} blob(s) no manifest named any more removed");
                Ok(())
            }
            Err(_) => self.huggingface_manager.delete_model(model_id),
        }
    }

    /// Repair a broken model - simplified for directory-based approach
    pub async fn repair_model(&self, model_id: &str) -> Result<ModelMetadata> {
        info!("Repairing model: {}", model_id);
        // In directory-based approach, repair is manual - just return the metadata
        let digest_info = self
            .get_model_path(model_id)
            .map(|p| compute_model_digest_info(&p))
            .unwrap_or_else(|| ModelDigestInfo {
                digest: "sha256:0000000000000000000000000000000000000000".to_string(),
                modified_at: chrono::Utc::now().to_rfc3339(),
            });

        let metadata = ModelMetadata {
            id: model_id.to_string(),
            name: model_id.to_string(),
            size: 0,
            downloaded_at: digest_info.modified_at,
            files: vec![],
            source: "unknown".to_string(),
            digest: digest_info.digest,
        };
        Ok(metadata)
    }

    /// The Ollama store, for the endpoints that build models from blobs.
    pub fn ollama(&self) -> &OllamaManager {
        &self.ollama_manager
    }

    /// Copies an Ollama model under another name.
    pub async fn copy_model(&self, source: &str, destination: &str) -> Result<()> {
        let (name, tag) = split_name_tag(source);
        let (to_name, to_tag) = split_name_tag(destination);
        self.ollama_manager.copy_model(name, tag, to_name, to_tag)
    }

    /// Get model size estimate - only supports ollama and huggingface sources
    pub async fn estimate_model_size(&self, model_id: &str, source: &str) -> Result<u64> {
        match source {
            "ollama" => {
                let parts: Vec<&str> = model_id.split(':').collect();
                let name = parts.first().copied().unwrap_or(model_id);
                let tag = parts.get(1).copied().unwrap_or("latest");
                self.ollama_manager.get_model_size(name, tag)
            }
            "huggingface" => self.huggingface_manager.get_model_size(model_id),
            other => Err(anyhow::anyhow!(
                "Unsupported model source: '{}'. Supported sources: 'ollama', 'huggingface'",
                other
            )),
        }
    }

    /// Get multi-device configuration for a model - simplified for directory-based approach
    pub async fn get_multi_device_config(
        &self,
        model_id: &str,
    ) -> Result<Option<serde_json::Value>> {
        info!("Getting multi-device config for {}", model_id);
        // In directory-based approach, this is handled by the system
        Ok(None)
    }
}

// Re-export the metadata types for use in other modules
pub use crate::inference::load::huggingface_manager::HuggingFaceModelMetadata;
pub use crate::inference::load::ollama_manager::OllamaModelMetadata;


/// `name:tag`, `latest` when no tag is given.
fn split_name_tag(model_id: &str) -> (&str, &str) {
    match model_id.rsplit_once(':') {
        Some((name, tag)) if !tag.contains('/') => (name, tag),
        _ => (model_id, "latest"),
    }
}
