//! API module providing HTTP server and client functionality
//!
//! This module contains the core API server implementation that handles
//! HTTP requests for model management and chat completions.
//! Supports Ollama-compatible API format.

pub(crate) mod handlers;
mod types;

/// Optional collaborators for the media endpoints.
pub mod assist;

/// The model-resolution helpers the render CLI shares with the HTTP handlers, so that
/// both agree on which family a name is, where its checkpoint lives, WHICH LOADER puts
/// it up and what recipe it renders at. Exported deliberately narrowly - the handler
/// module itself stays private.
///
/// `image_family_loader` + `load_image_family` are here because the CLI carried its own
/// if/else copy of the loader dispatch, ending in the same fall-through to Flux: it had
/// no boogu arm and no flux2 arm, so it rendered both families with FLUX.1-schnell.
/// Sharing the dispatch is what stops a fourth copy from being written.
#[cfg(feature = "image")]
pub mod image_paths {
    pub use super::handlers::{
        image_family, image_family_loader, image_model_defaults, load_image_family,
        requested_checkpoint, ImageLoader, ImageModelDefaults,
    };
}
mod anthropic;
mod client;
pub(crate) mod thinking;
mod tool_calls;

// Production features. Internal to the crate - none of these
// types are surfaced through loken::api re-exports today, so the
// `pub` was wider than needed. Promoting back to `pub` is a one-line
// change if/when an external embedder wants the auth or metrics
// surface; in the meantime keeping them private narrows the API contract.
mod gate;
pub(crate) mod rate_limiter;

// Core types
pub use types::{
    ChatCompletionRequest, ChatCompletionResponse, Choice, ListModelsResponse, Message, ModelInfo,
    StopSequences, Usage,
};

// Streaming types
pub use types::{ChatCompletionChunk, ChunkChoice, ChunkDelta};

// Ollama types
pub use types::{
    GetModelResponse, OllamaChatRequest, OllamaChatResponse, OllamaCopyRequest,
    OllamaCreateRequest, OllamaDeleteRequest, OllamaEmbedResponse, OllamaGenerateRequest,
    OllamaGenerateResponse, OllamaListModelsResponse, OllamaModel, OllamaModelDetails,
    OllamaPullRequest, OllamaPullResponse, OllamaShowRequest, OllamaShowResponse,
};

// Advanced feature types
pub use types::{ModelSource, StructuredOutput, Tool, ToolCall, ToolCallFunction, ToolFunction};

// Common types
pub use types::{
    GenerateRequest, GenerateResponse, LayerDistribution, ListLoadedModelsResponse,
    LoadModelResponse, LoadedModelInfo,
};

// Handlers and client
pub use handlers::parse_keep_alive;
pub use handlers::APIServer;
// Humanized validate_request lives in api/handlers/ (field-by-field
// errors). The earlier api::validator module had a duplicate function
// with the same name but a less useful "Validation failed: {...}" form;
// it's been deleted along with two unused validate_role/finish_reason
// helpers that were never wired up to a #[validate(custom = ...)] site.
pub use client::{Client, ClientError};
pub use handlers::validate_request;

use tracing::info;

/// Trait for Ollama client operations
#[async_trait::async_trait]
pub trait OllamaClient: Send + Sync + 'static {
    /// Pull a model from Ollama registry
    async fn pull_model(
        &self,
        model_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Trait for model management operations
#[async_trait::async_trait]
pub trait ModelManager: Send + Sync + 'static {
    /// List available models
    async fn list_models(&self)
        -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>>;

    /// Delete a model
    async fn delete_model(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Re-export GPU types from gpu module
pub use crate::gpu::{Device as GPUDevice, GPUManagerImpl, GPUManagerInterface};

/// Implementation of OllamaClient
pub struct OllamaClientImpl {
    base_url: String,
    http_client: reqwest::Client,
}

impl OllamaClientImpl {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            http_client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl OllamaClient for OllamaClientImpl {
    async fn pull_model(
        &self,
        model_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/api/pull", self.base_url);
        let response = self
            .http_client
            .post(&url)
            .json(&serde_json::json!({ "name": model_name }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Failed to pull model, status: {}", response.status()).into());
        }

        info!("Successfully pulled model: {}", model_name);
        Ok(())
    }
}

/// Implementation of ModelManager using the real model_manager module
pub struct ModelManagerImpl {
    inner: crate::inference::load::model_manager::ModelManager,
}

impl ModelManagerImpl {
    pub fn new(ollama_models_dir: String, huggingface_models_dir: String) -> Self {
        Self {
            inner: crate::inference::load::model_manager::ModelManager::new(
                &ollama_models_dir,
                &huggingface_models_dir,
            ),
        }
    }
}

#[async_trait::async_trait]
impl ModelManager for ModelManagerImpl {
    async fn list_models(
        &self,
    ) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>> {
        let metadata_list = self.inner.list_models().await?;
        let mut models = Vec::new();
        for metadata in metadata_list {
            let size_str = format_size(metadata.size);
            models.push(ModelInfo::new(
                metadata.id.clone(),
                size_str,
                metadata.size,
                metadata.downloaded_at,
            ));
        }
        Ok(models)
    }

    async fn delete_model(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.inner.delete_model(name).await?;
        info!("Successfully deleted model: {}", name);
        Ok(())
    }
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2}TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2}KB", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ollama_client() {
        let client = OllamaClientImpl::new("https://ollama.ai".to_string());
        assert_eq!(client.base_url, "https://ollama.ai");
    }

    #[tokio::test]
    async fn test_model_manager() {
        let manager = ModelManagerImpl::new("./models".to_string(), "./hf_models".to_string());
        // Just verify the function works - may return 0 models if directories are empty
        let models = manager.list_models().await.unwrap();
        // No assertion on length - empty list is valid during development/testing
        println!("Found {} models", models.len());
    }

    #[test]
    fn format_size_picks_largest_unit_with_two_decimals() {
        // Below 1 KB: integer bytes, no unit decimals.
        assert_eq!(format_size(0), "0B");
        assert_eq!(format_size(1023), "1023B");
        // KB boundary + KB scale (2 decimals).
        assert_eq!(format_size(1024), "1.00KB");
        assert_eq!(format_size(1024 + 512), "1.50KB");
        // MB, GB, TB boundaries.
        assert_eq!(format_size(1024 * 1024), "1.00MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00GB");
        assert_eq!(format_size(1024_u64.pow(4)), "1.00TB");
        // Real-world model sizes round consistently.
        // 7_000_000_000 bytes ≈ 6.52 GB (7e9 / 2^30 = 6.5193...)
        assert_eq!(format_size(7_000_000_000), "6.52GB");
        // 1_500_000 bytes ≈ 1.43 MB
        assert_eq!(format_size(1_500_000), "1.43MB");
    }
}
