//! HTTP client for the LLM server
//!
//! This module provides a client implementation for communicating with
//! the LLM server or Ollama server using the Ollama API protocol.

use reqwest::{Client as HttpClient, Response};
use serde::de::DeserializeOwned;
use std::time::Duration;

use crate::api::types::*;

/// HTTP client for the LLM server (Ollama-compatible)
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new client
    pub fn new(base_url: String) -> Self {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to initialize HTTP client");

        Self {
            base_url,
            http_client,
        }
    }

    /// List available models (GET /api/tags)
    pub async fn list_models(&self) -> Result<ListModelsResponse, ClientError> {
        let url = format!("{}/api/tags", self.base_url);

        let response = self.http_client.get(&url).send().await?;
        let ollama_response: OllamaListModelsResponse = self.handle_response(response).await?;
        Ok(ListModelsResponse::from_ollama(ollama_response))
    }

    /// Pull a model with explicit source (POST /api/pull)
    pub async fn pull_model_with_source(
        &self,
        model_name: &str,
        source: &str,
    ) -> Result<OllamaPullResponse, ClientError> {
        let url = format!("{}/api/pull", self.base_url);

        let mut request = OllamaPullRequest::new(model_name.to_string());
        request.source = source.to_string();

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(600)) // 10 minutes for large models
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Pull a model (POST /api/pull) - defaults to auto-detection
    pub async fn pull_model(&self, model_name: &str) -> Result<OllamaPullResponse, ClientError> {
        let url = format!("{}/api/pull", self.base_url);

        let request = OllamaPullRequest::new(model_name.to_string());

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(600)) // 10 minutes for large models
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Delete a model (DELETE /api/delete)
    pub async fn delete_model(&self, model_name: &str) -> Result<(), ClientError> {
        let url = format!("{}/api/delete", self.base_url);

        let request = OllamaDeleteRequest {
            name: model_name.to_string(),
        };

        let response = self.http_client.delete(&url).json(&request).send().await?;

        // Ollama returns empty response on success, just check status
        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            Err(ClientError::Http(format!(
                "Delete failed with status {}: {}",
                status, text
            )))
        }
    }

    /// Show model info (POST /api/show)
    pub async fn show_model(&self, model_name: &str) -> Result<OllamaShowResponse, ClientError> {
        let url = format!("{}/api/show", self.base_url);

        let request = OllamaShowRequest {
            model: model_name.to_string(),
            name: String::new(),
        };

        let response = self.http_client.post(&url).json(&request).send().await?;

        self.handle_response(response).await
    }

    /// Chat completion using Ollama API (POST /api/chat)
    pub async fn chat(
        &self,
        request: &OllamaChatRequest,
    ) -> Result<OllamaChatResponse, ClientError> {
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(300)) // 5 minutes for generation
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Chat completion with streaming (returns raw response for NDJSON parsing)
    /// Each line of the response is a JSON `OllamaChatResponse` object.
    /// The last line has `done: true`.
    pub async fn chat_stream(
        &self,
        request: &OllamaChatRequest,
    ) -> Result<reqwest::Response, ClientError> {
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(1800)) // 30 minutes for streaming
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ClientError::Http(format!(
                "Request failed with status {}: {}",
                status, text
            )));
        }

        Ok(response)
    }

    /// Generate text using Ollama API (POST /api/generate)
    pub async fn generate(
        &self,
        request: &OllamaGenerateRequest,
    ) -> Result<OllamaGenerateResponse, ClientError> {
        let url = format!("{}/api/generate", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(300)) // 5 minutes for generation
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Chat completion (OpenAI-compatible endpoint, converts to/from Ollama internally)
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ClientError> {
        // Convert to Ollama format
        let ollama_request = request.clone().to_ollama();

        // Call Ollama chat endpoint
        let ollama_response = self.chat(&ollama_request).await?;

        // Convert back to OpenAI format
        Ok(ChatCompletionResponse::from_ollama(ollama_response))
    }

    /// List loaded models (custom endpoint, may not be available on all servers)
    pub async fn list_loaded_models(&self) -> Result<ListLoadedModelsResponse, ClientError> {
        // Try custom endpoint first
        let url = format!("{}/api/models/loaded", self.base_url);

        let response = self.http_client.get(&url).send().await;

        match response {
            Ok(resp) if resp.status().is_success() => self.handle_response(resp).await,
            _ => {
                // Fallback: return empty list if endpoint not available
                Ok(ListLoadedModelsResponse::new(Vec::new()))
            }
        }
    }

    /// Load a model (POST /api/generate with empty prompt per Ollama spec)
    pub async fn load_model(&self, model_name: &str) -> Result<LoadModelResponse, ClientError> {
        let url = format!("{}/api/generate", self.base_url);

        // Load model by sending empty prompt with keep_alive parameter
        let mut request = OllamaGenerateRequest::new(
            model_name.to_string(),
            String::new(), // Empty prompt signals load-only operation
        );
        request.keep_alive = Some("5m".to_string()); // Keep loaded for 5 minutes

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(600)) // 10 minutes for loading large models
            .send()
            .await?;

        // Parse response and convert to LoadModelResponse
        let _ollama_response: OllamaGenerateResponse = self.handle_response(response).await?;
        Ok(LoadModelResponse {
            model: model_name.to_string(),
            status: "success".to_string(),
            message: format!("Model {} loaded", model_name),
        })
    }

    /// Unload a model (POST /api/generate with empty prompt and keep_alive: 0 per Ollama spec)
    pub async fn unload_model(&self, model_name: &str) -> Result<(), ClientError> {
        let url = format!("{}/api/generate", self.base_url);

        // Unload model by sending empty prompt with keep_alive: 0
        let mut request = OllamaGenerateRequest::new(
            model_name.to_string(),
            String::new(), // Empty prompt signals unload operation
        );
        request.keep_alive = Some("0".to_string()); // Keep_alive: 0 means unload immediately

        let response = self.http_client.post(&url).json(&request).send().await;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) if resp.status() == reqwest::StatusCode::NOT_FOUND => {
                // Endpoint not found, try Ollama-style unload
                // Ollama unloads by sending a generate request with keep_alive: 0
                let url = format!("{}/api/generate", self.base_url);
                let response = self
                    .http_client
                    .post(&url)
                    .json(&serde_json::json!({
                        "model": model_name,
                        "keep_alive": 0
                    }))
                    .send()
                    .await?;

                if response.status().is_success() {
                    Ok(())
                } else {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    Err(ClientError::Http(format!(
                        "Unload failed with status {}: {}",
                        status, text
                    )))
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(ClientError::Http(format!(
                    "Unload failed with status {}: {}",
                    status, text
                )))
            }
            Err(e) => {
                // Network error, try Ollama-style as fallback
                let url = format!("{}/api/generate", self.base_url);
                let response = self
                    .http_client
                    .post(&url)
                    .json(&serde_json::json!({
                        "model": model_name,
                        "keep_alive": 0
                    }))
                    .send()
                    .await?;

                if response.status().is_success() {
                    Ok(())
                } else {
                    Err(ClientError::Request(e.to_string()))
                }
            }
        }
    }

    /// Validate a model (custom endpoint, not part of the Ollama API)
    pub async fn validate_model(
        &self,
        model_name: &str,
    ) -> Result<crate::inference::load::model_manager::ModelValidationResult, ClientError> {
        let url = format!("{}/api/models/validate", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&serde_json::json!({ "name": model_name }))
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Repair a model (custom endpoint, not part of the Ollama API)
    pub async fn repair_model(
        &self,
        model_name: &str,
    ) -> Result<crate::inference::load::model_manager::ModelMetadata, ClientError> {
        let url = format!("{}/api/models/repair", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&serde_json::json!({ "name": model_name }))
            .timeout(Duration::from_secs(600)) // 10 minutes for re-downloading
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Handle HTTP response
    async fn handle_response<T: DeserializeOwned>(
        &self,
        response: Response,
    ) -> Result<T, ClientError> {
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ClientError::Http(format!(
                "Request failed with status {}: {}",
                status, text
            )));
        }

        let json = response.json::<T>().await?;
        Ok(json)
    }
}

/// Client error types
#[derive(thiserror::Error, Debug)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Request error: {0}")]
    Request(String),
}

impl From<validator::ValidationErrors> for ClientError {
    fn from(err: validator::ValidationErrors) -> Self {
        ClientError::Validation(format!("Validation failed: {}", err))
    }
}

impl From<reqwest::Error> for ClientError {
    fn from(err: reqwest::Error) -> Self {
        ClientError::Request(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = Client::new("http://localhost:11434".to_string());
        assert_eq!(client.base_url, "http://localhost:11434");
    }
}
